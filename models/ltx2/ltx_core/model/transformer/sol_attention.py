# SPDX-License-Identifier: Apache-2.0
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""LTX2 policy for the bundled Sol-Attn kernels."""

from __future__ import annotations

import torch
from mmgp import cuda_graphs, offload


SOL_ATTN_THRESH_TYPE = "diag"
SOL_ATTN_MIN_TOKENS = 8192
SOL_ATTN_HEAD_DIM = 128
SOL_ATTN_TAU_END = 0.8
SOL_ATTN_TAU_START_DEFAULT = 1.0
SOL_ATTN_TAU_START_KEY = "_sol_attention_sparsity"
# The first third of the denoising steps of a phase runs with full attention, so the face and the shape are built densely
# before the sparse path takes over.
SOL_ATTN_WARMUP_FRACTION = 1.0 / 3.0
# A block keeps one CUDA graph per tau value and a few graphs per block, so the schedule from Start Tau to End Tau is walked
# in that many levels.
SOL_ATTN_TAU_LEVELS = 5
# Past that share of the window is conditioning, the sparse path has nothing left to win and the window stays dense.
SOL_ATTN_DENSE_PREFIX_FRACTION = 0.25
# the active Sol state of the running forward (tau, sink tokens): the blocks pass them to the kernels as numbers, which CUDA
# graphs record, so they are part of the graph keys
_graph_state = None
cuda_graphs.register_key(lambda: _graph_state)


def first_modality(modality):
    """Return the first non-None Modality (or the modality itself)."""
    if modality is None:
        return None
    if isinstance(modality, (list, tuple)):
        for item in modality:
            if item is not None:
                return item
        return None
    return modality


def current_step_fraction(video, audio) -> float:
    """Progress of the current denoising step in [0, 1] for the tau schedule, read from the numbers a CUDA graph may record.
    Unknown progress counts as the first step, so a run starts dense and at the highest tau."""
    modality = first_modality(video) or first_modality(audio)
    if modality is None or modality.step_index is None or modality.sigma_schedule is None:
        return 0.0
    total = max(len(modality.sigma_schedule) - 1, 1)
    return min(max(float(modality.step_index) / total, 0.0), 1.0)


def sink_tokens_from_mask(denoise_mask):
    """Count of the leading tokens that carry conditioning (denoise_mask 0): the tokens of a reference video, a keyframe, a
    start image and a Multiple Subject Reference, which hold the identity of the person. The kernels keep that prefix in the
    exact set, and the policy attends to it densely, so the identity is never read through an approximation that dropped it.
    Returns None when the conditioned tokens of a sample are not one leading run, which the sparse path cannot express:
    the layers then keep full attention for that window. One sink prefix serves the batch, so the largest leading count
    keeps every conditioning token exact."""
    if denoise_mask is None:
        return None
    mask = denoise_mask.detach()
    if mask.ndim == 3:
        mask = mask[..., 0]
    conditioned = mask == 0
    if conditioned.is_cuda:
        conditioned = conditioned.cpu()  # the caller caches the answer, so this synchronization runs once per window
    leading = conditioned.int().cumprod(dim=1).sum(dim=1)
    if bool((leading != conditioned.sum(dim=1)).any()):
        return None
    return int(leading.max())


def quantize_tau(tau, tau_start):
    """The tau of the schedule rounded to one of SOL_ATTN_TAU_LEVELS levels between End Tau and Start Tau, in either
    direction, so a block keeps at most that many sparse graphs."""
    span = tau_start - SOL_ATTN_TAU_END
    if span == 0.0:
        return round(SOL_ATTN_TAU_END, 4)
    levels = SOL_ATTN_TAU_LEVELS - 1
    if span > 0.0:
        level = min(max(int(round((tau - SOL_ATTN_TAU_END) / span * levels)), 0), levels)
        return round(SOL_ATTN_TAU_END + span * level / levels, 4)
    span = -span
    level = min(max(int(round((tau - tau_start) / span * levels)), 0), levels)
    return round(tau_start + span * level / levels, 4)


class LTX2SolAttention:
    def __init__(self):
        self.enabled = False
        self.sparse = False
        self._runtime_validated = False
        self._runtime_available = False
        self._announced = False
        self.sink_tokens = 0
        self.tau = SOL_ATTN_TAU_END

    def begin_forward(self, device, dtype, video=None, audio=None):
        """Refresh the policy for one forward: the mode, the tau of the schedule, the conditioning tokens kept exact, and the
        state CUDA graphs watch."""
        global _graph_state
        self.sparse = False
        self.sink_tokens = 0
        self.enabled = offload.shared_state.get("_attention") == "sol"
        if not self.enabled:
            _graph_state = None
            return
        if self._runtime_validated and not self._runtime_available:
            self.enabled = False
            _graph_state = None
            return
        tau_start = float(offload.shared_state.get(SOL_ATTN_TAU_START_KEY, SOL_ATTN_TAU_START_DEFAULT))
        fraction = current_step_fraction(video, audio)
        self.tau = tau_start + (SOL_ATTN_TAU_END - tau_start) * fraction
        modality = first_modality(video) or first_modality(audio)
        sink = 0 if modality is None else modality.sol_sink_tokens
        tokens = 0 if modality is None else modality.latent.shape[1]
        # The warmup steps keep full attention, and so does a window made mostly of conditioning tokens: the sparse path has
        # nothing left to win there. A window whose conditioning tokens are scattered has no exact prefix to express, so it
        # keeps full attention too.
        self.sparse = (
            sink is not None
            and fraction >= SOL_ATTN_WARMUP_FRACTION
            and sink <= tokens * SOL_ATTN_DENSE_PREFIX_FRACTION
        )
        self.sink_tokens = sink if self.sparse and sink is not None else 0
        if self.sparse:
            self.tau = quantize_tau(self.tau, tau_start)
            _graph_state = (self.tau, self.sink_tokens)
        else:
            _graph_state = None
        if not self._runtime_validated:
            from shared.sol_attn import validate_runtime

            try:
                capability = validate_runtime(device, dtype)
            except RuntimeError as error:
                self._runtime_validated = True
                self._runtime_available = False
                self.enabled = False
                self.sparse = False
                self.sink_tokens = 0
                _graph_state = None
                if not self._announced:
                    print(f"[LTX2] Sol-Attn is not available on this system: {error}. Regular attention is used.")
                    self._announced = True
                return
            self._runtime_validated = True
            self._runtime_available = True
            if not self._announced:
                print(
                    f"[LTX2] Sol-Attn enabled with Triton on SM{capability[0]}{capability[1]} "
                    f"(tau {tau_start:g} to {SOL_ATTN_TAU_END:g} in {SOL_ATTN_TAU_LEVELS} levels, the first third of "
                    f"each phase dense, the conditioning tokens exact, {SOL_ATTN_THRESH_TYPE})"
                )
                self._announced = True

    def use_for_layer(self, tokens, head_dim):
        # A stream shorter than the conditioning prefix (the audio blocks) and a prefix of more than a quarter of the stream
        # keep full attention.
        return (
            self.sparse
            and head_dim == SOL_ATTN_HEAD_DIM
            and tokens >= SOL_ATTN_MIN_TOKENS
            and self.sink_tokens < tokens
            and self.sink_tokens <= tokens * SOL_ATTN_DENSE_PREFIX_FRACTION
        )

    def __call__(self, qkv_list):
        query, key, value = qkv_list
        qkv_list.clear()
        from shared.sol_attn import sol_attn

        tokens = query.shape[1]
        sink = self.sink_tokens if self.sink_tokens < tokens else 0
        if not sink:
            # Nothing to protect: the INT8 path, the fastest one.
            return sol_attn(query, key, value, tau=self.tau, thresh_type=SOL_ATTN_THRESH_TYPE,
                            sink_start=0, sink_tokens=0, int8_qk=True)
        # The conditioning rows carry the identity of the person, and the sparse kernels cannot compute them: query_start
        # leaves their output uninitialized. They are attended densely, and the generated rows take the sparse path with the
        # conditioning keys always kept in the exact set.
        from shared.sol_attn.interface import BLOCK_SIZE

        rows = min(-(-sink // BLOCK_SIZE) * BLOCK_SIZE, tokens)
        exact = torch.nn.functional.scaled_dot_product_attention(
            query[:, :rows].transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        ).transpose(1, 2)
        output = sol_attn(query, key, value, tau=self.tau, thresh_type=SOL_ATTN_THRESH_TYPE,
                          sink_start=0, sink_tokens=sink, query_start=rows, int8_qk=False, recycle_q=True)
        output[:, :rows] = exact
        return output


sol_attention = LTX2SolAttention()


__all__ = [
    "LTX2SolAttention",
    "SOL_ATTN_DENSE_PREFIX_FRACTION",
    "SOL_ATTN_HEAD_DIM",
    "SOL_ATTN_MIN_TOKENS",
    "SOL_ATTN_TAU_END",
    "SOL_ATTN_TAU_LEVELS",
    "SOL_ATTN_TAU_START_DEFAULT",
    "SOL_ATTN_TAU_START_KEY",
    "SOL_ATTN_THRESH_TYPE",
    "SOL_ATTN_WARMUP_FRACTION",
    "current_step_fraction",
    "first_modality",
    "quantize_tau",
    "sink_tokens_from_mask",
    "sol_attention",
]
