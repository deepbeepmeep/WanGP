# SPDX-License-Identifier: Apache-2.0
"""LTX2 policy for the bundled Sol-Attn kernels."""

from __future__ import annotations

import torch
from mmgp import offload


SOL_ATTN_THRESH_TYPE = "diag"
SOL_ATTN_MIN_TOKENS = 8192
SOL_ATTN_HEAD_DIM = 128
SOL_ATTN_TAU_END = 0.8
SOL_ATTN_TAU_START_DEFAULT = 1.3
SOL_ATTN_TAU_START_KEY = "_sol_attention_sparsity"


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
    """Progress of the current denoising step in [0, 1] for the tau schedule."""
    modality = first_modality(video) or first_modality(audio)
    if modality is None:
        return 1.0
    if modality.step_index is not None and modality.sigma_schedule is not None:
        total = max(len(modality.sigma_schedule) - 1, 1)
        return min(max(float(modality.step_index) / total, 0.0), 1.0)
    if modality.sigma is not None:
        # Flow-matching sigmas start at 1.0 and end at 0, so sigma is a direct progress proxy.
        value = float(modality.sigma.detach().max())
        return min(max(value, 0.0), 1.0)
    return 1.0


class LTX2SolAttention:
    def __init__(self):
        self.enabled = False
        self._runtime_validated = False
        self._announced = False
        self.sink_tokens = 0
        self.tau = SOL_ATTN_TAU_END

    def begin_forward(self, device, dtype, video=None, audio=None):
        self.enabled = offload.shared_state.get("_attention") == "sol"
        if not self.enabled:
            return
        tau_start = float(offload.shared_state.get(SOL_ATTN_TAU_START_KEY, SOL_ATTN_TAU_START_DEFAULT))
        fraction = current_step_fraction(video, audio)
        self.tau = SOL_ATTN_TAU_END + (tau_start - SOL_ATTN_TAU_END) * (1.0 - fraction)
        if not self._runtime_validated:
            from shared.sol_attn import validate_runtime

            capability = validate_runtime(device, dtype)
            self._runtime_validated = True
            if not self._announced:
                print(
                    f"[LTX2] Sol-Attn enabled with Triton on SM{capability[0]}{capability[1]} "
                    f"(tau start={tau_start:g} end={SOL_ATTN_TAU_END:g}, {SOL_ATTN_THRESH_TYPE})"
                )
                self._announced = True

    def use_for_layer(self, tokens, head_dim):
        return self.enabled and head_dim == SOL_ATTN_HEAD_DIM and tokens >= SOL_ATTN_MIN_TOKENS

    def __call__(self, qkv_list):
        query, key, value = qkv_list
        qkv_list.clear()
        from shared.sol_attn import sol_attn

        output = sol_attn(query, key, value, tau=self.tau, thresh_type=SOL_ATTN_THRESH_TYPE,
                          sink_start=0, sink_tokens=self.sink_tokens, int8_qk=True)
        return output


sol_attention = LTX2SolAttention()


__all__ = [
    "LTX2SolAttention",
    "SOL_ATTN_HEAD_DIM",
    "SOL_ATTN_MIN_TOKENS",
    "SOL_ATTN_TAU_END",
    "SOL_ATTN_TAU_START_DEFAULT",
    "SOL_ATTN_TAU_START_KEY",
    "current_step_fraction",
    "first_modality",
    "sol_attention",
]