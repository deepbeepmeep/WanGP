# SPDX-License-Identifier: Apache-2.0
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""MiniMax H3 policy for the bundled Sol-Attn kernels."""

from __future__ import annotations

import os

from mmgp import cuda_graphs, offload

from shared.attention import pay_attention
from shared.kernels import ops as denoiser_kernels


SOL_ATTN_THRESH_TYPE = "diag"
SOL_ATTN_MIN_TOKENS = 8192
# Development switch: "i8" or "fp4" run the CUDA kernels of shared.sol_attn.cuda (GeForce RTX 50xx) instead of the Triton ones
SOL_ATTN_CUDA = os.environ.get("WGP_SOL_CUDA", "i8")
# the active Sol state of the running forward (tau, sink tokens, CUDA kernels): the blocks pass them to the kernels as numbers, which CUDA graphs
# record, so they are part of the graph keys
_graph_state = None
cuda_graphs.register_key(lambda: _graph_state)


class MiniMaxH3SolAttention:
    def __init__(self):
        self.enabled = False
        self._runtime_validated = False
        self._announced = False
        self.sink_tokens = 0
        self.tau = 1.0
        self.cuda = ""

    def begin_forward(self, layout, device, dtype, tau, grouped_video=False):
        self.sink_tokens = int(layout.video_indices[layout.num_condition_video_rows])
        self.tau = float(tau)
        self.enabled = offload.shared_state.get("_attention") == "sol"
        self.cuda = ""
        if self.enabled and SOL_ATTN_CUDA in ("i8", "fp4"):
            from shared.sol_attn import cuda as sol_cuda
            if sol_cuda.supported(device, dtype):
                self.cuda = SOL_ATTN_CUDA
        global _graph_state
        _graph_state = (self.tau, self.sink_tokens, self.cuda) if self.enabled else None
        if self.enabled and grouped_video:
            raise ValueError("MiniMax H3 grouped masked denoising is not compatible with Sol Attention")
        if self.enabled and not self._runtime_validated:
            from shared.sol_attn import validate_runtime
            capability = validate_runtime(device, dtype)
            self._runtime_validated = True
            if not self._announced:
                kernels = {"": "Triton", "i8": "CUDA kernels", "fp4": "CUDA kernels, NVFP4 scores"}[self.cuda]
                print(f"[MiniMax H3] Sol-Attn enabled with {kernels} on SM{capability[0]}{capability[1]} (tau={self.tau:g}, {SOL_ATTN_THRESH_TYPE})")
                self._announced = True

    def use_for_layer(self, tokens):
        return self.enabled and tokens >= SOL_ATTN_MIN_TOKENS

    def staged(self, x, projections):
        """True when attention() can compute the q/k/v projections of x one tensor at a time on this GPU."""
        from shared.sol_attn import staged_supported
        return staged_supported(x.device, x.shape[0], SOL_ATTN_THRESH_TYPE) and offload.linear_rows_supported(projections, x)

    def attention(self, x_list, projections, norms, rope, eps, heads, head_dim):
        """Sol attention of the q/k/v projections of x (handed off), with the results of __call__: v first, then k (normalized,
        rotated and prepared as INT8, then released), then q, which receives the output. With Attention Head Split, one group of
        heads at a time into one output."""
        from shared import attention_kit
        from shared.sol_attn import cuda as sol_cuda, prepare_kv, sol_attn_prepared
        tokens = x_list[0].shape[0]
        groups = attention_kit.head_groups(heads, tokens)
        step = heads // groups
        linear_input = offload.prepare_linear_input(x_list, projections)
        (q_proj, k_proj, v_proj), (q_norm, k_norm) = projections, norms
        output = None
        for first in range(0, heads, step):
            project = lambda module: offload.linear_rows(module, linear_input, first * head_dim, (first + step) * head_dim).view(1, tokens, step, head_dim)
            value = project(v_proj)
            if self.cuda:  # each projection is packed and released before the next one exists
                state = sol_cuda.allocate(tokens, step, value.device, fp4=self.cuda == "fp4")
                sol_cuda.pack_values(state, value)
                value = None
            key = project(k_proj)
            denoiser_kernels.sol_rms_rope_(key, k_norm.weight, rope, eps)
            if self.cuda:
                sol_cuda.pack_keys(state, key)
                prepared = sol_cuda.finish(state)
                state = None
            else:
                prepared = prepare_kv(key, value)
            key = None
            query = project(q_proj)
            if first + step == heads:
                linear_input = project = None
            denoiser_kernels.sol_rms_rope_(query, q_norm.weight, rope, eps)
            if self.cuda:
                query = sol_cuda.attention(query, prepared, tau=self.tau, sink_start=0, sink_tokens=self.sink_tokens, out=query)
            else:
                query = sol_attn_prepared(query, value, prepared, tau=self.tau, sink_start=0, sink_tokens=self.sink_tokens, out=query)
            if groups == 1:
                return query
            value = prepared = None
            if output is None:
                output = query.new_empty((1, tokens, heads, head_dim))
            output[:, :, first:first + step] = query
            query = None
        return output

    def __call__(self, qkv_list, use_sol):
        if not use_sol:
            return pay_attention(qkv_list, recycle_q=True)

        query, key, value = qkv_list
        qkv_list.clear()
        if self.cuda:
            from shared.sol_attn import cuda as sol_cuda
            if sol_cuda.usable(query, key, value):
                prepared = sol_cuda.prepare(key, value, fp4=self.cuda == "fp4")
                del key, value
                return sol_cuda.attention(query, prepared, tau=self.tau, sink_start=0, sink_tokens=self.sink_tokens, out=query)
        from shared.sol_attn import sol_attn
        output = sol_attn(query, key, value, tau=self.tau, thresh_type=SOL_ATTN_THRESH_TYPE,
                          sink_start=0, sink_tokens=self.sink_tokens, int8_qk=True)
        return output


__all__ = ["MiniMaxH3SolAttention"]
