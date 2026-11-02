# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Length-independent launch setup for the FLA 0.4 inference kernels used by Qwen.

Reuse the installed FLA kernel bodies and their license/provenance. Adjust JIT
specialization, unused tuning labels, and the convolution launch grid. Triton
3.9 also needs block accesses lowered to ordinary masked loads and stores.
The package on disk is never edited. Install before the first model forward.
"""
from copy import copy
from importlib import import_module
from types import FunctionType

import torch
import triton
from triton.runtime.autotuner import Autotuner, Heuristics
from triton.runtime.jit import JITFunction

from .triton_compilation_log import install_triton_compilation_logger
from .triton_autotune import InferenceAutotuner

_INSTALLED = False
_CONV_BT = 16
_MASKED_POINTERS = tuple(map(int, triton.__version__.split(".")[:2])) >= (3, 9)
_KERNELS = {
    "fla.modules.convolution": ("causal_conv1d_fwd_kernel", "causal_conv1d_states_fwd_kernel"),
    "fla.modules.l2norm": ("l2norm_fwd_kernel",),
    "fla.modules.fused_norm_gate": ("layer_norm_gated_fwd_kernel",),
    "fla.ops.common.chunk_delta_h": ("chunk_gated_delta_rule_fwd_kernel_h_blockdim64",),
    "fla.ops.common.chunk_o": ("chunk_fwd_kernel_o",),
    "fla.ops.common.chunk_scaled_dot_kkt": ("chunk_scaled_dot_kkt_fwd_kernel",),
    "fla.ops.gated_delta_rule.wy_fast": ("recompute_w_u_fwd_kernel",),
    "fla.ops.gated_delta_rule.fused_recurrent": ("fused_recurrent_gated_delta_rule_fwd_kernel",),
    "fla.ops.utils.cumsum": ("chunk_local_cumsum_scalar_kernel",),
    "fla.ops.utils.solve_tril": ("merge_16x16_to_64x64_inverse_kernel",),
}


def _clone_kernel(kernel):
    if isinstance(kernel, JITFunction):
        original = kernel.fn
        fn = FunctionType(original.__code__, original.__globals__, original.__name__, original.__defaults__, original.__closure__)
        fn.__module__, fn.__qualname__ = original.__module__, original.__qualname__
        fn.__annotations__ = {name: value for name, value in original.__annotations__.items() if name not in ("B", "T", "NB")}
        runtime = tuple(name for name in ("B", "T", "NB") if name in kernel.arg_names)
        cloned = triton.jit(fn, do_not_specialize=tuple(kernel.do_not_specialize) + runtime,
                            do_not_specialize_on_alignment=tuple(kernel.do_not_specialize_on_alignment) + runtime)
        if _MASKED_POINTERS:
            from .fla_block_pointers import lower_block_pointers
            lower_block_pointers(cloned)
        return cloned
    if not isinstance(kernel, (Autotuner, Heuristics)):
        raise TypeError(f"Unsupported FLA kernel wrapper: {type(kernel).__name__}")
    cloned = copy(kernel)
    cloned.fn = _clone_kernel(kernel.fn)
    if isinstance(cloned, Autotuner):
        cloned.__class__ = InferenceAutotuner
        cloned.cache = {}
        cloned.keys = [key for key in cloned.keys if key not in ("B", "NB")]
        # Flags can change register/shared-memory needs even at the same width.
        leaf = cloned.fn
        while not isinstance(leaf, JITFunction):
            leaf = leaf.fn
        for param in leaf.params:
            if param.is_constexpr and param.name not in {"BT", "BD", "BK", "BV", "DOT_PRECISION"}:
                if param.name not in cloned.keys:
                    cloned.keys.append(param.name)
        cloned.cache_results = True
    return cloned


def _conv_forward(x, weight, bias, residual, initial_state=None, output_final_state=False,
                  activation=None, cu_seqlens=None, chunk_indices=None):
    # Launch the unmodified upstream body with one token tile for all lengths.
    # Rebuild the grid/varlen indices together: changing only BT would skip rows.
    from fla.modules import convolution as conv
    shape = x.shape
    if x.shape[-1] != weight.shape[0]:
        x = x.flatten(2)
    batch, tokens, channels = x.shape
    width = weight.shape[1]
    if cu_seqlens is not None:
        chunk_indices = conv.prepare_chunk_indices(cu_seqlens, _CONV_BT)
    chunks = len(chunk_indices) if cu_seqlens is not None else triton.cdiv(tokens, _CONV_BT)
    y = torch.empty_like(x)
    conv.causal_conv1d_fwd_kernel[lambda meta: (triton.cdiv(channels, meta["BD"]), chunks, batch)](
        x=x, y=y, weight=weight, bias=bias, residual=residual, cu_seqlens=cu_seqlens,
        initial_state=initial_state, chunk_indices=chunk_indices, B=batch, T=tokens,
        D=channels, W=width, BT=_CONV_BT, BW=triton.next_power_of_2(width), NB=1, ACTIVATION=activation,
    )
    final_state = conv.causal_conv1d_update_states(x=x, state_len=width, initial_state=initial_state,
                                                 cu_seqlens=cu_seqlens) if output_final_state else None
    return y.view(shape), final_state


def configure_fla_inference_kernels():
    global _INSTALLED
    if _INSTALLED:
        return
    # Build all replacements before installing any of them.
    replacements = [(import_module(module), name, _clone_kernel(getattr(import_module(module), name)))
                    for module, names in _KERNELS.items() for name in names]
    if _MASKED_POINTERS:
        conv = import_module("fla.modules.convolution")
        replacements.append((conv, "causal_conv1d_update_kernel", _clone_kernel(conv.causal_conv1d_update_kernel)))
    for module, name, kernel in replacements:
        setattr(module, name, kernel)
    conv = import_module("fla.modules.convolution")
    conv.causal_conv1d_fwd = conv.input_guard(_conv_forward)
    _INSTALLED = True


install_triton_compilation_logger()
