# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Independent AdaLN with native-dtype intermediate rounding and FP32 statistics.

Two-pass variance; no package-specific CUDA intrinsics or shape autotuning.
"""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['ROWS', 'INNER', 'S0', 'S1', 'H0', 'H1'],
            do_not_specialize_on_alignment=['ROWS', 'INNER', 'S0', 'S1', 'H0', 'H1'])
def _adaln(X, S, H, Y, ROWS, INNER, S0, S1, H0, H1, EPS, D, BLOCK: tl.constexpr):
    # D, EPS at runtime: one kernel per power-of-two range of widths (BLOCK), whatever the model
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BLOCK)
    x = tl.load(X + row * D + c, c < D, 0).to(tl.float32)
    mean = tl.div_rn(tl.sum(x, 0), D.to(tl.float32))
    centered = tl.where(c < D, x - mean, 0.)
    var = tl.div_rn(tl.sum(centered * centered, 0), D.to(tl.float32))
    inv = tl.div_rn(1., tl.sqrt_rn(var + EPS))
    dtype = Y.dtype.element_ty
    norm = (centered * inv).to(dtype).to(tl.float32)
    s = tl.load(S + row // INNER * S0 + row % INNER * S1 + c, c < D, 0).to(tl.float32)
    h = tl.load(H + row // INNER * H0 + row % INNER * H1 + c, c < D, 0).to(tl.float32)
    gain = (1. + s).to(dtype).to(tl.float32)
    y = (norm * gain).to(dtype).to(tl.float32) + h
    tl.store(Y + row * D + c, y.to(dtype), c < D)


def adaln(x, scale, shift, eps=1e-6):
    scale_view, shift_view = scale.expand(x.shape), shift.expand(x.shape)
    ss = (0, scale_view.stride(0)) if x.ndim == 2 else scale_view.stride()[:2]
    hs = (0, shift_view.stride(0)) if x.ndim == 2 else shift_view.stride()[:2]
    y = torch.empty_like(x)
    width = x.shape[-1]
    rows = x.numel() // width
    _adaln[(rows,)](x, scale, shift, y, rows, x.shape[-2], *ss, *hs, eps, width,
                    triton.next_power_of_2(width), num_warps=4 if width <= 4096 else 8, enable_fp_fusion=False)
    return y


from .triton_compilation_log import install_triton_compilation_logger  # noqa: E402

install_triton_compilation_logger()
