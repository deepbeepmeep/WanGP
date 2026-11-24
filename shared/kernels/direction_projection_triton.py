# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Rank-one projection of the rows of a tensor, in place: x <- x - scale * r (r . x) with a unit direction r.

A layer's output loses part of its component along r, as a rank-one adapter B (A x) with B = -r and A = r^T W would remove it (a direction
ablated at the residual writers). FP32 dot product and update, one rounding to the tensor's dtype; one program per row.
"""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['ROWS'], do_not_specialize_on_alignment=['ROWS'])
def _project_out(X, R, ROWS, D, SCALE, BLOCK: tl.constexpr):
    # D at runtime: one kernel per power-of-two range of widths (BLOCK)
    row = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BLOCK)
    x = tl.load(X + row * D + c, c < D, 0).to(tl.float32)
    r = tl.load(R + c, c < D, 0)
    d = tl.sum(x * r, 0) * SCALE
    tl.store(X + row * D + c, (x - d * r).to(X.dtype.element_ty), c < D)


def project_out_(x, direction, scale):
    """x (..., D) contiguous on a CUDA device; direction (D,) FP32 on the same device."""
    width = x.shape[-1]
    rows = x.numel() // width
    if rows:
        _project_out[(rows,)](x, direction, rows, width, float(scale), triton.next_power_of_2(width), num_warps=4 if width <= 4096 else 8)
        torch.autograd.graph.increment_version(x)
    return x


from .triton_compilation_log import install_triton_compilation_logger  # noqa: E402

install_triton_compilation_logger()
