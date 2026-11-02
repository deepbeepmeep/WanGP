# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Submanifold 3x3x3 sparse convolution of voxel features as one implicit GEMM, and the per-channel moments of a group norm over voxels.

Convolution: each program owns BM output voxels and BN output channels and walks the 27 kernel taps x the input channels by BK, gathering
the tile's neighbor rows straight from the features through the neighbor map (rows, 27) int32 (-1 for an inactive site), possibly through
a row map when the features live on a coarser level (each voxel reads its parent's row). With AFFINE the gathered rows are normalized on
the fly as the PyTorch path does it (x * scale + shift then SiLU in FP32, separate roundings, precise exponential and IEEE division,
rounded to the activation dtype; inactive sites stay zeros). Products accumulate in FP32 over the whole 27 * Ci reduction, the bias is
added in FP32 and the result rounded once; a residual is added to the rounded result in FP32 and rounded again (a 16-bit add). No
gathered (rows, 27 * Ci) copy exists, which the PyTorch path writes, normalizes in FP32 and reads back for every chunk of rows.
Weights keep the checkpoint layout (Co, 3, 3, 3, Ci) seen as (Co, 27 * Ci). Row counts and offsets are runtime values; input and output
widths are compile-time constants of a model: a new voxel count compiles nothing. The launch tile is tuned once per GPU, dtype, widths
and variant, on a scratch output.

Moments: FP64 sums of x and x^2 per channel over row blocks (exact products of 16-bit values), partials combined by PyTorch in a fixed
order: deterministic."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .triton_tuner import LaunchTuner

_TUNER = LaunchTuner("wan2gp_sparse_conv", 1, "Sparse Conv")
# (BM, BN, BK, warps, stages); the first is the untuned default
_CANDIDATES = ((128, 64, 32, 4, 3), (64, 64, 32, 4, 3), (128, 128, 32, 8, 3), (64, 128, 32, 4, 3), (128, 64, 64, 4, 3), (64, 64, 64, 4, 4),
               (128, 128, 64, 8, 2), (64, 32, 64, 4, 3), (256, 64, 32, 8, 3))
_TUNE_ROWS = 1 << 15
_RUNTIME = ["ROWS", "NOFF", "ROFF", "OOFF"]


@triton.jit(do_not_specialize=_RUNTIME, do_not_specialize_on_alignment=_RUNTIME)
def _sparse_conv(X, NBR, ROWMAP, W, BIAS, SCALE, SHIFT, RES, OUT, ROWS, NOFF, ROFF, OOFF, CI: tl.constexpr, CO: tl.constexpr,
                 HAS_ROWMAP: tl.constexpr, HAS_BIAS: tl.constexpr, AFFINE: tl.constexpr, HAS_RES: tl.constexpr, BM: tl.constexpr,
                 BN: tl.constexpr, BK: tl.constexpr):
    """Output rows [0, ROWS) of this launch: neighbor rows of row i at NBR + (NOFF + i) * 27, output at OUT + (OOFF + i) * CO, residual at
    RES + (ROFF + i) * CO."""
    rows = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM)
    ok = rows < ROWS
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    n_ok = n < CO
    k = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in range(0, 27 * CI, BK):
        tap = kk // CI
        c = kk - tap * CI + k
        index = tl.load(NBR + (NOFF + rows) * 27 + tap, mask=ok, other=-1)
        active = index >= 0
        if HAS_ROWMAP:
            index = tl.load(ROWMAP + index, mask=active, other=0)
        a = tl.load(X + index.to(tl.int64)[:, None] * CI + c[None, :], mask=active[:, None], other=0.0)
        if AFFINE:
            v = a.to(tl.float32) * tl.load(SCALE + c)[None, :] + tl.load(SHIFT + c)[None, :]
            v = tl.div_rn(v, 1.0 + libdevice.exp(-v))
            a = tl.where(active[:, None], v, 0.0).to(X.dtype.element_ty)
        b = tl.load(W + n[None, :].to(tl.int64) * (27 * CI) + (kk + k)[:, None], mask=n_ok[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
    if HAS_BIAS:
        acc += tl.load(BIAS + n, mask=n_ok, other=0.0).to(tl.float32)[None, :]
    y = acc.to(OUT.dtype.element_ty)
    mask = ok[:, None] & n_ok[None, :]
    if HAS_RES:
        y = (y.to(tl.float32) + tl.load(RES + (ROFF + rows)[:, None] * CO + n[None, :], mask=mask, other=0.0).to(tl.float32)).to(OUT.dtype.element_ty)
    tl.store(OUT + (OOFF + rows)[:, None] * CO + n[None, :], y, mask=mask)


def _candidates(device, ci, co, elsize):
    """Tiles whose K step divides the input width and whose pipelined tiles fit the device's shared memory."""
    limit = triton.runtime.driver.active.utils.get_device_properties(device.index if device.index is not None else torch.cuda.current_device())["max_shared_mem"]
    widest = max(16, triton.next_power_of_2(co))
    found = []
    for bm, bn, bk, warps, stages in _CANDIDATES:
        bn = min(bn, widest)
        if ci % bk == 0 and stages * (bm * bk + bk * bn) * elsize <= limit - 8192 and (bm, bn, bk, warps, stages) not in found:
            found.append((bm, bn, bk, warps, stages))
    return tuple(found)


def conv_(x, neighbors, weight, bias, out, *, rowmap=None, affine=None, residual=None, rows=None, neighbor_offset=0, out_offset=0, residual_offset=0):
    """out[out_offset + i] = conv of output row i (i < rows) whose neighbors are neighbors[neighbor_offset + i] (int32, -1 inactive): x (N, Ci)
    16-bit contiguous, weight (Co, 27 * Ci) contiguous, bias (Co,) or None, rowmap (int, the x row of each voxel of the neighbor map's level)
    or None, affine (scale, shift) FP32 (Ci,) or None, residual (rows, Co) at residual_offset or None, out (M, Co) contiguous."""
    ci, co = x.shape[1], out.shape[1]
    rows = neighbors.shape[0] - neighbor_offset if rows is None else rows
    if rows == 0:
        return out
    scale, shift = affine if affine is not None else (x, x)

    def launch(output, count, offset, config):
        bm, bn, bk, warps, stages = config
        _sparse_conv[(triton.cdiv(count, bm), triton.cdiv(co, bn))](
            x, neighbors, x if rowmap is None else rowmap, weight, x if bias is None else bias, scale, shift, x if residual is None else residual,
            output, count, neighbor_offset, residual_offset, offset, ci, co, rowmap is not None, bias is not None, affine is not None,
            residual is not None, bm, bn, bk, num_warps=warps, num_stages=stages, enable_fp_fusion=False)

    def prepare_scratch():  # the first rows into a scratch output: the inputs are only read
        count = min(rows, _TUNE_ROWS)
        scratch = out.new_empty((count, co))
        return lambda config: launch(scratch, count, 0, config)

    candidates = _candidates(x.device, ci, co, x.element_size())
    identity = f"{x.dtype}|ci{ci}|co{co}|M{int(rowmap is not None)}A{int(affine is not None)}B{int(bias is not None)}R{int(residual is not None)}"
    config = _TUNER.config("sparse_conv", identity, x.device, candidates, prepare_scratch, rows >= _TUNE_ROWS)
    launch(out, rows, out_offset, config)
    torch.autograd.graph.increment_version(out)
    return out


@triton.jit(do_not_specialize=["ROWS", "SPAN"], do_not_specialize_on_alignment=["ROWS", "SPAN"])
def _moments(X, PARTIAL, ROWS, SPAN, C: tl.constexpr, CP: tl.constexpr, BR: tl.constexpr):
    """FP64 sums of x and x^2 per channel over rows [pid * SPAN, (pid + 1) * SPAN) into PARTIAL (programs, 2, C)."""
    pid = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, CP)
    c_ok = c < C
    total = tl.zeros((CP,), tl.float64)
    squares = tl.zeros((CP,), tl.float64)
    for start in range(0, SPAN, BR):
        rows = pid * SPAN + start + tl.arange(0, BR)
        x = tl.load(X + rows[:, None] * C + c[None, :], mask=(rows < ROWS)[:, None] & c_ok[None, :], other=0.0).to(tl.float64)
        total += tl.sum(x, 0)
        squares += tl.sum(x * x, 0)
    tl.store(PARTIAL + pid * 2 * C + c, total, mask=c_ok)
    tl.store(PARTIAL + pid * 2 * C + C + c, squares, mask=c_ok)


_MOMENT_SPAN = 4096  # rows per program


def moments(x):
    """(2, C) FP64: the sums of x and of x^2 per channel of x (N, C) 16-bit contiguous."""
    rows, channels = x.shape
    programs = max(1, triton.cdiv(rows, _MOMENT_SPAN))
    partial = torch.empty((programs, 2, channels), dtype=torch.float64, device=x.device)
    block = max(1, 4096 // triton.next_power_of_2(channels))
    _moments[(programs,)](x, partial, rows, _MOMENT_SPAN, channels, triton.next_power_of_2(channels), block, num_warps=4)
    return partial.sum(0)


from shared.kernels.triton_compilation_log import install_triton_compilation_logger
install_triton_compilation_logger()
