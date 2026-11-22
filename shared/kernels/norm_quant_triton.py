# Copyright (c) 2024-2026 DeepBeepMeep. WanGP License 2.0.
"""RMSNorm, segmented modulation, ConvRot rotation and INT8 quantization of rows in one pass, into the WGPKernels 'block' format (INT8
values, one FP32 scale per row and 128-element block of K): the input of INT8 projections made from a residual without its 16-bit
norm output (mmgp offload.RowSource.fill_prepared). The norm and the modulation are computed in FP32 and rounded once, at the
quantization (the eager path rounds the norm output and each modulation step in 16 bits); the rotation and the quantization follow
the WGPKernels quantizer's arithmetic (the order of its Hadamard sums, 127 / amax, a zero scale for a zero block)."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from shared.kernels.convrot_int8_triton import _hadamard_step

CHUNK = 256  # columns per step: a ConvRot rotation group at most, two quantization blocks
BLOCK = 128  # columns per activation scale (the WGPKernels block path)
ROWS = 2  # rows per program (with WARPS: 0.49 ms for 25k x 5376 rows on an RTX 5090, 1.2 to 2.4x faster than 4 to 16 rows)
WARPS = 2
_INTS = ["rows", "LDX", "SS", "SH", "FIRST", "STEP"]


@triton.jit(do_not_specialize=_INTS, do_not_specialize_on_alignment=_INTS)
def _rms_modulate_quantize(X, W, SCALE, SHIFT, ROWMOD, Q, QS, Y, rows, LDX, SS, SH, EPS, COUNTER, FIRST, STEP, K: tl.constexpr, BM: tl.constexpr,
                           GROUP: tl.constexpr, NORM: tl.constexpr, HAS_W: tl.constexpr, WANT_Y: tl.constexpr, AT: tl.constexpr):
    """Rows r of X (BM per program): v = X[r] / RMS(X[r]) (times W) * (1 + SCALE[m]) + SHIFT[m], m = ROWMOD[r], in FP32; Y[r] = v in X's
    dtype (WANT_Y); v rotated per GROUP columns, then Q[r] = rint(v * 127 / amax) and QS[r] = amax * NORM / 127 per 128 columns.
    AT: X's and ROWMOD's rows from FIRST + counter * STEP (a CUDA graph loop's turn, the counter on the device)."""
    if AT:
        base = (FIRST + tl.load(COUNTER) * STEP).to(tl.int64)
        X += base * LDX
        ROWMOD += base
    r = tl.program_id(0) * BM + tl.arange(0, BM)
    inside = r < rows
    r64 = r.to(tl.int64)
    offsets = tl.arange(0, 256)
    squares = tl.zeros((BM,), tl.float32)
    for c in range(0, K, 256):
        a = tl.load(X + r64[:, None] * LDX + c + offsets[None, :], inside[:, None], 0).to(tl.float32)
        squares += tl.sum(a * a, axis=1)
    inv = tl.div_rn(1.0, tl.sqrt_rn(tl.div_rn(squares, K) + EPS))
    mod = tl.load(ROWMOD + r, inside, 0).to(tl.int64)
    for c in range(0, K, 256):
        cols = c + offsets
        v = tl.load(X + r64[:, None] * LDX + cols[None, :], inside[:, None], 0).to(tl.float32) * inv[:, None]
        if HAS_W:
            v = v * tl.load(W + cols).to(tl.float32)[None, :]
        scale = tl.load(SCALE + mod[:, None] * SS + cols[None, :], inside[:, None], 0).to(X.dtype.element_ty).to(tl.float32)
        shift = tl.load(SHIFT + mod[:, None] * SH + cols[None, :], inside[:, None], 0).to(X.dtype.element_ty).to(tl.float32)
        v = v * (1.0 + scale) + shift
        if WANT_Y:
            tl.store(Y + r64[:, None] * K + cols[None, :], v.to(Y.dtype.element_ty), inside[:, None])
        if GROUP >= 4:
            v = _hadamard_step(v, BM, 256, 1)
        if GROUP >= 16:
            v = _hadamard_step(v, BM, 256, 4)
        if GROUP >= 64:
            v = _hadamard_step(v, BM, 256, 16)
        if GROUP >= 256:
            v = _hadamard_step(v, BM, 256, 64)
        b = v.reshape((BM, 2, 128))
        amax = tl.max(tl.abs(b), axis=2)
        factor = tl.where(amax > 0.0, tl.div_rn(127.0, tl.where(amax > 0.0, amax, 1.0)), 0.0)
        q = libdevice.nearbyint(b * factor[:, :, None]).to(tl.int8).reshape((BM, 256))
        tl.store(Q + r64[:, None] * K + cols[None, :], q, inside[:, None])
        blocks = c // 128 + tl.arange(0, 2)
        tl.store(QS + r64[:, None] * (K // 128) + blocks[None, :], tl.div_rn(amax * NORM, 127.0), inside[:, None])


def supported(x, weight):
    """Whether rms_modulate_quantize takes these rows (a CUDA 16-bit (rows, K) view of rows one stride apart, K a multiple of 256)."""
    return (x.is_cuda and x.dim() == 2 and x.dtype in (torch.bfloat16, torch.float16) and x.stride(1) == 1 and x.shape[1] % CHUNK == 0
            and x.shape[1] <= 16384 and (weight is None or weight.dtype == x.dtype and weight.is_contiguous()))


def rms_modulate_quantize(x, weight, eps, scale, shift, rowmod, group, want_y, at=None):
    """(y or None, q, scales) of rows x (rows, K): the RMSNorm (times weight), the modulation of row r by scale / shift row rowmod[r]
    (int32, one per row), rotated by ConvRot groups of `group` columns (0: none) and quantized as the WGPKernels block path does.
    at: (counter, first, rows) to read `rows` rows of x and rowmod from first + counter * rows instead (a CUDA graph loop's turn)."""
    rows, k = x.shape if at is None else (at[2], x.shape[1])
    q = torch.empty((rows, k), dtype=torch.int8, device=x.device)
    scales = torch.empty((rows, k // BLOCK), dtype=torch.float32, device=x.device)
    y = torch.empty((rows, k), dtype=x.dtype, device=x.device) if want_y else None
    norm = {0: 1.0, 16: 0.25, 64: 0.125, 256: 0.0625}[group]
    _rms_modulate_quantize[(triton.cdiv(rows, ROWS),)](x, x if weight is None else weight, scale, shift, rowmod, q, scales, q if y is None else y,
                                                      rows, x.stride(0), scale.stride(0), shift.stride(0), eps,
                                                      rowmod if at is None else at[0], 0 if at is None else at[1], rows, k, ROWS, group, norm,
                                                      weight is not None, want_y, at is not None, num_warps=WARPS)
    return y, q, scales


from shared.kernels.triton_compilation_log import install_triton_compilation_logger
install_triton_compilation_logger()
