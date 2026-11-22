# Copyright (c) 2024-2026 DeepBeepMeep. WanGP License 2.0.
"""RMSNorm or LayerNorm, segmented modulation (optional), ConvRot rotation and INT8 quantization of rows in one pass, into a 'block' format (INT8 values, one
FP32 scale per row and block of K columns: 128 for WGPKernels, the GEMM's K tile for the Triton backend): the input of INT8 projections
made from a residual without its 16-bit norm output (mmgp offload.RowSource.fill_prepared). The norm and the modulation are computed in FP32 and rounded once, at the
quantization (the eager path rounds the norm output and each modulation step in 16 bits); the rotation and the quantization follow
the WGPKernels quantizer's arithmetic (the order of its Hadamard sums, 127 / amax, a zero scale for a zero block)."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from shared.kernels.convrot_int8_triton import _hadamard_step

CHUNK = 256  # columns per step: a ConvRot rotation group at most, whole quantization blocks
BLOCKS = (0, 64, 128, 256)  # columns per activation scale it makes (WGPKernels block path: 128; Triton: its GEMM's K tile; 0: one per row, Comfy Kitchen)
ROWS = 2  # rows per program (with WARPS: 0.49 ms for 25k x 5376 rows on an RTX 5090, 1.2 to 2.4x faster than 4 to 16 rows)
WARPS = 2
_INTS = ["rows", "LDX", "SS", "SH", "FIRST", "STEP"]


@triton.jit
def _statistics(X, r64, inside, offsets, LDX, EPS, K: tl.constexpr, BM: tl.constexpr, LAYER: tl.constexpr):
    """(mean, 1 / sqrt(variance + EPS)) of rows r64 in FP32: LayerNorm's (the mean, then the mean square of the centered values: two
    passes), RMSNorm's (a zero mean, the mean square)."""
    mean = tl.zeros((BM,), tl.float32)
    if LAYER:
        for c in range(0, K, 256):
            mean += tl.sum(tl.load(X + r64[:, None] * LDX + c + offsets[None, :], inside[:, None], 0).to(tl.float32), axis=1)
        mean = tl.div_rn(mean, K)
    squares = tl.zeros((BM,), tl.float32)
    for c in range(0, K, 256):
        a = tl.load(X + r64[:, None] * LDX + c + offsets[None, :], inside[:, None], 0).to(tl.float32) - mean[:, None]
        squares += tl.sum(tl.where(inside[:, None], a * a, 0.0), axis=1)
    return mean, tl.div_rn(1.0, tl.sqrt_rn(tl.div_rn(squares, K) + EPS))


@triton.jit
def _rotated(X, W, SCALE, SHIFT, Y, r64, inside, mean, inv, mod, cols, LDX, SS, SH, K: tl.constexpr, BM: tl.constexpr, GROUP: tl.constexpr,
             HAS_W: tl.constexpr, HAS_MOD: tl.constexpr, STORE_Y: tl.constexpr):
    """Columns cols (256) of rows r64: normalized, modulated (stored in Y with STORE_Y), rotated per GROUP columns, in FP32."""
    v = (tl.load(X + r64[:, None] * LDX + cols[None, :], inside[:, None], 0).to(tl.float32) - mean[:, None]) * inv[:, None]
    if HAS_W:
        v = v * tl.load(W + cols).to(tl.float32)[None, :]
    if HAS_MOD:
        scale = tl.load(SCALE + mod[:, None] * SS + cols[None, :], inside[:, None], 0).to(X.dtype.element_ty).to(tl.float32)
        shift = tl.load(SHIFT + mod[:, None] * SH + cols[None, :], inside[:, None], 0).to(X.dtype.element_ty).to(tl.float32)
        v = v * (1.0 + scale) + shift
    if STORE_Y:
        tl.store(Y + r64[:, None] * K + cols[None, :], v.to(Y.dtype.element_ty), inside[:, None])
    if GROUP >= 4:
        v = _hadamard_step(v, BM, 256, 1)
    if GROUP >= 16:
        v = _hadamard_step(v, BM, 256, 4)
    if GROUP >= 64:
        v = _hadamard_step(v, BM, 256, 16)
    if GROUP >= 256:
        v = _hadamard_step(v, BM, 256, 64)
    return v


@triton.jit(do_not_specialize=_INTS, do_not_specialize_on_alignment=_INTS)
def _rms_modulate_quantize(X, W, SCALE, SHIFT, ROWMOD, Q, QS, Y, rows, LDX, SS, SH, EPS, COUNTER, FIRST, STEP, K: tl.constexpr, BM: tl.constexpr,
                           GROUP: tl.constexpr, SB: tl.constexpr, NORM: tl.constexpr, HAS_W: tl.constexpr, WANT_Y: tl.constexpr, AT: tl.constexpr,
                           LAYER: tl.constexpr, HAS_MOD: tl.constexpr):
    """Rows r of X (BM per program): v = X[r] / RMS(X[r]) (times W) * (1 + SCALE[m]) + SHIFT[m], m = ROWMOD[r], in FP32; Y[r] = v in X's
    dtype (WANT_Y); v rotated per GROUP columns, then Q[r] = rint(v * 127 / amax) and QS[r] = amax * NORM / 127 per SB columns (SB 0:
    one scale per row, its amax taken in a pass of its own over the rotated row, which the quantization pass makes again).
    AT: X's and ROWMOD's rows from FIRST + counter * STEP (a CUDA graph loop's turn, the counter on the device)."""
    if AT:
        base = (FIRST + tl.load(COUNTER) * STEP).to(tl.int64)
        X += base * LDX
        ROWMOD += base
    r = tl.program_id(0) * BM + tl.arange(0, BM)
    inside = r < rows
    r64 = r.to(tl.int64)
    offsets = tl.arange(0, 256)
    mean, inv = _statistics(X, r64, inside, offsets, LDX, EPS, K, BM, LAYER)
    mod = r64
    if HAS_MOD:
        mod = tl.load(ROWMOD + r, inside, 0).to(tl.int64)
    if SB == 0:
        amax = tl.zeros((BM,), tl.float32)
        for c in range(0, K, 256):
            v = _rotated(X, W, SCALE, SHIFT, Y, r64, inside, mean, inv, mod, c + offsets, LDX, SS, SH, K, BM, GROUP, HAS_W, HAS_MOD, WANT_Y)
            amax = tl.maximum(amax, tl.max(tl.abs(v), axis=1))
        factor = tl.where(amax > 0.0, tl.div_rn(127.0, tl.where(amax > 0.0, amax, 1.0)), 0.0)
        for c in range(0, K, 256):
            cols = c + offsets
            v = _rotated(X, W, SCALE, SHIFT, Y, r64, inside, mean, inv, mod, cols, LDX, SS, SH, K, BM, GROUP, HAS_W, HAS_MOD, False)
            tl.store(Q + r64[:, None] * K + cols[None, :], libdevice.nearbyint(v * factor[:, None]).to(tl.int8), inside[:, None])
        tl.store(QS + r64, tl.div_rn(amax * NORM, 127.0), inside)
    else:
        for c in range(0, K, 256):
            cols = c + offsets
            v = _rotated(X, W, SCALE, SHIFT, Y, r64, inside, mean, inv, mod, cols, LDX, SS, SH, K, BM, GROUP, HAS_W, HAS_MOD, WANT_Y)
            b = v.reshape((BM, 256 // SB, SB))
            amax = tl.max(tl.abs(b), axis=2)
            factor = tl.where(amax > 0.0, tl.div_rn(127.0, tl.where(amax > 0.0, amax, 1.0)), 0.0)
            q = libdevice.nearbyint(b * factor[:, :, None]).to(tl.int8).reshape((BM, 256))
            tl.store(Q + r64[:, None] * K + cols[None, :], q, inside[:, None])
            blocks = c // SB + tl.arange(0, 256 // SB)
            tl.store(QS + r64[:, None] * (K // SB) + blocks[None, :], tl.div_rn(amax * NORM, 127.0), inside[:, None])


@triton.jit(do_not_specialize=_INTS, do_not_specialize_on_alignment=_INTS)
def _rms_modulate_fp8(X, W, SCALE, SHIFT, ROWMOD, Q, P, S, Y, rows, LDX, SS, SH, EPS, COUNTER, FIRST, STEP, MINV, MAXV, K: tl.constexpr,
                      BM: tl.constexpr, HAS_W: tl.constexpr, WANT_Y: tl.constexpr, AT: tl.constexpr, CAST: tl.constexpr, LAYER: tl.constexpr,
                      HAS_MOD: tl.constexpr):
    """The norm and modulation of _rms_modulate_quantize, for FP8 activations with one scale for all the rows. CAST False: P[program] = the
    absmax of its rows' values (Y[r] = v with WANT_Y). CAST: Q[r] = clamp(v / S, MINV, MAXV) in Q's FP8 dtype, S the FP32 scale."""
    if AT:
        base = (FIRST + tl.load(COUNTER) * STEP).to(tl.int64)
        X += base * LDX
        ROWMOD += base
    r = tl.program_id(0) * BM + tl.arange(0, BM)
    inside = r < rows
    r64 = r.to(tl.int64)
    offsets = tl.arange(0, 256)
    mean, inv = _statistics(X, r64, inside, offsets, LDX, EPS, K, BM, LAYER)
    mod = r64
    if HAS_MOD:
        mod = tl.load(ROWMOD + r, inside, 0).to(tl.int64)
    if CAST:
        scale = tl.load(S)
        for c in range(0, K, 256):
            cols = c + offsets
            v = _rotated(X, W, SCALE, SHIFT, Y, r64, inside, mean, inv, mod, cols, LDX, SS, SH, K, BM, 0, HAS_W, HAS_MOD, False)
            q = tl.minimum(tl.maximum(tl.div_rn(v, scale), MINV), MAXV)
            tl.store(Q + r64[:, None] * K + cols[None, :], q.to(Q.dtype.element_ty), inside[:, None])
    else:
        amax = tl.zeros((BM,), tl.float32)
        for c in range(0, K, 256):
            v = _rotated(X, W, SCALE, SHIFT, Y, r64, inside, mean, inv, mod, c + offsets, LDX, SS, SH, K, BM, 0, HAS_W, HAS_MOD, WANT_Y)
            amax = tl.maximum(amax, tl.max(tl.where(inside[:, None], tl.abs(v), 0.0), axis=1))
        tl.store(P + tl.program_id(0), tl.max(amax, axis=0))


def supported(x, weight):
    """Whether rms_modulate_quantize takes these rows (a CUDA 16-bit (rows, K) view of rows one stride apart, K a multiple of 256)."""
    return (x.is_cuda and x.dim() == 2 and x.dtype in (torch.bfloat16, torch.float16) and x.stride(1) == 1 and x.shape[1] % CHUNK == 0
            and x.shape[1] <= 16384 and (weight is None or weight.dtype == x.dtype and weight.is_contiguous()))


def row_fills(x, weight, eps, scale, shift, segments, layer=False):
    """(fill_prepared, fill_prepared_at) of an mmgp offload.RowSource of the norm of rows x (rows, K) (RMSNorm, or LayerNorm with layer:
    no bias; times weight when given) modulated by segments (start, stop, row) of the (rows, K) tables scale / shift (both None: no
    modulation): rms_modulate_quantize at given rows or at a CUDA graph loop's; (None, None) where that kernel does not take these rows."""
    from shared.kernels.row_loop_triton import segment_runs
    modulated = scale is not None
    if (not supported(x, weight) or modulated and (scale.dim() != 2 or shift.dim() != 2 or scale.stride(1) != 1 or shift.stride(1) != 1
                                                   or scale.dtype != shift.dtype)):
        return None, None
    rowmod = None
    if modulated:
        rowmod = torch.empty(x.shape[0], dtype=torch.int32, device=x.device)  # each row's modulation row
        for start, stop, row0, offset, span in segment_runs(segments):
            rowmod[start:stop] = torch.arange(offset, offset + stop - start, dtype=torch.int32, device=x.device).floor_divide_(span).add_(row0)

    def fill_prepared(start, stop, form, want_rows):
        return rms_modulate_quantize(x[start:stop], weight, eps, scale, shift, None if rowmod is None else rowmod[start:stop], form, want_rows, layer=layer)

    def fill_prepared_at(counter, first, rows, form, want_rows):
        return rms_modulate_quantize(x, weight, eps, scale, shift, rowmod, form, want_rows, (counter, first, rows), layer)
    return fill_prepared, fill_prepared_at


def rms_modulate_quantize(x, weight, eps, scale, shift, rowmod, form, want_y, at=None, layer=False):
    """(y or None, q, scales) of rows x (rows, K): the RMSNorm (LayerNorm with layer: no bias), times weight when given, the modulation
    of row r by scale / shift row rowmod[r] (int32, one per row; scale, shift and rowmod None: none), rotated by ConvRot groups of `group`
    columns (0: none) and quantized as the WGPKernels block path does, one scale per `block` columns (0: one per row, scales (rows, 1)):
    form = (group, block), a 'block' form of mmgp offload (the row projections' block_form), or ("fp8", dtype) (_fp8).
    at: (counter, first, rows) to read `rows` rows of x and rowmod from first + counter * rows instead (a CUDA graph loop's turn)."""
    if form[0] == "fp8":
        return _fp8(x, weight, eps, scale, shift, rowmod, form[1], want_y, at, layer)
    group, block = form
    rows, k = x.shape if at is None else (at[2], x.shape[1])
    q = torch.empty((rows, k), dtype=torch.int8, device=x.device)
    scales = torch.empty((rows, k // block if block else 1), dtype=torch.float32, device=x.device)
    y = torch.empty((rows, k), dtype=x.dtype, device=x.device) if want_y else None
    norm = {0: 1.0, 16: 0.25, 64: 0.125, 256: 0.0625}[group]
    modulated = scale is not None
    _rms_modulate_quantize[(triton.cdiv(rows, ROWS),)](x, x if weight is None else weight, scale if modulated else x, shift if modulated else x,
                                                      rowmod if modulated else x, q, scales, q if y is None else y, rows, x.stride(0),
                                                      scale.stride(0) if modulated else 0, shift.stride(0) if modulated else 0, eps,
                                                      x if at is None else at[0], 0 if at is None else at[1], rows, k, ROWS, group, block, norm,
                                                      weight is not None, want_y, at is not None, layer, modulated, num_warps=WARPS)
    return y, q, scales


def _fp8(x, weight, eps, scale, shift, rowmod, fp8_dtype, want_y, at, layer):
    """rms_modulate_quantize's FP8 form ("fp8", dtype): (y or None, the rows in fp8_dtype, their one FP32 scale (shape ())) as the FP8
    activation quantizer makes them (shared.qtypes.scaled_fp8: absmax / the format's max, 1 for zero rows), in two passes over x: the
    absmax of the modulated rows (per program, reduced in a fixed order), then the cast."""
    minv, maxv = float(torch.finfo(fp8_dtype).min), float(torch.finfo(fp8_dtype).max)
    rows, k = x.shape if at is None else (at[2], x.shape[1])
    programs = triton.cdiv(rows, ROWS)
    q = torch.empty((rows, k), dtype=fp8_dtype, device=x.device)
    partials = torch.empty(programs, dtype=torch.float32, device=x.device)
    y = torch.empty((rows, k), dtype=x.dtype, device=x.device) if want_y else None
    modulated = scale is not None
    common = (rows, x.stride(0), scale.stride(0) if modulated else 0, shift.stride(0) if modulated else 0, eps, x if at is None else at[0],
              0 if at is None else at[1], rows, minv, maxv, k, ROWS, weight is not None, want_y, at is not None)
    tables = (x if weight is None else weight, scale if modulated else x, shift if modulated else x, rowmod if modulated else x)
    _rms_modulate_fp8[(programs,)](x, *tables, q, partials, partials, q if y is None else y, *common, False, layer, modulated, num_warps=WARPS)
    absmax = partials.amax()
    s = torch.where(absmax > 0, absmax / maxv, torch.ones_like(absmax)).reshape(())
    _rms_modulate_fp8[(programs,)](x, *tables, q, partials, s, q if y is None else y, *common, True, layer, modulated, num_warps=WARPS)
    return y, q, s


from shared.kernels.triton_compilation_log import install_triton_compilation_logger
install_triton_compilation_logger()
