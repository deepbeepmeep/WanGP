# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""ConvRot INT8 linears in Triton: tile-local rotation for small decode inputs (no activation workspace), and for large inputs a
rotation fused into the activation quantization, by chunks of rows, followed by a block-scaled INT8 GEMM."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from shared.kernels.triton_tuner import LaunchTuner


@triton.jit
def _regular_step(a, BM: tl.constexpr, STRIDE: tl.constexpr):
    v = a.reshape((BM, 256 // (4 * STRIDE), 4, STRIDE))
    reverse = tl.broadcast_to((3 - tl.arange(0, 4))[None, None, :, None], (BM, 256 // (4 * STRIDE), 4, STRIDE))
    return (tl.sum(v, axis=2)[:, :, None, :] - 2.0 * tl.gather(v, reverse, axis=2)).reshape((BM, 256))


@triton.jit
def _convrot_int8_kernel(x, w, scale, out, bias, M, N, K,
                         SX0: tl.constexpr, SX1: tl.constexpr, SW0: tl.constexpr, SW1: tl.constexpr,
                         BM: tl.constexpr, BN: tl.constexpr, HAS_BIAS: tl.constexpr):
    pid = tl.program_id(0)
    tiles_m = tl.cdiv(M, BM)
    rows = (pid % tiles_m) * BM + tl.arange(0, BM)
    cols = (pid // tiles_m) * BN + tl.arange(0, BN)
    kk = tl.arange(0, 256)
    qk = tl.arange(0, 64)
    acc = tl.full((BM, BN), 0, tl.float32)
    for base in range(0, K, 256):
        a = tl.load(x + rows[:, None] * SX0 + (base + kk[None, :]) * SX1, rows[:, None] < M, 0).to(tl.float32)
        a = _regular_step(a, BM, 1)
        a = _regular_step(a, BM, 4)
        a = _regular_step(a, BM, 16)
        a = _regular_step(a, BM, 64)
        # Preserve BF16 rounding before the existing 64-element INT8 quantization.
        a = (a * 0.0625).to(x.dtype.element_ty).to(tl.float32).reshape((BM, 4, 64))
        scales = tl.max(tl.abs(a), axis=2) / 127.0
        scales = tl.where(scales > 0, scales, 1.0)
        quant = libdevice.nearbyint(a / scales[:, :, None])
        quant = tl.maximum(tl.minimum(quant, 127), -128).to(tl.int8).reshape((BM, 256))
        for g in tl.static_range(4):
            qa = tl.gather(quant, tl.broadcast_to((g * 64 + qk)[None, :], (BM, 64)), axis=1)
            s = tl.gather(scales, tl.full((BM, 1), g, tl.int32), axis=1).reshape((BM,))
            b = tl.load(w + cols[None, :] * SW0 + (base + g * 64 + qk[:, None]) * SW1, cols[None, :] < N, 0).to(tl.int8)
            acc += tl.dot(qa, b).to(tl.float32) * s[:, None]
    result = acc * tl.load(scale + cols, cols < N, 0)[None, :].to(tl.float32)
    if HAS_BIAS:
        result = result.to(out.dtype.element_ty).to(tl.float32) + tl.load(bias + cols, cols < N, 0)[None, :].to(tl.float32)
    tl.store(out + rows[:, None] * N + cols[None, :], result, (rows[:, None] < M) & (cols[None, :] < N))


def convrot_int8_mm(x, weight, scale, out, bias, cfg):
    m, k = x.shape
    n = weight.shape[0]
    bm, bn, _, warps, stages = cfg
    _convrot_int8_kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](x, weight, scale, out, bias, m, n, k, *x.stride(), *weight.stride(), bm, bn, bias is not None, num_warps=warps, num_stages=stages)


# Large inputs: the rotation and the INT8 quantization of the activations in one pass over a chunk of rows (INT8 + one FP32 scale per row
# and 64 columns, no rotated copy), then an INT8 GEMM applying those scales; rows run in chunks bounded by SCRATCH_BYTES.
SCALE_BLOCK = 64
SCRATCH_BYTES = 16 * 1024 * 1024


@triton.jit
def _hadamard_step(a, BM: tl.constexpr, SEG: tl.constexpr, STRIDE: tl.constexpr):
    """The 4-point regular Hadamard (rows of ones with -1 at 3 - i) over elements STRIDE apart, summed in a fixed order whatever the launch
    configuration: tuned and untuned launches give the same bits."""
    v = tl.permute(a.reshape((BM, SEG // (4 * STRIDE), 2, 2, STRIDE)), (0, 1, 4, 2, 3))
    even, odd = tl.split(v)
    e0, e2 = tl.split(even)
    e1, e3 = tl.split(odd)
    total = (e0 + e1) + (e2 + e3)
    out = tl.join(tl.join(total - 2.0 * e3, total - 2.0 * e2), tl.join(total - 2.0 * e1, total - 2.0 * e0))
    return tl.permute(out, (0, 1, 4, 3, 2)).reshape((BM, SEG))


@triton.jit(do_not_specialize=["m", "stride_x", "first", "step"], do_not_specialize_on_alignment=["x", "q", "scales"])  # ranges of columns start at any 4-byte scale
def _rotate_quantize_kernel(x, q, scales, m, k, stride_x, BM: tl.constexpr, SEG: tl.constexpr, GROUP: tl.constexpr, NORM: tl.constexpr,
                            BK: tl.constexpr, counter=None, first=0, step=0, AT: tl.constexpr = False):
    if AT:  # x's rows from first + counter * step (a CUDA graph loop's turn)
        x += (first + tl.load(counter) * step) * stride_x
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * SEG + tl.arange(0, SEG)
    inside = rows[:, None] < m
    a = tl.load(x + rows[:, None].to(tl.int64) * stride_x + cols[None, :], inside, 0).to(tl.float32)
    a = _hadamard_step(a, BM, SEG, 1)
    a = _hadamard_step(a, BM, SEG, 4)
    if GROUP >= 64:
        a = _hadamard_step(a, BM, SEG, 16)
    if GROUP >= 256:
        a = _hadamard_step(a, BM, SEG, 64)
    v = (a * NORM).reshape((BM, SEG // BK, BK))  # NORM = 1 / sqrt(GROUP), a power of two: exact
    scale = tl.div_rn(tl.max(tl.abs(v), axis=2), 127.0)
    scale = tl.where(scale > 0.0, scale, 1.0)
    quant = libdevice.nearbyint(tl.div_rn(v, tl.broadcast_to(scale[:, :, None], (BM, SEG // BK, BK))))
    quant = tl.maximum(tl.minimum(quant, 127.0), -127.0).to(tl.int8).reshape((BM, SEG))
    tl.store(q + rows[:, None].to(tl.int64) * k + cols[None, :], quant, inside)
    blocks = tl.program_id(1) * (SEG // BK) + tl.arange(0, SEG // BK)
    tl.store(scales + rows[:, None].to(tl.int64) * (k // BK) + blocks[None, :], scale, inside)


@triton.jit(do_not_specialize=["m"], do_not_specialize_on_alignment=["q", "scales", "out"])  # ranges of rows of a kept input
def _blockscale_gemm_kernel(q, scales, w, weight_scale, bias, out, m, n, k, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                            HAS_BIAS: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    offsets = tl.arange(0, BK)
    blocks = k // BK
    rows64, cols64 = rows.to(tl.int64), cols.to(tl.int64)
    acc = tl.zeros((BM, BN), tl.float32)
    for block in range(0, blocks):
        kk = block * BK + offsets
        a = tl.load(q + rows64[:, None] * k + kk[None, :], rows[:, None] < m, 0)
        b = tl.load(w + cols64[None, :] * k + kk[:, None], cols[None, :] < n, 0)
        s = tl.load(scales + rows64 * blocks + block, rows < m, 0)
        acc += tl.dot(a, b).to(tl.float32) * s[:, None]
    acc = acc * tl.load(weight_scale + cols, cols < n, 0).to(tl.float32)[None, :]
    if HAS_BIAS:
        acc += tl.load(bias + cols, cols < n, 0).to(tl.float32)[None, :]
    tl.store(out + rows64[:, None] * n + cols[None, :], acc, (rows[:, None] < m) & (cols[None, :] < n))


_TUNER = LaunchTuner("wan2gp_convrot_int8_triton", 2, "INT8")
_QUANTIZE_CANDIDATES = ((8, 8), (8, 4), (16, 8), (16, 4), (32, 4), (32, 8))  # (rows per program, warps)
_GEMM_CANDIDATES = ((64, 128, 8, 3), (64, 128, 4, 3), (128, 64, 4, 4), (64, 128, 8, 4), (128, 128, 8, 3), (128, 256, 8, 2), (128, 256, 8, 3),
                    (64, 256, 8, 3))  # (BM, BN, warps, stages)
_TUNE_ROWS = 2048


def _quantize(x, q, scales, count, k, group, config, at=None):
    """at: (counter, first, step), x's rows from first + counter * step instead of its first rows."""
    rows, warps = config
    seg = max(group, SCALE_BLOCK)
    _rotate_quantize_kernel[(triton.cdiv(count, rows), k // seg)](x, q, scales, count, k, x.stride(0), rows, seg, group, group ** -0.5, SCALE_BLOCK,
                                                                 *(() if at is None else (*at, True)), num_warps=warps)


def _gemm(q, scales, weight, scale, bias, out, count, n, k, config):
    bm, bn, warps, stages = config
    _blockscale_gemm_kernel[(triton.cdiv(count, bm), triton.cdiv(n, bn))](q, scales, weight, scale, bias, out, count, n, k, bm, bn, SCALE_BLOCK,
                                                                          bias is not None, num_warps=warps, num_stages=stages)


def _quantize_identity(k, group, dtype):
    return f"k{k} g{group} {str(dtype)[6:]}"


def _gemm_identity(n, k, dtype, bias):
    return f"n{triton.next_power_of_2(n)} k{k} {str(dtype)[6:]}{' bias' if bias is not None else ''}"


def _gemm_candidates(n):
    static = (128, 256, 8, 2) if n >= 4096 else (64, 128, 8, 3)
    return (static,) + tuple(c for c in _GEMM_CANDIDATES if c != static)


def scratch_rows(k):
    """Rows per launch of rotated_linear for an input of k columns (SCRATCH_BYTES of INT8 activations and scales)."""
    return max(128, SCRATCH_BYTES // (k + 4 * (k // SCALE_BLOCK)) // 128 * 128)


def loop_configs(dtype, device, n, k, group, bias):
    """rotated_linear's (quantize, GEMM) launch configurations as its calls inside a CUDA graph capture take them: the tuned winners, else the
    static ones (no tuning)."""
    return (_TUNER.config("rotate_quantize", _quantize_identity(k, group, dtype), device, _QUANTIZE_CANDIDATES, None, False),
            _TUNER.config("blockscale_gemm", _gemm_identity(n, k, dtype, bias), device, _gemm_candidates(n), None, False))


def rotated_linear(x, weight, scale, bias, group):
    """ConvRot INT8 linear for large inputs: x (M, K) BF16/FP16 with unit column stride; weight (N, K) contiguous INT8, rotated rows; scale
    (N,) FP32; group 16, 64 or 256, K a multiple of max(group, 64). The launch configurations are tuned on scratch copies of the call's
    first rows (LaunchTuner)."""
    m, k = x.shape
    n = weight.shape[0]
    rows = scratch_rows(k)
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    q = torch.empty((min(rows, m), k), dtype=torch.int8, device=x.device)
    scales = torch.empty((min(rows, m), k // SCALE_BLOCK), dtype=torch.float32, device=x.device)
    quantize_config = gemm_config = None
    for start in range(0, m, rows):
        count = min(rows, m - start)
        part = x[start:start + count]
        if quantize_config is None:
            def quantize_scratch():
                sample = part[:_TUNE_ROWS].clone()
                q_scratch, scales_scratch = torch.empty_like(q[:len(sample)]), torch.empty_like(scales[:len(sample)])
                return lambda config: _quantize(sample, q_scratch, scales_scratch, len(sample), k, group, config)
            quantize_config = _TUNER.config("rotate_quantize", _quantize_identity(k, group, x.dtype), x.device, _QUANTIZE_CANDIDATES, quantize_scratch,
                                            count >= 256)
        _quantize(part, q, scales, count, k, group, quantize_config)
        if gemm_config is None:
            def gemm_scratch():
                sample_q, sample_scales = q[:min(count, _TUNE_ROWS)].clone(), scales[:min(count, _TUNE_ROWS)].clone()
                out_scratch = torch.empty((len(sample_q), n), dtype=x.dtype, device=x.device)
                return lambda config: _gemm(sample_q, sample_scales, weight, scale, bias, out_scratch, len(sample_q), n, k, config)
            gemm_config = _TUNER.config("blockscale_gemm", _gemm_identity(n, k, x.dtype, bias), x.device, _gemm_candidates(n), gemm_scratch, count >= 256)
        _gemm(q, scales, weight, scale, bias, out[start:start + count], count, n, k, gemm_config)
    return out


def quantize_columns(x, q, scales, first, group, block=SCALE_BLOCK):
    """x (M, columns) BF16/FP16 with unit column stride, whole rotation groups and scale blocks (max(group, block) columns each),
    rotated and quantized in place into the columns [first, first + columns) of q (M, K) INT8 and their scales (M, K / block), as
    rotated_linear quantizes its input: nothing in it spans more columns, so an input received by ranges of columns is quantized
    as it would be whole. block: columns per activation scale (128: the form the WGPKernels block GEMM reads)."""
    (m, k), columns, seg = q.shape, x.shape[1], max(group, block)

    def launch(source, target, target_scales, count, width, row_stride, config):
        rows, warps = config
        _rotate_quantize_kernel[(triton.cdiv(count, rows), columns // seg)](source, target, target_scales, count, width, row_stride, rows, seg, group,
                                                                            group ** -0.5, block, num_warps=warps)

    def scratch():
        sample = x[:_TUNE_ROWS].clone()
        q_scratch = torch.empty((len(sample), columns), dtype=torch.int8, device=x.device)
        scales_scratch = torch.empty((len(sample), columns // block), dtype=torch.float32, device=x.device)
        return lambda config: launch(sample, q_scratch, scales_scratch, len(sample), columns, sample.stride(0), config)
    identity = f"k{k} g{group}{'' if block == SCALE_BLOCK else f' b{block}'} {str(x.dtype)[6:]}"
    config = _TUNER.config("rotate_quantize", identity, x.device, _QUANTIZE_CANDIDATES, scratch, m >= 256)
    launch(x, q[:, first:], scales[:, first // block:], m, k, x.stride(0), config)


def prepared_linear(q, scales, weight, scale, bias, out):
    """rotated_linear's block-scaled GEMM on an input quantized beforehand (quantize_columns), written into out (M, N): its rows as
    rotated_linear's."""
    (m, k), n, dtype = q.shape, weight.shape[0], out.dtype

    def scratch():
        sample_q, sample_scales = q[:_TUNE_ROWS].clone(), scales[:_TUNE_ROWS].clone()
        out_scratch = torch.empty((len(sample_q), n), dtype=dtype, device=q.device)
        return lambda config: _gemm(sample_q, sample_scales, weight, scale, bias, out_scratch, len(sample_q), n, k, config)
    config = _TUNER.config("blockscale_gemm", _gemm_identity(n, k, dtype, bias), q.device, _gemm_candidates(n), scratch, m >= 256)
    _gemm(q, scales, weight, scale, bias, out, m, n, k, config)
    return out


from shared.kernels.triton_compilation_log import install_triton_compilation_logger
install_triton_compilation_logger()
