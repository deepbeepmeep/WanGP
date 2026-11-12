# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Weight-only FP8 matmul (W8A16) for block-scaled FP8 weights (shared.qtypes.block_scaled: E4M3 (N, K) with FP32 scales per 128 x 128
block or per output channel and 128-element block of K) on every CUDA GPU, with 16-bit activations: the weights are decoded exactly in
registers, each 128-element block of K accumulated in FP32 on the tensor cores, then promoted with its scales into FP32 accumulators
(the block semantics of DeepGEMM's FP8 GEMM, with exact activations instead of E4M3 ones). The weight matrix is never materialized and
the activations are never quantized: more precise than the reference dequantization + 16-bit matmul (whose dequantized weights are
rounded to the compute dtype) and than the FP8 tensor-core kernels (whose activations are quantized to E4M3).

Tensor cores: sm80+ multiply the activations in their own dtype (BF16 or FP16 MMAs); Turing (sm75) has FP16 MMAs only, so BF16
activations are first written as FP16 with one power-of-two scale per row and 128-element block (exact for BF16 values within 2^28
of their block's maximum: the scaling is exact and FP16 holds BF16's 8-bit mantissa), the scale folded into the promotion. GPUs before
sm89 cannot convert float8 in Triton: the E4M3 bytes are placed as FP16 bits (the value times 2^-8, exact, subnormals included) with
2^8 folded into the promotion, or decoded with integer operations for BF16 MMAs.

Tile configurations are static per GPU generation and tuned once per layer shape and token-count slot on a real call (outside CUDA
graph recording and compilation), cached in ~/.triton/autotune/; the kernel is not specialized on the number of tokens, so a new
resolution compiles nothing. WAN2GP_FP8_TRITON_AUTOTUNE=0 keeps the static configurations."""
import json
import os
import threading
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

_CACHE_VERSION = 3
_CACHE_PATH = Path(os.environ.get("WAN2GP_FP8_BLOCK_TRITON_CACHE", Path.home() / ".triton" / "autotune" / "wan2gp_fp8_block_cache.json")).expanduser()
_TUNING = os.environ.get("WAN2GP_FP8_TRITON_AUTOTUNE", "1") != "0"
DECODE_NATIVE, DECODE_BITS_F32, DECODE_BITS_F16 = 0, 1, 2
WS_BLOCK, WS_GROUP = 0, 3  # weight scale layouts (the values of wgpkernels.fp8)
# (BLOCK_M, BLOCK_N, BLOCK_K, warps, stages); the first of each slot is its static configuration. Turing: 64 KiB of shared memory.
# A 128-element k-tile holds one scale block, so every iteration promotes (RTX 2080 Ti: 43 TFLOPS against 7 for 32-element k-tiles).
_CANDIDATES = {
    "sm75": {"small": [(32, 64, 128, 4, 3), (64, 64, 128, 4, 3), (16, 64, 128, 4, 3), (32, 128, 128, 4, 2)],
             "large": [(64, 64, 128, 4, 3), (64, 128, 128, 4, 2), (128, 64, 128, 4, 2), (64, 64, 128, 4, 4), (64, 64, 128, 8, 3), (128, 128, 32, 8, 3)]},
    "sm80": {"small": [(32, 64, 128, 4, 4), (64, 64, 128, 4, 4), (64, 128, 64, 4, 3)],
             "large": [(128, 128, 128, 4, 3), (64, 128, 128, 4, 4), (128, 64, 128, 4, 4), (128, 128, 64, 4, 4), (128, 128, 128, 8, 3), (128, 256, 64, 8, 3)]},
}
_lock = threading.Lock()
_tuned = None
_device_names = {}


@triton.jit
def _bf16_bits_to_f32(bits):
    return (bits.to(tl.uint32) << 16).to(tl.float32, bitcast=True)


@triton.jit
def _f32_to_bf16_bits(value):
    # Round to nearest even without BF16 instructions (unavailable before sm80).
    bits = value.to(tl.uint32, bitcast=True)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return tl.where(value != value, 0x7FC0, rounded).to(tl.uint16)


@triton.jit
def _e4m3_bits_to_f32(bits):
    """float8_e4m3fn bytes as fp32 (GPUs without float8 conversions)."""
    b = bits.to(tl.int32)
    e = (b >> 3) & 0xF
    m = b & 0x7
    value = tl.where(e == 0, m.to(tl.float32) * 0.001953125, (((e + 120) << 23) | (m << 20)).to(tl.float32, bitcast=True))
    value = tl.where((b & 0x7F) == 0x7F, float("nan"), value)
    return tl.where((b & 0x80) != 0, -value, value)


@triton.jit
def _e4m3_bits_to_f16_scaled(bits):
    """float8_e4m3fn bytes as fp16 values times 2^-8 (exact: the exponent and mantissa bits move into the low FP16 exponent and high
    mantissa bits, E4M3 subnormals becoming FP16 subnormals of the same mantissa)."""
    b = bits.to(tl.uint16)
    return (((b & 0x80) << 8) | ((b & 0x7F) << 7)).to(tl.float16, bitcast=True)


@triton.jit(do_not_specialize=["M"])
def _prepare_a16_kernel(x, q, sa, M, K, sxm, sxk, kblocks, BM: tl.constexpr, EVEN_K: tl.constexpr):
    """FP16 rows of BF16 x (bits) with one power-of-two scale per row and 128-element block, the block's maximum scaled into [2^14, 2^15)."""
    pid_m, kb = tl.program_id(0), tl.program_id(1)
    rm = (pid_m * BM + tl.arange(0, BM)).to(tl.int64)  # 64 bits: rows x width passes 2^31 on long high resolution videos
    rk = kb * 128 + tl.arange(0, 128)
    mask = rm[:, None] < M
    if not EVEN_K:
        mask = mask & (rk[None, :] < K)
    v = _bf16_bits_to_f32(tl.load(x + rm[:, None] * sxm + rk[None, :] * sxk, mask=mask, other=0))
    amax = tl.max(tl.abs(v), axis=1)
    exponent = (amax.to(tl.uint32, bitcast=True) >> 23) & 0xFF  # biased exponent of the block maximum (0 for an all-zero block)
    scale_bits = tl.where(amax > 0, tl.maximum(exponent, 15) - 14, 127).to(tl.uint32) << 23  # 2^(floor(log2 amax) - 14), 1 for zeros
    scale = scale_bits.to(tl.float32, bitcast=True)
    tl.store(sa + rm * kblocks + kb, scale, mask=rm < M)
    tl.store(q + rm[:, None] * K + rk[None, :], (v * (1.0 / scale)[:, None]).to(tl.float16), mask=mask)


@triton.jit(do_not_specialize=["M"])  # one binary per configuration whatever the number of tokens
def _block_kernel(x, w, sa, sb, bias, out, M, N, K, kblocks, sxm, sxk, swn, swk, som, son,
                  WMODE: tl.constexpr, HAS_SA: tl.constexpr, HAS_BIAS: tl.constexpr, DECODE: tl.constexpr, BITS_IO: tl.constexpr,
                  EVEN_K: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
    """out = x @ (w * scales).T + bias, x (M, K) FP16 or BF16 (BITS_IO: the BF16 bias and output are uint16 bits, for GPUs without BF16
    conversions, whose x is FP16), w (N, K) E4M3 bytes (uint8 when DECODE != 0),
    sa (M, kblocks) FP32 row-block scales or none, sb (ceil(N / 128), kblocks) block scales or (N, kblocks) group scales: each 128-element
    block of K is accumulated from zero in FP32 and promoted with sa[row] * sb[block] (times 2^8 for the FP16-bits decode)."""
    pid = tl.program_id(0)
    tiles_m, tiles_n = tl.cdiv(M, BM), tl.cdiv(N, BN)
    group = GROUP_M * tiles_n
    first_m = (pid // group) * GROUP_M
    size_m = tl.minimum(tiles_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group) % size_m
    pid_n = (pid % group) // size_m
    rm = (pid_m * BM + tl.arange(0, BM)).to(tl.int64)  # 64 bits: rows x width passes 2^31 on long high resolution videos
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    m_mask, n_mask = rm < M, rn < N
    a_ptr = x + rm[:, None] * sxm + rk[None, :] * sxk
    b_ptr = w + rn[None, :] * swn + rk[:, None] * swk
    if WMODE == 0:
        sb_ptr = sb + (rn // 128) * kblocks
    else:
        sb_ptr = sb + rn * kblocks
    accf = tl.zeros((BM, BN), dtype=tl.float32)
    KPB: tl.constexpr = 128 // BK
    for kb in range(0, kblocks):
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for j in tl.static_range(KPB):
            if EVEN_K:
                a = tl.load(a_ptr, mask=m_mask[:, None], other=0)
                b = tl.load(b_ptr, mask=n_mask[None, :])
            else:
                k_mask = rk < K - (kb * KPB + j) * BK
                a = tl.load(a_ptr, mask=m_mask[:, None] & k_mask[None, :], other=0)
                b = tl.load(b_ptr, mask=n_mask[None, :] & k_mask[:, None])
            if DECODE == 2:
                b = _e4m3_bits_to_f16_scaled(b)
            elif DECODE == 1:
                b = _e4m3_bits_to_f32(b).to(a.dtype)
            else:
                b = b.to(tl.float32).to(a.dtype)
            if not EVEN_K:  # columns beyond K would multiply undefined bytes into the sums
                b = tl.where(k_mask[:, None], b, 0.0)
            acc = tl.dot(a, b, acc)
            a_ptr += BK * sxk
            b_ptr += BK * swk
        s = tl.load(sb_ptr + kb, mask=n_mask, other=0.0)[None, :]
        if DECODE == 2:
            s = s * 256.0
        if HAS_SA:
            s = s * tl.load(sa + rm * kblocks + kb, mask=m_mask, other=0.0)[:, None]
        accf += acc * s
    if HAS_BIAS:
        bias_value = tl.load(bias + rn, mask=n_mask, other=0)[None, :]
        if BITS_IO:
            bias_value = _bf16_bits_to_f32(bias_value)
        else:
            bias_value = bias_value.to(tl.float32)
        accf += bias_value
    out_ptr = out + rm[:, None] * som + rn[None, :] * son
    if BITS_IO:
        tl.store(out_ptr, _f32_to_bf16_bits(accf), mask=m_mask[:, None] & n_mask[None, :])
    else:
        tl.store(out_ptr, accf.to(out.dtype.element_ty), mask=m_mask[:, None] & n_mask[None, :])


def scale_mode(scale, n, k):
    """WS_BLOCK for (ceil(N / 128), ceil(K / 128)) scales, WS_GROUP for (N, ceil(K / 128))."""
    kblocks = -(-k // 128)
    if scale.dim() == 2 and tuple(scale.shape) == (-(-n // 128), kblocks):
        return WS_BLOCK
    if scale.dim() == 2 and tuple(scale.shape) == (n, kblocks):
        return WS_GROUP
    raise ValueError(f"FP8 weight scales of shape {tuple(scale.shape)} match neither ({-(-n // 128)}, {kblocks}) blocks nor ({n}, {kblocks}) channel blocks")


def prepare_a16(x):
    """(FP16 copy of BF16 x (M, K), FP32 scales (M, ceil(K / 128))): x = q * scale per row and 128-element block, exactly for values within
    2^28 of their block's maximum (Turing's FP16 tensor cores)."""
    m, k = x.shape
    kblocks = -(-k // 128)
    q = torch.empty((m, k), dtype=torch.float16, device=x.device)
    sa = torch.empty((m, kblocks), dtype=torch.float32, device=x.device)
    bm = 32
    _prepare_a16_kernel[(triton.cdiv(m, bm), kblocks)](x.view(torch.uint16), q, sa, m, k, x.stride(0), x.stride(1), kblocks, bm, k % 128 == 0, num_warps=4)
    return q, sa


def _generation(device):
    return "sm75" if torch.cuda.get_device_capability(device) < (8, 0) else "sm80"


def _launch(a, sa, weight, scale, wmode, bias, out, config):
    bm, bn, bk, warps, stages = config
    m, k = a.shape
    n = weight.shape[0]
    capability = torch.cuda.get_device_capability(a.device)
    bits_io = out.dtype == torch.bfloat16 and capability < (8, 0)
    if a.dtype == torch.float16:
        decode = DECODE_BITS_F16 if capability < (8, 9) or os.environ.get("WAN2GP_FP8_TRITON_DECODE", "") == "bits" else DECODE_NATIVE
    else:
        decode = DECODE_NATIVE if capability >= (8, 9) and os.environ.get("WAN2GP_FP8_TRITON_DECODE", "") != "bits" else DECODE_BITS_F32
    w = weight if decode == DECODE_NATIVE else weight.view(torch.uint8)
    if bits_io:
        out = out.view(torch.uint16)
        if bias is not None:
            bias = bias.view(torch.uint16)
    kblocks = -(-k // 128)
    _block_kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
        a, w, sa, scale, bias, out, m, n, k, kblocks, a.stride(0), a.stride(1), w.stride(0), w.stride(1), out.stride(0), out.stride(1),
        wmode, sa is not None, bias is not None, decode, bits_io, k % 128 == 0, bm, bn, bk, 8, num_warps=warps, num_stages=stages)


def _load_cache():
    global _tuned
    if _tuned is None:
        try:
            payload = json.loads(_CACHE_PATH.read_text(encoding="utf-8"))
            _tuned = payload["configs"] if payload.get("version") == _CACHE_VERSION else {}
        except (OSError, ValueError, KeyError):
            _tuned = {}
    return _tuned


def _save_cache():
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        _CACHE_PATH.write_text(json.dumps({"version": _CACHE_VERSION, "configs": _tuned}, indent=1), encoding="utf-8")
    except OSError as exc:
        print(f"[FP8] Triton block tuning results not saved ({exc}).")


def _time(launch, reps):
    launch()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        launch()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / reps


def _tune(key, a, sa, weight, scale, wmode, bias, out, candidates):
    """The slot's configuration, timed on at most 2048 rows of the call (its own input and output), kept when at least 2% faster than
    the static one; configurations this GPU cannot run are skipped."""
    rows = min(a.shape[0], 2048)
    bench = (a[:rows], None if sa is None else sa[:rows], weight, scale, wmode, bias, out[:rows])
    timings = {}
    for config in candidates:
        try:
            timings[config] = _time(lambda: _launch(*bench, config), 3)
        except (triton.runtime.errors.OutOfResources, RuntimeError):
            continue
    for config in list(timings):  # a second alternated round: the first configurations timed are not penalized by the GPU warming up
        timings[config] = min(timings[config], _time(lambda: _launch(*bench, config), 3))
    best = min(timings, key=timings.get)
    if candidates[0] in timings and timings[best] * 1.02 > timings[candidates[0]]:
        best = candidates[0]
    with _lock:
        _load_cache()[key] = list(best)
        _save_cache()
    return best


def linear(x, weight, scale, bias=None, out=None):
    """x (M, K) BF16/FP16 @ (weight (N, K) E4M3 * scale per 128 x 128 block or per channel and 128-element block of K).T + bias, in x's
    dtype; K a multiple of 16."""
    m, k = x.shape
    n = weight.shape[0]
    wmode = scale_mode(scale, n, k)
    scale = scale if scale.dtype == torch.float32 and scale.is_contiguous() else scale.float().contiguous()
    out = torch.empty((m, n), dtype=x.dtype, device=x.device) if out is None else out
    if bias is not None and bias.dtype != x.dtype:
        bias = bias.to(x.dtype)
    generation = _generation(x.device)
    if x.dtype == torch.bfloat16 and generation == "sm75":
        a, sa = prepare_a16(x)  # FP16 tensor cores with exact power-of-two block scales
    else:
        a, sa = x, None
    slot = "small" if m <= 128 else "large"
    device = _device_names.get(x.device.index)
    if device is None:
        device = _device_names[x.device.index] = torch.cuda.get_device_name(x.device)
    key = f"{device}|{slot}|{n}|{k}|{wmode}|{str(x.dtype)[6:]}|{int(bias is not None)}"
    config = _load_cache().get(key)
    if config is None:
        candidates = _CANDIDATES[generation][slot]
        tune = _TUNING and (slot == "small" or m >= 1024) and not torch.compiler.is_compiling() and not torch.cuda.is_current_stream_capturing()
        config = _tune(key, a, sa, weight, scale, wmode, bias, out, candidates) if tune else candidates[0]
    _launch(a, sa, weight, scale, wmode, bias, out, tuple(config))
    return out


from shared.kernels.triton_compilation_log import install_triton_compilation_logger, log_compilation_tuning  # noqa: E402
install_triton_compilation_logger()
_tune = log_compilation_tuning(_tune)
