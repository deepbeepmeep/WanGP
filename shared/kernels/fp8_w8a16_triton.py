# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Weight-only FP8 matmul (W8A16) for Quanto qfloat8 and scaled FP8 weights: each tile of FP8 weights is decoded in registers,
multiplied by its scale and rounded to the compute dtype exactly as the dequantization does, then multiplied with the 16-bit
activations on the tensor cores (fp32 accumulation). The weight matrix is never materialized: same values as dequantization + matmul
up to the order of the sums.

GPUs before sm89 and AMD GPUs cannot convert float8 in Triton: their weights are read as bytes and decoded with integer operations
(same values). Before sm80, BF16 and FP32 products use IEEE FP32 arithmetic, preserving BF16's range without requiring BF16 or TF32
tensor cores; AMD GPUs multiply with IEEE precision as well (Triton's AMD backend has no TF32).

Tile sizes, as the INT8 Triton tuner does: the calls are grouped in slots (a range of tokens: tiny <= 16, small <= 128, large, for a
layer shape N x K and kernel variant). A slot starts with a static configuration and is tuned once, outside CUDA graph recording and
compilation, on representative rows of a real call (the call's own input and output: no extra activation-sized buffer; a large slot
waits for a call of at least 4096 tokens and uses 4096 of them), keeping a candidate only when it is at least 2% faster; the results go
to a cache file shared by the sessions. A new resolution reuses its slots: no tuning, and no compilation (the kernel is not
specialized on the number of tokens).
WAN2GP_FP8_TRITON_AUTOTUNE=0 keeps the static configurations; WAN2GP_FP8_TRITON_AUTOTUNE_MAX_SLOTS bounds the slots tuned per process."""
import json
import os
import threading
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

_CACHE_VERSION = 6  # bump when the kernel, the slots or the benchmark change: tuned configurations of other versions are ignored
_CACHE_PATH = Path(os.environ.get("WAN2GP_FP8_TRITON_CACHE", Path.home() / ".triton" / "autotune" / "wan2gp_fp8_w8a16_cache.json")).expanduser()
_DECODE = {"native": 0, torch.float8_e4m3fn: 1, torch.float8_e5m2: 2}
# (BLOCK_M, BLOCK_N, BLOCK_K, warps, stages); the first one of each slot is its static configuration (the most frequent winner on
# an RTX 5090 for LTX-2, Z-Image and synthetic shapes)
_CANDIDATES = {
    "tiny": [(16, 128, 128, 4, 4), (16, 64, 128, 4, 4), (16, 128, 64, 4, 4), (32, 64, 128, 4, 4), (16, 256, 128, 8, 3)],
    "small": [(32, 128, 64, 4, 4), (64, 128, 64, 4, 4), (64, 64, 64, 4, 4), (64, 128, 128, 4, 3), (128, 64, 64, 4, 4)],
    "large": [(128, 128, 32, 4, 3), (128, 64, 32, 4, 5), (128, 128, 32, 4, 4), (128, 128, 64, 8, 3), (128, 128, 64, 4, 4), (128, 64, 64, 4, 4),
              (256, 128, 64, 8, 3), (128, 256, 64, 8, 3)],
}
_LARGE_MIN_TOKENS = _LARGE_TUNE_TOKENS = 4096  # shorter calls tie between configurations: they would not pick the one of long calls
_TUNING = os.environ.get("WAN2GP_FP8_TRITON_AUTOTUNE", "1") != "0"
_MAX_SLOTS = int(os.environ.get("WAN2GP_FP8_TRITON_AUTOTUNE_MAX_SLOTS", "32"))
_MIN_SPEEDUP = 1.02
_lock = threading.Lock()
_tuned = None
_tuned_slots = 0
_device_names = {}


@triton.jit
def _decode_e4m3(bits):
    """float8_e4m3fn bytes as fp32 (GPUs without float8 conversions)."""
    b = bits.to(tl.int32)
    e = (b >> 3) & 0xF
    m = b & 0x7
    value = tl.where(e == 0, m.to(tl.float32) * 0.001953125, (((e + 120) << 23) | (m << 20)).to(tl.float32, bitcast=True))
    value = tl.where((b & 0x7F) == 0x7F, float("nan"), value)
    return tl.where((b & 0x80) != 0, -value, value)


@triton.jit
def _decode_e5m2(bits):
    """float8_e5m2 bytes as fp32: they are the upper byte of a float16."""
    return (bits.to(tl.uint16) << 8).to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _decode_bf16(bits):
    return (bits.to(tl.uint32) << 16).to(tl.float32, bitcast=True)


@triton.jit
def _encode_bf16(value):
    # Round to nearest even without BF16 instructions (unavailable before sm80).
    bits = value.to(tl.uint32, bitcast=True)
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16
    return tl.where(value != value, 0x7FC0, rounded).to(tl.uint16)


@triton.jit(do_not_specialize=["M"])  # one binary per configuration whatever the number of tokens
def _w8a16_kernel(x, w, scale, bias, out, M, N, K, sxm, sxk, swn, swk, som, son,
                  SCALE_PER_N: tl.constexpr, HAS_BIAS: tl.constexpr, DECODE: tl.constexpr, EVEN_K: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr, BF16_SIMT: tl.constexpr, IEEE: tl.constexpr):
    pid = tl.program_id(0)
    tiles_m, tiles_n = tl.cdiv(M, BM), tl.cdiv(N, BN)
    group = GROUP_M * tiles_n
    first_m = (pid // group) * GROUP_M
    size_m = tl.minimum(tiles_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group) % size_m
    pid_n = (pid % group) // size_m
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    dtype = out.dtype.element_ty
    if SCALE_PER_N:
        s = tl.load(scale + rn, mask=rn < N, other=0)[None, :]
    else:
        s = tl.load(scale)
    if BF16_SIMT:
        s = _decode_bf16(s)
    else:
        s = s.to(tl.float32)
    a_ptr = x + rm[:, None] * sxm + rk[None, :] * sxk
    b_ptr = w + rn[None, :] * swn + rk[:, None] * swk
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        k_mask = rk < K - k * BK
        if EVEN_K:
            a = tl.load(a_ptr, mask=rm[:, None] < M, other=0.0)
        else:
            a = tl.load(a_ptr, mask=(rm[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(b_ptr, mask=(rn[None, :] < N) & k_mask[:, None])  # float8 cannot take a fill value: masked rows are zeroed below
        if DECODE == 1:
            b = _decode_e4m3(b)
        elif DECODE == 2:
            b = _decode_e5m2(b)
        else:
            b = b.to(tl.float32)
        if not EVEN_K:  # rows beyond K would multiply undefined values into the sums
            b = tl.where(k_mask[:, None], b, 0.0)
        # Round dequantized weights to the compute dtype before multiplying, as Quanto does.
        if BF16_SIMT:
            acc = tl.dot(_decode_bf16(a), _decode_bf16(_encode_bf16(b * s)), acc, input_precision="ieee")
        else:
            acc = tl.dot(a, (b * s).to(dtype), acc, input_precision="ieee" if IEEE else "tf32")
        a_ptr += BK * sxk
        b_ptr += BK * swk
    if HAS_BIAS:
        bias_value = tl.load(bias + rn, mask=rn < N, other=0)[None, :]
        if bias.dtype.element_ty == tl.uint16:
            bias_value = _decode_bf16(bias_value)
        else:
            bias_value = bias_value.to(tl.float32)
    if BF16_SIMT:
        result = _encode_bf16(acc)
        if HAS_BIAS:
            result = _encode_bf16(_decode_bf16(result) + bias_value)
    else:
        result = acc.to(dtype)
        if HAS_BIAS:  # quanto adds the bias to the rounded output
            result = (result.to(tl.float32) + bias_value).to(dtype)
    tl.store(out + rm[:, None] * som + rn[None, :] * son, result, mask=(rm[:, None] < M) & (rn[None, :] < N))


def _decode_mode(weight):
    # ROCm reports the gfx version as capability and Triton does not convert float8 on every AMD GPU: bytes decoded there
    native = os.environ.get("WAN2GP_FP8_TRITON_DECODE", "") != "bits" and torch.version.hip is None and torch.cuda.get_device_capability(weight.device) >= (8, 9)
    return _DECODE["native"] if native else _DECODE[weight.dtype]


def _launch(x, weight, scale, bias, out, decode, config):
    bm, bn, bk, warps, stages = config
    m, k = x.shape
    n = weight.shape[0]
    w = weight if decode == 0 else weight.view(torch.uint8)
    pre_sm80 = torch.cuda.get_device_capability(x.device) < (8, 0)
    bf16_simt = out.dtype == torch.bfloat16 and pre_sm80
    if bf16_simt:
        x, scale, out = x.view(torch.uint16), scale.view(torch.uint16), out.view(torch.uint16)
    if pre_sm80 and bias is not None and bias.dtype == torch.bfloat16:
        bias = bias.view(torch.uint16)
    _w8a16_kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
        x, w, scale, bias, out, m, n, k, x.stride(0), x.stride(1), w.stride(0), w.stride(1), out.stride(0), out.stride(1),
        scale.numel() > 1, bias is not None, decode, k % bk == 0, bm, bn, bk, 8,
        BF16_SIMT=bf16_simt, IEEE=pre_sm80 or torch.version.hip is not None, num_warps=warps, num_stages=stages)  # AMD's tl.dot has no tf32


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
        print(f"[FP8] Triton tuning results not saved ({exc}).")


def _can_tune(slot, m):
    return (_TUNING and _tuned_slots < _MAX_SLOTS and (slot != "large" or m >= _LARGE_MIN_TOKENS)
            and not torch.compiler.is_compiling() and not torch.cuda.is_current_stream_capturing())


def _time(x, weight, scale, bias, out, decode, config, reps):
    _launch(x, weight, scale, bias, out, decode, config)  # compilation and warm-up
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        _launch(x, weight, scale, bias, out, decode, config)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / reps


def _tune(key, slot, x, weight, scale, bias, out, decode, candidates):
    """The slot's configuration: candidates checked on a small slice against the static one, timed on representative rows of the call
    (its own input and output), kept when at least 2% faster than the static one."""
    global _tuned_slots
    started = time.time()
    _tuned_slots += 1
    rows = min(x.shape[0], _LARGE_TUNE_TOKENS)
    bench_x, bench_out = x[:rows], out[:rows]
    probe_rows, probe_cols = min(rows, 256), min(weight.shape[0], 256)
    probe = (x[:probe_rows], weight[:probe_cols], scale if scale.numel() == 1 else scale.reshape(-1)[:probe_cols], None if bias is None else bias[:probe_cols])
    reference = out.new_empty((probe_rows, probe_cols))
    _launch(*probe, reference, decode, candidates[0])
    trial = torch.empty_like(reference)
    valid = []
    for config in candidates:
        try:
            _launch(*probe, trial, decode, config)  # compilation and validation
            if torch.allclose(trial.float(), reference.float(), rtol=2e-2, atol=2e-2 * reference.float().abs().max().item()):
                _launch(bench_x, weight, scale, bias, bench_out, decode, config)
                valid.append(config)
        except (triton.runtime.errors.OutOfResources, RuntimeError):  # a configuration this GPU cannot run
            continue
    timings = {}
    for _ in range(2):  # two alternated rounds, the best time of each: the first configurations timed are not penalized by the GPU warming up
        for config in valid:
            elapsed = _time(bench_x, weight, scale, bias, bench_out, decode, config, 3 if slot == "large" else 10)
            timings[config] = min(timings.get(config, elapsed), elapsed)
    baseline = timings.get(candidates[0])
    best = min(timings, key=timings.get)
    if baseline is not None and timings[best] * _MIN_SPEEDUP > baseline:
        best = candidates[0]
    with _lock:
        _load_cache()[key] = list(best)
        _save_cache()
    gain = f" ({baseline / timings[best]:.2f}x the static configuration)" if baseline is not None and best != candidates[0] else ""
    # The shared compilation logger reports the selected configuration once.
    return best


def linear(x, weight, scale, bias, out):
    """out (M, N) = x (M, K) @ (weight (N, K) float8 * scale (N or 1 values))^T + bias, in out's dtype (that of x and scale)."""
    m, k = x.shape
    n = weight.shape[0]
    decode = _decode_mode(weight)
    slot = "tiny" if m <= 16 else "small" if m <= 128 else "large"
    device = _device_names.get(x.device.index)
    if device is None:
        device = _device_names[x.device.index] = torch.cuda.get_device_name(x.device)
    key = f"{device}|{slot}|{n}|{k}|{decode}|{str(weight.dtype)[6:]}|{str(out.dtype)[6:]}|{int(scale.numel() > 1)}|{int(bias is not None)}"
    config = _load_cache().get(key)
    if config is None:
        candidates = _CANDIDATES[slot]
        if slot == "tiny" and x.dtype in (torch.bfloat16, torch.float32) and torch.cuda.get_device_capability(x.device) < (8, 0):
            candidates = candidates[1:]  # the first tile's FP32 operands need 72 KiB; Turing has 64 KiB
        config = _tune(key, slot, x, weight, scale, bias, out, decode, candidates) if _can_tune(slot, m) else candidates[0]
    _launch(x, weight, scale, bias, out, decode, tuple(config))
    return out


from shared.kernels.triton_compilation_log import install_triton_compilation_logger, log_compilation_tuning
install_triton_compilation_logger()
_tune = log_compilation_tuning(_tune)
