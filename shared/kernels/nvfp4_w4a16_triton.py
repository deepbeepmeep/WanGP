# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Weight-only NVFP4 matmul (W4A16) for GPUs without FP4 tensor cores (RTX 30xx/40xx, Hopper): each tile of packed e2m1 weights is
decoded in registers, multiplied by its e4m3 block scale (exact in FP32: at most four significant bits) and by the per-tensor factor,
rounded to the compute dtype exactly as NVFP4WeightTensor.dequantize() does, then multiplied with the 16-bit activations on the tensor
cores (FP32 accumulation). The weight matrix is never materialized: same values as dequantization + matmul up to the order of the sums,
with no activation quantization (more precise than the FP4 tensor-core path the student was trained under).

Block scales are read in the stored 128x4 swizzled layout (cuBLAS / cutlass order shared by both WanGP NVFP4 layouts), so no copy of the
weight or its scales is kept. The e4m3 scales are decoded with integer operations, so the kernel runs from sm80 (BF16 tensor cores).

Launch configurations follow fp8_w8a16_triton: slots by token count, a static configuration per slot tuned once on representative rows
of a real call, winners in a versioned cache file; the kernel is not specialized on the number of tokens."""
import json
import os
import threading
import time
from pathlib import Path

import torch
import triton
import triton.language as tl

from shared.kernels.fp8_w8a16_triton import _decode_e4m3

_CACHE_VERSION = 2
_CACHE_PATH = Path(os.environ.get("WAN2GP_NVFP4_TRITON_CACHE", Path.home() / ".triton" / "autotune" / "wan2gp_nvfp4_w4a16_cache.json")).expanduser()
# (BLOCK_M, BLOCK_N, BLOCK_K, warps, stages); the first one of each slot is its static configuration
_CANDIDATES = {
    "tiny": [(16, 128, 64, 4, 3), (16, 64, 128, 4, 3), (32, 128, 64, 4, 3), (16, 256, 64, 8, 3)],
    "small": [(64, 128, 64, 4, 3), (32, 128, 64, 4, 3), (64, 64, 64, 4, 4), (64, 128, 128, 4, 3), (128, 64, 64, 4, 4)],
    "large": [(128, 128, 64, 4, 3), (128, 128, 64, 8, 3), (128, 64, 64, 4, 4), (128, 128, 128, 8, 3), (64, 128, 64, 4, 4), (256, 128, 64, 8, 3),
              (128, 256, 64, 8, 3)],
}
_LARGE_MIN_TOKENS = _LARGE_TUNE_TOKENS = 4096
_TUNING = os.environ.get("WAN2GP_NVFP4_TRITON_AUTOTUNE", "1") != "0"
_MAX_SLOTS = int(os.environ.get("WAN2GP_NVFP4_TRITON_AUTOTUNE_MAX_SLOTS", "32"))
_MIN_SPEEDUP = 1.02
_lock = threading.Lock()
_tuned = None
_tuned_slots = 0
_device_names = {}


@triton.jit
def _decode_e2m1(code):
    """4-bit e2m1 codes (sign in bit 3) as fp32: 0, .5, 1, 1.5, 2, 3, 4, 6."""
    magnitude = (code & 7).to(tl.float32)
    value = tl.where(magnitude < 4, magnitude * 0.5, tl.where(magnitude < 6, magnitude - 2.0, magnitude * 2.0 - 8.0))
    return tl.where((code & 8) != 0, -value, value)


@triton.jit(do_not_specialize=["M"])  # one binary per configuration whatever the number of tokens
def _w4a16_kernel(x, w, scale, bias, out, M, N, K, k_tiles, sxm, sxk, swn, som, son, factor,
                  HAS_BIAS: tl.constexpr, HIGH_FIRST: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
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
    rkb = tl.arange(0, BK // 2)  # packed bytes of a k tile
    rg = tl.arange(0, BK // 16)  # scale groups of a k tile
    dtype = out.dtype.element_ty
    n_mask = rn < N
    # 128x4 swizzled scale layout: byte index of (row, group) = (((row // 128) * k_tiles + group // 4) * 32 + row % 32) * 4 + (row // 32) % 4) * 4 + group % 4
    row_term = (((rn // 128) * k_tiles) * 32 + rn % 32) * 16 + ((rn // 32) % 4) * 4
    a_ptr = x + rm[:, None] * sxm + rk[None, :] * sxk
    w_ptr = w + rn[:, None] * swn + rkb[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptr, mask=rm[:, None] < M, other=0.0)
        packed = tl.load(w_ptr, mask=n_mask[:, None], other=0)
        low, high = packed & 15, packed >> 4
        codes = tl.interleave(high, low) if HIGH_FIRST else tl.interleave(low, high)  # [BN, BK]
        values = _decode_e2m1(codes)
        groups = k * (BK // 16) + rg
        scale_index = row_term[:, None] + ((groups // 4) * 512 + groups % 4)[None, :]
        s = _decode_e4m3(tl.load(scale + scale_index, mask=n_mask[:, None], other=0))  # [BN, BK // 16]
        scaled = tl.reshape(values, (BN, BK // 16, 16)) * tl.reshape(s, (BN, BK // 16, 1))
        # the dequantization rounds (value * block scale) * factor once from fp32 to the compute dtype
        b = (tl.reshape(scaled, (BN, BK)) * factor).to(dtype)
        acc = tl.dot(a, tl.trans(b), acc)
        a_ptr += BK * sxk
        w_ptr += BK // 2
    if HAS_BIAS:
        acc += tl.load(bias + rn, mask=n_mask, other=0.0).to(tl.float32)[None, :]
    tl.store(out + rm[:, None] * som + rn[None, :] * son, acc.to(dtype), mask=(rm[:, None] < M) & n_mask[None, :])


def supported(weight, x):
    """Whether this kernel computes x @ weight.T: 2-D 16-bit CUDA activations, K a multiple of 64 (scale tiles), N a multiple of 128."""
    n, half_k = weight._data.shape
    return (x.is_cuda and x.dtype in (torch.bfloat16, torch.float16) and half_k % 32 == 0 and n % 128 == 0
            and weight._scale.shape[0] >= n and weight._scale.shape[1] * 16 == 2 * half_k and torch.cuda.get_device_capability(x.device) >= (8, 0))


def _factor(weight):
    from shared.qtypes.nvfp4 import _NVFP4_LAYOUT_TENSORCORE
    alpha = weight._alpha.to(torch.float32)
    if weight._layout == _NVFP4_LAYOUT_TENSORCORE:
        return alpha, True
    return alpha * weight._input_global_scale.to(torch.float32), False


class RowSlice:
    """The rows [first, last) of an NVFP4 weight, first a multiple of 128 (one swizzled scale tile), for linear_rows."""

    def __init__(self, weight, first, last):
        self._data = weight._data[first:last]
        self._scale = weight._scale[first:(last + 127) // 128 * 128]
        self._alpha, self._input_global_scale, self._layout = weight._alpha, weight._input_global_scale, weight._layout


def _launch(x, weight, bias, out, config):
    bm, bn, bk, warps, stages = config
    m, k = x.shape
    n = weight._data.shape[0]
    factor, high_first = _factor(weight)
    scale = weight._scale.view(torch.uint8)
    _w4a16_kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](
        x, weight._data, scale, bias, out, m, n, k, k // 64, x.stride(0), x.stride(1), weight._data.stride(0), out.stride(0), out.stride(1),
        factor.item(), bias is not None, high_first, bm, bn, bk, 8, num_warps=warps, num_stages=stages, enable_fp_fusion=False)


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
        print(f"[NVFP4] Triton tuning results not saved ({exc}).")


def _can_tune(slot, m):
    return (_TUNING and _tuned_slots < _MAX_SLOTS and (slot != "large" or m >= _LARGE_MIN_TOKENS)
            and not torch.compiler.is_compiling() and not torch.cuda.is_current_stream_capturing())


def _time(x, weight, bias, out, config, reps):
    _launch(x, weight, bias, out, config)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(reps):
        _launch(x, weight, bias, out, config)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / reps


def _tune(key, slot, x, weight, bias, out, candidates):
    """The slot's configuration: candidates validated on the first rows against the static one, timed on representative rows of the
    call (its own input and output), kept when at least 2% faster than the static one."""
    global _tuned_slots
    _tuned_slots += 1
    rows = min(x.shape[0], _LARGE_TUNE_TOKENS)
    bench_x, bench_out = x[:rows], out[:rows]
    probe_rows = min(rows, 256)
    reference = out.new_empty((probe_rows, out.shape[1]))
    _launch(x[:probe_rows], weight, bias, reference, candidates[0])
    trial = torch.empty_like(reference)
    valid = []
    for config in candidates:
        try:
            _launch(x[:probe_rows], weight, bias, trial, config)
            if torch.allclose(trial.float(), reference.float(), rtol=2e-2, atol=2e-2 * reference.float().abs().max().item()):
                _launch(bench_x, weight, bias, bench_out, config)
                valid.append(config)
        except (triton.runtime.errors.OutOfResources, RuntimeError):
            continue
    timings = {}
    for _ in range(2):
        for config in valid:
            elapsed = _time(bench_x, weight, bias, bench_out, config, 3 if slot == "large" else 10)
            timings[config] = min(timings.get(config, elapsed), elapsed)
    baseline = timings.get(candidates[0])
    best = min(timings, key=timings.get)
    if baseline is not None and timings[best] * _MIN_SPEEDUP > baseline:
        best = candidates[0]
    with _lock:
        _load_cache()[key] = list(best)
        _save_cache()
    return best


def linear(x, weight, bias=None, out=None):
    """x (M, K) 16-bit @ NVFP4 weight (N, K)^T + bias, in x's dtype. x may be 2-D or more (rows flattened)."""
    x2d = x.reshape(-1, x.shape[-1])
    if not x2d.is_contiguous():
        x2d = x2d.contiguous()
    m, k = x2d.shape
    n = weight._data.shape[0]
    if out is None:
        out = torch.empty((m, n), dtype=x2d.dtype, device=x2d.device)
    if bias is not None and (bias.dtype != x2d.dtype or not bias.is_contiguous()):
        bias = bias.to(x2d.dtype).contiguous()
    slot = "tiny" if m <= 16 else "small" if m <= 128 else "large"
    device = _device_names.get(x2d.device.index)
    if device is None:
        device = _device_names[x2d.device.index] = torch.cuda.get_device_name(x2d.device)
    key = f"{device}|{slot}|{n}|{k}|{str(x2d.dtype)[6:]}|{int(bias is not None)}"
    config = _load_cache().get(key)
    if config is None:
        candidates = _CANDIDATES[slot]
        config = _tune(key, slot, x2d, weight, bias, out, candidates) if _can_tune(slot, m) else candidates[0]
    _launch(x2d, weight, bias, out, tuple(config))
    return out.reshape(*x.shape[:-1], n)


from shared.kernels.triton_compilation_log import install_triton_compilation_logger, log_compilation_tuning
install_triton_compilation_logger()
_tune = log_compilation_tuning(_tune)
