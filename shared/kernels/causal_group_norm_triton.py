# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Fused GroupNorm + SiLU + replicate padding of the causal 3D VAEs with GroupNorm (HunyuanVideo, Kandinsky 5 and 6: the input of each
3x3x3 convolution): one read for the statistics,
one read and one write of the padded convolution input, instead of the GroupNorm, SiLU and padding passes. FP32 arithmetic in registers;
the norm output is rounded to the activation dtype before the SiLU, as in the PyTorch path, and the per-channel affine follows PyTorch's
fused parameters (a * x + b, from the group mean and rstd rounded to the activation dtype, as PyTorch stores them). Only the group
statistics are summed in another order: rounding-level differences.

One binary per kernel and activation dtype: the sizes are not specialized, so a new resolution, video length or tile size compiles nothing.
Launch configurations start from a static choice and are tuned once per kernel, dtype and GPU on scratch outputs of a large call."""
import functools

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from shared.kernels.triton_tuner import LaunchTuner

_TUNER = LaunchTuner("wan2gp_causal_vae_kernels", 4, "Causal 3D VAE")
_MAX_CHUNKS = 256  # partial statistics per group
_CHUNK_ELEMENTS = 1 << 14  # at least this many elements per statistics program
_MAX_GROUP_CHANNELS = 64
_TUNE_ELEMENTS = 1 << 24  # calls of at least this many elements tune the configurations
_TRITON_DTYPES = {torch.float16: tl.float16, torch.bfloat16: tl.bfloat16, torch.float32: tl.float32}


@functools.cache
def supported(device):
    """CUDA GPUs with compute capability 8.0+; other devices use the PyTorch path."""
    return device.type == "cuda" and torch.version.hip is None and torch.cuda.get_device_capability(device) >= (8, 0)


@triton.jit(do_not_specialize=["ROW", "CHUNK", "NCHUNK"])
def _group_stats_kernel(X, PART, ROW, CHUNK, NCHUNK, BLOCK: tl.constexpr):
    """Count, mean and sum of squared deviations of one chunk of a group row (the group's channels are contiguous)."""
    row = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    start = chunk.to(tl.int64) * CHUNK
    end = tl.minimum(start + CHUNK, ROW)
    cols = tl.arange(0, BLOCK)
    count = 0.0
    mean = 0.0
    m2 = 0.0
    for offset in range(start, end, BLOCK):
        idx = offset + cols
        valid = idx < end
        x = tl.load(X + row * ROW + idx, valid, 0.0).to(tl.float32)
        n = tl.sum(valid.to(tl.float32), axis=0)
        tile_mean = tl.sum(x, axis=0) / n
        deviation = tl.where(valid, x - tile_mean, 0.0)
        total = count + n
        delta = tile_mean - mean
        mean += delta * (n / total)
        m2 += tl.sum(deviation * deviation, axis=0) + delta * delta * (count * n / total)
        count = total
    out = PART + (row * NCHUNK + chunk) * 3
    tl.store(out, count)
    tl.store(out + 1, mean)
    tl.store(out + 2, m2)


@triton.jit
def _merged_partials(part, valid):
    n = tl.load(part, valid, 0.0)
    m = tl.load(part + 1, valid, 0.0)
    q = tl.load(part + 2, valid, 0.0)
    total = tl.sum(n, axis=0)
    mean = tl.sum(n * m, axis=0) / total
    deviation = m - mean
    return total, mean, tl.sum(q + n * deviation * deviation, axis=0)


@triton.jit(do_not_specialize=["NCHUNK", "GROUP_CHANNELS", "C", "G", "BATCH"])
def _group_affine_kernel(PART, RUN, WEIGHT, BIAS, A, B, NCHUNK, GROUP_CHANNELS, C, G, BATCH, EPS, STATS_DTYPE: tl.constexpr,
                         RUN_MODE: tl.constexpr, SHARED: tl.constexpr, MAX_CHUNKS: tl.constexpr, MAX_CG: tl.constexpr):
    """PyTorch's fused per-channel parameters a = rstd * weight, b = bias - a * mean of a group, from (RUN_MODE 0) the partial statistics of
    x merged (Chan), (1) those merged first into the running statistics of the previous chunks of a streamed decode (count, mean, M2 in
    FP64, updated), or (2) the given statistics RUN (frozen). SHARED: one program per group for all the batch items (several parts of one
    video streamed together), whose statistics are merged."""
    row = tl.program_id(0)
    run = RUN + row.to(tl.int64) * 3
    if RUN_MODE == 2:
        count = tl.load(run)
        mean = tl.load(run + 1).to(tl.float32)
        var = tl.maximum((tl.load(run + 2) / count).to(tl.float32), 0.0)
    else:
        chunks = tl.arange(0, MAX_CHUNKS)
        valid = chunks < NCHUNK
        if SHARED:
            total = 0.0
            mean = 0.0
            m2 = 0.0
            for item in range(BATCH):
                item_total, item_mean, item_m2 = _merged_partials(PART + (item * G + row).to(tl.int64) * NCHUNK * 3 + chunks * 3, valid)
                merged_total = total + item_total
                delta = item_mean - mean
                mean += delta * (item_total / merged_total)
                m2 += item_m2 + delta * delta * (total * item_total / merged_total)
                total = merged_total
        else:
            total, mean, m2 = _merged_partials(PART + row.to(tl.int64) * NCHUNK * 3 + chunks * 3, valid)
        if RUN_MODE == 1:
            count, run_mean = tl.load(run), tl.load(run + 1)
            merged = count + total.to(tl.float64)
            delta = mean.to(tl.float64) - run_mean
            run_mean += delta * (total.to(tl.float64) / merged)
            run_m2 = tl.load(run + 2) + m2.to(tl.float64) + delta * delta * (count * total.to(tl.float64) / merged)
            tl.store(run, merged)
            tl.store(run + 1, run_mean)
            tl.store(run + 2, run_m2)
            mean = run_mean.to(tl.float32)
            var = tl.maximum((run_m2 / merged).to(tl.float32), 0.0)
        else:
            var = tl.maximum(m2 / total, 0.0)
    rstd = libdevice.rsqrt(var + EPS).to(STATS_DTYPE).to(tl.float32)
    mean = mean.to(STATS_DTYPE).to(tl.float32)
    lanes = tl.arange(0, MAX_CG)
    lane_valid = lanes < GROUP_CHANNELS
    channels = (row % G) * GROUP_CHANNELS + lanes
    a = rstd * tl.load(WEIGHT + channels, lane_valid, 1.0).to(tl.float32)
    b = -a * mean + tl.load(BIAS + channels, lane_valid, 0.0).to(tl.float32)
    if SHARED:
        for item in range(BATCH):
            tl.store(A + item * C + channels, a, lane_valid)
            tl.store(B + item * C + channels, b, lane_valid)
    else:
        out = (row // G) * C + channels
        tl.store(A + out, a, lane_valid)
        tl.store(B + out, b, lane_valid)


@triton.jit(do_not_specialize=["T", "H", "W", "TP", "HP", "WP", "PT", "PH", "PW", "OUT_TP", "OUT_T0"])
def _norm_silu_pad_kernel(X, OUT, A, B, T, H, W, TP, HP, WP, PT, PH, PW, OUT_TP, OUT_T0, BLOCK: tl.constexpr, SILU: tl.constexpr):
    """One block of a padded output frame: the replicated border reads the nearest input position. The TP padded frames are written from
    frame OUT_T0 of an output of OUT_TP frames."""
    plane = tl.program_id(0).to(tl.int64)  # (batch * channels + channel) * TP + padded frame
    volume = plane // TP
    t = tl.minimum(tl.maximum(plane % TP - PT, 0), T - 1)
    idx = tl.program_id(1).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = idx < HP * WP
    h = tl.minimum(tl.maximum(idx // WP - PH, 0), H - 1)
    w = tl.minimum(tl.maximum(idx % WP - PW, 0), W - 1)
    x = tl.load(X + (volume * T + t) * H * W + h * W + w, valid, 0.0).to(tl.float32)
    y = (x * tl.load(A + volume) + tl.load(B + volume)).to(OUT.dtype.element_ty)
    if SILU:
        y = y.to(tl.float32)
        y = tl.div_rn(y, 1.0 + libdevice.exp(-y)).to(OUT.dtype.element_ty)  # IEEE division and expf, as PyTorch
    tl.store(OUT + (volume * OUT_TP + OUT_T0 + plane % TP) * HP * WP + idx, y, valid)


def group_norm_silu_pad(x, norm, pad=(0, 0, 0, 0, 0, 0), silu=True, running=None, frozen=False, shared=False, out=None, out_frame=0, apply=True):
    """silu(norm(x)) of x (batch, channels, frames, height, width), replicate-padded by pad = (left, right, top, bottom, front, back) as
    F.pad: a new tensor of x's dtype, or written from frame out_frame of out. running: FP64 (batch * groups, 3) count, mean and M2 of the
    previous chunks of a streamed decode, merged with x's statistics (and updated) before normalizing, or used as given when frozen.
    shared: the batch items are parts of one video, normalized with statistics merged over all of them (running: (groups, 3)).
    apply=False only merges x's statistics into running."""
    x = x.contiguous()
    batch, channels, frames, height, width = x.shape
    groups = norm.num_groups
    group_channels = channels // groups
    row = group_channels * frames * height * width
    nchunk = min(_MAX_CHUNKS, triton.cdiv(row, _CHUNK_ELEMENTS))
    chunk = triton.cdiv(row, nchunk)
    tunable = x.numel() >= _TUNE_ELEMENTS
    part = torch.empty(batch * groups * nchunk * 3, dtype=torch.float32, device=x.device)

    def stats(rows, partials, config):
        block, warps = config
        _group_stats_kernel[(rows, nchunk)](x, partials, row, chunk, nchunk, block, num_warps=warps)

    def stats_scratch():
        partials = torch.empty_like(part)
        return lambda config: stats(min(batch * groups, 8), partials, config)

    if not frozen:
        stats(batch * groups, part, _TUNER.config("group_stats", x.dtype, x.device, ((2048, 8), (1024, 4), (4096, 8), (2048, 4)), stats_scratch, tunable))
    a = torch.empty(batch * channels, dtype=torch.float32, device=x.device)
    b = torch.empty_like(a)
    _group_affine_kernel[(groups if shared else batch * groups,)](part, part if running is None else running, norm.weight, norm.bias, a, b, nchunk,
                                                                  group_channels, channels, groups, batch, norm.eps, _TRITON_DTYPES[x.dtype],
                                                                  0 if running is None else 2 if frozen else 1, shared, _MAX_CHUNKS,
                                                                  _MAX_GROUP_CHANNELS, num_warps=4)
    if not apply:
        return None
    left, right, top, bottom, front, back = pad
    tp, hp, wp = frames + front + back, height + top + bottom, width + left + right
    if out is None:
        out = torch.empty((batch, channels, tp, hp, wp), dtype=x.dtype, device=x.device)

    def apply(target, planes, target_frames, target_start, config):
        block, warps = config
        _norm_silu_pad_kernel[(planes, triton.cdiv(hp * wp, block))](x, target, a, b, frames, height, width, tp, hp, wp, front, top, left,
                                                                     target_frames, target_start, block, silu, num_warps=warps)  # a * x + b: FMA, as nvcc

    def apply_scratch():
        planes = min(batch * channels * tp, 64)
        target = torch.empty(planes * hp * wp, dtype=x.dtype, device=x.device)
        return lambda config: apply(target, planes, tp, 0, config)

    apply(out, batch * channels * tp, out.shape[2], out_frame,
          _TUNER.config("norm_silu_pad", x.dtype, x.device, ((1024, 4), (2048, 8), (512, 4), (4096, 8)), apply_scratch, tunable))
    return out


@functools.cache
def fp32_scores_supported():
    """cuBLAS products of 16-bit inputs with FP32 outputs (torch.mm out_dtype, PyTorch 2.8+), needed by attention()."""
    return any("out_dtype" in str(schema) for schema in torch._C._jit_get_schemas_for_operator("aten::mm"))


@triton.jit(do_not_specialize=["N", "STRIDE"])
def _exp_rows_kernel(S, P, SUMS, N, STRIDE, SCALE, BLOCK: tl.constexpr):
    """One row of FP32 attention scores: the maximum and the sum of exp((s - max) * scale) (FP32, online), then exp((s - max) * scale)
    rounded to P's dtype, written over the first half of the row's bytes (P views the scores' storage, twice the row stride)."""
    row = tl.program_id(0).to(tl.int64)
    scores = S + row * STRIDE
    cols = tl.arange(0, BLOCK)
    high = tl.full([BLOCK], -3.0e38, tl.float32)
    total = tl.zeros([BLOCK], tl.float32)
    for start in range(0, N, BLOCK):
        idx = start + cols
        valid = idx < N
        s = tl.load(scores + idx, valid, -3.0e38) * SCALE
        new_high = tl.maximum(high, s)
        total = total * tl.exp(high - new_high) + tl.where(valid, tl.exp(s - new_high), 0.0)
        high = new_high
    row_max = tl.max(high, axis=0)
    tl.store(SUMS + row, tl.sum(total * tl.exp(high - row_max), axis=0))
    probs = P + row * (2 * STRIDE)
    for start in range(0, N, BLOCK):
        idx = start + cols
        valid = idx < N
        s = tl.load(scores + idx, valid, 0.0) * SCALE
        tl.debug_barrier()  # the values written overwrite scores of this block or earlier ones: every thread has read them
        tl.store(probs + idx, tl.exp(s - row_max).to(P.dtype.element_ty), valid)


def attention(q, k, v, rows=256):
    """softmax(q k^T / sqrt(d)) v of one head, q (batch, queries, d) and k / v (batch, keys, d) in a 16-bit dtype: the scores of `rows`
    queries at a time in FP32 (cuBLAS), turned in place into exp(s - max) in q's dtype and the FP32 row sums, then (P v) in FP32 divided
    by the sums. The rounding points of PyTorch's memory-efficient SDPA kernel (FP32 scores and softmax, probabilities rounded to the input
    dtype for the second product, FP32 accumulation and normalization) at cuBLAS speed for the 512-wide head, which that kernel handles
    slowly. Extra memory: one block of FP32 scores."""
    batch, queries, dim = q.shape
    keys = k.shape[1]
    rows = triton.cdiv(queries, triton.cdiv(queries, rows))  # equal blocks
    scores = torch.empty(rows, keys, dtype=torch.float32, device=q.device)
    probs = scores.view(q.dtype)
    sums = torch.empty(rows, dtype=torch.float32, device=q.device)
    out = torch.empty_like(q)

    def exp_rows(target, count, config):
        block, warps = config
        _exp_rows_kernel[(count,)](target, target.view(q.dtype), sums, keys, keys, dim ** -0.5, block, num_warps=warps, num_stages=1)

    def exp_scratch():
        target = torch.zeros(min(rows, 64), keys, dtype=torch.float32, device=q.device)
        return lambda config: exp_rows(target, target.shape[0], config)

    config = _TUNER.config("exp_rows", q.dtype, q.device, ((4096, 8), (2048, 4), (8192, 8), (4096, 4)), exp_scratch, rows * keys >= _TUNE_ELEMENTS)
    for item in range(batch):
        for start in range(0, queries, rows):
            count = min(rows, queries - start)
            torch.mm(q[item, start:start + count], k[item].T, out_dtype=torch.float32, out=scores[:count])
            exp_rows(scores, count, config)
            out[item, start:start + count] = torch.mm(probs[:count, :keys], v[item], out_dtype=torch.float32).div_(sums[:count, None])
    return out


from shared.kernels.triton_compilation_log import install_triton_compilation_logger  # noqa: E402

install_triton_compilation_logger()
