# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Decoding attention over a fixed-capacity BF16/FP16 cache (the MTP draft cache) whose length is read on the GPU, for CUDA graphs on
GPUs without FlashAttention (Turing): each query row attends the keys up to its own position only, so the slots past the written ones
are never read and no copy of the cache is made. The 16-bit values are decoded from their bits and every product, sum and exponential
is FP32 (IEEE), without BF16 instructions. Split over the keys (flash decoding); the partial results are merged with PyTorch."""
import triton
import triton.language as tl
from triton.language.extra import libdevice

_RUNTIME = ["COUNT", "SPLITS"]  # query rows of the call and key splits: one binary whatever their values


@triton.jit
def _load16(ptrs, mask, BF16: tl.constexpr):
    if BF16:
        return (tl.load(ptrs, mask=mask, other=0).to(tl.uint32) << 16).to(tl.float32, bitcast=True)
    return tl.load(ptrs, mask=mask, other=0.0).to(tl.float32)


@triton.jit(do_not_specialize=_RUNTIME, do_not_specialize_on_alignment=_RUNTIME)
def _static_decode(Q, K, V, SEQLENS, PM, PL, PACC, COUNT, SPLITS, scale, STRIDE_QC, STRIDE_KN, GROUPS: tl.constexpr, KV_HEADS: tl.constexpr,
                   HEAD_DIM: tl.constexpr, ROWS_PAD: tl.constexpr, BF16: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_N: tl.constexpr):
    kv, split, row_block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seqlen = tl.load(SEQLENS)
    total = seqlen + COUNT  # the keys written so far, this call's included
    span = tl.cdiv(tl.cdiv(total, SPLITS), BLOCK_N) * BLOCK_N
    start = split * span
    stop = tl.minimum(start + span, total)
    rows = row_block * BLOCK_R + tl.arange(0, BLOCK_R)  # row r: query i = r % COUNT of head kv * GROUPS + r // COUNT
    row_ok = rows < GROUPS * COUNT
    position = seqlen + rows % COUNT
    head = kv * GROUPS + rows // COUNT
    d = tl.arange(0, HEAD_DIM)
    q = _load16(Q + (rows % COUNT)[:, None] * STRIDE_QC + head[:, None] * HEAD_DIM + d[None, :], row_ok[:, None], BF16)
    m = tl.full((BLOCK_R,), float("-inf"), tl.float32)
    l = tl.zeros((BLOCK_R,), tl.float32)
    acc = tl.zeros((BLOCK_R, HEAD_DIM), tl.float32)
    for first in range(start, stop, BLOCK_N):
        n = first + tl.arange(0, BLOCK_N)
        key_ok = n < stop
        offsets = n[:, None] * STRIDE_KN + kv * HEAD_DIM + d[None, :]
        k = _load16(K + offsets, key_ok[:, None], BF16)
        s = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
        s = tl.where(key_ok[None, :] & (n[None, :] <= position[:, None]), s, float("-inf"))
        m_new = tl.maximum(m, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)  # a row with no visible key yet keeps zero weights
        alpha = libdevice.exp(m - m_safe)
        p = libdevice.exp(s - m_safe[:, None])
        v = _load16(V + offsets, key_ok[:, None], BF16)
        l = l * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
        m = m_new
    out = (split * KV_HEADS + kv) * ROWS_PAD + rows
    tl.store(PM + out, m)
    tl.store(PL + out, l)
    tl.store(PACC + out[:, None] * HEAD_DIM + d[None, :], acc)


@triton.jit(do_not_specialize=_RUNTIME, do_not_specialize_on_alignment=_RUNTIME)
def _static_decode_tensor(Q, K, V, SEQLENS, PM, PL, PACC, COUNT, SPLITS, scale, STRIDE_QC, STRIDE_KN, GROUPS: tl.constexpr, KV_HEADS: tl.constexpr,
                          HEAD_DIM: tl.constexpr, ROWS_PAD: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_N: tl.constexpr):
    """_static_decode on tensor cores, for a BF16 cache (sm_80 and newer, approximate kernels): BF16 scores and weights, as FlashAttention. About 6x faster at thousands of keys."""
    kv, split, row_block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    seqlen = tl.load(SEQLENS)
    total = seqlen + COUNT
    span = tl.cdiv(tl.cdiv(total, SPLITS), BLOCK_N) * BLOCK_N
    start = split * span
    stop = tl.minimum(start + span, total)
    if start < total:  # the splits past the written keys hold nothing that _static_merge reads
        rows = row_block * BLOCK_R + tl.arange(0, BLOCK_R)
        row_ok = rows < GROUPS * COUNT
        position = seqlen + rows % COUNT
        head = kv * GROUPS + rows // COUNT
        d = tl.arange(0, HEAD_DIM)
        q = tl.load(Q + (rows % COUNT)[:, None] * STRIDE_QC + head[:, None] * HEAD_DIM + d[None, :], mask=row_ok[:, None], other=0.0)
        m = tl.full((BLOCK_R,), float("-inf"), tl.float32)
        l = tl.zeros((BLOCK_R,), tl.float32)
        acc = tl.zeros((BLOCK_R, HEAD_DIM), tl.float32)
        for first in range(start, stop, BLOCK_N):
            n = first + tl.arange(0, BLOCK_N)
            key_ok = n < stop
            offsets = n[:, None] * STRIDE_KN + kv * HEAD_DIM + d[None, :]
            k = tl.load(K + offsets, mask=key_ok[:, None], other=0.0)
            s = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
            s = tl.where(key_ok[None, :] & (n[None, :] <= position[:, None]), s, float("-inf"))
            m_new = tl.maximum(m, tl.max(s, 1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.exp(m - m_safe)
            p = tl.exp(s - m_safe[:, None])
            v = tl.load(V + offsets, mask=key_ok[:, None], other=0.0)
            l = l * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), v, out_dtype=tl.float32)
            m = m_new
        out = (split * KV_HEADS + kv) * ROWS_PAD + rows
        tl.store(PM + out, m)
        tl.store(PL + out, l)
        tl.store(PACC + out[:, None] * HEAD_DIM + d[None, :], acc)


@triton.jit(do_not_specialize=_RUNTIME, do_not_specialize_on_alignment=_RUNTIME)
def _static_merge(SEQLENS, PM, PL, PACC, OUT, COUNT, SPLITS, STRIDE_QC, GROUPS: tl.constexpr, KV_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr, ROWS_PAD: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_N: tl.constexpr):
    """The splits of _static_decode_tensor that hold keys into BF16 attention outputs, in the layout of the query."""
    kv, row_block = tl.program_id(0), tl.program_id(1)
    total = tl.load(SEQLENS) + COUNT
    span = tl.cdiv(tl.cdiv(total, SPLITS), BLOCK_N) * BLOCK_N
    used = tl.cdiv(total, span)
    rows = row_block * BLOCK_R + tl.arange(0, BLOCK_R)
    row_ok = rows < GROUPS * COUNT
    d = tl.arange(0, HEAD_DIM)
    top = tl.full((BLOCK_R,), float("-inf"), tl.float32)
    for split in range(0, used):
        top = tl.maximum(top, tl.load(PM + (split * KV_HEADS + kv) * ROWS_PAD + rows, mask=row_ok, other=float("-inf")))
    l = tl.zeros((BLOCK_R,), tl.float32)
    acc = tl.zeros((BLOCK_R, HEAD_DIM), tl.float32)
    for split in range(0, used):
        at = (split * KV_HEADS + kv) * ROWS_PAD + rows
        weight = tl.exp(tl.load(PM + at, mask=row_ok, other=float("-inf")) - top)  # a split without keys for a row: exp(-inf) = 0
        l += weight * tl.load(PL + at, mask=row_ok, other=0.0)
        acc += weight[:, None] * tl.load(PACC + at[:, None] * HEAD_DIM + d[None, :], mask=row_ok[:, None], other=0.0)
    head = kv * GROUPS + rows // COUNT
    tl.store(OUT + (rows % COUNT)[:, None] * STRIDE_QC + head[:, None] * HEAD_DIM + d[None, :], (acc / l[:, None]).to(tl.bfloat16), mask=row_ok[:, None])


# Below the kernels, whose line numbers are part of Triton's cache keys.
import os

import torch
from shared.kernels import kernel_policy
from shared.kernels.triton_tuner import LaunchTuner

_TUNER = LaunchTuner("wan2gp_static_cache_attention", 1, "Static Cache Attention")
# (BLOCK_N, warps, stages); the first is the static configuration. FP32 tiles: q (16 rows), and one key and one value tile per stage
_CONFIGS = [(16, 4, 1), (32, 4, 1), (16, 8, 1), (32, 8, 1), (16, 4, 2), (64, 8, 1)]
_BLOCK_R = 16
_SPLITS_PER_SM = 2
_TENSOR_CONFIGS = [(64, 8, 1), (32, 4, 2), (32, 8, 2), (64, 4, 1), (16, 4, 3), (16, 8, 3)]  # of _static_decode_tensor
_TENSOR_SPLITS_PER_SM = 1  # at about 100 KB of shared memory, two blocks per SM at most; with the splits past the keys doing nothing
_TENSOR = "draft_tensor" not in os.environ.get("WGP_FN_DISABLE", "").split(",")  # until the switches are removed


def _fits(config, head_dim, device):
    # the FP32 query and key tiles, plus a little per key (Triton 3.2 and 3.6 for sm_75, head_dim 256: 33.9 / 51.3 / 86.1 KB for 16 / 32 /
    # 64 keys, whatever the stages): Turing's 64 KB take up to 32 keys
    shared = (_BLOCK_R + config[0]) * head_dim * 4 + 128 * config[0]
    return shared <= torch.cuda.get_device_properties(device).shared_memory_per_block_optin


def _fits_tensor(config, head_dim, device):
    # BF16 key and value tiles per stage, BF16 query and FP32 sum (Triton 3.6 for sm_120, head_dim 256, (64, 8, 2): 143 KB)
    shared = 4 * config[0] * head_dim * config[2] + 6 * _BLOCK_R * head_dim
    return shared <= torch.cuda.get_device_properties(device).shared_memory_per_block_optin


def usable(query, cache):
    return query.is_cuda and query.dtype in (torch.bfloat16, torch.float16) and cache.dtype == query.dtype and cache.is_contiguous() \
        and query.is_contiguous() and query.shape[-1] in (64, 128, 256)


def attention(query, key_cache, value_cache, seqlens, scale):
    """query (1, count, heads, head_dim) at positions seqlens + 0..count-1, whose keys and values are already written; key_cache and
    value_cache (1, capacity, kv_heads, head_dim), contiguous; seqlens: int32 (1,) on the GPU. Returns (1, count, heads, head_dim)."""
    _, count, heads, head_dim = query.shape
    kv_heads = key_cache.shape[2]
    groups = heads // kv_heads
    rows = groups * count
    row_blocks = triton.cdiv(rows, _BLOCK_R)
    rows_pad = row_blocks * _BLOCK_R
    bf16 = query.dtype == torch.bfloat16
    bits = (lambda t: t.view(torch.uint16)) if bf16 else (lambda t: t)
    q, k, v = bits(query), bits(key_cache), bits(value_cache)
    tensor = _TENSOR and bf16 and kernel_policy.allow_approximate() and torch.cuda.get_device_capability(query.device) >= (8, 0)
    splits = max(1, torch.cuda.get_device_properties(query.device).multi_processor_count * (_TENSOR_SPLITS_PER_SM if tensor else _SPLITS_PER_SM) // (kv_heads * row_blocks))
    candidates = [config for config in _TENSOR_CONFIGS if _fits_tensor(config, head_dim, query.device)] if tensor else [config for config in _CONFIGS if _fits(config, head_dim, query.device)]

    def launch(pm, pl, pacc, config):
        block_n, warps, stages = config
        if tensor:
            _static_decode_tensor[(kv_heads, splits, row_blocks)](query, key_cache, value_cache, seqlens, pm, pl, pacc, count, splits, scale, query.stride(1), key_cache.stride(1),
                                                                  GROUPS=groups, KV_HEADS=kv_heads, HEAD_DIM=head_dim, ROWS_PAD=rows_pad, BLOCK_R=_BLOCK_R, BLOCK_N=block_n, num_warps=warps, num_stages=stages)
            return
        _static_decode[(kv_heads, splits, row_blocks)](q, k, v, seqlens, pm, pl, pacc, count, splits, scale, query.stride(1), key_cache.stride(1),
                                                       GROUPS=groups, KV_HEADS=kv_heads, HEAD_DIM=head_dim, ROWS_PAD=rows_pad, BF16=bf16,
                                                       BLOCK_R=_BLOCK_R, BLOCK_N=block_n, num_warps=warps, num_stages=stages)

    def partials():
        return (torch.empty((splits, kv_heads, rows_pad), dtype=torch.float32, device=query.device),
                torch.empty((splits, kv_heads, rows_pad), dtype=torch.float32, device=query.device),
                torch.empty((splits, kv_heads, rows_pad, head_dim), dtype=torch.float32, device=query.device))

    def prepare_scratch():  # the cache and the query are only read: the candidates write scratch partial results
        scratch = partials()
        return lambda config: launch(*scratch, config)

    config = _TUNER.config("decode_tensor" if tensor else "decode", f"{query.dtype}|D{head_dim}|G{groups}|KV{kv_heads}", query.device, candidates, prepare_scratch, True)
    pm, pl, pacc = partials()
    launch(pm, pl, pacc, config)
    if tensor:
        output = torch.empty_like(query)
        _static_merge[(kv_heads, row_blocks)](seqlens, pm, pl, pacc, output, count, splits, query.stride(1), GROUPS=groups, KV_HEADS=kv_heads, HEAD_DIM=head_dim, ROWS_PAD=rows_pad, BLOCK_R=_BLOCK_R, BLOCK_N=config[0], num_warps=4)
        return output
    pm, pl, pacc = pm[:, :, :rows], pl[:, :, :rows], pacc[:, :, :rows]
    weights = torch.exp(pm - pm.amax(0))  # a split without keys: exp(-inf) = 0
    output = (weights.unsqueeze(-1) * pacc).sum(0).div_((weights * pl).sum(0).unsqueeze(-1))  # (kv_heads, groups * count, head_dim)
    return output.view(kv_heads, groups, count, head_dim).permute(2, 0, 1, 3).reshape(1, count, heads, head_dim).to(query.dtype)


from shared.kernels.triton_compilation_log import install_triton_compilation_logger
install_triton_compilation_logger()
