# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Attention of the queries of a prompt pass (hundreds to thousands of tokens) of a model with grouped queries, a relative-position bias
per head and backward distance, a sink logit per head and query and a sigmoid gate on the output (Inkling 2), without the scores in
memory: query tiles over key tiles with a running float32 softmax, on the tensor cores.

A tile's rows are (token, query head of the KV head) pairs of consecutive tokens: the 3 query heads of a KV head share each key / value
tile. Keys and values are read in place: the pages of a paged cache through the sequence's block table (BF16, or a quantized body with
its exact overlays, dequantized as the decoding kernels of ring_attention_triton read them: the same BF16 values as the PyTorch path),
or rows of consecutive positions (the window before a pass and the pass's own keys). Scores are rounded as the PyTorch path rounds them
(the BF16 product, its scaling, the added bias); the bias of a backward distance below its extent is read from the pass's BF16 table
(the product of each query's low-rank term and the projection, as the PyTorch path computes it); the weights of the keys are rounded to
BF16 before the values (relative to the largest logit so far) and their rounded values are summed for the normalization. Long contexts
with few queries split the keys of a tile in parts merged in their order with the sink. Head counts, widths and the bias extent are
architecture constants; tokens, positions, the context, the window and the splits are runtime arguments (a new prompt length compiles
nothing). NVIDIA GPUs from compute capability 7.5 (below 8.0, and on AMD where HIP_TILES says so, the sliced kernel: _tiles) and the AMD
architectures of shared.kernels.bf16_bits.HIP_ARCHS; elsewhere the caller keeps its PyTorch path."""
from functools import cache

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from shared.kernels.bf16_bits import bf16_bits_to_fp32, bits_view, convert, fp32_to_bf16_bits, portable_supported, reads_bits, to_float
from shared.kernels.ring_attention_triton import FORMATS, _rounded, _step_scores, _step_values, _store_slices, _tile

BLOCK_M = 64  # (token, query head) rows of a program
BLOCK_N = 32  # keys per step (64: register spills at 8 warps)
WARPS = 8  # 4: spills
STAGES = 2
PRECISE = True  # the weights of the keys as two BF16 parts (about 16 bits) against the values, their float32 sum for the normalization: less error than the PyTorch path's BF16 probabilities; False: rounded to BF16 as that path does
WAVE_PROGRAMS = 2 * 170  # programs that fill a large GPU about twice: fewer query tiles split their keys


@cache
def supported(device):
    return portable_supported(device)


TURING_TILES = (16, 16, 8, 1, 2)  # BLOCK_M, BLOCK_N, WARPS, STAGES, head-dimension slices below compute capability 8.0 (BF16 bits, float32 FMA products): no stack frame on sm_75, every layout
HIP_TILES = {"gfx1200": (64, 32, 16, 1, 2), "gfx1201": (64, 32, 16, 1, 2),  # by AMD architecture, the same (the least scratch: none but on RDNA3; gfx942 keeps the defaults)
             **dict.fromkeys(("gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151"), (32, 16, 8, 1, 4))}


@cache
def _tiles(device):
    if reads_bits(device):
        return TURING_TILES
    if torch.version.hip is not None:
        return HIP_TILES.get(torch.cuda.get_device_properties(device).gcnArchName.split(":")[0])
    return None


@triton.jit(do_not_specialize=("TOKENS", "SIZE", "WINDOW", "KEY_SHIFT", "SPLITS", "scale", "SINK_ROWS", "RING_MASK"),
            do_not_specialize_on_alignment=("Q", "BIAS", "KEYS", "VALUES", "TABLE", "POS", "SINK", "GATE", "OUT", "PARTS", "K_SCALE", "K_MIN", "K_OVER", "K_OVER_SCALE", "V_SCALE", "V_MIN",
                                            "V_OVER", "V_OVER_SCALE", "BASE", "RING_START"))
def _prefill_attention(Q, BIAS, KEYS, VALUES, TABLE, POS, SINK, GATE, OUT, PARTS, K_SCALE, K_MIN, K_OVER, K_OVER_SCALE, V_SCALE, V_MIN, V_OVER, V_OVER_SCALE, BASE, RING_START,
                       TOKENS, SIZE, WINDOW, KEY_SHIFT, SPLITS, scale, SINK_ROWS, RING_MASK,
                       HEADS: tl.constexpr, KV: tl.constexpr, D: tl.constexpr, EXTENT: tl.constexpr, M: tl.constexpr, N: tl.constexpr, PAGED: tl.constexpr,
                       K_FORMAT: tl.constexpr, V_FORMAT: tl.constexpr, OVERLAY: tl.constexpr, SPLIT: tl.constexpr, PRECISE: tl.constexpr, REAL, HAS_REAL: tl.constexpr):
    # Rows [tile * M, tile * M + M) of the (token, query head) pairs of KV head kv, over part `split` of the keys their tokens read.
    # PAGED: key index = position (the cache's pages through TABLE, SIZE rows each); else row j of KEYS holds position POS[0] - KEY_SHIFT + j.
    # SPLIT: the running softmax of the part into PARTS [splits, tokens * heads, D + 2] (merged by _merge); else the sink, the normalization and the gate here, into OUT.
    HPK: tl.constexpr = HEADS // KV
    tile = tl.program_id(0)
    kv = tl.program_id(1)
    split = tl.program_id(2)
    origin = tl.load(POS).to(tl.int32)  # the position of the pass's first token (positions, keys and distances in 32 bits: [M, N] tensors of 64-bit indices spill)
    rows = tile * M + tl.arange(0, M)
    token = rows // HPK
    real = token < TOKENS
    heads = kv * HPK + rows % HPK
    position = origin + token  # [M]
    first = origin + tile * M // HPK
    last = tl.minimum(origin + (tile * M + M - 1) // HPK, origin + TOKENS - 1)
    if HAS_REAL:  # a padded pass: no row reads the keys of the padded rows (written nowhere)
        last = tl.minimum(last, origin + tl.load(REAL) - 1)
    base_key = 0 if PAGED else origin - KEY_SHIFT  # the position of key index 0
    place = token.to(tl.int64) * HEADS + heads  # the (token, head) row of q, the bias table, the gate and the output
    low = tl.where(WINDOW > 0, tl.maximum(first - WINDOW + 1, 0), 0)  # positions the tile's tokens read
    low = tl.maximum(low - base_key, 0)  # as key indices
    high = last - base_key + 1
    steps = (high - low + N - 1) // N
    per = (steps + SPLITS - 1) // SPLITS
    start_step = split * per
    stop_step = tl.minimum(start_step + per, steps)
    width = tl.arange(0, D)
    within = tl.arange(0, N)
    q = tl.load(Q + place[:, None] * D + width[None, :], mask=real[:, None], other=0.0)  # [M, D] BF16
    if SPLIT:
        largest = tl.full((M,), float("-inf"), tl.float32)
        total = tl.zeros((M,), tl.float32)
    else:
        largest = tl.load(SINK + place, mask=real, other=0.0).to(tl.float32)  # the sink: a logit without a value
        total = tl.zeros((M,), tl.float32) + 1.0
    values = tl.zeros((M, D), tl.float32)
    if OVERLAY:
        overlay_base = tl.load(BASE).to(tl.int64)
        ring_start = tl.load(RING_START)
    for step in tl.range(start_step, stop_step):
        key = low + step * N + within  # key indices
        key_position = base_key + key
        valid = (key < high) & (key_position >= 0)
        if PAGED:
            row = tl.load(TABLE + key // SIZE, mask=valid, other=0).to(tl.int64) * SIZE + key % SIZE
        else:
            row = key.to(tl.int64)
        exact, exact_row = valid, row
        if OVERLAY:  # as the decoding kernels: the positions whose exact rows are read (sink pages, the ring of the last positions)
            exact = valid & (overlay_base >= 0) & ((key_position < SINK_ROWS) | (key_position >= ring_start))
            exact_row = tl.maximum(overlay_base, 0) + tl.where(key_position < SINK_ROWS, key_position, SINK_ROWS + (key_position & RING_MASK))
        k = _tile(KEYS, K_SCALE, K_MIN, K_OVER, K_OVER_SCALE, row, exact_row, exact, valid, kv, KV, D, K_FORMAT, OVERLAY)  # [N, D] BF16
        score = tl.dot(q, tl.trans(k)).to(tl.bfloat16).to(tl.float32)  # [M, N]: the BF16 product
        score = (score * scale).to(tl.bfloat16).to(tl.float32)
        distance = position[:, None] - key_position[None, :]
        allowed = real[:, None] & valid[None, :] & (distance >= 0)
        if WINDOW > 0:
            allowed = allowed & (distance < WINDOW)
        if first - (base_key + low + step * N + N - 1) < EXTENT:  # a key of the step within the bias extent of some row: the table's BF16 bias
            near = allowed & (distance < EXTENT)
            bias = tl.load((BIAS + place * EXTENT)[:, None] + distance, mask=near, other=0.0).to(tl.float32)
            score = (score + bias).to(tl.bfloat16).to(tl.float32)
        score = tl.where(allowed, score, float("-inf"))
        top = tl.maximum(largest, tl.max(score, axis=1))
        decay = tl.where(largest == top, 1.0, libdevice.exp(largest - top))  # (a row may have no key in a step)
        weight = tl.where(allowed, libdevice.exp(score - top[:, None]), 0.0)
        v = _tile(VALUES, V_SCALE, V_MIN, V_OVER, V_OVER_SCALE, row, exact_row, exact, valid, kv, KV, D, V_FORMAT, OVERLAY)
        if PRECISE:
            total = total * decay + tl.sum(weight, axis=1)
            head_part = weight.to(tl.bfloat16)
            values = tl.dot((weight - head_part.to(tl.float32)).to(tl.bfloat16), v, values * decay[:, None])
            values = tl.dot(head_part, v, values)
        else:  # rounded as the PyTorch path rounds its probabilities
            weight = weight.to(tl.bfloat16)
            total = total * decay + tl.sum(weight.to(tl.float32), axis=1)
            values = values * decay[:, None] + tl.dot(weight, v)
        largest = top
    if SPLIT:
        target = PARTS + (split * TOKENS * HEADS + place) * (D + 2)
        tl.store(target[:, None] + width[None, :], values, mask=real[:, None])
        tl.store(target + D, largest, mask=real)
        tl.store(target + D + 1, total, mask=real)
    else:
        out = tl.div_rn(values, total[:, None])
        gate = tl.load(GATE + place[:, None] * D + width[None, :], mask=real[:, None], other=0.0).to(tl.float32)
        out = out.to(tl.bfloat16).to(tl.float32) * tl.div_rn(1.0, 1.0 + libdevice.exp(-gate))
        tl.store(OUT + place[:, None] * D + width[None, :], out.to(OUT.dtype.element_ty), mask=real[:, None])


@triton.jit(do_not_specialize=("TOKENS", "SPLITS"), do_not_specialize_on_alignment=("PARTS", "SINK", "GATE", "OUT"))
def _merge(PARTS, SINK, GATE, OUT, TOKENS, SPLITS, HEADS: tl.constexpr, D: tl.constexpr, M: tl.constexpr, BITS: tl.constexpr = False):
    # The parts of M (token, head) rows merged in their order with the sink, then the gate.
    rows = tl.program_id(0) * M + tl.arange(0, M)
    real = rows < TOKENS * HEADS
    width = tl.arange(0, D)
    if BITS:  # (the BF16 tensors as their bits)
        largest = bf16_bits_to_fp32(tl.load(SINK + rows, mask=real, other=0))
    else:
        largest = tl.load(SINK + rows, mask=real, other=0.0).to(tl.float32)
    total = tl.zeros((M,), tl.float32) + 1.0
    values = tl.zeros((M, D), tl.float32)
    for split in tl.range(SPLITS):
        source = PARTS + (split * TOKENS * HEADS + rows.to(tl.int64)) * (D + 2)
        other = tl.load(source + D, mask=real, other=float("-inf"))
        top = tl.maximum(largest, other)
        decay = libdevice.exp(largest - top)
        weight = libdevice.exp(other - top)
        total = total * decay + tl.load(source + D + 1, mask=real, other=0.0) * weight
        values = values * decay[:, None] + tl.load(source[:, None] + width[None, :], mask=real[:, None], other=0.0) * weight[:, None]
        largest = top
    out = tl.div_rn(values, total[:, None])
    place = rows.to(tl.int64)[:, None] * D + width[None, :]
    if BITS:
        gate = bf16_bits_to_fp32(tl.load(GATE + place, mask=real[:, None], other=0))
        out = bf16_bits_to_fp32(fp32_to_bf16_bits(out)) * tl.div_rn(1.0, 1.0 + libdevice.exp(-gate))
        tl.store(OUT + place, fp32_to_bf16_bits(out), mask=real[:, None])
    else:
        gate = tl.load(GATE + place, mask=real[:, None], other=0.0).to(tl.float32)
        out = out.to(tl.bfloat16).to(tl.float32) * tl.div_rn(1.0, 1.0 + libdevice.exp(-gate))
        tl.store(OUT + place, out.to(OUT.dtype.element_ty), mask=real[:, None])


def attend(q, bias, sinks, keys, values, positions, scale, gate, window=0, key_shift=0, table=None, context=None, quantized=None, dtype=torch.bfloat16, real=None):
    """[tokens, heads * dim] in dtype: the queries q [tokens, heads, dim] of consecutive positions (positions[0]: the first) over the keys of the positions up to theirs.
    bias [tokens, heads, extent] BF16: the logit added at each backward distance below extent; sinks [tokens, heads]: the sink's logit; gate [tokens, heads * dim]: the output
    rounded to BF16 times its sigmoid. Keys and values: with table (int32 [pages], the sequence's pages in order) the pages [pages, rows, kv_heads, ...] of a cache, key index =
    position, quantized as ring_attention_triton.attend takes it; else rows [count, kv_heads, dim] of consecutive positions, row j at positions[0] - key_shift + j. window: the
    positions a query reads, itself included (0: all). context: the positions the last query reads at most (the pass's end), for the split of long contexts. real: int32 [1] on
    the device, a padded pass's real rows (the first ones): no row reads the keys of the others (they are stored nowhere)."""
    tokens, heads, dim = q.shape
    kv_heads = keys.shape[-2]
    if not (q.is_contiguous() and bias.is_contiguous() and sinks.is_contiguous() and gate.is_contiguous() and keys.is_contiguous() and values.is_contiguous()):
        raise ValueError("The prompt attention takes contiguous tensors")
    if not (q.is_cuda and keys.is_cuda and positions.is_cuda and bias.is_cuda):
        raise ValueError("The prompt attention runs on CUDA tensors")
    out = torch.empty(tokens, heads * dim, dtype=dtype, device=q.device)
    chosen = _tiles(q.device)
    block_m, block_n = (BLOCK_M, BLOCK_N) if chosen is None else chosen[:2]
    tiles = -(-tokens * (heads // kv_heads) // block_m)
    reach = (window if window > 0 else (context or tokens)) + block_m // (heads // kv_heads)  # keys a tile reads at most
    splits = max(1, min(-(-WAVE_PROGRAMS // (tiles * kv_heads)), reach // (8 * block_n)))  # parts of at least 8 steps
    spare = positions  # an argument a launch does not read
    k_format = v_format = 0
    k_extra = v_extra = (spare,) * 4
    base = ring_start = spare
    sink_rows = ring_mask = 0
    overlay = False
    if quantized is not None:
        k_format, v_format = FORMATS[quantized["k"]], FORMATS[quantized["v"]]
        overlay = quantized.get("overlay") is not None
        extras = [(quantized["scales"][index], spare if quantized["mins"][index] is None else quantized["mins"][index], *((quantized["overlay"][index], quantized["overlay_scales"][index]) if overlay else (spare, spare)))
                  for index in range(2)]
        k_extra, v_extra = extras
        if overlay:
            base, ring_start, sink_rows, ring_mask = quantized["base"], quantized["ring_start"], quantized["sink"], max(quantized["recent"], 1) - 1
        if not all(tensor.is_contiguous() for tensor in (*k_extra, *v_extra)):
            raise ValueError("The prompt attention reads contiguous cache tensors")
    parts = torch.empty(splits, tokens * heads, dim + 2, dtype=torch.float32, device=q.device) if splits > 1 else spare
    size = keys.shape[-3] if table is not None else 0
    if chosen is not None:  # (Turing, AMD) other tiles, the head dimension by slices
        _, _, warps, stages, chunks = chosen
        bits, result = reads_bits(q.device), out
        q, bias, sinks, gate, keys, values, out = (bits_view(tensor, bits) for tensor in (q, bias, sinks, gate, keys, values, out))
        (_prefill_attention_sliced if chunks else _prefill_attention)[(tiles, kv_heads, splits)](
            q, bias, keys, values, positions if table is None else table, positions, sinks, gate, out, parts, *k_extra, *v_extra, base, ring_start, tokens, size, window, key_shift, splits, scale, sink_rows,
            ring_mask, heads, kv_heads, dim, bias.shape[-1], block_m, block_n, table is not None, k_format, v_format, overlay, splits > 1, PRECISE, positions if real is None else real, real is not None,
            *((chunks, bits) if chunks else ()), num_warps=warps, num_stages=stages, enable_fp_fusion=False)
        if splits > 1:
            _merge[(-(-tokens * heads // 32),)](parts, sinks, gate, out, tokens, splits, heads, dim, 32, bits, num_warps=4, enable_fp_fusion=False)
        return result
    _prefill_attention[(tiles, kv_heads, splits)](q, bias, keys, values, positions if table is None else table, positions, sinks, gate, out, parts, *k_extra, *v_extra, base, ring_start,
                                                  tokens, size, window, key_shift, splits, scale, sink_rows, ring_mask, heads, kv_heads, dim, bias.shape[-1], BLOCK_M, BLOCK_N, table is not None,
                                                  k_format, v_format, overlay, splits > 1, PRECISE, positions if real is None else real, real is not None, num_warps=WARPS, num_stages=STAGES, enable_fp_fusion=False)
    if splits > 1:
        _merge[(-(-tokens * heads // 32),)](parts, sinks, gate, out, tokens, splits, heads, dim, 32, num_warps=4, enable_fp_fusion=False)
    return out


@triton.jit(do_not_specialize=("TOKENS", "SIZE", "WINDOW", "KEY_SHIFT", "SPLITS", "scale", "SINK_ROWS", "RING_MASK"),
            do_not_specialize_on_alignment=("Q", "BIAS", "KEYS", "VALUES", "TABLE", "POS", "SINK", "GATE", "OUT", "PARTS", "K_SCALE", "K_MIN", "K_OVER", "K_OVER_SCALE", "V_SCALE", "V_MIN",
                                            "V_OVER", "V_OVER_SCALE", "BASE", "RING_START"))
def _prefill_attention_sliced(Q, BIAS, KEYS, VALUES, TABLE, POS, SINK, GATE, OUT, PARTS, K_SCALE, K_MIN, K_OVER, K_OVER_SCALE, V_SCALE, V_MIN, V_OVER, V_OVER_SCALE, BASE, RING_START,
                              TOKENS, SIZE, WINDOW, KEY_SHIFT, SPLITS, scale, SINK_ROWS, RING_MASK,
                              HEADS: tl.constexpr, KV: tl.constexpr, D: tl.constexpr, EXTENT: tl.constexpr, M: tl.constexpr, N: tl.constexpr, PAGED: tl.constexpr,
                              K_FORMAT: tl.constexpr, V_FORMAT: tl.constexpr, OVERLAY: tl.constexpr, SPLIT: tl.constexpr, PRECISE: tl.constexpr, REAL, HAS_REAL: tl.constexpr,
                              CHUNKS: tl.constexpr, BITS: tl.constexpr):
    # _prefill_attention by slices of the head dimension (ring_attention_triton's sliced kernels: a step holds a slice of a key / value tile). BITS: the
    # BF16 tensors as their bits and float32 FMA products, the weights of the keys in float32 against the values (PRECISE or not: FMAs take them whole).
    HPK: tl.constexpr = HEADS // KV
    W: tl.constexpr = D // CHUNKS
    tile = tl.program_id(0)
    kv = tl.program_id(1)
    split = tl.program_id(2)
    origin = tl.load(POS).to(tl.int32)
    rows = tile * M + tl.arange(0, M)
    token = rows // HPK
    real = token < TOKENS
    heads = kv * HPK + rows % HPK
    position = origin + token
    first = origin + tile * M // HPK
    last = tl.minimum(origin + (tile * M + M - 1) // HPK, origin + TOKENS - 1)
    if HAS_REAL:
        last = tl.minimum(last, origin + tl.load(REAL) - 1)
    base_key = 0 if PAGED else origin - KEY_SHIFT
    place = token.to(tl.int64) * HEADS + heads
    low = tl.where(WINDOW > 0, tl.maximum(first - WINDOW + 1, 0), 0)
    low = tl.maximum(low - base_key, 0)
    high = last - base_key + 1
    steps = (high - low + N - 1) // N
    per = (steps + SPLITS - 1) // SPLITS
    start_step = split * per
    stop_step = tl.minimum(start_step + per, steps)
    within = tl.arange(0, N)
    if SPLIT:
        largest = tl.full((M,), float("-inf"), tl.float32)
        total = tl.zeros((M,), tl.float32)
    else:
        largest = to_float(tl.load(SINK + place, mask=real, other=0))
        total = tl.zeros((M,), tl.float32) + 1.0
    values0 = tl.zeros((M, W), tl.float32)
    values1 = tl.zeros((M, W), tl.float32)
    values2 = tl.zeros((M, W), tl.float32)
    values3 = tl.zeros((M, W), tl.float32)
    if OVERLAY:
        overlay_base = tl.load(BASE).to(tl.int64)
        ring_start = tl.load(RING_START)
    for step in tl.range(start_step, stop_step):
        key = low + step * N + within
        key_position = base_key + key
        valid = (key < high) & (key_position >= 0)
        if PAGED:
            row = tl.load(TABLE + key // SIZE, mask=valid, other=0).to(tl.int64) * SIZE + key % SIZE
        else:
            row = key.to(tl.int64)
        exact, exact_row = valid, row
        if OVERLAY:
            exact = valid & (overlay_base >= 0) & ((key_position < SINK_ROWS) | (key_position >= ring_start))
            exact_row = tl.maximum(overlay_base, 0) + tl.where(key_position < SINK_ROWS, key_position, SINK_ROWS + (key_position & RING_MASK))
        score = _step_scores(Q, place, real, KEYS, K_SCALE, K_MIN, K_OVER, K_OVER_SCALE, row, exact_row, exact, valid, kv, scale, KV, D, K_FORMAT, OVERLAY, M, N, CHUNKS, BITS)
        distance = position[:, None] - key_position[None, :]
        allowed = real[:, None] & valid[None, :] & (distance >= 0)
        if WINDOW > 0:
            allowed = allowed & (distance < WINDOW)
        if first - (base_key + low + step * N + N - 1) < EXTENT:
            near = allowed & (distance < EXTENT)
            bias = to_float(tl.load((BIAS + place * EXTENT)[:, None] + distance, mask=near, other=0))
            score = _rounded(score + bias, BITS)
        score = tl.where(allowed, score, float("-inf"))
        top = tl.maximum(largest, tl.max(score, axis=1))
        decay = tl.where(largest == top, 1.0, libdevice.exp(largest - top))
        weight = tl.where(allowed, libdevice.exp(score - top[:, None]), 0.0)
        if BITS:
            total = total * decay + tl.sum(weight, axis=1)
            values0, values1, values2, values3 = _step_values(values0, values1, values2, values3, weight, None, decay, VALUES, V_SCALE, V_MIN, V_OVER, V_OVER_SCALE, row, exact_row, exact, valid,
                                                              kv, KV, D, V_FORMAT, OVERLAY, CHUNKS, BITS)
        elif PRECISE:
            total = total * decay + tl.sum(weight, axis=1)
            head_part = weight.to(tl.bfloat16)
            values0, values1, values2, values3 = _step_values(values0, values1, values2, values3, head_part, (weight - head_part.to(tl.float32)).to(tl.bfloat16), decay, VALUES, V_SCALE, V_MIN,
                                                              V_OVER, V_OVER_SCALE, row, exact_row, exact, valid, kv, KV, D, V_FORMAT, OVERLAY, CHUNKS, BITS, True)
        else:
            weight = weight.to(tl.bfloat16)
            total = total * decay + tl.sum(weight.to(tl.float32), axis=1)
            values0, values1, values2, values3 = _step_values(values0, values1, values2, values3, weight, None, decay, VALUES, V_SCALE, V_MIN, V_OVER, V_OVER_SCALE, row, exact_row, exact, valid,
                                                              kv, KV, D, V_FORMAT, OVERLAY, CHUNKS, BITS)
        largest = top
    if SPLIT:
        target = PARTS + (split * TOKENS * HEADS + place) * (D + 2)
        _store_slices(target, values0, values1, values2, values3, real, W, CHUNKS)
        tl.store(target + D, largest, mask=real)
        tl.store(target + D + 1, total, mask=real)
    else:
        for c in tl.static_range(CHUNKS):
            width = c * W + tl.arange(0, W)
            if c == 0:
                out = tl.div_rn(values0, total[:, None])
            elif c == 1:
                out = tl.div_rn(values1, total[:, None])
            elif c == 2:
                out = tl.div_rn(values2, total[:, None])
            else:
                out = tl.div_rn(values3, total[:, None])
            gate = to_float(tl.load(GATE + place[:, None] * D + width[None, :], mask=real[:, None], other=0))
            out = _rounded(out, BITS) * tl.div_rn(1.0, 1.0 + libdevice.exp(-gate))
            tl.store(OUT + place[:, None] * D + width[None, :], convert(out, OUT), mask=real[:, None])


from shared.kernels.triton_compilation_log import install_triton_compilation_logger  # noqa: E402

install_triton_compilation_logger()
