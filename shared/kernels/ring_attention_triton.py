# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Attention of the few queries of a decoding pass over keys and values kept by position: a ring of the last positions (sliding-window
layers) or the pages of a BF16 paged KV cache through its block table (full-attention layers). Grouped queries, a relative-position bias
per head and backward distance (the BF16 product of a low-rank query term and a table), a sink logit per head and query, an optional
sigmoid gate on the output.

Two kernels. Programs per token, KV head and part of the context (an eighth of a window, a page) read the keys and values of their part
in place, once, with a running float32 softmax (largest logit so far, sum and weighted values rescaled when it grows); a program per
token and KV head then merges the parts in their order with the sink. Only the positions of the context are read, whatever the size of
the cache; the parts run side by side, so a launch takes about the same time at any context length; a query's result does not depend on
the other queries of the pass. Scores are rounded as the PyTorch path stores them (the BF16 product, its scaling, the added bias); the
weights of the keys are rounded to BF16 before the values as that path's probabilities are (relative to the largest logit of their part,
not to the final sum). The products run on the tensor cores (BF16 tiles of 16 query rows, float32 sums). Head counts, widths and the
bias extent are architecture constants; tokens, positions, the ring size and the window are runtime arguments. NVIDIA GPUs from compute
capability 7.5 and the AMD architectures of shared.kernels.bf16_bits.HIP_ARCHS: from 8.0 as described; elsewhere by the sliced kernels below (_tiles).
attend(..., shared=True): the parts are aligned on the key index and one program per KV head and part serves every token of the pass (a
verification window; in bands of up to 10 tokens, the 32 rows of a tile), so that a tile is read and dequantized once for all its rows; a token's result does not depend on the others there
either (the bits of a pass of that token alone in the same mode)."""
import os
from functools import cache

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from shared.kernels.bf16_bits import bf16_bits_to_fp32, bits_view, fp32_to_bf16_bits, portable_supported, reads_bits, to_float

ROWS = 16  # tokens of a pass (decoding, verification): larger passes keep the PyTorch path, whose GEMMs serve many queries at once
CHUNK = 64  # keys per step of a program
WARPS = 8
STEPS_RING, STEPS_PAGED = 2, 4  # steps of a program: an eighth of a window of 1024 positions, a page of 256
STEPS_SHARED_RING = 2  # with shared tiles, parts aligned on the key index: a window straddles one part more
_DISABLED = os.environ.get("WGP_INKLING2_DISABLE", "").split(",")
WIDE_K = tl.constexpr("wide_tiles" not in _DISABLED and "wide_k" not in _DISABLED)  # quantized key / value tiles read a record's bytes and its group scales in contiguous loads: the same values, 3.3 times faster on 4-bit pages (the loads of one per value bound the kernel); the products of 4-bit tiles then round about 1 output in 2,000 differently (a row alone = in a window still). Disabled: a load per value
WIDE_V = tl.constexpr("wide_tiles" not in _DISABLED and "wide_v" not in _DISABLED)
MERGE_SLICES = 1 if "merge_slices" in _DISABLED else 16  # programs per token and KV head of the shared tiles' merge, each over its 8 columns of the values (the same bits; RTX 5090, 28K: 8 paged + 40 window merges 0.9 ms faster)
SHARED_ROWS = 32  # rows of a shared-tile program at most (the query heads of up to 10 tokens): 64-row tiles spill to the stack; longer passes run in bands of tokens


@cache
def supported(device):
    return portable_supported(device)


TURING_TILES = (4, 16, 8, 16)  # below compute capability 8.0 (BF16 bits, float32 FMA products): head dimension in 4 slices, 16 keys per step, 8 warps, 16 rows per shared-tile program (no stack frame on sm_75 with Triton 3.2 and 3.6, every cache layout)
HIP_TILES = {"gfx1200": (2, 16, 16, 32), "gfx1201": (2, 16, 16, 32), "gfx942": (0, 32, 8, 32),  # by AMD architecture, the same fields (slices 0: the tensor-core
             **dict.fromkeys(("gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151"), (2, 16, 16, 32))}  # kernels whole), the least scratch (none but on RDNA3: tests/inkling2_port sweeps)


@cache
def _tiles(device):
    """The tiles of the part kernels on device (TURING_TILES, HIP_TILES), None for the tensor-core kernels as they are (NVIDIA from 8.0)."""
    if reads_bits(device):
        return TURING_TILES
    if torch.version.hip is not None:
        return HIP_TILES.get(torch.cuda.get_device_properties(device).gcnArchName.split(":")[0])
    return None


@triton.jit
def _tile(DATA, SCALE, MIN, OVER, OVER_SCALE, row, exact_row, exact, valid, kv, KV: tl.constexpr, D: tl.constexpr, FORMAT: tl.constexpr, OVERLAY: tl.constexpr, WIDE: tl.constexpr = False):
    """BF16 [keys, D]: the keys (or values) of a KV head at the rows of a step. FORMAT 0: BF16 rows; else a quantized body (layers.kv_formats: 1 q8, 2 q6_0, 3 q4_0, 4 q4_1; a
    scale and for q4_1 a minimum per group of 32 values), with OVERLAY the exact int8 rows of the positions that have one (exact) read instead. Dequantized in float32 and rounded
    to BF16, as the PyTorch path hands them to its attention."""
    width = tl.arange(0, D)
    if FORMAT == 0:
        return tl.load(DATA + (row * KV + kv)[:, None] * D + width[None, :], mask=valid[:, None], other=0.0)
    GROUPS: tl.constexpr = D // 32
    BYTES: tl.constexpr = 32 if FORMAT == 1 else 24 if FORMAT == 2 else 16
    N: tl.constexpr = row.shape[0]
    group, lane = (width // 32)[None, :], (width % 32)[None, :]
    body = valid & (exact.to(tl.int32) == 0) if OVERLAY else valid
    record = (row * KV + kv)[:, None]
    if WIDE and FORMAT != 2:  # a record's bytes in one contiguous load and its group scales once (not per value): the loads of the per-value form bound the kernel (Nsight: the LSU queue)
        scales = tl.reshape(tl.load(SCALE + record * GROUPS + tl.arange(0, GROUPS)[None, :], mask=body[:, None], other=0.0).to(tl.float32), (N, GROUPS, 1))
        if FORMAT == 1:
            values = tl.reshape(tl.load(DATA + record * D + width[None, :], mask=body[:, None], other=0).to(tl.float32), (N, GROUPS, 32)) * scales
        else:
            packed = tl.reshape(tl.load(DATA + record * (GROUPS * BYTES) + tl.arange(0, GROUPS * BYTES)[None, :], mask=body[:, None], other=0).to(tl.int32), (N, GROUPS, 16))
            code = tl.reshape(tl.permute(tl.join(packed & 15, packed >> 4), (0, 1, 3, 2)), (N, GROUPS, 32))  # lanes 0-15 of a group: the low nibbles, 16-31 the high ones
            if FORMAT == 4:
                values = code.to(tl.float32) * scales + tl.reshape(tl.load(MIN + record * GROUPS + tl.arange(0, GROUPS)[None, :], mask=body[:, None], other=0.0).to(tl.float32), (N, GROUPS, 1))
            else:
                values = (code - 8).to(tl.float32) * scales
        values = tl.reshape(values, (N, D))
    else:
        scales = tl.load(SCALE + record * GROUPS + group, mask=body[:, None], other=0.0).to(tl.float32)
        if FORMAT == 1:
            values = tl.load(DATA + record * D + width[None, :], mask=body[:, None], other=0).to(tl.float32) * scales
        else:
            packed = tl.load(DATA + record * (GROUPS * BYTES) + group * BYTES + (lane & 15), mask=body[:, None], other=0).to(tl.int32)
            code = tl.where(lane < 16, packed & 15, packed >> 4)
            if FORMAT == 2:
                plane = tl.load(DATA + record * (GROUPS * BYTES) + group * BYTES + 16 + (lane >> 4) * 4 + (lane & 3), mask=body[:, None], other=0).to(tl.int32)
                code = code | (((plane >> (((lane >> 2) & 3) * 2)) & 3) << 4)
            if FORMAT == 4:
                values = code.to(tl.float32) * scales + tl.load(MIN + record * GROUPS + group, mask=body[:, None], other=0.0).to(tl.float32)
            else:
                values = (code - (32 if FORMAT == 2 else 8)).to(tl.float32) * scales
    if OVERLAY:
        chosen = (valid & (exact.to(tl.int32) != 0))[:, None]
        record = (exact_row * KV + kv)[:, None]
        if WIDE:
            over_scales = tl.reshape(tl.load(OVER_SCALE + record * GROUPS + tl.arange(0, GROUPS)[None, :], mask=chosen, other=0.0).to(tl.float32), (N, GROUPS, 1))
            over = tl.reshape(tl.reshape(tl.load(OVER + record * D + width[None, :], mask=chosen, other=0).to(tl.float32), (N, GROUPS, 32)) * over_scales, (N, D))
        else:
            over = tl.load(OVER + record * D + width[None, :], mask=chosen, other=0).to(tl.float32) * tl.load(OVER_SCALE + record * GROUPS + group, mask=chosen, other=0.0).to(tl.float32)
        values = tl.where(chosen, over, values)
    return values.to(tl.bfloat16)


@triton.jit(do_not_specialize=("SIZE", "WINDOW", "scale", "SINK_ROWS", "RING_MASK"),
            do_not_specialize_on_alignment=("Q", "REL", "PROJ", "KEYS", "VALUES", "TABLE", "POS", "PARTS", "K_SCALE", "K_MIN", "K_OVER", "K_OVER_SCALE", "V_SCALE", "V_MIN", "V_OVER", "V_OVER_SCALE", "BASE", "RING_START"))
def _attention_part(Q, REL, PROJ, KEYS, VALUES, TABLE, POS, PARTS, K_SCALE, K_MIN, K_OVER, K_OVER_SCALE, V_SCALE, V_MIN, V_OVER, V_OVER_SCALE, BASE, RING_START, SIZE, WINDOW, scale, SINK_ROWS, RING_MASK,
                    HEADS: tl.constexpr, KV: tl.constexpr, D: tl.constexpr, R: tl.constexpr, EXTENT: tl.constexpr, STEP: tl.constexpr, STEPS: tl.constexpr, GROUP: tl.constexpr, PAGED: tl.constexpr,
                    K_FORMAT: tl.constexpr, V_FORMAT: tl.constexpr, OVERLAY: tl.constexpr):
    # A part of the context of a token and KV head (STEPS * STEP backward distances): per query head the largest logit of its keys, the sum of their weights relative to it
    # and the values weighted so, into PARTS [tokens, KV, parts, query heads of a KV head, D + 2]. SIZE: rows of the ring, or of a page; WINDOW: positions a query reads, itself included (0: all).
    token = tl.program_id(0).to(tl.int64)
    kv = tl.program_id(1)
    part = tl.program_id(2)
    position = tl.load(POS) + token
    count = position + 1
    if WINDOW > 0:
        count = tl.minimum(count, WINDOW)
    if part * (STEPS * STEP) < count:
        lanes = tl.arange(0, GROUP)  # the query heads of a KV head, then empty rows: GROUP is the 16 rows of a tensor-core tile
        real = lanes < HEADS // KV
        heads = kv * (HEADS // KV) + lanes
        width = tl.arange(0, D)
        within = tl.arange(0, STEP)
        q = tl.load(Q + (token * HEADS + heads)[:, None] * D + width[None, :], mask=real[:, None], other=0.0)  # [GROUP, D], BF16: the products below run on the tensor cores
        rel = tl.load(REL + (token * HEADS + heads)[:, None] * R + tl.arange(0, R)[None, :], mask=real[:, None], other=0.0)  # [GROUP, R]
        largest = tl.full((GROUP,), float("-inf"), tl.float32)
        total = tl.zeros((GROUP,), tl.float32)
        values = tl.zeros((GROUP, D), tl.float32)
        for step in tl.range(STEPS):  # a runtime loop: unrolled, the steps' tiles overlap and spill (stack frames beyond the context's 256-byte limit, which a launch then raises for good)
            distance = (part * STEPS + step) * STEP + within
            valid = distance < count
            key = position - distance
            if PAGED:
                row = tl.load(TABLE + key // SIZE, mask=valid, other=0).to(tl.int64) * SIZE + key % SIZE
            else:
                row = key % SIZE
            exact, exact_row = valid, row
            if OVERLAY:  # the positions whose exact rows are read: the sink pages and the ring of the last positions, of the sequence's overlay slot
                base = tl.load(BASE).to(tl.int64)
                exact = valid & (base >= 0) & ((key < SINK_ROWS) | (key >= tl.load(RING_START)))
                exact_row = tl.maximum(base, 0) + tl.where(key < SINK_ROWS, key, SINK_ROWS + (key & RING_MASK))
            k = _tile(KEYS, K_SCALE, K_MIN, K_OVER, K_OVER_SCALE, row, exact_row, exact, valid, kv, KV, D, K_FORMAT, OVERLAY)  # [STEP, D]
            score = tl.dot(q, tl.trans(k)).to(tl.bfloat16).to(tl.float32)  # [GROUP, STEP]: float32 sums of BF16 products, rounded as the BF16 product
            score = (score * scale).to(tl.bfloat16).to(tl.float32)
            near = valid & (distance < EXTENT)
            proj = tl.load(PROJ + tl.arange(0, R)[:, None] * EXTENT + distance[None, :], mask=near[None, :], other=0.0)  # [R, STEP]
            score = (score + tl.dot(rel, proj).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16).to(tl.float32)  # the bias: a BF16 product, as its sum with the score
            score = tl.where(valid[None, :], score, float("-inf"))
            top = tl.maximum(largest, tl.max(score, axis=1))
            decay = tl.where(largest == top, 1.0, libdevice.exp(largest - top))  # (the keys of a step may all lie beyond the context)
            weight = tl.where(valid[None, :], libdevice.exp(score - top[:, None]), 0.0)
            total = total * decay + tl.sum(weight, axis=1)
            v = _tile(VALUES, V_SCALE, V_MIN, V_OVER, V_OVER_SCALE, row, exact_row, exact, valid, kv, KV, D, V_FORMAT, OVERLAY)
            values = values * decay[:, None] + tl.dot(weight.to(tl.bfloat16), v)  # the weights rounded to BF16, as the PyTorch path rounds its probabilities
            largest = top
        target = PARTS + (((token * KV + kv) * tl.num_programs(2) + part) * (HEADS // KV) + lanes) * (D + 2)  # the query heads only
        tl.store(target[:, None] + width[None, :], values, mask=real[:, None])
        tl.store(target + D, largest, mask=real)
        tl.store(target + D + 1, total, mask=real)


@triton.jit(do_not_specialize=("WINDOW", "KEYS_PER_PART", "COUNT"), do_not_specialize_on_alignment=("PARTS", "SINK", "GATE", "POS", "OUT"))
def _attention_merge(PARTS, SINK, GATE, POS, OUT, WINDOW, KEYS_PER_PART, COUNT, HEADS: tl.constexpr, KV: tl.constexpr, D: tl.constexpr, GROUP: tl.constexpr, HAS_GATE: tl.constexpr,
                     BITS: tl.constexpr = False):
    # The parts of a token and KV head merged in their order with the sink (a logit without a value), then the gate. COUNT: the parts of the launch.
    token = tl.program_id(0).to(tl.int64)
    kv = tl.program_id(1)
    count = tl.load(POS) + token + 1
    if WINDOW > 0:
        count = tl.minimum(count, WINDOW)
    lanes = tl.arange(0, GROUP)
    real = lanes < HEADS // KV
    heads = kv * (HEADS // KV) + lanes
    width = tl.arange(0, D)
    if BITS:  # (the BF16 tensors as their bits)
        largest = bf16_bits_to_fp32(tl.load(SINK + token * HEADS + heads, mask=real, other=0))
    else:
        largest = tl.load(SINK + token * HEADS + heads, mask=real, other=0.0).to(tl.float32)
    total = tl.zeros((GROUP,), tl.float32) + 1.0
    values = tl.zeros((GROUP, D), tl.float32)
    for part in range((count + KEYS_PER_PART - 1) // KEYS_PER_PART):
        source = PARTS + (((token * KV + kv) * COUNT + part) * (HEADS // KV) + lanes) * (D + 2)
        other = tl.load(source + D, mask=real, other=0.0)
        top = tl.maximum(largest, other)
        decay = libdevice.exp(largest - top)
        weight = libdevice.exp(other - top)
        total = total * decay + tl.load(source + D + 1, mask=real, other=0.0) * weight
        values = values * decay[:, None] + tl.load(source[:, None] + width[None, :], mask=real[:, None], other=0.0) * weight[:, None]
        largest = top
    out = tl.div_rn(values, total[:, None])
    place = (token * HEADS + heads)[:, None] * D + width[None, :]
    if BITS:
        if HAS_GATE:
            gate = bf16_bits_to_fp32(tl.load(GATE + place, mask=real[:, None], other=0))
            out = bf16_bits_to_fp32(fp32_to_bf16_bits(out)) * tl.div_rn(1.0, 1.0 + libdevice.exp(-gate))
        tl.store(OUT + place, fp32_to_bf16_bits(out), mask=real[:, None])
    else:
        if HAS_GATE:
            gate = tl.load(GATE + place, mask=real[:, None], other=0.0).to(tl.float32)
            out = out.to(tl.bfloat16).to(tl.float32) * tl.div_rn(1.0, 1.0 + libdevice.exp(-gate))
        tl.store(OUT + place, out.to(OUT.dtype.element_ty), mask=real[:, None])


@triton.jit(do_not_specialize=("TOKENS", "BAND_TOKENS", "SIZE", "WINDOW", "scale", "SINK_ROWS", "RING_MASK"),
            do_not_specialize_on_alignment=("Q", "REL", "PROJ", "TABLE", "POS", "PARTS", "BASE", "RING_START"))  # the cache tensors keep their alignment: wide loads (WIDE)
def _shared_part(Q, REL, PROJ, KEYS, VALUES, TABLE, POS, PARTS, K_SCALE, K_MIN, K_OVER, K_OVER_SCALE, V_SCALE, V_MIN, V_OVER, V_OVER_SCALE, BASE, RING_START, TOKENS, BAND_TOKENS, SIZE, WINDOW, scale, SINK_ROWS, RING_MASK,
                 HEADS: tl.constexpr, KV: tl.constexpr, D: tl.constexpr, R: tl.constexpr, EXTENT: tl.constexpr, STEP: tl.constexpr, STEPS: tl.constexpr, GROUP: tl.constexpr, PAGED: tl.constexpr,
                 K_FORMAT: tl.constexpr, V_FORMAT: tl.constexpr, OVERLAY: tl.constexpr):
    # As _attention_part, with the parts aligned on the key index (part p: the keys [p * STEPS * STEP, (p + 1) * STEPS * STEP)) and one program per (KV head, part) for every token of
    # the pass: its tiles are read and dequantized once, the query heads of all the tokens in the GROUP rows of one tensor-core product (row: token * query heads of a KV head + head).
    # A token's rows are computed as they are in a pass of that token alone (each output of the products depends on its own row only; keys outside a token's reach weigh 0), so a
    # verification pass gives each row the bits of plain decoding. Part 0 of the launch: the first part of the first token's reach.
    # Bands: the tokens of a pass in groups of BAND_TOKENS (at most GROUP // HPK: 32 rows at most, larger tiles spill), one band per program_id(0) (the fastest axis: the bands of a part run side by side); a band reads and dequantizes the tiles of its
    # own reach, with the part numbering of the whole pass.
    HPK: tl.constexpr = HEADS // KV
    PART: tl.constexpr = STEPS * STEP
    band = tl.program_id(0)
    kv = tl.program_id(1)
    origin = tl.load(POS).to(tl.int64)  # the pass's first position: the parts' numbering
    first = origin + band * BAND_TOKENS  # the band's
    count = tl.minimum(TOKENS - band * BAND_TOKENS, BAND_TOKENS)
    last = first + count - 1
    low = tl.where(WINDOW > 0, tl.maximum(first - WINDOW + 1, 0), 0)
    part = tl.where(WINDOW > 0, tl.maximum(origin - WINDOW + 1, 0), 0) // PART + tl.program_id(2)
    if (part * PART <= last) & ((part + 1) * PART > low):
        lanes = tl.arange(0, GROUP)
        local = lanes // HPK
        token = (band * BAND_TOKENS + local).to(tl.int64)
        real = local < count
        heads = kv * HPK + lanes % HPK
        width = tl.arange(0, D)
        within = tl.arange(0, STEP)
        q = tl.load(Q + (token * HEADS + heads)[:, None] * D + width[None, :], mask=real[:, None], other=0.0)  # [GROUP, D], BF16
        rel = tl.load(REL + (token * HEADS + heads)[:, None] * R + tl.arange(0, R)[None, :], mask=real[:, None], other=0.0)  # [GROUP, R]
        largest = tl.full((GROUP,), float("-inf"), tl.float32)
        total = tl.zeros((GROUP,), tl.float32)
        values = tl.zeros((GROUP, D), tl.float32)
        for step in tl.range(STEPS):  # a runtime loop, as in _attention_part
            start = part * PART + step * STEP
            key = start + within
            valid = (key >= low) & (key <= last)  # the keys some token of the band reads
            if PAGED:
                row = tl.load(TABLE + key // SIZE, mask=valid, other=0).to(tl.int64) * SIZE + key % SIZE
            else:
                row = key % SIZE
            exact, exact_row = valid, row
            if OVERLAY:
                base = tl.load(BASE).to(tl.int64)
                exact = valid & (base >= 0) & ((key < SINK_ROWS) | (key >= tl.load(RING_START)))
                exact_row = tl.maximum(base, 0) + tl.where(key < SINK_ROWS, key, SINK_ROWS + (key & RING_MASK))
            k = _tile(KEYS, K_SCALE, K_MIN, K_OVER, K_OVER_SCALE, row, exact_row, exact, valid, kv, KV, D, K_FORMAT, OVERLAY, WIDE_K)  # [STEP, D]
            score = tl.dot(q, tl.trans(k)).to(tl.bfloat16).to(tl.float32)
            score = (score * scale).to(tl.bfloat16).to(tl.float32)
            distance = local[:, None] - (key - first).to(tl.int32)[None, :]  # [GROUP, STEP]: a token's position minus the key's, in 32 bits
            allowed = real[:, None] & (distance >= 0) & ((WINDOW == 0) | (distance < WINDOW))
            bias = tl.zeros((GROUP, STEP), tl.float32)
            if first - (start + STEP - 1) < EXTENT:  # a key of the step within the bias extent of some token: each token's distances read their own columns of the projection
                if count == 1:  # (the same bits as the loop: it would select these rows)
                    gap = first - key
                    proj = tl.load(PROJ + tl.arange(0, R)[:, None] * EXTENT + gap[None, :], mask=((gap >= 0) & (gap < EXTENT))[None, :], other=0.0)
                    bias = tl.dot(rel, proj).to(tl.bfloat16).to(tl.float32)
                else:
                    for t in range(count):
                        gap = first + t - key
                        near = (gap >= 0) & (gap < EXTENT)
                        proj = tl.load(PROJ + tl.arange(0, R)[:, None] * EXTENT + gap[None, :], mask=near[None, :], other=0.0)  # [R, STEP]
                        bias = tl.where((local == t)[:, None], tl.dot(rel, proj).to(tl.bfloat16).to(tl.float32), bias)
            score = (score + bias).to(tl.bfloat16).to(tl.float32)
            score = tl.where(allowed, score, float("-inf"))
            top = tl.maximum(largest, tl.max(score, axis=1))
            decay = tl.where(largest == top, 1.0, libdevice.exp(largest - top))
            weight = tl.where(allowed, libdevice.exp(score - top[:, None]), 0.0)
            # the sum of the weights as the values get them (rounded to BF16), by the same tensor-core product against a column of ones: a float32 tl.sum rounds a row differently
            # with its place in the tile (a token alone and in a window), and the normalization then matches the weights the values were summed with
            weight = weight.to(tl.bfloat16)
            ones = tl.where(tl.arange(0, 16)[None, :] == 0, 1.0, 0.0).to(tl.bfloat16) + tl.zeros((STEP, 16), tl.bfloat16)
            sums = total[:, None] * decay[:, None] + tl.dot(weight, ones)  # [GROUP, 16]: the running total in column 0
            total = tl.sum(tl.where(tl.arange(0, 16)[None, :] == 0, sums, 0.0), axis=1)
            v = _tile(VALUES, V_SCALE, V_MIN, V_OVER, V_OVER_SCALE, row, exact_row, exact, valid, kv, KV, D, V_FORMAT, OVERLAY, WIDE_V)
            values = values * decay[:, None] + tl.dot(weight, v)
            largest = top
        target = PARTS + (((token * KV + kv) * tl.num_programs(2) + tl.program_id(2)) * HPK + lanes % HPK) * (D + 2)
        tl.store(target[:, None] + width[None, :], values, mask=real[:, None])
        tl.store(target + D, largest, mask=real)
        tl.store(target + D + 1, total, mask=real)


@triton.jit
def _rescaled(running, decay, added):
    """running * decay + added in one FMA whatever the layout: written as a product and an add, ptxas fuses some of them into FMAs on sm_120 (packed float32 pairs), by layout."""
    return tl.fma(running, decay, added)


@triton.jit(do_not_specialize=("WINDOW", "PART", "COUNT"), do_not_specialize_on_alignment=("PARTS", "SINK", "GATE", "POS", "OUT"))
def _shared_merge(PARTS, SINK, GATE, POS, OUT, WINDOW, PART, COUNT, HEADS: tl.constexpr, KV: tl.constexpr, D: tl.constexpr, GROUP: tl.constexpr, HAS_GATE: tl.constexpr, SLICE: tl.constexpr,
                  BITS: tl.constexpr = False):
    # As _attention_merge over the parts of _shared_part: a token merges the parts of its own reach (by key index), in their order, as in a pass of that token alone.
    # program_id(2): the SLICE columns of the values it merges (each program computes the same running largest logits and totals, and every output as one program did).
    token = tl.program_id(0).to(tl.int64)
    kv = tl.program_id(1)
    first = tl.load(POS).to(tl.int64)
    position = first + token
    start = tl.where(WINDOW > 0, tl.maximum(first - WINDOW + 1, 0), 0) // PART
    own = tl.where(WINDOW > 0, tl.maximum(position - WINDOW + 1, 0), 0) // PART
    lanes = tl.arange(0, GROUP)
    real = lanes < HEADS // KV
    heads = kv * (HEADS // KV) + lanes
    width = tl.program_id(2) * SLICE + tl.arange(0, SLICE)
    if BITS:  # (the BF16 tensors as their bits)
        largest = bf16_bits_to_fp32(tl.load(SINK + token * HEADS + heads, mask=real, other=0))
    else:
        largest = tl.load(SINK + token * HEADS + heads, mask=real, other=0.0).to(tl.float32)
    total = tl.zeros((GROUP,), tl.float32) + 1.0
    values = tl.zeros((GROUP, SLICE), tl.float32)
    for part in tl.range(own - start, position // PART - start + 1, loop_unroll_factor=8):  # unrolled: the loads of several parts in flight
        source = PARTS + (((token * KV + kv) * COUNT + part) * (HEADS // KV) + lanes) * (D + 2)
        other = tl.load(source + D, mask=real, other=0.0)
        top = tl.maximum(largest, other)
        decay = libdevice.exp(largest - top)
        weight = libdevice.exp(other - top)
        total = _rescaled(total, decay, tl.load(source + D + 1, mask=real, other=0.0) * weight)
        values = _rescaled(values, decay[:, None], tl.load(source[:, None] + width[None, :], mask=real[:, None], other=0.0) * weight[:, None])
        largest = top
    out = tl.div_rn(values, total[:, None])
    place = (token * HEADS + heads)[:, None] * D + width[None, :]
    if BITS:
        if HAS_GATE:
            gate = bf16_bits_to_fp32(tl.load(GATE + place, mask=real[:, None], other=0))
            out = bf16_bits_to_fp32(fp32_to_bf16_bits(out)) * tl.div_rn(1.0, 1.0 + libdevice.exp(-gate))
        tl.store(OUT + place, fp32_to_bf16_bits(out), mask=real[:, None])
    else:
        if HAS_GATE:
            gate = tl.load(GATE + place, mask=real[:, None], other=0.0).to(tl.float32)
            out = out.to(tl.bfloat16).to(tl.float32) * tl.div_rn(1.0, 1.0 + libdevice.exp(-gate))
        tl.store(OUT + place, out.to(OUT.dtype.element_ty), mask=real[:, None])


FORMATS = {"q8": 1, "q6_0": 2, "q4_0": 3, "q4_1": 4}  # layers.kv_formats names of the quantized bodies


def attend(q, rel, proj, sinks, keys, values, positions, scale, window=0, table=None, gate=None, dtype=torch.bfloat16, quantized=None, shared=False):
    """[tokens, heads * dim] in dtype: the queries q [tokens, heads, dim] of consecutive positions (positions[0]: the first) over the keys and values of the
    positions up to their own, which hold them already. keys, values: a ring [rows, kv_heads, dim] indexed by position modulo its rows (window: the positions
    a query reads, itself included), or with table (int32 [pages]: the pages of the sequence, in order) the pages [pages, rows, kv_heads, dim] of a BF16
    cache (window 0: the whole context). rel [tokens, heads, rank] and proj [rank, extent]: their BF16 product is the logit added at each backward distance
    below extent; sinks [tokens, heads]: the logit of the sink; gate [tokens, heads * dim]: the BF16 output times its sigmoid.
    quantized: the pages are a quantized cache's (keys, values: the packed bodies [pages, rows, kv_heads, bytes]): a dict of formats ("k", "v": FORMATS names), "scales" and "mins"
    ((keys', values') tensors [pages, rows, kv_heads, groups], a minimum None for a format without one), and with exact overlays "overlay" and "overlay_scales" ((keys', values')
    int8 [rows, kv_heads, dim] and their scales), "base" and "ring_start" (device tensors: the first overlay row of the sequence or -1, the first position its ring holds),
    "sink" (the positions of the sink pages) and "recent" (the ring's size).
    shared: the tokens of the pass share each key / value tile (one program per KV head and part of the context, parts aligned on the key index): a pass of several tokens
    costs about one of a single token; each token's result is the one it gets alone in this mode (not the bits of shared=False, whose parts follow each token's position)."""
    tokens, heads, dim = q.shape
    kv_heads = keys.shape[-2]
    if tokens > ROWS or not (q.is_contiguous() and rel.is_contiguous() and proj.is_contiguous() and sinks.is_contiguous() and keys.is_contiguous() and values.is_contiguous()):
        raise ValueError("The fused attention takes the contiguous queries of a short pass")
    if not (q.is_cuda and keys.is_cuda and positions.is_cuda):
        raise ValueError("The fused attention runs on CUDA tensors")
    out = torch.empty(tokens, heads * dim, dtype=dtype, device=q.device)
    group = max(16, triton.next_power_of_2(heads // kv_heads))
    steps = STEPS_PAGED if table is not None else STEPS_RING
    reach = window if table is None else table.shape[0] * keys.shape[-3]  # the positions a query can read: its programs beyond its context do nothing
    if shared and table is None:
        steps = STEPS_SHARED_RING
        reach = window + tokens - 1 + steps * CHUNK  # the parts from the first token's first key to the last token: one more where the reach straddles a part
    count = -(-reach // (steps * CHUNK))
    parts = torch.empty(tokens, kv_heads, count, heads // kv_heads, dim + 2, dtype=torch.float32, device=q.device)
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
            raise ValueError("The fused attention reads contiguous cache tensors")
    tiles = _tiles(q.device)
    if tiles is not None:  # (Turing, AMD) the parts of the same positions, by other tiles: steps of fewer keys, the head dimension by slices
        chunks, step, warps, most = tiles
        bits, result = reads_bits(q.device), out
        q, rel, proj, sinks, keys, values, out = (bits_view(tensor, bits) for tensor in (q, rel, proj, sinks, keys, values, out))
        gate = None if gate is None else bits_view(gate, bits)
        common = (q, rel, proj, keys, values, positions if table is None else table, positions, parts, *k_extra, *v_extra, base, ring_start)
        shape = (heads, kv_heads, dim, proj.shape[0], proj.shape[1], step, steps * CHUNK // step)
        sliced = (chunks, bits) if chunks else ()
        if shared:
            rows = min(most, max(16, triton.next_power_of_2(tokens * (heads // kv_heads))))
            band = rows // (heads // kv_heads)
            (_shared_part_sliced if chunks else _shared_part)[(-(-tokens // band), kv_heads, count)](*common, tokens, band, keys.shape[-3], window, scale, sink_rows, ring_mask, *shape, rows, table is not None,
                                                                                                    k_format, v_format, overlay, *sliced, num_warps=warps, enable_fp_fusion=False)
            _shared_merge[(tokens, kv_heads, MERGE_SLICES)](parts, sinks, q if gate is None else gate, positions, out, window, steps * CHUNK, count, heads, kv_heads, dim, triton.next_power_of_2(heads // kv_heads), gate is not None,
                                                            dim // MERGE_SLICES, bits, num_warps=max(1, 4 // MERGE_SLICES), enable_fp_fusion=False)
            return result
        (_attention_part_sliced if chunks else _attention_part)[(tokens, kv_heads, count)](*common, keys.shape[-3], window, scale, sink_rows, ring_mask, *shape, group, table is not None, k_format, v_format, overlay,
                                                                                           *sliced, num_warps=warps, enable_fp_fusion=False)
        _attention_merge[(tokens, kv_heads)](parts, sinks, q if gate is None else gate, positions, out, window, steps * CHUNK, count, heads, kv_heads, dim, group, gate is not None, bits, num_warps=4, enable_fp_fusion=False)
        return result
    if shared:
        rows = min(SHARED_ROWS, max(16, triton.next_power_of_2(tokens * (heads // kv_heads))))  # the query heads of the tokens of a band: 16 or 32 rows of a tensor-core tile
        band = rows // (heads // kv_heads)
        _shared_part[(-(-tokens // band), kv_heads, count)](q, rel, proj, keys, values, positions if table is None else table, positions, parts, *k_extra, *v_extra, base, ring_start, tokens, band, keys.shape[-3], window, scale, sink_rows, ring_mask,
                                        heads, kv_heads, dim, proj.shape[0], proj.shape[1], CHUNK, steps, rows, table is not None, k_format, v_format, overlay, num_warps=WARPS, enable_fp_fusion=False)
        _shared_merge[(tokens, kv_heads, MERGE_SLICES)](parts, sinks, q if gate is None else gate, positions, out, window, steps * CHUNK, count, heads, kv_heads, dim, triton.next_power_of_2(heads // kv_heads), gate is not None, dim // MERGE_SLICES,
                                                        num_warps=max(1, 4 // MERGE_SLICES), enable_fp_fusion=False)
        return out
    _attention_part[(tokens, kv_heads, count)](q, rel, proj, keys, values, positions if table is None else table, positions, parts, *k_extra, *v_extra, base, ring_start, keys.shape[-3], window, scale, sink_rows, ring_mask,
                                               heads, kv_heads, dim, proj.shape[0], proj.shape[1], CHUNK, steps, group, table is not None, k_format, v_format, overlay, num_warps=WARPS, enable_fp_fusion=False)
    _attention_merge[(tokens, kv_heads)](parts, sinks, q if gate is None else gate, positions, out, window, steps * CHUNK, count, heads, kv_heads, dim, group, gate is not None, num_warps=4, enable_fp_fusion=False)
    return out


@triton.jit(do_not_specialize=("SIZE", "WINDOW", "STEPS", "scale"), do_not_specialize_on_alignment=("Q", "KEYS", "VALUES", "START", "LOW", "PARTS"))
def _block_part(Q, KEYS, VALUES, START, LOW, PARTS, SIZE, WINDOW, STEPS, scale, TOKENS: tl.constexpr, HEADS: tl.constexpr, KV: tl.constexpr, D: tl.constexpr, STEP: tl.constexpr, GROUP: tl.constexpr):
    # A part of the keys a block of TOKENS queries reads together (no causality among them, no bias, no sink), for every query row of a KV head (row: token * query heads of a KV head
    # + head): the largest logit, the sum of the weights relative to it and the values weighted so, into PARTS [KV, parts, rows, D + 2]. Part p of the launch: the positions
    # [(first // PART + p) * PART, + PART), PART = STEPS * STEP, first: the first position read. Rounding as _attention_part (BF16 product, its scaling, BF16 weights before the values).
    HPK: tl.constexpr = HEADS // KV
    kv = tl.program_id(0)
    end = tl.load(START).to(tl.int64) + TOKENS
    low = tl.maximum(tl.load(LOW).to(tl.int64), 0)
    if WINDOW > 0:
        low = tl.maximum(low, end - WINDOW)
    span = STEPS * STEP
    part = low // span + tl.program_id(1)
    if part * span < end:
        lanes = tl.arange(0, GROUP)
        real = lanes < TOKENS * HPK
        heads = (lanes // HPK).to(tl.int64) * HEADS + kv * HPK + lanes % HPK
        width = tl.arange(0, D)
        within = tl.arange(0, STEP)
        q = tl.load(Q + heads[:, None] * D + width[None, :], mask=real[:, None], other=0.0)  # [GROUP, D] BF16
        largest = tl.full((GROUP,), float("-inf"), tl.float32)
        total = tl.zeros((GROUP,), tl.float32)
        values = tl.zeros((GROUP, D), tl.float32)
        for step in tl.range(STEPS):  # a runtime loop (unrolled steps spill, as in _attention_part)
            key = part * span + step * STEP + within
            valid = (key >= low) & (key < end)
            record = ((key % SIZE) * KV + kv)[:, None] * D + width[None, :]
            k = tl.load(KEYS + record, mask=valid[:, None], other=0.0)
            score = tl.dot(q, tl.trans(k)).to(tl.bfloat16).to(tl.float32)
            score = (score * scale).to(tl.bfloat16).to(tl.float32)
            score = tl.where(valid[None, :], score, float("-inf"))
            top = tl.maximum(largest, tl.max(score, axis=1))
            decay = tl.where(largest == top, 1.0, libdevice.exp(largest - top))
            weight = tl.where(valid[None, :], libdevice.exp(score - top[:, None]), 0.0)
            total = total * decay + tl.sum(weight, axis=1)
            v = tl.load(VALUES + record, mask=valid[:, None], other=0.0)
            values = values * decay[:, None] + tl.dot(weight.to(tl.bfloat16), v)
            largest = top
        target = PARTS + ((kv * tl.num_programs(1) + tl.program_id(1)) * (TOKENS * HPK) + lanes) * (D + 2)
        tl.store(target[:, None] + width[None, :], values, mask=real[:, None])
        tl.store(target + D, largest, mask=real)
        tl.store(target + D + 1, total, mask=real)


@triton.jit(do_not_specialize=("WINDOW", "SPAN", "COUNT"), do_not_specialize_on_alignment=("PARTS", "START", "LOW", "OUT"))
def _block_merge(PARTS, START, LOW, OUT, WINDOW, SPAN, COUNT, TOKENS: tl.constexpr, HEADS: tl.constexpr, KV: tl.constexpr, D: tl.constexpr, GROUP: tl.constexpr, BITS: tl.constexpr = False):
    # The parts of _block_part of a token and KV head merged in their order (COUNT: the parts of the launch, SPAN: positions per part).
    HPK: tl.constexpr = HEADS // KV
    token = tl.program_id(0).to(tl.int64)
    kv = tl.program_id(1)
    end = tl.load(START).to(tl.int64) + TOKENS
    low = tl.maximum(tl.load(LOW).to(tl.int64), 0)
    if WINDOW > 0:
        low = tl.maximum(low, end - WINDOW)
    lanes = tl.arange(0, GROUP)
    real = lanes < HPK
    width = tl.arange(0, D)
    largest = tl.full((GROUP,), float("-inf"), tl.float32)
    total = tl.zeros((GROUP,), tl.float32)
    values = tl.zeros((GROUP, D), tl.float32)
    for part in tl.range(0, (end - 1) // SPAN - low // SPAN + 1, loop_unroll_factor=8):
        source = PARTS + ((kv * COUNT + part) * (TOKENS * HPK) + token * HPK + lanes) * (D + 2)
        other = tl.load(source + D, mask=real, other=0.0)
        top = tl.maximum(largest, other)
        decay = libdevice.exp(largest - top)
        weight = libdevice.exp(other - top)
        total = total * decay + tl.load(source + D + 1, mask=real, other=0.0) * weight
        values = values * decay[:, None] + tl.load(source[:, None] + width[None, :], mask=real[:, None], other=0.0) * weight[:, None]
        largest = top
    out = tl.div_rn(values, total[:, None])
    if BITS:  # (the BF16 output as its bits)
        tl.store(OUT + (token * HEADS + kv * HPK + lanes)[:, None] * D + width[None, :], fp32_to_bf16_bits(out), mask=real[:, None])
    else:
        tl.store(OUT + (token * HEADS + kv * HPK + lanes)[:, None] * D + width[None, :], out.to(OUT.dtype.element_ty), mask=real[:, None])


BLOCK_PARTS = 64  # parts of a block's reach at most: a part covers more positions as the reach grows, so that the parts' buffer stays [kv_heads, 65, rows, dim + 2]


def block_attend(q, keys, values, start, low, scale, window=0):
    """[tokens, heads * dim] BF16: the queries q [tokens, heads, dim] of a block of consecutive positions from start (a device tensor [1]), all of
    them over the keys and values of every position from low (a device tensor [1], at most start) to the block's last, its own included (no causality
    inside the block, as a drafter's block reads them). keys, values: BF16 [rows, kv_heads, dim], position p at row p % rows (a ring, or a cache by
    position); window: the positions read at most, the block's included (0: from low). Only those positions are read, whatever the rows of the cache."""
    tokens, heads, dim = q.shape
    size, kv_heads = keys.shape[0], keys.shape[1]
    if not (q.is_contiguous() and keys.is_contiguous() and values.is_contiguous() and q.dtype == keys.dtype == values.dtype == torch.bfloat16):
        raise ValueError("The block attention takes contiguous BF16 queries and caches")
    if not (q.is_cuda and keys.is_cuda and values.is_cuda and start.is_cuda and low.is_cuda):
        raise ValueError("The block attention runs on CUDA tensors")
    reach = window or size
    steps = max(STEPS_RING, -(-reach // (CHUNK * BLOCK_PARTS)))
    count = -(-reach // (steps * CHUNK)) + 1  # one more where the reach straddles a part
    rows = tokens * (heads // kv_heads)
    parts = torch.empty(kv_heads, count, rows, dim + 2, dtype=torch.float32, device=q.device)
    out = torch.empty(tokens, heads * dim, dtype=q.dtype, device=q.device)
    tiles = _tiles(q.device)
    if tiles is not None:  # (Turing, AMD) as attend
        chunks, step, warps, _ = tiles
        bits = reads_bits(q.device)
        (_block_part_sliced if chunks else _block_part)[(kv_heads, count)](bits_view(q, bits), bits_view(keys, bits), bits_view(values, bits), start, low, parts, size, window, steps * CHUNK // step, scale, tokens,
                                                                           heads, kv_heads, dim, step, max(16, triton.next_power_of_2(rows)), *((chunks, bits) if chunks else ()), num_warps=warps,
                                                                           enable_fp_fusion=False)
        _block_merge[(tokens, kv_heads)](parts, start, low, bits_view(out, bits), window, steps * CHUNK, count, tokens, heads, kv_heads, dim, max(2, triton.next_power_of_2(heads // kv_heads)), bits,
                                         num_warps=4, enable_fp_fusion=False)
        return out
    _block_part[(kv_heads, count)](q, keys, values, start, low, parts, size, window, steps, scale, tokens, heads, kv_heads, dim, CHUNK, max(16, triton.next_power_of_2(rows)), num_warps=4, enable_fp_fusion=False)
    _block_merge[(tokens, kv_heads)](parts, start, low, out, window, steps * CHUNK, count, tokens, heads, kv_heads, dim, max(2, triton.next_power_of_2(heads // kv_heads)), num_warps=4, enable_fp_fusion=False)
    return out


# Sliced variants of the part kernels (the head dimension in CHUNKS slices of D // CHUNKS columns, whole scale groups): a step dequantizes and
# multiplies one slice of a key / value tile at a time, the scores summed slice by slice and the values kept by slice, so that it holds a slice of
# a tile, not the tile (the full tiles spill on AMD's RDNA and on the float32 products below). BITS (NVIDIA below 8.0, no BF16 tensor cores
# nor BF16 conversions in Triton): the BF16 tensors read as their bits, the BF16 values multiplied in float32 by FMAs (exact products, float32
# sums), every rounding of the tensor-core kernels kept (shared.kernels.bf16_bits).


@triton.jit
def _bf16(x, BITS: tl.constexpr):
    """float32 x rounded to BF16: BF16, or with BITS float32 holding the BF16 value (the products' operands)."""
    if BITS:
        return bf16_bits_to_fp32(fp32_to_bf16_bits(x))
    else:
        return x.to(tl.bfloat16)


@triton.jit
def _rounded(x, BITS: tl.constexpr):
    """float32 x rounded to BF16, as float32."""
    if BITS:
        return bf16_bits_to_fp32(fp32_to_bf16_bits(x))
    else:
        return x.to(tl.bfloat16).to(tl.float32)


@triton.jit
def _operand(x, BITS: tl.constexpr):
    """Loaded BF16 values as the products take them: BF16, or with BITS float32 (from BF16, or from its bits as int16)."""
    if BITS:
        return to_float(x)
    else:
        return x


@triton.jit
def _product(a, b, BITS: tl.constexpr):
    if BITS:
        return tl.dot(a, b, input_precision="ieee")
    else:
        return tl.dot(a, b)


@triton.jit
def _accumulate(a, b, acc, BITS: tl.constexpr):
    """acc + a b (a None accumulator reaches a jit function as a constexpr in Triton 3.2: _product for none)."""
    if BITS:
        return tl.dot(a, b, acc, input_precision="ieee")
    else:
        return tl.dot(a, b, acc)


@triton.jit
def _tile_slice(DATA, SCALE, MIN, OVER, OVER_SCALE, row, exact_row, exact, valid, kv, KV: tl.constexpr, D: tl.constexpr, FORMAT: tl.constexpr, OVERLAY: tl.constexpr,
                SLICE: tl.constexpr, W: tl.constexpr, BITS: tl.constexpr):
    """_tile's columns [SLICE * W, SLICE * W + W) (the per-value loads), the same values: BF16, or with BITS float32 holding them."""
    width = SLICE * W + tl.arange(0, W)
    if FORMAT == 0:
        return _operand(tl.load(DATA + (row * KV + kv)[:, None] * D + width[None, :], mask=valid[:, None], other=0), BITS)
    else:
        GROUPS: tl.constexpr = D // 32
        BYTES: tl.constexpr = 32 if FORMAT == 1 else 24 if FORMAT == 2 else 16
        group, lane = (width // 32)[None, :], (width % 32)[None, :]
        body = valid & (exact.to(tl.int32) == 0) if OVERLAY else valid
        record = (row * KV + kv)[:, None]
        scales = tl.load(SCALE + record * GROUPS + group, mask=body[:, None], other=0.0).to(tl.float32)
        if FORMAT == 1:
            values = tl.load(DATA + record * D + width[None, :], mask=body[:, None], other=0).to(tl.float32) * scales
        else:
            packed = tl.load(DATA + record * (GROUPS * BYTES) + group * BYTES + (lane & 15), mask=body[:, None], other=0).to(tl.int32)
            code = tl.where(lane < 16, packed & 15, packed >> 4)
            if FORMAT == 2:
                plane = tl.load(DATA + record * (GROUPS * BYTES) + group * BYTES + 16 + (lane >> 4) * 4 + (lane & 3), mask=body[:, None], other=0).to(tl.int32)
                code = code | (((plane >> (((lane >> 2) & 3) * 2)) & 3) << 4)
            if FORMAT == 4:
                values = code.to(tl.float32) * scales + tl.load(MIN + record * GROUPS + group, mask=body[:, None], other=0.0).to(tl.float32)
            else:
                values = (code - (32 if FORMAT == 2 else 8)).to(tl.float32) * scales
        if OVERLAY:
            chosen = (valid & (exact.to(tl.int32) != 0))[:, None]
            record = (exact_row * KV + kv)[:, None]
            over = tl.load(OVER + record * D + width[None, :], mask=chosen, other=0).to(tl.float32) * tl.load(OVER_SCALE + record * GROUPS + group, mask=chosen, other=0.0).to(tl.float32)
            values = tl.where(chosen, over, values)
        return _bf16(values, BITS)


@triton.jit
def _q_slice(Q, place, real, SLICE: tl.constexpr, W: tl.constexpr, D: tl.constexpr, BITS: tl.constexpr):
    """Columns [SLICE * W, SLICE * W + W) of the query rows place (token * heads + head) [ROWS, W]."""
    return _operand(tl.load(Q + place[:, None] * D + SLICE * W + tl.arange(0, W)[None, :], mask=real[:, None], other=0), BITS)


@triton.jit
def _keep(values0, values1, values2, values3, decay, weighted, SLICE: tl.constexpr):
    """The running values of slice SLICE rescaled by decay plus its weighted values of a step."""
    if SLICE == 0:
        values0 = values0 * decay[:, None] + weighted
    elif SLICE == 1:
        values1 = values1 * decay[:, None] + weighted
    elif SLICE == 2:
        values2 = values2 * decay[:, None] + weighted
    else:
        values3 = values3 * decay[:, None] + weighted
    return values0, values1, values2, values3


@triton.jit
def _store_slices(target, values0, values1, values2, values3, real, W: tl.constexpr, CHUNKS: tl.constexpr):
    for c in tl.static_range(CHUNKS):
        width = c * W + tl.arange(0, W)
        if c == 0:
            tl.store(target[:, None] + width[None, :], values0, mask=real[:, None])
        elif c == 1:
            tl.store(target[:, None] + width[None, :], values1, mask=real[:, None])
        elif c == 2:
            tl.store(target[:, None] + width[None, :], values2, mask=real[:, None])
        else:
            tl.store(target[:, None] + width[None, :], values3, mask=real[:, None])


@triton.jit
def _step_scores(Q, place, real, KEYS, K_SCALE, K_MIN, K_OVER, K_OVER_SCALE, row, exact_row, exact, valid, kv, scale, KV: tl.constexpr, D: tl.constexpr, K_FORMAT: tl.constexpr,
                 OVERLAY: tl.constexpr, ROWS: tl.constexpr, STEP: tl.constexpr, CHUNKS: tl.constexpr, BITS: tl.constexpr):
    """The rounded, scaled BF16 scores [ROWS, STEP] of a step: the BF16 product of the queries and the keys, slice by slice."""
    W: tl.constexpr = D // CHUNKS
    score = tl.zeros((ROWS, STEP), tl.float32)
    for c in tl.static_range(CHUNKS):
        k = _tile_slice(KEYS, K_SCALE, K_MIN, K_OVER, K_OVER_SCALE, row, exact_row, exact, valid, kv, KV, D, K_FORMAT, OVERLAY, c, W, BITS)
        score = _accumulate(_q_slice(Q, place, real, c, W, D, BITS), tl.trans(k), score, BITS)
    return _rounded(_rounded(score, BITS) * scale, BITS)


@triton.jit
def _step_values(values0, values1, values2, values3, weight, low, decay, VALUES, V_SCALE, V_MIN, V_OVER, V_OVER_SCALE, row, exact_row, exact, valid, kv, KV: tl.constexpr, D: tl.constexpr,
                 V_FORMAT: tl.constexpr, OVERLAY: tl.constexpr, CHUNKS: tl.constexpr, BITS: tl.constexpr, TWO: tl.constexpr = False):
    """The running values of every slice after a step: rescaled by decay, plus the weights times the step's values of the slice (TWO: weight + low,
    two BF16 parts of float32 weights, summed in the product)."""
    W: tl.constexpr = D // CHUNKS
    for c in tl.static_range(CHUNKS):
        v = _tile_slice(VALUES, V_SCALE, V_MIN, V_OVER, V_OVER_SCALE, row, exact_row, exact, valid, kv, KV, D, V_FORMAT, OVERLAY, c, W, BITS)
        if TWO:
            weighted = _accumulate(weight, v, _product(low, v, BITS), BITS)
        else:
            weighted = _product(weight, v, BITS)
        values0, values1, values2, values3 = _keep(values0, values1, values2, values3, decay, weighted, c)
    return values0, values1, values2, values3


@triton.jit(do_not_specialize=("SIZE", "WINDOW", "scale", "SINK_ROWS", "RING_MASK"),
            do_not_specialize_on_alignment=("Q", "REL", "PROJ", "KEYS", "VALUES", "TABLE", "POS", "PARTS", "K_SCALE", "K_MIN", "K_OVER", "K_OVER_SCALE", "V_SCALE", "V_MIN", "V_OVER", "V_OVER_SCALE", "BASE", "RING_START"))
def _attention_part_sliced(Q, REL, PROJ, KEYS, VALUES, TABLE, POS, PARTS, K_SCALE, K_MIN, K_OVER, K_OVER_SCALE, V_SCALE, V_MIN, V_OVER, V_OVER_SCALE, BASE, RING_START, SIZE, WINDOW, scale, SINK_ROWS, RING_MASK,
                           HEADS: tl.constexpr, KV: tl.constexpr, D: tl.constexpr, R: tl.constexpr, EXTENT: tl.constexpr, STEP: tl.constexpr, STEPS: tl.constexpr, GROUP: tl.constexpr, PAGED: tl.constexpr,
                           K_FORMAT: tl.constexpr, V_FORMAT: tl.constexpr, OVERLAY: tl.constexpr, CHUNKS: tl.constexpr, BITS: tl.constexpr):
    # _attention_part by slices of the head dimension (see above)
    W: tl.constexpr = D // CHUNKS
    token = tl.program_id(0).to(tl.int64)
    kv = tl.program_id(1)
    part = tl.program_id(2)
    position = tl.load(POS) + token
    count = position + 1
    if WINDOW > 0:
        count = tl.minimum(count, WINDOW)
    if part * (STEPS * STEP) < count:
        lanes = tl.arange(0, GROUP)
        real = lanes < HEADS // KV
        place = token * HEADS + kv * (HEADS // KV) + lanes
        within = tl.arange(0, STEP)
        rel = _operand(tl.load(REL + place[:, None] * R + tl.arange(0, R)[None, :], mask=real[:, None], other=0), BITS)
        largest = tl.full((GROUP,), float("-inf"), tl.float32)
        total = tl.zeros((GROUP,), tl.float32)
        values0 = tl.zeros((GROUP, W), tl.float32)
        values1 = tl.zeros((GROUP, W), tl.float32)
        values2 = tl.zeros((GROUP, W), tl.float32)
        values3 = tl.zeros((GROUP, W), tl.float32)
        for step in tl.range(STEPS):
            distance = (part * STEPS + step) * STEP + within
            valid = distance < count
            key = position - distance
            if PAGED:
                row = tl.load(TABLE + key // SIZE, mask=valid, other=0).to(tl.int64) * SIZE + key % SIZE
            else:
                row = key % SIZE
            exact, exact_row = valid, row
            if OVERLAY:
                base = tl.load(BASE).to(tl.int64)
                exact = valid & (base >= 0) & ((key < SINK_ROWS) | (key >= tl.load(RING_START)))
                exact_row = tl.maximum(base, 0) + tl.where(key < SINK_ROWS, key, SINK_ROWS + (key & RING_MASK))
            score = _step_scores(Q, place, real, KEYS, K_SCALE, K_MIN, K_OVER, K_OVER_SCALE, row, exact_row, exact, valid, kv, scale, KV, D, K_FORMAT, OVERLAY, GROUP, STEP, CHUNKS, BITS)
            near = valid & (distance < EXTENT)
            proj = _operand(tl.load(PROJ + tl.arange(0, R)[:, None] * EXTENT + distance[None, :], mask=near[None, :], other=0), BITS)
            score = _rounded(score + _rounded(_product(rel, proj, BITS), BITS), BITS)
            score = tl.where(valid[None, :], score, float("-inf"))
            top = tl.maximum(largest, tl.max(score, axis=1))
            decay = tl.where(largest == top, 1.0, libdevice.exp(largest - top))
            weight = tl.where(valid[None, :], libdevice.exp(score - top[:, None]), 0.0)
            total = total * decay + tl.sum(weight, axis=1)
            values0, values1, values2, values3 = _step_values(values0, values1, values2, values3, _bf16(weight, BITS), None, decay, VALUES, V_SCALE, V_MIN, V_OVER, V_OVER_SCALE, row, exact_row, exact,
                                                              valid, kv, KV, D, V_FORMAT, OVERLAY, CHUNKS, BITS)
            largest = top
        target = PARTS + (((token * KV + kv) * tl.num_programs(2) + part) * (HEADS // KV) + lanes) * (D + 2)
        _store_slices(target, values0, values1, values2, values3, real, W, CHUNKS)
        tl.store(target + D, largest, mask=real)
        tl.store(target + D + 1, total, mask=real)


@triton.jit(do_not_specialize=("TOKENS", "BAND_TOKENS", "SIZE", "WINDOW", "scale", "SINK_ROWS", "RING_MASK"),
            do_not_specialize_on_alignment=("Q", "REL", "PROJ", "TABLE", "POS", "PARTS", "BASE", "RING_START"))
def _shared_part_sliced(Q, REL, PROJ, KEYS, VALUES, TABLE, POS, PARTS, K_SCALE, K_MIN, K_OVER, K_OVER_SCALE, V_SCALE, V_MIN, V_OVER, V_OVER_SCALE, BASE, RING_START, TOKENS, BAND_TOKENS, SIZE, WINDOW, scale, SINK_ROWS,
                        RING_MASK, HEADS: tl.constexpr, KV: tl.constexpr, D: tl.constexpr, R: tl.constexpr, EXTENT: tl.constexpr, STEP: tl.constexpr, STEPS: tl.constexpr, GROUP: tl.constexpr,
                        PAGED: tl.constexpr, K_FORMAT: tl.constexpr, V_FORMAT: tl.constexpr, OVERLAY: tl.constexpr, CHUNKS: tl.constexpr, BITS: tl.constexpr):
    # _shared_part by slices of the head dimension (see above): a token's rows still computed as in a pass of that token alone
    HPK: tl.constexpr = HEADS // KV
    PART: tl.constexpr = STEPS * STEP
    W: tl.constexpr = D // CHUNKS
    band = tl.program_id(0)
    kv = tl.program_id(1)
    origin = tl.load(POS).to(tl.int64)
    first = origin + band * BAND_TOKENS
    count = tl.minimum(TOKENS - band * BAND_TOKENS, BAND_TOKENS)
    last = first + count - 1
    low = tl.where(WINDOW > 0, tl.maximum(first - WINDOW + 1, 0), 0)
    part = tl.where(WINDOW > 0, tl.maximum(origin - WINDOW + 1, 0), 0) // PART + tl.program_id(2)
    if (part * PART <= last) & ((part + 1) * PART > low):
        lanes = tl.arange(0, GROUP)
        local = lanes // HPK
        token = (band * BAND_TOKENS + local).to(tl.int64)
        real = local < count
        place = token * HEADS + kv * HPK + lanes % HPK
        within = tl.arange(0, STEP)
        rel = _operand(tl.load(REL + place[:, None] * R + tl.arange(0, R)[None, :], mask=real[:, None], other=0), BITS)
        largest = tl.full((GROUP,), float("-inf"), tl.float32)
        total = tl.zeros((GROUP,), tl.float32)
        values0 = tl.zeros((GROUP, W), tl.float32)
        values1 = tl.zeros((GROUP, W), tl.float32)
        values2 = tl.zeros((GROUP, W), tl.float32)
        values3 = tl.zeros((GROUP, W), tl.float32)
        for step in tl.range(STEPS):
            start = part * PART + step * STEP
            key = start + within
            valid = (key >= low) & (key <= last)
            if PAGED:
                row = tl.load(TABLE + key // SIZE, mask=valid, other=0).to(tl.int64) * SIZE + key % SIZE
            else:
                row = key % SIZE
            exact, exact_row = valid, row
            if OVERLAY:
                base = tl.load(BASE).to(tl.int64)
                exact = valid & (base >= 0) & ((key < SINK_ROWS) | (key >= tl.load(RING_START)))
                exact_row = tl.maximum(base, 0) + tl.where(key < SINK_ROWS, key, SINK_ROWS + (key & RING_MASK))
            score = _step_scores(Q, place, real, KEYS, K_SCALE, K_MIN, K_OVER, K_OVER_SCALE, row, exact_row, exact, valid, kv, scale, KV, D, K_FORMAT, OVERLAY, GROUP, STEP, CHUNKS, BITS)
            distance = local[:, None] - (key - first).to(tl.int32)[None, :]
            allowed = real[:, None] & (distance >= 0) & ((WINDOW == 0) | (distance < WINDOW))
            bias = tl.zeros((GROUP, STEP), tl.float32)
            if first - (start + STEP - 1) < EXTENT:
                for t in range(count):
                    gap = first + t - key
                    near = (gap >= 0) & (gap < EXTENT)
                    proj = _operand(tl.load(PROJ + tl.arange(0, R)[:, None] * EXTENT + gap[None, :], mask=near[None, :], other=0), BITS)
                    bias = tl.where((local == t)[:, None], _rounded(_product(rel, proj, BITS), BITS), bias)
            score = _rounded(score + bias, BITS)
            score = tl.where(allowed, score, float("-inf"))
            top = tl.maximum(largest, tl.max(score, axis=1))
            decay = tl.where(largest == top, 1.0, libdevice.exp(largest - top))
            weight = _bf16(tl.where(allowed, libdevice.exp(score - top[:, None]), 0.0), BITS)
            if BITS:
                ones = tl.where(tl.arange(0, 16)[None, :] == 0, 1.0, 0.0) + tl.zeros((STEP, 16), tl.float32)
            else:
                ones = tl.where(tl.arange(0, 16)[None, :] == 0, 1.0, 0.0).to(tl.bfloat16) + tl.zeros((STEP, 16), tl.bfloat16)
            sums = total[:, None] * decay[:, None] + _product(weight, ones, BITS)
            total = tl.sum(tl.where(tl.arange(0, 16)[None, :] == 0, sums, 0.0), axis=1)
            values0, values1, values2, values3 = _step_values(values0, values1, values2, values3, weight, None, decay, VALUES, V_SCALE, V_MIN, V_OVER, V_OVER_SCALE, row, exact_row, exact, valid, kv, KV, D,
                                                              V_FORMAT, OVERLAY, CHUNKS, BITS)
            largest = top
        target = PARTS + (((token * KV + kv) * tl.num_programs(2) + tl.program_id(2)) * HPK + lanes % HPK) * (D + 2)
        _store_slices(target, values0, values1, values2, values3, real, W, CHUNKS)
        tl.store(target + D, largest, mask=real)
        tl.store(target + D + 1, total, mask=real)


@triton.jit(do_not_specialize=("SIZE", "WINDOW", "STEPS", "scale"), do_not_specialize_on_alignment=("Q", "KEYS", "VALUES", "START", "LOW", "PARTS"))
def _block_part_sliced(Q, KEYS, VALUES, START, LOW, PARTS, SIZE, WINDOW, STEPS, scale, TOKENS: tl.constexpr, HEADS: tl.constexpr, KV: tl.constexpr, D: tl.constexpr, STEP: tl.constexpr,
                       GROUP: tl.constexpr, CHUNKS: tl.constexpr, BITS: tl.constexpr):
    # _block_part by slices of the head dimension (see above)
    HPK: tl.constexpr = HEADS // KV
    W: tl.constexpr = D // CHUNKS
    kv = tl.program_id(0)
    end = tl.load(START).to(tl.int64) + TOKENS
    low = tl.maximum(tl.load(LOW).to(tl.int64), 0)
    if WINDOW > 0:
        low = tl.maximum(low, end - WINDOW)
    span = STEPS * STEP
    part = low // span + tl.program_id(1)
    if part * span < end:
        lanes = tl.arange(0, GROUP)
        real = lanes < TOKENS * HPK
        place = (lanes // HPK).to(tl.int64) * HEADS + kv * HPK + lanes % HPK
        within = tl.arange(0, STEP)
        largest = tl.full((GROUP,), float("-inf"), tl.float32)
        total = tl.zeros((GROUP,), tl.float32)
        values0 = tl.zeros((GROUP, W), tl.float32)
        values1 = tl.zeros((GROUP, W), tl.float32)
        values2 = tl.zeros((GROUP, W), tl.float32)
        values3 = tl.zeros((GROUP, W), tl.float32)
        for step in tl.range(STEPS):
            key = part * span + step * STEP + within
            valid = (key >= low) & (key < end)
            row = key % SIZE
            score = _step_scores(Q, place, real, KEYS, KEYS, KEYS, KEYS, KEYS, row, row, valid, valid, kv, scale, KV, D, 0, False, GROUP, STEP, CHUNKS, BITS)
            score = tl.where(valid[None, :], score, float("-inf"))
            top = tl.maximum(largest, tl.max(score, axis=1))
            decay = tl.where(largest == top, 1.0, libdevice.exp(largest - top))
            weight = tl.where(valid[None, :], libdevice.exp(score - top[:, None]), 0.0)
            total = total * decay + tl.sum(weight, axis=1)
            values0, values1, values2, values3 = _step_values(values0, values1, values2, values3, _bf16(weight, BITS), None, decay, VALUES, VALUES, VALUES, VALUES, VALUES, row, row, valid, valid, kv, KV, D,
                                                              0, False, CHUNKS, BITS)
            largest = top
        target = PARTS + ((kv * tl.num_programs(1) + tl.program_id(1)) * (TOKENS * HPK) + lanes) * (D + 2)
        _store_slices(target, values0, values1, values2, values3, real, W, CHUNKS)
        tl.store(target + D, largest, mask=real)
        tl.store(target + D + 1, total, mask=real)


from shared.kernels.triton_compilation_log import install_triton_compilation_logger  # noqa: E402

install_triton_compilation_logger()
