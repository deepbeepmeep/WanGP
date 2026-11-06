# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
# Fused kernels of indexed sparse attention (lmgp.layers.attention_indexed), with ideas of Strata (MIT License, https://github.com/Niko1221/Strata): the RMS norm and rotation of a
# projection in one launch, the block scores in one scan of the indexer's pages, the selection by radix in a kernel, attention over the selected blocks in FP32 and the gate in its
# reduction. The launches do not depend on the context or the number of queries: a capture holds for any position.
"""Selection is deterministic: the largest scores, ties to the lowest block, in ascending block order. A query attends to its selected blocks and to the incomplete block at its end
(the last entry of its list), so that the tokens of a selection are ascending: the sink, the body and the ring of a mixed KV layout (kv_formats) each hold a contiguous range of it."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from shared.kernels import dispatch
from shared.llm_engines.lmgp.layers import attention_mixed

PROJECT_ROWS, PROJECT_COLUMNS, PROJECT_WARPS = 4, 512, 4  # the output rows and input columns of a program of the indexer's projections
PROJECT_MAX_TOKENS = 16  # longer passes (prompt processing) keep the linears
SELECT_CHUNK = 1024  # scores a selection pass reads at a time
SELECT_WARPS = 8
SELECT_SAMPLE = 32  # one score in so many bounds the threshold of a long context
LIST_FACTOR = 8  # the scores above the bound a selection can keep, in budgets
TILE_TOKENS = 32  # tokens of an attention tile
MAX_CHUNKS = 64  # attention programs per query and KV head, at most: one tile each while the budget allows
PRECISION = "tf32x3"  # FP32 operands on the tensor cores: keys, values and codes are exact in TF32; measured no less precise than the FP32 BLAS of the PyTorch path
SPLIT_BF16 = "bf16x2"  # prompts, approximate kernels allowed: queries and softmax weights as BF16 pairs (value, remainder), codes exact in BF16: a sixth of the products of tf32x3


@triton.jit
def _project_pair_kernel(X, W_A, W_B, OUT_A, OUT_B, ROWS_A: tl.constexpr, ROWS_B: tl.constexpr, WIDTH: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    """M output rows of one of two projections of a token, the rows of the second after those of the first: float32 activation and accumulation."""
    group, token = tl.program_id(0), tl.program_id(1)
    first = group * M < ROWS_A
    rows = tl.where(first, group * M, group * M - ROWS_A) + tl.arange(0, M)
    valid = tl.where(first, rows < ROWS_A, rows < ROWS_B)
    weight = tl.where(first, W_A, W_B)
    total = tl.zeros((M,), tl.float32)
    for start in range(0, WIDTH, N):
        columns = start + tl.arange(0, N)
        x = tl.load(X + token * WIDTH + columns).to(tl.float32)
        total += tl.sum(tl.load(weight + rows[:, None] * WIDTH + columns[None, :], mask=valid[:, None], other=0.0).to(tl.float32) * x[None, :], 1)
    tl.store(tl.where(first, OUT_A + token * ROWS_A + rows, OUT_B + token * ROWS_B + rows), total, mask=valid)


def project_pair(x, weight_a, weight_b):
    """Float32 [tokens, rows] of two projections of x [tokens, width] with their weights [rows, width] in VRAM, in one launch, for a few tokens: no copy of x per projection and no rounding of the output."""
    tokens, width = x.shape
    rows_a, rows_b = weight_a.shape[0], weight_b.shape[0]
    assert x.is_cuda and weight_a.is_cuda and weight_b.is_cuda and rows_a % PROJECT_ROWS == 0 and width % PROJECT_COLUMNS == 0
    out_a = torch.empty(tokens, rows_a, dtype=torch.float32, device=x.device)
    out_b = torch.empty(tokens, rows_b, dtype=torch.float32, device=x.device)
    _project_pair_kernel[(triton.cdiv(rows_a + rows_b, PROJECT_ROWS), tokens)](x, weight_a, weight_b, out_a, out_b, rows_a, rows_b, width, PROJECT_ROWS, PROJECT_COLUMNS, num_warps=PROJECT_WARPS)
    return out_a, out_b


@triton.jit(do_not_specialize=("IN_ROW", "IN_HEAD", "OUT_ROW", "OUT_HEAD", "ROT_ROW"), do_not_specialize_on_alignment=("IN_ROW", "IN_HEAD", "OUT_ROW", "OUT_HEAD", "ROT_ROW"))
def _norm_rope_kernel(X, OUT, WEIGHT, COS, SIN, IN_ROW, IN_HEAD, OUT_ROW, OUT_HEAD, ROT_ROW, EPS: tl.constexpr, D: tl.constexpr, ROTARY: tl.constexpr):
    """x [rows, heads, >= D] into float32 out [rows, heads, D]: per row and head the RMS norm over D values, the weights, the rotation of the first ROTARY values."""
    row, head = tl.program_id(0), tl.program_id(1)
    dims = tl.arange(0, D)
    source = X + row * IN_ROW + head * IN_HEAD
    x = tl.load(source + dims).to(tl.float32)
    wide = x.to(tl.float64)  # the factor of a whole row: its statistics in FP64, as the error of the sum would reach every value
    factor = tl.div_rn(1.0, tl.sqrt_rn((tl.sum(wide * wide, 0) * (1.0 / D) + EPS).to(tl.float32)))
    half: tl.constexpr = ROTARY // 2
    turning = dims < ROTARY
    first = dims < half
    pair = tl.where(first, dims + half, dims - half)
    partner = tl.load(source + pair, mask=turning, other=0.0).to(tl.float32) * factor * tl.load(WEIGHT + pair, mask=turning, other=0.0).to(tl.float32)
    cos = tl.load(COS + row * ROT_ROW + dims, mask=turning, other=1.0)
    sin = tl.load(SIN + row * ROT_ROW + dims, mask=turning, other=0.0)
    x = x * factor * tl.load(WEIGHT + dims).to(tl.float32)
    tl.store(OUT + row * OUT_ROW + head * OUT_HEAD + dims, x * cos + tl.where(first, -partner, partner) * sin)


def norm_rope(x, weight, cos, sin, eps, heads, dim, stride_dim=None):
    """float32 [rows, heads, dim] from x [rows, heads, stride_dim or dim] (contiguous, in a float dtype Triton reads): RMS norm per head, weights, rotation in one launch. cos, sin [rows, rotary]."""
    rows, stride_dim = x.shape[0], stride_dim or dim
    assert x.is_cuda and weight.is_cuda and cos.is_cuda and sin.is_cuda
    out = torch.empty(rows, heads, dim, dtype=torch.float32, device=x.device)
    _norm_rope_kernel[(rows, heads)](x, out, weight, cos, sin, heads * stride_dim, stride_dim, heads * dim, dim, cos.shape[1], eps, dim, cos.shape[1], num_warps=2, num_stages=1)
    return out


@triton.jit(do_not_specialize=("TABLE_WIDTH",), do_not_specialize_on_alignment=("TABLE_WIDTH",))
def _append_kernel(RAW, COS, SIN, SEQUENCE, TABLE, RAW_RING, COS_RING, SIN_RING, POOL, WEIGHT, TABLE_WIDTH, EPS: tl.constexpr, D: tl.constexpr, ROTARY: tl.constexpr, RATIO: tl.constexpr, PER_PAGE: tl.constexpr,
                   RING: tl.constexpr, WRITE_RINGS: tl.constexpr):
    """One program per token of the pass: the pooled key of the block it completes into the paged cache, its rows into the rings. A member before the pass is read from the rings, which
    the programs of this launch do not overwrite: the tokens of the pass and the RATIO - 1 before them fit in the rings, or WRITE_RINGS is off."""
    row = tl.program_id(0)
    position = tl.load(SEQUENCE + row).to(tl.int32)
    dims = tl.arange(0, D)
    block = position // RATIO
    page_index = block // PER_PAGE
    if ((position + 1) % RATIO == 0) & (page_index < TABLE_WIDTH):
        page = tl.load(TABLE + page_index)
        if page >= 0:
            half: tl.constexpr = ROTARY // 2
            turning = dims < ROTARY
            first = dims < half
            pair = tl.where(first, dims + half, dims - half)
            total = tl.zeros((D,), tl.float32)
            other = tl.zeros((D,), tl.float32)
            for back in tl.static_range(RATIO):
                member = row - back
                if member >= 0:
                    total += tl.load(RAW + member * D + dims).to(tl.float32)
                    other += tl.load(RAW + member * D + pair, mask=turning, other=0.0).to(tl.float32)
                else:
                    total += tl.load(RAW_RING + ((position - back) % RING) * D + dims)
                    other += tl.load(RAW_RING + ((position - back) % RING) * D + pair, mask=turning, other=0.0)
            start = row - (RATIO - 1)
            if start >= 0:
                cos = tl.load(COS + start * ROTARY + dims, mask=turning, other=1.0)
                sin = tl.load(SIN + start * ROTARY + dims, mask=turning, other=0.0)
            else:
                ring = (position - (RATIO - 1)) % RING
                cos = tl.load(COS_RING + ring * ROTARY + dims, mask=turning, other=1.0)
                sin = tl.load(SIN_RING + ring * ROTARY + dims, mask=turning, other=0.0)
            total = total * (1.0 / RATIO)
            factor = tl.div_rn(1.0, tl.sqrt_rn(tl.sum(total * total, 0) * (1.0 / D) + EPS))
            other = other * (1.0 / RATIO) * factor * tl.load(WEIGHT + pair, mask=turning, other=0.0).to(tl.float32)
            total = total * factor * tl.load(WEIGHT + dims).to(tl.float32)
            pooled = total * cos + tl.where(first, -other, other) * sin
            tl.store(POOL + (page * PER_PAGE + block % PER_PAGE) * D + dims, pooled.to(POOL.dtype.element_ty))
    if WRITE_RINGS:
        ring = position % RING
        turns = tl.arange(0, ROTARY)
        tl.store(RAW_RING + ring * D + dims, tl.load(RAW + row * D + dims).to(tl.float32))
        tl.store(COS_RING + ring * ROTARY + turns, tl.load(COS + row * ROTARY + turns))
        tl.store(SIN_RING + ring * ROTARY + turns, tl.load(SIN + row * ROTARY + turns))


def append_index(raw, cos, sin, sequence, table, raw_ring, cos_ring, sin_ring, pool, weight, ratio, eps):
    """raw [tokens, dim] in a float dtype Triton reads; the rows of the pass into the rings, the pooled keys of the blocks it completes into pool [pages, page // ratio, dim] (table: the pages of the sequence)."""
    tokens, dim = raw.shape
    assert raw.is_cuda and cos.is_cuda and sin.is_cuda and sequence.is_cuda and table.is_cuda and raw_ring.is_cuda and pool.is_cuda and weight.is_cuda
    rotary, ring = cos.shape[1], raw_ring.shape[0]
    fits = tokens + ratio - 1 <= ring
    _append_kernel[(tokens,)](raw, cos, sin, sequence, table, raw_ring, cos_ring, sin_ring, pool, weight, table.shape[0], eps, dim, rotary, ratio, pool.shape[1], ring, fits, num_warps=2, num_stages=1)
    if not fits:  # a long pass: the end of it, after the programs that read the rings
        rows = sequence[-ring:] % ring
        raw_ring[rows] = raw[-ring:].float()
        cos_ring[rows] = cos[-ring:]
        sin_ring[rows] = sin[-ring:]
    torch.autograd.graph.increment_version((raw_ring, cos_ring, sin_ring, pool))


@triton.jit(do_not_specialize=("STRIDE",), do_not_specialize_on_alignment=("STRIDE",))
def _scores_kernel(Q, POOL, TABLE, SEQUENCE, SCORES, STRIDE, H: tl.constexpr, D: tl.constexpr, PER_PAGE: tl.constexpr, RATIO: tl.constexpr, PRECISION: tl.constexpr):
    """The scores of the blocks of a page for a query, written by block: the relus of the head products, summed (the indexer's scale changes no order)."""
    query, index = tl.program_id(0), tl.program_id(1)
    if index * PER_PAGE < (tl.load(SEQUENCE + query).to(tl.int32) + 1) // RATIO:
        page = tl.load(TABLE + index)
        blocks = tl.arange(0, PER_PAGE)
        if page >= 0:
            rows = tl.arange(0, H)
            dims = tl.arange(0, D)
            q = tl.load(Q + (query * H + rows[:, None]) * D + dims[None, :])
            keys = tl.load(POOL + (page * PER_PAGE + blocks[None, :]) * D + dims[:, None]).to(tl.float32)
            scores = tl.sum(tl.maximum(tl.dot(q, keys, input_precision=PRECISION), 0.0), 0)
        else:
            scores = tl.zeros((PER_PAGE,), tl.float32)
        tl.store(SCORES + query * STRIDE + index * PER_PAGE + blocks, scores)


@triton.jit
def _radix_digit(source, stride, count, threshold, above, BUDGET, SHIFT: tl.constexpr, WIDTH: tl.constexpr, C: tl.constexpr):
    """The next WIDTH bits of the threshold (the largest value whose count of keys at or above it reaches BUDGET), and the count of keys above the digits fixed so far."""
    BINS: tl.constexpr = 1 << WIDTH
    bins = tl.arange(0, BINS)
    total = tl.zeros((BINS,), tl.int32)
    for start in tl.range(0, count, C):
        index = start + tl.arange(0, C)
        valid = index < count
        keys = tl.load(source + index * stride, mask=valid, other=0) & 0x7fffffff
        total += tl.histogram((keys >> SHIFT) & (BINS - 1), BINS, mask=valid & (keys >> (SHIFT + WIDTH) == threshold >> (SHIFT + WIDTH)))
    digit = tl.max(tl.where(above + tl.cumsum(total, axis=0, reverse=True) >= BUDGET, bins, 0), axis=0)
    return threshold | (digit << SHIFT), above + tl.sum(tl.where(bins > digit, total, 0), axis=0)


@triton.jit(do_not_specialize=("STRIDE", "BUDGET", "ENTRIES", "SINK", "CAP"), do_not_specialize_on_alignment=("STRIDE", "BUDGET", "ENTRIES", "SINK", "CAP"))
def _select_kernel(SCORES, SEQUENCE, RING_START, IDS, COUNTS, LIST_KEYS, LIST_IDS, STRIDE, BUDGET, ENTRIES, SINK, CAP, RATIO: tl.constexpr, C: tl.constexpr, SAMPLE: tl.constexpr):
    """Per query, the list of its best BUDGET complete blocks in ascending order and its incomplete block, and three counts: the entries of the list, those in the sink, and those before the
    ring. SCORES as int32: their bits order them, being sums of relus (a negative zero is a zero). Radix selection of the threshold: the exponent, then the mantissa. Beyond a few chunks of
    scores, every SAMPLE-th score bounds the threshold from below, and only the scores above the bound (about 3 BUDGET, whatever the context) are read again; the rare query whose bound was
    wrong reads all of them. As Strata's one-read selection."""
    query = tl.program_id(0)
    position = tl.load(SEQUENCE + query).to(tl.int32)
    visible = (position + 1) // RATIO
    scores = SCORES + query * STRIDE
    entries = IDS + query * ENTRIES
    listed_keys = LIST_KEYS + query * CAP
    listed_ids = LIST_IDS + query * CAP
    lanes = tl.arange(0, C)
    source = scores
    count = visible
    from_list = False
    threshold = -1
    ties = 0
    if visible > BUDGET:
        need = 2 * (BUDGET // SAMPLE) + 16
        samples = tl.cdiv(visible, SAMPLE)
        if (visible > 2 * C) & (samples > need):
            bound, above = _radix_digit(scores, SAMPLE, samples, 0, 0, need, 23, 8, C)
            bound, above = _radix_digit(scores, SAMPLE, samples, bound, above, need, 15, 8, C)
            listed = 0
            for start in tl.range(0, visible, C):
                index = start + lanes
                valid = index < visible
                keys = tl.load(scores + index, mask=valid, other=0) & 0x7fffffff
                above_bound = valid & (keys >= bound)
                flags = above_bound.to(tl.int32)
                target = listed + tl.cumsum(flags, axis=0) - 1
                kept = above_bound & (target < CAP)
                tl.store(listed_keys + target, keys, mask=kept)
                tl.store(listed_ids + target, index, mask=kept)
                listed += tl.sum(flags, axis=0)
            if (listed >= BUDGET) & (listed <= CAP):
                source = listed_keys
                count = listed
                from_list = True
        threshold, above = _radix_digit(source, 1, count, 0, 0, BUDGET, 23, 8, C)
        threshold, above = _radix_digit(source, 1, count, threshold, above, BUDGET, 15, 8, C)
        threshold, above = _radix_digit(source, 1, count, threshold, above, BUDGET, 7, 8, C)
        threshold, above = _radix_digit(source, 1, count, threshold, above, BUDGET, 0, 7, C)
        ties = BUDGET - above
    written = 0
    seen = 0
    sink = 0
    ring_start = tl.load(RING_START)
    before_ring = 0
    for start in tl.range(0, count, C):
        index = start + lanes
        valid = index < count
        keys = tl.load(source + index, mask=valid, other=0) & 0x7fffffff
        blocks = tl.where(from_list, tl.load(listed_ids + index, mask=valid & from_list, other=0), index)
        tie = valid & (keys == threshold)
        chosen = (valid & (keys > threshold)) | (tie & (seen + tl.cumsum(tie.to(tl.int32), axis=0) <= ties))
        selected = chosen.to(tl.int32)
        tl.store(entries + written + tl.cumsum(selected, axis=0) - 1, blocks, mask=chosen)
        written += tl.sum(selected, axis=0)
        seen += tl.sum(tie.to(tl.int32), axis=0)
        sink += tl.sum(tl.where(blocks * RATIO < SINK, selected, 0), axis=0)
        before_ring += tl.sum(tl.where(blocks * RATIO < ring_start, selected, 0), axis=0)
    if visible * RATIO <= position:  # the incomplete block at the end of the query
        tl.store(entries + written, visible)
        written += 1
        if visible * RATIO < SINK:
            sink += 1
        if visible * RATIO < ring_start:
            before_ring += 1
    tl.store(COUNTS + query * 3, written)
    tl.store(COUNTS + query * 3 + 1, sink)
    tl.store(COUNTS + query * 3 + 2, before_ring)


def select_blocks(q, pool, table, sequence, budget, ratio, ring_start=None, sink=0, precision=PRECISION):
    """(ids [queries, budget + 1] int32, counts [queries, 3] int32): the selection of each query q [queries, heads, dim] float32 at positions sequence [queries] over the blocks of pool through table, as
    _select_kernel writes it. ring_start: one int32 for the sequence, or None without overlay."""
    queries, heads, dim = q.shape
    assert q.is_cuda and pool.is_cuda and table.is_cuda and sequence.is_cuda
    device, per_page = q.device, pool.shape[1]
    stride = table.shape[0] * per_page
    scores = torch.empty(queries, stride, dtype=torch.float32, device=device)
    _scores_kernel[(queries, table.shape[0])](q, pool, table, sequence, scores, stride, heads, dim, per_page, ratio, precision, num_warps=4, num_stages=1)
    ids = torch.empty(queries, budget + 1, dtype=torch.int32, device=device)
    counts = torch.empty(queries, 3, dtype=torch.int32, device=device)
    ring = torch.full((1,), 1 << 30, dtype=torch.int32, device=device) if ring_start is None else ring_start
    capacity = max(LIST_FACTOR * budget, SELECT_CHUNK)
    listed_keys, listed_ids = torch.empty(2, queries, capacity, dtype=torch.int32, device=device) if stride > 2 * SELECT_CHUNK else (ids, ids)  # a table this short reads the scores only
    _select_kernel[(queries,)](scores.view(torch.int32), sequence, ring, ids, counts, listed_keys, listed_ids, stride, budget, budget + 1, sink, capacity, ratio, SELECT_CHUNK, SELECT_SAMPLE, num_warps=SELECT_WARPS, num_stages=1)
    return ids, counts


@triton.jit
def _codes(data_ptr, records, mask, D: tl.constexpr, BITS: tl.constexpr, ASYM: tl.constexpr, TRANSPOSED: tl.constexpr, SPLIT: tl.constexpr):
    """Both lane halves of every group, in FP32, or BF16 with SPLIT: the codes of a quantized record (centred), the values of a BF16 one. [groups, 16, N] for keys (TRANSPOSED), [groups, N, 16] for values."""
    groups = tl.arange(0, D // 32)
    lanes = tl.arange(0, 16)
    lane0 = tl.arange(0, 1)
    if TRANSPOSED:
        if BITS == 16:
            base = data_ptr + records[None, None, :] + groups[:, None, None] * 32 + lanes[None, :, None]
            a, b = tl.load(base, mask=mask[None, None, :], other=0.0), tl.load(base + 16, mask=mask[None, None, :], other=0.0)
        else:
            a, b = attention_mixed._group_codes(data_ptr, records[None, None, :], groups[:, None, None], lanes[None, :, None], lane0[None, :, None], mask[None, None, :], mask[None, None, :], BITS, ASYM)
    else:
        if BITS == 16:
            base = data_ptr + records[None, :, None] + groups[:, None, None] * 32 + lanes[None, None, :]
            a, b = tl.load(base, mask=mask[None, :, None], other=0.0), tl.load(base + 16, mask=mask[None, :, None], other=0.0)
        else:
            a, b = attention_mixed._group_codes(data_ptr, records[None, :, None], groups[:, None, None], lanes[None, None, :], lane0[None, None, :], mask[None, :, None], mask[None, :, None], BITS, ASYM)
    if ASYM:  # centred: a sum of codes and the minimum would cancel, and round more
        a, b = a - 8, b - 8
    a, b = a.to(tl.float32), b.to(tl.float32)
    if SPLIT:
        a, b = a.to(tl.bfloat16), b.to(tl.bfloat16)
    return a, b


@triton.jit
def _tile(q_a, q_b, q_a_rest, q_b_rest, q_sum, k_ptr, v_ptr, ks_ptr, vs_ptr, km_ptr, vm_ptr, k_records, v_records, scale_records, valid, maximum, denominator, acc_a, acc_b, D: tl.constexpr, SCALE: tl.constexpr,
          K_BITS: tl.constexpr, K_ASYM: tl.constexpr, V_BITS: tl.constexpr, V_ASYM: tl.constexpr, PRECISION: tl.constexpr, SPLIT: tl.constexpr):
    """One tile of tokens: the online softmax of the queries of a KV head over them, in FP32. The products of a group of 32 head dimensions are summed apart, then the groups: a smaller rounding error on
    the scores than one sum over the head, as the BLAS of the PyTorch path; the scales and minimums of a group apply once to its sum."""
    groups = tl.arange(0, D // 32)
    k_a, k_b = _codes(k_ptr, k_records, valid, D, K_BITS, K_ASYM, True, SPLIT)
    if SPLIT:
        dots = tl.dot(q_a, k_a)
        dots = tl.dot(q_a_rest, k_a, dots)
        dots = tl.dot(q_b, k_b, dots)
        dots = tl.dot(q_b_rest, k_b, dots)
    else:
        dots = tl.dot(q_a, k_a, input_precision=PRECISION)
        dots = tl.dot(q_b, k_b, dots, input_precision=PRECISION)
    if K_BITS != 16:
        k_scale = tl.load(ks_ptr + scale_records[None, :] + groups[:, None], mask=valid[None, :], other=0.0).to(tl.float32)
        dots = dots * k_scale[:, None, :]
        if K_ASYM:
            k_min = tl.load(km_ptr + scale_records[None, :] + groups[:, None], mask=valid[None, :], other=0.0).to(tl.float32)
            dots = dots + q_sum[:, :, None] * (k_min + 8.0 * k_scale)[:, None, :]
    scores = tl.sum(dots, axis=0) * SCALE
    scores = tl.where(valid[None, :], scores, -1.0e20)
    new_maximum = tl.maximum(maximum, tl.max(scores, 1))
    correction = libdevice.exp(maximum - new_maximum)
    p = tl.where(valid[None, :], libdevice.exp(scores - new_maximum[:, None]), 0.0)
    denominator = denominator * correction + tl.sum(p, 1)
    v_a, v_b = _codes(v_ptr, v_records, valid, D, V_BITS, V_ASYM, False, SPLIT)
    acc_a = acc_a * correction[None, :, None]
    acc_b = acc_b * correction[None, :, None]
    if V_BITS != 16:
        v_scale = tl.load(vs_ptr + scale_records[None, :] + groups[:, None], mask=valid[None, :], other=0.0).to(tl.float32)
        weighted = p[None, :, :] * v_scale[:, None, :]
    else:
        weighted = tl.broadcast_to(p[None, :, :], (D // 32, p.shape[0], p.shape[1]))
    if SPLIT:
        high = weighted.to(tl.bfloat16)
        rest = (weighted - high.to(tl.float32)).to(tl.bfloat16)
        acc_a = tl.dot(rest, v_a, tl.dot(high, v_a, acc_a))
        acc_b = tl.dot(rest, v_b, tl.dot(high, v_b, acc_b))
    else:
        acc_a = tl.dot(weighted, v_a, acc_a, input_precision=PRECISION)
        acc_b = tl.dot(weighted, v_b, acc_b, input_precision=PRECISION)
    if V_ASYM:
        v_min = tl.load(vm_ptr + scale_records[None, :] + groups[:, None], mask=valid[None, :], other=0.0).to(tl.float32)
        minimum = tl.sum(p[None, :, :] * (v_min + 8.0 * v_scale)[:, None, :], axis=2)[:, :, None]
        acc_a += minimum
        acc_b += minimum
    return new_maximum, denominator, acc_a, acc_b


@triton.jit(do_not_specialize=("ENTRIES", "TABLE_WIDTH", "STEP"), do_not_specialize_on_alignment=("ENTRIES", "TABLE_WIDTH", "STEP"))
def _attend_kernel(Q, K, V, KS, VS, KM, VM, KO, VO, KOS, VOS, IDS, COUNTS, SEQUENCE, TABLE, BASE, RING_START, PARTIAL, MAXIMUM, SUM, ENTRIES, TABLE_WIDTH, STEP, H_Q: tl.constexpr, H_KV: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr,
                   RATIO: tl.constexpr, CHUNKS: tl.constexpr, M: tl.constexpr, N: tl.constexpr, SCALE: tl.constexpr, K_BITS: tl.constexpr, K_ASYM: tl.constexpr, V_BITS: tl.constexpr, V_ASYM: tl.constexpr,
                   OVERLAY: tl.constexpr, SINK: tl.constexpr, RING: tl.constexpr, PRECISION: tl.constexpr, SPLIT: tl.constexpr):
    """The partial softmaxes of one query's heads sharing a KV head over one chunk of its list: STEP entries, the last chunk taking what is left (the incomplete block)."""
    query, kv_head, chunk = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    group: tl.constexpr = H_Q // H_KV
    QB: tl.constexpr = D // 32
    K_WIDTH: tl.constexpr = D if K_BITS == 16 else QB * 4 * K_BITS
    V_WIDTH: tl.constexpr = D if V_BITS == 16 else QB * 4 * V_BITS
    PER: tl.constexpr = N // RATIO
    rows = tl.arange(0, M)
    valid_rows = rows < group
    blocks = tl.arange(0, QB)
    half = tl.arange(0, 16)
    q_base = (query * H_Q + kv_head * group + rows[None, :, None]) * D + blocks[:, None, None] * 32 + half[None, None, :]
    q_a = tl.load(Q + q_base, mask=valid_rows[None, :, None], other=0.0)
    q_b = tl.load(Q + q_base + 16, mask=valid_rows[None, :, None], other=0.0)
    q_sum = tl.sum(q_a, axis=2) + tl.sum(q_b, axis=2)
    if SPLIT:
        high_a, high_b = q_a.to(tl.bfloat16), q_b.to(tl.bfloat16)
        q_a_rest, q_b_rest = (q_a - high_a.to(tl.float32)).to(tl.bfloat16), (q_b - high_b.to(tl.float32)).to(tl.bfloat16)
        q_a, q_b = high_a, high_b
    else:
        q_a_rest, q_b_rest = q_a, q_b
    count = tl.load(COUNTS + query * 3)
    position = tl.load(SEQUENCE + query).to(tl.int32)
    entries = IDS + query * ENTRIES
    lanes = tl.arange(0, N)
    begin = chunk * STEP
    end = tl.where(chunk == CHUNKS - 1, count, tl.minimum(begin + STEP, count))
    if OVERLAY:
        base = tl.load(BASE)
        sink_end = tl.where(base >= 0, tl.minimum(tl.maximum(begin, tl.load(COUNTS + query * 3 + 1)), end), begin)
        ring_begin = tl.where(base >= 0, tl.maximum(tl.minimum(tl.load(COUNTS + query * 3 + 2), end), sink_end), end)
    else:
        base = -1
        sink_end = begin
        ring_begin = end
    maximum = tl.full((M,), -1.0e20, tl.float32)
    denominator = tl.zeros((M,), tl.float32)
    acc_a = tl.zeros((QB, M, 16), tl.float32)
    acc_b = tl.zeros((QB, M, 16), tl.float32)
    if OVERLAY:
        for start in tl.range(begin, sink_end, PER):
            index = start + lanes // RATIO
            mask = index < sink_end
            columns = tl.load(entries + index, mask=mask, other=0) * RATIO + lanes % RATIO
            valid = mask & (columns <= position)
            records = (base + columns) * H_KV + kv_head
            maximum, denominator, acc_a, acc_b = _tile(q_a, q_b, q_a_rest, q_b_rest, q_sum, KO, VO, KOS, VOS, KOS, VOS, records * D, records * D, records * QB, valid, maximum, denominator, acc_a, acc_b, D, SCALE, 8, False, 8, False, PRECISION, SPLIT)
    for start in tl.range(sink_end, ring_begin, PER):
        index = start + lanes // RATIO
        mask = index < ring_begin
        columns = tl.load(entries + index, mask=mask, other=0) * RATIO + lanes % RATIO
        inside = mask & (columns <= position) & (columns // PAGE < TABLE_WIDTH)
        page = tl.load(TABLE + columns // PAGE, mask=inside, other=-1)
        valid = inside & (page >= 0)
        records = (page * PAGE + columns % PAGE) * H_KV + kv_head
        maximum, denominator, acc_a, acc_b = _tile(q_a, q_b, q_a_rest, q_b_rest, q_sum, K, V, KS, VS, KM, VM, records * K_WIDTH, records * V_WIDTH, records * QB, valid, maximum, denominator, acc_a, acc_b, D, SCALE, K_BITS, K_ASYM, V_BITS, V_ASYM, PRECISION, SPLIT)
    if OVERLAY:
        for start in tl.range(ring_begin, end, PER):
            index = start + lanes // RATIO
            mask = index < end
            columns = tl.load(entries + index, mask=mask, other=0) * RATIO + lanes % RATIO
            valid = mask & (columns <= position)
            records = (base + SINK + (columns & (RING - 1))) * H_KV + kv_head
            maximum, denominator, acc_a, acc_b = _tile(q_a, q_b, q_a_rest, q_b_rest, q_sum, KO, VO, KOS, VOS, KOS, VOS, records * D, records * D, records * QB, valid, maximum, denominator, acc_a, acc_b, D, SCALE, 8, False, 8, False, PRECISION, SPLIT)
    target = (query * H_Q + kv_head * group + rows) * CHUNKS + chunk
    out_base = PARTIAL + target[None, :, None] * D + blocks[:, None, None] * 32 + half[None, None, :]
    tl.store(out_base, acc_a, mask=valid_rows[None, :, None])
    tl.store(out_base + 16, acc_b, mask=valid_rows[None, :, None])
    tl.store(MAXIMUM + target, maximum, mask=valid_rows)
    tl.store(SUM + target, denominator, mask=valid_rows)


@triton.jit
def _reduce_gate_kernel(PARTIAL, MAXIMUM, SUM, GATE, OUT, CHUNKS: tl.constexpr, D: tl.constexpr):
    """A head's chunks merged, its gate applied: the output in the dtype of OUT."""
    row = tl.program_id(0)
    chunks = tl.arange(0, CHUNKS)
    dims = tl.arange(0, D)
    maximum = tl.load(MAXIMUM + row * CHUNKS + chunks)
    weights = libdevice.exp(maximum - tl.max(maximum, 0))
    denominator = tl.sum(tl.load(SUM + row * CHUNKS + chunks) * weights, 0)
    partial = tl.load(PARTIAL + (row * CHUNKS + chunks[:, None]) * D + dims[None, :])
    output = tl.div_rn(tl.sum(partial * weights[:, None], 0), tl.maximum(denominator, 1.0e-20))
    gate = tl.load(GATE + row * 2 * D + D + dims).to(tl.float32)
    output = output * tl.div_rn(1.0, 1.0 + libdevice.exp(-gate))
    tl.store(OUT + row * D + dims, output.to(OUT.dtype.element_ty))


@triton.jit
def _gate_kernel(ATTENDED, GATE, OUT, D: tl.constexpr):
    """A head's attention behind its gate: the second half of its row of the query's projection."""
    row = tl.program_id(0)
    dims = tl.arange(0, D)
    gate = tl.load(GATE + row * 2 * D + D + dims).to(tl.float32)
    output = tl.load(ATTENDED + row * D + dims).to(tl.float32) * tl.div_rn(1.0, 1.0 + libdevice.exp(-gate))
    tl.store(OUT + row * D + dims, output.to(OUT.dtype.element_ty))


def gate_rows(attended, q_full, rows, dim, dtype):
    """[rows, dim] in dtype, the rows of heads: attended [rows, dim] by the sigmoid of the gates in q_full [rows, 2 * dim]."""
    output = torch.empty(rows, dim, dtype=dtype, device=attended.device)
    _gate_kernel[(rows,)](attended, q_full, output, dim, num_warps=4, num_stages=1)
    return output.view(attended.shape[0], -1)


def attend_selected(q, q_full, ids, counts, sequence, table, module, base, ring_start, precision=PRECISION, dtype=torch.bfloat16, chunks=None, output=None):
    """[queries, heads * dim] in dtype: the attention of q [queries, heads, dim] float32 over the tokens of each query's list, gated by the second half of each head's row of q_full [queries, heads * 2 * dim].
    chunks: programs per query and KV head; by default as many as the list has tiles, for a few queries. A prompt has its parallelism in the queries: one chunk reads each query once and needs no partial softmaxes. output: the tensor to write into."""
    queries, heads, dim = q.shape
    device = q.device
    layout, k_cache, v_cache = module.kv_layout, module.k_cache, module.v_cache
    assert q.is_cuda and q_full.is_cuda and ids.is_cuda and counts.is_cuda and sequence.is_cuda and table.is_cuda and k_cache.is_cuda and v_cache.is_cuda
    quantized = k_cache.dtype in (torch.int8, torch.uint8)
    if quantized:
        k_min, v_min, overlay = attention_mixed._operands(module)[1:]
        k_scale, v_scale = module.k_scale, module.v_scale
        k_bits, k_asym, v_bits, v_asym = layout.k.bits, layout.k.asymmetric, layout.v.bits, layout.v.asymmetric
    else:
        k_scale, v_scale, k_min, v_min, overlay = k_cache, v_cache, k_cache, v_cache, (k_cache, v_cache, k_cache, v_cache)
        k_bits, k_asym, v_bits, v_asym = 16, False, 16, False
    has_overlay = quantized and layout.overlay and module.k_overlay is not None and base is not None
    dummy = table[:1]
    page = k_cache.shape[1]
    per_tile = TILE_TOKENS // module.ratio
    chunks = chunks or min(MAX_CHUNKS, triton.next_power_of_2(triton.cdiv(module.top_blocks, per_tile)))
    partial = torch.empty(queries, heads, chunks, dim, dtype=torch.float32, device=device)
    maximum = torch.empty(queries, heads, chunks, dtype=torch.float32, device=device)
    denominator = torch.empty_like(maximum)
    groups = heads // module.num_kv_heads
    _attend_kernel[(queries, module.num_kv_heads, chunks)](q, k_cache, v_cache, k_scale, v_scale, k_min, v_min, *overlay, ids, counts, sequence, table, base if has_overlay else dummy, ring_start if has_overlay else dummy, partial, maximum, denominator,
                                                           ids.shape[1], table.shape[0], triton.cdiv(module.top_blocks, chunks), heads, module.num_kv_heads, dim, page, module.ratio, chunks, triton.next_power_of_2(groups), TILE_TOKENS, module.scale, k_bits, k_asym, v_bits, v_asym,
                                                           has_overlay, layout.sink_tokens(page) if has_overlay else 0, max(layout.recent_tokens, TILE_TOKENS) if has_overlay else TILE_TOKENS, PRECISION if precision == SPLIT_BF16 else precision, precision == SPLIT_BF16, num_warps=4, num_stages=1)
    output = torch.empty(queries, heads * dim, dtype=dtype, device=device) if output is None else output
    _reduce_gate_kernel[(queries * heads,)](partial, maximum, denominator, q_full, output, chunks, dim, num_warps=4, num_stages=1)
    return output


def available(x, module):
    """The GPU runs these kernels: BF16 in Triton, no HIP validation yet; the blocks fit a tile of tokens and the queries of a KV head a tile of rows."""
    return (x.is_cuda and torch.version.hip is None and dispatch.triton_dtype_supported(x.device, torch.bfloat16) and TILE_TOKENS % module.ratio == 0 and module.num_heads // module.num_kv_heads <= 32
            and module.head_dim in (32, 64, 128, 256))


# Register after definitions to preserve Triton's line-number-sensitive cache keys.
from shared.kernels.triton_compilation_log import install_triton_compilation_logger  # noqa: E402

install_triton_compilation_logger()
