# Part of WanGP - WanGP Community License 2.0 (see LICENSE.txt), Copyright (c) 2026 DeepBeepMeep
"""Paged attention over a mixed-precision KV cache (kv_formats): a 4-, 6- or 8-bit body per component plus the exact Q8 overlays
(sink pages and the ring of the most recent tokens). The kernels are the Q8 kernels of attention.py with the loads generalized: a
tile of 32 positions belongs to one region (boundaries are tile aligned), so every pass runs three loops, sink, body and ring, with
the region's loads and one online softmax carried across them.

A group of 32 head dimensions is read as two halves of 16 lanes (kv_formats: values 0..15 in the low nibbles of 16 contiguous bytes,
16..31 in their high nibbles; a Q8 record keeps its 32 bytes), each half loaded straight into the layout its dot needs, so no reorder
follows a load; every product over the head dimension is two half-depth dots. The earlier loads with tl.interleave and permutes, and
lane-indexed byte gathers, measured 1.4-2.7x slower (specs/KV_MIXED_PRECISION_STUDY_20261022.md).

Turing (compute capability 7.5, shared.kernels.bf16_bits.applies): Triton has no BF16 conversion or BF16 tensor-core product there, so
the decoding / verification kernel reads BF16 queries as bits and computes its products in FP32 (IEEE, CUDA cores; decoding is bound by
the cache reads), and the reduction writes the BF16 output as bits. The prefill kernel keeps the GPUs with BF16 support (Turing prefills
through the dense SDPA path, which is not captured in a CUDA graph)."""
import torch

from shared.kernels import bf16_bits, dispatch
from . import kv_formats as kvf

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _group_codes(data_ptr, records, groups, lanes, lane0, mask, mask0, BITS: tl.constexpr, ASYM: tl.constexpr):
        """(codes of lanes 0..15, codes of lanes 16..31) of the groups of some records; records, groups, lanes and mask broadcast to the
        output shape, each nibble byte loaded once. lane0 / mask0 index the 6-bit high-bit words: a size-1 lane axis loads one word per group,
        a copy of lanes loads it per lane. Symmetric formats below 8 bits are centred; Q8 codes are the stored int8 values."""
        if BITS == 8:
            a = tl.load(data_ptr + records + groups * 32 + lanes, mask=mask, other=0).to(tl.int32)
            b = tl.load(data_ptr + records + groups * 32 + 16 + lanes, mask=mask, other=0).to(tl.int32)
            return a, b
        BYTES: tl.constexpr = 4 * BITS
        nibbles = tl.load(data_ptr + records + groups * BYTES + lanes, mask=mask, other=0).to(tl.int32)
        a = nibbles & 15
        b = nibbles >> 4
        if BITS == 6:
            shifts = (lanes % 4) * 8 + (lanes // 4) * 2
            words = (data_ptr + records + groups * BYTES + 16 + lane0).to(tl.pointer_type(tl.int32), bitcast=True)
            a = a | (((tl.load(words, mask=mask0, other=0) >> shifts) & 3) << 4)
            b = b | (((tl.load(words + 1, mask=mask0, other=0) >> shifts) & 3) << 4)
        if ASYM:
            return a, b
        return a - (1 << (BITS - 1)), b - (1 << (BITS - 1))


    @triton.jit
    def _grouped_halves(data_ptr, scale_ptr, min_ptr, records, scale_records, mask, D: tl.constexpr, BITS: tl.constexpr, ASYM: tl.constexpr, TRANSPOSED: tl.constexpr):
        """Dequantized FP32 values of both lane halves of every group: [D/2, N] each for keys (TRANSPOSED) or [N, D/2] for values."""
        dims = tl.arange(0, D // 2)
        groups = dims // 16
        lanes = dims % 16
        if TRANSPOSED:
            a, b = _group_codes(data_ptr, records[None, :], groups[:, None], lanes[:, None], lanes[:, None] * 0, mask[None, :], mask[None, :], BITS, ASYM)
            scale = tl.load(scale_ptr + scale_records[None, :] + groups[:, None], mask=mask[None, :], other=0).to(tl.float32)
            a = a.to(tl.float32) * scale
            b = b.to(tl.float32) * scale
            if ASYM:
                minimum = tl.load(min_ptr + scale_records[None, :] + groups[:, None], mask=mask[None, :], other=0).to(tl.float32)
                a += minimum
                b += minimum
        else:
            a, b = _group_codes(data_ptr, records[:, None], groups[None, :], lanes[None, :], lanes[None, :] * 0, mask[:, None], mask[:, None], BITS, ASYM)
            scale = tl.load(scale_ptr + scale_records[:, None] + groups[None, :], mask=mask[:, None], other=0).to(tl.float32)
            a = a.to(tl.float32) * scale
            b = b.to(tl.float32) * scale
            if ASYM:
                minimum = tl.load(min_ptr + scale_records[:, None] + groups[None, :], mask=mask[:, None], other=0).to(tl.float32)
                a += minimum
                b += minimum
        return a, b


    @triton.jit
    def _grouped_tile(q_a, q_b, k_ptr, v_ptr, ks_ptr, vs_ptr, km_ptr, vm_ptr, k_records, v_records, scale_records, columns, valid_rows, valid_columns, rows_limit,
                      maximum, denominator, acc_a, acc_b, D: tl.constexpr, SCALE: tl.constexpr, K_BITS: tl.constexpr, K_ASYM: tl.constexpr, V_BITS: tl.constexpr, V_ASYM: tl.constexpr,
                      IEEE: tl.constexpr = False):
        """One tile of the grouped kernel (the Q8 kernel's arithmetic: FP32 values, high/low BF16 pairs on the tensor cores; IEEE: FP32
        queries and FP32 products for GPUs without BF16 in Triton), the head dimension split in the two lane halves a and b."""
        k_a, k_b = _grouped_halves(k_ptr, ks_ptr, km_ptr, k_records, scale_records, valid_columns, D, K_BITS, K_ASYM, True)
        if IEEE:
            scores = tl.dot(q_a, k_a, input_precision="ieee")
            scores = tl.dot(q_b, k_b, scores, input_precision="ieee") * SCALE
        else:
            k_hi = k_a.to(q_a.dtype)
            k_lo = (k_a - k_hi.to(tl.float32)).to(q_a.dtype)
            scores = tl.dot(q_a, k_hi)
            scores = tl.dot(q_a, k_lo, scores)
            k_hi = k_b.to(q_a.dtype)
            k_lo = (k_b - k_hi.to(tl.float32)).to(q_a.dtype)
            scores = tl.dot(q_b, k_hi, scores)
            scores = tl.dot(q_b, k_lo, scores) * SCALE
        valid = valid_rows[:, None] & valid_columns[None, :] & (columns[None, :] <= rows_limit[:, None])
        scores = tl.where(valid, scores, -1.0e20)
        new_maximum = tl.maximum(maximum, tl.max(scores, 1))
        correction = tl.exp2((maximum - new_maximum) * 1.4426950408889634)
        p = tl.where(valid, tl.exp2((scores - new_maximum[:, None]) * 1.4426950408889634), 0.0)
        denominator = denominator * correction + tl.sum(p, 1)
        v_a, v_b = _grouped_halves(v_ptr, vs_ptr, vm_ptr, v_records, scale_records, valid_columns, D, V_BITS, V_ASYM, False)
        if IEEE:
            acc_a = tl.dot(p, v_a, acc_a * correction[:, None], input_precision="ieee")
            acc_b = tl.dot(p, v_b, acc_b * correction[:, None], input_precision="ieee")
            return new_maximum, denominator, acc_a, acc_b
        p_hi = p.to(q_a.dtype)
        p_lo = (p - p_hi.to(tl.float32)).to(q_a.dtype)
        v_hi = v_a.to(q_a.dtype)
        v_lo = (v_a - v_hi.to(tl.float32)).to(q_a.dtype)
        acc_a *= correction[:, None]
        acc_a = tl.dot(p_hi, v_hi, acc_a)
        acc_a = tl.dot(p_lo, v_hi, acc_a)
        acc_a = tl.dot(p_hi, v_lo, acc_a)
        acc_a = tl.dot(p_lo, v_lo, acc_a)
        v_hi = v_b.to(q_a.dtype)
        v_lo = (v_b - v_hi.to(tl.float32)).to(q_a.dtype)
        acc_b *= correction[:, None]
        acc_b = tl.dot(p_hi, v_hi, acc_b)
        acc_b = tl.dot(p_lo, v_hi, acc_b)
        acc_b = tl.dot(p_hi, v_lo, acc_b)
        acc_b = tl.dot(p_lo, v_lo, acc_b)
        return new_maximum, denominator, acc_a, acc_b


    @triton.jit
    def mixed_grouped_partials_kernel(q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, km_ptr, vm_ptr, ko_ptr, vo_ptr, kos_ptr, vos_ptr, tables_ptr, lengths_ptr, base_ptr, ring_ptr,
                                      partial_ptr, maximum_ptr, sum_ptr, Q_PER_SEQ, H_Q: tl.constexpr, H_KV: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr, TABLE_WIDTH,
                                      SPLITS: tl.constexpr, M: tl.constexpr, N: tl.constexpr, SCALE: tl.constexpr,
                                      K_BITS: tl.constexpr, K_ASYM: tl.constexpr, V_BITS: tl.constexpr, V_ASYM: tl.constexpr, OVERLAY: tl.constexpr, SINK: tl.constexpr, RING: tl.constexpr,
                                      Q_BITS: tl.constexpr = False):
        """Decoding and verification: the queries of a sequence that share a KV head attend together; the context is split across programs.
        Q_BITS: BF16 queries read as bits (an int16 pointer), FP32 products (Turing)."""
        sequence, kv_head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        split, query_block = tile % SPLITS, tile // SPLITS
        group: tl.constexpr = H_Q // H_KV
        QB: tl.constexpr = D // 32
        K_WIDTH: tl.constexpr = QB * 4 * K_BITS
        V_WIDTH: tl.constexpr = QB * 4 * V_BITS
        rows = query_block * M + tl.arange(0, M)
        query_index = sequence * Q_PER_SEQ + rows // group
        query_head = kv_head * group + rows % group
        valid_rows = rows < Q_PER_SEQ * group
        half = tl.arange(0, D // 2)
        dims_a = (half // 16) * 32 + half % 16  # the head dimensions of lane half a; b is 16 further
        q_base = (query_index[:, None] * H_Q + query_head[:, None]) * D + dims_a[None, :]
        if Q_BITS:
            q_a = bf16_bits.bf16_bits_to_fp32(tl.load(q_ptr + q_base, mask=valid_rows[:, None], other=0))
            q_b = bf16_bits.bf16_bits_to_fp32(tl.load(q_ptr + q_base + 16, mask=valid_rows[:, None], other=0))
        else:
            q_a = tl.load(q_ptr + q_base, mask=valid_rows[:, None], other=0.0)
            q_b = tl.load(q_ptr + q_base + 16, mask=valid_rows[:, None], other=0.0)
        context_len = tl.load(lengths_ptr + sequence)
        prefix_len = context_len - Q_PER_SEQ
        rows_limit = prefix_len + rows // group
        tiles_per_split = tl.cdiv(tl.cdiv(context_len, N), SPLITS)
        begin = split * tiles_per_split * N
        end = tl.minimum(context_len, (split + 1) * tiles_per_split * N)
        if OVERLAY:
            base = tl.load(base_ptr + sequence)
            ring_start = tl.load(ring_ptr + sequence)
            sink_end = tl.where(base >= 0, tl.minimum(tl.maximum(begin, SINK), end), begin)
            ring_begin = tl.where(base >= 0, tl.maximum(tl.minimum(ring_start, end), sink_end), end)
        else:
            base = -1
            sink_end = begin
            ring_begin = end
        maximum = tl.full((M,), -1.0e20, tl.float32)
        denominator = tl.zeros((M,), tl.float32)
        acc_a = tl.zeros((M, D // 2), tl.float32)
        acc_b = tl.zeros((M, D // 2), tl.float32)
        if OVERLAY:
            for start in tl.range(begin, sink_end, N):
                columns = start + tl.arange(0, N)
                valid_columns = columns < sink_end
                records = (base + columns) * H_KV + kv_head
                maximum, denominator, acc_a, acc_b = _grouped_tile(q_a, q_b, ko_ptr, vo_ptr, kos_ptr, vos_ptr, kos_ptr, vos_ptr, records * D, records * D, records * QB, columns, valid_rows, valid_columns, rows_limit,
                                                                   maximum, denominator, acc_a, acc_b, D, SCALE, 8, False, 8, False, Q_BITS)
        for start in tl.range(sink_end, ring_begin, N):
            columns = start + tl.arange(0, N)
            valid_columns = columns < ring_begin
            page = tl.load(tables_ptr + sequence * TABLE_WIDTH + columns // PAGE, mask=valid_columns, other=0)
            records = (page * PAGE + columns % PAGE) * H_KV + kv_head
            maximum, denominator, acc_a, acc_b = _grouped_tile(q_a, q_b, k_ptr, v_ptr, ks_ptr, vs_ptr, km_ptr, vm_ptr, records * K_WIDTH, records * V_WIDTH, records * QB, columns, valid_rows, valid_columns, rows_limit,
                                                               maximum, denominator, acc_a, acc_b, D, SCALE, K_BITS, K_ASYM, V_BITS, V_ASYM, Q_BITS)
        if OVERLAY:
            for start in tl.range(ring_begin, end, N):
                columns = start + tl.arange(0, N)
                valid_columns = columns < end
                records = (base + SINK + (columns & (RING - 1))) * H_KV + kv_head
                maximum, denominator, acc_a, acc_b = _grouped_tile(q_a, q_b, ko_ptr, vo_ptr, kos_ptr, vos_ptr, kos_ptr, vos_ptr, records * D, records * D, records * QB, columns, valid_rows, valid_columns, rows_limit,
                                                                   maximum, denominator, acc_a, acc_b, D, SCALE, 8, False, 8, False, Q_BITS)
        index = (query_index * H_Q + query_head) * SPLITS + split
        tl.store(partial_ptr + index[:, None] * D + dims_a[None, :], acc_a, mask=valid_rows[:, None])
        tl.store(partial_ptr + index[:, None] * D + dims_a[None, :] + 16, acc_b, mask=valid_rows[:, None])
        tl.store(maximum_ptr + index, maximum, mask=valid_rows)
        tl.store(sum_ptr + index, denominator, mask=valid_rows)


    @triton.jit
    def _exact_tile(q_a, q_b, qsum, k_ptr, v_ptr, ks_ptr, vs_ptr, km_ptr, vm_ptr, k_records, v_records, scale_records, columns, row_mask, column_mask, rows_limit,
                    maximum, denominator, acc_a, acc_b, dtype: tl.constexpr, QB: tl.constexpr, softmax_scale: tl.constexpr,
                    K_BITS: tl.constexpr, K_ASYM: tl.constexpr, V_BITS: tl.constexpr, V_ASYM: tl.constexpr):
        """One tile of the exact prefill kernel: integer codes on the tensor cores, per-block scales (and minimums) on the FP32 partial sums;
        the keys as [QB, 16, N] halves, the values as [QB, N, 16] halves."""
        blocks = tl.arange(0, QB)
        lanes = tl.arange(0, 16)
        lane0 = tl.arange(0, 1)
        k_a, k_b = _group_codes(k_ptr, k_records[None, None, :], blocks[:, None, None], lanes[None, :, None], lane0[None, :, None], column_mask[None, None, :], column_mask[None, None, :], K_BITS, K_ASYM)
        dots = tl.dot(q_a, k_a.to(dtype))
        dots = tl.dot(q_b, k_b.to(dtype), dots)
        k_scale = tl.load(ks_ptr + scale_records[None, :] + blocks[:, None], mask=column_mask[None, :], other=0.0).to(tl.float32)
        scores = tl.sum(dots * k_scale[:, None, :], axis=0)
        if K_ASYM:
            k_min = tl.load(km_ptr + scale_records[None, :] + blocks[:, None], mask=column_mask[None, :], other=0.0).to(tl.float32)
            scores += tl.sum(qsum[:, :, None] * k_min[:, None, :], axis=0)
        scores = scores * softmax_scale
        causal = columns[None, :] <= rows_limit[:, None]
        scores = tl.where(row_mask[:, None] & column_mask[None, :] & causal, scores, -float("inf"))
        tile_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
        correction = tl.exp2((maximum - tile_maximum) * 1.4426950408889634)
        probabilities = tl.exp2((scores - tile_maximum[:, None]) * 1.4426950408889634)
        denominator = denominator * correction + tl.sum(probabilities, axis=1)
        v_scale = tl.load(vs_ptr + scale_records[None, :] + blocks[:, None], mask=column_mask[None, :], other=0.0).to(tl.float32)
        weighted = probabilities[None, :, :] * (v_scale[:, None, :] * 64.0)
        w_high = weighted.to(dtype)
        w_low = (weighted - w_high.to(tl.float32)).to(dtype)
        v_a, v_b = _group_codes(v_ptr, v_records[None, :, None], blocks[:, None, None], lanes[None, None, :], lane0[None, None, :], column_mask[None, :, None], column_mask[None, :, None], V_BITS, V_ASYM)
        v = v_a.to(dtype)
        acc_a = acc_a * correction[None, :, None]
        acc_a = tl.dot(w_high, v, acc_a)
        acc_a = tl.dot(w_low, v, acc_a)
        v = v_b.to(dtype)
        acc_b = acc_b * correction[None, :, None]
        acc_b = tl.dot(w_high, v, acc_b)
        acc_b = tl.dot(w_low, v, acc_b)
        if V_ASYM:
            v_min = tl.load(vm_ptr + scale_records[None, :] + blocks[:, None], mask=column_mask[None, :], other=0.0).to(tl.float32)
            minimum = (tl.sum(probabilities[None, :, :] * v_min[:, None, :], axis=2) * 64.0)[:, :, None]
            acc_a += minimum
            acc_b += minimum
        return tile_maximum, denominator, acc_a, acc_b


    @triton.jit
    def mixed_prefill_exact_kernel(q_ptr, k_ptr, v_ptr, ks_ptr, vs_ptr, km_ptr, vm_ptr, ko_ptr, vo_ptr, kos_ptr, vos_ptr, block_tables_ptr, cu_q_ptr, cu_k_ptr, base_ptr, ring_ptr, out_ptr,
                                   q_stride_t: tl.constexpr, q_stride_h: tl.constexpr, bt_stride, out_stride_t: tl.constexpr, out_stride_h: tl.constexpr, softmax_scale: tl.constexpr,
                                   H_Q: tl.constexpr, H_KV: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                                   K_BITS: tl.constexpr, K_ASYM: tl.constexpr, V_BITS: tl.constexpr, V_ASYM: tl.constexpr, OVERLAY: tl.constexpr, SINK: tl.constexpr, RING: tl.constexpr):
        """Prefill of a chunk over its sequence's pages: exact integer products with per-block scales, as q8_prefill_exact_kernel."""
        QB: tl.constexpr = D // 32
        K_WIDTH: tl.constexpr = QB * 4 * K_BITS
        V_WIDTH: tl.constexpr = QB * 4 * V_BITS
        sequence = tl.program_id(0)
        query_head = tl.program_id(1)
        query_block = tl.program_id(2)
        q_start = tl.load(cu_q_ptr + sequence)
        q_end = tl.load(cu_q_ptr + sequence + 1)
        k_end = tl.load(cu_k_ptr + sequence + 1) - tl.load(cu_k_ptr + sequence)
        q_len = q_end - q_start
        prefix_len = k_end - q_len
        rows = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < q_len
        rows_limit = prefix_len + rows
        blocks = tl.arange(0, QB)
        lanes = tl.arange(0, 16)
        q_base = (q_start + rows[None, :, None]) * q_stride_t + query_head * q_stride_h + blocks[:, None, None] * 32 + lanes[None, None, :]
        q_a = tl.load(q_ptr + q_base, mask=row_mask[None, :, None], other=0.0)
        q_b = tl.load(q_ptr + q_base + 16, mask=row_mask[None, :, None], other=0.0)
        qsum = tl.sum(q_a.to(tl.float32), axis=2) + tl.sum(q_b.to(tl.float32), axis=2)
        maximum = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        denominator = tl.zeros((BLOCK_M,), tl.float32)
        acc_a = tl.zeros((QB, BLOCK_M, 16), tl.float32)
        acc_b = tl.zeros((QB, BLOCK_M, 16), tl.float32)
        kv_head = query_head // (H_Q // H_KV)
        limit = tl.minimum(k_end, prefix_len + (query_block + 1) * BLOCK_M)
        if OVERLAY:
            base = tl.load(base_ptr + sequence)
            ring_start = tl.load(ring_ptr + sequence)
            sink_end = tl.where(base >= 0, tl.minimum(limit, SINK), 0)
            ring_begin = tl.where(base >= 0, tl.maximum(tl.minimum(ring_start, limit), sink_end), limit)
        else:
            base = -1
            sink_end = 0
            ring_begin = limit
        if OVERLAY:
            for key_start in tl.range(0, sink_end, BLOCK_N):
                columns = key_start + tl.arange(0, BLOCK_N)
                column_mask = columns < sink_end
                records = (base + columns) * H_KV + kv_head
                maximum, denominator, acc_a, acc_b = _exact_tile(q_a, q_b, qsum, ko_ptr, vo_ptr, kos_ptr, vos_ptr, kos_ptr, vos_ptr, records * D, records * D, records * QB, columns, row_mask, column_mask, rows_limit,
                                                                 maximum, denominator, acc_a, acc_b, q_a.dtype, QB, softmax_scale, 8, False, 8, False)
        for key_start in tl.range(sink_end, ring_begin, BLOCK_N):
            columns = key_start + tl.arange(0, BLOCK_N)
            column_mask = columns < ring_begin
            physical_block = tl.load(block_tables_ptr + sequence * bt_stride + columns // PAGE, mask=column_mask, other=0)
            records = (physical_block * PAGE + columns % PAGE) * H_KV + kv_head
            maximum, denominator, acc_a, acc_b = _exact_tile(q_a, q_b, qsum, k_ptr, v_ptr, ks_ptr, vs_ptr, km_ptr, vm_ptr, records * K_WIDTH, records * V_WIDTH, records * QB, columns, row_mask, column_mask, rows_limit,
                                                             maximum, denominator, acc_a, acc_b, q_a.dtype, QB, softmax_scale, K_BITS, K_ASYM, V_BITS, V_ASYM)
        if OVERLAY:
            for key_start in tl.range(ring_begin, limit, BLOCK_N):
                columns = key_start + tl.arange(0, BLOCK_N)
                column_mask = columns < limit
                records = (base + SINK + (columns & (RING - 1))) * H_KV + kv_head
                maximum, denominator, acc_a, acc_b = _exact_tile(q_a, q_b, qsum, ko_ptr, vo_ptr, kos_ptr, vos_ptr, kos_ptr, vos_ptr, records * D, records * D, records * QB, columns, row_mask, column_mask, rows_limit,
                                                                 maximum, denominator, acc_a, acc_b, q_a.dtype, QB, softmax_scale, 8, False, 8, False)
        scale = 1.0 / (denominator[None, :, None] * 64.0)
        out_base = (q_start + rows[None, :, None]) * out_stride_t + query_head * out_stride_h + blocks[:, None, None] * 32 + lanes[None, None, :]
        tl.store(out_ptr + out_base, acc_a * scale, mask=row_mask[None, :, None])
        tl.store(out_ptr + out_base + 16, acc_b * scale, mask=row_mask[None, :, None])


    @triton.jit(do_not_specialize=["start", "count"])
    def dense_kernel(data_ptr, scale_ptr, min_ptr, overlay_ptr, overlay_scale_ptr, tables_ptr, cu_k_ptr, base_ptr, ring_ptr, out_ptr, TABLE_WIDTH, start, count,
                     H_KV: tl.constexpr, D: tl.constexpr, PAGE: tl.constexpr, N: tl.constexpr, BITS: tl.constexpr, ASYM: tl.constexpr,
                     OVERLAY: tl.constexpr, SINK: tl.constexpr, RING: tl.constexpr):
        """A tile of the positions start .. start + count of a sequence's keys or values dequantized to out [those positions of every sequence,
        H_KV, D] in out's type: the body's codes with their scales (and minimums), the exact Q8 rows of the sink and of the ring where the
        layout has them."""
        sequence, kv_head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        QB: tl.constexpr = D // 32
        WIDTH: tl.constexpr = QB * 4 * BITS
        first = tl.load(cu_k_ptr + sequence) - start
        length = tl.minimum(tl.load(cu_k_ptr + sequence + 1) - tl.load(cu_k_ptr + sequence), start + count)
        columns = start + tile * N + tl.arange(0, N)
        valid = columns < length
        half = tl.arange(0, D // 2)
        dims_a = (half // 16) * 32 + half % 16
        page = tl.load(tables_ptr + sequence * TABLE_WIDTH + columns // PAGE, mask=valid, other=0)
        records = (page * PAGE + columns % PAGE) * H_KV + kv_head
        if OVERLAY:  # the other region's loads are masked out
            base = tl.load(base_ptr + sequence)
            exact = (base >= 0) & ((columns < SINK) | (columns >= tl.load(ring_ptr + sequence)))
            rows = (base + tl.where(columns < SINK, columns, SINK + (columns & (RING - 1)))) * H_KV + kv_head
            a, b = _grouped_halves(data_ptr, scale_ptr, min_ptr, records * WIDTH, records * QB, valid & (exact == 0), D, BITS, ASYM, False)
            exact_a, exact_b = _grouped_halves(overlay_ptr, overlay_scale_ptr, overlay_scale_ptr, rows * D, rows * QB, valid & exact, D, 8, False, False)
            a = tl.where(exact[:, None], exact_a, a)
            b = tl.where(exact[:, None], exact_b, b)
        else:
            a, b = _grouped_halves(data_ptr, scale_ptr, min_ptr, records * WIDTH, records * QB, valid, D, BITS, ASYM, False)
        out = ((first + columns) * H_KV + kv_head)[:, None] * D + dims_a[None, :]
        tl.store(out_ptr + out, a, mask=valid[:, None])
        tl.store(out_ptr + out + 16, b, mask=valid[:, None])


def _operands(module):
    layout = module.kv_layout
    k_min = module.k_min if module.k_min is not None else module.k_scale
    v_min = module.v_min if module.v_min is not None else module.v_scale
    if layout.overlay and module.k_overlay is not None:
        overlay = (module.k_overlay, module.v_overlay, module.k_overlay_scale, module.v_overlay_scale)
    else:
        overlay = (module.k_cache, module.v_cache, module.k_scale, module.v_scale)
    return layout, k_min, v_min, overlay


def _overlay_context(context, layout, device):
    if layout.overlay and context.overlay_base is not None:
        return True, context.overlay_base, context.ring_start
    dummy = torch.zeros(1, dtype=torch.int32, device=device)
    return False, dummy, dummy


def grouped_attention(q: torch.Tensor, module, context, softmax_scale: float, block_m: int | None = None, num_warps: int = 4, num_stages: int = 1) -> torch.Tensor:
    """Decoding and speculative verification over a mixed cache; q [queries, H_Q, D], context_lens from the context."""
    from .attention import q8_grouped_reduce_kernel
    layout, k_min, v_min, overlay = _operands(module)
    context_lens = context.cu_seqlens_k[1:] if context.speculative_verify else context.context_lens
    k_cache, v_cache = module.k_cache, module.v_cache
    kv_heads, head_dim = k_cache.shape[2], q.shape[2]
    queries_per_sequence = q.shape[0] // context_lens.numel()
    groups = q.shape[1] // kv_heads
    query_rows = queries_per_sequence * groups
    bits = bf16_bits.applies(q)  # Turing: BF16 q read as bits, FP32 products, BF16 output written as bits
    if block_m is None:  # Turing's FP32 operands of 32 rows exceed its 64 KiB of shared memory: 16 rows there
        block_m = 32 if query_rows > 16 and not bits else 16
    query_blocks = triton.cdiv(query_rows, block_m)
    blocks = context_lens.numel() * kv_heads * triton.cdiv(query_rows, 16)
    splits = min(128, triton.next_power_of_2(triton.cdiv(4 * torch.cuda.get_device_properties(q.device).multi_processor_count, blocks)))
    partial = torch.empty((*q.shape[:2], splits, head_dim), device=q.device, dtype=torch.float32)
    maximum = torch.empty((*q.shape[:2], splits), device=q.device, dtype=torch.float32)
    denominator = torch.empty_like(maximum)
    output = torch.empty_like(q)
    has_overlay, base, ring = _overlay_context(context, layout, q.device)
    mixed_grouped_partials_kernel[(context_lens.numel(), kv_heads, query_blocks * splits)](
        q.view(torch.int16) if bits else q, k_cache, v_cache, module.k_scale, module.v_scale, k_min, v_min, *overlay, context.block_tables, context_lens, base, ring, partial, maximum, denominator,
        queries_per_sequence, q.shape[1], kv_heads, head_dim, k_cache.shape[1], context.block_tables.shape[1], splits, block_m, 32, softmax_scale,
        layout.k.bits, layout.k.asymmetric, layout.v.bits, layout.v.asymmetric, has_overlay, layout.sink_tokens(k_cache.shape[1]), max(layout.recent_tokens, 32),
        Q_BITS=bits, num_warps=num_warps, num_stages=num_stages)
    if bits:
        bf16_bits._q8_attention_reduce_kernel[(q.shape[0] * q.shape[1],)](partial, maximum, denominator, output.view(torch.int16), splits, head_dim, num_warps=4)
    else:
        q8_grouped_reduce_kernel[(q.shape[0] * q.shape[1],)](partial, maximum, denominator, output, splits, head_dim, num_warps=4)
    return output


def prefill(q: torch.Tensor, module, context, softmax_scale: float, num_warps: int = 4, num_stages: int = 1) -> torch.Tensor:
    """Prefill of chunks over their sequences' pages (prefix cache) with a mixed cache."""
    layout, k_min, v_min, overlay = _operands(module)
    k_cache, v_cache = module.k_cache, module.v_cache
    output = torch.empty_like(q)
    query_lengths = context.cu_seqlens_q[1:] - context.cu_seqlens_q[:-1]
    grid = (query_lengths.numel(), q.shape[1], triton.cdiv(q.shape[0], 32))
    has_overlay, base, ring = _overlay_context(context, layout, q.device)
    mixed_prefill_exact_kernel[grid](
        q, k_cache, v_cache, module.k_scale, module.v_scale, k_min, v_min, *overlay, context.block_tables, context.cu_seqlens_q, context.cu_seqlens_k, base, ring, output,
        q.stride(0), q.stride(1), context.block_tables.stride(0), output.stride(0), output.stride(1), softmax_scale,
        q.shape[1], k_cache.shape[2], q.shape[2], k_cache.shape[1], 32, 32,
        layout.k.bits, layout.k.asymmetric, layout.v.bits, layout.v.asymmetric, has_overlay, layout.sink_tokens(k_cache.shape[1]), max(layout.recent_tokens, 32),
        num_warps=num_warps, num_stages=num_stages)
    return output


def dense(module, context, dtype: torch.dtype, out: tuple[torch.Tensor, torch.Tensor] | None = None, start: int = 0, count: int | None = None, capacity: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """The keys and values of every sequence's context dequantized to [tokens, H_KV, D] in dtype (cu_seqlens_k offsets), for an attention
    over dense operands: 2 x H_KV x D values per context token, each rounded to dtype. out: a pair from an earlier call of the same pass.
    start, count: only these positions of a single sequence, written from row 0 of out (capacity rows when it is allocated here)."""
    layout, k_min, v_min, overlay = _operands(module)
    k_cache = module.k_cache
    kv_heads, head_dim, page = k_cache.shape[2], module.head_dim, k_cache.shape[1]
    if count is None:
        count = int(context.max_seqlen_k)
    has_overlay, base, ring = _overlay_context(context, layout, k_cache.device)
    grid = (context.cu_seqlens_k.numel() - 1, kv_heads, triton.cdiv(count, 32))
    if out is None:
        out = tuple(torch.empty((capacity or int(context.cu_seqlens_k[-1]), kv_heads, head_dim), dtype=dtype, device=k_cache.device) for _ in range(2))
    for target, data, scale, minimum, rows, row_scale, fmt in ((out[0], k_cache, module.k_scale, k_min, overlay[0], overlay[2], layout.k), (out[1], module.v_cache, module.v_scale, v_min, overlay[1], overlay[3], layout.v)):
        dense_kernel[grid](data, scale, minimum, rows, row_scale, context.block_tables, context.cu_seqlens_k, base, ring, target, context.block_tables.shape[1], start, count,
                           kv_heads, head_dim, page, 32, fmt.bits, fmt.asymmetric, has_overlay, layout.sink_tokens(page), max(layout.recent_tokens, 32), num_warps=4)
    return out


def available(q: torch.Tensor, module, prefill: bool = False) -> bool:
    """The Triton kernels serve this call: CUDA (not HIP), a supported head size, and Triton with q's dtype conversions, or on Turing
    (bf16_bits.applies) a decoding / verification call with BF16 queries (read as bits); Turing prefills through the dense path."""
    return (triton is not None and q.is_cuda and torch.version.hip is None and q.dtype in (torch.float16, torch.bfloat16) and q.is_contiguous()
            and module.head_dim in (32, 64, 128, 256) and (dispatch.triton_dtype_supported(q.device, q.dtype) or not prefill and bf16_bits.applies(q)))


if triton is not None:
    from shared.kernels.triton_compilation_log import install_triton_compilation_logger
    install_triton_compilation_logger()
