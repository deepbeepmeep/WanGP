# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Kernels addressing rows through positions known on the device: the counter of a CUDA graph loop (mmgp.cuda_graphs.loop), a device int64
giving rows first + counter * step of a tensor, and the segments of a segmented modulation. copy_rows_at / store_rows_at gather and store a
turn's rows (exactly), residual_add_at adds rows to them as add_ / addcmul_ do, lora_residual_at is lora_triton.residual_ for them, rms_norm_modulate is a weightless RMSNorm with a segment
modulation in one FP32 pass (fixed rows or a turn's), and modulate_runs is norm_modulate's modulation, one launch per run of segments
(bit-identical to the per-segment kernels it replaced)."""
import torch
import triton
import triton.language as tl

from shared.kernels.bf16_bits import bf16_bits_to_fp32, fp32_to_bf16_bits
from shared.kernels.denoiser_bits_triton import _round_bf16

_INTS = ["N", "FIRST", "STEP", "LDX", "SPAN"]


_MOVE_INTS = ["X_FIRST", "X_STEP", "Y_FIRST", "Y_STEP", "LDX", "LDY", "ROWS"]


@triton.jit(do_not_specialize=_MOVE_INTS, do_not_specialize_on_alignment=_MOVE_INTS)
def _move_rows(X, Y, COUNTER, X_FIRST, X_STEP, Y_FIRST, Y_STEP, LDX, LDY, ROWS, D, R: tl.constexpr, B: tl.constexpr):
    """Rows r < ROWS: row X_FIRST + counter * X_STEP + r of X to row Y_FIRST + counter * Y_STEP + r of Y, by tiles of R rows and B columns."""
    turn = tl.load(COUNTER)
    rows = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
    cols = tl.program_id(1) * B + tl.arange(0, B)
    valid = (rows < ROWS)[:, None] & (cols < D)[None, :]
    source = X + (X_FIRST + turn * X_STEP + rows)[:, None] * LDX + cols[None, :]
    tl.store(Y + (Y_FIRST + turn * Y_STEP + rows)[:, None] * LDY + cols[None, :], tl.load(source, valid, 0), valid)


_FLAT_INTS = ["X_FIRST", "X_STEP", "Y_FIRST", "Y_STEP", "N", "D"]


@triton.jit(do_not_specialize=_FLAT_INTS, do_not_specialize_on_alignment=_FLAT_INTS)
def _move_flat(X, Y, COUNTER, X_FIRST, X_STEP, Y_FIRST, Y_STEP, N, D, BLOCK: tl.constexpr):
    """_move_rows for rows that are contiguous on both sides: N elements from element (X_FIRST + counter * X_STEP) * D of X."""
    turn = tl.load(COUNTER)
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    tl.store(Y + (Y_FIRST + turn * Y_STEP) * D + i, tl.load(X + (X_FIRST + turn * X_STEP) * D + i, valid, 0), valid)


def _move(source, target, counter, x_first, x_step, y_first, y_step, count):
    width, size = source.shape[1], source.element_size()
    if source.stride(0) == width and target.stride(0) == width:  # contiguous rows: one flat span each side
        if size < 4 and width * size % 4 == 0 and source.data_ptr() % 4 == 0 and target.data_ptr() % 4 == 0:  # moved as 4-byte words
            source, target, width, size = source.view(torch.int32), target.view(torch.int32), width * size // 4, 4
        block = 16384 // size
        _move_flat[(triton.cdiv(count * width, block),)](source, target, counter, x_first, x_step, y_first, y_step, count * width, width, block, num_warps=8)
        return
    block = min(triton.next_power_of_2(width), max(16, 4096 // size))  # 4 KB of a row at most per tile row
    rows = max(1, min(64, 16384 // (block * size)))  # tiles of 16 KB
    _move_rows[(triton.cdiv(count, rows), triton.cdiv(width, block))](source, target, counter, x_first, x_step, y_first, y_step, source.stride(0), target.stride(0),
                                                                    count, width, rows, block, num_warps=4)


_GATE_INTS = ["FIRST", "STEP", "SPAN", "D"]


@triton.jit(do_not_specialize=_GATE_INTS, do_not_specialize_on_alignment=_GATE_INTS)
def _segment_row_at(TABLE, OUT, COUNTER, FIRST, STEP, SPAN, LDT, D, BLOCK: tl.constexpr):
    """OUT = row (FIRST + counter * STEP) // SPAN of TABLE: the vector of a turn's segment (segments of SPAN rows)."""
    row = (FIRST + tl.load(COUNTER) * STEP) // SPAN
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(OUT + cols, tl.load(TABLE + row * LDT + cols, cols < D, 0), cols < D)


def segment_row_at(table, counter, first, step, span):
    """table (segments, D)'s row of the segment holding row first + counter[0] * step (segments of span rows), as a new contiguous vector:
    a CUDA graph loop's per-turn gate."""
    out = torch.empty(table.shape[1], dtype=table.dtype, device=table.device)
    block = min(1024, triton.next_power_of_2(table.shape[1]))
    _segment_row_at[(triton.cdiv(table.shape[1], block),)](table, out, counter, first, step, span, table.stride(0), table.shape[1], block)
    return out


def copy_rows_at(rows, counter, first, step, count):
    """rows (R, D) with unit column stride: its `count` rows from first + counter[0] * step, as a new contiguous tensor."""
    out = torch.empty((count, rows.shape[1]), dtype=rows.dtype, device=rows.device)
    _move(rows, out, counter, first, step, 0, 0, count)
    return out


def store_rows_at(target, rows, counter, first, step):
    """target's rows from first + counter[0] * step = rows (count, ...), of target's dtype, rows of unit column stride (2-D after their first
    dimension): copy_rows_at's other way."""
    rows2 = rows.reshape(rows.shape[0], -1)
    _move(rows2, target.view(target.shape[0], -1), counter, 0, 0, first, step, rows2.shape[0])
    torch.autograd.graph.increment_version(target)


_ADD_INTS = ["FIRST", "STEP", "SPAN", "LDY", "LDO", "ROWS", "D"]


@triton.jit(do_not_specialize=_ADD_INTS, do_not_specialize_on_alignment=_ADD_INTS)
def _residual_add_at(Y, OUT, SCALE, COUNTER, FIRST, STEP, SPAN, LDY, LDO, ROWS, D, HAS_SCALE: tl.constexpr, R: tl.constexpr, B: tl.constexpr):
    """OUT's rows FIRST + counter * STEP + r += Y's rows r (times SCALE's vector, of their segment's row when SPAN > 0), each rounded in FP32
    then to OUT's dtype: PyTorch's add_ / addcmul_ (launched without FMA contraction: the product rounds before the sum, as theirs)."""
    shift = FIRST + tl.load(COUNTER) * STEP
    rows = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
    cols = tl.program_id(1) * B + tl.arange(0, B)
    valid = (rows < ROWS)[:, None] & (cols < D)[None, :]
    y = tl.load(Y + rows[:, None] * LDY + cols[None, :], valid, 0).to(tl.float32)
    if HAS_SCALE:
        if SPAN > 0:
            SCALE += (shift // SPAN) * D
        y = y * tl.load(SCALE + cols, cols < D, 0).to(tl.float32)[None, :]
    target = OUT + (shift + rows)[:, None] * LDO + cols[None, :]
    tl.store(target, (tl.load(target, valid, 0).to(tl.float32) + y).to(OUT.dtype.element_ty), valid)


def residual_add_at(target, y, scale, counter, first, step, span=0):
    """target's rows from first + counter[0] * step += y (count, D), or addcmul_ with scale: one vector (span 0) or a table (segments, D) of
    contiguous vectors, segments of span rows from target's row 0. The bits of target[rows].add_(y) / .addcmul_(y, vector)."""
    width, size = y.shape[1], y.element_size()
    block = min(triton.next_power_of_2(width), max(16, 4096 // size))
    rows = max(1, min(64, 16384 // (block * size)))
    _residual_add_at[(triton.cdiv(y.shape[0], rows), triton.cdiv(width, block))](y, target, y if scale is None else scale, counter, first, step, span, y.stride(0),
                                                                              target.stride(0), y.shape[0], width, scale is not None, rows, block, num_warps=4,
                                                                              enable_fp_fusion=False)
    torch.autograd.graph.increment_version(target)


@triton.jit(do_not_specialize=["M", "K", "SCALE"] + _INTS, do_not_specialize_on_alignment=["Y", "FACTOR", "OUT", "RES", "GATE", "M", "K"] + _INTS)
def _lora_residual_at(Y, FACTOR, OUT, RES, GATE, COUNTER, FIRST, STEP, SPAN, M, N, K, SY, SFN, SFK, SO, SR, SCALE, HAS_GATE: tl.constexpr,
                      RANK_FIRST: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """lora_triton._output_residual with the residual rows from FIRST + counter * STEP and, SPAN > 0, the gate row of their segment."""
    shift = FIRST + tl.load(COUNTER) * STEP
    if HAS_GATE and SPAN > 0:
        GATE += (shift // SPAN) * N
    rm = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    m_mask, n_mask = rm < M, rn < N
    y_ptr = Y + rm[:, None] * SY + rk[None, :]
    factor_ptr = FACTOR + rk[:, None] * SFK + rn[None, :] if RANK_FIRST else FACTOR + rn[:, None] * SFN + rk[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        y = tl.load(y_ptr, mask=m_mask[:, None], other=0.0)
        if RANK_FIRST:
            acc = tl.dot(y, tl.load(factor_ptr, mask=(k * BK + rk < K)[:, None] & n_mask[None, :], other=0.0), acc)
        else:
            acc = tl.dot(y, tl.trans(tl.load(factor_ptr, mask=n_mask[:, None], other=0.0)), acc)
        y_ptr += BK
        factor_ptr += BK * (SFK if RANK_FIRST else 1)
    mask = m_mask[:, None] & n_mask[None, :]
    dtype = RES.dtype.element_ty
    out = (acc * SCALE + tl.load(OUT + rm[:, None] * SO + rn[None, :], mask=mask, other=0.0).to(tl.float32)).to(dtype).to(tl.float32)
    if HAS_GATE:
        out = out * tl.load(GATE + rn, mask=n_mask, other=0.0).to(tl.float32)[None, :]
    res_ptr = RES + (shift + rm[:, None]) * SR + rn[None, :]  # the row offset after the products: nothing held through the K loop
    tl.store(res_ptr, (tl.load(res_ptr, mask=mask, other=0.0).to(tl.float32) + out).to(dtype), mask=mask)


def lora_residual_at(residual, out, y, factor, scale, rank_first, gate, counter, first, step, span):
    """lora_triton.residual_ (residual += gate * (out + scale * y @ factor)) writing the `out.shape[0]` rows of residual from
    first + counter * step, with the gate vector of their segment (gate (segments, outputs), span rows each) or one vector (span 0), in
    residual_'s tuned tile configuration (the same bits)."""
    from shared.kernels import lora_triton
    y = lora_triton._pad(y)
    config = lora_triton._TUNER.config("output_residual", f"gate={gate is not None} rank_first={rank_first}", residual.device, lora_triton._CANDIDATES, None, False)
    bm, bn, warps, stages = config
    m, k, n = y.shape[0], factor.shape[0] if rank_first else factor.shape[1], factor.shape[1] if rank_first else factor.shape[0]
    _lora_residual_at[(triton.cdiv(m, bm), triton.cdiv(n, bn))](y, factor, out, residual, residual if gate is None else gate, counter, first, step, span, m, n, k, y.stride(0),
                                                               0 if rank_first else factor.stride(0), factor.stride(0) if rank_first else 0, out.stride(0), residual.stride(0),
                                                               scale, gate is not None, rank_first, bm, bn, lora_triton.BK, num_warps=warps, num_stages=stages,
                                                               enable_fp_fusion=False)
    torch.autograd.graph.increment_version(residual)


_MOD_INTS = ["ROW0", "OFFSET", "SPAN", "SS", "SH"]


@triton.jit(do_not_specialize=_MOD_INTS, do_not_specialize_on_alignment=_MOD_INTS)
def _modulate_rows(X, SCALE, SHIFT, ROW0, OFFSET, SPAN, SS, SH, D, BITS: tl.constexpr, ROWS_BITS: tl.constexpr, BLOCK: tl.constexpr):
    """The segmented modulation of norm_modulate, one program per row and block of columns: row r of X takes modulation row
    ROW0 + (OFFSET + r) // SPAN (segments of SPAN rows), with the arithmetic of the per-segment
    kernels it replaced (BITS: BF16 as int16 bits, the modulation rows BF16 bits or FP32 rounded to BF16), element for element."""
    row = tl.program_id(0).to(tl.int64)
    mod_row = ROW0 + (OFFSET + row) // SPAN
    SCALE += mod_row * SS
    SHIFT += mod_row * SH
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    valid = col < D
    pointer = X + row * D + col
    if BITS:
        x = bf16_bits_to_fp32(tl.load(pointer, valid, 0))
        if ROWS_BITS:
            scale = bf16_bits_to_fp32(tl.load(SCALE + col, valid, 0))
            shift = bf16_bits_to_fp32(tl.load(SHIFT + col, valid, 0))
        else:
            scale = _round_bf16(tl.load(SCALE + col, valid, 0))
            shift = _round_bf16(tl.load(SHIFT + col, valid, 0))
        factor = _round_bf16(1.0 + scale)
        scaled = _round_bf16(x * factor)
        tl.store(pointer, fp32_to_bf16_bits(scaled + shift), valid)
    else:
        x = tl.load(pointer, valid, 0).to(tl.float32)
        scale = tl.load(SCALE + col, valid, 0).to(X.dtype.element_ty).to(tl.float32)
        shift = tl.load(SHIFT + col, valid, 0).to(X.dtype.element_ty).to(tl.float32)
        factor = (1.0 + scale).to(X.dtype.element_ty).to(tl.float32)
        scaled = (x * factor).to(X.dtype.element_ty).to(tl.float32)
        tl.store(pointer, scaled + shift, valid)


def segment_runs(segments):
    """segments (start, stop, row) of a segmented modulation as runs (start, stop, row0, offset, span), one launch each: consecutive
    segments of consecutive modulation rows, all span rows long but the first (its segment's last rows, offset rows in) and the last (its
    segment's first rows). Other segments are runs of their own."""
    chains = []  # segments in contiguous rows with consecutive modulation rows
    for segment in segments:
        if segment[1] <= segment[0]:
            continue
        if chains and segment[0] == chains[-1][-1][1] and segment[2] == chains[-1][-1][2] + 1:
            chains[-1].append(segment)
        else:
            chains.append([segment])
    runs = []
    for chain in chains:
        lengths = [stop - start for start, stop, _ in chain]
        span = max(lengths)
        if all(length == span for length in lengths[1:-1]):
            runs.append((chain[0][0], chain[-1][1], chain[0][2], span - lengths[0], span))
        else:
            runs += [(start, stop, row, 0, stop - start) for start, stop, row in chain]
    return runs


def _launch_modulate(target, scale, shift, row0, offset, span, bits):
    width = target.shape[-1]
    rows_bits = bits and scale.dtype == torch.bfloat16
    scale, shift = (scale.view(torch.int16), shift.view(torch.int16)) if rows_bits else (scale, shift)
    block = min(1024, triton.next_power_of_2(width))
    _modulate_rows[(target.shape[0], triton.cdiv(width, block))](target, scale, shift, row0, offset, span, scale.stride(0), shift.stride(0), width, bits, rows_bits,
                                                               block, enable_fp_fusion=False)


def modulate_runs(output, shift, scale, segments, bits):
    """norm_modulate's modulation of a normalized output in place (bits: denoiser_bits_triton's, else denoiser_triton's), one launch per
    run of segments (segment_runs) instead of one per segment."""
    target = output.view(torch.int16) if bits else output
    for start, stop, row0, offset, span in segment_runs(segments):
        _launch_modulate(target[start:stop], scale, shift, row0, offset, span, bits)
    torch.autograd.graph.increment_version(output)
    return output


def _norm_warps(block):
    return min(8, max(4, block // 2048))  # 4 warps up to 8192 columns: fastest or tied at every width measured (1536 to 6144)


_NORM_INTS = ["FIRST", "STEP", "LDX", "SPAN", "SS", "SH"]


@triton.jit(do_not_specialize=_NORM_INTS, do_not_specialize_on_alignment=_NORM_INTS)
def _rms_norm_modulate(X, Y, SCALE, SHIFT, COUNTER, FIRST, STEP, LDX, SPAN, SS, SH, EPS, D, BLOCK: tl.constexpr,
                       HAS_COUNTER: tl.constexpr, MODULATED: tl.constexpr):
    """Row r of Y = the weightless RMSNorm of row FIRST (+ counter * STEP) + r of X, times 1 + SCALE and plus SHIFT of the row's segment
    of SPAN rows (with MODULATED): FP32 from the loaded values to the one rounding of the store. D runtime, BLOCK its power-of-two range."""
    row = tl.program_id(0).to(tl.int64)
    source = FIRST + row
    if HAS_COUNTER:
        source += tl.load(COUNTER) * STEP
    col = tl.arange(0, BLOCK)
    valid = col < D
    x = tl.load(X + source * LDX + col, valid, 0).to(tl.float32)
    mean_square = tl.div_rn(tl.sum(x * x, 0), D.to(tl.float32))
    y = x * tl.div_rn(1.0, tl.sqrt_rn(mean_square + EPS))
    if MODULATED:
        segment = source // SPAN
        scale = tl.load(SCALE + segment * SS + col, valid, 0).to(tl.float32)
        shift = tl.load(SHIFT + segment * SH + col, valid, 0).to(tl.float32)
        y = y * (1.0 + scale) + shift
    tl.store(Y + row * D + col, y.to(Y.dtype.element_ty), valid)


def rms_norm_modulate(rows, eps, count, first=0, shift=None, scale=None, span=0, counter=None, step=0):
    """The weightless RMSNorm of the `count` rows of rows (R, D) from first (+ counter * step: a CUDA graph loop's turn), and with
    shift / scale (segments, D) x * (1 + scale[s]) + shift[s] for the segment s = row // span: one pass in FP32, rounded once, a new
    tensor (count, D) of rows' dtype. D at most 16384."""
    width = rows.shape[1]
    out = torch.empty((count, width), dtype=rows.dtype, device=rows.device)
    block = triton.next_power_of_2(width)
    modulated = shift is not None
    _rms_norm_modulate[(count,)](rows, out, scale if modulated else rows, shift if modulated else rows, rows if counter is None else counter, first, step,
                                 rows.stride(0), max(span, 1), scale.stride(0) if modulated else 0, shift.stride(0) if modulated else 0, eps, width, block,
                                 counter is not None, modulated, num_warps=_norm_warps(block))
    return out


@triton.jit(do_not_specialize=["LDX"], do_not_specialize_on_alignment=["LDX"])
def _rms_norm_rows_(X, WEIGHT, LDX, EPS, D, BLOCK: tl.constexpr, WEIGHTED: tl.constexpr):
    """Row r of X normalized in place by its RMS, times WEIGHT (with WEIGHTED): FP32 from the loaded values to the one rounding of the store.
    D runtime, BLOCK its power-of-two range."""
    row = tl.program_id(0).to(tl.int64)
    col = tl.arange(0, BLOCK)
    valid = col < D
    pointer = X + row * LDX + col
    x = tl.load(pointer, valid, 0).to(tl.float32)
    y = x * tl.div_rn(1.0, tl.sqrt_rn(tl.div_rn(tl.sum(x * x, 0), D.to(tl.float32)) + EPS))
    if WEIGHTED:
        y = y * tl.load(WEIGHT + col, valid, 0).to(tl.float32)
    tl.store(pointer, y.to(X.dtype.element_ty), valid)


def rms_norm_rows_(rows, eps, weight=None):
    """rows (R, D), unit column stride, RMS-normalized in place (times weight, D values): one pass, FP32, rounded once. D at most 16384."""
    block = triton.next_power_of_2(rows.shape[1])
    _rms_norm_rows_[(rows.shape[0],)](rows, rows if weight is None else weight, rows.stride(0), eps, rows.shape[1], block, weight is not None,
                                      num_warps=_norm_warps(block))
    torch.autograd.graph.increment_version(rows)
    return rows


@triton.jit(do_not_specialize=["FIRST", "STEP", "LDG", "HEADS"], do_not_specialize_on_alignment=["FIRST", "STEP", "LDG", "HEADS"])
def _gate_heads_at(Q, GATE, COUNTER, FIRST, STEP, LDG, HEADS, D, HB: tl.constexpr, DB: tl.constexpr):
    """One program per row: its HEADS * D values times the gate of their head, as an (HB, DB) tile of head and head-dim ranges (no
    division per element)."""
    row = tl.program_id(0).to(tl.int64)
    head, dim = tl.arange(0, HB), tl.arange(0, DB)
    valid = (head < HEADS)[:, None] & (dim < D)[None, :]
    gate = tl.load(GATE + (FIRST + tl.load(COUNTER) * STEP + row) * LDG + head, head < HEADS, 0).to(tl.float32)
    pointer = Q + row * HEADS * D + head[:, None] * D + dim[None, :]
    tl.store(pointer, (tl.load(pointer, valid, 0).to(tl.float32) * gate[:, None]).to(Q.dtype.element_ty), valid)


def gate_heads_at(query, gates, counter, first, step):
    """query (rows, heads, head_dim) contiguous, each head of each row times its gate: gates (R, heads) at rows first + counter * step (a
    CUDA graph loop's turn), the product rounded once as query.mul_(gates[rows].unsqueeze(-1)) rounds it."""
    rows, heads, head_dim = query.shape
    hb, db = triton.next_power_of_2(heads), triton.next_power_of_2(head_dim)
    _gate_heads_at[(rows,)](query, gates, counter, first, step, gates.stride(0), heads, head_dim, hb, db, num_warps=min(16, max(4, hb * db // 512)))
    torch.autograd.graph.increment_version(query)
    return query


from shared.kernels.triton_compilation_log import install_triton_compilation_logger
install_triton_compilation_logger()
