# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Triton kernels for GPUs that run Triton without BF16 conversion instructions (Turing, sm_75): BF16 tensors are read and written as
their bits (shift into FP32 bits; round to nearest even as PyTorch's casts, NaN -> 0x7FC0). For the Qwen3.5/3.8 language models:
- the norms and the activation, which otherwise run as chains of small PyTorch kernels there. The norms round once to BF16 (the PyTorch
  route rounds each FP32 step, then to BF16): RMSNorm keeps sums and products exact as FP32 pairs (error-free transformations; Turing
  runs FP64 at 1/32 of the FP32 rate) with the inverse square root in FP64, the gated norm of 128 values computes in FP64; SiLU*mul
  reproduces the PyTorch route bit for bit;
- the paged Q8 attention of decoding and speculative verification, on FP16 tensor cores (sm_75 has no BF16 ones), each K/V page read
  once for the query heads of a KV head and all the queries of a verification, every row computed alike whatever the query count.
Also helpers for kernels written once for every GPU (the row kernels of Inkling 2): to_float / round_bf16 / convert take BF16 or its
bits, and portable_supported / reads_bits / bits_view choose them on the host."""
import functools

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError:
    triton = None


@functools.cache
def eligible() -> bool:
    """Validated on compute capability 7.5 only; other GPUs keep their routes."""
    return triton is not None and torch.cuda.is_available() and torch.version.hip is None and torch.cuda.get_device_capability() == (7, 5)


def applies(x) -> bool:
    """Whether a kernel reading x takes these kernels in place of a native Triton one: a BF16 CUDA tensor on a GPU where they are
    validated (eligible), whose Triton has no BF16 conversions."""
    return x.dtype == torch.bfloat16 and x.is_cuda and eligible()


if triton is not None:
    @triton.jit
    def bf16_bits_to_fp32(bits):
        return (bits.to(tl.int32) << 16).to(tl.float32, bitcast=True)

    @triton.jit
    def fp32_to_bf16_bits(x):
        bits = x.to(tl.uint32, bitcast=True)
        rounded = tl.where(x != x, 0x7FC0, (bits + 0x7FFF + ((bits >> 16) & 1)) >> 16)
        return rounded.to(tl.uint16).to(tl.int16, bitcast=True)

    @triton.jit
    def fp64_to_bf16_bits(x):
        # correctly rounded: to FP32 rounded to odd (truncated, lowest bit set when inexact), then to nearest even BF16
        f = x.to(tl.float32)
        back = f.to(tl.float64)
        bits = f.to(tl.uint32, bitcast=True)
        bits = tl.where(back != x, tl.where(tl.abs(back) > tl.abs(x), bits - 1, bits) | 1, bits)
        return fp32_to_bf16_bits(bits.to(tl.float32, bitcast=True))

    @triton.jit
    def _two_sum(a, b):
        s = a + b
        bb = s - a
        return s, (a - (s - bb)) + (b - bb)

    @triton.jit
    def _add_pairs(a_hi, a_lo, b_hi, b_lo):
        s, e = _two_sum(a_hi, b_hi)
        return s, e + (a_lo + b_lo)

    @triton.jit
    def pair_to_bf16_bits(hi, lo):
        # hi + lo (|lo| below an ulp of hi) correctly rounded: renormalized, rounded to odd in FP32, then to nearest even BF16
        s, e = _two_sum(hi, lo)
        e = tl.where(tl.abs(s) < float("inf"), e, 0.0)
        bits = s.to(tl.uint32, bitcast=True)
        bits = tl.where(e != 0, tl.where((e < 0) != (s < 0), bits - 1, bits) | 1, bits)
        return fp32_to_bf16_bits(bits.to(tl.float32, bitcast=True))

    @triton.jit
    def _load_bf16_fp64(PTR, mask):
        return bf16_bits_to_fp32(tl.load(PTR, mask, 0)).to(tl.float64)

    @triton.jit
    def _weight(W, offs, mask, W_BITS: tl.constexpr):
        if W_BITS:
            return bf16_bits_to_fp32(tl.load(W + offs, mask, 0))
        return tl.load(W + offs, mask, 0).to(tl.float32)

    @triton.jit
    def _inverse_rms(x, n_cols, eps):
        mean = libdevice.div_rn(tl.sum(x * x, axis=0), n_cols.to(tl.float64))
        return libdevice.div_rn(tl.full((), 1.0, tl.float64), libdevice.sqrt(mean + eps))

    @triton.jit(do_not_specialize=("x_stride", "r_stride", "y_stride", "ro_stride"))
    def _rms_norm_kernel(X, RES, W, Y, RES_OUT, x_stride, r_stride, y_stride, ro_stride, n_cols, eps,
                         HAS_RESIDUAL: tl.constexpr, W_BITS: tl.constexpr, BLOCK: tl.constexpr):
        # RMSNorm._fallback_rms_forward / _fallback_add_rms_forward: (x [+ residual]) * rsqrt(mean(x^2) + eps) * weight, with
        # x [+ residual] = x + lo exactly and the products as (value, error) pairs; launched without FMA contraction
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK)
        mask = offs < n_cols
        x = bf16_bits_to_fp32(tl.load(X + row * x_stride + offs, mask, 0))
        lo = tl.zeros_like(x)
        if HAS_RESIDUAL:
            x, lo = _two_sum(x, bf16_bits_to_fp32(tl.load(RES + row * r_stride + offs, mask, 0)))
            lo = tl.where(tl.abs(x) < float("inf"), lo, 0.0)
            tl.store(RES_OUT + row * ro_stride + offs, pair_to_bf16_bits(x, lo), mask)
        # squares scaled by a power of two from the largest magnitude (exact; no FP32 overflow), as exact pairs plus the cross term
        top = (tl.max(tl.abs(x), axis=0).to(tl.uint32, bitcast=True) >> 23).to(tl.int32)
        top = tl.minimum(tl.maximum(top, 1), 253)
        scale = ((254 - top) << 23).to(tl.float32, bitcast=True)
        xs, ls = x * scale, lo * scale
        square = xs * xs
        total, total_lo = tl.reduce((square, tl.fma(xs, xs, -square) + 2 * xs * ls), 0, _add_pairs)
        total_lo = tl.where(tl.abs(total) < float("inf"), total_lo, 0.0)
        unscale = ((top + 896).to(tl.int64) << 52).to(tl.float64, bitcast=True)  # 2^(top - 127), squared below
        mean = libdevice.div_rn((total.to(tl.float64) + total_lo.to(tl.float64)) * unscale * unscale, n_cols.to(tl.float64))
        inverse = libdevice.div_rn(tl.full((), 1.0, tl.float64), libdevice.sqrt(mean + eps))
        inverse_hi = inverse.to(tl.float32)
        inverse_lo = (inverse - inverse_hi.to(tl.float64)).to(tl.float32)
        a = x * inverse_hi  # the normalized value first, as the PyTorch route: no subnormal or overflowing intermediate
        a_lo = tl.fma(x, inverse_hi, -a) + (x * inverse_lo + lo * inverse_hi)
        w = _weight(W, offs, mask, W_BITS)
        y = a * w
        tl.store(Y + row * y_stride + offs, pair_to_bf16_bits(y, tl.fma(a, w, -y) + a_lo * w), mask)

    @triton.jit(do_not_specialize=("x_stride", "g_stride"))
    def _gated_rms_norm_kernel(X, G, W, Y, x_stride, g_stride, n_cols, eps, W_BITS: tl.constexpr, BLOCK: tl.constexpr):
        # Qwen3_5RMSNormGated.forward_list: x * rsqrt(mean(x^2) + eps) * weight * silu(gate), with silu(gate) in FP32 as F.silu
        row = tl.program_id(0).to(tl.int64)
        offs = tl.arange(0, BLOCK)
        mask = offs < n_cols
        x = _load_bf16_fp64(X + row * x_stride + offs, mask)
        gate = bf16_bits_to_fp32(tl.load(G + row * g_stride + offs, mask, 0))
        silu = tl.div_rn(gate, 1.0 + libdevice.exp(-gate)).to(tl.float64)
        y = x * _weight(W, offs, mask, W_BITS).to(tl.float64) * _inverse_rms(x, n_cols, eps) * silu
        tl.store(Y + row * n_cols + offs, fp64_to_bf16_bits(y), mask)

    @triton.jit(do_not_specialize=("ROWS",), do_not_specialize_on_alignment=("ROWS",))
    def _silu_mul_kernel(X, OUT, ROWS, COLUMNS, R: tl.constexpr, BLOCK: tl.constexpr):
        # silu_mul's PyTorch route on BF16: silu(gate) rounded to BF16 (in place), then the product rounded. Tiles of R rows and BLOCK
        # columns with the width at runtime: one compile per tile shape for every FFN width
        rows = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
        columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        valid = (rows < ROWS)[:, None] & (columns < COLUMNS)[None, :]
        row, column = rows[:, None], columns[None, :]
        source = row * (2 * COLUMNS) + column
        gate = bf16_bits_to_fp32(tl.load(X + source, valid, 0))
        value = bf16_bits_to_fp32(tl.load(X + source + COLUMNS, valid, 0))
        activated = bf16_bits_to_fp32(fp32_to_bf16_bits(tl.div_rn(gate, 1.0 + libdevice.exp(-gate))))
        tl.store(OUT + row * COLUMNS + column, fp32_to_bf16_bits(activated * value), valid)

    @triton.jit
    def _power_of_two(top, offset: tl.constexpr):
        # 2^(offset - top + 127) for a biased FP32 exponent top: 2^(offset - unbiased exponent)
        return ((offset + 254 - top) << 23).to(tl.float32, bitcast=True)

    @triton.jit
    def _fp16_pair(x):
        hi = x.to(tl.float16)
        return hi, (x - hi.to(tl.float32)).to(tl.float16)

    @triton.jit
    def _exponent(x):
        return (x.to(tl.uint32, bitcast=True) >> 23).to(tl.int32)

    @triton.jit
    def _query_block(Q, row_offsets, valid_rows, scale, b: tl.constexpr):
        dims = b * 32 + tl.arange(0, 32)
        return (bf16_bits_to_fp32(tl.load(Q + row_offsets[:, None] + dims[None, :], mask=valid_rows[:, None], other=0)) * scale[:, None]).to(tl.float16)

    @triton.jit
    def _scores_block(q, K, KS, offsets, scale_offsets, valid_columns, b: tl.constexpr):
        # int8 keys are exact in FP16; their scale (one per 32 values) multiplies the block's dot products
        k = tl.load(K + offsets[None, :] + b * 32 + tl.arange(0, 32)[:, None], mask=valid_columns[None, :], other=0).to(tl.float16)
        return tl.dot(q, k) * tl.load(KS + scale_offsets + b, mask=valid_columns, other=0).to(tl.float32)[None, :]

    @triton.jit
    def _values_block(acc, p, correction, V, VS, offsets, scale_offsets, valid_columns, b: tl.constexpr):
        # probabilities times the value scales of the block, scaled per row to [1, 2) and split in FP16 (value, remainder)
        weights = p * tl.load(VS + scale_offsets + b, mask=valid_columns, other=0).to(tl.float32)[None, :]
        top = tl.maximum(_exponent(tl.max(weights, 1)), 1)
        hi, lo = _fp16_pair(weights * _power_of_two(top, 0)[:, None])
        v = tl.load(V + offsets[:, None] + b * 32 + tl.arange(0, 32)[None, :], mask=valid_columns[:, None], other=0).to(tl.float16)
        weighted = tl.dot(hi, v)
        weighted = tl.dot(lo, v, weighted)
        return acc * correction[:, None] + weighted * _power_of_two(254 - top, 0)[:, None]

    @triton.jit
    def _store_block(PARTIAL, index, valid_rows, acc, b: tl.constexpr):
        tl.store(PARTIAL + index[:, None] * 256 + b * 32 + tl.arange(0, 32)[None, :], acc, mask=valid_rows[:, None])

    @triton.jit(do_not_specialize=("Q_PER_SEQ", "TABLE_WIDTH", "H_Q", "H_KV"), do_not_specialize_on_alignment=("Q_PER_SEQ", "TABLE_WIDTH", "H_Q", "H_KV"))
    def _q8_attention_partials_kernel(Q, K, V, KS, VS, TABLES, LENGTHS, PARTIAL, MAXIMUM, SUM, Q_PER_SEQ, TABLE_WIDTH,
                                      H_Q, H_KV, PAGE: tl.constexpr, SPLITS: tl.constexpr,
                                      M: tl.constexpr, N: tl.constexpr, SCALE):
        # H_Q, H_KV and SCALE at runtime: one compile for every model's head counts and attention scale
        # Paged Q8 attention (head_dim 256) of the query heads sharing a KV head and of all the queries of a sequence (decode or
        # verification rows, causal): each K/V page is read once per split. FP16 tensor-core dots with FP32 accumulation on the
        # int8 values (exact in FP16), one per block of 32 values, the FP16 scales applied to the block results: queries scaled
        # by a power of two that keeps their BF16 values exact in FP16, probabilities times value scales as FP16 (value,
        # remainder) pairs scaled per row. Tiles and splits do not depend on the query count.
        sequence, kv_head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        split, query_block = tile % SPLITS, tile // SPLITS
        group = H_Q // H_KV
        rows = query_block * M + tl.arange(0, M)
        row_query = rows // group  # the row's query within the sequence, once (the key tiles below reuse it)
        query_index = sequence * Q_PER_SEQ + row_query
        query_head = kv_head * group + rows % group
        valid_rows = rows < Q_PER_SEQ * group
        row_offsets = (query_index * H_Q + query_head) * 256
        q = bf16_bits_to_fp32(tl.load(Q + row_offsets[:, None] + tl.arange(0, 256)[None, :], mask=valid_rows[:, None], other=0))
        top = tl.minimum(tl.maximum(_exponent(tl.max(tl.abs(q), 1)), 15), 253)
        q_scale = _power_of_two(top, 14)  # largest |q| of the row in [2^14, 2^15)
        row_scale = SCALE * _power_of_two(254 - top, -14)  # 2^(top - 141) undoes it
        q0 = _query_block(Q, row_offsets, valid_rows, q_scale, 0)
        q1 = _query_block(Q, row_offsets, valid_rows, q_scale, 1)
        q2 = _query_block(Q, row_offsets, valid_rows, q_scale, 2)
        q3 = _query_block(Q, row_offsets, valid_rows, q_scale, 3)
        q4 = _query_block(Q, row_offsets, valid_rows, q_scale, 4)
        q5 = _query_block(Q, row_offsets, valid_rows, q_scale, 5)
        q6 = _query_block(Q, row_offsets, valid_rows, q_scale, 6)
        q7 = _query_block(Q, row_offsets, valid_rows, q_scale, 7)
        context_len = tl.load(LENGTHS + sequence)
        prefix_len = context_len - Q_PER_SEQ
        tiles_per_split = tl.cdiv(tl.cdiv(context_len, N), SPLITS)
        begin = split * tiles_per_split * N
        end = tl.minimum(context_len, (split + 1) * tiles_per_split * N)
        maximum = tl.full((M,), -1.0e20, tl.float32)
        denominator = tl.zeros((M,), tl.float32)
        acc0 = tl.zeros((M, 32), tl.float32)
        acc1 = tl.zeros((M, 32), tl.float32)
        acc2 = tl.zeros((M, 32), tl.float32)
        acc3 = tl.zeros((M, 32), tl.float32)
        acc4 = tl.zeros((M, 32), tl.float32)
        acc5 = tl.zeros((M, 32), tl.float32)
        acc6 = tl.zeros((M, 32), tl.float32)
        acc7 = tl.zeros((M, 32), tl.float32)
        for start in tl.range(begin, end, N):
            columns = start + tl.arange(0, N)
            valid_columns = columns < end
            page = tl.load(TABLES + sequence * TABLE_WIDTH + columns // PAGE, mask=valid_columns, other=0)
            slot = (page * PAGE + columns % PAGE).to(tl.int64)
            offsets = (slot * H_KV + kv_head) * 256
            scale_offsets = (slot * H_KV + kv_head) * 8
            scores = _scores_block(q0, K, KS, offsets, scale_offsets, valid_columns, 0)
            scores += _scores_block(q1, K, KS, offsets, scale_offsets, valid_columns, 1)
            scores += _scores_block(q2, K, KS, offsets, scale_offsets, valid_columns, 2)
            scores += _scores_block(q3, K, KS, offsets, scale_offsets, valid_columns, 3)
            scores += _scores_block(q4, K, KS, offsets, scale_offsets, valid_columns, 4)
            scores += _scores_block(q5, K, KS, offsets, scale_offsets, valid_columns, 5)
            scores += _scores_block(q6, K, KS, offsets, scale_offsets, valid_columns, 6)
            scores += _scores_block(q7, K, KS, offsets, scale_offsets, valid_columns, 7)
            valid = valid_rows[:, None] & valid_columns[None, :] & (columns[None, :] <= prefix_len + row_query[:, None])
            scores = tl.where(valid, scores * row_scale[:, None], -1.0e20)
            new_maximum = tl.maximum(maximum, tl.max(scores, 1))
            correction = tl.exp2((maximum - new_maximum) * 1.4426950408889634)
            p = tl.where(valid, tl.exp2((scores - new_maximum[:, None]) * 1.4426950408889634), 0.0)
            denominator = denominator * correction + tl.sum(p, 1)
            acc0 = _values_block(acc0, p, correction, V, VS, offsets, scale_offsets, valid_columns, 0)
            acc1 = _values_block(acc1, p, correction, V, VS, offsets, scale_offsets, valid_columns, 1)
            acc2 = _values_block(acc2, p, correction, V, VS, offsets, scale_offsets, valid_columns, 2)
            acc3 = _values_block(acc3, p, correction, V, VS, offsets, scale_offsets, valid_columns, 3)
            acc4 = _values_block(acc4, p, correction, V, VS, offsets, scale_offsets, valid_columns, 4)
            acc5 = _values_block(acc5, p, correction, V, VS, offsets, scale_offsets, valid_columns, 5)
            acc6 = _values_block(acc6, p, correction, V, VS, offsets, scale_offsets, valid_columns, 6)
            acc7 = _values_block(acc7, p, correction, V, VS, offsets, scale_offsets, valid_columns, 7)
            maximum = new_maximum
        index = (query_index * H_Q + query_head) * SPLITS + split
        _store_block(PARTIAL, index, valid_rows, acc0, 0)
        _store_block(PARTIAL, index, valid_rows, acc1, 1)
        _store_block(PARTIAL, index, valid_rows, acc2, 2)
        _store_block(PARTIAL, index, valid_rows, acc3, 3)
        _store_block(PARTIAL, index, valid_rows, acc4, 4)
        _store_block(PARTIAL, index, valid_rows, acc5, 5)
        _store_block(PARTIAL, index, valid_rows, acc6, 6)
        _store_block(PARTIAL, index, valid_rows, acc7, 7)
        tl.store(MAXIMUM + index, maximum, mask=valid_rows)
        tl.store(SUM + index, denominator, mask=valid_rows)

    @triton.jit
    def _q8_attention_reduce_kernel(PARTIAL, MAXIMUM, SUM, OUT, SPLITS: tl.constexpr, D: tl.constexpr):
        row = tl.program_id(0)
        splits = tl.arange(0, SPLITS)
        dims = tl.arange(0, D)
        maximum = tl.load(MAXIMUM + row * SPLITS + splits)
        weights = tl.exp2((maximum - tl.max(maximum, 0)) * 1.4426950408889634)
        denominator = tl.sum(tl.load(SUM + row * SPLITS + splits) * weights, 0)
        partial = tl.load(PARTIAL + (row * SPLITS + splits[:, None]) * D + dims[None, :])
        tl.store(OUT + row * D + dims, fp32_to_bf16_bits(tl.div_rn(tl.sum(partial * weights[:, None], 0), tl.maximum(denominator, 1.0e-20))))


if triton is not None:
    # Kernels written once for every GPU: their BF16 tensors arrive as BF16, or as int16 bits where Triton has no BF16 conversions
    # (reads_bits), and these helpers choose at compile time. With BF16 tensors they are the plain conversions.
    @triton.jit
    def to_float(x):
        """float32 of loaded values: BF16 bits (int16) shifted into place, any other type converted."""
        if x.dtype == tl.int16:
            return bf16_bits_to_fp32(x)
        return x.to(tl.float32)

    @triton.jit
    def round_bf16(x, BITS: tl.constexpr):
        """float32 x rounded to BF16, as float32."""
        if BITS:
            return bf16_bits_to_fp32(fp32_to_bf16_bits(x))
        return x.to(tl.bfloat16).to(tl.float32)

    @triton.jit
    def convert(x, pointer):
        """x in the element type of pointer, int16 on either side standing for BF16 bits."""
        if (x.dtype == tl.int16) == (pointer.dtype.element_ty == tl.int16):
            return x.to(pointer.dtype.element_ty)
        elif x.dtype == tl.int16:
            return bf16_bits_to_fp32(x).to(pointer.dtype.element_ty)
        else:
            return fp32_to_bf16_bits(x.to(tl.float32))


HIP_ARCHS = ("gfx942", "gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151", "gfx1200", "gfx1201")  # AMD GPUs for which Triton compiles the kernels written with the helpers above


@functools.cache
def reads_bits(device) -> bool:
    """Whether BF16 tensors reach Triton kernels on this CUDA device as their int16 bits: NVIDIA GPUs below compute capability 8.0, whose
    Triton (3.2 for Turing) has no BF16 conversions."""
    return torch.version.hip is None and torch.cuda.get_device_capability(device) < (8, 0)


@functools.cache
def portable_supported(device) -> bool:
    """Whether the kernels written with the helpers above run on device: NVIDIA GPUs from compute capability 7.5 (BF16 bits below 8.0),
    the AMD architectures of HIP_ARCHS."""
    device = torch.device(device)
    if device.type != "cuda" or triton is None:
        return False
    if torch.version.hip is not None:
        return torch.cuda.get_device_properties(device).gcnArchName.split(":")[0] in HIP_ARCHS
    return torch.cuda.get_device_capability(device) >= (7, 5)


def bits_view(tensor, bits):
    """A BF16 tensor as its int16 bits when bits (reads_bits), any other tensor as it is."""
    return tensor.view(torch.int16) if bits and tensor.dtype == torch.bfloat16 else tensor


def _bits(tensor):
    return tensor.view(torch.int16) if tensor.dtype == torch.bfloat16 else tensor


def rms_norm(x, residual, weight, eps):
    """BF16 x (and residual) with unit last stride: y, or (y, x + residual) as RMSNorm.forward."""
    x_2d = x.reshape(-1, x.shape[-1])
    residual_2d = x_2d if residual is None else residual.reshape(-1, residual.shape[-1])
    y = torch.empty(x_2d.shape, device=x.device, dtype=x.dtype)
    residual_out = torch.empty_like(y) if residual is not None else y
    _rms_norm_kernel[(x_2d.shape[0],)](_bits(x_2d), _bits(residual_2d), _bits(weight), _bits(y), _bits(residual_out),
                                       x_2d.stride(0), residual_2d.stride(0), y.stride(0), residual_out.stride(0), x_2d.shape[1], eps,
                                       residual is not None, weight.dtype == torch.bfloat16, triton.next_power_of_2(x_2d.shape[1]),
                                       num_warps=8 if x_2d.shape[1] > 1024 else 1, enable_fp_fusion=False)
    if residual is None:
        return y.view_as(x)
    return y.view_as(x), residual_out.view_as(x)


def gated_rms_norm(x, gate, weight, eps):
    """BF16 x and gate of shape (rows, dim) with unit last stride, as Qwen3_5RMSNormGated.forward_list."""
    y = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    _gated_rms_norm_kernel[(x.shape[0],)](_bits(x), _bits(gate), _bits(weight), _bits(y), x.stride(0), gate.stride(0), x.shape[1], eps,
                                          weight.dtype == torch.bfloat16, triton.next_power_of_2(x.shape[1]), num_warps=1)
    return y


def silu_mul(x):
    """BF16 x = [gate, value] on the last dimension: silu(gate) * value."""
    x = x.contiguous()
    output = torch.empty((*x.shape[:-1], x.shape[-1] // 2), device=x.device, dtype=x.dtype)
    columns = output.shape[-1]
    rows = output.numel() // columns
    block = min(1024, triton.next_power_of_2(columns))
    tile_rows = 1024 // block  # narrow rows: several per program
    _silu_mul_kernel[(triton.cdiv(rows, tile_rows), triton.cdiv(columns, block))](_bits(x), _bits(output), rows, columns, tile_rows, block)
    return output


Q8_ATTENTION_SPLITS = 32  # context splits per KV head: fixed, so CUDA graphs replay at any context length and every query count reads alike
Q8_ATTENTION_ROWS = 16  # query rows per tile (query heads of a KV head times queries): decode and verification rows compute alike
Q8_ATTENTION_KEYS = 64  # keys per tile (16 x 64 tiles on 2 warps: fastest 5-row verification on sm_75, faster than llama.cpp's adapter for 1 row)


def q8_paged_attention(q, k_cache, v_cache, k_scale, v_scale, block_tables, context_lens, softmax_scale):
    """BF16 q (sequences * queries, heads, 256) against the paged Q8 cache (int8 values, FP16 scales per 32 values): BF16 output of q's
    shape. context_lens counts each sequence's keys including its queries, which attend causally (decode: one query)."""
    queries_per_sequence = q.shape[0] // context_lens.numel()
    query_blocks = triton.cdiv(queries_per_sequence * (q.shape[1] // k_cache.shape[2]), Q8_ATTENTION_ROWS)
    partial = torch.empty((*q.shape[:2], Q8_ATTENTION_SPLITS, q.shape[2]), device=q.device, dtype=torch.float32)
    maximum = torch.empty((*q.shape[:2], Q8_ATTENTION_SPLITS), device=q.device, dtype=torch.float32)
    denominator = torch.empty_like(maximum)
    output = torch.empty_like(q)
    _q8_attention_partials_kernel[(context_lens.numel(), k_cache.shape[2], query_blocks * Q8_ATTENTION_SPLITS)](
        _bits(q), k_cache, v_cache, k_scale, v_scale, block_tables, context_lens, partial, maximum, denominator, queries_per_sequence,
        block_tables.shape[1], q.shape[1], k_cache.shape[2], k_cache.shape[1], Q8_ATTENTION_SPLITS, Q8_ATTENTION_ROWS, Q8_ATTENTION_KEYS,
        softmax_scale, num_warps=2, num_stages=1)
    _q8_attention_reduce_kernel[(q.shape[0] * q.shape[1],)](partial, maximum, denominator, _bits(output), Q8_ATTENTION_SPLITS, q.shape[2], num_warps=4)
    return output


# Register after definitions to preserve Triton's line-number-sensitive cache keys.
if triton is not None:
    from shared.kernels.triton_compilation_log import install_triton_compilation_logger
    install_triton_compilation_logger()
