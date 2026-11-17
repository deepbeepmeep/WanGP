# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""H3 denoiser fusions of denoiser_triton.py for GPUs whose Triton has no BF16 conversion instructions (compute capability 7.5): BF16
tensors are passed as int16 views and converted by the bit helpers of bf16_bits.py (shift to FP32 bits; round to nearest even back,
as PyTorch's casts). Same operations and rounding points as the original kernels: the modulation is bit-identical to the eager path,
the Q/K RMSNorm + RoPE rounds once after the rotation (error against FP64 below the eager path's: 2.22e-3 vs 2.44e-3 at the H3 block
shape on an RTX 2080 Ti, specs/TURING_H3_FL2VA_AUDIT_20261021.md). ops.py selects them where the device is below the BF16 floor."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from shared.kernels.bf16_bits import bf16_bits_to_fp32, fp32_to_bf16_bits
from shared.kernels.denoiser_triton import _ROPE_CONFIGS, _TUNE_ROWS, _TUNER, _axis_ids, _axis_row, modulation_stride, scale_shift_block

BF16_ONE = tl.constexpr(0x3F80)


@triton.jit
def _round_bf16(v):
    return bf16_bits_to_fp32(fp32_to_bf16_bits(v))


def norm_modulate(x, norm, shift, scale, segments):
    """norm(x) (BF16), then per segment x * (1 + scale[row]) + shift[row] with the eager path's BF16 roundings, in place on the norm
    output; the modulation rows FP32 (rounded to BF16 as the eager cast) or BF16 (read as bits). One launch per run of consecutive
    segments (row_loop_triton.modulate_runs)."""
    from shared.kernels import row_loop_triton
    return row_loop_triton.modulate_runs(norm(x), shift, scale, segments, True)


@triton.jit(do_not_specialize=['ROWS', 'HEADS', 'SEQ', 'XB', 'XS', 'XH', 'RB', 'RS'], do_not_specialize_on_alignment=['ROWS', 'HEADS', 'SEQ', 'XB', 'XS', 'XH', 'RB', 'RS'])
def _rms_rope_bits(X, W, ROPE, ROWS, HEADS, SEQ, XB, XS, XH, RB, RS, EPS, RP: tl.constexpr, RC: tl.constexpr, D: tl.constexpr,
                   PAIRS: tl.constexpr, ROPE_BITS: tl.constexpr, BLOCK: tl.constexpr):
    # D and PAIRS stay constexprs: head-dim classes shared by models (as denoiser_triton._rms_rope)
    rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    ok = rows < ROWS
    head = rows % HEADS
    token = rows // HEADS
    seq = token % SEQ
    batch = token // SEQ
    c = tl.arange(0, D)
    base = batch * XB + seq * XS + head * XH
    ptr = X + base[:, None] + c[None, :]
    x = bf16_bits_to_fp32(tl.load(ptr, ok[:, None], 0))
    rotated = c < 2 * PAIRS
    partner = tl.where(c < PAIRS, c + PAIRS, tl.where(rotated, c - PAIRS, c))
    other = bf16_bits_to_fp32(tl.load(X + base[:, None] + partner[None, :], ok[:, None], 0))
    if D == 128:
        lanes = tl.reshape(tl.permute(tl.reshape(x, (BLOCK, 4, 32)), 0, 2, 1), (BLOCK, 32, 2, 2))
        even, odd = tl.split(lanes)
        x0, x2 = tl.split(even)
        x1, x3 = tl.split(odd)
        total = tl.sum(tl.fma(x3, x3, tl.fma(x2, x2, tl.fma(x1, x1, x0 * x0))), 1)
    else:
        total = tl.sum(x * x, 1)
    inv = libdevice.rsqrt(total / D + EPS)[:, None]
    w = bf16_bits_to_fp32(tl.load(W + c))[None, :]
    w_other = bf16_bits_to_fp32(tl.load(W + partner))[None, :]
    y = _round_bf16(x * inv * w)
    other = _round_bf16(other * inv * w_other)
    angle = batch[:, None] * RB + seq[:, None] * RS + tl.where(c < PAIRS, c, c - PAIRS)[None, :] * RP
    mask = ok[:, None] & rotated[None, :]
    if ROPE_BITS:  # a BF16 table, read as bits
        cos = bf16_bits_to_fp32(tl.load(ROPE + angle, mask, BF16_ONE))
        sin = bf16_bits_to_fp32(tl.load(ROPE + angle + RC, mask, 0))
    else:  # an FP32 table (ACE-Step 1.5), read as it is
        cos = tl.load(ROPE + angle, mask, 1.0)
        sin = tl.load(ROPE + angle + RC, mask, 0.0)
    first = c[None, :] < PAIRS
    out = tl.fma(tl.where(first, cos, sin), tl.where(first, y, other), tl.where(first, -sin, cos) * tl.where(first, other, y))
    tl.store(ptr, fp32_to_bf16_bits(tl.where(rotated[None, :], out, y)), ok[:, None])


def rms_rope_(x, weight, rope, eps):
    """In place on Q or K (batch, tokens, heads, head_dim) in BF16, any strides with contiguous channels; weight the BF16 RMSNorm
    weight; rope the BF16 or FP32 (1 | batch, tokens, 1, pairs, 2) cos / sin table of the split-half rotation of the first 2 * pairs
    channels (a BF16 table is read as bits; an FP32 one as it is: viewed as 16-bit words, its strides would not address its values)."""
    batch, seq, heads, dim = x.shape
    rope_batch = 0 if rope.shape[0] == 1 else rope.stride(0)
    table_bits = rope.dtype == torch.bfloat16
    x_bits, w_bits, rope_table = x.view(torch.int16), weight.view(torch.int16), rope.view(torch.int16) if table_bits else rope

    def launch(xx, config):
        block, warps = config
        rows = xx.shape[0] * xx.shape[1] * heads
        _rms_rope_bits[(triton.cdiv(rows, block),)](xx, w_bits, rope_table, rows, heads, xx.shape[1], *xx.stride()[:3], rope_batch, rope.stride(1), eps,
                                                    rope.stride(3), rope.stride(4), dim, rope.shape[-2], table_bits, block, num_warps=warps, enable_fp_fusion=False)

    def prepare_scratch():  # a copy of the first rows, in the head layout of x
        tokens = max(1, min(seq, _TUNE_ROWS // heads))
        scratch = x_bits.new_empty((1, tokens, heads, x.stride(2)))[..., :dim]
        scratch.copy_(x_bits[:1, :tokens])
        return lambda config: launch(scratch, config)

    # no head count in the identity: every Attention Head Split level (group size) reuses one launch configuration
    config = _TUNER.config("rms_rope_bits", f"{x.dtype}|d{dim}|p{rope.shape[-2]}|s{x.stride(2)}" + ("" if table_bits else "|fp32 table"), x.device, _ROPE_CONFIGS, prepare_scratch,
                           batch * seq * heads >= _TUNE_ROWS)
    launch(x_bits, config)
    torch.autograd.graph.increment_version(x)
    return x


@triton.jit(do_not_specialize=['N', 'MOD_STRIDE'], do_not_specialize_on_alignment=['N', 'MOD_STRIDE'])
def _scale_shift_bits(X, SCALE, SHIFT, Y, N, MOD_STRIDE, D, PER_TOKEN: tl.constexpr, BLOCK: tl.constexpr, DIVIDES: tl.constexpr):
    # denoiser_triton._scale_shift with BF16 x, scale, shift and y as bits: x * (1 + scale) + shift with the eager roundings
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    group = tl.program_id(1).to(tl.int64)
    offset = group * N + i
    if PER_TOKEN:
        s = i
    elif DIVIDES:
        s = group * MOD_STRIDE + (tl.program_id(0) % (D // BLOCK)) * BLOCK + tl.arange(0, BLOCK)
    else:
        s = group * MOD_STRIDE + i % D
    x = bf16_bits_to_fp32(tl.load(X + offset, i < N, 0))
    scale = bf16_bits_to_fp32(tl.load(SCALE + s, i < N, 0))
    shift = bf16_bits_to_fp32(tl.load(SHIFT + s, i < N, 0))
    factor = _round_bf16(1.0 + scale)
    value = _round_bf16(x * factor)
    tl.store(Y + offset, fp32_to_bf16_bits(value + shift), i < N)


def scale_shift(x, scale, shift, in_place):
    """denoiser_triton.scale_shift for BF16 x, scale and shift (same shapes; the modulation rows may be strided views): in place or into a new tensor."""
    out = x if in_place else torch.empty_like(x)
    per_token = scale.shape[1] == x.shape[1] and scale.is_contiguous()  # strided per-token rows go through the row groups
    groups = 1 if per_token else scale.shape[0] * scale.shape[1]
    size = x.numel() // groups
    block, divides = scale_shift_block(x.shape[2], per_token)
    _scale_shift_bits[(triton.cdiv(size, block), groups)](
        x.view(torch.int16), scale.view(torch.int16), shift.view(torch.int16), out.view(torch.int16), size, modulation_stride(scale), x.shape[2],
        per_token, block, divides,
        enable_fp_fusion=False)
    if in_place:
        torch.autograd.graph.increment_version(out)
    return out


@triton.jit(do_not_specialize=['N', 'TOKENS', 'WIDTH', 'G0', 'G1', 'G2', 'BATCH_STRIDE', 'HEAD_START', 'PAD', 'FREQ_WIDTH'], do_not_specialize_on_alignment=['N', 'TOKENS', 'WIDTH', 'G0', 'G1', 'G2', 'BATCH_STRIDE', 'HEAD_START', 'PAD', 'FREQ_WIDTH'])
def _split_rope_bits(X, C0, S0, C1, S1, C2, S2, N, TOKENS,
                     WIDTH, HALF: tl.constexpr, PAD,
                     AXES: tl.constexpr, G0, G1, G2, A0: tl.constexpr, A1: tl.constexpr, A2: tl.constexpr,
                     FREQ_WIDTH, BATCH_STRIDE, HEAD_START,
                     FP32: tl.constexpr, FREQ_BITS: tl.constexpr, BLOCK: tl.constexpr):
    # denoiser_triton._split_rope with BF16 x as bits (in place) and, with FREQ_BITS, BF16 cos / sin tables as bits
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    f = i % (WIDTH // 2)
    token = i // (WIDTH // 2) % TOKENS
    batch = i // (WIDTH // 2) // TOKENS
    head, local = f // HALF, f % HALF
    offset = batch * BATCH_STRIDE + token * WIDTH + head * HALF * 2 + local
    f = f + HEAD_START * HALF
    axis = (f - PAD) % AXES
    col = (f - PAD) // AXES
    c = tl.full((BLOCK,), 1.0, tl.float32)
    s = tl.full((BLOCK,), 0.0, tl.float32)
    for a in tl.static_range(AXES):
        valid = (i < N) & (f >= PAD) & (axis == a)
        if a == 0:
            row = _axis_row(token, A0, G0, G1, G2)
            cr = tl.load(C0 + row * FREQ_WIDTH + col, valid, 0)
            sr = tl.load(S0 + row * FREQ_WIDTH + col, valid, 0)
        elif a == 1:
            row = _axis_row(token, A1, G0, G1, G2)
            cr = tl.load(C1 + row * FREQ_WIDTH + col, valid, 0)
            sr = tl.load(S1 + row * FREQ_WIDTH + col, valid, 0)
        else:
            row = _axis_row(token, A2, G0, G1, G2)
            cr = tl.load(C2 + row * FREQ_WIDTH + col, valid, 0)
            sr = tl.load(S2 + row * FREQ_WIDTH + col, valid, 0)
        if FREQ_BITS:
            cv, sv = bf16_bits_to_fp32(cr), bf16_bits_to_fp32(sr)
        else:
            cv, sv = cr.to(tl.float32), sr.to(tl.float32)
        c = tl.where(valid, cv, c)
        s = tl.where(valid, sv, s)
    x0 = bf16_bits_to_fp32(tl.load(X + offset, i < N, 0))
    x1 = bf16_bits_to_fp32(tl.load(X + offset + HALF, i < N, 0))
    a, b = libdevice.mul_rn(x0, c), libdevice.mul_rn(x1, c)
    if not FP32:
        a = _round_bf16(a)
        b = _round_bf16(b)
    valid = (i < N) & (((head + HEAD_START) % AXES != 0) | (local >= PAD))
    tl.store(X + offset, fp32_to_bf16_bits(libdevice.sub_rn(a, libdevice.mul_rn(x1, s))), valid)
    tl.store(X + offset + HALF, fp32_to_bf16_bits(libdevice.add_rn(b, libdevice.mul_rn(x0, s))), valid)


def split_rope(x, cache, layout, head_start=0, dim_head=None):
    """denoiser_triton.split_rope for a BF16 x (in place); the cos / sin tables FP32 (cache.use_fp32_freqs) or BF16, read as bits."""
    cos, sin = layout.axis_cos, layout.axis_sin
    bits = not cache.use_fp32_freqs
    tables = tuple(t.view(torch.int16) if bits else t for a in range(3) for t in (cos[min(a, len(cos) - 1)], sin[min(a, len(sin) - 1)]))
    size = x.numel() // 2
    return _split_rope_bits[(triton.cdiv(size, 256),)](
        x.view(torch.int16), *tables, size, x.shape[1], x.shape[2], (dim_head or x.shape[2] // cache.num_attention_heads) // 2,
        cache.pad_size, len(cos), *(layout.grid_sizes + (1,) * (3 - len(layout.grid_sizes))), *_axis_ids(cache.rope_axes), cos[0].shape[1],
        x.stride(0), head_start, cache.use_fp32_freqs, bits, 256, enable_fp_fusion=False)


from shared.kernels.triton_compilation_log import install_triton_compilation_logger  # noqa: E402

install_triton_compilation_logger()
