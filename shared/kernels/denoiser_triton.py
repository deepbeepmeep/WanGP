"""H3 denoiser fusions: modulation (native RMSNorm and BF16 rounding boundaries kept) and Q/K RMSNorm + RoPE."""
import triton
import triton.language as tl


@triton.jit(do_not_specialize=['N'], do_not_specialize_on_alignment=['N'])
def _modulate(X, SCALE, SHIFT, N, D: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    valid = i < N
    if D % BLOCK == 0:
        col = (tl.program_id(0) % (D // BLOCK)) * BLOCK + tl.arange(0, BLOCK)
    else:
        col = i % D
    x = tl.load(X + i, valid, 0).to(tl.float32)
    scale = tl.load(SCALE + col, valid, 0).to(X.dtype.element_ty).to(tl.float32)
    shift = tl.load(SHIFT + col, valid, 0).to(X.dtype.element_ty).to(tl.float32)
    factor = (1.0 + scale).to(X.dtype.element_ty).to(tl.float32)
    scaled = (x * factor).to(X.dtype.element_ty).to(tl.float32)
    tl.store(X + i, scaled + shift, valid)


def norm_modulate(x, norm, shift, scale, segments, dtype=None):
    output = norm(x)
    if dtype is not None:
        output = output.to(dtype)
    width = x.shape[-1]
    for start, stop, row in segments:
        if stop > start:
            size = (stop-start) * width
            _modulate[(triton.cdiv(size, 1024),)](
                output[start:stop], scale[row], shift[row], size, width, 1024,
                enable_fp_fusion=False)
    torch.autograd.graph.increment_version(output)
    return output


# Below the modulation kernel, whose line numbers are part of Triton's cache keys.
import torch
from triton.language.extra import libdevice
from shared.kernels.triton_tuner import LaunchTuner

_TUNER = LaunchTuner("wan2gp_h3_denoiser_kernels", 3, "H3")
_ROPE_CONFIGS = ((8, 1), (16, 2), (16, 4), (32, 4), (64, 8))
_TUNE_ROWS = 1 << 17  # cap tuning scratch
_ROPE_RUNTIME = ['ROWS', 'HEADS', 'SEQ', 'XB', 'XS', 'XH', 'RB', 'RS', 'RH', 'WH']


@triton.jit(do_not_specialize=_ROPE_RUNTIME, do_not_specialize_on_alignment=_ROPE_RUNTIME)
def _rms_rope(X, W, ROPE, ROWS, HEADS, SEQ, XB, XS, XH, RB, RS, RH, WH, EPS, RP: tl.constexpr, RC: tl.constexpr, D: tl.constexpr,
              PAIRS: tl.constexpr, BLOCK: tl.constexpr, INTERLEAVED: tl.constexpr, ROPE_FIRST: tl.constexpr, PER_HEAD: tl.constexpr,
              DP: tl.constexpr):
    """Rows (batch, token, head) of Q or K, loaded whole (DP: D rounded up to a power of two): weighted RMSNorm in FP32 rounded to the
    activation dtype (the norm output), then the rotation of the first 2 * PAIRS channels in FP32 with one final rounding (cos / sin
    pairs of ROPE): channel p with p + PAIRS (split halves) or, INTERLEAVED, 2p with 2p + 1 (a complex table viewed as real); the other
    channels keep the norm output. ROPE_FIRST: the rotation first, then the norm of the rotated row, in FP32 with one final rounding.
    The weight is one row (D,), or with PER_HEAD a row per head (heads, D) at stride WH; RH: the head stride of a table with one
    rotation per head, 0 for one shared by the heads; PAIRS 0: the norm alone. 128-channel heads normalized first sum the
    squares as Comfy Kitchen does (lane sums of c, c + 32, c + 64, c + 96, then the lanes): its results bit for bit."""
    rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    ok = rows < ROWS
    head = rows % HEADS
    token = rows // HEADS
    seq = token % SEQ
    batch = token // SEQ
    c = tl.arange(0, DP)
    valid = ok[:, None] & (c < D)[None, :]
    base = batch * XB + seq * XS + head * XH
    ptr = X + base[:, None] + c[None, :]
    x = tl.load(ptr, valid, 0).to(tl.float32)
    rotated = c < 2 * PAIRS
    if INTERLEAVED:
        leading = c % 2 == 0
        partner = tl.where(rotated, c ^ 1, c)
        pair = c // 2
    else:
        leading = c < PAIRS
        partner = tl.where(leading, c + PAIRS, tl.where(rotated, c - PAIRS, c))
        pair = tl.where(leading, c, c - PAIRS)
    other = tl.load(X + base[:, None] + partner[None, :], valid, 0).to(tl.float32)
    if PER_HEAD:
        weight = tl.load(W + head[:, None] * WH + c[None, :], valid, 0).to(tl.float32)
        partner_weight = tl.load(W + head[:, None] * WH + partner[None, :], valid, 0).to(tl.float32)
    else:
        weight = tl.load(W + c, c < D, 0).to(tl.float32)[None, :]
        partner_weight = tl.load(W + partner, c < D, 0).to(tl.float32)[None, :]
    angle = batch[:, None] * RB + seq[:, None] * RS + head[:, None] * RH + pair[None, :] * RP
    mask = ok[:, None] & rotated[None, :]
    cos = tl.load(ROPE + angle, mask, 1.0).to(tl.float32)
    sin = tl.load(ROPE + angle + RC, mask, 0.0).to(tl.float32)
    first = leading[None, :]
    if ROPE_FIRST:
        r = tl.fma(tl.where(first, cos, sin), tl.where(first, x, other), tl.where(first, -sin, cos) * tl.where(first, other, x))
        r = tl.where(rotated[None, :], r, x)
        inv = libdevice.rsqrt(tl.sum(r * r, 1) / D + EPS)[:, None]
        tl.store(ptr, r * inv * weight, valid)
    else:
        if D == 128:
            lanes = tl.reshape(tl.permute(tl.reshape(x, (BLOCK, 4, 32)), 0, 2, 1), (BLOCK, 32, 2, 2))  # variadic dims: Triton 3.2 rejects the tuple form
            even, odd = tl.split(lanes)
            x0, x2 = tl.split(even)
            x1, x3 = tl.split(odd)
            total = tl.sum(tl.fma(x3, x3, tl.fma(x2, x2, tl.fma(x1, x1, x0 * x0))), 1)
        else:
            total = tl.sum(x * x, 1)
        inv = libdevice.rsqrt(total / D + EPS)[:, None]
        y = (x * inv * weight).to(X.dtype.element_ty).to(tl.float32)
        other = (other * inv * partner_weight).to(X.dtype.element_ty).to(tl.float32)
        out = tl.fma(tl.where(first, cos, sin), tl.where(first, y, other), tl.where(first, -sin, cos) * tl.where(first, other, y))
        tl.store(ptr, tl.where(rotated[None, :], out, y), valid)


def rms_rope_(x, weight, rope, eps, *, interleaved=False, rope_first=False):
    """In place on Q or K (batch, tokens, heads, head_dim), any strides with contiguous channels (views of a fused projection); weight:
    the RMSNorm weight (head_dim,), or one per head (heads, head_dim); rope: (1 | batch, tokens, 1 | heads, pairs, 2) cos / sin of the
    rotation of the first 2 * pairs channels (one table for all heads, or one per head), split halves or interleaved pairs (2p, 2p + 1: a
    complex table viewed as real), None for the norm alone; rope_first: the rotation before the norm (both in FP32, rounded once)."""
    batch, seq, heads, dim = x.shape
    rope_strides = (0, 0, 0, 0, 0) if rope is None else (0 if rope.shape[0] == 1 else rope.stride(0), rope.stride(1), rope.stride(2) if rope.shape[2] > 1 else 0,
                                                         rope.stride(3), rope.stride(4))
    pairs = 0 if rope is None else rope.shape[-2]
    per_head = weight.dim() == 2

    def launch(xx, config):
        block, warps = config
        rows = xx.shape[0] * xx.shape[1] * heads
        _rms_rope[(triton.cdiv(rows, block),)](xx, weight, xx if rope is None else rope, rows, heads, xx.shape[1], *xx.stride()[:3], *rope_strides[:3],
                                               weight.stride(0) if per_head else 0, eps, *rope_strides[3:], dim, pairs, block,
                                               interleaved, rope_first, per_head, triton.next_power_of_2(dim), num_warps=warps, enable_fp_fusion=False)

    def prepare_scratch():  # a copy of the first rows, in the head layout of x
        tokens = max(1, min(seq, _TUNE_ROWS // heads))
        scratch = x.new_empty((1, tokens, heads, x.stride(2)))[..., :dim]
        scratch.copy_(x[:1, :tokens])
        return lambda config: launch(scratch, config)

    variant = ("|F" if rope_first else "") + ("|H" if per_head else "")
    # no head count in the identity: every Attention Head Split level (group size) reuses one launch configuration, compiling nothing
    config = _TUNER.config("rms_rope", f"{x.dtype}|d{dim}|p{pairs}|s{x.stride(2)}|I{int(interleaved)}{variant}", x.device, _ROPE_CONFIGS,
                           prepare_scratch, batch * seq * heads >= _TUNE_ROWS)
    launch(x, config)
    torch.autograd.graph.increment_version(x)
    return x


from shared.kernels.triton_compilation_log import install_triton_compilation_logger
install_triton_compilation_logger()


@triton.jit(do_not_specialize=['N', 'MOD_STRIDE'], do_not_specialize_on_alignment=['N', 'MOD_STRIDE'])
def _scale_shift(X, SCALE, SHIFT, Y, N, MOD_STRIDE, D: tl.constexpr, PER_TOKEN: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    # A grid row owns one conditioning row and all tokens that borrow it.
    # No runtime integer divisions are needed to recover batch/frame indices.
    group = tl.program_id(1).to(tl.int64)
    offset = group * N + i
    if PER_TOKEN:
        s = i
    elif D % BLOCK == 0:
        s = group * MOD_STRIDE + (tl.program_id(0) % (D // BLOCK)) * BLOCK + tl.arange(0, BLOCK)
    else:
        s = group * MOD_STRIDE + i % D
    x = tl.load(X + offset, i < N, 0).to(tl.float32)
    scale = tl.load(SCALE + s, i < N, 0).to(tl.float32)
    shift = tl.load(SHIFT + s, i < N, 0).to(tl.float32)
    factor = (1.0 + scale).to(X.dtype.element_ty).to(tl.float32)
    value = (x * factor).to(X.dtype.element_ty).to(tl.float32)
    tl.store(Y + offset, value + shift, i < N)


def scale_shift(x, scale, shift, in_place):
    out = x if in_place else torch.empty_like(x)
    per_token = scale.shape[1] == x.shape[1] and scale.is_contiguous()  # strided per-token rows go through the row groups
    groups = 1 if per_token else scale.shape[0] * scale.shape[1]
    size = x.numel() // groups
    _scale_shift[(triton.cdiv(size, 1024), groups)](
        x, scale, shift, out, size, modulation_stride(scale), x.shape[2], per_token, 1024,
        enable_fp_fusion=False)
    if in_place:
        torch.autograd.graph.increment_version(out)
    return out


@triton.jit
def _axis_row(token, AXIS: tl.constexpr, G0, G1, G2):
    # The token's position along grid axis AXIS (0: first, 1: second, else: third) whose frequency table the kernel reads.
    if AXIS == 0:
        row = token // (G1 * G2) % G0
    elif AXIS == 1:
        row = token // G2 % G1
    else:
        row = token % G2
    return row


def _axis_ids(rope_axes):
    """The grid axis of each frequency table as three constexpr ints (a constexpr tuple is not subscriptable on Triton 3.2)."""
    return tuple(rope_axes) + (0,) * (3 - len(rope_axes))


def modulation_stride(scale):
    """Elements between consecutive modulation rows: LTX hands unbind views of a (B, T, K, D) projection, stride K * D."""
    return scale.stride(1) if scale.shape[1] > 1 else scale.stride(0)


@triton.jit(do_not_specialize=['N', 'TOKENS', 'WIDTH', 'G0', 'G1', 'G2', 'BATCH_STRIDE', 'HEAD_START'], do_not_specialize_on_alignment=['N', 'TOKENS', 'WIDTH', 'G0', 'G1', 'G2', 'BATCH_STRIDE', 'HEAD_START'])
def _split_rope(X, C0, S0, C1, S1, C2, S2, N, TOKENS,
                WIDTH, HALF: tl.constexpr, PAD: tl.constexpr,
                AXES: tl.constexpr, G0, G1, G2, A0: tl.constexpr, A1: tl.constexpr, A2: tl.constexpr,
                FREQ_WIDTH: tl.constexpr, BATCH_STRIDE, HEAD_START,
                FP32: tl.constexpr, BLOCK: tl.constexpr):
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
            cv = tl.load(C0 + row * FREQ_WIDTH + col, valid, 0).to(tl.float32)
            sv = tl.load(S0 + row * FREQ_WIDTH + col, valid, 0).to(tl.float32)
        elif a == 1:
            row = _axis_row(token, A1, G0, G1, G2)
            cv = tl.load(C1 + row * FREQ_WIDTH + col, valid, 0).to(tl.float32)
            sv = tl.load(S1 + row * FREQ_WIDTH + col, valid, 0).to(tl.float32)
        else:
            row = _axis_row(token, A2, G0, G1, G2)
            cv = tl.load(C2 + row * FREQ_WIDTH + col, valid, 0).to(tl.float32)
            sv = tl.load(S2 + row * FREQ_WIDTH + col, valid, 0).to(tl.float32)
        c = tl.where(valid, cv, c)
        s = tl.where(valid, sv, s)
    x0 = tl.load(X + offset, i < N, 0).to(tl.float32)
    x1 = tl.load(X + offset + HALF, i < N, 0).to(tl.float32)
    # CUDA rounding intrinsics preserve the existing path's denormal behavior
    # and prevent packed arithmetic substitutions. HIP has no matching libdevice
    # intrinsics; dispatch keeps this path on the validated NVIDIA SM120 target.
    a, b = libdevice.mul_rn(x0, c), libdevice.mul_rn(x1, c)
    if not FP32:
        a = a.to(X.dtype.element_ty).to(tl.float32)
        b = b.to(X.dtype.element_ty).to(tl.float32)
    # Preserve the native grouped layout's padding in every head group.
    valid = (i < N) & (((head + HEAD_START) % AXES != 0) | (local >= PAD))
    tl.store(X + offset, libdevice.sub_rn(a, libdevice.mul_rn(x1, s)), valid)
    tl.store(X + offset + HALF, libdevice.add_rn(b, libdevice.mul_rn(x0, s)), valid)


def split_rope(x, cache, layout, head_start=0, dim_head=None):
    cos, sin = layout.axis_cos, layout.axis_sin
    pointers = tuple(value for a in range(3) for value in (cos[min(a, len(cos)-1)], sin[min(a, len(sin)-1)]))
    size = x.numel() // 2
    return _split_rope[(triton.cdiv(size, 256),)](
        x, *pointers, size, x.shape[1], x.shape[2], (dim_head or x.shape[2] // cache.num_attention_heads) // 2,
        cache.pad_size, len(cos), *(layout.grid_sizes + (1,) * (3 - len(layout.grid_sizes))), *_axis_ids(cache.rope_axes), cos[0].shape[1],
        x.stride(0), head_start, cache.use_fp32_freqs, 256, enable_fp_fusion=False)
