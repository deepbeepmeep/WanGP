# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""H3 decoder inference fusions. Preserve the eager path's dtype boundaries."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from shared.kernels.triton_tuner import LaunchTuner

_TUNER = LaunchTuner("wan2gp_h3_vae_kernels", 5, "H3 VAE")


@triton.jit(do_not_specialize=['ROWS', 'TOKENS', 'HEADS', 'SB', 'ST', 'SH', 'CB', 'CT'], do_not_specialize_on_alignment=['ROWS', 'TOKENS', 'HEADS', 'SB', 'ST', 'SH', 'CB', 'CT'])
def _cast_rope(X, OUT, C, S, ROWS, TOKENS, HEADS,
             SB, ST, SH, CB, CT,
             D: tl.constexpr, R: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    cols = tl.arange(0, D)
    token = rows // HEADS % TOKENS
    batch = rows // (TOKENS * HEADS)
    offsets = batch * SB + token * ST + (rows % HEADS) * SH
    ptr = X
    out = OUT
    x = tl.load(ptr + offsets[:, None] + cols[None, :], rows[:, None] < ROWS, 0).to(tl.float32)
    # Keep native FP32 RMSNorm outside this kernel. Preserve its BF16 cast and
    # the rounding after each eager BF16 multiply before the rotation add/sub.
    x = x.to(out.dtype.element_ty).to(tl.float32)
    if R:
        partner = tl.where(cols < R // 2, cols + R // 2, cols - R // 2)
        partner = tl.where(cols < R, partner, cols)
        other = tl.gather(x, tl.broadcast_to(partner[None, :], (BLOCK, D)), 1)
        co = batch[:, None] * CB + token[:, None] * CT + cols[None, :]
        cos = tl.load(C + co, (rows[:, None] < ROWS) & (cols[None, :] < R), 0).to(out.dtype.element_ty).to(tl.float32)
        sin = tl.load(S + co, (rows[:, None] < ROWS) & (cols[None, :] < R), 0).to(out.dtype.element_ty).to(tl.float32)
        a = (x * cos).to(out.dtype.element_ty).to(tl.float32)
        b = (other * sin).to(out.dtype.element_ty).to(tl.float32)
        rotated = tl.where(cols[None, :] < R // 2, a - b, a + b)
        x = tl.where(cols[None, :] < R, rotated, x)
    tl.store(out + rows[:, None] * D + cols[None, :], x, rows[:, None] < ROWS)


# The elementwise kernels below take the width D at runtime on a grid of (R rows, BLOCK columns) tiles: one compile per tile shape
# for every channel width, no division per element.
_ROW_INTS = ["ROWS"]  # D keeps its divisible-by-16 specialization (vector loads)


@triton.jit
def _row_tile(ROWS, D, R: tl.constexpr, BLOCK: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * R + tl.arange(0, R)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    return rows[:, None], cols[None, :], (rows < ROWS)[:, None] & (cols < D)[None, :]


@triton.jit(do_not_specialize=_ROW_INTS, do_not_specialize_on_alignment=["ROWS"])
def _swiglu(X, ROWS, D, R: tl.constexpr, BLOCK: tl.constexpr):
    row, col, valid = _row_tile(ROWS, D, R, BLOCK)
    offset = row * (2 * D) + col
    gate = tl.load(X + offset, valid, 0).to(tl.float32)
    value = tl.load(X + offset + D, valid, 0).to(tl.float32)
    activated = (gate / (1.0 + tl.exp(-gate))).to(X.dtype.element_ty).to(tl.float32)
    tl.store(X + offset, activated * value, valid)


@triton.jit(do_not_specialize=_ROW_INTS, do_not_specialize_on_alignment=["ROWS"])
def _residual(X, SCALE, RESIDUAL, ROWS, D, R: tl.constexpr, BLOCK: tl.constexpr):
    """branch * scale + residual with the two roundings of branch.mul_(scale).add_(residual): bit-identical."""
    row, col, valid = _row_tile(ROWS, D, R, BLOCK)
    i = row * D + col
    x = tl.load(X + i, valid, 0).to(tl.float32)
    scale = tl.load(SCALE + col, valid, 0).to(tl.float32)
    residual = tl.load(RESIDUAL + i, valid, 0).to(tl.float32)
    scaled = (x * scale).to(X.dtype.element_ty).to(tl.float32)
    tl.store(X + i, scaled + residual, valid)


def row_tiles(rows, width):
    """(grid, R, BLOCK) of the row-tile kernels: BLOCK a power of two up to 1024 columns, R rows so that a tile holds 1024 elements
    (narrow rows: programs of several rows)."""
    block = min(1024, triton.next_power_of_2(width))
    tile_rows = 1024 // block
    return (triton.cdiv(rows, tile_rows), triton.cdiv(width, block)), tile_rows, block


CAST_ROPE_SUPPORTED = hasattr(tl, "gather")  # _cast_rope gathers the rotation partners: Triton 3.3+ (3.2 of the PyTorch 2.7 stack has no tl.gather)


def cast_rope(q, rotary):
    b, t, heads, dim = q.shape
    cos, sin = (q, q) if rotary is None else rotary
    r = 0 if rotary is None else cos.shape[-1]
    oq = torch.empty(q.shape, device=q.device, dtype=torch.bfloat16)
    _cast_rope[(triton.cdiv(b * t * heads, 16),)](q, oq, cos, sin, b * t * heads, t, heads, *q.stride()[:3], cos.stride(0), cos.stride(1), dim, r, 16, enable_fp_fusion=False)
    return oq


def swiglu_(expanded):
    dim = expanded.shape[-1] // 2
    rows = expanded.numel() // (2 * dim)
    grid, tile_rows, block = row_tiles(rows, dim)
    _swiglu[grid](expanded, rows, dim, tile_rows, block, enable_fp_fusion=False)
    torch.autograd.graph.increment_version(expanded)
    return expanded[..., :dim]


def residual_(branch, scale, residual):
    width = branch.shape[-1]
    rows = branch.numel() // width
    grid, tile_rows, block = row_tiles(rows, width)
    _residual[grid](branch, scale, residual, rows, width, tile_rows, block, enable_fp_fusion=False)
    torch.autograd.graph.increment_version(branch)
    return branch


@triton.jit(do_not_specialize=_ROW_INTS, do_not_specialize_on_alignment=["ROWS"])
def _swiglu_contiguous(X, OUT, ROWS, D, R: tl.constexpr, BLOCK: tl.constexpr):
    row, col, valid = _row_tile(ROWS, D, R, BLOCK)
    offset = row * (2 * D) + col
    gate = tl.load(X + offset, valid, 0).to(tl.float32)
    value = tl.load(X + offset + D, valid, 0).to(tl.float32)
    activated = (gate / (1.0 + tl.exp(-gate))).to(X.dtype.element_ty).to(tl.float32)
    tl.store(OUT + row * D + col, activated * value, valid)


@triton.jit(do_not_specialize=_ROW_INTS, do_not_specialize_on_alignment=["ROWS"])
def _swiglu_rn(X, OUT, ROWS, D, R: tl.constexpr, BLOCK: tl.constexpr):
    """silu(gate) rounded to the activation dtype, then times the value: F.silu(gate).mul_(value) bit for bit (IEEE division and
    expf as PyTorch's kernel)."""
    row, col, valid = _row_tile(ROWS, D, R, BLOCK)
    offset = row * (2 * D) + col
    gate = tl.load(X + offset, valid, 0).to(tl.float32)
    value = tl.load(X + offset + D, valid, 0).to(tl.float32)
    activated = tl.div_rn(gate, 1.0 + libdevice.exp(-gate)).to(X.dtype.element_ty).to(tl.float32)
    tl.store(OUT + row * D + col, activated * value, valid)


def swiglu_rn(expanded):
    """Rows of [gate | value] to silu(gate) * value in a new contiguous tensor (the twice-wide projection can be released)."""
    dim = expanded.shape[-1] // 2
    out = torch.empty((*expanded.shape[:-1], dim), device=expanded.device, dtype=expanded.dtype)
    rows = out.numel() // dim
    grid, tile_rows, block = row_tiles(rows, dim)
    _swiglu_rn[grid](expanded, out, rows, dim, tile_rows, block, enable_fp_fusion=False)
    return out


@triton.jit
def _norm_rope_rows(X, base, ok, cos, sin, partner, keep, EPS, D: tl.constexpr, HALF: tl.constexpr, BLOCK: tl.constexpr):
    """Rows of one head, loaded whole: RMSNorm without weight in FP32 rounded to the activation dtype (the norm output), then the
    split-half rotation of the rotary channels in FP32 with one final rounding; the other channels keep the norm output. The partner
    channels are a second load of the same rows (cache hits)."""
    c = tl.arange(0, D)
    ptr = X + base[:, None] + c[None, :]
    x = tl.load(ptr, ok[:, None], 0).to(tl.float32)
    other = tl.load(X + base[:, None] + partner[None, :], ok[:, None], 0).to(tl.float32)
    if D == 64:
        # Match Kitchen's lane sums (c, c + 32), then its warp reduction. Reuse
        # the loaded row: a differently ordered sum can change FP16 rounding.
        first, second = tl.split(tl.trans(tl.reshape(x, (BLOCK, 2, 32)), 0, 2, 1))  # variadic dims: Triton 3.2 rejects the tuple form
        squares = tl.fma(second, second, first * first)
        total = tl.sum(squares, 1)
    else:
        total = tl.sum(x * x, 1)
    inv = libdevice.rsqrt(total / D + EPS)[:, None]
    y = (x * inv).to(X.dtype.element_ty).to(tl.float32)
    other = (other * inv).to(X.dtype.element_ty).to(tl.float32)
    # Preserve which matrix product is rounded before the CUDA rotation's FMA.
    # Algebraically equivalent contractions can amplify into visible outliers.
    first_half = c[None, :] < HALF
    rotated = tl.fma(tl.where(first_half, cos, sin), tl.where(first_half, y, other),
                     tl.where(first_half, -sin, cos) * tl.where(first_half, other, y))
    tl.store(ptr, tl.where(keep[None, :], y, rotated), ok[:, None])


@triton.jit(do_not_specialize=['ROWS', 'HEADS', 'SEQ', 'QB', 'QS', 'QH', 'KB', 'KS', 'KH', 'CB', 'CS'], do_not_specialize_on_alignment=['ROWS', 'HEADS', 'SEQ', 'QB', 'QS', 'QH', 'KB', 'KS', 'KH', 'CB', 'CS'])
def _qk_rms_rope(Q, K, COS, SIN, ROWS, HEADS, SEQ, QB, QS, QH, KB, KS, KH, CB, CS, EPS, D: tl.constexpr, HALF: tl.constexpr,
                 BLOCK: tl.constexpr):
    rows = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)  # (batch, token, head) rows: the heads of a token share its angles
    ok = rows < ROWS
    head = rows % HEADS
    token = rows // HEADS
    seq = token % SEQ
    batch = token // SEQ
    c = tl.arange(0, D)
    keep = c >= 2 * HALF
    partner = tl.where(c < HALF, c + HALF, tl.where(keep, c, c - HALF))
    rotated = ok[:, None] & (c[None, :] < 2 * HALF)
    angle = batch[:, None] * CB + seq[:, None] * CS + c[None, :]
    cos = tl.load(COS + angle, rotated, 0)
    sin = tl.load(SIN + angle, rotated, 0)
    _norm_rope_rows(Q, batch * QB + seq * QS + head * QH, ok, cos, sin, partner, keep, EPS, D, HALF, BLOCK)
    _norm_rope_rows(K, batch * KB + seq * KS + head * KH, ok, cos, sin, partner, keep, EPS, D, HALF, BLOCK)


_ROPE_CONFIGS = ((8, 1), (16, 2), (16, 4), (32, 4), (64, 8))
_TUNE_ROWS = 1 << 17  # cap tuning scratch for strided projection views


def can_rms_rope(dim, rot):
    """Head widths the kernel takes: a power of two, an even number of rotary channels."""
    return dim & (dim - 1) == 0 and rot % 2 == 0 and rot <= dim


def rms_rope_(q, k, rotary, eps):
    """In place on q and k (batch, tokens, heads, head_dim), any strides with contiguous channels (views of a fused projection): per head
    RMSNorm without weight, then the split-half RoPE of the first rotary channels with the angles of rotary (cos, sin), FP32 (batch,
    tokens, 1, rotary channels) whose two halves are equal. H3's 64-channel heads preserve Comfy Kitchen's reduction and rotation
    rounding, including its FP32 fused multiply-add order. Not bit-identical to the PyTorch 16-bit rotation but more precise (one rounding
    after the rotation instead of rounded cos / sin and products)."""
    cos, sin = rotary[0], rotary[1]
    batch, seq, heads, dim = q.shape
    rows = batch * seq * heads

    def launch(qq, kk, config):
        block, warps = config
        n = qq.shape[0] * qq.shape[1] * heads
        _qk_rms_rope[(triton.cdiv(n, block),)](qq, kk, cos, sin, n, heads, qq.shape[1], *qq.stride()[:3], *kk.stride()[:3], cos.stride(0),
                                                cos.stride(1), eps, dim, cos.shape[-1] // 2, block, num_warps=warps, enable_fp_fusion=False)

    def prepare_scratch():  # copies of the first rows, in the head layout of q and k (views of a fused projection)
        tokens = max(1, min(seq, _TUNE_ROWS // heads))
        items = max(1, min(batch, _TUNE_ROWS // (tokens * heads)))
        scratch_q, scratch_k = (x.new_empty((items, tokens, heads, x.stride(2)))[..., :dim] for x in (q, k))
        scratch_q.copy_(q[:items, :tokens])
        scratch_k.copy_(k[:items, :tokens])
        return lambda config: launch(scratch_q, scratch_k, config)

    config = _TUNER.config("qk_rms_rope", f"{q.dtype}|{heads}x{dim}|r{cos.shape[-1]}|s{q.stride(2)}", q.device, _ROPE_CONFIGS,
                           prepare_scratch, rows >= _TUNE_ROWS)
    launch(q, k, config)
    torch.autograd.graph.increment_version(q)
    torch.autograd.graph.increment_version(k)
    return q, k


def swiglu(expanded):
    """Release the twice-wide projection before the following linear operation."""
    dim = expanded.shape[-1] // 2
    out = torch.empty((*expanded.shape[:-1], dim), device=expanded.device, dtype=expanded.dtype)
    rows = out.numel() // dim
    grid, tile_rows, block = row_tiles(rows, dim)
    _swiglu_contiguous[grid](expanded, out, rows, dim, tile_rows, block, enable_fp_fusion=False)
    return out


@triton.jit(do_not_specialize=['C', 'T', 'H', 'W'], do_not_specialize_on_alignment=['C', 'T', 'H', 'W'])
def _silu_pad_frames(X, OUT, C, T,
                     H, W, P: tl.constexpr,
                     FRONT: tl.constexpr, BLOCK: tl.constexpr):
    # one program row per output plane (batch, channel, padded frame): the plane's indices once per program, the channel count at
    # runtime (one compile for every VAE width), only the position in the plane split per element
    plane = tl.program_id(0).to(tl.int64)
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    t = plane % (T + FRONT) - FRONT
    c = plane // (T + FRONT) % C
    b = plane // ((T + FRONT) * C)
    col = i % (W + 2 * P) - P
    row = i // (W + 2 * P) - P
    inside = row < H + P
    col = tl.where(col < 0, -col, tl.where(col >= W, 2 * W - 2 - col, col))
    row = tl.where(row < 0, -row, tl.where(row >= H, 2 * H - 2 - row, row))
    offset = (((b * T + t) * C + c) * H + row) * W + col
    x = tl.load(X + offset, inside & (t >= 0), 0).to(tl.float32)
    if X.dtype.element_ty == tl.float16:
        value = tl.div_rn(x, 1.0 + libdevice.exp(-x))
    else:
        value = x / (1.0 + tl.exp(-x))
    tl.store(OUT + plane * ((H + 2 * P) * (W + 2 * P)) + i, value, inside)


def silu_pad_frames(normalized, batch, frames, padding, front):
    _, channels, _, height, width = normalized.shape
    out = torch.empty((batch, channels, frames + front, height + 2 * padding, width + 2 * padding),
                      device=normalized.device, dtype=normalized.dtype)
    plane = out.shape[3] * out.shape[4]
    _silu_pad_frames[(batch * channels * (frames + front), triton.cdiv(plane, 256))](normalized, out, channels, frames, height, width,
                                                                                    padding, front, 256, enable_fp_fusion=False)
    return out


from shared.kernels.triton_compilation_log import install_triton_compilation_logger
install_triton_compilation_logger()
