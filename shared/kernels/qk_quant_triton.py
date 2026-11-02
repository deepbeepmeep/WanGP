# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Staged SageAttention 2 quantization of q and k fused with their weighted RMSNorm and split-half RoPE: one pass over q from the
projection output to INT8 rows and scales, two passes over k (the partial sums of its normalized, rotated rows for the mean SageAttention
subtracts, then the quantization with that mean, the norm and rotation recomputed from the projection output). The norm reproduces the
in-place kernel of shared.kernels.denoiser_triton (FP32 statistics, the norm output rounded to the activation dtype), the rotation runs in
FP32, and the quantization reads the FP32 rotated values where the unfused path reads them rounded to the activation dtype. The
quantization arithmetic (block scales max|x| / 127, half-away-from-zero rounding, the approximate FP32 division) is that of
SageAttention 2's quantizers (Apache 2.0, https://github.com/thu-ml/SageAttention, sageattention/triton/quant_per_thread.py and
quant_per_block.py), so the quantization-only mode reproduces them bit for bit; the row groups of a scale follow the layouts
LAYOUT_PER_BLOCK (one scale per block of rows), LAYOUT_THREAD_Q (eight scales per 32 rows, rows r, r + 8, r + 16, r + 24),
LAYOUT_THREAD_K (four scales per 64 rows, row pairs) and LAYOUT_WARP_Q (four scales per 128 rows, 32 consecutive rows each).
The rotation pairs channel p with p + PAIRS (split halves) or, interleaved, 2p with 2p + 1; the cos / sin of pair p are read at the
table's pair stride, the sine at its last-dimension stride from the cosine, so a (..., pairs, 2) table and the 2 x 2 matrices
[[cos, -sin], [sin, cos]] of a (..., pairs, 2, 2) table both work as they are.

A program owns one block of rows of one head and walks it by sub-blocks of 8 rows, the live tile of the in-place RoPE kernel: the
maxima of the layout's row groups first, then the rows again from the cache for the quantization (the norm and rotation recomputed,
cheaper than holding or storing the block)."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .triton_tuner import LaunchTuner

LAYOUT_PER_BLOCK, LAYOUT_THREAD_Q, LAYOUT_THREAD_K, LAYOUT_WARP_Q = 0, 1, 2, 3
SCALES_PER_BLOCK = {LAYOUT_PER_BLOCK: 1, LAYOUT_THREAD_Q: 8, LAYOUT_THREAD_K: 4, LAYOUT_WARP_Q: 4}
PADDED_ROWS = {LAYOUT_THREAD_Q: 128}  # SageAttention sizes the per-thread q scales by 128-row blocks: the tail blocks of 32 rows beyond the tokens get the epsilon scale
_TUNER = LaunchTuner("wan2gp_qk_quant", 5, "Attention")
_LAUNCH_CONFIGS = ((1, 1), (1, 2), (2, 1))  # (num_warps, num_stages): one warp per program of 8-row steps; wider or deeper launches spill (RTX 4090: 2-10x slower)
_TUNE_ROWS = 1 << 16
_RUNTIME = ['SEQ', 'XB', 'XS', 'XH', 'RB', 'RS', 'RH', 'OB', 'OS', 'OH', 'SB', 'SH', 'MB', 'MH', 'PB', 'PH', 'PK', 'WH']  # WROW: a pointer


def _blocks(seq, layout, block):
    """Blocks of rows the launch covers (their scales are all written): the tokens padded as SageAttention's quantizer pads them."""
    pad = PADDED_ROWS.get(layout, block)
    return triton.cdiv(seq, pad) * (pad // block)


@triton.jit
def _rows(X, W, WROW, ROPE, MEAN, base, seq, ok, c, batch, head, RB, RS, RH, MB, MH, WH, EPS, RP: tl.constexpr, RC: tl.constexpr, D: tl.constexpr,
          PAIRS: tl.constexpr, NORM: tl.constexpr, ROTATE: tl.constexpr, HAS_MEAN: tl.constexpr, SUB: tl.constexpr, LANES: tl.constexpr,
          INTERLEAVED: tl.constexpr, SEGMENTED: tl.constexpr, ROPE_FIRST: tl.constexpr):
    """The rows at base (one head; c spans D rounded up to a power of two, the channels beyond D are zeros): weighted RMSNorm in FP32
    rounded to the activation dtype (NORM; the weight row of each token from WROW with SEGMENTED, else that of the head at stride WH,
    0 for one row), the rotation of the first 2 * PAIRS channels in FP32 (ROTATE, cos / sin pairs of ROPE), minus the head's MEAN row
    (HAS_MEAN); with ROPE_FIRST the rotation comes first and the norm of the rotated rows stays in FP32; rows beyond SEQ are zeros. RH:
    the head stride of a table with one rotation per head, 0 for one shared by the heads."""
    inside = ok[:, None] & (c < D)[None, :]
    x = tl.load(X + base[:, None] + c[None, :], inside, 0).to(tl.float32)
    if SEGMENTED:
        wrow = tl.load(WROW + seq, ok, 0).to(tl.int64)[:, None] * D
    else:
        wrow = tl.zeros([SUB, 1], tl.int64) + head * WH
    if NORM and not ROPE_FIRST:
        if LANES and D == 128:  # lane sums of c, c + 32, c + 64, c + 96, then the lanes (shared.kernels.denoiser_triton._rms_rope)
            lanes = tl.reshape(tl.permute(tl.reshape(x, (SUB, 4, 32)), 0, 2, 1), (SUB, 32, 2, 2))
            even, odd = tl.split(lanes)
            x0, x2 = tl.split(even)
            x1, x3 = tl.split(odd)
            total = tl.sum(tl.fma(x3, x3, tl.fma(x2, x2, tl.fma(x1, x1, x0 * x0))), 1)
        else:
            total = tl.sum(x * x, 1)
        inv = libdevice.rsqrt(total / D + EPS)[:, None]
        y = (x * inv * tl.load(W + wrow + c[None, :], (c < D)[None, :], 0).to(tl.float32)).to(X.dtype.element_ty).to(tl.float32)
    else:
        y = x
    if ROTATE:
        rotated = c < 2 * PAIRS
        if INTERLEAVED:  # channel 2p with 2p + 1
            leading = c % 2 == 0
            partner = tl.where(rotated, c ^ 1, c)
            pair = c // 2
        else:  # channel p with p + PAIRS
            leading = c < PAIRS
            partner = tl.where(leading, c + PAIRS, tl.where(rotated, c - PAIRS, c))
            pair = tl.where(leading, c, c - PAIRS)
        other = tl.load(X + base[:, None] + partner[None, :], inside, 0).to(tl.float32)
        if NORM and not ROPE_FIRST:
            other = (other * inv * tl.load(W + wrow + partner[None, :], (c < D)[None, :], 0).to(tl.float32)).to(X.dtype.element_ty).to(tl.float32)
        angle = batch * RB + seq[:, None] * RS + head * RH + pair[None, :] * RP
        mask = ok[:, None] & rotated[None, :]
        cos = tl.load(ROPE + angle, mask, 1.0).to(tl.float32)
        sin = tl.load(ROPE + angle + RC, mask, 0.0).to(tl.float32)
        first = leading[None, :]
        out = tl.fma(tl.where(first, cos, sin), tl.where(first, y, other), tl.where(first, -sin, cos) * tl.where(first, other, y))
        y = tl.where(rotated[None, :], out, y)
    if NORM and ROPE_FIRST:
        inv = libdevice.rsqrt(tl.sum(y * y, 1) / D + EPS)[:, None]
        y = y * inv * tl.load(W + wrow + c[None, :], (c < D)[None, :], 0).to(tl.float32)
    if HAS_MEAN:  # rows beyond SEQ stay zeros: they must not reach the scale of their row group
        y = tl.where(ok[:, None], y - tl.load(MEAN + batch * MB + head * MH + c, c < D, 0).to(tl.float32)[None, :], 0.0)
    return y


@triton.jit(do_not_specialize=_RUNTIME, do_not_specialize_on_alignment=_RUNTIME)
def _norm_rope_quant(X, W, WROW, ROPE, OUT, SCALE, MEAN, SUMS, SEQ, XB, XS, XH, RB, RS, RH, OB, OS, OH, SB, SH, MB, MH, PB, PH, PK, WH, EPS, SM_SCALE,
                     SCALE_EPS, RP: tl.constexpr, RC: tl.constexpr, D: tl.constexpr, PAIRS: tl.constexpr, NORM: tl.constexpr, ROTATE: tl.constexpr,
                     SUM_ONLY: tl.constexpr, HAS_MEAN: tl.constexpr, LAYOUT: tl.constexpr, BLOCK: tl.constexpr, GROUPS: tl.constexpr,
                     SUB: tl.constexpr, LANES: tl.constexpr, INTERLEAVED: tl.constexpr, SEGMENTED: tl.constexpr, ROPE_FIRST: tl.constexpr,
                     DP: tl.constexpr, OW: tl.constexpr):
    """One block of BLOCK rows (tokens) of one head of one batch item of X (batch, tokens, heads, D), by steps of SUB rows (_rows): the
    sum of the rows into SUMS (SUM_ONLY), or the rows times SM_SCALE as INT8 into OUT with the GROUPS scales of the LAYOUT into SCALE.
    DP: D rounded up to a power of two; OW: the channels written per row of OUT, D or DP (an attention that pads the heads to DP
    reads zeros beyond D)."""
    blk = tl.program_id(0).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    batch = tl.program_id(2).to(tl.int64)
    c = tl.arange(0, DP)
    r = tl.arange(0, SUB)
    g = tl.arange(0, GROUPS)
    if SUM_ONLY:
        total = tl.zeros([DP], tl.float32)
        for start in range(0, BLOCK, SUB):
            seq = blk * BLOCK + start + r
            ok = seq < SEQ
            total += tl.sum(_rows(X, W, WROW, ROPE, MEAN, batch * XB + seq * XS + head * XH, seq, ok, c, batch, head, RB, RS, RH, MB, MH, WH, EPS,
                                  RP, RC, D, PAIRS, NORM, ROTATE, HAS_MEAN, SUB, LANES, INTERLEAVED, SEGMENTED, ROPE_FIRST), 0)
        tl.store(SUMS + batch * PB + head * PH + blk * PK + c, total, c < D)
        return
    amax = tl.zeros([GROUPS], tl.float32)
    for start in range(0, BLOCK, SUB):
        seq = blk * BLOCK + start + r
        ok = seq < SEQ
        v = _rows(X, W, WROW, ROPE, MEAN, batch * XB + seq * XS + head * XH, seq, ok, c, batch, head, RB, RS, RH, MB, MH, WH, EPS,
                  RP, RC, D, PAIRS, NORM, ROTATE, HAS_MEAN, SUB, LANES, INTERLEAVED, SEGMENTED, ROPE_FIRST) * SM_SCALE
        m = tl.max(tl.abs(v), 1)
        if LAYOUT == 1:  # row j of every step belongs to group j % 8
            amax = tl.maximum(amax, tl.max(tl.reshape(m, (SUB // 8, GROUPS)), 0))
        elif LAYOUT == 2:  # row j of every step belongs to group (j % 8) // 2
            amax = tl.maximum(amax, tl.max(tl.max(tl.reshape(m, (SUB // 8, GROUPS, 2)), 2), 0))
        elif LAYOUT == 3:  # the rows of this step belong to the group of their 32-row slice
            amax = tl.where(g == start // 32, tl.maximum(amax, tl.max(m, 0)), amax)
        else:
            amax = tl.maximum(amax, tl.max(m, 0))
    scale = amax / 127. + SCALE_EPS
    tl.store(SCALE + batch * SB + head * SH + blk * GROUPS + g, scale)
    for start in range(0, BLOCK, SUB):
        seq = blk * BLOCK + start + r
        ok = seq < SEQ
        v = _rows(X, W, WROW, ROPE, MEAN, batch * XB + seq * XS + head * XH, seq, ok, c, batch, head, RB, RS, RH, MB, MH, WH, EPS,
                  RP, RC, D, PAIRS, NORM, ROTATE, HAS_MEAN, SUB, LANES, INTERLEAVED, SEGMENTED, ROPE_FIRST) * SM_SCALE
        if LAYOUT == 1:
            q = v / tl.reshape(tl.broadcast_to(scale[None, :], (SUB // 8, GROUPS)), (SUB,))[:, None]
        elif LAYOUT == 2:
            q = v / tl.reshape(tl.broadcast_to(tl.join(scale, scale)[None, :, :], (SUB // 8, GROUPS, 2)), (SUB,))[:, None]
        elif LAYOUT == 3:
            q = v / tl.sum(tl.where(g == start // 32, scale, 0.0), 0)
        else:
            q = v / tl.sum(scale, 0)
        q += 0.5 * tl.where(q >= 0, 1, -1)
        tl.store(OUT + batch * OB + seq[:, None] * OS + head * OH + c[None, :], q.to(tl.int8), ok[:, None] & (c < OW)[None, :])


SUB_ROWS, LANE_SUMS = 8, True  # rows per step (the live tile of the in-place RoPE kernel), and the sum of squares in that kernel's lane order (same speed, norm statistics identical to it and independent of the warp count)


def _launch(x, weight, rope, out, scale, mean, sums, eps, sm_scale, scale_eps, pairs, norm, rotate, sum_only, layout, block, warps, scratch=None,
            stages=2, sub=None, lanes=None, interleaved=False, weight_rows=None, rope_first=False):
    xx = x if scratch is None else scratch
    batch, seq, heads, dim = xx.shape
    blocks = _blocks(seq, layout, block) if not sum_only else triton.cdiv(seq, block)
    rope_batch = 0 if rope is None or rope.shape[0] == 1 else rope.stride(0)
    rope_args = (rope, rope_batch, rope.stride(1), rope.stride(2) if rope.shape[2] > 1 else 0, rope.stride(3), rope.stride(4)) if rope is not None else (xx, 0, 0, 0, 0, 0)
    out_args = (out, *out.stride()[:3]) if out is not None else (xx, 0, 0, 0)
    scale_args = (scale, *scale.stride()[:2]) if scale is not None else (xx, 0, 0)
    mean_args = (mean, *mean.stride()[:2]) if mean is not None else (xx, 0, 0)
    sums_args = (sums, *sums.stride()[:3]) if sums is not None else (xx, 0, 0, 0)
    _norm_rope_quant[(blocks, heads, batch)](
        xx, weight if weight is not None else xx, weight_rows if weight_rows is not None else xx, rope_args[0], out_args[0], scale_args[0], mean_args[0], sums_args[0],
        seq, *xx.stride()[:3], *rope_args[1:4], *out_args[1:], *scale_args[1:], *mean_args[1:], *sums_args[1:],
        weight.stride(0) if _per_head(weight, weight_rows) else 0, eps, sm_scale, scale_eps, rope_args[4], rope_args[5], dim, pairs, norm, rotate,
        sum_only, mean is not None, layout, block, SCALES_PER_BLOCK[layout], SUB_ROWS if sub is None else sub, LANE_SUMS if lanes is None else lanes,
        interleaved, weight_rows is not None, rope_first, triton.next_power_of_2(dim), dim if out is None else out.shape[-1],
        num_warps=warps, num_stages=stages, enable_fp_fusion=False)


def _per_head(weight, weight_rows):
    """A weight row per head (heads, D), as opposed to one row (D,) or the stacked rows of token segments (weight_rows)."""
    return weight is not None and weight.dim() == 2 and weight_rows is None


def _variant(weight, weight_rows, rope_first):
    """Tuning identity suffix of the kernel variants added after the first cache entries (empty for those)."""
    return ("F" if rope_first else "") + ("H" if _per_head(weight, weight_rows) else "")


def _pairs(rope):
    """Rotated channel pairs of a table (1 | batch, tokens, 1 | heads, pairs, 2) or (1 | batch, tokens, 1 | heads, pairs, 2, 2)."""
    return 0 if rope is None else rope.shape[3]


def _config(x, name, identity, block, norm, rotate, sum_only, layout, launch):
    batch, seq, heads, dim = x.shape

    def prepare_scratch():  # a copy of the first rows, in the head layout of x
        tokens = max(block, min(seq, _TUNE_ROWS // heads)) // block * block
        scratch = x.new_empty((1, min(seq, tokens), heads, x.stride(2)))[..., :dim]
        scratch.copy_(x[:1, :scratch.shape[1]])
        return lambda config: launch(scratch, config)

    # no head count in the identity: every Attention Head Split level (group size) reuses one launch configuration, compiling nothing
    return _TUNER.config(name, f"{identity}|{x.dtype}|d{dim}|s{x.stride(2)}|N{int(norm)}R{int(rotate)}S{int(sum_only)}L{layout}B{block}",
                         x.device, _LAUNCH_CONFIGS, prepare_scratch, batch * seq * heads >= _TUNE_ROWS)


def norm_rope_mean(x, weight, rope, eps, block=64, *, interleaved=False, weight_rows=None, rope_first=False):
    """Mean over the tokens of the normalized, rotated rows of x (batch, tokens, heads, D) per batch item and head, (batch, heads, D) FP32,
    from FP32 partial sums per block of rows; x is not modified. weight (D,) or one row per head (heads, D); None skips the norm, rope
    None the rotation (a table per head allowed: (1 | batch, tokens, heads, pairs, 2)); interleaved: pairs (2p, 2p + 1) instead of
    (p, p + pairs); weight_rows (tokens,) int32: the row of weight (rows, D)
    of each token (segments of tokens with their own norm weights); rope_first: the rotation before the norm, kept in FP32."""
    batch, seq, heads, dim = x.shape
    sums = torch.empty((batch, heads, triton.cdiv(seq, block), dim), dtype=torch.float32, device=x.device)
    pairs = _pairs(rope)

    def launch(xx, config):
        _launch(x, weight, rope, None, None, None, sums if xx is x else sums[:1, :, :triton.cdiv(xx.shape[1], block)], eps, 1.0, 0.0, pairs,
                weight is not None, rope is not None, True, 0, block, config[0], scratch=None if xx is x else xx, stages=config[1], interleaved=interleaved,
                weight_rows=weight_rows, rope_first=rope_first)

    launch(x, _config(x, "norm_rope_sums", f"p{pairs}I{int(interleaved)}G{int(weight_rows is not None)}{_variant(weight, weight_rows, rope_first)}", block,
                      weight is not None, rope is not None, True, 0, launch))
    return sums.sum(dim=2).div_(seq)


def norm_rope_quantize(x, weight, rope, eps, layout, block, *, mean=None, sm_scale=1.0, scale_eps=1e-7, interleaved=False, weight_rows=None,
                       rope_first=False, width=None):
    """INT8 rows and FP32 scales (batch, heads, blocks * scales per block) of the normalized (weight, eps), rotated (rope, interleaved,
    rope_first as norm_rope_mean) rows of x, minus mean (batch, heads, D) when given and times sm_scale, in the given scale layout and
    block of rows; x is not modified. width: the INT8 rows' width when the attention pads the heads (D rounded up to a power of two),
    zeros beyond D."""
    batch, seq, heads, dim = x.shape
    out = torch.empty((batch, seq, heads, width or dim), dtype=torch.int8, device=x.device)
    scale = torch.empty((batch, heads, _blocks(seq, layout, block) * SCALES_PER_BLOCK[layout]), dtype=torch.float32, device=x.device)
    pairs = _pairs(rope)

    def launch(xx, config):
        if xx is x:
            _launch(x, weight, rope, out, scale, mean, None, eps, sm_scale, scale_eps, pairs, weight is not None, rope is not None, False, layout, block,
                    config[0], stages=config[1], interleaved=interleaved, weight_rows=weight_rows, rope_first=rope_first)
        else:
            rows = xx.shape[1]
            _launch(x, weight, rope, out[:1, :rows], scale[:1, :, :_blocks(rows, layout, block) * SCALES_PER_BLOCK[layout]], mean, None, eps, sm_scale, scale_eps,
                    pairs, weight is not None, rope is not None, False, layout, block, config[0], scratch=xx, stages=config[1], interleaved=interleaved,
                    weight_rows=weight_rows, rope_first=rope_first)

    launch(x, _config(x, "norm_rope_quant", f"p{pairs}M{int(mean is not None)}I{int(interleaved)}G{int(weight_rows is not None)}{_variant(weight, weight_rows, rope_first)}"
                      f"{'' if width in (None, dim) else f'W{width}'}", block, weight is not None, rope is not None, False, layout, launch))
    return out, scale


from shared.kernels.triton_compilation_log import install_triton_compilation_logger
install_triton_compilation_logger()
