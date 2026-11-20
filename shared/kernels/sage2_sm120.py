# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""GeForce Blackwell kernels for SageAttention 2.2's per-warp INT8 Q/K and per-channel E4M3 V.
The FP8 layout, scales and buffered softmax convention follow the upstream SageAttention CUDA kernels
(https://github.com/thu-ml/SageAttention, Apache-2.0)."""
import torch
import triton
import triton.language as tl

from .dispatch import choose


@triton.jit
def scaled_e4m3(x, peak, BF16: tl.constexpr):
    # A finite BF16 subnormal can have an infinite FP32 reciprocal. Lift those channels by an
    # exact power of two before forming their multiplier; the output scale uses the original peak.
    if BF16:
        tiny = (peak > 0.) & (peak < 1.e-37)
        if tl.max(tiny.to(tl.int32), 0):
            lift = tl.where(tiny, 18446744073709551616., 1.)
            peak *= lift
            x *= lift[None, :]
    return (x * tl.where(peak == 0., 0., 2.25 / peak)[None, :]).to(tl.float8e4nv)


@triton.jit(do_not_specialize=["N", "H"])
def _value_amax(V, PARTIAL, N, H, D: tl.constexpr, VZ, VT, VH, ROWS: tl.constexpr):
    block, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    row = block.to(tl.int64) * ROWS + tl.arange(0, ROWS)
    channel = tl.arange(0, D)
    x = tl.load(V + batch * VZ + head * VH + row[:, None] * VT + channel[None, :], row[:, None] < N, 0)
    peak = tl.max(tl.abs(x.to(tl.float32)), 0)
    tl.store(PARTIAL + ((batch * H + head) * tl.cdiv(N, ROWS) + block) * D + channel, peak)


@triton.jit(do_not_specialize=["N", "H"])
def _pack_value(V, PARTIAL, SCALE, OUT, N, H, D: tl.constexpr, VZ, VT, VH, OZ, OD, OH,
                ROWS: tl.constexpr, CHANNELS: tl.constexpr, GROUPS: tl.constexpr, PEAK_ROWS: tl.constexpr,
                TILE_WIDTH: tl.constexpr):
    tile, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    block = tile // (D // CHANNELS)
    channel = tile % (D // CHANNELS) * CHANNELS + tl.arange(0, CHANNELS)
    row = block.to(tl.int64) * ROWS + tl.arange(0, ROWS)
    # Sage2's FP8 MMA consumes each 16-token group as 0, 1, 8, 9, 2, 3, 10, 11, ...
    src = row // 16 * 16 + row % 4 // 2 * 8 + row % 16 // 4 * 2 + row % 2
    x = tl.load(V + batch * VZ + head * VH + src[:, None] * VT + channel[None, :], src[:, None] < N, 0).to(tl.float32)
    count = tl.cdiv(N, PEAK_ROWS)
    if GROUPS:
        group = tl.arange(0, GROUPS)
        peak = tl.max(tl.load(PARTIAL + ((batch * H + head) * count + group[:, None]) * D + channel[None, :],
                              group[:, None] < count, 0.), 0)
    else:
        peak = tl.load(PARTIAL + (batch * H + head) * D + channel)
    if block == 0:
        tl.store(SCALE + (batch * H + head) * D + channel, peak / 2.25)
    fp8 = scaled_e4m3(x, peak, V.dtype.element_ty == tl.bfloat16)
    if TILE_WIDTH:
        base = ((batch * H + head) * tl.cdiv(N, 64) + row // 64) * (64 * TILE_WIDTH)
        if TILE_WIDTH == 292:
            base += 6400
        swizzle = ((row[:, None] % 64 // 16) ^ (channel[None, :] // 2 % 4)) * 16
        tl.store(OUT + base[:, None] + channel[None, :] * 64 + swizzle + row[:, None] % 16, fp8,
                 row[:, None] < tl.cdiv(N, 64) * 64)
    else:
        tl.store(OUT + batch * OZ + head * OH + channel[:, None] * OD + row[None, :], tl.trans(fp8),
                 row[None, :] < tl.cdiv(N, 64) * 64)


@triton.jit(do_not_specialize=["N", "H"])
def _channel_peak(PARTIAL, PEAK, N, H, D: tl.constexpr, GROUPS: tl.constexpr, PEAK_ROWS: tl.constexpr):
    head, batch = tl.program_id(1), tl.program_id(2).to(tl.int64)
    channel = tl.program_id(0) * 16 + tl.arange(0, 16)
    group = tl.arange(0, GROUPS)
    count = tl.cdiv(N, PEAK_ROWS)
    peak = tl.max(tl.load(PARTIAL + ((batch * H + head) * count + group[:, None]) * D + channel[None, :],
                          group[:, None] < count, 0.), 0)
    tl.store(PEAK + (batch * H + head) * D + channel, peak)


def pack_value(v, nhd, *, partial=None, peak_rows=128, tiles=None):
    """Pack 16-bit V, optionally using maxima from its producer. Routed HND calls can supply their K/V tiles."""
    x = v if nhd else v.transpose(1, 2)
    batch, tokens, heads, dim = x.shape
    padded = triton.cdiv(tokens, 64) * 64
    groups = triton.cdiv(tokens, peak_rows)
    if partial is None:
        partial = torch.empty((batch, heads, groups, dim), dtype=torch.float32, device=x.device)
        _value_amax[(groups, heads, batch)](x, partial, tokens, heads, dim, *x.stride()[:3], peak_rows, num_warps=4)
    scale = torch.empty((batch, heads, dim), dtype=torch.float32, device=x.device)
    shape = (batch, dim, heads, padded) if nhd else (batch, heads, dim, padded)
    if tiles is None:
        out = torch.empty(shape, dtype=torch.float8_e4m3fn, device=x.device)
    else:
        out = tiles.view(torch.float8_e4m3fn).as_strided(shape, (tiles.stride(0), tiles.stride(1), padded, 1))
    logical = out if nhd else out.transpose(1, 2)
    reduction = triton.next_power_of_2(groups)
    if groups >= 32:
        peak = torch.empty_like(scale)
        _channel_peak[(dim // 16, heads, batch)](partial, peak, tokens, heads, dim, reduction, peak_rows, num_warps=4)
        partial, reduction = peak, 0
    _pack_value[(padded // 32 * (dim // 64), heads, batch)](
        x, partial, scale, out, tokens, heads, dim, *x.stride()[:3], *logical.stride()[:3],
        32, 64, reduction, peak_rows, 0 if tiles is None else tiles.size(-1), num_warps=4)
    if tiles is not None:
        torch.autograd.graph.increment_version(tiles)
    return out, scale, None


# A short query gives the stock CUDA forward only one CTA per head, regardless of the number of keys. Split the keys across CTAs.
# Independent softmax maxima per split change the FP8 rounding of every subsequent 64-key tile. Measure those maxima first, so each
# split starts with the same running maximum as the serial kernel. This first pass retains just one float per query and 128 keys.
@triton.jit(do_not_specialize=["T", "N", "H", "HK"])
def _key_maxima(Q, K, QS, KS, MAX, T, N, H, HK, D: tl.constexpr, QZ, QT, QH, KZ, KT, KH, SCALE):
    chunk = tl.program_id(0)
    row = tl.program_id(1).to(tl.int64) * 16 + tl.arange(0, 16)
    zh = tl.program_id(2).to(tl.int64)
    batch, head = zh // H, zh % H
    kv_head = head // (H // HK)
    key = chunk.to(tl.int64) * 128 + tl.arange(0, 128)
    d = tl.arange(0, D)
    q = tl.load(Q + batch * QZ + head * QH + row[:, None] * QT + d[None, :], row[:, None] < T, 0)
    k = tl.load(K + batch * KZ + kv_head * KH + key[:, None] * KT + d[None, :], key[:, None] < N, 0)
    qs = tl.load(QS + zh * tl.cdiv(T, 128) * 4 + row // 32, row < T, 0.)
    ks = tl.load(KS + (batch * HK + kv_head) * tl.cdiv(N, 64) + key // 64, key < N, 0.)
    dot = tl.dot(q, tl.trans(k), out_dtype=tl.int32).to(tl.float32)
    score = dot * ((qs[:, None] * ks[None, :]) * SCALE)
    hi = tl.max(tl.where(key[None, :] < N, score, -float("inf")), 1) - 8.807
    tl.store(MAX + (zh * T + row) * tl.cdiv(N, 128) + chunk, hi, row < T)


@triton.jit(do_not_specialize=["T", "N", "H", "HK"])
def _attend_range(Q, K, V, QS, KS, MAX, SUM, TOP, OUT, T, N, H, HK, D: tl.constexpr,
                  QZ, QT, QH, KZ, KT, KH, VZ, VD, VH, SCALE, PARTS: tl.constexpr, CHUNKS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * 32 + tl.arange(0, 32)
    head = tl.program_id(1)
    zp = tl.program_id(2).to(tl.int64)
    batch, part = zp // PARTS, zp % PARTS
    kv_head = head // (H // HK)
    zh = batch * H + head
    d, j = tl.arange(0, D), tl.arange(0, 64)
    key_order = j // 16 * 16 + j % 4 // 2 * 8 + j % 16 // 4 * 2 + j % 2
    q = tl.load(Q + batch * QZ + head * QH + row[:, None] * QT + d[None, :], row[:, None] < T, 0)
    qs = tl.load(QS + zh * tl.cdiv(T, 128) * 4 + row // 32, row < T, 0.)
    count = tl.cdiv(N, 128)
    start = part * count // PARTS
    hi = tl.full((32,), -float("inf"), tl.float32)
    if part > 0:
        ch = tl.arange(0, CHUNKS)
        hi = tl.max(tl.load(MAX + (zh * T + row[:, None]) * count + ch[None, :],
                            (row[:, None] < T) & (ch[None, :] < start), -float("inf")), 1)
    denom = tl.full((32,), 0, tl.float32)
    numer = tl.full((32, D), 0, tl.float32)
    for block in range(start * 2, tl.minimum(tl.cdiv(N, 64), ((part + 1) * count // PARTS) * 2)):
        key = block * 64 + key_order
        k = tl.load(K + batch * KZ + kv_head * KH + key[:, None] * KT + d[None, :], key[:, None] < N, 0)
        ks = tl.load(KS + (batch * HK + kv_head) * tl.cdiv(N, 64) + block)
        score = tl.dot(q, tl.trans(k), out_dtype=tl.int32).to(tl.float32) * ((qs * ks) * SCALE)[:, None]
        score = tl.where(key[None, :] < N, score, -float("inf"))
        new_hi = tl.maximum(hi, tl.max(score, 1) - 8.807)
        prob = tl.exp2(score - new_hi[:, None])
        factor = tl.exp2(hi - new_hi)
        denom = denom * factor + tl.sum(prob, 1)
        # V is already padded to 64 tokens. K above was gathered in V's permuted order.
        vt = tl.load(V + batch * VZ + kv_head * VH + d[:, None] * VD + (block * 64 + j)[None, :])
        numer = numer * factor[:, None] + tl.dot(prob.to(tl.float8e4nv), tl.trans(vt), out_dtype=tl.float16)
        hi = new_hi
    ptr = (zp * H + head) * T + row
    tl.store(SUM + ptr, denom, row < T)
    tl.store(TOP + ptr, hi, row < T)
    tl.store(OUT + (ptr[:, None] * D + d[None, :]), numer, row[:, None] < T)


@triton.jit(do_not_specialize=["T", "H", "HK"])
def _finish_ranges(P, SUM, TOP, VS, OUT, LSE, T, H, HK, D: tl.constexpr, OZ, OT, OH, PARTS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64) * 4 + tl.arange(0, 4)
    head, batch = tl.program_id(1), tl.program_id(2).to(tl.int64)
    part, d = tl.arange(0, PARTS), tl.arange(0, D)
    ptr = ((batch * PARTS + part[:, None]) * H + head) * T + row[None, :]
    hi = tl.load(TOP + ptr, row[None, :] < T, 0.)
    max_hi = tl.max(hi, 0)
    factor = tl.exp2(hi - max_hi[None, :])
    denom = tl.sum(tl.load(SUM + ptr, row[None, :] < T, 0.) * factor, 0)
    values = tl.load(P + ptr[:, :, None] * D + d[None, None, :], row[None, :, None] < T, 0.)
    out = tl.sum(values * factor[:, :, None], 0) / denom[:, None]
    scale = tl.load(VS + (batch * HK + head // (H // HK)) * D + d)
    tl.store(OUT + batch * OZ + head * OH + row[:, None] * OT + d[None, :], out * scale[None, :], row[:, None] < T)
    if LSE is not None:
        tl.store(LSE + (batch * H + head) * T + row, max_hi + tl.log2(denom), row < T)


def split_attention(out, q, k, v, qs, ks, vs, sm_scale, return_lse):
    """Return the requested LSE or out, and None for shapes better served by the regular CUDA forward."""
    batch, rows, heads, dim = q.shape
    keys, kv_heads = k.shape[1:3]
    if (not 0 < rows <= 128 or not 2048 <= keys <= 32768 or dim not in (64, 128)
            or not batch or not heads or not kv_heads or heads % kv_heads or k.shape[0] != batch or k.shape[3] != dim
            or v.shape[:3] != (batch, dim, kv_heads) or v.shape[3] < triton.cdiv(keys, 64) * 64
            or any(t.stride(-1) != 1 for t in (q, k, v, out)) or not all(t.is_contiguous() for t in (qs, ks, vs))):
        return None
    sms = torch.cuda.get_device_properties(q.device).multi_processor_count
    query_tiles = batch * heads * triton.cdiv(rows, 32)
    # At 128 channels, the auxiliary score pass outweighs the parallelism once these query tiles
    # already occupy most of the SMs. Keep the CUDA forward for those head groups.
    if batch * heads > sms // 4 or (dim == 128 and query_tiles > 3 * sms // 4):
        return None
    choose("Sage2 Short-Query Attention", owned_triton=True)
    chunks = triton.cdiv(keys, 128)
    parts = min(16, chunks, max(4, triton.next_power_of_2(triton.cdiv(2 * sms, query_tiles))))
    device = q.device
    shape = (batch, parts, heads, rows)
    maxima = torch.empty((batch, heads, rows, chunks), dtype=torch.float32, device=device)
    sums = torch.empty(shape, dtype=torch.float32, device=device)
    top = torch.empty_like(sums)
    partial = torch.empty((*shape, dim), dtype=torch.float32, device=device)
    lse = torch.empty((batch, heads, rows), dtype=torch.float32, device=device) if return_lse else None
    scale = sm_scale * 1.4426950408889634
    strides = (*q.stride()[:3], *k.stride()[:3])
    _key_maxima[(chunks, triton.cdiv(rows, 16), batch * heads)](
        q, k, qs, ks, maxima, rows, keys, heads, kv_heads, dim, *strides, scale, num_warps=4)
    _attend_range[(triton.cdiv(rows, 32), heads, batch * parts)](
        q, k, v, qs, ks, maxima, sums, top, partial, rows, keys, heads, kv_heads, dim,
        *strides, *v.stride()[:3], scale, parts, triton.next_power_of_2(chunks), num_warps=4, num_stages=1)
    _finish_ranges[(triton.cdiv(rows, 4), heads, batch)](
        partial, sums, top, vs, out, lse, rows, heads, kv_heads, dim, *out.stride()[:3], parts, num_warps=4)
    torch.autograd.graph.increment_version(out)
    return lse if return_lse else out

from shared.kernels.triton_compilation_log import install_triton_compilation_logger
install_triton_compilation_logger()
