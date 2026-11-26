# SPDX-License-Identifier: Apache-2.0
# Adapted from DeepBeepMeep/SparseGP v1.0.0.
"""JIT fallback: skip empty key tiles; INT8 QK and 16-bit PV accumulate in 32 bits."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack(X, Y, S, D: tl.constexpr, ROWS: tl.constexpr):
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    d = tl.arange(0, D)
    offset = r[:, None].to(tl.int64) * D + d[None, :]
    x = tl.load(X + offset).to(tl.float32)
    peak = tl.max(tl.abs(x), 1)
    step = tl.where(peak > 0, peak / 127.0, 1.0)
    y = x / step[:, None]
    tl.store(Y + offset, tl.where(y >= 0.0, y + 0.5, y - 0.5).to(tl.int8))
    tl.store(S + r, step)


@triton.jit
def _attend(
    Q,
    K,
    V,
    QS,
    KS,
    BITS,
    O,
    QP: tl.constexpr,
    KP: tl.constexpr,
    KEYS: tl.constexpr,
    QB: tl.constexpr,
    SCALE: tl.constexpr,
    REPS: tl.constexpr,
    CAUSAL: tl.constexpr,
    D: tl.constexpr,
    M: tl.constexpr,
):
    g = tl.program_id(1).to(tl.int64)
    tile = tl.program_id(0)
    r = tile * M + tl.arange(0, M)
    d = tl.arange(0, D)
    c = tl.arange(0, 64)
    nk = tl.cdiv(KEYS, 64)
    base = BITS + (g * tl.cdiv(QP, QB) + tile * M // QB) * nk * 2
    query = tl.load(Q + ((g * QP + r[:, None]) * D + d[None, :]), r[:, None] < QP, 0)
    qs = tl.load(QS + g * QP + r, r < QP, 0.0)
    maximum = tl.full((M,), -float("inf"), tl.float32)
    denom = tl.full((M,), 0, tl.float32)
    num = tl.full((M, D), 0, tl.float32)
    for t in range(nk + (tl.cdiv(nk, 64) if REPS else 0)):
        col = t * 64 + c
        rep = t >= nk
        if rep:
            j = col - nk * 64
            lo = tl.load(base + 2 * j, j < nk, 0).to(tl.uint32)
            hi = tl.load(base + 2 * j + 1, j < nk, 0).to(tl.uint32)
            # Portable SWAR popcount, lowered to native bit-count instructions.
            a = lo - ((lo >> 1) & 0x55555555)
            b = hi - ((hi >> 1) & 0x55555555)
            a = (a & 0x33333333) + ((a >> 2) & 0x33333333)
            b = (b & 0x33333333) + ((b >> 2) & 0x33333333)
            a = ((((a + (a >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24) & 255
            b = ((((b + (b >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24) & 255
            mass = tl.where(j < nk, tl.minimum(64, KEYS - j * 64) - a.to(tl.int32) - b.to(tl.int32), 0)
        else:
            bits = tl.load(base + t * 2 + c // 32).to(tl.uint32)
            mass = ((bits >> (c % 32)) & 1).to(tl.int32) * (col < KEYS)
        if tl.max(mass, 0) > 0:
            k = tl.load(K + ((g * KP + col[:, None]) * D + d[None, :]), col[:, None] < KP, 0)
            v = tl.load(V + ((g * KP + col[:, None]) * D + d[None, :]), col[:, None] < KP, 0)
            ks = tl.load(KS + g * KP + col, col < KP, 0.0)
            score = tl.dot(query, tl.trans(k)).to(tl.float32) * (
                qs[:, None] * ks[None, :] * (SCALE * 1.4426950408889634)
            )
            score += tl.log2(tl.maximum(mass, 1).to(tl.float32))[None, :]
            ok = (mass > 0)[None, :]
            if CAUSAL:
                score = tl.where(col[None, :] <= r[:, None], score, -float("inf"))
            score = tl.where(ok, score, -float("inf"))
            m = tl.maximum(maximum, tl.max(score, 1))
            safe = tl.where(m == -float("inf"), 0.0, m)
            alpha = tl.exp2(maximum - safe)
            p = tl.exp2(score - safe[:, None])
            num = num * alpha[:, None] + tl.dot(p.to(tl.float16), v)
            denom = denom * alpha + tl.sum(p, 1)
            maximum = m
    tl.store(
        O + ((g * QP + r[:, None]) * D + d[None, :]),
        num / tl.where(denom > 0, denom, 1.0)[:, None],
        r[:, None] < QP,
    )


def pack(x):
    out = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty(x.shape[:-1], device=x.device, dtype=torch.float32)
    _pack[(scales.numel() // 16,)](x, out, scales, x.shape[-1], 16)
    return out, scales


def _simt(device):
    # Turing and RDNA 2 have packed four-byte integer dot products.
    if torch.version.hip:
        return torch.cuda.get_device_properties(device).gcnArchName.startswith("gfx10")
    return torch.cuda.get_device_capability(device) < (8, 0)


@triton.jit
def _attend_dp4a(
    Q,
    K,
    V,
    QS,
    KS,
    BITS,
    O,
    QP: tl.constexpr,
    KP: tl.constexpr,
    T: tl.constexpr,
    QB: tl.constexpr,
    SCALE: tl.constexpr,
    REPS: tl.constexpr,
    CAUSAL: tl.constexpr,
    D: tl.constexpr,
    HIP: tl.constexpr,
):
    row, g = tl.program_id(0), tl.program_id(1).to(tl.int64)
    nk = tl.cdiv(T, 64)
    bits = BITS + (g * tl.cdiv(QP, QB) + row // QB) * nk * 2
    ds = tl.arange(0, D)
    wd = tl.arange(0, D // 4)
    c = tl.arange(0, 64)
    q = tl.load(Q.to(tl.pointer_type(tl.int32)) + (g * QP + row) * (D // 4) + wd)
    qs = tl.load(QS + g * QP + row)
    maximum = -float("inf")
    denom = 0.0
    num = tl.full((D,), 0.0, tl.float32)
    for tile in range(nk + (tl.cdiv(nk, 64) if REPS else 0)):
        col = tile * 64 + c
        if tile < nk:
            word = tl.load(bits + tile * 2 + c // 32).to(tl.uint32)
            mass = (((word >> (c % 32)) & 1).to(tl.int32)) * (col < T)
        else:
            j = col - nk * 64
            lo = tl.load(bits + 2 * j, j < nk, 0)
            hi = tl.load(bits + 2 * j + 1, j < nk, 0)
            if HIP:
                used = tl.inline_asm_elementwise(
                    "v_bcnt_u32_b32 $0, $1, 0", "=v,v", [lo], dtype=tl.int32, is_pure=True, pack=1
                ) + tl.inline_asm_elementwise(
                    "v_bcnt_u32_b32 $0, $1, 0", "=v,v", [hi], dtype=tl.int32, is_pure=True, pack=1
                )
            else:
                used = tl.inline_asm_elementwise(
                    "popc.b32 $0, $1;", "=r,r", [lo], dtype=tl.int32, is_pure=True, pack=1
                ) + tl.inline_asm_elementwise(
                    "popc.b32 $0, $1;", "=r,r", [hi], dtype=tl.int32, is_pure=True, pack=1
                )
            mass = tl.where(j < nk, tl.minimum(64, T - j * 64) - used, 0)
        if CAUSAL:
            mass = tl.where(col <= row, mass, 0)
        if tl.max(mass, 0) > 0:
            k = tl.load(
                K.to(tl.pointer_type(tl.int32)) + ((g * KP + col[:, None]) * (D // 4) + wd[None, :]),
                col[:, None] < KP,
                0,
            )
            if HIP:
                dot = tl.inline_asm_elementwise(
                    "v_dot4_i32_i8 $0, $1, $2, 0", "=v,v,v", [k, q[None, :]], dtype=tl.int32, is_pure=True, pack=1
                )
            else:
                dot = tl.inline_asm_elementwise(
                    "dp4a.s32.s32 $0, $1, $2, 0;", "=r,r,r", [k, q[None, :]], dtype=tl.int32, is_pure=True, pack=1
                )
            scale = tl.load(KS + g * KP + col, col < KP, 0) * qs * SCALE * 1.4426950408889634
            score = tl.sum(dot, 1).to(tl.float32) * scale + tl.log2(tl.maximum(mass, 1).to(tl.float32))
            score = tl.where(mass > 0, score, -float("inf"))
            m = tl.maximum(maximum, tl.max(score, 0))
            alpha = tl.exp2(maximum - m)
            p = tl.exp2(score - m)
            v = tl.load(V + ((g * KP + col[:, None]) * D + ds[None, :]), col[:, None] < KP, 0)
            num = num * alpha + tl.sum(v * p[:, None], 0)
            denom = denom * alpha + tl.sum(p, 0)
            maximum = m
    tl.store(O + (g * QP + row) * D + ds, num / tl.maximum(denom, 1.0e-20))


def attend(q, k, v, qs, ks, bits, keys, block_q, scale, representatives, causal):
    groups, qp, d = q.shape
    out = torch.empty_like(q, dtype=torch.float16)
    if _simt(q.device):
        _attend_dp4a[(qp, groups)](
            q,
            k,
            v,
            qs,
            ks,
            bits,
            out,
            qp,
            k.size(1),
            keys,
            block_q,
            scale,
            representatives,
            causal,
            d,
            bool(torch.version.hip),
            num_warps=4,
        )
        return out
    rows = 32 if d == 128 or torch.version.hip else 64
    _attend[(triton.cdiv(qp, rows), groups)](
        q,
        k,
        v,
        qs,
        ks,
        bits,
        out,
        qp,
        k.size(1),
        keys,
        block_q,
        scale,
        representatives,
        causal,
        d,
        rows,
        num_warps=4,
        num_stages=1 if torch.version.hip else 2,
    )
    return out


@triton.jit
def _select(
    Q,
    K,
    F,
    STATS,
    CUT,
    BITS,
    N: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    WRITE: tl.constexpr,
):
    tile, kb, group = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    r = tile * 16 + tl.arange(0, 16)
    c = tl.arange(0, 64)
    d = tl.arange(0, D)
    ix = (group * N + r) * N + kb
    flag = tl.load(F + ix, r < N, 0)
    score = tl.full((16, 64), 0.0, tl.float32)
    if tl.max((flag == 1).to(tl.int32), 0) > 0:
        a = tl.load(Q + ((group * N + r[:, None]) * D + d[None, :]), r[:, None] < N, 0.0)
        b = tl.load(K + (((group * N + kb) * 64 + c[:, None]) * D + d[None, :]))
        score = tl.dot(a, tl.trans(b)) * SCALE
    valid = (kb * 64 + c) < T
    if WRITE:
        cut = tl.load(CUT + group * N + r, r < N, 0.0)
        chosen = valid[None, :] & ((flag < 0)[:, None] | ((flag == 1)[:, None] & (score > cut[:, None])))
        lanes = tl.arange(0, 32).to(tl.uint32)
        parts = tl.reshape(chosen, (16, 2, 32))
        packed = tl.sum(parts.to(tl.uint32) << lanes[None, None, :], 2)
        tl.store(BITS + ix[:, None] * 2 + tl.arange(0, 2)[None, :], packed, r[:, None] < N)
    else:
        keep = (flag == 1)[:, None] & valid[None, :]
        value = tl.where(keep, score, 0.0)
        tl.store(STATS + ix * 3, tl.sum(keep.to(tl.int32), 1), r < N)
        tl.store(STATS + ix * 3 + 1, tl.sum(value, 1), r < N)
        tl.store(STATS + ix * 3 + 2, tl.sum(value * value, 1), r < N)


@triton.jit
def _select_simt(
    Q,
    K,
    F,
    STATS,
    CUT,
    BITS,
    N: tl.constexpr,
    T: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    WRITE: tl.constexpr,
):
    row, kb, g = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    ix = (g * N + row) * N + kb
    flag = tl.load(F + ix)
    c = tl.arange(0, 64)
    d = tl.arange(0, D)
    valid = kb * 64 + c < T
    score = tl.full((64,), 0.0, tl.float32)
    if flag == 1:
        q = tl.load(Q + (g * N + row) * D + d).to(tl.float32)
        k = tl.load(K + (((g * N + kb) * 64 + c[:, None]) * D + d[None, :])).to(tl.float32)
        score = tl.sum(k * q[None, :], 1) * SCALE
    if WRITE:
        chosen = valid & ((flag < 0) | ((flag == 1) & (score > tl.load(CUT + g * N + row))))
        p = tl.reshape(chosen, (2, 32)).to(tl.uint32) << tl.arange(0, 32)[None, :]
        tl.store(BITS + ix * 2 + tl.arange(0, 2), tl.sum(p, 1))
    else:
        kept = valid & (flag == 1)
        v = tl.where(kept, score, 0.0)
        tl.store(STATS + ix * 3, tl.sum(kept.to(tl.int32), 0))
        tl.store(STATS + ix * 3 + 1, tl.sum(v, 0))
        tl.store(STATS + ix * 3 + 2, tl.sum(v * v, 0))


def select(qmean, keys, flags, tokens, scale):
    g, n, d = qmean.shape
    stats = torch.empty((g, n, n, 3), device=keys.device, dtype=torch.float32)
    bits = torch.empty((g, n, n, 2), device=keys.device, dtype=torch.int32)
    simt = _simt(keys.device)
    grid = (n if simt else triton.cdiv(n, 16), n, g)
    kernel = _select_simt if simt else _select
    kernel[grid](qmean, keys, flags, stats, stats, bits, n, tokens, d, scale, False)
    count, total, squared = stats.sum(2).unbind(-1)
    count.clamp_min_(1.0)
    mean = total / count
    cut = mean + 1.125 * (torch.clamp_min(squared / count - mean.square(), 0.0) + 1.0e-6).sqrt()
    kernel[grid](qmean, keys, flags, stats, cut, bits, n, tokens, d, scale, True)
    return bits
