# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""SM120 Sage2 per-block INT8/FP16 attention for wide heads and reusable boolean mask tiles.
The per-block scales, online softmax and FP16 tile product follow SageAttention's attn_qk_int8_per_block
(https://github.com/thu-ml/SageAttention, Apache-2.0)."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .dispatch import choose


@triton.jit(do_not_specialize=["T", "N", "H", "HK"])
def _wide_forward(Q, K, V, QS, KS, OUT, LSE, T, N, H, HK, QZ, QT, QH, KZ, KT, KH, VZ, VT, VH, OZ, OT, OH,
                  TAIL: tl.constexpr):
    # Two CTAs compute the two halves of a 256-channel output. Repeating its INT8 QK dot costs less
    # than spilling the wide FP32 numerator. The key tile and FP16 PV accumulation match Sage2's.
    tile = tl.program_id(0)
    head = tl.program_id(1).to(tl.int64)
    zp = tl.program_id(2).to(tl.int64)
    batch, half = zp // 2, zp % 2
    kv_head = head // (H // HK)
    row = tile.to(tl.int64) * 128 + tl.arange(0, 128)
    d = tl.arange(0, 256)
    c = half * 128 + tl.arange(0, 128)
    q = tl.load(Q + batch * QZ + head * QH + row[:, None] * QT + d[None, :], row[:, None] < T, 0)
    qs = tl.load(QS + (batch * H + head) * tl.cdiv(T, 128) + tile)
    hi = tl.full((128,), -float("inf"), tl.float32)
    denom = tl.full((128,), 1., tl.float32)
    numer = tl.full((128, 128), 0., tl.float32)
    for block in range(tl.cdiv(N, 64)):
        key = block.to(tl.int64) * 64 + tl.arange(0, 64)
        valid = (key < N) if TAIL else tl.full((64,), True, tl.int1)
        k = tl.load(K + batch * KZ + kv_head * KH + key[:, None] * KT + d[None, :], valid[:, None], 0)
        v = tl.load(V + batch * VZ + kv_head * VH + key[:, None] * VT + c[None, :], valid[:, None], 0.)
        ks = tl.load(KS + (batch * HK + kv_head) * tl.cdiv(N, 64) + block)
        score = libdevice.mul_rn(tl.dot(q, tl.trans(k), out_dtype=tl.int32).to(tl.float32), qs * ks)
        if TAIL:
            score += tl.where(valid[None, :], 0., -1.e6)
        top = tl.maximum(hi, tl.max(score, 1))
        p = tl.exp2(score - top[:, None])
        factor = tl.exp2(hi - top)
        # The stock D256 kernel assigns eight adjacent scores to each warp. Add the two probabilities
        # per lane, reduce the four lanes at XOR distances 2 and 1, then the eight warp sums at 4, 2, 1.
        # Splitting output channels changes Triton's default order. Preserve these roundings through sampling.
        sums = tl.sum(tl.reshape(p, (128, 8, 8)), 2)
        for stage in tl.static_range(3):
            left, right = tl.split(tl.permute(tl.reshape(sums, (128, 2, 4 >> stage)), 0, 2, 1))
            sums = left + right
        denom = tl.fma(denom, factor, tl.reshape(sums, (128,)))
        numer = numer * factor[:, None] + tl.dot(p.to(tl.float16), v, out_dtype=tl.float16)
        hi = top
    tl.store(OUT + batch * OZ + head * OH + row[:, None] * OT + c[None, :], numer / denom[:, None], row[:, None] < T)
    if LSE is not None:
        if half == 0:
            tl.store(LSE + (batch * H + head) * T + row, hi + tl.log2(denom), row < T)


@triton.jit(do_not_specialize=["N", "MASK_HEADS"])
def _active_tiles(MASK, TILES, COUNTS, N, MASK_HEADS, MZ, MH, MN, GROUPS: tl.constexpr):
    owner = tl.program_id(0).to(tl.int64)
    batch, head = owner // MASK_HEADS, owner % MASK_HEADS
    tile = tl.arange(0, GROUPS)
    key = tile[:, None] * 64 + tl.arange(0, 64)[None, :]
    keep = tl.load(MASK + batch * MZ + head * MH + key.to(tl.int64) * MN, key < N, 0)
    active = tl.max(keep.to(tl.int32), 1)
    index = tl.cumsum(active) - 1
    tl.store(TILES + owner * tl.cdiv(N, 64) + index, tile, active != 0)
    tl.store(COUNTS + owner, tl.sum(active, 0))


@triton.jit(do_not_specialize=["T", "N"])
def _classify_tiles(MASK, FLAGS, T, N, MZ, MT, MN):
    qt, kt, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2).to(tl.int64)
    row = qt.to(tl.int64) * 64 + tl.arange(0, 64)
    key = kt.to(tl.int64) * 64 + tl.arange(0, 64)
    keep = tl.load(MASK + batch * MZ + row[:, None] * MT + key[None, :] * MN,
                   (row[:, None] < T) & (key[None, :] < N), False)
    count = tl.sum(tl.sum(keep.to(tl.int32), 0), 0)
    tl.store(FLAGS + (batch * tl.cdiv(T, 64) + qt) * tl.cdiv(N, 64) + kt,
             tl.where(count == 0, 0, tl.where(count == 64 * 64, 2, 1)))


@triton.jit(do_not_specialize=["BLOCKS"])
def _index_tiles(FLAGS, TILES, COUNTS, BLOCKS, GROUPS: tl.constexpr):
    owner = tl.program_id(0).to(tl.int64)
    tile = tl.arange(0, GROUPS)
    flag = tl.load(FLAGS + owner * BLOCKS + tile, tile < BLOCKS, 0)
    active = (flag != 0).to(tl.int32)
    index = tl.cumsum(active) - 1
    # A complemented index marks a partial tile. Full tiles need no mask load in the attention loop.
    tl.store(TILES + owner * BLOCKS + index, tl.where(flag == 2, tile, ~tile), active != 0)
    tl.store(COUNTS + owner, tl.sum(active, 0))


@triton.jit(do_not_specialize=["T", "N", "H", "HK"])
def _masked_forward(Q, K, V, QS, KS, OUT, MASK, LSE, TILES, COUNTS, T, N, H, HK, D: tl.constexpr,
                    QZ, QT, QH, KZ, KT, KH, VZ, VT, VH, OZ, OT, OH, MZ, MH, MT, MN, Q_GROUP: tl.constexpr, ROWS: tl.constexpr,
                    BATCH_MASK: tl.constexpr, HEAD_MASK: tl.constexpr, TAIL: tl.constexpr):
    batch = tl.program_id(2).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    kv_head = head // (H // HK)
    row = tl.program_id(0).to(tl.int64) * ROWS + tl.arange(0, ROWS)
    d, col = tl.arange(0, D), tl.arange(0, 64)
    q = tl.load(Q + batch * QZ + head * QH + row[:, None] * QT + d[None, :], row[:, None] < T, 0)
    query_scales = QS + (batch * H + head) * tl.cdiv(T, Q_GROUP)
    if ROWS == 64:
        qs = tl.load(query_scales + tl.program_id(0))
    else:
        qs = tl.load(query_scales + row // Q_GROUP, row < T, 0.)
    hi = tl.full((ROWS,), -float("inf"), tl.float32)
    denom = tl.full((ROWS,), 1., tl.float32)
    numer = tl.full((ROWS, D), 0., tl.float32)
    owner = (batch if BATCH_MASK else 0) * (H if HEAD_MASK else 1) + (head if HEAD_MASK else 0)
    blocks = tl.cdiv(N, 64)
    if ROWS == 64:
        owner = owner * tl.cdiv(T, ROWS) + tl.program_id(0)
    count = tl.load(COUNTS + owner)
    for i in range(count):
        code = tl.load(TILES + owner * blocks + i)
        tile = tl.where(code < 0, ~code, code) if ROWS == 64 else code
        key = tile.to(tl.int64) * 64 + col
        valid = (key < N) if TAIL else tl.full((64,), True, tl.int1)
        k = tl.load(K + batch * KZ + kv_head * KH + key[:, None] * KT + d[None, :], valid[:, None], 0)
        v = tl.load(V + batch * VZ + kv_head * VH + key[:, None] * VT + d[None, :], valid[:, None], 0.)
        ks = tl.load(KS + (batch * HK + kv_head) * blocks + tile)
        dot = tl.dot(q, tl.trans(k), out_dtype=tl.int32).to(tl.float32)
        if ROWS == 64:
            score = dot * (qs * ks)
        else:
            score = dot * (qs * ks)[:, None]
        if ROWS == 128:
            keep = tl.load(MASK + batch * MZ + head * MH + key * MN, valid, False)
            score += tl.where(keep[None, :], 0., -1.e6)
        else:
            # A full tile issues no mask loads. Keeping the bias on the common arithmetic path preserves
            # the FP16 tile product of Sage2's masked forward, including for rows with few visible keys.
            keep = tl.load(MASK + batch * MZ + row[:, None] * MT + key[None, :] * MN,
                           (code < 0) & (row[:, None] < T) & valid[None, :], False)
            score += tl.where(keep | (code >= 0), 0., -1.e6)
        top = tl.maximum(hi, tl.max(score, 1))
        p = tl.exp2(score - top[:, None])
        factor = tl.exp2(hi - top)
        denom = denom * factor + tl.sum(p, 1)
        numer = numer * factor[:, None] + tl.dot(p.to(tl.float16), v, out_dtype=tl.float16)
        hi = top
    tl.store(OUT + batch * OZ + head * OH + row[:, None] * OT + d[None, :], numer / denom[:, None], row[:, None] < T)
    if LSE is not None:
        tl.store(LSE + (batch * H + head) * T + row, hi + tl.log2(denom), row < T)


def attention(q, k, v, qs, ks, out, mask, lse, quant_rows):
    """Handle wide unmasked heads or compact the live 64-key tiles once per distinct mask."""
    batch, rows, heads, dim = q.shape
    keys, kv_heads = k.shape[1:3]
    if (rows < 2048 or not 2048 <= keys <= 32768
            or quant_rows not in (64, 128) or not kv_heads or not heads or heads % kv_heads
            or k.shape != v.shape or k.shape[0] != batch or k.shape[3] != dim
            or any(x.stride(-1) != 1 for x in (q, k, v, out)) or not qs.is_contiguous() or not ks.is_contiguous()):
        return False
    if mask is None:
        if dim != 256 or quant_rows != 128:
            return False
        choose("Sage2 Wide-Head Attention", owned_triton=True)
        _wide_forward[(triton.cdiv(rows, 128), heads, batch * 2)](
            q, k, v, qs, ks, out, lse, rows, keys, heads, kv_heads, *q.stride()[:3], *k.stride()[:3],
            *v.stride()[:3], *out.stride()[:3], keys % 64 != 0, num_warps=8, num_stages=2)
        torch.autograd.graph.increment_version(out)
        return True
    if mask.dtype != torch.bool or dim not in (64, 128):
        return False
    mz, mh, mt, mn = mask.stride()
    if mt and (dim != 64 or quant_rows != 64 or mh or mn != 1 or heads < 8 or not 4096 <= rows <= 32768):
        return False
    choose("Sage2 Boolean Mask Tiles", owned_triton=True)
    mask_heads = heads if mh else 1
    owners = (batch if mz else 1) * mask_heads
    tile_rows = 64 if mt else 128
    query_tiles = triton.cdiv(rows, tile_rows)
    blocks = triton.cdiv(keys, 64)
    mask_batches = owners
    if mt:
        owners *= query_tiles
    tiles = torch.empty((owners, blocks), dtype=torch.int32, device=q.device)
    counts = torch.empty((owners,), dtype=torch.int32, device=q.device)
    span = triton.next_power_of_2(blocks)
    if mt:
        flags = torch.empty((owners, blocks), dtype=torch.uint8, device=q.device)
        _classify_tiles[(query_tiles, blocks, mask_batches)](mask, flags, rows, keys, mz, mt, mn, num_warps=4)
        _index_tiles[(owners,)](flags, tiles, counts, blocks, span, num_warps=4)
    else:
        _active_tiles[(owners,)](mask, tiles, counts, keys, mask_heads, mz, mh, mn, span, num_warps=4)
    _masked_forward[(query_tiles, heads, batch)](
        q, k, v, qs, ks, out, mask, lse, tiles, counts, rows, keys, heads, kv_heads, dim,
        *q.stride()[:3], *k.stride()[:3], *v.stride()[:3], *out.stride()[:3], mz, mh, mt, mn, quant_rows, tile_rows,
        bool(mz), bool(mh), keys % 64 != 0, num_warps=4 if mt else 8, num_stages=2)
    torch.autograd.graph.increment_version(out)
    return True

from shared.kernels.triton_compilation_log import install_triton_compilation_logger
install_triton_compilation_logger()
