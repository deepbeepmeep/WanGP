# Copyright (c) 2024-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Merge of the parts of an attention whose keys were split (split-KV): S partial outputs over disjoint key ranges, each with its
base-2 log-sum-exp, into one output, weighted by exp2(lse - max) in FP32 and rounded once to the output's dtype."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

_MERGE_RUNTIME = ['TOKENS', 'PS', 'PT', 'PH', 'LS', 'LH', 'OB', 'OT', 'OH']


@triton.jit(do_not_specialize=_MERGE_RUNTIME, do_not_specialize_on_alignment=_MERGE_RUNTIME)
def _merge_parts_kernel(P, L, O, TOKENS, PS, PT, PH, LS, LH, OB, OT, OH, PARTS: tl.constexpr, D: tl.constexpr, BLOCK_T: tl.constexpr):
    tokens = tl.program_id(0).to(tl.int64) * BLOCK_T + tl.arange(0, BLOCK_T)
    head = tl.program_id(1).to(tl.int64)
    valid = tokens < TOKENS
    top = tl.full((BLOCK_T,), float("-inf"), tl.float32)
    for part in tl.static_range(PARTS):
        top = tl.maximum(top, tl.load(L + part * LS + head * LH + tokens, valid, float("-inf")))
    total = tl.zeros((BLOCK_T,), tl.float32)
    acc = tl.zeros((BLOCK_T, D), tl.float32)
    c, rows = tl.arange(0, D)[None, :], tokens[:, None]
    for part in tl.static_range(PARTS):
        weight = libdevice.exp2(tl.load(L + part * LS + head * LH + tokens, valid, float("-inf")) - top)
        total += weight
        acc += tl.load(P + part * PS + rows * PT + head * PH + c, valid[:, None], 0.0).to(tl.float32) * weight[:, None]
    acc = acc * tl.div_rn(tl.full((BLOCK_T,), 1.0, tl.float32), total)[:, None]
    tl.store(O + rows * OT + head * OH + c, acc.to(O.dtype.element_ty), valid[:, None])


def merge_parts_(out, parts, lse):
    """out (1, tokens, heads, head_dim), through its strides (a range of heads of a larger output) = the merge of parts (S, tokens,
    heads, head_dim) with lse (S, heads, tokens): one pass, FP32 weights."""
    count, tokens, heads, dim = parts.shape
    block = 32 if dim >= 128 else 64
    _merge_parts_kernel[(triton.cdiv(tokens, block), heads)](
        parts, lse, out, tokens, parts.stride(0), parts.stride(1), parts.stride(2), lse.stride(0), lse.stride(1),
        out.stride(0), out.stride(1), out.stride(2), count, dim, block, num_warps=4)
    return out


_SPLIT_RUNTIME = ['SQZ', 'SQH', 'SQN', 'SKZ', 'SKH', 'SKN', 'SVZ', 'SVH', 'SVN', 'SOZ', 'SOH', 'SON', 'QO_LEN', 'KV_LEN', 'H', 'GROUPS', 'SPLITS', 'SPLIT_LEN']


# The per-block INT8 SageAttention 2 kernel of the Triton path (sageattention/triton/attn_qk_int8_per_block.py, Copyright (c) 2024 by
# SageAttention team, Apache License 2.0) with split keys: program z = batch * SPLITS + split attends the keys [split * SPLIT_LEN,
# ...) of its batch item and writes its output and lse at z. No mask; 64-bit offsets; head counts at run time (one compilation).
@triton.jit(do_not_specialize=_SPLIT_RUNTIME, do_not_specialize_on_alignment=_SPLIT_RUNTIME)
def _split_attn_kernel(Q, K, V, Q_scale, K_scale, Out, Lse, SQZ, SQH, SQN, SKZ, SKH, SKN, SVZ, SVH, SVN, SOZ, SOH, SON,
                       QO_LEN, KV_LEN, H, GROUPS, SPLITS, SPLIT_LEN, HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    start_m = tl.program_id(0)
    off_h = tl.program_id(1).to(tl.int64)
    off_out = tl.program_id(2).to(tl.int64)
    off_z = off_out // SPLITS
    kv_start = (off_out % SPLITS) * SPLIT_LEN
    local = tl.minimum(KV_LEN - kv_start, SPLIT_LEN)
    kv_head = off_h // GROUPS
    q_scale_ptr = Q_scale + (off_z * H + off_h) * tl.cdiv(QO_LEN, BLOCK_M) + start_m
    k_scale_ptr = K_scale + (off_z * (H // GROUPS) + kv_head) * tl.cdiv(KV_LEN, BLOCK_N) + kv_start // BLOCK_N
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    q = tl.load(Q + off_z * SQZ + off_h * SQH + offs_m[:, None] * SQN + offs_k[None, :], mask=offs_m[:, None] < QO_LEN)
    q_scale = tl.load(q_scale_ptr)
    k_ptrs = K + off_z * SKZ + kv_head * SKH + (kv_start + offs_n[None, :]) * SKN + offs_k[:, None]
    v_ptrs = V + off_z * SVZ + kv_head * SVH + (kv_start + offs_n[:, None]) * SVN + offs_k[None, :]
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    for start_n in range(0, local, BLOCK_N):
        k_mask = offs_n[None, :] < (local - start_n)
        k = tl.load(k_ptrs, mask=k_mask)
        k_scale = tl.load(k_scale_ptr)
        qk = tl.dot(q, k).to(tl.float32) * (q_scale * k_scale)
        qk += tl.where(k_mask, 0, -1.0e6)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        acc = acc * alpha[:, None]
        v = tl.load(v_ptrs, mask=offs_n[:, None] < (local - start_n))
        acc += tl.dot(p.to(tl.float16), v, out_dtype=tl.float16)
        m_i = m_ij
        k_ptrs += BLOCK_N * SKN
        k_scale_ptr += 1
        v_ptrs += BLOCK_N * SVN
    acc = acc / l_i[:, None]
    tl.store(Out + off_out * SOZ + off_h * SOH + offs_m[:, None] * SON + offs_k[None, :], acc.to(Out.type.element_ty), mask=offs_m[:, None] < QO_LEN)
    tl.store(Lse + off_out * QO_LEN * H + off_h * QO_LEN + offs_m, tl.log2(l_i) + m_i, mask=offs_m < QO_LEN)


def triton_split_attention(scratch, q, k, v, q_scale, k_scale, splits, block_m, launch):
    """The per-block Triton attention of q (batch, tokens, heads, dim) INT8 against k INT8 and 16-bit v (batch, kv tokens, kv heads,
    dim), its keys split: scratch (batch * splits, tokens, heads, dim) gets the parts; returns their lse (batch * splits, heads, tokens)."""
    batch, qo_len, heads, dim = q.shape
    kv_len, kv_heads = k.shape[1], k.shape[2]
    split_len = triton.cdiv(triton.cdiv(kv_len, 64), splits) * 64
    lse = torch.empty((batch * splits, heads, qo_len), dtype=torch.float32, device=q.device)
    num_warps, num_stages = launch or ((4, 3) if dim == 64 else (8, 4))
    _split_attn_kernel[(triton.cdiv(qo_len, block_m), heads, batch * splits)](
        q, k, v, q_scale, k_scale, scratch, lse, q.stride(0), q.stride(2), q.stride(1), k.stride(0), k.stride(2), k.stride(1),
        v.stride(0), v.stride(2), v.stride(1), scratch.stride(0), scratch.stride(2), scratch.stride(1), qo_len, kv_len, heads, heads // kv_heads,
        splits, split_len, dim, block_m, 64, num_warps=num_warps, num_stages=num_stages)
    return lse


from shared.kernels.triton_compilation_log import install_triton_compilation_logger  # noqa: E402 (after the kernels: Triton's line-number cache keys)
install_triton_compilation_logger()
