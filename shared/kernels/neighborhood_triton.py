# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: ANN001, ANN202, PLR0912, PLR0913, PLR0915
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Triton 3D neighborhood attention using NATTEN ``na3d`` semantics.

Vendored from the official LTX-2.5 implementation, which in turn vendors the
Comfy Kitchen kernel. One program handles a run of queries along W with online
softmax and fp32 accumulation, without materializing the attention scores.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


_NEG_INF = tl.constexpr(-3.0e38)


@triton.jit(do_not_specialize=['t_size', 'h_size', 'w_size', 'num_heads', 's_b', 's_t', 's_h', 's_w', 's_n', 'kt', 'kh', 'kw'],
            do_not_specialize_on_alignment=['t_size', 'h_size', 'w_size', 'num_heads', 's_b', 's_t', 's_h', 's_w', 's_n', 'kt', 'kh', 'kw'])
def _na3d_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    out_ptr,
    t_size,
    h_size,
    w_size,
    num_heads,
    s_b,
    s_t,
    s_h,
    s_w,
    s_n,
    scale,
    kt,
    kh,
    kw,
    causal_t: tl.constexpr,
    causal_h: tl.constexpr,
    causal_w: tl.constexpr,
    hd: tl.constexpr,
    hd_pad: tl.constexpr,
    block_q: tl.constexpr,
    block_k: tl.constexpr,
    is_fp32: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_th = tl.program_id(1)
    pid_bn = tl.program_id(2)

    t_q = pid_th // h_size
    h_q = pid_th % h_size
    base = (pid_bn // num_heads) * s_b + (pid_bn % num_heads) * s_n

    w_off = pid_w * block_q + tl.arange(0, block_q)
    w_valid = w_off < w_size
    d_off = tl.arange(0, hd_pad)
    d_mask = d_off < hd

    q_ptrs = q_ptr + base + t_q * s_t + h_q * s_h + w_off[:, None] * s_w + d_off[None, :]
    q_blk = tl.load(q_ptrs, mask=w_valid[:, None] & d_mask[None, :], other=0.0)

    if causal_t:
        t_lo = tl.maximum(t_q - kt + 1, 0)
        t_hi = t_q + 1
    else:
        t_lo = tl.minimum(tl.maximum(t_q - kt // 2, 0), t_size - kt)
        t_hi = t_lo + kt
    if causal_h:
        h_lo = tl.maximum(h_q - kh + 1, 0)
        h_hi = h_q + 1
    else:
        h_lo = tl.minimum(tl.maximum(h_q - kh // 2, 0), h_size - kh)
        h_hi = h_lo + kh

    w_q = tl.where(w_valid, w_off, w_size - 1)
    if causal_w:
        w_start = tl.maximum(w_q - kw + 1, 0)
        w_end = w_q + 1
        blk_first = tl.minimum(pid_w * block_q, w_size - 1)
        blk_last = tl.minimum(pid_w * block_q + block_q - 1, w_size - 1)
        w_lo = tl.maximum(blk_first - kw + 1, 0)
        w_hi = blk_last + 1
    else:
        w_start = tl.minimum(tl.maximum(w_q - kw // 2, 0), w_size - kw)
        w_end = w_start + kw
        blk_first = tl.minimum(pid_w * block_q, w_size - 1)
        blk_last = tl.minimum(pid_w * block_q + block_q - 1, w_size - 1)
        w_lo = tl.minimum(tl.maximum(blk_first - kw // 2, 0), w_size - kw)
        w_hi = tl.minimum(tl.maximum(blk_last - kw // 2, 0), w_size - kw) + kw

    m_i = tl.full((block_q,), _NEG_INF, dtype=tl.float32)
    l_i = tl.zeros((block_q,), dtype=tl.float32)
    acc = tl.zeros((block_q, hd_pad), dtype=tl.float32)

    for tk in range(t_lo, t_hi):
        for hk in range(h_lo, h_hi):
            plane = base + tk * s_t + hk * s_h
            for wk0 in range(w_lo, w_hi, block_k):
                wk = wk0 + tl.arange(0, block_k)
                kmask = wk < w_hi
                kv_ptrs = plane + wk[:, None] * s_w + d_off[None, :]
                kv_mask = kmask[:, None] & d_mask[None, :]
                k_blk = tl.load(k_ptr + kv_ptrs, mask=kv_mask, other=0.0)
                if is_fp32:
                    scores = tl.dot(q_blk, tl.trans(k_blk), input_precision="ieee") * scale
                else:
                    scores = tl.dot(q_blk, tl.trans(k_blk)) * scale
                visible = (wk[None, :] >= w_start[:, None]) & (wk[None, :] < w_end[:, None]) & kmask[None, :]
                scores = tl.where(visible, scores, _NEG_INF)
                m_new = tl.maximum(m_i, tl.max(scores, 1))
                alpha = tl.exp(m_i - m_new)
                probabilities = tl.exp(scores - m_new[:, None])
                l_i = l_i * alpha + tl.sum(probabilities, 1)
                v_blk = tl.load(v_ptr + kv_ptrs, mask=kv_mask, other=0.0)
                if is_fp32:
                    acc = acc * alpha[:, None] + tl.dot(probabilities, v_blk, input_precision="ieee")
                else:
                    acc = acc * alpha[:, None] + tl.dot(probabilities.to(v_blk.dtype), v_blk)
                m_i = m_new

    out = acc / tl.maximum(l_i, 1e-30)[:, None]
    out_ptrs = out_ptr + base + t_q * s_t + h_q * s_h + w_off[:, None] * s_w + d_off[None, :]
    tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=w_valid[:, None] & d_mask[None, :])


@triton.jit
def _fold_w_run(m_i, l_i, acc, q_blk, k_ptr, v_ptr, row_base, row_ok, w_lo, w_hi, w_start, w_end, s_w, d_off, d_mask, block_k: tl.constexpr, is_fp32: tl.constexpr):
    """Fold one key row (frame or plane, h) into the running online softmax; ``row_ok`` gates loads and visibility."""
    for wk0 in range(w_lo, w_hi, block_k):
        wk = wk0 + tl.arange(0, block_k)
        kmask = wk < w_hi
        kv_ptrs = row_base + wk[:, None] * s_w + d_off[None, :]
        kv_mask = kmask[:, None] & d_mask[None, :] & row_ok
        k_blk = tl.load(k_ptr + kv_ptrs, mask=kv_mask, other=0.0)
        if is_fp32:
            scores = tl.dot(q_blk, tl.trans(k_blk), input_precision="ieee")
        else:
            scores = tl.dot(q_blk, tl.trans(k_blk))
        visible = (wk[None, :] >= w_start[:, None]) & (wk[None, :] < w_end[:, None]) & kmask[None, :] & row_ok
        scores = tl.where(visible, scores, _NEG_INF)
        m_new = tl.maximum(m_i, tl.max(scores, 1))
        alpha = tl.exp(m_i - m_new)
        probabilities = tl.where(visible, tl.exp(scores - m_new[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(probabilities, 1)
        v_blk = tl.load(v_ptr + kv_ptrs, mask=kv_mask, other=0.0)
        if is_fp32:
            acc = acc * alpha[:, None] + tl.dot(probabilities, v_blk, input_precision="ieee")
        else:
            acc = acc * alpha[:, None] + tl.dot(probabilities.to(v_blk.dtype), v_blk)
        m_i = m_new
    return m_i, l_i, acc


@triton.jit(do_not_specialize=['own_t_size', 'h_size', 'w_size', 'num_heads', 'bn_count', 'own_s_b', 'other_s_b', 's_t', 's_h', 's_w', 's_n', 'kt', 'kh', 'kw'],
            do_not_specialize_on_alignment=['own_t_size', 'h_size', 'w_size', 'num_heads', 'bn_count', 'own_s_b', 'other_s_b', 's_t', 's_h', 's_w', 's_n', 'kt', 'kh', 'kw'])
def _joint_kernel(
    q_ptr,
    own_k_ptr,
    own_v_ptr,
    other_k_ptr,
    other_v_ptr,
    slot_ptr,
    out_ptr,
    own_t_size,
    h_size,
    w_size,
    num_heads,
    bn_count,
    own_s_b,
    other_s_b,
    s_t,
    s_h,
    s_w,
    s_n,
    kt,
    kh,
    kw,
    num_slots: tl.constexpr,
    hd: tl.constexpr,
    hd_pad: tl.constexpr,
    block_q: tl.constexpr,
    block_k: tl.constexpr,
    is_fp32: tl.constexpr,
):
    """Queries of one stream: their own centered ``kt x kh x kw`` window (clamped and masked, ``kt = 1`` for planes)
    plus the ``kh x kw`` window on ``num_slots`` rows of the other stream, all in one online softmax."""
    pid_w = tl.program_id(0)
    h_q = tl.program_id(1)
    pid_tbn = tl.program_id(2)
    t_q = pid_tbn // bn_count
    pid_bn = pid_tbn % bn_count
    batch_i = pid_bn // num_heads
    head_i = pid_bn % num_heads
    own_base = batch_i.to(tl.int64) * own_s_b + head_i.to(tl.int64) * s_n
    other_base = batch_i.to(tl.int64) * other_s_b + head_i.to(tl.int64) * s_n

    w_off = pid_w * block_q + tl.arange(0, block_q)
    w_valid = w_off < w_size
    d_off = tl.arange(0, hd_pad)
    d_mask = d_off < hd
    q_ptrs = q_ptr + own_base + t_q.to(tl.int64) * s_t + h_q.to(tl.int64) * s_h + w_off[:, None] * s_w + d_off[None, :]
    q_blk = tl.load(q_ptrs, mask=w_valid[:, None] & d_mask[None, :], other=0.0)

    w_q = tl.where(w_valid, w_off, 0)
    w_start = tl.maximum(w_q - kw // 2, 0)
    w_end = tl.minimum(w_q - kw // 2 + kw, w_size)
    blk_last = tl.minimum(pid_w * block_q + block_q - 1, w_size - 1)
    w_lo = tl.maximum(pid_w * block_q - kw // 2, 0)
    w_hi = tl.minimum(blk_last - kw // 2 + kw, w_size)

    m_i = tl.full((block_q,), _NEG_INF, dtype=tl.float32)
    l_i = tl.zeros((block_q,), dtype=tl.float32)
    acc = tl.zeros((block_q, hd_pad), dtype=tl.float32)

    for d_t in range(kt):
        tk = t_q - kt // 2 + d_t
        t_ok = (tk >= 0) & (tk < own_t_size)
        tk_c = tl.maximum(tl.minimum(tk, own_t_size - 1), 0)
        for d_h in range(kh):
            hk = h_q - kh // 2 + d_h
            row_ok = t_ok & (hk >= 0) & (hk < h_size)
            hk_c = tl.maximum(tl.minimum(hk, h_size - 1), 0)
            m_i, l_i, acc = _fold_w_run(m_i, l_i, acc, q_blk, own_k_ptr, own_v_ptr, own_base + tk_c.to(tl.int64) * s_t + hk_c.to(tl.int64) * s_h, row_ok, w_lo, w_hi, w_start, w_end, s_w, d_off, d_mask, block_k=block_k, is_fp32=is_fp32)

    for slot in range(num_slots):
        row = tl.load(slot_ptr + t_q * num_slots + slot)
        slot_ok = row >= 0
        row_c = tl.maximum(row, 0)
        for d_h in range(kh):
            hk = h_q - kh // 2 + d_h
            row_ok = slot_ok & (hk >= 0) & (hk < h_size)
            hk_c = tl.maximum(tl.minimum(hk, h_size - 1), 0)
            m_i, l_i, acc = _fold_w_run(m_i, l_i, acc, q_blk, other_k_ptr, other_v_ptr, other_base + row_c.to(tl.int64) * s_t + hk_c.to(tl.int64) * s_h, row_ok, w_lo, w_hi, w_start, w_end, s_w, d_off, d_mask, block_k=block_k, is_fp32=is_fp32)

    out = acc / tl.maximum(l_i, 1e-30)[:, None]
    out_ptrs = out_ptr + own_base + t_q.to(tl.int64) * s_t + h_q.to(tl.int64) * s_h + w_off[:, None] * s_w + d_off[None, :]
    tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=w_valid[:, None] & d_mask[None, :])


def joint_na3d(qkv_list: list[torch.Tensor], video_slots: torch.Tensor, keyframe_slots: torch.Tensor, kernel_size: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Joint video + keyframe-plane neighborhood attention over contiguous ``(B, A, H, W, heads, head_dim)`` tensors (Q pre-scaled).

    ``video_slots`` ``(T, 2)`` / ``keyframe_slots`` ``(P, 2)`` hold the nearest rows of the other stream (``-1`` = empty)."""
    q, k, v, keyframe_q, keyframe_k, keyframe_v = qkv_list
    qkv_list.clear()
    batch, time, height, width, num_heads, head_dim = q.shape
    planes = keyframe_q.shape[1]
    kt, kh, kw = kernel_size
    video_out, keyframe_out = q, keyframe_q
    common = dict(hd=head_dim, hd_pad=max(16, triton.next_power_of_2(head_dim)), block_q=16, block_k=max(16, min(32, triton.next_power_of_2(min(width, 16 + kw)))), is_fp32=q.dtype == torch.float32, num_warps=4)
    strides = q.stride(1), q.stride(2), q.stride(3), q.stride(4)
    bn_count = batch * num_heads
    video_slots = video_slots.to(device=q.device, dtype=torch.int32).contiguous()
    keyframe_slots = keyframe_slots.to(device=q.device, dtype=torch.int32).contiguous()
    _joint_kernel[(triton.cdiv(width, 16), height, time * bn_count)](q, k, v, keyframe_k, keyframe_v, video_slots, video_out, time, height, width, num_heads, bn_count, q.stride(0), keyframe_q.stride(0), *strides, kt=kt, kh=kh, kw=kw, num_slots=video_slots.shape[1], **common)
    _joint_kernel[(triton.cdiv(width, 16), height, planes * bn_count)](keyframe_q, keyframe_k, keyframe_v, k, v, keyframe_slots, keyframe_out, planes, height, width, num_heads, bn_count, keyframe_q.stride(0), q.stride(0), *strides, kt=1, kh=kh, kw=kw, num_slots=keyframe_slots.shape[1], **common)
    torch.autograd.graph.increment_version(video_out)
    torch.autograd.graph.increment_version(keyframe_out)
    return video_out, keyframe_out


def na3d(qkv_list: list[torch.Tensor], kernel_size: tuple[int, int, int], scale: float = 1.0) -> torch.Tensor:
    """Run neighborhood attention over ``(B, T, H, W, heads, head_dim)`` tensors."""
    q, k, v = qkv_list
    qkv_list.clear()
    batch, t, h, w, num_heads, head_dim = q.shape
    kt, kh, kw = (min(kernel, dim) for kernel, dim in zip(kernel_size, (t, h, w), strict=True))
    out = q

    head_dim_pad = max(16, triton.next_power_of_2(head_dim))
    block_q = 16
    block_k = max(16, min(32, triton.next_power_of_2(min(w, block_q + kw))))
    grid = triton.cdiv(w, block_q), t * h, batch * num_heads
    _na3d_kernel[grid](
        q,
        k,
        v,
        out,
        t,
        h,
        w,
        num_heads,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        q.stride(4),
        scale,
        kt=kt,
        kh=kh,
        kw=kw,
        causal_t=False,
        causal_h=False,
        causal_w=False,
        hd=head_dim,
        hd_pad=head_dim_pad,
        block_q=block_q,
        block_k=block_k,
        is_fp32=q.dtype == torch.float32,
        num_warps=4,
    )
    torch.autograd.graph.increment_version(out)
    return out


from shared.kernels.triton_compilation_log import install_triton_compilation_logger
install_triton_compilation_logger()
