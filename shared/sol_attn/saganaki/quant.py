# SPDX-License-Identifier: Apache-2.0
# Bundled from ComfyUI-sol-attn v0.6.2 (commit 930a4d6).
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""INT8 quantization for the optional SageAttention-style q/k path.

k is quantized per token after centering by its 64-token block mean, so only
the small per-block residual goes through the int8 dot; the mean term is the
routing score the forward kernel already computes exactly in bf16 and is added
back there. The TMA/materialized reference quantizes q per token together with
the diagonal routing threshold; SM86/SM89/SM120 instead perform both operations in
the pointer forward from its resident Q tile. Loads use plain pointers with
explicit strides so the preprocessing also runs on pre-Hopper architectures.
"""

import torch
import triton
import triton.language as tl

BLOCK_SIZE = 64


@triton.jit
def _round_to_int8(x):
    """Round-to-nearest-int via int32 truncation, avoiding libdevice.rint
    (unavailable on some Triton builds)."""
    return tl.where(x >= 0, (x + 0.5).to(tl.int32), (x - 0.5).to(tl.int32)).to(tl.int8)


@triton.jit(do_not_specialize=['T', 's_b', 's_t', 's_h', 'NB', 'H'], do_not_specialize_on_alignment=['T', 's_b', 's_t', 's_h', 'NB', 'H'])
def _reduce_quant_k_kernel(
    k_ptr,
    kc_ptr,
    k8_ptr,
    k_scale,
    T,
    s_b, s_t, s_h,
    H,
    NB,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    WRITE_QUANT: tl.constexpr,
):
    """Fused block-mean reduce + residual int8 quantization: one K read.

    The block sum is accumulated in fp32 so the result is independent of the
    compiler-chosen reduction order (fp32 order noise rounds away at bf16
    storage). Both the bf16 and int8 paths obtain kc from this kernel, so
    routing decisions agree between them. With WRITE_QUANT=False the quant
    half is dead-code-eliminated and only kc is produced.
    """
    block, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    block_len = tl.minimum(BLOCK, T - block * BLOCK)
    rows = block * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, D)
    valid = rows < T
    tile = tl.load(
        k_ptr + batch * s_b + rows[:, None].to(tl.int64) * s_t + head * s_h + d[None, :],
        mask=valid[:, None],
        other=0.0,
    )
    summary = tl.sum(tile.to(tl.float32), axis=0) / block_len
    tl.store(
        kc_ptr + ((batch * NB + block) * H + head) * D + d,
        summary,
    )
    if WRITE_QUANT:
        # The residual is computed against the mean AFTER it has been rounded
        # to bf16 storage precision, replicating store-then-load.
        mean_stored = summary.to(tl.bfloat16).to(tl.float32)
        centered = tl.where(valid[:, None], tile.to(tl.float32) - mean_stored[None, :], 0.0)
        amax = tl.max(tl.abs(centered), axis=1)
        scale = tl.maximum(amax / 127.0, 1e-8)
        k8 = _round_to_int8(centered / scale[:, None])
        tl.store(
            k8_ptr + ((batch * T + rows[:, None]) * H + head) * D + d[None, :],
            k8,
            mask=valid[:, None],
        )
        tl.store(k_scale + (batch * T + rows) * H + head, scale, mask=valid)


@triton.jit(do_not_specialize=['T', 's_b', 's_t', 's_h', 'NB', 'H'], do_not_specialize_on_alignment=['T', 's_b', 's_t', 's_h', 'NB', 'H'])
def _quantize_k_kernel(
    k_ptr,
    kc_ptr,
    k8_ptr,
    k_scale,
    T,
    s_b, s_t, s_h,
    H,
    NB,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    block, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    rows = block * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, D)
    valid = rows < T
    tile = tl.load(
        k_ptr + batch * s_b + rows[:, None].to(tl.int64) * s_t + head * s_h + d[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    mean = tl.load(kc_ptr + ((batch * NB + block) * H + head) * D + d).to(tl.float32)
    centered = tl.where(valid[:, None], tile - mean[None, :], 0.0)
    amax = tl.max(tl.abs(centered), axis=1)
    scale = tl.maximum(amax / 127.0, 1e-8)
    k8 = _round_to_int8(centered / scale[:, None]).to(tl.int8)
    tl.store(
        k8_ptr + ((batch * T + rows[:, None]) * H + head) * D + d[None, :],
        k8,
        mask=valid[:, None],
    )
    tl.store(k_scale + (batch * T + rows) * H + head, scale, mask=valid)


@triton.jit(do_not_specialize=['T', 's_b', 's_t', 's_h', 'N', 'H'], do_not_specialize_on_alignment=['T', 's_b', 's_t', 's_h', 'N', 'H'])
def _q_quant_threshold_kernel(
    q_ptr,
    kc_mean_ptr,
    kc_var_ptr,
    q8_ptr,
    q_scale,
    thr_ptr,
    softmax_scale,
    T,
    s_b, s_t, s_h,
    H,
    N,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
    TAU,
):
    """Per-token q quantization fused with the diag routing threshold: one q load."""
    q_block, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    rows = q_block * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, D)
    valid = rows < T
    q_len = tl.minimum(BLOCK, T - q_block * BLOCK).to(tl.float32)
    tile = tl.load(
        q_ptr + batch * s_b + rows[:, None].to(tl.int64) * s_t + head * s_h + d[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)

    amax = tl.max(tl.abs(tile), axis=1)
    scale = tl.maximum(amax / 127.0, 1e-8)
    q8 = _round_to_int8(tile / scale[:, None]).to(tl.int8)
    tl.store(
        q8_ptr + ((batch * T + rows[:, None]) * H + head) * D + d[None, :],
        q8,
        mask=valid[:, None],
    )
    tl.store(q_scale + (batch * T + rows) * H + head, scale, mask=valid)

    centroid = tl.sum(tile, axis=0) / q_len
    mean_kc = tl.load(kc_mean_ptr + batch_head * D + d)
    var_kc = tl.load(kc_var_ptr + batch_head * D + d)
    log2_scale = softmax_scale * 1.4426950408889634
    mean = tl.sum(centroid * mean_kc, axis=0) * log2_scale
    variance = tl.sum(centroid * centroid * var_kc, axis=0) * (log2_scale * log2_scale)
    std = tl.sqrt(tl.maximum(variance, 0.0) + 1.0e-6)
    tl.store(thr_ptr + (batch * N + q_block) * H + head, mean + TAU * std)


def quantize_k(k, kc):
    """k [B, T, H, D] bf16; kc [B, NB, H, D] bf16 block means.

    Returns k_int8 [B, T, H, D] (per-block-mean residual), k_scale [B, T, H] fp32.
    """
    batch, tokens, heads, head_dim = k.shape
    blocks = triton.cdiv(tokens, BLOCK_SIZE)
    k8 = torch.empty(k.shape, device=k.device, dtype=torch.int8)
    k_scale = torch.empty((batch, tokens, heads), device=k.device, dtype=torch.float32)
    _quantize_k_kernel[(blocks, batch * heads)](
        k,
        kc,
        k8,
        k_scale,
        tokens,
        k.stride(0), k.stride(1), k.stride(2),
        heads,
        blocks,
        head_dim,
        BLOCK_SIZE,
        num_warps=4,
    )
    return k8, k_scale


def reduce_quantize_k(k, quant=True):
    """Fused block-mean reduce + residual int8 quantization; one K read.

    k [B, T, H, D] bf16. With quant=True returns (kc, k_int8, k_scale); with
    quant=False returns (kc, None, None) — the kc producer for the bf16 path,
    so routing decisions agree between the bf16 and int8 paths.
    """
    batch, tokens, heads, head_dim = k.shape
    blocks = triton.cdiv(tokens, BLOCK_SIZE)
    kc = torch.empty((batch, blocks, heads, head_dim), device=k.device, dtype=torch.bfloat16)
    k8 = torch.empty(k.shape, device=k.device, dtype=torch.int8) if quant else kc
    k_scale = (
        torch.empty((batch, tokens, heads), device=k.device, dtype=torch.float32)
        if quant else kc
    )
    _reduce_quant_k_kernel[(blocks, batch * heads)](
        k,
        kc,
        k8,
        k_scale,
        tokens,
        k.stride(0), k.stride(1), k.stride(2),
        heads,
        blocks,
        head_dim,
        BLOCK_SIZE,
        quant,
        num_warps=4,
    )
    return kc, k8, k_scale


def quantize_q_with_threshold(q, kc_mean, kc_var, *, scale, tau):
    """q [B, T, H, D] bf16; kc stats [B, H, D] fp32 on the smoothed summaries.

    Returns q_int8 [B, T, H, D], q_scale [B, T, H] fp32, threshold [B, NB, H] fp32.
    """
    batch, tokens, heads, head_dim = q.shape
    blocks = triton.cdiv(tokens, BLOCK_SIZE)
    q8 = torch.empty(q.shape, device=q.device, dtype=torch.int8)
    q_scale = torch.empty((batch, tokens, heads), device=q.device, dtype=torch.float32)
    threshold = torch.empty((batch, blocks, heads), device=q.device, dtype=torch.float32)
    _q_quant_threshold_kernel[(blocks, batch * heads)](
        q,
        kc_mean,
        kc_var,
        q8,
        q_scale,
        threshold,
        scale,
        tokens,
        q.stride(0), q.stride(1), q.stride(2),
        heads,
        blocks,
        head_dim,
        BLOCK_SIZE,
        tau,
        num_warps=4,
    )
    return q8, q_scale, threshold


@triton.jit(do_not_specialize=['T', 's_b', 's_t', 's_h', 'H'], do_not_specialize_on_alignment=['T', 's_b', 's_t', 's_h', 'H'])
def _quantize_v_kernel(
    v_ptr,
    scale_ptr,
    vi_ptr,
    T,
    s_b, s_t, s_h,
    H,
    D: tl.constexpr,
    ROWS: tl.constexpr,
):
    """Per-channel INT8 V quantization: one scale per (batch*head, channel)."""
    row_tile, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head = batch_head // H, batch_head % H
    rows = row_tile * ROWS + tl.arange(0, ROWS)
    d = tl.arange(0, D)
    valid = rows < T
    tile = tl.load(
        v_ptr + batch * s_b + rows[:, None].to(tl.int64) * s_t + head * s_h + d[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    s = tl.load(scale_ptr + batch_head * D + d)
    v8 = _round_to_int8(tile / s[None, :]).to(tl.int8)
    tl.store(
        vi_ptr + ((batch * T + rows[:, None]).to(tl.int64) * H + head) * D + d[None, :],
        v8,
        mask=valid[:, None],
    )


def quantize_v_per_channel(v, rows=16, absmax=None):
    """v [B, T, H, D] bf16. Returns v_int8 [B, T, H, D], v_scale [B*H, D] fp32.

    PV is out[m,d] = sum_k P[m,k] V[k,d], so a per-channel V scale and a
    per-row P scale both factor straight out of the int32 dot.

    ``absmax`` is V's per-(head, channel) |max| as [B, H, D]. Pass it in from a
    pass that already reads V; computing it here costs an extra full read.
    """
    batch, tokens, heads, head_dim = v.shape
    if absmax is None:
        absmax = v.abs().amax(dim=1)
    scale = absmax.float().div(127.0).clamp_min_(1e-8)
    scale = scale.reshape(batch * heads, head_dim).contiguous()
    vi = torch.empty(v.shape, device=v.device, dtype=torch.int8)
    _quantize_v_kernel[(triton.cdiv(tokens, rows), batch * heads)](
        v,
        scale,
        vi,
        tokens,
        v.stride(0), v.stride(1), v.stride(2),
        heads,
        head_dim,
        rows,
        num_warps=4,
    )
    return vi, scale


__all__ = ["quantize_k", "reduce_quantize_k", "quantize_q_with_threshold", "quantize_v_per_channel"]


from shared.kernels.triton_compilation_log import install_triton_compilation_logger
install_triton_compilation_logger()
