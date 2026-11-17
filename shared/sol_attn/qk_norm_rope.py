# SPDX-License-Identifier: Apache-2.0
# Modified for WanGP.
"""In-place Q/K RMSNorm + RoPE for strided fused-projection views."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _round_bf16(value):
    return tl.inline_asm_elementwise("cvt.rn.bf16.f32 $0, $1;", "=h,f", [value], dtype=tl.bfloat16, is_pure=True, pack=1)


@triton.jit(do_not_specialize=['SXB', 'SXT', 'SXH', 'SRB', 'SRT', 'SRP', 'SRC', 'T', 'H'], do_not_specialize_on_alignment=['SXB', 'SXT', 'SXH', 'SRB', 'SRT', 'SRP', 'SRC', 'T', 'H'])
def _rms_norm_rope_kernel(
    x,
    weight,
    rope,
    eps,
    SXB,
    SXT,
    SXH,
    SRB,
    SRT,
    SRP,
    SRC,
    T,
    H,
    D: tl.constexpr,
    PAIRS: tl.constexpr,
):
    row = tl.program_id(0)
    batch, token_head = row // (T * H), row % (T * H)
    token, head = token_head // H, token_head % H
    dims = tl.arange(0, D)
    offsets = batch * SXB + token * SXT + head * SXH + dims
    values = tl.load(x + offsets).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(values * values, axis=0) / D + eps)

    pair_offsets = tl.arange(0, D // 2)
    valid_pair = pair_offsets < PAIRS
    first_offsets = batch * SXB + token * SXT + head * SXH + pair_offsets
    second_offsets = first_offsets + PAIRS
    first = tl.load(x + first_offsets, mask=valid_pair, other=0.0).to(tl.float32)
    second = tl.load(x + second_offsets, mask=valid_pair, other=0.0).to(tl.float32)
    first = _round_bf16(first * inv_rms * tl.load(weight + pair_offsets, mask=valid_pair, other=0.0).to(tl.float32))
    second = _round_bf16(second * inv_rms * tl.load(weight + PAIRS + pair_offsets, mask=valid_pair, other=0.0).to(tl.float32))
    rope_offsets = batch * SRB + token * SRT + pair_offsets * SRP
    cosine = tl.load(rope + rope_offsets, mask=valid_pair, other=1.0).to(tl.float32)
    sine = tl.load(rope + rope_offsets + SRC, mask=valid_pair, other=0.0).to(tl.float32)
    first_cosine = _round_bf16(first.to(tl.float32) * cosine).to(tl.float32)
    first_sine = _round_bf16(first.to(tl.float32) * sine).to(tl.float32)
    second_cosine = _round_bf16(second.to(tl.float32) * cosine).to(tl.float32)
    second_sine = _round_bf16(second.to(tl.float32) * sine).to(tl.float32)
    tl.store(x + first_offsets, first_cosine - second_sine, mask=valid_pair)
    tl.store(x + second_offsets, second_cosine + first_sine, mask=valid_pair)

    remaining = dims >= 2 * PAIRS
    normalized = values * inv_rms * tl.load(weight + dims).to(tl.float32)
    tl.store(x + offsets, normalized, mask=remaining)


def rms_norm_rope_(x: torch.Tensor, weight: torch.Tensor, rope: torch.Tensor, eps: float) -> None:
    """Normalize and rotate one of Q or K in place."""
    batch, tokens, heads, head_dim = x.shape
    pairs = rope.shape[-2]
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16 or rope.dtype != torch.bfloat16:
        raise TypeError("fused RMSNorm + RoPE requires bfloat16 tensors")
    if x.device.type != "cuda" or weight.device != x.device or rope.device != x.device:
        raise ValueError("fused RMSNorm + RoPE tensors must share a CUDA device")
    if x.stride(-1) != 1 or weight.ndim != 1 or weight.numel() != head_dim:
        raise ValueError("fused RMSNorm + RoPE requires a contiguous head dimension and one weight per channel")
    if rope.ndim != 5 or rope.shape[0] not in (1, batch) or rope.shape[1] != tokens or rope.shape[2] != 1 or pairs * 2 > head_dim:
        raise ValueError("rope must have shape [1|B, T, 1, P, 2] with 2P <= head dimension")
    rope_batch_stride = 0 if rope.shape[0] == 1 else rope.stride(0)
    _rms_norm_rope_kernel[(batch * tokens * heads,)](
        x, weight, rope, float(eps), x.stride(0), x.stride(1), x.stride(2),
        rope_batch_stride, rope.stride(1), rope.stride(3), rope.stride(4), T=tokens, H=heads, D=head_dim,
        PAIRS=pairs, num_warps=4, num_stages=1,
    )


def qk_rms_norm_rope_(q: torch.Tensor, k: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor,
                       rope: torch.Tensor, eps: float) -> None:
    """Normalize and rotate Q/K in place without materializing their strided views."""

    rms_norm_rope_(q, q_weight, rope, eps)
    rms_norm_rope_(k, k_weight, rope, eps)


__all__ = ["qk_rms_norm_rope_", "rms_norm_rope_"]


from shared.kernels.triton_compilation_log import install_triton_compilation_logger
install_triton_compilation_logger()
