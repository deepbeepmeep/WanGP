# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Pack an owned Sage query in its original rows. The block scale and rounding match Sage's per-block INT8 quantizer."""
import torch
import triton
import triton.language as tl

from .bf16_bits import bf16_bits_to_fp32


def recyclable_query(x):
    """A vector-aligned NHD or head-major view with disjoint rows and 32-bit byte offsets."""
    b, n, h, d = x.shape
    sb, st, sh, sc = x.stride()
    span = (n - 1) * st + (h - 1) * sh + d
    return (x.dtype in (torch.bfloat16, torch.float16) and sc == 1 and x.data_ptr() % 16 == 0
            and st % 8 == 0 and sh % 8 == 0 and (b == 1 or (sb % 8 == 0 and sb >= span))
            and 2 * max(sb, st, sh, (b - 1) * sb + span) < 2**31
            and ((sh >= d and st >= (h - 1) * sh + d) or (st >= d and sh >= (n - 1) * st + d)))


@triton.jit
def _pack(X, Q, S, N, XB, XT, XH, SB, ST, SH, softmax_scale, D: tl.constexpr, TILE: tl.constexpr, BITS: tl.constexpr):
    tile, h, b = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    row = tile * TILE + tl.arange(0, TILE)
    col = tl.arange(0, D)
    offset = b * XB + row * XT + h * XH
    x = tl.load(X + offset[:, None] + col[None, :], row[:, None] < N, 0)
    value = (bf16_bits_to_fp32(x) if BITS else x.to(tl.float32)) * softmax_scale
    scale = tl.max(tl.abs(value)) / 127.
    code = value / scale
    code += tl.where(code >= 0, 0.5, -0.5)
    # Other lanes can still be reading the original 16-bit channels of the same row.
    tl.debug_barrier()
    tl.store(Q + 2 * offset[:, None] + col[None, :], code.to(tl.int8), row[:, None] < N)
    tl.store(S + b * SB + tile * ST + h * SH, scale)


def pack_query(q, tile, softmax_scale, *, varlen=False):
    batch, tokens, heads, dim = q.shape
    blocks = triton.cdiv(tokens, tile)
    packed = q.view(torch.int8)[..., :dim]
    shape = (batch, blocks, heads) if varlen else (batch, heads, blocks)
    scale = torch.empty(shape, dtype=torch.float32, device=q.device)
    scales_bth = scale if varlen else scale.transpose(1, 2)
    bits = q.dtype == torch.bfloat16
    _pack[(blocks, heads, batch)](q.view(torch.int16) if bits else q, packed, scale, tokens, *q.stride()[:3],
                                  *scales_bth.stride(), softmax_scale, dim, tile, bits,
                                  num_warps=max(4, tile * dim // 2048))
    torch.autograd.graph.increment_version(q)
    return packed, scale
