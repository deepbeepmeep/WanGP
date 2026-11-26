"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch
import triton
import triton.language as tl

from shared.kernels.dispatch import triton_dtype_supported


def _bf16_bits(tensor):
    # GPUs without BF16 conversion instructions (Turing) read a BF16 tensor as its bits
    return tensor.dtype == torch.bfloat16 and not triton_dtype_supported(tensor.device, tensor.dtype)


@triton.jit
def quant_per_block_int8_kernel(Input, Output, Scale, L,
                                stride_iz, stride_ih, stride_in,
                                stride_oz, stride_oh, stride_on,
                                stride_sz, stride_sh,
                                sm_scale,
                                C: tl.constexpr, BLK: tl.constexpr, BF16_BITS: tl.constexpr):
    off_blk = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    offs_n = off_blk * BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, C)

    input_ptrs = Input + off_b * stride_iz + off_h * stride_ih + offs_n[:, None] * stride_in + offs_k[None, :]
    output_ptrs = Output + off_b * stride_oz + off_h * stride_oh + offs_n[:, None] * stride_on + offs_k[None, :]
    scale_ptrs = Scale + off_b * stride_sz + off_h * stride_sh + off_blk

    x = tl.load(input_ptrs, mask=offs_n[:, None] < L, other=0.0)
    if BF16_BITS:  # the bits of BF16 values, shifted into FP32 bits: the exact conversion
        x = (x.to(tl.int32) << 16).to(tl.float32, bitcast=True)
    x = x.to(tl.float32)
    x *= sm_scale
    peak = tl.max(tl.abs(x))
    scale = tl.where(peak > 0, peak / 127., 1.)
    x_int8 = x / scale
    x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8.to(tl.int8)
    tl.store(output_ptrs, x_int8, mask=offs_n[:, None] < L)
    tl.store(scale_ptrs, scale)

def quantize(x, block, sm_scale, tensor_layout="NHD"):
    rows = x if tensor_layout == "NHD" else x.transpose(1, 2)
    b, length, heads, dim = rows.shape
    out = torch.empty(rows.shape, dtype=torch.int8, device=x.device)
    scales = torch.empty((b, heads, triton.cdiv(length, block)), dtype=torch.float32, device=x.device)
    bits = _bf16_bits(rows)
    quant_per_block_int8_kernel[(scales.shape[-1], heads, b)](
        rows.view(torch.int16) if bits else rows, out, scales, length,
        rows.stride(0), rows.stride(2), rows.stride(1), out.stride(0), out.stride(2), out.stride(1),
        scales.stride(0), scales.stride(1), sm_scale=sm_scale, C=dim, BLK=block, BF16_BITS=bits)
    return (out if tensor_layout == "NHD" else out.transpose(1, 2)), scales


def per_block_int8(q, k, km=None, BLKQ=128, BLKK=64, sm_scale=None, tensor_layout="HND"):
    if tensor_layout not in ("NHD", "HND"):
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")
    scale = q.size(-1) ** -0.5 if sm_scale is None else sm_scale
    return (*quantize(q, BLKQ, scale * 1.44269504, tensor_layout),
            *quantize(k if km is None else k - km, BLKK, 1.0, tensor_layout))
