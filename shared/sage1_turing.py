# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
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

The SageAttention (1.0.6) attention kernel of attn_qk_int8_block_varlen.py for GPUs whose Triton has no BF16 conversion instructions
(compute capability 7.5): the output is written as the bits of a BF16 tensor, rounded once from the FP32 accumulator as the kernel's
own BF16 store does on sm_80+, and with V_BITS v is read as the bits of a BF16 tensor and converted to FP16 for the P.V product. INT8
q/k inputs, block scales, tiling and arithmetic are those of the original kernel. shared.sage1_core launches it on such GPUs instead
of writing an FP16 output and copying it back to BF16, and for short sequences instead of copying v to FP16. MASK_MODE adds an
attention mask in SageAttention's (batch, heads, q, k) layout (strides, zero for broadcast dimensions): 1 a boolean mask (uint8
view) whose zero entries get no weight, 2 an additive FP32 mask as SDPA adds it to the scaled scores. RETURN_LSE also writes the base-2
log-sum-exp of each query's scores, to merge attentions over separate key sets (a cached prefix, shared.attention_kit).
"""
import triton
import triton.language as tl

from shared.kernels.bf16_bits import bf16_bits_to_fp32, fp32_to_bf16_bits


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q, q_scale, kv_len, qo_len,
                    K_ptrs, K_scale_ptr, V_ptrs, M_ptrs, stride_kn, stride_vn, stride_mk,
                    start_m,
                    H,
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr, V_BITS: tl.constexpr, MASK_MODE: tl.constexpr,
                    ):
    lo, hi = 0, kv_len
    for start_n in range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_mask = offs_n[None, :] < (kv_len - start_n)
        k = tl.load(K_ptrs, mask=k_mask)
        k_scale = tl.load(K_scale_ptr)
        qk = tl.dot(q, k).to(tl.float32) * q_scale * k_scale
        if MASK_MODE == 1:
            # boolean mask (batch, heads, q, k) as uint8, broadcast through zero strides: a masked key gets no weight (-inf before the
            # softmax, as SDPA); keys beyond kv_len keep the kernel's behavior (unmasked)
            keep = tl.load(M_ptrs, mask=(offs_m[:, None] < qo_len) & k_mask, other=1)
            qk = tl.where(keep != 0, qk, float("-inf"))
        elif MASK_MODE == 2:
            # additive FP32 mask in the natural-log domain of SDPA: the scores are in base 2 here (log2 e folded into q_scale)
            qk = qk + tl.load(M_ptrs, mask=(offs_m[:, None] < qo_len) & k_mask, other=0.0) * 1.4426950408889634
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk = qk - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)

        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij

        acc = acc * alpha[:, None]

        if V_BITS:
            # BF16 v read as bits (int16 pointer), exactly representable in FP16 unless below its normal range. Converted again by
            # every query block: worth it for short sequences only (sage1_core decides), an FP16 copy converts each value once.
            v = bf16_bits_to_fp32(tl.load(V_ptrs, mask=offs_n[:, None] < (kv_len - start_n))).to(tl.float16)
        else:
            v = tl.load(V_ptrs, mask=offs_n[:, None] < (kv_len - start_n))
        p = p.to(tl.float16)

        acc += tl.dot(p, v, out_dtype=tl.float16)
        m_i = m_ij
        K_ptrs += BLOCK_N * stride_kn
        K_scale_ptr += H
        V_ptrs += BLOCK_N * stride_vn
        M_ptrs += BLOCK_N * stride_mk
    return acc, l_i, m_i


@triton.jit(do_not_specialize=["H", "num_kv_groups"])  # head counts at runtime: one kernel for every Head Split group size
def _attn_fwd(Q, K, V,
              cu_seqlens_q, cu_seqlens_k,
              Q_scale, K_scale, cu_seqlens_q_scale, cu_seqlens_k_scale,
              Out, Mask, Lse,
              stride_qh, stride_qn,
              stride_kh, stride_kn,
              stride_vh, stride_vn,
              stride_oh, stride_on,
              stride_mz, stride_mh, stride_mq, stride_mk,
              H, num_kv_groups,
              HEAD_DIM: tl.constexpr,
              BLOCK_M: tl.constexpr,
              BLOCK_N: tl.constexpr,
              STAGE: tl.constexpr,
              V_BITS: tl.constexpr,
              MASK_MODE: tl.constexpr,
              RETURN_LSE: tl.constexpr = False
              ):
    start_m = tl.program_id(0)

    off_z = tl.program_id(2).to(tl.int64)
    off_h = tl.program_id(1).to(tl.int64)

    cu_seqlens_q_start = tl.load(cu_seqlens_q + off_z)
    cu_seqlens_q_end = tl.load(cu_seqlens_q + off_z + 1)

    qo_len = cu_seqlens_q_end - cu_seqlens_q_start

    if (start_m * BLOCK_M) >= qo_len:
        return

    cu_seq_lens_q_scale_start = tl.load(cu_seqlens_q_scale + off_z)
    cu_seq_lens_k_scale_start = tl.load(cu_seqlens_k_scale + off_z)

    q_scale_offset = cu_seq_lens_q_scale_start * H + off_h + start_m * H
    k_scale_offset = cu_seq_lens_k_scale_start * (H // num_kv_groups) + off_h // num_kv_groups

    cu_seqlens_k_start = tl.load(cu_seqlens_k + off_z)
    cu_seqlens_k_end = tl.load(cu_seqlens_k + off_z + 1)

    kv_len = cu_seqlens_k_end - cu_seqlens_k_start

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, HEAD_DIM)
    Q_ptrs = Q + (cu_seqlens_q_start * stride_qn + off_h * stride_qh) + offs_m[:, None] * stride_qn + offs_k[None, :]
    Q_scale_ptr = Q_scale + q_scale_offset
    K_ptrs = K + (cu_seqlens_k_start * stride_kn + (off_h // num_kv_groups) * stride_kh) + offs_n[None, :] * stride_kn + offs_k[:, None]
    K_scale_ptr = K_scale + k_scale_offset
    V_ptrs = V + (cu_seqlens_k_start * stride_vn + (off_h // num_kv_groups) * stride_vh) + offs_n[:, None] * stride_vn + offs_k[None, :]
    O_block_ptr = Out + (cu_seqlens_q_start * stride_on + off_h * stride_oh) + offs_m[:, None] * stride_on + offs_k[None, :]
    # the mask in SageAttention's (batch, heads, q, k) layout, broadcast dimensions with stride 0; q and k local to the batch item
    M_ptrs = Mask + off_z * stride_mz + off_h * stride_mh + offs_m[:, None] * stride_mq + offs_n[None, :] * stride_mk

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q = tl.load(Q_ptrs, mask=offs_m[:, None] < qo_len)
    q_scale = tl.load(Q_scale_ptr)
    acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, q_scale, kv_len, qo_len, K_ptrs, K_scale_ptr, V_ptrs, M_ptrs, stride_kn, stride_vn, stride_mk,
                                    start_m,
                                    H // num_kv_groups,
                                    BLOCK_M, HEAD_DIM, BLOCK_N,
                                    4 - STAGE, offs_m, offs_n, V_BITS, MASK_MODE
                                    )
    acc = acc / l_i[:, None]
    # BF16 output written as bits (int16 pointer): one rounding of the FP32 accumulator (round to nearest even)
    tl.store(O_block_ptr, fp32_to_bf16_bits(acc), mask=(offs_m[:, None] < qo_len))
    if RETURN_LSE:  # base-2 log-sum-exp of the scores (log2 e is folded into q_scale), Lse (batch, heads, tokens) FP32
        tl.store(Lse + (off_z * H + off_h) * qo_len + offs_m, m_i + tl.math.log2(l_i), mask=offs_m < qo_len)


@triton.jit(do_not_specialize=["H"])
def quant_per_block_int8_bits_kernel(Input, Output, Scale, Mean,
                                     cu_seqlens_input, cu_seqlens_scale,
                                     stride_ih, stride_in,
                                     stride_oh, stride_on,
                                     sm_scale,
                                     H,
                                     C: tl.constexpr, BLK: tl.constexpr, HAS_MEAN: tl.constexpr):
    """quant_per_block_int8_kernel of quant_per_block_varlen.py reading a BF16 input as bits (int16 pointer): no FP16 copy on GPUs without
    BF16 conversions. With HAS_MEAN, Mean (heads, C) in FP32 is subtracted first (SageAttention's key smoothing), so k itself is neither
    copied nor modified."""
    off_blk = tl.program_id(0)
    off_h = tl.program_id(1)
    off_b = tl.program_id(2)

    cu_seqlens_input_start = tl.load(cu_seqlens_input + off_b)
    cu_seqlens_input_end = tl.load(cu_seqlens_input + off_b + 1)

    L = cu_seqlens_input_end - cu_seqlens_input_start

    if (off_blk * BLK) >= L:
        return

    cu_seqlens_scale_start = tl.load(cu_seqlens_scale + off_b)

    offs_n = off_blk * BLK + tl.arange(0, BLK)
    offs_k = tl.arange(0, C)

    input_ptrs = Input + cu_seqlens_input_start * stride_in + off_h * stride_ih + offs_n[:, None] * stride_in + offs_k[None, :]
    output_ptrs = Output + cu_seqlens_input_start * stride_on + off_h * stride_oh + offs_n[:, None] * stride_on + offs_k[None, :]
    scale_ptrs = Scale + cu_seqlens_scale_start * H + off_h + off_blk * H

    x = bf16_bits_to_fp32(tl.load(input_ptrs, mask=offs_n[:, None] < L, other=0))
    if HAS_MEAN:
        x = x - tl.load(Mean + off_h * C + offs_k)[None, :]
    x *= sm_scale
    scale = tl.max(tl.abs(x)) / 127.
    x_int8 = x / scale
    x_int8 += 0.5 * tl.where(x_int8 >= 0, 1, -1)
    x_int8 = x_int8.to(tl.int8)
    tl.store(output_ptrs, x_int8, mask=offs_n[:, None] < L)
    tl.store(scale_ptrs, scale)
