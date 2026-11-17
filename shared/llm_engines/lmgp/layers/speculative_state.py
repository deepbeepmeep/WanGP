"""Multi-token verification with snapshots for every committable prefix.

Adapted from FLA's convolution and gated delta rule kernels (MIT):
https://github.com/fla-org/flash-linear-attention
Copyright (c) 2023-2025, Songlin Yang, Yu Zhang.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import torch
import triton
import triton.language as tl
from fla.ops.utils.op import exp


@triton.jit(do_not_specialize=("B", "T"), do_not_specialize_on_alignment=("B", "T"))
def conv_verify_kernel(x, state, weight, bias, output, snapshots,
                       B, T, C, W: tl.constexpr,
                       HAS_BIAS: tl.constexpr, BC: tl.constexpr, BW: tl.constexpr):
    channel = tl.program_id(0) * BC + tl.arange(0, BC)
    batch = tl.program_id(1)
    window = tl.arange(0, BW)
    for token in range(T):
        source = token + 1 + window
        history = tl.load(state + (batch * C + channel[:, None]) * W + source[None, :], mask=(channel[:, None] < C) & (window[None, :] < W) & (source[None, :] < W), other=0).to(tl.float32)
        current = tl.load(x + (batch * T + source[None, :] - W) * C + channel[:, None], mask=(channel[:, None] < C) & (window[None, :] < W) & (source[None, :] >= W), other=0).to(tl.float32)
        values = history + current
        weights = tl.load(weight + channel[:, None] * W + window[None, :], mask=(channel[:, None] < C) & (window[None, :] < W), other=0)
        result = tl.sum(values * weights, 1)
        if HAS_BIAS:
            result += tl.load(bias + channel, mask=channel < C, other=0)
        result *= tl.sigmoid(result)
        tl.store(output + (batch * T + token) * C + channel, result, mask=channel < C)
        if token + 1 < T:
            tl.store(snapshots + ((token * B + batch) * C + channel[:, None]) * W + window[None, :], values, mask=(channel[:, None] < C) & (window[None, :] < W))
        else:
            tl.store(state + (batch * C + channel[:, None]) * W + window[None, :], values, mask=(channel[:, None] < C) & (window[None, :] < W))


def conv_verify(x, state, weight, bias, snapshots):
    """Store prefix snapshots and update the final convolution state in place."""
    b, t, c = x.shape
    w = weight.shape[-1]
    output = torch.empty_like(x)
    conv_verify_kernel[(triton.cdiv(c, 32), b)](x, state, weight, bias, output, snapshots, b, t, c, w, bias is not None, 32, triton.next_power_of_2(w), num_warps=4)
    return output, state


@triton.jit(do_not_specialize=("B", "T", "H", "HV"), do_not_specialize_on_alignment=("B", "T", "H", "HV"))
def recurrent_verify_kernel(q, k, v, g, beta, initial, output, snapshots,
                            B, T, H, HV,
                            K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr,
                            SAVE_PREFIX: tl.constexpr):
    vblock, batch_head = tl.program_id(0), tl.program_id(1)
    batch, head_v = batch_head // HV, batch_head % HV
    head_k = head_v // (HV // H)
    keys = tl.arange(0, BK)
    values = vblock * BV + tl.arange(0, BV)
    state_offset = batch_head * K * V + keys[:, None] * V + values[None, :]
    mask = (keys[:, None] < K) & (values[None, :] < V)
    state = tl.load(initial + state_offset, mask=mask, other=0).to(tl.float32)
    for token in range(T):
        offset_k = ((batch * T + token) * H + head_k) * K + keys
        offset_v = ((batch * T + token) * HV + head_v) * V + values
        query = tl.load(q + offset_k, mask=keys < K, other=0).to(tl.float32)
        key = tl.load(k + offset_k, mask=keys < K, other=0).to(tl.float32)
        value = tl.load(v + offset_v, mask=values < V, other=0).to(tl.float32)
        query = query / tl.sqrt(tl.sum(query * query) + 1e-6)
        key = key / tl.sqrt(tl.sum(key * key) + 1e-6)
        query = query * K ** -0.5
        decay = tl.load(g + (batch * T + token) * HV + head_v).to(tl.float32)
        strength = tl.load(beta + (batch * T + token) * HV + head_v).to(tl.float32)
        state *= exp(decay)
        value = strength * (value - tl.sum(state * key[:, None], 0))
        state += key[:, None] * value
        result = tl.sum(state * query[:, None], 0)
        tl.store(output + offset_v, result, mask=values < V)
        # Triton type-checks the dynamic loop branch even for T == 1, so the
        # optional pointer must be guarded by a compile-time condition.
        if SAVE_PREFIX:
            if token + 1 < T:
                tl.store(snapshots + token * B * HV * K * V + state_offset, state, mask=mask)
    tl.store(initial + state_offset, state, mask=mask)


def recurrent_verify(q, k, v, g, beta, initial, snapshots):
    """Carry FP32 state; write prefixes to snapshots and the final state in place."""
    q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
    b, t, h, dim_k = q.shape
    hv, dim_v = v.shape[-2:]
    output = torch.empty_like(v)
    recurrent_verify_kernel[(triton.cdiv(dim_v, 8), b * hv)](q, k, v, g, beta, initial, output, snapshots, b, t, h, hv, dim_k, dim_v, triton.next_power_of_2(dim_k), 8, snapshots is not None, num_warps=1, num_stages=3)
    return output, initial


# Register after definitions to preserve Triton's line-number-sensitive cache keys.
from shared.kernels.triton_compilation_log import install_triton_compilation_logger
install_triton_compilation_logger()
