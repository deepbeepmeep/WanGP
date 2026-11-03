# Copyright (c) 2026 DeepBeepMeep - WanGP Community License 2.0 (see LICENSE.txt)
"""Selected resident experts of routed banks (SwiGLU, or squared ReLU without a gate), addressed individually through a pointer table
(one row per expert: input projection weights, down weights, input projection scales, down scales), for the FP8 block, INT4 group and
INT8 rows formats.

This vLLM backend uses float32 activations/accumulation, matching the CPU backend. It never transfers weights. Nonresident ids
contribute zero to the GPU result."""
from functools import cache

import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_gate_up(X, IDs, PTR, Mid, H: tl.constexpr, I: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr, ROWS: tl.constexpr, GATED: tl.constexpr):
    # One (token, slot) pair and ROWS rows of its gate and up projections (up only without a gate), a scale block of columns at a time.
    pair = tl.program_id(0)
    rows = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    expert = tl.load(IDs + pair).to(tl.int64)
    address = tl.load(PTR + expert * 4)
    gate = tl.zeros((ROWS,), tl.float32)
    up = tl.zeros((ROWS,), tl.float32)
    if address != 0:
        weight = address.to(tl.pointer_type(tl.float8e4nv))
        scales = tl.load(PTR + expert * 4 + 2).to(tl.pointer_type(tl.float32))
        cols = tl.arange(0, BLOCK)
        for block in range(H // BLOCK):
            x = tl.load(X + (pair // K) * H + block * BLOCK + cols).to(tl.float32)
            g = tl.load(weight + rows[:, None] * H + block * BLOCK + cols[None, :]).to(tl.float32)
            gate += tl.sum(g * x[None, :], 1) * tl.load(scales + (rows // BLOCK) * (H // BLOCK) + block)
            if GATED:
                u = tl.load(weight + (rows[:, None] + I) * H + block * BLOCK + cols[None, :]).to(tl.float32)
                up += tl.sum(u * x[None, :], 1) * tl.load(scales + ((rows + I) // BLOCK) * (H // BLOCK) + block)
    if GATED:
        tl.store(Mid + pair * I + rows, gate * tl.sigmoid(gate) * up)
    else:
        gate = tl.maximum(gate, 0.0)
        tl.store(Mid + pair * I + rows, gate * gate)


@triton.jit
def _fp8_down(Mid, IDs, Probs, PTR, Out, H: tl.constexpr, I: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr, ROWS: tl.constexpr):
    # ROWS output rows of one token, summed over its resident slots.
    token = tl.program_id(0)
    rows = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK)
    total = tl.zeros((ROWS,), tl.float32)
    for slot in tl.static_range(K):
        pair = token * K + slot
        expert = tl.load(IDs + pair).to(tl.int64)
        address = tl.load(PTR + expert * 4 + 1)
        if address != 0:
            weight = address.to(tl.pointer_type(tl.float8e4nv))
            scales = tl.load(PTR + expert * 4 + 3).to(tl.pointer_type(tl.float32))
            value = tl.zeros((ROWS,), tl.float32)
            for block in range(I // BLOCK):
                x = tl.load(Mid + pair * I + block * BLOCK + cols)
                w = tl.load(weight + rows[:, None] * I + block * BLOCK + cols[None, :]).to(tl.float32)
                value += tl.sum(w * x[None, :], 1) * tl.load(scales + (rows // BLOCK) * (I // BLOCK) + block)
            total += value * tl.load(Probs + pair).to(tl.float32)
    tl.store(Out + token * H + rows, total)


@triton.jit
def _int4_gate_up(X, IDs, PTR, Mid, H: tl.constexpr, I: tl.constexpr, K: tl.constexpr, GROUP: tl.constexpr, ROWS: tl.constexpr, GATED: tl.constexpr):
    # As _fp8_gate_up, a scale group at a time: each byte holds an even column (low nibble) and the next one, offset by 8.
    pair = tl.program_id(0)
    rows = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    expert = tl.load(IDs + pair).to(tl.int64)
    address = tl.load(PTR + expert * 4)
    gate = tl.zeros((ROWS,), tl.float32)
    up = tl.zeros((ROWS,), tl.float32)
    if address != 0:
        weight = address.to(tl.pointer_type(tl.uint8))
        scales = tl.load(PTR + expert * 4 + 2).to(tl.pointer_type(tl.float16))
        cols = tl.arange(0, GROUP // 2)
        for group in range(H // GROUP):
            x = X + (pair // K) * H + group * GROUP + 2 * cols
            even, odd = tl.load(x).to(tl.float32), tl.load(x + 1).to(tl.float32)
            g = tl.load(weight + rows[:, None] * (H // 2) + group * (GROUP // 2) + cols[None, :])
            g_sum = tl.sum(((g & 15).to(tl.float32) - 8.0) * even[None, :] + ((g >> 4).to(tl.float32) - 8.0) * odd[None, :], 1)
            gate += g_sum * tl.load(scales + rows * (H // GROUP) + group).to(tl.float32)
            if GATED:
                u = tl.load(weight + (rows[:, None] + I) * (H // 2) + group * (GROUP // 2) + cols[None, :])
                u_sum = tl.sum(((u & 15).to(tl.float32) - 8.0) * even[None, :] + ((u >> 4).to(tl.float32) - 8.0) * odd[None, :], 1)
                up += u_sum * tl.load(scales + (rows + I) * (H // GROUP) + group).to(tl.float32)
    if GATED:
        tl.store(Mid + pair * I + rows, gate * tl.sigmoid(gate) * up)
    else:
        gate = tl.maximum(gate, 0.0)
        tl.store(Mid + pair * I + rows, gate * gate)


@triton.jit
def _int4_down(Mid, IDs, Probs, PTR, Out, H: tl.constexpr, I: tl.constexpr, K: tl.constexpr, GROUP: tl.constexpr, ROWS: tl.constexpr):
    token = tl.program_id(0)
    rows = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, GROUP // 2)
    total = tl.zeros((ROWS,), tl.float32)
    for slot in tl.static_range(K):
        pair = token * K + slot
        expert = tl.load(IDs + pair).to(tl.int64)
        address = tl.load(PTR + expert * 4 + 1)
        if address != 0:
            weight = address.to(tl.pointer_type(tl.uint8))
            scales = tl.load(PTR + expert * 4 + 3).to(tl.pointer_type(tl.float16))
            value = tl.zeros((ROWS,), tl.float32)
            for group in range(I // GROUP):
                x = Mid + pair * I + group * GROUP + 2 * cols
                even, odd = tl.load(x).to(tl.float32), tl.load(x + 1).to(tl.float32)
                w = tl.load(weight + rows[:, None] * (I // 2) + group * (GROUP // 2) + cols[None, :])
                w_sum = tl.sum(((w & 15).to(tl.float32) - 8.0) * even[None, :] + ((w >> 4).to(tl.float32) - 8.0) * odd[None, :], 1)
                value += w_sum * tl.load(scales + rows * (I // GROUP) + group).to(tl.float32)
            total += value * tl.load(Probs + pair).to(tl.float32)
    tl.store(Out + token * H + rows, total)


@triton.jit
def _int8_gate_up(X, IDs, PTR, Mid, H: tl.constexpr, I: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr, ROWS: tl.constexpr, GATED: tl.constexpr):
    # As _fp8_gate_up for INT8 rows: BLOCK columns at a time, the row scale applied to the whole sum.
    pair = tl.program_id(0)
    rows = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    expert = tl.load(IDs + pair).to(tl.int64)
    address = tl.load(PTR + expert * 4)
    gate = tl.zeros((ROWS,), tl.float32)
    up = tl.zeros((ROWS,), tl.float32)
    if address != 0:
        weight = address.to(tl.pointer_type(tl.int8))
        scales = tl.load(PTR + expert * 4 + 2).to(tl.pointer_type(tl.float32))
        cols = tl.arange(0, BLOCK)
        for block in range(H // BLOCK):
            x = tl.load(X + (pair // K) * H + block * BLOCK + cols).to(tl.float32)
            g = tl.load(weight + rows[:, None] * H + block * BLOCK + cols[None, :]).to(tl.float32)
            gate += tl.sum(g * x[None, :], 1)
            if GATED:
                u = tl.load(weight + (rows[:, None] + I) * H + block * BLOCK + cols[None, :]).to(tl.float32)
                up += tl.sum(u * x[None, :], 1)
        gate = gate * tl.load(scales + rows)
        if GATED:
            up = up * tl.load(scales + rows + I)
    if GATED:
        tl.store(Mid + pair * I + rows, gate * tl.sigmoid(gate) * up)
    else:
        gate = tl.maximum(gate, 0.0)
        tl.store(Mid + pair * I + rows, gate * gate)


@triton.jit
def _int8_down(Mid, IDs, Probs, PTR, Out, H: tl.constexpr, I: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr, ROWS: tl.constexpr):
    token = tl.program_id(0)
    rows = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    cols = tl.arange(0, BLOCK)
    total = tl.zeros((ROWS,), tl.float32)
    for slot in tl.static_range(K):
        pair = token * K + slot
        expert = tl.load(IDs + pair).to(tl.int64)
        address = tl.load(PTR + expert * 4 + 1)
        if address != 0:
            weight = address.to(tl.pointer_type(tl.int8))
            scales = tl.load(PTR + expert * 4 + 3).to(tl.pointer_type(tl.float32))
            value = tl.zeros((ROWS,), tl.float32)
            for block in range(I // BLOCK):
                x = tl.load(Mid + pair * I + block * BLOCK + cols)
                w = tl.load(weight + rows[:, None] * I + block * BLOCK + cols[None, :]).to(tl.float32)
                value += tl.sum(w * x[None, :], 1)
            total += value * tl.load(scales + rows) * tl.load(Probs + pair).to(tl.float32)
    tl.store(Out + token * H + rows, total)


_KERNELS = {"fp8_block": (_fp8_gate_up, _fp8_down), "int4_group": (_int4_gate_up, _int4_down), "int8_rows": (_int8_gate_up, _int8_down)}


@cache
def _needs_fp32_inputs(device):
    return torch.version.hip is None and torch.cuda.get_device_capability(device) < (8, 0)


def forward(kind, x, ids, probabilities, pointers, intermediate, blocks, out=None, mid=None, rotation=0, gated=True, rows=8):
    """float32 [tokens, hidden] sum of the resident selected experts (zero for the others); out and the float32 [tokens * topk,
    intermediate] scratch mid may be preallocated. blocks: the scale blocks of the input and down projections; rotation: the ConvRot
    group of the activations of both projections, or 0; gated: SwiGLU (else squared ReLU without a gate)."""
    tokens, hidden = x.shape
    topk = ids.shape[-1]
    # These kernels already compute in FP32. Convert exactly before launch on GPUs without native BF16 conversion instructions.
    if _needs_fp32_inputs(x.device):
        x, probabilities = x.float(), probabilities.float()
    gate_up, down = _KERNELS[kind]
    if kind == "int8_rows":  # one scale per row: any column tile dividing both widths
        tile = 64 if hidden % 64 == 0 and intermediate % 64 == 0 else 32
        blocks = (tile, tile)
    mid = torch.empty(tokens * topk, intermediate, dtype=torch.float32, device=x.device) if mid is None else mid
    out = torch.empty(tokens, hidden, dtype=torch.float32, device=x.device) if out is None else out
    if rotation:
        from shared.qtypes.int8_convrot import _rotate_activation
        x = _rotate_activation(x.float(), rotation)
    gate_up[(tokens * topk, intermediate // rows)](x, ids, pointers, mid, hidden, intermediate, topk, blocks[0], rows, gated)
    if rotation:
        mid = _rotate_activation(mid, rotation)
    down[(tokens, hidden // rows)](mid, ids, probabilities, pointers, out, hidden, intermediate, topk, blocks[1], rows)
    return out


from shared.kernels.triton_compilation_log import install_triton_compilation_logger
install_triton_compilation_logger()
