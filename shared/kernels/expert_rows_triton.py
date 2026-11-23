# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Routed experts whose rows carry their own block scales, read in their checkpoint storage through a pointer table (one row per expert:
input projection weights, down weights, input projection scales, down scales, then for NVFP4 the float32 factors of both projections;
zero for an expert without a GPU copy, which contributes nothing):
- NVFP4 rows: two E2M1 codes per byte (the even column in the low nibble), an E4M3 scale per 16 values of a row, a factor per expert;
- MXFP8 rows: E4M3 values, a scale 2^(byte - 127) per 32 values of a row.
The rows of the input projection alternate gate and up. Activation: SwiGLU, with soft caps (cap * tanh(v / cap) on the gate and on the
up outputs) when caps are given.

Decoding passes (forward): one program per (token, slot) pair and row, 32 lanes keeping their float32 partial sums of the exact decoded
weights over the row (the addresses of the table are 4-byte aligned, as MMGP's slots and regions are). Passes of many tokens (GroupedPass, from compute capability 8.0): the pairs routed to an expert form groups, and a value times its block scale
is exact in BF16 (at most 6 significant bits), so that tiles of weights and of BF16 activations multiply on the tensor cores with
float32 accumulation; the factor of an expert applies once to the sums. The output projection reads the activation in BF16 (as the
released model computes it) where approximate kernels are allowed, else as the three BF16 values whose sum is its float32 value. No weight is ever
dequantized to memory, and no load passes the rows of a tensor."""
import os
from functools import cache

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from shared.kernels import kernel_policy
from shared.kernels.bf16_bits import portable_supported, reads_bits
from shared.kernels.expert_gguf_triton import _four_activations
from shared.kernels.fp8_w8a16_triton import _decode_e4m3  # the block scales of the NVFP4 tiles
from shared.kernels.nvfp4_w4a16_triton import _decode_e2m1

NVFP4, MXFP8 = 0, 1
_DISABLED = os.environ.get("WGP_INKLING2_DISABLE", "").split(",")
SHARED = "shared_rows" not in _DISABLED  # with window_rows disabled: the leader kernels for small passes (off too: a launch of the row kernels per token, a comparison)
WINDOW_ROWS = "window_rows" not in _DISABLED  # small passes through the row kernels in one launch (off: the leader kernels _gate_up_shared / _down_shared)
UNROLLED = tl.constexpr("unrolled_rows" not in _DISABLED)  # the step loops of the decoding kernels unrolled (off: loops; the same bits)
_NVIDIA = tl.constexpr(torch.version.hip is None)
TENSORS = {NVFP4: 6, MXFP8: 4}  # columns of the pointer table a kind reads
_UNIT = tl.constexpr(64)  # columns per step of the grouped kernels (UNIT): whole scale blocks of both kinds
GROUP_TILES = {True: (128, 32, 8), False: (64, 16, 4)}  # rows and pairs per program, warps of the grouped kernels: BF16 activation for the output projection, float32 (RTX 5090, 512 tokens)
HIP_GROUP_TILES = {"gfx1200": (32, 32, 8), "gfx1201": (32, 32, 8), "gfx942": (64, 32, 8),  # by AMD architecture: (rows, pairs, warps) of the grouped kernels in both modes, where
                   **dict.fromkeys(("gfx1100", "gfx1101", "gfx1102", "gfx1150", "gfx1151"), (32, 32, 8))}  # GROUP_TILES spill (the least scratch: tests/inkling2_port sweeps)
GROUPED_PAIRS = 256  # eager calls of this many (token, slot) pairs run by groups
DOWN_PAIRS = "down_pairs" not in _DISABLED  # windows of DOWN_PAIR_TOKENS tokens or more: _down_rows with two output rows a program (the bits of _down_row)
DOWN_PAIR_TOKENS = 8
SHARED_ROWS = 16  # tokens of a decoding pass (a speculative verification) whose rows read an expert they share once
SHARED_PROGRAM = 1  # rows of a projection per program there (8 measured slower on an RTX 5090: 265 + 148 against 157 + 123 microseconds per layer at 8 tokens)


@triton.jit
def _e4m3(byte):
    """E4M3 bytes (as int32) over 256, float32: their bits are those of an FP16 value, subnormals included. The NaN encodings (0x7F, 0xFF) are not decoded: expert checkpoints hold none."""
    return (((byte & 127) << 7) | ((byte & 128) << 8)).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)


@triton.jit
def _values(weight, rows, unit, COLS: tl.constexpr, ROWS: tl.constexpr, KIND: tl.constexpr):
    """UNIT columns of ROWS rows of a packed matrix of COLS columns, before their block scales: float32 [ROWS, UNIT]."""
    if KIND == 0:
        packed = tl.load(weight + rows[:, None] * (COLS // 2) + unit * (_UNIT // 2) + tl.arange(0, _UNIT // 2)[None, :])
        return _decode_e2m1(tl.interleave(packed & 15, packed >> 4))
    else:  # over 256: _scales() gives it back
        return _e4m3(tl.load(weight + rows[:, None] * COLS + unit * _UNIT + tl.arange(0, _UNIT)[None, :]).to(tl.int32))


@triton.jit
def _scales(scale, rows, unit, COLS: tl.constexpr, KIND: tl.constexpr):
    """Their block scales: float32 [ROWS, UNIT // block]."""
    if KIND == 0:
        return _decode_e4m3(tl.load(scale + rows[:, None] * (COLS // 16) + unit * (_UNIT // 16) + tl.arange(0, _UNIT // 16)[None, :]))
    else:
        exponent = tl.load(scale + rows[:, None] * (COLS // 32) + unit * (_UNIT // 32) + tl.arange(0, _UNIT // 32)[None, :]).to(tl.int32)
        return ((exponent + 8) << 23).to(tl.float32, bitcast=True)  # 2^(exponent - 127) times the 256 of _values()


@triton.jit
def _unit_bf16(weight, scale, rows, unit, COLS: tl.constexpr, ROWS: tl.constexpr, KIND: tl.constexpr):
    """BF16 [ROWS, UNIT]: the values times their block scales, exact."""
    BLOCK: tl.constexpr = 16 if KIND == 0 else 32
    values = tl.reshape(_values(weight, rows, unit, COLS, ROWS, KIND), (ROWS, _UNIT // BLOCK, BLOCK))
    return tl.reshape(values * tl.reshape(_scales(scale, rows, unit, COLS, KIND), (ROWS, _UNIT // BLOCK, 1)), (ROWS, _UNIT)).to(tl.bfloat16)


@triton.jit
def _add(total, lost, value):
    """total + value and what its rounding lost, kept in lost (two-sum)."""
    new = total + value
    part = new - total
    return new, lost + (total - (new - part)) + (value - part)


@triton.jit
def _activate(gate, up, CAP_GATE, CAP_UP, SOFTCAP: tl.constexpr):
    if SOFTCAP:
        one = tl.zeros_like(gate) + 1.0
        sigmoid = tl.div_rn(one, 1.0 + libdevice.exp(-gate))
        return libdevice.tanh(tl.div_rn(gate, one * CAP_GATE)) * CAP_GATE * sigmoid * (libdevice.tanh(tl.div_rn(up, one * CAP_UP)) * CAP_UP)
    else:
        return tl.div_rn(gate, 1.0 + libdevice.exp(-gate)) * up


# Decoding kernels: a program computes one row with 32 lanes, each lane reading 4 consecutive columns per step of 128 (a 16-bit load of 4 E2M1 codes, or a 32-bit load of
# 4 E4M3 values) and keeping its float32 partial sum over the row, reduced once at its end: no value moves between threads inside the row. The steps are unrolled (UNROLLED: the
# loads of every step of a row in flight; 1.5 times faster at one token on an RTX 5090) and every sum is a separately rounded add (_radd): unrolled, LLVM pairs the gate and up
# chains into packed f32x2 multiplies and adds, which ptxas then fuses into FMAs on sm_120 whatever enable_fp_fusion says, and a token would no longer get the same bits in
# every kernel that computes it. On AMD GPUs the adds are plain (the PTX instruction does not exist there; bit identity between the kernels not validated there).


@triton.jit
def _lane_weights(weight, scale, step, lanes, KIND: tl.constexpr):
    """The 4 columns of each lane at a step of a row (weight, scale: its packed values and its block scales), then the lane's block scale. NVFP4: the scale over 256; MXFP8: the values over
    256 and the scale times 256."""
    if KIND == 0:
        word = tl.load((weight + step * 64 + lanes * 2).to(tl.pointer_type(tl.uint16))).to(tl.int32)
        block = _e4m3(tl.load(scale + step * 8 + (lanes >> 2)).to(tl.int32))
        return _decode_e2m1(word & 15), _decode_e2m1((word >> 4) & 15), _decode_e2m1((word >> 8) & 15), _decode_e2m1(word >> 12), block
    else:
        word = tl.load((weight + step * 128 + lanes * 4).to(tl.pointer_type(tl.int32)))
        block = ((tl.load(scale + step * 4 + (lanes >> 3)).to(tl.int32) + 8) << 23).to(tl.float32, bitcast=True)
        return _e4m3(word & 255), _e4m3((word >> 8) & 255), _e4m3((word >> 16) & 255), _e4m3((word >> 24) & 255), block


@triton.jit
def _radd(a, b):
    """a + b rounded on its own (never part of an FMA)."""
    if _NVIDIA:
        return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=r,r,r", [a, b], dtype=tl.float32, is_pure=True, pack=1)
    else:
        return a + b


@triton.jit
def _rsum(x):
    """The sum of the lanes, [1]."""
    return tl.reduce(x, 0, _radd, keep_dims=True)


@triton.jit
def _lane_dot(w0, w1, w2, w3, block, x0, x1, x2, x3):
    """A lane's 4 weights and block scale (_lane_weights) times its 4 activations (_four_activations), as values: Triton 3.2 (Turing) passes no tuple to a function."""
    return _radd(_radd(_radd(w0 * x0, w1 * x1), w2 * x2), w3 * x3) * block


@triton.jit
def _gate_up_step(gate, up, weights, scales, activations, step, lanes, WIDTH: tl.constexpr, SCALES: tl.constexpr, KIND: tl.constexpr):
    x0, x1, x2, x3 = _four_activations(activations + step * 128, 0)
    w0, w1, w2, w3, block = _lane_weights(weights, scales, step, lanes, KIND)
    gate = _radd(gate, _lane_dot(w0, w1, w2, w3, block, x0, x1, x2, x3))
    w0, w1, w2, w3, block = _lane_weights(weights + WIDTH, scales + SCALES, step, lanes, KIND)
    return gate, _radd(up, _lane_dot(w0, w1, w2, w3, block, x0, x1, x2, x3))


@triton.jit
def _down_step(value, weights, scales, activations, step, lanes, KIND: tl.constexpr):
    w0, w1, w2, w3, block = _lane_weights(weights, scales, step, lanes, KIND)
    x0, x1, x2, x3 = _four_activations(activations + step * 128, 0)
    return _radd(value, _lane_dot(w0, w1, w2, w3, block, x0, x1, x2, x3))


@triton.jit
def _gate_up_lanes(weights, scales, activations, lanes, H: tl.constexpr, WIDTH: tl.constexpr, SCALES: tl.constexpr, KIND: tl.constexpr):
    """The lanes' partial sums of a row of the gate and of the up projection for one token."""
    gate = tl.zeros((32,), tl.float32)
    up = tl.zeros((32,), tl.float32)
    if UNROLLED:
        for step in tl.static_range(H // 128):
            gate, up = _gate_up_step(gate, up, weights, scales, activations, step, lanes, WIDTH, SCALES, KIND)
    else:
        for step in range(H // 128):
            gate, up = _gate_up_step(gate, up, weights, scales, activations, step, lanes, WIDTH, SCALES, KIND)
    return gate, up


@triton.jit
def _down_lanes(weights, scales, activations, lanes, I: tl.constexpr, KIND: tl.constexpr):
    """The lanes' partial sums of an output row of the down projection for one pair."""
    value = tl.zeros((32,), tl.float32)
    if UNROLLED:
        for step in tl.static_range(I // 128):
            value = _down_step(value, weights, scales, activations, step, lanes, KIND)
    else:
        for step in range(I // 128):
            value = _down_step(value, weights, scales, activations, step, lanes, KIND)
    return value


@triton.jit(do_not_specialize=("STRIDE", "CAP_GATE", "CAP_UP"), do_not_specialize_on_alignment=("X", "IDs", "PTR", "Mid"))
def _gate_up_row(X, IDs, PTR, Mid, STRIDE, CAP_GATE, CAP_UP, H: tl.constexpr, I: tl.constexpr, K: tl.constexpr, KIND: tl.constexpr, SOFTCAP: tl.constexpr):
    # One row of the gate and up projections of one (token, slot) pair.
    pair = tl.program_id(0)
    if not _NVIDIA:  # 64-bit offsets from the pair: Triton 3.6's AMD buffer-op conversion fails on this kernel's 32-bit ones (TritonAMDGPUConvertToBufferOps)
        pair = pair.to(tl.int64)
    index = tl.program_id(1)
    expert = tl.load(IDs + pair).to(tl.int64)
    row = PTR + tl.maximum(expert, 0) * STRIDE
    address = tl.where(expert >= 0, tl.load(row), 0)
    gate = tl.zeros((32,), tl.float32)
    up = tl.zeros((32,), tl.float32)
    WIDTH: tl.constexpr = H // 2 if KIND == 0 else H
    SCALES: tl.constexpr = H // 16 if KIND == 0 else H // 32
    if address != 0:
        weights = address.to(tl.pointer_type(tl.uint8)) + index * (2 * WIDTH)
        scales = tl.load(row + 2).to(tl.pointer_type(tl.uint8)) + index * (2 * SCALES)
        lanes = tl.arange(0, 32)
        gate, up = _gate_up_lanes(weights, scales, X + (pair // K) * H + lanes * 4, lanes, H, WIDTH, SCALES, KIND)
        if KIND == 0:
            factor = tl.load(tl.load(row + 4).to(tl.pointer_type(tl.float32))) * 256.0
            gate, up = gate * factor, up * factor
    tl.store(Mid + pair * I + index + tl.arange(0, 1), _activate(_rsum(gate), _rsum(up), CAP_GATE, CAP_UP, SOFTCAP))


@triton.jit(do_not_specialize=("STRIDE",), do_not_specialize_on_alignment=("Mid", "IDs", "Probs", "PTR", "Out"))
def _down_row(Mid, IDs, Probs, PTR, Out, STRIDE, H: tl.constexpr, I: tl.constexpr, K: tl.constexpr, KIND: tl.constexpr):
    # One output row of one token, summed over its slots whose experts have a GPU copy.
    token = tl.program_id(0)
    index = tl.program_id(1)
    lanes = tl.arange(0, 32)
    total = tl.zeros((1,), tl.float32)  # slot after slot, each one's lanes summed first: what the leader kernels of a wider pass compute for a token (_down_shared, _sum_slots)
    WIDTH: tl.constexpr = I // 2 if KIND == 0 else I
    SCALES: tl.constexpr = I // 16 if KIND == 0 else I // 32
    for slot in range(K):
        pair = token * K + slot
        expert = tl.load(IDs + pair).to(tl.int64)
        row = PTR + tl.maximum(expert, 0) * STRIDE
        address = tl.where(expert >= 0, tl.load(row + 1), 0)
        if address != 0:
            weights = address.to(tl.pointer_type(tl.uint8)) + index * WIDTH
            scales = tl.load(row + 3).to(tl.pointer_type(tl.uint8)) + index * SCALES
            value = _down_lanes(weights, scales, Mid + pair * I + lanes * 4, lanes, I, KIND)
            probability = tl.load(Probs + pair).to(tl.float32)
            if KIND == 0:
                probability = probability * (tl.load(tl.load(row + 5).to(tl.pointer_type(tl.float32))) * 256.0)
            total = _radd(total, _rsum(value * probability))
    tl.store(Out + token * H + index + tl.arange(0, 1), total)


@triton.jit(do_not_specialize=("STRIDE",), do_not_specialize_on_alignment=("Mid", "IDs", "Probs", "PTR", "Out"))
def _down_rows(Mid, IDs, Probs, PTR, Out, STRIDE, H: tl.constexpr, I: tl.constexpr, K: tl.constexpr, KIND: tl.constexpr, R: tl.constexpr):
    # _down_row for R output rows a program: the slot's activations of a step loaded once for the R rows; each row keeps _down_row's lanes, steps and adds (the same bits)
    token = tl.program_id(0)
    index = tl.program_id(1) * R + tl.arange(0, R)
    lanes = tl.arange(0, 32)
    total = tl.zeros((R, 1), tl.float32)
    WIDTH: tl.constexpr = I // 2 if KIND == 0 else I
    SCALES: tl.constexpr = I // 16 if KIND == 0 else I // 32
    for slot in range(K):
        pair = token * K + slot
        expert = tl.load(IDs + pair).to(tl.int64)
        row = PTR + tl.maximum(expert, 0) * STRIDE
        address = tl.where(expert >= 0, tl.load(row + 1), 0)
        if address != 0:
            weights = address.to(tl.pointer_type(tl.uint8)) + (index * WIDTH)[:, None]
            scales = tl.load(row + 3).to(tl.pointer_type(tl.uint8)) + (index * SCALES)[:, None]
            activations = Mid + pair * I + lanes * 4
            value = tl.zeros((R, 32), tl.float32)
            for step in tl.static_range(I // 128):
                value = _down_step(value, weights, scales, activations, step, lanes[None, :], KIND)
            probability = tl.load(Probs + pair).to(tl.float32)
            if KIND == 0:
                probability = probability * (tl.load(tl.load(row + 5).to(tl.pointer_type(tl.float32))) * 256.0)
            total = _radd(total, tl.reduce(value * probability, 1, _radd, keep_dims=True))
    tl.store(Out + token * H + index[:, None], total)


# Decoding passes of a few tokens (verification windows) run the row kernels above over all their pairs in one launch by default (WINDOW_ROWS): the pairs of a row index run side
# by side, so the tokens that route to an expert read its row from VRAM once and from the L2 cache after (RTX 5090, real routing, 8 tokens: 10.6 ms of experts per pass against 12.5
# for the leader kernels below, which decode a shared row once per token anyway). WGP_INKLING2_DISABLE=window_rows: the leader kernels, where the first (token, slot) pair of an
# expert, its leader, computes every pair of that expert and the others do nothing; Members gives a leader the pair of each token (-1: none), found on the GPU: nothing returns to
# the host. Both give a token the bits of a pass of one row.


@triton.jit(do_not_specialize=("TOKENS",), do_not_specialize_on_alignment=("IDs", "Members"))
def _members(IDs, Members, TOKENS, K: tl.constexpr, ROWS: tl.constexpr):
    pair = tl.program_id(0)
    expert = tl.load(IDs + pair).to(tl.int64)
    token = tl.arange(0, ROWS)[:, None]
    slot = tl.arange(0, K)[None, :]
    inside = (token < TOKENS) & (slot >= 0)
    same = (tl.load(IDs + token * K + slot, mask=inside, other=-1).to(tl.int64) == expert) & inside & (expert >= 0)
    leader = tl.sum((same & (token < pair // K)).to(tl.int32)) == 0
    found = tl.max(tl.where(same, token * K + slot, -1), axis=1)  # a token routes to an expert at most once
    tl.store(Members + pair * ROWS + tl.arange(0, ROWS), tl.where(leader, found, -1))


@triton.jit(do_not_specialize=("STRIDE", "CAP_GATE", "CAP_UP", "TOKENS"), do_not_specialize_on_alignment=("X", "IDs", "PTR", "Mid", "Members"))
def _gate_up_shared(X, IDs, PTR, Mid, Members, STRIDE, CAP_GATE, CAP_UP, TOKENS, H: tl.constexpr, I: tl.constexpr, K: tl.constexpr, KIND: tl.constexpr, SOFTCAP: tl.constexpr, ROWS: tl.constexpr,
                    PROGRAM: tl.constexpr):
    # SHARED_PROGRAM rows of the gate and up projections of an expert, for every pair routed to it (the program of its leader; the others end at once).
    pair = tl.program_id(0)
    expert = tl.load(IDs + pair).to(tl.int64)
    row = PTR + tl.maximum(expert, 0) * STRIDE
    address = tl.where(expert >= 0, tl.load(row), 0)
    WIDTH: tl.constexpr = H // 2 if KIND == 0 else H
    SCALES: tl.constexpr = H // 16 if KIND == 0 else H // 32
    leader = tl.load(Members + pair * ROWS + pair // K) >= 0
    if (address != 0) & leader:
        lanes = tl.arange(0, 32)
        factor = 1.0
        if KIND == 0:
            factor = tl.load(tl.load(row + 4).to(tl.pointer_type(tl.float32))) * 256.0
        for line in range(PROGRAM):
            index = tl.program_id(1) * PROGRAM + line
            weights = address.to(tl.pointer_type(tl.uint8)) + index * (2 * WIDTH)
            scales = tl.load(row + 2).to(tl.pointer_type(tl.uint8)) + index * (2 * SCALES)
            for token in range(TOKENS):
                member = tl.load(Members + pair * ROWS + token)
                if member >= 0:
                    gate, up = _gate_up_lanes(weights, scales, X + token * H + lanes * 4, lanes, H, WIDTH, SCALES, KIND)
                    if KIND == 0:
                        gate, up = gate * factor, up * factor
                    tl.store(Mid + member.to(tl.int64) * I + index + tl.arange(0, 1), _activate(_rsum(gate), _rsum(up), CAP_GATE, CAP_UP, SOFTCAP))


@triton.jit(do_not_specialize=("STRIDE", "TOKENS"), do_not_specialize_on_alignment=("Mid", "IDs", "Probs", "PTR", "Parts", "Members"))
def _down_shared(Mid, IDs, Probs, PTR, Parts, Members, STRIDE, TOKENS, H: tl.constexpr, I: tl.constexpr, K: tl.constexpr, KIND: tl.constexpr, ROWS: tl.constexpr, PROGRAM: tl.constexpr):
    # PROGRAM output rows of an expert for every pair routed to it, each into the part of its pair (zero for a pair whose expert has no GPU copy).
    pair = tl.program_id(0)
    expert = tl.load(IDs + pair).to(tl.int64)
    row = PTR + tl.maximum(expert, 0) * STRIDE
    address = tl.where(expert >= 0, tl.load(row + 1), 0)
    WIDTH: tl.constexpr = I // 2 if KIND == 0 else I
    SCALES: tl.constexpr = I // 16 if KIND == 0 else I // 32
    if address != 0:
        if tl.load(Members + pair * ROWS + pair // K) >= 0:  # the leader of its expert
            lanes = tl.arange(0, 32)
            factor = 1.0
            if KIND == 0:
                factor = tl.load(tl.load(row + 5).to(tl.pointer_type(tl.float32))) * 256.0
            for line in range(PROGRAM):
                index = tl.program_id(1) * PROGRAM + line
                weights = address.to(tl.pointer_type(tl.uint8)) + index * WIDTH
                scales = tl.load(row + 3).to(tl.pointer_type(tl.uint8)) + index * SCALES
                for token in range(TOKENS):
                    member = tl.load(Members + pair * ROWS + token)
                    if member >= 0:
                        value = _down_lanes(weights, scales, Mid + member.to(tl.int64) * I + lanes * 4, lanes, I, KIND)
                        tl.store(Parts + member.to(tl.int64) * H + index + tl.arange(0, 1), _rsum(value * (tl.load(Probs + member).to(tl.float32) * factor)))
    else:
        tl.store(Parts + pair.to(tl.int64) * H + tl.program_id(1) * PROGRAM + tl.arange(0, PROGRAM), tl.zeros((PROGRAM,), tl.float32))


@triton.jit(do_not_specialize_on_alignment=("Parts", "Out"))
def _sum_slots(Parts, Out, H: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    # the parts of a token's slots added in their order, as _down_row adds them
    token = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    total = tl.zeros((BLOCK,), tl.float32)
    for slot in tl.static_range(K):
        total = _radd(total, tl.load(Parts + (token * K + slot) * H + columns))
    tl.store(Out + token * H + columns, total)


@triton.jit(do_not_specialize=("FIRST", "STRIDE", "CAP_GATE", "CAP_UP"), do_not_specialize_on_alignment=("X", "Pairs", "Experts", "PTR", "Mid"))
def _gate_up_group(X, Pairs, Experts, PTR, Mid, FIRST, STRIDE, CAP_GATE, CAP_UP, H: tl.constexpr, I: tl.constexpr, K: tl.constexpr, KIND: tl.constexpr, ROWS: tl.constexpr, GROUP: tl.constexpr,
                   SOFTCAP: tl.constexpr):
    # ROWS rows of the gate and up projections of one expert for up to GROUP pairs routed to it, into Mid by place in the groups (float32 or BF16).
    group = FIRST + tl.program_id(0)
    rows = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    within = tl.arange(0, GROUP)
    row = PTR + tl.load(Experts + group).to(tl.int64) * STRIDE
    address = tl.load(row)
    if address != 0:
        weight = address.to(tl.pointer_type(tl.uint8))
        scale = tl.load(row + 2).to(tl.pointer_type(tl.uint8))
        token = (tl.maximum(tl.load(Pairs + group * GROUP + within), 0) // K).to(tl.int64)  # what the places past the end compute is not read
        activations = X + token[:, None] * H + tl.arange(0, _UNIT)[None, :]
        gate = tl.zeros((ROWS, GROUP), tl.float32)
        up = tl.zeros((ROWS, GROUP), tl.float32)
        gate_lost = tl.zeros((ROWS, GROUP), tl.float32)
        up_lost = tl.zeros((ROWS, GROUP), tl.float32)
        for unit in range(H // _UNIT):
            x = tl.trans(tl.load(activations + unit * _UNIT))
            if Mid.dtype.element_ty == tl.bfloat16:
                gate = tl.dot(_unit_bf16(weight, scale, 2 * rows, unit, H, ROWS, KIND), x, gate)
                up = tl.dot(_unit_bf16(weight, scale, 2 * rows + 1, unit, H, ROWS, KIND), x, up)
            else:  # the sums of the tensor cores are truncated: each unit summed from zero, the units with what their sums round kept
                gate, gate_lost = _add(gate, gate_lost, tl.dot(_unit_bf16(weight, scale, 2 * rows, unit, H, ROWS, KIND), x))
                up, up_lost = _add(up, up_lost, tl.dot(_unit_bf16(weight, scale, 2 * rows + 1, unit, H, ROWS, KIND), x))
        gate, up = gate + gate_lost, up + up_lost
        if KIND == 0:
            factor = tl.load(tl.load(row + 4).to(tl.pointer_type(tl.float32)))
            gate, up = gate * factor, up * factor
        place = group.to(tl.int64) * GROUP + within
        tl.store(Mid + place[None, :] * I + rows[:, None], _activate(gate, up, CAP_GATE, CAP_UP, SOFTCAP).to(Mid.dtype.element_ty))


@triton.jit(do_not_specialize=("FIRST", "STRIDE"), do_not_specialize_on_alignment=("Mid", "Pairs", "Experts", "PTR", "Probs", "Parts"))
def _down_group(Mid, Pairs, Experts, PTR, Probs, Parts, FIRST, STRIDE, H: tl.constexpr, I: tl.constexpr, KIND: tl.constexpr, ROWS: tl.constexpr, GROUP: tl.constexpr):
    # ROWS output rows of one expert for the pairs of a group, each into the part of its pair, times its probability.
    group = FIRST + tl.program_id(0)
    rows = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    within = tl.arange(0, GROUP)
    row = PTR + tl.load(Experts + group).to(tl.int64) * STRIDE
    address = tl.load(row + 1)
    if address != 0:
        weight = address.to(tl.pointer_type(tl.uint8))
        scale = tl.load(row + 3).to(tl.pointer_type(tl.uint8))
        pair = tl.load(Pairs + group * GROUP + within)
        inputs = Mid + (group.to(tl.int64) * GROUP + within)[:, None] * I + tl.arange(0, _UNIT)[None, :]
        value = tl.zeros((ROWS, GROUP), tl.float32)
        lost = tl.zeros((ROWS, GROUP), tl.float32)
        for unit in range(I // _UNIT):
            w = _unit_bf16(weight, scale, rows, unit, I, ROWS, KIND)
            mid = tl.trans(tl.load(inputs + unit * _UNIT))
            if Mid.dtype.element_ty == tl.bfloat16:
                value = tl.dot(w, mid, value)
            else:
                high = mid.to(tl.bfloat16)  # a float32 value is the sum of three BF16 values (8 bits each), exactly
                rest = mid - high.to(tl.float32)
                middle = rest.to(tl.bfloat16)
                value, lost = _add(value, lost, tl.dot(w, high))
                value, lost = _add(value, lost, tl.dot(w, middle))
                value, lost = _add(value, lost, tl.dot(w, (rest - middle.to(tl.float32)).to(tl.bfloat16)))
        value = value + lost
        valid = pair >= 0
        probability = tl.load(Probs + pair, mask=valid, other=0.0).to(tl.float32)
        if KIND == 0:
            probability = probability * tl.load(tl.load(row + 5).to(tl.pointer_type(tl.float32)))
        tl.store(Parts + pair.to(tl.int64)[:, None] * H + rows[None, :], tl.trans(value) * probability[:, None], mask=valid[:, None])


@cache
def _needs_fp32_inputs(device):
    return torch.version.hip is None and torch.cuda.get_device_capability(device) < (8, 0)


@cache
def group_tiles(device, approximate):
    """(rows, pairs, warps) of the grouped kernels on device: GROUP_TILES, or HIP_GROUP_TILES on the AMD architectures that have them."""
    if torch.version.hip is not None:
        return HIP_GROUP_TILES.get(torch.cuda.get_device_properties(device).gcnArchName.split(":")[0], GROUP_TILES[approximate])
    return GROUP_TILES[approximate]


@cache
def pass_supported(device):
    """The grouped kernels: BF16 tensor cores, NVIDIA GPUs from compute capability 8.0 and the AMD architectures of bf16_bits.HIP_ARCHS (compiled, not run there)."""
    return portable_supported(device) and not reads_bits(device)


def _check(kind, pointers, hidden, intermediate, *tensors):
    # the kernels read raw addresses: nothing but CUDA memory of the expected layout may reach them
    if hidden % 128 or intermediate % 128:
        raise ValueError(f"The row-scaled expert kernels take widths that are multiples of 128, not {hidden} and {intermediate}")
    if not (pointers.is_cuda and pointers.dtype == torch.int64 and pointers.dim() == 2 and pointers.shape[1] >= TENSORS[kind] and pointers.is_contiguous()):
        raise ValueError(f"The expert pointer table must be a contiguous CUDA int64 [experts, {TENSORS[kind]} or more] tensor")
    if not all(tensor.is_cuda and tensor.is_contiguous() for tensor in tensors):
        raise ValueError("The row-scaled expert kernels read contiguous CUDA tensors")


def _groups(ids, group, order):
    """On the host: (pairs int32 [groups, group] of pair indices, -1 past the end of an expert; experts int32 [groups]; first int64 [len(order) + 1]: the first group of each expert of order). Ids of -1 belong to no expert."""
    flat = ids.reshape(-1).numpy()
    sorted_pairs = np.argsort(flat, kind="stable").astype(np.int32)
    counts = np.bincount(flat[flat >= 0], minlength=int(order.max()) + 1 if len(order) else 0)
    starts = len(flat) - counts.sum() + np.concatenate(([0], np.cumsum(counts)))
    needed = (counts[order] + group - 1) // group
    first = np.zeros(len(order) + 1, dtype=np.int64)
    first[1:] = np.cumsum(needed)
    pairs = np.full(int(first[-1]) * group, -1, dtype=np.int32)
    for index, expert in enumerate(order.tolist()):  # at most the experts of a bank
        pairs[first[index] * group:first[index] * group + counts[expert]] = sorted_pairs[starts[expert]:starts[expert + 1]]
    return pairs.reshape(-1, group), np.repeat(order.astype(np.int32), needed), first


class GroupedPass:
    """A pass of many tokens: run() computes experts of the order of processing (some may arrive later, streamed), finish() gives the float32 [tokens, hidden] sum. ids: the routing on the host;
    order: the routed experts to compute, an int array (by default all of them); zero: the pairs of the other experts count for nothing (else every routed expert must run)."""
    def __init__(self, kind, x, ids, probabilities, intermediate, caps=None, order=None, zero=False):
        if x.dtype != torch.bfloat16:
            raise ValueError(f"The grouped expert kernels take BF16 activations, not {x.dtype}")
        self.kind, self.x, self.probabilities, self.intermediate, self.caps = kind, x.contiguous(), probabilities.reshape(-1).float().contiguous(), intermediate, caps
        self.tokens, self.hidden = x.shape
        self.topk = ids.shape[-1]
        approximate = kernel_policy.allow_approximate()
        self.rows, self.group, self.warps = group_tiles(x.device, approximate)
        if order is None:
            order = np.unique(ids.numpy())
            order = order[order >= 0]
        pairs, experts, self.first = _groups(ids, self.group, np.asarray(order, dtype=np.int64))
        self.count, device = pairs.shape[0], x.device
        self.pairs, self.experts = torch.from_numpy(pairs).to(device), torch.from_numpy(experts).to(device)
        # BF16 activations for the output projection, as the released model computes them; otherwise float32, read as three BF16 values: the precision of float32
        self.mid = torch.empty(self.count * self.group, intermediate, dtype=torch.bfloat16 if approximate else torch.float32, device=device)
        self.parts = (torch.zeros if zero else torch.empty)(self.tokens * self.topk, self.hidden, dtype=torch.float32, device=device)

    def run(self, pointers, start=0, stop=None):
        """The experts [start, stop) of the order of processing, whose rows pointers gives."""
        first = int(self.first[start])
        count = int(self.first[len(self.first) - 1 if stop is None else stop]) - first
        if not count:
            return
        _check(self.kind, pointers, self.hidden, self.intermediate, self.x)
        hidden, intermediate, stride = self.hidden, self.intermediate, pointers.shape[1]
        cap_gate, cap_up = self.caps or (0.0, 0.0)
        _gate_up_group[(count, intermediate // self.rows)](self.x, self.pairs, self.experts, pointers, self.mid, first, stride, cap_gate, cap_up, hidden, intermediate, self.topk, self.kind, self.rows, self.group,
                                                           self.caps is not None, num_warps=self.warps, num_stages=1)
        _down_group[(count, hidden // self.rows)](self.mid, self.pairs, self.experts, pointers, self.probabilities, self.parts, first, stride, hidden, intermediate, self.kind, self.rows, self.group,
                                                  num_warps=self.warps, num_stages=1)

    def finish(self):
        self.mid = self.pairs = self.experts = None
        output = self.parts.view(self.tokens, self.topk, self.hidden).sum(1)
        self.parts = None
        return output


def forward(kind, x, ids, probabilities, pointers, intermediate, caps=None, out=None, mid=None):
    """float32 [tokens, hidden] sum of the selected experts having GPU copies (zero for the others, and for ids of -1); out and the float32 [tokens * topk, intermediate] scratch mid may be
    preallocated. kind: NVFP4 or MXFP8; pointers: int64 [experts, 4 or more] (6 for NVFP4); caps: (gate cap, up cap) of the soft caps, None for plain SwiGLU. Eager calls of many pairs run by
    groups (they read the routing and the table on the host)."""
    tokens, hidden = x.shape
    topk = ids.shape[-1]
    if out is None and tokens * topk >= GROUPED_PAIRS and x.dtype == torch.bfloat16 and pass_supported(x.device) and not torch.cuda.is_current_stream_capturing():
        host_ids = ids.to("cpu")
        routed = np.bincount(host_ids.numpy().ravel().clip(0), minlength=pointers.shape[0]) > 0
        job = GroupedPass(kind, x, host_ids, probabilities, intermediate, caps, np.nonzero(routed & (pointers[:, 0].to("cpu").numpy() != 0))[0], zero=True)
        job.run(pointers)
        return job.finish()
    if _needs_fp32_inputs(x.device):  # these kernels compute in float32: converted exactly before launch on GPUs without BF16 conversions
        x, probabilities = x.float(), probabilities.float()
    mid = torch.empty(tokens * topk, intermediate, dtype=torch.float32, device=x.device) if mid is None else mid
    out = torch.empty(tokens, hidden, dtype=torch.float32, device=x.device) if out is None else out
    _check(kind, pointers, hidden, intermediate, x, ids, probabilities, mid, out)
    if mid.dtype != torch.float32 or out.dtype != torch.float32 or mid.numel() < tokens * topk * intermediate or out.numel() != tokens * hidden or ids.numel() != tokens * topk:
        raise ValueError("The scratch and the output of the row-scaled expert kernels are float32 tensors of the call's sizes")
    cap_gate, cap_up = caps or (0.0, 0.0)
    if (x.data_ptr() | mid.data_ptr()) & 7 or x.dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("The decoding kernels read BF16 or float32 activations and a scratch aligned on 8 bytes")
    if not WINDOW_ROWS and SHARED and 1 < tokens <= SHARED_ROWS:  # a verification window through the leader kernels: each routed expert read once for the tokens that share it
        members = torch.empty(tokens * topk, SHARED_ROWS, dtype=torch.int32, device=x.device)
        parts = torch.empty(tokens * topk, hidden, dtype=torch.float32, device=x.device)
        _members[(tokens * topk,)](ids, members, tokens, topk, SHARED_ROWS, num_warps=1)
        _gate_up_shared[(tokens * topk, intermediate // SHARED_PROGRAM)](x, ids, pointers, mid, members, pointers.shape[1], cap_gate, cap_up, tokens, hidden, intermediate, topk, kind, caps is not None, SHARED_ROWS,
                                                                         SHARED_PROGRAM, num_warps=1, enable_fp_fusion=False)
        _down_shared[(tokens * topk, hidden // SHARED_PROGRAM)](mid, ids, probabilities, pointers, parts, members, pointers.shape[1], tokens, hidden, intermediate, topk, kind, SHARED_ROWS, SHARED_PROGRAM, num_warps=1,
                                                                enable_fp_fusion=False)
        _sum_slots[(tokens, hidden // 256)](parts, out, hidden, topk, 256, num_warps=1)
        return out
    if not WINDOW_ROWS and not SHARED and tokens > 1:  # a launch per token (a comparison)
        for token in range(tokens):
            forward(kind, x[token:token + 1], ids[token:token + 1], probabilities[token:token + 1], pointers, intermediate, caps, out[token:token + 1], mid[token * topk:(token + 1) * topk])
        return out
    # (no contraction of a product and a sum: a token gets the same result in passes of any rows and in either kernel family)
    _gate_up_row[(tokens * topk, intermediate)](x, ids, pointers, mid, pointers.shape[1], cap_gate, cap_up, hidden, intermediate, topk, kind, caps is not None, num_warps=1, enable_fp_fusion=False)
    if DOWN_PAIRS and tokens >= DOWN_PAIR_TOKENS and UNROLLED:  # a wide window: two output rows a program read a slot's activations once (5 % faster at 8 rows, slower below)
        _down_rows[(tokens, hidden // 2)](mid, ids, probabilities, pointers, out, pointers.shape[1], hidden, intermediate, topk, kind, 2, num_warps=1, enable_fp_fusion=False)
    else:
        _down_row[(tokens, hidden)](mid, ids, probabilities, pointers, out, pointers.shape[1], hidden, intermediate, topk, kind, num_warps=1, enable_fp_fusion=False)
    return out


@triton.jit(do_not_specialize=("PAIRS",), do_not_specialize_on_alignment=("IDs", "Counts"))
def _count_pairs(IDs, Counts, PAIRS, BLOCK: tl.constexpr):
    pairs = tl.arange(0, BLOCK)
    expert = tl.load(IDs + pairs, mask=pairs < PAIRS, other=-1)
    tl.atomic_add(Counts + tl.maximum(expert, 0), tl.full((BLOCK,), 1, tl.int32), mask=(pairs < PAIRS) & (expert >= 0))  # integer adds: the counts do not depend on their order


COUNT_BLOCK = 1024  # (token, slot) pairs of a decoding or verification pass at most


def count_pairs(ids, counts):
    """counts[e] += the pairs of ids routed to expert e (int32 or int64 ids, -1 for none; int32 counts), in one program: graph-safe, no host synchronization."""
    if ids.numel() > COUNT_BLOCK or ids.dtype not in (torch.int32, torch.int64) or counts.dtype != torch.int32 or not ids.is_contiguous():
        raise ValueError("The pair counts take the int32 / int64 ids of a short pass")  # int64: the PyTorch router's (fused row kernels off)
    _count_pairs[(1,)](ids, counts, ids.numel(), COUNT_BLOCK, num_warps=4)
    torch.autograd.graph.increment_version(counts)


from shared.kernels.triton_compilation_log import install_triton_compilation_logger  # noqa: E402

install_triton_compilation_logger()
