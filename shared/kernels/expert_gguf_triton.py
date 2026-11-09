# Copyright (c) 2026 DeepBeepMeep - WanGP Community License 2.0 (see LICENSE.txt)
"""Selected experts of routed banks in GGUF storage, on the GPU: the compact rows of the checkpoint stay as they are in VRAM (more experts fit) and are decoded as they are read,
each expert addressed through a pointer table (one row per expert: gate, up and down rows), so that a CUDA graph replays any routing. IQ2_S, IQ3_S, IQ3_XXS, IQ4_XS and Q2_0 for
the input projections, IQ4_NL and Q2_0 for the output projection. The block layouts are ggml's (MIT License); float32 activations and accumulation, the values of the PyTorch
dequantization. Experts without a GPU copy contribute zero.

Decoding passes read one (token, slot) pair at a time. Passes of many tokens (prompt processing) group the tokens routed to each expert, which decodes its rows once for all of them."""
import os
from functools import cache

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

IQ3_XXS, IQ4_NL, IQ3_S, IQ2_S, IQ4_XS, Q2_0 = 18, 20, 21, 22, 23, 42  # ggml types (literal in the kernels)


@triton.jit
def _input_unit(row, unit, x, GRID, VALUES, TYPE: tl.constexpr, ROWS: tl.constexpr):
    """The product of ROWS rows (pointers to their blocks) with x, over the 32 columns (or 64 for Q2_0) of a unit: [ROWS] float32."""
    c2 = tl.arange(0, 2)
    c4 = tl.arange(0, 4)
    c8 = tl.arange(0, 8)
    c16 = tl.arange(0, 16)
    if TYPE == 21:  # 2: scale, 64: codes, 8: 9th bits, 32: signs, 4: scales of the units; a code gives 4 magnitudes
        block, s = row + (unit // 8) * 110, unit % 8
        codes = tl.load(block[:, None] + 2 + s * 8 + c8[None, :]).to(tl.int32) | (((tl.load(block + 66 + s).to(tl.int32)[:, None] >> c8[None, :]) & 1) << 8)
        magnitudes = (tl.load(GRID + codes)[:, :, None] >> (c4[None, None, :] * 8)) & 255
        signs = (tl.load(block[:, None] + 74 + s * 4 + c4[None, :]).to(tl.int32)[:, :, None] >> c8[None, None, :]) & 1
        w = tl.reshape(magnitudes, (ROWS, 32)).to(tl.float32) * (1 - 2 * tl.reshape(signs, (ROWS, 32))).to(tl.float32)
        scale = tl.load(block.to(tl.pointer_type(tl.float16))).to(tl.float32) * (1 + 2 * ((tl.load(block + 106 + s // 2).to(tl.int32) >> (4 * (s % 2))) & 15)).to(tl.float32)
        return tl.sum(w * x[None, :], 1) * scale
    elif TYPE == 18:  # 2: scale, 64: codes, 32: a word per unit (4 scale bits, 4 x 7 sign bits, the eighth by parity); a code gives 4 magnitudes
        block, s = row + (unit // 8) * 98, unit % 8
        codes = tl.load(block[:, None] + 2 + s * 8 + c8[None, :]).to(tl.int32)
        magnitudes = (tl.load(GRID + codes)[:, :, None] >> (c4[None, None, :] * 8)) & 255
        word = tl.sum(tl.load(block[:, None] + 66 + s * 4 + c4[None, :]).to(tl.int32) << (c4[None, :] * 8), 1)  # bytes: the words are not aligned
        groups = (word[:, None] >> (c4[None, :] * 7)) & 127
        bits = (groups[:, :, None] >> c8[None, None, :]) & 1  # bit 7 of a group is 0 here: the parity below
        parity = tl.sum(bits, 2) & 1
        bits = bits + (c8[None, None, :] == 7).to(tl.int32) * parity[:, :, None]
        w = tl.reshape(magnitudes, (ROWS, 32)).to(tl.float32) * (1 - 2 * tl.reshape(bits, (ROWS, 32))).to(tl.float32)
        scale = tl.load(block.to(tl.pointer_type(tl.float16))).to(tl.float32) * (0.5 + ((word >> 28) & 15).to(tl.float32)) * 0.5
        return tl.sum(w * x[None, :], 1) * scale
    elif TYPE == 22:  # 2: scale, 32: codes, 32: signs, 8: 2 high bits per code, 8: a scale per 16 values; a code gives 8 magnitudes
        block, s = row + (unit // 8) * 82, unit % 8
        codes = tl.load(block[:, None] + 2 + s * 4 + c4[None, :]).to(tl.int32) | (((tl.load(block + 66 + s).to(tl.int32)[:, None] >> (c4[None, :] * 2)) & 3) << 8)
        magnitudes = (tl.load(GRID + codes)[:, :, None] >> (c8[None, None, :] * 8)) & 255
        signs = (tl.load(block[:, None] + 34 + s * 4 + c4[None, :]).to(tl.int32)[:, :, None] >> c8[None, None, :]) & 1
        w = tl.reshape(magnitudes, (ROWS, 32)).to(tl.float32) * (1 - 2 * tl.reshape(signs, (ROWS, 32))).to(tl.float32)
        d = tl.load(block.to(tl.pointer_type(tl.float16))).to(tl.float32)
        scales = tl.load(block + 74 + s).to(tl.int32)
        low, high = (scales & 15).to(tl.float32) + 0.5, ((scales >> 4) & 15).to(tl.float32) + 0.5
        first = (tl.arange(0, 32) < 16).to(tl.float32)[None, :]
        return tl.sum(w * x[None, :] * (low[:, None] * first + high[:, None] * (1.0 - first)), 1) * d * 0.25
    elif TYPE == 23:  # 2: scale, 2: high scale bits, 4: low scale bits, 128: values, 16 bytes per unit, low nibbles first
        block, s = row + (unit // 8) * 136, unit % 8
        q = tl.load(block[:, None] + 8 + s * 16 + c16[None, :]).to(tl.int32)
        w = tl.reshape(tl.load(VALUES + ((q[:, None, :] >> (c2[None, :, None] * 4)) & 15)), (ROWS, 32))
        low = (tl.load(block + 4 + s // 2).to(tl.int32) >> (4 * (s % 2))) & 15
        high = (tl.load((block + 2).to(tl.pointer_type(tl.uint16))).to(tl.int32) >> (2 * s)) & 3
        scale = tl.load(block.to(tl.pointer_type(tl.float16))).to(tl.float32) * ((low | (high << 4)) - 32).to(tl.float32)
        return tl.sum(w * x[None, :], 1) * scale
    else:  # Q2_0, 64 columns: 2: scale, 16: four codes per byte, value code - 1
        block = row + unit * 18
        q = tl.load(block[:, None] + 2 + c16[None, :]).to(tl.int32)
        w = tl.reshape((q[:, :, None] >> (c4[None, None, :] * 2)) & 3, (ROWS, 64)).to(tl.float32) - 1.0
        return tl.sum(w * x[None, :], 1) * tl.load(block.to(tl.pointer_type(tl.float16))).to(tl.float32)


@triton.jit
def _input_unit_weights(row, unit, GRID, VALUES, TYPE: tl.constexpr, ROWS: tl.constexpr):
    """The weights of ROWS rows (pointers to their blocks) over the 32 columns (or 64 for Q2_0) of a unit: [ROWS, columns] float32."""
    c2 = tl.arange(0, 2)
    c4 = tl.arange(0, 4)
    c8 = tl.arange(0, 8)
    c16 = tl.arange(0, 16)
    if TYPE == 21:
        block, s = row + (unit // 8) * 110, unit % 8
        codes = tl.load(block[:, None] + 2 + s * 8 + c8[None, :]).to(tl.int32) | (((tl.load(block + 66 + s).to(tl.int32)[:, None] >> c8[None, :]) & 1) << 8)
        magnitudes = (tl.load(GRID + codes)[:, :, None] >> (c4[None, None, :] * 8)) & 255
        signs = (tl.load(block[:, None] + 74 + s * 4 + c4[None, :]).to(tl.int32)[:, :, None] >> c8[None, None, :]) & 1
        w = tl.reshape(magnitudes, (ROWS, 32)).to(tl.float32) * (1 - 2 * tl.reshape(signs, (ROWS, 32))).to(tl.float32)
        scale = tl.load(block.to(tl.pointer_type(tl.float16))).to(tl.float32) * (1 + 2 * ((tl.load(block + 106 + s // 2).to(tl.int32) >> (4 * (s % 2))) & 15)).to(tl.float32)
        return w * scale[:, None]
    elif TYPE == 18:
        block, s = row + (unit // 8) * 98, unit % 8
        codes = tl.load(block[:, None] + 2 + s * 8 + c8[None, :]).to(tl.int32)
        magnitudes = (tl.load(GRID + codes)[:, :, None] >> (c4[None, None, :] * 8)) & 255
        word = tl.sum(tl.load(block[:, None] + 66 + s * 4 + c4[None, :]).to(tl.int32) << (c4[None, :] * 8), 1)
        groups = (word[:, None] >> (c4[None, :] * 7)) & 127
        bits = (groups[:, :, None] >> c8[None, None, :]) & 1
        parity = tl.sum(bits, 2) & 1
        bits = bits + (c8[None, None, :] == 7).to(tl.int32) * parity[:, :, None]
        w = tl.reshape(magnitudes, (ROWS, 32)).to(tl.float32) * (1 - 2 * tl.reshape(bits, (ROWS, 32))).to(tl.float32)
        scale = tl.load(block.to(tl.pointer_type(tl.float16))).to(tl.float32) * (0.5 + ((word >> 28) & 15).to(tl.float32)) * 0.5
        return w * scale[:, None]
    elif TYPE == 22:
        block, s = row + (unit // 8) * 82, unit % 8
        codes = tl.load(block[:, None] + 2 + s * 4 + c4[None, :]).to(tl.int32) | (((tl.load(block + 66 + s).to(tl.int32)[:, None] >> (c4[None, :] * 2)) & 3) << 8)
        magnitudes = (tl.load(GRID + codes)[:, :, None] >> (c8[None, None, :] * 8)) & 255
        signs = (tl.load(block[:, None] + 34 + s * 4 + c4[None, :]).to(tl.int32)[:, :, None] >> c8[None, None, :]) & 1
        w = tl.reshape(magnitudes, (ROWS, 32)).to(tl.float32) * (1 - 2 * tl.reshape(signs, (ROWS, 32))).to(tl.float32)
        d = tl.load(block.to(tl.pointer_type(tl.float16))).to(tl.float32)
        scales = tl.load(block + 74 + s).to(tl.int32)
        low, high = (scales & 15).to(tl.float32) + 0.5, ((scales >> 4) & 15).to(tl.float32) + 0.5
        first = (tl.arange(0, 32) < 16).to(tl.float32)[None, :]
        return w * (low[:, None] * first + high[:, None] * (1.0 - first)) * d[:, None] * 0.25
    elif TYPE == 23:
        block, s = row + (unit // 8) * 136, unit % 8
        q = tl.load(block[:, None] + 8 + s * 16 + c16[None, :]).to(tl.int32)
        w = tl.reshape(tl.load(VALUES + ((q[:, None, :] >> (c2[None, :, None] * 4)) & 15)), (ROWS, 32))
        low = (tl.load(block + 4 + s // 2).to(tl.int32) >> (4 * (s % 2))) & 15
        high = (tl.load((block + 2).to(tl.pointer_type(tl.uint16))).to(tl.int32) >> (2 * s)) & 3
        scale = tl.load(block.to(tl.pointer_type(tl.float16))).to(tl.float32) * ((low | (high << 4)) - 32).to(tl.float32)
        return w * scale[:, None]
    else:
        block = row + unit * 18
        q = tl.load(block[:, None] + 2 + c16[None, :]).to(tl.int32)
        w = tl.reshape((q[:, :, None] >> (c4[None, None, :] * 2)) & 3, (ROWS, 64)).to(tl.float32) - 1.0
        return w * tl.load(block.to(tl.pointer_type(tl.float16))).to(tl.float32)[:, None]


@triton.jit
def _gate_up_groups(X, Pairs, Experts, PTR, GRID, VALUES, Mid, H: tl.constexpr, I: tl.constexpr, K: tl.constexpr, COLUMNS: tl.constexpr, ROW_BYTES, TYPE: tl.constexpr, ROWS: tl.constexpr, GROUP: tl.constexpr):
    # ROWS rows of the gate and up projections of one expert, for up to GROUP pairs routed to it (-1: none), a unit of columns at a time.
    group = tl.program_id(0)
    rows = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    pair = tl.load(Pairs + group * GROUP + tl.arange(0, GROUP))
    valid = pair >= 0
    expert = tl.load(Experts + group).to(tl.int64)
    gate_rows = tl.load(PTR + expert * 3).to(tl.pointer_type(tl.uint8)) + rows * ROW_BYTES
    up_rows = tl.load(PTR + expert * 3 + 1).to(tl.pointer_type(tl.uint8)) + rows * ROW_BYTES
    columns = tl.arange(0, COLUMNS)
    activations = (pair // K).to(tl.int64)[:, None] * H + columns[None, :]
    gate = tl.zeros((ROWS, GROUP), tl.float32)
    up = tl.zeros((ROWS, GROUP), tl.float32)
    for unit in range(H // COLUMNS):
        x = tl.trans(tl.load(X + activations + unit * COLUMNS, mask=valid[:, None], other=0.0).to(tl.float32))
        gate += tl.dot(_input_unit_weights(gate_rows, unit, GRID, VALUES, TYPE, ROWS), x, input_precision="ieee")
        up += tl.dot(_input_unit_weights(up_rows, unit, GRID, VALUES, TYPE, ROWS), x, input_precision="ieee")
    tl.store(Mid + pair.to(tl.int64)[None, :] * I + rows[:, None], tl.div_rn(gate, 1.0 + libdevice.exp(-gate)) * up, mask=valid[None, :])


@triton.jit
def _down_groups(Mid, Pairs, Experts, PTR, VALUES, Parts, H: tl.constexpr, I: tl.constexpr, ROW_BYTES, TYPE: tl.constexpr, ROWS: tl.constexpr, GROUP: tl.constexpr):
    # ROWS output rows of one expert for up to GROUP pairs routed to it.
    group = tl.program_id(0)
    rows = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    pair = tl.load(Pairs + group * GROUP + tl.arange(0, GROUP))
    valid = pair >= 0
    expert = tl.load(Experts + group).to(tl.int64)
    row = tl.load(PTR + expert * 3 + 2).to(tl.pointer_type(tl.uint8)) + rows * ROW_BYTES
    c2 = tl.arange(0, 2)
    c4 = tl.arange(0, 4)
    c16 = tl.arange(0, 16)
    value = tl.zeros((ROWS, GROUP), tl.float32)
    if TYPE == 20:
        columns = tl.arange(0, 32)
        inputs = pair.to(tl.int64)[:, None] * I + columns[None, :]
        for unit in range(I // 32):
            block = row + unit * 18
            q = tl.load(block[:, None] + 2 + c16[None, :]).to(tl.int32)
            w = tl.reshape(tl.load(VALUES + ((q[:, None, :] >> (c2[None, :, None] * 4)) & 15)), (ROWS, 32)) * tl.load(block.to(tl.pointer_type(tl.float16))).to(tl.float32)[:, None]
            value += tl.dot(w, tl.trans(tl.load(Mid + inputs + unit * 32, mask=valid[:, None], other=0.0)), input_precision="ieee")
    else:
        columns = tl.arange(0, 64)
        inputs = pair.to(tl.int64)[:, None] * I + columns[None, :]
        for unit in range(I // 64):
            block = row + unit * 18
            q = tl.load(block[:, None] + 2 + c16[None, :]).to(tl.int32)
            w = (tl.reshape((q[:, :, None] >> (c4[None, None, :] * 2)) & 3, (ROWS, 64)).to(tl.float32) - 1.0) * tl.load(block.to(tl.pointer_type(tl.float16))).to(tl.float32)[:, None]
            value += tl.dot(w, tl.trans(tl.load(Mid + inputs + unit * 64, mask=valid[:, None], other=0.0)), input_precision="ieee")
    tl.store(Parts + pair.to(tl.int64)[None, :] * H + rows[:, None], value, mask=valid[None, :])


@triton.jit
def _gate_up_kernel(X, IDs, PTR, GRID, VALUES, Mid, H: tl.constexpr, I: tl.constexpr, K: tl.constexpr, COLUMNS: tl.constexpr, ROW_BYTES, TYPE: tl.constexpr, ROWS: tl.constexpr):
    # One (token, slot) pair and ROWS rows of its gate and up projections, a unit of columns at a time.
    pair = tl.program_id(0)
    rows = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    expert = tl.load(IDs + pair).to(tl.int64)
    address = tl.load(PTR + expert * 3)
    gate = tl.zeros((ROWS,), tl.float32)
    up = tl.zeros((ROWS,), tl.float32)
    if address != 0:
        gate_rows = address.to(tl.pointer_type(tl.uint8)) + rows * ROW_BYTES
        up_rows = tl.load(PTR + expert * 3 + 1).to(tl.pointer_type(tl.uint8)) + rows * ROW_BYTES
        columns = tl.arange(0, COLUMNS)
        activations = X + (pair // K) * H
        for unit in range(H // COLUMNS):
            x = tl.load(activations + unit * COLUMNS + columns).to(tl.float32)
            gate += _input_unit(gate_rows, unit, x, GRID, VALUES, TYPE, ROWS)
            up += _input_unit(up_rows, unit, x, GRID, VALUES, TYPE, ROWS)
    tl.store(Mid + pair * I + rows, gate * tl.sigmoid(gate) * up)


@triton.jit
def _down_kernel(Mid, IDs, Probs, PTR, VALUES, Out, H: tl.constexpr, I: tl.constexpr, K: tl.constexpr, ROW_BYTES, TYPE: tl.constexpr, ROWS: tl.constexpr):
    # ROWS output rows of one token, summed over its experts with a GPU copy.
    token = tl.program_id(0)
    rows = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    c2 = tl.arange(0, 2)
    c4 = tl.arange(0, 4)
    c16 = tl.arange(0, 16)
    total = tl.zeros((ROWS,), tl.float32)
    for slot in tl.static_range(K):
        pair = token * K + slot
        expert = tl.load(IDs + pair).to(tl.int64)
        address = tl.load(PTR + expert * 3 + 2)
        if address != 0:
            row = address.to(tl.pointer_type(tl.uint8)) + rows * ROW_BYTES
            value = tl.zeros((ROWS,), tl.float32)
            if TYPE == 20:  # 32 values per block of 18 bytes: scale, then 16 bytes, low nibbles first
                columns = tl.arange(0, 32)
                for unit in range(I // 32):
                    block = row + unit * 18
                    q = tl.load(block[:, None] + 2 + c16[None, :]).to(tl.int32)
                    w = tl.reshape(tl.load(VALUES + ((q[:, None, :] >> (c2[None, :, None] * 4)) & 15)), (ROWS, 32))
                    x = tl.load(Mid + pair * I + unit * 32 + columns)
                    value += tl.sum(w * x[None, :], 1) * tl.load(block.to(tl.pointer_type(tl.float16))).to(tl.float32)
            else:  # Q2_0: 64 values per block of 18 bytes
                columns = tl.arange(0, 64)
                for unit in range(I // 64):
                    block = row + unit * 18
                    q = tl.load(block[:, None] + 2 + c16[None, :]).to(tl.int32)
                    w = tl.reshape((q[:, :, None] >> (c4[None, None, :] * 2)) & 3, (ROWS, 64)).to(tl.float32) - 1.0
                    x = tl.load(Mid + pair * I + unit * 64 + columns)
                    value += tl.sum(w * x[None, :], 1) * tl.load(block.to(tl.pointer_type(tl.float16))).to(tl.float32)
            total += value * tl.load(Probs + pair).to(tl.float32)
    tl.store(Out + token * H + rows, total)


# Decoding passes, row kernels, with the ideas of Strata (MIT License, https://github.com/Niko1221/Strata): a program is one warp and one row, a lane takes the 4 columns of a code (8 for IQ2_S),
# whose weights it decodes once in registers: a load or a shift serves several columns, each lane keeps its float32 partial sum over the row, reduced once at its end.


@triton.jit
def _bf16_words(word):
    """The two BF16 values of an int32, as float32: the same bits, in the upper half."""
    return tl.cast(word << 16, tl.float32, bitcast=True), tl.cast(word & -65536, tl.float32, bitcast=True)


@triton.jit
def _float_pair(pointer, column):
    """Two float32 from one 8-byte load, at an even column of an 8-byte aligned row."""
    word = tl.load((pointer + column).to(tl.pointer_type(tl.int64)))
    return tl.cast(word.to(tl.int32), tl.float32, bitcast=True), tl.cast((word >> 32).to(tl.int32), tl.float32, bitcast=True)


@triton.jit
def _four_activations(X, column):
    if X.dtype.element_ty == tl.bfloat16:
        word = tl.load((X + column).to(tl.pointer_type(tl.int64)))
        return _bf16_words(word.to(tl.int32)) + _bf16_words((word >> 32).to(tl.int32))
    else:
        return _float_pair(X, column) + _float_pair(X, column + 2)


@triton.jit
def _two_activations(X, column):
    if X.dtype.element_ty == tl.bfloat16:
        return _bf16_words(tl.load((X + column).to(tl.pointer_type(tl.int32))))
    else:
        return _float_pair(X, column)


@triton.jit
def _step_activations(X, half, lanes, TYPE: tl.constexpr):
    """The activations of the lanes of half a step: 8 per lane for IQ2_S (a block), else 4, consecutive or, for IQ4 rows, two pairs 16 columns apart."""
    if TYPE == 22:
        return _four_activations(X + lanes * 8, 0) + _four_activations(X + lanes * 8, 4)
    elif TYPE == 23 or TYPE == 20:
        pointer = X + (lanes >> 3) * 32 + (lanes & 7) * 2
        return _two_activations(pointer, half * 128) + _two_activations(pointer, half * 128 + 16)
    else:
        return _four_activations(X + lanes * 4, half * 128)


@triton.jit
def _signed_grid(grid, signs):
    """The four magnitudes of a grid word as float32, negated by 4 sign bits (no magnitude is 0: no carry between bytes)."""
    mask = (signs * 0x00204081) & 0x01010101
    packed = (grid ^ (mask * 255)) + mask
    return ((packed << 24) >> 24).to(tl.float32), ((packed << 16) >> 24).to(tl.float32), ((packed << 8) >> 24).to(tl.float32), (packed >> 24).to(tl.float32)


@triton.jit
def _nibble_weights(NIBBLES, word):
    """The values over 128 of the four nibbles of two IQ4 bytes, in the order of their columns: column c and c + 1, then c + 16 and c + 17."""
    return tl.load(NIBBLES + (word & 15)), tl.load(NIBBLES + ((word >> 8) & 15)), tl.load(NIBBLES + ((word >> 4) & 15)), tl.load(NIBBLES + (word >> 12))


@triton.jit
def _block_scale(block):
    return tl.load(block.to(tl.pointer_type(tl.float16))).to(tl.float32)


@triton.jit
def _step_weights(block, half, scale, lanes, GRID, NIBBLES, TYPE: tl.constexpr):
    """The weights of the lanes, 4 per lane (8 for IQ2_S), then their common scale. block: the start of a step, half: its half, scale: the scale of its block when it has one."""
    if TYPE == 21:  # 110 bytes: 2 scale, 64 codes, 8 ninth bits, 32 signs, 4 unit scales
        value = tl.load(block + lanes + 2 + half * 32).to(tl.int32) | (((tl.load(block + (lanes >> 3) + 66 + half * 4).to(tl.int32) >> (lanes & 7)) & 1) << 8)
        signs = (tl.load(block + (lanes >> 1) + 74 + half * 16).to(tl.int32) >> ((lanes & 1) * 4)) & 15
        unit = (tl.load(block + (lanes >> 4) + 106 + half * 2).to(tl.int32) >> (((lanes >> 3) & 1) * 4)) & 15
        return _signed_grid(tl.load(GRID + value), signs) + ((unit * 2 + 1).to(tl.float32) * scale,)
    elif TYPE == 18:  # 98 bytes: 2 scale, 64 codes, 32: a word per unit (4 x 7 sign bits, the eighth by parity, 4 scale bits at the top)
        words = (block + (lanes >> 3) * 4 + 66 + half * 16).to(tl.pointer_type(tl.uint16))
        word = tl.load(words).to(tl.int32) | (tl.load(words + 1).to(tl.int32) << 16)
        group = (word >> (((lanes & 7) >> 1) * 7)) & 127
        parity = group ^ (group >> 4)
        parity = parity ^ (parity >> 2)
        parity = (parity ^ (parity >> 1)) & 1
        signs = ((group | (parity << 7)) >> ((lanes & 1) * 4)) & 15
        return _signed_grid(tl.load(GRID + tl.load(block + lanes + 2 + half * 32).to(tl.int32)), signs) + ((((word >> 28) & 15) * 2 + 1).to(tl.float32) * scale * 0.25,)
    elif TYPE == 22:  # 82 bytes: 2 scale, 32 codes, 32 signs, 8: 2 high bits per code, 8: scale nibbles per 2 codes
        value = tl.load(block + lanes + 2).to(tl.int32) | (((tl.load(block + (lanes >> 2) + 66).to(tl.int32) >> ((lanes & 3) * 2)) & 3) << 8)
        signs = tl.load(block + lanes + 34).to(tl.int32)
        grid = tl.load(GRID + value)
        unit = (tl.load(block + (lanes >> 2) + 74).to(tl.int32) >> (((lanes & 3) >> 1) * 4)) & 15
        return _signed_grid(grid.to(tl.int32), signs & 15) + _signed_grid((grid >> 32).to(tl.int32), signs >> 4) + ((unit * 2 + 1).to(tl.float32) * scale * 0.125,)
    elif TYPE == 23:  # 136 bytes: 2 scale, 2 high scale bits, 4 low scale bits, 16 bytes per unit, low nibbles first
        unit = half * 4 + (lanes >> 3)
        word = tl.load((block + 8 + unit * 16 + (lanes & 7) * 2).to(tl.pointer_type(tl.uint16))).to(tl.int32)
        low = (tl.load(block + 4 + (unit >> 1)).to(tl.int32) >> ((unit & 1) * 4)) & 15
        high = (tl.load((block + 2).to(tl.pointer_type(tl.uint16))).to(tl.int32) >> (unit * 2)) & 3
        return _nibble_weights(NIBBLES, word) + ((((low | (high << 4)) - 32) * 128).to(tl.float32) * scale,)
    elif TYPE == 20:  # IQ4_NL, 18 bytes per 32 values: scale, 16 bytes, low nibbles first; a step of 4 blocks
        lane_block = block + (lanes >> 3) * 18
        word = tl.load((lane_block + 2 + (lanes & 7) * 2).to(tl.pointer_type(tl.uint16))).to(tl.int32)
        return _nibble_weights(NIBBLES, word) + (_block_scale(lane_block),)
    else:  # Q2_0, 18 bytes per 64 values: scale, 16 bytes of 4 codes, value = code - 1; a step of 2 blocks
        lane_block = block + (lanes >> 4) * 18
        q = tl.load(lane_block + 2 + (lanes & 15)).to(tl.int32)
        return (q & 3).to(tl.float32) - 1.0, ((q >> 2) & 3).to(tl.float32) - 1.0, ((q >> 4) & 3).to(tl.float32) - 1.0, (q >> 6).to(tl.float32) - 1.0, _block_scale(lane_block)


@triton.jit
def _step_dot(w, x, TYPE: tl.constexpr):
    if TYPE == 22:
        return (w[0] * x[0] + w[1] * x[1] + w[2] * x[2] + w[3] * x[3] + w[4] * x[4] + w[5] * x[5] + w[6] * x[6] + w[7] * x[7]) * w[8]
    else:
        return (w[0] * x[0] + w[1] * x[1] + w[2] * x[2] + w[3] * x[3]) * w[4]


@triton.jit
def _store_swiglu(Mid, pair, row, I: tl.constexpr, gate, up):
    gate, up = tl.sum(gate), tl.sum(up)
    tl.store(Mid + pair * I + row, tl.div_rn(gate, 1.0 + libdevice.exp(-gate)) * up)


@triton.jit(do_not_specialize_on_alignment=("X", "IDs", "Mid"))
def _gate_up_row(X, IDs, PTR, GRID, NIBBLES, Mid, H: tl.constexpr, I: tl.constexpr, K: tl.constexpr, TYPE: tl.constexpr, COLUMNS: tl.constexpr, WIDTH: tl.constexpr, HALVES: tl.constexpr):
    # One row of the gate and up projections of one (token, slot) pair.
    pair = tl.program_id(0)
    row = tl.program_id(1)
    expert = tl.load(IDs + pair).to(tl.int64)
    address = tl.load(PTR + expert * 3)
    if address != 0:
        gate_row = address.to(tl.pointer_type(tl.uint8)) + row * (H // COLUMNS * WIDTH)
        up_row = tl.load(PTR + expert * 3 + 1).to(tl.pointer_type(tl.uint8)) + row * (H // COLUMNS * WIDTH)
        gate = tl.zeros((32,), tl.float32)
        up = tl.zeros((32,), tl.float32)
        lanes = tl.arange(0, 32)
        activations = X + (pair // K) * H
        for step in range(H // COLUMNS):
            gate_block, up_block = gate_row + step * WIDTH, up_row + step * WIDTH
            gate_scale, up_scale = _block_scale(gate_block), _block_scale(up_block)
            for half in tl.static_range(HALVES):
                x = _step_activations(activations + step * COLUMNS, half, lanes, TYPE)
                gate += _step_dot(_step_weights(gate_block, half, gate_scale, lanes, GRID, NIBBLES, TYPE), x, TYPE)
                up += _step_dot(_step_weights(up_block, half, up_scale, lanes, GRID, NIBBLES, TYPE), x, TYPE)
        _store_swiglu(Mid, pair, row, I, gate, up)


@triton.jit(do_not_specialize_on_alignment=("Mid", "IDs", "Probs", "Out"))
def _down_row(Mid, IDs, Probs, PTR, NIBBLES, Out, H: tl.constexpr, I: tl.constexpr, K: tl.constexpr, TYPE: tl.constexpr, WIDTH: tl.constexpr):
    # One output row of one token, summed over its experts with a GPU copy.
    token = tl.program_id(0)
    row = tl.program_id(1)
    total = tl.zeros((32,), tl.float32)
    lanes = tl.arange(0, 32)
    for slot in range(K):
        pair = token * K + slot
        expert = tl.load(IDs + pair).to(tl.int64)
        address = tl.load(PTR + expert * 3 + 2)
        if address != 0:
            start = address.to(tl.pointer_type(tl.uint8)) + row * (I // 128 * WIDTH)
            value = tl.zeros((32,), tl.float32)
            mid = Mid + pair * I
            for step in range(I // 128):
                value += _step_dot(_step_weights(start + step * WIDTH, 0, None, lanes, None, NIBBLES, TYPE), _step_activations(mid + step * 128, 0, lanes, TYPE), TYPE)
            total += value * (tl.load(Probs + pair).to(tl.float32) * (128.0 if TYPE == 20 else 1.0))
    tl.store(Out + token * H + row, tl.sum(total))


# The experts of a window: tokens that read the same expert decode its rows once. Each pair's leader is found in the graph (any routing); a group holds at most 4 pairs, packed as pair + 1 in the bytes of an int64, 0 for pairs that lead nothing.


@triton.jit(do_not_specialize=("T",), do_not_specialize_on_alignment=("IDs", "PTR", "Packed"))
def _group_leaders(IDs, PTR, Packed, T, K: tl.constexpr):
    pair = tl.program_id(0)
    expert = tl.load(IDs + pair).to(tl.int64)
    tokens = tl.arange(0, 8)[:, None]
    slots = tl.arange(0, 16)[None, :]
    ids = tl.load(IDs + tokens * K + slots, mask=(tokens < T) & (slots < K), other=-1).to(tl.int64)
    slot = tl.sum(tl.where(ids == expert, slots + 1, 0), axis=1)  # of each token: 0 when it does not read this expert
    found = slot > 0
    order = tl.cumsum(found.to(tl.int64), axis=0) - 1
    token = tl.arange(0, 8)
    own = tl.sum(tl.where(token == pair // K, order, 0))
    relative = order - own
    members = tl.sum(tl.where(found & (relative >= 0) & (relative < 4), (token * K + slot).to(tl.int64) << (relative * 8), 0))
    tl.store(Packed + pair, tl.where(((own & 3) == 0) & (tl.load(PTR + expert * 3) != 0), members, 0))


@triton.jit
def _members_row(X, Mid, gate_row, up_row, packed, row, GRID, NIBBLES, H: tl.constexpr, I: tl.constexpr, K: tl.constexpr, TYPE: tl.constexpr, COLUMNS: tl.constexpr, WIDTH: tl.constexpr, HALVES: tl.constexpr, M: tl.constexpr):
    """One row of the gate and up projections of an expert, decoded once, for its M tokens at most."""
    members = tl.arange(0, M)
    pairs = ((packed >> (members.to(tl.int64) * 8)) & 255).to(tl.int32) - 1
    present = pairs >= 0
    activations = X + (tl.where(present, pairs, 0) // K)[:, None] * H
    lanes = tl.arange(0, 32)
    wide = lanes[None, :]
    gate = tl.zeros((M, 32), tl.float32)
    up = tl.zeros((M, 32), tl.float32)
    for step in range(H // COLUMNS):
        gate_block, up_block = gate_row + step * WIDTH, up_row + step * WIDTH
        gate_scale, up_scale = _block_scale(gate_block), _block_scale(up_block)
        for half in tl.static_range(HALVES):
            x = _step_activations(activations + step * COLUMNS, half, wide, TYPE)
            gate += _step_dot(_step_weights(gate_block, half, gate_scale, lanes, GRID, NIBBLES, TYPE), x, TYPE)
            up += _step_dot(_step_weights(up_block, half, up_scale, lanes, GRID, NIBBLES, TYPE), x, TYPE)
    gate, up = tl.sum(gate, axis=1), tl.sum(up, axis=1)
    tl.store(Mid + pairs * I + row, tl.div_rn(gate, 1.0 + libdevice.exp(-gate)) * up, mask=present)


@triton.jit(do_not_specialize=("N", "STRIDE"), do_not_specialize_on_alignment=("X", "IDs", "Packed", "Mid"))
def _gate_up_group(X, IDs, Packed, PTR, GRID, NIBBLES, Mid, N, STRIDE, H: tl.constexpr, I: tl.constexpr, K: tl.constexpr, TYPE: tl.constexpr, COLUMNS: tl.constexpr, WIDTH: tl.constexpr, HALVES: tl.constexpr):
    row = tl.program_id(1)
    for leader in range(tl.program_id(0), N, STRIDE):
        packed = tl.load(Packed + leader)
        if packed != 0:
            expert = tl.load(IDs + leader).to(tl.int64)
            gate_row = tl.load(PTR + expert * 3).to(tl.pointer_type(tl.uint8)) + row * (H // COLUMNS * WIDTH)
            up_row = tl.load(PTR + expert * 3 + 1).to(tl.pointer_type(tl.uint8)) + row * (H // COLUMNS * WIDTH)
            if packed < 256:
                _members_row(X, Mid, gate_row, up_row, packed, row, GRID, NIBBLES, H, I, K, TYPE, COLUMNS, WIDTH, HALVES, 1)
            elif packed < 65536:
                _members_row(X, Mid, gate_row, up_row, packed, row, GRID, NIBBLES, H, I, K, TYPE, COLUMNS, WIDTH, HALVES, 2)
            else:
                _members_row(X, Mid, gate_row, up_row, packed, row, GRID, NIBBLES, H, I, K, TYPE, COLUMNS, WIDTH, HALVES, 4)


_tables = {}
_TILE_ROWS, _DOWN_ROWS, _WARPS = 8, 8, 8  # the best of real decoding on a 5090: 4 warps looks 15-20% faster per kernel in isolation and decodes 7% slower (a table of launch configurations per GPU is still to do)
import importlib  # noqa: E402
import importlib.util  # noqa: E402

from shared.kernels import kernel_policy  # noqa: E402

_GROUP_PAIRS = 16  # pairs per program of the grouped kernels
_GROUP_ROWS = 16
GROUPED_PAIRS = 256  # from this many (token, slot) pairs, the tokens of each expert are grouped
_WIDTHS = {IQ2_S: 82, IQ3_XXS: 98, IQ3_S: 110, IQ4_XS: 136}
ROW_KERNELS = "experts" not in os.environ.get("WGP_FN_DISABLE", "").split(",")  # the old kernels, for comparisons, until the switches are removed
GROUPS = "groups" in os.environ.get("WGP_FN_ENABLE", "").split(",")  # each expert of a window read once, until measured
WINDOW_KERNELS = "native_window" not in os.environ.get("WGP_FN_DISABLE", "").split(",")  # until the switches are removed
GROUP_TOKENS, GROUP_STRIDE = 8, 8  # the most tokens of a window for it; programs per row, each taking one pair in so many
_STEPS = {IQ3_S: (256, 110, 2), IQ3_XXS: (256, 98, 2), IQ2_S: (256, 82, 1), IQ4_XS: (256, 136, 2), Q2_0: (128, 36, 1), IQ4_NL: (128, 72, 1)}  # the steps of the row kernels: columns, bytes, halves


def prepare(device):
    """The tables of the kernels on device (kept for the life of the process, a few KB), made outside graph captures."""
    tables = _tables.get(device)
    if tables is None:
        import gguf
        from mmgp import allocator
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("The GGUF expert tables must be prepared before graph capture")
        packed = {}
        for type, quant, size in ((IQ3_S, gguf.quants.IQ3_S, torch.int32), (IQ3_XXS, gguf.quants.IQ3_XXS, torch.int32), (IQ2_S, gguf.quants.IQ2_S, torch.int64)):
            quant.init_grid()
            grid = torch.from_numpy(quant.grid.copy()).reshape(quant.grid_shape).to(torch.int64)
            word = (grid << (torch.arange(grid.shape[1], device=grid.device) * 8)).sum(1)
            packed[type] = word.to(size)
        with allocator.normal_allocations():
            tables = {type: table.to(device) for type, table in packed.items()}
            tables[IQ4_NL] = torch.tensor(gguf.quants.IQ4_NL.kvalues, dtype=torch.float32, device=device)
            tables["codes"] = tables[IQ4_NL].to(torch.int8)  # the same integers
            tables["nibbles"] = tables[IQ4_NL] / 128  # below 1: the partial sums of huge activations overflow no sooner than the weights times the activations (the kernels multiply by 128 after)
        _tables[device] = tables
    return tables


@cache
def _needs_fp32_inputs(device):
    return torch.version.hip is None and torch.cuda.get_device_capability(device) < (8, 0)


def _groups(ids, pointers):
    """The pairs routed to each expert having a GPU copy, by groups: int32 [groups, _GROUP_PAIRS] of pair indices (-1 past the end), int32 [groups] of experts."""
    flat = ids.reshape(-1)
    flat = flat.to(torch.int64)
    order = torch.nonzero(pointers[flat, 0] != 0).flatten()
    if not order.numel():
        return None, None
    order = order[torch.argsort(flat[order], stable=True)]
    experts = flat[order]
    first = torch.ones_like(experts, dtype=torch.bool)
    first[1:] = experts[1:] != experts[:-1]
    start = torch.cummax(torch.arange(experts.numel(), device=ids.device) * first, 0).values
    within = torch.arange(experts.numel(), device=ids.device) - start
    new_group = first | (within % _GROUP_PAIRS == 0)
    group = torch.cumsum(new_group, 0) - 1
    pairs = torch.full((int(group[-1]) + 1, _GROUP_PAIRS), -1, dtype=torch.int32, device=ids.device)
    pairs[group, within % _GROUP_PAIRS] = order.to(torch.int32)
    return pairs, experts[new_group].to(torch.int32)


def forward_groups(x, ids, probabilities, pointers, intermediate, types):
    """As forward(), for many pairs: float32 [tokens, hidden]."""
    tokens, hidden = x.shape
    topk = ids.shape[-1]
    gate, up, down = types
    if gate != up:
        raise NotImplementedError("The GPU kernels take one format for the gate and up rows")
    if _needs_fp32_inputs(x.device):
        x = x.float()
    tables = prepare(x.device)
    pairs, experts = _groups(ids, pointers)
    parts = torch.zeros(tokens * topk, hidden, dtype=torch.float32, device=x.device)
    if pairs is not None:
        count = pairs.shape[0]
        mid = torch.empty(tokens * topk, intermediate, dtype=torch.float32, device=x.device)
        columns, row_bytes = (64, hidden // 64 * 18) if gate == Q2_0 else (32, hidden // 256 * _WIDTHS[gate])
        _gate_up_groups[(count, intermediate // _GROUP_ROWS)](x, pairs, experts, pointers, tables.get(gate, tables[IQ3_S]), tables[IQ4_NL], mid, hidden, intermediate, topk, columns, row_bytes, gate, _GROUP_ROWS, _GROUP_PAIRS, num_warps=8)
        down_bytes = intermediate // 32 * 18 if down == IQ4_NL else intermediate // 64 * 18
        _down_groups[(count, hidden // _GROUP_ROWS)](mid, pairs, experts, pointers, tables[IQ4_NL], parts, hidden, intermediate, down_bytes, down, _GROUP_ROWS, _GROUP_PAIRS, num_warps=8)
        mid = pairs = experts = None
    return (parts.view(tokens, topk, hidden) * probabilities.float()[..., None]).sum(1)


def gate_up_groups(x, ids, pointers, mid, hidden, intermediate, gate, tables):
    """The gate and up rows of a window into mid, as _gate_up_row, each expert decoded once for the tokens that read it."""
    tokens, topk = ids.shape
    packed = torch.empty(tokens * topk, dtype=torch.int64, device=x.device)
    _group_leaders[(tokens * topk,)](ids, pointers, packed, tokens, topk, num_warps=1)
    _gate_up_group[(GROUP_STRIDE, intermediate)](x, ids, packed, pointers, tables.get(gate, tables[IQ3_S]), tables["nibbles"], mid, tokens * topk, GROUP_STRIDE, hidden, intermediate, topk, gate, *_STEPS[gate], num_warps=1)


@cache
def _window_kernels(device, gate, down):
    # the decoding kernels of WGPKernels, when the installed version has them for these formats
    native = importlib.import_module("wgpkernels.experts") if importlib.util.find_spec("wgpkernels.experts") else None
    return native if native is not None and hasattr(native, "window_supported") and native.window_supported(device, gate, down) else None


def forward(x, ids, probabilities, pointers, intermediate, types, out=None, mid=None):
    """float32 [tokens, hidden] sum of the selected experts having GPU copies (pointers: int64 [experts, 3] table of the gate, up and down rows, zero for the others); out and the
    float32 [tokens * topk, intermediate] scratch mid may be preallocated. types: the ggml types of the three projections."""
    tokens, hidden = x.shape
    topk = ids.shape[-1]
    if out is None and tokens * topk >= GROUPED_PAIRS:
        return forward_groups(x, ids, probabilities, pointers, intermediate, types)
    gate, up, down = types
    if gate != up:
        raise NotImplementedError("The GPU kernels take one format for the gate and up rows")
    if _needs_fp32_inputs(x.device):
        x, probabilities = x.float(), probabilities.float()
    mid = torch.empty(tokens * topk, intermediate, dtype=torch.float32, device=x.device) if mid is None else mid
    out = torch.empty(tokens, hidden, dtype=torch.float32, device=x.device) if out is None else out
    # each expert of the window read once (their activations in int8: an error of about 4e-5 per layer, against 1.6e-3 for the BF16 output): decoding's largest GPU cost, a third less at 5 tokens
    native = _window_kernels(x.device, gate, down) if WINDOW_KERNELS and gate == up and kernel_policy.allow_approximate() and hidden % 256 == 0 and intermediate % 128 == 0 and x.dtype in (torch.bfloat16, torch.float32) else None
    if native is not None and tokens <= native.WINDOW_MAX_TOKENS and tokens * topk <= native.WINDOW_MAX_PAIRS:
        native.forward_window(x, ids, probabilities, pointers, intermediate, types, out, mid.view(-1).view(torch.uint8))  # mid is larger than its scratch
        return out
    tables = prepare(x.device)
    columns, row_bytes = (64, hidden // 64 * 18) if gate == Q2_0 else (32, hidden // 256 * _WIDTHS[gate])
    if ROW_KERNELS and hidden % 256 == 0 and intermediate % 128 == 0:
        if not (x.is_cuda and ids.is_cuda and probabilities.is_cuda and pointers.is_cuda and mid.is_cuda and out.is_cuda) or (x.data_ptr() | mid.data_ptr()) & 7:
            raise ValueError("The GGUF row kernels read CUDA tensors, the activations and the scratch aligned on 8 bytes")
        if GROUPS and 2 <= tokens <= GROUP_TOKENS and topk <= 16:
            gate_up_groups(x, ids, pointers, mid, hidden, intermediate, gate, tables)
        else:
            _gate_up_row[(tokens * topk, intermediate)](x, ids, pointers, tables.get(gate, tables[IQ3_S]), tables["nibbles"], mid, hidden, intermediate, topk, gate, *_STEPS[gate], num_warps=1)
        _down_row[(tokens, hidden)](mid, ids, probabilities, pointers, tables["nibbles"], out, hidden, intermediate, topk, down, _STEPS[down][1], num_warps=1)
        return out
    _gate_up_kernel[(tokens * topk, intermediate // _TILE_ROWS)](x, ids, pointers, tables.get(gate, tables[IQ3_S]), tables[IQ4_NL], mid, hidden, intermediate, topk, columns, row_bytes, gate, _TILE_ROWS, num_warps=_WARPS)
    down_bytes = intermediate // 32 * 18 if down == IQ4_NL else intermediate // 64 * 18
    _down_kernel[(tokens, hidden // _DOWN_ROWS)](mid, ids, probabilities, pointers, tables[IQ4_NL], out, hidden, intermediate, topk, down_bytes, down, _DOWN_ROWS, num_warps=_WARPS)
    return out


from shared.kernels.triton_compilation_log import install_triton_compilation_logger  # noqa: E402

install_triton_compilation_logger()
