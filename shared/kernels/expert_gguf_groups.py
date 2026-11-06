# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
# With ideas of Strata (MIT License, https://github.com/Niko1221/Strata) for reading prompts: routed experts grouped by expert, weights left quantized.
"""Routed experts of a pass of many tokens (prompt reading) on tensor cores, the weights of the GPU copies staying in their GGUF storage: the (token, slot) pairs routed to an expert form groups, and each block of a weight row is
an exact small integer in BF16 times its float32 scale, so that the dot product of a block with BF16 activations (exact in BF16) accumulates in float32 and takes its scale once. The output projection reads float32 activations: on FMA units under strict
precision, as BF16 pairs on tensor cores otherwise. Against float64, on real weights: 1.2e-7 strict (the PyTorch path: 1.5-2.2e-7), 2e-6 otherwise, whatever the expert format. The block layouts are ggml's (MIT License)."""
import importlib
import importlib.util
import os
from functools import cache

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from shared.kernels import kernel_policy
from shared.kernels.expert_gguf_triton import _tables as expert_tables, IQ2_S, IQ3_S, IQ3_XXS, IQ4_NL, IQ4_XS, Q2_0, prepare

ROW_BYTES = {IQ2_S: 82, IQ3_XXS: 98, IQ3_S: 110, IQ4_XS: 136}  # per 256 columns
# Launch configurations of the best measured on a 5090, each kernel alone at 8K tokens: rows per program, warps. The gate/up kernel takes about 0.1 us per pair, IQ2_S 0.15 (its dots hold 16 columns), and the down kernel 0.085 on BF16 pairs, 0.16-0.2 on float32.
GATE_UP_TILES = {IQ2_S: (32, 8), IQ3_XXS: (64, 4), IQ3_S: (64, 4), IQ4_XS: (64, 4), Q2_0: (64, 4)}
DOWN_TILE = (64, 4)
LONG_GROUPS_TOKENS = 2048  # from this many tokens, the experts of a pass read 64 pairs at a time, 32 below: nearly full tiles either way
COLUMNS = {IQ2_S: 16, IQ3_XXS: 32, IQ3_S: 32, IQ4_XS: 32, Q2_0: 64}  # per block of a same scale, gate and up rows
MID_FP32, MID_BF16, MID_CODES = 0, 1, 2  # how the SwiGLU of a pair waits for the output projection
NATIVE_PASS = "native" not in os.environ.get("WGP_FN_DISABLE", "").split(",")  # until the switches are removed


@triton.jit
def _unit(row, unit, GRID, VALUES, TYPE: tl.constexpr, ROWS: tl.constexpr, CODES: tl.constexpr):
    """Unit (a run of columns of a same scale: COLUMNS[TYPE]) of ROWS rows (pointers to their blocks): integer weights [ROWS, columns] in BF16, or int8 with CODES, and their scale [ROWS] in float32. The blocks are 2-aligned: 16-bit loads.
    With CODES a Q2_0 unit is half a block: the scale of an activation block spans 32 columns."""
    c2 = tl.arange(0, 2)
    c4 = tl.arange(0, 4)
    c8 = tl.arange(0, 8)
    c16 = tl.arange(0, 16)
    if TYPE == 21:  # 2: scale, 64: codes, 8: 9th bits, 32: signs, 4: scales of the units; a code gives 4 magnitudes
        block, s = row + (unit // 8) * 110, unit % 8
        words = tl.load((block[:, None] + 2 + s * 8 + c4[None, :] * 2).to(tl.pointer_type(tl.uint16))).to(tl.int32)
        codes = tl.reshape((words[:, :, None] >> (c2[None, None, :] * 8)) & 255, (ROWS, 8)) | (((tl.load(block + 66 + s).to(tl.int32)[:, None] >> c8[None, :]) & 1) << 8)
        words = tl.load((block[:, None] + 74 + s * 4 + c2[None, :] * 2).to(tl.pointer_type(tl.uint16))).to(tl.int32)
        signs = (words[:, :, None] >> c16[None, None, :]) & 1
        magnitudes = (tl.load(GRID + codes)[:, :, None] >> (c4[None, None, :] * 8)) & 255
        w = tl.reshape(magnitudes, (ROWS, 32)) * (1 - 2 * tl.reshape(signs, (ROWS, 32)))
        scale = tl.load(block.to(tl.pointer_type(tl.float16))).to(tl.float32) * (1 + 2 * ((tl.load(block + 106 + s // 2).to(tl.int32) >> (4 * (s % 2))) & 15)).to(tl.float32)
    elif TYPE == 18:  # 2: scale, 64: codes, 32: a word per unit (4 scale bits, 4 x 7 sign bits, the eighth by parity); a code gives 4 magnitudes
        block, s = row + (unit // 8) * 98, unit % 8
        words = tl.load((block[:, None] + 2 + s * 8 + c4[None, :] * 2).to(tl.pointer_type(tl.uint16))).to(tl.int32)
        codes = tl.reshape((words[:, :, None] >> (c2[None, None, :] * 8)) & 255, (ROWS, 8))
        word = tl.load((block + 66 + s * 4).to(tl.pointer_type(tl.uint16))).to(tl.int32) | (tl.load((block + 68 + s * 4).to(tl.pointer_type(tl.uint16))).to(tl.int32) << 16)
        magnitudes = (tl.load(GRID + codes)[:, :, None] >> (c4[None, None, :] * 8)) & 255
        groups = (word[:, None] >> (c4[None, :] * 7)) & 127
        parity = groups ^ (groups >> 4)  # of the 7 bits: the eighth
        parity = parity ^ (parity >> 2)
        parity = (parity ^ (parity >> 1)) & 1
        bits = ((groups | (parity << 7))[:, :, None] >> c8[None, None, :]) & 1
        w = tl.reshape(magnitudes, (ROWS, 32)) * (1 - 2 * tl.reshape(bits, (ROWS, 32)))
        scale = tl.load(block.to(tl.pointer_type(tl.float16))).to(tl.float32) * (0.5 + ((word >> 28) & 15).to(tl.float32)) * 0.5
    elif TYPE == 22:  # 2: scale, 32: codes, 32: signs, 8: 2 high bits per code, 8: a scale per 16 values; a code gives 8 magnitudes. A unit of 16 columns: half of a sub-block
        block, s, half = row + (unit // 16) * 82, (unit // 2) % 8, unit % 2
        high = ((tl.load(block + 66 + s).to(tl.int32)[:, None] >> ((half * 2 + c2[None, :]) * 2)) & 3) << 8
        words = tl.load((block + 2 + s * 4 + half * 2).to(tl.pointer_type(tl.uint16))).to(tl.int32)
        codes = ((words[:, None] >> (c2[None, :] * 8)) & 255) | high
        signs = (tl.load((block + 34 + s * 4 + half * 2).to(tl.pointer_type(tl.uint16))).to(tl.int32)[:, None] >> c16[None, :]) & 1
        magnitudes = (tl.load(GRID + codes)[:, :, None] >> (c8[None, None, :] * 8)) & 255
        w = tl.reshape(magnitudes, (ROWS, 16)).to(tl.int32) * (1 - 2 * signs)
        scale = tl.load(block.to(tl.pointer_type(tl.float16))).to(tl.float32) * (((tl.load(block + 74 + s).to(tl.int32) >> (half * 4)) & 15).to(tl.float32) + 0.5) * 0.25
    elif TYPE == 23:  # 2: scale, 2: high scale bits, 4: low scale bits, 128: values, 16 bytes per unit, low nibbles first
        block, s = row + (unit // 8) * 136, unit % 8
        q = tl.load(block[:, None] + 8 + s * 16 + c16[None, :]).to(tl.int32)
        w = tl.reshape(tl.load(VALUES + ((q[:, None, :] >> (c2[None, :, None] * 4)) & 15)), (ROWS, 32))
        low = (tl.load(block + 4 + s // 2).to(tl.int32) >> (4 * (s % 2))) & 15
        high = (tl.load((block + 2).to(tl.pointer_type(tl.uint16))).to(tl.int32) >> (2 * s)) & 3
        scale = tl.load(block.to(tl.pointer_type(tl.float16))).to(tl.float32) * ((low | (high << 4)) - 32).to(tl.float32)
    elif CODES:  # Q2_0, half a block
        block = row + (unit // 2) * 18
        q = tl.load(block[:, None] + 2 + (unit % 2) * 8 + c8[None, :]).to(tl.int32)
        w = tl.reshape((q[:, :, None] >> (c4[None, None, :] * 2)) & 3, (ROWS, 32)) - 1
        scale = tl.load(block.to(tl.pointer_type(tl.float16))).to(tl.float32)
    else:  # Q2_0, 64 columns: 2: scale, 16: four codes per byte, value code - 1
        block = row + unit * 18
        q = tl.load(block[:, None] + 2 + c16[None, :]).to(tl.int32)
        w = tl.reshape((q[:, :, None] >> (c4[None, None, :] * 2)) & 3, (ROWS, 64)) - 1
        scale = tl.load(block.to(tl.pointer_type(tl.float16))).to(tl.float32)
    if CODES:
        w = w.to(tl.int8)
    else:
        w = w.to(tl.float32).to(tl.bfloat16)
    return w, scale


@triton.jit
def _down_unit(row, unit, VALUES, TYPE: tl.constexpr, ROWS: tl.constexpr):
    """Unit of ROWS output rows: float32 integer weights [ROWS, columns] and their float32 scale [ROWS]."""
    block = row + unit * 18
    q = tl.load(block[:, None] + 2 + tl.arange(0, 16)[None, :]).to(tl.int32)
    if TYPE == 20:  # 32 values per block of 18 bytes: scale, then 16 bytes, low nibbles first
        w = tl.reshape(tl.load(VALUES + ((q[:, None, :] >> (tl.arange(0, 2)[None, :, None] * 4)) & 15)), (ROWS, 32))
    else:  # Q2_0: 64 values per block of 18 bytes
        w = (tl.reshape((q[:, :, None] >> (tl.arange(0, 4)[None, None, :] * 2)) & 3, (ROWS, 64)).to(tl.float32) - 1.0)
    return w, tl.load(block.to(tl.pointer_type(tl.float16))).to(tl.float32)


@triton.jit
def _add(total, carry, value):
    """total + value and what its rounding lost, kept in carry (two-sum)."""
    new = total + value
    part = new - total
    return new, carry + (total - (new - part)) + (value - part)


@triton.jit
def _store_mid(gate, up, Mid, MidScales, group, within, block, rows, I: tl.constexpr, ROWS: tl.constexpr, GROUP: tl.constexpr, MID: tl.constexpr):
    """The SwiGLU to Mid, by place in the groups: float32 (MID_FP32), BF16 (MID_BF16), or int8 codes with a float32 scale per 32 of its values and per pair (MID_CODES)."""
    activated = tl.div_rn(gate, 1.0 + libdevice.exp(-gate)) * up
    place = group.to(tl.int64) * GROUP + within
    if MID == 2:
        blocks = tl.reshape(activated, (ROWS // 32, 32, GROUP))
        scale = tl.maximum(tl.max(tl.abs(blocks), 1) / 127.0, 1.0e-30)
        codes = tl.reshape(tl.floor(blocks / scale[:, None, :] + 0.5), (ROWS, GROUP)).to(tl.int8)
        tl.store(Mid + place[:, None] * I + rows[None, :], tl.trans(codes))
        tl.store(MidScales + place[:, None] * (I // 32) + block * (ROWS // 32) + tl.arange(0, ROWS // 32)[None, :], tl.trans(scale))
    elif MID == 1:
        tl.store(Mid + place[:, None] * I + rows[None, :], tl.trans(activated).to(tl.bfloat16))
    else:
        tl.store(Mid + place[:, None] * I + rows[None, :], tl.trans(activated))


@triton.jit(do_not_specialize=("FIRST",))
def _gate_up(X, Pairs, Experts, PTR, GRID, VALUES, Mid, MidScales, FIRST, H: tl.constexpr, I: tl.constexpr, K: tl.constexpr, UNIT: tl.constexpr, ROW_BYTES, TYPE: tl.constexpr, ROWS: tl.constexpr, GROUP: tl.constexpr, EXACT: tl.constexpr, MID: tl.constexpr):
    # ROWS rows of the gate and up projections of one expert for up to GROUP pairs routed to it (-1: none), a unit of columns at a time. Their SwiGLU goes to Mid, by place in the groups. EXACT: what the sums of the units round is kept.
    group = FIRST + tl.program_id(0)
    rows = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    within = tl.arange(0, GROUP)
    pair = tl.load(Pairs + group * GROUP + within)
    expert = tl.load(Experts + group).to(tl.int64)
    gate_rows = tl.load(PTR + expert * 3).to(tl.pointer_type(tl.uint8)) + rows * ROW_BYTES
    up_rows = tl.load(PTR + expert * 3 + 1).to(tl.pointer_type(tl.uint8)) + rows * ROW_BYTES
    activations = X + (tl.maximum(pair, 0) // K).to(tl.int64)[:, None] * H + tl.arange(0, UNIT)[None, :]  # what the places past the end compute is not read
    gate = tl.zeros((ROWS, GROUP), tl.float32)
    up = tl.zeros((ROWS, GROUP), tl.float32)
    if EXACT:
        gate_lost = tl.zeros((ROWS, GROUP), tl.float32)
        up_lost = tl.zeros((ROWS, GROUP), tl.float32)
    for unit in range(H // UNIT):
        x = tl.trans(tl.load(activations + unit * UNIT))
        w, scale = _unit(gate_rows, unit, GRID, VALUES, TYPE, ROWS, False)
        if EXACT:
            gate, gate_lost = _add(gate, gate_lost, tl.dot(w, x) * scale[:, None])
        else:
            gate += tl.dot(w, x) * scale[:, None]
        w, scale = _unit(up_rows, unit, GRID, VALUES, TYPE, ROWS, False)
        if EXACT:
            up, up_lost = _add(up, up_lost, tl.dot(w, x) * scale[:, None])
        else:
            up += tl.dot(w, x) * scale[:, None]
    if EXACT:
        gate, up = gate + gate_lost, up + up_lost
    _store_mid(gate, up, Mid, MidScales, group, within, tl.program_id(1), rows, I, ROWS, GROUP, MID)


@triton.jit(do_not_specialize=("FIRST",))
def _down(Mid, Pairs, Experts, PTR, VALUES, Probs, Parts, FIRST, H: tl.constexpr, I: tl.constexpr, ROW_BYTES, TYPE: tl.constexpr, ROWS: tl.constexpr, GROUP: tl.constexpr, EXACT: tl.constexpr, MID: tl.constexpr):
    # ROWS output rows of one expert for up to GROUP pairs routed to it. EXACT: float32 activations on FMA units, what the sums of the units round being kept; MID_BF16: BF16 activations on tensor cores; otherwise each activation as a BF16 pair (value, remainder).
    group = FIRST + tl.program_id(0)
    rows = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    within = tl.arange(0, GROUP)
    pair = tl.load(Pairs + group * GROUP + within)
    expert = tl.load(Experts + group).to(tl.int64)
    row = tl.load(PTR + expert * 3 + 2).to(tl.pointer_type(tl.uint8)) + rows * ROW_BYTES
    unit_columns: tl.constexpr = 32 if TYPE == 20 else 64
    inputs = Mid + (group.to(tl.int64) * GROUP + within)[:, None] * I + tl.arange(0, unit_columns)[None, :]
    value = tl.zeros((ROWS, GROUP), tl.float32)
    lost = tl.zeros((ROWS, GROUP), tl.float32)
    for unit in range(I // unit_columns):
        w, scale = _down_unit(row, unit, VALUES, TYPE, ROWS)
        mid = tl.trans(tl.load(inputs + unit * unit_columns))
        if EXACT:
            value, lost = _add(value, lost, tl.dot(w, mid, input_precision="ieee") * scale[:, None])
        elif MID == 1:
            value += tl.dot(w.to(tl.bfloat16), mid) * scale[:, None]
        else:
            hi = mid.to(tl.bfloat16)
            w16 = w.to(tl.bfloat16)
            value += (tl.dot(w16, hi) + tl.dot(w16, (mid - hi.to(tl.float32)).to(tl.bfloat16))) * scale[:, None]
    valid = pair >= 0
    if EXACT:
        value = value + lost
    tl.store(Parts + pair.to(tl.int64)[:, None] * H + rows[None, :], tl.trans(value) * tl.load(Probs + pair, mask=valid, other=0.0).to(tl.float32)[:, None], mask=valid[:, None])


@triton.jit
def _quantize_rows(X, CODES, SCALES, H: tl.constexpr, UNITS: tl.constexpr):
    """A few blocks of 32 columns of a token: int8 codes and a float32 scale per block."""
    place, token = tl.program_id(0), tl.program_id(1)
    units = place * UNITS + tl.arange(0, UNITS)
    offsets = token * H + units[:, None] * 32 + tl.arange(0, 32)[None, :]
    x = tl.load(X + offsets).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), 1) / 127.0, 1.0e-30)
    tl.store(CODES + offsets, tl.floor(x / scale[:, None] + 0.5).to(tl.int8))
    tl.store(SCALES + token * (H // 32) + units, scale)


@triton.jit(do_not_specialize=("FIRST",))
def _gate_up_codes(Codes, Scales, Pairs, Experts, PTR, GRID, VALUES, Mid, MidScales, FIRST, H: tl.constexpr, I: tl.constexpr, K: tl.constexpr, UNIT: tl.constexpr, ROW_BYTES, TYPE: tl.constexpr, ROWS: tl.constexpr, GROUP: tl.constexpr, MID: tl.constexpr):
    # As _gate_up, on int8 activations: each dot of a unit is an exact integer, its scales apply once.
    group = FIRST + tl.program_id(0)
    block = tl.program_id(1)
    rows = block * ROWS + tl.arange(0, ROWS)
    within = tl.arange(0, GROUP)
    pair = tl.load(Pairs + group * GROUP + within)
    expert = tl.load(Experts + group).to(tl.int64)
    gate_rows = tl.load(PTR + expert * 3).to(tl.pointer_type(tl.uint8)) + rows * ROW_BYTES
    up_rows = tl.load(PTR + expert * 3 + 1).to(tl.pointer_type(tl.uint8)) + rows * ROW_BYTES
    token = (tl.maximum(pair, 0) // K).to(tl.int64)  # what the places past the end compute is not read
    activations = Codes + token[:, None] * H + tl.arange(0, UNIT)[None, :]
    scales = Scales + token * (H // 32)
    gate = tl.zeros((ROWS, GROUP), tl.float32)
    up = tl.zeros((ROWS, GROUP), tl.float32)
    for unit in range(H // UNIT):
        x = tl.trans(tl.load(activations + unit * UNIT))
        scale_x = tl.load(scales + unit * UNIT // 32)[None, :]
        w, scale = _unit(gate_rows, unit, GRID, VALUES, TYPE, ROWS, True)
        gate += tl.dot(w, x, out_dtype=tl.int32).to(tl.float32) * scale[:, None] * scale_x
        w, scale = _unit(up_rows, unit, GRID, VALUES, TYPE, ROWS, True)
        up += tl.dot(w, x, out_dtype=tl.int32).to(tl.float32) * scale[:, None] * scale_x
    _store_mid(gate, up, Mid, MidScales, group, within, block, rows, I, ROWS, GROUP, MID)


@triton.jit
def _down_codes_unit(row, unit, VALUES, TYPE: tl.constexpr, ROWS: tl.constexpr):
    """32 columns of ROWS output rows: int8 weights [ROWS, 32] and their float32 scale [ROWS]."""
    if TYPE == 20:
        block = row + unit * 18
        q = tl.load(block[:, None] + 2 + tl.arange(0, 16)[None, :]).to(tl.int32)
        w = tl.reshape(tl.load(VALUES + ((q[:, None, :] >> (tl.arange(0, 2)[None, :, None] * 4)) & 15)), (ROWS, 32))
    else:  # Q2_0: half a block
        block = row + (unit // 2) * 18
        q = tl.load(block[:, None] + 2 + (unit % 2) * 8 + tl.arange(0, 8)[None, :]).to(tl.int32)
        w = (tl.reshape((q[:, :, None] >> (tl.arange(0, 4)[None, None, :] * 2)) & 3, (ROWS, 32)) - 1).to(tl.int8)
    return w, tl.load(block.to(tl.pointer_type(tl.float16))).to(tl.float32)


@triton.jit(do_not_specialize=("FIRST",))
def _down_codes(Mid, MidScales, Pairs, Experts, PTR, VALUES, Probs, Parts, FIRST, H: tl.constexpr, I: tl.constexpr, ROW_BYTES, TYPE: tl.constexpr, ROWS: tl.constexpr, GROUP: tl.constexpr):
    # As _down, on the codes of _gate_up_codes.
    group = FIRST + tl.program_id(0)
    rows = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    within = tl.arange(0, GROUP)
    pair = tl.load(Pairs + group * GROUP + within)
    expert = tl.load(Experts + group).to(tl.int64)
    row = tl.load(PTR + expert * 3 + 2).to(tl.pointer_type(tl.uint8)) + rows * ROW_BYTES
    place = group.to(tl.int64) * GROUP + within
    inputs = Mid + place[:, None] * I + tl.arange(0, 32)[None, :]
    scales = MidScales + place * (I // 32)
    value = tl.zeros((ROWS, GROUP), tl.float32)
    for unit in range(I // 32):
        w, scale = _down_codes_unit(row, unit, VALUES, TYPE, ROWS)
        value += tl.dot(w, tl.trans(tl.load(inputs + unit * 32)), out_dtype=tl.int32).to(tl.float32) * scale[:, None] * tl.load(scales + unit)[None, :]
    valid = pair >= 0
    tl.store(Parts + pair.to(tl.int64)[:, None] * H + rows[None, :], tl.trans(value) * tl.load(Probs + pair, mask=valid, other=0.0).to(tl.float32)[:, None], mask=valid[:, None])


@cache
def supported(device):
    """Tensor-core kernels on BF16 activations: BF16 Triton is there from compute capability 8.0 (AMD: not validated)."""
    return torch.version.hip is None and torch.cuda.get_device_capability(device) >= (8, 0)


@triton.jit(do_not_specialize=("ROW_BYTES",))
def _decode_rows(RAW, GRID, VALUES, OUT, ROW_BYTES, COLUMN_COUNT: tl.constexpr, UNIT: tl.constexpr, TYPE: tl.constexpr):
    """Packed rows to their values, a unit of columns per program: what a table lookup needs, in one launch."""
    rows = tl.program_id(0) + tl.arange(0, 1)
    w, scale = _unit(RAW.to(tl.pointer_type(tl.uint8)) + rows * ROW_BYTES, tl.program_id(1), GRID, VALUES, TYPE, 1, False)
    tl.store(OUT + rows[:, None] * COLUMN_COUNT + tl.program_id(1) * UNIT + tl.arange(0, UNIT)[None, :], (w.to(tl.float32) * scale[:, None]).to(OUT.dtype.element_ty))


def decode_rows_supported(device, type):
    """The formats of the kernels, their tables made: they cannot be during a graph capture."""
    return type in COLUMNS and supported(device) and (not torch.cuda.is_current_stream_capturing() or device in expert_tables)


def decode_rows(raw, type, rows, columns, dtype):
    """raw: the packed rows, one after the other, on the GPU. float32 or BF16 [rows, columns], the values of the PyTorch dequantization."""
    tables = prepare(raw.device)
    unit = COLUMNS[type]
    out = torch.empty(rows, columns, dtype=dtype, device=raw.device)
    _decode_rows[(rows, columns // unit)](raw, tables.get(type, tables[IQ3_S]), tables[IQ4_NL], out, columns // 64 * 18 if type == Q2_0 else columns // 256 * ROW_BYTES[type], columns, unit, type, num_warps=1)
    return out


def groups(ids, group, order=None):
    """On the host: (pairs int32 [groups, group] of pair indices, -1 past the end of an expert; experts int32 [groups]; first int64 [len(order) + 1]: the first group of each expert in order). order: the routed experts in their order of processing, by default ascending."""
    flat = ids.reshape(-1).numpy().astype(np.int16)
    sorted_pairs = np.argsort(flat, kind="stable").astype(np.int32)
    counts = np.bincount(flat)
    if order is None:
        order = np.nonzero(counts)[0]
    starts = np.zeros(len(counts) + 1, dtype=np.int64)
    starts[1:] = np.cumsum(counts)
    needed = (counts[order] + group - 1) // group
    first = np.zeros(len(order) + 1, dtype=np.int64)
    first[1:] = np.cumsum(needed)
    pairs = np.full((int(first[-1]), group), -1, dtype=np.int32)
    experts = np.repeat(order.astype(np.int32), needed)
    flat_pairs = pairs.reshape(-1)
    for index, expert in enumerate(order.tolist()):  # at most the experts of a bank
        flat_pairs[first[index] * group:first[index] * group + counts[expert]] = sorted_pairs[starts[expert]:starts[expert + 1]]
    return pairs, experts, first


class GroupedPass:
    """A pass of many tokens: run() computes groups of experts (some may arrive later, streamed), finish() gives the float32 [tokens, hidden] sum."""
    def __init__(self, x, ids, probabilities, intermediate, types, order=None, rows=None, group=None, warps=None, exact=None, zero=False, down_rows=None, down_warps=None, codes=False, mid=None):
        if x.dtype != torch.bfloat16:
            raise ValueError(f"The grouped expert kernels take BF16 activations, not {x.dtype}")
        self.x, self.probabilities, self.intermediate, self.types = x, probabilities.reshape(-1).float().contiguous(), intermediate, types
        self.tokens, self.hidden = x.shape
        self.topk = ids.shape[-1]
        group = group or (64 if x.shape[0] >= LONG_GROUPS_TOKENS else 32)
        rows, warps = (rows, warps) if rows else GATE_UP_TILES[types[0]]
        self.rows, self.group, self.warps = rows, group, warps
        self.down_rows, self.down_warps = (down_rows, down_warps) if down_rows else DOWN_TILE
        self.exact = not kernel_policy.allow_approximate() if exact is None else exact
        self.gate, up, self.down = types
        if self.gate != up:
            raise NotImplementedError("The GPU kernels take one format for the gate and up rows")
        pairs, experts, self.first = groups(ids.to("cpu"), group, None if order is None else order.astype(np.int64))
        self.count = pairs.shape[0]
        device = x.device
        self.pairs = torch.from_numpy(pairs).to(device)
        self.experts = torch.from_numpy(experts).to(device)
        mid = (MID_FP32 if self.exact else MID_BF16) if mid is None else mid  # BF16 activations for the output projection: one dot instead of two, an error of the order of the rounding of its input and output rows (1.2e-3 against 2e-6 for pairs)
        self.codes, self.mid_kind = codes and self.gate != IQ2_S, mid  # IQ2_S has a scale per 16 columns: integer dots take 32 at least
        if self.codes:  # int8 activations with a scale per 32 of them, as the block path of INT8 linears: integer products on tensor cores, each dot exact
            hidden = self.hidden
            units = min(16, (hidden // 32) & -(hidden // 32))
            self.x_codes = torch.empty(self.tokens, hidden, dtype=torch.int8, device=device)
            self.x_scales = torch.empty(self.tokens, hidden // 32, dtype=torch.float32, device=device)
            _quantize_rows[(hidden // 32 // units, self.tokens)](x, self.x_codes, self.x_scales, hidden, units, num_warps=2)
        self.mid = torch.empty(self.count * group, intermediate, dtype=(torch.float32, torch.bfloat16, torch.int8)[mid], device=device)
        self.mid_scales = torch.empty(self.count * group, intermediate // 32, dtype=torch.float32, device=device) if mid == MID_CODES else self.mid
        self.parts = (torch.zeros if zero else torch.empty)(self.tokens * self.topk, self.hidden, dtype=torch.float32, device=device)
        self.tables = prepare(device)
        self.unit = COLUMNS[self.gate]
        self.row_bytes = self.hidden // 64 * 18 if self.gate == Q2_0 else self.hidden // 256 * ROW_BYTES[self.gate]
        self.down_bytes = intermediate // 32 * 18 if self.down == IQ4_NL else intermediate // 64 * 18

    def run(self, pointers, start=0, stop=None, parts="both"):
        """The experts [start, stop) of the order of processing, whose rows pointers gives."""
        first = int(self.first[start])
        count = int(self.first[len(self.first) - 1 if stop is None else stop]) - first
        if not count:
            return
        tables = self.tables
        if parts != "down":
            if self.codes:
                _gate_up_codes[(count, self.intermediate // self.rows)](self.x_codes, self.x_scales, self.pairs, self.experts, pointers, tables.get(self.gate, tables[IQ3_S]), tables["codes"], self.mid, self.mid_scales, first, self.hidden, self.intermediate, self.topk, 32 if self.gate == Q2_0 else self.unit, self.row_bytes, self.gate, self.rows, self.group, self.mid_kind, num_warps=self.warps, num_stages=1)
            else:
                _gate_up[(count, self.intermediate // self.rows)](self.x, self.pairs, self.experts, pointers, tables.get(self.gate, tables[IQ3_S]), tables[IQ4_NL], self.mid, self.mid_scales if self.mid_kind == MID_CODES else self.mid, first, self.hidden, self.intermediate, self.topk, self.unit, self.row_bytes, self.gate, self.rows, self.group, self.exact, self.mid_kind, num_warps=self.warps, num_stages=1)
        if parts != "up":
            if self.mid_kind == MID_CODES:
                _down_codes[(count, self.hidden // self.down_rows)](self.mid, self.mid_scales, self.pairs, self.experts, pointers, tables["codes"], self.probabilities, self.parts, first, self.hidden, self.intermediate, self.down_bytes, self.down, self.down_rows, self.group, num_warps=self.down_warps, num_stages=1)
            else:
                _down[(count, self.hidden // self.down_rows)](self.mid, self.pairs, self.experts, pointers, tables[IQ4_NL], self.probabilities, self.parts, first, self.hidden, self.intermediate, self.down_bytes, self.down, self.down_rows, self.group, self.exact, self.mid_kind, num_warps=self.down_warps, num_stages=1)

    def finish(self):
        self.mid = self.pairs = self.experts = self.x_codes = self.x_scales = self.mid_scales = None
        output = self.parts.view(self.tokens, self.topk, self.hidden).sum(1)
        self.parts = None
        return output


@cache
def _native_experts():
    # the int8 kernels of WGPKernels, when the installed version has them
    return importlib.import_module("wgpkernels.experts") if importlib.util.find_spec("wgpkernels.experts") else None


def prompt_pass(x, ids, probabilities, intermediate, types, order=None, zero=False):
    """The pass of many tokens for these weights: the native int8 kernels (three times faster, their activations and SwiGLU in int8: an error of about 1% per layer against 0.1%) where approximate kernels are allowed and the device and the formats have them, else the kernels above."""
    if NATIVE_PASS and kernel_policy.allow_approximate() and types[0] == types[1] and (native := _native_experts()) is not None and native.supported(x.device, types[0], types[2]):
        return native.ExpertPass(x, ids, probabilities, intermediate, types, order, zero=zero)
    return GroupedPass(x, ids, probabilities, intermediate, types, order, zero=zero)


from shared.kernels.triton_compilation_log import install_triton_compilation_logger  # noqa: E402

install_triton_compilation_logger()
