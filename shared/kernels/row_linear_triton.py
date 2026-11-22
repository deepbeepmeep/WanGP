# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""BF16 linear layers over the few rows of an LM decoding pass, each row computed on its own: an output of a token is the same float32
sum, in the same order, whatever the number of tokens of the pass. A speculative verification pass (2 to 16 rows) then gives each of its
rows what a decoding pass of one token gives, bit for bit, which library GEMMs do not (their kernels follow the row count: about 0.15 %
of BF16 outputs one rounding apart, measured on cuBLAS; a model that amplifies rounding then decodes other tokens at near ties).

Tensor-core path (default): a program multiplies TILE output rows by the 16 token columns of the pass (absent tokens read as zeros),
UNIT input columns per step, float32 accumulation as library GEMMs do. A column of a tensor-core product depends on that column of its
input only, and the steps run in the same order whatever the tokens: a token gets the same bits in a pass of one row and among 16.
linears() runs several weights of one input width in one launch (the projections of an attention block), so that the launch has enough
programs to read the weights at the card's bandwidth. WGP_INKLING2_DISABLE=tc_linear: the previous kernel, one program per output row
(a float32 dot with each token in turn; other rounding).

Optional soft cap of the rounded result (an LM head: cap * tanh(y / cap), float32). The input width is an architecture constant; tokens,
strides and output rows are runtime arguments. NVIDIA GPUs from compute capability 8.0."""
import os
from functools import cache

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

ROWS = 16  # tokens of a pass: beyond, a GEMM reads the weights once for many more rows than a program holds
TENSOR_CORES = "tc_linear" not in os.environ.get("WGP_INKLING2_DISABLE", "").split(",")
TILE, UNIT, WARPS, STAGES = 16, 256, 4, 3  # output rows and input columns per step of a program (RTX 5090: the 7 projections of 48 block applications 1.93 ms at 1 row, 2.03 at 8)
SEGMENTS = 4  # weights of one launch


@cache
def supported(device):
    device = torch.device(device)
    return device.type == "cuda" and torch.version.hip is None and torch.cuda.get_device_capability(device) >= (8, 0)


@triton.jit(do_not_specialize=("TOKENS", "STRIDE", "OUT", "cap", "inverse"), do_not_specialize_on_alignment=("X", "W", "Y"))
def _linear(X, W, Y, TOKENS, STRIDE, OUT, cap, inverse, IN: tl.constexpr, BLOCK: tl.constexpr, SOFTCAP: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, BLOCK)
    inside = columns < IN
    weights = tl.load(W + row * IN + columns, mask=inside, other=0.0).to(tl.float32)
    for token in range(TOKENS):  # one token after the other: the same sum for a token whatever the tokens of the pass
        y = tl.sum(weights * tl.load(X + token * STRIDE + columns, mask=inside, other=0.0).to(tl.float32))
        if SOFTCAP:
            y = libdevice.tanh(y.to(tl.bfloat16).to(tl.float32) * inverse) * cap
        tl.store(Y + token * OUT + row, y.to(Y.dtype.element_ty))


@triton.jit
def _tile(X, W, Y, STRIDE, OUT, first, tile, TOKENS, cap, inverse, IN: tl.constexpr, TILE: tl.constexpr, UNIT: tl.constexpr, SOFTCAP: tl.constexpr):
    rows = (tile - first) * TILE + tl.arange(0, TILE)
    token = tl.arange(0, 16)
    inside_rows = rows < OUT
    inside_tokens = token < TOKENS
    columns = tl.arange(0, UNIT)
    total = tl.zeros((TILE, 16), tl.float32)
    for unit in range(tl.cdiv(IN, UNIT)):
        column = unit * UNIT + columns
        w = tl.load(W + rows.to(tl.int64)[:, None] * IN + column[None, :], mask=inside_rows[:, None] & (column[None, :] < IN), other=0.0)
        x = tl.load(X + token[:, None] * STRIDE + column[None, :], mask=inside_tokens[:, None] & (column[None, :] < IN), other=0.0)
        total = tl.dot(w, tl.trans(x), total)
    if SOFTCAP:
        total = libdevice.tanh(total.to(tl.bfloat16).to(tl.float32) * inverse) * cap
    tl.store(Y + token[None, :] * OUT + rows[:, None], total.to(Y.dtype.element_ty), mask=inside_rows[:, None] & inside_tokens[None, :])


@triton.jit(do_not_specialize=("TOKENS", "S0", "S1", "S2", "S3", "O0", "O1", "O2", "O3", "T1", "T2", "T3", "cap", "inverse"),
            do_not_specialize_on_alignment=("X0", "X1", "X2", "X3"))
def _linears(X0, W0, Y0, X1, W1, Y1, X2, W2, Y2, X3, W3, Y3, S0, S1, S2, S3, O0, O1, O2, O3, T1, T2, T3, TOKENS, cap, inverse, IN: tl.constexpr, TILE: tl.constexpr, UNIT: tl.constexpr,
             SOFTCAP: tl.constexpr):
    # up to four weights, each over its own rows of the same input width; T1 to T3: the first tile of the second to fourth
    tile = tl.program_id(0)
    if tile < T1:
        _tile(X0, W0, Y0, S0, O0, 0, tile, TOKENS, cap, inverse, IN, TILE, UNIT, SOFTCAP)
    elif tile < T2:
        _tile(X1, W1, Y1, S1, O1, T1, tile, TOKENS, cap, inverse, IN, TILE, UNIT, SOFTCAP)
    elif tile < T3:
        _tile(X2, W2, Y2, S2, O2, T2, tile, TOKENS, cap, inverse, IN, TILE, UNIT, SOFTCAP)
    else:
        _tile(X3, W3, Y3, S3, O3, T3, tile, TOKENS, cap, inverse, IN, TILE, UNIT, SOFTCAP)


def _cap(softcap):
    cap = float(softcap or 0.0)
    return cap, float(np.float32(1.0) / np.float32(cap)) if cap else 0.0


def linears(pairs, dtype=None, softcap=None):
    """[(x [tokens, inputs], weight [outputs, inputs] BF16)] of one input width and token count (at most SEGMENTS): each x times its weight, in
    dtype (x's by default; float32 with softcap), in one launch."""
    tokens, inputs = pairs[0][0].shape
    pairs = [(x if x.stride(-1) == 1 else x.contiguous(), weight) for x, weight in pairs]
    if tokens > ROWS or len(pairs) > SEGMENTS or any(x.shape != (tokens, inputs) or weight.shape[1] != inputs or not weight.is_contiguous() or weight.dtype != torch.bfloat16 or not (x.is_cuda and weight.is_cuda)
                                                    for x, weight in pairs):
        raise ValueError("The row linears take the CUDA rows of a short pass, of one width, and contiguous BF16 weights")
    outs = [torch.empty(tokens, weight.shape[0], dtype=torch.float32 if softcap is not None else dtype or x.dtype, device=x.device) for x, weight in pairs]
    cap, inverse = _cap(softcap)
    if not TENSOR_CORES:
        for (x, weight), out in zip(pairs, outs):
            _linear[(weight.shape[0],)](x, weight, out, tokens, x.stride(0), weight.shape[0], cap, inverse, inputs, triton.next_power_of_2(inputs), softcap is not None, num_warps=4, enable_fp_fusion=False)
        return outs
    starts = np.cumsum([0] + [triton.cdiv(weight.shape[0], TILE) for _, weight in pairs]).tolist()
    last = len(pairs) - 1
    segments = [(*pairs[min(index, last)], outs[min(index, last)]) for index in range(SEGMENTS)]
    bounds = (starts[1:] + [starts[-1]] * SEGMENTS)[:SEGMENTS - 1]
    _linears[(starts[-1],)](*(t for segment in segments for t in segment), *(x.stride(0) for x, _, _ in segments), *(weight.shape[0] for _, weight, _ in segments), *bounds, tokens, cap, inverse, inputs, TILE, UNIT,
                            softcap is not None, num_warps=WARPS, num_stages=STAGES)
    return outs


def linear(x, weight, dtype=None, softcap=None):
    """weight [outputs, inputs] (BF16) times each row of x [tokens, inputs], in dtype (x's by default). softcap: the result rounded to x's dtype, then cap * tanh(. / cap) in float32."""
    return linears([(x, weight)], dtype, softcap)[0]


from shared.kernels.triton_compilation_log import install_triton_compilation_logger  # noqa: E402

install_triton_compilation_logger()
