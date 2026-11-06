# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
# Written for the LM engine with ideas of Strata (MIT License, https://github.com/Niko1221/Strata): the kernels of a hyper-connection read their weights once.
"""Fused kernels of hyper-connections (parallel residual streams mixed through a low-rank gate) and of softmax routers, for decoding and verification passes (a few rows): BF16
projections with float32 activations and accumulation, float32 norms and streams, one launch per weight pass instead of about 25 small operations. Compiled once per model: the row
count is the grid, never a specialization. Passes of many rows (prompts) keep BLAS for the projections and fuse what surrounds them: the injection and the norm in one pass over the streams, the gate and the mean in another."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from shared.kernels.triton_compilation_log import install_triton_compilation_logger

MAX_ROWS = 16  # larger passes (prompt processing) keep the PyTorch path

# Launch configurations: rows per program, columns per tile, warps
DOWN_CONFIG = (4, 2048, 8)
UP_CONFIG = (8, 64, 4)  # columns of the output per program, tile of the rank, warps
ROUTER_CONFIG = (4, 512, 4)


@triton.jit
def _sigmoid(x):
    return tl.div_rn(1.0, 1.0 + libdevice.exp(-x))


@triton.jit
def _silu(x):
    return tl.div_rn(x, 1.0 + libdevice.exp(-x))


@triton.jit
def _norm_kernel(STATE, NEXT, OUTPUT, WEIGHTS, NORM, NORMED, LOW, EPS, HIDDEN: tl.constexpr, STREAMS: tl.constexpr, FOLD: tl.constexpr, HAS_LOW: tl.constexpr, BLOCK: tl.constexpr):
    """RMS norm of a stream of a row, weighted; with FOLD, the output of the mixer before it injected first (combine); with HAS_LOW, in BF16 as well, for projections."""
    token, stream = tl.program_id(0), tl.program_id(1)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < HIDDEN
    base = token * (STREAMS * HIDDEN) + stream * HIDDEN + offsets
    x = tl.load(STATE + base, mask=mask, other=0.0)
    if FOLD:
        x = x + tl.load(OUTPUT + token * HIDDEN + offsets, mask=mask, other=0.0).to(tl.float32) * tl.load(WEIGHTS + token * STREAMS + stream)
        tl.store(NEXT + base, x, mask=mask)
    scale = tl.div_rn(1.0, tl.sqrt_rn(tl.div_rn(tl.sum(x * x, 0), float(HIDDEN)) + EPS))
    normed = x * scale * tl.load(NORM + stream * HIDDEN + offsets, mask=mask, other=0.0)
    tl.store(NORMED + base, normed, mask=mask)
    if HAS_LOW:
        tl.store(LOW + base, normed.to(tl.bfloat16), mask=mask)


@triton.jit
def _mix_kernel(GATE, NORMED, MIXED, HIDDEN: tl.constexpr, STREAMS: tl.constexpr, COLUMNS: tl.constexpr):
    """COLUMNS columns of the mix of a row: each stream's normed value behind the sigmoid of its gate, then their mean, one rounding to the output's dtype."""
    chunk, token = tl.program_id(0), tl.program_id(1)
    index = tl.arange(0, STREAMS * COLUMNS)
    places = token * (STREAMS * HIDDEN) + index // COLUMNS * HIDDEN + chunk * COLUMNS + index % COLUMNS
    gated = tl.load(NORMED + places) * _sigmoid(tl.load(GATE + places).to(tl.float32))
    mixed = tl.div_rn(tl.sum(tl.reshape(gated, (STREAMS, COLUMNS)), 0), float(STREAMS))
    tl.store(MIXED + token * HIDDEN + chunk * COLUMNS + tl.arange(0, COLUMNS), mixed.to(MIXED.dtype.element_ty))


@triton.jit
def _down_kernel(NORMED, W_DOWN, W_INJECT, LOW, INJECT, RANK: tl.constexpr, WIDTH: tl.constexpr, STREAMS: tl.constexpr, GROUPS: tl.constexpr, M: tl.constexpr, N: tl.constexpr, FULL: tl.constexpr):
    """One group of rows of the low-rank projection (silu of the dot over the streams), the last group being the rows of the inject weights (2 sigmoid of the same)."""
    group, token = tl.program_id(0), tl.program_id(1)
    inject = group >= GROUPS
    rows = (tl.where(inject, group - GROUPS, group) * M + tl.arange(0, M)).to(tl.int64)
    weight = tl.where(inject, W_INJECT, W_DOWN)
    valid = tl.where(inject, rows < STREAMS, rows < RANK)
    row_offsets = rows[:, None] * WIDTH
    acc = tl.zeros((M,), tl.float32)
    for start in range(0, WIDTH, N):
        columns = start + tl.arange(0, N)
        x = tl.load(NORMED + token * WIDTH + columns)
        if FULL:  # every group has M rows
            w = tl.load(weight + row_offsets + columns[None, :])
        else:
            w = tl.load(weight + row_offsets + columns[None, :], mask=valid[:, None], other=0.0)
        acc += tl.sum(w.to(tl.float32) * x[None, :], 1)
    x = tl.div_rn(acc, float(STREAMS))
    output = tl.where(inject, INJECT + token * STREAMS + rows, LOW + token * RANK + rows)
    tl.store(output, tl.where(inject, 2.0 * _sigmoid(x), _silu(x)), mask=valid)


@triton.jit
def _up_kernel(LOW, NORMED, W_UP, MIXED, UNROUNDED, RANK: tl.constexpr, HIDDEN: tl.constexpr, STREAMS: tl.constexpr, COLUMNS: tl.constexpr, K: tl.constexpr, KEEP_FP32: tl.constexpr):
    """COLUMNS columns of the mix: each stream's normed value behind its sigmoid gate (the gate: the row of the up projection, over the low-rank values), then their mean; with
    KEEP_FP32, also in float32 (for consumers reading its unrounded values)."""
    group, token = tl.program_id(0), tl.program_id(1)
    index = tl.arange(0, STREAMS * COLUMNS)
    rows = (index // COLUMNS * HIDDEN + group * COLUMNS + index % COLUMNS).to(tl.int64)
    acc = tl.zeros((STREAMS * COLUMNS,), tl.float32)
    for start in range(0, RANK, K):
        columns = start + tl.arange(0, K)
        low = tl.load(LOW + token * RANK + columns)
        acc += tl.sum(tl.load(W_UP + rows[:, None] * RANK + columns[None, :]).to(tl.float32) * low[None, :], 1)
    gated = tl.load(NORMED + token * (STREAMS * HIDDEN) + rows) * _sigmoid(acc)
    mixed = tl.div_rn(tl.sum(tl.reshape(gated, (STREAMS, COLUMNS)), 0), float(STREAMS))
    offsets = token * HIDDEN + group * COLUMNS + tl.arange(0, COLUMNS)
    tl.store(MIXED + offsets, mixed.to(MIXED.dtype.element_ty))
    if KEEP_FP32:
        tl.store(UNROUNDED + offsets, mixed)


@triton.jit
def _combine_kernel(STATE, ROUTED, SHARED, GATE, WEIGHTS, HIDDEN: tl.constexpr, STREAMS: tl.constexpr, HAS_SHARED: tl.constexpr, BLOCK: tl.constexpr):
    """state += (routed + gate * shared) * weight of the stream, in float32."""
    chunk, stream, token = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    offsets = chunk * BLOCK + tl.arange(0, BLOCK)
    output = tl.load(ROUTED + token * HIDDEN + offsets).to(tl.float32)
    if HAS_SHARED:
        output = output + tl.load(GATE + token) * tl.load(SHARED + token * HIDDEN + offsets).to(tl.float32)
    place = STATE + token * (STREAMS * HIDDEN) + stream * HIDDEN + offsets
    tl.store(place, tl.load(place) + output * tl.load(WEIGHTS + token * STREAMS + stream))


@triton.jit
def _router_kernel(X, W_ROUTER, W_GATE, LOGITS, GATE, EXPERTS: tl.constexpr, WIDTH: tl.constexpr, GROUPS: tl.constexpr, M: tl.constexpr, N: tl.constexpr):
    """A group of rows of the router's float32 logits, the last group being the one row of the shared expert's gate (its sigmoid)."""
    group, token = tl.program_id(0), tl.program_id(1)
    shared = group == GROUPS
    rows = (tl.where(shared, 0, group) * M + tl.arange(0, M)).to(tl.int64)
    weight = tl.where(shared, W_GATE, W_ROUTER)
    valid = tl.where(shared, rows < 1, rows < EXPERTS)
    acc = tl.zeros((M,), tl.float32)
    for start in range(0, WIDTH, N):
        columns = start + tl.arange(0, N)
        x = tl.load(X + token * WIDTH + columns).to(tl.float32)
        acc += tl.sum(tl.load(weight + rows[:, None] * WIDTH + columns[None, :], mask=valid[:, None], other=0.0).to(tl.float32) * x[None, :], 1)
    tl.store(tl.where(shared, GATE + token + rows, LOGITS + token * EXPERTS + rows), tl.where(shared, _sigmoid(acc), acc), mask=valid)


@triton.jit
def _larger(a, a_id, b, b_id):  # the larger probability, the first expert of a tie
    take = (b > a) | ((b == a) & (b_id < a_id))
    return tl.where(take, b, a), tl.where(take, b_id, a_id)


@triton.jit(do_not_specialize_on_alignment=("IDS", "WEIGHTS"))  # views of a payload at offsets depending on the row count
def _route_kernel(LOGITS, IDS, WEIGHTS, EXPERTS: tl.constexpr, TOP: tl.constexpr, BLOCK: tl.constexpr, TOP_BLOCK: tl.constexpr):
    """Softmax of a row's logits, its TOP largest probabilities in descending order (the first expert of a tie), normalized over them."""
    token = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    mask = offsets < EXPERTS
    logits = tl.load(LOGITS + token * EXPERTS + offsets, mask=mask, other=float("-inf"))
    exponentials = tl.where(mask, libdevice.exp(logits - tl.max(logits, 0)), 0.0)
    probabilities = tl.where(mask, tl.div_rn(exponentials, tl.sum(exponentials, 0)), float("-inf"))
    ranks = tl.arange(0, TOP_BLOCK)
    ids = tl.zeros((TOP_BLOCK,), tl.int32)
    values = tl.zeros((TOP_BLOCK,), tl.float32)
    for rank in range(TOP):
        value, best = tl.reduce((probabilities, offsets), 0, _larger)
        ids = tl.where(ranks == rank, best, ids)
        values = tl.where(ranks == rank, value, values)
        probabilities = tl.where(offsets == best, float("-inf"), probabilities)
    tl.store(IDS + token * TOP + ranks, ids, mask=ranks < TOP)
    tl.store(WEIGHTS + token * TOP + ranks, tl.div_rn(values, tl.sum(values, 0)), mask=ranks < TOP)


@triton.jit
def _silu_product_kernel(GATE, VALUE, OUT, WIDTH: tl.constexpr, BLOCK: tl.constexpr):
    """silu(gate) * value, one rounding to the output's dtype."""
    chunk, token = tl.program_id(0), tl.program_id(1)
    offsets = token * WIDTH + chunk * BLOCK + tl.arange(0, BLOCK)
    tl.store(OUT + offsets, (_silu(tl.load(GATE + offsets).to(tl.float32)) * tl.load(VALUE + offsets).to(tl.float32)).to(OUT.dtype.element_ty))


install_triton_compilation_logger()


def launch_norm(state, next_state, output, weights, norm, normed, eps, streams, hidden, low=None):
    fold = output is not None
    assert state.is_cuda and norm.is_cuda and (not fold or (output.is_cuda and weights.is_cuda))
    _norm_kernel[(state.shape[0], streams)](state, next_state, output if fold else state, weights if fold else state, norm, normed, normed if low is None else low, eps, hidden, streams, fold, low is not None, triton.next_power_of_2(hidden), num_warps=8)


def launch_down(normed, down, inject, low, connection, streams, config=DOWN_CONFIG):
    rows, columns, warps = config
    rank, width = down.shape
    assert width % columns == 0 and normed.is_cuda and down.is_cuda and low.is_cuda
    groups = triton.cdiv(rank, rows)
    _down_kernel[(groups + (0 if inject is None else triton.cdiv(streams, rows)), normed.shape[0])](normed, down, down if inject is None else inject, low, low if inject is None else connection, rank, width, streams, groups, rows, columns, rank % rows == 0 and streams % rows == 0, num_warps=warps)


def launch_up(low, normed, up, mixed, unrounded, streams, hidden, config=UP_CONFIG):
    columns, tile, warps = config
    keep = unrounded is not None
    assert up.shape[1] % tile == 0 and hidden % columns == 0 and normed.is_cuda and up.is_cuda and low.is_cuda
    _up_kernel[(hidden // columns, normed.shape[0])](low, normed, up, mixed, unrounded if keep else low, up.shape[1], hidden, streams, columns, tile, keep, num_warps=warps)


def read(state, norm, down, up, inject, streams, hidden, eps, mixed_dtype, output=None, weights=None, unrounded=None, config=None):
    """state [tokens, streams * hidden] float32. (mixed [tokens, hidden] in mixed_dtype, weights [tokens, streams] or None, next state). With output and weights, the mixer's output
    before this connection is injected first, into a new state (combine). unrounded: a float32 tensor [tokens, hidden] to write the mix into as well."""
    down_config, up_config = config or (DOWN_CONFIG, UP_CONFIG)
    tokens, device = state.shape[0], state.device
    next_state = state if output is None else torch.empty(state.shape, dtype=state.dtype, device=device)
    normed = torch.empty(tokens, streams * hidden, dtype=torch.float32, device=device)
    launch_norm(state, next_state, output, weights, norm, normed, eps, streams, hidden)
    low = torch.empty(tokens, down.shape[0], dtype=torch.float32, device=device)
    connection = torch.empty(tokens, streams, dtype=torch.float32, device=device) if inject is not None else None
    launch_down(normed, down, inject, low, connection, streams, down_config)
    mixed = torch.empty(tokens, hidden, dtype=mixed_dtype, device=device)
    launch_up(low, normed, up, mixed, unrounded, streams, hidden, up_config)
    return mixed, connection, next_state


def norm_rows(state, norm, eps, streams, hidden, output=None, weights=None):
    """state [tokens, streams * hidden] float32, written in place when the mixer's output before this connection is injected first: (normed float32, the same in BF16)."""
    normed = torch.empty(state.shape, dtype=torch.float32, device=state.device)
    low = torch.empty(state.shape, dtype=torch.bfloat16, device=state.device)
    launch_norm(state, state, output, weights, norm, normed, eps, streams, hidden, low)
    if output is not None:
        torch.autograd.graph.increment_version(state)
    return normed, low


def mix_rows(gate, normed, streams, hidden, dtype):
    """[tokens, hidden] in dtype: the mean over the streams of normed behind the sigmoid of gate, both [tokens, streams * hidden]."""
    columns = min(512, hidden & -hidden)
    assert gate.is_cuda and normed.is_cuda
    mixed = torch.empty(gate.shape[0], hidden, dtype=dtype, device=gate.device)
    _mix_kernel[(hidden // columns, gate.shape[0])](gate, normed, mixed, hidden, streams, columns, num_warps=8)
    return mixed


def combine_(state, routed, weights, streams, hidden, shared=None, gate=None):
    """state += (routed [+ gate * shared]) * weights, all [tokens, ...], in place."""
    tokens = state.shape[0]
    block, has_shared = 256, shared is not None
    assert hidden % block == 0 and state.is_cuda and routed.is_cuda and weights.is_cuda and (not has_shared or (shared.is_cuda and gate.is_cuda))
    _combine_kernel[(hidden // block, streams, tokens)](state, routed, shared if has_shared else routed, gate if has_shared else weights, weights, hidden, streams, has_shared, block, num_warps=4)
    torch.autograd.graph.increment_version(state)
    return state


def router_rows(x, router, gate_row, config=ROUTER_CONFIG):
    """x [tokens, hidden]: float32 logits [tokens, experts] of router (BF16 [experts, hidden]) and sigmoid gate [tokens, 1] of gate_row (BF16 [1, hidden])."""
    rows, columns, warps = config
    tokens, device = x.shape[0], x.device
    experts, width = router.shape
    assert width % columns == 0 and x.is_cuda and router.is_cuda and gate_row.is_cuda
    logits = torch.empty(tokens, experts, dtype=torch.float32, device=device)
    gate = torch.empty(tokens, 1, dtype=torch.float32, device=device)
    groups = triton.cdiv(experts, rows)
    _router_kernel[(groups + 1, tokens)](x, router, gate_row, logits, gate, experts, width, groups, rows, columns, num_warps=warps)
    return logits, gate


def softmax_topk(logits, top, ids=None, weights=None):
    """float32 logits [tokens, experts]: ids int32, weights float32 [tokens, top], as softmax, topk and normalization, written into ids and weights when given."""
    tokens, experts = logits.shape
    ids = torch.empty(tokens, top, dtype=torch.int32, device=logits.device) if ids is None else ids
    weights = torch.empty(tokens, top, dtype=torch.float32, device=logits.device) if weights is None else weights
    _route_kernel[(tokens,)](logits, ids, weights, experts, top, triton.next_power_of_2(experts), triton.next_power_of_2(top), num_warps=1)
    return ids, weights


def silu_product(gate, value):
    """silu(gate) * value of two tensors [tokens, width], in gate's dtype."""
    tokens, width = gate.shape
    output = torch.empty(gate.shape, dtype=gate.dtype, device=gate.device)
    block = 64
    _silu_product_kernel[(width // block, tokens)](gate, value, output, width, block, num_warps=2)
    return output
