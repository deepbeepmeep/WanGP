# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Fused row operations of LM prompt passes (hundreds to thousands of tokens) of the Inkling 2 model, beside the decoding kernels of
row_fused_triton (passes of up to 16 rows): a prompt pass made of many small PyTorch operations is bound by their launches when it is
short, and by their round trips through memory when it is long.

Each kernel computes in float32 registers in the order of the PyTorch expression it replaces and rounds where that expression stores
BF16, so that its results are those of that expression. Widths, taps and ranks are architecture constants; tokens and positions are
runtime arguments. NVIDIA GPUs from compute capability 7.5 (below 8.0 the BF16 tensors as their bits: shared.kernels.bf16_bits) and the
AMD architectures of bf16_bits.HIP_ARCHS; elsewhere the callers keep their PyTorch path. DeviceGroupedPass runs the grouped expert kernels,
which take expert_rows_triton.pass_supported."""
from functools import cache

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from shared.kernels.bf16_bits import bits_view, convert, portable_supported, reads_bits, round_bf16, to_float


@cache
def supported(device):
    return portable_supported(device)


def _block(width):
    return triton.next_power_of_2(width)


@triton.jit(do_not_specialize=("RING", "eps", "threshold", "inverse"), do_not_specialize_on_alignment=("X", "R", "W", "DYN", "POS", "Y"))
def _conv_rows(X, R, W, DYN, POS, Y, RING, eps, threshold, inverse, C: tl.constexpr, TAPS: tl.constexpr, N: tl.constexpr, POST: tl.constexpr, DYNAMIC: tl.constexpr, BLOCK: tl.constexpr,
               BITS: tl.constexpr):
    # Causal depthwise convolution with its residual over the rows of a pass, the rows before it read from the ring R by position (which this
    # kernel does not write: a pass's rows may reach the ring rows its first tokens read); a program: N channels of a token. POST 0: nothing;
    # 1: the BF16 result clipped (l2_clip); 2: normalized (rms). DYNAMIC: the taps W [TAPS, C] by backward distance plus those of the token
    # and of its group of N channels, DYN [tokens, groups * TAPS]; else W [C, TAPS], the last tap on the token.
    token = tl.program_id(0).to(tl.int64)
    channels = tl.program_id(1) * N + tl.arange(0, BLOCK)
    inside = tl.arange(0, BLOCK) < N
    first = tl.load(POS)
    position = first + token
    total = tl.zeros((BLOCK,), tl.float32)
    for tap in tl.static_range(TAPS):
        source = position - tap if DYNAMIC else position - (TAPS - 1) + tap
        if source >= first:
            value = to_float(tl.load(X + (source - first) * C + channels, mask=inside, other=0.0))
        else:
            value = to_float(tl.load(R + (source % RING) * C + channels, mask=inside & (source >= 0), other=0.0))
        if DYNAMIC:
            total += value * (to_float(tl.load(W + tap * C + channels, mask=inside, other=0.0)) + to_float(tl.load(DYN + (token * (C // N) + tl.program_id(1)) * TAPS + tap)))
        else:
            total += value * to_float(tl.load(W + channels * TAPS + tap, mask=inside, other=0.0))
    x = tl.load(X + token * C + channels, mask=inside, other=0.0)
    y = (total + to_float(x)) * inverse  # PyTorch divides by a scalar this way
    if POST == 1:
        y = round_bf16(y, BITS)
        y = y * tl.minimum(tl.div_rn(threshold, tl.sqrt_rn(tl.sum(y * y)) + 1e-6), 1.0)
    if POST == 2:
        y = round_bf16(y, BITS)
        y = y * tl.rsqrt(tl.div_rn(tl.sum(y * y), N * 1.0) + eps)
    tl.store(Y + token * C + channels, convert(y, Y), mask=inside)


def causal_conv(x, ring, weight, positions, dtype, group=None, post=0, eps=0.0, threshold=0.0, dynamic=None, real=None):
    """row_fused_triton.causal_conv for a pass of any rows of consecutive positions (x [tokens, channels] contiguous, model dtype): the rows before the pass from
    the ring by position, then the ring takes the pass's last rows (as many as it holds), after the convolution."""
    tokens, channels = x.shape
    taps = weight.shape[-1] if dynamic is None else weight.shape[0]
    if not x.is_contiguous() or (dynamic is not None and not dynamic.is_contiguous()):
        raise ValueError("The fused convolution takes contiguous rows")
    if not (x.is_cuda and ring.is_cuda and positions.is_cuda):
        raise ValueError("The fused convolution runs on CUDA tensors")
    group = group or min(channels, 256)
    out = torch.empty(x.shape, dtype=dtype, device=x.device)
    inverse = float(np.float32(1.0) / np.float32(((taps + 1) if dynamic is None else (2 * taps + 1)) ** 0.5))
    bits = reads_bits(x.device)
    x_ = bits_view(x, bits)
    _conv_rows[(tokens, channels // group)](x_, bits_view(ring, bits), bits_view(weight, bits), x_ if dynamic is None else bits_view(dynamic, bits), positions, bits_view(out, bits), ring.shape[0], eps, threshold, inverse,
                                            channels, taps, group, post, dynamic is not None, _block(group), bits, num_warps=1, enable_fp_fusion=False)
    written = min(tokens, ring.shape[0])
    if real is not None:  # a padded pass (a graph of a short prompt): the ring takes the last rows of the real ones only
        ring_rows_store(x, ring, positions, real)
    elif RING_KERNEL:
        _ring_store[(written,)](x_, bits_view(ring, bits), positions, tokens - written, ring.shape[0], channels, _block(channels), num_warps=4)
        torch.autograd.graph.increment_version(ring)
    else:
        ring[positions[-written:] % ring.shape[0]] = x[-written:]
    return out


@triton.jit(do_not_specialize=("FIRST_ROW", "RING"), do_not_specialize_on_alignment=("X", "R", "POS"))
def _ring_store(X, R, POS, FIRST_ROW, RING, C: tl.constexpr, BLOCK: tl.constexpr):
    # row FIRST_ROW + program of the pass into the ring at its position (the last rows of a pass, as many as the ring holds: one row per slot)
    row = FIRST_ROW + tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, BLOCK)
    inside = columns < C
    slot = (tl.load(POS) + row) % RING
    tl.store(R + slot * C + columns, tl.load(X + row * C + columns, mask=inside, other=0.0), mask=inside)


@triton.jit(do_not_specialize=("RING", "TOKENS"), do_not_specialize_on_alignment=("X", "R", "POS", "REAL"))
def _ring_store_real(X, R, POS, REAL, RING, TOKENS, C: tl.constexpr, BLOCK: tl.constexpr):
    # of the first REAL rows of a pass (REAL on the device: a padded pass), the last ones the ring holds, into the ring at their positions
    real = tl.minimum(tl.load(REAL), TOKENS)
    count = tl.minimum(real, RING)
    index = tl.program_id(0)
    if index < count:
        row = (real - count + index).to(tl.int64)
        columns = tl.arange(0, BLOCK)
        inside = columns < C
        slot = (tl.load(POS) + row) % RING
        tl.store(R + slot * C + columns, tl.load(X + row * C + columns, mask=inside, other=0.0), mask=inside)


def ring_rows_store(x, ring, positions, real):
    """The ring [rows, ...] takes, at their positions (consecutive from positions[0]), the last rows it holds of the first real (int32 [1] on the device) rows of x [tokens, ...]: the rows
    of a padded pass beyond the real ones write nothing."""
    tokens, channels = x.shape[0], x[0].numel()
    bits = reads_bits(x.device)
    _ring_store_real[(min(tokens, ring.shape[0]),)](bits_view(x, bits), bits_view(ring, bits), positions, real, ring.shape[0], tokens, channels, _block(channels), num_warps=4)
    torch.autograd.graph.increment_version(ring)


RING_KERNEL = True  # the ring write of causal_conv in one launch (False: the PyTorch index write of _SlotState.history, the same rows)


@triton.jit(do_not_specialize=("eps", "STRIDE_A", "STRIDE_B"), do_not_specialize_on_alignment=("A", "B", "Y"))
def _gate_inputs(A, B, Y, eps, STRIDE_A, STRIDE_B, H: tl.constexpr, BLOCK: tl.constexpr):
    # [rms(a) | rms(b)] of a token rounded to BF16 (inkling2_core.rms of the float32 rows: x * rsqrt(mean(x^2) + eps))
    token = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, BLOCK)
    inside = columns < H
    a = to_float(tl.load(A + token * STRIDE_A + columns, mask=inside, other=0.0))
    b = to_float(tl.load(B + token * STRIDE_B + columns, mask=inside, other=0.0))
    a = a * tl.rsqrt(tl.div_rn(tl.sum(a * a), H * 1.0) + eps)
    b = b * tl.rsqrt(tl.div_rn(tl.sum(b * b), H * 1.0) + eps)
    tl.store(Y + token * (2 * H) + columns, convert(a, Y), mask=inside)
    tl.store(Y + token * (2 * H) + H + columns, convert(b, Y), mask=inside)


@triton.jit(do_not_specialize=("STRIDE_A", "STRIDE_B"), do_not_specialize_on_alignment=("A", "B", "F", "N", "Y"))
def _gate_combine(A, B, F, N, Y, STRIDE_A, STRIDE_B, H: tl.constexpr, BLOCK: tl.constexpr):
    # (1 - tanh(f)) a + tanh(n) b in float32, f and n [tokens, H] the forget and gain products
    token = tl.program_id(0).to(tl.int64)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    a = to_float(tl.load(A + token * STRIDE_A + columns))
    b = to_float(tl.load(B + token * STRIDE_B + columns))
    forget = libdevice.tanh(tl.load(F + token * H + columns))
    gain = libdevice.tanh(tl.load(N + token * H + columns))
    tl.store(Y + token * H + columns, (1.0 - forget) * a + gain * b)


def _rows(x):
    if x.stride(-1) != 1:
        x = x.contiguous()
    return x, x.stride(0)


def message_gate(hidden, message, down, forget, gain, eps):
    """inkling2.message_gate for a pass of any rows, its three products through the library as the PyTorch path computes them:
    (1 - tanh(F g)) hidden + tanh(N g) message in float32, g = down [rms(hidden) | rms(message)] (its inputs and itself rounded to BF16)."""
    (hidden, stride_a), (message, stride_b) = _rows(hidden), _rows(message)
    tokens, width = hidden.shape
    inputs = torch.empty(tokens, 2 * width, dtype=torch.bfloat16, device=hidden.device)
    bits = reads_bits(hidden.device)
    _gate_inputs[(tokens,)](bits_view(hidden, bits), bits_view(message, bits), bits_view(inputs, bits), eps, stride_a, stride_b, width, _block(width), num_warps=4, enable_fp_fusion=False)
    g = torch.nn.functional.linear(inputs, down).float()
    inputs = None
    out = torch.empty(tokens, width, dtype=torch.float32, device=hidden.device)
    f, n = torch.nn.functional.linear(g, forget.float()), torch.nn.functional.linear(g, gain.float())
    g = None
    _gate_combine[(tokens, width // 256)](bits_view(hidden, bits), bits_view(message, bits), f, n, out, stride_a, stride_b, width, 256, num_warps=2, enable_fp_fusion=False)
    return out


@triton.jit(do_not_specialize=("TOTAL",), do_not_specialize_on_alignment=("IDS", "COUNTS"))
def _bin_count(IDS, COUNTS, TOTAL, BLOCK: tl.constexpr):
    # the pairs routed to each expert (integer atomics: the counts do not depend on their order)
    pairs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inside = pairs < TOTAL
    tl.atomic_add(COUNTS + tl.load(IDS + pairs, mask=inside, other=0).to(tl.int32), 1, mask=inside)


@triton.jit(do_not_specialize=("COUNT",), do_not_specialize_on_alignment=("COUNTS", "FIRST", "CURSOR", "OWNERS", "PAIRS"))
def _bin_plan(COUNTS, FIRST, CURSOR, OWNERS, PAIRS, COUNT, E: tl.constexpr, EXPERTS: tl.constexpr, GROUP: tl.constexpr, BLOCK: tl.constexpr, OWNER_BLOCK: tl.constexpr):
    # one program: each expert's first group (its groups one after the other, in expert order), the owner of each group (E past the last), the pair table emptied
    experts = tl.arange(0, EXPERTS)
    real = experts < E
    needed = (tl.load(COUNTS + experts, mask=real, other=0) + GROUP - 1) // GROUP
    ends = tl.cumsum(needed, 0)
    tl.store(FIRST + experts, ends - needed, mask=real)
    tl.store(CURSOR + experts, tl.zeros((EXPERTS,), tl.int32), mask=real)
    ends = tl.where(real, ends, 1 << 30)
    for start in range(0, COUNT, OWNER_BLOCK):  # (OWNER_BLOCK x EXPERTS comparisons at a time)
        groups = start + tl.arange(0, OWNER_BLOCK)
        owner = tl.sum((ends[None, :] <= groups[:, None]).to(tl.int32), axis=1)  # the expert whose groups hold it (searchsorted, right)
        tl.store(OWNERS + groups, owner, mask=groups < COUNT)
    for start in range(0, COUNT * GROUP, BLOCK):
        places = start + tl.arange(0, BLOCK)
        tl.store(PAIRS + places, tl.full((BLOCK,), -1, tl.int32), mask=places < COUNT * GROUP)


@triton.jit(do_not_specialize=("TOTAL",), do_not_specialize_on_alignment=("IDS", "FIRST", "CURSOR", "PAIRS"))
def _bin_place(IDS, FIRST, CURSOR, PAIRS, TOTAL, GROUP: tl.constexpr, BLOCK: tl.constexpr):
    # each pair into the next place of its expert's groups (the order of the pairs within an expert follows the atomics: a pair's result does not depend on its place)
    pairs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    inside = pairs < TOTAL
    expert = tl.load(IDS + pairs, mask=inside, other=0).to(tl.int32)
    slot = tl.atomic_add(CURSOR + expert, 1, mask=inside)
    place = (tl.load(FIRST + expert, mask=inside, other=0) + slot // GROUP) * GROUP + slot % GROUP
    tl.store(PAIRS + place, pairs.to(tl.int32), mask=inside)


@triton.jit(do_not_specialize=("scaling",), do_not_specialize_on_alignment=("L", "B", "IDS", "WEIGHTS"))
def _route_top(L, B, IDS, WEIGHTS, scaling, EXPERTS: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    # sigmoid of a token's float32 router logits (as torch.sigmoid: 1 / (1 + exp(-x))), the K largest score + bias, the highest first (the lowest id among equals), and their
    # scores over their sum, times scaling (row_fused_triton._router_top on logits)
    token = tl.program_id(0).to(tl.int64)
    experts = tl.arange(0, BLOCK)
    inside = experts < EXPERTS
    scores = tl.div_rn(1.0, 1.0 + libdevice.exp(-tl.load(L + token * EXPERTS + experts, mask=inside, other=0.0)))
    biased = tl.where(inside, scores + tl.load(B + experts, mask=inside, other=0.0), float("-inf"))
    slots = tl.arange(0, K)
    ids = tl.zeros((K,), tl.int32)
    weights = tl.zeros((K,), tl.float32)
    for slot in tl.static_range(K):
        best = tl.argmax(biased, 0)
        ids = tl.where(slots == slot, best, ids)
        weights = tl.where(slots == slot, tl.sum(tl.where(experts == best, scores, 0.0)), weights)
        biased = tl.where(experts == best, float("-inf"), biased)
    tl.store(IDS + token * K + slots, ids)
    tl.store(WEIGHTS + token * K + slots, tl.div_rn(weights, tl.sum(weights) + 1e-20) * scaling)


def route(x, weight, bias, top_k, scaling):
    """(ids int32 [tokens, top_k], float32 weights) of a prompt pass: the router's float32 product through the library (as moe.sigmoid_bias_topk's caller computes it), then the
    selection in one kernel; the experts are listed highest first (the PyTorch top-k lists them in its own order: the slot sums then add in another order)."""
    logits = torch.nn.functional.linear(x.float(), weight.float())
    tokens, experts = logits.shape
    ids = torch.empty(tokens, top_k, dtype=torch.int32, device=x.device)
    weights = torch.empty(tokens, top_k, dtype=torch.float32, device=x.device)
    _route_top[(tokens,)](logits, bias, ids, weights, scaling, experts, top_k, _block(experts), num_warps=1, enable_fp_fusion=False)
    return ids, weights


MIX_STATES = 8  # states a learned query reads at most (the block boundaries before a token and its running state)
MIX_ROWS = 256  # passes of up to this many tokens take the mix kernel (its float64 sums cost little there; longer passes are not bound by their launches)


@triton.jit(do_not_specialize=("STATES", "eps", "STRIDE", "QUERIES"), do_not_specialize_on_alignment=("S", "P", "QUERY", "Y"))
def _mix64(S, P, QUERY, Y, STATES, eps, STRIDE, QUERIES, H: tl.constexpr, Q: tl.constexpr, HAS_PARTIAL: tl.constexpr, SLOTS: tl.constexpr, CHUNK: tl.constexpr):
    # row_fused_triton._mix with its sums in float64 (scores, the running state's norm, softmax, weighted sum), stored in float32: less error than the PyTorch path's float32
    # products. The width by chunks of CHUNK columns (float64 tiles of the whole width would not fit the registers).
    token = tl.program_id(0).to(tl.int64)
    slots = tl.arange(0, SLOTS)[None, :]
    rows = tl.arange(0, Q)
    scores = tl.zeros((Q, SLOTS), tl.float64)
    square = tl.zeros((CHUNK,), tl.float64)
    for start in range(0, H, CHUNK):
        columns = start + tl.arange(0, CHUNK)
        inside = columns < H
        query = tl.load(QUERY + rows[:, None] * H + columns[None, :], mask=(rows < QUERIES)[:, None] & inside[None, :], other=0.0).to(tl.float64)
        for state in range(STATES):
            values = tl.load(S + (token * STATES + state) * H + columns, mask=inside, other=0.0).to(tl.float64)
            scores += tl.where(slots == state, tl.sum(query * values[None, :], axis=1)[:, None], 0.0)
        if HAS_PARTIAL:
            partial = tl.load(P + token * STRIDE + columns, mask=inside, other=0.0)
            if partial.dtype == tl.int16:  # BF16 bits (reads_bits)
                partial = to_float(partial)
            partial = partial.to(tl.float64)
            square += partial * partial
            scores += tl.where(slots == STATES, tl.sum(query * partial[None, :], axis=1)[:, None], 0.0)
    inverse = 1.0 / tl.sqrt(tl.sum(square) / H + eps)  # the running state's norm (its products above taken unnormalized)
    if HAS_PARTIAL:
        scores = tl.where(slots == STATES, scores * inverse, scores)
    used = slots < STATES + HAS_PARTIAL
    scores = tl.where(used, scores / tl.sqrt(H * 1.0), float("-inf"))
    weights = tl.exp(scores - tl.max(scores, axis=1)[:, None])
    weights = weights / tl.sum(weights, axis=1)[:, None]
    for start in range(0, H, CHUNK):
        columns = start + tl.arange(0, CHUNK)
        inside = columns < H
        out = tl.zeros((Q, CHUNK), tl.float64)
        for state in range(STATES):
            values = tl.load(S + (token * STATES + state) * H + columns, mask=inside, other=0.0).to(tl.float64)
            out += tl.sum(tl.where(slots == state, weights, 0.0), axis=1)[:, None] * values[None, :]
        if HAS_PARTIAL:
            partial = tl.load(P + token * STRIDE + columns, mask=inside, other=0.0)
            if partial.dtype == tl.int16:
                partial = to_float(partial)
            partial = partial.to(tl.float64) * inverse
            out += tl.sum(tl.where(slots == STATES, weights, 0.0), axis=1)[:, None] * partial[None, :]
        tl.store(Y + (token * QUERIES + rows)[:, None] * H + columns[None, :], out.to(tl.float32), mask=(rows < QUERIES)[:, None] & inside[None, :])


def mix(query, states, partial=None, eps=0.0):
    """inkling2.mix of a short prompt pass (float32 [tokens, queries, width]) with float64 sums: softmax((query @ states^T) * width^-0.5) @ states over the float32 states
    [tokens, states, width] of each token and, with partial [tokens, width], the rms of that running state as one more state. query: float32 [queries, width]."""
    tokens, count, width = states.shape
    queries = query.shape[0]
    if count + (partial is not None) > MIX_STATES or states.dtype != torch.float32 or query.dtype != torch.float32:
        raise ValueError("The fused mix reads up to 8 float32 states per token")
    states = states if states.is_contiguous() else states.contiguous()
    if partial is not None:
        partial, stride = _rows(partial)
    out = torch.empty(tokens, queries, width, dtype=torch.float32, device=states.device)
    _mix64[(tokens,)](states, states if partial is None else bits_view(partial, reads_bits(states.device)), query.contiguous(), out, count, eps, 0 if partial is None else stride, queries, width, triton.next_power_of_2(queries), partial is not None,
                      MIX_STATES, 256, num_warps=4)
    return out


EXPERT_TILES = {}  # by approximate mode: (rows, pairs, warps, stages) of the grouped kernels for these passes; absent: expert_rows_triton.GROUP_TILES and one stage
BIN_KERNELS = True  # the groups of DeviceGroupedPass by three Triton launches (False: about 20 PyTorch operations, a stable sort; the same results)


class DeviceGroupedPass:
    """expert_rows_triton.GroupedPass for a bank whose experts all have a GPU copy, its groups built on the device (no host synchronization): the (token, slot) pairs sorted by
    expert (counting sort through a stable argsort), cut in groups of GROUP pairs, the groups of an expert one after the other in expert order, as GroupedPass builds them on the host.
    The launches cover the most groups a routing can need (pairs / GROUP + experts); the groups past the last are given the extra all-zero row of the pointer table, which the
    kernels skip. A pair's result does not depend on its group or place in it (each column of a tensor-core product is its own), so the output is GroupedPass's, bit for bit.
    run(pointers): pointers int64 [experts + 1, columns] with a zero last row; finish(): the float32 [tokens, hidden] sum. ids int32 / int64 [tokens, topk] on the device."""
    def __init__(self, kind, x, ids, probabilities, intermediate, experts, caps=None):
        from shared.kernels import expert_rows_triton as rows, kernel_policy
        if x.dtype != torch.bfloat16:
            raise ValueError(f"The grouped expert kernels take BF16 activations, not {x.dtype}")
        self.rows_module, self.kind, self.x, self.intermediate, self.caps = rows, kind, x.contiguous(), intermediate, caps
        self.probabilities = probabilities.reshape(-1).float().contiguous()
        self.tokens, self.hidden = x.shape
        self.topk = ids.shape[-1]
        approximate = kernel_policy.allow_approximate()
        self.rows, self.group, self.warps, self.stages = EXPERT_TILES.get(approximate) or (*rows.group_tiles(x.device, approximate), 1)
        self.ids = ids.reshape(-1).long()
        self.experts = experts
        self.count = -(-self.ids.numel() // self.group) + min(self.ids.numel(), experts)  # the groups of any routing: each expert adds at most one partial group
        self.mid = torch.empty(self.count * self.group, intermediate, dtype=torch.bfloat16 if approximate else torch.float32, device=x.device)
        self.parts = torch.empty(self.tokens * self.topk, self.hidden, dtype=torch.float32, device=x.device)
        self._planned = None

    def groups(self):
        """pairs int32 [count, group] (-1 past an expert's last pair), experts int32 [count] (experts for the groups past the last), counts [experts]."""
        flat, group, experts = self.ids, self.group, self.experts
        if BIN_KERNELS:
            device, total = flat.device, flat.numel()
            counts = torch.zeros(3, experts, dtype=torch.int32, device=device)  # counts, first group, cursor
            owners = torch.empty(self.count, dtype=torch.int32, device=device)
            pairs = torch.empty(self.count * group, dtype=torch.int32, device=device)
            _bin_count[(-(-total // 1024),)](flat, counts[0], total, 1024, num_warps=4)
            _bin_plan[(1,)](counts[0], counts[1], counts[2], owners, pairs, self.count, experts, triton.next_power_of_2(experts), group, 1024, 32, num_warps=4)
            _bin_place[(-(-total // 1024),)](flat, counts[1], counts[2], pairs, total, group, 1024, num_warps=4)
            return pairs.view(self.count, group), owners, counts[0]
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=experts)
        needed = (counts + group - 1) // group
        first = torch.cumsum(needed, 0) - needed  # the first group of each expert
        start = torch.cumsum(counts, 0) - counts  # its first sorted pair
        expert = flat[order]
        rank = torch.arange(flat.numel(), device=flat.device) - start[expert]
        pairs = torch.full((self.count * group,), -1, dtype=torch.int32, device=flat.device)
        pairs[(first[expert] + rank // group) * group + rank % group] = order.to(torch.int32)
        owners = torch.searchsorted(torch.cumsum(needed, 0), torch.arange(self.count, device=flat.device), right=True).to(torch.int32)  # experts for the groups past the last
        return pairs.view(self.count, group), owners, counts

    def run(self, pointers):
        """The routed experts having an address in pointers (a pair is computed by every run whose table gives its expert, the same values each time); returns the pairs per expert (on
        the device). The groups are built at the first run."""
        rows = self.rows_module
        if pointers.shape[0] != self.experts + 1:
            raise ValueError("The device-grouped pass reads a pointer table with one more, all-zero row (padded_pointers)")
        if self._planned is None:
            self._planned = self.groups()
        pairs, owners, counts = self._planned
        rows._check(self.kind, pointers, self.hidden, self.intermediate, self.x)
        cap_gate, cap_up = self.caps or (0.0, 0.0)
        stride = pointers.shape[1]
        rows._gate_up_group[(self.count, self.intermediate // self.rows)](self.x, pairs, owners, pointers, self.mid, 0, stride, cap_gate, cap_up, self.hidden, self.intermediate, self.topk, self.kind,
                                                                           self.rows, self.group, self.caps is not None, num_warps=self.warps, num_stages=self.stages)
        rows._down_group[(self.count, self.hidden // self.rows)](self.mid, pairs, owners, pointers, self.probabilities, self.parts, 0, stride, self.hidden, self.intermediate, self.kind, self.rows,
                                                                 self.group, num_warps=self.warps, num_stages=self.stages)
        return counts

    def finish(self):
        self.mid = self._planned = None
        output = self.parts.view(self.tokens, self.topk, self.hidden).sum(1)
        self.parts = None
        return output


@triton.jit
def _pad_table(T, OUT, ENTRIES, COLUMNS, BLOCK: tl.constexpr):
    # the table's entries, then a row of zeros
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(OUT + index, tl.load(T + index, mask=index < ENTRIES, other=0), mask=index < ENTRIES + COLUMNS)


def padded_pointers(pointers):
    """The pointer table with one more, all-zero row (index experts): the groups past the last of a DeviceGroupedPass. A kernel copies it, as the kernels that use it read it: the
    placement's table stays at its address while the model is loaded (graphs recorded with this copy read the placement of each replay)."""
    rows, columns = pointers.shape
    if not pointers.is_contiguous():
        raise ValueError("padded_pointers takes a contiguous table")
    out = torch.empty(rows + 1, columns, dtype=pointers.dtype, device=pointers.device)
    entries = rows * columns
    _pad_table[(-(-(entries + columns) // 1024),)](pointers, out, entries, columns, 1024, num_warps=4)
    return out


from shared.kernels.triton_compilation_log import install_triton_compilation_logger  # noqa: E402

install_triton_compilation_logger()
