# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Fused row operations of LM decoding passes: a pass of a few tokens through a model made of many small PyTorch operations is bound by
its number of kernels (a few microseconds each inside a CUDA graph, whatever they compute), so each chain below is one kernel.

Every kernel computes in float32 registers, in the order of the PyTorch expression it replaces, and rounds where that expression stores
BF16 (ROUND_IN: an input the expression rounds to BF16 first); reductions sum in another order. Widths (hidden size, head size, taps,
ranks) are architecture constants; rows, tokens and positions are runtime arguments. NVIDIA GPUs from compute capability 8.0 (BF16
conversions); elsewhere the callers keep their PyTorch path."""
from functools import cache

import numpy as np
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@cache
def supported(device):
    device = torch.device(device)
    return device.type == "cuda" and torch.version.hip is None and torch.cuda.get_device_capability(device) >= (8, 0)


def _block(width):
    return max(16, triton.next_power_of_2(width))


@triton.jit
def _bf16(x, ROUND: tl.constexpr):
    """float32 x, through BF16 when the replaced expression stores it there."""
    if ROUND:
        return x.to(tl.bfloat16).to(tl.float32)
    return x.to(tl.float32)


@triton.jit(do_not_specialize=("eps", "STRIDE"), do_not_specialize_on_alignment=("X", "W", "Y"))
def _rms(X, W, Y, eps, STRIDE, N: tl.constexpr, HAS_WEIGHT: tl.constexpr, ROUND_IN: tl.constexpr, BLOCK: tl.constexpr):
    # y = x * rsqrt(mean(x^2) + eps) [* weight], a row of N values per program
    row = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, BLOCK)
    inside = columns < N
    x = _bf16(tl.load(X + row * STRIDE + columns, mask=inside, other=0.0), ROUND_IN)
    y = x * tl.rsqrt(tl.div_rn(tl.sum(x * x), N * 1.0) + eps)
    if HAS_WEIGHT:
        y = tl.load(W + columns, mask=inside, other=0.0).to(tl.float32) * y
    tl.store(Y + row * N + columns, y.to(Y.dtype.element_ty), mask=inside)


def rms(x, eps, weight=None, dtype=torch.float32, round_input=False):
    """x * rsqrt(mean(x^2, -1) + eps) [* weight] in dtype. x: rows of its last dimension, contiguous, or a [rows, width] view with a row stride."""
    width = x.shape[-1]
    if not x.is_contiguous() and not (x.dim() == 2 and x.stride(1) == 1):
        x = x.contiguous()
    out = torch.empty(x.shape, dtype=dtype, device=x.device)
    _rms[(x.numel() // width,)](x, x if weight is None else weight, out, eps, x.stride(0) if x.dim() == 2 else width, width, weight is not None, round_input, _block(width), num_warps=4 if width > 512 else 1, enable_fp_fusion=False)
    return out


@triton.jit(do_not_specialize=("threshold", "STRIDE"), do_not_specialize_on_alignment=("X", "S", "TAU", "Y"))
def _clip(X, S, TAU, Y, threshold, STRIDE, GROUPS: tl.constexpr, N: tl.constexpr, HAS_SCALE: tl.constexpr, HAS_TAU: tl.constexpr, ROUND_IN: tl.constexpr, BLOCK: tl.constexpr):
    # y = l2_clip(x * scale, threshold) rounded to BF16 [then * tau of its token, rounded again]: a group of N values per program
    row = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, BLOCK)
    inside = columns < N
    x = _bf16(tl.load(X + (row // GROUPS) * STRIDE + (row % GROUPS) * N + columns, mask=inside, other=0.0), ROUND_IN)
    if HAS_SCALE:
        x = x * tl.load(S + (row % GROUPS) * N + columns, mask=inside, other=0.0).to(tl.float32)
    y = x * tl.minimum(tl.div_rn(threshold, tl.sqrt_rn(tl.sum(x * x)) + 1e-6), 1.0)
    if HAS_TAU:
        y = y.to(tl.bfloat16).to(tl.float32) * tl.load(TAU + row // GROUPS)
    tl.store(Y + row * N + columns, y.to(Y.dtype.element_ty), mask=inside)


def clip(x, threshold, scale=None, tau=None, dtype=torch.bfloat16, round_input=False):
    """l2_clip(x * scale, threshold) in dtype, x [tokens, groups, width] (a row of groups per token, or a [tokens, groups * width] view with a row stride); tau float32 [tokens]: the BF16 result times tau."""
    tokens, groups, width = x.shape
    if not x.is_contiguous():
        x = x.contiguous()
    out = torch.empty(x.shape, dtype=dtype, device=x.device)
    _clip[(tokens * groups,)](x, x if scale is None else scale, out if tau is None else tau, out, threshold, groups * width, groups, width, scale is not None, tau is not None, round_input, _block(width), num_warps=1, enable_fp_fusion=False)
    return out


@triton.jit(do_not_specialize=("RING", "eps", "threshold", "inverse"), do_not_specialize_on_alignment=("X", "R", "W", "POS", "Y"))
def _causal(X, R, W, DYN, POS, Y, RING, eps, threshold, inverse, C: tl.constexpr, TAPS: tl.constexpr, N: tl.constexpr, POST: tl.constexpr, DYNAMIC: tl.constexpr, BLOCK: tl.constexpr):
    # Causal depthwise convolution with its residual over the rows of a pass and the rows before them, kept by position in the ring R (which takes
    # the rows of the pass); a program: N channels of a token. POST 0: nothing; 1: the BF16 result clipped (l2_clip); 2: normalized (rms).
    # DYNAMIC: the taps W [TAPS, C], by backward distance, plus those of the token and of its group of N channels, DYN [tokens, groups * TAPS].
    token = tl.program_id(0).to(tl.int64)
    channels = tl.program_id(1) * N + tl.arange(0, BLOCK)
    inside = tl.arange(0, BLOCK) < N
    first = tl.load(POS)
    position = first + token
    total = tl.zeros((BLOCK,), tl.float32)
    for tap in tl.static_range(TAPS):
        source = position - tap if DYNAMIC else position - (TAPS - 1) + tap
        if source >= first:
            value = tl.load(X + (source - first) * C + channels, mask=inside, other=0.0).to(tl.float32)
        else:
            value = tl.load(R + (source % RING) * C + channels, mask=inside & (source >= 0), other=0.0).to(tl.float32)
        if DYNAMIC:
            total += value * (tl.load(W + tap * C + channels, mask=inside, other=0.0).to(tl.float32) + tl.load(DYN + (token * (C // N) + tl.program_id(1)) * TAPS + tap).to(tl.float32))
        else:
            total += value * tl.load(W + channels * TAPS + tap, mask=inside, other=0.0).to(tl.float32)
    x = tl.load(X + token * C + channels, mask=inside, other=0.0)
    y = (total + x.to(tl.float32)) * inverse  # PyTorch divides by a scalar this way
    tl.store(R + (position % RING) * C + channels, x.to(R.dtype.element_ty), mask=inside)
    if POST == 1:
        y = y.to(tl.bfloat16).to(tl.float32)
        y = y * tl.minimum(tl.div_rn(threshold, tl.sqrt_rn(tl.sum(y * y)) + 1e-6), 1.0)
    if POST == 2:
        y = y.to(tl.bfloat16).to(tl.float32)
        y = y * tl.rsqrt(tl.div_rn(tl.sum(y * y), N * 1.0) + eps)
    tl.store(Y + token * C + channels, y.to(Y.dtype.element_ty), mask=inside)


CAUSAL_ROWS = 16  # tokens of a pass whose convolutions run fused: its rows must not reach the ring rows it reads


def causal_conv(x, ring, weight, positions, dtype, group=None, post=0, eps=0.0, threshold=0.0, dynamic=None):
    """((taps over the rows before each token, the last on the token itself) * weight + x) / sqrt(taps + 1) in dtype, x [tokens, channels] of consecutive
    positions; ring [rows, channels] keeps the inputs by position (zeros before position 0) and takes x. weight [channels, 1, taps]. group: channels of a
    program (a head: post 1 clips its BF16 result to threshold, post 2 normalizes it with eps). dynamic [tokens, groups * taps]: weight is [taps, channels]
    by backward distance, a token's group adds its own taps, and the divisor is sqrt(2 taps + 1)."""
    tokens, channels = x.shape
    taps = weight.shape[-1] if dynamic is None else weight.shape[0]
    if tokens > min(CAUSAL_ROWS, ring.shape[0] - taps + 1) or not x.is_contiguous():
        raise ValueError("The fused convolution takes the contiguous rows of a short pass")
    group = group or min(channels, 256)
    out = torch.empty(x.shape, dtype=dtype, device=x.device)
    inverse = float(np.float32(1.0) / np.float32(((taps + 1) if dynamic is None else (2 * taps + 1)) ** 0.5))
    _causal[(tokens, channels // group)](x, ring, weight, x if dynamic is None else dynamic, positions, out, ring.shape[0], eps, threshold, inverse, channels, taps, group, post, dynamic is not None, _block(group),
                                         num_warps=1, enable_fp_fusion=False)
    torch.autograd.graph.increment_version(ring)
    return out


@triton.jit(do_not_specialize=("eps", "STRIDE_A", "STRIDE_B"), do_not_specialize_on_alignment=("A", "B", "W", "G"))
def _gate_down(A, B, W, G, eps, STRIDE_A, STRIDE_B, H: tl.constexpr, RANK: tl.constexpr, BLOCK: tl.constexpr):
    # g = W [rank, 2H] (BF16) times [rms(a) | rms(b)] rounded to BF16, itself rounded to BF16: a rank row of a token per program
    token = tl.program_id(0).to(tl.int64)
    rank = tl.program_id(1)
    columns = tl.arange(0, BLOCK)
    inside = columns < H
    a = tl.load(A + token * STRIDE_A + columns, mask=inside, other=0.0).to(tl.float32)
    b = tl.load(B + token * STRIDE_B + columns, mask=inside, other=0.0).to(tl.float32)
    a = (a * tl.rsqrt(tl.div_rn(tl.sum(a * a), H * 1.0) + eps)).to(tl.bfloat16).to(tl.float32)
    b = (b * tl.rsqrt(tl.div_rn(tl.sum(b * b), H * 1.0) + eps)).to(tl.bfloat16).to(tl.float32)
    total = tl.sum(a * tl.load(W + rank * (2 * H) + columns, mask=inside, other=0.0).to(tl.float32)) + tl.sum(b * tl.load(W + rank * (2 * H) + H + columns, mask=inside, other=0.0).to(tl.float32))
    tl.store(G + token * RANK + rank, total.to(tl.bfloat16).to(tl.float32))


@triton.jit(do_not_specialize=("STRIDE_A", "STRIDE_B"), do_not_specialize_on_alignment=("A", "B", "G", "F", "N", "Y"))
def _gate_mix(A, B, G, F, N, Y, STRIDE_A, STRIDE_B, H: tl.constexpr, RANK: tl.constexpr, ROWS: tl.constexpr):
    # y = (1 - tanh(F g)) a + tanh(N g) b, float32: ROWS channels of a token per program
    token = tl.program_id(0).to(tl.int64)
    channels = tl.program_id(1) * ROWS + tl.arange(0, ROWS)
    g = tl.load(G + token * RANK + tl.arange(0, RANK))[None, :]
    place = channels[:, None] * RANK + tl.arange(0, RANK)[None, :]
    forget = libdevice.tanh(tl.sum(tl.load(F + place).to(tl.float32) * g, axis=1))
    gain = libdevice.tanh(tl.sum(tl.load(N + place).to(tl.float32) * g, axis=1))
    a = tl.load(A + token * STRIDE_A + channels).to(tl.float32)
    b = tl.load(B + token * STRIDE_B + channels).to(tl.float32)
    tl.store(Y + token * H + channels, (1.0 - forget) * a + gain * b)


def _rows(x):
    """A [tokens, width] tensor whose rows are contiguous, and their stride."""
    if x.stride(-1) != 1:
        x = x.contiguous()
    return x, x.stride(0)


def message_gate(hidden, message, down, forget, gain, eps):
    """(1 - tanh(forget g)) hidden + tanh(gain g) message in float32, g = down [rms(hidden) | rms(message)] with its inputs and itself rounded to BF16.
    hidden, message [tokens, width] (rows with a stride); down [rank, 2 width], forget and gain [width, rank] in BF16."""
    (hidden, stride_a), (message, stride_b) = _rows(hidden), _rows(message)
    tokens, width = hidden.shape
    rank = down.shape[0]
    g = torch.empty(tokens, rank, dtype=torch.float32, device=hidden.device)
    out = torch.empty(tokens, width, dtype=torch.float32, device=hidden.device)
    _gate_down[(tokens, rank)](hidden, message, down, g, eps, stride_a, stride_b, width, rank, _block(width), num_warps=4, enable_fp_fusion=False)
    _gate_mix[(tokens, width // 64)](hidden, message, g, forget, gain, out, stride_a, stride_b, width, rank, 64, num_warps=4, enable_fp_fusion=False)
    return out


@triton.jit(do_not_specialize=("STRIDE",), do_not_specialize_on_alignment=("X", "W", "S"))
def _router_scores(X, W, S, STRIDE, H: tl.constexpr, EXPERTS: tl.constexpr, BLOCK: tl.constexpr):
    # sigmoid(W x), float32: an expert of a token per program
    token = tl.program_id(0).to(tl.int64)
    expert = tl.program_id(1)
    columns = tl.arange(0, BLOCK)
    inside = columns < H
    total = tl.sum(tl.load(X + token * STRIDE + columns, mask=inside, other=0.0).to(tl.float32) * tl.load(W + expert * H + columns, mask=inside, other=0.0).to(tl.float32))
    tl.store(S + token * EXPERTS + expert, tl.div_rn(1.0, 1.0 + libdevice.exp(-total)))


@triton.jit(do_not_specialize=("scaling",), do_not_specialize_on_alignment=("S", "B", "IDS", "WEIGHTS"))
def _router_top(S, B, IDS, WEIGHTS, scaling, EXPERTS: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    # the K largest score + bias of a token, the highest first (the lowest id among equals), and their scores over their sum, times scaling
    token = tl.program_id(0).to(tl.int64)
    experts = tl.arange(0, BLOCK)
    inside = experts < EXPERTS
    scores = tl.load(S + token * EXPERTS + experts, mask=inside, other=0.0)
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
    """(ids int32 [tokens, top_k], float32 weights): the experts of the top_k largest sigmoid(weight x) + bias, their sigmoid scores normalized to scaling.
    x [tokens, width] and weight [experts, width] in BF16, bias float32 [experts]."""
    x, stride = _rows(x)
    tokens, width = x.shape
    experts = weight.shape[0]
    scores = torch.empty(tokens, experts, dtype=torch.float32, device=x.device)
    ids = torch.empty(tokens, top_k, dtype=torch.int32, device=x.device)
    weights = torch.empty(tokens, top_k, dtype=torch.float32, device=x.device)
    _router_scores[(tokens, experts)](x, weight, scores, stride, width, experts, _block(width), num_warps=4, enable_fp_fusion=False)
    _router_top[(tokens,)](scores, bias, ids, weights, scaling, experts, top_k, _block(experts), num_warps=1, enable_fp_fusion=False)
    return ids, weights


MIX_STATES = 8  # states a learned query reads at most (the block boundaries before a token and its running state)


@triton.jit(do_not_specialize=("STATES", "eps", "scale", "STRIDE", "QUERIES"), do_not_specialize_on_alignment=("S", "P", "QUERY", "Y"))
def _mix(S, P, QUERY, Y, STATES, eps, scale, STRIDE, QUERIES, H: tl.constexpr, Q: tl.constexpr, HAS_PARTIAL: tl.constexpr, SLOTS: tl.constexpr, BLOCK: tl.constexpr):
    # softmax(query . states * scale) @ states for the QUERIES queries of a token (Q: their count rounded up to a power of two): its STATES float32 states [H], then (HAS_PARTIAL) the rms of its running state
    token = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, BLOCK)
    inside = columns < H
    slots = tl.arange(0, SLOTS)[None, :]
    real = (tl.arange(0, Q) < QUERIES)[:, None] & inside[None, :]
    query = tl.load(QUERY + tl.arange(0, Q)[:, None] * H + columns[None, :], mask=real, other=0.0).to(tl.float32)
    scores = tl.full((Q, SLOTS), float("-inf"), tl.float32)
    for state in range(STATES):
        values = tl.load(S + (token * STATES + state) * H + columns, mask=inside, other=0.0).to(tl.float32)
        scores = tl.where(slots == state, tl.sum(query * values[None, :], axis=1)[:, None], scores)
    partial = tl.zeros((BLOCK,), tl.float32)
    if HAS_PARTIAL:
        partial = tl.load(P + token * STRIDE + columns, mask=inside, other=0.0).to(tl.float32)
        partial = partial * tl.rsqrt(tl.div_rn(tl.sum(partial * partial), H * 1.0) + eps)
        scores = tl.where(slots == STATES, tl.sum(query * partial[None, :], axis=1)[:, None], scores)
    scores = scores * scale
    weights = libdevice.exp(scores - tl.max(scores, axis=1)[:, None])
    weights = tl.div_rn(weights, tl.sum(weights, axis=1)[:, None])
    out = tl.zeros((Q, BLOCK), tl.float32)
    for state in range(STATES):
        values = tl.load(S + (token * STATES + state) * H + columns, mask=inside, other=0.0).to(tl.float32)
        out += tl.sum(tl.where(slots == state, weights, 0.0), axis=1)[:, None] * values[None, :]
    if HAS_PARTIAL:
        out += tl.sum(tl.where(slots == STATES, weights, 0.0), axis=1)[:, None] * partial[None, :]
    tl.store(Y + (token * QUERIES + tl.arange(0, Q))[:, None] * H + columns[None, :], out, mask=real)


def mix(query, states, partial=None, eps=0.0):
    """float32 [tokens, queries, width]: softmax((query @ states^T) * width^-0.5) @ states over the float32 states [tokens, states, width] of each token, then, with
    partial [tokens, width], the rms of that running state as one more state. query: float32 [queries, width]."""
    tokens, count, width = states.shape
    queries = query.shape[0]
    if count + (partial is not None) > MIX_STATES or states.dtype != torch.float32 or query.dtype != torch.float32:
        raise ValueError("The fused mix reads up to 8 float32 states per token")
    states = states if states.is_contiguous() else states.contiguous()
    if partial is not None:
        partial, stride = _rows(partial)
    out = torch.empty(tokens, queries, width, dtype=torch.float32, device=states.device)
    _mix[(tokens,)](states, states if partial is None else partial, query, out, count, eps, float(np.float32(width ** -0.5)), 0 if partial is None else stride, queries, width, triton.next_power_of_2(queries), partial is not None, MIX_STATES, _block(width),
                    num_warps=8, enable_fp_fusion=False)
    return out


from shared.kernels.triton_compilation_log import install_triton_compilation_logger  # noqa: E402

install_triton_compilation_logger()
