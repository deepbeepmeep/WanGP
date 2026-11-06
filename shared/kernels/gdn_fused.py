# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
# The recurrence is adapted from flash-linear-attention (https://github.com/fla-org/flash-linear-attention), Copyright (c) 2023-2026 Songlin Yang, Yu Zhang,
# Zhiyuan Li, MIT License (see docs/third_party_licenses/MIT.txt), with ideas of Strata (MIT License, https://github.com/Niko1221/Strata).
"""The Gated DeltaNet mixer of a decoding pass or a speculative verification, between its projections: the short convolution with the alpha / beta gates in one launch, the
recurrence, then its output norm and gate. They exchange FP32 values (normalized queries and keys, values, gates) instead of BF16 ones, and a verification keeps them for the
replay of an accepted prefix. One compilation per kernel and model: token counts are not specialized."""
import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

ENABLED = "gdn" not in os.environ.get("WGP_FN_DISABLE", "").split(",")
STEP_COLUMNS, STEP_WARPS = 16, 2  # value columns per program of the recurrence, measured on sm_120
STATE_DTYPE = torch.float32  # a BF16 state rounded at each token drifts with the square root of the tokens (2.5 % of the output after 5000)


@triton.jit
def _rounded(x, pointer):  # x rounded to the dtype of pointer's tensor, as stored there
    return x.to(pointer.dtype.element_ty).to(tl.float32)


@triton.jit(do_not_specialize=("T",), do_not_specialize_on_alignment=("T",))
def _conv_gates_kernel(QKV, X, HISTORY, CONV_W, WA, WB, SSM_A, SSM_DT, H, GATE, BETA, SNAPSHOTS, T,
                       C: tl.constexpr, N: tl.constexpr, HV: tl.constexpr, QK_BLOCKS: tl.constexpr, BC: tl.constexpr, BN: tl.constexpr, CHUNKS: tl.constexpr):
    program = tl.program_id(0)
    if program < HV:
        columns = tl.arange(0, BN)
        for token in range(T):
            a = tl.zeros((BN,), dtype=tl.float32)
            b = tl.zeros((BN,), dtype=tl.float32)
            for chunk in tl.static_range(CHUNKS):
                offset = chunk * BN + columns
                x = tl.load(X + token * N + offset).to(tl.float32)
                a += tl.load(WA + program * N + offset).to(tl.float32) * x
                b += tl.load(WB + program * N + offset).to(tl.float32) * x
            alpha = tl.sum(a) + tl.load(SSM_DT + program).to(tl.float32)
            softplus = tl.where(alpha > 20., alpha, libdevice.log1p(libdevice.exp(alpha)))
            tl.store(GATE + token * HV + program, libdevice.exp(tl.load(SSM_A + program).to(tl.float32) * softplus))
            tl.store(BETA + token * HV + program, tl.div_rn(1.0, 1.0 + libdevice.exp(-tl.sum(b))))
    else:
        block = program - HV
        channels = block * BC + tl.arange(0, BC)
        first = channels * 4
        w0, w1 = tl.load(CONV_W + first).to(tl.float32), tl.load(CONV_W + first + 1).to(tl.float32)
        w2, w3 = tl.load(CONV_W + first + 2).to(tl.float32), tl.load(CONV_W + first + 3).to(tl.float32)
        h0, h1 = tl.load(HISTORY + first).to(tl.float32), tl.load(HISTORY + first + 1).to(tl.float32)
        h2, h3 = tl.load(HISTORY + first + 2).to(tl.float32), tl.load(HISTORY + first + 3).to(tl.float32)
        for token in range(T):
            x = tl.load(QKV + token * C + channels).to(tl.float32)
            y = w0 * h1 + w1 * h2 + w2 * h3 + w3 * x
            y = tl.div_rn(y, 1.0 + libdevice.exp(-y))
            if block < QK_BLOCKS:
                y = y * tl.div_rn(1.0, tl.sqrt_rn(tl.sum(y * y) + 1e-6))
            tl.store(H + token * C + channels, y)
            h0, h1, h2, h3 = h1, h2, h3, _rounded(x, HISTORY)
            if token + 1 < T:  # what a commit of this prefix restores
                snapshot = (token * C + channels) * 4
                tl.store(SNAPSHOTS + snapshot, h0)
                tl.store(SNAPSHOTS + snapshot + 1, h1)
                tl.store(SNAPSHOTS + snapshot + 2, h2)
                tl.store(SNAPSHOTS + snapshot + 3, h3)
        tl.store(HISTORY + first, h0)
        tl.store(HISTORY + first + 1, h1)
        tl.store(HISTORY + first + 2, h2)
        tl.store(HISTORY + first + 3, h3)


def conv_gates(qkv, x, history, conv_weight, weight_alpha, weight_beta, ssm_a, ssm_dt, h, gate, beta, snapshots):
    """qkv, x: [tokens, columns]. Writes h [tokens, channels] FP32 (queries and keys normalized), gate (exp of the decay) and beta [tokens, value heads] FP32, updates history in place,
    keeps the history after each token but the last in snapshots. In checkpoint head order; ssm_a and ssm_dt too."""
    tokens, channels = qkv.shape
    value_heads, columns = weight_alpha.shape
    head_dim = 128
    query_key_blocks = (channels - value_heads * head_dim) // head_dim
    block_columns = 512
    _conv_gates_kernel[(value_heads + channels // head_dim,)](
        qkv, x, history, conv_weight, weight_alpha, weight_beta, ssm_a, ssm_dt, h, gate, beta, snapshots, tokens,
        channels, columns, value_heads, query_key_blocks, head_dim, block_columns, columns // block_columns, num_warps=4)
    torch.autograd.graph.increment_version(history)


@triton.jit(do_not_specialize=("T",), do_not_specialize_on_alignment=("T",))
def _step_kernel(H, GATE, BETA, STATE, PRE, OUT, T,
                 C: tl.constexpr, KEY_DIM: tl.constexpr, HV: tl.constexpr, HK: tl.constexpr, DK: tl.constexpr, DV: tl.constexpr, BV: tl.constexpr, OUTPUTS: tl.constexpr):
    block, head = tl.program_id(0), tl.program_id(1)
    repeats = HV // HK
    key_head = head // repeats
    checkpoint_head = (head % repeats) * HK + key_head
    keys = tl.arange(0, DK)
    values = block * BV + tl.arange(0, BV)
    index = head * DK * DV + keys[:, None] * DV + values[None, :]
    if OUTPUTS:
        raw = tl.load(STATE + index)
        if T > 1:  # what the replay of an accepted prefix starts from
            tl.store(PRE + index, raw)
    else:
        raw = tl.load(PRE + index)
    state = raw.to(tl.float32)
    for token in range(T):
        row = token * C
        k = tl.load(H + row + KEY_DIM + key_head * DK + keys)
        v = tl.load(H + row + 2 * KEY_DIM + checkpoint_head * DV + values)
        state *= tl.load(GATE + token * HV + checkpoint_head)
        v = tl.load(BETA + token * HV + checkpoint_head) * (v - tl.sum(state * k[:, None], 0))
        state += k[:, None] * v[None, :]
        if OUTPUTS:
            q = tl.load(H + row + key_head * DK + keys)
            tl.store(OUT + token * HV * DV + checkpoint_head * DV + values, tl.sum(state * q[:, None], 0))
    tl.store(STATE + index, state)


@triton.jit(do_not_specialize=("eps",))
def _out_norm_kernel(OUT, Z, GAMMA, Y, eps, DK: tl.constexpr, DV: tl.constexpr):
    row = tl.program_id(0)
    values = tl.arange(0, DV)
    o = tl.load(OUT + row * DV + values) * DK ** -0.5
    scale = tl.div_rn(1.0, tl.sqrt_rn(tl.sum(o * o) / DV + eps))
    gate = tl.load(Z + row * DV + values).to(tl.float32)
    tl.store(Y + row * DV + values, tl.div_rn(o * scale * tl.load(GAMMA + values).to(tl.float32), 1.0 + libdevice.exp(-gate)))


def step_norm(h, gate, beta, z, gamma, state, pre, y, eps):
    """h, gate, beta from conv_gates. state [1, value heads, key dim, value dim] (grouped heads) in place, keeping the state before a verification in pre; y [tokens, value heads * value dim] in checkpoint order."""
    tokens, channels = h.shape
    _, value_heads, dim_k, dim_v = state.shape
    key_dim = (channels - value_heads * dim_v) // 2
    _step_kernel[(dim_v // STEP_COLUMNS, value_heads)](h, gate, beta, state, pre, y, tokens, channels, key_dim, value_heads, key_dim // dim_k, dim_k, dim_v, STEP_COLUMNS, True, num_warps=STEP_WARPS)
    _out_norm_kernel[(tokens * value_heads,)](y, z, gamma, y, eps, dim_k, dim_v, num_warps=2)
    torch.autograd.graph.increment_version(state)


def step_replay(h, gate, beta, state, pre, tokens):
    """The state after the first tokens of the last verification, from the state before it."""
    _, value_heads, dim_k, dim_v = state.shape
    channels = h.shape[-1]
    key_dim = (channels - value_heads * dim_v) // 2
    _step_kernel[(dim_v // STEP_COLUMNS, value_heads)](h, gate, beta, state, pre, None, tokens, channels, key_dim, value_heads, key_dim // dim_k, dim_k, dim_v, STEP_COLUMNS, False, num_warps=STEP_WARPS)
    torch.autograd.graph.increment_version(state)


class FusedReplay:
    """What a speculative verification leaves for the replay of an accepted prefix: the state before it, and the values of its tokens as the kernels exchange them. The values are not copied:
    the convolution writes them here."""

    def __init__(self, state, channels, tokens):
        value_heads = state.shape[1]
        self.tokens = tokens
        self.pre = torch.empty_like(state)
        self.h = state.new_empty((tokens, channels), dtype=torch.float32)
        self.gate = state.new_empty((tokens, value_heads), dtype=torch.float32)
        self.beta = state.new_empty((tokens, value_heads), dtype=torch.float32)

    def fits(self, state, tokens, *_dtypes):
        return self.tokens >= tokens and self.pre.shape == state.shape and self.pre.dtype == state.dtype and self.pre.device == state.device


def _float_output(module, x):
    """A GGUF projection with an FP32 output: the kernel's own product, without its conversion to the compute dtype."""
    weight = module.qweight
    dtype, weight._gguf_default_dtype = weight._gguf_default_dtype, torch.float32
    try:
        return module(x)
    finally:
        weight._gguf_default_dtype = dtype


def eligible(block):
    """A block with separate GGUF projections and BF16 gates, heads as its checkpoint stores them, power-of-two head sizes, a sigmoid output gate. Other blocks keep the established kernels."""
    from shared.qtypes.gguf import GGUFWeightTensor
    return (block.attn_qkv is not None and block.attn_gate is not None and block.ssm_alpha is not None and block.conv_kernel_size == 4 and block.head_k_dim == block.head_v_dim == 128
            and all(isinstance(module.weight, GGUFWeightTensor) for module in (block.attn_qkv, block.attn_gate, block.ssm_out))
            and block._gguf_v_head_reordered and block._gguf_ssm_param_reordered and not block._gguf_interleave_ssm_ab and not block._log_ssm_a
            and getattr(block.ssm_norm, "sigmoid_gate", False))


def mixer(block, x, history, state, verify):
    """The mixer of a decoding pass (one row) or of a speculative verification, on x [1, rows, hidden]: FP32 output [1, rows, hidden]."""
    rows = x.shape[1]
    qkv = _float_output(block.attn_qkv, x)[0]
    z = _float_output(block.attn_gate, x)[0]
    x = x[0]
    replay = block.speculative_replay if verify else None
    if replay is not None:
        h, gate, beta, pre = replay.h[:rows], replay.gate[:rows], replay.beta[:rows], replay.pre  # a window may be shorter than the buffers
    else:
        h = qkv.new_empty(qkv.shape)
        gate = qkv.new_empty((rows, block.num_v_heads))
        beta = qkv.new_empty((rows, block.num_v_heads))
        pre = state
    conv_gates(qkv, x, history, block.ssm_conv1d.weight, block.ssm_alpha.weight, block.ssm_beta.weight, block.ssm_a, block.ssm_dt, h, gate, beta, block.speculative_conv_state_buffer if verify else history)
    qkv = None
    y = z.new_empty(z.shape)
    step_norm(h, gate, beta, z, block.ssm_norm.weight, state, pre, y, block.ssm_norm.eps)
    h = gate = beta = z = None
    return _float_output(block.ssm_out, y[None])


from shared.kernels.triton_compilation_log import install_triton_compilation_logger
install_triton_compilation_logger()
