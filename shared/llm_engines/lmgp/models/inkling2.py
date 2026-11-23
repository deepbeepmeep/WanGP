# Copyright WanGP. Licensed under the WanGP License 2.0.
# Inkling 2 (looped MoE) for the LM engine. Architecture after the Hugging Face Transformers implementation (Apache 2.0) and llama.cpp
# PR #32694 (MIT); numerical specification: lmgp.models.inkling2_core.
"""Inkling 2 in the LM engine: the weights under the checkpoint's own names (24 blocks in one tower that MMGP streams), 48 block
applications, each with its own state (a "slot"): blocks 6-17 run three times.

- The 8 slots of the full-attention blocks (5, 11, 17, 23) write their keys and values into the engine's paged KV cache, in any of its
  layouts, and read the rows of their context back from it (Inkling2PagedState).
- The 40 sliding-window slots keep the keys and values of the last positions in a ring indexed by position (Inkling2WindowState): they
  cost the same VRAM whatever the context.
- The four causal convolutions of a slot keep their last inputs by position too: a rejected speculative draft is simply overwritten.

Every shape of a pass follows its number of tokens and the width of the page table only, and positions are read from the engine's
context as tensors: nothing returns to Python between the token ids and the hidden states. One sequence at a time.
"""
import contextlib
import math
import os

import torch
from torch import nn
from torch.nn import functional as F
from mmgp import cuda_graphs, runtime
from mmgp.expert_compute import RoutedExperts

from ..layers import kv_formats as kvf
from ..layers.attention import Attention, store_kvcache
from ..layers.attention_indexed import IndexedAttention
from ..layers.moe import routed_experts, sigmoid_bias_topk
from ..utils.context import get_context
from .inkling2_core import DRAFT_FEATURES, E2M1, Inkling2Config, l2_clip, rms, softcap

CONV_RING = 32  # the convolution inputs kept by position: the longest verification pass and the taps before it
CONV_FUSED_ROWS = 16  # passes of up to this many rows (decoding, verification) run their static convolutions in one kernel each
WINDOW_SLACK = 32  # rows of a window ring beyond the window: a verification pass must not overwrite the keys its rejected drafts hid
PROMPT_BYTES_PER_TOKEN = 512 << 10  # VRAM of a prompt pass per token beside the ring of its streamed experts (block states, attention and expert scratch)
PROMPT_VRAM_MARGIN = 768 << 20  # left free by a prompt pass beside its ring: a short suffix peaks 340 to 570 MiB over the ring (graphs of a new pass shape, the slots of swapped blocks); while the pass runs its banks are in use and their experts cannot leave for it
ATTENTION_ELEMENTS = 1 << 21  # queries x keys attended together (per head: scores, their float32 softmax and the probabilities)
ROUTED = "routed" not in os.environ.get("WGP_INKLING2_DISABLE", "").split(",")  # the experts in MMGP's routed-expert banks (GPU and CPU kernels on the packed rows, elastic placement); disabled: the PyTorch reference below (Inkling2PackedExperts), for comparisons
STREAMED = ROUTED and "stream" not in os.environ.get("WGP_INKLING2_DISABLE", "").split(",")  # prompt passes: the experts out of VRAM stream through MMGP's ring; disabled, each bank uploads its own (a comparison)
FUSED_RING = "fused_ring" not in os.environ.get("WGP_INKLING2_DISABLE", "").split(",")  # short passes: the key / value convolutions of a window slot write its ring rows (the same bits); disabled: index writes in keep()
FUSED_RESIDUAL = "fused_residual" not in os.environ.get("WGP_INKLING2_DISABLE", "").split(",")  # short passes: the attention and expert convolutions add their block's residual in their own kernel (the same bits); disabled: PyTorch adds
DOORBELL_FREE = "doorbell_free" not in os.environ.get("WGP_INKLING2_DISABLE", "").split(",")  # with every expert in VRAM, decoding passes leave out the CPU expert job (its ask, wait and read kernels); disabled: the job runs in every pass (the same bits)
EXPERT_CACHE = 128  # interim experts: decoded experts kept in VRAM between eager passes, all layers together (16 MiB each in BF16)
MIX_EPS = 1e-6  # of the norm of the states the learned queries read
FUSED = "fused" not in os.environ.get("WGP_INKLING2_DISABLE", "").split(",")  # comparison switch of the fused row kernels (shared.kernels.row_fused_triton)
FUSED_ATTENTION = FUSED and "fused_attention" not in os.environ.get("WGP_INKLING2_DISABLE", "").split(",")  # and of the one-kernel attention of short passes (shared.kernels.ring_attention_triton)
SHARED_TILES = "shared_tiles" not in os.environ.get("WGP_INKLING2_DISABLE", "").split(",")  # the tokens of a short pass share the key / value tiles of the one-kernel attention (one read and dequantization for a verification window); disabled, a program per token (the previous path, other rounding)
FUSED_LINEAR = FUSED and "fused_linear" not in os.environ.get("WGP_INKLING2_DISABLE", "").split(",")  # and of the linears computed row by row (shared.kernels.row_linear_triton)
GROUPED_LINEAR = "grouped_linear" not in os.environ.get("WGP_INKLING2_DISABLE", "").split(",")  # the projections of one attention input in one launch of the row linears (each row as alone; disabled: a launch per weight)
PREFILL_ATTENTION = "prefill_attention" not in os.environ.get("WGP_INKLING2_DISABLE", "").split(",")  # prompt passes: the attention without its scores in memory (shared.kernels.prefill_attention_triton); disabled, the PyTorch attend below
PREFILL_IN_PLACE = True  # the full-attention slots' pages read in place by the prompt attention; False: the rows append gathers (a comparison, the same values)
_KERNELS = _ATTENTION = _LINEAR = _PREFILL_ATTENTION = None  # those modules, once a model with the Triton feature exists


def linear(x, weight):
    """A BF16 linear of the model over the rows of a pass: each row on its own for a short pass (the same sums whatever the rows computed with it), the library's otherwise."""
    if _LINEAR is not None and x.is_cuda and x.shape[0] <= _LINEAR.ROWS and _LINEAR.supported(x.device):
        return _LINEAR.linear(x, weight)
    return F.linear(x, weight)


PREFILL_FUSED = "prefill_fused" not in os.environ.get("WGP_INKLING2_DISABLE", "").split(",")  # prompt passes: convolutions and message gates in fused kernels with the PyTorch path's rounding (shared.kernels.prefill_fused_triton); disabled, the PyTorch expressions
PREFILL_ROWS = "prefill_rows" not in os.environ.get("WGP_INKLING2_DISABLE", "").split(",")  # prompt passes: the router's selection in one kernel (the experts listed highest first: their weights summed in another order); disabled, sigmoid_bias_topk (the learned-query mixes stay in PyTorch: the decoding kernel's error against FP64 is a little larger there)
PREFILL_MIX = "prefill_mix" not in os.environ.get("WGP_INKLING2_DISABLE", "").split(",")  # short prompt passes: the learned-query mixes in one kernel with float64 sums (less error than the PyTorch path's); disabled, PyTorch
_PREFILL_FUSED = None
_PROMPT_REAL = None  # during a padded prompt pass (a graph of a short prompt, Inkling2ForCausalLM.lm_engine_prompt_rows): int32 [1] on the device, its real rows; the others write no state
PROMPT_GRAPH_ROWS = (32, 64, 128, 256)  # the padded sizes of the short prompt passes the engine may replay as graphs (lm_engine_prompt_graph_rows)
PROMPT_GRAPHS = "prompt_graphs" in os.environ.get("WGP_INKLING2_ENABLE", "").split(",")  # short prompt passes replayed as CUDA graphs of padded sizes (opt-in while it is validated); off, eager


def _prefill_fused(x):
    """The fused kernels of prompt passes (more than CONV_FUSED_ROWS rows) on x's device, None where the PyTorch path runs."""
    return _PREFILL_FUSED if _PREFILL_FUSED is not None and PREFILL_FUSED and x.is_cuda and _PREFILL_FUSED.supported(x.device) else None


def _fused(x):
    """The fused row kernels for a pass on x's device, None where the PyTorch path runs."""
    return _KERNELS if _KERNELS is not None and x.is_cuda and _KERNELS.supported(x.device) else None


def norm(x, eps, weight=None, dtype=torch.float32, rounded=False):
    """rms over the last dimension, in dtype (rounded: of x rounded to dtype first)."""
    kernels = _fused(x)
    if kernels is not None:
        return kernels.rms(x, eps, weight, dtype, rounded and x.dtype != dtype)
    return rms(x.to(dtype) if rounded else x, eps, weight).to(dtype)


def clipped(x, threshold, scale=None, tau=None, rounded=False):
    """l2_clip(x * scale, threshold) in the model's dtype (rounded: of x rounded to it first), x [tokens, groups, width]; tau [tokens]: that result times tau, rounded again."""
    dtype, kernels = torch.bfloat16, _fused(x)
    if kernels is not None:
        return kernels.clip(x, threshold, scale, tau, dtype, rounded and x.dtype != dtype)
    x = (x.to(dtype) if rounded else x).float()
    y = l2_clip(x if scale is None else x * scale.float(), threshold).to(dtype)
    return y if tau is None else (y.float() * tau[:, None, None]).to(dtype)


class Inkling2EngineConfig:
    """What the engine reads of a model configuration, with the architecture (core) from the checkpoint's config.json."""
    model_type = "inkling2_loop"
    tie_word_embeddings = False
    dtype = "bfloat16"

    def __init__(self, config):
        self.core = core = Inkling2Config.from_json(config)
        self.hidden_size, self.num_hidden_layers, self.num_attention_heads = core.hidden, core.layers, core.heads
        self.num_key_value_heads, self.head_dim, self.rms_norm_eps = core.kv_heads, core.head_dim, core.eps
        self.vocab_size = core.vocab  # the rows of the head that are tokens: the size of the logits
        self.max_position_embeddings = int(config["text_config"]["model_max_length"])
        self.eos_token_id = config["eos_token_id"]
        self.mtp_config, self.vision_config = config.get("mtp_config"), config.get("vision_config")


class _Tensors(nn.Module):
    """Parameters under the checkpoint's names. Float32 ones stay float32 (the model's _convertWeightsFloatTo is None)."""
    def __init__(self, dtype=torch.bfloat16, **shapes):
        super().__init__()
        for name, shape in shapes.items():
            setattr(self, name, nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False))


def _locked(module):
    """A module holding float32 parameters of a BF16 model: MMGP's precision lock keeps them as stored."""
    module._lock_dtype = torch.float32
    return module


class _ParameterList(nn.ParameterList):
    _lock_dtype = torch.float32  # the float32 queries of the block inputs


def _linear(inputs, outputs):
    return nn.Linear(inputs, outputs, bias=False, dtype=torch.bfloat16)


def mix(query, states, partial=None):
    """Attention of learned queries [queries, hidden] over normalized float32 states [tokens, states, hidden] and, with partial [tokens, hidden], over the
    norm of that running state after them: [tokens, queries, hidden]."""
    kernels = _fused(states)
    if kernels is not None and states.shape[0] <= CONV_FUSED_ROWS:
        return kernels.mix(query, states, partial, MIX_EPS)
    prefill = _prefill_fused(states) if PREFILL_MIX else None
    if prefill is not None and states.shape[0] <= prefill.MIX_ROWS:  # a short prompt pass: one kernel, float64 sums (bound by its launches, not by these)
        return prefill.mix(query, states, partial, MIX_EPS)
    if partial is not None:
        states = with_state(states, partial)
    weights = torch.softmax((query.float() @ states.transpose(-1, -2)) * states.shape[-1] ** -0.5, -1)
    return weights @ states


def with_state(normed, partial):
    return torch.cat([normed, norm(partial, MIX_EPS).unsqueeze(-2)], -2)


class _SlotState:
    """State of one block application: the last inputs of its four convolutions (attention keys, values, output and experts output), kept
    by position in rings of the model's dtype (these inputs are rounded to it before their convolution)."""
    def _init_slot(self, c):
        self.taps = c.conv_taps
        self.conv_widths = {"k": c.kv_heads * c.head_dim, "v": c.kv_heads * c.head_dim, "attn": c.hidden, "mlp": c.hidden}
        self.conv = None

    def _state(self):
        return list(self.conv.values()) if self.conv is not None else []

    def prepare_sequence_state(self, max_batch_size, device, dtype):
        if self.conv is None or self.conv["k"].device != device or self.conv["k"].dtype != dtype:
            self.conv = {name: torch.zeros(CONV_RING, width, dtype=dtype, device=device) for name, width in self.conv_widths.items()}
            self._prepare(device, dtype)
        else:
            self.reset_sequence_state()

    def _prepare(self, device, dtype):
        pass

    def reset_sequence_state(self):
        for tensor in self._state():
            tensor.zero_()

    def release_sequence_state(self):
        self.conv = None

    def snapshot_sequence_state(self):
        return [tensor.to("cpu", copy=True) for tensor in self._state()]

    def restore_sequence_state(self, snapshot):
        for tensor, saved in zip(self._state(), snapshot or ()):
            tensor.copy_(saved)  # into the live rings, which the graphs read

    def history(self, name, x, positions):
        """float32 x [tokens, channels] preceded by the taps - 1 inputs before it (zeros before the sequence); keeps the new ones."""
        ring, before = self.conv[name], self.taps - 1
        previous = positions[0] - before + torch.arange(before, device=x.device)
        rows = ring[previous % CONV_RING].float() * (previous >= 0).unsqueeze(-1)
        written = min(x.shape[0], CONV_RING)
        ring[positions[-written:] % CONV_RING] = x[-written:]
        return torch.cat([rows, x.float()])


class Inkling2WindowState(_SlotState, nn.Module):
    """A sliding-window slot: the keys and values of the last window positions in a ring indexed by position."""
    def __init__(self, c):
        super().__init__()
        self._init_slot(c)
        self.window, self.ring, self.shape = c.window, c.window + WINDOW_SLACK, (c.kv_heads, c.head_dim)
        self.keys = self.values = None

    def _state(self):
        return super()._state() + ([self.keys, self.values] if self.keys is not None else [])

    def _prepare(self, device, dtype):
        self.keys, self.values = (torch.zeros(self.ring, *self.shape, dtype=dtype, device=device) for _ in range(2))

    def release_sequence_state(self):
        self.conv = self.keys = self.values = None

    def keep(self, k, v, positions, stored=False):
        """Keeps the keys and values of a short pass (stored: its convolutions wrote them already); returns what the one-kernel attention reads: (keys, values, no page table,
        the positions a query reads)."""
        if not stored:
            rows = positions % self.ring
            self.keys[rows], self.values[rows] = k, v
        return self.keys, self.values, None, self.window

    def append(self, k, v, positions):
        """The keys and values a pass attends to (the window before it, then its own), with their positions; keeps the new ones."""
        previous = positions[0] - (self.window - 1) + torch.arange(self.window - 1, device=k.device)
        rows = previous % self.ring
        keys, values = torch.cat([self.keys[rows], k]), torch.cat([self.values[rows], v])
        if _PROMPT_REAL is not None:  # a padded pass: the ring takes the real rows only
            _PREFILL_FUSED.ring_rows_store(k, self.keys, positions, _PROMPT_REAL)
            _PREFILL_FUSED.ring_rows_store(v, self.values, positions, _PROMPT_REAL)
        else:
            written = min(k.shape[0], self.ring)
            rows = positions[-written:] % self.ring
            self.keys[rows], self.values[rows] = k[-written:], v[-written:]
        return keys, values, torch.cat([previous, positions])


class Inkling2PagedState(_SlotState, Attention):
    """A full-attention slot: its keys and values in the engine's paged KV cache (one layer of it, bound by the runner), read back by
    pages in any layout."""
    _read_rows, _read_kv, _read_overlay = IndexedAttention._read_rows, IndexedAttention._read_kv, staticmethod(IndexedAttention._read_overlay)

    def __init__(self, c):
        Attention.__init__(self, c.heads, c.head_dim, c.head_dim ** -0.5, c.kv_heads)
        self._init_slot(c)

    def _store(self, k, v):
        context = get_context()
        store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping, self.k_scale, self.v_scale, use_triton_kv_cache=self.use_triton_kv_cache,
                      k_min=self.k_min, v_min=self.v_min, k_overlay=self.k_overlay, v_overlay=self.v_overlay, k_overlay_scale=self.k_overlay_scale,
                      v_overlay_scale=self.v_overlay_scale, overlay_mapping=context.overlay_mapping, layout=self.kv_layout)
        return context

    def keep(self, k, v, positions):
        """As Inkling2WindowState.keep: the cache's pages through the sequence's table, the whole context; for a quantized cache, with what the attention needs to read its
        bodies and exact overlays in place (ring_attention_triton.attend)."""
        context = self._store(k, v)
        if self.k_cache.is_floating_point():
            return self.k_cache, self.v_cache, context.block_tables[0], 0
        layout = self.kv_layout
        quantized = dict(k=layout.k.name, v=layout.v.name, scales=(self.k_scale, self.v_scale), mins=(self.k_min, self.v_min))
        if layout.overlay and self.k_overlay is not None and context.overlay_base is not None:
            quantized.update(overlay=(self.k_overlay, self.v_overlay), overlay_scales=(self.k_overlay_scale, self.v_overlay_scale), base=context.overlay_base, ring_start=context.ring_start,
                             sink=layout.sink_tokens(self.k_cache.shape[1]), recent=layout.recent_tokens)
        return self.k_cache, self.v_cache, context.block_tables[0], 0, quantized

    def append(self, k, v, positions):
        """Writes the keys and values of a pass into the cache; returns those of the pages of its context (the sequence's pages: the
        rows after a query, written or not, are masked by position), in k's dtype, with their positions."""
        context = self._store(k, v)
        table, page = context.block_tables[0], self.k_cache.shape[1]
        pages = table.shape[0]
        if context.is_prefill and not context.speculative_verify:
            pages = min(pages, -(-int(context.max_seqlen_k) // page))
        key_positions = torch.arange(pages * page, device=k.device)
        if self.k_cache.is_floating_point():
            rows = table[:pages].clamp_min(0).long()
            return self.k_cache[rows].flatten(0, 1), self.v_cache[rows].flatten(0, 1), key_positions
        keys, values = self._read_kv(key_positions, table, 0)
        return keys.to(k.dtype), values.to(k.dtype), key_positions


def attend(c, q, rel_logits, sinks, keys, values, positions, key_positions, window=None):
    """q [tokens, heads, dim], rel_logits [tokens, heads, extent] (the bias of each backward distance), sinks [tokens, heads], keys and
    values [keys, kv_heads, dim] at key_positions: [tokens, heads * dim]. Scores and probabilities are rounded to q's dtype around a
    float32 softmax over the keys and the sink, as the reference does."""
    tokens, count, group, dtype = q.shape[0], keys.shape[0], c.heads // c.kv_heads, q.dtype
    live = ((key_positions >= 0) & (key_positions <= positions[-1]))[:, None, None]  # rows beyond the context (pages not written yet, ring slots before position 0) may hold anything, NaN included:
    keys, values = torch.where(live, keys, 0), torch.where(live, values, 0)  # their weights are 0, and 0 times NaN would still be NaN
    keys, values = keys.permute(1, 2, 0), values.transpose(0, 1)  # [kv_heads, dim, keys], [kv_heads, keys, dim]
    step, outputs = max(1, ATTENTION_ELEMENTS // max(count, 1)), []
    for start in range(0, tokens, step):
        rows = slice(start, start + step)
        queries = q[rows]
        n = queries.shape[0]
        distance = positions[rows, None] - key_positions[None, :]
        allowed = (distance >= 0) & (key_positions >= 0)[None, :]
        if window is not None:
            allowed = allowed & (distance < window)
        bias = rel_logits[rows].transpose(0, 1).gather(-1, distance.clamp(0, c.rel_extent - 1).expand(c.heads, -1, -1))
        bias = bias.masked_fill(~((distance >= 0) & (distance < c.rel_extent)), 0.0)
        scores = torch.bmm(queries.reshape(n, c.kv_heads, group, c.head_dim).permute(1, 0, 2, 3).reshape(c.kv_heads, n * group, c.head_dim), keys)
        scores = scores.view(c.kv_heads, n, group, count).permute(0, 2, 1, 3).reshape(c.heads, n, count) * c.head_dim ** -0.5 + bias
        bias = None
        scores = scores.masked_fill(~allowed, torch.finfo(dtype).min)
        probabilities = torch.softmax(torch.cat([scores, sinks[rows].T.unsqueeze(-1)], -1).float(), -1)[..., :-1].to(dtype)
        scores = None
        output = torch.bmm(probabilities.reshape(c.kv_heads, group * n, count), values)
        outputs.append(output.view(c.kv_heads, group, n, c.head_dim).permute(2, 0, 1, 3).reshape(n, -1))
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs)


class Inkling2Attention(nn.Module):
    """Queries with a learned scale and an L2 clip, keys and values through a short causal convolution, a relative-position bias for the
    last rel_extent positions, a sink logit per head, grouped queries, and a sigmoid gate on the output."""
    def __init__(self, c, full):
        super().__init__()
        self.c, self.full = c, full
        kv, taps = c.kv_heads * c.head_dim, c.conv_taps
        self.wq_du, self.wk_dv, self.wv_dv = _linear(c.hidden, c.heads * c.head_dim), _linear(c.hidden, kv), _linear(c.hidden, kv)
        self.wr_du, self.wo_ud = _linear(c.hidden, c.heads * c.d_rel), _linear(c.heads * c.head_dim, c.hidden)
        self.gate_attn, self.attn_sink_proj = _linear(c.hidden, c.heads * c.head_dim), _linear(c.hidden, c.heads)
        self.q_l2_clip, self.rel_l2_clip = _Tensors(scale=(c.heads, c.head_dim)), _Tensors(scale=(c.heads, c.d_rel))
        self.rel_logits_proj = _Tensors(proj=(c.d_rel, c.rel_extent))
        conv = (lambda: _Tensors(static_w=(taps, kv), weight_proj=(c.kv_heads * taps, c.hidden))) if full else (lambda: _Tensors(weight=(kv, 1, taps)))
        self.k_sconv, self.v_sconv = conv(), conv()

    def _conv(self, state, name, x, source, weights, positions):
        if not self.full:
            return static_conv(state, name, x, weights.weight, positions)
        # taps static per channel plus per token and KV head; tap index = backward distance
        c, taps = self.c, self.c.conv_taps
        dynamic = linear(source, weights.weight_proj).float().view(-1, c.kv_heads, 1, taps)
        taps_weights = weights.static_w.float().T.reshape(c.kv_heads, -1, taps) + dynamic
        windows = state.history(name, x, positions).unfold(0, taps, 1).flip(-1).reshape(x.shape[0], c.kv_heads, -1, taps)
        return ((windows * taps_weights).sum(-1).reshape(x.shape) + x.float()) / math.sqrt(2 * taps + 1)

    def forward(self, state, q_in, k_in, v_in, positions):
        c, dtype, tokens = self.c, q_in.dtype, q_in.shape[0]
        # queries grow with the logarithm of their position beyond log_floor
        tau = 1.0 + c.log_alpha * torch.log(((positions + 1).float() / c.log_floor).clamp(min=1.0)) if self.full else None
        batched = _LINEAR is not None and GROUPED_LINEAR and q_in.is_cuda and tokens <= _LINEAR.ROWS and _LINEAR.supported(q_in.device)
        if batched:  # the projections of one input in one launch (each row as alone)
            q, rel, sink, gate = _LINEAR.linears([(q_in, self.wq_du.weight), (q_in, self.wr_du.weight), (q_in, self.attn_sink_proj.weight), (q_in, self.gate_attn.weight)])
        else:
            q, rel, sink, gate = linear(q_in, self.wq_du.weight), linear(q_in, self.wr_du.weight), None, None
        q = clipped(q.view(tokens, c.heads, c.head_dim), c.qk_clip, self.q_l2_clip.scale, tau)
        rel = clipped(rel.view(tokens, c.heads, c.d_rel), c.rel_clip, self.rel_l2_clip.scale, tau)
        kernels = _fused(q_in) if tokens <= CONV_FUSED_ROWS else _prefill_fused(q_in)
        if kernels is not None:  # the convolution of each head and its clip or norm in one kernel
            if batched:
                k, v, *dynamic = _LINEAR.linears([(k_in, self.wk_dv.weight), (v_in, self.wv_dv.weight)] + ([(k_in, self.k_sconv.weight_proj), (v_in, self.v_sconv.weight_proj)] if self.full else []))
            else:
                k, v, dynamic = linear(k_in, self.wk_dv.weight), linear(v_in, self.wv_dv.weight), [linear(k_in, self.k_sconv.weight_proj), linear(v_in, self.v_sconv.weight_proj)] if self.full else []
            taps = (lambda weights, dynamic: dict(weight=weights.static_w, dynamic=dynamic)) if self.full else (lambda weights, dynamic: dict(weight=weights.weight))
            padded = {} if _PROMPT_REAL is None else dict(real=_PROMPT_REAL)  # (a padded prompt pass: the prompt kernels)
            stored = FUSED_RING and not self.full and tokens <= CONV_FUSED_ROWS and _ATTENTION is not None and q.is_cuda and tokens <= _ATTENTION.ROWS and _ATTENTION.supported(q.device)
            ring = (lambda name: dict(store=getattr(state, name))) if stored else (lambda name: {})  # a window slot of a short pass: its keys and values into the ring by the convolutions
            k = kernels.causal_conv(k, state.conv["k"], positions=positions, dtype=dtype, group=c.head_dim, post=1, threshold=c.qk_clip, **taps(self.k_sconv, dynamic[0] if self.full else None), **padded, **ring("keys")).view(tokens, c.kv_heads, c.head_dim)
            v = kernels.causal_conv(v, state.conv["v"], positions=positions, dtype=dtype, group=c.head_dim, post=2, eps=c.eps, **taps(self.v_sconv, dynamic[1] if self.full else None), **padded, **ring("values")).view(tokens, c.kv_heads, c.head_dim)
        else:
            stored = False
            k = self._conv(state, "k", linear(k_in, self.wk_dv.weight), k_in, self.k_sconv, positions)
            v = self._conv(state, "v", linear(v_in, self.wv_dv.weight), v_in, self.v_sconv, positions)
            k = clipped(k.view(tokens, c.kv_heads, c.head_dim), c.qk_clip, rounded=True)
            v = norm(v.view(tokens, c.kv_heads, c.head_dim), c.eps, None, dtype, True)
        fused = _ATTENTION if _ATTENTION is not None and q.is_cuda and tokens <= _ATTENTION.ROWS and _ATTENTION.supported(q.device) else None
        source = (state.keep(k, v, positions, stored=True) if stored else state.keep(k, v, positions)) if fused is not None else None
        if source is not None:  # the positions of the context read once, in place
            keys, values, table, window, *quantized = source
            sink, gate = (sink, gate) if batched else (linear(q_in, self.attn_sink_proj.weight), linear(q_in, self.gate_attn.weight))
            return linear(fused.attend(q, rel, self.rel_logits_proj.proj, sink, keys, values, positions, c.head_dim ** -0.5, window, table, gate, dtype,
                                       *quantized, shared=SHARED_TILES), self.wo_ud.weight)
        prefill = _PREFILL_ATTENTION if _PREFILL_ATTENTION is not None and PREFILL_ATTENTION and tokens > CONV_FUSED_ROWS and q.is_cuda and _PREFILL_ATTENTION.supported(q.device) else None
        if prefill is not None:  # the cache's pages read in place, the window rows as append gives them (consecutive positions)
            sink, gate = (sink, gate) if batched else (linear(q_in, self.attn_sink_proj.weight), linear(q_in, self.gate_attn.weight))
            bias = rel @ self.rel_logits_proj.proj
            if self.full and PREFILL_IN_PLACE:
                keys, values, table, _, *quantized = state.keep(k, v, positions)
                output = prefill.attend(q, bias, sink, keys, values, positions, c.head_dim ** -0.5, gate, table=table, context=int(get_context().max_seqlen_k), quantized=quantized[0] if quantized else None, dtype=dtype, real=_PROMPT_REAL)
            elif self.full:  # (a comparison: the rows append reads, from position 0)
                keys, values, _ = state.append(k, v, positions)
                output = prefill.attend(q, bias, sink, keys, values, positions, c.head_dim ** -0.5, gate, key_shift=int(positions[0]), context=int(get_context().max_seqlen_k), dtype=dtype)
            else:
                keys, values, _ = state.append(k, v, positions)
                output = prefill.attend(q, bias, sink, keys, values, positions, c.head_dim ** -0.5, gate, window=c.window, key_shift=c.window - 1, dtype=dtype, real=_PROMPT_REAL)
            k = v = keys = values = bias = None
            return linear(output, self.wo_ud.weight)
        keys, values, key_positions = state.append(k, v, positions)
        k = v = None
        output = attend(c, q, rel @ self.rel_logits_proj.proj, sink if batched else linear(q_in, self.attn_sink_proj.weight), keys, values, positions, key_positions, None if self.full else c.window)
        keys = values = None
        return linear((output.float() * torch.sigmoid((gate if batched else linear(q_in, self.gate_attn.weight)).float())).to(dtype), self.wo_ud.weight)


def static_conv_to(state, name, x, weight, positions, dtype):
    """static_conv in dtype: one kernel for the rows of a short pass."""
    kernels = _fused(x) if x.shape[0] <= CONV_FUSED_ROWS else _prefill_fused(x)
    if kernels is not None:
        return kernels.causal_conv(x, state.conv[name], weight, positions, dtype, **({} if _PROMPT_REAL is None else dict(real=_PROMPT_REAL)))
    return static_conv(state, name, x, weight, positions).to(dtype)


def static_conv(state, name, x, weight, positions):
    """Causal depthwise convolution with its residual. weight [channels, 1, taps], the last tap on the current token."""
    taps = weight.shape[-1]
    windows = state.history(name, x, positions).unfold(0, taps, 1)  # [tokens, channels, taps]
    return ((windows * weight.float().squeeze(1)).sum(-1) + x.float()) / math.sqrt(taps + 1)


def message_gate(controller, hidden, message):
    """A stream of a loop pass gated with its message from the previous pass (float32)."""
    down, eps, kernels = controller.down_weight, 1e-6, _fused(hidden)
    if kernels is not None and hidden.shape[0] <= CONV_FUSED_ROWS:  # a short pass: two kernels (its programs read their rows once per rank row)
        return kernels.message_gate(hidden, message, down, controller.forget_weight, controller.gain_weight, eps)
    kernels = _prefill_fused(hidden) if hidden.shape[0] > CONV_FUSED_ROWS else None
    if kernels is not None:  # a prompt pass: the norms in one kernel, the products as below, the tanh mix in one kernel
        return kernels.message_gate(hidden, message, down, controller.forget_weight, controller.gain_weight, eps)
    hidden, message = hidden.float(), message.float()
    g = F.linear(torch.cat([rms(hidden, eps), rms(message, eps)], -1).to(down.dtype), down).float()
    return (1 - torch.tanh(F.linear(g, controller.forget_weight.float()))) * hidden + torch.tanh(F.linear(g, controller.gain_weight.float())) * message


class Inkling2PackedExperts(nn.Module):
    """PyTorch reference of the routed experts (WGP_INKLING2_DISABLE=routed) on the checkpoint's packed tensors (NVFP4: two E2M1 codes per byte, an FP8 scale per 16 values, amax /
    denominator per expert; MXFP8: FP8 values, a power-of-two scale per 32), decoded per expert as inkling2_core does (bit-identical decode), where
    mmgp.expert_compute.RoutedExperts computes on the packed rows. Same tensor names; the buffers are MMGP's to place."""
    def __init__(self, c):
        super().__init__()
        self.c, self.packed = c, None

    def _load_from_state_dict(self, state_dict, prefix, *args):
        # the format is the checkpoint's: its tensors as buffers, in their storage (NVFP4 holds an amax per expert)
        for name, tensor in state_dict.items():
            if name.startswith(prefix):
                self.register_buffer(name[len(prefix):], torch.empty(tensor.shape, dtype=tensor.dtype, device="meta"))
        self.packed = "nvfp4" if prefix + "w13_factor" in state_dict else "mxfp8"
        super()._load_from_state_dict(state_dict, prefix, *args)

    def release_sequence_state(self):
        _DECODED.clear()

    def _decode(self, name, index, dtype):
        """The experts `index` (a tensor) of projection name [experts, rows, columns] in dtype, with inkling2_core.dequantize_experts' arithmetic."""
        weight, scale = (getattr(self, name + part).view(torch.uint8)[index].int() for part in ("_weight", "_weight_scale"))  # FP8 as bytes: table lookups
        pairs, fp8 = _expert_tables(weight.device)
        if self.packed == "mxfp8":
            return (fp8[weight].unflatten(-1, (-1, 32)) * torch.exp2(scale.float() - 127).unsqueeze(-1)).flatten(-2).to(dtype)
        factor = getattr(self, name + "_factor")[index]
        values = pairs[weight].unflatten(-2, (-1, 8)).flatten(-2) * fp8[scale].unsqueeze(-1)
        return (values * factor.reshape(-1, 1, 1, 1)).flatten(-2).to(dtype)

    def _expert(self, index, dtype):
        key = (id(self), index, self.w13_weight.device)
        hit = _DECODED.pop(key, None)
        if hit is None:
            select = torch.tensor([index], device=self.w13_weight.device)
            hit = self._decode("w13", select, dtype)[0], self._decode("w2", select, dtype)[0]
            while len(_DECODED) >= EXPERT_CACHE:
                _DECODED.pop(next(iter(_DECODED)))
        _DECODED[key] = hit
        return hit

    def _activation(self, gate_up, dtype):
        c, gate, up = self.c, gate_up[..., 0::2], gate_up[..., 1::2]  # rows of w13 alternate gate and up
        return (softcap(gate, c.gate_softcap) * torch.sigmoid(gate) * softcap(up, c.up_softcap)).to(dtype)

    def forward(self, x, ids, weights):
        """x [tokens, hidden] in the model dtype, ids and float32 weights [tokens, top_k]: the weighted sum of each token's experts
        (float32 accumulation), in x's dtype."""
        if cuda_graphs.recording() or torch.cuda.is_current_stream_capturing():
            return self._forward_fixed(x, ids, weights)
        output = torch.zeros_like(x, dtype=torch.float32)
        for index in ids.unique().tolist():
            token, rank = torch.where(ids == index)
            w13, w2 = self._expert(index, x.dtype)
            activation = self._activation(F.linear(x[token], w13).float(), x.dtype)
            output.index_add_(0, token, (F.linear(activation, w2) * weights[token, rank, None]).float())
        return output.to(x.dtype)

    def _forward_fixed(self, x, ids, weights):
        """The same sum with shapes that follow the rows only, for the passes recorded in CUDA graphs: each token's experts are decoded
        at each pass."""
        outputs = []
        for token in range(x.shape[0]):
            activation = self._activation(torch.matmul(self._decode("w13", ids[token], x.dtype), x[token]).float(), x.dtype)
            output = torch.bmm(self._decode("w2", ids[token], x.dtype), activation.unsqueeze(-1)).squeeze(-1)
            outputs.append((output * weights[token, :, None]).float().sum(0))
        return torch.stack(outputs).to(x.dtype)


_EXPERT_TABLES = {}
_DECODED = {}  # (bank, expert, device) -> its decoded projections, the least recently used first


def _expert_tables(device):
    """float32 tables of a byte: its two E2M1 values (low nibble first), and its FP8 E4M3 value."""
    if device not in _EXPERT_TABLES:
        byte = torch.arange(256, dtype=torch.int32, device=device)
        e2m1 = torch.tensor(E2M1, device=device)
        exponent, mantissa = (byte >> 3) & 15, (byte & 7).float()
        fp8 = torch.where(exponent == 0, mantissa / 8 * 2.0 ** -6, (1 + mantissa / 8) * torch.exp2(exponent.float() - 7)) * (1.0 - 2.0 * (byte >> 7))
        _EXPERT_TABLES[device] = cuda_graphs.register_constant(torch.stack((e2m1[byte & 15], e2m1[byte >> 4]), -1)), cuda_graphs.register_constant(fp8)
    return _EXPERT_TABLES[device]


class Inkling2MoE(nn.Module):
    """Routed experts: float32 sigmoid scores, the top_k of score + bias, weights normalized to route_scale."""
    def __init__(self, c, use_triton=False):
        super().__init__()
        self.c = c
        self.gate = _Tensors(weight=(c.experts, c.hidden))
        self.gate.bias = nn.Parameter(torch.empty(c.experts, dtype=torch.float32), requires_grad=False)
        _locked(self.gate).weight._lock_dtype = None  # a float32 bias beside its weight, which stays as stored (BF16)
        if ROUTED:  # the format of the bank is that of the tensors it loads (NVFP4 or MXFP8 rows)
            self.experts = RoutedExperts(c.experts, c.hidden, c.intermediate, activation="swiglu_softcap", caps=(c.gate_softcap, c.up_softcap), use_triton=use_triton)
            self.experts.streamed_pairs = 1 if use_triton else None  # prompt passes run every routed expert on the GPU
            self.experts.doorbell_free = DOORBELL_FREE
        else:
            self.experts = Inkling2PackedExperts(c)

    def forward(self, x):
        kernels = _fused(x)
        if kernels is not None and x.shape[0] <= CONV_FUSED_ROWS:  # the passes whose experts are queued: no routing returns to the host
            ids, weights = kernels.route(x, self.gate.weight, self.gate.bias, self.c.top_k, self.c.route_scale)
        elif PREFILL_ROWS and x.shape[0] > CONV_FUSED_ROWS and (prefill := _prefill_fused(x)) is not None:  # a prompt pass: the product as below, the selection in one kernel
            ids, weights = prefill.route(x, self.gate.weight, self.gate.bias, self.c.top_k, self.c.route_scale)
        else:
            ids, weights = sigmoid_bias_topk(F.linear(x.float(), self.gate.weight.float()), self.gate.bias, self.c.top_k, True, self.c.route_scale)
        return routed_experts(self.experts, x, ids, weights) if ROUTED else self.experts(x, ids, weights)


class Inkling2Block(nn.Module):
    layer_type = "full_attention"  # no recurrent state of the kind the assistant runtime lists

    def __init__(self, c, layer, use_triton=False):
        super().__init__()
        self.c, self.layer = c, layer
        self.full, self.boundary = layer not in c.local_layers, layer % c.block == 0
        looped = c.loop[0] <= layer < c.loop[1]
        self.attn = Inkling2Attention(c, self.full)
        if self.full:
            self.q_input_norm, self.k_input_norm, self.v_input_norm = (_Tensors(weight=(c.hidden,)) for _ in range(3))
        else:
            self.attn_norm = _Tensors(weight=(c.hidden,))
        self.attn_sconv, self.mlp_sconv = _Tensors(weight=(c.hidden, 1, c.conv_taps)), _Tensors(weight=(c.hidden, 1, c.conv_taps))
        self.mlp_norm = _Tensors(weight=(c.hidden,))
        self.mlp = Inkling2MoE(c, use_triton)
        if looped:
            self.controllers = nn.ModuleDict({name: _Tensors(down_weight=(c.gate_rank, 2 * c.hidden), forget_weight=(c.hidden, c.gate_rank), gain_weight=(c.hidden, c.gate_rank))
                                              for name in ("attn", "k", "v", "mlp")})
        # one state per application of the block: its passes of the loop
        self.states = nn.ModuleList((Inkling2PagedState if self.full else Inkling2WindowState)(c) for _ in range(c.passes if looped else 1))

    def forward(self, carry, positions, attn_query, mlp_query, message_states=None, message_query=None, loop_pass=0):
        """carry: [hidden states [tokens, hidden] (the residual stream since the last block boundary), the normalized states of the
        block boundaries before it, float32 [tokens, boundaries, hidden]]. message_states: in passes after the first, the normalized
        states [tokens, 2, hidden] of the previous pass that the four message queries [4, hidden] read."""
        hidden, normed = carry
        carry.clear()
        c, dtype, state, eps = self.c, hidden.dtype, self.states[loop_pass], self.c.eps
        messages = None if message_states is None else mix(message_query, message_states)  # query, key, value, experts
        if attn_query is None:
            q_h = k_h = v_h = hidden
        else:
            mixed = mix(attn_query.view(-1, c.hidden), normed, hidden)
            q_h, k_h, v_h = mixed.unbind(-2) if self.full else (mixed[:, 0],) * 3
        if messages is not None:
            q_h, k_h, v_h = (message_gate(self.controllers[name], h, messages[:, index]) for index, (name, h) in enumerate((("attn", q_h), ("k", k_h), ("v", v_h))))
        if self.full:
            q_in, k_in, v_in = (norm(h, eps, weights.weight, dtype) for h, weights in ((q_h, self.q_input_norm), (k_h, self.k_input_norm), (v_h, self.v_input_norm)))
        elif messages is not None:
            q_in, k_in, v_in = (norm(h, eps, self.attn_norm.weight, dtype) for h in (q_h, k_h, v_h))
        else:
            q_in = k_in = v_in = norm(q_h, eps, self.attn_norm.weight, dtype)
        q_h = k_h = v_h = mixed = None
        attn = self.attn(state, q_in, k_in, v_in, positions)
        q_in = k_in = v_in = None
        kernels = _fused(attn) if FUSED_RESIDUAL and dtype == torch.bfloat16 and attn.shape[0] <= CONV_FUSED_ROWS else None
        if kernels is not None:  # the convolution, its BF16 rounding and the residual add in one kernel (the same bits)
            residual = kernels.causal_conv(attn, state.conv["attn"], self.attn_sconv.weight, positions, dtype, residual=None if self.boundary else hidden.contiguous(), out_dtype=torch.float32)
            if self.boundary:
                normed = with_state(normed, hidden)
        else:
            attn = static_conv_to(state, "attn", attn, self.attn_sconv.weight, positions, dtype).float()
            if self.boundary:  # the residual stream restarts from the attention; the block's input stays readable among the states
                normed, residual = with_state(normed, hidden), attn
            else:
                residual = hidden.float() + attn
        hidden = attn = None
        mlp_in = mix(mlp_query.view(-1, c.hidden), normed, residual)[:, 0]
        if messages is not None:
            mlp_in = message_gate(self.controllers["mlp"], mlp_in, messages[:, 3])
        mlp = self.mlp(norm(mlp_in, eps, self.mlp_norm.weight, dtype))
        if kernels is not None:
            return [kernels.causal_conv(mlp, state.conv["mlp"], self.mlp_sconv.weight, positions, dtype, residual=residual, out_dtype=dtype), normed]
        mlp = static_conv_to(state, "mlp", mlp, self.mlp_sconv.weight, positions, dtype)
        return [(residual + mlp.float()).to(dtype), normed]


class Inkling2Head(nn.Module):
    """The head: the token rows of the unembedding, soft-capped float32 logits."""
    def __init__(self, c):
        super().__init__()
        self.vocab, self.cap = c.vocab, c.logit_softcap
        self.weight = nn.Parameter(torch.empty(c.vocab_rows, c.hidden, dtype=torch.bfloat16), requires_grad=False)

    def forward(self, hidden):
        rows = hidden.reshape(-1, hidden.shape[-1])
        if _LINEAR is not None and hidden.is_cuda and rows.shape[0] <= _LINEAR.ROWS and _LINEAR.supported(hidden.device):  # row by row, as the blocks' linears
            return _LINEAR.linear(rows, self.weight[:self.vocab], softcap=self.cap).view(*hidden.shape[:-1], self.vocab)
        return softcap(F.linear(hidden, self.weight[:self.vocab]).float(), self.cap)

    def lm_engine_graph_forward(self, hidden, logits):
        """The logits of a verification window into their buffer (cuda_graphs.run stage)."""
        logits.copy_(self(hidden))


class Inkling2ForCausalLM(nn.Module):
    """The LM engine's Inkling 2. forward returns the final normalized hidden states [1, tokens, hidden]; compute_logits (or output)
    applies the head. draft_features [1, tokens, 4 * hidden], when given, receives what the DSpark drafter reads: the states entering
    the applications DRAFT_FEATURES, then the returned states."""
    _lm_engine_tower_prefix = "layers."
    _lm_engine_prefill_block_tables = True  # its full-attention slots read the cache in every pass
    _lm_engine_adaptive_experts = ROUTED  # its GPU experts follow the routing
    _offload_base_modules = ("mtp",)  # the drafter (when loaded) stays with the base: its stages and its context writes read its weights outside a hooked forward

    def __init__(self, config, use_triton=False):
        super().__init__()
        self.config, self.c = config, config.core
        c = self.c
        if use_triton and FUSED:
            global _KERNELS, _ATTENTION, _LINEAR, _PREFILL_FUSED, _PREFILL_ATTENTION
            from shared.kernels import row_fused_triton as _KERNELS
            from shared.kernels import prefill_fused_triton as _PREFILL_FUSED
            if FUSED_ATTENTION:
                from shared.kernels import ring_attention_triton as _ATTENTION
            if FUSED_LINEAR:
                from shared.kernels import row_linear_triton as _LINEAR
            from shared.kernels import prefill_attention_triton as _PREFILL_ATTENTION
        self.embed = nn.Embedding(c.vocab_rows, c.hidden, dtype=torch.bfloat16)
        self.embed_norm, self.norm = _Tensors(weight=(c.hidden,)), _Tensors(weight=(c.hidden,))
        self.attn_res = _locked(nn.Module())
        _locked(self)  # the float32 queries of the loop messages are its own; its forward takes no floating-point argument
        full = lambda layer: layer not in c.local_layers
        self.attn_res.attn_queries = _ParameterList(nn.Parameter(torch.empty((3, c.hidden) if full(layer) else (c.hidden,), dtype=torch.float32), requires_grad=False) for layer in range(1, c.layers))
        self.attn_res.mlp_queries = _ParameterList(nn.Parameter(torch.empty(c.hidden, dtype=torch.float32), requires_grad=False) for _ in range(c.layers))
        self.attn_res.output_query = nn.Parameter(torch.empty(c.hidden, dtype=torch.float32), requires_grad=False)
        loop_queries = 2 * (c.loop[1] - c.loop[0])
        self.middle_cycle_cross_pass_queries = nn.Parameter(torch.empty(loop_queries, c.hidden, dtype=torch.float32), requires_grad=False)  # rows [query, experts] per loop block
        self.middle_cycle_cross_pass_kv_queries = nn.Parameter(torch.empty(loop_queries, c.hidden, dtype=torch.float32), requires_grad=False)  # rows [key, value]
        self.layers = nn.ModuleList(Inkling2Block(c, layer, use_triton) for layer in range(c.layers))
        self.unembed = Inkling2Head(c)
        self.schedule = c.schedule()
        self.mtp = None  # the DSpark drafter (inkling2_draft.Inkling2Draft on mtp.safetensors), when speculative decoding is on
        self._block_draft, self._block_draft_tree, self._draft_features = False, None, None  # its drafts verify as a chain

    @property
    def device(self):
        return self.embed_norm.weight.device

    @property
    def embed_tokens(self):
        return self.embed

    @property
    def token_embd(self):
        return self.embed

    @property
    def blk(self):
        return self.layers

    @property
    def output(self):
        return self.unembed

    @property
    def lm_head(self):
        return self.unembed

    def embed_ids(self, input_ids):
        """The hidden states the blocks read for these tokens (the embedding rows through their norm): what inputs_embeds stands for."""
        return norm(self.embed(input_ids), self.c.eps, self.embed_norm.weight, self.embed_norm.weight.dtype)

    def _sequence_positions(self, tokens, device):
        """The positions in the sequence of the tokens of this pass."""
        context = get_context()
        if context.speculative_verify:
            first = context.cu_seqlens_k[1] - tokens
        elif context.is_prefill:
            first = context.cu_seqlens_k[1] - context.cu_seqlens_q[1]
        else:
            first = context.context_lens[0] - 1
        # a graph capture's pass on dummy pages has no context: position 0 keeps what it reads and writes finite
        return first.long().clamp_min(0) + torch.arange(tokens, dtype=torch.long, device=device)

    def forward(self, input_ids=None, positions=None, inputs_embeds=None, draft_features=None, **_kwargs):
        c = self.c
        hidden = self.embed_ids(input_ids.reshape(-1)) if inputs_embeds is None else inputs_embeds.reshape(-1, c.hidden)
        tokens, start = hidden.shape[0], c.loop[0]
        sequence = self._sequence_positions(tokens, hidden.device)
        if draft_features is None and self._block_draft:  # an eager pass (a prompt chunk): the drafter's features of its tokens, for the runner
            draft_features = self._draft_features = torch.empty(1, tokens, (len(DRAFT_FEATURES) + 1) * c.hidden, dtype=hidden.dtype, device=hidden.device)
        carry = [hidden, torch.zeros(tokens, 0, c.hidden, dtype=torch.float32, device=hidden.device)]
        hidden = before = message_states = message_query = None
        queries, kv_queries = self.middle_cycle_cross_pass_queries.unflatten(0, (-1, 2)), self.middle_cycle_cross_pass_kv_queries.unflatten(0, (-1, 2))
        # a prompt pass: the experts out of VRAM stream ahead of their use (mmgp.expert_stream), a stream per walk over the banks: the loop reads its banks three times
        streamed = STREAMED and tokens > CONV_FUSED_ROWS and carry[0].is_cuda and not torch.cuda.is_current_stream_capturing()
        with contextlib.ExitStack() as streams:
            for application, (layer, _slot, loop_pass, index) in enumerate(self.schedule):
                if streamed and (application == 0 or layer == start and loop_pass):
                    streams.close()
                    stop = next((later for later in range(application + 1, len(self.schedule)) if self.schedule[later][0] == start and self.schedule[later][2]), len(self.schedule))
                    streams.enter_context(runtime.stream_experts(self, [self.layers[item[0]].mlp.experts for item in self.schedule[application:stop]], tokens, tokens * PROMPT_BYTES_PER_TOKEN + PROMPT_VRAM_MARGIN))
                if layer == start:
                    if loop_pass == 0:
                        before = list(carry)
                    else:  # the next pass restarts from the loop's input and reads the previous pass: its states since its first boundary
                        message_states = with_state(carry[1][:, before[1].shape[1] + 1:], carry[0])
                        carry = list(before)
                if draft_features is not None and application in DRAFT_FEATURES:
                    self._tap(draft_features, DRAFT_FEATURES.index(application), carry[0])
                if loop_pass:
                    message_query = torch.cat([queries[index, :1], kv_queries[index], queries[index, 1:]])
                carry = self.layers[layer](carry, sequence, self.attn_res.attn_queries[layer - 1] if layer else None, self.attn_res.mlp_queries[layer],
                                           message_states if loop_pass else None, message_query if loop_pass else None, loop_pass or 0)
        before = message_states = None
        final = mix(self.attn_res.output_query.view(-1, c.hidden), carry[1], carry[0])[:, 0]
        final = norm(final, c.eps, self.norm.weight, carry[0].dtype)
        if draft_features is not None:
            self._tap(draft_features, len(DRAFT_FEATURES), final)
        return final.unsqueeze(0)

    def _tap(self, draft_features, index, states):
        hidden = self.c.hidden
        draft_features[0, :, index * hidden:(index + 1) * hidden].copy_(states)

    def lm_engine_graph_forward(self, input_ids, positions, outputs, draft_features=None):
        """A decoding pass of the engine (cuda_graphs.run stage): the hidden states of input_ids written into outputs."""
        outputs.copy_(self(input_ids, positions, draft_features=draft_features).view(outputs.shape))

    @property
    def lm_engine_prompt_graph_rows(self):
        """The padded sizes of the short prompt passes the engine may replay as graphs (none without the prompt kernels)."""
        return PROMPT_GRAPH_ROWS if PROMPT_GRAPHS and _PREFILL_FUSED is not None and _PREFILL_ATTENTION is not None and ROUTED else ()

    def lm_engine_prompt_graph_ready(self):
        """Whether a short prompt pass may replay its graph now: the prompt kernels on (the PyTorch paths would write the padded rows' state) and every expert in VRAM (a graph
        streams nothing)."""
        from mmgp import expert_compute
        return (PROMPT_GRAPHS and PREFILL_FUSED and PREFILL_ATTENTION and PREFILL_IN_PLACE and expert_compute.DEVICE_GROUPS
                and all(bool(block.mlp.experts.resident_flags.all()) for block in self.layers))

    def lm_engine_prompt_forward(self, input_ids, positions, outputs, real, draft_features=None):
        """A padded short prompt pass (cuda_graphs.run stage): its hidden states into outputs; real (int32 [1], on the device): the rows that are the prompt's, the others write
        no state (KV slots of -1 from the engine, the rings and the window keys of the real rows only)."""
        global _PROMPT_REAL
        _PROMPT_REAL = real
        try:
            outputs.copy_(self(input_ids, positions, draft_features=draft_features).view(outputs.shape))
        finally:
            _PROMPT_REAL = None

    def compute_logits(self, hidden_states):
        context = get_context()
        if hidden_states.ndim == 3:
            if context.is_prefill and context.cu_seqlens_q is not None:
                hidden_states = hidden_states[:, int(context.cu_seqlens_q[-1].item()) - 1, :]
            elif hidden_states.shape[-2] == 1:
                hidden_states = hidden_states[:, 0, :]
        return self.unembed(hidden_states)


def llm_state_dict(state_dict, quant_map=None, tied_map=None, prefix="model.llm."):
    """The language model's tensors of the checkpoint under this module tree's names (a preprocess_sd of MMGP's loader). The vision
    tower is a file of its own."""
    if "fuse_weight" in state_dict:  # the drafter's file (a module of the load)
        from .inkling2_draft import draft_state_dict
        return draft_state_dict(state_dict), quant_map, tied_map
    tensors = {name[len(prefix):]: tensor for name, tensor in state_dict.items() if name.startswith(prefix)}
    for name in [name for name in tensors if name.endswith("_weight_amax")]:  # NVFP4 experts: one float32 factor per expert, as the expert banks hold it
        base = name[:-len("_weight_amax")]
        tensors[base + "_factor"] = (tensors.pop(name).float() / tensors.pop(base + "_weight_outer_scale_denominator").float()).view(-1, 1)
    return tensors, quant_map, tied_map


def kv_cache_bytes(c, kv_cache_layout, tokens, page=256):
    """VRAM of the keys and values of a context of tokens: the paged cache of the full-attention slots (BF16, or the body and exact
    overlays of a quantized layout) and the rings of the sliding-window slots, which do not grow with it."""
    slots = c.schedule()
    full = sum(layer not in c.local_layers for layer, *_ in slots)
    rings = (len(slots) - full) * (c.window + WINDOW_SLACK) * 2 * c.kv_heads * c.head_dim * 2
    if not kv_cache_layout:
        return rings + full * 2 * c.kv_heads * c.head_dim * 2 * tokens
    layout = kvf.parse_layout(kv_cache_layout)
    return rings + layout.body_bytes_per_token(full, c.kv_heads, c.head_dim) * tokens + layout.overlay_bytes(full, c.kv_heads, c.head_dim, page, 1)


def set_expert_priorities(model):
    """The first placement of the experts (expert_priority of each bank, before MMGP takes the model): measured evaluations per token and per byte, comparable across
    the layers. The counts of inkling2_expert_usage.json are per layer over the three passes of the loop: an expert of blocks 6-17 is read up to three times per
    token, and a share of its own layer's routing would hide it. Decoding then follows the routing of the text (the banks count each call)."""
    import json
    with open(os.path.join(os.path.dirname(__file__), "inkling2_expert_usage.json"), encoding="utf-8") as reader:
        usage = torch.tensor(json.load(reader)["counts"], dtype=torch.float64)
    tokens = usage.sum() / (len(model.schedule) * model.c.top_k)
    for block, counts in zip(model.layers, usage):
        bank = block.mlp.experts
        bank.expert_priority = ((counts + 1) / tokens * (2 << 20) / bank.expert_nbytes).tolist()
