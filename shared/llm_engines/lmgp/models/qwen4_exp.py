# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
# Qwen3.8 Flash-Next (qwen4_exp), written for the LM engine after the Hugging Face Transformers implementation (Apache 2.0,
# models/qwen4_exp/modular_qwen4_exp.py) and llama.cpp's qwen4exp graph (MIT License, src/models/qwen4exp.cpp), with ideas of Strata (MIT License,
# https://github.com/Niko1221/Strata) for running it beyond the VRAM. Hyper-connections keep parallel residual streams, the mixers are the Gated DeltaNet of
# Qwen3.5 (shared with lmgp.models.qwen3_5) and indexed sparse attention, every layer ends in routed experts (MMGP expert banks in their GGUF storage),
# and one layer reads hashed n-gram embeddings from a table left on the drive. One sequence at a time.

import math
import os
from copy import copy

import torch
from torch import nn
from torch.nn import functional as F
from mmgp import cuda_graphs, expert_stream, offload, runtime
from mmgp.expert_compute import RoutedExperts

from ..layers.attention_indexed import IndexedAttention, fused_kernels, rms_norm, rotate_
from ..layers.layernorm import RMSNorm
from ..layers.linear import ColumnParallelLinear, RowParallelLinear
from ..layers.moe import routed_experts, softmax_topk
from ..layers.ngram_table import QUEUED_ROWS
from ..utils.context import get_context
from .qwen3_5 import Qwen3_5Block, Qwen3_5RMSNormGated, Qwen3_5StaticCache, Qwen3_5TextRotaryEmbedding, _build_mrope_freq_cache, _fla_fused_norm_gate
from shared.kernels.dispatch import triton_dtype_supported

try:
    from shared.kernels import hyper_connection_triton as hc_kernels
except ImportError:
    hc_kernels = None

DRAFT = "draft" not in os.environ.get("WGP_FN_DISABLE", "").split(",")  # the drafter's norm of the target's streams, one per stream
SHARED_BESIDE = "shared_beside" not in os.environ.get("WGP_FN_DISABLE", "").split(",")  # the shared expert of a decoding pass computed while the CPU computes the routed experts out of VRAM, until the switches go
FUSED_HC = hc_kernels is not None and "hc" not in os.environ.get("WGP_FN_DISABLE", "").split(",")  # validated: a switch for the comparisons of the other work under way
STATE_RING = 32  # the token ids and convolution inputs of the n-gram layer, by position: a verification pass and its history
PROMPT_CHUNKS = (32768, 24576, 16384, 12288, 8192, 6144, 4096, 3072, 2048, 1024)  # the prompt chunks to choose from
PROMPT_BYTES_PER_TOKEN = 340 << 10  # VRAM of a prompt pass per token: streams, activations, the routed experts' rows
PROMPT_VRAM_MARGIN = 256 << 20  # left free by a prompt pass


def _rotary_tables(rotary, positions):
    """float32 cos, sin [1, tokens, rotary]: the rotation of queries and keys and of the blocks of the indexer. Computed here rather than by the module, whose
    hook gives its float32 arguments the model's compute dtype."""
    if positions.ndim == 1:
        positions = positions.unsqueeze(0)
    table = _build_mrope_freq_cache(positions.device, max_position=rotary.max_position_embeddings, rotary_dim=rotary.rotary_dim, rope_theta=rotary.rope_theta)
    if positions.ndim == 2:  # text: the same position on the three axes, which their interleaving keeps
        freqs = table[positions.long()]
    else:
        freqs = rotary.apply_interleaved_mrope(torch.stack([table[positions[axis].long()] for axis in range(3)], dim=0))
    angles = torch.cat((freqs, freqs), dim=-1)
    return angles.cos(), angles.sin()


def _grouped_rms(x, weight, hidden, eps):
    """float32 RMS norm of x [..., streams * hidden] over each stream, weighted over all of them."""
    y = x.float().view(*x.shape[:-1], -1, hidden)
    y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
    return y.flatten(-2) * weight.float()


class Qwen4ExpHyperConnection(nn.Module):
    """Mixes the residual streams into a mixer's input and gives the weights injecting its output back into them (a learned mix replacing layer norms)."""
    def __init__(self, config, inject=True):
        super().__init__()
        hidden, self.streams = int(config.hidden_size), int(config.hc_count)
        self.hidden, self.eps = hidden, float(config.rms_norm_eps)
        width, rank = hidden * self.streams, int(config.hc_lowrank)
        self.norm = nn.Parameter(torch.ones(width))
        self.down = ColumnParallelLinear(width, rank, bias=False)
        self.up = ColumnParallelLinear(rank, width, bias=False)
        self.inject = ColumnParallelLinear(width, self.streams, bias=False) if inject else None
        self._lock_dtype = torch.float32  # precision lock of its own float32 norm weights; its linears are not concerned

    def fused(self, state):
        """The fused kernels: a few rows of a float32 state on a GPU whose Triton computes in the dtype of the projections, these in VRAM (MMGP's CPU compute may keep them in RAM)."""
        return (FUSED_HC and state.is_cuda and state.dtype == torch.float32 and state.numel() <= hc_kernels.MAX_ROWS * state.shape[-1] and triton_dtype_supported(state.device, self.down.weight.dtype)
                and self.down.weight.is_cuda and self.up.weight.is_cuda and self.norm.is_cuda and (self.inject is None or self.inject.weight.is_cuda))

    def fused_rows(self, state):
        """The fused norms and mixes of a pass of many rows, BLAS keeping the projections."""
        return FUSED_HC and state.is_cuda and state.dtype == torch.float32 and triton_dtype_supported(state.device, self.down.weight.dtype) and self.down.weight.dtype == torch.bfloat16 and self.norm.is_cuda

    def read_rows(self, state, output=None, weights=None):
        """state [tokens, streams * hidden]: (mixed, weights), the output of the mixer before it injected first into state, in place."""
        normed, low = hc_kernels.norm_rows(state, self.norm, self.eps, self.streams, self.hidden, output, weights)
        gate = self.up(F.silu(self.down(low) / self.streams))
        injection = None if self.inject is None else (self.inject(low).float() / self.streams).sigmoid() * 2
        low = None
        return hc_kernels.mix_rows(gate, normed, self.streams, self.hidden, self.down.weight.dtype), injection

    def read(self, state, output=None, weights=None, unrounded=None):
        """The fused kernels on rows [tokens, streams * hidden], with the injection of a previous output if any (see hc_kernels.read)."""
        return hc_kernels.read(state, self.norm, self.down.weight, self.up.weight, None if self.inject is None else self.inject.weight, self.streams, self.hidden, self.eps, self.down.weight.dtype, output, weights, unrounded)

    def forward(self, state):
        if self.fused(state):
            mixed, weights, _ = self.read(state.reshape(-1, state.shape[-1]))
            mixed = mixed.view(*state.shape[:-1], self.hidden)
            return mixed if self.inject is None else (mixed, weights.view(*state.shape[:-1], self.streams))
        dtype = self.down.weight.dtype  # the compute dtype of its projections
        normed = _grouped_rms(state, self.norm, self.hidden, self.eps)
        low = normed.to(dtype)
        gate = self.up(F.silu(self.down(low) / self.streams)).float().sigmoid()
        mixed = (gate * normed).view(*state.shape[:-1], self.streams, self.hidden).mean(-2).to(dtype)
        if self.inject is None:
            return mixed
        return mixed, (self.inject(low).float() / self.streams).sigmoid() * 2

    @staticmethod
    def combine(state, output, weights):
        """The mixer's output injected into every stream, each with its weight. A pass of many tokens owns its streams and writes them in place: the VRAM of a copy of the streams less."""
        if expert_stream.ENABLED and state.shape[-2] > QUEUED_ROWS:
            hidden, output = output.shape[-1], output.float()
            for stream in range(weights.shape[-1]):
                state[..., stream * hidden:(stream + 1) * hidden].add_(output * weights[..., stream:stream + 1])
            return state
        return state + (output.float().unsqueeze(-2) * weights.unsqueeze(-1)).flatten(-2)


class Qwen4ExpMoE(nn.Module):
    """Routed experts (softmax top-k, an MMGP expert bank) and a shared expert behind a sigmoid gate."""
    def __init__(self, config, use_triton=False):
        super().__init__()
        hidden, intermediate = int(config.hidden_size), int(config.moe_intermediate_size)
        self.top_k = int(config.num_experts_per_tok)
        self.router = ColumnParallelLinear(hidden, int(config.num_experts), bias=False)
        self.experts = RoutedExperts(int(config.num_experts), hidden, intermediate, activation="swiglu", use_triton=use_triton)
        self.experts.streamed_pairs = 1 if use_triton else None  # the passes that do not run queued (more rows than QUEUED_ROWS)
        shared = int(config.shared_expert_intermediate_size)
        self.shared_router = ColumnParallelLinear(hidden, 1, bias=False)
        self.shared_gate = ColumnParallelLinear(hidden, shared, bias=False)
        self.shared_up = ColumnParallelLinear(hidden, shared, bias=False)
        self.shared_down = RowParallelLinear(shared, hidden, bias=False)

    def forward(self, x):
        rows = x.reshape(-1, x.shape[-1])
        ids, weights = softmax_topk(self.router(rows), self.top_k, True)
        output = routed_experts(self.experts, rows, ids, weights.float()).float()
        ids = weights = None
        shared = self.shared_down(F.silu(self.shared_gate(rows)) * self.shared_up(rows)).float()
        output = output + self.shared_router(rows).float().sigmoid() * shared
        return output.view(x.shape).to(x.dtype)

    def fused(self, hc, state):
        return hc.fused(state) and self.router.weight.is_cuda and self.shared_router.weight.is_cuda

    def forward_after(self, hc, state, output, weights):
        """The second half of a layer on the fused kernels: the output of its mixer injected (state, output, weights [1, tokens, ...]), the streams read through hc, the experts,
        their output injected. The rows of the routed experts stay in float32, in the host job's payload along with the routing."""
        state, output, weights = state[0], output[0], weights[0]
        tokens, hidden = state.shape[0], self.router.weight.shape[1]
        payload = torch.empty(tokens * hidden + 2 * tokens * self.top_k, dtype=torch.int32, device=state.device)
        rows = payload[:tokens * hidden].view(torch.float32).view(tokens, hidden)
        mixed, weights, state = hc.read(state, output, weights, rows)
        output = None
        logits, gate = hc_kernels.router_rows(rows, self.router.weight, self.shared_router.weight)
        ids = payload[tokens * (hidden + self.top_k):].view(tokens, self.top_k)
        probabilities = payload[tokens * hidden:tokens * (hidden + self.top_k)].view(torch.float32).view(tokens, self.top_k)
        hc_kernels.softmax_topk(logits, self.top_k, ids, probabilities)
        logits = None
        if SHARED_BESIDE:  # the shared expert reads nothing of the routed ones: queued before the wait for the CPU's, it runs while the CPU works
            beside = []
            routed = routed_experts(self.experts, rows, ids, probabilities, payload, lambda: beside.append(self.shared_down(hc_kernels.silu_product(self.shared_gate(mixed), self.shared_up(mixed)))))
            shared = beside[0]
        else:
            routed = routed_experts(self.experts, rows, ids, probabilities, payload)
            shared = self.shared_down(hc_kernels.silu_product(self.shared_gate(mixed), self.shared_up(mixed)))
        rows = ids = probabilities = payload = None
        return hc_kernels.combine_(state, routed, weights, hc.streams, hc.hidden, shared, gate)[None]


    def forward_rows(self, hc, state, output, weights):
        """The second half of a layer in a pass of many rows: the mixer's output injected and the streams read through hc in one pass, the experts, then their output and the shared expert's injected in one pass."""
        state, output, weights = state[0], output[0], weights[0]
        rows, weights = hc.read_rows(state, output, weights)
        output = None
        ids, probabilities = softmax_topk(self.router(rows), self.top_k, True)
        routed = routed_experts(self.experts, rows, ids, probabilities.float()).float()
        ids = probabilities = None
        shared = self.shared_down(hc_kernels.silu_product(self.shared_gate(rows), self.shared_up(rows)))
        gate = self.shared_router(rows).float().sigmoid()
        rows = None
        return hc_kernels.combine_(state, routed, weights, hc.streams, hc.hidden, shared, gate)[None]


class Qwen4ExpPLE(nn.Module):
    """Per-layer embedding: hashed n-grams of the tokens gate a value per stream, a dilated convolution adds their local context. The rows come from a table
    on the drive (table: lmgp.layers.ngram_table.NgramTable, set by the loader). The ids and convolution inputs are kept by position: a rejected draft is
    simply overwritten."""
    def __init__(self, config):
        super().__init__()
        hidden, self.streams = int(config.hidden_size), int(config.hc_count)
        self.hidden, self.eps = hidden, float(config.rms_norm_eps)
        width = hidden * self.streams
        embedding = int(config.ple_embed_dim)
        self.key = ColumnParallelLinear(embedding, width, bias=False)
        self.value = ColumnParallelLinear(embedding, hidden, bias=False)
        self.norm_key = nn.Parameter(torch.ones(width))
        self.norm_query = nn.Parameter(torch.ones(width))
        self.norm_conv = nn.Parameter(torch.ones(width))
        self.conv = nn.Parameter(torch.zeros(width, int(config.ple_conv_kernel_size)))
        self.kernel, self.dilation = int(config.ple_conv_kernel_size), int(config.ngram_size)
        self.context_length = int(config.ngram_size) - 1
        eos = config.eos_token_id
        self.eos = int(eos[0] if isinstance(eos, (list, tuple)) else eos)
        self.table = None
        self.id_ring = self.conv_ring = None
        self._lock_dtype = torch.float32  # its own float32 norm and convolution weights; its linears are not concerned

    def prepare_sequence_state(self, max_batch_size, device, dtype):
        if self.id_ring is None or self.id_ring.device != device:
            self.id_ring = torch.full((STATE_RING,), self.eos, dtype=torch.int32, device=device)
            self.conv_ring = torch.zeros(STATE_RING, self.streams * self.hidden, dtype=torch.float32, device=device)
        else:
            self.reset_sequence_state()

    def reset_sequence_state(self):
        if self.id_ring is not None:
            self.id_ring.fill_(self.eos)
            self.conv_ring.zero_()

    def release_sequence_state(self):
        self.id_ring = self.conv_ring = None
        if self.table is not None:
            self.table.release_jobs()

    def snapshot_sequence_state(self):
        return {name: getattr(self, name).to("cpu", copy=True) for name in ("id_ring", "conv_ring") if getattr(self, name) is not None}

    def restore_sequence_state(self, snapshot):
        for name, copy in (snapshot or {}).items():
            getattr(self, name).copy_(copy)  # into the live rings, which a new engine has reallocated and its graphs read

    def read_ahead(self, ids):
        """ids: int32 CPU tensor, the tokens of the next pass of a prompt after the context_length tokens before it: their rows are read while the GPU works."""
        before, tokens = self.context_length, ids.shape[0] - self.context_length
        columns, cut = [ids[before:]], torch.zeros(tokens, dtype=torch.bool, device="cpu")
        for back in range(1, before + 1):
            token = ids[before - back:before - back + tokens]
            cut = cut | (token == self.eos)
            columns.append(torch.where(cut, self.eos, token))
        self.table.read_ahead(torch.stack(columns, 1).contiguous())

    def _token_context(self, ids, sequence):
        """int32 [tokens, ngram_size]: each token and its predecessors; an end token cuts what is before it, and the start of the sequence reads as end tokens."""
        tokens = ids.shape[0]
        before = self.context_length
        previous = self.id_ring[(sequence[0] - before + torch.arange(before, device=ids.device)) % STATE_RING]
        window = torch.cat((previous, ids.to(torch.int32)))
        written = min(tokens, STATE_RING)
        self.id_ring[sequence[-written:] % STATE_RING] = ids[-written:].to(torch.int32)
        columns, cut = [window[before:]], torch.zeros(tokens, dtype=torch.bool, device=ids.device)
        for back in range(1, before + 1):
            token = window[before - back:before - back + tokens]
            cut = cut | (sequence < back) | (token == self.eos)
            columns.append(torch.where(cut, self.eos, token))
        return torch.stack(columns, 1)

    def forward(self, state, ids, sequence):
        tokens = state.shape[0]
        context = self._token_context(ids, sequence)
        rows = self.table.queue_embed(context) if tokens <= QUEUED_ROWS and state.is_cuda else self.table.embed(context, state.device)
        rows = rows.to(self.key.weight.dtype)
        key = _grouped_rms(self.key(rows), self.norm_key, self.hidden, self.eps).view(tokens, self.streams, self.hidden)
        value = self.value(rows).float()
        rows = None
        query = _grouped_rms(state, self.norm_query, self.hidden, self.eps).view(tokens, self.streams, self.hidden)
        score = (key * query).sum(-1, keepdim=True) / math.sqrt(self.hidden)
        key = query = None
        gate = (score.abs().clamp_min(1e-6).sqrt() * score.sign()).sigmoid()
        gated = (gate * value[:, None, :]).flatten(1)
        normalized = _grouped_rms(gated, self.norm_conv, self.hidden, self.eps)
        history = (self.kernel - 1) * self.dilation
        engine = get_context()
        if engine.is_prefill and not engine.has_previous_state and not engine.speculative_verify:
            previous = torch.zeros(history, normalized.shape[1], dtype=normalized.dtype, device=normalized.device)
        else:
            previous = self.conv_ring[(sequence[0] - history + torch.arange(history, device=ids.device)) % STATE_RING]
        padded = torch.cat((previous, normalized))
        written = min(tokens, STATE_RING)
        self.conv_ring[sequence[-written:] % STATE_RING] = normalized[-written:]
        output = gated
        tapped = None
        for tap in range(self.kernel):
            start = history - (self.kernel - 1 - tap) * self.dilation
            term = padded[start:start + tokens] * self.conv[:, tap].float()
            tapped = term if tapped is None else tapped + term
        return (state.float() + output + F.silu(tapped)).to(state.dtype)


class Qwen4ExpSigmoidGatedNorm(Qwen3_5RMSNormGated):
    """The Gated DeltaNet's output norm of Qwen3.5 with a sigmoid gate."""
    sigmoid_gate = True

    def forward(self, hidden_states, gate):
        return self.forward_list([hidden_states, gate])

    def forward_list(self, state_list):
        hidden_states, gate = state_list
        state_list.clear()
        if self.use_triton and hidden_states.is_cuda and gate.dtype == hidden_states.dtype and _fla_fused_norm_gate is not None and triton_dtype_supported(hidden_states.device, hidden_states.dtype):
            return _fla_fused_norm_gate.rms_norm_gated(hidden_states, gate, self.weight, None, "sigmoid", eps=self.eps)
        dtype = hidden_states.dtype
        output = rms_norm(hidden_states.float(), self.weight, self.eps) * gate.float().sigmoid()
        return output.to(dtype)


class Qwen4ExpBlock(Qwen3_5Block):
    def __init__(self, config, layer_idx, use_triton=False, ple=False):
        super().__init__(config, layer_idx)
        # the layer norms and the feed-forward of Qwen3.5 are replaced by the hyper-connections and the routed experts
        del self.attn_norm, self.post_attention_norm, self.ffn_gate, self.ffn_up, self.ffn_gate_up, self.ffn_down, self.mlp_act_fn
        self.eps = float(config.rms_norm_eps)
        self.hc_attn = Qwen4ExpHyperConnection(config)
        self.hc_ffn = Qwen4ExpHyperConnection(config)
        self.moe = Qwen4ExpMoE(config, use_triton)
        self.ple = Qwen4ExpPLE(config) if ple else None
        hidden = int(config.hidden_size)
        if self.layer_type == "full_attention":
            self.index_heads, self.index_dim = int(config.indexer_n_heads), int(config.indexer_head_dim)
            self.index_q = ColumnParallelLinear(hidden, self.index_heads * self.index_dim, bias=False)
            self.index_k = ColumnParallelLinear(hidden, self.index_dim, bias=False)
            self.index_q_norm = nn.Parameter(torch.ones(self.index_dim))
            self.index_k_norm = nn.Parameter(torch.ones(self.index_dim))
            attention = IndexedAttention(self.num_heads, self.head_dim, self.scaling, self.num_kv_heads, self.index_heads, self.index_dim, int(config.indexer_budget), int(config.indexer_compress_ratio), self.eps)
            attention.use_triton_kv_cache = self.attn.use_triton_kv_cache
            attention.flash_attn_varlen_func = attention.flash_attn_with_kvcache = None
            self.attn = attention
            self._lock_dtype = torch.float32  # its index norm weights; its linears are not concerned
        else:
            use = self.ssm_norm.use_triton
            self.ssm_norm = Qwen4ExpSigmoidGatedNorm(self.head_v_dim, eps=float(config.rms_norm_eps))
            self.ssm_norm.use_triton = use
            self._lock_dtype = torch.float32  # its float32 SSM parameters; its linears are not concerned

    def _forward_indexed(self, x, cos, sin, sequence):
        tokens = x.shape[0]
        if self.attn.fused(x, self.index_q_norm, self.index_k_norm, self.attn_q_norm.weight, self.attn_k_norm.weight):
            q_full = self.attn_q(x)
            if tokens <= fused_kernels.PROJECT_MAX_TOKENS and self.index_q.weight.is_cuda and self.index_k.weight.is_cuda:
                index_q, index_raw = fused_kernels.project_pair(x, self.index_q.weight, self.index_k.weight)
            else:
                index_q, index_raw = self.index_q(x), self.index_k(x)
            index_q = fused_kernels.norm_rope(index_q, self.index_q_norm, cos, sin, self.eps, self.index_heads, self.index_dim)
            q = fused_kernels.norm_rope(q_full, self.attn_q_norm.weight, cos, sin, self.eps, self.num_heads, self.head_dim, self.head_dim * 2)
            k = fused_kernels.norm_rope(self.attn_k(x), self.attn_k_norm.weight, cos, sin, self.eps, self.num_kv_heads, self.head_dim)
            output = self.attn.forward_fused(q, q_full, k, self.attn_v(x).view(tokens, self.num_kv_heads, self.head_dim), index_q, index_raw, cos, sin, sequence, self.index_k_norm)
            q = k = q_full = index_q = index_raw = None
            return self.attn_output(output.to(x.dtype))
        index_q = rotate_(rms_norm(self.index_q(x).float().view(tokens, self.index_heads, self.index_dim), self.index_q_norm, self.eps), cos, sin)
        index_raw = self.index_k(x).float()
        q, gate = self.attn_q(x).view(tokens, self.num_heads, self.head_dim * 2).chunk(2, dim=-1)
        q = rotate_(rms_norm(q.float(), self.attn_q_norm.weight, self.eps), cos, sin)
        k = rotate_(rms_norm(self.attn_k(x).float().view(tokens, self.num_kv_heads, self.head_dim), self.attn_k_norm.weight, self.eps), cos, sin)
        v = self.attn_v(x).float().view(tokens, self.num_kv_heads, self.head_dim)
        output = self.attn.forward_indexed(q, k, v, index_q, index_raw, cos, sin, sequence, self.index_k_norm)
        q = k = v = index_q = index_raw = None
        output = output * gate.float().sigmoid().reshape(tokens, -1)
        return self.attn_output(output.to(x.dtype))

    def forward(self, state, cos, sin, sequence, ids, layer_idx):
        """state [1, tokens, streams * hidden]"""
        if self.ple is not None:
            state = self.ple(state[0], ids, sequence)[None]
        many = state.shape[1] > QUEUED_ROWS and self.hc_attn.fused_rows(state)
        if many:
            mixed, weights = self.hc_attn.read_rows(state[0])
            mixed = mixed[None]
            weights = weights[None]
        else:
            mixed, weights = self.hc_attn(state)
        if self.layer_type == "full_attention":
            output = self._forward_indexed(mixed[0], cos, sin, sequence)[None]
        else:
            output = self._forward_linear_attention([mixed], layer_idx, None, None)
        mixed = None
        if many:
            return self.moe.forward_rows(self.hc_ffn, state, output, weights)
        if self.moe.fused(self.hc_ffn, state):
            return self.moe.forward_after(self.hc_ffn, state, output, weights)
        state = self.hc_attn.combine(state, output, weights)
        output = weights = None
        mixed, weights = self.hc_ffn(state)
        return self.hc_ffn.combine(state, self.moe(mixed), weights)


class Qwen4ExpOutput(nn.Module):
    """The last mix of the streams, which stands for the output norm, then the head."""
    def __init__(self, config):
        super().__init__()
        self.mixer = Qwen4ExpHyperConnection(config, inject=False)
        self.head = nn.Linear(int(config.hidden_size), int(config.vocab_size), bias=False)

    @property
    def weight(self):
        return self.head.weight

    def forward(self, state):
        return self.head(self.mixer(state))

    def lm_engine_graph_forward(self, state, logits):
        """The logits of a verification window into their buffer (cuda_graphs.run stage): no Python between the pass and the sampling."""
        logits.copy_(self(state))


class Qwen4ExpForCausalLM(nn.Module):
    """The LM engine's Qwen3.8 Flash-Next. forward returns the residual streams [1, tokens, streams * hidden], which the drafter also reads; output (or
    compute_logits) mixes them and applies the head."""
    _lm_engine_tower_prefix = "blk."
    _lm_engine_prefill_block_tables = True  # its attention reads the cache in every pass
    _lm_engine_ids_with_embeds = True  # its n-gram layer reads the token ids of embedded prompts too
    _lm_engine_adaptive_experts = True  # its routing follows the topic of the text: its GPU experts follow the routing

    def __init__(self, config, use_triton=False):
        super().__init__()
        self.config = config
        ple_layers = {layer - 1 for layer in config.ple_layer_ids}  # one-indexed in the configuration
        self.token_embd = nn.Embedding(int(config.vocab_size), int(config.hidden_size))
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config)
        self.blk = nn.ModuleList([Qwen4ExpBlock(config, index, use_triton, index in ple_layers) for index in range(int(config.num_hidden_layers))])
        self.output = Qwen4ExpOutput(config)
        self.streams = int(config.hc_count)
        self._lm_engine_output_size = self.streams * int(config.hidden_size)
        self.mtp = Qwen4ExpMTP(config, use_triton) if bool(getattr(config, "_prompt_enhancer_enable_mtp_speculative", False)) else None
        self._block_draft = False
        self._draft_features = None

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def embed_tokens(self):
        return self.token_embd

    @property
    def layers(self):
        return self.blk

    @property
    def lm_head(self):
        return self.output.head

    def lm_engine_read_ahead(self, token_ids, start, stop):
        """The tokens [start, stop) of token_ids are the next pass of a prompt: what a layer reads from a drive for them is read now."""
        if expert_stream.ENABLED and stop - start > QUEUED_ROWS:
            for block in self.blk:
                if block.ple is not None:
                    before = block.ple.context_length
                    ids = torch.tensor(token_ids[max(0, start - before):stop], dtype=torch.int32, device="cpu")
                    block.ple.read_ahead(torch.cat((torch.full((max(0, before - start),), block.ple.eos, dtype=torch.int32, device="cpu"), ids)))

    def lm_engine_prefill_chunk(self, tokens):
        """The tokens of a prompt read together: a whole prompt if it fits in the free VRAM, or else the longest chunk that does. Longer chunks read each expert out of VRAM once for more tokens."""
        if not expert_stream.ENABLED:
            return self._prefill_chunk_tokens
        room = runtime.free_vram() + runtime.lendable_experts(self) // 2 - PROMPT_VRAM_MARGIN - expert_stream.REGION_BYTES * expert_stream.MIN_RING_REGIONS
        if tokens * PROMPT_BYTES_PER_TOKEN <= room:
            return max(tokens, PROMPT_CHUNKS[-1])
        return next((chunk for chunk in PROMPT_CHUNKS if chunk * PROMPT_BYTES_PER_TOKEN <= room), PROMPT_CHUNKS[-1])

    def _sequence_positions(self, tokens, device):
        """The positions in the cache of the tokens of this pass."""
        context = get_context()
        if context.speculative_verify:
            first = context.cu_seqlens_k[1] - tokens
        elif context.is_prefill:
            first = context.cu_seqlens_k[1] - context.cu_seqlens_q[1]
        else:
            first = context.context_lens[0] - 1
        # a graph capture's pass on dummy pages has no context: position 0 keeps its attention, and what it writes, finite
        return first.long().clamp_min(0) + torch.arange(tokens, dtype=torch.long, device=device)

    def forward(self, input_ids=None, positions=None, inputs_embeds=None, **_kwargs):
        if input_ids is not None and input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        hidden = self.token_embd(input_ids) if inputs_embeds is None else inputs_embeds
        tokens, device = hidden.shape[1], hidden.device
        if positions.ndim == 3 and positions.shape[0] == 4:
            positions = positions[1:]
        cos, sin = _rotary_tables(self.rotary_emb, positions)
        sequence = self._sequence_positions(tokens, device)
        carry = [hidden.float().repeat(1, 1, self.streams)]  # the residual streams accumulate in float32: small contributions are lost in BF16
        hidden = None
        # MMGP: inside a graph scope, each run of blocks kept in VRAM is one graph, the streamed blocks go through their hooks
        if tokens <= QUEUED_ROWS or torch.cuda.is_current_stream_capturing():
            (state,) = cuda_graphs.run_blocks(self, self.blk, self._run_block, carry, cos[0], sin[0], sequence, None if input_ids is None else input_ids[0])
        else:
            with runtime.stream_experts(self, [block.moe.experts for block in self.blk], tokens, tokens * PROMPT_BYTES_PER_TOKEN + PROMPT_VRAM_MARGIN):  # the experts of a prompt chunk stream ahead of their use
                (state,) = cuda_graphs.run_blocks(self, self.blk, self._run_prompt_block, carry, cos[0], sin[0], sequence, None if input_ids is None else input_ids[0])
        return state

    @staticmethod
    def _run_block(layer_idx, block, carry, cos, sin, sequence, ids):
        state = carry[0]
        carry.clear()
        return [block(state, cos, sin, sequence, ids, layer_idx)]

    def _run_prompt_block(self, layer_idx, block, carry, cos, sin, sequence, ids):
        """A layer of a prompt pass, whose progress reaches a loading context (a first call reads the experts out of VRAM from the drive)."""
        carry = self._run_block(layer_idx, block, carry, cos, sin, sequence, ids)
        offload.report_progress("Reading the Prompt", layer_idx + 1, len(self.blk))
        return carry

    def lm_engine_prefetch(self, token_ids, first, offset=0):
        """The n-gram rows of the coming pass start reaching the cache before the pass: token_ids (a list), the tokens from position first being those of the pass. offset: the position of token_ids[0] in the sequence."""
        for block in self.blk:
            if block.ple is not None:
                ple = block.ple
                eos, before = ple.eos, ple.context_length
                context = []
                for position in range(first, len(token_ids)):
                    cut = False
                    context.append(token_ids[position])
                    for back in range(1, before + 1):
                        token = token_ids[position - back] if position - back >= 0 else eos
                        cut = cut or position - back + offset < 0 or token == eos
                        context.append(eos if cut else token)
                ple.table.prefetch(context, len(token_ids) - first)
                return

    def lm_engine_graph_forward(self, input_ids, positions, outputs, draft_features=None):
        """A decoding pass of the engine (cuda_graphs.run stage): the streams of input_ids written into outputs."""
        outputs.copy_(self(input_ids, positions).view(outputs.shape))

    def compute_logits(self, hidden_states):
        context = get_context()
        if hidden_states.ndim == 3:
            if context.is_prefill and context.cu_seqlens_q is not None:
                hidden_states = hidden_states[:, int(context.cu_seqlens_q[-1].item()) - 1, :]
            elif hidden_states.shape[-2] == 1:
                hidden_states = hidden_states[:, 0, :]
        return self.output(hidden_states)


class Qwen4ExpDraftBlock(nn.Module):
    """The drafter's layer: dense attention over its own K/V (no indexer), then routed experts."""
    def __init__(self, config, use_triton=False):
        super().__init__()
        hidden = int(config.hidden_size)
        self.heads, self.kv_heads, self.dim = int(config.num_attention_heads), int(config.num_key_value_heads), int(config.head_dim)
        self.eps, self.scaling = float(config.rms_norm_eps), self.dim ** -0.5
        self.hc_attn = Qwen4ExpHyperConnection(config)
        self.hc_ffn = Qwen4ExpHyperConnection(config)
        self.attn_q = ColumnParallelLinear(hidden, self.heads * self.dim * 2, bias=False)
        self.attn_k = ColumnParallelLinear(hidden, self.kv_heads * self.dim, bias=False)
        self.attn_v = ColumnParallelLinear(hidden, self.kv_heads * self.dim, bias=False)
        self.attn_output = RowParallelLinear(self.heads * self.dim, hidden, bias=False)
        self.attn_q_norm = RMSNorm(self.dim, eps=self.eps)
        self.attn_k_norm = RMSNorm(self.dim, eps=self.eps)
        self.attn_q_norm._lock_dtype = self.attn_k_norm._lock_dtype = torch.float32  # float32 weights, as stored
        self.moe = Qwen4ExpMoE(config, use_triton)


class Qwen4ExpMTP(nn.Module):
    """The single-layer next-token predictor of Flash-Next: the target's streams and the next token's embedding make its own streams, one layer refines them, a last mix and the
    target's head give the logits. Its streams feed the next draft. Its experts are small (Q2_0 rows) and all kept in VRAM: a draft must cost less than a target pass. Tokens of a
    prompt only write their K/V (compute_logits=False): the rest of the layer reads them, it needs nothing else from them."""
    def __init__(self, config, use_triton=False):
        super().__init__()
        hidden, self.streams = int(config.hidden_size), int(config.hc_count)
        self.hidden, self.eps = hidden, float(config.rms_norm_eps)
        self.embed_tokens = nn.Embedding(int(config.vocab_size), hidden)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config)
        self.pre_fc_norm_embedding = nn.Parameter(torch.ones(hidden))
        self.pre_fc_norm_hidden = nn.Parameter(torch.ones(hidden * self.streams))
        self.fc_embedding = ColumnParallelLinear(hidden, hidden, bias=False)
        self.fc_hidden = ColumnParallelLinear(hidden, hidden, bias=False)
        self.block = Qwen4ExpDraftBlock(config, use_triton)
        self.mixer = Qwen4ExpHyperConnection(config, inject=False)
        self.lm_head = nn.Linear(hidden, int(config.vocab_size), bias=False)
        self._lock_dtype = torch.float32  # its float32 norm weights; its linears are not concerned
        self._mm_graph_keys, self._mm_graph_check_once, self._mm_graph_placement_key = 32, True, True  # as the blocks of the target (lm_engine_graph_forward)
        self._cache = None
        self._cache_config = copy(config)
        self._cache_config.num_hidden_layers = 1
        self._cache_config.layer_types = ["full_attention"]

    def lm_engine_graph_forward(self, input_ids, positions, hidden_states, outputs, logits, next_token):
        """A pass of the engine (cuda_graphs.run stage): the streams, the logits and the likeliest token of the last token into their buffers."""
        state, head = self(input_ids, positions, hidden_states, last_logits_only=True, cache_prepared=True)
        outputs.copy_(state)
        logits.copy_(head)
        next_token.copy_(head[0, -1].argmax().reshape(1))

    def prepare_cache(self, max_cache_len, device, dtype):
        if self._cache is None or self._cache.max_cache_len != int(max_cache_len) or self._cache.key_cache[0].device != device or self._cache.key_cache[0].dtype != dtype:
            self._cache = Qwen3_5StaticCache(self._cache_config, max_cache_len, device, dtype)
        else:
            self._cache.reset()

    def reset_sequence_state(self):
        if self._cache is not None:
            self._cache.reset()

    def release_sequence_state(self):
        clear_head_cache = getattr(self.lm_head, "clear_cache", None)
        if callable(clear_head_cache):
            clear_head_cache()
        self._cache = None

    def snapshot_sequence_state(self, previous=None, reuse_tokens=0):
        return None if self._cache is None else self._cache.snapshot(previous=previous, reuse_tokens=reuse_tokens)

    def restore_sequence_state(self, snapshot):
        if snapshot is None:
            self.reset_sequence_state()
        else:
            self._cache.restore(snapshot)

    def get_cache_length(self):
        return self._cache.get_seq_length()

    def truncate_cache(self, seq_length):
        self._cache.truncate(seq_length)

    def forward(self, input_ids, positions, hidden_states, inputs_embeds=None, compute_logits=True, last_logits_only=False, cache_prepared=False):
        """hidden_states: the streams [1, tokens, streams * hidden], of the target or of the previous draft. (streams, logits), or nothing for the tokens of a prompt."""
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        tokens, block, cache = hidden_states.shape[1], self.block, self._cache
        if inputs_embeds is None:
            embedded = self.embed_tokens(input_ids)
        elif inputs_embeds.shape[1] == tokens - 1:
            embedded = torch.cat((inputs_embeds, self.embed_tokens(input_ids[:, -1:])), dim=1)
        else:
            embedded = inputs_embeds
        dtype = self.fc_hidden.weight.dtype
        fused = (DRAFT and block.hc_attn.fused(hidden_states[0]) and self.pre_fc_norm_hidden.is_cuda and self.pre_fc_norm_embedding.is_cuda and block.attn_q_norm.weight.is_cuda and block.attn_k_norm.weight.is_cuda
                 and dtype == torch.bfloat16 and fused_kernels is not None)  # a drafter's pass has about 100 small operations: one launch per norm, its rotation with it
        if fused:
            token = self.fc_embedding(hc_kernels.norm_rows(embedded[0].float(), self.pre_fc_norm_embedding, self.eps, 1, self.hidden)[1])
            embedded = None
            hidden = self.fc_hidden(hc_kernels.norm_rows(hidden_states[0], self.pre_fc_norm_hidden, self.eps, self.streams, self.hidden)[1].view(tokens * self.streams, self.hidden))
        else:
            token = self.fc_embedding(rms_norm(embedded[0].float(), self.pre_fc_norm_embedding, self.eps).to(dtype))
            embedded = None
            normed = _grouped_rms(hidden_states[0], self.pre_fc_norm_hidden, self.hidden, self.eps) if DRAFT else rms_norm(hidden_states[0].float(), self.pre_fc_norm_hidden, self.eps)
            hidden = self.fc_hidden(normed.view(tokens * self.streams, self.hidden).to(dtype))
            normed = None
        state = (hidden.view(tokens, self.streams, self.hidden) + token[:, None]).float().flatten(1)[None]
        hidden = token = None
        cos, sin = _rotary_tables(self.rotary_emb, positions)
        cos, sin = cos[0], sin[0]
        if not cache_prepared:
            cache.prepare_append()
        mixed, weights = block.hc_attn(state)
        rows = mixed[0]
        mixed = None
        if fused:
            k = fused_kernels.norm_rope(block.attn_k(rows), block.attn_k_norm.weight, cos, sin, self.eps, block.kv_heads, block.dim)
        else:
            k = rotate_(rms_norm(block.attn_k(rows).float().view(tokens, block.kv_heads, block.dim), block.attn_k_norm.weight, self.eps), cos, sin)
        v = block.attn_v(rows).view(tokens, block.kv_heads, block.dim)
        key, value = cache.key_cache[0], cache.value_cache[0]
        slots = cache.cache_seqlens + cache.cache_positions[:tokens]
        key.index_copy_(1, slots, k[None].to(key.dtype))
        value.index_copy_(1, slots, v[None].to(value.dtype))
        k = v = None
        if not compute_logits:
            if not cache_prepared:
                cache.advance(tokens)
            return None, None
        seqlens = cache.cache_seqlens
        if last_logits_only and tokens > 1:  # the tokens of a catch-up before the last one only write their K/V, as those of a prompt
            state, weights, rows, cos, sin = state[:, -1:], weights[:, -1:], rows[-1:], cos[-1:], sin[-1:]
            seqlens = seqlens + tokens - 1
            tokens = 1
        if fused:
            q_full = block.attn_q(rows)
            q = fused_kernels.norm_rope(q_full, block.attn_q_norm.weight, cos, sin, self.eps, block.heads, block.dim, block.dim * 2)
            attended = cache.attend_written(q[None].to(key.dtype), seqlens, 0, block.scaling)
            q = None
            output = block.attn_output(fused_kernels.gate_rows(attended[0].contiguous(), q_full, tokens * block.heads, block.dim, dtype))
            q_full = None
        else:
            q, gate = block.attn_q(rows).view(tokens, block.heads, block.dim * 2).chunk(2, dim=-1)
            q = rotate_(rms_norm(q.float(), block.attn_q_norm.weight, self.eps), cos, sin)
            attended = cache.attend_written(q[None].to(key.dtype), seqlens, 0, block.scaling)
            q = None
            output = block.attn_output((attended[0].float() * gate.float().sigmoid()).reshape(tokens, -1).to(dtype))
            gate = None
        attended = rows = None
        if block.moe.fused(block.hc_ffn, state):
            state = block.moe.forward_after(block.hc_ffn, state, output[None], weights)
        else:
            state = block.hc_attn.combine(state, output[None], weights)
            output = weights = None
            mixed, weights = block.hc_ffn(state)
            state = block.hc_ffn.combine(state, block.moe(mixed), weights)
        output = mixed = weights = None
        if not cache_prepared:
            cache.advance(hidden_states.shape[1])
        return state, self.lm_head(self.mixer(state))


# ----------------------------------------------------------------------------------------------------------------------------------------
# Loading from llama.cpp's GGUF layout

_HYPER_CONNECTIONS = {"norm.weight": "norm", "down.weight": "down.weight", "up.weight": "up.weight", "inject.weight": "inject.weight"}
_EXPERT_TENSORS = {"ffn_gate_exps.weight": "moe.experts.gate_exps", "ffn_up_exps.weight": "moe.experts.up_exps", "ffn_down_exps.weight": "moe.experts.down_exps"}
_BLOCK_TENSORS = {"ffn_gate_inp.weight": "moe.router.weight", "ffn_gate_shexp.weight": "moe.shared_gate.weight", "ffn_up_shexp.weight": "moe.shared_up.weight", "ffn_down_shexp.weight": "moe.shared_down.weight",
                  "indexer.q_proj.weight": "index_q.weight", "indexer.k_proj.weight": "index_k.weight", "indexer.q_norm.weight": "index_q_norm", "indexer.k_norm.weight": "index_k_norm",
                  "ple_key.weight": "ple.key.weight", "ple_value.weight": "ple.value.weight", "ple_norm_key.weight": "ple.norm_key", "ple_norm_query.weight": "ple.norm_query", "ple_norm_conv.weight": "ple.norm_conv",
                  "ssm_dt.bias": "ssm_dt"}


def gguf_state_dict_names(state_dict, quant_map=None, tied_map=None):
    """The GGUF tensors under the names of this module tree (a preprocess_sd of MMGP's loader). Expert rows become plain tensors in their storage."""
    new = {}
    for name, tensor in state_dict.items():
        if name == "per_layer_token_embd.weight":
            continue  # the n-gram table stays on the drive
        if name.startswith("output_hc_"):
            new["output.mixer." + _HYPER_CONNECTIONS[name[len("output_hc_"):]]] = tensor
        elif name == "output.weight":
            new["output.head.weight"] = tensor
        elif name.startswith("blk."):
            _, layer, rest = name.split(".", 2)
            prefix = f"blk.{layer}."
            if rest in _EXPERT_TENSORS:
                new[prefix + _EXPERT_TENSORS[rest]] = tensor.as_subclass(torch.Tensor)
            elif rest == "ffn_gate_inp_shexp.weight":
                new[prefix + "moe.shared_router.weight"] = tensor.as_subclass(torch.Tensor).view(1, -1)
            elif rest == "ple_conv1d.weight":
                new[prefix + "ple.conv"] = tensor.as_subclass(torch.Tensor).float()  # a small kernel stored in FP16, used in float32
            elif rest == "ssm_conv1d.weight":
                new[prefix + rest] = tensor.unsqueeze(1)
            elif rest.startswith(("hc_attn_", "hc_ffn_")):
                group, part = rest.split("_", 2)[1:]
                new[f"{prefix}hc_{group}." + _HYPER_CONNECTIONS[part]] = tensor
            else:
                new[prefix + _BLOCK_TENSORS.get(rest, rest)] = tensor
        else:
            new[name] = tensor
    return new, quant_map, tied_map


def draft_shared_weights(state_dict, quant_map=None, tied_map=None):
    """The drafter reads the target's token table and head (a postprocess_sd of MMGP's loader)."""
    for source, target in (("token_embd.weight", "mtp.embed_tokens.weight"), ("output.head.weight", "mtp.lm_head.weight")):
        state_dict[target] = state_dict[source]
        if quant_map is not None and source.rsplit(".", 1)[0] in quant_map:
            quant_map[target.rsplit(".", 1)[0]] = dict(quant_map[source.rsplit(".", 1)[0]])
    return state_dict, quant_map, tied_map


def open_ngram_table(table_path, metadata_path, config):
    """The n-gram table of the second shard: its place in the file, the hash from the first shard's metadata."""
    from shared.qtypes import gguf
    from .gguf_metadata import read_metadata
    from ..layers.ngram_table import NgramTable
    info = next(tensor for tensor in gguf._gguf_get_index(table_path).tensor_infos if tensor.name == "per_layer_token_embd.weight")
    values, rows = info.raw_shape
    block, size = gguf._quantization_sizes(info.tensor_type)
    if block != 32 or size != 18:
        raise NotImplementedError(f"The n-gram table reader expects IQ4_NL rows, not {info.tensor_type}")
    metadata = read_metadata(metadata_path, ("qwen4exp.ple.layer_multipliers", "qwen4exp.ple.head_offsets", "qwen4exp.ple.head_vocab_sizes"))
    return NgramTable(table_path, info.data_offset, rows, values // block * size, values, int(config.ngram_size), int(config.heads_per_ngram),
                      metadata["qwen4exp.ple.layer_multipliers"], metadata["qwen4exp.ple.head_offsets"], metadata["qwen4exp.ple.head_vocab_sizes"])
