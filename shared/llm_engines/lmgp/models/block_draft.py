"""Single-GPU DSpark and DFlash2 inference (SpecForge/DeepSpec, MIT).

Checkpoint-native BF16 weights; MMGP owns weight placement. Draft context
contains only verified target features, never rejected proposal states.
"""
from __future__ import annotations

import torch
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm, Qwen3RotaryEmbedding, rotate_half

from ..layers import attention as shared_attention
from shared.utils.cancellation import check_cancelled
from mmgp import transfer


def rotate(x, cos, sin):
    return x * cos.unsqueeze(2) + rotate_half(x) * sin.unsqueeze(2)


class GroupedConv(nn.Module):
    def __init__(self, config):
        super().__init__()
        cfg = config.dflash_config
        self.taps = cfg["conv_kernel_size"]
        self.group_size = cfg["conv_group_size"]
        self.groups = config.hidden_size // self.group_size
        self.base_kernel = nn.Parameter(torch.empty(2, self.taps, config.hidden_size))
        self.kernel_projection = nn.Linear(config.hidden_size, 2 * self.taps * self.groups, bias=False)

    def convolve(self, x, delta, side):
        coefficients = self.base_kernel[side].reshape(1, 1, self.taps, self.groups, self.group_size) + delta.unsqueeze(-1)
        grouped = x.reshape(*x.shape[:2], self.groups, self.group_size)
        out = coefficients[:, :, 0] * grouped
        for tap in range(1, self.taps):
            out[:, tap:] += coefficients[:, tap:, tap] * grouped[:, :-tap]
        return out.reshape_as(x)

    def prepare(self, x):
        delta = self.kernel_projection(x).reshape(*x.shape[:2], 2, self.taps, self.groups)
        return self.convolve(x, delta[:, :, 0], 0), delta[:, :, 1]


class DraftAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.head_dim
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.q_proj = nn.Linear(config.hidden_size, self.heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.kv_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def context_kv(self, x, cos, sin):
        k = self.k_norm(self.k_proj(x).reshape(*x.shape[:2], self.kv_heads, self.head_dim))
        v = self.v_proj(x).reshape(*x.shape[:2], self.kv_heads, self.head_dim)
        return rotate(k, cos, sin), v

    def forward(self, x, cos, sin, context_k, context_v, window, cache_lengths):
        q = self.q_norm(self.q_proj(x).reshape(*x.shape[:2], self.heads, self.head_dim))
        q = rotate(q, cos, sin)
        k, v = self.context_kv(x, cos, sin)
        result = shared_attention.flash_attn_with_kvcache(q, context_k, context_v, k=k, v=v, cache_seqlens=cache_lengths, causal=False, window_size=(-1, -1) if window is None else (window - 1, -1))
        return self.o_proj(result.reshape(*x.shape[:2], -1))


class DraftLayer(nn.Module):
    def __init__(self, config, method):
        super().__init__()
        self.self_attn = DraftAttention(config)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        if method == "dflash2":
            self.attention_conv = GroupedConv(config)
            self.mlp_conv = GroupedConv(config)
        else:
            self.attention_conv = self.mlp_conv = None

    def forward(self, x, cos, sin, k, v, window, cache_lengths):
        y = self.input_layernorm(x)
        if self.attention_conv is not None:
            y, delta = self.attention_conv.prepare(y)
        y = self.self_attn(y, cos, sin, k, v, window, cache_lengths)
        if self.attention_conv is not None:
            y = self.attention_conv.convolve(y, delta, 1)
        x = x + y
        y = self.post_attention_layernorm(x)
        if self.mlp_conv is not None:
            y, delta = self.mlp_conv.prepare(y)
        y = self.mlp(y)
        if self.mlp_conv is not None:
            y = self.mlp_conv.convolve(y, delta, 1)
        return x + y


class MarkovHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.markov_w1 = nn.Embedding(config.vocab_size, config.markov_rank)
        self.markov_w2 = nn.Linear(config.markov_rank, config.vocab_size, bias=False)


class CandidateSelector(nn.Module):
    def __init__(self, config):
        super().__init__()
        rank = config.dflash_config["selector_rank"]
        self.top_k = config.dflash_config["selector_top_k"]
        self.predecessor_codebook = nn.Parameter(torch.empty(config.vocab_size, rank))
        self.successor_codebook = nn.Parameter(torch.empty(config.vocab_size, rank))
        self.hidden_projection = nn.Linear(config.hidden_size, rank, bias=False)


class BlockDraft(nn.Module):
    def __init__(self, config, method):
        super().__init__()
        self.config = config
        self.method = method
        self.target_layer_ids = config.dflash_config["target_layer_ids"]
        self.block_size = config.dflash_config.get("block_size", getattr(config, "block_size", 8))
        self.mask_token_id = config.dflash_config["mask_token_id"]
        self.layers = nn.ModuleList([DraftLayer(config, method) for _ in range(config.num_hidden_layers)])
        self.fc = nn.Linear(len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        if method == "dspark":
            self.markov_head = MarkovHead(config)
            self.confidence_head = nn.Module()
            self.confidence_head.proj = nn.Linear(config.hidden_size + config.markov_rank, 1)
        else:
            self.candidate_selector = CandidateSelector(config)
        self._keys = self._values = None
        self._length = 0
        self._proposal_graph = None
        self._append_graphs = {}
        self._append_graph_pool = None
        self.draft_vocab_size = config.vocab_size

    def prepare_cache(self, max_length, device, dtype):
        self._append_graphs.clear()
        self._append_graph_pool = None
        # Preserve the drafter's checkpoint precision independently of the target.
        dtype = self.fc.weight.dtype
        shape = (len(self.layers), 1, max_length + self.block_size, self.config.num_key_value_heads, self.config.head_dim)
        self._keys = torch.empty(shape, dtype=dtype, device=device)
        self._values = torch.empty_like(self._keys)
        self._positions = torch.empty(max_length, dtype=torch.long, device=device)
        self._length = 0
        self._cache_lengths = torch.zeros(1, dtype=torch.int32, device=device)
        self._proposal_graph = None
        self._proposal_buffers = None

    def get_cache_length(self):
        return self._length

    def truncate_cache(self, length):
        self._length = length
        self._cache_lengths.fill_(length)

    def reset_sequence_state(self):
        self._length = 0
        if self._keys is not None:
            self._cache_lengths.zero_()

    def release_sequence_state(self):
        self._append_graphs.clear()
        self._append_graph_pool = None
        self._proposal_graph = None
        self._proposal_buffers = None
        self._keys = self._values = None
        self._positions = None
        self._length = 0

    def append_context(self, features, positions):
        count = features.shape[1]
        if count == 0:
            return
        check_cancelled()
        positions = positions[0, 0] if positions.ndim == 3 else positions.reshape(-1)
        start, end = self._length, self._length + count
        if features.is_cuda and count <= self.block_size:
            state = self._append_graphs.get(count)
            if state is None:
                inputs = torch.empty_like(features, memory_format=torch.contiguous_format)
                fixed_positions = torch.empty_like(positions)
                indices = torch.arange(start, end, device=positions.device)
                inputs.copy_(features)
                fixed_positions.copy_(positions)
                self._append_context_tensors(inputs, fixed_positions, indices)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=self._append_graph_pool, capture_error_mode="thread_local"):
                    self._append_context_tensors(inputs, fixed_positions, indices)
                self._append_graph_pool = graph.pool()
                state = (graph, inputs, fixed_positions, indices)
                self._append_graphs[count] = state
            graph, inputs, fixed_positions, indices = state
            inputs.copy_(features)
            fixed_positions.copy_(positions)
            torch.arange(start, end, device=indices.device, out=indices)
            graph.replay()
        else:
            self._append_context_tensors(features, positions)
        self._length = end
        self._cache_lengths.fill_(end)

    def _append_context_tensors(self, features, positions, indices=None):
        x = self.hidden_norm(self.fc(features.to(self.fc.weight.dtype)))
        cos, sin = self.rotary_emb(x, positions.unsqueeze(0))
        start, end = self._length, self._length + x.shape[1]
        for index, layer in enumerate(self.layers):
            check_cancelled()
            k, v = layer.self_attn.context_kv(x, cos, sin)
            if indices is None:
                self._keys[index, :, start:end].copy_(k)
                self._values[index, :, start:end].copy_(v)
            else:
                self._keys[index].index_copy_(1, indices, k)
                self._values[index].index_copy_(1, indices, v)
        if indices is None:
            self._positions[start:end].copy_(positions)
        else:
            self._positions.index_copy_(0, indices, positions)

    def forward(self, anchor, start_position, embedding, output):
        device = self._keys.device
        ids = torch.full((1, self.block_size), self.mask_token_id, dtype=torch.long, device=device)
        ids[0, 0].fill_(anchor)
        positions = torch.arange(start_position, start_position + self.block_size, device=device).unsqueeze(0)
        return self._forward_block(ids, positions, embedding, output)

    def _forward_block(self, ids, positions, embedding, output):
        x = embedding(ids).to(self.fc.weight.dtype)
        cos, sin = self.rotary_emb(x, positions)
        window = self.config.sliding_window if self.config.use_sliding_window else None
        for index, layer in enumerate(self.layers):
            check_cancelled()
            x = layer(x, cos, sin, self._keys[index], self._values[index], window, self._cache_lengths)
        x = self.norm(x)
        if self.method == "dflash2":
            x = x[:, 1:]
        # Target GGUF output handles its own compute dtype and Bonsai rotation.
        return x[0], output(x)[0].float()

    def propose(self, anchor, start_position, embedding, output):
        check_cancelled()
        if self._proposal_graph is None:
            ids = torch.full((1, self.block_size), self.mask_token_id, dtype=torch.long, device=self._keys.device)
            ids[0, 0].fill_(anchor)
            positions = torch.arange(start_position, start_position + self.block_size, device=ids.device).unsqueeze(0)
            self._forward_block(ids, positions, embedding, output)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                hidden, logits = self._forward_block(ids, positions, embedding, output)
            self._proposal_buffers = (ids, positions, hidden, logits)
            self._proposal_graph = graph
        ids, positions, hidden, logits = self._proposal_buffers
        ids[0, 0].fill_(anchor)
        torch.arange(start_position, start_position + self.block_size, device=positions.device, out=positions[0])
        self._proposal_graph.replay()
        return hidden, logits

    def proposal_logits(self, hidden, logits, previous):
        previous = torch.as_tensor(previous, dtype=torch.long, device=hidden.device)
        if self.method == "dspark":
            markov = self.markov_head.markov_w1(previous)
            scores = logits + self.markov_head.markov_w2(markov).float()
            confidence = self.confidence_head.proj(torch.cat((hidden, markov), dim=-1)).sigmoid()
            return scores, confidence
        scores, candidates = self.proposal_candidates(hidden, logits, previous)
        full = torch.full_like(logits, -torch.inf)
        full.scatter_(0, candidates, scores)
        return full, None

    def proposal_candidates(self, hidden, logits, previous):
        selector = self.candidate_selector
        unary, candidates = logits.topk(selector.top_k)
        if torch.is_tensor(previous):
            predecessor = selector.predecessor_codebook.index_select(0, previous.reshape(1))[0]
        else:
            predecessor = selector.predecessor_codebook[previous]
        state = predecessor * selector.hidden_projection(hidden)
        scores = unary + torch.einsum("r,kr->k", state, selector.successor_codebook[candidates]).float()
        return scores, candidates

    def snapshot_sequence_state(self, previous=None, reuse_tokens=0):
        # CPU copies are blocking; snapshots survive vision lending and teardown.
        if self._keys is None:  # no cache: an empty state
            return None
        return {"seq_length": self._length, "keys": transfer.move(self._keys[:, :, :self._length], "cpu"), "values": transfer.move(self._values[:, :, :self._length], "cpu"), "positions": self._positions[:self._length].to("cpu", copy=True)}

    def restore_sequence_state(self, snapshot):
        if snapshot is None:
            self.reset_sequence_state()
            return
        self._length = snapshot["seq_length"]
        transfer.copy_(self._keys[:, :, :self._length], snapshot["keys"])
        transfer.copy_(self._values[:, :, :self._length], snapshot["values"])
        self._positions[:self._length].copy_(snapshot["positions"])
        self._cache_lengths.fill_(self._length)
