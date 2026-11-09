"""Single-GPU DSpark and DFlash2 inference (SpecForge/DeepSpec, MIT) and DSpine inference (Tencent DSpine, Apache 2.0, see
DSPINE_LICENSE.txt: the code bridge and its serial last-write refinement follow sglang_dspine's reference arithmetic).

Checkpoint-native BF16 weights; MMGP owns weight placement. Draft context
contains only verified target features, never rejected proposal states.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP, Qwen3RMSNorm, Qwen3RotaryEmbedding, rotate_half

from ..layers import attention as shared_attention
from shared.utils.cancellation import check_cancelled
from mmgp import transfer

_REWIND_ROWS = 4096  # rows kept beyond a sliding window: a rewind up to this far stays exact
_SHIFT_ROWS = 2048  # rows appended between two moves of the kept rows


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
        self.causal = bool(config.dflash_config.get("draft_block_causal", False))  # DSpine: causal inside the block, every context row visible
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
        result = shared_attention.flash_attn_with_kvcache(q, context_k, context_v, k=k, v=v, cache_seqlens=cache_lengths, causal=self.causal, window_size=(-1, -1) if window is None else (window - 1, -1))
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


class CodeBridge(nn.Module):
    """DSpine's code bridge: after every draft layer each block row receives the code of the token its predecessor predicts (the anchor
    row, the exact code of its token). The last layer's pre-write state feeds the serial refinement, which redoes that write with the
    exact code of each candidate sender."""
    def __init__(self, config):
        super().__init__()
        cfg = config.dflash_config
        bridge = cfg["selected_prefix_code_bridge"]
        if [int(layer) for layer in cfg["selected_prefix_feature_layers"]] != list(range(1, config.num_hidden_layers + 1)) or cfg.get("selected_prefix_terminal", True) \
                or cfg.get("selected_prefix_feature_write_mode") != "independent" or cfg["selected_prefix_last_write_refine"]["kind"] != "exact_sender":
            raise ValueError("Unsupported DSpine draft: a terminal-disabled bridge with independent writes after every layer and an exact-sender refinement is expected.")
        self.code_size, self.rank, self.eps = int(bridge["code_size"]), int(bridge["rank"]), float(bridge["eps"])
        layers, hidden = config.num_hidden_layers, config.hidden_size
        self.prev = nn.Linear(self.code_size, self.rank, bias=False)
        self.gate = nn.Linear(self.code_size, self.rank, bias=True)
        self.predecessor_gate = nn.Linear(self.code_size, self.rank, bias=False)
        self.codebook = nn.Parameter(torch.empty(config.vocab_size, self.code_size))
        self.read_weight = nn.Parameter(torch.empty(self.code_size, hidden))
        self.early_read_weights = nn.Parameter(torch.empty(layers - 1, self.code_size, hidden))
        self.feature_write_weights = nn.Parameter(torch.empty(layers, hidden, self.rank))

    def forward(self, out, anchor, layer, next_norm):
        """The write after draft layer `layer` on the BF16 stream out [1, T, H]: (next_norm(updated), updated, pre-write (out, rms, query of rows 1..))."""
        last = layer == self.feature_write_weights.shape[0] - 1
        scale = math.sqrt(self.code_size)
        f = out.float()
        rms = f.square().mean(-1).clamp_min(torch.finfo(torch.float32).tiny).sqrt()
        projected = F.linear((f / (rms.unsqueeze(-1) + self.eps)).to(out.dtype), self.read_weight if last else self.early_read_weights[layer]).float()
        codes = (scale * projected / (projected.norm(dim=-1, keepdim=True) + self.eps)).to(out.dtype)
        codes[:, 0] = (scale * self.codebook[anchor].float()).to(out.dtype)
        query = F.linear(codes[:, 1:], self.gate.weight)
        gate = torch.sigmoid(query.float() + F.linear(codes[:, :-1], self.predecessor_gate.weight).float() + self.gate.bias.float())
        message = (gate * F.linear(codes[:, :-1], self.prev.weight).float()).to(out.dtype)
        delta = F.linear(message, self.feature_write_weights[layer]).float()
        updated = torch.cat((out[:, :1], (f[:, 1:] + rms[:, 1:, None] * delta).to(out.dtype)), dim=1)
        return next_norm(updated), updated, (out, rms, query)

    def refinement_table(self, raw, rms, query, anchor, candidates, head_rows, norm):
        """FP32 [n, K, K] logits of proposal row i, sender lane j (row 0: the anchor), candidate c: the last write redone with the exact
        code of the sender's token, normalized and projected on the target head rows of the candidates (sglang_dspine serial_rs_table).
        raw [n, H] BF16 pre-write stream, rms [n], query [n, R], candidates [n, K], head_rows [n, K, H]."""
        f32 = torch.float32
        senders = torch.cat((anchor.reshape(1, 1).expand(1, candidates.shape[1]), candidates[:-1]))
        exact = math.sqrt(self.code_size) * self.codebook[senders].to(f32)
        gate = torch.sigmoid(query.to(f32).unsqueeze(1) + exact @ self.predecessor_gate.weight.to(f32).T + self.gate.bias.to(f32))
        message = gate * (exact @ self.prev.weight.to(f32).T)
        refined = raw.to(f32).unsqueeze(1) + rms[:, None, None] * (message @ self.feature_write_weights[-1].to(f32).T)
        normalized = refined / torch.sqrt(refined.square().mean(-1, keepdim=True) + norm.variance_epsilon) * norm.weight.to(f32)
        return torch.einsum("njh,nch->njc", normalized, head_rows.to(f32))


class BlockDraft(nn.Module):
    def __init__(self, config, method, block_size=None):
        super().__init__()
        self.config = config
        self.method = method
        self.target_layer_ids = config.dflash_config["target_layer_ids"]
        # DSpine's block attention is causal: a shorter served block yields exactly the first rows of its trained block.
        self.block_size = block_size or config.dflash_config.get("block_size", getattr(config, "block_size", 8))
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
        elif method == "dspine":
            self.selected_prefix_bridge = CodeBridge(config)
            self.head_topk = int(config.dflash_config["selected_prefix_last_write_refine"]["head_topk"])
        else:
            self.candidate_selector = CandidateSelector(config)
        self._keys = self._values = None
        self._length = self._offset = 0
        self._proposal_graph = None
        self._append_graphs = {}
        self._append_graph_pool = None
        self.draft_vocab_size = config.vocab_size

    def prepare_cache(self, max_length, device, dtype):
        self._append_graphs.clear()
        self._append_graph_pool = None
        # Preserve the drafter's checkpoint precision independently of the target.
        dtype = self.fc.weight.dtype
        # A sliding-window drafter reads the last `window` context rows only: the cache keeps them plus a rewind margin, and drops its
        # oldest rows in one move when full. Rows hold consecutive positions from _offset on, so attention distances are unchanged.
        self._window = self.config.sliding_window if self.config.use_sliding_window else None
        self._capacity = max_length if self._window is None else min(max_length, self._window + _REWIND_ROWS + _SHIFT_ROWS)
        shape = (len(self.layers), 1, self._capacity + self.block_size, self.config.num_key_value_heads, self.config.head_dim)
        self._keys = torch.empty(shape, dtype=dtype, device=device)
        self._values = torch.empty_like(self._keys)
        self._positions = torch.empty(self._capacity, dtype=torch.long, device=device)
        self._length = self._offset = 0
        self._cache_lengths = torch.zeros(1, dtype=torch.int32, device=device)
        self._proposal_graph = None
        self._proposal_buffers = None

    def get_cache_length(self):
        return self._length

    def truncate_cache(self, length):
        # A rewind past the kept rows leaves the next blocks a shorter context until it refills: drafts only, verification is exact.
        self._length, self._offset = length, min(self._offset, length)
        self._cache_lengths.fill_(length - self._offset)

    def reset_sequence_state(self):
        self._length = self._offset = 0
        if self._keys is not None:
            self._cache_lengths.zero_()

    def _make_room(self, count):
        """Before appending `count` rows to a full cache: the newest window and rewind margin move to its start."""
        rows = self._length - self._offset
        if rows + count <= self._capacity:
            return
        keep = min(rows, self._window + _REWIND_ROWS, self._capacity - count)
        for cache in (self._keys, self._values):
            for layer in cache:  # overlapping ranges: one layer's rows at a time through a copy
                layer[:, :keep] = layer[:, rows - keep:rows].clone()
        self._positions[:keep] = self._positions[rows - keep:rows].clone()
        self._offset = self._length - keep

    def release_sequence_state(self):
        self._append_graphs.clear()
        self._append_graph_pool = None
        self._proposal_graph = None
        self._proposal_buffers = None
        self._keys = self._values = None
        self._positions = None
        self._length = self._offset = 0

    def append_context(self, features, positions):
        count = features.shape[1]
        if count == 0:
            return
        check_cancelled()
        positions = positions[0, 0] if positions.ndim == 3 else positions.reshape(-1)
        if count > self._capacity:  # a long suffix: only its newest rows can be read again
            self._length += count - self._capacity
            self._offset = self._length
            features, positions, count = features[:, -self._capacity:], positions[-self._capacity:], self._capacity
        self._make_room(count)
        start, end = self._length - self._offset, self._length - self._offset + count
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
        self._length += count
        self._cache_lengths.fill_(end)

    def _append_context_tensors(self, features, positions, indices=None):
        x = self.hidden_norm(self.fc(features.to(self.fc.weight.dtype)))
        cos, sin = self.rotary_emb(x, positions.unsqueeze(0))
        start, end = self._length - self._offset, self._length - self._offset + x.shape[1]
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
        if self.method == "dspine":
            return self._forward_dspine(ids[:, 0], x, cos, sin, window, output)
        for index, layer in enumerate(self.layers):
            check_cancelled()
            x = layer(x, cos, sin, self._keys[index], self._values[index], window, self._cache_lengths)
        x = self.norm(x)
        if self.method == "dflash2":
            x = x[:, 1:]
        # Target GGUF output handles its own compute dtype and Bonsai rotation.
        return x[0], output(x)[0].float()

    def _forward_dspine(self, anchor, x, cos, sin, window, output):
        """(refinement table [n, K, K], candidate ids [n, K] ascending) of the block's proposal rows 1.."""
        bridge, last = self.selected_prefix_bridge, len(self.layers) - 1
        residual, y = x, self.layers[0].input_layernorm(x)
        for index, layer in enumerate(self.layers):
            check_cancelled()
            h = residual + layer.self_attn(y, cos, sin, self._keys[index], self._values[index], window, self._cache_lengths)
            out = h + layer.mlp(layer.post_attention_layernorm(h))
            y, residual, (raw, rms, query) = bridge(out, anchor, index, self.layers[index + 1].input_layernorm if index < last else self.norm)
        candidates = output(y[:, 1:])[0].float().topk(self.head_topk).indices.sort().values
        head_rows = output.weight.embedding(candidates, dtype=torch.float32)  # the GGUF head's packed rows of the candidates, decoded
        return bridge.refinement_table(raw[0, 1:], rms[0, 1:], query[0], anchor, candidates, head_rows, self.norm), candidates

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
        rows = self._length - self._offset
        return {"seq_length": self._length, "offset": self._offset, "keys": transfer.move(self._keys[:, :, :rows], "cpu"), "values": transfer.move(self._values[:, :, :rows], "cpu"), "positions": self._positions[:rows].to("cpu", copy=True)}

    def restore_sequence_state(self, snapshot):
        if snapshot is None:
            self.reset_sequence_state()
            return
        self._length, self._offset = snapshot["seq_length"], snapshot["offset"]
        rows = self._length - self._offset
        transfer.copy_(self._keys[:, :, :rows], snapshot["keys"])
        transfer.copy_(self._values[:, :, :rows], snapshot["values"])
        self._positions[:rows].copy_(snapshot["positions"])
        self._cache_lengths.fill_(rows)
