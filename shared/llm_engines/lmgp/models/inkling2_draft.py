# Copyright WanGP. Licensed under the WanGP License 2.0.
# DSpark block drafter of Inkling 2 (mtp.safetensors) for the LM engine. Algorithm after vLLM's Inkling 2 DSpark support (Apache 2.0),
# NVIDIA NeMo Automodel (Apache 2.0) and llama.cpp (MIT); numerical specification: lmgp.models.inkling2_draft_core.
"""Inkling 2's DSpark drafter behind the engine's block-drafter protocol (models.block_draft.BlockDraft, engine.block_draft_runner).

A draft is one parallel pass over a block of 8 rows (the last accepted token, then 7 masked slots) that attends keys and values made
from the target's features of every committed position, then a small recurrent head that turns the 7 slots into tokens one after the
other (proposal_step, run by the runner's chain).

- The context keys and values are kept by position: every position for the layer that reads the whole context, a ring of the last
  positions for the two sliding-window layers. A pass of the target is appended whole (the rows of its buffer) and the cache length
  moves by the accepted rows only: what a rejected draft wrote lies beyond the length, is never read, and is overwritten by the
  next pass. A block writes its own keys and values there too (the rows of its positions, beyond the length) and attends the caches in
  place, only the positions it reads (block_attend of the target's attention kernels, or the live rows in PyTorch).
- The head's logits U (h + H latent + b) are computed as U h, once for the 7 rows, plus (U H) latent + U b per token: U H and U b
  are made when the checkpoint loads (draft_head_tensors), so that a token costs a product over 257 columns instead of the vocabulary's 2304.
  Drafts only: verification by the target is exact whatever the drafter computes.
- Its passes are stages of the target's MMGP graphs (the cache and the buffers below are the engine's registered state)."""
import os

import torch
from torch import nn
from torch.nn import functional as F
from mmgp import cuda_graphs, transfer

from . import inkling2
from .inkling2 import _Tensors, _linear
from .inkling2_core import DRAFT_FEATURES, rms, softcap

TIED = ("embed", "embed_norm", "unembed")  # the target's modules whose weights the drafter's stages read (Inkling2Draft.tied)
REWIND_ROWS = 4096  # positions a window ring keeps beyond its window: a rewind up to this far leaves the drafts their whole context
CONFIDENT = "confident_drafts" not in os.environ.get("WGP_INKLING2_DISABLE", "").split(",")  # disabled: the automatic count from the measured history only (MeasuredDrafts), for comparisons
IN_PLACE = "draft_attention" not in os.environ.get("WGP_INKLING2_DISABLE", "").split(",")  # the block's keys and values written at its positions in the caches, attended there up to the block (the target's attention kernels when it has them); disabled: the previous path (copies of the whole caches, scores over all their rows), for comparisons


def _rotate(x, cos, sin):
    """Half-split rotation of x [rows, heads, dim] (float32) by the angles of its rows [rows, 1, dim / 2]."""
    a, b = x.float().chunk(2, -1)
    return torch.cat((a * cos - b * sin, b * cos + a * sin), -1)


class _DraftAttention(nn.Module):
    def __init__(self, hidden, heads, kv_heads, dim):
        super().__init__()
        self.q_proj, self.o_proj = _linear(hidden, heads * dim), _linear(heads * dim, hidden)
        self.k_proj, self.v_proj, self.k_proj_ctx, self.v_proj_ctx = (_linear(hidden, kv_heads * dim) for _ in range(4))
        self.q_norm, self.k_norm, self.k_norm_ctx = (_Tensors(weight=(dim,)) for _ in range(3))


class _DraftLayer(nn.Module):
    def __init__(self, hidden, intermediate, heads, kv_heads, dim):
        super().__init__()
        self.self_attn = _DraftAttention(hidden, heads, kv_heads, dim)
        self.mlp = nn.Module()
        self.mlp.gate_proj, self.mlp.up_proj, self.mlp.down_proj = _linear(hidden, intermediate), _linear(hidden, intermediate), _linear(intermediate, hidden)
        self.dflare_norm, self.input_layernorm, self.post_attention_layernorm = (_Tensors(weight=(hidden,)) for _ in range(3))


class Inkling2Draft(nn.Module):
    method = "dspark"
    confident_drafts = CONFIDENT  # an automatic count chooses each window's drafts from the head's probabilities of that window's drafts (engine.draft_sources.ConfidentDrafts)

    def __init__(self, config, target):
        """config: the checkpoint's mtp_config; target: the Inkling2ForCausalLM whose token table, head and features it reads."""
        super().__init__()
        self.__dict__["target"] = target  # not a submodule: the drafter is one of the target's
        hidden, self.heads, self.kv_heads, self.dim = config["hidden_size"], config["num_attention_heads"], config["num_key_value_heads"], config["head_dim"]
        self.hidden, self.block_size, self.window, self.local = hidden, config["block_size"], config["sliding_window_size"], tuple(config["local_layer_ids"])
        self.eps, self.theta, self.rank = config["rms_norm_eps"], config["rope_theta"], config["markov_rank"]
        self.target_layer_ids = list(config["target_layer_ids"])
        if self.target_layer_ids != [*DRAFT_FEATURES, len(target.schedule)]:
            raise ValueError(f"The drafter reads the target features {self.target_layer_ids}, the model gives {[*DRAFT_FEATURES, len(target.schedule)]}.")
        self.draft_vocab_size, self.cap = target.c.vocab, target.c.logit_softcap
        self.layers = nn.ModuleList(_DraftLayer(hidden, config["intermediate_size"], self.heads, self.kv_heads, self.dim) for _ in range(config["num_hidden_layers"]))
        self.fuse_weight = nn.Parameter(torch.empty(len(self.layers) + 1, len(self.target_layer_ids), dtype=torch.bfloat16), requires_grad=False)
        self.mask_embed = nn.Parameter(torch.empty(hidden, dtype=torch.bfloat16), requires_grad=False)
        self.norm = _Tensors(weight=(hidden,))
        head = self.markov_head = nn.Module()
        head.dflare_norm = _Tensors(weight=(hidden,))
        head.markov_w1 = nn.Embedding(target.c.vocab_rows, self.rank, dtype=torch.bfloat16)
        head.rnn_proj = nn.Linear(2 * self.rank + 2 * hidden, 3 * self.rank, dtype=torch.bfloat16)
        head.hidden_out = nn.Linear(self.rank, hidden, dtype=torch.bfloat16)
        self.register_buffer("head_latent", torch.empty(self.draft_vocab_size, self.rank + 1, dtype=torch.bfloat16))  # U H | U b (draft_head_tensors)
        # the target's tensors its stages read (the anchor's embedding, the head), tied to them (draft_head_tensors): the same tensors, which MMGP
        # counts once and places as the target's, are also the drafter's weights, which its graphs may read and whose changes they follow
        self.tied = _Tensors(**{name: getattr(target, name).weight.shape for name in TIED})
        self._keys = self._values = self._rnn = None
        self._length = 0

    # the context cache -------------------------------------------------------------------------------------------------------------
    def prepare_cache(self, max_length, device, dtype):
        dtype, block = self.mask_embed.dtype, self.block_size
        self._ring = min(max_length, self.window + REWIND_ROWS) + block
        sizes = [self._ring if index in self.local else max_length + block for index in range(len(self.layers))]
        if self._keys is None or self._keys[0].device != device or [k.shape[0] for k in self._keys] != sizes:
            shape = (self.kv_heads, self.dim)
            self._keys, self._values = ([torch.zeros(size, *shape, dtype=dtype, device=device) for size in sizes] for _ in range(2))
            self._rnn = torch.zeros(self._ring, 3 * self.rank, dtype=torch.float32, device=device)  # the head's input from the features of each position
            self._cursor, self._high = (torch.zeros(1, dtype=torch.long, device=device) for _ in range(2))  # the cache length; the end of what was ever written
            self._anchor = torch.zeros(1, dtype=torch.long, device=device)
            self._append_in = torch.zeros(block, len(self.target_layer_ids), self.hidden, dtype=dtype, device=device)
            self._base = torch.zeros(block - 1, 3 * self.rank, dtype=torch.float32, device=device)
            self._unary = torch.zeros(block - 1, self.draft_vocab_size, dtype=torch.float32, device=device)
        self.reset_sequence_state()

    def get_cache_length(self):
        return self._length

    def _set_length(self, length):
        self._length = length
        self._cursor.fill_(length)

    def truncate_cache(self, length):
        # positions beyond a window ring's reach read as absent until the ring refills: shorter drafts' context, verification is exact
        if length != self._length:
            self._set_length(length)

    def reset_sequence_state(self):
        self._length = 0
        if self._keys is not None:
            self._cursor.zero_()
            self._high.zero_()

    def release_sequence_state(self):
        self._keys = self._values = self._rnn = self._cursor = self._high = self._anchor = self._append_in = self._base = self._unary = None
        self._length = 0

    def _kept(self):
        """The first position the window rings still hold, on the host."""
        return max(0, int(self._high.item()) - self._ring)

    def snapshot_sequence_state(self, previous=None, reuse_tokens=0):
        # CPU copies are blocking; snapshots survive vision lending and teardown
        if self._keys is None:
            return None
        length, first = self._length, min(self._length, self._kept())
        rows = torch.arange(first, length, device=self._cursor.device) % self._ring
        take = lambda index, cache: transfer.move(cache[index][rows] if index in self.local else cache[index][:length], "cpu")
        return {"seq_length": length, "first": first, "keys": [take(index, self._keys) for index in range(len(self.layers))],
                "values": [take(index, self._values) for index in range(len(self.layers))], "rnn": self._rnn[rows].to("cpu")}

    def restore_sequence_state(self, snapshot):
        if snapshot is None:
            self.reset_sequence_state()
            return
        length, first = snapshot["seq_length"], snapshot["first"]
        rows = torch.arange(first, length, device=self._cursor.device) % self._ring
        for index in range(len(self.layers)):
            for cache, saved in ((self._keys[index], snapshot["keys"][index]), (self._values[index], snapshot["values"][index])):
                if index in self.local:
                    cache[rows] = saved.to(cache.device)
                else:
                    transfer.copy_(cache[:length], saved)
        self._rnn[rows] = snapshot["rnn"].to(self._rnn.device)
        self._set_length(length)
        self._high.fill_(length if first == 0 else first + self._ring)  # what lies before `first` is no longer there

    def _angles(self, positions):
        half = self.dim // 2
        inverse = self.theta ** (-torch.arange(half, device=positions.device, dtype=torch.float32) / half)
        angle = positions.float()[:, None, None] * inverse
        return angle.cos(), angle.sin()

    def _normed(self, x, heads, norm, cos, sin):
        dtype = x.dtype
        return _rotate(rms(x.view(-1, heads, self.dim), self.eps, norm.weight).to(dtype), cos, sin).to(dtype)

    def _append(self, features, first):
        """Keys, values and head input of the positions first.. from their target features [rows, 4, hidden], written by position."""
        dtype, count = self.mask_embed.dtype, features.shape[0]
        positions = first + torch.arange(count, device=features.device)
        mixed = torch.softmax(self.fuse_weight.float(), -1) @ features.float()  # [rows, layers + 1, hidden]
        cos, sin = self._angles(positions)
        for index, layer in enumerate(self.layers):
            attention = layer.self_attn
            context = rms(mixed[:, index], self.eps, layer.dflare_norm.weight).to(dtype)
            rows = positions % self._keys[index].shape[0]
            self._keys[index].index_copy_(0, rows, self._normed(attention.k_proj_ctx(context), self.kv_heads, attention.k_norm_ctx, cos, sin))
            self._values[index].index_copy_(0, rows, attention.v_proj_ctx(context).view(count, self.kv_heads, self.dim))
        head = self.markov_head
        self._rnn.index_copy_(0, positions % self._ring, F.linear(rms(mixed[:, -1], self.eps, head.dflare_norm.weight), head.rnn_proj.weight[:, -self.hidden:].float()))
        self._high.copy_(torch.maximum(self._high, positions[-1:] + 1))

    def append_stage(self, features):
        """A pass of the target appended at the cache length (cuda_graphs.run stage): all the rows of the buffer, of which the caller counts the committed ones."""
        self._append(features, self._cursor)

    def append_context(self, features, positions=None):
        """features [1, rows, 4 * hidden] of the committed positions that follow the cache: positions are the cache's own."""
        count = features.shape[1]
        if count == 0:
            return
        features = features.reshape(count, len(self.target_layer_ids), self.hidden)
        if count <= self.block_size:
            self._append_in[:count].copy_(features)
            with cuda_graphs.scope(gated=False):
                cuda_graphs.run(self, self._append_in, owner=self.target, method="append_stage", enabled=self._graphed())
        else:
            for start in range(0, count, self._ring):  # a ring takes each of its rows once per call, the later positions last
                self._append(features[start:start + self._ring], self._cursor + start)
        self._set_length(self._length + count)

    def _graphed(self):
        return bool(getattr(self.target, "_lm_engine_mmgp_graphs", False)) and self._keys[0].is_cuda

    # the block ---------------------------------------------------------------------------------------------------------------------
    def _context(self, index, cursor):
        """Keys and values a layer's block reads before its own, and which of them are committed positions."""
        keys, values = self._keys[index], self._values[index]
        if index not in self.local:
            return keys, values, torch.arange(keys.shape[0], device=keys.device) < cursor
        positions = cursor - self.window + torch.arange(self.window, device=keys.device)
        rows = positions.clamp_min(0) % self._ring
        return keys[rows], values[rows], positions >= (self._high - self._ring).clamp_min(0)

    def _scores(self, q, keys, values, valid):
        """[rows, heads * dim]: the block's queries over keys / values [count, kv_heads, dim] (views of the caches), valid [count]: the positions read."""
        rows, group, dtype = q.shape[0], self.heads // self.kv_heads, q.dtype
        scores = torch.bmm(q.view(rows, self.kv_heads, group, self.dim).permute(1, 0, 2, 3).reshape(self.kv_heads, rows * group, self.dim), keys.permute(1, 2, 0)) * self.dim ** -0.5
        scores = scores.masked_fill(~valid, torch.finfo(dtype).min)
        output = torch.bmm(torch.softmax(scores.float(), -1).to(dtype), values.transpose(0, 1))  # [kv_heads, rows * group, dim]
        return output.view(self.kv_heads, rows, group, self.dim).permute(1, 0, 2, 3).reshape(rows, -1)

    def _attend(self, index, q, k, v, places, kept):
        """The block's attention in a layer: its keys k and values v written at its positions places in the layer's cache (rows beyond the cache length,
        which the next append overwrites; the rings' slots of these positions, as an append takes them), then the positions from the first one the
        layer reads to the block's last read in place. kept: the first position the rings hold (at most the block's first)."""
        keys, values, rows = self._keys[index], self._values[index], self.block_size
        slots = places % keys.shape[0]
        keys.index_copy_(0, slots, k)
        values.index_copy_(0, slots, v)
        local = index in self.local
        low = kept if local else torch.zeros_like(kept)
        kernels = inkling2._ATTENTION
        if kernels is not None and q.is_cuda and kernels.supported(q.device):
            return kernels.block_attend(q, keys, values, self._cursor, low, self.dim ** -0.5, self.window + rows if local else 0)
        if local:  # the window's positions gathered from the ring (window + block rows)
            positions = places[-1] + 1 - (self.window + rows) + torch.arange(self.window + rows, device=q.device)
            slots = positions.clamp_min(0) % keys.shape[0]
            return self._scores(q, keys[slots], values[slots], positions >= low)
        # the cache's rows up to the block, in place: the live ones when the length is known here, all of them in a graph (fixed shapes)
        count = keys.shape[0] if keys.is_cuda and torch.cuda.is_current_stream_capturing() else self._length + rows
        return self._scores(q, keys[:count], values[:count], torch.arange(count, device=q.device) <= places[-1])

    def _attend_copies(self, index, q, k, v, cursor):
        """The previous path (WGP_INKLING2_DISABLE=draft_attention): the layer's context and the block's rows concatenated, every row of the global cache scored."""
        keys, values, valid = self._context(index, cursor)
        own = torch.ones(self.block_size, dtype=torch.bool, device=q.device)
        return self._scores(q, torch.cat((keys, k)), torch.cat((values, v)), torch.cat((valid, own)))

    def _block(self, anchor):
        """The states [block, hidden] of the block that starts at the cache length with the token anchor."""
        dtype, rows = self.mask_embed.dtype, self.block_size
        cursor = self._cursor
        h = torch.cat((self.target.embed_ids(anchor.reshape(1)), self.mask_embed.expand(rows - 1, -1))).to(dtype)
        places = cursor + torch.arange(rows, device=h.device)
        cos, sin = self._angles(places)
        if IN_PLACE:
            self._high.copy_(torch.maximum(self._high, places[-1:] + 1))  # the block's rows take the rings' slots of its positions
            kept = torch.minimum((self._high - self._ring).clamp_min(0), cursor)
        for index, layer in enumerate(self.layers):
            attention = layer.self_attn
            x = rms(h, self.eps, layer.input_layernorm.weight).to(dtype)
            q = self._normed(attention.q_proj(x), self.heads, attention.q_norm, cos, sin)
            k = self._normed(attention.k_proj(x), self.kv_heads, attention.k_norm, cos, sin)
            v = attention.v_proj(x).view(rows, self.kv_heads, self.dim)
            output = self._attend(index, q, k, v, places, kept) if IN_PLACE else self._attend_copies(index, q, k, v, cursor)
            h = h + attention.o_proj(output)
            q = k = v = output = None
            x = rms(h, self.eps, layer.post_attention_layernorm.weight).to(dtype)
            h = h + layer.mlp.down_proj((F.silu(layer.mlp.gate_proj(x).float()) * layer.mlp.up_proj(x).float()).to(dtype))
        return rms(h, self.eps, self.norm.weight).to(dtype)

    def draft_stage(self, anchor, base, unary):
        """The block after the token anchor (cuda_graphs.run stage). Into base [7, 3 * rank]: what the head's recurrence reads of each
        slot and of the last committed position; into unary [7, vocab]: the slots through the target's head, before the head's own term."""
        head, states = self.markov_head, self._block(anchor)[1:]
        last = self._rnn.index_select(0, (self._cursor - 1).clamp_min(0) % self._ring)
        last = last * (self._cursor > (self._high - self._ring).clamp_min(0)).to(last.dtype)  # no committed position, or one the ring lost
        base.copy_(F.linear(states.float(), head.rnn_proj.weight[:, 2 * self.rank:2 * self.rank + self.hidden].float(), head.rnn_proj.bias.float()) + last)
        unary.copy_(F.linear(states, self.tied.unembed[:self.draft_vocab_size]))

    def propose(self, anchor, start_position=None, embedding=None, output=None):
        """(base, unary) of the block after the token anchor: what proposal_step turns into the logits of each draft."""
        self._anchor.fill_(anchor)
        with cuda_graphs.scope(gated=False):
            cuda_graphs.run(self, self._anchor, self._base, self._unary, owner=self.target, method="draft_stage", enabled=self._graphed())
        return self._base, self._unary

    def proposal_step(self, index, base, unary, previous, state=None):
        """Draft index of the chain: (float32 logits, no confidence, the recurrence's state for the next draft). previous: the token before it."""
        head, rank = self.markov_head, self.rank
        previous = torch.as_tensor(previous, dtype=torch.long, device=base.device)
        weight = head.rnn_proj.weight[:, :2 * rank].float()
        raw = base + F.linear(head.markov_w1(previous).float(), weight[:, rank:])
        if state is not None:
            raw = raw + F.linear(state, weight[:, :rank])
        gate, candidate, latent = torch.sigmoid(raw[:rank]), torch.tanh(raw[rank:2 * rank]), torch.tanh(raw[2 * rank:])
        state = (1 - gate) * candidate if state is None else gate * state + (1 - gate) * candidate
        own = F.linear(torch.cat((latent, latent.new_ones(1))).to(self.head_latent.dtype), self.head_latent)
        return softcap(unary + own.float(), self.cap), None, state


def draft_state_dict(state_dict, prefix="mtp."):
    """The drafter's file under the names of the target's module tree."""
    return {prefix + name: tensor for name, tensor in state_dict.items()}


def draft_head_tensors(vocab, rows=16384):
    """A postprocess_sd of MMGP's loader: the drafter's head through the target's, [U H | U b] over the vocab token rows of U (see the
    module docstring), in the drafter's dtype."""
    def add(state_dict, quant_map=None, tied_map=None):
        unembed, head = state_dict["unembed.weight"], "mtp.markov_head.hidden_out."
        extended = torch.cat((state_dict[head + "weight"].float(), state_dict[head + "bias"].float()[:, None]), 1)
        state_dict["mtp.head_latent"] = torch.cat([(unembed[start:min(vocab, start + rows)].float() @ extended).to(unembed.dtype) for start in range(0, vocab, rows)])
        state_dict.update({f"mtp.tied.{name}": state_dict[f"{name}.weight"] for name in TIED})  # the same tensors: tied weights
        return state_dict, quant_map, tied_map
    return add
