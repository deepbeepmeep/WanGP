# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""ZONOS3-Mini language model (Zyphra, Apache-2.0 weights; architecture of the MIT reference implementation of the ZONOS3-Mini
repository). Each row of the sequence holds the 10 codec codes of a 15 Hz frame (codebook j delayed by j rows) and one text token;
one forward predicts the 10 codes of the next row. 11 layers: gated DeltaNet, a global attention layer every 4th, a dense SwiGLU then
latent MoE feed-forwards, and Full AttnRes residuals: each sublayer reads a learned softmax mix of the embedding and every earlier
sublayer output (no residual add)."""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from mmgp import cuda_graphs, runtime
from shared import attention_kit
from shared.kernels import ops
from shared.llm_engines.lmgp.layerwise import cached_attention, delta_rule
from shared.llm_engines.lmgp.models.qwen3_5 import torch_chunk_gated_delta_rule, torch_recurrent_gated_delta_rule

BF16 = torch.bfloat16


class GenerationCancelled(Exception):
    pass


@dataclass
class LayerContext:
    input_pos: torch.Tensor  # [T] absolute positions of the rows
    cos: torch.Tensor  # [T, head_dim / 2] RoPE tables (float32)
    sin: torch.Tensor
    kv_len: int  # attended cache prefix
    continue_state: bool  # the rows continue a cached sequence


def rope_interleaved(x, cos, sin):
    """Rotate adjacent channel pairs of x (batch, tokens, heads, head_dim) in float32."""
    pairs = x.float().unflatten(-1, (-1, 2))
    a, b = pairs[..., 0], pairs[..., 1]
    cos, sin = cos[None, :, None], sin[None, :, None]
    return torch.stack((a * cos - b * sin, a * sin + b * cos), -1).flatten(-2).to(x.dtype)


def _torch_delta_rule(recurrent):
    """The shared PyTorch gated delta rule on float32 q / k / v (the reference normalizes and accumulates them in float32)."""
    def rule(query, key, value, decay, beta, initial_state):
        query, key, value = query.float(), key.float(), value.float()
        if recurrent:
            return torch_recurrent_gated_delta_rule(query, key, value, decay, beta, initial_state, True)
        return torch_chunk_gated_delta_rule(query, key, value, decay, beta, initial_state=initial_state, output_final_state=True)
    return rule


class GatedDeltaNet(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.heads, self.head_dim, self.kernel = c["gdn_heads"], c["gdn_head_dim"], c["gdn_kernel"]
        inner = self.heads * self.head_dim
        self.inner, self.channels = inner, 3 * inner
        self.in_proj = nn.Linear(c["dim"], 4 * inner + 2 * self.heads, bias=False, dtype=BF16)  # q k v | z | a | b
        self.conv = nn.Conv1d(self.channels, self.channels, self.kernel, groups=self.channels, bias=False, dtype=BF16)
        self.out_weight = nn.Parameter(torch.empty(self.head_dim, dtype=BF16))
        self.A_log = nn.Parameter(torch.empty(self.heads, dtype=BF16))
        self.dt_bias = nn.Parameter(torch.empty(self.heads, dtype=BF16))
        self.wo = nn.Linear(inner, c["dim"], bias=False, dtype=BF16)
        self.features = None

    def forward(self, x, context, cache):
        conv_state, state = cache
        batch, length, _ = x.shape
        qkv, z, a, b = self.in_proj(x).split((self.channels, self.inner, self.heads, self.heads), -1)
        previous = conv_state if context.continue_state else qkv.new_zeros(batch, self.channels, self.kernel - 1)
        padded = torch.cat((previous, qkv.transpose(1, 2)), -1)
        conv_state.copy_(padded[..., -(self.kernel - 1):])
        mixed = F.silu(F.conv1d(padded, self.conv.weight, groups=self.channels)).transpose(1, 2)
        query, key, value = (part.reshape(batch, length, self.heads, self.head_dim) for part in mixed.split(self.inner, -1))
        decay = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())
        beta = b.float().sigmoid()
        recurrent = context.continue_state and length == 1
        core, final = delta_rule(query, key, value, decay, beta, state if context.continue_state else None, features=self.features,
                                 recurrent=recurrent, torch_impl=_torch_delta_rule(recurrent))
        state.copy_(final)
        normalized = F.rms_norm(core.to(x.dtype), (self.head_dim,), weight=self.out_weight, eps=1e-5)
        gated = (normalized.float() * F.silu(z.reshape(batch, length, self.heads, self.head_dim).float())).to(x.dtype)
        return self.wo(gated.reshape(batch, length, self.inner))


class GlobalAttention(nn.Module):
    """Grouped-query attention with weightless q/k RMSNorm, a learned temperature per query head, interleaved RoPE and a sigmoid output
    gate per head. The released k|v projection (wkv) is split into wk and wv at load."""

    def __init__(self, c):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = c["n_heads"], c["n_kv_heads"], c["head_dim"]
        self.wq = nn.Linear(c["dim"], self.heads * self.head_dim, bias=False, dtype=BF16)
        self.wk = nn.Linear(c["dim"], self.kv_heads * self.head_dim, bias=False, dtype=BF16)
        self.wv = nn.Linear(c["dim"], self.kv_heads * self.head_dim, bias=False, dtype=BF16)
        self.wo = nn.Linear(self.heads * self.head_dim, c["dim"], bias=False, dtype=BF16)
        self.temp = nn.Parameter(torch.empty(1, self.heads, 1, dtype=BF16))
        self.gater = nn.Linear(c["dim"], self.heads, bias=False, dtype=BF16)
        self.features = None

    def forward(self, x, context, cache):
        batch, length, _ = x.shape
        gate = self.gater(x).sigmoid()
        dim = self.head_dim

        def normalize(query, key, group):
            if query is not None:
                temp = self.temp.abs().view(-1, 1)[group.q.start:group.q.stop]  # a slice: an index tensor would be a host copy (no graph capture)
                attention_kit.rows_(query, lambda part, first, last: rope_interleaved(F.rms_norm(part, (dim,), eps=1e-6) * temp, context.cos[first:last], context.sin[first:last]))
            if key is not None:
                attention_kit.rows_(key, lambda part, first, last: rope_interleaved(F.rms_norm(part, (dim,), eps=1e-6), context.cos[first:last], context.sin[first:last]))

        attended = attention_kit.qkv_attention([x], self.wq, self.wk, self.wv, self.heads, dim, normalize, kv_heads=self.kv_heads,
            attention_fn=lambda qkv, group: cached_attention(qkv, group, cache, context.input_pos, context.kv_len, self.features))
        return self.wo((attended * gate[..., None]).reshape(batch, length, -1))

    def text_attention(self, x, context, keys, weights):
        """The alignment probe of one decoded row x [1, 1, dim]: the attention distributions of its query heads over the text keys
        [T, kv_heads, head_dim] (computed again in float32, softmax over the text alone), summed with the head weights [heads]."""
        query = self.wq(x).view(1, 1, self.heads, self.head_dim)
        query = rope_interleaved(F.rms_norm(query, (self.head_dim,), eps=1e-6) * self.temp.abs().view(-1, 1), context.cos, context.sin)[0, 0]
        logits = torch.einsum("hd,thd->ht", query.float(), keys.repeat_interleave(self.heads // self.kv_heads, 1).float()) * self.head_dim ** -0.5
        return (logits.softmax(-1) * weights[:, None]).sum(0)


class DenseFFN(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.w_in = nn.Linear(c["dim"], 2 * c["ffn_hidden"], bias=False, dtype=BF16)  # gate | value rows (reordered in the WanGP checkpoint)
        self.w_out = nn.Linear(c["ffn_hidden"], c["dim"], bias=False, dtype=BF16)

    def forward(self, x_list):
        return ops.mlp(x_list, self.w_in, self.w_out, "swiglu")


class Experts(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.empty(c["n_experts"], 2 * c["moe_hidden"], c["moe_routed_dim"], dtype=BF16))  # gate | up rows
        self.down_proj = nn.Parameter(torch.empty(c["n_experts"], c["moe_routed_dim"], c["moe_hidden"], dtype=BF16))


class Router(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.router_linear = nn.Linear(c["dim"], c["n_experts"], bias=False, dtype=BF16)
        self.rmsnorm_eda = nn.Module()
        self.rmsnorm_eda.weight = nn.Parameter(torch.empty(c["dim"], dtype=BF16))
        self.balancing_biases = nn.Parameter(torch.empty(c["n_experts"], dtype=torch.float32))
        self._lock_dtype = torch.float32  # the balancing biases stay float32 (released), whatever MMGP's model dtype


class LatentMoE(nn.Module):
    """Top-k of n SwiGLU experts in a narrower latent space (latent_down / latent_up around them). Experts are chosen on the router
    logits minus the balancing biases, weighted by the softmax of their raw logits."""
    GATHER_PAIRS = 32  # up to this many (row, expert) pairs, the chosen experts are gathered (graph-safe); above, rows are grouped by expert

    def __init__(self, c, eps):
        super().__init__()
        self.topk, self.hidden, self.eps = c["moe_topk"], c["moe_hidden"], eps
        self.experts = Experts(c)
        self.latent_down = nn.Linear(c["dim"], c["moe_routed_dim"], bias=False, dtype=BF16)
        self.latent_up = nn.Linear(c["moe_routed_dim"], c["dim"], bias=False, dtype=BF16)
        self.router = Router(c)

    def forward(self, x_list):
        x = x_list.pop()
        flat = x.reshape(-1, x.shape[-1])
        router = self.router
        logits = router.router_linear(F.rms_norm(flat, (flat.shape[-1],), weight=router.rmsnorm_eda.weight, eps=self.eps)).float()
        choices = torch.topk(logits - router.balancing_biases, self.topk, -1).indices
        weights = logits.gather(-1, choices).softmax(-1)
        latent = self.latent_down(flat)
        del x, flat, logits
        ids = choices.reshape(-1)
        rows = latent.repeat_interleave(self.topk, 0) if self.topk > 1 else latent
        if len(ids) <= self.GATHER_PAIRS:
            gate_up = torch.bmm(self.experts.gate_up_proj[ids], rows[:, :, None])[..., 0]
            routed = torch.bmm(self.experts.down_proj[ids], (F.silu(gate_up[:, :self.hidden]) * gate_up[:, self.hidden:])[:, :, None])[..., 0]
        else:
            routed = self._grouped(ids, rows)
        routed = (routed.float() * weights.reshape(-1, 1)).to(latent.dtype).view(-1, self.topk, latent.shape[-1]).sum(1)
        return self.latent_up(routed)

    def _grouped(self, ids, rows):
        """Rows grouped by expert, each expert's projection run once over its rows (prompt processing)."""
        order = ids.argsort(stable=True)
        counts = torch.bincount(ids, minlength=self.experts.gate_up_proj.shape[0]).tolist()
        ordered, out, start = rows[order], torch.empty_like(rows), 0
        for expert, count in enumerate(counts):
            if count:
                part = ordered[start:start + count]
                gate_up = F.linear(part, self.experts.gate_up_proj[expert])
                ordered[start:start + count] = F.linear(F.silu(gate_up[:, :self.hidden]) * gate_up[:, self.hidden:], self.experts.down_proj[expert])
                start += count
        out[order] = ordered
        return out


class Layer(nn.Module):
    def __init__(self, c, index):
        super().__init__()
        self.attention_norm = nn.Module()
        self.attention_norm.weight = nn.Parameter(torch.empty(c["dim"], dtype=BF16))
        self.ffn_norm = nn.Module()
        self.ffn_norm.weight = nn.Parameter(torch.empty(c["dim"], dtype=BF16))
        self.is_global = (index + 1) % c["global_period"] == 0
        self.attention = GlobalAttention(c) if self.is_global else GatedDeltaNet(c)
        self.feed_forward = DenseFFN(c) if index == 0 else LatentMoE(c, c["norm_eps"])


class MultiEmbedder(nn.Module):
    def __init__(self, c):
        super().__init__()
        audio_vocab = c["codebook_size"] + 2
        self.embedders = nn.ModuleList([nn.Embedding(audio_vocab, c["dim"], dtype=BF16) for _ in range(c["n_codebooks"])] + [nn.Embedding(c["text_rows"], c["dim"], dtype=BF16)])


class Zonos3LM(nn.Module):
    def __init__(self, config, features):
        super().__init__()
        c = self.config = config
        self.features = features
        self.n_codebooks, self.audio_vocab, self.eps = c["n_codebooks"], c["codebook_size"] + 2, c["norm_eps"]
        self.multi_embedder = MultiEmbedder(c)
        self.layers = nn.ModuleList([Layer(c, index) for index in range(c["n_layers"])])
        self.out_norm = nn.Module()
        self.out_norm.weight = nn.Parameter(torch.empty(c["dim"], dtype=BF16))
        self.multi_output = nn.Linear(c["dim"], c["n_codebooks"] * self.audio_vocab, bias=False, dtype=BF16)
        self.block_attnres_queries = nn.Parameter(torch.empty(2 * c["n_layers"] + 1, c["dim"], dtype=BF16))
        self.speaker_projection = nn.Linear(c["speaker_embedding_dim"], c["dim"], bias=True, dtype=BF16)
        self.condition_embeddings = nn.ModuleList([nn.Embedding(count + 1, c["dim"], dtype=BF16) for count in c["condition_bucket_counts"]])
        for layer in self.layers:
            layer.attention.features = features
        self.steered_sources = ()  # the Full AttnRes bank sources that steering deltas add to (set by the pipeline)
        # The alignment probe of a decoding pass, set by the pipeline for takes with inline cues: (first text row, text rows,
        # {layer index: head weights}). Each decoded row then writes the text row it reads most into `reading`.
        self.probe = None
        self._convertWeightsFloatTo = None  # the routers' balancing biases stay float32, as released
        self._model_dtype = BF16
        self.kv = self.conv = self.state = self.reading = None
        self.views = []  # each layer's view of the caches (not "cache": wgp resets transformer.cache before a generation)
        self.capacity = 0
        self.abort_requested = lambda: False

    # -- caches -------------------------------------------------------------------------------------------------------------------------

    def setup_caches(self, capacity, batch=1):
        """The caches of every layer in three stacked tensors: the global layers' keys / values, the DeltaNet layers' convolution history
        and float32 recurrent states."""
        device, dtype = runtime.execution_device(self), self.multi_output.weight.dtype
        global_layers = [layer.attention for layer in self.layers if layer.is_global]
        delta_layers = [layer.attention for layer in self.layers if not layer.is_global]
        first = global_layers[0]
        self.kv = torch.zeros(2, len(global_layers), batch, capacity, first.kv_heads, first.head_dim, dtype=dtype, device=device)
        gdn = delta_layers[0]
        self.conv = torch.zeros(len(delta_layers), batch, gdn.channels, gdn.kernel - 1, dtype=dtype, device=device)
        self.state = torch.zeros(len(delta_layers), batch, gdn.heads, gdn.head_dim, gdn.head_dim, dtype=torch.float32, device=device)
        self.reading = torch.zeros(1, dtype=torch.long, device=device)
        self.views = self.layer_caches(self.kv, self.conv, self.state)
        self.capacity = capacity
        cuda_graphs.register_state(self, "zonos3_caches", (self.kv, self.conv, self.state, self.reading))

    def layer_caches(self, kv, conv, state):
        caches, globals_, deltas = [], 0, 0
        for layer in self.layers:
            if layer.is_global:
                caches.append((kv[0, globals_], kv[1, globals_]))
                globals_ += 1
            else:
                caches.append((conv[deltas], state[deltas]))
                deltas += 1
        return caches

    def set_probe(self, start, length, heads):
        """Starts the alignment probe of the decoded rows over the text rows [start, start + length): heads, [layer, head] pairs."""
        device, weights = runtime.execution_device(self), {}
        for layer, head in heads:
            weights.setdefault(layer, torch.zeros(self.config["n_heads"], device=device))[head] = 1.0
        self.probe = (start, length, weights)
        cuda_graphs.register_state(self, "zonos3_probe", tuple(weights.values()))

    def release_cache(self):
        if self.probe is not None:
            cuda_graphs.release_state(self, "zonos3_probe")
            self.probe = None
        self.views.clear()
        self.kv = self.conv = self.state = self.reading = None
        self.capacity = 0

    # -- forward ------------------------------------------------------------------------------------------------------------------------

    def layer_context(self, input_pos, kv_len, continue_state):
        dim = self.config["head_dim"]
        inverse = self.config["rope_theta"] ** (-torch.arange(0, dim, 2, dtype=torch.float32, device=input_pos.device) / dim)
        angles = input_pos.float()[:, None] * inverse[None, :]
        return LayerContext(input_pos, angles.cos(), angles.sin(), kv_len, continue_state)

    def kv_length(self, start, length):
        end = start + length
        if end > self.capacity:
            raise ValueError("ZONOS3 cache capacity exceeded")
        # One attended length per 256 positions while decoding: the decoding graph is reused, positions and masks remain inputs.
        return min(self.capacity, -(-end // 256) * 256) if length == 1 else end

    def embed(self, ids, speaker=None, speaker_row=-1, condition_rows=()):
        """Rows [T, n_codebooks + 1] -> the summed embeddings [T, dim] before the weightless norm, the speaker row replaced by the
        projected speaker embedding and each condition row (row, table, bucket) by its bucket's embedding."""
        embedders = self.multi_embedder.embedders
        x = embedders[0](ids[:, 0])
        for index in range(1, len(embedders)):
            x = x + embedders[index](ids[:, index])
        if speaker is not None and 0 <= speaker_row < len(x):
            x[speaker_row] = self.speaker_projection(speaker.to(device=x.device, dtype=x.dtype))
        for row, table, bucket in condition_rows:
            if 0 <= row < len(x):
                x[row] = self.condition_embeddings[table].weight[bucket]
        return F.rms_norm(x, (x.shape[-1],), eps=self.eps)

    @staticmethod
    def _route(bank, keys, count, query):
        scores = (keys[:count] * query).sum(-1, dtype=torch.float32)
        return (bank[:count] * scores.softmax(0).to(bank.dtype).unsqueeze(-1)).sum(0)

    def _add_source(self, bank, keys, index, value, deltas=None):
        if deltas is not None and index in self.steered_sources:
            value = value + deltas[self.steered_sources.index(index)].to(value.dtype)  # a steering delta on this bank source
        bank[index] = value
        keys[index] = F.rms_norm(value, (value.shape[-1],), eps=self.eps)

    def layers_forward(self, x, context, caches, check_abort=True, deltas=None, probe=None):
        """Normalized embeddings [T, dim] -> the Full AttnRes bank [2L+1, T, dim] and its normalized keys. deltas: steering deltas
        [len(steered_sources), dim] added to those bank sources. probe (one decoded row): see `probe`; the text row read is written
        into `reading`."""
        queries = self.block_attnres_queries
        sources = 2 * len(self.layers) + 1
        bank, keys = x.new_empty(sources, *x.shape), x.new_empty(sources, *x.shape)
        self._add_source(bank, keys, 0, x, deltas)
        del x
        read = None
        for index, (layer, cache) in enumerate(zip(self.layers, caches)):
            if check_abort and self.abort_requested():
                raise GenerationCancelled()
            mixed = self._route(bank, keys, 2 * index + 1, queries[2 * index])
            attention_in = F.rms_norm(mixed, (mixed.shape[-1],), weight=layer.attention_norm.weight, eps=self.eps)[None]
            del mixed
            if probe is not None and index in probe[2]:
                start, length, weights = probe
                part = layer.attention.text_attention(attention_in, context, cache[0][0, start:start + length], weights[index])
                read = part if read is None else read + part
            self._add_source(bank, keys, 2 * index + 1, layer.attention(attention_in, context, cache)[0], deltas)
            del attention_in
            mixed = self._route(bank, keys, 2 * index + 2, queries[2 * index + 1])
            ffn_in = [F.rms_norm(mixed, (mixed.shape[-1],), weight=layer.ffn_norm.weight, eps=self.eps)[None]]
            del mixed
            self._add_source(bank, keys, 2 * index + 2, layer.feed_forward(ffn_in).reshape(bank.shape[1:]), deltas)
        if read is not None:
            self.reading.copy_(read.argmax()[None])
        return bank, keys

    def head(self, bank, keys):
        """Logits [rows, n_codebooks, audio_vocab] of the bank's rows, softcapped."""
        mixed = self._route(bank, keys, bank.shape[0], self.block_attnres_queries[-1])
        logits = self.multi_output(F.rms_norm(mixed, (mixed.shape[-1],), weight=self.out_norm.weight, eps=self.eps))
        softcap = self.config["loss_softcap"]
        return (softcap * torch.tanh(logits / softcap)).view(-1, self.n_codebooks, self.audio_vocab)

    def prefill(self, ids, positions, start, speaker=None, speaker_row=-1, condition_rows=()):
        """Prompt rows [T, n_codebooks + 1] at positions start..start+T-1 (start > 0 continues the caches): the logits of the last row.
        speaker_row and the condition rows are absolute prompt rows."""
        runtime.ensure_loaded(self)
        device = runtime.execution_device(self)
        ids = ids.to(device)
        x = self.embed(ids, speaker, speaker_row - start, [(row - start, table, bucket) for row, table, bucket in condition_rows])
        context = self.layer_context(positions.to(device), self.kv_length(start, len(ids)), start > 0)
        bank, keys = self.layers_forward(x, context, self.views)
        return self.head(bank[:, -1:], keys[:, -1:])[0]

    def decode(self, row, position, start, deltas=None):
        """One generated row [1, n_codebooks + 1] at position tensor [1] (start: its host value): the logits of the next row. deltas:
        the steering deltas of this frame (see layers_forward)."""
        kv_len = self.kv_length(start, 1)
        if self.abort_requested():
            raise GenerationCancelled()
        if cuda_graphs.is_enabled(self) and runtime.weights_resident(self):
            return cuda_graphs.run(self, row, position, kv_len, self.kv, self.conv, self.state, deltas, method="decode_token")
        runtime.ensure_loaded(self)
        return self.decode_token(row, position, kv_len, self.kv, self.conv, self.state, deltas)

    def decode_token(self, row, position, kv_len, kv, conv, state, deltas=None):
        x = self.embed(row)
        bank, keys = self.layers_forward(x, self.layer_context(position, kv_len, True), self.layer_caches(kv, conv, state), check_abort=False, deltas=deltas,
                                         probe=self.probe)
        return self.head(bank, keys)[0]
