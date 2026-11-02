# Adapted from fishaudio/fish-speech, revision 1ff3c3ed2f80d3106a83909dd25b83129c3d816f.
# Copyright Fish Audio contributors. See LICENSE_FISH_AUDIO for the upstream license.
import math
import torch
from torch import nn
from torch.nn import functional as F

from mmgp import cuda_graphs, runtime
from mmgp.expert_compute import SwiGLUExperts
from shared.llm_engines.lmgp.layers.moe import softmax_topk
from .layers import FullAttention, GatedDeltaNet, LayerContext, ZeroCenteredRMSNorm


class GenerationCancelled(Exception):
    pass


class DenseMLP(nn.Module):
    def __init__(self, hidden, intermediate):
        super().__init__()
        self.gate_proj = nn.Linear(hidden, intermediate, bias=False, dtype=torch.bfloat16)
        self.up_proj = nn.Linear(hidden, intermediate, bias=False, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(intermediate, hidden, bias=False, dtype=torch.bfloat16)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SparseMoeBlock(nn.Module):
    def __init__(self, config, features):
        super().__init__()
        hidden = config.hidden_size
        self.top_k, self.norm_topk_prob = config.num_experts_per_tok, config.norm_topk_prob
        self.gate = nn.Linear(hidden, config.num_experts, bias=False, dtype=torch.bfloat16)
        self.shared_expert = DenseMLP(hidden, config.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(hidden, 1, bias=False, dtype=torch.bfloat16)
        self.experts = SwiGLUExperts(config.num_experts, hidden, config.moe_intermediate_size, device="meta", use_triton=features.triton)

    def route(self, x):
        flat = x.reshape(-1, x.shape[-1])
        ids, weights = softmax_topk(self.gate(flat), self.top_k, self.norm_topk_prob)
        shared = self.shared_expert(flat) * self.shared_expert_gate(flat).sigmoid()
        return flat, ids, weights.to(flat.dtype), shared


class DecoderLayer(nn.Module):
    _cuda_graph_segments = True

    def __init__(self, config, index, features):
        super().__init__()
        self.layer_type = config.layer_types[index]
        if self.layer_type == "full_attention":
            self.self_attn = FullAttention(config, dtype=torch.bfloat16)
            self.self_attn.features = features
        else:
            self.linear_attn = GatedDeltaNet(config, dtype=torch.bfloat16)
            self.linear_attn.features = features
        self.mlp = SparseMoeBlock(config, features)
        self.input_layernorm = ZeroCenteredRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = ZeroCenteredRMSNorm(config.hidden_size, config.rms_norm_eps)

    def gpu_stage(self, x, context, cache):
        mixer = self.self_attn if self.layer_type == "full_attention" else self.linear_attn
        x = x + mixer(self.input_layernorm(x), context, cache)
        flat, ids, weights, shared = self.mlp.route(self.post_attention_layernorm(x))
        return x, flat, ids, weights, shared, SwiGLUExperts.pack(flat, ids, weights)

    def forward(self, x, context, cache):
        residual, flat, ids, weights, shared, packed = cuda_graphs.run(self, x, context, cache, method="gpu_stage")
        routed = self.mlp.experts(flat, ids, weights, packed)
        return residual + (routed + shared).reshape_as(residual)

    def decode(self, x, context, cache):
        """forward() inside the backbone's decoding graph: the CPU experts run as a host function queued on the stream."""
        residual, flat, ids, weights, shared, packed = self.gpu_stage(x, context, cache)
        routed = self.mlp.experts.queue_forward(flat, ids, weights, packed)
        return residual + (routed + shared).reshape_as(residual)


class TextModel(nn.Module):
    def __init__(self, config, features):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config, index, features) for index in range(config.num_hidden_layers)])
        self.norm = ZeroCenteredRMSNorm(config.hidden_size, config.rms_norm_eps)


class FishOmniBackbone(nn.Module):
    def __init__(self, config, features, head_rows, vq_codebooks):
        super().__init__()
        self.config, self.features = config, features
        self.model = TextModel(config, features)
        # Decoding samples only eos and the semantic band: the head keeps vocabulary rows head_rows = [start, stop) (the loader
        # slices the checkpoint's full head), and its logits are indexed from start.
        self.head_rows = head_rows
        self.lm_head = nn.Linear(config.hidden_size, head_rows[1] - head_rows[0], bias=False, dtype=torch.bfloat16)
        self.vq_codebooks = vq_codebooks  # codes 1..K of a semantic frame are added to its token embedding
        self._convertWeightsFloatTo = None
        self._model_dtype = torch.bfloat16
        self.cache = []
        self.kv = self.conv = self.state = None
        self.cache_capacity = 0
        self.abort_requested = lambda: False

    def setup_caches(self, capacity):
        """The caches of all layers in three stacked tensors (a few graph arguments); self.cache holds each layer's views."""
        device = runtime.execution_device(self)
        full = [layer.self_attn for layer in self.model.layers if layer.layer_type == "full_attention"]
        linear = [layer.linear_attn for layer in self.model.layers if layer.layer_type != "full_attention"]
        self.kv = torch.zeros(2, len(full), 1, capacity, full[0].num_kv_heads, full[0].head_dim, dtype=self._model_dtype, device=device)
        self.conv = torch.zeros(len(linear), 1, linear[0].conv_dim, linear[0].conv_kernel_size - 1, dtype=self._model_dtype, device=device)
        self.state = torch.zeros(len(linear), 1, linear[0].num_v_heads, linear[0].head_k_dim, linear[0].head_v_dim, dtype=torch.float32, device=device)
        self.cache = self.layer_caches(self.kv, self.conv, self.state)
        self.cache_capacity = capacity

    def layer_caches(self, kv, conv, state):
        caches, full, linear = [], 0, 0
        for layer in self.model.layers:
            if layer.layer_type == "full_attention":
                caches.append((kv[0, full], kv[1, full]))
                full += 1
            else:
                caches.append((conv[linear], state[linear]))
                linear += 1
        return caches

    def release_cache(self):
        self.cache.clear()
        self.kv = self.conv = self.state = None
        self.cache_capacity = 0

    def embed(self, input_ids):
        runtime.ensure_loaded(self)
        return self.model.embed_tokens(input_ids.to(runtime.execution_device(self)))

    def mix_embeddings(self, tokens, extra, semantic):
        """Backbone input of sampled frames: a semantic frame adds the summed embeddings of its codes 1..K (extra) to its token's."""
        text = self.model.embed_tokens(tokens)
        mixed = (text + extra) / math.sqrt(1 + self.vq_codebooks)
        return torch.where(semantic[:, None], mixed, text)

    def layer_context(self, input_pos, kv_len, continue_state):
        width = self.config.rotary_dim
        inverse = self.config.rope_theta ** (-torch.arange(0, width, 2, dtype=torch.float32, device=input_pos.device) / width)
        angles = input_pos.float()[:, None] * inverse[None, :]
        angles = torch.cat((angles, angles), dim=-1)
        return LayerContext(input_pos=input_pos, cos=angles.cos(), sin=angles.sin(), kv_len=kv_len, continue_state=continue_state, use_cache=True)

    def kv_length(self, start, length):
        end = start + length
        if end > self.cache_capacity:
            raise ValueError("Backbone cache capacity exceeded")
        # Fixed attended buckets allow graph reuse while the positions/masks remain tensor inputs.
        return min(self.cache_capacity, math.ceil(end / 256) * 256) if length == 1 else end

    def forward(self, inputs_embeds, input_pos, start):
        """Layer by layer. input_pos: the rows' device positions, the first one being the host int start."""
        hidden, context = inputs_embeds, self.layer_context(input_pos, self.kv_length(start, inputs_embeds.shape[1]), start > 0)
        for layer, cache in zip(self.model.layers, self.cache):
            if self.abort_requested():
                raise GenerationCancelled()
            hidden = layer(hidden, context, cache)
        if self.abort_requested():
            raise GenerationCancelled()
        return self.model.norm(hidden)

    def decode(self, tokens, extra, semantic, input_pos, start):
        """One decoding step from the sampled main tokens and the tower's embeddings of their codes: (normed hidden, logits)."""
        kv_len = self.kv_length(start, 1)
        if self.abort_requested():
            raise GenerationCancelled()
        if cuda_graphs.is_enabled(self) and runtime.weights_resident(self):
            # The whole step is one graph: each layer's CPU experts run as a host function of the stream.
            return cuda_graphs.run(self, tokens, extra, semantic, input_pos, kv_len, start > 0, self.kv, self.conv, self.state, method="decode_token")
        runtime.ensure_loaded(self)
        hidden = self.forward(self.mix_embeddings(tokens, extra, semantic)[None], input_pos, start)[0]
        return hidden, self.lm_head(hidden)

    def decode_token(self, tokens, extra, semantic, input_pos, kv_len, continue_state, kv, conv, state):
        hidden = self.mix_embeddings(tokens, extra, semantic)[None]
        context = self.layer_context(input_pos, kv_len, continue_state)
        for layer, cache in zip(self.model.layers, self.layer_caches(kv, conv, state)):
            hidden = layer.decode(hidden, context, cache)
        hidden = self.model.norm(hidden)[0]
        return hidden, self.lm_head(hidden)

    def raise_expert_errors(self):
        """Once the stream is synchronized: report a failed CPU expert job of the decoding graph."""
        for layer in self.model.layers:
            layer.mlp.experts.raise_job_errors()

    def compute_logits(self, hidden):
        """Logits of vocabulary rows head_rows[0]:head_rows[1] only."""
        runtime.ensure_loaded(self)
        return self.lm_head(hidden)
