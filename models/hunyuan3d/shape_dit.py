# Hunyuan3D 2.1 shape denoiser HunYuanDiTPlain, ported from Tencent Hunyuan3D 2.1 hy3dshape/models/denoisers/hunyuandit.py and
# moe_layers.py (Copyright (C) 2025 Tencent; Tencent Hunyuan 3D 2.1 Community License, see models/hunyuan3d/LICENSE and Notice.txt).
# Modified by DeepBeepMeep for WanGP: inference only; self-attention through shared.attention_kit and cross-attention through
# pay_attention on per-head q/k/v weights (rows reordered in the WanGP checkpoint, same products); the classifier-free guidance
# branches run one after the other inside each block; the zero-image unconditional cross-attention is its exact output, the output
# projection bias; unused options (decoupled cross-attention, attention pooling, positional embedding, timestep modulation) removed.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import math

import torch
import torch.nn.functional as F
from torch import nn

from shared import attention_kit
from shared.attention import pay_attention
from shared.kernels import ops
from shared.utils.phase_progress import check_abort


class Timesteps(nn.Module):
    def __init__(self, num_channels, max_period=10000):
        super().__init__()
        self.num_channels, self.max_period = num_channels, max_period

    def forward(self, timesteps):
        half_dim = self.num_channels // 2
        exponent = -math.log(self.max_period) * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=timesteps.device)
        emb = timesteps[:, None].float() * torch.exp(exponent / half_dim)[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(hidden_size, frequency_embedding_size), nn.GELU(), nn.Linear(frequency_embedding_size, hidden_size))
        self.time_embed = Timesteps(hidden_size)

    def forward(self, t):
        return self.mlp(self.time_embed(t).type(self.mlp[0].weight.dtype)).unsqueeze(dim=1)


def _rms_norm_(tensor, norm):
    """norm (RMSNorm over head_dim) of q or k (batch, tokens, heads, head_dim) in place, by chunks of tokens."""
    attention_kit.rows_(tensor, lambda part, first, last: F.rms_norm(part, norm.normalized_shape, norm.weight, norm.eps))


class Attention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.out_proj = nn.Linear(dim, dim)

    def _norm_(self, query, key, group):
        for tensor, norm in ((query, self.q_norm), (key, self.k_norm)):
            if tensor is not None:
                _rms_norm_(tensor, norm)

    def forward(self, x_list):
        return attention_kit.qkv_attention(x_list, self.to_q, self.to_k, self.to_v, self.num_heads, self.head_dim, self._norm_,
                                           fused_quant=lambda query, key, group, core, settings: ops.split_half_norm_rope_quant_(query, key, None, self.q_norm, self.k_norm, core, settings),
                                           out_proj=self.out_proj)


class CrossAttention(nn.Module):
    def __init__(self, qdim, kdim, num_heads):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, qdim // num_heads
        self.to_q = nn.Linear(qdim, qdim, bias=False)
        self.to_k = nn.Linear(kdim, qdim, bias=False)
        self.to_v = nn.Linear(kdim, qdim, bias=False)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=1e-6)
        self.out_proj = nn.Linear(qdim, qdim)

    def forward(self, x, context):
        query = self.to_q(x).unflatten(-1, (self.num_heads, self.head_dim))
        key = self.to_k(context).unflatten(-1, (self.num_heads, self.head_dim))
        _rms_norm_(query, self.q_norm)
        _rms_norm_(key, self.k_norm)
        qkv_list = [query, key, self.to_v(context).unflatten(-1, (self.num_heads, self.head_dim))]
        query = key = None
        return self.out_proj(pay_attention(qkv_list, recycle_q=True).flatten(2))


class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc1 = nn.Linear(width, width * 4)
        self.fc2 = nn.Linear(width * 4, width)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class FeedForward(nn.Module):
    """diffusers FeedForward(activation_fn="gelu"), keys net.0.proj and net.2."""

    def __init__(self, dim, inner_dim):
        super().__init__()
        self.net = nn.ModuleList([nn.Module(), nn.Identity(), nn.Linear(inner_dim, dim)])
        self.net[0].proj = nn.Linear(dim, inner_dim)

    def forward(self, x):
        return self.net[2](F.gelu(self.net[0].proj(x)))


class MoEBlock(nn.Module):
    def __init__(self, dim, num_experts, top_k):
        super().__init__()
        self.top_k = top_k
        self.experts = nn.ModuleList([FeedForward(dim, dim * 4) for _ in range(num_experts)])
        self.gate = nn.Module()
        self.gate.weight = nn.Parameter(torch.empty(num_experts, dim))
        self.shared_experts = FeedForward(dim, dim * 4)

    def forward(self, x):
        """Upstream moe_infer: tokens sorted by expert, each expert run on its tokens, the weighted outputs summed per token in expert
        order (one host read of the expert sizes per call)."""
        flat = x.reshape(-1, x.shape[-1])
        scores = F.linear(flat, self.gate.weight).softmax(dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        del scores
        flat_idx, flat_weight = topk_idx.view(-1), topk_weight.view(-1, 1)
        order = flat_idx.argsort()
        ends = flat_idx.bincount().cpu().numpy().cumsum(0).tolist()
        token_idx = order // self.top_k
        out = torch.zeros_like(flat)
        start = 0
        for expert, end in zip(self.experts, ends):
            if end > start:
                tokens = token_idx[start:end]
                y = expert(flat[tokens])
                out.index_add_(0, tokens, y.mul_(flat_weight[order[start:end]]))
                del y
            start = end
        return out.view(x.shape).add_(self.shared_experts(x))


class HunYuanDiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, context_dim, skip_connection, use_moe, num_experts, moe_top_k):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn1 = Attention(hidden_size, num_heads)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1e-6)
        self.attn2 = CrossAttention(hidden_size, context_dim, num_heads)
        self.norm3 = nn.LayerNorm(hidden_size, eps=1e-6)
        if skip_connection:
            self.skip_norm = nn.LayerNorm(hidden_size, eps=1e-6)
            self.skip_linear = nn.Linear(2 * hidden_size, hidden_size)
        else:
            self.skip_linear = None
        if use_moe:
            self.moe = MoEBlock(hidden_size, num_experts, moe_top_k)
        else:
            self.mlp = MLP(hidden_size)
        self.use_moe = use_moe

    def forward(self, x, context, skip_value=None):
        """context None: the zero-image branch, whose cross-attention output is exactly attn2.out_proj.bias (all keys and values are
        zero)."""
        if self.skip_linear is not None:
            x = self.skip_norm(self.skip_linear(torch.cat([skip_value, x], dim=-1)))
        x = x + self.attn1([self.norm1(x)])
        if context is None:
            x.add_(self.attn2.out_proj.bias)
        else:
            x.add_(self.attn2(self.norm2(x), context))
        ffn = self.moe if self.use_moe else self.mlp
        return x.add_(ffn(self.norm3(x)))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_channels)

    def forward(self, x):
        return self.linear(self.norm_final(x)[:, 1:])


class HunYuanDiTPlain(nn.Module):
    def __init__(self, in_channels=64, hidden_size=2048, context_dim=1024, depth=21, num_heads=16, num_moe_layers=6, num_experts=8, moe_top_k=2, **_):
        super().__init__()
        self.depth = depth
        self.x_embedder = nn.Linear(in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size, hidden_size * 4)
        self.blocks = nn.ModuleList([HunYuanDiTBlock(hidden_size, num_heads, context_dim, layer > depth // 2, depth - layer <= num_moe_layers, num_experts, moe_top_k)
                                     for layer in range(depth)])
        self.final_layer = FinalLayer(hidden_size, in_channels)

    def forward(self, x, t, contexts):
        """x (batch, latents, channels), t (batch,) in [0, 1]; contexts: the image conditioning of each guidance branch, (batch, tokens,
        context_dim), or None for the zero-image branch. The branches share the input and run one after the other in each block.
        Returns one prediction per branch."""
        x = torch.cat([self.t_embedder(t), self.x_embedder(x)], dim=1)
        states = [x] * len(contexts)
        skips = [[] for _ in contexts]
        del x
        for layer, block in enumerate(self.blocks):
            for branch, context in enumerate(contexts):
                skip_value = skips[branch].pop() if layer > self.depth // 2 else None
                states[branch] = block(states[branch], context, skip_value)
                del skip_value
                if layer < self.depth // 2:
                    skips[branch].append(states[branch])
            check_abort()
        return [self.final_layer(state) for state in states]
