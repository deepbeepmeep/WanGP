# TripoSG and PartCrafter shape denoiser, ported from VAST-AI-Research/TripoSG triposg/models/transformers/triposg_transformer.py and
# triposg/models/attention_processor.py (Copyright (c) 2025 VAST-AI-Research and contributors, MIT) and wgsxm/PartCrafter
# src/models/transformers/partcrafter_transformer.py and src/models/attention_processor.py (Copyright (c) 2025 Yuchen Lin, MIT).
# This code is based on Tencent HunyuanDiT, licensed under the Tencent Hunyuan Community License Agreement (see
# LICENSE_TENCENT_HUNYUAN_DIT.txt in this folder): Tencent Hunyuan is licensed under the Tencent Hunyuan Community License Agreement,
# Copyright (C) 2024 Tencent. All Rights Reserved. The trademark rights of "Tencent Hunyuan" are owned by Tencent or its affiliate.
# Modified by DeepBeepMeep for WanGP: inference only, without diffusers; self-attention through shared.attention_kit and cross-attention
# through pay_attention on per-head q/k/v weights (rows reordered in the WanGP checkpoint, same products); the classifier-free
# guidance branches run one after the other inside each block; the zero-image unconditional cross-attention is its exact output, the
# output projection bias; PartCrafter's global attention blocks attend the tokens of all parts joined (its cross-attention, whose
# image keys are the same for every part, is computed once for all parts, as its global blocks do).
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import math

import torch
import torch.nn.functional as F
from torch import nn

from shared import attention_kit
from shared.attention import pay_attention
from shared.utils.phase_progress import check_abort


class FP32LayerNorm(nn.LayerNorm):
    """diffusers FP32LayerNorm: normalized in FP32, returned in the input dtype."""

    def forward(self, x):
        return F.layer_norm(x.float(), self.normalized_shape, self.weight.float(), self.bias.float(), self.eps).to(x.dtype)


def _rms_norm_(tensor, norm):
    """diffusers RMSNorm over head_dim of q or k (batch, tokens, heads, head_dim), in place by chunks of tokens: FP32 statistics, the
    normalized values rounded to the tensor dtype before the weight."""
    def normalize(part, first, last):
        values = part.float()
        return (values * torch.rsqrt(values.pow(2).mean(-1, keepdim=True) + norm.eps)).to(part.dtype) * norm.weight

    attention_kit.rows_(tensor, normalize)


def timestep_embedding(timesteps, channels, max_period=10000):
    """diffusers Timesteps(channels, flip_sin_to_cos=False, downscale_freq_shift=0): [sin, cos] in FP32."""
    half = channels // 2
    exponent = -math.log(max_period) * torch.arange(0, half, dtype=torch.float32, device=timesteps.device) / half
    emb = timesteps[:, None].float() * torch.exp(exponent)[None, :]
    return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class Attention(nn.Module):
    """diffusers Attention as TripoSG configures it (no q/k/v bias, optional RMS q/k norm, optional LayerNorm of the context)."""

    def __init__(self, dim, heads, context_dim=None, qk_norm=True, norm_cross=False):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(context_dim or dim, dim, bias=False)
        self.to_v = nn.Linear(context_dim or dim, dim, bias=False)
        if qk_norm:
            self.norm_q = nn.RMSNorm(self.head_dim, eps=1e-6)
            self.norm_k = nn.RMSNorm(self.head_dim, eps=1e-6)
        if norm_cross:
            self.norm_cross = nn.LayerNorm(context_dim)
        self.to_out = nn.ModuleList([nn.Linear(dim, dim)])
        self.qk_norm = qk_norm

    def _norm_(self, query, key, group):
        if self.qk_norm:
            for tensor, norm in ((query, self.norm_q), (key, self.norm_k)):
                if tensor is not None:
                    _rms_norm_(tensor, norm)

    def forward(self, x_list):
        """Self-attention of x (handed off in x_list), (batch, tokens, dim)."""
        out = attention_kit.qkv_attention(x_list, self.to_q, self.to_k, self.to_v, self.heads, self.head_dim, self._norm_)
        return self.to_out[0](out.flatten(2))

    def keys_values(self, context):
        """Normalized keys and values (1, tokens, heads, head_dim) of a context (1, tokens, context_dim)."""
        if hasattr(self, "norm_cross"):
            context = self.norm_cross(context)
        key = self.to_k(context).unflatten(-1, (self.heads, self.head_dim))
        if self.qk_norm:
            _rms_norm_(key, self.norm_k)
        return key, self.to_v(context).unflatten(-1, (self.heads, self.head_dim))

    def cross(self, x, keys_values):
        """Cross-attention of x (batch, tokens, dim) to the keys and values shared by every batch item: the queries of all items are
        attended together (each query row is independent of the others)."""
        query = self.to_q(x).reshape(1, -1, self.heads, self.head_dim)
        if self.qk_norm:
            _rms_norm_(query, self.norm_q)
        qkv_list = [query, *keys_values]
        query = None
        return self.to_out[0](pay_attention(qkv_list, recycle_q=True).reshape(x.shape))


class FeedForward(nn.Module):
    """diffusers FeedForward(activation_fn="gelu"), keys net.0.proj and net.2."""

    def __init__(self, dim, inner_dim):
        super().__init__()
        self.net = nn.ModuleList([nn.Module(), nn.Identity(), nn.Linear(inner_dim, dim)])
        self.net[0].proj = nn.Linear(dim, inner_dim)

    def forward(self, x):
        return self.net[2](F.gelu(self.net[0].proj(x)))


class DiTBlock(nn.Module):
    """TripoSG DiTBlock (skip_concat_front, skip_norm_last: the long skip is [skip, x] -> linear -> norm)."""

    def __init__(self, dim, heads, context_dim=None, self_attention=True, cross_attention=True, skip=False, qk_norm=True, norm_cross=False):
        super().__init__()
        if self_attention:
            self.norm1 = FP32LayerNorm(dim, eps=1e-5)
            self.attn1 = Attention(dim, heads, qk_norm=qk_norm)
        if cross_attention:
            self.norm2 = FP32LayerNorm(dim, eps=1e-5)
            self.attn2 = Attention(dim, heads, context_dim, qk_norm=qk_norm, norm_cross=norm_cross)
        self.norm3 = FP32LayerNorm(dim, eps=1e-5)
        self.ff = FeedForward(dim, dim * 4)
        if skip:
            self.skip_norm = FP32LayerNorm(dim, eps=1e-5)
            self.skip_linear = nn.Linear(2 * dim, dim)
        self.self_attention, self.cross_attention, self.skip = self_attention, cross_attention, skip

    def forward(self, x, keys_values=None, skip=None, joined=False):
        """x (batch, tokens, dim). keys_values: the context keys and values of attn2 (see Attention.cross), or None for the zero-image
        branch, whose cross-attention output is exactly attn2's output bias (all its keys and values are zero). joined: the self-attention
        attends the tokens of all batch items together (PartCrafter's global blocks)."""
        if self.skip:
            x = self.skip_norm(self.skip_linear(torch.cat([skip, x], dim=-1)))
        if self.self_attention:
            hidden = self.norm1(x)
            x = x + self.attn1([hidden.reshape(1, -1, hidden.shape[-1]) if joined else hidden]).reshape(x.shape)
            del hidden
        if self.cross_attention:
            if keys_values is None:
                x = x + self.attn2.to_out[0].bias
            else:
                x = x + self.attn2.cross(self.norm2(x), keys_values)
        return x.add_(self.ff(self.norm3(x)))


class TripoSGDiT(nn.Module):
    """TripoSGDiTModel, and PartCrafterDiTModel when max_num_parts is set (part embeddings and global attention blocks)."""

    def __init__(self, in_channels=64, width=2048, num_layers=21, num_attention_heads=16, cross_attention_dim=1024, max_num_parts=0,
                 global_attn_block_ids=(), **_):
        super().__init__()
        self.width, self.num_layers, self.global_blocks = width, num_layers, set(global_attn_block_ids)
        self.time_proj = nn.Module()
        self.time_proj.linear_1 = nn.Linear(width, width * 4)
        self.time_proj.linear_2 = nn.Linear(width * 4, width)
        if max_num_parts:
            self.part_embedding = nn.Embedding(max_num_parts, width)
        self.proj_in = nn.Linear(in_channels, width)
        self.blocks = nn.ModuleList([DiTBlock(width, num_attention_heads, cross_attention_dim, skip=layer > num_layers // 2) for layer in range(num_layers)])
        self.norm_out = nn.LayerNorm(width)
        self.proj_out = nn.Linear(width, in_channels)
        # It holds no weights itself: its lock keeps MMGP from rounding the float32 timestep to FP16 before the FP32 sinusoid (the
        # pipeline passes FP16 latents, whose dtype the embedding is cast to, and FP16 contexts)
        self._lock_dtype = torch.float32

    def forward(self, x, t, contexts, part_ids=None):
        """x (batch, latents, channels) and t (batch,) shared by the guidance branches; contexts: the image conditioning of each branch,
        (1, tokens, cross_attention_dim), or None for the zero-image branch; part_ids: the part embedding indices (batch,) of each branch
        (PartCrafter). The branches run one after the other in each block. Returns one prediction per branch."""
        temb = self.time_proj.linear_2(F.gelu(self.time_proj.linear_1(timestep_embedding(t, self.width).to(x.dtype))))
        x = torch.cat([temb[:, None], self.proj_in(x)], dim=1)
        del temb
        states = [x] * len(contexts) if part_ids is None else [x + self.part_embedding(ids)[:, None] for ids in part_ids]
        del x
        skips = [[] for _ in contexts]
        for layer, block in enumerate(self.blocks):
            joined = layer in self.global_blocks
            for branch, context in enumerate(contexts):
                keys_values = None if context is None else block.attn2.keys_values(context)
                skip = skips[branch].pop() if layer > self.num_layers // 2 else None
                states[branch] = block(states[branch], keys_values, skip, joined)
                del keys_values, skip
                if layer < self.num_layers // 2:
                    skips[branch].append(states[branch])
            check_abort()
        return [self.proj_out(self.norm_out(state)[:, 1:]) for state in states]
