# Transformer layers shared by the TripoSplat flow model and Gaussian decoder, ported from VAST-AI-Research/TripoSplat model.py (MIT
# License, Copyright (c) 2026 VAST, see LICENSE in this folder). Modified by DeepBeepMeep for WanGP: fused q/k/v projections split at load
# for the WanGP attention kit, q/k norms and RoPE in the shared kernels (shared.kernels.ops: in place, or fused with the SageAttention 2
# quantization), the output projections and FFNs written into their residuals by the shared operations. The arithmetic follows upstream
# (FP16 checkpoint, FP32 layer norms, q/k RMS norms and RoPE), except that the rotation and the norm after it are rounded once instead
# of twice, and a (gated) residual update once instead of twice.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared import attention_kit
from shared.kernels import ops


def layer_norm(x, weight=None, bias=None, eps=1e-6):
    """Upstream LayerNorm32: computed in FP32, rounded to the activation dtype."""
    return F.layer_norm(x.float(), x.shape[-1:], None if weight is None else weight.float(), None if bias is None else bias.float(), eps).to(x.dtype)


def sinusoidal_embedding(t, dim, max_period, level):
    """Upstream TimestepEmbedder (level False) / LevelEmbedder (level True: angles times 2 pi) embeddings of t (batch,). The frequencies are
    computed on the CPU, as upstream builds them without a device, then moved to the device of t."""
    half = dim // 2
    freqs = torch.exp(-np.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device="cpu") / half).to(device=t.device)
    args = t[:, None].float() * freqs[None] * 2 * torch.pi if level else t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class EmbedderMLP(nn.Module):
    """TimestepEmbedder (max_period 10000) and LevelEmbedder (max_period 1024, level True)."""

    def __init__(self, hidden_size, max_period=10000, level=False, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
        self.max_period, self.level, self.frequency_embedding_size = max_period, level, frequency_embedding_size

    def forward(self, t):
        emb = sinusoidal_embedding(t, self.frequency_embedding_size, self.max_period, self.level)
        return self.mlp(emb.to(self.mlp[0].weight.dtype))


class FeedForward(nn.Module):
    """Linear, GELU (tanh), Linear: run by shared.kernels.ops.mlp on mlp[0] and mlp[2]."""

    def __init__(self, channels, mlp_ratio=4.0):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(channels, int(channels * mlp_ratio)), nn.GELU(approximate="tanh"), nn.Linear(int(channels * mlp_ratio), channels))


class MultiHeadRMSNorm(nn.Module):
    """Upstream F.normalize(x) * gamma * sqrt(head_dim) in FP32: the RMSNorm with eps 1e-24 / head_dim (F.normalize's 1e-12 floor on
    the norm) and a weight row per head."""

    def __init__(self, dim, heads):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def heads(self, heads):
        """(weight, eps) of the shared q/k kernels for the head range heads."""
        return self.gamma[heads.start:heads.stop], 1e-24 / self.gamma.shape[-1]


class SelfAttention(nn.Module):
    """Upstream RopeMultiHeadAttention / MultiHeadAttention (self) with q/k RMS norms, its fused qkv projection split at load."""

    def __init__(self, channels, heads, names=("q", "k", "v", "q_norm", "k_norm", "out")):
        super().__init__()
        self.heads, self.head_dim, self.names = heads, channels // heads, names
        for name in names[:3] + names[5:]:
            setattr(self, name, nn.Linear(channels, channels))
        for name in names[3:5]:
            setattr(self, name, MultiHeadRMSNorm(self.head_dim, heads))

    @property
    def output(self):
        """The output projection (out in the flow, to_out in the Gaussian decoder)."""
        return getattr(self, self.names[5])

    def attend(self, x_list, phases=None):
        """Attention output (batch, tokens, heads, head_dim) before the output projection. phases: complex64 (batch, tokens, heads,
        head_dim / 2) of upstream's learned RoPE, applied to consecutive channel pairs before the norms (one rounding after both), or None."""
        q, k, v, q_norm, k_norm = (getattr(self, name) for name in self.names[:5])
        rope = None if phases is None else torch.view_as_real(phases)
        head_rope = lambda heads: None if rope is None else rope[:, :, heads.start:heads.stop]  # q and kv heads coincide (no grouped keys)

        def norm_rope(query, key, group):
            ops.interleaved_norm_rope_(query, key, head_rope(group.q), q_norm.heads(group.q), k_norm.heads(group.kv), rope_first=True)

        def fused_quant(query, key, group, core, settings):
            return ops.split_half_norm_rope_quant_(query, key, head_rope(group.q), q_norm.heads(group.q), k_norm.heads(group.kv), core, settings,
                                                   interleaved=True, rope_first=True)

        return attention_kit.qkv_attention(x_list, q, k, v, self.heads, self.head_dim, norm_rope, fused_quant=fused_quant)


class CrossAttention(nn.Module):
    """Upstream MultiHeadAttention (cross) with q/k RMS norms, its fused kv projection split at load."""

    def __init__(self, channels, ctx_channels, heads):
        super().__init__()
        self.heads, self.head_dim = heads, channels // heads
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(ctx_channels, channels)
        self.to_v = nn.Linear(ctx_channels, channels)
        self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, heads)
        self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, heads)
        self.to_out = nn.Linear(channels, channels)

    def attend(self, x_list, context):
        """Attention output (batch, tokens, heads, head_dim) before to_out."""
        def norm(query, key, group):
            ops.interleaved_norm_rope_(query, key, None, self.q_rms_norm.heads(group.q), self.k_rms_norm.heads(group.kv))

        def fused_quant(query, key, group, core, settings):
            return ops.split_half_norm_rope_quant_(query, key, None, self.q_rms_norm.heads(group.q), self.k_rms_norm.heads(group.kv), core, settings)

        return attention_kit.cross_qkv_attention(x_list, [context], self.to_q, self.to_k, self.to_v, self.heads, self.head_dim, norm, fused_quant=fused_quant)
