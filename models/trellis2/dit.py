# TRELLIS.2 flow transformer, inference port of microsoft/TRELLIS.2 (trellis2/models/sparse_structure_flow.py, structured_latent_flow.py,
# trellis2/modules/{transformer,sparse/transformer,attention,sparse/attention}, MIT License, see LICENSE in this folder).
# One class serves the five flow models: the sparse-structure flow attends over its dense 16^3 grid flattened to 4096 tokens, the shape
# and texture flows over the active voxels of one sample (batch 1, so the sparse attention is a plain attention over all of them).
# Precision follows upstream: the blocks run in BF16 with FP32 norms, q/k RMS norms and RoPE; the timestep embedder, the shared
# AdaLN projection and the input/output layers compute in FP32 (upstream keeps them FP32 and loads the BF16 checkpoint into them).
# The q/k norms and RoPE run in the shared kernels (shared.kernels.ops), fused with the SageAttention 2 quantization when it is used;
# the AdaLN modulation, the output projections with their (gated) residuals and the FFN run in the shared operations as well.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared import attention_kit
from shared.kernels import ops
from shared.utils.phase_progress import check_abort
from .sparse import row_chunks

def fused_split_map(channels):
    """mmgp split map of the fused upstream projections: self-attention to_qkv and cross-attention to_kv become separate q/k/v linears."""
    return {"to_qkv": {"mapped_modules": ["to_q", "to_k", "to_v"], "split_sizes": [channels] * 3},
            "to_kv": {"mapped_modules": ["to_k", "to_v"], "split_sizes": [channels] * 2}}


def rope_phases(coords, head_dim, rope_freq=(1.0, 10000.0)):
    """Upstream RotaryPositionEmbedder phases (tokens, head_dim / 2) complex64 of integer voxel coords (tokens, 3), computed on the device
    of coords: per axis head_dim // 6 frequencies, the remaining pair left unrotated."""
    freq_dim = head_dim // 2 // 3
    freqs = torch.arange(freq_dim, dtype=torch.float32, device="cpu") / freq_dim
    freqs = (rope_freq[0] / (rope_freq[1] ** freqs)).to(coords.device)
    phases = torch.outer(coords.reshape(-1), freqs)
    phases = torch.polar(torch.ones_like(phases), phases).reshape(coords.shape[0], -1)
    pad = head_dim // 2 - phases.shape[-1]
    if pad:
        shape = (coords.shape[0], pad)
        phases = torch.cat([phases, torch.polar(torch.ones(shape, device=coords.device), torch.zeros(shape, device=coords.device))], dim=-1)
    return phases


class MultiHeadRMSNorm(nn.Module):
    """Upstream F.normalize(x) * gamma * sqrt(head_dim) in FP32, rounded: the RMSNorm with eps 1e-24 / head_dim (F.normalize's
    1e-12 floor on the norm) and a weight row per head."""

    def __init__(self, dim, heads):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def heads(self, heads):
        """(weight, eps) of the shared q/k kernels for the head range heads."""
        return self.gamma[heads.start:heads.stop], 1e-24 / self.gamma.shape[-1]


class SelfAttention(nn.Module):
    def __init__(self, channels, heads):
        super().__init__()
        self.heads, self.head_dim = heads, channels // heads
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, heads)
        self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, heads)
        self.to_out = nn.Linear(channels, channels)

    def attend(self, x_list, rope):
        """Attention output (batch, tokens, heads, head_dim) before to_out. Upstream norm then RoPE: the norm output rounded, then
        rotated in FP32 and rounded (or quantized for SageAttention)."""
        def norm_rope(query, key, group):
            ops.interleaved_norm_rope_(query, key, rope, self.q_rms_norm.heads(group.q), self.k_rms_norm.heads(group.kv))

        def fused_quant(query, key, group, core, settings):
            return ops.split_half_norm_rope_quant_(query, key, rope, self.q_rms_norm.heads(group.q), self.k_rms_norm.heads(group.kv), core, settings,
                                                   interleaved=True)

        return attention_kit.qkv_attention(x_list, self.to_q, self.to_k, self.to_v, self.heads, self.head_dim, norm_rope, fused_quant=fused_quant)


class CrossAttention(nn.Module):
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


class FeedForward(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, channels))


class TransformerBlock(nn.Module):
    """ModulatedTransformerCrossBlock / ModulatedSparseTransformerCrossBlock with a shared AdaLN modulation and a per-block offset."""

    def __init__(self, channels, ctx_channels, heads, mlp_ratio):
        super().__init__()
        self.hidden = int(channels * mlp_ratio)
        self.norm1 = nn.LayerNorm(channels, eps=1e-6, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6, elementwise_affine=True)
        self.norm3 = nn.LayerNorm(channels, eps=1e-6, elementwise_affine=False)
        self.self_attn = SelfAttention(channels, heads)
        self.cross_attn = CrossAttention(channels, ctx_channels, heads)
        self.mlp = FeedForward(channels, self.hidden)
        self.modulation = nn.Parameter(torch.zeros(6 * channels))

    def forward(self, x, mod, context, rope):
        """x (1, tokens, channels) BF16, updated in place; mod (1, 6 * channels) BF16; rope (1, tokens, 1, head_dim / 2, 2) FP32."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.modulation.float() + mod).to(mod.dtype).chunk(6, dim=1)
        rows = x.view(-1, x.shape[-1])
        spans = [(0, rows.shape[0], 0)]
        # each attention input is handed off in its list: the attention releases it after its projections
        y = self.self_attn.attend([ops.norm_modulate([rows], self.norm1, shift_msa, scale_msa, spans).view(x.shape)], rope)
        ops.gated_linear(self.self_attn.to_out, [y.view(rows.shape)], rows, gate_msa, spans)
        y = self.cross_attn.attend([self.norm2(x)], context)
        ops.linear_residual_(self.cross_attn.to_out, [y.view(rows.shape)], rows)
        del y
        ops.mlp([ops.norm_modulate([rows], self.norm3, shift_mlp, scale_mlp, spans)], self.mlp.mlp[0], self.mlp.mlp[2], "gelu", residual=rows, gate=gate_mlp[0])
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))

    def forward(self, t):
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(start=0, end=half, dtype=torch.float32, device="cpu") / half).to(t.device)
        args = t[:, None].float() * freqs[None]
        return self.mlp(torch.cat([torch.cos(args), torch.sin(args)], dim=-1))


class TrellisFlowDiT(nn.Module):
    """SparseStructureFlowModel / SLatFlowModel (share_mod, QK RMS norms, 3D RoPE) over the tokens of one sample."""

    def __init__(self, in_channels, out_channels, model_channels=1536, cond_channels=1024, num_blocks=30, num_heads=12, mlp_ratio=5.3334,
                 rope_freq=(1.0, 10000.0), resolution=None):
        super().__init__()
        self.in_channels, self.out_channels, self.resolution = in_channels, out_channels, resolution
        self.head_dim, self.rope_freq = model_channels // num_heads, tuple(rope_freq)
        self.t_embedder = TimestepEmbedder(model_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 6 * model_channels))
        self.input_layer = nn.Linear(in_channels, model_channels)
        self.blocks = nn.ModuleList([TransformerBlock(model_channels, cond_channels, num_heads, mlp_ratio) for _ in range(num_blocks)])
        self.out_layer = nn.Linear(model_channels, out_channels)

    def phases(self, coords):
        return rope_phases(coords, self.head_dim, self.rope_freq)

    def forward(self, x, t, contexts, phases):
        """x (1, tokens, in_channels) FP32, t (1,) FP32 timestep in [0, 1000], contexts: list of (1, cond tokens, cond_channels) BF16
        conditionings (CFG branches, joint pass: each block runs every branch while its weights are loaded), phases (tokens, head_dim / 2)
        complex64. Returns one (1, tokens, out_channels) FP32 prediction per context."""
        with torch.autocast(device_type=x.device.type, dtype=torch.float32):
            h = self.input_layer(x).to(torch.bfloat16)
            mod = self.adaLN_modulation(self.t_embedder(t)).to(torch.bfloat16)
        hidden = [h] + [h.clone() for _ in contexts[1:]]
        del h
        rope = torch.view_as_real(phases)[None, :, None]  # cos / sin of each channel pair
        for block in self.blocks:
            for h, context in zip(hidden, contexts):
                block(h, mod, context, rope)
            check_abort()
        outputs = []
        while hidden:
            h = hidden.pop(0)
            out = h.new_empty((*h.shape[:-1], self.out_channels), dtype=torch.float32)
            with torch.autocast(device_type=x.device.type, dtype=torch.float32):
                for start, stop in row_chunks(h.shape[1], h.shape[-1]):
                    part = h[:, start:stop].float()
                    out[:, start:stop] = self.out_layer(F.layer_norm(part, part.shape[-1:]))
            outputs.append(out)
            del h
        return outputs
