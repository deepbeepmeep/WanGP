# Hi3D v1.0 sparse flow transformer, inference port of hi3d-official/Hi3Dv1.0 (sparc3d/models/transformers/sparc3d_transformer.py,
# sparc3d/nn/sparse/{attention,transformer,spatial,conv}.py and sparc3d/nn/rope.py: Hi3D Open License Agreement 1.0, with parts adapted
# from Microsoft TRELLIS under the MIT License; see LICENSE, NOTICE and THIRD_PARTY_LICENSES.md in this folder).
# One sample (batch 1). The I/O residual blocks run on the active voxels, the 24 transformer blocks on the voxels pooled 2x (average of
# the children); self-attention rotates q/k (3D RoPE with upstream's long-grid frequency scaling) before their per-head RMS norms.
# Modified by DeepBeepMeep for WanGP: spconv replaced by the shared submanifold convolution (shared.kernels.vae_ops.sparse_conv, same
# weight layout), flash-attn by the shared attention kit (q/k rotation, norm and quantization in one pass), the classifier-free guidance
# branches run block by block (joint pass), the input path shared by both branches, and the upsampling output block works on the pooled
# rows (its norms and skip projection are row-wise, so the 2x2x2 copies are only gathered by its first convolution).
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.trellis2.sparse import SparseTensor, layer_norm_rows_
from shared import attention_kit
from shared.kernels import ops, vae_ops
from shared.utils.lora_mapping import convert_lora_keys
from shared.utils.phase_progress import check_abort


def fused_split_map(channels):
    """mmgp split map of the fused upstream projections: self-attention to_qkv and cross-attention to_kv become separate linears."""
    return {"to_qkv": {"mapped_modules": ["to_q", "to_k", "to_v"], "split_sizes": [channels] * 3},
            "to_kv": {"mapped_modules": ["to_k", "to_v"], "split_sizes": [channels] * 2}}


class SparseConv3d(nn.Module):
    """Submanifold 3x3x3 convolution, weight (Co, 3, 3, 3, Ci) under .conv as in the spconv checkpoints."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Module()
        self.conv.weight = nn.Parameter(torch.empty(out_channels, 3, 3, 3, in_channels))
        self.conv.bias = nn.Parameter(torch.empty(out_channels))

    def forward(self, feats, level, rowmap=None):
        """feats (rows, Ci); level: SparseTensor of the output coords (neighbor lookup); rowmap: row of feats of each level voxel
        (feats on the pooled grid), None when feats are the level's own rows. Returns (level rows, Co)."""
        return vae_ops.sparse_conv(feats, level, self.conv.weight, self.conv.bias, rowmap=rowmap)


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(nn.Linear(frequency_embedding_size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))

    def forward(self, t, dtype):
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        return self.mlp(torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(dtype))


def rope_table(coords, head_dim):
    """Upstream RotaryPositionEmbedder phases of the integer coords (tokens, 3) as a cos / sin table (1, tokens, 1, head_dim / 2, 2) FP32
    of the interleaved channel pairs: head_dim // 6 frequencies per axis, axis-major, rescaled as upstream when the largest coordinate
    reaches the 64 positions of training."""
    freq_dim = head_dim // 3 // 2
    freqs = 1.0 / (10000 ** (torch.arange(freq_dim, dtype=torch.float32, device=coords.device) / freq_dim))
    longest = int(coords.max())
    if longest >= 64:
        train, infer = 64, (64 + longest) // 2
        wavelengths = freqs * train / (2 * math.pi)
        ratio = (wavelengths - 1) / (24 - 1)
        ratio = ratio + (1 - ratio) * train / infer
        ratio = torch.where(wavelengths > 24, 1.0, ratio)
        freqs = freqs * torch.where(wavelengths < 1, train / infer, ratio)
    phases = torch.outer(coords.reshape(-1).float(), freqs).reshape(coords.shape[0], -1)
    pad = head_dim // 2 - phases.shape[1]
    if pad:
        phases = F.pad(phases, (0, pad))
    return torch.stack((phases.cos(), phases.sin()), dim=-1)[None, :, None]


class MultiHeadRMSNorm(nn.Module):
    """Upstream's F.normalize(x) * gamma * sqrt(dim) with a gamma per head: an RMS norm of epsilon 1e-24 / dim (F.normalize's 1e-12 floor
    of the L2 norm)."""

    def __init__(self, dim, heads):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(heads, dim))

    def heads(self, heads):
        """(weight, eps) of the shared norm operations for the heads of a head group."""
        return self.gamma[heads.start:heads.stop], 1e-24 / self.gamma.shape[-1]


class SelfAttention(nn.Module):
    def __init__(self, channels, heads):
        super().__init__()
        self.heads, self.head_dim = heads, channels // heads
        self.to_q, self.to_k, self.to_v = (nn.Linear(channels, channels) for _ in range(3))
        self.q_rms_norm = MultiHeadRMSNorm(self.head_dim, heads)
        self.k_rms_norm = MultiHeadRMSNorm(self.head_dim, heads)
        self.to_out = nn.Linear(channels, channels)

    def attend(self, x_list, rope):
        """Upstream rotates q and k before their norms: rotation and norm in FP32, rounded once (or quantized for SageAttention)."""
        def rope_norm(query, key, group):
            ops.interleaved_norm_rope_(query, key, rope, self.q_rms_norm.heads(group.q), self.k_rms_norm.heads(group.kv), rope_first=True)

        def fused_quant(query, key, group, core, settings):
            return ops.split_half_norm_rope_quant_(query, key, rope, self.q_rms_norm.heads(group.q), self.k_rms_norm.heads(group.kv), core, settings,
                                                   interleaved=True, rope_first=True)

        return attention_kit.qkv_attention(x_list, self.to_q, self.to_k, self.to_v, self.heads, self.head_dim, rope_norm, fused_quant=fused_quant)


class CrossAttention(nn.Module):
    def __init__(self, channels, ctx_channels, heads):
        super().__init__()
        self.heads, self.head_dim = heads, channels // heads
        self.to_q = nn.Linear(channels, channels)
        self.to_k, self.to_v = nn.Linear(ctx_channels, channels), nn.Linear(ctx_channels, channels)
        self.to_out = nn.Linear(channels, channels)

    def attend(self, x_list, context):
        return attention_kit.cross_qkv_attention(x_list, [context], self.to_q, self.to_k, self.to_v, self.heads, self.head_dim, lambda query, key, group: None,
                                                 fused_quant=lambda query, key, group, core, settings: ops.split_half_norm_rope_quant_(query, key, None, None, None, core, settings))

    def null_output(self, dtype):
        """Output (1, channels) for the all-zero condition of the guidance branch: its keys and values are the projection biases, the
        same for every condition token, so the attention weights are uniform and every query gets the value bias."""
        return self.to_out(self.to_v.bias.to(dtype).view(1, -1))


class FeedForward(nn.Module):
    def __init__(self, channels, hidden):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, channels))


class TransformerBlock(nn.Module):
    """ModulatedSparseTransformerCrossBlock: shared AdaLN modulation plus a per-block offset."""

    def __init__(self, channels, ctx_channels, heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels, eps=1e-6, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6)
        self.norm3 = nn.LayerNorm(channels, eps=1e-6, elementwise_affine=False)
        self.self_attn = SelfAttention(channels, heads)
        self.cross_attn = CrossAttention(channels, ctx_channels, heads)
        self.mlp = FeedForward(channels, int(channels * mlp_ratio))
        self.modulation = nn.Parameter(torch.zeros(1, 6 * channels))

    def forward(self, x, mod, context, rope):
        """x (tokens, channels) updated in place; mod (1, 6 * channels); context (1, cond tokens, ctx_channels), None for the all-zero
        condition of the unconditional guidance branch."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.modulation.float() + mod.float()).chunk(6, dim=1)
        spans = [(0, x.shape[0], 0)]
        y = ops.norm_modulate([x], self.norm1, shift_msa, scale_msa, spans)
        y = self.self_attn.attend([y.unsqueeze(0)], rope)
        ops.gated_linear(self.self_attn.to_out, [y.reshape(x.shape)], x, gate_msa, spans)
        if context is None:
            x.add_(self.cross_attn.null_output(x.dtype))
        else:
            y = self.cross_attn.attend([self.norm2(x).unsqueeze(0)], context)
            ops.linear_residual_(self.cross_attn.to_out, [y.reshape(x.shape)], x)
            del y
        y = ops.norm_modulate([x], self.norm3, shift_mlp, scale_mlp, spans)
        ops.mlp([y], self.mlp.mlp[0], self.mlp.mlp[2], "gelu", residual=x, gate=gate_mlp[0])
        return x


class ResBlock(nn.Module):
    """SparseResBlock3d: timestep-modulated residual block of two submanifold convolutions, with the 2x pooling before it or the 2x
    upsampling (restored coords of the paired pooling) at its start."""

    def __init__(self, channels, emb_channels, out_channels=None, downsample=False, upsample=False):
        super().__init__()
        out_channels = out_channels or channels
        hidden = out_channels // 2
        self.downsample, self.upsample = downsample, upsample
        self.norm1 = nn.LayerNorm(channels, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden, eps=1e-6, elementwise_affine=False)
        self.conv1 = SparseConv3d(channels, hidden)
        self.conv2 = SparseConv3d(hidden, hidden)
        self.linear = nn.Linear(hidden, out_channels)
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(emb_channels, 2 * hidden))
        self.skip_connection = nn.Linear(channels, out_channels) if channels != out_channels else None

    def forward(self, x_list, emb, plan):
        """x handed off in x_list (rows, channels): the fine voxels, or with upsample the pooled ones. Returns the output rows: pooled
        with downsample, fine otherwise."""
        x = x_list.pop()
        if self.downsample:
            x = plan.pool(x)
        level = plan.coarse if self.downsample else plan.fine
        rowmap = plan.inverse if self.upsample else None
        scale, shift = self.emb_layers(emb).chunk(2, dim=1)
        skip = x if self.skip_connection is None else self.skip_connection(x)
        h = layer_norm_rows_(x.clone() if skip is x else x, self.norm1.weight, self.norm1.bias, 1e-6, F.silu)
        del x
        h = self.conv1(h, level, rowmap)
        h = F.silu(ops.norm_modulate([h], self.norm2, shift, scale, [(0, h.shape[0], 0)]), inplace=True)
        h = F.silu(self.conv2(h, level), inplace=True)
        out = self.linear(h)
        del h
        return out.add_(skip if rowmap is None else skip.index_select(0, rowmap))


class SparsePlan:
    """Per-pass sparse structures of one set of voxels: the fine level, the 2x pooled level, the pooling map and the RoPE tables of the
    pooled tokens. Upstream pools by sorted unique codes, so the pooled voxels are in (x, y, z) order."""

    def __init__(self, coords, head_dim):
        self.fine = SparseTensor(None, coords).keep_neighbors()
        parent = coords.long() // 2
        span = int(parent.max()) + 1
        codes, self.inverse = torch.unique((parent[:, 0] * span + parent[:, 1]) * span + parent[:, 2], return_inverse=True)
        pooled = torch.stack([codes // (span * span), codes // span % span, codes % span], dim=1).to(coords.dtype)
        self.counts = torch.bincount(self.inverse, minlength=codes.shape[0]).unsqueeze(1) + 1  # upstream's scatter_reduce mean includes the zero init
        self.coarse = SparseTensor(None, pooled).keep_neighbors()
        self.rope = rope_table(pooled, head_dim)

    def pool(self, x):
        """Upstream SparseDownsample: the sum of the fine rows of each pooled voxel over their count plus one (torch.scatter_reduce "mean"
        with its default include_self over a zero tensor), FP32 sums rounded once."""
        out = torch.zeros((self.counts.shape[0], x.shape[1]), dtype=torch.float32, device=x.device)
        return out.index_add_(0, self.inverse, x.float()).div_(self.counts).to(x.dtype)


class Sparc3DTransformer(nn.Module):
    """Sparc3DTransformer: sparse I/O residual stages around shared-modulation transformer blocks."""

    def __init__(self, resolution=64, in_channels=32, out_channels=32, model_channels=2112, cond_channels=2048, num_blocks=24, num_heads=22,
                 mlp_ratio=4.0, patch_size=2, num_io_res_blocks=2, io_block_channels=(256,), use_fp16=True, latent_mean=None, latent_std=None):
        super().__init__()
        assert patch_size == 2 and len(io_block_channels) == 1, "one 2x I/O stage, as the released checkpoints"
        self.resolution, self.model_channels, self.head_dim = resolution, model_channels, model_channels // num_heads
        self.dtype = torch.bfloat16  # compute dtype of the blocks, set by the loader to the checkpoint's
        self.latent_mean, self.latent_std = latent_mean, latent_std
        io = io_block_channels[0]
        self.t_embedder = TimestepEmbedder(model_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 6 * model_channels))
        self.input_layer = nn.Linear(in_channels, io)
        self.input_blocks = nn.ModuleList([ResBlock(io, model_channels) for _ in range(num_io_res_blocks - 1)]
                                          + [ResBlock(io, model_channels, model_channels, downsample=True)])
        self.blocks = nn.ModuleList([TransformerBlock(model_channels, cond_channels, num_heads, mlp_ratio) for _ in range(num_blocks)])
        self.out_blocks = nn.ModuleList([ResBlock(model_channels * 2, model_channels, io, upsample=True)]
                                        + [ResBlock(io * 2, model_channels, io) for _ in range(num_io_res_blocks - 1)])
        self.out_layer = nn.Linear(io, out_channels)

    def plan(self, coords):
        return SparsePlan(coords, self.head_dim)

    def preprocess_loras(self, model_type, state_dict):
        return convert_lora_keys(state_dict, dict(self.named_modules()), fused_split_map=fused_split_map(self.model_channels))

    def forward(self, x, t, contexts, plan):
        """x (voxels, in_channels) FP32 noisy latents, t (1,) FP32 model time (1000 * sigma), contexts: the conditionings of the guidance
        branches (1, cond tokens, cond_channels), or None for the all-zero condition, each run block by block while the block's weights
        are loaded. Returns one (voxels, out_channels) FP32 velocity per context."""
        dtype = self.dtype
        t_emb = self.t_embedder(t, dtype)
        emb = t_emb.to(dtype)
        mod = self.adaLN_modulation(t_emb).to(dtype)
        h = self.input_layer(x.to(dtype))
        skips = []
        for block in self.input_blocks:
            h = block([h], emb, plan)
            skips.append(h)
        hidden = [h.clone() for _ in contexts]  # the blocks update in place; h stays the skip of the first output block
        del h
        for block in self.blocks:
            for h, context in zip(hidden, contexts):
                block(h, mod, context, plan.rope)
            check_abort()
        outputs = []
        while hidden:
            h = hidden.pop(0)
            for block, skip in zip(self.out_blocks, reversed(skips)):
                h = block([torch.cat((h, skip), dim=1)], emb, plan)
            outputs.append(self.out_layer(F.layer_norm(h, h.shape[-1:])).float())
            del h
        return outputs
