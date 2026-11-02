# TripoSplat Gaussian decoder (OctreeGaussianDecoder: OctreeProbabilityFixedlenDecoder + ElasticGaussianFixedlenDecoder, and the Gaussian
# activations of triposplat.py), ported from VAST-AI-Research/TripoSplat model.py and triposplat.py (MIT License, Copyright (c) 2026 VAST,
# see LICENSE in this folder). Modified by DeepBeepMeep for WanGP: random draws from the generation's generator (upstream uses the global
# RNG), fused projections split at load for the attention kit, in-place residual updates, progress and abort checks, and the Gaussian
# activations computed in FP32 from the FP16 network outputs (upstream applies them in FP16).
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.kernels import ops
from shared.utils.phase_progress import check_abort
from .layers import CrossAttention, EmbedderMLP, FeedForward, SelfAttention, layer_norm

CHILD_OFFSETS = [[i, j, k] for k in (0, 1) for j in (0, 1) for i in (0, 1)]


def fused_split_map(channels):
    """mmgp split map of the fused upstream projections: self-attention to_qkv and cross-attention to_kv."""
    return {"to_qkv": {"mapped_modules": ["to_q", "to_k", "to_v"], "split_sizes": [channels] * 3},
            "to_kv": {"mapped_modules": ["to_k", "to_v"], "split_sizes": [channels] * 2}}


def point_embedding(points, channels, max_res=10):
    """Upstream PcdAbsolutePositionEmbedderV2 of points (N, 3) FP32: sin / cos at 2^k frequencies, k evenly spaced in [0, max_res]."""
    freqs = torch.pow(2.0, torch.linspace(0.0, float(max_res), steps=channels // 3 // 2, dtype=torch.float32, device=points.device))
    angles = points.unsqueeze(-1) * freqs * torch.pi
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1).reshape(points.shape[0], -1)
    return F.pad(embedding, (0, channels - embedding.shape[1]))


def sample_counts(probs, counts, generator):
    """Upstream sample_probs ("systematic"): splits counts (B, P) among the 8 children of each parent with probs (B, P, 8) by systematic
    resampling of each parent's count. Returns (B, P * 8) long."""
    batch_shape = counts.shape
    rows, size = counts.numel(), probs.shape[-1]
    probs = probs.reshape(rows, size).to(torch.float32).clamp_min_(0)
    counts = counts.reshape(rows).long()
    row_sums = probs.sum(1, keepdim=True)
    zero = row_sums.eq(0)
    probs = probs / row_sums.clamp_min_(1)
    if zero.any():
        probs[zero.expand_as(probs)] = 1.0 / size
    out = torch.zeros(rows, size, dtype=torch.long, device=probs.device)
    cdf = probs.cumsum(dim=1).clamp(max=1.0 - 1e-12)
    unique, inverse = counts.unique(sorted=False, return_inverse=True)
    for i, n in enumerate(unique.tolist()):
        if n == 0:
            continue
        selected = (inverse == i).nonzero(as_tuple=False).squeeze(1)
        start = torch.rand(selected.numel(), 1, device=probs.device, generator=generator) / float(n)
        grid = torch.arange(n, device=probs.device, dtype=torch.float32)[None, :] / float(n)
        index = torch.searchsorted(cdf.index_select(0, selected), (start + grid).clamp(max=1.0 - 1e-12)).clamp_max(size - 1)
        hits = torch.zeros(selected.numel(), size, dtype=torch.float32, device=probs.device)
        hits.scatter_add_(1, index, torch.ones_like(index, dtype=hits.dtype))
        out.index_copy_(0, selected, hits.long())
    return out.view(*batch_shape, size).flatten(1, 2)


class OctreeBlock(nn.Module):
    """ModulatedTransformerCrossOnlyBlock with a shared modulation: cross-attention to the latent and an MLP. Updates x in place."""

    def __init__(self, channels, ctx_channels, num_heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels, eps=1e-6, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6, elementwise_affine=False)
        self.cross_attn = CrossAttention(channels, ctx_channels, num_heads)
        self.mlp = FeedForward(channels, mlp_ratio)

    def forward(self, x, mod, context):
        """x (batch, cells, channels), updated in place; mod (batch, 6 * channels): one modulation row per batch item."""
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        batch, cells, channels = x.shape
        rows = x.view(-1, channels)
        spans = [(b * cells, (b + 1) * cells, b) for b in range(batch)]
        y = self.cross_attn.attend([ops.norm_modulate([rows], self.norm1, shift_msa, scale_msa, spans).view(x.shape)], context)
        ops.gated_linear(self.cross_attn.to_out, [y.view(rows.shape)], rows, gate_msa, spans)
        del y
        ops.mlp([ops.norm_modulate([rows], self.norm2, shift_mlp, scale_mlp, spans)], self.mlp.mlp[0], self.mlp.mlp[2], "gelu", residual=rows,
                segments=ops.gate_segments(gate_mlp.unsqueeze(1), batch, cells))
        return x


class OctreeDecoder(nn.Module):
    """OctreeProbabilityFixedlenDecoder: probabilities of the 8 children of each occupied octree cell, given the cell centres."""

    def __init__(self, model_channels, cond_channels, num_blocks, num_heads, mlp_ratio):
        super().__init__()
        self.model_channels = model_channels
        self.in_proj = nn.Linear(3, model_channels)
        self.input_layer = nn.Linear(model_channels, model_channels)
        self.l_embedder = EmbedderMLP(model_channels, max_period=1024, level=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 6 * model_channels))
        self.blocks = nn.ModuleList([OctreeBlock(model_channels, cond_channels, num_heads, mlp_ratio) for _ in range(num_blocks)])
        self.out_proj = nn.Linear(model_channels, 8)

    def forward(self, centres, resolution, cond):
        """Logits (B, cells, 8) of the cells centred at centres (B, cells, 3) FP32 in [0, 1], at resolution (B,) long."""
        dtype = self.out_proj.bias.dtype
        batch, cells, _ = centres.shape
        h = self.in_proj(centres.to(dtype)) + point_embedding(centres.reshape(-1, 3), self.model_channels).reshape(batch, cells, -1).to(dtype)
        h = self.input_layer(h)
        mod = self.adaLN_modulation(self.l_embedder(resolution))
        cond = cond.to(dtype)
        for block in self.blocks:
            h = block(h, mod, cond)
        return self.out_proj(F.layer_norm(h.float(), h.shape[-1:]).to(dtype))


class GaussianBlock(nn.Module):
    """TransformerCrossBlock: self-attention, cross-attention to the latent, MLP. Updates x in place."""

    def __init__(self, channels, ctx_channels, num_heads, mlp_ratio):
        super().__init__()
        self.norm2 = nn.LayerNorm(channels, eps=1e-6)
        self.self_attn = SelfAttention(channels, num_heads, ("to_q", "to_k", "to_v", "q_rms_norm", "k_rms_norm", "to_out"))
        self.cross_attn = CrossAttention(channels, ctx_channels, num_heads)
        self.mlp = FeedForward(channels, mlp_ratio)

    def forward(self, x, context):
        rows = x.view(-1, x.shape[-1])
        ops.linear_residual_(self.self_attn.output, [self.self_attn.attend([layer_norm(x)]).view(rows.shape)], rows)
        ops.linear_residual_(self.cross_attn.to_out, [self.cross_attn.attend([layer_norm(x, self.norm2.weight, self.norm2.bias, self.norm2.eps)], context).view(rows.shape)], rows)
        ops.mlp([layer_norm(rows)], self.mlp.mlp[0], self.mlp.mlp[2], "gelu", residual=rows)
        return x


class GaussianDecoder(nn.Module):
    """ElasticGaussianFixedlenDecoder: num_gaussians Gaussians around each sampled point, then upstream's Gaussian activations."""

    LAYOUT = (("_xyz", 3), ("_features_dc", 3), ("_scaling", 3), ("_rotation", 4), ("_opacity", 1), ("_offset_scale", 1))

    def __init__(self, in_channels, model_channels, cond_channels, num_blocks, num_heads, mlp_ratio, num_gaussians, lr, perturbe_size,
                 filter_kernel_size_3d, scaling_bias, opacity_bias):
        super().__init__()
        self.model_channels, self.num_gaussians, self.lr, self.perturbe_size = model_channels, num_gaussians, lr, perturbe_size
        self.kernel_size, self.scaling_bias, self.opacity_bias = filter_kernel_size_3d, scaling_bias, opacity_bias
        self.in_proj = nn.Linear(in_channels, model_channels)
        self.input_layer = nn.Linear(model_channels, model_channels)
        self.blocks = nn.ModuleList([GaussianBlock(model_channels, cond_channels, num_heads, mlp_ratio) for _ in range(num_blocks)])
        self.out_proj = nn.Linear(model_channels, num_gaussians * sum(size for _, size in self.LAYOUT))
        # upstream buffers stored in the checkpoint: atanh of a 3D Hammersley set rescaled to (-1, 1) / perturbe_size, and the inverse
        # softplus of offset_scale
        self.register_buffer("points_offset_perturbation", torch.zeros(num_gaussians, 3))
        self.register_buffer("base_offset_scale", torch.zeros(()))

    def forward(self, points, cond, progress=None):
        """Gaussians (glTF-agnostic upstream axes, centred on the origin, unit box) of points (1, P, 3) FP32 in [0, 1] and the latent cond:
        dict of means, linear scales, unit quaternions (w, x, y, z), linear opacities and SH DC colours, FP32."""
        dtype = self.out_proj.bias.dtype
        batch, count, _ = points.shape
        h = self.in_proj(points.to(dtype)) + point_embedding(points.reshape(-1, 3), self.model_channels).reshape(batch, count, -1).to(dtype)
        h = self.input_layer(h)
        cond = cond.to(dtype)
        for block in self.blocks:
            h = block(h, cond)
            check_abort()
            if progress is not None:
                progress()
        features = self.out_proj(F.layer_norm(h.float(), h.shape[-1:]).to(dtype)).float()
        del h
        fields, start = {}, 0
        for name, size in self.LAYOUT:
            fields[name] = features[..., start:start + size * self.num_gaussians].reshape(batch, count, self.num_gaussians, size)
            start += size * self.num_gaussians
        offset_scale = F.softplus(fields["_offset_scale"] + self.base_offset_scale.float())
        offset = fields["_xyz"] * self.lr["_xyz"] + self.points_offset_perturbation.float()
        offset = torch.tanh(offset) * 0.5 * self.perturbe_size * offset_scale
        flat = lambda name: (fields[name] * self.lr[name]).flatten(1, 2)[0]
        # upstream Gaussian activations: biases are computed in FP32 on the CPU, then applied on the device
        scale_bias = torch.tensor(self.scaling_bias, device="cpu")
        scale_bias = (scale_bias + torch.log(-torch.expm1(-scale_bias))).to(points.device)
        opacity_bias = torch.tensor(self.opacity_bias, device="cpu")
        opacity_bias = torch.log(opacity_bias / (1 - opacity_bias)).to(points.device)
        scales = F.softplus(flat("_scaling") + scale_bias)
        rotations = flat("_rotation")
        rotations[:, 0] += 1
        return {"means": (points[:, :, None, :] + offset).flatten(1, 2)[0] - 0.5,
                "scales": torch.sqrt(torch.square(scales) + self.kernel_size ** 2),
                "rotations": F.normalize(rotations, dim=1),
                "opacities": torch.sigmoid(flat("_opacity") + opacity_bias)[:, 0],
                "sh_dc": flat("_features_dc"), "sh_rest": None, "sh_degree": 0}


class TripoSplatDecoder(nn.Module):
    """OctreeGaussianDecoder: octree sampling of the Gaussian anchor points from the latent, then the Gaussians around each point."""

    def __init__(self, octree, gaussians, octree_levels):
        super().__init__()
        self.octree_levels = octree_levels
        self.octree = OctreeDecoder(**octree)
        self.gs = GaussianDecoder(**gaussians)

    def sample_points(self, latent, num_points, generator, progress=None):
        """Upstream OctreeProbabilityFixedlenDecoder.sample (temperature 1, systematic): num_points cell centres (1, num_points, 3) FP32 in
        [0, 1], drawn level by level down the octree, with a uniform jitter inside the finest cells."""
        device, batch = latent.device, latent.shape[0]
        child_offset = torch.tensor(CHILD_OFFSETS, dtype=torch.long, device=device)
        coords = torch.zeros(batch, 1, 3, dtype=torch.long, device=device)
        counts = torch.full((batch, 1), num_points, dtype=torch.long, device=device)
        batch_index = torch.arange(batch, device=device).unsqueeze(1)
        for level in range(1, self.octree_levels + 1):
            centres = (coords.to(torch.float32) + 0.5) / (1 << (level - 1))
            logits = self.octree(centres, torch.full((batch,), 1 << level, dtype=torch.long, device=device), latent)
            sampled = sample_counts(torch.softmax(logits, dim=-1), counts, generator)
            children = (coords[:, :, None, :] * 2 + child_offset[None, None]).flatten(1, 2)
            mask = sampled > 0
            slots = mask.cumsum(dim=1)[mask] - 1
            rows = batch_index.expand_as(mask)[mask]
            width = int(mask.sum(dim=1).max())
            coords = torch.zeros(batch, width, 3, dtype=torch.long, device=device)
            coords[rows, slots] = children[mask]
            counts = torch.zeros(batch, width, dtype=torch.long, device=device)
            counts[rows, slots] = sampled[mask]
            check_abort()
            if progress is not None:
                progress()
        coords = torch.repeat_interleave(coords.flatten(0, 1), counts.flatten(0, 1), dim=0).reshape(batch, num_points, -1)
        jitter = torch.rand(coords.shape, dtype=torch.float32, device=device, generator=generator)
        return (coords.to(torch.float32) + jitter) / (1 << self.octree_levels)

    def forward(self, latent, num_points, generator, progress=None):
        return self.gs(self.sample_points(latent, num_points, generator, progress), latent, progress)
