# TripoSplat flow-matching transformer (LatentSeqMMFlowModel), ported from VAST-AI-Research/TripoSplat model.py (MIT License, Copyright (c)
# 2026 VAST, see LICENSE in this folder). Modified by DeepBeepMeep for WanGP: joint-pass CFG (the conditional and unconditional sequences
# go through each block one after the other), the noise, camera and context refiners computed once for both branches (the context
# refiners once per generation: they do not depend on the noisy latent or the timestep), in-place residual updates, abort checks.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.kernels import ops
from shared.utils.phase_progress import check_abort
from .layers import EmbedderMLP, FeedForward, SelfAttention, layer_norm


def fused_split_map(channels):
    """mmgp split map of the fused upstream self-attention projection."""
    return {"qkv": {"mapped_modules": ["q", "k", "v"], "split_sizes": [channels] * 3}}


class RePo3DRotaryEmbedding(nn.Module):
    """Learned 3D RoPE: per token and head, three positions predicted from the hidden states drive three groups of frequencies. Returns
    complex64 phases (batch, tokens, heads, head_dim / 2)."""

    def __init__(self, model_channels, num_heads, head_dim, repo_hidden_ratio=0.125, max_freq=16.0):
        super().__init__()
        self.num_heads = num_heads
        hidden = int(model_channels * repo_hidden_ratio)
        self.norm = nn.LayerNorm(model_channels)
        self.gate_map = nn.Linear(model_channels, hidden, bias=False)
        self.content_map = nn.Linear(model_channels, hidden, bias=False)
        self.final_map = nn.Linear(hidden, 3 * num_heads, bias=False)
        dims = (2 * (head_dim // 6), 2 * (head_dim // 6), head_dim - 4 * (head_dim // 6))
        for i, dim in enumerate(dims):
            setattr(self, f"freqs_{i}", nn.Parameter(torch.linspace(1.0, float(max_freq), steps=dim // 2, dtype=torch.float32)))

    def forward(self, hidden_states):
        h = layer_norm(hidden_states, self.norm.weight, self.norm.bias, self.norm.eps)
        delta = self.final_map(F.silu(self.gate_map(h)) * self.content_map(h)).unflatten(-1, (self.num_heads, 3))
        angles = []
        for i, freqs in enumerate((self.freqs_0, self.freqs_1, self.freqs_2)):
            x, tanh = delta[..., i].unsqueeze(-1), freqs.tanh()
            angles.append((x * tanh + x * (freqs - tanh)) * torch.pi)  # upstream clamp_mul (its straight-through gradient trick)
        angles = torch.cat(angles, dim=-1).float()
        return torch.polar(torch.ones_like(angles), angles)


def position_embedding(points, channels, max_res=16):
    """Upstream PcdAbsolutePositionEmbedder of points (..., 3) FP32: sin / cos of each coordinate at 2^k frequencies, zero padded."""
    freq_dim = channels // 3 // 2
    res_dim = max(0, freq_dim - max_res)
    freqs = torch.cat([torch.arange(max_res, dtype=torch.float32, device=points.device),
                       torch.arange(res_dim, dtype=torch.float32, device=points.device) / max(res_dim, 1) * max_res])[:freq_dim]
    out = torch.outer(points.reshape(-1), torch.pow(2.0, freqs)) * 2 * torch.pi
    out = torch.cat([out.sin(), out.cos()], dim=-1).reshape(*points.shape[:-1], -1)
    return F.pad(out, (0, channels - out.shape[-1]))


class TransformerBlock(nn.Module):
    """UnifiedTransformerBlock: shared AdaLN modulation plus a per-block shift table (modulation), or affine layer norms (context
    refiner). Updates x in place."""

    def __init__(self, channels, num_heads, mlp_ratio, modulation):
        super().__init__()
        self.modulation = modulation
        self.norm1 = nn.LayerNorm(channels, eps=1e-6, elementwise_affine=not modulation)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6, elementwise_affine=not modulation)
        self.attn = SelfAttention(channels, num_heads)
        self.mlp = FeedForward(channels, mlp_ratio)
        if modulation:
            self.shift_table = nn.Parameter(torch.zeros(1, 6 * channels))

    def forward(self, x, mod, phases):
        """x (1, tokens, channels), updated in place; mod (1, 6 * channels) or None for the context refiners."""
        rows = x.view(-1, x.shape[-1])
        fc1, fc2 = self.mlp.mlp[0], self.mlp.mlp[2]
        if not self.modulation:
            y = self.attn.attend([layer_norm(x, self.norm1.weight, self.norm1.bias, self.norm1.eps)], phases)
            ops.linear_residual_(self.attn.output, [y.view(rows.shape)], rows)
            del y
            ops.mlp([layer_norm(rows, self.norm2.weight, self.norm2.bias, self.norm2.eps)], fc1, fc2, "gelu", residual=rows)
            return x
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (mod + self.shift_table.type(mod.dtype)).chunk(6, dim=1)
        spans = [(0, rows.shape[0], 0)]
        y = self.attn.attend([ops.norm_modulate([rows], self.norm1, shift_msa, scale_msa, spans).view(x.shape)], phases)
        ops.gated_linear(self.attn.output, [y.view(rows.shape)], rows, gate_msa, spans)
        del y
        ops.mlp([ops.norm_modulate([rows], self.norm2, shift_mlp, scale_mlp, spans)], fc1, fc2, "gelu", residual=rows, gate=gate_mlp[0])
        return x


class CameraMLP(nn.Module):
    def __init__(self, cam_channels, model_channels):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(cam_channels, model_channels), nn.GELU(approximate="tanh"), nn.Linear(model_channels, model_channels))


class TripoSplatFlow(nn.Module):
    def __init__(self, q_token_length, in_channels, cam_channels, out_channels, model_channels, cond_channels, cond2_channels, num_refiner_blocks,
                 num_blocks, num_heads, mlp_ratio):
        super().__init__()
        self.q_token_length, self.in_channels, self.cam_channels, self.model_channels = q_token_length, in_channels, cam_channels, model_channels
        head_dim = model_channels // num_heads
        self.t_embedder = EmbedderMLP(model_channels)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(model_channels, 6 * model_channels))
        self.input_layer = nn.Linear(in_channels, model_channels)
        self.cond_embedder = nn.Linear(cond_channels, model_channels)
        self.cond_embedder2 = nn.Linear(cond2_channels, model_channels)
        # the latent token positions: a scrambled Sobol sequence drawn on the CPU, as upstream
        self.positions = torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=123).draw(q_token_length)[None]
        repo = lambda count: nn.ModuleList([RePo3DRotaryEmbedding(model_channels, num_heads, head_dim) for _ in range(count)])
        self.noise_repo_layers, self.context_repo_layers, self.repo_layers = repo(num_refiner_blocks), repo(num_refiner_blocks), repo(num_blocks)
        blocks = lambda count, modulation: nn.ModuleList([TransformerBlock(model_channels, num_heads, mlp_ratio, modulation) for _ in range(count)])
        self.noise_refiner, self.context_refiner = blocks(num_refiner_blocks, True), blocks(num_refiner_blocks, False)
        self.cam_refiner = CameraMLP(cam_channels, model_channels)
        self.blocks = blocks(num_blocks, True)
        self.shift_table = nn.Parameter(torch.zeros(1, 2, model_channels))
        self.out_layer = nn.Linear(model_channels, out_channels)
        self.cam_out_layer = nn.Linear(model_channels, cam_channels)

    def encode_context(self, feature1, feature2):
        """Image tokens after the context refiners (1, tokens, channels): DINOv3 features feature1 and FLUX.2 VAE latents feature2."""
        dtype = self.shift_table.dtype
        h = self.cond_embedder(feature1.to(dtype)) + self.cond_embedder2(feature2.to(dtype))
        for block, repo in zip(self.context_refiner, self.context_repo_layers):
            h = block(h, None, repo(h))
        return h

    def forward(self, latent, camera, t, contexts):
        """Velocities (latent, camera) for each context of contexts (refined image tokens: conditional, then unconditional), the noisy
        latent (1, tokens, in_channels) and camera (1, 1, cam_channels) FP32 at timestep t (1,) FP32 (times 1000)."""
        dtype = self.shift_table.dtype
        t_emb = self.t_embedder(t)
        t_mod = self.adaLN_modulation(t_emb)
        h_x = self.input_layer(latent.to(dtype)) + position_embedding(self.positions.to(latent.device), self.model_channels).to(dtype)
        for block, repo in zip(self.noise_refiner, self.noise_repo_layers):
            h_x = block(h_x, t_mod, repo(h_x))
        h_cam = self.cam_refiner.mlp(camera.to(dtype))
        hidden = [torch.cat([h_x, context, h_cam], dim=1) for context in contexts]
        del h_x, h_cam
        for block, repo in zip(self.blocks, self.repo_layers):
            for h in hidden:
                block(h, t_mod, repo(h))
            check_abort()
        shift, scale = (self.shift_table + t_emb.unsqueeze(1)).chunk(2, dim=1)
        outputs = []
        for h in hidden:
            h_x = F.layer_norm(h[:, :latent.shape[1]].float(), h.shape[-1:]).type(dtype)
            h_cam = F.layer_norm(h[:, -camera.shape[1]:].float(), h.shape[-1:]).type(dtype)
            outputs.append((self.out_layer(h_x * (1 + scale) + shift), self.cam_out_layer(h_cam * (1 + scale) + shift)))
        return outputs
