# Adapted from Alaya-EVOKE (https://github.com/AlayaLab/Evoke, evoke/modules/transformer_evoke.py), Copyright 2025 The Helios Team and The
# HuggingFace Team, Copyright 2026 The Evoke Team, Apache License 2.0. Modified for WanGP.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""EVOKE-Turbo DiT: the Wan2.1-T2V-14B blocks (Helios), denoising the 9 latent frames of a chunk while attending to its history in one
bidirectional self-attention: the history enters as tokens of three tiers (long: 16 latent frames patched 4x8x8, mid: 2 frames 2x4x4,
short: the start image latent, the memory view of the chunk and the previous chunk's last latent, 1x2x2), modulated at timestep 0 and
left out of the cross-attention and of the output head. Weights keep WanGP's Wan names (blocks.N.self_attn.q, ...)."""
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.wan.modules.model import WanLayerNorm, WanSelfAttention, WanT2VCrossAttention
from shared import attention_kit
from shared.kernels import ops
from shared.utils.lora_mapping import convert_lora_keys
from shared.utils.phase_progress import check_abort

DIM, FFN_DIM, HEADS, HEAD_DIM, DEPTH, TEXT_DIM, FREQ_DIM, LATENT_CHANNELS, EPS = 5120, 13824, 40, 128, 40, 4096, 256, 16, 1e-6
ROPE_DIMS, ROPE_THETA = (44, 42, 42), 10000.0
TIERS = {"long": (4, 8, 8), "mid": (2, 4, 4), "short": (1, 2, 2)}  # patch (t, h, w) of the history tiers, in token order
HISTORY_SIZES = (16, 2, 1)  # latent frames of the long, mid and 1x tiers (the last of them is the previous chunk's last latent)


def timestep_embedding(steps):
    """The source's sinusoidal timestep features (diffusers Timesteps, flip_sin_to_cos, no shift): [cos, sin] of steps * exp(-ln(10000) k / 128)."""
    frequencies = torch.exp(-math.log(10000.0) * torch.arange(FREQ_DIM // 2, dtype=torch.float32, device=steps.device) / (FREQ_DIM // 2))
    angles = steps.float()[:, None] * frequencies[None]
    return torch.cat([angles.cos(), angles.sin()], dim=-1)


def rope_grid(frame_indices, height, width, device):
    """(cos, sin), each (len(frame_indices), height, width, 128) FP32, the source's 3D RoPE (t / h / w on 44 / 42 / 42 channels, every
    frequency repeated for its interleaved channel pair) at integer grid positions."""
    grids = torch.meshgrid(torch.as_tensor(frame_indices, device=device, dtype=torch.float32), torch.arange(height, device=device, dtype=torch.float32),
                           torch.arange(width, device=device, dtype=torch.float32), indexing="ij")
    angles = []
    for grid, dim in zip(grids, ROPE_DIMS):
        base = 1.0 / (ROPE_THETA ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device)[: dim // 2] / dim))
        angles.append(torch.einsum("d,thw->thwd", base, grid).repeat_interleave(2, dim=-1))
    angles = torch.cat(angles, dim=-1)
    return angles.cos(), angles.sin()


def _pooled(table, kernel):
    """A (T, H, W, 128) table averaged over kernel (t, h, w) blocks: the source's mid / long tier RoPE (averaged rotations)."""
    table = table.permute(3, 0, 1, 2).unsqueeze(0)
    pad = [(k - s % k) % k for s, k in zip(table.shape[2:], kernel)]
    if any(pad):
        table = F.pad(table, (0, pad[2], 0, pad[1], 0, pad[0]), mode="replicate")
    return F.avg_pool3d(table, kernel, stride=kernel)[0].permute(1, 2, 3, 0)


@dataclass
class TokenPlan:
    """The token layout of one forward: [long | mid | short (kept tokens) | noise]. rope: (cos, sin) (tokens, 128); keep: indices of the
    short tier tokens kept (None: all); history / noise: their token counts; noise_shape: (frames, height, width) of the noise latents."""
    rope: tuple
    keep: torch.Tensor | None
    history: int
    noise: int
    noise_shape: tuple


def token_plan(indices, short_frames, latent_hw, noise_hw, keep, device):
    """TokenPlan of a forward. indices: frame positions {"long", "mid", "short", "noise"}; short_frames: latent frames of the short tier;
    latent_hw: the full latent size (the short tier's, whose token grid the long and mid tables are pooled from); noise_hw: the noise
    latents' size at this pyramid stage (an unscaled grid of its own, as the source without warp_rope_noise_center_align); keep: kept short
    tokens (indices) or None."""
    grid_h, grid_w = latent_hw[0] // 2, latent_hw[1] // 2
    tables = []
    for name, pool in (("long", (4, 4, 4)), ("mid", (2, 2, 2))):
        cos, sin = rope_grid(indices[name], grid_h, grid_w, device)
        tables.append((_pooled(cos, pool).reshape(-1, 128), _pooled(sin, pool).reshape(-1, 128)))
    cos, sin = (table.reshape(-1, 128) for table in rope_grid(indices["short"], grid_h, grid_w, device))
    if keep is not None:
        cos, sin = cos[keep], sin[keep]
    tables.append((cos, sin))
    noise_h, noise_w = noise_hw[0] // 2, noise_hw[1] // 2
    tables.append(tuple(table.reshape(-1, 128) for table in rope_grid(indices["noise"], noise_h, noise_w, device)))
    rope = (torch.cat([cos for cos, _ in tables]).contiguous(), torch.cat([sin for _, sin in tables]).contiguous())
    noise = len(indices["noise"]) * noise_h * noise_w
    return TokenPlan(rope, keep, rope[0].shape[0] - noise, noise, (len(indices["noise"]), *noise_hw))


class EvokeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = WanLayerNorm(DIM, EPS)
        self.self_attn = WanSelfAttention(DIM, HEADS, qk_norm=True, eps=EPS)
        self.norm3 = WanLayerNorm(DIM, EPS, elementwise_affine=True)  # before the cross-attention (the source's norm2)
        self.cross_attn = WanT2VCrossAttention(DIM, HEADS, qk_norm=True, eps=EPS)
        self.norm2 = WanLayerNorm(DIM, EPS)
        self.ffn = nn.Sequential(nn.Linear(DIM, FFN_DIM), nn.GELU(approximate="tanh"), nn.Linear(FFN_DIM, DIM))
        self.modulation = nn.Parameter(torch.empty(1, 6, DIM))

    def forward(self, x, context, e0, spans, history, rope):
        """x (1, tokens, DIM) updated in place; e0 (2, 6, DIM): the timestep projections of the history (row 0, t = 0) and of the noise
        (row 1); spans [(start, stop, row)] of the token rows."""
        shift, scale, gate, c_shift, c_scale, c_gate = (self.modulation.float() + e0.float()).unbind(1)
        rows = x.view(-1, DIM)
        y = ops.norm_modulate([rows], self.norm1, shift, scale, spans).view(x.shape)
        attention = self.self_attn
        attention_kit.qkv_attention([y], attention.q, attention.k, attention.v, HEADS, HEAD_DIM,
                                    lambda query, key, group: attention._norm_rope_(query, key, group, rope), norm_spans_heads=True,
                                    out_proj=attention.o, residual=rows, gate=gate, segments=spans)  # rows += gate * o(attention)
        del y
        noise = x[:, history:]  # the history skips the cross-attention (guidance_cross_attn)
        noise += self.cross_attn([self.norm3(noise)], context, None)
        y = ops.norm_modulate([rows], self.norm2, c_shift, c_scale, spans)
        ops.mlp([y], self.ffn[0], self.ffn[2], "gelu", residual=rows, segments=[(start, stop, c_gate[row]) for start, stop, row in spans])
        return x


class EvokeHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = WanLayerNorm(DIM, EPS)
        self.head = nn.Linear(DIM, LATENT_CHANNELS * 4)
        self.modulation = nn.Parameter(torch.empty(1, 2, DIM))

    def forward(self, x, e):
        """x (tokens, DIM) of the noise tokens, e (DIM,) the noise timestep embedding."""
        shift, scale = (self.modulation.float() + e.float()).unbind(1)
        y = ops.norm_modulate([x], self.norm, shift, scale, [(0, x.shape[0], 0)])
        return self.head(y)


class EvokeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embedding = nn.Conv3d(LATENT_CHANNELS, DIM, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.patch_short = nn.Conv3d(LATENT_CHANNELS, DIM, kernel_size=TIERS["short"], stride=TIERS["short"])
        self.patch_mid = nn.Conv3d(LATENT_CHANNELS, DIM, kernel_size=TIERS["mid"], stride=TIERS["mid"])
        self.patch_long = nn.Conv3d(LATENT_CHANNELS, DIM, kernel_size=TIERS["long"], stride=TIERS["long"])
        self.text_embedding = nn.Sequential(nn.Linear(TEXT_DIM, DIM), nn.GELU(approximate="tanh"), nn.Linear(DIM, DIM))
        self.time_embedding = nn.Sequential(nn.Linear(FREQ_DIM, DIM), nn.SiLU(), nn.Linear(DIM, DIM))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(DIM, DIM * 6))
        self.blocks = nn.ModuleList([EvokeBlock() for _ in range(DEPTH)])
        self.head = EvokeHead()
        self.compute_dtype = torch.bfloat16

    def lock_layers_dtypes(self):
        """The modules the source keeps in FP32 (_keep_in_fp32_modules: time embedder, modulation tables, the affine norm), stored in FP32 in
        the WanGP checkpoint and kept so by MMGP; the rest runs in BF16."""
        for module in [self.time_embedding, *self.time_embedding, self.head, *self.blocks, *(block.norm3 for block in self.blocks)]:  # head / blocks: their modulation
            module._lock_dtype = torch.float32

    def preprocess_loras(self, model_type, state_dict):
        return convert_lora_keys(state_dict, dict(self.named_modules()))

    def _tokens(self, conv, latents, kernel=None):
        if kernel is not None:
            pad = [(k - s % k) % k for s, k in zip(latents.shape[2:], kernel)]
            if any(pad):
                latents = F.pad(latents, (0, pad[2], 0, pad[1], 0, pad[0]), mode="replicate")
        return conv(latents.to(self.compute_dtype)).flatten(2).transpose(1, 2)

    def forward(self, noise, timestep, context, tiers, plan):
        """noise (1, 16, frames, h, w) at the stage's size -> the predicted velocity (FP32, same shape). timestep: the stage's model timestep;
        context: the UMT5 states (1, 512, 4096) zero-padded; tiers: {"long", "mid", "short"} history latents (1, 16, T, H, W) at the full
        latent size; plan: token_plan's layout."""
        context = self.text_embedding(context.to(self.compute_dtype))
        parts = [self._tokens(self.patch_long, tiers["long"], TIERS["long"]), self._tokens(self.patch_mid, tiers["mid"], TIERS["mid"])]
        short = self._tokens(self.patch_short, tiers["short"])
        parts.append(short if plan.keep is None else short[:, plan.keep])
        parts.append(self._tokens(self.patch_embedding, noise))
        x = torch.cat(parts, dim=1)
        del parts, short
        steps = torch.stack([torch.zeros_like(timestep), timestep]).float().reshape(2)
        e = self.time_embedding(timestep_embedding(steps))  # FP32 (locked)
        e = e.to(self.compute_dtype)  # the source casts the embedder's output to the transformer dtype
        e0 = self.time_projection(e).unflatten(1, (6, DIM))
        spans = [(0, plan.history, 0), (plan.history, plan.history + plan.noise, 1)]
        for block in self.blocks:
            block(x, context, e0, spans, plan.history, plan.rope)
            check_abort()
        out = self.head(x[0, plan.history:], e[1])
        del x
        frames, height, width = plan.noise_shape
        out = out.float().view(1, frames, height // 2, width // 2, 1, 2, 2, LATENT_CHANNELS)
        return out.permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(1, LATENT_CHANNELS, frames, height, width)
