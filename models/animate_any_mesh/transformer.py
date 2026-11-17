# Text-to-trajectory rectified-flow transformer of AnimateAnyMesh (DyMeshMMDiT), from JarrentWu1031/AnimateAnyMesh
# networks/transformer.py and networks/util.py (Apache License 2.0, see LICENSE in this folder; it builds on Point-E and CogVideoX).
# Modified by DeepBeepMeep for WanGP: inference only; the frozen CLIP text tower is a separate model (shared clip_vit_large_patch14);
# the fused q/k/v projection is split at load (rows reordered in the WanGP checkpoint) and attended through
# shared.attention_kit; the conditional and unconditional branches of classifier-free guidance run one after the other in each block.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import math

import torch
from torch import nn

from shared import attention_kit
from shared.utils.phase_progress import check_abort


def timestep_embedding(timesteps, dim, max_period=10000):
    """Sinusoidal embedding [cos, sin] of continuous times; the frequencies are computed on the CPU, as upstream."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device="cpu") / half).to(timesteps.device)
    args = timesteps[:, None].to(timesteps.dtype) * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


def fused_split_map(width):
    """mmgp split map of the fused self-attention projection c_qkv (rows [q; k; v] after conversion)."""
    return {"c_qkv": {"mapped_modules": ["to_q", "to_k", "to_v"], "split_sizes": [width] * 3}}


class MLP(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.c_fc, self.gelu, self.c_proj = nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.to_q, self.to_k, self.to_v = nn.Linear(width, width), nn.Linear(width, width), nn.Linear(width, width)
        self.c_proj = nn.Linear(width, width)

    def forward(self, x_list, text_list):
        """Self-attention of the motion tokens and the text tokens joined in this order (upstream concatenates them)."""
        projections = (self.to_q, self.to_k, self.to_v)
        # FP32 checkpoint: SDPA, as upstream (the 16-bit kernels do not take FP32); no q/k norm or RoPE
        return attention_kit.joint_qkv_attention([(x_list, projections), (text_list, projections)], self.heads, self.head_dim, lambda query, key, group: None,
                                                 force_attention="sdpa", out_proj=self.c_proj)


class AdaLayerNormZero(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.silu, self.linear = nn.SiLU(), nn.Linear(width, 6 * width)
        self.norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.linear(self.silu(emb)).chunk(6, dim=1)
        return self.norm(x).mul_(1 + scale_msa[:, None]).add_(shift_msa[:, None]), gate_msa, shift_mlp, scale_mlp, gate_mlp


class CogXAttentionBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.attn, self.mlp = Attention(width, heads), MLP(width)
        self.adaln_x, self.adaln_text = AdaLayerNormZero(width), AdaLayerNormZero(width)
        self.ln_x, self.ln_text = nn.LayerNorm(width), nn.LayerNorm(width)

    def forward(self, x, text, t_emb):
        tokens = x.shape[1]
        norm_x, gate_msa_x, shift_mlp_x, scale_mlp_x, gate_mlp_x = self.adaln_x(x, t_emb)
        norm_text, gate_msa_text, shift_mlp_text, scale_mlp_text, gate_mlp_text = self.adaln_text(text, t_emb)
        attention = self.attn([norm_x], [norm_text])
        del norm_x, norm_text
        x = x + gate_msa_x[:, None] * attention[:, :tokens]
        text = text + gate_msa_text[:, None] * attention[:, tokens:]
        del attention
        x = x + gate_mlp_x[:, None] * self.mlp(self.ln_x(x).mul_(1 + scale_mlp_x[:, None]).add_(shift_mlp_x[:, None]))
        text = text + gate_mlp_text[:, None] * self.mlp(self.ln_text(text).mul_(1 + scale_mlp_text[:, None]).add_(shift_mlp_text[:, None]))
        return x, text


class Backbone(nn.Module):
    def __init__(self, width, layers, heads):
        super().__init__()
        self.width = width
        self.resblocks = nn.ModuleList([CogXAttentionBlock(width, heads) for _ in range(layers)])


class DyMeshMMDiT(nn.Module):
    def __init__(self, input_channels=64, output_channels=64, width=512, layers=12, heads=8, text_channels=768, **kwargs):
        super().__init__()
        self.time_embed = MLP(width)
        self.backbone = Backbone(width, layers, heads)
        self.ln_pre, self.ln_pre_text, self.ln_post = nn.LayerNorm(width), nn.LayerNorm(width), nn.LayerNorm(width)
        self.clip_token_mlp = nn.Linear(text_channels, width)
        self.input_proj, self.output_proj = nn.Linear(input_channels, width), nn.Linear(width, output_channels)

    def forward(self, x, t, text_states, guidance_scale):
        """Guided flow of x (1, tokens, channels) at time t (1,): text_states (1, 77, 768) are the CLIP hidden states of the prompt.
        The unconditional branch (upstream's empty prompt, whose text tokens are masked to zero) is skipped when guidance_scale is 1."""
        t_emb = self.time_embed(timestep_embedding(t, self.backbone.width))
        h = self.ln_pre(self.input_proj(x))
        texts = [self.clip_token_mlp(text_states)]
        if guidance_scale != 1:
            texts.append(torch.zeros_like(texts[0]))
        texts = [self.ln_pre_text(text) for text in texts]
        streams = [h] + [h.clone() for _ in texts[1:]]
        del h
        for block in self.backbone.resblocks:
            for branch, text in enumerate(texts):
                streams[branch], texts[branch] = block(streams[branch], text, t_emb)
            check_abort()
        flows = [self.output_proj(self.ln_post(stream)) for stream in streams]
        return flows[0] if len(flows) == 1 else flows[1] + guidance_scale * (flows[0] - flows[1])
