# WorldMirror 2.0 visual geometry transformer, ported from Tencent HY-World 2.0 (hyworld2/worldrecon/hyworldmirror/models:
# models/visual_transformer.py, layers/{block,attention,mlp,layer_scale,norm_rope,patch_embed,vision_transformer}.py; the DINOv2 layers
# derive from facebookresearch/dinov2, Apache-2.0). Copyright (C) 2026 Tencent, licensed under the Tencent HY-WORLD 2.0 Community License
# Agreement (LICENSE in this folder). Modified by DeepBeepMeep for WanGP (2026-10-08): inference on one GPU only (no sequence parallel,
# FSDP, training or prior-conditioning paths), q/k/v split at load and attention through shared.attention_kit, RoPE tables computed once
# per forward, in-place residuals and chunked MLPs, and the upstream BF16 autocast made explicit: matmul and convolution weights in BF16,
# norms, layer scales, learned tokens and the residual stream in FP32.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import math

import torch
import torch.nn.functional as F
from torch import nn

from shared import attention_kit

COMPUTE_DTYPE = torch.bfloat16
RESNET_MEAN = (0.485, 0.456, 0.406)
RESNET_STD = (0.229, 0.224, 0.225)
ROW_CHUNK = 16384  # tokens per residual / MLP chunk: bounds the 4x MLP activation, rows are independent


def qkv_split_map(dim):
    """mmgp split map of the fused upstream `qkv` projections into the q / k / v linears of the attention kit."""
    return {"qkv": {"mapped_modules": ["q", "k", "v"], "split_sizes": [dim] * 3}}


def lock_fp32(modules):
    """Parameters that stay FP32 (MMGP _lock_dtype): upstream's autocast runs these modules in FP32."""
    for module in modules:
        module._lock_dtype = torch.float32


def rope_periods(head_dim, base):
    """NormalizedRotaryPositionEmbedding2D periods, computed on the CPU like the upstream buffer (FP32)."""
    return base ** (2 * torch.arange(head_dim // 4, device="cpu") / (head_dim // 2))


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class LayerScale(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(dim))


class Mlp(nn.Module):
    def __init__(self, dim, hidden, out=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, out or dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Attention(nn.Module):
    def __init__(self, dim, heads, qk_norm):
        super().__init__()
        self.heads, self.head_dim = heads, dim // heads
        self.q, self.k, self.v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.q_norm = nn.LayerNorm(self.head_dim) if qk_norm else None
        self.k_norm = nn.LayerNorm(self.head_dim) if qk_norm else None
        self.proj = nn.Linear(dim, dim)

    def forward(self, x_list, rope=None, force_attention=None):
        """x_list: the normalized input, handed off. rope: (cos, sin) tables (tokens, head_dim) of the q/k dtype, indexed by token."""
        def norm_rows(norm):
            def rows(part, first, last):
                # upstream: LayerNorm of the BF16 projection in FP32 (autocast), cast back, then q * cos + rotate_half(q) * sin in BF16
                y = F.layer_norm(part.float(), (self.head_dim,), norm.weight, norm.bias, norm.eps).to(part.dtype)
                return y * rope[0][first:last, None] + _rotate_half(y) * rope[1][first:last, None]
            return rows

        def norm_rope(query, key, group):
            for tensor, norm in ((query, self.q_norm), (key, self.k_norm)):
                if tensor is not None and norm is not None:
                    attention_kit.rows_(tensor, norm_rows(norm))

        return attention_kit.qkv_attention(x_list, self.q, self.k, self.v, self.heads, self.head_dim, norm_rope, force_attention=force_attention, out_proj=self.proj)


class Block(nn.Module):
    """Pre-norm transformer block with layer scales. The FP32 residual x is updated in place; the attention and the MLP read their
    normalized input in compute_dtype (BF16 in the trunk as upstream's autocast, FP32 in the camera head)."""

    def __init__(self, dim, heads, qk_norm=False, eps=1e-5, compute_dtype=COMPUTE_DTYPE):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = Attention(dim, heads, qk_norm)
        self.ls1 = LayerScale(dim)
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = Mlp(dim, 4 * dim)
        self.ls2 = LayerScale(dim)

    def forward(self, x, rope=None, force_attention=None):
        out = self.attn([self.norm1(x).to(self.compute_dtype)], rope, force_attention).view(-1, x.shape[-1])
        rows = x.view(-1, x.shape[-1])
        for start in range(0, len(rows), ROW_CHUNK):
            rows[start:start + ROW_CHUNK].add_(out[start:start + ROW_CHUNK] * self.ls1.gamma)
        del out
        for start in range(0, len(rows), ROW_CHUNK):
            part = rows[start:start + ROW_CHUNK]
            part.add_(self.mlp(self.norm2(part).to(self.compute_dtype)) * self.ls2.gamma)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, patch_size, dim):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)


class DinoViT(nn.Module):
    """DINOv2 ViT-L/14 with 4 registers (upstream vision_transformer.vit_large): the normalized patch tokens of each view."""

    def __init__(self, dim, depth, heads, patch_size, img_size, registers):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(patch_size, dim)
        self.cls_token = nn.Parameter(torch.empty(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.empty(1, (img_size // patch_size) ** 2 + 1, dim))
        self.register_tokens = nn.Parameter(torch.empty(1, registers, dim))
        self.mask_token = nn.Parameter(torch.empty(1, dim))  # training only
        self.blocks = nn.ModuleList([Block(dim, heads, eps=1e-6) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)

    def position_table(self, height, width):
        """interpolate_pos_encoding with interpolate_offset 0: the 37x37 table resized by antialiased bicubic interpolation in FP32."""
        grid = self.pos_embed.shape[1] - 1
        if (height // self.patch_size) * (width // self.patch_size) == grid and height == width:
            return self.pos_embed
        side, table = math.isqrt(grid), self.pos_embed.float()
        patch = F.interpolate(table[:, 1:].reshape(1, side, side, -1).permute(0, 3, 1, 2), size=(height // self.patch_size, width // self.patch_size),
                              mode="bicubic", antialias=True)
        return torch.cat((table[:, :1], patch.permute(0, 2, 3, 1).reshape(1, -1, table.shape[-1])), dim=1)

    def forward(self, images, progress):
        """images: normalized FP32 (views, 3, H, W) -> FP32 (views, patches, dim)."""
        x = self.patch_embed.proj(images.to(COMPUTE_DTYPE)).flatten(2).transpose(1, 2)
        x = torch.cat((self.cls_token.expand(len(x), -1, -1), x), dim=1)  # FP32 by type promotion, as under the upstream autocast
        x += self.position_table(*images.shape[-2:])
        x = torch.cat((x[:, :1], self.register_tokens.expand(len(x), -1, -1), x[:, 1:]), dim=1)
        for block in self.blocks:
            block(x)
            progress.advance()
        return self.norm(x[:, 1 + self.register_tokens.shape[1]:])


class VisualGeometryTransformer(nn.Module):
    """Alternating frame / global attention over the tokens of all views (VGGT style). Returns the concatenated frame and global tokens
    (views, tokens, 2 * dim) FP32 of the intermediate layers; each view's tokens are camera, 4 registers, the (unused here) pose and ray
    condition slots, then its patches."""

    def __init__(self, embed_dim, depth, num_heads, mlp_ratio, num_register_tokens, patch_size, img_size, rope_base, intermediate_layers):
        super().__init__()
        assert mlp_ratio == 4.0
        self.patch_size, self.head_dim, self.rope_base = patch_size, embed_dim // num_heads, rope_base
        self.intermediate_layers = intermediate_layers
        self.patch_start_idx = 1 + num_register_tokens + 2
        self.patch_embed = DinoViT(embed_dim, depth, num_heads, patch_size, img_size, num_register_tokens)
        self.frame_blocks = nn.ModuleList([Block(embed_dim, num_heads, qk_norm=True) for _ in range(depth)])
        self.global_blocks = nn.ModuleList([Block(embed_dim, num_heads, qk_norm=True) for _ in range(depth)])
        self.cam_token = nn.Parameter(torch.empty(1, 2, 1, embed_dim))
        self.reg_token = nn.Parameter(torch.empty(1, 2, num_register_tokens, embed_dim))

    @property
    def layer_count(self):
        return len(self.patch_embed.blocks) + len(self.frame_blocks)

    def lock_fp32_modules(self):
        """The learned tokens, norms and layer scales stay FP32; the roots are also locked so that MMGP keeps their FP32 inputs."""
        lock_fp32([self, self.patch_embed] + [m for m in self.modules() if isinstance(m, (nn.LayerNorm, LayerScale))])

    def rope_tables(self, patches_h, patches_w, device):
        """NormalizedRotaryPositionEmbedding2D ("separate" coordinates) of one view's tokens, in BF16 like upstream (the q/k dtype): the
        special tokens sit at cell (0, 0) and patch (i, j) at cell (i + 1, j + 1) of a (patches_h + 1) x (patches_w + 1) grid."""
        height, width = patches_h + 1, patches_w + 1
        dd = dict(device=device, dtype=COMPUTE_DTYPE)
        coords = torch.stack(torch.meshgrid(torch.arange(0.5, height, **dd) / height, torch.arange(0.5, width, **dd) / width, indexing="ij"), dim=-1)
        coords = 2.0 * coords.flatten(0, 1) - 1.0
        periods = rope_periods(self.head_dim, self.rope_base).to(**dd)
        angles = ((2 * math.pi * coords[:, :, None]) / periods[None, None, :]).flatten(1, 2)
        angles = torch.cat((angles, angles), dim=-1)
        patch_cells = (torch.arange(1, height, device=device)[:, None] * width + torch.arange(1, width, device=device)).flatten()
        index = torch.cat([torch.zeros(self.patch_start_idx, dtype=torch.long, device=device), patch_cells])
        return torch.cos(angles)[index], torch.sin(angles)[index]

    def forward(self, images, progress):
        """images: (views, 3, H, W) FP32 in [0, 1], H and W multiples of the patch size."""
        views, _, height, width = images.shape
        mean, std = (torch.tensor(value, dtype=torch.float32, device=images.device).view(1, 3, 1, 1) for value in (RESNET_MEAN, RESNET_STD))
        patches = self.patch_embed((images - mean) / std, progress)
        dim = patches.shape[-1]

        def per_view(token):  # the first view uses token[:, 0], the others token[:, 1]
            return torch.cat([token[:, :1], token[:, 1:].expand(1, views - 1, *token.shape[2:])], dim=1).reshape(views, *token.shape[2:])

        conditions = patches.new_zeros(views, 2, dim)  # pose and ray tokens without priors
        x = torch.cat([per_view(self.cam_token), per_view(self.reg_token), conditions, patches], dim=1)
        del patches, conditions
        tokens = x.shape[1]
        cos, sin = self.rope_tables(height // self.patch_size, width // self.patch_size, images.device)
        frame_rope, global_rope = (cos, sin), (cos.repeat(views, 1), sin.repeat(views, 1))
        outputs = []
        for index, (frame_block, global_block) in enumerate(zip(self.frame_blocks, self.global_blocks)):
            frame_block(x, frame_rope)
            if index in self.intermediate_layers:
                outputs.append(x.new_empty(views, tokens, 2 * dim))
                outputs[-1][..., :dim].copy_(x)
            global_block(x.view(1, views * tokens, dim), global_rope)
            if index in self.intermediate_layers:
                outputs[-1][..., dim:].copy_(x)
            progress.advance()
        return outputs
