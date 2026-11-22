# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
# Architecture after the Inkling 2 vision classes of Hugging Face Transformers (pull request 51050, Apache License 2.0,
# Copyright 2026 the HuggingFace Team), rewritten on the checkpoint's own tensor names.
"""Vision tower of Inkling 2: image cells of 40x40 pixels to the language model's image-token embeddings.

A ViT over 20-pixel patches (class token in front of each image, interpolated learned positions, 2D axial RoPE, layer scale), then the
2x2 patches of a cell are concatenated and projected to the language model's hidden size. Its weights are a file of their own (the
checkpoint's `model.visual.*` tensors, apart from the language model's file); state-dict keys are those names after `model.visual.`."""
from __future__ import annotations

import json
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

VISUAL_PREFIX = "model.visual."
IMAGE_TOKEN, IMAGE_TOKEN_ID = "<|unused_200054|>", 200054


@dataclass(frozen=True)
class Inkling2VisionConfig:
    hidden_size: int = 1024
    num_layers: int = 23
    num_heads: int = 16
    mlp_ratio: float = 4.0
    patch_size: int = 20
    merge_size: int = 2
    posemb_grid_size: int = 32
    projection_hidden_dim: int = 4096
    text_hidden_size: int = 2304
    layer_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    num_channels: int = 3

    @classmethod
    def from_json(cls, values: dict) -> "Inkling2VisionConfig":
        """`values`: the `vision_config` entry of the checkpoint's config.json."""
        if values["projector_bias"]:
            raise ValueError("Inkling 2 vision projector with biases is not supported.")
        return cls(hidden_size=values["hidden_size"], num_layers=values["num_hidden_layers"], num_heads=values["num_attention_heads"], mlp_ratio=values["mlp_ratio"],
                   patch_size=values["native_patch_size"], merge_size=values["adapter_merge_size"], posemb_grid_size=values["posemb_grid_size"],
                   projection_hidden_dim=values["projection_hidden_dim"], text_hidden_size=values["text_hidden_size"], layer_norm_eps=values["layer_norm_eps"])

    @property
    def cell_size(self) -> int:
        """Pixels per side of the square one image token covers."""
        return self.patch_size * self.merge_size


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """RoPE over interleaved channel pairs, in FP32. x [tokens, heads, head_dim]; cos, sin [tokens, 1, head_dim / 2]."""
    a, b = x.float().unflatten(-1, (-1, 2)).unbind(-1)
    return torch.stack((a * cos - b * sin, b * cos + a * sin), -1).flatten(-2).to(x.dtype)


def _axis_taps(index: torch.Tensor, size: int, side: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Bilinear taps (align_corners=False, border clamped) of an axis of `size` positions into a table of `side` entries."""
    src = (index.to(torch.float32) + 0.5) * side / size - 0.5
    floor = torch.floor(src)
    offsets = torch.arange(0, 2, device=index.device)
    taps = (floor.long()[:, None] + offsets).clamp(0, side - 1)
    return taps, (1 - (src[:, None] - floor[:, None] - offsets).abs()).clamp(min=0)


class Inkling2VisionAttention(nn.Module):
    def __init__(self, config: Inkling2VisionConfig):
        super().__init__()
        self.num_heads, self.scale = config.num_heads, (config.hidden_size // config.num_heads) ** -0.5
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = (nn.Linear(config.hidden_size, config.hidden_size) for _ in range(4))

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        q, k, v = (proj(x).unflatten(-1, (self.num_heads, -1)) for proj in (self.q_proj, self.k_proj, self.v_proj))
        q, k, v = _rotate(q, cos, sin).transpose(0, 1), _rotate(k, cos, sin).transpose(0, 1), v.transpose(0, 1)
        if len(lengths) == 1:
            x = F.scaled_dot_product_attention(q[None], k[None], v[None], scale=self.scale)[0]
        else:  # each image attends to itself only
            x = torch.cat([F.scaled_dot_product_attention(qi[None], ki[None], vi[None], scale=self.scale)[0] for qi, ki, vi in zip(q.split(lengths, 1), k.split(lengths, 1), v.split(lengths, 1))], 1)
        return self.out_proj(x.transpose(0, 1).flatten(1))


class Inkling2VisionMLP(nn.Module):
    def __init__(self, config: Inkling2VisionConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.hidden_size, int(config.hidden_size * config.mlp_ratio))
        self.c_proj = nn.Linear(int(config.hidden_size * config.mlp_ratio), config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(x)))


class Inkling2LayerScale(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.gamma * x


class Inkling2VisionBlock(nn.Module):
    def __init__(self, config: Inkling2VisionConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.self_attn = Inkling2VisionAttention(config)
        self.ls_1 = Inkling2LayerScale(config.hidden_size)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.mlp = Inkling2VisionMLP(config)
        self.ls_2 = Inkling2LayerScale(config.hidden_size)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        x = x + self.ls_1(self.self_attn(self.ln_1(x), cos, sin, lengths))
        return x + self.ls_2(self.mlp(self.ln_2(x)))


class Inkling2VisionEmbeddings(nn.Module):
    def __init__(self, config: Inkling2VisionConfig):
        super().__init__()
        self.patch_size, self.merge_size, self.grid_size = config.patch_size, config.merge_size, config.posemb_grid_size
        self.class_embedding = nn.Parameter(torch.zeros(config.hidden_size))
        self.cls_position_embedding = nn.Parameter(torch.zeros(config.hidden_size))
        self.patch_embedding = nn.Linear(config.num_channels * config.patch_size**2, config.hidden_size, bias=False)
        self.position_embedding = nn.Parameter(torch.zeros(config.posemb_grid_size**2, config.hidden_size))

    def forward(self, pixel_patches: torch.Tensor, grids: list[tuple[int, int]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sequence [class token, patches] of each image, the sequence index of every patch and its 1-based (row, column)."""
        merge, patch, side, device = self.merge_size, self.patch_size, self.grid_size, pixel_patches.device
        x = pixel_patches.to(self.patch_embedding.weight.dtype).unflatten(1, (merge, patch)).unflatten(3, (merge, patch)).transpose(2, 3)
        x = self.patch_embedding(x.flatten(3).flatten(0, 2))
        positions, indices, weights = [], [], []
        for rows, columns in grids:  # patches in the order of the cells (row-major), the merge x merge patches of a cell row-major
            row = torch.arange(rows * merge, device=device).view(rows, 1, merge, 1).expand(rows, columns, merge, merge).flatten()
            column = torch.arange(columns * merge, device=device).view(1, columns, 1, merge).expand(rows, columns, merge, merge).flatten()
            (row_taps, row_weights), (column_taps, column_weights) = _axis_taps(row, rows * merge, side), _axis_taps(column, columns * merge, side)
            indices.append((row_taps[:, :, None] * side + column_taps[:, None, :]).flatten(1))
            weights.append((row_weights[:, :, None] * column_weights[:, None, :]).flatten(1))
            positions.append(torch.stack((row, column), -1) + 1)
        x = x + (self.position_embedding[torch.cat(indices)] * torch.cat(weights)[:, :, None]).sum(1).to(x.dtype)  # FP32 mix of the learned grid
        counts = torch.tensor([rows * columns * merge * merge for rows, columns in grids], device=device)
        patch_index = torch.arange(x.shape[0], device=device) + torch.repeat_interleave(torch.arange(1, len(grids) + 1, device=device), counts)
        sequence = (self.class_embedding + self.cls_position_embedding).to(x.dtype).repeat(x.shape[0] + len(grids), 1)
        sequence[patch_index] = x
        return sequence, patch_index, torch.cat(positions)


class Inkling2VisionBackbone(nn.Module):
    def __init__(self, config: Inkling2VisionConfig):
        super().__init__()
        self.embeddings = Inkling2VisionEmbeddings(config)
        self.ln_pre = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layers = nn.ModuleList([Inkling2VisionBlock(config) for _ in range(config.num_layers)])


class Inkling2VisionTower(nn.Module):
    """forward(pixel_patches, grids) -> [image tokens, text_hidden_size] in the weights' dtype, on pixel_patches' device.

    pixel_patches: [image tokens, 40, 40, 3] cells in [-1, 1] (image_processing.preprocess_images), all images one after the other, cells
    row-major; grids: (rows, columns) in cells per image. One output row per cell, in the same order: the order of the image placeholder
    tokens in the prompt. Images in one call do not see each other: the result of an image does not depend on its neighbours."""

    def __init__(self, config: Inkling2VisionConfig):
        super().__init__()
        self.config = config
        self.backbone = Inkling2VisionBackbone(config)
        self.proj_in = nn.Linear(config.merge_size**2 * config.hidden_size, config.projection_hidden_dim, bias=False)
        self.proj_out = nn.Linear(config.projection_hidden_dim, config.text_hidden_size, bias=False)

    @property
    def blocks(self) -> nn.ModuleList:
        """The transformer blocks, under the name WanGP's vision progress and cancellation hooks use."""
        return self.backbone.layers

    def forward(self, pixel_patches: torch.Tensor, grids) -> torch.Tensor:
        config, device = self.config, pixel_patches.device
        grids = [(int(rows), int(columns)) for rows, columns in (grids.tolist() if torch.is_tensor(grids) else grids)]
        if sum(rows * columns for rows, columns in grids) != pixel_patches.shape[0]:
            raise ValueError(f"Inkling 2 vision: {pixel_patches.shape[0]} cells for grids {grids}.")
        x, patch_index, positions = self.backbone.embeddings(pixel_patches, grids)
        x = self.backbone.ln_pre(x)
        # axial RoPE: the first half of the channel pairs turns with the column, the second half with the row; class tokens at (0, 0)
        quarter = config.hidden_size // config.num_heads // 2
        inv_freq = (1.0 / (config.rope_theta ** (torch.arange(0, quarter, 2, dtype=torch.float32, device="cpu") / quarter))).to(device)
        angles = positions.new_zeros(x.shape[0], 2).index_copy(0, patch_index, positions)[:, :, None].float() * inv_freq
        angles = torch.cat((angles[:, 1], angles[:, 0]), -1)[:, None]
        cos, sin = angles.cos(), angles.sin()
        lengths = [rows * columns * config.merge_size**2 + 1 for rows, columns in grids]
        for block in self.backbone.layers:
            x = block(x, cos, sin, lengths)
        return self.proj_out(F.gelu(self.proj_in(x[patch_index].unflatten(0, (-1, config.merge_size**2)).flatten(1))))


def visual_tensors(state_dict: dict) -> dict:
    """The tensors of the vision file (stored under the checkpoint's `model.visual.` names) under the tower's names (MMGP preprocess_sd)."""
    return {name[len(VISUAL_PREFIX):]: tensor for name, tensor in state_dict.items()}


def load_inkling2_vision_tower(checkpoint_path: str, config_path: str) -> Inkling2VisionTower:
    """The tower on the CPU with its vision file's BF16 tensors mapped read-only, ready for an MMGP pipe as `vision_tower_model`."""
    from mmgp import offload
    with open(config_path, "r", encoding="utf-8") as reader:
        config = Inkling2VisionConfig.from_json(json.load(reader)["vision_config"])
    with torch.device("meta"):
        tower = Inkling2VisionTower(config)
    offload.load_model_data(tower, checkpoint_path, preprocess_sd=visual_tensors, writable_tensors=False, default_dtype=torch.bfloat16)
    return tower.eval()
