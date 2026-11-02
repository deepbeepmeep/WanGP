# Hi3D v1.0 image conditioning, inference port of hi3d-official/Hi3Dv1.0 (sparc3d/models/conditioners/image.py and
# sparc3d/utils/image.py: Hi3D Open License Agreement 1.0; see LICENSE and NOTICE in this folder): the object is cropped to the square
# around its alpha, composited on white with a 1/1.2 margin, then encoded by DINOv2-large at 756 px (2917 tokens) and CLIP ViT-L/14 at
# 224 px (its pooled embedding repeated on every token), both L2-normalized: a (1, 2917, 2048) conditioning shared by every pass.
# The CLIP vision tower follows the architecture and parameter names of Hugging Face transformers' CLIPVisionModel (Apache-2.0,
# Copyright OpenAI and The HuggingFace Inc. team), reimplemented for WanGP with the shared attention kit; DINOv2 is WanGP's port in
# models/hunyuan3d/dinov2.py. Modified by DeepBeepMeep for WanGP: 16-bit encoders, rembg u2net for images without a usable alpha.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps

from shared import attention_kit

SUBJECT_PADDING = 1.2
DINO_SIZE, CLIP_SIZE = 756, 224
DINO_MEAN, DINO_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


def has_alpha(image):
    return image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info)


def prepare_image(image):
    """Upstream prepare_image: the square around the opaque subject (overhang transparent), composited on white, centred on a square
    1.2 times larger. An image without transparency keeps its whole frame."""
    rgba = ImageOps.exif_transpose(image).convert("RGBA")
    bbox = rgba.getchannel("A").getbbox()
    if bbox is not None:
        left, upper, right, lower = bbox
        side = max(right - left, lower - upper)
        left -= (side - (right - left)) // 2
        upper -= (side - (lower - upper)) // 2
        rgba = rgba.crop((left, upper, left + side, upper + side))
    white = Image.new("RGB", rgba.size, (255, 255, 255))
    white.paste(rgba, (0, 0), rgba)
    side = int(max(white.size) * SUBJECT_PADDING)
    padded = Image.new("RGB", (side, side), (255, 255, 255))
    padded.paste(white, ((side - white.width) // 2, (side - white.height) // 2))
    return padded


def _pixels(image, device):
    return torch.from_numpy(np.asarray(image.convert("RGB"), dtype=np.float32).copy()).permute(2, 0, 1).unsqueeze(0).to(device).div_(255.0)


def _normalize(pixels, mean, std):
    return (pixels - torch.tensor(mean, device=pixels.device).view(1, 3, 1, 1)) / torch.tensor(std, device=pixels.device).view(1, 3, 1, 1)


def dino_pixels(image, device):
    """The prepared image resized bilinearly to 756 px on the GPU (FP32), ImageNet-normalized."""
    return _normalize(F.interpolate(_pixels(image, device), size=(DINO_SIZE, DINO_SIZE), mode="bilinear", align_corners=False), DINO_MEAN, DINO_STD)


def clip_pixels(image, device):
    """The prepared image resized to 224 px with Lanczos (PIL), CLIP-normalized."""
    return _normalize(_pixels(image.convert("RGB").resize((CLIP_SIZE, CLIP_SIZE), resample=Image.Resampling.LANCZOS), device), CLIP_MEAN, CLIP_STD)


def condition_tokens(dino_hidden, clip_pooled):
    """(1, 2917, 2048) FP32: the L2-normalized DINOv2 tokens and the L2-normalized CLIP embedding repeated on each."""
    dino = F.normalize(dino_hidden.float(), p=2, dim=-1)
    clip = F.normalize(clip_pooled.float(), p=2, dim=-1)
    return torch.cat((dino, clip[:, None, :].expand(-1, dino.shape[1], -1)), dim=-1)


class CLIPEncoderLayer(nn.Module):
    def __init__(self, hidden_size, heads, intermediate_size, eps):
        super().__init__()
        self.heads = heads
        self.self_attn = nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(self.self_attn, name, nn.Linear(hidden_size, hidden_size))
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=eps)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.mlp.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, x):
        attn = self.self_attn
        out = attention_kit.qkv_attention([self.layer_norm1(x)], attn.q_proj, attn.k_proj, attn.v_proj, self.heads, x.shape[-1] // self.heads,
                                          lambda query, key, group: None)
        x = attn.out_proj(out.flatten(2)).add_(x)
        del out
        h = self.mlp.fc1(self.layer_norm2(x))
        return self.mlp.fc2(h.mul_(torch.sigmoid(1.702 * h))).add_(x)  # quick_gelu


class CLIPVisionModel(nn.Module):
    """CLIP ViT vision tower: pooler output = post_layernorm(class token)."""

    def __init__(self, hidden_size=1024, num_hidden_layers=24, num_attention_heads=16, intermediate_size=4096, image_size=224, patch_size=14,
                 layer_norm_eps=1e-5):
        super().__init__()
        self.vision_model = vision = nn.Module()
        vision.embeddings = nn.Module()
        vision.embeddings.class_embedding = nn.Parameter(torch.empty(hidden_size))
        vision.embeddings.patch_embedding = nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size, bias=False)
        vision.embeddings.position_embedding = nn.Embedding((image_size // patch_size) ** 2 + 1, hidden_size)
        vision.pre_layrnorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        vision.encoder = nn.Module()
        vision.encoder.layers = nn.ModuleList([CLIPEncoderLayer(hidden_size, num_attention_heads, intermediate_size, layer_norm_eps)
                                               for _ in range(num_hidden_layers)])
        vision.post_layernorm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def forward(self, pixel_values):
        vision = self.vision_model
        embeddings = vision.embeddings
        patch = embeddings.patch_embedding
        x = patch(pixel_values.to(patch.weight.dtype)).flatten(2).transpose(1, 2)
        x = torch.cat([embeddings.class_embedding.view(1, 1, -1).expand(x.shape[0], -1, -1).to(x.dtype), x], dim=1)
        x = vision.pre_layrnorm(x + embeddings.position_embedding.weight.unsqueeze(0).to(x.dtype))
        for layer in vision.encoder.layers:
            x = layer(x)
        return vision.post_layernorm(x[:, 0])
