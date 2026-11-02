# NAF feature upsampler (Neighborhood Attention Filtering, valeoai/NAF, src/model/naf.py and src/layers, Apache License 2.0), inference
# port without NATTEN: queries are image features at the target size (two convolutional encoders, axial RoPE), keys the queries pooled to
# the low-resolution feature grid and values the low-resolution features. NATTEN's dilated neighborhood attention over the nearest-upsampled
# keys and values is, for an upsampling factor equal to its dilation, an attention of each target pixel to the 9 x 9 window of
# low-resolution tokens around its own token (the window kept inside the grid): one batched product per low-resolution cell here.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.utils.phase_progress import check_abort

DIM, HEADS, KERNEL = 256, 4, 9
_ROWS = 4  # low-resolution rows of cells per attention chunk


class EncBlock(nn.Module):
    """GroupNorm, SiLU, convolution, twice (no residual: NAF's encoder setting)."""

    def __init__(self, channels, kernel_size):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, padding_mode="reflect")
        self.norm2 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=kernel_size // 2, padding_mode="reflect")

    def forward(self, x):
        return self.conv2(F.silu(self.norm2(self.conv1(F.silu(self.norm1(x))))))


def _encoder(channels, kernel_size):
    return nn.Sequential(nn.Conv2d(3, channels, kernel_size, padding=kernel_size // 2, padding_mode="reflect"), EncBlock(channels, kernel_size), EncBlock(channels, kernel_size))


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = _encoder(DIM // 2, 1)
        self.sem_encoder = _encoder(DIM // 2, 3)
        self.rope = nn.Module()
        self.rope.register_buffer("periods", torch.zeros(DIM // HEADS // 4))

    def forward(self, image, size):
        """Queries (1, DIM, size, size) of image (1, 3, H, W) in [0, 1]."""
        if image.shape[-2] > 4 * size or image.shape[-1] > 4 * size:
            image = F.interpolate(image, size=(min(image.shape[-2], 4 * size), min(image.shape[-1], 4 * size)), mode="bilinear", align_corners=False)
        x = F.adaptive_avg_pool2d(torch.cat([self.encoder(image), self.sem_encoder(image)], dim=1), (size, size))
        # DINOv3-style axial RoPE (rope_rotate_half): per head, (u, v) coordinates in [-1, 1] at pixel centers, frequencies 2 pi / periods
        coords = (torch.arange(size, dtype=torch.float32, device=x.device) + 0.5) / size * 2 - 1
        coords = torch.stack(torch.meshgrid(coords, coords, indexing="ij"), dim=-1).reshape(-1, 2)
        angles = (2 * math.pi * coords[:, :, None] / self.rope.periods.float()[None, None]).flatten(1).tile(2)
        x = x.view(1, HEADS, DIM // HEADS, size * size).transpose(2, 3)
        first, second = x.chunk(2, dim=-1)
        x = (x * angles.cos() + torch.cat([-second, first], dim=-1) * angles.sin()).to(x.dtype)
        return x.transpose(2, 3).reshape(1, DIM, size, size)


class NAF(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder()

    def forward(self, image, features, size):
        """features (1, C, h, w) upsampled to (1, C, size, size) BF16 guided by image (1, 3, H, W) in [0, 1]: the encoders in the weights'
        dtype, the attention in FP32."""
        queries = self.image_encoder(image.to(self.image_encoder.encoder[0].weight.dtype), size).float()
        h, w = features.shape[-2:]
        keys = F.adaptive_avg_pool2d(queries, (h, w))
        dilation, head_dim, value_dim = size // h, DIM // HEADS, features.shape[1] // HEADS
        q = queries.view(HEADS, head_dim, h, dilation, w, dilation).permute(2, 4, 0, 3, 5, 1).reshape(h, w, HEADS, dilation * dilation, head_dim)
        k = keys.view(HEADS, head_dim, h, w).permute(2, 3, 0, 1)
        v = features.float().view(HEADS, value_dim, h, w).permute(2, 3, 0, 1)
        rows = (torch.arange(h, device=features.device) - KERNEL // 2).clamp(0, h - KERNEL)[:, None] + torch.arange(KERNEL, device=features.device)
        cols = (torch.arange(w, device=features.device) - KERNEL // 2).clamp(0, w - KERNEL)[:, None] + torch.arange(KERNEL, device=features.device)
        out = torch.empty((HEADS, value_dim, h, dilation, w, dilation), dtype=torch.bfloat16, device=features.device)
        scale = head_dim ** -0.5
        for start in range(0, h, _ROWS):
            stop = min(start + _ROWS, h)
            index_r, index_c = rows[start:stop, None, :, None], cols[None, :, None, :]  # (r, 1, K, 1), (1, w, 1, K)
            window_k = k[index_r, index_c].reshape(stop - start, w, KERNEL * KERNEL, HEADS, head_dim).permute(0, 1, 3, 4, 2)
            scores = torch.matmul(q[start:stop], window_k).mul_(scale).softmax(dim=-1)  # (r, w, heads, d^2, K^2)
            del window_k
            window_v = v[index_r, index_c].reshape(stop - start, w, KERNEL * KERNEL, HEADS, value_dim).transpose(2, 3)
            part = torch.matmul(scores, window_v)  # (r, w, heads, d^2, value_dim)
            del scores, window_v
            out[:, :, start:stop] = part.view(stop - start, w, HEADS, dilation, dilation, value_dim).permute(2, 5, 0, 3, 1, 4).to(torch.bfloat16)
            del part
            check_abort()
        return out.view(1, features.shape[1], size, size)
