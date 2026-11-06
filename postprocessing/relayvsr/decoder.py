# Adapted from RelayVSR (https://github.com/kopperx/RelayVSR, Apache-2.0, see LICENSE in this folder): its decoder ("dual-memory video
# transformer"), inference only. Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""RelayVSR's decoder: each low-resolution frame encoded to tokens at the keyframe latents' resolution (a quarter of its own for the native x4
checkpoint, an eighth for the x2 one), ten transformer blocks attending the two keyframe latents of its window (each keyframe primed once
through the same blocks) and the previous frame, then a bicubic upscale of the frame plus the predicted residual (16 x 16 pixels per token). Keys and values: per-head RMSNorm; RoPE on channel pairs (16 temporal, 24 + 24 spatial channels), queries, conditions
and the current frame at temporal position 0, the previous frame at -1."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared import attention_kit
from shared.kernels import ops as kernels

from .kernels import centered_window_attention

DIM, DEPTH, HEADS, KV_HEADS, HEAD_DIM = 512, 10, 8, 2, 64
HIDDEN = math.ceil(DIM * 8 / 3 / 64) * 64  # 1408
ROPE_DIMS = (16, 24, 24)  # temporal, rows, columns
CONDITION_WINDOW, VIDEO_WINDOW = 48, 24  # tokens around each query, conditions and video frames
TOKEN_PIXELS = 16  # upscaled pixels per token, side
EPS = 1e-5


def rope_table(height, width, device):
    """(1, height * width, 1, 32, 2) cos / sin of the pairs at temporal position 0 (the temporal pairs unrotated), FP32 from float64 angles."""
    angles = []
    for dim, positions in zip(ROPE_DIMS, (torch.zeros(1, dtype=torch.float64), torch.arange(height, dtype=torch.float64), torch.arange(width, dtype=torch.float64))):
        angles.append(torch.outer(positions, 1.0 / 10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)))
    t, h, w = angles
    grid = torch.cat([t[:, None, None].expand(1, height, width, -1), h[None, :, None].expand(1, height, width, -1), w[None, None].expand(1, height, width, -1)], dim=-1)
    grid = grid.reshape(1, height * width, 1, -1)
    return torch.stack((grid.cos(), grid.sin()), dim=-1).to(device=device, dtype=torch.float32)


def previous_frame_rotation(device):
    """cos / sin (8,) of the temporal pairs rotating a key from position 0 to -1."""
    angles = -1.0 / 10000.0 ** (torch.arange(0, ROPE_DIMS[0], 2, dtype=torch.float64) / ROPE_DIMS[0])
    return angles.cos().to(device=device, dtype=torch.float32), angles.sin().to(device=device, dtype=torch.float32)


def _rope_rows_(x, rope):
    """x (1, tokens, heads, head_dim) rotated in place by rope (1, tokens, 1, pairs, 2), in FP32 with one rounding."""
    attention_kit.rows_(x, lambda part, first, last: kernels.paired_rope_(part.float(), rope[:, first:last, ..., 0], rope[:, first:last, ..., 1], interleaved=True).to(x.dtype))


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        kv_dim = KV_HEADS * HEAD_DIM
        self.query = nn.Linear(DIM, DIM)
        self.key = nn.Linear(DIM, kv_dim)
        self.value = nn.Linear(DIM, kv_dim)
        self.condition_key = nn.Linear(DIM, kv_dim)
        self.condition_value = nn.Linear(DIM, kv_dim)
        self.key_norm = nn.RMSNorm(HEAD_DIM, eps=EPS)
        self.value_norm = nn.RMSNorm(HEAD_DIM, eps=EPS)
        self.output = nn.Linear(DIM, DIM)

    def _norm_rope(self, rope):
        def norm_rope(query, key, group):
            if key is not None:
                kernels.interleaved_norm_rope_(None, key, rope, None, self.key_norm)
            if query is not None:
                _rope_rows_(query, rope)
        return norm_rope

    def _norm_v(self, value, group):
        attention_kit.rows_(value, lambda part, first, last: F.rms_norm(part, (HEAD_DIM,), self.value_norm.weight, EPS))

    def attend(self, x_list, rope, attention_fn, condition):
        """The attention output (1, tokens, DIM) of x (handed off): the queries and the keys / values of the condition or video projections."""
        key, value = (self.condition_key, self.condition_value) if condition else (self.key, self.value)
        out = attention_kit.qkv_attention(x_list, self.query, key, value, HEADS, HEAD_DIM, self._norm_rope(rope), kv_heads=KV_HEADS, norm_v=self._norm_v, attention_fn=attention_fn)
        return self.output(out.flatten(2))


class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Linear(DIM, 2 * HIDDEN, bias=False)
        self.down = nn.Linear(HIDDEN, DIM, bias=False)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention_norm = nn.RMSNorm(DIM, eps=EPS)
        self.attention = Attention()
        self.feed_forward_norm = nn.RMSNorm(DIM, eps=EPS)
        self.feed_forward = SwiGLU()

    def forward(self, x, rope, attention_fn, condition=False):
        """x (1, tokens, DIM) updated in place and returned."""
        x += self.attention.attend([self.attention_norm(x)], rope, attention_fn, condition)
        kernels.mlp([self.feed_forward_norm(x)], self.feed_forward.gate_up_proj, self.feed_forward.down, "swiglu", residual=x)
        return x


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList(Block() for _ in range(DEPTH))


class LowResolutionEncoder(nn.Module):
    def __init__(self, scale, channels=(64, 128, 128)):
        super().__init__()
        layers, previous = [], 3
        for out_channels, stride in zip(channels, (2, 2, TOKEN_PIXELS // scale // 4)):
            layers += [nn.Conv2d(previous, out_channels, 3, stride=stride, padding=1), nn.SiLU()]
            previous = out_channels
        self.layers = nn.Sequential(*layers, nn.Conv2d(previous, DIM, 1))
        self.norm = nn.RMSNorm(DIM, eps=EPS)

    def forward(self, frame):
        return self.norm(self.layers(frame).flatten(2).transpose(1, 2))


class ConditionEncoder(nn.Module):
    def __init__(self, channels=48):
        super().__init__()
        self.projection = nn.Conv2d(channels, DIM, 1)
        self.norm = nn.RMSNorm(DIM, eps=EPS)

    def forward(self, latent):
        return self.norm(self.projection(latent).flatten(2).transpose(1, 2))


class Condition:
    """The keys / values of one keyframe at every block: (1, tokens, KV_HEADS, HEAD_DIM), normalized and rotated."""

    def __init__(self):
        self.keys, self.values = [], []


class Stream:
    """The decoding state of a window: its two keyframes, the video slots of every block (the previous frame and the current one)."""

    def __init__(self, start, end, tokens, height, width, dtype, device):
        self.start, self.end, self.height, self.width = start, end, height, width
        self.frame = 0
        self.slots = [(torch.empty((2, tokens, KV_HEADS, HEAD_DIM), dtype=dtype, device=device), torch.empty((2, tokens, KV_HEADS, HEAD_DIM), dtype=dtype, device=device)) for _ in range(DEPTH)]

    def release(self):
        self.slots = self.start = self.end = None


class RelayVSRDecoder(nn.Module):
    """RelayVSR's decoder: prime(latent) once per keyframe, then decode(frame, stream) for each frame of a window."""
    _offload_hooks = ["prime", "decode"]

    def __init__(self, scale=4):
        super().__init__()
        self.scale = scale  # the checkpoint's native upscaling, 4 or 2
        self.lr_encoder = LowResolutionEncoder(scale)
        self.condition_encoder = ConditionEncoder()
        self.video_type_embedding = nn.Parameter(torch.zeros(1, 1, DIM))
        self.condition_type_embedding = nn.Parameter(torch.zeros(1, 1, DIM))
        self.backbone = Backbone()
        self.spatial_head = nn.Sequential(nn.Linear(DIM, 1024), nn.SiLU())
        self.residual_projection = nn.Linear(1024, 3 * TOKEN_PIXELS ** 2)

    def prime(self, latent, rope, abort=None):
        """The Condition of a keyframe latent (1, 48, height, width): the latent's tokens through the blocks, each attending the latent's own keys."""
        height, width = latent.shape[-2:]
        x = self.condition_encoder(latent.to(self.condition_encoder.projection.weight.dtype))
        x += self.condition_type_embedding
        condition = Condition()

        def attend(qkv_list, group):
            q, k, v = qkv_list  # the kit repeats each kv head for its query heads: one of each
            qkv_list.clear()
            k, v = k[:, :, ::HEADS // KV_HEADS], v[:, :, ::HEADS // KV_HEADS]
            heads = slice(group.kv.start, group.kv.stop)
            if group.kv.start == 0:
                condition.keys.append(torch.empty((1, k.shape[1], KV_HEADS, HEAD_DIM), dtype=k.dtype, device=k.device))
                condition.values.append(torch.empty_like(condition.keys[-1]))
            keys, values = condition.keys[-1], condition.values[-1]
            keys[:, :, heads].copy_(k)
            values[:, :, heads].copy_(v)
            del k, v
            return centered_window_attention(q, [(keys, values)], CONDITION_WINDOW, None, VIDEO_WINDOW, -1, -1, height, width, HEADS // KV_HEADS, group.kv.start)

        for block in self.backbone.blocks:
            if abort is not None and abort():
                return None
            x = block(x, rope, attend, condition=True)
        return condition

    def decode(self, frame, stream, rope, rotation, abort=None):
        """The upscaled frame (1, 3, scale * height, scale * width) in [-1, 1] of a low-resolution frame (1, 3, height, width), height and width
        multiples of 16 / scale."""
        current, previous = stream.frame % 2, (stream.frame - 1) % 2 if stream.frame else -1
        x = self.lr_encoder(frame)
        x += self.video_type_embedding
        index = [0]

        def attend(qkv_list, group):
            q, k, v = qkv_list  # the kit repeats each kv head for its query heads: one of each
            qkv_list.clear()
            k, v = k[:, :, ::HEADS // KV_HEADS], v[:, :, ::HEADS // KV_HEADS]
            heads = slice(group.kv.start, group.kv.stop)
            keys, values = stream.slots[index[0]]
            keys[current, :, heads].copy_(k[0])
            values[current, :, heads].copy_(v[0])
            del k, v
            if previous >= 0:  # the previous frame's keys move to temporal position -1
                pairs = keys[previous, :, heads, :ROPE_DIMS[0]].unflatten(-1, (-1, 2)).float()
                cos, sin = rotation
                rotated = torch.stack((pairs[..., 0] * cos - pairs[..., 1] * sin, pairs[..., 0] * sin + pairs[..., 1] * cos), dim=-1)
                keys[previous, :, heads, :ROPE_DIMS[0]] = rotated.flatten(-2).to(keys.dtype)
                del pairs, rotated
            start, end = stream.start, stream.end
            return centered_window_attention(q, [(start.keys[index[0]], start.values[index[0]]), (end.keys[index[0]], end.values[index[0]])], CONDITION_WINDOW,
                                             (keys, values), VIDEO_WINDOW, previous, current, stream.height, stream.width, HEADS // KV_HEADS, group.kv.start)

        for number, block in enumerate(self.backbone.blocks):
            if abort is not None and abort():
                return None
            index[0] = number
            x = block(x, rope, attend)
        stream.frame += 1
        pixels = self.residual_projection(self.spatial_head(x))
        x = None
        pixels = F.pixel_shuffle(pixels.transpose(1, 2).reshape(1, -1, stream.height, stream.width), TOKEN_PIXELS)
        return pixels.add_(F.interpolate(frame, scale_factor=self.scale, mode="bicubic", align_corners=False))


__all__ = ["RelayVSRDecoder", "Stream", "previous_frame_rotation", "rope_table"]
