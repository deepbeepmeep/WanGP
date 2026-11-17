# SPDX-FileCopyrightText: Copyright (c) 2025 Lightricks. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""LTX-2.5 NAD diffusion video decoder.

This inference-only port keeps WanGP's VAE interface and MMGP ownership of
device placement. Neighborhood attention uses Triton when supported and the
existing bounded-workspace eager SDPA implementation otherwise.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Callable, Iterator, Sequence

import torch
from shared.utils.phase_progress import GenerationAborted, PhaseProgress
import torch.nn.functional as F
from einops import rearrange
from shared.kernels import ops as kernels, vae_ops
from torch import nn
from tqdm import tqdm

from ..transformer.timestep_embedding import PixArtAlphaCombinedTimestepSizeEmbeddings
from ...types import SpatioTemporalScaleFactors, VideoLatentShape
from .keyframes import DecodeKeyframes, joint_na3d_eager, keyframe_stage_times, keyframe_video_slots, planes_for_tile, video_keyframe_slots
from .ops import PerChannelStatistics, patchify, unpatchify
from .tiling import TilingConfig, compute_trapezoidal_mask_1d


_NA_SCORE_BUDGET = 2**25
_NA_KV_STACK_BUDGET = 2**28
_TRITON_NA_AVAILABLE, _TRITON_VERSION = vae_ops.neighborhood_status()


def _attention_backend_message() -> str:
    if _TRITON_NA_AVAILABLE:
        return f"Triton {_TRITON_VERSION}"
    return f"eager tiled SDPA (Triton {_TRITON_VERSION}; minimum 3.2.0)"


class _DecodeInterrupted(GenerationAborted):
    pass


def _check_interrupt(interrupt_check: Callable[[], bool] | None) -> None:
    if interrupt_check is not None and interrupt_check():
        raise _DecodeInterrupted


def _take_tensor(x_list: list[torch.Tensor]) -> torch.Tensor:
    x = x_list[0]
    x_list.clear()
    return x


def _linear_disposable(linear: nn.Module, x_list: list[torch.Tensor]) -> torch.Tensor:
    return linear(_take_tensor(x_list))


def _rms_norm_disposable(norm: nn.RMSNorm, x_list: list[torch.Tensor]) -> torch.Tensor:
    return norm(_take_tensor(x_list))


def _to_disposable(x_list: list[torch.Tensor], device: torch.device | str, dtype: torch.dtype | None = None) -> torch.Tensor:
    x = _take_tensor(x_list)
    target_dtype = x.dtype if dtype is None else dtype
    if x.device == torch.device(device) and x.dtype == target_dtype:
        return x
    out = torch.empty_like(x, device=device, dtype=target_dtype)
    out.copy_(x)
    return out


def _contiguous_disposable(x_list: list[torch.Tensor]) -> torch.Tensor:
    x = _take_tensor(x_list)
    if x.is_contiguous():
        return x
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    out.copy_(x)
    return out


class ChannelLinear(nn.Linear):
    @property
    def in_channels(self) -> int:
        return self.in_features

    @property
    def out_channels(self) -> int:
        return self.out_features


class LinearPixelShuffleUpsample(nn.Module):
    def __init__(self, in_channels: int, stride: tuple[int, int, int], out_channels_reduction_factor: int = 1):
        super().__init__()
        self.stride = tuple(stride)
        self.proj_out_channels = math.prod(stride) * in_channels // out_channels_reduction_factor
        self.out_channels = self.proj_out_channels // math.prod(stride)
        self.proj = nn.Linear(in_channels, self.proj_out_channels, bias=True)

    def forward(self, x_list: list[torch.Tensor], drop_leading_frame: bool = True) -> torch.Tensor:
        x = _linear_disposable(self.proj, x_list)
        x = rearrange(x, "b t h w (c pt ph pw) -> b (t pt) (h ph) (w pw) c", pt=self.stride[0], ph=self.stride[1], pw=self.stride[2])
        if self.stride[0] == 2 and drop_leading_frame:
            x = x[:, 1:]
        return x


class AdaLNZero(nn.Module):
    NUM_CHUNKS = 7

    def __init__(self, dim: int, t_emb_dim: int):
        super().__init__()
        self.proj = nn.Linear(t_emb_dim, self.NUM_CHUNKS * dim, bias=True)

    def forward(self, t_emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(chunk[:, None, None, None] for chunk in self.proj(F.silu(t_emb)).chunk(self.NUM_CHUNKS, dim=-1))


class QKVProjections(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.to_q = nn.Linear(dim, dim, bias=True)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)
        return q, k, v


def _default_rope_dim_split(head_dim: int) -> tuple[int, int, int]:
    d_t = (head_dim // 4) // 2 * 2
    d_hw = (head_dim - d_t) // 2
    if d_hw % 2:
        d_t -= 2
        d_hw = (head_dim - d_t) // 2
    return d_t, d_hw, d_hw


def _rope_inv_freqs(dim: int, base: float = 10000.0) -> torch.Tensor:
    return 1.0 / base ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim).to(torch.float32)


def _rotate_axis_(x: torch.Tensor, positions: torch.Tensor, inv_freq: torch.Tensor, axis: int) -> None:
    for w_start in range(0, x.shape[3], 8):
        part = x[:, :, :, w_start:w_start + 8]
        source = torch.empty_like(part, dtype=torch.float32)
        source.copy_(part)
        source = source.reshape(*part.shape[:-1], part.shape[-1] // 2, 2)
        even, odd = source[..., 0], source[..., 1]
        axis_positions = positions[w_start:w_start + 8] if axis == 3 else positions
        shape = [1] * even.ndim
        shape[axis] = axis_positions.shape[0]
        shape[-1] = inv_freq.shape[0]
        angles = (axis_positions[:, None] * inv_freq[None].to(axis_positions.device)).reshape(shape)
        cos, sin = angles.cos(), angles.sin()
        target = part.reshape(*part.shape[:-1], part.shape[-1] // 2, 2)
        target[..., 0].copy_(even * cos - odd * sin)
        target[..., 1].copy_(odd * cos + even * sin)


def _apply_rope_(x: torch.Tensor, split: tuple[int, int, int], inv_freqs: tuple[torch.Tensor, ...], t_pos: torch.Tensor | None = None) -> None:
    d_t, d_h, _ = split
    if t_pos is None:
        t_pos = torch.arange(x.shape[1], dtype=torch.float32, device=x.device)
    h_pos = torch.arange(x.shape[2], dtype=torch.float32, device=x.device)
    w_pos = torch.arange(x.shape[3], dtype=torch.float32, device=x.device)
    _rotate_axis_(x[..., :d_t], t_pos, inv_freqs[0], 1)
    _rotate_axis_(x[..., d_t:d_t + d_h], h_pos, inv_freqs[1], 2)
    _rotate_axis_(x[..., d_t + d_h:], w_pos, inv_freqs[2], 3)


def _window_bounds(length: int, kernel: int) -> tuple[list[int], list[int]]:
    kernel = min(kernel, length)
    lo, half = length - kernel, kernel // 2
    starts = [min(max(i - half, 0), lo) for i in range(length)]
    return starts, [start + kernel for start in starts]


def _pick_na_tiles(dims: tuple[int, int, int], kernels: list[int]) -> list[int]:
    tiles = list(dims)

    def cost() -> int:
        nq = math.prod(tiles)
        nk = math.prod(min(dim, tile + kernel - 1) for tile, kernel, dim in zip(tiles, kernels, dims, strict=True))
        return nq * nk

    while cost() > _NA_SCORE_BUDGET and max(tiles) > 1:
        axis = max(range(3), key=lambda i: tiles[i] / kernels[i])
        tiles[axis] = max(1, (tiles[axis] + 1) // 2)
    return tiles


def _na_mask(rel_bounds: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    visible_axes = []
    for starts, ends in rel_bounds:
        start = torch.tensor(starts, device=device)
        end = torch.tensor(ends, device=device)
        keys = torch.arange(int(end.max()), device=device)
        visible_axes.append((keys[None] >= start[:, None]) & (keys[None] < end[:, None]))
    visible = visible_axes[0][:, None, None, :, None, None] & visible_axes[1][None, :, None, None, :, None] & visible_axes[2][None, None, :, None, None, :]
    nq, nk = math.prod(visible.shape[:3]), math.prod(visible.shape[3:])
    mask = torch.zeros((nq, nk), dtype=dtype, device=device)
    mask.masked_fill_(~visible.reshape(nq, nk), torch.finfo(dtype).min)
    return mask.reshape(1, 1, nq, nk)


def _na3d_eager(qkv_list: list[torch.Tensor], kernel_size: tuple[int, int, int]) -> torch.Tensor:
    q, k, v = qkv_list
    qkv_list.clear()
    batch, t, h, w, heads, head_dim = q.shape
    dims = t, h, w
    kernels = [min(kernel, dim) for kernel, dim in zip(kernel_size, dims, strict=True)]
    bounds = [_window_bounds(dim, kernel) for dim, kernel in zip(dims, kernels, strict=True)]
    tile_t, tile_h, tile_w = _pick_na_tiles(dims, kernels)
    groups: dict[tuple, list[tuple[tuple[slice, ...], tuple[slice, ...]]]] = {}
    for t0 in range(0, t, tile_t):
        t1 = min(t0 + tile_t, t)
        rt0, rt1 = bounds[0][0][t0], bounds[0][1][t1 - 1]
        rel_t = tuple(start - rt0 for start in bounds[0][0][t0:t1]), tuple(end - rt0 for end in bounds[0][1][t0:t1])
        for h0 in range(0, h, tile_h):
            h1 = min(h0 + tile_h, h)
            rh0, rh1 = bounds[1][0][h0], bounds[1][1][h1 - 1]
            rel_h = tuple(start - rh0 for start in bounds[1][0][h0:h1]), tuple(end - rh0 for end in bounds[1][1][h0:h1])
            for w0 in range(0, w, tile_w):
                w1 = min(w0 + tile_w, w)
                rw0, rw1 = bounds[2][0][w0], bounds[2][1][w1 - 1]
                rel_w = tuple(start - rw0 for start in bounds[2][0][w0:w1]), tuple(end - rw0 for end in bounds[2][1][w0:w1])
                groups.setdefault((rel_t, rel_h, rel_w), []).append(((slice(t0, t1), slice(h0, h1), slice(w0, w1)), (slice(rt0, rt1), slice(rh0, rh1), slice(rw0, rw1))))

    out = q
    for rel, tiles in groups.items():
        mask = _na_mask(rel, q.dtype, q.device)
        nq, nk = mask.shape[-2:]
        group_max = max(1, _NA_KV_STACK_BUDGET // max(1, batch * heads * nk * head_dim * 2)) if q.is_cuda else 1
        first_query = tiles[0][0]
        query_shape = tuple(axis.stop - axis.start for axis in first_query)
        for start in range(0, len(tiles), group_max):
            chunk = tiles[start:start + group_max]
            q_stack = torch.stack([q[:, qs[0], qs[1], qs[2]] for qs, _ in chunk]).permute(0, 1, 5, 2, 3, 4, 6).reshape(-1, heads, nq, head_dim)
            k_stack = torch.stack([k[:, rs[0], rs[1], rs[2]] for _, rs in chunk]).permute(0, 1, 5, 2, 3, 4, 6).reshape(-1, heads, nk, head_dim)
            v_stack = torch.stack([v[:, rs[0], rs[1], rs[2]] for _, rs in chunk]).permute(0, 1, 5, 2, 3, 4, 6).reshape(-1, heads, nk, head_dim)
            result = F.scaled_dot_product_attention(q_stack, k_stack, v_stack, attn_mask=mask, scale=1.0)
            result = result.view(len(chunk), batch, heads, *query_shape, head_dim).permute(0, 1, 3, 4, 5, 2, 6)
            for index, (query_slice, _) in enumerate(chunk):
                out[:, query_slice[0], query_slice[1], query_slice[2]] = result[index]
    return out


def _na3d(qkv_list: list[torch.Tensor], kernel_size: tuple[int, int, int]) -> torch.Tensor:
    return vae_ops.neighborhood_attention(qkv_list, kernel_size, _na3d_eager)


def _joint_na3d(qkv_list: list[torch.Tensor], keyframe_times: torch.Tensor, kernel_size: tuple[int, int, int]) -> tuple[torch.Tensor, torch.Tensor]:
    return vae_ops.neighborhood_attention(qkv_list, kernel_size, joint_na3d_eager, keyframe_times=keyframe_times,
                                          slots=(video_keyframe_slots, keyframe_video_slots))


class NeighborhoodAttention3D(nn.Module):
    def __init__(self, dim: int, kernel_size: tuple[int, int, int], head_dim: int = 64, rope_dim_split: tuple[int, int, int] | None = None):
        super().__init__()
        self.dim = dim
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.kernel_size = tuple(kernel_size)
        self.scale = head_dim**-0.5
        self.rope_dim_split = rope_dim_split or _default_rope_dim_split(head_dim)
        self.register_buffer("rope_inv_t", _rope_inv_freqs(self.rope_dim_split[0]), persistent=False)
        self.register_buffer("rope_inv_h", _rope_inv_freqs(self.rope_dim_split[1]), persistent=False)
        self.register_buffer("rope_inv_w", _rope_inv_freqs(self.rope_dim_split[2]), persistent=False)
        self.qkv = QKVProjections(dim)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.q_norm = nn.RMSNorm(head_dim, eps=1e-6)
        self.k_norm = nn.RMSNorm(head_dim, eps=1e-6)

    def _qkv_rope(self, x_list: list[torch.Tensor], t_pos: torch.Tensor | None = None) -> list[torch.Tensor]:
        x = _take_tensor(x_list)
        shape = *x.shape[:4], self.num_heads, self.head_dim
        device = x.device
        q, k, v = self.qkv(x)
        x = None
        q = q.view(shape)
        k = k.view(shape)
        v = v.view(shape)
        q_input = [q]
        q = None
        q = _rms_norm_disposable(self.q_norm, q_input)
        q.mul_(self.scale)
        k_input = [k]
        k = None
        k = _rms_norm_disposable(self.k_norm, k_input)
        inv_freqs = self.rope_inv_t.to(device), self.rope_inv_h.to(device), self.rope_inv_w.to(device)
        _apply_rope_(q, self.rope_dim_split, inv_freqs, t_pos)
        _apply_rope_(k, self.rope_dim_split, inv_freqs, t_pos)
        return [q, k, v]

    def forward(self, x_list: list[torch.Tensor]) -> torch.Tensor:
        batch, t, h, w, _ = x_list[0].shape
        if any(size < kernel for size, kernel in zip((t, h, w), self.kernel_size, strict=True)):
            raise ValueError(f"NAD attention volume {(t, h, w)} is smaller than kernel {self.kernel_size}")
        out = _na3d(self._qkv_rope(x_list), self.kernel_size).reshape(batch, t, h, w, self.dim)
        out_list = [out]
        out = None
        return _linear_disposable(self.proj, out_list)

    def forward_with_keyframes(self, x_list: list[torch.Tensor], keyframe_x_list: list[torch.Tensor], keyframe_times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Joint attention over video and keyframe planes (centered windows masked at the edges, so no kernel-size floor)."""
        batch, t, h, w, _ = x_list[0].shape
        planes = keyframe_x_list[0].shape[1]
        out, keyframe_out = _joint_na3d(self._qkv_rope(x_list) + self._qkv_rope(keyframe_x_list, keyframe_times), keyframe_times, self.kernel_size)
        out_list = [out.reshape(batch, t, h, w, self.dim)]
        out = None
        return _linear_disposable(self.proj, out_list), self.proj(keyframe_out.reshape(batch, planes, h, w, self.dim))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x_list: list[torch.Tensor]) -> torch.Tensor:
        """w_down(silu(w_gate(x)) * w_up(x)) by chunks of tokens (shared chunked FFN), written over x (handed off, contiguous)."""
        return kernels.mlp(x_list, self.w_gate, self.w_down, "swiglu", up=self.w_up)


class NABlock(nn.Module):
    def __init__(self, dim: int, kernel_size: tuple[int, int, int], head_dim: int = 64, rope_dim_split: tuple[int, int, int] | None = None):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = NeighborhoodAttention3D(dim, kernel_size, head_dim, rope_dim_split)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        hidden = (int(dim * 4.0) + 15) // 16 * 16
        self.mlp = SwiGLU(dim, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_input = self.norm1(x)
        attention_list = [attention_input]
        attention_input = None
        attention_out = self.attn(attention_list)
        x.add_(attention_out)
        attention_out = None
        mlp_input = self.norm2(x)
        mlp_list = [mlp_input]
        mlp_input = None
        mlp_out = self.mlp(mlp_list)
        x.add_(mlp_out)
        return x

    def forward_with_keyframes(self, x: torch.Tensor, keyframe_x: torch.Tensor, keyframe_times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        attention_out, keyframe_attention = self.attn.forward_with_keyframes([self.norm1(x)], [self.norm1(keyframe_x)], keyframe_times)
        x.add_(attention_out)
        keyframe_x.add_(keyframe_attention)
        attention_out = keyframe_attention = None
        x.add_(self.mlp([self.norm2(x)]))
        keyframe_x.add_(self.mlp([self.norm2(keyframe_x)]))
        return x, keyframe_x


def _norm_modulate(norm: nn.RMSNorm, x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """norm(x) * (1 + scale) + shift in place of the norm output, x (batch, t, h, w, channels) borrowed, one modulation per batch row:
    the shared scale/shift on a (batch, tokens, channels) view."""
    batch = x.shape[0]
    return kernels.norm_scale_shift([x.view(batch, -1, x.shape[-1])], norm, scale.view(batch, 1, -1), shift.view(batch, 1, -1)).view(x.shape)


class DiffusionNABlock(nn.Module):
    def __init__(self, dim: int, kernel_size: tuple[int, int, int], context_channels: int, head_dim: int = 64, rope_dim_split: tuple[int, int, int] | None = None):
        super().__init__()
        self.context_proj = nn.Linear(context_channels, dim, bias=True)
        self.scale_shift_table = nn.Parameter(torch.zeros(7, dim))
        self.norm1 = nn.RMSNorm(dim, eps=1e-6)
        self.attn = NeighborhoodAttention3D(dim, kernel_size, head_dim, rope_dim_split)
        self.norm2 = nn.RMSNorm(dim, eps=1e-6)
        hidden = (int(dim * 4.0) + 15) // 16 * 16
        self.mlp = SwiGLU(dim, hidden)

    def forward_with_keyframes(self, context: torch.Tensor, x: torch.Tensor, keyframe_context: torch.Tensor, keyframe_x: torch.Tensor, modulation: tuple[torch.Tensor, ...], keyframe_times: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Both streams take their own context injection and the same modulation, then meet in the joint attention."""
        scale_msa, shift_msa, _, scale_mlp, shift_mlp, _, _ = [modulation[index] + self.scale_shift_table[index].view(1, 1, 1, 1, -1) for index in range(7)]
        x.add_(self.context_proj(context))
        keyframe_x.add_(self.context_proj(keyframe_context))
        attention_out, keyframe_attention = self.attn.forward_with_keyframes([_norm_modulate(self.norm1, x, scale_msa, shift_msa)], [_norm_modulate(self.norm1, keyframe_x, scale_msa, shift_msa)], keyframe_times)
        x.add_(attention_out)
        keyframe_x.add_(keyframe_attention)
        attention_out = keyframe_attention = None
        x.add_(self.mlp([_norm_modulate(self.norm2, x, scale_mlp, shift_mlp)]))
        keyframe_x.add_(self.mlp([_norm_modulate(self.norm2, keyframe_x, scale_mlp, shift_mlp)]))
        return x, keyframe_x

    def forward(self, context: torch.Tensor, x: torch.Tensor, modulation: tuple[torch.Tensor, ...]) -> torch.Tensor:
        scale_msa, shift_msa, _, scale_mlp, shift_mlp, _, _ = [modulation[index] + self.scale_shift_table[index].view(1, 1, 1, 1, -1) for index in range(7)]
        x.add_(self.context_proj(context))
        attention_input = _norm_modulate(self.norm1, x, scale_msa, shift_msa)
        attention_list = [attention_input]
        attention_input = None
        attention_out = self.attn(attention_list)
        x.add_(attention_out)
        attention_out = None
        mlp_input = _norm_modulate(self.norm2, x, scale_mlp, shift_mlp)
        mlp_list = [mlp_input]
        mlp_input = None
        mlp_out = self.mlp(mlp_list)
        x.add_(mlp_out)
        return x


@dataclass(frozen=True)
class _AxisInterval:
    start: int
    end: int
    left_ramp: int
    right_ramp: int


@dataclass(frozen=True)
class _Tile:
    t: _AxisInterval
    h: _AxisInterval
    w: _AxisInterval
    out_t: slice
    out_h: slice
    out_w: slice
    masks: tuple[torch.Tensor, torch.Tensor, torch.Tensor]


@dataclass(frozen=True)
class _AxisPad:
    before: int
    after: int


def _resize_axis(x_list: list[torch.Tensor], dim: int, size: int, symmetric: bool) -> tuple[torch.Tensor, _AxisPad]:
    x = _take_tensor(x_list)
    length = x.shape[dim]
    if length == size:
        return x, _AxisPad(0, 0)
    if length > size:
        before = (length - size) // 2 if symmetric else 0
        return x.narrow(dim, before, size).contiguous(), _AxisPad(before, length - size - before)
    before = (size - length) // 2 if symmetric else 0
    after = size - length - before
    parts = []
    if before:
        shape = list(x.shape)
        shape[dim] = before
        parts.append(x.narrow(dim, 0, 1).expand(shape))
    parts.append(x)
    if after:
        shape = list(x.shape)
        shape[dim] = after
        parts.append(x.narrow(dim, length - 1, 1).expand(shape))
    return torch.cat(parts, dim=dim), _AxisPad(before, after)


def _append_trailing_frames(x_list: list[torch.Tensor], count: int) -> torch.Tensor:
    x = _take_tensor(x_list)
    if count == 0:
        return x
    shape = list(x.shape)
    shape[2] += count
    out = torch.empty(shape, device=x.device, dtype=x.dtype)
    out[:, :, :x.shape[2]].copy_(x)
    out[:, :, x.shape[2]:].copy_(x[:, :, -1:])
    return out


def _split_axis(length: int, size: int, overlap: int, minimum: int) -> list[_AxisInterval]:
    if length <= size:
        return [_AxisInterval(0, length, 0, 0)]
    intervals = []
    start = 0
    while start < length:
        end = min(start + size, length)
        if end == length and end - start < minimum:
            start = max(0, end - minimum)
        left = 0 if not intervals else intervals[-1].end - start
        if intervals:
            previous = intervals[-1]
            intervals[-1] = _AxisInterval(previous.start, previous.end, previous.left_ramp, left)
        intervals.append(_AxisInterval(start, end, left, 0))
        if end == length:
            break
        start = end - overlap
    return intervals


class DiffusionVideoDecoder(nn.Module):
    is_diffusion_decoder = True

    def __init__(self, in_channels: int, out_channels: int, patch_size: int, head_dim: int, stage_channels: Sequence[int], stage_depths: Sequence[int], stage_kernels: Sequence[Sequence[int]], upsamples: Sequence[Sequence], stage5_kernel: Sequence[int], timestep_scale_multiplier: float, default_num_inference_steps: int, model_output_type: str):
        super().__init__()
        print(f"LTX-2.5 NAD neighborhood attention backend: {_attention_backend_message()}")
        self.patch_size = int(patch_size)
        self.out_channels = int(out_channels)
        self.stage_channels = tuple(int(value) for value in stage_channels)
        self.stage_depths = tuple(int(value) for value in stage_depths)
        self.stage_kernels = tuple(tuple(int(axis) for axis in kernel) for kernel in stage_kernels)
        self.stage5_kernel = tuple(int(axis) for axis in stage5_kernel)
        self.video_downscale_factors = SpatioTemporalScaleFactors.default()
        self.per_channel_statistics = PerChannelStatistics(latent_channels=in_channels)
        self.conv_in = ChannelLinear(in_channels, self.stage_channels[0], bias=True)
        # Keyframe-stream tag added to un-normalized keyframe latents before the shared conv_in.
        self.type_emb = nn.Parameter(torch.zeros(in_channels))
        self.det_stages = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for index in range(len(self.stage_channels) - 1):
            self.det_stages.append(nn.ModuleList([NABlock(self.stage_channels[index], self.stage_kernels[index], head_dim) for _ in range(self.stage_depths[index])]))
            stride, reduction = upsamples[index]
            self.upsamples.append(LinearPixelShuffleUpsample(self.stage_channels[index], tuple(stride), int(reduction)))
        self.t_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(embedding_dim=384, size_emb_dim=0)
        final_channels = self.stage_channels[-1]
        noised_channels = self.out_channels * self.patch_size**2
        self.conv_in_x_t = ChannelLinear(noised_channels, final_channels, bias=True)
        self.shared_adaln = AdaLNZero(final_channels, 384)
        self.diff_blocks = nn.ModuleList([DiffusionNABlock(final_channels, self.stage5_kernel, final_channels, head_dim) for _ in range(self.stage_depths[-1])])
        self.norm_out = nn.RMSNorm(final_channels, eps=1e-6)
        self.conv_out = ChannelLinear(final_channels, noised_channels, bias=True)
        self.timestep_scale_multiplier = float(timestep_scale_multiplier)
        self.model_output_type = model_output_type
        self.register_buffer("default_inference_timesteps", torch.linspace(1.0, 1.0 / default_num_inference_steps, default_num_inference_steps), persistent=False)
        self._trailing_latent_frames = (self.stage_kernels[0][0] // 2) * 2
        self._stage_minimum = self._all_stages_minimum(upsamples)
        stride4 = tuple(self.upsamples[3].stride)
        self._tile_minimum = tuple(max(self.stage_kernels[3][axis], math.ceil(self.stage5_kernel[axis] / stride4[axis])) for axis in range(3))
        halo4 = tuple(self.stage_depths[3] * (self.stage_kernels[3][axis] // 2) for axis in range(3))
        halo5 = tuple(math.ceil(self.stage_depths[-1] * (self.stage5_kernel[axis] // 2) / stride4[axis]) for axis in range(3))
        self._tile_halo = tuple(max(halo4[axis], halo5[axis]) for axis in range(3))
        # Temporal upsampling still to come at each stage input, then 1 at stage 5: keyframe plane time units.
        self._keyframe_time_strides = tuple(math.prod(int(upsample[0][0]) for upsample in upsamples[index:]) for index in range(len(upsamples) + 1))

    @classmethod
    def from_config(cls, config: dict) -> "DiffusionVideoDecoder":
        vae_config = config["vae"]
        decoder = vae_config["decoder"]
        return cls(
            in_channels=decoder["in_channels"], out_channels=decoder["out_channels"], patch_size=decoder["patch_size"],
            head_dim=decoder["head_dim"], stage_channels=decoder["stage_channels"], stage_depths=decoder["stage_depths"],
            stage_kernels=decoder["stage_kernels"], upsamples=decoder["upsamples"], stage5_kernel=decoder["stage5_kernel"],
            timestep_scale_multiplier=decoder["timestep_scale_multiplier"], default_num_inference_steps=decoder["default_num_inference_steps"],
            model_output_type=vae_config["model_output_type"],
        )

    def _all_stages_minimum(self, upsamples: Sequence[Sequence]) -> tuple[int, int, int]:
        cumulative = [1, 1, 1]
        minimum = [1, 1, 1]
        for index in range(4):
            for axis in range(3):
                minimum[axis] = max(minimum[axis], math.ceil(self.stage_kernels[index][axis] / cumulative[axis]))
            for axis in range(3):
                cumulative[axis] *= int(upsamples[index][0][axis])
        for axis in range(3):
            minimum[axis] = max(minimum[axis], math.ceil(self.stage5_kernel[axis] / cumulative[axis]))
        return tuple(minimum)

    def _run_stage(self, x_list: list[torch.Tensor], index: int, drop_leading_frame: bool, interrupt_check: Callable[[], bool] | None, keyframe_x: torch.Tensor | None = None, keyframe_times: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = _take_tensor(x_list)
        for block in self.det_stages[index]:
            if keyframe_x is None:
                x = block(x)
            else:
                x, keyframe_x = block.forward_with_keyframes(x, keyframe_x, keyframe_times)
            _check_interrupt(interrupt_check)
        upsample_input = [x]
        x = None
        x = self.upsamples[index](upsample_input, drop_leading_frame=drop_leading_frame)
        if keyframe_x is not None:
            # Each plane is upsampled as its own one-frame clip: temporal strides collapse, only H/W grow.
            batch, planes = keyframe_x.shape[:2]
            keyframe_x = self.upsamples[index]([keyframe_x.flatten(0, 1).unsqueeze(1)], drop_leading_frame=True).squeeze(1).unflatten(0, (batch, planes))
        return x, keyframe_x

    def _keyframe_times(self, index: int, keyframe_indices: tuple[int, ...] | None, origin: float, device: torch.device) -> torch.Tensor | None:
        return None if keyframe_indices is None else keyframe_stage_times(keyframe_indices, self._keyframe_time_strides[index], origin, device)

    def _stages_1_to_3(self, latent_list: list[torch.Tensor], interrupt_check: Callable[[], bool] | None, keyframes: DecodeKeyframes | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        latent = _take_tensor(latent_list)
        std = self.per_channel_statistics.get_buffer("std-of-means").view(1, -1, 1, 1, 1).to(latent)
        mean = self.per_channel_statistics.get_buffer("mean-of-means").view(1, -1, 1, 1, 1).to(latent)
        latent.mul_(std).add_(mean)
        x = latent.permute(0, 2, 3, 4, 1)
        latent = None
        conv_input = [x]
        x = None
        x = _linear_disposable(self.conv_in, conv_input)
        device = x.device
        keyframe_x = keyframe_indices = None
        if keyframes is not None:
            keyframe_indices = keyframes.pixel_frame_indices
            keyframe_x = self.conv_in((keyframes.latents.to(std) * std + mean).permute(0, 2, 3, 4, 1) + self.type_emb.to(std))
        for index in range(3):
            stage_input = [x]
            x = None
            x, keyframe_x = self._run_stage(stage_input, index, True, interrupt_check, keyframe_x, self._keyframe_times(index, keyframe_indices, 0.0, device))
        return x, keyframe_x

    def _stage_4(self, x_list: list[torch.Tensor], is_origin: bool, pad_trailing: bool, interrupt_check: Callable[[], bool] | None, keyframe_x: torch.Tensor | None = None, keyframe_times: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        x, keyframe_x = self._run_stage(x_list, 3, is_origin, interrupt_check, keyframe_x, keyframe_times)
        if pad_trailing:
            ghost = self._trailing_latent_frames * self.video_downscale_factors.time
            content = max(x.shape[1] - ghost, 1)
            target = min(x.shape[1], max(content, self.stage5_kernel[0]))
            resize_input = [x]
            x = None
            x, _ = _resize_axis(resize_input, 1, target, False)
        return x, keyframe_x

    def _stage5_pixels(self, x_list: list[torch.Tensor], high_precision: bool = False) -> torch.Tensor:
        if high_precision:
            # HDR codes need more than bf16 output steps (which band shadows and highlights): run the output head in fp32.
            x = F.rms_norm(_take_tensor(x_list).float(), (self.norm_out.normalized_shape[0],), self.norm_out.weight.float(), self.norm_out.eps)
            x = F.linear(x, self.conv_out.weight.float(), self.conv_out.bias.float()).permute(0, 4, 1, 2, 3)
        else:
            x = _rms_norm_disposable(self.norm_out, x_list)
            conv_input = [x]
            x = None
            x = _linear_disposable(self.conv_out, conv_input).permute(0, 4, 1, 2, 3)
        contiguous_input = [x]
        x = None
        x = _contiguous_disposable(contiguous_input)
        return unpatchify(x, patch_size_hw=self.patch_size)

    def _diffusion_step(self, context: torch.Tensor, noise_list: list[torch.Tensor], timestep: torch.Tensor, interrupt_check: Callable[[], bool] | None, keyframe_context: torch.Tensor | None = None, keyframe_noise: torch.Tensor | None = None, keyframe_times: torch.Tensor | None = None, high_precision: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        """One stage-5 step; with keyframes the planes are a second pixel stream that the video attends to."""
        noise = _take_tensor(noise_list)
        x = patchify(noise, patch_size_hw=self.patch_size).permute(0, 2, 3, 4, 1)
        noise = None
        conv_input = [x]
        x = None
        x = _linear_disposable(self.conv_in_x_t, conv_input)
        t_emb = self.t_embedder(self.timestep_scale_multiplier * timestep, hidden_dtype=x.dtype)
        modulation = self.shared_adaln(t_emb)
        keyframe_x = None if keyframe_context is None else self.conv_in_x_t(patchify(keyframe_noise, patch_size_hw=self.patch_size).permute(0, 2, 3, 4, 1))
        for block in self.diff_blocks:
            if keyframe_x is None:
                x = block(context, x, modulation)
            else:
                x, keyframe_x = block.forward_with_keyframes(context, x, keyframe_context, keyframe_x, modulation, keyframe_times)
            _check_interrupt(interrupt_check)
        norm_input = [x]
        x = None
        return self._stage5_pixels(norm_input, high_precision), None if keyframe_x is None else self._stage5_pixels([keyframe_x], high_precision)

    def _euler_update(self, x_list: list[torch.Tensor], model_out_list: list[torch.Tensor], timestep: torch.Tensor, next_timestep: torch.Tensor) -> torch.Tensor:
        x = _take_tensor(x_list)
        device = x.device
        dt = (timestep - next_timestep).view(-1, 1, 1, 1, 1).float()
        velocity = _to_disposable(model_out_list, device, torch.float32)
        if self.model_output_type == "x0":
            velocity.neg_().add_(x).div_(timestep.view(-1, 1, 1, 1, 1).float())
        x_dtype = x.dtype
        velocity.mul_(-dt).add_(x)
        x = None
        velocity_list = [velocity]
        velocity = None
        return _to_disposable(velocity_list, device, x_dtype)

    def _decode_tile(self, features_list: list[torch.Tensor], noise_list: list[torch.Tensor], is_origin: bool, pad_trailing: bool, interrupt_check: Callable[[], bool] | None, keyframes: tuple | None = None, high_precision: bool = False) -> torch.Tensor:
        """``keyframes`` is ``(stage-4 plane features, plane noise, global pixel indices, stage-4 origin, pixel origin)`` of this tile."""
        features = _take_tensor(features_list)
        device, batch = features.device, features.shape[0]
        stage_input = [features]
        features = None
        keyframe_features, keyframe_noise, keyframe_indices, stage4_origin, pixel_origin = keyframes or (None, None, None, 0, 0)
        context, keyframe_context = self._stage_4(stage_input, is_origin, pad_trailing, interrupt_check, keyframe_features, self._keyframe_times(3, keyframe_indices, stage4_origin, device))
        keyframe_features = None
        keyframe_times = self._keyframe_times(4, keyframe_indices, pixel_origin, device)
        timesteps = self.default_inference_timesteps.to(device).unsqueeze(0).expand(batch, -1)
        x = _take_tensor(noise_list)
        for index, timestep in enumerate(timesteps.unbind(1)):
            if index == timesteps.shape[1] - 1 and self.model_output_type == "x0":
                noise_input = [x]
                x = None
                return self._diffusion_step(context, noise_input, timestep, interrupt_check, keyframe_context, keyframe_noise, keyframe_times, high_precision)[0]
            model_out, keyframe_out = self._diffusion_step(context, [x], timestep, interrupt_check, keyframe_context, keyframe_noise, keyframe_times, high_precision)
            next_timestep = timesteps[:, index + 1] if index + 1 < timesteps.shape[1] else torch.zeros_like(timestep)
            x_input, model_output_list = [x], [model_out]
            x = model_out = None
            x = self._euler_update(x_input, model_output_list, timestep, next_timestep)
            if keyframe_out is not None:
                keyframe_noise = self._euler_update([keyframe_noise], [keyframe_out], timestep, next_timestep)
                keyframe_out = None
        return x

    def _schedule(self, shape: VideoLatentShape, tiling_config: TilingConfig | None, device: torch.device) -> list[_Tile]:
        stage4_t = shape.frames
        stage4_h = shape.height
        stage4_w = shape.width
        for upsample in self.upsamples[:3]:
            stage4_t = stage4_t * upsample.stride[0] - (1 if upsample.stride[0] == 2 else 0)
            stage4_h *= upsample.stride[1]
            stage4_w *= upsample.stride[2]
        pixel_scale = self.upsamples[3].stride[0], self.upsamples[3].stride[1] * self.patch_size, self.upsamples[3].stride[2] * self.patch_size
        spatial = tiling_config.spatial_config if tiling_config is not None else None
        temporal = tiling_config.temporal_config if tiling_config is not None else None
        whole = max(stage4_h * pixel_scale[1], stage4_w * pixel_scale[2])
        spatial_tile = whole if spatial is None else spatial.tile_size_in_pixels if spatial.diffusion_tile_in_pixels is None else spatial.diffusion_tile_in_pixels or whole
        spatial_tile = max(192, spatial_tile)
        temporal_tile = temporal.tile_size_in_frames if temporal is not None else stage4_t * pixel_scale[0] - 1
        temporal_tile = max(48, temporal_tile)
        required_overlap = tuple(self._tile_halo[axis] * pixel_scale[axis] for axis in range(3))
        spatial_overlap = max(required_overlap[1], required_overlap[2])
        temporal_overlap = required_overlap[0]
        t_intervals = _split_axis(stage4_t, math.ceil(temporal_tile / pixel_scale[0]), math.ceil(temporal_overlap / pixel_scale[0]), self._tile_minimum[0])
        if len(t_intervals) > 1:
            first_frames = (t_intervals[0].end - t_intervals[0].start) * pixel_scale[0] - (pixel_scale[0] - 1)
            next_frames = (t_intervals[1].end - t_intervals[1].start) * pixel_scale[0]
            shift = math.ceil(max(0, next_frames - first_frames) / pixel_scale[0])
            if shift and t_intervals[-1].end - t_intervals[-1].start - shift >= self._tile_minimum[0]:
                t_intervals = [
                    _AxisInterval(
                        interval.start + (shift if index else 0),
                        interval.end + (shift if index < len(t_intervals) - 1 else 0),
                        interval.left_ramp,
                        interval.right_ramp,
                    )
                    for index, interval in enumerate(t_intervals)
                ]
        h_intervals = _split_axis(stage4_h, math.ceil(spatial_tile / pixel_scale[1]), math.ceil(spatial_overlap / pixel_scale[1]), self._tile_minimum[1])
        w_intervals = _split_axis(stage4_w, math.ceil(spatial_tile / pixel_scale[2]), math.ceil(spatial_overlap / pixel_scale[2]), self._tile_minimum[2])
        tiles = []
        for t_interval, h_interval, w_interval in itertools.product(t_intervals, h_intervals, w_intervals):
            t_start = t_interval.start * pixel_scale[0] - (pixel_scale[0] - 1 if t_interval.start else 0)
            t_end = t_interval.end * pixel_scale[0] - (pixel_scale[0] - 1)
            out_t = slice(t_start, t_end)
            out_h = slice(h_interval.start * pixel_scale[1], h_interval.end * pixel_scale[1])
            out_w = slice(w_interval.start * pixel_scale[2], w_interval.end * pixel_scale[2])
            masks = (
                compute_trapezoidal_mask_1d(t_end - t_start, t_interval.left_ramp * pixel_scale[0], t_interval.right_ramp * pixel_scale[0], device=device),
                compute_trapezoidal_mask_1d(out_h.stop - out_h.start, h_interval.left_ramp * pixel_scale[1], h_interval.right_ramp * pixel_scale[1], device=device),
                compute_trapezoidal_mask_1d(out_w.stop - out_w.start, w_interval.left_ramp * pixel_scale[2], w_interval.right_ramp * pixel_scale[2], device=device),
            )
            tiles.append(_Tile(t_interval, h_interval, w_interval, out_t, out_h, out_w, masks))
        return tiles

    @staticmethod
    def _crop_emit(buffer: torch.Tensor, weights: torch.Tensor, global_start: int, content_shape: VideoLatentShape, h_pad: _AxisPad, w_pad: _AxisPad) -> torch.Tensor | None:
        if global_start >= content_shape.frames:
            return None
        frames = min(buffer.shape[2], content_shape.frames - global_start)
        if frames < 1:
            return None
        h_start = h_pad.before * 32
        w_start = w_pad.before * 32
        buffer = buffer[:, :, :frames, h_start:h_start + content_shape.height, w_start:w_start + content_shape.width]
        weights = weights[:, :, :frames, h_start:h_start + content_shape.height, w_start:w_start + content_shape.width]
        return buffer.div_(weights.clamp_min_(1e-8)).to(buffer.dtype)

    def _decode_pixels(self, latent_list: list[torch.Tensor], tiling_config: TilingConfig | None, generator: torch.Generator | None, interrupt_check: Callable[[], bool] | None, output_uint8: bool = False, keyframes: DecodeKeyframes | None = None, high_precision: bool = False) -> Iterator[torch.Tensor]:
        """``high_precision`` (HDR) computes the output head and accumulates the tiles in fp32."""
        latent = _take_tensor(latent_list)
        content_latent = VideoLatentShape.from_torch_shape(latent.shape)
        content_pixels = content_latent.upscale(self.video_downscale_factors)._replace(channels=self.out_channels)
        target = max(latent.shape[2], self._stage_minimum[0])
        resize_input = [latent]
        latent = None
        latent, _ = _resize_axis(resize_input, 2, target, False)
        target = max(latent.shape[3], self._stage_minimum[1])
        resize_input = [latent]
        latent = None
        latent, h_pad = _resize_axis(resize_input, 3, target, True)
        target = max(latent.shape[4], self._stage_minimum[2])
        resize_input = [latent]
        latent = None
        latent, w_pad = _resize_axis(resize_input, 4, target, True)
        if keyframes is not None:
            # Same symmetric spatial pad as the video, or every plane would be offset from it.
            keyframe_latents, _ = _resize_axis([keyframes.latents], 3, latent.shape[3], True)
            keyframe_latents, _ = _resize_axis([keyframe_latents], 4, latent.shape[4], True)
            keyframes = DecodeKeyframes(keyframe_latents, keyframes.pixel_frame_indices)
            keyframe_latents = None
        work_latent = VideoLatentShape.from_torch_shape(latent.shape)
        work_pixels = work_latent.upscale(self.video_downscale_factors)._replace(channels=self.out_channels)
        tiles = self._schedule(work_latent, tiling_config, latent.device)
        with PhaseProgress(len(tiles)) as phase_progress:
            trailing_input = [latent]
            latent = None
            latent = _append_trailing_frames(trailing_input, self._trailing_latent_frames)
            stage_input = [latent]
            latent = None
            features, keyframe_features = self._stages_1_to_3(stage_input, interrupt_check, keyframes)
            decode_device = features.device
            features_list = [features]
            features = None
            features = _to_disposable(features_list, "cpu")
            content_stage4_frames = (work_latent.frames - 1) * 4 + 1
            temporal_groups: list[list[_Tile]] = []
            for tile in tiles:
                if not temporal_groups or temporal_groups[-1][0].out_t != tile.out_t:
                    temporal_groups.append([tile])
                else:
                    temporal_groups[-1].append(tile)
            overlap_buffer = overlap_weights = None
            progress = tqdm(total=len(tiles), desc="NAD VAE Decoding", leave=False)
            try:
                for group_index, group in enumerate(temporal_groups):
                    _check_interrupt(interrupt_check)
                    group_start, group_stop = group[0].out_t.start, group[0].out_t.stop
                    group_frames = group_stop - group_start
                    accum_dtype = torch.float32 if high_precision else torch.float16 if features.dtype == torch.bfloat16 else features.dtype
                    buffer = torch.zeros((work_pixels.batch, self.out_channels, group_frames, work_pixels.height, work_pixels.width), device="cpu", dtype=accum_dtype)
                    weights = torch.zeros((1, 1, group_frames, work_pixels.height, work_pixels.width), device="cpu", dtype=accum_dtype)
                    for tile in group:
                        is_origin = tile.t.start == 0
                        pad_trailing = tile.t.end == content_stage4_frames
                        t_stop = features.shape[1] if pad_trailing else tile.t.end
                        feature_tile = features[:, tile.t.start:t_stop, tile.h.start:tile.h.end, tile.w.start:tile.w.end]
                        feature_list = [feature_tile]
                        feature_tile = None
                        feature_tile = _to_disposable(feature_list, decode_device)
                        stage5_frames = (tile.t.end - tile.t.start) * self.upsamples[3].stride[0] - (1 if is_origin else 0)
                        if pad_trailing:
                            stage5_frames = max(stage5_frames, self.stage5_kernel[0])
                        noise_shape = (work_pixels.batch, self.out_channels, stage5_frames, (tile.h.end - tile.h.start) * 8, (tile.w.end - tile.w.start) * 8)
                        noise_device = generator.device if generator is not None else decode_device
                        noise = torch.randn(noise_shape, dtype=features.dtype, device=noise_device, generator=generator)
                        noise_list = [noise]
                        noise = None
                        noise = _to_disposable(noise_list, decode_device)
                        feature_list, noise_list = [feature_tile], [noise]
                        feature_tile = noise = None
                        tile_keyframes = None
                        if keyframe_features is not None:
                            keep = planes_for_tile(keyframes.pixel_frame_indices, tile.out_t.start, tile.out_t.stop - 1)
                            keyframe_noise = torch.randn((*noise_shape[:2], len(keep), *noise_shape[3:]), dtype=features.dtype, device=noise_device, generator=generator).to(decode_device)
                            tile_keyframes = (keyframe_features[:, keep, tile.h.start:tile.h.end, tile.w.start:tile.w.end], keyframe_noise, tuple(keyframes.pixel_frame_indices[index] for index in keep), tile.t.start, tile.out_t.start)
                            keyframe_noise = None
                        decoded = self._decode_tile(feature_list, noise_list, is_origin, pad_trailing, interrupt_check, tile_keyframes, high_precision)
                        tile_keyframes = None
                        expected = (tile.out_t.stop - tile.out_t.start, tile.out_h.stop - tile.out_h.start, tile.out_w.stop - tile.out_w.start)
                        resize_input = [decoded]
                        decoded = None
                        decoded, _ = _resize_axis(resize_input, 2, expected[0], False)
                        resize_input = [decoded]
                        decoded = None
                        decoded, _ = _resize_axis(resize_input, 3, expected[1], False)
                        resize_input = [decoded]
                        decoded = None
                        decoded, _ = _resize_axis(resize_input, 4, expected[2], False)
                        decoded_list = [decoded]
                        decoded = None
                        decoded = _to_disposable(decoded_list, decode_device, accum_dtype)
                        mask_t, mask_h, mask_w = (mask.to(device=decoded.device, dtype=accum_dtype) for mask in tile.masks)
                        decoded.mul_(mask_t.view(1, 1, -1, 1, 1)).mul_(mask_h.view(1, 1, 1, -1, 1)).mul_(mask_w.view(1, 1, 1, 1, -1))
                        mask_t = mask_h = mask_w = None
                        decoded_list = [decoded]
                        decoded = None
                        decoded = _to_disposable(decoded_list, "cpu")
                        local_t = slice(tile.out_t.start - group_start, tile.out_t.stop - group_start)
                        coords = (slice(None), slice(None), local_t, tile.out_h, tile.out_w)
                        buffer[coords] += decoded
                        mask_t, mask_h, mask_w = (mask.to(device="cpu", dtype=accum_dtype) for mask in tile.masks)
                        strength = mask_t.view(1, 1, -1, 1, 1) * mask_h.view(1, 1, 1, -1, 1) * mask_w.view(1, 1, 1, 1, -1)
                        weights[coords] += strength
                        del decoded, strength
                        progress.update(1)
                        phase_progress.advance()
                        _check_interrupt(interrupt_check)
                    if overlap_buffer is not None:
                        overlap_frames = overlap_buffer.shape[2]
                        buffer[:, :, :overlap_frames] += overlap_buffer
                        weights[:, :, :overlap_frames] += overlap_weights
                    next_start = temporal_groups[group_index + 1][0].out_t.start if group_index + 1 < len(temporal_groups) else group_stop
                    emit_frames = next_start - group_start
                    emitted = self._crop_emit(buffer[:, :, :emit_frames], weights[:, :, :emit_frames], group_start, content_pixels, h_pad, w_pad)
                    if emitted is not None:
                        emitted = emitted.add_(1.0).mul_(127.5).clamp_(0.0, 255.0).to(torch.uint8) if output_uint8 else emitted.clone()
                    if next_start < group_stop:
                        overlap_buffer = buffer[:, :, emit_frames:].clone()
                        overlap_weights = weights[:, :, emit_frames:].clone()
                    else:
                        overlap_buffer = overlap_weights = None
                    del buffer, weights
                    if emitted is not None:
                        yield emitted
            finally:
                progress.close()

    def forward(self, sample: torch.Tensor | list[torch.Tensor], generator: torch.Generator | None = None, interrupt_check: Callable[[], bool] | None = None, keyframes: DecodeKeyframes | None = None, high_precision: bool = False) -> torch.Tensor:
        sample_list = sample if isinstance(sample, list) else [sample]
        sample = None
        return next(self._decode_pixels(sample_list, None, generator, interrupt_check, keyframes=keyframes, high_precision=high_precision))

    def tiled_decode(self, latent: torch.Tensor | list[torch.Tensor], tiling_config: TilingConfig, generator: torch.Generator | None = None, interrupt_check: Callable[[], bool] | None = None, copy_emitted_chunks: bool = False, output_uint8: bool = False, keyframes: DecodeKeyframes | None = None, high_precision: bool = False) -> Iterator[torch.Tensor]:
        try:
            latent_list = latent if isinstance(latent, list) else [latent]
            latent = None
            for chunk in self._decode_pixels(latent_list, tiling_config, generator, interrupt_check, output_uint8, keyframes, high_precision):
                yield chunk.clone() if copy_emitted_chunks else chunk
        except _DecodeInterrupted:
            return
