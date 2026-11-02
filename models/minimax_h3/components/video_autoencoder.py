# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmgp import transfer

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import AttentionMixin
from diffusers.models.autoencoders.vae import AutoencoderMixin, DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.utils import logging
from diffusers.utils.accelerate_utils import apply_forward_hook

from shared import attention_kit
from shared.attention import pay_attention

from ..interrupt import GenerationInterrupted
from shared.utils.phase_progress import vae_encoding_progress, PhaseProgress
from shared.kernels import ops as kernels, vae_ops

logger = logging.get_logger(__name__)
_FUSED_DTYPES = (torch.bfloat16, torch.float16)


class MiniMaxH3VideoCausalConv3d(nn.Conv3d):
    r"""
    3D convolution used throughout the MiniMax-H3 video encoder.

    Spatial padding is symmetric and uses `spatial_padding_mode` (`"reflect"` in the released checkpoint); temporal
    padding is causal, i.e. `kernel_size_t - 1` zero frames are prepended and nothing is appended.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        stride: int | tuple[int, int, int] = 1,
        spatial_padding: int = 0,
        temporal_padding: int = 0,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0)
        self.spatial_padding = spatial_padding
        self.temporal_padding = temporal_padding
        self.spatial_padding_mode = spatial_padding_mode

    def forward(self, hidden_states: torch.Tensor, *, pre_padded: bool = False, residual: torch.Tensor | None = None) -> torch.Tensor:
        if self.spatial_padding > 0 and not pre_padded:
            padding = self.spatial_padding
            hidden_states = F.pad(
                hidden_states, (padding, padding, padding, padding, 0, 0), mode=self.spatial_padding_mode
            )
        if self.temporal_padding > 0 and not pre_padded:
            hidden_states = F.pad(hidden_states, (0, 0, 0, 0, self.temporal_padding, 0), mode="constant")
        x_list, hidden_states = [hidden_states], None
        return vae_ops.conv3d(x_list, self, residual)


class MiniMaxH3VideoGroupNorm(nn.GroupNorm):
    r"""
    Group normalization applied to each latent frame in isolation (`use_t_isolated_gn` in the original config): the
    temporal axis is folded into the batch axis so statistics never mix across frames.
    """

    def forward(self, hidden_states: torch.Tensor, *, silu_pad=None) -> torch.Tensor:
        x_list, hidden_states = [hidden_states], None
        return vae_ops.frame_group_norm(x_list, self, silu_pad)


def _norm_silu_conv(hidden_states: list[torch.Tensor], norm, conv, residual=None):
    """`hidden_states` holds the input, handed over so that it is released as soon as it is normalized."""
    hidden_states = hidden_states.pop()
    if conv.spatial_padding_mode != "reflect":
        hidden_states = F.silu(norm(hidden_states))
        return conv(hidden_states, residual=residual)
    p = conv.spatial_padding
    hidden_states = norm(hidden_states, silu_pad=(p, p, p, p, conv.temporal_padding))
    return conv(hidden_states, pre_padded=True, residual=residual)


class MiniMaxH3VideoResnetBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.norm1 = MiniMaxH3VideoGroupNorm(norm_num_groups, in_channels, eps=norm_eps, affine=True)
        self.conv1 = MiniMaxH3VideoCausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.norm2 = MiniMaxH3VideoGroupNorm(norm_num_groups, out_channels, eps=norm_eps, affine=True)
        self.conv2 = MiniMaxH3VideoCausalConv3d(
            out_channels,
            out_channels,
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.nin_shortcut = None
        if in_channels != out_channels:
            self.nin_shortcut = MiniMaxH3VideoCausalConv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = [_norm_silu_conv([hidden_states], self.norm1, self.conv1)]
        if self.nin_shortcut is not None:
            residual = self.nin_shortcut(residual)
        return _norm_silu_conv(hidden_states, self.norm2, self.conv2, residual)


class MiniMaxH3VideoDownsample3d(nn.Module):
    r"""
    Strided 3x3x3 downsampling convolution. A spatial stride of 2 is preceded by an asymmetric bottom/right pad of 1
    (the convolution itself carries no spatial padding), so the output is exactly `ceil(size / 2)`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        temporal_stride: int = 1,
        spatial_stride: int = 2,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.spatial_stride = spatial_stride
        self.spatial_padding_mode = spatial_padding_mode
        self.conv = MiniMaxH3VideoCausalConv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=(temporal_stride, spatial_stride, spatial_stride),
            spatial_padding=0,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(self, hidden_states: list[torch.Tensor]) -> torch.Tensor:
        """`hidden_states` holds the input, handed over: with reflect padding, the bottom/right pad and the causal frames are
        written into one buffer and the input is released before the convolution."""
        hidden_states = hidden_states.pop()
        if self.spatial_stride != 2:
            return self.conv(hidden_states)
        if self.spatial_padding_mode != "reflect":
            return self.conv(F.pad(hidden_states, (0, 1, 0, 1, 0, 0), mode=self.spatial_padding_mode))
        front, (height, width) = self.conv.temporal_padding, hidden_states.shape[-2:]
        # channels-last activations (fused convolutions) keep their layout: the copies stay coalesced and the convolution reads the buffer as is
        layout = torch.channels_last_3d if hidden_states.is_contiguous(memory_format=torch.channels_last_3d) else torch.contiguous_format
        padded = torch.empty((*hidden_states.shape[:2], front + hidden_states.shape[2], height + 1, width + 1), dtype=hidden_states.dtype,
                             device=hidden_states.device, memory_format=layout)
        padded[:, :, :front].zero_()
        frames = padded[:, :, front:]
        frames[..., :height, :width].copy_(hidden_states)
        frames[..., height, :width].copy_(hidden_states[..., height - 2, :])
        hidden_states = None
        frames[..., width].copy_(frames[..., width - 2])  # the corner reflects both ways, like F.pad
        frames = None
        return self.conv(padded, pre_padded=True)


class MiniMaxH3VideoDownBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_layers: int,
        temporal_downsample_factor: int,
        spatial_downsample_factor: int,
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()
        self.block = nn.ModuleList(
            [
                MiniMaxH3VideoResnetBlock3d(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    spatial_padding_mode=spatial_padding_mode,
                )
                for i in range(num_layers)
            ]
        )
        self.downsample = None
        if temporal_downsample_factor * spatial_downsample_factor > 1:
            self.downsample = MiniMaxH3VideoDownsample3d(
                out_channels,
                out_channels,
                temporal_stride=temporal_downsample_factor,
                spatial_stride=spatial_downsample_factor,
                spatial_padding_mode=spatial_padding_mode,
            )

        self.gradient_checkpointing = False

    def forward(self, hidden_states: list[torch.Tensor]) -> torch.Tensor:
        """`hidden_states` holds the input, handed over so that it is released once the first resnet no longer needs it."""
        hidden_states = hidden_states.pop()
        for resnet in self.block:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states)
            else:
                hidden_states = resnet(hidden_states)
        if self.downsample is not None:
            hidden_states = [hidden_states]
            hidden_states = self.downsample(hidden_states)
        return hidden_states


class MiniMaxH3VideoEncoder3d(nn.Module):
    r"""
    Causal 3D CNN encoder. `block_out_channels` gives the channel count of every level; the per-level
    `spatial_downsample_factors` / `temporal_downsample_factors` multiply out to the total compression ratios.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 48,
        block_out_channels: tuple[int, ...] = (128, 256, 256, 512, 512, 1024),
        layers_per_block: int = 2,
        spatial_downsample_factors: tuple[int, ...] = (2, 2, 2, 2, 1, 1),
        temporal_downsample_factors: tuple[int, ...] = (1, 2, 2, 1, 1, 1),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
    ) -> None:
        super().__init__()

        self.conv_in = MiniMaxH3VideoCausalConv3d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )

        block_in_channels = (block_out_channels[0],) + tuple(block_out_channels[:-1])
        self.down = nn.ModuleList(
            [
                MiniMaxH3VideoDownBlock3d(
                    in_channels=block_in_channels[i],
                    out_channels=block_out_channels[i],
                    num_layers=layers_per_block,
                    temporal_downsample_factor=temporal_downsample_factors[i],
                    spatial_downsample_factor=spatial_downsample_factors[i],
                    norm_num_groups=norm_num_groups,
                    norm_eps=norm_eps,
                    spatial_padding_mode=spatial_padding_mode,
                )
                for i in range(len(block_out_channels))
            ]
        )

        self.norm_out = MiniMaxH3VideoGroupNorm(norm_num_groups, block_out_channels[-1], eps=norm_eps, affine=True)
        self.conv_out = MiniMaxH3VideoCausalConv3d(
            block_out_channels[-1],
            out_channels,
            kernel_size=3,
            spatial_padding=1,
            temporal_padding=2,
            spatial_padding_mode=spatial_padding_mode,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = [self.conv_in(hidden_states)]
        for down_block in self.down:
            hidden_states = [down_block(hidden_states)]  # each block takes its input out of the list and releases it
            if getattr(self, "_interrupt", False):
                raise GenerationInterrupted
        return _norm_silu_conv(hidden_states, self.norm_out, self.conv_out)


class MiniMaxH3VideoRotaryPosEmbed(nn.Module):
    r"""
    3-axis rotary embedding for the ViT decoder. Coordinates are length-normalized to `[-1, 1)` per axis and scaled by
    `2 * pi`, and the resulting `(t, h, w)` angles are concatenated and then duplicated, so the first
    `rope_dim_ratio * attention_head_dim` channels of every head are rotated.
    """

    def __init__(self, dim: int, theta: float = 100.0, num_axes: int = 3) -> None:
        super().__init__()
        if dim % (2 * num_axes) != 0:
            raise ValueError(f"`dim` {dim} must be divisible by `2 * num_axes` {2 * num_axes}.")
        inv_freq = 1.0 / theta ** torch.arange(0, 1, 2 * num_axes / dim, dtype=torch.float32)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        angles = 2.0 * math.pi * position_ids[:, :, :, None] * self.inv_freq[None, None, None, :]
        angles = angles.flatten(2, 3).tile(2).unsqueeze(2)
        return vae_ops.prepare_rotary(angles.cos(), angles.sin())


def _take(x_list: list[torch.Tensor]) -> torch.Tensor:
    hidden_states = x_list[0]
    x_list.clear()
    return hidden_states


def group_qkv_rows(state_dict, num_attention_heads: int = 32, attention_head_dim: int = 64):
    """The rows of the fused to_qkv projections (weights, the per-row scales of INT8 checkpoints, biases), interleaved head by head
    (heads, q/k/v, head_dim) in the released checkpoints, grouped (q/k/v, heads, head_dim): MMGP then splits them into to_q / to_k / to_v
    as row ranges, quantized or not."""
    rows = 3 * num_attention_heads * attention_head_dim
    for key in [key for key in state_dict if key.endswith((".attn.to_qkv.weight", ".attn.to_qkv.weight_scale", ".attn.to_qkv.bias"))]:
        tensor = state_dict[key]
        if tensor.shape[0] == rows:  # a per-tensor scale stays as it is
            state_dict[key] = tensor.reshape(num_attention_heads, 3, attention_head_dim, *tensor.shape[1:]).transpose(0, 1).reshape(tensor.shape).contiguous()
    return state_dict


def get_linear_split_map(inner_dim: int = 2048):
    return {"to_qkv": {"mapped_modules": ["to_q", "to_k", "to_v"], "split_sizes": [inner_dim, inner_dim, inner_dim]}}


def _rotate_(x, rotary, first, last):
    """The split-half RoPE of the first rotary channels of FP32 rows x (batch, tokens first..last, heads, head_dim), in place."""
    cos, sin = (table[:, first:last] for table in rotary)
    half = cos.shape[-1] // 2
    a, b = x[..., :half], x[..., half:2 * half]
    rotated = a * cos[..., :half] - b * sin[..., :half]
    b.mul_(cos[..., half:]).add_(a * sin[..., half:])
    a.copy_(rotated)
    return x


def _norm_rope_(query, key, attn, rotary_emb, plan):
    """In place on query and / or key (batch, tokens, heads, head_dim; the other None): the per-head RMSNorm without weight then the
    split-half RoPE of the first rotary channels. Both together go through the fused kernel of rope_plan when there is one; alone (the
    staged quantization projects k before q), FP32 rows rounded once."""
    if query is not None and key is not None and plan[0] is not None:
        for x, done in zip((query, key), vae_ops.finish_rope_(query, key, rotary_emb, attn.norm_q.eps, plan)):
            if done is not x:
                x.copy_(done)
        return
    for x, norm in ((query, attn.norm_q), (key, attn.norm_k)):
        if x is not None:
            attention_kit.rows_(x, lambda part, first, last: (norm(part.float()) if rotary_emb is None else _rotate_(norm(part.float()), rotary_emb, first, last)).to(x.dtype))


class MiniMaxH3VideoAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __call__(
        self,
        attn: "MiniMaxH3VideoAttention",
        hidden_states: list[torch.Tensor],
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        pre_norm=None, residual_scale=None,
    ) -> torch.Tensor:
        hidden_states = _take(hidden_states)
        if hasattr(attn, "to_q") and hidden_states.dtype in _FUSED_DTYPES:
            return self._kit_attention(attn, hidden_states, rotary_emb, pre_norm, residual_scale)
        batch_size, seq_len, _ = hidden_states.shape
        plan = vae_ops.rope_plan(hidden_states, attn.dim_head, rotary_emb)

        def normalize(projected, norm):
            x_list, projected = [projected], None
            return vae_ops.normalize_projection(x_list, norm, rotary_emb, plan)

        if hasattr(attn, "to_q"):
            query = normalize(attn.to_q(hidden_states).view(batch_size, seq_len, attn.heads, attn.dim_head), attn.norm_q)
            key = normalize(attn.to_k(hidden_states).view(batch_size, seq_len, attn.heads, attn.dim_head), attn.norm_k)
            value = attn.to_v(hidden_states).view(batch_size, seq_len, attn.heads, attn.dim_head)
        else:
            qkv = (kernels.linear(attn.to_qkv, [hidden_states], input_act="rms_norm", act_weight=pre_norm.weight, act_eps=pre_norm.eps)
                   if pre_norm is not None else attn.to_qkv(hidden_states))
            qkv = qkv.view(batch_size, seq_len, attn.heads, 3, attn.dim_head)
            query, key, value = qkv.unbind(dim=3)  # strided views of the head-interleaved projection: no copy
            query = normalize(query, attn.norm_q)
            key = normalize(key, attn.norm_k)
            del qkv
        query, key = vae_ops.finish_rope_(query, key, rotary_emb, attn.norm_q.eps, plan)
        residual = hidden_states if pre_norm is not None else None
        del hidden_states

        output_dtype = query.dtype
        if output_dtype == torch.float32:
            query, key, value = query.half(), key.half(), value.half()
        # a strided query (view of the fused projection) gets a new contiguous output, which the output projection reads without a copy;
        # handed off, the projection is released once quantized, before that output is allocated
        qkv_list, recycle_q = [query, key, value], query.is_contiguous()
        query = key = value = None
        hidden_states = pay_attention(qkv_list, causal=False, recycle_q=recycle_q)
        projected = hidden_states.flatten(2, 3).to(output_dtype)
        del hidden_states
        if residual is not None:
            projected_list, projected = [projected], None
            return kernels.linear(attn.to_out, projected_list, residual=residual, residual_scale=residual_scale)
        return attn.to_out(projected)

    @staticmethod
    def _kit_attention(attn, hidden_states, rotary_emb, pre_norm, residual_scale):
        """q, k and v projected one after the other from one prepared input (attention_kit): with SageAttention 2 staged, each is
        quantized as soon as it is projected, the per-head norm and the RoPE of q and k fused with their quantization. The fused
        projection's whole output (three times the input) never exists."""
        residual = hidden_states if pre_norm is not None else None
        x = pre_norm(hidden_states) if pre_norm is not None else hidden_states
        del hidden_states
        plan = vae_ops.rope_plan(x, attn.dim_head, rotary_emb)
        # the fused kernels take the rotation as (batch, tokens, 1, pairs, cos / sin) and a weight for the norm: 1 (this one has none)
        table = None if rotary_emb is None else torch.stack((rotary_emb[0][..., :rotary_emb[0].shape[-1] // 2], rotary_emb[1][..., :rotary_emb[1].shape[-1] // 2]), dim=-1)
        unit = (torch.ones(attn.dim_head, dtype=x.dtype, device=x.device), attn.norm_q.eps)
        x_list, x = [x], None
        attention = attention_kit.qkv_attention(x_list, attn.to_q, attn.to_k, attn.to_v, attn.heads, attn.dim_head,
                                                lambda query, key, group: _norm_rope_(query, key, attn, rotary_emb, plan),
                                                fused_quant=lambda query, key, group, core, settings: kernels.split_half_norm_rope_quant_(query, key, table, unit, unit, core, settings))
        projected_list, attention = [attention.flatten(2, 3)], None
        if residual is not None:
            return kernels.linear(attn.to_out, projected_list, residual=residual, residual_scale=residual_scale)
        return attn.to_out(_take(projected_list))


class MiniMaxH3VideoAttention(nn.Module):
    _default_processor_cls = MiniMaxH3VideoAttnProcessor
    _available_processors = [MiniMaxH3VideoAttnProcessor]

    def __init__(self, dim: int, heads: int, dim_head: int, eps: float = 1e-5, bias: bool = True) -> None:
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.use_bias = bias
        inner_dim = heads * dim_head

        self.norm_q = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.norm_k = nn.RMSNorm(dim_head, eps=eps, elementwise_affine=False)
        self.to_qkv = nn.Linear(dim, 3 * inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, dim, bias=bias)

        self.processor = MiniMaxH3VideoAttnProcessor()

    def forward(
        self, hidden_states: list[torch.Tensor], rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        pre_norm=None, residual_scale=None,
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, rotary_emb, pre_norm, residual_scale)


class MiniMaxH3VideoFeedForward(nn.Module):
    def __init__(self, dim: int, mult: int, bias: bool = True) -> None:
        super().__init__()
        inner_dim = dim * mult
        self.w1 = nn.Linear(dim, 2 * inner_dim, bias=bias)
        self.w2 = nn.Linear(inner_dim, dim, bias=bias)

    def _project(self, hidden_states: list[torch.Tensor]) -> torch.Tensor:
        hidden_states = _take(hidden_states)
        expanded = self.w1(hidden_states)
        del hidden_states
        x_list, expanded = [expanded], None
        gate = vae_ops.swiglu(x_list)
        return self.w2(gate)

    def forward(self, hidden_states: list[torch.Tensor], pre_norm=None, residual_scale=None) -> torch.Tensor:
        hidden_states = _take(hidden_states)
        # chunks of whole rows of the flattened tokens (contiguous: no input or residual copies, linears with their bias fused), each
        # expanded projection about as large as the input; the output overwrites its own rows
        rows = hidden_states.view(-1, hidden_states.shape[-1])
        chunk_rows = max(1, rows.numel() // self.w1.out_features)
        for start in range(0, rows.shape[0], chunk_rows):
            x = rows[start:start + chunk_rows]
            if pre_norm is None:
                x.copy_(self._project([x]))
                continue
            expanded = kernels.linear(self.w1, [x], input_act="rms_norm", act_weight=pre_norm.weight, act_eps=pre_norm.eps)
            expanded_list, expanded = [expanded], None
            kernels.linear(self.w2, expanded_list, input_act="swiglu", residual=x, residual_scale=residual_scale, out=x)
        return hidden_states


class MiniMaxH3VideoTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        ffn_mult: int = 4,
        eps: float = 1e-5,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.norm1 = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.attn = MiniMaxH3VideoAttention(dim=dim, heads=heads, dim_head=dim_head, eps=eps, bias=bias)
        self.scale1 = nn.Parameter(torch.zeros(dim))
        self.norm2 = nn.RMSNorm(dim, eps=eps, elementwise_affine=True)
        self.ff = MiniMaxH3VideoFeedForward(dim, ffn_mult, bias=bias)
        self.scale2 = nn.Parameter(torch.zeros(dim))

    def forward(
        self, hidden_states: list[torch.Tensor], rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> torch.Tensor:
        hidden_states = _take(hidden_states)
        # split q / k / v projections (attention kit) take the normalized input; a fused one gets the norm fused into its kernel
        fused_qkv = (self.attn.to_qkv,) if hasattr(self.attn, 'to_qkv') else ()
        if (all(kernels.can_linear(layer, hidden_states, dtypes=_FUSED_DTYPES)
                for layer in (*fused_qkv, self.attn.to_out, self.ff.w1, self.ff.w2))
                and all(p.device == hidden_states.device and p.dtype == hidden_states.dtype
                        for p in (self.norm1.weight, self.norm2.weight, self.scale1, self.scale2))):
            hidden_states = self.attn([hidden_states], rotary_emb, pre_norm=self.norm1, residual_scale=self.scale1)
            return self.ff([hidden_states], pre_norm=self.norm2, residual_scale=self.scale2)
        norm_hidden_states = self.norm1(hidden_states.to(self.norm1.weight.dtype)).to(hidden_states.dtype)
        normalized_list, norm_hidden_states = [norm_hidden_states], None
        branch = self.attn(normalized_list, rotary_emb)
        branch_list, branch = [branch], None
        branch = vae_ops.scaled_residual(branch_list, self.scale1, hidden_states)
        del hidden_states
        hidden_states = branch
        norm_hidden_states = self.norm2(hidden_states.to(self.norm2.weight.dtype)).to(hidden_states.dtype)
        normalized_list, norm_hidden_states = [norm_hidden_states], None
        branch = self.ff(normalized_list)
        branch_list, branch = [branch], None
        branch = vae_ops.scaled_residual(branch_list, self.scale2, hidden_states)
        del hidden_states
        return branch


class MiniMaxH3VideoViTDecoder3d(nn.Module):
    r"""
    Non-causal ViT decoder. Every latent voxel becomes one token; `num_register_tokens` learned register tokens plus a
    single all-zero token are appended (all at position `0`), attended over with full self-attention, and dropped
    again before the patch projection expands each token into a `patch_size_t x patch_size x patch_size` pixel block.
    """

    def __init__(
        self,
        in_channels: int = 24,
        out_channels: int = 3,
        patch_size: int = 16,
        patch_size_t: int = 4,
        num_layers: int = 36,
        num_attention_heads: int = 32,
        attention_head_dim: int = 64,
        num_register_tokens: int = 4,
        ffn_mult: int = 4,
        rope_theta: float = 100.0,
        rope_dim_ratio: float = 0.75,
        norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        dim = num_attention_heads * attention_head_dim
        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = out_channels
        self.num_register_tokens = num_register_tokens

        self.rope = MiniMaxH3VideoRotaryPosEmbed(int(attention_head_dim * rope_dim_ratio), theta=rope_theta)
        self.x_embedder = nn.Linear(in_channels, dim)
        self.register_tokens = nn.Parameter(torch.zeros(1, num_register_tokens, dim))
        self.register_buffer("mask_token", torch.zeros(1, 1, dim))
        self.transformer_blocks = nn.ModuleList(
            [
                MiniMaxH3VideoTransformerBlock(
                    dim=dim,
                    heads=num_attention_heads,
                    dim_head=attention_head_dim,
                    ffn_mult=ffn_mult,
                    eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=True, eps=norm_eps)
        self.proj_out = nn.Linear(dim, out_channels * patch_size_t * patch_size * patch_size)

        self.gradient_checkpointing = False

    def forward(self, hidden_states: torch.Tensor, tokens: bool = False) -> torch.Tensor:
        """tokens: return the normalized output tokens (batch, latent positions, dim), to project and unpatchify part by part."""
        batch_size, num_channels, num_frames, height, width = hidden_states.shape

        hidden_states = hidden_states.permute(0, 2, 3, 4, 1).reshape(
            batch_size, num_frames * height * width, num_channels
        )
        hidden_states = self.x_embedder(hidden_states)
        num_patches = hidden_states.shape[1]

        register_tokens = self.register_tokens.expand(batch_size, -1, -1)
        cls_token = torch.zeros_like(hidden_states[:, :1, :])
        hidden_states = torch.cat([hidden_states, register_tokens, cls_token], dim=1)

        grids = [
            2.0 * (torch.arange(0.5, size, dtype=torch.float32, device=hidden_states.device) / size) - 1.0
            for size in (num_frames, height, width)
        ]
        position_ids = torch.stack(torch.meshgrid(*grids, indexing="ij"), dim=-1).flatten(0, 2)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1, -1)
        suffix_ids = position_ids.new_zeros((batch_size, self.num_register_tokens + 1, 3))
        position_ids = torch.cat([position_ids, suffix_ids], dim=1)
        rotary_emb = self.rope(position_ids)

        for block in self.transformer_blocks:
            if getattr(self, "_interrupt", False):
                raise GenerationInterrupted
            hidden_states = block([hidden_states], rotary_emb)

        hidden_states = self.norm_out(hidden_states)
        if tokens:
            return hidden_states[:, :num_patches, :]
        hidden_states = self.proj_out(hidden_states)
        hidden_states = hidden_states[:, :num_patches, :]
        return self.unpatchify(hidden_states, num_frames, height, width)

    def unpatchify(self, hidden_states: torch.Tensor, num_frames: int, height: int, width: int) -> torch.Tensor:
        """Output tokens (batch, latent positions, channels * patch volume) to pixels (batch, channels, frames, height, width)."""
        batch_size, patch_size, patch_size_t = hidden_states.shape[0], self.patch_size, self.patch_size_t
        hidden_states = hidden_states.view(batch_size, num_frames, height, width, self.out_channels, patch_size_t, patch_size, patch_size)
        hidden_states = hidden_states.permute(0, 4, 1, 5, 2, 6, 3, 7).contiguous()
        return hidden_states.reshape(batch_size, self.out_channels, num_frames * patch_size_t, height * patch_size, width * patch_size)


class AutoencoderKLMiniMaxH3(ModelMixin, ConfigMixin, AttentionMixin, AutoencoderMixin):
    r"""
    A VAE model with a causal 3D CNN encoder and a non-causal ViT decoder, used in
    [MiniMax-H3](https://huggingface.co/MiniMaxAI).

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Latents are normalized with per-channel `latents_mean` / `latents_std` rather than a `scaling_factor`; a pipeline
    encodes with `(latent - latents_mean) / latents_std` and decodes with `latent * latents_std + latents_mean`.

    The pixel convention is ImageNet-normalized RGB over a `[0, 1]` base range, not the usual `[-1, 1]`: `encode`
    expects `(pixel - imagenet_mean) / imagenet_std` and `decode` returns values in that same space, so a pipeline has
    to apply `sample * imagenet_std + imagenet_mean` (mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`) and
    clamp to `[0, 1]` before postprocessing.

    The temporal geometry is fixed by `clip_length` (17 pixel frames per encoder chunk) and `token_drop` (3 trailing
    latent frames dropped per encode): `17 * n + 5` pixel frames map to `5 * n + 2` latent frames.

    Unlike most autoencoders in the library, spatial tiling is **on by default**: MiniMax-H3 was released with tiling
    enabled for both encoding and decoding, and the released frames are the blended-tile ones, so disabling tiling
    changes the output. Use `enable_tiling` to change the tile geometry, `disable_tiling` to turn it off.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["MiniMaxH3VideoResnetBlock3d", "MiniMaxH3VideoTransformerBlock"]
    _repeated_blocks = ["MiniMaxH3VideoTransformerBlock"]
    _skip_layerwise_casting_patterns = ["norm"]
    # The released checkpoint is float32 and the verified decode recipe is float16 *autocast over float32 weights*
    # (see `decode`). A pipeline-level `torch_dtype=torch.bfloat16` must therefore not downcast the weights, so every
    # top-level module is pinned, mirroring the transformer's mixed-precision contract.
    _keep_in_fp32_modules = ["encoder", "decoder", "quant_conv", "post_quant_conv"]

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 24,
        block_out_channels: tuple[int, ...] = (128, 256, 256, 512, 512, 1024),
        layers_per_block: int = 2,
        spatial_downsample_factors: tuple[int, ...] = (2, 2, 2, 2, 1, 1),
        temporal_downsample_factors: tuple[int, ...] = (1, 2, 2, 1, 1, 1),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        spatial_padding_mode: str = "reflect",
        decoder_num_layers: int = 36,
        decoder_num_attention_heads: int = 32,
        decoder_attention_head_dim: int = 64,
        decoder_num_register_tokens: int = 4,
        decoder_ffn_mult: int = 4,
        decoder_rope_theta: float = 100.0,
        decoder_rope_dim_ratio: float = 0.75,
        decoder_norm_eps: float = 1e-5,
        clip_length: int = 17,
        token_drop: int = 3,
        latents_mean: tuple[float, ...] = (0.0,) * 24,
        latents_std: tuple[float, ...] = (1.0,) * 24,
    ) -> None:
        super().__init__()

        self.spatial_compression_ratio = math.prod(spatial_downsample_factors)
        self.temporal_compression_ratio = math.prod(temporal_downsample_factors)

        self.encoder = MiniMaxH3VideoEncoder3d(
            in_channels=in_channels,
            out_channels=2 * latent_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            spatial_downsample_factors=spatial_downsample_factors,
            temporal_downsample_factors=temporal_downsample_factors,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            spatial_padding_mode=spatial_padding_mode,
        )
        self.quant_conv = nn.Conv3d(2 * latent_channels, 2 * latent_channels, kernel_size=1)
        self.post_quant_conv = nn.Conv3d(latent_channels, latent_channels, kernel_size=1)
        self.decoder = MiniMaxH3VideoViTDecoder3d(
            in_channels=latent_channels,
            out_channels=out_channels,
            patch_size=self.spatial_compression_ratio,
            patch_size_t=self.temporal_compression_ratio,
            num_layers=decoder_num_layers,
            num_attention_heads=decoder_num_attention_heads,
            attention_head_dim=decoder_attention_head_dim,
            num_register_tokens=decoder_num_register_tokens,
            ffn_mult=decoder_ffn_mult,
            rope_theta=decoder_rope_theta,
            rope_dim_ratio=decoder_rope_dim_ratio,
            norm_eps=decoder_norm_eps,
        )

        # Derived temporal-chunking geometry. `clip_length` pixel frames are encoded at a time; because
        # `clip_length` is not a multiple of `temporal_compression_ratio`, the decoder has to re-derive the
        # implicit leading pad (`frame_pre_padding`) and the overlap that `token_drop` leaves behind.
        self.frame_pre_padding = (-clip_length) % self.temporal_compression_ratio
        self.tokens_chunk_size = math.ceil(clip_length / self.temporal_compression_ratio)
        self.token_overlap = (-token_drop) % self.tokens_chunk_size
        self.frame_overlap = max(self.token_overlap * self.temporal_compression_ratio - self.frame_pre_padding, 0)

        # When decoding a batch of video latents at a time, one can save memory by slicing across the batch dimension
        # to perform decoding of a single video latent at a time.
        self.use_slicing = False

        # When encoding/decoding spatially large videos, the memory requirement is very high. By splitting the frames
        # into smaller tiles, running the encoder/decoder per tile and blending the overlaps, the memory requirement
        # can be lowered. MiniMax-H3 ships with tiling enabled.
        self.use_tiling = True

        # The tile size in pixel space, and the minimum overlap between two neighbouring tiles. The actual overlaps are
        # widened (in multiples of `spatial_compression_ratio`) so that the tiles cover the frame exactly.
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_min_overlap_height = 64
        self.tile_sample_min_overlap_width = 64

    def enable_tiling(
        self,
        tile_sample_min_height: int | None = None,
        tile_sample_min_width: int | None = None,
        tile_sample_min_overlap_height: int | None = None,
        tile_sample_min_overlap_width: int | None = None,
    ) -> None:
        r"""
        Enable tiled VAE encoding/decoding. When this option is enabled, the VAE splits the frames into tiles, encodes
        or decodes each tile separately and linearly blends the overlaps back together. This lowers the memory
        requirement and allows processing larger frames.

        Args:
            tile_sample_min_height (`int`, *optional*):
                The tile height in pixel space. Frames taller than this are split along the height dimension.
            tile_sample_min_width (`int`, *optional*):
                The tile width in pixel space. Frames wider than this are split along the width dimension.
            tile_sample_min_overlap_height (`int`, *optional*):
                The minimum overlap, in pixels, between two consecutive vertical tiles.
            tile_sample_min_overlap_width (`int`, *optional*):
                The minimum overlap, in pixels, between two consecutive horizontal tiles.
        """
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_sample_min_overlap_height = tile_sample_min_overlap_height or self.tile_sample_min_overlap_height
        self.tile_sample_min_overlap_width = tile_sample_min_overlap_width or self.tile_sample_min_overlap_width

    def _split_tiles(self, length: int, tile_size: int, min_overlap: int) -> tuple[list[int], list[int], list[int]]:
        r"""
        Lay `tile_size`-wide tiles over `length` pixels. The number of tiles is the smallest one whose union can cover
        `length` while keeping every overlap at least `min_overlap`; the slack is then distributed round-robin over the
        overlaps in whole `spatial_compression_ratio` steps so that every tile boundary stays latent-aligned.
        """
        if tile_size >= length:
            return [0], [length], []

        num_tiles = math.ceil(length / tile_size)
        while tile_size * num_tiles - min_overlap * (num_tiles - 1) - length < 0:
            num_tiles += 1

        overlaps = [min_overlap] * (num_tiles - 1)
        remaining = tile_size * num_tiles - sum(overlaps) - length
        for i in range(remaining // self.spatial_compression_ratio):
            overlaps[i % (num_tiles - 1)] += self.spatial_compression_ratio

        tile_start_indices = [0]
        for i in range(num_tiles - 1):
            tile_start_indices.append(tile_start_indices[-1] + tile_size - overlaps[i])
        return tile_start_indices, [tile_size] * num_tiles, overlaps

    def _blend(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int, dim: int) -> torch.Tensor:
        blend_extent = min(a.shape[dim], b.shape[dim], blend_extent)
        positions = torch.arange(blend_extent, device=b.device, dtype=b.dtype)
        shape = [1] * a.ndim
        shape[dim] = blend_extent
        weight_a = (1 - positions / blend_extent).view(shape)
        weight_b = (positions / blend_extent).view(shape)

        slice_a = [slice(None)] * a.ndim
        slice_a[dim] = slice(-blend_extent, None)
        slice_b = [slice(None)] * b.ndim
        slice_b[dim] = slice(0, blend_extent)
        blended = a[tuple(slice_a)] * weight_a + b[tuple(slice_b)] * weight_b

        if blend_extent == b.shape[dim]:
            return blended
        slice_rest = [slice(None)] * b.ndim
        slice_rest[dim] = slice(blend_extent, None)
        return torch.cat([blended, b[tuple(slice_rest)]], dim=dim)

    def _stitch_tiles(
        self,
        tiles: list[list[torch.Tensor]],
        height_overlaps: list[int],
        width_overlaps: list[int],
    ) -> torch.Tensor:
        result_rows = []
        for i, row in enumerate(tiles):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self._blend(tiles[i - 1][j], tile, height_overlaps[i - 1], dim=-2)
                if j > 0:
                    tile = self._blend(row[j - 1], tile, width_overlaps[j - 1], dim=-1)
                if i < len(tiles) - 1:
                    tile = tile[..., : -height_overlaps[i], :]
                if j < len(row) - 1:
                    tile = tile[..., :, : -width_overlaps[j]]
                result_row.append(tile)
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    @apply_forward_hook
    def _encode_clip(self, x: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
        r"""
        Encode one temporal clip, spatially tiled when tiling is enabled. `x` may stay in RAM: each row of tiles is moved to
        `device` and prepared (`_prepare_encoder_clip`) on its own.

        MiniMax-H3 encodes a keyframe or an image reference through this method rather than through [`~encode`],
        because a single frame must not go through the temporal chunking, so it carries the offload hook too.
        """
        if not self.use_tiling:
            return self.quant_conv(self.encoder(self._prepare_encoder_clip(x, device)))

        height, width = x.shape[-2], x.shape[-1]
        y_indices, y_lengths, y_overlaps = self._split_tiles(
            height, self.tile_sample_min_height, self.tile_sample_min_overlap_height
        )
        x_indices, x_lengths, x_overlaps = self._split_tiles(
            width, self.tile_sample_min_width, self.tile_sample_min_overlap_width
        )

        rows = []
        for i_pos, i_len in zip(y_indices, y_lengths):
            band = self._prepare_encoder_clip(x[..., i_pos : i_pos + i_len, :], device)
            row = []
            for j_pos, j_len in zip(x_indices, x_lengths):
                # Compact the tile so padding does not inherit full-video strides.
                tile = band[..., j_pos : j_pos + j_len].contiguous()
                row.append(self.quant_conv(self.encoder(tile)))
            rows.append(row)
            band = tile = None

        latent_y_overlaps = [overlap // self.spatial_compression_ratio for overlap in y_overlaps]
        latent_x_overlaps = [overlap // self.spatial_compression_ratio for overlap in x_overlaps]
        return self._stitch_tiles(rows, latent_y_overlaps, latent_x_overlaps)

    def _decode_clip(self, z: torch.Tensor) -> torch.Tensor:
        r"""Decode one temporal clip, spatially tiled when tiling is enabled."""
        if not self.use_tiling:
            return self.decoder(self.post_quant_conv(z))

        # Tiles are laid out in pixel space and then mapped back onto the latent grid.
        height = z.shape[-2] * self.spatial_compression_ratio
        width = z.shape[-1] * self.spatial_compression_ratio
        y_indices, y_lengths, y_overlaps = self._split_tiles(
            height, self.tile_sample_min_height, self.tile_sample_min_overlap_height
        )
        x_indices, x_lengths, x_overlaps = self._split_tiles(
            width, self.tile_sample_min_width, self.tile_sample_min_overlap_width
        )

        ratio = self.spatial_compression_ratio
        tiles = []
        for i_pos, i_len in zip(y_indices, y_lengths):
            for j_pos, j_len in zip(x_indices, x_lengths):
                tiles.append(z[..., i_pos // ratio : i_pos // ratio + i_len // ratio,
                               j_pos // ratio : j_pos // ratio + j_len // ratio])
        batch_size = z.shape[0]
        tile_batch = torch.cat(tiles, dim=0)
        del tiles
        hidden_states = self.post_quant_conv(tile_batch)
        del tile_batch
        tile_dims = hidden_states.shape[2:]
        decoded = self.decoder(hidden_states, tokens=True)  # projected and unpatchified tile by tile below: no full-size output tensors
        del hidden_states

        canvas = torch.empty(batch_size, self.decoder.out_channels, tile_dims[0] * self.decoder.patch_size_t, height, width,
                             dtype=decoded.dtype, device=decoded.device)
        row_tails = []
        out_y = 0
        tile_index = 0
        for i in range(len(y_indices)):
            new_tails = []
            left_tail = None
            out_x = 0
            for j in range(len(x_indices)):
                tile = self.decoder.unpatchify(self.decoder.proj_out(decoded[tile_index * batch_size : (tile_index + 1) * batch_size]), *tile_dims)
                tile_index += 1
                if i > 0:
                    tile = self._blend(row_tails[j], tile, y_overlaps[i - 1], dim=-2)
                if j > 0:
                    tile = self._blend(left_tail, tile, x_overlaps[j - 1], dim=-1)
                # Preserve both-axis contributions at corners and when three spatial tiles overlap.
                if i < len(y_indices) - 1:
                    new_tails.append(tile[..., -y_overlaps[i] :, :].clone())
                next_left_tail = tile[..., :, -x_overlaps[j] :].clone() if j < len(x_indices) - 1 else None
                left_tail = next_left_tail
                if i < len(y_indices) - 1:
                    tile = tile[..., : -y_overlaps[i], :]
                if j < len(x_indices) - 1:
                    tile = tile[..., :, : -x_overlaps[j]]
                if canvas is None:
                    canvas = torch.empty(*tile.shape[:-2], height, width, dtype=tile.dtype, device=tile.device)
                canvas[..., out_y : out_y + tile.shape[-2], out_x : out_x + tile.shape[-1]].copy_(tile)
                out_x += tile.shape[-1]
            row_tails = new_tails
            out_y += tile.shape[-2]
        del decoded
        return canvas

    def _prepare_encoder_clip(self, clip: torch.Tensor, device: torch.device | None) -> torch.Tensor:
        """Convert a part of the video to encode (a row of tiles of one clip); the video itself may stay on CPU."""
        return clip.to(device=device)

    @apply_forward_hook
    def _encode(self, x: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
        r"""
        Encode a video in `clip_length`-frame chunks and drop the `token_drop` trailing latent frames.

        MiniMax-H3 encodes a video reference through this method rather than through [`~encode`], because the
        posterior is sampled under a fixed generator rather than through the distribution object, so it carries the
        offload hook too. `x` may stay in RAM (`_encode_clip` moves each row of tiles to `device`), and only the last clip
        is padded by repeating the last frame, so no full-video copy is made.
        """
        clip_length = self.config.clip_length
        clips = []
        for start in range(0, x.shape[2], clip_length):
            clip = x[:, :, start : start + clip_length]
            if clip.shape[2] < clip_length:
                clip = torch.cat([clip, x[:, :, -1:].repeat(1, 1, clip_length - clip.shape[2], 1, 1)], dim=2)
            clips.append(self._encode_clip(clip, device))
            clip = None
        moments = torch.cat(clips, dim=2)
        clips = None
        if self.config.token_drop > 0:
            moments = moments[:, :, : -self.config.token_drop]
        return moments

    def _prepare_decoded_chunk(self, chunk: torch.Tensor, uint8_rounding: str | None = None) -> torch.Tensor:
        """Convert finalized pixels before copying them to the CPU output buffer."""
        return chunk

    def _decode(self, z: torch.Tensor, uint8_rounding: str | None = None) -> torch.Tensor:
        r"""
        Decode a latent video, mirroring the chunking that `_encode` applied.

        `token_drop` removed the tail of every encoded chunk, so consecutive decoded chunks overlap by
        `frame_overlap` pixel frames and are linearly cross-faded. Latent frames are repeated at the end when the
        length is not a whole number of chunks; the extra pixel frames are cut off again at the end.
        Finalized frames are accumulated on CPU; only the current clip and its overlap stay on GPU.
        """
        tokens_chunk_size = self.tokens_chunk_size
        token_drop = self.config.token_drop
        temporal_ratio = self.temporal_compression_ratio
        chunk_num_frames = tokens_chunk_size * temporal_ratio

        if z.shape[2] == 1:
            # Fizgig H3 Still's decode recipe: five identical latents, then keep pixel frame 3.
            # https://github.com/shootthesound/ComfyUI-Fizgig-H3-Still (MIT)
            z = z.repeat(1, 1, tokens_chunk_size, 1, 1)
            num_chunks, output_frames = 0, 1
        else:
            num_tokens = z.shape[2] + token_drop
            pad_tokens = (-num_tokens) % tokens_chunk_size
            num_chunks = (num_tokens + pad_tokens) // tokens_chunk_size - int(token_drop > 0)
            if pad_tokens > 0:
                z = torch.cat([z, z[:, :, -1:].repeat(1, 1, pad_tokens, 1, 1)], dim=2)

            intra_tail = self.config.clip_length % temporal_ratio
            num_tokens_before_pad = z.shape[2] - pad_tokens
            pad_frames = sum(
                intra_tail if intra_tail and (num_tokens_before_pad + k) % tokens_chunk_size == 0 else temporal_ratio
                for k in range(pad_tokens)
            )
            output_frames = num_chunks * (chunk_num_frames - self.frame_pre_padding) + self.frame_overlap - pad_frames
        spatial_tiles = 1
        if self.use_tiling:
            height, width = z.shape[-2] * self.spatial_compression_ratio, z.shape[-1] * self.spatial_compression_ratio
            rows = self._split_tiles(height, self.tile_sample_min_height, self.tile_sample_min_overlap_height)[0]
            cols = self._split_tiles(width, self.tile_sample_min_width, self.tile_sample_min_overlap_width)[0]
            spatial_tiles = len(rows) * len(cols)
        with PhaseProgress(max(1, num_chunks) * spatial_tiles) as progress, progress.track(self.decoder, spatial_tiles):
            decoded = None
            write_position = 0
            overlap = None

            def write_chunk(chunk):
                # finalized pixels converted and copied to the CPU output by groups of frames (no full-clip conversion copies)
                nonlocal decoded, write_position
                copy_frames = min(chunk.shape[2], output_frames - write_position)
                for start in range(0, copy_frames, 4):
                    part = self._prepare_decoded_chunk(chunk[:, :, start : min(start + 4, copy_frames)], uint8_rounding)
                    if decoded is None:
                        decoded = torch.empty(*part.shape[:2], output_frames, *part.shape[3:], dtype=part.dtype, device="cpu")
                    transfer.copy_(decoded[:, :, write_position : write_position + part.shape[2]], part)
                    write_position += part.shape[2]
                    del part

            if num_chunks == 0:
                # Short videos still need one decode, without an overlapping temporal chunk.
                write_chunk(self._decode_clip(z)[:, :, self.frame_pre_padding : self.frame_pre_padding + output_frames])
                return decoded

            for i in range(num_chunks):
                start = i * tokens_chunk_size
                clip = self._decode_clip(z[:, :, start : start + tokens_chunk_size + self.token_overlap])
                for j in range(int(token_drop > 0) + 1):
                    frame_start = j * chunk_num_frames
                    chunk = clip[:, :, frame_start : frame_start + chunk_num_frames]
                    chunk = chunk[:, :, self.frame_pre_padding :]
                    if j == 0:
                        if overlap is not None:
                            chunk = self._blend(overlap, chunk, self.frame_overlap, dim=-3)
                        write_chunk(chunk)
                    else:
                        overlap = chunk.contiguous()
                del chunk, clip
            if overlap is not None:
                write_chunk(overlap)
            if write_position != output_frames:
                raise RuntimeError(f"MiniMax H3 VAE decoded {write_position} frames, expected {output_frames}")
            return decoded

    @apply_forward_hook
    def encode(self, x: torch.Tensor, return_dict: bool = True, device: torch.device | None = None) -> AutoencoderKLOutput | tuple[torch.Tensor]:
        r"""
        Encode a batch of videos into latents.

        Args:
            x (`torch.Tensor`):
                Input batch of videos, shape `(batch_size, in_channels, num_frames, height, width)`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoders.autoencoder_kl.AutoencoderKLOutput`] instead of a plain
                tuple.
            device (`torch.device`, *optional*):
                Device each clip is moved to before encoding, so `x` may stay on CPU.

        Returns:
            The latent distribution of the encoded videos. Note that MiniMax-H3 normalizes the sampled latents with
            `latents_mean` / `latents_std` afterwards.
        """
        tiles = (x.shape[0] if self.use_slicing else 1) * ((x.shape[2] + self.config.clip_length - 1) // self.config.clip_length)
        if self.use_tiling:
            rows = self._split_tiles(x.shape[-2], self.tile_sample_min_height, self.tile_sample_min_overlap_height)[0]
            cols = self._split_tiles(x.shape[-1], self.tile_sample_min_width, self.tile_sample_min_overlap_width)[0]
            tiles *= len(rows) * len(cols)
        with vae_encoding_progress(tiles, self.encoder, enabled=x.shape[2] > 1):
            if self.use_slicing and x.shape[0] > 1:
                moments = torch.cat([self._encode(x_slice, device) for x_slice in x.split(1)])
            else:
                moments = self._encode(x, device)
            posterior = DiagonalGaussianDistribution(moments)
            if not return_dict:
                return (posterior,)
            return AutoencoderKLOutput(latent_dist=posterior)

    @apply_forward_hook
    def decode(self, z: torch.Tensor, return_dict: bool = True, uint8_rounding: str | None = None) -> DecoderOutput | tuple[torch.Tensor]:
        r"""
        Decode a batch of latent videos.

        Args:
            z (`torch.Tensor`):
                Input batch of latent videos, shape `(batch_size, latent_channels, num_latent_frames, height, width)`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoders.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.autoencoders.vae.DecoderOutput`] or `tuple`:
                The decoded CPU videos, shape `(batch_size, out_channels, num_frames, height, width)`.
        """
        if self.use_slicing and z.shape[0] > 1:
            decoded = torch.cat([self._decode(z_slice, uint8_rounding) for z_slice in z.split(1)])
        else:
            decoded = self._decode(z, uint8_rounding)
        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        generator: torch.Generator | None = None,
        return_dict: bool = True,
    ) -> DecoderOutput | tuple[torch.Tensor]:
        r"""
        Encode then decode a batch of videos.

        Args:
            sample (`torch.Tensor`):
                Input batch of videos, shape `(batch_size, in_channels, num_frames, height, width)`.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample the posterior instead of taking its mode.
            generator (`torch.Generator`, *optional*):
                Generator used when `sample_posterior=True`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoders.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.autoencoders.vae.DecoderOutput`] or `tuple`:
                The round-tripped videos, shape `(batch_size, out_channels, num_frames, height, width)`.
        """
        posterior = self.encode(sample).latent_dist
        z = posterior.sample(generator=generator) if sample_posterior else posterior.mode()
        return self.decode(z, return_dict=return_dict)
