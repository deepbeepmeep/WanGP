# Copyright 2026 The Qwen Team and The HuggingFace Team. All rights reserved.
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
# Modified for WanGP.

import functools
import math

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.autoencoders.vae import AutoencoderMixin, DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.activations import get_activation
from diffusers.utils import logging
from shared.utils.phase_progress import check_abort
from shared.attention import pay_attention
from shared.image_vae import blocks, tiling
from shared.utils.conv_bands import BAND_BYTES, band_convs, band_rows, conv_bands, conv2d_weights, downsample_bands, kernel_layout
import torch
import torch.nn as nn
import torch.nn.functional as F
logger = logging.get_logger(__name__)
CACHE_T = 2

class QwenImage21AvgDown3D(nn.Module):

    def __init__(self, in_channels, out_channels, factor_t, factor_s=1):
        super().__init__()
        factor = factor_t * factor_s * factor_s
        if in_channels * factor % out_channels != 0:
            raise ValueError(f'`in_channels` ({in_channels}) times the downsampling factor ({factor}) must be divisible by `out_channels` ({out_channels}).')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = factor
        self.group_size = in_channels * factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        pad = (0, 0, 0, 0, pad_t, 0)
        x = F.pad(x, pad)
        B, C, T, H, W = x.shape
        x = x.reshape(B, C, T // self.factor_t, self.factor_t, H // self.factor_s, self.factor_s, W // self.factor_s, self.factor_s)  # channels-last or not
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(B, C * self.factor, T // self.factor_t, H // self.factor_s, W // self.factor_s)
        x = x.view(B, self.out_channels, self.group_size, T // self.factor_t, H // self.factor_s, W // self.factor_s)
        x = x.mean(dim=2)
        return x

class QwenImage21DupUp3D(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, factor_t, factor_s=1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s
        assert out_channels * self.factor % in_channels == 0
        self.repeats = out_channels * self.factor // in_channels

    def forward(self, x: torch.Tensor, first_chunk=False) -> torch.Tensor:
        x = x.repeat_interleave(self.repeats, dim=1)
        x = x.reshape(x.size(0), self.out_channels, self.factor_t, self.factor_s, self.factor_s, x.size(2), x.size(3), x.size(4))  # channels-last or not
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(x.size(0), self.out_channels, x.size(2) * self.factor_t, x.size(4) * self.factor_s, x.size(6) * self.factor_s)
        if first_chunk:
            x = x[:, :, self.factor_t - 1:, :, :]
        return x

    def add_(self, x, source, first_chunk=False):
        """x + self(source, first_chunk), added into x (the caller's own tensor) for one frame of a spatial x2: no duplicated or rearranged
        copy of source, no new sum."""
        if x.numel() * x.element_size() <= BAND_BYTES:  # small: fewer operations than position by position
            return x.add_(self(source, first_chunk=first_chunk))
        if self.factor_s == 2 and source.shape[2] == 1 and (self.factor_t == 1 or first_chunk):
            blocks.add_channel_shortcut_(x.squeeze(2), source.squeeze(2), self.repeats, 4 * self.factor_t, 4 * (self.factor_t - 1))
            return x
        return x + self(source, first_chunk=first_chunk)

class QwenImage21CausalConv3d(nn.Conv2d):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | tuple[int | int | int], stride: int | tuple[int | int | int]=1, padding: int | tuple[int | int | int]=0) -> None:
        super().__init__(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self._padding = (self.padding[1], self.padding[1], self.padding[0], self.padding[0])
        self.padding = (0, 0)

    def forward(self, x, cache_x=None, window=None, out=None, out_row=0, add=False):
        """x: one frame (batch, channels, 1, height, width), or rows of it (batch, channels, rows, width) with the same result layout.
        3x3 convolutions also take window = (first, count), the only output rows wanted, and out, written into its rows from out_row
        (added to them with add) instead of a new tensor."""
        check_abort()
        padding = list(self._padding)
        if cache_x is not None:
            raise ValueError("This convolution is the image specialization of Wan's causal 3D one: it folds the single frame away and has no temporal context to prepend, so it cannot take a feature cache.")
        frame = x.dim() == 5
        rows = x.squeeze(2) if frame else x
        # zero padding by the convolution itself (no padded copy), by bands of rows (no full-image cuDNN workspace)
        conv = functools.partial(F.conv2d, weight=self.weight, bias=self.bias, stride=self.stride, padding=(padding[2], padding[0]))
        if self.kernel_size == (3, 3) and self.stride == (1, 1) and padding[0] == padding[2] == 1:
            y = conv2d_weights(rows, self.weight, self.bias, conv, window, out, out_row, add)  # the row-window kernel where it is the faster one
        else:
            y = conv_bands(rows, conv)
        return y.unsqueeze(2) if frame else y

class QwenImage21RMS_norm(nn.Module):

    def __init__(self, dim: int, channel_first: bool=True, images: bool=True, bias: bool=False) -> None:
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        self.channel_first = channel_first
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape, device="cpu"))
        self.bias = nn.Parameter(torch.zeros(shape, device="cpu")) if bias else 0.0

    def forward(self, x, rows=False, out=None):
        """rows: x is a band of rows (batch, channels, rows, width) of one frame and the result is silu(norm(x)), in a new tensor or
        written into out (x itself: in place). The norm is per pixel: the arithmetic below on the band, its float32 temporaries
        included, instead of on the whole frame."""
        if rows:
            # x.float().norm() without the float32 copy, then the float32 quotient rounded once, straight into out: the same values in
            # fewer operations (a band is bound by how fast they are issued). PyTorch sums a pixel's channels in one order whatever
            # the number of rows when x is channels-last; for planes the order can follow the shape (1152 channels at 128 x 128): such
            # a band's norms then round as PyTorch rounds that band, not the frame.
            if out is None:
                out = torch.empty(x.shape, dtype=x.dtype, device=x.device, memory_format=torch.channels_last if x.stride(1) == 1 else torch.contiguous_format)
            torch.div(x, torch.linalg.vector_norm(x, 2.0, dim=1, keepdim=True, dtype=torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else None).clamp_min_(1e-12), out=out)
            out.mul_(self.scale).mul_(self.gamma.view(-1, 1, 1))
            if not isinstance(self.bias, float):  # (without a bias it is the number 0.0)
                out.add_(self.bias.view(-1, 1, 1))
            return F.silu(out, inplace=True)
        dim = 1 if self.channel_first else -1
        if x.dtype in (torch.float16, torch.bfloat16):
            # F.normalize(x.float()).to(x.dtype) with the same float32 norm, then the division by chunks of rows: no full-size float32 result
            denom = x.float().norm(2.0, dim, keepdim=True).clamp_min_(1e-12)
            normalized, rows = torch.empty_like(x), max(1, (64 << 20) * x.shape[-2] // (x.numel() * 4))
            for start in range(0, x.shape[-2], rows):
                normalized[..., start:start + rows, :] = x[..., start:start + rows, :] / denom[..., start:start + rows, :]
        else:
            normalized = F.normalize(x, dim=dim)
        return normalized.mul_(self.scale).mul_(self.gamma).add_(self.bias)  # in place: one full-size tensor

class QwenImage21Upsample(nn.Upsample):

    def forward(self, x):
        return super().forward(x.float()).type_as(x)

class QwenImage21Resample(nn.Module):

    def __init__(self, dim: int, mode: str, upsample_out_dim: int=None) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode
        if upsample_out_dim is None:
            upsample_out_dim = dim // 2
        if mode == 'upsample2d':
            self.resample = nn.Sequential(QwenImage21Upsample(scale_factor=(2.0, 2.0), mode='nearest-exact'), nn.Conv2d(dim, upsample_out_dim, 3, padding=1))
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(QwenImage21Upsample(scale_factor=(2.0, 2.0), mode='nearest-exact'), nn.Conv2d(dim, upsample_out_dim, 3, padding=1))
            self.time_conv = QwenImage21CausalConv3d(dim, dim * 2, (1, 1), padding=(0, 0))
        elif mode == 'downsample2d':
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(nn.ZeroPad2d((0, 1, 0, 1)), nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = QwenImage21CausalConv3d(dim, dim, (1, 1), stride=(1, 1), padding=(0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_idx is None:
            feat_idx = [0]
        b, c, t, h, w = x.size()
        if self.mode == 'upsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and (feat_cache[idx] != 'Rep'):
                        cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                    if cache_x.shape[2] < 2 and feat_cache[idx] is not None and (feat_cache[idx] == 'Rep'):
                        cache_x = torch.cat([torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2)
                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        if self.mode in ('upsample2d', 'upsample3d'):  # nearest x2 + conv by bands of rows: no full-size upsampled input
            x = conv_bands(x, self.resample[1], upsample=True, mode='nearest-exact')
        elif self.mode in ('downsample2d', 'downsample3d'):  # the padded stride-2 convolution by bands of rows: no padded copy of the input
            x = downsample_bands(x, self.resample[1])
        else:
            x = self.resample(x)
        x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)
        if self.mode == 'downsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x

class QwenImage21ResidualBlock(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, dropout: float=0.0) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.nonlinearity = get_activation('silu')
        self.norm1 = QwenImage21RMS_norm(in_dim, images=False)
        self.conv1 = QwenImage21CausalConv3d(in_dim, out_dim, 3, padding=1)
        self.norm2 = QwenImage21RMS_norm(out_dim, images=False)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = QwenImage21CausalConv3d(out_dim, out_dim, 3, padding=1)
        self.conv_shortcut = QwenImage21CausalConv3d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def _ops(self):
        return (functools.partial(self.norm1, rows=True), self.conv1, functools.partial(self.norm2, rows=True), self.conv2,
                None if isinstance(self.conv_shortcut, nn.Identity) else self.conv_shortcut)

    def rows(self, x):
        """This block on the rows of one frame, x a tensor (batch, channels, height, width) or blocks.Rows: its output as rows
        computed when read."""
        return blocks.ResnetRows(x, *self._ops())

    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_idx is None:
            feat_idx = [0]
        if feat_cache is None and x.shape[2] == 1:  # a picture: by bands of rows, with only x and the result at full size
            return blocks.pixel_resnet_block(x.squeeze(2), *self._ops()).unsqueeze(2)
        h = self.conv_shortcut(x)
        x = self.norm1(x)
        x = self.nonlinearity(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)
        x = self.norm2(x)
        x = self.nonlinearity(x)
        x = self.dropout(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv2(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv2(x)
        return x.add_(h)

class QwenImage21AttentionBlock(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.norm = QwenImage21RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        identity = x
        batch_size, channels, time, height, width = x.size()
        x = x.permute(0, 2, 1, 3, 4).reshape(batch_size * time, channels, height, width)
        x = self.norm(x)
        qkv = self.to_qkv(x)
        qkv = qkv.reshape(batch_size * time, 1, channels * 3, -1)
        qkv = qkv.permute(0, 1, 3, 2).contiguous()
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        del qkv, x
        if q.shape[1] > 8192:
            # The encoder attends over the assembled bottleneck. Bound the
            # attention matrix at 4K without changing its global key/value set.
            attended = torch.empty_like(q)
            for start in range(0, q.shape[1], 256):
                check_abort()
                stop = min(start + 256, q.shape[1])
                attended[:, start:stop] = pay_attention([q[:, start:stop], k, v], force_attention="sdpa")
            x = attended.transpose(1, 2)
        else:
            x = pay_attention([q, k, v], force_attention="sdpa").transpose(1, 2)
        del q, k, v
        x = x.squeeze(1).permute(0, 2, 1).reshape(batch_size * time, channels, height, width)
        x = self.proj(x)
        x = x.view(batch_size, time, channels, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        return x.add_(identity)

class QwenImage21MidBlock(nn.Module):

    def __init__(self, dim: int, dropout: float=0.0, num_layers: int=1):
        super().__init__()
        self.dim = dim
        resnets = [QwenImage21ResidualBlock(dim, dim, dropout)]
        attentions = []
        for _ in range(num_layers):
            attentions.append(QwenImage21AttentionBlock(dim))
            resnets.append(QwenImage21ResidualBlock(dim, dim, dropout))
        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)
        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_idx is None:
            feat_idx = [0]
        x = self.resnets[0](x, feat_cache=feat_cache, feat_idx=feat_idx)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                x = attn(x)
            x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
        return x

class QwenImage21ResidualDownBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout, num_res_blocks, temperal_downsample=False, down_flag=False):
        super().__init__()
        self.avg_shortcut = QwenImage21AvgDown3D(in_dim, out_dim, factor_t=2 if temperal_downsample else 1, factor_s=2 if down_flag else 1)
        resnets = []
        for _ in range(num_res_blocks):
            resnets.append(QwenImage21ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim
        self.resnets = nn.ModuleList(resnets)
        if down_flag:
            mode = 'downsample3d' if temperal_downsample else 'downsample2d'
            self.downsampler = QwenImage21Resample(out_dim, mode=mode)
        else:
            self.downsampler = None

    def forward(self, x, feat_cache=None, feat_idx=None):
        """x as blocks.Rows (a picture's level computed by rows when read; this level must downsample): its blocks and its
        downsampling are rows in turn, and none of their activations exists whole."""
        if feat_idx is None:
            feat_idx = [0]
        if isinstance(x, blocks.Rows):
            rows = x
            for resnet in self.resnets:
                rows = resnet.rows(rows)
            x.retain = 2 * len(self.resnets) + 2  # the shortcut reads x again, that many rows behind the main path at most
            return blocks.DownsampledRows(rows, self.downsampler.resample[1], x, lambda source: self.avg_shortcut(source.unsqueeze(2)).squeeze(2))
        x_copy = x
        for resnet in self.resnets:
            x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
        if self.downsampler is not None:
            x = self.downsampler(x, feat_cache=feat_cache, feat_idx=feat_idx)
        return x + self.avg_shortcut(x_copy)

class QwenImage21Encoder3d(nn.Module):

    def _chain(self, x):
        """(number of first levels computed as a chain of rows, rows per band of the input) for the picture x: the levels that
        downsample, from the first on, while their activation takes more than blocks.CHAIN_MIN_BANDS bands and their share of a band
        keeps the size from which the row-window convolution is used (blocks.chain_levels' rule)."""
        n, _, _, height, width = x.shape
        count = band_rows(height, n * self.conv_in.out_channels * width * x.element_size())
        chain, channels = 0, self.conv_in.out_channels
        for block in self.down_blocks:
            if not (isinstance(block, QwenImage21ResidualDownBlock) and block.downsampler is not None and height % 2 == 0 and width % 2 == 0
                    and n * channels * height * width * x.element_size() > blocks.CHAIN_MIN_BANDS * BAND_BYTES and (count >> chain) * width >= blocks.CHAIN_MIN_PIXELS):
                break
            chain, channels, height, width = chain + 1, block.downsampler.dim, height // 2, width // 2
        return chain, count

    def __init__(self, in_channels: int=3, dim=128, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[], temperal_downsample=[True, True, False], dropout=0.0, is_residual: bool=False):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.nonlinearity = get_activation('silu')
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0
        self.conv_in = QwenImage21CausalConv3d(in_channels, dims[0], 3, padding=1)
        self.down_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if is_residual:
                self.down_blocks.append(QwenImage21ResidualDownBlock(in_dim, out_dim, dropout, num_res_blocks, temperal_downsample=temperal_downsample[i] if i != len(dim_mult) - 1 else False, down_flag=i != len(dim_mult) - 1))
            else:
                for _ in range(num_res_blocks):
                    self.down_blocks.append(QwenImage21ResidualBlock(in_dim, out_dim, dropout))
                    if scale in attn_scales:
                        self.down_blocks.append(QwenImage21AttentionBlock(out_dim))
                    in_dim = out_dim
                if i != len(dim_mult) - 1:
                    mode = 'downsample3d' if temperal_downsample[i] else 'downsample2d'
                    self.down_blocks.append(QwenImage21Resample(out_dim, mode=mode))
                    scale /= 2.0
        self.mid_block = QwenImage21MidBlock(out_dim, dropout, num_layers=1)
        self.norm_out = QwenImage21RMS_norm(out_dim, images=False)
        self.conv_out = QwenImage21CausalConv3d(out_dim, z_dim, 3, padding=1)
        self.gradient_checkpointing = False

    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_idx is None:
            feat_idx = [0]
        x = kernel_layout(x)
        chain, count = self._chain(x) if feat_cache is None and x.shape[2] == 1 else (0, 0)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        elif chain:  # a large picture: the first convolution and the large levels as rows, whole from the first small level on
            rows = blocks.ConvRows(x.squeeze(2), self.conv_in)
            for layer in self.down_blocks[:chain]:
                rows = layer(rows)
            x = blocks.collect(rows, max(1, count >> chain)).unsqueeze(2)
            del rows
        else:
            x = self.conv_in(x)
        for layer in self.down_blocks[chain:]:
            if feat_cache is not None:
                x = layer(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = layer(x)
        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)
        x = self.norm_out(x)
        x = self.nonlinearity(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x.contiguous()  # the callers' layout, whatever the convolutions ran in

class QwenImage21ResidualUpBlock(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, num_res_blocks: int, dropout: float=0.0, temperal_upsample: bool=False, up_flag: bool=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        if up_flag:
            self.avg_shortcut = QwenImage21DupUp3D(in_dim, out_dim, factor_t=2 if temperal_upsample else 1, factor_s=2)
        else:
            self.avg_shortcut = None
        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(QwenImage21ResidualBlock(current_dim, out_dim, dropout))
            current_dim = out_dim
        self.resnets = nn.ModuleList(resnets)
        if up_flag:
            upsample_mode = 'upsample3d' if temperal_upsample else 'upsample2d'
            self.upsampler = QwenImage21Resample(out_dim, mode=upsample_mode, upsample_out_dim=out_dim)
        else:
            self.upsampler = None
        self.gradient_checkpointing = False

    def chainable(self, first_chunk):
        """Whether this level's output can be rows computed when read (blocks.Rows): one frame of a spatial x2, or no upsampling."""
        shortcut = self.avg_shortcut
        return self.upsampler is None or shortcut.factor_s == 2 and (shortcut.factor_t == 1 or first_chunk)

    def _upsampled_rows(self, x, source):
        shortcut = self.avg_shortcut
        return blocks.UpsampledRows(x, self.upsampler.resample[1], 'nearest-exact', source, shortcut.repeats, 4 * shortcut.factor_t, 4 * (shortcut.factor_t - 1))

    def forward(self, x_list, feat_cache=None, feat_idx=None, first_chunk=False, lazy=False):
        """x_list: [input], emptied so that the caller holds no reference to it (released once the first block has read it).
        lazy (a picture, a chainable level): the upsampling and its shortcut are returned as rows to compute when read
        (blocks.Rows) instead of the x2 activation. A level that receives such rows continues the chain: its blocks and its own
        upsampling are rows in turn, and none of their activations exists whole."""
        if feat_idx is None:
            feat_idx = [0]
        x = x_list.pop()
        if isinstance(x, blocks.Rows):
            rows = x
            for resnet in self.resnets:
                rows = resnet.rows(rows)
            if self.upsampler is None:
                return rows
            x.retain = 2 * len(self.resnets) + 2  # the shortcut reads x again, that many rows behind the main path at most
            return self._upsampled_rows(rows, x)
        x_copy = x if self.avg_shortcut is not None else None
        for resnet in self.resnets:
            if feat_cache is not None:
                x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = resnet(x)
        if self.upsampler is not None:
            if lazy:
                return self._upsampled_rows(x.squeeze(2), x_copy.squeeze(2))
            if feat_cache is not None:
                x = self.upsampler(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = self.upsampler(x)
        if self.avg_shortcut is not None:
            x = self.avg_shortcut.add_(x, x_copy, first_chunk)  # into the upsampler's output
        return x

class QwenImage21UpBlock(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, num_res_blocks: int, dropout: float=0.0, upsample_mode: str | None=None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        resnets = []
        current_dim = in_dim
        for _ in range(num_res_blocks + 1):
            resnets.append(QwenImage21ResidualBlock(current_dim, out_dim, dropout))
            current_dim = out_dim
        self.resnets = nn.ModuleList(resnets)
        self.upsamplers = None
        if upsample_mode is not None:
            self.upsamplers = nn.ModuleList([QwenImage21Resample(out_dim, mode=upsample_mode)])
        self.gradient_checkpointing = False

    def forward(self, x_list, feat_cache=None, feat_idx=None, first_chunk=None):
        """x_list: [input], emptied so that the caller holds no reference to it."""
        if feat_idx is None:
            feat_idx = [0]
        x = x_list.pop()
        for resnet in self.resnets:
            if feat_cache is not None:
                x = resnet(x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = resnet(x)
        if self.upsamplers is not None:
            if feat_cache is not None:
                x = self.upsamplers[0](x, feat_cache=feat_cache, feat_idx=feat_idx)
            else:
                x = self.upsamplers[0](x)
        return x

class QwenImage21Decoder3d(nn.Module):

    def __init__(self, dim=128, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[], temperal_upsample=[False, True, True], dropout=0.0, out_channels: int=3, is_residual: bool=False):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample
        self.nonlinearity = get_activation('silu')
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        self.conv_in = QwenImage21CausalConv3d(z_dim, dims[0], 3, padding=1)
        self.mid_block = QwenImage21MidBlock(dims[0], dropout, num_layers=1)
        self.up_blocks = nn.ModuleList([])
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            if i > 0 and (not is_residual):
                in_dim = in_dim // 2
            up_flag = i != len(dim_mult) - 1
            upsample_mode = None
            if up_flag and temperal_upsample[i]:
                upsample_mode = 'upsample3d'
            elif up_flag:
                upsample_mode = 'upsample2d'
            if is_residual:
                up_block = QwenImage21ResidualUpBlock(in_dim=in_dim, out_dim=out_dim, num_res_blocks=num_res_blocks, dropout=dropout, temperal_upsample=temperal_upsample[i] if up_flag else False, up_flag=up_flag)
            else:
                up_block = QwenImage21UpBlock(in_dim=in_dim, out_dim=out_dim, num_res_blocks=num_res_blocks, dropout=dropout, upsample_mode=upsample_mode)
            self.up_blocks.append(up_block)
        self.norm_out = QwenImage21RMS_norm(out_dim, images=False)
        self.conv_out = QwenImage21CausalConv3d(out_dim, out_channels, 3, padding=1)
        self.gradient_checkpointing = False

    def _chain(self, x, first_chunk):
        """(first level computed as a chain of rows, rows per band of the output) for the picture x entering the levels; no level:
        len(up_blocks) (blocks.chain_levels)."""
        n, _, _, height, width = x.shape
        levels, size, widest = [], None, 0
        for index, block in enumerate(self.up_blocks):  # a level and those after it must be chainable, and so the one before it
            levels.append(size if all(isinstance(b, QwenImage21ResidualUpBlock) and b.chainable(first_chunk) for b in self.up_blocks[max(index - 1, 0):]) else None)
            if getattr(block, "upsampler", None) is not None:
                height, width, widest = 2 * height, 2 * width, block.out_dim
                size = (height, width, widest)
            else:
                size = None
        return blocks.chain_levels(levels, (height, width, widest), n, x.element_size())

    def forward(self, x, feat_cache=None, feat_idx=None, first_chunk=False, sink=None):
        """sink (a picture): called with (first row, rows (batch, channels, 1, rows, width)) for the output, band after band down the
        image where the last levels run as a chain of rows, else once with the whole output; nothing is returned then."""
        if feat_idx is None:
            feat_idx = [0]
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_in(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_in(x)
        x = self.mid_block(x, feat_cache=feat_cache, feat_idx=feat_idx)
        picture = feat_cache is None and x.shape[2] == 1
        chain, count = self._chain(x, first_chunk) if picture else (len(self.up_blocks), 0)
        for level, up_block in enumerate(self.up_blocks):
            x_list, x = [x], None
            x = up_block(x_list, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk, **({"lazy": True} if level + 1 == chain else {}))
        if isinstance(x, blocks.Rows):  # the large levels as rows: the output convolution reads them band after band
            rows = blocks.NormConvRows(x, functools.partial(self.norm_out, rows=True), self.conv_out)
            x = None
            if sink is None:
                return blocks.collect(rows, count).unsqueeze(2)
            for first, _, band in blocks.bands(rows, count):
                sink(first, band.unsqueeze(2))
            return None
        if picture:  # the output convolution by bands of rows, no normalized copy
            x = blocks.pixel_norm_conv(x.squeeze(2), functools.partial(self.norm_out, rows=True), self.conv_out).unsqueeze(2)
            return x if sink is None else sink(0, x)
        x = self.norm_out(x)
        x = self.nonlinearity(x)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
            x = self.conv_out(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv_out(x)
        return x

def _patchify(x, patch_size):
    if patch_size == 1:
        return x
    if x.dim() != 5:
        raise ValueError(f'Invalid input shape: {x.shape}')
    batch_size, channels, frames, height, width = x.shape
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(f'Height ({height}) and width ({width}) must be divisible by patch_size ({patch_size})')
    x = x.view(batch_size, channels, frames, height // patch_size, patch_size, width // patch_size, patch_size)
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    x = x.view(batch_size, channels * patch_size * patch_size, frames, height // patch_size, width // patch_size)
    return x

def _unpatchify(x, patch_size):
    if patch_size == 1:
        return x
    if x.dim() != 5:
        raise ValueError(f'Invalid input shape: {x.shape}')
    batch_size, c_patches, frames, height, width = x.shape
    channels = c_patches // (patch_size * patch_size)
    x = x.reshape(batch_size, channels, patch_size, patch_size, frames, height, width)  # channels-last or not
    x = x.permute(0, 1, 4, 5, 3, 6, 2).contiguous()
    x = x.view(batch_size, channels, frames, height * patch_size, width * patch_size)
    return x

class AutoencoderKLQwenImage21(ModelMixin, ConfigMixin, AutoencoderMixin):

    # measured VRAM (shared.image_vae.tiling): one frame, BF16, the larger of the routes with and without the Triton convolutions, + 10 %
    COSTS = {"weights": 0.63, "decode": [(0.065, 0.2), (0.26, 0.45), (1.05, 0.8), (4.2, 1.45)], "encode": [(1.05, 0.8), (4.2, 1.15)]}

    @staticmethod
    def get_VAE_tile_size(vae_config, device_mem_capacity, mixed_precision, output_height=None, output_width=None):
        """(tiling, tile size in pixels): the largest tile within the VAE Tiling setting's share of the VRAM, no tiling when the whole
        picture fits (tiling.choose_tile)."""
        tile = tiling.choose_tile(vae_config, device_mem_capacity, AutoencoderKLQwenImage21.COSTS, output_height, output_width, (2048, 1024, 512, 256))
        return tile > 0, tile or 256

    def configure_tiling(self, tile_setting, width, height, batch_size=1, reference_pixels=0):
        if tile_setting is None or (isinstance(tile_setting, int) and not isinstance(tile_setting, bool)):
            capacity = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1048576 if torch.cuda.is_available() else 0
            enabled, size = self.get_VAE_tile_size(tile_setting or 0, capacity, False, height, width)
        elif isinstance(tile_setting, bool):
            enabled, size = tile_setting, 256
        else:
            enabled, size = tile_setting
        if enabled is None:  # Older Python callers supplied the Auto sentinel.
            enabled = True
        self.use_tiling = bool(enabled)
        if enabled:
            self.enable_tiling(tile_sample_min_height=size, tile_sample_min_width=size, tile_sample_stride_height=size * 3 // 4, tile_sample_stride_width=size * 3 // 4)
    _supports_gradient_checkpointing = False
    _group_offload_block_modules = ['quant_conv', 'post_quant_conv', 'encoder', 'decoder']
    _skip_keys = ['feat_cache', 'feat_idx']

    @register_to_config
    def __init__(self, base_dim: int=96, decoder_base_dim: int | None=144, z_dim: int=64, dim_mult: list[int]=[1, 2, 4, 8, 8], num_res_blocks: int=2, attn_scales: list[float]=[], temperal_downsample: list[bool]=[False, True, True, True], dropout: float=0.0, latents_mean: list[float]=[0.5126, 0.7721, -0.0631, 1.3506, -0.7855, -2.1025, -0.3458, 1.3722, 1.8873, -1.7177, -0.651, 0.2732, 0.7562, -0.6163, -1.0277, 3.8363, 2.021, 0.0472, 0.932, 2.0087, 2.4954, -0.1391, -1.4249, 1.8464, -0.5236, 1.2826, 3.7046, -1.3035, 2.7286, -1.4518, -1.9036, -1.9955, -0.0342, -1.0265, -0.7636, 3.0555, 0.0746, -3.0751, -0.1076, 1.7376, -1.0914, -1.9435, -0.2784, -1.368, 0.4809, -0.4433, 0.3764, 0.5729, -2.0595, 1.096, -1.326, -2.0211, -5.0179, 0.5275, 4.0162, 1.8505, 0.3026, 1.9373, 1.4937, 0.2632, 0.5547, -1.7121, -0.1562, 0.0304], latents_std: list[float]=[3.2001, 3.2936, 3.4321, 3.0091, 3.1061, 4.0379, 4.0705, 3.791, 3.0785, 3.65, 3.9308, 3.0904, 2.8778, 3.7675, 3.732, 5.0756, 3.2864, 4.0397, 3.1317, 4.0443, 2.9249, 3.9454, 3.0988, 4.2489, 3.4896, 3.8513, 3.9323, 3.4719, 3.7498, 4.283, 3.5694, 4.2467, 3.9037, 3.2947, 5.077, 3.5075, 3.27, 3.4767, 2.8063, 5.1125, 3.5327, 4.7833, 3.1286, 4.1819, 3.8527, 3.8312, 3.5605, 4.3875, 3.9624, 4.0168, 3.5643, 4.055, 5.5614, 4.2963, 4.408, 3.4959, 3.8747, 3.7608, 3.5735, 3.149, 3.7662, 3.6746, 3.4563, 3.8161], is_residual: bool=True, in_channels: int=4, out_channels: int=4, patch_size: int | None=None, scale_factor_temporal: int | None=8, scale_factor_spatial: int | None=16) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]
        if decoder_base_dim is None:
            decoder_base_dim = base_dim
        self.encoder = QwenImage21Encoder3d(in_channels=in_channels, dim=base_dim, z_dim=z_dim * 2, dim_mult=dim_mult, num_res_blocks=num_res_blocks, attn_scales=attn_scales, temperal_downsample=temperal_downsample, dropout=dropout, is_residual=is_residual)
        self.quant_conv = QwenImage21CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.post_quant_conv = QwenImage21CausalConv3d(z_dim, z_dim, 1)
        self.decoder = QwenImage21Decoder3d(dim=decoder_base_dim, z_dim=z_dim, dim_mult=dim_mult, num_res_blocks=num_res_blocks, attn_scales=attn_scales, temperal_upsample=self.temperal_upsample, dropout=dropout, out_channels=out_channels, is_residual=is_residual)
        for module in self.modules():  # every activation follows a normalization (a fresh tensor): in place, without a second full-size tensor
            if isinstance(module, nn.SiLU):
                module.inplace = True
        band_convs(self)  # the upsampling convolutions: by rows, or the row-window kernel
        self.spatial_compression_ratio = scale_factor_spatial
        self.use_slicing = False
        self.use_tiling = False
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192
        self._cached_conv_counts = {'decoder': sum((isinstance(m, QwenImage21CausalConv3d) for m in self.decoder.modules())) if self.decoder is not None else 0, 'encoder': sum((isinstance(m, QwenImage21CausalConv3d) for m in self.encoder.modules())) if self.encoder is not None else 0}

    def enable_tiling(self, tile_sample_min_height: int | None=None, tile_sample_min_width: int | None=None, tile_sample_stride_height: float | None=None, tile_sample_stride_width: float | None=None) -> None:
        self.use_tiling = True
        self.tile_sample_min_height = tile_sample_min_height or self.tile_sample_min_height
        self.tile_sample_min_width = tile_sample_min_width or self.tile_sample_min_width
        self.tile_sample_stride_height = tile_sample_stride_height or self.tile_sample_stride_height
        self.tile_sample_stride_width = tile_sample_stride_width or self.tile_sample_stride_width

    def clear_cache(self):
        self._conv_num = self._cached_conv_counts['decoder']
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        self._enc_conv_num = self._cached_conv_counts['encoder']
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num

    def _encode(self, x: torch.Tensor):
        _, _, num_frame, height, width = x.shape
        self.clear_cache()
        if self.config.patch_size is not None:
            x = _patchify(x, patch_size=self.config.patch_size)
        if self.use_tiling and (width > min(self.tile_sample_min_width, 256) or height > min(self.tile_sample_min_height, 256)):
            return self.tiled_encode(x)
        iter_ = 1 + (num_frame - 1) // 4
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out = self.encoder(x[:, :, :1, :, :], feat_cache=None, feat_idx=self._enc_conv_idx)
            else:
                out_ = self.encoder(x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :], feat_cache=None, feat_idx=self._enc_conv_idx)
                out = torch.cat([out, out_], 2)
        enc = self.quant_conv(out)
        self.clear_cache()
        return enc

    def encode(self, x: torch.Tensor, return_dict: bool=True) -> AutoencoderKLOutput | tuple[DiagonalGaussianDistribution]:
        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [self._encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = self._encode(x)
        posterior = DiagonalGaussianDistribution(h)
        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(self, z: torch.Tensor, return_dict: bool=True):
        _, _, num_frame, height, width = z.shape
        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        if self.use_tiling and (width > tile_latent_min_width or height > tile_latent_min_height):
            return self.tiled_decode(z, return_dict=return_dict)
        self.clear_cache()
        x = self.post_quant_conv(z)
        for i in range(num_frame):
            self._conv_idx = [0]
            if i == 0:
                out = self.decoder(x[:, :, i:i + 1, :, :], feat_cache=None, feat_idx=self._conv_idx, first_chunk=True)
            else:
                out_ = self.decoder(x[:, :, i:i + 1, :, :], feat_cache=None, feat_idx=self._conv_idx)
                out = torch.cat([out, out_], 2)
        if self.config.patch_size is not None:
            out = _unpatchify(out, patch_size=self.config.patch_size)
        out = torch.clamp(out, min=-1.0, max=1.0)
        self.clear_cache()
        if not return_dict:
            return (out,)
        return DecoderOutput(sample=out)

    def decode(self, z: torch.Tensor, return_dict: bool=True) -> DecoderOutput | torch.Tensor:
        if self.use_slicing and z.shape[0] > 1:
            decoded_slices = [self._decode(z_slice).sample for z_slice in z.split(1)]
            decoded = torch.cat(decoded_slices)
        else:
            decoded = self._decode(z).sample
        if not return_dict:
            return (decoded,)
        return DecoderOutput(sample=decoded)

    def _blend(self, a, b, blend_extent, dimension):
        extent = min(a.shape[dimension], b.shape[dimension], blend_extent)
        if extent == 0:
            return b
        # Zero endpoint weights suppress padding artifacts from each tile's edge.
        ramp = torch.linspace(0, 1, extent, device=b.device, dtype=b.dtype)
        ramp = ((ramp - 0.125) / 0.75).clamp_(0, 1)
        ramp = (1 - torch.cos(ramp * math.pi)) * 0.5
        shape = [1] * b.ndim
        shape[dimension] = extent
        ramp = ramp.reshape(shape)
        previous = a.narrow(dimension, a.shape[dimension] - extent, extent)
        current = b.narrow(dimension, 0, extent)
        current.mul_(ramp).add_(previous * (1 - ramp))
        return b

    def blend_v(self, a, b, blend_extent):
        return self._blend(a, b, blend_extent, 3)

    def blend_h(self, a, b, blend_extent):
        return self._blend(a, b, blend_extent, 4)

    def tiled_encode(self, x: torch.Tensor) -> torch.Tensor:
        _, _, num_frames, height, width = x.shape
        if num_frames != 1 or self.config.patch_size is not None:
            raise ValueError("Qwen Image 2.1 tiled VAE encoding expects one unpatched image.")
        ratio = self.spatial_compression_ratio
        core_height = self.tile_sample_min_height
        core_width = self.tile_sample_min_width
        if any(size % ratio for size in (height, width, core_height, core_width)):
            raise ValueError("Qwen Image 2.1 tiled VAE encoding requires dimensions aligned to its spatial compression ratio.")
        # The down path has a 281-pixel receptive field. The 160-pixel halo
        # gives every retained feature its full local context.
        halo = 160
        features = None
        for top in range(0, height, core_height):
            for left in range(0, width, core_width):
                patch_top = max(0, top - halo)
                patch_left = max(0, left - halo)
                patch_bottom = min(height, top + core_height + halo)
                patch_right = min(width, left + core_width + halo)
                patch = x[:, :, :, patch_top:patch_bottom, patch_left:patch_right]
                local = self.encoder.conv_in(patch)
                for block in self.encoder.down_blocks:
                    local = block(local)
                if features is None:
                    features = local.new_empty((local.shape[0], local.shape[1], local.shape[2], height // ratio, width // ratio))
                output_height = min(core_height, height - top) // ratio
                output_width = min(core_width, width - left) // ratio
                local_top = (top - patch_top) // ratio
                local_left = (left - patch_left) // ratio
                features[:, :, :, top // ratio:top // ratio + output_height, left // ratio:left // ratio + output_width] = (
                    local[:, :, :, local_top:local_top + output_height, local_left:local_left + output_width]
                )
                del patch, local
        # The bottleneck attention must see the entire image, not one tile.
        features = self.encoder.mid_block(features)
        features = self.encoder.norm_out(features)
        features = self.encoder.nonlinearity(features)
        enc = self.quant_conv(self.encoder.conv_out(features))
        self.clear_cache()
        return enc

    def tiled_decode(self, z: torch.Tensor, return_dict: bool=True) -> DecoderOutput | torch.Tensor:
        _, _, num_frames, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio
        tile_latent_min_height = self.tile_sample_min_height // self.spatial_compression_ratio
        tile_latent_min_width = self.tile_sample_min_width // self.spatial_compression_ratio
        tile_latent_stride_height = self.tile_sample_stride_height // self.spatial_compression_ratio
        tile_latent_stride_width = self.tile_sample_stride_width // self.spatial_compression_ratio
        tile_sample_stride_height = self.tile_sample_stride_height
        tile_sample_stride_width = self.tile_sample_stride_width
        if self.config.patch_size is not None:
            sample_height = sample_height // self.config.patch_size
            sample_width = sample_width // self.config.patch_size
            tile_sample_stride_height = tile_sample_stride_height // self.config.patch_size
            tile_sample_stride_width = tile_sample_stride_width // self.config.patch_size
            blend_height = self.tile_sample_min_height // self.config.patch_size - tile_sample_stride_height
            blend_width = self.tile_sample_min_width // self.config.patch_size - tile_sample_stride_width
        else:
            blend_height = self.tile_sample_min_height - tile_sample_stride_height
            blend_width = self.tile_sample_min_width - tile_sample_stride_width
        rows = []
        for i in range(0, height, tile_latent_stride_height):
            row = []
            for j in range(0, width, tile_latent_stride_width):
                self.clear_cache()
                time = []
                for k in range(num_frames):
                    self._conv_idx = [0]
                    tile = z[:, :, k:k + 1, i:i + tile_latent_min_height, j:j + tile_latent_min_width]
                    tile = self.post_quant_conv(tile)
                    decoded = self.decoder(tile, feat_cache=None, feat_idx=self._conv_idx, first_chunk=k == 0)
                    time.append(decoded)
                row.append(torch.cat(time, dim=2))
            rows.append(row)
        self.clear_cache()
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                result_row.append(tile[:, :, :, :tile_sample_stride_height, :tile_sample_stride_width])
            result_rows.append(torch.cat(result_row, dim=-1))
        dec = torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]
        if self.config.patch_size is not None:
            dec = _unpatchify(dec, patch_size=self.config.patch_size)
        dec = torch.clamp(dec, min=-1.0, max=1.0)
        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)

    @torch.inference_mode()
    def decode_to_cpu_uint8(self, latents, output_channels=None):
        """Stream tiles into one CPU RGB/RGBA byte buffer, retaining only overlap edges.

        Like Qwen v1, weights remain owned by MMGP and overlap state lives on CPU.
        Blending precedes patch unpacking and quantization, matching float decoding.
        """
        output_channels = self.config.out_channels if output_channels is None else output_channels
        if output_channels not in (3, self.config.out_channels):
            raise ValueError("Qwen Image 2.1 output must have RGB or RGBA channels.")
        z = latents.pop() if isinstance(latents, list) else latents
        if z.shape[2] != 1:
            raise ValueError("Qwen Image 2.1 decodes single-frame images.")
        device = z.device
        batch, _, _, height, width = z.shape
        scale = self.spatial_compression_ratio
        patch = self.config.patch_size or 1
        full_height, full_width = height * scale, width * scale
        tile_h, tile_w = height, width
        stride_h, stride_w = height, width
        tiled = self.use_tiling and (full_height > self.tile_sample_min_height or full_width > self.tile_sample_min_width)
        if tiled:
            tile_h, tile_w = self.tile_sample_min_height // scale, self.tile_sample_min_width // scale
            stride_h, stride_w = self.tile_sample_stride_height // scale, self.tile_sample_stride_width // scale
        blend_h, blend_w = (tile_h - stride_h) * scale // patch, (tile_w - stride_w) * scale // patch
        # Single-frame inference never reuses temporal feature caches. Encoder
        # and decoder calls below therefore pass feat_cache=None.
        source = z.detach().cpu() if tiled else z
        del z, latents
        output = torch.empty((batch, output_channels, 1, full_height, full_width), dtype=torch.uint8, device="cpu")
        previous_edges, row_edges = [], []
        left_edge = None
        try:
            for batch_index in range(batch):
                previous_edges = []
                for row_index, y in enumerate(range(0, height, stride_h)):
                    row_edges, left_edge = [], None
                    for column_index, x in enumerate(range(0, width, stride_w)):
                        check_abort()
                        self.clear_cache()
                        tile_latents = source[batch_index:batch_index + 1, :, :, y:y + tile_h, x:x + tile_w].to(device=device)
                        tile = self.post_quant_conv(tile_latents)
                        if not tiled:
                            # no blending: the decoder hands its rows over as they are decoded and they become bytes at once, a quarter of
                            # the output's 16-bit size (16 MB at 2048 x 2048), which reach RAM in one transfer: a copy per band waits for
                            # the GPU each time (13 % slower), and asynchronous ones cost more to set up than one picture's bands gain
                            pixels = torch.empty((1, output_channels, 1, full_height, full_width), dtype=torch.uint8, device=device)

                            def sink(first, rows):
                                rows = _unpatchify(rows, patch_size=patch) if patch != 1 else rows
                                pixels[:, :, :, first * patch:first * patch + rows.shape[3]].copy_(rows[:, :output_channels].float().add_(1).mul_(127.5).clamp_(0, 255))

                            self.decoder(tile, feat_cache=None, feat_idx=self._conv_idx, first_chunk=True, sink=sink)
                            output[batch_index:batch_index + 1].copy_(pixels)
                            del tile, tile_latents, pixels
                            continue
                        tile = self.decoder(tile, feat_cache=None, feat_idx=self._conv_idx, first_chunk=True)
                        del tile_latents
                        if row_index:
                            edge = previous_edges[column_index].to(device=device)
                            self.blend_v(edge, tile, blend_h)
                            previous_edges[column_index] = None
                            del edge
                        if left_edge is not None:
                            edge = left_edge.to(device=device)
                            self.blend_h(edge, tile, blend_w)
                            del edge, left_edge
                        # Only the bottom strip of the previous row and the right strip survive.
                        bottom = tile[:, :, :, -blend_h:, :].detach().cpu() if blend_h and y + stride_h < height else None
                        left_edge = tile[:, :, :, :, -blend_w:].detach().cpu() if blend_w and x + stride_w < width else None
                        row_edges.append(bottom)
                        write_h, write_w = min(stride_h, height - y) * scale, min(stride_w, width - x) * scale
                        tile = tile[:, :, :, :write_h // patch, :write_w // patch]
                        if patch != 1:
                            tile = _unpatchify(tile, patch_size=patch)
                        # Quantize the small RGBA tile in FP32 even for BF16
                        # inference, avoiding a second low-precision rounding.
                        pixels = tile[:, :output_channels].float().add_(1).mul_(127.5).clamp_(0, 255).to(torch.uint8).cpu()
                        output[batch_index:batch_index + 1, :, :, y * scale:y * scale + write_h, x * scale:x * scale + write_w].copy_(pixels)
                        del tile, pixels, bottom
                    previous_edges = row_edges
                previous_edges.clear()
            return output
        finally:
            previous_edges.clear()
            row_edges.clear()
            left_edge = None
            self.clear_cache()

    def forward(self, sample: torch.Tensor, sample_posterior: bool=False, return_dict: bool=True, generator: torch.Generator | None=None) -> DecoderOutput | torch.Tensor:
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z, return_dict=return_dict)
        return dec
