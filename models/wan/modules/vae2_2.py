# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modifications Copyright (c) 2025-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import logging

import torch
from shared.utils.phase_progress import vae_encoding_progress, PhaseProgress, check_abort, set_phase_status
from shared.cuda_memory import mmgp_vram_allocator
from shared.kernels import vae_ops
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from .vae import _blend_h_edge_, _blend_v_edge_, _vae_float_to_cpu_uint8, _kernels, _use_fast, _causal_input, _residual_fast, _resample_fast, _run_part, _exact_segments, _exact_calls, _exact_decode, _exact_encode, _attention_fast

__all__ = [
    "Wan2_2_VAE",
]

CACHE_T = 2


class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolusion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0,
        )
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None, padded=False, residual=None):
        if padded:  # fast path: x is already the padded channels-last input; residual is added to the output
            if _kernels.conv_eligible(x, self):
                return _kernels.conv(x, self.weight, self.bias, residual)
            x = F.conv3d(x, _kernels.channels_last_weight(self.weight), self.bias, self.stride)  # cuDNN, without PyTorch's slower weight layout copy
            return x if residual is None else x.add_(residual)
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
            cache_x = None
        x = F.pad(x, padding)
        try:
            out = super().forward(x)
            return out
        except RuntimeError as e:
            if "miopenStatus" in str(e):
                print("⚠️ MIOpen fallback: AMD gets upset when trying to work with large areas, and so CPU will be "
                      "used for this decoding (which is very slow). Consider using tiled VAE Decoding.")
                x_cpu = x.float().cpu()
                weight_cpu = self.weight.float().cpu()
                bias_cpu = self.bias.float().cpu() if self.bias is not None else None
                print(f"[Fallback] x shape: {x_cpu.shape}, weight shape: {weight_cpu.shape}")
                out = F.conv3d(x_cpu, weight_cpu, bias_cpu,
                               self.stride, (0, 0, 0),  # avoid double padding here
                               self.dilation, self.groups)
                out = out.to(x.device)
                if x.dtype in (torch.float16, torch.bfloat16):
                    out = out.half()
                if x.dtype != out.dtype:
                    out = out.to(x.dtype)
                return out
            raise
class RMS_norm(nn.Module):

    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x, pad=None, cache=None, keep=0):
        if pad is not None:  # fast path: silu(norm(x)) as the padded input of the next causal convolution
            return _kernels.causal_pad(x, pad, cache, self, keep)
        return (F.normalize(x, dim=(1 if self.channel_first else -1)) *
                self.scale * self.gamma + self.bias)


class Upsample(nn.Upsample):

    def forward(self, x):
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):

    def __init__(self, dim, mode):
        assert mode in (
            "none",
            "upsample2d",
            "upsample3d",
            "downsample2d",
            "downsample3d",
        )
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2.0, 2.0), mode="nearest-exact"),
                nn.Conv2d(dim, dim, 3, padding=1),
                # nn.Conv2d(dim, dim//2, 3, padding=1)
            )
            self.time_conv = CausalConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))
        elif mode == "downsample2d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == "downsample3d":
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0], fast=False):
        if fast:
            return _resample_fast(self, x, feat_cache, feat_idx)
        b, c, t, h, w = x.size()
        if self.mode == "upsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = "Rep"
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -CACHE_T:, :, :].clone()
                    if (cache_x.shape[2] < 2 and feat_cache[idx] is not None and
                            feat_cache[idx] != "Rep"):
                        # cache last frame of last two chunk
                        cache_x = torch.cat(
                            [
                                feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                    cache_x.device),
                                cache_x,
                            ],
                            dim=2,
                        )
                    if (cache_x.shape[2] < 2 and feat_cache[idx] is not None and
                            feat_cache[idx] == "Rep"):
                        cache_x = torch.cat(
                            [
                                torch.zeros_like(cache_x).to(cache_x.device),
                                cache_x
                            ],
                            dim=2,
                        )
                    if feat_cache[idx] == "Rep":
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]),
                                    3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", t=t)

        if self.mode == "downsample3d":
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x.clone()
                    feat_idx[0] += 1
                else:
                    cache_x = x[:, :, -1:, :, :].clone()
                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1
        return x

    def init_weight(self, conv):
        conv_weight = conv.weight.detach().clone()
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        conv_weight.data[:, :, 1, 0, 0] = init_matrix  # * 0.5
        conv.weight = nn.Parameter(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        conv_weight = conv.weight.data.detach().clone()
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        conv_weight[:c1 // 2, :, -1, 0, 0] = init_matrix
        conv_weight[c1 // 2:, :, -1, 0, 0] = init_matrix
        conv.weight = nn.Parameter(conv_weight)
        nn.init.zeros_(conv.bias.data)


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False),
            nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1),
        )
        self.shortcut = (
            CausalConv3d(in_dim, out_dim, 1)
            if in_dim != out_dim else nn.Identity())

    def forward(self, x, feat_cache=None, feat_idx=[0], fast=False):
        if fast:
            return _residual_fast(self, x, feat_cache, feat_idx)
        h = self.shortcut(x)
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x + h


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x, fast=False):
        if fast:
            return _attention_fast(self, x)
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.norm(x)
        # compute query, key, value
        q, k, v = (
            self.to_qkv(x).reshape(b * t, 1, c * 3,
                                   -1).permute(0, 1, 3,
                                               2).contiguous().chunk(3, dim=-1))

        # apply attention (by chunks of queries where SDPA has no memory-efficient kernel for the VAE's dtype)
        x = vae_ops.attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        # output
        x = self.proj(x)
        x = rearrange(x, "(b t) c h w-> b c t h w", t=t)
        return x + identity


def patchify(x, patch_size):
    if patch_size == 1:
        return x
    if x.dim() == 4:
        x = rearrange(
            x, "b c (h q) (w r) -> b (c r q) h w", q=patch_size, r=patch_size)
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b c f (h q) (w r) -> b (c r q) f h w",
            q=patch_size,
            r=patch_size,
        )
    else:
        raise ValueError(f"Invalid input shape: {x.shape}")

    return x


def unpatchify(x, patch_size):
    if patch_size == 1:
        return x

    if x.dim() == 4:
        x = rearrange(
            x, "b (c r q) h w -> b c (h q) (w r)", q=patch_size, r=patch_size)
    elif x.dim() == 5:
        x = rearrange(
            x,
            "b (c r q) f h w -> b c f (h q) (w r)",
            q=patch_size,
            r=patch_size,
        )
    return x


class AvgDown3D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        factor_t,
        factor_s=1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.factor_t = factor_t
        self.factor_s = factor_s
        self.factor = self.factor_t * self.factor_s * self.factor_s

        assert in_channels * self.factor % out_channels == 0
        self.group_size = in_channels * self.factor // out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_t = (self.factor_t - x.shape[2] % self.factor_t) % self.factor_t
        pad = (0, 0, 0, 0, pad_t, 0)
        x = F.pad(x, pad)
        B, C, T, H, W = x.shape
        x = x.view(
            B,
            C,
            T // self.factor_t,
            self.factor_t,
            H // self.factor_s,
            self.factor_s,
            W // self.factor_s,
            self.factor_s,
        )
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        x = x.view(
            B,
            C * self.factor,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )
        x = x.view(
            B,
            self.out_channels,
            self.group_size,
            T // self.factor_t,
            H // self.factor_s,
            W // self.factor_s,
        )
        x = x.mean(dim=2)
        return x


class DupUp3D(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor_t,
        factor_s=1,
    ):
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
        x = x.view(
            x.size(0),
            self.out_channels,
            self.factor_t,
            self.factor_s,
            self.factor_s,
            x.size(2),
            x.size(3),
            x.size(4),
        )
        x = x.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        x = x.view(
            x.size(0),
            self.out_channels,
            x.size(2) * self.factor_t,
            x.size(4) * self.factor_s,
            x.size(6) * self.factor_s,
        )
        if first_chunk:
            x = x[:, :, self.factor_t - 1:, :, :]
        return x


class Down_ResidualBlock(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 dropout,
                 mult,
                 temperal_downsample=False,
                 down_flag=False):
        super().__init__()

        # Shortcut path with downsample
        self.avg_shortcut = AvgDown3D(
            in_dim,
            out_dim,
            factor_t=2 if temperal_downsample else 1,
            factor_s=2 if down_flag else 1,
        )

        # Main path with residual blocks and downsample
        downsamples = []
        for _ in range(mult):
            downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim

        # Add the final downsample block
        if down_flag:
            mode = "downsample3d" if temperal_downsample else "downsample2d"
            downsamples.append(Resample(out_dim, mode=mode))

        self.downsamples = nn.Sequential(*downsamples)

    def forward(self, x, feat_cache=None, feat_idx=[0], fast=False):
        if fast:  # the modules leave their input unchanged: no copy, the shortcut added in place
            shortcut = x
            for module in self.downsamples:
                x = module(x, feat_cache, feat_idx, fast=True)
            return x.add_(self.avg_shortcut(shortcut))
        x_copy = x.clone()
        for module in self.downsamples:
            x = module(x, feat_cache, feat_idx)

        return x + self.avg_shortcut(x_copy)


class Up_ResidualBlock(nn.Module):

    def __init__(self,
                 in_dim,
                 out_dim,
                 dropout,
                 mult,
                 temperal_upsample=False,
                 up_flag=False):
        super().__init__()
        # Shortcut path with upsample
        if up_flag:
            self.avg_shortcut = DupUp3D(
                in_dim,
                out_dim,
                factor_t=2 if temperal_upsample else 1,
                factor_s=2 if up_flag else 1,
            )
        else:
            self.avg_shortcut = None

        # Main path with residual blocks and upsample
        upsamples = []
        for _ in range(mult):
            upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
            in_dim = out_dim

        # Add the final upsample block
        if up_flag:
            mode = "upsample3d" if temperal_upsample else "upsample2d"
            upsamples.append(Resample(out_dim, mode=mode))

        self.upsamples = nn.Sequential(*upsamples)

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False, fast=False):
        if fast:  # the modules leave their input unchanged: no copy, the shortcut added in place
            x_main = x
            for module in self.upsamples:
                x_main = module(x_main, feat_cache, feat_idx, fast=True)
            return x_main if self.avg_shortcut is None else x_main.add_(self.avg_shortcut(x, first_chunk))
        x_main = x.clone()
        for module in self.upsamples:
            x_main = module(x_main, feat_cache, feat_idx)
        if self.avg_shortcut is not None:
            x_shortcut = self.avg_shortcut(x, first_chunk)
            return x_main + x_shortcut
        else:
            return x_main


class Encoder3d(nn.Module):

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(12, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_down_flag = (
                temperal_downsample[i]
                if i < len(temperal_downsample) else False)
            downsamples.append(
                Down_ResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks,
                    temperal_downsample=t_down_flag,
                    down_flag=i != len(dim_mult) - 1,
                ))
            scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout),
            AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout),
        )

        # # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1),
        )

    def _stages(self):
        return [self.conv1, *self.downsamples, *self.middle, self.head]

    def _forward_fast(self, x, feat_cache, feat_idx):
        x = self.conv1(_causal_input(x, self.conv1, feat_cache, feat_idx), padded=True)
        for layer in self.downsamples:
            x = layer(x, feat_cache, feat_idx, fast=True)
        for layer in self.middle:
            x = layer(x, feat_cache, feat_idx, fast=True) if isinstance(layer, ResidualBlock) else layer(x, fast=True)
        x = self.head[2](_causal_input(x, self.head[2], feat_cache, feat_idx, self.head[0]), padded=True)
        return x.contiguous()

    def forward(self, x, feat_cache=None, feat_idx=[0], part=None):
        if part is not None:  # exact tiled encoding: some stages, on the whole frame or a region
            return _run_part(self, x, part, feat_cache, feat_idx)
        if _use_fast(x):
            return self._forward_fast(x, feat_cache, feat_idx)

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        ## downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)

        return x


class Decoder3d(nn.Module):

    def __init__(
        self,
        dim=128,
        z_dim=4,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_upsample=[False, True, True],
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)
        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout),
            AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout),
        )

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            t_up_flag = temperal_upsample[i] if i < len(
                temperal_upsample) else False
            upsamples.append(
                Up_ResidualBlock(
                    in_dim=in_dim,
                    out_dim=out_dim,
                    dropout=dropout,
                    mult=num_res_blocks + 1,
                    temperal_upsample=t_up_flag,
                    up_flag=i != len(dim_mult) - 1,
                ))
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False),
            nn.SiLU(),
            CausalConv3d(out_dim, 12, 3, padding=1),
        )

    def _stages(self):
        return [self.conv1, *self.middle, *self.upsamples, self.head]

    def _forward_fast(self, x, feat_cache, feat_idx, first_chunk):
        x = self.conv1(_causal_input(x, self.conv1, feat_cache, feat_idx), padded=True)
        for layer in self.middle:
            check_abort()
            x = layer(x, feat_cache, feat_idx, fast=True) if isinstance(layer, ResidualBlock) else layer(x, fast=True)
        for layer in self.upsamples:
            check_abort()
            x = layer(x, feat_cache, feat_idx, first_chunk, fast=True)
        check_abort()
        x = self.head[2](_causal_input(x, self.head[2], feat_cache, feat_idx, self.head[0]), padded=True)
        return x.contiguous()

    def forward(self, x, feat_cache=None, feat_idx=[0], first_chunk=False, part=None):
        if part is not None:  # exact tiled decoding: some stages, on the whole frame or a region
            return _run_part(self, x, part, feat_cache, feat_idx)
        if _use_fast(x):
            return self._forward_fast(x, feat_cache, feat_idx, first_chunk)
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                cache_x = torch.cat(
                    [
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device),
                        cache_x,
                    ],
                    dim=2,
                )
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)

        for layer in self.middle:
            check_abort()
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## upsamples
        for layer in self.upsamples:
            check_abort()
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx, first_chunk)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            check_abort()
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


class WanVAE_(nn.Module):
    _offload_hooks = ['encode', 'decode']
    def __init__(
        self,
        dim=160,
        dec_dim=256,
        z_dim=16,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, False],
        dropout=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]

        # modules
        self.encoder = Encoder3d(
            dim,
            z_dim * 2,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_downsample,
            dropout,
        )
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(
            dec_dim,
            z_dim,
            dim_mult,
            num_res_blocks,
            attn_scales,
            self.temperal_upsample,
            dropout,
        )

    def forward(self, x, scale=[0, 1]):
        mu = self.encode(x, scale)
        x_recon = self.decode(mu, scale)
        return x_recon, mu

    def encode(self, x, scale = None, any_end_frame = False):
        self.clear_cache()
        x = patchify(x, patch_size=2)        
        ## cache
        t = x.shape[2]
        if any_end_frame:
            iter_ = 2 + (t - 2) // 4
        else:
            iter_ = 1 + (t - 1) // 4
        ## 对encode输入的x，按时间拆分为1、4、4、4....
        out_list = []
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out_list.append(self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx))
            elif any_end_frame and i== iter_ -1:
                out_list.append(self.encoder(
                    x[:, :, -1:, :, :],
                    feat_cache= None,
                    feat_idx=self._enc_conv_idx))
            else:
                out_list.append(self.encoder(
                    x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx))

        self.clear_cache()
        out = torch.cat(out_list, 2)
        out_list = None

        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if scale != None:
            if isinstance(scale[0], torch.Tensor):
                mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                    1, self.z_dim, 1, 1, 1)
            else:
                mu = (mu - scale[0]) * scale[1]
        return mu


    def decode(self, z, scale=None, any_end_frame = False):
        self.clear_cache()
        # z: [b,c,t,h,w]
        if scale != None:
            if isinstance(scale[0], torch.Tensor):
                z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                    1, self.z_dim, 1, 1, 1)
            else:
                z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        out_list = []
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out_list.append(self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx,
                    first_chunk = True)
                    )
            elif any_end_frame and i==iter_-1:
                out_list.append(self.decoder(
                    x[:, :, -1:, :, :],
                    feat_cache=None ,
                    feat_idx=self._conv_idx))
            else:
                out_list.append(self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx))
        self.clear_cache()
        out = torch.cat(out_list, 2)
        out = unpatchify(out, patch_size=2)        
        return out

   
    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (y / blend_extent)
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (x / blend_extent)
        return b
    
    def spatial_tiled_encode(self, x, scale, tile_size, any_end_frame=False):
        """Tiled encoding equal to the untiled one, fast path only (tiles in patches, half the pixels)."""
        return _exact_encode(self, patchify(x, patch_size=2), scale, tile_size // 2, any_end_frame)

    def spatial_tiled_decode(self, z, scale, tile_size, any_end_frame= False):
        tile_sample_min_size = tile_size
        tile_latent_min_size = int(tile_sample_min_size / 16)
        tile_overlap_factor = 0.25

        # z: [b,c,t,h,w]

        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(
                1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        if _use_fast(z):  # tiles with halos: the untiled result (decoder tiles in patches, half the pixels)
            out = torch.empty((z.shape[0], 3, (z.shape[2] - 2) * 4 + 2 if any_end_frame else (z.shape[2] - 1) * 4 + 1, z.shape[-2] * 16, z.shape[-1] * 16), dtype=z.dtype, device=z.device)

            def sink(i, offset, region, y):
                out[:, :, offset:offset + y.shape[2], region[0][0] * 2:region[0][1] * 2, region[1][0] * 2:region[1][1] * 2] = unpatchify(y.contiguous(), patch_size=2)
            _exact_decode(self, z, _exact_segments(self.decoder, tuple(z.shape[-2:]), tile_size // 2, True), any_end_frame, sink)
            return out

        overlap_size = int(tile_latent_min_size * (1 - tile_overlap_factor)) #8 0.75
        blend_extent = int(tile_sample_min_size * tile_overlap_factor) #256 0.25
        row_limit = tile_sample_min_size - blend_extent

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, z.shape[-2], overlap_size):
            row = []
            for j in range(0, z.shape[-1], overlap_size):
                tile = z[:, :, :, i: i + tile_latent_min_size, j: j + tile_latent_min_size]
                decoded = self.decode(tile, any_end_frame= any_end_frame)
                row.append(decoded)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))

        return torch.cat(result_rows, dim=-2)

    def decode_tile_chunks(self, z, any_end_frame=False):
        self.clear_cache()
        x = self.conv2(z)
        frame_start = 0
        try:
            for i in range(x.shape[2]):
                self._conv_idx = [0]
                if i == 0:
                    tile = self.decoder(x[:, :, i:i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx, first_chunk=True)
                elif any_end_frame and i == x.shape[2] - 1:
                    tile = self.decoder(x[:, :, -1:, :, :], feat_cache=None, feat_idx=self._conv_idx)
                else:
                    tile = self.decoder(x[:, :, i:i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
                tile = unpatchify(tile, patch_size=2)
                yield frame_start, tile
                frame_start += int(tile.shape[2])
                del tile
        finally:
            del x
            self.clear_cache()

    def decode_to_cpu_uint8(self, z, scale=None, tile_size=0, target_frames=None, target_height=None, target_width=None, any_end_frame=False, device=None):
        device = torch.device(device) if device is not None else (scale[0].device if scale is not None and isinstance(scale[0], torch.Tensor) else z.device)
        dtype = getattr(self, "_model_dtype", z.dtype)
        tile_size = int(tile_size or 0)
        latent_source = z.detach()
        if tile_size > 0 and latent_source.device.type != "cpu":
            latent_source = latent_source.to("cpu")
        decoded_frame_count = 0 if latent_source.shape[2] <= 0 else (int(latent_source.shape[2]) - 1) * 4 + 1
        target_frames = decoded_frame_count if target_frames is None else min(int(target_frames), decoded_frame_count)
        needed_latents = 0 if target_frames <= 0 else min(int(latent_source.shape[2]), (max(int(target_frames), 1) - 1 + 3) // 4 + 1)
        latent_source = latent_source[:, :, :needed_latents]
        full_height = latent_source.shape[-2] * 16
        full_width = latent_source.shape[-1] * 16
        target_height = full_height if target_height is None else min(int(target_height), full_height)
        target_width = full_width if target_width is None else min(int(target_width), full_width)
        if target_frames <= 0:
            return torch.empty((latent_source.shape[0], 3, 0, target_height, target_width), dtype=torch.uint8, device="cpu")
        spatial_tiles = 1
        segments = _exact_segments(self.decoder, tuple(latent_source.shape[-2:]), tile_size // 2, True) if tile_size > 0 and _use_fast(torch.empty(0, dtype=dtype, device=device)) else None
        if segments is not None:  # tiles with halos (the untiled result): whole-frame calls, then the tiles
            spatial_tiles = _exact_calls(segments, 1)
        elif tile_size > 0:
            stride = max(1, int(max(1, tile_size // 16) * 0.75))
            row_limit = max(1, tile_size - int(tile_size * 0.25))
            rows = min(len(range(0, latent_source.shape[-2], stride)), (target_height + row_limit - 1) // row_limit)
            cols = min(len(range(0, latent_source.shape[-1], stride)), (target_width + row_limit - 1) // row_limit)
            spatial_tiles = rows * cols
        with PhaseProgress(spatial_tiles * needed_latents) as progress, progress.track(self.decoder):
            try:
                if scale is not None and isinstance(scale[0], torch.Tensor):
                    scale = [u.to(device=device) for u in scale]
                if segments is not None:
                    z = latent_source.to(device=device, dtype=dtype)
                    if scale is not None:
                        z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(1, self.z_dim, 1, 1, 1) if isinstance(scale[0], torch.Tensor) else z / scale[1] + scale[0]
                    decoded = torch.empty((latent_source.shape[0], 3, target_frames, target_height, target_width), dtype=torch.uint8, device="cpu")

                    def sink(i, offset, region, y):
                        (r0, r1), (c0, c1) = ((lo * 2, min(hi * 2, limit)) for (lo, hi), limit in zip(region, (target_height, target_width)))
                        b = min(offset + y.shape[2], target_frames)
                        if offset < b and r0 < r1 and c0 < c1:
                            decoded[:, :, offset:b, r0:r1, c0:c1] = _vae_float_to_cpu_uint8(unpatchify(y[:, :, :b - offset].contiguous(), patch_size=2)[..., :r1 - r0, :c1 - c0])
                    _exact_decode(self, z, segments, any_end_frame, sink)
                    return decoded
                if tile_size <= 0:
                    z = latent_source.to(device=device, dtype=dtype)
                    frames = self.decode(z, scale, any_end_frame=any_end_frame)[:, :, :target_frames, :target_height, :target_width]
                    decoded = _vae_float_to_cpu_uint8(frames)
                    del z, frames
                    return decoded

                tile_sample_min_size = tile_size
                tile_latent_min_size = max(1, int(tile_sample_min_size / 16))
                tile_overlap_factor = 0.25
                overlap_size = max(1, int(tile_latent_min_size * (1 - tile_overlap_factor)))
                blend_extent = int(tile_sample_min_size * tile_overlap_factor)
                row_limit = max(1, tile_sample_min_size - blend_extent)
                decoded = torch.empty((latent_source.shape[0], 3, target_frames, target_height, target_width), dtype=torch.uint8, device="cpu")
                previous_row_edges = []
                row_index = 0
                for latent_y in range(0, latent_source.shape[-2], overlap_size):
                    current_row_edges = []
                    left_edge = None
                    col_index = 0
                    write_y0 = row_index * row_limit
                    write_y1 = min(write_y0 + row_limit, target_height)
                    has_next_row = write_y1 < target_height
                    if write_y1 <= write_y0:
                        break
                    for latent_x in range(0, latent_source.shape[-1], overlap_size):
                        write_x0 = col_index * row_limit
                        write_x1 = min(write_x0 + row_limit, target_width)
                        has_next_col = write_x1 < target_width
                        if write_x1 <= write_x0:
                            break
                        tile_latents = latent_source[:, :, :, latent_y:latent_y + tile_latent_min_size, latent_x:latent_x + tile_latent_min_size].to(device=device, dtype=dtype)
                        if scale is not None:
                            if isinstance(scale[0], torch.Tensor):
                                tile_latents.div_(scale[1].view(1, self.z_dim, 1, 1, 1)).add_(scale[0].view(1, self.z_dim, 1, 1, 1))
                            else:
                                tile_latents.div_(scale[1]).add_(scale[0])
                        bottom_edge = None
                        right_edge = None
                        previous_edge = previous_row_edges[col_index] if row_index > 0 and col_index < len(previous_row_edges) else None
                        for frame_start, tile in self.decode_tile_chunks(tile_latents, any_end_frame=any_end_frame):
                            if frame_start >= target_frames:
                                break
                            frame_end = min(frame_start + int(tile.shape[2]), target_frames)
                            tile = tile[:, :, :frame_end - frame_start]
                            if previous_edge is not None:
                                _blend_v_edge_(previous_edge[:, :, frame_start:frame_end], tile, blend_extent)
                            if left_edge is not None:
                                _blend_h_edge_(left_edge[:, :, frame_start:frame_end], tile, blend_extent)
                            if has_next_row:
                                edge = tile[:, :, :, -min(blend_extent, tile.shape[-2]):, :].detach().cpu()
                                if bottom_edge is None:
                                    bottom_edge = torch.empty((edge.shape[0], edge.shape[1], target_frames, edge.shape[3], edge.shape[4]), dtype=edge.dtype, device="cpu")
                                bottom_edge[:, :, frame_start:frame_end].copy_(edge)
                                del edge
                            if has_next_col:
                                edge = tile[:, :, :, :, -min(blend_extent, tile.shape[-1]):].detach().cpu()
                                if right_edge is None:
                                    right_edge = torch.empty((edge.shape[0], edge.shape[1], target_frames, edge.shape[3], edge.shape[4]), dtype=edge.dtype, device="cpu")
                                right_edge[:, :, frame_start:frame_end].copy_(edge)
                                del edge
                            tile = tile[:, :, :, :write_y1 - write_y0, :write_x1 - write_x0]
                            decoded[:, :, frame_start:frame_end, write_y0:write_y0 + tile.shape[-2], write_x0:write_x0 + tile.shape[-1]].copy_(_vae_float_to_cpu_uint8(tile))
                            del tile
                        current_row_edges.append(bottom_edge)
                        left_edge = right_edge
                        del tile_latents, previous_edge
                        col_index += 1
                    left_edge = None
                    previous_row_edges = current_row_edges
                    row_index += 1
                return decoded

            finally:
                self.clear_cache()

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, imgs, deterministic=False):
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        # cache encode
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


def _video_vae(pretrained_path=None, z_dim=16, dim=160, device="cpu", *, dtype, **kwargs):
    # params
    cfg = dict(
        dim=dim,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[True, True, True],
        dropout=0.0,
    )
    cfg.update(**kwargs)

    # init model
    with torch.device("meta"):
        model = WanVAE_(**cfg)

    from mmgp import offload
    # load checkpoint
    logging.info(f"loading {pretrained_path}")
    # model.load_state_dict(
    #     torch.load(pretrained_path, map_location=device), assign=True)
    # offload.save_model(model, "Wan_vae_2_2.safetensors")
    # model.to(torch.bfloat16)
    # offload.save_model(model, "Wan_vae_2_2_bf16.safetensors")
    offload.load_model_data(model, pretrained_path.replace(".pth", ".safetensors"), writable_tensors=False, default_dtype=dtype)

    return model


class Wan2_2_VAE:

    def __init__(
        self,
        z_dim=48,
        c_dim=160,
        vae_pth=None,
        dim_mult=[1, 2, 4, 4],
        temperal_downsample=[False, True, True],
        dtype=torch.float,
        upsampler_factor=1,
        device="cuda",
    ):

        self.dtype = dtype
        self.device = device
        self.vae_pth = vae_pth
        assert upsampler_factor == 1
        mean = torch.tensor(
            [
                -0.2289,
                -0.0052,
                -0.1323,
                -0.2339,
                -0.2799,
                0.0174,
                0.1838,
                0.1557,
                -0.1382,
                0.0542,
                0.2813,
                0.0891,
                0.1570,
                -0.0098,
                0.0375,
                -0.1825,
                -0.2246,
                -0.1207,
                -0.0698,
                0.5109,
                0.2665,
                -0.2108,
                -0.2158,
                0.2502,
                -0.2055,
                -0.0322,
                0.1109,
                0.1567,
                -0.0729,
                0.0899,
                -0.2799,
                -0.1230,
                -0.0313,
                -0.1649,
                0.0117,
                0.0723,
                -0.2839,
                -0.2083,
                -0.0520,
                0.3748,
                0.0152,
                0.1957,
                0.1433,
                -0.2944,
                0.3573,
                -0.0548,
                -0.1681,
                -0.0667,
            ],
            dtype=dtype,
            device=device,
        )
        std = torch.tensor(
            [
                0.4765,
                1.0364,
                0.4514,
                1.1677,
                0.5313,
                0.4990,
                0.4818,
                0.5013,
                0.8158,
                1.0344,
                0.5894,
                1.0901,
                0.6885,
                0.6165,
                0.8454,
                0.4978,
                0.5759,
                0.3523,
                0.7135,
                0.6804,
                0.5833,
                1.4146,
                0.8986,
                0.5659,
                0.7069,
                0.5338,
                0.4889,
                0.4917,
                0.4069,
                0.4999,
                0.6866,
                0.4093,
                0.5709,
                0.6065,
                0.6415,
                0.4944,
                0.5726,
                1.2042,
                0.5458,
                1.6887,
                0.3971,
                1.0600,
                0.3943,
                0.5537,
                0.5444,
                0.4089,
                0.7468,
                0.7744,
            ],
            dtype=dtype,
            device=device,
        )
        self.scale = [mean, 1.0 / std]

        # init model
        self.model = (
            _video_vae(
                pretrained_path=vae_pth,
                dtype=dtype,
                z_dim=z_dim,
                dim=c_dim,
                dim_mult=dim_mult,
                temperal_downsample=temperal_downsample,
            ).eval().requires_grad_(False).to(device))
        
        self.model._model_dtype = dtype


    @staticmethod
    def get_VAE_tile_size(vae_config, device_mem_capacity, mixed_precision):
        if mmgp_vram_allocator():  # no allocator fragmentation to guard against: Default and Low VRAM levels
            return 256 if vae_config == 3 or device_mem_capacity < 8000 else 512
        # VAE Tiling
        if vae_config == 0:
            if mixed_precision:
                device_mem_capacity = device_mem_capacity / 2
            if device_mem_capacity >= 24000:
                use_vae_config = 1            
            elif device_mem_capacity >= 8000:
                use_vae_config = 2
            else:          
                use_vae_config = 3
        else:
            use_vae_config = vae_config

        if use_vae_config == 1:
            VAE_tile_size = 0  
        elif use_vae_config == 2:
            VAE_tile_size = 256  
        else: 
            VAE_tile_size = 128  

        return  VAE_tile_size

    def encode(self, videos, tile_size = 256, any_end_frame = False):
        """
        videos: A list of videos each with shape [C, T, H, W].
        """
        set_phase_status("VAE Encoding")
        # tiles with halos (the untiled result) only with the MMGP allocator: Low VRAM (256) always, Default (512) for frames large for the GPU
        # (untiled up to 1280x720 from 12 GB, up to 1920x1088 from 24 GB); PyTorch's allocator: untiled as before
        fast = tile_size > 0 and mmgp_vram_allocator() and _use_fast(torch.empty(0, dtype=self.dtype, device=self.device))
        memory = torch.cuda.get_device_properties(self.device).total_memory if fast else 0
        tiling = [0 if not fast or tile_size > 256 and (memory >= 12e9 and u.shape[-2] * u.shape[-1] <= 1280 * 720 or memory >= 24e9 and u.shape[-2] * u.shape[-1] <= 1920 * 1088) else tile_size for u in videos]
        tiles = 0
        for video, tile in zip(videos, tiling):
            temporal_tiles = 2 + (video.shape[1] - 2) // 4 if any_end_frame else 1 + (video.shape[1] - 1) // 4
            if tile:
                temporal_tiles = _exact_calls(_exact_segments(self.model.encoder, tuple(s // 2 for s in video.shape[-2:]), tile // 2, False), temporal_tiles)
            tiles += temporal_tiles
        with vae_encoding_progress(tiles, self.model.encoder, cleanup=self.model.clear_cache, enabled=any(video.shape[1] > 1 for video in videos)):
            scale = [u.to(device = self.device) for u in self.scale]
            return [(self.model.spatial_tiled_encode(u.to(self.dtype).unsqueeze(0), scale, tile, any_end_frame=any_end_frame) if tile else
                     self.model.encode(u.to(self.dtype).unsqueeze(0), scale, any_end_frame=any_end_frame)).float().squeeze(0) for u, tile in zip(videos, tiling)]


    def decode(self, zs, tile_size = 256, any_end_frame = False):
        scale = [u.to(device = self.device) for u in self.scale]  
        if tile_size > 0 :
            return [ self.model.spatial_tiled_decode(u.to(self.dtype).unsqueeze(0), scale, tile_size, any_end_frame=any_end_frame).clamp_(-1, 1).float().squeeze(0) for u in zs ]
        else:
            return [ self.model.decode(u.to(self.dtype).unsqueeze(0), scale, any_end_frame=any_end_frame).clamp_(-1, 1).float().squeeze(0) for u in zs ]

    def decode_to_cpu_uint8(self, zs, tile_size=256, target_frames=None, target_height=None, target_width=None, any_end_frame=False):
        scale = [u.to(device=self.device) for u in self.scale]
        tile_size = int(tile_size or 0)
        return [
            self.model.decode_to_cpu_uint8(u.detach().to(device="cpu" if tile_size > 0 else u.device, dtype=self.dtype).unsqueeze(0), scale, tile_size, target_frames=target_frames, target_height=target_height, target_width=target_width, any_end_frame=any_end_frame, device=self.device).squeeze(0)
            for u in zs
        ]

