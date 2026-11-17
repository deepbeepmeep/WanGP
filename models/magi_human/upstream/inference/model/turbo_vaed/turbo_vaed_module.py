# Copyright (c) 2026 SandAI. All Rights Reserved.
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

from mmgp import runtime, transfer
from shared.image_vae.output import blend_edge_
from shared.image_vae.tiles import halo_plan
from shared.utils.phase_progress import vae_decoding_progress
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange


__all__ = ["TurboVAED"]

ACT2CLS = {"swish": nn.SiLU, "silu": nn.SiLU, "mish": nn.Mish, "gelu": nn.GELU, "relu": nn.ReLU}


def _identity_compile(fn):
    return fn


def get_activation(act_fn: str) -> nn.Module:
    act_fn = act_fn.lower()
    if act_fn in ACT2CLS:
        return ACT2CLS[act_fn]()
    else:
        raise ValueError(f"activation function {act_fn} not found in ACT2FN mapping {list(ACT2CLS.keys())}")


def unpatchify(x, patch_size):
    """
    Unpatchify operation: convert patched representation back to original spatial resolution.
    Similar to Wan VAE's unpatchify.

    Args:
        x: Input tensor with shape [batch_size, (channels * patch_size * patch_size), frame, height, width]
        patch_size: The patch size used during patchification

    Returns:
        Tensor with shape [batch_size, channels, frame, height * patch_size, width * patch_size]
    """
    if patch_size == 1:
        return x

    if x.dim() != 5:
        raise ValueError(f"Invalid input shape: {x.shape}")

    # x shape: [batch_size, (channels * patch_size * patch_size), frame, height, width]
    batch_size, c_patches, frames, height, width = x.shape
    channels = c_patches // (patch_size * patch_size)

    # Reshape to [b, c, patch_size, patch_size, f, h, w]
    x = x.view(batch_size, channels, patch_size, patch_size, frames, height, width)

    # Rearrange to [b, c, f, h * patch_size, w * patch_size]
    x = x.permute(0, 1, 4, 5, 3, 6, 2).contiguous()
    x = x.view(batch_size, channels, frames, height * patch_size, width * patch_size)

    return x


class RMSNorm(nn.Module):
    r"""
    RMS Norm as introduced in https://huggingface.co/papers/1910.07467 by Zhang et al.

    Args:
        dim (`int`): Number of dimensions to use for `weights`. Only effective when `elementwise_affine` is True.
        eps (`float`): Small value to use when calculating the reciprocal of the square-root.
        elementwise_affine (`bool`, defaults to `True`):
            Boolean flag to denote if affine transformation should be applied.
        bias (`bool`, defaults to False): If also training the `bias` param.
    """

    def __init__(self, dim, eps: float, elementwise_affine: bool = True, bias: bool = False):
        super().__init__()

        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if isinstance(dim, int):
            dim = (dim,)

        self.dim = torch.Size(dim)

        self.weight = None

        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        if self.weight is not None:
            # convert into half-precision if necessary
            if self.weight.dtype in [torch.float16, torch.bfloat16]:
                hidden_states = hidden_states.to(self.weight.dtype)
            hidden_states = hidden_states * self.weight
        else:
            hidden_states = hidden_states.to(input_dtype)

        return hidden_states


class TurboVAEDConv2dSplitUpsampler(nn.Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        stride: Union[int, Tuple[int, int]] = 1,
        upscale_factor: int = 1,
        padding_mode: str = "zeros",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.upscale_factor = upscale_factor

        out_channels = in_channels

        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)

        height_pad = self.kernel_size[0] // 2
        width_pad = self.kernel_size[1] // 2
        padding = (height_pad, width_pad)

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=padding,
            padding_mode=padding_mode,
        )

    @_identity_compile
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv(hidden_states)
        hidden_states = torch.nn.functional.pixel_shuffle(hidden_states, self.stride[0])

        return hidden_states


class TurboVAEDCausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        groups: int = 1,
        padding_mode: str = "zeros",
        is_causal: bool = False,
    ):
        super().__init__()

        assert is_causal == False
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_causal = is_causal
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)

        dilation = dilation if isinstance(dilation, tuple) else (dilation, 1, 1)
        stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        height_pad = self.kernel_size[1] // 2
        width_pad = self.kernel_size[2] // 2
        padding = (0, height_pad, width_pad)

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            padding=padding,
            padding_mode=padding_mode,
        )

    @_identity_compile
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        time_kernel_size = self.kernel_size[0]

        if time_kernel_size > 1:
            pad_left = hidden_states[:, :, :1, :, :].repeat((1, 1, (time_kernel_size - 1) // 2, 1, 1))
            pad_right = hidden_states[:, :, -1:, :, :].repeat((1, 1, (time_kernel_size - 1) // 2, 1, 1))
            hidden_states = torch.cat([pad_left, hidden_states, pad_right], dim=2)

        hidden_states = self.conv(hidden_states)
        return hidden_states


class TurboVAEDCausalDepthwiseSeperableConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        padding_mode: str = "zeros",
        is_causal: bool = True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.is_causal = is_causal
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, stride)
        self.dilation = dilation if isinstance(dilation, tuple) else (dilation, 1, 1)

        # Calculate padding for height and width dimensions
        height_pad = self.kernel_size[1] // 2
        width_pad = self.kernel_size[2] // 2
        self.padding = (0, height_pad, width_pad)

        # Depthwise Convolution
        self.depthwise_conv = nn.Conv3d(
            in_channels,
            in_channels,
            self.kernel_size,
            stride=self.stride,
            dilation=self.dilation,
            groups=in_channels,  # Each input channel is convolved separately
            padding=self.padding,
            padding_mode=padding_mode,
        )

        # Pointwise Convolution
        self.pointwise_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)  # 1x1x1 convolution to mix channels

    @_identity_compile
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        time_kernel_size = self.kernel_size[0]
        if time_kernel_size > 1:
            pad_count = (time_kernel_size - 1) // 2
            pad_left = hidden_states[:, :, :1, :, :].repeat((1, 1, pad_count, 1, 1))
            pad_right = hidden_states[:, :, -1:, :, :].repeat((1, 1, pad_count, 1, 1))
            hidden_states = torch.cat([pad_left, hidden_states, pad_right], dim=2)

        # Apply depthwise convolution
        hidden_states = self.depthwise_conv(hidden_states)
        # Apply pointwise convolution
        hidden_states = self.pointwise_conv(hidden_states)

        return hidden_states


class TurboVAEDResnetBlock3d(nn.Module):
    r"""
    A 3D ResNet block used in the TurboVAED model.

    Args:
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`, *optional*):
            Number of output channels. If None, defaults to `in_channels`.
        dropout (`float`, defaults to `0.0`):
            Dropout rate.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        elementwise_affine (`bool`, defaults to `False`):
            Whether to enable elementwise affinity in the normalization layers.
        non_linearity (`str`, defaults to `"swish"`):
            Activation function to use.
        conv_shortcut (bool, defaults to `False`):
            Whether or not to use a convolution shortcut.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        eps: float = 1e-6,
        elementwise_affine: bool = False,
        non_linearity: str = "swish",
        is_causal: bool = True,
        is_upsampler_modified: bool = False,
        is_dw_conv: bool = False,
        dw_kernel_size: int = 3,
    ) -> None:
        super().__init__()

        out_channels = out_channels or in_channels

        self.nonlinearity = get_activation(non_linearity)

        self.conv_operation = TurboVAEDCausalConv3d if not is_dw_conv else TurboVAEDCausalDepthwiseSeperableConv3d
        self.kernel_size = 3 if not is_dw_conv else dw_kernel_size

        self.is_upsampler_modified = is_upsampler_modified
        self.replace_nonlinearity = get_activation("relu")

        self.norm1 = RMSNorm(in_channels, eps=1e-8, elementwise_affine=elementwise_affine)
        self.conv1 = self.conv_operation(
            in_channels=in_channels, out_channels=out_channels, kernel_size=self.kernel_size, is_causal=is_causal
        )

        self.norm2 = RMSNorm(out_channels, eps=1e-8, elementwise_affine=elementwise_affine)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = self.conv_operation(
            in_channels=out_channels, out_channels=out_channels, kernel_size=self.kernel_size, is_causal=is_causal
        )

        self.norm3 = None
        self.conv_shortcut = None
        if in_channels != out_channels:
            self.norm3 = RMSNorm(in_channels, eps=eps, elementwise_affine=elementwise_affine)
            self.conv_shortcut = self.conv_operation(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, is_causal=is_causal
            )

    @_identity_compile
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden_states = inputs

        hidden_states = self.norm1(hidden_states)

        if self.is_upsampler_modified:
            hidden_states = self.replace_nonlinearity(hidden_states)
        else:
            hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = self.conv2(hidden_states)

        if self.norm3 is not None:
            inputs = self.norm3(inputs)

        if self.conv_shortcut is not None:
            inputs = self.conv_shortcut(inputs)

        hidden_states = hidden_states + inputs
        return hidden_states


class WanUpsample(nn.Upsample):
    r"""
    Perform upsampling while ensuring the output tensor has the same data type as the input.

    Args:
        x (torch.Tensor): Input tensor to be upsampled.

    Returns:
        torch.Tensor: Upsampled tensor with the same data type as the input.
    """

    def forward(self, x):
        return super().forward(x.float()).type_as(x)


class WanResample(nn.Module):
    r"""
    A custom resampling module for 2D and 3D data.

    Args:
        dim (int): The number of input/output channels.
        mode (str): The resampling mode. Must be one of:
            - 'none': No resampling (identity operation).
            - 'upsample2d': 2D upsampling with nearest-exact interpolation and convolution.
            - 'upsample3d': 3D upsampling with nearest-exact interpolation, convolution, and causal 3D convolution.
    """

    def __init__(self, dim: int, mode: str, upsample_out_dim: int = None) -> None:
        super().__init__()
        self.dim = dim
        self.mode = mode

        # default to dim //2
        if upsample_out_dim is None:
            upsample_out_dim = dim // 2

        # layers
        if mode == "upsample2d":
            self.resample = nn.Sequential(
                WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"), nn.Conv2d(dim, upsample_out_dim, 3, padding=1)
            )
        elif mode == "upsample3d":
            self.resample = nn.Sequential(
                WanUpsample(scale_factor=(2.0, 2.0), mode="nearest-exact"), nn.Conv2d(dim, upsample_out_dim, 3, padding=1)
            )
            self.time_conv = TurboVAEDCausalConv3d(dim, dim * 2, (3, 1, 1))
        else:
            self.resample = nn.Identity()

    def forward(self, x, is_first_chunk: bool = True):
        b, c, t, h, w = x.shape
        if self.mode == "upsample3d":
            x = self.time_conv(x)
            x = rearrange(x, 'b (n_split c) t h w -> b c (t n_split) h w', n_split=2)
            assert x.shape == (b, c, t * 2, h, w), "x.shape: {}, expected: {}".format(x.shape, (b, c, t * 2, h, w))
            if is_first_chunk:
                x = x[:, :, 1:]

        x = rearrange(x, "b c t h w -> (b t) c h w")
        x = self.resample(x)
        x = rearrange(x, "(b t) c h w -> b c t h w", b=b)

        return x


class TurboVAEDMidBlock3d(nn.Module):
    r"""
    A middle block used in the TurboVAED model.

    Args:
        in_channels (`int`):
            Number of input channels.
        num_layers (`int`, defaults to `1`):
            Number of resnet layers.
        dropout (`float`, defaults to `0.0`):
            Dropout rate.
        resnet_eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        resnet_act_fn (`str`, defaults to `"swish"`):
            Activation function to use.
        is_causal (`bool`, defaults to `True`):
            Whether this layer behaves causally (future frames depend only on past frames) or not.
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        is_causal: bool = True,
        is_dw_conv: bool = False,
        dw_kernel_size: int = 3,
    ) -> None:
        super().__init__()

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                TurboVAEDResnetBlock3d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    dropout=dropout,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    is_causal=is_causal,
                    is_dw_conv=is_dw_conv,
                    dw_kernel_size=dw_kernel_size,
                )
            )
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    @_identity_compile
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        r"""Forward method of the `LTXMidBlock3D` class."""

        for i, resnet in enumerate(self.resnets):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states)
            else:
                hidden_states = resnet(hidden_states)

        return hidden_states


class TurboVAEDUpBlock3d(nn.Module):
    r"""
    Up block used in the TurboVAED model.

    Args:
        in_channels (`int`):
            Number of input channels.
        out_channels (`int`, *optional*):
            Number of output channels. If None, defaults to `in_channels`.
        num_layers (`int`, defaults to `1`):
            Number of resnet layers.
        dropout (`float`, defaults to `0.0`):
            Dropout rate.
        resnet_eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        resnet_act_fn (`str`, defaults to `"swish"`):
            Activation function to use.
        spatio_temporal_scale (`bool`, defaults to `True`):
            Whether or not to use a downsampling layer. If not used, output dimension would be same as input dimension.
            Whether or not to downsample across temporal dimension.
        is_causal (`bool`, defaults to `True`):
            Whether this layer behaves causally (future frames depend only on past frames) or not.
    """

    _supports_gradient_checkpointing = True

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        spatio_temporal_scale: bool = True,
        is_causal: bool = True,
        is_dw_conv: bool = False,
        dw_kernel_size: int = 3,
        spatio_only: bool = False,
    ):
        super().__init__()

        out_channels = out_channels or in_channels

        self.conv_in = None
        if in_channels != out_channels:
            self.conv_in = TurboVAEDResnetBlock3d(
                in_channels=in_channels,
                out_channels=out_channels,
                dropout=dropout,
                eps=resnet_eps,
                non_linearity=resnet_act_fn,
                is_causal=is_causal,
                is_dw_conv=is_dw_conv,
                dw_kernel_size=dw_kernel_size,
            )

        self.upsamplers = None
        if spatio_temporal_scale:
            self.upsamplers = nn.ModuleList(
                [
                    WanResample(
                        dim=out_channels, mode="upsample2d" if spatio_only else "upsample3d", upsample_out_dim=out_channels
                    )
                ]
            )

        resnets = []
        for _ in range(num_layers):
            resnets.append(
                TurboVAEDResnetBlock3d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    dropout=dropout,
                    eps=resnet_eps,
                    non_linearity=resnet_act_fn,
                    is_causal=is_causal,
                    is_dw_conv=is_dw_conv,
                    dw_kernel_size=dw_kernel_size,
                    is_upsampler_modified=(spatio_temporal_scale),
                )
            )
        self.resnets = nn.ModuleList(resnets)

        self.gradient_checkpointing = False

    @_identity_compile
    def forward(self, hidden_states: torch.Tensor, is_first_chunk: bool) -> torch.Tensor:
        if self.conv_in is not None:
            hidden_states = self.conv_in(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states, is_first_chunk=is_first_chunk)

        for i, resnet in enumerate(self.resnets):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                hidden_states = self._gradient_checkpointing_func(resnet, hidden_states)
            else:
                hidden_states = resnet(hidden_states)

        return hidden_states


class TurboVAEDDecoder3d(nn.Module):
    r"""
    The `TurboVAEDDecoder3d` layer of a variational autoencoder that decodes its latent representation into an output
    sample.

    Args:
        in_channels (`int`, defaults to 128):
            Number of latent channels.
        out_channels (`int`, defaults to 3):
            Number of output channels.
        block_out_channels (`Tuple[int, ...]`, defaults to `(128, 256, 512, 512)`):
            The number of output channels for each block.
        spatio_temporal_scaling (`Tuple[bool, ...], defaults to `(True, True, True, False)`:
            Whether a block should contain spatio-temporal upscaling layers or not.
        layers_per_block (`Tuple[int, ...]`, defaults to `(4, 3, 3, 3, 4)`):
            The number of layers per block.
        patch_size (`int`, defaults to `4`):
            The size of spatial patches.
        patch_size_t (`int`, defaults to `1`):
            The size of temporal patches.
        resnet_norm_eps (`float`, defaults to `1e-6`):
            Epsilon value for ResNet normalization layers.
        is_causal (`bool`, defaults to `False`):
            Whether this layer behaves causally (future frames depend only on past frames) or not.
    """

    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        spatio_temporal_scaling: Tuple[bool, ...] = (True, True, True, False),
        layers_per_block: Tuple[int, ...] = (4, 3, 3, 3, 4),
        patch_size: int = 4,
        patch_size_t: int = 1,
        resnet_norm_eps: float = 1e-6,
        is_causal: bool = False,
        decoder_is_dw_conv: Tuple[bool, ...] = (False, False, False, False, False),
        decoder_dw_kernel_size: int = 3,
        spatio_only: Tuple[bool, ...] = (False, False, False, False),
        upsampling: bool = False,
        use_unpatchify: bool = False,
    ) -> None:
        super().__init__()

        self.patch_size = patch_size
        self.patch_size_t = patch_size_t
        self.out_channels = out_channels

        self.upsampling = upsampling
        self.use_unpatchify = use_unpatchify

        block_out_channels = tuple(reversed(block_out_channels))
        spatio_temporal_scaling = tuple(reversed(spatio_temporal_scaling))
        layers_per_block = tuple(reversed(layers_per_block))
        decoder_is_dw_conv = tuple(reversed(decoder_is_dw_conv))
        spatio_only = tuple(reversed(spatio_only))
        output_channel = block_out_channels[0]

        self.conv_in = TurboVAEDCausalConv3d(
            in_channels=in_channels, out_channels=output_channel, kernel_size=3, stride=1, is_causal=is_causal
        )

        self.mid_block = TurboVAEDMidBlock3d(
            in_channels=output_channel,
            num_layers=layers_per_block[0],
            resnet_eps=resnet_norm_eps,
            is_causal=is_causal,
            is_dw_conv=decoder_is_dw_conv[0],
            dw_kernel_size=decoder_dw_kernel_size,
        )

        # up blocks
        num_block_out_channels = len(block_out_channels)
        self.up_blocks = nn.ModuleList([])
        for i in range(num_block_out_channels):
            input_channel = output_channel
            output_channel = block_out_channels[i]

            up_block = TurboVAEDUpBlock3d(
                in_channels=input_channel,
                out_channels=output_channel,
                num_layers=layers_per_block[i + 1],
                resnet_eps=resnet_norm_eps,
                spatio_temporal_scale=spatio_temporal_scaling[i],
                is_causal=is_causal,
                is_dw_conv=decoder_is_dw_conv[i + 1],
                dw_kernel_size=decoder_dw_kernel_size,
                spatio_only=spatio_only[i],
            )

            self.up_blocks.append(up_block)

        # out
        assert self.patch_size == 2
        if not self.use_unpatchify:
            self.norm_up_1 = RMSNorm(output_channel, eps=1e-8, elementwise_affine=False)
            self.upsampler2d_1 = TurboVAEDConv2dSplitUpsampler(in_channels=output_channel, kernel_size=3, stride=(2, 2))
            output_channel = output_channel // (2 * 2)

        self.conv_act = nn.SiLU()

        # When use_unpatchify=True, conv_out outputs more channels (out_channels * patch_size^2)
        # and unpatchify will recover the spatial resolution
        conv_out_channels = self.out_channels
        if self.use_unpatchify and self.patch_size >= 2:
            conv_out_channels = self.out_channels * self.patch_size * self.patch_size

        self.conv_out = TurboVAEDCausalConv3d(
            in_channels=output_channel, out_channels=conv_out_channels, kernel_size=3, stride=1, is_causal=is_causal
        )

        self.gradient_checkpointing = False

    @_identity_compile
    def forward(self, hidden_states: torch.Tensor, is_first_chunk: bool) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states)

        if torch.is_grad_enabled() and self.gradient_checkpointing:

            def create_custom_forward(module):
                def create_forward(*inputs):
                    return module(*inputs)

                return create_forward

            hidden_states = torch.utils.checkpoint.checkpoint(create_custom_forward(self.mid_block), hidden_states)

            for up_block in self.up_blocks:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(up_block), hidden_states, is_first_chunk
                )
        else:
            hidden_states = self.mid_block(hidden_states)

            for index, up_block in enumerate(self.up_blocks):
                hidden_states = up_block(hidden_states, is_first_chunk=is_first_chunk)

        if not self.use_unpatchify:
            hidden_states = self.norm_up_1(hidden_states)
            hidden_states = self.conv_act(hidden_states)

            hidden_states_array = []
            for t in range(hidden_states.shape[2]):
                h = self.upsampler2d_1(hidden_states[:, :, t, :, :])
                hidden_states_array.append(h)
            hidden_states = torch.stack(hidden_states_array, dim=2)

        # RMSNorm
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + 1e-8)
        hidden_states = hidden_states.to(input_dtype)

        hidden_states = self.conv_act(hidden_states)

        hidden_states = self.conv_out(hidden_states)

        if self.use_unpatchify:
            hidden_states = unpatchify(hidden_states, self.patch_size)

        return hidden_states


class TurboVAED(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,  # useless arg for compatibility, we only use latent channels
        out_channels: int = 3,
        latent_channels: int = 128,
        decoder_block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        decoder_layers_per_block: Tuple[int, ...] = (4, 3, 3, 3, 4),
        decoder_spatio_temporal_scaling: Tuple[bool, ...] = (True, True, True, False),
        patch_size: int = 4,
        patch_size_t: int = 1,
        resnet_norm_eps: float = 1e-6,
        scaling_factor: float = 1.0,
        decoder_causal: bool = False,
        decoder_is_dw_conv: Tuple[bool, ...] = (False, False, False, False, False),
        decoder_dw_kernel_size: int = 3,
        decoder_spatio_only: Tuple[bool, ...] = (False, False, False, False),
        first_chunk_size: int = 3,
        step_size: int = 5,
        spatial_compression_ratio: int = 16,
        temporal_compression_ratio: int = 4,
        use_unpatchify: bool = False,
    ):
        super().__init__()

        self.decoder = TurboVAEDDecoder3d(
            in_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=decoder_block_out_channels,
            spatio_temporal_scaling=decoder_spatio_temporal_scaling,
            layers_per_block=decoder_layers_per_block,
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            resnet_norm_eps=resnet_norm_eps,
            is_causal=decoder_causal,
            decoder_is_dw_conv=decoder_is_dw_conv,
            decoder_dw_kernel_size=decoder_dw_kernel_size,
            spatio_only=decoder_spatio_only,
            use_unpatchify=use_unpatchify,
        )

        self.first_chunk_size = first_chunk_size
        self.step_size = step_size

        self.spatial_compression_ratio = spatial_compression_ratio
        self.temporal_compression_ratio = temporal_compression_ratio

        self.z_dim = latent_channels
        self.mean = torch.tensor(
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
            dtype=torch.float32,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.std = torch.tensor(
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
            dtype=torch.float32,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.scale = [self.mean, 1.0 / self.std]

    def _sliding_window_decode(self, z, output_offload=False, temporal_chunk_size: int = 0):
        """The decoded clip of z: its windows (_decode_windows) concatenated, in RAM with output_offload."""
        return torch.cat([piece.cpu() if output_offload else piece for piece in self._decode_windows(z, temporal_chunk_size)], dim=2)

    def _decode_windows(self, z, temporal_chunk_size: int = 0, decode=None):
        """The decoded clip of z window after window, in order, on z's device: each window is decoded with the latent frame before
        it and the one after it, whose pixel frames are dropped; the frames decoded from the padding of the last window as well.
        decode: the function of (window, is_first_chunk) that decodes a window (the decoder by default)."""
        decode = self.decoder if decode is None else decode
        z_dtype = z.dtype
        scale = [value.to(z.device) for value in self.scale]
        assert isinstance(scale[0], torch.Tensor), "scale[0] must be a tensor"
        z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(1, self.z_dim, 1, 1, 1)
        z = z.to(z_dtype)

        if int(temporal_chunk_size) > 0:
            first_chunk_size = max(1, int(temporal_chunk_size))
            step = first_chunk_size
        else:
            first_chunk_size = self.first_chunk_size
            step = self.step_size

        # Context mapping: 1 latent frame context -> temporal_compression_ratio pixel frames overlap
        num_overlap_pixel_frames = 1 * self.temporal_compression_ratio

        _, _, num_frames, _, _ = z.shape

        # 1. Pad frames to satisfy chunking requirements
        #     The total number of frames must follow the formula:
        #     num_frames = first_chunk_size + n * step_size
        num_padding_frames = 0

        if num_frames < first_chunk_size:
            # if input is shorter than first_chunk_size
            num_padding_frames = first_chunk_size - num_frames
        elif (num_frames - first_chunk_size) % step != 0:
            num_padding_frames = step - (num_frames - first_chunk_size) % step

        if num_padding_frames > 0:
            z = torch.cat([z, z[:, :, -1:].repeat(1, 1, num_padding_frames, 1, 1)], dim=2)
            num_frames = num_frames + num_padding_frames

        # 2. Decode with overlapping windows; 3. the frames of the padding (fewer than a window's) are dropped from the last one
        padded_pixel_frames = num_padding_frames * self.temporal_compression_ratio
        unpadded = lambda out: out[:, :, : out.shape[2] - padded_pixel_frames]

        if num_frames == first_chunk_size:
            # if only one chunk, decode directly
            yield unpadded(decode(z, is_first_chunk=True))
        else:
            # first chunk: attach the right frame
            out = decode(z[:, :, 0 : first_chunk_size + 1, :, :], is_first_chunk=True)
            yield out[:, :, :-num_overlap_pixel_frames]
            del out

            # middle chunk: attach the left and right frames
            # last chunk: attach the left frame
            for i in range(first_chunk_size, num_frames, step):
                is_last_chunk = i + step == num_frames
                left = i - 1
                right = i + step + 1 if not is_last_chunk else i + step

                assert left >= 0 and right <= num_frames, f"left: {left}, right: {right}, num_frames: {num_frames}"

                out_ = decode(z[:, :, left:right, :, :], is_first_chunk=False)

                if is_last_chunk:
                    yield unpadded(out_[:, :, num_overlap_pixel_frames:])
                else:
                    yield out_[:, :, num_overlap_pixel_frames:-num_overlap_pixel_frames]
                del out_

    def _exact_steps(self):
        """The decoder's layers in forward order as (function of (x, is_first_chunk), halo_plan step): every layer is local in space
        (norms per pixel, no attention), so a window decodes exactly by tiles with their halo."""
        decoder = self.decoder
        reach = lambda conv: conv.kernel_size[1] // 2 if len(conv.kernel_size) == 3 else conv.kernel_size[0] // 2
        block = lambda resnet: max(reach(resnet.conv1) + reach(resnet.conv2), 0 if resnet.conv_shortcut is None else reach(resnet.conv_shortcut))
        steps = [(lambda x, first: decoder.conv_in(x), ("same", reach(decoder.conv_in)))]
        steps += [(lambda x, first, resnet=resnet: resnet(x), ("same", block(resnet))) for resnet in decoder.mid_block.resnets]
        for up in decoder.up_blocks:
            if up.conv_in is not None:
                steps.append((lambda x, first, resnet=up.conv_in: resnet(x), ("same", block(up.conv_in))))
            for upsampler in up.upsamplers or ():  # nearest x2 then a 3x3 convolution (its time convolution is 3x1x1)
                steps.append((lambda x, first, upsampler=upsampler: upsampler(x, is_first_chunk=first), ("nearest2",)))
            steps += [(lambda x, first, resnet=resnet: resnet(x), ("same", block(resnet))) for resnet in up.resnets]
        assert decoder.use_unpatchify

        def head(x, first):
            variance = x.to(torch.float32).pow(2).mean(1, keepdim=True)  # (the decoder's own RMS norm and output layers)
            return decoder.conv_out(decoder.conv_act((x * torch.rsqrt(variance + 1e-8)).to(x.dtype)))

        steps.append((head, ("same", reach(decoder.conv_out))))
        steps.append((lambda x, first: unpatchify(x, decoder.patch_size), ("expand", decoder.patch_size)))
        return steps

    def _exact_tiles(self, x, is_first_chunk, tile):
        """decoder(x, is_first_chunk) by tiles of `tile` latent pixels: each tile computed on its region plus the receptive field still
        ahead of every layer (halo_plan), cropped after each one: the untiled result with the memory of a tile."""
        steps = self._exact_steps()
        plan = [step for _, step in steps]
        scale = self.spatial_compression_ratio
        height, width = x.shape[-2:]
        out = None
        for y0 in range(0, height, tile):
            (ya, yb), y_crops = halo_plan(plan, y0 * scale, min(y0 + tile, height) * scale, height)
            for x0 in range(0, width, tile):
                (xa, xb), x_crops = halo_plan(plan, x0 * scale, min(x0 + tile, width) * scale, width)
                h = x[..., ya:yb, xa:xb]
                for (step, _), rows, columns in zip(steps, y_crops, x_crops):
                    h = step(h, is_first_chunk)[..., slice(*rows), slice(*columns)]
                if out is None:
                    out = h.new_empty(*h.shape[:3], height * scale, width * scale)
                out[..., y0 * scale : y0 * scale + h.shape[-2], x0 * scale : x0 * scale + h.shape[-1]] = h
                del h
        return out

    def decode_to_ram(self, z, to_ram, tile_size: int = 0, temporal_chunk_size: int = 0, exact: bool = True):
        """The decoded video in RAM in its final form (to_ram: a function of a decoded piece on the GPU, a dtype conversion or bytes),
        each piece written to its place without waiting for the GPU. exact (with a tile_size): every window decoded by exact tiles
        (_exact_tiles: the untiled result, no seam); else the video decode(z, tile_size, auto_tile=False) gives, tiles blended as
        _spatial_tiled_decode blends them, each window of each tile once blended with the edges kept from the tiles above it and on
        its left. VRAM holds one decoded window (and the edges of a row of blended tiles); RAM holds the video in its final form only
        (decode keeps every decoded tile of the clip in RAM, in 16 bits, to blend them there)."""
        runtime.ensure_loaded(self)  # (an entry point of its own: the hooked forwards are not all called)
        if exact and int(tile_size) > 0:
            frames = 1 + (z.shape[2] - 1) * self.temporal_compression_ratio
            tile = max(1, int(tile_size) // self.spatial_compression_ratio)
            first = int(temporal_chunk_size) if int(temporal_chunk_size) > 0 else self.first_chunk_size
            step = int(temporal_chunk_size) if int(temporal_chunk_size) > 0 else self.step_size
            windows = 1 + max(0, (z.shape[2] - first + step - 1) // step)
            pieces = windows * -(-z.shape[-2] // tile) * -(-z.shape[-1] // tile)
            out, position = None, 0
            with vae_decoding_progress(pieces, self.decoder.conv_out), transfer.RamWriter("decoded video", depth=64, max_bytes=256 << 20) as writer:
                for piece in self._decode_windows(z, temporal_chunk_size, decode=lambda x, is_first_chunk: self._exact_tiles(x, is_first_chunk, tile)):
                    final = to_ram(piece)
                    if out is None:
                        with torch.inference_mode(False):
                            out = torch.empty((*final.shape[:2], frames, *final.shape[3:]), dtype=final.dtype, device="cpu")
                    writer.write(out[:, :, position : position + final.shape[2]], final)
                    position += final.shape[2]
                    del piece, final
            if position != frames:
                raise RuntimeError(f"TurboVAED decoded {position} frames, {frames} expected")
            return out
        frames = 1 + (z.shape[2] - 1) * self.temporal_compression_ratio
        height, width = z.shape[-2] * self.spatial_compression_ratio, z.shape[-1] * self.spatial_compression_ratio
        tiled = int(tile_size) > 0
        latent_tile = max(1, int(tile_size) // 16) if tiled else max(z.shape[-2:])
        stride = max(1, int(latent_tile * 0.75)) if tiled else latent_tile
        extent, limit = (int(tile_size * 0.25), tile_size - int(tile_size * 0.25)) if tiled else (0, max(height, width))
        rows, columns = list(range(0, z.shape[-2], stride)), list(range(0, z.shape[-1], stride))
        first = int(temporal_chunk_size) if int(temporal_chunk_size) > 0 else self.first_chunk_size
        step = int(temporal_chunk_size) if int(temporal_chunk_size) > 0 else self.step_size
        windows = 1 + max(0, (z.shape[2] - first + step - 1) // step)
        out, bottoms, y = None, [None] * len(columns), 0
        with vae_decoding_progress(windows * len(rows) * len(columns), self.decoder), transfer.RamWriter("decoded video", depth=64, max_bytes=256 << 20) as writer:
            for i, r0 in enumerate(rows):
                left, x = None, 0
                for j, c0 in enumerate(columns):
                    bottom = right = None
                    frame = 0
                    for piece in self._decode_windows(z[:, :, :, r0 : r0 + latent_tile, c0 : c0 + latent_tile], temporal_chunk_size):
                        count = piece.shape[2]
                        if i > 0:
                            e = min(bottoms[j].shape[-2], piece.shape[-2], extent)
                            blend_edge_(piece[:, :, :, :e], bottoms[j][:, :, frame : frame + count, bottoms[j].shape[-2] - e :], 3)
                        if j > 0:
                            e = min(left.shape[-1], piece.shape[-1], extent)
                            blend_edge_(piece[..., :e], left[:, :, frame : frame + count, :, left.shape[-1] - e :], 4)
                        if tiled:  # the edges the next tiles blend with (rows below, columns on the right), as blended here
                            if i < len(rows) - 1:
                                if bottom is None:
                                    bottom = piece.new_empty(*piece.shape[:2], frames, min(extent, piece.shape[-2]), piece.shape[-1])
                                bottom[:, :, frame : frame + count] = piece[:, :, :, piece.shape[-2] - bottom.shape[-2] :]
                            if j < len(columns) - 1:
                                if right is None:
                                    right = piece.new_empty(*piece.shape[:2], frames, piece.shape[-2], min(extent, piece.shape[-1]))
                                right[:, :, frame : frame + count] = piece[..., piece.shape[-1] - right.shape[-1] :]
                            piece = piece[:, :, :, :limit, :limit]
                        tile_height, tile_width = piece.shape[-2:]
                        final = to_ram(piece)
                        if out is None:
                            with torch.inference_mode(False):  # (an ordinary tensor: the caller may edit the video in place)
                                out = torch.empty((*final.shape[:2], frames, height, width), dtype=final.dtype, device="cpu")
                        writer.write(out[:, :, frame : frame + count, y : y + tile_height, x : x + tile_width], final)
                        frame += count
                        del piece, final
                    if frame != frames:
                        raise RuntimeError(f"TurboVAED decoded {frame} frames, {frames} expected")
                    left, bottoms[j] = right, bottom
                    x += tile_width
                y += tile_height
        return out

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            alpha = y / blend_extent
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - alpha) + b[:, :, :, y, :] * alpha
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            alpha = x / blend_extent
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - alpha) + b[:, :, :, :, x] * alpha
        return b

    def _spatial_tiled_decode(self, z: torch.Tensor, tile_size: int, output_offload: bool = False, temporal_chunk_size: int = 0):
        tile_sample_min_size = int(tile_size)
        tile_latent_min_size = max(1, tile_sample_min_size // 16)
        tile_overlap_factor = 0.25
        overlap_size = max(1, int(tile_latent_min_size * (1 - tile_overlap_factor)))
        blend_extent = int(tile_sample_min_size * tile_overlap_factor)
        row_limit = tile_sample_min_size - blend_extent

        rows = []
        for i in range(0, z.shape[-2], overlap_size):
            row = []
            for j in range(0, z.shape[-1], overlap_size):
                tile = z[:, :, :, i : i + tile_latent_min_size, j : j + tile_latent_min_size]
                decoded = self._sliding_window_decode(tile, output_offload=output_offload, temporal_chunk_size=temporal_chunk_size)
                row.append(decoded)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))
        return torch.cat(result_rows, dim=-2)

    def _resolve_spatial_tile_size(self, z: torch.Tensor, tile_size: int) -> int:
        tile_size = int(tile_size or 0)
        if tile_size > 0:
            return tile_size
        sample_h = int(z.shape[-2]) * self.spatial_compression_ratio
        sample_w = int(z.shape[-1]) * self.spatial_compression_ratio
        sample_pixels = sample_h * sample_w
        if sample_pixels > 1920 * 1088:
            return 128
        if sample_pixels > 832 * 480:
            return 256
        return 0

    def decode(self, z: torch.Tensor, output_offload: bool = False, tile_size: int = 0, temporal_chunk_size: int = 0, auto_tile: bool = True):
        """auto_tile: without a tile_size, tiles from the frame size (_resolve_spatial_tile_size); False: no tiles then."""
        requested_tile_size = int(tile_size or 0)
        tile_size = self._resolve_spatial_tile_size(z, requested_tile_size) if auto_tile else requested_tile_size
        if z.shape[-2] * z.shape[-1] >= 68 * 120 or tile_size > 0 or int(temporal_chunk_size) > 0:
            spatial_grid = None
            if tile_size > 0:
                tile_latent_min_size = max(1, tile_size // self.spatial_compression_ratio)
                overlap_size = max(1, int(tile_latent_min_size * 0.75))
                rows = max(1, (z.shape[-2] + overlap_size - 1) // overlap_size)
                cols = max(1, (z.shape[-1] + overlap_size - 1) // overlap_size)
                spatial_grid = f"{rows}x{cols}"
            print(
                f"[Magi][TurboVAE] decode latent={tuple(z.shape)} requested_tile_size={requested_tile_size} "
                f"tile_size={tile_size} temporal_chunk_size={int(temporal_chunk_size)} "
                f"output_offload={bool(output_offload)} grid={spatial_grid}",
                flush=True,
            )
        first = int(temporal_chunk_size) if int(temporal_chunk_size) > 0 else self.first_chunk_size
        step = int(temporal_chunk_size) if int(temporal_chunk_size) > 0 else self.step_size
        tiles = 1 + max(0, (z.shape[2] - first + step - 1) // step)
        if tile_size > 0:
            stride = max(1, int(max(1, tile_size // self.spatial_compression_ratio) * 0.75))
            tiles *= len(range(0, z.shape[-2], stride)) * len(range(0, z.shape[-1], stride))
        with vae_decoding_progress(tiles, self.decoder):
            if tile_size > 0:
                return self._spatial_tiled_decode(z, tile_size, output_offload=output_offload, temporal_chunk_size=temporal_chunk_size)
            return self._sliding_window_decode(z, output_offload=output_offload, temporal_chunk_size=temporal_chunk_size)
