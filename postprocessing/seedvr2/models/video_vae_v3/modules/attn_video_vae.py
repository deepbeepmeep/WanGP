# Copyright (c) 2023 HuggingFace Team
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache License, Version 2.0 (the "License")
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 1st June 2025
#
# Original file was released under Apache License, Version 2.0 (the "License"), with the full license text
# available at http://www.apache.org/licenses/LICENSE-2.0.
#
# This modified file is released under the same license.
# Modified for WanGP.


from contextlib import nullcontext
from itertools import product
from typing import Literal, Optional, Tuple, Union
import diffusers
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention, SpatialNorm
from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
from diffusers.models.downsampling import Downsample2D
from diffusers.models.lora import LoRACompatibleConv
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.unets.unet_2d_blocks import DownEncoderBlock2D, UpDecoderBlock2D
from diffusers.models.upsampling import Upsample2D
from diffusers.utils import is_torch_version
from diffusers.utils.accelerate_utils import apply_forward_hook
from einops import rearrange
from mmgp import transfer
from shared.attention import pay_attention
from tqdm import tqdm
from ....common.half_precision_fixes import safe_pad_operation, safe_interpolate_operation

from ....common.logger import get_logger
from .causal_inflation_lib import (
    InflatedCausalConv3d,
    causal_norm_wrapper,
    init_causal_conv3d,
    remove_head,
)
from .global_config import set_norm_limit
from .types import (
    CausalAutoencoderOutput,
    CausalDecoderOutput,
    CausalEncoderOutput,
    MemoryState,
    _inflation_mode_t,
    _memory_device_t,
    _receptive_field_t,
)
from ....optimization.memory_manager import retry_on_oom

logger = get_logger(__name__)  # pylint: disable=invalid-name


def _pay_vae_attention(attn, hidden_states, temb):
    residual = hidden_states
    if attn.spatial_norm is not None:
        hidden_states = attn.spatial_norm(hidden_states, temb)
    batch_size, channels, height, width = hidden_states.shape
    hidden_states = hidden_states.view(batch_size, channels, height * width).transpose(1, 2)
    if attn.group_norm is not None:
        hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)
    query, key, value = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)
    head_dim = key.shape[-1] // attn.heads
    query = query.view(batch_size, -1, attn.heads, head_dim)
    key = key.view(batch_size, -1, attn.heads, head_dim)
    value = value.view(batch_size, -1, attn.heads, head_dim)
    if attn.norm_q is not None:
        query = attn.norm_q(query.transpose(1, 2)).transpose(1, 2)
    if attn.norm_k is not None:
        key = attn.norm_k(key.transpose(1, 2)).transpose(1, 2)
    hidden_states = pay_attention([query, key, value], force_attention="sdpa").reshape(batch_size, -1, attn.heads * head_dim)
    hidden_states = attn.to_out[1](attn.to_out[0](hidden_states))
    hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, channels, height, width)
    if attn.residual_connection:
        hidden_states = hidden_states + residual
    return hidden_states / attn.rescale_output_factor

class Upsample3D(Upsample2D):
    """A 3D upsampling layer with an optional convolution."""

    def __init__(
        self,
        *args,
        inflation_mode: _inflation_mode_t = "tail",
        temporal_up: bool = False,
        spatial_up: bool = True,
        slicing: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        conv = self.conv if self.name == "conv" else self.Conv2d_0

        assert type(conv) is not nn.ConvTranspose2d
        # Note: lora_layer is not passed into constructor in the original implementation.
        # So we make a simplification.
        conv = init_causal_conv3d(
            self.channels,
            self.out_channels,
            3,
            padding=1,
            inflation_mode=inflation_mode,
        )

        self.temporal_up = temporal_up
        self.spatial_up = spatial_up
        self.temporal_ratio = 2 if temporal_up else 1
        self.spatial_ratio = 2 if spatial_up else 1
        self.slicing = slicing

        assert not self.interpolate
        # [Override] MAGViT v2 implementation
        if not self.interpolate:
            upscale_ratio = (self.spatial_ratio**2) * self.temporal_ratio
            self.upscale_conv = nn.Conv3d(
                self.channels, self.channels * upscale_ratio, kernel_size=1, padding=0
            )
            identity = (
                torch.eye(self.channels)
                .repeat(upscale_ratio, 1)
                .reshape_as(self.upscale_conv.weight)
            )
            self.upscale_conv.weight.data.copy_(identity)
            nn.init.zeros_(self.upscale_conv.bias)

        if self.name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(
        self,
        hidden_states_list: list,
        output_size: Optional[int] = None,
        memory_state: MemoryState = MemoryState.DISABLED,
        **kwargs,
    ) -> torch.FloatTensor:
        hidden_states = hidden_states_list.pop()  # the caller holds no reference: the input is released once upscaled
        assert hidden_states.shape[1] == self.channels

        if hasattr(self, "norm") and self.norm is not None:
            # [Overridden] change to causal norm.
            hidden_states = causal_norm_wrapper(self.norm, hidden_states)

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        if self.slicing:
            split_size = hidden_states.size(2) // 2
            hidden_states = list(
                hidden_states.split([split_size, hidden_states.size(2) - split_size], dim=2)
            )
        else:
            hidden_states = [hidden_states]

        for i in range(len(hidden_states)):
            def upscale_and_rearrange():
                temp = self.upscale_conv(hidden_states[i])
                return rearrange(
                    temp,
                    "b (x y z c) f h w -> b c (f z) (h x) (w y)",
                    x=self.spatial_ratio,
                    y=self.spatial_ratio,
                    z=self.temporal_ratio,
                )
            
            hidden_states[i] = retry_on_oom(
                upscale_and_rearrange,
                debug=getattr(self, 'debug', None),
                operation_name="Upsample3D.upscale_conv"
            )

        # [Overridden] For causal temporal conv
        if self.temporal_up and memory_state != MemoryState.ACTIVE:
            hidden_states[0] = remove_head(hidden_states[0])

        if not self.slicing:
            hidden_states = hidden_states[0]

        if self.use_conv:
            def apply_conv():
                if self.name == "conv":
                    return self.conv(hidden_states, memory_state=memory_state)
                else:
                    return self.Conv2d_0(hidden_states, memory_state=memory_state)
            
            hidden_states = retry_on_oom(
                apply_conv,
                debug=getattr(self, 'debug', None),
                operation_name="Upsample3D.conv"
            )

        if not self.slicing:
            return hidden_states
        else:
            return torch.cat(hidden_states, dim=2)


class Downsample3D(Downsample2D):
    """A 3D downsampling layer with an optional convolution."""

    def __init__(
        self,
        *args,
        inflation_mode: _inflation_mode_t = "tail",
        spatial_down: bool = False,
        temporal_down: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        conv = self.conv
        self.temporal_down = temporal_down
        self.spatial_down = spatial_down

        self.temporal_ratio = 2 if temporal_down else 1
        self.spatial_ratio = 2 if spatial_down else 1

        self.temporal_kernel = 3 if temporal_down else 1
        self.spatial_kernel = 3 if spatial_down else 1

        if type(conv) in [nn.Conv2d, LoRACompatibleConv]:
            # Note: lora_layer is not passed into constructor in the original implementation.
            # So we make a simplification.
            conv = init_causal_conv3d(
                self.channels,
                self.out_channels,
                kernel_size=(self.temporal_kernel, self.spatial_kernel, self.spatial_kernel),
                stride=(self.temporal_ratio, self.spatial_ratio, self.spatial_ratio),
                padding=(
                    1 if self.temporal_down else 0,
                    self.padding if self.spatial_down else 0,
                    self.padding if self.spatial_down else 0,
                ),
                inflation_mode=inflation_mode,
            )
        elif type(conv) is nn.AvgPool2d:
            assert self.channels == self.out_channels
            conv = nn.AvgPool3d(
                kernel_size=(self.temporal_ratio, self.spatial_ratio, self.spatial_ratio),
                stride=(self.temporal_ratio, self.spatial_ratio, self.spatial_ratio),
            )
        else:
            raise NotImplementedError

        if self.name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        else:
            self.conv = conv

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        memory_state: MemoryState = MemoryState.DISABLED,
        **kwargs,
    ) -> torch.FloatTensor:

        assert hidden_states.shape[1] == self.channels

        if hasattr(self, "norm") and self.norm is not None:
            # [Overridden] change to causal norm.
            hidden_states = causal_norm_wrapper(self.norm, hidden_states)

        if self.use_conv and self.padding == 0 and self.spatial_down:
            pad = (0, 1, 0, 1)
            hidden_states = safe_pad_operation(hidden_states, pad, mode="constant", value=0)

        assert hidden_states.shape[1] == self.channels

        hidden_states = self.conv(hidden_states, memory_state=memory_state)

        return hidden_states


class ResnetBlock3D(ResnetBlock2D):
    def __init__(
        self,
        *args,
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "half",
        slicing: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.conv1 = init_causal_conv3d(
            self.in_channels,
            self.out_channels,
            kernel_size=(1, 3, 3) if time_receptive_field == "half" else (3, 3, 3),
            stride=1,
            padding=(0, 1, 1) if time_receptive_field == "half" else (1, 1, 1),
            inflation_mode=inflation_mode,
        )

        self.conv2 = init_causal_conv3d(
            self.out_channels,
            self.conv2.out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            inflation_mode=inflation_mode,
        )

        if self.up:
            assert type(self.upsample) is Upsample2D
            self.upsample = Upsample3D(
                self.in_channels,
                use_conv=False,
                inflation_mode=inflation_mode,
                slicing=slicing,
            )
        elif self.down:
            assert type(self.downsample) is Downsample2D
            self.downsample = Downsample3D(
                self.in_channels,
                use_conv=False,
                padding=1,
                name="op",
                inflation_mode=inflation_mode,
            )

        if self.use_in_shortcut:
            self.conv_shortcut = init_causal_conv3d(
                self.in_channels,
                self.conv_shortcut.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=(self.conv_shortcut.bias is not None),
                inflation_mode=inflation_mode,
            )


    def forward(
        self, input_list, temb, memory_state: MemoryState = MemoryState.DISABLED, **kwargs
    ):
        # input_list: [input], emptied so that the input is released once its shortcut projection is taken (before conv1)
        input_tensor = input_list.pop()

        hidden_states = causal_norm_wrapper(self.norm1, input_tensor)
        hidden_states = retry_on_oom(
            self.nonlinearity,
            hidden_states,
            debug=getattr(self, 'debug', None),
            operation_name="ResnetBlock3D.nonlinearity"
        )

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes.
            # see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = self.upsample([input_tensor], memory_state=memory_state)
            hidden_states = self.upsample([hidden_states], memory_state=memory_state)
        elif self.downsample is not None:
            input_tensor = self.downsample(input_tensor, memory_state=memory_state)
            hidden_states = self.downsample(hidden_states, memory_state=memory_state)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor, memory_state=memory_state)

        hidden_states = self.conv1(hidden_states, memory_state=memory_state)

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = self.time_emb_proj(temb)[:, :, None, None]

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        hidden_states = causal_norm_wrapper(self.norm2, hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states, memory_state=memory_state)

        return hidden_states.add_(input_tensor).div_(self.output_scale_factor)  # in place on conv2's output: the same values


class DownEncoderBlock3D(DownEncoderBlock2D):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_downsample: bool = True,
        downsample_padding: int = 1,
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "half",
        temporal_down: bool = True,
        spatial_down: bool = True,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            num_layers=num_layers,
            resnet_eps=resnet_eps,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_pre_norm=resnet_pre_norm,
            output_scale_factor=output_scale_factor,
            add_downsample=add_downsample,
            downsample_padding=downsample_padding,
        )
        resnets = []
        temporal_modules = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                # [Override] Replace module.
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    temb_channels=None,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    inflation_mode=inflation_mode,
                    time_receptive_field=time_receptive_field,
                )
            )
            temporal_modules.append(nn.Identity())

        self.resnets = nn.ModuleList(resnets)
        self.temporal_modules = nn.ModuleList(temporal_modules)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    # [Override] Replace module.
                    Downsample3D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        name="op",
                        temporal_down=temporal_down,
                        spatial_down=spatial_down,
                        inflation_mode=inflation_mode,
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(
        self,
        hidden_states_list: list,
        memory_state: MemoryState = MemoryState.DISABLED,
        **kwargs,
    ) -> torch.FloatTensor:
        hidden_states = hidden_states_list.pop()  # the caller holds no reference: the input is released after the first resnet
        for resnet, temporal in zip(self.resnets, self.temporal_modules):
            resnet_input, hidden_states = [hidden_states], None
            hidden_states = resnet(resnet_input, temb=None, memory_state=memory_state)
            hidden_states = temporal(hidden_states)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, memory_state=memory_state)

        return hidden_states


class UpDecoderBlock3D(UpDecoderBlock2D):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        temb_channels: Optional[int] = None,
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "half",
        temporal_up: bool = True,
        spatial_up: bool = True,
        slicing: bool = False,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            num_layers=num_layers,
            resnet_eps=resnet_eps,
            resnet_time_scale_shift=resnet_time_scale_shift,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_pre_norm=resnet_pre_norm,
            output_scale_factor=output_scale_factor,
            add_upsample=add_upsample,
            temb_channels=temb_channels,
        )
        resnets = []
        temporal_modules = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                # [Override] Replace module.
                ResnetBlock3D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    inflation_mode=inflation_mode,
                    time_receptive_field=time_receptive_field,
                    slicing=slicing,
                )
            )

            temporal_modules.append(nn.Identity())

        self.resnets = nn.ModuleList(resnets)
        self.temporal_modules = nn.ModuleList(temporal_modules)

        if add_upsample:
            # [Override] Replace module & use learnable upsample
            self.upsamplers = nn.ModuleList(
                [
                    Upsample3D(
                        out_channels,
                        use_conv=True,
                        out_channels=out_channels,
                        temporal_up=temporal_up,
                        spatial_up=spatial_up,
                        interpolate=False,
                        inflation_mode=inflation_mode,
                        slicing=slicing,
                    )
                ]
            )
        else:
            self.upsamplers = None

    def forward(
        self,
        hidden_states_list: list,
        temb: Optional[torch.FloatTensor] = None,
        memory_state: MemoryState = MemoryState.DISABLED,
    ) -> torch.FloatTensor:
        hidden_states = hidden_states_list.pop()  # the caller holds no reference: the input is released after the first resnet
        for resnet, temporal in zip(self.resnets, self.temporal_modules):
            resnet_input, hidden_states = [hidden_states], None
            hidden_states = resnet(resnet_input, temb=None, memory_state=memory_state)
            hidden_states = temporal(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                upsampler_input, hidden_states = [hidden_states], None
                hidden_states = upsampler(upsampler_input, memory_state=memory_state)

        return hidden_states


class UNetMidBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        temb_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        add_attention: bool = True,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "half",
    ):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention

        # there is always at least one resnet
        resnets = [
            # [Override] Replace module.
            ResnetBlock3D(
                in_channels=in_channels,
                out_channels=in_channels,
                temb_channels=temb_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                time_embedding_norm=resnet_time_scale_shift,
                non_linearity=resnet_act_fn,
                output_scale_factor=output_scale_factor,
                pre_norm=resnet_pre_norm,
                inflation_mode=inflation_mode,
                time_receptive_field=time_receptive_field,
            )
        ]
        attentions = []

        if attention_head_dim is None:
            logger.warn(
                f"It is not recommend to pass `attention_head_dim=None`. "
                f"Defaulting `attention_head_dim` to `in_channels`: {in_channels}."
            )
            attention_head_dim = in_channels

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=(
                            resnet_groups if resnet_time_scale_shift == "default" else None
                        ),
                        spatial_norm_dim=(
                            temb_channels if resnet_time_scale_shift == "spatial" else None
                        ),
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                ResnetBlock3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    temb_channels=temb_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    time_embedding_norm=resnet_time_scale_shift,
                    non_linearity=resnet_act_fn,
                    output_scale_factor=output_scale_factor,
                    pre_norm=resnet_pre_norm,
                    inflation_mode=inflation_mode,
                    time_receptive_field=time_receptive_field,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states, temb=None, memory_state: MemoryState = MemoryState.DISABLED):
        video_length, frame_height, frame_width = hidden_states.size()[-3:]
        hidden_states = self.resnets[0]([hidden_states], temb, memory_state=memory_state)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
                hidden_states = _pay_vae_attention(attn, hidden_states, temb)
                hidden_states = rearrange(
                    hidden_states, "(b f) c h w -> b c f h w", f=video_length
                )
            hidden_states = resnet([hidden_states], temb, memory_state=memory_state)

        return hidden_states


class Encoder3D(nn.Module):
    r"""
    [Override] override most logics to support extra condition input and causal conv

    The `Encoder` layer of a variational autoencoder that encodes
    its input into a latent representation.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        down_block_types (`Tuple[str, ...]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            The types of down blocks to use.
            See `~diffusers.models.unet_2d_blocks.get_down_block`
            for available options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use.
            See `~diffusers.models.activations.get_activation` for available options.
        double_z (`bool`, *optional*, defaults to `True`):
            Whether to double the number of output channels for the last block.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlock3D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        mid_block_add_attention=True,
        # [Override] add extra_cond_dim, temporal down num
        temporal_down_num: int = 2,
        extra_cond_dim: int = None,
        gradient_checkpoint: bool = False,
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "half",
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.temporal_down_num = temporal_down_num

        self.conv_in = init_causal_conv3d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            stride=1,
            padding=1,
            inflation_mode=inflation_mode,
        )

        self.mid_block = None
        self.down_blocks = nn.ModuleList([])
        self.extra_cond_dim = extra_cond_dim

        self.conv_extra_cond = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            # [Override] to support temporal down block design
            is_temporal_down_block = i >= len(block_out_channels) - self.temporal_down_num - 1
            # Note: take the last ones

            assert down_block_type == "DownEncoderBlock3D"

            down_block = DownEncoderBlock3D(
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=not is_final_block,
                resnet_eps=1e-6,
                downsample_padding=0,
                # Note: Don't know why set it as 0
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                temporal_down=is_temporal_down_block,
                spatial_down=True,
                inflation_mode=inflation_mode,
                time_receptive_field=time_receptive_field,
            )
            self.down_blocks.append(down_block)

            def zero_module(module):
                # Zero out the parameters of a module and return it.
                for p in module.parameters():
                    p.detach().zero_()
                return module

            self.conv_extra_cond.append(
                zero_module(
                    nn.Conv3d(extra_cond_dim, output_channel, kernel_size=1, stride=1, padding=0)
                )
                if self.extra_cond_dim is not None and self.extra_cond_dim > 0
                else None
            )

        # mid
        self.mid_block = UNetMidBlock3D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=mid_block_add_attention,
            inflation_mode=inflation_mode,
            time_receptive_field=time_receptive_field,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6
        )
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = init_causal_conv3d(
            block_out_channels[-1], conv_out_channels, 3, padding=1, inflation_mode=inflation_mode
        )

        self.gradient_checkpointing = gradient_checkpoint

    def forward(
        self,
        sample: torch.FloatTensor,
        extra_cond=None,
        memory_state: MemoryState = MemoryState.DISABLED,
    ) -> torch.FloatTensor:
        r"""The forward method of the `Encoder` class."""
        sample = self.conv_in(sample, memory_state=memory_state)
        if callable(getattr(self, "abort_callback", None)) and self.abort_callback():
            raise InterruptedError("SeedVR2 upscaling aborted")
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # down
            # [Override] add extra block and extra cond
            for down_block, extra_block in zip(self.down_blocks, self.conv_extra_cond):
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(down_block), [sample], memory_state, use_reentrant=False
                )
                if extra_block is not None:
                    sample = sample + safe_interpolate_operation(extra_block(extra_cond), size=sample.shape[2:])

            # middle
            sample = self.mid_block(sample, memory_state=memory_state)

            # sample = torch.utils.checkpoint.checkpoint(
            #     create_custom_forward(self.mid_block), sample, use_reentrant=False
            # )

        else:
            # down
            # [Override] add extra block and extra cond
            for down_block, extra_block in zip(self.down_blocks, self.conv_extra_cond):
                block_input, sample = [sample], None
                sample = down_block(block_input, memory_state=memory_state)
                if extra_block is not None:
                    sample = sample + safe_interpolate_operation(extra_block(extra_cond), size=sample.shape[2:])
                if callable(getattr(self, "abort_callback", None)) and self.abort_callback():
                    raise InterruptedError("SeedVR2 upscaling aborted")

            # middle
            sample = self.mid_block(sample, memory_state=memory_state)
            if callable(getattr(self, "abort_callback", None)) and self.abort_callback():
                raise InterruptedError("SeedVR2 upscaling aborted")

        # post-process
        sample = causal_norm_wrapper(self.conv_norm_out, sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample, memory_state=memory_state)

        return sample


class Decoder3D(nn.Module):
    r"""
    The `Decoder` layer of a variational autoencoder that
    decodes its latent representation into an output sample.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        up_block_types (`Tuple[str, ...]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            The types of up blocks to use.
            See `~diffusers.models.unet_2d_blocks.get_up_block` for available options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use.
            See `~diffusers.models.activations.get_activation` for available options.
        norm_type (`str`, *optional*, defaults to `"group"`):
            The normalization type to use. Can be either `"group"` or `"spatial"`.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = ("UpDecoderBlock3D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        norm_type: str = "group",  # group, spatial
        mid_block_add_attention=True,
        # [Override] add temporal up block
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "half",
        temporal_up_num: int = 2,
        slicing_up_num: int = 0,
        gradient_checkpoint: bool = False,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block
        self.temporal_up_num = temporal_up_num

        self.conv_in = init_causal_conv3d(
            in_channels,
            block_out_channels[-1],
            kernel_size=3,
            stride=1,
            padding=1,
            inflation_mode=inflation_mode,
        )

        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        temb_channels = in_channels if norm_type == "spatial" else None

        # mid
        self.mid_block = UNetMidBlock3D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default" if norm_type == "group" else norm_type,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=temb_channels,
            add_attention=mid_block_add_attention,
            inflation_mode=inflation_mode,
            time_receptive_field=time_receptive_field,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        #print(f"slicing_up_num: {slicing_up_num}")
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]

            is_final_block = i == len(block_out_channels) - 1
            is_temporal_up_block = i < self.temporal_up_num
            is_slicing_up_block = i >= len(block_out_channels) - slicing_up_num
            # Note: Keep symmetric

            assert up_block_type == "UpDecoderBlock3D"
            up_block = UpDecoderBlock3D(
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                add_upsample=not is_final_block,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                resnet_time_scale_shift=norm_type,
                temb_channels=temb_channels,
                temporal_up=is_temporal_up_block,
                slicing=is_slicing_up_block,
                inflation_mode=inflation_mode,
                time_receptive_field=time_receptive_field,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if norm_type == "spatial":
            self.conv_norm_out = SpatialNorm(block_out_channels[0], temb_channels)
        else:
            self.conv_norm_out = nn.GroupNorm(
                num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6
            )
        self.conv_act = nn.SiLU()
        self.conv_out = init_causal_conv3d(
            block_out_channels[0], out_channels, 3, padding=1, inflation_mode=inflation_mode
        )

        self.gradient_checkpointing = gradient_checkpoint

    # Note: Just copy from Decoder.
    def forward(
        self,
        sample: torch.FloatTensor,
        latent_embeds: Optional[torch.FloatTensor] = None,
        memory_state: MemoryState = MemoryState.DISABLED,
    ) -> torch.FloatTensor:
        r"""The forward method of the `Decoder` class."""

        sample = self.conv_in(sample, memory_state=memory_state)
        if callable(getattr(self, "abort_callback", None)) and self.abort_callback():
            raise InterruptedError("SeedVR2 upscaling aborted")

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                sample = self.mid_block(sample, latent_embeds, memory_state=memory_state)

                # up
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block),
                        [sample],
                        latent_embeds,
                        memory_state,
                        use_reentrant=False,
                    )
            else:
                # middle
                sample = self.mid_block(sample, latent_embeds, memory_state=memory_state)

                # up
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block), [sample], latent_embeds, memory_state
                    )
        else:
            # middle
            sample = self.mid_block(sample, latent_embeds, memory_state=memory_state)
            if callable(getattr(self, "abort_callback", None)) and self.abort_callback():
                raise InterruptedError("SeedVR2 upscaling aborted")

            # up
            for up_block in self.up_blocks:
                block_input, sample = [sample], None
                sample = up_block(block_input, latent_embeds, memory_state=memory_state)
                if callable(getattr(self, "abort_callback", None)) and self.abort_callback():
                    raise InterruptedError("SeedVR2 upscaling aborted")

        # post-process
        sample = causal_norm_wrapper(self.conv_norm_out, sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample, memory_state=memory_state)

        return sample


class AutoencoderKL(diffusers.AutoencoderKL):
    """
    We simply inherit the model code from diffusers
    """

    def __init__(self, attention: bool = True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # A hacky way to remove attention.
        if not attention:
            self.encoder.mid_block.attentions = torch.nn.ModuleList([None])
            self.decoder.mid_block.attentions = torch.nn.ModuleList([None])

    def load_state_dict(self, state_dict, strict=True, assign=False):
        # Newer version of diffusers changed the model keys,
        # causing incompatibility with old checkpoints.
        # They provided a method for conversion. We call conversion before loading state_dict.
        convert_deprecated_attention_blocks = getattr(
            self, "_convert_deprecated_attention_blocks", None
        )
        if callable(convert_deprecated_attention_blocks):
            convert_deprecated_attention_blocks(state_dict)
        return super().load_state_dict(state_dict, strict, assign)


class VideoAutoencoderKL(diffusers.AutoencoderKL):
    """
    We simply inherit the model code from diffusers
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str] = ("DownEncoderBlock3D",),
        up_block_types: Tuple[str] = ("UpDecoderBlock3D",),
        block_out_channels: Tuple[int] = (64,),
        layers_per_block: int = 1,
        act_fn: str = "silu",
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        sample_size: int = 32,
        scaling_factor: float = 0.18215,
        force_upcast: float = True,
        attention: bool = True,
        temporal_scale_num: int = 2,
        slicing_up_num: int = 0,
        gradient_checkpoint: bool = False,
        inflation_mode: _inflation_mode_t = "tail",
        time_receptive_field: _receptive_field_t = "full",
        slicing_sample_min_size: int = 32,
        use_quant_conv: bool = True,
        use_post_quant_conv: bool = True,
        *args,
        **kwargs,
    ):
        extra_cond_dim = kwargs.pop("extra_cond_dim") if "extra_cond_dim" in kwargs else None
        self.slicing_sample_min_size = slicing_sample_min_size
        self.slicing_latent_min_size = max(1, slicing_sample_min_size // (2**temporal_scale_num))

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            # [Override] make sure it can be normally initialized
            down_block_types=tuple(
                [down_block_type.replace("3D", "2D") for down_block_type in down_block_types]
            ),
            up_block_types=tuple(
                [up_block_type.replace("3D", "2D") for up_block_type in up_block_types]
            ),
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            latent_channels=latent_channels,
            norm_num_groups=norm_num_groups,
            sample_size=sample_size,
            scaling_factor=scaling_factor,
            force_upcast=force_upcast,
            *args,
            **kwargs,
        )

        # pass init params to Encoder
        self.encoder = Encoder3D(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            act_fn=act_fn,
            norm_num_groups=norm_num_groups,
            double_z=True,
            extra_cond_dim=extra_cond_dim,
            # [Override] add temporal_down_num parameter
            temporal_down_num=temporal_scale_num,
            gradient_checkpoint=gradient_checkpoint,
            inflation_mode=inflation_mode,
            time_receptive_field=time_receptive_field,
        )

        # pass init params to Decoder
        self.decoder = Decoder3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            # [Override] add temporal_up_num parameter
            temporal_up_num=temporal_scale_num,
            slicing_up_num=slicing_up_num,
            gradient_checkpoint=gradient_checkpoint,
            inflation_mode=inflation_mode,
            time_receptive_field=time_receptive_field,
        )

        self.quant_conv = (
            init_causal_conv3d(
                in_channels=2 * latent_channels,
                out_channels=2 * latent_channels,
                kernel_size=1,
                inflation_mode=inflation_mode,
            )
            if use_quant_conv
            else None
        )
        self.post_quant_conv = (
            init_causal_conv3d(
                in_channels=latent_channels,
                out_channels=latent_channels,
                kernel_size=1,
                inflation_mode=inflation_mode,
            )
            if use_post_quant_conv
            else None
        )

        for module in self.modules():  # every activation follows a normalization (a fresh tensor): in place, without a second full-size tensor
            if isinstance(module, nn.SiLU):
                module.inplace = True

        # A hacky way to remove attention.
        if not attention:
            self.encoder.mid_block.attentions = torch.nn.ModuleList([None])
            self.decoder.mid_block.attentions = torch.nn.ModuleList([None])

    @apply_forward_hook
    def encode(self, x: torch.FloatTensor, return_dict: bool = True, 
               tiled: bool = False, tile_size: Tuple[int, int] = (512, 512), 
               tile_overlap: Tuple[int, int] = (64, 64), progress_callback=None) -> AutoencoderKLOutput:
        if tiled:
            h = self.tiled_encode(x, tile_size=tile_size, tile_overlap=tile_overlap, progress_callback=progress_callback)
        else:
            total_steps = self._slicing_steps(x, self.slicing_sample_min_size)
            with tqdm(total=total_steps, desc="SeedVR2 VAE encode", unit="chunk") as progress:
                def report(current, _total):
                    if current == progress.n and current > 0:
                        return
                    progress.update(current - progress.n)
                    if callable(progress_callback):
                        progress_callback(current, total_steps)
                h = self.slicing_encode(x, progress_callback=report)

        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    @apply_forward_hook
    def decode(self, z: torch.Tensor, return_dict: bool = True, 
               tiled: bool = False, tile_size: Tuple[int, int] = (512, 512), 
               tile_overlap: Tuple[int, int] = (64, 64), progress_callback=None) -> Union[DecoderOutput, torch.Tensor]:

        if tiled:
            decoded = self.tiled_decode(z, tile_size=tile_size, tile_overlap=tile_overlap, progress_callback=progress_callback)
        else:
            total_steps = self._slicing_steps(z, self.slicing_latent_min_size)
            with tqdm(total=total_steps, desc="SeedVR2 VAE decode", unit="chunk") as progress:
                def report(current, _total):
                    if current == progress.n and current > 0:
                        return
                    progress.update(current - progress.n)
                    if callable(progress_callback):
                        progress_callback(current, total_steps)
                decoded = self.slicing_decode(z, progress_callback=report)

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def _encode(
        self, x: torch.Tensor, memory_state: MemoryState = MemoryState.DISABLED) -> torch.Tensor:
        h = self.encoder(x, memory_state=memory_state)
        
        if self.quant_conv is not None:
            output = self.quant_conv(h, memory_state=memory_state)
        else:
            output = h
        
        return output

    def _decode(
        self, z: torch.Tensor, memory_state: MemoryState = MemoryState.DISABLED) -> torch.Tensor:
        if self.post_quant_conv is not None:
            z = self.post_quant_conv(z, memory_state=memory_state)
        
        output = self.decoder(z, memory_state=memory_state)
        return output

    def _slicing_steps(self, tensor: torch.Tensor, split_size: int) -> int:
        sliced_frames = tensor.shape[2] - 1
        return (sliced_frames + split_size - 1) // split_size if self.use_slicing and sliced_frames > split_size else 1

    def slicing_encode(self, x: torch.Tensor, progress_callback=None) -> torch.Tensor:
        sp_size = 1
        if self.use_slicing and (x.shape[2] - 1) > self.slicing_sample_min_size * sp_size:
            x_slices = x[:, :, 1:].split(split_size=self.slicing_sample_min_size * sp_size, dim=2)
            if callable(progress_callback):
                progress_callback(0, len(x_slices))
            encoded_slices = [
                self._encode(
                    torch.cat((x[:, :, :1], x_slices[0]), dim=2),
                    memory_state=MemoryState.INITIALIZING,
                )
            ]
            if callable(progress_callback):
                progress_callback(1, len(x_slices))
            for x_idx in range(1, len(x_slices)):
                encoded_slices.append(
                    self._encode(x_slices[x_idx], memory_state=MemoryState.ACTIVE)
                )
                if callable(progress_callback):
                    progress_callback(x_idx + 1, len(x_slices))
            out = torch.cat(encoded_slices, dim=2)
            # Clear memory efficiently
            modules_with_memory = [m for m in self.modules() 
                                if isinstance(m, InflatedCausalConv3d) and m.memory is not None]
            for m in modules_with_memory:
                m.memory = None
            return out
        else:
            if callable(progress_callback):
                progress_callback(0, 1)
            out = self._encode(x)
            if callable(progress_callback):
                progress_callback(1, 1)
            return out

    def slicing_decode(self, z: torch.Tensor, progress_callback=None) -> torch.Tensor:
        sp_size = 1
        if self.use_slicing and (z.shape[2] - 1) > self.slicing_latent_min_size * sp_size:
            z_slices = z[:, :, 1:].split(split_size=self.slicing_latent_min_size * sp_size, dim=2)
            if callable(progress_callback):
                progress_callback(0, len(z_slices))
            decoded_slices = [
                self._decode(
                    torch.cat((z[:, :, :1], z_slices[0]), dim=2),
                    memory_state=MemoryState.INITIALIZING
                )
            ]
            if callable(progress_callback):
                progress_callback(1, len(z_slices))
            for z_idx in range(1, len(z_slices)):
                decoded_slices.append(
                    self._decode(z_slices[z_idx], memory_state=MemoryState.ACTIVE)
                )
                if callable(progress_callback):
                    progress_callback(z_idx + 1, len(z_slices))
            out = torch.cat(decoded_slices, dim=2)
            # Clear memory efficiently
            modules_with_memory = [m for m in self.modules() 
                                if isinstance(m, InflatedCausalConv3d) and m.memory is not None]
            for m in modules_with_memory:
                m.memory = None
            return out
        else:
            if callable(progress_callback):
                progress_callback(0, 1)
            out = self._decode(z)
            if callable(progress_callback):
                progress_callback(1, 1)
            return out

    def tiled_encode(self, x: torch.Tensor, tile_size: Tuple[int, int] = (512, 512), 
                     tile_overlap: Tuple[int, int] = (64, 64), progress_callback=None) -> torch.Tensor:
        r"""
        Encodes an input tensor `x` by splitting it into spatial tiles in latent space. Temporal is handled by `slicing_encode`.
        `tile_size` and `tile_overlap` are interpreted in output-space pixels and converted to latent-space.
        """
        # Ensure 5D [B, C, F, H, W]
        if x.ndim != 5:
            x = x.unsqueeze(2)

        b, c, f, H, W = x.shape
        tile_h, tile_w = tile_size
        
        # Only tile if input resolution requires multiple tiles
        if H <= tile_h and W <= tile_w:
            total_steps = self._slicing_steps(x, self.slicing_sample_min_size)
            with tqdm(total=total_steps, desc="SeedVR2 VAE encode", unit="chunk") as progress:
                def report(current, _total):
                    if current == progress.n and current > 0:
                        return
                    progress.update(current - progress.n)
                    if callable(progress_callback):
                        progress_callback(current, total_steps)
                return self.slicing_encode(x, progress_callback=report)
        else:
            if self.debug:
                self.debug.log(f"Using VAE tiled encoding (Tile: {tile_size}, Overlap: {tile_overlap})", category="vae", force=True, indent_level=1)

        # Spatial scale factor (output/latent)
        scale_factor = self.spatial_downsample_factor

        # Convert output-space tiling params to latent-space
        tile_h, tile_w = tile_size
        overlap_h, overlap_w = tile_overlap
        
        latent_tile_h = max(1, tile_h // scale_factor)
        latent_tile_w = max(1, tile_w // scale_factor)
        latent_overlap_h = max(0, min((overlap_h // scale_factor), latent_tile_h - 1))
        latent_overlap_w = max(0, min((overlap_w // scale_factor), latent_tile_w - 1))

        stride_h = max(1, latent_tile_h - latent_overlap_h)
        stride_w = max(1, latent_tile_w - latent_overlap_w)

        H_lat_total = (H + scale_factor - 1) // scale_factor
        W_lat_total = (W + scale_factor - 1) // scale_factor

        result = None
        count = None

        num_tiles = ((max(H_lat_total - latent_overlap_h, 1) + stride_h - 1) // stride_h) \
                  * ((max(W_lat_total - latent_overlap_w, 1) + stride_w - 1) // stride_w)
        temporal_steps = self._slicing_steps(x, self.slicing_sample_min_size)
        total_steps = num_tiles * temporal_steps

        # Log once at start instead of per-tile
        if self.debug:
            self.debug.log(
                f"Encoding {num_tiles} tiles (Tile: {tile_size}, Overlap: {tile_overlap})",
                category="vae",
            )

        ramp_cache = None

        tile_id = 0
        with tqdm(total=total_steps, desc="SeedVR2 VAE encode", unit="chunk") as progress:
            for y_lat, x_lat in product(range(0, H_lat_total, stride_h), range(0, W_lat_total, stride_w)):
                y_lat_end = min(y_lat + latent_tile_h, H_lat_total)
                x_lat_end = min(x_lat + latent_tile_w, W_lat_total)

                # Skip if fully within overlap of previous tiles
                if (y_lat > 0 and (y_lat_end - y_lat) <= latent_overlap_h) or \
                   (x_lat > 0 and (x_lat_end - x_lat) <= latent_overlap_w):
                    continue

                # Map latent tile to output-space crop
                y_out = y_lat * scale_factor
                x_out = x_lat * scale_factor
                y_out_end = min(y_lat_end * scale_factor, H)
                x_out_end = min(x_lat_end * scale_factor, W)

                tile_id += 1

                # Store tile boundary info for debug visualization
                if self.debug and hasattr(self.debug, 'encode_tile_boundaries'):
                    self.debug.encode_tile_boundaries.append({
                        'id': tile_id,
                        'y': y_out,
                        'x': x_out,
                        'h': y_out_end - y_out,
                        'w': x_out_end - x_out
                    })

                tile_sample = x[:, :, :, y_out:y_out_end, x_out:x_out_end]

                # Log progress periodically instead of every tile (at 1, 6, 11, 16, ...)
                if self.debug and (tile_id % 5 == 1 or tile_id == num_tiles):
                    if tile_id == num_tiles:
                        # Only log final tile if not covered by previous range
                        if (tile_id - 1) % 5 == 0:
                            self.debug.log(f"Encoding tile {tile_id} / {num_tiles}", category="vae", indent_level=1)
                    else:
                        end_tile = min(tile_id + 4, num_tiles)
                        self.debug.log(f"Encoding tiles {tile_id}-{end_tile} / {num_tiles}", category="vae", indent_level=1)

                tile_offset = (tile_id - 1) * temporal_steps
                def report(current, _total, offset=tile_offset):
                    current += offset
                    if current == progress.n and current > 0:
                        return
                    progress.update(current - progress.n)
                    if callable(progress_callback):
                        progress_callback(current, total_steps)
                encoded_tile = self.slicing_encode(tile_sample, progress_callback=report)

                if ramp_cache is None:
                    ramp_cache = {}
                    if latent_overlap_h > 0:
                        t_h = torch.linspace(0, 1, steps=latent_overlap_h, device=encoded_tile.device, dtype=encoded_tile.dtype)
                        ramp_cache['h'] = 0.5 - 0.5 * torch.cos(t_h * torch.pi)
                    if latent_overlap_w > 0:
                        t_w = torch.linspace(0, 1, steps=latent_overlap_w, device=encoded_tile.device, dtype=encoded_tile.dtype)
                        ramp_cache['w'] = 0.5 - 0.5 * torch.cos(t_w * torch.pi)

                # Initialize output size using first encoded tile
                if result is None:
                    b_out, c_out, f_lat, _, _ = encoded_tile.shape
                    
                    result = torch.zeros(
                        (b_out, c_out, f_lat, H_lat_total, W_lat_total),
                        device=encoded_tile.device,
                        dtype=encoded_tile.dtype,
                    )
                    count = torch.zeros((1, 1, 1, H_lat_total, W_lat_total), device=encoded_tile.device, dtype=encoded_tile.dtype)

                eff_h_lat = min(y_lat_end - y_lat, encoded_tile.shape[3], result.shape[3] - y_lat)
                eff_w_lat = min(x_lat_end - x_lat, encoded_tile.shape[4], result.shape[4] - x_lat)

                encoded_tile = encoded_tile[:, :, : result.shape[2], :eff_h_lat, :eff_w_lat]

                # Build faded masks
                ov_h = max(0, min(latent_overlap_h, eff_h_lat - 1))
                ov_w = max(0, min(latent_overlap_w, eff_w_lat - 1))
                
                weight_h = torch.ones((eff_h_lat,), device=encoded_tile.device, dtype=encoded_tile.dtype)
                weight_w = torch.ones((eff_w_lat,), device=encoded_tile.device, dtype=encoded_tile.dtype)

                # Apply fades only on interior edges using cached ramps (avoid fading on outer image borders)
                if ov_h > 0:
                    if y_lat > 0:  # Not top edge
                        weight_h[:ov_h] = ramp_cache['h'][:ov_h]
                    if y_lat_end < H_lat_total:  # Not bottom edge
                        weight_h[-ov_h:] = 1 - ramp_cache['h'][:ov_h]
                if ov_w > 0:
                    if x_lat > 0:  # Not left edge
                        weight_w[:ov_w] = ramp_cache['w'][:ov_w]
                    if x_lat_end < W_lat_total:  # Not right edge
                        weight_w[-ov_w:] = 1 - ramp_cache['w'][:ov_w]

                # Separable application (no 2D mask to save memory)
                weight_h_5d = weight_h.view(1, 1, 1, eff_h_lat, 1)
                weight_w_5d = weight_w.view(1, 1, 1, 1, eff_w_lat)
                encoded_tile.mul_(weight_h_5d).mul_(weight_w_5d)

                result[:, :, : encoded_tile.shape[2], y_lat : y_lat + eff_h_lat, x_lat : x_lat + eff_w_lat] += encoded_tile
                count[:, :, :, y_lat : y_lat + eff_h_lat, x_lat : x_lat + eff_w_lat].addcmul_(weight_h_5d, weight_w_5d)

        result.div_(count.clamp(min=1e-6))

        if x.shape[2] == 1:  # single frame
            result = result.squeeze(2)

        return result

    def tiled_decode(self, z: torch.Tensor, tile_size: Tuple[int, int] = (512, 512), tile_overlap: Tuple[int, int] = (64, 64), progress_callback=None) -> torch.Tensor:
        r"""
        Decodes a latent tensor `z` by splitting it into spatial tiles only. Temporal is handled by `slicing_decode`.
        """
        if z.ndim != 5:
            z = z.unsqueeze(2)

        b, c, f, H, W = z.shape

        # Spatial scale factor (output/latent)
        scale_factor = self.spatial_downsample_factor

        # Convert output-space tiling params to latent-space for spatial tiling
        tile_h, tile_w = tile_size
        overlap_h, overlap_w = tile_overlap
        
        latent_tile_h = max(1, tile_h // scale_factor)
        latent_tile_w = max(1, tile_w // scale_factor)
        
        # Only tile if latent resolution requires multiple tiles
        if H <= latent_tile_h and W <= latent_tile_w:
            total_steps = self._slicing_steps(z, self.slicing_latent_min_size)
            with tqdm(total=total_steps, desc="SeedVR2 VAE decode", unit="chunk") as progress:
                def report(current, _total):
                    if current == progress.n and current > 0:
                        return
                    progress.update(current - progress.n)
                    if callable(progress_callback):
                        progress_callback(current, total_steps)
                return self.slicing_decode(z, progress_callback=report)
        else:
            if self.debug:
                self.debug.log(f"Using VAE tiled decoding (Tile: {tile_size}, Overlap: {tile_overlap})", category="vae", force=True, indent_level=1)
        
        latent_overlap_h = max(0, min((overlap_h // scale_factor), latent_tile_h - 1))
        latent_overlap_w = max(0, min((overlap_w // scale_factor), latent_tile_w - 1))

        stride_h = max(1, latent_tile_h - latent_overlap_h)
        stride_w = max(1, latent_tile_w - latent_overlap_w)

        # Tiles in raster order (rows of tiles), without those fully within the overlap of previous tiles
        tile_rows = [y for y in range(0, H, stride_h) if not (y > 0 and min(y + latent_tile_h, H) - y <= latent_overlap_h)]
        tile_cols = [x for x in range(0, W, stride_w) if not (x > 0 and min(x + latent_tile_w, W) - x <= latent_overlap_w)]
        assert 2 * latent_overlap_h <= latent_tile_h and 2 * latent_overlap_w <= latent_tile_w  # only neighbouring tiles overlap

        # The blended output is written to RAM as soon as no later tile adds to it: only the pending sums stay in VRAM, the overlap below
        # the current row of tiles (carried to the next row) and the overlap right of the current tile (carried to the next tile). Each
        # tile adds the pending sums of its region into itself: the same additions in the same order as with a full-size canvas.
        result = None
        count = None
        pending_rows = None

        num_tiles = ((max(H - latent_overlap_h, 1) + stride_h - 1) // stride_h) \
                  * ((max(W - latent_overlap_w, 1) + stride_w - 1) // stride_w)
        temporal_steps = self._slicing_steps(z, self.slicing_latent_min_size)
        total_steps = num_tiles * temporal_steps

        # Log once at start instead of per-tile
        if self.debug:
            self.debug.log(
                f"Decoding {num_tiles} tiles (Tile: {tile_size}, Overlap: {tile_overlap})",
                category="vae",
            )

        # Pre-compute common ramp values (small memory, big time save)
        ramp_cache = {}
        if overlap_h > 0:
            t_h = torch.linspace(0, 1, steps=overlap_h, device=z.device, dtype=z.dtype)
            ramp_cache['h'] = 0.5 - 0.5 * torch.cos(t_h * torch.pi)
        if overlap_w > 0:
            t_w = torch.linspace(0, 1, steps=overlap_w, device=z.device, dtype=z.dtype)
            ramp_cache['w'] = 0.5 - 0.5 * torch.cos(t_w * torch.pi)

        tile_id = 0
        with tqdm(total=total_steps, desc="SeedVR2 VAE decode", unit="chunk") as progress:
            for (row, y_lat), (col, x_lat) in product(enumerate(tile_rows), enumerate(tile_cols)):
                y_lat_end = min(y_lat + latent_tile_h, H)
                x_lat_end = min(x_lat + latent_tile_w, W)
                if col == 0:
                    above, pending_rows, pending_cols = pending_rows, None, None

                tile_id += 1
                
                # Store tile boundary info for debug visualization
                if self.debug and hasattr(self.debug, 'decode_tile_boundaries'):
                    # Map to output space
                    y_out = y_lat * scale_factor
                    x_out = x_lat * scale_factor
                    y_out_end = y_lat_end * scale_factor
                    x_out_end = x_lat_end * scale_factor
                    self.debug.decode_tile_boundaries.append({
                        'id': tile_id,
                        'y': y_out,
                        'x': x_out,
                        'h': y_out_end - y_out,
                        'w': x_out_end - x_out
                    })
                
                tile_latent = z[:, :, :, y_lat:y_lat_end, x_lat:x_lat_end]

                # Log progress periodically instead of every tile (at 1, 6, 11, 16, ...)
                if self.debug and (tile_id % 5 == 1 or tile_id == num_tiles):
                    if tile_id == num_tiles:
                        # Only log final tile if not covered by previous range
                        if (tile_id - 1) % 5 == 0:
                            self.debug.log(f"Decoding tile {tile_id} / {num_tiles}", category="vae", indent_level=1)
                    else:
                        end_tile = min(tile_id + 4, num_tiles)
                        self.debug.log(f"Decoding tiles {tile_id}-{end_tile} / {num_tiles}", category="vae", indent_level=1)

                tile_offset = (tile_id - 1) * temporal_steps
                def report(current, _total, offset=tile_offset):
                    current += offset
                    if current == progress.n and current > 0:
                        return
                    progress.update(current - progress.n)
                    if callable(progress_callback):
                        progress_callback(current, total_steps)
                decoded_tile = self.slicing_decode(tile_latent, progress_callback=report)

                # Initialize result tensors using actual decoded shapes on first tile
                if result is None:
                    b_out, c_out, out_f_tile, _, _ = decoded_tile.shape
                    output_h = H * scale_factor
                    output_w = W * scale_factor
                    
                    result = torch.empty((b_out, c_out, out_f_tile, output_h, output_w), device="cpu", dtype=decoded_tile.dtype)
                    count = torch.zeros((1, 1, 1, output_h, output_w), device=decoded_tile.device, dtype=decoded_tile.dtype)

                # Corresponding output-space placement
                y_out, y_out_end = y_lat * scale_factor, y_lat_end * scale_factor
                x_out, x_out_end = x_lat * scale_factor, x_lat_end * scale_factor

                h_out = y_out_end - y_out
                w_out = x_out_end - x_out

                # Build faded masks
                ov_h_out = max(0, min(overlap_h, h_out - 1))
                ov_w_out = max(0, min(overlap_w, w_out - 1))
                
                weight_h = torch.ones((h_out,), device=decoded_tile.device, dtype=decoded_tile.dtype)
                weight_w = torch.ones((w_out,), device=decoded_tile.device, dtype=decoded_tile.dtype)

                # Apply fades only on interior edges using cached ramps (avoid fading on outer image borders)
                if ov_h_out > 0:
                    if y_lat > 0:  # Not top edge
                        weight_h[:ov_h_out] = ramp_cache['h'][:ov_h_out]
                    if y_lat_end < H:  # Not bottom edge
                        weight_h[-ov_h_out:] = 1 - ramp_cache['h'][:ov_h_out]
                if ov_w_out > 0:
                    if x_lat > 0:  # Not left edge
                        weight_w[:ov_w_out] = ramp_cache['w'][:ov_w_out]
                    if x_lat_end < W:  # Not right edge
                        weight_w[-ov_w_out:] = 1 - ramp_cache['w'][:ov_w_out]

                # Separable application (no 2D mask to save memory)
                weight_h_5d = weight_h.view(1, 1, 1, h_out, 1)
                weight_w_5d = weight_w.view(1, 1, 1, 1, w_out)
                decoded_tile.mul_(weight_h_5d).mul_(weight_w_5d)

                # Sums of earlier tiles over this tile: the overlap with the previous tile (which already includes the row above), then the
                # rest of the overlap with the row above
                left = 0 if pending_cols is None else pending_cols.shape[-1]
                if left:
                    decoded_tile[..., :left].add_(pending_cols)
                if above is not None:
                    decoded_tile[:, :, :, :above.shape[3], left:].add_(above[..., x_out + left:x_out_end])
                count[:, :, :, y_out:y_out_end, x_out:x_out_end].addcmul_(weight_h_5d, weight_w_5d)

                # Rows from next_y and columns from next_x get contributions from later tiles; the rest is final
                next_y = tile_rows[row + 1] * scale_factor - y_out if row + 1 < len(tile_rows) else h_out
                next_x = tile_cols[col + 1] * scale_factor - x_out if col + 1 < len(tile_cols) else w_out
                pending_cols = decoded_tile[..., next_x:].clone() if next_x < w_out else None
                if next_y < h_out:
                    if pending_rows is None:
                        pending_rows = decoded_tile.new_empty((*decoded_tile.shape[:3], h_out - next_y, output_w))
                    pending_rows[..., x_out:x_out + next_x].copy_(decoded_tile[:, :, :, next_y:, :next_x])
                final = decoded_tile[:, :, :, :next_y, :next_x].div_(count[:, :, :, y_out:y_out + next_y, x_out:x_out + next_x].clamp(min=1e-6))
                transfer.copy_(result[:, :, : decoded_tile.shape[2], y_out:y_out + next_y, x_out:x_out + next_x], final)  # (to RAM, without a GPU copy of the tile)
                final = decoded_tile = None

        if z.shape[2] == 1:  # single frame
            result = result.squeeze(2)

        return result

    def forward(
        self, x: torch.FloatTensor, mode: Literal["encode", "decode", "all"] = "all", **kwargs
    ):
        # x: [b c t h w]
        if mode == "encode":
            h = self.encode(x)
            return h.latent_dist
        elif mode == "decode":
            h = self.decode(x)
            return h.sample
        else:
            h = self.encode(x)
            h = self.decode(h.latent_dist.mode())
            return h.sample

    def load_state_dict(self, state_dict, strict=False, assign=False):
        # Newer version of diffusers changed the model keys,
        # causing incompatibility with old checkpoints.
        # They provided a method for conversion.
        # We call conversion before loading state_dict.
        convert_deprecated_attention_blocks = getattr(
            self, "_convert_deprecated_attention_blocks", None
        )
        if callable(convert_deprecated_attention_blocks):
            convert_deprecated_attention_blocks(state_dict)
        return super().load_state_dict(state_dict, strict, assign)


class VideoAutoencoderKLWrapper(VideoAutoencoderKL):
    def __init__(
        self,
        *args,
        spatial_downsample_factor: int,
        temporal_downsample_factor: int,
        freeze_encoder: bool,
        **kwargs,
    ):
        self.spatial_downsample_factor = spatial_downsample_factor
        self.temporal_downsample_factor = temporal_downsample_factor
        self.freeze_encoder = freeze_encoder
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.FloatTensor) -> CausalAutoencoderOutput:
        with torch.no_grad() if self.freeze_encoder else nullcontext():
            z, p = self.encode(x)
        x = self.decode(z).sample
        return CausalAutoencoderOutput(x, z, p)

    def encode(self, x: torch.FloatTensor, return_dict: bool = True, 
               tiled: bool = False, tile_size: Tuple[int, int] = (512, 512), 
               tile_overlap: Tuple[int, int] = (64, 64), progress_callback=None) -> CausalEncoderOutput:
        if x.ndim == 4:
            x = x.unsqueeze(2)
        p = super().encode(x, return_dict=return_dict, tiled=tiled, tile_size=tile_size,
                          tile_overlap=tile_overlap, progress_callback=progress_callback).latent_dist
        # Use deterministic mode for tiled encoding to avoid artifacts
        z = p.mode().squeeze(2)
        return CausalEncoderOutput(z, p)

    def decode(self, z: torch.Tensor, return_dict: bool = True, 
               tiled: bool = False, tile_size: Tuple[int, int] = (512, 512), 
               tile_overlap: Tuple[int, int] = (64, 64), progress_callback=None) -> CausalDecoderOutput:
        if z.ndim == 4:
            z = z.unsqueeze(2)
        x = super().decode(z, return_dict=return_dict, tiled=tiled, tile_size=tile_size,
                          tile_overlap=tile_overlap, progress_callback=progress_callback).sample.squeeze(2)
        return CausalDecoderOutput(x)

    def preprocess(self, x: torch.Tensor):
        # x should in [B, C, T, H, W], [B, C, H, W]
        assert x.ndim == 4 or x.size(2) % 4 == 1
        return x

    def postprocess(self, x: torch.Tensor):
        # x should in [B, C, T, H, W], [B, C, H, W]
        return x

    def set_causal_slicing(
        self,
        *,
        split_size: Optional[int],
        memory_device: _memory_device_t,
    ):
        assert (
            split_size is None or memory_device is not None
        ), "if split_size is set, memory_device must not be None."
        if split_size is not None:
            self.enable_slicing()
            self.slicing_sample_min_size = split_size
            self.slicing_latent_min_size = max(1, split_size // self.temporal_downsample_factor)
        else:
            self.disable_slicing()
        for module in self.modules():
            if isinstance(module, InflatedCausalConv3d):
                module.set_memory_device(memory_device)

    def set_memory_limit(self, conv_max_mem: Optional[float], norm_max_mem: Optional[float]):
        set_norm_limit(norm_max_mem)
        for m in self.modules():
            if isinstance(m, InflatedCausalConv3d):
                m.set_memory_limit(conv_max_mem if conv_max_mem is not None else float("inf"))
                
