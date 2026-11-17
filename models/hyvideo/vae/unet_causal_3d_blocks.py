# Copyright 2023 The HuggingFace Team. All rights reserved.
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
# Modifications Copyright (c) 2025-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange

from diffusers.utils import is_torch_version, logging
from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import SpatialNorm
from diffusers.models.attention_processor import Attention
from diffusers.models.normalization import AdaGroupNorm
from diffusers.models.normalization import RMSNorm

from shared.kernels import kernel_policy, vae_ops

try:
    from shared.kernels import causal_group_norm_triton as _fused
except (ImportError, OSError, RuntimeError):
    _fused = None


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

# Sub-pixel form of a nearest x2 upsample followed by a 3-tap replicate-padded convolution: output parity p reads 2 low-resolution taps
# (m - 1, m) for p = 0 and (m, m + 1) for p = 1, which combine the 3 high-resolution taps as below (also in time: (k - 1, k) for both parities).
_SUBPIXEL_TAPS = ((1.0, 0.0, 0.0), (0.0, 1.0, 1.0)), ((1.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def fused_norm_silu(x):
    """Triton GroupNorm + SiLU + padding kernels for the convolution inputs ("fast" kernel precision, CUDA GPUs with Triton)."""
    return vae_ops.causal_norm_kernels(x)


class CausalConv3d(nn.Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        pad_mode = 'replicate',
        disable_causal=False,
        **kwargs
    ):
        super().__init__()

        self.pad_mode = pad_mode
        if disable_causal:
            padding = (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2)
        else:
            padding = (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size - 1, 0) # W, H, T
        self.time_causal_padding = padding

        self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride = stride, dilation = dilation, **kwargs)

    def forward(self, x):
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)

    def norm_silu_forward(self, x, norm):
        """self(silu(norm(x))) with the norm, the SiLU and the replicate padding in one pass (when fused_norm_silu(x))."""
        return self.conv(_fused.group_norm_silu_pad(x, norm, self.time_causal_padding))

    def stream_norm_silu(self, x, norm, state):
        """norm_silu_forward of one chunk of frames of a streamed decode: the norm uses the running statistics of its input over the chunks
        so far (or, for the norms in state["frozen"], the statistics given in state), and the previous chunk's last input frames replace the
        replicate padding of the first frame."""
        left, right, top, bottom, front, _ = self.time_causal_padding
        shared = state.get("shared", False)
        running = state.get(norm)
        if running is None:
            running = state[norm] = x.new_zeros((1 if shared else x.shape[0]) * norm.num_groups, 3, dtype=torch.float64)
        frozen = norm in state["frozen"]
        cache = state.get(self)
        if cache is None:
            padded = vae_ops.causal_group_norm_silu_pad(x, norm, self.time_causal_padding, running=running, frozen=frozen, shared=shared)
            state[self] = padded[:, :, -front:].clone()
        else:
            batch, channels, frames, height, width = x.shape
            padded = x.new_empty(batch, channels, front + frames, top + height + bottom, left + width + right)
            padded[:, :, :front] = cache
            vae_ops.causal_group_norm_silu_pad(x, norm, (left, right, top, bottom, 0, 0), running=running, frozen=frozen, shared=shared, out=padded,
                                               out_frame=front)
            cache.copy_(padded[:, :, -front:])
        return self.conv(padded)

class CausalAvgPool3d(nn.Module):
    def __init__(
        self,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]],
        pad_mode = 'replicate',
        disable_causal=False,
        **kwargs
    ):
        super().__init__()

        self.pad_mode = pad_mode
        if disable_causal:
            padding = (0, 0, 0, 0, 0, 0)
        else:
            padding = (0, 0, 0, 0, stride - 1, 0) # W, H, T
        self.time_causal_padding = padding

        self.conv = nn.AvgPool3d(kernel_size, stride=stride, ceil_mode=True, **kwargs)
        self.pad_mode = pad_mode

    def forward(self, x):
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)

class UpsampleCausal3D(nn.Module):
    """A 3D upsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        use_conv_transpose (`bool`, default `False`):
            option to use a convolution transpose.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        name (`str`, default `conv`):
            name of the upsampling 3D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        use_conv_transpose: bool = False,
        out_channels: Optional[int] = None,
        name: str = "conv",
        kernel_size: Optional[int] = None,
        padding=1,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
        interpolate=True,
        upsample_factor=(2, 2, 2),
        disable_causal=False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.interpolate = interpolate
        self.upsample_factor = upsample_factor
        self.disable_causal = disable_causal

        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(channels, eps, elementwise_affine)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        conv = None
        if use_conv_transpose:
            assert False, "Not Implement yet"
            if kernel_size is None:
                kernel_size = 4
            conv = nn.ConvTranspose2d(
                channels, self.out_channels, kernel_size=kernel_size, stride=2, padding=padding, bias=bias
            )
        elif use_conv:
            if kernel_size is None:
                kernel_size = 3
            conv = CausalConv3d(self.channels, self.out_channels, kernel_size=kernel_size, bias=bias, disable_causal=disable_causal)

        if name == "conv":
            self.conv = conv
        else:
            self.Conv2d_0 = conv

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        output_size: Optional[int] = None,
        scale: float = 1.0,
        state=None,
    ) -> torch.FloatTensor:
        if state is not None:  # one chunk of a streamed decode, through forward so that the offload hooks load the weights first
            return self.stream(hidden_states, state)
        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            assert False, "Not Implement yet"
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.use_conv_transpose:
            return self.conv(hidden_states)

        if self.use_conv and not self.disable_causal and output_size is None and kernel_policy.allow_approximate():
            return self._subpixel_forward(hidden_states)

        # Cast to float32 to as 'upsample_nearest2d_out_frame' op does not support bfloat16
        # https://github.com/pytorch/pytorch/issues/86679
        dtype = hidden_states.dtype
        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(torch.float32)

        # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
        if hidden_states.shape[0] >= 64:
            hidden_states = hidden_states.contiguous()

        # if `output_size` is passed we force the interpolation output
        # size and do not make use of `scale_factor=2`
        if self.interpolate:
            B, C, T, H, W = hidden_states.shape
            if not self.disable_causal:
                first_h, other_h = hidden_states.split((1, T-1), dim=2)
                if output_size is None:
                    if T > 1:
                        other_h = F.interpolate(other_h, scale_factor=self.upsample_factor, mode="nearest")
    
                    first_h = first_h.squeeze(2)
                    first_h = F.interpolate(first_h, scale_factor=self.upsample_factor[1:], mode="nearest")
                    first_h = first_h.unsqueeze(2)
                else:
                    assert False, "Not Implement yet"
                    other_h = F.interpolate(other_h, size=output_size, mode="nearest")
            
                if T > 1:
                    hidden_states = torch.cat((first_h, other_h), dim=2)
                else:
                    hidden_states = first_h
            else:
                hidden_states = F.interpolate(hidden_states, scale_factor=self.upsample_factor, mode="nearest")                

        if dtype == torch.bfloat16:
            hidden_states = hidden_states.to(dtype)

        if self.use_conv:
            if self.name == "conv":
                hidden_states = self.conv(hidden_states)
            else:
                hidden_states = self.Conv2d_0(hidden_states)

        return hidden_states

    def _subpixel_weights(self):
        """Kernels of the 3x3x3 convolution of the nearest-upsampled tensor rewritten as one convolution of the replicate-padded
        low-resolution input, whose 2x2(x2) kernels hold every output parity: exact in real arithmetic (the summed kernel taps are rounded to
        the weight dtype once), up to 3.4x fewer operations."""
        conv = self.conv.conv
        taps = torch.tensor(_SUBPIXEL_TAPS, dtype=torch.float32, device=conv.weight.device)
        if self.upsample_factor[0] == 2:
            weight = torch.einsum("oityx,pat,qby,rcx->pqroiabc", conv.weight.float(), taps, taps, taps).flatten(0, 3)
        else:
            weight = torch.einsum("oityx,qby,rcx->qroitbc", conv.weight.float(), taps, taps).flatten(0, 2)
        return weight.to(conv.weight.dtype), conv.bias.repeat(weight.shape[0] // conv.out_channels)

    def _subpixel_conv(self, padded, weights):
        """(batch, time parity, row parity, column parity, channels, windows, height + 1, width + 1) from _subpixel_weights."""
        return F.conv3d(padded, *weights).unflatten(1, (-1, 2, 2, self.conv.conv.out_channels))

    @staticmethod
    def _interleave(y, height, width, first):
        """Output frames of the parity windows: time x2 gives even frame 2k from window k and odd frame 2k - 1 from window k >= 1 (a later
        chunk of a stream starts with its first window's odd frame); without time upsampling, frame k from window k."""
        batch, time_parities, _, _, channels, windows = y.shape[:6]
        frames = windows if time_parities == 1 else 2 * windows - first
        out = y.new_empty(batch, channels, frames, 2 * height, 2 * width)
        for pt in range(time_parities):
            frames_out = slice(None) if time_parities == 1 else slice(pt, None, 2) if first else slice(1 - pt, None, 2)
            windows_in = slice(pt, None) if first else slice(None)
            for py in range(2):
                for px in range(2):
                    out[:, :, frames_out, py::2, px::2] = y[:, pt, py, px, :, windows_in, py:py + height, px:px + width]
        return out

    def _subpixel_forward(self, x):
        """Same upsampling as forward, through _subpixel_conv (no upsampled copy). Replicate padding of the low-resolution input equals
        replicate padding of the upsampled tensor; causal frames: windows (k - 1, k), or 3 taps without time upsampling."""
        padded = F.pad(x, (1, 1, 1, 1, 1 if self.upsample_factor[0] == 2 else 2, 0), mode="replicate")
        return self._interleave(self._subpixel_conv(padded, self._subpixel_weights()), x.shape[3], x.shape[4], True)

    def _stream_exact(self, x, state):
        """forward of one chunk of frames of a streamed decode with the layer's own arithmetic (the nearest upsampling, then the causal
        convolution): the upsampled frames are written straight into the padded input of the convolution, whose first two frames are
        the previous chunk's last ones (the replicated first frame for the first chunk). The first frame of the clip is not doubled in
        time."""
        cache = state.get(self)
        batch, channels, frames, height, width = x.shape
        doubled = self.upsample_factor[0] == 2
        first = doubled and cache is None
        padded = x.new_empty(batch, channels, 2 + frames * (2 if doubled else 1) - first, 2 * height + 2, 2 * width + 2)
        body = padded[:, :, 2:]
        for py in range(2):
            for px in range(2):
                inner = body[:, :, :, 1 + py:1 + 2 * height:2, 1 + px:1 + 2 * width:2]
                if not doubled:
                    inner.copy_(x)
                elif first:
                    inner[:, :, :1] = x[:, :, :1]
                    inner[:, :, 1::2] = x[:, :, 1:]
                    inner[:, :, 2::2] = x[:, :, 1:]
                else:
                    inner[:, :, 0::2] = x
                    inner[:, :, 1::2] = x
        body[:, :, :, :1, 1:-1] = body[:, :, :, 1:2, 1:-1]
        body[:, :, :, -1:, 1:-1] = body[:, :, :, -2:-1, 1:-1]
        body[..., :1] = body[..., 1:2]
        body[..., -1:] = body[..., -2:-1]
        if cache is None:
            padded[:, :, :2] = body[:, :, :1]
            state[self] = padded[:, :, -2:].clone()
        else:
            padded[:, :, :2] = cache
            cache.copy_(padded[:, :, -2:])
        return self.conv.conv(padded)

    def stream(self, x, state):
        """One chunk of frames of a streamed decode. "fast" kernel precision: _subpixel_forward of the chunk, the previous chunk's last
        frames replacing the replicate padding; state["weights"] keeps the kernels for the whole decode. Else _stream_exact (the summed
        kernel taps of the sub-pixel form are rounded once more than the layer's own)."""
        if not kernel_policy.allow_approximate():
            return self._stream_exact(x, state)
        weights = state["weights"].get(self)
        if weights is None:
            weights = state["weights"][self] = self._subpixel_weights()
        context = 1 if self.upsample_factor[0] == 2 else 2
        cache = state.get(self)
        padded = F.pad(x, (1, 1, 1, 1, context if cache is None else 0, 0), mode="replicate")
        if cache is None:
            state[self] = padded[:, :, -context:].clone()
        else:
            padded = torch.cat([cache, padded], 2)
            cache.copy_(padded[:, :, -context:])
        return self._interleave(self._subpixel_conv(padded, weights), x.shape[3], x.shape[4], cache is None)

class DownsampleCausal3D(nn.Module):
    """A 3D downsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        padding (`int`, default `1`):
            padding for the convolution.
        name (`str`, default `conv`):
            name of the downsampling 3D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        name: str = "conv",
        kernel_size=3,
        norm_type=None,
        eps=None,
        elementwise_affine=None,
        bias=True,
        stride=2,
        disable_causal=False,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = stride
        self.name = name

        if norm_type == "ln_norm":
            self.norm = nn.LayerNorm(channels, eps, elementwise_affine)
        elif norm_type == "rms_norm":
            self.norm = RMSNorm(channels, eps, elementwise_affine)
        elif norm_type is None:
            self.norm = None
        else:
            raise ValueError(f"unknown norm_type: {norm_type}")

        if use_conv:
            conv = CausalConv3d(
                self.channels, self.out_channels, kernel_size=kernel_size, stride=stride, disable_causal=disable_causal, bias=bias
            )
        else:
            raise NotImplementedError
        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv

    def forward(self, hidden_states: torch.FloatTensor, scale: float = 1.0) -> torch.FloatTensor:
        assert hidden_states.shape[1] == self.channels

        if self.norm is not None:
            hidden_states = self.norm(hidden_states.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        assert hidden_states.shape[1] == self.channels

        hidden_states = self.conv(hidden_states)

        return hidden_states

class ResnetBlockCausal3D(nn.Module):
    r"""
    A Resnet block.

    Parameters:
        in_channels (`int`): The number of channels in the input.
        out_channels (`int`, *optional*, default to be `None`):
            The number of output channels for the first conv2d layer. If None, same as `in_channels`.
        dropout (`float`, *optional*, defaults to `0.0`): The dropout probability to use.
        temb_channels (`int`, *optional*, default to `512`): the number of channels in timestep embedding.
        groups (`int`, *optional*, default to `32`): The number of groups to use for the first normalization layer.
        groups_out (`int`, *optional*, default to None):
            The number of groups to use for the second normalization layer. if set to None, same as `groups`.
        eps (`float`, *optional*, defaults to `1e-6`): The epsilon to use for the normalization.
        non_linearity (`str`, *optional*, default to `"swish"`): the activation function to use.
        time_embedding_norm (`str`, *optional*, default to `"default"` ): Time scale shift config.
            By default, apply timestep embedding conditioning with a simple shift mechanism. Choose "scale_shift" or
            "ada_group" for a stronger conditioning with scale and shift.
        kernel (`torch.FloatTensor`, optional, default to None): FIR filter, see
            [`~models.resnet.FirUpsample2D`] and [`~models.resnet.FirDownsample2D`].
        output_scale_factor (`float`, *optional*, default to be `1.0`): the scale factor to use for the output.
        use_in_shortcut (`bool`, *optional*, default to `True`):
            If `True`, add a 1x1 nn.conv2d layer for skip-connection.
        up (`bool`, *optional*, default to `False`): If `True`, add an upsample layer.
        down (`bool`, *optional*, default to `False`): If `True`, add a downsample layer.
        conv_shortcut_bias (`bool`, *optional*, default to `True`):  If `True`, adds a learnable bias to the
            `conv_shortcut` output.
        conv_3d_out_channels (`int`, *optional*, default to `None`): the number of channels in the output.
            If None, same as `out_channels`.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: Optional[int] = None,
        conv_shortcut: bool = False,
        dropout: float = 0.0,
        temb_channels: int = 512,
        groups: int = 32,
        groups_out: Optional[int] = None,
        pre_norm: bool = True,
        eps: float = 1e-6,
        non_linearity: str = "swish",
        skip_time_act: bool = False,
        time_embedding_norm: str = "default",  # default, scale_shift, ada_group, spatial
        kernel: Optional[torch.FloatTensor] = None,
        output_scale_factor: float = 1.0,
        use_in_shortcut: Optional[bool] = None,
        up: bool = False,
        down: bool = False,
        conv_shortcut_bias: bool = True,
        conv_3d_out_channels: Optional[int] = None,
        disable_causal: bool = False,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_norm = True
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.up = up
        self.down = down
        self.output_scale_factor = output_scale_factor
        self.time_embedding_norm = time_embedding_norm
        self.skip_time_act = skip_time_act

        linear_cls = nn.Linear

        if groups_out is None:
            groups_out = groups

        if self.time_embedding_norm == "ada_group":
            self.norm1 = AdaGroupNorm(temb_channels, in_channels, groups, eps=eps)
        elif self.time_embedding_norm == "spatial":
            self.norm1 = SpatialNorm(in_channels, temb_channels)
        else:
            self.norm1 = torch.nn.GroupNorm(num_groups=groups, num_channels=in_channels, eps=eps, affine=True)

        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3, stride=1, disable_causal=disable_causal)

        if temb_channels is not None:
            if self.time_embedding_norm == "default":
                self.time_emb_proj = linear_cls(temb_channels, out_channels)
            elif self.time_embedding_norm == "scale_shift":
                self.time_emb_proj = linear_cls(temb_channels, 2 * out_channels)
            elif self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
                self.time_emb_proj = None
            else:
                raise ValueError(f"unknown time_embedding_norm : {self.time_embedding_norm} ")
        else:
            self.time_emb_proj = None

        if self.time_embedding_norm == "ada_group":
            self.norm2 = AdaGroupNorm(temb_channels, out_channels, groups_out, eps=eps)
        elif self.time_embedding_norm == "spatial":
            self.norm2 = SpatialNorm(out_channels, temb_channels)
        else:
            self.norm2 = torch.nn.GroupNorm(num_groups=groups_out, num_channels=out_channels, eps=eps, affine=True)

        self.dropout = torch.nn.Dropout(dropout)
        conv_3d_out_channels = conv_3d_out_channels or out_channels
        self.conv2 = CausalConv3d(out_channels, conv_3d_out_channels, kernel_size=3, stride=1, disable_causal=disable_causal)

        self.nonlinearity = get_activation(non_linearity)

        self.upsample = self.downsample = None
        if self.up:
            self.upsample = UpsampleCausal3D(in_channels, use_conv=False, disable_causal=disable_causal)
        elif self.down:
            self.downsample = DownsampleCausal3D(in_channels, use_conv=False, disable_causal=disable_causal, name="op")

        self.use_in_shortcut = self.in_channels != conv_3d_out_channels if use_in_shortcut is None else use_in_shortcut

        self.conv_shortcut = None
        if self.use_in_shortcut:
            self.conv_shortcut = CausalConv3d(
                in_channels,
                conv_3d_out_channels,
                kernel_size=1,
                stride=1,
                disable_causal=disable_causal,
                bias=conv_shortcut_bias,
            )

    def _residual(self, hidden_states, input_tensor):
        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut.conv(input_tensor)  # 1x1x1: no padding
        hidden_states.add_(input_tensor)
        return hidden_states if self.output_scale_factor == 1 else hidden_states.div_(self.output_scale_factor)

    def stream(self, x, state):
        """forward of one chunk of frames of a streamed decode (see CausalConv3d.stream_norm_silu)."""
        return self._residual(self.conv2.stream_norm_silu(self.conv1.stream_norm_silu(x, self.norm1, state), self.norm2, state), x)

    def _whole_clip(self, x, chunk_frames):
        """forward of the whole clip by chunks of frames: each GroupNorm's statistics over its whole input first, then norm + SiLU +
        convolution chunk by chunk with causal caches (stream_norm_silu with frozen statistics), the residual written in place. The module's
        values without its whole padded copies, both convolution outputs alive at once and the workspace of whole-clip convolutions."""
        def frozen(norm, tensor):
            stats = tensor.new_zeros(tensor.shape[0] * norm.num_groups, 3, dtype=torch.float64)
            vae_ops.causal_group_norm_silu_pad(tensor, norm, running=stats, apply=False)
            return {norm: stats, "frozen": {norm}}

        frames = x.shape[2]
        state = frozen(self.norm1, x)
        hidden = x.new_empty(x.shape[0], self.conv1.conv.out_channels, *x.shape[2:])
        for start in range(0, frames, chunk_frames):
            hidden[:, :, start:start + chunk_frames] = self.conv1.stream_norm_silu(x[:, :, start:start + chunk_frames], self.norm1, state)
        state = frozen(self.norm2, hidden)
        out = x if self.conv_shortcut is None else x.new_empty(x.shape[0], self.conv2.conv.out_channels, *x.shape[2:])
        for start in range(0, frames, chunk_frames):
            chunk = slice(start, start + chunk_frames)
            out[:, :, chunk] = self._residual(self.conv2.stream_norm_silu(hidden[:, :, chunk], self.norm2, state), x[:, :, chunk])
        return out

    def forward(
        self,
        input_tensor: torch.FloatTensor,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        state=None,
        chunk_frames=None,
    ) -> torch.FloatTensor:
        # parts of a streamed decode, through forward so that the offload hooks load the weights first
        if state is not None:
            return self.stream(input_tensor, state)
        if chunk_frames is not None:
            return self._whole_clip(input_tensor, chunk_frames)
        if temb is None and self.upsample is None and self.downsample is None and fused_norm_silu(input_tensor):
            hidden_states = self.conv2.norm_silu_forward(self.conv1.norm_silu_forward(input_tensor, self.norm1), self.norm2)
            return self._residual(hidden_states, input_tensor)

        hidden_states = input_tensor

        if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
            hidden_states = self.norm1(hidden_states, temb)
        else:
            hidden_states = self.norm1(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)

        if self.upsample is not None:
            # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
            if hidden_states.shape[0] >= 64:
                input_tensor = input_tensor.contiguous()
                hidden_states = hidden_states.contiguous()
            input_tensor = (
                self.upsample(input_tensor, scale=scale)
            )
            hidden_states = (
                self.upsample(hidden_states, scale=scale)
            )
        elif self.downsample is not None:
            input_tensor = (
                self.downsample(input_tensor, scale=scale)
            )
            hidden_states = (
                self.downsample(hidden_states, scale=scale)
            )

        hidden_states = self.conv1(hidden_states)

        if self.time_emb_proj is not None:
            if not self.skip_time_act:
                temb = self.nonlinearity(temb)
            temb = (
                self.time_emb_proj(temb, scale)[:, :, None, None]
            )

        if temb is not None and self.time_embedding_norm == "default":
            hidden_states = hidden_states + temb

        if self.time_embedding_norm == "ada_group" or self.time_embedding_norm == "spatial":
            hidden_states = self.norm2(hidden_states, temb)
        else:
            hidden_states = self.norm2(hidden_states)

        if temb is not None and self.time_embedding_norm == "scale_shift":
            scale, shift = torch.chunk(temb, 2, dim=1)
            hidden_states = hidden_states * (1 + scale) + shift

        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = (
                self.conv_shortcut(input_tensor)
            )

        output_tensor = (input_tensor + hidden_states) / self.output_scale_factor

        return output_tensor

def get_down_block3d(
    down_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    temb_channels: int,
    add_downsample: bool,
    downsample_stride: int,
    resnet_eps: float,
    resnet_act_fn: str,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = None,
    resnet_groups: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    downsample_padding: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = False,
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    attention_type: str = "default",
    resnet_skip_time_act: bool = False,
    resnet_out_scale_factor: float = 1.0,
    cross_attention_norm: Optional[str] = None,
    attention_head_dim: Optional[int] = None,
    downsample_type: Optional[str] = None,
    dropout: float = 0.0,
    disable_causal: bool = False,
):
    # If attn head dim is not defined, we default it to the number of heads
    if attention_head_dim is None:
        logger.warn(
            f"It is recommended to provide `attention_head_dim` when calling `get_down_block`. Defaulting `attention_head_dim` to {num_attention_heads}."
        )
        attention_head_dim = num_attention_heads

    down_block_type = down_block_type[7:] if down_block_type.startswith("UNetRes") else down_block_type
    if down_block_type == "DownEncoderBlockCausal3D":
        return DownEncoderBlockCausal3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            add_downsample=add_downsample,
            downsample_stride=downsample_stride,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            downsample_padding=downsample_padding,
            resnet_time_scale_shift=resnet_time_scale_shift,
            disable_causal=disable_causal,
        )
    raise ValueError(f"{down_block_type} does not exist.")

def get_up_block3d(
    up_block_type: str,
    num_layers: int,
    in_channels: int,
    out_channels: int,
    prev_output_channel: int,
    temb_channels: int,
    add_upsample: bool,
    upsample_scale_factor: Tuple,
    resnet_eps: float,
    resnet_act_fn: str,
    resolution_idx: Optional[int] = None,
    transformer_layers_per_block: int = 1,
    num_attention_heads: Optional[int] = None,
    resnet_groups: Optional[int] = None,
    cross_attention_dim: Optional[int] = None,
    dual_cross_attention: bool = False,
    use_linear_projection: bool = False,
    only_cross_attention: bool = False,
    upcast_attention: bool = False,
    resnet_time_scale_shift: str = "default",
    attention_type: str = "default",
    resnet_skip_time_act: bool = False,
    resnet_out_scale_factor: float = 1.0,
    cross_attention_norm: Optional[str] = None,
    attention_head_dim: Optional[int] = None,
    upsample_type: Optional[str] = None,
    dropout: float = 0.0,
    disable_causal: bool = False,
) -> nn.Module:
    # If attn head dim is not defined, we default it to the number of heads
    if attention_head_dim is None:
        logger.warn(
            f"It is recommended to provide `attention_head_dim` when calling `get_up_block`. Defaulting `attention_head_dim` to {num_attention_heads}."
        )
        attention_head_dim = num_attention_heads

    up_block_type = up_block_type[7:] if up_block_type.startswith("UNetRes") else up_block_type
    if up_block_type == "UpDecoderBlockCausal3D":
        return UpDecoderBlockCausal3D(
            num_layers=num_layers,
            in_channels=in_channels,
            out_channels=out_channels,
            resolution_idx=resolution_idx,
            dropout=dropout,
            add_upsample=add_upsample,
            upsample_scale_factor=upsample_scale_factor,
            resnet_eps=resnet_eps,
            resnet_act_fn=resnet_act_fn,
            resnet_groups=resnet_groups,
            resnet_time_scale_shift=resnet_time_scale_shift,
            temb_channels=temb_channels,
            disable_causal=disable_causal,
        )
    raise ValueError(f"{up_block_type} does not exist.")


class UNetMidBlockCausal3D(nn.Module):
    """
    A 3D UNet mid-block [`UNetMidBlockCausal3D`] with multiple residual blocks and optional attention blocks.

    Args:
        in_channels (`int`): The number of input channels.
        temb_channels (`int`): The number of temporal embedding channels.
        dropout (`float`, *optional*, defaults to 0.0): The dropout rate.
        num_layers (`int`, *optional*, defaults to 1): The number of residual blocks.
        resnet_eps (`float`, *optional*, 1e-6 ): The epsilon value for the resnet blocks.
        resnet_time_scale_shift (`str`, *optional*, defaults to `default`):
            The type of normalization to apply to the time embeddings. This can help to improve the performance of the
            model on tasks with long-range temporal dependencies.
        resnet_act_fn (`str`, *optional*, defaults to `swish`): The activation function for the resnet blocks.
        resnet_groups (`int`, *optional*, defaults to 32):
            The number of groups to use in the group normalization layers of the resnet blocks.
        attn_groups (`Optional[int]`, *optional*, defaults to None): The number of groups for the attention blocks.
        resnet_pre_norm (`bool`, *optional*, defaults to `True`):
            Whether to use pre-normalization for the resnet blocks.
        add_attention (`bool`, *optional*, defaults to `True`): Whether to add attention blocks.
        attention_head_dim (`int`, *optional*, defaults to 1):
            Dimension of a single attention head. The number of attention heads is determined based on this value and
            the number of input channels.
        output_scale_factor (`float`, *optional*, defaults to 1.0): The output scale factor.

    Returns:
        `torch.FloatTensor`: The output of the last residual block, which is a tensor of shape `(batch_size,
        in_channels, height, width)`.

    """

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
        attn_groups: Optional[int] = None,
        resnet_pre_norm: bool = True,
        add_attention: bool = True,
        attention_head_dim: int = 1,
        output_scale_factor: float = 1.0,
        disable_causal: bool = False,
        causal_attention: bool = False,
    ):
        super().__init__()
        resnet_groups = resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        self.add_attention = add_attention
        self.causal_attention = causal_attention

        if attn_groups is None:
            attn_groups = resnet_groups if resnet_time_scale_shift == "default" else None

        # there is always at least one resnet
        resnets = [
            ResnetBlockCausal3D(
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
                disable_causal=disable_causal,
            )
        ]
        attentions = []

        if attention_head_dim is None:
            logger.warn(
                f"It is not recommend to pass `attention_head_dim=None`. Defaulting `attention_head_dim` to `in_channels`: {in_channels}."
            )
            attention_head_dim = in_channels

        for _ in range(num_layers):
            if self.add_attention:
                #assert False, "Not implemented yet"
                attentions.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        rescale_output_factor=output_scale_factor,
                        eps=resnet_eps,
                        norm_num_groups=attn_groups,
                        spatial_norm_dim=temb_channels if resnet_time_scale_shift == "spatial" else None,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                ResnetBlockCausal3D(
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
                    disable_causal=disable_causal,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    @staticmethod
    def _frame_causal_attention(attn, hidden_states):
        """attn (group norm, projections, SDPA, output projection, residual) with upstream's frame-causal mask, without building the
        tokens x tokens mask: the queries of each frame attend to the keys of that frame and the earlier ones."""
        frames, height, width = hidden_states.shape[2:]
        tokens = height * width
        residual = hidden_states.flatten(2).transpose(1, 2)
        normed = attn.group_norm(residual.transpose(1, 2)).transpose(1, 2)
        query, key, value = (projection(normed).unflatten(2, (attn.heads, -1)).transpose(1, 2) for projection in (attn.to_q, attn.to_k, attn.to_v))
        del normed
        out = torch.empty_like(query)
        for stop in range(tokens, frames * tokens + 1, tokens):
            out[:, :, stop - tokens:stop] = F.scaled_dot_product_attention(query[:, :, stop - tokens:stop], key[:, :, :stop], value[:, :, :stop])
        del query, key, value
        out = attn.to_out[0](out.transpose(1, 2).flatten(2))
        return ((out + residual) / attn.rescale_output_factor).transpose(1, 2).unflatten(2, (frames, height, width))

    def forward(self, hidden_states: torch.FloatTensor, temb: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
        hidden_states = self.resnets[0](hidden_states, temb)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None and self.causal_attention:
                hidden_states = self._frame_causal_attention(attn, hidden_states)
            elif attn is not None:
                B, C, T, H, W = hidden_states.shape
                hidden_states = rearrange(hidden_states, "b c f h w -> b (f h w) c")
                hidden_states = attn(hidden_states, temb=temb)
                hidden_states = rearrange(hidden_states, "b (f h w) c -> b c f h w", f=T, h=H, w=W)
            hidden_states = resnet(hidden_states, temb)

        return hidden_states


class DownEncoderBlockCausal3D(nn.Module):
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
        downsample_stride: int = 2,
        downsample_padding: int = 1,
        disable_causal: bool = False,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlockCausal3D(
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
                    disable_causal=disable_causal,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    DownsampleCausal3D(
                        out_channels, 
                        use_conv=True, 
                        out_channels=out_channels, 
                        padding=downsample_padding, 
                        name="op", 
                        stride=downsample_stride,
                        disable_causal=disable_causal,
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(self, hidden_states: torch.FloatTensor, scale: float = 1.0) -> torch.FloatTensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=None, scale=scale)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states, scale)

        return hidden_states


class UpDecoderBlockCausal3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        resolution_idx: Optional[int] = None,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_time_scale_shift: str = "default",  # default, spatial
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        resnet_pre_norm: bool = True,
        output_scale_factor: float = 1.0,
        add_upsample: bool = True,
        upsample_scale_factor = (2, 2, 2),
        temb_channels: Optional[int] = None,
        disable_causal: bool = False,
    ):
        super().__init__()
        resnets = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                ResnetBlockCausal3D(
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
                    disable_causal=disable_causal,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [
                    UpsampleCausal3D(
                        out_channels, 
                        use_conv=True, 
                        out_channels=out_channels, 
                        upsample_factor=upsample_scale_factor, 
                        disable_causal=disable_causal
                    )
                ]
            )
        else:
            self.upsamplers = None

        self.resolution_idx = resolution_idx

    def forward(
        self, hidden_states: torch.FloatTensor, temb: Optional[torch.FloatTensor] = None, scale: float = 1.0
    ) -> torch.FloatTensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states, temb=temb, scale=scale)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states

