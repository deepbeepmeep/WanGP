# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanVideo-1.5/blob/main/LICENSE
#
# Unless and only to the extent required by applicable law, the Tencent Hunyuan works and any
# output and results therefrom are provided "AS IS" without any express or implied warranties of
# any kind including any warranties of title, merchantability, noninfringement, course of dealing,
# usage of trade, or fitness for a particular purpose. You are solely responsible for determining the
# appropriateness of using, reproducing, modifying, performing, displaying or distributing any of
# the Tencent Hunyuan works or outputs and assume any and all risks associated with your or a
# third party's use or distribution of any of the Tencent Hunyuan works or outputs and your exercise
# of rights and permissions under this agreement.
# See the License for the specific language governing permissions and limitations under the License.
# Modified for WanGP.

from shared.utils.phase_progress import vae_decoding_progress, set_phase_status
from mmgp import transfer
from shared.cuda_memory import mmgp_vram_allocator
from shared.image_vae import tiling
from shared.image_vae.output import blend_edge_
from shared.image_vae.tiles import halo_plan
from shared.kernels import vae_ops
from shared.utils.phase_progress import check_abort
import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import loguru
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch.nn import Conv3d
from torch import distributed as dist

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.autoencoders.vae import BaseOutput, DiagonalGaussianDistribution
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin



@dataclass
class DecoderOutput(BaseOutput):
    sample: torch.FloatTensor
    posterior: Optional[DiagonalGaussianDistribution] = None


def swish(x: Tensor, inplace=False) -> Tensor:
    """Applies the swish activation function (SiLU) with optional inplace support."""
    return F.silu(x, inplace=inplace)


def forward_with_checkpointing(module, *inputs, use_checkpointing=False):
    """Forward with optional gradient checkpointing."""
    def create_custom_forward(module):
        def custom_forward(*inputs):
            return module(*inputs)
        return custom_forward

    if use_checkpointing:
        return torch.utils.checkpoint.checkpoint(create_custom_forward(module), *inputs, use_reentrant=False)
    else:
        return module(*inputs)


# Optimized implementation of CogVideoXSafeConv3d
# https://github.com/huggingface/diffusers/blob/c9ff360966327ace3faad3807dc871a4e5447501/src/diffusers/models/autoencoders/autoencoder_kl_cogvideox.py#L38
class PatchCausalConv3d(nn.Conv3d):
    r"""Causal Conv3d with efficient patch processing for large tensors."""
    def find_split_indices(self, seq_len, part_num):
        ideal_interval = seq_len / part_num
        possible_indices = list(range(0, seq_len, self.stride[0]))
        selected_indices = []

        for i in range(1, part_num):
            closest = min(possible_indices, key=lambda x: abs(x - round(i * ideal_interval)))
            if closest not in selected_indices:
                selected_indices.append(closest)

        merged_indices = []
        prev_idx = 0
        for idx in selected_indices:
            if idx - prev_idx >= self.kernel_size[0]:
                merged_indices.append(idx)
                prev_idx = idx

        return merged_indices

    def forward(self, input):
        T = input.shape[2]  # input: NCTHW
        memory_count = torch.prod(torch.tensor(input.shape)).item() * 2 / 1024**3
        if T > self.kernel_size[0] and memory_count > 0.6:
            kernel_size = self.kernel_size[0]
            part_num = int(memory_count / 2) + 1
            split_indices = self.find_split_indices(T, part_num)
            input_chunks = torch.tensor_split(input, split_indices, dim=2) if len(split_indices) > 0 else [input]
            if kernel_size > 1:
                input_chunks = [input_chunks[0]] + [
                    torch.cat((input_chunks[i - 1][:, :, -kernel_size + 1:], input_chunks[i]), dim=2)
                    for i in range(1, len(input_chunks))
                ]
            output_chunks = []
            for input_chunk in input_chunks:
                output_chunks.append(super().forward(input_chunk))
            output = torch.cat(output_chunks, dim=2)
            return output
        else:
            return super().forward(input)




class RMS_norm(nn.Module):
    """Root Mean Square Layer Normalization for Channel-First or Last"""

    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x):
        return F.normalize(
            x, dim=(1 if self.channel_first else -1)
        ) * self.scale * self.gamma + self.bias


class CausalConv3d(nn.Module):
    """Causal Conv3d with configurable padding for temporal axis."""

    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        pad_mode='replicate',
        disable_causal=False,
        enable_patch_conv=False,
        **kwargs
    ):
        super().__init__()

        self.pad_mode = pad_mode
        if disable_causal:
            padding = (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2)
        else:
            padding = (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size - 1, 0)  # W, H, T
        self.time_causal_padding = padding

        if enable_patch_conv:
            self.conv = PatchCausalConv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)
        else:
            self.conv = nn.Conv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, x):
        x = F.pad(x, self.time_causal_padding, mode=self.pad_mode)
        return self.conv(x)

    def stream(self, x, state, residual=None, norm=None):
        """forward of one chunk of frames of a streamed clip, of swish(norm(x)) with norm, plus residual: the last input frames of the
        previous chunk (kept in state) replace the replicate padding of the first frame."""
        out, state[self] = vae_ops.replicate_padded_conv3d(x, self.conv, self.time_causal_padding[:5], state.get(self), residual, norm)
        return out


def prepare_causal_attention_mask(n_frame: int, n_hw: int, dtype, device, batch_size: int = None):
    """Prepare a causal attention mask for 3D videos.

    Args:
        n_frame (int): Number of frames (temporal length).
        n_hw (int): Product of height and width.
        dtype: Desired mask dtype.
        device: Device for the mask.
        batch_size (int, optional): If set, expands for batch.

    Returns:
        torch.Tensor: Causal attention mask.
    """
    seq_len = n_frame * n_hw
    mask = torch.full((seq_len, seq_len), float("-inf"), dtype=dtype, device=device)
    for i in range(seq_len):
        i_frame = i // n_hw
        mask[i, : (i_frame + 1) * n_hw] = 0
    if batch_size is not None:
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return mask


class AttnBlock(nn.Module):
    """Self-attention block for 3D video tensors."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels

        self.norm = RMS_norm(in_channels, images=False)

        self.q = Conv3d(in_channels, in_channels, kernel_size=1)
        self.k = Conv3d(in_channels, in_channels, kernel_size=1)
        self.v = Conv3d(in_channels, in_channels, kernel_size=1)
        self.proj_out = Conv3d(in_channels, in_channels, kernel_size=1)

    def attention(self, h_: Tensor) -> Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, f, h, w = q.shape
        q = rearrange(q, "b c f h w -> b 1 (f h w) c").contiguous()
        k = rearrange(k, "b c f h w -> b 1 (f h w) c").contiguous()
        v = rearrange(v, "b c f h w -> b 1 (f h w) c").contiguous()
        attention_mask = prepare_causal_attention_mask(f, h * w, h_.dtype, h_.device, batch_size=b)
        h_ = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask.unsqueeze(1))

        return rearrange(h_, "b 1 (f h w) c -> b c f h w", f=f, h=h, w=w, c=c, b=b)

    def forward(self, x: Tensor, by_frames: bool = False) -> Tensor:
        if by_frames:
            return self.causal_frames_(x)
        return x + self.proj_out(self.attention(x))

    def causal_frames_(self, x, chunk_bytes=32 << 20):
        """forward on a whole clip, in place: the keys and values of all the frames, then the queries of one frame at a time attending
        to that frame and the earlier ones (the mask of attention(), without its tokens x tokens tensor nor its loop over the tokens),
        each frame's result added to x. The 1x1x1 projections are matrix products on the tokens: cuDNN's 3D convolution reserves a
        workspace of 0.9 GiB for them on a 1280x720 x 65 clip, 7 times the clip at this resolution."""
        batch, channels, frames, height, width = x.shape
        tokens = height * width
        step = max(1, chunk_bytes // (batch * channels * tokens * x.element_size()))
        project = lambda conv, rows: F.linear(rows, conv.weight.flatten(1), conv.bias)
        normed = lambda first, last: self.norm(x[:, :, first:last]).flatten(2).transpose(1, 2)  # (batch, frames * tokens, channels)
        key, value = x.new_empty(batch, 1, frames * tokens, channels), x.new_empty(batch, 1, frames * tokens, channels)
        for start in range(0, frames, step):
            rows = normed(start, start + step)
            span = slice(start * tokens, start * tokens + rows.shape[1])
            key[:, 0, span], value[:, 0, span] = project(self.k, rows), project(self.v, rows)
            del rows
        for frame in range(frames):
            visible = (frame + 1) * tokens
            out = vae_ops.attention(project(self.q, normed(frame, frame + 1)).unsqueeze(1), key[:, :, :visible], value[:, :, :visible])[:, 0]
            x[:, :, frame:frame + 1] += project(self.proj_out, out).transpose(1, 2).unflatten(2, (1, height, width))
            del out
            check_abort()
        return x


class ResnetBlock(nn.Module):
    """ResNet-style block for 3D video tensors."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels

        self.norm1 = RMS_norm(in_channels, images=False)
        self.conv1 = CausalConv3d(in_channels, out_channels, kernel_size=3)

        self.norm2 = RMS_norm(out_channels, images=False)
        self.conv2 = CausalConv3d(out_channels, out_channels, kernel_size=3)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_list, state=None):
        """state: the causal caches of a streamed clip, of which x is one chunk of frames (see CausalConv3d.stream)."""
        x = x_list[0]
        x_list.clear()
        if state is not None:
            h = self.conv1.stream(x, state, norm=self.norm1)
            return self.conv2.stream(h, state, self.nin_shortcut(x) if self.in_channels != self.out_channels else x, self.norm2)
        dtype = x.dtype
        h = x
        h = self.norm1(h).to(dtype)
        h = swish(h, inplace=True)
        h = self.conv1(h)

        h = self.norm2(h).to(dtype)
        h = swish(h, inplace=True)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        h += x 
        return h


class Downsample(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, add_temporal_downsample: bool = True):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_downsample else 1 * 2 * 2
        assert out_channels % factor == 0
        self.conv = CausalConv3d(in_channels, out_channels // factor, kernel_size=3)
        self.add_temporal_downsample = add_temporal_downsample
        self.group_size = factor * in_channels // out_channels

    def forward(self, x_list, state=None):
        """state: the causal caches of a streamed clip, of which x is one chunk of frames: only the first frame of the clip (the first
        chunk's) is not paired in time; the other frames of a chunk go by pairs."""
        x = x_list[0]
        x_list.clear()
        r1 = 2 if self.add_temporal_downsample else 1
        first = state is None or self.conv not in state
        h = self.conv(x) if state is None else self.conv.stream(x, state)
        if self.add_temporal_downsample and first:
            h_first = h[:, :, :1, :, :]
            h_first = rearrange(h_first, "b c f (h r2) (w r3) -> b (r2 r3 c) f h w", r2=2, r3=2)
            h_first = torch.cat([h_first, h_first], dim=1)
            h_next = h[:, :, 1:, :, :]
            h_next = rearrange(h_next, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
            h = torch.cat([h_first, h_next], dim=2)
            # shortcut computation
            x_first = x[:, :, :1, :, :]
            x_first = rearrange(x_first, "b c f (h r2) (w r3) -> b (r2 r3 c) f h w", r2=2, r3=2)
            B, C, T, H, W = x_first.shape
            x_first = x_first.view(B, h.shape[1], self.group_size // 2, T, H, W).mean(dim=2)

            x_next = x[:, :, 1:, :, :]
            del x
            x_next = rearrange(x_next, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
            B, C, T, H, W = x_next.shape
            x_next = x_next.view(B, h.shape[1], self.group_size, T, H, W).mean(dim=2)
            shortcut = torch.cat([x_first, x_next], dim=2)
        else:
            h = rearrange(h, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
            shortcut = rearrange(x, "b c (f r1) (h r2) (w r3) -> b (r1 r2 r3 c) f h w", r1=r1, r2=2, r3=2)
            B, C, T, H, W = shortcut.shape
            shortcut = shortcut.view(B, h.shape[1], self.group_size, T, H, W).mean(dim=2)

        h += shortcut
        return h 




class Upsample(nn.Module):
    """Hierarchical upsampling with temporal/ spatial support."""

    def __init__(self, in_channels: int, out_channels: int, add_temporal_upsample: bool = True):
        super().__init__()
        factor = 2 * 2 * 2 if add_temporal_upsample else 1 * 2 * 2
        self.conv = CausalConv3d(in_channels, out_channels * factor, kernel_size=3)
        self.add_temporal_upsample = add_temporal_upsample
        self.repeats = factor * out_channels // in_channels

    def forward(self, x_list: Tensor, state=None):
        """state: the causal caches of a streamed clip, of which x is one chunk of frames: only the first frame of the clip (the first
        chunk's) is not doubled in time."""
        x = x_list[0]
        x_list.clear()
        r1 = 2 if self.add_temporal_upsample else 1
        first = state is None or self.conv not in state
        h = self.conv(x) if state is None else self.conv.stream(x, state)
        if self.add_temporal_upsample and first:
            h_first = h[:, :, :1, :, :]
            h_first = rearrange(h_first, "b (r2 r3 c) f h w -> b c f (h r2) (w r3)", r2=2, r3=2)
            h_first = h_first[:, : h_first.shape[1] // 2]
            h_next = h[:, :, 1:, :, :]
            h_next = rearrange(h_next, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
            h = torch.cat([h_first, h_next], dim=2)

            # shortcut computation
            x_first = x[:, :, :1, :, :]
            x_first = rearrange(x_first, "b (r2 r3 c) f h w -> b c f (h r2) (w r3)", r2=2, r3=2)
            x_first = x_first.repeat_interleave(repeats=self.repeats // 2, dim=1)

            x_next = x[:, :, 1:, :, :]
            del x
            x_next = rearrange(x_next, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
            x_next = x_next.repeat_interleave(repeats=self.repeats, dim=1)
            shortcut = torch.cat([x_first, x_next], dim=2)

        else:
            h = rearrange(h, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
            shortcut = x.repeat_interleave(repeats=self.repeats, dim=1)
            del x
            shortcut = rearrange(shortcut, "b (r1 r2 r3 c) f h w -> b c (f r1) (h r2) (w r3)", r1=r1, r2=2, r3=2)
        h += shortcut
        return h


class Encoder(nn.Module):
    """Hierarchical video encoder with temporal and spatial factorization."""

    def __init__(
        self,
        in_channels: int,
        z_channels: int,
        block_out_channels: Tuple[int, ...],
        num_res_blocks: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        downsample_match_channel: bool = True,
    ):
        super().__init__()
        assert block_out_channels[-1] % (2 * z_channels) == 0

        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks

        # downsampling
        self.conv_in = CausalConv3d(in_channels, block_out_channels[0], kernel_size=3)

        self.down = nn.ModuleList()
        block_in = block_out_channels[0]
        for i_level, ch in enumerate(block_out_channels):
            block = nn.ModuleList()
            block_out = ch
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block

            add_spatial_downsample = bool(i_level < np.log2(ffactor_spatial))
            add_temporal_downsample = add_spatial_downsample and bool(i_level >= np.log2(ffactor_spatial // ffactor_temporal))
            if add_spatial_downsample or add_temporal_downsample:
                assert i_level < len(block_out_channels) - 1
                block_out = block_out_channels[i_level + 1] if downsample_match_channel else block_in
                down.downsample = Downsample(block_in, block_out, add_temporal_downsample)
                block_in = block_out
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = RMS_norm(block_in, images=False)
        self.conv_out = CausalConv3d(block_in, 2 * z_channels, kernel_size=3)

        self.gradient_checkpointing = False

    def stream(self, x, tile, chunk_frames):
        """Encodes x exactly, as forward() does it (no tile overlap, no blending), with the memory of one tile: conv_in and the levels
        that reduce the frame are local (norms per pixel) and causal, so each tile of the reduced frame (side `tile` latent pixels)
        goes through all the frames alone by chunks of chunk_frames latent frames with its own causal caches, every layer on the
        tile's region plus the receptive field still ahead of it, its output cropped to what the later layers need; then the layers at
        the latent resolution, whose attention spans the frames, on the whole clip by chunks of frames, in place."""
        modules = [module for level in self.down for module in (*level.block, *([level.downsample] if hasattr(level, "downsample") else []))]
        local = max(index for index, module in enumerate(modules) if isinstance(module, Downsample)) + 1
        steps = [("same", 1)] + [("unshuffle2",) if isinstance(module, Downsample) else ("same", 2) for module in modules[:local]]
        spatial = 2 ** sum(isinstance(module, Downsample) for module in modules)
        temporal = 2 ** sum(isinstance(module, Downsample) and module.add_temporal_downsample for module in modules)
        batch, _, frames, height, width = x.shape
        latent_frames, latent_height, latent_width = 1 + (frames - 1) // temporal, height // spatial, width // spatial
        h = None
        for y0 in range(0, latent_height, tile):
            (ya, yb), y_crops = halo_plan(steps, y0, min(y0 + tile, latent_height), height)
            for x0 in range(0, latent_width, tile):
                (xa, xb), x_crops = halo_plan(steps, x0, min(x0 + tile, latent_width), width)
                crops = [(slice(*rows), slice(*columns)) for rows, columns in zip(y_crops, x_crops)]
                state = {}
                for start in range(0, latent_frames, chunk_frames):
                    stop = min(start + chunk_frames, latent_frames)
                    part = x[:, :, (0 if start == 0 else 1 + (start - 1) * temporal):1 + (stop - 1) * temporal, ya:yb, xa:xb]
                    rows, columns = crops[0]
                    part = self.conv_in.stream(part, state)[..., rows, columns]
                    for module, (rows, columns) in zip(modules, crops[1:]):
                        part = module([part], state=state)[..., rows, columns]
                        check_abort()
                    if h is None:
                        h = part.new_empty(batch, part.shape[1], latent_frames, latent_height, latent_width)
                    h[:, :, start:stop, y0:y0 + part.shape[3], x0:x0 + part.shape[4]] = part
                    del part
                del state
        step = max(1, (32 << 20) // (h[:, :, 0].numel() * h.element_size()))
        for block in (*modules[local:], self.mid.block_1, self.mid.attn_1, self.mid.block_2):
            if block is self.mid.attn_1:
                h = block(h, by_frames=True)
                continue
            state = {}
            for start in range(0, latent_frames, step):
                h[:, :, start:start + step] = block([h[:, :, start:start + step]], state=state)
                check_abort()
        out, state = h.new_empty(batch, 2 * self.z_channels, latent_frames, latent_height, latent_width), {}
        for start in range(0, latent_frames, step):
            part = h[:, :, start:start + step]
            shortcut = part.unflatten(1, (2 * self.z_channels, -1)).mean(dim=2)
            out[:, :, start:start + step] = self.conv_out.stream(part, state, shortcut, self.norm_out)
        return out

    def forward(self, x: Tensor, stream=None) -> Tensor:
        """Forward pass through the encoder. stream: (tile, chunk_frames) of a streamed encode (see stream), run through forward so
        that the offload hooks load the weights first."""
        if stream is not None:
            return self.stream(x, *stream)
        use_checkpointing = bool(self.training and self.gradient_checkpointing)

        # downsampling
        h = self.conv_in(x)
        for i_level in range(len(self.block_out_channels)):
            for i_block in range(self.num_res_blocks):
                x_list = [h]
                del h
                h = self.down[i_level].block[i_block](x_list)
            if hasattr(self.down[i_level], "downsample"):
                x_list = [h]
                del h
                h = self.down[i_level].downsample(x_list)

        # middle
        x_list = [h]
        del h
        h = self.mid.block_1(x_list)
        h = self.mid.attn_1(h) 
        x_list = [h]
        del h
        h = self.mid.block_2(x_list)

        # end
        group_size = self.block_out_channels[-1] // (2 * self.z_channels)
        shortcut = rearrange(h, "b (c r) f h w -> b c r f h w", r=group_size).mean(dim=2)
        h = self.norm_out(h)
        h = swish(h, inplace=True)
        h = self.conv_out(h)
        h += shortcut
        return h


class Decoder(nn.Module):
    """Hierarchical video decoder with upsampling factories."""

    def __init__(
        self,
        z_channels: int,
        out_channels: int,
        block_out_channels: Tuple[int, ...],
        num_res_blocks: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        upsample_match_channel: bool = True,
    ):
        super().__init__()
        assert block_out_channels[0] % z_channels == 0

        self.z_channels = z_channels
        self.block_out_channels = block_out_channels
        self.num_res_blocks = num_res_blocks

        block_in = block_out_channels[0]
        self.conv_in = CausalConv3d(z_channels, block_in, kernel_size=3)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # upsampling
        self.up = nn.ModuleList()
        for i_level, ch in enumerate(block_out_channels):
            block = nn.ModuleList()
            block_out = ch
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block

            add_spatial_upsample = bool(i_level < np.log2(ffactor_spatial))
            add_temporal_upsample = bool(i_level < np.log2(ffactor_temporal))
            if add_spatial_upsample or add_temporal_upsample:
                assert i_level < len(block_out_channels) - 1
                block_out = block_out_channels[i_level + 1] if upsample_match_channel else block_in
                up.upsample = Upsample(block_in, block_out, add_temporal_upsample)
                block_in = block_out
            self.up.append(up)

        # end
        self.norm_out = RMS_norm(block_in, images=False)
        self.conv_out = CausalConv3d(block_in, out_channels, kernel_size=3)

        self.gradient_checkpointing = False

    def _stream_start(self, z, chunk_bytes=32 << 20):
        """The part of the decoder that sees the whole clip, at the latent resolution: conv_in and the mid block, whose attention spans
        the frames (each frame attends to itself and to the earlier ones). Its residual blocks run by chunks of frames with their
        causal caches, in place: the values of forward(), with the clip itself as the largest tensor."""
        h = self.conv_in(z)
        h.unflatten(1, (self.z_channels, -1)).add_(z.unsqueeze(2))  # (+ z.repeat_interleave(repeats), without the repeated copy)
        step = max(1, chunk_bytes // (h[:, :, 0].numel() * h.element_size()))
        for block in (self.mid.block_1, self.mid.attn_1, self.mid.block_2):
            if block is self.mid.attn_1:
                h = block(h, by_frames=True)
                continue
            state = {}
            for start in range(0, h.shape[2], step):
                h[:, :, start:start + step] = block([h[:, :, start:start + step]], state=state)
                check_abort()
        return h

    def _stream_modules(self):
        """The layers after the mid block, in order, and their halo_plan steps (then the output layers')."""
        modules = [module for level in self.up for module in (*level.block, *([level.upsample] if hasattr(level, "upsample") else []))]
        return modules, [("shuffle2",) if isinstance(module, Upsample) else ("same", 2) for module in modules] + [("same", 1)]

    def stream_shape(self, z):
        """(frames, height, width) of the video decoded from the latent z."""
        levels = [level.upsample for level in self.up if hasattr(level, "upsample")]
        temporal = 2 ** sum(upsample.add_temporal_upsample for upsample in levels)
        return 1 + (z.shape[2] - 1) * temporal, z.shape[3] * 2 ** len(levels), z.shape[4] * 2 ** len(levels)

    def stream(self, z, tile, chunk_frames, sink):
        """Decodes z exactly, as forward() does it (no tile overlap, no blending), with the memory of one tile: the start that sees the
        whole clip (_stream_start) at the latent resolution, kept in RAM; then every other layer is local (norms per pixel) and causal,
        so each output tile (side `tile` latent pixels) goes through all the frames alone by chunks of chunk_frames latent frames with
        its own causal caches, every layer on the tile's region plus the receptive field still ahead of it, its output cropped to what
        the later layers need. sink(first frame, rows, columns, pixels) receives each tile's chunk of frames as it is decoded."""
        sample = self._stream_start(z)
        device, sample = sample.device, sample.cpu()
        batch, _, frames, height, width = sample.shape
        modules, steps = self._stream_modules()
        spatial = 2 ** sum(isinstance(module, Upsample) for module in modules)
        temporal = 2 ** sum(isinstance(module, Upsample) and module.add_temporal_upsample for module in modules)
        for y0 in range(0, height, tile):
            (ya, yb), y_crops = halo_plan(steps, y0 * spatial, min(y0 + tile, height) * spatial, height)
            for x0 in range(0, width, tile):
                (xa, xb), x_crops = halo_plan(steps, x0 * spatial, min(x0 + tile, width) * spatial, width)
                crops = [(slice(*rows), slice(*columns)) for rows, columns in zip(y_crops, x_crops)]
                region = transfer.from_ram(sample[:, :, :, ya:yb, xa:xb], device)
                state = {}
                for start in range(0, frames, chunk_frames):
                    h = region[:, :, start:start + chunk_frames]
                    for module, (rows, columns) in zip(modules, crops):
                        h = module([h], state=state)[..., rows, columns]
                        check_abort()
                    rows, columns = crops[-1]
                    h = self.conv_out.stream(h, state, norm=self.norm_out)[..., rows, columns]
                    first_frame = 0 if start == 0 else 1 + (start - 1) * temporal
                    sink(first_frame, slice(y0 * spatial, y0 * spatial + h.shape[3]), slice(x0 * spatial, x0 * spatial + h.shape[4]), h)
                    del h
                del region, state

    def forward(self, z: Tensor, stream=None) -> Tensor:
        """Forward pass through the decoder. stream: (tile, chunk_frames, sink) of a streamed decode (see stream), run through forward
        so that the offload hooks load the weights first."""
        if stream is not None:
            return self.stream(z, *stream)

        # z to block_in
        repeats = self.block_out_channels[0] // (self.z_channels)
        h = self.conv_in(z) + z.repeat_interleave(repeats=repeats, dim=1)

        # middle
        h_list = [h]
        del h
        h = self.mid.block_1(h_list)
        # h_list = [h]
        # del h
        h = self.mid.attn_1(h)
        h_list = [h]
        del h
        h = self.mid.block_2(h_list)

        # upsampling
        for i_level in range(len(self.block_out_channels)):
            for i_block in range(self.num_res_blocks + 1):
                x_list = [h]
                del h
                h = self.up[i_level].block[i_block](x_list )
            if hasattr(self.up[i_level], "upsample"):
                x_list = [h]
                del h
                h = self.up[i_level].upsample(x_list )

        # end
        h = self.norm_out(h).to(z.dtype)
        h = swish(h, inplace=True)
        h = self.conv_out(h)
        return h


def _stream_tile(vae_config, device_mem_capacity, costs, stage, height, width, minimum, ratio):
    """Latent pixels of the largest streamed tile of `stage` ("decode", "encode") that the VAE Tiling budget holds, by steps of 64
    pixels: 0 for the whole height x width frame (costs["whole_" + stage]: GiB per megapixel), None when not even a tile of `minimum`
    pixels fits. Half the room above the weights without the MMGP VRAM allocator."""
    room = (tiling.vram_budget(vae_config, device_mem_capacity) - costs["weights"] * tiling.GIB) / (1 if mmgp_vram_allocator() else 2)
    if height and width and costs["whole_" + stage] * height * width / 1e6 * tiling.GIB <= room:
        return 0
    if tiling.estimate(costs[stage], minimum * minimum) > room:
        return None
    return tiling.largest_tile(vae_config, device_mem_capacity, {"weights": costs["weights"], stage: costs[stage]}, height, width, step=64, minimum=minimum) // ratio


class AutoencoderKLConv3D(ModelMixin, ConfigMixin):
    """KL regularized 3D Conv VAE with advanced tiling and slicing strategies."""
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        latent_channels: int,
        block_out_channels: Tuple[int, ...],
        layers_per_block: int,
        ffactor_spatial: int,
        ffactor_temporal: int,
        sample_size: int,
        sample_tsize: int,
        scaling_factor: float = None,
        shift_factor: Optional[float] = None,
        downsample_match_channel: bool = True,
        upsample_match_channel: bool = True,
    ):
        super().__init__()
        self.ffactor_spatial = ffactor_spatial
        self.ffactor_temporal = ffactor_temporal
        self.scaling_factor = scaling_factor
        self.shift_factor = shift_factor

        self.encoder = Encoder(
            in_channels=in_channels,
            z_channels=latent_channels,
            block_out_channels=block_out_channels,
            num_res_blocks=layers_per_block,
            ffactor_spatial=ffactor_spatial,
            ffactor_temporal=ffactor_temporal,
            downsample_match_channel=downsample_match_channel,
        )
        self.decoder = Decoder(
            z_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=list(reversed(block_out_channels)),
            num_res_blocks=layers_per_block,
            ffactor_spatial=ffactor_spatial,
            ffactor_temporal=ffactor_temporal,
            upsample_match_channel=upsample_match_channel,
        )

        self.use_slicing = False
        self.use_spatial_tiling = False
        self.use_temporal_tiling = False

        # only relevant if vae tiling is enabled
        self.tile_sample_min_size = sample_size
        self.tile_latent_min_size = sample_size // ffactor_spatial
        self.tile_sample_min_tsize = sample_tsize
        self.tile_latent_min_tsize = sample_tsize // ffactor_temporal
        self.tile_overlap_factor = 0.25

        self._tile_parallelism_enabled = False

    def set_tile_sample_min_size(self, sample_size: int, tile_overlap_factor: float = 0.25, sample_tsize: Optional[int] = None):
        self.tile_sample_min_size = sample_size
        self.tile_latent_min_size = sample_size // self.ffactor_spatial
        self.tile_overlap_factor = tile_overlap_factor

        if sample_tsize is not None:
            self.tile_sample_min_tsize = sample_tsize
            self.tile_latent_min_tsize = sample_tsize // self.ffactor_temporal

        assert (self.tile_latent_min_size * self.tile_overlap_factor).is_integer(), \
            "self.tile_latent_min_size multiplied by tile_overlap_factor must be an integer"
        assert (self.tile_latent_min_tsize * self.tile_overlap_factor).is_integer(), \
            "self.tile_latent_min_tsize multiplied by tile_overlap_factor must be an integer"


    def _set_gradient_checkpointing(self, module, value=False):
        """Enable or disable gradient checkpointing on encoder and decoder."""
        if isinstance(module, (Encoder, Decoder)):
            module.gradient_checkpointing = value


    def enable_temporal_tiling(self, use_tiling: bool = True):
        self.use_temporal_tiling = use_tiling

    def disable_temporal_tiling(self):
        self.enable_temporal_tiling(False)

    def enable_spatial_tiling(self, use_tiling: bool = True):
        self.use_spatial_tiling = use_tiling

    def disable_spatial_tiling(self):
        self.enable_spatial_tiling(False)

    def enable_tiling(self, use_tiling: bool = True):
        self.enable_spatial_tiling(use_tiling)
        self.enable_temporal_tiling(use_tiling)

    def disable_tiling(self):
        self.disable_spatial_tiling()
        self.disable_temporal_tiling()

    def enable_slicing(self):
        self.use_slicing = True

    def disable_slicing(self):
        self.use_slicing = False

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int):
        """Blend tensor b horizontally into a at blend_extent region."""
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (x / blend_extent)
        return b

    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int):
        """Blend tensor b vertically into a at blend_extent region."""
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (y / blend_extent)
        return b

    def blend_t(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int):
        """Blend tensor b temporally into a at blend_extent region."""
        blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
        for x in range(blend_extent):
            b[:, :, x, :, :] = a[:, :, -blend_extent + x, :, :] * (1 - x / blend_extent) + b[:, :, x, :, :] * (x / blend_extent)
        return b

    def spatial_tiled_encode(self, x: torch.Tensor):
        """Tiled spatial encoding for large inputs via overlapping."""
        B, C, T, H, W = x.shape
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        rows = []
        for i in range(0, H, overlap_size):
            row = []
            for j in range(0, W, overlap_size):
                tile = x[:, :, :, i: i + self.tile_sample_min_size, j: j + self.tile_sample_min_size]
                tile = self.encoder(tile)
                row.append(tile)
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
        moments = torch.cat(result_rows, dim=-2)
        return moments

    def temporal_tiled_encode(self, x: torch.Tensor):
        """Tiled temporal encoding for large video sequences."""
        B, C, T, H, W = x.shape
        overlap_size = int(self.tile_sample_min_tsize * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_tsize * self.tile_overlap_factor)
        t_limit = self.tile_latent_min_tsize - blend_extent

        row = []
        for i in range(0, T, overlap_size):
            tile = x[:, :, i: i + self.tile_sample_min_tsize + 1, :, :]
            if self.use_spatial_tiling and (tile.shape[-1] > self.tile_sample_min_size or tile.shape[-2] > self.tile_sample_min_size):
                tile = self.spatial_tiled_encode(tile)
            else:
                tile = self.encoder(tile)
            if i > 0:
                tile = tile[:, :, 1:, :, :]
            row.append(tile)
        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_extent)
                result_row.append(tile[:, :, :t_limit, :, :])
            else:
                result_row.append(tile[:, :, :t_limit + 1, :, :])
        moments = torch.cat(result_row, dim=-3)
        return moments

    def enable_tile_parallelism(self):
        self._tile_parallelism_enabled = True
    
    def disable_tile_parallelism(self):
        self._tile_parallelism_enabled = False

    def tile_parallel_spatial_tiled_decode(self, z: torch.Tensor):
        B, C, T, H, W = z.shape
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent


        rank = get_parallel_state().sp_rank
        world_size = get_parallel_state().sp

        num_rows = math.ceil(H / overlap_size)
        num_cols = math.ceil(W / overlap_size)
        total_tiles = num_rows * num_cols
        tiles_per_rank = math.ceil(total_tiles / world_size)


        my_linear_indices = list(range(rank, total_tiles, world_size))
        decoded_tiles = []
        decoded_metas = []
        H_out_std = self.tile_sample_min_size
        W_out_std = self.tile_sample_min_size
        for lin_idx in my_linear_indices:
            ri = lin_idx // num_cols
            rj = lin_idx % num_cols
            i = ri * overlap_size
            j = rj * overlap_size
            tile = z[:, :, :, i:i + self.tile_latent_min_size, j:j + self.tile_latent_min_size]
            dec = self.decoder(tile)

            pad_h = max(0, H_out_std - dec.shape[-2])
            pad_w = max(0, W_out_std - dec.shape[-1])
            if pad_h > 0 or pad_w > 0:
                dec = F.pad(dec, (0, pad_w, 0, pad_h, 0, 0), "constant", 0)
            decoded_tiles.append(dec)
            decoded_metas.append(torch.tensor([ri, rj, pad_w, pad_h], device=z.device, dtype=torch.int64))
        
        while len(decoded_tiles) < tiles_per_rank:
            zero_tile = torch.zeros(
                [1, 3, (T - 1) * self.ffactor_temporal + 1, self.tile_sample_min_size, self.tile_sample_min_size],
                device=dec.device,
                dtype=dec.dtype
            )
            decoded_tiles.append(zero_tile)
            meta_tensor = torch.tensor(
                [-1, -1, self.tile_sample_min_size, self.tile_sample_min_size],
                device=z.device,
                dtype=torch.int64
            )
            decoded_metas.append(meta_tensor)

        decoded_tiles = torch.stack(decoded_tiles, dim=0)
        decoded_metas = torch.stack(decoded_metas, dim=0)
        
        tiles_gather_list = [torch.empty_like(decoded_tiles) for _ in range(world_size)]
        metas_gather_list = [torch.empty_like(decoded_metas) for _ in range(world_size)]

        dist.all_gather(tiles_gather_list, decoded_tiles, group=get_parallel_state().sp_group)
        dist.all_gather(metas_gather_list, decoded_metas, group=get_parallel_state().sp_group)

        if rank != 0:
            return torch.empty(0, device=z.device)

        rows = [[None for _ in range(num_cols)] for _ in range(num_rows)]
        for r in range(world_size):
            gathered_tiles_r = tiles_gather_list[r]  # [tiles_per_rank, B, C, T, H, W]
            gathered_metas_r = metas_gather_list[r]  # [tiles_per_rank, 4]
            for k in range(gathered_tiles_r.shape[0]):
                ri = int(gathered_metas_r[k][0])
                rj = int(gathered_metas_r[k][1])
                if ri < 0 or rj < 0:
                    continue
                if ri < num_rows and rj < num_cols:
                    # remove padding
                    pad_w = int(gathered_metas_r[k][2])
                    pad_h = int(gathered_metas_r[k][3])
                    h_end = None if pad_h == 0 else -pad_h
                    w_end = None if pad_w == 0 else -pad_w
                    rows[ri][rj] = gathered_tiles_r[k][:, :, :, :h_end, :w_end]

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if tile is None:
                    continue
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))

        dec = torch.cat(result_rows, dim=-2)
        return dec

    def spatial_tiled_decode(self, z: torch.Tensor):
        if self._tile_parallelism_enabled:
            return self.tile_parallel_spatial_tiled_decode(z)

        B, C, T, H, W = z.shape
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent

        rows = []
        for i in range(0, H, overlap_size):
            row = []
            for j in range(0, W, overlap_size):
                tile = z[:, :, :, i: i + self.tile_latent_min_size, j: j + self.tile_latent_min_size]
                decoded = self.decoder(tile)
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
        dec = torch.cat(result_rows, dim=-2)
        return dec

    def temporal_tiled_decode(self, z: torch.Tensor):
        """Tiled temporal decoding for long sequence latents."""
        B, C, T, H, W = z.shape
        overlap_size = int(self.tile_latent_min_tsize * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_tsize * self.tile_overlap_factor)
        t_limit = self.tile_sample_min_tsize - blend_extent
        assert 0 < overlap_size < self.tile_latent_min_tsize

        row = []
        for i in range(0, T, overlap_size):
            tile = z[:, :, i: i + self.tile_latent_min_tsize + 1, :, :]
            if self.use_spatial_tiling and (tile.shape[-1] > self.tile_latent_min_size or tile.shape[-2] > self.tile_latent_min_size):
                decoded = self.spatial_tiled_decode(tile)
            else:
                decoded = self.decoder(tile)
            if i > 0:
                decoded = decoded[:, :, 1:, :, :]
            row.append(decoded)

        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_extent)
                result_row.append(tile[:, :, :t_limit, :, :])
            else:
                result_row.append(tile[:, :, :t_limit + 1, :, :])
        dec = torch.cat(result_row, dim=-3)
        return dec

    _blend_ = staticmethod(blend_edge_)  # (the values of blend_v / blend_h / blend_t)

    def decode_to_ram(self, z: torch.Tensor, to_ram):
        """The tiled decode of z (temporal tiles, each by spatial tiles, blended as temporal_tiled_decode and spatial_tiled_decode
        blend them) returned in RAM: each tile, once blended with the edges kept from its neighbours, goes through to_ram (a function of
        a decoded piece on the GPU: its final form, a dtype conversion or bytes) and is written to its place without waiting for the
        GPU. VRAM holds one decoded tile, the edges of a row of tiles and the frames shared with the next temporal tile, instead of
        every decoded tile and the assembled clip."""
        B, C, T, H, W = z.shape
        fs, ft = self.ffactor_spatial, self.ffactor_temporal
        temporal = self.use_temporal_tiling and T > self.tile_latent_min_tsize
        spatial = self.use_spatial_tiling and (W > self.tile_latent_min_size or H > self.tile_latent_min_size)
        t_stride = int(self.tile_latent_min_tsize * (1 - self.tile_overlap_factor))
        t_extent = int(self.tile_sample_min_tsize * self.tile_overlap_factor)
        t_limit = self.tile_sample_min_tsize - t_extent
        stride = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - extent
        t_starts = list(range(0, T, t_stride)) if temporal else [0]
        rows, columns = (list(range(0, H, stride)), list(range(0, W, stride))) if spatial else ([0], [0])
        out = previous = current = None
        position = previous_frames = 0
        with vae_decoding_progress(len(t_starts) * len(rows) * len(columns), self.decoder), transfer.RamWriter("decoded video", depth=64, max_bytes=256 << 20) as writer:
            for k, t0 in enumerate(t_starts):
                part = z[:, :, t0 : t0 + self.tile_latent_min_tsize + 1] if temporal else z
                bottoms, y, kept = [None] * len(columns), 0, 0
                for i, r0 in enumerate(rows):
                    left, x = None, 0
                    for j, c0 in enumerate(columns):
                        tile = self.decoder(part[:, :, :, r0 : r0 + self.tile_latent_min_size, c0 : c0 + self.tile_latent_min_size] if spatial else part)
                        if k > 0:
                            tile = tile[:, :, 1:]
                        if i > 0:
                            e = min(bottoms[j].shape[-2], tile.shape[-2], extent)
                            self._blend_(tile[:, :, :, :e], bottoms[j][:, :, :, bottoms[j].shape[-2] - e :], 3)
                        if j > 0:
                            e = min(left.shape[-1], tile.shape[-1], extent)
                            self._blend_(tile[..., :e], left[..., left.shape[-1] - e :], 4)
                        if spatial:  # the edges the next tiles blend with (rows below, columns on the right), as blended here
                            bottoms[j] = tile[:, :, :, -extent:].clone() if i < len(rows) - 1 else None
                            left = tile[..., -extent:].clone() if j < len(columns) - 1 else None
                            tile = tile[:, :, :, :row_limit, :row_limit]
                        frames, height, width = tile.shape[2:]
                        if k > 0:  # the first frames blended with the last ones of the temporal tile before
                            e = min(previous_frames, frames, t_extent)
                            self._blend_(tile[:, :, :e], previous[:, :, previous.shape[2] - e :, y : y + height, x : x + width], 2)
                        if temporal and k < len(t_starts) - 1:
                            if current is None:
                                current = torch.empty((*tile.shape[:2], min(frames, t_extent), H * fs, W * fs), dtype=tile.dtype, device=tile.device)
                            current[:, :, :, y : y + height, x : x + width].copy_(tile[:, :, frames - current.shape[2] :])
                        kept = min(frames, t_limit + (1 if k == 0 else 0)) if temporal else frames
                        piece = to_ram(tile[:, :, :kept])
                        if out is None:
                            total = sum(min((min(self.tile_latent_min_tsize + 1, T - start) - 1) * ft + (1 if n == 0 else 0), t_limit + (1 if n == 0 else 0))
                                        for n, start in enumerate(t_starts)) if temporal else frames
                            out = torch.empty((*piece.shape[:2], total, H * fs, W * fs), dtype=piece.dtype, device="cpu")
                        writer.write(out[:, :, position : position + kept, y : y + height, x : x + width], piece)
                        x += width
                        del tile, piece
                    y += height
                position += kept
                previous, current, previous_frames = current, None, frames
        return out

    def streamed_decode(self, z, tile, chunk_frames, to_ram=None):
        """The exact decode of z (Decoder.stream: no tile overlap, no blending) by tiles of `tile` latent pixels and chunks of
        chunk_frames latent frames. to_ram: a function of each decoded piece on the GPU (its final form: a dtype conversion, bytes); the
        video is then returned in RAM, each piece written to its place as it is decoded, without the caller waiting for the GPU."""
        frames, height, width = self.decoder.stream_shape(z)
        out = None

        def sink(frame, rows, columns, pixels):
            nonlocal out
            piece = pixels if to_ram is None else to_ram(pixels)
            if out is None:
                out = torch.empty((*piece.shape[:2], frames, height, width), dtype=piece.dtype, device="cpu" if to_ram is not None else piece.device)
            if to_ram is None:
                out[:, :, frame:frame + piece.shape[2], rows, columns] = piece
            else:
                writer.write(out[:, :, frame:frame + piece.shape[2], rows, columns], piece)

        pieces = -(-z.shape[3] // tile) * -(-z.shape[4] // tile) * -(-z.shape[2] // chunk_frames)
        with vae_decoding_progress(pieces, self.decoder.conv_out.conv), transfer.RamWriter("decoded video", depth=64, max_bytes=256 << 20) as writer:
            self.decoder(z, stream=(tile, chunk_frames, sink))
        return out

    def streamed_encode(self, x, tile, chunk_frames):
        """The exact encode of x (Encoder.stream: no tile overlap, no blending) by tiles of `tile` latent pixels and chunks of
        chunk_frames latent frames: the moments of the latent distribution."""
        return self.encoder(x, stream=(tile, chunk_frames))

    def encode(self, x: Tensor, return_dict: bool = True):

        set_phase_status("VAE Encoding")
        fs, ft = self.ffactor_spatial, self.ffactor_temporal
        tile = self.picture_tile if x.shape[2] == 1 else self.encode_tile
        if ((self.use_spatial_tiling or self.use_temporal_tiling) and x.is_cuda and tile is not None and (x.shape[2] - 1) % ft == 0
                and x.shape[3] % fs == 0 and x.shape[4] % fs == 0):
            # the exact streamed encode, by tiles of the size the VAE Tiling budget holds (0: the whole frame)
            tile = tile or max(x.shape[3:]) // fs
            h = torch.cat([self.streamed_encode(part, tile, self.STREAM_CHUNK) for part in (x.split(1) if self.use_slicing and x.shape[0] > 1 else [x])])
            posterior = DiagonalGaussianDistribution(h)
            return AutoencoderKLOutput(latent_dist=posterior) if return_dict else (posterior,)

        def _encode(x):
            if self.use_temporal_tiling and x.shape[-3] > self.tile_sample_min_tsize:
                return self.temporal_tiled_encode(x)
            if self.use_spatial_tiling and (x.shape[-1] > self.tile_sample_min_size or x.shape[-2] > self.tile_sample_min_size):
                return self.spatial_tiled_encode(x)
            return self.encoder(x)

        assert len(x.shape) == 5  # (B, C, T, H, W)

        if self.use_slicing and x.shape[0] > 1:
            encoded_slices = [_encode(x_slice) for x_slice in x.split(1)]
            h = torch.cat(encoded_slices)
        else:
            h = _encode(x)
        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)

        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z: Tensor, return_dict: bool = True, generator=None, to_ram=None):
        """to_ram: a function of a decoded piece on the GPU (its final form); the video is then returned in RAM, written piece by piece
        as it is decoded. With tiling on a CUDA device: the exact streamed decode (streamed_decode), by tiles of the size the VAE Tiling
        budget holds (stream_tile, 0: the whole frame) and chunks of STREAM_CHUNK latent frames; elsewhere blended tiles."""
        if (self.use_spatial_tiling or self.use_temporal_tiling) and z.is_cuda and not self._tile_parallelism_enabled:
            tile = self.tile_latent_min_size if self.stream_tile is None else self.stream_tile or max(z.shape[3:])
            pieces = [self.streamed_decode(part, tile, self.STREAM_CHUNK, to_ram) for part in (z.split(1) if self.use_slicing and z.shape[0] > 1 else [z])]
            decoded = pieces[0] if len(pieces) == 1 else torch.cat(pieces)
            return DecoderOutput(sample=decoded) if return_dict else (decoded,)
        if to_ram is not None and not self._tile_parallelism_enabled and not (self.use_slicing and z.shape[0] > 1):
            decoded = self.decode_to_ram(z, to_ram)
            return DecoderOutput(sample=decoded) if return_dict else (decoded,)

        tiles = z.shape[0] if self.use_slicing else 1
        if self.use_temporal_tiling and z.shape[2] > self.tile_latent_min_tsize:
            tiles *= len(range(0, z.shape[2], int(self.tile_latent_min_tsize * (1 - self.tile_overlap_factor))))
        if self.use_spatial_tiling and (z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size):
            stride = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
            tiles *= len(range(0, z.shape[-2], stride)) * len(range(0, z.shape[-1], stride))
        with vae_decoding_progress(tiles, self.decoder):
            def _decode(z):
                if self.use_temporal_tiling and z.shape[-3] > self.tile_latent_min_tsize:
                    return self.temporal_tiled_decode(z)
                if self.use_spatial_tiling and (z.shape[-1] > self.tile_latent_min_size or z.shape[-2] > self.tile_latent_min_size):
                    return self.spatial_tiled_decode(z)
                return self.decoder(z)

            if self.use_slicing and z.shape[0] > 1:
                decoded_slices = [_decode(z_slice) for z_slice in z.split(1)]
                decoded = torch.cat(decoded_slices)
            else:
                decoded = _decode(z)
            if to_ram is not None:
                decoded = to_ram(decoded).cpu()

            if not return_dict:
                return (decoded,)

            return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_posterior: bool = True,
        return_dict: bool = True
    ):
        """Forward autoencoder pass. Returns both reconstruction and optionally the posterior."""
        posterior = self.encode(sample).latent_dist
        z = posterior.sample() if sample_posterior else posterior.mode()
        dec = self.decode(z).sample
        return DecoderOutput(sample=dec, posterior=posterior) if return_dict else (dec, posterior)





    # CUDA devices: exact streamed tiles, to decode and to encode. Their result does not depend on their size (70 dB between sizes:
    # FP16 rounding), their time does: each tile computes a halo of its own (wider for the encoder), so the largest tile the VAE
    # Tiling budget holds is the fastest, the whole frame when it fits (_stream_tile). 1280x720 x 65, tiles of 192 / 256 / 384 / 512
    # pixels / whole frames: decode 20.1 / 15.7 / 12.7 / 11.7 / 9.0 s (blended tiles of 128:48: 29.2 s), encode 14.2 / 10.5 / 8.4 /
    # 7.5 / 5.0 s (17.3 s). Tiles of 128 pixels are slower than the blended ones (30.2 and 22.0 s): the decoder starts at
    # 192 pixels, whose VRAM is that of the blended tiles of 128:32; the encoder keeps its blended tiles (TILES) below 192.
    # STREAM_COSTS, with the Triton kernels (True) and with PyTorch's layers (False): megapixels of a tile and GiB above the weights
    # (+ 10 %), then GiB per megapixel of a whole frame (no halo), chunks of STREAM_CHUNK latent frames (more frames per chunk: the
    # same time, more VRAM). A single picture (a start image) holds PICTURE of a chunk of 4 frames (measured 0.25 to 0.38: tiles of
    # 192 pixels 0.37 GiB, 0.64 with PyTorch's layers; blended tiles: 0.06 GiB but 2.5 times the time and latents 38 to 43 dB from
    # the untiled encode's, where the stream is at 68 to 70): it always streams, at 192 pixels when nothing larger fits.
    # Blended tiles (the encoder of videos on small budgets; devices without CUDA), measured on footage (832x480 x 65 and 1280x720 x 33, FP16):
    # the size of a tile in pixels changes neither the quality (0.2 dB) nor the time (a few %), its frames do: tiles of 32 frames
    # lose 1 to 2 dB against 48 or 64. TILES: (tile pixels, frames): GiB above the weights (+ 10 %), the first that the budget
    # holds; WEIGHTS: encoder and decoder, GiB.
    STREAM_COSTS = {True: {"weights": 2.35, "decode": [(0.0369, 0.96), (0.0655, 1.25), (0.147, 2.3), (0.262, 3.95), (0.553, 7.65), (0.737, 10.15)], "whole_decode": 13.6,
                           "encode": [(0.0369, 1.62), (0.0655, 2.2), (0.147, 3.06), (0.262, 4.72), (0.553, 7.05)], "whole_encode": 10.4},
                    False: {"weights": 2.35, "decode": [(0.0369, 1.0), (0.0655, 1.4), (0.262, 4.65)], "whole_decode": 16.1,
                            "encode": [(0.0369, 1.95), (0.0655, 2.65), (0.262, 5.75)], "whole_encode": 14.0}}
    STREAM_CHUNK, STREAM_MINIMUM, PICTURE = 1, 192, 0.4
    stream_tile = None  # latent pixels of a streamed decode tile, chosen by get_VAE_tile_size (0: the whole frame; None: not chosen)
    encode_tile = None  # the same for the streamed encode of a video (None: blended tiles)
    picture_tile = None  # the same for the streamed encode of a single picture (None: not chosen, blended tiles)
    TILES, WEIGHTS = (((128, 48), 1.35), ((128, 32), 0.95)), 2.35

    def get_VAE_tile_size(self, vae_config, device_mem_capacity, mixed_precision, output_height=None, output_width=None):
        self.stream_tile = self.encode_tile = self.picture_tile = None
        if torch.cuda.is_available():
            costs = self.STREAM_COSTS[vae_ops.replicate_conv_kernels(torch.device("cuda", torch.cuda.current_device()))]
            picture = dict(costs, encode=[(megapixels, need * self.PICTURE) for megapixels, need in costs["encode"]], whole_encode=costs["whole_encode"] * self.PICTURE)
            choose = lambda costs, stage: _stream_tile(vae_config, device_mem_capacity, costs, stage, output_height, output_width, self.STREAM_MINIMUM, self.ffactor_spatial)
            floor = lambda tile: self.STREAM_MINIMUM // self.ffactor_spatial if tile is None else tile
            self.encode_tile, self.picture_tile, self.stream_tile = choose(costs, "encode"), floor(choose(picture, "encode")), floor(choose(costs, "decode"))
        room = tiling.vram_budget(vae_config, device_mem_capacity) / tiling.GIB - self.WEIGHTS
        if not mmgp_vram_allocator():  # (PyTorch's allocator reserves more than it allocates)
            room /= 2
        sample_size, sample_tsize = next((tile for tile, need in self.TILES if need <= room), self.TILES[-1][0])
        return {"tile_sample_min_size": sample_size, "tile_sample_min_tsize": sample_tsize, "tile_latent_min_size": sample_size // self.ffactor_spatial,
                "tile_latent_min_tsize": sample_tsize // self.ffactor_temporal, "tile_overlap_factor": 0.25}
