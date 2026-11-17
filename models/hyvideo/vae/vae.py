from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.utils import BaseOutput, is_torch_version
from diffusers.utils.torch_utils import randn_tensor
from diffusers.models.attention_processor import SpatialNorm
from mmgp import transfer
from shared.image_vae.tiles import halo_plan
from shared.kernels import vae_ops
from shared.utils.phase_progress import check_abort
from .unet_causal_3d_blocks import _fused
from .unet_causal_3d_blocks import (
    CausalConv3d,
    ResnetBlockCausal3D,
    UNetMidBlockCausal3D,
    UpsampleCausal3D,
    fused_norm_silu,
    get_down_block3d,
    get_up_block3d,
)


def _stream_plan(modules, start, stop, length):
    """Backward plan along one spatial axis of DecoderCausal3D.stream_tiles (shared.image_vae.tiles.halo_plan): for the output range
    [start, stop), the input range of the first module (axis length `length` at its resolution) and the crop of each module's output,
    then of the output layers' output: the receptive field still ahead (2 per resnet, 1 per output convolution, the 2-tap parity
    windows of an upsampler), clamped to the frame where the replicate padding is the decoder's own."""
    steps = [("nearest2",) if isinstance(module, UpsampleCausal3D) else ("same", 2) for module in modules] + [("same", 1)]
    return halo_plan(steps, start, stop, length)


def _unmerged(total, part):
    """FP64 (count, mean, sum of squared deviations) rows of what was merged with `part` to give `total` (the inverse of the merge)."""
    count = total[:, 0] - part[:, 0]
    mean = (total[:, 0] * total[:, 1] - part[:, 0] * part[:, 1]) / count
    delta = mean - part[:, 1]
    return torch.stack((count, mean, (total[:, 2] - part[:, 2] - delta * delta * (part[:, 0] * count / total[:, 0])).clamp_(min=0)), 1)


@dataclass
class DecoderOutput(BaseOutput):
    r"""
    Output of decoding method.

    Args:
        sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            The decoded output sample from the last layer of the model.
    """

    sample: torch.FloatTensor


class EncoderCausal3D(nn.Module):
    r"""
    The `EncoderCausal3D` layer of a variational autoencoder that encodes its input into a latent representation.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        down_block_types (`Tuple[str, ...]`, *optional*, defaults to `("DownEncoderBlock2D",)`):
            The types of down blocks to use. See `~diffusers.models.unet_2d_blocks.get_down_block` for available
            options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        double_z (`bool`, *optional*, defaults to `True`):
            Whether to double the number of output channels for the last block.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = ("DownEncoderBlockCausal3D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        mid_block_add_attention=True,
        time_compression_ratio: int = 4,
        spatial_compression_ratio: int = 8,
        disable_causal: bool = False,
        mid_block_causal_attn: bool = True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = CausalConv3d(in_channels, block_out_channels[0], kernel_size=3, stride=1, disable_causal=disable_causal)
        self.mid_block = None
        self.down_blocks = nn.ModuleList([])

        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            num_spatial_downsample_layers = int(np.log2(spatial_compression_ratio))
            num_time_downsample_layers = int(np.log2(time_compression_ratio))

            if time_compression_ratio == 4:
                add_spatial_downsample = bool(i < num_spatial_downsample_layers)
                add_time_downsample = bool(i >= (len(block_out_channels) - 1 - num_time_downsample_layers) and not is_final_block)
            elif time_compression_ratio == 8:
                add_spatial_downsample = bool(i < num_spatial_downsample_layers)
                add_time_downsample = bool(i < num_time_downsample_layers)
            else:
                raise ValueError(f"Unsupported time_compression_ratio: {time_compression_ratio}")
            
            downsample_stride_HW = (2, 2) if add_spatial_downsample else (1, 1)
            downsample_stride_T = (2, ) if add_time_downsample else (1, )
            downsample_stride = tuple(downsample_stride_T + downsample_stride_HW)
            down_block = get_down_block3d(
                down_block_type,
                num_layers=self.layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=bool(add_spatial_downsample or add_time_downsample),
                downsample_stride=downsample_stride,
                resnet_eps=1e-6,
                downsample_padding=0,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=None,
                disable_causal=disable_causal,
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = UNetMidBlockCausal3D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default",
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=None,
            add_attention=mid_block_add_attention,
            disable_causal=disable_causal,
            causal_attention=mid_block_causal_attn,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = CausalConv3d(block_out_channels[-1], conv_out_channels, kernel_size=3, disable_causal=disable_causal)

        self.gradient_checkpointing = False

    def forward(self, sample: torch.FloatTensor) -> torch.FloatTensor:
        r"""The forward method of the `EncoderCausal3D` class."""
        assert len(sample.shape) == 5, "The input tensor should have 5 dimensions"

        sample = self.conv_in(sample)

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # down
            if is_torch_version(">=", "1.11.0"):
                for down_block in self.down_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(down_block), sample, use_reentrant=False
                    )
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, use_reentrant=False
                )
            else:
                for down_block in self.down_blocks:
                    sample = torch.utils.checkpoint.checkpoint(create_custom_forward(down_block), sample)
                # middle
                sample = torch.utils.checkpoint.checkpoint(create_custom_forward(self.mid_block), sample)

        else:
            # down
            for down_block in self.down_blocks:
                sample = down_block(sample)

            # middle
            sample = self.mid_block(sample)

        # post-process
        if fused_norm_silu(sample):
            return self.conv_out.norm_silu_forward(sample, self.conv_norm_out)
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample


class DecoderCausal3D(nn.Module):
    r"""
    The `DecoderCausal3D` layer of a variational autoencoder that decodes its latent representation into an output sample.

    Args:
        in_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        out_channels (`int`, *optional*, defaults to 3):
            The number of output channels.
        up_block_types (`Tuple[str, ...]`, *optional*, defaults to `("UpDecoderBlock2D",)`):
            The types of up blocks to use. See `~diffusers.models.unet_2d_blocks.get_up_block` for available options.
        block_out_channels (`Tuple[int, ...]`, *optional*, defaults to `(64,)`):
            The number of output channels for each block.
        layers_per_block (`int`, *optional*, defaults to 2):
            The number of layers per block.
        norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups for normalization.
        act_fn (`str`, *optional*, defaults to `"silu"`):
            The activation function to use. See `~diffusers.models.activations.get_activation` for available options.
        norm_type (`str`, *optional*, defaults to `"group"`):
            The normalization type to use. Can be either `"group"` or `"spatial"`.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = ("UpDecoderBlockCausal3D",),
        block_out_channels: Tuple[int, ...] = (64,),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        norm_type: str = "group",  # group, spatial
        mid_block_add_attention=True,
        time_compression_ratio: int = 4,
        spatial_compression_ratio: int = 8,
        disable_causal: bool = False,
        mid_block_causal_attn: bool = True,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = CausalConv3d(in_channels, block_out_channels[-1], kernel_size=3, stride=1, disable_causal=disable_causal)
        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        temb_channels = in_channels if norm_type == "spatial" else None

        # mid
        self.mid_block = UNetMidBlockCausal3D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            output_scale_factor=1,
            resnet_time_scale_shift="default" if norm_type == "group" else norm_type,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            temb_channels=temb_channels,
            add_attention=mid_block_add_attention,
            disable_causal=disable_causal,
            causal_attention=mid_block_causal_attn,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            num_spatial_upsample_layers = int(np.log2(spatial_compression_ratio))
            num_time_upsample_layers = int(np.log2(time_compression_ratio))

            if time_compression_ratio == 4:
                add_spatial_upsample = bool(i < num_spatial_upsample_layers) 
                add_time_upsample = bool(i >= len(block_out_channels) - 1 - num_time_upsample_layers and not is_final_block)
            elif time_compression_ratio == 8:
                add_spatial_upsample = bool(i >= len(block_out_channels) - num_spatial_upsample_layers)
                add_time_upsample = bool(i >= len(block_out_channels) - num_time_upsample_layers)
            else:
                raise ValueError(f"Unsupported time_compression_ratio: {time_compression_ratio}")

            upsample_scale_factor_HW = (2, 2) if add_spatial_upsample else (1, 1)
            upsample_scale_factor_T = (2, ) if add_time_upsample else (1, )
            upsample_scale_factor = tuple(upsample_scale_factor_T + upsample_scale_factor_HW)
            up_block = get_up_block3d(
                up_block_type,
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                prev_output_channel=None,
                add_upsample=bool(add_spatial_upsample or add_time_upsample),
                upsample_scale_factor=upsample_scale_factor,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                attention_head_dim=output_channel,
                temb_channels=temb_channels,
                resnet_time_scale_shift=norm_type,
                disable_causal=disable_causal,
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        if norm_type == "spatial":
            self.conv_norm_out = SpatialNorm(block_out_channels[0], temb_channels)
        else:
            self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = CausalConv3d(block_out_channels[0], out_channels, kernel_size=3, disable_causal=disable_causal)

        self.gradient_checkpointing = False

    def _stream_mid_block(self, x):
        """The mid block with its attention computed by chunks of query frames: the group statistics of the whole clip first, then the keys
        and values of all the frames and each chunk's queries from chunks normalized with them, each chunk's output added to the residual
        in place. Same values as the module without its normalized copy, query and output tensors. With upstream's frame-causal mask, each
        frame's queries attend to that frame and all the earlier ones (restricting them to windows of frames degrades moving clips).
        16-bit attention runs by blocks of FP32 scores (cuBLAS speed for the 512-wide head, SDPA's rounding points), one frame of queries
        at a time."""
        mid = self.mid_block
        chunk_frames = max(1, (1 << 25) // x[0, :, 0].numel())  # whole-clip resnets by chunks of ~32M elements
        x = mid.resnets[0](x, chunk_frames=chunk_frames)
        for attn, resnet in zip(mid.attentions, mid.resnets[1:]):
            batch, channels, frames, height, width = x.shape
            stats = x.new_zeros(batch * attn.group_norm.num_groups, 3, dtype=torch.float64)
            vae_ops.causal_group_norm_silu_pad(x, attn.group_norm, silu=False, running=stats, apply=False)
            tokens = height * width
            blocks = x.dtype != torch.float32 and fused_norm_silu(x) and _fused.fp32_scores_supported()
            step = 1 if mid.causal_attention or blocks else max(1, (1 << 17) // tokens)  # SDPA: frames of queries per call, few calls

            def normalized(start, stop):
                part = vae_ops.causal_group_norm_silu_pad(x[:, :, start:stop], attn.group_norm, silu=False, running=stats, frozen=True)
                return part.flatten(2).transpose(1, 2)

            key, value = x.new_empty(batch, 1, frames * tokens, channels), x.new_empty(batch, 1, frames * tokens, channels)
            for start in range(0, frames, step):
                part = normalized(start, min(start + step, frames))
                key[:, 0, start * tokens:start * tokens + part.shape[1]] = attn.to_k(part)
                value[:, 0, start * tokens:start * tokens + part.shape[1]] = attn.to_v(part)
            for start in range(0, frames, step):
                stop = min(start + step, frames)
                visible = stop * tokens if mid.causal_attention else frames * tokens
                query = attn.to_q(normalized(start, stop))
                if blocks:
                    out = attn.to_out[0](_fused.attention(query, key[:, 0, :visible], value[:, 0, :visible]))
                else:
                    out = attn.to_out[0](vae_ops.attention(query[:, None], key[:, :, :visible], value[:, :, :visible])[:, 0])
                x[:, :, start:stop].add_(out.transpose(1, 2).unflatten(2, (stop - start, height, width))).div_(attn.rescale_output_factor)
                check_abort()
            del key, value
            x = resnet(x, chunk_frames=chunk_frames)
        return x

    def stream_start(self, sample, global_modules):
        """Global start of a streamed decode (see AutoencoderKLCausal3D.streamed_decode): conv_in, the mid block and the first
        global_modules modules of the up blocks on the whole clip. Returns their output and the modules left to stream."""
        sample = self._stream_mid_block(self.conv_in(sample))
        modules = [module for block in self.up_blocks for module in (*block.resnets, *(block.upsamplers or ()))]
        for module in modules[:global_modules]:
            if isinstance(module, ResnetBlockCausal3D):
                sample = module(sample, chunk_frames=max(1, (1 << 25) // sample[0, :, 0].numel()))
            else:
                sample = module(sample)
        return sample, modules[global_modules:]

    def stream_statistics(self, sample, modules, boxes, chunk_frames, group):
        """Whole-clip estimate of the streamed GroupNorms' statistics for stream_tiles: equal boxes (latent row and column slices spread over
        the frame) stream through all the frames chunk by chunk, `group` boxes at a time in the batch dimension with their causal caches,
        one group after the other through the whole clip, normalizing with running statistics shared by all of them; returns these
        statistics at the end, per batch item of sample. A first group of boxes spread over the frame goes through the clip before the
        others, so that every group is normalized with statistics of the whole clip; what it added to the statistics, computed while
        they were still forming, is removed at the end. The memory is that of one group whatever the frame size (all the groups
        advancing together chunk by chunk held one set of caches per box: 2.3 GiB at 1280x720, 5.1 at 1920x1088)."""
        norms = [norm for module in modules if isinstance(module, ResnetBlockCausal3D) for norm in (module.norm1, module.norm2)] + [self.conv_norm_out]
        device = self.conv_out.conv.weight.device
        stats, weights = {}, {}
        for item in range(sample.shape[0]):
            running = {norm: torch.zeros(norm.num_groups, 3, dtype=torch.float64, device=device) for norm in norms}
            first = None
            for part in [boxes[::max(1, len(boxes) // group)][:group]] + [boxes[start:start + group] for start in range(0, len(boxes), group)]:
                parts = torch.cat([transfer.from_ram(sample[item:item + 1, :, :, rows, columns], device) for rows, columns in part])
                state = dict(running, frozen=set(), shared=True, weights=weights)
                for start in range(0, sample.shape[2], chunk_frames):
                    x = parts[:, :, start:start + chunk_frames]
                    for module in modules:
                        x = module(x, state=state)
                        check_abort()
                    vae_ops.causal_group_norm_silu_pad(x, self.conv_norm_out, running=running[self.conv_norm_out], shared=True, apply=False)
                del parts, state, x
                first = first or {norm: value.clone() for norm, value in running.items()}
            for norm, value in running.items():
                value = _unmerged(value, first[norm])
                stats[norm] = torch.cat([stats[norm], value]) if norm in stats else value
        return stats

    def stream_shape(self, sample, modules):
        """(frames, height, width) of the video stream_tiles decodes from sample."""
        spatial = 2 ** sum(isinstance(module, UpsampleCausal3D) for module in modules)
        temporal = 2 ** sum(isinstance(module, UpsampleCausal3D) and module.upsample_factor[0] == 2 for module in modules)
        return 1 + (sample.shape[2] - 1) * temporal, sample.shape[3] * spatial, sample.shape[4] * spatial

    def stream_tiles(self, sample, modules, stats, tile, chunk_frames, sink=None):
        """Tile-major streamed decode with frozen GroupNorm statistics (stats: norm -> FP64 (count, mean, M2) per group): each output tile
        streams through all the frames alone, holding only its own causal caches; every module runs on the tile's region plus the receptive
        field still ahead of it, and its output is cropped to what the later modules need. Exact for the given statistics, so the tiles join
        without overlap or blending, and the memory follows the tile size instead of the frame size. tile: tile side at sample resolution.
        sink(first frame, rows, columns, pixels): receives each tile's chunk of frames as it is decoded instead of the returned video (no
        decoded clip in VRAM)."""
        batch, _, frames, height, width = sample.shape
        device = self.conv_out.conv.weight.device
        spatial = 2 ** sum(isinstance(module, UpsampleCausal3D) for module in modules)
        temporal = 2 ** sum(isinstance(module, UpsampleCausal3D) and module.upsample_factor[0] == 2 for module in modules)
        out = None if sink is not None else torch.empty(batch, self.conv_out.conv.out_channels, 1 + (frames - 1) * temporal, height * spatial, width * spatial,
                                                        dtype=sample.dtype, device=device)
        weights = {}
        for y0 in range(0, height, tile):
            (ya, yb), y_crops = _stream_plan(modules, y0 * spatial, min(y0 + tile, height) * spatial, height)
            for x0 in range(0, width, tile):
                (xa, xb), x_crops = _stream_plan(modules, x0 * spatial, min(x0 + tile, width) * spatial, width)
                crops = [(slice(*rows), slice(*columns)) for rows, columns in zip(y_crops, x_crops)]
                state = dict(stats, frozen=set(stats), weights=weights)
                region = transfer.from_ram(sample[:, :, :, ya:yb, xa:xb], device)  # (gathered on this thread alone)
                for start in range(0, frames, chunk_frames):
                    x = region[:, :, start:start + chunk_frames]
                    for module, (rows, columns) in zip(modules, crops):
                        x = module(x, state=state)[..., rows, columns]
                        check_abort()
                    rows, columns = crops[-1]
                    x = self.conv_out.stream_norm_silu(x, self.conv_norm_out, state)[..., rows, columns]
                    first_frame = 0 if start == 0 else 1 + (start - 1) * temporal
                    rows, columns = slice(y0 * spatial, y0 * spatial + x.shape[3]), slice(x0 * spatial, x0 * spatial + x.shape[4])
                    if sink is None:
                        out[:, :, first_frame:first_frame + x.shape[2], rows, columns] = x
                    else:
                        sink(first_frame, rows, columns, x)
        return out

    def forward(
        self,
        sample: torch.FloatTensor,
        latent_embeds: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        r"""The forward method of the `DecoderCausal3D` class."""
        assert len(sample.shape) == 5, "The input tensor should have 5 dimensions"
        
        sample = self.conv_in(sample)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype
        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            if is_torch_version(">=", "1.11.0"):
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block),
                    sample,
                    latent_embeds,
                    use_reentrant=False,
                )
                sample = sample.to(upscale_dtype)

                # up
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(up_block),
                        sample,
                        latent_embeds,
                        use_reentrant=False,
                    )
            else:
                # middle
                sample = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(self.mid_block), sample, latent_embeds
                )
                sample = sample.to(upscale_dtype)

                # up
                for up_block in self.up_blocks:
                    sample = torch.utils.checkpoint.checkpoint(create_custom_forward(up_block), sample, latent_embeds)
        else:
            # middle
            sample = self.mid_block(sample, latent_embeds)
            sample = sample.to(upscale_dtype)

            # up
            for up_block in self.up_blocks:
                sample = up_block(sample, latent_embeds)

        # post-process
        if latent_embeds is None and fused_norm_silu(sample):
            return self.conv_out.norm_silu_forward(sample, self.conv_norm_out)
        if latent_embeds is None:
            sample = self.conv_norm_out(sample)
        else:
            sample = self.conv_norm_out(sample, latent_embeds)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)   

        return sample


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        if parameters.ndim == 3:
            dim = 2 # (B, L, C)
        elif parameters.ndim == 5 or parameters.ndim == 4:
            dim = 1 # (B, C, T, H ,W) / (B, C, H, W)
        else:
            raise NotImplementedError
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=dim)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.FloatTensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = randn_tensor(
            self.mean.shape,
            generator=generator,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        x = self.mean + self.std * sample
        return x

    def kl(self, other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            reduce_dim = list(range(1, self.mean.ndim))
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=reduce_dim,
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=reduce_dim,
                )

    def nll(self, sample: torch.Tensor, dims: Tuple[int, ...] = [1, 2, 3]) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> torch.Tensor:
        return self.mean
