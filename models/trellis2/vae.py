# TRELLIS.2 decoders, inference port of microsoft/TRELLIS.2 (trellis2/models/sc_vaes/sparse_unet_vae.py, sc_vaes/fdg_vae.py and
# trellis2/models/sparse_structure_vae.py, MIT License, see LICENSE in this folder).
# Precision follows upstream: the blocks run in FP16 (the checkpoint dtype) with FP32 layer norms; the latent projection, the final norm
# and the output layer compute in FP32 (upstream keeps them FP32 and loads the FP16 checkpoint into them). The sparse decoders work by
# chunks of voxels: each block writes a new tensor row chunk by row chunk, and the channel-to-space upsampling keeps only the flagged
# children of each convolution chunk instead of materializing the 8x wider convolution output; its second convolution adds the skip
# rows as it writes its output.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.utils.phase_progress import check_abort
from shared.kernels import vae_ops
from .sparse import SparseConv3d, SparseTensor, conv_row_ranges, conv_rows, layer_norm_rows_, row_chunks, subdivide


def _layer_norm32(x, norm, eps=1e-6):
    return F.layer_norm(x.float(), x.shape[-1:], norm.weight.float(), norm.bias.float(), eps).to(x.dtype)


class ConvNeXtBlock(nn.Module):
    """SparseConvNeXtBlock3d: x + mlp(norm(conv(x)))."""

    def __init__(self, channels, mlp_ratio=4.0):
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-6)
        self.conv = SparseConv3d(channels, channels)
        self.mlp = nn.Sequential(nn.Linear(channels, int(channels * mlp_ratio)), nn.SiLU(), nn.Linear(int(channels * mlp_ratio), channels))

    def forward(self, x):
        out = torch.empty_like(x.feats)
        for start, stop in conv_row_ranges(x, x.feats.shape[1]):
            h = _layer_norm32(conv_rows(x, self.conv.weight, self.conv.bias, start, stop), self.norm)
            out[start:stop] = self.mlp(h).add_(x.feats[start:stop])
        return x.replace(out)


class C2SBlock(nn.Module):
    """SparseResBlockC2S3d: upsampling residual block whose convolution output holds the features of the 8 children of each voxel;
    the children kept are given by subdivision flags (predicted from the input when pred_subdiv)."""

    def __init__(self, channels, out_channels, pred_subdiv=True):
        super().__init__()
        self.channels, self.out_channels, self.pred_subdiv = channels, out_channels, pred_subdiv
        self.norm1 = nn.LayerNorm(channels, eps=1e-6)
        self.conv1 = SparseConv3d(channels, out_channels * 8)
        self.conv2 = SparseConv3d(out_channels, out_channels)
        if pred_subdiv:
            self.to_subdiv = nn.Linear(channels, 8)

    def forward(self, x_list, subdivision=None, coords_only=False):
        """x handed off in x_list. Returns (children SparseTensor, subdivision (N, 8) bool), or (children coords, subdivision)."""
        x = x_list.pop()
        if self.pred_subdiv:
            subdivision = self.to_subdiv(x.feats) > 0
        child_coords, _, slot = subdivide(x.coords, subdivision)
        if coords_only:
            return child_coords, subdivision
        channels, out_channels = self.channels, self.out_channels
        h = x.replace(layer_norm_rows_(x.feats.clone(), self.norm1.weight, self.norm1.bias, 1e-6, F.silu))
        ranges = conv_row_ranges(h, channels)
        bounds = torch.tensor([0] + [stop for _, stop in ranges], device=slot.device)
        child_bounds = torch.cat([subdivision.new_zeros(1, dtype=torch.int64), subdivision.sum(1).cumsum(0)])[bounds].tolist()
        up = x.feats.new_empty((slot.shape[0], out_channels))
        for (start, stop), first, last in zip(ranges, child_bounds[:-1], child_bounds[1:]):
            rows = conv_rows(h, self.conv1.weight, self.conv1.bias, start, stop).view(-1, out_channels)
            up[first:last] = rows.index_select(0, slot[first:last] - start * 8)
            del rows
        del h
        skip = x.feats.reshape(len(x) * 8, channels // 8).index_select(0, slot)
        del x, slot
        y = SparseTensor(layer_norm_rows_(up, eps=1e-6, activation=F.silu), child_coords)
        repeat = out_channels // (channels // 8)
        out = vae_ops.sparse_conv(y.feats, y, self.conv2.weight, self.conv2.bias, residual=lambda start, stop: skip[start:stop].repeat_interleave(repeat, dim=1))
        return y.replace(out), subdivision


class SparseUnetDecoder(nn.Module):
    """SparseUnetVaeDecoder: latent voxels upsampled 16x through four channel-to-space levels."""

    def __init__(self, out_channels, model_channels, latent_channels, num_blocks, block_type=None, up_block_type=None, block_args=None,
                 use_fp16=True, pred_subdiv=True, resolution=None, voxel_margin=0.5):
        super().__init__()
        self.out_channels, self.resolution, self.voxel_margin = out_channels, resolution, voxel_margin
        self.dtype = torch.float16 if use_fp16 else torch.float32  # dtype of the blocks (checkpoint FP16)
        self.from_latent = nn.Linear(latent_channels, model_channels[0])
        self.output_layer = nn.Linear(model_channels[-1], out_channels)
        self.blocks = nn.ModuleList()
        for i, (channels, count) in enumerate(zip(model_channels, num_blocks)):
            level = [ConvNeXtBlock(channels) for _ in range(count)]
            if i < len(model_channels) - 1:
                level.append(C2SBlock(channels, model_channels[i + 1], pred_subdiv))
            self.blocks.append(nn.ModuleList(level))

    def forward(self, x_list, subdivisions=None, coords_only=False, progress=None):
        """x: latent SparseTensor (FP32), handed off in x_list. subdivisions: the flags of each upsampling level (texture decoder).
        Returns (output SparseTensor FP32, subdivisions), or with coords_only the coords of the last level (cascade upsampling: the
        last upsampling block only predicts its subdivision)."""
        x = x_list.pop()
        with torch.autocast(device_type=x.feats.device.type, dtype=torch.float32):
            h = SparseTensor(self.from_latent(x.feats).to(self.dtype), x.coords)
        del x
        flags = []
        last_level = len(self.blocks) - 2
        for i, level in enumerate(self.blocks):
            if len(level) > 1:
                h.keep_neighbors()  # the convolutions of a level share its coords
            for block in level:
                if isinstance(block, C2SBlock):
                    h_list, h = [h], None
                    h, subdivision = block(h_list, None if subdivisions is None else subdivisions[i], coords_only and i == last_level)
                    flags.append(subdivision)
                    if coords_only and i == last_level:
                        return h
                else:
                    h = block(h)
                if progress is not None:
                    progress()
                check_abort()
        out = h.feats.new_empty((len(h), self.out_channels), dtype=torch.float32)
        with torch.autocast(device_type=out.device.type, dtype=torch.float32):
            for start, stop in row_chunks(len(h), h.feats.shape[1]):
                part = h.feats[start:stop].float()
                out[start:stop] = self.output_layer(F.layer_norm(part, part.shape[-1:]))
        return SparseTensor(out, h.coords), flags

    def block_count(self, coords_only=False):
        return sum(len(level) for level in self.blocks) - (1 if coords_only else 0)


# Dense sparse-structure decoder (TRELLIS ss_dec_conv3d_16l8_fp16: FP32 input/output layers and norms, FP16 block convolutions)

class ChannelLayerNorm32(nn.LayerNorm):
    def forward(self, x):
        dims = x.dim()
        h = F.layer_norm(x.permute(0, *range(2, dims), 1).contiguous().float(), self.normalized_shape, self.weight, self.bias, self.eps).to(x.dtype)
        return h.permute(0, dims - 1, *range(1, dims - 1)).contiguous()


class ResBlock3d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = ChannelLayerNorm32(channels)
        self.norm2 = ChannelLayerNorm32(channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv2(F.silu(self.norm2(self.conv1(F.silu(self.norm1(x)))))) + x


def pixel_shuffle_3d(x, scale_factor):
    batch, channels, height, width, depth = x.shape
    x = x.reshape(batch, channels // scale_factor ** 3, scale_factor, scale_factor, scale_factor, height, width, depth)
    return x.permute(0, 1, 5, 2, 6, 3, 7, 4).reshape(batch, channels // scale_factor ** 3, height * scale_factor, width * scale_factor, depth * scale_factor)


class UpsampleBlock3d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels * 8, 3, padding=1)

    def forward(self, x):
        return pixel_shuffle_3d(self.conv(x), 2)


class SparseStructureDecoder(nn.Module):
    """Occupancy decoder of the 16^3 x 8 structure latent to a 64^3 grid (TRELLIS SparseStructureDecoder, layer norms)."""

    def __init__(self, out_channels=1, latent_channels=8, num_res_blocks=2, channels=(512, 128, 32), num_res_blocks_middle=2, use_fp16=True, **kwargs):
        super().__init__()
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.input_layer = nn.Conv3d(latent_channels, channels[0], 3, padding=1)
        self.middle_block = nn.Sequential(*[ResBlock3d(channels[0]) for _ in range(num_res_blocks_middle)])
        blocks = []
        for i, ch in enumerate(channels):
            blocks += [ResBlock3d(ch) for _ in range(num_res_blocks)]
            if i < len(channels) - 1:
                blocks.append(UpsampleBlock3d(ch, channels[i + 1]))
        self.blocks = nn.ModuleList(blocks)
        self.out_layer = nn.Sequential(ChannelLayerNorm32(channels[-1]), nn.SiLU(), nn.Conv3d(channels[-1], out_channels, 3, padding=1))

    def lock_checkpoint_fp32_modules(self):
        """The checkpoint stores the input/output layers and the norms in FP32, the block convolutions in FP16."""
        for module in [self.input_layer, *self.out_layer] + [m for m in self.modules() if isinstance(m, ChannelLayerNorm32)]:
            module._lock_dtype = torch.float32

    def forward(self, x):
        h = self.input_layer(x).to(self.dtype)
        h = self.middle_block(h)
        for block in self.blocks:
            h = block(h)
            check_abort()
        return self.out_layer(h.float())
