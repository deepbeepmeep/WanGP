# TRELLIS.2 sparse UNet VAE encoders (shape: flexible dual grid, texture: PBR attributes), inference port of microsoft/TRELLIS.2
# (trellis2/models/sc_vaes/sparse_unet_vae.py SparseUnetVaeEncoder, sc_vaes/fdg_vae.py FlexiDualGridVaeEncoder, MIT License, see
# LICENSE_TRELLIS2 in this folder) on the sparse operators of WanGP's TRELLIS.2 port (models/trellis2).
# Precision follows upstream: the blocks run in FP16 (the checkpoint dtype) with FP32 layer norms; the input layer, the final norm and
# the latent projection compute in FP32. Each downsampling level records which children of its parent voxels are active: the texture
# decoder is driven by these flags instead of its own predictions, so that it decodes exactly the encoded voxels (upstream reuses the
# encoder's spatial maps through the sparse tensor caches).
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.kernels import vae_ops
from shared.utils.phase_progress import check_abort
from models.trellis2.sparse import SparseConv3d, SparseTensor, layer_norm_rows_, row_chunks, spatial2channel, submanifold_conv3d, subdivide
from models.trellis2.vae import ConvNeXtBlock


class S2CBlock(nn.Module):
    """SparseResBlockS2C3d: downsampling residual block packing the 8 children of each parent voxel into its channels; the skip
    connection averages the packed input channels by groups."""

    def __init__(self, channels, out_channels):
        super().__init__()
        self.channels, self.out_channels = channels, out_channels
        self.norm1 = nn.LayerNorm(channels, eps=1e-6)
        self.conv1 = SparseConv3d(channels, out_channels // 8)
        self.conv2 = SparseConv3d(out_channels, out_channels)

    def forward(self, x_list):
        """x handed off in x_list. Returns (parents SparseTensor, (parent coords, children flags (P, 8) bool))."""
        x = x_list.pop()
        h = x.replace(layer_norm_rows_(x.feats.clone(), self.norm1.weight, self.norm1.bias, 1e-6, F.silu))
        h = h.replace(submanifold_conv3d(h, self.conv1.weight, self.conv1.bias))
        h, parent, child = spatial2channel(h)
        group = self.channels * 8 // self.out_channels
        means = x.feats.view(len(x), self.channels // group, group).mean(-1)  # mean of each group of packed channels, by child
        del x
        width = means.shape[1]
        skip = means.new_zeros((len(h), 8, width))
        skip[parent, child] = means
        del means
        flags = torch.zeros((len(h), 8), dtype=torch.bool, device=parent.device)
        flags[parent, child] = True
        layer_norm_rows_(h.feats, eps=1e-6, activation=F.silu)
        out = vae_ops.sparse_conv(h.feats, h, self.conv2.weight, self.conv2.bias, residual=skip.view(len(h), -1))
        return h.replace(out), (h.coords, flags)


class SparseUnetEncoder(nn.Module):
    """SparseUnetVaeEncoder: voxels downsampled 16x through four space-to-channel levels to the latent mean (no posterior sampling)."""

    def __init__(self, in_channels, model_channels, latent_channels, num_blocks, block_type=None, down_block_type=None, block_args=None,
                 use_fp16=True):
        super().__init__()
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.latent_channels = latent_channels
        self.input_layer = nn.Linear(in_channels, model_channels[0])
        self.to_latent = nn.Linear(model_channels[-1], 2 * latent_channels)
        self.blocks = nn.ModuleList()
        for i, (channels, count) in enumerate(zip(model_channels, num_blocks)):
            level = [ConvNeXtBlock(channels) for _ in range(count)]
            if i < len(model_channels) - 1:
                level.append(S2CBlock(channels, model_channels[i + 1]))
            self.blocks.append(nn.ModuleList(level))

    def block_count(self):
        return sum(len(level) for level in self.blocks)

    def forward(self, x_list, progress=None):
        """x: SparseTensor of FP32 input features (handed off in x_list). Returns (latent mean SparseTensor FP32, subdivision levels):
        the (parent coords, children flags) of each downsampling, finest first."""
        x = x_list.pop()
        feats = torch.empty((len(x), self.input_layer.out_features), dtype=self.dtype, device=x.feats.device)
        with torch.autocast(device_type=feats.device.type, dtype=torch.float32):
            for start, stop in row_chunks(len(x), x.feats.shape[1] + feats.shape[1]):
                feats[start:stop] = self.input_layer(x.feats[start:stop].float())
        h = x.replace(feats)
        del x, feats
        levels = []
        for level in self.blocks:
            if len(level) > 1:
                h.keep_neighbors()  # the convolutions of a level share its coords
            for block in level:
                if isinstance(block, S2CBlock):
                    h_list, h = [h], None
                    h, flags = block(h_list)
                    levels.append(flags)
                else:
                    h = block(h)
                if progress is not None:
                    progress()
                check_abort()
        out = torch.empty((len(h), self.latent_channels), dtype=torch.float32, device=h.feats.device)
        with torch.autocast(device_type=out.device.type, dtype=torch.float32):
            for start, stop in row_chunks(len(h), h.feats.shape[1]):
                part = h.feats[start:stop].float()
                out[start:stop] = self.to_latent(F.layer_norm(part, part.shape[-1:]))[:, :self.latent_channels]
        return SparseTensor(out, h.coords), levels


def shape_input(voxels, offsets, intersected):
    """FlexiDualGridVaeEncoder input: dual vertex offsets and intersected flags, centered."""
    return SparseTensor(torch.cat([offsets.float() - 0.5, intersected.float() - 0.5], dim=1), voxels.int())


def decoder_subdivisions(levels, coords):
    """Children flags of the decoder levels (coarsest first), aligned with the rows the decoder produces from the latent coords: the
    decoder subdivides with these flags, so it rebuilds the encoded voxels."""
    flags = []
    for parent_coords, children in reversed(levels):
        found = SparseTensor(None, parent_coords).find(coords)
        if bool((found < 0).any()):
            raise RuntimeError("decoder voxels missing from the encoder hierarchy")
        flags.append(children[found.long()])
        coords = subdivide(coords, flags[-1])[0]
    return flags
