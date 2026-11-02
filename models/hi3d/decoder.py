# Hi3D v1.0 sparse mesh decoder, inference port of hi3d-official/Hi3Dv1.0 (sparc3d/models/autoencoders/{ptv3,blocks,sparconv_vae}.py,
# sparc3d/structures/point.py and sparc3d/utils/serialization: Hi3D Open License Agreement 1.0, with parts adapted from Pointcept /
# Point Transformer V3 and O-CNN under the MIT License; see LICENSE, NOTICE and THIRD_PARTY_LICENSES.md in this folder).
# The PTv3 decoder maps the 32-channel latents of a pass to per-corner SDF signs, distances and deformations: a serialized windowed
# attention stage on the latent voxels, then three 2x unpooling stages whose occupancy head keeps the children near the surface.
# Modified by DeepBeepMeep for WanGP: 16-bit weights and features, spconv replaced by the submanifold convolution of
# shared.kernels.vae_ops (an implicit GEMM reading the neighbors in place), flash-attn by the shared attention, and a bounded working
# set for the 8x larger last stages:
# - the group norms take their statistics in FP64 (deterministic) and are applied with their Swish as a convolution reads its
#   neighbors, so no normalized copy of the features exists;
# - the residual of a block is added into its input by the convolution, and the unpooling projects and keeps the children by chunks;
# so a residual block of the last stage holds its input and one intermediate (two 16-bit tensors) instead of upstream's FP32 copies.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.trellis2.sparse import SparseTensor, row_chunks
from shared.attention import pay_attention
from shared.kernels import vae_ops
from shared.utils.phase_progress import check_abort

PATCH_SIZE = 48  # points per attention window
CHILD_OFFSETS = ((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1))  # upstream inv_grid_coord
KEEP_NEIGHBORS_ROWS = 4_000_000  # below this many voxels a level keeps its whole neighbor map (27 int32 per voxel)
SKIP_POINTS = 12_000_000  # upstream: a residual block over more voxels passes its input through (the released weights' behavior)


def _level(coords):
    level = SparseTensor(None, coords)
    return level.keep_neighbors() if len(level) <= KEEP_NEIGHBORS_ROWS else level


class Conv(nn.Module):
    """Submanifold 3x3x3 convolution, weight (Co, 3, 3, 3, Ci) as spconv's SubMConv3d."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, 3, 3, 3, in_channels))
        self.bias = nn.Parameter(torch.empty(out_channels))

    def forward(self, feats, level, affine=None, residual=None, out=None):
        """The convolution of feats over level; affine: per-channel (scale, shift) FP32 of a group norm, applied with a Swish to the
        gathered neighbors (inactive sites stay zero); residual: added to the result (rows or a callable of a row range)."""
        return vae_ops.sparse_conv(feats, level, self.weight, self.bias, affine=affine, residual=residual, out=out)


class GroupNorm(nn.Module):
    """PointGroupNorm (32 groups of at least 4 channels, eps 1e-6, statistics over all the voxels in FP32), as a per-channel affine."""

    def __init__(self, channels, group=32, min_group_channels=4, eps=1e-6):
        super().__init__()
        self.groups = min(group, channels // min_group_channels)
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(1, channels))
        self.bias = nn.Parameter(torch.zeros(1, channels))

    def affine(self, feats):
        """(scale, shift) FP32 per channel such that feats * scale + shift is the normalized and scaled feats."""
        groups = self.groups
        sums = vae_ops.channel_moments(feats).view(2, groups, -1).sum(-1)
        count = feats.shape[0] * (feats.shape[1] // groups) + self.eps
        mean = sums[0] / count
        inv_std = torch.rsqrt(sums[1] / count - mean.square() + self.eps)
        per_channel = feats.shape[1] // groups
        scale = (inv_std.repeat_interleave(per_channel).float() * self.weights.float()[0])
        shift = self.bias.float()[0] - mean.repeat_interleave(per_channel).float() * scale
        return scale, shift


class ResnetBlock(nn.Module):
    """SparseResnetBlock: x + conv(swish(norm(conv(swish(norm(x)))))) (res_conv(x) when the width changes)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.ModuleList([GroupNorm(in_channels), nn.SiLU(), Conv(in_channels, out_channels), GroupNorm(out_channels), nn.SiLU(),
                                    Conv(out_channels, out_channels)])
        self.res_conv = nn.Linear(in_channels, out_channels) if in_channels != out_channels else None

    def forward(self, x_list, level):
        x = x_list.pop()
        if x.shape[0] > SKIP_POINTS:
            return x
        h = self.block[2](x, level, self.block[0].affine(x))
        if self.res_conv is None:  # the residual added into x, which only the residual reads
            return self.block[5](h, level, self.block[3].affine(h), residual=x, out=x)
        return self.block[5](h, level, self.block[3].affine(h), residual=lambda start, stop: self.res_conv(x[start:stop]))


def _zorder(coords, depth):
    """Upstream xyz2key Morton code of (x, y, z) (x the most significant of each bit triple)."""
    coords = coords.long()
    key = torch.zeros_like(coords[:, 0])
    for bit in range(depth):
        for axis in range(3):
            key |= ((coords[:, axis] >> bit) & 1) << (3 * bit + 2 - axis)
    return key


class WindowAttention(nn.Module):
    """SerializedAttention: attention within windows of 48 consecutive voxels along a space-filling curve; a last partial window is
    filled with the voxels before it (upstream's padding), whose outputs come from their own window."""

    def __init__(self, channels, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)

    def forward(self, x, order):
        n, channels = x.shape
        qkv = self.qkv(x).index_select(0, order).view(n, 3, self.heads, -1)
        out = torch.empty((n, self.heads, qkv.shape[-1]), dtype=x.dtype, device=x.device)

        def attend(start, stop):
            q, k, v = (qkv[start:stop, i].reshape(-1, min(PATCH_SIZE, n), self.heads, qkv.shape[-1]) for i in range(3))
            return pay_attention([q, k, v], force_attention="sdpa").reshape(stop - start, self.heads, -1)

        full = n // PATCH_SIZE * PATCH_SIZE if n > PATCH_SIZE else n
        out[:full] = attend(0, full)
        if full < n:  # the last window: the last 48 voxels, of which the first ones already have their output
            out[full:] = attend(n - PATCH_SIZE, n)[PATCH_SIZE - (n - full):]
        del qkv
        result = torch.empty_like(out)
        result[order] = out
        return self.proj(result.view(n, channels))


class AttentionBlock(nn.Module):
    """SparseAttentionBlock: conditional position encoding (sparse conv), windowed attention and MLP, pre-norm residuals."""

    def __init__(self, channels, heads, mlp_ratio=4.0):
        super().__init__()
        self.cpe = nn.ModuleList([Conv(channels, channels), nn.Linear(channels, channels), nn.LayerNorm(channels)])
        self.norm1 = nn.ModuleList([nn.LayerNorm(channels)])
        self.attn = WindowAttention(channels, heads)
        self.norm2 = nn.ModuleList([nn.LayerNorm(channels)])
        self.mlp = nn.ModuleList([nn.Module()])
        self.mlp[0].fc1 = nn.Linear(channels, int(channels * mlp_ratio))
        self.mlp[0].fc2 = nn.Linear(int(channels * mlp_ratio), channels)

    @staticmethod
    def _norm(norm, x):
        return F.layer_norm(x.float(), x.shape[-1:], norm.weight.float(), norm.bias.float(), norm.eps).to(x.dtype)

    def forward(self, x, level, order):
        x = x + self._norm(self.cpe[2], self.cpe[1](self.cpe[0](x, level)))
        x = x + self.attn(self._norm(self.norm1[0], x), order)
        mlp = self.mlp[0]
        return x + mlp.fc2(F.gelu(mlp.fc1(self._norm(self.norm2[0], x))))


class Unpool(nn.Module):
    """SerializedUnpooling: each voxel splits into its 8 children; the occupancy head keeps those with a positive logit, which take the
    matching channel group of the projection."""

    def __init__(self, channels):
        super().__init__()
        self.cls_head = nn.Module()
        self.cls_head.fc = nn.ModuleList([GroupNorm(channels), nn.SiLU(), Conv(channels, 8)])
        self.proj = nn.Linear(channels, channels * 8, bias=False)

    def forward(self, x_list, coords, level):
        x = x_list.pop()
        keep = self.cls_head.fc[2](x, level, self.cls_head.fc[0].affine(x)) > 0
        parent, child = keep.nonzero(as_tuple=True)
        offsets = torch.tensor(CHILD_OFFSETS, dtype=coords.dtype, device=coords.device)
        child_coords = coords.index_select(0, parent) * 2 + offsets.index_select(0, child)
        bounds = torch.cat([keep.new_zeros(1, dtype=torch.int64), keep.sum(1).cumsum(0)])
        out = x.new_empty((parent.shape[0], x.shape[1]))
        del parent, child
        for start, stop in row_chunks(x.shape[0], 8 * x.shape[1]):
            first, last = int(bounds[start]), int(bounds[stop])
            out[first:last] = self.proj(x[start:stop]).view(-1, x.shape[1])[keep[start:stop].view(-1)]
        return out, child_coords


class Stage(nn.Module):
    def __init__(self, in_channels, out_channels, depth, upsample):
        super().__init__()
        for i in range(depth):
            self.add_module(f"block{i}", ResnetBlock(in_channels if i == 0 else out_channels, out_channels))
        self.depth = depth
        if upsample:
            self.up = nn.Module()
            self.up.upsample = Unpool(out_channels)


class PTv3MeshDecoder(nn.Module):
    def __init__(self, out_channels=32, depths=(2, 2, 2, 2), channels=(128, 256, 512, 512), mid_depth=2, mid_num_head=32, z_channels=32,
                 patch_size=PATCH_SIZE, mlp_ratio=4, order=("z", "z-trans")):
        super().__init__()
        self.order = order
        mid = channels[-1]
        self.z_to_f = nn.Module()
        self.z_to_f.conv = nn.ModuleList([Conv(z_channels, mid)])
        self.mid = nn.Module()
        self.mid.block_in = ResnetBlock(mid, mid)
        for i in range(mid_depth):
            self.mid.add_module(f"block{i}", AttentionBlock(mid, mid_num_head, mlp_ratio))
        self.mid.block_out = ResnetBlock(mid, mid)
        self.mid_depth = mid_depth
        self.dec = nn.Module()
        for s in reversed(range(len(depths))):
            self.dec.add_module(f"dec{s}", Stage(channels[s], channels[s - 1] if s > 0 else channels[0], depths[s], s != 0))
        self.stages = len(depths)
        self.conv_out = nn.Module()
        self.conv_out.conv = nn.ModuleList([GroupNorm(channels[0]), nn.SiLU(), Conv(channels[0], 8 + out_channels)])

    def _orders(self, coords):
        depth = max(1, int(coords.max()).bit_length())
        codes = [_zorder(coords if name == "z" else coords[:, [1, 0, 2]], depth) for name in self.order]
        return [torch.argsort(code) for code in codes]

    def forward(self, latents, coords, progress=None):
        """latents (voxels, z_channels) FP32 in the VAE latent space, coords (voxels, 3) int32 of the pass's structure grid. Returns
        (attributes (N, 40) 16-bit: 8 SDF sign logits then the per-corner SDF and deformation, coords (N, 3) int32 of the 8x grid)."""
        dtype = self.z_to_f.conv[0].weight.dtype
        level = _level(coords)
        x = self.z_to_f.conv[0](latents.to(dtype), level)
        orders = self._orders(coords)
        x = self.mid.block_in([x], level)
        for i in range(self.mid_depth):
            x = getattr(self.mid, f"block{i}")(x, level, orders[i % len(orders)])
        del orders
        x = self.mid.block_out([x], level)
        check_abort()
        for s in reversed(range(self.stages)):
            stage = getattr(self.dec, f"dec{s}")
            for i in range(stage.depth):
                x = getattr(stage, f"block{i}")([x], level)
                check_abort()
            if s != 0:
                x, coords = stage.up.upsample([x], coords, level)
                del level
                level = _level(coords)
            if progress is not None:
                progress()
        norm, conv = self.conv_out.conv[0], self.conv_out.conv[2]
        out = conv(x, level, norm.affine(x))
        return out, coords


class SparconvVAE(nn.Module):
    def __init__(self, **decoder_cfg):
        super().__init__()
        self.decoder = PTv3MeshDecoder(**decoder_cfg)

    def forward(self, latents, coords, progress=None):
        return self.decoder(latents, coords, progress)
