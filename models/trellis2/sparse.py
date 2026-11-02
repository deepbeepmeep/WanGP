# Sparse voxel tensors and the sparse operators of TRELLIS.2, without compiled extensions: the convolutions run through
# shared.kernels.vae_ops.sparse_conv (the WanGP Triton implicit GEMM, else gathered rows and addmm), the rest in PyTorch.
# Semantics ported from microsoft/TRELLIS.2 (trellis2/modules/sparse, MIT License, see LICENSE in this folder) and from the FlexGEMM
# submanifold convolution it uses (github.com/JeffreyXiang/FlexGEMM, MIT License): the neighbor map follows FlexGEMM's kernel order
# v = (dx + 1) * 9 + (dy + 1) * 3 + (dz + 1) over the coordinates (x, y, z), and weights keep the checkpoint layout (Co, 3, 3, 3, Ci).
# Neighbors are found by sorting the voxel keys and binary searching them (torch.searchsorted) instead of a CUDA hash map.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.kernels import vae_ops

SCRATCH_BYTES = 256 << 20  # bound of the gathered (im2col) neighbor features of one chunk of rows
ROW_ELEMENTS = 1 << 24  # bound of the FP32 scratch of the row-wise passes (norms, activations)
_KERNEL_OFFSETS = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]


def row_chunks(rows, row_elements, budget=ROW_ELEMENTS):
    step = max(1, budget // max(1, row_elements))
    return [(start, min(start + step, rows)) for start in range(0, rows, step)]


class SparseTensor:
    """Features of the active voxels of one sample: feats (N, C) and coords (N, 3) int32 voxel indices (x, y, z). Tensors made from
    the same coords with replace() share the lookup structures cached for them (sorted keys, neighbor maps)."""

    def __init__(self, feats, coords, cache=None):
        self.feats, self.coords = feats, coords
        self.cache = {} if cache is None else cache

    def replace(self, feats):
        return SparseTensor(feats, self.coords, self.cache)

    def __len__(self):
        return self.coords.shape[0]

    def _keys(self, coords, span, low):
        coords = coords.long()
        return ((coords[..., 0] - low) * span + coords[..., 1] - low) * span + coords[..., 2] - low

    def lookup(self):
        """(span, low, sorted keys, voxel index of each sorted key) of these coords; a key encodes coords - low in base span, low being
        one below the smallest coordinate (and at most -1), so the -1 and max + 1 neighbors of every voxel have keys of their own."""
        if "lookup" not in self.cache:
            low = min(int(self.coords.min()), 0) - 1
            span = int(self.coords.max()) - low + 2
            keys, order = torch.sort(self._keys(self.coords, span, low))
            self.cache["lookup"] = (span, low, keys, order.to(torch.int32))
        return self.cache["lookup"]

    def find(self, query):
        """Index of the voxel at each query coordinate (..., 3), -1 where there is none."""
        span, low, keys, order = self.lookup()
        query = query.long()
        inside = ((query >= low) & (query < low + span)).all(-1)
        wanted = self._keys(query.clamp(low, low + span - 1), span, low)
        position = torch.searchsorted(keys, wanted.reshape(-1)).clamp_(max=keys.shape[0] - 1).view(wanted.shape)
        return torch.where(inside & (keys[position] == wanted), order[position], -1)

    def _neighbor_rows(self, start, stop):
        span, low, keys, order = self.lookup()
        deltas = torch.tensor([(dx * span + dy) * span + dz for dx, dy, dz in _KERNEL_OFFSETS], dtype=torch.int64, device=keys.device)
        wanted = self._keys(self.coords[start:stop], span, low)[:, None] + deltas
        position = torch.searchsorted(keys, wanted.view(-1)).clamp_(max=keys.shape[0] - 1).view(wanted.shape)
        return torch.where(keys[position] == wanted, order[position], -1)

    def neighbors(self, start, stop):
        """(stop - start, 27) int32 indices of the 3x3x3 neighbors of rows [start, stop), -1 for inactive sites."""
        full = self.cache.get("neighbors")
        return self._neighbor_rows(start, stop) if full is None else full[start:stop]

    def keep_neighbors(self):
        """Builds and caches the whole neighbor map (27 int32 per voxel), worth it when several convolutions share these coords."""
        if "neighbors" not in self.cache:
            rows = len(self)
            full = torch.empty((rows, 27), dtype=torch.int32, device=self.coords.device)
            for start, stop in row_chunks(rows, 27 * 4, SCRATCH_BYTES // 8):
                full[start:stop] = self._neighbor_rows(start, stop)
            self.cache["neighbors"] = full
        return self


def conv_rows(x, weight, bias, start, stop):
    """Submanifold 3x3x3 convolution of x for the output rows [start, stop): weight (Co, 3, 3, 3, Ci) as stored by FlexGEMM, FP32
    accumulation over all the taps, rounded once to the activation dtype as FlexGEMM does."""
    return vae_ops.sparse_conv(x.feats, x, weight, bias, rows=(start, stop))


def conv_row_ranges(x, in_channels):
    return row_chunks(len(x), 27 * in_channels * x.feats.element_size(), SCRATCH_BYTES)


def submanifold_conv3d(x, weight, bias):
    """Whole submanifold convolution of x: a new (N, Co) tensor."""
    return vae_ops.sparse_conv(x.feats, x, weight, bias)


class SparseConv3d(nn.Module):
    """Submanifold 3x3x3 sparse convolution, weight (Co, 3, 3, 3, Ci) as in the FlexGEMM checkpoints."""

    def __init__(self, in_channels, out_channels, bias=True):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.weight = nn.Parameter(torch.empty(out_channels, 3, 3, 3, in_channels))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None

    def forward(self, x):
        return x.replace(submanifold_conv3d(x, self.weight, self.bias))


def subdivide(coords, subdivision):
    """Children of the voxels coords (N, 3) whose 8 children are flagged by subdivision (N, 8) bool, in upstream order (parents in
    order, children by increasing index; child bit i offsets axis i): (child coords (M, 3), parent index (M,), child slot (M,) =
    parent * 8 + child index)."""
    parent, child = subdivision.nonzero(as_tuple=True)
    child_coords = coords[parent] * 2 + torch.stack([(child >> axis) & 1 for axis in range(3)], dim=1).to(coords.dtype)
    return child_coords, parent, parent * 8 + child


def channel2spatial(x, subdivision):
    """Upstream SparseChannel2Spatial(2): the channel group g of a voxel becomes the features of its child g, for the flagged children.
    Returns (children SparseTensor, parent index, child slot)."""
    child_coords, parent, slot = subdivide(x.coords, subdivision)
    feats = x.feats.reshape(len(x) * 8, -1).index_select(0, slot)
    return SparseTensor(feats, child_coords), parent, slot


def spatial2channel(x):
    """Upstream SparseSpatial2Channel(2): the features of the children of a parent voxel are packed as 8 channel groups, missing children
    are zeros. Parents are sorted by (x, y, z) as upstream's unique codes. Returns (parents SparseTensor, parent index, child index)."""
    coords = x.coords.long()
    parents = coords // 2
    child = ((coords % 2) * torch.tensor([1, 2, 4], device=coords.device)).sum(-1)
    span = int(parents.max()) + 1
    codes = (parents[:, 0] * span + parents[:, 1]) * span + parents[:, 2]
    codes, parent = torch.unique(codes, return_inverse=True)
    parent_coords = torch.stack([codes // (span * span), codes // span % span, codes % span], dim=1).to(x.coords.dtype)
    feats = x.feats.new_zeros((codes.shape[0] * 8, x.feats.shape[1]))
    feats[parent * 8 + child] = x.feats
    return SparseTensor(feats.view(codes.shape[0], -1), parent_coords), parent, child


def layer_norm_rows_(feats, weight=None, bias=None, eps=1e-6, activation=None):
    """Upstream LayerNorm32 (FP32 statistics and affine, rounded back to the activation dtype) in place by chunks of rows, optionally
    followed by an activation applied to the rounded values (separate PyTorch ops in upstream)."""
    weight = None if weight is None else weight.float()
    bias = None if bias is None else bias.float()
    for start, stop in row_chunks(feats.shape[0], feats.shape[1]):
        part = feats[start:stop]
        normed = F.layer_norm(part.float(), part.shape[-1:], weight, bias, eps).to(part.dtype)
        part.copy_(normed if activation is None else activation(normed))
    return feats


def sparse_trilinear(feats, x, points):
    """Trilinear sampling of the sparse voxel features at points (P, 3) in voxel units (voxel i spans [i, i + 1]): FlexGEMM's
    grid_sample_3d semantics (the 8 voxels whose centers surround a point, weights renormalized over the active ones). Returns
    (P, C) FP32 values and (P,) FP32 weight sums (0 where no surrounding voxel is active)."""
    corners = torch.tensor([[a, b, c] for a in (-0.5, 0.5) for b in (-0.5, 0.5) for c in (-0.5, 0.5)], device=points.device)
    neighbor = (points[:, None] + corners).to(torch.int32)  # truncation toward zero, as FlexGEMM's .int()
    index = x.find(neighbor)
    weight = (1 - (neighbor.float() + 0.5 - points[:, None]).abs()).prod(-1)
    weight = torch.where(index >= 0, weight, 0)
    values = feats.float().index_select(0, index.clamp(min=0).view(-1)).view(*index.shape, -1)
    total = weight.sum(1)
    values = (values * weight.unsqueeze(-1)).sum(1) / total.clamp_min(1e-12).unsqueeze(-1)
    return values, total
