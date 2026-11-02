# Hi3D v1.0 surface voxelization, port of hi3d-official/Hi3Dv1.0 (sparc3d/ops/voxelize.py and sparc3d/ops/triton/voxelize.py, after
# Zhihao Li's pytorch-voxelization: Hi3D Open License Agreement 1.0; see LICENSE, NOTICE and AUTHORS in this folder): the mesh is
# centred and scaled into the grid with a 3-voxel margin, and every voxel overlapping a triangle (separating-axis test: the box, the
# triangle plane and the nine edge cross products) is active. When the active voxels exceed a pass's budget the grid shrinks by 5 %.
# Modified by DeepBeepMeep for WanGP: the overlap tests run as batched PyTorch operations over the candidate voxels of each triangle's
# bounding box, by bounded chunks, instead of a Triton kernel per triangle; same arithmetic, same voxel order (sorted z, y, x keys).
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from dataclasses import dataclass

import torch

PAD = 3
EPS = 1e-6
MIN_RESOLUTION = 2 * PAD + 1
CANDIDATES_PER_CHUNK = 1 << 23


@dataclass
class GridFrame:
    """Mesh coordinates to voxel space: v_voxel = (v - center) * scale + resolution / 2."""
    resolution: int
    center: torch.Tensor
    scale: torch.Tensor

    def to_voxels(self, vertices):
        return (vertices - self.center) * self.scale + self.resolution / 2.0

    def to_mesh(self, voxels):
        return (voxels - self.resolution / 2.0) / self.scale + self.center


def grid_frame(vertices, resolution):
    low, high = vertices.min(dim=0).values - EPS, vertices.max(dim=0).values + EPS
    return GridFrame(resolution, (low + high) / 2.0, (resolution - PAD * 2 - 2 * EPS) / (high - low).max())


def _axis_overlaps(v0, v1, v2, axis):
    p0, p1, p2 = (v0 * axis).sum(-1), (v1 * axis).sum(-1), (v2 * axis).sum(-1)
    radius = 0.5 * axis.abs().sum(-1)
    return (torch.minimum(torch.minimum(p0, p1), p2) <= radius) & (torch.maximum(torch.maximum(p0, p1), p2) >= -radius)


def _overlaps(triangles, voxels):
    """Separating-axis test of triangles (K, 3, 3) against the unit voxels whose low corners are voxels (K, 3)."""
    v0, v1, v2 = (triangles[:, i] - voxels - 0.5 for i in range(3))
    lower, upper = torch.minimum(torch.minimum(v0, v1), v2), torch.maximum(torch.maximum(v0, v1), v2)
    hit = ((lower <= 0.5) & (upper >= -0.5)).all(-1)
    zero = torch.zeros_like(v0[:, 0])
    for edge in (v1 - v0, v2 - v1, v0 - v2):
        ex, ey, ez = edge.unbind(-1)
        for axis in ((zero, ez, -ey), (-ez, zero, ex), (ey, -ex, zero)):
            hit &= _axis_overlaps(v0, v1, v2, torch.stack(axis, dim=-1))
    e0, e1 = v1 - v0, v2 - v1
    normal = torch.cross(e0, e1, dim=-1)
    return hit & ((normal * v0).sum(-1).abs() <= 0.5 * normal.abs().sum(-1))


def surface_voxels(vertices, faces, resolution):
    """Active voxels (N, 3) int32 (x, y, z) of the triangles of a mesh in voxel space, sorted like upstream's z, y, x keys."""
    triangles = vertices.float()[faces]
    low = triangles.amin(1).floor().clamp_(0, resolution - 1).long()
    high = triangles.amax(1).floor().clamp_(0, resolution - 1).long()
    extent = (high - low + 1).clamp_(min=0)
    counts = extent.prod(-1)
    ends = counts.cumsum(0)
    keys, first = [], 0
    while first < len(faces):
        last = int(torch.searchsorted(ends, ends[first] - counts[first] + CANDIDATES_PER_CHUNK, right=True))
        last = max(last, first + 1)
        face = torch.repeat_interleave(torch.arange(first, last, device=faces.device), counts[first:last])
        local = torch.arange(face.shape[0], device=faces.device) + (ends[first] - counts[first]) - (ends[face] - counts[face])
        size = extent[face]
        offset = torch.stack([local % size[:, 0], local // size[:, 0] % size[:, 1], local // (size[:, 0] * size[:, 1])], dim=1)
        voxel = low[face] + offset
        del local, size, offset
        hit = _overlaps(triangles[face], voxel.float())
        voxel = voxel[hit]
        keys.append((voxel[:, 2] << 22) | (voxel[:, 1] << 11) | voxel[:, 0])
        del face, voxel, hit
        first = last
    keys = torch.unique(torch.cat(keys))
    return torch.stack([keys & 2047, (keys >> 11) & 2047, keys >> 22], dim=1).int()


def adaptive_voxels(vertices, faces, resolution, budget):
    """Upstream adaptive_vox: voxelize at int(resolution * scale), scale shrinking by 5 % while the voxels exceed budget. Returns
    (voxels (N, 3) int32, GridFrame of the grid used)."""
    scale = 1.0
    while True:
        current = int(resolution * scale)
        if current < MIN_RESOLUTION:
            raise ValueError(f"The input mesh is too complex for a {resolution} grid within {budget} voxels.")
        frame = grid_frame(vertices, current)
        voxels = surface_voxels(frame.to_voxels(vertices), faces, current)
        if voxels.shape[0] <= budget:
            return voxels, frame
        scale *= 0.95
