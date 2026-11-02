# Hi3D v1.0 surface extraction, inference port of hi3d-official/Hi3Dv1.0 (sparc3d/sparcubes/{extract,sparse_cube,fill_hole}.py and
# sparc3d/structures/point.py: Hi3D Open License Agreement 1.0; see LICENSE and NOTICE in this folder): the per-corner attributes of
# the decoded voxels are averaged on the shared cube corners, FlexiCubes extracts the zero level set (the decoded sign decides the
# side, the distance its magnitude) and ear clipping closes the remaining boundary loops.
# Modified by DeepBeepMeep for WanGP: unique corners and edges as int64 keys instead of row-wise uniques, positions kept in grid units
# (the caller maps them back to the input mesh's frame).
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import numpy as np
import torch

from .flexicubes import FlexiCubes

CUBE_CORNERS = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1), (1, 1, 1))  # FlexiCubes' corner order
_extractors = {}


def _flexicubes(device):
    if device not in _extractors:
        _extractors[device] = FlexiCubes(device=device)
    return _extractors[device]


def extract_surface(attributes, coords, resolution):
    """attributes (N, 40) decoder output (8 SDF sign logits, 8 corner distances, 8 x 3 corner deformations, before their activations),
    coords (N, 3) int32 voxels of a grid of `resolution`. Returns (vertices (V, 3) FP32 in grid units, faces (F, 3) int64), empty when
    no voxel straddles the surface."""
    device = coords.device
    span = resolution + 2
    index = (coords[:, 0].long() * span + coords[:, 1].long()) * span + coords[:, 2].long()
    index, order = torch.sort(index)  # FlexiCubes' sparse ambiguity resolution expects ordered cubes
    coords = coords.index_select(0, order)
    offsets = torch.tensor([(dx * span + dy) * span + dz for dx, dy, dz in CUBE_CORNERS], dtype=torch.int64, device=device)
    keys, cubes = torch.unique((index.unsqueeze(1) + offsets).view(-1), return_inverse=True)
    del index
    cubes = cubes.to(torch.int32)
    count = torch.bincount(cubes, minlength=keys.shape[0]).unsqueeze(1).float()
    # the mean of each corner attribute over the cubes sharing the corner, channel group by channel group (sign, distance, offset)
    means = []
    for first, last, activation in ((0, 8, torch.sigmoid), (8, 16, torch.tanh), (16, 40, lambda x: torch.tanh(x) * 0.5)):
        values = activation(attributes[:, first:last].index_select(0, order).float()).view(-1, (last - first) // 8)
        means.append(torch.zeros((keys.shape[0], values.shape[1]), dtype=torch.float32, device=device).index_add_(0, cubes, values).div_(count))
        del values
    del order, count
    sign, distance, deform = means
    grid = torch.stack([keys // (span * span), keys // span % span, keys % span], dim=1).float()
    del keys
    positions = grid.add_(deform)
    sdf = torch.where(sign[:, 0] >= 0.5, 1.0, -1.0) * distance[:, 0].abs()
    del sign, distance, deform, means
    vertices, faces = _flexicubes(device)(positions, sdf, cubes.view(-1, 8), resolution, cube_index_map=coords)
    del positions, sdf, cubes
    if faces.shape[0] == 0:
        return vertices, faces.long()
    return vertices, fill_holes(vertices, faces).long()


def fill_holes(vertices, faces, length_thresh=10000):
    """Upstream fill_mesh_holes: each boundary loop of at most length_thresh edges triangulated by ear clipping (on the CPU)."""
    directed = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    keys, order = torch.sort(directed.min(dim=1).values.long() * vertices.shape[0] + directed.max(dim=1).values)
    once = torch.ones_like(keys, dtype=torch.bool)  # undirected edges of a single face, kept in upstream's (directed) order
    once[1:] &= keys[1:] != keys[:-1]
    once[:-1] &= keys[:-1] != keys[1:]
    del keys
    boundary = directed[order[once].sort().values].cpu().numpy()
    del directed, order, once
    if len(boundary) == 0:
        return faces
    points = vertices.cpu().numpy()
    loops = [loop for loop in _boundary_loops(boundary) if 3 <= len(loop) <= length_thresh]
    if not loops:
        return faces
    by_length = {}
    for loop in loops:
        by_length.setdefault(len(loop), []).append(loop)
    new_faces = np.vstack([_ear_clip(points, np.array(group, dtype=np.int64)) for group in by_length.values()])
    return torch.cat([faces, torch.from_numpy(new_faces).to(faces)])


def _boundary_loops(boundary):
    """Boundary half-edges (B, 2) chained into closed vertex loops, each half-edge used once."""
    outgoing = {}
    for idx, (start, _) in enumerate(boundary):
        outgoing.setdefault(start, []).append(idx)
    visited = [False] * len(boundary)
    loops = []
    for i in range(len(boundary)):
        if visited[i]:
            continue
        visited[i] = True
        start = boundary[i][0]
        loop = [start, boundary[i][1]]
        while loop[-1] != start:
            for next_idx in outgoing.get(loop[-1], []):
                if not visited[next_idx]:
                    visited[next_idx] = True
                    loop.append(boundary[next_idx][1])
                    break
            else:
                break  # dead end: not a closed loop
        if loop[0] == loop[-1] and len(loop) > 3:
            loop.pop()
            loops.append(loop)
    return loops


def _ear_clip(points, loops):
    """Triangulates boundary loops of the same length (B, L), clipping the sharpest corner of each loop first (upstream scoring, a
    folded corner first of all), all loops of the group at once."""
    rows = np.arange(loops.shape[0])
    line, faces = points[loops], []
    while loops.shape[1] > 3:
        length = loops.shape[1]
        next_pts = np.roll(line, -1, axis=1)
        tangent = next_pts - line
        tangent = tangent / (np.linalg.norm(tangent, axis=2, keepdims=True) + 1e-6)
        tangent_prev = np.roll(tangent, 1, axis=1)
        normal = np.cross(-tangent_prev, tangent)
        center = line.mean(axis=1, keepdims=True)
        sin_sign = np.sign((np.cross(next_pts - center, line - center) * normal).sum(axis=2, keepdims=True))
        sin_val = np.linalg.norm(normal, axis=2, keepdims=True) * sin_sign
        cos_val = -(tangent_prev * tangent).sum(axis=2, keepdims=True)
        angle = np.arctan2(sin_val, cos_val)[..., 0]
        collapsed = angle == 0.0
        ear = np.argmax(np.where(collapsed, np.inf, 1.0 / np.where(collapsed, 1.0, angle)), axis=1)
        faces.append(np.stack([loops[rows, (ear + 1) % length], loops[rows, ear], loops[rows, (ear - 1) % length]], axis=1))
        keep = np.ones(loops.shape, dtype=bool)
        keep[rows, ear] = False
        loops, line = loops[keep].reshape(-1, length - 1), line[keep].reshape(-1, length - 1, 3)
    faces.append(loops[:, [2, 1, 0]])
    return np.vstack(faces)
