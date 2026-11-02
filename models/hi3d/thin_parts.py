# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Thin parts of the input mesh that Hi3D loses. Hi3D rebuilds the surface from signed distances on its voxel grid: a sheet thinner
than a few voxels (a leaf, fabric, paper, a stem) gets perforated or vanishes, while the surface stays closed. The input surface of the
thin parts with no refined surface near it is added back to the refined mesh with a margin around it; the refined fragments left inside
(lace of a perforated sheet) are removed, and the outer rings of the margin overlap the refined surface beyond, so that no crack
remains. Solid parts are never touched (Hi3D may carve them deeper)."""
import numpy as np
import torch
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from shared.mesh3d import Mesh
from shared.mesh3d.cleanup import merge_vertices
from .voxelize import grid_frame, surface_voxels

THIN_GRID = 256  # resolution of the thin-part search
INTERIOR_REACH = 4  # surface voxels with no interior voxel within this many voxels are thin parts (about 1.5 % of the object's size)
COVER_GRID = 512  # resolution of the coverage test: the refined surface's voxels met by the line through each input face
COVER_REACH = 3.0  # length of that line on either side of the face (voxels of COVER_GRID)
CLOSING_RINGS = 4  # a lost region also takes the covered islands it surrounds within this many rings (the lace of a perforated sheet)
MIN_REGION = 1e-3  # lost regions smaller than this share of the input's area are left to Hi3D (it reshapes details of solid parts)
MARGIN_RINGS = 4  # rings of input faces added around a lost region: it replaces the refined fragments of the first ones
OVERLAP_RINGS = 2  # the last rings overlap the refined surface kept beyond


def _keys(voxels):
    voxels = voxels.long()
    return (voxels[:, 2] << 22) | (voxels[:, 1] << 11) | voxels[:, 0]


def thin_faces(vertices, faces, resolution=THIN_GRID):
    """Faces (F,) bool of the parts of a mesh with no inside behind them on a resolution^3 grid: the inside is what the outside cannot reach
    once the surface voxels are dilated by one (which seals small holes of the surface); the surface voxels of a solid part lie within
    INTERIOR_REACH voxels of it (an oblique surface spans two or three voxels)."""
    frame = grid_frame(vertices, resolution)
    voxels = surface_voxels(frame.to_voxels(vertices), faces, resolution).long().cpu().numpy()
    occupied = np.zeros((resolution,) * 3, bool)
    occupied[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = True
    sealed = ndimage.binary_dilation(occupied)
    labels, _ = ndimage.label(~sealed)
    inside = ~sealed & (labels != labels[0, 0, 0])  # the grid's margin keeps its corner outside
    del sealed, labels
    thin = occupied & ~ndimage.binary_dilation(inside, iterations=INTERIOR_REACH)
    centers = frame.to_voxels(vertices[faces].mean(1)).floor().long().clamp_(0, resolution - 1).cpu().numpy()
    return torch.from_numpy(thin[centers[:, 0], centers[:, 1], centers[:, 2]]).to(vertices.device)


def _count_near(points, occupied, frame, radius):
    """Number (P,) of voxels of occupied (sorted keys) within radius voxels of points (P, 3)."""
    voxel = frame.to_voxels(points).floor().long()
    count = torch.zeros(points.shape[0], dtype=torch.int32, device=points.device)
    steps = torch.arange(-radius, radius + 1, device=points.device)
    for offset in torch.cartesian_prod(steps, steps, steps):
        keys = _keys(voxel + offset)
        count += occupied[torch.searchsorted(occupied, keys).clamp_(max=len(occupied) - 1)] == keys
    return count


def _surface_keys(mesh_vertices, mesh_faces, frame):
    return torch.sort(_keys(surface_voxels(frame.to_voxels(mesh_vertices.float()), mesh_faces, frame.resolution))).values


def lost_faces(vertices, faces, refined, resolution=COVER_GRID, reach=COVER_REACH):
    """Faces (F,) bool of a mesh where the line through the face along its normal meets no surface voxel of the refined mesh within reach
    voxels on either side (the grid of the mesh): the surface is gone there, or perforated (a lace sheet keeps refined surface around
    the face, not across it)."""
    frame = grid_frame(vertices, resolution)
    occupied = _surface_keys(refined.vertices, refined.faces, frame)
    corners = frame.to_voxels(vertices[faces])
    centers = corners.mean(1)
    normals = torch.nn.functional.normalize(torch.linalg.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), dim=1)
    met = torch.zeros(faces.shape[0], dtype=torch.bool, device=vertices.device)
    for step in torch.arange(-reach, reach + 0.25, 0.5).tolist():  # half-voxel steps: no voxel crossed by the line is skipped
        keys = _keys((centers + step * normals).floor())
        met |= occupied[torch.searchsorted(occupied, keys).clamp_(max=len(occupied) - 1)] == keys
    return ~met


def _large_regions(vertices, faces, selected, min_share):
    """selected (F,) bool restricted to its connected regions (faces sharing a vertex) holding at least min_share of the mesh's area."""
    index = torch.nonzero(selected).flatten()
    corners = vertices[faces]
    area = torch.linalg.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]).norm(dim=1)
    chosen = faces[index].cpu().numpy()
    count = len(chosen)
    graph = coo_matrix((np.ones(3 * count), (np.repeat(np.arange(count), 3), chosen.flatten() + count)), shape=(count + len(vertices),) * 2)
    labels = torch.from_numpy(connected_components(graph, directed=False)[1][:count]).to(faces.device)
    region_area = torch.zeros(int(labels.max()) + 1, device=faces.device).index_add_(0, labels, area[index])
    kept = torch.zeros_like(selected)
    kept[index] = region_area[labels] >= min_share * area.sum()
    return kept


def restore_thin_parts(source, refined):
    """refined (shared.mesh3d Mesh, the refinement of source in its frame) with the faces of source's thin parts it lost. Returns
    (mesh, number of faces added)."""
    source = merge_vertices(Mesh(source.vertices.float(), source.faces.long()))  # welded through UV seams: the rings cross them
    vertices, faces = source.vertices, source.faces
    thin = thin_faces(vertices, faces)
    lost = lost_faces(vertices, faces, refined)
    restored = thin & lost
    if not bool(restored.any()):
        return refined, 0

    def ring(selected, allowed):
        marked = torch.zeros(vertices.shape[0], dtype=torch.bool, device=vertices.device)
        marked[faces[selected].flatten()] = True
        return selected | (allowed & marked[faces].any(1))

    while True:  # a lost region extends over the lost solid-looking bits of the same part (knots of a stem)
        grown = ring(restored, lost)
        if bool((grown == restored).all()):
            break
        restored = grown
    closed = restored
    for _ in range(CLOSING_RINGS):
        closed = ring(closed, thin | lost)
    for _ in range(CLOSING_RINGS):  # erosion: the faces of closed touching no face outside it
        closed = closed & ~ring(~closed, closed)
    restored = _large_regions(vertices, faces, restored | closed, MIN_REGION)
    if not bool(restored.any()):
        return refined, 0
    for _ in range(MARGIN_RINGS - OVERLAP_RINGS):
        restored = ring(restored, thin | lost)
    frame = grid_frame(vertices, COVER_GRID)
    replaced = _count_near(refined.vertices[refined.faces].float().mean(1), _surface_keys(vertices, faces[restored], frame), frame, 1) > 0
    for _ in range(OVERLAP_RINGS):
        restored = ring(restored, thin | lost)
    kept = refined.faces[~replaced]
    used = torch.unique(faces[restored])
    index = torch.full((vertices.shape[0],), -1, dtype=torch.long, device=vertices.device)
    index[used] = torch.arange(len(used), device=vertices.device) + refined.vertices.shape[0]
    mesh = Mesh(torch.cat([refined.vertices, vertices[used].to(refined.vertices.dtype)]), torch.cat([kept, index[faces[restored]].to(kept.dtype)]))
    return mesh, int(restored.sum())
