# SUPRA-Tex visible-surface material observations, inference port of supra3d/supra-tex (supratex/inference/native_observation.py and
# the observation part of supratex/inference/pipeline.py, MIT License, see LICENSE in this folder) in PyTorch on the GPU: the
# pixel-first voxel traversal (upstream: a Numba CPU kernel) steps all the camera rays together, the finished ones dropped at each step.
# Canonical frame: the object in the [-0.5, 0.5]^3 cube at RESOLUTION^3 voxels, Y up, the camera on +Z looking at the origin.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import math

import torch
import torch.nn.functional as F

from shared.utils.phase_progress import check_abort
from models.trellis2.sparse import SparseTensor

RESOLUTION = 1024
LATENT_RESOLUTION = 64
DEPTH_EDGE_RTOL = 0.04
TRUSTED_ALPHA = 254
ERODE = 5


def focal_pixels(camera_angle_x):
    """Focal length in pixels of the RESOLUTION-square observation."""
    return 16.0 / math.tan(camera_angle_x / 2.0) * RESOLUTION / 32.0


def trusted_mask(alpha):
    """Pixels whose alpha is at least TRUSTED_ALPHA, resized to the observation (nearest) and eroded by ERODE x ERODE: (R, R) bool."""
    trusted = (alpha >= TRUSTED_ALPHA).float()[None, None]
    trusted = F.interpolate(trusted, size=(RESOLUTION, RESOLUTION), mode="nearest-exact")
    return (-F.max_pool2d(-trusted, ERODE, 1, ERODE // 2))[0, 0] > 0.5


def _occupancy(coords):
    """Packed occupancy bits of the voxels (N, 3) of the RESOLUTION^3 grid, and their bounding box in voxels."""
    keys = (coords[:, 0].long() * RESOLUTION + coords[:, 1]) * RESOLUTION + coords[:, 2]
    bits = torch.zeros((RESOLUTION ** 3 + 7) // 8, dtype=torch.uint8, device=coords.device)
    bits.index_put_((keys >> 3,), (1 << (keys & 7)).to(torch.uint8), accumulate=True)  # keys are unique: no carry between bits
    return bits, coords.amin(0).long(), coords.amax(0).long()


def first_occupied(origin, directions, bits, low, high):
    """Upstream pixel-first DDA: the first occupied voxel along each ray (R, 3) int64, -1 where the ray leaves the occupied voxels'
    bounding box (low, high) without one. Rays start where they enter that box (no occupied voxel lies before it); every axis
    crossed at the same distance steps together (diagonal moves), as upstream."""
    cell = 1.0 / RESOLUTION
    box_min, box_max = low.double() * cell - 0.5, (high + 1).double() * cell - 0.5
    origin = origin.double()
    count = directions.shape[0]
    owners = torch.full((count, 3), -1, dtype=torch.long, device=directions.device)
    with torch.no_grad():
        flat = directions.abs() < 1e-14
        inv = 1.0 / torch.where(flat, 1.0, directions)
        t1, t2 = (box_min - origin) * inv, (box_max - origin) * inv
        t_entry = torch.where(flat, -torch.inf, torch.minimum(t1, t2)).amax(1)
        t_exit = torch.where(flat, torch.inf, torch.maximum(t1, t2)).amin(1)
        outside = (flat & ((origin < box_min) | (origin > box_max))).any(1)
        rays = (~outside & (t_entry <= t_exit) & (t_exit >= 0)).nonzero().squeeze(1)
        d, t_exit = directions[rays], t_exit[rays]
        t_start = t_entry[rays].clamp_min(0)
        point = origin + (t_start + (t_start.abs() * 1e-12).clamp_min(1e-12))[:, None] * d
        current = ((point + 0.5) / cell).floor().long()
        current = torch.maximum(torch.minimum(current, high), low)
        step = torch.where(d > 1e-14, 1, torch.where(d < -1e-14, -1, 0))
        boundary = (current + (step > 0).long()).double() * cell - 0.5
        moving = step != 0
        t_max = torch.where(moving, (boundary - origin) / torch.where(moving, d, 1.0), torch.inf)
        t_delta = torch.where(moving, cell / d.abs().clamp_min(1e-300), torch.inf)
        for iteration in range(3 * RESOLUTION + 3):
            if not len(rays):
                break
            key = (current[:, 0] * RESOLUTION + current[:, 1]) * RESOLUTION + current[:, 2]
            hit = ((bits[key >> 3] >> (key & 7).to(torch.uint8)) & 1).bool()
            owners[rays[hit]] = current[hit]
            next_t = t_max.amin(1)
            crossed = (t_max - next_t[:, None]).abs() <= 1e-12 * next_t.abs().clamp_min(1.0)[:, None]
            current = current + step * crossed
            t_max = t_max + torch.where(crossed, t_delta, 0)
            alive = ~hit & torch.isfinite(next_t) & (next_t <= t_exit + 1e-12) & ((current >= low) & (current <= high)).all(1)
            rays, current, t_max, t_delta, step, t_exit = (value[alive] for value in (rays, current, t_max, t_delta, step, t_exit))
            if iteration % 64 == 63:
                check_abort()
    return owners


def visible_cells(coords, camera_angle_x, distance, trusted, stats=None):
    """Upstream pixel_first_occupied for the canonical camera: the first occupied voxel of every trusted pixel, keeping the pixels
    whose first-hit depth has no discontinuity in their 3 x 3 neighborhood (relative range above DEPTH_EDGE_RTOL). Returns the unique
    visible voxels (V, 3) int64."""
    device = coords.device
    bits, low, high = _occupancy(coords)
    ys, xs = trusted.nonzero(as_tuple=True)
    focal = focal_pixels(camera_angle_x)
    directions = torch.stack([(xs.double() + 0.5 - RESOLUTION / 2) / focal, -(ys.double() + 0.5 - RESOLUTION / 2) / focal,
                              -torch.ones_like(xs, dtype=torch.float64)], 1)
    origin = torch.tensor([0.0, 0.0, distance], dtype=torch.float64, device=device)
    owners = first_occupied(origin, directions, bits, low, high)
    del bits
    hit = owners[:, 0] >= 0
    if not bool(hit.any()):
        raise RuntimeError("No pixel of the object image reaches the mesh: the image and the mesh do not match")
    depth = torch.full((RESOLUTION, RESOLUTION), torch.nan, dtype=torch.float32, device=device)
    depth[ys[hit], xs[hit]] = (distance - ((owners[hit, 2].double() + 0.5) / RESOLUTION - 0.5)).float()
    filled = depth.nan_to_num(-torch.inf)[None, None]
    local_max = F.max_pool2d(filled, 3, 1, 1)[0, 0]
    local_min = -F.max_pool2d(-depth.nan_to_num(torch.inf)[None, None], 3, 1, 1)[0, 0]
    edge = ((local_max - local_min) / depth).nan_to_num(nan=0.0, posinf=0.0, neginf=0.0) > DEPTH_EDGE_RTOL
    admitted = hit & ~edge[ys, xs]
    if stats is not None:
        stats.update(trusted=len(ys), first_hits=int(hit.sum()), admitted=int(admitted.sum()))
    if not bool(admitted.any()):
        raise RuntimeError("No pixel of the object image gives a reliable observation of the mesh")
    keys = torch.unique((owners[admitted, 0] * RESOLUTION + owners[admitted, 1]) * RESOLUTION + owners[admitted, 2])
    return torch.stack([keys // (RESOLUTION * RESOLUTION), keys // RESOLUTION % RESOLUTION, keys % RESOLUTION], 1)


def project(coords, camera_angle_x, distance):
    """Image coordinates (N, 2) in [0, 1] (u right, v down) of the voxel centers seen by the canonical camera."""
    points = (coords.double() + 0.5) / RESOLUTION - 0.5
    fx = 0.5 / math.tan(camera_angle_x / 2.0)
    depth = (distance - points[:, 2]).clamp_min(1e-8)
    return torch.stack([0.5 + fx * points[:, 0] / depth, 0.5 - fx * points[:, 1] / depth], 1).float()


def sample_maps(maps, uv):
    """Bilinear samples (N, C) in [0, 1] of the maps (C, H, W) at uv (N, 2), border padding (upstream grid_sample)."""
    grid = (uv * 2 - 1).view(1, 1, -1, 2)
    return F.grid_sample(maps[None].float(), grid, mode="bilinear", align_corners=False, padding_mode="border")[0, :, 0].T.clamp(0, 1)


def _cell_keys(coords):
    cells = coords.long() // (RESOLUTION // LATENT_RESOLUTION)
    return (cells[:, 0] * LATENT_RESOLUTION + cells[:, 1]) * LATENT_RESOLUTION + cells[:, 2]


def boundary_distance(surface_counts, observed_counts):
    """Six-neighbor distance of every active latent cell to the nearest active unobserved one (0 for those, -1 when none)."""
    active = (surface_counts > 0).view((LATENT_RESOLUTION,) * 3)
    frontier = active & ~(observed_counts > 0).view_as(active)
    distance = torch.full(active.shape, -1, dtype=torch.int16, device=active.device)
    distance[frontier] = 0
    step = 0
    while bool(frontier.any()):
        grown = torch.zeros_like(frontier)
        grown[1:] |= frontier[:-1]; grown[:-1] |= frontier[1:]
        grown[:, 1:] |= frontier[:, :-1]; grown[:, :-1] |= frontier[:, 1:]
        grown[:, :, 1:] |= frontier[:, :, :-1]; grown[:, :, :-1] |= frontier[:, :, 1:]
        frontier = grown & active & (distance < 0)
        step += 1
        distance[frontier] = step
    return distance.view(-1)


def observation_message(encoded, surface_coords, visible, shape_coords, normalization):
    """The 34-channel observation (upstream encode_partial): the material latents of the observed cells normalized with the partial
    statistics, the fraction of each cell's surface voxels that is observed and the cell's distance to the unobserved surface.
    encoded: texture encoder latents of the visible voxels; surface_coords (N, 3) and visible (N,) bool: the surface voxels."""
    keys = _cell_keys(surface_coords)
    all_counts = torch.bincount(keys, minlength=LATENT_RESOLUTION ** 3)
    observed_counts = torch.bincount(keys[visible], minlength=LATENT_RESOLUTION ** 3)
    latent_keys = (encoded.coords[:, 0].long() * LATENT_RESOLUTION + encoded.coords[:, 1]) * LATENT_RESOLUTION + encoded.coords[:, 2]
    if not torch.equal(torch.sort(latent_keys).values, torch.nonzero(observed_counts).flatten()):
        raise RuntimeError("encoded observation cells differ from the observed surface cells")
    shape_keys = (shape_coords[:, 0].long() * LATENT_RESOLUTION + shape_coords[:, 1]) * LATENT_RESOLUTION + shape_coords[:, 2]
    if not torch.equal(torch.sort(shape_keys).values, torch.nonzero(all_counts).flatten()):
        raise RuntimeError("surface cells differ from the shape latent cells")
    coverage = (observed_counts[latent_keys].float() / all_counts[latent_keys].float()).unsqueeze(1)
    distance = boundary_distance(all_counts, observed_counts)[latent_keys].float().unsqueeze(1)
    mean, std = normalization
    return SparseTensor(torch.cat([(encoded.feats.float() - mean) / std, coverage, distance], 1), encoded.coords)


def texture_input(surface_coords, visible, albedo, mra, uv):
    """Texture encoder input of the visible surface voxels: albedo and metallic / roughness / coverage maps sampled at their
    projections, in [-1, 1]."""
    features = torch.cat([sample_maps(albedo, uv[visible]), sample_maps(mra, uv[visible])], 1) * 2 - 1
    return SparseTensor(features, surface_coords[visible].int())
