# Placement of an existing mesh in SUPRA-Tex's canonical frame (the [-0.5, 0.5]^3 cube, Y up, a perspective camera on +Z looking at
# the origin whose image spans the cube's width at the origin): rotation, scale and offset matching the mesh silhouette to the object
# image's alpha. Upstream receives this placement from its own geometry stage (Pixal3D generates the mesh in the camera frame); meshes of
# other models or files are fitted here. WanGP meshes face +Z with Y up: the search covers the front half, seen from level to nearly
# straight down (product photos are often taken from above), then the other half when it fits clearly better.
# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from shared.mesh3d.raster import rasterize
from shared.utils.phase_progress import check_abort

SIZE = 192  # silhouette resolution of the search
CUBE_HALF = 0.49  # the placed mesh stays inside the canonical cube
FOVS = (30.0, 37.5, 45.0, 52.5)  # horizontal fields of view tried (degrees)
ROTATION_MARGIN = 0.03  # IoU a mesh facing another axis must gain over the front placements
FRONT_YAWS = tuple(float(yaw) for yaw in range(-90, 91, 15))
BACK_YAWS = (120.0, 150.0, 180.0, 210.0, 240.0)
PITCHES = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0)  # the camera's elevation above the object
SEARCH_FACES = 20000  # faces of the decimated copy the silhouettes are rasterized from
FIT_STEPS = ((0.04, 2), (0.02, 1), (0.01, 0.5))  # (scale factor, offset in pixels) of the pattern search
REFINE_STEPS = ((0.02, 1), (0.01, 0.5))
CANDIDATES = 4  # coarse placements given a full fit


@dataclass
class Placement:
    rotation: torch.Tensor  # (3, 3)
    center: torch.Tensor  # (3,) mesh point placed at the origin before scaling
    scale: float
    offset: torch.Tensor  # (3,)
    camera_angle_x: float
    distance: float
    iou: float
    yaw: float = 0.0
    pitch: float = 0.0

    def apply(self, vertices):
        """Canonical positions of mesh vertices (V, 3)."""
        return ((vertices.float() - self.center) @ self.rotation.T) * self.scale + self.offset


def _rotation(yaw, pitch, device):
    a, b = math.radians(yaw), math.radians(pitch)
    ry = torch.tensor([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]], dtype=torch.float32, device=device)
    rx = torch.tensor([[1, 0, 0], [0, math.cos(b), -math.sin(b)], [0, math.sin(b), math.cos(b)]], dtype=torch.float32, device=device)
    return rx @ ry


def _silhouette(points, faces, distance, focal, size=SIZE):
    """Coverage (size, size) of the placed points seen by the canonical camera (rows downward)."""
    depth = distance - points[:, 2]
    clip = torch.stack([2 * focal * points[:, 0], -2 * focal * points[:, 1], torch.zeros_like(depth), depth], 1)
    return rasterize(clip, faces, size)[0][0] >= 0


def _iou(a, b):
    return float((a & b).sum()) / max(1.0, float((a | b).sum()))


def _box(mask):
    rows, cols = mask.any(1).nonzero(), mask.any(0).nonzero()
    if not len(rows):
        return None
    return float(cols[0]), float(rows[0]), float(cols[-1] + 1), float(rows[-1] + 1)


def _search(points, faces, target, distance, focal, scale, offset, steps):
    """Pattern search of scale and image-plane offset on the IoU with the target silhouette, by decreasing steps (scale factor,
    offset in pixels of the target). Returns (iou, scale, offset)."""
    size = target.shape[0]
    best = _iou(_silhouette(points * scale + offset, faces, distance, focal, size), target)
    pixel = distance / focal / size
    for step_scale, step in steps:
        step *= pixel
        improved = True
        while improved:
            improved = False
            for factor, delta in ((1 + step_scale, (0, 0)), (1 - step_scale, (0, 0)), (1, (step, 0)), (1, (-step, 0)), (1, (0, step)), (1, (0, -step))):
                trial_scale, trial_offset = scale * factor, offset + torch.tensor([delta[0], delta[1], 0.0], device=points.device)
                score = _iou(_silhouette(points * trial_scale + trial_offset, faces, distance, focal, size), target)
                if score > best + 1e-4:
                    best, scale, offset, improved = score, trial_scale, trial_offset, True
        check_abort()
    return best, scale, offset


def _fit(points, faces, target, distance, focal, steps=FIT_STEPS):
    """Scale and image-plane offset of the points (centered) matching the target silhouette: bounding box matching, then a
    pattern search on the IoU (none without steps). Returns (iou, scale, offset)."""
    goal = _box(target)
    extent = (points[:, :2].abs().amax() * 2).item()
    scale = (goal[3] - goal[1] if goal[3] - goal[1] > goal[2] - goal[0] else goal[2] - goal[0]) / SIZE * distance / (2 * focal * max(extent, 1e-6)) * 2
    offset = torch.zeros(3, device=points.device)
    for _ in range(4):
        box = _box(_silhouette(points * scale + offset, faces, distance, focal))
        if box is None:
            break
        size = max(box[2] - box[0], box[3] - box[1])
        scale *= max(goal[2] - goal[0], goal[3] - goal[1]) / max(size, 1.0)
        shift_u = ((goal[0] + goal[2]) - (box[0] + box[2])) / 2 / SIZE
        shift_v = ((goal[1] + goal[3]) - (box[1] + box[3])) / 2 / SIZE
        offset = offset + torch.tensor([shift_u * distance / focal, -shift_v * distance / focal, 0.0], device=points.device)
    return _search(points, faces, target, distance, focal, scale, offset, steps)


def place_mesh(vertices, faces, alpha, *, max_faces=SEARCH_FACES):
    """Placement of a mesh (vertices (V, 3), faces (F, 3) on the GPU) in the canonical frame matching alpha (H, W) uint8, the object
    image's alpha on its square canvas. The search runs on a decimated copy of the mesh: a coarse grid of views fitted by their bounding
    box, full fits of the best ones, then yaw and pitch refined by halving steps and the field of view."""
    device = vertices.device
    if faces.shape[0] > max_faces:
        from shared.mesh3d import Mesh
        from shared.mesh3d.cleanup import decimate

        reduced = decimate(Mesh(vertices.cpu(), faces.cpu()), max_faces)
        vertices, faces = reduced.vertices.to(device), reduced.faces.to(device)
    target = F.interpolate((alpha.float() / 255)[None, None], size=(SIZE, SIZE), mode="area")[0, 0] > 0.5
    low, high = vertices.amin(0), vertices.amax(0)
    center = (low + high) / 2
    centered = vertices.float() - center

    def evaluate(yaw, pitch, fov, steps=FIT_STEPS, start=None):
        focal = 0.5 / math.tan(math.radians(fov) / 2)
        distance = 0.5 / math.tan(math.radians(fov) / 2)
        rotation = _rotation(yaw, pitch, device)
        points = centered @ rotation.T
        if start is None:
            iou, scale, offset = _fit(points, faces, target, distance, focal, steps)
        else:  # a neighbouring view: its scale and offset start from the current best
            iou, scale, offset = _search(points, faces, target, distance, focal, *start, steps)
        return iou, (yaw, pitch, fov, rotation, scale, offset, distance)

    fov = FOVS[1]
    coarse = sorted(((evaluate(yaw, pitch, fov, steps=()), yaw, pitch) for yaw in FRONT_YAWS for pitch in PITCHES), key=lambda one: one[0][0], reverse=True)
    best = max((evaluate(yaw, pitch, fov) for _, yaw, pitch in coarse[:CANDIDATES]), key=lambda candidate: candidate[0])
    # a mesh facing another axis (front and back silhouettes are mirrors: the front is preferred)
    back = max(((evaluate(yaw, pitch, fov, steps=()), yaw, pitch) for yaw in BACK_YAWS for pitch in PITCHES[:3]), key=lambda one: one[0][0])
    if back[0][0] > coarse[0][0][0] + ROTATION_MARGIN:
        candidate = evaluate(back[1], back[2], fov)
        if candidate[0] > best[0] + ROTATION_MARGIN:
            best = candidate
    for step in (7.5, 3.75, 1.875):
        improved = True
        while improved:
            improved = False
            for delta in ((-step, 0.0), (step, 0.0), (0.0, -step), (0.0, step)):
                yaw, pitch = best[1][0] + delta[0], min(max(best[1][1] + delta[1], -30.0), 89.0)
                candidate = evaluate(yaw, pitch, fov, REFINE_STEPS, start=best[1][4:6])
                if candidate[0] > best[0] + 1e-4:
                    best, improved = candidate, True
            check_abort()
    yaw, pitch = best[1][:2]
    for fov in FOVS:
        if fov != best[1][2]:
            candidate = evaluate(yaw, pitch, fov)
            if candidate[0] > best[0]:
                best = candidate
    iou, (yaw, pitch, fov, rotation, scale, offset, distance) = best
    points = centered @ rotation.T
    fine = F.interpolate((alpha.float() / 255)[None, None], size=(2 * SIZE, 2 * SIZE), mode="area")[0, 0] > 0.5
    iou, scale, offset = _search(points, faces, fine, distance, distance, scale, offset, ((0.01, 1), (0.004, 0.5), (0.002, 0.25)))
    placed = points * scale + offset
    shrink = min(1.0, CUBE_HALF / placed.abs().amax().item())  # mesh and camera scaled together: the image is unchanged
    return Placement(rotation, center, scale * shrink, offset * shrink, math.radians(fov), distance * shrink, iou, yaw, pitch)
