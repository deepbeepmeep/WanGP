# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Triangle mesh container shared by the 3D models, the file formats, the rasterizer and the renderer.

Conventions (those of glTF 2.0, so that a Mesh is written to GLB without conversion):
- Right-handed coordinates, +Y up, the front of the asset faces +Z (-X is the asset's right), units in metres.
- Faces are counter-clockwise seen from outside: the right-hand-rule normal (v1 - v0) x (v2 - v0) points outwards.
- One index buffer for every vertex attribute: vertices are duplicated along UV seams (and wherever an attribute differs).
- uvs: (0, 0) is the top-left corner of the texture image, u grows to the right and v downwards, so texel [r, c] of an
  [H, W, C] image covers u in [c / W, (c + 1) / W] and v in [r / H, (r + 1) / H]. OBJ/PLY files store 1 - v; their readers
  and writers convert.
- colors: per-vertex sRGB-encoded values in 0..1 (3 or 4 channels, alpha linear). The GLB writer stores them as linear
  COLOR_0, as glTF requires, and the GLB reader converts them back.
- base_color: uint8 sRGB texture [H, W, 3|4]; metallic_roughness: uint8 linear texture [H, W, 3] with G = roughness and
  B = metallic (R unused, the glTF packing); normal_map: uint8 linear tangent-space normal map [H, W, 3], OpenGL/glTF
  convention (+X red, +Y green, +Z blue). Image rows are stored top to bottom.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch

VERTEX_FIELDS = ("vertices", "normals", "uvs", "colors")
TEXTURE_FIELDS = ("base_color", "metallic_roughness", "normal_map")


@dataclass
class Mesh:
    vertices: torch.Tensor                            # [V, 3] float32
    faces: torch.Tensor                               # [F, 3] int64
    normals: torch.Tensor | None = None               # [V, 3] unit vertex normals
    uvs: torch.Tensor | None = None                   # [V, 2] (seams duplicated)
    colors: torch.Tensor | None = None                # [V, 3|4] float sRGB 0..1
    base_color: torch.Tensor | None = None            # [H, W, 3|4] uint8 sRGB
    metallic_roughness: torch.Tensor | None = None    # [H, W, 3] uint8 (G roughness, B metallic)
    normal_map: torch.Tensor | None = None            # [H, W, 3] uint8 tangent space

    @property
    def device(self) -> torch.device:
        return self.vertices.device

    def to(self, device) -> Mesh:
        return Mesh(**{f.name: None if (t := getattr(self, f.name)) is None else t.to(device) for f in dataclasses.fields(self)})

    def replace(self, **changes) -> Mesh:
        return dataclasses.replace(self, **changes)


def select_vertices(mesh: Mesh, index: torch.Tensor, faces: torch.Tensor) -> Mesh:
    """Mesh whose vertex i is the old vertex index[i] (every per-vertex attribute follows), with the given faces."""
    return mesh.replace(faces=faces, **{k: getattr(mesh, k)[index] for k in VERTEX_FIELDS if getattr(mesh, k) is not None})


def submesh(mesh: Mesh, face_mask: torch.Tensor) -> Mesh:
    """Keeps the selected faces (bool mask or indices) and only the vertices they use (in their original order)."""
    used, faces = torch.unique(mesh.faces[face_mask], return_inverse=True)
    return select_vertices(mesh, used, faces)


def weld(vertices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unique positions (exact float equality, -0.0 == 0.0) in lexicographic order and, for each vertex, the index of its unique
    position: torch.unique(vertices, dim=0, return_inverse=True). On the CPU, where torch.unique compares rows one by one (seconds for
    large meshes), the rows are ordered by stable radix sorts of integer keys that order like the coordinates, one column at a time."""
    if vertices.is_cuda:
        return torch.unique(vertices, dim=0, return_inverse=True)
    width = vertices.element_size() * 8
    keys = (vertices + 0.0).view(getattr(torch, f"int{width}"))  # + 0.0: -0.0 becomes 0.0
    keys = keys ^ ((keys >> (width - 1)) & ((1 << (width - 1)) - 1))  # negative floats: reversed magnitude bits
    order = torch.arange(len(vertices), device=vertices.device)
    for axis in reversed(range(vertices.shape[1])):
        order = order[torch.sort(keys[order, axis], stable=True).indices]
    keys = keys[order]
    first = torch.ones(len(order), dtype=torch.bool, device=vertices.device)
    first[1:] = (keys[1:] != keys[:-1]).any(1)
    inverse = torch.empty_like(order)
    inverse[order] = torch.cumsum(first, 0) - 1
    return vertices[order[first]], inverse


def face_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Unnormalized face normals (v1 - v0) x (v2 - v0); their length is twice the face area."""
    v = vertices[faces]
    return torch.linalg.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])


def vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Area-weighted unit vertex normals, smooth across vertices that share a position (UV seams). Vertices without any
    non-degenerate face get +Y, so that the result is always a valid glTF NORMAL."""
    welded, inverse = weld(vertices)
    corner_normals = face_normals(vertices.float(), faces).repeat_interleave(3, 0)
    sums = torch.zeros_like(welded, dtype=torch.float32).index_add_(0, inverse[faces].reshape(-1), corner_normals)[inverse]
    length = sums.norm(dim=1, keepdim=True)
    up = torch.tensor([0.0, 1.0, 0.0], device=vertices.device)
    return torch.where(length > 0, sums / length.clamp_min(1e-30), up)


def bounding_box(vertices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return vertices.amin(0), vertices.amax(0)


def srgb_to_linear(x: torch.Tensor) -> torch.Tensor:
    """sRGB transfer function decode (IEC 61966-2-1) of values in 0..1."""
    return torch.where(x <= 0.04045, x / 12.92, ((x.clamp_min(0.04045) + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: torch.Tensor) -> torch.Tensor:
    """sRGB transfer function encode of linear values in 0..1."""
    return torch.where(x <= 0.0031308, x * 12.92, 1.055 * x.clamp_min(0.0031308) ** (1 / 2.4) - 0.055)
