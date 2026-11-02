# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""glTF 2.0 container code: GLB writing and reading (and .gltf reading), generation metadata in asset.extras, quick file
statistics, and the format dispatchers load_mesh / export_mesh / mesh_file_stats (OBJ, PLY and STL are in formats.py).

Written GLB files follow the Mesh conventions (Y-up, +Z front, metres): one node, mesh, primitive and material;
little-endian buffers; a JSON chunk padded with spaces and a BIN chunk padded with zeros; every bufferView 4-byte aligned;
POSITION with min/max, NORMAL (computed when the mesh has none), TEXCOORD_0, linear COLOR_0 (float); uint16 or uint32
indices; a PBR metallic-roughness material whose textures (base colour sRGB, metallic-roughness and normal map linear) are
embedded as PNG or JPEG (RGBA textures always PNG).
"""
from __future__ import annotations

import base64
import json
import math
import os
import struct
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import torch

from .images import MIME_TYPES, bytes_reader, decode_image, encode_image, file_reader, image_size, pack_textures, remap_uvs, to_channels
from .mesh import Mesh, linear_to_srgb, srgb_to_linear, vertex_normals

GLB_MAGIC, CHUNK_JSON, CHUNK_BIN = 0x46546C67, 0x4E4F534A, 0x004E4942
ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER = 34962, 34963
COMPONENT_TYPES = {np.dtype(np.int8): 5120, np.dtype(np.uint8): 5121, np.dtype(np.int16): 5122, np.dtype(np.uint16): 5123, np.dtype(np.uint32): 5125, np.dtype(np.float32): 5126}
DTYPES = {code: dtype.newbyteorder("<") for dtype, code in COMPONENT_TYPES.items()}
TYPE_SIZES = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}  # MAT4: float skin inverse bind matrices
NORMALIZERS = {5120: 127.0, 5121: 255.0, 5122: 32767.0, 5123: 65535.0}
UNSUPPORTED_EXTENSIONS = {"KHR_draco_mesh_compression", "EXT_meshopt_compression", "KHR_texture_basisu"}


# ------------------------------------------------------------------------------------------------------------ writing
def _numpy(tensor: torch.Tensor, dtype) -> np.ndarray:
    return tensor.detach().to(device="cpu").numpy().astype(dtype, copy=False)


def _pack_glb(gltf: dict, chunks: bytes) -> bytes:
    """GLB file from the JSON document and the already framed chunks that follow the JSON chunk."""
    text = json.dumps(gltf, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    text += b" " * (-len(text) % 4)
    body = struct.pack("<II", len(text), CHUNK_JSON) + text + chunks
    return struct.pack("<III", GLB_MAGIC, 2, 12 + len(body)) + body


class _GltfWriter:
    """Accumulates the binary buffer and the glTF objects of one GLB file: one node, mesh, primitive and material per
    added Mesh, all nodes in the default scene."""

    def __init__(self, image_format: str):
        self.image_format, self.parts, self.size = image_format, [], 0
        self.gltf = {"nodes": [], "meshes": [], "materials": [], "images": [], "textures": [], "animations": [], "bufferViews": [], "accessors": []}

    def view(self, data: bytes, target: int | None = None) -> int:
        if pad := -self.size % 4:
            self.parts.append(bytes(pad))
            self.size += pad
        self.gltf["bufferViews"].append({"buffer": 0, "byteOffset": self.size, "byteLength": len(data)} | ({"target": target} if target else {}))
        self.parts.append(data)
        self.size += len(data)
        return len(self.gltf["bufferViews"]) - 1

    def accessor(self, array: np.ndarray, target: int | None, bounds: bool = False) -> int:
        array = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))
        accessor = {"bufferView": self.view(array.tobytes(), target), "componentType": COMPONENT_TYPES[np.dtype(array.dtype.type)],
                    "count": len(array), "type": "SCALAR" if array.ndim == 1 else f"VEC{array.shape[1]}"}
        if bounds:
            accessor |= {"min": np.atleast_1d(array.min(0)).tolist(), "max": np.atleast_1d(array.max(0)).tolist()}
        self.gltf["accessors"].append(accessor)
        return len(self.gltf["accessors"]) - 1

    def texture(self, tensor: torch.Tensor) -> dict:
        array = _numpy(tensor, np.uint8)
        fmt = "png" if array.shape[2] == 4 else self.image_format
        self.gltf["images"].append({"bufferView": self.view(encode_image(array, fmt)), "mimeType": MIME_TYPES[fmt]})
        self.gltf["textures"].append({"sampler": 0, "source": len(self.gltf["images"]) - 1})
        return {"index": len(self.gltf["textures"]) - 1}

    def add_mesh(self, mesh: Mesh, name: str | None = None) -> dict:
        """Adds a node, mesh, primitive and material for `mesh`; returns the primitive (e.g. to add morph targets)."""
        if len(mesh.faces) == 0:
            raise ValueError("Cannot write an empty mesh")
        if mesh.uvs is None and any(t is not None for t in (mesh.base_color, mesh.metallic_roughness, mesh.normal_map)):
            raise ValueError("Textured mesh without uvs")
        normals = vertex_normals(mesh.vertices, mesh.faces) if mesh.normals is None else mesh.normals
        attributes = {"POSITION": self.accessor(_numpy(mesh.vertices, np.float32), ARRAY_BUFFER, bounds=True),
                      "NORMAL": self.accessor(_numpy(normals, np.float32), ARRAY_BUFFER)}
        if mesh.uvs is not None:
            attributes["TEXCOORD_0"] = self.accessor(_numpy(mesh.uvs, np.float32), ARRAY_BUFFER)
        blend = False
        if mesh.colors is not None:
            colors = mesh.colors.float()
            colors = torch.cat([srgb_to_linear(colors[:, :3].clamp(0, 1)), colors[:, 3:].clamp(0, 1)], 1)
            attributes["COLOR_0"] = self.accessor(_numpy(colors, np.float32), ARRAY_BUFFER)
            blend = colors.shape[1] == 4 and bool((colors[:, 3] < 1).any())
        index_dtype = np.uint16 if len(mesh.vertices) <= 65535 else np.uint32
        pbr = {"baseColorFactor": [1.0, 1.0, 1.0, 1.0], "metallicFactor": 0.0, "roughnessFactor": 1.0}
        material = {"name": f"{name}_material" if name else "material", "pbrMetallicRoughness": pbr}
        if mesh.base_color is not None:
            pbr["baseColorTexture"] = self.texture(mesh.base_color)
            blend |= mesh.base_color.shape[2] == 4 and bool((mesh.base_color[..., 3] < 255).any())
        if mesh.metallic_roughness is not None:
            pbr |= {"metallicFactor": 1.0, "metallicRoughnessTexture": self.texture(mesh.metallic_roughness)}
        if mesh.normal_map is not None:
            material["normalTexture"] = self.texture(mesh.normal_map)
        if blend:
            material["alphaMode"] = "BLEND"
        self.gltf["materials"].append(material)
        primitive = {"attributes": attributes, "indices": self.accessor(_numpy(mesh.faces, index_dtype).reshape(-1), ELEMENT_ARRAY_BUFFER),
                     "material": len(self.gltf["materials"]) - 1, "mode": 4}
        named = {"name": name} if name else {}
        self.gltf["meshes"].append(named | {"primitives": [primitive]})
        self.gltf["nodes"].append(named | {"mesh": len(self.gltf["meshes"]) - 1})
        return primitive

    def glb(self, extras: dict | None) -> bytes:
        gltf = {"asset": {"version": "2.0", "generator": "WanGP mesh3d"} | ({"extras": extras} if extras else {}),
                "scene": 0, "scenes": [{"nodes": list(range(len(self.gltf["nodes"])))}]}
        for key, value in self.gltf.items():
            if key == "images" and value:
                gltf["samplers"] = [{"magFilter": 9729, "minFilter": 9987}]
            if value and key not in ("bufferViews", "accessors"):
                gltf[key] = value
        binary = b"".join(self.parts) + bytes(-self.size % 4)
        gltf |= {"buffers": [{"byteLength": len(binary)}], "bufferViews": self.gltf["bufferViews"], "accessors": self.gltf["accessors"]}
        return _pack_glb(gltf, struct.pack("<II", len(binary), CHUNK_BIN) + binary)


def mesh_to_glb(mesh: Mesh, *, image_format: str = "png", extras: dict | None = None) -> bytes:
    """Binary glTF 2.0 of a Mesh (see the module docstring). image_format "png" or "jpeg" applies to RGB textures.

    The material is PBR metallic-roughness: baseColorFactor 1, metallic/roughness factors 1 with a metallic-roughness
    texture, otherwise metallic 0 and roughness 1; alphaMode BLEND when the base colour texture or the vertex colours
    have a non-opaque alpha channel. extras (JSON-serializable) becomes asset.extras."""
    writer = _GltfWriter(image_format)
    writer.add_mesh(mesh)
    return writer.glb(extras)


def meshes_to_glb(parts: list[Mesh], names: list[str] | None = None, *, image_format: str = "png", extras: dict | None = None) -> bytes:
    """Multi-part GLB: one node, mesh and material per part (each part keeps its own textures and colours), named
    names[i] or part_0, part_1, ... All parts share the scene's coordinate frame (no node transforms). glb_to_mesh
    merges them back into one Mesh."""
    writer = _GltfWriter(image_format)
    for i, part in enumerate(parts):
        writer.add_mesh(part, names[i] if names else f"part_{i}")
    return writer.glb(extras)


def animated_mesh_to_glb(mesh: Mesh, vertex_frames: torch.Tensor, fps: float, *, normals: bool = True, image_format: str = "png",
                         extras: dict | None = None) -> bytes:
    """Animated GLB of a mesh whose vertices move: vertex_frames [F, V, 3] are the vertex positions of each frame (same
    vertices and faces as mesh). Each frame becomes a morph target (POSITION displacement from mesh.vertices, plus the
    NORMAL displacement of the recomputed smooth normals when normals is True), and one animation named "animation"
    drives the target weights: frame i, at time i / fps, has weight 1 on target i, interpolated linearly between frames,
    so playback is smooth at any display rate. The rest pose (all weights 0) is mesh.vertices. Supported by glTF
    viewers including model-viewer (three.js handles up to 256 morph targets on every WebGL2 device: subsample longer
    clips). File size grows by 12 (24 with normals) bytes per vertex and frame."""
    writer = _GltfWriter(image_format)
    primitive = writer.add_mesh(mesh.replace(normals=None) if normals else mesh)
    base, base_normals = mesh.vertices.float(), vertex_normals(mesh.vertices, mesh.faces)
    targets = []
    for frame in vertex_frames:
        target = {"POSITION": writer.accessor(_numpy(frame.float() - base, np.float32), ARRAY_BUFFER, bounds=True)}
        if normals:
            target["NORMAL"] = writer.accessor(_numpy(vertex_normals(frame.float(), mesh.faces) - base_normals, np.float32), ARRAY_BUFFER)
        targets.append(target)
    primitive["targets"] = targets
    count = len(targets)
    writer.gltf["meshes"][0]["weights"] = [0.0] * count
    times = writer.accessor(np.arange(count, dtype=np.float32) / np.float32(fps), None, bounds=True)
    weights = writer.accessor(np.eye(count, dtype=np.float32).reshape(-1), None)
    writer.gltf["animations"].append({"name": "animation", "samplers": [{"input": times, "output": weights, "interpolation": "LINEAR"}],
                                      "channels": [{"sampler": 0, "target": {"node": 0, "path": "weights"}}]})
    return writer.glb(extras)


# ------------------------------------------------------------------------------------------------------------ extras
def _split_glb(data) -> tuple[dict, bytes | None, int]:
    """(JSON document, BIN chunk payload or None, offset of the first chunk after the JSON chunk)."""
    if len(data) < 20:
        raise ValueError("Truncated GLB file")
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC or version != 2:
        raise ValueError("Not a glTF 2.0 binary (GLB) file")
    json_length, kind = struct.unpack_from("<II", data, 12)
    if kind != CHUNK_JSON:
        raise ValueError("GLB file without a leading JSON chunk")
    gltf = json.loads(bytes(data[20:20 + json_length]).rstrip(b" \0").decode("utf-8"))
    binary, pos, end = None, 20 + json_length, min(length, len(data))
    while pos + 8 <= end:
        chunk_length, kind = struct.unpack_from("<II", data, pos)
        if kind == CHUNK_BIN and binary is None:
            binary = memoryview(data)[pos + 8:pos + 8 + chunk_length]
        pos += 8 + chunk_length
    return gltf, binary, 20 + json_length


def glb_compress_images(data: bytes, max_size: int | None = None, quality: int = 88) -> bytes:
    """GLB whose embedded pictures are re-encoded for delivery: JPEG, or PNG when the picture uses its alpha, the longer side reduced to
    max_size. Geometry, the other buffer views and the rest of the document are kept; pictures PIL cannot read stay as they are."""
    import io

    from PIL import Image

    gltf, binary, _ = _split_glb(data)
    images = {image["bufferView"]: index for index, image in enumerate(gltf.get("images", [])) if "bufferView" in image}
    if binary is None or not images or any(view.get("buffer", 0) != 0 for view in gltf["bufferViews"]):
        return bytes(data)
    parts, size = [], 0
    for index, view in enumerate(gltf["bufferViews"]):
        start = view.get("byteOffset", 0)
        chunk = bytes(binary[start:start + view["byteLength"]])
        if index in images:
            try:
                with Image.open(io.BytesIO(chunk)) as picture:
                    picture.load()
                    alpha = "A" in picture.getbands() and picture.getchannel("A").getextrema()[0] < 255
                    if max_size and max(picture.size) > max_size:
                        picture.thumbnail((max_size, max_size), Image.LANCZOS)
                    encoded = io.BytesIO()
                    picture.save(encoded, "PNG", optimize=True) if alpha else picture.convert("RGB").save(encoded, "JPEG", quality=quality)
                chunk = encoded.getvalue()
                gltf["images"][images[index]]["mimeType"] = "image/png" if alpha else "image/jpeg"
            except OSError:
                pass
        parts.append(bytes(-size % 4))
        size += len(parts[-1])
        view["byteOffset"], view["byteLength"] = size, len(chunk)
        parts.append(chunk)
        size += len(chunk)
    payload = b"".join(parts)
    payload += bytes(-len(payload) % 4)
    gltf["buffers"][0]["byteLength"] = len(payload)
    return _pack_glb(gltf, struct.pack("<II", len(payload), CHUNK_BIN) + payload)


def glb_write_extras(data: bytes, extras: dict, *, merge: bool = True) -> bytes:
    """GLB with asset.extras set to extras (merged into the existing asset.extras dict when merge is True). Only the JSON
    chunk is rewritten (re-padded to 4 bytes); the following chunks, including BIN, are kept byte-identical."""
    gltf, _, json_end = _split_glb(data)
    asset = gltf.setdefault("asset", {"version": "2.0"})
    old = asset.get("extras")
    asset["extras"] = {**old, **extras} if merge and isinstance(old, dict) else dict(extras)
    return _pack_glb(gltf, bytes(data[json_end:struct.unpack_from("<I", data, 8)[0]]))


def _read_document(source) -> dict:
    """glTF JSON of a GLB or .gltf path / bytes, reading only the JSON chunk of a GLB file."""
    if isinstance(source, (bytes, bytearray, memoryview)):
        return _split_glb(source)[0] if bytes(source[:4]) == b"glTF" else json.loads(bytes(source).decode("utf-8"))
    with open(source, "rb") as handle:
        head = handle.read(20)
        if head[:4] != b"glTF":
            return json.loads((head + handle.read()).decode("utf-8"))
        return _split_glb(head + handle.read(struct.unpack_from("<I", head, 12)[0]))[0]


def glb_add_files(data: bytes, files: dict[str, list[bytes]], key: str) -> bytes:
    """GLB with files (encoded pictures by role, e.g. {"image_refs": [PNG bytes]}) appended to its BIN chunk, one buffer view each, listed
    in asset.extras[key] as {role: [buffer view index]}. No texture uses them, so viewers ignore them, and the JSON chunk stays small for
    the readers of the settings. A GLB that already lists files under key is returned unchanged."""
    gltf, binary, _ = _split_glb(data)
    extras = gltf.setdefault("asset", {"version": "2.0"}).setdefault("extras", {})
    if key in extras or not any(files.values()):
        return bytes(data)
    payload, views, listed = bytearray(binary or b""), gltf.setdefault("bufferViews", []), {}
    for role, contents in files.items():
        for content in contents:
            payload += bytes(-len(payload) % 4)
            views.append({"buffer": 0, "byteOffset": len(payload), "byteLength": len(content)})
            listed.setdefault(role, []).append(len(views) - 1)
            payload += content
    payload += bytes(-len(payload) % 4)
    gltf.setdefault("buffers", [{}])[0]["byteLength"] = len(payload)
    extras[key] = listed
    return _pack_glb(gltf, struct.pack("<II", len(payload), CHUNK_BIN) + bytes(payload))


def glb_read_files(source, key: str) -> dict[str, list[bytes]]:
    """The files glb_add_files stored under asset.extras[key] of a GLB (path or bytes): {role: [bytes]}, {} without any."""
    data = source if isinstance(source, (bytes, bytearray, memoryview)) else Path(source).read_bytes()
    gltf, binary, _ = _split_glb(data)
    listed, views = gltf.get("asset", {}).get("extras", {}).get(key), gltf.get("bufferViews", [])
    if not isinstance(listed, dict) or binary is None:
        return {}
    span = lambda view: bytes(binary[view.get("byteOffset", 0):view.get("byteOffset", 0) + view["byteLength"]])
    return {role: [span(views[index]) for index in indices] for role, indices in listed.items()}


def glb_read_extras(source) -> dict:
    """asset.extras of a GLB/.gltf file (path or bytes), {} when absent. Reads only the JSON chunk."""
    extras = _read_document(source).get("asset", {}).get("extras")
    return extras if isinstance(extras, dict) else {}


# ------------------------------------------------------------------------------------------------------------ reading
def _uri_bytes(uri: str, base: Path | None) -> bytes:
    if uri.startswith("data:"):
        header, _, payload = uri.partition(",")
        return base64.b64decode(payload) if header.endswith(";base64") else unquote(payload).encode("latin-1")
    if base is None:
        raise ValueError(f"External glTF resource {uri!r} without a base folder")
    path = (base / unquote(uri)).resolve()
    if base.resolve() not in path.parents:
        raise ValueError(f"glTF resource {uri!r} lies outside the folder of the glTF file")
    return path.read_bytes()


def _load_document(source) -> tuple[dict, list, Path | None]:
    if isinstance(source, (bytes, bytearray, memoryview)):
        data, base = source, None
    else:
        data, base = Path(source).read_bytes(), Path(source).parent
    gltf, binary = _split_glb(data)[:2] if bytes(data[:4]) == b"glTF" else (json.loads(bytes(data).decode("utf-8")), None)
    if unsupported := UNSUPPORTED_EXTENSIONS & set(gltf.get("extensionsRequired", [])):
        raise ValueError(f"Unsupported glTF extensions: {', '.join(sorted(unsupported))}")
    buffers = [_uri_bytes(b["uri"], base) if "uri" in b else binary for b in gltf.get("buffers", [])]
    return gltf, buffers, base


def _view_bytes(gltf: dict, buffers: list, index: int):
    view = gltf["bufferViews"][index]
    start = view.get("byteOffset", 0)
    return memoryview(buffers[view["buffer"]])[start:start + view["byteLength"]]


def _accessor(gltf: dict, buffers: list, index: int) -> np.ndarray:
    """Accessor data as [count] (SCALAR) or [count, n]; float32 for float and normalized integer data."""
    accessor = gltf["accessors"][index]
    code, size, count = accessor["componentType"], TYPE_SIZES[accessor["type"]], accessor["count"]
    dtype = DTYPES[code]
    if "bufferView" in accessor:
        view = gltf["bufferViews"][accessor["bufferView"]]
        stride = view.get("byteStride") or size * dtype.itemsize
        array = np.ndarray((count, size), dtype, buffer=_view_bytes(gltf, buffers, accessor["bufferView"]), offset=accessor.get("byteOffset", 0), strides=(stride, dtype.itemsize)).copy()
    else:
        array = np.zeros((count, size), dtype)
    if sparse := accessor.get("sparse"):
        indices, values = sparse["indices"], sparse["values"]
        rows = np.frombuffer(_view_bytes(gltf, buffers, indices["bufferView"]), DTYPES[indices["componentType"]], sparse["count"], indices.get("byteOffset", 0))
        array[rows] = np.frombuffer(_view_bytes(gltf, buffers, values["bufferView"]), dtype, sparse["count"] * size, values.get("byteOffset", 0)).reshape(-1, size)
    if accessor.get("normalized"):
        array = np.maximum(array / NORMALIZERS[code], -1.0)
    if code == 5126 or accessor.get("normalized"):
        array = array.astype(np.float32)
    return array[:, 0] if size == 1 else array


def _local_matrix(node: dict, pose: dict | None = None) -> np.ndarray:
    """Local matrix of a node; pose overrides its translation / rotation / scale (animation sample)."""
    if "matrix" in node:
        return np.array(node["matrix"], np.float64).reshape(4, 4).T
    pose = pose or {}
    x, y, z, w = pose.get("rotation", node.get("rotation", (0.0, 0.0, 0.0, 1.0)))
    rotation = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                         [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                         [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    matrix = np.eye(4)
    matrix[:3, :3] = rotation * np.array(pose.get("scale", node.get("scale", (1.0, 1.0, 1.0))))
    matrix[:3, 3] = pose.get("translation", node.get("translation", (0.0, 0.0, 0.0)))
    return matrix


def _world_matrices(gltf: dict, poses: dict | None = None) -> dict[int, np.ndarray]:
    """World matrix of every node of the default scene (all root nodes without scenes), in depth-first order."""
    nodes = gltf["nodes"]
    scenes = gltf.get("scenes")
    if scenes:
        roots = scenes[gltf.get("scene", 0)].get("nodes", [])
    else:
        children = {c for node in nodes for c in node.get("children", [])}
        roots = [i for i in range(len(nodes)) if i not in children]
    matrices, stack = {}, [(i, np.eye(4)) for i in reversed(roots)]
    while stack:
        index, parent = stack.pop()
        matrices[index] = parent @ _local_matrix(nodes[index], (poses or {}).get(index))
        stack.extend((c, matrices[index]) for c in reversed(nodes[index].get("children", [])))
    return matrices


def _mesh_instances(gltf: dict, poses: dict | None = None):
    """(node index or None, mesh index, world matrix) of every mesh instance of the default scene."""
    nodes = gltf.get("nodes", [])
    if not nodes:
        yield from ((None, i, np.eye(4)) for i in range(len(gltf.get("meshes", []))))
        return
    for index, matrix in _world_matrices(gltf, poses).items():
        if "mesh" in nodes[index]:
            yield index, nodes[index]["mesh"], matrix


# ------------------------------------------------------------------------------------------------------------ animation
def animation_timing(gltf: dict, animation: int = 0) -> dict | None:
    """Frames of a glTF animation, from the min / max / count of its sampler inputs (glTF requires their bounds, so
    only the JSON is read): the densest track sets the frame rate (WanGP's animated GLBs key every frame), and frame i
    is at start + i / fps. None without animation; frames, fps and duration are None when the bounds are missing."""
    animations = gltf.get("animations", [])
    if animation >= len(animations):
        return None
    inputs = [gltf["accessors"][sampler["input"]] for sampler in animations[animation]["samplers"]]
    timing = {"name": animations[animation].get("name", ""), "frames": None, "fps": None, "duration": None, "start": 0.0}
    if not inputs or not all("min" in a and "max" in a for a in inputs):
        return timing
    start, end = min(a["min"][0] for a in inputs), max(a["max"][0] for a in inputs)
    rates = [(a["count"] - 1) / (a["max"][0] - a["min"][0]) for a in inputs if a["count"] > 1 and a["max"][0] > a["min"][0]]
    fps = max(rates) if rates else None
    return {**timing, "frames": int(round((end - start) * fps)) + 1 if fps else 1, "fps": fps, "duration": end - start, "start": start}


def _sample_track(gltf: dict, buffers: list, sampler: dict, time: float, rotation: bool) -> np.ndarray:
    """Value of an animation sampler at `time` (clamped to its keys): STEP, LINEAR (slerp for rotations) or CUBICSPLINE."""
    times = _accessor(gltf, buffers, sampler["input"]).astype(np.float64)
    interpolation = sampler.get("interpolation", "LINEAR")
    cubic = interpolation == "CUBICSPLINE"
    values = _accessor(gltf, buffers, sampler["output"]).astype(np.float64).reshape(len(times), 3 if cubic else 1, -1)
    middle = 1 if cubic else 0
    index = int(np.clip(np.searchsorted(times, time, side="right") - 1, 0, len(times) - 1))
    if time <= times[0] or index == len(times) - 1 or interpolation == "STEP":
        return values[index, middle]
    u = (time - times[index]) / (times[index + 1] - times[index])
    a, b = values[index, middle], values[index + 1, middle]
    if cubic:
        span = times[index + 1] - times[index]
        return (2 * u ** 3 - 3 * u ** 2 + 1) * a + (u ** 3 - 2 * u ** 2 + u) * span * values[index, 2] + (3 * u ** 2 - 2 * u ** 3) * b + (u ** 3 - u ** 2) * span * values[index + 1, 0]
    if not rotation:
        return a + u * (b - a)
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    if dot > 0.9995:
        return a + u * (b - a)
    angle = math.acos(dot)
    return (math.sin((1 - u) * angle) * a + math.sin(u * angle) * b) / math.sin(angle)


def _animation_poses(gltf: dict, buffers: list, time: float, animation: int = 0) -> dict[int, dict]:
    """Animated node properties at `time`: {node: {"translation" | "rotation" | "scale" | "weights": value}}."""
    clip, poses = gltf["animations"][animation], {}
    for channel in clip["channels"]:
        target = channel["target"]
        if "node" not in target or target["path"] not in ("translation", "rotation", "scale", "weights"):
            continue
        value = _sample_track(gltf, buffers, clip["samplers"][channel["sampler"]], time, target["path"] == "rotation")
        poses.setdefault(target["node"], {})[target["path"]] = value / np.linalg.norm(value) if target["path"] == "rotation" else value
    return poses


def _skin(points: np.ndarray, joints: np.ndarray, weights: np.ndarray, matrices: np.ndarray, translate: bool) -> np.ndarray:
    """Linear blend skinning of points [V, 3] with joint matrices [J, 4, 4] (translation dropped for normals)."""
    result = np.zeros_like(points)
    for k in range(joints.shape[1]):
        matrix = matrices[joints[:, k]]
        moved = np.einsum("vij,vj->vi", matrix[:, :3, :3], points)
        result += weights[:, k:k + 1] * (moved + matrix[:, :3, 3] if translate else moved)
    return result


def _triangles(indices: np.ndarray, mode: int) -> np.ndarray:
    if mode == 4:
        return indices[:len(indices) // 3 * 3].reshape(-1, 3)
    i = np.arange(max(len(indices) - 2, 0))
    if mode == 5:  # strip: odd triangles swap their last two corners to keep the winding
        odd = i % 2
        return np.stack([indices[i], indices[i + 1 + odd], indices[i + 2 - odd]], 1)
    return np.stack([indices[i + 1], indices[i + 2], np.full_like(i, indices[0]) if len(i) else i], 1)  # fan


def _texture_array(gltf: dict, buffers: list, base: Path | None, info: dict | None, channels: int) -> np.ndarray | None:
    if info is None:
        return None
    texture = gltf["textures"][info["index"]]
    source = texture.get("extensions", {}).get("EXT_texture_webp", {}).get("source", texture.get("source"))
    if source is None:
        return None
    image = gltf["images"][source]
    data = _view_bytes(gltf, buffers, image["bufferView"]) if "bufferView" in image else _uri_bytes(image["uri"], base)
    return to_channels(decode_image(bytes(data)), channels)


def _gltf_to_mesh(gltf: dict, buffers: list, base: Path | None, device, time: float | None = None) -> Mesh:
    poses = _animation_poses(gltf, buffers, time) if time is not None and gltf.get("animations") else None
    worlds = _world_matrices(gltf, poses) if poses is not None and gltf.get("skins") else None
    parts = []
    for node_index, mesh_index, matrix in _mesh_instances(gltf, poses):
        node = {} if node_index is None else gltf["nodes"][node_index]
        mesh_def = gltf["meshes"][mesh_index]
        # posed only at an animation time: morph target weights, and skins (whose node transform glTF ignores)
        weights = None if poses is None else np.asarray(poses.get(node_index, {}).get("weights", node.get("weights", mesh_def.get("weights", []))), np.float64)
        skin = gltf["skins"][node["skin"]] if worlds is not None and "skin" in node else None
        if skin is not None:
            inverse_bind = _accessor(gltf, buffers, skin["inverseBindMatrices"]).reshape(-1, 4, 4).transpose(0, 2, 1) if "inverseBindMatrices" in skin else np.eye(4)[None]
            joint_matrices = np.stack([worlds[joint] for joint in skin["joints"]]) @ inverse_bind
        flip = skin is None and np.linalg.det(matrix[:3, :3]) < 0
        normal_matrix = np.linalg.inv(matrix[:3, :3]).T
        for primitive in mesh_def["primitives"]:
            mode = primitive.get("mode", 4)
            if mode not in (4, 5, 6):
                continue
            if UNSUPPORTED_EXTENSIONS & set(primitive.get("extensions", {})):
                raise ValueError("Compressed glTF primitives are not supported")
            attributes = primitive["attributes"]
            positions = _accessor(gltf, buffers, attributes["POSITION"]).astype(np.float64)
            normals = _accessor(gltf, buffers, attributes["NORMAL"]).astype(np.float64) if "NORMAL" in attributes else None
            for weight, target in zip(weights if weights is not None else [], primitive.get("targets", [])):
                if weight != 0 and "POSITION" in target:
                    positions += weight * _accessor(gltf, buffers, target["POSITION"])
                if weight != 0 and "NORMAL" in target and normals is not None:
                    normals += weight * _accessor(gltf, buffers, target["NORMAL"])
            indices = _accessor(gltf, buffers, primitive["indices"]).astype(np.int64) if "indices" in primitive else np.arange(len(positions))
            faces = _triangles(indices, mode)
            sets = [n for n in range(4) if f"JOINTS_{n}" in attributes and f"WEIGHTS_{n}" in attributes] if skin is not None else []
            if sets:
                joints = np.concatenate([_accessor(gltf, buffers, attributes[f"JOINTS_{n}"]).reshape(len(positions), -1).astype(np.int64) for n in sets], 1)
                joint_weights = np.concatenate([_accessor(gltf, buffers, attributes[f"WEIGHTS_{n}"]).reshape(len(positions), -1) for n in sets], 1)
                vertices = _skin(positions, joints, joint_weights, joint_matrices, True)
                normals = None if normals is None else _skin(normals, joints, joint_weights, joint_matrices, False)
            else:
                vertices = positions @ matrix[:3, :3].T + matrix[:3, 3]
                normals = None if normals is None else normals @ normal_matrix.T
            part = {"vertices": vertices, "faces": faces[:, ::-1] if flip else faces, "material": primitive.get("material")}
            if normals is not None:
                part["normals"] = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
            material = gltf["materials"][part["material"]] if part["material"] is not None else {}
            pbr = material.get("pbrMetallicRoughness", {})
            uv_set = pbr.get("baseColorTexture", {}).get("texCoord", 0)
            if f"TEXCOORD_{uv_set}" in attributes:
                part["uvs"] = _accessor(gltf, buffers, attributes[f"TEXCOORD_{uv_set}"])
            if "COLOR_0" in attributes:
                part["colors"] = _accessor(gltf, buffers, attributes["COLOR_0"])
            part["factor"] = np.array(pbr.get("baseColorFactor", (1.0, 1.0, 1.0, 1.0)), np.float32)
            part["textures"] = {"base_color": pbr.get("baseColorTexture"), "metallic_roughness": pbr.get("metallicRoughnessTexture"), "normal_map": material.get("normalTexture")}
            parts.append(part)
    if not parts:
        raise ValueError("The glTF file contains no triangle mesh")

    # One material keeps its textures. Several base colour textures are packed into one atlas (uvs remapped; the
    # metallic-roughness and normal maps are then dropped). Base colour factors x COLOR_0 become vertex colours unless
    # all white: renderers and viewers multiply them with the texture, as glTF does.
    textures, uvs = {}, None
    if len({p["material"] for p in parts}) == 1 and all("uvs" in p for p in parts) and any(v is not None for v in parts[0]["textures"].values()):
        textures = {k: _texture_array(gltf, buffers, base, info, 4 if k == "base_color" else 3) for k, info in parts[0]["textures"].items()}
        uvs = [p["uvs"] for p in parts]
    elif any(p["textures"]["base_color"] is not None and "uvs" in p for p in parts):
        source = [p["textures"]["base_color"]["index"] if p["textures"]["base_color"] is not None and "uvs" in p else None for p in parts]
        keys = list(dict.fromkeys(k for k in source if k is not None))
        atlas, boxes = pack_textures([_texture_array(gltf, buffers, base, {"index": k}, 4) for k in keys] + [np.full((4, 4, 4), 255, np.uint8)])
        uvs = [remap_uvs(p["uvs"], boxes[keys.index(k)], atlas.shape) if k is not None else remap_uvs(np.full((len(p["vertices"]), 2), 0.5), boxes[-1], atlas.shape)
               for p, k in zip(parts, source)]
        textures = {"base_color": atlas}
    if textures.get("base_color") is not None and (textures["base_color"][..., 3] == 255).all():
        textures["base_color"] = np.ascontiguousarray(textures["base_color"][..., :3])
    colors = []
    for part in parts:
        color = np.broadcast_to(part["factor"], (len(part["vertices"]), 4)).copy()
        if "colors" in part:
            color[:, :part["colors"].shape[1]] *= part["colors"]
        colors.append(color)
    if all((c == 1).all() for c in colors):
        colors = None

    offsets = np.cumsum([0] + [len(p["vertices"]) for p in parts])
    tensor = lambda arrays, dtype: torch.from_numpy(np.concatenate(arrays).astype(dtype)).to(device)
    mesh = Mesh(vertices=tensor([p["vertices"] for p in parts], np.float32), faces=tensor([p["faces"] + o for p, o in zip(parts, offsets)], np.int64))
    if all("normals" in p for p in parts):
        mesh.normals = tensor([p["normals"] for p in parts], np.float32)
    if uvs is not None:
        mesh.uvs = tensor(uvs, np.float32)
        for key, array in textures.items():
            setattr(mesh, key, None if array is None else torch.from_numpy(array).to(device))
    if colors is not None:  # linear -> sRGB, opaque alpha dropped
        width = max(c.shape[1] for c in colors)
        colors = torch.from_numpy(np.concatenate([np.pad(c, ((0, 0), (0, width - c.shape[1])), constant_values=1.0) for c in colors]).astype(np.float32))
        if width == 4 and bool((colors[:, 3] == 1).all()):
            colors = colors[:, :3]
        mesh.colors = torch.cat([linear_to_srgb(colors[:, :3].clamp(0, 1)), colors[:, 3:]], 1).to(device)
    return mesh


def glb_to_mesh(source, *, device="cpu", time: float | None = None) -> Mesh:
    """Mesh from a GLB (path or bytes) or .gltf file (path; external buffers and images must lie in its folder).

    Every triangle primitive of the default scene is merged with its node transform applied. All textures are kept when
    the primitives share one material; with several materials, their base colour textures are packed into one atlas
    (uvs remapped) and their metallic-roughness / normal maps dropped. Base colour factors x COLOR_0 become sRGB vertex
    colours unless all white. Points and lines are ignored. Without `time`, morph targets and skins are ignored (rest
    pose); with `time` (seconds, see animation_timing), the first animation is sampled there: node translations,
    rotations and scales, morph target weights and linear blend skinning. Tensors are created on `device`."""
    return _gltf_to_mesh(*_load_document(source), device, time)


# ------------------------------------------------------------------------------------------------------------ statistics
def _gltf_stats(path: str) -> dict:
    with open(path, "rb") as handle:
        head = handle.read(20)
        if head[:4] == b"glTF":
            gltf = _split_glb(head + handle.read(struct.unpack_from("<I", head, 12)[0]))[0]
            bin_start = handle.tell() + 8
        else:
            gltf, bin_start = json.loads((head + handle.read()).decode("utf-8")), None
        vertices = faces = 0
        has_uv = has_colors = False
        low, high = np.full(3, np.inf), np.full(3, -np.inf)
        accessors = gltf.get("accessors", [])
        for _node, mesh_index, matrix in _mesh_instances(gltf):
            for primitive in gltf["meshes"][mesh_index]["primitives"]:
                mode = primitive.get("mode", 4)
                if mode not in (4, 5, 6):
                    continue
                attributes = primitive["attributes"]
                position = accessors[attributes["POSITION"]]
                count = accessors[primitive["indices"]]["count"] if "indices" in primitive else position["count"]
                vertices += position["count"]
                faces += count // 3 if mode == 4 else max(count - 2, 0)
                has_uv |= "TEXCOORD_0" in attributes
                has_colors |= "COLOR_0" in attributes
                if "min" in position and "max" in position:
                    corners = np.array(np.meshgrid(*zip(position["min"], position["max"]), indexing="ij")).reshape(3, -1).T
                    corners = corners @ matrix[:3, :3].T + matrix[:3, 3]
                    low, high = np.minimum(low, corners.min(0)), np.maximum(high, corners.max(0))
        textures = []
        for image in gltf.get("images", []):
            if "bufferView" in image:
                view = gltf["bufferViews"][image["bufferView"]]
                if gltf["buffers"][view["buffer"]].get("uri") is None and bin_start is not None:
                    start = bin_start + view.get("byteOffset", 0)
                    textures.append(image_size(file_reader(handle, start, start + view["byteLength"])))
                else:
                    textures.append(None)
            elif image.get("uri", "").startswith("data:"):
                textures.append(image_size(bytes_reader(_uri_bytes(image["uri"], None))))
            else:
                with open(Path(path).parent / unquote(image.get("uri", "")), "rb") as image_file:
                    textures.append(image_size(file_reader(image_file)))
    return {"format": "glb" if head[:4] == b"glTF" else "gltf", "vertices": vertices, "faces": faces, "materials": len(gltf.get("materials", [])),
            "textures": textures, "bbox": [low.tolist(), high.tolist()] if np.isfinite(low).all() else None,
            "has_uv": has_uv, "has_colors": has_colors, "animations": len(gltf.get("animations", [])), "animation": animation_timing(gltf), "splats": 0,
            "file_size": os.path.getsize(path)}


# ------------------------------------------------------------------------------------------------------------ dispatch
MESH_EXTENSIONS = (".glb", ".gltf", ".obj", ".ply", ".stl")
EXPORT_FORMATS = ("glb", "obj", "stl", "ply")


def load_mesh(path, *, device="cpu", time: float | None = None) -> Mesh:
    """Mesh from a .glb, .gltf, .obj (+ .mtl and textures), .ply or .stl file, with tensors on `device`; `time` poses an
    animated glTF (see glb_to_mesh), the other formats have no animation."""
    from . import formats
    readers = {".glb": glb_to_mesh, ".gltf": glb_to_mesh, ".obj": formats.read_obj, ".ply": formats.read_ply, ".stl": formats.read_stl}
    suffix = Path(path).suffix.lower()
    if suffix not in readers:
        raise ValueError(f"Unsupported mesh file {path!r}")
    return glb_to_mesh(path, device=device, time=time) if suffix in (".glb", ".gltf") else readers[suffix](path, device=device)


def mesh_file_stats(path) -> dict:
    """Quick statistics without decoding textures: format, vertices, faces (triangles), materials, textures [(w, h) or
    None], bbox [[x, y, z] min, max] or None, has_uv, has_colors, animations (glTF animations), animation (timing of the
    first one, see animation_timing; glTF only), splats (Gaussians of a
    3DGS splat PLY, else 0), file_size (bytes of the file itself).

    GLB/glTF: counts per mesh instance from the accessors, bbox from the POSITION min/max under the node transforms,
    texture sizes from the image headers. OBJ: positions ("v" lines) and triangles after fan triangulation. PLY: header
    counts (polygons count as faces) and the bbox of the vertex positions. STL: triangles and distinct positions."""
    from . import formats
    suffix = Path(path).suffix.lower()
    stats = {".glb": _gltf_stats, ".gltf": _gltf_stats, ".obj": formats.obj_stats, ".ply": formats.ply_stats, ".stl": formats.stl_stats}
    if suffix not in stats:
        raise ValueError(f"Unsupported mesh file {path!r}")
    return stats[suffix](str(path))


def export_mesh(mesh: Mesh, fmt: str, *, name: str = "mesh", z_up: bool = False, scale: float = 1.0, image_format: str = "png") -> dict[str, bytes]:
    """Files of a mesh export, as {file name: bytes}:
    - "glb": {name.glb} (always Y-up);
    - "obj": {name.obj, name.mtl, name_basecolor.png, name_roughness.png, name_metallic.png, name_normal.png} (textures
      present only; v = 1 - glTF v; welded positions/uvs/normals with separate indices);
    - "stl": {name.stl} binary STL (geometry only);
    - "ply": {name.ply} binary little-endian PLY with vertex normals, uvs (s, t) and colours (vertex colours, or the base
      colour texture sampled at the vertices).
    z_up rotates Y-up to Z-up (front -Y, Blender/printing convention) for obj/stl/ply; scale multiplies the positions
    (e.g. 1000 for millimetres)."""
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"Unsupported export format {fmt!r}")
    if z_up and fmt == "glb":
        raise ValueError("glTF is always Y-up")
    if z_up or scale != 1.0:
        rotate = (lambda v: torch.stack([v[:, 0], -v[:, 2], v[:, 1]], 1)) if z_up else (lambda v: v)
        mesh = mesh.replace(vertices=rotate(mesh.vertices) * scale, normals=None if mesh.normals is None else rotate(mesh.normals))
    if fmt == "glb":
        return {f"{name}.glb": mesh_to_glb(mesh, image_format=image_format)}
    from . import formats
    return {"obj": formats.write_obj, "stl": formats.write_stl, "ply": formats.write_ply}[fmt](mesh, name)

