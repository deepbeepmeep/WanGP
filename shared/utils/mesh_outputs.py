# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""3D gallery files and their preview render `<stem>.png`:
- meshes: GLB with the generation settings in the glTF `asset.extras["wangp"]` and, with the Embed Source Images option, the source
  pictures (lossless, alpha kept) in its binary chunk, listed in `asset.extras["wangp_source_images"]`;
- Gaussian splats: INRIA 3DGS PLY with the settings in a `comment wangp <json>` header line. Every PLY in the 3D gallery is
  a splat file: mesh PLYs are converted to GLB on import.

The preview is the 3D gallery thumbnail and the image shown where a 3D asset cannot be viewed interactively (Deepy, workspaces).
"""
import json
import os
from pathlib import Path

from shared.utils.media_imports import persist_gallery_import

SETTINGS_KEY = "wangp"
SOURCE_IMAGES_KEY = "wangp_source_images"


def preview_path(path):
    return os.path.splitext(path)[0] + ".png"


def is_splat_file(path):
    return Path(path).suffix.lower() == ".ply"


def asset_stats(path):
    """Header-level description of a 3D gallery file: {"kind": "splats", "gaussians", "sh_degree", ...} or {"kind": "mesh", **mesh_file_stats}."""
    if is_splat_file(path):
        from shared.mesh3d.splats import splat_file_stats
        return {"kind": "splats", **splat_file_stats(path)}
    from shared.mesh3d.glb import mesh_file_stats
    return {"kind": "mesh", **mesh_file_stats(path)}


def asset_summary(path):
    """Compact facts of a 3D gallery file for Deepy and MCP: kind (mesh or splats), faces / vertices / textured or
    gaussians, frame_count (1 unless animated), and fps / duration_seconds for an animated mesh."""
    stats = asset_stats(path)
    if stats["kind"] == "splats":
        return {"kind": "splats", "gaussians": stats["gaussians"], "frame_count": 1}
    summary = {"kind": "mesh", "faces": stats["faces"], "vertices": stats["vertices"], "textured": bool(stats["textures"]), "frame_count": 1}
    animation = stats["animation"]
    if animation and animation["frames"]:
        summary.update(frame_count=animation["frames"], fps=round(animation["fps"], 3) if animation["fps"] else None, duration_seconds=round(animation["duration"], 3))
    return summary


def animation_time(path, frame_no):
    """Animation time (seconds) of frame `frame_no` of a 3D gallery file; ValueError when it has no such frame."""
    animation = None if is_splat_file(path) else asset_stats(path)["animation"]
    if not animation or not animation["frames"]:
        raise ValueError("This 3D asset has no animation frames.")
    if not 0 <= frame_no < animation["frames"]:
        raise ValueError(f"frame_no must be between 0 and {animation['frames'] - 1} (the animation has {animation['frames']} frames).")
    return animation["start"] + frame_no / animation["fps"] if animation["fps"] else animation["start"]


def render_asset_views(path, angles, resolution, **options):
    """Views of a 3D gallery file (mesh or Gaussian splats) as PIL images, one scale shared by every view (angles: names
    of shared.mesh3d.render.VIEWS or (azimuth, elevation); options of iter_asset_views)."""
    return list(iter_asset_views(path, angles, resolution, chunk=len(angles), **options))


def iter_asset_views(path, angles, resolution, *, chunk, frame_no=None, mode="shaded", projection="orthographic", zoom=1.0, focus=(0.5, 0.5, 0.5), background=None):
    """Views of a 3D gallery file as a stream of PIL images rendered `chunk` at a time, every view keeping the scale of
    the whole sequence, so a long one (a turntable) holds only a few views in memory; depth views are normalized per
    chunk. See orbit_cameras for zoom and focus: RGBA with a transparent background for the colour modes unless
    background is given. frame_no poses an animated mesh at that frame (rest pose without it). Rendered on the GPU when
    there is one (a few hundred MB for typical assets)."""
    import torch
    from PIL import Image
    from shared.mesh3d.render import VIEWS, asset_cameras, render_splat_views, render_views
    device = "cuda" if torch.cuda.is_available() else "cpu"
    angles = [VIEWS[angle] if isinstance(angle, str) else angle for angle in angles]
    time = None if frame_no is None else animation_time(path, frame_no)
    with torch.inference_mode():
        if is_splat_file(path):
            from shared.mesh3d.splats import ply_to_splats
            asset, render = ply_to_splats(path, device=device), render_splat_views
        else:
            from shared.mesh3d.glb import load_mesh
            asset, render = load_mesh(path, device=device, time=time), render_views
        cameras = asset_cameras(asset, angles, resolution, projection=projection, zoom=zoom, focus=focus)
    for start in range(0, len(angles), chunk):
        with torch.inference_mode():
            images = render(asset, None, resolution, mode, background=background, cameras=tuple(camera[start:start + chunk] for camera in cameras))
            arrays = (images.clamp(0, 1) * 255).round().byte().cpu().numpy()
        del images
        yield from (Image.fromarray(array) for array in arrays)  # RGBA or RGB from the channel count


def _encoded_image(image):
    """A source picture as file bytes: a file keeps its own bytes, a PIL image (the one the model received) is stored as a lossless PNG."""
    if isinstance(image, (str, os.PathLike)):
        return Path(image).read_bytes()
    import io
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def save_mesh_metadata(path, settings, images=None):
    """images: the source pictures by setting name ({"image_refs": [...]}, the Embed Source Images option), stored in a GLB only."""
    if is_splat_file(path):
        from shared.mesh3d.splats import ply_write_comments
        data = ply_write_comments(Path(path).read_bytes(), [f"{SETTINGS_KEY} {json.dumps(settings)}"])
    else:
        from shared.mesh3d.glb import glb_add_files, glb_read_extras, glb_write_extras
        data = glb_write_extras(Path(path).read_bytes(), {**glb_read_extras(path), SETTINGS_KEY: settings})
        files = {name: [_encoded_image(image) for image in (value if isinstance(value, list) else [value]) if image is not None] for name, value in (images or {}).items()}
        data = glb_add_files(data, files, SOURCE_IMAGES_KEY)
    Path(path).write_bytes(data)


def extract_mesh_source_images(path, output_dir=None):
    """The source pictures embedded in a GLB, written as files: {setting name: [paths]} as video_metadata.extract_source_images, {}
    without any."""
    if Path(path).suffix.lower() != ".glb":
        return {}
    import tempfile
    from shared.mesh3d.glb import glb_read_files
    files = glb_read_files(path, SOURCE_IMAGES_KEY)
    if not files:
        return {}
    output_dir = output_dir or tempfile.mkdtemp()
    os.makedirs(output_dir, exist_ok=True)
    extracted = {}
    for name, contents in files.items():
        for index, content in enumerate(contents):
            suffix = ".png" if content.startswith(b"\x89PNG") else ".jpg"
            target = Path(output_dir) / f"{Path(path).stem}_{name}_{index}{suffix}"
            target.write_bytes(content)
            extracted.setdefault(name, []).append(str(target))
    return extracted


def read_mesh_metadata(path):
    if Path(path).suffix.lower() not in (".glb", ".gltf", ".ply"):
        return None  # OBJ and STL carry no settings
    if is_splat_file(path):
        from shared.mesh3d.splats import ply_comments
        prefix = SETTINGS_KEY + " "
        settings = next((json.loads(comment[len(prefix):]) for comment in ply_comments(path) if comment.startswith(prefix)), None)
    else:
        from shared.mesh3d.glb import glb_read_extras
        settings = glb_read_extras(path).get(SETTINGS_KEY)
    return settings if isinstance(settings, dict) else None


def save_preview(preview, path):
    """Writes a [C, H, W] preview in [-1, 1] (C = 3 or 4) as the PNG preview of the asset at `path`."""
    from PIL import Image
    array = ((preview.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(array).save(preview_path(path))  # RGBA or RGB from the channel count


def ensure_preview(path):
    """Renders the preview of a 3D asset that has none (imported assets)."""
    if os.path.isfile(preview_path(path)):
        return
    if is_splat_file(path):
        from shared.mesh3d.render import splat_preview
        from shared.mesh3d.splats import ply_to_splats
        save_preview(splat_preview(ply_to_splats(path)), path)
    else:
        from shared.mesh3d.glb import load_mesh
        from shared.mesh3d.render import preview
        save_preview(preview(load_mesh(path)), path)


def import_mesh_file(source, output_dir, filename=None):
    """Copies a 3D file into the 3D gallery folder (splat PLYs as they are, other meshes converted to GLB) and makes sure
    it has a preview. Returns (path, reused) like persist_gallery_import."""
    from shared.mesh3d.splats import is_splat_ply
    source = str(getattr(source, "path", source))
    suffix = Path(source).suffix.lower()
    keep = suffix == ".glb" or suffix == ".ply" and is_splat_ply(source)
    name = Path(filename or source).stem + (suffix if keep else ".glb")
    if keep:
        path, reused = persist_gallery_import(source, output_dir, filename=name)
    else:
        import tempfile
        from shared.mesh3d.glb import load_mesh, mesh_to_glb
        with tempfile.TemporaryDirectory() as temporary:
            converted = Path(temporary) / name
            converted.write_bytes(mesh_to_glb(load_mesh(source)))
            path, reused = persist_gallery_import(converted, output_dir, filename=name)
    ensure_preview(path)
    return path, reused
