# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Post-processing of 3D assets: the 3D refiner of the spatial slot (``spatial_upsampling``), then the texturer of the temporal slot
(``temporal_upsampling``), both discovered from the registered processors whose definition lists the media kind ``3d``. They run
on the meshes of a generation, with its reference image, or on a gallery mesh (late post-processing), with the image the user
supplies in the reference images input shared by both slots (postprocessing_reference_images).

Model definitions describe their 3D outputs:
- ``mesh_texture``: True, or a callable of the generation inputs, when the generated meshes carry a texture. A texturer is then
  redundant, and a refiner would discard the texture: validation refuses both.
- ``mesh_postprocessing: False``: the outputs are not static meshes (Gaussian splats, animated or segmented meshes)."""

from __future__ import annotations

import torch

from postprocessing import parameter_ui
from postprocessing import spatial_upsamplers as upsampler_api
from postprocessing import temporal_upsamplers as temporal_upsampler_api

MEDIA_3D = "3d"


def selected(spatial_upsampling, temporal_upsampling) -> bool:
    return bool(str(spatial_upsampling or "").strip() or str(temporal_upsampling or "").strip())


def mesh_texture(model_def, inputs) -> bool:
    texture = model_def.get("mesh_texture", False)
    return bool(texture(inputs) if callable(texture) else texture)


def _label(value, temporal: bool) -> str:
    return (temporal_upsampler_api.format_temporal_upsampling_label(value) if temporal else upsampler_api.format_upsampling_label(value)) or str(value)


def validate_processors(spatial_upsampling, temporal_upsampling) -> str:
    """The processors must be registered and process 3D assets."""
    return (upsampler_api.validate_postprocessing_spatial_upsampling(str(spatial_upsampling or ""), 0, media_kind=MEDIA_3D)
            or temporal_upsampler_api.validate_temporal_upsampling(temporal_upsampling, media_kind=MEDIA_3D))


# asset.extras key of a mesh generated in the frame of its object image (Pixal3D): that image's camera ({"camera_angle_x", "distance"}),
# which a texturer projecting the image (SUPRA-Tex) uses instead of fitting one; a refiner keeps the mesh's frame, so it stays valid
OBJECT_CAMERA = "wangp_object_camera"


def validate(model_def, inputs, spatial_upsampling, temporal_upsampling) -> str:
    """Error text for the 3D post-processing of a generation with these inputs, "" when it is valid."""
    if not selected(spatial_upsampling, temporal_upsampling):
        return ""
    if model_def.get("mesh_postprocessing", True) is False:
        return "The 3D outputs of this model are not static meshes: they can not go through a 3D Refiner or a Texturer."
    error = validate_processors(spatial_upsampling, temporal_upsampling)
    if error:
        return error
    if mesh_texture(model_def, inputs):
        if str(temporal_upsampling or "").strip():
            return (f"The Texturer '{_label(temporal_upsampling, True)}' is redundant: this generation already textures its meshes. Select the "
                    "model's untextured mode (for instance Shape Only) to texture them with a Texturer.")
        return (f"The 3D Refiner '{_label(spatial_upsampling, False)}' would discard the texture this generation paints. Select the model's "
                "untextured mode (for instance Shape Only), then a Texturer to paint the refined mesh.")
    return ""


def process_mesh(mesh, spatial_upsampling, temporal_upsampling, *, spatial_values=None, temporal_values=None, spatial_kwargs=None, temporal_kwargs=None, **common):
    """The mesh refined by the 3D refiner, then textured by the texturer (either may be absent). spatial_values / temporal_values: the
    slots' runtime values (parameter_ui.runtime_values); common: the arguments of both processors (reference_image: the object image, PIL
    RGBA preferred, unless the parameters supply one;
    seed, abort_callback, progress_callback, main_offloadobj, loaded_model_context, process_files); spatial_kwargs / temporal_kwargs: those
    of one slot (init_pipe of its memory policy)."""
    if str(spatial_upsampling or "").strip():
        handler = upsampler_api.find_postprocessing_upsampler(spatial_upsampling)
        mesh = upsampler_api.upscale_postprocessing(handler, mesh, spatial_upsampling, parameter_values=spatial_values, **common, **dict(spatial_kwargs or {}))
    if str(temporal_upsampling or "").strip():
        mesh = temporal_upsampler_api.texture_mesh(temporal_upsampling, mesh, parameter_values=temporal_values, **common, **dict(temporal_kwargs or {}))
    return mesh


def process_glb(data: bytes, spatial_upsampling, temporal_upsampling, **kwargs) -> tuple[bytes, torch.Tensor]:
    """A GLB (bytes) through process_mesh: the processed GLB, keeping the source's extras, and its [C, H, W] preview. The processors
    receive the source's OBJECT_CAMERA (object_camera)."""
    from shared.mesh3d import glb_read_extras, glb_to_mesh, preview

    extras = glb_read_extras(data)
    mesh = process_mesh(glb_to_mesh(data), spatial_upsampling, temporal_upsampling, object_camera=extras.get(OBJECT_CAMERA), **kwargs)
    return _processed_glb(mesh, data, extras), preview(mesh).cpu()


def _processed_glb(mesh, source, extras) -> bytes:
    """The GLB of a processed mesh with the extras of its GLB source and the source pictures embedded there (they index its buffer)."""
    from shared.mesh3d.glb import glb_add_files, glb_read_files, mesh_to_glb
    from shared.utils.mesh_outputs import SOURCE_IMAGES_KEY

    files = glb_read_files(source, SOURCE_IMAGES_KEY) if SOURCE_IMAGES_KEY in extras else {}
    return glb_add_files(mesh_to_glb(mesh.to("cpu"), extras={key: value for key, value in extras.items() if key != SOURCE_IMAGES_KEY}), files, SOURCE_IMAGES_KEY)


def is_mesh_file(path) -> bool:
    """A static mesh file of the 3D gallery (GLB, glTF, OBJ, STL); PLY files there are Gaussian splats."""
    from shared.mesh3d.glb import MESH_EXTENSIONS
    from shared.utils.mesh_outputs import is_splat_file
    return str(path).lower().endswith(MESH_EXTENSIONS) and not is_splat_file(path)


def embedded_object_images(path, extract=False):
    """The object images embedded in a gallery GLB (the Embed Source Images option): whether there are any (reading only the GLB's JSON
    chunk), or with extract their files."""
    if not str(path or "").lower().endswith(".glb"):
        return [] if extract else False
    from shared.utils.mesh_outputs import SOURCE_IMAGES_KEY, extract_mesh_source_images
    if extract:
        return extract_mesh_source_images(path).get("image_refs", [])
    from shared.mesh3d import glb_read_extras
    return bool((glb_read_extras(path).get(SOURCE_IMAGES_KEY) or {}).get("image_refs"))


def validate_late(spatial_upsampling, temporal_upsampling, spatial_values, temporal_values, source=None) -> str:
    """Late post-processing of a gallery mesh: at least one 3D processor, and the parameters they require in that context (such as the
    reference image, unless the GLB source embeds its object image), from the slots' runtime values (parameter_ui.runtime_values)."""
    if not selected(spatial_upsampling, temporal_upsampling):
        return "You must choose a 3D Refiner or a Texturer"
    error = validate_processors(spatial_upsampling, temporal_upsampling)
    if error:
        return error
    checks = []
    if str(spatial_upsampling or "").strip():
        method = upsampler_api.split_upsampling_value(spatial_upsampling)[0]
        checks += [(parameter, spatial_values) for parameter in upsampler_api.method_parameters(method, ui_context=upsampler_api.PARAMETER_UI_LATE_POSTPROCESSING)]
    if str(temporal_upsampling or "").strip():
        method = temporal_upsampler_api.split_temporal_upsampling_value(temporal_upsampling)[0]
        checks += [(parameter, temporal_values) for parameter in temporal_upsampler_api.method_parameters(method, ui_context=temporal_upsampler_api.PARAMETER_UI_LATE_POSTPROCESSING)]
    for parameter, values in checks:
        value = parameter_ui.parameter_value(parameter, values)
        if parameter.get("required", False) and (value is None or value == "" or value == []):
            if parameter_ui.parameter_component_type(parameter) == "images" and embedded_object_images(source):
                continue
            return f"{parameter.get('label', parameter['name'])} is required"
    return ""


def process_file(path, spatial_upsampling, temporal_upsampling, **kwargs) -> tuple[bytes, torch.Tensor]:
    """A mesh file through process_mesh: the processed GLB (keeping a GLB source's extras) and its [C, H, W] preview."""
    from shared.mesh3d import glb_read_extras, load_mesh, preview

    extras = glb_read_extras(path) if str(path).lower().endswith(".glb") else {}
    mesh = process_mesh(load_mesh(path), spatial_upsampling, temporal_upsampling, object_camera=extras.get(OBJECT_CAMERA), **kwargs)
    return _processed_glb(mesh, path, extras), preview(mesh).cpu()


def reference_image(value):
    """The object image a 3D processor conditions on, from a PIL image, a file path, a (image, caption) gallery entry or a list of
    those (the first one); None when absent. Its alpha is kept."""
    from PIL import Image

    if isinstance(value, (list, tuple)) and value and not isinstance(value[0], (int, float)):
        if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], (str, type(None))) and not isinstance(value[0], (list, tuple)):
            value = value[0]  # (image, caption)
        else:
            return reference_image(value[0])
    if value is None or value == [] or value == "":
        return None
    if isinstance(value, Image.Image):
        return value
    return Image.open(value)


def pipeline_callbacks(pipeline, abort_callback=None, progress_callback=None):
    """(callback, set_progress_status) that drive the progress and abort of a WanGP pipeline method wrapped by generation_progress from
    a post-processor's abort_callback() and progress_callback(phase, current, total, unit)."""
    pipeline._interrupt = False

    def callback(step=-1, latents=None, force_refresh=False, override_num_inference_steps=None, progress_title=None, progress_unit=None, **_kwargs):
        if progress_callback is not None and progress_title is not None:
            progress_callback(progress_title, max(0, int(step) + 1) if override_num_inference_steps else None, override_num_inference_steps, progress_unit)

    def checkpoint():
        if abort_callback is not None and abort_callback():
            pipeline._interrupt = True

    def set_status(text):
        if progress_callback is not None:
            progress_callback(text)

    callback.checkpoint = checkpoint
    return callback, set_status
