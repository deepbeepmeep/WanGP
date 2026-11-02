# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""TRELLIS.2 Texture as a Texturer of meshes (temporal slot, media 3d): PBR textures (base colour, metallic, roughness) generated in 3D from
the object image by TRELLIS.2's texture flow, with WanGP's trellis2_texture model type (models/trellis2). During a generation the image is the
generation's reference; in late post-processing the user supplies it."""

from __future__ import annotations

from typing import Any

from postprocessing.mesh3d_models import MethodValueMixin, ModelRuntime
from postprocessing.temporal_upsamplers import CATEGORY_TEXTURER, MEDIA_3D, PARAMETER_UI_LATE_POSTPROCESSING, PARAMETER_UI_POSTPROCESSING
from shared.utils.mesh_postprocessing import pipeline_callbacks, reference_image as object_image

METHOD = "trellis2_texture"
MODEL_TYPE = "trellis2_texture"
NAME = "TRELLIS.2 Texture"
_BOTH = (PARAMETER_UI_POSTPROCESSING, PARAMETER_UI_LATE_POSTPROCESSING)


class Trellis2Texturer(MethodValueMixin):
    def __init__(self, server_config: dict[str, Any] | None = None, files_locator=None):
        self.runtime = ModelRuntime(NAME, MODEL_TYPE)

    @staticmethod
    def persistent_models() -> bool:
        return True  # kept between generations like a 3D Refiner: MMGP frees the RAM of an idle model under pressure

    @classmethod
    def query_temporal_upsampler_def(cls) -> dict[str, Any]:
        return {
            "name": NAME,
            "media": (MEDIA_3D,),
            "postprocessing_category": CATEGORY_TEXTURER,
            "config_key": "trellis2_texture",
            "pos": 30,
            "methods": [("TRELLIS.2 Texture", METHOD)],
            "description": "Generates PBR textures (base colour, metallic, roughness) all around the mesh from the object image, in 3D. Several times faster "
                           "than the other Texturers, but the least faithful: colours are approximate and change with the seed, print or patterns are not reproduced.",
            "method_parameters": {METHOD: [
                {"name": "texture_resolution", "setting": "resolution", "type": "integer", "component": "dropdown", "ui": _BOTH, "required": False, "default": 1024,
                 "choices": [("512 (Faster)", 512), ("1024", 1024)], "enum": [512, 1024], "label": "Texture Voxels",
                 "description": "Resolution of the voxels the texture is generated in: 1024 gives finer colour detail."},
                {"name": "texture_size", "setting": "texture_size", "type": "integer", "component": "dropdown", "ui": _BOTH, "required": False, "default": 2048,
                 "choices": [("1024", 1024), ("2048", 2048), ("4096", 4096)], "enum": [1024, 2048, 4096], "label": "Texture Size", "description": "Size of the PBR texture images."},
                {"name": "texture_max_faces", "setting": "max_faces", "type": "integer", "component": "slider", "ui": _BOTH, "required": False, "default": 0,
                 "minimum": 0, "maximum": 4000000, "step": 50000, "label": "Textured Max Faces (0 = keep all)",
                 "description": "Face count the mesh is reduced to before texturing; 0 keeps every face (a refined mesh keeps its detail, unwrapping takes longer)."},
                {"name": "texture_guidance", "setting": "guidance_scale", "type": "number", "component": "slider", "ui": _BOTH, "required": False, "default": 1.0,
                 "minimum": 1.0, "maximum": 10.0, "step": 0.5, "label": "Texture Guidance",
                 "description": "1 = off (the official setting); higher values follow the image more strongly but can saturate the colours."},
                {"name": "reference_images", "setting": "reference_image", "type": "array", "component": "images", "ui": (PARAMETER_UI_LATE_POSTPROCESSING,), "required": True,
                 "default": [], "label": "Object Image", "description": "Picture of the object the mesh represents (its first image is used; a transparent background is best).",
                 "multiple": True, "media_type": "image"},
            ]},
        }

    def enabled(self) -> bool:
        return True

    def validate_upsampling(self, temporal_upsampling, *, source_is_image: bool = False) -> str:
        return "" if self.split_value(temporal_upsampling) is not None else "Invalid TRELLIS.2 Texture selection"

    def supports_loaded_model(self, temporal_upsampling, context, **_kwargs) -> bool:
        return context.base_model_type == MODEL_TYPE

    def texture(self, mesh, temporal_upsampling, *, loaded_model_context=None, reference_image=None, resolution=1024, texture_size=2048, max_faces=0,
                guidance_scale=1.0, seed=0, abort_callback=None, progress_callback=None, **_kwargs):
        image = object_image(reference_image)
        if image is None:
            raise ValueError("TRELLIS.2 Texture needs the Object Image of the mesh")
        model = self.runtime.model(loaded_model_context)
        callback, set_status = pipeline_callbacks(model, abort_callback, progress_callback)
        return self.runtime.run(model.texture_mesh, mesh, image, int(seed), guidance_scale=float(guidance_scale), resolution=int(resolution), max_faces=int(max_faces),
                                texture_size=int(texture_size), callback=callback, set_progress_status=set_status)

    def release_vram(self) -> None:
        self.runtime.release()


__all__ = ["Trellis2Texturer"]
