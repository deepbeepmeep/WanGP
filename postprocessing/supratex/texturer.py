# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""SUPRA-Tex as a Texturer of meshes (temporal slot, media 3d): PBR textures (base colour, metallic, roughness) whose visible side keeps
the object image's colours and print and whose hidden side is completed in 3D, with WanGP's supratex_texture model type (models/supratex).
During a generation the image is the generation's reference; in late post-processing the user supplies it."""

from __future__ import annotations

from typing import Any

from postprocessing.mesh3d_models import MethodValueMixin, ModelRuntime
from postprocessing.temporal_upsamplers import CATEGORY_TEXTURER, MEDIA_3D, PARAMETER_UI_LATE_POSTPROCESSING, PARAMETER_UI_POSTPROCESSING
from shared.utils.mesh_postprocessing import pipeline_callbacks, reference_image as object_image

METHOD = "supratex"
MODEL_TYPE = "supratex_texture"
NAME = "SUPRA-Tex"
_BOTH = (PARAMETER_UI_POSTPROCESSING, PARAMETER_UI_LATE_POSTPROCESSING)


class SupraTexTexturer(MethodValueMixin):
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
            "config_key": "supratex",
            "pos": 10,
            "methods": [("SUPRA-Tex", METHOD)],
            "description": "The most faithful Texturer when the image lines up with the mesh. Completes PBR textures (base colour, metallic, roughness) in 3D from the object image: the visible side keeps the image's colours, "
                           "print and patterns, the hidden side is completed consistently. The mesh and the image must show the same object, facing the same way. "
                           "Its material stage is non-commercial (Qwen Research License).",
            "method_parameters": {METHOD: [
                {"name": "texture_size", "setting": "texture_size", "type": "integer", "component": "dropdown", "ui": _BOTH, "required": False, "default": 2048,
                 "choices": [("1024", 1024), ("2048", 2048), ("4096", 4096)], "enum": [1024, 2048, 4096], "label": "Texture Size", "description": "Size of the PBR texture images."},
                {"name": "texture_max_faces", "setting": "max_faces", "type": "integer", "component": "slider", "ui": _BOTH, "required": False, "default": 0,
                 "minimum": 0, "maximum": 4000000, "step": 50000, "label": "Textured Max Faces (0 = keep all)",
                 "description": "Face count the mesh is reduced to before texturing; 0 keeps every face (a refined mesh keeps its detail, unwrapping takes longer)."},
                {"name": "reference_images", "setting": "reference_image", "type": "array", "component": "images", "ui": (PARAMETER_UI_LATE_POSTPROCESSING,), "required": True,
                 "default": [], "label": "Object Image", "description": "Picture of the object the mesh represents, seen from the mesh's front (its first image is used; a transparent background is best).",
                 "multiple": True, "media_type": "image"},
            ]},
        }

    def enabled(self) -> bool:
        return True

    def validate_upsampling(self, temporal_upsampling, *, source_is_image: bool = False) -> str:
        return "" if self.split_value(temporal_upsampling) is not None else "Invalid SUPRA-Tex selection"

    def supports_loaded_model(self, temporal_upsampling, context, **_kwargs) -> bool:
        return context.base_model_type == MODEL_TYPE

    def texture(self, mesh, temporal_upsampling, *, loaded_model_context=None, reference_image=None, texture_size=2048, max_faces=0, seed=0, object_camera=None,
                abort_callback=None, progress_callback=None, **_kwargs):
        """object_camera: the camera of a Pixal3D mesh (its GLB's OBJECT_CAMERA), used instead of fitting one to the image's silhouette."""
        image = object_image(reference_image)
        if image is None:
            raise ValueError("SUPRA-Tex needs the Object Image of the mesh")
        model = self.runtime.model(loaded_model_context)
        callback, set_status = pipeline_callbacks(model, abort_callback, progress_callback)
        return self.runtime.run(model.texture_mesh, mesh, image, int(seed), int(max_faces), int(texture_size), object_camera=object_camera, callback=callback,
                                set_progress_status=set_status)

    def release_vram(self) -> None:
        self.runtime.release()


__all__ = ["SupraTexTexturer"]
