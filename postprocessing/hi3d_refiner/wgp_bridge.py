# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Hi3D v1.0 as the 3D Refiner of meshes (spatial slot, media 3d): it adds the detail of the object image to a coarse mesh, in the mesh's
own frame, with WanGP's hi3d model type (models/hi3d). During a generation the image is the generation's reference; in late
post-processing the user supplies it."""

from __future__ import annotations

from typing import Any

from postprocessing.mesh3d_models import ModelRuntime
from postprocessing.spatial_upsamplers import (PARAMETER_UI_LATE_POSTPROCESSING, PARAMETER_UI_POSTPROCESSING, POSTPROCESSING_CATEGORY_REFINER, SimpleScaleSuffixMixin,
                                               UPSAMPLER_PROFILE_3D, UPSAMPLER_PROFILE_IMAGE, UPSAMPLER_TYPE_POSTPROCESSING)
from shared.utils.mesh_postprocessing import pipeline_callbacks, reference_image as object_image

METHOD = "hi3d"
MODEL_TYPE = "hi3d"
NAME = "Hi3D v1.0"
_BOTH = (PARAMETER_UI_POSTPROCESSING, PARAMETER_UI_LATE_POSTPROCESSING)


class Hi3DRefinerBridge(SimpleScaleSuffixMixin):
    def __init__(self, server_config: dict[str, Any] | None = None, files_locator=None):
        self.runtime = ModelRuntime(NAME, MODEL_TYPE)

    @classmethod
    def query_upsampler_def(cls) -> dict[str, Any]:
        return {
            "name": NAME,
            "upsampler_types": (UPSAMPLER_TYPE_POSTPROCESSING,),
            "media": (UPSAMPLER_PROFILE_3D,),
            "profile": UPSAMPLER_PROFILE_IMAGE,
            "config_key": "hi3d_refiner",
            "pos": 60,
            "methods": [("Hi3D Refiner", METHOD)],
            "vae_methods": [],
            "default_spatial_upsampling": METHOD,
            "postprocessing_category": POSTPROCESSING_CATEGORY_REFINER,
            "progress_label": "Refining Geometry",
            "description": "Adds the fine geometric detail of the object image (tufting, scales, teeth, panel lines) to a coarse mesh, keeping its position, "
                           "orientation and scale. Its detail goes to solid parts: thin sheets such as leaves or fabric keep the coarse surface. It replaces the geometry, "
                           "so the result is untextured: add a Texturer to paint it.",
            "media_descriptions": {UPSAMPLER_PROFILE_3D: "Hi3D v1.0 refines the mesh in passes up to the chosen level: 1024 suits most objects, 1536 adds a little more "
                                                         "at about three times the time. The mesh and the image must show the same object, facing the same way."},
            "method_parameters": {METHOD: [
                {"name": "hi3d_level", "setting": "level", "type": "integer", "component": "dropdown", "ui": _BOTH, "required": False, "default": 1024,
                 "choices": [("512 (Fast)", 512), ("1024 (Recommended)", 1024), ("1536 (Finest, Slowest)", 1536)], "enum": [512, 1024, 1536], "label": "Refinement Level",
                 "description": "Resolution of the last refinement pass: higher levels add finer detail and take longer (about 12 s, 50 s and 2 min on a fast GPU)."},
                {"name": "hi3d_max_faces", "setting": "max_faces", "type": "integer", "component": "slider", "ui": _BOTH, "required": False, "default": 0,
                 "minimum": 0, "maximum": 4000000, "step": 50000, "label": "Refined Max Faces (0 = 1M, 2M at 1536)", "description": "Face count the refined mesh is reduced to; 0 selects 1 million faces, 2 million at 1536."},
                {"name": "reference_images", "setting": "reference_image", "type": "array", "component": "images", "ui": (PARAMETER_UI_LATE_POSTPROCESSING,), "required": True,
                 "default": [], "label": "Object Image", "description": "Picture of the object the mesh represents (its first image is used; a transparent background is best).",
                 "multiple": True, "media_type": "image"},
            ]},
        }

    def enabled(self) -> bool:
        return True

    def validate_upsampling(self, spatial_upsampling, image_mode: int) -> str:
        split = self.split_value(spatial_upsampling)
        return "" if split is not None and split[0] == METHOD else "Invalid Hi3D Refiner selection"

    def supports_loaded_model(self, spatial_upsampling, context, **_kwargs) -> bool:
        return context.base_model_type == MODEL_TYPE

    def upscale(self, mesh, spatial_upsampling, *, loaded_model_context=None, reference_image=None, level=1024, max_faces=0, seed=0, abort_callback=None,
                progress_callback=None, **_kwargs):
        image = object_image(reference_image)
        if image is None:
            raise ValueError("The Hi3D Refiner needs the Object Image of the mesh")
        model = self.runtime.model(loaded_model_context)
        callback, set_status = pipeline_callbacks(model, abort_callback, progress_callback)
        return self.runtime.run(model.refine_mesh, mesh, image, int(level), int(max_faces), 25, int(seed), callback=callback, set_progress_status=set_status)

    def release_vram(self) -> None:
        self.runtime.release()


__all__ = ["Hi3DRefinerBridge"]
