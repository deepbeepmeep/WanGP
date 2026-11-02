# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Hunyuan3D 2.1 Paint as a Texturer of meshes (temporal slot, media 3d): PBR textures (base colour, metallic, roughness) painted from the
object image with multi-view diffusion, with WanGP's hunyuan3d_21_paint model type (models/hunyuan3d). During a generation the image is the
generation's reference; in late post-processing the user supplies it."""

from __future__ import annotations

from typing import Any

from postprocessing.mesh3d_models import MethodValueMixin, ModelRuntime
from postprocessing.temporal_upsamplers import CATEGORY_TEXTURER, MEDIA_3D, PARAMETER_UI_LATE_POSTPROCESSING, PARAMETER_UI_POSTPROCESSING
from shared.utils.mesh_postprocessing import pipeline_callbacks, reference_image as object_image

METHOD = "hunyuan3d_paint"
MODEL_TYPE = "hunyuan3d_21_paint"
NAME = "Hunyuan3D 2.1 Paint"
_BOTH = (PARAMETER_UI_POSTPROCESSING, PARAMETER_UI_LATE_POSTPROCESSING)


class Hunyuan3DPaintTexturer(MethodValueMixin):
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
            "config_key": "hunyuan3d_paint",
            "pos": 20,
            "methods": [("Hunyuan3D 2.1 Paint", METHOD)],
            "description": "Paints PBR textures (base colour, metallic, roughness) on the mesh from the object image: views of the mesh are generated around it "
                           "and baked into a UV atlas. Colours follow the image, fine print comes out blurred. Robust to a mesh that does not match the image exactly.",
            "method_parameters": {METHOD: [
                {"name": "texture_views", "setting": "texture_views", "type": "string", "component": "dropdown", "ui": _BOTH, "required": False, "default": "8_768",
                 "choices": [("6 Views at 512 px (fast)", "6_512"), ("8 Views at 768 px", "8_768"), ("9 Views at 768 px", "9_768")], "enum": ["6_512", "8_768", "9_768"],
                 "label": "Texture Views", "description": "Views generated to paint the texture: more views and 768 px cover the object better and look sharper."},
                {"name": "texture_size", "setting": "texture_size", "type": "integer", "component": "dropdown", "ui": _BOTH, "required": False, "default": 2048,
                 "choices": [("1024", 1024), ("2048", 2048), ("4096", 4096)], "enum": [1024, 2048, 4096], "label": "Texture Size", "description": "Size of the PBR texture images."},
                {"name": "texture_max_faces", "setting": "max_faces", "type": "integer", "component": "slider", "ui": _BOTH, "required": False, "default": 0,
                 "minimum": 0, "maximum": 4000000, "step": 1000, "label": "Textured Max Faces (0 = keep all)",
                 "description": "Face count the mesh is reduced to before texturing; 0 keeps every face (a refined mesh keeps its detail, unwrapping takes longer)."},
                {"name": "texture_guidance", "setting": "guidance_scale", "type": "number", "component": "slider", "ui": _BOTH, "required": False, "default": 3.0,
                 "minimum": 1.0, "maximum": 10.0, "step": 0.5, "label": "Texture Guidance", "description": "Guidance of the generated texture views (3.0 upstream)."},
                {"name": "reference_images", "setting": "reference_image", "type": "array", "component": "images", "ui": (PARAMETER_UI_LATE_POSTPROCESSING,), "required": True,
                 "default": [], "label": "Object Image", "description": "Picture of the object the mesh represents (its first image is used; a transparent background is best).", "multiple": True, "media_type": "image"},
            ]},
        }

    def enabled(self) -> bool:
        return True

    def validate_upsampling(self, temporal_upsampling, *, source_is_image: bool = False) -> str:
        return "" if self.split_value(temporal_upsampling) is not None else "Invalid Hunyuan3D Paint selection"

    def supports_loaded_model(self, temporal_upsampling, context, **_kwargs) -> bool:
        return context.base_model_type == MODEL_TYPE

    def texture(self, mesh, temporal_upsampling, *, loaded_model_context=None, reference_image=None, texture_views="8_768", texture_size=2048, max_faces=0,
                guidance_scale=3.0, seed=0, abort_callback=None, progress_callback=None, **_kwargs):
        image = object_image(reference_image)
        if image is None:
            raise ValueError("Hunyuan3D Paint needs the Object Image of the mesh")
        model = self.runtime.model(loaded_model_context)
        callback, set_status = pipeline_callbacks(model, abort_callback, progress_callback)
        return self.runtime.run(model.texture_mesh, mesh, image, int(seed), float(guidance_scale), int(max_faces), texture_views, int(texture_size), callback=callback,
                                set_progress_status=set_status)

    def release_vram(self) -> None:
        self.runtime.release()


__all__ = ["Hunyuan3DPaintTexturer"]
