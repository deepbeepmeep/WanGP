# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from models import goals_3d
from preprocessing.rembg.assets import OBJECT_MODEL

from .prompts import (SHAPE_DEEPY_INFOS, SHAPE_INFOS, TEXTURE_DEEPY_INFOS, TEXTURE_INFOS, TRELLIS2_DEEPY_PROMPT_INFOS, TRELLIS2_PROMPT_INFOS)
from .settings import CUSTOM_SETTINGS, MODEL_MODES, TEXTURE_CUSTOM_SETTINGS, TEXTURE_MODEL_MODES

SHAPE_ARCH = "trellis2_shape"
TEXTURE_ARCH = "trellis2_texture"
# TRELLIS.2 4B (shape and texture in one model type), split into the Shape model + the TRELLIS.2 Texture Texturer (2026-10-29)
RETIRED_ARCH = "trellis2"
TEXTURER = "trellis2_texture"  # its method in the temporal slot (postprocessing/trellis2_texture)
REPO = "DeepBeepMeep/3D"
MESH_FILE_TYPES = [".glb", ".obj", ".ply", ".stl"]


class family_handler:
    @staticmethod
    def query_supported_types():
        return [SHAPE_ARCH, TEXTURE_ARCH]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_retired_model_types():
        return {RETIRED_ARCH: SHAPE_ARCH}

    @staticmethod
    def convert_retired_settings(model_type, settings):
        """Settings of TRELLIS.2 4B: the Shape model, with the TRELLIS.2 Texture Texturer (its texture settings in the temporal slot) unless
        the surface was Shape Only or another Texturer is already selected. Returns the new model type."""
        custom = dict(settings.get("custom_settings") or {})
        texture_mode, texture_size = custom.pop("texture_mode", "texture"), custom.pop("texture_size", 2048)
        custom.pop("alpha_mode", None)
        texture_guidance = settings.pop("alt_guidance_scale", 1.0)
        settings["custom_settings"] = custom
        if texture_mode != "none" and not settings.get("temporal_upsampling"):
            settings.update(temporal_upsampling=TEXTURER, temporal_postprocessing_parameters={"texture_resolution": 512 if settings.get("model_mode") == 0 else 1024,
                            "texture_size": texture_size, "texture_max_faces": 0, "texture_guidance": texture_guidance})
        return SHAPE_ARCH

    @staticmethod
    def query_model_family():
        return "trellis2"

    @staticmethod
    def query_family_infos():
        return {"trellis2": (3020, "TRELLIS.2"), **goals_3d.family_infos(goals_3d.IMAGE_TO_3D, goals_3d.TEXTURE_MESH)}

    @staticmethod
    def get_lora_dir(base_model_type):
        return base_model_type

    @staticmethod
    def query_model_def(base_model_type, model_def):
        texture = base_model_type == TEXTURE_ARCH
        extra_model_def = {
            "group": goals_3d.TEXTURE_MESH if texture else goals_3d.IMAGE_TO_3D,
            "image_outputs": True,
            "mesh_outputs": True,
            "mesh_texture": texture,
            "device_explicit": True,
            "dtype": "bf16",
            "image_ref_choices": {"choices": [("Object Image", "I")], "letters_filter": "I", "default": "I"},
            "one_image_ref_needed": True,
            "one_image_ref_only": True,
            "image_refs_keep_alpha": True,
            "background_removal_model": OBJECT_MODEL,  # BiRefNet: no cast shadows in the object cutout
            "fit_into_canvas_image_refs": 0,
            "prompt_description": "Asset Name / Description (optional, used for file names)",
            "prompt_optional": True,  # the prompt only names the output files
            "no_negative_prompt": True,
            "embedded_guidance": False,
            "inference_steps": True,
            "guidance_max_phases": 1,
            "prompt_enhancer_def": {"selection": [], "labels": {}, "default": ""},
            "profiles_dir": [base_model_type],
            "prompt_infos": TRELLIS2_PROMPT_INFOS,
            "deepy_prompt_infos": TRELLIS2_DEEPY_PROMPT_INFOS,
        }
        if texture:
            extra_model_def.update({
                "custom_guide": {"id": "custom_guide", "name": "Mesh", "label": "Input Mesh (.glb/.obj/.ply/.stl)", "type": "file", "default": None, "required": True,
                                 "file_types": MESH_FILE_TYPES},
                "model_modes": {**TEXTURE_MODEL_MODES, "choices": list(TEXTURE_MODEL_MODES["choices"])},
                "custom_settings": [one.copy() for one in TEXTURE_CUSTOM_SETTINGS],
                "infos": TEXTURE_INFOS,
                "deepy_infos": TEXTURE_DEEPY_INFOS,
                "specialities": [{"name": "approximate PBR texturing", "description": "Generates PBR textures for a supplied mesh whose colours only approximate "
                                  "the picture: the least faithful Texturer, SUPRA-Tex and Hunyuan3D 2.1 Paint follow the picture much more closely."}],
            })
        else:
            extra_model_def.update({
                "model_modes": {**MODEL_MODES, "choices": list(MODEL_MODES["choices"])},
                "custom_settings": [one.copy() for one in CUSTOM_SETTINGS],
                "preset_profiles_dir": ["trellis2_shape_presets"],  # recipes: this shape with a 3D Refiner or a Texturer
                "infos": SHAPE_INFOS,
                "deepy_infos": SHAPE_DEEPY_INFOS,
                "specialities": [{"name": "image to 3D", "aliases": ["3D mesh", "GLB", "3D asset"],
                                  "description": "Turns one picture of an object into a 3D mesh, keeping thin and intricate structures."}],
            })
        return extra_model_def

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        decoders = ["trellis2_shape_encoder_fp16.safetensors", "trellis2_texture_decoder_fp16.safetensors"] if base_model_type == TEXTURE_ARCH else \
            ["trellis2_ss_decoder_fp16.safetensors", "trellis2_shape_decoder_fp16.safetensors"]
        return [{"repoId": REPO, "sourceFolderList": ["trellis2", "dinov3_vitl16"], "fileList": [decoders, ["dinov3_vitl16_lvd1689m.safetensors", "LICENSE_DINOv3.md"]]}]

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, save_quantized=False, **kwargs):
        from .trellis2_main import model_factory

        model = model_factory(model_filename, model_type, base_model_type, save_quantized=save_quantized)
        return model, model.pipe()

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        texture = base_model_type == TEXTURE_ARCH
        ui_defaults.update({
            "image_mode": 1,
            "video_prompt_type": "I",
            "num_inference_steps": 12,
            "guidance_scale": 1.0 if texture else 7.5,  # the texture model's guidance is the texture guidance (1 = off, the official setting)
            "model_mode": (TEXTURE_MODEL_MODES if texture else MODEL_MODES)["default"],
            "batch_size": 1,
            "remove_background_images_ref": 1,
        })
