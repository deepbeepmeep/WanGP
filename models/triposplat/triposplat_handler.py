# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from models import goals_3d
from preprocessing.rembg.assets import OBJECT_MODEL

from .prompts import ASSET_NAME_INSTRUCTIONS, TRIPOSPLAT_DEEPY_INFOS, TRIPOSPLAT_DEEPY_PROMPT_INFOS, TRIPOSPLAT_INFOS, TRIPOSPLAT_PROMPT_INFOS
from .settings import CUSTOM_SETTINGS

ARCHITECTURE = "triposplat"
REPO = "DeepBeepMeep/3D"


class family_handler:
    @staticmethod
    def query_supported_types():
        return [ARCHITECTURE]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "triposplat"

    @staticmethod
    def query_family_infos():
        return {"triposplat": (3030, "TripoSplat"), **goals_3d.family_infos(goals_3d.IMAGE_TO_3D)}

    @staticmethod
    def get_lora_dir(base_model_type):
        return base_model_type

    @staticmethod
    def query_model_def(base_model_type, model_def):
        return {
            "group": goals_3d.IMAGE_TO_3D,
            "image_outputs": True,
            "mesh_outputs": True,
            "mesh_postprocessing": False,  # Gaussian splats
            "device_explicit": True,
            "dtype": "fp16",
            "image_ref_choices": {"choices": [("Object Image", "I")], "letters_filter": "I", "default": "I"},
            "one_image_ref_needed": True,
            "one_image_ref_only": True,
            "image_refs_keep_alpha": True,
            "background_removal_model": OBJECT_MODEL,  # BiRefNet: no cast shadows in the object cutout
            "prompt_description": "Asset Name / Description (optional, used for file names)",
            "prompt_optional": True,  # the prompt only names the output files
            "no_negative_prompt": True,
            "guidance_max_phases": 1,
            "embedded_guidance": False,
            "NAG": False,
            "inference_steps": True,
            "flow_shift": True,
            "profiles_dir": [base_model_type],
            "custom_settings": [one.copy() for one in CUSTOM_SETTINGS],
            "prompt_enhancer_def": {"selection": ["TI"], "labels": {"TI": "An Asset Name from the {image_inputs}"}, "default": ""},
            "image_prompt_enhancer_instructions": ASSET_NAME_INSTRUCTIONS,
            "image_prompt_enhancer_max_tokens": 32,
            "infos": TRIPOSPLAT_INFOS,
            "prompt_infos": TRIPOSPLAT_PROMPT_INFOS,
            "deepy_infos": TRIPOSPLAT_DEEPY_INFOS,
            "deepy_prompt_infos": TRIPOSPLAT_DEEPY_PROMPT_INFOS,
            "specialities": [{"name": "image to 3D Gaussian splat", "aliases": ["3D Gaussian splatting", "3DGS", "splat", "PLY splat"],
                              "description": "Turns one picture of an object into a 3D Gaussian splat (PLY) for photographic 3D viewing."}],
        }

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return [
            {"repoId": REPO, "sourceFolderList": ["triposplat", "dinov3_vith16plus"],
             "fileList": [["triposplat_vae_decoder_fp16.safetensors"], ["dinov3_vith16plus_lvd1689m_bf16.safetensors", "LICENSE_DINOv3.md"]]},
            {"repoId": "DeepBeepMeep/Flux2", "sourceFolderList": [""], "fileList": [["flux2_vae.safetensors"]]},
        ]

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, save_quantized=False, **kwargs):
        from .triposplat_main import model_factory

        model = model_factory(model_filename, model_type, base_model_type, save_quantized=save_quantized)
        return model, model.pipe()

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({
            "image_mode": 1,
            "video_prompt_type": "I",
            "num_inference_steps": 20,
            "guidance_scale": 3.0,
            "flow_shift": 3.0,
            "batch_size": 1,
            "remove_background_images_ref": 1,
            "prompt_enhancer": "",
        })
