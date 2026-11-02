# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from models import goals_3d

from .prompts import WORLDMIRROR_DEEPY_INFOS, WORLDMIRROR_DEEPY_PROMPT_INFOS, WORLDMIRROR_INFOS, WORLDMIRROR_PROMPT_INFOS
from .settings import CUSTOM_SETTINGS, MODEL_MODES

ARCHITECTURE = "worldmirror_2"
REPO = "DeepBeepMeep/3D"
PROJECT = "worldmirror"


class family_handler:
    @staticmethod
    def query_supported_types():
        return [ARCHITECTURE]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "worldmirror"

    @staticmethod
    def query_family_infos():
        return {"worldmirror": (3040, "WorldMirror"), **goals_3d.family_infos(goals_3d.RECONSTRUCTION)}

    @staticmethod
    def get_lora_dir(base_model_type):
        return base_model_type

    @staticmethod
    def query_model_def(base_model_type, model_def):
        return {
            "group": goals_3d.RECONSTRUCTION,
            "image_outputs": True,
            "mesh_outputs": True,
            "device_explicit": True,
            "image_ref_choices": {"choices": [("Photos", "I"), ("No Photos", "")], "letters_filter": "I", "default": "I"},
            "no_background_removal": True,
            "fit_into_canvas_image_refs": 0,
            "guide_custom_choices": {"choices": [("No Video", ""), ("Reference Video", "V")], "letters_filter": "V", "default": "", "label": "Video"},
            "reference_video_enabled": True,
            "skip_video_guide_preprocess": True,  # the pipeline picks and decodes its views from the source file
            "video_guide_label": "Reference Video",
            "prompt_description": "Scene Name / Description (optional, used for file names)",
            "prompt_optional": True,  # the prompt only names the output files
            "no_negative_prompt": True,
            "inference_steps": False,
            "lock_inference_steps": True,
            "guidance_max_phases": 0,
            "embedded_guidance": False,
            "image_batch_size_max": 1,
            "model_modes": {**MODEL_MODES, "choices": list(MODEL_MODES["choices"])},
            "custom_settings": [one.copy() for one in CUSTOM_SETTINGS],
            "prompt_enhancer_def": {"selection": [], "labels": {}, "default": ""},
            "profiles_dir": [base_model_type],
            "accelerated": "native",
            "specialities": [{"name": "3D reconstruction", "aliases": ["gaussian splats", "3DGS", "photogrammetry", "point cloud", "camera poses", "video to 3D"],
                              "description": "Reconstructs a static scene as Gaussian splats or a point cloud, with the cameras, from several photos or a video."}],
            "infos": WORLDMIRROR_INFOS,
            "prompt_infos": WORLDMIRROR_PROMPT_INFOS,
            "deepy_infos": WORLDMIRROR_DEEPY_INFOS,
            "deepy_prompt_infos": WORLDMIRROR_DEEPY_PROMPT_INFOS,
        }

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return [{"repoId": REPO, "sourceFolderList": [PROJECT], "fileList": [["worldmirror_2_heads.safetensors", "License.txt", "Notice.txt"]]}]

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        video_prompt_type = inputs["video_prompt_type"]
        if "I" not in video_prompt_type and "V" not in video_prompt_type:
            return "WorldMirror needs Photos, a Reference Video or both."
        images = len(inputs["image_refs"]) if "I" in video_prompt_type else 0
        max_views = int((inputs["custom_settings"] or {}).get("max_views", CUSTOM_SETTINGS[0]["default"]))
        if "V" in video_prompt_type and images >= max_views:
            return f"The {images} Photos already reach Max Views ({max_views}): raise Max Views or remove photos to add Reference Video frames."
        return None

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, save_quantized=False, **kwargs):
        from .worldmirror_main import model_factory

        model = model_factory(model_filename, model_type, base_model_type, save_quantized=save_quantized)
        return model, model.pipe()

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({
            "image_mode": 1,
            "video_prompt_type": "I",
            "model_mode": MODEL_MODES["default"],
            "batch_size": 1,
            "remove_background_images_ref": 0,
            "prompt_enhancer": "",
        })
