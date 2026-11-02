# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from models import goals_3d

from .prompts import (DA3_DEEPY_INFOS, DA3_INFOS, DEEPY_PROMPT_INFOS, MAPANYTHING_DEEPY_INFOS, MAPANYTHING_INFOS, MOGE2_DEEPY_INFOS, MOGE2_INFOS,
                      PROMPT_INFOS)
from .settings import CUSTOM_SETTINGS, MOGE2_MODES

ARCHITECTURES = ["moge2", "mapanything", "da3_points"]
INFOS = {"moge2": (MOGE2_INFOS, MOGE2_DEEPY_INFOS), "mapanything": (MAPANYTHING_INFOS, MAPANYTHING_DEEPY_INFOS), "da3_points": (DA3_INFOS, DA3_DEEPY_INFOS)}
SPECIALITIES = {
    "moge2": [{"name": "photo to 3D", "aliases": ["point cloud", "depth to 3D", "2.5D mesh"],
               "description": "Turns one photo into a metric point cloud or a textured mesh of its visible surface."}],
    "mapanything": [{"name": "multi-view reconstruction", "aliases": ["photogrammetry", "video to 3D", "camera poses", "point cloud"],
                     "description": "Reconstructs a scene in metres with its cameras from several photos or a video."}],
    "da3_points": [{"name": "video to point cloud", "aliases": ["depth to 3D", "camera poses"],
                    "description": "Fast point cloud and cameras from photos or video frames (relative scale)."}],
}


class family_handler:
    @staticmethod
    def query_supported_types():
        return ARCHITECTURES

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "recon3d"

    @staticmethod
    def query_family_infos():
        return goals_3d.family_infos(goals_3d.RECONSTRUCTION)

    @staticmethod
    def get_lora_dir(base_model_type):
        return base_model_type

    @staticmethod
    def query_model_def(base_model_type, model_def):
        infos, deepy_infos = INFOS[base_model_type]
        extra_model_def = {
            "image_outputs": True,
            "mesh_outputs": True,
            "mesh_postprocessing": False,  # reconstructed scenes, point clouds and splats
            "device_explicit": True,
            "no_model_swap": True,  # one managed model: kept loaded between samples and queued tasks of the same model
            "no_background_removal": True,
            "fit_into_canvas_image_refs": 0,
            "prompt_description": "Asset Name / Description (optional, used for file names)",
            "prompt_optional": True,  # the prompt only names the output files
            "no_negative_prompt": True,
            "embedded_guidance": False,
            "inference_steps": False,
            "lock_inference_steps": True,
            "image_batch_size_max": 1,
            "custom_settings": [one.copy() for one in CUSTOM_SETTINGS[base_model_type]],
            "prompt_enhancer_def": {"selection": [], "labels": {}, "default": ""},
            "profiles_dir": [base_model_type],
            "infos": infos,
            "prompt_infos": PROMPT_INFOS,
            "deepy_infos": deepy_infos,
            "deepy_prompt_infos": DEEPY_PROMPT_INFOS,
            "specialities": SPECIALITIES[base_model_type],
        }
        if base_model_type == "moge2":
            extra_model_def.update({
                "image_ref_choices": {"choices": [("Photo", "I")], "letters_filter": "I", "default": "I"},
                "one_image_ref_needed": True,
                "one_image_ref_only": True,
                "model_modes": {**MOGE2_MODES, "choices": list(MOGE2_MODES["choices"])},
            })
        else:
            extra_model_def.update({
                "image_ref_choices": {"choices": [("Photos", "I"), ("No Photos", "")], "letters_filter": "I", "default": "I"},
                "guide_custom_choices": {"choices": [("No Video", ""), ("Reference Video", "V")], "letters_filter": "V", "default": "", "label": "Video"},
                "reference_video_enabled": True,
                "skip_video_guide_preprocess": True,  # generate() samples its frames from the file
                "video_guide_label": "Reference Video",
            })
        return extra_model_def

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        if base_model_type != "da3_points":
            return []
        from preprocessing.depth_anything_v3.assets import query_download_def

        return [query_download_def(metric=False)]  # the checkpoint of WanGP's Depth Anything 3 depth preprocessor (DA3-LARGE-1.1)

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        if base_model_type != "moge2" and "I" not in inputs["video_prompt_type"] and "V" not in inputs["video_prompt_type"]:
            return "Add Photos, a Reference Video or both to reconstruct."
        return None

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, save_quantized=False, **kwargs):
        from .recon3d_main import model_factory

        model = model_factory(model_filename, model_type, base_model_type, save_quantized=save_quantized)
        return model, {"transformer": model.transformer}

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({"image_mode": 1, "video_prompt_type": "I", "batch_size": 1, "num_inference_steps": 1, "remove_background_images_ref": 0})
        if base_model_type == "moge2":
            ui_defaults["model_mode"] = MOGE2_MODES["default"]
