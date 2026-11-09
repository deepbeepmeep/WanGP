# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""CCORE preview weights from https://huggingface.co/ccore/Qwen3.5-4B-GLB-3D-Coder, revision
541152f42ceb889252554ad6d569f7142e3ed5c3 (Apache 2.0, see LICENSE-CCORE)."""
from pathlib import Path

from models import goals_3d
from shared.prompt_enhancer.assets import QWEN35_VARIANT_4B, QWEN35_VARIANT_SPECS
from shared.utils import files_locator as fl
from . import prompts
from .settings import CUSTOM_SETTINGS

ARCH = "qwen35_glb_coder"
ASSETS = QWEN35_VARIANT_SPECS[QWEN35_VARIANT_4B]
TEXT_FILES = ["config.json", "chat_template.jinja", "tokenizer.json", "tokenizer_config.json"]
VISION_FILES = ["video_preprocessor_config.json", ASSETS["vision_filename"]]


class family_handler:
    @staticmethod
    def query_supported_types():
        return [ARCH]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return ARCH

    @staticmethod
    def query_family_infos():
        return {ARCH: (3015, "GLB 3D Coder"), **goals_3d.family_infos(goals_3d.IMAGE_TO_3D)}

    @staticmethod
    def get_lora_dir(base_model_type):
        return ARCH

    @staticmethod
    def query_model_def(base_model_type, model_def):
        return {
            "group": goals_3d.IMAGE_TO_3D, "mesh_outputs": True, "image_outputs": True,
            "mesh_postprocessing": False,  # Keep scene nodes, materials and animation clips.
            "lm_decoder": True, "device_explicit": True, "profiles_dir": [ARCH], "profile_type": "image",
            "image_prompt_types_allowed": "", "no_negative_prompt": True, "embedded_guidance": False,
            "guidance_max_phases": 0, "inference_steps": False, "lock_inference_steps": True, "NAG": False,
            "prompt_description": "3D Asset Description",
            "custom_settings": [setting.copy() for setting in CUSTOM_SETTINGS],
            "system_configs2": {
                "_name": "Input", "_default_label": "Text Only",
                "_infos": "Text Only needs just an asset description. Text and Reference Image loads the vision encoder for an image-guided asset.",
                "vision": {"name": "Text and Reference Image", "glb_coder_vision": True,
                           "image_ref_choices": {"choices": [("Reference Image", "I")], "letters_filter": "I", "default": "I"},
                           "one_image_ref_needed": True, "one_image_ref_only": True},
            },
            "prompt_enhancer_def": {"selection": ["T"], "labels": {"T": "Expand the Asset Description"}, "default": ""},
            "text_prompt_enhancer_instructions": prompts.ENHANCER,
            "infos": prompts.INFOS, "prompt_infos": prompts.PROMPT_INFOS,
            "deepy_infos": prompts.DEEPY_INFOS, "deepy_prompt_infos": prompts.DEEPY_PROMPT_INFOS,
            "specialities": [{"name": "text to 3D", "aliases": ["3D from text", "low-poly GLB", "animated 3D"],
                              "description": "Writes a low-poly GLB from a text description (or one reference picture), with animated parts when the prompt asks for motion."}],
        }

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        files = TEXT_FILES + (VISION_FILES if model_def and model_def.get("glb_coder_vision") else [])
        return [{"repoId": ASSETS["root_repo"], "sourceFolderList": [ASSETS["repo_subfolder"]], "fileList": [files]}]

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, lm_features, save_quantized=False, **kwargs):
        from .pipeline import GLBCoder

        if save_quantized:
            raise ValueError("GLB 3D Coder is released as quantized GGUF only. Select Q4_K_M or Q8_0 under Model Weights.")
        vision = bool(model_def.get("glb_coder_vision"))
        files = {name: fl.locate_file(f"{ASSETS['repo_subfolder']}/{name}") for name in TEXT_FILES + (VISION_FILES if vision else [])}
        checkpoint, = model_filename
        pipeline = GLBCoder(checkpoint, Path(files["config.json"]).parent, files.get(ASSETS["vision_filename"]), lm_features)
        pipe = {"transformer": pipeline.transformer}
        if vision:
            pipe["image_encoder"] = pipeline.vision.vision_tower_model
        from shared.deepy.config import get_deepy_config_value
        cpu_compute = get_deepy_config_value("llm_cpu_compute", "auto")
        # Every token reads the full decoder. 5 GiB covers the larger Q8_0 release while MMGP can reclaim preloaded layers.
        return pipeline, {"pipe": pipe, "budgets": {"transformer": 5 * 1024}, "cudaGraphs": ["transformer"] if lm_features.graphs else False,
                          "cpuCompute": ["transformer"] if cpu_compute != "off" else False, "cpuComputeForce": cpu_compute == "on"}

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update(image_mode=1, video_prompt_type="I" if model_def.get("glb_coder_vision") else "",
                           prompt_enhancer="", batch_size=1, guidance_scale=1.0, num_inference_steps=1)
