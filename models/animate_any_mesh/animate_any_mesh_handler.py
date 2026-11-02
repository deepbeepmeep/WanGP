# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from models import goals_3d
from shared.utils.hf import build_hf_url

from .prompts import DEEPY_INFOS, DEEPY_PROMPT_INFOS, INFOS, MOTION_ENHANCER, PROMPT_INFOS
from .settings import CUSTOM_SETTINGS

ARCHITECTURE = "animate_any_mesh"
REPO = "DeepBeepMeep/3D"
PROJECT = "animate_any_mesh"
# SD 1.5's frozen CLIP ViT-L/14 text tower, bit-identical to the text tower of the shared FP32 CLIP ViT-L/14 checkpoint
TEXT_ENCODER_REPO, TEXT_ENCODER_FOLDER = "DeepBeepMeep/HunyuanVideo", "clip_vit_large_patch14"
TOKENIZER_FILES = ["text_config.json", "merges.txt", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"]
MESH_FILE_TYPES = [".glb", ".obj", ".ply", ".stl"]


class family_handler:
    @staticmethod
    def query_supported_types():
        return [ARCHITECTURE]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "animate_any_mesh"

    @staticmethod
    def query_family_infos():
        return {"animate_any_mesh": (3040, "AnimateAnyMesh"), **goals_3d.family_infos(goals_3d.ANIMATE_MESH)}

    @staticmethod
    def get_lora_dir(base_model_type):
        return base_model_type

    @staticmethod
    def query_model_def(base_model_type, model_def):
        return {
            "group": goals_3d.ANIMATE_MESH,
            "image_outputs": True,
            "mesh_outputs": True,
            "mesh_postprocessing": False,  # animated meshes
            "device_explicit": True,
            "no_negative_prompt": True,
            "embedded_guidance": False,
            "guidance_max_phases": 1,
            "inference_steps": True,
            "prompt_description": "Motion Description (English), e.g. The dragon flaps its wings and flies",
            "custom_guide": {"id": "custom_guide", "name": "Mesh", "label": "Input Mesh", "type": "file", "default": None, "required": True,
                             "file_types": MESH_FILE_TYPES},
            "custom_settings": [one.copy() for one in CUSTOM_SETTINGS],
            "text_encoder_URLs": [build_hf_url(TEXT_ENCODER_REPO, TEXT_ENCODER_FOLDER, "model.safetensors")],
            "text_encoder_folder": TEXT_ENCODER_FOLDER,
            "prompt_enhancer_def": {"selection": ["T"], "labels": {"T": "A Motion Description using the existing Text Prompt"}, "default": ""},
            "text_prompt_enhancer_instructions": MOTION_ENHANCER,
            "text_prompt_enhancer_max_tokens": 64,
            "profiles_dir": [base_model_type],
            "infos": INFOS,
            "prompt_infos": PROMPT_INFOS,
            "deepy_infos": DEEPY_INFOS,
            "deepy_prompt_infos": DEEPY_PROMPT_INFOS,
            "specialities": [{"name": "mesh animation", "aliases": ["animate a mesh", "animated 3D", "4D", "morph animation", "rigless animation"],
                              "description": "Animates a static 3D mesh (16 frames, no rig) from a text description of its motion."}],
        }

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return [{"repoId": REPO, "sourceFolderList": [PROJECT], "fileList": [["animate_any_mesh_vae_fp32.safetensors"]]},
                {"repoId": TEXT_ENCODER_REPO, "sourceFolderList": [TEXT_ENCODER_FOLDER], "fileList": [TOKENIZER_FILES]}]

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, text_encoder_filename, save_quantized=False, **kwargs):
        from .animate_any_mesh_main import model_factory

        model = model_factory(model_filename, model_type, base_model_type, text_encoder_filename, save_quantized=save_quantized)
        return model, model.pipe()

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({
            "image_mode": 1,
            "num_inference_steps": 63,
            "guidance_scale": 3.0,
            "batch_size": 1,
            "prompt_enhancer": "",
        })
