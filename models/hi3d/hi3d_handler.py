# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from models import goals_3d
from preprocessing.rembg.assets import OBJECT_MODEL

from .prompts import HI3D_DEEPY_INFOS, HI3D_DEEPY_PROMPT_INFOS, HI3D_INFOS, HI3D_PROMPT_INFOS

ARCHITECTURE = "hi3d"
REPO, PROJECT = "DeepBeepMeep/3D", "hi3d"
MESH_FILE_TYPES = [".glb", ".obj", ".ply", ".stl"]
CUSTOM_SETTINGS = [
    {"id": "level", "name": "Refinement Level", "label": "Refinement Level", "type": "dropdown", "default": 1024,
     "choices": [("512 (1 Pass, Fast)", 512), ("1024 (2 Passes, Recommended)", 1024), ("1536 (3 Passes, Finest)", 1536)],
     "info": "Detail of the refined surface: each level adds a pass on a finer grid and takes several times longer."},
    {"id": "max_faces", "name": "Max Faces", "label": "Max Faces (0 = 1M, 2M at 1536)", "type": "int", "min": 0, "max": 4000000, "inc": 50000,
     "default": 0, "info": "Reduces the refined mesh to this number of faces (quadric decimation)."},
]


class family_handler:
    @staticmethod
    def query_supported_types():
        return [ARCHITECTURE]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "hi3d"

    @staticmethod
    def query_family_infos():
        return {"hi3d": (3040, "Hi3D"), **goals_3d.family_infos(goals_3d.REFINE_MESH)}

    @staticmethod
    def get_lora_dir(base_model_type):
        return base_model_type

    @staticmethod
    def query_model_def(base_model_type, model_def):
        return {
            "group": goals_3d.REFINE_MESH,
            "image_outputs": True,
            "mesh_outputs": True,
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
            "NAG": False,
            "inference_steps": True,
            "profiles_dir": [base_model_type],
            "custom_guide": {"id": "custom_guide", "name": "Mesh", "label": "Input Mesh (.glb/.obj/.ply/.stl)", "type": "file", "default": None,
                             "required": True, "file_types": MESH_FILE_TYPES},
            "custom_settings": [one.copy() for one in CUSTOM_SETTINGS],
            "prompt_enhancer_def": {"selection": [], "labels": {}, "default": ""},
            "infos": HI3D_INFOS,
            "prompt_infos": HI3D_PROMPT_INFOS,
            "deepy_infos": HI3D_DEEPY_INFOS,
            "deepy_prompt_infos": HI3D_DEEPY_PROMPT_INFOS,
            "specialities": [{"name": "3D mesh refinement", "aliases": ["mesh detail", "refine mesh", "high-resolution geometry", "3D upscaling"],
                              "description": "Adds fine geometric detail to a coarse mesh (a Hunyuan3D, TripoSG or TRELLIS.2 shape) from a picture "
                                             "of the same object; geometry only, same frame as the input."}],
        }

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return [{"repoId": REPO, "sourceFolderList": [PROJECT, "hunyuan3d"],
                 "fileList": [["hi3d_vae_fp16.safetensors", "clip_vit_large_patch14_vision_fp16.safetensors", "LICENSE", "NOTICE", "THIRD_PARTY_LICENSES.md", "AUTHORS"],
                              ["dinov2_large_fp16.safetensors"]]}]

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, save_quantized=False, **kwargs):
        from .hi3d_main import model_factory

        model = model_factory(model_filename, model_type, base_model_type, save_quantized=save_quantized)
        return model, model.pipe()

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({
            "image_mode": 1,
            "video_prompt_type": "I",
            "num_inference_steps": 25,
            "batch_size": 1,
            "remove_background_images_ref": 1,
            "prompt_enhancer": "",
        })
