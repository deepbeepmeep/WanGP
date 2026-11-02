# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from models import goals_3d
from preprocessing.rembg.assets import OBJECT_MODEL

from .prompts import (ASSET_NAME_INSTRUCTIONS, DEEPY_PROMPT_INFOS, GEOSAM2_DEEPY_INFOS, GEOSAM2_DEEPY_PROMPT_INFOS, GEOSAM2_INFOS, GEOSAM2_PROMPT_INFOS,
                      PARTCRAFTER_DEEPY_INFOS, PARTCRAFTER_INFOS, PROMPT_INFOS, TRIPOSG_DEEPY_INFOS, TRIPOSG_INFOS)

TRIPOSG_ARCH, PARTCRAFTER_ARCH, GEOSAM2_ARCH = "triposg", "partcrafter", "geosam2"
REPO, PROJECT = "DeepBeepMeep/3D", "triposg"
DINO_REPO, DINO_PROJECT, DINO_FILE = "DeepBeepMeep/3D", "hunyuan3d", "dinov2_large_fp16.safetensors"  # stock DINOv2-L, shared with Hunyuan3D
VAE_FILE = "triposg_vae_fp16.safetensors"
MESH_FILE_TYPES = [".glb", ".obj", ".ply", ".stl"]
GEOSAM2_SETTINGS = [
    {"id": "postprocess_pa", "name": "Small Part Cleanup", "label": "Small Part Cleanup (fraction of the largest part)", "type": "dropdown", "default": 0.02,
     "choices": [("0.01 (Finer Pieces)", 0.01), ("0.02 (Standard)", 0.02), ("0.035 (Fewer, Larger Parts)", 0.035)],
     "info": "Fragments smaller than this fraction of the largest part are merged into their neighbours."},
    {"id": "auto_segment", "name": "Parts", "label": "Parts", "type": "dropdown", "default": 1,
     "choices": [("Automatic + Prompted Parts", 1), ("Prompted Parts Only", 0)],
     "info": "Without Point Prompts the segmentation is always automatic."},
]


def _custom_settings(parts):
    settings = [
        {"id": "mesh_resolution", "name": "Mesh Resolution", "label": "Mesh Resolution (surface grid)", "type": "dropdown", "default": 512,
         "choices": [("256 (Fast)", 256), ("512 (Standard)", 512)],
         "info": "Resolution of the grid the surface is extracted from: 512 keeps the fine details, 256 is faster with fewer faces."},
        {"id": "max_faces", "name": "Max Faces", "label": "Max Faces (0 = keep all)" + (", shared by the parts" if parts else ""), "type": "int",
         "min": 0, "max": 2000000, "inc": 10000, "default": 0, "info": "Reduces the mesh to this number of faces (quadric decimation)."},
    ]
    if parts:
        settings.insert(0, {"id": "num_parts", "name": "Parts", "label": "Number of Parts", "type": "int", "min": 1, "max": 16, "inc": 1, "default": 4,
                            "info": "Number of separate part meshes generated together; match the natural pieces of the object."})
    return settings


class family_handler:
    @staticmethod
    def query_supported_types():
        return [TRIPOSG_ARCH, PARTCRAFTER_ARCH, GEOSAM2_ARCH]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "triposg"

    @staticmethod
    def query_family_infos():
        return {"triposg": (3030, "TripoSG"), **goals_3d.family_infos(goals_3d.IMAGE_TO_3D, goals_3d.MESH_PARTS)}

    @staticmethod
    def get_lora_dir(base_model_type):
        return base_model_type

    @staticmethod
    def query_model_def(base_model_type, model_def):
        if base_model_type == GEOSAM2_ARCH:
            return {
                "group": goals_3d.MESH_PARTS,
                "image_outputs": True,
                "mesh_outputs": True,
                "mesh_postprocessing": False,  # segmented meshes: their parts would merge
                "device_explicit": True,
                "no_model_swap": True,  # one managed model: kept loaded between samples and queued tasks of the same model
                "image_prompt_types_allowed": "",
                "prompt_description": "Asset Name / Description (optional, used for file names)",
                "prompt_optional": True,  # the prompt only names the output files
                "no_negative_prompt": True,
                "embedded_guidance": False,
                "NAG": False,
                "inference_steps": False,
                "profiles_dir": [base_model_type],
                "custom_guide": {"id": "custom_guide", "name": "Mesh", "label": "Input Mesh (.glb/.obj/.ply/.stl)", "type": "file", "default": None,
                                 "required": True, "file_types": MESH_FILE_TYPES},
                "alt_prompt": {"label": "Point Prompts (optional, GeoSAM2 JSON: frame_idx, obj_id, point [x, y] in 1024 px, label)", "name": "Point Prompts",
                               "placeholder": '[{"frame_idx": 3, "obj_id": 1, "point": [512, 420], "label": 1}]', "lines": 3},
                "custom_settings": [one.copy() for one in GEOSAM2_SETTINGS],
                "infos": GEOSAM2_INFOS,
                "prompt_infos": GEOSAM2_PROMPT_INFOS,
                "deepy_infos": GEOSAM2_DEEPY_INFOS,
                "deepy_prompt_infos": GEOSAM2_DEEPY_PROMPT_INFOS,
                "specialities": [{"name": "3D part segmentation", "aliases": ["mesh segmentation", "split mesh into parts", "segment 3D model"],
                                  "description": "Splits an existing mesh into its parts, automatically or from point prompts."}],
            }
        parts = base_model_type == PARTCRAFTER_ARCH
        return {
            "mesh_postprocessing": not parts,  # PartCrafter's parts would merge
            "group": goals_3d.IMAGE_TO_3D,
            "image_outputs": True,
            "mesh_outputs": True,
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
            "profiles_dir": [base_model_type],
            **({} if parts else {"preset_profiles_dir": ["triposg_presets"]}),  # recipes: the shape with a 3D Refiner or a Texturer
            "custom_settings": _custom_settings(parts),
            "prompt_enhancer_def": {"selection": ["TI"], "labels": {"TI": "An Asset Name from the {image_inputs}"}, "default": ""},
            "image_prompt_enhancer_instructions": ASSET_NAME_INSTRUCTIONS,
            "image_prompt_enhancer_max_tokens": 32,
            "infos": PARTCRAFTER_INFOS if parts else TRIPOSG_INFOS,
            "prompt_infos": PROMPT_INFOS,
            "deepy_infos": PARTCRAFTER_DEEPY_INFOS if parts else TRIPOSG_DEEPY_INFOS,
            "deepy_prompt_infos": DEEPY_PROMPT_INFOS,
            "specialities": [{"name": "image to 3D parts", "aliases": ["part segmentation", "multi-part mesh", "separate parts"],
                              "description": "Turns one picture of an object into separate part meshes in one GLB."}] if parts else
                            [{"name": "image to 3D", "aliases": ["3D mesh", "GLB", "3D asset"], "description": "Turns one picture of an object into a detailed 3D mesh."}],
        }

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        if base_model_type == GEOSAM2_ARCH:
            return []
        return [{"repoId": REPO, "sourceFolderList": [PROJECT], "fileList": [[VAE_FILE]]},
                {"repoId": DINO_REPO, "sourceFolderList": [DINO_PROJECT], "fileList": [[DINO_FILE]]}]

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, save_quantized=False, **kwargs):
        if base_model_type == GEOSAM2_ARCH:
            from .geosam2_main import model_factory

            model = model_factory(model_filename, model_type, base_model_type, save_quantized=save_quantized)
            return model, {"transformer": model.transformer}
        from .triposg_main import model_factory

        model = model_factory(model_filename, model_type, base_model_type, save_quantized=save_quantized)
        return model, {"transformer": model.transformer, "vae": model.vae, "image_encoder": model.image_encoder}

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        if base_model_type == GEOSAM2_ARCH:
            ui_defaults.update({"image_mode": 1, "batch_size": 1, "video_prompt_type": "", "alt_prompt": "", "seed": 3})
            return
        ui_defaults.update({
            "image_mode": 1,
            "num_inference_steps": 50,
            "guidance_scale": 7.0,
            "batch_size": 1,
            "video_prompt_type": "I",
            "remove_background_images_ref": 1,
            "prompt_enhancer": "",
        })
