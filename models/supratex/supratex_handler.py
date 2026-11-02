# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from models import goals_3d
from preprocessing.rembg.assets import OBJECT_MODEL

from .prompts import (SHAPE_DEEPY_INFOS, SHAPE_INFOS, SHAPE_PROMPT_INFOS, TEXTURE_DEEPY_INFOS, TEXTURE_DEEPY_PROMPT_INFOS, TEXTURE_INFOS,
                      TEXTURE_PROMPT_INFOS)

TEXTURE_ARCH, SHAPE_ARCH = "supratex_texture", "pixal3d"
# Pixal3D + SUPRA-Tex in one model type, merged into Pixal3D + the SUPRA-Tex Texturer (2026-10-29)
RETIRED_ARCH = "supratex"
TEXTURER = "supratex"  # the SUPRA-Tex Texturer's method in the temporal slot (postprocessing/supratex)
COMPLETION_STEPS = 25  # the former full model's fixed texture steps
REPO, PROJECT = "DeepBeepMeep/3D", "supratex"
MESH_FILE_TYPES = [".glb", ".obj", ".ply", ".stl"]
TEXTURE_FILES = ["completion_config.json", "completion_partial_normalization.json", "supratex_material_lora_bf16.safetensors",
                 "supratex_material_embeddings_bf16.safetensors", "LICENSE_SUPRA-Tex.txt", "LICENSE_Qwen_Research.txt", "Notice_material.txt"]
TEXTURE_TRELLIS2_FILES = ["trellis2_shape_encoder_fp16.safetensors", "trellis2_texture_encoder_fp16.safetensors", "trellis2_texture_decoder_fp16.safetensors"]
SHAPE_FILES = ["naf_bf16.safetensors", "LICENSE_NAF.txt", "LICENSE_Pixal3D.txt"]
SHAPE_TRELLIS2_FILES = ["trellis2_ss_decoder_fp16.safetensors", "trellis2_shape_decoder_fp16.safetensors"]
DINOV3_FILES = ["dinov3_vitl16_lvd1689m.safetensors", "LICENSE_DINOv3.md"]


def _settings(base_model_type):
    shape = base_model_type == SHAPE_ARCH
    settings = [{"id": "max_faces", "name": "Max Faces", "label": "Max Faces (0 = keep all)", "type": "int", "min": 0, "max": 4000000, "inc": 50000,
                 "default": 500000 if shape else 0,
                 "info": "Reduces the generated mesh to this number of faces (quadric decimation)." if shape else
                         "Reduces the mesh to this number of faces before texturing (quadric decimation); 0 keeps every face."}]
    if not shape:
        settings.append({"id": "texture_size", "name": "Texture Size", "label": "Texture Size", "type": "dropdown", "default": 2048,
                         "choices": [("1024", 1024), ("2048", 2048), ("4096", 4096)], "info": "Size of the PBR texture images."})
    return settings


class family_handler:
    @staticmethod
    def query_supported_types():
        return [TEXTURE_ARCH, SHAPE_ARCH]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_retired_model_types():
        return {RETIRED_ARCH: SHAPE_ARCH}

    @staticmethod
    def convert_retired_settings(model_type, settings):
        """Settings of SUPRA-Tex (Pixal3D + SUPRA-Tex): Shape Only becomes Pixal3D, Texture a Mesh SUPRA-Tex Texture, Shape + PBR Texture
        Pixal3D with the SUPRA-Tex Texturer (which reuses the Pixal3D mesh's camera, as the full model did). Returns the new model type."""
        mode = settings.pop("model_mode", None) or "shape_texture"
        custom = dict(settings.get("custom_settings") or {})
        max_faces, texture_size = custom.get("max_faces", 500000), custom.get("texture_size", 2048)
        if mode == "texture":
            settings.update(custom_settings={"max_faces": max_faces, "texture_size": texture_size}, num_inference_steps=COMPLETION_STEPS)
            return TEXTURE_ARCH
        settings["custom_settings"] = {"max_faces": max_faces}
        if mode == "shape_texture":  # the full model textured every face of its mesh
            settings.update(temporal_upsampling=TEXTURER, temporal_postprocessing_parameters={"texture_size": texture_size, "texture_max_faces": 0})
        return SHAPE_ARCH

    @staticmethod
    def query_model_family():
        return "supratex"

    @staticmethod
    def query_family_infos():
        return {"supratex": (3050, "SUPRA-Tex / Pixal3D"), **goals_3d.family_infos(goals_3d.IMAGE_TO_3D, goals_3d.TEXTURE_MESH)}

    @staticmethod
    def get_lora_dir(base_model_type):
        return base_model_type

    @staticmethod
    def query_model_def(base_model_type, model_def):
        from models.qwen21.qwen21_handler import ENCODER, ENCODER_REPO
        from shared.utils.hf import build_hf_url

        textured = base_model_type == TEXTURE_ARCH
        definition = {
            "image_outputs": True,
            "mesh_outputs": True,
            "mesh_texture": textured,
            "group": goals_3d.TEXTURE_MESH if textured else goals_3d.IMAGE_TO_3D,
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
            "guidance_max_phases": 0 if textured else 1,
            "embedded_guidance": False,
            "NAG": False,
            "no_lora": True,
            "inference_steps": True,
            "profiles_dir": [],
            "custom_settings": _settings(base_model_type),
            "prompt_enhancer_def": {"selection": [], "labels": {}, "default": ""},
            "deepy_prompt_infos": TEXTURE_DEEPY_PROMPT_INFOS,
        }
        if textured:
            definition.update({
                "text_encoder_folder": ENCODER,
                "text_encoder_URLs": [build_hf_url(ENCODER_REPO, ENCODER, ENCODER + suffix) for suffix in ("_bf16.safetensors", "_int8_convrot.safetensors")],
                "custom_guide": {"id": "custom_guide", "name": "Mesh", "label": "Input Mesh (.glb/.obj/.ply/.stl)", "type": "file", "default": None, "required": True,
                                 "file_types": MESH_FILE_TYPES},
                "infos": TEXTURE_INFOS, "prompt_infos": TEXTURE_PROMPT_INFOS, "deepy_infos": TEXTURE_DEEPY_INFOS,
                "specialities": [{"name": "PBR texturing", "aliases": ["texture a mesh", "retexture", "mesh texture", "albedo metallic roughness"],
                                  "description": "Paints a mesh with base colour, metallic and roughness textures from one picture of the object, keeping "
                                                 "the picture's colours and print on the visible side."}],
            })
        else:
            definition.update({
                "preset_profiles_dir": ["pixal3d_presets"],  # recipes: this shape with the SUPRA-Tex Texturer, after the Hi3D 3D Refiner or not
                "infos": SHAPE_INFOS, "prompt_infos": SHAPE_PROMPT_INFOS, "deepy_infos": SHAPE_DEEPY_INFOS,
                "specialities": [{"name": "image to 3D", "aliases": ["3D mesh", "GLB", "3D asset", "pixel-aligned 3D"],
                                  "description": "Turns one picture of an object into a 3D mesh aligned with the picture's viewpoint."}],
            })
        return definition

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        if base_model_type == SHAPE_ARCH:
            return [{"repoId": REPO, "sourceFolderList": [PROJECT, "trellis2", "dinov3_vitl16"], "fileList": [SHAPE_FILES, SHAPE_TRELLIS2_FILES, DINOV3_FILES]}]
        from models.qwen21.qwen21_handler import family_handler as qwen21

        return [{"repoId": REPO, "sourceFolderList": [PROJECT, "trellis2"], "fileList": [TEXTURE_FILES, TEXTURE_TRELLIS2_FILES]},
                *qwen21.query_model_files(computeList, "qwen_image_21_7B", model_def)]

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, text_encoder_filename=None, save_quantized=False, **kwargs):
        from .supratex_main import model_factory

        model = model_factory(model_filename, model_type, base_model_type, text_encoder_filename=text_encoder_filename, save_quantized=save_quantized, **kwargs)
        if base_model_type == SHAPE_ARCH:
            return model, model.pipe()
        return model, {"pipe": model.pipe(), "loras": ["material_transformer"]}  # LoRA slots of the material adapter

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({
            "image_mode": 1,
            "video_prompt_type": "I",
            "num_inference_steps": COMPLETION_STEPS if base_model_type == TEXTURE_ARCH else 12,
            "batch_size": 1,
            "remove_background_images_ref": 1,
            "prompt_enhancer": "",
        })
        if base_model_type == SHAPE_ARCH:
            ui_defaults["guidance_scale"] = 7.5
