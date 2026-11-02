# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from models import goals_3d
from preprocessing.rembg.assets import OBJECT_MODEL

from .prompts import (ASSET_NAME_INSTRUCTIONS, DEEPY_PROMPT_INFOS, PAINT_DEEPY_INFOS, PAINT_INFOS, PROMPT_INFOS, SHAPE_DEEPY_INFOS, SHAPE_INFOS)

SHAPE_ARCH = "hunyuan3d_21_shape"
PAINT_ARCH = "hunyuan3d_21_paint"
# Hunyuan3D 2.1 PBR (shape and paint in one model type), merged into the Shape model + the Hunyuan3D 2.1 Paint Texturer (2026-10-29)
RETIRED_ARCH = "hunyuan3d_21"
PAINT_TEXTURER = "hunyuan3d_paint"  # its method in the temporal slot (postprocessing/hunyuan3d_paint)
REPO = "DeepBeepMeep/3D"
PROJECT = "hunyuan3d"
SHAPE_FILES = ["hunyuan3d_21_shape_vae_fp16.safetensors", "dinov2_large_fp16.safetensors"]
PAINT_FILES = ["hunyuan3d_21_paint_vae_fp16.safetensors", "dinov2_giant_fp16.safetensors", "realesrgan_x4plus_fp16.safetensors"]
MESH_FILE_TYPES = [".glb", ".obj", ".ply", ".stl"]
TEXTURE_VIEWS = [("6 Views at 512 px (fast)", "6_512"), ("8 Views at 768 px", "8_768"), ("9 Views at 768 px", "9_768")]
DEFAULT_PAINT_FACES = 40000  # the paint model reduces meshes to it (upstream)


def _custom_settings(paint):
    if paint:
        return [
            {"id": "max_faces", "name": "Max Faces", "label": "Max Faces (0 = 40000)", "type": "int", "min": 0, "max": 1000000, "inc": 1000, "default": DEFAULT_PAINT_FACES,
             "info": "The mesh is reduced to this number of faces before it is painted (quadric decimation)."},
            {"id": "texture_views", "name": "Texture Views", "label": "Texture Views", "type": "dropdown", "default": "8_768", "choices": TEXTURE_VIEWS,
             "info": "Views generated to paint the texture: more views and 768 px cover the object better and look sharper, 6 at 512 px is faster."},
            {"id": "texture_size", "name": "Texture Size", "label": "Texture Size", "type": "dropdown", "default": 2048,
             "choices": [("1024", 1024), ("2048", 2048), ("4096", 4096)], "info": "Size of the PBR texture images."},
        ]
    return [
        {"id": "octree_resolution", "name": "Mesh Resolution", "label": "Mesh Resolution (surface grid)", "type": "dropdown", "default": 256,
         "choices": [("196 (Fast)", 196), ("256 (Standard)", 256), ("384 (High)", 384), ("512 (Very High)", 512)],
         "info": "Resolution of the grid the surface is extracted from: higher values keep finer details and give more faces."},
        {"id": "max_faces", "name": "Max Faces", "label": "Max Faces (0 = keep all)", "type": "int", "min": 0, "max": 1000000, "inc": 1000, "default": 0,
         "info": "Reduces the mesh to this number of faces (quadric decimation)."},
    ]


class family_handler:
    @staticmethod
    def query_supported_types():
        return [SHAPE_ARCH, PAINT_ARCH]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_retired_model_types():
        return {RETIRED_ARCH: SHAPE_ARCH}

    @staticmethod
    def convert_retired_settings(model_type, settings):
        """Settings of Hunyuan3D 2.1 PBR: Shape Only becomes the Shape model, Texture a Mesh the Paint model, Shape + PBR Texture the Shape
        model with the Hunyuan3D 2.1 Paint Texturer (its texture settings in the temporal slot). Returns the new model type."""
        mode = settings.pop("model_mode", None) or "shape_texture"
        texture_guidance = settings.pop("alt_guidance_scale", 3.0)
        custom = dict(settings.get("custom_settings") or {})
        texture = {key: custom.pop(key) for key in ("texture_views", "texture_size") if key in custom}
        max_faces = custom.pop("max_faces", DEFAULT_PAINT_FACES)
        if mode == "texture":
            custom.pop("octree_resolution", None)
            settings.update(custom_settings={**texture, "max_faces": max_faces}, guidance_scale=texture_guidance)
            return PAINT_ARCH
        if mode == "shape":
            settings["custom_settings"] = {**custom, "max_faces": max_faces}
            return SHAPE_ARCH
        # the shape keeps its faces: the Texturer reduces it (0 meant the paint model's 40000)
        settings.update(custom_settings={**custom, "max_faces": 0}, temporal_upsampling=PAINT_TEXTURER,
                        temporal_postprocessing_parameters={**texture, "texture_max_faces": max_faces or DEFAULT_PAINT_FACES, "texture_guidance": texture_guidance})
        return SHAPE_ARCH

    @staticmethod
    def query_model_family():
        return "hunyuan3d"

    @staticmethod
    def query_family_infos():
        return {"hunyuan3d": (1300, "Hunyuan3D"), **goals_3d.family_infos(goals_3d.IMAGE_TO_3D, goals_3d.TEXTURE_MESH)}

    @staticmethod
    def get_lora_dir(base_model_type):
        return base_model_type

    @staticmethod
    def query_model_def(base_model_type, model_def):
        paint = base_model_type == PAINT_ARCH
        extra_model_def = {
            "image_outputs": True,
            "mesh_outputs": True,
            "mesh_texture": paint,
            "group": goals_3d.TEXTURE_MESH if paint else goals_3d.IMAGE_TO_3D,
            "device_explicit": True,
            "dtype": "fp16",
            "image_ref_choices": {"choices": [("Object Image", "I")], "letters_filter": "I", "default": "I"},
            "one_image_ref_needed": True,
            "one_image_ref_only": True,
            "image_refs_keep_alpha": True,
            "background_removal_model": OBJECT_MODEL,  # BiRefNet: no cast shadows in the object cutout
            "no_lora": True,
            "prompt_description": "Asset Name / Description (optional, used for file names)",
            "prompt_optional": True,  # the prompt only names the output files
            "no_negative_prompt": True,
            "guidance_max_phases": 1,
            "embedded_guidance": False,
            "NAG": False,
            "inference_steps": not paint,  # the paint model always runs its 15 texture steps
            "profiles_dir": [base_model_type],
            "custom_settings": _custom_settings(paint),
            "prompt_enhancer_def": {"selection": ["TI"], "labels": {"TI": "An Asset Name from the {image_inputs}"}, "default": ""},
            "image_prompt_enhancer_instructions": ASSET_NAME_INSTRUCTIONS,
            "image_prompt_enhancer_max_tokens": 32,
            "prompt_infos": PROMPT_INFOS,
            "deepy_prompt_infos": DEEPY_PROMPT_INFOS,
        }
        if paint:
            extra_model_def.update({
                "custom_guide": {"id": "custom_guide", "name": "Mesh", "label": "Input Mesh (.glb/.obj/.ply/.stl)", "type": "file", "default": None, "required": True,
                                 "file_types": MESH_FILE_TYPES},
                "infos": PAINT_INFOS,
                "deepy_infos": PAINT_DEEPY_INFOS,
                "specialities": [{"name": "PBR texturing", "aliases": ["textured mesh", "texture a mesh", "retexture"],
                                  "description": "Paints a supplied mesh with base colour and metallic-roughness textures matching a picture."}],
            })
        else:
            extra_model_def.update({
                "preset_profiles_dir": ["hunyuan3d_21_shape_presets"],  # recipes: this shape with a 3D Refiner or a Texturer
                "infos": SHAPE_INFOS,
                "deepy_infos": SHAPE_DEEPY_INFOS,
                "specialities": [{"name": "image to 3D", "aliases": ["3D mesh", "GLB", "3D asset"], "description": "Turns one picture of an object into a 3D mesh."}],
            })
        return extra_model_def

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return [{"repoId": REPO, "sourceFolderList": [PROJECT], "fileList": [PAINT_FILES if base_model_type == PAINT_ARCH else SHAPE_FILES]}]

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, save_quantized=False, submodel_no_list=None, **kwargs):
        from .hunyuan3d_main import model_factory

        model = model_factory(model_filename, model_type, base_model_type, save_quantized=save_quantized)
        if model.paint is not None:
            return model, {"transformer": model.paint.unet, "reference_unet": model.paint.unet_dual, "paint_vae": model.paint_vae,
                           "image_encoder": model.paint_image_encoder, "upscaler": model.upscaler}
        return model, {"transformer": model.transformer, "vae": model.vae, "image_encoder": model.image_encoder}

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({
            "image_mode": 1,
            "num_inference_steps": 30,
            "guidance_scale": 3.0 if base_model_type == PAINT_ARCH else 5.0,  # the paint model's guidance is the texture guidance
            "batch_size": 1,
            "video_prompt_type": "I",
            "remove_background_images_ref": 1,
            "prompt_enhancer": "",
        })
