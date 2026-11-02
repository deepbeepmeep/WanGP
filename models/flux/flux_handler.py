# Copyright (c) 2025-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from models.help_shared import merge_help
from .infos import flux_help
import torch
import gradio as gr
from PIL import Image
from shared.utils import files_locator as fl
from shared.utils.hf import build_hf_url

def test_flux2(base_model_type):
    return base_model_type in ["flux2_dev", "pi_flux2", "flux2_klein_4b", "flux2_klein_9b", "flux2_klein_9b_kv"]


def get_text_encoder_name(base_model_type, text_encoder_quantization):
    if base_model_type == "flux2_klein_4b":
        if text_encoder_quantization == "int8":
            return "qwen3_quanto_bf16_int8.safetensors"
        return "qwen3_bf16.safetensors"
    if base_model_type in ("flux2_klein_9b", "flux2_klein_9b_kv"):
        if text_encoder_quantization == "int8":
            return "qwen3_8b_int8_convrot.safetensors" if base_model_type == "flux2_klein_9b_kv" else "qwen3_8b_quanto_bf16_int8.safetensors"
        return "qwen3_8b_bf16.safetensors"
    if text_encoder_quantization == "int8":
        return "mistral3_small_quanto_bf16_int8.safetensors" if test_flux2(base_model_type) else "T5_xxl_1.1_enc_quanto_bf16_int8.safetensors"
    return "mistral3_small_bf16.safetensors" if test_flux2(base_model_type) else "T5_xxl_1.1_enc_bf16.safetensors"


class family_handler():
    @staticmethod
    def query_supported_types():
        return [
            "flux",
            "flux2_dev",
            "pi_flux2",
            "flux2_klein_4b",
            "flux2_klein_9b",
            "flux2_klein_9b_kv",
            "flux_chroma",
            "flux_chroma_radiance",
            "flux_dev_kontext",
            "flux_dev_umo",
            "flux_dev_uso",
            "flux_schnell",
            "flux_dev_kontext_dreamomni2",
        ]

    @staticmethod
    def query_family_maps():

        models_eqv_map = {
            "flux_dev_kontext" : "flux",
            "flux_dev_umo" : "flux",
            "flux_dev_uso" : "flux",
            "flux_schnell" : "flux",
            "flux_chroma" : "flux",
            "flux_chroma_radiance": "flux",
            "flux_dev_kontext_dreamomni2": "flux",
            "flux2_dev": "flux",
            "pi_flux2": "flux",
            "flux2_klein_4b": "flux",
            "flux2_klein_9b": "flux",
            "flux2_klein_9b_kv": "flux",
        }

        models_comp_map = {
                    "flux": ["flux2_dev", "pi_flux2", "flux2_klein_4b", "flux2_klein_9b", "flux_chroma", "flux_chroma_radiance", "flux_dev_kontext", "flux_dev_umo", "flux_dev_uso", "flux_schnell", "flux_dev_kontext_dreamomni2" ]
                    }
        models_comp_map["flux"].append("flux2_klein_9b_kv")
        return models_eqv_map, models_comp_map
    @staticmethod
    def query_model_def(base_model_type, model_def):
        flux_model = "flux-dev" if base_model_type == "flux" else base_model_type.replace("_", "-")
        flux_dev = base_model_type == "flux"
        pi_flux2 = base_model_type == "pi_flux2"
        flux2_klein_4b = base_model_type == "flux2_klein_4b"
        flux2_klein_kv = base_model_type == "flux2_klein_9b_kv"
        flux2_klein_9b = base_model_type in ("flux2_klein_9b", "flux2_klein_9b_kv")
        flux2_klein = flux2_klein_4b or flux2_klein_9b
        flux2 = flux_model.startswith("flux2") or pi_flux2
        flux_schnell = flux_model == "flux-schnell"
        flux_chroma = flux_model == "flux-chroma"
        flux_chroma_radiance = flux_model == "flux-chroma-radiance"
        flux_uso = flux_model == "flux-dev-uso"
        flux_umo = flux_model == "flux-dev-umo"
        flux_kontext = flux_model == "flux-dev-kontext"
        flux_kontext_dreamomni2 = flux_model == "flux-dev-kontext-dreamomni2"

        extra_model_def = {
            "image_outputs" : True,
            "no_negative_prompt" :  flux_chroma or flux_chroma_radiance,
            "flux-model": flux_model,
            "flux2": flux2,
            "device_explicit": base_model_type in ("flux", "flux_schnell", "flux2_klein_4b", "flux2_klein_9b", "flux2_klein_9b_kv", "flux2_dev", "flux_chroma_radiance", "flux_dev_uso", "pi_flux2"),
            "cuda_graphs": ["transformer"],
        }
        if flux_chroma or flux_chroma_radiance:
            extra_model_def["guidance_max_phases"] = 1
        else:
            extra_model_def["NAG"] = True
            extra_model_def["no_negative_prompt"] = False
        supports_inpaint = flux_kontext or flux_schnell or flux_dev or flux2
        if supports_inpaint:
            extra_model_def["inpaint_support"] = True
            if not pi_flux2:
                lanpaint_choices = [
                    ("LanPaint (2 steps): ~2x slower, easy task", 2),
                    ("LanPaint (5 steps): ~5x slower, medium task", 3),
                    ("LanPaint (10 steps): ~10x slower, hard task", 4),
                    ("LanPaint (15 steps): ~15x slower, very hard task", 5),
                ]
                if flux_dev or flux_schnell:
                    choices = lanpaint_choices
                    default_mode = 2
                    extra_model_def["inpaint_video_prompt_type"] = "VA"
                    extra_model_def["image_video_prompt_type"] = ""
                else:
                    choices = [("Masked Denoising : Inpainted area may reuse some content that has been masked", 0)] + lanpaint_choices
                    default_mode = 0
                extra_model_def["model_modes"] = {
                    "choices": choices,
                    "default": default_mode,
                    "label": "Inpainting Method",
                    "image_modes": [2],
                }
            extra_model_def["inpaint_color"] = "FF0000"
            extra_model_def["video_guide_outpainting"] = [1,2]
            extra_model_def["outpainting_quantize_margins"]= 16

        if flux2:
            if flux2_klein:
                if flux2_klein_4b:
                    text_encoder_folder = "Qwen3"
                    text_encoder_repo = "DeepBeepMeep/Z-Image"
                    text_encoder_urls = [
                        build_hf_url(text_encoder_repo, text_encoder_folder, "qwen3_bf16.safetensors"),
                        build_hf_url(text_encoder_repo, text_encoder_folder, "qwen3_quanto_bf16_int8.safetensors"),
                    ]
                else:
                    text_encoder_folder = "qwen3_8b"
                    text_encoder_repo = "DeepBeepMeep/Flux2"
                    text_encoder_urls = [
                        build_hf_url(text_encoder_repo, text_encoder_folder, "qwen3_8b_bf16.safetensors"),
                        build_hf_url(text_encoder_repo, text_encoder_folder, "qwen3_8b_int8_convrot.safetensors" if flux2_klein_kv else "qwen3_8b_quanto_bf16_int8.safetensors"),
                    ]
                extra_model_def["text_encoder_type"] = "qwen3"
                extra_model_def["text_encoder_URLs"] = text_encoder_urls
            else:
                text_encoder_folder = "mistral3small"
                extra_model_def["text_encoder_URLs"] = [
                    build_hf_url("DeepBeepMeep/Flux2", text_encoder_folder, "mistral3_small_bf16.safetensors"),
                    build_hf_url("DeepBeepMeep/Flux2", text_encoder_folder, "mistral3_small_quanto_bf16_int8.safetensors"),
                ]
                extra_model_def["text_encoder_type"] = "mistral3"
            extra_model_def["text_encoder_folder"] = text_encoder_folder
        else:
            text_encoder_folder = "T5_xxl_1.1"
            extra_model_def["text_encoder_URLs"] = [
                build_hf_url("DeepBeepMeep/LTX_Video", text_encoder_folder, "T5_xxl_1.1_enc_bf16.safetensors"),
                build_hf_url("DeepBeepMeep/LTX_Video", text_encoder_folder, "T5_xxl_1.1_enc_quanto_bf16_int8.safetensors"),
            ]
            extra_model_def["text_encoder_folder"] = text_encoder_folder
        if flux2_klein:
            extra_model_def["profiles_dir"] = ["flux2_klein_4b"] if flux2_klein_4b else ["flux2_klein_9b"]
            extra_model_def.update({
                "deepy_infos": "Text-to-image or editing with ordered `image_refs`. Preserve `video_prompt_type`: `KI` = first image is the main scene, then people/objects; `I` = people/objects. Inpainting combines source, mask and prompt.",
                "deepy_prompt_infos": "Text-only: describe the finished image. Editing: give a precise change and state what to preserve, e.g. 'Replace the woman in image 1 with the man from image 2; keep pose, lighting and background.' Number references in upload order and assign each a role. Quote exact visible text.",
                "infos": "Generate from text alone, or combine the Text Prompt (`prompt`) with ordered Reference Images (`image_refs`) for editing and composition. Reference mode `KI` treats the first image as the main scene, followed by people/objects; `I` supplies people/objects. Describe each image's role in the prompt. Inpainting combines the source image, mask and prompt to target a region.",
                "prompt_infos": 'For text-only generation, describe the finished image. For editing, give a precise instruction and identify what to retain: "Replace the woman in image 1 with the man from image 2; keep the pose, camera angle, lighting and background." For a new composition, describe the final scene and assign each reference a role, such as the person from image 1 wearing the coat from image 2. Image numbers follow upload order. Keep instructions compatible and quote exact visible text.',
            })
        else:
            extra_model_def["profiles_dir"] = [] if (flux_schnell or flux2) else ["flux"]
        if flux_chroma_radiance:
            extra_model_def["radiance"] = True
        elif not flux_schnell and not flux2_klein:
            extra_model_def["embedded_guidance"] = True
        if not flux_chroma_radiance:
            extra_model_def["vae_upsamplers"] = {"flux2_vae_pid" if flux2 else "flux_vae_pid": [1, 2]}
            extra_model_def["excluded_spatial_upsamplers"] = ["flux2_pid" if flux2 else "flux_pid"]
        if flux_uso :
            extra_model_def["any_image_refs_relative_size"] = True
            extra_model_def["image_refs_relative_size"] = {"min": 20, "max": 100, "step": 1}
            extra_model_def["no_background_removal"] = True
            extra_model_def["image_ref_choices"] = {
                "choices":[("First Image is a Reference Image, and then the next ones (up to two) are Style Images", "KI"),
                            ("Up to two Images are Style Images", "KIJ")],
                "default": "KI",
                "letters_filter": "KIJ",
                "label": "Reference Images / Style Images"
            }
        
        if flux_kontext or flux_kontext_dreamomni2 or flux2:
            extra_model_def["image_ref_choices"] = {
                "choices": [
                    ("None", ""),
                    ("Conditional Image is first Main Subject / Landscape and may be followed by People / Objects", "KI"),
                    ("Conditional Images are People / Objects", "I"),
                    ],
                "letters_filter": "KI",
            }
            if flux_kontext_dreamomni2:
                extra_model_def["no_background_removal"] = True
            else:                
                extra_model_def["background_removal_label"]= "Remove Backgrounds only behind People / Objects except main Subject / Landscape" 
        elif flux_umo:
            extra_model_def["image_ref_choices"] = {
                "choices": [
                    ("Conditional Images are People / Objects", "I"),
                    ],
                "letters_filter": "I",
                "visible": False
            }

        if flux2:
            extra_model_def["group"] ="flux2"
            extra_model_def["no_background_removal"] = True
            # extra_model_def["inpaint_support"] = True
            extra_model_def["mask_preprocessing"] = {
                "selection":[ ""],
                "visible": False
            }

            extra_model_def["mask_strength_always_enabled"] = True

            extra_model_def["guide_preprocessing"] = {
                "selection": ["", "PV", "MV"],
            }

            extra_model_def["mask_preprocessing"] = {
                    "selection": ["", "A", "NA"],
                    "visible": True,
                }


        if pi_flux2:
            extra_model_def["piflow"] = True

        if flux2_klein_kv:
            extra_model_def.update(NAG=False, no_negative_prompt=True, guidance_max_phases=0, profiles_dir=["flux2_klein_9b_kv"],
                                   vae_filename="flux2/flux2_vae_bf16.safetensors")
            for key in ("infos", "prompt_infos", "deepy_infos", "deepy_prompt_infos"):
                del extra_model_def[key]

        extra_model_def["fit_into_canvas_image_refs"] = 0
        if not flux_chroma_radiance:
            extra_model_def["tiny_vae_architecture"] = "flux2_klein_4b" if flux2 else "flux"

        return merge_help(extra_model_def, flux_help(base_model_type, {**extra_model_def, **model_def}))


    @staticmethod
    def get_rgb_factors(base_model_type ):
        if base_model_type in ["flux_chroma_radiance"]:
            return None, None
        from shared.RGB_factors import get_rgb_factors
        latent_rgb_factors, latent_rgb_factors_bias = get_rgb_factors("flux", sub_family= "flux2" if test_flux2(base_model_type) else "flux")
        return latent_rgb_factors, latent_rgb_factors_bias

    @staticmethod
    def preview_latents(base_model_type, latents, meta):
        if base_model_type != "flux_chroma_radiance":
            return None
        from .sampling import patches_to_image

        tensor = latents
        if not torch.is_tensor(tensor):
            return None
        tensor = tensor.detach()
        C, T, H, W = tensor.shape
        image = tensor.cpu().clamp(-1, 1)
        image = image.permute(0, 2, 1, 3) # (C, T, H, W) -> (C, H, T, W)        
        image = image.reshape(C, H, T * W) # (C, H, T, W) -> (C, H, T*W)
        image = image.add(1).mul(127.5).clamp(0, 255).to(torch.uint8)
        image = image.permute(1, 2, 0).numpy()
        preview = Image.fromarray(image)
        if preview.height > 0:
            scale = 200 / preview.height
            width_px = max(1, int(round(preview.width * scale)))
            resampling_module = getattr(Image, "Resampling", Image)
            resample_filter = getattr(resampling_module, "BILINEAR", Image.BILINEAR)
            preview = preview.resize((width_px, 200), resample=resample_filter)
        return preview


    @staticmethod
    def query_model_family():
        return "flux"

    @staticmethod
    def query_family_infos():
        return {"flux":(1100, "Flux 1"), "flux2":(1101, "Flux 2")}

    @staticmethod
    def get_lora_dir(base_model_type):
        if base_model_type == "flux2_klein_9b_kv":
            return "flux2_klein_9b_kv"
        if base_model_type == "flux2_klein_4b":
            return "flux2_klein_4b"
        if base_model_type == "flux2_klein_9b":
            return "flux2_klein_9b"
        if test_flux2(base_model_type):
            return "flux2"
        return "flux"

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        if base_model_type in ["flux2_klein_4b", "flux2_klein_9b", "flux2_klein_9b_kv"]:
            if base_model_type == "flux2_klein_4b":
                text_encoder_folder = "Qwen3"
                text_encoder_repo = "DeepBeepMeep/Z-Image"
            else:
                text_encoder_folder = "qwen3_8b"
                text_encoder_repo = "DeepBeepMeep/Flux2"

            tokenizer_files = ["config.json", "generation_config.json", "added_tokens.json", "chat_template.jinja", "merges.txt", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"]

            ret = [
                {
                    "repoId": text_encoder_repo,
                    "sourceFolderList": [text_encoder_folder],
                    "fileList": [tokenizer_files],
                },
                {
                    "repoId": "DeepBeepMeep/Flux2",
                    "sourceFolderList": ["flux2" if base_model_type == "flux2_klein_9b_kv" else ""],
                    "fileList": [["flux2_vae_bf16.safetensors" if base_model_type == "flux2_klein_9b_kv" else "flux2_vae.safetensors"]],
                },
            ]
        elif test_flux2(base_model_type):
            ret = [
                {
                "repoId": "DeepBeepMeep/Flux2",
                "sourceFolderList": ["mistral3small", ""],
                "fileList": [
                    [ "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "processor_config.json", "config.json", "preprocessor_config.json", "chat_template.jinja", ],
                    [ "flux2_vae.safetensors", ],
                ],
                }
            ]
        else:
            ret = [
                {  
                "repoId" : "DeepBeepMeep/LTX_Video", 
                "sourceFolderList" :  ["T5_xxl_1.1"],
                "fileList" : [ ["added_tokens.json", "special_tokens_map.json", "spiece.model", "tokenizer_config.json"]  ]   
                },
                {  
                "repoId" : "DeepBeepMeep/HunyuanVideo", 
                "sourceFolderList" :  [  "clip_vit_large_patch14",   ],
                "fileList" :[ 
                                ["text_config.json", "merges.txt", "model.safetensors", "preprocessor_config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"],
                                ]
                },
                {  
                "repoId" : "DeepBeepMeep/Flux", 
                "sourceFolderList" :  ["",],
                "fileList" : [ ["flux_vae.safetensors"] ]   
                }]

            if base_model_type in ["flux_dev_uso"]:
                ret += [
                    {  
                    "repoId" : "DeepBeepMeep/Flux", 
                    "sourceFolderList" :  ["siglip-so400m-patch14-384"],
                    "fileList" : [ ["vision_config.json", "preprocessor_config.json", "model.safetensors"] ]   
                    }]


            if base_model_type in ["flux_dev_kontext_dreamomni2"]:
                ret += [
                    {  
                    "repoId" : "DeepBeepMeep/Flux", 
                    "sourceFolderList" :  ["Qwen2.5-VL-7B-DreamOmni2"],
                    "fileList" : [ ["Qwen2.5-VL-7B-DreamOmni2_quanto_bf16_int8.safetensors", "merges.txt", "tokenizer_config.json", "config.json", "vocab.json", "video_preprocessor_config.json", "preprocessor_config.json", "chat_template.jinja"] ]
                    }]
            
        return ret

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, quantizeTransformer = False, text_encoder_quantization = None, dtype = torch.bfloat16, VAE_dtype = torch.float32, mixed_precision_transformer = False, save_quantized = False, submodel_no_list = None, text_encoder_filename = None, **kwargs):
        from .flux_main  import model_factory

        flux_model = model_factory(
            checkpoint_dir="ckpts",
            model_filename=model_filename,
            model_type = model_type, 
            model_def = model_def,
            base_model_type=base_model_type,
            text_encoder_filename= text_encoder_filename,
            quantizeTransformer = quantizeTransformer,
            dtype = dtype,
            VAE_dtype = VAE_dtype, 
            mixed_precision_transformer = mixed_precision_transformer,
            save_quantized = save_quantized,
        )

        pipe = { "transformer": flux_model.model, "vae" : flux_model.vae}
        if getattr(flux_model, "clip", None) is not None:
            pipe["text_encoder"] = flux_model.clip
        if getattr(flux_model, "t5", None) is not None:
            pipe["text_encoder_2"] = flux_model.t5
        if getattr(flux_model, "mistral", None) is not None:
            pipe["text_encoder"] = flux_model.mistral.model

        if flux_model.vision_encoder is not None:
            pipe["siglip_model"] = flux_model.vision_encoder 
        if flux_model.feature_embedder is not None:
            pipe["feature_embedder"] = flux_model.feature_embedder 
        if flux_model.vlm_model is not None:
            pipe["vlm_model"] = flux_model.vlm_model 
        return flux_model, pipe

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        flux_model = model_def.get("flux-model", "flux-dev")
        flux_uso = flux_model == "flux-dev-uso"
        if flux_uso and settings_version < 2.29:
            video_prompt_type = ui_defaults.get("video_prompt_type", "")
            if "I" in video_prompt_type:
                video_prompt_type = video_prompt_type.replace("I", "KI")
                ui_defaults["video_prompt_type"] = video_prompt_type 

        if settings_version < 2.34:
            ui_defaults["denoising_strength"] = 1.
        if flux_model.startswith("flux2"):
            ui_defaults["embedded_guidance_scale"] = ui_defaults.get("embedded_guidance_scale", 4.0)

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        flux_model = model_def.get("flux-model", "flux-dev")
        flux_uso = flux_model == "flux-dev-uso"
        flux_umo = flux_model == "flux-dev-umo"
        flux_kontext = flux_model == "flux-dev-kontext"
        flux_kontext_dreamomni2 = flux_model == "flux-dev-kontext-dreamomni2"
        flux2 = flux_model.startswith("flux2")
        flux2_klein = base_model_type in ["flux2_klein_4b", "flux2_klein_9b", "flux2_klein_9b_kv"]

        ui_defaults.update({
            "image_mode": 1,
            "embedded_guidance_scale":  2.5,
        })

        if flux2:
            ui_defaults.update({
                "embedded_guidance_scale": 1.0 if flux2_klein else 4.0,
                "denoising_strength": 1.0,
                "masking_strength": 0.25,
                "remove_background_images_ref" : 0,
            })
            if flux2_klein:
                ui_defaults["num_inference_steps"] = 4

        if flux_kontext or flux_uso or flux_kontext_dreamomni2:
            ui_defaults.update({
                "video_prompt_type": "KI",
                "denoising_strength": 1.,
            })
        elif flux_umo:
            ui_defaults.update({
                "video_prompt_type": "I",
                "remove_background_images_ref": 0,
            })
        

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        model_mode = inputs.get("model_mode")
        model_mode_int = None
        if model_mode is not None:
            try:
                model_mode_int = int(model_mode)
            except (TypeError, ValueError):
                model_mode_int = None
        if model_mode_int in (2, 3, 4, 5):
            denoising_strength = inputs.get("denoising_strength", 1)
            masking_strength = inputs.get("masking_strength", 1)
            if denoising_strength != 1 or masking_strength != 1:
                gr.Info("LanPaint forces Denoising Strength and Masking Strength to 1; non-1 values will be ignored.")


    @staticmethod
    def custom_prompt_preprocess(prompt, video_guide_outpainting, model_mode, **kwargs):
        if model_mode == 0:
            # from wgp import get_outpainting_dims
            outpainting_ratio = (kwargs.get("video_guide_outpainting_ratio") or "").strip()
            if ((len(video_guide_outpainting) and not video_guide_outpainting.startswith("#") and video_guide_outpainting != "0 0 0 0") or (len(outpainting_ratio) > 0 and not video_guide_outpainting.startswith("#"))):
                if not prompt.endswith("."): prompt += "."
                prompt += "Remove the red paddings on the sides and show what's behind them."
        return prompt  
