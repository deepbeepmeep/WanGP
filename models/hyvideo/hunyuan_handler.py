# Copyright (c) 2025-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from models.help_shared import merge_help
from .infos import hunyuan_help
import torch
from shared.utils.hf import build_hf_url

def test_hunyuan_1_5(base_model_type):
    return base_model_type in ["hunyuan_1_5_t2v", "hunyuan_1_5_i2v", "hunyuan_1_5_upsampler"]

class family_handler():

    @staticmethod
    def set_cache_parameters(cache_type, base_model_type, model_def, inputs, skip_steps_cache):
        resolution = inputs["resolution"]
        width, height = resolution.split("x")
        pixels = int(width) * int(height)

        if cache_type == "mag":
            skip_steps_cache.update({
            "magcache_thresh" : 0,
            "magcache_K" : 2,
            })
            if test_hunyuan_1_5(base_model_type):
                # Hunyuan 1.5 MagCache ratios (from ComfyUI-MagCache project)
                num_steps = inputs.get("num_inference_steps", 20)
                if num_steps > 20:
                    # 40 steps version for longer inference
                    skip_steps_cache.def_mag_ratios = [1.04199, 1.01953, 1.01172, 1.01855, 1.00293, 1.00586, 1.00098, 1.00195, 1.0, 1.00098, 0.99805, 0.99902, 0.99854, 0.99805, 0.99658, 0.99756, 0.99707, 0.99512, 0.99609, 0.99609, 0.99658, 0.99658, 0.99658, 0.99805, 0.99658, 0.99707, 0.99561, 0.99561, 0.99561, 0.99658, 0.99658, 0.99658, 0.99609, 0.99658, 0.99512, 0.99561, 0.99463, 0.99512, 0.99463, 0.99512, 0.99365, 0.99414, 0.99365, 0.99365, 0.99219, 0.99219, 0.99219, 0.99268, 0.99072, 0.99121, 0.99072, 0.99072, 0.98926, 0.98975, 0.9873, 0.98779, 0.98535, 0.98584, 0.9834, 0.9834, 0.97998, 0.97998, 0.97607, 0.97607, 0.96973, 0.96973, 0.96338, 0.96338, 0.9502, 0.9502, 0.93066, 0.93066, 0.896, 0.896, 0.81787, 0.81836]
                else:
                    # 20 steps version (default)
                    skip_steps_cache.def_mag_ratios = [1.01172, 1.00293, 0.9873, 1.0127, 1.01465, 0.98926, 0.99805, 1.00098, 0.99512, 0.99365, 0.99512, 0.99561, 0.99316, 0.99365, 0.99365, 0.99365, 0.99316, 0.99316, 0.99268, 0.99268, 0.9917, 0.9917, 0.99023, 0.99023, 0.98828, 0.98877, 0.98633, 0.98633, 0.9834, 0.9834, 0.97949, 0.97998, 0.97363, 0.97363, 0.96436, 0.96436, 0.94824, 0.94873]
            elif pixels >= 1280* 720:
                skip_steps_cache.def_mag_ratios = [1.0754, 1.27807, 1.11596, 1.09504, 1.05188, 1.00844, 1.05779, 1.00657, 1.04142, 1.03101, 1.00679, 1.02556, 1.00908, 1.06949, 1.05438, 1.02214, 1.02321, 1.03019, 1.00779, 1.03381, 1.01886, 1.01161, 1.02968, 1.00544, 1.02822, 1.00689, 1.02119, 1.0105, 1.01044, 1.01572, 1.02972, 1.0094, 1.02368, 1.0226, 0.98965, 1.01588, 1.02146, 1.0018, 1.01687, 0.99436, 1.00283, 1.01139, 0.97122, 0.98251, 0.94513, 0.97656, 0.90943, 0.85703, 0.75456]
            else:
                skip_steps_cache.def_mag_ratios = [1.06971, 1.29073, 1.11245, 1.09596, 1.05233, 1.01415, 1.05672, 1.00848, 1.03632, 1.02974, 1.00984, 1.03028, 1.00681, 1.06614, 1.05022, 1.02592, 1.01776, 1.02985, 1.00726, 1.03727, 1.01502, 1.00992, 1.03371, 0.9976, 1.02742, 1.0093, 1.01869, 1.00815, 1.01461, 1.01152, 1.03082, 1.0061, 1.02162, 1.01999, 0.99063, 1.01186, 1.0217, 0.99947, 1.01711, 0.9904, 1.00258, 1.00878, 0.97039, 0.97686, 0.94315, 0.97728, 0.91154, 0.86139, 0.76592]
        else:
            skip_steps_cache.coefficients = [7.33226126e+02, -4.01131952e+02,  6.75869174e+01, -3.14987800e+00, 9.61237896e-02]

    @staticmethod
    def query_model_def(base_model_type, model_def):
        extra_model_def = {"device_explicit": base_model_type in ("hunyuan_1_5_t2v", "hunyuan_1_5_i2v"), "tiny_vae_architecture": "hunyuan_1_5_t2v" if test_hunyuan_1_5(base_model_type) else "hunyuan"}
        if base_model_type != "hunyuan_avatar":
            extra_model_def["riflex"] = True

        if test_hunyuan_1_5(base_model_type):
            text_encoder_folder = "Qwen2.5-VL-7B-Instruct"
            extra_model_def["text_encoder_URLs"] = [
                build_hf_url("DeepBeepMeep/Qwen_image", text_encoder_folder, "Qwen2.5-VL-7B-Instruct_bf16.safetensors"),
                build_hf_url("DeepBeepMeep/Qwen_image", text_encoder_folder, "Qwen2.5-VL-7B-Instruct_quanto_bf16_int8.safetensors"),
            ]
            extra_model_def["text_encoder_folder"] = text_encoder_folder
        else:
            text_encoder_folder = "llava-llama-3-8b"
            extra_model_def["text_encoder_URLs"] = [
                build_hf_url("DeepBeepMeep/HunyuanVideo", text_encoder_folder, "llava-llama-3-8b-v1_1_vlm_fp16.safetensors"),
                build_hf_url("DeepBeepMeep/HunyuanVideo", text_encoder_folder, "llava-llama-3-8b-v1_1_vlm_quanto_int8.safetensors"),
            ]
            extra_model_def["text_encoder_folder"] = text_encoder_folder

        if base_model_type in ["hunyuan_avatar", "hunyuan_custom_audio"]:
            fps = 25
        elif base_model_type in ["hunyuan", "hunyuan_i2v", "hunyuan_custom_edit", "hunyuan_custom"] or test_hunyuan_1_5(base_model_type):
            fps = 24
        else:
            fps = 16

        if test_hunyuan_1_5(base_model_type):
            extra_model_def["group"] = "hunyuan_1_5"
            if base_model_type in ["hunyuan_1_5_upsampler"]:
                extra_model_def["profiles_dir"] = [""]
            else:
                extra_model_def["profiles_dir"] = ["hunyuan_1_5"]

        if base_model_type in [ "hunyuan_avatar", "hunyuan_custom_audio"]:
            extra_model_def["any_audio_prompt"] = True
            extra_model_def["returns_audio"] = True

        extra_model_def["fps"] = fps
        extra_model_def["frames_minimum"] = 5
        extra_model_def["frames_steps"] = 4
        extra_model_def["sliding_window"] = False
        extra_model_def["flow_shift"] = True

        if base_model_type in ["hunyuan", "hunyuan_i2v"]:
            extra_model_def["embedded_guidance"] = True
        else:
            extra_model_def["guidance_max_phases"] = 1

        extra_model_def["cfg_star"] =  base_model_type in [ "hunyuan_avatar", "hunyuan_custom_audio", "hunyuan_custom_edit", "hunyuan_custom"] or test_hunyuan_1_5(base_model_type)
        extra_model_def["tea_cache"] = not test_hunyuan_1_5(base_model_type)
        extra_model_def["mag_cache"] = True  # Enabled for all Hunyuan models including 1.5

        if base_model_type in ["hunyuan_custom_edit"]:
            extra_model_def["guide_preprocessing"] = {
                "selection": ["MV", "PV"],
            }

            extra_model_def["mask_preprocessing"] = {
                "selection": ["A", "NA"],
                "default" : "NA"
            }

        if base_model_type in ["hunyuan_custom_audio", "hunyuan_custom_edit", "hunyuan_custom"]:
            extra_model_def["image_ref_choices"] = {
                "choices": [("Reference Image", "I")],
                "letters_filter":"I",
                "visible": False,
            }

        if base_model_type in ["hunyuan_avatar"]: 
            extra_model_def["image_ref_choices"] = {
                "choices": [("Start Image", "KI")],
                "letters_filter":"KI",
                "visible": False,
            }
            extra_model_def["no_background_removal"] = True

        if base_model_type in ["hunyuan_custom", "hunyuan_custom_edit", "hunyuan_custom_audio", "hunyuan_avatar"]:
            extra_model_def["one_image_ref_needed"] = True


        if base_model_type in ["hunyuan_i2v", "hunyuan_1_5_i2v"]:
            extra_model_def["image_prompt_types_allowed"] = "SVL"

        if base_model_type in ["hunyuan_1_5_upsampler"]:
            extra_model_def["image_prompt_types_allowed"] = "TVL"
            extra_model_def["guide_custom_choices"] = {
            "choices":[
                ("Upsample", "V"),
            ],
            "default": "V",
            "letters_filter": "V",
            "label": "Type of Process",
            "scale": 3,
            "show_label" : False,
            "visible": False,
            }
        if test_hunyuan_1_5(base_model_type) and base_model_type not in ["hunyuan_1_5_t2v"]:
            extra_model_def["sliding_window"] = True
            extra_model_def["sliding_window_defaults"] = { "overlap_min" : 1, "overlap_max" : 1, "overlap_step": 0, "overlap_default": 1}


        return merge_help(extra_model_def, hunyuan_help(base_model_type))

    @staticmethod
    def query_supported_types():
        return ["hunyuan", "hunyuan_i2v", "hunyuan_custom", "hunyuan_custom_audio", "hunyuan_custom_edit", "hunyuan_avatar", "hunyuan_1_5_t2v", "hunyuan_1_5_i2v", "hunyuan_1_5_upsampler"]

    @staticmethod
    def query_family_maps():
        models_eqv_map = {
                    "hunyuan_1_5_t2v": "hunyuan",
                    "hunyuan_1_5_i2v": "hunyuan_i2v",
        }

        models_comp_map = { 
                    "hunyuan_custom":  ["hunyuan_custom_edit", "hunyuan_custom_audio"],
                    "hunyuan": ["hunyuan_1_5_t2v"],
                    "hunyuan_i2v": ["hunyuan_1_5_i2v"],
                    }

        return models_eqv_map, models_comp_map

    @staticmethod
    def query_model_family():
        return "hunyuan"

    @staticmethod
    def query_family_infos():
        return {"hunyuan":(20, "Hunyuan Video"), "hunyuan_1_5":(21, "Hunyuan Video 1.5")}

    @staticmethod
    def get_lora_dir(base_model_type):
        if test_hunyuan_1_5(base_model_type):
            return "hunyuan_1_5"
            
        elif "i2v" in base_model_type:
            return "hunyuan_i2v"

        return "hunyuan"

    @staticmethod
    def get_rgb_factors(base_model_type ):
        from shared.RGB_factors import get_rgb_factors
        latent_rgb_factors, latent_rgb_factors_bias = get_rgb_factors("hunyuan", sub_family = "hunyuan1.5" if test_hunyuan_1_5(base_model_type) else "")
        return latent_rgb_factors, latent_rgb_factors_bias

    @staticmethod
    def query_retired_files(model_type):
        """Checkpoints that earlier releases of a model type downloaded and its current definition no longer names (Models Manager
        lists them with the model, to delete them)."""
        return {
            "hunyuan": ["hunyuan_video_720_bf16.safetensors.safetensors"],
            "hunyuan_1_5_upsampler": ["hunyuan_video_1.5_upsampler_720_quanto_int8_bf16.safetensors"],
        }.get(model_type, [])

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        if test_hunyuan_1_5(base_model_type):
            download_def = [{  
            "repoId" : "DeepBeepMeep/Qwen_image", 
            "sourceFolderList" :  ["", "Qwen2.5-VL-7B-Instruct"],
            "fileList" : [ ["qwen_vae.safetensors", "qwen_vae_config.json"], ["merges.txt", "tokenizer_config.json", "config.json", "vocab.json", "video_preprocessor_config.json", "preprocessor_config.json", "chat_template.json"]  ]
            },
            {  
            "repoId" : "DeepBeepMeep/HunyuanVideo1.5", 
            "sourceFolderList" :  [ "Glyph-SDXL-v2", "Glyph-SDXL-v2/byt5-small",  "siglip_vision_model", ""  ],
            "fileList" :[ ["color_idx.json", "multilingual_10-lang_idx.json"] ,
                        [ "config.json", "model.safetensors", "byt5_model.safetensors"], # "byt5_model.pt", "pytorch_model.bin"],
                        [ "model.safetensors", "config.json", "preprocessor_config.json"],
                        [  "hunyuan_video_1_5_VAE_fp32.safetensors", "hunyuan_video_1_5_VAE.json"] ,
                        ]
            } ]

        else:   
            download_def= {  
            "repoId" : "DeepBeepMeep/HunyuanVideo", 
            "sourceFolderList" :  [ "llava-llama-3-8b", "clip_vit_large_patch14",  "whisper-tiny" , ""  ],
            "fileList" :[ ["config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"],
                            ["text_config.json", "merges.txt", "model.safetensors", "preprocessor_config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json"],
                            ["config.json", "model.safetensors", "preprocessor_config.json", "special_tokens_map.json", "tokenizer_config.json"],
                            [ "hunyuan_video_720_quanto_int8_map.json", "hunyuan_video_custom_VAE_fp32.safetensors", "hunyuan_video_custom_VAE_config.json", "hunyuan_video_VAE_fp32.safetensors", "hunyuan_video_VAE_config.json" , "hunyuan_video_720_quanto_int8_map.json"   ]  
                            ]
            }
            from .data_kits.assets import query_download_def
            download_def = [download_def, query_download_def()]
        return download_def 

    @staticmethod
    def load_model(model_filename, model_type = None,  base_model_type = None, model_def = None, quantizeTransformer = False, text_encoder_quantization = None, dtype = torch.bfloat16, VAE_dtype = torch.float32, mixed_precision_transformer = False, save_quantized = False, submodel_no_list = None, text_encoder_filename = None, **kwargs):
        from .hunyuan import HunyuanVideoSampler
        from mmgp import offload

        hunyuan_model = HunyuanVideoSampler.from_pretrained(
            model_filepath = model_filename,
            model_type = model_type, 
            base_model_type = base_model_type,
            model_def = model_def,
            text_encoder_filepath = text_encoder_filename,
            dtype = dtype,
            quantizeTransformer = quantizeTransformer,
            VAE_dtype = VAE_dtype, 
            mixed_precision_transformer = mixed_precision_transformer,
            save_quantized = save_quantized
        )

        pipe = { "transformer" : hunyuan_model.model, "text_encoder" : hunyuan_model.text_encoder, "text_encoder_2" : hunyuan_model.text_encoder_2, "vae" : hunyuan_model.vae  }
        if hunyuan_model.byt5_model is not None:
            pipe["byt5_model"] = hunyuan_model.byt5_model

        if hunyuan_model.vision_encoder is not None:
            pipe["vision_encoder"] = hunyuan_model.vision_encoder

        if hunyuan_model.upsampler is not None:
            pipe["upsampler"] = hunyuan_model.upsampler

        if hunyuan_model.wav2vec is not None:
            pipe["wav2vec"] = hunyuan_model.wav2vec


        # if hunyuan_model.align_instance != None:
        #     pipe["align_instance"] = hunyuan_model.align_instance.facedet.model


        from .modules.models import get_linear_split_map

        split_linear_modules_map = get_linear_split_map()
        hunyuan_model.model.split_linear_modules_map = split_linear_modules_map
        offload.split_linear_modules(hunyuan_model.model, split_linear_modules_map )


        return hunyuan_model, pipe

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        if settings_version<2.33:
            if base_model_type in ["hunyuan_custom_edit"]:
                video_prompt_type=  ui_defaults["video_prompt_type"]
                if "P" in video_prompt_type and "M" in video_prompt_type: 
                    video_prompt_type = video_prompt_type.replace("M","")
                    ui_defaults["video_prompt_type"] = video_prompt_type  

        if settings_version < 2.36:
            if base_model_type in ["hunyuan_avatar", "hunyuan_custom_audio"]:
                audio_prompt_type=  ui_defaults["audio_prompt_type"]
                if "A" not in audio_prompt_type:
                    audio_prompt_type += "A"
                    ui_defaults["audio_prompt_type"] = audio_prompt_type  


        if settings_version < 2.41:
            if base_model_type in ["hunyuan_i2v"]:
                ui_defaults["sliding_window_overlap"] = 1

        
    
    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults["embedded_guidance_scale"]= 6.0

        if base_model_type in ["hunyuan","hunyuan_i2v"]:
            ui_defaults.update({
                "guidance_scale": 7.0,
            })
    
        elif base_model_type in ["hunyuan_custom"]:
            ui_defaults.update({
                "guidance_scale": 7.5,
                "flow_shift": 13,
                "resolution": "1280x720",
                "video_prompt_type": "I",
            })
        elif base_model_type in ["hunyuan_custom_audio"]:
            ui_defaults.update({
                "guidance_scale": 7.5,
                "flow_shift": 13,
                "video_prompt_type": "I",
                "audio_prompt_type": "A",
            })
        elif base_model_type in ["hunyuan_custom_edit"]:
            ui_defaults.update({
                "guidance_scale": 7.5,
                "flow_shift": 13,
                "video_prompt_type": "MVAI",
                "sliding_window_size": 129,
            })
        elif base_model_type in ["hunyuan_avatar"]:
            ui_defaults.update({
                "guidance_scale": 7.5,
                "flow_shift": 5,
                "remove_background_images_ref": 0,
                "skip_steps_start_step_perc": 25, 
                "video_length": 129,
                "video_prompt_type": "KI",
                "audio_prompt_type": "A",
            })

        if base_model_type in ["hunyuan_1_5_i2v","hunyuan_i2v"]:
            ui_defaults.update({
                "image_prompt_type": "S",
                "sliding_window_overlap" : 1,
            })

        if base_model_type in ["hunyuan_1_5_upsampler"]:
            ui_defaults.update({
                "video_prompt_type": "V",
                "sliding_window_overlap" : 1,
            })
