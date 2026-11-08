# Copyright (c) 2025-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from models.help_shared import merge_help
from .infos import LTXV_HELP
import torch
from shared.utils.hf import build_hf_url

class family_handler():
    @staticmethod
    def query_model_def(base_model_type, model_def):
        LTXV_config = model_def.get("LTXV_config", "")
        distilled= "distilled" in LTXV_config 
        extra_model_def = {"device_explicit": distilled}
        if distilled:
            extra_model_def.update({
            "lock_inference_steps": True,
            "no_negative_prompt" : True,
        })

        text_encoder_folder = "T5_xxl_1.1"
        extra_model_def["text_encoder_URLs"] = [
            build_hf_url("DeepBeepMeep/LTX_Video", text_encoder_folder, "T5_xxl_1.1_enc_bf16.safetensors"),
            build_hf_url("DeepBeepMeep/LTX_Video", text_encoder_folder, "T5_xxl_1.1_enc_quanto_bf16_int8.safetensors"),
        ]
        extra_model_def["text_encoder_folder"] = text_encoder_folder

        extra_model_def["fps"] = 30
        extra_model_def["frames_minimum"] = 17
        extra_model_def["frames_steps"] = 8
        extra_model_def["sliding_window"] = True
        extra_model_def["image_prompt_types_allowed"] = "TSEV"

        extra_model_def["guide_preprocessing"] = {
            "selection": ["", "PV", "DV", "EV", "V"],
            "labels" : { "V": "Use LTXV raw format"}
        }

        extra_model_def["mask_preprocessing"] = {
            "selection": ["", "A", "NA", "XA", "XNA"],
        }

        extra_model_def["extra_control_frames"] = 1
        extra_model_def["dont_cat_preguide"]= True
        extra_model_def["vae_block_size"] = 32
        extra_model_def["multiple_images_as_text_prompts"] = True
        return merge_help(extra_model_def, LTXV_HELP)

    @staticmethod
    def query_supported_types():
        return ["ltxv_13B"]
    
    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def get_rgb_factors(base_model_type ):
        from shared.RGB_factors import get_rgb_factors
        latent_rgb_factors, latent_rgb_factors_bias = get_rgb_factors("ltxv")
        return latent_rgb_factors, latent_rgb_factors_bias

    @staticmethod
    def query_model_family():
        return "ltxv"

    @staticmethod
    def query_family_infos():
        return {"ltxv":(30, "LTX Video")}

    @staticmethod
    def get_lora_dir(base_model_type):
        return "ltxv"

    @staticmethod
    def query_retired_files(model_type):
        """Checkpoints that earlier releases of a model type downloaded and its current definition no longer names (Models Manager
        lists them with the model, to delete them)."""
        return {
            "ltxv_13B": ["ltxv_0.9.7_13B_dev_quanto_bf16_int8.safetensors"],
            "ltxv_distilled": ["ltxv_0.9.8_13B_distilled_dev_bf16.safetensors"],
        }.get(model_type, [])

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return {
            "repoId" : "DeepBeepMeep/LTX_Video", 
            "sourceFolderList" :  ["T5_xxl_1.1",  ""  ],
            "fileList" : [ ["added_tokens.json", "special_tokens_map.json", "spiece.model", "tokenizer_config.json"], ["ltxv_0.9.7_VAE.safetensors", "ltxv_0.9.7_spatial_upscaler.safetensors", "ltxv_scheduler.json"] ]   
        }


    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, quantizeTransformer = False, text_encoder_quantization = None, dtype = torch.bfloat16, VAE_dtype = torch.float32, mixed_precision_transformer = False, save_quantized = False, submodel_no_list = None, text_encoder_filename = None, **kwargs):
        from .ltxv import LTXV

        ltxv_model = LTXV(
            model_filepath = model_filename,
            text_encoder_filepath = text_encoder_filename,
            model_type = model_type, 
            base_model_type = base_model_type,
            model_def = model_def,
            dtype = dtype,
            # quantizeTransformer = quantizeTransformer,
            VAE_dtype = VAE_dtype, 
            mixed_precision_transformer = mixed_precision_transformer
        )

        pipeline = ltxv_model.pipeline 
        pipe = {"transformer" : pipeline.video_pipeline.transformer, "vae" : pipeline.vae, "text_encoder" : pipeline.video_pipeline.text_encoder, "latent_upsampler" : pipeline.latent_upsampler}

        return ltxv_model, pipe

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        pass
   
