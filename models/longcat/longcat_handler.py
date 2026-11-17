# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from models.help_shared import merge_help
from .infos import longcat_help
import torch
from shared.utils.hf import build_hf_url


LONGCAT_AVATAR_TYPES = {"longcat_avatar", "longcat_avatar_v1_5"}


class family_handler:
    @staticmethod
    def query_supported_types():
        return ["longcat_video", "longcat_avatar", "longcat_avatar_v1_5"]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "longcat"

    @staticmethod
    def query_family_infos():
        return {"longcat": (60, "LongCat")}

    @staticmethod
    def get_lora_dir(base_model_type):
        if base_model_type == "longcat_avatar":
            return "longcat_avatar"
        if base_model_type == "longcat_avatar_v1_5":
            return "longcat_avatar_v1_5"
        return "longcat"

    @staticmethod
    def query_model_def(base_model_type, model_def):
        extra_model_def = {
            "device_explicit": base_model_type == "longcat_avatar_v1_5",
            "cuda_graphs": ["transformer"],
            "custom_settings": [{
                "id": "longcat_kv_cache", "name": "KV Cache", "label": "KV Cache", "type": "dropdown",
                "default": "Disabled",
                "choices": [("Disabled", "Disabled"), ("INT8 in VRAM", "INT8"), ("INT8 Streamed from RAM", "INT8_RAM"),
                            ("Q4 in VRAM", "Q4"), ("Q4 Streamed from RAM", "Q4_RAM")],
                "info": "May speed up generation with a start image, reference image or video continuation. Streamed from RAM uses less graphics memory. Q4 needs less memory than INT8 but may change fine details. Speed depends on your hardware.",
            }],
            "tiny_vae_architecture": "t2v",
            "frames_minimum": 5,
            "frames_steps": 4,
            "sliding_window": True,
            "guidance_max_phases": 1,
            "image_prompt_types_allowed": "TSVL",
            "video_continuation": True,
            "sample_solvers": [
                ("Auto (Continuation = Enhanced HF)", "auto"),
                ("Default", ""),
                ("Enhanced HF", "enhance_hf"),
                ("Distill", "distill"),
            ],
        }
        text_encoder_folder = "umt5-xxl"
        extra_model_def["text_encoder_URLs"] = [
            build_hf_url("DeepBeepMeep/Wan2.1", text_encoder_folder, "models_t5_umt5-xxl-enc-bf16.safetensors"),
            build_hf_url("DeepBeepMeep/Wan2.1", text_encoder_folder, "models_t5_umt5-xxl-enc-quanto_int8.safetensors"),
        ]
        extra_model_def["text_encoder_folder"] = text_encoder_folder

        if base_model_type == "longcat_video":
            extra_model_def.update(
                {
                    "fps": 15,
                    "profiles_dir": ["longcat_video"],
                }
            )
        elif base_model_type in LONGCAT_AVATAR_TYPES:
            is_v1_5 = base_model_type == "longcat_avatar_v1_5"
            audio_sources = []
            if not model_def.get("multi_speakers_only", False):
                audio_sources.append("A")
            if not model_def.get("one_speaker_only", False):
                audio_sources.extend(["XA", "CAB", "PAB"])
            extra_model_def.update(
                {
                    "fps": 25 if is_v1_5 else 16,
                    "profiles_dir": [base_model_type],
                    "audio_guide_label": "Voice to follow",
                    "audio_guide2_label": "Voice to follow #2",
                    "audio_guidance": True,
                    "any_audio_prompt": True,
                    "returns_audio": True,
                    "audio_prompt_choices": True,
                    "audio_prompt_type_sources": {
                        "selection": audio_sources,
                        "default": audio_sources[0],
                        "label": "Voices",
                        "scale": 3,
                    },
                    "speaker_locations": True,
                    "one_image_ref_only": True,
                    "one_image_ref_only_with_background": True,
                    "image_ref_choices": {
                        "choices": [ ("Anchor Reference Image", "KI")],
                        "letters_filter": "KI",
                        "visible": False,
                        "label": "Anchor Reference Image",
                    },
                    "reference_image_enabled": True,
                    "no_background_removal": True,
                    "image_prompt_types_allowed": "TSVL",
                    "audio_stride": 1 if is_v1_5 else 2,
                }
            )
            if is_v1_5:
                extra_model_def.update(
                    {
                        "sample_solvers": [("Distill", "distill")],
                        "sample_solver": "distill",
                        "distill_only": True,
                        "num_distill_sample_steps": 8,
                        "scheduler_config": "models/longcat/configs/longcat_avatar_v1_5_scheduler.json",
                        "transformer_config": "models/longcat/configs/longcat_avatar_v1_5.json",
                        "audio_encoder_type": "whisper-large-v3",
                        "audio_encoder_folder": "whisper-large-v3",
                        "distill_lora_URL": build_hf_url("DeepBeepMeep/LongCat", "longcat_avatar_v1_5", "dmd_lora.safetensors"),
                    }
                )


        if model_def.get("avatar_version") == "1.6":
            extra_model_def.update({
                "distill_lora_URL": None, "accelerated": "native",
                "audio_encoder_file": "whisper_large_v3_encoder_bf16.safetensors",
                "text_encoder_URLs": [
                    build_hf_url("DeepBeepMeep/Wan2.1", text_encoder_folder, "models_t5_umt5-xxl-enc-bf16.safetensors"),
                    build_hf_url("DeepBeepMeep/LongCat", text_encoder_folder, "models_t5_umt5-xxl-enc-int8_convrot.safetensors"),
                ],
            })
        return merge_help(extra_model_def, longcat_help(base_model_type, model_def.get("avatar_version"),
                          one_speaker_only=model_def.get("one_speaker_only", False),
                          multi_speakers_only=model_def.get("multi_speakers_only", False)))

    @staticmethod
    def resolve_runtime_model_def(model_def, runtime_context):
        if model_def.get("avatar_version") != "1.6":
            return model_def
        quantization = runtime_context.get("text_encoder_quantization", "int8")
        suffix = "int8_convrot" if quantization == "int8" else "bf16"
        return {**model_def, "audio_encoder_file": f"whisper_large_v3_encoder_{suffix}.safetensors"}

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        if base_model_type in LONGCAT_AVATAR_TYPES:
            ui_defaults["video_prompt_type"] = "KI"

    @staticmethod
    def get_rgb_factors(base_model_type):
        from shared.RGB_factors import get_rgb_factors

        return get_rgb_factors("wan")

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        download_def = [
            {
                "repoId": "DeepBeepMeep/Wan2.1",
                "sourceFolderList": ["umt5-xxl"],
                "fileList": [["special_tokens_map.json", "spiece.model", "tokenizer.json", "tokenizer_config.json"]],
            }
        ]
        if base_model_type == "longcat_avatar_v1_5":
            download_def.append(
                {
                    "repoId": "DeepBeepMeep/LongCat",
                    "sourceFolderList": ["whisper-large-v3"],
                    "fileList": [["config.json", "generation_config.json", (model_def or {}).get("audio_encoder_file", "model.safetensors"), "preprocessor_config.json"]],
                }
            )
        elif base_model_type in LONGCAT_AVATAR_TYPES:
            from models.wan.multitalk.assets import query_download_def
            download_def.append(query_download_def(include_readme=True))
        download_def += [
            {
                "repoId": "DeepBeepMeep/Wan2.1",
                "sourceFolderList": [""],
                "fileList": [["Wan2.1_VAE_bf16.safetensors"]],
            }
        ]
        return download_def

    @staticmethod
    def load_model(
        model_filename,
        model_type,
        base_model_type,
        model_def,
        quantizeTransformer=False,
        text_encoder_quantization=None,
        dtype=torch.bfloat16,
        VAE_dtype=torch.float32,
        mixed_precision_transformer=False,
        save_quantized=False,
        submodel_no_list=None,
        text_encoder_filename=None,
        **kwargs,
    ):
        from .longcat_main import LongCatModel

        longcat_model = LongCatModel(
            checkpoint_dir="ckpts",
            model_filename=model_filename,
            model_type=model_type,
            model_def=model_def,
            base_model_type=base_model_type,
            text_encoder_filename=text_encoder_filename,
            quantizeTransformer=quantizeTransformer,
            dtype=dtype,
            VAE_dtype=VAE_dtype,
            mixed_precision_transformer=mixed_precision_transformer,
            save_quantized=save_quantized,
        )

        pipe = {
            "transformer": longcat_model.transformer,
            "vae": longcat_model.vae.model,
            "text_encoder": longcat_model.text_encoder.model,
        }
        if longcat_model.audio_encoder is not None:
            audio_key = "whisper" if getattr(longcat_model, "audio_encoder_name", "") == "whisper-large-v3" else "wav2vec"
            pipe[audio_key] = longcat_model.audio_encoder

        return longcat_model, pipe

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        if base_model_type == "longcat_avatar_v1_5":
            ui_defaults.update(
                {
                    "guidance_scale": 1.0,
                    "num_inference_steps": 8,
                    "audio_guidance_scale": 1.0,
                    "sliding_window_overlap": 13,
                    "sliding_window_size": 93,
                    "video_length": 93,
                    "video_prompt_type": "",
                    "sample_solver": "distill",
                }
            )
            return

        ui_defaults.update(
            {
                "guidance_scale": 4.0,
                "num_inference_steps": 50,
                "audio_guidance_scale": 4.0,
                "sliding_window_overlap": 13,
                "sliding_window_size": 93,                 
            }
        )
        if base_model_type == "longcat_video":
            ui_defaults.update({"video_length": 93})

        if base_model_type in LONGCAT_AVATAR_TYPES:
            ui_defaults.update({"video_length": 93, "video_prompt_type": ""})

        if ui_defaults.get("sample_solver", "") == "":
            ui_defaults["sample_solver"] = "auto"
