# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from models.help_shared import merge_help
from .infos import HIDREAM_HELP

import torch
from PIL import Image
from .prompt_enhancer import HIDREAM_PROMPT_ENHANCER_INSTRUCTIONS


_PROJECT_REPO = "DeepBeepMeep/HiDream"
_ASSET_FOLDER = "hidream_o1"
_ASSET_FILES = [
    "chat_template.json",
    "config.json",
    "configuration.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
]


class family_handler:
    @staticmethod
    def query_model_def(base_model_type, model_def):
        is_dev = base_model_type == "hidream_o1_dev"
        return merge_help({
            "image_outputs": True,
            "device_explicit": True,
            "no_model_swap": True,  # one managed model: kept loaded between samples and queued tasks of the same model
            "sample_solvers":[("Flash", "flash")] if is_dev else [("Default", "default")],
            "guidance_max_phases": 0 if is_dev else 1,
            "fit_into_canvas_image_refs": 0,
            "profiles_dir": [base_model_type],
            "flow_shift": True,
            "no_negative_prompt": True,
            "no_background_removal": True,
            "processor_folder": _ASSET_FOLDER,
            "vae_block_size": 32,
            "text_prompt_enhancer_instructions": HIDREAM_PROMPT_ENHANCER_INSTRUCTIONS,
            "image_prompt_enhancer_instructions": HIDREAM_PROMPT_ENHANCER_INSTRUCTIONS,
            "text_prompt_enhancer_max_tokens": 512,
            "image_prompt_enhancer_max_tokens": 512,
            "guide_preprocessing": {
                "selection": ["", "V", "PV", "DV", "EV"],
                "labels": {"V": "Use Control Image Unchanged"},
            },
            "image_ref_choices": {
                "choices": [
                    ("None", ""),
                    ("Conditional Image is first Main Subject / Landscape and may be followed by People / Objects", "KI"),
                    ("Conditional Images are References", "I"),
                ],
                "letters_filter": "KI",
                "default": "",
            },
        }, HIDREAM_HELP)

    @staticmethod
    def query_supported_types():
        return ["hidream_o1", "hidream_o1_dev"]

    @staticmethod
    def query_family_maps():
        return {}, {"hidream_o1": ["hidream_o1", "hidream_o1_dev"]}

    @staticmethod
    def query_model_family():
        return "hidream"

    @staticmethod
    def query_family_infos():
        return {"hidream": (1130, "HiDream")}

    @staticmethod
    def get_lora_dir(base_model_type):
        return "hidream_o1"

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return [
            {
                "repoId": _PROJECT_REPO,
                "sourceFolderList": [_ASSET_FOLDER],
                "fileList": [_ASSET_FILES],
            }
        ]

    @staticmethod
    def load_model(
        model_filename,
        model_type=None,
        base_model_type=None,
        model_def=None,
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
        from .hidream_main import model_factory

        pipe_processor = model_factory(
            checkpoint_dir="ckpts",
            model_filename=model_filename,
            model_type=model_type,
            model_def=model_def,
            base_model_type=base_model_type,
            quantizeTransformer=quantizeTransformer,
            dtype=dtype,
            save_quantized=save_quantized,
        )
        return pipe_processor, {"transformer": pipe_processor.transformer}

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        if base_model_type == "hidream_o1_dev":
            ui_defaults.update({
                "guidance_scale": 0,
                "num_inference_steps": 28,
                "sample_solver": "flash",
                "flow_shift": 1.0,
            })
        else:
            ui_defaults.update({
                "guidance_scale": 5,
                "num_inference_steps": 50,
                "sample_solver": "default",
                "flow_shift": 3.0,
            })

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        if base_model_type == "hidream_o1_dev" and ui_defaults.get("sample_solver", "") in ("", "default"):
            ui_defaults["sample_solver"] = "flash"
        elif ui_defaults.get("sample_solver", "") == "":
            ui_defaults["sample_solver"] = "default"

    @staticmethod
    def preview_latents(base_model_type, latents, meta):
        if not torch.is_tensor(latents) or latents.dim() != 4 or latents.shape[0] != 3:
            return None
        image = latents.detach().float().cpu().clamp(-1, 1)
        channels, frames, height, width = image.shape
        image = image.permute(0, 2, 1, 3).reshape(channels, height, frames * width)
        image = image.add(1).mul(127.5).clamp(0, 255).to(torch.uint8)
        preview = Image.fromarray(image.permute(1, 2, 0).numpy())
        if preview.height > 0:
            scale = 200 / preview.height
            resampling_module = getattr(Image, "Resampling", Image)
            preview = preview.resize((max(1, int(round(preview.width * scale))), 200), resample=getattr(resampling_module, "BILINEAR", Image.BILINEAR))
        return preview
