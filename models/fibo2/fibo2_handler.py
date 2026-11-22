# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from shared.utils.hf import build_hf_url
from .help import INFOS, PROMPT_INFOS, DEEPY_INFOS, DEEPY_PROMPT_INFOS, TURBO_INFOS, TURBO_PROMPT_INFOS, TURBO_DEEPY_INFOS, TURBO_DEEPY_PROMPT_INFOS
from .prompts import CLASSIC, STRUCTURED, EDIT, STRUCTURED_EDIT, RMBG3_ENHANCER, RMBG3_PROMPT

REPO = "DeepBeepMeep/Fibo2"
TEXT_FOLDER = "Qwen3-VL-4B-Instruct"


class family_handler:
    @staticmethod
    def query_supported_types():
        return ["fibo2_dev", "fibo2_edit_turbo"]

    @staticmethod
    def query_model_family():
        return "fibo2"

    @staticmethod
    def query_family_infos():
        return {"fibo2": (1165, "Fibo 2.0")}

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def get_lora_dir(base_model_type):
        return "fibo2"

    @staticmethod
    def query_model_def(base_model_type, model_def):
        turbo = base_model_type == "fibo2_edit_turbo"
        result = {
            "image_outputs": True, "device_explicit": True, "dtype": "bf16",
            "profiles_dir": ["fibo2"], "guidance_max_phases": 1, "embedded_guidance": False,
            "inference_steps": True, "no_negative_prompt": False, "no_background_removal": True,
            "vae_block_size": 16, "resolution_budget": "1152x1024", "fit_into_canvas_image_refs": 0,
            "max_width": 8192, "max_height": 8192,
            "tiny_vae_architecture": "flux2_klein_4b",
            "skip_prompt_template": True, "preserve_empty_prompt_lines": True,
            "text_encoder_folder": TEXT_FOLDER,
            "text_encoder_URLs": [build_hf_url(REPO, TEXT_FOLDER, f"Qwen3-VL-4B-Instruct_{fmt}.safetensors") for fmt in ("bf16", "int8_convrot")],
            "image_ref_choices": {"choices": [("Generate without Reference Images", ""),
                                             ("First Reference Defines Output Dimensions (Main Image Ref)", "KI"),
                                             ("Use References with Selected Output Dimensions", "I")],
                                  "letters_filter": "KI", "default": "", "label": "Reference Images"},
            "at_least_one_image_ref_needed": False,
            "inpaint_support": True, "inpaint_video_prompt_type": "VA", "inpaint_color": "808080", "uint8_guides": True,
            "guide_custom_choices_image": {"choices": [("No Control Image", ""), ("Control Image", "V")],
                                           "letters_filter": "V", "default": "", "visible": False},
            "infos": INFOS, "prompt_infos": PROMPT_INFOS, "deepy_infos": DEEPY_INFOS, "deepy_prompt_infos": DEEPY_PROMPT_INFOS,
            "prompt_enhancer_def": {"selection": ["T", "TI", "T1", "TI1"], "default": "",
                                    "labels": {"T": "A Detailed Image Caption", "TI": "An Editing Instruction using {image_inputs}",
                                               "T1": "A Structured Fibo Caption", "TI1": "A Structured Fibo Edit using {image_inputs}"}},
            "text_prompt_enhancer_instructions": CLASSIC, "image_prompt_enhancer_instructions": EDIT,
            "text_prompt_enhancer_instructions1": STRUCTURED, "image_prompt_enhancer_instructions1": STRUCTURED_EDIT,
            "text_prompt_enhancer_max_tokens1": 4096, "image_prompt_enhancer_max_tokens1": 4096,
        }
        if turbo:
            result.update({
                "accelerated": "native",
                "capability_overrides": {"text_to_image": False},
                "profiles_dir": ["fibo2_edit_turbo"], "guidance_max_phases": 0, "no_negative_prompt": True,
                "infos": TURBO_INFOS, "prompt_infos": TURBO_PROMPT_INFOS,
                "deepy_infos": TURBO_DEEPY_INFOS, "deepy_prompt_infos": TURBO_DEEPY_PROMPT_INFOS,
            })
            result["image_ref_choices"]["choices"] = result["image_ref_choices"]["choices"][1:] + [("No Reference Images (Masked Editing)", "")]
            result["image_ref_choices"]["default"] = "KI"
            result["prompt_enhancer_def"]["selection"] = ["TI", "TI1"]
        if model_def.get("matting"):
            result.update({
                "prompt_enhancer_def": {"selection": ["T"], "default": "", "labels": {"T": "Background Removal Prompt"}},
                "text_prompt_enhancer_instructions": RMBG3_ENHANCER,
                "image_prompt_enhancer_instructions": RMBG3_ENHANCER,
                "text_prompt_enhancer_instructions1": RMBG3_ENHANCER,
                "image_prompt_enhancer_instructions1": RMBG3_ENHANCER,
                "text_prompt_enhancer_max_tokens": 512, "image_prompt_enhancer_max_tokens": 512,
                "text_prompt_enhancer_max_tokens1": 512, "image_prompt_enhancer_max_tokens1": 512,
                "prompt_enhancer_video_duration": False,
                "custom_settings": [{"id": "rgba", "name": "RGBA", "label": "RGBA Cutout", "type": "dropdown", "default": "Disabled",
                                     "choices": [("Disabled (Grayscale Matte)", "Disabled"), ("Enabled (Transparent PNG Cutout)", "Enabled")],
                                     "info": "Disabled saves the grayscale alpha matte. Enabled saves the source image as an RGBA PNG whose alpha is that matte, gray levels included, so soft edges stay partly transparent. Requires the Main Image Ref."}],
                "specialities": [{"name": "alpha output", "aliases": ["RGBA", "transparent background", "background removal", "cutout"],
                                  "description": "Set custom_settings.rgba to Enabled to save the source image as an RGBA PNG cutout whose alpha is the generated matte (soft, partial transparency kept)."}],
                "infos": "RMBG-3.0 generates a grayscale alpha matte from a reference image in four steps. White keeps the subject, black removes the background, and gray preserves partial opacity. Use the default fixed prompt.\n\n**RGBA Cutout:** Disabled saves the matte itself. Enabled saves the source image as a transparent PNG: the matte becomes its alpha channel, so hair, blur and glass stay partly transparent. Semi-transparent edges keep a trace of the original background color. The cutout needs the Main Image Ref (the source sets the output size).\n\nWeights: CC BY-NC 4.0; commercial use requires a separate Bria license.",
                "prompt_infos": "Use the default matte instruction unchanged. The optional Background Removal Prompt enhancer restores that fixed instruction; it does not describe or edit the scene.",
                "deepy_infos": 'Alpha-matte generation: model_type="fibo2_rmbg3", image_mode=1, one image_refs source, video_prompt_type="KI", num_inference_steps=4, guidance_scale=1. Keep prompt_enhancer off unless requested. Default output is the grayscale matte; custom_settings.rgba="Enabled" saves the source image as an RGBA PNG cutout with the matte as alpha (partial transparency kept; edges keep a trace of the old background color; requires video_prompt_type="KI").',
                "deepy_prompt_infos": "Use this exact fixed prompt for every reference image: " + RMBG3_PROMPT,
            })
        return result

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return [{"repoId": REPO, "sourceFolderList": [TEXT_FOLDER, "fibo2"],
                 "fileList": [["fibo2_encoder_config.json", "fibo2_processor_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"],
                              ["flux2_vae_fp16.safetensors"]]}]

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, text_encoder_filename=None, save_quantized=False, **kwargs):
        from .pipeline import load_pipeline
        pipeline = load_pipeline(model_filename, text_encoder_filename, model_def, model_type, save_quantized)
        return pipeline, {"transformer": pipeline.transformer, "text_encoder": pipeline.text_encoder, "vae": pipeline.vae}

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({"image_mode": 1, "resolution": "1024x1024", "batch_size": 1, "num_inference_steps": 50,
                            "guidance_scale": 5.0, "prompt_enhancer": "", "video_prompt_type": "", "multi_prompts_gen_type": "FG",
                            "remove_background_images_ref": 0, "denoising_strength": 1.0, "masking_strength": 1.0})
        if base_model_type == "fibo2_edit_turbo":
            ui_defaults.update(num_inference_steps=4, guidance_scale=1.0, video_prompt_type="KI")

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        references = inputs.get("image_refs") or []
        if len(references) > 4:
            return "Fibo 2.0 supports up to four ordered reference images."
        if inputs["image_mode"] == 2 and references:
            return "Masked editing uses one Control Image and its mask; remove additional Reference Images."
        if base_model_type == "fibo2_edit_turbo":
            if inputs["image_mode"] != 2 and (not references or "I" not in inputs.get("video_prompt_type", "")):
                return "Fibo 2.0 Edit Turbo requires Reference Images or a Control Image with a mask. Use Dev for text-only generation."
            inputs["guidance_scale"] = 1.0
        if model_def.get("matting") and (inputs.get("custom_settings") or {}).get("rgba") == "Enabled":
            if inputs["image_mode"] == 2 or "K" not in inputs.get("video_prompt_type", "") or len(references) != 1:
                return "RGBA Cutout needs exactly one source image as Main Image Ref (First Reference Defines Output Dimensions)."
        inputs["denoising_strength"] = inputs["masking_strength"] = 1.0

    @staticmethod
    def get_rgb_factors(base_model_type):
        from shared.RGB_factors import get_rgb_factors
        return get_rgb_factors("flux", sub_family="flux2")
