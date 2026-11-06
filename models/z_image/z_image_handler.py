# Copyright (c) 2025-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from models.help_shared import merge_help
from .infos import z_image_help
import torch
from shared.utils.hf import build_hf_url


class family_handler:
    @staticmethod
    def query_model_def(base_model_type, model_def):
        z_image_base = base_model_type == "z_image_base"
        guidance_max_phases = 1 if z_image_base else 0
        extra_model_def = {
            "image_outputs": True,
            "dtype": "bf16",  # in FP16 the activations overflow from the first refiner block: black or prompt-less images
            "sage_v_scaling": z_image_base,  # v overflows the fp16 P.V sums of the SageAttention Triton kernels
            "device_explicit": base_model_type != "z_image_control",
            "cuda_graphs": [] if base_model_type in ("z_image_control", "z_image_control2", "z_image_control2_1") else ["transformer"],  # control blocks run control layers inside them
            "guidance_max_phases": guidance_max_phases,
            "fit_into_canvas_image_refs": 0,
            "profiles_dir": [],
            "vae_upsamplers": {"flux_vae_pid": [1, 2]},
            "excluded_spatial_upsamplers": ["flux_pid"],
            "tiny_vae_architecture": "flux",
        }
        text_encoder_folder = "Qwen3"
        extra_model_def["text_encoder_URLs"] = [
            build_hf_url("DeepBeepMeep/Z-Image", text_encoder_folder, "qwen3_bf16.safetensors"),
            build_hf_url("DeepBeepMeep/Z-Image", text_encoder_folder, "qwen3_quanto_bf16_int8.safetensors"),
        ]
        extra_model_def["text_encoder_folder"] = text_encoder_folder

        if base_model_type in ["z_image_control", "z_image_control2", "z_image_control2_1"]:
            extra_model_def["mask_preprocessing"] = {
                "selection":[ ""],
                "visible": False
            }

            extra_model_def["control_net_weight_name"] = "Control"
            extra_model_def["control_net_weight_size"] = 1

            extra_model_def["guide_preprocessing"] = {
                "selection": ["", "PV", "DV", "EV", "V"],
                "labels" : { "V": "Use Z-Image Raw Format"},
            }

        if base_model_type in ["z_image_control2", "z_image_control2_1"]:
            extra_model_def["mask_preprocessing"] = {
                "selection":[ "", "A", "NA"],
                "visible": False, 
            }
            extra_model_def["parent_model_type"] = "z_image_control"

            extra_model_def["inpaint_support"] = True
            extra_model_def["inpaint_video_prompt_type"]= "VA"

            # extra_model_def["image_ref_choices"] = {
            #     "choices":[("No Reference Image",""), ("Image is a Reference Image", "KI")],
            #     "default": "",
            #     "letters_filter": "KI",
            #     "label": "Reference Image for Inpainting",
            #     "visible": True,
            # }
         
        extra_model_def["flow_shift"] = z_image_base
        extra_model_def["NAG"] = base_model_type in ["z_image"]
        if base_model_type in ("z_image", "z_image_base"):
            extra_model_def.update({
                "deepy_infos": "Text-to-image: `prompt` describes the image; `resolution` sets its dimensions.",
                "deepy_prompt_infos": "Describe the finished image: subject/action, composition, setting, lighting and style. Put key details first and exact visible text in quotes. English and Chinese are supported.",
                "infos": "Generate an image from the Text Prompt (`prompt`) at the selected resolution. The prompt supplies the subjects, composition, setting and visual style together.",
                "prompt_infos": 'Write a description of the finished image in natural language: subject and action, composition, background, lighting, colors and medium. For example: "A man in a red coat stands beside a rainy bus stop, reflected neon on the pavement, documentary photograph." Put important details first and spell visible text exactly in quotes. English and Chinese descriptions are supported.',
            })
        if model_def.get("uniform_sigmas", False):
            extra_model_def["infos"] += "\n\n### Steps\n\nThis checkpoint is made for 1 to 4 steps. Choose the number of steps with its **Accelerator** profiles: each one also sets the matching **Shift Scale** and the strengths of the De-distillation LoRA, which cleans up the first step of a 2 to 4 step run. 2 steps is the default, 1 step the fastest. Other step counts or shifts still work but are not what the checkpoint was trained for."
            extra_model_def["deepy_infos"] += " Few-step checkpoint: `num_inference_steps` 1-4 (2 by default). Its accelerator profiles set the matching `flow_shift` (3 / 6 / 8 / 9) and the dedistill LoRA multipliers (`0.5,0` at 2 steps, none at 1 step)."
        return merge_help(extra_model_def, z_image_help(base_model_type))

    @staticmethod
    def query_supported_types():
        return ["z_image", "z_image_base", "z_image_control", "z_image_control2", "z_image_control2_1"]

    @staticmethod
    def query_family_maps():

        models_eqv_map = {
            "z_image_control2_1" : "z_image_control2",
            "z_image_base": "z_image",
        }

        models_comp_map = {}

        return models_eqv_map, models_comp_map

    @staticmethod
    def query_model_family():
        return "z_image"

    @staticmethod
    def query_family_infos():
        return {"z_image": (1120, "Z-Image") }

    @staticmethod
    def get_lora_dir(base_model_type):
        return "z_image"

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        download_def = [
            {
                "repoId": "DeepBeepMeep/Z-Image",
                "sourceFolderList": ["Qwen3", ""],
                "fileList": [                    
                    ["tokenizer.json", "tokenizer_config.json", "vocab.json", "config.json", "merges.txt"],
                    ["ZImageTurbo_VAE_bf16_config.json", "ZImageTurbo_VAE_bf16.safetensors", "ZImageTurbo_scheduler_config.json"],                
                ],
            }
        ]
        return download_def

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
        from .z_image_main import model_factory

        # Detect if this is a control variant (v1 or v2)
        is_control = base_model_type in ["z_image_control", "z_image_control2", "z_image_control2_1"]

        pipe_processor = model_factory(
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
            is_control=is_control,
        )

        pipe = {
            "transformer": pipe_processor.transformer,
            "text_encoder": pipe_processor.text_encoder,
            "vae": pipe_processor.vae,
        }
        return pipe_processor, pipe

    def get_rgb_factors(base_model_type ):
        from shared.RGB_factors import get_rgb_factors
        latent_rgb_factors, latent_rgb_factors_bias = get_rgb_factors("flux")
        return latent_rgb_factors, latent_rgb_factors_bias

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        z_image_base = base_model_type == "z_image_base" 
        ui_defaults["image_mode"] = 1

        if z_image_base:
            ui_defaults.update(
            {
                "guidance_scale": 4,
                "num_inference_steps": 30 ,    
                "flow_shift": 6.0,
            }
            )
        else:
            ui_defaults.update(
                {
                    "guidance_scale": 0,
                    "num_inference_steps": 8,    
                    "NAG_scale": 1.0,
                    "NAG_tau": 3.5,
                    "NAG_alpha": 0.5,
                }
            )

            # Add control defaults for z_image_control and z_image_control2
            if base_model_type in ["z_image_control", "z_image_control2", "z_image_control2_1"]:
                ui_defaults.update(
                    {
                        "control_net_weight":  0.75,
                    }
                )

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        ui_defaults.setdefault("image_mode", 1)
