# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)

import torch

from shared.utils.hf import build_hf_url
from .prompt_enhancer import IDEOGRAM4_PROMPT_ENHANCER, IDEOGRAM4_PROMPT_INFOS


_PROJECT_REPO = "DeepBeepMeep/Ideogram4"
_PROJECT_FOLDER = "ideogram4"
_TEXT_ENCODER_FOLDER = "Qwen3-VL-8B-Instruct"
_VAE_REPO = "DeepBeepMeep/Flux2"
_VAE_FILENAME = "flux2_vae.safetensors"
_TURBOTIME_MODEL_TYPE = "ideogram4_turbotime"
_PRESET_CHOICES = [
    ("Quality 48", "V4_QUALITY_48"),
    ("Default 20", "V4_DEFAULT_20"),
    ("Turbo 12", "V4_TURBO_12"),
]
_DEFAULT_PRESET = "V4_DEFAULT_20"
_DEFAULT_SAMPLE_SOLVER = "euler"
_DEFAULT_FLOW_SHIFT = 1.0
_SAMPLE_SOLVERS = [
    ("Euler", "euler"),
    ("RES 2M", "res_2m"),
    ("RES 2S", "res_2s"),
]
_SAMPLE_SOLVER_IDS = {value for _, value in _SAMPLE_SOLVERS}
_CUSTOM_SETTINGS = [
    {"id": "ideogram_mu", "label": "Scheduler Mu", "name": "Scheduler Mu", "type": "float", "default": 0.0, "min": -10.0, "max": 10.0, "inc": 0.05},
    {"id": "ideogram_std", "label": "Scheduler Std", "name": "Scheduler Std", "type": "float", "default": 1.75, "min": 0.1, "max": 5.0, "inc": 0.05},
]
IDEOGRAM4_INFOS = """## Ideogram 4

Ideogram 4 is an image model specialized for prompt-following, text rendering, layout, logos, posters, typography, and graphic design.

It was trained on structured JSON captions; use that format for good results. In WanGP, optional Magic Prompt can convert a normal text brief into Ideogram's JSON caption format. Without Magic Prompt, write the structured caption yourself; directly passing plain text is not the upstream-supported prompting workflow.


## Guidance phases

Ideogram 4 uses WanGP guidance phases for CFG switching.

Phase 1 uses Guidance during the high-noise part of denoising. Phase 2 uses Guidance2 after the first switch threshold. If Three Phases is enabled, phase 3 uses Guidance3 after the second switch threshold.

The official presets use two phases: most steps at CFG 7, then final low-noise polish at CFG 3.


## Scheduler custom settings

Scheduler Mu and Scheduler Std control the Ideogram logit-normal sampling schedule.

Mu shifts where denoising samples concentrate along the noise schedule. Std controls how wide that distribution is.

The official presets set these values for you. Change them mainly when matching a Comfy workflow or experimenting with a different schedule.


## LoRA multipliers

Ideogram 4 has two parallel transformers: the conditional transformer and the unconditional transformer.

A plain LoRA multiplier applies to both branches.

To set different values per branch, use ':' as conditional:unconditional. For example, 0.9:0.4 applies 0.9 to the conditional transformer and 0.4 to the unconditional transformer.

Guidance phases still use ';'. For example, 1:1.2;0.8:1.0 means phase 1 cond/uncond = 1/1.2 and phase 2 cond/uncond = 0.8/1.0."""
IDEOGRAM4_TURBOTIME_INFOS = """## Ideogram 4 TurboTime

Ideogram 4 TurboTime uses the Ideogram 4 conditional transformer with the TurboTime LoRA applied. It is designed for a few denoising steps with no CFG and no unconditional transformer branch.

LoRA multipliers apply to the conditional transformer only."""
_PRESET_SETTINGS = {
    "V4_QUALITY_48": {"num_inference_steps": 48, "mu": 0.0, "std": 1.5, "switch_threshold": 184},
    "V4_DEFAULT_20": {"num_inference_steps": 20, "mu": 0.0, "std": 1.75, "switch_threshold": 211},
    "V4_TURBO_12": {"num_inference_steps": 12, "mu": 0.5, "std": 1.75, "switch_threshold": 302},
}
_LEGACY_CFG_OVERRIDE_KEYS = (
    "ideogram_cfg_override_enabled",
    "ideogram_cfg_override_start_percent",
    "ideogram_cfg_override_end_percent",
)
_TURBOTIME_SETTINGS = {
    "num_inference_steps": 8,
    "guidance_phases": 0,
    "guidance_scale": 0,
    "guidance2_scale": 0,
    "guidance3_scale": 0,
    "switch_threshold": 0,
    "switch_threshold2": 0,
    "sample_solver": _DEFAULT_SAMPLE_SOLVER,
    "flow_shift": _DEFAULT_FLOW_SHIFT,
}
_TURBOTIME_CUSTOM_SETTINGS = {
    "ideogram_mu": 0.5,
    "ideogram_std": 1.75,
}


def _time_snr_shift(shift, sigma):
    return shift * sigma / (1.0 + (shift - 1.0) * sigma)


def _percent_to_switch_threshold(percent, shift=1.0):
    return int(round(_time_snr_shift(float(shift), 1.0 - float(percent)) * 1000.0))


def _drop_legacy_cfg_override_settings(custom_settings):
    for key in _LEGACY_CFG_OVERRIDE_KEYS:
        custom_settings.pop(key, None)


def _apply_turbotime_settings(ui_defaults, preserve_user_values=False):
    custom_settings = ui_defaults.get("custom_settings", None)
    if not isinstance(custom_settings, dict):
        custom_settings = {}
    _drop_legacy_cfg_override_settings(custom_settings)
    for key, value in _TURBOTIME_CUSTOM_SETTINGS.items():
        if preserve_user_values:
            custom_settings.setdefault(key, value)
        else:
            custom_settings[key] = value
    for key, value in _TURBOTIME_SETTINGS.items():
        if preserve_user_values and key not in {"guidance_phases", "guidance_scale", "guidance2_scale", "guidance3_scale", "switch_threshold", "switch_threshold2"}:
            ui_defaults.setdefault(key, value)
        else:
            ui_defaults[key] = value
    ui_defaults["custom_settings"] = custom_settings
    ui_defaults["model_mode"] = None


def _apply_preset_settings(ui_defaults, preset_name):
    preset = _PRESET_SETTINGS[preset_name]
    custom_settings = ui_defaults.get("custom_settings", None)
    if not isinstance(custom_settings, dict):
        custom_settings = {}
    custom_settings.update({
        "ideogram_mu": preset["mu"],
        "ideogram_std": preset["std"],
    })
    _drop_legacy_cfg_override_settings(custom_settings)
    ui_defaults.update({
        "num_inference_steps": preset["num_inference_steps"],
        "guidance_phases": 2,
        "guidance_scale": 7.0,
        "guidance2_scale": 3.0,
        "switch_threshold": preset["switch_threshold"],
        "sample_solver": _DEFAULT_SAMPLE_SOLVER,
        "flow_shift": _DEFAULT_FLOW_SHIFT,
        "custom_settings": custom_settings,
    })


def _model_uses_nf4(model_def):
    urls = []
    if isinstance(model_def, dict):
        urls.extend(model_def.get("URLs", []) or [])
        urls.extend(model_def.get("URLs2", []) or [])
    return any("_nf4" in str(url).lower() for url in urls)


def _conditional_transformer_only(base_model_type, model_def=None):
    return base_model_type == _TURBOTIME_MODEL_TYPE or bool((model_def or {}).get("conditional_transformer_only", False))


class family_handler:
    @staticmethod
    def query_model_def(base_model_type, model_def):
        conditional_only = _conditional_transformer_only(base_model_type, model_def)
        text_encoder_filename = "Qwen3-VL-8B-Instruct_nf4.safetensors" if _model_uses_nf4(model_def) else "Qwen3-VL-8B-Instruct_fp8.safetensors"
        model_def_update = {
            "image_outputs": True,
            "device_explicit": True,
            "flux2": True,
            "tiny_vae_architecture": "flux2_klein_4b",
            "vae_upsamplers": {"flux2_vae_pid": [1]},
            "excluded_spatial_upsamplers": ["flux2_pid"],
            "guidance_max_phases": 3,
            "lora_multiplier_branches": ["cond", "uncond"],
            "inference_steps": True,
            "fit_into_canvas_image_refs": 0,
            "profiles_dir": [base_model_type],
            "preset_profiles_dir": ["ideogram4_presets"],
            "sample_solvers": _SAMPLE_SOLVERS,
            "flow_shift": True,
            "custom_settings": [one.copy() for one in _CUSTOM_SETTINGS],
            "infos": IDEOGRAM4_INFOS,
            "specialities": [{"name": name} for name in ("logos", "posters", "typography", "graphic design", "text rendering")],
            "no_negative_prompt": True,
            "no_background_removal": True,
            "skip_prompt_template": True,
            "resolution_budget": "2048x2048",
            "vae_block_size": 16,
            "prompt_enhancer_button_label": "Magic Prompt",
            "prompt_infos": IDEOGRAM4_PROMPT_INFOS,
            "deepy_infos": "### Ideogram 4\n\nText-to-image for logos, posters, typography and graphic design. Trained on structured JSON captions: supply JSON, or optionally enable Magic Prompt to convert a plain-text brief. Direct plain text is not the upstream-supported workflow. Use the selected Quality/Default/Turbo preset; proofread text and layout. TurboTime has its own few-step settings. Sizes: each side 256 to 2048 pixels in multiples of 16, aspect ratio up to 6:1; 2048x2048 is the largest.",
            "deepy_prompt_infos": '### Prompt Format\n\nSupply JSON, not a bare prose prompt. Top-level order: high_level_description (recommended), style_description (optional), compositional_deconstruction (required: background, then elements). Style order: aesthetics, lighting, photo, medium for photos; aesthetics, lighting, medium, art_style otherwise; optional color_palette comes last. Use photo OR art_style. Element order: type, optional bbox, text for text elements only, desc, optional color_palette. Bboxes are [y_min,x_min,y_max,x_max], normalized 0–1000 from top left. Example: {"high_level_description":"A simple market poster.","style_description":{"aesthetics":"bold and cheerful","lighting":"flat","medium":"graphic_design","art_style":"screen print","color_palette":["#FFF4D6","#174A8B"]},"compositional_deconstruction":{"background":"Cream paper","elements":[{"type":"text","bbox":[100,100,300,900],"text":"MARKET","desc":"Large blue sans-serif lettering"}]}}. Quote every intended visible string exactly, escape embedded quotes, and close all JSON delimiters. No Markdown fences in the actual prompt.',
            "prompt_helper_popup_dims": [78, 100],
            "prompt_enhancer_def": {
                "selection": ["T"],
                "labels": {"T": "Ideogram JSON caption from Text Prompt"},
                "default": "",
            },
            "text_prompt_enhancer_instructions": IDEOGRAM4_PROMPT_ENHANCER,
            "image_prompt_enhancer_instructions": IDEOGRAM4_PROMPT_ENHANCER,
            "text_prompt_enhancer_max_tokens": 2048,
            "image_prompt_enhancer_max_tokens": 2048,
            "text_encoder_folder": _TEXT_ENCODER_FOLDER,
            "text_encoder_URLs": [
                build_hf_url(_PROJECT_REPO, _TEXT_ENCODER_FOLDER, text_encoder_filename),
            ],
        }
        if conditional_only:
            model_def_update.update({
                "conditional_transformer_only": True,
                "accelerated": "native",
                "guidance_max_phases": 0,
                "lora_multiplier_phases": 1,
                "preset_profiles_dir": [],
                "infos": IDEOGRAM4_TURBOTIME_INFOS,
            })
            model_def_update.pop("lora_multiplier_branches", None)
        return model_def_update

    @staticmethod
    def query_supported_types():
        return ["ideogram4", _TURBOTIME_MODEL_TYPE]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "ideogram4"

    @staticmethod
    def query_family_infos():
        return {"ideogram4": (1140, "Ideogram")}

    @staticmethod
    def render_prompt_helper(model_type, model_def, prompt_id, popup_id, prompt_elem_id, resolution_elem_id):
        from .prompt_helper import render_prompt_helper

        return render_prompt_helper(model_type, model_def, prompt_id, popup_id, prompt_elem_id, resolution_elem_id)

    @staticmethod
    def get_prompt_helper_css():
        from .prompt_helper import get_prompt_helper_css

        return get_prompt_helper_css()

    @staticmethod
    def get_prompt_helper_javascript():
        from .prompt_helper import get_prompt_helper_javascript

        return get_prompt_helper_javascript()

    @staticmethod
    def get_rgb_factors(base_model_type):
        from shared.RGB_factors import get_rgb_factors

        return get_rgb_factors("flux", sub_family="flux2")

    @staticmethod
    def get_lora_dir(base_model_type):
        return "ideogram4"

    @staticmethod
    def query_retired_files(model_type):
        """Checkpoints that earlier releases of a model type downloaded and its current definition no longer names (Models Manager
        lists them with the model, to delete them)."""
        return {"ideogram4": ["Ideogram4_Transformer_nvfp4.safetensors", "Ideogram4_Unconditional_Transformer_nvfp4.safetensors"]}.get(model_type, [])

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        transformer_configs = ["ideogram4_transformer_config.json"]
        if not _conditional_transformer_only(base_model_type, model_def):
            transformer_configs.append("ideogram4_unconditional_transformer_config.json")
        return [
            {
                "repoId": _VAE_REPO,
                "sourceFolderList": [""],
                "fileList": [[_VAE_FILENAME]],
            },
            {
                "repoId": _PROJECT_REPO,
                "sourceFolderList": [_PROJECT_FOLDER, _TEXT_ENCODER_FOLDER],
                "fileList": [
                    transformer_configs,
                    ["config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja"],
                ],
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
        from .ideogram4_main import model_factory

        pipe_processor = model_factory(
            checkpoint_dir="ckpts",
            model_filename=model_filename,
            model_type=model_type,
            model_def=model_def,
            base_model_type=base_model_type,
            text_encoder_filename=text_encoder_filename,
            dtype=dtype,
            VAE_dtype=VAE_dtype,
            save_quantized=save_quantized,
        )
        pipe = {
            "transformer": pipe_processor.conditional_transformer,
            "text_encoder": pipe_processor.text_encoder,
            "vae": pipe_processor.autoencoder,
        }
        co_tenants_map = {}
        if pipe_processor.unconditional_transformer is not None:
            pipe["transformer2"] = pipe_processor.unconditional_transformer
            co_tenants_map = {
                "transformer": ["transformer2"],
                "transformer2": ["transformer"],
            }
        pipe = {
            "pipe": pipe,
            "coTenantsMap": co_tenants_map,
        }
        return pipe_processor, pipe

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({"image_mode": 1, "model_mode": None, "batch_size": 1})
        if _conditional_transformer_only(base_model_type, model_def):
            _apply_turbotime_settings(ui_defaults)
            return
        _apply_preset_settings(ui_defaults, _DEFAULT_PRESET)

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        if _conditional_transformer_only(base_model_type, model_def):
            _apply_turbotime_settings(ui_defaults, preserve_user_values=True)
            return
        old_mode = ui_defaults.get("model_mode", None)
        if old_mode in _PRESET_SETTINGS:
            _apply_preset_settings(ui_defaults, old_mode)
        elif ui_defaults.get("sample_solver", None) in _PRESET_SETTINGS:
            _apply_preset_settings(ui_defaults, ui_defaults["sample_solver"])
        else:
            custom_settings = ui_defaults.get("custom_settings", None)
            if not isinstance(custom_settings, dict):
                custom_settings = {}
            defaults = _PRESET_SETTINGS[_DEFAULT_PRESET]
            if custom_settings.get("ideogram_cfg_override_enabled", None) not in (None, 0, "0", False) and "ideogram_cfg_override_start_percent" in custom_settings:
                ui_defaults["switch_threshold"] = _percent_to_switch_threshold(custom_settings["ideogram_cfg_override_start_percent"], ui_defaults.get("flow_shift", _DEFAULT_FLOW_SHIFT))
            _drop_legacy_cfg_override_settings(custom_settings)
            custom_settings.setdefault("ideogram_mu", defaults["mu"])
            custom_settings.setdefault("ideogram_std", defaults["std"])
            ui_defaults["custom_settings"] = custom_settings
            ui_defaults.setdefault("num_inference_steps", defaults["num_inference_steps"])
            ui_defaults.setdefault("guidance_scale", 7.0)
            ui_defaults.setdefault("guidance2_scale", 3.0)
            ui_defaults.setdefault("switch_threshold", defaults["switch_threshold"])
            if ui_defaults.get("sample_solver", "") not in _SAMPLE_SOLVER_IDS:
                ui_defaults["sample_solver"] = _DEFAULT_SAMPLE_SOLVER
            ui_defaults.setdefault("flow_shift", _DEFAULT_FLOW_SHIFT)
            ui_defaults.setdefault("guidance_phases", 2)
        ui_defaults["model_mode"] = None
        ui_defaults.setdefault("image_mode", 1)
