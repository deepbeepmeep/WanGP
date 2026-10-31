# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import os
import shutil
import sys
import torch
from shared.lora_paths import resolve_lora_dir
from shared.utils import files_locator as fl
from shared.utils.hf import build_hf_url
from shared.utils.loras_mutipliers import parse_loras_multipliers
import gradio as gr
from pathlib import Path

from .infos import LTX2_25_DEEPY_INFOS, LTX2_25_INFOS, LTX2_25_MSR_INFOS, LTX2_INFOS, LTX2_MSR_INFOS, LTX2_MSR_V2_INFOS, LTX2_SOL_INFOS
from .ltx_core.model.transformer.sol_attention import SOL_ATTN_TAU_START_DEFAULT
from .lora_utils import control_video_phase2_message
from .ltx2_runtime import LTX2_OUTPAINTING_METHOD

_GEMMA_FOLDER_URL = "https://huggingface.co/DeepBeepMeep/LTX-2/resolve/main/gemma-3-12b-it-qat-q4_0-unquantized/"
_GEMMA_FOLDER = "gemma-3-12b-it-qat-q4_0-unquantized"
_GEMMA_FILENAME = f"{_GEMMA_FOLDER}.safetensors"
_GEMMA_QUANTO_FILENAME = f"{_GEMMA_FOLDER}_quanto_bf16_int8.safetensors"
_GEMMA4_FOLDER = "gemma4-12b-ltx-v1"
_GEMMA4_FILENAME = f"{_GEMMA4_FOLDER}_bf16.safetensors"
_GEMMA4_INT8_FILENAME = f"{_GEMMA4_FOLDER}_int8_convrot.safetensors"
_PRUNAAI_VAE_FILENAME = "ltx-2.3-22b_PrunaAI_vae.safetensors"
_PRUNAAI_VAE_CONFIG_FILENAME = "ltx-2.3-22b_PrunaAI_vae.json"
_NAD_VAE_FILENAME = "ltx-2.5-22b_diffusion_video_vae_bf16.safetensors"
_MSR25_SLOT_FILENAME = "LTX-2.5-Licon-MSR-V1_slot_embeddings_bf16.safetensors"
_GEMMA_TOKENIZER_FILES = [
    "added_tokens.json",
    "chat_template.json",
    "config_light.json",
    "generation_config.json",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
]
_GEMMA4_TOKENIZER_FILES = ["config.json", "chat_template.jinja", "tokenizer.json", "tokenizer_config.json"]
_LORAS_MIGRATED = False
_LORA_SPEC_KEYS = ("distilled_lora", "distilled_1_1_lora", "pixel_spatial_upscaler_lora", "refine_details_lora", "union_control_lora", "id_lora", "outpaint_lora", "inpaint_lora", "ingredients_lora", "hdr_lora")
_SYSTEM_LORA_SPEC_KEYS = {
    "distilled": "distilled_lora",
    "distilled_1_1": "distilled_1_1_lora",
    "pixel_spatial_upscaler": "pixel_spatial_upscaler_lora",
    "refine_details": "refine_details_lora",
    "union_control": "union_control_lora",
    "id": "id_lora",
    "outpaint": "outpaint_lora",
    "inpaint": "inpaint_lora",
    "ingredients": "ingredients_lora",
    "hdr": "hdr_lora",
}
_EDITANYTHING_MODEL_DEF = {
    "ltx2_edit_anything": True,
    "ltx2_edit_anything_ref": True,
    "ltx2_edit_anything_ref_start_block": 12,
    "ltx2_edit_anything_ref_end_block": 35,
    "ltx2_edit_anything_ref_context_scale": 0.01,
    "ltx2_edit_anything_ref_token_scale": 0.25,
    "ltx2_edit_anything_adaln_scale": 2.0,
}
_PROMPT_RELAY_CUSTOM_SETTING = {
    "id": "prompt_relay_epsilon",
    "name": "Prompt Relay Epsilon",
    "label": "Prompt Relay Epsilon (higher values create softer transitions)",
    "type": "float",
    "default": 1e-3,
    "min": 1e-4,
    "max": 0.99,
    "inc": 1e-4,
    "video_prompt_type": "?",
}
_ARCH_SPECS = {
    "ltx2_19B": {
        "repo_id": "DeepBeepMeep/LTX-2",
        "config_file": "ltx2_19b_config.json",
        "spatial_upscaler": "ltx-2-spatial-upscaler-x2-1.0.safetensors",
        "temporal_upscaler": "ltx-2-temporal-upscaler-x2-1.0.safetensors",
        "distilled_lora": "ltx-2-19b-distilled-lora-384.safetensors",
        "union_control_lora": "ltx-2-19b-ic-lora-union-control-ref0.5.safetensors",
        "id_lora": "id-lora-celebvhq-ltx2.safetensors",
        "video_vae": "ltx-2-19b_vae.safetensors",
        "audio_vae": "ltx-2-19b_audio_vae.safetensors",
        "vocoder": "ltx-2-19b_vocoder.safetensors",
        "text_embedding_projection": "ltx-2-19b_text_embedding_projection.safetensors",
        "dev_embeddings_connector": "ltx-2-19b-dev_embeddings_connector.safetensors",
        "distilled_embeddings_connector": "ltx-2-19b-distilled_embeddings_connector.safetensors",
        "profiles_dir": "ltx2",
        "dev_profiles_dir": "ltx2_dev_accelerators",
        "preset_profiles_dir": "ltx2_presets",
        "distilled_preset_profiles_dir": "ltx2_distilled_presets",
        "lora_dir": "ltx2",
    },
    "ltx2_22B": {
        "repo_id": "DeepBeepMeep/LTX-2",
        "config_file": "ltx2_22b_config.json",
        "spatial_upscaler": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "temporal_upscaler": "ltx-2.3-temporal-upscaler-x2-1.0.safetensors",
        "distilled_lora": "ltx-2.3-22b-distilled-lora-384.safetensors",
        "distilled_1_1_lora": "ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
        "pixel_spatial_upscaler_lora": "ltx-2.3-22b-ic-lora-pixel-spatial-upscaler-x2-0.9.safetensors",
        "union_control_lora": "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
        "id_lora": "id-lora-celebvhq-ltx2.3.safetensors",
        "outpaint_lora": "ltx-2.3-22b-ic-lora-outpaint.safetensors",
        "inpaint_lora": "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors",
        "ingredients_lora": "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors",
        "hdr_lora": "ltx-2.3-22b-ic-lora-hdr-0.9.safetensors",
        "hdr_scene_embeddings": "ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors",
        "hdr_transform": "logc3",
        "video_vae": "ltx-2.3-22b_vae.safetensors",
        "diffusion_video_vae": _NAD_VAE_FILENAME,
        "audio_vae": "ltx-2.3-22b_audio_vae.safetensors",
        "vocoder": "ltx-2.3-22b_vocoder.safetensors",
        "text_embedding_projection": "ltx-2.3-22b_text_embedding_projection.safetensors",
        "embeddings_connector": "ltx-2.3-22b_embeddings_connector.safetensors",
        "profiles_dir": "ltx2",
        "dev_profiles_dir": "ltx2_dev_accelerators",
        "preset_profiles_dir": "ltx2_presets",
        "distilled_preset_profiles_dir": "ltx2_distilled_presets",
        "lora_dir": "ltx2",
    },
    "ltx2_25_22B": {
        "repo_id": "DeepBeepMeep/LTX-2",
        "config_file": "ltx2_25_22b_config.json",
        "spatial_upscaler": "ltx-2.5-spatial-upscaler-x2-1.0_bf16.safetensors",
        "temporal_upscaler": "ltx-2.5-temporal-upscaler-x2-1.0_bf16.safetensors",
        "distilled_lora": "ltx-2.5-22b-distilled-lora-450_bf16.safetensors",
        "pixel_spatial_upscaler_lora": "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors",
        "refine_details_lora": "ltx-2.5-22b-ic-lora-refine-details-1.0.safetensors",
        "union_control_lora": "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
        "id_lora": "id-lora-celebvhq-ltx2.3.safetensors",
        "outpaint_lora": "ltx-2.3-22b-ic-lora-outpaint.safetensors",
        "inpaint_lora": "ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors",
        "ingredients_lora": "ltx-2.5-22b-ic-lora-ingredients-0.9.safetensors",
        "hdr_lora": "ltx-2.5-22b-ic-lora-sdr-to-hdr-1.0.safetensors",
        "hdr_scene_embeddings": "ltx-2.5-22b-ic-lora-sdr-to-hdr-scene-emb.safetensors",
        "hdr_transform": "acescct",
        "video_vae": "ltx-2.5-22b_video_vae_bf16.safetensors",
        "diffusion_video_vae": _NAD_VAE_FILENAME,
        "audio_vae": "ltx-2.5-22b_audio_vae_bf16.safetensors",
        "vocoder": "ltx-2.5-22b_vocoder_bf16.safetensors",
        "text_embedding_projection": "ltx-2.5-22b_text_embedding_projection_bf16.safetensors",
        "video_embeddings_connector_bf16": "ltx-2.5-22b_video_embeddings_connector_bf16.safetensors",
        "video_embeddings_connector_int8": "ltx-2.5-22b_video_embeddings_connector_int8_convrot.safetensors",
        "audio_embeddings_connector_bf16": "ltx-2.5-22b_audio_embeddings_connector_bf16.safetensors",
        "audio_embeddings_connector_int8": "ltx-2.5-22b_audio_embeddings_connector_int8_convrot.safetensors",
        "profiles_dir": "ltx2",
        "dev_profiles_dir": "ltx2_25_dev_accelerators",
        "preset_profiles_dir": "ltx2_presets",
        "distilled_preset_profiles_dir": "ltx2_distilled_presets",
        "lora_dir": "ltx2",
    },
}
_ARCH_SPECS["ltx2_22B_msr"] = {
    **_ARCH_SPECS["ltx2_22B"],
    "profiles_dir": "ltx2_msr",
    "dev_profiles_dir": "ltx2_msr_dev_accelerators",
    "preset_profiles_dir": "ltx2_msr_presets",
    "distilled_preset_profiles_dir": "ltx2_msr_distilled_presets",
}
_ARCH_SPECS["ltx2_25_22B_msr"] = {
    **_ARCH_SPECS["ltx2_25_22B"],
    "profiles_dir": "ltx2_25_msr",
    "dev_profiles_dir": "ltx2_25_msr_dev_accelerators",
    "preset_profiles_dir": "ltx2_25_msr_presets",
    "distilled_preset_profiles_dir": "ltx2_25_msr_distilled_presets",
}
# LTX-2.5 VFX IC-LoRA workflows: same weights as LTX-2.5, listed as models of their own in the selector.
for model_type, profiles_name in (("ltx2_25_22B_alpha_gen", "ltx2_25_alpha_gen"), ("ltx2_25_22B_layout_to_render", "ltx2_25_layout_to_render")):
    _ARCH_SPECS[model_type] = {
        **_ARCH_SPECS["ltx2_25_22B"],
        "profiles_dir": profiles_name,
        "dev_profiles_dir": f"{profiles_name}_dev_accelerators",
        "preset_profiles_dir": f"{profiles_name}_presets",
        "distilled_preset_profiles_dir": f"{profiles_name}_distilled_presets",
    }
LTX2_22B_CLASS = {"ltx2_22B", "ltx2_22B_edit_anything", "ltx2_22B_msr", "joyai_echo"}
LTX2_25_CLASS = {"ltx2_25_22B", "ltx2_25_22B_msr", "ltx2_25_22B_alpha_gen", "ltx2_25_22B_layout_to_render"}
for model_type in LTX2_22B_CLASS:
    if model_type != "ltx2_22B" and model_type not in _ARCH_SPECS:
        _ARCH_SPECS[model_type]=_ARCH_SPECS["ltx2_22B"]

def _get_arch_spec(base_model_type: str | None) -> dict:
    return _ARCH_SPECS.get(base_model_type or "", _ARCH_SPECS["ltx2_19B"])


def _is_ltx25(base_model_type: str | None) -> bool:
    return base_model_type in LTX2_25_CLASS


def _supports_main_22b_loras(base_model_type: str | None) -> bool:
    return base_model_type == "ltx2_22B" or _is_ltx25(base_model_type)


def _ltx2_outpainting_method() -> int:
    return LTX2_OUTPAINTING_METHOD


def ltx2_uint8_guides(video_prompt_type, audio_prompt_type, any_outpainting):
    # outpainting edits the float control video in place (gamma, continuation frames) and the audio from the control video returns it as
    # the output video: float control videos for these modes, so that no uint8 copy stays alongside
    return not any_outpainting and "2" not in (audio_prompt_type or "")


def ltx2_guide_inpaint_color(video_prompt_type, any_outpainting, extra_settings):
    if "M" in (video_prompt_type or "") or (any_outpainting and LTX2_OUTPAINTING_METHOD == 2):
        return "66FF00"
    return 0


def ltx2_background_removal_color(video_prompt_type, extra_settings, **kwargs):
    return [255, 255, 255]


def _is_joyai_echo(base_model_type: str | None, model_def: dict | None = None) -> bool:
    return base_model_type == "joyai_echo" or bool((model_def or {}).get("joyai_echo", False))


def _joyai_prompt_types(video_prompt_type="", image_prompt_type=""):
    return "V1" if "V" in (video_prompt_type or "") else "", "V" if "V" in (image_prompt_type or "") else "S" if "S" in (image_prompt_type or "") else ""


def _joyai_settings(custom_settings=None, *, video_prompt_type="", image_prompt_type="", guidance_phases=2, runtime=False):
    video_prompt_type, image_prompt_type = _joyai_prompt_types(video_prompt_type, image_prompt_type)
    if not isinstance(custom_settings, dict):
        custom_settings = {}
    settings = {
        "num_inference_steps": 8,
        "guidance_scale": 1.0,
        "audio_guidance_scale": 1.0,
        "alt_guidance_scale": 1.0,
        "alt_scale": 0.0,
        "guidance_phases": max(1, min(2, int(guidance_phases or 2))),
        "audio_prompt_type": "",
        "video_prompt_type": video_prompt_type,
        "image_prompt_type": image_prompt_type,
        "custom_settings": custom_settings,
        "multi_prompts_gen_type": "PW",
    }
    if runtime:
        settings["audio_cfg_scale"] = 1.0
    return settings


def _validate_joyai_inputs(model_def, inputs):
    from shared.utils.audio_video import extract_audio_tracks
    from .joyai_echo import JOYAI_CONTROL_MEMORY_MAX_SECONDS, JOYAI_CONTROL_MEMORY_SETTING, parse_drop_mem_option, parse_load_mem_option, parse_store_mem_option, validate_control_memory_positions
    custom_settings = inputs.get("custom_settings", None)
    if not isinstance(custom_settings, dict):
        custom_settings = {}
    no_mem_notified = False
    for window in (inputs.get("frame_scheduler", {}) or {}).get("windows", []):
        model_options = window.get("model_options", {}) or {}
        if "no_mem" in model_options and not no_mem_notified:
            gr.Info("JoyAI-Echo /no_mem is deprecated because memories are no longer saved automatically. It will be ignored; use /store_mem=name only on windows you want to remember.")
            no_mem_notified = True
        store_mem = model_options.get("store_mem", None)
        if store_mem is not None:
            try:
                parse_store_mem_option(store_mem)
            except ValueError as exc:
                return str(exc)
        load_mem = model_options.get("load_mem", None)
        if load_mem is not None:
            try:
                parse_load_mem_option(load_mem)
            except ValueError as exc:
                return str(exc)
        drop_mem = model_options.get("drop_mem", None)
        if drop_mem is not None:
            try:
                parse_drop_mem_option(drop_mem)
            except ValueError as exc:
                return str(exc)
    memory_positions = str(custom_settings.get(JOYAI_CONTROL_MEMORY_SETTING, "") or "").strip()
    video_prompt_type, _ = _joyai_prompt_types(inputs.get("video_prompt_type", ""), inputs.get("image_prompt_type", ""))
    if not video_prompt_type:
        return None
    if inputs.get("video_guide") is None:
        return "JoyAI-Echo Control Video Memory requires a Control Video."
    if extract_audio_tracks(inputs.get("video_guide"), query_only=True) == 0:
        return "JoyAI-Echo Control Video Memory requires a Control Video with an audio track."
    return validate_control_memory_positions(memory_positions, float(model_def.get("fps", 25) or 25), max_seconds=JOYAI_CONTROL_MEMORY_MAX_SECONDS)


def _get_system_lora_urls(spec: dict) -> dict:
    return {
        f"ltx2_lora_{name}": build_hf_url(spec["repo_id"], spec[spec_key])
        for name, spec_key in _SYSTEM_LORA_SPEC_KEYS.items()
        if spec.get(spec_key)
    }


def _default_perturbation_layers(base_model_type: str | None) -> list[int]:
    return [28] if base_model_type in LTX2_22B_CLASS or _is_ltx25(base_model_type) else [29]


def _default_dev_settings(base_model_type: str | None) -> dict:
    if _is_ltx25(base_model_type):
        return {
            "num_inference_steps": 8,
            "video_length": 241,
            "resolution": "1280x704",
            "sample_solver": "distilled_8_steps_ancestral",
            "guidance_scale": 1.0,
            "audio_guidance_scale": 1.0,
            "alt_guidance_scale": 1.0,
            "alt_scale": 0.0,
            "perturbation_switch": 0,
            "perturbation_layers": _default_perturbation_layers(base_model_type),
            "perturbation_start_perc": 0,
            "perturbation_end_perc": 100,
            "apg_switch": 0,
            "cfg_star_switch": 0,
            "self_refiner_setting": 0,
            "guidance_phases": 2,
        }
    if base_model_type in LTX2_22B_CLASS:
        return {
            "num_inference_steps": 8,
            "video_length": 121,
            "resolution": "1280x720",
            "sample_solver": "distilled_8_steps",
            "guidance_scale": 1.0,
            "audio_guidance_scale": 1.0,
            "alt_guidance_scale": 1.0,
            "alt_scale": 0.0,
            "perturbation_switch": 0,
            "perturbation_layers": _default_perturbation_layers(base_model_type),
            "perturbation_start_perc": 0,
            "perturbation_end_perc": 100,
            "apg_switch": 0,
            "cfg_star_switch": 0,
            "self_refiner_setting": 0,
            "guidance_phases": 2,
        }
    return {
        "num_inference_steps": 40,
        "guidance_scale": 3.0,
        # "audio_guidance_scale": 7.0,
        # "alt_guidance_scale": 3.0,
        # "alt_scale": 0.7,
        # "perturbation_switch": 2,
        "perturbation_layers": _default_perturbation_layers(base_model_type),
        "perturbation_start_perc": 0,
        "perturbation_end_perc": 100,
        "apg_switch": 0,
        "cfg_star_switch": 0,
        "guidance_phases": 2,
    }


def _is_editanything_model(model_def) -> bool:
    return model_def.get("ltx2_edit_anything", False) or model_def.get("architecture","")=="ltx2_22B_edit_anything"


def _is_msr_model(base_model_type, model_def) -> bool:
    return base_model_type in {"ltx2_22B_msr", "ltx2_25_22B_msr"} or model_def.get("ltx2_msr", False)


def _is_distilled_model(model_def) -> bool:
    return model_def.get("ltx2_pipeline", "") == "distilled"


def _get_embeddings_connector_filename(model_def, base_model_type):
    spec = _get_arch_spec(base_model_type)
    shared_connector = spec.get("embeddings_connector")
    if shared_connector:
        return shared_connector
    pipeline_kind = (model_def or {}).get("ltx2_pipeline", "two_stage")
    if pipeline_kind == "distilled":
        return spec["distilled_embeddings_connector"]
    return spec["dev_embeddings_connector"]


def _get_multi_file_names(model_def, base_model_type):
    spec = _get_arch_spec(base_model_type)
    names = {
        "video_vae": model_def.get("ltx2_video_vae_file", spec["video_vae"]),
        "audio_vae": spec["audio_vae"],
        "vocoder": spec["vocoder"],
        "text_embedding_projection": spec["text_embedding_projection"],
    }
    if _is_ltx25(base_model_type):
        # LTX-2.5 connectors follow the transformer quantization setting: bf16, otherwise int8 ConvRot.
        connector_variant = "bf16" if sys.modules["wgp"].transformer_quantization == "bf16" else "int8"
        names["video_embeddings_connector"] = spec[f"video_embeddings_connector_{connector_variant}"]
        names["audio_embeddings_connector"] = spec[f"audio_embeddings_connector_{connector_variant}"]
    else:
        names["text_embeddings_connector"] = _get_embeddings_connector_filename(model_def, base_model_type)
    return names


def _resolve_multi_file_paths(model_def, base_model_type, include_spatial_upsampler=True):
    spec = _get_arch_spec(base_model_type)
    paths = {key: fl.locate_file(name) for key, name in _get_multi_file_names(model_def, base_model_type).items()}
    if include_spatial_upsampler:
        paths["spatial_upsampler"] = fl.locate_file(spec["spatial_upscaler"])
    model_config = os.path.join(os.path.dirname(__file__), "configs", spec["config_file"])
    if not os.path.isfile(model_config):
        raise FileNotFoundError(f"Missing LTX config file: {model_config}")
    paths["model_config"] = model_config
    video_vae_config_file = model_def.get("ltx2_video_vae_config_file")
    if video_vae_config_file:
        video_vae_config = os.path.join(os.path.dirname(__file__), video_vae_config_file)
        if not os.path.isfile(video_vae_config):
            raise FileNotFoundError(f"Missing LTX video VAE config file: {video_vae_config}")
        paths["video_vae_config"] = video_vae_config
    return paths


def _migrate_loras():
    global _LORAS_MIGRATED
    if _LORAS_MIGRATED:
        return
    wgp = sys.modules.get("wgp")
    lora_root = wgp.get_lora_root()

    moved = set()
    for spec in _ARCH_SPECS.values():
        lora_dir = Path(resolve_lora_dir(spec["lora_dir"], lora_root, wgp.args.lora_config))
        for key in _LORA_SPEC_KEYS:
            filename = spec.get(key, None)
            if filename is None or filename in moved:
                continue
            source = fl.locate_file(filename, error_if_none=False)
            if source is None:
                continue
            target = lora_dir / filename
            if Path(source).resolve() == target.resolve() or target.exists():
                moved.add(filename)
                continue
            shutil.move(source, target)
            print(f"[WAN2GP][LTX2] Moved {key} LoRA '{source}' -> '{target}'")
            moved.add(filename)
            
    _LORAS_MIGRATED = True


def _notify_control_video_phase2(base_model_type, model_def, inputs, any_outpainting):
    video_prompt_type = inputs.get("video_prompt_type", "") or ""
    if int(inputs.get("guidance_phases", 1)) != 2 or "V" not in video_prompt_type or inputs.get("video_guide") is None:
        return ""
    if "&" in video_prompt_type and _is_ltx25(base_model_type):
        return ""  # LTX-2.5 HDR conversion runs as a single phase
    wgp = sys.modules.get("wgp")
    lora_dir = wgp.get_lora_dir(base_model_type) if wgp is not None and hasattr(wgp, "get_lora_dir") else None
    selected = {os.path.basename(lora).lower() for lora in inputs.get("activated_loras", []) or []}
    spec = _get_arch_spec(base_model_type)
    outpainting_method = _ltx2_outpainting_method()
    new_outpainting = any_outpainting and outpainting_method == 2
    builtins = [
        spec.get("hdr_lora") if _supports_main_22b_loras(base_model_type) and "&" in video_prompt_type else None,
        spec.get("union_control_lora") if any(letter in video_prompt_type for letter in "OPDE") else None,
        spec.get("outpaint_lora") if _supports_main_22b_loras(base_model_type) and any_outpainting and outpainting_method == 1 else None,
        spec.get("inpaint_lora") if _supports_main_22b_loras(base_model_type) and ("M" in video_prompt_type and "A" in video_prompt_type or new_outpainting) else None,
    ]
    extra_loras = [os.path.join(lora_dir, name) if lora_dir else name for name in builtins if name and name.lower() not in selected]
    extra_mults = [1.0] * len(extra_loras)
    activated_loras = [os.path.join(lora_dir, os.path.basename(lora)) if lora_dir else lora for lora in inputs.get("activated_loras", []) or []]
    steps, switch_phase = int(inputs.get("num_inference_steps", 1)), inputs.get("model_switch_phase", 1)
    _, loras_slists, errors = parse_loras_multipliers(extra_mults, len(extra_loras), steps, nb_phases=2, model_switch_phase=switch_phase)
    if not errors:
        _, loras_slists, errors = parse_loras_multipliers(inputs.get("loras_multipliers", ""), len(activated_loras), steps, nb_phases=2, merge_slist=loras_slists, model_switch_phase=switch_phase)
    if errors:
        return f"Error parsing Loras: {errors}"
    loras_selected = extra_loras + activated_loras
    layout = model_def.get("ltx2_layout_to_render", False)
    msg = control_video_phase2_message(loras_selected, loras_slists, force_phase2_control=layout or _is_editanything_model(model_def), force_name="Layout to Render" if layout else "EditAnything")
    print(msg)
    gr.Info(msg)
    return ""


class family_handler:
    @staticmethod
    def query_supported_types():
        _migrate_loras()
        return ["ltx2_19B", "ltx2_22B", "ltx2_25_22B", "ltx2_22B_edit_anything", "ltx2_22B_msr", "ltx2_25_22B_msr", "ltx2_25_22B_alpha_gen", "ltx2_25_22B_layout_to_render", "joyai_echo"]

    @staticmethod
    def query_family_maps():

        models_eqv_map = {
            "ltx2_19B" : "ltx2_22B",
            "ltx2_25_22B" : "ltx2_22B",
            "ltx2_22B_edit_anything" : "ltx2_22B",
            "ltx2_22B_msr" : "ltx2_22B",
            "ltx2_25_22B_msr" : "ltx2_22B",
            "ltx2_25_22B_alpha_gen" : "ltx2_22B",
            "ltx2_25_22B_layout_to_render" : "ltx2_22B",
        }

        models_comp_map = { 
                    "ltx2_19B" : [ "ltx2_22B", "ltx2_22B_edit_anything", "ltx2_22B_msr"],
                    }
        return models_eqv_map, models_comp_map

    @staticmethod
    def query_model_family():
        return "ltx2"

    @staticmethod
    def query_family_infos():
        return {"ltx2": (40, "LTX-2")}

    @staticmethod
    def query_model_def(base_model_type, model_def):
        preload_urls = model_def.get("preload_URLs")
        spec = _get_arch_spec(base_model_type)
        ltx25 = _is_ltx25(base_model_type)
        joy = _is_joyai_echo(base_model_type, model_def)
        msr = _is_msr_model(base_model_type, model_def)
        msr_v2 = msr and any(
            "msr-v2" in str(value).lower()
            for value in (model_def.get("loras", []) if isinstance(model_def.get("loras", []), (list, tuple)) else [model_def.get("loras", "")])
        )
        if isinstance(preload_urls, list): 
            # migrate old finetunes
            lora_filenames = {spec[key] for key in _LORA_SPEC_KEYS if key in spec}
            def add_lora_dir_suffix(entry):
                if not isinstance(entry, str) or "|%lora_dir" in entry:
                    return entry
                source_entry = entry.split("|", 1)[0]
                if source_entry.startswith("http") and os.path.basename(source_entry) in lora_filenames:
                    return f"{source_entry}|%lora_dir"
                return entry
            model_def["preload_URLs"] = [add_lora_dir_suffix(entry) for entry in preload_urls]

        editanything_ref = _is_editanything_model(model_def)
        pipeline_kind = "distilled" if joy or _is_distilled_model(model_def) else "two_stage"

        distilled = pipeline_kind == "distilled"
        gemma_folder = _GEMMA4_FOLDER if ltx25 else _GEMMA_FOLDER
        gemma_files = (_GEMMA4_FILENAME, _GEMMA4_INT8_FILENAME) if ltx25 else (_GEMMA_FILENAME, _GEMMA_QUANTO_FILENAME)
        extra_model_def = {
            "device_explicit": True,
            "cuda_graphs": ["transformer", "transformer2"],
            "uint8_guides": ltx2_uint8_guides,
            "ltx2_22B_class": base_model_type in LTX2_22B_CLASS or ltx25,
            "ltx2_edit_anything": editanything_ref,
            "infos": (model_def.get("infos") or (LTX2_25_MSR_INFOS if ltx25 and msr else LTX2_25_INFOS if ltx25 else LTX2_MSR_V2_INFOS if msr_v2 else LTX2_MSR_INFOS if msr else LTX2_INFOS)) + LTX2_SOL_INFOS,
            "text_encoder_folder": gemma_folder,
            "text_encoder_URLs": [
                build_hf_url("DeepBeepMeep/LTX-2", gemma_folder, gemma_files[0]),
                build_hf_url("DeepBeepMeep/LTX-2", gemma_folder, gemma_files[1]),
            ],
            "dtype": "bf16",
            "fps": 24,
            "prompt_enhancer_video_duration": True,
            "frames_minimum": 17,
            "frames_steps": 8,
            "sliding_window": not msr,
            "returns_audio": True,
            "audio_window_alignment": True,
            "audio_overlap_repair_ms": 80,  # reserve two 40 ms audio latents at the join
            "auto_null_audio": True,
            "multimedia_generation": True,
            "image_end_frame_position": True,
            "profiles_dir": [spec["profiles_dir"]] if distilled else [spec["profiles_dir"], spec["dev_profiles_dir"]],
            "ltx2_spatial_upscaler_file": spec["spatial_upscaler"],
            "ltx2_hdr_lora_file": spec.get("hdr_lora", ""),
            "ltx2_hdr_scene_embeddings_file": spec.get("hdr_scene_embeddings", ""),
            "ltx2_hdr_transform": spec.get("hdr_transform"),
            "self_refiner": True,
            "self_refiner_max_plans": 2,
            "custom_settings": [_PROMPT_RELAY_CUSTOM_SETTING.copy()],
            "custom_attention_modes": {
                "sol": {"label": "Sol sparse attention, requires Triton and an NVIDIA RTX 30/40/50-series GPU or newer", "supports_sparsity": True},
            },
            "attention_sparsity": {
                "label": "Start Tau (higher = more sparse/faster; lower = more faithful; End Tau = 0.8)",
                "start": 0.0,
                "end": 4.0,
                "inc": 0.05,
            },
            # "no_background_removal": True,
            "vae_block_size": 64,
            "keep_frames_video_guide_not_supported": True,
            # LTX-2.3/2.5 variants share the 2.3 video latent space; conditioning tokens are removed before previews.
            "tiny_vae_architecture": "ltx2_22B" if base_model_type in LTX2_22B_CLASS or ltx25 else "ltx2_19B",
        }
        extra_model_def["prompt_enhancer_button_label"] = "Write"
        if base_model_type in LTX2_22B_CLASS or ltx25:
            extra_model_def["system_configs"] = {
                "_name": "VAE",
                "_default_label": "Default VAE",
                "PrunaAI VAE": {
                    "name": "PrunaAI VAE (faster, slightly worse quality)",
                    "ltx2_video_vae_file": _PRUNAAI_VAE_FILENAME,
                    "ltx2_video_vae_config_file": _PRUNAAI_VAE_CONFIG_FILENAME,
                    "ltx2_pruna_vae": True,
                },
                "NAD Diffusion Decoder": {
                    "name": "NAD Diffusion Decoder (slower, higher VRAM, better motion)",
                    "ltx2_video_vae_file": spec["diffusion_video_vae"],
                },
            }
        extra_model_def.update(_get_system_lora_urls(spec))
        if distilled:
            extra_model_def["ltx2_pipeline"] = "distilled"
            extra_model_def["accelerated"] = "native"
        else:
            extra_model_def["finetune_custom_urls"] =  [ "ltx2_lora_distilled"]

            
        if editanything_ref:
            extra_model_def.update(_EDITANYTHING_MODEL_DEF)
        
        if _supports_main_22b_loras(base_model_type):

            extra_model_def["video_guide_outpainting"] = [0,1]
            extra_model_def["video_guide_outpainting_label"] = "Enable Spatial Outpainting on Control Video using LTX2 Outpainting IC-LoRA"
            extra_model_def["guide_inpaint_color"] = ltx2_guide_inpaint_color
            extra_model_def["background_removal_color"] = ltx2_background_removal_color

        extra_model_def["preset_profiles_dir"] = [spec.get("distilled_preset_profiles_dir") if distilled else spec.get("preset_profiles_dir")]
        extra_model_def["extra_control_frames"] = 1
        extra_model_def["dont_cat_preguide"] = True
        extra_model_def["input_video_strength"] = {
            "label": "Start Image / Source Strength (lower values may create more motion)",
            "name": "Start Image / Source Strength",
        }
        if joy:
            from .joyai_echo import JOYAI_CONTROL_MEMORY_SETTING, JOYAI_ECHO_INFOS, JOYAI_ECHO_PROMPT_ENHANCER, JOYAI_ECHO_PROMPT_INFOS

            extra_model_def.update(
                {
                    "joyai_echo": True,
                    "joyai_audio_memory": True,
                    "joyai_memory_max_size": 7,
                    "joyai_memory_num_fix_frames": 3,
                    "joyai_memory_downscale_factor": 1,
                    "joyai_audio_memory_window_size": 96,
                    "prompt_slash_commands": ["no_mem", "store_mem", "load_mem", "drop_mem"],
                    "preserve_empty_prompt_lines": True,
                    "skip_video_guide_preprocess": True,
                    "NAG": True,
                    "infos": model_def.get("infos", JOYAI_ECHO_INFOS) + LTX2_SOL_INFOS,
                    "fps": 25,
                    "image_prompt_types_allowed": "TSV",
                    "prompt_infos": JOYAI_ECHO_PROMPT_INFOS,
                    "prompt_enhancer_def": {"selection": ["TM", "TIM"], "labels": {"TM": "A JoyAI-Echo multi-shot prompt using existing Text Prompt", "TIM": "A JoyAI-Echo multi-shot prompt using existing Text Prompt and {image_inputs}"}, "default": ""},
                    "text_prompt_enhancer_instructions1": JOYAI_ECHO_PROMPT_ENHANCER,
                    "video_prompt_enhancer_instructions1": JOYAI_ECHO_PROMPT_ENHANCER,
                    "image_prompt_enhancer_instructions1": JOYAI_ECHO_PROMPT_ENHANCER,
                    "text_prompt_enhancer_max_tokens1": 1536,
                    "video_prompt_enhancer_max_tokens1": 1536,
                    "image_prompt_enhancer_max_tokens1": 1536,
                    "guide_custom_choices": {"choices": [("No Control Video Memory", ""), ("JoyAI-Echo Control Video Memory", "V1")], "letters_filter": "V1", "default": "", "label": "Control Video Memory"},
                    "custom_settings": [
                        _PROMPT_RELAY_CUSTOM_SETTING.copy(),
                        {"id": JOYAI_CONTROL_MEMORY_SETTING, "name": "Control Video Memory Positions", "label": "Joy Memory Positions from Control Video (frames or seconds, comma-separated)", "type": "text", "default": "", "video_prompt_type": "1"},
                    ],
                }
            )
        else:
            from .prompt_enhancer import LTX2_25_DEEPY_PROMPT_INFOS, LTX2_PROMPT_INFOS, LTX2_RELAYED_IMAGE_PROMPT, LTX2_RELAYED_PROMPT

            if ltx25 and not (msr or editanything_ref):
                extra_model_def.update({"deepy_infos": LTX2_25_DEEPY_INFOS, "deepy_prompt_infos": LTX2_25_DEEPY_PROMPT_INFOS})

            if msr:
                audio_prompt_selection = ["", "A"]
            elif editanything_ref and not distilled:
                audio_prompt_selection = ["", "A", "K"]
            else:
                audio_prompt_selection = ["", "A", "K", "2", "A1OF"]
            audio_prompt_labels = {
                "": "Generate Video & Soundtrack based on Text Prompt",
                "A": "Generate Video based on Soundtrack and Text Prompt",
                "K": "Generate Video based on Control Video + its Audio Track and Text Prompt",
                "2": "Generate Audio based on Control Video and Text Prompt",
                "A1OF": "Generate Video based on Reference Voice (ID-LoRA) and Text Prompt",
            }
            if ltx25:
                audio_prompt_selection = [choice for choice in audio_prompt_selection if choice != "A1OF"]
                audio_prompt_labels.pop("A1OF")
            extra_model_def.update(
                {
                    "image_prompt_types_allowed": "TSEVL",
                    "end_frames_always_enabled": True,
                    "any_audio_prompt": True,
                    "audio_prompt_choices": True,
                    "one_speaker_only": True,
                    "audio_guide_label": "Audio Prompt (Soundtrack, leave blank to to use a Null Audio)",
                    "audio_scale_name": "Prompt Audio Strength",
                    "audio_prompt_type_sources": {
                        "selection": audio_prompt_selection,
                        "labels": audio_prompt_labels,
                        "custom_flags": {"2": audio_prompt_labels["2"]} if ltx25 else {
                            "1": "Reference Voice (ID-LoRA)",
                            "2": "Generate Audio based on Control Video and Text Prompt",
                        },
                        "letters_filter": "A1OFK2",
                        "show_label": False,
                        "default": "K" if editanything_ref else "",
                    },
                    "prompt_infos": LTX2_PROMPT_INFOS,
                    "prompt_enhancer_def": {
                        "selection": ["T", "TI", "T1", "TI1"],
                        "labels": {
                            "T": "An Enhanced Prompt using existing Text Prompt",
                            "TIV": "An Enhanced Prompt using existing Text Prompt and {image_inputs}",
                            "T1V": "An Enhanced Relayed Prompt using existing Text Prompt",
                            "TI1V": "An Enhanced Relayed Prompt using existing Text Prompt and {image_inputs}",
                        },
                        "default": "",
                    },
                    "text_prompt_enhancer_instructions1": LTX2_RELAYED_PROMPT,
                    "video_prompt_enhancer_instructions1": LTX2_RELAYED_IMAGE_PROMPT,
                    "text_prompt_enhancer_max_tokens1": 1024,
                    "video_prompt_enhancer_max_tokens1": 1024,
                    "audio_guide_window_slicing": True,
                    "video_length_not_limited_by_audio": True,
                    "output_audio_is_input_audio": True,
                    "multiple_images_as_text_prompts": True,
                    "custom_denoising_strength": distilled,
                    "NAG": True,
                    "v2i_switch_supported": True,
                    "image_batch_size_max": 1,
                }
            )
            extra_model_def["denoising_strength"] = {
                "label": "Control Video Strength (higher = closer to the Control Video)",
                "name": "Control Video Strength",
            }
            extra_model_def["masking_strength"] = {
                "label": "Unmasked Area Strength (higher = unmasked area closer to control video)",
                "name": "Unmasked Area Strength",
            }
            if msr:
                control_choices = [("No Control Video", "")]
            elif base_model_type in ["ltx2_22B_edit_anything"]:
                control_choices = [("EditAnything Source Video", "VGI")]
            else:
                control_choices = [("No Video Process", "")]
                control_choices += [("Transfer Human Motion", "PVG"), ("Transfer Human Motion With Pose Alignment", "OVG"), ("Transfer Depth", "DVG"), ("Transfer Canny Edges", "EVG"), ("LTX2 Raw Format / Control Video for Ic Lora", "VG")]
                if _supports_main_22b_loras(base_model_type):
                    control_choices += [("Inpaint Masked Area", "MVG"), ("Ingredients Reference Sheet", "I"), ("Convert SDR to HDR (IC-LoRA)", f"V&G")]
                control_choices += [("Inject Frames", "KFI")]
            control_choices_image = [(label, value) for label, value in control_choices if value not in ("OVG", "MVG", "I", "KFI", "V&G")]
            if not msr:
                guide_custom_choices = {
                    "choices": control_choices,
                    "letters_filter": "OPDEMVG&KFI",
                    "default": "VGI" if editanything_ref else "",
                    "label": "Control Video / Frames Injection",
                    "visible":  not editanything_ref  ,
                }
                extra_model_def["guide_custom_choices"] = guide_custom_choices
                extra_model_def["guide_custom_choices_image"] = {**guide_custom_choices, "choices": control_choices_image, "label": "Control Image"}
            extra_model_def["custom_frames_injection"] = True
            extra_model_def["one_image_ref_only"] = True
            if editanything_ref:
                extra_model_def["one_image_ref_needed"] = True
            extra_model_def["mask_preprocessing"] = {"selection": [""], "visible": False} if editanything_ref or msr else {"selection": ["", "A", "NA", "XA", "XNA"]}
            if msr:
                extra_model_def.update(
                    {
                        "ltx2_msr": True,
                        "image_prompt_types_allowed": "TE",
                        "image_ref_choices": {
                            "choices": [("Up to 5 Subjects / Objects", "I"), ("Background + Up to 4 Subjects", "KI")],
                            "letters_filter": "KI",
                            "default": "KI",
                            "label": "MSR Reference Images",
                            "show_label": True,
                        },
                        "one_image_ref_only": False,
                        "at_least_one_image_ref_needed": True,
                        "custom_frames_injection": False,
                        "all_image_refs_are_background_ref": False,
                        "fit_into_canvas_image_refs": 1,
                        "background_removal_label": "Remove Background behind MSR Subjects / Objects",
                        "multiple_images_as_text_prompts": False,
                        "ltx2_msr_frame_count": int(model_def.get("ltx2_msr_frame_count", 41)),
                    }
                )
                if ltx25:
                    extra_model_def["ltx2_msr_slot_embeddings_file"] = _MSR25_SLOT_FILENAME
                    extra_model_def["image_ref_choices"]["choices"][0] = ("Up to 4 Subjects / Objects", "I")
                if msr_v2:
                    extra_model_def["custom_settings"] = list(model_def.get("custom_settings", [])) if isinstance(model_def.get("custom_settings", []), list) else []
                    extra_model_def["custom_settings"].append(
                        {
                            "id": "msr_reference_frames",
                            "name": "MSR Reference Video Length",
                            "label": "MSR Reference Video Length",
                            "type": "dropdown",
                            "default": "",
                            "choices": [("Auto", "")] + [(str(value), value) for value in (17, 25, 33, 41, 49, 57, 65)],
                        }
                    )
        extra_model_def["sliding_window_defaults"] = {
            "overlap_min": 1,
            "overlap_max": 97,
            "overlap_step": 8,
            "overlap_default": 9,
            "window_min": 5,
            "window_max": 501,
            "window_step": 4,
            "window_default": 241,
        }
        if distilled:
            extra_model_def.update(
                {
                    "lock_inference_steps": True,
                }
            )
        else:
            extra_model_def.update(
                {
                    "audio_guidance": True,
                    "adaptive_projected_guidance": True,
                    "cfg_star": True,
                    "perturbation": True,
                    "alt_guidance": "Modality Guidance",
                    "alt_scale": "Guidance Rescale",
                    "perturbation_choices": [
                        ("Off", 0),
                        ("Skip Layer Guidance", 1),
                        ("Skip Self Attention", 2),
                    ],
                    "perturbation_layers_max": 48,
                }
            )
            if ltx25:
                extra_model_def["sample_solvers"] = [("Distilled 8 Steps (Euler Ancestral)", "distilled_8_steps_ancestral"), ("Distilled 8 Steps (Euler)", "distilled_8_steps"), ("Euler", "euler"), ("HQ (res2s)", "res2s")]
            elif base_model_type in LTX2_22B_CLASS:
                extra_model_def["sample_solvers"] = [("Distilled 8 Steps", "distilled_8_steps"), ("Euler", "euler"), ("HQ (res2s)", "res2s")]
        extra_model_def["guidance_max_phases"] = 2
        extra_model_def["custom_sigmas"] = True
        extra_model_def["custom_sigmas_switch_scale"] = None  # stages have independent schedules, no noise-switch control
        extra_model_def["custom_sigmas_schedulers"] = ["", "euler", "res2s"]
        extra_model_def["visible_phases"] = 0 if distilled else 1
        if ltx25:
            extra_model_def["self_refiner"] = False
        # extra_model_def["lock_guidance_phases"] = True

        # extra_model_def["custom_video_selection"] = {
        #     "choices":[
        #         ("None", ""),
        #         ("Inject Frames", "FI"),
        #     ],
        #     "label": "Inject Frames",
        #     "type": "checkbox",
        #     "letters_filter": "FI",
        #     "show_label" : False,
        #     "scale": 1,
        #     }

        from models.help_shared import merge_help
        from .infos import compact_help
        extra_model_def = merge_help(extra_model_def, compact_help(msr=msr, joy=joy, editanything=editanything_ref, ltx25=ltx25))
        extra_model_def["infos"] += "\n\n**Custom Sigmas:** leave empty for the normal schedules. Separate values with commas and phase slots with `;`; a blank slot keeps that phase's defaults. Explicit schedules override its step count. The **i** beside the field explains the syntax. Fixed eight-step schedulers require this field to stay empty."
        extra_model_def["deepy_infos"] += " `custom_sigmas` defaults to empty; use comma-separated positive descending values below 1 and one semicolon-separated slot per phase, e.g. `;0.9,0.6,0.3`. Blank slots retain defaults; optional final zero is ignored. Explicit sigmas are used directly without another flow shift and determine that phase's step count. Each phase restarts its own schedule. Compatible selectable schedulers: `euler`, `res2s`; fixed `distilled_8_steps` / `distilled_8_steps_ancestral` require empty custom sigmas."
        extra_model_def["phase_2_spatial_tiling"] = not (msr or joy or editanything_ref)
        if extra_model_def["phase_2_spatial_tiling"]:
            tiling_help = "\n\n**Two Phases with Tiling** refines overlapping spatial tiles in phase 2, blending predictions at every step. Phase 1 establishes the whole scene. This reduces each transformer call's spatial extent, takes more calls, and can change fine detail. VAE tiling is independent."
            for key in ("infos", "deepy_infos"):
                if key in extra_model_def:
                    extra_model_def[key] += tiling_help
        from .vfx import vfx_model_def
        extra_model_def.update(vfx_model_def(model_def))
        if ltx25:
            extra_model_def["resolution_budget"] = "1920x1088"
            if extra_model_def["phase_2_spatial_tiling"]:
                extra_model_def["resolution_budget_tiled"] = "3840x2176"
        return extra_model_def

    @staticmethod
    def custom_preprocess(base_model_type, video_guide, video_mask, video_prompt_type, **kwargs):
        from .vfx import preprocess_alpha_source
        return preprocess_alpha_source(video_guide, video_mask, video_prompt_type)

    @staticmethod
    def get_rgb_factors(base_model_type):
        from shared.RGB_factors import get_rgb_factors

        return get_rgb_factors("ltx2", "ltx2_22B" if base_model_type in LTX2_22B_CLASS or _is_ltx25(base_model_type) else base_model_type)

    @staticmethod
    def get_lora_dir(base_model_type):
        return "ltx2"

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        spec = _get_arch_spec(base_model_type)

        file_list = [spec["spatial_upscaler"]] if _is_joyai_echo(base_model_type, model_def) else [spec["spatial_upscaler"], spec["temporal_upscaler"]]
        component_names = _get_multi_file_names(model_def, base_model_type)
        if _is_ltx25(base_model_type):
            component_names["diffusion_video_vae"] = spec["diffusion_video_vae"]
        for name in component_names.values():
            if name not in file_list:
                file_list.append(name)
        if _is_ltx25(base_model_type) and _is_msr_model(base_model_type, model_def):
            file_list.append(_MSR25_SLOT_FILENAME)

        download_def = [
            {
                "repoId": spec["repo_id"],
                "sourceFolderList": [""],
                "fileList": [file_list],
            },
            {
                "repoId": "DeepBeepMeep/LTX-2",
                "sourceFolderList": [_GEMMA4_FOLDER if _is_ltx25(base_model_type) else _GEMMA_FOLDER],
                "fileList": [_GEMMA4_TOKENIZER_FILES if _is_ltx25(base_model_type) else _GEMMA_TOKENIZER_FILES],
            },
        ]
        return download_def

    def validate_generative_settings(base_model_type, model_def, inputs):
        pipeline_kind = model_def.get("ltx2_pipeline", "two_stage")
        alpha = model_def.get("ltx2_alpha_gen", False)
        layout = model_def.get("ltx2_layout_to_render", False)
        if alpha or layout:
            if not inputs.get("video_guide") or "V" not in inputs.get("video_prompt_type", ""):
                return "Alpha Gen requires a Source Video." if alpha else "Layout to Render requires a Layout Video."
            if alpha:
                if inputs.get("video_length", 121) > 145:
                    return "Alpha Gen supports at most 145 frames. Trim or split the source clip."
                inputs.update(guidance_phases=1, masking_strength=0.0, denoising_strength=1.0, prompt_enhancer="", audio_prompt_type="")
            elif not inputs.get("image_refs"):
                return "Layout to Render requires an Appearance Reference made from the layout's first frame."
        if _is_ltx25(base_model_type):
            inputs["self_refiner_setting"] = 0
        if _is_joyai_echo(base_model_type, model_def):
            error = _validate_joyai_inputs(model_def, inputs)
            if error:
                return error
            inputs.update(_joyai_settings(inputs.get("custom_settings"), video_prompt_type=inputs.get("video_prompt_type", ""), image_prompt_type=inputs.get("image_prompt_type", ""), guidance_phases=inputs.get("guidance_phases", 2), runtime=True))
            return
        if pipeline_kind == "distilled":
            inputs.update(
                {
                    "num_inference_steps": 8,
                    "guidance_scale": 1.0,
                    "audio_guidance_scale": 1.0,
                    "audio_cfg_scale": 1.0,
                    "alt_guidance_scale": 1.0,
                    "alt_scale": 0.0,
                }
            )
            if inputs.get("perturbation",0) == 2:
                inputs["perturbation"] = 0
        else:
            sampler_enabled = base_model_type in LTX2_22B_CLASS or _is_ltx25(base_model_type)
            sample_solver = inputs.get("sample_solver", "euler" if sampler_enabled else "").lower()
            if sampler_enabled:
                supported_samplers = {"distilled_8_steps", "euler", "res2s"}
                if _is_ltx25(base_model_type):
                    supported_samplers.add("distilled_8_steps_ancestral")
                if sample_solver not in supported_samplers:
                    return f"Unsupported LTX2 sampler '{sample_solver}'."
                inputs["sample_solver"] = sample_solver
                if sample_solver in {"distilled_8_steps", "distilled_8_steps_ancestral"}:
                    inputs["num_inference_steps"] = 8
                if sample_solver == "res2s":
                    if inputs.get("apg_switch", 0):
                        return "HQ sampler does not support APG yet."
                    if inputs.get("cfg_star_switch", 0):
                        return "HQ sampler does not support CFG Star yet."
                    if inputs.get("self_refiner_setting", 0):
                        return "HQ sampler does not support Self Refiner yet."
                    if inputs.get("perturbation_switch", 0) not in (0, 2):
                        return "HQ sampler supports only Off or Skip Self Attention guidance."
            elif sample_solver not in {"", "euler"}:
                return f"Sampler '{sample_solver}' is not supported for {base_model_type}."
        video_guide_outpainting = inputs.get("video_guide_outpainting", None) 
        video_guide_outpainting_ratio = inputs.get("video_guide_outpainting_ratio", "") 
        video_prompt_type = inputs.get("video_prompt_type", "") or ""
        image_prompt_type = inputs.get("image_prompt_type", "") or ""
        audio_prompt_type = inputs.get("audio_prompt_type", "") or ""
        from shared.utils.utils import get_outpainting_dims

        any_outpainting = get_outpainting_dims(video_guide_outpainting, video_guide_outpainting_ratio) is not None
        if _is_msr_model(base_model_type, model_def):
            if any(letter in image_prompt_type for letter in "SVL"):
                return "LTX2 MSR does not support Start Image, Continue Video, or Continue Last Video."
            if "I" not in video_prompt_type:
                return "LTX2 MSR requires the MSR Reference Images mode."
            if any(letter in video_prompt_type for letter in "OPDEVG&FA"):
                return "LTX2 MSR supports only MSR reference images for video conditioning."
            if any(letter in audio_prompt_type for letter in "K2"):
                return "LTX2 MSR does not support Control Video audio options."
            image_refs = inputs.get("image_refs") or []
            if "K" in video_prompt_type:
                if not 2 <= len(image_refs) <= 5:
                    return "LTX2 MSR Background + Subjects mode requires 2 to 5 reference images, with the background image first."
            elif not 1 <= len(image_refs) <= 4:
                return "LTX2 MSR Subjects / Objects only mode requires 1 to 4 reference images."
        if "2" in audio_prompt_type:
            if any(letter in audio_prompt_type for letter in "AK"):
                return "LTX2 audio generation from Control Video must use the dedicated audio option, without an Audio Source or Control Video Audio Track prompt."
            if "V" not in video_prompt_type or "G" not in video_prompt_type:
                return "LTX2 audio generation from Control Video requires 'LTX2 Raw Format / Control Video for Ic Lora'."
            if any(letter in video_prompt_type for letter in "OPDE&AFKI") or any_outpainting:
                return "LTX2 audio generation from Control Video supports only raw Control Video, without Pose/Depth/Canny/HDR/Outpaint/Mask/Inject Frames."
            if inputs.get("video_guide") is None:
                return "You must provide a Control Video to generate audio from it."
        if "&" in video_prompt_type:
            if not _supports_main_22b_loras(base_model_type):
                return "LTX2 HDR IC-LoRA is supported only with the main 22B models."
            if any(letter in video_prompt_type for letter in "OPDE") or any_outpainting:
                return "LTX2 HDR IC-LoRA is not compatible with Pose/Depth/Canny/Outpaint control modes."
            if "F" in video_prompt_type:
                return "LTX2 HDR IC-LoRA is not yet compatible with Inject Frames."
            if _is_ltx25(base_model_type) and (any(letter in image_prompt_type for letter in "SE") or "A" in video_prompt_type):
                return "LTX-2.5 SDR to HDR conversion is driven by the whole Control Video only: Start / End Images and Video Masks are not supported."
        if "M" in video_prompt_type:
            if not _supports_main_22b_loras(base_model_type):
                return "LTX2 inpainting IC-LoRA is supported only with the main 22B models."
            if "A" not in video_prompt_type:
                return "LTX2 inpainting requires a Video Mask."
            if float(inputs.get("masking_strength", 0)) != 0.0:
                return "LTX2 inpainting IC-LoRA requires Unmasked Area Strength to be 0 as Unmasked Area is managed also by Inpainting."
            if float(inputs.get("denoising_strength", 1)) != 1.0:
                return "LTX2 inpainting IC-LoRA requires Control Video Strength to be 1."
        if any_outpainting:
            if "V" in video_prompt_type :
                if any(letter in video_prompt_type for letter in "OPDE"):
                    return "LTX2 outpainting on Control Video supports only LTX2 Raw Format  / Contro Video for Ic Lora."
                if "1" in audio_prompt_type:
                    return "LTX2 outpainting on Control Video is not compatible with the ID-LoRA option."
                if "F" in video_prompt_type :
                    return "LTX2 outpainting is not yet compatible with Inject Frames."
                if "A" in video_prompt_type and _ltx2_outpainting_method() == 1:
                    return "LTX2 outpainting doesnt support Video Mask."

        guide_phases = inputs.get("guidance_phases", 1)
        if guide_phases !=1 and "V" in video_prompt_type and any_outpainting and _ltx2_outpainting_method() == 1:
            inputs["guidance_phases"]=  1            
            gr.Info("Number of Phases has been set to 1 as Outpainting is enabled")
        if "2" not in audio_prompt_type:
            error = _notify_control_video_phase2(base_model_type, model_def, inputs, any_outpainting)
            if error:
                return error
        if "A" in audio_prompt_type and inputs.get("audio_guide") is None:
            audio_source = inputs.get("audio_source")
            if audio_source is not None:
                inputs["audio_guide"] = audio_source

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
        from .ltx2 import LTX2, LTX2_ENABLE_EMBEDDING_LORAS

        transformer_modules = []
        if isinstance(model_filename, (list, tuple)):
            submodel_no_list = submodel_no_list or [1] * len(model_filename)
            transformer_path = [path for path, submodel_no in zip(model_filename, submodel_no_list) if submodel_no == 1]
            transformer_modules = [path for path, submodel_no in zip(model_filename, submodel_no_list) if submodel_no == 0]
            if len(transformer_path) == 1:
                transformer_path = transformer_path[0]
        else:
            transformer_path = model_filename
        checkpoint_paths = _resolve_multi_file_paths(model_def, base_model_type, include_spatial_upsampler=not model_type.startswith("ltx2_upsampler_"))
        checkpoint_paths["transformer"] = transformer_path
        if transformer_modules:
            checkpoint_paths["transformer_modules"] = transformer_modules

        ltx2_model = LTX2(
            model_filename=model_filename,
            model_type=model_type,
            base_model_type=base_model_type,
            model_def=model_def,
            dtype=dtype,
            VAE_dtype=VAE_dtype,
            text_encoder_filename=text_encoder_filename,
            text_encoder_filepath=(
                model_def.get("text_encoder_folder")
                or (os.path.dirname(text_encoder_filename) if text_encoder_filename else None)
            ),
            checkpoint_paths=checkpoint_paths,
        )

        if save_quantized:
            from wgp import save_quantized_model

            quantized_source = transformer_path[0] if isinstance(transformer_path, (list, tuple)) else transformer_path
            quantized_transformer = getattr(ltx2_model.model, "velocity_model", ltx2_model.model)
            save_quantized_model(
                quantized_transformer,
                model_type,
                quantized_source,
                dtype,
                checkpoint_paths["model_config"],
            )

        pipe = {
            "transformer": ltx2_model.model,
            "text_encoder": ltx2_model.text_encoder,
            "text_embedding_projection": ltx2_model.text_embedding_projection,
            "vae": ltx2_model.video_decoder,
            "video_encoder": ltx2_model.video_encoder,
            "audio_encoder": ltx2_model.audio_encoder,
            "audio_decoder": ltx2_model.audio_decoder,
            "vocoder": ltx2_model.vocoder,
        }
        if ltx2_model.spatial_upsampler is not None:
            pipe["spatial_upsampler"] = ltx2_model.spatial_upsampler
        if ltx2_model.split_text_connectors:
            pipe["video_embeddings_connector"] = ltx2_model.text_embeddings_connector.video_embeddings_connector
            pipe["audio_embeddings_connector"] = ltx2_model.text_embeddings_connector.audio_embeddings_connector
        else:
            pipe["text_embeddings_connector"] = ltx2_model.text_embeddings_connector
        if ltx2_model.model2 is not None:
            pipe["transformer2"] = ltx2_model.model2

        if LTX2_ENABLE_EMBEDDING_LORAS:
            embedding_lora_modules = ["text_embedding_projection", "video_embeddings_connector", "audio_embeddings_connector"] if ltx2_model.split_text_connectors else ["text_embedding_projection", "text_embeddings_connector"]
            pipe = {"pipe": pipe, "loras": embedding_lora_modules}

        return ltx2_model, pipe

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        default_perturbation_layers = _default_perturbation_layers(base_model_type)
        pipeline_kind = model_def.get("ltx2_pipeline", "two_stage")
        ui_defaults.setdefault("attention_sparsity", SOL_ATTN_TAU_START_DEFAULT)
        if _is_joyai_echo(base_model_type, model_def):
            ui_defaults.setdefault("resolution", "1280x720")
            ui_defaults.setdefault("video_length", 129)
            ui_defaults.update(_joyai_settings(ui_defaults.get("custom_settings"), video_prompt_type=ui_defaults.get("video_prompt_type", ""), image_prompt_type=ui_defaults.get("image_prompt_type", ""), guidance_phases=ui_defaults.get("guidance_phases", 2)))
            return
        if pipeline_kind != "distilled" and ui_defaults.get("sample_solver", "") in {"", None}:
            ui_defaults["sample_solver"] = "euler"

        if settings_version < 2.43:
            ui_defaults.update(
                {
                    "denoising_strength": 1.0,
                    "masking_strength": 0,
                }
            )

        if settings_version < 2.45:
            ui_defaults.update(
                {
                    "alt_guidance_scale": 1.0,
                    "perturbation_layers": default_perturbation_layers,
                }
            )

        if settings_version < 2.49:
            ui_defaults.update(
                {
                    "self_refiner_plan": "2-8:3",
                }
            )

        if settings_version < 2.55 and pipeline_kind != "distilled":
            ui_defaults.update({
                "audio_guidance_scale": 1.0,
                "alt_guidance_scale": 1.0,
                "alt_scale": 0.0,
                })

                # _default_dev_settings(base_model_type)

        if settings_version < 2.52:
            plan = ui_defaults.get("self_refiner_plan")
            if isinstance(plan, list):
                from shared.utils.self_refiner import convert_refiner_list_to_string
                ui_defaults["self_refiner_plan"] = convert_refiner_list_to_string(plan)

        if settings_version < 2.58 and pipeline_kind == "distilled":
            ui_defaults["guidance_phases"]=2

        if settings_version < 2.65: 
            audio_prompt_type = ui_defaults.get("audio_prompt_type", None)
            if audio_prompt_type is not None and "L" in audio_prompt_type:
                audio_prompt_type =audio_prompt_type.replace("L", "")
                ui_defaults["audio_prompt_type"] = audio_prompt_type

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        default_perturbation_layers = _default_perturbation_layers(base_model_type)
        ui_defaults.update(
            {
                "sliding_window_size": 145 if model_def.get("ltx2_alpha_gen", False) else 481,
                "sliding_window_overlap": 17,
                "denoising_strength": 1.0,
                "masking_strength": 0,
                "audio_prompt_type": "",
                "perturbation_layers": default_perturbation_layers,
                "guidance_phases": 2,
                "attention_sparsity": SOL_ATTN_TAU_START_DEFAULT,
	            }
        )
        ui_defaults.setdefault("audio_scale", 1.0)
        pipeline_kind = model_def.get("ltx2_pipeline", "two_stage")
        if pipeline_kind != "distilled":
            ui_defaults.update(_default_dev_settings(base_model_type))
            ui_defaults.setdefault("sample_solver", "euler")
        if _is_editanything_model(model_def):
            ui_defaults.update(
                {
                    "audio_prompt_type": "K",
                    "video_prompt_type": "VGI",
                    "remove_background_images_ref": 1,
                }
            )
        if _is_joyai_echo(base_model_type, model_def):
            ui_defaults.update(_joyai_settings(ui_defaults.get("custom_settings")))
        if _is_msr_model(base_model_type, model_def):
            ui_defaults.update(
                {
                    "video_prompt_type": "KI",
                    "audio_prompt_type": "",
                    "video_length": 145,
                    "resolution": "1280x704" if _is_ltx25(base_model_type) else "1280x720",
                    "force_fps": "",
                    "remove_background_images_ref": 1,
                    "guidance_phases": 2,
                }
            )

    @staticmethod
    def get_custom_prompt_enhancer_instructions(model_type, prompt_enhancer_mode, is_image, enhancer_kwargs):
        if model_type == "joyai_echo":
            from .joyai_echo import JOYAI_ECHO_PROMPT_ENHANCER

            return JOYAI_ECHO_PROMPT_ENHANCER, 4096
        from .prompt_enhancer import  get_custom_prompt_enhancer_instructions
        return get_custom_prompt_enhancer_instructions(model_type, prompt_enhancer_mode, is_image, enhancer_kwargs)
