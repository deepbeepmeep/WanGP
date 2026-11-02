# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Setting-name validation shared by every generation entry point.

Accepted names are the shared catalog in models/_settings.json, the non-setting keys that travel with
settings (identity, versioning, UI transport and the metadata WanGP writes into saved media), and retired
names that older saved settings, queues and defaults may still carry; retired names are ignored rather than rejected.
Other media metadata (written by tools and plugins) is not a setting: split_settings separates it before reuse.
"""

import difflib
import json
from functools import cache
from pathlib import Path

SETTINGS_CATALOG = Path(__file__).resolve().parents[2] / "models" / "_settings.json"

NON_SETTING_KEYS = frozenset({
    "model_type", "base_model_type", "model_filename", "type", "settings_version", "mode", "state",
    "priority", "frame_scheduler", "send_cmd", "task", "target", "image_mask_guide", "loras_choices", "plugin_data",
    "profile_priority", "help", "creation_date", "creation_timestamp", "generation_time", "is_image", "window_no", "custom_guide_used", "_settings_bundle_task_count",
    "image_quality", "video_quality", "hdr", "hdr_video_crf", "modules", "transformer_loras_filenames", "transformer_loras_multipliers",
    "gallery_media_ids", "deepy_media_id", "deepy_media_fingerprint", "deepy_session_id",
})

DEPRECATED_SETTING_KEYS = frozenset({
    "prompts", "lset_name", "slg_switch", "slg_layers", "slg_start_perc", "slg_end_perc", "tea_cache_setting", "tea_cache_start_step_perc", "spatial_upsampler_face_count", "spatial_upsampler_parameters", "postprocessing_parameters",
    "temporal_upsampler_parameters", "spatial_upsampler_prompt", "spatial_upsampler_reference_images", "spatial_upsampler_param", "spatial_upsampler_param2",
    "scheduler_type", "shift", "adaptive_switch", "MMAudio_setting", "MMAudio_prompt", "MMAudio_neg_prompt", "seedvc_voice_sample", "seedvc_voice_sample2",
    "fps", "sampling_steps", "sampler_solver", "sample_solvers", "enhanced_prompt", "enhanced_alt_prompt", "audio_postprocess", "audio_cfg_scale", "pace", "exaggeration",
    "override_profile", "override_compile", "override_attention",
})

# Names commonly guessed for an existing setting, and what to use instead.
SETTING_HINTS = {
    "steps": "num_inference_steps", "num_steps": "num_inference_steps", "inference_steps": "num_inference_steps",
    "width": 'resolution ("WIDTHxHEIGHT")', "height": 'resolution ("WIDTHxHEIGHT")', "size": 'resolution ("WIDTHxHEIGHT")', "aspect_ratio": 'resolution ("WIDTHxHEIGHT")',
    "cfg": "guidance_scale", "cfg_scale": "guidance_scale", "guidance": "guidance_scale",
    "sampler": "sample_solver", "scheduler": "sample_solver", "solver": "sample_solver",
    "frames": "video_length", "num_frames": "video_length", "duration": "duration_seconds", "negative": "negative_prompt",
    "lora": "activated_loras", "loras": "activated_loras", **dict.fromkeys(("name", "filename", "output_name"), "output_filename"),
    **dict.fromkeys(("accelerator", "accelerator_profile", "profile"), "an accelerator profile's settings"),
}
# Model information returned beside template settings and in model metadata: never a setting.
MODEL_INFO_KEYS = frozenset({"accelerated", "media_inputs", "resolution_budget", "resolution_budget_tiled", "resolution_min", "max_width", "max_height", "lock_resolutions", "allowed_resolutions", "vae_block_size"})


@cache
def catalog_setting_names() -> frozenset:
    return frozenset(json.loads(SETTINGS_CATALOG.read_text(encoding="utf-8")))


# String settings that agents easily send as a list or a number, which the generation code cannot parse.
STRING_FORMATS = {"frames_positions": 'frame numbers separated by spaces, such as "181" or "1 97 181"', "resolution": '"WIDTHxHEIGHT"'}


def accepted_setting_names() -> frozenset:
    return catalog_setting_names() | NON_SETTING_KEYS | DEPRECATED_SETTING_KEYS


def unknown_setting_names(settings) -> list:
    accepted = accepted_setting_names()
    return [name for name in settings if name not in accepted]


def split_settings(settings) -> tuple:
    """Separate reusable settings from other metadata, such as saved-media details written by tools."""
    accepted = accepted_setting_names()
    return {name: value for name, value in settings.items() if name in accepted}, {name: value for name, value in settings.items() if name not in accepted}


def settings_error(settings) -> str:
    """Unknown setting names, or a string setting given another type, as one message; empty when the settings are acceptable."""
    wrong = [name for name in STRING_FORMATS if settings.get(name) is not None and not isinstance(settings[name], str)]
    if wrong:
        return " ".join(f"{name} must be a string{': ' + STRING_FORMATS[name] if name in STRING_FORMATS else ''}, not {type(settings[name]).__name__} {settings[name]!r}." for name in wrong)
    unknown = unknown_setting_names(settings)
    if not unknown:
        return ""
    described = []
    for name in unknown:
        if name in MODEL_INFO_KEYS:
            described.append(f"{name} (model information, not a setting: omit it)")
            continue
        hint = SETTING_HINTS.get(name.lower()) or next(iter(difflib.get_close_matches(name, catalog_setting_names(), n=1, cutoff=0.75)), "")
        described.append(f"{name} (use {hint})" if hint else name)
    return f"Unknown setting{'s' if len(unknown) > 1 else ''}: {', '.join(described)}. Model-specific options belong in custom_settings."
