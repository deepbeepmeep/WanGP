# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

from typing import Any

from shared import extra_settings


_SLIDING = {
    "sliding_window_size", "sliding_window_overlap", "sub_parallel_window_size", "sub_parallel_window_overlap",
    "sliding_window_color_correction_strength", "sliding_window_overlap_noise", "sliding_window_discard_last_frames", "sliding_window_trim_first_frames",
}
_AUDIO_POSTPROCESSING = {"postprocess_audio", "postprocess_audio_prompt", "postprocess_audio_neg_prompt"}
_VISUAL_POSTPROCESSING = {"temporal_upsampling", "spatial_upsampling", "film_grain_intensity", "film_grain_saturation"}
_VIDEO_MEDIA_OPTIONS = {
    "force_fps", "keep_frames_video_source", "keep_frames_video_guide", "video_guide_outpainting", "video_guide_outpainting_ratio",
    "mask_expand", "frames_positions", "image_refs_relative_size", "remove_background_images_ref", "input_video_strength",
}
_NEUTRAL = {
    "client_id": {None, ""},
}


def _drop(settings: dict[str, Any], keys) -> None:
    for key in keys:
        settings.pop(key, None)


def _number(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clean_metadata_settings(settings: dict[str, Any], model_def: dict[str, Any], *, attention_mode: str = "", prompt_enhancer_visible: bool = False) -> dict[str, Any]:
    model_type = str(settings.get("model_type", "") or "")
    audio_only = bool(model_def.get("audio_only", False))
    context = {"model_type": model_type, "guidance_phases": settings.get("guidance_phases", 1)}
    declared = extra_settings.iter_defs(model_def, **context)
    visible = extra_settings.iter_defs(model_def, only_visible=True, **context)
    _drop(settings, declared.keys() - visible.keys())

    image_output = not audio_only and int(settings.get("image_mode", 0) or 0) > 0
    video_output = not audio_only and not image_output
    image_prompt_type = "" if audio_only else settings.get("image_prompt_type", "")
    video_prompt_type = "" if audio_only else settings.get("video_prompt_type", "")
    guide_choices = model_def.get("guide_custom_choices_image") if image_output and model_def.get("guide_custom_choices_image") is not None else model_def.get("guide_custom_choices")
    guidance_max_phases = int(model_def.get("guidance_max_phases", 0) or 0)
    guidance_phases = int(settings.get("guidance_phases", 1) or 0)
    postprocess_audio_meta = None
    if not image_output and str(settings.get("postprocess_audio", "") or ""):
        from postprocessing import audio_processors

        postprocess_audio_meta = audio_processors.method_metadata(settings["postprocess_audio"])

    capabilities = (
        (audio_only and model_def.get("temperature", True), {"temperature"}),
        (model_def.get("duration_slider") is not None, {"duration_seconds"}),
        (model_def.get("pause_between_sentences", False), {"pause_seconds"}),
        (model_def.get("world_outputs", False), {"world_output"}),
        (model_def.get("inference_steps", True), {"num_inference_steps"}),
        (model_def.get("sample_solvers") is not None, {"sample_solver"}),
        (model_def.get("self_refiner", False), {"self_refiner_setting", "self_refiner_plan", "self_refiner_f_uncertainty", "self_refiner_certain_percentage"}),
        (model_def.get("audio_guidance", False), {"speakers_locations"}),
        (any(model_def.get(key, False) for key in ("tea_cache", "mag_cache", "spectrum_cache", "first_block_cache")), {"skip_steps_cache_type", "skip_steps_multiplier", "skip_steps_start_step_perc"}),
        (model_def.get("perturbation", False), {"perturbation_switch", "perturbation_layers", "perturbation_start_perc", "perturbation_end_perc"}),
        (model_def.get("cfg_zero", False), {"cfg_zero_step"}),
        (model_def.get("cfg_star", False), {"cfg_star_switch"}),
        (model_def.get("adaptive_projected_guidance", False), {"apg_switch"}),
        (model_def.get("NAG", False), {"NAG_scale", "NAG_tau", "NAG_alpha"}),
        (model_def.get("riflex", False), {"RIFLEx_setting"}),
        (guidance_max_phases >= 1, {"guidance_scale", "guidance_phases"}),
        (guidance_max_phases >= 2 and guidance_phases >= 2, {"guidance2_scale", "switch_threshold"}),
        (guidance_max_phases >= 3 and guidance_phases >= 3, {"guidance3_scale", "switch_threshold2", "model_switch_phase"}),
        (model_def.get("alt_prompt") is not None, {"alt_prompt"}),
        (model_def.get("custom_settings") is not None, {"custom_settings"}),
        (model_def.get("model_modes") is not None, {"model_mode"}),
        (model_def.get("custom_attention_modes", {}).get(attention_mode, {}).get("supports_sparsity", False), {"attention_sparsity"}),
        (not audio_only, {"resolution", "batch_size", "video_length", "force_fps", "image_mode", "image_prompt_type", "video_prompt_type", "RIFLEx_setting", "skip_steps_cache_type", *_SLIDING, *_VISUAL_POSTPROCESSING, *_VIDEO_MEDIA_OPTIONS}),
        (video_output, {"video_length", "force_fps", "keep_frames_video_source", "keep_frames_video_guide", "RIFLEx_setting", "skip_steps_cache_type", *_SLIDING}),
        (video_output or model_def.get("mesh_outputs", False), {"temporal_upsampling"}),  # the Texturer of 3D assets
        (image_output, {"batch_size"}),
        (not image_output, _AUDIO_POSTPROCESSING),
        (guide_choices is not None or model_def.get("guide_preprocessing") is not None, {"keep_frames_video_guide", "mask_expand"}),
        ("G" in video_prompt_type or model_def.get("mask_strength_always_enabled", False), {"masking_strength"}),
        (model_def.get("input_video_strength") is not None, {"input_video_strength"}),
        (model_def.get("any_image_refs_relative_size", False), {"image_refs_relative_size"}),
        (model_def.get("video_guide_outpainting") is not None, {"video_guide_outpainting", "video_guide_outpainting_ratio"}),
        (model_def.get("vace_class", False) or model_def.get("t2v_class", False), {"min_frames_if_references"}),
        (model_def.get("multiple_images_as_text_prompts", False), {"multi_images_gen_type"}),
        (not model_def.get("no_negative_prompt", False), {"negative_prompt"}),
        (model_def.get("audio_scale_name") is not None, {"audio_scale"}),
        (postprocess_audio_meta is not None and (postprocess_audio_meta["needs_prompt"] or postprocess_audio_meta["needs_negative_prompt"]) and not (model_def.get("returns_audio", False) or model_def.get("any_audio_prompt", False)), {"postprocess_audio_prompt", "postprocess_audio_neg_prompt"}),
        (not model_def.get("no_lora", False), {"activated_loras", "loras_multipliers"}),
        (video_output and model_def.get("sliding_window", False), _SLIDING),
        (video_output and model_def.get("sliding_window", False) and model_def.get("sub_parallel_windows", False), {"sub_parallel_window_size", "sub_parallel_window_overlap"}),
    )
    for supported, keys in capabilities:
        if not supported:
            _drop(settings, keys)

    for key, neutral_values in _NEUTRAL.items():
        if settings.get(key) in neutral_values:
            settings.pop(key, None)

    if "G" not in video_prompt_type:
        settings.pop("denoising_strength", None)
    if not any(flag in image_prompt_type for flag in "VL"):
        settings.pop("keep_frames_video_source", None)
    if "V" not in video_prompt_type:
        settings.pop("keep_frames_video_guide", None)
    if "A" not in video_prompt_type:
        settings.pop("mask_expand", None)
    if "I" not in video_prompt_type:
        _drop(settings, {"image_refs_relative_size", "remove_background_images_ref"})
    if "F" not in video_prompt_type:
        settings.pop("frames_positions", None)

    if _number(settings.get("film_grain_intensity"), 0) <= 0:
        _drop(settings, {"film_grain_intensity", "film_grain_saturation"})
    if not str(settings.get("postprocess_audio", "") or ""):
        _drop(settings, {"postprocess_audio_prompt", "postprocess_audio_neg_prompt"})

    if len(settings.get("skip_steps_cache_type", "")) == 0 :
        _drop(settings, {"skip_steps_multiplier", "skip_steps_start_step_perc"})
    if _number(settings.get("perturbation_switch"), 0) <= 0:
        _drop(settings, {"perturbation_layers", "perturbation_start_perc", "perturbation_end_perc"})
    if _number(settings.get("self_refiner_setting"), 0) <= 0:
        _drop(settings, {"self_refiner_plan", "self_refiner_f_uncertainty", "self_refiner_certain_percentage"})
    if _number(settings.get("NAG_scale"), 1) <= 1:
        _drop(settings, {"NAG_tau", "NAG_alpha"})

    _drop(settings, [key for key, value in settings.items() if value is None])
    return settings


__all__ = ["clean_metadata_settings"]
