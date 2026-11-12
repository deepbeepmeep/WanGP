from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr
import torch
from safetensors.torch import save_file

from postprocessing import spatial_upsamplers as upsampler_api
from postprocessing.flashvsr.wgp_bridge import FlashVSRBridge
from shared.utils.virtual_media import build_virtual_media_path


class FlashVSRProcessHandler:
    system_handler = "flashvsr"
    model_type = "__system_flashvsr"
    model_label = "WanGP System Video Postprocessing"
    target_control_label = "Upsampling"
    target_control_choices = [(f"x{FlashVSRBridge.format_ratio_label(scale)}", FlashVSRBridge.upsampling_value(scale)) for scale in FlashVSRBridge.UPSAMPLING_RATIOS]
    default_target_control = FlashVSRBridge.upsampling_value(2.0)
    default_chunk_size_seconds = 3.0
    frame_step = 1
    minimum_requested_frames = 1
    hide_sliding_window_overlap = True
    hide_output_resolution = True
    hide_prompt = True

    def configured_method(self, method: str) -> str:
        bridge = upsampler_api.require_upsampler_by_method(method)
        model = bridge.model()
        two_pass = bridge.is_two_pass_upsampling(method)
        return next(key for key, version in bridge.METHOD_MODELS.items() if version == model and bridge.is_two_pass_upsampling(key) == two_pass)

    @property
    def overlap_frames(self) -> int:
        """Default overlap for the original FlashVSR v1.1 process."""
        return 11

    def get_overlap_frames(self, chunk_frames: int) -> int:
        return max(0, min(int(self.overlap_frames), int(chunk_frames) - 1))

    def get_overlap_frames_for_target(self, chunk_frames: int, target_control: str) -> int:
        split = FlashVSRBridge.split_value(target_control)
        if split is None:
            return self.get_overlap_frames(chunk_frames)
        overlap = upsampler_api.require_upsampler_by_method(split[0]).overlap_frames(target_control)
        return max(0, min(overlap, int(chunk_frames) - 1))

    def normalize_target_control(self, value: str | None) -> str:
        value = str(value or "").strip()
        scale = scale_for_lanczos(value)
        if scale is not None:
            return upsampling_value("lanczos", scale)
        split = FlashVSRBridge.split_value(value)
        if split is not None:
            return upsampling_value(*split)
        scale = scale_for_any(value)
        return FlashVSRBridge.upsampling_value(scale) if scale is not None else self.default_target_control

    def target_control_choices_for_process(self, process_settings: dict) -> list[tuple[str, str]]:
        prefix = upsampling_prefix_for_process(process_settings)
        return [(f"x{FlashVSRBridge.format_ratio_label(scale)}", upsampling_value(prefix, scale)) for scale in FlashVSRBridge.UPSAMPLING_RATIOS]

    def target_control_default_for_process(self, process_settings: dict) -> str:
        return self.normalize_target_control_for_process(process_settings.get("target_ratio"), process_settings)

    def normalize_target_control_for_process(self, value: str | None, process_settings: dict) -> str:
        scale = scale_for_any(value) or scale_for_any(process_settings.get("target_ratio")) or 2.0
        return upsampling_value(upsampling_prefix_for_process(process_settings), scale if scale in FlashVSRBridge.UPSAMPLING_RATIOS else 2.0)

    def output_resolution_token(self, value: str | None) -> str:
        value = self.normalize_target_control(value)
        scale = scale_for_lanczos(value) or FlashVSRBridge.scale_for_upsampling(value) or 2.0
        method = "lanczos" if value.startswith("lanczos") else FlashVSRBridge.split_value(value)[0]
        prefix = "" if method == FlashVSRBridge.UPSAMPLING_VALUE_PREFIX else f"{method}-"
        return f"{prefix}x{FlashVSRBridge.format_ratio(scale)}"

    def build_queue_settings(self, process_settings: dict, *, source_path: str, start_frame: int, frame_count: int, target_control: str, seed: int, continue_cache: Any, audio_track_no: int | None = None) -> dict:
        target_control = self.normalize_target_control_for_process(target_control, process_settings)
        video_path = build_virtual_media_path(source_path, start_frame=start_frame, end_frame=start_frame + frame_count - 1, audio_track_no=audio_track_no)
        api_options = dict(process_settings.get("_api", {})) if isinstance(process_settings.get("_api"), dict) else {}
        api_options.update({"return_media": True, "suppress_source_audio": False, "suppress_metadata_images": True})
        if self.supports_continue_cache_for_target(target_control):
            api_options.update({"return_flashvsr_continue_cache": True, "flashvsr_continue_cache": continue_cache})
        else:
            api_options.pop("return_flashvsr_continue_cache", None)
            api_options.pop("flashvsr_continue_cache", None)
        settings = dict(process_settings)
        settings.update({
            "mode": "edit_postprocessing",
            "model_type": self.model_type,
            "prompt": str(settings.get("prompt") or "FlashVSR upsampling"),
            "image_mode": 0,
            "video_source": video_path,
            "video_length": int(frame_count),
            "keep_frames_video_source": str(int(frame_count)),
            "temporal_upsampling": "",
            "spatial_upsampling": target_control,
            "film_grain_intensity": 0,
            "film_grain_saturation": 0.5,
            "postprocess_audio": "",
            "repeat_generation": 1,
            "batch_size": 1,
            "seed": int(seed),
            "_api": api_options,
        })
        return settings

    def build_image_queue_settings(self, process_settings: dict, *, source_path: str, target_control: str, seed: int) -> dict:
        target_control = self.normalize_target_control_for_process(target_control, process_settings)
        api_options = dict(process_settings.get("_api", {})) if isinstance(process_settings.get("_api"), dict) else {}
        api_options.update({"return_media": True, "suppress_source_audio": True, "suppress_metadata_images": True})
        settings = dict(process_settings)
        settings.update({
            "mode": "edit_postprocessing",
            "model_type": str(settings.get("model_type") or "__system_image_postprocessing"),
            "prompt": str(settings.get("prompt") or "Image upsampling"),
            "image_mode": 1,
            "video_source": source_path,
            "video_length": 1,
            "keep_frames_video_source": "1",
            "temporal_upsampling": "",
            "spatial_upsampling": target_control,
            "film_grain_intensity": 0,
            "film_grain_saturation": 0.5,
            "postprocess_audio": "",
            "repeat_generation": 1,
            "batch_size": 1,
            "seed": int(seed),
            "_api": api_options,
        })
        return settings

    def supports_continue_cache(self) -> bool:
        return True

    def supports_continue_cache_for_target(self, value: str | None) -> bool:
        value = self.normalize_target_control(value)
        return FlashVSRBridge.scale_for_upsampling(value) is not None

    carries_cache = supports_continue_cache_for_target  # each chunk hands its last frames to the next one

    def cache_sidecar_path(self, output_filename: str) -> str:
        output_path = Path(output_filename).resolve()
        return str(output_path.with_suffix(output_path.suffix + ".flashvsr_cache.safetensors"))

    def can_resume_without_output_metadata(self, output_filename: str) -> bool:
        return Path(self.cache_sidecar_path(output_filename)).is_file() or Path(output_filename).is_file()

    def move_continue_cache(self, source_output_filename: str, target_output_filename: str) -> bool:
        source_path = Path(self.cache_sidecar_path(source_output_filename))
        if not source_path.is_file():
            return False
        target_path = Path(self.cache_sidecar_path(target_output_filename))
        target_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.replace(target_path)
        return True

    def delete_continue_cache(self, output_filename: str) -> None:
        cache_path = Path(self.cache_sidecar_path(output_filename))
        if cache_path.is_file():
            cache_path.unlink()

    def save_continue_cache(self, cache: Any, output_filename: str, metadata: dict | None = None) -> str:
        if not isinstance(cache, dict):
            return ""
        tail = _cache_tail_to_uint8(cache.get("tail_frames"))
        if tail is None:
            return ""
        tensors = {"tail_frames": tail}
        shifted_tail = _cache_tail_to_uint8(cache.get("tail_frames_shifted"))
        if shifted_tail is not None:
            tensors["tail_frames_shifted"] = shifted_tail
        cache_metadata = {
            "version": "2" if shifted_tail is not None else "1",
            "handler": self.system_handler,
            "scale": str(cache.get("scale", "")),
            "variant": str(cache.get("variant", "")),
            "metadata": json.dumps(metadata or {}, ensure_ascii=True, sort_keys=True),
        }
        cache_metadata.update({key: str(cache[key]) for key in ("two_pass", "shift_y", "shift_x", "out_shift_y", "out_shift_x") if key in cache})
        sidecar_path = self.cache_sidecar_path(output_filename)
        Path(sidecar_path).parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, sidecar_path, metadata=cache_metadata)
        return sidecar_path

    def load_continue_cache(self, output_filename: str) -> Any:
        sidecar_path = self.cache_sidecar_path(output_filename)
        if not Path(sidecar_path).is_file():
            raise gr.Error(f"FlashVSR continuation cache is missing: {sidecar_path}")
        from safetensors import safe_open
        with safe_open(sidecar_path, framework="pt", device="cpu") as handle:
            metadata = dict(handle.metadata() or {})
            cache = {"tail_frames": _load_tail_tensor(handle, "tail_frames", sidecar_path), "scale": _coerce_float(metadata.get("scale"), 0.0), "variant": str(metadata.get("variant") or "")}
            if "tail_frames_shifted" in set(handle.keys()):
                cache["tail_frames_shifted"] = _load_tail_tensor(handle, "tail_frames_shifted", sidecar_path)
        cache.update({key: _coerce_float(metadata.get(key), 0.0) for key in ("shift_y", "shift_x", "out_shift_y", "out_shift_x") if key in metadata})
        if "two_pass" in metadata:
            cache["two_pass"] = str(metadata.get("two_pass")).lower() == "true"
        return cache

    def continue_cache_from_tail_frames(self, tail_frames: Any, target_control: str | None = None) -> Any:
        tail = _cache_tail_to_uint8(tail_frames)
        if tail is None:
            return None
        return {"tail_frames": tail, "scale": FlashVSRBridge.scale_for_upsampling(self.normalize_target_control(target_control)) or 0.0, "variant": "", "fallback": True}


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _cache_tail_to_uint8(tail: Any) -> torch.Tensor | None:
    if not torch.is_tensor(tail) or tail.ndim != 4 or int(tail.shape[1]) <= 0:
        return None
    if tail.dtype == torch.uint8:
        return tail.detach().cpu().contiguous()
    return tail.detach().cpu().float().clamp(-1.0, 1.0).add(1.0).mul_(127.5).round_().clamp_(0, 255).to(torch.uint8).contiguous()


def _load_tail_tensor(handle, key: str, sidecar_path: str) -> torch.Tensor:
    tail = handle.get_tensor(key)
    if not torch.is_tensor(tail) or tail.ndim != 4:
        raise gr.Error(f"FlashVSR continuation cache is invalid: {sidecar_path}")
    return tail.clone().contiguous() if tail.dtype == torch.uint8 else tail.float().clamp_(-1.0, 1.0).contiguous()


def scale_for_lanczos(value: str | None) -> float | None:
    scale = upsampler_api.parse_multiplier_suffix(value, "lanczos", 2.0)
    return scale if scale in FlashVSRBridge.UPSAMPLING_RATIOS else None


def scale_for_any(value: str | None) -> float | None:
    text = str(value or "").strip()
    if len(text) == 0:
        return None
    scale = scale_for_lanczos(text) or FlashVSRBridge.scale_for_upsampling(text)
    if scale is not None:
        return scale
    try:
        scale = float(text)
    except ValueError:
        return None
    return scale if scale in FlashVSRBridge.UPSAMPLING_RATIOS else None


def upsampling_prefix_for_process(process_settings: dict | None) -> str:
    settings = process_settings if isinstance(process_settings, dict) else {}
    method = str(settings.get("spatial_upsampling_method") or "").strip().lower()
    if method == "lanczos" or method in FlashVSRBridge.METHOD_MODELS:
        return method
    target = str(settings.get("target_ratio") or "").strip().lower()
    if target.startswith("lanczos"):
        return "lanczos"
    split = FlashVSRBridge.split_value(target)
    return split[0] if split is not None else FlashVSRBridge.UPSAMPLING_VALUE_PREFIX


def upsampling_value(prefix: str, scale: float) -> str:
    return upsampler_api.format_multiplier_value(prefix, scale)


HANDLER = FlashVSRProcessHandler()
