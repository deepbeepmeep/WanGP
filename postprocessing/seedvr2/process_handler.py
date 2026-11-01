from __future__ import annotations

from typing import Any

from postprocessing import spatial_upsamplers as upsampler_api
from shared.utils.virtual_media import build_virtual_media_path
from .wgp_bridge import SeedVR2Bridge


class SeedVR2ProcessHandler:
    system_handler = "seedvr2"
    model_type = "__system_seedvr2"
    model_label = "WanGP System Video Postprocessing"
    target_control_label = "Upsampling"
    target_control_choices = [(f"x{SeedVR2Bridge.format_ratio(scale)}", SeedVR2Bridge.upsampling_value(scale)) for scale in SeedVR2Bridge.UPSAMPLING_RATIOS]
    default_target_control = SeedVR2Bridge.upsampling_value(2.0)
    default_chunk_size_seconds = 3.0
    chunk_frame_step = 4
    frame_step = 1
    minimum_requested_frames = 1
    overlap_frames = SeedVR2Bridge.WINDOW_OVERLAP_FRAMES
    crossfade_overlap_outputs = True
    hide_chunk_size = True
    hide_sliding_window_overlap = True
    hide_output_resolution = True
    hide_prompt = True

    def configured_method(self, method: str = SeedVR2Bridge.UPSAMPLING_VALUE_PREFIX) -> str:
        return upsampler_api.require_upsampler_by_method(method).method()

    def get_overlap_frames(self, chunk_frames: int) -> int:
        return max(0, min(self.overlap_frames, int(chunk_frames) - 1))

    def get_chunk_frames(self, selected_frame_count: int) -> int:
        window_size = SeedVR2Bridge.resolve_model_window_size(upsampler_api.config_for_method(SeedVR2Bridge.UPSAMPLING_VALUE_PREFIX)["window_size"])
        return int(selected_frame_count) if window_size == SeedVR2Bridge.WINDOW_SIZE_UNLIMITED else window_size

    def normalize_target_control(self, value: str | None) -> str:
        """The scale of a SeedVR2 value on the model chosen in Configuration > Extensions."""
        return self.normalize_target_control_for_process(value, {})

    def target_control_choices_for_process(self, process_settings: dict) -> list[tuple[str, str]]:
        method = process_settings.get("spatial_upsampling_method") or self.configured_method()
        return [(f"x{SeedVR2Bridge.format_ratio(scale)}", SeedVR2Bridge.upsampling_value(scale, method)) for scale in SeedVR2Bridge.UPSAMPLING_RATIOS]

    def target_control_default_for_process(self, process_settings: dict) -> str:
        return self.normalize_target_control_for_process(None, process_settings)

    def normalize_target_control_for_process(self, value: str | None, process_settings: dict) -> str:
        method = process_settings.get("spatial_upsampling_method") or self.configured_method()
        scale = _scale_for_value(value or process_settings.get("target_ratio"))
        return SeedVR2Bridge.upsampling_value(scale if scale in SeedVR2Bridge.UPSAMPLING_RATIOS else 2.0, method)

    def output_resolution_token(self, value: str | None) -> str:
        method, scale = upsampler_api.split_upsampling_value(value) or (self.configured_method(), 2.0)
        model = SeedVR2Bridge.METHOD_MODELS[method]
        prefix = "" if model == SeedVR2Bridge.MODEL_3B else f"{model}-"
        return f"{prefix}x{SeedVR2Bridge.format_ratio(scale)}"

    def build_queue_settings(self, process_settings: dict, *, source_path: str, start_frame: int, frame_count: int, target_control: str, seed: int, continue_cache: Any, audio_track_no: int | None = None) -> dict:
        target_control = self.normalize_target_control_for_process(target_control, process_settings)
        video_path = build_virtual_media_path(source_path, start_frame=start_frame, end_frame=start_frame + frame_count - 1, audio_track_no=audio_track_no)
        api_options = dict(process_settings.get("_api", {})) if isinstance(process_settings.get("_api"), dict) else {}
        api_options.update({"return_media": True, "suppress_source_audio": False, "suppress_metadata_images": True})
        chunk_seed = int(process_settings.get("seed", 0))
        settings = dict(process_settings)
        settings.update({
            "mode": "edit_postprocessing",
            "model_type": self.model_type,
            "prompt": str(settings.get("prompt") or "SeedVR2 upsampling"),
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
            "seed": chunk_seed,
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
            "prompt": str(settings.get("prompt") or "SeedVR2 image upsampling"),
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
        return False

    def supports_continue_cache_for_target(self, value: str | None) -> bool:
        return False


def _scale_for_value(value: str | None) -> float | None:
    split = upsampler_api.split_upsampling_value(value)
    return split[1] if split is not None and split[0] in SeedVR2Bridge.METHOD_MODELS else None


HANDLER = SeedVR2ProcessHandler()
