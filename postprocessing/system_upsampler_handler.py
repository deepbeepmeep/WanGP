# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shared.utils.virtual_media import build_virtual_media_path


@dataclass(frozen=True)
class SystemUpsamplerProcessHandler:
    system_handler: str
    method: str
    temporal: bool
    prompt: str
    overlap_frames: int = 1  # source frames a chunk shares with the one before it

    model_type = "__system_flashvsr"
    model_label = "WanGP System Video Postprocessing"
    default_chunk_size_seconds = 3.0
    frame_step = 1
    hide_sliding_window_overlap = True
    hide_output_resolution = True
    hide_prompt = True

    @property
    def target_control_label(self) -> str:
        return f"{'Temporal' if self.temporal else 'Spatial'} Upsampling Multiplier"

    @property
    def minimum_requested_frames(self) -> int:
        return 2 if self.temporal else 1

    @property
    def crossfade_overlap_outputs(self) -> bool:
        return not self.temporal

    @property
    def disable_continuation(self) -> bool:
        return self.temporal

    def _api(self):
        if self.temporal:
            from postprocessing import temporal_upsamplers
            return temporal_upsamplers
        from postprocessing import spatial_upsamplers
        return spatial_upsamplers

    def _default_scale(self) -> float:
        return self._api().default_multiplier_for_method(self.method)

    def _scale_for_value(self, value: str | float | None) -> float | None:
        api = self._api()
        split = api.split_temporal_upsampling_value(value) if self.temporal else api.split_upsampling_value(value)
        if split is not None and split[0] == self.method:
            return split[1]
        try:
            scale = float(value)
        except (TypeError, ValueError):
            return None
        return scale if scale in api.method_multipliers(self.method) else None

    def _build_value(self, scale: float) -> str:
        api = self._api()
        value = api.build_temporal_upsampling_value(self.method, scale) if self.temporal else api.build_upsampling_value(self.method, scale)
        if value is None:
            raise RuntimeError(f"No registered upsampler owns Media Flow method '{self.method}'")
        return value

    def normalize_target_control(self, value: str | float | None) -> str:
        return self._build_value(self._scale_for_value(value) or self._default_scale())

    def target_control_choices_for_process(self, process_settings: dict) -> list[tuple[str, str]]:
        api = self._api()
        choices = api.multiplier_choices_for_method(self.method) if self.temporal else api.ratio_choices_for_method(self.method)
        return [(label, self._build_value(scale)) for label, scale in choices]

    def target_control_default_for_process(self, process_settings: dict) -> str:
        return self.normalize_target_control(process_settings.get("target_ratio"))

    def normalize_target_control_for_process(self, value: str | None, process_settings: dict) -> str:
        return self.normalize_target_control(value or process_settings.get("target_ratio"))

    def output_resolution_token(self, value: str | None) -> str:
        scale = self._scale_for_value(self.normalize_target_control(value)) or self._default_scale()
        return f"{self.system_handler}-x{self._api().format_multiplier(scale)}"

    def get_overlap_frames(self, chunk_frames: int) -> int:
        return max(0, min(self.overlap_frames, int(chunk_frames) - 1))

    def build_queue_settings(self, process_settings: dict, *, source_path: str, start_frame: int, frame_count: int, target_control: str, seed: int, continue_cache: Any, audio_track_no: int | None = None) -> dict:
        target_control = self.normalize_target_control_for_process(target_control, process_settings)
        video_path = build_virtual_media_path(source_path, start_frame=start_frame, end_frame=start_frame + frame_count - 1, audio_track_no=audio_track_no)
        api_options = dict(process_settings.get("_api", {})) if isinstance(process_settings.get("_api"), dict) else {}
        api_options.update({"return_media": True, "suppress_source_audio": False, "suppress_metadata_images": True})
        settings = dict(process_settings)
        settings.update({
            "mode": "edit_postprocessing",
            "model_type": self.model_type,
            "prompt": str(settings.get("prompt") or self.prompt),
            "image_mode": 0,
            "video_source": video_path,
            "video_length": int(frame_count),
            "keep_frames_video_source": str(int(frame_count)),
            "temporal_upsampling": target_control if self.temporal else "",
            "spatial_upsampling": "" if self.temporal else target_control,
            "film_grain_intensity": 0,
            "film_grain_saturation": 0.5,
            "postprocess_audio": "",
            "repeat_generation": 1,
            "batch_size": 1,
            "seed": int(settings.get("seed", seed)),
            "_api": api_options,
        })
        return settings

    def build_image_queue_settings(self, process_settings: dict, *, source_path: str, target_control: str, seed: int) -> dict:
        settings = self.build_queue_settings(process_settings, source_path=source_path, start_frame=0, frame_count=1, target_control=target_control, seed=seed, continue_cache=None)
        settings.update({"model_type": "__system_image_postprocessing", "image_mode": 1, "video_source": source_path})
        settings["_api"]["suppress_source_audio"] = True
        return settings

    def expected_output_frame_count(self, input_frame_count: int, target_control: str) -> int:
        if not self.temporal:
            return int(input_frame_count)
        scale = int(self._scale_for_value(target_control) or self._default_scale())
        return (int(input_frame_count) - 1) * scale + 1

    def output_write_range(self, returned_frame_count: int, *, source_write_start: int, source_write_end: int, next_overlap_frames: int) -> tuple[int, int]:
        if not self.temporal:
            return source_write_start, source_write_end
        return source_write_start, int(returned_frame_count) - int(next_overlap_frames)

    def output_fps(self, source_fps: float, target_control: str) -> float:
        scale = self._scale_for_value(target_control) or self._default_scale()
        return float(source_fps) * scale if self.temporal else float(source_fps)

    def supports_continue_cache(self) -> bool:
        return False

    def supports_continue_cache_for_target(self, value: str | None) -> bool:
        return False


RIFE_HANDLER = SystemUpsamplerProcessHandler("rife", "rife", True, "RIFE temporal upsampling")
DLSSG_HANDLER = SystemUpsamplerProcessHandler("dlssg", "dlssg", True, "DLSS 5 Frame Generation")
# DLSS 5 builds each frame from the ones before it: a chunk starts cold, and fades in over 16 shared frames while it settles (with a single
# shared frame the picture changed in one frame at every chunk start)
DLSS5_HANDLER = SystemUpsamplerProcessHandler("dlss5", "dlss5", False, "DLSS 5 Neural Rendering", overlap_frames=16)
