# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Media Flow process of RelayVSR: chunks of 15 n + 1 frames, each starting on the keyframe the previous one ended on, the stream handed from
chunk to chunk in memory (runtime.CarriedStream), so that the chunks join as one long upscale would. A chunk that no stream continues (the
first one of a job started or resumed after the source's first frame) rebuilds the keyframe memory from the keyframes before it
(runtime.StreamReplay). Nothing is kept on disk: the stream holds several GB."""
from __future__ import annotations

from typing import Any

from postprocessing import spatial_upsamplers as upsampler_api
from shared.utils.virtual_media import build_virtual_media_path
from .wgp_bridge import METHOD

KEYFRAME_SPACING = 15  # runtime.WINDOW_FRAMES - 1
REPLAY_KEYFRAMES = 3  # runtime.REPLAY_KEYFRAMES


class RelayVSRProcessHandler:
    system_handler = METHOD
    model_type = "__system_flashvsr"
    model_label = "WanGP System Video Postprocessing"
    target_control_label = "Spatial Upsampling Multiplier"
    default_chunk_size_seconds = 3.0
    frame_step = 1
    chunk_frame_step = KEYFRAME_SPACING  # chunks of 15 n + 1 frames: each ends on a keyframe, where the next one starts
    minimum_requested_frames = 1
    overlap_frames = 1  # that keyframe, which the next chunk writes as the previous chunk decoded it
    crossfade_overlap_outputs = False
    hide_sliding_window_overlap = True
    hide_output_resolution = True
    hide_prompt = True

    def _scale_for_value(self, value) -> float | None:
        split = upsampler_api.split_upsampling_value(value)
        return split[1] if split is not None and split[0] == METHOD else None

    def normalize_target_control(self, value) -> str:
        return upsampler_api.build_upsampling_value(METHOD, self._scale_for_value(value) or upsampler_api.default_multiplier_for_method(METHOD))

    def target_control_choices_for_process(self, process_settings: dict) -> list[tuple[str, str]]:
        return [(label, upsampler_api.build_upsampling_value(METHOD, scale)) for label, scale in upsampler_api.ratio_choices_for_method(METHOD)]

    def target_control_default_for_process(self, process_settings: dict) -> str:
        return self.normalize_target_control(process_settings.get("target_ratio"))

    def normalize_target_control_for_process(self, value, process_settings: dict) -> str:
        return self.normalize_target_control(value or process_settings.get("target_ratio"))

    def output_resolution_token(self, value) -> str:
        return f"{METHOD}-x{upsampler_api.format_multiplier(self._scale_for_value(self.normalize_target_control(value)))}"

    def get_overlap_frames(self, chunk_frames: int) -> int:
        return max(0, min(self.overlap_frames, int(chunk_frames) - 1))

    def build_queue_settings(self, process_settings: dict, *, source_path: str, start_frame: int, frame_count: int, target_control: str, seed: int, continue_cache: Any, audio_track_no: int | None = None) -> dict:
        """continue_cache: the stream the previous chunk returned; without one, a chunk after the source's first frame replays the keyframes
        before it. One seed for the whole job (the keyframes' noise follows their source frame)."""
        if continue_cache is None and start_frame > 0:
            from .runtime import StreamReplay
            frames = tuple(frame for frame in range(start_frame - REPLAY_KEYFRAMES * KEYFRAME_SPACING, start_frame, KEYFRAME_SPACING) if frame >= 0)
            continue_cache = StreamReplay(source_path, frames) if frames else None
        api_options = dict(process_settings.get("_api", {})) if isinstance(process_settings.get("_api"), dict) else {}
        api_options.update({"return_media": True, "suppress_source_audio": False, "suppress_metadata_images": True, "upsampler_frame_offset": int(start_frame),
                            "return_flashvsr_continue_cache": True, "flashvsr_continue_cache": continue_cache})
        settings = dict(process_settings)
        settings.update({
            "mode": "edit_postprocessing",
            "model_type": self.model_type,
            "prompt": str(settings.get("prompt") or "RelayVSR upsampling"),
            "image_mode": 0,
            "video_source": build_virtual_media_path(source_path, start_frame=start_frame, end_frame=start_frame + frame_count - 1, audio_track_no=audio_track_no),
            "video_length": int(frame_count),
            "keep_frames_video_source": str(int(frame_count)),
            "temporal_upsampling": "",
            "spatial_upsampling": self.normalize_target_control_for_process(target_control, process_settings),
            "film_grain_intensity": 0,
            "film_grain_saturation": 0.5,
            "postprocess_audio": "",
            "repeat_generation": 1,
            "batch_size": 1,
            "seed": int(process_settings.get("seed", 0)),
            "_api": api_options,
        })
        return settings

    def build_image_queue_settings(self, process_settings: dict, *, source_path: str, target_control: str, seed: int) -> dict:
        settings = self.build_queue_settings(process_settings, source_path=source_path, start_frame=0, frame_count=1, target_control=target_control, seed=seed, continue_cache=None)
        settings.update({"model_type": "__system_image_postprocessing", "image_mode": 1, "video_source": source_path, "prompt": str(process_settings.get("prompt") or "RelayVSR image upsampling")})
        settings["_api"].update({"suppress_source_audio": True, "return_flashvsr_continue_cache": False, "flashvsr_continue_cache": None})
        settings["_api"].pop("upsampler_frame_offset")
        return settings

    def expected_output_frame_count(self, input_frame_count: int, target_control: str) -> int:
        return int(input_frame_count)

    def output_fps(self, source_fps: float, target_control: str) -> float:
        return float(source_fps)

    def carries_cache(self, value) -> bool:
        return True  # the stream each chunk hands to the next one

    def supports_continue_cache(self) -> bool:
        return False  # nothing to save beside the output: a resumed job rebuilds the keyframe memory

    def supports_continue_cache_for_target(self, value) -> bool:
        return False

    def release_continue_cache(self) -> None:
        """The end of a Media Flow run, completed, stopped or failed: the stream its last chunk returned is released."""
        from .runtime import release_stream
        release_stream()


HANDLER = RelayVSRProcessHandler()
