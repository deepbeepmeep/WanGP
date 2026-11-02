from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import gradio as gr
from safetensors import safe_open
from safetensors.torch import save_file

from postprocessing import spatial_upsamplers as upsampler_api
from shared.utils.virtual_media import build_virtual_media_path

from . import runtime as refine_runtime
from .runtime import REFINE_CARRY_KEY, REFINE_CARRY_START_KEY, REFINE_FUSION_KEY, REFINE_METHOD, REFINE_MULTIPLIERS, REFINE_RESUME_KEY, REFINE_TAIL_PADDING_FRAMES, TEMPORAL_STRIDE, refine_fusion_overlap_latents, refine_overlap_frames


def _chunk_mode() -> str:
    """``runtime.REFINE_MEDIA_FLOW_CHUNK_MODE``: "fusion", "conditioning" or "crossfade"."""
    return refine_runtime.REFINE_MEDIA_FLOW_CHUNK_MODE


def _queue_settings(process_settings: dict, model_type: str, source_path: str, start_frame: int, frame_count: int, spatial_upsampling: str, audio_track_no: int | None) -> dict:
    video_path = build_virtual_media_path(source_path, start_frame=start_frame, end_frame=start_frame + frame_count - 1, audio_track_no=audio_track_no)
    api_options = dict(process_settings.get("_api", {})) if isinstance(process_settings.get("_api"), dict) else {}
    api_options.update({"return_media": True, "suppress_source_audio": False, "suppress_metadata_images": True, "upsampler_frame_offset": int(start_frame)})
    settings = dict(process_settings)
    settings.update({
        "mode": "edit_postprocessing",
        "model_type": model_type,
        "image_mode": 0,
        "video_source": video_path,
        "video_length": int(frame_count),
        "keep_frames_video_source": str(int(frame_count)),
        "temporal_upsampling": "",
        "spatial_upsampling": spatial_upsampling,
        "film_grain_intensity": 0,
        "film_grain_saturation": 0.5,
        "postprocess_audio": "",
        "repeat_generation": 1,
        "batch_size": 1,
        "seed": int(settings.get("seed", 0)),
        "_api": api_options,
    })
    return settings


class LTXVideoUpsamplerProcessHandler:
    system_handler = "ltx2_upsampler"
    model_type = "__system_flashvsr"
    model_label = "WanGP System Video Postprocessing"
    target_control_label = "Upsampling"
    default_target_control = "ltx23*2"
    default_chunk_size_seconds = 3.0
    frame_step = 1
    minimum_requested_frames = 1
    crossfade_overlap_outputs = True
    hide_chunk_size = True
    hide_sliding_window_overlap = True
    hide_output_resolution = True
    hide_prompt = False

    @staticmethod
    def config() -> dict[str, Any]:
        from postprocessing import spatial_upsamplers as upsampler_api
        return upsampler_api.config_for_method("ltx23")

    @property
    def overlap_frames(self) -> int:
        return self.config()["window_overlap"]

    def get_overlap_frames(self, chunk_frames: int) -> int:
        return max(0, min(self.overlap_frames, int(chunk_frames) - 1))

    def get_chunk_frames(self, selected_frame_count: int) -> int:
        return min(self.config()["window_size"], int(selected_frame_count))

    @staticmethod
    def normalize_target_control(value: str | None) -> str:
        for method in ("ltx23", "ltx25"):
            scale = upsampler_api.parse_multiplier_suffix(value, method, 2.0)
            if scale == 2.0:
                return upsampler_api.format_multiplier_value(method, scale)
        return LTXVideoUpsamplerProcessHandler.default_target_control

    def target_control_choices_for_process(self, process_settings: dict) -> list[tuple[str, str]]:
        target = self.normalize_target_control(process_settings["target_ratio"])
        return [("x2", target)]

    def target_control_default_for_process(self, process_settings: dict) -> str:
        return self.normalize_target_control(process_settings["target_ratio"])

    def normalize_target_control_for_process(self, value: str | None, process_settings: dict) -> str:
        return self.normalize_target_control(process_settings["target_ratio"])

    def output_resolution_token(self, value: str | None) -> str:
        target = self.normalize_target_control(value)
        return f"{target.partition('*')[0]}-x2"

    def build_queue_settings(self, process_settings: dict, *, source_path: str, start_frame: int, frame_count: int, target_control: str, seed: int, continue_cache: Any, audio_track_no: int | None = None) -> dict:
        return _queue_settings(process_settings, self.model_type, source_path, start_frame, frame_count, self.normalize_target_control_for_process(target_control, process_settings), audio_track_no)

    @staticmethod
    def supports_continue_cache() -> bool:
        return False

    @staticmethod
    def supports_continue_cache_for_target(value: str | None) -> bool:
        return False


class LTXDetailRefinerProcessHandler:
    """Media Flow process: each chunk is one refiner temporal window."""

    system_handler = "ltx2_refiner"
    model_type = "__system_flashvsr"
    model_label = "WanGP System Video Postprocessing"
    target_control_label = "Upsampling"
    target_control_choices = [(f"x{scale:g}", upsampler_api.format_multiplier_value(REFINE_METHOD, scale)) for scale in REFINE_MULTIPLIERS]
    default_target_control = upsampler_api.format_multiplier_value(REFINE_METHOD, 2.0)
    default_chunk_size_seconds = 4.0
    frame_step = 1
    minimum_requested_frames = 1
    hide_chunk_size = True
    hide_sliding_window_overlap = True
    hide_output_resolution = True
    hide_prompt = False

    @staticmethod
    def config() -> dict[str, Any]:
        return upsampler_api.config_for_method(REFINE_METHOD)

    crossfade_overlap_outputs = True

    def overlap_output_split(self, overlap_frames: int) -> int | None:
        """Overlap frames written from the previous chunk, the rest from this one; None crossfades.

        A chunk's first latents follow its own clip start (causal VAE): conditioned ones are VAE round trips of the
        previous frames (softer), fused ones mix two encoding contexts and flicker. Conditioning keeps the previous
        frames unless ``REFINE_MEDIA_FLOW_CONDITIONING_CROSSFADE``; fusion switches mid-overlap.
        """
        mode = _chunk_mode()
        if mode == "fusion":
            return (overlap_frames // TEMPORAL_STRIDE + 1) // 2 * TEMPORAL_STRIDE
        return overlap_frames if mode == "conditioning" and not refine_runtime.REFINE_MEDIA_FLOW_CONDITIONING_CROSSFADE else None

    @property
    def overlap_frames(self) -> int:
        return self.config()["window_overlap"]

    def get_overlap_frames(self, chunk_frames: int) -> int:
        # Must match what the runtime hands to the next chunk (fusion: whole 8-frame latents, so that chunk starts and
        # resume points, which Media Flow places one overlap earlier, stay on the previous chunk's latent grid).
        if _chunk_mode() == "fusion":
            return min(refine_fusion_overlap_latents(self.overlap_frames, chunk_frames) * TEMPORAL_STRIDE, max(0, int(chunk_frames) - 1))
        return refine_overlap_frames(self.overlap_frames, chunk_frames)

    def get_chunk_frames(self, selected_frame_count: int) -> int:
        # The refiner pads each clip by 8 frames (fusion chunks also by one leading frame): every chunk stays one temporal window.
        return min(self.config()["refiner_window_size"] - REFINE_TAIL_PADDING_FRAMES - int(_chunk_mode() == "fusion"), int(selected_frame_count))

    def target_control_choices_for_process(self, process_settings: dict) -> list[tuple[str, str]]:
        return self.target_control_choices

    def normalize_target_control(self, value: str | None) -> str:
        scale = upsampler_api.parse_multiplier_suffix(value, REFINE_METHOD, 2.0)
        return upsampler_api.format_multiplier_value(REFINE_METHOD, scale) if scale in REFINE_MULTIPLIERS else self.default_target_control

    def target_control_default_for_process(self, process_settings: dict) -> str:
        return self.normalize_target_control(process_settings.get("target_ratio"))

    def normalize_target_control_for_process(self, value: str | None, process_settings: dict) -> str:
        return self.normalize_target_control(value or process_settings.get("target_ratio"))

    def output_resolution_token(self, value: str | None) -> str:
        return f"refine-x{upsampler_api.parse_multiplier_suffix(self.normalize_target_control(value), REFINE_METHOD, 2.0):g}"

    def build_queue_settings(self, process_settings: dict, *, source_path: str, start_frame: int, frame_count: int, target_control: str, seed: int, continue_cache: Any, audio_track_no: int | None = None) -> dict:
        settings = _queue_settings(process_settings, self.model_type, source_path, start_frame, frame_count, self.normalize_target_control(target_control), audio_track_no)
        settings["spatial_postprocessing_parameters"] = {**dict(settings.get("spatial_postprocessing_parameters") or {}), "refine_prompt": str(settings.get("prompt") or "")}
        if _chunk_mode() in ("fusion", "conditioning"):
            carry = continue_cache if continue_cache is not None and continue_cache[REFINE_CARRY_START_KEY] == int(start_frame) else None
            if continue_cache is not None and carry is None:
                print(f"[LTX 2.5 Detail Refiner] Continuation data starts at source frame {continue_cache[REFINE_CARRY_START_KEY]}, but the chunk starts at frame {start_frame}: refining it without the previous frames.")
            settings["_api"].update({"return_flashvsr_continue_cache": True, "flashvsr_continue_cache": carry})
        return settings

    # Resume sidecar: fusion stores the per-step predictions handed to the next chunk, conditioning the latents of
    # the last written overlap frames, so a resumed run continues from lossless refiner data, not the encoded output.
    @staticmethod
    def supports_continue_cache() -> bool:
        return _chunk_mode() in ("fusion", "conditioning")

    @staticmethod
    def supports_continue_cache_for_target(value: str | None) -> bool:
        return _chunk_mode() in ("fusion", "conditioning")

    @staticmethod
    def cache_sidecar_path(output_filename: str) -> str:
        output_path = Path(output_filename).resolve()
        return str(output_path.with_suffix(output_path.suffix + ".ltx25_refine_cache.safetensors"))

    def can_resume_without_output_metadata(self, output_filename: str) -> bool:
        return Path(self.cache_sidecar_path(output_filename)).is_file()

    def move_continue_cache(self, source_output_filename: str, target_output_filename: str) -> bool:
        source_path = Path(self.cache_sidecar_path(source_output_filename))
        if not source_path.is_file():
            return False
        target_path = Path(self.cache_sidecar_path(target_output_filename))
        target_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.replace(target_path)
        return True

    def delete_continue_cache(self, output_filename: str) -> None:
        Path(self.cache_sidecar_path(output_filename)).unlink(missing_ok=True)

    def save_continue_cache(self, cache: Any, output_filename: str, metadata: dict | None = None) -> str:
        resume = cache[REFINE_RESUME_KEY]
        mode = "fusion" if REFINE_FUSION_KEY in resume else "conditioning"
        sidecar_path = self.cache_sidecar_path(output_filename)
        Path(sidecar_path).parent.mkdir(parents=True, exist_ok=True)
        save_file({"latents": resume[REFINE_FUSION_KEY if mode == "fusion" else REFINE_CARRY_KEY].contiguous()}, sidecar_path, metadata={"version": "2", "handler": self.system_handler, "mode": mode, "start_frame": str(resume[REFINE_CARRY_START_KEY]), "metadata": json.dumps(metadata or {}, ensure_ascii=True, sort_keys=True)})
        return sidecar_path

    def load_continue_cache(self, output_filename: str) -> Any:
        sidecar_path = self.cache_sidecar_path(output_filename)
        if not Path(sidecar_path).is_file():
            print(f"[LTX 2.5 Detail Refiner] Continuation data is missing ({sidecar_path}): the first resumed chunk is refined without the previous frames.")
            return None
        with safe_open(sidecar_path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            latents = handle.get_tensor("latents").clone() if "latents" in handle.keys() else None
        mode = metadata.get("mode")
        if metadata.get("handler") != self.system_handler or mode not in ("fusion", "conditioning") or latents is None or latents.ndim != 5 or not str(metadata.get("start_frame", "")).isdigit():
            raise gr.Error(f"LTX 2.5 Detail Refiner continuation data is invalid: {sidecar_path}")
        if mode != _chunk_mode():
            print(f"[LTX 2.5 Detail Refiner] Continuation data was made in {mode} mode: the first resumed chunk is refined without it.")
            return None
        # Until a new chunk is written, the loaded data also remains the resume point.
        resume = {REFINE_FUSION_KEY if mode == "fusion" else REFINE_CARRY_KEY: latents, REFINE_CARRY_START_KEY: int(metadata["start_frame"])}
        return dict(resume, **{REFINE_RESUME_KEY: resume})


HANDLER = LTXVideoUpsamplerProcessHandler()
REFINER_HANDLER = LTXDetailRefinerProcessHandler()
