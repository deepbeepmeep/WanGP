# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import gradio as gr

from shared.utils.audio_video import extract_audio_tracks
from shared.utils.utils import get_video_info_details
from shared.utils.video_decode import resolve_media_binary
from shared.utils.virtual_media import build_virtual_media_path

from . import common
from . import frame_planning as frames
from . import media_io as media
from . import output_paths
from . import process_metadata
from . import video_buffers as video


class ProcessInfoExit(Exception):
    def __init__(self, message: str, *, output_path: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.output_path = output_path


@dataclass(frozen=True)
class PreparedRun:
    verbose_level: int
    start_frame: int
    end_frame_exclusive: int
    fps_float: float
    total_source_frames: int
    processing_fps: float
    selected_audio_track: int | None
    frame_plan_rules: frames.FramePlanRules
    budget_resolution: str
    chunk_frames: int
    overlap_frames: int
    full_plans: list[frames.ChunkPlan]
    requested_unique_frames: int
    requested_source_segment: str
    output_path: str
    resume_existing_output: bool
    ffmpeg_path: str
    ffprobe_path: str
    output_container: str
    merged_continuation_signatures: list[dict]


def get_mmgp_verbose_level() -> int:
    try:
        from mmgp import offload
        return int(getattr(offload, "default_verboseLevel", 0) or 0)
    except Exception:
        return 0


def prepare_run(
    *,
    plugin,
    get_model_def,
    process_name: str,
    process_display_name: str,
    source_path: str,
    output_path: str,
    output_resolution: str,
    active_target_ratio: str,
    continue_enabled: bool,
    source_audio_track: str,
    chunk_size_seconds: float,
    sliding_window_overlap: int,
    sliding_window_size: int | None,
    start_seconds: float,
    end_seconds: float | None,
    model_type: str,
    process_is_hdr: bool,
    uses_builtin_outpaint_ui: bool,
    system_handler=None,
    system_target_control: str = "",
    expected_spatial_upsampling: str = "",
) -> PreparedRun:
    verbose_level = get_mmgp_verbose_level()
    try:
        metadata = get_video_info_details(source_path)
    except Exception as exc:
        raise gr.Error(f"Unable to read source video metadata: {source_path}") from exc
    start_frame, end_frame_exclusive, fps_float, total_source_frames = video.compute_selected_frame_range(metadata, start_seconds, end_seconds)
    processing_fps = video.get_processing_fps(fps_float)
    try:
        audio_track_count = int(extract_audio_tracks(source_path, query_only=True))
    except Exception as exc:
        raise gr.Error(f"Unable to inspect source audio tracks in: {source_path}") from exc
    selected_audio_track = None
    if len(source_audio_track) > 0:
        if not source_audio_track.isdigit() or int(source_audio_track) <= 0:
            raise gr.Error("Source Audio must be Auto or a whole track number.")
        if audio_track_count <= 0:
            raise gr.Error("Source video contains no audio track. Leave Source Audio empty.")
        selected_audio_track = int(source_audio_track)
    elif audio_track_count > 0:
        selected_audio_track = 1
    if selected_audio_track is not None and (selected_audio_track <= 0 or selected_audio_track > audio_track_count):
            raise gr.Error(f"Source Audio must be between 1 and {audio_track_count}.")
    if system_handler is None:
        frame_plan_rules = frames.get_frame_plan_rules(model_type, get_model_def)
        budget_resolution = output_paths.choose_resolution(output_resolution)
        output_resolution_token = output_resolution
    else:
        frame_plan_rules = frames.FramePlanRules(frame_step=int(getattr(system_handler, "frame_step", 1)), minimum_requested_frames=int(getattr(system_handler, "minimum_requested_frames", 1)), frame_offset=int(getattr(system_handler, "frame_offset", 1)))
        budget_resolution = ""
        output_resolution_token = system_handler.output_resolution_token(system_target_control) if hasattr(system_handler, "output_resolution_token") else output_resolution
    try:
        chunk_frame_step = int(getattr(system_handler, "chunk_frame_step", frame_plan_rules.frame_step)) if system_handler is not None else frame_plan_rules.frame_step
        get_chunk_frames = getattr(system_handler, "get_chunk_frames", None) if system_handler is not None else None
        chunk_frames = int(get_chunk_frames(end_frame_exclusive - start_frame)) if callable(get_chunk_frames) else frames.normalize_chunk_frames(chunk_size_seconds, processing_fps, frame_step=chunk_frame_step, minimum_requested_frames=frame_plan_rules.minimum_requested_frames, frame_offset=frame_plan_rules.frame_offset)
        # WanGP splits a longer chunk into its own sliding windows: no gain, a coarser resume and a step counter that restarts in each window.
        if system_handler is None and sliding_window_size and frames.require_model_def(model_type, get_model_def).get("sliding_window", False):
            window_frames = frames.align_requested_frames(int(sliding_window_size), frame_step=frame_plan_rules.frame_step, frame_offset=frame_plan_rules.frame_offset, round_up=False)
            if chunk_frames > window_frames >= frame_plan_rules.minimum_requested_frames:
                common.plugin_info(f"Chunk Size reduced to {window_frames / processing_fps:.2f} seconds ({window_frames} frames), the longest piece this process generates in one pass.")
                chunk_frames = window_frames
        if system_handler is not None:
            get_target_overlap_frames = getattr(system_handler, "get_overlap_frames_for_target", None)
            get_overlap_frames = getattr(system_handler, "get_overlap_frames", None)
            if callable(get_target_overlap_frames):
                overlap_frames = int(get_target_overlap_frames(chunk_frames, system_target_control))
            else:
                overlap_frames = int(get_overlap_frames(chunk_frames)) if callable(get_overlap_frames) else int(getattr(system_handler, "overlap_frames", 0))
        else:
            overlap_frames = frames.normalize_overlap_frames(sliding_window_overlap, frame_step=frame_plan_rules.frame_step)
    except frames.ChunkSizeBelowMinimumError as exc:
        raise ProcessInfoExit(str(exc)) from exc
    except frames.FramePlanningError as exc:
        raise gr.Error(str(exc)) from exc
    if overlap_frames >= chunk_frames:
        raise gr.Error(f"Sliding Window Overlap must stay below the computed chunk size ({chunk_frames} frame(s)).")
    selected_unique_frames = end_frame_exclusive - start_frame
    try:
        full_plans = frames.build_chunk_plan(
            start_frame,
            end_frame_exclusive,
            total_source_frames,
            chunk_frames,
            frame_step=frame_plan_rules.frame_step,
            minimum_requested_frames=frame_plan_rules.minimum_requested_frames,
            frame_offset=frame_plan_rules.frame_offset,
            overlap_frames=overlap_frames,
        )
    except frames.FramePlanningError as exc:
        raise gr.Error(str(exc)) from exc
    requested_unique_frames = frames.count_planned_unique_frames(full_plans)
    requested_source_segment = build_virtual_media_path(source_path, start_frame=start_frame, end_frame=start_frame + requested_unique_frames - 1, audio_track_no=selected_audio_track)
    default_output_container = media.normalize_container_name(plugin.server_config.get("video_container", "mp4"))
    requested_output_path = str(output_paths.build_requested_output_path(source_path, output_path, process_display_name, active_target_ratio, output_resolution_token, start_seconds, end_seconds, has_outpaint=uses_builtin_outpaint_ui, default_container=default_output_container))
    if continue_enabled:
        existing_identity = process_metadata.read_output_identity(requested_output_path)
        identity_mismatch_message = process_metadata.get_output_identity_mismatch_message(requested_output_path, existing_identity, process_name=process_display_name, source_path=source_path, source_segment=requested_source_segment, spatial_upsampling=expected_spatial_upsampling)
        if identity_mismatch_message is not None:
            system_supports_continue_cache = system_handler is not None and (not callable(getattr(system_handler, "supports_continue_cache_for_target", None)) or system_handler.supports_continue_cache_for_target(system_target_control))
            can_resume_from_system_state = existing_identity is None and system_supports_continue_cache and callable(getattr(system_handler, "can_resume_without_output_metadata", None)) and system_handler.can_resume_without_output_metadata(requested_output_path)
            if not can_resume_from_system_state:
                raise ProcessInfoExit(identity_mismatch_message, output_path=requested_output_path)
            common.plugin_info(f"Existing output has no readable WanGP metadata, but the system process can continue from its cache or decoded output tail: {requested_output_path}")
    resolved_output_path, resume_existing_output = output_paths.resolve_output_path(source_path, output_path, process_display_name, active_target_ratio, output_resolution_token, start_seconds, end_seconds, continue_enabled, has_outpaint=uses_builtin_outpaint_ui, default_container=default_output_container, notify=common.plugin_info)
    try:
        Path(resolved_output_path).parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise gr.Error(f"Unable to create the output folder for: {resolved_output_path}") from exc
    dropped_tail_frames = selected_unique_frames - requested_unique_frames
    if dropped_tail_frames > 0:
        common.plugin_info(f"Dropping the last {dropped_tail_frames} source frame(s) so the selected range fits the current model chunk shape.")
    ffmpeg_path = resolve_media_binary("ffmpeg")
    if ffmpeg_path is None:
        raise gr.Error("ffmpeg binary not found.")
    ffprobe_path = resolve_media_binary("ffprobe")
    if ffprobe_path is None:
        raise gr.Error("ffprobe binary not found.")
    output_container = media.normalize_container_name(Path(resolved_output_path).suffix.lstrip(".") or plugin.server_config.get("video_container", "mp4"))
    output_video_codec = "libx265_8" if process_is_hdr else plugin.server_config.get("video_output_codec", "libx264_8")
    media.validate_output_codec_container(output_video_codec, output_container, output_path=resolved_output_path)
    if selected_audio_track is not None:
        media.validate_audio_copy_container(ffprobe_path, source_path, output_container, selected_audio_track)
    return PreparedRun(
        verbose_level=verbose_level,
        start_frame=start_frame,
        end_frame_exclusive=end_frame_exclusive,
        fps_float=fps_float,
        total_source_frames=total_source_frames,
        processing_fps=processing_fps,
        selected_audio_track=selected_audio_track,
        frame_plan_rules=frame_plan_rules,
        budget_resolution=budget_resolution,
        chunk_frames=chunk_frames,
        overlap_frames=overlap_frames,
        full_plans=full_plans,
        requested_unique_frames=requested_unique_frames,
        requested_source_segment=requested_source_segment,
        output_path=resolved_output_path,
        resume_existing_output=resume_existing_output,
        ffmpeg_path=ffmpeg_path,
        ffprobe_path=ffprobe_path,
        output_container=output_container,
        merged_continuation_signatures=process_metadata.read_merged_continuation_signatures(resolved_output_path),
    )
