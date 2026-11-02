# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path

from shared.utils.video_metadata import read_metadata_from_video, save_video_metadata
from shared.utils.virtual_media import build_virtual_media_path


PLUGIN_NAME = "MediaFlow"
PROCESS_FULL_VIDEO_METADATA_KEY = "fill_process_video"


def format_time_hms(seconds: float | None) -> str:
    total_seconds = max(0, int(round(float(seconds or 0.0))))
    minutes, seconds_only = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_only:02d}"


def make_continuation_signature(file_path: str) -> dict | None:
    if not isinstance(file_path, str) or not os.path.isfile(file_path):
        return None
    try:
        stats = os.stat(file_path)
    except OSError:
        return None
    return {"path": str(Path(file_path).resolve()), "size": int(stats.st_size), "mtime_ns": int(getattr(stats, "st_mtime_ns", int(stats.st_mtime * 1_000_000_000)))}


def continuation_signature_key(signature: dict) -> tuple[str, int, int] | None:
    if not isinstance(signature, dict):
        return None
    path = str(signature.get("path") or "").strip()
    if len(path) == 0:
        return None
    try:
        return str(Path(path).resolve()), max(0, int(signature.get("size"))), max(0, int(signature.get("mtime_ns")))
    except (TypeError, ValueError):
        return None


def normalize_merged_continuation_signatures(signatures) -> list[dict]:
    normalized: list[dict] = []
    seen: set[tuple[str, int, int]] = set()
    for signature in list(signatures or []):
        key = continuation_signature_key(signature)
        if key is None or key in seen:
            continue
        seen.add(key)
        path, size, mtime_ns = key
        normalized.append({"path": path, "size": size, "mtime_ns": mtime_ns})
    return normalized


def append_merged_continuation_signature(signatures: list[dict], signature: dict | None) -> list[dict]:
    return normalize_merged_continuation_signatures([*list(signatures or []), *( [] if signature is None else [signature] )])


def read_merged_continuation_signatures(output_path: str) -> list[dict]:
    if not isinstance(output_path, str) or not os.path.isfile(output_path):
        return []
    metadata = read_metadata_from_video(output_path)
    if not isinstance(metadata, dict):
        return []
    process_metadata = metadata.get(PROCESS_FULL_VIDEO_METADATA_KEY)
    return [] if not isinstance(process_metadata, dict) else normalize_merged_continuation_signatures(process_metadata.get("merged_continuations"))


def store_process_progress(output_path: str, *, written_unique_frames: int, merged_signatures: list[dict], verbose_level: int = 0) -> bool:
    if not isinstance(output_path, str) or not os.path.isfile(output_path):
        return False
    metadata = read_metadata_from_video(output_path)
    if not isinstance(metadata, dict) or len(metadata) == 0:
        return False
    process_metadata = metadata.get(PROCESS_FULL_VIDEO_METADATA_KEY)
    process_metadata = {} if not isinstance(process_metadata, dict) else process_metadata.copy()
    process_metadata["written_unique_frames"] = int(written_unique_frames)
    process_metadata["merged_continuations"] = normalize_merged_continuation_signatures(merged_signatures)
    metadata[PROCESS_FULL_VIDEO_METADATA_KEY] = process_metadata
    metadata["video_length"] = int(written_unique_frames)
    metadata["frame_count"] = int(written_unique_frames)
    if save_video_metadata(output_path, metadata, allow_inplace_update=True, verbose_level=verbose_level):
        return True
    print(f"[MediaFlow] Warning: failed to store process progress in {output_path}")
    return False


def read_recorded_written_unique_frames(output_path: str) -> int:
    if not isinstance(output_path, str) or not os.path.isfile(output_path):
        return 0
    metadata = read_metadata_from_video(output_path)
    if not isinstance(metadata, dict):
        return 0
    process_metadata = metadata.get(PROCESS_FULL_VIDEO_METADATA_KEY)
    if not isinstance(process_metadata, dict):
        return 0
    try:
        return max(0, int(process_metadata.get("written_unique_frames") or 0))
    except (TypeError, ValueError):
        return 0


def normalize_identity_path(path_value: str) -> str:
    text = str(path_value or "").strip()
    if len(text) == 0:
        return ""
    try:
        return str(Path(text).resolve()).casefold()
    except (OSError, RuntimeError, ValueError):
        return text.casefold()


def read_output_identity(output_path: str) -> tuple[str, str, str, str] | None:
    if not isinstance(output_path, str) or not os.path.isfile(output_path):
        return None
    metadata = read_metadata_from_video(output_path)
    if not isinstance(metadata, dict) or len(metadata) == 0:
        return None
    process_metadata = metadata.get(PROCESS_FULL_VIDEO_METADATA_KEY)
    process_metadata = process_metadata if isinstance(process_metadata, dict) else {}
    process_name = str(process_metadata.get("process") or "").strip()
    if len(process_name) == 0:
        segments = metadata.get("segments")
        first_segment = next((segment for segment in list(segments or []) if isinstance(segment, dict)), None)
        if first_segment is not None:
            process_name = str(first_segment.get("process") or "").strip()
    source_video = str(process_metadata.get("source_video") or metadata.get("source_video") or "").strip()
    source_segment = str(process_metadata.get("source_segment") or metadata.get("source_segment") or "").strip()
    return process_name, source_video, source_segment, str(process_metadata.get("spatial_upsampling") or metadata.get("spatial_upsampling") or "")


def get_output_identity_mismatch_message(output_path: str, identity, *, process_name: str, source_path: str, source_segment: str, spatial_upsampling: str = "") -> str | None:
    if not isinstance(output_path, str) or not os.path.isfile(output_path):
        return None
    if identity is None:
        return f"Output file already exists at {output_path}, but it does not contain readable WanGP metadata. Processing was stopped."
    existing_process, existing_source_video, existing_source_segment, existing_upsampling = identity
    if existing_upsampling and spatial_upsampling:
        from postprocessing.spatial_upsamplers import normalize_upsampling_value
        if normalize_upsampling_value(existing_upsampling) != spatial_upsampling:
            return f"Cannot continue {output_path}: it used {existing_upsampling} and this run requests {spatial_upsampling}. Choose another output or turn off Continue."
    mismatches: list[str] = []
    if str(existing_process or "").strip() != str(process_name or "").strip():
        mismatches.append("process")
    if normalize_identity_path(existing_source_video) != normalize_identity_path(source_path):
        mismatches.append("source_video")
    if str(existing_source_segment or "").strip() != str(source_segment or "").strip():
        mismatches.append("source_segment")
    if len(mismatches) == 0:
        return None
    mismatch_text = ", ".join(mismatches)
    return f"Output file already exists at {output_path}, but its metadata does not match the current {mismatch_text}. Processing was stopped."


def log_existing_output_metadata(output_path: str, verbose_level: int) -> None:
    if int(verbose_level or 0) < 2 or not os.path.isfile(output_path):
        return
    metadata = read_metadata_from_video(output_path)
    if not isinstance(metadata, dict) or len(metadata) == 0:
        print(f"[MediaFlow] Existing output metadata not found in {output_path}")
        return
    creation_date = str(metadata.get("creation_date") or "unknown")
    generation_time = metadata.get("generation_time")
    generation_time_text = "unknown" if generation_time in (None, "") else str(generation_time)
    print(f"[MediaFlow] Existing output metadata found: creation_date={creation_date}, generation_time={generation_time_text}")


def store_output_metadata(output_path: str, last_segment_path: str | None, *, source_path: str, process_name: str, source_start_seconds: float, start_frame: int, fps_float: float, selected_audio_track: int | None, total_generation_time: float, actual_frame_count: int, source_frame_count: int | None = None, process_metadata: dict | None = None, verbose_level: int = 0) -> bool:
    if not os.path.isfile(output_path):
        return False
    metadata = {}
    if last_segment_path and os.path.isfile(last_segment_path):
        loaded_metadata = read_metadata_from_video(last_segment_path)
        if isinstance(loaded_metadata, dict) and len(loaded_metadata) > 0:
            metadata = loaded_metadata
        elif Path(last_segment_path).resolve() != Path(output_path).resolve():
            print(f"[MediaFlow] Warning: failed to read WanGP metadata from {last_segment_path}")
    elif last_segment_path:
        print(f"[MediaFlow] Warning: no segment metadata source was available for {output_path}")
    final_metadata = metadata.copy()
    source_name = os.path.basename(source_path)
    source_frame_count = int(actual_frame_count) if source_frame_count is None else int(source_frame_count)
    end_frame = max(int(start_frame), int(start_frame) + max(0, source_frame_count) - 1)
    start_seconds = max(0.0, float(source_start_seconds or 0.0))
    end_seconds = start_seconds + max(0, source_frame_count) / float(fps_float)
    final_metadata["video_guide"] = build_virtual_media_path(source_path, start_frame=start_frame, end_frame=end_frame, audio_track_no=selected_audio_track)
    final_metadata["video_length"] = int(actual_frame_count)
    final_metadata["frame_count"] = int(actual_frame_count)
    final_metadata["generation_time"] = max(0.0, float(total_generation_time))
    operation_comment = f'{PLUGIN_NAME}: {process_name} on "{source_name}" Start { format_time_hms(start_seconds) } End { format_time_hms(end_seconds) }'
    existing_comments = str(final_metadata.get("comments") or "").strip()
    final_metadata["comments"] = operation_comment if len(existing_comments) == 0 else f"{existing_comments}\n{operation_comment}"
    final_metadata["segments"] = [{
        "plugin": PLUGIN_NAME,
        "process": process_name,
        "source_file": source_name,
        "start_frame": int(start_frame),
        "end_frame": end_frame,
        "start_seconds": start_seconds,
        "end_seconds": end_seconds,
    }]
    final_metadata[PROCESS_FULL_VIDEO_METADATA_KEY] = process_metadata.copy() if isinstance(process_metadata, dict) else {}
    for key in ("plugin", "process", "source_video", "source_segment", "video_source"):
        final_metadata.pop(key, None)
    final_metadata["creation_date"] = datetime.now().isoformat(timespec="seconds")
    final_metadata["creation_timestamp"] = int(time.time())
    if save_video_metadata(output_path, final_metadata, allow_inplace_update=True, verbose_level=verbose_level):
        return True
        print(f"[MediaFlow] Warning: failed to write metadata to {output_path}")
    return False


def read_metadata_generation_time(video_path: str | None) -> float:
    if not video_path or not os.path.isfile(video_path):
        return 0.0
    metadata = read_metadata_from_video(video_path)
    if not isinstance(metadata, dict):
        return 0.0
    try:
        return max(0.0, float(metadata.get("generation_time") or 0.0))
    except (TypeError, ValueError):
        return 0.0
