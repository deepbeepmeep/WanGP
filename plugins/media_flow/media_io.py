# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Callable
from pathlib import Path

import gradio as gr

from shared.utils.audio_video import get_hdr_video_encode_args, get_video_encode_args
from shared.utils.video_codecs import normalize_video_container, validate_video_output_settings
from shared.utils.utils import get_video_info_details
from shared.utils.video_decode import resolve_media_binary
from shared.utils.video_metadata import DEFAULT_RESERVED_VIDEO_METADATA_BYTES, write_reserved_video_ffmetadata

from . import constants
from shared.utils.subprocess_stdin import child_kwargs

DEFAULT_FFMPEG_TIMEOUT_SECONDS = float(os.getenv("WAN2GP_FFMPEG_TIMEOUT", "300.0"))

class ContinuationMergeOutputLockedError(PermissionError):
    def __init__(self, output_path: str) -> None:
        self.output_path = str(output_path or "")
        super().__init__(f"Unable to replace locked output file: {self.output_path}")


def probe_resume_frame_count(ffprobe_path: str, output_path: str, fps_float: float) -> tuple[int, str]:
    probe = subprocess.run([ffprobe_path, "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries", "stream=nb_read_packets", "-of", "json", output_path], capture_output=True, text=True, encoding="utf-8", errors="ignore", check=False, **child_kwargs())
    if probe.returncode == 0:
        try:
            frame_count = int((((json.loads(probe.stdout).get("streams") or [{}])[0]).get("nb_read_packets")) or 0)
        except (TypeError, ValueError, json.JSONDecodeError, IndexError):
            frame_count = 0
        if frame_count > 0:
            return frame_count, ""
    probe = subprocess.run([ffprobe_path, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries", "stream=nb_read_frames", "-of", "json", output_path], capture_output=True, text=True, encoding="utf-8", errors="ignore", check=False, **child_kwargs())
    if probe.returncode == 0:
        try:
            frame_count = int((((json.loads(probe.stdout).get("streams") or [{}])[0]).get("nb_read_frames")) or 0)
        except (TypeError, ValueError, json.JSONDecodeError, IndexError):
            frame_count = 0
        if frame_count > 0:
            return frame_count, ""
    metadata = get_video_info_details(output_path)
    frame_count = int(metadata.get("frame_count") or 0)
    if frame_count > 0:
        return frame_count, ""
    duration = float(metadata.get("duration") or 0.0)
    if duration > 0 and fps_float > 0:
        return int(round(duration * fps_float)), ""
    stderr = (probe.stderr or "").strip()
    return 0, stderr or "existing output contains no readable frame count or duration metadata"


def normalize_container_name(video_container: str | None) -> str:
    return normalize_video_container(video_container)


SUPPORTED_OUTPUT_CONTAINERS = constants.SUPPORTED_OUTPUT_CONTAINERS
LIVE_AUDIO_LOOP_PADDING_SECONDS = 5.0


def is_supported_output_container(video_container: str | None) -> bool:
    return normalize_container_name(video_container) in SUPPORTED_OUTPUT_CONTAINERS


def is_quicktime_output_container(video_container: str | None) -> bool:
    return normalize_container_name(video_container) in {"mp4", "mov"}


def _get_live_mux_output_args(video_container: str | None) -> list[str]:
    video_container = normalize_container_name(video_container)
    if video_container == "mkv":
        return ["-fflags", "+flush_packets", "-flush_packets", "1", "-f", "matroska", "-live", "1"]
    if is_quicktime_output_container(video_container):
        return ["-movflags", "+frag_keyframe+empty_moov+default_base_moof"]
    return []


def _probe_media_duration(ffprobe_path: str, media_path: str) -> float:
    result = subprocess.run([ffprobe_path, "-v", "error", "-show_entries", "format=duration", "-of", "json", media_path], capture_output=True, text=True, encoding="utf-8", errors="ignore", check=False, **child_kwargs())
    if result.returncode != 0:
        return 0.0
    try:
        return max(0.0, float((((json.loads(result.stdout) or {}).get("format") or {}).get("duration")) or 0.0))
    except (TypeError, ValueError, json.JSONDecodeError):
        return 0.0


def _probe_audio_end_time(ffprobe_path: str, media_path: str, audio_index: int) -> float:
    stream_selector = f"a:{max(0, int(audio_index))}"
    result = subprocess.run([ffprobe_path, "-v", "error", "-select_streams", stream_selector, "-show_entries", "stream=start_time,duration", "-of", "json", media_path], capture_output=True, text=True, encoding="utf-8", errors="ignore", check=False, **child_kwargs())
    if result.returncode == 0:
        try:
            stream = (((json.loads(result.stdout) or {}).get("streams") or [{}])[0])
            start_time = 0.0 if stream.get("start_time") in (None, "", "N/A") else max(0.0, float(stream.get("start_time")))
            duration_time = 0.0 if stream.get("duration") in (None, "", "N/A") else max(0.0, float(stream.get("duration")))
            if duration_time > 0.0:
                return start_time + duration_time
        except (TypeError, ValueError, json.JSONDecodeError, IndexError):
            pass
    approximate_end = _probe_media_duration(ffprobe_path, media_path)
    if approximate_end <= 0.0:
        return 0.0
    for window_seconds in (2.0, 8.0, 32.0, 128.0):
        probe_start = max(0.0, approximate_end - window_seconds)
        probe_span = max(0.25, approximate_end - probe_start + 0.25)
        result = subprocess.run(
            [ffprobe_path, "-v", "error", "-select_streams", stream_selector, "-read_intervals", f"{probe_start:.6f}%+{probe_span:.6f}", "-show_packets", "-show_entries", "packet=pts_time,duration_time", "-of", "csv=p=0", media_path],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
         **child_kwargs())
        if result.returncode != 0:
            continue
        last_end = 0.0
        for raw_line in result.stdout.splitlines():
            fields = [field.strip() for field in str(raw_line or "").strip().split(",")]
            if len(fields) <= 0 or len(fields[0]) == 0:
                continue
            try:
                pts_time = float(fields[0])
            except (TypeError, ValueError):
                continue
            try:
                duration_time = float(fields[1]) if len(fields) > 1 and len(fields[1]) > 0 else 0.0
            except (TypeError, ValueError):
                duration_time = 0.0
            last_end = max(last_end, pts_time, pts_time + max(0.0, duration_time))
        if last_end > 0.0:
            return last_end
    return approximate_end


def _probe_selected_audio_end_time(ffprobe_path: str, media_path: str, audio_track_no: int | None) -> float:
    _, audio_stream_count = _probe_media_stream_layout(ffprobe_path, media_path)
    if audio_stream_count <= 0:
        return 0.0
    if audio_track_no is None:
        audio_indices = range(audio_stream_count)
    else:
        audio_indices = [max(0, min(audio_stream_count - 1, int(audio_track_no) - 1))]
    return max((_probe_audio_end_time(ffprobe_path, media_path, audio_index) for audio_index in audio_indices), default=0.0)


def probe_selected_audio_overhang(ffprobe_path: str, media_path: str, audio_track_no: int | None, visible_duration_seconds: float) -> float:
    visible_duration_seconds = max(0.0, float(visible_duration_seconds or 0.0))
    _, audio_stream_count = _probe_media_stream_layout(ffprobe_path, media_path)
    if audio_stream_count <= 0:
        return 0.0
    if audio_track_no is None:
        audio_indices = range(audio_stream_count)
    else:
        audio_indices = [max(0, min(audio_stream_count - 1, int(audio_track_no) - 1))]
    video_end_seconds = _probe_primary_video_start_time(ffprobe_path, media_path) + visible_duration_seconds
    seam_start = max(0.0, video_end_seconds - 1.0)
    for probe_span in (2.0, 8.0, 32.0, 128.0):
        last_end = 0.0
        for audio_index in audio_indices:
            result = subprocess.run(
                [
                    ffprobe_path,
                    "-v",
                    "error",
                    "-select_streams",
                    f"a:{int(audio_index)}",
                    "-read_intervals",
                    f"{seam_start:.6f}%+{probe_span:.6f}",
                    "-show_packets",
                    "-show_entries",
                    "packet=pts_time,duration_time",
                    "-of",
                    "csv=p=0",
                    media_path,
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                check=False,
             **child_kwargs())
            if result.returncode != 0:
                continue
            for raw_line in result.stdout.splitlines():
                fields = [field.strip() for field in str(raw_line or "").strip().split(",")]
                if len(fields) <= 0 or len(fields[0]) == 0:
                    continue
                try:
                    pts_time = float(fields[0])
                except (TypeError, ValueError):
                    continue
                try:
                    duration_time = float(fields[1]) if len(fields) > 1 and len(fields[1]) > 0 else 0.0
                except (TypeError, ValueError):
                    duration_time = 0.0
                last_end = max(last_end, pts_time, pts_time + max(0.0, duration_time))
        if last_end > video_end_seconds:
            return max(0.0, last_end - video_end_seconds)
    return max(0.0, _probe_selected_audio_end_time(ffprobe_path, media_path, audio_track_no) - video_end_seconds)


def _probe_audio_stream_codecs(ffprobe_path: str, media_path: str) -> list[str]:
    result = subprocess.run([ffprobe_path, "-v", "error", "-select_streams", "a", "-show_entries", "stream=index,codec_name", "-of", "json", media_path], capture_output=True, text=True, encoding="utf-8", errors="ignore", check=False, **child_kwargs())
    if result.returncode != 0:
        raise gr.Error((result.stderr or result.stdout or f"ffprobe failed for {media_path}").strip())
    try:
        streams = list((json.loads(result.stdout) or {}).get("streams") or [])
    except json.JSONDecodeError as exc:
        raise gr.Error(f"Unable to read audio codecs for {media_path}") from exc
    return [str(stream.get("codec_name") or "").strip().lower() for stream in streams]


def validate_audio_copy_container(ffprobe_path: str, source_path: str, video_container: str, audio_track_no: int | None) -> None:
    video_container = normalize_container_name(video_container)
    if video_container not in {"mp4", "mov"}:
        return
    supported_codecs = {"aac", "ac3", "alac", "eac3", "flac", "mp3", "opus"} if video_container == "mp4" else {"aac", "ac3", "alac", "eac3", "mp3", "pcm_s16le", "pcm_s24le", "pcm_s32le"}
    audio_codecs = _probe_audio_stream_codecs(ffprobe_path, source_path)
    if audio_track_no is None:
        selected_codecs = [codec for codec in audio_codecs if len(codec) > 0]
    else:
        selected_index = max(0, int(audio_track_no) - 1)
        selected_codecs = [audio_codecs[selected_index]] if selected_index < len(audio_codecs) and len(audio_codecs[selected_index]) > 0 else []
    incompatible_codecs = [codec for codec in selected_codecs if codec not in supported_codecs]
    if len(incompatible_codecs) > 0:
        track_label = f"audio track {int(audio_track_no)}" if audio_track_no is not None else "the selected audio tracks"
        codecs_text = ", ".join(sorted(set(incompatible_codecs)))
        raise gr.Error(f"{video_container.upper()} output cannot packet-copy {track_label} with codec(s): {codecs_text}. Use an .mkv output path or choose a compatible track.")


def validate_output_codec_container(video_codec: str, video_container: str, audio_codec: str | None = None, width: int | None = None, height: int | None = None, output_path: str | None = None) -> None:
    error = validate_video_output_settings(video_codec, video_container, audio_codec=audio_codec, width=width, height=height, allowed_containers=SUPPORTED_OUTPUT_CONTAINERS)
    if error is None:
        return
    suffix = Path(output_path).suffix if output_path else f".{normalize_container_name(video_container)}"
    raise gr.Error(f"Output file extension '{suffix}' is incompatible with the current video codec/container settings. {error} Choose a compatible output extension or update the Configuration plugin settings.")


def start_video_mux_process(ffmpeg_path: str, output_path: str, width: int, height: int, fps_float: float, video_codec: str, video_container: str, reserved_metadata_path: str | None = None) -> subprocess.Popen:
    validate_output_codec_container(video_codec, video_container, width=width, height=height, output_path=output_path)
    command = [
        ffmpeg_path,
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        f"{float(fps_float):.12g}",
        "-i",
        "pipe:0",
    ]
    metadata_input_index = None
    if reserved_metadata_path and os.path.isfile(reserved_metadata_path):
        command += ["-f", "ffmetadata", "-i", reserved_metadata_path]
        metadata_input_index = 1
    if metadata_input_index is not None:
        command += ["-map_metadata", str(metadata_input_index)]
    command += ["-map", "0:v:0"]
    command += get_video_encode_args(video_codec, video_container)
    command += _get_live_mux_output_args(video_container)
    command += [output_path]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, bufsize=0)


def start_av_mux_process(ffmpeg_path: str, output_path: str, width: int, height: int, fps_float: float, video_codec: str, video_container: str, source_path: str, start_seconds: float, audio_track_no: int | None, reserved_metadata_path: str | None = None, source_audio_duration_seconds: float | None = None) -> subprocess.Popen:
    validate_output_codec_container(video_codec, video_container, width=width, height=height, output_path=output_path)
    command = [
        ffmpeg_path,
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        f"{float(fps_float):.12g}",
        "-i",
        "pipe:0",
        "-stream_loop",
        "-1",
        "-ss",
        f"{max(0.0, float(start_seconds)):.12g}",
    ]
    if source_audio_duration_seconds is not None and source_audio_duration_seconds > 0.0:
        command += ["-t", f"{max(0.0, float(source_audio_duration_seconds) + LIVE_AUDIO_LOOP_PADDING_SECONDS):.12g}"]
    command += ["-i", source_path]
    metadata_input_index = None
    if reserved_metadata_path and os.path.isfile(reserved_metadata_path):
        command += ["-f", "ffmetadata", "-i", reserved_metadata_path]
        metadata_input_index = 2
    else:
        metadata_input_index = 1
    command += ["-map_metadata", str(metadata_input_index), "-map", "0:v:0"]
    if audio_track_no is None:
        command += ["-map", "1:a?"]
    else:
        command += ["-map", f"1:a:{max(0, int(audio_track_no) - 1)}?"]
    command += get_video_encode_args(video_codec, video_container) + ["-c:a", "copy", "-shortest"]
    if normalize_container_name(video_container) == "mp4":
        command += ["-strict", "-2"]
    command += _get_live_mux_output_args(video_container)
    command += [output_path]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, bufsize=0)


def start_hdr_video_mux_process(ffmpeg_path: str, output_path: str, width: int, height: int, fps_float: float, video_codec: str, video_container: str, reserved_metadata_path: str | None = None) -> subprocess.Popen:
    command = [
        ffmpeg_path,
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gbrpf32le",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        f"{float(fps_float):.12g}",
        "-i",
        "pipe:0",
    ]
    metadata_input_index = None
    if reserved_metadata_path and os.path.isfile(reserved_metadata_path):
        command += ["-f", "ffmetadata", "-i", reserved_metadata_path]
        metadata_input_index = 1
    if metadata_input_index is not None:
        command += ["-map_metadata", str(metadata_input_index)]
    command += ["-map", "0:v:0"]
    command += get_hdr_video_encode_args(video_codec, video_container)
    command += _get_live_mux_output_args(video_container)
    command += [output_path]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, bufsize=0)


def start_hdr_av_mux_process(ffmpeg_path: str, output_path: str, width: int, height: int, fps_float: float, video_codec: str, video_container: str, source_path: str, start_seconds: float, audio_track_no: int | None, reserved_metadata_path: str | None = None, source_audio_duration_seconds: float | None = None) -> subprocess.Popen:
    command = [
        ffmpeg_path,
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gbrpf32le",
        "-video_size",
        f"{width}x{height}",
        "-framerate",
        f"{float(fps_float):.12g}",
        "-i",
        "pipe:0",
        "-stream_loop",
        "-1",
        "-ss",
        f"{max(0.0, float(start_seconds)):.12g}",
    ]
    if source_audio_duration_seconds is not None and source_audio_duration_seconds > 0.0:
        command += ["-t", f"{max(0.0, float(source_audio_duration_seconds) + LIVE_AUDIO_LOOP_PADDING_SECONDS):.12g}"]
    command += ["-i", source_path]
    if reserved_metadata_path and os.path.isfile(reserved_metadata_path):
        command += ["-f", "ffmetadata", "-i", reserved_metadata_path, "-map_metadata", "2"]
    else:
        command += ["-map_metadata", "1"]
    command += ["-map", "0:v:0"]
    if audio_track_no is None:
        command += ["-map", "1:a?"]
    else:
        command += ["-map", f"1:a:{max(0, int(audio_track_no) - 1)}?"]
    command += get_hdr_video_encode_args(video_codec, video_container) + ["-c:a", "copy", "-shortest"]
    command += _get_live_mux_output_args(video_container)
    command += [output_path]
    return subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, bufsize=0)


def finalize_mux_process(process: subprocess.Popen, *, timeout_seconds: float = DEFAULT_FFMPEG_TIMEOUT_SECONDS, label: str = "") -> tuple[int, str, bool]:
    if process.stdin is not None and not process.stdin.closed:
        try:
            process.stdin.close()
        except OSError:
            pass
    forced_termination = False
    target_desc = label.strip() if label and label.strip() else "output"
    print(f"[MediaFlow] Finalizing ffmpeg {target_desc} (timeout: {timeout_seconds:.0f}s)...")
    try:
        return_code = process.wait(timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        forced_termination = True
        print(f"[MediaFlow] Warning: ffmpeg did not exit within {timeout_seconds}s timeout; terminating process.")
        process.kill()
        return_code = process.wait(timeout=5)
    stderr = process.stderr.read().decode("utf-8", errors="ignore").strip() if process.stderr is not None else ""
    return return_code, stderr, forced_termination


def _reserve_sidecar_path(preferred_path: Path, fallback_path_factory) -> str:
    for attempt in range(1000):
        candidate = preferred_path if attempt == 0 else fallback_path_factory(uuid.uuid4().hex[:8])
        try:
            handle = os.open(str(candidate), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            continue
        try:
            os.close(handle)
        except OSError:
            pass
        return str(candidate)
    raise gr.Error(f"Unable to reserve a scratch filename near {preferred_path.parent}")


def reserve_working_output_path(output_path: str) -> str:
    output = Path(output_path).resolve()
    preferred = output.with_name(f"{output.stem}_working{output.suffix}")
    return _reserve_sidecar_path(preferred, lambda token: output.with_name(f"{output.stem}_working_{token}{output.suffix}"))


def create_reserved_metadata_file(output_path: str, metadata: dict | None = None) -> str:
    output = Path(output_path).resolve()
    preferred = output.with_name(f"{output.name}.ffmeta")
    reserved_metadata_path = _reserve_sidecar_path(preferred, lambda token: output.with_name(f"{output.name}.{token}.ffmeta"))
    write_reserved_video_ffmetadata(reserved_metadata_path, DEFAULT_RESERVED_VIDEO_METADATA_BYTES, metadata)
    return reserved_metadata_path


def delete_file_if_exists(file_path: str | None, *, label: str) -> None:
    if not isinstance(file_path, str) or not os.path.isfile(file_path):
        return
    try:
        os.remove(file_path)
    except OSError as exc:
        print(f"[MediaFlow] Warning: failed to delete {label} {file_path}: {exc}")


def _normalize_artifact_paths(paths) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for path in list(paths or []):
        if not isinstance(path, str) or len(path.strip()) == 0:
            continue
        resolved = str(Path(path).resolve())
        if resolved in seen:
            continue
        normalized.append(resolved)
        seen.add(resolved)
    return normalized


def result_generated_artifact_paths(result) -> list[str]:
    paths = list(getattr(result, "generated_files", []) or [])
    for artifact in tuple(getattr(result, "artifacts", ()) or ()):
        artifact_path = getattr(artifact, "path", None)
        if isinstance(artifact_path, str) and len(artifact_path.strip()) > 0:
            paths.append(artifact_path)
    return _normalize_artifact_paths(paths)


def remember_generated_artifacts(active_job: dict, paths) -> None:
    active_job["generated_artifact_paths"] = _normalize_artifact_paths(list(active_job.get("generated_artifact_paths", []) or []) + list(paths or []))


def _remove_paths_from_gallery(file_list: list, file_settings_list: list, deleted_paths: set[str]) -> bool:
    if len(deleted_paths) == 0:
        return False
    kept_files = []
    kept_settings = []
    changed = False
    for index, file_path in enumerate(list(file_list or [])):
        resolved = str(Path(file_path).resolve()) if isinstance(file_path, str) and len(file_path.strip()) > 0 else ""
        if resolved in deleted_paths:
            changed = True
            continue
        kept_files.append(file_path)
        kept_settings.append(file_settings_list[index] if index < len(file_settings_list) else None)
    if changed:
        file_list[:] = kept_files
        file_settings_list[:] = kept_settings
    return changed


def _clamp_gallery_selection(state: dict, *, video_changed: bool, audio_changed: bool) -> None:
    gen = state.get("gen", {})
    if not isinstance(gen, dict):
        return
    if video_changed:
        gen["selected"] = min(max(int(gen.get("selected", 0) or 0), 0), max(len(gen.get("file_list", []) or []) - 1, 0))
    if audio_changed:
        gen["audio_selected"] = min(max(int(gen.get("audio_selected", -1) or -1), -1), max(len(gen.get("audio_file_list", []) or []) - 1, -1))


def prune_generated_artifacts(
    state: dict,
    artifact_paths: list[str],
    preserve_count: int,
    *,
    preserve_paths: list[str] | None = None,
    get_video_gallery: Callable,
    get_audio_gallery: Callable,
) -> tuple[list[str], bool]:
    normalized_paths = _normalize_artifact_paths(artifact_paths)
    preserve_count = max(1, int(preserve_count))
    protected_paths = set(normalized_paths[-preserve_count:])
    protected_paths.update(_normalize_artifact_paths(preserve_paths))
    deleted_paths: set[str] = set()
    kept_paths: list[str] = []
    for path in normalized_paths:
        if path in protected_paths:
            kept_paths.append(path)
            continue
        if os.path.isfile(path):
            try:
                os.remove(path)
            except OSError as exc:
                print(f"[MediaFlow] Warning: failed to delete old generated artifact {path}: {exc}")
                kept_paths.append(path)
                continue
        deleted_paths.add(path)
    if len(deleted_paths) == 0:
        return kept_paths, False
    video_file_list, video_file_settings_list = get_video_gallery(state)
    audio_file_list, audio_file_settings_list = get_audio_gallery(state)
    video_changed = _remove_paths_from_gallery(video_file_list, video_file_settings_list, deleted_paths)
    audio_changed = _remove_paths_from_gallery(audio_file_list, audio_file_settings_list, deleted_paths)
    _clamp_gallery_selection(state, video_changed=video_changed, audio_changed=audio_changed)
    return kept_paths, video_changed or audio_changed


def prune_active_job_artifacts(plugin, active_job: dict, state: dict, *, preserve_paths: list[str] | None = None) -> tuple[list[str], bool]:
    from . import process_catalog as catalog

    kept_paths, gallery_changed = prune_generated_artifacts(
        state,
        list(active_job.get("generated_artifact_paths", []) or []),
        catalog.normalize_preserve_artifact_count(active_job.get(catalog.PRESERVE_ARTIFACTS_STORAGE_KEY)),
        preserve_paths=preserve_paths,
        get_video_gallery=plugin.get_current_video_gallery,
        get_audio_gallery=plugin.get_current_audio_gallery,
    )
    active_job["generated_artifact_paths"] = kept_paths
    return kept_paths, gallery_changed


def delete_released_chunk_outputs(state: dict, chunk_output_paths: list[str], *, preserve_paths: list[str] | None = None) -> list[str]:
    return _normalize_artifact_paths(chunk_output_paths)


def get_last_generated_video_path(paths: list[str]) -> str | None:
    video_paths = [str(Path(path).resolve()) for path in paths if isinstance(path, str) and os.path.isfile(path) and str(Path(path).suffix).lower() in {".mp4", ".mkv", ".mov"}]
    return video_paths[-1] if len(video_paths) > 0 else None


def _make_output_temp_dir(output_path: str, prefix: str) -> str:
    return tempfile.mkdtemp(prefix=prefix, dir=str(Path(output_path).resolve().parent))


def _probe_media_stream_layout(ffprobe_path: str, media_path: str) -> tuple[int, int]:
    result = subprocess.run([ffprobe_path, "-v", "error", "-show_entries", "stream=codec_type", "-of", "json", media_path], capture_output=True, text=True, **child_kwargs())
    if result.returncode != 0:
        raise gr.Error((result.stderr or result.stdout or f"ffprobe failed for {media_path}").strip())
    try:
        streams = list((json.loads(result.stdout) or {}).get("streams") or [])
    except json.JSONDecodeError as exc:
        raise gr.Error(f"Unable to read media stream layout for {media_path}") from exc
    video_count = sum(1 for stream in streams if str(stream.get("codec_type") or "").lower() == "video")
    audio_count = sum(1 for stream in streams if str(stream.get("codec_type") or "").lower() == "audio")
    return video_count, audio_count


def _probe_primary_video_codec(ffprobe_path: str, media_path: str) -> str:
    result = subprocess.run([ffprobe_path, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name", "-of", "json", media_path], capture_output=True, text=True, encoding="utf-8", errors="ignore", check=False, **child_kwargs())
    if result.returncode != 0:
        raise gr.Error((result.stderr or result.stdout or f"ffprobe failed for {media_path}").strip())
    try:
        codec_name = str((((json.loads(result.stdout) or {}).get("streams") or [{}])[0]).get("codec_name") or "").strip().lower()
    except (TypeError, ValueError, json.JSONDecodeError, IndexError):
        codec_name = ""
    if len(codec_name) == 0:
        raise gr.Error(f"Unable to detect the video codec of {media_path}")
    return codec_name


def _probe_primary_video_start_time(ffprobe_path: str, media_path: str) -> float:
    result = subprocess.run([ffprobe_path, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=start_time", "-of", "json", media_path], capture_output=True, text=True, encoding="utf-8", errors="ignore", check=False, **child_kwargs())
    if result.returncode == 0:
        try:
            stream = (((json.loads(result.stdout) or {}).get("streams") or [{}])[0])
            start_time = stream.get("start_time")
            if start_time not in (None, "", "N/A"):
                return max(0.0, float(start_time))
        except (TypeError, ValueError, json.JSONDecodeError, IndexError):
            pass
    packet_times = _probe_video_packet_times(ffprobe_path, media_path, start_seconds=0.0, duration_seconds=1.0)
    return max(0.0, min(packet_times)) if len(packet_times) > 0 else 0.0


def _probe_video_packet_times(ffprobe_path: str, media_path: str, *, start_seconds: float | None = None, duration_seconds: float | None = None) -> list[float]:
    command = [ffprobe_path, "-v", "error", "-select_streams", "v:0"]
    if start_seconds is not None and duration_seconds is not None and float(duration_seconds) > 0.0:
        command += ["-read_intervals", f"{max(0.0, float(start_seconds)):.6f}%+{max(0.05, float(duration_seconds)):.6f}"]
    command += ["-show_packets", "-show_entries", "packet=dts_time,pts_time", "-of", "json", media_path]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="ignore", check=False, **child_kwargs())
    if result.returncode != 0:
        raise gr.Error((result.stderr or result.stdout or f"ffprobe failed for {media_path}").strip())
    try:
        return sorted(
            float(packet.get("dts_time") if packet.get("dts_time") is not None else packet.get("pts_time"))
            for packet in ((json.loads(result.stdout) or {}).get("packets") or [])
            if packet.get("dts_time") is not None or packet.get("pts_time") is not None
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return []


def _probe_video_frame_gap(ffprobe_path: str, media_path: str, fps_float: float, *, near_time: float | None = None, window_seconds: float = 4.0) -> tuple[float, float, float] | None:
    fps_value = float(fps_float or 0.0)
    if fps_value <= 0.0 or not os.path.isfile(media_path):
        return None
    max_delta = max(1.0 / fps_value * 1.5, 0.05)
    if near_time is None:
        packet_times = _probe_video_packet_times(ffprobe_path, media_path)
        if len(packet_times) < 2:
            return None
        for current_pts, next_pts in zip(packet_times, packet_times[1:]):
            delta = float(next_pts) - float(current_pts)
            if delta > max_delta:
                return float(current_pts), float(next_pts), float(delta)
        return None
    seam_time = max(0.0, float(near_time))
    local_window = max(1.0, float(window_seconds))
    seam_margin = max_delta * 2.0
    total_duration = max(seam_time, _probe_media_duration(ffprobe_path, media_path))
    probe_start = max(0.0, seam_time - max(0.25, seam_margin))
    probe_duration = min(max(local_window, 4.0), max(0.05, total_duration - probe_start))
    while probe_duration > 0.0:
        packet_times = _probe_video_packet_times(ffprobe_path, media_path, start_seconds=probe_start, duration_seconds=probe_duration)
        prev_candidates = [packet_time for packet_time in packet_times if packet_time <= seam_time + seam_margin]
        if len(prev_candidates) > 0:
            prev_time = prev_candidates[-1]
            next_candidates = [packet_time for packet_time in packet_times if packet_time > prev_time + 1e-9]
            if len(next_candidates) > 0:
                next_time = next_candidates[0]
                delta = float(next_time) - float(prev_time)
                return None if delta <= max_delta else (float(prev_time), float(next_time), float(delta))
        if probe_start + probe_duration >= total_duration - 1e-6:
            return None
        probe_duration = min(max(2.0 * probe_duration, local_window), max(0.05, total_duration - probe_start))
    return None


def _write_concat_list(list_path: str, media_paths: list[str]) -> None:
    with open(list_path, "w", encoding="utf-8") as handle:
        for media_path in media_paths:
            escaped_path = str(media_path).replace("'", "'\\''")
            handle.write(f"file '{escaped_path}'\n")


def _split_sparse_fine_seek(seconds: float) -> tuple[float, float]:
    seconds = max(0.0, float(seconds or 0.0))
    coarse_seconds = max(0.0, seconds - 1.0)
    return coarse_seconds, seconds - coarse_seconds


def _build_video_reconstruct_bsf(fps_float: float) -> str:
    fps_value = max(float(fps_float or 0.0), 1.0)
    frame_duration_expr = f"1/({fps_value:.15g}*TB)"
    return (
        "setts="
        f"pts='if(eq(N,0),PTS,PREV_OUTPTS+(PTS-PREV_INPTS)-(PREV_INDURATION-DURATION))':"
        f"dts='if(eq(N,0),DTS,PREV_OUTDTS+(DTS-PREV_INDTS)-(PREV_INDURATION-DURATION))':"
        f"duration='if(eq(N,0),{frame_duration_expr},DURATION)'"
    )


def _concat_audio_segments(ffmpeg_path: str, segment_paths: list[str], output_path: str, work_dir: str, *, segment_trim_seconds: list[float] | None = None, segment_duration_seconds: list[float | None] | None = None, audio_stream_indices: list[int] | None = None) -> None:
    extracted_paths: list[str] = []
    list_path = os.path.join(work_dir, "audio.txt")
    try:
        for segment_no, segment_path in enumerate(segment_paths, start=1):
            extracted_path = os.path.join(work_dir, f"segment_{segment_no}_audio.mka")
            trim_seconds = max(0.0, float(segment_trim_seconds[segment_no - 1])) if segment_trim_seconds is not None and segment_no - 1 < len(segment_trim_seconds) else 0.0
            duration_seconds = None
            if segment_duration_seconds is not None and segment_no - 1 < len(segment_duration_seconds) and segment_duration_seconds[segment_no - 1] is not None:
                duration_seconds = max(0.0, float(segment_duration_seconds[segment_no - 1]))
            audio_stream_index = max(0, int(audio_stream_indices[segment_no - 1])) if audio_stream_indices is not None and segment_no - 1 < len(audio_stream_indices) else 0
            command = [ffmpeg_path, "-y", "-v", "error"]
            coarse_seek_seconds, fine_seek_seconds = _split_sparse_fine_seek(trim_seconds)
            if coarse_seek_seconds > 0.0:
                command += ["-ss", f"{coarse_seek_seconds:.12g}"]
            command += ["-i", segment_path]
            if fine_seek_seconds > 0.0:
                command += ["-ss", f"{fine_seek_seconds:.12g}"]
            if duration_seconds is not None and duration_seconds > 0.0:
                command += ["-t", f"{duration_seconds:.12g}"]
            command += ["-map", f"0:a:{audio_stream_index}?", "-c", "copy", "-avoid_negative_ts", "make_zero", extracted_path]
            result = subprocess.run(command, capture_output=True, text=True, **child_kwargs())
            if result.returncode != 0 or not os.path.isfile(extracted_path):
                raise gr.Error((result.stderr or result.stdout or f"ffmpeg failed to extract audio from {segment_path}").strip())
            extracted_paths.append(extracted_path)
        _write_concat_list(list_path, extracted_paths)
        result = subprocess.run([ffmpeg_path, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", output_path], capture_output=True, text=True, **child_kwargs())
        if result.returncode != 0 or not os.path.isfile(output_path):
            raise gr.Error((result.stderr or result.stdout or "ffmpeg failed to concatenate audio streams").strip())
    finally:
        for extracted_path in extracted_paths:
            if os.path.isfile(extracted_path):
                os.remove(extracted_path)
        if os.path.isfile(list_path):
            os.remove(list_path)


def concat_video_segments(
    ffmpeg_path: str,
    segment_paths: list[str],
    output_path: str,
    video_codec: str,
    video_container: str,
    audio_codec_key: str,
    *,
    segment_audio_trim_seconds: list[float] | None = None,
    segment_audio_duration_seconds: list[float | None] | None = None,
    fps_float: float | None = None,
    selected_audio_track_no: int | None = None,
    reserved_metadata_path: str | None = None,
    source_audio_path: str | None = None,
    source_audio_start_seconds: float | None = None,
    source_audio_duration_seconds: float | None = None,
    source_audio_track_no: int | None = None,
) -> None:
    normalized_segment_paths: list[str] = []
    for path in segment_paths:
        segment_path = str(Path(str(path).strip()).resolve())
        if len(segment_path) == 0 or not os.path.isfile(segment_path):
            raise gr.Error(f"Output segment is missing and cannot be merged: {path}")
        normalized_segment_paths.append(segment_path)
    segment_paths = normalized_segment_paths
    if len(segment_paths) == 0:
        raise gr.Error("No output segments available to merge.")
    if len(segment_paths) == 1:
        if str(Path(segment_paths[0]).resolve()) != str(Path(output_path).resolve()):
            try:
                os.replace(segment_paths[0], output_path)
            except OSError as exc:
                raise ContinuationMergeOutputLockedError(output_path) from exc
        return
    ffprobe_path = resolve_media_binary("ffprobe")
    layouts = [_probe_media_stream_layout(ffprobe_path, path) for path in segment_paths]
    if any(video_count != 1 for video_count, _ in layouts):
        raise gr.Error("All continuation segments must contain exactly one video stream.")
    use_source_audio_merge = isinstance(source_audio_path, str) and len(str(source_audio_path).strip()) > 0
    if use_source_audio_merge:
        validate_audio_copy_container(ffprobe_path, str(source_audio_path), video_container, source_audio_track_no)
    audio_stream_counts = [audio_count for _, audio_count in layouts]
    has_audio = use_source_audio_merge or any(audio_count > 0 for audio_count in audio_stream_counts)
    if not use_source_audio_merge and has_audio and any(audio_count <= 0 for audio_count in audio_stream_counts):
        raise gr.Error("All continuation segments must expose an audio stream.")
    fps_value = float(fps_float or 0.0)
    concat_dir = _make_output_temp_dir(output_path, "wangp_process_full_video_concat_")
    merged_video_path = os.path.join(concat_dir, "merged_video.mkv")
    temp_output_path = os.path.join(concat_dir, f"{Path(output_path).stem}_merged{Path(output_path).suffix}")
    try:
        video_codec_name = _probe_primary_video_codec(ffprobe_path, segment_paths[0])
        video_bsf = "h264_mp4toannexb" if video_codec_name == "h264" else "hevc_mp4toannexb" if video_codec_name in ("hevc", "h265") else ""
        if len(video_bsf) == 0:
            raise gr.Error(f"Unsupported continuation video codec for no-reencode merge: {video_codec_name}")
        reconstructed_video_bsf = f"{_build_video_reconstruct_bsf(fps_value)},{video_bsf}"
        if use_source_audio_merge:
            command = [ffmpeg_path, "-y", "-v", "error", "-f", "mpegts", "-i", "pipe:0", "-ss", f"{max(0.0, float(source_audio_start_seconds or 0.0)):.12g}", "-t", f"{max(0.0, float(source_audio_duration_seconds or 0.0)):.12g}", "-i", str(source_audio_path)]
            if reserved_metadata_path and os.path.isfile(reserved_metadata_path):
                command += ["-f", "ffmetadata", "-i", reserved_metadata_path, "-map_metadata", "2"]
            else:
                command += ["-map_metadata", "1"]
            command += ["-map", "0:v:0"]
            if source_audio_track_no is None:
                command += ["-map", "1:a?"]
            else:
                command += ["-map", f"1:a:{max(0, int(source_audio_track_no) - 1)}?"]
            command += ["-c", "copy"]
            if is_quicktime_output_container(video_container):
                command += ["-movflags", "+faststart"]
            command += [temp_output_path]
            mux_process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, bufsize=0)
            try:
                if mux_process.stdin is None:
                    raise gr.Error("ffmpeg source-audio merge did not expose a writable video pipe.")
                for segment_path in segment_paths:
                    segment_process = subprocess.Popen([ffmpeg_path, "-v", "error", "-i", segment_path, "-map", "0:v:0", "-c", "copy", "-bsf:v", reconstructed_video_bsf, "-f", "mpegts", "-"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0, **child_kwargs())
                    try:
                        if segment_process.stdout is None:
                            raise gr.Error(f"ffmpeg failed to expose the TS stream for {segment_path}.")
                        shutil.copyfileobj(segment_process.stdout, mux_process.stdin, length=1024 * 1024)
                    except Exception:
                        segment_process.kill()
                        segment_process.wait(timeout=5)
                        raise
                    finally:
                        if segment_process.stdout is not None:
                            segment_process.stdout.close()
                    segment_returncode = segment_process.wait()
                    segment_stderr = segment_process.stderr.read().decode("utf-8", errors="ignore").strip() if segment_process.stderr is not None else ""
                    if segment_returncode != 0:
                        raise gr.Error((segment_stderr or f"ffmpeg failed to stream {segment_path} for concat").strip())
                mux_returncode, mux_stderr, _ = finalize_mux_process(mux_process, label="merged continuation")
            except Exception:
                try:
                    if mux_process.stdin is not None and not mux_process.stdin.closed:
                        mux_process.stdin.close()
                except OSError:
                    pass
                mux_process.kill()
                mux_process.wait(timeout=5)
                raise
            if mux_returncode != 0 or not os.path.isfile(temp_output_path):
                raise gr.Error((mux_stderr or "ffmpeg source-audio merge failed").strip())
        else:
            concat_ts_path = os.path.join(concat_dir, "segments.ts")
            ts_paths: list[str] = []
            for index, segment_path in enumerate(segment_paths, start=1):
                ts_path = os.path.join(concat_dir, f"segment_{index}.ts")
                result = subprocess.run([ffmpeg_path, "-y", "-v", "error", "-i", segment_path, "-map", "0:v:0", "-c", "copy", "-bsf:v", reconstructed_video_bsf, "-f", "mpegts", ts_path], capture_output=True, text=True, **child_kwargs())
                if result.returncode != 0 or not os.path.isfile(ts_path):
                    raise gr.Error((result.stderr or result.stdout or f"ffmpeg failed to prepare {segment_path} for concat").strip())
                ts_paths.append(ts_path)
            with open(concat_ts_path, "wb") as handle:
                for ts_path in ts_paths:
                    with open(ts_path, "rb") as ts_file:
                        shutil.copyfileobj(ts_file, handle, length=1024 * 1024)
            result = subprocess.run([ffmpeg_path, "-y", "-v", "error", "-i", concat_ts_path, "-map", "0:v:0", "-c", "copy", merged_video_path], capture_output=True, text=True, **child_kwargs())
            if result.returncode != 0 or not os.path.isfile(merged_video_path):
                raise gr.Error((result.stderr or result.stdout or "ffmpeg failed to concatenate video stream").strip())
            command = [ffmpeg_path, "-y", "-v", "error", "-i", merged_video_path]
            if has_audio:
                merged_audio_path = os.path.join(concat_dir, "merged_audio.mka")
                selected_audio_index = max(0, int(selected_audio_track_no or 1) - 1)
                audio_stream_indices = [0 if audio_count <= 1 else min(audio_count - 1, selected_audio_index) for audio_count in audio_stream_counts]
                resolved_audio_duration_seconds = [None] * len(segment_paths)
                if segment_audio_duration_seconds is not None:
                    for segment_index, duration_seconds in enumerate(segment_audio_duration_seconds[:len(segment_paths)]):
                        if duration_seconds is not None:
                            resolved_audio_duration_seconds[segment_index] = max(0.0, float(duration_seconds))
                elif fps_value > 0.0:
                    for segment_index, segment_path in enumerate(segment_paths):
                        segment_frame_count, _ = probe_resume_frame_count(ffprobe_path, segment_path, fps_value)
                        if segment_frame_count > 0:
                            resolved_audio_duration_seconds[segment_index] = float(segment_frame_count) / fps_value
                _concat_audio_segments(ffmpeg_path, segment_paths, merged_audio_path, concat_dir, segment_trim_seconds=segment_audio_trim_seconds, segment_duration_seconds=resolved_audio_duration_seconds, audio_stream_indices=audio_stream_indices)
                command += ["-i", merged_audio_path]
            if reserved_metadata_path and os.path.isfile(reserved_metadata_path):
                command += ["-f", "ffmetadata", "-i", reserved_metadata_path]
                command += ["-map_metadata", "2" if has_audio else "1"]
            else:
                command += ["-map_metadata", "0"]
            command += ["-map", "0:v:0"]
            if has_audio:
                command += ["-map", "1:a:0"]
            command += ["-c", "copy"]
            if is_quicktime_output_container(video_container):
                command += ["-movflags", "+faststart"]
            command += [temp_output_path]
            result = subprocess.run(command, capture_output=True, text=True, **child_kwargs())
            if result.returncode != 0 or not os.path.isfile(temp_output_path):
                raise gr.Error((result.stderr or result.stdout or "ffmpeg concat failed").strip())
        timeline_gap = _probe_video_frame_gap(ffprobe_path, temp_output_path, fps_value, near_time=_probe_media_duration(ffprobe_path, segment_paths[0]))
        if timeline_gap is not None:
            gap_start, gap_end, gap_seconds = timeline_gap
            raise gr.Error(f"Merged video timeline contains a {gap_seconds:.6f}s gap near {gap_start:.3f}s -> {gap_end:.3f}s.")
        try:
            os.replace(temp_output_path, output_path)
        except OSError as exc:
            raise ContinuationMergeOutputLockedError(output_path) from exc
    finally:
        shutil.rmtree(concat_dir, ignore_errors=True)
