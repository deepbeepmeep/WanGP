# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import json
import os
import shutil
import subprocess
import threading
from functools import lru_cache

import numpy as np
import torch

from .virtual_media import clamp_virtual_frame_range, get_virtual_media_entry, parse_virtual_media_path, strip_virtual_media_suffix
from shared.utils.subprocess_stdin import child_kwargs

_ZSCALE_TRANSFER_MAP = {"smpte2084": "smpte2084", "arib-std-b67": "arib-std-b67", "bt709": "bt709", "bt2020-10": "2020_10", "bt2020-12": "2020_12"}
_ZSCALE_PRIMARIES_MAP = {"bt2020": "2020", "bt709": "709", "smpte170m": "170m", "bt470bg": "470bg"}
_ZSCALE_MATRIX_MAP = {"bt2020nc": "2020_ncl", "bt2020c": "2020_cl", "bt709": "709", "smpte170m": "170m", "bt470bg": "470bg"}
_ZSCALE_RANGE_MAP = {"tv": "limited", "limited": "limited", "pc": "full", "full": "full"}
_HDR_REFERENCE_WHITE_NITS = 203
_VIRTUAL_MEDIA_PRESEEK_FRAMES = 64
_VIRTUAL_MEDIA_LOCAL_SEARCH_FRAMES = 8


def _parse_media_ratio(value, default=None):
    if value in [None, "", "N/A", "0:1", "0/0"]:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if ":" in text:
        num, den = text.split(":", 1)
    elif "/" in text:
        num, den = text.split("/", 1)
    else:
        try:
            return float(text)
        except (TypeError, ValueError):
            return default
    try:
        num = float(num)
        den = float(den)
    except (TypeError, ValueError):
        return default
    return default if den == 0 else num / den


def _parse_media_duration(value, default=0.0):
    if value in [None, "", "N/A"]:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if ":" not in text:
        try:
            return float(text)
        except (TypeError, ValueError):
            return default
    try:
        total = 0.0
        for part in text.split(":"):
            total = total * 60.0 + float(part)
        return total
    except (TypeError, ValueError):
        return default


def _resample_frame_indices(video_fps, video_frames_count, max_target_frames_count, target_fps, start_target_frame):
    import math

    video_frame_duration = 1 / video_fps
    target_frame_duration = 1 / target_fps
    target_time = start_target_frame * target_frame_duration
    frame_no = math.ceil(target_time / video_frame_duration)
    cur_time = frame_no * video_frame_duration
    frame_ids = []
    while True:
        if max_target_frames_count != 0 and len(frame_ids) >= max_target_frames_count:
            break
        diff = round((target_time - cur_time) / video_frame_duration, 5)
        add_frames_count = math.ceil(diff)
        frame_no += add_frames_count
        if frame_no >= video_frames_count:
            break
        frame_ids.append(frame_no)
        cur_time += add_frames_count * video_frame_duration
        target_time += target_frame_duration
    return frame_ids[:max_target_frames_count]


def _resolve_media_binary(binary_name: str):
    env_map = {"ffmpeg": "FFMPEG_BINARY", "ffprobe": "FFPROBE_BINARY", "ffplay": "FFPLAY_BINARY"}
    binary_path = os.environ.get(env_map.get(binary_name, ""), "")
    if len(binary_path) > 0 and os.path.isfile(binary_path):
        return binary_path
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    candidate = os.path.join(repo_root, "ffmpeg_bins", binary_name + (".exe" if os.name == "nt" else ""))
    if os.path.isfile(candidate):
        return candidate
    return shutil.which(binary_name + (".exe" if os.name == "nt" else "")) or shutil.which(binary_name)


def resolve_media_binary(binary_name: str):
    return _resolve_media_binary(binary_name)


def _augment_virtual_metadata(video_path, metadata):
    spec = parse_virtual_media_path(video_path)
    if spec is None or metadata is None:
        return metadata
    total_frames = int(metadata.get("frame_count") or 0)
    start_frame, end_frame = clamp_virtual_frame_range(spec, total_frames)
    virtual_metadata = dict(metadata)
    virtual_metadata["source_path"] = spec.source_path
    virtual_metadata["virtual_start_frame"] = start_frame
    virtual_metadata["virtual_end_frame"] = end_frame
    if end_frame is None:
        return virtual_metadata
    virtual_frame_count = max(0, end_frame - start_frame + 1)
    virtual_metadata["frame_count"] = virtual_frame_count
    fps_float = float(virtual_metadata.get("fps_float") or 0.0)
    fps = int(virtual_metadata.get("fps") or 0)
    effective_fps = fps_float if fps_float > 0 else float(fps or 0)
    if effective_fps > 0:
        virtual_metadata["duration"] = virtual_frame_count / effective_fps
    return virtual_metadata


def _build_vsource_metadata(video_path, entry):
    if not isinstance(entry, dict):
        return None
    if entry.get("kind") == "image":
        image = entry.get("image")
        if image is None:
            return None
        width, height = image.size
        fps_float = 1.0
        frame_count = 1
    elif entry.get("kind") == "video":
        tensor = entry.get("tensor")
        if tensor is None or int(getattr(tensor, "ndim", 0)) != 4:
            return None
        width = int(tensor.shape[3])
        height = int(tensor.shape[2])
        frame_count = int(tensor.shape[1])
        fps_float = max(float(entry.get("fps") or 0.0), 1.0)
    else:
        return None
    return _augment_virtual_metadata(video_path, {
        "source_path": parse_virtual_media_path(video_path).source_path if parse_virtual_media_path(video_path) is not None else "",
        "width": width,
        "height": height,
        "display_width": width,
        "display_height": height,
        "fps_float": fps_float,
        "fps": int(round(fps_float)),
        "frame_count": frame_count,
        "duration": float(frame_count / fps_float) if fps_float > 0 else 0.0,
        "start_time": 0.0,
        "sample_aspect_ratio": "1:1",
        "display_aspect_ratio": "",
        "color_transfer": "",
        "color_primaries": "",
        "color_space": "",
        "color_range": "",
        "needs_sar_fix": False,
        "needs_tonemap": False,
        "hdr": bool(entry.get("hdr")),
    })


def _frame_timestamp(frame):
    for key in ("best_effort_timestamp_time", "pts_time", "pkt_pts_time"):
        try:
            return float(frame.get(key))
        except (TypeError, ValueError):
            pass
    return None


def _probe_mkv_sparse_frame_count(source_path, ffprobe_path, duration, start_time, fps_float):
    if duration <= 0 or fps_float <= 0:
        return 0
    candidate = int(round(max(0.0, duration - max(0.0, start_time)) * fps_float))
    if candidate <= 0:
        return 0
    tail_frames = 64
    tail_start = max(0.0, start_time + max(0, candidate - tail_frames) / fps_float)
    probe_cmd = [ffprobe_path, "-v", "error", "-select_streams", "v:0", "-read_intervals", f"{tail_start:.6f}%+#96", "-show_entries", "frame=best_effort_timestamp_time,pts_time,pkt_pts_time", "-of", "json", source_path]
    probe = subprocess.run(probe_cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore", check=False, **child_kwargs())
    if probe.returncode != 0:
        return candidate
    try:
        timestamps = [timestamp for timestamp in (_frame_timestamp(frame) for frame in (json.loads(probe.stdout).get("frames") or [])) if timestamp is not None]
    except (json.JSONDecodeError, TypeError, ValueError):
        return candidate
    if not timestamps or min(timestamps) < tail_start - tail_frames / fps_float:
        return candidate
    return max(1, int(round((max(timestamps) - start_time) * fps_float)) + 1)


@lru_cache(maxsize=128)
def probe_video_stream_metadata(video_path, *, file_version=None):
    # Callers needing fresh file facts include (mtime_ns, size) in the cache key.
    video_path = os.fspath(video_path)
    if (entry := get_virtual_media_entry(video_path)) is not None:
        return _build_vsource_metadata(video_path, entry)
    source_path = os.fspath(strip_virtual_media_suffix(video_path))
    ffprobe_path = _resolve_media_binary("ffprobe")
    if ffprobe_path is None:
        return None
    probe_cmd = [ffprobe_path, "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_format", "-of", "json", source_path]
    probe = subprocess.run(probe_cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore", check=False, **child_kwargs())
    if probe.returncode != 0:
        return None
    try:
        probe_data = json.loads(probe.stdout)
    except json.JSONDecodeError:
        return None
    streams = probe_data.get("streams") or []
    if len(streams) == 0:
        return None
    stream = streams[0]
    codec_name = str(stream.get("codec_name") or "")
    codec_profile = str(stream.get("profile") or "")
    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if width <= 0 or height <= 0:
        return None
    sar = _parse_media_ratio(stream.get("sample_aspect_ratio"), 1.0) or 1.0
    dar = _parse_media_ratio(stream.get("display_aspect_ratio"))
    display_width = width
    if abs(sar - 1.0) > 1e-6:
        display_width = max(2, (int(width * sar) // 2) * 2)
    elif dar is not None and dar > 0:
        display_width = max(2, (int(height * dar) // 2) * 2)
    fps_float = _parse_media_ratio(stream.get("avg_frame_rate"), 0.0) or _parse_media_ratio(stream.get("r_frame_rate"), 0.0) or 0.0
    duration = stream.get("duration") or (probe_data.get("format") or {}).get("duration") or 0.0
    try:
        duration = float(duration)
    except (TypeError, ValueError):
        duration = 0.0
    try:
        start_time = float(stream.get("start_time") or 0.0)
    except (TypeError, ValueError):
        start_time = 0.0
    try:
        frame_count = int(stream.get("nb_frames"))
    except (TypeError, ValueError):
        frame_count = 0
        if source_path.lower().endswith(".mkv"):
            frame_count = _probe_mkv_sparse_frame_count(source_path, ffprobe_path, _parse_media_duration((stream.get("tags") or {}).get("DURATION"), duration), start_time, fps_float)
        if frame_count <= 0:
            frame_count = int(round(duration * fps_float)) if duration > 0 and fps_float > 0 else 0
    side_data = stream.get("side_data_list") or []
    rotation = 0
    for item in side_data:
        if item.get("rotation") is not None:
            try:
                rotation = int(round(float(item["rotation"])))
            except (TypeError, ValueError, OverflowError):
                pass
    needs_sar_fix = display_width != width
    display_height = height
    # FFmpeg autorotates before our filters; preserve the stored-raster SAR decision.
    if abs(rotation) % 180 == 90:
        display_width, display_height = display_height, display_width
    color_transfer = str(stream.get("color_transfer") or "").lower()
    color_primaries = str(stream.get("color_primaries") or "").lower()
    color_space = str(stream.get("color_space") or "").lower()
    color_range = str(stream.get("color_range") or "").lower()
    sample_aspect_ratio = str(stream.get("sample_aspect_ratio") or "1:1")
    display_aspect_ratio = str(stream.get("display_aspect_ratio") or "")
    is_hdr = color_transfer in {"smpte2084", "arib-std-b67"} or color_primaries == "bt2020" or any(
        str(item.get("side_data_type") or "").lower() in {"mastering display metadata", "content light level metadata"} for item in side_data
    )
    return _augment_virtual_metadata(video_path, {
        "codec_name": codec_name,
        "codec_profile": codec_profile,
        "width": width,
        "height": height,
        "display_width": display_width,
        "display_height": display_height,
        "rotation": rotation,
        "fps_float": fps_float,
        "fps": int(round(fps_float)) if fps_float > 0 else 0,
        "frame_count": frame_count,
        "duration": duration,
        "start_time": start_time,
        "sample_aspect_ratio": sample_aspect_ratio,
        "display_aspect_ratio": display_aspect_ratio,
        "color_transfer": color_transfer,
        "color_primaries": color_primaries,
        "color_space": color_space,
        "color_range": color_range,
        "needs_sar_fix": needs_sar_fix,
        "needs_tonemap": is_hdr,
    })


def _decode_virtual_media_frames(video_path, metadata, entry, start_frame, max_frames, target_fps, bridge):
    if entry.get("kind") == "image":
        if int(start_frame) > 0 or int(max_frames) <= 0:
            frames = torch.empty((0, metadata["display_height"], metadata["display_width"], 3), dtype=torch.uint8)
        else:
            image = np.asarray(entry["image"].convert("RGB"), dtype=np.uint8)[None]
            frames = torch.from_numpy(image)
    else:
        tensor = entry["tensor"]
        start_index = int(metadata.get("virtual_start_frame") or 0)
        end_index = metadata.get("virtual_end_frame")
        tensor = tensor[:, start_index:] if end_index is None else tensor[:, start_index:int(end_index) + 1]
        frame_count = int(tensor.shape[1])
        if target_fps is None or float(target_fps) <= 0:
            start_index = max(0, int(start_frame))
            frames = tensor[:, start_index:start_index + max(0, int(max_frames))].permute(1, 2, 3, 0)
        else:
            source_fps = metadata["fps"] if metadata["fps"] > 0 else max(1, int(round(metadata["fps_float"] or 0)))
            frame_nos = _resample_frame_indices(source_fps, frame_count, int(max_frames), float(target_fps), int(start_frame))
            frames = tensor[:, frame_nos].permute(1, 2, 3, 0) if len(frame_nos) > 0 else tensor[:, :0].permute(1, 2, 3, 0)
        if entry.get("hdr"):
            frames = frames.to(torch.float32).contiguous()
        else:
            frames = frames.add(1.0).mul(127.5).clamp_(0, 255).to(torch.uint8).contiguous()
    return frames if bridge == "torch" else frames.numpy()


def video_needs_corrected_decode(video_path):
    metadata = probe_video_stream_metadata(video_path)
    return metadata is not None and (metadata["needs_sar_fix"] or metadata["needs_tonemap"])


def _build_hdr_tonemap_filter(metadata):
    zscale_parts = ["t=linear", f"npl={_HDR_REFERENCE_WHITE_NITS}"]
    if transfer := _ZSCALE_TRANSFER_MAP.get(metadata["color_transfer"]):
        zscale_parts.append(f"tin={transfer}")
    if primaries := _ZSCALE_PRIMARIES_MAP.get(metadata["color_primaries"]):
        zscale_parts.append(f"pin={primaries}")
    if matrix := _ZSCALE_MATRIX_MAP.get(metadata["color_space"]):
        zscale_parts.append(f"min={matrix}")
    if color_range := _ZSCALE_RANGE_MAP.get(metadata.get("color_range")):
        zscale_parts.append(f"rin={color_range}")
    return ["zscale=" + ":".join(zscale_parts), "format=gbrpf32le", "tonemap=reinhard", "zscale=t=bt709:p=bt709:m=bt709:r=limited"]


def _build_hdr_linear_filter(metadata):
    zscale_parts = [f"npl={_HDR_REFERENCE_WHITE_NITS}", "t=linear", "p=709", "m=gbr", "r=full"]
    if transfer := _ZSCALE_TRANSFER_MAP.get(metadata["color_transfer"]):
        zscale_parts.append(f"tin={transfer}")
    if primaries := _ZSCALE_PRIMARIES_MAP.get(metadata["color_primaries"]):
        zscale_parts.append(f"pin={primaries}")
    if matrix := _ZSCALE_MATRIX_MAP.get(metadata["color_space"]):
        zscale_parts.append(f"min={matrix}")
    if color_range := _ZSCALE_RANGE_MAP.get(metadata.get("color_range")):
        zscale_parts.append(f"rin={color_range}")
    return ["zscale=" + ":".join(zscale_parts), "format=gbrpf32le"]


def _build_corrected_video_filter(metadata, target_fps=None, start_frame=0, end_frame=None, hdr_linear=False):
    filters = []
    if target_fps is not None and float(target_fps) > 0:
        filters.append(f"fps={float(target_fps):.12g}")
    if start_frame > 0 or end_frame is not None:
        trim_parts = [f"start_frame={int(start_frame)}"]
        if end_frame is not None:
            trim_parts.append(f"end_frame={int(end_frame)}")
        filters.append("trim=" + ":".join(trim_parts))
        filters.append("setpts=PTS-STARTPTS")
    if metadata["needs_sar_fix"]:
        filters += [f"scale={int(metadata['display_width'])}:{int(metadata['display_height'])}:flags=lanczos", "setsar=1"]
    if hdr_linear:
        filters += _build_hdr_linear_filter(metadata)
        return ",".join(filters)
    if metadata["needs_tonemap"]:
        filters += _build_hdr_tonemap_filter(metadata)
    return ",".join(filters)


def _read_exact(stream, size):
    buf = bytearray(size)
    view = memoryview(buf)
    read_pos = 0
    while read_pos < size:
        chunk = stream.read(size - read_pos)
        if not chunk:
            return None if read_pos == 0 else bytes(view[:read_pos])
        view[read_pos:read_pos + len(chunk)] = chunk
        read_pos += len(chunk)
    return buf


def _drain_stream(stream, chunks):
    while True:
        chunk = stream.read(65536)
        if not chunk:
            break
        chunks.append(chunk)


def _parse_first_showinfo_pts_time(stderr_text):
    for line in str(stderr_text or "").splitlines():
        pts_marker = " pts_time:"
        pts_index = line.find(pts_marker)
        if pts_index < 0:
            continue
        pts_text = line[pts_index + len(pts_marker):].split(None, 1)[0].strip()
        try:
            return float(pts_text)
        except (TypeError, ValueError):
            continue
    return None


def _decode_contiguous_video_frames_ffmpeg(video_path, start_frame, max_frames, bridge="torch", hdr_linear=False):
    metadata = probe_video_stream_metadata(video_path)
    if metadata is None:
        raise RuntimeError(f"Unable to probe video metadata for {video_path}")
    virtual_spec = parse_virtual_media_path(video_path)
    decode_path = os.fspath(metadata.get("source_path") or strip_virtual_media_suffix(video_path))
    ffmpeg_path = _resolve_media_binary("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg binary not found")
    start_frame = int(start_frame)
    max_frames = int(max_frames)
    # The caller's limit can greatly exceed the clip length (for example, postprocessing).
    if metadata["frame_count"] > 0 or metadata.get("virtual_end_frame") is not None:
        available_frames = max(0, int(metadata["frame_count"]) - max(0, start_frame))
        max_frames = min(max_frames, available_frames)
    if max_frames <= 0:
        empty_dtype = np.float32 if hdr_linear else np.uint8
        empty = np.empty((0, metadata["display_height"], metadata["display_width"], 3), dtype=empty_dtype)
        return torch.from_numpy(empty) if bridge == "torch" else empty
    actual_start = start_frame + int(metadata.get("virtual_start_frame") or 0)
    fps_float = float(metadata.get("fps_float") or metadata.get("fps") or 0.0)
    actual_end_exclusive = actual_start + max_frames
    filter_start_frame = 0
    filter_end_frame = None
    decode_seek_frame = actual_start
    local_search_enabled = virtual_spec is not None and fps_float > 0 and actual_start > 0
    requested_frames = max_frames
    if virtual_spec is not None:
        if local_search_enabled:
            decode_seek_frame = max(0, actual_start - _VIRTUAL_MEDIA_PRESEEK_FRAMES - _VIRTUAL_MEDIA_LOCAL_SEARCH_FRAMES)
            filter_start_frame = actual_start - decode_seek_frame
            requested_frames = filter_start_frame + max_frames + _VIRTUAL_MEDIA_LOCAL_SEARCH_FRAMES
        elif fps_float > 0 and actual_start > 0:
            decode_seek_frame = max(0, actual_start - _VIRTUAL_MEDIA_PRESEEK_FRAMES)
            filter_start_frame = actual_start - decode_seek_frame
            filter_end_frame = filter_start_frame + max_frames
        else:
            filter_start_frame = actual_start
            filter_end_frame = actual_end_exclusive
    video_filter = _build_corrected_video_filter(metadata, start_frame=filter_start_frame if virtual_spec is not None and not local_search_enabled else 0, end_frame=filter_end_frame if virtual_spec is not None and not local_search_enabled else None, hdr_linear=hdr_linear)
    if local_search_enabled:
        video_filter = "showinfo" if len(video_filter) == 0 else video_filter + ",showinfo"
    cmd = [ffmpeg_path, "-v", "info" if local_search_enabled else "error", "-nostdin", "-threads", "0"]
    if local_search_enabled:
        cmd += ["-copyts"]
    if fps_float > 0 and decode_seek_frame > 0:
        cmd += ["-ss", f"{float(metadata.get('start_time') or 0.0) + (decode_seek_frame / fps_float):.12g}"]
    cmd += ["-i", decode_path, "-an", "-sn"]
    if len(video_filter) > 0:
        cmd += ["-vf", video_filter]
    out_pix_fmt = "gbrpf32le" if hdr_linear else "rgb24"
    cmd += ["-fps_mode", "passthrough", "-frames:v", str(requested_frames), "-f", "rawvideo", "-pix_fmt", out_pix_fmt, "pipe:1"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**7, **child_kwargs())
    frame_bytes = metadata["display_width"] * metadata["display_height"] * 3 * (4 if hdr_linear else 1)
    frame_dtype = np.float32 if hdr_linear else np.uint8
    frames_shape = (requested_frames, 3, metadata["display_height"], metadata["display_width"]) if hdr_linear else (requested_frames, metadata["display_height"], metadata["display_width"], 3)
    frames = np.empty(frames_shape, dtype=frame_dtype)
    frame_count = 0
    stderr_chunks = []
    stderr_thread = None
    try:
        if process.stderr is not None:
            stderr_thread = threading.Thread(target=_drain_stream, args=(process.stderr, stderr_chunks), daemon=True)
            stderr_thread.start()
        while frame_count < requested_frames:
            raw_frame = _read_exact(process.stdout, frame_bytes)
            if raw_frame is None or len(raw_frame) < frame_bytes:
                break
            if hdr_linear:
                frames[frame_count] = np.frombuffer(raw_frame, dtype=np.float32).reshape(3, metadata["display_height"], metadata["display_width"])
            else:
                frames[frame_count] = np.frombuffer(raw_frame, dtype=np.uint8).reshape(metadata["display_height"], metadata["display_width"], 3)
            frame_count += 1
        return_code = process.wait()
        if stderr_thread is not None:
            stderr_thread.join()
        stderr = b"".join(stderr_chunks).decode("utf-8", errors="ignore").strip()
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
    if return_code != 0 and frame_count == 0:
        raise RuntimeError(f"ffmpeg decode failed for {video_path}: {stderr}")
    frames = frames[:frame_count]
    if local_search_enabled and frame_count > 0:
        first_pts_time = _parse_first_showinfo_pts_time(stderr)
        target_pts_time = float(metadata.get("start_time") or 0.0) + (actual_start / fps_float)
        local_start_frame = filter_start_frame if first_pts_time is None else max(0, int(round((target_pts_time - first_pts_time) * fps_float)))
        frames = frames[local_start_frame:local_start_frame + max_frames]
    if hdr_linear:
        frames = np.ascontiguousarray(frames[:, [2, 0, 1]].transpose(0, 2, 3, 1))
    return torch.from_numpy(frames) if bridge == "torch" else frames


def decode_video_frame_indices_ffmpeg(video_path, frame_indices, bridge="torch", hdr_linear=False):
    if torch.is_tensor(frame_indices):
        frame_indices = frame_indices.detach().cpu().tolist()
    frame_indices = [int(frame_index) for frame_index in frame_indices]
    metadata = probe_video_stream_metadata(video_path)
    if metadata is None:
        raise RuntimeError(f"Unable to probe video metadata for {video_path}")
    if len(frame_indices) == 0:
        empty_dtype = np.float32 if hdr_linear else np.uint8
        empty = np.empty((0, metadata["display_height"], metadata["display_width"], 3), dtype=empty_dtype)
        return torch.from_numpy(empty) if bridge == "torch" else empty
    start_frame = min(frame_indices)
    if (entry := get_virtual_media_entry(video_path)) is not None:
        decoded = _decode_virtual_media_frames(video_path, metadata, entry, start_frame, max(frame_indices) - start_frame + 1, None, "torch")
        frames = decoded[[frame_index - start_frame for frame_index in frame_indices]]
        return frames if bridge == "torch" else frames.numpy()
    unique_indices = sorted(set(frame_indices))
    span = max(unique_indices) - start_frame + 1
    if span <= len(unique_indices) * 3:
        decoded = decode_video_frames_ffmpeg(video_path, start_frame, span, target_fps=None, bridge="torch", hdr_linear=hdr_linear)
        frames = decoded[[frame_index - start_frame for frame_index in frame_indices]]
        return frames if bridge == "torch" else frames.numpy()
    ffmpeg_path = _resolve_media_binary("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg binary not found")
    decode_path = os.fspath(metadata.get("source_path") or strip_virtual_media_suffix(video_path))
    fps_float = float(metadata.get("fps_float") or metadata.get("fps") or 0.0)
    actual_start = start_frame + int(metadata.get("virtual_start_frame") or 0)
    rel_indices = [frame_index - start_frame for frame_index in unique_indices]
    select_expr = "+".join(f"eq(n\\,{frame_index})" for frame_index in rel_indices)
    video_filter = f"select={select_expr},setpts=N/FRAME_RATE/TB"
    corrected_filter = _build_corrected_video_filter(metadata, hdr_linear=hdr_linear)
    if len(corrected_filter) > 0:
        video_filter += "," + corrected_filter
    cmd = [ffmpeg_path, "-v", "error", "-nostdin", "-threads", "0"]
    if fps_float > 0 and actual_start > 0:
        cmd += ["-ss", f"{float(metadata.get('start_time') or 0.0) + (actual_start / fps_float):.12g}"]
    cmd += ["-i", decode_path, "-an", "-sn", "-vf", video_filter]
    out_pix_fmt = "gbrpf32le" if hdr_linear else "rgb24"
    cmd += ["-fps_mode", "passthrough", "-frames:v", str(len(unique_indices)), "-f", "rawvideo", "-pix_fmt", out_pix_fmt, "pipe:1"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**7, **child_kwargs())
    frame_bytes = metadata["display_width"] * metadata["display_height"] * 3 * (4 if hdr_linear else 1)
    frame_dtype = np.float32 if hdr_linear else np.uint8
    frames_shape = (len(unique_indices), 3, metadata["display_height"], metadata["display_width"]) if hdr_linear else (len(unique_indices), metadata["display_height"], metadata["display_width"], 3)
    frames_np = np.empty(frames_shape, dtype=frame_dtype)
    frame_count = 0
    stderr_chunks = []
    stderr_thread = None
    try:
        if process.stderr is not None:
            stderr_thread = threading.Thread(target=_drain_stream, args=(process.stderr, stderr_chunks), daemon=True)
            stderr_thread.start()
        while frame_count < len(unique_indices):
            raw_frame = _read_exact(process.stdout, frame_bytes)
            if raw_frame is None or len(raw_frame) < frame_bytes:
                break
            if hdr_linear:
                frames_np[frame_count] = np.frombuffer(raw_frame, dtype=np.float32).reshape(3, metadata["display_height"], metadata["display_width"])
            else:
                frames_np[frame_count] = np.frombuffer(raw_frame, dtype=np.uint8).reshape(metadata["display_height"], metadata["display_width"], 3)
            frame_count += 1
        return_code = process.wait()
        if stderr_thread is not None:
            stderr_thread.join()
        stderr = b"".join(stderr_chunks).decode("utf-8", errors="ignore").strip()
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            process.stderr.close()
    if return_code != 0 or frame_count != len(unique_indices):
        raise RuntimeError(f"ffmpeg indexed decode failed for {video_path}: {stderr}")
    if hdr_linear:
        frames_np = np.ascontiguousarray(frames_np[:, [2, 0, 1]].transpose(0, 2, 3, 1))
    frames = torch.from_numpy(frames_np)
    positions = {frame_index: pos for pos, frame_index in enumerate(unique_indices)}
    frames = frames[[positions[frame_index] for frame_index in frame_indices]]
    return frames if bridge == "torch" else frames.numpy()


def decode_video_frames_ffmpeg(video_path, start_frame, max_frames, target_fps=None, bridge="torch", hdr_linear=False):
    metadata = probe_video_stream_metadata(video_path)
    if metadata is None:
        raise RuntimeError(f"Unable to probe video metadata for {video_path}")
    if (entry := get_virtual_media_entry(video_path)) is not None:
        return _decode_virtual_media_frames(video_path, metadata, entry, start_frame, max_frames, target_fps, bridge)
    start_frame = int(start_frame)
    if metadata.get("virtual_end_frame") is not None and start_frame >= int(metadata["frame_count"]):
        empty_dtype = np.float32 if hdr_linear else np.uint8
        empty = np.empty((0, metadata["display_height"], metadata["display_width"], 3), dtype=empty_dtype)
        return torch.from_numpy(empty) if bridge == "torch" else empty
    if target_fps is None or float(target_fps) <= 0:
        return _decode_contiguous_video_frames_ffmpeg(video_path, start_frame, max_frames, bridge, hdr_linear=hdr_linear)
    source_fps = metadata["fps"] if metadata["fps"] > 0 else max(1, int(round(metadata["fps_float"] or 0)))
    frame_nos = _resample_frame_indices(source_fps, metadata["frame_count"], int(max_frames), float(target_fps), int(start_frame))
    if len(frame_nos) == 0:
        empty_dtype = np.float32 if hdr_linear else np.uint8
        empty = np.empty((0, metadata["display_height"], metadata["display_width"], 3), dtype=empty_dtype)
        return torch.from_numpy(empty) if bridge == "torch" else empty
    decode_start = frame_nos[0]
    decoded = _decode_contiguous_video_frames_ffmpeg(video_path, decode_start, frame_nos[-1] - decode_start + 1, bridge, hdr_linear=hdr_linear)
    index_list = [frame_no - decode_start for frame_no in frame_nos if frame_no - decode_start < decoded.shape[0]]
    if bridge == "torch":
        return decoded[index_list]
    return decoded[index_list]


def get_video_summary_extras(video_path):
    metadata = probe_video_stream_metadata(video_path)
    if metadata is None:
        return [], []
    values, labels = [], []
    if metadata["needs_sar_fix"]:
        values += [f"{metadata['width']}x{metadata['height']}", metadata["sample_aspect_ratio"]]
        labels += ["Stored Raster", "Pixel Aspect Ratio"]
        if len(metadata["display_aspect_ratio"]) > 0:
            values += [f"{metadata['display_aspect_ratio']} (square-pixel {metadata['display_width']}x{metadata['display_height']})"]
            labels += ["Display Aspect Ratio"]
    if metadata["needs_tonemap"]:
        hdr_parts = []
        if metadata["color_transfer"] == "smpte2084":
            hdr_parts += ["HDR PQ"]
        elif metadata["color_transfer"] == "arib-std-b67":
            hdr_parts += ["HDR HLG"]
        elif len(metadata["color_transfer"]) > 0:
            hdr_parts += [metadata["color_transfer"].upper()]
        if len(metadata["color_primaries"]) > 0:
            hdr_parts += [metadata["color_primaries"].upper()]
        if len(metadata["color_space"]) > 0 and metadata["color_space"] != metadata["color_primaries"]:
            hdr_parts += [metadata["color_space"].upper()]
        values += [" / ".join(hdr_parts) if len(hdr_parts) > 0 else "HDR source"]
        labels += ["Color"]
    return values, labels
