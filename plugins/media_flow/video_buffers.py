# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import os
import subprocess
import tempfile

import gradio as gr
import torch
from PIL import Image

from shared.utils.hdr import iter_hdr_gbrpf32_frames, tonemap_hdr_tensor_to_uint8
from shared.utils.utils import get_video_info_details
from shared.utils.video_decode import decode_video_frames_ffmpeg, resolve_media_binary
from shared.utils.virtual_media import build_virtual_media_path, clear_virtual_media_source, get_virtual_video, store_virtual_video
from shared.utils.subprocess_stdin import child_kwargs

PROCESS_FULL_VIDEO_VSOURCE = "process_full_video"
PROCESS_FULL_VIDEO_VFILE = "last_frames.mp4"


def frame_to_image(frame_tensor: torch.Tensor) -> Image.Image:
    return Image.fromarray(frame_tensor.permute(1, 2, 0).cpu().numpy())


def clear_process_full_video_source() -> None:
    clear_virtual_media_source(PROCESS_FULL_VIDEO_VSOURCE)


def build_process_full_video_source_path(*, hdr: bool = False) -> str:
    extras = {"vsource": PROCESS_FULL_VIDEO_VSOURCE}
    if hdr:
        extras["hdr"] = "1"
    return build_virtual_media_path(PROCESS_FULL_VIDEO_VFILE, extras=extras)


def set_process_full_video_overlap_buffer(overlap_tensor: torch.Tensor | None, fps_float: float, *, hdr: bool = False) -> None:
    if overlap_tensor is None or not torch.is_tensor(overlap_tensor) or int(overlap_tensor.shape[1]) <= 0:
        clear_virtual_media_source(PROCESS_FULL_VIDEO_VSOURCE)
        return
    store_virtual_video(PROCESS_FULL_VIDEO_VSOURCE, PROCESS_FULL_VIDEO_VFILE, overlap_tensor.contiguous(), fps_float, hdr=hdr)


def load_process_full_video_overlap_buffer(video_path: str, overlap_frames: int, actual_frame_count: int) -> torch.Tensor | None:
    if overlap_frames <= 0 or actual_frame_count <= 0 or not os.path.isfile(video_path):
        return None
    if overlap_frames > actual_frame_count:
        return None
    frames = decode_video_frames_ffmpeg(build_virtual_media_path(video_path, start_frame=-overlap_frames, end_frame=-1, extras={"frame_count": actual_frame_count}), 0, overlap_frames, target_fps=None, bridge="torch")
    return None if not torch.is_tensor(frames) or int(frames.shape[0]) <= 0 else frames.permute(3, 0, 1, 2).float().div_(127.5).sub_(1.0).contiguous()


def update_process_full_video_overlap_buffer(committed_tensor: torch.Tensor, overlap_frames: int, fps_float: float, *, hdr: bool = False) -> torch.Tensor | None:
    if overlap_frames <= 0 or not torch.is_tensor(committed_tensor) or int(committed_tensor.shape[1]) <= 0:
        clear_virtual_media_source(PROCESS_FULL_VIDEO_VSOURCE)
        return None
    overlap_tensor = committed_tensor.detach().cpu().to(torch.float32).contiguous()
    if not hdr:
        overlap_tensor = overlap_tensor.div_(127.5).sub_(1.0)
    previous_overlap = get_virtual_video(PROCESS_FULL_VIDEO_VSOURCE, PROCESS_FULL_VIDEO_VFILE)
    if previous_overlap is not None and int(previous_overlap.shape[1]) > 0:
        overlap_tensor = torch.cat([previous_overlap, overlap_tensor], dim=1)
    available_frames = int(overlap_tensor.shape[1])
    kept_frames = overlap_frames if overlap_frames < available_frames else available_frames
    overlap_tensor = overlap_tensor[:, -kept_frames:].contiguous()
    set_process_full_video_overlap_buffer(overlap_tensor, fps_float, hdr=hdr)
    return overlap_tensor


def load_process_full_video_hdr_overlap_buffer(output_path: str, overlap_frames: int, actual_frame_count: int) -> torch.Tensor | None:
    if overlap_frames <= 0 or actual_frame_count <= 0:
        return None
    if overlap_frames > actual_frame_count:
        return None
    frames = decode_video_frames_ffmpeg(build_virtual_media_path(output_path, start_frame=-overlap_frames, end_frame=-1, extras={"frame_count": actual_frame_count}), 0, overlap_frames, target_fps=None, bridge="torch", hdr_linear=True)
    return None if not torch.is_tensor(frames) or int(frames.shape[0]) <= 0 else frames.permute(3, 0, 1, 2).contiguous()


def _extract_exact_frame_image(video_path: str, frame_no: int) -> Image.Image:
    ffmpeg_path = resolve_media_binary("ffmpeg")
    if ffmpeg_path is None or not os.path.isfile(video_path) or int(frame_no) < 0:
        raise gr.Error(f"Unable to decode frame {frame_no} from {video_path}")
    with tempfile.TemporaryDirectory(prefix="wangp_tail_frame_") as temp_dir:
        output_path = os.path.join(temp_dir, "frame.png")
        command = [ffmpeg_path, "-v", "error", "-y", "-i", video_path, "-an", "-sn", "-vf", f"select=eq(n\\,{int(frame_no)})", "-frames:v", "1", output_path]
        result = subprocess.run(command, capture_output=True, text=True, **child_kwargs())
        if result.returncode != 0 or not os.path.isfile(output_path):
            raise gr.Error(f"Unable to decode frame {frame_no} from {video_path}")
        with Image.open(output_path) as frame_image:
            return frame_image.convert("RGB").copy()


def resolve_resume_last_frame(video_path: str, reported_frame_count: int) -> tuple[int, Image.Image | None, str]:
    if reported_frame_count <= 0:
        return 0, None, "existing output contains no decodable frame"
    for backtrack in (0, 1, 2, 4, 8, 16, 32, 64, 128, 256):
        frame_no = reported_frame_count - 1 - backtrack
        if frame_no < 0:
            continue
        try:
            frame_image = _extract_exact_frame_image(video_path, frame_no)
        except gr.Error:
            continue
        actual_frame_count = frame_no + 1
        message = "" if actual_frame_count == reported_frame_count else f"Adjusted continuation point to {actual_frame_count} decodable frame(s) from the existing output."
        return actual_frame_count, frame_image, message
    return 0, None, f"Unable to decode a valid tail frame from {video_path}"


def probe_existing_output_resolution(output_path: str) -> tuple[str, int, int]:
    metadata = get_video_info_details(output_path)
    width = int(metadata.get("display_width") or metadata.get("width") or 0)
    height = int(metadata.get("display_height") or metadata.get("height") or 0)
    if width <= 0 or height <= 0:
        raise gr.Error(f"Unable to read the resolution of existing output: {output_path}")
    return f"{width}x{height}", width, height


def get_video_tensor_resolution(video_tensor_uint8: torch.Tensor) -> tuple[int, int]:
    if not torch.is_tensor(video_tensor_uint8) or video_tensor_uint8.ndim != 4:
        raise gr.Error("WanGP API returned an invalid video tensor.")
    return int(video_tensor_uint8.shape[3]), int(video_tensor_uint8.shape[2])


def load_video_tensor_from_file(video_path: str) -> torch.Tensor:
    metadata = get_video_info_details(video_path)
    frame_count = int(metadata.get("frame_count") or 0)
    if frame_count <= 0:
        raise gr.Error(f"Unable to read the frame count of generated chunk: {video_path}")
    frames = decode_video_frames_ffmpeg(video_path, 0, frame_count, target_fps=None, bridge="torch")
    if frames.shape[0] <= 0:
        raise gr.Error(f"Unable to decode generated chunk: {video_path}")
    return frames.permute(3, 0, 1, 2).contiguous()


def write_video_chunk(process, video_tensor_uint8: torch.Tensor, *, start_frame: int, frame_count: int) -> torch.Tensor:
    if frame_count <= 0:
        raise RuntimeError("No frames available to write.")
    end_frame = start_frame + frame_count
    batch_frames = 8
    for batch_start in range(start_frame, end_frame, batch_frames):
        batch_end = batch_start + batch_frames
        if batch_end > end_frame:
            batch_end = end_frame
        batch = video_tensor_uint8[:, batch_start:batch_end].permute(1, 2, 3, 0).contiguous()
        try:
            process.stdin.write(batch.numpy().tobytes())
            process.stdin.flush()
        except BrokenPipeError as exc:
            stderr = process.stderr.read().decode("utf-8", errors="ignore").strip() if process.stderr is not None and process.poll() is not None else ""
            raise RuntimeError(stderr or "ffmpeg stopped receiving video frames while streaming a chunk") from exc
        if process.poll() not in (None, 0):
            stderr = process.stderr.read().decode("utf-8", errors="ignore").strip() if process.stderr is not None else ""
            raise RuntimeError(stderr or "ffmpeg exited while streaming a chunk")
    return video_tensor_uint8[:, start_frame + frame_count - 1]


def write_hdr_video_chunk(process, video_tensor_hdr: torch.Tensor, *, start_frame: int, frame_count: int) -> torch.Tensor:
    if frame_count <= 0:
        raise RuntimeError("No HDR frames available to write.")
    chunk = video_tensor_hdr[:, start_frame:start_frame + frame_count].detach().cpu()
    try:
        for frame_bytes in iter_hdr_gbrpf32_frames(chunk):
            process.stdin.write(frame_bytes)
        process.stdin.flush()
    except BrokenPipeError as exc:
        stderr = process.stderr.read().decode("utf-8", errors="ignore").strip() if process.stderr is not None and process.poll() is not None else ""
        raise RuntimeError(stderr or "ffmpeg stopped receiving HDR video frames while streaming a chunk") from exc
    if process.poll() not in (None, 0):
        stderr = process.stderr.read().decode("utf-8", errors="ignore").strip() if process.stderr is not None else ""
        raise RuntimeError(stderr or "ffmpeg exited while streaming HDR video")
    return tonemap_hdr_tensor_to_uint8(video_tensor_hdr[:, start_frame + frame_count - 1:start_frame + frame_count])[:, 0]


def compute_selected_frame_range(metadata: dict, start_seconds: float | None, end_seconds: float | None) -> tuple[int, int, float, int]:
    fps_float = float(metadata.get("fps_float") or metadata.get("fps") or 0.0)
    total_frames = int(metadata.get("frame_count") or 0)
    if fps_float <= 0 or total_frames <= 0:
        raise gr.Error("Unable to read the source video FPS or frame count.")
    start_frame = int(round(float(start_seconds or 0.0) * fps_float))
    if start_frame < 0:
        start_frame = 0
    if start_frame >= total_frames:
        start_frame = total_frames - 1
    if end_seconds in (None, ""):
        end_frame_exclusive = total_frames
    else:
        end_frame_exclusive = int(round(float(end_seconds) * fps_float))
        if end_frame_exclusive <= start_frame:
            end_frame_exclusive = start_frame + 1
        if end_frame_exclusive > total_frames:
            end_frame_exclusive = total_frames
    if end_frame_exclusive <= start_frame:
        raise gr.Error("End must be greater than Start.")
    return start_frame, end_frame_exclusive, fps_float, total_frames


def get_processing_fps(fps_float: float) -> float:
    processing_fps = int(round(float(fps_float)))
    return float(processing_fps if processing_fps > 0 else 1)


