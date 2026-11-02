# Copyright (c) 2025-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import subprocess
import tempfile, os
import ffmpeg
import struct
from typing import Any
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import cv2
import tempfile
import imageio
from shared import imageio_ffmpeg_patch  # Configure all imageio FFmpeg readers/writers before use.
import binascii
import torchvision
import torch
from PIL import Image
import os.path as osp
import json
import numpy as np
import soundfile as sf
import zlib
import re
from .hdr import hdr10_x265_params, hdr10_zscale_filter, iter_hdr_gbrpf32_frames, iter_video_chunks

from .video_decode import probe_video_stream_metadata, resolve_media_binary
from .video_codecs import SUPPORTED_VIDEO_CONTAINERS, get_imageio_codec_params, get_video_encode_args, validate_video_output_settings
from .virtual_media import get_virtual_media_entry, parse_virtual_media_path, strip_virtual_media_suffix

def _ffmpeg_binary():
    return resolve_media_binary("ffmpeg") or "ffmpeg"


def _ffprobe_binary():
    return resolve_media_binary("ffprobe") or "ffprobe"


def rand_name(length=8, suffix=''):
    name = binascii.b2a_hex(os.urandom(length)).decode('utf-8')
    if suffix:
        if not suffix.startswith('.'):
            suffix = '.' + suffix
        name += suffix
    return name


def _prepare_audio_array(audio_data):
    if torch.is_tensor(audio_data):
        audio_data = audio_data.detach().cpu().float().numpy()
    else:
        audio_data = np.asarray(audio_data, dtype=np.float32)
    if audio_data.ndim == 2 and audio_data.size == 0:
        # Empty tracks have zero frames; the other axis retains the channel count.
        return audio_data.reshape(0, max(audio_data.shape))
    if audio_data.ndim == 2 and audio_data.shape[0] <= 8 and audio_data.shape[1] > audio_data.shape[0]:
        audio_data = audio_data.T
    return audio_data


def write_wav_file(path, audio_data, sample_rate):
    audio_array = _prepare_audio_array(audio_data)
    sf.write(path, audio_array, int(sample_rate))
    return path


def resample_audio_array(audio_data, source_sample_rate, target_sample_rate):
    audio_array = np.asarray(audio_data, dtype=np.float32)
    source_sample_rate = int(source_sample_rate or 0)
    target_sample_rate = int(target_sample_rate or 0)
    if audio_array.size == 0 or source_sample_rate <= 0 or target_sample_rate <= 0 or source_sample_rate == target_sample_rate:
        return audio_array.astype(np.float32, copy=False)
    import torchaudio.functional as taF
    wave = torch.from_numpy(audio_array.T.copy() if audio_array.ndim == 2 else audio_array[None].copy()).to(dtype=torch.float32)
    resampled = taF.resample(wave, source_sample_rate, target_sample_rate).cpu().numpy()
    return (resampled.T if audio_array.ndim == 2 else resampled[0]).astype(np.float32, copy=False)


def append_sliding_window_audio(existing_audio_data, existing_audio_path, generated_audio, audio_sampling_rate, committed_audio_samples, existing_audio_sample_rate=None, prefix_start_samples=0, keep_existing=False, trim_ranges=(), video_fps=None, align_to_video=False, repair_samples=0):
    # keep_existing: a kept soundtrack also replaces the generated audio of the current window for as long as it lasts
    generated_audio = np.asarray(generated_audio, dtype=np.float32)
    if generated_audio.size == 0:
        return generated_audio
    prefix_sample_rate = int(existing_audio_sample_rate or audio_sampling_rate)
    if existing_audio_data is not None:
        prefix_audio = np.asarray(existing_audio_data, dtype=np.float32)
    elif existing_audio_path:
        prefix_audio, prefix_sample_rate = sf.read(os.fspath(existing_audio_path), dtype="float32", always_2d=generated_audio.ndim == 2)
    else:
        return generated_audio
    if prefix_sample_rate != int(audio_sampling_rate):
        prefix_audio = resample_audio_array(prefix_audio, prefix_sample_rate, audio_sampling_rate)
    prefix_start_samples, committed_audio_samples = max(0, int(prefix_start_samples)), max(0, int(committed_audio_samples))
    prefix_audio = prefix_audio[prefix_start_samples:]
    prefix_audio = trim_audio_ranges(prefix_audio, trim_ranges, video_fps, audio_sampling_rate)
    join_sample = committed_audio_samples - repair_samples
    prefix_audio = prefix_audio[:join_sample + (generated_audio.shape[0] if keep_existing else 0)]
    if align_to_video and not keep_existing and prefix_audio.shape[0] < join_sample:
        if repair_samples:
            raise ValueError("Audio overlap is too short to repair the previous window; increase the video overlap.")
        # A deliberate cut has no overlap to regenerate: retain its timestamp with silence.
        prefix_audio = np.pad(prefix_audio, [(0, join_sample - prefix_audio.shape[0])] + [(0, 0)] * (prefix_audio.ndim - 1))
    if prefix_audio.size == 0:
        return generated_audio
    if prefix_audio.ndim != generated_audio.ndim:
        prefix_audio = prefix_audio[:, None] if prefix_audio.ndim == 1 else prefix_audio
        generated_audio = generated_audio[:, None] if generated_audio.ndim == 1 else generated_audio
    if prefix_audio.ndim == 2 and prefix_audio.shape[1] != generated_audio.shape[1]:
        prefix_audio = np.repeat(prefix_audio[:, :1], generated_audio.shape[1], axis=1) if prefix_audio.shape[1] == 1 else prefix_audio[:, :generated_audio.shape[1]]
    return np.concatenate([prefix_audio, generated_audio[max(0, prefix_audio.shape[0] - join_sample):]], axis=0)


def source_audio_repair_ms(repair_ms, audio_path, codec, video_seconds):
    """Repair margin for continuing a source video: the model margin, unless the source audio runs past the continued frames (an audio
    tail saved with the video, or frames kept before its end) by at least that margin and two codec frames, a lossy encoder dropping a
    partial frame and coding the ~40 ms before the end of a stream less faithfully."""
    codec, info = str(codec or "").lower(), sf.info(audio_path)
    clear_ms = max(repair_ms, 0.0 if codec.startswith("pcm") or codec in ("flac", "alac") else 2000 * (1152 if codec == "mp3" else 1024) / info.samplerate)
    return 0 if info.frames / info.samplerate - video_seconds >= clear_ms / 1000 else repair_ms


def audio_overlap_repair_seconds(overlap_frames, video_fps, repair_ms, overlap_audio=None, sample_rate=0):
    overlap_seconds = overlap_frames / video_fps
    shortfall = max(0.0, overlap_seconds - len(overlap_audio) / sample_rate) if overlap_audio is not None else 0.0
    repair_seconds = repair_ms / 1000 + shortfall
    if repair_seconds > overlap_seconds + 1e-9:
        raise ValueError(f"Audio continuation needs more than {overlap_frames} overlap frames at {video_fps:g} FPS; increase the video overlap (repair requires {repair_seconds * 1000:.1f} ms).")
    return min(repair_seconds, overlap_seconds)


def create_silent_wav_file(output_dir=None, duration_seconds=0.0, sample_rate=16000, prefix="null_audio_"):
    sample_rate = int(sample_rate)
    num_samples = max(1, int(np.ceil(float(duration_seconds) * sample_rate)))
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".wav", dir=output_dir)
    os.close(fd)
    return write_wav_file(path, np.zeros(num_samples, dtype=np.float32), sample_rate)


def truncate_audio(generated_audio, trim_video_frames_beginning, trim_video_frames_end, video_fps, audio_sampling_rate):
    samples_per_frame = audio_sampling_rate / video_fps
    start = int(trim_video_frames_beginning * samples_per_frame)
    end = generated_audio.shape[0] - int(trim_video_frames_end * samples_per_frame)
    return generated_audio[start:end if end > 0 else None]


def shift_audio_trim_ranges(trim_ranges, offset_frames):
    shifted = []
    for start_frame, frame_count in trim_ranges:
        start_frame -= offset_frames
        end_frame = start_frame + frame_count
        if end_frame <= 0:
            continue
        start_frame = max(0, start_frame)
        shifted.append((start_frame, end_frame - start_frame))
    return shifted


def trim_audio_ranges(audio_data, trim_ranges, video_fps, audio_sampling_rate):
    if len(trim_ranges) == 0:
        return audio_data
    audio_data = np.asarray(audio_data, dtype=np.float32)
    samples_per_frame = audio_sampling_rate / video_fps
    pieces = []
    cursor = 0
    total_samples = audio_data.shape[0]
    removed_frames = 0
    for start_frame, frame_count in sorted(trim_ranges):
        source_start_frame = start_frame + removed_frames
        source_end_frame = source_start_frame + frame_count
        start = max(cursor, min(total_samples, int(source_start_frame * samples_per_frame)))
        end = max(start, min(total_samples, int(source_end_frame * samples_per_frame)))
        if start > cursor:
            pieces.append(audio_data[cursor:start])
        cursor = end
        removed_frames += frame_count
    if cursor < total_samples:
        pieces.append(audio_data[cursor:])
    return np.concatenate(pieces, axis=0) if len(pieces) > 0 else audio_data[:0]


def trim_audio_file_ranges(audio_path, trim_ranges, video_fps, output_path):
    if len(trim_ranges) == 0:
        return audio_path
    audio_data, audio_sampling_rate = sf.read(audio_path, dtype="float32")
    trimmed_audio = trim_audio_ranges(audio_data, trim_ranges, video_fps, audio_sampling_rate)
    return write_wav_file(output_path, trimmed_audio, audio_sampling_rate)


def slice_audio_window(audio_path, start_frame, num_frames, fps, output_dir=None, suffix="", pad_head=True, pad_tail=True):
    start_sec = float(start_frame) / float(fps)
    duration_sec = float(num_frames) / float(fps)

    with sf.SoundFile(audio_path) as audio_file:
        sample_rate = audio_file.samplerate
        channels = audio_file.channels
        total_frames = len(audio_file)
        start_sample = int(round(start_sec * sample_rate))
        pad_start = 0
        if start_sample < 0 and pad_head:
            pad_start = -start_sample
        if start_sample < 0:
            start_sample = 0
        frames_to_read = int(round(duration_sec * sample_rate))
        if start_sample > total_frames:
            data = np.zeros((0, channels), dtype=np.float32)
        else:
            audio_file.seek(min(start_sample, total_frames))
            data = audio_file.read(frames_to_read, dtype="float32", always_2d=True)

    if pad_head and pad_start > 0:
        data = np.concatenate([np.zeros((pad_start, channels), dtype=np.float32), data], axis=0)
    if pad_tail:
        target_frames = (pad_start if pad_head else 0) + frames_to_read
        if data.shape[0] < target_frames:
            pad_end = target_frames - data.shape[0]
            data = np.concatenate([data, np.zeros((pad_end, channels), dtype=np.float32)], axis=0)
    return data, sample_rate


def get_audio_file_sample_rate(audio_path):
    probe = ffmpeg.probe(os.fspath(audio_path), cmd=_ffprobe_binary())
    audio_stream = next((stream for stream in probe["streams"] if stream.get("codec_type") == "audio"), None)
    if audio_stream is None or not audio_stream.get("sample_rate"):
        raise ValueError(f"Unable to read audio sample rate from {audio_path}")
    return int(audio_stream["sample_rate"])


def get_audio_file_channels(audio_path):
    probe = ffmpeg.probe(os.fspath(audio_path), cmd=_ffprobe_binary())
    audio_stream = next((stream for stream in probe["streams"] if stream.get("codec_type") == "audio"), None)
    if audio_stream is None or not audio_stream.get("channels"):
        raise ValueError(f"Unable to read audio channel count from {audio_path}")
    return int(audio_stream["channels"])


def get_media_duration_and_audio_layouts(media_path):
    """Read file duration and each audio track's channel layout, without decoding."""
    probe = ffmpeg.probe(os.fspath(media_path), cmd=_ffprobe_binary())
    duration = probe.get("format", {}).get("duration")
    duration = float(duration) if duration not in (None, "", "N/A") else None
    layouts = []
    for stream in probe.get("streams", []):
        if stream.get("codec_type") != "audio":
            continue
        channels = int(stream.get("channels", 0))
        layout = stream.get("channel_layout", "")
        if channels in (1, 2):
            label = "Mono" if channels == 1 else "Stereo"
        elif channels:
            label = f"{layout} ({channels} channels)" if layout and layout != "unknown" else f"{channels} channels"
        else:
            label = "Unknown channel layout"
        layouts.append(label)
    return duration, layouts


def resolve_mux_audio_sampling_rate(default_rate, source_audio_metadata=None, audio_paths=None):
    sample_rates = [int(default_rate)]
    for meta in source_audio_metadata or []:
        sample_rate = int(meta.get("sample_rate", 0) or 0)
        if sample_rate > 0:
            sample_rates.append(sample_rate)
    for audio_path in audio_paths or []:
        if audio_path:
            sample_rates.append(get_audio_file_sample_rate(audio_path))
    return max(sample_rates)


def _compute_active_abs_amplitude(audio_data, active_mask=None):
    audio_data = np.asarray(audio_data, dtype=np.float32)
    if active_mask is not None:
        active_mask = np.asarray(active_mask, dtype=np.float32).reshape(-1) > 0.5
        if audio_data.ndim == 1:
            active_mask = active_mask[:audio_data.shape[0]]
            audio_data = audio_data[:active_mask.shape[0]][active_mask]
        else:
            active_mask = active_mask[:audio_data.shape[0]]
            audio_data = audio_data[:active_mask.shape[0]][active_mask]
    abs_audio = np.abs(audio_data).reshape(-1)
    if abs_audio.size == 0:
        return 0.0, 0.0
    avg_abs = float(abs_audio.mean())
    if avg_abs <= 0.0:
        return 0.0, 0.0
    threshold = 0.1 * avg_abs
    active_mask = abs_audio > threshold
    active_avg_abs = float(abs_audio[active_mask].mean()) if np.any(active_mask) else avg_abs
    return avg_abs, active_avg_abs


def normalize_audio_volumes(audios):
    """Scale each audio so its active loudness matches the average active loudness of all of them."""
    audios = [np.asarray(audio, dtype=np.float32) for audio in audios]
    levels = [_compute_active_abs_amplitude(audio)[1] for audio in audios]
    target = sum(levels) / len(levels)
    return [np.clip(audio * float(target / level if level > 1e-8 else 1.0), -1.0, 1.0) for audio, level in zip(audios, levels)]


def normalize_audio_volumes_to_temp_files(audio_paths, output_dir=None, prefix="audio_norm_", max_duration_seconds=None):
    audios = [sf.read(os.fspath(path), frames=-1 if max_duration_seconds is None else round(float(max_duration_seconds) * sf.info(os.fspath(path)).samplerate), dtype="float32", always_2d=False) for path in audio_paths]
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    outputs = []
    for index, (normalized, (_, sample_rate)) in enumerate(zip(normalize_audio_volumes([audio for audio, _ in audios]), audios), 1):
        fd, output = tempfile.mkstemp(prefix=f"{prefix}{index}_", suffix=".wav", dir=output_dir)
        os.close(fd)
        sf.write(output, normalized, int(sample_rate))
        outputs.append(output)
    return outputs


def _get_audio_codec_settings(codec_key):
    if not codec_key:
        codec_key = "wav"
    codec_key = str(codec_key).lower()
    if codec_key == "mp3":
        codec_key = "mp3_192"
    settings = {
        "wav": {"ext": "wav", "format": "wav"},
        "flac": {"ext": "flac", "format": "flac"},
        "mp3_128": {"ext": "mp3", "format": "mp3", "bitrate": "128k"},
        "mp3_192": {"ext": "mp3", "format": "mp3", "bitrate": "192k"},
        "mp3_320": {"ext": "mp3", "format": "mp3", "bitrate": "320k"},
    }
    return settings.get(codec_key, settings["wav"])


def get_mp4_audio_codec_settings(codec_key):
    codec_key = "aac_128" if not codec_key else str(codec_key).lower()
    settings = {
        "aac_128": {"codec": "aac", "bitrate": "128k", "ext": ".aac"},
        "aac_192": {"codec": "aac", "bitrate": "192k", "ext": ".aac"},
        "aac_256": {"codec": "aac", "bitrate": "256k", "ext": ".aac"},
        "aac_320": {"codec": "aac", "bitrate": "320k", "ext": ".aac"},
        "alac": {"codec": "alac", "bitrate": None, "ext": ".m4a"},
        "flac": {"codec": "flac", "bitrate": None, "ext": ".flac", "sample_fmt": "s16"},
    }
    return settings.get(codec_key, settings["aac_128"])


def get_video_audio_extra_args(codec_key, output_path=None):
    if str(codec_key or "").lower() != "flac":
        return []
    if output_path is not None and osp.splitext(os.fspath(output_path))[1].lower() == ".mov":
        raise ValueError("FLAC soundtracks require MP4 or MKV; choose AAC or ALAC for MOV.")
    return ["-sample_fmt", "s16", "-strict", "-2"]


def _infer_video_dimensions(tensor):
    if torch.is_tensor(tensor):
        if tensor.ndim == 5:
            return int(tensor.shape[-1]), int(tensor.shape[-2])
        if tensor.ndim == 4:
            if tensor.shape[-1] in (1, 3, 4):
                return int(tensor.shape[2]), int(tensor.shape[1])
            return int(tensor.shape[-1]), int(tensor.shape[-2])
    if isinstance(tensor, (list, tuple)):
        for chunk in tensor:
            dims = _infer_video_dimensions(chunk)
            if dims is not None:
                return dims
    return None


def _validate_video_save_settings(codec_type, container, tensor):
    dims = _infer_video_dimensions(tensor)
    width = height = None
    if dims is not None:
        width, height = dims
    error = validate_video_output_settings(codec_type, container, width=width, height=height, allowed_containers=SUPPORTED_VIDEO_CONTAINERS)
    if error is not None:
        raise RuntimeError(error)


def _video_frame_tensor_to_uint8(frame, normalize=True, value_range=(-1, 1)):
    if frame.dtype == torch.uint8:
        return frame.detach().cpu().contiguous().numpy()
    frame = frame.detach().to(dtype=torch.float32, copy=True)
    if normalize:
        value_min, value_max = min(value_range), max(value_range)
        frame.clamp_(value_min, value_max).sub_(value_min).div_(value_max - value_min)
    return frame.mul_(255.0).round_().clamp_(0, 255).to(torch.uint8).cpu().contiguous().numpy()


def _crf_from_video_codec(codec_key: str | None, default: str = "18") -> str:
    codec_key = str(codec_key or "").strip().lower()
    if re.fullmatch(r"\d+", codec_key):
        return codec_key
    match = re.search(r"_(\d+)$", codec_key)
    return match.group(1) if match is not None else str(default)


def get_hdr_video_encode_args(codec_key: str | None, container: str | None) -> list[str]:
    crf = _crf_from_video_codec(codec_key, default="18")
    return [
        "-vf", hdr10_zscale_filter(),
        "-c:v", "libx265",
        "-preset", "medium",
        "-crf", crf,
        "-pix_fmt", "yuv420p10le",
        "-tag:v", "hvc1",
        "-color_primaries", "bt2020",
        "-color_trc", "smpte2084",
        "-colorspace", "bt2020nc",
        "-x265-params", hdr10_x265_params(),
    ]


def get_audio_codec_extension(codec_key):
    return _get_audio_codec_settings(codec_key)["ext"]


def _run_ffmpeg_encode(input_path, output_path, codec, bitrate=None, sample_rate=None, drop_video=False):
    cmd = [_ffmpeg_binary(), "-y", "-v", "error", "-i", input_path]
    if drop_video:
        cmd.append("-vn")
    cmd += ["-c:a", codec]
    if bitrate:
        cmd += ["-b:a", bitrate]
    if sample_rate:
        cmd += ["-ar", str(int(sample_rate))]
    cmd.append(output_path)
    subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")


def save_audio_file(path, audio_data, sample_rate, codec_key="wav"):
    settings = _get_audio_codec_settings(codec_key)
    ext = settings["ext"]
    if not path.lower().endswith(f".{ext}"):
        path = osp.splitext(path)[0] + f".{ext}"
    if settings["format"] == "wav":
        return write_wav_file(path, audio_data, sample_rate)
    if settings["format"] == "flac":
        sf.write(path, _prepare_audio_array(audio_data), int(sample_rate), format="FLAC", subtype="PCM_16")
        return path
    fd, tmp_path = tempfile.mkstemp(suffix=".wav", prefix="audio_")
    os.close(fd)
    try:
        write_wav_file(tmp_path, audio_data, sample_rate)
        _run_ffmpeg_encode(tmp_path, path, "libmp3lame", bitrate=settings.get("bitrate"), sample_rate=sample_rate)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
    return path


def _resolve_virtual_audio_segment(video_path: str) -> tuple[str, dict[str, Any], int]:
    if isinstance(video_path, Image.Image):
        return "", {}, 0
    if get_virtual_media_entry(video_path) is not None:
        return "", {}, 0
    spec = parse_virtual_media_path(video_path)
    source_path = os.fspath(strip_virtual_media_suffix(video_path))
    time_args: dict[str, Any] = {}
    if spec is None:
        return source_path, time_args, 0
    metadata = probe_video_stream_metadata(video_path)
    if metadata is not None and metadata.get("virtual_end_frame") is not None:
        start_frame = int(metadata.get("virtual_start_frame") or 0)
        end_frame = int(metadata.get("virtual_end_frame") or start_frame)
        fps_float = float(metadata.get("fps_float") or metadata.get("fps") or 0.0)
        if fps_float > 0:
            time_args["ss"] = max(0.0, start_frame / fps_float)
            time_args["to"] = max(time_args["ss"], (end_frame + 1) / fps_float)
    audio_track_no = 1 if spec.audio_track_no is None else max(1, int(spec.audio_track_no))
    return source_path, time_args, audio_track_no - 1


def extract_audio_track_to_wav(video_path, output_path):
    if not video_path:
        return None
    if isinstance(video_path, Image.Image):
        return None
    video_path = os.fspath(video_path)
    source_path, time_args, audio_track_index = _resolve_virtual_audio_segment(video_path)
    if len(source_path) == 0:
        return None
    import ffmpeg
    try:
        output_kwargs = {"map": f"0:a:{audio_track_index}", "acodec": "pcm_s16le"}
        ffmpeg.input(source_path, **time_args).output(output_path, **output_kwargs).overwrite_output().run(cmd=_ffmpeg_binary(), quiet=True)
    except ffmpeg.Error as err:
        stderr = getattr(err, "stderr", b"")
        if isinstance(stderr, (bytes, bytearray)):
            stderr = stderr.decode("utf-8", errors="ignore")
        stderr = (stderr or str(err)).strip()
        raise RuntimeError(f"ffmpeg audio extract failed for {source_path} -> {output_path}: {stderr}") from err
    return output_path



def extract_audio_tracks(source_video, verbose=False, query_only=False, codec_key="aac_128", temp_format=None):
    """
    Extract all audio tracks from a source video into temporary audio files.

    Returns:
        Tuple:
          - List of temp file paths for extracted audio tracks
          - List of corresponding metadata dicts:
              {'codec', 'sample_rate', 'channels', 'duration', 'language'}
              where 'duration' is set to container duration (for consistency).
    """
    if isinstance(source_video, Image.Image):
        return 0 if query_only else ([], [])
    source_path, time_args, selected_track_index = _resolve_virtual_audio_segment(source_video)
    if len(source_path) == 0:
        return 0 if query_only else ([], [])
    if not os.path.exists(source_path):
        msg = f"ffprobe skipped; file not found: {source_video}"
        if verbose:
            print(msg)
        raise FileNotFoundError(msg)

    try:
        probe = ffmpeg.probe(source_path, cmd=_ffprobe_binary())
    except ffmpeg.Error as err:
        stderr = getattr(err, 'stderr', b'')
        if isinstance(stderr, (bytes, bytearray)):
            stderr = stderr.decode('utf-8', errors='ignore')
        stderr = (stderr or str(err)).strip()
        message = f"ffprobe failed for {source_path}: {stderr}"
        if verbose:
            print(message)
        raise RuntimeError(message) from err
    audio_streams = [s for s in probe['streams'] if s['codec_type'] == 'audio']
    container_duration = float(probe['format'].get('duration', 0.0))
    if selected_track_index is not None:
        audio_streams = [audio_streams[selected_track_index]] if 0 <= selected_track_index < len(audio_streams) else []

    if not audio_streams:
        if query_only: return 0
        if verbose: print(f"No audio track found in {source_video}")
        return [], []

    if query_only:
        return len(audio_streams)

    if verbose:
        print(f"Found {len(audio_streams)} audio track(s), container duration = {container_duration:.3f}s")

    file_paths = []
    metadata = []
    if temp_format == "wav":
        audio_settings = {"codec": "pcm_s16le", "bitrate": None, "ext": ".wav"}
    else:
        audio_settings = get_mp4_audio_codec_settings(codec_key)

    for i, stream in enumerate(audio_streams):
        fd, temp_path = tempfile.mkstemp(suffix=f'_track{i}{audio_settings["ext"]}', prefix='audio_')
        os.close(fd)

        file_paths.append(temp_path)
        metadata.append({
            'codec': stream.get('codec_name'),
            'sample_rate': int(stream.get('sample_rate', 0)),
            'channels': int(stream.get('channels', 0)),
            'duration': container_duration,
            'language': stream.get('tags', {}).get('language', None)
        })

        stream_index = i if selected_track_index is None else selected_track_index
        output_kwargs = {f'map': f'0:a:{stream_index}', 'acodec': audio_settings["codec"]}
        if audio_settings["bitrate"]:
            output_kwargs['b:a'] = audio_settings["bitrate"]
        if audio_settings.get("sample_fmt"):
            output_kwargs["sample_fmt"] = audio_settings["sample_fmt"]
        ffmpeg.input(source_path, **time_args).output(temp_path, **output_kwargs).overwrite_output().run(cmd=_ffmpeg_binary(), quiet=not verbose)

    return file_paths, metadata



def combine_and_concatenate_video_with_audio_tracks(
    save_path_tmp, video_path,
    source_audio_tracks, new_audio_tracks,
    source_audio_duration, audio_sampling_rate,
    new_audio_from_start=False,
    source_audio_metadata=None,
    audio_codec_key="aac_128",
    verbose = False,
    pad_audio_to_video=False,
    audio_tail_seconds=0.0,
):
    """audio_tail_seconds keeps that much audio past the last video frame (instead of cutting at the video end), so the end of the
    video's own audio stays clear of the encoder's end effects."""
    extra_audio_args = get_video_audio_extra_args(audio_codec_key, save_path_tmp)
    audio_settings = get_mp4_audio_codec_settings(audio_codec_key)
    audio_codec = audio_settings["codec"]
    audio_bitrate = audio_settings["bitrate"]
    inputs, filters, maps, idx = ['-i', video_path], [], ['-map', '0:v'], 1
    metadata_args = []
    sources = source_audio_tracks or []
    news = new_audio_tracks or []

    duplicate_source = len(sources) == 1 and len(news) > 1
    N = len(news) if source_audio_duration == 0 else max(len(sources), len(news)) or 1

    for i in range(N):
        s = (sources[i] if i < len(sources)
             else sources[0] if duplicate_source else None)
        n = news[i] if len(news) == N else (news[0] if news else None)
        meta = source_audio_metadata[i] if source_audio_metadata and i < len(source_audio_metadata) else {}
        source_channels = int(meta.get('channels', 0) or 0) if s else 0
        if s and source_channels == 0:
            source_channels = get_audio_file_channels(s)
        new_channels = get_audio_file_channels(n) if n else 0
        channel_layout = 'stereo' if max(source_channels, new_channels) >= 2 else 'mono'

        if source_audio_duration == 0:
            if n:
                inputs += ['-i', n]
                filters.append(f'[{idx}:a]aformat=channel_layouts={channel_layout},apad=pad_dur=100[aout{i}]')
                idx += 1
            else:
                filters.append(f'anullsrc=r={audio_sampling_rate}:cl={channel_layout},apad=pad_dur=100[aout{i}]')
        else:
            if s:
                inputs += ['-i', s]
                needs_filter = (
                    meta.get('codec') != audio_codec or
                    meta.get('sample_rate') != audio_sampling_rate or
                    source_channels != (2 if channel_layout == 'stereo' else 1) or
                    meta.get('duration', 0) < source_audio_duration
                )
                if needs_filter:
                    filters.append(
                        f'[{idx}:a]aresample={audio_sampling_rate},aformat=channel_layouts={channel_layout},'
                        f'apad=pad_dur={source_audio_duration},atrim=0:{source_audio_duration},asetpts=PTS-STARTPTS[s{i}]')
                else:
                    filters.append(
                        f'[{idx}:a]apad=pad_dur={source_audio_duration},atrim=0:{source_audio_duration},asetpts=PTS-STARTPTS[s{i}]')
                if lang := meta.get('language'):
                    metadata_args += ['-metadata:s:a:' + str(i), f'language={lang}']
                idx += 1
            else:
                filters.append(
                    f'anullsrc=r={audio_sampling_rate}:cl={channel_layout},atrim=0:{source_audio_duration},asetpts=PTS-STARTPTS[s{i}]')

            if n:
                inputs += ['-i', n]
                start = '0' if new_audio_from_start else source_audio_duration
                filters.append(
                    f'[{idx}:a]aresample={audio_sampling_rate},aformat=channel_layouts={channel_layout},'
                    f'atrim=start={start},asetpts=PTS-STARTPTS[n{i}]')
                padding = ',apad=pad_dur=100' if pad_audio_to_video else ''
                filters.append(f'[s{i}][n{i}]concat=n=2:v=0:a=1{padding}[aout{i}]')
                idx += 1
            else:
                filters.append(f'[s{i}]apad=pad_dur=100[aout{i}]')

        maps += ['-map', f'[aout{i}]']

    duration_args = ['-shortest']
    if audio_tail_seconds > 0:
        video_duration = float(next(s for s in ffmpeg.probe(video_path, cmd=_ffprobe_binary())['streams'] if s['codec_type'] == 'video')['duration'])
        duration_args = ['-t', str(video_duration + audio_tail_seconds)]
    cmd = [_ffmpeg_binary(), '-y', *inputs,
           '-filter_complex', ';'.join(filters),  # ✅ Only change made
           *maps, *metadata_args,
           '-c:v', 'copy',
           '-c:a', audio_codec, *extra_audio_args,
           *(['-b:a', audio_bitrate] if audio_bitrate else []),
           '-ar', str(audio_sampling_rate),
           *duration_args, save_path_tmp]

    if verbose:
        print(f"ffmpeg command: {cmd}")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except subprocess.CalledProcessError as e:
        raise Exception(f"FFmpeg error: {e.stderr}")


def subtitle_stream_count(path):
    if get_virtual_media_entry(path) is not None:
        return 0
    return sum(stream['codec_type'] == 'subtitle' for stream in ffmpeg.probe(strip_virtual_media_suffix(path), cmd=_ffprobe_binary())['streams'])


def combine_video_with_audio_tracks(target_video, audio_tracks, output_video,
                                     audio_metadata=None, audio_codec_key="aac_128", verbose=False, video_duration=None, subtitle_source=None):
    """Mux audio tracks, and the subtitle tracks of subtitle_source when given, with the video stream of target_video."""
    extra_audio_args = get_video_audio_extra_args(audio_codec_key, output_video)
    if not audio_tracks and subtitle_source is None:
        if verbose: print("No audio tracks to combine."); return False

    if video_duration is None:
        video_duration = float(next(s for s in ffmpeg.probe(target_video, cmd=_ffprobe_binary())['streams']
                               if s['codec_type'] == 'video')['duration'])
    dur = video_duration
    if verbose: print(f"Video duration: {dur:.3f}s")

    cmd = [_ffmpeg_binary(), '-y', '-i', target_video]
    for path in audio_tracks:
        cmd += ['-i', path]
    if subtitle_source is not None:
        source_path, time_args, _ = _resolve_virtual_audio_segment(subtitle_source)
        for key, value in time_args.items():
            cmd += [f'-{key}', str(value)]
        cmd += ['-i', source_path]
        if time_args:
            # Seeking leaves subtitle packets before zero and across the chunk boundaries.
            # Drop packets outside the chunk and clip overlapping cues before muxing.
            subtitle_duration = min(float(dur), time_args['to'] - time_args['ss'])
            cmd += ['-bsf:s', f"noise=amount=0:drop='lte(pts+duration,0)+gte(pts*tb,{subtitle_duration})',"
                    f"setts=ts='max(TS,0)':duration='min(PTS+DURATION,{subtitle_duration}/TB)-max(PTS,0)'"]

    cmd += ['-map', '0:v']
    for i in range(len(audio_tracks)):
        cmd += ['-map', f'{i+1}:a']
    if subtitle_source is not None:
        # Text subtitles keep their language and title; MP4/MOV store them as mov_text, Matroska as SRT.
        cmd += ['-map', f'{len(audio_tracks) + 1}:s', '-c:s', 'mov_text' if os.path.splitext(output_video)[1].lower() in ('.mp4', '.mov', '.m4v') else 'srt']

    for i, meta in enumerate(audio_metadata or []):
        if (lang := meta.get('language')):
            cmd += ['-metadata:s:a:' + str(i), f'language={lang}']

    audio_settings = get_mp4_audio_codec_settings(audio_codec_key)
    cmd += ['-c:v', 'copy', '-c:a', audio_settings["codec"], *extra_audio_args]
    if audio_settings["bitrate"]:
        cmd += ['-b:a', audio_settings["bitrate"]]
    cmd += ['-t', str(dur), output_video]

    result = subprocess.run(cmd, capture_output=not verbose, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise Exception(f"FFmpeg error:\n{result.stderr}")
    if verbose:
        print(f"Created {output_video} with {len(audio_tracks)} audio track(s)")
    return True


def subtitled_video_extension(source_video):
    """The container of a copy of source_video with an added text subtitle track: MP4 and MOV keep theirs (mov_text), others become MKV."""
    extension = os.path.splitext(source_video)[1].lower()
    return ".mov" if extension == ".mov" else ".mp4" if extension in (".mp4", ".m4v") else ".mkv"


def add_subtitle_track(source_video, subtitles, output_video, title=None):
    """Copy the video and audio streams of source_video (no re-encoding) into output_video with the SRT file subtitles as its subtitle track,
    shown by default (the source's own subtitle tracks are not kept)."""
    cmd = [_ffmpeg_binary(), '-y', '-v', 'error', '-i', source_video, '-i', subtitles, '-map', '0:V', '-map', '0:a?', '-map', '1:s', '-c:v', 'copy', '-c:a', 'copy',
           '-c:s', 'srt' if output_video.lower().endswith('.mkv') else 'mov_text', '-disposition:s:0', 'default', *(['-metadata:s:s:0', f'title={title}'] if title else []), output_video]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise Exception(f"FFmpeg error:\n{result.stderr}")


def cleanup_temp_audio_files(audio_tracks, verbose=False):
    """
    Clean up temporary audio files.
    
    Args:
        audio_tracks: List of audio file paths to delete
        verbose: Enable verbose output (default: False)
        
    Returns:
        Number of files successfully deleted
    """
    deleted_count = 0
    
    for audio_path in audio_tracks:
        try:
            if os.path.exists(audio_path):
                os.unlink(audio_path)
                deleted_count += 1
                if verbose:
                    print(f"Cleaned up {audio_path}")
        except PermissionError:
            print(f"Warning: Could not delete {audio_path} (file may be in use)")
        except Exception as e:
            print(f"Warning: Error deleting {audio_path}: {e}")
    
    if verbose and deleted_count > 0:
        print(f"Successfully deleted {deleted_count} temporary audio file(s)")
    
    return deleted_count


def save_video(tensor,
                save_file=None,
                fps=30,
                codec_type='libx264_8',
                container='mp4',
                nrow=8,
                normalize=True,
                value_range=(-1, 1),
                retry=5):
    """Save tensor as video with configurable codec and container options."""
        
    if torch.is_tensor(tensor) and len(tensor.shape) == 4:
        tensor = tensor.unsqueeze(0)

    _validate_video_save_settings(codec_type, container, tensor)
        
    suffix = f'.{container}'
    cache_file = osp.join('/tmp', rand_name(suffix=suffix)) if save_file is None else save_file
    if not cache_file.endswith(suffix):
        cache_file = osp.splitext(cache_file)[0] + suffix
    
    # Configure codec parameters
    codec_params = _get_codec_params(codec_type, container)
    
    # Process and save
    error = None
    for _ in range(retry):
        try:
            # Write video (silence ffmpeg logs)
            writer = imageio.get_writer(cache_file, fps=fps, ffmpeg_log_level='error', **codec_params)
            try:
                if torch.is_tensor(tensor):
                    # Stream frames to avoid materializing the full video on CPU.
                    if tensor.dtype == torch.uint8 and tensor.ndim == 5 and tensor.shape[0] == 1 and nrow == 1:
                        frames = tensor[0].permute(1, 2, 3, 0)
                        for frame in frames:
                            writer.append_data(frame.cpu().numpy())
                    else:
                        if tensor.dtype == torch.uint8:
                            tensor = tensor.float().div_(127.5).sub_(1.0)
                        for u in tensor.unbind(2):
                            u = u.clamp(min(value_range), max(value_range))
                            grid = torchvision.utils.make_grid(
                                u, nrow=nrow, normalize=normalize, value_range=value_range
                            )
                            frame = grid.mul(255).type(torch.uint8).permute(1, 2, 0).cpu().numpy()
                            writer.append_data(frame)
                elif isinstance(tensor, (list, tuple)) and tensor and torch.is_tensor(tensor[0]):
                    for chunk in tensor:
                        if chunk is None:
                            continue
                        if chunk.ndim == 4:
                            if chunk.shape[-1] in (1, 3, 4):
                                frames = chunk
                            else:
                                frames = chunk.permute(1, 2, 3, 0)
                            for frame in frames:
                                writer.append_data(_video_frame_tensor_to_uint8(frame, normalize=normalize, value_range=value_range))
                        else:
                            writer.append_data(chunk)
                else:
                    for frame in tensor:
                        writer.append_data(frame)
            finally:
                writer.close()

            return cache_file

        except Exception as e:
            error = e
            print(f"error saving {save_file}: {e}")


def save_hdr_video(
                tensor,
                save_file=None,
                fps=30,
                codec_type='libx264_8',
                container='mp4',
                preview_exposure=0.0,
                retry=5):
    """Save linear HDR video as a tagged 10-bit HEVC HDR file."""
    suffix = f'.{container}'
    output_file = osp.join('/tmp', rand_name(suffix=suffix)) if save_file is None else save_file
    if not output_file.endswith(suffix):
        output_file = osp.splitext(output_file)[0] + suffix
    ffmpeg_path = resolve_media_binary("ffmpeg")
    if ffmpeg_path is None:
        raise RuntimeError("ffmpeg binary not found")

    width = height = None
    for chunk in iter_video_chunks(tensor):
        if chunk is None:
            continue
        cur = chunk[0] if chunk.ndim == 5 and chunk.shape[0] == 1 else chunk
        if cur.ndim == 4:
            height, width = int(cur.shape[2]), int(cur.shape[3])
            break
    if width is None or height is None:
        raise RuntimeError("Unable to determine HDR video dimensions.")
    error = None
    for _ in range(retry):
        cmd = [
            ffmpeg_path, "-y", "-v", "error",
            "-f", "rawvideo",
            "-pix_fmt", "gbrpf32le",
            "-video_size", f"{width}x{height}",
            "-framerate", f"{float(fps):.12g}",
            "-i", "pipe:0",
            *get_hdr_video_encode_args(codec_type, container),
            "-an",
            output_file,
        ]
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            assert process.stdin is not None
            wrote_frame = False
            for frame_bytes in iter_hdr_gbrpf32_frames(tensor):
                process.stdin.write(frame_bytes)
                wrote_frame = True
            if not wrote_frame:
                raise RuntimeError("No HDR frames available to save.")
            process.stdin.close()
            stderr = process.stderr.read().decode("utf-8", errors="ignore").strip() if process.stderr is not None else ""
            ret = process.wait()
            if ret != 0:
                raise RuntimeError(stderr or "ffmpeg HDR encode failed")
            return output_file
        except Exception as e:
            error = e
            try:
                if process.stdin is not None and not process.stdin.closed:
                    process.stdin.close()
            except Exception:
                pass
            process.kill()
            print(f"error saving HDR {save_file}: {e}")
    raise error or RuntimeError(f"Failed to save HDR video: {save_file}")


def _get_codec_params(codec_type, container):
    """Get codec parameters based on codec type and container."""
    return get_imageio_codec_params(codec_type, container)




def save_image(tensor,
                save_file,
                nrow=8,
                normalize=True,
                value_range=(-1, 1),
                quality='jpeg_95',  # 'jpeg_95', 'jpeg_85', 'jpeg_70', 'jpeg_50', 'webp_95', 'webp_85', 'webp_70', 'webp_50', 'png', 'webp_lossless'
                retry=5):
    """Save tensor as image with configurable format and quality."""

    RGBA = tensor.shape[0] == 4

    # Get format and quality settings
    format_info = get_image_format(quality, rgba=RGBA)
    
    # Rename file extension to match requested format
    save_file = osp.splitext(save_file)[0] + format_info['ext']
    
    # Save image
    error = None
                         
    for _ in range(retry):
        try:
            if format_info['use_pil'] or RGBA:
                # Use PIL for WebP and advanced options
                if tensor.dtype == torch.uint8:
                    grid = torchvision.utils.make_grid(tensor, nrow=nrow, normalize=False).permute(1, 2, 0).cpu().numpy()
                else:
                    tensor = tensor.clamp(min(value_range), max(value_range))
                    grid = torchvision.utils.make_grid(tensor, nrow=nrow, normalize=normalize, value_range=value_range)
                    grid = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                mode = 'RGBA' if RGBA else 'RGB'
                img = Image.fromarray(grid, mode=mode)
                img.save(save_file, **format_info['params'])
            else:
                # Use torchvision for JPEG and PNG
                was_uint8 = tensor.dtype == torch.uint8
                tensor = tensor.float().div_(255.0) if was_uint8 else tensor.clamp(min(value_range), max(value_range))
                torchvision.utils.save_image(tensor, save_file, nrow=nrow, normalize=False if was_uint8 else normalize, value_range=value_range, **format_info['params'])
            break
        except Exception as e:
            error = e
            continue
    else:
        raise OSError(f"Failed to save image '{save_file}' after {retry} attempts: {error}") from error
    
    return save_file


def get_image_format(quality, rgba=False):
    """Get the saved format and parameters; transparent images require PNG."""
    if rgba:
        quality = "png"
    formats = {
        # JPEG with PIL (so 'quality' works)
        'jpeg_95': {'ext': '.jpg', 'params': {'quality': 95}, 'use_pil': True},
        'jpeg_85': {'ext': '.jpg', 'params': {'quality': 85}, 'use_pil': True},
        'jpeg_70': {'ext': '.jpg', 'params': {'quality': 70}, 'use_pil': True},
        'jpeg_50': {'ext': '.jpg', 'params': {'quality': 50}, 'use_pil': True},

        # PNG with torchvision
        'png': {'ext': '.png', 'params': {}, 'use_pil': False},

        # WebP with PIL (for quality control)
        'webp_95': {'ext': '.webp', 'params': {'quality': 95}, 'use_pil': True},
        'webp_85': {'ext': '.webp', 'params': {'quality': 85}, 'use_pil': True},
        'webp_70': {'ext': '.webp', 'params': {'quality': 70}, 'use_pil': True},
        'webp_50': {'ext': '.webp', 'params': {'quality': 50}, 'use_pil': True},
        'webp_lossless': {'ext': '.webp', 'params': {'lossless': True}, 'use_pil': True},
    }
    return formats.get(quality, formats['jpeg_95'])


from PIL import Image, PngImagePlugin

def _enc_uc(s):
    try: return b"ASCII\0\0\0" + s.encode("ascii")
    except UnicodeEncodeError: return b"UNICODE\0" + s.encode("utf-16le")

def _dec_uc(b):
    if not isinstance(b, (bytes, bytearray)):
        try: b = bytes(b)
        except Exception: return None
    if b.startswith(b"ASCII\0\0\0"): return b[8:].decode("ascii", "ignore")
    if b.startswith(b"UNICODE\0"):   return b[8:].decode("utf-16le", "ignore")
    return b.decode("utf-8", "ignore")


def _blank_exif_dict():
    return {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}


def _load_exif_dict(image_path, ext):
    import piexif
    try:
        if ext in (".jpg", ".jpeg"):
            return piexif.load(image_path)
        if ext == ".webp":
            with Image.open(image_path) as im:
                exif_bytes = im.info.get("exif")
            return piexif.load(exif_bytes) if exif_bytes else _blank_exif_dict()
    except Exception:
        pass
    return _blank_exif_dict()


def _insert_exif_user_comment(image_path, comment_text, ext):
    import piexif
    exif_dict = _load_exif_dict(image_path, ext)
    exif_dict.setdefault("Exif", {})
    exif_dict["Exif"][piexif.ExifIFD.UserComment] = _enc_uc(comment_text)
    piexif.insert(piexif.dump(exif_dict), image_path)


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _build_png_chunk(chunk_type, data):
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xffffffff)


def _is_png_comment_chunk(chunk_type, data):
    if chunk_type not in {b"tEXt", b"zTXt", b"iTXt"}:
        return False
    return data.split(b"\x00", 1)[0] == b"comment"


def _write_png_comment_metadata(image_path, comment_text):
    raw = open(image_path, "rb").read()
    if not raw.startswith(_PNG_SIGNATURE):
        raise ValueError("Invalid PNG signature")
    comment_chunk = _build_png_chunk(b"iTXt", b"comment\x00\x00\x00\x00\x00" + comment_text.encode("utf-8"))
    out = bytearray(_PNG_SIGNATURE)
    pos = len(_PNG_SIGNATURE)
    inserted = False
    while pos < len(raw):
        if pos + 8 > len(raw):
            raise ValueError("Corrupted PNG chunk header")
        length = struct.unpack(">I", raw[pos:pos + 4])[0]
        chunk_type = raw[pos + 4:pos + 8]
        end = pos + 12 + length
        if end > len(raw):
            raise ValueError("Corrupted PNG chunk payload")
        chunk_data = raw[pos + 8:pos + 8 + length]
        chunk = raw[pos:end]
        pos = end
        if _is_png_comment_chunk(chunk_type, chunk_data):
            continue
        if not inserted and chunk_type == b"IDAT":
            out.extend(comment_chunk)
            inserted = True
        out.extend(chunk)
    if not inserted:
        raise ValueError("PNG image data chunk not found")
    with open(image_path, "wb") as writer:
        writer.write(out)

def save_image_metadata(image_path, metadata_dict, **save_kwargs):
    try:
        j = json.dumps(metadata_dict, ensure_ascii=False)
        ext = os.path.splitext(image_path)[1].lower()
        if ext == ".png":
            _write_png_comment_metadata(image_path, j); return True
        if ext in (".jpg", ".jpeg", ".webp"):
            _insert_exif_user_comment(image_path, j, ext); return True
        raise ValueError("Unsupported format")
    except Exception as e:
        print(f"Error saving metadata: {e}"); return False

def read_image_metadata(image_path):
    try:
        ext = os.path.splitext(image_path)[1].lower()
        if ext == ".png":
            # Read text chunks without inflating IDAT (including text after IDAT).
            with open(image_path, "rb") as fp:
                if fp.read(8) != _PNG_SIGNATURE:
                    raise ValueError("Invalid PNG signature")
                png = PngImagePlugin.PngStream(fp)
                try:
                    while True:
                        cid, pos, length = png.read()
                        if cid == b"IEND":
                            break
                        if cid in (b"tEXt", b"zTXt", b"iTXt"):
                            data = png.call(cid, pos, length)
                            png.crc(cid, data)
                        else:
                            fp.seek(length + 4, 1)
                    val = png.im_text.get("comment")
                    return json.loads(val) if val else None
                finally:
                    png.close()
        with Image.open(image_path) as im:
            if ext in (".jpg", ".jpeg"):
                import piexif
                try:
                    uc = piexif.load(image_path).get("Exif", {}).get(piexif.ExifIFD.UserComment)
                    s = _dec_uc(uc) if uc else None
                    if s:
                        return json.loads(s)
                except Exception:
                    pass
                val = im.info.get("comment")
                if isinstance(val, (bytes, bytearray)): val = val.decode("utf-8", "ignore")
                if val:
                    try: return json.loads(val)
                    except Exception: pass
                exif = getattr(im, "getexif", lambda: None)()
                if exif:
                    uc = exif.get(37510)  # UserComment
                    s = _dec_uc(uc) if uc else None
                    if s:
                        try: return json.loads(s)
                        except Exception: pass
                return None
            if ext == ".webp":
                import piexif
                exif_bytes = im.info.get("exif")
                if not exif_bytes: return None
                uc = piexif.load(exif_bytes).get("Exif", {}).get(piexif.ExifIFD.UserComment)
                s = _dec_uc(uc) if uc else None
                return json.loads(s) if s else None
            return None
    except Exception as e:
        print(f"Error reading metadata: {e}"); return None

