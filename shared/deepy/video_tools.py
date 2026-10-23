# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import base64
import contextlib
import hashlib
import math
import os
import re
import shutil
import subprocess
from collections import OrderedDict
from tempfile import TemporaryDirectory

import ffmpeg
from PIL import Image, ImageDraw, ImageFont

from shared.ffmpeg_setup import download_ffmpeg
from shared.utils.audio_video import get_mp4_audio_codec_settings, get_video_audio_extra_args
from shared.utils.video_codecs import get_video_container_extension, get_video_encode_args, validate_video_output_settings
from shared.utils.utils import get_video_frame, get_video_info
from shared.utils.subprocess_stdin import child_kwargs
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".webm", ".avi"}
_THUMB_DATA_URL_CACHE: OrderedDict[str, str] = OrderedDict()
_THUMB_CACHE_MAX = 64


def _get_ffmpeg_path() -> str:
    download_ffmpeg()
    ffmpeg_path = os.environ.get("FFMPEG_BINARY", "") or shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg is not available.")
    return ffmpeg_path


def _run_ffmpeg(args: list[str]) -> None:
    result = subprocess.run([_get_ffmpeg_path(), "-y", "-v", "error", *args], capture_output=True, text=True, encoding="utf-8", errors="replace", **child_kwargs())
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "ffmpeg command failed").strip())


def _format_timestamp(seconds: float | int | None) -> str | None:
    if seconds is None:
        return None
    seconds = float(seconds)
    if seconds < 0:
        raise ValueError("Time values must be >= 0.")
    return f"{seconds:.6f}".rstrip("0").rstrip(".")


def _resolve_segment_args(start_time: float | int | None = None, end_time: float | int | None = None, duration: float | int | None = None) -> tuple[str | None, str | None]:
    if end_time is not None and duration is not None:
        raise ValueError("Specify either end_time or duration, not both.")
    start_str = _format_timestamp(0 if start_time is None else start_time)
    end_str = _format_timestamp(end_time)
    duration_str = _format_timestamp(duration)
    if end_str is not None and float(end_str) <= float(start_str):
        raise ValueError("end_time must be greater than start_time.")
    if duration_str is not None and float(duration_str) <= 0:
        raise ValueError("duration must be > 0.")
    # input seeking (-ss before -i) restarts timestamps at the start time, so an end time becomes the remaining duration
    return start_str, duration_str if end_str is None else _format_timestamp(float(end_str) - float(start_str))


def get_audio_standalone_extension(codec_key: str | None) -> str:
    codec_key = str(codec_key or "wav").strip().lower() or "wav"
    if codec_key == "mp3":
        codec_key = "mp3_192"
    if codec_key == "flac":
        return ".flac"
    if codec_key in {"m4a", "alac"}:
        return ".m4a"
    return ".wav" if codec_key == "wav" else ".mp3"


def _get_mp4_audio_encode_args(codec_key: str | None) -> list[str]:
    settings = get_mp4_audio_codec_settings(codec_key)
    args = ["-c:a", settings["codec"], *get_video_audio_extra_args(codec_key)]
    if settings.get("bitrate"):
        args += ["-b:a", settings["bitrate"]]
    return args


def _get_standalone_audio_encode_args(codec_key: str | None) -> list[str]:
    codec_key = str(codec_key or "wav").strip().lower() or "wav"
    if codec_key == "mp3":
        codec_key = "mp3_192"
    if codec_key == "wav":
        return ["-c:a", "pcm_s16le"]
    if codec_key == "flac":
        return ["-c:a", "flac", "-sample_fmt", "s16"]
    if codec_key in {"m4a", "alac"}:
        return ["-c:a", "alac"]
    bitrate = {"mp3_128": "128k", "mp3_192": "192k", "mp3_320": "320k"}.get(codec_key, "192k")
    return ["-c:a", "libmp3lame", "-b:a", bitrate]


def has_video_extension(path: str) -> bool:
    return os.path.splitext(str(path or "").strip())[1].lower() in VIDEO_EXTENSIONS


def get_video_thumbnail_data_url(video_path: str) -> str:
    video_path = os.path.abspath(os.path.normpath(str(video_path or "").strip()))
    if len(video_path) == 0 or not os.path.isfile(video_path) or not has_video_extension(video_path):
        return ""
    stat = os.stat(video_path)
    cache_key = hashlib.sha1(f"{os.path.abspath(video_path)}|{stat.st_mtime_ns}|{stat.st_size}".encode("utf-8")).hexdigest()
    cached = _THUMB_DATA_URL_CACHE.get(cache_key, "")
    if len(cached) > 0:
        _THUMB_DATA_URL_CACHE.move_to_end(cache_key)
        return cached
    result = subprocess.run(
        [
            _get_ffmpeg_path(),
            "-v",
            "error",
            "-threads",
            "1",
            "-i",
            video_path,
            "-frames:v",
            "1",
            "-vf",
            "scale=320:-1:force_original_aspect_ratio=decrease",
            "-q:v",
            "4",
            "-f",
            "image2pipe",
            "-vcodec",
            "mjpeg",
            "-threads",
            "1",
            "-",
        ],
        capture_output=True,
     **child_kwargs())
    if result.returncode != 0 or len(result.stdout) == 0:
        raise RuntimeError((result.stderr.decode("utf-8", errors="ignore").strip() or "ffmpeg thumbnail extraction failed").strip())
    data_url = f"data:image/jpeg;base64,{base64.b64encode(result.stdout).decode('ascii')}"
    _THUMB_DATA_URL_CACHE[cache_key] = data_url
    while len(_THUMB_DATA_URL_CACHE) > _THUMB_CACHE_MAX:
        _THUMB_DATA_URL_CACHE.popitem(last=False)
    return data_url


def picture_checks(path: str, start_s: float = 0.0, end_s: float | None = None, *, min_black_s: float = 1.0) -> dict[str, list]:
    """Measured picture events in [start_s, end_s] (seconds), in one decode: "black_ranges_seconds" (ffmpeg blackdetect, nearly every
    pixel below 10 % luminance) and "cut_times_seconds" (hard cuts: a frame whose ffmpeg scene score is at least 0.08 and 3 times the
    highest of the 6 frames on either side; cuts score 0.14-1, down to 0.14 between dark shots, while continuous motion stays at most
    0.1 and close to its neighbours). Only non-empty entries are returned."""
    span = [] if end_s is None or not math.isfinite(end_s) else ["-t", f"{end_s - start_s:.3f}"]
    picture = f"scale=160:-2,blackdetect=d={min_black_s:g}:pix_th=0.10,select='gte(scene,0)',metadata=print:key=lavfi.scene_score"
    process = subprocess.run([_get_ffmpeg_path(), "-v", "info", "-ss", f"{start_s:.3f}", *span, "-i", path, "-an", "-vf", picture, "-f", "null", "-"], capture_output=True, text=True, encoding="utf-8", errors="replace", **child_kwargs())
    if process.returncode:
        raise RuntimeError(process.stderr.strip()[-500:] or "ffmpeg could not decode the video.")
    times, scores = [float(time) for time in re.findall(r"pts_time:([\d.]+)", process.stderr)], [float(score) for score in re.findall(r"lavfi\.scene_score=([\d.]+)", process.stderr)]
    cuts = [time for index, (time, score) in enumerate(zip(times, scores)) if score >= 0.08 and score >= 3 * max(scores[max(0, index - 6):index] + scores[index + 1:index + 7] or [0])]
    checks = {"black_ranges_seconds": [[round(start_s + float(begin), 2), round(start_s + float(end), 2)] for begin, end in re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", process.stderr)],
              "cut_times_seconds": [round(start_s + time, 2) for time in cuts]}
    return {key: value for key, value in checks.items() if value}


def quiet_ranges(path: str, *, threshold_db: float = -45.0, min_seconds: float = 1.0) -> list[list[float]]:
    """Stretches (seconds) where the first audio track stays below threshold_db: ffmpeg silencedetect."""
    command = [_get_ffmpeg_path(), "-v", "info", "-i", path, "-vn", "-af", f"silencedetect=n={threshold_db:g}dB:d={min_seconds:g}", "-f", "null", "-"]
    process = subprocess.run(command, capture_output=True, text=True, errors="replace", **child_kwargs())
    if process.returncode:
        raise RuntimeError(process.stderr.strip()[-500:] or "ffmpeg could not decode the audio.")
    duration = get_media_duration(path)
    starts = [max(0.0, float(value)) for value in re.findall(r"silence_start: (-?[\d.]+)", process.stderr)]
    ends = [float(value) for value in re.findall(r"silence_end: ([\d.]+)", process.stderr)]
    ends += [duration] * (len(starts) - len(ends))  # a silence lasting to the end
    ranges = [[round(start, 2), round(end, 2)] for start, end in zip(starts, ends)]
    audio_end = max((float(stream["duration"]) for stream in ffmpeg.probe(path)["streams"] if stream.get("codec_type") == "audio" and "duration" in stream), default=duration)
    if duration - audio_end >= min_seconds:  # the audio stream stops before the picture: nothing plays after it
        if ranges and ranges[-1][1] >= audio_end - 0.05:
            ranges[-1][1] = round(duration, 2)
        else:
            ranges.append([round(audio_end, 2), round(duration, 2)])
    return ranges


def _pipe_frames(command: list[str], first, frames, failure: str) -> None:
    """Run ffmpeg with the PIL frames (first, then the rest) written to its standard input."""
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        process.stdin.write(first.tobytes())
        for frame in frames:
            process.stdin.write(frame.tobytes())
    except BrokenPipeError:
        pass
    finally:
        with contextlib.suppress(BrokenPipeError):
            process.stdin.close()  # also when a frame fails to render, so that ffmpeg stops waiting for input
        error = process.stderr.read().decode(errors="replace")
        if process.wait():
            raise RuntimeError(error.strip() or failure)


def encode_frames(frames, output_path: str, fps: float, *, video_codec: str | None = None, video_container: str | None = None) -> str:
    """Encode RGB PIL frames of one size as a silent video; odd sizes get one black pixel row/column for 4:2:0 encoders."""
    frames = iter(frames)
    first = next(frames)
    width, height = first.size
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    command = [_get_ffmpeg_path(), "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "pipe:0",
               *(["-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"] if width % 2 or height % 2 else []), *get_video_encode_args(video_codec, video_container),
               *(["-movflags", "+faststart"] if str(video_container or "mp4").strip().lower() == "mp4" else []), output_path]
    _pipe_frames(command, first, frames, "ffmpeg could not encode the frames.")
    return output_path


def overlay_frames(frames, video_path: str, output_path: str, fps: float, start_s: float = 0.0, *, video_codec: str | None = None, video_container: str | None = None, audio_codec: str | None = None) -> str:
    """Burn RGBA PIL frames of the video's size into its picture from start_s, one per video frame; the overlay's alpha shows the video
    through. Audio is copied when the container stays the same, and subtitle tracks are kept."""
    frames = iter(frames)
    first = next(frames)
    streams = ffmpeg.probe(video_path).get("streams", [])
    video = next(stream for stream in streams if stream.get("codec_type") == "video")
    if first.size != (int(video["width"]), int(video["height"])):
        raise ValueError(f"The drawing renders at {first.width}x{first.height} over a {video['width']}x{video['height']} video: draw it in the video's aspect ratio, e.g. width=\"{video['width']}\" height=\"{video['height']}\" viewBox=\"0 0 {video['width']} {video['height']}\".")
    output_extension = os.path.splitext(output_path)[1].lower()
    if os.path.splitext(video_path)[1].lower() != output_extension:
        get_video_audio_extra_args(audio_codec, output_path)
    subtitle_args = [arg for index, stream in enumerate(stream for stream in streams if stream.get("codec_type") == "subtitle") for arg in _subtitle_codec_args(index, stream, output_extension)]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    command = [_get_ffmpeg_path(), "-y", "-v", "error", "-i", video_path, "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{first.width}x{first.height}", "-framerate", f"{fps:g}", "-i", "pipe:0",
               "-filter_complex", f"[1:v]setpts=PTS-STARTPTS+{float(start_s):g}/TB[drawing];[0:v][drawing]overlay=eof_action=pass[video]", "-map", "[video]", "-map", "0:a?", "-map", "0:s?",
               *get_video_encode_args(video_codec, video_container), *(["-c:a", "copy"] if os.path.splitext(video_path)[1].lower() == output_extension else _get_mp4_audio_encode_args(audio_codec)), *subtitle_args,
               *(["-movflags", "+faststart"] if str(video_container or "mp4").strip().lower() == "mp4" else []), output_path]
    _pipe_frames(command, first, frames, "ffmpeg could not draw over the video.")
    return output_path


MERGE_TRANSITIONS = {"cut": None, "crossfade": "fade", "fade_black": "fadeblack", "fade_white": "fadewhite"}  # names for agents -> ffmpeg xfade


def merge_videos(videos: list[str], output_path: str, *, transitions: str | list[str] | None = None, transition_s: float = 1.0, video_codec: str | None = None, video_container: str | None = None, audio_codec: str | None = None) -> str:
    """Join videos in order with a cut, crossfade or fade through black/white at each join (transitions: one per join, or one for all).
    Clips are fitted to the first one's size and frame rate; a transition overlaps the two clips by transition_s, and the soundtracks
    follow the picture (silence for a clip without sound)."""
    error = validate_video_output_settings(video_codec, video_container, audio_codec)
    if error is not None:
        raise ValueError(error)
    videos = [os.path.normpath(str(path or "").strip()) for path in videos]
    if not 2 <= len(videos) <= 32:
        raise ValueError("Merge between 2 and 32 videos.")
    for path in videos:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Video not found: {path}")
    transitions = [transitions] if isinstance(transitions, str) else list(transitions or ["cut"])
    transitions = transitions * (len(videos) - 1) if len(transitions) == 1 else transitions
    if len(transitions) != len(videos) - 1 or any(name not in MERGE_TRANSITIONS for name in transitions):
        raise ValueError(f"transitions needs one value per join ({len(videos) - 1}) or one for all, among {', '.join(MERGE_TRANSITIONS)}.")
    first = _probe_video_stream(videos[0])
    width, height, rate = int(first["width"]), int(first["height"]), first.get("avg_frame_rate") or first["r_frame_rate"]
    durations = [get_media_duration(path) for path in videos]
    if any(not duration or duration <= 0 for duration in durations):
        raise ValueError("Every video needs a known duration.")
    with_audio = [_has_audio_stream(path) for path in videos]
    audio = any(with_audio)
    filters = []
    for index, (duration, has_audio) in enumerate(zip(durations, with_audio)):
        filters.append(f"[{index}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,fps={rate},format=yuv420p,settb=AVTB,setpts=PTS-STARTPTS[v{index}]")
        if audio:  # every clip brings exactly its own length of sound
            source = f"[{index}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo,asetpts=PTS-STARTPTS," if has_audio else "anullsrc=r=48000:cl=stereo,"
            filters.append(f"{source}apad,atrim=0:{duration:.6f}[a{index}]")
    video, sound, total = "[v0]", "[a0]", durations[0]
    for index, (name, duration) in enumerate(zip(transitions, durations[1:]), 1):
        if MERGE_TRANSITIONS[name] is None:
            filters.append(f"{video}{sound if audio else ''}[v{index}]{f'[a{index}]' if audio else ''}concat=n=2:v=1:a={int(audio)}[vj{index}]{f'[aj{index}]' if audio else ''}")
            total += duration
        else:
            if not 0.05 <= transition_s < min(total, duration):
                raise ValueError(f"transition_s must be at least 0.05 s and shorter than the clips it joins ({min(total, duration):.2f} s at join {index}).")
            filters.append(f"{video}[v{index}]xfade=transition={MERGE_TRANSITIONS[name]}:duration={transition_s:g}:offset={total - transition_s:.6f}[vj{index}]")
            if audio:
                filters.append(f"{sound}[a{index}]acrossfade=d={transition_s:g}[aj{index}]")
            total += duration - transition_s
        video, sound = f"[vj{index}]", f"[aj{index}]"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    command = [arg for path in videos for arg in ("-i", path)] + ["-filter_complex", ";".join(filters), "-map", video, *(["-map", sound] if audio else []), *get_video_encode_args(video_codec, video_container)]
    if audio:
        command += _get_mp4_audio_encode_args(audio_codec)
    if str(video_container or "mp4").strip().lower() == "mp4":
        command += ["-movflags", "+faststart"]
    _run_ffmpeg([*command, output_path])
    return output_path


def _side_by_side_layout(count: int, layout: str | None) -> tuple[int, int, str]:
    layout = str(layout or "horizontal").strip().lower() or "horizontal"
    if layout == "horizontal":
        return count, 1, layout
    if layout == "vertical":
        return 1, count, layout
    if layout == "grid":
        columns = math.ceil(math.sqrt(count))
        return columns, math.ceil(count / columns), layout
    match = re.fullmatch(r"(\d+)x(\d+)", layout)
    if match is None:
        raise ValueError("layout must be horizontal, vertical, grid, or COLSxROWS.")
    columns, rows = map(int, match.groups())
    if columns < 1 or rows < 1 or columns * rows < count:
        raise ValueError(f"layout {layout} does not have room for {count} media items.")
    return columns, rows, layout


def _side_by_side_legend_image(width: int, height: int, text: str) -> Image.Image:
    image = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(image)
    font_size = max(12, min(36, height // 2))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        font = ImageFont.load_default()
    original_text = str(text or "").strip().replace("\n", " ")
    text = original_text
    while text and draw.textlength(text, font=font) > width - 16:
        text = text[:-1].rstrip()
    if text != original_text:
        suffix = "..."
        while text and draw.textlength(f"{text.rstrip('.')}{suffix}", font=font) > width - 16:
            text = text[:-1].rstrip()
        text = f"{text.rstrip('.')}{suffix}" if draw.textlength(suffix, font=font) <= width - 16 else ""
    if text:
        box = draw.textbbox((0, 0), text, font=font)
        draw.text(((width - box[2] + box[0]) / 2, (height - box[3] + box[1]) / 2), text, fill="white", font=font)
    return image


def side_by_side_media(source_paths: list[str], output_path: str, layout: str | None = None, legends: list[str] | None = None, *, video_codec: str | None = None, video_container: str | None = None, audio_codec: str | None = None) -> str:
    error = validate_video_output_settings(video_codec, video_container, audio_codec)
    if error is not None:
        raise ValueError(error)
    source_paths = [os.path.normpath(str(path or "").strip()) for path in source_paths]
    if not source_paths:
        raise ValueError("media_ids must contain at least one image or video.")
    missing = next((path for path in source_paths if not os.path.isfile(path)), None)
    if missing is not None:
        raise FileNotFoundError(f"Media not found: {missing}")
    if legends is not None and not isinstance(legends, list):
        raise ValueError("legends must be an array of strings.")
    legends = [] if legends is None else [str(legend or "").strip() for legend in legends]
    if len(legends) > len(source_paths):
        raise ValueError("legends cannot contain more entries than media_ids.")
    legends += [""] * (len(source_paths) - len(legends))
    columns, rows, _resolved_layout = _side_by_side_layout(len(source_paths), layout)
    videos = [has_video_extension(path) for path in source_paths]
    sizes = []
    for path, is_video in zip(source_paths, videos):
        if is_video:
            stream = _probe_video_stream(path)
            sizes.append((int(stream["width"]), int(stream["height"])))
        else:
            with Image.open(path) as image:
                sizes.append(image.size)
    tile_width, tile_height = sizes[0]
    has_legends = any(legends)
    legend_height = max(32, min(96, tile_height // 8)) if has_legends else 0
    cell_height = tile_height + legend_height
    output_path = os.path.normpath(str(output_path or "").strip())
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if not any(videos):
        canvas = Image.new("RGB", (columns * tile_width, rows * cell_height), "black")
        for index, (path, legend) in enumerate(zip(source_paths, legends)):
            with Image.open(path) as source:
                image = source.convert("RGB")
                image.thumbnail((tile_width, tile_height), Image.Resampling.LANCZOS)
                x = index % columns * tile_width
                y = index // columns * cell_height
                canvas.paste(image, (x + (tile_width - image.width) // 2, y + (tile_height - image.height) // 2))
                if has_legends:
                    canvas.paste(_side_by_side_legend_image(tile_width, legend_height, legend), (x, y + tile_height))
        canvas.save(output_path)
        return output_path

    tile_width += tile_width % 2
    tile_height += tile_height % 2
    legend_height += legend_height % 2
    cell_height = tile_height + legend_height
    durations = []
    for path, is_video in zip(source_paths, videos):
        if not is_video:
            continue
        duration = get_media_duration(path)
        if duration is None or duration <= 0:
            fps, _width, _height, frame_count = get_video_info(path)
            duration = frame_count / fps if fps > 0 else 0
        if duration <= 0:
            raise ValueError(f"Could not determine video duration: {path}")
        durations.append(duration)
    duration = max(durations)
    first_video = source_paths[videos.index(True)]
    fps = get_precise_video_fps(first_video)
    if fps is None or fps <= 0:
        fps = float(get_video_info(first_video)[0])
    fps = fps if fps > 0 else 24.0
    fps_text = f"{fps:.6f}".rstrip("0").rstrip(".")
    duration_text = f"{duration:.6f}".rstrip("0").rstrip(".")
    input_args = []
    for path, is_video in zip(source_paths, videos):
        input_args += ["-i", path] if is_video else ["-loop", "1", "-framerate", fps_text, "-i", path]
    with TemporaryDirectory(prefix="wangp_side_by_side_") as temp_dir:
        if has_legends:
            for index, legend in enumerate(legends):
                legend_path = os.path.join(temp_dir, f"legend_{index}.png")
                _side_by_side_legend_image(tile_width, legend_height, legend).save(legend_path)
                input_args += ["-loop", "1", "-framerate", fps_text, "-i", legend_path]
        filters = []
        tile_labels = []
        for index in range(len(source_paths)):
            filters.append(f"[{index}:v]setpts=PTS-STARTPTS,fps={fps_text},scale={tile_width}:{tile_height}:force_original_aspect_ratio=decrease,pad={tile_width}:{tile_height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,tpad=stop_mode=clone:stop_duration={duration_text},trim=duration={duration_text}[v{index}]")
            if has_legends:
                filters.append(f"[{len(source_paths) + index}:v]setpts=PTS-STARTPTS,fps={fps_text},trim=duration={duration_text}[l{index}]")
                filters.append(f"[v{index}][l{index}]vstack=inputs=2[t{index}]")
                tile_labels.append(f"[t{index}]")
            else:
                tile_labels.append(f"[v{index}]")
        output_width, output_height = columns * tile_width, rows * cell_height
        if len(source_paths) == 1:
            filters.append(f"{tile_labels[0]}pad={output_width}:{output_height}:0:0:color=black[out]")
        else:
            positions = "|".join(f"{index % columns * tile_width}_{index // columns * cell_height}" for index in range(len(source_paths)))
            filters.append(f"{''.join(tile_labels)}xstack=inputs={len(source_paths)}:layout={positions}:fill=black:shortest=1[grid]")
            filters.append(f"[grid]pad={output_width}:{output_height}:0:0:color=black[out]")
        cmd = [*input_args, "-filter_complex", ";".join(filters), "-map", "[out]"]
        audio_index = next((index for index, (path, is_video) in enumerate(zip(source_paths, videos)) if is_video and _has_audio_stream(path)), None)
        if audio_index is not None:
            cmd += ["-map", f"{audio_index}:a:0", "-af", "apad", *_get_mp4_audio_encode_args(audio_codec)]
        cmd += [*get_video_encode_args(video_codec, video_container), "-t", duration_text]
        if str(video_container or "mp4").strip().lower() == "mp4":
            cmd += ["-movflags", "+faststart"]
        cmd += [output_path]
        _run_ffmpeg(cmd)
    return output_path


def extract_video(source_path: str, output_path: str, start_time: float | int = 0, end_time: float | int | None = None, duration: float | int | None = None, *, video_codec: str | None = None, video_container: str | None = None, audio_codec: str | None = None) -> str:
    error = validate_video_output_settings(video_codec, video_container, audio_codec)
    if error is not None:
        raise ValueError(error)
    source_path = os.path.normpath(str(source_path or "").strip())
    output_path = os.path.normpath(str(output_path or "").strip())
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Video not found: {source_path}")
    start_str, duration_str = _resolve_segment_args(start_time, end_time, duration)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cmd = ["-ss", start_str, "-i", source_path]
    if duration_str is not None:
        cmd += ["-t", duration_str]
    cmd += ["-map", "0:v:0", "-map", "0:a?"]
    cmd += get_video_encode_args(video_codec, video_container)
    cmd += _get_mp4_audio_encode_args(audio_codec)
    if str(video_container or "mp4").strip().lower() == "mp4":
        cmd += ["-movflags", "+faststart"]
    cmd += [output_path]
    _run_ffmpeg(cmd)
    return output_path


def extract_audio(source_path: str, output_path: str, start_time: float | int | None = None, end_time: float | int | None = None, duration: float | int | None = None, audio_track_no: int | None = None, *, audio_codec: str | None = None) -> str:
    source_path = os.path.normpath(str(source_path or "").strip())
    output_path = os.path.normpath(str(output_path or "").strip())
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Media not found: {source_path}")
    probe = ffmpeg.probe(source_path)
    audio_streams = [stream for stream in probe.get("streams", []) if str(stream.get("codec_type", "")).strip().lower() == "audio"]
    if len(audio_streams) == 0:
        raise RuntimeError(f"No audio stream found in {source_path}")
    audio_track_no = 1 if audio_track_no is None else int(audio_track_no)
    if audio_track_no <= 0:
        raise ValueError("audio_track_no must be >= 1.")
    if audio_track_no > len(audio_streams):
        raise ValueError(f"audio_track_no must be between 1 and {len(audio_streams)}.")
    start_str, duration_str = _resolve_segment_args(start_time, end_time, duration)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cmd = []
    if start_str is not None:
        cmd += ["-ss", start_str]
    cmd += ["-i", source_path]
    if duration_str is not None:
        cmd += ["-t", duration_str]
    cmd += ["-map", f"0:a:{audio_track_no - 1}", "-vn", *_get_standalone_audio_encode_args(audio_codec), output_path]
    _run_ffmpeg(cmd)
    return output_path


def extract_video_frame(source_path: str, output_path: str, *, frame_no: int | None = None, time_seconds: float | int | None = None) -> str:
    source_path = os.path.normpath(str(source_path or "").strip())
    output_path = os.path.normpath(str(output_path or "").strip())
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Video not found: {source_path}")
    if time_seconds is None and frame_no is None:
        raise ValueError("frame_no or time_seconds is required.")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    frame_no = resolve_video_frame_no(source_path, frame_no=frame_no, time_seconds=time_seconds)
    image = get_video_frame(source_path, frame_no, return_last_if_missing=True, return_PIL=True).convert("RGB")
    image.save(output_path)
    if not os.path.isfile(output_path):
        raise RuntimeError("Frame extraction did not produce an output image.")
    return output_path


def _resolve_crop_anchor(crop_anchor: str | None) -> tuple[str, str]:
    normalized = str(crop_anchor or "center").strip().lower().replace("-", "_").replace(" ", "_") or "center"
    anchor_aliases = {
        "centre": "center",
        "middle": "center",
        "center_left": "left",
        "left_center": "left",
        "center_right": "right",
        "right_center": "right",
        "top_center": "top",
        "center_top": "top",
        "bottom_center": "bottom",
        "center_bottom": "bottom",
    }
    normalized = anchor_aliases.get(normalized, normalized)
    if normalized in {"center", "left", "right", "top", "bottom"}:
        return {
            "center": ("center", "center"),
            "left": ("left", "center"),
            "right": ("right", "center"),
            "top": ("center", "top"),
            "bottom": ("center", "bottom"),
        }[normalized]
    corner_anchors = {
        "top_left": ("left", "top"),
        "top_right": ("right", "top"),
        "bottom_left": ("left", "bottom"),
        "bottom_right": ("right", "bottom"),
    }
    if normalized in corner_anchors:
        return corner_anchors[normalized]
    raise ValueError("crop_anchor must be one of center, left, right, top, bottom, top_left, top_right, bottom_left, or bottom_right.")


def _resolve_resize_crop_geometry(
    source_width: int,
    source_height: int,
    *,
    crop_left: float = 0,
    crop_top: float = 0,
    crop_right: float = 0,
    crop_bottom: float = 0,
    crop_unit: str = "pixels",
    target_width: int | None = None,
    target_height: int | None = None,
    preserve_aspect_ratio: bool = True,
    crop_anchor: str = "center",
) -> tuple[int, int, int, int]:
    crop_unit = str(crop_unit or "pixels").strip().lower() or "pixels"
    if crop_unit not in {"pixels", "percent"}:
        raise ValueError("crop_unit must be 'pixels' or 'percent'.")
    align_x, align_y = _resolve_crop_anchor(crop_anchor)

    def resolve_crop(value: float, total: int) -> int:
        value = float(value or 0)
        if value < 0:
            raise ValueError("Crop values must be >= 0.")
        return round(total * value / 100.0) if crop_unit == "percent" else round(value)

    left = resolve_crop(crop_left, source_width)
    right = resolve_crop(crop_right, source_width)
    top = resolve_crop(crop_top, source_height)
    bottom = resolve_crop(crop_bottom, source_height)
    cropped_width = source_width - left - right
    cropped_height = source_height - top - bottom
    if cropped_width <= 0 or cropped_height <= 0:
        raise ValueError("Crop values remove the whole frame.")
    if not preserve_aspect_ratio or target_width is None or target_height is None:
        return left, top, cropped_width, cropped_height
    if target_width <= 0 or target_height <= 0:
        raise ValueError("width and height must be > 0 when provided.")
    target_ratio = float(target_width) / float(target_height)
    current_ratio = float(cropped_width) / float(cropped_height)
    if math.isclose(current_ratio, target_ratio, rel_tol=0.0, abs_tol=1e-6):
        return left, top, cropped_width, cropped_height
    if current_ratio > target_ratio:
        adjusted_width = min(cropped_width, max(1, int(round(cropped_height * target_ratio))))
        trim = cropped_width - adjusted_width
        left += 0 if align_x == "left" else trim if align_x == "right" else trim // 2
        cropped_width = adjusted_width
    else:
        adjusted_height = min(cropped_height, max(1, int(round(cropped_width / target_ratio))))
        trim = cropped_height - adjusted_height
        top += 0 if align_y == "top" else trim if align_y == "bottom" else trim // 2
        cropped_height = adjusted_height
    if cropped_width <= 0 or cropped_height <= 0:
        raise ValueError("Unable to preserve aspect ratio with the requested crop.")
    return left, top, cropped_width, cropped_height


def mute_video(source_path: str, output_path: str) -> str:
    source_path = os.path.normpath(str(source_path or "").strip())
    output_path = os.path.normpath(str(output_path or "").strip())
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Video not found: {source_path}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    _run_ffmpeg(["-i", source_path, "-map", "0:v:0", "-c:v", "copy", "-an", output_path])
    return output_path


def replace_audio(video_path: str, audio_path: str, output_path: str, *, audio_codec: str | None = None) -> str:
    get_video_audio_extra_args(audio_codec, output_path)
    video_path = os.path.normpath(str(video_path or "").strip())
    audio_path = os.path.normpath(str(audio_path or "").strip())
    output_path = os.path.normpath(str(output_path or "").strip())
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    if not _has_audio_stream(audio_path):
        raise RuntimeError(f"No audio stream found in {audio_path}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    _run_ffmpeg(["-i", video_path, "-i", audio_path, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", *_get_mp4_audio_encode_args(audio_codec), "-shortest", output_path])
    return output_path


def _subtitle_codec_args(index: int, stream: dict, output_extension: str) -> list[str]:
    """Codec of output subtitle stream index for a text subtitle stream: MP4/MOV store mov_text, Matroska SRT."""
    codec = str(stream.get("codec_name", "")).lower()
    if output_extension in {".mp4", ".mov"}:
        if codec not in {"mov_text", "subrip", "ass", "ssa", "webvtt", "text"}:
            raise ValueError(f"Subtitle codec {codec} cannot be written to {output_extension}; choose MKV.")
        return [f"-c:s:{index}", "copy" if codec == "mov_text" else "mov_text"]
    if output_extension == ".mkv":
        return [f"-c:s:{index}", "subrip" if codec == "mov_text" else "copy"]
    return []


def remux_media(audio_paths: list[str], output_path: str, *, mode: str, video_path: str | None = None, include_video_audio: bool = False, gains_db: list[float] | None = None, audio_codec: str | None = None, standalone_audio_codec: str | None = None, subtitle_tracks: list[dict] | None = None, include_video_subtitles: bool = True,
                offsets_s: list[float] | None = None, fade_in_s: list[float] | None = None, fade_out_s: list[float] | None = None, video_audio_gain_db: float | None = None, normalize_lufs: float | None = None,
                trim_start_s: list[float] | None = None, trim_duration_s: list[float] | None = None, gain_points: list | None = None, ducking: dict | None = None, audio_languages: list[str] | None = None) -> str:
    """Copy video, combine or retain audio, and attach selectable subtitle streams. Mixing can trim, offset, fade, gain and automate each
    file, lower some files under speech (ducking) and normalize the result. Files add at their own levels, under a true-peak limiter."""
    mode = str(mode or "").strip().lower()
    if mode not in {"copy", "replace", "mix", "multitrack"}:
        raise ValueError("mode must be copy, replace, mix, or multitrack.")
    if audio_languages is not None and (mode != "multitrack" or len(audio_languages) != len(audio_paths)):
        raise ValueError("audio_languages needs multitrack mode and one language code per file of audio_ids.")
    if not isinstance(audio_paths, list) or (mode != "copy" and not audio_paths and not (mode == "mix" and video_path and include_video_audio)):  # a mix of the video's own soundtrack masters it
        raise ValueError("audio_paths must contain audio files for this mode.")
    audio_paths = [os.path.normpath(str(path or "").strip()) for path in audio_paths]
    for path in audio_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Audio not found: {path}")
        if not _has_audio_stream(path):
            raise ValueError(f"No audio stream found in {path}")
    video_path = os.path.normpath(str(video_path).strip()) if video_path else None
    if video_path and not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    mix_options = any(value is not None for value in (gains_db, offsets_s, fade_in_s, fade_out_s, video_audio_gain_db, normalize_lufs, trim_start_s, trim_duration_s, gain_points, ducking))
    if mode == "copy" and (not video_path or audio_paths or include_video_audio or mix_options):
        raise ValueError("copy requires a video and no audio changes.")
    if mode == "copy" and not subtitle_tracks and include_video_subtitles:
        raise ValueError("copy requires subtitle_tracks or include_video_subtitles=false.")
    if subtitle_tracks and not video_path:
        raise ValueError("Subtitle tracks require a video input.")
    if not isinstance(subtitle_tracks, (list, type(None))):
        raise ValueError("subtitle_tracks must be an array.")
    subtitle_tracks = subtitle_tracks or []
    if sum(track.get("default") is True for track in subtitle_tracks if isinstance(track, dict)) > 1:
        raise ValueError("Only one new subtitle track can be the default.")
    for track in subtitle_tracks:
        if not isinstance(track, dict) or not str(track.get("path", "")).strip():
            raise ValueError("Each subtitle track needs a path.")
        if os.path.splitext(track["path"])[1].lower() not in {".srt", ".vtt", ".ass", ".ssa"}:
            raise ValueError("Subtitle files must be SRT, VTT, ASS, or SSA.")
        if not os.path.isfile(track["path"]):
            raise FileNotFoundError(f"Subtitle not found: {track['path']}")
        if not any(stream.get("codec_type") == "subtitle" for stream in ffmpeg.probe(track["path"]).get("streams", [])):
            raise ValueError(f"No subtitle stream found in {track['path']}")
    if include_video_audio and (not video_path or not _has_audio_stream(video_path)):
        raise ValueError("include_video_audio requires a video with an audio stream.")
    if not str(output_path or "").strip():
        raise ValueError("output_path is required.")
    output_path = os.path.normpath(str(output_path).strip())
    output_extension = os.path.splitext(output_path)[1].lower()
    source_subtitles = []
    if video_path and include_video_subtitles:
        source_subtitles = [stream for stream in ffmpeg.probe(video_path).get("streams", []) if stream.get("codec_type") == "subtitle"]
    subtitle_inputs = []
    subtitle_args = []
    if video_path:
        first_subtitle_input = 1 + len(audio_paths)
        for track in subtitle_tracks:
            subtitle_inputs += ["-i", track["path"]]
        if source_subtitles:
            subtitle_args += ["-map", "0:s?"]
        for index, track in enumerate(subtitle_tracks):
            subtitle_args += ["-map", f"{first_subtitle_input + index}:s:0"]
        for index, stream in enumerate(source_subtitles + [next(item for item in ffmpeg.probe(track["path"]).get("streams", []) if item.get("codec_type") == "subtitle") for track in subtitle_tracks]):
            subtitle_args += _subtitle_codec_args(index, stream, output_extension)
            if index < len(source_subtitles):
                tags = stream.get("tags", {})
                title = tags.get("title") or tags.get("handler_name")
                if title:
                    key = "handler_name" if output_extension in {".mp4", ".mov"} else "title"
                    subtitle_args += [f"-metadata:s:s:{index}", f"{key}={title}"]
        default_subtitle_index = next((len(source_subtitles) + index for index, track in enumerate(subtitle_tracks) if track.get("default") is True), None)
        if default_subtitle_index is not None:
            for index in range(len(source_subtitles) + len(subtitle_tracks)):
                subtitle_args += [f"-disposition:s:{index}", "default" if index == default_subtitle_index else "0"]
        for index, track in enumerate(subtitle_tracks, start=len(source_subtitles)):
            if track.get("language"):
                subtitle_args += [f"-metadata:s:s:{index}", f"language={track['language']}"]
            if track.get("title"):
                key = "handler_name" if output_extension in {".mp4", ".mov"} else "title"
                subtitle_args += [f"-metadata:s:s:{index}", f"{key}={track['title']}"]
            if default_subtitle_index is None and track.get("default") is not None:
                subtitle_args += [f"-disposition:s:{index}", "default" if track["default"] else "0"]
    if mode == "copy":
        command = ["-i", video_path, *subtitle_inputs, "-map", "0:v:0", "-map", "0:a?", *subtitle_args, "-c:v", "copy"]
        if output_extension == ".mkv":
            command += ["-c:a", "copy"]
        else:
            audio_streams = [stream for stream in ffmpeg.probe(video_path).get("streams", []) if stream.get("codec_type") == "audio"]
            for index, stream in enumerate(audio_streams):
                codec = str(stream.get("codec_name", "")).lower()
                if codec in {"aac", "alac"}:
                    command += [f"-c:a:{index}", "copy"]
                else:
                    encode_args = _get_mp4_audio_encode_args(audio_codec)
                    command += [f"-c:a:{index}" if arg == "-c:a" else f"-b:a:{index}" if arg == "-b:a" else arg for arg in encode_args]
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        _run_ffmpeg([*command, output_path])
        return output_path
    if mode == "replace":
        if not video_path or len(audio_paths) != 1 or include_video_audio or mix_options:
            raise ValueError("replace requires a video and exactly one audio file, without mixing options.")
        audio_stream = next(stream for stream in ffmpeg.probe(audio_paths[0]).get("streams", []) if stream.get("codec_type") == "audio")
        source_codec = str(audio_stream.get("codec_name", "")).lower()
        copy_audio = output_extension == ".mkv" or (output_extension in {".mp4", ".mov"} and source_codec in {"aac", "alac"})
        command = ["-i", video_path, "-i", audio_paths[0], *subtitle_inputs, "-map", "0:v:0", "-map", "1:a:0", *subtitle_args, "-c:v", "copy", *(["-c:a", "copy"] if copy_audio else _get_mp4_audio_encode_args(audio_codec))]
        video_duration = get_media_duration(video_path)
        if video_duration is not None and video_duration > 0:
            command += ["-t", f"{video_duration:.6f}"]
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        _run_ffmpeg([*command, output_path])
        return output_path
    if len(audio_paths) + int(include_video_audio) < (1 if mode == "mix" else 2):  # a one-file mix masters it: trim, fades, gain, loudness, limiter
        raise ValueError("mix requires an audio stream." if mode == "mix" else "multitrack requires at least two audio streams.")
    def per_file(name, values, low, high, unit):
        if values is None:
            return None
        if mode != "mix" or len(values) != len(audio_paths):
            raise ValueError(f"{name} is available for mix only and needs one value per audio file.")
        values = [0.0 if value is None else float(value) for value in values]  # null keeps that file's default (no gain, offset, fade or trim)
        if any(not math.isfinite(value) or value < low or value > high for value in values):
            raise ValueError(f"Each {name} value must be a finite number from {low} to {high} {unit}.")
        return values

    gains_db, offsets_s = per_file("gains_db", gains_db, -60, 24, "dB"), per_file("offsets_s", offsets_s, 0, 3600, "seconds")
    fade_in_s, fade_out_s = per_file("fade_in_s", fade_in_s, 0, 600, "seconds"), per_file("fade_out_s", fade_out_s, 0, 600, "seconds")
    trim_start_s, trim_duration_s = per_file("trim_start_s", trim_start_s, 0, 3600, "seconds"), per_file("trim_duration_s", trim_duration_s, 0, 3600, "seconds")
    excerpt = lambda index: (trim_start_s[index] if trim_start_s else 0.0, trim_duration_s[index] if trim_duration_s else 0.0)
    automation = {}  # source index (-1 for the video's own soundtrack) -> envelopes of (time, gain dB) points on the mix timeline
    if gain_points is not None:
        if mode != "mix" or len(gain_points) != len(audio_paths):
            raise ValueError("gain_points is available for mix only and needs one entry (a list of [time_s, gain_db] points, or null) per audio file.")
        for index, points in enumerate(gain_points):
            if points:
                points = sorted((float(time), float(gain)) for time, gain in points)
                if any(not (math.isfinite(time) and math.isfinite(gain) and 0 <= time <= 3600 and -60 <= gain <= 24) for time, gain in points):
                    raise ValueError("Each gain point is [time_s, gain_db] with a time from 0 to 3600 s on the mix timeline and a gain from -60 to 24 dB.")
                automation.setdefault(index, []).append(points)
    if ducking is not None:
        from shared.deepy.audio_inspection import speech_intervals
        tracks, under = ducking.get("tracks"), ducking.get("under")
        depth, attack, release = float(ducking.get("depth_db", 10)), float(ducking.get("attack_s", 0.2)), float(ducking.get("release_s", 0.6))
        valid = lambda value: isinstance(value, int) and not isinstance(value, bool) and 0 <= value < len(audio_paths)
        own = lambda item: valid(item) or item == "video" and include_video_audio  # "video": the video's own soundtrack
        if mode != "mix" or not tracks or not under or not all(map(own, tracks)) or not all(map(own, under)) or set(tracks) & set(under):
            raise ValueError('ducking needs mix mode, "tracks" (what dips: indexes of audio_ids, or "video" with include_video_audio) and "under" (the voices: indexes of audio_ids, '
                             'or "video" for the speech of lip-synced shots in the video), with no item in both.')
        if not (1 <= depth <= 40 and 0.01 <= attack <= 5 and 0.01 <= release <= 5):
            raise ValueError("ducking depth_db must be 1..40, attack_s and release_s 0.01..5 seconds.")
        spans = sorted([*((start + (offsets_s[index] if offsets_s else 0.0), end + (offsets_s[index] if offsets_s else 0.0))
                          for index in under if index != "video" for start, end in speech_intervals(audio_paths[index], start_time=excerpt(index)[0], duration=excerpt(index)[1] or None, isolated=True)),
                        *(speech_intervals(video_path) if "video" in under else ())])  # the video's soundtrack may hold ambience: speech found against its background
        merged = []
        for start, end in spans:  # pauses shorter than a full dip and recovery keep the music down
            if merged and start - merged[-1][1] < attack + release:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        points = [point for start, end in merged for point in ((start - attack, 0.0), (start, -depth), (end, -depth), (end + release, 0.0))]
        for item in tracks:
            if points:
                automation.setdefault(-1 if item == "video" else item, []).append(points)
    if video_audio_gain_db is not None and (mode != "mix" or not include_video_audio or not -60 <= float(video_audio_gain_db) <= 24):
        raise ValueError("video_audio_gain_db needs mix mode with include_video_audio=true and a gain from -60 to 24 dB.")
    if normalize_lufs is not None and (mode != "mix" or not -40 <= float(normalize_lufs) <= -5):
        raise ValueError("normalize_lufs needs mix mode and a target from -40 to -5 LUFS.")
    expected_extension = (".mkv" if video_path else ".m4a") if mode == "multitrack" else (None if video_path else get_audio_standalone_extension(standalone_audio_codec))
    if expected_extension and os.path.splitext(output_path)[1].lower() != expected_extension:
        raise ValueError(f"{mode} output must use {expected_extension}.")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    inputs = (["-i", video_path] if video_path else [])
    for path in audio_paths:
        inputs += ["-i", path]
    inputs += subtitle_inputs
    first_audio_index = 1 if video_path else 0
    sources = (["0:a:0"] if include_video_audio else []) + [f"{first_audio_index + index}:a:0" for index in range(len(audio_paths))]
    maps = ["-map", "0:v:0", "-c:v", "copy"] if video_path else []
    if mode == "multitrack":
        for index, source in enumerate(sources):
            maps += ["-map", source]
        command = [*inputs, *maps, *subtitle_args]
        if video_path:
            command += ["-c:a", "copy"]
            video_duration = get_media_duration(video_path)
            if video_duration is not None and video_duration > 0:
                command += ["-t", f"{video_duration:.6f}"]
        else:
            for index, path in enumerate(audio_paths):
                audio_stream = next(stream for stream in ffmpeg.probe(path).get("streams", []) if stream.get("codec_type") == "audio")
                source_codec = str(audio_stream.get("codec_name", "")).lower()
                command += [f"-c:a:{index}", "copy" if source_codec in {"aac", "alac"} else "alac"]
        names = (["Original Soundtrack"] if include_video_audio else []) + [os.path.splitext(os.path.basename(path))[0] for path in audio_paths]
        for index, name in enumerate(names):
            command += [f"-metadata:s:a:{index}", f"title={name}"]
        for index, language in enumerate(audio_languages or [], start=1 if include_video_audio else 0):
            command += [f"-metadata:s:a:{index}", f"language={language}"]
    else:
        # amix would otherwise adopt the rate and layout negotiated for its first input, downsampling or downmixing richer sources
        streams = [next(stream for stream in ffmpeg.probe(path).get("streams", []) if stream.get("codec_type") == "audio") for path in ([video_path] if include_video_audio else []) + audio_paths]
        richest = max(streams, key=lambda stream: int(stream["channels"]))
        layout = richest.get("channel_layout", "unknown")
        if layout == "unknown":  # as ffmpeg itself guesses unknown layouts (guess_layout_max=2)
            layout = {1: "mono", 2: "stereo"}.get(int(richest["channels"]))
        rate = max(int(stream['sample_rate']) for stream in streams)
        audio_format = f"aformat=sample_rates={rate}" + (f":channel_layouts={layout}" if layout else "")
        filters = []
        labels = []
        for index, source in enumerate(sources):
            label, file_index = f"track{index}", index - int(include_video_audio)
            if file_index < 0:  # the video's own soundtrack
                chain = [f"volume={video_audio_gain_db or 0:g}dB", audio_format]
            else:  # the trimmed excerpt is faded on its own timeline, then the offset places it in the mix
                start, length = excerpt(file_index)
                chain = [f"atrim=start={start:g}" + (f":duration={length:g}" if length else "") + ",asetpts=PTS-STARTPTS"] if start or length else []
                chain += [f"volume={gains_db[file_index] if gains_db else 0:g}dB", audio_format]
                if fade_in_s and fade_in_s[file_index]:
                    chain.append(f"afade=t=in:st=0:d={fade_in_s[file_index]:g}")
                if fade_out_s and fade_out_s[file_index]:
                    length = min(length or math.inf, get_media_duration(audio_paths[file_index]) - start)
                    chain.append(f"afade=t=out:st={max(0.0, length - fade_out_s[file_index]):g}:d={fade_out_s[file_index]:g}")
                if offsets_s and offsets_s[file_index]:
                    chain.append(f"adelay={round(offsets_s[file_index] * 1000)}:all=1")
            if automation.get(file_index):  # 10 ms audio frames keep the gain ramps smooth
                chain += ["asetnsamples=n=480:p=0", *(f"volume='{_gain_expression(points)}':eval=frame" for points in automation[file_index])]
            filters.append(f"[{source}]{','.join(chain)}[{label}]")
            labels.append(f"[{label}]")
        # Files add at their own levels: averaging would change a file's level whenever another one starts or ends.
        filters.append(f"{''.join(labels)}amix=inputs={len(sources)}:duration=longest:normalize=0{',apad' if video_path else ''}[mixed]")
        video_duration = get_media_duration(video_path) if video_path else None
        duration_args = ["-t", f"{video_duration:.6f}"] if video_duration is not None and video_duration > 0 else []
        gain = ""
        if normalize_lufs is not None:  # one linear gain to the measured integrated loudness
            loudness = _graph_loudness([*inputs, "-filter_complex", ";".join([*filters, "[mixed]ebur128=framelog=verbose[measure]"]), "-map", "[measure]", *duration_args])
            gain = f"volume={float(normalize_lufs) - loudness:g}dB,"
        encode_args = _get_mp4_audio_encode_args(audio_codec) if video_path else _get_standalone_audio_encode_args(standalone_audio_codec)
        codec = encode_args[encode_args.index("-c:a") + 1]
        ceiling = -2 if codec.startswith("pcm_") or codec in ("flac", "alac") else -3  # AAC and MP3 encoding add about 1 dB of true-peak overshoot
        # A true-peak limiter: at 4x the rate it also catches peaks between samples, keeping the encoded file below -1 dBTP.
        filters.append(f"[mixed]{gain}aresample={4 * rate},alimiter=limit={10 ** (ceiling / 20):.4f}:level=0:latency=1,aresample={rate}[limited]")
        mixed = "[limited]"
        command = [*inputs, "-filter_complex", ";".join(filters), *maps, "-map", mixed, *subtitle_args, *encode_args, *duration_args]
    _run_ffmpeg([*command, output_path])
    return output_path


def _gain_expression(points: list[tuple[float, float]]) -> str:
    """Linear gain of a piecewise-linear dB envelope at the filter time t; the first and last gains hold outside the points.
    The caller quotes it in the filter graph."""
    expression = f"{points[-1][1]:g}"
    for (time, gain), (next_time, next_gain) in reversed(list(zip(points, points[1:]))):
        expression = f"if(lt(t,{next_time:g}),{gain:g}+({next_gain - gain:g})*(t-{time:g})/{max(next_time - time, 1e-6):g},{expression})"
    return f"pow(10,(if(lt(t,{points[0][0]:g}),{points[0][1]:g},{expression}))/20)"


def _graph_loudness(args: list[str]) -> float:
    """Integrated loudness (LUFS) of a filter graph's [measure] output, from FFmpeg's EBU R128 summary."""
    result = subprocess.run([_get_ffmpeg_path(), "-hide_banner", "-nostats", *args, "-f", "null", "-"], capture_output=True, text=True, encoding="utf-8", errors="replace", **child_kwargs())
    if result.returncode != 0:
        raise RuntimeError((result.stderr or "ffmpeg loudness measurement failed").strip())
    loudness = float(re.findall(r"\bI:\s*(-?[\d.]+)\s*LUFS", result.stderr)[-1])
    if loudness <= -70:
        raise ValueError("The mix is silent; it cannot be loudness-normalized.")
    return loudness


def resize_crop_video(source_path: str, output_path: str, *, width: int | None = None, height: int | None = None, crop_left: float = 0, crop_top: float = 0, crop_right: float = 0, crop_bottom: float = 0, crop_unit: str = "pixels", preserve_aspect_ratio: bool = True, crop_anchor: str = "center", video_codec: str | None = None, video_container: str | None = None, audio_codec: str | None = None) -> str:
    error = validate_video_output_settings(video_codec, video_container, audio_codec)
    if error is not None:
        raise ValueError(error)
    source_path = os.path.normpath(str(source_path or "").strip())
    output_path = os.path.normpath(str(output_path or "").strip())
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Video not found: {source_path}")
    stream = _probe_video_stream(source_path)
    source_width = int(stream["width"])
    source_height = int(stream["height"])
    left, top, cropped_width, cropped_height = _resolve_resize_crop_geometry(
        source_width,
        source_height,
        crop_left=crop_left,
        crop_top=crop_top,
        crop_right=crop_right,
        crop_bottom=crop_bottom,
        crop_unit=crop_unit,
        target_width=width,
        target_height=height,
        preserve_aspect_ratio=preserve_aspect_ratio,
        crop_anchor=crop_anchor,
    )
    filters = [f"crop={cropped_width}:{cropped_height}:{left}:{top}"]
    if width is not None or height is not None:
        scale_w = int(width) if width is not None else -1
        scale_h = int(height) if height is not None else -1
        if width is not None and int(width) <= 0 or height is not None and int(height) <= 0:
            raise ValueError("width and height must be > 0 when provided.")
        filters.append(f"scale={scale_w}:{scale_h}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cmd = ["-i", source_path, "-vf", ",".join(filters), "-map", "0:v:0", "-map", "0:a?"]
    cmd += get_video_encode_args(video_codec, video_container)
    cmd += _get_mp4_audio_encode_args(audio_codec)
    if str(video_container or "mp4").strip().lower() == "mp4":
        cmd += ["-movflags", "+faststart"]
    cmd += [output_path]
    _run_ffmpeg(cmd)
    return output_path


def resize_crop_image(source_path: str, output_path: str, *, width: int | None = None, height: int | None = None, crop_left: float = 0, crop_top: float = 0, crop_right: float = 0, crop_bottom: float = 0, crop_unit: str = "pixels", preserve_aspect_ratio: bool = True, crop_anchor: str = "center") -> str:
    source_path = os.path.normpath(str(source_path or "").strip())
    output_path = os.path.normpath(str(output_path or "").strip())
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Image not found: {source_path}")
    with Image.open(source_path) as image:
        source_width, source_height = image.size
        left, top, cropped_width, cropped_height = _resolve_resize_crop_geometry(
            source_width,
            source_height,
            crop_left=crop_left,
            crop_top=crop_top,
            crop_right=crop_right,
            crop_bottom=crop_bottom,
            crop_unit=crop_unit,
            target_width=width,
            target_height=height,
            preserve_aspect_ratio=preserve_aspect_ratio,
            crop_anchor=crop_anchor,
        )
        output_image = image.crop((left, top, left + cropped_width, top + cropped_height))
        if width is not None or height is not None:
            if width is not None and int(width) <= 0 or height is not None and int(height) <= 0:
                raise ValueError("width and height must be > 0 when provided.")
            target_width = int(width) if width is not None else max(1, round(cropped_width * (int(height) / cropped_height)))
            target_height = int(height) if height is not None else max(1, round(cropped_height * (int(width) / cropped_width)))
            output_image = output_image.resize((target_width, target_height), Image.Resampling.LANCZOS)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        output_ext = os.path.splitext(output_path)[1].lower()
        save_kwargs = {"quality": 95} if output_ext in {".jpg", ".jpeg", ".webp"} else {}
        if output_ext in {".jpg", ".jpeg"} and output_image.mode not in {"RGB", "L"}:
            output_image = output_image.convert("RGB")
        output_image.save(output_path, **save_kwargs)
    return output_path


def get_media_duration(path: str) -> float | None:
    probe = ffmpeg.probe(path)
    format_info = probe.get("format", {}) or {}
    duration = format_info.get("duration", None)
    try:
        return None if duration is None else float(duration)
    except Exception:
        return None


def get_precise_video_fps(path: str) -> float | None:
    try:
        stream = _probe_video_stream(path)
    except Exception:
        return None
    for key in ("avg_frame_rate", "r_frame_rate"):
        value = str(stream.get(key, "") or "").strip()
        if len(value) == 0 or value in {"0/0", "N/A"}:
            continue
        if "/" in value:
            num, den = value.split("/", 1)
            try:
                num = float(num)
                den = float(den)
                if den != 0:
                    return num / den
            except Exception:
                continue
        else:
            try:
                return float(value)
            except Exception:
                continue
    return None


def resolve_video_frame_no(path: str, *, frame_no: int | None = None, time_seconds: float | int | None = None) -> int:
    if time_seconds is None and frame_no is None:
        raise ValueError("frame_no or time_seconds is required.")
    fps, _width, _height, frame_count = get_video_info(path)
    precise_fps = get_precise_video_fps(path)
    effective_fps = float(precise_fps) if precise_fps is not None and precise_fps > 0 else float(fps)
    max_frame = max(0, int(frame_count) - 1)
    if frame_no is None:
        if effective_fps <= 0:
            frame_no = 0
        else:
            frame_no = int(math.floor(max(0.0, float(time_seconds or 0.0)) * effective_fps + 1e-9))
    frame_no = int(frame_no)
    if frame_no < 0:
        frame_no = 0
    if frame_no > max_frame:
        frame_no = max_frame
    return frame_no


def has_audio_stream(path: str) -> bool:
    return _has_audio_stream(path)


def has_video_stream(path: str) -> bool:
    try:
        _probe_video_stream(path)
        return True
    except Exception:
        return False


def _probe_video_stream(video_path: str) -> dict:
    probe = ffmpeg.probe(video_path)
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            return stream
    raise RuntimeError(f"No video stream found in {video_path}")


def _has_audio_stream(video_path: str) -> bool:
    probe = ffmpeg.probe(video_path)
    return any(stream.get("codec_type") == "audio" for stream in probe.get("streams", []))

