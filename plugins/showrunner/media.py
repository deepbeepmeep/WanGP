# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Media facts for the Showrunner UI, all on the CPU: probing, poster frames, waveforms, and the beats and sections of a song."""
from __future__ import annotations

import hashlib
import os
import json
import shutil
import subprocess
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from shared.utils.video_decode import resolve_media_binary

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus"}
ANALYSIS_VERSION = 3  # part of the song analysis cache key: bump when analyze_song changes


def kind_of(path: str | Path) -> str | None:
    suffix = Path(path).suffix.lower()
    return "image" if suffix in IMAGE_EXTENSIONS else "video" if suffix in VIDEO_EXTENSIONS else "audio" if suffix in AUDIO_EXTENSIONS else None


def _run(arguments: list[str], **kwargs) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(arguments, capture_output=True, check=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), **kwargs)
    except subprocess.CalledProcessError as error:  # the tool's own last words say what went wrong, not its exit status
        stderr = error.stderr.decode("utf-8", "replace") if isinstance(error.stderr, bytes) else str(error.stderr or "")
        lines = [line for line in stderr.splitlines() if line.strip()]
        raise RuntimeError(f"{Path(arguments[0]).stem} failed: {' | '.join(lines[:3]) or f'exit status {error.returncode}'}") from None


@lru_cache(maxsize=1)
def ffmpeg() -> str:
    return resolve_media_binary("ffmpeg")


@lru_cache(maxsize=1)
def ffprobe() -> str:
    return resolve_media_binary("ffprobe")


def probe(path: str | Path) -> dict[str, Any]:
    """Duration, size, frame rate and audio presence of a media file (images: size only)."""
    path = Path(path)
    kind = kind_of(path)
    if kind == "image":
        from PIL import Image
        from shared.deepy.image_channels import upright_size
        with Image.open(path) as image:
            width, height = upright_size(image)  # a phone photo as it is shown: its EXIF orientation applied
            return {"kind": "image", "width": width, "height": height, "duration_s": None, "has_audio": False}
    output = _run([ffprobe(), "-v", "error", "-show_entries", "format=duration:stream=codec_type,codec_name,width,height,avg_frame_rate,r_frame_rate,duration", "-of", "json", str(path)]).stdout
    data = json.loads(output or b"{}")
    streams = data.get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    length = (video or {}).get("duration") or data.get("format", {}).get("duration")  # a video's picture: its audio stream may run longer than its frames
    info = {"kind": kind or ("video" if video else "audio"), "duration_s": round(float(length or 0.0), 3) or None,
            "has_audio": any(stream.get("codec_type") == "audio" for stream in streams)}
    if video:
        rate = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1"
        numerator, _, denominator = rate.partition("/")
        info.update(width=video.get("width"), height=video.get("height"), codec=video.get("codec_name"), fps=round(float(numerator) / float(denominator or 1), 3) if float(denominator or 1) else None)
    return info


def generation_time(path: str | Path) -> float | None:
    """The seconds WanGP took to generate a video, as it records them in the file's metadata (None for other files)."""
    path = Path(path)
    if kind_of(path) != "video" or not path.is_file():
        return None
    stat = path.stat()
    return _generation_time(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=4096)
def _generation_time(path: str, mtime_ns: int, size: int) -> float | None:
    from shared.utils.video_metadata import read_metadata_from_video
    metadata = read_metadata_from_video(path)
    return float(metadata["generation_time"]) if isinstance(metadata, dict) and metadata.get("generation_time") else None


def poster(path: str | Path, out: str | Path, *, at_s: float | None = None, width: int = 640) -> Path:
    """A JPEG frame of a video (a third of the way in by default) or a resized copy of an image."""
    path, out = Path(path), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if kind_of(path) == "image":
        from PIL import Image
        from shared.deepy.image_channels import upright
        with Image.open(path) as image:
            image = upright(image).convert("RGB")
            image.thumbnail((width, width * 4))
            image.save(out, quality=88)
        return out
    if at_s is None:
        duration = probe(path).get("duration_s") or 0.0
        at_s = duration / 3
    _run([ffmpeg(), "-y", "-v", "error", "-ss", f"{max(0.0, at_s):.3f}", "-i", str(path), "-frames:v", "1", "-vf", f"scale={width}:-2", "-q:v", "3", str(out)])
    return out


def filmstrip(path: str | Path, out: str | Path, *, frames: int = 8, height: int = 90) -> Path:
    """One image holding `frames` evenly spaced frames of a video side by side (timeline thumbnails)."""
    path, out = Path(path), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    duration = probe(path).get("duration_s") or 1.0
    _run([ffmpeg(), "-y", "-v", "error", "-i", str(path), "-vf", f"fps={frames / max(duration, 0.1):.5f},scale=-2:{height},tile={frames}x1", "-frames:v", "1", "-q:v", "4", str(out)])
    return out


def subtitles_vtt(path: str | Path) -> str:
    """The first subtitle track of a video as WebVTT, the format browsers show ("" without one)."""
    tracks = _run([ffprobe(), "-v", "error", "-select_streams", "s", "-show_entries", "stream=index", "-of", "csv=p=0", str(path)]).stdout.split()
    return _run([ffmpeg(), "-v", "error", "-i", str(path), "-map", "0:s:0", "-f", "webvtt", "-"]).stdout.decode("utf-8", "replace") if tracks else ""


def subtitle_cues(path: str | Path) -> list[tuple[float, float, str]]:
    """The cues of a video's first subtitle track: (start_s, end_s, text)."""
    seconds = lambda stamp: sum(float(part) * 60 ** index for index, part in enumerate(reversed(stamp.split(":"))))
    cues = []
    for block in subtitles_vtt(path).replace("\r", "").split("\n\n"):
        lines = block.strip().split("\n")
        timing = next((index for index, line in enumerate(lines) if " --> " in line), None)
        if timing is not None:
            start, end = lines[timing].split(" --> ")
            cues.append((seconds(start.strip()), seconds(end.split()[0]), " ".join(lines[timing + 1:]).strip()))
    return cues


NEAR_RATE = 0.1  # frames per second: 23.976 for 24, 29.97 for 30
OWN_RATES = (23.5, 30.5)  # keep their frames at their rate: a real render at 25 and at 30 matches its render at 24 (43 and 41 dB). Twice that would cost twice the frames for the same seconds


def _rates(path: str | Path) -> list[float]:
    found = _run([ffprobe(), "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=avg_frame_rate,r_frame_rate", "-of", "csv=p=0", str(path)]).stdout.decode("ascii", "replace").strip().split("\n")[0].split(",")
    return [float(numerator) / float(denominator or 1) if numerator and float(denominator or 1) else 0.0 for numerator, _, denominator in (rate.partition("/") for rate in found)]


def frame_rate(path: str | Path) -> float:
    """A video's real frame rate, 23.976023976 and not a rounded 24."""
    return _rates(path)[-1]


def _steady_rate(path: str | Path) -> bool:
    rates = _rates(path)
    return max(rates) - min(rates) < 0.01


def normalized_video(path: str | Path, fps: float, out_folder: str | Path) -> Path:
    """A remake's source, whose frames are the numbers the plan speaks in. A video at a steady rate near the model's is used as it is, frame for frame, at its real rate: WanGP reads, renders, times its audio
    and records at that rate (force_fps). Another (a higher rate, a variable one) is converted to the model's rate once and reused."""
    if (abs(frame_rate(path) - fps) < NEAR_RATE or OWN_RATES[0] <= frame_rate(path) <= OWN_RATES[1]) and _steady_rate(path):
        return Path(path)
    out = Path(out_folder) / f"{cache_key(path, fps)}.mp4"
    if not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        _run([ffmpeg(), "-y", "-v", "error", "-i", str(path), "-vf", f"fps={fps:g}", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k", str(out)])
    return out


def video_subtitles(path: str | Path, out_folder: str | Path) -> list[dict[str, Any]]:
    """The text subtitle tracks of a video as SRT files with their languages, converted once and reused. Subtitles drawn as pictures are left out."""
    streams = json.loads(_run([ffprobe(), "-v", "error", "-select_streams", "s", "-show_entries", "stream=codec_name:stream_tags=language", "-of", "json", str(path)]).stdout or b"{}").get("streams", [])
    tracks = []
    for number, stream in enumerate(streams):
        if stream.get("codec_name") in ("mov_text", "subrip", "srt", "ass", "ssa", "webvtt", "text"):
            out = Path(out_folder) / f"{Path(path).stem}-subtitles-{number}.srt"
            if not out.is_file():
                out.parent.mkdir(parents=True, exist_ok=True)
                _run([ffmpeg(), "-y", "-v", "error", "-i", str(path), "-map", f"0:s:{number}", "-f", "srt", str(out)])
            tracks.append({"path": out, "language": (stream.get("tags") or {}).get("language") or "und"})
    return tracks


def video_window(path: str | Path, start: int, frames: int, fps: float, out_folder: str | Path, size: str | None = None, hold: tuple[int, int] = (0, 0)) -> Path:
    """Exactly `frames` frames of a video from frame `start`, with their sound, cut once and reused, for the page's players: nothing delivered reads it. size: "WIDTHxHEIGHT" to scale them to.
    hold: how many of its first and last frames are replaced by the frame next to them (a render's cut that landed a frame or two off: cut_flash)."""
    out = Path(out_folder) / f"{Path(path).stem}-{start}-{frames}{f'-{size}' if size else ''}{f'-hold{hold[0]}-{hold[1]}' if any(hold) else ''}.mp4"
    if not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        scale = f",scale={size.replace('x', ':')}" if size else ""
        held = f",trim=start_frame={hold[0]}:end_frame={frames - hold[1]},setpts=PTS-STARTPTS,tpad=start={hold[0]}:start_mode=clone:stop={hold[1]}:stop_mode=clone" if any(hold) else ""
        # inputs that end with the part: nothing before it is decoded, and half a frame early lands the seek on its first frame
        inputs = ["-ss", f"{max(0.0, start - 0.5) / fps:.6f}", "-t", f"{(frames + 2) / fps * 1.01:.6f}", "-i", str(path), "-ss", f"{start / fps:.6f}", "-t", f"{frames / fps:.6f}", "-i", str(path)]
        making = out.with_name(f"{out.stem}.{uuid.uuid4().hex[:8]}.mp4")  # another request for this file, a mouse over a card and its player, must not read it half written
        _run([ffmpeg(), "-y", "-v", "error", *inputs, "-filter_complex", f"[0:v]trim=end_frame={frames},setpts=PTS-STARTPTS{held}{scale}[picture]", "-map", "[picture]", "-map", "1:a?", "-frames:v", str(frames),
              "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", "-r", f"{fps:.6f}", "-c:a", "aac", "-b:a", "256k", str(making)])
        os.replace(making, out)
    return out


def cut_flash(path: str | Path, cut: int, fps: float) -> tuple[int, int]:
    """At a cut inside a render (`cut`: the new shot's first frame), the frames on the wrong side of it, as the Viggle Remake MCP server's deflash finds them:
    of the three frames before the cut, how many look like the shot after it (the cut came early), and of the three from it, how many look like the shot
    before (it came late). Each frame is compared with the frames 4 before and 3 after the cut, at 160x90. Upstream holds two; Skate Queen had a cut 3
    frames early."""
    data = _run([ffmpeg(), "-v", "error", "-i", str(path), "-vf", f"select=between(n\\,{cut - 4}\\,{cut + 3}),scale=160:90", "-fps_mode", "passthrough", "-frames:v", "8",  # by index: a seek can land a frame off
                 "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]).stdout
    frames = np.frombuffer(data, dtype=np.uint8).reshape(-1, 90, 160, 3).astype(np.float32)
    if len(frames) < 8 or cut < 4:
        return 0, 0
    before = lambda frame: np.abs(frame - frames[0]).mean() < np.abs(frame - frames[7]).mean()  # noqa: E731
    early = next((count for count, index in enumerate((3, 2, 1)) if before(frames[index])), 3)  # frames cut-1, cut-2, cut-3
    late = next((count for count, index in enumerate((4, 5, 6)) if not before(frames[index])), 3)  # frames cut, cut+1, cut+2
    return early, late


def changed_share(before: str | Path, after: str | Path) -> float:
    """The share of a picture that really differs from another, past the small shifts and tones of a redraw: colours at 64x48, more than 30 of 255 apart."""
    from PIL import Image

    pictures = [np.asarray(Image.open(path).convert("RGB").resize((64, 48), Image.BOX), dtype=np.float32) for path in (before, after)]
    return float((np.abs(pictures[0] - pictures[1]).max(axis=2) > 30).mean())


def fitted_canvas(path: str | Path, resolution: str, pixels: int | None = None) -> str:
    """The size a video's shape takes at a resolution of the brief, on the 32-pixel grid: a wider video keeps its shape below the model's largest size (pixels)."""
    facts = probe(path)
    short = {"480p": 480, "720p": 736}.get(resolution, 736)
    wide, tall = (facts["width"], facts["height"]) if facts["width"] >= facts["height"] else (facts["height"], facts["width"])
    long = round(short * wide / tall / 32) * 32
    while pixels and short * long > pixels:
        short -= 32
        long = round(short * wide / tall / 32) * 32
    return f"{long}x{short}" if facts["width"] >= facts["height"] else f"{short}x{long}"


def generation_seed(path: str | Path) -> int | None:
    """The seed a generated video was made with, from the settings WanGP keeps in its file."""
    found = _run([ffprobe(), "-v", "error", "-show_entries", "format_tags=comment", "-of", "default=nw=1:nk=1", str(path)]).stdout.decode("utf-8", "replace").strip()
    try:
        return int(json.loads(found)["seed"])
    except (ValueError, KeyError, TypeError):  # a file from elsewhere, dropped on the shot
        return None


def video_frame(path: str | Path, frame: int, out_folder: str | Path) -> Path:
    """One frame of a video as a PNG, cut once and reused."""
    out = Path(out_folder) / f"{Path(path).stem}-frame{frame}.png"
    if not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        _run([ffmpeg(), "-y", "-v", "error", "-i", str(path), "-vf", f"select=eq(n\\,{frame})", "-frames:v", "1", "-fps_mode", "passthrough", str(out)])
    return out


def fit_picture(path: str | Path, size: str, out_folder: str | Path) -> Path:
    """A picture at a size ("1280x704"), cropped from its centre to that shape when it has another: a painted frame from elsewhere takes the render's size."""
    from PIL import Image, ImageOps
    width, height = (int(value) for value in size.split("x"))
    out = Path(out_folder) / f"{Path(path).stem}-{width}x{height}.png"
    with Image.open(path) as picture:
        ImageOps.fit(ImageOps.exif_transpose(picture).convert("RGB"), (width, height), Image.LANCZOS).save(out)
    return out


def audio_slice(path: str | Path, start_s: float, end_s: float, out_folder: str | Path) -> Path:
    """The [start_s, end_s] excerpt of an audio file as a WAV in out_folder, cut once and reused (a shot's driving audio)."""
    out = Path(out_folder) / f"{Path(path).stem}-{start_s:.3f}-{end_s:.3f}.wav"
    if not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        _run([ffmpeg(), "-y", "-v", "error", "-ss", f"{start_s:.3f}", "-t", f"{end_s - start_s:.3f}", "-i", str(path), "-vn", "-c:a", "pcm_s16le", str(out)])
    return out


def take_sound(path: str | Path, speed: float, out_folder: str | Path, tempo: str = "") -> Path:
    """The whole sound of a take as a WAV in out_folder, at the speed it plays in the edit (tempo: its filters), made once."""
    out = Path(out_folder) / f"{Path(path).stem}-sound{'' if speed == 1.0 else f'-{speed:g}x'}.wav"
    if not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        _run([ffmpeg(), "-y", "-v", "error", "-i", str(path), "-vn", *(["-af", tempo] if tempo else []), "-c:a", "pcm_s16le", str(out)])
    return out


def padded_audio(path: str | Path, duration_s: float, out_folder: str | Path) -> Path:
    """An audio file followed by silence up to duration_s, as a WAV in out_folder (a short line driving a longer lip-synced shot)."""
    out = Path(out_folder) / f"{Path(path).stem}-to-{duration_s:.3f}.wav"
    if not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        _run([ffmpeg(), "-y", "-v", "error", "-i", str(path), "-vn", "-af", f"apad=whole_dur={duration_s:.3f}", "-c:a", "pcm_s16le", str(out)])
    return out


def joined_audio(paths: list[str | Path], out_folder: str | Path, gap_s: float = 0.3) -> Path:
    """Audio files one after the other with gap_s of silence between them, as a WAV in out_folder (the lines of a shot driving it together)."""
    key = hashlib.sha1(json.dumps([[str(path), Path(path).stat().st_mtime_ns] for path in paths] + [gap_s]).encode("utf-8")).hexdigest()[:12]
    out = Path(out_folder) / f"lines-{key}.wav"
    if not out.is_file():
        out.parent.mkdir(parents=True, exist_ok=True)
        inputs = [arg for path in paths for arg in ("-i", str(path))]
        chains = [f"[{index}:a]aresample=48000,aformat=channel_layouts=mono" + (f",apad=pad_dur={gap_s:.3f}" if index < len(paths) - 1 else "") + f"[l{index}]" for index in range(len(paths))]
        graph = ";".join(chains) + ";" + "".join(f"[l{index}]" for index in range(len(paths))) + f"concat=n={len(paths)}:v=0:a=1[out]"
        _run([ffmpeg(), "-y", "-v", "error", *inputs, "-filter_complex", graph, "-map", "[out]", "-c:a", "pcm_s16le", str(out)])
    return out


def _decode_mono(path: str | Path, rate: int) -> np.ndarray:
    raw = _run([ffmpeg(), "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(rate), "-f", "f32le", "-"]).stdout
    return np.frombuffer(raw, dtype=np.float32)


def peaks(path: str | Path, buckets: int = 1600) -> dict[str, Any]:
    """Waveform envelope: per bucket the peak and RMS of the mono mix, normalized to the loudest peak."""
    rate = 8000
    samples = _decode_mono(path, rate)
    if samples.size == 0:
        return {"duration_s": 0.0, "peak": [], "rms": []}
    buckets = max(1, min(buckets, samples.size))
    edges = np.linspace(0, samples.size, buckets + 1).astype(np.int64)
    peak = np.maximum.reduceat(np.abs(samples), edges[:-1])
    rms = np.sqrt(np.add.reduceat(samples * samples, edges[:-1]) / np.maximum(np.diff(edges), 1))
    scale = float(peak.max()) or 1.0
    return {"duration_s": round(samples.size / rate, 3), "peak": np.round(peak / scale, 4).tolist(), "rms": np.round(rms / scale, 4).tolist()}


def analyze_song(path: str | Path) -> dict[str, Any]:
    """Tempo, beats, bars (every fourth beat), an energy curve and sections (repeated parts share a letter) of a song."""
    import librosa

    rate = 22050
    samples = _decode_mono(path, rate)
    duration = samples.size / rate
    if duration < 1.0:
        return {"duration_s": round(duration, 3), "bpm": None, "beats": [], "bars": [], "energy": [], "sections": []}
    hop = 512
    onset = librosa.onset.onset_strength(y=samples, sr=rate, hop_length=hop)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset, sr=rate, hop_length=hop, units="frames")
    beats = librosa.frames_to_time(beat_frames, sr=rate, hop_length=hop)
    # Bars: the beat phase whose onsets are strongest is taken as the downbeat.
    phase = int(np.argmax([onset[beat_frames[offset::4]].mean() if len(beat_frames[offset::4]) else 0 for offset in range(4)])) if len(beat_frames) >= 8 else 0
    bars = beats[phase::4]
    rms = librosa.feature.rms(y=samples, hop_length=hop)[0]
    points = 400
    energy_edges = np.linspace(0, rms.size, points + 1).astype(np.int64)
    energy = np.add.reduceat(rms, energy_edges[:-1]) / np.maximum(np.diff(energy_edges), 1)
    energy = energy / (energy.max() or 1.0)
    return {"duration_s": round(duration, 3), "bpm": round(float(np.atleast_1d(tempo)[0]), 1), "beats": np.round(beats, 3).tolist(), "bars": np.round(bars, 3).tolist(),
            "energy": np.round(energy, 4).tolist(), "sections": _sections(samples, rate, hop, duration, beats, energy) if duration >= 12 else []}


def _sections(samples, rate, hop, duration, beats, energy) -> list[dict[str, Any]]:
    """Foote novelty on a self-similarity matrix of half-second frames (chroma, timbre, loudness), boundaries refined to the beat
    with the strongest before/after contrast; sections whose sound matches share a letter, and the loudest repeated letter is the chorus."""
    import librosa
    import scipy.signal

    def standard(matrix):
        return (matrix - matrix.mean(axis=1, keepdims=True)) / (matrix.std(axis=1, keepdims=True) + 1e-6)

    features = np.vstack([standard(librosa.feature.chroma_cqt(y=samples, sr=rate, hop_length=hop)), standard(librosa.feature.mfcc(y=samples, sr=rate, hop_length=hop, n_mfcc=13)),
                          2 * standard(np.log(librosa.feature.rms(y=samples, hop_length=hop) + 1e-5))])
    frame = 0.5
    per = max(1, int(round(frame * rate / hop)))
    count = features.shape[1] // per
    grid = features[:, :count * per].reshape(features.shape[0], count, per).mean(axis=2)
    unit = grid / (np.linalg.norm(grid, axis=0, keepdims=True) + 1e-9)
    similarity = unit.T @ unit
    half = 12
    window = scipy.signal.windows.gaussian(2 * half, half / 2)
    kernel = np.outer(window, window) * np.block([[np.ones((half, half)), -np.ones((half, half))], [-np.ones((half, half)), np.ones((half, half))]])
    padded = np.pad(similarity, half, mode="edge")
    novelty = np.maximum(np.array([np.sum(padded[index:index + 2 * half, index:index + 2 * half] * kernel) for index in range(count)]), 0)
    novelty /= novelty.max() or 1.0
    peaks, _ = scipy.signal.find_peaks(novelty, distance=int(6 / frame), height=0.3, prominence=0.1)

    def contrast(time):
        center = int(time / frame)
        before, after = grid[:, max(0, center - 4):center], grid[:, center:center + 4]
        return float(np.linalg.norm(before.mean(axis=1) - after.mean(axis=1))) if before.size and after.size else 0.0

    bounds = []
    for peak in peaks:
        time = peak * frame
        candidates = [beat for beat in beats if abs(beat - time) <= 1.5] or [time]
        bounds.append(max(candidates, key=contrast))
    bounds = [0.0] + sorted(round(float(value), 3) for value in bounds if 3.0 < value < duration - 3.0) + [round(duration, 3)]
    spans = list(zip(bounds[:-1], bounds[1:]))
    # A repeated part plays the same chords in the same order: compare sections along the diagonals of their chroma self-similarity block
    # (best lag over at least 60 % of the shorter one), then group them by average linkage. Averaged features miss repeats, chords change inside a part.
    chroma = grid[:12] / (np.linalg.norm(grid[:12], axis=0, keepdims=True) + 1e-9)
    frames = [(min(int(start / frame), count - 1), min(count, max(int(start / frame) + 1, int(end / frame)))) for start, end in spans]

    def repeat(first, second):
        block = chroma[:, frames[first][0]:frames[first][1]].T @ chroma[:, frames[second][0]:frames[second][1]]
        shortest = min(block.shape)
        return max((float(np.diagonal(block, offset).mean()) for offset in range(-block.shape[0] + 1, block.shape[1]) if len(np.diagonal(block, offset)) >= 0.6 * shortest), default=-1.0)

    scores = np.array([[repeat(first, second) if first != second else 1.0 for second in range(len(spans))] for first in range(len(spans))])
    groups = [[index] for index in range(len(spans))]
    while len(groups) > 1:
        linkage, left, right = max((float(scores[np.ix_(groups[i], groups[j])].mean()), i, j) for i in range(len(groups)) for j in range(i + 1, len(groups)))
        if linkage < 0.6:
            break
        groups[left] += groups.pop(right)
    groups.sort(key=min)
    labels = [chr(ord("A") + next(number for number, group in enumerate(groups) if index in group)) for index in range(len(spans))]
    points = len(energy)
    loudness = [float(energy[int(start / duration * points):max(int(start / duration * points) + 1, int(end / duration * points))].mean()) for start, end in spans]
    repeated = {label for label in labels if labels.count(label) > 1}
    chorus = max(repeated, key=lambda label: np.mean([value for value, item in zip(loudness, labels) if item == label])) if repeated else None
    verse = next((label for label in labels if label in repeated and label != chorus), None)  # the first other repeated part; later ones are refrains
    sections = []
    for index, ((start, end), label) in enumerate(zip(spans, labels)):
        if label == chorus:
            name = "Chorus"
        elif index == len(spans) - 1 and len(spans) > 2 and (label not in repeated or loudness[index] < 0.6 * max(loudness)):
            name = "Outro"
        elif label in repeated:
            name = "Verse" if label == verse else "Refrain"
        elif index == 0 and len(spans) > 1:
            name = "Intro"
        else:
            name = "Bridge" if len(spans) > 1 else "Song"
        sections.append({"start_s": start, "end_s": end, "label": label, "name": name, "energy": round(loudness[index], 3)})
    return sections


def import_file(source: str | Path, media_folder: str | Path, *, move: bool = False, name: str | None = None) -> Path:
    """Copy (or move) a file into a production's media folder under a unique, readable name (name: the original file name)."""
    source, media_folder = Path(source), Path(media_folder)
    media_folder.mkdir(parents=True, exist_ok=True)
    original = Path(name or source.name)
    stem = "".join(char if char.isalnum() or char in "-_" else "_" for char in original.stem)[:48] or "media"
    target = media_folder / f"{stem}-{uuid.uuid4().hex[:6]}{original.suffix.lower()}"
    (shutil.move if move else shutil.copy2)(str(source), str(target))
    return target


def cache_key(path: str | Path, *parts: Any) -> str:
    path = Path(path)
    stat = path.stat()
    return hashlib.sha1(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}|{parts}".encode()).hexdigest()[:20]
