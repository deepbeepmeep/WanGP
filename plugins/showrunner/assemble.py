# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""The edit of a production rendered as one video: stream copy when the cut allows it, otherwise a single encode with the user's codec."""
from __future__ import annotations

import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from . import media, production as model

XFADE = {"crossfade": "fade", "fade_black": "fadeblack", "fade_white": "fadewhite"}
CEILING_DB = -2.0  # the limiter's ceiling, at 4x oversampling (inter-sample peaks included): AAC adds up to ~1 dB, the delivery stays under about -1 dBTP
DELIVERY_PEAK_DB = -1.0  # the true peak the delivered file keeps, measured after the audio codec
MAX_GAIN_DB = 20.0  # a nearly silent mix is raised this much at most: its noise floor must not become hiss
DEFAULT_VIDEO_ARGS = ["-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p"]
DEFAULT_AUDIO_ARGS = ["-c:a", "aac", "-b:a", "192k"]


def _canvas(doc: dict[str, Any], takes: list[dict[str, Any]]) -> tuple[int, int, float]:
    first = next((take for take in takes if take.get("width") and take.get("height")), None)
    width, height = (int(first["width"]), int(first["height"])) if first else (1280, 720)
    fps = float(doc["edit"].get("fps") or next((take.get("fps") for take in takes if take.get("fps")), None) or 24.0)
    return width - width % 2, height - height % 2, fps


def plan(doc: dict[str, Any], resolve) -> dict[str, Any]:
    """The clips of the edit with their source window, and whether a stream copy is possible."""
    clips = []
    for shot, take in model.sequence(doc):
        path = resolve(take["path"])
        start, end = (list(shot.get("trim") or [0.0, None]) + [None, None])[:2]
        length = take.get("duration_s") or shot.get("duration_s") or 5.0
        start = float(start or 0.0)
        end = float(end) if end is not None else model.out_point(shot, length)
        clips.append({"shot": shot["id"], "take": take["id"], "path": path, "kind": take.get("kind", "video"), "start": start, "end": max(start + 0.04, end),
                      "transition": shot.get("transition") or {"type": "cut"}, "has_audio": take.get("has_audio"), "codec": take.get("codec"),
                      "width": take.get("width"), "height": take.get("height"), "fps": take.get("fps"), "sung": bool(shot.get("audio_slice")), "dialogue": shot.get("dialogue") or [],
                      "line": (shot.get("generation") or {}).get("settings", {}).get("audio_guide"), "length": float(length),
                      "trimmed": start > 0.0 or end < float(length) - 0.01})
    edit = doc["edit"]
    if doc["brief"].get("song") and edit.get("audio_mode") in ("song", "mix"):  # the song is the clock: a transition keeps every shot on its part
        for before, clip in zip(clips, clips[1:]):
            if clip["transition"].get("type", "cut") in XFADE:  # the outgoing shot runs into it: spare frames, then its last frame held (tpad, 1 s)
                overlap = min(float(clip["transition"].get("duration_s") or 0.5), 1.0)
                before["end"] += overlap
                clip["transition"] = {**clip["transition"], "duration_s": overlap}
    copyable = bool(clips) and all(clip["kind"] == "video" and not clip["trimmed"] and clip["transition"].get("type", "cut") == "cut" for clip in clips) \
        and len({(clip["codec"], clip["width"], clip["height"], clip["fps"]) for clip in clips}) == 1 and edit.get("audio_mode") == "clips" and not edit.get("audio_tracks") and not edit.get("overlays") \
        and not edit.get("subtitles") and not any(clip["dialogue"] for clip in clips)  # subtitles need the encode
    return {"clips": clips, "copy": copyable}


def render(doc: dict[str, Any], resolve, out: str | Path, *, video_args: list[str] | None = None, audio_args: list[str] | None = None, progress=None) -> dict[str, Any]:
    """Render the edit to `out`. resolve maps document paths to files. Returns the output facts and how it was made."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    layout = plan(doc, resolve)
    clips = layout["clips"]
    if not clips:
        raise ValueError("The edit has no shot with a take yet.")
    started = time.time()
    if layout["copy"]:
        with tempfile.TemporaryDirectory() as folder:
            listing = Path(folder) / "clips.txt"
            listing.write_text("".join(f"file '{Path(clip['path']).as_posix()}'\n" for clip in clips), encoding="utf-8")
            # the picture is copied untouched; the sound gets the broadcast level of every mix
            sound = ["-c:a", "copy"]
            if all(clip["has_audio"] for clip in clips):
                gain = _gain(-16.0, _loudness([media.ffmpeg(), "-f", "concat", "-safe", "0", "-i", str(listing), "-vn", "-af", MEASURE, "-f", "null", "-"]))
                sound = ["-af", _level(gain), *(audio_args or DEFAULT_AUDIO_ARGS)]
            media._run([media.ffmpeg(), "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(listing), "-c:v", "copy", *sound, "-movflags", "+faststart", str(out)])
        method = "stream copy"
    else:
        length = sum(clip["end"] - clip["start"] for clip in clips)
        measure = _encode_command(doc, clips, resolve, out, video_args or DEFAULT_VIDEO_ARGS, audio_args or DEFAULT_AUDIO_ARGS, measure=True)
        gain = _gain(-14.0 if doc["edit"].get("audio_mode") == "song" and not doc["edit"].get("audio_tracks") else -16.0, _loudness(measure, timeout=max(120.0, 5.0 * length))) if measure else 0.0
        command = _encode_command(doc, clips, resolve, out, video_args or DEFAULT_VIDEO_ARGS, audio_args or DEFAULT_AUDIO_ARGS, gain_db=gain)
        _run_graph(command, timeout=max(300.0, 30.0 * length))  # an encode that stalls fails instead of blocking the director
        method = "single encode"
    _tame_peaks(out, audio_args or DEFAULT_AUDIO_ARGS)
    info = media.probe(out)
    return {"path": str(out), "method": method, "seconds": round(time.time() - started, 2), "clips": len(clips), **info}


# Delivery level: the mix is measured once (EBU R128 integrated loudness), then raised or lowered by one fixed gain under a limiter. A one-pass
# loudnorm rides the gain instead: it pumps the music and lifts quiet passages (a take's -65 dB noise floor became -17 dB hiss).
MEASURE = "ebur128=peak=true:framelog=quiet"


def _run_graph(command: list[str], **kwargs):
    """Run an ffmpeg command whose filter graph goes in a file: the graph of a long edit outgrows the Windows command line (32K characters)."""
    with tempfile.TemporaryDirectory() as folder:
        graph = command.index("-filter_complex")
        script = Path(folder) / "graph.txt"
        script.write_text(command[graph + 1], encoding="utf-8")
        return media._run([*command[:graph], "-filter_complex_script", str(script), *command[graph + 2:]], **kwargs)


def _true_peak(path: Path) -> float | None:
    """The true peak (dBTP) of a file's sound, None without sound."""
    result = media._run([media.ffmpeg(), "-hide_banner", "-nostats", "-v", "info", "-i", str(path), "-vn", "-af", MEASURE, "-f", "null", "-"])
    found = re.findall(r"Peak:\s+(-?[\d.]+|-inf) dBFS", result.stderr.decode("utf-8", "replace") if isinstance(result.stderr, bytes) else result.stderr)
    return float(found[-1]) if found and found[-1] != "-inf" else None


def delivery(path: Path) -> dict[str, Any]:
    """What a delivered file holds, measured: its sound and subtitle streams, integrated loudness and true peak (so that the showrunner
    checks the export from these instead of parsing the file)."""
    kinds = media._run([media.ffprobe(), "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)]).stdout.decode().split()
    report = media._run([media.ffmpeg(), "-hide_banner", "-nostats", "-v", "info", "-i", str(path), "-vn", "-af", MEASURE, "-f", "null", "-"]).stderr.decode("utf-8", "replace") \
        if "audio" in kinds else ""
    loudness, peak = re.findall(r"I:\s+(-?[\d.]+) LUFS", report), re.findall(r"Peak:\s+(-?[\d.]+) dBFS", report)
    return {"sound_streams": kinds.count("audio"), "subtitle_streams": kinds.count("subtitle"), "loudness_lufs": float(loudness[-1]) if loudness else None,
            "true_peak_dbtp": float(peak[-1]) if peak else None}


def _tame_peaks(out: Path, audio_args: list[str]) -> None:
    """The audio codec pushes the peaks of a dense master above the limiter's ceiling (AAC added 1.7 dB to a song mastered at -14 LUFS):
    the delivered file is measured, and its sound limited again by the excess when needed, with the picture and subtitles copied."""
    ceiling = CEILING_DB
    for _ in range(3):
        peak = _true_peak(out)
        if peak is None or peak <= DELIVERY_PEAK_DB:
            return
        ceiling -= peak - DELIVERY_PEAK_DB + 0.3  # each pass only tightens the ceiling of the last
        retry = out.with_name(f"{out.stem}.peaks{out.suffix}")
        media._run([media.ffmpeg(), "-y", "-v", "error", "-i", str(out), "-map", "0", "-c", "copy", "-af",
                    f"aresample=192000,alimiter=limit={10 ** (ceiling / 20):.4f}:attack=2:release=40:level=disabled,aresample=48000", *audio_args,
                    "-movflags", "+faststart", str(retry)])
        retry.replace(out)


def _loudness(command: list[str], **kwargs) -> float | None:
    """Integrated loudness (LUFS) printed by the ebur128 filter of a measuring command; None for silence."""
    result = (_run_graph if "-filter_complex" in command else media._run)([command[0], "-hide_banner", "-nostats", "-v", "info", *command[1:]], **kwargs)
    found = re.findall(r"I:\s+(-?[\d.]+) LUFS", result.stderr.decode("utf-8", "replace") if isinstance(result.stderr, bytes) else result.stderr)
    return float(found[-1]) if found and float(found[-1]) > -70.0 else None


def _gain(target: float, loudness: float | None) -> float:
    return 0.0 if loudness is None else max(-30.0, min(MAX_GAIN_DB, target - loudness))


def _level(gain: float) -> str:
    """One fixed gain, then a limiter that keeps the peaks under the ceiling without raising anything (level=disabled), run at 4x the rate so
    that peaks between samples (true peaks) are caught too."""
    return f"aresample=192000,volume={gain:.2f}dB,alimiter=limit={10 ** (CEILING_DB / 20):.4f}:attack=2:release=40:level=disabled,aresample=48000"


def _encode_command(doc, clips, resolve, out, video_args, audio_args, gain_db: float = 0.0, measure: bool = False) -> list[str] | None:
    """The single encode of the edit; measure=True: the command that measures its mixed sound alone (None without sound)."""
    width, height, fps = _canvas(doc, [{"width": clip["width"], "height": clip["height"], "fps": clip["fps"]} for clip in clips])
    # Every clip lasts whole frames, and its picture is cut to exactly that many: the sound, subtitles and text placed at the clips'
    # starts then stay on the picture (fractions of a frame left at each cut added up to 0.1 s of lip-sync drift over a 13-shot film).
    clips = [{**clip, "frames": (frames := max(1, round((clip["end"] - clip["start"]) * fps))), "end": clip["start"] + frames / fps} for clip in clips]
    inputs, filters = [], []
    for index, clip in enumerate(clips):
        duration, frames = clip["end"] - clip["start"], clip["frames"]
        if clip["kind"] == "image":  # a still gets a slow push-in (6 % over the shot), cropped to the canvas
            inputs += ["-loop", "1", "-framerate", f"{fps:g}", "-t", f"{duration + 1 / fps:.4f}", "-i", str(clip["path"])]
            source = (f"[{index}:v]scale={2 * width}:{2 * height}:force_original_aspect_ratio=increase,crop={2 * width}:{2 * height},"
                      f"zoompan=z='1+0.06*on/{frames}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={width}x{height}:fps={fps:g},")
        else:
            inputs += ["-i", str(clip["path"])]
            source = f"[{index}:v]trim=start={clip['start']:.4f},"
        filters.append(f"{source}setpts=PTS-STARTPTS,scale={width}:{height}:force_original_aspect_ratio=decrease,"
                       f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1,fps={fps:g},tpad=stop_mode=clone:stop_duration=1,"
                       f"trim=end_frame={frames},setpts=PTS-STARTPTS,settb=AVTB,format=yuv420p[v{index}]")
    # Video chain: cuts concatenate, transitions overlap the two clips by their duration.
    label, total = "v0", clips[0]["end"] - clips[0]["start"]
    offsets = [0.0]
    for index in range(1, len(clips)):
        clip = clips[index]
        duration = clip["end"] - clip["start"]
        transition = clip["transition"].get("type", "cut")
        overlap = min(float(clip["transition"].get("duration_s") or 0.5), duration / 2, total / 2) if transition in XFADE else 0.0
        target = f"x{index}"
        if transition in XFADE and overlap > 0.04:
            filters.append(f"[{label}][v{index}]xfade=transition={XFADE[transition]}:duration={overlap:.3f}:offset={total - overlap:.3f}[{target}]")
            offsets.append(total - overlap)
            total += duration - overlap
        else:
            filters.append(f"[{label}][v{index}]concat=n=2:v=1:a=0[{target}]")
            offsets.append(total)
            total += duration
        label = target
    first_audio = len(filters)
    audio_label = _audio_chain(doc, clips, resolve, inputs, filters, offsets, total, None if measure else gain_db)
    if measure:  # the sound alone: no picture is decoded or filtered
        return [media.ffmpeg(), *inputs, "-filter_complex", ";".join(filters[first_audio:] + [f"[{audio_label}]{MEASURE}[measured]"]), "-map", "[measured]", "-f", "null", "-"] if audio_label else None
    label = _overlays(doc, resolve, inputs, filters, label, width, height, fps, total)
    command = [media.ffmpeg(), "-y", "-v", "error", *inputs, "-filter_complex", ";".join(filters), "-map", f"[{label}]"]
    if audio_label:
        command += ["-map", f"[{audio_label}]", *audio_args]
    subtitles = [item for item in doc["edit"].get("subtitles", []) if item.get("path") and resolve(item["path"]).is_file()]
    if not subtitles and any(clip["dialogue"] for clip in clips):  # the shots' lines, timed on this cut (an SRT of the edit's own replaces them)
        srt = out.with_suffix(".srt")
        srt.write_text(dialogue_srt(clips, offsets, resolve), encoding="utf-8")
        subtitles = [{"path": str(srt), "language": LANGUAGES.get(str(doc["brief"].get("language") or "en")[:2].lower(), "und")}]
    first_subtitle = inputs.count("-i")
    command[command.index("-filter_complex"):command.index("-filter_complex")] = [arg for item in subtitles for arg in ("-i", str(resolve(item["path"])))]
    for offset, item in enumerate(subtitles):
        command += ["-map", f"{first_subtitle + offset}:s", f"-metadata:s:s:{offset}", f"language={item.get('language', 'eng')}", f"-disposition:s:{offset}", "0"]  # the viewer turns them on
    if subtitles:
        command += ["-c:s", "mov_text" if out.suffix.lower() in (".mp4", ".mov", ".m4v") else "srt"]
    return [*command, *video_args, "-r", f"{fps:g}", "-t", f"{total:.3f}", "-movflags", "+faststart", str(out)]


LANGUAGES = {"en": "eng", "fr": "fra", "de": "deu", "es": "spa", "it": "ita", "pt": "por", "nl": "nld", "ja": "jpn", "zh": "zho", "ko": "kor", "ru": "rus", "ar": "ara", "hi": "hin"}


def dialogue_srt(clips, offsets, resolve) -> str:
    """Subtitles of the shots' dialogue at their place in this cut: a line starts with its shot and lasts its recorded length (the shot's
    length without one); several lines share their shot by their number of characters."""
    def stamp(seconds):
        milliseconds = int(round(seconds * 1000))
        return f"{milliseconds // 3600000:02d}:{milliseconds // 60000 % 60:02d}:{milliseconds // 1000 % 60:02d},{milliseconds % 1000:03d}"
    cues = []
    for clip, offset in zip(clips, offsets):
        lines = [item for item in clip["dialogue"] if item.get("line", "").strip()]
        if not lines:
            continue
        shown = clip["end"] - clip["start"]
        recorded = media.probe(resolve(clip["line"]))["duration_s"] if clip.get("line") and resolve(clip["line"]).is_file() else None
        span = min(shown, recorded + 0.3) if recorded and len(lines) == 1 else shown
        total = sum(len(item["line"]) for item in lines)
        start = offset
        for item in lines:
            end = start + span * len(item["line"]) / total
            cues.append((start, end - 0.05, item["line"].strip()))
            start = end
    return "".join(f"{number}\n{stamp(start)} --> {stamp(end)}\n{text}\n\n" for number, (start, end, text) in enumerate(cues, 1))


def _overlays(doc, resolve, inputs, filters, label, width, height, fps, total) -> str:
    """Text and pictures laid over the cut, each over its own time range with a fade in and out (after the audio inputs, whose indexes are fixed)."""
    for number, overlay in enumerate(doc["edit"].get("overlays", [])):
        start, end = float(overlay["start_s"]), min(float(overlay["end_s"]), total)
        if end - start < 0.05 or not overlay.get("path") or not resolve(overlay["path"]).is_file():
            continue
        duration = end - start
        fade = min(float(overlay.get("fade_s") or 0.0), duration / 3)
        index = inputs.count("-i")
        inputs += ["-loop", "1", "-framerate", f"{fps:g}", "-t", f"{duration:.3f}", "-i", str(resolve(overlay["path"]))]
        fades = f",fade=t=in:st=0:d={fade:.3f}:alpha=1,fade=t=out:st={duration - fade:.3f}:d={fade:.3f}:alpha=1" if fade > 0.01 else ""
        filters.append(f"[{index}:v]format=rgba,scale={width}:{height}{fades},setpts=PTS-STARTPTS+{start:.3f}/TB[o{number}]")
        filters.append(f"[{label}][o{number}]overlay=eof_action=pass,format=yuv420p[vo{number}]")
        label = f"vo{number}"
    return label


def _audio_chain(doc, clips, resolve, inputs, filters, offsets, total, gain_db) -> str | None:
    edit, parts = doc["edit"], []
    mode = edit.get("audio_mode", "clips")
    index = len(clips)
    if mode in ("clips", "mix"):
        for position, clip in enumerate(clips):
            duration = clip["end"] - clip["start"]
            delay = int(offsets[position] * 1000)
            if clip["kind"] == "video" and clip["has_audio"] and not (mode == "mix" and clip["sung"] and doc["brief"].get("song")):  # the song already sings it
                filters.append(f"[{position}:a]atrim=start={clip['start']:.3f}:end={clip['end']:.3f},asetpts=PTS-STARTPTS,aresample=48000,adelay={delay}|{delay}[a{position}]")
                parts.append((f"a{position}", 1.0))
    song = doc["brief"].get("song")
    if mode in ("song", "mix") and song and resolve(song).is_file():
        inputs += ["-i", str(resolve(song))]
        start = float(edit.get("song_start_s") or 0.0)
        fade = min(2.0, total / 4) if start + total < (media.probe(resolve(song))["duration_s"] or 0.0) - 0.05 else 0.0  # fade out only a song the edit cuts short
        filters.append(f"[{index}:a]atrim=start={start:.3f}:end={start + total:.3f},asetpts=PTS-STARTPTS,aresample=48000" + (f",afade=t=out:st={total - fade:.3f}:d={fade:.3f}" if fade else "") + "[song]")
        parts.append(("song", 1.0 if mode == "song" else 0.5))
        index += 1
    for number, track in enumerate(edit.get("audio_tracks", [])):
        if not track.get("path") or not resolve(track["path"]).is_file():
            continue
        inputs += ["-i", str(resolve(track["path"]))]
        delay = int(float(track.get("start_s") or 0.0) * 1000)
        filters.append(f"[{index}:a]asetpts=PTS-STARTPTS,aresample=48000,adelay={delay}|{delay}[t{number}]")
        parts.append((f"t{number}", 10 ** (float(track.get("gain_db") or 0.0) / 20)))
        index += 1
    if not parts:
        return None
    # the delivery level (render measures the mix first): a song alone the music level, a mix with speech the broadcast one
    level = "" if gain_db is None else "," + _level(gain_db)
    if len(parts) == 1:
        filters.append(f"[{parts[0][0]}]volume={parts[0][1]:.3f}{level},apad=whole_dur={total:.3f},atrim=end={total:.3f}[aout]")
    else:
        weights = " ".join(f"{weight:.3f}" for _, weight in parts)
        filters.append("".join(f"[{name}]" for name, _ in parts) + f"amix=inputs={len(parts)}:duration=longest:normalize=0:weights={weights}{level},apad=whole_dur={total:.3f},atrim=end={total:.3f}[aout]")
    return "aout"
