# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""The edit of a production rendered as one video: stream copy when the cut allows it, otherwise a single encode with the user's codec."""
from __future__ import annotations

import re
import statistics
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
WORD_CUT_FADE_S = 0.1  # the sound where words outside a shot's lines were cut: a word or a breath must not stop dead
VOICE_GAIN_DB = 15.0  # how far Even Voices moves a shot at most: past it, what the viewer hears is the shot's noise floor
DEFAULT_VIDEO_ARGS = ["-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p"]
DEFAULT_AUDIO_ARGS = ["-c:a", "aac", "-b:a", "192k"]


def _canvas(doc: dict[str, Any], takes: list[dict[str, Any]]) -> tuple[int, int, float]:
    """The edit is made at the size of its takes: a draft edit stays a draft, and a mix takes the largest. A part kept from the source has no size of its own."""
    sized = [take for take in takes if take.get("width") and take.get("height")]
    largest = max([take for take in sized if not take.get("kept")] or sized, key=lambda take: int(take["width"]) * int(take["height"]), default=None)
    width, height = (int(largest["width"]), int(largest["height"])) if largest else (1280, 720)
    fps = float(doc["edit"].get("fps") or next((take.get("fps") for take in takes if take.get("fps")), None) or 24.0)
    return width - width % 2, height - height % 2, fps


def plan(doc: dict[str, Any], resolve) -> dict[str, Any]:
    """The clips of the edit with their source window, and whether a stream copy is possible."""
    clips = []
    for shot, take in model.sequence(doc):
        kept = bool(take.get("kept_source"))  # read from the source the cuts were measured on, the upload itself at a rate that rounds to the model's: from its full size, with no encode of its own
        path = resolve(doc["remake"]["path"] if kept else take["path"])
        made = doc.get("remake") and shot.get("source_range")  # a remake's shot: its frames are the source's one to one, at the source's real rate, and so is its sound
        start, end = (list(shot.get("trim") or [0.0, None]) + [None, None])[:2]
        length = take.get("duration_s") or shot.get("duration_s") or 5.0
        start = float(start or 0.0)
        end = float(end) if end is not None else model.out_point(shot, length)
        clips.append({"shot": shot["id"], "take": take["id"], "path": path, "kind": take.get("kind", "video"), "start": start, "end": max(start + 0.04, end), "speed": model.speed_of(shot, take),
                      "transition": shot.get("transition") or {"type": "cut"}, "has_audio": take.get("has_audio"), "sound": shot.get("sound", True), "codec": take.get("codec"),
                      "width": take.get("width"), "height": take.get("height"), "fps": take.get("fps"), "sung": bool(shot.get("audio_slice")), "dialogue": shot.get("dialogue") or [], "line_cut": shot.get("line_cut") if shot.get("line_cut") == shot.get("trim") else None,
                      "line": (shot.get("generation") or {}).get("settings", {}).get("audio_guide"), "length": float(length),
                      "trimmed": start > 0.0 or end < float(length) - 0.01, "kept": kept, "seek": shot["source_range"][0] / doc["remake"]["fps"] if kept else 0.0,
                      **({"exact": max(1, round((end - start) * doc["remake"]["fps"])), "sound_at": shot["source_range"][0] / doc["remake"]["fps"] + start} if made else {})})
    edit = doc["edit"]
    layout_sound_untouched = bool(doc.get("remake")) and edit.get("audio_mode") == "clips" and not edit.get("audio_tracks") and all(clip.get("sound_at") is not None or not clip["has_audio"] for clip in clips)
    if doc["brief"].get("song") and edit.get("audio_mode") in ("song", "mix"):  # the song is the clock: a transition keeps every shot on its part
        for before, clip in zip(clips, clips[1:]):
            if clip["transition"].get("type", "cut") in XFADE:  # the outgoing shot runs into it: spare frames, then its last frame held (tpad, 1 s)
                overlap = min(float(clip["transition"].get("duration_s") or 0.5), 1.0)
                before["end"] += overlap * before["speed"]
                clip["transition"] = {**clip["transition"], "duration_s": overlap}
    copyable = bool(clips) and all(clip["kind"] == "video" and not clip["trimmed"] and not clip["kept"] and clip["speed"] == 1.0 and clip["sound"] and clip["transition"].get("type", "cut") == "cut" for clip in clips) \
        and len({(clip["codec"], clip["width"], clip["height"], clip["fps"]) for clip in clips}) == 1 and edit.get("audio_mode") == "clips" and not edit.get("audio_tracks") and not edit.get("overlays") \
        and not edit.get("subtitles") and not any(clip["dialogue"] for clip in clips)  # subtitles need the encode
    return {"clips": clips, "copy": copyable, "sound_untouched": layout_sound_untouched}


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
        length = sum((clip["end"] - clip["start"]) / clip["speed"] for clip in clips)
        _level_voices(doc, clips)
        measure = _encode_command(doc, clips, resolve, out, video_args or DEFAULT_VIDEO_ARGS, audio_args or DEFAULT_AUDIO_ARGS, measure=True)
        gain = None if layout["sound_untouched"] else _gain(-14.0 if doc["edit"].get("audio_mode") == "song" and not doc["edit"].get("audio_tracks") else -16.0, _loudness(measure, timeout=max(120.0, 5.0 * length))) if measure else 0.0  # a remake's sound alone stays at its own level
        command = _encode_command(doc, clips, resolve, out, video_args or DEFAULT_VIDEO_ARGS, audio_args or DEFAULT_AUDIO_ARGS, gain_db=gain)
        _run_graph(command, timeout=max(300.0, 30.0 * length))  # an encode that stalls fails instead of blocking the director
        method = "single encode"
    if not layout["sound_untouched"]:
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
        args = [*command[:graph], "-/filter_complex", str(script), *command[graph + 2:]]
        try:
            return media._run(args, **kwargs)
        except RuntimeError as error:
            # Before FFmpeg 7, only the dedicated graph-file option was available.
            if "Unrecognized option '/filter_complex'" not in str(error):
                raise
        args[graph] = "-filter_complex_script"
        return media._run(args, **kwargs)


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
            "true_peak_dbtp": float(peak[-1]) if peak else None, "silent_from_s": _silent_from(path) if "audio" in kinds else None}


def _silent_from(path: Path) -> float | None:
    """Where the silence that ends the file starts, when its last second or more is silent."""
    report = media._run([media.ffmpeg(), "-hide_banner", "-nostats", "-v", "info", "-i", str(path), "-vn", "-sn", "-af", "silencedetect=n=-50dB:d=1", "-f", "null", "-"]).stderr.decode("utf-8", "replace")
    starts, ends = re.findall(r"silence_start: (-?[\d.]+)", report), re.findall(r"silence_end: (-?[\d.]+)", report)
    duration = re.search(r"time=(\d+):(\d+):([\d.]+)", report)
    if not starts:
        return None
    if len(ends) < len(starts):  # still silent when the file ended
        return float(starts[-1])
    total = int(duration[1]) * 3600 + int(duration[2]) * 60 + float(duration[3]) if duration else None
    return float(starts[-1]) if total is not None and float(ends[-1]) >= total - 0.2 else None


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


def _level_voices(doc: dict[str, Any], clips: list[dict[str, Any]]) -> None:
    """The edit's level_voices: a video model says each shot's lines at a level of its own, and the mix's one gain would leave the jumps. Each shot with lines moves to their median,
    measured on what the cut plays of it."""
    if not doc["edit"].get("level_voices") or not doc["brief"].get("voices_from_model"):
        return
    voiced = [clip for clip in clips if clip["kind"] == "video" and clip["has_audio"] and clip["sound"] and clip["dialogue"] and not clip["sung"]]
    measured = [(clip, _loudness([media.ffmpeg(), "-ss", f"{clip['start']:.3f}", "-to", f"{clip['end']:.3f}", "-i", str(clip["path"]), "-vn", "-af", MEASURE, "-f", "null", "-"])) for clip in voiced]
    measured = [(clip, loudness) for clip, loudness in measured if loudness is not None]
    if len(measured) > 1:
        middle = statistics.median(loudness for _, loudness in measured)
        for clip, loudness in measured:
            clip["voice_gain_db"] = max(-VOICE_GAIN_DB, min(VOICE_GAIN_DB, middle - loudness))


def _gain(target: float, loudness: float | None) -> float:
    return 0.0 if loudness is None else max(-30.0, min(MAX_GAIN_DB, target - loudness))


def _level(gain: float) -> str:
    """One fixed gain, then a limiter that keeps the peaks under the ceiling without raising anything (level=disabled), run at 4x the rate so
    that peaks between samples (true peaks) are caught too."""
    return f"aresample=192000,volume={gain:.2f}dB,alimiter=limit={10 ** (CEILING_DB / 20):.4f}:attack=2:release=40:level=disabled,aresample=48000"


def _in_frames(clips, fps) -> list[dict[str, Any]]:
    """Every clip lasts whole frames (span: its seconds in the cut), and its picture is cut to exactly that many: the sound, subtitles and text placed
    at the clips' starts then stay on the picture (fractions of a frame left at each cut added up to 0.1 s of lip-sync drift over a 13-shot film)."""
    return [{**clip, "frames": (frames := clip["exact"] if clip.get("exact") and clip["speed"] == 1.0 else max(1, round((clip["end"] - clip["start"]) / clip["speed"] * fps))), "span": frames / fps, "end": clip["start"] + frames / fps * clip["speed"]} for clip in clips]


def _places(clips) -> tuple[list[float], list[float], float]:
    """Where each clip starts in the cut and how much it overlaps the one before (cuts follow each other, transitions overlap the two clips), and the cut's length."""
    offsets, overlaps, total = [0.0], [0.0], clips[0]["span"]
    for clip in clips[1:]:
        overlap = min(float(clip["transition"].get("duration_s") or 0.5), clip["span"] / 2, total / 2) if clip["transition"].get("type", "cut") in XFADE else 0.0
        overlap = overlap if overlap > 0.04 else 0.0
        offsets.append(total - overlap)
        overlaps.append(overlap)
        total += clip["span"] - overlap
    return offsets, overlaps, total


def placed_clip(doc: dict[str, Any], resolve, shot_id: str) -> dict[str, Any] | None:
    """A shot's clip as the export cuts it, with where it starts in the cut (offset); None for a shot outside the edit."""
    clips = plan(doc, resolve)["clips"]
    if not clips:
        return None
    clips = _in_frames(clips, _canvas(doc, clips)[2])
    return next(({**clip, "offset": offset} for clip, offset in zip(clips, _places(clips)[0]) if clip["shot"] == shot_id), None)


def _atempo(speed: float) -> str:
    """A speed as audio filters that keep the pitch: atempo goes no lower than half speed at a time."""
    halves = []
    while speed < 0.5:
        halves.append("atempo=0.5")
        speed /= 0.5
    return ",".join([*halves, f"atempo={speed:g}"])


def _encode_command(doc, clips, resolve, out, video_args, audio_args, gain_db: float = 0.0, measure: bool = False) -> list[str] | None:
    """The single encode of the edit; measure=True: the command that measures its mixed sound alone (None without sound)."""
    width, height, fps = _canvas(doc, clips)
    clips = _in_frames(clips, fps)
    offsets, overlaps, total = _places(clips)
    inputs, filters = [], []
    for index, clip in enumerate(clips):
        duration, frames = clip["span"], clip["frames"]
        exact = clip.get("exact") and clip["speed"] == 1.0  # frame for frame, nothing repeated or dropped: its frames only take the edit's clock
        timing = f"N/({fps:.6f}*TB)" if exact else "PTS-STARTPTS" if clip["speed"] == 1.0 else f"(PTS-STARTPTS)/{clip['speed']:g}"  # slowed down, the fps filter below repeats frames
        if clip["kind"] == "image":  # a still gets a slow push-in (6 % over the shot), cropped to the canvas
            inputs += ["-loop", "1", "-framerate", f"{fps:g}", "-t", f"{duration + 1 / fps:.4f}", "-i", str(clip["path"])]
            source = (f"[{index}:v]scale={2 * width}:{2 * height}:force_original_aspect_ratio=increase,crop={2 * width}:{2 * height},"
                      f"zoompan=z='1+0.06*on/{frames}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d=1:s={width}x{height}:fps={fps:g},")
        else:
            inputs += [*(["-ss", f"{max(0.0, clip['seek'] - 0.0001):.6f}"] if clip.get("seek") else []), "-i", str(clip["path"])]  # just before the frame: at it, rounding can pass it by
            source = f"[{index}:v]trim=start={clip['start']:.4f},"
        fit = f"scale={width}:{height}" if clip.get("kept") else f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black"  # a kept part stretches as a render does: no bars
        filters.append(f"{source}setpts={timing},{fit},setsar=1,{'' if exact else f'fps={fps:g},'}tpad=stop_mode=clone:stop_duration=1,"
                       f"trim=end_frame={frames},setpts=PTS-STARTPTS,settb=AVTB,format=yuv420p[v{index}]")
    # Video chain: cuts concatenate, transitions overlap the two clips by their duration.
    label = "v0"
    for index in range(1, len(clips)):
        target = f"x{index}"
        if overlaps[index]:
            filters.append(f"[{label}][v{index}]xfade=transition={XFADE[clips[index]['transition']['type']]}:duration={overlaps[index]:.3f}:offset={offsets[index]:.3f}[{target}]")
        else:
            filters.append(f"[{label}][v{index}]concat=n=2:v=1:a=0[{target}]")
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
    if not (doc["brief"].get("song") and doc["edit"].get("audio_mode") != "clips"):  # on a song, subtitles follow the song, which does not move
        places = model.cut_places(doc)
        for index, item in enumerate(subtitles):
            if item.get("cut") and item["cut"] != places:  # timed on an earlier cut: a copy on this one, the file itself stays as it is
                moved = out.with_name(f"{out.stem}-{index}.srt")
                moved.write_text(_srt(recut_cues(_srt_cues(resolve(item["path"]).read_text(encoding="utf-8")), item["cut"], places)), encoding="utf-8")
                subtitles[index] = {**item, "path": str(moved)}
    if not subtitles and any(clip["dialogue"] for clip in clips):  # the shots' lines, timed on this cut (an SRT of the edit's own replaces them)
        srt = out.with_suffix(".srt")
        sung = model.checked_lyrics(doc) if model.on_the_song(doc) else None
        srt.write_text(dialogue_srt(clips, offsets, resolve) if sung is None else sung_srt(sung, float(doc["edit"].get("song_start_s") or 0.0), total), encoding="utf-8")
        subtitles = [{"path": str(srt), "language": LANGUAGES.get(str(doc["brief"].get("language") or "en")[:2].lower(), "und")}]
    first_subtitle = inputs.count("-i")
    command[command.index("-filter_complex"):command.index("-filter_complex")] = [arg for item in subtitles for arg in ("-i", str(resolve(item["path"])))]
    for offset, item in enumerate(subtitles):
        command += ["-map", f"{first_subtitle + offset}:s", f"-metadata:s:s:{offset}", f"language={item.get('language', 'eng')}", f"-disposition:s:{offset}", "0"]  # the viewer turns them on
    if subtitles:
        command += ["-c:s", "mov_text" if out.suffix.lower() in (".mp4", ".mov", ".m4v") else "srt"]
    return [*command, *video_args, "-r", f"{fps:.6f}".rstrip("0").rstrip("."), "-t", f"{total:.3f}", "-movflags", "+faststart", str(out)]


LANGUAGES = {"en": "eng", "fr": "fra", "de": "deu", "es": "spa", "it": "ita", "pt": "por", "nl": "nld", "ja": "jpn", "zh": "zho", "ko": "kor", "ru": "rus", "ar": "ara", "hi": "hin"}


def _srt_cues(text: str) -> list[tuple[float, float, str]]:
    seconds = lambda stamp: sum(float(part) * 60 ** index for index, part in enumerate(reversed(stamp.strip().replace(",", ".").split(":"))))
    cues = []
    for block in text.replace("\r", "").strip().split("\n\n"):
        lines = block.strip().split("\n")
        timing = next((index for index, line in enumerate(lines) if " --> " in line), None)
        if timing is not None:
            start, end = lines[timing].split(" --> ")
            cues.append((seconds(start), seconds(end.split()[0]), "\n".join(lines[timing + 1:]).strip()))
    return cues


def recut_cues(cues, before, now) -> list[tuple[float, float, str]]:
    """Subtitles timed on an earlier cut, placed on this one: each line stays where it was in the picture of the shot it started on. A line goes with its shot when the shot is gone or no
    longer plays that part of its take."""
    shots = {item["shot"]: item for item in now}
    moved = []
    for start, end, text in cues:
        old = next((item for item in before if item["start_s"] <= start < item["start_s"] + item["duration_s"]), None)
        new = shots.get(old["shot"]) if old else None
        if new is None:
            continue
        place = new["start_s"] + start - old["start_s"] + old.get("in_s", 0.0) - new.get("in_s", 0.0)
        if place + end - start > new["start_s"] and place < new["start_s"] + new["duration_s"]:
            moved.append((max(place, new["start_s"]), place + end - start, text))
    return moved


def _srt(cues) -> str:
    def stamp(seconds):
        milliseconds = int(round(seconds * 1000))
        return f"{milliseconds // 3600000:02d}:{milliseconds // 60000 % 60:02d}:{milliseconds // 1000 % 60:02d},{milliseconds % 1000:03d}"
    return "".join(f"{number}\n{stamp(start)} --> {stamp(end)}\n{text}\n\n" for number, (start, end, text) in enumerate(cues, 1))


def sung_srt(lines, song_start: float, total: float) -> str:
    """Subtitles of a cut that plays the song, from when its lines are really sung (check_song): a line stays until the next one when the pause between them is short; a line the song
    does not sing gets no subtitle (one it sings unclearly keeps its place)."""
    sung = [(line["start_s"] - song_start, line["end_s"] - song_start, line["line"]) for line in lines if line["start_s"] is not None and 0.0 <= line["start_s"] - song_start < total]
    cues = []
    for (start, end, text), following in zip(sung, sung[1:] + [None]):
        cues.append((start, following[0] - 0.05 if following and following[0] - end < 1.0 else min(end + 0.4, total), text))
    return _srt(cues)


def dialogue_srt(clips, offsets, resolve) -> str:
    """Subtitles of the shots' dialogue at their place in this cut: a line starts with its shot and lasts its recorded length, and several recorded lines
    follow each other as they do in the sound that drives the shot; lines not recorded yet share their shot by their number of characters."""
    cues = []
    for clip, offset in zip(clips, offsets):
        lines = [item for item in clip["dialogue"] if item.get("line", "").strip()]
        if not lines:
            continue
        shown = clip["span"]
        if all(item.get("duration_s") for item in lines):
            start = offset
            for item in lines:
                if start >= offset + shown:
                    break
                end = start + item["duration_s"] / clip["speed"]
                cues.append((start, min(end + model.BETWEEN_LINES, offset + shown) - 0.05, item["line"].strip()))
                start = end + model.BETWEEN_LINES / clip["speed"]
            continue
        recorded = lines[0].get("duration_s")  # the line itself: the clip that drives the shot may be padded with silence
        if not recorded and clip.get("line") and resolve(clip["line"]).is_file():
            recorded = media.probe(resolve(clip["line"]))["duration_s"]
        span = min(shown, recorded / clip["speed"] + 0.3) if recorded and len(lines) == 1 else shown
        total = sum(len(item["line"]) for item in lines)
        start = offset
        for item in lines:
            end = start + span * len(item["line"]) / total
            cues.append((start, end - 0.05, item["line"].strip()))
            start = end
    return _srt(cues)


END_CARD_GAP = 0.25  # seconds before the end of the cut within which an end card closes the film


def _overlays(doc, resolve, inputs, filters, label, width, height, fps, total) -> str:
    """Text and pictures laid over the cut, each over its own time range with a fade in and out (after the audio inputs, whose indexes are fixed)."""
    for number, overlay in enumerate(doc["edit"].get("overlays", [])):
        start, end = float(overlay["start_s"]), min(float(overlay["end_s"]), total)
        closes = overlay.get("style") == "end_card" and total - end < END_CARD_GAP  # the last thing seen: it stays to the last frame, and the shot behind it does not come back
        end = total if closes else end
        if end - start < 0.05 or not overlay.get("path") or not resolve(overlay["path"]).is_file():
            continue
        duration = end - start
        fade = min(float(overlay.get("fade_s") or 0.0), duration if closes else duration / 3)
        index = inputs.count("-i")
        inputs += ["-loop", "1", "-framerate", f"{fps:g}", "-t", f"{duration + 1 if closes else duration:.3f}", "-i", str(resolve(overlay["path"]))]  # longer than the cut, which cuts it: its last frame is covered too
        fades = f",fade=t=in:st=0:d={fade:.3f}:alpha=1" + ("" if closes else f",fade=t=out:st={duration - fade:.3f}:d={fade:.3f}:alpha=1") if fade > 0.01 else ""
        filters.append(f"[{index}:v]format=rgba,scale={width}:{height}{fades},setpts=PTS-STARTPTS+{start:.3f}/TB[o{number}]")
        filters.append(f"[{label}][o{number}]overlay=eof_action=pass,format=yuv420p[vo{number}]")
        label = f"vo{number}"
    return label


def _track_cut(track) -> str:
    """The part of its file a sound track plays (in_s to out_s, seconds of the file)."""
    start, stop = float(track.get("in_s") or 0.0), track.get("out_s")
    return f"atrim=start={start:.3f}" + (f":end={float(stop):.3f}" if stop is not None else "") + "," if start or stop is not None else ""


def _track_fades(track, path) -> str:
    fade_in, fade_out = float(track.get("fade_in_s") or 0.0), float(track.get("fade_out_s") or 0.0)
    fades = f",afade=t=in:st=0:d={fade_in:.3f}" if fade_in else ""
    if fade_out:
        length = media.probe(path)["duration_s"] or 0.0
        length = (min(float(track["out_s"]), length) if track.get("out_s") is not None else length) - float(track.get("in_s") or 0.0)
        fades += f",afade=t=out:st={max(0.0, length - fade_out):.3f}:d={fade_out:.3f}"
    return fades


def _source_stretches(clips, offsets) -> list[dict[str, float]]:
    """The parts of a remake's source its sound plays: shots that follow each other in the source and in the edit make one stretch, with nothing at their cut."""
    stretches = []
    for clip, offset in zip(clips, offsets):
        if clip.get("sound_at") is None or not clip["has_audio"] or not clip["sound"] or clip["speed"] != 1.0:
            continue
        last = stretches[-1] if stretches else None
        if last and abs(last["end"] - clip["sound_at"]) < 0.001 and abs(last["offset"] + last["end"] - last["start"] - offset) < 0.001:
            last["end"] += clip["span"]
        else:
            stretches.append({"start": clip["sound_at"], "end": clip["sound_at"] + clip["span"], "offset": offset})
    return stretches


def _audio_chain(doc, clips, resolve, inputs, filters, offsets, total, gain_db) -> str | None:
    edit, parts = doc["edit"], []
    mode = edit.get("audio_mode", "clips")
    index = len(clips)
    if mode in ("clips", "mix") and doc.get("remake"):  # the source's own sound, taken once from the upload: the takes hold copies of it
        for number, stretch in enumerate(_source_stretches(clips, offsets)):
            inputs += ["-ss", f"{stretch['start']:.6f}", "-t", f"{stretch['end'] - stretch['start']:.6f}", "-i", str(resolve(doc["brief"]["source"]))]
            length, delay = stretch["end"] - stretch["start"], f"{round(stretch['offset'] * 48000)}S"  # to the sample
            filters.append(f"[{index}:a]asetpts=PTS-STARTPTS,aresample=48000,afade=t=in:d=0.015,afade=t=out:st={max(0.0, length - 0.03):.3f}:d=0.03,adelay={delay}|{delay}[s{number}]")  # no click where a stretch starts or stops
            parts.append((f"s{number}", 1.0))
            index += 1
    if mode in ("clips", "mix"):
        for position, clip in enumerate(clips):
            delay = int(offsets[position] * 1000)
            if clip.get("sound_at") is not None and clip["speed"] == 1.0:  # in a stretch of the source
                continue
            if clip["kind"] == "video" and clip["has_audio"] and clip["sound"] and not (mode == "mix" and clip["sung"] and doc["brief"].get("song")):  # the song already sings it
                tempo = "" if clip["speed"] == 1.0 else _atempo(clip["speed"]) + ","
                played = (clip["end"] - clip["start"]) / clip["speed"]
                cut = clip.get("line_cut") or [0.0, None]
                fade_in, fade_out = WORD_CUT_FADE_S if cut[0] else 0.015, WORD_CUT_FADE_S if cut[1] is not None else 0.03
                fades = f"afade=t=in:d={fade_in},afade=t=out:st={played - fade_out:.3f}:d={fade_out}," if played > 0.1 else ""  # no click where the sound of one clip stops and the next starts
                voice = f"volume={clip['voice_gain_db']:.2f}dB," if clip.get("voice_gain_db") else ""
                filters.append(f"[{position}:a]atrim=start={clip['start']:.3f}:end={clip['end']:.3f},asetpts=PTS-STARTPTS,{tempo}aresample=48000,{voice}{fades}adelay={delay}|{delay}[a{position}]")
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
        filters.append(f"[{index}:a]{_track_cut(track)}asetpts=PTS-STARTPTS,aresample=48000{_track_fades(track, resolve(track['path']))},adelay={delay}|{delay}[t{number}]")
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
