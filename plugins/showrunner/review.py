# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Deep Review: a finished version measured against its script, after the fact and on request: what is really said and sung, when subtitles show, whose voice it is, whether
lips follow, how the sound and the picture hold. Findings say where and what, a fault when it is measured, a check when it is a suspicion for the user's eyes."""
from __future__ import annotations

from collections import Counter
import difflib
import math
import re
import subprocess
from typing import Any

import numpy as np

from . import media, production as model

RATE = 16000
SUBTITLE_OFF = 0.7  # seconds between a subtitle and its words that a viewer notices
SAID_ELSEWHERE = 6.0  # seconds beyond which words heard are the same words at another moment (a refrain), not these
LEVEL_JUMP = 12.0  # dB between two neighbouring shots
SPEECH_FLOOR = -45.0  # dB: below it, what a transcriber hears is what it invents on silence ("Thank you.")
AROUND_LINE = 0.5  # seconds around a line whose words belong to it: a transcript misses or splits words at its ends


def finding(kind: str, level: str, at_s: float | None, shots: list[str], text: str) -> dict[str, Any]:
    return {"kind": kind, "level": level, "at_s": None if at_s is None else round(float(at_s), 2), "shots": shots, "text": text}


def shots_between(places: list[dict[str, Any]], start: float, end: float) -> list[str]:
    return [place["shot"] for place in places if place["start_s"] < end and place["start_s"] + place["duration_s"] > start]


def shot_start(places: list[dict[str, Any]], shot: str | None) -> float | None:
    return next((place["start_s"] for place in places if place["shot"] == shot), None)


def script_lines(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """What the version should say, in order: the lyrics of a cut that plays the song, the shots' lines otherwise, with who says them and the shot they belong to."""
    if model.on_the_song(doc) and model.lyric_lines(doc):
        return [{"line": line, "speaker": None, "shot": None} for line in model.lyric_lines(doc)]
    return [{"line": line["line"], "speaker": line.get("speaker"), "shot": shot["id"]} for shot in doc["shots"] if shot.get("included", True) for line in shot.get("dialogue") or [] if line.get("line")]


def speaking_shots(doc: dict[str, Any], moving: set[str]) -> set[str]:
    """Generated shots where someone on screen speaks or sings."""
    return {shot["id"] for shot in doc["shots"] if shot["id"] in moving and (shot.get("audio_slice") or any(line.get("speaker") in (shot.get("cast") or []) for line in shot.get("dialogue") or []))}


def decode(path) -> np.ndarray:
    raw = subprocess.run([media.ffmpeg(), "-v", "error", "-i", str(path), "-vn", "-ac", "1", "-ar", str(RATE), "-f", "f32le", "-"], capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def level_db(audio: np.ndarray, start: float, end: float) -> float:
    part = audio[max(0, int(start * RATE)):max(0, int(end * RATE))]
    return float(20 * np.log10(np.sqrt(np.mean(part ** 2)) + 1e-6)) if len(part) else -120.0


def word_findings(lines: list[dict[str, Any]], measured: list[dict[str, Any]], words: list[tuple[str, float, float]], places: list[dict[str, Any]], audio: np.ndarray) -> list[dict[str, Any]]:
    """Lines not said as written, with what is heard instead where they should be (neighbouring lines of a song together), and on a film what is said outside the script."""
    used = [False] * len(words)

    def spend(start: float, end: float) -> list[str]:
        heard = [index for index, (_, at, _) in enumerate(words) if start <= at < end]
        for index in heard:
            used[index] = True
        return [words[index][0].strip() for index in heard]

    for item in measured:
        if item["sung"]:
            spend(item["start_s"] - AROUND_LINE, item["end_s"] + AROUND_LINE)
    missing: dict[tuple, list[str]] = {}
    for index, (line, item) in enumerate(zip(lines, measured)):
        if not item["sung"]:
            before = next((other["end_s"] for other in reversed(measured[:index]) if other["sung"]), 0.0)
            after = next((other["start_s"] for other in measured[index + 1:] if other["sung"]), math.inf)
            missing.setdefault((before, after, line["shot"]), []).append(line["line"])
    found = []
    for (before, after, shot), written in missing.items():
        instead, at_s = spend(before, after), shot_start(places, shot)
        text = "Not said as written: " + " / ".join(f'"{line}"' for line in written) + (f'. Heard here: "{" ".join(instead[:24])}"' if instead else ". Nothing is heard here.")
        found.append(finding("words", "fault", before if at_s is None else at_s, [shot] if shot else shots_between(places, before, after), text))
    if lines and lines[0]["shot"]:  # a song has its ad-libs
        run = []
        for index, word in enumerate(words + [("", math.inf, math.inf)]):
            if index < len(words) and not used[index]:
                run.append(word)
                continue
            if run and level_db(audio, run[0][1], run[-1][2]) > SPEECH_FLOOR:
                found.append(finding("words", "check", run[0][1], shots_between(places, run[0][1], run[-1][2]), 'Words that are not in the script: "' + " ".join(item[0] for item in run)[:90] + '"'))
            run = []
    return found


def subtitle_findings(cues: list[tuple[float, float, str]], words: list[tuple[str, float, float]], places: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Subtitles against the words heard: away from when they are said, or showing words nobody says near them (neighbouring ones as one finding). A cue in brackets describes a sound."""
    letters = lambda text: re.sub(r"[^a-z0-9]", "", text.lower())
    heard = [letters(word) for word, _, _ in words]
    found, unsaid = [], []

    def close_unsaid() -> None:
        if unsaid:
            found.append(finding("subtitles", "check", unsaid[0][0], list(dict.fromkeys(shot for start, end, _ in unsaid for shot in shots_between(places, start, end))),
                                 ("A subtitle shows words that are not heard: " if len(unsaid) == 1 else f"{len(unsaid)} subtitles show words that are not heard: ") + " / ".join(f'"{text}"' for _, _, text in unsaid)))
            unsaid.clear()

    for start, end, text in cues:
        wanted = letters(text)
        if not wanted or text.lstrip().startswith(("[", "(")):
            continue
        said, last = [], -2
        for first in range(len(heard)):
            blocks = difflib.SequenceMatcher(None, wanted, "".join(heard[first:first + len(text.split()) + 2]), autojunk=False).get_matching_blocks()
            if blocks[0].b == 0 and sum(block.size for block in blocks) >= 0.7 * len(wanted):  # words starting here that read as the subtitle's, a first one missed or not
                if first != last + 1:  # a run of matches is one saying
                    said.append(words[first][1] - blocks[0].a / len(wanted) * (end - start))
                last = first
        off = start - min(said, key=lambda at: abs(at - start)) if said else None
        if off is None or abs(off) > SAID_ELSEWHERE:
            if unsaid and start - unsaid[-1][1] > 3.0:
                close_unsaid()
            unsaid.append((start, end, text))
            continue
        close_unsaid()
        if abs(off) > SUBTITLE_OFF:
            found.append(finding("subtitles", "fault", start, shots_between(places, start, end), f'A subtitle comes {abs(off):.1f} s {"late" if off > 0 else "early"}: "{text}"'))
    close_unsaid()
    return found


def voice_findings(doc: dict[str, Any], places: list[dict[str, Any]], pitch, resolve) -> list[dict[str, Any]]:
    """Voices against their cards, and lines against their speaker's voice."""
    found = []
    for member in doc["cast"]:
        if not member.get("voice"):
            continue
        sample, kind = pitch(resolve(member["voice"])), model.voice_kind(member.get("voice_description") or member.get("description"))
        if model.voice_off(kind, sample):
            found.append(finding("voice", "fault", None, [], f"{member['name']}'s voice measures {sample:.0f} Hz, which reads as a {'man' if kind == 'woman' else 'woman'}, and their card says a {kind}."))
        for shot in doc["shots"]:
            for line in (shot.get("dialogue") or []) if shot.get("included", True) else []:
                if line.get("speaker") == member["id"] and line.get("audio") and model.line_off_voice(sample, heard := pitch(resolve(line["audio"]))):
                    found.append(finding("voice", "fault", shot_start(places, shot["id"]), [shot["id"]], f'{member["name"]} does not sound like themselves here: this line measures {heard:.0f} Hz, their voice {sample:.0f} Hz: "{line["line"][:60]}"'))
    heard = (doc["audio"].get("lyrics") or {}).get("voice_hz") if model.on_the_song(doc) else None
    kinds = model.singer_kinds(doc)
    if heard and len(kinds) == 1 and model.voice_off(next(iter(kinds)), heard):
        found.append(finding("voice", "fault", None, [], f"The song's lead voice measures {heard:.0f} Hz, and the singer on screen is a {next(iter(kinds))}."))
    return found


def speaker_findings(doc: dict[str, Any], lines: list[dict[str, Any]], turns: list[tuple[str, str]], places: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Who the transcriber hears against who the script says, from (speaker number, words) turns: two characters on one voice, a line on another voice than its speaker's."""
    names = {member["id"]: member["name"] for member in doc["cast"]}
    normal = lambda text: re.findall(r"[a-z0-9']+", text.lower())
    heard: dict[str, Counter] = {}
    where, position = [], 0
    for line in lines:
        if not line["speaker"]:
            continue
        expected, best = normal(line["line"]), None
        for index in range(position, len(turns)):
            size = sum(block.size for block in difflib.SequenceMatcher(None, expected, normal(turns[index][1]), autojunk=False).get_matching_blocks())
            if size * 2 >= len(expected) and (best is None or size > best[1]):
                best = (index, size)
                if size == len(expected):
                    break
        if best:
            position = best[0]
            heard.setdefault(line["speaker"], Counter())[turns[best[0]][0]] += 1
            where.append((line, turns[best[0]][0]))
    usual = {speaker: count.most_common(1)[0][0] for speaker, count in heard.items()}
    found = []
    for one, voice in usual.items():
        for other, voice_other in usual.items():
            if one < other and voice == voice_other:
                found.append(finding("speakers", "check", None, [], f"{names.get(one, one)} and {names.get(other, other)} are heard as one voice: they may sound alike."))
    for line, voice in where:
        if voice != usual[line["speaker"]]:
            found.append(finding("speakers", "check", shot_start(places, line["shot"]), [line["shot"]], f'This line of {names.get(line["speaker"], line["speaker"])} is heard in another voice: "{line["line"][:60]}"'))
    return found


def sound_findings(audio: np.ndarray, places: list[dict[str, Any]], talking: set[str], silent_from: float | None, total: float) -> list[dict[str, Any]]:
    """The sound stopping before the picture, and a jump of level between two shots that both have lines or both have none: from a line to a quiet shot is how a film sounds."""
    found = []
    levels = [(place, level_db(audio, place["start_s"], place["start_s"] + place["duration_s"])) for place in places]
    for (before, loud_before), (after, loud_after) in zip(levels, levels[1:]):
        if loud_before > -60 and loud_after > -60 and abs(loud_after - loud_before) > LEVEL_JUMP and (before["shot"] in talking) == (after["shot"] in talking):
            found.append(finding("sound", "check", after["start_s"], [before["shot"], after["shot"]], f"The sound jumps by {abs(loud_after - loud_before):.0f} dB at this cut."))
    if silent_from is not None and total - silent_from > 1.0:
        found.append(finding("sound", "fault", silent_from, shots_between(places, silent_from, total), f"The sound stops at {silent_from:.1f} s and the video runs to {total:.1f} s."))
    return found


def picture_findings(path, places: list[dict[str, Any]], moving: set[str]) -> list[dict[str, Any]]:
    """Black or frozen picture in shots that should move (a card or a still is meant to hold)."""
    report = subprocess.run([media.ffmpeg(), "-hide_banner", "-nostats", "-v", "info", "-i", str(path), "-an", "-sn", "-vf", "fps=8,scale=320:-2,blackdetect=d=0.3:pic_th=0.97,freezedetect=n=0.002:d=1.2", "-f", "null", "-"], capture_output=True).stderr.decode("utf-8", "replace")
    found = []
    for start, end in re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", report):
        shots = [shot for shot in shots_between(places, float(start), float(end)) if shot in moving]
        if shots:
            found.append(finding("picture", "fault", float(start), shots, f"Black picture for {float(end) - float(start):.1f} s."))
    for start, end in re.findall(r"freeze_start: ([\d.]+).*?freeze_end: ([\d.]+)", report, flags=re.S):
        shots = [shot for shot in shots_between(places, float(start), float(end)) if shot in moving]
        if shots:
            found.append(finding("picture", "check", float(start), shots, f"The picture is frozen for {float(end) - float(start):.1f} s."))
    return found


def lip_finding(shot: str, at_s: float, measured: dict[str, Any] | None) -> dict[str, Any] | None:
    """A check, never a fault: the measure is noisy on small faces."""
    from shared.deepy.lip_sync import verdict

    return finding("lips", "check", at_s, [shot], "The lips may not follow the voice here. Worth a look.") if verdict(measured) == "may_not_follow" else None
