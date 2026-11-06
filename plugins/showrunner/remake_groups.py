# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""A remake's short shots render together. Viggle Remake follows speech and singing much better over a long window than over a single short shot,
so consecutive shots join one request: one view over their part of the source, each shot's painted frames pinned at their own frames, their
prompts joined as [Shot 1], [Shot 2]... Each shot then keeps its own part of the render as its take."""
from __future__ import annotations

import re
from typing import Any

SECTIONS = ("subject_definitions", "summary", "retention_analysis", "detailed_description", "overall_soundscape", "non_diegetic_music")
MAX_CUTS, MAX_PICTURES = 3, 4  # upstream: the model drops and reorders short shots in a window with many cuts; the model takes four painted frames
_TOKEN = re.compile(r"<(Picture|Subject) (\d+)>|\(S(\d+)\)|\[Shot (\d+)\]")
_CLOCK = re.compile(r"\b(\d{2}):(\d{2}\.\d{3})\b")
_KEY = re.compile(r"<(?:Video|Picture|Subject) \d+>")
_SUBJECT = re.compile(r"^<Subject (\d+)> is (.+?) in ((?:<Picture \d+>(?:,? and |, )?)+):\s*(.*)$")


def kept(shot: dict[str, Any]) -> bool:
    """A shot nothing changes in: its take is its part of the source, never the model's."""
    return not shot["cast"] and all(item["path"] == item["source"] for item in shot.get("painted_frames") or [])


def frames(group: list[dict[str, Any]]) -> int:
    return group[-1]["source_range"][1] - group[0]["source_range"][0]


def _valid(group: list[dict[str, Any]], maximum: int) -> bool:
    """Shots that can render as one: consecutive, of one template, no mask (a shot's mask covers its own clip), within one window of `maximum`
    frames, MAX_CUTS cuts and MAX_PICTURES painted frames. One shot always can."""
    templates = {item["generation"].get("template") for item in group if not kept(item)}
    return len(group) == 1 or (all(one["source_range"][1] == two["source_range"][0] for one, two in zip(group, group[1:])) and len(templates) <= 1
                               and not any((item["generation"].get("settings") or {}).get("video_mask") for item in group) and frames(group) <= maximum
                               and len(group) <= MAX_CUTS + 1 and sum(len(item.get("painted_frames") or []) for item in group) <= MAX_PICTURES)


def plan(doc: dict[str, Any], minimum: int, maximum: int, wanted: list[str] | tuple[str, ...] | None = None) -> list[list[dict[str, Any]]]:
    """The renders of a remake's shots asked for (`wanted`, all of them by default), in source order: consecutive shots asked for render together while
    they fit (_valid); a render still shorter than `minimum` frames takes the shots around it that no other render holds, the next one first, then
    shots from the end of the render before it while that one keeps its minimum."""
    shots = sorted((item for item in doc["shots"] if item.get("source_range")), key=lambda item: item["source_range"][0])
    asked = {shot["id"] for shot in shots} if wanted is None else set(wanted)
    if minimum <= 0:  # no minimum (Group rendering off): every shot alone
        return [[shot] for shot in shots if shot["id"] in asked]
    groups: list[list[dict[str, Any]]] = []
    for shot in shots:
        if shot["id"] not in asked:
            continue
        if groups and _valid([*groups[-1], shot], maximum):
            groups[-1].append(shot)
        else:
            groups.append([shot])
    taken, order = {item["id"] for group in groups for item in group}, [shot["id"] for shot in shots]
    for group in groups:
        while frames(group) < minimum:
            after, before = order.index(group[-1]["id"]) + 1, order.index(group[0]["id"]) - 1
            candidates = [[*group, shots[after]] if after < len(shots) and shots[after]["id"] not in taken else None, [shots[before], *group] if before >= 0 and shots[before]["id"] not in taken else None]
            grown = next((candidate for candidate in candidates if candidate and _valid(candidate, maximum)), None)
            if grown is None:
                break
            taken.update(item["id"] for item in grown)
            group[:] = grown
    for previous, group in zip(groups, groups[1:]):
        while frames(group) < minimum and len(previous) > 1 and frames(previous[:-1]) >= minimum and _valid([previous[-1], *group], maximum):
            group.insert(0, previous.pop())
    return groups


def group_of(groups: list[list[dict[str, Any]]], shot_id: str) -> list[dict[str, Any]]:
    return next(group for group in groups if any(item["id"] == shot_id for item in group))


def _clock(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{seconds % 60:06.3f}"


def _sections(prompt: str, shot_id: str) -> dict[str, str]:
    parts = re.split(rf"^({'|'.join(SECTIONS)}):[ \t]*$", str(prompt or "").strip(), flags=re.M)
    found = {name: body.strip() for name, body in zip(parts[1::2], parts[2::2])}
    if tuple(found) != SECTIONS or parts[0].strip():
        raise ValueError(f"{shot_id}'s prompt is not in the template's six sections ({', '.join(SECTIONS)}): write it as the template's guidance says, then render it.")
    return found


def joined_prompt(group: list[dict[str, Any]], fps: float) -> str:
    """The prompts of a group's shots as one, in the window's time. Each shot's pictures are numbered after the shots before it; a cast member
    seen in several shots stays one subject, with all their pictures; each shot's [Shot N] and times move to where the shot starts in the window.
    A shot left as it is says so, from what was seen in it."""
    start = group[0]["source_range"][0]
    videos, pictures, summary, details = [], [], [], []
    subjects: dict[int, dict[str, Any]] = {}  # the window's subjects: name, pictures, the line that defines them
    kept_lines: dict[str, str] = {}  # retention lines by what they keep: <Picture N>, <Subject N>
    head = {"tag": "", "video": "", "style": "", "soundscape": "", "music": ""}
    picture_count = shot_count = 0
    for shot in group:
        offset = (shot["source_range"][0] - start) / fps
        own = len(shot.get("painted_frames") or [])
        if kept(shot):
            seen = re.split(r"(?<=\.)\s", str(shot.get("seen") or "").strip(), maxsplit=1)[0] or "The shot as it was."
            shot_count += 1
            videos.append(seen.rstrip("."))
            pictures += [f"<Picture {picture_count + number}> is a frame of the target video, from [Shot {shot_count}], exactly as <Video 1> shows it." for number in range(1, own + 1)]
            details.append(f"[Shot {shot_count}] At {_clock(offset)} the shot of <Video 1> plays as it is: {seen}")
            picture_count += own
            continue
        found = _sections(shot["generation"]["settings"].get("prompt"), shot["id"])
        local: dict[int, int] = {}  # this shot's subject numbers in the window
        for line in found["subject_definitions"].splitlines():
            line = line.strip()
            if not line.startswith("<Subject "):
                continue
            match = _SUBJECT.match(line)
            again = next((number for number, item in subjects.items() if match and item["match"] and item["name"].casefold() == match[2].casefold()), None)
            local[int(re.match(r"<Subject (\d+)>", line)[1])] = again or len(subjects) + 1
            if again is None:
                subjects[len(subjects) + 1] = {"name": match[2] if match else "", "pictures": [], "line": line, "match": match}

        def renumber(text: str, shift: bool = False) -> str:
            def token(match):
                kind, number, spoken, shot_no = match.groups()
                return (f"<Picture {picture_count + int(number)}>" if kind == "Picture" else f"<Subject {local[int(number)]}>" if kind == "Subject"
                        else f"(S{local[int(spoken)]})" if spoken else f"[Shot {shot_count + int(shot_no)}]")
            text = _TOKEN.sub(token, text)
            return _CLOCK.sub(lambda match: _clock(int(match[1]) * 60 + float(match[2]) + offset), text) if shift else text

        for line in found["subject_definitions"].splitlines():
            line = line.strip()
            if line.startswith("<Video 1>"):
                videos.append(line.split(":", 1)[-1].strip().rstrip("."))
            elif line.startswith("<Picture "):
                pictures.append(renumber(line).rstrip(".") + f", from [Shot {shot_count + 1}].")
            elif match := _SUBJECT.match(line):
                subjects[local[int(match[1])]]["pictures"] += [renumber(picture) for picture in re.findall(r"<Picture \d+>", match[3])]
        tag = re.match(r"^\[[^\]]+\]\s*", found["summary"])
        head["tag"] = head["tag"] or (tag[0].strip() if tag else "")
        summary.append(renumber(found["summary"][tag.end() if tag else 0:]))
        for line in found["retention_analysis"].splitlines():
            line = renumber(line.strip())
            if not (key := _KEY.match(line)):
                continue
            if key[0] == "<Video 1>":
                head["video"] = head["video"] or line
            elif key[0] not in kept_lines:
                kept_lines[key[0]] = line
            elif f"[Shot {shot_count + 1}]" not in kept_lines[key[0]]:  # a cast member kept by an earlier shot: this shot joins where they appear
                kept_lines[key[0]] = re.sub(r"\(appears in (.+?)\)", lambda match: f"(appears in {match[1]} and [Shot {shot_count + 1}])", kept_lines[key[0]], count=1)
        lines = [line.strip() for line in found["detailed_description"].splitlines() if line.strip()]
        head["style"] = head["style"] or lines[0]
        details += [renumber(line, shift=True) for line in lines[1:]]
        head["soundscape"], head["music"] = head["soundscape"] or found["overall_soundscape"], head["music"] or found["non_diegetic_music"]
        picture_count += own
        shot_count += max([int(number) for number in re.findall(r"\[Shot (\d+)\]", found["detailed_description"])] or [1])
    definitions = [f"<Video 1> is the source video, {shot_count} shots: " + "; then ".join(videos) + ".", *pictures]
    for number, item in subjects.items():
        definitions.append(f"<Subject {number}> is {item['name']} in {' and '.join(dict.fromkeys(item['pictures']))}: {item['match'][4]}" if item["match"]
                           else re.sub(r"<Subject \d+>", f"<Subject {number}>", item["line"], count=1))
    retention = [head["video"] or "<Video 1> (source video editing): fully_preserved - the camera, the motion and the timing of every shot."]
    retention += [line for key, line in kept_lines.items() if key.startswith("<Picture")] + [line for key, line in kept_lines.items() if key.startswith("<Subject")]
    sentences = dict.fromkeys(sentence for text in summary for sentence in re.split(r"(?<=\.)\s+", text.strip()) if sentence)  # what every shot keeps is said once
    blocks = {"subject_definitions": definitions, "summary": [" ".join(filter(None, [head["tag"], *sentences]))], "retention_analysis": retention,
              "detailed_description": [head["style"] or "Realistic footage.", *details], "overall_soundscape": [head["soundscape"] or "The soundtrack of <Video 1>, unchanged."],
              "non_diegetic_music": [head["music"] or "N/A"]}
    return "\n\n".join(f"{name}:\n" + "\n".join(blocks[name]) for name in SECTIONS)
