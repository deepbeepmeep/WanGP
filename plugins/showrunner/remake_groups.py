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


def prompt_warnings(shot: dict[str, Any]) -> list[str]:
    """What the Viggle Remake MCP server's prompt checks say of a shot's prompt (set_prompt and its lint), for the showrunner when it writes one:
    render refuses a prompt out of the six sections, and the rest goes wrong in the take."""
    prompt, sid = str((shot["generation"].get("settings") or {}).get("prompt") or ""), shot["id"]
    if not prompt.strip() or kept(shot):
        return []
    try:
        _sections(prompt, sid)
    except ValueError as error:
        return [str(error)]
    warnings, count = [], len(shot.get("painted_frames") or [])
    if missing := [number for number in range(1, count + 1) if f"<Picture {number}>" not in prompt]:
        warnings.append(f"{sid}'s prompt never names {_listed([f'<Picture {number}>' for number in missing])}: name each painted frame, in frame order.")
    if extra := sorted({int(number) for number in re.findall(r"<Picture (\d+)>", prompt) if int(number) > count}):
        warnings.append(f"{sid}'s prompt names {_listed([f'<Picture {number}>' for number in extra])}, but the shot has {count} painted frame{'s' if count != 1 else ''}.")
    if re.search(r"\bdim(?:ly)?\b", prompt, re.IGNORECASE):
        warnings.append(f"{sid}'s prompt says 'dim': it darkens the whole render.")
    if re.search(r"[^\x00-\x7f]", re.sub(r"<d>.*?</d>", "", prompt, flags=re.S)):
        warnings.append(f"{sid}'s prompt has non-ASCII characters: plain ASCII punctuation.")
    if any(words.strip() != "[unclear]" for words in re.findall(r"<d>(.*?)</d>", prompt, flags=re.S)):
        warnings.append(f"{sid}'s prompt writes words in <d>: write <d>[unclear]</d>, never the words; the sound comes from the source.")
    if re.search(r"\bonly (?:the|her|his|their) face\b|\bkeeps? (?:the|her|his|their) original (?:clothes|outfit|costume)", prompt, re.IGNORECASE):
        warnings.append(f"{sid}'s prompt keeps part of the original person ('only the face', 'the original clothes'): describe the whole new character, the outfit as theirs.")
    return warnings


def _listed(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _opening(text: str) -> str:
    """A shot's first sentence after "the shot cuts": its framing ("A close-up of...") joins it, anything else follows it."""
    return f"the shot cuts to {text[0].lower()}{text[1:]}" if re.match(r"(?:A|An|The|One|Another)\s", text) else f"the shot cuts. {text}"


def joined_prompt(group: list[dict[str, Any]], fps: float, swap: bool = True) -> str:
    """The prompt of one render over a group of consecutive shots (one shot too), in the window's time and in upstream's window format: one line
    placing every painted frame, each cast member bound to every painted frame of the shots they are in, the frames as `reference` (swap) or
    `target keyframe` (edit), the cuts named in <Video 1>'s retention, and each later shot opening on its cut. Each shot's own prompt gives its
    content: its pictures are numbered after the shots before it, a cast member seen in several shots stays one subject, and each shot's
    [Shot N] and times move to where the shot starts in the window. A shot left as it is says so, from what was seen in it."""
    start = group[0]["source_range"][0]
    videos, places, summaries, details = [], [], [], []  # places: (picture, shot, time) for every painted frame
    subjects: dict[int, dict[str, Any]] = {}  # the window's subjects: name, pictures, shots, the line that defines them
    kept_lines: dict[str, str] = {}  # each subject's retention line, from the first shot that has it
    head = {"tag": "", "video": "", "style": "", "soundscape": "", "music": ""}
    picture_count = shot_count = 0
    cuts = [_clock((shot["source_range"][0] - start) / fps) for shot in group[1:]]
    for shot in group:
        offset = (shot["source_range"][0] - start) / fps
        own = list(range(picture_count + 1, picture_count + len(shot.get("painted_frames") or []) + 1))
        places += [(number, shot_count + 1, _clock((painted["frame"] - start) / fps)) for number, painted in zip(own, shot.get("painted_frames") or [])]
        if kept(shot):
            seen = str(shot.get("seen") or "").strip() or "The shot as it was."
            shot_count += 1
            videos.append(re.split(r"(?<=\.)\s", seen, maxsplit=1)[0].rstrip("."))
            details.append(f"[Shot {shot_count}] " + (f"At {_clock(offset)}, {_opening(seen)}" if offset else seen) + " It plays as in <Video 1>.")
            picture_count += len(own)
            continue
        found = _sections(shot["generation"]["settings"].get("prompt"), shot["id"])
        local: dict[int, int] = {}  # this shot's subject numbers in the window
        created = []  # the subjects this shot names first, with their lines as it writes them
        for line in found["subject_definitions"].splitlines():
            line = line.strip()
            if not line.startswith("<Subject "):
                continue
            match = _SUBJECT.match(line)
            again = next((number for number, item in subjects.items() if match and item["match"] and item["name"].casefold() == match[2].casefold()), None)
            local[int(re.match(r"<Subject (\d+)>", line)[1])] = again or len(subjects) + 1
            if again is None:
                subjects[len(subjects) + 1] = {"name": match[2] if match else "", "pictures": [], "shots": [], "match": match}
                created.append((len(subjects), line, match))

        def renumber(text: str, shift: bool = False) -> str:
            def token(match):
                kind, number, spoken, shot_no = match.groups()
                return (f"<Picture {picture_count + int(number)}>" if kind == "Picture" else f"<Subject {local[int(number)]}>" if kind == "Subject"
                        else f"(S{local[int(spoken)]})" if spoken else f"[Shot {shot_count + int(shot_no)}]")
            text = _TOKEN.sub(token, text)
            return _CLOCK.sub(lambda match: _clock(int(match[1]) * 60 + float(match[2]) + offset), text) if shift else text

        for number, line, match in created:
            subjects[number].update(line=renumber(line), looks=renumber(match[4]) if match else "")

        cast_here = [match for line in found["subject_definitions"].splitlines() if (match := _SUBJECT.match(line.strip()))]
        for line in found["subject_definitions"].splitlines():
            line = line.strip()
            if line.startswith("<Video 1>"):
                videos.append(line.split(":", 1)[-1].strip().rstrip("."))
            elif match := _SUBJECT.match(line):  # bound to the pictures it names; the only cast member of a shot shows in all its pictures
                named = [renumber(picture) for picture in re.findall(r"<Picture \d+>", match[3])]
                subjects[local[int(match[1])]]["pictures"] += named + ([f"<Picture {number}>" for number in own] if len(cast_here) == 1 else [])
        for number in local.values():
            subjects[number]["shots"].append(shot_count + 1)
        tag = re.match(r"^\[[^\]]+\]\s*", found["summary"])
        head["tag"] = head["tag"] or (tag[0].strip() if tag else "")
        summaries.append(renumber(found["summary"][tag.end() if tag else 0:]))
        for line in found["retention_analysis"].splitlines():
            line = renumber(line.strip())
            if not (key := _KEY.match(line)):
                continue
            if key[0] == "<Video 1>":
                head["video"] = head["video"] or line
            elif key[0].startswith("<Subject") and key[0] not in kept_lines:  # a painted frame's line is written below, as the template's format has it
                kept_lines[key[0]] = line
        lines = [line.strip() for line in found["detailed_description"].splitlines() if line.strip()]
        head["style"] = head["style"] or lines[0]
        shot_lines = [renumber(line, shift=True) for line in lines[1:]]
        if offset and shot_lines and (opened := re.match(rf"\[Shot {shot_count + 1}\]\s+(?:At {re.escape(_clock(offset))},?\s+)?(.*)", shot_lines[0])):
            shot_lines[0] = f"[Shot {shot_count + 1}] At {_clock(offset)}, {_opening(opened[1])}"
        details += shot_lines
        head["soundscape"], head["music"] = head["soundscape"] or found["overall_soundscape"], head["music"] or found["non_diegetic_music"]
        picture_count += len(own)
        shot_count += max([int(number) for number in re.findall(r"\[Shot (\d+)\]", found["detailed_description"])] or [1])
    numbered = lambda numbers: [f"[Shot {number}]" for number in dict.fromkeys(numbers)]  # noqa: E731
    definitions = [f"<Video 1> is the source video for {'the editing task' if swap else 'the target video edit'}: " + (f"{shot_count} shots: " if shot_count > 1 else "") + "; then ".join(videos) + "."]
    if swap and places:
        definitions.append(f"{_listed([f'<Picture {number}>' for number, _, _ in places])} {'is a frame' if len(places) == 1 else 'are frames'} of the target video, "
                           + ", ".join(f"<Picture {number}> from [Shot {shot}] at {time}" for number, shot, time in places) + ".")
    else:
        definitions += [f"<Picture {number}> is a frame of the target video, exactly as the target video shows it at that moment, from [Shot {shot}] at {time}." for number, shot, time in places]
    for number, item in subjects.items():
        definitions.append(f"<Subject {number}> is {item['name']} in {_listed(list(dict.fromkeys(item['pictures'])))}: {item['looks']}" if item["match"] and item["pictures"] else item["line"])
    video = head["video"] or "<Video 1> (source video editing): fully_preserved - the camera, the motion and the timing of every shot."
    if cuts:
        video = re.sub(r"fully_preserved\s*-\s*", f"fully_preserved - the {shot_count} shots and the cuts at {_listed(cuts)}, ", video, count=1)
    retention = [video]
    for number, shot, _ in places:
        retention.append(f"<Picture {number}> (appears in [Shot {shot}]): reference - a frame of the target video. The people in it, their faces and clothing, and the background, framing and lighting are taken from <Picture {number}>." if swap
                         else f"<Picture {number}> (target keyframe): fully_preserved - the target video passes through this exact frame.")
    if swap and len(places) > 1:
        retention[-1] += " A person shown in several pictures is the same person."
    retention += [re.sub(r"\(appears in [^)]*\)", f"(appears in {', '.join(numbered(subjects[int(key[9:-1])]['shots']))})", line, count=1) for key, line in kept_lines.items()]
    sentences = list(dict.fromkeys(sentence for sentence in re.split(r"(?<=\.)\s+", summaries[0].strip()) if sentence)) if summaries else []
    for text in summaries[1:]:  # a later shot adds what it says of someone not named yet: the rest is the same change again
        sentences += [sentence for sentence in re.split(r"(?<=\.)\s+", text.strip()) if sentence and set(re.findall(r"<Subject \d+>", sentence)) - set(re.findall(r"<Subject \d+>", " ".join(sentences)))]
    if shot_count > 1:  # the window's shots and cuts, as one: what a single shot said it keeps would leave them out
        sentences = [sentence for sentence in sentences if not ("<Video 1>" in sentence and re.search(r"\b(?:kept|preserved|unchanged)\b", sentence))]
        sentences.append("All other people, the place and the soundtrack are unchanged; the shots, cuts, camera, motion and timing of <Video 1> are kept.")
    blocks = {"subject_definitions": definitions, "summary": [" ".join(filter(None, [head["tag"], *sentences]))], "retention_analysis": retention,
              "detailed_description": [head["style"] or "Realistic footage.", *details], "overall_soundscape": [head["soundscape"] or "The soundtrack of <Video 1>, unchanged."],
              "non_diegetic_music": [head["music"] or "N/A"]}
    return "\n\n".join(f"{name}:\n" + "\n".join(blocks[name]) for name in SECTIONS)
