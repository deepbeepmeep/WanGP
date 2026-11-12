# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""The production document: brief, cast, style, shots with their takes, edit, deliverables and the director's log.

It is the single source of truth of a production. The Showrunner UI renders it; the user's edits and Deepy's toolbox actions
change it through the same operations (apply_ops), with lock and revision rules so that neither overwrites the other's work."""
from __future__ import annotations

import copy
import difflib
import re
import hashlib
import json
import time
import uuid
from typing import Any

SCHEMA = 1
KINDS = ("music_video", "short_film", "ad", "trailer", "story", "remake", "free")
STAGES = ("brief", "planning", "review", "shooting", "editing", "delivered")
SHOT_STATUSES = ("planned", "queued", "rendering", "rendered", "approved", "rejected", "failed")
TRANSITIONS = ("cut", "crossfade", "fade_black", "fade_white")
ASPECTS = ("16:9", "9:16", "1:1", "4:3", "2.39:1")
AUDIO_MODES = ("song", "clips", "mix")
# The brief's format as a generation resolution, from WanGP's standard choices (shared/resolutions.py) where one exists.
RESOLUTIONS = {"16:9": {"480p": "832x480", "720p": "1280x720", "1080p": "1920x1088"}, "9:16": {"480p": "480x832", "720p": "720x1280", "1080p": "1088x1920"},
               "1:1": {"480p": "720x720", "720p": "960x960", "1080p": "1440x1440"}, "4:3": {"480p": "832x624", "720p": "1104x832", "1080p": "1440x1088"},
               "2.39:1": {"480p": "1152x480", "720p": "1280x544", "1080p": "1920x832"}}


def resolution(brief: dict[str, Any]) -> str:
    """The brief's target: every picture is made at it, and every shot shot final."""
    return RESOLUTIONS[brief["aspect"]][brief["resolution"]]


def tier_name(brief: dict[str, Any], final: bool = False) -> str:
    """The resolution a video shot renders at: the draft's, when the brief has one, until the shot is shot final."""
    return brief["resolution"] if final or not brief.get("draft_resolution") else brief["draft_resolution"]


class OperationError(ValueError):
    """A refused operation; current holds the item as it is now when another author changed it meanwhile."""

    def __init__(self, message: str, current: Any = None):
        super().__init__(message)
        self.current = current


def now() -> float:
    return round(time.time(), 3)


def new_document(title: str, kind: str = "free", **brief: Any) -> dict[str, Any]:
    doc = {"schema": SCHEMA, "id": "", "title": title.strip() or "Untitled production", "created": now(), "updated": now(), "revision": 0, "stage": "brief",
           "brief": {"kind": kind}, "cast": [], "style": {}, "audio": {"analysis": None}, "shots": [], "edit": {}, "deliverables": [], "log": [],
           "director": {"state": "idle"}, "budget": {}}
    doc["brief"].update(brief)
    return normalize(doc)


def normalize(doc: dict[str, Any]) -> dict[str, Any]:
    """Fill the defaults of every section in place (older or hand-edited documents load the same way)."""
    brief = doc.setdefault("brief", {})
    for key, value in {"kind": "free", "text": "", "song": None, "script": "", "references": [], "duration_s": None, "aspect": "16:9", "resolution": "720p", "draft_resolution": None,
                       "style_tags": [], "style": "", "pacing": 0, "language": "en", "plan_first": True, "budget_minutes": None, "templates": {},
                       "read_other_productions": True, "review_before_edit": False, "shot_min_s": None, "shot_max_s": None, "voices_from_model": False,
                       "group_renders": True, "render_min_s": RENDER_MIN_S, "render_max_s": None,  # a remake's renders: shots join until one lasts render_min_s, up to render_max_s or its template's window
                       "song_voices": []}.items():  # who sings the song the showrunner composes, first heard first: cast ids (song models take no voice)
        brief.setdefault(key, copy.deepcopy(value))
    if brief["kind"] not in KINDS:
        brief["kind"] = "free"
    for member in doc.setdefault("cast", []):
        for key, value in {"name": "", "role": "", "description": "", "images": [], "master": None, "voice": None, "voice_description": "", "voice_text": "", "lora": None,
                           "locked": False, "looks": []}.items():
            member.setdefault(key, copy.deepcopy(value))
        if not isinstance(member["lora"], dict):  # {file, folder (its model family's LoRA folder), multiplier, trigger}
            member["lora"] = None
        for look in member["looks"]:  # a costume or an age of the member: its master for the sequences it is worn in
            for key, value in {"name": "", "master": None, "sections": [], "shots": [], "description": ""}.items():
                look.setdefault(key, copy.deepcopy(value))
    style = doc.setdefault("style", {})
    for key in ("look", "palette", "camera", "lighting", "notes", "text_design"):
        style.setdefault(key, "")
    style.setdefault("props", [])
    doc.setdefault("audio", {}).setdefault("analysis", None)
    for sequence in doc.setdefault("outline", []):
        for key, value in {"name": "", "summary": "", "duration_s": None}.items():
            sequence.setdefault(key, value)
    for shot in doc.setdefault("shots", []):
        normalize_shot(shot)
        shot["cast"] = cast_ids(doc, shot["cast"])
    edit = doc.setdefault("edit", {})
    for key, value in {"audio_mode": "song" if brief["kind"] == "music_video" else "clips", "audio_tracks": [], "subtitles": [], "overlays": [], "fps": None, "level_voices": False}.items():
        edit.setdefault(key, copy.deepcopy(value))
    for overlay in edit["overlays"]:
        for key, value in {"text": "", "subtitle": "", "style": "", "design": "", "start_s": 0.0, "end_s": 3.0, "fade_s": 0.5, "by": "user"}.items():
            overlay.setdefault(key, value)
    doc.setdefault("deliverables", [])
    doc.setdefault("archived", False)  # hidden from the home grid, everything kept
    doc.setdefault("bin", [])
    doc.setdefault("log", [])
    doc.setdefault("director", {}).setdefault("state", "idle")
    doc.setdefault("budget", {}).setdefault("gpu_seconds", 0.0)
    if doc.get("stage") not in STAGES:
        doc["stage"] = "brief"
    doc["edit_key"] = edit_key(doc)
    return doc


def normalize_shot(shot: dict[str, Any]) -> dict[str, Any]:
    for key, value in {"title": "", "section": "", "intent": "", "camera": "", "cast": [], "dialogue": [], "duration_s": 5.0, "audio_slice": None, "generation": {},
                       "status": "planned", "locked": False, "included": True, "takes": [], "chosen_take": None, "trim": [0.0, None], "speed": 1.0, "sound": True,
                       "transition": {"type": "cut", "duration_s": 0.0}, "notes": "", "updated_by": "", "updated_rev": 0, "cut_extra_words": True, "line_cut": None,
                       "rendering_with": None}.items():  # rendering_with: the shot whose render renders this one too, while it runs
        shot.setdefault(key, copy.deepcopy(value))
    generation = shot["generation"]
    generation.setdefault("template", "gen_video")
    generation.setdefault("settings", {})
    if shot["status"] not in SHOT_STATUSES:
        shot["status"] = "planned"
    for take in shot["takes"]:
        for key, value in {"kind": "video", "settings": {}, "by": "deepy", "created": now(), "checks": {}, "verdict": None, "note": "", "gpu_s": None}.items():
            take.setdefault(key, copy.deepcopy(value))
        # the showrunner's own verdict, kept beside the user's (takes judged before it was kept: the verdict when it is the showrunner's)
        take.setdefault("showrunner_verdict", copy.deepcopy(take["verdict"]) if (take["verdict"] or {}).get("by", "user") != "user" else None)
    return shot


def _next_id(prefix: str, items: list[dict[str, Any]]) -> str:
    used = {item.get("id") for item in items}
    index = len(items) + 1
    while f"{prefix}{index}" in used:
        index += 1
    return f"{prefix}{index}"


def find(items: list[dict[str, Any]], item_id: str, label: str) -> dict[str, Any]:
    for item in items:
        if item.get("id") == item_id:
            return item
    raise OperationError(f"No {label} has the id {item_id!r}.")


def shot_prompt(shot: dict[str, Any]) -> str:
    return str(shot["generation"].get("settings", {}).get("prompt", "") or "")


_QUOTED_CLAUSE = re.compile(r"""[^.;!?]*(?<!\w)['"\u201c\u2018\u00ab][^'"\u201d\u2019\u00bb\n]{2,}?['"\u201d\u2019\u00bb](?!\w)[^.;!?]*[.;!?]?""")


def without_quoted_words(text: str) -> str:
    """A text without the clauses that quote someone's words: a model that voices its prompt says them, whoever they were written for."""
    return _QUOTED_CLAUSE.sub("", text).strip(" ;,")


def compiled_prompt(doc: dict[str, Any], shot: dict[str, Any], lip_sync: bool = True) -> str:
    """A shot's prompt from what the user wrote on its card and in the brief, for a shot rendered without the showrunner: no language model rewrites it. Empty until the shot says what happens.
    lip_sync: its speakers follow a recording, and the prompt asks for it. Off, its model says what the prompt quotes, so the cast's quoted words stay out."""
    if not str(shot.get("intent") or "").strip():
        return ""
    members = {member["id"]: member for member in doc["cast"]}
    shown = [members[cast_id] for cast_id in shot.get("cast") or [] if cast_id in members]
    speakers = {line.get("speaker") for line in shot.get("dialogue") or []}
    look = ", ".join(str(item).strip() for item in [*(doc["brief"].get("style_tags") or []), doc["brief"].get("style")] if str(item or "").strip())
    described = lambda member: member["description"] if lip_sync else without_quoted_words(member["description"])
    parts = [shot["intent"], shot.get("camera"), *(f"{member['name']}: {described(member)}" for member in shown if member.get("description") and described(member)),
             *(f"{member['name']} is speaking with natural lip movements synchronized" for member in shown if lip_sync and member["id"] in speakers), look]
    return " ".join(f"{part.strip().rstrip('.')}." for part in parts if str(part or "").strip())


def chosen_take(shot: dict[str, Any]) -> dict[str, Any] | None:
    takes = shot.get("takes") or []
    if shot.get("chosen_take"):
        for take in takes:
            if take["id"] == shot["chosen_take"]:
                return take
    return takes[-1] if takes else None


def sequence(doc: dict[str, Any]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """(shot, take) pairs of the edit, in playing order: included shots that have a take."""
    pairs = []
    for shot in doc["shots"]:
        take = chosen_take(shot)
        if shot.get("included", True) and take is not None and take.get("kind", "video") in ("video", "image"):
            pairs.append((shot, take))
    return pairs


def awaiting_approval(doc: dict[str, Any]) -> list[str]:
    """Included shots whose take in the edit the user has not approved yet (or that have no take)."""
    approved = lambda take: bool(take) and (take.get("verdict") or {}).get("by") == "user" and take["verdict"].get("ok")
    return [shot["id"] for shot in doc["shots"] if shot.get("included", True) and not approved(chosen_take(shot))]


def shoot_rates(own: list[dict[str, Any]], others: list[dict[str, Any]], size: str) -> dict[str, dict[str, Any]]:
    """Measured generation time per template, from timed takes ({template, kind, seconds, duration_s, width, height}): seconds per second of
    video (per picture for images). This production's takes first, then other productions' takes; only takes of about the planned frame
    size count (generation time grows fast with the size)."""
    width, height = (int(value) for value in size.split("x"))
    near = lambda item: item["width"] and item["height"] and abs(item["width"] * item["height"] / (width * height) - 1) <= 0.15
    rates: dict[str, dict[str, Any]] = {}
    for source, samples in (("this production", own), ("other productions", others)):
        found: dict[str, list[dict[str, Any]]] = {}
        for item in samples:
            if item["template"] not in rates and item["seconds"] and near(item) and (item["kind"] == "image" or item["duration_s"]):
                found.setdefault(item["template"], []).append(item)
        for template, items in found.items():
            videos = [item for item in items if item["kind"] != "image"]
            rate = {"per_second": round(sum(item["seconds"] for item in videos) / sum(float(item["duration_s"]) for item in videos), 2)} if videos else \
                {"per_picture": round(sum(item["seconds"] for item in items) / len(items), 2)}
            rates[template] = {**rate, "takes": len(videos or items), "from": source}
    return rates


def speed_of(shot: dict[str, Any], take: dict[str, Any] | None) -> float:
    """How fast a shot's take plays in the edit (a still has no speed). Its trim stays in seconds of the take."""
    return 1.0 if (take or {}).get("kind") == "image" else float(shot.get("speed") or 1.0)


def out_point(shot: dict[str, Any], length: float) -> float:
    """Where a take ends in the edit without a trim of the user's: at the shot's planned length when the take came out longer. Models round
    lengths up to their frame steps (10.22 s for a 10.125 s plan), and the cuts the director planned on the bars, and the song slices of
    lip-synced shots, must not drift by that much per shot."""
    planned = shot.get("duration_s")
    return min(float(length), float(planned)) if planned else float(length)


def next_step(doc: dict[str, Any]) -> str:
    """What is up to the user once the showrunner's task is over, in a sentence for a notification (the page says it too: nextStep in app.js)."""
    included = [shot for shot in doc["shots"] if shot.get("included", True)]
    many = lambda number, word: f"{number} {word}{'' if number == 1 else 's'}"
    todo = sum(not shot.get("locked") and (not shot["takes"] or shot["status"] in ("rejected", "failed")) for shot in included)
    review = sum(shot["status"] == "rendered" for shot in included)
    if doc["shots"] and not any(shot["takes"] for shot in included):
        return f"The plan is ready: check its {many(len(doc['shots']), 'shot')} on the Board, then press Shoot."
    if todo:
        return f"{many(todo, 'shot')} still {'needs' if todo == 1 else 'need'} a take: press Shoot on the Board."
    if doc["shots"] and doc["brief"].get("review_before_edit") and review:
        return f"{many(review, 'take')} {'waits' if review == 1 else 'wait'} for your review on the Board."
    if doc["shots"] and not doc["deliverables"]:
        return "Every shot has a take: press Assemble to make the video."
    if doc["deliverables"] and doc["deliverables"][-1].get("edit_key") == edit_key(doc):
        return f"Version {len(doc['deliverables'])} is ready: watch it on Deliver."
    return "The showrunner has finished its task."


def take_length(doc: dict[str, Any], shot: dict[str, Any]) -> float:
    """The seconds of take a shot's place in the edit uses at most: its planned length or the end of its trim, and on the song, what it plays on through
    a transition into the next shot (the song keeps that shot on its part: the outgoing one runs into it, at most 1 s)."""
    length = max(float(shot.get("duration_s") or 0.0), float((list(shot.get("trim") or []) + [None, None])[1] or 0.0))
    if doc["brief"].get("song") and doc["edit"].get("audio_mode") in ("song", "mix"):
        included = [item for item in doc["shots"] if item.get("included", True)]
        following = next((after for before, after in zip(included, included[1:]) if before["id"] == shot["id"]), None)
        if following is not None and (following.get("transition") or {}).get("type", "cut") != "cut":
            length += min(float(following["transition"].get("duration_s") or 0.5), 1.0) * float(shot.get("speed") or 1.0)
    return length


def cut_lines(doc: dict[str, Any]) -> list[str]:
    """Lines driving a shot's lips that are longer than what the cut plays of the shot: their end is lost."""
    played = {item["shot"]: item["duration_s"] for item in timeline(doc)}
    found = []
    for shot in doc["shots"]:
        take = chosen_take(shot)
        spoken = sum(float(line.get("duration_s") or 0.0) for line in spoken_lines(doc, shot))
        if shot.get("included", True) and take and (take.get("settings") or {}).get("audio_guide") and spoken > played.get(shot["id"], 0.0) + 0.05:
            found.append(f"Shot {shot['id']} plays {played[shot['id']]:.2f} s and its line lasts {spoken:.2f} s: the end of the line is cut. Set its `trim` to hold the whole line (take it again if the take is too short).")
        elif shot.get("included", True) and take and spoken and played.get(shot["id"], 9.0) < 2.0:
            found.append(f"Shot {shot['id']} plays only {played[shot['id']]:.2f} s for its line: too short to register. Follow it with a silent reaction shot, or set a longer `trim` after `transcribe_media` shows that part of the take says nothing.")
    return found


def on_the_song(doc: dict[str, Any]) -> bool:
    """The cut plays the song straight through: its shots keep their place on it, and their lines are lyrics."""
    return bool(doc["brief"].get("song")) and doc["edit"].get("audio_mode") == "song"


def lyric_lines(doc: dict[str, Any]) -> list[str]:
    """The brief's lyrics, line by line, without their section markers."""
    return [line.strip() for line in str(doc["brief"].get("script") or "").splitlines() if line.strip() and not line.strip().startswith("[")]


_SCRIPT_LABEL = re.compile(r"^[*_\s]*([^\W\d_][^:()\d]{0,38}?)[*_\s]*(?:\([^)]*\))?[*_\s]*:[*_\s]*(\S.*)$")
_SCRIPT_HEADINGS = {"scene", "act", "shot", "chapter", "part", "sequence", "title", "note", "notes", "setting", "location", "action", "description", "int", "ext", "fade", "cut"}


def script_speech(doc: dict[str, Any]) -> list[str]:
    """What the brief's script has spoken, line by line: after a speaker's label (NARRATOR: ..., MIA (whispering): ...) or under a speaker's name in capitals. Text without a speaker may be anything, and says nothing."""
    spoken, speaking = [], False
    for text in (line.strip() for line in str(doc["brief"].get("script") or "").splitlines()):
        label = _SCRIPT_LABEL.match(text)
        if not text:
            speaking = False
        elif text.startswith(("[", "#")):
            continue
        elif label and len(label[1].split()) <= 3 and label[1].split()[0].casefold().rstrip(".") not in _SCRIPT_HEADINGS:
            spoken.append(label[2])
            speaking = False
        elif text.isupper() and len(text.split()) <= 3 and not text.endswith((".", ":")) and text.split()[0].rstrip(".") not in ("INT", "EXT"):
            speaking = True
        elif speaking:
            spoken.append(text)
    return spoken


def _script_words(text: str) -> list[str]:
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*", re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", text).casefold().replace("\u2019", "'"))


def says(text: str, line: str) -> bool:
    """Whether a text has a line word for word: capitals, punctuation and directions do not count."""
    run = lambda words: f" {' '.join(words)} "
    return run(_script_words(line)) in run(_script_words(text))


def script_differences(doc: dict[str, Any]) -> tuple[list[str], list[str]]:
    """The script's spoken lines that no shot in the edit says word for word, and the lines of shots (other than those the user edited) that the script does not have.
    Capitals and punctuation do not count, and a line can run over several shots."""
    run = lambda lines: f" {' '.join(word for line in lines for word in _script_words(line))} "
    written = [line for line in script_speech(doc) if _script_words(line)]
    shots = [shot for shot in doc["shots"] if shot.get("included", True)]
    said, script = run(line["line"] for shot in shots for line in shot.get("dialogue") or []), run(written)
    missing = [line for line in written if run([line]) not in said]
    added = [f"{shot['id']}: {line['line']}" for shot in shots if written and shot.get("updated_by") != "user" for line in shot.get("dialogue") or [] if _script_words(line["line"]) and run([line["line"]]) not in script]
    return missing, added


def _lyric_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower().replace("\u2019", "'"))


def _same_word(written: str, heard: str) -> bool:
    return written == heard or (len(written) > 3 and written[:4] == heard[:4])  # groove / grooves, skate / skates


def _aligned(written: list[str], heard: list[str]) -> list[tuple[int, int]]:
    """The words two lists share in order, as many as possible (choruses repeat: pairing the longest passage first can pair a chorus with another one)."""
    best = [[0] * (len(heard) + 1) for _ in range(len(written) + 1)]
    for row in range(len(written) - 1, -1, -1):
        for column in range(len(heard) - 1, -1, -1):
            best[row][column] = max(best[row + 1][column], best[row][column + 1], best[row + 1][column + 1] + 1 if _same_word(written[row], heard[column]) else 0)
    pairs, row, column = [], 0, 0
    while row < len(written) and column < len(heard):
        if _same_word(written[row], heard[column]) and best[row][column] == best[row + 1][column + 1] + 1:
            pairs.append((row, column))
            row, column = row + 1, column + 1
        elif best[row + 1][column] >= best[row][column + 1]:
            row += 1
        else:
            column += 1
    return pairs


def sung_lines(lines: list[str], words: list[tuple[str, float, float]]) -> list[dict[str, Any]]:
    """Which lines of the lyrics a song really sings, and when, from the words heard in it [(word, start, end)]. A line is sung when at least half of its words are heard, in
    order and close together: a transcript misses words here and there, but it does not lose a whole line that is sung (Whisper reads most of a song's clear lines word for word)."""
    said = [(_lyric_words(word) or [""])[0] for word, _, _ in words]
    written, first_word = [], []
    for line in lines:
        first_word.append(len(written))
        written += _lyric_words(line)
    found = [[] for _ in lines]
    for row, column in _aligned(written, said):
        found[next(index for index in range(len(lines) - 1, -1, -1) if first_word[index] <= row)].append((row, column))
    counts = [len(_lyric_words(line)) for line in lines]
    sung = [bool(pairs) and len(pairs) * 2 >= count and words[pairs[-1][1]][2] - words[pairs[0][1]][1] <= 1.5 * count + 2.0 for pairs, count in zip(found, counts)]
    result, floor, taken = [], 0.0, -1  # taken: the last word heard of the lines before
    for index, line in enumerate(lines):
        count, pairs = counts[index], found[index]
        item = {"line": line, "heard": len(pairs), "words": count, "sung": False, "start_s": None, "end_s": None}
        if sung[index]:
            following = next((found[later][0][1] for later in range(index + 1, len(lines)) if sung[later]), len(words))
            first, before = _reach(words, pairs[0][1], -1, pairs[0][0] - first_word[index], taken)  # the words it misses are heard next to it, read otherwise
            last, after = _reach(words, pairs[-1][1], 1, first_word[index] + count - 1 - pairs[-1][0], following)
            start = max(floor, words[first][1] - 0.3 * before)  # about 0.3 s a word missed and not heard
            floor, taken = words[last][2] + 0.3 * after, last
            item.update(sung=True, start_s=round(start, 2), end_s=round(floor, 2))
        result.append(item)
    _place_unclear(result, words)
    return result


def _reach(words: list[tuple[str, float, float]], column: int, step: int, missed: int, bound: int) -> tuple[int, int]:
    """The heard word a line reaches from its word heard at `column`, going over at most `missed` neighbours sung on without a pause and never to `bound`;
    with the number of missed words left over."""
    while missed and bound != column + step and abs(words[column + step][1 if step > 0 else 2] - words[column][2 if step > 0 else 1]) <= 1.0:
        column, missed = column + step, missed - 1
    return column, missed


def _place_unclear(result: list[dict[str, Any]], words: list[tuple[str, float, float]]) -> None:
    """Lines a transcript cannot read, sung between two lines it reads: where the words heard between them read like these lines, they are these lines sung unclearly,
    shared by their number of words. They get their times (and a subtitle) with unclear=True, and stay not sung as written."""
    letters = lambda text: re.sub(r"[^a-z0-9]", "", text.lower())
    index = 0
    while index < len(result):
        if result[index]["sung"]:
            index += 1
            continue
        run = [index]
        while run[-1] + 1 < len(result) and not result[run[-1] + 1]["sung"]:
            run.append(run[-1] + 1)
        index = run[-1] + 1
        before = result[run[0] - 1]["end_s"] if run[0] else -1.0
        after = result[index]["start_s"] if index < len(result) else float("inf")
        heard = [word for word in words if before <= word[1] < after]
        if not heard or difflib.SequenceMatcher(None, letters(" ".join(result[line]["line"] for line in run)), letters(" ".join(word for word, _, _ in heard)), autojunk=False).ratio() < 0.5:
            continue
        total, done = sum(result[line]["words"] for line in run), 0
        for line in run:
            share = heard[round(len(heard) * done / total):round(len(heard) * (done + result[line]["words"]) / total)]
            done += result[line]["words"]
            if share:
                result[line].update(unclear=True, start_s=round(max(share[0][1], before), 2), end_s=round(share[-1][2], 2))


WORD_CUT_MARGIN_S = 0.1  # a cut before a word heard: its start is known to about this much


def _heard_tokens(text: str) -> list[str]:
    """A text's words, in any alphabet; each character of the scripts written without spaces."""
    return re.findall(r"[\u3040-\u30ff\u3400-\u9fff]|[^\W_\u3040-\u30ff\u3400-\u9fff]+(?:'[^\W_]+)*", text.casefold().replace("\u2019", "'"))


def heard_lines(lines: list[str], words: list[tuple[str, float, float]]) -> dict[str, Any]:
    """A take against the lines its model was asked to say, from the words heard in it [(word, start, end)]: the lines it does not say, those it says with a word heard differently, the words
    it adds before, between and after them, and where to cut. A line is found where most of its words come close together, not where half of them are scattered: a model that gets a line wrong
    still says some of its words. Nothing is cut around a line it did not find, nor between two lines: that would jump."""
    heard = [(token, start, end) for word, start, end in words for token in _heard_tokens(word)]
    said, spans, missing, inexact, offset = [token for token, _, _ in heard], [], [], [], 0
    for line in lines:
        wanted, best = _heard_tokens(line), None
        for start in range(offset, len(said)):
            pairs = _aligned(wanted, said[start:start + len(wanted) + 2])
            if pairs and (best is None or len(pairs) > len(best[1])):
                best = (start, pairs)
        if best is None or len(best[1]) * 5 < len(wanted) * 4:
            missing.append(line)
            continue
        spans.append((best[0] + best[1][0][1], best[0] + best[1][-1][1]))
        offset = spans[-1][1] + 1
        if len(best[1]) < len(wanted):
            inexact.append(line)
    before = said[:spans[0][0]] if spans else []
    after = said[spans[-1][1] + 1:] if spans else []
    result = {"text": " ".join(word.strip() for word, _, _ in words), "missing": missing, "inexact": inexact, "before": " ".join(before), "after": " ".join(after),
              "between": [" ".join(said[left[1] + 1:right[0]]) for left, right in zip(spans, spans[1:]) if right[0] > left[1] + 1], "cut": None}
    if spans and not missing and (before or after):
        first, last = spans[0][0], spans[-1][1]
        start = heard[first][1] - min(WORD_CUT_MARGIN_S, max(0.0, heard[first][1] - heard[first - 1][2]) / 2) if before else 0.0
        end = heard[last + 1][1] - min(WORD_CUT_MARGIN_S, max(0.0, heard[last + 1][1] - heard[last][2]) / 2) if after else None
        result["cut"] = [round(start, 2), None if end is None else round(end, 2)]
    return result


def follow_line_cut(shot: dict[str, Any]) -> None:
    """A trim made around a take's own lines follows the take in use and the shot's switch; any other trim was set on purpose, and stays."""
    if (shot.get("trim") or [0.0, None]) != (shot.get("line_cut") or [0.0, None]):
        return
    cut = ((chosen_take(shot) or {}).get("heard") or {}).get("cut") if shot.get("cut_extra_words", True) else None
    shot["line_cut"], shot["trim"] = copy.deepcopy(cut), copy.deepcopy(cut) if cut else [0.0, None]


def checked_lyrics(doc: dict[str, Any]) -> list[dict[str, Any]] | None:
    """The lines check_song measured on the production's current song; None before it ran on this song."""
    checked = doc["audio"].get("lyrics")
    return checked["lines"] if checked and doc["brief"].get("song") and checked["song"] == str(doc["brief"]["song"]).replace("\\", "/").rsplit("/", 1)[-1] else None


def song_words(doc: dict[str, Any]) -> list[str]:
    """On a cut that plays a song: lyrics the production does not have, a song whose words were not checked, and lines it does not sing."""
    if not on_the_song(doc):
        return []
    if not lyric_lines(doc):
        return ["The production has the song but not its lyrics, so nothing says which lines it really sings or times the subtitles on them: pass the words it was made from to check_song(lyrics=...), unless it has none."]
    lines = checked_lyrics(doc)
    if lines is None:
        return ["The song's words are not checked: check_song says which lines it really sings, and times the subtitles on them."]
    missing = [line["line"] for line in lines if not line["sung"]]
    untimed = sum(line["start_s"] is None for line in lines)
    warnings = [f"The song does not sing {len(missing)} of its {len(lines)} lines as written" + (", and they get no subtitle" if untimed == len(missing) else f" ({untimed} get no subtitle, the others are subtitled where they are sung unclearly)" if untimed else " (they are subtitled where they are sung unclearly)") + ": " + " / ".join(f'"{line}"' for line in missing[:8]) + (" ..." if len(missing) > 8 else "") + ". Compose it again before shooting on it, with fewer lines or more length."] if missing else []
    pitch, kinds = doc["audio"]["lyrics"].get("voice_hz"), singer_kinds(doc)
    if len(kinds) == 1 and voice_off(next(iter(kinds)), pitch):
        warnings.append(f"The song's lead voice measures {pitch:.0f} Hz, which reads as a {'man' if next(iter(kinds)) == 'woman' else 'woman'}, and the singer on screen is a {next(iter(kinds))}: compose it again, with the lead voice named first in its style.")
    return warnings


def song_drift(doc: dict[str, Any]) -> list[str]:
    """On a cut that plays the song: shots away from the part of the song their lips were made from, and takes made from a recording the cut does not play."""
    if not on_the_song(doc):
        return []
    start, cut, warnings = float(doc["edit"].get("song_start_s") or 0.0), {item["shot"]: item for item in timeline(doc)}, []
    away, spoken, short, longer = [], [], [], []
    for shot in doc["shots"]:
        item, take = cut.get(shot["id"]), chosen_take(shot)
        if item is None or take is None:
            continue
        if has_lips(doc, shot) and abs(item["start_s"] + start - shot["audio_slice"][0] - _trim_of(shot)[0]) > 0.1:
            away.append((shot["id"], shot["audio_slice"][0] + _trim_of(shot)[0] - start - item["start_s"]))
        if has_lips(doc, shot) and float(shot["duration_s"]) > shot["audio_slice"][1] - shot["audio_slice"][0] + 0.1:
            longer.append(shot["id"])
        if (take.get("settings") or {}).get("audio_guide") and not shot.get("audio_slice"):
            spoken.append(shot["id"])
        if abs(item["duration_s"] - float(shot["duration_s"])) > 0.1 and not away:  # the shots before the first one away from its part
            short.append(f"{shot['id']} plays {item['duration_s']:.2f} s of its planned {float(shot['duration_s']):.2f} s")
    if away:
        worst = max(away, key=lambda found: abs(found[1]))
        warnings.append(f"From {away[0][0]} on, shots start up to {abs(worst[1]):.2f} s away from the part of the song their lips follow, as the song plays straight through"
                        + (f": {'; '.join(short)}. Give it back its planned length" if short else ". Move their cuts back onto their slices") + " (update_shot: duration_s, trim null).")
    if spoken:
        warnings.append(f"Shots {', '.join(spoken)} follow a recording the cut does not play: on a song their lines are lyrics. Give each its audio_slice, the part of the song under it, and render it again.")
    if longer:
        warnings.append(f"Shots {', '.join(longer)} last longer than the part of the song they were made from, and their lips stop following it: give each a longer audio_slice and render it again.")
    return warnings


def missing_ends(doc: dict[str, Any]) -> list[str]:
    """A cut with no title or no end card: the viewer never reads its name, and it stops on its last image."""
    included = [shot for shot in doc["shots"] if shot.get("included", True)]
    if not included or doc["brief"].get("kind") == "remake":  # a remake is the source video changed: it adds nothing the user did not ask for
        return []
    styles = {item.get("style") for item in doc["edit"]["overlays"]}
    notes = [str((chosen_take(shot) or {}).get("note") or "") for shot in included]
    title = styles & {"title", "chapter"} or any(note.startswith("Chapter card") for note in notes) or included[0]["generation"].get("template") == "draw_svg"
    end = "end_card" in styles or any(note.startswith("End card") for note in notes) or included[-1]["generation"].get("template") == "draw_svg"
    missing = [name for name, there in (("title", title), ("end card", end)) if not there]
    return [f"The cut has no {' and no '.join(missing)}: add {'them' if len(missing) > 1 else 'it'}, in the production's text design, unless the user asked for none."] if missing else []


def singing_without_song(doc: dict[str, Any]) -> list[str]:
    """Shots whose prompt has someone singing with no sung audio driving their lips: the mouth moves, out of time with the song."""
    if doc["brief"].get("kind") != "music_video":
        return []
    def driven(shot):  # a take by what it was made with, a shot still to shoot by what it asks
        take = chosen_take(shot)
        return bool((take.get("settings") or {}).get("audio_guide")) if take else bool(shot.get("audio_slice") or (shot["generation"].get("settings") or {}).get("audio_guide"))
    found = [shot["id"] for shot in doc["shots"] if shot.get("included", True) and shot.get("cast") and re.search(r"\bsing(s|ing)?\b", shot_prompt(shot), re.IGNORECASE) and not driven(shot)]
    if not found:
        return []
    return [f"Shots {', '.join(found)} show someone singing and are not driven by the song: their lips will not match. Give each its `audio_slice` and the song as `audio_guide` and shoot it again, or rewrite it to show something else with the mouth closed."]


def planned_length(shot: dict[str, Any]) -> float:
    """How long a shot without a take will play: its planned length, or less when someone it shows speaks, as the shot ends after the line (about 0.37 s a word until the line is spoken)."""
    planned = float(shot.get("duration_s") or 0.0)
    lines = [line for line in shot.get("dialogue") or [] if line.get("line") and line.get("speaker") in (shot.get("cast") or [])]
    if not lines or not planned:
        return planned
    spoken = sum(float(line.get("duration_s") or 0.37 * len(_lyric_words(line["line"]))) for line in lines)
    return min(planned, spoken + BETWEEN_LINES * (len(lines) - 1) + AFTER_LINE)


def timeline(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Start time and duration of every included shot (planned durations where no take exists yet), in whole frames of the edit as the
    assembly cuts them, so that sound and text placed at a shot's start stay on its picture."""
    included = [shot for shot in doc["shots"] if shot.get("included", True)]
    fps = float(doc.get("edit", {}).get("fps") or next(((take or {}).get("fps") for shot in included if (take := chosen_take(shot)) and take.get("fps")), None) or 24.0)
    song_driven = bool(doc["brief"].get("song")) and doc.get("edit", {}).get("audio_mode") != "clips"  # the song is the clock: transitions keep every shot on its part
    position, result = 0.0, []
    for shot in included:
        take = chosen_take(shot)
        start_trim, end_trim = (list(shot.get("trim") or [0.0, None]) + [None, None])[:2]
        length = (take or {}).get("duration_s") or (shot.get("duration_s") if song_driven or doc["brief"].get("voices_from_model") else planned_length(shot)) or 0.0
        end = end_trim if end_trim is not None else length if take is None else out_point(shot, length)
        duration = round(max(0.0, float(end) - float(start_trim or 0.0)) / speed_of(shot, take) * fps) / fps
        transition = shot.get("transition") or {}
        if result and not song_driven and transition.get("type", "cut") != "cut":  # as the export: a transition overlaps the two shots, and what follows starts earlier
            overlap = min(float(transition.get("duration_s") or 0.5), duration / 2, position / 2)
            position -= overlap if overlap > 0.04 else 0.0
        result.append({"shot": shot["id"], "start_s": round(position, 3), "duration_s": round(duration, 3)})
        position += duration
    return result


def _trim_of(shot: dict[str, Any]) -> tuple[float, float | None]:
    start, end = (list(shot.get("trim") or [0.0, None]) + [None, None])[:2]
    return float(start or 0.0), None if end is None else float(end)


def _plays_until(shot: dict[str, Any]) -> float:
    """Where a shot stops in its take (or in its plan, before it has one)."""
    end = _trim_of(shot)[1]
    return end if end is not None else out_point(shot, (chosen_take(shot) or {}).get("duration_s") or shot.get("duration_s") or 0.0)


def _end_shot_at(shot: dict[str, Any], end: float) -> None:
    if chosen_take(shot):
        shot["trim"] = [_trim_of(shot)[0], round(end, 3)]
    else:
        shot["duration_s"] = round(end - _trim_of(shot)[0], 3)


NOT_SINGING = ("place", "location", "set", "object", "prop", "vehicle")


def has_lips(doc: dict[str, Any], shot: dict[str, Any]) -> bool:
    """Lips follow the song in this shot: its slice alone does not say so, as a plan may give every shot its part of the song. A shot listing only places and things has none."""
    if not shot.get("audio_slice"):
        return False
    cast = set(shot.get("cast") or [])
    return not cast or any(member["id"] in cast and str(member.get("role") or "").strip().casefold() not in NOT_SINGING for member in doc["cast"])


def keep_on_the_song(doc: dict[str, Any], asked: set[str] = frozenset()) -> list[str]:
    """On a cut that plays the song, the song is the clock and lips follow it: what is inserted or lengthened covers what comes next instead of pushing it off the song, so the cut keeps its
    length and every later shot its place. Time comes first from the shots without lips before a lip-synced one (they keep their start), then from its head: it plays from further into its
    take, which is still the song at that moment. A shot covered entirely is left out. A gap that nothing can fill stays, and song_drift reports it: a lip-synced shot has no picture for
    the song before its slice. Locked shots stay as they are, and so does the length asked of a shot (asked: the shots just added or given a length)."""
    if not on_the_song(doc):
        return []
    start, notes, free = float(doc["edit"].get("song_start_s") or 0.0), [], []  # free: the shots without lips since the last lip-synced one
    for shot in [item for item in doc["shots"] if item.get("included", True)]:
        if not has_lips(doc, shot) or shot.get("locked"):
            free = [] if has_lips(doc, shot) else [*free, *([] if shot["id"] in asked or shot.get("locked") else [shot])]
            continue
        head = _trim_of(shot)[0]
        here = next(item["start_s"] for item in timeline(doc) if item["shot"] == shot["id"])
        late = here + start - shot["audio_slice"][0] - head
        for other in reversed(free if late > 0.03 else []):
            if late > 0.03:  # at least 1.5 s of its action stays
                give = min(late, _plays_until(other) - _trim_of(other)[0] - 1.5)
                if give > 0.03:
                    _end_shot_at(other, _plays_until(other) - give)
                    notes.append(f"{other['id']} now plays {give:.2f} s less.")
                    late -= give
        if late < -0.03 and head:  # the time is back: it starts earlier in its take, as far as it has picture for the song
            back = min(-late, head)
            head -= back
            late += back
            shot["trim"] = [round(head, 3), _plays_until(shot)]
        for other in reversed(free if late < -0.03 else []):
            if late < -0.03:
                take = chosen_take(other)
                give = min(-late, take["duration_s"] - _plays_until(other)) if take and take.get("duration_s") else -late
                if give > 0.03:
                    _end_shot_at(other, _plays_until(other) + give)
                    notes.append(f"{other['id']} now plays {give:.2f} s more.")
                    late += give
        if late > 0.03:
            if _plays_until(shot) - head - late < 0.5:
                shot["included"] = False
                notes.append(f"{shot['id']} is entirely covered and left out of the edit.")
            else:
                shot["trim"] = [round(head + late, 3), round(_plays_until(shot), 3)]
                notes.append(f"{shot['id']} now starts {head + late:.2f} s into its take, on the song.")
        free = []
    return notes


def cut_places(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Where each shot plays in the cut and from where in its take: what a subtitle file timed on this cut needs to follow a later one."""
    shots = {shot["id"]: shot for shot in doc["shots"]}
    return [{**item, "in_s": _trim_of(shots[item["shot"]])[0]} for item in timeline(doc)]


def cut_length(doc: dict[str, Any]) -> float:
    """The cut's length, transitions overlapping."""
    cut = timeline(doc)
    return round(cut[-1]["start_s"] + cut[-1]["duration_s"], 2) if cut else 0.0


def _follow_cut(doc: dict[str, Any], before: list[dict[str, Any]], old_edit: dict[str, Any]) -> None:
    """Texts and sound tracks stay on the picture they were placed on when shots are added, removed, moved, retimed or left out: each one
    the operations did not place themselves keeps its offset into the shot it started on, wherever that shot now plays (from the start
    of the next remaining shot when its own is gone). One starting at 0 stays at the opening."""
    after = timeline(doc)
    if [(item["shot"], item["start_s"], item["duration_s"]) for item in after] == [(item["shot"], item["start_s"], item["duration_s"]) for item in before]:
        return
    starts = {item["shot"]: item["start_s"] for item in after}

    def moved(start: float) -> float:
        index = next((index for index, item in enumerate(before) if item["start_s"] <= start < item["start_s"] + item["duration_s"]), None)
        if start <= 0 or index is None:
            return start
        survivor = next((item for item in before[index:] if item["shot"] in starts), None)
        return start if survivor is None else round(starts[survivor["shot"]] + (start - before[index]["start_s"] if survivor is before[index] else 0.0), 3)

    old = {overlay["id"]: overlay for overlay in old_edit["overlays"]}
    for overlay in doc["edit"]["overlays"]:
        previous = old.get(overlay["id"])
        if previous and (previous["start_s"], previous["end_s"]) == (overlay["start_s"], overlay["end_s"]):
            start = moved(overlay["start_s"])
            overlay["start_s"], overlay["end_s"] = start, round(overlay["end_s"] + start - overlay["start_s"], 3)
    if doc["edit"]["audio_tracks"] == old_edit["audio_tracks"]:
        for track in doc["edit"]["audio_tracks"]:
            track["start_s"] = moved(float(track.get("start_s") or 0.0))


def outline(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """The sequences of the production with their shots: the outline's sequences first, then the other sections in shot order."""
    sequences = {item["name"]: {**item, "shots": [], "planned_s": 0.0, "status": {}} for item in doc.get("outline", [])}
    for shot in doc["shots"]:
        sequence = sequences.setdefault(shot["section"], {"name": shot["section"], "summary": "", "duration_s": None, "shots": [], "planned_s": 0.0, "status": {}})
        sequence["shots"].append(shot["id"])
        sequence["planned_s"] = round(sequence["planned_s"] + (float(shot["duration_s"] or 0.0) if shot.get("included", True) else 0.0), 2)
        sequence["status"][shot["status"]] = sequence["status"].get(shot["status"], 0) + 1
    return list(sequences.values())


def edit_key(doc: dict[str, Any]) -> str:
    """What an export depends on (shots in the cut with their takes, trims and transitions, the sound, text and subtitles): a delivered
    version whose key differs from the document's is out of date."""
    cut = [[shot["id"], take["path"], shot.get("trim"), shot.get("transition"), *([speed_of(shot, take)] if speed_of(shot, take) != 1.0 else []), *([False] if shot.get("sound") is False else [])]
           for shot, take in sequence(doc)]  # listed only when set: older exports stay up to date
    edit = {key: doc["edit"].get(key) for key in ("audio_mode", "song_start_s", "audio_tracks", "subtitles", "fps")}
    edit["overlays"] = [{key: item.get(key) for key in ("path", "start_s", "end_s", "fade_s")} for item in doc["edit"].get("overlays", [])]
    if doc["edit"].get("level_voices") and doc["brief"].get("voices_from_model"):  # listed only when set: older exports stay up to date
        edit["level_voices"] = True
    return hashlib.sha1(json.dumps([cut, edit, doc["brief"].get("song")], sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def summary(doc: dict[str, Any], sequence: str | None = None) -> dict[str, Any]:
    """A compact view for agents: every shot in one line of facts, without settings or take histories (one sequence's shots when named;
    shorter prompts in long productions)."""
    counts: dict[str, int] = {}
    for shot in doc["shots"]:
        counts[shot["status"]] = counts.get(shot["status"], 0) + 1
    shots = [shot for shot in doc["shots"] if sequence is None or shot["section"].casefold() == sequence.casefold()]
    prompt_chars = 240 if sequence or len(shots) <= 24 else 90
    result = {
        "id": doc["id"], "title": doc["title"], "revision": doc["revision"], "stage": doc["stage"], "brief": doc["brief"],
        "cast": [{key: member[key] for key in ("id", "name", "role", "description", "master", "locked")} for member in doc["cast"]],
        "style": doc["style"], "shot_counts": counts, "total_s": cut_length(doc),
        "shots": [{"id": shot["id"], "title": shot["title"], "section": shot["section"], "duration_s": shot["duration_s"], "status": shot["status"], "locked": shot["locked"],
                   "included": shot["included"], "cast": shot["cast"], "prompt": shot_prompt(shot)[:prompt_chars], "takes": len(shot["takes"]), "chosen_take": shot["chosen_take"],
                   **({"dialogue": shot["dialogue"]} if shot.get("dialogue") else {}), **({"speed": shot["speed"]} if shot["speed"] != 1.0 else {}), **({"sound": False} if not shot["sound"] else {}), **({"cut_extra_words": False} if not shot.get("cut_extra_words", True) else {}),
                   "verdict": (chosen_take(shot) or {}).get("verdict"), "take_path": (chosen_take(shot) or {}).get("path"), "notes": shot["notes"],
                   **({"changed_by_user": True} if shot.get("updated_by") == "user" else {})}
                  for shot in shots],
        # the user's requests and notes, with what became of the showrunner's runs (a request whose run failed reads as such)
        "user_changes": [entry for entry in doc["log"][-20:] if entry.get("by") in ("user", "system")],
        "deliverables": doc["deliverables"][-3:],
    }
    if doc.get("outline") or len(doc["shots"]) > 24:
        result["outline"] = [{key: item[key] for key in ("name", "summary", "duration_s", "planned_s", "status")} | {"shots": len(item["shots"])} for item in outline(doc)]
    return result


# Operations ---------------------------------------------------------------------------------------------------------------------------

BRIEF_FIELDS = {"kind", "title", "text", "song", "source", "script", "references", "duration_s", "aspect", "resolution", "draft_resolution", "style_tags", "style", "pacing", "language", "plan_first", "budget_minutes",
                "templates", "read_other_productions", "review_before_edit", "shot_min_s", "shot_max_s", "voices_from_model", "group_renders", "render_min_s", "render_max_s", "song_voices"}
DEFAULT_LENGTH_S = 30.0  # what the Brief shows and plans for until the user sets a length
RENDER_MIN_S = 8.0  # a remake's shorter renders lose the lips: Skate Queen s3 alone (5.8 s) did not follow the singing, with its neighbour (9.5 s) it did


def shot_range(doc: dict[str, Any]) -> tuple[float | None, float | None]:
    """The length the user wants for video shots, in seconds: either end may be open, and both are for the showrunner's own choice."""
    low, high = doc["brief"].get("shot_min_s") or None, doc["brief"].get("shot_max_s") or None
    return (high, low) if low and high and low > high else (low, high)


def shot_range_words(doc: dict[str, Any]) -> str:
    low, high = shot_range(doc)
    return f"{low:g} to {high:g} s" if low and high else f"at least {low:g} s" if low else f"at most {high:g} s" if high else ""
GENERATION_SHORTHAND = {"prompt", "negative_prompt", "audio_guide", "image_refs", "image_start", "video_length", "seed", "template", "settings"}
SHOT_FIELDS = {"title", "section", "intent", "camera", "cast", "dialogue", "duration_s", "audio_slice", "generation", "status", "locked", "included", "chosen_take", "trim", "speed", "sound",
               "transition", "notes", "cut_extra_words", "source_range", "source_poster", "painted_frames", "seen", "looked", "rendering_with"}
SPEEDS = (0.25, 4.0)  # what a browser plays with its sound and the export's filters handle
CAST_FIELDS = {"name", "role", "description", "images", "master", "voice", "voice_description", "voice_text", "lora", "locked", "looks"}
MEDIA_SETTINGS = {"image_refs", "image_start", "image_end", "image_guide", "image_mask", "video_guide", "video_mask", "video_source", "audio_guide", "audio_guide2",
                  "audio_source", "custom_guide"}  # the generation settings that name media files


def look_for(member: dict[str, Any], shot: dict[str, Any]) -> dict[str, Any] | None:
    """The member's look worn in a shot: one naming the shot, else one worn in its sequence (its section name)."""
    looks = [look for look in member.get("looks") or [] if look.get("master")]
    section = str(shot.get("section") or "").casefold()
    return next((look for look in looks if shot.get("id") in (look.get("shots") or [])), None) \
        or next((look for look in looks if any(name.casefold() == section for name in look.get("sections") or [])), None)
USER_ONLY_SHOT_FIELDS = {"locked"}


def _check_shot_write(shot: dict[str, Any], by: str, fields: set[str], base_revision: int | None) -> None:
    if by == "user":
        return
    if fields & USER_ONLY_SHOT_FIELDS:
        raise OperationError(f"Only the user can lock or unlock shot {shot['id']}.")
    if shot["locked"]:
        raise OperationError(f"Shot {shot['id']} is locked by the user: leave it as it is.", current=copy.deepcopy(shot))
    if base_revision is not None and shot.get("updated_by") == "user" and shot.get("updated_rev", 0) > base_revision:
        raise OperationError(f"The user changed shot {shot['id']} after revision {base_revision}: read it again before changing it.", current=copy.deepcopy(shot))


def _generation_fields(fields: dict[str, Any]) -> dict[str, Any]:
    """A shot's settings or template written beside its fields, or a setting written directly in its generation, go where they are read: there is only one thing they can mean."""
    loose = {key: fields[key] for key in fields if key in GENERATION_SHORTHAND}
    given = fields.get("generation")
    nested = {key: given[key] for key in given if key not in {"template", "settings"}} if isinstance(given, dict) else {}
    if not loose and not nested:
        return fields
    rest = {key: value for key, value in fields.items() if key not in loose}
    generation = {key: value for key, value in (given or {}).items() if key not in nested}
    settings = {**nested, **(generation.get("settings") or {})}
    if "template" in loose:
        generation["template"] = loose.pop("template")
    settings.update(loose.pop("settings", None) or {})
    settings.update(loose)
    if settings:
        generation["settings"] = settings
    rest["generation"] = generation
    return rest


def _set_shot_fields(shot: dict[str, Any], fields: dict[str, Any], by: str, revision: int) -> list[str]:
    fields = _generation_fields(fields)
    unknown = set(fields) - SHOT_FIELDS
    if unknown:
        raise OperationError(f"Unknown shot fields: {', '.join(sorted(unknown))}.")
    changed = []
    for key, value in fields.items():
        if key == "generation":
            generation = shot["generation"]
            value = dict(value or {})
            if "settings" in value:  # merged; a null value removes the setting
                merged = {**generation.get("settings", {}), **dict(value.pop("settings") or {})}
                generation["settings"] = {key: item for key, item in merged.items() if item is not None}
            generation.update(value)
        elif key == "status" and value not in SHOT_STATUSES:
            raise OperationError(f"Shot status must be one of {', '.join(SHOT_STATUSES)}.")
        elif key == "chosen_take" and value is not None:
            find(shot["takes"], value, "take")
            shot[key] = value
        elif key == "dialogue":  # the words spoken or sung in the shot: [{speaker, line, note}], speaker a cast id or a name (NARRATOR)
            shot[key] = _dialogue(shot.get("dialogue") or [], value or [])
        elif key == "transition":
            transition = {**shot["transition"], **dict(value or {})}
            if transition["type"] not in TRANSITIONS:
                raise OperationError(f"Transition type must be one of {', '.join(TRANSITIONS)}.")
            shot[key] = transition
        elif key == "speed":
            if not SPEEDS[0] <= float(value) <= SPEEDS[1]:
                raise OperationError(f"A shot's speed goes from {SPEEDS[0]:g} to {SPEEDS[1]:g} times.")
            shot[key] = round(float(value), 3)
        else:
            shot[key] = copy.deepcopy(value)
        changed.append(key)
    if {"chosen_take", "cut_extra_words"} & set(fields):
        follow_line_cut(shot)
    if by == "user" and set(fields) - {"status"}:  # the user's last change of the shot: it protects it from the director's stale writes and replans
        shot["updated_by"], shot["updated_rev"] = by, revision
    return changed


LINE_AUDIO = ("audio", "voice", "spoken", "duration_s")  # a line's spoken audio: its file, the voice sample it was spoken from, the words it says


def line_key(line: dict[str, Any]) -> str:
    return "|".join(str(line.get(key) or "") for key in ("speaker", "line", "note"))


def _dialogue(old: list[dict[str, Any]], lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """New dialogue lines; a line keeps its spoken audio while its speaker, words and note stay the same (audio of other words is dropped)."""
    spoken = {item.get("spoken"): item for item in old if item.get("audio")}
    result = []
    for item in lines:
        line = {"speaker": str(item.get("speaker") or ""), "line": str(item.get("line") or ""), "note": str(item.get("note") or "")}
        if not line["line"].strip():
            continue
        source = item if item.get("audio") and item.get("spoken") == line_key(line) else spoken.get(line_key(line))
        result.append({**line, **({key: source.get(key) for key in LINE_AUDIO} if source else {})})
    return result


def speaker(doc: dict[str, Any], name: str) -> dict[str, Any] | None:
    """The cast member speaking a line: its cast id or name, NARRATOR (or VOICE-OVER) the member whose role or name is narrator."""
    name = str(name or "").strip()
    for member in doc["cast"]:
        if member["id"] == name or member["name"].casefold() == name.casefold():
            return member
    if name.upper().replace("-", " ") in ("NARRATOR", "VOICE OVER", "VOICEOVER", "V.O."):
        return next((member for member in doc["cast"] if "narrator" in f"{member['role']} {member['name']}".casefold()), None)
    return None


def spoken_lines(doc: dict[str, Any], shot: dict[str, Any]) -> list[dict[str, Any]]:
    """The shot's dialogue lines whose recording is in their speaker's current voice."""
    return [line for line in shot.get("dialogue") or [] if line.get("audio") and line.get("voice") == ((speaker(doc, line["speaker"]) or {}).get("voice") or "")]


def cast_ids(doc: dict[str, Any], cast: list[str]) -> list[str]:
    """A shot's cast as cast ids: members named by their name instead (as an agent may write them) become their id."""
    by_name = {member["name"].casefold(): member["id"] for member in doc.get("cast", []) if member.get("name")}
    return [entry if any(member["id"] == entry for member in doc.get("cast", [])) else by_name.get(str(entry).casefold(), entry) for entry in cast or []]


def speaker_ids(doc: dict[str, Any], dialogue: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Dialogue whose speakers named by a cast member's name (ELENA for Elena) become that member's id."""
    return [{**item, "speaker": cast_ids(doc, [item.get("speaker")])[0] if item.get("speaker") else item.get("speaker")} for item in dialogue or [] if isinstance(item, dict)]


def _new_shot(doc: dict[str, Any], fields: dict[str, Any], by: str, revision: int) -> dict[str, Any]:
    # Shot ids are never reused: a removed shot's id may still appear in Deepy's context or in the log.
    counters = doc.setdefault("next_ids", {})
    requested = str(fields.get("id") or "")
    if requested[:1] == "s" and requested[1:].isdigit() and int(requested[1:]) <= counters.get("s", 0) and all(shot["id"] != requested for shot in doc["shots"]):
        shot = normalize_shot({"id": requested})  # the director re-plans a shot it replaced under its id: its notes and log keep meaning
    else:
        number = max([counters.get("s", 0)] + [int(shot["id"][1:]) for shot in doc["shots"] if shot["id"][1:].isdigit()]) + 1
        counters["s"] = number
        shot = normalize_shot({"id": f"s{number}"})
    _set_shot_fields(shot, {key: cast_ids(doc, value) if key == "cast" else speaker_ids(doc, value) if key == "dialogue" else value for key, value in fields.items() if key != "id"},
                     by, revision)
    return shot


def _add_take(shot: dict[str, Any], take: dict[str, Any], by: str, choose: bool = True) -> dict[str, Any]:
    if not take.get("path"):
        raise OperationError("A take needs the path of its media file.")
    record = {"id": _next_id("t", shot["takes"]), "path": take["path"], "media": take.get("media"), "kind": take.get("kind", "video"),
              "duration_s": take.get("duration_s"), "width": take.get("width"), "height": take.get("height"), "fps": take.get("fps"), "poster": take.get("poster"),
              "has_audio": take.get("has_audio"), "strip": take.get("strip"), "codec": take.get("codec"), "settings": copy.deepcopy(take.get("settings") or {}),
              "by": by, "created": now(), "checks": copy.deepcopy(take.get("checks") or {}), "verdict": None, "showrunner_verdict": None, "note": take.get("note", ""),
              "gpu_s": take.get("gpu_s")}  # gpu_s: the measured generation time, for shoot estimates
    if take.get("base_loras"):  # its LoRAs before the cast's (activated, multipliers): a new take adds the cast's as they are then
        record["base_loras"] = copy.deepcopy(take["base_loras"])
    record.update({key: take[key] for key in ("tier", "kept_source") if take.get(key)})  # tier: "draft" until it is shot final; kept_source: the edit reads the shot from the source itself
    shot["takes"].append(record)
    if choose or shot["chosen_take"] is None:  # not chosen: another shot's render gave it this take, beside the one it uses
        shot["chosen_take"] = record["id"]
        if shot["status"] in ("planned", "queued", "rendering", "failed", "rejected"):
            shot["status"] = "rendered"
    follow_line_cut(shot)
    return record


def apply_ops(doc: dict[str, Any], ops: list[dict[str, Any]], *, by: str = "user", base_revision: int | None = None) -> list[dict[str, Any]]:
    """Apply operations in order (all or none) and return what each produced. by is "user", "deepy" or "system"."""
    work = copy.deepcopy(doc)
    revision = work["revision"] + 1
    results = []
    cut = timeline(work)
    for op in ops:
        name = op.get("op")
        handler = _OPS.get(name)
        if handler is None:
            raise OperationError(f"Unknown operation {name!r}. Known: {', '.join(sorted(_OPS))}.")
        results.append(handler(work, op, by, revision, base_revision) or {})
    if not any(op.get("op") in ("restore_version", "bring_back") for op in ops):  # those put back an exact state
        if results and any(shot["takes"] for shot in work["shots"]) and any(op.get("op") in ("add_shots", "remove_shot", "add_take") or (op.get("op") == "update_shot" and {"duration_s", "included", "chosen_take"} & set(op.get("fields") or {})) for op in ops):
            asked = {op["shot"] for op in ops if op.get("op") == "update_shot" and "duration_s" in (op.get("fields") or {})} | {shot for result in results for shot in result.get("shots", [])}
            notes = keep_on_the_song(work, asked)
            if notes:
                results[-1] = {**results[-1], "on_the_song": notes}
                _op_log(work, {"text": "To keep the song in place: " + " ".join(notes)}, "system", revision, base_revision)
        _follow_cut(work, cut, doc["edit"])
    work["revision"], work["updated"] = revision, now()
    work["edit_key"] = edit_key(work)  # the page compares it with the last export's to say whether that export is current
    doc.clear()
    doc.update(work)
    return results


def _op_brief(doc, op, by, revision, base_revision):
    fields = dict(op.get("fields") or {})
    unknown = set(fields) - BRIEF_FIELDS
    if unknown:
        raise OperationError(f"Unknown brief fields: {', '.join(sorted(unknown))}.")
    if "title" in fields:
        doc["title"] = str(fields.pop("title")).strip() or doc["title"]
    singers = fields.get("song_voices", [])
    if not isinstance(singers, list) or len(singers) > 2 or any(not any(member["id"] == singer for member in doc["cast"]) for singer in singers):
        raise OperationError("song_voices: one or two cast ids, first heard first (SeedVC tells two voices apart, not more).")
    doc["brief"].update(copy.deepcopy(fields))
    normalize(doc)


def _op_style(doc, op, by, revision, base_revision):
    doc["style"].update(copy.deepcopy(dict(op.get("fields") or {})))


def _op_stage(doc, op, by, revision, base_revision):
    if op.get("stage") not in STAGES:
        raise OperationError(f"Stage must be one of {', '.join(STAGES)}.")
    doc["stage"] = op["stage"]


_WOMAN_WORDS = {"woman", "women", "female", "she", "her", "lady", "mother", "motherly", "grandmother", "grandmotherly", "soprano", "alto", "feminine"}
_MAN_WORDS = {"man", "men", "male", "he", "his", "him", "gentleman", "father", "fatherly", "grandfather", "grandfatherly", "baritone", "bass", "masculine"}
_CHILD_WORDS = {"boy", "girl", "child", "kid", "teen", "teenage", "teenager"}  # a child's voice is high either way


def voice_kind(text: str) -> str | None:
    """Who a description says this is: "woman", "man", or None for a child or when it leaves it open."""
    words = set(re.findall(r"[a-z]+", str(text or "").casefold()))
    if words & _CHILD_WORDS:
        return None
    woman, man = bool(words & _WOMAN_WORDS), bool(words & _MAN_WORDS)
    return "woman" if woman and not man else "man" if man and not woman else None


def voice_off(kind: str | None, pitch: float | None) -> bool:
    """A voice whose median pitch reads as the other sex. Measured on the productions' voice samples: men 98 to 154 Hz, women 150 to 372 Hz, with an overlap left unjudged."""
    return bool(pitch) and (kind == "woman" and pitch < 165.0 or kind == "man" and pitch > 200.0)


def line_off_voice(sample: float | None, line: float | None) -> bool:
    """A spoken line far from its speaker's voice sample: acting moves pitch by a quarter at most, beyond that it is another voice (one sample at 154 Hz gave lines at 235 to 333 Hz)."""
    return bool(sample and line) and not 0.75 <= line / sample <= 1.35


def singer_kinds(doc: dict[str, Any]) -> set[str]:
    """Who sings on screen: the people of the lip-synced shots, or the whole cast before there are any."""
    singing = {member for shot in doc["shots"] if shot.get("audio_slice") for member in shot.get("cast") or []}
    return {kind for member in doc["cast"] if not is_place(member) and (not singing or member["id"] in singing) and (kind := voice_kind(member.get("voice_description") or member.get("description")))}


def song_singers(doc: dict[str, Any]) -> str:
    """Who the brief says sings the song the showrunner composes, in words ("" for the song model's own voice)."""
    names = [f"{member['name'] or member['id']} ({member['id']})" for singer in doc["brief"]["song_voices"] for member in doc["cast"] if member["id"] == singer]
    return " then ".join(names) + (", a duet in the order they are first heard" if len(names) == 2 else "")


def is_place(member: dict[str, Any]) -> bool:
    """A location kept the same from shot to shot: a cast card whose picture shows the empty set."""
    return str(member.get("role") or "").strip().casefold() in ("place", "location", "set")


def _op_add_cast(doc, op, by, revision, base_revision):
    member = {"id": _next_id("c", doc["cast"])}
    fields = dict(op.get("fields") or {})
    unknown = set(fields) - CAST_FIELDS
    if unknown:
        raise OperationError(f"Unknown cast fields: {', '.join(sorted(unknown))}.")
    member.update(copy.deepcopy(fields))
    doc["cast"].append(member)
    normalize(doc)
    return {"cast": member["id"]}


def _op_update_cast(doc, op, by, revision, base_revision):
    member = find(doc["cast"], op.get("cast"), "cast member")
    fields = dict(op.get("fields") or {})
    if by != "user" and (member["locked"] or "locked" in fields):
        raise OperationError(f"Cast member {member['id']} is locked by the user." if member["locked"] else "Only the user can lock a cast member.", current=copy.deepcopy(member))
    unknown = set(fields) - CAST_FIELDS
    if unknown:
        raise OperationError(f"Unknown cast fields: {', '.join(sorted(unknown))}.")
    before = member["name"]
    member.update(copy.deepcopy(fields))
    if member["name"] != before and before:  # what is written about them in the plan follows; the prompts stay the showrunner's to rewrite
        name = re.compile(rf"\b{re.escape(before)}\b")
        for item in doc.get("outline") or []:
            item["summary"] = name.sub(member["name"], item.get("summary") or "")
        for shot in doc["shots"]:
            shot["title"], shot["intent"] = name.sub(member["name"], shot.get("title") or ""), name.sub(member["name"], shot.get("intent") or "")


def _op_remove_cast(doc, op, by, revision, base_revision):
    member = find(doc["cast"], op.get("cast"), "cast member")
    if by != "user" and member["locked"]:
        raise OperationError(f"Cast member {member['id']} is locked by the user.")
    doc["cast"].remove(member)
    for shot in doc["shots"]:
        shot["cast"] = [item for item in shot["cast"] if item != member["id"]]
    doc["brief"]["song_voices"] = [singer for singer in doc["brief"]["song_voices"] if singer != member["id"]]  # they sing no more


def _op_remake(doc, op, by, revision, base_revision):
    """A remake's source as ingest_source prepared it (path, fps, frames, cuts). Only a remake's document has this section and its shots' source_range, source_poster, painted_frames and seen."""
    doc["remake"] = copy.deepcopy(op.get("source"))


def _op_add_shots(doc, op, by, revision, base_revision):
    position, added = op.get("position"), []
    for offset, fields in enumerate(op.get("shots") or []):
        fields = dict(fields)
        take = fields.pop("take", None)  # a shot made from a file of the user's arrives with it as its take
        shot = _new_shot(doc, fields, by, revision)  # inserted at once: the next shot's id accounts for it
        doc["shots"].insert(len(doc["shots"]) if position is None else int(position) + offset, shot)
        if take:
            _add_take(shot, take, by)
        added.append(shot["id"])
    return {"shots": added}


def media_shot(doc: dict[str, Any], media: dict[str, Any], position: int | None = None) -> tuple[int, dict[str, Any]]:
    """Where a shot made from a video or a picture of the user's goes, and its fields with the file as its take: placed as Add a shot places one
    (at the position given, else at the end, before a closing card), named after the file, in the part and with the model of the shot before it.
    The showrunner then treats it as any shot with a take."""
    shots, drawn = doc["shots"], lambda shot: (shot.get("generation") or {}).get("template") == "draw_svg"
    position = (len(shots) - 1 if shots and drawn(shots[-1]) else len(shots)) if position is None else max(0, min(int(position), len(shots)))
    pattern = next((shot for shot in reversed(shots[:position]) if not drawn(shot)), None) or next((shot for shot in shots if not drawn(shot)), None)
    file = str(media.get("name") or media["path"]).replace("\\", "/").rsplit("/", 1)[-1]
    name = re.sub(r"^\d{4}-\d\d-\d\d-\d\dh\d\dm\d\ds_seed\d+_|\.\w+$", "", file).replace("_", " ").strip()  # a generated file: the words of its prompt
    return position, {"title": name[:60] or "Your shot", "section": shots[position - 1]["section"] if position else "",
                      "duration_s": round(float(media["duration_s"]), 2) if media.get("kind") == "video" and media.get("duration_s") else 4.0,
                      "generation": {"template": (pattern or {}).get("generation", {}).get("template") or "gen_video", "settings": {"prompt": ""}}, "take": {**media, "note": "Your file"}}


def _op_replace_plan(doc, op, by, revision, base_revision):
    """Deepy's plan: shots locked or edited by the user are kept as they are, the others are replaced."""
    doc["shots"] = [shot for shot in doc["shots"] if shot["locked"] or shot["takes"] or shot.get("updated_by") == "user"]
    kept = [shot["id"] for shot in doc["shots"]]
    added = _op_add_shots(doc, {"shots": op.get("shots") or []}, by, revision, base_revision)
    return {"kept": kept, **added}


def _op_update_shot(doc, op, by, revision, base_revision):
    shot = find(doc["shots"], op.get("shot"), "shot")
    fields = dict(op.get("fields") or {})
    _check_shot_write(shot, by, set(fields), op.get("revision", base_revision))
    if "cast" in fields:
        fields["cast"] = cast_ids(doc, fields["cast"])
    if "dialogue" in fields:
        fields["dialogue"] = speaker_ids(doc, fields["dialogue"])
    return {"changed": _set_shot_fields(shot, fields, by, revision)}


def _op_remove_shot(doc, op, by, revision, base_revision):
    shot = find(doc["shots"], op.get("shot"), "shot")
    _check_shot_write(shot, by, set(), op.get("revision", base_revision))
    doc["shots"].remove(shot)
    for number, take in enumerate(shot["takes"], 1):  # its takes wait in the bin, to come back as a shot or a take of another
        if take.get("path") and not any(item["path"] == take["path"] for item in doc["bin"]):
            doc["bin"].append({"id": _next_id("b", doc["bin"]), "created": now(), "by": take.get("by") or by, "name": f"{shot['title'] or shot['id']} T{number}",
                               **{key: take.get(key) for key in ("path", "kind", "duration_s", "width", "height", "fps", "poster", "has_audio", "codec", "strip")}})


def _op_move_shots(doc, op, by, revision, base_revision):
    """Move shots (in their current relative order) before shot `before`, or to the end."""
    ids = list(op.get("shots") or [])
    moving = [find(doc["shots"], shot_id, "shot") for shot_id in ids]
    if by != "user":
        for shot in moving:
            _check_shot_write(shot, by, set(), op.get("revision", base_revision))
    moving.sort(key=doc["shots"].index)
    remaining = [shot for shot in doc["shots"] if shot not in moving]
    before = op.get("before")
    index = len(remaining) if before is None else remaining.index(find(remaining, before, "shot"))
    doc["shots"] = remaining[:index] + moving + remaining[index:]


def _op_reorder(doc, op, by, revision, base_revision):
    order = list(op.get("order") or [])
    if sorted(order) != sorted(shot["id"] for shot in doc["shots"]):
        raise OperationError("reorder needs every shot id exactly once.")
    by_id = {shot["id"]: shot for shot in doc["shots"]}
    doc["shots"] = [by_id[shot_id] for shot_id in order]


def _op_add_take(doc, op, by, revision, base_revision):
    shot = find(doc["shots"], op.get("shot"), "shot")
    if by != "user" and shot["locked"]:
        raise OperationError(f"Shot {shot['id']} is locked by the user: leave it as it is.")
    take = _add_take(shot, dict(op.get("take") or {}), by, choose=op.get("choose", True))
    doc["bin"] = [item for item in doc["bin"] if item.get("by") != "deepy" or item.get("path") != take.get("path")]  # it was in the bin until it became a take
    if by == "user":
        shot["updated_by"], shot["updated_rev"] = by, revision
    return {"shot": shot["id"], "take": take["id"]}


def _op_verdict(doc, op, by, revision, base_revision):
    shot = find(doc["shots"], op.get("shot"), "shot")
    take = find(shot["takes"], op.get("take") or shot["chosen_take"], "take")
    if by != "user" and (take.get("verdict") or {}).get("by") == "user":
        raise OperationError(f"The user already judged take {take['id']} of shot {shot['id']}: {take['verdict']}.", current=copy.deepcopy(take["verdict"]))
    if by == "user" and op.get("ok") is None:  # the user takes their verdict back: the take waits for review again, with the showrunner's verdict
        take["verdict"] = copy.deepcopy(take.get("showrunner_verdict"))
        shot["status"] = "rejected" if take["verdict"] and not take["verdict"]["ok"] else "rendered"
        shot["updated_by"], shot["updated_rev"] = by, revision
        return
    take["verdict"] = {"by": by, "ok": bool(op.get("ok")), "note": str(op.get("note") or ""), "time": now()}
    if by != "user":  # the user's verdict replaces it as the take's verdict; the showrunner's note stays readable beside it
        take["showrunner_verdict"] = copy.deepcopy(take["verdict"])
    if op.get("checks"):
        take["checks"].update(copy.deepcopy(op["checks"]))
    if by == "user":
        shot["status"] = "approved" if op.get("ok") else "rejected"
        shot["chosen_take"] = take["id"] if op.get("ok") else shot["chosen_take"]
        shot["updated_by"], shot["updated_rev"] = by, revision
    elif shot["status"] != "approved":  # only the user approves: a good verdict of the director leaves the take for the user to review
        shot["status"] = "rendered" if op.get("ok") else "rejected"


def _op_heard(doc, op, by, revision, base_revision):
    """What was heard in a take whose model says the shot's lines (see heard_lines)."""
    shot = find(doc["shots"], op.get("shot"), "shot")
    find(shot["takes"], op.get("take"), "take")["heard"] = copy.deepcopy(op.get("heard"))
    follow_line_cut(shot)


def _op_remove_take(doc, op, by, revision, base_revision):
    shot = find(doc["shots"], op.get("shot"), "shot")
    take = find(shot["takes"], op.get("take"), "take")
    if by != "user":
        raise OperationError("Only the user removes takes.")
    shot["takes"].remove(take)
    if shot["chosen_take"] == take["id"]:
        shot["chosen_take"] = shot["takes"][-1]["id"] if shot["takes"] else None
        follow_line_cut(shot)
    if not shot["takes"]:
        shot["status"] = "planned"


def _op_edit(doc, op, by, revision, base_revision):
    fields = dict(op.get("fields") or {})
    if "audio_mode" in fields and fields["audio_mode"] not in AUDIO_MODES:
        raise OperationError(f"audio_mode must be one of {', '.join(AUDIO_MODES)}.")
    doc["edit"].update(copy.deepcopy(fields))
    if "subtitles" in fields:  # a subtitle file is timed on the cut of this moment: assembling a later cut moves its lines with their shots
        places = cut_places(doc)
        for item in doc["edit"]["subtitles"] if places else []:
            item.setdefault("cut", places)


def _op_outline(doc, op, by, revision, base_revision):
    """The production's sequences in order: [{name, summary, duration_s}]; shots belong to the sequence their section names."""
    sequences = []
    for item in op.get("outline") or []:
        name = " ".join(str(item.get("name") or item.get("title") or item.get("section") or "").split())
        if not name:
            raise OperationError('Every sequence of the outline needs a name: [{"name": ..., "summary": ..., "duration_s": ...}].')
        sequences.append({"name": name, "summary": str(item.get("summary") or ""), "duration_s": float(item["duration_s"]) if item.get("duration_s") else None})
    doc["outline"] = sequences


OVERLAY_FIELDS = {"path", "text", "subtitle", "style", "design", "start_s", "end_s", "fade_s"}


def _overlay_fields(fields: dict[str, Any]) -> dict[str, Any]:
    unknown = set(fields) - OVERLAY_FIELDS
    if unknown:
        raise OperationError(f"Unknown overlay fields: {', '.join(sorted(unknown))}.")
    fields = copy.deepcopy(fields)
    for key in ("start_s", "end_s", "fade_s"):
        if key in fields:
            fields[key] = round(max(0.0, float(fields[key])), 3)
    return fields


def _op_add_overlay(doc, op, by, revision, base_revision):
    """Text or a picture laid over the edit from start_s to end_s (timeline seconds), fading in and out."""
    fields = _overlay_fields(dict(op.get("fields") or {}))
    if not fields.get("path"):
        raise OperationError("An overlay needs the path of its picture.")
    overlay = {"id": _next_id("o", doc["edit"]["overlays"]), "by": by, **fields}
    doc["edit"]["overlays"].append(overlay)
    normalize(doc)
    if overlay["end_s"] <= overlay["start_s"]:
        raise OperationError("An overlay must end after it starts.")
    return {"overlay": overlay["id"]}


def _op_update_overlay(doc, op, by, revision, base_revision):
    overlay = find(doc["edit"]["overlays"], op.get("overlay"), "overlay")
    if by != "user" and overlay["by"] == "user":
        raise OperationError(f"Overlay {overlay['id']} is the user's: leave it as it is.", current=copy.deepcopy(overlay))
    overlay.update(_overlay_fields(dict(op.get("fields") or {})))
    if overlay["end_s"] <= overlay["start_s"]:
        raise OperationError("An overlay must end after it starts.")


def _op_remove_overlay(doc, op, by, revision, base_revision):
    overlay = find(doc["edit"]["overlays"], op.get("overlay"), "overlay")
    if by != "user" and overlay["by"] == "user":
        raise OperationError(f"Overlay {overlay['id']} is the user's: leave it as it is.")
    doc["edit"]["overlays"].remove(overlay)


def _op_log(doc, op, by, revision, base_revision):
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(" ".join(line.split()) for line in str(op.get("text") or "").strip().splitlines()))  # its lines kept: Deepy's tables and lists
    if text:
        doc["log"].append({"id": uuid.uuid4().hex[:8], "time": now(), "by": by, "text": text[:4000], **({"shot": op["shot"]} if op.get("shot") else {})})
        del doc["log"][:-500]


AFTER_LINE = 0.35  # how long a shot plays on after its last word
BETWEEN_LINES = 0.3  # the silence between two lines driving one shot


def _lines_length(shot: dict[str, Any]) -> float:
    spoken = [float(line["duration_s"]) for line in shot.get("dialogue") or [] if line.get("duration_s")]
    return sum(spoken) + BETWEEN_LINES * max(0, len(spoken) - 1)


def _ends_after_lines(shot: dict[str, Any]) -> bool:
    start, end = _trim_of(shot)
    return end is not None and bool(_lines_length(shot)) and abs(end - start - _lines_length(shot) - AFTER_LINE) < 0.03


def _op_line_audio(doc, op, by, revision, base_revision):
    """The spoken audio of the shot's lines saying exactly these words (op["spoken"], from line_key), from the voice sample op["voice"]."""
    shot = find(doc["shots"], op.get("shot"), "shot")
    lines = [line for line in shot["dialogue"] if line_key(line) == op.get("spoken")]
    if not lines:
        raise OperationError(f"Shot {shot['id']} no longer has this line: its words changed while it was spoken.")
    ended_after_lines = _ends_after_lines(shot)
    for line in lines:
        line.update(audio=op.get("audio"), voice=op.get("voice"), spoken=op["spoken"], duration_s=op.get("duration_s"))
    if ended_after_lines and shot.get("updated_by") != "user":  # it ended just after its words, as render left it: it still does; a trim chosen otherwise stays
        shot["trim"] = [_trim_of(shot)[0], round(_lines_length(shot) + _trim_of(shot)[0] + AFTER_LINE, 2)]


# Versions -----------------------------------------------------------------------------------------------------------------------------

def _numbers(doc: dict[str, Any]) -> dict[str, int]:
    return {shot["id"]: index + 1 for index, shot in enumerate(item for item in doc["shots"] if item.get("included", True))}


def version_changes(old: dict[str, Any], new: dict[str, Any], limit: int = 40, lengths: tuple[float | None, float | None] = (None, None)) -> list[str]:
    """What changed from one state of a production to a later one, in plain words: shots named by their place in the later cut. lengths:
    the delivered lengths of the two states when they are versions (else the length of their edit)."""
    lines, retimed, numbers = [], [], _numbers(new)
    label = lambda shot: f"Shot {numbers[shot['id']]}" if shot["id"] in numbers else f"«{shot.get('title') or shot['id']}»"
    take = lambda take_id: f"take {str(take_id)[1:]}" if take_id else "no take"
    before_shots, after_shots = {shot["id"]: shot for shot in old["shots"]}, {shot["id"]: shot for shot in new["shots"]}
    for shot in new["shots"]:
        before = before_shots.get(shot["id"])
        if before is None:
            lines.append(f"Added {label(shot).lower()} ({shot.get('title') or 'untitled'})")
            continue
        if before.get("included", True) != shot.get("included", True):
            lines.append(f"{'Included' if shot.get('included', True) else 'Excluded'} {label(shot).lower()} ({shot.get('title') or 'untitled'})")
        if before.get("chosen_take") != shot.get("chosen_take"):
            lines.append(f"{label(shot)}: {take(before.get('chosen_take'))} → {take(shot.get('chosen_take'))}")
        if before.get("trim") != shot.get("trim") or before.get("duration_s") != shot.get("duration_s"):
            retimed.append(label(shot))
        if before.get("speed", 1.0) != shot.get("speed", 1.0):
            lines.append(f"{label(shot)}: speed {before.get('speed', 1.0):g}× → {shot.get('speed', 1.0):g}×")
        if before.get("sound", True) != shot.get("sound", True):
            lines.append(f"{label(shot)}: {'its own sound back' if shot.get('sound', True) else 'sound detached'}")
        if [line.get("line") for line in before.get("dialogue") or []] != [line.get("line") for line in shot.get("dialogue") or []]:
            lines.append(f"{label(shot)}: words changed")
        if (before.get("transition") or {}).get("type") != (shot.get("transition") or {}).get("type"):
            lines.append(f"{label(shot)}: transition {(before.get('transition') or {}).get('type', 'cut')} → {(shot.get('transition') or {}).get('type', 'cut')}")
    if retimed:
        lines.append(f"{retimed[0]} retimed" if len(retimed) == 1 else f"{len(retimed)} shots retimed ({', '.join(item.removeprefix('Shot ') for item in retimed[:8])}{', …' if len(retimed) > 8 else ''})")
    lines += [f"Removed «{shot.get('title') or shot['id']}»" for shot in old["shots"] if shot["id"] not in after_shots]
    kept_before = [shot["id"] for shot in old["shots"] if shot["id"] in after_shots]
    if kept_before != [shot["id"] for shot in new["shots"] if shot["id"] in before_shots]:
        lines.append("Shots reordered")
    before_cast = {member["id"]: member for member in old["cast"]}
    for member in new["cast"]:
        before = before_cast.pop(member["id"], None)
        if before is None:
            lines.append(f"Added {member['name'] or member['id']} to the cast")
            continue
        if before["name"] != member["name"]:
            lines.append(f"{before['name'] or member['id']} recast as {member['name'] or member['id']}")
        if (before.get("master"), before.get("images")) != (member.get("master"), member.get("images")):
            lines.append(f"{member['name'] or member['id']}: new picture")
        if before.get("looks") != member.get("looks"):
            lines.append(f"{member['name'] or member['id']}: looks changed")
        if before.get("voice") != member.get("voice"):
            lines.append(f"{member['name'] or member['id']}: new voice")
        if before.get("lora") != member.get("lora"):
            lines.append(f"{member['name'] or member['id']}: LoRA changed")
    lines += [f"Removed {member['name'] or member['id']} from the cast" for member in before_cast.values()]
    texts = lambda doc: [(item.get("style"), item.get("text")) for item in doc["edit"].get("overlays", [])]
    lines += [f"Text added: «{text}»" for style, text in texts(new) if (style, text) not in texts(old)]
    lines += [f"Text removed: «{text}»" for style, text in texts(old) if (style, text) not in texts(new)]
    timing = lambda doc: {(item.get("style"), item.get("text")): tuple(item.get(key) for key in ("start_s", "end_s", "fade_s")) for item in doc["edit"].get("overlays", [])}
    lines += [f"Text «{key[1]}» moved" for key, when in timing(new).items() if key in timing(old) and timing(old)[key] != when]
    if old["edit"].get("audio_mode") != new["edit"].get("audio_mode"):
        lines.append(f"Soundtrack: {old['edit'].get('audio_mode')} → {new['edit'].get('audio_mode')}")
    if bool(old["edit"].get("level_voices")) != bool(new["edit"].get("level_voices")):
        lines.append("Voices evened out" if new["edit"].get("level_voices") else "Voices as the model made them")
    cuts = lambda doc: [[track.get(key) for key in ("in_s", "out_s", "fade_in_s", "fade_out_s")] for track in doc["edit"].get("audio_tracks", [])]
    if [track.get("path") for track in old["edit"].get("audio_tracks", [])] != [track.get("path") for track in new["edit"].get("audio_tracks", [])]:
        counts = len(old["edit"].get("audio_tracks", [])), len(new["edit"].get("audio_tracks", []))
        lines.append("Sound track replaced" if counts == (1, 1) else "Sound tracks replaced" if counts[0] == counts[1] else f"Sound tracks: {counts[0]} → {counts[1]}")
    elif cuts(old) != cuts(new):
        lines.append("Sound tracks trimmed")
    if old["brief"].get("song") != new["brief"].get("song"):
        lines.append("New song")
    if (old.get("style") or {}).get("text_design") != (new.get("style") or {}).get("text_design"):
        lines.append(f"Text design: {(old.get('style') or {}).get('text_design') or 'classic'} → {(new.get('style') or {}).get('text_design') or 'classic'}")
    before, after = (given if given is not None else cut_length(doc) for given, doc in zip(lengths, (old, new)))
    if abs(before - after) >= 0.05:
        lines.append(f"Length {before:.1f} s → {after:.1f} s")
    return lines[:limit] + ([f"…and {len(lines) - limit} more changes"] if len(lines) > limit else [])


def _op_bring_back(doc, op, by, revision, base_revision):
    """One piece of an earlier version (op["doc"]) back in the production: a shot's take as that version used it, a shot removed since, a cast
    member's pictures, looks and voice, a text, a sound track or a subtitle file."""
    version, kind, item = op["doc"], op.get("kind"), op.get("item")
    if kind == "take":
        old = find(version["shots"], item, "shot in that version")
        used = chosen_take(old)
        if used is None:
            raise OperationError(f"Shot {item} had no take in that version.")
        shot = next((shot for shot in doc["shots"] if shot["id"] == item), None)
        if shot is None:
            raise OperationError(f"Shot {item} is no longer in the production: bring back the shot itself.")
        if by != "user" and shot["locked"]:
            raise OperationError(f"Shot {shot['id']} is locked by the user: leave it as it is.")
        same = next((take for take in shot["takes"] if take["path"] == used["path"]), None)
        if same is None:
            same = _add_take(shot, used, used.get("by", by))
        shot["chosen_take"], shot["trim"], shot["speed"] = same["id"], copy.deepcopy(old.get("trim") or [0.0, None]), old.get("speed", 1.0)
        if shot["status"] in ("planned", "failed", "rejected"):
            shot["status"] = "rendered"
        return {"shot": shot["id"], "take": same["id"]}
    if kind == "shot":
        if any(shot["id"] == item for shot in doc["shots"]):
            raise OperationError(f"Shot {item} is still in the production: bring back its take instead.")
        old = find(version["shots"], item, "shot in that version")
        order, current = [shot["id"] for shot in version["shots"]], [shot["id"] for shot in doc["shots"]]
        before = next((shot_id for shot_id in reversed(order[:order.index(item)]) if shot_id in current), None)  # back after the shot it followed
        after = next((shot_id for shot_id in order[order.index(item) + 1:] if shot_id in current), None)
        position = current.index(before) + 1 if before else current.index(after) if after else len(current)
        doc["shots"].insert(position, normalize_shot(copy.deepcopy(old)))
        return {"shot": item}
    if kind == "cast":
        old = find(version["cast"], item, "cast member in that version")
        fields = {key: copy.deepcopy(old.get(key)) for key in ("name", "role", "description", "images", "master", "voice", "voice_description", "voice_text", "looks", "lora")}
        member = next((member for member in doc["cast"] if member["id"] == item), None)
        if member is None:
            doc["cast"].append({"id": item, **fields})
        elif by != "user" and member["locked"]:
            raise OperationError(f"Cast member {item} is locked by the user.")
        else:
            member.update(fields)
        normalize(doc)
        return {"cast": item}
    if kind == "overlay":
        old = find(version["edit"].get("overlays", []), item, "text in that version")
        overlay = {"id": _next_id("o", doc["edit"]["overlays"]), **{key: copy.deepcopy(old[key]) for key in OVERLAY_FIELDS if key in old}, "by": by}
        doc["edit"]["overlays"].append(overlay)
        return {"overlay": overlay["id"]}
    if kind in ("track", "subtitle"):
        key = "audio_tracks" if kind == "track" else "subtitles"
        entries = version["edit"].get(key, [])
        if not str(item).isdigit() or int(item) >= len(entries):
            raise OperationError(f"That version has no {kind} {item}.")
        entry = copy.deepcopy(entries[int(item)])
        if any(existing.get("path") == entry.get("path") for existing in doc["edit"][key]):
            raise OperationError(f"That {kind} is already in the edit.")
        doc["edit"][key].append(entry)
        return {kind: entry.get("path")}
    raise OperationError("Bring back a take, shot, cast, overlay, track or subtitle.")


def _op_restore_version(doc, op, by, revision, base_revision):
    """The production as it was in an earlier version (op["doc"]): its brief, cast, style, outline, shots and edit. Nothing is lost: takes made
    since stay on their shots, unused; shots and cast members added since stay, the shots excluded from the edit; exports, log and bin stay."""
    version = copy.deepcopy(op["doc"])
    current = {shot["id"]: shot for shot in doc["shots"]}
    shots = []
    for old in version["shots"]:
        now = current.pop(old["id"], None)
        if now is not None:
            takes, used = list(now["takes"]), chosen_take(old)
            for take in old["takes"]:
                if all(item["path"] != take["path"] for item in takes):
                    takes.append({**take, "id": _next_id("t", takes)})
            old["takes"], old["chosen_take"] = takes, next((take["id"] for take in takes if used and take["path"] == used["path"]), None)
        shots.append(old)
    for added in current.values():
        added["included"] = False
        shots.append(added)
    doc["shots"] = shots
    doc["cast"] = version["cast"] + [member for member in doc["cast"] if all(member["id"] != old["id"] for old in version["cast"])]
    for key in ("brief", "style", "outline", "edit"):
        doc[key] = version.get(key, doc.get(key))
    normalize(doc)
    return {"restored": version.get("revision")}


def _op_director(doc, op, by, revision, base_revision):
    doc["director"].update(copy.deepcopy(dict(op.get("fields") or {})))


def _op_deliverable(doc, op, by, revision, base_revision):
    item = {"id": _next_id("d", doc["deliverables"]), "time": now(), **copy.deepcopy(dict(op.get("deliverable") or {}))}
    doc["deliverables"].append(item)
    return {"deliverable": item["id"]}


def _op_budget(doc, op, by, revision, base_revision):
    doc["budget"]["gpu_seconds"] = round(float(doc["budget"].get("gpu_seconds", 0.0)) + float(op.get("gpu_seconds") or 0.0), 2)


def _op_audio(doc, op, by, revision, base_revision):
    doc["audio"]["analysis"] = copy.deepcopy(op.get("analysis"))


def _op_review(doc, op, by, revision, base_revision):
    """What a deep review found on a version: its findings stay with that version."""
    item = find(doc["deliverables"], op.get("deliverable"), "version")
    item["review"] = {**copy.deepcopy(op.get("review") or {}), "time": now()}


def _op_lyrics(doc, op, by, revision, base_revision):
    doc["audio"]["lyrics"] = copy.deepcopy(op.get("lyrics"))


def _op_bin_add(doc, op, by, revision, base_revision):
    """The production's media bin: uploads and Deepy's media that no take records, ready to be dragged onto shots or cast."""
    item = dict(op.get("item") or {})
    if not item.get("path"):
        raise OperationError("A bin item needs a path.")
    if any(existing["path"] == item["path"] for existing in doc["bin"]):
        return {"bin": next(existing["id"] for existing in doc["bin"] if existing["path"] == item["path"])}
    record = {"id": _next_id("b", doc["bin"]), "created": now(), "by": by, **copy.deepcopy(item)}
    doc["bin"].append(record)
    return {"bin": record["id"]}


def _op_bin_remove(doc, op, by, revision, base_revision):
    doc["bin"].remove(find(doc["bin"], op.get("item"), "bin item"))


def _op_bin_to_take(doc, op, by, revision, base_revision):
    item = find(doc["bin"], op.get("item"), "bin item")
    if item.get("kind") == "audio":
        raise OperationError("A sound is not a take: use bin_to_track to lay it over the edit from this shot.")
    result = _op_add_take(doc, {"shot": op.get("shot"), "take": {**item, "note": "From the bin"}}, by, revision, base_revision)
    if op.get("move", True):
        doc["bin"].remove(item)
    return result


def _op_bin_to_track(doc, op, by, revision, base_revision):
    """A sound from the bin over the edit, from the start of a shot (its whole file: the Timeline shortens it), then following that shot's
    picture when the cut changes. A sound is never a shot's take."""
    item = find(doc["bin"], op.get("item"), "bin item")
    if item.get("kind") != "audio":
        raise OperationError("Only a sound goes over the edit: a picture or a video becomes a take of a shot.")
    start = next((part["start_s"] for part in timeline(doc) if part["shot"] == op.get("shot")), None)
    if start is None:
        raise OperationError(f"Shot {op.get('shot')} is not in the edit: include it first.")
    doc["edit"]["audio_tracks"].append({"path": item["path"], "start_s": round(start, 3), "gain_db": 0.0})
    if doc["edit"]["audio_mode"] == "song":  # the song keeps playing beside it
        doc["edit"]["audio_mode"] = "mix"
    doc["bin"].remove(item)
    return {"track": len(doc["edit"]["audio_tracks"]) - 1, "start_s": round(start, 3)}


def _op_bin_to_shot(doc, op, by, revision, base_revision):
    """A video or a picture from the bin as a new shot of its own, its take (position: where it goes, else at the end)."""
    item = find(doc["bin"], op.get("item"), "bin item")
    if item.get("kind") == "audio":
        raise OperationError("A sound is no shot: it plays over the edit from a shot (bin_to_track).")
    position, fields = media_shot(doc, item, op.get("position"))
    doc["bin"].remove(item)
    return _op_add_shots(doc, {"position": position, "shots": [fields]}, by, revision, base_revision)


_OPS = {"heard": _op_heard, "remake": _op_remake, "brief": _op_brief, "style": _op_style, "stage": _op_stage, "add_cast": _op_add_cast, "update_cast": _op_update_cast, "remove_cast": _op_remove_cast,
        "add_shots": _op_add_shots, "replace_plan": _op_replace_plan, "update_shot": _op_update_shot, "remove_shot": _op_remove_shot, "move_shots": _op_move_shots,
        "reorder": _op_reorder, "add_take": _op_add_take, "verdict": _op_verdict, "remove_take": _op_remove_take, "edit": _op_edit, "log": _op_log,
        "director": _op_director, "line_audio": _op_line_audio, "bring_back": _op_bring_back, "restore_version": _op_restore_version, "outline": _op_outline, "add_overlay": _op_add_overlay, "update_overlay": _op_update_overlay,
        "remove_overlay": _op_remove_overlay, "deliverable": _op_deliverable, "budget": _op_budget, "audio": _op_audio, "lyrics": _op_lyrics, "review": _op_review, "bin_add": _op_bin_add, "bin_remove": _op_bin_remove,
        "bin_to_take": _op_bin_to_take, "bin_to_track": _op_bin_to_track, "bin_to_shot": _op_bin_to_shot}
