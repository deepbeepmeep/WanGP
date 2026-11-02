# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""The production document: brief, cast, style, shots with their takes, edit, deliverables and the director's log.

It is the single source of truth of a production. The Showrunner UI renders it; the user's edits and Deepy's toolbox actions
change it through the same operations (apply_ops), with lock and revision rules so that neither overwrites the other's work."""
from __future__ import annotations

import copy
import hashlib
import json
import time
import uuid
from typing import Any

SCHEMA = 1
KINDS = ("music_video", "short_film", "ad", "trailer", "story", "free")
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
    return RESOLUTIONS[brief["aspect"]][brief["resolution"]]


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
    for key, value in {"kind": "free", "text": "", "song": None, "script": "", "references": [], "duration_s": None, "aspect": "16:9", "resolution": "720p",
                       "style_tags": [], "style": "", "pacing": 0, "language": "en", "plan_first": True, "budget_minutes": None, "templates": {},
                       "read_other_productions": True, "review_before_edit": False}.items():
        brief.setdefault(key, copy.deepcopy(value))
    if brief["kind"] not in KINDS:
        brief["kind"] = "free"
    for member in doc.setdefault("cast", []):
        for key, value in {"name": "", "role": "", "description": "", "images": [], "master": None, "voice": None, "voice_description": "", "voice_text": "", "lora": None,
                           "locked": False, "looks": []}.items():
            member.setdefault(key, copy.deepcopy(value))
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
    for key, value in {"audio_mode": "song" if brief["kind"] == "music_video" else "clips", "audio_tracks": [], "subtitles": [], "overlays": [], "fps": None}.items():
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
                       "status": "planned", "locked": False, "included": True, "takes": [], "chosen_take": None, "trim": [0.0, None],
                       "transition": {"type": "cut", "duration_s": 0.0}, "notes": "", "updated_by": "", "updated_rev": 0}.items():
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


def out_point(shot: dict[str, Any], length: float) -> float:
    """Where a take ends in the edit without a trim of the user's: at the shot's planned length when the take came out longer. Models round
    lengths up to their frame steps (10.22 s for a 10.125 s plan), and the cuts the director planned on the bars, and the song slices of
    lip-synced shots, must not drift by that much per shot."""
    planned = shot.get("duration_s")
    return min(float(length), float(planned)) if planned else float(length)


def timeline(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Start time and duration of every included shot (planned durations where no take exists yet), in whole frames of the edit as the
    assembly cuts them, so that sound and text placed at a shot's start stay on its picture."""
    included = [shot for shot in doc["shots"] if shot.get("included", True)]
    fps = float(doc.get("edit", {}).get("fps") or next(((take or {}).get("fps") for shot in included if (take := chosen_take(shot)) and take.get("fps")), None) or 24.0)
    position, result = 0.0, []
    for shot in included:
        take = chosen_take(shot)
        start_trim, end_trim = (shot.get("trim") or [0.0, None]) + [None] * (2 - len(shot.get("trim") or []))
        length = (take or {}).get("duration_s") or shot.get("duration_s") or 0.0
        end = end_trim if end_trim is not None else out_point(shot, length)
        duration = round(max(0.0, float(end) - float(start_trim or 0.0)) * fps) / fps
        result.append({"shot": shot["id"], "start_s": round(position, 3), "duration_s": round(duration, 3)})
        position += duration
    return result


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
    cut = [[shot["id"], take["path"], shot.get("trim"), shot.get("transition")] for shot, take in sequence(doc)]
    edit = {key: doc["edit"].get(key) for key in ("audio_mode", "song_start_s", "audio_tracks", "subtitles", "fps")}
    edit["overlays"] = [{key: item.get(key) for key in ("path", "start_s", "end_s", "fade_s")} for item in doc["edit"].get("overlays", [])]
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
        "style": doc["style"], "shot_counts": counts, "total_s": round(sum(item["duration_s"] for item in timeline(doc)), 2),
        "shots": [{"id": shot["id"], "title": shot["title"], "section": shot["section"], "duration_s": shot["duration_s"], "status": shot["status"], "locked": shot["locked"],
                   "included": shot["included"], "cast": shot["cast"], "prompt": shot_prompt(shot)[:prompt_chars], "takes": len(shot["takes"]), "chosen_take": shot["chosen_take"],
                   **({"dialogue": shot["dialogue"]} if shot.get("dialogue") else {}),
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

BRIEF_FIELDS = {"kind", "title", "text", "song", "script", "references", "duration_s", "aspect", "resolution", "style_tags", "style", "pacing", "language", "plan_first", "budget_minutes",
                "templates", "read_other_productions", "review_before_edit"}
SHOT_FIELDS = {"title", "section", "intent", "camera", "cast", "dialogue", "duration_s", "audio_slice", "generation", "status", "locked", "included", "chosen_take", "trim",
               "transition", "notes"}
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


def _set_shot_fields(shot: dict[str, Any], fields: dict[str, Any], by: str, revision: int) -> list[str]:
    unknown = set(fields) - SHOT_FIELDS
    if unknown:
        settings = sorted(unknown & {"prompt", "negative_prompt", "audio_guide", "image_refs", "image_start", "video_length", "seed", "template"})
        raise OperationError(f"Unknown shot fields: {', '.join(sorted(unknown))}." + (f" {', '.join(settings)} belong in the shot's generation: "
                             '{"generation": {"settings": {...}}} (the template as generation.template).' if settings else ""))
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
        else:
            shot[key] = copy.deepcopy(value)
        changed.append(key)
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


def _add_take(shot: dict[str, Any], take: dict[str, Any], by: str) -> dict[str, Any]:
    if not take.get("path"):
        raise OperationError("A take needs the path of its media file.")
    record = {"id": _next_id("t", shot["takes"]), "path": take["path"], "media": take.get("media"), "kind": take.get("kind", "video"),
              "duration_s": take.get("duration_s"), "width": take.get("width"), "height": take.get("height"), "fps": take.get("fps"), "poster": take.get("poster"),
              "has_audio": take.get("has_audio"), "strip": take.get("strip"), "codec": take.get("codec"), "settings": copy.deepcopy(take.get("settings") or {}),
              "by": by, "created": now(), "checks": copy.deepcopy(take.get("checks") or {}), "verdict": None, "showrunner_verdict": None, "note": take.get("note", ""),
              "gpu_s": take.get("gpu_s")}  # gpu_s: the measured generation time, for shoot estimates
    shot["takes"].append(record)
    shot["chosen_take"] = record["id"]
    if shot["status"] in ("planned", "queued", "rendering", "failed", "rejected"):
        shot["status"] = "rendered"
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
    doc["brief"].update(copy.deepcopy(fields))
    normalize(doc)


def _op_style(doc, op, by, revision, base_revision):
    doc["style"].update(copy.deepcopy(dict(op.get("fields") or {})))


def _op_stage(doc, op, by, revision, base_revision):
    if op.get("stage") not in STAGES:
        raise OperationError(f"Stage must be one of {', '.join(STAGES)}.")
    doc["stage"] = op["stage"]


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
    member.update(copy.deepcopy(fields))


def _op_remove_cast(doc, op, by, revision, base_revision):
    member = find(doc["cast"], op.get("cast"), "cast member")
    if by != "user" and member["locked"]:
        raise OperationError(f"Cast member {member['id']} is locked by the user.")
    doc["cast"].remove(member)
    for shot in doc["shots"]:
        shot["cast"] = [item for item in shot["cast"] if item != member["id"]]


def _op_add_shots(doc, op, by, revision, base_revision):
    position, added = op.get("position"), []
    for offset, fields in enumerate(op.get("shots") or []):
        shot = _new_shot(doc, dict(fields), by, revision)  # inserted at once: the next shot's id accounts for it
        doc["shots"].insert(len(doc["shots"]) if position is None else int(position) + offset, shot)
        added.append(shot["id"])
    return {"shots": added}


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
    take = _add_take(shot, dict(op.get("take") or {}), by)
    if by == "user":
        shot["updated_by"], shot["updated_rev"] = by, revision
    return {"shot": shot["id"], "take": take["id"]}


def _op_verdict(doc, op, by, revision, base_revision):
    shot = find(doc["shots"], op.get("shot"), "shot")
    take = find(shot["takes"], op.get("take") or shot["chosen_take"], "take")
    if by != "user" and (take.get("verdict") or {}).get("by") == "user":
        raise OperationError(f"The user already judged take {take['id']} of shot {shot['id']}: {take['verdict']}.", current=copy.deepcopy(take["verdict"]))
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


def _op_remove_take(doc, op, by, revision, base_revision):
    shot = find(doc["shots"], op.get("shot"), "shot")
    take = find(shot["takes"], op.get("take"), "take")
    if by != "user":
        raise OperationError("Only the user removes takes.")
    shot["takes"].remove(take)
    if shot["chosen_take"] == take["id"]:
        shot["chosen_take"] = shot["takes"][-1]["id"] if shot["takes"] else None
    if not shot["takes"]:
        shot["status"] = "planned"


def _op_edit(doc, op, by, revision, base_revision):
    fields = dict(op.get("fields") or {})
    if "audio_mode" in fields and fields["audio_mode"] not in AUDIO_MODES:
        raise OperationError(f"audio_mode must be one of {', '.join(AUDIO_MODES)}.")
    doc["edit"].update(copy.deepcopy(fields))


def _op_outline(doc, op, by, revision, base_revision):
    """The production's sequences in order: [{name, summary, duration_s}]; shots belong to the sequence their section names."""
    sequences = []
    for item in op.get("outline") or []:
        name = " ".join(str(item.get("name") or "").split())
        if not name:
            raise OperationError("Every sequence of the outline needs a name.")
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
    text = " ".join(str(op.get("text") or "").split())
    if text:
        doc["log"].append({"id": uuid.uuid4().hex[:8], "time": now(), "by": by, "text": text[:2000], **({"shot": op["shot"]} if op.get("shot") else {})})
        del doc["log"][:-500]


def _op_line_audio(doc, op, by, revision, base_revision):
    """The spoken audio of the shot's lines saying exactly these words (op["spoken"], from line_key), from the voice sample op["voice"]."""
    shot = find(doc["shots"], op.get("shot"), "shot")
    lines = [line for line in shot["dialogue"] if line_key(line) == op.get("spoken")]
    if not lines:
        raise OperationError(f"Shot {shot['id']} no longer has this line: its words changed while it was spoken.")
    for line in lines:
        line.update(audio=op.get("audio"), voice=op.get("voice"), spoken=op["spoken"], duration_s=op.get("duration_s"))


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
    lines += [f"Removed {member['name'] or member['id']} from the cast" for member in before_cast.values()]
    texts = lambda doc: [(item.get("style"), item.get("text")) for item in doc["edit"].get("overlays", [])]
    lines += [f"Text added: «{text}»" for style, text in texts(new) if (style, text) not in texts(old)]
    lines += [f"Text removed: «{text}»" for style, text in texts(old) if (style, text) not in texts(new)]
    timing = lambda doc: {(item.get("style"), item.get("text")): tuple(item.get(key) for key in ("start_s", "end_s", "fade_s")) for item in doc["edit"].get("overlays", [])}
    lines += [f"Text «{key[1]}» moved" for key, when in timing(new).items() if key in timing(old) and timing(old)[key] != when]
    if old["edit"].get("audio_mode") != new["edit"].get("audio_mode"):
        lines.append(f"Soundtrack: {old['edit'].get('audio_mode')} → {new['edit'].get('audio_mode')}")
    if [track.get("path") for track in old["edit"].get("audio_tracks", [])] != [track.get("path") for track in new["edit"].get("audio_tracks", [])]:
        counts = len(old["edit"].get("audio_tracks", [])), len(new["edit"].get("audio_tracks", []))
        lines.append("Sound track replaced" if counts == (1, 1) else "Sound tracks replaced" if counts[0] == counts[1] else f"Sound tracks: {counts[0]} → {counts[1]}")
    if old["brief"].get("song") != new["brief"].get("song"):
        lines.append("New song")
    if (old.get("style") or {}).get("text_design") != (new.get("style") or {}).get("text_design"):
        lines.append(f"Text design: {(old.get('style') or {}).get('text_design') or 'classic'} → {(new.get('style') or {}).get('text_design') or 'classic'}")
    before, after = (given if given is not None else sum(item["duration_s"] for item in timeline(doc)) for given, doc in zip(lengths, (old, new)))
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
        shot["chosen_take"], shot["trim"] = same["id"], copy.deepcopy(old.get("trim") or [0.0, None])
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
        fields = {key: copy.deepcopy(old.get(key)) for key in ("name", "role", "description", "images", "master", "voice", "voice_description", "voice_text", "looks")}
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
    result = _op_add_take(doc, {"shot": op.get("shot"), "take": {**item, "note": "From the bin"}}, by, revision, base_revision)
    if op.get("move", True):
        doc["bin"].remove(item)
    return result


_OPS = {"brief": _op_brief, "style": _op_style, "stage": _op_stage, "add_cast": _op_add_cast, "update_cast": _op_update_cast, "remove_cast": _op_remove_cast,
        "add_shots": _op_add_shots, "replace_plan": _op_replace_plan, "update_shot": _op_update_shot, "remove_shot": _op_remove_shot, "move_shots": _op_move_shots,
        "reorder": _op_reorder, "add_take": _op_add_take, "verdict": _op_verdict, "remove_take": _op_remove_take, "edit": _op_edit, "log": _op_log,
        "director": _op_director, "line_audio": _op_line_audio, "bring_back": _op_bring_back, "restore_version": _op_restore_version, "outline": _op_outline, "add_overlay": _op_add_overlay, "update_overlay": _op_update_overlay,
        "remove_overlay": _op_remove_overlay, "deliverable": _op_deliverable, "budget": _op_budget, "audio": _op_audio, "bin_add": _op_bin_add, "bin_remove": _op_bin_remove,
        "bin_to_take": _op_bin_to_take}
