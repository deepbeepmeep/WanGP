# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Which Deepy template makes each kind of shot or media of a production: the director's advice, the user's picks and the warnings.

A start image becomes the first frame of the shot (its framing, aspect and setting leak into it), so recurring people go through
reference images. The catalog is each Deepy category's template list as Deepy's settings show it; nothing here names a model."""
from __future__ import annotations

from typing import Any

VIDEO_TASKS = ("gen_video", "gen_video_with_speech", "gen_video_with_refs")
DRAWN = "draw_svg"  # the template of a shot the director draws (a card, a diagram, a map) and records with record_take
RULE = ("Shots that show a cast member take that member's pictures as reference images (a template whose inputs include references), "
        "not as image_start: a start image becomes the shot's first frame, so its framing, aspect and setting leak into the shot. "
        "Lip-synced shots also need a template that takes the song slice as driving audio.")
# The kinds of shots and media the user picks a template for, one per Deepy category in the order of Deepy's settings: the advice key,
# the category, and what the showrunner makes with its template (shown under the category's name).
ROLES = (
    {"id": "images", "task": "gen_image", "hint": "Stills, illustrated pages and backgrounds"},
    {"id": "image_edits", "task": "edit_image", "hint": "The cast's reference pictures and looks, made from their photos"},
    {"id": "shots_without_cast", "task": "gen_video", "hint": "Shots without the cast: landscapes, objects, crowds, inserts"},
    {"id": "cast_shots_lip_synced", "task": "gen_video_with_speech", "hint": "Shots where someone sings or speaks, lip-synced to the song or to their line"},
    {"id": "cast_shots", "task": "gen_video_with_refs", "hint": "Shots with the cast, keeping each person's face, hair and outfit from their pictures"},
    {"id": "music", "task": "gen_song", "hint": "The songs and scores the showrunner composes"},
    {"id": "voices", "task": "gen_speech_from_description", "hint": "Each speaker's voice, made once from the voice described on their cast card"},
    {"id": "cloned_voices", "task": "gen_speech_from_sample", "hint": "Every spoken line (dialogue, narration), in its speaker's voice"},
)
ROLE_IDS = tuple(role["id"] for role in ROLES)
TASKS = {role["id"]: role["task"] for role in ROLES}


def entry_name(entry: dict[str, Any]) -> str:
    return f"{entry['task']}/{entry['name']}"


def resolve(catalog: list[dict[str, Any]], template: str | None) -> dict[str, Any] | None:
    """The catalog entry of "task" (its default) or "task/name"."""
    task, _, name = str(template or "").partition("/")
    name = name.strip()
    for entry in catalog:
        if entry["task"] == task and (entry["name"].casefold() == name.casefold() if name else entry["default"]):
            return entry
    return None


def role_of(entry: dict[str, Any]) -> str | None:
    """The kind of shot or media a template makes: its Deepy category's."""
    return next((role["id"] for role in ROLES if role["task"] == entry["task"]), None)


def _fits(role: str, entry: dict[str, Any]) -> bool:
    return entry["task"] == TASKS[role]


def automatic(catalog: list[dict[str, Any]]) -> dict[str, str]:
    """The template of each role when the user picks none: the one selected in Deepy's settings for its category."""
    return {role["id"]: entry_name(entry) for role in ROLES if (entry := resolve(catalog, role["task"])) is not None}


def chosen(doc: dict[str, Any], catalog: list[dict[str, Any]]) -> dict[str, str]:
    """The user's picks that are still templates of their category."""
    picks = doc["brief"].get("templates") or {}
    return {role: template for role, template in picks.items() if role in ROLE_IDS and (entry := resolve(catalog, template)) is not None and _fits(role, entry)}


def named(doc: dict[str, Any]) -> list[str]:
    """The templates a production names (its picks and its shots'): the catalog keeps listing a custom one that Deepy's settings dropped."""
    return [*(doc["brief"].get("templates") or {}).values(), *(shot["generation"].get("template") for shot in doc["shots"])]


def effective(doc: dict[str, Any], catalog: list[dict[str, Any]]) -> dict[str, str]:
    return {**automatic(catalog), **chosen(doc, catalog)}


def choices(doc: dict[str, Any], catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The picker: every role under its Deepy category's name, with that category's templates, the automatic one and the user's pick."""
    auto, picks = automatic(catalog), chosen(doc, catalog)
    return [{**role, "label": next(entry["category"] for entry in catalog if _fits(role["id"], entry)), "auto": auto.get(role["id"]), "value": picks.get(role["id"]),
             "choices": [{"value": entry_name(entry), "label": entry.get("label") or entry["name"], "refs": entry.get("refs", False), "audio": entry.get("audio", False)}
                         for entry in catalog if _fits(role["id"], entry)]}
            for role in ROLES if any(_fits(role["id"], entry) for entry in catalog)]


def pick(picks: dict[str, str], catalog: list[dict[str, Any]], role: str, template: str | None) -> dict[str, str]:
    """`picks` with `template` as the role's pick (None: automatic again)."""
    if role not in ROLE_IDS:
        raise ValueError(f"Unknown template role {role!r}. Known: {', '.join(ROLE_IDS)}.")
    entry = resolve(catalog, template) if template else None
    if template and (entry is None or not _fits(role, entry)):
        raise ValueError(f"{template} is not a template for {role}.")
    return {key: value for key, value in picks.items() if key != role} | ({role: entry_name(entry)} if entry else {})


def choose(doc: dict[str, Any], catalog: list[dict[str, Any]], role: str, template: str | None) -> list[dict[str, Any]]:
    """The operations that make `template` the role's pick (None: automatic again). Shots still on the role's previous template follow it;
    shots given another template one by one keep theirs."""
    before = effective(doc, catalog).get(role)
    picks = pick(doc["brief"].get("templates") or {}, catalog, role, template)
    ops = [{"op": "brief", "fields": {"templates": picks}}]
    after = effective({**doc, "brief": {**doc["brief"], "templates": picks}}, catalog).get(role)
    if role in ("cast_shots", "cast_shots_lip_synced", "shots_without_cast") and before and after and after != before:
        previous = resolve(catalog, before)
        ops += [{"op": "update_shot", "shot": shot["id"], "fields": {"generation": {"template": after}}} for shot in doc["shots"]
                if not shot.get("locked") and resolve(catalog, shot["generation"].get("template")) is previous]
    return ops


def speech_settings(entry: dict[str, Any], text: str, *, note: str = "", description: str = "", sample: str | None = None, sample_text: str = "") -> dict[str, Any]:
    """Settings of one spoken line for a speech template: the words, with the delivery note as a leading cue for the models that perform
    cues; the voice from a sample (with its transcript for the models that read one) or from a description."""
    family = str(entry.get("model_type") or "")
    cues = family.startswith(("fish_", "index_tts"))
    settings = {"template": entry_name(entry), "prompt": f"[{note}] {text}" if note and cues else text}
    if sample:
        settings["audio_guide"] = sample
        if sample_text and family.startswith("fish_"):
            settings["alt_prompt"] = sample_text
    elif description:
        if family.startswith("fish_"):  # a direction the model performs: the voice it invents follows it
            settings["prompt"] = f"[{description}] {text}"
        else:
            settings["alt_prompt"] = description
    return settings


def cast_with_pictures(doc: dict[str, Any]) -> list[str]:
    return [member["id"] for member in doc["cast"] if member.get("images") or member.get("master")]


def lip_synced_with_cast(picks: dict[str, str], catalog: list[dict[str, Any]]) -> str | None:
    """The template of lip-synced shots that show the cast: Video With Speech's when it takes the cast's pictures, else Video Generator with
    Ref.'s when its model can follow a soundtrack (render then gives it the song slice or the spoken line as one)."""
    entries = [resolve(catalog, picks.get(role)) for role in ("cast_shots_lip_synced", "cast_shots")]
    return next((entry_name(entry) for entry in entries if entry and entry["refs"] and (entry["audio"] or entry["soundtrack"])), None)


def advice(doc: dict[str, Any], catalog: list[dict[str, Any]]) -> dict[str, Any]:
    """What the director should choose for this production's shots and media."""
    picks, user = effective(doc, catalog), chosen(doc, catalog)
    lip, with_cast = picks.get("cast_shots_lip_synced"), lip_synced_with_cast(picks, catalog)
    result = {"rule": RULE, "cast_with_pictures": cast_with_pictures(doc)}
    result.update({role: picks[role] for role in ("cast_shots", "shots_without_cast") if role in picks})
    if with_cast or lip:
        result["cast_shots_lip_synced"] = with_cast or lip
    if lip and with_cast and lip != with_cast:  # Video With Speech's template does not take the cast's pictures
        result["lip_synced_without_cast"] = lip
        result["lip_sync"] = ("cast_shots_lip_synced is a reference template: give a singing shot its song slice as audio_guide (render gives a "
                              "speaking shot its line), and render makes the template follow it as a kept soundtrack.")
    if "image_edits" in picks:  # a master in the production's look made from the member's own pictures (or from a master, for a look)
        result["masters_from_pictures"] = picks["image_edits"]
    if doc["brief"].get("kind") == "story":  # picture-book pages are stills: the cast's go through the image editing template
        result.update({key: picks[role] for key, role in (("pages", "images"), ("pages_with_cast", "image_edits")) if role in picks})
    if user:
        result["user_choices"] = user
        result["user_rule"] = "user_choices are the user's picks for these kinds of shots and media: use them, never another template for the same kind."
    if doc["brief"].get("kind") == "music_video" and doc["brief"].get("song") and "cast_shots_lip_synced" in result:
        result["song"] = "Give each lip-synced shot its exact slice of the song as driving audio."
    return result


def lint(doc: dict[str, Any], catalog: list[dict[str, Any]], shot_ids: list[str] | None = None) -> list[str]:
    """Warnings for shots showing a cast member with pictures whose template takes no reference images, and for shots on another
    template than the one the user picked for their kind."""
    pictured = set(cast_with_pictures(doc))
    picks, user = effective(doc, catalog), chosen(doc, catalog)
    warnings = []
    for shot in doc["shots"]:
        if shot_ids is not None and shot["id"] not in shot_ids or shot.get("locked"):
            continue
        entry = resolve(catalog, shot["generation"].get("template"))
        if entry is None:
            continue
        shown = pictured & set(shot.get("cast") or [])
        lines, settings = shot.get("dialogue") or [], shot["generation"].get("settings") or {}
        # driven by audio: a lip-sync template, or a reference template that follows a soundtrack when a speaker the shot shows talks
        driven = entry["audio"] or entry["soundtrack"] and any(line.get("speaker") in (shot.get("cast") or []) for line in lines)
        if shown and not entry["refs"]:  # a still keeps the cast through the image editing template, a moving shot through a video one
            suggestion = lip_synced_with_cast(picks, catalog) if entry["audio"] else picks.get("image_edits") if entry["task"] == "gen_image" else picks.get("cast_shots")
            warnings.append(f"{shot['id']} shows {', '.join(sorted(shown))} but its template {entry_name(entry)} takes no reference images"
                            + (f": use {suggestion} with their pictures as references." if suggestion else "."))
        elif driven and not settings.get("audio_guide") and not shot.get("audio_slice") and not lines:  # render speaks and joins a shot's lines itself
            warnings.append(f"{shot['id']} is driven by audio but has none: " + ("give its audio_slice and the song as audio_guide." if doc["brief"].get("song")
                            else "write its dialogue (render speaks it) or give an audio_guide."))
        elif (role := role_of(entry)) in user and user[role] != entry_name(entry) and shot.get("updated_by") != "user" \
                and (shown or not resolve(catalog, user[role])["refs"]):  # a pick that needs references does not fit a shot without the cast
            warnings.append(f"{shot['id']} uses {entry_name(entry)} but the user picked {user[role]} for this kind of shot: use it.")
        if lines and "<d>" in str(settings.get("prompt") or ""):  # the model would speak in a voice of its own, not the speaker's sample
            warnings.append(f"{shot['id']} writes its spoken words in the prompt (<d>): the model voices them itself, in a voice of its own. "
                            "Speak the lines with `speak` and leave their words out of the prompt: render drives the lips with the recordings.")
    return warnings
