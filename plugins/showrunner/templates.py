# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Which Deepy template makes each kind of shot or media of a production: the director's advice, the user's picks and the warnings.

A start image becomes the first frame of the shot (its framing, aspect and setting leak into it), so recurring people go through
reference images. The catalog is each Deepy category's template list as Deepy's settings show it; nothing here names a model."""
from __future__ import annotations

import math
import re
from typing import Any

from . import production as model

VIDEO_TASKS = ("gen_video", "gen_video_with_speech", "gen_video_with_refs")
EDIT_TASKS = ("swap_video", "edit_video")  # they change a remake's source video: never offered for a shot made from nothing
DRAWN ="draw_svg"  # the template of a shot the director draws (a card, a diagram, a map) and records with record_take
SHOT_RANGE_MARGIN_S = 0.5  # a model's lengths come in steps: the nearest one to the user's range is not outside it
LIP_SYNC_WORDS = re.compile(r"\s+with natural lip[- ]movements?(?:\s+synchronized)?", re.I)  # for a face that follows a recording
WORD_S, SPARE_S = 0.4, 2.0  # a word's time when a model says it, and the time left in a shot past which it adds words of its own (measured on H3)
LANGUAGE_NAMES = {"ar": "Arabic", "de": "German", "en": "English", "es": "Spanish", "fr": "French", "hi": "Hindi", "it": "Italian", "ja": "Japanese", "ko": "Korean",
                  "nl": "Dutch", "pl": "Polish", "pt": "Portuguese", "ru": "Russian", "tr": "Turkish", "zh": "Chinese"}  # the brief's language, as a prompt names it
RULE = ("Shots that show a cast member take that member's pictures as reference images (a template whose inputs include references), "
        "not as image_start: a start image becomes the shot's first frame, so its framing, aspect and setting leak into the shot. "
        "Lip-synced shots also need a template that takes the song slice as driving audio.")
PROMPT_RULES_NOTE = ("user_prompt_rules are the user's own rules for the prompts written with these templates: follow them in every prompt you write for one "
                     "(shots, pictures, songs, voices). What the user asks in the chat comes first.")
# The kinds of shots and media the user picks a template for, one per Deepy category in the order of Deepy's settings: the advice key,
# the category, and what the showrunner makes with its template (shown under the category's name).
ROLES = (
    {"id": "images", "task": "gen_image", "hint": "Stills, illustrated pages and backgrounds"},
    {"id": "image_edits", "task": "edit_image", "hint": "The cast's reference pictures and looks, made from their photos"},
    # Its own Auto, not Deepy's image editor: H3-Image keeps poses and expressions and handles several people with someone staying
    # (12 draws of two remakes: 8 clean against Klein's 9, Viggle Image Swap's 2 and Qwen 2.1 Turbo's swaps that left people unchanged: specs).
    {"id": "frame_painter", "task": "edit_image", "label": "Frame Painter", "hint": "A remake's frames where people are replaced by the cast, painted before the video",
     "auto_template": "edit_image/H3-Image Pruned 6 Steps"},
    {"id": "shots_without_cast", "task": "gen_video", "hint": "Shots without the cast: landscapes, objects, crowds, inserts"},
    {"id": "cast_shots_lip_synced", "task": "gen_video_with_speech", "hint": "Shots where someone sings or speaks, lip-synced to the song or to their line"},
    {"id": "cast_shots", "task": "gen_video_with_refs", "hint": "Shots with the cast, keeping each person's face, hair and outfit from their pictures"},
    {"id": "people_edits", "task": "swap_video", "hint": "A remake's shots where people or characters are replaced"},
    {"id": "scene_edits", "task": "edit_video", "hint": "A remake's shots where objects, the setting, weather or light change"},
    {"id": "music", "task": "gen_song", "hint": "The songs and scores the showrunner composes"},
    {"id": "voices", "task": "gen_speech_from_description", "hint": "Each speaker's voice, made once from the voice described on their cast card"},
    {"id": "cloned_voices", "task": "gen_speech_from_sample", "hint": "Every spoken line (dialogue, narration), in its speaker's voice"},
)
ROLE_IDS = tuple(role["id"] for role in ROLES)
TASKS = {role["id"]: role["task"] for role in ROLES}
REMAKE_ROLES = ("people_edits", "scene_edits", "frame_painter")  # used by remakes alone


def planned_length(doc: dict[str, Any], shot: dict[str, Any], entry: dict[str, Any] | None) -> dict[str, str]:
    """The length a video shot asks of its generation, unless its settings name one: what the edit uses of its take. Without it, the take lasts
    as long as the template's example and the edit throws the rest away (10 s rendered for a 3.4 s shot)."""
    length = model.take_length(doc, shot)
    if entry is None or entry["task"] not in VIDEO_TASKS or length <= 0 or {"video_length", "duration_seconds"} & set(shot["generation"].get("settings") or {}):
        return {}
    return {"video_length": f"{math.ceil(length * 1000) / 1000:g}s"}


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
    """The template of each role when the user picks none: the one selected in Deepy's settings for its category, or the role's own."""
    return {role["id"]: entry_name(entry) for role in ROLES if (entry := resolve(catalog, role.get("auto_template") or role["task"])) is not None}


def chosen(doc: dict[str, Any], catalog: list[dict[str, Any]]) -> dict[str, str]:
    """The user's picks that are still templates of their category."""
    picks = doc["brief"].get("templates") or {}
    return {role: template for role, template in picks.items() if role in ROLE_IDS and (entry := resolve(catalog, template)) is not None and _fits(role, entry)}


def named(doc: dict[str, Any]) -> list[str]:
    """The templates a production names (its picks and its shots'): the catalog keeps listing a custom one that Deepy's settings dropped."""
    return [*(doc["brief"].get("templates") or {}).values(), *(shot["generation"].get("template") for shot in doc["shots"])]


def effective(doc: dict[str, Any], catalog: list[dict[str, Any]]) -> dict[str, str]:
    return {**automatic(catalog), **chosen(doc, catalog)}


def choices(doc: dict[str, Any], catalog: list[dict[str, Any]], rules: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """The picker: every role under its Deepy category's name, with that category's templates (and the user's prompt rules for each), the
    automatic one and the user's pick."""
    auto, picks, rules = automatic(catalog), chosen(doc, catalog), rules or {}
    return [{**role, "label": role.get("label") or next(entry["category"] for entry in catalog if _fits(role["id"], entry)), "auto": auto.get(role["id"]), "value": picks.get(role["id"]),
             "choices": [{"value": entry_name(entry), "label": entry.get("label") or entry["name"], "refs": entry.get("refs", False), "audio": entry.get("audio", False), "speaks": speaks_its_lines(entry),
                          "rules": rules.get(entry_name(entry), "")}
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


def speaks_its_lines(entry: dict[str, Any] | None) -> bool:
    """A template whose model voices words written in its prompt and follows no audio given to it."""
    return bool(entry and entry.get("spoken_line") and not entry["audio"])


def model_speaks(doc: dict[str, Any], shot: dict[str, Any], entry: dict[str, Any] | None) -> bool:
    """The user's Voices choice holds for this shot: its own model says its lines, and nothing is recorded. A cut on the song has lyrics, and a shot given its own audio follows it."""
    return bool(doc["brief"].get("voices_from_model")) and speaks_its_lines(entry) and not cut_on_the_song(doc) and not shot.get("audio_slice") and not (shot["generation"].get("settings") or {}).get("audio_guide")


def voicing_template(doc: dict[str, Any], catalog: list[dict[str, Any]], shown: bool) -> str | None:
    """The template of a shot whose model says its lines: the brief's for what the shot shows when its model can, else the first that can."""
    wanted = resolve(catalog, effective(doc, catalog).get("cast_shots" if shown else "shots_without_cast"))
    entries = [wanted, *(entry for entry in catalog if entry["task"] in VIDEO_TASKS and (entry["refs"] or not shown))]
    return next((entry_name(entry) for entry in entries if speaks_its_lines(entry)), None)


def with_spoken_lines(doc: dict[str, Any], shot: dict[str, Any], entry: dict[str, Any], prompt: str) -> str:
    """A shot's prompt with the lines its model says, as its model reads them, before what the prompt says about the sound. The speaker's described voice goes with them, the one thing
    that carries a voice from shot to shot, and the card of a speaker on screen whom the prompt does not name: the lines name people, and a prompt may only describe them. A line the prompt
    already has stays where it is. The showrunner's words for a recording go; a prompt the user changed keeps what it says."""
    if shot.get("updated_by") != "user":
        prompt = LIP_SYNC_WORDS.sub("", prompt)
    code, form = str(doc["brief"].get("language") or "en")[:2].lower(), entry["spoken_line"]
    ids, voices, lines = {}, [], []
    for line in shot.get("dialogue") or []:
        if not str(line.get("line") or "").strip() or model.says(prompt, line["line"]):
            continue
        member = model.speaker(doc, line["speaker"])
        name = member["name"] if member and member["name"] else "A voice"
        card = model.without_quoted_words(str((member or {}).get("description") or "")).rstrip(".")
        if line["speaker"] not in ids and card and line["speaker"] in (shot.get("cast") or []) and not re.search(rf"\b{re.escape(name)}\b", prompt, re.I):
            voices.append(f"{name}: {card}.")
        if line["speaker"] not in ids and member and str(member.get("voice_description") or "").strip():
            voices.append(f"{name}'s voice: {member['voice_description'].strip().rstrip('.')}.")
        number = ids.setdefault(line["speaker"], len(ids) + 1)
        note = str(line.get("note") or "").strip()
        lines.append(("" if line["speaker"] in (shot.get("cast") or []) else "Off screen, ")
                     + form["line"].format(speaker=name, note=f", {note}," if note else "", id=f"S{number}", language=LANGUAGE_NAMES.get(code, code), line=line["line"].strip()))
    if not lines:
        return prompt
    text = " ".join([*voices, *lines])
    at = prompt.find(form["before"]) if form.get("before") else -1
    closed = lambda part: part if part[-1:] in (".", "!", "?", ":", ";", "") else f"{part}."  # a prompt may end without its full stop: the lines start a sentence
    return f"{closed(prompt[:at].rstrip())} {text}\n{prompt[at:]}" if at >= 0 else f"{closed(prompt.rstrip())} {text}"


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


def cut_on_the_song(doc: dict[str, Any]) -> bool:
    return doc["brief"].get("kind") == "music_video" or model.on_the_song(doc)


def _seconds(frames: int, fps: float) -> float:
    return math.floor(frames / fps * 1000) / 1000  # rounded down: the model rounds a length up to its next frame count, which is this one


def take_lengths(entry: dict[str, Any], count: int = 5) -> list[float]:
    """The lengths a template's model really makes, shortest first: its first length, then its frame steps."""
    first, step, fps = entry["take_frames"]
    return [_seconds(first + index * step, fps) for index in range(count)]


def take_lengths_in_range(entry: dict[str, Any], low: float | None, high: float | None) -> list[float]:
    """The lengths a template's model really makes inside the user's range, up to its longest window unless its model continues a take past it
    (sliding windows: WanGP joins the windows into one take); the nearest one when none falls inside."""
    first, step, fps = entry["take_frames"]
    last = entry["longest_frames"] or first + 4 * step
    if entry["sliding_window"] and high:
        last = max(last, math.ceil(high * fps))
    lengths = [_seconds(frames, fps) for frames in range(first, last + 1, step)]
    margin = step / fps / 2  # what the model makes nearest to the end of the range
    inside = [length for length in lengths if (low is None or length >= low - margin) and (high is None or length <= high + margin)]
    return inside or [min(lengths, key=lambda length: abs(length - (low or high)))]


def outside_shot_range(doc: dict[str, Any], duration: float) -> bool:
    low, high = model.shot_range(doc)
    return bool(low and duration < low - SHOT_RANGE_MARGIN_S or high and duration > high + SHOT_RANGE_MARGIN_S)


def whole_take(doc: dict[str, Any], catalog: list[dict[str, Any]], shot: dict[str, Any]) -> float:
    """The nearest length a shot's model really makes when the shot's plan is another one, else 0: the edit plays a take from its start, so a shot shorter than its take
    would lose the end of its action. A speaking shot ends after its line (the model fills what follows with invented words), the song decides the cuts of a music video
    (its bars and lines are often shorter, and a lip-synced shot only loses the silence after its slice), and what the user set stays."""
    entry = resolve(catalog, shot["generation"].get("template"))
    if entry is None or entry["task"] not in VIDEO_TASKS or shot.get("locked") or shot.get("updated_by") == "user" or cut_on_the_song(doc) or shot.get("audio_slice"):
        return 0.0
    speaks = entry["audio"] or entry["soundtrack"] and any(line.get("speaker") in (shot.get("cast") or []) for line in shot.get("dialogue") or [])
    if speaks and shot.get("dialogue") and not shot.get("audio_slice") and not model_speaks(doc, shot, entry):
        return 0.0
    if entry["shortest_s"] < 1:  # its model has no first length: it makes the length it is asked for
        return 0.0
    first, step, fps = entry["take_frames"]
    planned = float(shot.get("duration_s") or 0.0)
    length = _seconds(first + max(0, round((planned * fps - first) / step)) * step, fps)
    return length if abs(length - planned) > 0.01 else 0.0


def shot_entry(doc: dict[str, Any], shot: dict[str, Any], catalog: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The template a shot renders with: one left on gen_video, nobody's choice, takes the brief's Models for what it shows and says."""
    template = shot["generation"].get("template")
    return resolve(catalog, template == "gen_video" and template_for(doc, shot, catalog) or template)


def rules_in_use(doc: dict[str, Any], catalog: list[dict[str, Any]], rules: dict[str, str]) -> dict[str, str]:
    """The user's prompt rules of the templates this production makes its shots and media with (a remake's editing templates only in a remake)."""
    if not rules:
        return {}
    remake = doc["brief"].get("kind") == "remake"
    names = [*(template for role, template in effective(doc, catalog).items() if remake or role not in REMAKE_ROLES), *(entry_name(entry) for shot in doc["shots"] if (entry := shot_entry(doc, shot, catalog)))]
    return {name: rules[name] for name in dict.fromkeys(names) if rules.get(name) and (remake or name.partition("/")[0] not in EDIT_TASKS)}


def advice(doc: dict[str, Any], catalog: list[dict[str, Any]], any_length: bool = False, rules: dict[str, str] | None = None) -> dict[str, Any]:
    """What the director should choose for this production's shots and media, with the user's prompt rules of the templates it uses."""
    picks, user = effective(doc, catalog), chosen(doc, catalog)
    if doc["brief"].get("kind") != "remake":  # the editing templates are a remake's: nothing about them in any other production
        user = {role: template for role, template in user.items() if role not in REMAKE_ROLES}
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
    if doc["brief"].get("kind") == "remake":  # every shot changes the source: the two editing templates, the painter of frames whose people are replaced, and the image editor for the rest
        result = {"rule": "Each shot takes the template of what changes in it: people_edits for people or characters, scene_edits for objects, setting, weather or light. Read its input guidance with wangp_templates first. "
                          "paint_frame paints a frame whose people are replaced with frame_painter, any other frame and the character sheets with image_edits.",
                  **{role: picks[role] for role in ("people_edits", "scene_edits", "frame_painter", "image_edits") if role in picks}}
    if user:
        result["user_choices"] = user
        result["user_rule"] = "user_choices are the user's picks for these kinds of shots and media: use them, never another template for the same kind."
    if doc["brief"].get("kind") == "music_video" and doc["brief"].get("song") and "cast_shots_lip_synced" in result:
        result["song"] = "Give each lip-synced shot its exact slice of the song as driving audio."
    voiced = bool(doc["brief"].get("voices_from_model")) and not cut_on_the_song(doc) and voicing_template(doc, catalog, False) is not None
    if voiced:  # the user's choice: the video model says the lines, with no recording for its lips to follow
        for key in ("cast_shots_lip_synced", "lip_synced_without_cast", "lip_sync"):
            result.pop(key, None)
        result.update({role: template for role, shown in (("cast_shots", True), ("shots_without_cast", False)) if role not in user and (template := voicing_template(doc, catalog, shown))})
        result["spoken_lines"] = ("The user chose that the video model says the lines (the brief's Voices). Write them in the shots' dialogue as usual and do not call speak: render writes them into the prompt. "
                                  "Leave their words and any lip-sync wording out of the prompt, quote nobody there (the model says what a prompt quotes) and name who speaks. Give a shot about 0.4 s a word plus a second: with more time, the model adds words of its own. A shot with lines takes a template that can say them: "
                                  + ", ".join(entry_name(entry) for entry in catalog if entry["task"] in VIDEO_TASKS and speaks_its_lines(entry)) + ".")
    speaking_exception = "" if voiced else " A speaking shot is the exception, as it ends after its line."
    shortest = {template: entry["shortest_s"] for key in ("cast_shots", "shots_without_cast", "cast_shots_lip_synced", "lip_synced_without_cast")
                if (template := result.get(key)) and (entry := resolve(catalog, template)) and entry["shortest_s"] >= 1}
    if shortest and cut_on_the_song(doc):
        result["shot_length"] = ("The song decides the cuts, on its bars and lines, so a shot can be shorter than its model's shortest take (shortest_take_s): it plays the start of the take. "
                                 "A lip-synced shot only loses the silence after its slice; give any other shot an action that fits its length.")
        result["shortest_take_s"] = shortest
    elif shortest and any_length:
        result["shot_length"] = ("The user lets you pick any shot length. A template's model makes no take shorter than these seconds, and a shot plays the start of its take: "
                                 "under them, write an action that fits the shot, or its end is lost.")
        result["shortest_take_s"] = shortest
    elif shortest:
        result["shot_length"] = ("A shot lasts one of its template's take_lengths_s, the only lengths its model makes (others are moved to the nearest): shorter, the end of its action would be lost."
                                 f"{speaking_exception} Count your shots from this: a faster pace takes fewer words per shot, not shorter shots.")
        result["take_lengths_s"] = {template: take_lengths(resolve(catalog, template)) for template in shortest}
    if wanted := model.shot_range_words(doc):
        result["shot_length_range"] = (f"The user wants video shots of {wanted}: plan them so, whatever the pacing. " + ("Cut on the song's bars inside this range, as far as the song allows."
                                       if cut_on_the_song(doc) else speaking_exception.strip()))
        if "take_lengths_s" in result:
            result["take_lengths_s"] = {template: take_lengths_in_range(resolve(catalog, template), *model.shot_range(doc)) for template in shortest}
    if used := rules_in_use(doc, catalog, rules or {}):
        result.update(user_prompt_rules=used, user_prompt_rules_note=PROMPT_RULES_NOTE)
    return result


def places_missing(doc: dict[str, Any], catalog: list[dict[str, Any]]) -> list[str]:
    """Video shots that happen in no place of the cast: words in a prompt give a different set each time, only a picture keeps it."""
    videos = [shot for shot in doc["shots"] if shot.get("included", True) and (entry := resolve(catalog, shot["generation"].get("template"))) and entry["task"] in VIDEO_TASKS]
    places = {member["id"] for member in doc["cast"] if model.is_place(member) and (member.get("master") or member.get("images"))}
    if len(videos) < 3:
        return []
    if not places:
        return [f"The cast has no place with a picture, so each of the {len(videos)} shots will invent its own set. Unless every shot happens somewhere else, add each recurring location as a cast member with role \"place\" and a picture of it empty, and list it in the cast of the shots there."]
    without = [shot["id"] for shot in videos if not places & set(shot.get("cast") or [])]
    return [f"Shots {', '.join(without)} list no place in their cast, so they invent their own set: list the place they happen in." + (" A template that takes no reference images cannot hold it." if without else "")] if without else []


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
        driven = (entry["audio"] or entry["soundtrack"] and any(line.get("speaker") in (shot.get("cast") or []) for line in lines)) and not model_speaks(doc, shot, entry)
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
        if entry["task"] in VIDEO_TASKS and shot.get("included", True) and shot.get("updated_by") != "user" and not driven and not cut_on_the_song(doc) and outside_shot_range(doc, float(shot.get("duration_s") or 0.0)):
            warnings.append(f"{shot['id']} lasts {float(shot.get('duration_s') or 0.0):g} s, but the user wants video shots of {model.shot_range_words(doc)}: change its length, or say why it must differ.")
        if doc["brief"].get("voices_from_model") and lines and entry["task"] in VIDEO_TASKS and not speaks_its_lines(entry) and not cut_on_the_song(doc) and not shot.get("audio_slice") and not settings.get("audio_guide") \
                and (able := voicing_template(doc, catalog, bool(shown))):
            warnings.append(f"{shot['id']} has lines and the user wants the video model to say them, but {entry_name(entry)} cannot: use {able}.")
        if model_speaks(doc, shot, entry) and lines and shot.get("updated_by") != "user":
            words, duration = sum(len(line["line"].split()) for line in lines), float(shot.get("duration_s") or 0.0)
            if duration - words * WORD_S > SPARE_S and duration > entry["shortest_s"] + 0.1 and not any(model.shot_range(doc)):
                warnings.append(f"{shot['id']} gives {words} words {duration:g} s: with time left, its model adds words of its own. Make it about {words * WORD_S + 1:.0f} s, as near as its model's lengths allow, or give it more to say.")
            if model.without_quoted_words(prompt := re.sub(r"<d>.*?</d>", " ", str(settings.get("prompt") or ""), flags=re.S)) != prompt.strip(" ;,"):
                warnings.append(f"{shot['id']}'s prompt quotes words that are not its lines: its model will say them. Describe them instead.")
        if lines and "<d>" in str(settings.get("prompt") or "") and not model_speaks(doc, shot, entry):  # the model would speak in a voice of its own, not the speaker's sample
            warnings.append(f"{shot['id']} writes its spoken words in the prompt (<d>): the model voices them itself, in a voice of its own. "
                            "Speak the lines with `speak` and leave their words out of the prompt: render drives the lips with the recordings.")
    return warnings


def script_warnings(doc: dict[str, Any]) -> list[str]:
    """For a plan being written: what it loses or adds to the user's script. Not in lint, which every read of the production shows: a line cut on the user's request stays cut. The song's words have their own check."""
    if not doc["shots"] or cut_on_the_song(doc):
        return []
    missing, added = model.script_differences(doc)
    listed = lambda lines: " | ".join(lines[:6]) + (f" | and {len(lines) - 6} more" if len(lines) > 6 else "")
    return [*([f"No shot says these lines of the user's script yet, word for word: {listed(missing)}. Give them to the shots that should say them, exactly as written, unless the user asked to change them."] if missing else []),
            *([f"These lines are not in the user's script: {listed(added)}. Use the script's words, unless the user asked for this line."] if added else [])]


def template_for(doc: dict[str, Any], shot: dict[str, Any], catalog: list[dict[str, Any]]) -> str | None:
    """The template of a shot that names none, from the brief's Models and what the shot shows and says."""
    picks = effective(doc, catalog)
    shown = set(cast_with_pictures(doc)) & set(shot.get("cast") or [])
    if doc["brief"]["kind"] == "story":
        return picks.get("image_edits" if shown else "images")
    if doc["brief"].get("voices_from_model") and shot.get("dialogue") and not shot.get("audio_slice") and not cut_on_the_song(doc) and (template := voicing_template(doc, catalog, bool(shown))):
        return template
    if shot.get("audio_slice") or shot.get("dialogue") and not model.on_the_song(doc):
        return (lip_synced_with_cast(picks, catalog) if shown else None) or picks.get("cast_shots_lip_synced")
    return picks.get("cast_shots" if shown else "shots_without_cast")
