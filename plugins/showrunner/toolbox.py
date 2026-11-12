# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""The showrunner toolbox: how Deepy (and other agents) read and write a production while directing it."""
from __future__ import annotations

import copy
import difflib
import json
import math
import re
import time
from pathlib import Path
from typing import Any, Literal

from tqdm import tqdm

from shared.utils.virtual_media import write_view

from . import assemble, media, production as model, remake_groups, review, templates
from .director import media_entries

MEDIA_SETTINGS = model.MEDIA_SETTINGS
_REFERENCE = re.compile(r"^(@|[a-z]+:[0-9a-f]{6,}$)")  # an @alias path or a Gallery media id
CUT_FAULTS = {"dropped": "The source cuts to another camera setup at this frame of the take, and the take does not: everything after that cut in the source is missing, however few frames the number suggests.",
              "moved": "The take makes this cut of the source late or early.", "added": "The take cuts here and the source does not."}

# The words of a voice sample made from a description: neutral, about ten seconds, so that every line spoken from it keeps its voice.
CASTING_LINE = ("Some stories begin quietly, on an ordinary morning. Listen closely, and you will hear how this one goes, from the very first word "
                "to the last.")
SUMMARY = "Showrunner productions (music videos, films, ads, stories): read the production document, plan shots, render and judge takes, assemble the edit."
GUIDE = """Read wangp://skills/showrunner/directing before directing a production: it is the method (planning, template choice, shooting, review, edit).

- The production document is the plan: read `production` first and after long steps, and record every change with these actions.
- Locked shots and cast members are the user's, and the user's verdicts override yours.
- Choose templates as `template_advice` says (the cast's pictures as references, never as image_start) and fix the `warnings` returned.
- A shot's settings hold only what the shot changes (prompt, media): its template brings the model, LoRAs and steps, the brief the resolution, its duration_s the length. `plan` validates them and returns what they will render at.
- Paths are given as you can use them in settings."""
FULL_DOCUMENT_CHARACTERS = 28000  # what Deepy can read of one call
SHOT_ROLES = ("cast_shots", "shots_without_cast", "cast_shots_lip_synced", "lip_synced_without_cast", "people_edits", "scene_edits", "pages", "pages_with_cast")  # template_advice's templates for shots
# How plan and add_shots name what render sends with each shot, beside the prompt format of its model.
PICTURE_ORDER = ("Write each shot's prompt in the format its template's model has in prompt_guides. reference_pictures: the cast members whose "
                 "picture render gives each shot, in this order, as reference images (a model that numbers images counts a start or end image "
                 "first): bind each to its picture in the prompt. A song slice or recorded line render keeps as the shot's soundtrack is no audio "
                 "reference: ask for the singer's or speaker's lips moving in sync instead.")


def mistyped_words(doc: dict[str, Any], texts: list[str]) -> list[str]:
    """Words of a text that are close to a word of the user's without being it: a mistyped copy ("Crranes" for the title's "Cranes")."""
    written = [doc["title"], doc["brief"].get("text") or "", doc["brief"].get("script") or "", doc["brief"].get("lyrics") or "",
               *(entry["text"] for entry in doc["log"] if entry.get("by") == "user"), *(member["name"] for member in doc["cast"])]
    known = set(re.findall(r"[^\W\d_]{3,}", " ".join(written).casefold()))
    found = []
    for text in texts:
        for word in re.findall(r"[^\W\d_]{4,}", text):
            close = None if word.casefold() in known else difflib.get_close_matches(word.casefold(), known, n=1, cutoff=0.85)
            if close:
                found.append(f'"{word}" in "{text}": the user wrote "{close[0]}"')
    return found


# A character sheet follows the Viggle Remake MCP server (Copyright Viggle AI, Apache License 2.0): a portrait and three whole views give the painter a person's looks, and no framing or expression to copy.
SHEET_LAYOUT = ("Left third: a large head-and-shoulders portrait facing the camera. Right two thirds: the same character three times full body, head to shoes visible, standing upright with arms relaxed - front view, three-quarter "
                "view, back view. Identical in every view. No text.")
SHEET_FROM_WORDS = "Character reference sheet, {style} on a plain light-grey background, soft even lighting. The character: {description}. " + SHEET_LAYOUT
SHEET_FROM_PHOTO = ("Character reference sheet of the person in the photo, {style} on a plain light-grey background, soft even lighting. Exactly the same real face, hairstyle, skin tone, age and body build as the photo. "
                    "The outfit: {description}. " + SHEET_LAYOUT)
SHEET_SIZE = "1216x832"
# The visual inspector on one picture, source or painted, in the same words: told it compares two, it says sizes match that do not. It is never told the brief.
FRAME_PEOPLE = ("This is one frame of a video. List every person or character visible, even if only partly, from behind or out of focus, one line each, with no names and nothing you cannot see: where they are in the picture (left, "
                "centre or right; foreground or background); which part of them shows; how tall their head is as a percentage of the picture's height, and whether the picture's edge cuts it; their hair or what covers their head, and "
                "their clothes; which way the head faces; the eyes, brows and mouth (open or closed, smiling or not); sharp or blurred. Then one line for the camera (extreme close-up, close-up, medium shot or wide) and one for the background.")
# The same without hair and clothes, for the painter's directions: told a replaced person's hair and clothes, it keeps them (s7 of the A24 teaser: Batman in 3 of 3 draws without those words, 1 of 3 with them).
FRAME_POSES = ("This is one frame of a video. List every person or character visible, even if only partly, from behind or out of focus, one line each, with no names and nothing you cannot see: where they are in the picture (left, "
               "centre or right; foreground or background); which part of them shows; how tall their head is as a percentage of the picture's height, and whether the picture's edge cuts it; which way the head faces; the eyes, brows "
               "and mouth (open or closed, smiling or not); sharp or blurred. Then one line for the camera (extreme close-up, close-up, medium shot or wide) and one for the background.")
PAINT_DRAWS = 3  # a draw that leaves the picture as it was is drawn again: the painter often does
LOOKS_WORDS = re.compile(r"\b(?:hair|haired|blonde?|brunette|redhead|bald|glasses|sunglasses|spectacles|beard|bearded|moustache|mustache|face)\b", re.IGNORECASE)  # what a replaced person loses (colours name clothes too)
BUILD_WORDS = re.compile(r"\b(?:slim|slender|thin|skinny|lanky|lean|wiry|petite|small|tiny|little|tall|short|stocky|broad-shouldered|muscular|athletic|plump|chubby|stout|heavy(?:set)?|round|curvy|big|large|frail)\b", re.IGNORECASE)  # a card's build, from its first phrase ("a lanky clown")
HEAD_SHARE = r"(\d{1,3})(?:\s*-\s*(\d{1,3}))?\s*(?:%|percent)"  # the inspector's head sizes: "22 %", "20-25%", "22 percent"
CUT_GAP, FRAME_GAP = 3, 72  # as the Viggle Remake MCP server: a painted frame keeps 3 frames of its own shot from a cut, and two of one shot are 3 s apart
LARGE_HEAD = 30  # percent of the picture's height: below it a real change can touch too little of the picture to measure
SAME_PICTURE = 0.15  # media.changed_share of a draw that changed nobody (0 to 0.13 on the teaser; 0.16 and up for a person replaced)
PAINT_VERIFY = ("Verify this picture before going on: read `before` and `after`, person by person. It fails when a replaced person's old hair, face or clothes are still there; a head is much larger or smaller, or elsewhere; someone "
                "seen from behind now shows a face, or someone new appears; eyes, brows or mouth differ from `before`; the camera or background changed; the outfit is not the one on the character's sheet. On a failure call paint_frame "
                "again with the same people: each try draws differently. After 3 tries keep the closest (pass its `picture` back) and say in your report who is wrong and how.")


class ShowrunnerToolbox:
    def __init__(self, showrunner):
        self.showrunner = showrunner

    @property
    def store(self):
        return self.showrunner.store

    def register(self, plugin) -> None:
        plugin.register_deepy_toolbox(SUMMARY, guide=GUIDE, toolbox_id="showrunner", display_name="Showrunner")
        routing = "A production opened in Showrunner, or any request naming a Showrunner production id"
        plugin.register_deepy_action(self.production, uses_gpu=False, routing=routing, summary="Read a production: summary, or one section, shot, its files or the full document.")
        plugin.register_deepy_action(self.list_productions, uses_gpu=False, summary="List the user's productions, one line each (to read one with production, and bring its files with import_media).")
        plugin.register_deepy_action(self.import_media, uses_gpu=False, summary="Copy files of another production (its cast pictures, voices, takes) into this one, ready to use.")
        plugin.register_deepy_action(self.plan, uses_gpu=False, summary="Write cast, style and shots in one call (keeps the user's shots) and validate their settings.", instead_of="wangp_io")
        plugin.register_deepy_action(self.update_shot, uses_gpu=False, summary="Change fields of one shot (refused on locked shots or when the user changed it after your revision).")
        plugin.register_deepy_action(self.add_shots, uses_gpu=False, summary="Insert shots at a position (or at the end).")
        plugin.register_deepy_action(self.remove_shot, uses_gpu=False, summary="Remove a shot that is not locked.")
        plugin.register_deepy_action(self.move_shots, uses_gpu=False, summary="Move shots before another shot (or to the end).")
        plugin.register_deepy_action(self.cast, paths={"images": "read", "master": "read", "voice": "read"}, uses_gpu=False,
                                     summary="Add, update or remove a cast member: name, role, description, reference images, master image, voice description or sample.")
        plugin.register_deepy_action(self.speak, summary="Speak a shot's dialogue lines, each speaker in one voice for the whole production, and record the audio on the lines.")
        plugin.register_deepy_action(self.render, summary="Generate a take of each listed shot from its document settings, record it and return its media id.")
        plugin.register_deepy_action(self.record_take, paths={"media": "read"}, uses_gpu=False, summary="Record media you made another way as a take of a shot.")
        plugin.register_deepy_action(self.verdict, uses_gpu=False, summary="Record your inspection of a take: ok or not, a short note and the measured values.")
        plugin.register_deepy_action(self.edit, paths={"song": "read"}, uses_gpu=False,
                                     summary="Set the edit's sound and subtitles: soundtrack mode, the production's song, narration or score tracks, SRT files.")
        plugin.register_deepy_action(self.change_singer_voice, paths={"voices": "read"},
                                     summary="Only when the user asks: the production's song sung in a cast member's voice or a voice sample (two for a duet), its music kept (SeedVC).")
        plugin.register_deepy_action(self.check_song, summary="Listen to the production's song: which lines of the lyrics it really sings, and when (the subtitles follow it).")
        plugin.register_deepy_action(self.deep_review, summary="Measure the last version against its script: words, subtitles, voices, lips, sound and picture, with the findings recorded on the version.")
        plugin.register_deepy_action(self.assemble, uses_gpu=False, summary="Render the edit (chosen takes in order, trims, transitions, soundtrack) as the production's deliverable.")
        plugin.register_deepy_action(self.ingest_source, uses_gpu=False, listed=False, summary="A remake: read its source video, find the cuts and make one shot per camera shot, each with its part of the source.")
        plugin.register_deepy_action(self.make_sheet, listed=False, summary="A remake: make a cast member's character sheet, a portrait and three whole views, as their picture for the painter.")
        plugin.register_deepy_action(self.look_at_frame, uses_gpu=False, listed=False, summary="A remake: what the visual inspector sees in one frame of a shot's source, person by person: where, how large, what shows, expression, focus.")
        plugin.register_deepy_action(self.paint_frame, paths={"picture": "read", "references": "read"}, listed=False, summary="A remake: paint one frame of a shot's source with the image editor from what happens to each person in it, or attach your own picture, as that shot's painted frame, and describe the result to verify it.")
        plugin.register_deepy_action(self.check_take, uses_gpu=False, listed=False, summary="A remake: measure a take against its source: cuts moved, dropped or added, black bands at its edges.")
        plugin.register_deepy_action(self.log, uses_gpu=False, summary="Add a short line to the director's log the user reads.")
        plugin.register_deepy_action(self.status, uses_gpu=False, summary="Shots by status, takes waiting for a verdict, the latest deliverable (nothing renders in the background: render returns when done).")

    # Helpers ----------------------------------------------------------------------------------------------------------------------
    def _own(self, ctx):
        """The production whose director is calling: the caller works in its media folder (its workspace). None for other callers (the user's
        Deepy chat, agents, scripts), who may name any production."""
        folder = Path(ctx.outputs_dir).resolve()
        return self.store.get(folder.parent.name) if folder.name == "media" and folder.parent.parent == self.store.root.resolve() else None

    def _production(self, ctx, production: str, read: bool = False):
        """The production an action works on. A director changes only its own production, and reads another one only when its brief lets it
        (read_other_productions, on by default); it is never told which others exist unless it may read them."""
        own = self._own(ctx)
        if own is not None and production != own.id:
            shared = own.snapshot()["brief"]["read_other_productions"]
            if not (read and shared):
                raise ValueError(f"You direct the production {own.id!r} ({own.snapshot()['title']}): it is the only one you can {'change' if shared else 'use'}.")
            if production not in self.store.ids():
                raise ValueError(f"No production has the id {production!r}: `list_productions` lists them.")
        try:
            return self.store.get(production)
        except KeyError as error:
            raise ValueError(f"{error.args[0]} Known productions: {', '.join(self.store.ids()[:12]) or 'none'}.") from None

    def _fixing(self, ctx, target) -> list[str] | None:
        """The shots the user named for the director's fix or shoot, none in a review, None in any other task."""
        director = target.snapshot()["director"]
        if self._own(ctx) is None or director.get("state") not in ("queued", "running", "pausing"):
            return None
        named = director.get("shots") or []
        return {"reshoot": named, "shoot": named or None, "review": []}.get(director.get("task"))

    def _only_the_fix(self, ctx, target, shots=(), what: str = "") -> None:
        """The user named shots to fix: in that task the director changes nothing else, because rounds that spread beyond what was asked made films worse than their first version. A global
        change stays possible, in a task of its own: the user asks for it in the chat, or lets the fix change other things, or says yes when the director explains that the fault needs it."""
        allowed = self._fixing(ctx, target)
        if allowed is None:
            return
        outside = what or ", ".join(shot for shot in shots if shot not in allowed)
        if outside and target.snapshot()["director"]["task"] == "review":
            raise ValueError("A review changes nothing: end your turn saying what you would change, and the user decides.")
        if outside:
            raise ValueError(f"You are working on {', '.join(allowed)} at the user's request and everything else stays as it is, so {outside} cannot change in this task. If the fault really needs it, "
                             "end your turn saying what you would change and what it costs: the user's answer starts a task that can do it.")

    @staticmethod
    def _since_last_version(target, doc: dict[str, Any]) -> dict[str, Any]:
        """What this version changes from the one before, for the director's report: all of it, in plain words."""
        versions = doc["deliverables"]
        if len(versions) < 2:
            return {}
        return {"changed_since_last_version": model.version_changes(target.version(versions[-2]["id"]), target.version(versions[-1]["id"]), 20, (versions[-2].get("duration_s"), versions[-1].get("duration_s"))) or ["nothing in the edit"]}

    def _catalog(self, doc: dict[str, Any]) -> list[dict[str, Any]]:
        return self.showrunner.catalog(doc)

    def _script_warnings(self, ctx, target) -> list[str]:
        """While a plan is written; a fix keeps everything else as it is."""
        return [] if self._fixing(ctx, target) is not None else templates.script_warnings(target.snapshot())

    def _warnings(self, production, shot_ids=None) -> list[str]:
        doc = production.snapshot()
        remake = [warning for shot in doc["shots"] if shot.get("source_range") and (shot_ids is None or shot["id"] in shot_ids) for warning in remake_groups.prompt_warnings(shot)]
        return self._hold_lengths(production, doc, shot_ids) + templates.lint(doc, self._catalog(doc), shot_ids) + remake

    def _hold_lengths(self, production, doc: dict[str, Any], shot_ids=None) -> list[str]:
        """Every shot plays its whole take, on a length its model really makes, unless the user's settings let Deepy pick any length (`whole_take` says which shots this concerns)."""
        if self.showrunner.host.settings()["any_shot_length"]:
            return []
        catalog, ops, notes = self._catalog(doc), [], []
        for shot in doc["shots"]:
            length = templates.whole_take(doc, catalog, shot) if shot_ids is None or shot["id"] in shot_ids else 0.0
            if length:
                ops.append({"op": "update_shot", "shot": shot["id"], "fields": {"duration_s": length}})
                notes.append(f"{shot['id']} now lasts {length:g} s instead of {shot['duration_s']:g} s, the nearest length its model makes.")
        if ops:
            production.apply(ops, by="system", record=False)
        return notes

    @staticmethod
    def _usable(ctx, production, doc: dict[str, Any]) -> dict[str, Any]:
        """The document with every media path as the caller names it (an @alias such as @outputs/uploads/song.wav), never a full path."""
        doc = copy.deepcopy(doc)
        resolve = lambda path: ctx.virtualize(str(production.resolve(path))) if path else path
        brief = doc["brief"]
        brief["song"] = resolve(brief.get("song"))
        brief["source"] = resolve(brief.get("source"))
        brief["references"] = [resolve(path) for path in brief.get("references", [])]
        voices = {member["id"]: member.get("voice") for member in doc.get("cast", [])}  # as the document keeps them, before they are renamed below
        for member in doc.get("cast", []):
            member.update(images=[resolve(path) for path in member.get("images", [])], master=resolve(member.get("master")), voice=resolve(member.get("voice")),
                          looks=[{**look, "master": resolve(look.get("master"))} for look in member.get("looks", [])])
        def media(settings):  # as the caller passes them back in settings
            for key in MEDIA_SETTINGS & set(settings):
                settings[key] = [resolve(item) for item in settings[key]] if isinstance(settings[key], list) else resolve(settings[key])
        for shot in doc.get("shots", []):
            shot.pop("source_poster", None)
            media((shot.get("generation") or {}).get("settings") or {})
            for painted in shot.get("painted_frames") or []:
                painted.update(path=resolve(painted["path"]), source=resolve(painted["source"]))
            for line in shot.get("dialogue") or []:  # a line's audio while it is in its speaker's current voice
                if line.get("audio") and line.get("voice") == voices.get((model.speaker(doc, line.get("speaker")) or {}).get("id")):
                    line["audio"] = resolve(line["audio"])
                else:
                    line.pop("audio", None), line.pop("duration_s", None)
                line.pop("voice", None), line.pop("spoken", None)
            for take in shot.get("takes", []) if isinstance(shot.get("takes"), list) else []:
                take["path"] = resolve(take.get("path"))
                media(take.get("settings") or {})
                take.pop("poster", None), take.pop("strip", None)
        for item in doc.get("bin", []) + doc.get("deliverables", []):
            item["path"] = resolve(item.get("path"))
            item.pop("poster", None), item.pop("strip", None)
        for item in doc["edit"].get("audio_tracks", []) + doc["edit"].get("subtitles", []) + doc["edit"].get("overlays", []):
            item["path"] = resolve(item.get("path"))
        return doc

    def _generation(self, ctx, target, shot: dict[str, Any], final: bool = False) -> dict[str, Any]:
        """A shot's settings as render sends them: the brief's format as a draft or final, media named the caller's way, the cast's pictures as references."""
        size = self.showrunner.shot_size(target, target.snapshot(), shot, final=final)
        settings = {"template": shot["generation"].get("template", "gen_video"), "resolution": size,
                    **self._resolve_settings(ctx, target, shot["generation"].get("settings", {}))}  # the brief's format unless the shot sets its own
        if shot.get("source_range"):  # a shot made from a video follows its video's shape, whatever an older plan wrote
            settings["resolution"] = size
        entry = templates.resolve(self._catalog(target.snapshot()), settings["template"])
        settings.update(templates.planned_length(target.snapshot(), shot, entry))
        pictures = self._cast_pictures(target.snapshot(), shot)
        if entry is not None and entry["refs"] and pictures and not settings.get("image_refs") and entry["task"] not in templates.EDIT_TASKS:  # the cast's identity in every shot that shows them; a remake's shot takes its painted frames instead
            settings["image_refs"] = [ctx.virtualize(str(target.resolve(path))) for path in pictures]
        if entry is not None and entry["task"] == "edit_image" and entry.get("sized_by_ref") and settings.get("image_refs") and "video_prompt_type" not in settings:
            settings["video_prompt_type"] = "I"  # an illustrated page keeps the production's format: its references bring the cast, not their picture's size
        if templates.model_speaks(target.snapshot(), shot, entry) and settings.get("prompt"):
            settings["prompt"] = templates.with_spoken_lines(target.snapshot(), shot, entry, settings["prompt"])
        lines = self._driving_lines(target.snapshot(), shot, entry)
        if lines and not settings.get("audio_guide") and not shot.get("audio_slice"):  # several lines drive the shot one after the other
            paths = [target.resolve(line["audio"]) for line in lines]
            settings["audio_guide"] = ctx.virtualize(str(paths[0] if len(paths) == 1 else media.joined_audio(paths, target.folder / "media" / "slices")))
        song = target.snapshot()["brief"].get("song")
        if song and shot.get("audio_slice") and not settings.get("audio_guide") and entry is not None and (entry["audio"] or entry["soundtrack"]):  # a slice is of the song: render cuts it
            settings["audio_guide"] = ctx.virtualize(str(target.resolve(song)))
        if letters := templates.soundtrack_letters(shot, entry, settings):
            settings["audio_prompt_type"] = letters  # the driving audio as a kept soundtrack the picture follows, not a voice reference
        if entry is not None and entry.get("video_edit") and settings.get("video_mask"):
            settings["video_prompt_type"] = settings.get("video_prompt_type", "IVU").replace("U", "A")  # WanGP ignores a mask beside U
        return settings

    @staticmethod
    def _driving_lines(doc: dict[str, Any], shot: dict[str, Any], entry: dict[str, Any] | None, spoken: bool = True) -> list[dict[str, Any]]:
        """The dialogue lines that drive a shot's picture, in order (spoken=False: whether recorded or not): every line for a template driven
        by speech, the lines of the people it shows for a reference template that follows a soundtrack, never a narrator's line over them."""
        lines = model.spoken_lines(doc, shot) if spoken else shot.get("dialogue") or []
        if entry is None or model.on_the_song(doc) or templates.model_speaks(doc, shot, entry):  # on a cut that plays the song, lines are lyrics: the song drives the lips, they only give the subtitles
            return []
        shown = [line for line in lines if line.get("speaker") in shot.get("cast", [])]
        return shown if entry["soundtrack"] and not entry["audio"] else shown or lines if entry["audio"] else []

    def _generation_with_loras(self, ctx, target, shot: dict[str, Any], final: bool = False) -> tuple[dict[str, Any], list[Any] | None]:
        """A shot's settings with the LoRAs of its cast (Settings > LoRAs on cast cards), and the LoRAs before them."""
        settings, doc = self._generation(ctx, target, shot, final), target.snapshot()
        return self.showrunner.with_cast_loras(doc, shot, templates.resolve(self._catalog(doc), settings["template"]), settings)

    def _validate(self, ctx, target, shot_ids: list[str]) -> tuple[list[str], dict[str, dict[str, float | None]]]:
        """The listed shots' settings checked as render will send them (drawn and locked shots skipped): their problems, one line per shot ("s4: ..."),
        and what they will render at as the validator measures it, {size: {shot: seconds}}."""
        problems, batch = [], []
        for shot in [item for item in target.snapshot()["shots"] if item["id"] in shot_ids and not item["locked"] and item["generation"].get("template") != templates.DRAWN]:
            try:
                batch.append((shot["id"], self._generation_with_loras(ctx, target, shot)[0]))
            except Exception as error:
                problems.append(f"{shot['id']}: {error}")
        found, renders = self._check_settings(ctx, batch) if batch else ([], {})
        return problems + found, renders

    def _check_settings(self, ctx, batch: list[tuple[str, dict[str, Any]]]) -> tuple[list[str], dict[str, dict[str, float | None]]]:
        """One validation of all the shots (one step in the window); an error it cannot tie to a task stopped the check of them all: each shot is checked alone then, to name it."""
        renders: dict[str, dict[str, float | None]] = {}
        try:
            validation = ctx.call("wangp_validate", {"settings": [settings for _, settings in batch]})
            errors = [item for item in validation.get("diagnostics", []) if item.get("severity") == "error"]
            errors += [{"message": validation.get("error")}] if validation.get("status") == "error" else []
            for task in validation.get("tasks", []):
                task = {**validation.get("every_task", {}), **task}  # a batch states once what every shot shares
                size = (task.get("resolution_assessment") or {}).get("aligned")
                seconds = (task.get("window_plan") or {}).get("total_seconds") or (task.get("requested_video") or {}).get("duration_seconds")
                if size:
                    renders.setdefault(size, {})[batch[task["index"]][0]] = round(seconds, 2) if seconds else None
        except Exception as error:
            errors = [{"message": str(error)}]
        if len(batch) > 1 and any("task" not in item for item in errors):
            problems = []
            for one in batch:
                found, measured = self._check_settings(ctx, [one])
                problems += found
                for size, shots in measured.items():
                    renders.setdefault(size, {}).update(shots)
            return problems, renders
        return [f"{batch[item.get('task', 0)][0]}: {item['message']}" for item in errors], renders

    def _apply(self, production, ops, revision=None, by="deepy"):
        try:
            return production.apply(ops, by=by, base_revision=revision)
        except model.OperationError as error:
            raise ValueError(str(error) + (f" Current value: {error.current}" if error.current is not None else "")) from None

    # Actions ----------------------------------------------------------------------------------------------------------------------
    def production(self, ctx, production: str, section: Literal["summary", "brief", "cast", "style", "timeline", "shots", "shot", "edit", "log", "bin", "files", "full"] = "summary",
                   shot: str | None = None, sequence: str | None = None) -> dict[str, Any]:
        """Read a production. summary: brief, cast, style, outline and one line per shot (only the shots of `sequence` when named: long
        productions); timeline: where each shot of the cut starts and how long it lasts; shot: one shot with its takes and settings; files: its media (cast
        pictures and voices, chosen takes, tracks, exports, bin); full: everything, while it is short enough to read at once. It also reads the user's other productions (listed by `list_productions`) when the brief allows it: you never change
        them, and their files become yours with import_media."""
        target = self._production(ctx, production, read=True)
        if section == "files":
            own = self._own(ctx)
            return {"revision": target.doc["revision"], "files": self._files(ctx, target, own is None or own.id == target.id)}
        doc = self._usable(ctx, target, target.snapshot())
        if section == "summary":
            result = model.summary(doc, sequence)
            result["timeline"] = model.timeline(doc)
            result["references"] = doc["brief"]["references"]
            result["note"] = 'Prompts are cut short here: production(section="shot", shot=id) gives a shot\'s whole prompt and settings.'
            result["song"] = doc["brief"]["song"]
            result["format"] = (f"{model.resolution(doc['brief'])}: the brief's, what shots render at (fitted to their model's grid) unless one sets its own resolution" if not doc["brief"].get("draft_resolution")
                                else f"{model.resolution(doc['brief'])}: the brief's target, which every picture is made at. Video shots render at {doc['brief']['draft_resolution']} (a draft, fitted to their model's grid) until the user shoots them final")
            result["cast"] = [{**member, "images": full["images"], "voice": full["voice"], **({"voice_description": full["voice_description"]} if full["voice_description"] else {})}
                              for member, full in zip(result["cast"], doc["cast"])]
            catalog = self._catalog(doc)
            result["template_advice"] = templates.advice(doc, catalog, self.showrunner.host.settings()["any_shot_length"], self.showrunner.host.prompt_rules())
            if not doc["shots"] and (guides := self._guides(catalog, [result["template_advice"][key] for key in SHOT_ROLES if key in result["template_advice"]])):
                result["template_advice"]["prompt_guides"] = guides  # read before the first prompt is written; plan and add_shots repeat them
            warnings = templates.lint(doc, catalog) + templates.places_missing(doc, catalog) + model.cut_lines(doc) + model.singing_without_song(doc) + model.song_drift(doc) + model.song_words(doc) + self._length_warning(doc)
            if warnings:
                result["warnings"] = warnings
            if doc["audio"].get("analysis"):
                analysis = doc["audio"]["analysis"]
                result["song_analysis"] = {key: analysis.get(key) for key in ("duration_s", "bpm", "sections")} | {"bars": analysis.get("bars", [])[:200]}
            return result
        if section == "shot":  # read before a reshoot or a new prompt: the user's rules for its template go with it
            result = {"revision": doc["revision"], "shot": model.find(doc["shots"], shot, "shot")}
            rules = self.showrunner.host.prompt_rules()
            if rules and (entry := templates.shot_entry(doc, result["shot"], self._catalog(doc))) and (text := rules.get(name := templates.entry_name(entry))):
                result.update(user_prompt_rules={name: text}, user_prompt_rules_note=templates.PROMPT_RULES_NOTE)
            return result
        if section == "timeline":  # before assembling: what assemble would warn about, while it costs no version
            warnings = self._music_warnings(target, target.snapshot(), model.cut_length(doc)) + model.cut_lines(doc) + model.singing_without_song(doc) + model.song_drift(doc) + model.song_words(doc) + model.missing_ends(doc) + self._length_warning(doc)
            return {"revision": doc["revision"], "total_s": model.cut_length(doc), "timeline": model.timeline(doc), **({"warnings": warnings} if warnings else {})}
        if section == "full":
            size = len(json.dumps(doc, default=str))
            if size > FULL_DOCUMENT_CHARACTERS:
                return {**model.summary(doc, sequence), "note": f"The full document has {size:,} characters, too many to read at once: this is its summary. Read one shot with "
                        'section="shot", or the sections "brief", "cast", "style", "timeline", "edit", "log", "bin", "files".'}
            return doc
        if section == "log":
            return {"revision": doc["revision"], "log": doc["log"][-60:]}
        if section == "shots" and len(json.dumps(doc["shots"], default=str)) > FULL_DOCUMENT_CHARACTERS:
            return {"revision": doc["revision"], "shots": model.summary(doc, sequence)["shots"], "note": 'All the shots in full are too long to read at once: this is one line for each. production(section="shot", shot=id) reads one in full.'}
        return {"revision": doc["revision"], section: doc[section]}

    def list_productions(self, ctx, query: str | None = None) -> dict[str, Any]:
        """The user's productions, one line each: id, title, kind, length, stage, shots and the brief's first words (query: only those whose title
        or brief mention it). Read one with `production` (section files for its media) and copy its files into yours with import_media."""
        own = self._own(ctx)
        if own is not None and not own.snapshot()["brief"]["read_other_productions"]:
            raise ValueError("This production's brief does not let you see the user's other productions.")
        items = []
        for production_id in self.store.ids():
            if own is not None and production_id == own.id:
                continue
            doc = self.store.get(production_id).snapshot()
            about = " ".join(str(doc["brief"].get("text") or "").split())
            if query and query.casefold() not in f"{doc['title']} {about}".casefold():
                continue
            items.append({"id": production_id, "title": doc["title"], "kind": doc["brief"]["kind"], "duration_s": model.cut_length(doc),
                          "stage": doc["stage"], "shots": len(doc["shots"]), "about": about[:100]})
        return {"productions": items}

    def import_media(self, ctx, production: str, source: str, files: list[str]) -> dict[str, Any]:
        """Copy files of the production `source` into this production's media/imports: the paths its files section lists (cast pictures,
        voices, takes). Returns each copy's path, usable in settings, cast and your other tools. The source production is not changed."""
        target = self._production(ctx, production)
        origin = self._production(ctx, source, read=True)
        copied = []
        for name in files:
            path = origin.resolve(name)
            if path is None or not path.is_file() or not origin.owns(path):
                raise ValueError(f"{name} is not a file of the production {source}: its files section lists them.")
            copied.append({"file": name, "path": ctx.virtualize(str(target.resolve(target.adopt(path))))})
        return {"revision": target.doc["revision"], "copied": copied}

    @staticmethod
    def _files(ctx, target, own: bool) -> list[dict[str, Any]]:
        """A production's media, named as the caller uses them (its own) or as import_media takes them (another production's)."""
        doc = target.snapshot()
        name = (lambda path: ctx.virtualize(str(target.resolve(path)))) if own else (lambda path: path)
        files = []
        song = target.resolve(doc["brief"].get("song"))
        if song is not None and song.with_name(f"{song.stem}_post.wav").is_file():
            files.append({"file": name(f"media/{song.stem}_post.wav"), "what": "the song's vocal stem, the voice alone"})
        for member in doc["cast"]:
            pictures = dict.fromkeys([member.get("master"), *member.get("images", []), *(look.get("master") for look in member.get("looks", []))])
            files += [{"file": name(path), "what": f"picture of {member['name']}"} for path in pictures if path]
            if member.get("voice"):
                files.append({"file": name(member["voice"]), "what": f"voice of {member['name']}"})
        for shot in doc["shots"]:
            if take := model.chosen_take(shot):
                files.append({"file": name(take["path"]), "what": f"take of {shot['id']} ({shot['title']})", "kind": take.get("kind"), "duration_s": take.get("duration_s")})
        files += [{"file": name(item["path"]), "what": "sound track"} for item in doc["edit"]["audio_tracks"]]
        files += [{"file": name(doc["brief"]["song"]), "what": "song"}] if doc["brief"].get("song") else []
        files += [{"file": name(item["path"]), "what": f"bin: {item.get('name') or ''}"} for item in doc.get("bin", [])]
        files += [{"file": name(item["path"]), "what": "export", "duration_s": item.get("duration_s")} for item in doc["deliverables"][-2:]]
        return files

    @staticmethod
    def _source_decides_shots(target) -> None:
        """A remake's shots are its source's camera shots, with their ranges and painted frames: a plan written over them loses both, and with them what they render from."""
        if target.snapshot().get("remake"):
            raise ValueError("A remake's shots come from its source and stay as ingest_source made them: write each shot's prompt and cast with update_shot. Nothing was changed.")

    def plan(self, ctx, production: str, shots: list[dict[str, Any]] | None = None, cast: list[dict[str, Any]] | None = None, style: dict[str, Any] | None = None,
             outline: list[dict[str, Any]] | None = None, validate: bool = True) -> dict[str, Any]:
        """Write the plan. shots: [{id?, title, section, intent, camera, cast: [cast ids], dialogue: [{speaker, line, note}], duration_s, audio_slice: [start_s, end_s],
        generation: {template, settings}}] replaces the shots the user neither locked, edited nor shot (a shot given the id of a replaced one keeps it).
        cast: [{name, role, description, master, images, voice, voice_description}] adds members or updates the one of that name. outline: [{name, summary, duration_s}]
        the sequences in order (a shot's section names its sequence); long productions plan it first, then the shots of each sequence with add_shots.
        Every shot is validated as render will send it (the cast's pictures as references): fix the warnings with update_shot, which validates again;
        plan() without shots validates the whole document. renders in the result: what the shots will come out at, measured, {size: {shot: seconds}}."""
        target = self._production(ctx, production)
        if shots:
            self._only_the_fix(ctx, target, what="the plan of the shots (update_shot changes a shot you are fixing)")
            self._source_decides_shots(target)
        ops = [{"op": "outline", "outline": outline}] if outline is not None else []
        if style:
            ops.append({"op": "style", "fields": style})
        for member in cast or []:  # a member already in the cast under this name is updated, not added again
            if not str(member.get("name") or "").strip():
                raise ValueError("Every cast member of the plan needs a name.")
            fields = {key: member[key] for key in ("name", "role", "description", "voice_description") if key in member}
            fields.update({key: target.adopt(ctx.resolve(member[key])) for key in ("master", "voice") if member.get(key)})
            if member.get("images"):
                fields["images"] = [target.adopt(ctx.resolve(path)) for path in member["images"]]
            existing = next((item for item in target.doc["cast"] if item["name"].casefold() == member["name"].strip().casefold()), None)
            ops.append({"op": "update_cast", "cast": existing["id"], "fields": fields} if existing else {"op": "add_cast", "fields": fields})
        if shots is not None:
            ops.append({"op": "replace_plan", "shots": self._portable_shots(ctx, target, shots)})
        results = self._apply(target, ops) if ops else []
        response = {"revision": target.doc["revision"], "results": results}
        added = [shot_id for result in results for shot_id in result.get("shots", [])]
        warnings = self._warnings(target, added or (None if shots else [])) + templates.places_missing(target.snapshot(), self._catalog(target.snapshot()))  # a plan of the cast alone leaves the shots as they are
        if shots or not (cast or style or outline):
            warnings += self._script_warnings(ctx, target)
        if warnings:
            response["warnings"] = warnings
        if validate:  # the new shots, or the whole document when the call only re-checks it
            checked = added if shots is not None else [shot["id"] for shot in target.snapshot()["shots"]]
            problems, renders = self._validate(ctx, target, checked)
            response["validated"] = len(checked)
            if problems:  # problems become warnings next to the shot they concern, where they get fixed
                response["warnings"] = response.get("warnings", []) + problems
            if renders:
                response["renders"] = renders
        response.update(self._prompt_guides(target, added if shots is not None else [] if cast or style or outline else [shot["id"] for shot in target.snapshot()["shots"]]))
        if target.doc["stage"] in ("brief", "planning"):  # a plan() that only reads leaves the revision as it was
            target.apply([{"op": "stage", "stage": "review"}], by="system", record=False)
            response["revision"] = target.doc["revision"]
        return response

    def _prompt_guides(self, target, shot_ids: list[str]) -> dict[str, Any]:
        """The prompt format of each model these shots use, and who the reference pictures render gives each shot are: a plain description leaves
        out what such models are trained on (which reference to keep, the lip sync they are asked for)."""
        doc = target.snapshot()
        catalog = self._catalog(doc)
        names, pictures = [], {}
        for shot in doc["shots"]:
            if shot["id"] not in shot_ids or (entry := templates.shot_entry(doc, shot, catalog)) is None:
                continue
            names.append(templates.entry_name(entry))
            if entry["refs"] and entry["task"] not in templates.EDIT_TASKS and not shot["generation"].get("settings", {}).get("image_refs") and (shown := self._pictured_cast(doc, shot)):
                pictures[shot["id"]] = [member["name"] for member, _ in shown]  # as _generation sends them
        guides = self._guides(catalog, names)
        return {"prompt_guides": guides, **({"reference_pictures": pictures, "reference_pictures_note": PICTURE_ORDER} if pictures else {})} if guides else {}

    def _guides(self, catalog: list[dict[str, Any]], names: list[str]) -> dict[str, str]:
        """{template: its model's prompt format} for these templates, those whose model has one."""
        entries = {name: entry for name in dict.fromkeys(names) if (entry := templates.resolve(catalog, name)) is not None and entry["model_type"]}
        return {name: guide for name, entry in entries.items() if (guide := self.showrunner.host.prompt_guide(entry["model_type"]))}

    def update_shot(self, ctx, production: str, shot: str, fields: dict[str, Any], revision: int | None = None) -> dict[str, Any]:
        """fields: any of title, section, intent, camera, cast, dialogue, duration_s, audio_slice, generation (settings are merged; null removes one), status, included, chosen_take,
        trim [start_s, end_s], speed (0.25 to 4, how fast the take plays in the edit; trim stays in seconds of the take), sound (false: the take plays silent),
        transition {type: cut|crossfade|fade_black|fade_white, duration_s}, notes. revision: the one you last read, which guards
        against the user's edits since then, never against your own or the system's writes: one revision serves a batch of calls. Changed settings or cast are validated again."""
        target = self._production(ctx, production)
        self._only_the_fix(ctx, target, [shot])
        if isinstance(fields.get("generation"), dict) and isinstance(fields["generation"].get("settings"), dict):
            fields = {**fields, "generation": {**fields["generation"], "settings": self._portable(ctx, target, fields["generation"]["settings"])}}
        result = self._apply(target, [{"op": "update_shot", "shot": shot, "fields": fields, "revision": revision}], revision)
        problems, renders = self._validate(ctx, target, [shot]) if {"generation", "cast"} & set(fields) else ([], {})
        warnings = self._warnings(target, [shot]) + problems
        return {"revision": target.doc["revision"], **result[0], **({"warnings": warnings} if warnings else {}), **({"renders": renders} if renders else {})}

    def add_shots(self, ctx, production: str, shots: list[dict[str, Any]], position: int | None = None) -> dict[str, Any]:
        """Insert shots (same fields as plan) at a zero-based position, or at the end."""
        target = self._production(ctx, production)
        self._source_decides_shots(target)
        result = self._apply(target, [{"op": "add_shots", "shots": self._portable_shots(ctx, target, shots), "position": position}])
        fixing = self._fixing(ctx, target)
        if fixing is not None:  # a shot inserted in a fix is part of it; the shots around it keep their takes
            target.apply([{"op": "director", "fields": {"shots": [*fixing, *result[0]["shots"]]}}], by="system", record=False)
        problems, renders = self._validate(ctx, target, result[0]["shots"])
        warnings = self._warnings(target, result[0]["shots"]) + problems + ([] if target.snapshot()["deliverables"] else self._script_warnings(ctx, target))  # a shot added to a finished film is about that shot
        return {"revision": target.doc["revision"], **result[0], **({"warnings": warnings} if warnings else {}), **({"renders": renders} if renders else {}), **self._prompt_guides(target, result[0]["shots"])}

    def remove_shot(self, ctx, production: str, shot: str, revision: int | None = None) -> dict[str, Any]:
        target = self._production(ctx, production)
        self._source_decides_shots(target)
        self._only_the_fix(ctx, target, [shot])
        result = self._apply(target, [{"op": "remove_shot", "shot": shot, "revision": revision}], revision)
        return {"revision": target.doc["revision"], "removed": shot, **result[0]}

    def move_shots(self, ctx, production: str, shots: list[str], before: str | None = None) -> dict[str, Any]:
        target = self._production(ctx, production)
        self._source_decides_shots(target)
        self._only_the_fix(ctx, target, shots)
        self._apply(target, [{"op": "move_shots", "shots": shots, "before": before}])
        return {"revision": target.doc["revision"], "order": [item["id"] for item in target.doc["shots"]]}

    def cast(self, ctx, production: str, name: str | None = None, cast_id: str | None = None, role: str | None = None, description: str | None = None,
             images: list[str] | None = None, master: str | None = None, voice: str | None = None, voice_description: str | None = None,
             remove: bool = False, looks: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Add a cast member or update one (cast_id, or the member of that name); remove=true removes the member cast_id (and takes it out of every
        shot). images, master and voice are media ids or paths (voice="" removes the voice sample: `speak` makes a new one from the description).
        voice_description: how the member sounds (age, timbre, accent, pace), from which `speak` makes the member's voice sample once; the
        narrator is a member with role "narrator" and no pictures. looks: [{name, master, sections: [sequence names], shots: [shot ids], description}] the member's
        costumes, ages or states, replacing the current ones: a shot in one of a look's sequences gets that look's master as reference."""
        target = self._production(ctx, production)
        if remove:
            self._apply(target, [{"op": "remove_cast", "cast": cast_id}])
            return {"revision": target.doc["revision"], "removed": cast_id}
        fields = {key: value for key, value in {"name": name, "role": role, "description": description, "images": images, "master": master, "voice": voice,
                                                 "voice_description": voice_description}.items() if value is not None}
        if looks is not None:
            fields["looks"] = [{"name": str(look.get("name") or ""), "master": target.adopt(ctx.resolve(look["master"])) if look.get("master") else None,
                                "sections": [str(section) for section in look.get("sections") or []], "shots": [str(shot) for shot in look.get("shots") or []],
                                "description": str(look.get("description") or "")} for look in looks]
        for key in ("images",):
            if key in fields:
                fields[key] = [target.adopt(path) for path in fields[key]]
        for key in ("master", "voice"):
            if key in fields:
                fields[key] = target.adopt(ctx.resolve(fields[key])) if fields[key] else None
        if cast_id is None and name:
            cast_id = next((member["id"] for member in target.doc["cast"] if member["name"].casefold() == name.strip().casefold()), None)
        if cast_id is None:
            result = self._apply(target, [{"op": "add_cast", "fields": fields}])
            cast_id = result[0]["cast"]
        else:
            self._apply(target, [{"op": "update_cast", "cast": cast_id, "fields": fields}])
        return {"revision": target.doc["revision"], "cast": cast_id}

    def render(self, ctx, production: str, shots: list[str], new_seed: bool | int = False, final: bool = False) -> dict[str, Any]:
        """Generate one take per listed shot, in order, from the shot's generation (template + settings), and record it. Returns each take's media id
        to inspect; record a verdict afterwards. new_seed: true draws a new seed instead of the settings' one; a number is the seed to use.
        final: at the brief's target resolution when it has a draft one, with the seed of the shot's take; only when the user asks for it."""
        target = self._production(ctx, production)
        self._only_the_fix(ctx, target, shots)
        if target.doc["stage"] in ("planning", "review"):  # a plan the user chose not to review goes on to its shoot
            target.apply([{"op": "stage", "stage": "shooting"}], by="system", record=False)
        renders, covered = [], {}  # covered: shots another shot's render already gave a take in this call
        for index, shot_id in enumerate(shots):
            if ctx.cancelled():
                break
            if shot_id in covered:
                renders.append({"shot": shot_id, "rendered_with": covered[shot_id], "note": f"rendered together with {covered[shot_id]}: its take is in that render's result"})
                continue
            if self.showrunner.director.pausing(production):
                renders.append({"paused": True, "not_rendered": shots[index:], "note": "the user paused you: stop here, they will resume the task later"})
                break
            if self.showrunner.director.note_waiting(production) and index:
                renders.append({"stopped": True, "not_rendered": shots[index:], "note": "the user just wrote to you: read it, then decide what to render"})
                break
            render = self.render_shot(ctx, target, shot_id, new_seed, index, len(shots), final=final, requested=shots)
            covered.update({item["shot"]: shot_id for item in render.get("rendered_with") or []})
            if render.get("paused"):
                renders.append({"paused": True, "not_rendered": shots[index:], "note": "the user paused you: stop here, they will resume the task later"})
                break
            if render.get("error"):  # what the user's Render now writes too: the Board shows it under the shot
                target.apply([{"op": "log", "text": f"Render of {shot_id} failed: {render['error']}"}], by="system", record=False)
            renders.append(render)
        ctx.progress(done=len(renders), total=len(shots), text="Rendered", force=True)
        return {"revision": target.doc["revision"], "renders": renders}

    def render_shot(self, ctx, target, shot_id: str, new_seed: bool | int = False, index: int = 0, total: int = 1, by: str = "deepy", final: bool = False,
                    requested: list[str] | tuple[str, ...] = ()) -> dict[str, Any]:
        """One take of a shot, from its generation: its lines spoken first when they are not yet. What happened, for its report. by "user": the user's Render now, which no showrunner directs.
        final: at the target resolution, with the seed of the take it follows. A remake's shot rendered with its neighbours (remake_groups) gives them a take too: as their
        chosen one when they are in `requested` or have none, else beside the one they use."""
        shot = model.find(target.snapshot()["shots"], shot_id, "shot")
        if shot["locked"] and by == "deepy":
            return {"shot": shot_id, "error": "locked by the user"}
        if shot["generation"].get("template") == templates.DRAWN:
            return {"shot": shot_id, "error": "a drawn shot: draw it with draw_svg and record it with record_take (or record_take card=...)"}
        doc = target.snapshot()
        catalog = self._catalog(doc)
        entry = templates.resolve(catalog, shot["generation"].get("template", "gen_video"))
        if shot.get("source_range") and ((entry or {}).get("video_edit") or {}).get("input") == "painted_frames" and not shot["cast"] and all(item["path"] == item["source"] for item in shot.get("painted_frames") or []):
            return self._keep_source(ctx, target, shot, doc, by)
        if shot.get("source_range") and (entry or {}).get("video_edit") and len(group := remake_groups.group_of(self._render_groups(doc, entry, requested or [shot_id]), shot_id)) > 1:
            return self._render_group(ctx, target, group, shot_id, new_seed, index, total, by, final, requested, entry)
        if templates.model_speaks(doc, shot, entry) and (shot.get("trim") or [0.0, None])[1] is not None and shot.get("updated_by") != "user":  # a trim made for a recorded line would cut what the model says
            target.apply([{"op": "update_shot", "shot": shot_id, "fields": {"trim": [0.0, None]}}], by="system", record=False)
            shot = model.find(target.snapshot()["shots"], shot_id, "shot")
        recorded = model.spoken_lines(doc, shot)
        missing = [number for number, line in enumerate(shot.get("dialogue") or []) if line in self._driving_lines(doc, shot, entry, spoken=False) and line not in recorded]
        if missing and not shot.get("audio_slice") and not (shot["generation"].get("settings") or {}).get("audio_guide"):
            # a line not spoken yet: the model would invent its words, in a voice of its own; spoken in the speaker's voice first, it drives the shot
            said = []
            if self._speak_lines(ctx, target, target.id, shot_id, missing, False, catalog, templates.effective(doc, catalog), [], said):
                return {"paused": True}
            if errors := [f"line {item['line']}: {item['error']}" for item in said if item.get("error")]:
                return {"shot": shot_id, "error": "its lines could not be spoken (" + "; ".join(errors) + "): fix the speakers' voices, then render it"}
            shot = model.find(target.snapshot()["shots"], shot_id, "shot")
        settings, base_loras = self._generation_with_loras(ctx, target, shot, final)
        if shot.get("source_range") and (entry or {}).get("video_edit") and settings.get("multi_prompts_gen_type") != "PW":  # one window: its prompt in the template's window format, as a group's
            try:
                settings["prompt"] = remake_groups.joined_prompt([shot], doc["remake"]["fps"], swap=entry["video_edit"].get("kind") == "swap")
            except ValueError as error:
                return {"shot": shot_id, "error": str(error)}
        tier = "draft" if doc["brief"].get("draft_resolution") and not final else "final"
        follows = model.chosen_take(shot) if final and not new_seed else None
        if follows is not None and follows.get("kind", "video") == "video" and (seed := media.generation_seed(target.resolve(follows["path"]))) is not None:
            settings["seed"] = seed  # the take you approved at its draft size
        song = target.resolve(target.snapshot()["brief"].get("song"))
        if shot.get("audio_slice") and song is not None and settings.get("audio_guide") == ctx.virtualize(str(song)):
            start_s, end_s = shot["audio_slice"]  # the song drives the shot: its own slice, not the song from the start
            settings["audio_guide"] = ctx.virtualize(str(media.audio_slice(song, float(start_s), float(end_s), target.folder / "media" / "slices")))
        elif settings.get("audio_guide") and shot.get("duration_s"):  # a line shorter than its shot: silence after it, or the model invents more speech
            guide = ctx.resolve(settings["audio_guide"])
            spoken = media.probe(guide)["duration_s"] or 0.0
            start, end = (list(shot.get("trim") or [0.0, None]) + [None, None])[:2]
            if end is not None and end < spoken + 0.1:  # a trim from an older, shorter line (a new voice, a rewritten line) would cut the end of this one
                target.apply([{"op": "update_shot", "shot": shot_id, "fields": {"trim": [start, round(float(start or 0.0) + spoken + 0.35, 2)]}}], by="system", record=False)
                shot = model.find(target.snapshot()["shots"], shot_id, "shot")
                end = shot["trim"][1]
            if spoken < float(shot["duration_s"]) - 0.25:
                settings["audio_guide"] = ctx.virtualize(str(media.padded_audio(guide, float(shot["duration_s"]), target.folder / "media" / "slices")))
                if end is None and shot.get("updated_by") != "user":  # the model fills the padding with invented words: the edit ends after the line
                    target.apply([{"op": "update_shot", "shot": shot_id, "fields": {"trim": [0.0, round(spoken + 0.35, 2)]}}], by="system", record=False)
            elif spoken > float(shot["duration_s"]):  # the edit ends a take at its planned length: the shot must hold its whole line
                target.apply([{"op": "update_shot", "shot": shot_id, "fields": {"duration_s": round(spoken + 0.1, 2)}}], by="system", record=False)
                shot = model.find(target.snapshot()["shots"], shot_id, "shot")
            settings.update(templates.planned_length(target.snapshot(), shot, entry))
        if not str(settings.get("prompt") or "").strip():
            return {"shot": shot_id, "error": "the shot's settings have no prompt"}
        if new_seed:
            settings["seed"] = -1 if new_seed is True else int(new_seed)
        ctx.progress(done=index, total=total, text=f"Rendering {shot_id} ({index + 1}/{total})", force=True)
        target.apply([{"op": "update_shot", "shot": shot_id, "fields": {"status": "rendering"}}], by=by, record=False)
        started = time.time()
        try:
            response = ctx.call("wangp_generate", {"settings": settings})
            generated = time.time() - started
        except Exception as error:
            target.apply([{"op": "update_shot", "shot": shot_id, "fields": {"status": "failed"}}], by=by, record=False)
            return {"shot": shot_id, "error": str(error)}
        entries = media_entries(response)
        if not entries:
            target.apply([{"op": "update_shot", "shot": shot_id, "fields": {"status": "failed"}}], by=by, record=False)
            result = (response.get("result") or {}) if isinstance(response, dict) else {}
            reasons = [str(item["message"] if isinstance(item, dict) else item) for item in [response.get("error") if isinstance(response, dict) else None, *(result.get("errors") or [])] if item]  # WanGP's errors are {message, task_index, stage...}
            return {"shot": shot_id, "error": "the generation returned no media" + (f": {'; '.join(reasons)}" if reasons else "")}
        entry = entries[0]
        try:
            path = ctx.resolve(entry["path"])  # tool results name files as the caller does
            used = entry.get("settings") if isinstance(entry.get("settings"), dict) else settings
            take = self.showrunner.add_take(target.id, shot_id, path, by=by, settings=self._real_paths(ctx, used), media_id=entry.get("media_id") if by == "deepy" else None, gpu_s=generated, base_loras=base_loras, tier=tier)
        except Exception as error:  # the media exists: report it with the shot and render the next shots
            target.apply([{"op": "update_shot", "shot": shot_id, "fields": {"status": "failed"}}], by=by, record=False)
            return {"shot": shot_id, "media_id": entry.get("media_id"), "error": f"the take could not be recorded ({error}): record it with record_take"}
        target.apply([{"op": "budget", "gpu_seconds": time.time() - started}], by="system", record=False)
        heard = self.showrunner.hear_take(target, shot_id, take["take"], listening=lambda: ctx.progress(done=index, total=total, text=f"Listening to {shot_id} ({index + 1}/{total})", force=True))
        measured = {key: value for key, value in self.check_take(ctx, target.id, shot_id, take["take"]).items() if key in ("ok", "cut_faults", "edge_bands")} if shot.get("source_range") else None  # a remake's take against its source
        return {"shot": shot_id, "take": take["take"], "media_id": entry.get("media_id"), "path": ctx.virtualize(path), "duration_s": take.get("duration_s"), "size": f"{take.get('width')}x{take.get('height')}", **({"heard": heard} if heard else {}), **({"check": measured} if measured else {}),
                        **{key: response[key] for key in ("cut_times_seconds", "canvas") if isinstance(response, dict) and key in response}}

    def _keep_source(self, ctx, target, shot: dict[str, Any], doc: dict[str, Any], by: str) -> dict[str, Any]:
        """A remake's shot nothing changes in: its take is its part of the source. The model could only make it worse, at the cost of a render."""
        note = "Nothing changes in this shot: its take is its part of the source, without the model."
        if (kept := model.chosen_take(shot)) and kept.get("kept_source"):  # the edit reads it from the source at whatever size it is made
            target.apply([{"op": "update_shot", "shot": shot["id"], "fields": {"status": "rendered"}}], by="system", record=False)
            return {"shot": shot["id"], "take": kept["id"], "kept_source": True, "note": note}
        start, stop = shot["source_range"]
        clip = media.video_window(target.resolve(doc["remake"]["path"]), start, stop - start, doc["remake"]["fps"], target.folder / "media" / "source", size=self.showrunner.shot_size(target, doc, shot))  # what the page plays
        take = self.showrunner.add_take(target.id, shot["id"], clip, by=by, note=note, gpu_s=0.0, kept_source=True)
        target.apply([{"op": "verdict", "shot": shot["id"], "ok": True, "note": note}], by="system", record=False)
        return {"shot": shot["id"], "take": take["take"], "kept_source": True, "note": note, "path": ctx.virtualize(clip), "duration_s": take.get("duration_s"), "size": f"{take.get('width')}x{take.get('height')}"}

    @staticmethod
    def _render_groups(doc: dict[str, Any], entry: dict[str, Any], wanted: list[str] | tuple[str, ...] | None = None) -> list[list[dict[str, Any]]]:
        """The renders of a remake's shots asked for (remake_groups.plan): the brief's lengths in frames, up to the model's largest window (Auto: all of it)."""
        fps, brief = doc["remake"]["fps"], doc["brief"]
        minimum = round(float(brief["render_min_s"] or 0.0) * fps) if brief["group_renders"] else 0  # Group rendering off on the Board: every shot alone
        largest = entry["window_max"] or entry["longest_frames"]
        return remake_groups.plan(doc, minimum, min(round(float(brief["render_max_s"]) * fps), largest) if brief["render_max_s"] else largest, wanted)

    def _render_group(self, ctx, target, group: list[dict[str, Any]], shot_id: str, new_seed: bool | int, index: int, total: int, by: str, final: bool, requested,
                      entry: dict[str, Any]) -> dict[str, Any]:
        """A remake's shot and its neighbours in one request (remake_groups): one view over their part of the source, every painted frame pinned at its own frame, their
        prompts joined. Each changed shot gets its own part of the render as a take; the shots left as they are only give the window its length."""
        doc = target.snapshot()
        fps, changed = doc["remake"]["fps"], [shot for shot in group if not remake_groups.kept(shot)]
        others = [shot["id"] for shot in changed if shot["id"] != shot_id]
        together = ", ".join(shot["id"] for shot in group if shot["id"] != shot_id)
        if missing := [shot["id"] for shot in changed if not shot.get("painted_frames")]:
            return {"shot": shot_id, "error": f"{shot_id} renders together with {together}: paint a frame of {', '.join(missing)} first"}
        try:
            prompt = remake_groups.joined_prompt(group, fps, swap=entry["video_edit"].get("kind") == "swap")
        except ValueError as error:
            return {"shot": shot_id, "error": f"{shot_id} renders together with {together}: {error}"}
        lead = model.find(doc["shots"], shot_id, "shot")
        settings, base_loras = self._generation_with_loras(ctx, target, lead, final)
        start, stop = group[0]["source_range"][0], group[-1]["source_range"][1]
        source = target.resolve(doc["remake"]["path"])
        view = write_view(target.folder / "media" / "views" / f"{source.stem}-{start}-{stop - start}{source.suffix}", source, start, stop - 1)
        painted = [item for shot in group for item in shot.get("painted_frames") or []]
        pinned = any("F" in str((shot["generation"].get("settings") or {}).get("video_prompt_type") or "") for shot in changed)
        settings.update(video_guide=ctx.virtualize(view), video_length=stop - start, prompt=prompt, image_refs=[ctx.virtualize(str(target.resolve(item["path"]))) for item in painted],
                        video_prompt_type="FIVU" if pinned else "IVU", frames_positions=" ".join(str(item["frame"] - start + 1) for item in painted) if pinned else "")
        first, step, _ = entry["take_frames"]  # one window holds the whole render: WanGP would otherwise cut it into sliding windows, a seam inside a shot
        settings["sliding_window_size"] = max(int(settings.get("sliding_window_size") or entry["longest_frames"] or 0), first + -(-max(0, stop - start - first) // step) * step)
        follows = model.chosen_take(lead) if final and not new_seed else None
        if follows is not None and (seed := (follows.get("settings") or {}).get("seed")) not in (None, -1):
            settings["seed"] = seed  # the take you approved at its draft size: its part was cut from a render with this seed
        if new_seed:
            settings["seed"] = -1 if new_seed is True else int(new_seed)
        tier = "draft" if doc["brief"].get("draft_resolution") and not final else "final"
        ctx.progress(done=index, total=total, text=f"Rendering {shot_id} with {together} ({index + 1}/{total})", force=True)
        before = {shot["id"]: shot["status"] for shot in changed}  # every shot of the render shows as rendering, the ones you did not ask for too
        target.apply([{"op": "update_shot", "shot": shot["id"], "fields": {"status": "rendering", "rendering_with": None if shot["id"] == shot_id else shot_id}} for shot in changed], by=by, record=False)
        started = time.time()
        try:
            response = ctx.call("wangp_generate", {"settings": settings})
        except Exception as error:
            response = {"error": str(error)}
        generated, entries = time.time() - started, media_entries(response)
        if not entries:
            target.apply([{"op": "update_shot", "shot": shot["id"], "fields": {"status": "failed" if not shot["takes"] or shot["id"] == shot_id else before[shot["id"]], "rendering_with": None}} for shot in changed], by=by, record=False)
            result = (response.get("result") or {}) if isinstance(response, dict) else {}
            reasons = [str(item["message"] if isinstance(item, dict) else item) for item in [response.get("error") if isinstance(response, dict) else None, *(result.get("errors") or [])] if item]
            return {"shot": shot_id, "error": f"the generation of {shot_id} with {together} returned no media" + (f": {'; '.join(reasons)}" if reasons else "")}
        path = ctx.resolve(entries[0]["path"])
        used = entries[0].get("settings") if isinstance(entries[0].get("settings"), dict) else settings
        seed = media.generation_seed(path)
        used = self._real_paths(ctx, {**used, **({"seed": seed} if seed is not None else {})})
        target.apply([{"op": "budget", "gpu_seconds": generated}], by="system", record=False)
        takes = {}
        flashes = {item["source_range"][0] - start: media.cut_flash(path, item["source_range"][0] - start, fps) for item in group[1:]}  # the render's cuts a frame or two off: their frames are held
        for shot in changed:  # each shot's own frames, with their sound, as a take of its own: the edit, the checks and the players read them as any take
            own = shot["source_range"][1] - shot["source_range"][0]
            first = shot["source_range"][0] - start
            hold = (flashes.get(first, (0, 0))[1], flashes.get(first + own, (0, 0))[0])
            part = media.video_window(path, first, own, fps, target.folder / "media" / "parts", hold=hold)
            choose = shot["id"] in (shot_id, *requested) or not shot["takes"]
            mates = ", ".join(item["id"] for item in group if item["id"] != shot["id"])
            added = self.showrunner.add_take(target.id, shot["id"], part, by=by, settings=used, note=f"Rendered together with {mates}", gpu_s=generated * own / (stop - start),
                                             base_loras=base_loras, tier=tier, choose=choose)
            target.apply([{"op": "update_shot", "shot": shot["id"], "fields": {"rendering_with": None, **({} if choose else {"status": before[shot["id"]]})}}], by="system", record=False)  # a take beside the one it uses: as it was
            checked = {key: value for key, value in self.check_take(ctx, target.id, shot["id"], added["take"]).items() if key in ("ok", "cut_faults", "edge_bands")}
            takes[shot["id"]] = {"shot": shot["id"], "take": added["take"], "chosen": choose, "path": ctx.virtualize(part), "check": checked}
        extra = [item for shot, item in takes.items() if shot != shot_id]
        if extra:
            target.apply([{"op": "log", "text": f"{shot_id} rendered together with {together}: " + "; ".join(f"{item['shot']} got take {item['take']}" + ("" if item["chosen"] else ", beside the take it uses")
                                                                                                         for item in extra) + ".", "shot": shot_id}], by="system", record=False)
        mine = takes[shot_id]
        return {**mine, "media_id": entries[0].get("media_id"), "duration_s": round((lead["source_range"][1] - lead["source_range"][0]) / fps, 3), "rendered_with": extra,
                **({"note": f"{shot_id} rendered in one request with {together}, which gives the model a window long enough to follow speech and singing; " + ", ".join(item["shot"] for item in extra)
                    + " got a take too: judge them as well."} if extra else {"note": f"{shot_id} rendered in one request with {together} (left as they are), for a longer window."})}

    def speak(self, ctx, production: str, shot: str | list[str], lines: list[int] | None = None, new_take: bool = False) -> dict[str, Any]:
        """Speak the dialogue lines of a shot, or of several shots in one call (shot: a list of ids), and record each line's audio on it;
        lines: zero-based indexes, with one shot. Every speaker keeps one voice for the whole production: the cast member's voice sample, made
        once from its voice_description when it has none (NARRATOR is the member whose role is narrator). A line already spoken in its
        speaker's voice with the same words and note is returned as it is, unless new_take. Give a lip-synced line as the shot's audio_guide;
        lay narration over the edit with edit(tracks) at its shot's start."""
        target = self._production(ctx, production)
        self._only_the_fix(ctx, target, [shot] if isinstance(shot, str) else shot)
        shots = [shot] if isinstance(shot, str) else list(shot)
        if lines is not None and len(shots) != 1:
            raise ValueError("lines picks lines of one shot: give a single shot with it.")
        doc = target.snapshot()
        catalog = self._catalog(doc)
        picks = templates.effective(doc, catalog)
        voices, spoken = [], []
        for shot_id in shots:
            item = model.find(doc["shots"], shot_id, "shot")
            count = len(item["dialogue"])
            indexes = list(range(count)) if lines is None else [int(index) for index in lines]
            own_voice = templates.model_speaks(doc, item, templates.resolve(catalog, item["generation"].get("template")))
            if own_voice or item["locked"] or not count or any(not 0 <= index < count for index in indexes):
                problem = "its video model says its lines, as the user chose in the brief: render writes them into its prompt, and nothing is recorded" if own_voice else "locked by the user: leave it as it is" if item["locked"] else f"{count} dialogue lines" + (f" (0 to {count - 1})" if count else ": write them with update_shot first")
                if len(shots) == 1:
                    raise ValueError(f"Shot {shot_id}: {problem}.")
                spoken.append({"shot": shot_id, "error": problem})
                continue
            if self._speak_lines(ctx, target, production, shot_id, indexes, new_take, catalog, picks, voices, spoken):
                break
        doc = target.snapshot()
        heard = {item.get("speaker") for item in spoken}
        off = [f"{member['name']}'s voice measures {hz:.0f} Hz, which reads as a {'man' if kind == 'woman' else 'woman'}, and the card says a {kind}: give a clearer voice_description or another sample, as every line follows it."
               for member in doc["cast"] if member["name"] in heard and member.get("voice") and (kind := model.voice_kind(member.get("voice_description") or member.get("description"))) and model.voice_off(kind, hz := self.showrunner.host.voice_pitch(target.resolve(member["voice"])))]
        return {"revision": target.doc["revision"], "lines": spoken, **({"voices_made": voices} if voices else {}), **({"warnings": off} if off else {})}

    def _speak_lines(self, ctx, target, production, shot, indexes, new_take, catalog, picks, voices, spoken) -> bool:
        """Speak one shot's lines into `spoken`; True when the user paused the director (the rest waits)."""
        for position, index in enumerate(indexes):
            if ctx.cancelled() or self.showrunner.director.pausing(production):
                spoken.append({"shot": shot, "paused": True, "not_spoken": indexes[position:]})
                return True
            doc = target.snapshot()
            line = model.find(doc["shots"], shot, "shot")["dialogue"][index]
            member = model.speaker(doc, line["speaker"])
            if member is None:
                spoken.append({"shot": shot, "line": index, "speaker": line["speaker"], "error": "no cast member speaks it: add the speaker to the cast (the narrator with role "
                                                                                                 "narrator) with a voice_description"})
                continue
            try:
                voice, made = self._voice(ctx, target, member, catalog, picks)
            except (ValueError, RuntimeError) as error:
                spoken.append({"shot": shot, "line": index, "speaker": member["name"], "error": str(error)})
                continue
            if made:
                voices.append({"cast": member["id"], "name": member["name"], "voice": ctx.virtualize(str(target.resolve(voice)))})
            if line.get("audio") and line.get("voice") == voice and not new_take and target.resolve(line["audio"]).is_file():
                spoken.append({"shot": shot, "line": index, "speaker": member["name"], "path": ctx.virtualize(str(target.resolve(line["audio"]))),
                               "duration_s": line.get("duration_s"), "kept": True})
                continue
            entry = templates.resolve(catalog, picks.get("cloned_voices"))
            if entry is None:
                raise ValueError("No template speaks from a voice sample: add one to Deepy's Speech From Sample templates.")
            member = model.speaker(target.snapshot(), line["speaker"])
            settings = templates.speech_settings(entry, line["line"], note=line["note"], sample=ctx.virtualize(str(target.resolve(voice))), sample_text=member.get("voice_text") or "")
            ctx.progress(done=len(spoken), total=None, text=f"Speaking {shot} line {index + 1} ({member['name']})", force=True)
            pitch = self.showrunner.host.voice_pitch
            sample_pitch = pitch(target.resolve(voice))
            try:
                path, media_id = self._generate_audio(ctx, settings)
                if model.line_off_voice(sample_pitch, pitch(path)):  # the delivery note can carry a line away from the voice: once more without it, and the closer one stays
                    plain = templates.speech_settings(entry, line["line"], sample=settings["audio_guide"], sample_text=member.get("voice_text") or "")
                    other, other_id = self._generate_audio(ctx, plain)
                    if abs(math.log(pitch(other) / sample_pitch)) < abs(math.log(pitch(path) / sample_pitch)):
                        path, media_id = other, other_id
            except RuntimeError as error:
                spoken.append({"shot": shot, "line": index, "speaker": member["name"], "error": str(error)})
                continue
            line_pitch = pitch(path)
            drifted = {"warning": f"This line measures {line_pitch:.0f} Hz and {member['name']}'s voice {sample_pitch:.0f} Hz: it may not sound like them."} if model.line_off_voice(sample_pitch, line_pitch) else {}
            relative = target.adopt(path)
            duration = media.probe(target.resolve(relative))["duration_s"]
            target.apply([{"op": "line_audio", "shot": shot, "spoken": model.line_key(line), "audio": relative, "voice": voice, "duration_s": duration}], by="deepy")
            spoken.append({"shot": shot, "line": index, "speaker": member["name"], "text": line["line"], "media_id": media_id, "path": ctx.virtualize(str(path)), "duration_s": duration, **drifted})
        return False

    def _voice(self, ctx, target, member: dict[str, Any], catalog, picks) -> tuple[str, bool]:
        """The member's voice sample (a production path) and whether it was just made: once, from the member's voice description."""
        if member.get("voice"):
            return member["voice"], False
        if not str(member.get("voice_description") or "").strip():
            raise ValueError(f"{member['name'] or member['id']} has no voice: give them a voice_description with cast (or a voice sample).")
        entry = templates.resolve(catalog, picks.get("voices"))
        if entry is None:
            raise ValueError("No template makes a voice from a description: add one to Deepy's Speech From Description templates.")
        kind = model.voice_kind(member["voice_description"])
        for _ in range(3):  # a voice that reads as the other sex than its card says is made again
            path, _media = self._generate_audio(ctx, templates.speech_settings(entry, CASTING_LINE, description=member["voice_description"]))
            if not model.voice_off(kind, self.showrunner.host.voice_pitch(path)):
                break
        relative = target.adopt(path)
        self._apply(target, [{"op": "update_cast", "cast": member["id"], "fields": {"voice": relative, "voice_text": CASTING_LINE}}])
        return relative, True

    @staticmethod
    def _generate_audio(ctx, settings: dict[str, Any]) -> tuple[str, str | None]:
        response = ctx.call("wangp_generate", {"settings": settings})
        entries = media_entries(response)
        if not entries:
            details = {key: response.get(key) for key in ("status", "error", "errors") if isinstance(response, dict) and key in response}
            raise RuntimeError(f"the speech generation returned no audio {details}")
        return str(ctx.resolve(entries[0]["path"])), entries[0].get("media_id")

    @staticmethod
    def _pictured_cast(doc: dict[str, Any], shot: dict[str, Any]) -> list[tuple[dict[str, Any], str]]:
        """Each cast member the shot shows with one picture of them: the look worn in the shot's sequence, else the master, else the first picture."""
        members = {member["id"]: member for member in doc["cast"]}
        def picture(member):
            look = model.look_for(member, shot)
            return look["master"] if look else member.get("master") or (member.get("images") or [None])[0]
        return [(members[cast_id], path) for cast_id in shot.get("cast") or [] if cast_id in members and (path := picture(members[cast_id]))]

    @classmethod
    def _cast_pictures(cls, doc: dict[str, Any], shot: dict[str, Any]) -> list[str]:
        return [path for _, path in cls._pictured_cast(doc, shot)]

    @staticmethod
    def _portable(ctx, target, settings: dict[str, Any]) -> dict[str, Any]:
        """Media in a shot's settings as the document keeps them: production-relative paths, not Gallery media ids or @aliases, which belong to
        one workspace (a duplicated production, the user's reruns and other agents read the same document)."""
        def portable(value):
            if isinstance(value, list):
                return [portable(item) for item in value]
            if isinstance(value, str) and _REFERENCE.match(value.strip()):
                return target.adopt(ctx.resolve(value.strip()))
            return value
        return {key: portable(value) if key in MEDIA_SETTINGS else value for key, value in settings.items()}

    def _portable_shots(self, ctx, target, shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{**shot, "generation": {**shot["generation"], "settings": self._portable(ctx, target, shot["generation"].get("settings") or {})}}
                if isinstance(shot.get("generation"), dict) else shot for shot in shots]

    @staticmethod
    def _resolve_settings(ctx, target, settings: dict[str, Any]) -> dict[str, Any]:
        """Media paths in settings become the caller's names for them (tools refuse absolute paths without Read Everywhere). Only the media
        settings: a prompt is text, and a long one is no valid file name (Errno 36)."""
        def resolve(value):
            if isinstance(value, str) and value and ":" not in value[:12].replace(":\\", "").replace(":/", "") and target.resolve(value).is_file():
                return ctx.virtualize(str(target.resolve(value)))
            if isinstance(value, list):
                return [resolve(item) for item in value]
            return value
        return {key: resolve(value) if key in MEDIA_SETTINGS else value for key, value in settings.items()}

    @staticmethod
    def _real_paths(ctx, settings: dict[str, Any]) -> dict[str, Any]:
        """A take's settings with the caller's @alias paths as real paths: the user's reruns run them without Deepy."""
        def real(value):
            if isinstance(value, str) and value.startswith("@"):
                return str(ctx.resolve(value))
            return [real(item) for item in value] if isinstance(value, list) else value
        return {key: real(value) for key, value in settings.items()}

    def record_take(self, ctx, production: str, shot: str, media: str | None = None, note: str = "", card: dict[str, Any] | None = None) -> dict[str, Any]:
        """Record a media file (media id or path) as a new take of a shot, or a full-frame text card drawn for it:
        card {text, subtitle, style: chapter|end_card, background, accent, design, picture} (CSS colours in the production's palette: a pale page gets
        dark words; design: classic|bold|elegant|minimal|typewriter|storybook, else the production's style.text_design; picture: a shot id, whose
        chosen take shows dimmed behind the words, or an image or video)."""
        target = self._production(ctx, production)
        self._only_the_fix(ctx, target, [shot])
        if model.find(target.snapshot()["shots"], shot, "shot")["locked"]:
            raise ValueError(f"Shot {shot} is locked by the user: leave it as it is.")
        if card:
            picture = card.get("picture")
            if picture and not any(item["id"] == picture for item in target.snapshot()["shots"]):
                picture = target.adopt(ctx.resolve(picture))
            result = self.showrunner.add_card(production, shot, str(card.get("text") or ""), str(card.get("subtitle") or ""), card.get("style") or "chapter", by="deepy",
                                              accent=card.get("accent"), background=card.get("background"), design=card.get("design"), picture=picture)
        elif media:
            result = self.showrunner.add_take(production, shot, media, by="deepy", note=note)
            self._end_after_line(target, shot)
        else:
            raise ValueError("Give the media to record, or a card to draw.")
        return {"revision": target.doc["revision"], "shot": shot, "take": result["take"], "duration_s": result.get("duration_s"), "path": ctx.virtualize(str(target.resolve(result["path"])))}

    def _end_after_line(self, target, shot_id: str) -> None:
        """A speaking take recorded from elsewhere ends shortly after its line, as render's do: speaking models fill the rest with invented
        or repeated words. A trim the user set, or a shot planned no longer than its line, is kept."""
        doc = target.snapshot()
        shot = model.find(doc["shots"], shot_id, "shot")
        entry = templates.resolve(self._catalog(doc), shot["generation"].get("template"))
        spoken = model.spoken_lines(doc, shot)
        if entry is None or not entry["audio"] or len(spoken) != 1 or shot.get("updated_by") == "user" or (shot.get("trim") or [0.0, None])[1] is not None:
            return
        line = float(spoken[0].get("duration_s") or 0.0)
        if 0 < line < float(shot["duration_s"]) - 0.25:
            target.apply([{"op": "update_shot", "shot": shot_id, "fields": {"trim": [0.0, round(line + 0.35, 2)]}}], by="system", record=False)

    def verdict(self, ctx, production: str, shot: str, ok: bool, note: str = "", take: str | None = None, checks: dict[str, Any] | list[Any] | None = None) -> dict[str, Any]:
        """Your judgment of a take (the chosen one by default) after inspecting it; checks: measured values (black_ranges_seconds, cut_times_seconds, lufs...)
        or a list of what you checked."""
        target = self._production(ctx, production)
        self._only_the_fix(ctx, target, [shot])
        if isinstance(checks, list):
            checks = {"checked": checks}
        take = f"t{take}" if take is not None and str(take).isdigit() else take
        self._apply(target, [{"op": "verdict", "shot": shot, "take": take, "ok": ok, "note": note, "checks": checks}])
        return {"revision": target.doc["revision"], "shot": shot, "status": model.find(target.doc["shots"], shot, "shot")["status"]}

    @staticmethod
    def _length_warning(doc: dict[str, Any]) -> list[str]:
        """A plan far longer than the brief asks, or a finished cut short of it (a shorter plan may still be growing, sequence by sequence). A remake lasts as long as its source."""
        if doc["brief"].get("kind") == "remake":
            return []
        wanted, cut = doc["brief"].get("duration_s") or (0.0 if doc["brief"].get("song") else model.DEFAULT_LENGTH_S), model.cut_length(doc)
        if wanted and cut < wanted * 0.85 and all(shot["takes"] for shot in doc["shots"] if shot.get("included", True)):
            return [f"The cut lasts {cut:.0f} s and the brief asks for about {wanted:.0f} s. Bring it close (more room on the shots that need it, another shot); if you keep it, say plainly in your report how far it falls short."]
        if not wanted or cut <= wanted * 1.3:
            return []
        return [f"The cut lasts {cut:.0f} s and the brief asks for about {wanted:.0f} s. Bring it close, or say plainly why it differs and let the user choose before shooting."]

    @staticmethod
    def _music_warnings(target, doc: dict[str, Any], duration_s: float | None) -> list[str]:
        """Music that stops before the video does (a score from before shots were added): the sound of the rest is only the clips'."""
        ends = []
        if doc["brief"].get("song") and doc["edit"].get("audio_mode") in ("song", "mix") and (doc["audio"].get("analysis") or {}).get("duration_s"):
            ends.append(float(doc["audio"]["analysis"]["duration_s"]) - float(doc["edit"].get("song_start_s") or 0.0))
        for track in doc["edit"]["audio_tracks"]:
            if track.get("line") is None and track.get("path"):
                length = float(track["out_s"]) if track.get("out_s") is not None else float(media.probe(target.resolve(track["path"]))["duration_s"] or 0.0)
                ends.append(float(track.get("start_s") or 0.0) + length - float(track.get("in_s") or 0.0))
        if not ends or not duration_s or max(ends) > duration_s - 2.0:
            return []
        return [f"The music stops at {max(ends):.1f} s and the video runs to {duration_s:.1f} s. Make it as long as the cut (a longer score or song, a track for the tail) or tell the user plainly."]

    def edit(self, ctx, production: str, audio_mode: Literal["song", "clips", "mix"] | None = None, song: str | None = None, song_start_s: float | None = None,
             tracks: list[dict[str, Any]] | None = None, subtitles: list[dict[str, Any]] | None = None, overlays: list[dict[str, Any]] | None = None, level_voices: bool | None = None, voices_from_model: bool | None = None) -> dict[str, Any]:
        """Set the edit's sound, subtitles and text. voices_from_model: only when the user asks in words, switches the brief's Voices: true, the video model says the lines with nothing recorded first; false, they are recorded first again. level_voices: when the user asks, brings the shots whose lines their video model says to one level in the export (with the brief's Voices on the video model). audio_mode: song (the production's song alone), clips (each take's own sound: dialogue, effects) or
        mix (both). song: a song you made (media id or path) becomes the production's song, analysed for bars and sections. tracks:
        [{media, start_s, gain_db, in_s, out_s, fade_in_s, fade_out_s}] narration, voice-over or score laid over the edit, replacing the current tracks (a score under speech sits
        around -12 dB; in_s and out_s: the part of the file it plays; keep what `production` lists for tracks you leave as they are); [{shot, line, gain_db}] lays a line spoken with `speak` from its shot's start (start_s to place it elsewhere). subtitles: [{media, language}] SRT files, kept as removable tracks. overlays: text or pictures over the cut, replacing
        yours (the user's stay): [{text, subtitle, style: title|lower_third|caption|chapter|end_card, design, start_s, duration_s, fade_s}] drawn for
        you in the production's text design (style.text_design, or design; an end card reaching the end of the cut only fades in, over fade_s, and stays to the last frame), or [{media, start_s, duration_s, fade_s}] a transparent picture you drew
        at the production's format (draw_svg)."""
        target = self._production(ctx, production)
        if song is not None or audio_mode is not None or level_voices is not None or voices_from_model is not None:
            self._only_the_fix(ctx, target, what="the song and how the cut plays it")
        if voices_from_model:
            if target.snapshot()["brief"]["kind"] in ("music_video", "story"):
                raise ValueError("A music video's words follow its song and an illustrated story's its narrator, so the video model cannot say them here: tell the user.")
            if templates.voicing_template(target.snapshot(), self._catalog(target.snapshot()), False) is None:
                raise ValueError("None of the user's video templates can say lines, so they stay recorded first: tell the user.")
        fields = {key: value for key, value in {"audio_mode": audio_mode, "song_start_s": song_start_s, "level_voices": level_voices}.items() if value is not None}
        if tracks is not None:
            fields["audio_tracks"] = [self._track(ctx, target, track) for track in tracks]
        if subtitles is not None:
            fields["subtitles"] = [{"path": target.adopt(ctx.resolve(item["media"])), "language": str(item.get("language") or "eng")} for item in subtitles]
        ops = [{"op": "edit", "fields": fields}] if fields else []
        if overlays is not None:
            ops += [{"op": "remove_overlay", "overlay": item["id"]} for item in target.snapshot()["edit"]["overlays"] if item["by"] != "user"]
            for item in overlays:
                start = float(item.get("start_s") or 0.0)
                fields = {"start_s": start, "end_s": start + float(item.get("duration_s") or 3.0), "fade_s": float(item.get("fade_s", 0.5))}
                if item.get("text"):
                    style = item.get("style") or "title"
                    fields.update(text=item["text"], subtitle=item.get("subtitle") or "", style=style, design=item.get("design") or "",
                                  path=self.showrunner.render_text(target, item["text"], item.get("subtitle") or "", style, design=item.get("design")))
                else:
                    fields["path"] = target.adopt(ctx.resolve(item["media"]))
                ops.append({"op": "add_overlay", "fields": fields})
        if song:
            ops.append({"op": "brief", "fields": {"song": target.adopt(song)}})
        if voices_from_model is not None:
            ops.append({"op": "brief", "fields": {"voices_from_model": voices_from_model}})
        self._apply(target, ops)
        if song:
            self.showrunner.analysis(production, refresh=True)
        voices = {}
        if voices_from_model is not None:
            target.apply([{"op": "log", "text": "Voices: the video model now says the lines, with nothing recorded first." if voices_from_model else "Voices: the lines are recorded first again.", "shot": None}], by="deepy", record=False)
            state, catalog = target.snapshot(), self._catalog(target.snapshot())
            voices = {"voices_from_model": voices_from_model, "template_advice": templates.advice(state, catalog, self.showrunner.host.settings()["any_shot_length"]), "warnings": templates.lint(state, catalog),
                      "tell_the_user": ("You switched the Brief's Voices to the video model: a voice can change from shot to shot, and a shot already shot needs a new take to change." if voices_from_model
                                        else "You switched the Brief's Voices back to recordings: a shot already shot needs a new take to change.")}
        doc = self._usable(ctx, target, target.snapshot())
        typed = [str(text) for item in overlays or [] for text in (item.get("text"), item.get("subtitle")) if text]
        misspelled = mistyped_words(doc, typed)
        sung = {"next": f"The brief asks the song sung by {model.song_singers(doc)}: call change_singer_voice(cast_ids={doc['brief']['song_voices']}) now, then check_song."
                } if song and doc["brief"]["song_voices"] else {}
        return {**({"check_spelling": misspelled} if misspelled else {}), **voices, **sung, "revision": doc["revision"], "edit": doc["edit"], "song": doc["brief"]["song"]}

    @staticmethod
    def _track(ctx, target, track: dict[str, Any]) -> dict[str, Any]:
        """A sound track of the edit: a media file, or a shot's spoken line ({shot, line}), which starts with its shot unless start_s is given."""
        gain = float(track.get("gain_db") or 0.0)
        cut = {key: float(track[key]) for key in ("in_s", "out_s", "fade_in_s", "fade_out_s") if track.get(key) is not None}
        if track.get("media") or track.get("path"):  # path: as `production` lists a track that stays
            return {"path": target.adopt(ctx.resolve(track.get("media") or track["path"])), "start_s": float(track.get("start_s") or 0.0), "gain_db": gain, **cut}
        if not track.get("shot"):
            raise ValueError("A track names media (a file) or a shot and its line.")
        doc = target.snapshot()
        shot = model.find(doc["shots"], track.get("shot"), "shot")
        index = int(track.get("line") or 0)
        line = shot["dialogue"][index] if 0 <= index < len(shot["dialogue"]) else None
        member = model.speaker(doc, line["speaker"]) if line else None
        if not line or not line.get("audio") or member is None or line.get("voice") != member.get("voice"):
            raise ValueError(f"Shot {shot['id']} line {index} has no audio in its speaker's voice: speak it first (a track names media or a spoken line).")
        start = track.get("start_s")
        if start is None:
            start = next((item["start_s"] for item in model.timeline(doc) if item["shot"] == shot["id"]), 0.0)
        return {"path": line["audio"], "start_s": float(start), "gain_db": gain, **cut}

    def change_singer_voice(self, ctx, production: str, cast_ids: list[str] | None = None, voices: list[str] | None = None) -> dict[str, Any]:
        """Only when the user asks: the song sung in other voices by WanGP's SeedVC, its music, words and timing as they are, so takes already shot
        keep their lip sync. Who sings, in the order their voices are first heard in the song, one or two for a duet (SeedVC tells two voices
        apart, not more): cast_ids (their voice sample, made from their voice description when they have none) or voices, voice samples (media id
        or path). Requires SeedVC v1.0 Singing in Configuration > Extensions. The new version becomes the production's song, the former one goes to the bin."""
        target = self._production(ctx, production)
        self._only_the_fix(ctx, target, what="the song")
        if unavailable := self.showrunner.host.singer_voice_unavailable():
            raise ValueError(f"{unavailable} Tell the user.")
        if bool(cast_ids) == bool(voices):
            raise ValueError("Give cast_ids (the cast members who sing) or voices (voice samples), one of them.")
        doc = target.snapshot()
        members = [model.find(doc["cast"], cast_id, "cast") for cast_id in cast_ids or []]
        self.showrunner.singers(doc, members or [None] * len(voices))
        catalog = self._catalog(doc)
        samples = [target.resolve(self._voice(ctx, target, member, catalog, templates.effective(doc, catalog))[0]) for member in members] or [ctx.resolve(voice) for voice in voices]
        ctx.progress(text="Changing the singer's voice")
        return self.showrunner.change_singer_voice(target, samples, by="deepy")

    def _song_stem(self, ctx, song: Path) -> Path | None:
        """The song's voice alone, isolated the first time."""
        stem = song.with_name(f"{song.stem}_post.wav")
        if stem.is_file():
            return stem
        ctx.progress(text="Isolating the song's voice")
        made = (ctx.call("wangp_postprocess", {"media": ctx.virtualize(str(song)), "action": "remove_background", "arguments": {}}).get("result") or {}).get("generated_files") or []
        return ctx.resolve(made[0]) if made else None

    def check_song(self, ctx, production: str, lyrics: str | None = None) -> dict[str, Any]:
        """Listen to the production's song and compare it with the lyrics of the brief: which lines it really sings, and when. Run it on every new song, before shooting on it: a
        transcript that cannot read a line while it reads the others means the line is not sung as written. Subtitles of a cut that plays the song follow these times; lines
        that are not sung get none, unless they are sung unclearly where they belong. lyrics: the words the song was made from, when you wrote them yourself: they become the production's script, as written, section markers included."""
        target = self._production(ctx, production)
        if lyrics:
            self._apply(target, [{"op": "brief", "fields": {"script": lyrics}}])
        doc = target.snapshot()
        song, lines = target.resolve(doc["brief"].get("song")), model.lyric_lines(doc)
        if song is None or not song.is_file():
            raise ValueError("The production has no song: make one with edit(song=...).")
        if not lines:
            raise ValueError("The production has no lyrics to compare the song with: call check_song again with lyrics, the words the song was made from.")
        stem = self._song_stem(ctx, song)  # the voice alone: a pitch means nothing on a full mix, and a transcript of the mix stretches words over the music between lines
        ctx.progress(text="Listening to the song")
        measured = model.sung_lines(lines, self.showrunner.host.song_words(stem or song, str(doc["brief"].get("language") or "") or None))
        voice_hz = self.showrunner.host.voice_pitch(stem) if stem else None
        self._apply(target, [{"op": "lyrics", "lyrics": {"song": song.name, "lines": measured, "voice_hz": voice_hz}}])
        missing = [line["line"] for line in measured if not line["sung"]]
        return {"lines": len(measured), "sung": len(measured) - len(missing), "lead_voice_hz": voice_hz, **({"not_sung": missing} if missing else {}),
                "times": [f"{line['start_s']:.1f}-{line['end_s']:.1f} {line['line']}" + (" (sung unclearly)" if line.get("unclear") else "") for line in measured if line["start_s"] is not None],
                **({"warnings": warned} if (warned := model.song_words(target.snapshot())) else {})}

    def deep_review(self, ctx, production: str, add: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Measure the last version against its script (takes minutes, changes nothing else): what is really said or sung and when, subtitles against the words, voices against their cards and
        their lines, lips against the voice, the sound's level and silences, black or frozen picture, shots without a place; with Nemotron enabled in the Deepy configuration, who is heard against who the
        script says. Findings are recorded on the version: a fault is measured, a check is a suspicion for the user's eyes. add: [{shot, at_s, text}] what you saw yourself, added to its findings."""
        target = self._production(ctx, production)
        doc = target.snapshot()
        if not doc["deliverables"]:
            raise ValueError("There is no version to review: assemble one first.")
        version = doc["deliverables"][-1]
        if add is not None:
            seen = [review.finding("look", "check", item.get("at_s"), [item["shot"]] if item.get("shot") else [], str(item.get("text") or "")) for item in add]
            current = version.get("review") or {"findings": []}
            findings = sorted(current["findings"] + seen, key=lambda item: (item["at_s"] is None, item["at_s"] or 0.0))
            self._apply(target, [{"op": "review", "deliverable": version["id"], "review": {**current, "findings": findings}}])
            return {"findings": len(findings), "added": len(seen)}
        state = target.version(version["id"])
        host, resolve = self.showrunner.host, target.resolve
        clips = assemble.plan(state, resolve)["clips"]
        fps = assemble._canvas(state, clips)[2]
        clips = assemble._in_frames(clips, fps)
        offsets, _, total = assemble._places(clips)
        places = [{"shot": clip["shot"], "start_s": offset, "duration_s": clip["span"]} for clip, offset in zip(clips, offsets)]
        path = resolve(version["path"])
        ctx.progress(text="Listening to the version")
        audio, lines = review.decode(path), review.script_lines(state)
        words = host.song_words(path, str(state["brief"].get("language") or "") or None)
        measured = model.sung_lines([line["line"] for line in lines], words)
        findings = review.word_findings(lines, measured, words, places, audio) + review.subtitle_findings(media.subtitle_cues(path), words, places) + review.voice_findings(state, places, host.voice_pitch, resolve)
        ctx.progress(text="Checking the sound and the picture")
        catalog = self._catalog(state)
        moving = {shot["id"] for shot in state["shots"] if (entry := templates.resolve(catalog, shot["generation"].get("template"))) and entry["task"] in templates.VIDEO_TASKS}
        talking = {shot["id"] for shot in state["shots"] if any(line.get("line") for line in shot.get("dialogue") or [])}
        findings += review.sound_findings(audio, places, talking, assemble.delivery(path).get("silent_from_s"), total) + review.picture_findings(path, places, moving)
        findings += [review.finding("place", "check", None, [], text) for text in templates.places_missing(state, catalog)]
        speaking = review.speaking_shots(state, moving)
        watched = [(clip, place) for clip, place in zip(clips, places) if clip["shot"] in speaking and clip["kind"] == "video"]
        if watched:
            stem = self._song_stem(ctx, resolve(state["brief"]["song"])) if model.on_the_song(state) else None
            voice, song_start = (review.decode(stem), float(state["edit"].get("song_start_s") or 0.0)) if stem else (audio, 0.0)
            read_lips = host.lip_reader()
            for index, (clip, place) in enumerate(watched):
                if ctx.cancelled():
                    return {"cancelled": True, "note": "the review stopped before its end: nothing was recorded"}
                ctx.progress(done=index, total=len(watched), text=f"Looking at lips, {clip['shot']}", force=True)
                tqdm.write(f"[Showrunner] Deep Review: looking at lips, {clip['shot']} ({index + 1}/{len(watched)})")
                findings += filter(None, [review.lip_finding(clip["shot"], place["start_s"], read_lips(clip["path"], clip["start"], clip["frames"], fps, voice, song_start + place["start_s"]))])
            del read_lips  # its model's VRAM, before the slower check below
        order = lambda item: (item["at_s"] is None, item["at_s"] or 0.0)
        self._apply(target, [{"op": "review", "deliverable": version["id"], "review": {"findings": sorted(findings, key=order)}}])  # kept whatever happens to the slower check below
        if any(line["speaker"] for line in lines) and (turns := host.speaker_turns(ctx, path)):  # a song's lyrics name nobody to compare with
            findings += review.speaker_findings(state, lines, turns, places)
            self._apply(target, [{"op": "review", "deliverable": version["id"], "review": {"findings": sorted(findings, key=order), "speakers_heard": True}}])
        else:
            turns = []
        findings.sort(key=order)
        return {"version": len(doc["deliverables"]), "faults": sum(item["level"] == "fault" for item in findings), "checks": sum(item["level"] == "check" for item in findings), "findings": findings,
                "shots_at": {place["shot"]: round(place["start_s"], 2) for place in places}, "speakers_heard": bool(turns)}

    def assemble(self, ctx, production: str) -> dict[str, Any]:
        """Render the edit: included shots in order with their chosen takes, trims and transitions, and the soundtrack chosen in the edit.
        With the brief's review_before_edit, refused until the user approved the take of every shot in the edit (unless the user asked you
        to assemble)."""
        target = self._production(ctx, production)
        doc = target.snapshot()
        own = self._own(ctx) is not None
        if own and doc["director"].get("new_version") is False:
            raise ValueError("The user asked for no new version in this task: do not assemble. Stop here and say how the new takes came out; the user assembles the edit when they are ready.")
        asked = own and doc["director"].get("task") == "assemble"
        waiting = model.awaiting_approval(doc) if doc["brief"]["review_before_edit"] and not asked else []
        if waiting:
            raise ValueError(f"The user reviews every take before the edit (the brief's review_before_edit): do not assemble on your own. Stop here and ask the user to "
                             f"review the takes of shots {', '.join(waiting)} (approve each one, or mark it for a redo with a note). Assembling works once every take "
                             "in the edit is approved; the user can also assemble the edit themselves.")
        result = self.showrunner.assemble(production, by="deepy")
        doc, path = target.snapshot(), target.resolve(result["path"])
        cut = model.sequence(doc)
        lines = sum(len(shot.get("dialogue") or []) for shot, _ in cut)
        subtitles = {"source": "your SRT", "files": len(doc["edit"]["subtitles"])} if doc["edit"]["subtitles"] else {"source": "the shots' dialogue"} if lines else None
        if subtitles:  # the delivered file's cues with their times: checking them needs no parsing of the file
            cues = media.subtitle_cues(path)
            subtitles.update(count=len(cues), cues=[f"{start:.2f}-{end:.2f} {text}" for start, end, text in cues[:40]], **({"cues_listed": 40} if len(cues) > 40 else {}))
        delivered = assemble.delivery(path)
        warnings = self._music_warnings(target, doc, result.get("duration_s")) + model.cut_lines(doc) + model.singing_without_song(doc) + model.song_drift(doc) + model.song_words(doc) + model.missing_ends(doc) + self._length_warning(doc)
        cues = media.subtitle_cues(path)
        if cues and result.get("duration_s") and cues[-1][1] > result["duration_s"] - 0.7:
            warnings.append(f'The last words end at {cues[-1][1]:.2f} s and the video at {result["duration_s"]:.2f} s: it stops on them. Hold the end for a moment (a longer last shot, or a card after them).')
        for overlay in doc["edit"]["overlays"]:
            if overlay.get("style") in ("end_card", "chapter"):
                spoken = [text for start, end, text in media.subtitle_cues(path) if start < overlay["end_s"] - 0.1 and end > overlay["start_s"] + 0.1]
                if spoken:
                    warnings.append(f'The {overlay["style"].replace("_", " ")} at {overlay["start_s"]:.2f}-{overlay["end_s"]:.2f} s hides the picture while these words are spoken: "{spoken[0]}". Start it after the line.')
        if delivered.get("silent_from_s") is not None and result.get("duration_s"):
            warnings.append(f"The sound stops at {delivered['silent_from_s']:.1f} s and the video runs to {result['duration_s']:.1f} s: the end is silent. Give it sound (a song as long as "
                            "the cut, a score on the tail), or tell the user plainly.")
        return {**({"warnings": warnings} if warnings else {}), **self._since_last_version(target, doc), "revision": doc["revision"], "version": len(doc["deliverables"]), "path": ctx.virtualize(str(path)), "method": result["method"], "duration_s": result.get("duration_s"),
                "shots": len(cut), "left_out": [shot["id"] for shot in doc["shots"] if shot["included"] and not shot["takes"]], "sound": doc["edit"]["audio_mode"],
                "tracks": len(doc["edit"]["audio_tracks"]), "text_overlays": len(doc["edit"]["overlays"]), "subtitles": subtitles,
                "delivered": delivered, "targets": "-16 LUFS for a mix, -14 for a song alone; true peak under -1 dBTP"}  # measured: no need to parse the file

    def ingest_source(self, ctx, production: str, change: Literal["people", "scene"], cuts: list[int] | None = None, max_frames: int | None = None) -> dict[str, Any]:
        """A remake's first step: bring the brief's source video to the template's frame rate, find its cuts and write one shot per camera shot. A shot too long for one window becomes a chain its
        template continues by itself. change, from what the brief asks, picks the template of the shots: people (people_edits) or scene (scene_edits); give another shot the other one with
        update_shot. cuts: your corrections, as first frames of the new shots, when the returned ones are wrong: call again before anything is rendered. Returns the shots with their source
        ranges and `seen`, what Inspect Video finds in each, and what the template needs prepared: painted frames, references, a mask, or only the prompt."""
        from shared.deepy import video_edit

        target = self._production(ctx, production)
        doc = target.snapshot()
        if doc["brief"].get("kind") != "remake" or not doc["brief"].get("source"):
            raise ValueError("ingest_source needs a remake with a source video in its brief.")
        if any(shot["takes"] for shot in doc["shots"]):
            raise ValueError("This remake's shots already have takes: its windows stay as they are.")
        catalog = self._catalog(doc)
        template = templates.effective(doc, catalog)["people_edits" if change == "people" else "scene_edits"]
        entry = templates.resolve(catalog, template)
        minimum, step, fps = entry["take_frames"]
        offset, maximum = minimum % step, int(max_frames or entry["longest_frames"] or video_edit.DEFAULT_MAX_FRAMES)
        hop = (entry.get("video_edit") or {}).get("continuation_overlap") or step + offset
        ctx.progress(done=0, total=1, text="Reading the source video", force=True)
        folder = target.folder / "media" / "source"
        source = media.normalized_video(target.resolve(doc["brief"]["source"]), fps, folder)
        rate = media.frame_rate(source)  # one clock, the source's real rate: its frames are the takes' one to one, and the sound of every window is counted on it
        detected = video_edit.detect_cuts(str(source), rate, own_frames=True)
        chosen = detected["cuts"] if cuts is None else sorted(int(cut) for cut in cuts)
        groups = video_edit.plan_camera_shots(chosen, detected["frames"], max_frames=maximum, min_frames=minimum, hop=hop, step=step, offset=offset)
        canvas = media.fitted_canvas(source, doc["brief"]["resolution"], entry.get("pixels"))
        shots = []
        for group in groups:
            start, stop = group["own"]
            poster = media.poster(source, folder / f"{source.stem}-{start}-{stop - start}-poster.jpg", at_s=(start + stop) / 2 / rate)  # what the Board shows until the shot has a take
            # the shot's part of the source, read as it is: WanGP fits a virtual control video to the lengths its model renders, and delivers exactly its frames
            guide = target.rel(write_view(target.folder / "media" / "views" / f"{source.stem}-{start}-{stop - start}{source.suffix}", source, start, stop - 1))
            settings = {"video_guide": guide, "video_length": stop - start, **({"force_fps": str(rate)} if abs(rate - fps) >= media.NEAR_RATE else {})}
            if group["windows"] > 1:
                settings.update(sliding_window_size=maximum, sliding_window_overlap=hop, multi_prompts_gen_type="PW")
            shots.append({"title": f"{start / rate:.1f} to {stop / rate:.1f} s", "duration_s": round((stop - start) / rate, 3), "source_range": [start, stop], "source_poster": target.rel(poster),
                          "trim": [0.0, round((stop - start) / rate, 3)], "generation": {"template": template, "settings": settings}})
        subtitles = [{"path": target.rel(track["path"]), "language": track["language"]} for track in media.video_subtitles(target.resolve(doc["brief"]["source"]), folder)]  # its timing stays: so do the words shown over it
        results = self._apply(target, [{"op": "remake", "source": {"path": target.rel(source), "fps": rate, "frames": detected["frames"], "cuts": chosen}},
                                       {"op": "replace_plan", "shots": shots}, {"op": "brief", "fields": {"duration_s": round(detected["frames"] / rate, 3)}}, {"op": "edit", "fields": {"fps": rate, **({"subtitles": subtitles} if subtitles else {})}}])
        target.apply([{"op": "stage", "stage": "planning"}], by="system", record=False)
        created = next(result["shots"] for result in results if "shots" in result)
        edit = entry.get("video_edit") or {}
        source_name = ctx.virtualize(str(source))
        seen = self._look_at_shots(ctx, target, source_name, created, [group["own"] for group in groups], rate)
        return {"revision": target.doc["revision"], "source": source_name, "fps": rate, "frames": detected["frames"], "cuts": chosen, "canvas": canvas, "template": template,
                "needs": {"painted_frames": edit.get("input") == "painted_frames", "references": entry["refs"] and edit.get("input") != "painted_frames", "mask": entry["needs_mask"], "max_images": edit.get("max_images")},
                "shots": [{"shot": shot_id, "source_range": item["source_range"], "windows": group["windows"], "seen": seen.get(shot_id)} for shot_id, item, group in zip(created, shots, groups)]}

    def _look_at_shots(self, ctx, target, source: str, shots: list[str], owns: list[list[int]], fps: float) -> dict[str, str]:
        """Inspect Video on each camera shot, for what the plan decides from: is anything the brief changes in this shot, and how many painted frames does it take. A director that looks by itself takes a
        frame or two per shot, all shots in one question. It is not told the brief: told of a figure to replace, it finds that figure even where there is none. Its range stays 3 frames from each cut: it
        samples up to the range's last frame, and a range ending on the cut would show the next camera shot."""
        seen = {}
        for index, (shot, (own_start, own_end)) in enumerate(zip(shots, owns)):
            if ctx.cancelled():
                break
            ctx.progress(done=index, total=len(shots), text=f"Looking at Shot {index + 1}/{len(shots)}", force=True)
            start_s, end_s = (own_start + 3) / fps, (own_end - 4) / fps
            question = (f"These pictures are evenly spaced frames of one camera shot of a video, in order, from {start_s:.1f} s to {end_s:.1f} s of the video. Say who and what is in the shot: every person, animal, robot or character, "
                        "or nobody. Say how each one's face shows over the shot: frontal, in profile, from behind, hidden or too small to see. Say what changes and at about which second of the video: someone entering or leaving, "
                        "a face turning, the framing, the place (a change of place is a cut). At most five short sentences.")
            try:
                result = ctx.call("wangp_toolbox", {"action": "inspect_video", "arguments": {"media_id": source, "start_s": start_s, "end_s": end_s, "question": question, "mid_res_sampling": True}})
            except Exception as error:  # the shots are written: the director looks by itself
                print(f"[Showrunner] Could not look at {shot}: {error}", flush=True)
                continue
            result = result.get("result") or result
            if answer := result.get("answer"):
                measured = [f"Measured: the picture cuts at frame {round(time * fps)}." for time in result.get("cut_times_seconds") or []] + [f"Measured: black from {begin:g} to {stop:g} s." for begin, stop in result.get("black_ranges_seconds") or []]
                seen[shot] = " ".join([answer, *measured])
                target.apply([{"op": "update_shot", "shot": shot, "fields": {"seen": seen[shot]}}], by="system", record=False)
        return seen

    @staticmethod
    def _from_behind(person: dict[str, Any]) -> bool:
        return re.search(r"\bback of\b|from behind", str(person.get("where") or ""), re.IGNORECASE) is not None

    @classmethod
    def _swap_request(cls, people: list[dict[str, Any]], named: dict[str, tuple[dict[str, Any], str]], note: str | None = None, clothes: str = "sheet", declared: dict[str, str] | None = None) -> tuple[str, list[str]]:
        """What the painter reads to replace people, and the cast pictures it gets after the frame. Each person is found by `who`, their clothes and place in
        this frame, and becomes a cast member: the face and hair from the member's photo (their sheet when they have none, or for a look of the shot's own,
        or when the frame shows their back) with the looks of their card, the build and clothes of their sheet (clothes="frame": the frame's clothes, fitted
        to that build), and the frame's expression, mouth, head angle, gaze, pose, hands and crop; those seen from behind last. Tried on several remakes'
        frames (specs): the inspector's words for the person's hair and glasses read as something to keep, an open mouth closed unless the prompt holds it,
        an outfit written out in words drew a whole figure into close shots, and without a word on the build the original's body stayed under the new head.
        A member with a sheet alone takes the original's place: told that the original "becomes" them, the painter kept a robot behind its clown, put a bear's
        head on a man's body and left people in their own clothes (better in 7 of 11 frames of 7 remakes, the same in 3).
        named: each member by name, with the shot's picture of them. declared: the sentences the painter's model was trained on (its character_swap_prompt),
        filled instead, with the frame as image 1 and each member's character sheet after it: such a model takes their outfit from the sheet whatever `clothes` says."""
        replaced = sorted((person for person in people if person.get("becomes")), key=cls._from_behind)
        keys = list(dict.fromkeys(str(person["becomes"]).casefold() for person in replaced))
        if declared:
            if str(note or "").strip():
                raise ValueError("This Frame Painter takes only the sentences its model was trained on: paint without `instruction`, or ask the user to pick another Frame Painter in the brief's Models. Nothing was painted.")
            pictures = []
            for key in keys:
                member, path = named[key]
                sheet = member.get("master") if path in (member.get("images") or [])[:1] else path  # the sheet, or the shot's own look of them
                if not sheet:
                    raise ValueError(f"{member['name']} has no character sheet, which this Frame Painter needs: make_sheet first. Nothing was painted.")
                pictures.append(sheet)
            stays = sum(1 for person in people if not person.get("becomes"))
            end = declared["end_others"].format(others="other person" if stays == 1 else "other people") if stays else declared["end"]
            return " ".join([*(declared["person"].format(who=person["who"], image=keys.index(str(person["becomes"]).casefold()) + 2) for person in replaced), end]), pictures
        pictures, lines, numbers = [], ["Picture 1 is a frame from a video."], {}
        for key in keys:
            member, path = named[key]
            behind = all(cls._from_behind(person) for person in replaced if str(person["becomes"]).casefold() == key)
            photo = member["images"][0] if member.get("images") and path in (member.get("master"), member["images"][0]) and not (behind and member.get("master")) else None
            sheet = path if path != photo else member.get("master")
            numbers[key] = {"face": len(pictures) + 2}
            pictures.append(photo or path)
            lines.append(f"Picture {numbers[key]['face']} is {'a photo' if photo else 'the character sheet'} of {member['name']}.")
            if photo and sheet and sheet != photo:  # the face from the photo, the build and the clothes from the sheet; a sheet that is the photo is sent once
                numbers[key]["body"] = len(pictures) + 2
                pictures.append(sheet)
                lines.append(f"Picture {numbers[key]['body']} is the character sheet of {member['name']}.")
            elif not photo:
                numbers[key]["body"] = numbers[key]["face"]
        for person in replaced:
            key = str(person["becomes"]).casefold()
            member, number = named[key][0], numbers[key]
            name = member["name"]
            if cls._from_behind(person):
                lines.append(f"In Picture 1, {person['who']} becomes {name} seen from behind, as in the back view of Picture {number['face']}: only the back of the head and shoulders, no face, the same size and place, as blurred as in Picture 1.")
                continue
            looks = str(member.get("description") or "").partition(" wearing ")[0].strip().rstrip(",.")  # the face and hair: the outfit is a picture
            body = f"of the character sheet in Picture {number['body']}" if "body" in number else f"of the photo in Picture {number['face']}"  # no sheet: the photo shows the build and clothes
            worn = (f"{name} wears the clothes of that {'character sheet' if 'body' in number else 'photo'}, only where Picture 1 shows them." if clothes == "sheet"
                    else f"{name} wears the clothes of Picture 1, fitted to that build.")
            if number.get("body") == number["face"]:  # a sheet alone: the whole character in the original's place, with the card's build words only (told of a red nose, it painted one on a robot's eye)
                who, build = person["who"], ", ".join(dict.fromkeys(word.lower() for word in BUILD_WORDS.findall(looks.partition(" with ")[0])))
                whole = "with face, hair, body and clothes" if clothes == "sheet" else f"with face, hair and body, in the clothes {who} wore, fitted to {name}'s build"
                lines += [f"In Picture 1, {name} takes the place of {who}: {who} is no longer in the picture and nothing of them remains, not their body, head, arms or legs, in front of {name} or behind.",
                          f"{name} is the whole character of the character sheet in Picture {number['face']}, {build + ', ' if build else ''}{whole}, as large in the picture as {who} was."]
            else:
                lines += [f"In Picture 1, {person['who']} becomes {name}, with {name}'s own face and hair from Picture {number['face']}" + (f" ({looks})." if looks else "."),
                          f"{name}'s body build is that {body}, not the original's: where {name} is slimmer or smaller, draw {name} so and fill the freed space with the background. {worn}",
                          "Nothing of the original remains: not their face, their hair colour or cut, their glasses or their jewellery."]
            lines += [f"{name} keeps exactly the expression and mouth shape of Picture 1, an open mouth stays open, and the same head angle, gaze, pose and hands.",
                      f"The framing and crop stay exactly as in Picture 1: only what Picture 1 shows of {name} is painted."]
        where = lambda person: f" ({person['where']})" if str(person.get("where") or "").strip() else ""  # noqa: E731
        stays = "; ".join(f"{person['who']}{where(person)}" for person in people if not person.get("becomes"))
        return " ".join(filter(None, [*lines, f"Leave unchanged: {stays}." if stays else "", str(note or "").strip(),
                                      "Everything else stays exactly as in Picture 1: the framing, background, light, sharpness and the other people."])), pictures

    def _paint_people(self, ctx, target, template: str, frame_path: Path, people: list[dict[str, Any]], named: dict[str, tuple[dict[str, Any], str]], note: str | None, size: str,
                      clothes: str = "sheet", declared: dict[str, str] | None = None) -> Path | None:
        """The frame with everyone replaced in one draw (_swap_request). A draw that leaves a large person as they were is drawn again. Returns the picture, or None when nothing changed."""
        prompt, pictures = self._swap_request(people, named, note, clothes, declared)
        settings = {"template": template, "prompt": prompt, "resolution": size,  # a picture is made once, at the target: a draft and its final use the same one
                    "image_refs": [ctx.virtualize(str(path)) for path in (frame_path, *(target.resolve(picture) for picture in pictures))]}
        heads = [int(value) for person in people for match in re.findall(HEAD_SHARE, str(person.get("where") or "")) for value in match if value]
        for draw in range(PAINT_DRAWS):
            ctx.progress(done=draw, total=PAINT_DRAWS, text="Painting the Frame", force=True)
            entries = media_entries(ctx.call("wangp_generate", {"settings": settings}))
            drawn = ctx.resolve(entries[0]["path"]) if entries else None
            if drawn is not None and (max(heads, default=0) < LARGE_HEAD or media.changed_share(frame_path, drawn) >= SAME_PICTURE):
                return drawn
        return None

    @staticmethod
    def _shot_frame(doc: dict[str, Any], shot: str, frame: int) -> dict[str, Any]:
        item = model.find(doc["shots"], shot, "shot")
        if not item.get("source_range") or not doc.get("remake"):
            raise ValueError("This is for a remake's shots: run ingest_source first.")
        start, stop = item["source_range"]
        if not start <= int(frame) < stop:
            raise ValueError(f"Frame {frame} is outside this shot's source range {start} to {stop - 1}.")
        return item

    @staticmethod
    def _people_in(ctx, picture: str, question: str = FRAME_PEOPLE) -> str | None:
        """The visual inspector on one picture. None for a caller without an inspector, who looks with its own eyes."""
        try:
            result = ctx.call("wangp_toolbox", {"action": "inspect_media", "arguments": {"media_inputs": [{"media_id": picture}], "question": question}})
        except Exception as error:
            print(f"[Showrunner] The inspector did not look at {picture}: {error}", flush=True)
            return None
        return (result.get("result") or result).get("answer") or None

    @staticmethod
    def _painted_faults(before: str | None, after: str | None) -> list[str]:
        """What can be told from the inspector's numbers and words alone, the two pictures described apart. Its head sizes are rough: only a large loss counts."""
        if not before or not after:
            return []
        faults = []
        heads = [max((int(value) for match in re.findall(HEAD_SHARE, text) for value in match if value), default=0) for text in (before, after)]
        if heads[0] and heads[1] and heads[1] < 0.6 * heads[0]:
            faults.append(f"The largest head went from about {heads[0]} % of the picture's height to about {heads[1]} %: the framing is lost.")

        def camera(text):
            found = re.search(r"camera[^\n]*(?:\n[^\n]*)?", text, re.IGNORECASE)
            return next((rank for word, rank in (("extreme", 0), ("medium", 2), ("wide", 3), ("close", 1)) if found and word in found.group(0).lower()), None)
        ranks = camera(before), camera(after)
        if None not in ranks and ranks[1] > ranks[0]:
            faults.append("The camera is wider than in the source.")
        return faults

    def make_sheet(self, ctx, production: str, member: str, style: str = "as a photorealistic studio photograph") -> dict[str, Any]:
        """A remake: a cast member's character sheet, which becomes their picture: a large portrait beside their whole body from the front, the side and the back, on plain grey. From their `description` (their looks and complete
        outfit, nothing about whom they replace), or from their photo when they have `images`. style: how it is drawn, for instance "as a stylized 3D animated character". The painter takes their looks from it and, unlike from a
        posed picture, no framing or expression."""
        target = self._production(ctx, production)
        doc = target.snapshot()
        found = next((item for item in doc["cast"] if member == item["id"] or member.casefold() == item["name"].casefold()), None)
        if found is None:
            raise ValueError(f"{member} is not in the cast. Add them with cast first.")
        description = str(found.get("description") or "").strip().rstrip(".")
        if not description:
            raise ValueError(f"{found['name']} has no description: give their looks and complete outfit with cast first.")
        picks = templates.effective(doc, self._catalog(doc))
        photo = (found.get("images") or [None])[0]
        settings = {"template": picks.get("image_edits"), "prompt": SHEET_FROM_PHOTO.format(style=style, description=description), "image_refs": [ctx.virtualize(str(target.resolve(photo)))], "video_prompt_type": "I", "resolution": SHEET_SIZE} if photo else \
            {"template": picks.get("images"), "prompt": SHEET_FROM_WORDS.format(style=style, description=description), "resolution": SHEET_SIZE}
        ctx.progress(done=0, total=1, text=f"Drawing the Character Sheet of {found['name']}", force=True)
        entries = media_entries(ctx.call("wangp_generate", {"settings": settings}))
        if not entries:
            return {"member": found["name"], "error": "the image model returned no picture"}
        sheet = target.adopt(ctx.resolve(entries[0]["path"]))
        self._apply(target, [{"op": "update_cast", "cast": found["id"], "fields": {"master": sheet}}])
        return {"revision": target.doc["revision"], "member": found["name"], "sheet": ctx.virtualize(str(target.resolve(sheet))), "note": "A sheet keeps this size whatever the production's format: do not crop or resize it, its views would be cut."}

    def look_at_frame(self, ctx, production: str, shot: str, frame: int) -> dict[str, Any]:
        """A remake: what the visual inspector sees in a frame of the shot's source, numbered as in the source, person by person: where they are, how large, what shows of them, their expression, sharp or blurred. Call it before
        paint_frame: you direct the painter from these words, for this frame, and it is what the painted picture is verified against."""
        target = self._production(ctx, production)
        doc = target.snapshot()
        item = self._shot_frame(doc, shot, frame)
        source_frame = media.video_frame(target.resolve(doc["remake"]["path"]), int(frame), target.folder / "media" / "painted")
        name = ctx.virtualize(str(source_frame))
        seen = self._people_in(ctx, name, FRAME_POSES)
        if not seen:
            raise ValueError(f"The inspector could not look at frame {frame}: look at {name} with your own eyes, and paint it with your instruction and references.")
        target.apply([{"op": "update_shot", "shot": shot, "fields": {"looked": {**(item.get("looked") or {}), str(int(frame)): {"poses": seen, "before": self._people_in(ctx, name)}}}}], by="system", record=False)
        return {"shot": shot, "frame": int(frame), "source_frame": name, "people": seen,
                "next": "paint_frame with `people`: one entry per person listed here, with who (for someone replaced, their clothes and place in this frame only: the woman in the turquoise jumpsuit in the centre), where (the inspector's words for them, unchanged) and becomes (a cast member's name, or null for someone who stays as they are)."}

    def paint_frame(self, ctx, production: str, shot: str, frame: int, people: list[dict[str, Any]] | None = None, instruction: str | None = None, picture: str | None = None, references: list[str] | None = None, pin: bool = True, unchanged: bool = False, replaces: int | None = None,
                    clothes: Literal["sheet", "frame"] = "sheet") -> dict[str, Any]:
        """A remake: a frame of the shot's source, numbered as in the source, changed by the image editor. When people are replaced: look_at_frame first, then `people`, one entry per person it listed: who (the woman), where (the
        inspector's words for them), becomes (a cast member's name, or null for someone who stays as they are); for someone replaced, `who` names them by their clothes and place in this frame only, never
        their hair, face or glasses, which the painter would keep; clothes: the cast member's own outfit from their sheet, or `frame` when the brief keeps the
        person's costume (a Frame Painter trained on swaps, such as Viggle Image Swap, always takes the sheet's outfit and no `instruction`: give the card that costume). `instruction` then adds what is special in this frame. For any other change: `instruction`, with `references`. Or
        your own finished `picture`, or the source frame as it is (`unchanged`, for a camera setup the change does not touch, to hold it in the shot). It becomes one of the shot's painted frames, in source order; painting the same
        frame again replaces it, and each try draws differently; `replaces`: another painted frame of the shot that this one takes the place of. pin: the take passes through that picture at that moment (all frames of a shot are pinned or none); keep it on when people are replaced. Returns the source frame and the picture; when people are
        replaced `before` and `after`, what the inspector sees in each, `faults` measured from them, and `verify`; otherwise `comparison`: read them before you go on."""
        target = self._production(ctx, production)
        doc = target.snapshot()
        item = self._shot_frame(doc, shot, frame)
        first, last = self._paintable(doc, item)
        if not first <= int(frame) <= last:
            raise ValueError(f"Frame {frame} is within {CUT_GAP} frames of a cut: paint one from {first} to {last}"
                             f" (a shot under {2 * CUT_GAP + 1} frames has none: it takes the look of the shots it renders with). Nothing was painted.")
        catalog = self._catalog(doc)
        entry = templates.resolve(catalog, item["generation"].get("template"))
        limit = ((entry or {}).get("video_edit") or {}).get("max_images") or 4
        dropped = {int(frame), *([int(replaces)] if replaces is not None else [])}
        kept = [painted for painted in item.get("painted_frames") or [] if painted["frame"] not in dropped]
        if close := [painted["frame"] for painted in kept if abs(painted["frame"] - int(frame)) < FRAME_GAP]:  # upstream: two pictures of nearly one framing snap the render between them
            raise ValueError(f"Frame {frame} is within {FRAME_GAP} frames (3 s) of this shot's painted frame {close[0]}: a shot takes a second one only where a face turns to profile or back, or someone new enters, "
                             f"at least 3 s after the first. Paint frame {close[0]} again, pass replaces={close[0]} to paint this frame in its place, or pick a frame at least {FRAME_GAP} frames from it. "
                             "A shot's only painted frame shows the face: from behind alone, the model invents the face. Nothing was painted.")
        if len(kept) >= limit:
            raise ValueError(f"This shot's template takes {limit} painted frames: paint one of those again, or give the shot fewer.")
        swap = ((entry or {}).get("video_edit") or {}).get("kind") == "swap"
        looked = (item.get("looked") or {}).get(str(int(frame)))
        before, after = (looked or {}).get("before"), None
        folder = target.folder / "media" / "painted"
        source_frame = media.video_frame(target.resolve(doc["remake"]["path"]), int(frame), folder)
        if unchanged:
            painted_path = source_frame
        elif picture:
            painted_path = ctx.resolve(picture)
        else:
            directed = swap and (people is not None or references is None)  # people replaced: the cast's own pictures, whatever references say
            pictured = [] if references is not None and not directed else self._pictured_cast(doc, item)
            if not pictured and references is None and (swap or templates.cast_with_pictures(doc)):
                raise ValueError(f"Shot {shot} has no cast with a picture, so the painter would draw from words alone and the same person would look different in every frame. Give the shot its cast with update_shot first, "
                                 "or pass references=[] when nobody of the cast is in this frame. Nothing was painted.")
            template = templates.effective(doc, catalog).get("frame_painter" if directed else "image_edits")  # people replaced: the brief's Frame Painter
            size = self.showrunner.shot_size(target, doc, item, final=True)
            if directed:
                if not looked:
                    raise ValueError(f"Look at frame {frame} first with look_at_frame: you direct the painter from what this frame shows. Nothing was painted.")
                if not people or not all(str(person.get("who") or "").strip() for person in people) or not any(person.get("becomes") for person in people):
                    raise ValueError("Say in `people` what happens to everyone look_at_frame listed: who, where (its words for them) and becomes (a cast member's name, or null to stay as they are). Nothing was painted.")
                if said := [person["who"] for person in people if person.get("becomes") and LOOKS_WORDS.search(str(person["who"]))]:
                    raise ValueError(f"`who` names the hair, face or glasses of someone replaced ({'; '.join(said)}): the painter keeps what it reads there. Name them by their clothes and place in this frame only "
                                     "(\"the woman in the turquoise jumpsuit in the centre\"). Nothing was painted.")
                named = {member["name"].casefold(): (member, path) for member, path in pictured}
                if missing := list(dict.fromkeys(str(person["becomes"]) for person in people if person.get("becomes") and str(person["becomes"]).casefold() not in named)):
                    raise ValueError(f"{', '.join(missing)} is not in this shot's cast with a picture: give the shot its cast with update_shot first. Nothing was painted.")
                for _ in range(PAINT_DRAWS):  # a close shot drawn as a full figure is drawn again: the painter does it about one draw in two, and its pin pulls the render out
                    painted_path = self._paint_people(ctx, target, template, source_frame, people, named, instruction, size, clothes, templates.resolve(catalog, template)["swap_prompt"])
                    if painted_path is None:
                        return {"shot": shot, "frame": int(frame), "error": f"The painter changed nobody in {PAINT_DRAWS} draws, so nothing was recorded. Call paint_frame again with the same people, or try another frame of this shot."}
                    ctx.progress(done=0, total=1, text=f"Checking the Painted Frame of {shot}", force=True)
                    before = before or self._people_in(ctx, ctx.virtualize(str(source_frame)))
                    after = self._people_in(ctx, ctx.virtualize(str(painted_path)))
                    if not any("framing is lost" in fault or "wider" in fault for fault in self._painted_faults(before, after)):  # the size of a head kept on a full figure made a caricature
                        break
            else:
                if not str(instruction or "").strip():
                    raise ValueError("Give an instruction for the image editor, your own picture, or unchanged.")
                pictures = [ctx.resolve(path) for path in references] if references is not None else [target.resolve(path) for _, path in pictured]
                settings = {"template": template, "prompt": instruction, "image_refs": [ctx.virtualize(str(path)) for path in (source_frame, *pictures)], "resolution": size}  # a picture is made once, at the target: a draft and its final use the same one
                ctx.progress(done=0, total=1, text=f"Painting a frame of {shot}", force=True)
                entries = media_entries(ctx.call("wangp_generate", {"settings": settings}))
                if not entries:
                    return {"shot": shot, "frame": int(frame), "error": "the image editor returned no picture"}
                painted_path = ctx.resolve(entries[0]["path"])
        painted = self._record_painted(target, shot, int(frame), painted_path, source_frame, dropped, limit, pin)
        tries = next(value["tries"] for value in painted if value["frame"] == int(frame))
        source_name, picture_name = ctx.virtualize(str(source_frame)), ctx.virtualize(str(target.resolve(painted[[value["frame"] for value in painted].index(int(frame))]["path"])))
        result = {"revision": target.doc["revision"], "shot": shot, "frame": int(frame), "source_frame": source_name, "picture": picture_name, "painted_frames": [value["frame"] for value in painted], "pinned": pin}
        if unchanged:
            return {**result, "comparison": None}
        if not swap:
            return {**result, "comparison": self._compare_painted(ctx, source_name, picture_name)}
        ctx.progress(done=0, total=1, text=f"Checking the Painted Frame of {shot}", force=True)
        before = before or self._people_in(ctx, source_name)
        after = after or self._people_in(ctx, picture_name)
        return {**result, "before": before, "after": after, "faults": self._painted_faults(before, after), "tries": tries, "verify": PAINT_VERIFY}

    @staticmethod
    def _paintable(doc: dict[str, Any], item: dict[str, Any]) -> tuple[int, int]:
        """The first and last frame of a shot that may be painted: upstream keeps CUT_GAP frames from a cut, next to which a frame takes the other shot's look."""
        start, stop = item["source_range"]
        return start + CUT_GAP if start else start, stop - CUT_GAP - 1 if stop < doc["remake"]["frames"] else stop - 1

    def _record_painted(self, target, shot: str, frame: int, picture, source_frame, dropped: set[int], limit: int, pin: bool, by: str = "deepy") -> list[dict[str, Any]]:
        """A picture as the shot's painted frame `frame`, in source order with those not `dropped`, the shot's pictures and pins following them. Read again before writing: what
        another call recorded while this one painted stays. Returns the shot's painted frames."""
        item = model.find(target.snapshot()["shots"], shot, "shot")
        kept = [painted for painted in item.get("painted_frames") or [] if painted["frame"] not in dropped]
        if len(kept) >= limit:
            raise ValueError(f"This shot's template takes {limit} painted frames: paint one of those again, or give the shot fewer.")
        tries = next((painted.get("tries", 1) for painted in item.get("painted_frames") or [] if painted["frame"] == frame), 0) + 1
        painted = sorted([*kept, {"frame": frame, "path": target.adopt(picture), "source": target.rel(source_frame), "tries": tries, **({"by": "user"} if by == "user" else {})}], key=lambda value: value["frame"])
        start = item["source_range"][0]
        settings = {**item["generation"]["settings"], "image_refs": [value["path"] for value in painted], "video_prompt_type": "FIVU" if pin else "IVU",
                    "frames_positions": " ".join(str(value["frame"] - start + 1) for value in painted) if pin else ""}  # WanGP counts positions from 1
        self._apply(target, [{"op": "update_shot", "shot": shot, "fields": {"painted_frames": painted, "generation": {**item["generation"], "settings": settings}}}], by=by)
        return painted

    def place_picture(self, target, shot: str, frame: int, picture: Path) -> list[dict[str, Any]]:
        """The user's own picture as a painted frame of a remake's shot, fitted to the size the shot renders at. It takes the place of the shot's painted frames less
        than FRAME_GAP frames from it, and keeps the shot's pins as they are. Returns the shot's painted frames."""
        doc = target.snapshot()
        item = self._shot_frame(doc, shot, frame)
        first, last = self._paintable(doc, item)
        if first > last:
            raise ValueError(f"Shot {shot} is too short for a painted frame of its own: it takes the look of the shots it renders with.")
        if not first <= frame <= last:
            raise ValueError(f"Frame {frame} is too close to a cut, where the render takes the other shot's look: pick a frame of {shot} from {first} to {last}.")
        entry = templates.resolve(self._catalog(doc), item["generation"].get("template"))
        limit = ((entry or {}).get("video_edit") or {}).get("max_images") or 4
        painted = item.get("painted_frames") or []
        dropped = {frame, *(value["frame"] for value in painted if abs(value["frame"] - frame) < FRAME_GAP)}
        if len([value for value in painted if value["frame"] not in dropped]) >= limit:
            raise ValueError(f"Shot {shot} already has the {limit} painted frames its model takes: replace one of them instead.")
        folder = target.folder / "media" / "painted"
        source_frame = media.video_frame(target.resolve(doc["remake"]["path"]), frame, folder)
        fitted = media.fit_picture(picture, self.showrunner.shot_size(target, doc, item, final=True), folder)
        pin = not painted or "F" in str(item["generation"]["settings"].get("video_prompt_type") or "")
        return self._record_painted(target, shot, frame, fitted, source_frame, dropped, limit, pin, by="user")

    @staticmethod
    def _compare_painted(ctx, source_frame: str, picture: str) -> str | None:
        """The visual inspector on both pictures: who is in each and what differs. A wrong frame pulls its whole render toward it, and a director that keeps its plan in its head misses it. It is not told
        what was asked: told of a figure to replace, it finds that figure in the source even where there is none. None for a caller without an inspector, who looks with its own eyes."""
        question = ("The first picture is a frame of a video, the second the same frame after an edit. Say who is in each picture and where they stand, counting them; say so when a picture has nobody in it. Then say in one or two short "
                    "sentences what differs: a person, animal or object added, removed or swapped, a pose, the framing, the background.")
        try:
            result = ctx.call("wangp_toolbox", {"action": "inspect_media", "arguments": {"media_inputs": [{"media_id": source_frame}, {"media_id": picture}], "question": question}})
        except Exception as error:  # the frame is painted and recorded: only its check is missing
            print(f"[Showrunner] No comparison of the painted frame: {error}", flush=True)
            return None
        return (result.get("result") or result).get("answer") or None

    def check_take(self, ctx, production: str, shot: str, take: str | None = None) -> dict[str, Any]:
        """A remake: measure a shot's take (by default its last) against the source clip it was rendered from: source cuts moved by 8 frames or more or dropped, cuts added, black bands growing at an
        edge. A band comes from a painted frame carrying it: repaint that frame. It measures timing and borders, not identity: still look at the take. Give the result to verdict as checks."""
        from shared.deepy import video_edit

        target = self._production(ctx, production, read=True)
        doc = target.snapshot()
        item = model.find(doc["shots"], shot, "shot")
        if not item.get("source_range") or not doc.get("remake"):
            raise ValueError("check_take is for a remake's shots.")
        recorded = model.find(item["takes"], take, "take") if take else (item["takes"] or [None])[-1]
        if recorded is None:
            raise ValueError(f"Shot {shot} has no take yet.")
        fps, start = doc["remake"]["fps"], item["source_range"][0]
        known = tuple(cut - start for cut in doc["remake"]["cuts"] if start < cut < item["source_range"][1])
        trim = (list(item.get("trim") or []) + [0.0, None])[:2]
        result = video_edit.check_take(str(target.resolve(doc["remake"]["path"])), str(target.resolve(recorded["path"])), source_start=start, own_start=round(float(trim[0] or 0.0) * fps),
                                       own_end=None if trim[1] is None else round(float(trim[1]) * fps), known_cuts=known, fps=None)
        result["cut_faults"] = [{**fault, "means": CUT_FAULTS[fault["fault"]]} for fault in result["cut_faults"]]
        return {"shot": shot, "take": recorded["id"], **result}

    def log(self, ctx, production: str, text: str, shot: str | None = None) -> dict[str, Any]:
        target = self._production(ctx, production)
        target.apply([{"op": "log", "text": text, "shot": shot}], by="deepy", record=False)
        return {"revision": target.doc["revision"]}

    def status(self, ctx, production: str) -> dict[str, Any]:
        target = self._production(ctx, production, read=True)
        doc = target.snapshot()
        waiting = [{"shot": shot["id"], "take": take["id"], "path": ctx.virtualize(str(target.resolve(take["path"])))} for shot in doc["shots"]
                   if (take := model.chosen_take(shot)) and not take.get("verdict")]
        return {"revision": doc["revision"], "stage": doc["stage"], "shots": {status: [shot["id"] for shot in doc["shots"] if shot["status"] == status] for status in model.SHOT_STATUSES if any(shot["status"] == status for shot in doc["shots"])},
                "awaiting_verdict": waiting, "deliverable": (doc["deliverables"] or [None])[-1], "gpu_seconds": doc["budget"].get("gpu_seconds", 0)}
