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

from . import assemble, media, production as model, review, templates
from .director import media_entries

MEDIA_SETTINGS = model.MEDIA_SETTINGS
_REFERENCE = re.compile(r"^(@|[a-z]+:[0-9a-f]{6,}$)")  # an @alias path or a Gallery media id

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
        plugin.register_deepy_action(self.check_song, summary="Listen to the production's song: which lines of the lyrics it really sings, and when (the subtitles follow it).")
        plugin.register_deepy_action(self.deep_review, summary="Measure the last version against its script: words, subtitles, voices, lips, sound and picture, with the findings recorded on the version.")
        plugin.register_deepy_action(self.assemble, uses_gpu=False, summary="Render the edit (chosen takes in order, trims, transitions, soundtrack) as the production's deliverable.")
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

    def _warnings(self, production, shot_ids=None) -> list[str]:
        doc = production.snapshot()
        return self._hold_lengths(production, doc, shot_ids) + templates.lint(doc, self._catalog(doc), shot_ids)

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
        brief["references"] = [resolve(path) for path in brief.get("references", [])]
        voices = {member["id"]: member.get("voice") for member in doc.get("cast", [])}  # as the document keeps them, before they are renamed below
        for member in doc.get("cast", []):
            member.update(images=[resolve(path) for path in member.get("images", [])], master=resolve(member.get("master")), voice=resolve(member.get("voice")),
                          looks=[{**look, "master": resolve(look.get("master"))} for look in member.get("looks", [])])
        def media(settings):  # as the caller passes them back in settings
            for key in MEDIA_SETTINGS & set(settings):
                settings[key] = [resolve(item) for item in settings[key]] if isinstance(settings[key], list) else resolve(settings[key])
        for shot in doc.get("shots", []):
            media((shot.get("generation") or {}).get("settings") or {})
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

    def _generation(self, ctx, target, shot: dict[str, Any]) -> dict[str, Any]:
        """A shot's settings as render sends them: the brief's format, media named the caller's way, the cast's pictures as references."""
        settings = {"template": shot["generation"].get("template", "gen_video"), "resolution": model.resolution(target.snapshot()["brief"]),
                    **self._resolve_settings(ctx, target, shot["generation"].get("settings", {}))}  # the brief's format unless the shot sets its own
        entry = templates.resolve(self._catalog(target.snapshot()), settings["template"])
        settings.update(templates.planned_length(target.snapshot(), shot, entry))
        pictures = self._cast_pictures(target.snapshot(), shot)
        if entry is not None and entry["refs"] and pictures and not settings.get("image_refs"):  # the cast's identity in every shot that shows them
            settings["image_refs"] = [ctx.virtualize(str(target.resolve(path))) for path in pictures]
        if entry is not None and entry["task"] == "edit_image" and entry.get("sized_by_ref") and settings.get("image_refs") and "video_prompt_type" not in settings:
            settings["video_prompt_type"] = "I"  # an illustrated page keeps the production's format: its references bring the cast, not their picture's size
        lines = self._driving_lines(target.snapshot(), shot, entry)
        if lines and not settings.get("audio_guide") and not shot.get("audio_slice"):  # several lines drive the shot one after the other
            paths = [target.resolve(line["audio"]) for line in lines]
            settings["audio_guide"] = ctx.virtualize(str(paths[0] if len(paths) == 1 else media.joined_audio(paths, target.folder / "media" / "slices")))
        song = target.snapshot()["brief"].get("song")
        if song and shot.get("audio_slice") and not settings.get("audio_guide") and entry is not None and (entry["audio"] or entry["soundtrack"]):  # a slice is of the song: render cuts it
            settings["audio_guide"] = ctx.virtualize(str(target.resolve(song)))
        if entry is not None and entry["soundtrack"] and not entry["audio"] and settings.get("audio_guide") and not settings.get("audio_prompt_type"):
            settings["audio_prompt_type"] = "AS"  # the driving audio as a kept soundtrack the picture follows, not a voice reference
        return settings

    @staticmethod
    def _driving_lines(doc: dict[str, Any], shot: dict[str, Any], entry: dict[str, Any] | None, spoken: bool = True) -> list[dict[str, Any]]:
        """The dialogue lines that drive a shot's picture, in order (spoken=False: whether recorded or not): every line for a template driven
        by speech, the lines of the people it shows for a reference template that follows a soundtrack, never a narrator's line over them."""
        lines = model.spoken_lines(doc, shot) if spoken else shot.get("dialogue") or []
        if entry is None or model.on_the_song(doc):  # on a cut that plays the song, lines are lyrics: the song drives the lips, they only give the subtitles
            return []
        shown = [line for line in lines if line.get("speaker") in shot.get("cast", [])]
        return shown if entry["soundtrack"] and not entry["audio"] else shown or lines if entry["audio"] else []

    def _generation_with_loras(self, ctx, target, shot: dict[str, Any]) -> tuple[dict[str, Any], list[Any] | None]:
        """A shot's settings with the LoRAs of its cast (Settings > LoRAs on cast cards), and the LoRAs before them."""
        settings, doc = self._generation(ctx, target, shot), target.snapshot()
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

    def _apply(self, production, ops, revision=None):
        try:
            return production.apply(ops, by="deepy", base_revision=revision)
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
            result["format"] = f"{model.resolution(doc['brief'])}: the brief's, what shots render at (fitted to their model's grid) unless one sets its own resolution"
            result["cast"] = [{**member, "images": full["images"], "voice": full["voice"], **({"voice_description": full["voice_description"]} if full["voice_description"] else {})}
                              for member, full in zip(result["cast"], doc["cast"])]
            catalog = self._catalog(doc)
            result["template_advice"] = templates.advice(doc, catalog, self.showrunner.host.settings()["any_shot_length"])
            warnings = templates.lint(doc, catalog) + templates.places_missing(doc, catalog) + model.cut_lines(doc) + model.singing_without_song(doc) + model.song_drift(doc) + model.song_words(doc) + self._length_warning(doc)
            if warnings:
                result["warnings"] = warnings
            if doc["audio"].get("analysis"):
                analysis = doc["audio"]["analysis"]
                result["song_analysis"] = {key: analysis.get(key) for key in ("duration_s", "bpm", "sections")} | {"bars": analysis.get("bars", [])[:200]}
            return result
        if section == "shot":
            return {"revision": doc["revision"], "shot": model.find(doc["shots"], shot, "shot")}
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
        target.apply([{"op": "stage", "stage": "review" if target.doc["stage"] in ("brief", "planning") else target.doc["stage"]}], by="system", record=False)
        return response

    def update_shot(self, ctx, production: str, shot: str, fields: dict[str, Any], revision: int | None = None) -> dict[str, Any]:
        """fields: any of title, section, intent, camera, cast, dialogue, duration_s, audio_slice, generation (settings are merged; null removes one), status, included, chosen_take,
        trim [start_s, end_s], speed (0.25 to 4, how fast the take plays in the edit; trim stays in seconds of the take), sound (false: the take plays silent),
        transition {type: cut|crossfade|fade_black|fade_white, duration_s}, notes. revision: the one you read, which guards
        against the user's edits since your read (your own writes never conflict). Changed settings or cast are validated again."""
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
        result = self._apply(target, [{"op": "add_shots", "shots": self._portable_shots(ctx, target, shots), "position": position}])
        fixing = self._fixing(ctx, target)
        if fixing is not None:  # a shot inserted in a fix is part of it; the shots around it keep their takes
            target.apply([{"op": "director", "fields": {"shots": [*fixing, *result[0]["shots"]]}}], by="system", record=False)
        problems, renders = self._validate(ctx, target, result[0]["shots"])
        warnings = self._warnings(target, result[0]["shots"]) + problems
        return {"revision": target.doc["revision"], **result[0], **({"warnings": warnings} if warnings else {}), **({"renders": renders} if renders else {})}

    def remove_shot(self, ctx, production: str, shot: str, revision: int | None = None) -> dict[str, Any]:
        target = self._production(ctx, production)
        self._only_the_fix(ctx, target, [shot])
        result = self._apply(target, [{"op": "remove_shot", "shot": shot, "revision": revision}], revision)
        return {"revision": target.doc["revision"], "removed": shot, **result[0]}

    def move_shots(self, ctx, production: str, shots: list[str], before: str | None = None) -> dict[str, Any]:
        target = self._production(ctx, production)
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

    def render(self, ctx, production: str, shots: list[str], new_seed: bool | int = False) -> dict[str, Any]:
        """Generate one take per listed shot, in order, from the shot's generation (template + settings), and record it. Returns each take's media id
        to inspect; record a verdict afterwards. new_seed: true draws a new seed instead of the settings' one; a number is the seed to use."""
        target = self._production(ctx, production)
        self._only_the_fix(ctx, target, shots)
        renders = []
        for index, shot_id in enumerate(shots):
            if ctx.cancelled():
                break
            if self.showrunner.director.pausing(production):
                renders.append({"paused": True, "not_rendered": shots[index:], "note": "the user paused you: stop here, they will resume the task later"})
                break
            if self.showrunner.director.note_waiting(production) and index:
                renders.append({"stopped": True, "not_rendered": shots[index:], "note": "the user just wrote to you: read it, then decide what to render"})
                break
            shot = model.find(target.snapshot()["shots"], shot_id, "shot")
            if shot["locked"]:
                renders.append({"shot": shot_id, "error": "locked by the user"})
                continue
            if shot["generation"].get("template") == templates.DRAWN:
                renders.append({"shot": shot_id, "error": "a drawn shot: draw it with draw_svg and record it with record_take (or record_take card=...)"})
                continue
            doc = target.snapshot()
            catalog = self._catalog(doc)
            entry = templates.resolve(catalog, shot["generation"].get("template", "gen_video"))
            recorded = model.spoken_lines(doc, shot)
            missing = [number for number, line in enumerate(shot.get("dialogue") or []) if line in self._driving_lines(doc, shot, entry, spoken=False) and line not in recorded]
            if missing and not shot.get("audio_slice") and not (shot["generation"].get("settings") or {}).get("audio_guide"):
                # a line not spoken yet: the model would invent its words, in a voice of its own; spoken in the speaker's voice first, it drives the shot
                said = []
                if self._speak_lines(ctx, target, production, shot_id, missing, False, catalog, templates.effective(doc, catalog), [], said):
                    renders.append({"paused": True, "not_rendered": shots[index:], "note": "the user paused you: stop here, they will resume the task later"})
                    break
                if errors := [f"line {item['line']}: {item['error']}" for item in said if item.get("error")]:
                    renders.append({"shot": shot_id, "error": "its lines could not be spoken (" + "; ".join(errors) + "): fix the speakers' voices, then render it"})
                    continue
                shot = model.find(target.snapshot()["shots"], shot_id, "shot")
            settings, base_loras = self._generation_with_loras(ctx, target, shot)
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
                renders.append({"shot": shot_id, "error": "the shot's settings have no prompt"})
                continue
            if new_seed:
                settings["seed"] = -1 if new_seed is True else int(new_seed)
            ctx.progress(done=index, total=len(shots), text=f"Rendering {shot_id} ({index + 1}/{len(shots)})", force=True)
            target.apply([{"op": "update_shot", "shot": shot_id, "fields": {"status": "rendering"}}], by="deepy", record=False)
            started = time.time()
            try:
                response = ctx.call("wangp_generate", {"settings": settings})
                generated = time.time() - started
            except Exception as error:
                target.apply([{"op": "update_shot", "shot": shot_id, "fields": {"status": "failed"}}], by="deepy", record=False)
                renders.append({"shot": shot_id, "error": str(error)})
                continue
            entries = media_entries(response)
            if not entries:
                target.apply([{"op": "update_shot", "shot": shot_id, "fields": {"status": "failed"}}], by="deepy", record=False)
                result = (response.get("result") or {}) if isinstance(response, dict) else {}
                reasons = [str(item) for item in [response.get("error") if isinstance(response, dict) else None, *(result.get("errors") or [])] if item]
                renders.append({"shot": shot_id, "error": "the generation returned no media" + (f": {'; '.join(reasons)}" if reasons else "")})
                continue
            entry = entries[0]
            try:
                path = ctx.resolve(entry["path"])  # tool results name files as the caller does
                used = entry.get("settings") if isinstance(entry.get("settings"), dict) else settings
                take = self.showrunner.add_take(production, shot_id, path, by="deepy", settings=self._real_paths(ctx, used), media_id=entry.get("media_id"), gpu_s=generated, base_loras=base_loras)
            except Exception as error:  # the media exists: report it with the shot and render the next shots
                target.apply([{"op": "update_shot", "shot": shot_id, "fields": {"status": "failed"}}], by="deepy", record=False)
                renders.append({"shot": shot_id, "media_id": entry.get("media_id"), "error": f"the take could not be recorded ({error}): record it with record_take"})
                continue
            target.apply([{"op": "budget", "gpu_seconds": time.time() - started}], by="system", record=False)
            renders.append({"shot": shot_id, "take": take["take"], "media_id": entry.get("media_id"), "path": ctx.virtualize(path), "duration_s": take.get("duration_s"), "size": f"{take.get('width')}x{take.get('height')}",
                            **{key: response[key] for key in ("cut_times_seconds", "canvas") if isinstance(response, dict) and key in response}})
        ctx.progress(done=len(renders), total=len(shots), text="Rendered", force=True)
        return {"revision": target.doc["revision"], "renders": renders}

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
            if item["locked"] or not count or any(not 0 <= index < count for index in indexes):
                problem = "locked by the user: leave it as it is" if item["locked"] else f"{count} dialogue lines" + (f" (0 to {count - 1})" if count else ": write them with update_shot first")
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
    def _cast_pictures(doc: dict[str, Any], shot: dict[str, Any]) -> list[str]:
        """One picture per cast member the shot shows: the look worn in the shot's sequence, else the master, else the first picture."""
        members = {member["id"]: member for member in doc["cast"]}
        def picture(member):
            look = model.look_for(member, shot)
            return look["master"] if look else member.get("master") or (member.get("images") or [None])[0]
        return [path for cast_id in shot.get("cast") or [] if cast_id in members and (path := picture(members[cast_id]))]

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
        """Media paths in settings become the caller's names for them (tools refuse absolute paths without Read Everywhere)."""
        def resolve(value):
            if isinstance(value, str) and value and ":" not in value[:12].replace(":\\", "").replace(":/", "") and target.resolve(value).is_file():
                return ctx.virtualize(str(target.resolve(value)))
            if isinstance(value, list):
                return [resolve(item) for item in value]
            return value
        return {key: resolve(value) for key, value in settings.items()}

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
        """A plan far longer than the brief asks, or a finished cut short of it (a shorter plan may still be growing, sequence by sequence)."""
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
             tracks: list[dict[str, Any]] | None = None, subtitles: list[dict[str, Any]] | None = None, overlays: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Set the edit's sound, subtitles and text. audio_mode: song (the production's song alone), clips (each take's own sound: dialogue, effects) or
        mix (both). song: a song you made (media id or path) becomes the production's song, analysed for bars and sections. tracks:
        [{media, start_s, gain_db, in_s, out_s, fade_in_s, fade_out_s}] narration, voice-over or score laid over the edit, replacing the current tracks (a score under speech sits
        around -12 dB; in_s and out_s: the part of the file it plays; keep what `production` lists for tracks you leave as they are); [{shot, line, gain_db}] lays a line spoken with `speak` from its shot's start (start_s to place it elsewhere). subtitles: [{media, language}] SRT files, kept as removable tracks. overlays: text or pictures over the cut, replacing
        yours (the user's stay): [{text, subtitle, style: title|lower_third|caption|chapter|end_card, design, start_s, duration_s, fade_s}] drawn for
        you in the production's text design (style.text_design, or design; an end card reaching the end of the cut only fades in, over fade_s, and stays to the last frame), or [{media, start_s, duration_s, fade_s}] a transparent picture you drew
        at the production's format (draw_svg)."""
        target = self._production(ctx, production)
        if song is not None or audio_mode is not None:
            self._only_the_fix(ctx, target, what="the song and how the cut plays it")
        fields = {key: value for key, value in {"audio_mode": audio_mode, "song_start_s": song_start_s}.items() if value is not None}
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
        self._apply(target, ops)
        if song:
            self.showrunner.analysis(production, refresh=True)
        doc = self._usable(ctx, target, target.snapshot())
        typed = [str(text) for item in overlays or [] for text in (item.get("text"), item.get("subtitle")) if text]
        misspelled = mistyped_words(doc, typed)
        return {**({"check_spelling": misspelled} if misspelled else {}), "revision": doc["revision"], "edit": doc["edit"], "song": doc["brief"]["song"]}

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
        transcript that cannot read a line while it reads the others means the line is not sung as written. Subtitles of a cut that plays the song follow these times, and lines
        that are not sung get none. lyrics: the words the song was made from, when you wrote them yourself: they become the production's script, as written, section markers included."""
        target = self._production(ctx, production)
        if lyrics:
            self._apply(target, [{"op": "brief", "fields": {"script": lyrics}}])
        doc = target.snapshot()
        song, lines = target.resolve(doc["brief"].get("song")), model.lyric_lines(doc)
        if song is None or not song.is_file():
            raise ValueError("The production has no song: make one with edit(song=...).")
        if not lines:
            raise ValueError("The production has no lyrics to compare the song with: call check_song again with lyrics, the words the song was made from.")
        ctx.progress(text="Listening to the song")
        measured = model.sung_lines(lines, self.showrunner.host.song_words(song, str(doc["brief"].get("language") or "") or None))
        stem = self._song_stem(ctx, song)  # the voice alone: a pitch means nothing on a full mix
        voice_hz = self.showrunner.host.voice_pitch(stem) if stem else None
        self._apply(target, [{"op": "lyrics", "lyrics": {"song": song.name, "lines": measured, "voice_hz": voice_hz}}])
        missing = [line["line"] for line in measured if not line["sung"]]
        return {"lines": len(measured), "sung": len(measured) - len(missing), "lead_voice_hz": voice_hz, **({"not_sung": missing} if missing else {}), "times": [f"{line['start_s']:.1f}-{line['end_s']:.1f} {line['line']}" for line in measured if line["sung"]],
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
        asked = self._own(ctx) is not None and doc["director"].get("task") == "assemble"
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
