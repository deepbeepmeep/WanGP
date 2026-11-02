# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""What the Showrunner page and Deepy's toolbox both do: imports, takes with their posters, song analysis, reruns and assembly."""
from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from pathlib import Path
from typing import Any

from . import assemble, media, production as model, templates, titles
from .director import Director
from .store import ProductionStore


# What a take's settings say about the shot itself, kept when the shot moves to another template (the rest belongs to the old model).
CONTENT_SETTINGS = ("prompt", "negative_prompt", "alt_prompt", "image_refs", "image_start", "image_end", "audio_guide", "video_length", "duration_seconds", "resolution", "seed")


class Showrunner:
    def __init__(self, store: ProductionStore, host):
        self.store, self.host = store, host
        self.director = Director(store, host, settle=self.settle)
        self._renders: dict[tuple[str, str], Any] = {}
        self._render_lock = threading.Lock()
        self._analysis_lock = threading.Lock()

    # Media ----------------------------------------------------------------------------------------------------------------------------
    def take_facts(self, production, path: str | Path) -> dict[str, Any]:
        """A take's (or bin item's) facts: probe, poster and filmstrip, with paths relative to the production."""
        path = Path(path)
        info = media.probe(path)
        facts = {"path": production.rel(path), "kind": info["kind"], **{key: info.get(key) for key in ("duration_s", "width", "height", "fps", "has_audio", "codec")}}
        if info["kind"] in ("video", "image"):
            key = media.cache_key(path)
            facts["poster"] = production.rel(media.poster(path, production.folder / "cache" / f"{key}.jpg"))
            if info["kind"] == "video":
                facts["strip"] = production.rel(media.filmstrip(path, production.folder / "cache" / f"{key}-strip.jpg"))
        return facts

    def upload(self, production_id: str, filename: str, data_path: Path, role: str = "bin", target: str | None = None) -> dict[str, Any]:
        """Import a dropped file into the production's media folder, then use it as: bin, song, reference, cast (target = cast id or new), take (target = shot id),
        track (narration, score or effects over the edit) or subtitles (an SRT file)."""
        production = self.store.get(production_id)
        suffix = Path(filename).suffix.lower()
        if role == "subtitles":
            if suffix != ".srt":
                raise ValueError("Subtitles are SRT files.")
            stored = media.import_file(data_path, production.folder / "media" / "uploads", move=True, name=filename)
            subtitles = production.snapshot()["edit"]["subtitles"] + [{"path": production.rel(stored), "language": "eng"}]
            production.apply([{"op": "edit", "fields": {"subtitles": subtitles}}])
            return {"path": production.rel(stored), "role": role, "target": None, "kind": "subtitles"}
        if media.kind_of(f"x{suffix}") is None or role == "track" and media.kind_of(f"x{suffix}") == "image":
            raise ValueError(f"Unsupported file type: {suffix or filename}.")
        stored = media.import_file(data_path, production.folder / "media" / "uploads", move=True, name=filename)
        facts = self.take_facts(production, stored)
        relative = facts["path"]
        if role == "song":
            production.apply([{"op": "brief", "fields": {"song": relative}}])
            self.analysis(production_id, refresh=True)
        elif role == "reference":
            references = production.snapshot()["brief"]["references"] + [relative]
            production.apply([{"op": "brief", "fields": {"references": references}}])
        elif role == "cast":
            doc = production.snapshot()
            member = next((item for item in doc["cast"] if item["id"] == target), None)
            if member is None:
                result = production.apply([{"op": "add_cast", "fields": {"name": Path(filename).stem.replace("_", " ").title()[:40], "images": [relative], "master": relative}}])
                target = result[0]["cast"]
            elif facts["kind"] == "audio":
                production.apply([{"op": "update_cast", "cast": member["id"], "fields": {"voice": relative}}])
            else:
                production.apply([{"op": "update_cast", "cast": member["id"], "fields": {"images": member["images"] + [relative], "master": member["master"] or relative}}])
        elif role == "take":
            production.apply([{"op": "add_take", "shot": target, "take": {**facts, "note": "Uploaded"}}])
        elif role == "track":
            tracks = production.snapshot()["edit"]["audio_tracks"] + [{"path": relative, "start_s": 0.0, "gain_db": 0.0}]
            production.apply([{"op": "edit", "fields": {"audio_tracks": tracks, **({"audio_mode": "mix"} if production.snapshot()["edit"]["audio_mode"] == "song" else {})}}])
        else:
            production.apply([{"op": "bin_add", "item": {**facts, "name": Path(filename).name, "by": "user"}}])
        return {"path": relative, "role": role, "target": target, **facts}

    def add_take(self, production_id: str, shot_id: str, path: str | Path, *, by: str, settings: dict[str, Any] | None = None, media_id: str | None = None,
                 note: str = "", gpu_s: float | None = None) -> dict[str, Any]:
        production = self.store.get(production_id)
        facts = self.take_facts(production, production.resolve(production.adopt(path)))  # a take from outside the production is copied in
        take = {**facts, "media": media_id, "settings": production.rel_settings(settings or {}), "note": note, "gpu_s": None if gpu_s is None else round(gpu_s, 1)}
        result = production.apply([{"op": "add_take", "shot": shot_id, "take": take}], by=by)
        return {**result[0], **facts}

    def shoot_rates(self, production_id: str) -> dict[str, Any]:
        """How long each template takes to generate at this production's frame size, measured on takes (see model.shoot_rates): the page
        estimates a shoot from it. A take without its own measure reads the generation time WanGP wrote into the file."""
        def timed(production):
            with production.lock:
                takes = [(take, (take.get("settings") or {}).get("template")) for shot in production.doc["shots"] for take in shot["takes"]]
            return [{"template": template, "kind": take.get("kind"), "duration_s": take.get("duration_s"), "width": take.get("width"), "height": take.get("height"),
                     "seconds": take.get("gpu_s") or media.generation_time(production.resolve(take["path"]))} for take, template in takes if template]
        production, others = self.store.get(production_id), []
        for other in self.store.ids():
            try:
                others += timed(self.store.get(other)) if other != production_id else []
            except (OSError, ValueError, KeyError):  # a damaged production folder: the home page lists it with its error
                continue
        return {"rates": model.shoot_rates(timed(production), others, model.resolution(production.snapshot()["brief"]))}

    def analysis(self, production_id: str, refresh: bool = False) -> dict[str, Any] | None:
        """The song's waveform, beats, bars, energy and sections, computed once per song file and cached in the document."""
        production = self.store.get(production_id)
        doc = production.snapshot()
        song = production.resolve(doc["brief"].get("song"))
        if song is None or not song.is_file():
            return None
        key = media.cache_key(song, media.ANALYSIS_VERSION)
        current = doc["audio"].get("analysis")
        if current and current.get("key") == key and not refresh:
            return current
        with self._analysis_lock:
            cache = production.folder / "cache" / f"{key}-analysis.json"
            if cache.is_file():
                analysis = json.loads(cache.read_text(encoding="utf-8"))
            else:
                analysis = {"key": key, **media.analyze_song(song), "waveform": media.peaks(song)}
                cache.write_text(json.dumps(analysis), encoding="utf-8")
        production.apply([{"op": "audio", "analysis": analysis}], by="system", record=False)
        return analysis

    def waveform(self, production_id: str, path: str) -> dict[str, Any]:
        production = self.store.get(production_id)
        file = production.resolve(path)
        if file is None or not production.owns(file):
            raise PermissionError("This file does not belong to the production.")
        cache = production.folder / "cache" / f"{media.cache_key(file)}-peaks.json"
        if cache.is_file():
            return json.loads(cache.read_text(encoding="utf-8"))
        peaks = media.peaks(file, 800)
        cache.write_text(json.dumps(peaks), encoding="utf-8")
        return peaks

    def subtitles(self, production_id: str, path: str) -> str:
        """The subtitles a video of the production carries (an assembled version's), as WebVTT for the page's player; "" without any."""
        production = self.store.get(production_id)
        file = production.resolve(path)
        if file is None or not file.is_file() or not production.owns(file):
            raise PermissionError("This file does not belong to the production.")
        cache = production.folder / "cache" / f"{media.cache_key(file)}-subtitles.vtt"
        if not cache.is_file():
            cache.write_text(media.subtitles_vtt(file), encoding="utf-8")
        return cache.read_text(encoding="utf-8")

    # Templates and text -------------------------------------------------------------------------------------------------------------
    def catalog(self, doc: dict[str, Any]) -> list[dict[str, Any]]:
        """Deepy's templates, with the custom ones the production (or the defaults) still names."""
        return self.host.template_catalog(templates.named(doc))

    def template_roles(self, production_id: str) -> dict[str, Any]:
        """The template picker: each Deepy category with its templates, the automatic one (Deepy's selection) and the user's pick."""
        doc = self.store.get(production_id).snapshot()
        return {"roles": templates.choices(doc, self.catalog(doc))}

    def choose_template(self, production_id: str, role: str, template: str | None) -> dict[str, Any]:
        production = self.store.get(production_id)
        doc = production.snapshot()
        ops = templates.choose(doc, self.catalog(doc), role, template)
        production.apply(ops[:1])
        if ops[1:]:  # the shots follow the pick without becoming the user's edits (a new plan may still replace them); undo reverts both
            production.apply(ops[1:], by="system", record=False)
        return self.template_roles(production_id)

    def _defaults(self) -> dict[str, Any]:
        return {"brief": {"templates": self.host.default_templates()}, "shots": []}

    def default_template_roles(self) -> dict[str, Any]:
        """The same picker for the templates new productions start with."""
        return {"roles": templates.choices(self._defaults(), self.catalog(self._defaults()))}

    def choose_default_template(self, role: str, template: str | None) -> dict[str, Any]:
        self.host.save_default_templates(templates.pick(self.host.default_templates(), self.catalog(self._defaults()), role, template))
        return self.default_template_roles()

    def create(self, title: str, kind: str, **brief: Any):
        """A new production, starting with the default templates."""
        return self.store.create(title, kind, **{"templates": templates.chosen(self._defaults(), self.catalog(self._defaults())), **brief})

    def render_text(self, production, text: str, subtitle: str = "", style: str = "title", accent: str | None = None, background: str | None = None,
                    design: str | None = None, picture: str | Path | None = None) -> str:
        """A picture of designed text at the production's format (transparent around it; opaque for cards, on `background` or over `picture`, an
        image or video file), cached by content. design: one of titles.DESIGNS, else the production's text design."""
        doc = production.snapshot()
        size = tuple(int(value) for value in model.resolution(doc["brief"]).split("x"))
        if not str(text).strip():
            raise ValueError("The text is empty.")
        design = design or doc["style"].get("text_design") or "classic"
        if design not in titles.DESIGNS:
            raise ValueError(f"Text design must be one of {', '.join(titles.DESIGNS)}.")
        if picture and media.kind_of(picture) == "video":  # a frame of the video, at the card's width
            picture = media.poster(picture, production.folder / "cache" / f"{media.cache_key(picture)}-still{size[0]}.jpg", width=size[0])
        key = hashlib.sha1(json.dumps([text, subtitle, style, size, accent, background, design, str(picture or "")]).encode("utf-8")).hexdigest()[:12]
        out = production.folder / "media" / "titles" / f"{style}-{key}.png"
        if not out.is_file():
            titles.render(text, subtitle, style, size, out, accent, background, design, picture)
        return production.rel(out)

    def add_text(self, production_id: str, text: str, subtitle: str = "", style: str = "title", start_s: float = 0.0, duration_s: float = 3.0, fade_s: float = 0.5,
                 *, by: str = "user", design: str | None = None) -> dict[str, Any]:
        """Lay designed text over the edit from start_s for duration_s (in `design`, else the production's text design)."""
        production = self.store.get(production_id)
        path = self.render_text(production, text, subtitle, style, design=design)
        result = production.apply([{"op": "add_overlay", "fields": {"path": path, "text": text, "subtitle": subtitle, "style": style, "design": design or "", "start_s": start_s,
                                                                    "end_s": float(start_s) + float(duration_s), "fade_s": fade_s}}], by=by)
        return {"overlay": result[0]["overlay"], "path": path}

    def update_text(self, production_id: str, overlay_id: str, fields: dict[str, Any], *, by: str = "user") -> dict[str, Any]:
        """Change an overlay's timing or text; new words or a new style draw it again."""
        production = self.store.get(production_id)
        overlay = model.find(production.snapshot()["edit"]["overlays"], overlay_id, "overlay")
        fields = dict(fields)
        if overlay.get("style") and {"text", "subtitle", "style", "design"} & set(fields):
            merged = {key: fields.get(key, overlay.get(key) or "") for key in ("text", "subtitle", "style", "design")}
            fields["path"] = self.render_text(production, merged["text"], merged["subtitle"], merged["style"], design=merged["design"] or None)
        production.apply([{"op": "update_overlay", "overlay": overlay_id, "fields": fields}], by=by)
        return {"overlay": overlay_id}

    def add_card(self, production_id: str, shot_id: str, text: str, subtitle: str = "", style: str = "chapter", *, by: str = "user", accent: str | None = None,
                 background: str | None = None, design: str | None = None, picture: str | None = None) -> dict[str, Any]:
        """A full-frame card (chapter or end card) recorded as an image take of a shot, on `background` (a CSS colour) with `accent` (line and subtitle),
        in `design` (else the production's), over `picture` when given: a shot id (a frame of its chosen take) or a media file of the production."""
        if style not in ("chapter", "end_card"):
            raise ValueError("A card is a chapter or an end card.")
        production = self.store.get(production_id)
        path = production.resolve(self.render_text(production, text, subtitle, style, accent, background, design, self.card_picture(production, picture)))
        return self.add_take(production_id, shot_id, path, by=by, note=f"{titles.STYLES[style]}: {text}")

    @staticmethod
    def card_picture(production, picture: str | None) -> Path | None:
        """The file behind a card: a shot id names its chosen take (a frame of a video), anything else is a file of the production."""
        if not picture:
            return None
        shot = next((item for item in production.snapshot()["shots"] if item["id"] == picture), None)
        if shot is not None:
            take = model.chosen_take(shot)
            if take is None or take.get("kind") not in ("video", "image"):
                raise ValueError(f"Shot {picture} has no picture to put behind the card yet: render it first.")
            return production.resolve(take["path"])
        path = production.resolve(picture)
        if not path.is_file() or media.kind_of(path) not in ("video", "image"):
            raise ValueError(f"The card's picture must be a shot id, an image or a video: {picture}.")
        return path

    # Versions ---------------------------------------------------------------------------------------------------------------------------
    def version(self, production_id: str, deliverable_id: str, against: str = "current") -> dict[str, Any]:
        """A version's state, and what changed between it and `against` (the current production, or another version), oldest first."""
        production = self.store.get(production_id)
        doc = production.version(deliverable_id)
        other = production.snapshot() if against == "current" else production.version(against)
        deliverables = production.snapshot()["deliverables"]
        if against == "current":  # both lengths from the edit: versions assembled before cuts kept the planned lengths played whole takes
            changes = model.version_changes(doc, other)
        else:
            (older, older_length), (newer, newer_length) = sorted([(doc, model.find(deliverables, deliverable_id, "version").get("duration_s")),
                                                                   (other, model.find(deliverables, against, "version").get("duration_s"))], key=lambda pair: pair[0]["revision"])
            changes = model.version_changes(older, newer, lengths=(older_length, newer_length))
        return {"doc": doc, "deliverable": model.find(deliverables, deliverable_id, "version"), "changes": changes}

    def bring_back(self, production_id: str, deliverable_id: str, kind: str, item: str, *, by: str = "user") -> dict[str, Any]:
        """One piece of that version (a take, a shot, a cast member, a text, a sound track or subtitles) back in the production, as one change."""
        production = self.store.get(production_id)
        return production.apply([{"op": "bring_back", "doc": production.version(deliverable_id), "kind": kind, "item": item}], by=by)[0]

    def restore_version(self, production_id: str, deliverable_id: str, *, by: str = "user") -> dict[str, Any]:
        production = self.store.get(production_id)
        return production.apply([{"op": "restore_version", "doc": production.version(deliverable_id)}], by=by)[0]

    def duplicate_version(self, production_id: str, deliverable_id: str) -> dict[str, Any]:
        production = self.store.get(production_id)
        doc, deliverables = production.version(deliverable_id), production.snapshot()["deliverables"]
        doc["deliverables"] = deliverables[:deliverables.index(model.find(deliverables, deliverable_id, "version")) + 1]  # the copy's versions end with this one
        copy_ = self.store.duplicate(production_id, f"{production.snapshot()['title']} (version {len(doc['deliverables'])})", doc=doc)
        return copy_.snapshot()

    # Undo and interrupted work ---------------------------------------------------------------------------------------------------------
    def settle(self, production_id: str) -> list[str]:
        """Shots shown rendering while nothing renders them (an undo, a WanGP restart or a stopped job left the status) go back to what they
        are: rendered with a take, planned without one. Called with no director job running; the user's reruns keep theirs."""
        production = self.store.get(production_id)
        with self._render_lock:
            busy = {shot for key, shot in self._renders if key == production_id}
        stale = [shot for shot in production.snapshot()["shots"] if shot["status"] in ("rendering", "queued") and shot["id"] not in busy]
        if stale:
            production.apply([{"op": "update_shot", "shot": shot["id"], "fields": {"status": "rendered" if shot["takes"] else "planned"}} for shot in stale], by="system", record=False)
        return [shot["id"] for shot in stale]

    def undo(self, production_id: str, redo: bool = False) -> dict[str, Any]:
        """Undo (or redo) the last change, then settle shots that the restored document shows rendering while nothing renders them."""
        production = self.store.get(production_id)
        revision = production.redo() if redo else production.undo()
        if revision is not None and not self.director.status(production_id)["running"]:
            self.settle(production_id)
        return {"revision": revision, "undone": None if redo else getattr(production, "undone", None), "doc": production.snapshot()}

    # Rendering without the director ---------------------------------------------------------------------------------------------------
    def rerun(self, production_id: str, shot_id: str, *, new_seed: bool = True) -> dict[str, Any]:
        """Render a new take of a shot from its last take's settings with the shot's current prompt, without Deepy."""
        production = self.store.get(production_id)
        doc = production.snapshot()
        shot = model.find(doc["shots"], shot_id, "shot")
        take = model.chosen_take(shot)
        if take is None or not take.get("settings"):
            raise ValueError("This shot has no take with saved settings yet: ask the showrunner to render it.")
        settings = production.resolve_settings(take["settings"])
        settings.update({key: value for key, value in shot["generation"].get("settings", {}).items() if key in ("prompt", "alt_prompt", "negative_prompt")})
        template = shot["generation"].get("template")
        if template and settings.get("template") and template != settings["template"]:  # the user picked another model: the shot's content on it, not the old model's settings
            settings = {"template": template, **{key: value for key, value in settings.items() if key in CONTENT_SETTINGS}}
            entry = templates.resolve(self.catalog(doc), template)
            if entry is not None and not entry["refs"]:
                settings.pop("image_refs", None)
            if entry is not None and entry["soundtrack"] and not entry["audio"] and settings.get("audio_guide"):
                settings["audio_prompt_type"] = "AS"  # as render: a reference template follows the driving audio as its soundtrack
        if new_seed:
            settings["seed"] = random.randint(0, 2 ** 31 - 1)
        key = (production_id, shot_id)
        with self._render_lock:
            if key in self._renders:
                raise RuntimeError("This shot is already rendering.")
            workspace = self.host.production_workspace(doc["title"], production.folder / "media")
            job = self.host.submit_generation(settings, workspace, on_progress=lambda update: self.store.publish(
                {"type": "render", "production": production_id, "shot": shot_id, "progress": getattr(update, "progress", None), "status": getattr(update, "status", "")}))
            self._renders[key] = job
        production.apply([{"op": "update_shot", "shot": shot_id, "fields": {"status": "rendering"}}], by="user", record=False)
        threading.Thread(target=self._finish_rerun, args=(production_id, shot_id, job, settings), daemon=True).start()
        return {"shot": shot_id, "status": "rendering"}

    def _finish_rerun(self, production_id, shot_id, job, settings):
        production = self.store.get(production_id)
        started = time.time()
        try:
            result = job.result()
            files = list(getattr(result, "generated_files", None) or [])
            if getattr(result, "success", False) and files:
                self.add_take(production_id, shot_id, files[0], by="user", settings=settings, note="Rerun", gpu_s=time.time() - started)
                production.apply([{"op": "budget", "gpu_seconds": time.time() - started}], by="system", record=False)
            else:
                errors = list(getattr(result, "errors", None) or [])
                status = "planned" if getattr(result, "cancelled", False) else "failed"
                production.apply([{"op": "update_shot", "shot": shot_id, "fields": {"status": status}}, *([{"op": "log", "text": f"Rerun of {shot_id} failed: {errors[0]}"}] if errors else [])],
                                 by="system", record=False)
        except Exception as error:
            production.apply([{"op": "update_shot", "shot": shot_id, "fields": {"status": "failed"}}, {"op": "log", "text": f"Rerun of {shot_id} failed: {error}"}], by="system", record=False)
        finally:
            with self._render_lock:
                self._renders.pop((production_id, shot_id), None)
            self.store.publish({"type": "render", "production": production_id, "shot": shot_id, "done": True})

    def cancel_rerun(self, production_id: str, shot_id: str) -> None:
        with self._render_lock:
            job = self._renders.get((production_id, shot_id))
        if job is not None:
            job.cancel()

    # Assembly -------------------------------------------------------------------------------------------------------------------------
    def assemble(self, production_id: str, by: str = "user") -> dict[str, Any]:
        production = self.store.get(production_id)
        doc = production.snapshot()
        video_args, audio_args, extension = self.host.encode_settings()
        name = f"{_safe_name(doc['title'], '')[:40] or 'production'}-r{doc['revision']}{extension}"
        out = production.folder / "media" / "exports" / name  # inside the production's workspace: Deepy can inspect the edit
        self.store.publish({"type": "assemble", "production": production_id, "state": "running"})
        try:
            result = assemble.render(doc, production.resolve, out, video_args=video_args, audio_args=audio_args)
        except Exception as error:
            self.store.publish({"type": "assemble", "production": production_id, "state": "error", "error": str(error)})
            raise
        facts = self.take_facts(production, out)
        finished = all(shot["takes"] and shot["status"] in ("approved", "rendered") for shot in doc["shots"] if shot["included"])
        production.apply([{"op": "deliverable", "deliverable": {**facts, "kind": "video", "method": result["method"], "revision": doc["revision"], "by": by,
                                                                "edit_key": model.edit_key(doc), "snapshot": production.save_version(doc)}},
                          {"op": "stage", "stage": "delivered" if finished else "editing"}], by=by)  # delivered only when no shot is left to shoot or redo
        self.store.publish({"type": "assemble", "production": production_id, "state": "done", "path": facts["path"]})
        return {**facts, "method": result["method"], "seconds": result["seconds"]}


def _safe_name(filename: str, tag: str) -> str:
    path = Path(filename)
    stem = "".join(char if char.isalnum() or char in "-_ " else "_" for char in path.stem).strip()[:48] or "media"
    return f"{stem}{'-' + tag if tag else ''}{path.suffix.lower()}"
