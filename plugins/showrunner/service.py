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

from shared.utils.loras_mutipliers import preparse_loras_multipliers

from . import assemble, media, production as model, templates, titles
from .director import Director
from .store import ProductionStore
from .toolbox import ShowrunnerToolbox


# What a take's settings say about the shot itself, kept when the shot moves to another template (the rest belongs to the old model).
CONTENT_SETTINGS = ("prompt", "negative_prompt", "alt_prompt", "image_refs", "image_start", "image_end", "audio_guide", "video_length", "duration_seconds", "resolution", "seed")


class _DirectContext:
    """What the render action asks of its caller, for shots the user renders without the showrunner: generations go straight to WanGP's queue, no language model takes part, and files keep their real paths."""

    def __init__(self, showrunner, production):
        self.showrunner, self.production = showrunner, production
        self.shot, self.job, self.stopped = None, None, False

    @property
    def outputs_dir(self) -> Path:
        return self.production.folder / "media"

    def virtualize(self, path) -> str:
        return str(path)

    def resolve(self, path) -> Path:
        return Path(path)

    def cancelled(self) -> bool:
        return self.stopped

    def cancel(self) -> None:
        self.stopped = True
        if self.job is not None:
            self.job.cancel()

    def progress(self, done=None, total=None, text="", *, force=False) -> None:
        pass

    def call(self, tool, arguments) -> dict[str, Any]:
        settings = arguments["settings"]
        if settings.get("seed") == -1:  # the take keeps the seed it really had, for Same again
            settings = {**settings, "seed": random.randint(0, 2 ** 31 - 1)}
        host, production_id = self.showrunner.host, self.production.id
        self.job = host.submit_generation(settings, host.production_workspace(self.production.snapshot()["title"], self.outputs_dir), on_progress=lambda update: self.showrunner.store.publish(
            {"type": "render", "production": production_id, "shot": self.shot, "progress": getattr(update, "progress", None), "status": getattr(update, "status", "")}))
        result = self.job.result()
        files = list(getattr(result, "generated_files", None) or [])
        if getattr(result, "success", False) and files:
            return {"outputs": [{"path": str(files[0]), "media_id": f"file:{Path(files[0]).name}", "settings": settings}]}
        errors = list(getattr(result, "errors", None) or [])
        return {"errors": errors, "result": {"errors": errors}}


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
        """Import a dropped file into the production's media folder, then use it as: bin, song, reference, cast (target = cast id or new), take (target = shot id; a sound
        plays over the edit from that shot), shot (a video or a picture as a new shot, target = its position or the end), painted (target = "shot:frame",
        a remake's painted frame), track (narration, score or effects over the edit, from shot target or 0 s) or subtitles (an SRT file)."""
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
        elif role == "source":
            if facts["kind"] != "video":
                raise ValueError("A remake's source is a video.")
            production.apply([{"op": "brief", "fields": {"source": relative}}])
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
        elif role == "track" or role == "take" and facts["kind"] == "audio":  # a sound dropped on a shot plays over the edit from that shot, never as its take
            doc = production.snapshot()
            start = next((part["start_s"] for part in model.timeline(doc) if part["shot"] == target), 0.0)
            tracks = doc["edit"]["audio_tracks"] + [{"path": relative, "start_s": round(start, 3), "gain_db": 0.0}]
            production.apply([{"op": "edit", "fields": {"audio_tracks": tracks, **({"audio_mode": "mix"} if doc["edit"]["audio_mode"] == "song" else {})}}])
        elif role == "take":
            production.apply([{"op": "add_take", "shot": target, "take": {**facts, "note": "Uploaded"}}])
        elif role == "shot":  # a video or a picture of the user's as a new shot (target: its position, else the end)
            if facts["kind"] == "audio":
                raise ValueError("A shot is a video or a picture: drop a sound on a shot to play it from there.")
            position, fields = model.media_shot(production.snapshot(), {**facts, "name": filename}, int(target) if str(target or "").isdigit() else None)
            target = production.apply([{"op": "add_shots", "position": position, "shots": [fields]}])[0]["shots"][0]
        elif role == "painted":  # target "shot:frame": the user's own picture of that frame of a remake's shot
            shot_id, _, frame = str(target or "").rpartition(":")
            if facts["kind"] != "image" or not frame.isdigit():
                raise ValueError("A painted frame is a picture, for one frame of a shot.")
            ShowrunnerToolbox(self).place_picture(production, shot_id, int(frame), stored)
        else:
            production.apply([{"op": "bin_add", "item": {**facts, "name": Path(filename).name, "by": "user"}}])
        return {"path": relative, "role": role, "target": target, **facts}

    def add_take(self, production_id: str, shot_id: str, path: str | Path, *, by: str, settings: dict[str, Any] | None = None, media_id: str | None = None,
                 note: str = "", gpu_s: float | None = None, base_loras: list[Any] | None = None, tier: str | None = None, kept_source: bool = False, choose: bool = True) -> dict[str, Any]:
        production = self.store.get(production_id)
        facts = self.take_facts(production, production.resolve(production.adopt(path)))  # a take from outside the production is copied in
        take = {**facts, "media": media_id, "settings": production.rel_settings(settings or {}), "note": note, "gpu_s": None if gpu_s is None else round(gpu_s, 1), "base_loras": base_loras, "tier": tier, "kept_source": kept_source}
        result = production.apply([{"op": "add_take", "shot": shot_id, "take": take, "choose": choose}], by=by)
        return {**result[0], **facts}

    def hear_take(self, production, shot_id: str, take_id: str, listening=None) -> dict[str, Any] | None:
        """Listens to a new take whose model said the shot's lines: the take keeps what was heard, the shot is cut around its lines as its switch says, and a line not said as written goes to the log.
        A transcript that fails leaves the take whole, and says so. listening: called just before it starts."""
        doc = production.snapshot()
        shot = model.find(doc["shots"], shot_id, "shot")
        take = model.find(shot["takes"], take_id, "take")
        lines = [line["line"] for line in shot.get("dialogue") or [] if str(line.get("line") or "").strip()]
        if not lines or take.get("kind") != "video" or not take.get("has_audio") or (take.get("settings") or {}).get("audio_guide") \
                or not templates.model_speaks(doc, shot, templates.resolve(self.catalog(doc), shot["generation"].get("template"))):
            return None
        if listening:
            listening()
        try:
            heard = model.heard_lines(lines, self.host.song_words(production.resolve(take["path"]), str(doc["brief"].get("language") or "") or None))
        except Exception as error:  # the take exists: the words are only checked
            production.apply([{"op": "log", "text": f"{shot_id} could not be checked against its lines ({error}): listen to it yourself.", "shot": shot_id}], by="system", record=False)
            return None
        wrong = [f"it does not say: {' / '.join(heard['missing'])}"] if heard["missing"] else []
        wrong += [f"it adds between its lines: {' / '.join(heard['between'])}"] if heard["between"] else []
        production.apply([{"op": "heard", "shot": shot_id, "take": take_id, "heard": heard}, *([{"op": "log", "text": f"{shot_id}: {'; '.join(wrong)}. Heard: {heard['text'] or 'no words'}.", "shot": shot_id}] if wrong else [])], by="system", record=False)
        return heard

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

    @staticmethod
    def _delivered(shot: dict[str, Any], fps: float) -> tuple[int, int]:
        """The frames of a remake's shot that the edit keeps, from the start of what it renders: its first and how many."""
        keep_from, keep_to = (list(shot.get("trim") or []) + [0.0, None])[:2]
        first, last = round(float(keep_from or 0.0) * fps), shot["source_range"][1] - shot["source_range"][0] if keep_to is None else round(float(keep_to) * fps)
        return first, last - first

    def shot_size(self, production, doc: dict[str, Any], shot: dict[str, Any], *, final: bool = False) -> str:
        """The size a video shot renders at, final or as a draft. A shot made from a video keeps that video's shape, below its model's largest size."""
        name = model.tier_name(doc["brief"], final)
        if not shot.get("source_range"):
            return model.RESOLUTIONS[doc["brief"]["aspect"]][name]
        entry = templates.resolve(self.catalog(doc), shot["generation"].get("template"))
        return media.fitted_canvas(production.resolve(doc["remake"]["path"]), name, (entry or {}).get("pixels"))

    def media_of(self, production_id: str, media_id: str) -> Path | None:
        """The file a Gallery media id the showrunner names: the take or the bin item that recorded it, else its media in the production's
        workspace Gallery (a picture used as a shot's start, say); None when neither has it."""
        production = self.store.get(production_id)
        doc = production.snapshot()
        path = next((item["path"] for item in [*doc["bin"], *(take for shot in doc["shots"] for take in shot["takes"])] if item.get("media") == media_id), None)
        if path:
            return production.resolve(path)
        found = doc["director"].get("workspace") and self.host.gallery_media(doc["director"]["workspace"], media_id)
        return Path(found) if found else None

    def source_part(self, production_id: str, shot_id: str) -> Path:
        """What a remake's shot delivers, as a video of its own for the page's player: cut once, the first time it is looked at."""
        production = self.store.get(production_id)
        doc = production.snapshot()
        shot, fps = model.find(doc["shots"], shot_id, "shot"), doc["remake"]["fps"]
        first, frames = self._delivered(shot, fps)
        return media.video_window(production.resolve(doc["remake"]["path"]), shot["source_range"][0] + first, frames, fps, production.folder / "media" / "source")

    def source_frame(self, production_id: str, shot_id: str, frame: int) -> Path:
        """One frame of a remake's source, numbered as in the source, for the user to paint elsewhere."""
        production = self.store.get(production_id)
        doc = production.snapshot()
        ShowrunnerToolbox._shot_frame(doc, shot_id, frame)
        return media.video_frame(production.resolve(doc["remake"]["path"]), frame, production.folder / "media" / "painted")

    def take_part(self, production_id: str, shot_id: str, take_id: str) -> Path:
        """The same of a take: a render runs on past its shot's cut, and those frames, never in the edit, show the next camera shot as it was."""
        production = self.store.get(production_id)
        doc = production.snapshot()
        shot, fps = model.find(doc["shots"], shot_id, "shot"), doc["remake"]["fps"]
        first, frames = self._delivered(shot, fps)
        return media.video_window(production.resolve(model.find(shot["takes"], take_id, "take")["path"]), first, frames, fps, production.folder / "media" / "parts")

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
        return {"roles": templates.choices(doc, self.catalog(doc), self.host.prompt_rules())}

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
        return {"roles": templates.choices(self._defaults(), self.catalog(self._defaults()), self.host.prompt_rules())}

    def choose_default_template(self, role: str, template: str | None) -> dict[str, Any]:
        self.host.save_default_templates(templates.pick(self.host.default_templates(), self.catalog(self._defaults()), role, template))
        return self.default_template_roles()

    def set_prompt_rules(self, template: str, text: str) -> dict[str, Any]:
        """The user's rules for the prompts the showrunner writes with a template, in every production (empty text: none)."""
        entry = templates.resolve(self.host.template_catalog([template]), template)
        if entry is None:
            raise ValueError(f"{template} is not one of Deepy's templates.")
        name, text = templates.entry_name(entry), text.strip()
        self.host.save_prompt_rules({key: value for key, value in self.host.prompt_rules().items() if key != name} | ({name: text} if text else {}))
        return {"template": name, "rules": text}

    def frame_sizes(self, production_id: str) -> dict[str, Any]:
        """What the shots of the edit come out at: their take's real size (shot true); not shot yet, what their model makes of the brief's format or of
        their own resolution (size None: their first reference decides it)."""
        doc = self.store.get(production_id).snapshot()
        catalog, sizes = self.catalog(doc), {}
        for shot in doc["shots"]:
            take = model.chosen_take(shot)
            if not shot["included"]:
                continue
            if take is not None and take.get("width") and take.get("height"):
                sizes[shot["id"]] = {"size": f"{take['width']}x{take['height']}", "shot": True}
                continue
            entry = templates.resolve(catalog, shot["generation"].get("template"))
            requested = shot["generation"].get("settings", {}).get("resolution") or self.shot_size(self.store.get(production_id), doc, shot)
            size = None if entry is not None and entry.get("sized_by_ref") and entry["task"] != "edit_image" else self.host.frame_size(entry, requested) if entry is not None and entry.get("model_type") else requested
            sizes[shot["id"]] = {"size": size, "shot": False}
        return {"sizes": sizes}

    # Cast LoRAs (Settings > LoRAs on cast cards; off by default) --------------------------------------------------------------------------
    def cast_lora_choices(self, production_id: str) -> dict[str, Any]:
        """What the cast cards offer: the installed LoRAs of each model family the production's shots use, and the family of each shot."""
        doc = self.store.get(production_id).snapshot()
        catalog = self.catalog(doc)
        picks = templates.effective(doc, catalog)
        names = [picks[role] for role in ("cast_shots", "cast_shots_lip_synced", "image_edits", "images", "shots_without_cast") if role in picks]
        names += [shot["generation"].get("template") for shot in doc["shots"]]
        families, folders = {}, {}
        for name in dict.fromkeys(names):
            entry = templates.resolve(catalog, name)
            if entry is None or not entry.get("model_type"):
                continue
            folder = folders[name] = self.host.lora_folder(entry["model_type"])
            if folder and folder not in families:
                families[folder] = {"folder": folder, "label": entry["label"], "loras": self.host.list_loras(entry["model_type"])}
        return {"families": list(families.values()), "shots": {shot["id"]: folders.get(shot["generation"].get("template")) for shot in doc["shots"] if shot["included"]}}

    def with_cast_loras(self, doc: dict[str, Any], shot: dict[str, Any], entry: dict[str, Any] | None, settings: dict[str, Any]) -> tuple[dict[str, Any], list[Any] | None]:
        """A shot's settings with the LoRAs of the cast members it shows, after the template's, when they fit the shot's model; a trigger word missing from
        the prompt opens it. Also returns the LoRAs before them, which a new take starts from (None when none applies)."""
        loras = [member["lora"] for member in doc["cast"] if member["id"] in shot["cast"] and member["lora"]]
        if not loras or entry is None or not entry.get("model_type") or not self.host.settings()["cast_loras"]:
            return settings, None
        folder = self.host.lora_folder(entry["model_type"])
        loras = [lora for lora in loras if lora["folder"] == folder]
        if not loras:
            return settings, None
        before = (list(settings["activated_loras"]), settings.get("loras_multipliers") or "") if "activated_loras" in settings else self.host.template_loras(templates.entry_name(entry))
        present = {Path(name).name for name in before[0]}
        added = [lora for lora in loras if Path(lora["file"]).name not in present]
        tokens = preparse_loras_multipliers(before[1])  # one per LoRA, 1 when missing, or the added ones would take another's place
        kept = before[1] if len(tokens) == len(before[0]) else " ".join((tokens + ["1"] * len(before[0]))[:len(before[0])])
        multipliers = "\n".join(filter(None, [kept, " ".join(f"{float(lora['multiplier']):g}" for lora in added)]))
        prompt = str(settings.get("prompt") or "")
        words = [lora["trigger"].strip() for lora in loras if lora["trigger"].strip() and lora["trigger"].strip().casefold() not in prompt.casefold()]
        return {**settings, "activated_loras": [*before[0], *(lora["file"] for lora in added)], "loras_multipliers": multipliers, **({"prompt": ", ".join([*words, prompt])} if words else {})}, list(before)

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
        stale = [shot for shot in production.snapshot()["shots"] if shot["status"] in ("rendering", "queued") and shot["id"] not in busy and shot.get("rendering_with") not in busy]  # a remake's shot rendering with another is as busy as it
        if stale:
            production.apply([{"op": "update_shot", "shot": shot["id"], "fields": {"status": "rendered" if shot["takes"] else "planned", "rendering_with": None}} for shot in stale], by="system", record=False)
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
        if shot.get("source_range") and shot["takes"]:  # a remake's shot renders from its own settings, with the neighbours it renders together with
            self.render_directly(production_id, [shot_id])
            return {"shot": shot_id, "status": "rendering"}
        take = model.chosen_take(shot)
        if take is None or not take.get("settings"):
            raise ValueError("This shot has no take with saved settings yet: use Render now.")
        tier = take.get("tier")
        settings = production.resolve_settings(take["settings"])
        settings.update({key: value for key, value in shot["generation"].get("settings", {}).items() if key in ("prompt", "alt_prompt", "negative_prompt")})
        if take.get("base_loras"):  # the cast cards say which LoRAs of the cast this take gets, as they are now
            settings["activated_loras"], settings["loras_multipliers"] = take["base_loras"]
        template = shot["generation"].get("template")
        entry = templates.resolve(self.catalog(doc), template or settings.get("template"))
        if template and settings.get("template") and template != settings["template"]:  # the user picked another model: the shot's content on it, not the old model's settings
            settings = {"template": template, **{key: value for key, value in settings.items() if key in CONTENT_SETTINGS}}
            if entry is not None and not entry["refs"]:
                settings.pop("image_refs", None)
            if entry is not None and entry["soundtrack"] and not entry["audio"] and settings.get("audio_guide"):
                settings["audio_prompt_type"] = "AS"  # as render: a reference template follows the driving audio as its soundtrack
        settings.update(templates.planned_length(doc, shot, entry))  # the shot as it is now, not its last take's length
        if not settings.get("audio_guide") and templates.model_speaks(doc, shot, entry) and settings.get("prompt"):
            settings["prompt"] = templates.with_spoken_lines(doc, shot, entry, settings["prompt"])  # the shot's prompt does not hold its lines
        settings, base_loras = self.with_cast_loras(doc, shot, entry, settings)
        if new_seed:
            settings["seed"] = random.randint(0, 2 ** 31 - 1)
        key, context = (production_id, shot_id), _DirectContext(self, production)  # cancels it while it waits for its turn as while it renders
        with self._render_lock:
            if key in self._renders:
                raise RuntimeError("This shot is already rendering.")
            self._renders[key] = context
        workspace = self.host.production_workspace(doc["title"], production.folder / "media")
        production.apply([{"op": "update_shot", "shot": shot_id, "fields": {"status": "rendering"}}], by="user", record=False)
        threading.Thread(target=self._finish_rerun, args=(production_id, shot_id, context, workspace, settings, base_loras, tier), daemon=True).start()
        return {"shot": shot_id, "status": "rendering"}

    def _finish_rerun(self, production_id, shot_id, context, workspace, settings, base_loras=None, tier=None):
        production = self.store.get(production_id)
        try:
            with self.host.gpu_turn(context.cancelled):  # after the showrunners at work in other productions, never beside a language model
                if context.stopped:
                    production.apply([{"op": "update_shot", "shot": shot_id, "fields": {"status": "planned"}}], by="system", record=False)
                    return
                started = time.time()
                context.job = self.host.submit_generation(settings, workspace, on_progress=lambda update: self.store.publish(
                    {"type": "render", "production": production_id, "shot": shot_id, "progress": getattr(update, "progress", None), "status": getattr(update, "status", "")}))
                result = context.job.result()
                files = list(getattr(result, "generated_files", None) or [])
                if getattr(result, "success", False) and files:
                    taken = self.add_take(production_id, shot_id, files[0], by="user", settings=settings, note="Rerun", gpu_s=time.time() - started, base_loras=base_loras, tier=tier)
                    production.apply([{"op": "budget", "gpu_seconds": time.time() - started}], by="system", record=False)
                    self.hear_take(production, shot_id, taken["take"])
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

    def render_groups(self, production_id: str, shots: list[str] | None = None) -> list[list[str]]:
        """The renders of a remake's shots asked for (all by default), one list of ids each (remake_groups); nothing for another kind of production."""
        doc = self.store.get(production_id).snapshot()
        catalog = self.catalog(doc)
        entry = templates.resolve(catalog, templates.effective(doc, catalog).get("people_edits"))
        return [[shot["id"] for shot in group] for group in ShowrunnerToolbox._render_groups(doc, entry, shots)] if doc.get("remake") and entry is not None else []

    def render_directly(self, production_id: str, shot_ids: list[str], final: bool = False, new_seed: bool | None = None) -> dict[str, Any]:
        """Render shots now, one after the other, without the showrunner: the render action's own work (the cast's pictures, spoken lines, lengths) with no language model and no verdict.
        A shot nobody chose a model for gets the brief's for what it shows, a shot without a prompt one made from what it says, kept on the shot, where it can be edited.
        new_seed: None draws a new seed for a draft and keeps the take's for a final (its draft at the target size); True, a new one either way (Reshoot)."""
        production = self.store.get(production_id)
        if self.director.status(production_id)["running"]:
            raise RuntimeError("The showrunner is working on this production: this comes back when it finishes, or after Pause or Stop.")
        doc = production.snapshot()
        catalog, ops, statuses = self.catalog(doc), [], {}
        for shot_id in shot_ids:
            shot = model.find(doc["shots"], shot_id, "shot")
            statuses[shot_id], generation = shot["status"], {}
            if shot["generation"].get("template") == "gen_video" and (template := templates.template_for(doc, shot, catalog)):  # every shot's default: nobody chose its model
                generation["template"] = template
            if not model.shot_prompt(shot).strip():
                entry = templates.resolve(catalog, generation.get("template") or shot["generation"].get("template"))
                if not (prompt := model.compiled_prompt(doc, shot, lip_sync=not templates.model_speaks(doc, shot, entry))):
                    raise ValueError(f"{shot['title'] or shot_id} has nothing to render yet: write what happens in it, or its prompt.")
                generation["settings"] = {"prompt": prompt}
            ops.append({"op": "update_shot", "shot": shot_id, "fields": {"status": "rendering", **({"generation": generation} if generation else {})}})
        context = _DirectContext(self, production)
        with self._render_lock:
            if busy := [shot_id for shot_id in shot_ids if (production_id, shot_id) in self._renders]:
                raise RuntimeError(f"Already rendering: {', '.join(busy)}.")
            self._renders.update({(production_id, shot_id): context for shot_id in shot_ids})
        production.apply(ops, by="user", record=False)
        threading.Thread(target=self._render_directly, args=(production, shot_ids, statuses, context, final, not final if new_seed is None else new_seed), daemon=True).start()
        return {"shots": shot_ids, "status": "rendering"}

    def _render_directly(self, production, shot_ids, statuses, context, final=False, new_seed=True):
        with self.host.gpu_turn(context.cancelled):  # after the showrunners at work in other productions, never beside a language model
            self._render_shots(production, shot_ids, statuses, context, final, new_seed)

    def _render_shots(self, production, shot_ids, statuses, context, final, new_seed=True):
        toolbox, covered = ShowrunnerToolbox(self), set()  # covered: shots a render of this run already gave a take (remake shots rendered together)
        for shot_id in shot_ids:
            context.shot = shot_id
            try:
                try:
                    render = {} if context.stopped or shot_id in covered else toolbox.render_shot(context, production, shot_id, new_seed=new_seed, by="user", final=final, requested=shot_ids)
                    covered.update(item["shot"] for item in render.get("rendered_with") or [])
                    error = render.get("error")
                except Exception as failure:
                    error = str(failure)
                if context.stopped:  # as it was: a shot marked for a redo stays marked
                    production.apply([{"op": "update_shot", "shot": shot_id, "fields": {"status": statuses[shot_id]}}], by="system", record=False)
                elif error:
                    production.apply([{"op": "update_shot", "shot": shot_id, "fields": {"status": "failed"}}, {"op": "log", "text": f"Render of {shot_id} failed: {error}"}], by="system", record=False)
            except model.OperationError:  # deleted while it waited: the next shots still render
                pass
            finally:
                with self._render_lock:
                    self._renders.pop((production.id, shot_id), None)
                self.store.publish({"type": "render", "production": production.id, "shot": shot_id, "done": True})

    def cancel_rerun(self, production_id: str, shot_id: str) -> None:
        with self._render_lock:
            job = self._renders.get((production_id, shot_id))
        if job is not None:
            job.cancel()

    # Assembly -------------------------------------------------------------------------------------------------------------------------
    def detach_sound(self, production_id: str, shot_id: str, *, by: str = "user") -> dict[str, Any]:
        """A shot's sound as a track of its own at its place in the cut, the shot then playing silent. The track holds the whole take's sound (at the
        shot's speed), so that it can be extended past the cut as well as shortened."""
        production = self.store.get(production_id)
        doc = production.snapshot()
        shot = model.find(doc["shots"], shot_id, "shot")
        take = model.chosen_take(shot)
        if take is None or take.get("kind") != "video" or not take.get("has_audio"):
            raise ValueError("This shot's take has no sound.")
        if not shot["sound"]:
            raise ValueError("This shot's sound is already detached.")
        clip = assemble.placed_clip(doc, production.resolve, shot_id)
        if clip is None:
            raise ValueError("Include this shot in the edit first: its sound goes where it plays.")
        speed = clip["speed"]
        file = media.take_sound(production.resolve(take["path"]), speed, production.folder / "media" / "tracks", "" if speed == 1.0 else assemble._atempo(speed))
        track = {"path": production.rel(file), "start_s": round(clip["offset"], 3), "gain_db": 0.0, "in_s": round(clip["start"] / speed, 3), "out_s": round(clip["end"] / speed, 3),
                 "name": f"Sound of {shot['title'] or shot_id}", "shot": shot_id, "take": take["id"]}
        production.apply([{"op": "edit", "fields": {"audio_tracks": [*doc["edit"]["audio_tracks"], track]}}, {"op": "update_shot", "shot": shot_id, "fields": {"sound": False}}], by=by)
        return track

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
