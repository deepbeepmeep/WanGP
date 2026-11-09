# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Productions on disk: one self-contained folder each, an atomic document with its revision history, and change notifications."""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Callable

from . import production as model

DOCUMENT = "production.json"
HISTORY_LIMIT = 200  # revisions kept for undo
FOLDERS = ("media", "cache", "exports", "history", "session", "versions")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:32].rstrip("-") or "production"


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    for attempt in range(40):  # Windows refuses to replace a file another process has open, for as long as it reads it: an antivirus, a search index, a backup
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.05)


def settings_folder(settings_root: str | os.PathLike | None) -> Path:
    """Showrunner's configuration folder, in WanGP's settings folder (--settings)."""
    return Path(settings_root or "settings").resolve() / "showrunner"


def rename_templates(folder: str | os.PathLike, renames: dict[str, str]) -> int:
    """Deepy templates renamed ("task/old name": "task/new name"): the references Showrunner saved, in its configuration `folder` (the default
    picks, the keys of the prompt rules) and in every production (its document, versions and undo history). Returns the files rewritten."""
    def renamed(value):
        if isinstance(value, str):
            return renames.get(value, value)
        if isinstance(value, list):
            return [renamed(item) for item in value]
        return {renames.get(key, key): renamed(item) for key, item in value.items()} if isinstance(value, dict) else value
    config_path, count = Path(folder) / "config.json", 0
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
    if (new := renamed(config)) != config:
        config_path.write_text(json.dumps(new, indent=1), encoding="utf-8")
        count += 1
    root = Path(os.path.expanduser(config.get("productions_root") or Path(folder) / "productions"))
    for production in (path for path in root.iterdir() if (path / DOCUMENT).is_file()) if root.is_dir() else ():
        for path in (production / DOCUMENT, *(production / "versions").glob("*.json"), *(production / "history").glob("*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (new := renamed(payload)) != payload:
                _write_json(path, new)
                count += 1
    return count


def _media_settings(settings: dict[str, Any], convert: Callable[[str], str]) -> dict[str, Any]:
    one = lambda value: [one(item) for item in value] if isinstance(value, list) else convert(value) if isinstance(value, str) and value else value
    return {key: one(value) if key in model.MEDIA_SETTINGS else value for key, value in settings.items()}


class Production:
    """One production folder. Every change goes through apply(), under the production's lock."""

    def __init__(self, store: "ProductionStore", folder: Path):
        self.store, self.folder = store, folder
        self.lock = threading.RLock()
        self.doc = model.normalize(json.loads((folder / DOCUMENT).read_text(encoding="utf-8")))
        self.doc["id"] = folder.name
        for shot in self.doc["shots"]:  # takes recorded before their settings were kept relative
            for take in shot["takes"]:
                take["settings"] = self.rel_settings(take["settings"])
        for name in FOLDERS:
            (folder / name).mkdir(exist_ok=True)
        self._redo: list[dict[str, Any]] = []

    @property
    def id(self) -> str:
        return self.folder.name

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return copy.deepcopy(self.doc)

    def save_version(self, doc: dict[str, Any]) -> str:
        """Keep the state a version was assembled from (versions/<revision>.json), whatever the undo history keeps."""
        path = self.folder / "versions" / f"{doc['revision']:06d}.json"
        _write_json(path, doc)
        return self.rel(path)

    def version(self, deliverable_id: str) -> dict[str, Any]:
        """The state of the production when that version was assembled: its saved snapshot, or for a version made before snapshots were
        kept, its revision in the undo history while the history still holds it."""
        item = model.find(self.snapshot()["deliverables"], deliverable_id, "version")
        path = self.resolve(item.get("snapshot")) if item.get("snapshot") else self.folder / "history" / f"{int(item.get('revision') or -1):06d}.json"
        if path is None or not path.is_file():
            raise model.OperationError("This version was assembled before versions kept their state, and the undo history no longer holds it.")
        doc = json.loads(path.read_text(encoding="utf-8"))
        return model.normalize(doc.get("doc", doc) if "ops" in doc else doc)

    def apply(self, ops: list[dict[str, Any]], *, by: str = "user", base_revision: int | None = None, record: bool = True) -> list[dict[str, Any]]:
        with self.lock:
            previous = copy.deepcopy(self.doc)
            results = model.apply_ops(self.doc, ops, by=by, base_revision=base_revision)
            if record:
                _write_json(self.folder / "history" / f"{previous['revision']:06d}.json", {"by": by, "ops": [op.get("op") for op in ops], "doc": previous})
                self._trim_history()
                if by == "user":
                    self._redo.clear()
            self.save()
            revision = self.doc["revision"]
        self.store.publish({"type": "doc", "production": self.id, "revision": revision, "by": by, "ops": [op.get("op") for op in ops]})
        return results

    def save(self) -> None:
        with self.lock:
            _write_json(self.folder / DOCUMENT, self.doc)

    def _trim_history(self) -> None:
        files = sorted((self.folder / "history").glob("*.json"))
        for path in files[:-HISTORY_LIMIT]:
            path.unlink(missing_ok=True)

    def undo(self) -> int | None:
        """Restore the document as it was before the last change (shot media stay on disk); returns the new revision."""
        with self.lock:
            files = sorted((self.folder / "history").glob("*.json"))
            if not files:
                return None
            entry = json.loads(files[-1].read_text(encoding="utf-8"))
            files[-1].unlink()
            self.undone = {"by": entry.get("by"), "ops": entry.get("ops", [])}  # what the page tells the user it undid
            self._redo.append(copy.deepcopy(self.doc))
            restored = model.normalize(entry["doc"])
            restored["revision"], restored["id"], restored["director"] = self.doc["revision"] + 1, self.id, self.doc["director"]
            self.doc = restored
            self.save()
            revision = self.doc["revision"]
        self.store.publish({"type": "doc", "production": self.id, "revision": revision, "by": "user", "ops": ["undo"]})
        return revision

    def redo(self) -> int | None:
        with self.lock:
            if not self._redo:
                return None
            _write_json(self.folder / "history" / f"{self.doc['revision']:06d}.json", {"by": "user", "ops": ["redo"], "doc": copy.deepcopy(self.doc)})
            restored = self._redo.pop()
            restored["revision"], restored["director"] = self.doc["revision"] + 1, self.doc["director"]
            self.doc = restored
            self.save()
            revision = self.doc["revision"]
        self.store.publish({"type": "doc", "production": self.id, "revision": revision, "by": "user", "ops": ["redo"]})
        return revision

    # Paths: files inside the folder are stored relative to it, so that a copied folder keeps working.
    def rel(self, path: str | os.PathLike) -> str:
        path = Path(path).resolve()
        try:
            return path.relative_to(self.folder.resolve()).as_posix()
        except ValueError:
            return str(path)

    def adopt(self, path: str | os.PathLike) -> str:
        """The document path of a file the production uses: a file outside its folder (an agent's documents, another workspace) is copied into
        media/imports first, so that a duplicated, moved or archived production keeps everything it uses."""
        relative = self.rel(path)
        if not Path(relative).is_absolute():
            return relative
        from .media import import_file
        return self.rel(import_file(path, self.folder / "media" / "imports"))

    def resolve(self, path: str | None) -> Path | None:
        if not path:
            return None
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.folder / candidate

    def rel_settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """A take's generation settings as the document keeps them: media of the folder relative to it, so that reruns of a duplicated, moved or
        copied production read its own files."""
        def portable(path):
            relative = self.rel(path) if Path(path).is_absolute() else path
            return path if Path(relative).is_absolute() else relative  # a file outside the folder keeps its own spelling
        return _media_settings(settings, portable)

    def resolve_settings(self, settings: dict[str, Any]) -> dict[str, Any]:
        """A take's generation settings as WanGP runs them: media of the folder as full paths."""
        return _media_settings(settings, lambda path: str(self.folder / path) if not Path(path).is_absolute() and (self.folder / path).is_file() else path)

    def owns(self, path: str | os.PathLike) -> bool:
        """Whether a file may be served: inside the folder, or referenced by the document (takes, cast, brief, deliverables)."""
        resolved = Path(path).resolve()
        try:
            resolved.relative_to(self.folder.resolve())
            return True
        except ValueError:
            pass
        return str(resolved) in self._referenced()

    def _referenced(self) -> set[str]:
        with self.lock:
            doc = self.doc
            paths = [doc["brief"].get("song"), *doc["brief"].get("references", [])]
            for member in doc["cast"]:
                paths += [*member.get("images", []), member.get("master"), member.get("voice")]
            for shot in doc["shots"]:
                paths += [take.get("path") for take in shot["takes"]]
            paths += [item.get("path") for item in doc["deliverables"]] + [item.get("path") for item in doc.get("bin", [])]
            paths += [track.get("path") for track in doc["edit"].get("audio_tracks", [])]
        return {str(self.resolve(path).resolve()) for path in paths if path}


class ProductionStore:
    def __init__(self, root: str | os.PathLike):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._open: dict[str, Production] = {}
        self._lock = threading.RLock()
        self._listeners: list[Callable[[dict[str, Any]], None]] = []

    def subscribe(self, listener: Callable[[dict[str, Any]], None]) -> Callable[[], None]:
        with self._lock:
            self._listeners.append(listener)
        return lambda: self._listeners.remove(listener) if listener in self._listeners else None

    def publish(self, event: dict[str, Any]) -> None:
        for listener in list(self._listeners):
            try:
                listener(event)
            except Exception as error:  # a closed page must not stop the others
                print(f"[Showrunner] Event listener failed: {error}")

    def ids(self) -> list[str]:
        return sorted((path.name for path in self.root.iterdir() if (path / DOCUMENT).is_file()), reverse=True)

    def get(self, production_id: str) -> Production:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", production_id or "") or not (self.root / production_id / DOCUMENT).is_file():
            raise KeyError(f"No production has the id {production_id!r}.")
        with self._lock:
            if production_id not in self._open:
                self._open[production_id] = Production(self, self.root / production_id)
            return self._open[production_id]

    def create(self, title: str, kind: str = "free", **brief: Any) -> Production:
        with self._lock:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            production_id = f"{stamp}-{_slug(title)}"
            suffix = 1
            while (self.root / production_id).exists():
                suffix += 1
                production_id = f"{stamp}-{_slug(title)}-{suffix}"
            folder = self.root / production_id
            folder.mkdir()
            doc = model.new_document(title, kind, **brief)
            doc["id"] = production_id
            _write_json(folder / DOCUMENT, doc)
            production = self.get(production_id)
        self.publish({"type": "productions"})
        return production

    def duplicate(self, production_id: str, title: str | None = None, doc: dict[str, Any] | None = None) -> Production:
        """A copy of the production (doc: from that state of it, such as one of its versions), with its media; the showrunner starts afresh."""
        source = self.get(production_id)
        with source.lock:
            doc = copy.deepcopy(doc) if doc is not None else source.snapshot()
        copy_ = self.create(title or f"{doc['title']} (copy)", doc["brief"]["kind"])
        for name in ("media", "cache"):  # cache: the posters and filmstrips of the takes
            shutil.copytree(source.folder / name, copy_.folder / name, dirs_exist_ok=True)
        for item in doc["deliverables"]:  # the saved states of its versions (older ones from the undo history); the revision goes on from the copied one
            saved = source.folder / (item.get("snapshot") or f"history/{int(item.get('revision') or -1):06d}.json")
            if saved.is_file():
                item["snapshot"] = item.get("snapshot") or f"versions/{saved.name}"
                shutil.copy2(saved, copy_.folder / item["snapshot"])
        with copy_.lock:
            doc.update(id=copy_.id, title=copy_.doc["title"], director={"state": "idle"}, created=model.now(), updated=model.now(), archived=False)
            copy_.doc = model.normalize(doc)
            copy_.save()
        self.publish({"type": "productions"})
        return copy_

    def export(self, production_id: str, target: Path) -> Path:
        """The production as one .zip, to keep or to open in another WanGP (import_zip): its document, media, versions with their saved states,
        exports and posters. As in a duplicate, the showrunner's conversation and the undo history stay behind."""
        source = self.get(production_id)
        doc = source.snapshot()
        with zipfile.ZipFile(target, "w", allowZip64=True) as archive:
            for item in doc["deliverables"]:  # versions older than saved states travel with their state from the undo history
                saved = source.folder / f"history/{int(item.get('revision') or -1):06d}.json"
                if not item.get("snapshot") and saved.is_file():
                    item["snapshot"] = f"versions/{saved.name}"
                    archive.write(saved, item["snapshot"], zipfile.ZIP_DEFLATED)
            archive.writestr(DOCUMENT, json.dumps({**doc, "director": {"state": "idle"}}, indent=1, ensure_ascii=False), zipfile.ZIP_DEFLATED)
            for name in ("media", "cache", "versions", "exports"):
                for path in sorted((source.folder / name).rglob("*")):
                    if path.is_file():  # media are compressed already: stored as they are
                        archive.write(path, path.relative_to(source.folder).as_posix(), zipfile.ZIP_DEFLATED if path.suffix.lower() in (".json", ".srt", ".vtt", ".txt", ".svg") else zipfile.ZIP_STORED)
        return target

    def import_zip(self, path: Path) -> Production:
        """A production exported as a .zip, as a new production here (its own id; the showrunner starts afresh). The file comes from elsewhere:
        only a production's own folders are read from it, and nothing outside the new production's folder is written."""
        with zipfile.ZipFile(path) as archive:
            if DOCUMENT not in archive.namelist():
                raise ValueError("This .zip is not a Showrunner production: it has no production.json.")
            doc = json.loads(archive.read(DOCUMENT).decode("utf-8"))
            if not isinstance(doc, dict) or not isinstance(doc.get("brief"), dict):
                raise ValueError("Its production.json is not a Showrunner production.")
            production = self.create(str(doc.get("title") or "Imported production")[:80], str((doc.get("brief") or {}).get("kind") or "free"))
            folder = production.folder.resolve()
            try:
                for entry in archive.infolist():
                    top, target = entry.filename.split("/", 1)[0], (folder / entry.filename).resolve()
                    if entry.is_dir() or top not in ("media", "cache", "versions", "exports") or not target.is_relative_to(folder / top):
                        continue
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(entry) as source, open(target, "wb") as out:
                        shutil.copyfileobj(source, out, 1 << 20)
                with production.lock:
                    doc.update(id=production.id, title=production.doc["title"], director={"state": "idle"}, created=model.now(), updated=model.now(), archived=False)
                    production.doc = model.normalize(doc)
                    production.save()
            except Exception:  # a broken file leaves no half production behind
                with self._lock:
                    self._open.pop(production.id, None)
                shutil.rmtree(production.folder, ignore_errors=True)
                raise
        self.publish({"type": "productions"})
        return production

    def archive(self, production_id: str, archived: bool = True) -> None:
        """Hide a production from the home grid (or show it again): a flag of the production, outside its undo history."""
        production = self.get(production_id)
        with production.lock:
            production.doc["archived"] = bool(archived)
            production.save()
        self.publish({"type": "productions"})

    def trash(self, production_id: str) -> None:
        """Move a production to the store's .trash folder (recoverable by hand)."""
        production = self.get(production_id)
        with self._lock:
            self._open.pop(production_id, None)
            trash = self.root / ".trash"
            trash.mkdir(exist_ok=True)
            shutil.move(str(production.folder), str(trash / f"{production_id}-{int(time.time())}"))
        self.publish({"type": "productions"})

    def catalog(self) -> list[dict[str, Any]]:
        items = []
        for production_id in self.ids():
            try:
                doc = self.get(production_id).snapshot()
            except (OSError, ValueError, KeyError) as error:
                items.append({"id": production_id, "title": production_id, "error": str(error)})
                continue
            posters = [poster for shot in doc["shots"] if (poster := (model.chosen_take(shot) or {}).get("poster") or shot.get("source_poster"))][:4]  # a remake's shot without a take shows its source
            items.append({"id": production_id, "title": doc["title"], "kind": doc["brief"]["kind"], "stage": doc["stage"], "updated": doc["updated"],
                          "created": doc["created"], "shots": len(doc["shots"]), "approved": sum(shot["status"] == "approved" for shot in doc["shots"]),
                          "rendered": sum(bool(shot["takes"]) for shot in doc["shots"]), "duration_s": model.cut_length(doc),
                          "aspect": doc["brief"]["aspect"], "posters": posters, "director": doc["director"].get("state", "idle"), "archived": doc["archived"],
                          "deliverables": len(doc["deliverables"]), "latest": (doc["deliverables"] or [{}])[-1].get("path"),
                          "changed_since_delivery": bool(doc["deliverables"]) and doc["deliverables"][-1].get("edit_key") not in (None, model.edit_key(doc))})  # exports made before edit keys: unknown
        return items
