# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import atexit
import copy
import hashlib
import json
import logging
import math
import os
import re
import shutil
import tempfile
import threading
import urllib.parse
import uuid
import zipfile
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from shared.utils.gallery_media import disambiguate_gallery_media_ids, gallery_media_ids
from shared.utils.gallery_sources import MESH_EXTENSIONS, WORLD_EXTENSIONS
from shared.utils.readable_paths import available_path, safe_name


SESSION_SCHEMA_VERSION = 1
CONTEXT_SCHEMA_VERSION = 1
UI_JOURNAL_SCHEMA_VERSION = 1
DEFAULT_SESSIONS_FOLDER = "deepy_sessions"
MONO_SESSION_POINTER_FILENAME = ".mono-session"
UI_JOURNAL_FILENAME = "cards.jsonl"
GALLERY_MEDIA_LINK = "link"
GALLERY_MEDIA_COPY = "copy"
RESET_MODE_NEW = "new_session"
RESET_MODE_RESET = "reset_session"
_MEDIA_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".jfif", ".pjpeg", ".mkv", ".mov", ".mp4", ".m4v", ".webm", ".avi", ".wav", ".mp3", ".aac", ".m4a", ".flac", ".ogg", ".opus", ".wma"}
_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_WINDOWS_PATH_RE = re.compile(r"(?<!\w)(?:[A-Za-z]:[\\/]|\\\\)[^\s\]\[(){}<>\"']+")
_POSIX_PATH_RE = re.compile(r"(?<!\w)/(?:[^\s\]\[(){}<>\"']+/)*[^\s\]\[(){}<>\"']+")
_MARKDOWN_LINK_RE = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_ATTACHMENT_RE = re.compile(r"(?:@file\([^)]*\)|<attachment\b[^>]*>|\[attachment[^\]]*\])", re.IGNORECASE)
_SESSION_ID_RE = re.compile(r"^\d{8}-\d{6}-[a-f0-9]{8}$")
_MONO_SESSION_DIR_RE = re.compile(r"^mono-[a-f0-9]{32}$")
_ROOT_LOCK = threading.RLock()
_SESSIONS_ROOT = (Path.cwd() / DEFAULT_SESSIONS_FOLDER).resolve()
_WRITER = ThreadPoolExecutor(max_workers=1, thread_name_prefix="deepy-session-save")
_OWNED_LOCKS: dict[str, tuple[str, Path]] = {}
_UI_JOURNAL_STATES: dict[str, dict[str, Any]] = {}
# Sessions opened from folders outside the catalog (plugin sessions, parked conversations), by id.
_EXTERNAL: dict[str, Path] = {}


class SessionStoreError(RuntimeError):
    pass


class SessionLockedError(SessionStoreError):
    pass


def configure_sessions_root(path: str | os.PathLike[str] | None) -> Path:
    global _SESSIONS_ROOT
    candidate = Path(path).expanduser() if str(path or "").strip() else Path.cwd() / DEFAULT_SESSIONS_FOLDER
    with _ROOT_LOCK:
        _SESSIONS_ROOT = candidate.resolve()
        _UI_JOURNAL_STATES.clear()
    return _SESSIONS_ROOT


def sessions_root() -> Path:
    with _ROOT_LOCK:
        return _SESSIONS_ROOT


def ensure_mono_session_workspace(chat_session_id: str) -> Path:
    directory_name = f"mono-{str(chat_session_id or '').strip().lower()}"
    if not _MONO_SESSION_DIR_RE.fullmatch(directory_name):
        raise SessionStoreError("Invalid temporary Deepy session identifier.")
    with _ROOT_LOCK:
        root = _SESSIONS_ROOT
        root.mkdir(parents=True, exist_ok=True)
        pointer = root / MONO_SESSION_POINTER_FILENAME
        try:
            previous_name = pointer.read_text(encoding="utf-8").strip()
        except OSError:
            previous_name = ""
        if previous_name != directory_name and _MONO_SESSION_DIR_RE.fullmatch(previous_name):
            try:
                shutil.rmtree(root / previous_name)
            except OSError:
                pass
        workspace = root / directory_name / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        temporary = pointer.with_name(f".{pointer.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(directory_name, encoding="utf-8")
            os.replace(temporary, pointer)
        except OSError:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
        return workspace


def normalize_gallery_media_mode(value: Any) -> str:
    return GALLERY_MEDIA_COPY if str(value or "").strip().lower() == GALLERY_MEDIA_COPY else GALLERY_MEDIA_LINK


def normalize_reset_mode(value: Any) -> str:
    return RESET_MODE_RESET if str(value or "").strip().lower() == RESET_MODE_RESET else RESET_MODE_NEW


def automatic_title(first_request: str, max_words: int = 10, max_chars: int = 60) -> str:
    text = _ATTACHMENT_RE.sub(" ", str(first_request or ""))
    text = _MARKDOWN_LINK_RE.sub(lambda match: f" {match.group(1)} ", text)
    text = _URL_RE.sub(" ", text)
    text = _WINDOWS_PATH_RE.sub(" ", text)
    text = _POSIX_PATH_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip(" \t\r\n-–—:;,.!?…")
    if not text:
        return "Deepy session"
    clause = re.split(r"(?:[.!?…]+|\s+[–—]\s+|\n)", text, maxsplit=1)[0].strip()
    words = clause.split()
    title = " ".join(words[:max(1, int(max_words))]).strip()
    if len(title) > max_chars:
        title = title[:max_chars].rsplit(" ", 1)[0].rstrip(" -–—:;,.!?") or title[:max_chars].rstrip()
    return title or "Deepy session"


def _unique_title(title: str, storage_id: str = "") -> str:
    title = re.sub(r"\s+", " ", str(title or "")).strip()[:120]
    if not title:
        raise SessionStoreError("Session title cannot be empty.")
    used_titles = {str(item["title"]).casefold() for item in list_sessions() if item["id"] != storage_id}
    candidate = title
    number = 2
    while candidate.casefold() in used_titles:
        suffix = f" ({number})"
        candidate = title[:120 - len(suffix)].rstrip() + suffix
        number += 1
    return candidate


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _new_storage_id(root: Path | None = None) -> str:
    root = sessions_root() if root is None else root
    used = _session_manifests(root)
    while True:
        candidate = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}"
        if candidate not in used and not (root / candidate).exists():
            return candidate


def _session_manifests(root: Path | None = None) -> dict[str, tuple[Path, dict[str, Any]]]:
    root = sessions_root() if root is None else root
    records = {}
    if not root.is_dir():
        return records
    for directory in root.iterdir():
        if directory.name.startswith('.') or directory.name == '_exports' or not directory.is_dir() or directory.resolve() != directory:
            continue
        manifest_path = directory / 'session.json'
        if not manifest_path.is_file():
            continue
        try:
            manifest = _read_json(manifest_path)
        except SessionStoreError:
            continue
        identity = str(manifest.get('id', ''))
        if not _SESSION_ID_RE.fullmatch(identity):
            continue
        if identity in records:
            raise SessionStoreError(f'Duplicate Deepy session identifier: {identity}')
        records[identity] = (directory, manifest)
    return records


def _new_session_directory(root: Path, title: str) -> Path:
    name = safe_name(title)
    if _MONO_SESSION_DIR_RE.fullmatch(name):
        name = '_' + name
    reserved = ['_exports']
    for _directory, manifest in _session_manifests(root).values():
        reserved.extend(Path(path).name for path in manifest.get('previous_directories', []) if Path(path).parent == root)
    return available_path(root, name, reserved=reserved)


def _validated_session_dir(storage_id: str) -> Path:
    storage_id = str(storage_id or "").strip()
    if not _SESSION_ID_RE.fullmatch(storage_id):
        raise SessionStoreError("Invalid Deepy session identifier.")
    records = _session_manifests()
    if storage_id not in records:
        raise SessionStoreError(f"Deepy session does not exist: {storage_id}")
    return records[storage_id][0]


def _session_dir(storage_id: str) -> Path:
    with _ROOT_LOCK:
        external = _EXTERNAL.get(storage_id)
    return external if external is not None else _validated_session_dir(storage_id)


def open_external_session(directory: str | os.PathLike[str]) -> str | None:
    """Register a session folder kept outside the catalog. Returns its id, or None for an empty folder."""
    directory = Path(directory).expanduser().resolve()
    root = sessions_root()
    parts = directory.relative_to(root).parts if directory.is_relative_to(root) else None
    if parts is not None and (not parts or not parts[0].startswith('.')):
        raise SessionStoreError("Choose a session folder outside the Deepy sessions folder.")
    if not (directory / "session.json").is_file():
        if directory.exists() and any(directory.iterdir()):
            raise SessionStoreError(f"{directory} is not empty and holds no Deepy session.")
        return None
    storage_id = str(_read_json(directory / "session.json").get("id", ""))
    if not _SESSION_ID_RE.fullmatch(storage_id):
        raise SessionStoreError("Invalid Deepy session identifier.")
    with _ROOT_LOCK:
        if _EXTERNAL.get(storage_id, directory) != directory or storage_id in _session_manifests():
            raise SessionStoreError(f"Another Deepy session already uses the identifier {storage_id}.")
        _EXTERNAL[storage_id] = directory
    return storage_id


def close_external_session(storage_id: str) -> None:
    with _ROOT_LOCK:
        _EXTERNAL.pop(storage_id, None)


def session_folder_in_use(directory: Path) -> bool:
    lock_path = directory / ".session.lock"
    return lock_path.is_file() and _process_alive(int(_read_json(lock_path).get("pid", 0) or 0))


class SessionReferences:
    """Resolve persistent session IDs independently of their current folder names."""
    def __init__(self):
        self.records = _session_manifests()
        self.roots = {}
        for identity, (directory, manifest) in self.records.items():
            for root in [str(directory), *manifest.get('previous_directories', [])]:
                key = os.path.normcase(os.path.abspath(root))
                if key in self.roots and self.roots[key] != identity:
                    raise SessionStoreError(f'Ambiguous Deepy session folder: {root}')
                self.roots[key] = identity

    def encode(self, path):
        if isinstance(path, dict):
            self.validate(path)
            return path
        if not isinstance(path, str):
            raise SessionStoreError('Invalid session media path.')
        candidate = Path(os.path.abspath(path))
        for parent in candidate.parents:
            identity = self.roots.get(os.path.normcase(str(parent)))
            if identity is not None:
                reference = {'deepy_session_id': identity, 'path': candidate.relative_to(parent).as_posix()}
                return reference
        return path

    @staticmethod
    def validate(reference):
        if not isinstance(reference, dict) or set(reference) != {'deepy_session_id', 'path'}:
            raise SessionStoreError('Invalid Deepy session media reference.')
        identity, relative = reference['deepy_session_id'], reference['path']
        if not isinstance(identity, str) or not _SESSION_ID_RE.fullmatch(identity):
            raise SessionStoreError('Invalid Deepy session identifier in media reference.')
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or '..' in Path(relative.replace('\\', '/')).parts or ':' in relative or relative.startswith(('\\', '/')):
            raise SessionStoreError('Invalid relative Deepy session media path.')
        return identity, relative

    def resolve(self, reference):
        identity, relative = self.validate(reference)
        if identity not in self.records:
            return None
        try:
            return str(_session_member_path(self.records[identity][0], relative))
        except (OSError, SessionStoreError):
            # A replaced folder/link must never grant access outside this session.
            return None

    def path(self, value):
        reference = self.encode(value)
        return (self.resolve(reference) or reference) if isinstance(reference, dict) else reference

    def rebase(self, value):
        replacements = {}
        for _identity, (directory, manifest) in self.records.items():
            for original in manifest.get('previous_directories', []):
                if original == str(directory):
                    continue
                for before, after in ((original, str(directory)), (Path(original).as_posix(), directory.as_posix())):
                    # Include the separator so a sibling with the same prefix is untouched.
                    separator = '\\' if '\\' in before else '/'
                    replacements[before + separator] = after + separator
                    replacements[urllib.parse.quote(before + separator, safe='/')] = urllib.parse.quote(after + separator, safe='/')
                    replacements[json.dumps(before + separator, ensure_ascii=False)[1:-1]] = json.dumps(after + separator, ensure_ascii=False)[1:-1]
        return _replace_paths(value, replacements)


def migrate_session_folders(*, blocked=()):
    """One-time rename of ID-named folders; locked/failed folders stay usable in place."""
    logger = logging.getLogger('wangp.workspaces')
    renamed, skipped = [], []
    with _ROOT_LOCK:
        for identity, (directory, manifest) in _session_manifests().items():
            if directory.name != identity:
                continue
            if identity in blocked:
                skipped.append(identity)
                logger.warning('Deepy session folder rename skipped for %s: used as a dedicated workspace folder. No retry on later startups.', directory)
                continue
            lease = SimpleNamespace(session_lock_path='', session_lock_token='', pending_session_save=None, storage_session_dir='')
            try:
                _acquire_session_lock(lease, directory)
                destination = _new_session_directory(directory.parent, manifest['title'])
                aliases = list(dict.fromkeys([*manifest.get('previous_directories', []), str(directory)]))
                # Publish the old root before moving: interruption on either side is recoverable.
                _atomic_json(directory / 'session.json', {**manifest, 'previous_directories': aliases})
                directory.rename(destination)
                lease.session_lock_path = str(destination / '.session.lock')
                _OWNED_LOCKS[lease.session_lock_token] = (str(destination), Path(lease.session_lock_path))
                _forget_ui_journal(directory)
                renamed.append(identity)
            except (OSError, SessionStoreError) as error:
                skipped.append(identity)
                logger.warning('Deepy session folder rename skipped for %s: %s No retry on later startups.', directory, error)
            finally:
                try:
                    release_session_lock(lease)
                except OSError as error:
                    logger.warning('Cannot release Deepy migration lock for %s: %s', directory, error)
    return {'renamed': renamed, 'skipped': skipped}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(child) for child in value]
    if isinstance(value, os.PathLike):
        return str(value)
    return str(value)


def _tool_arguments_as_mappings(messages: list[Any]) -> list[Any]:
    """Sessions saved after a turn of an external engine hold its tool-call arguments as JSON text: the local engines' chat templates
    read them as a mapping."""
    for message in messages:
        for call in (message.get("tool_calls") or []) if isinstance(message, dict) else []:
            function = call.get("function") if isinstance(call, dict) else None
            if isinstance(function, dict) and isinstance(function.get("arguments"), str):
                try:
                    arguments = json.loads(function["arguments"] or "{}")
                except json.JSONDecodeError:
                    arguments = None
                function["arguments"] = arguments if isinstance(arguments, dict) else {}
    return messages


def _read_json(path: Path) -> dict[str, Any]:
    try:
        # Windows cannot replace a file while another thread has it open.
        with _ROOT_LOCK:
            content = path.read_text(encoding="utf-8")
        value = json.loads(content)
    except (OSError, ValueError) as exc:
        raise SessionStoreError(f"Cannot read Deepy session file: {path.name}") from exc
    if not isinstance(value, dict):
        raise SessionStoreError(f"Invalid Deepy session file: {path.name}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(_json_safe(value), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
        with _ROOT_LOCK:
            os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _clean_ui_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_ui_value(child) for key, child in value.items() if str(key) != "_published_text"}
    if isinstance(value, (list, tuple)):
        return [_clean_ui_value(child) for child in value]
    return _json_safe(value)


_REPLAY_EVENT_TYPES = {"upsert_message", "remove_message", "upsert_block", "remove_block"}
_LEGACY_CHAT_CLASS_PREFIX = "wangp-assistant-chat"
_CHAT_CLASS_ATTRIBUTE_RE = re.compile(r"(\bclass\s*=\s*)(?P<quote>['\"])(?P<classes>[^'\"]*)(?P=quote)", re.IGNORECASE)


def _compact_replay_html_classes(markup: str) -> str:
    if _LEGACY_CHAT_CLASS_PREFIX not in markup:
        return markup

    def compact_attribute(match: re.Match) -> str:
        classes = []
        for class_name in match.group("classes").split():
            if class_name == _LEGACY_CHAT_CLASS_PREFIX or class_name.startswith(f"{_LEGACY_CHAT_CLASS_PREFIX}__") or class_name.startswith(f"{_LEGACY_CHAT_CLASS_PREFIX}--"):
                class_name = f"chat{class_name[len(_LEGACY_CHAT_CLASS_PREFIX):]}"
            classes.append(class_name)
        quote = match.group("quote")
        return f"{match.group(1)}{quote}{' '.join(classes)}{quote}"

    return _CHAT_CLASS_ATTRIBUTE_RE.sub(compact_attribute, markup)


def _compact_replay_event_classes(event: dict[str, Any]) -> dict[str, Any]:
    if str(event.get("type", "") or "") == "upsert_message" and isinstance(event.get("message"), dict):
        message = event["message"]
        if isinstance(message.get("html"), str):
            message["html"] = _compact_replay_html_classes(message["html"])
    if str(event.get("type", "") or "") == "upsert_block" and isinstance(event.get("html"), str):
        event["html"] = _compact_replay_html_classes(event["html"])
    return event


def read_ui_updates(directory: str | os.PathLike, after: int = 0) -> dict[str, Any]:
    """What Deepy's window needs to show a saved conversation: the updates its journal gained since `after` (a position in the file, 0 for all of it),
    each message and block at its latest. A journal that went back (the conversation was rewound) starts again, with `restart`."""
    path = Path(directory) / UI_JOURNAL_FILENAME
    if not path.is_file():
        return {"next": 0, "restart": after > 0, "events": []}
    restart = after > path.stat().st_size
    with path.open("rb") as file:
        file.seek(0 if restart else after)
        data = file.read()
    data = data[:data.rfind(b"\n") + 1]  # a line still being written waits for the next call
    latest: dict[tuple, dict[str, Any]] = {}
    for line in data.splitlines():
        event = json.loads(line).get("event") or {}
        if event.get("type") == "upsert_message":  # it holds its blocks
            latest = {key: value for key, value in latest.items() if key[0] != event["message"]["id"]}
            latest[(event["message"]["id"], "")] = event
        elif event.get("type") == "upsert_block":
            latest[(event["message_id"], event["block_id"])] = event
        elif event.get("type") in ("remove_message", "remove_block"):
            latest[(event["message_id"], event.get("block_id") or "", "removed", len(latest))] = event
    return {"next": (0 if restart else after) + len(data), "restart": restart, "events": list(latest.values())}


def _read_ui_journal(directory: Path, *, max_sequence: int | None = None) -> tuple[list[dict[str, Any]], int, int]:
    path = directory / UI_JOURNAL_FILENAME
    if not path.is_file():
        return [], 0, 0
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise SessionStoreError("Cannot read Deepy cards journal.") from exc
    commands: list[dict[str, Any]] = []
    previous_sequence = 0
    applied_sequence = 0
    nonempty_indices = [index for index, line in enumerate(lines) if line.strip()]
    final_nonempty_index = nonempty_indices[-1] if nonempty_indices else -1
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            operation = json.loads(line)
        except ValueError as exc:
            if index == final_nonempty_index:
                break
            raise SessionStoreError("Deepy cards journal contains invalid JSON.") from exc
        if not isinstance(operation, dict):
            raise SessionStoreError("Deepy cards journal contains an invalid operation.")
        schema_version = int(operation.get("schema_version", 0) or 0)
        sequence = int(operation.get("sequence", 0) or 0)
        if schema_version > UI_JOURNAL_SCHEMA_VERSION:
            raise SessionStoreError("This Deepy cards journal was created by a newer format.")
        if sequence != previous_sequence + 1:
            raise SessionStoreError("Deepy cards journal sequence is not contiguous.")
        previous_sequence = sequence
        if max_sequence is not None and sequence > max_sequence:
            continue
        if str(operation.get("cmd", "") or "") != "chat_output" or not isinstance(operation.get("event"), dict):
            raise SessionStoreError("Deepy cards journal contains an invalid client command.")
        if str(operation["event"].get("type", "") or "") not in _REPLAY_EVENT_TYPES:
            raise SessionStoreError("Deepy cards journal contains an unsupported client command.")
        commands.append({"cmd": "chat_output", "event": _compact_replay_event_classes(_clean_ui_value(operation["event"]))})
        applied_sequence = sequence
    return commands, applied_sequence, previous_sequence


def _truncate_ui_journal(directory: Path, last_sequence: int) -> None:
    path = directory / UI_JOURNAL_FILENAME
    if not path.is_file():
        return
    try:
        with path.open("rb+") as journal:
            keep_offset = 0
            while True:
                line = journal.readline()
                if not line:
                    break
                if not line.strip():
                    continue
                try:
                    operation = json.loads(line.decode("utf-8"))
                    sequence = int(operation.get("sequence", 0) or 0) if isinstance(operation, dict) else 0
                except (UnicodeDecodeError, ValueError):
                    break
                if sequence > last_sequence:
                    break
                keep_offset = journal.tell()
            journal.seek(0, os.SEEK_END)
            changed = journal.tell() != keep_offset
            if changed:
                journal.truncate(keep_offset)
            if keep_offset > 0:
                journal.seek(keep_offset - 1)
                if journal.read(1) != b"\n":
                    journal.seek(keep_offset)
                    journal.write(b"\n")
                    changed = True
            if changed:
                journal.flush()
                os.fsync(journal.fileno())
    except OSError as exc:
        raise SessionStoreError("Cannot recover Deepy cards journal.") from exc


def _message_frame(record: dict[str, Any]) -> dict[str, Any]:
    return {key: _clean_ui_value(value) for key, value in record.items() if key != "blocks"}


def _message_replay_event(record: dict[str, Any], message_index: int) -> dict[str, Any]:
    from shared.gradio import assistant_chat

    return {"type": "upsert_message", "message": assistant_chat._render_message_payload(record), "message_index": int(message_index)}


def _block_replay_event(record: dict[str, Any], block: dict[str, Any], message_index: int, block_index: int) -> dict[str, Any]:
    from shared.gradio import assistant_chat

    block_type = str(block.get("type", "markdown") or "markdown")
    return {
        "type": "upsert_block",
        "message_id": str(record.get("id", "") or ""),
        "message": assistant_chat._message_frame_payload(record),
        "message_index": int(message_index),
        "block_id": str(block.get("id", "") or ""),
        "block_type": block_type,
        "block_index": int(block_index),
        "html": assistant_chat._render_block_html(record, block, streaming=False),
        "text": str(block.get("text", "") or "") if block_type in {"markdown", "reasoning", "context_summary", "context_thinking"} else "",
        "streaming": False,
    }


def _consolidated_ui_transcript(previous: list[dict[str, Any]], current: list[dict[str, Any]]) -> list[dict[str, Any]]:
    previous_by_id = {str(record.get("id", "")): record for record in previous if isinstance(record, dict) and str(record.get("id", ""))}
    consolidated = []
    for raw_record in current:
        if not isinstance(raw_record, dict):
            continue
        record = _clean_ui_value(raw_record)
        message_id = str(record.get("id", "") or "")
        previous_record = previous_by_id.get(message_id, {})
        previous_blocks = {str(block.get("id", "")): block for block in list(previous_record.get("blocks", []) or []) if isinstance(block, dict) and str(block.get("id", ""))}
        blocks = []
        for block in list(record.get("blocks", []) or []):
            if not isinstance(block, dict):
                continue
            block_id = str(block.get("id", "") or "")
            block_type = str(block.get("type", "markdown") or "markdown")
            incomplete_text = block_type in {"markdown", "reasoning", "context_summary", "context_thinking"} and bool(block.get("streaming", False))
            incomplete_tool = block_type == "tool" and (bool(block.get("request_pending", False)) or str(block.get("status", "") or "").strip().lower() not in {"done", "error", "failed", "interrupted", "cancelled"})
            if incomplete_text or incomplete_tool:
                if block_id in previous_blocks:
                    blocks.append(_clean_ui_value(previous_blocks[block_id]))
                continue
            blocks.append(block)
        record["blocks"] = blocks
        renderable = str(record.get("role", "") or "") != "assistant" or bool(blocks) or bool(record.get("attachments"))
        if renderable:
            consolidated.append(record)
    return consolidated


def _build_ui_replay_commands(previous: list[dict[str, Any]], current: list[dict[str, Any]]) -> list[dict[str, Any]]:
    commands = []
    previous_by_id = {str(record.get("id", "")): record for record in previous if isinstance(record, dict) and str(record.get("id", ""))}
    current_by_id = {str(record.get("id", "")): record for record in current if isinstance(record, dict) and str(record.get("id", ""))}
    for message_id in previous_by_id.keys() - current_by_id.keys():
        commands.append({"cmd": "chat_output", "event": {"type": "remove_message", "message_id": message_id}})
    previous_order = [str(record.get("id", "")) for record in previous if isinstance(record, dict) and str(record.get("id", ""))]
    previous_indices = {message_id: index for index, message_id in enumerate(previous_order)}
    for message_index, record in enumerate(current):
        message_id = str(record.get("id", "") or "").strip()
        if not message_id:
            continue
        previous_record = previous_by_id.get(message_id)
        moved = previous_indices.get(message_id, message_index) != message_index
        if previous_record is None or moved or _message_frame(record) != _message_frame(previous_record):
            commands.append({"cmd": "chat_output", "event": _message_replay_event(record, message_index)})
            continue
        previous_blocks = {str(block.get("id", "")): block for block in list(previous_record.get("blocks", []) or []) if isinstance(block, dict) and str(block.get("id", ""))}
        current_blocks = {str(block.get("id", "")): block for block in list(record.get("blocks", []) or []) if isinstance(block, dict) and str(block.get("id", ""))}
        for block_id in previous_blocks.keys() - current_blocks.keys():
            commands.append({"cmd": "chat_output", "event": {"type": "remove_block", "message_id": message_id, "block_id": block_id, "block_type": str(previous_blocks[block_id].get("type", "markdown") or "markdown")}})
        for block_index, block in enumerate(list(record.get("blocks", []) or [])):
            if not isinstance(block, dict):
                continue
            block_id = str(block.get("id", "") or "").strip()
            if block_id and _clean_ui_value(block) != _clean_ui_value(previous_blocks.get(block_id)):
                commands.append({"cmd": "chat_output", "event": _block_replay_event(record, block, message_index, block_index)})
    return commands


def _append_ui_journal(directory: Path, transcript: list[dict[str, Any]]) -> tuple[int, int, list[dict[str, Any]]]:
    cache_key = str(directory.resolve())
    state = _UI_JOURNAL_STATES.get(cache_key)
    if state is None:
        previous = []
        committed_sequence = 0
        context_path = directory / "context.json"
        if context_path.is_file():
            stored_context = _read_json(context_path)
            stored_transcript = list(stored_context.get("chat", {}).get("transcript", []) or [])
            replay = stored_context.get("ui", {}).get("replay", {})
            if replay and not isinstance(replay, dict):
                raise SessionStoreError("Invalid Deepy UI replay descriptor.")
            if isinstance(replay, dict) and replay:
                if str(replay.get("path", "") or "") != UI_JOURNAL_FILENAME:
                    raise SessionStoreError("Invalid Deepy UI replay journal path.")
                committed_sequence = int(replay.get("last_sequence", 0) or 0)
                _commands, applied_sequence, _physical_sequence = _read_ui_journal(directory, max_sequence=committed_sequence)
                if applied_sequence != committed_sequence:
                    raise SessionStoreError("Deepy UI replay journal is incomplete.")
                previous = stored_transcript
                if committed_sequence == 0 and _consolidated_ui_transcript([], previous):
                    previous = []
        _truncate_ui_journal(directory, committed_sequence)
        state = {"transcript": _clean_ui_value(previous), "sequence": committed_sequence}
    current = _consolidated_ui_transcript(state["transcript"], transcript)
    commands = _build_ui_replay_commands(state["transcript"], current)
    sequence = int(state["sequence"] or 0)
    if commands:
        with (directory / UI_JOURNAL_FILENAME).open("a", encoding="utf-8", newline="\n") as writer:
            for command in commands:
                sequence += 1
                payload = {"schema_version": UI_JOURNAL_SCHEMA_VERSION, "sequence": sequence, **_clean_ui_value(command)}
                writer.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            writer.flush()
            os.fsync(writer.fileno())
    _UI_JOURNAL_STATES[cache_key] = {"transcript": current, "sequence": sequence}
    return sequence, len(current), current


def _forget_ui_journal(directory: Path) -> None:
    _UI_JOURNAL_STATES.pop(str(directory.resolve()), None)


def _process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        open_process.restype = wintypes.HANDLE
        get_exit_code = kernel32.GetExitCodeProcess
        get_exit_code.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
        get_exit_code.restype = wintypes.BOOL
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = (wintypes.HANDLE,)
        close_handle.restype = wintypes.BOOL
        handle = open_process(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not handle:
            error = ctypes.get_last_error()
            if error == 5:  # ERROR_ACCESS_DENIED: the process exists but cannot be queried.
                return True
            if error == 87:  # ERROR_INVALID_PARAMETER: no process owns this PID.
                return False
            raise ctypes.WinError(error)
        exit_code = wintypes.DWORD()
        try:
            if not get_exit_code(handle, ctypes.byref(exit_code)):
                raise ctypes.WinError(ctypes.get_last_error())
            return exit_code.value == 259  # STILL_ACTIVE
        finally:
            close_handle(handle)
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return True
    except OSError:
        return False


def _acquire_session_lock(session, directory: Path) -> None:
    current_path = str(session.session_lock_path or "")
    if current_path and Path(current_path) == directory / ".session.lock":
        return
    lock_path = directory / ".session.lock"
    token = uuid.uuid4().hex
    payload = {"pid": os.getpid(), "token": token, "created_at": _utc_now()}
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            existing = _read_json(lock_path)
            existing_pid = int(existing.get("pid", 0) or 0)
        except Exception:
            existing_pid = 0
        if _process_alive(existing_pid):
            raise SessionLockedError(f"Deepy session is already open by process {existing_pid}.")
        lock_path.unlink(missing_ok=True)
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as writer:
        json.dump(payload, writer)
    session.session_lock_path = str(lock_path)
    session.session_lock_token = token
    with _ROOT_LOCK:
        _OWNED_LOCKS[token] = (str(directory), lock_path)


def release_session_lock(session) -> None:
    pending = session.pending_session_save
    if isinstance(pending, Future) and not pending.done():
        try:
            pending.result()
        except Exception as exc:
            session.session_save_error = str(exc)
    lock_path = Path(str(session.session_lock_path)) if str(session.session_lock_path or "") else None
    token = str(session.session_lock_token or "")
    if lock_path is not None and lock_path.is_file():
        try:
            payload = _read_json(lock_path)
        except SessionStoreError:
            payload = {}
        if str(payload.get("token", "")) == token:
            lock_path.unlink(missing_ok=True)
    with _ROOT_LOCK:
        _OWNED_LOCKS.pop(token, None)
    if str(session.storage_session_dir or ""):
        _forget_ui_journal(Path(session.storage_session_dir))
    session.session_lock_path = ""
    session.session_lock_token = ""


def _release_all_locks() -> None:
    with _ROOT_LOCK:
        owned = list(_OWNED_LOCKS.items())
        _OWNED_LOCKS.clear()
    for token, (_directory, lock_path) in owned:
        try:
            payload = _read_json(lock_path)
            if str(payload.get("token", "")) == token:
                lock_path.unlink(missing_ok=True)
        except Exception:
            pass


atexit.register(_release_all_locks)


def bind_session_persistence(session) -> None:
    session.safe_checkpoint_callback = schedule_autosave


def ensure_session(session, first_request: str, deepy_type: str, gallery_media_mode: str, environment: dict[str, Any] | None = None, *, media_import=False, directory: Path | None = None) -> dict[str, Any]:
    deepy_type = str(deepy_type or "").strip().lower()
    if session.storage_session_id:
        if session.storage_deepy_type != deepy_type:
            raise SessionStoreError("A Deepy session cannot change between Prime and Zero.")
        session.gallery_media_mode = normalize_gallery_media_mode(gallery_media_mode or session.gallery_media_mode)
        session.session_environment = _json_safe(environment or session.session_environment)
        if session.storage_title_pending and str(first_request or '').strip():
            session.storage_title = _unique_title(automatic_title(first_request), session.storage_session_id)
            session.storage_title_pending = False
        bind_session_persistence(session)
        return session_metadata(session)
    request = str(first_request or "").strip()
    if not request and not media_import:
        raise SessionStoreError("A Deepy session is created only when its first request is sent.")
    root = sessions_root()
    root.mkdir(parents=True, exist_ok=True)
    with _ROOT_LOCK:
        storage_id = _new_storage_id(root)
        title = session.storage_title or ('New Deepy session' if media_import else automatic_title(request))
        if directory is None:
            title = _unique_title(title, storage_id)
            directory = _new_session_directory(root, title)
            directory.mkdir()
        else:
            # An external folder (see open_external_session) keeps its caller's title.
            directory.mkdir(parents=True, exist_ok=True)
            _EXTERNAL[storage_id] = directory
    (directory / "workspace").mkdir()
    _acquire_session_lock(session, directory)
    now = _utc_now()
    session.storage_session_id = storage_id
    session.storage_session_dir = str(directory)
    session.storage_title_pending = media_import and not session.storage_title
    session.storage_title = title
    session.storage_deepy_type = deepy_type
    session.storage_created_at = now
    session.storage_updated_at = now
    session.gallery_media_mode = normalize_gallery_media_mode(gallery_media_mode)
    session.session_environment = _json_safe(environment or {})
    session.chat_session_id = storage_id
    _atomic_json(directory / 'session.json', {'schema_version': SESSION_SCHEMA_VERSION, **session_metadata(session)})
    bind_session_persistence(session)
    return session_metadata(session)


def session_metadata(session) -> dict[str, Any]:
    return {
        "id": str(session.storage_session_id or ""),
        "title": str(session.storage_title or ""),
        **({"title_pending": True} if session.storage_title_pending else {}),
        "deepy_type": str(session.storage_deepy_type or ""),
        "created_at": str(session.storage_created_at or ""),
        "updated_at": str(session.storage_updated_at or ""),
        "gallery_media_mode": normalize_gallery_media_mode(session.gallery_media_mode),
        **({"gallery_workspace_id": session.gallery_workspace_id} if session.gallery_workspace_id else {}),
    }


def session_workspace(session) -> Path | None:
    if not session.storage_session_dir:
        return None
    return Path(session.storage_session_dir) / "workspace"


def _pending_action_from_runtime(runtime: Any) -> dict[str, Any] | None:
    if not isinstance(runtime, dict):
        raise SessionStoreError("Invalid Deepy runtime context.")
    replay = runtime.get("pending_action")
    if replay is None:
        return None
    if not isinstance(replay, dict) or int(replay.get("schema_version", 0) or 0) != 1:
        raise SessionStoreError("Invalid Deepy pending-action replay descriptor.")
    required = {
        "phase", "completion_prefix", "generation_state", "max_new_tokens", "seed", "do_sample", "temperature", "top_p", "top_k", "thinking_enabled",
        "user_message_id", "user_text", "turn_messages_len", "assistant_message_id", "assistant_badge", "completed_thought_content",
        "selected_visual_media_snapshot", "selected_audio_media_snapshot", "stream_answer_text", "stream_answer_block_id", "stream_reasoning_text",
        "stream_reasoning_block_id", "recent_steps", "pending_natural_thought", "loop_answer_checkpoint", "model_passes", "incomplete_stop_retries",
    }
    if not required.issubset(replay) or str(replay["phase"]) not in {"thought", "statement", "tool"} or not isinstance(replay["generation_state"], dict):
        raise SessionStoreError("Incomplete Deepy pending-action replay descriptor.")
    generation_state = replay["generation_state"]
    if not {"sampling_enabled", "sampling_state", "presence"}.issubset(generation_state) or not isinstance(generation_state["sampling_state"], list):
        raise SessionStoreError("Invalid Deepy pending-action generation state.")
    return _json_safe(copy.deepcopy(replay))


def _capture_snapshot(session) -> dict[str, Any] | None:
    if not session.storage_session_id or not session.storage_session_dir:
        return None
    with session.turn_lock:
        checkpoint = session.current_turn
        pending_action = copy.deepcopy(session.pending_action_replay)
        if pending_action is not None:
            saved_messages = copy.deepcopy(session.pending_action_replay_messages)
            saved_transcript = copy.deepcopy(session.pending_action_replay_transcript)
        elif isinstance(checkpoint, dict):
            message_limit = max(0, min(int(checkpoint.get("persisted_messages_len", len(session.messages)) or 0), len(session.messages)))
            saved_messages = copy.deepcopy(session.messages[:message_limit])
            completed_thought = str(checkpoint.get("completed_thought_content", "") or "").strip()
            if completed_thought:
                saved_messages.append({"role": "assistant", "content": completed_thought})
            saved_transcript = copy.deepcopy(session.chat_transcript)
        else:
            saved_messages = copy.deepcopy(session.messages)
            saved_transcript = copy.deepcopy(session.chat_transcript)
        context = {
            "schema_version": CONTEXT_SCHEMA_VERSION,
            "session_id": session.storage_session_id,
            "chat": {
                "messages": saved_messages,
                "transcript": saved_transcript,
                "turn_durations": dict(session.chat_turn_durations),
                "transcript_counter": int(session.chat_transcript_counter or 0),
                "revision": int(session.chat_revision or 0),
                "interruption_notice": str(session.interruption_notice or ""),
                "interruption_history": copy.deepcopy(session.interruption_history),
            },
            "media": copy.deepcopy(session.media_registry),
            "ui": {"tool_settings": copy.deepcopy(session.tool_ui_settings), "model_selection_runtime_signature": session.model_selection_runtime_signature},
            "runtime": {
                "pending_chat_media": copy.deepcopy(session.pending_chat_media),
                "generated_client_ids": list(session.generated_client_ids),
                "selected_visual_signature": str(session.selected_visual_runtime_signature or ""),
                "selected_audio_signature": str(session.selected_audio_runtime_signature or ""),
                "selected_mesh_signature": str(session.selected_mesh_runtime_signature or ""),
                "selected_world_signature": str(session.selected_world_runtime_signature or ""),
                "pending_action": pending_action,
            },
            "environment": copy.deepcopy(session.session_environment),
            "workspace": {"path": "workspace", "media": []},
        }
        revision = int(session.safe_checkpoint_revision or 0)
        metadata = session_metadata(session)
        lock_token = str(session.session_lock_token or "")
    return {"directory": str(session.storage_session_dir), "context": _json_safe(context), "metadata": metadata, "revision": revision, "lock_token": lock_token}


def _fingerprint(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_size:x}:{stat.st_mtime_ns:x}"


def _workspace_media(directory: Path) -> list[dict[str, Any]]:
    workspace = directory / "workspace"
    if not workspace.is_dir():
        return []
    sandbox = workspace / "sandbox"  # the Python sandbox's scratch (frames, work files) stays private; its deliverables are published to the outputs
    records = []
    for path in sorted(workspace.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _MEDIA_EXTENSIONS or path.is_relative_to(sandbox):
            continue
        relative = path.relative_to(directory).as_posix()
        digest = hashlib.sha1(relative.casefold().encode("utf-8")).hexdigest()[:12]
        records.append({"media_id": f"workspace_{digest}", "media_type": _detect_media_type(path), "path": str(path), "path_key": os.path.normcase(str(path.resolve())), "source": "workspace", "client_id": "", "settings": {}, "label": path.stem.replace("_", " "), "prompt_summary": "", "prompt": "", "filename": path.name, "access": ["write"], "fingerprint": _fingerprint(path), "session_relative_path": relative})
    return records


def _prepare_media(directory: Path, records: list[dict[str, Any]], mode: str) -> list[dict[str, Any]]:
    prepared = []
    known_paths = set()
    references = SessionReferences()
    for raw_record in records:
        if not isinstance(raw_record, dict) or raw_record.get("source") == "workspace":  # rescanned from the workspace folder below
            continue
        record = _json_safe(copy.deepcopy(raw_record))
        accesses = {str(value).strip().lower() for value in list(record.get("access", []) or [])}
        if not accesses and str(record.get("source", "")).strip().lower() == "deepy":
            accesses.add("write")
        if not accesses:
            continue
        record["access"] = sorted(accesses)
        raw_path = str(record.get("path", "") or "").strip()
        path = Path(raw_path) if raw_path else None
        reference = references.encode(raw_path) if raw_path else None
        if isinstance(reference, dict):
            record['session_reference'] = reference
        elif path is not None and path.is_file():
            record.pop('session_reference', None)
        if path is not None and path.is_file():
            try:
                record["fingerprint"] = _fingerprint(path)
            except OSError:
                pass
            if mode == GALLERY_MEDIA_COPY and str(record.get("source", "")) != "workspace":
                safe_id = re.sub(r"[^A-Za-z0-9_-]+", "_", str(record.get("media_id", "media"))).strip("_") or "media"
                destination = directory / "media" / f"{safe_id}_{path.name}"
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.is_file() or destination.stat().st_size != path.stat().st_size or destination.stat().st_mtime_ns != path.stat().st_mtime_ns:
                    shutil.copy2(path, destination)
                record["session_copy_path"] = destination.relative_to(directory).as_posix()
        path_key = os.path.normcase(str(path.resolve())) if path is not None else ""
        if path_key and path_key in known_paths:
            continue
        if path_key:
            known_paths.add(path_key)
        prepared.append(record)
    for record in _workspace_media(directory):
        if record["path_key"] not in known_paths:
            # Folders outside the catalog have no references; their workspace media keep a session-relative path.
            reference = references.encode(record['path'])
            if isinstance(reference, dict):
                record['session_reference'] = reference
            prepared.append(record)
    return prepared


def _preserve_gallery_media_ids(context: dict[str, Any], *, rebind_paths: bool = False) -> None:
    referenced_ids = list(dict.fromkeys(re.findall(r"(?:visual|audio):[a-f0-9]{12}", json.dumps(context.get("chat", {})), re.IGNORECASE)))
    items = []
    for record in context.get("media", []):
        if not isinstance(record, dict) or not record.get("path"):
            continue
        settings = record["settings"] = dict(record.get("settings") or {})
        gallery = "audio" if record.get("media_type") == "audio" else "visual"
        ids = gallery_media_ids(record["path"], gallery)
        if rebind_paths:
            settings["gallery_media_ids"] = list(dict.fromkeys([*settings["gallery_media_ids"], *ids]))
        elif not settings.get("gallery_media_ids"):
            # Legacy contexts may have saved the absolute path after issuing an ID for its relative spelling.
            settings["gallery_media_ids"] = list(dict.fromkeys([*(media_id.lower() for media_id in referenced_ids if media_id.lower() in ids), *ids]))
        items.append((record["path"], gallery, settings))
    for (_path, _gallery, settings), ids in zip(items, disambiguate_gallery_media_ids(items)):
        settings["gallery_media_ids"] = ids


def _write_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    directory = Path(snapshot["directory"])
    if not directory.is_dir():
        raise SessionStoreError("Deepy session directory disappeared while it was being saved.")
    lock_path = directory / ".session.lock"
    lock = _read_json(lock_path)
    if str(lock.get("token", "")) != snapshot["lock_token"]:
        raise SessionLockedError("Deepy session lock changed before the save completed.")
    context = snapshot["context"]
    metadata = snapshot["metadata"]
    replay_sequence, card_count, consolidated_transcript = _append_ui_journal(directory, list(context["chat"].get("transcript", []) or []))
    context["chat"]["transcript"] = consolidated_transcript
    context["ui"]["replay"] = {"schema_version": UI_JOURNAL_SCHEMA_VERSION, "path": UI_JOURNAL_FILENAME, "last_sequence": replay_sequence, "card_count": card_count}
    mode = normalize_gallery_media_mode(metadata["gallery_media_mode"])
    media = _prepare_media(directory, list(context.get("media", []) or []), mode)
    context["media"] = media
    _preserve_gallery_media_ids(context)
    context["workspace"]["media"] = [record for record in media if record.get("source") == "workspace"]
    updated_at = _utc_now()
    manifest = {
        "schema_version": SESSION_SCHEMA_VERSION,
        "id": metadata["id"],
        "title": metadata["title"],
        **({"title_pending": True} if metadata.get("title_pending") else {}),
        "deepy_type": metadata["deepy_type"],
        "created_at": metadata["created_at"],
        "updated_at": updated_at,
        "gallery_media_mode": mode,
        **({"gallery_workspace_id": metadata["gallery_workspace_id"]} if metadata.get("gallery_workspace_id") else {}),
        "context_revision": int(snapshot["revision"]),
        "message_count": len(context["chat"]["messages"]),
        "card_count": card_count,
        "replay_sequence": replay_sequence,
        "media_count": len(media),
    }
    previous_directories = _read_json(directory / 'session.json').get('previous_directories', [])
    if previous_directories:
        manifest['previous_directories'] = previous_directories
    context["saved_at"] = updated_at
    context["checkpoint_revision"] = int(snapshot["revision"])
    _atomic_json(directory / "context.json", context)
    _atomic_json(directory / "session.json", manifest)
    return {"manifest": manifest, "revision": int(snapshot["revision"]), "replay_sequence": replay_sequence}


def _save_completed(session, storage_id: str, future: Future) -> None:
    try:
        result = future.result()
    except Exception as exc:
        if session.storage_session_id == storage_id:
            session.session_save_error = str(exc)
            print(f"[Assistant] Continuous session save failed: {exc}")
    else:
        if session.storage_session_id == storage_id == result["manifest"]["id"]:
            session.storage_updated_at = str(result["manifest"]["updated_at"])
            session.saved_checkpoint_revision = max(int(session.saved_checkpoint_revision or 0), int(result["revision"]))
            session.ui_replay_sequence = int(result.get("replay_sequence", getattr(session, "ui_replay_sequence", 0)) or 0)
            session.session_save_error = ""


def schedule_autosave(session) -> Future | None:
    snapshot = _capture_snapshot(session)
    if snapshot is None:
        return None
    future = _WRITER.submit(_write_snapshot, snapshot)
    session.pending_session_save = future
    storage_id = str(snapshot["metadata"]["id"])
    future.add_done_callback(lambda completed: _save_completed(session, storage_id, completed))
    return future


def flush_session(session) -> dict[str, Any] | None:
    future = schedule_autosave(session)
    if future is None:
        return None
    result = future.result()
    return result["manifest"]


def list_sessions(deepy_type: str | None = None) -> list[dict[str, Any]]:
    root = sessions_root()
    if not root.is_dir():
        return []
    normalized_type = str(deepy_type or "").strip().lower()
    sessions = []
    for _directory, manifest in _session_manifests(root).values():
        if manifest.get("archived_at"):
            continue
        if normalized_type and str(manifest.get("deepy_type", "")).strip().lower() != normalized_type:
            continue
        sessions.append({key: manifest.get(key) for key in ("id", "title", "deepy_type", "created_at", "updated_at", "gallery_media_mode", "message_count", "media_count")})
    sessions.sort(key=lambda item: str(item.get("updated_at", "")), reverse=True)
    return sessions


def _replace_paths(value: Any, replacements: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: _replace_paths(child, replacements) for key, child in value.items()}
    if isinstance(value, list):
        return [_replace_paths(child, replacements) for child in value]
    if isinstance(value, str):
        if value in replacements:
            return replacements[value]
        updated = value
        for original, replacement in replacements.items():
            updated = updated.replace(original, replacement)
        return updated
    return value


def _rebind_session_references(value, old_id, new_id):
    if isinstance(value, dict):
        if set(value) == {'deepy_session_id', 'path'} and value['deepy_session_id'] == old_id:
            return {**value, 'deepy_session_id': new_id}
        return {key: _rebind_session_references(child, old_id, new_id) for key, child in value.items()}
    if isinstance(value, list):
        return [_rebind_session_references(child, old_id, new_id) for child in value]
    return value


def _register_path_replacement(replacements: dict[str, str], original: str, replacement: str) -> None:
    if not original or original == replacement:
        return
    replacements[original] = replacement
    normalized_original = os.path.normpath(original).replace("\\", "/")
    normalized_replacement = os.path.normpath(replacement).replace("\\", "/")
    replacements[normalized_original] = normalized_replacement
    replacements[urllib.parse.quote(normalized_original, safe="/")] = urllib.parse.quote(normalized_replacement, safe="/")
    replacements[json.dumps(original, ensure_ascii=False)[1:-1]] = json.dumps(replacement, ensure_ascii=False)[1:-1]


def _session_member_path(directory: Path, relative_path: str) -> Path:
    candidate = (directory / str(relative_path or "").replace("\\", "/")).resolve()
    try:
        candidate.relative_to(directory)
    except ValueError as exc:
        raise SessionStoreError("Deepy session context contains an unsafe relative path.") from exc
    return candidate


def _session_replay_commands(directory: Path, context: dict[str, Any]) -> tuple[list[dict[str, Any]], int]:
    replay = context.get("ui", {}).get("replay", {})
    if replay and not isinstance(replay, dict):
        raise SessionStoreError("Invalid Deepy UI replay descriptor.")
    if isinstance(replay, dict) and replay:
        if str(replay.get("path", "") or "") != UI_JOURNAL_FILENAME:
            raise SessionStoreError("Invalid Deepy UI replay journal path.")
        expected_sequence = int(replay.get("last_sequence", 0) or 0)
        commands, applied_sequence, _last_sequence = _read_ui_journal(directory, max_sequence=expected_sequence)
        if applied_sequence != expected_sequence:
            raise SessionStoreError("Deepy UI replay journal is incomplete.")
        return commands, applied_sequence
    transcript = list(context.get("chat", {}).get("transcript", []) or [])
    return _build_ui_replay_commands([], _consolidated_ui_transcript([], transcript)), 0


def validate_session(storage_id: str, deepy_type: str, active_session=None) -> dict[str, Any]:
    directory = _session_dir(storage_id)
    manifest = _read_json(directory / "session.json")
    if manifest.get("archived_at"):
        raise SessionStoreError("Restore this archived session and its workspace before resuming it.")
    context = _read_json(directory / "context.json")
    if str(manifest.get("id", "")) != storage_id or str(context.get("session_id", "")) != storage_id:
        raise SessionStoreError("Deepy session identifiers do not match its directory.")
    if int(manifest.get("schema_version", 0) or 0) > SESSION_SCHEMA_VERSION or int(context.get("schema_version", 0) or 0) > CONTEXT_SCHEMA_VERSION:
        raise SessionStoreError("This Deepy session was created by a newer format.")
    requested_type = str(deepy_type or "").strip().lower()
    stored_type = str(manifest.get("deepy_type", "")).strip().lower()
    if requested_type != stored_type:
        raise SessionStoreError(f"This is a Deepy {stored_type.title()} session and cannot be opened in Deepy {requested_type.title()}.")
    if not isinstance(context.get("chat", {}), dict):
        raise SessionStoreError("Invalid Deepy session chat context.")
    _pending_action_from_runtime(context.get("runtime", {}))
    _session_replay_commands(directory, context)
    for record in list(context.get("media", []) or []):
        if not isinstance(record, dict):
            continue
        if 'session_reference' in record:
            SessionReferences.validate(record['session_reference'])
        for key in ("session_relative_path", "session_copy_path"):
            relative_path = str(record.get(key, "") or "")
            if relative_path:
                _session_member_path(directory, relative_path)
    lock_path = directory / ".session.lock"
    active_lock = active_session is not None and active_session.storage_session_id == storage_id and Path(str(active_session.session_lock_path or "")) == lock_path
    if lock_path.is_file() and not active_lock:
        lock = _read_json(lock_path)
        if _process_alive(int(lock.get("pid", 0) or 0)):
            raise SessionLockedError(f"Deepy session is already open by process {int(lock.get('pid', 0) or 0)}.")
    return manifest


def load_session(session, storage_id: str, deepy_type: str) -> dict[str, Any]:
    directory = _session_dir(storage_id)
    manifest = _read_json(directory / "session.json")
    if manifest.get("archived_at"):
        raise SessionStoreError("Restore this archived session and its workspace before resuming it.")
    context = _read_json(directory / "context.json")
    if str(manifest.get("id", "")) != storage_id or str(context.get("session_id", "")) != storage_id:
        raise SessionStoreError("Deepy session identifiers do not match its directory.")
    if int(manifest.get("schema_version", 0) or 0) > SESSION_SCHEMA_VERSION or int(context.get("schema_version", 0) or 0) > CONTEXT_SCHEMA_VERSION:
        raise SessionStoreError("This Deepy session was created by a newer format.")
    requested_type = str(deepy_type or "").strip().lower()
    stored_type = str(manifest.get("deepy_type", "")).strip().lower()
    if requested_type != stored_type:
        raise SessionStoreError(f"This is a Deepy {stored_type.title()} session and cannot be opened in Deepy {requested_type.title()}.")
    _preserve_gallery_media_ids(context)
    references = SessionReferences()
    context = references.rebase(context)
    replacements = {}
    media_records = list(context.get("media", []) or [])
    missing_media = []
    for record in media_records:
        if not isinstance(record, dict):
            continue
        original = str(record.get("path", "") or record.get('unavailable_path', '') or "")
        if 'session_reference' in record:
            resolved = references.resolve(record['session_reference'])
            if resolved and Path(resolved).is_file():
                if original:
                    _register_path_replacement(replacements, original, resolved)
                record['path'] = resolved
                record['path_key'] = os.path.normcase(resolved)
                record.pop('unavailable_path', None)
                continue
        session_relative = str(record.get("session_relative_path", "") or "")
        if session_relative:
            relative_path = _session_member_path(directory, session_relative)
            if relative_path.is_file():
                if original and original != str(relative_path):
                    _register_path_replacement(replacements, original, str(relative_path))
                record["path"] = str(relative_path)
                record["path_key"] = os.path.normcase(str(relative_path))
                continue
        if 'session_reference' not in record and original and Path(original).is_file():
            continue
        copied = str(record.get("session_copy_path", "") or "")
        copied_path = _session_member_path(directory, copied) if copied else None
        if copied_path is not None and copied_path.is_file():
            if original:
                _register_path_replacement(replacements, original, str(copied_path))
            record["path"] = str(copied_path)
            record["path_key"] = os.path.normcase(str(copied_path.resolve()))
        elif original:
            missing_media.append(original)
            if 'session_reference' in record:
                record['unavailable_path'] = original
                record['path'] = record['path_key'] = ''
    context = _replace_paths(context, replacements)
    _preserve_gallery_media_ids(context, rebind_paths=True)
    replay_commands, replay_sequence = _session_replay_commands(directory, context)
    replay_commands = references.rebase(_replace_paths(replay_commands, replacements))
    chat = context.get("chat", {})
    if not isinstance(chat, dict):
        raise SessionStoreError("Invalid Deepy session chat context.")
    durations = chat.get("turn_durations", {})
    if not isinstance(durations, dict) or any(not isinstance(key, str) or type(value) not in (int, float) or not math.isfinite(value) or value < 0 for key, value in durations.items()):
        raise SessionStoreError("Invalid Deepy session turn durations.")
    _acquire_session_lock(session, directory)
    session.storage_session_id = storage_id
    session.storage_session_dir = str(directory)
    session.storage_title = str(manifest.get("title", "") or "Deepy session")
    session.storage_title_pending = bool(manifest.get("title_pending", False))
    session.storage_deepy_type = stored_type
    session.storage_created_at = str(manifest.get("created_at", "") or "")
    session.storage_updated_at = str(manifest.get("updated_at", "") or "")
    session.gallery_media_mode = normalize_gallery_media_mode(manifest.get("gallery_media_mode"))
    session.gallery_workspace_id = str(manifest.get("gallery_workspace_id", "") or "")
    session.session_environment = _json_safe(context.get("environment", {}))
    session.chat_session_id = storage_id
    session.messages = _tool_arguments_as_mappings(list(chat.get("messages", []) or []))
    session.chat_turn_durations = dict(durations)
    session.chat_transcript = list(chat.get("transcript", []) or [])
    for record_index, record in enumerate(session.chat_transcript):
        if isinstance(record, dict) and record.get("queued"):
            record["queued"] = False
            record["badge"] = "Not run"
            replay_commands.append({"cmd": "chat_output", "event": _message_replay_event(record, record_index)})
    session.chat_transcript_counter = int(chat.get("transcript_counter", 0) or 0)
    session.chat_revision = int(chat.get("revision", 0) or 0) + 1
    session.chat_event_sequence = 0
    session.ui_replay_commands = replay_commands
    session.ui_replay_sequence = replay_sequence
    session.interruption_notice = str(chat.get("interruption_notice", "") or "")
    session.interruption_history = list(chat.get("interruption_history", []) or [])
    session.media_registry = list(context.get("media", []) or [])
    from shared.deepy.media_links import restore_media_links

    restore_media_links(session.chat_transcript, replay_commands, session.media_registry)
    session.media_registry_counter = max([int(str(record.get("media_id", "_0")).rsplit("_", 1)[-1]) for record in session.media_registry if str(record.get("media_id", "")).rsplit("_", 1)[-1].isdigit()] or [0])
    session.tool_ui_settings = dict(context.get("ui", {}).get("tool_settings", {}) or {})
    session.model_selection_runtime_signature = str(context.get("ui", {}).get("model_selection_runtime_signature", ""))
    runtime = context.get("runtime", {})
    session.pending_chat_media = list(runtime.get("pending_chat_media", []))
    pending_action = _pending_action_from_runtime(runtime)
    session.generated_client_ids = list(runtime.get("generated_client_ids", []) or [])
    session.selected_visual_runtime_signature = str(runtime.get("selected_visual_signature", "") or "")
    session.selected_audio_runtime_signature = str(runtime.get("selected_audio_signature", "") or "")
    session.selected_mesh_runtime_signature = str(runtime.get("selected_mesh_signature", "") or "")
    session.selected_world_runtime_signature = str(runtime.get("selected_world_signature", "") or "")
    session.rendered_token_ids = []
    session.rendered_messages_len = 0
    session.runtime_snapshot = None
    session.paused_runtime_snapshot = None
    session.pending_action_replay = pending_action
    session.pending_action_replay_messages = copy.deepcopy(session.messages) if pending_action is not None else []
    session.pending_action_replay_transcript = copy.deepcopy(session.chat_transcript) if pending_action is not None else []
    session.pending_replay_reason = "restored session requires a full context prefill"
    session.safe_checkpoint_revision = int(context.get("checkpoint_revision", 0) or 0)
    session.saved_checkpoint_revision = session.safe_checkpoint_revision
    session.chat_epoch = int(session.chat_epoch or 0) + 1
    session.active_skills = list(session.session_environment.get("skills", []) or [])
    bind_session_persistence(session)
    return {"manifest": manifest, "missing_media": missing_media, "environment": session.session_environment, "replay_commands": replay_commands}


def inject_session_media(session, gen: dict[str, Any]) -> dict[str, Any]:
    from shared.deepy import media_registry
    from shared.utils.media_settings import peek_settings

    from shared.utils.gallery_sources import GALLERY_SOURCES, source_for_kind

    galleries = {source.id: (gen.setdefault(source.files_key, []), gen.setdefault(source.settings_key, [])) for source in GALLERY_SOURCES}
    visual_paths, _visual_settings = galleries["video"]
    canonical_paths = {}
    settings_locations = {}
    client_keys = {}
    fingerprints = {}
    for paths, settings_list in galleries.values():
        for index, path in enumerate(paths):
            settings = peek_settings(settings_list, index) or {}
            entry = (str(path), settings)
            canonical = os.path.normcase(str(Path(str(path)).resolve()))
            canonical_paths[canonical] = entry
            settings_locations[canonical] = (settings_list, index)
            client_id = str(settings.get("client_id", "") or "")
            if client_id:
                client_keys[(_detect_media_type(Path(str(path))), client_id)] = entry
            fingerprint = str(settings.get("deepy_media_fingerprint", "") or "")
            existing_path = Path(str(path or ""))
            if not fingerprint and existing_path.is_file():
                fingerprint = _fingerprint(existing_path)
            if fingerprint:
                fingerprints[fingerprint] = entry
    injected = 0
    missing = []
    # The registry is newest-first; Gallery appends in chronological order.
    for record in reversed(session.media_registry):
        if not isinstance(record, dict):
            continue
        path = Path(str(record.get("path", "") or ""))
        if not path.is_file():
            missing.append(str(path))
            continue
        media_type = str(record.get("media_type", "") or _detect_media_type(path))
        canonical = os.path.normcase(str(path.resolve()))
        client_id = str(record.get("client_id", "") or "")
        fingerprint = str(record.get("fingerprint", "") or "")
        settings = dict(media_registry._resolve_settings(str(path), record.get("settings")))
        settings.update({"deepy_session_id": session.storage_session_id, "deepy_media_id": record.get("media_id", ""), "deepy_media_fingerprint": fingerprint})
        target = source_for_kind(media_type)
        gallery = target.media_ids
        settings["gallery_media_ids"] = gallery_media_ids(str(path), gallery, settings)
        existing = canonical_paths.get(canonical)
        candidate = client_keys.get((media_type, client_id)) or fingerprints.get(fingerprint)
        # A job can produce several files; matching size/mtime also does not establish identity.
        if existing is None and candidate is not None and Path(candidate[0]).is_file():
            with path.open("rb") as source_file, open(candidate[0], "rb") as candidate_file:
                while True:
                    chunk = source_file.read(1024 * 1024)
                    if chunk != candidate_file.read(1024 * 1024):
                        break
                    if not chunk:
                        existing = candidate
                        break
        if existing is not None:
            existing_path, existing_settings = existing
            existing_settings.update(media_registry._resolve_settings(existing_path, {**settings, **existing_settings}))
            target_list, target_index = settings_locations[os.path.normcase(str(Path(existing_path).resolve()))]
            target_list[target_index] = existing_settings
            ids = gallery_media_ids(existing_path, gallery, existing_settings)
            existing_settings["gallery_media_ids"] = list(dict.fromkeys([*settings["gallery_media_ids"], *ids]))
            record.update(path=existing_path, path_key=os.path.normcase(str(Path(existing_path).resolve())))
            record["settings"] = {**settings, **existing_settings}
            continue
        if client_id:
            settings["client_id"] = client_id
        record["settings"] = settings.copy()
        target_paths, target_settings = galleries[target.id]
        target_paths.append(str(path))
        target_settings.append(settings)
        entry = (str(path), settings)
        canonical_paths[canonical] = entry
        settings_locations[canonical] = (target_settings, len(target_settings) - 1)
        if client_id:
            client_keys[(media_type, client_id)] = entry
        if fingerprint:
            fingerprints[fingerprint] = entry
        injected += 1
    for source in GALLERY_SOURCES:
        gen[source.selected_key] = len(galleries[source.id][0]) - 1
        gen[source.last_selected_key] = True
    gen["selected_video_time"] = 0.0 if visual_paths and _detect_media_type(Path(visual_paths[-1])) == "video" else None
    media_registry.sync_tool_call_gallery_media(session, gen)
    for source, (paths, _settings) in galleries.items():
        setattr(session, f"seen_{source}_gallery_paths", [str(path) for path in paths])
    return {"injected": injected, "missing": missing}


def rename_session(session, title: str) -> dict[str, Any]:
    session.storage_title = _unique_title(title, session.storage_session_id)
    session.storage_title_pending = False
    manifest = flush_session(session)
    return manifest or session_metadata(session)


def rename_stored_session(storage_id: str, title: str, active_session=None) -> dict[str, Any]:
    if active_session is not None and active_session.storage_session_id == storage_id:
        return rename_session(active_session, title)
    directory = _validated_session_dir(storage_id)
    if (directory / ".session.lock").is_file():
        lock = _read_json(directory / ".session.lock")
        if _process_alive(int(lock.get("pid", 0) or 0)):
            raise SessionLockedError("Close this Deepy session before renaming it.")
    normalized_title = _unique_title(title, storage_id)
    manifest = _read_json(directory / "session.json")
    manifest["title"] = normalized_title
    manifest.pop("title_pending", None)
    manifest["updated_at"] = _utc_now()
    _atomic_json(directory / "session.json", manifest)
    return manifest


def duplicate_stored_session(storage_id: str, active_session=None) -> dict[str, Any]:
    if active_session is not None and active_session.storage_session_id == storage_id:
        flush_session(active_session)
    directory = _validated_session_dir(storage_id)
    manifest = validate_session(storage_id, str(_read_json(directory / "session.json").get("deepy_type", "")), active_session=active_session)
    context = _read_json(directory / "context.json")
    _preserve_gallery_media_ids(context)
    references = SessionReferences()
    context = references.rebase(context)
    root = sessions_root()
    new_id = _new_storage_id(root)
    existing_titles = {str(item.get("title", "") or "").casefold() for item in list_sessions()}
    base_title = re.sub(r"\s+\(copy(?: \d+)?\)$", "", str(manifest.get("title", "") or "Deepy session"), flags=re.IGNORECASE).strip()
    copy_number = 1
    while True:
        suffix = " (copy)" if copy_number == 1 else f" (copy {copy_number})"
        duplicate_title = f"{base_title[: max(1, 120 - len(suffix))].rstrip()}{suffix}"
        if duplicate_title.casefold() not in existing_titles:
            break
        copy_number += 1
    with _ROOT_LOCK:
        destination = _new_session_directory(root, duplicate_title)
        destination.mkdir()
    now = _utc_now()
    manifest.update(id=new_id, title=duplicate_title, created_at=now, updated_at=now)
    manifest.pop('previous_directories', None)
    context["session_id"] = new_id
    context["saved_at"] = now
    replacements: dict[str, str] = {}
    _register_path_replacement(replacements, str(directory), str(destination))
    context = _replace_paths(context, replacements)
    context = _rebind_session_references(context, storage_id, new_id)
    _preserve_gallery_media_ids(context, rebind_paths=True)
    try:
        shutil.copytree(directory, destination, dirs_exist_ok=True, ignore=lambda source, names: ['.session.lock', *(['session.json', 'context.json'] if Path(source) == directory else [])])
        journal = destination / UI_JOURNAL_FILENAME
        if journal.is_file():
            journal_text = references.rebase(journal.read_text(encoding="utf-8"))
            for original, replacement in replacements.items():
                journal_text = journal_text.replace(original, replacement)
            journal.write_text(journal_text, encoding="utf-8", newline="\n")
        _atomic_json(destination / "context.json", context)
        _atomic_json(destination / "session.json", manifest)
    except Exception:
        if destination.is_dir():
            destination.resolve().relative_to(root)
            shutil.rmtree(destination)
        raise
    return manifest


def set_gallery_media_mode(session, mode: str) -> dict[str, Any] | None:
    session.gallery_media_mode = normalize_gallery_media_mode(mode)
    return flush_session(session) if session.storage_session_id else None


def start_new_session(session, *, save_current: bool = True) -> None:
    if save_current and session.storage_session_id:
        flush_session(session)
    release_session_lock(session)
    session.storage_session_id = ""
    session.storage_session_dir = ""
    session.storage_title = ""
    session.storage_title_pending = False
    session.storage_deepy_type = ""
    session.storage_created_at = ""
    session.storage_updated_at = ""
    session.session_environment = {}
    session.model_selection_runtime_signature = ""
    session.safe_checkpoint_revision = 0
    session.saved_checkpoint_revision = 0
    session.pending_session_save = None
    session.session_save_error = ""
    session.safe_checkpoint_callback = None
    session.ui_replay_commands = []
    session.ui_replay_sequence = 0
    session.pending_action_replay = None
    session.pending_action_replay_messages = []
    session.pending_action_replay_transcript = []
    session.chat_session_id = uuid.uuid4().hex
    session.chat_epoch = int(session.chat_epoch or 0) + 1


def reset_session_files(session) -> None:
    if not session.storage_session_id:
        return
    directory = _validated_session_dir(session.storage_session_id)
    _forget_ui_journal(directory)
    (directory / UI_JOURNAL_FILENAME).unlink(missing_ok=True)
    _UI_JOURNAL_STATES[str(directory.resolve())] = {"transcript": [], "sequence": 0}
    session.ui_replay_commands = []
    session.ui_replay_sequence = 0
    for child_name in ("workspace", "media"):
        child = (directory / child_name).resolve()
        child.relative_to(directory)
        if child.is_dir():
            shutil.rmtree(child)
        if child_name == "workspace" or session.gallery_media_mode == GALLERY_MEDIA_COPY:
            child.mkdir()
    session.safe_checkpoint_revision = int(session.safe_checkpoint_revision or 0) + 1
    flush_session(session)


@contextmanager
def archiving_session(storage_id: str):
    """Hide a session without moving linked artifacts; undo if its workspace cannot move."""
    directory = _validated_session_dir(storage_id)
    lease = SimpleNamespace(session_lock_path="", session_lock_token="", pending_session_save=None, storage_session_dir="")
    _acquire_session_lock(lease, directory)
    try:
        path = directory / "session.json"
        manifest = _read_json(path)
        if manifest.get("id") != storage_id:
            raise SessionStoreError("Deepy session identifier does not match its directory.")
        _atomic_json(path, {**manifest, "archived_at": _utc_now()})
        try:
            yield
        except Exception:
            _atomic_json(path, manifest)
            raise
    finally:
        release_session_lock(lease)


def delete_session(storage_id: str, active_session=None) -> Path:
    directory = _validated_session_dir(storage_id)
    _forget_ui_journal(directory)
    if active_session is not None and active_session.storage_session_id == storage_id:
        release_session_lock(active_session)
    elif (directory / ".session.lock").is_file():
        lock = _read_json(directory / ".session.lock")
        if _process_alive(int(lock.get("pid", 0) or 0)):
            raise SessionLockedError("Close this Deepy session before deleting it.")
        (directory / ".session.lock").unlink(missing_ok=True)
    trash = sessions_root() / ".trash"
    trash.mkdir(parents=True, exist_ok=True)
    destination = available_path(trash, directory.name)
    shutil.move(str(directory), str(destination))
    return destination


def export_session(session) -> Path:
    if not session.storage_session_id:
        raise SessionStoreError("There is no materialized Deepy session to export.")
    flush_session(session)
    return export_stored_session(session.storage_session_id, active_session=session)


def export_stored_session(storage_id: str, active_session=None) -> Path:
    if active_session is not None and active_session.storage_session_id == storage_id:
        flush_session(active_session)
    directory = _validated_session_dir(storage_id)
    lock_path = directory / ".session.lock"
    active_lock = active_session is not None and active_session.storage_session_id == storage_id and Path(str(active_session.session_lock_path or "")) == lock_path
    if lock_path.is_file() and not active_lock:
        lock = _read_json(lock_path)
        if _process_alive(int(lock.get("pid", 0) or 0)):
            raise SessionLockedError("Close this Deepy session before exporting it.")
    exports = sessions_root() / "_exports"
    exports.mkdir(parents=True, exist_ok=True)
    destination = exports / f"{directory.name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.zip"
    manifest = _read_json(directory / "session.json")
    context = _read_json(directory / "context.json")
    references = SessionReferences()
    _preserve_gallery_media_ids(context)
    context = references.rebase(context)
    directory_files = [path for path in directory.rglob("*") if path.is_file() and path.name != '.session.lock' and path not in (directory / 'context.json', directory / 'session.json')]
    archive_names = {path.relative_to(directory).as_posix() for path in directory_files}
    external_files: list[tuple[Path, str]] = []
    missing_media = []
    for record in list(context.get("media", []) or []):
        if not isinstance(record, dict):
            continue
        reference = references.encode(record['path']) if record.get('path') else None
        if isinstance(reference, dict):
            record['session_reference'] = reference
        if str(record.get("source", "")) == "workspace":
            continue
        copied = str(record.get("session_copy_path", "") or "")
        if copied and _session_member_path(directory, copied).is_file():
            continue
        source_text = str(record.get("path", "") or "").strip()
        if 'session_reference' in record:
            source_text = references.resolve(record['session_reference']) or ''
        source = Path(source_text) if source_text else None
        if source is None or not source.is_file():
            if source_text:
                missing_media.append(source_text)
            continue
        safe_id = re.sub(r"[^A-Za-z0-9_-]+", "_", str(record.get("media_id", "media"))).strip("_") or "media"
        stem = f"{safe_id}_{source.name}"
        archive_name = f"media/{stem}"
        suffix = 2
        while archive_name in archive_names:
            archive_name = f"media/{safe_id}_{suffix}_{source.name}"
            suffix += 1
        archive_names.add(archive_name)
        record["session_copy_path"] = archive_name
        external_files.append((source, archive_name))
    context["export"] = {"packaged_at": _utc_now(), "missing_media": missing_media}
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        archive.writestr("session.json", json.dumps(_json_safe(manifest), ensure_ascii=False, indent=2))
        archive.writestr("context.json", json.dumps(_json_safe(context), ensure_ascii=False, indent=2))
        for path in directory_files:
            if path == directory / UI_JOURNAL_FILENAME:
                archive.writestr(UI_JOURNAL_FILENAME, references.rebase(path.read_text(encoding='utf-8')))
            else:
                archive.write(path, path.relative_to(directory).as_posix(), compress_type=zipfile.ZIP_STORED if path.suffix.lower() in _MEDIA_EXTENSIONS else zipfile.ZIP_DEFLATED)
        for path, archive_name in external_files:
            archive.write(path, archive_name, compress_type=zipfile.ZIP_STORED)
    return destination


def import_session(archive_path: str | os.PathLike[str]) -> dict[str, Any]:
    archive_path = Path(archive_path).expanduser().resolve()
    if not archive_path.is_file() or not zipfile.is_zipfile(archive_path):
        raise SessionStoreError("Select a valid Deepy session ZIP archive.")
    root = sessions_root()
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        if len(members) > 10000 or sum(max(0, member.file_size) for member in members) > 50 * 1024 * 1024 * 1024:
            raise SessionStoreError("Deepy session archive is too large.")
        for member in members:
            member_path = Path(member.filename.replace("\\", "/"))
            if member_path.is_absolute() or ".." in member_path.parts:
                raise SessionStoreError("Deepy session archive contains an unsafe path.")
        with tempfile.TemporaryDirectory(prefix=".deepy-import-", dir=root) as temporary_name:
            temporary = Path(temporary_name)
            archive.extractall(temporary)
            content_root = temporary
            if not (content_root / "session.json").is_file():
                candidates = [path.parent for path in content_root.rglob("session.json") if path.parent.joinpath("context.json").is_file()]
                if len(candidates) != 1:
                    raise SessionStoreError("Archive does not contain one Deepy session.")
                content_root = candidates[0]
            manifest = _read_json(content_root / "session.json")
            context = _read_json(content_root / "context.json")
            original_id = str(manifest.get("id", "") or "")
            if int(manifest.get("schema_version", 0) or 0) > SESSION_SCHEMA_VERSION or int(context.get("schema_version", 0) or 0) > CONTEXT_SCHEMA_VERSION:
                raise SessionStoreError("This Deepy session archive was created by a newer format.")
            if str(manifest.get("deepy_type", "")).strip().lower() not in {"prime", "zero"} or not isinstance(context.get("chat", {}), dict):
                raise SessionStoreError("Archive contains an invalid Deepy session context.")
            _session_replay_commands(content_root, context)
            if original_id and str(context.get("session_id", "") or "") != original_id:
                raise SessionStoreError("Archive contains inconsistent Deepy session identifiers.")
            for record in list(context.get("media", []) or []):
                if not isinstance(record, dict):
                    continue
                if 'session_reference' in record:
                    SessionReferences.validate(record['session_reference'])
                for key in ("session_relative_path", "session_copy_path"):
                    relative_path = str(record.get(key, "") or "")
                    if relative_path:
                        _session_member_path(content_root, relative_path)
            storage_id = original_id if _SESSION_ID_RE.fullmatch(original_id) and original_id not in _session_manifests(root) and not (root / original_id).exists() else _new_storage_id(root)
            manifest["id"] = storage_id
            manifest.pop('previous_directories', None)
            manifest["title"] = str(manifest.get("title", "") or "Imported Deepy session")
            manifest["updated_at"] = _utc_now()
            context["session_id"] = storage_id
            context = _rebind_session_references(context, original_id, storage_id)
            with _ROOT_LOCK:
                destination = _new_session_directory(root, manifest['title'])
                destination.mkdir()
            try:
                shutil.copytree(content_root, destination, dirs_exist_ok=True, ignore=lambda source, names: ['.session.lock', *(['session.json', 'context.json'] if Path(source) == content_root else [])])
                _atomic_json(destination / "context.json", context)
                _atomic_json(destination / "session.json", manifest)
            except Exception:
                destination.resolve().relative_to(root)
                shutil.rmtree(destination)
                raise
    return manifest


def _detect_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".wav", ".mp3", ".aac", ".m4a", ".flac", ".ogg", ".opus", ".wma"}:
        return "audio"
    if suffix in {".mkv", ".mov", ".mp4", ".m4v", ".webm", ".avi"}:
        return "video"
    if suffix in MESH_EXTENSIONS:
        return "3d"
    if suffix in WORLD_EXTENSIONS:
        return "world"
    return "image"


__all__ = [
    "CONTEXT_SCHEMA_VERSION",
    "DEFAULT_SESSIONS_FOLDER",
    "GALLERY_MEDIA_COPY",
    "GALLERY_MEDIA_LINK",
    "MONO_SESSION_POINTER_FILENAME",
    "RESET_MODE_NEW",
    "RESET_MODE_RESET",
    "SESSION_SCHEMA_VERSION",
    "UI_JOURNAL_FILENAME",
    "UI_JOURNAL_SCHEMA_VERSION",
    "SessionLockedError",
    "SessionStoreError",
    "archiving_session",
    "automatic_title",
    "bind_session_persistence",
    "close_external_session",
    "configure_sessions_root",
    "delete_session",
    "duplicate_stored_session",
    "ensure_mono_session_workspace",
    "ensure_session",
    "export_session",
    "export_stored_session",
    "flush_session",
    "import_session",
    "inject_session_media",
    "list_sessions",
    "load_session",
    "normalize_gallery_media_mode",
    "normalize_reset_mode",
    "open_external_session",
    "release_session_lock",
    "rename_session",
    "rename_stored_session",
    "reset_session_files",
    "schedule_autosave",
    "session_folder_in_use",
    "session_metadata",
    "session_workspace",
    "sessions_root",
    "set_gallery_media_mode",
    "start_new_session",
    "validate_session",
]
