# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import os
import re
import time
from copy import deepcopy
from functools import lru_cache
from typing import Any

from shared.utils.audio_video import read_image_metadata
from shared.utils.gallery_media import gallery_media_ids
from shared.mesh3d.glb import MESH_EXTENSIONS as MESH_FILE_EXTENSIONS
from shared.utils.gallery_sources import APP_EXTENSIONS, GALLERY_SOURCES, WORLD_EXTENSIONS, source_for_kind
from shared.utils.media_settings import peek_settings


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff", ".jfif", ".pjpeg"}
_VIDEO_EXTENSIONS = {".mkv", ".mov", ".mp4", ".m4v", ".webm", ".avi"}
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".aac", ".m4a", ".flac", ".ogg", ".opus", ".wma"}
_MEDIA_TYPES = {"image", "video", "audio", "3d", "world", "app", "any", "all"}
PROMPT_SUMMARY_MAX_CHARS = 128
_TYPE_HINTS = {
    "image": ("image", "images", "picture", "photo", "photos", "pic", "pics"),
    "video": ("video", "videos", "clip", "movie", "footage"),
    "audio": ("audio", "sound", "song", "music", "track", "voice"),
    "3d": ("3d", "mesh", "glb", "splat", "splats", "gaussians"),
    "world": ("world", "worlds", "wgpworld"),
    "app": ("app", "apps", "wgpapp"),
}
_ALIAS_PREVIOUS_RE = re.compile(r"\b(previous|prior|before\s+last|second\s+last|penultimate)\b", flags=re.IGNORECASE)
_ALIAS_LAST_RE = re.compile(r"\b(last|latest|most\s+recent)\b", flags=re.IGNORECASE)
_REFERENCE_STOPWORDS = {
    "a",
    "an",
    "and",
    "any",
    "audio",
    "clip",
    "file",
    "generated",
    "image",
    "latest",
    "last",
    "media",
    "most",
    "my",
    "of",
    "on",
    "photo",
    "picture",
    "please",
    "prior",
    "recent",
    "show",
    "sound",
    "that",
    "the",
    "this",
    "track",
    "use",
    "video",
}
_SEARCH_LIMIT = 5
_GALLERY_MEDIA_ID_RE = re.compile(r"(?:visual|audio|3d|world):[a-f0-9]{12}", re.IGNORECASE)
_VIRTUAL_MEDIA_PATH_RE = re.compile(r"@[A-Za-z0-9_-]+/[^\r\n<>]*?\.(?:png|jpe?g|webp|bmp|gif|tiff?|jfif|pjpeg|mkv|mov|mp4|m4v|webm|avi|wav|mp3|aac|m4a|flac|ogg|opus|wma|glb|ply|wgpworld)(?=$|[\s\]\[(){}\"'`,.;:!?])", re.IGNORECASE)


def normalize_media_type(media_type: str | None, reference: str | None = None) -> str:
    normalized = str(media_type or "").strip().lower()
    if normalized not in _MEDIA_TYPES:
        normalized = "any"
    elif normalized == "all":
        normalized = "any"
    if normalized != "any":
        return normalized
    reference_text = _normalize_text(reference)
    for candidate_type, hints in _TYPE_HINTS.items():
        if any(hint in reference_text for hint in hints):
            return candidate_type
    return "any"


def detect_media_type(path: str) -> str:
    """Return the media kind inferred from a gallery file extension."""

    return _detect_media_type(path)


def get_media_record(session, media_id: str) -> dict[str, Any] | None:
    lookup_id = str(media_id or "").strip()
    if len(lookup_id) == 0:
        return None
    normalized_lookup_id = _normalize_media_id_lookup(lookup_id)
    for record in session.media_registry:
        record_id = str(record.get("media_id", "")).strip()
        if record_id == lookup_id or record_id == normalized_lookup_id:
            return record
    return None


def register_media(
    session,
    path: str,
    settings: dict[str, Any] | None = None,
    *,
    source: str = "wangp",
    client_id: str = "",
    label: str | None = None,
    media_type: str | None = None,
    access: str = "",
) -> dict[str, Any] | None:
    path = str(path or "").strip()
    if len(path) == 0:
        return None
    detected_type = normalize_media_type(media_type or _detect_media_type(path))
    if detected_type == "any":
        return None
    resolved_settings = _resolve_settings(path, settings)
    prompt = str((resolved_settings or {}).get("prompt", "") or "").strip()
    prompt_summary = summarize_prompt(prompt, detected_type)
    path_key = _normalize_path_key(path)
    existing = None
    for record in session.media_registry:
        if record.get("path_key") == path_key:
            existing = record
            break
    new_record = existing is None
    if new_record:
        session.media_registry_counter += 1
        existing = {
            "media_id": f"{detected_type}_{session.media_registry_counter}",
            "path_key": path_key,
        }
    else:
        session.media_registry.remove(existing)
        session.media_registry.insert(0, existing)
    gallery = source_for_kind(detected_type).media_ids
    ids = gallery_media_ids(path, gallery, existing.get("settings"))
    if (resolved_settings or {}).get("gallery_media_ids"):
        ids = list(dict.fromkeys([*gallery_media_ids(path, gallery, resolved_settings), *ids]))
    accesses = {str(value).strip().lower() for value in list(existing.get("access", []) or []) if str(value).strip().lower() in {"read", "write"}}
    if str(access or "").strip().lower() in {"read", "write"}:
        accesses.add(str(access).strip().lower())
    existing.update(
        {
            "media_type": detected_type,
            "path": path,
            "source": str(source or "wangp").strip() or "wangp",
            "client_id": str(client_id or "").strip(),
            "settings": {**(resolved_settings or {}), "gallery_media_ids": ids},
            "label": str(label or prompt_summary or _default_label(path, detected_type)).strip() or _default_label(path, detected_type),
            "prompt_summary": prompt_summary,
            "prompt": prompt,
            "filename": os.path.basename(path),
            "updated_at": float(time.time()),
            "access": sorted(accesses),
        }
    )
    # Gallery snapshots can run while IDs are resolved. Publish complete records.
    if new_record:
        session.media_registry.insert(0, existing)
    return existing


def mark_media_access(record: dict[str, Any] | None, access: str) -> dict[str, Any] | None:
    if not isinstance(record, dict):
        return record
    normalized = str(access or "").strip().lower()
    if normalized not in {"read", "write"}:
        raise ValueError(f"Unsupported media access kind: {access}")
    accesses = {str(value).strip().lower() for value in list(record.get("access", []) or []) if str(value).strip().lower() in {"read", "write"}}
    accesses.add(normalized)
    record["access"] = sorted(accesses)
    record["updated_at"] = float(time.time())
    return record


def record_media_access(
    session,
    path: str,
    settings: dict[str, Any] | None = None,
    *,
    source: str = "wangp",
    client_id: str = "",
    access: str,
) -> dict[str, Any] | None:
    path_key = _normalize_path_key(path)
    existing = next((record for record in session.media_registry if record.get("path_key") == path_key), None)
    if existing is not None:
        return mark_media_access(existing, access)
    return register_media(session, path, settings=settings, source=source, client_id=client_id, access=access)


def sync_tool_call_gallery_media(session, gen: dict[str, Any]) -> list[dict[str, Any]]:
    gallery_records = {}
    gallery_defs = [(source.media_ids, gen.get(source.files_key, []), gen.get(source.settings_key, [])) for source in GALLERY_SOURCES]
    for gallery, paths, settings_list in gallery_defs:
        for index, path in enumerate(paths):
            normalized_path = str(path or "").strip()
            if not normalized_path:
                continue
            settings = peek_settings(settings_list, index) if index < len(settings_list) else None
            for media_id in gallery_media_ids(normalized_path, gallery, settings):
                gallery_records[media_id] = (normalized_path, settings)

    referenced_ids = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                collect(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                collect(child)
        elif isinstance(value, str) and _GALLERY_MEDIA_ID_RE.fullmatch(value.strip()):
            referenced_ids.append(value.strip().casefold())

    for message in list(session.messages or []):
        if not isinstance(message, dict) or str(message.get("role", "")).strip().lower() != "assistant":
            continue
        for tool_call in list(message.get("tool_calls", []) or []):
            if isinstance(tool_call, dict):
                collect(tool_call.get("function", {}).get("arguments", {}))

    synced = []
    for media_id in dict.fromkeys(referenced_ids):
        gallery_record = gallery_records.get(media_id)
        if gallery_record is None:
            continue
        path, settings = gallery_record
        client_id = str((settings or {}).get("client_id", "") or "").strip()
        record = record_media_access(session, path, settings=settings, source=_resolve_source(settings), client_id=client_id, access="read")
        if record is not None:
            synced.append(record)
    return synced


def sync_context_media_paths(session, file_access_policy) -> list[dict[str, Any]]:
    references = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                collect(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                collect(child)
        elif isinstance(value, str):
            references.extend(match.group(0) for match in _VIRTUAL_MEDIA_PATH_RE.finditer(value))

    collect(list(session.messages or []))
    synced = []
    for reference in dict.fromkeys(references):
        try:
            path = file_access_policy.resolve_path(reference)
        except (OSError, PermissionError, ValueError):
            continue
        if not path.is_file() or detect_media_type(str(path)) == "any":
            continue
        record = record_media_access(session, str(path), source="wangp", access="read")
        if record is not None:
            synced.append(record)
    return synced


def collapse_gallery_media(file_list: list[Any], file_settings_list: list[Any], max_items: int = 0) -> tuple[list[Any], list[Any]]:
    gallery_files = list(file_list or [])
    gallery_settings = file_settings_list or []
    collapsed_pairs: list[tuple[Any, Any]] = []
    seen_client_keys: set[tuple[str, str]] = set()
    for index in range(len(gallery_files) - 1, -1, -1):
        path = gallery_files[index]
        settings = gallery_settings[index] if index < len(gallery_settings) else None
        client_key = _gallery_client_media_key(path, settings)
        if client_key is not None:
            if client_key in seen_client_keys:
                continue
            seen_client_keys.add(client_key)
        if path is None and settings is None:
            continue
        collapsed_pairs.append((path, settings))
        if max_items > 0 and len(collapsed_pairs) >= max_items:
            break
    collapsed_pairs.reverse()
    collapsed_files = []
    collapsed_settings = []
    for path, settings in collapsed_pairs:
        if path is None and settings is None:
            continue
        collapsed_files.append(path)
        collapsed_settings.append(settings)
    return collapsed_files, collapsed_settings


def find_last_gallery_media_by_client(file_list: list[Any], file_settings_list: list[Any], client_id: str, *, media_type: str | None = None) -> tuple[str | None, dict[str, Any] | None]:
    lookup_client_id = str(client_id or "").strip()
    if len(lookup_client_id) == 0:
        return None, None
    gallery_files = list(file_list or [])
    gallery_settings = file_settings_list or []
    expected_media_type = normalize_media_type(media_type)
    for index in range(len(gallery_files) - 1, -1, -1):
        settings = gallery_settings[index] if index < len(gallery_settings) else None
        if not isinstance(settings, dict):
            continue
        if str(settings.get("client_id", "") or "").strip() != lookup_client_id:
            continue
        path = str(gallery_files[index] or "").strip()
        if len(path) == 0:
            continue
        detected_media_type = _detect_media_type(path)
        if expected_media_type != "any" and detected_media_type != expected_media_type:
            continue
        return path, settings
    return None, None


def sync_recent_generated_media(session, file_list: list[Any], file_settings_list: list[Any], max_items: int = _SEARCH_LIMIT) -> list[dict[str, Any]]:
    collapsed_file_list, collapsed_settings_list = collapse_gallery_media(file_list, file_settings_list, max_items=max_items)
    recent_items = list(zip(collapsed_file_list, collapsed_settings_list))
    synced = []
    for path, settings in recent_items:
        record = register_media(
            session,
            str(path or ""),
            settings=settings if isinstance(settings, dict) else None,
            source=_resolve_source(settings),
            client_id=str((settings or {}).get("client_id", "") or "").strip(),
            label=_label_from_settings(settings, str(path or "")),
        )
        if record is not None:
            synced.append(record)
    return synced


def resolve_media_reference(session, reference: str, media_type: str = "any", limit: int = _SEARCH_LIMIT) -> dict[str, Any]:
    resolved_type = normalize_media_type(media_type, reference=reference)
    filtered = _filter_records(session.media_registry, resolved_type)
    reference_text = str(reference or "").strip()
    if len(filtered) == 0:
        return {"status": "not_found", "media_type": resolved_type, "reference": reference_text, "matches": []}
    alias_record = _resolve_alias(filtered, reference_text)
    if alias_record is not None:
        mark_media_access(alias_record, "read")
        return {"status": "resolved", "media_type": resolved_type, "reference": reference_text, "media": _compact_media(alias_record, why="matched recent alias")}
    direct_record = get_media_record(session, reference_text)
    if direct_record is not None and (resolved_type == "any" or direct_record.get("media_type") == resolved_type):
        mark_media_access(direct_record, "read")
        return {"status": "resolved", "media_type": resolved_type, "reference": reference_text, "media": _compact_media(direct_record, why="matched media id")}
    ranked = _rank_records(filtered, reference_text)
    if len(ranked) == 0:
        return {
            "status": "not_found",
            "media_type": resolved_type,
            "reference": reference_text,
            "matches": [_compact_media(record, why="recent") for record in filtered[: max(1, limit)]],
        }
    if len(ranked) == 1:
        mark_media_access(ranked[0][0], "read")
        return {"status": "resolved", "media_type": resolved_type, "reference": reference_text, "media": _compact_media(ranked[0][0], why=ranked[0][1])}
    return {
        "status": "candidates",
        "media_type": resolved_type,
        "reference": reference_text,
        "matches": [_compact_media(record, why=why) for record, why in ranked[: max(2, limit)]],
    }


@lru_cache(maxsize=512)
def _file_settings(path: str, mtime_ns: int, size: int, ctime_ns: int):
    # Cache absent metadata too; a new conversation must not reprobe the same files.
    from shared.utils.video_metadata import read_metadata_from_video

    reader = {"image": read_image_metadata, "video": read_metadata_from_video}.get(_detect_media_type(path))
    return reader(path) if reader is not None else None


def _resolve_settings(path: str, settings: dict[str, Any] | None) -> dict[str, Any]:
    # Session/gallery identifiers alone are not the file's generation metadata.
    if isinstance(settings, dict) and settings.keys() - {"gallery_media_ids", "deepy_session_id", "deepy_media_id", "deepy_media_fingerprint", "client_id"}:
        return settings
    if os.path.isfile(path):
        stat = os.stat(path)
        metadata = _file_settings(os.path.normcase(os.path.abspath(path)), stat.st_mtime_ns, stat.st_size, stat.st_ctime_ns)
        if isinstance(metadata, dict):
            return {**deepcopy(metadata), **(settings or {})}
    return settings or {}


def _resolve_alias(records: list[dict[str, Any]], reference: str) -> dict[str, Any] | None:
    normalized = _normalize_text(reference)
    if len(normalized) == 0:
        return records[0]
    if _ALIAS_PREVIOUS_RE.search(normalized):
        return records[1] if len(records) > 1 else None
    if _ALIAS_LAST_RE.search(normalized):
        return records[0]
    return None


def _rank_records(records: list[dict[str, Any]], reference: str) -> list[tuple[dict[str, Any], str]]:
    normalized_reference = _normalize_text(reference)
    reference_tokens = [token for token in _tokenize(reference) if token not in _REFERENCE_STOPWORDS]
    ranked = []
    for index, record in enumerate(records):
        search_blob = _normalize_text(" ".join(filter(None, [record.get("label", ""), record.get("prompt_summary", ""), record.get("prompt", ""), record.get("filename", "")])))
        if len(search_blob) == 0:
            continue
        score = 0
        why = ""
        if len(normalized_reference) > 0 and normalized_reference in search_blob:
            score += 100
            why = "exact text match"
        search_tokens = set(_tokenize(search_blob))
        token_hits = [token for token in reference_tokens if token in search_tokens]
        if token_hits:
            score += 15 * len(token_hits)
            why = f"matched {', '.join(token_hits[:3])}"
        score += max(0, 10 - index)
        if score <= 0:
            continue
        ranked.append((score, record, why or "recent semantic match"))
    ranked.sort(key=lambda item: (-item[0], item[1].get("updated_at", 0.0)), reverse=False)
    return [(record, why) for _score, record, why in ranked]


def _compact_media(record: dict[str, Any], why: str = "") -> dict[str, Any]:
    payload = {
        "media_id": record.get("media_id", ""),
        "media_type": record.get("media_type", ""),
        "label": record.get("label", ""),
        "source": record.get("source", ""),
        "filename": record.get("filename", ""),
    }
    prompt_summary = str(record.get("prompt_summary", "") or "").strip()
    if len(prompt_summary) > 0:
        payload["prompt_summary"] = prompt_summary
    if len(str(why or "").strip()) > 0:
        payload["why"] = str(why).strip()
    return payload


def _filter_records(records: list[dict[str, Any]], media_type: str) -> list[dict[str, Any]]:
    if media_type == "any":
        return list(records)
    return [record for record in records if record.get("media_type") == media_type]


def _normalize_path_key(path: str) -> str:
    return os.path.normcase(os.path.abspath(str(path or "").strip()))


def _normalize_media_id_lookup(media_id: str) -> str:
    normalized = re.sub(r"\s+", "_", str(media_id or "").strip().lower())
    return normalized


def _normalize_text(text: str | None) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(text or "").strip().lower())).strip()


def _tokenize(text: str | None) -> list[str]:
    return [token for token in _normalize_text(text).split(" ") if len(token) > 0]


def _detect_media_type(path: str) -> str:
    ext = os.path.splitext(str(path or ""))[1].lower()
    if ext in _IMAGE_EXTENSIONS:
        return "image"
    if ext in _VIDEO_EXTENSIONS:
        return "video"
    if ext in _AUDIO_EXTENSIONS:
        return "audio"
    if ext in MESH_FILE_EXTENSIONS:  # every mesh format the loader reads, not only the two the 3D gallery stores
        return "3d"
    if ext in WORLD_EXTENSIONS:
        return "world"
    if ext in APP_EXTENSIONS:
        return "app"
    return "any"


def _default_label(path: str, media_type: str) -> str:
    base_name = os.path.splitext(os.path.basename(str(path or "").strip()))[0].replace("_", " ").strip()
    return base_name or f"Generated {media_type}"


def summarize_prompt(prompt: str, media_type: str) -> str:
    meaningful_lines = []
    for raw_line in str(prompt or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("!"):
            continue
        line = re.sub(r"\[[^\]\r\n]*\]", " ", line)
        line = re.sub(r"\(\s*(?:from\s*)?\)", " ", line)  # parentheses emptied by the removed labels
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            meaningful_lines.append(line)
    summary = " ".join(meaningful_lines)[:PROMPT_SUMMARY_MAX_CHARS].rstrip()
    return summary or media_type.capitalize()


def _resolve_source(settings: dict[str, Any] | None) -> str:
    client_id = str((settings or {}).get("client_id", "") or "").strip()
    if client_id.startswith("ai"):
        return "deepy"
    return "wangp"


def _label_from_settings(settings: dict[str, Any] | None, path: str) -> str | None:
    if not isinstance(settings, dict):
        return None
    prompt = str(settings.get("prompt", "") or "").strip()
    media_type = _detect_media_type(path)
    if len(prompt) == 0:
        return None
    return summarize_prompt(prompt, media_type)


def _gallery_client_media_key(path: Any, settings: Any) -> tuple[str, str] | None:
    if not isinstance(settings, dict):
        return None
    client_id = str(settings.get("client_id", "") or "").strip()
    if len(client_id) == 0:
        return None
    media_type = _detect_media_type(str(path or "").strip())
    if media_type not in {"video", "audio"}:
        return None
    return media_type, client_id


__all__ = [
    "collapse_gallery_media",
    "find_last_gallery_media_by_client",
    "get_media_record",
    "mark_media_access",
    "normalize_media_type",
    "record_media_access",
    "register_media",
    "resolve_media_reference",
    "summarize_prompt",
    "sync_context_media_paths",
    "sync_tool_call_gallery_media",
    "sync_recent_generated_media",
]
