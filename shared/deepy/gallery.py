# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import os
from typing import Any

from shared.deepy import media_registry
from shared.deepy.engine import get_or_create_assistant_session
from shared.utils.gallery_sources import APP_EXTENSIONS, GALLERY_SOURCES, VIDEO, WORLD_EXTENSIONS, gallery_source, source_for_kind
from shared.utils.utils import get_video_info, has_mesh_file_extension

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff", ".jfif", ".pjpeg"}
_VIDEO_EXTENSIONS = {".mkv", ".mov", ".mp4", ".webm"}
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".aac", ".flac", ".m4a", ".ogg", ".wma", ".opus"}
_SCOPE_GROUPS = {"media": ("Media", VIDEO.id, None), "image": ("Images", VIDEO.id, "image"), "video": ("Videos", VIDEO.id, "video"), "audio": ("Audio", "audio", None), "3d": ("3D Assets", "mesh", None), "world": ("Worlds", "world", None), "app": ("Apps", "app", None)}


class Gallery:
    def __init__(self, deps, state: dict[str, Any]):
        self._deps = deps
        self._state = state
        self._session = get_or_create_assistant_session(state)
        self._record_views = {}

    def _gen(self) -> dict[str, Any]:
        return self._deps.get_gen_info(self._state)

    def _detect_media_type(self, path: str) -> str:
        ext = os.path.splitext(str(path or "").strip())[1].lower()
        if ext in _IMAGE_EXTENSIONS:
            return "image"
        if ext in _VIDEO_EXTENSIONS:
            return "video"
        if ext in _AUDIO_EXTENSIONS:
            return "audio"
        if has_mesh_file_extension(path):
            return "3d"
        if ext in WORLD_EXTENSIONS:
            return "world"
        if ext in APP_EXTENSIONS:
            return "app"
        return "any"

    def _resolve_lists(self, source_id: str) -> tuple[list[Any], list[Any]]:
        gen = self._gen()
        source = gallery_source(source_id)
        return gen[source.files_key], gen[source.settings_key]

    def _select_index(self, index: int, source_id: str) -> None:
        gen = self._gen()
        source = gallery_source(source_id)
        file_list, _file_settings_list = self._resolve_lists(source_id)
        if len(file_list) == 0:
            index = -1
        else:
            index = max(0, min(int(index), len(file_list) - 1))
        gen[source.selected_key] = index
        gen[source.last_selected_key] = (index + 1) >= len(file_list)
        gen["current_gallery_source"] = source.id
        selected_path = str(file_list[index] or "").strip() if source is VIDEO and 0 <= index < len(file_list) else ""
        gen["selected_video_time"] = 0.0 if self._detect_media_type(selected_path) == "video" else None

    def _register(self, path: str, settings: dict[str, Any] | None = None) -> dict[str, Any] | None:
        settings = settings if isinstance(settings, dict) else None
        client_id = "" if settings is None else str(settings.get("client_id", "") or "").strip()
        source = "deepy" if client_id.startswith("ai_") else "wangp"
        return media_registry.register_media(self._session, path, settings=settings, source=source, client_id=client_id)

    def add_path(self, raw_path: str, preferred_type: str = "any") -> dict[str, Any]:
        path = os.path.abspath(os.path.normpath(str(raw_path or "").strip().strip('"')))
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        detected_type = self._detect_media_type(path)
        if detected_type == "any":
            raise ValueError("Unsupported media type. Use an image, video, audio or 3D (GLB) file.")
        preferred_type = str(preferred_type or "any").strip().lower() or "any"
        if preferred_type != "any" and detected_type != preferred_type:
            raise ValueError(f"Expected a {preferred_type} file, got a {detected_type} file.")
        configs, _any_image_or_video, any_audio = self._deps.get_settings_from_file(self._state, path, False, False, False)
        source_id = "audio" if any_audio else source_for_kind(detected_type).id
        file_list, file_settings_list = self._resolve_lists(source_id)
        normalized_path = os.path.normcase(path)
        # Like the WebUI import, media already in the gallery move to its end with their settings.
        existing = next((index for index, existing_path in enumerate(file_list) if os.path.normcase(str(existing_path or "")) == normalized_path), None)
        if existing is not None:
            file_list.pop(existing)
            configs = file_settings_list.pop(existing)
        file_list.append(path)
        file_settings_list.append(configs if isinstance(configs, dict) else None)
        self._select_index(len(file_list) - 1, source_id)
        record = self._register(path, configs)
        return {"record": record, "added": existing is None}

    def counts(self) -> dict[str, int]:
        gen = self._gen()
        return {source.id: len(gen[source.files_key]) for source in GALLERY_SOURCES}

    def sync_latest_generated(self, before_counts: dict[str, int]) -> None:
        gen = self._gen()
        for source in GALLERY_SOURCES:
            file_list, file_settings_list = gen[source.files_key], gen[source.settings_key]
            if len(file_list) > before_counts[source.id]:
                if gen[source.last_selected_key]:
                    self._select_index(len(file_list) - 1, source.id)
                self._register(str(file_list[-1] or ""), file_settings_list[-1] if file_settings_list else None)

    def sync_refresh_path(self, payload: Any) -> None:
        if not isinstance(payload, dict):
            return
        path = str(payload.get("path", "") or "").strip()
        if len(path) == 0:
            return
        detected_type = self._detect_media_type(path)
        if detected_type == "any":
            return
        source = source_for_kind(detected_type)
        file_list = self._gen()[source.files_key]
        if path in file_list and self._gen()[source.last_selected_key]:
            self._select_index(file_list.index(path), source.id)

    def _iter_records(self, source_id: str | None = None) -> list[tuple[str, int, dict[str, Any], str]]:
        gen = self._gen()
        groups = []
        # Rendering a large retained workspace must not re-register every media
        # (registry insertion/search is linear). Invalidate on metadata replacement
        # or session registry reset, while retaining the original registration path.
        registered = {id(record) for record in self._session.media_registry}
        for source in GALLERY_SOURCES:
            if source_id is not None and source.id != source_id:
                continue
            file_list, file_settings_list = self._resolve_lists(source.id)
            for index, path in enumerate(file_list):
                settings = file_settings_list[index]
                cached = self._record_views.get(path)
                if cached is not None and cached[0] is settings and id(cached[1]) in registered:
                    record = cached[1]
                else:
                    record = self._register(path, settings)
                    if record is not None:
                        self._record_views[path] = (settings, record)
                        registered.add(id(record))
                if record is not None:
                    groups.append((source.id, index, record, path))
        if source_id is None:
            active = {path for source in GALLERY_SOURCES for path in gen[source.files_key]}
            self._record_views = {path: value for path, value in self._record_views.items() if path in active}
        return groups

    def list_lines(self, media_scope: str = "all") -> list[str]:
        scope = str(media_scope or "all").strip().lower()
        if scope not in {"all", *_SCOPE_GROUPS}:
            scope = "all"
        gen = self._gen()
        lines = []
        groups = []
        for group_scope in (("media", "audio", "3d", "world", "app") if scope == "all" else (scope,)):
            header, source_id, media_type = _SCOPE_GROUPS[group_scope]
            items = self._iter_records(source_id)
            groups.append((header, items if media_type is None else [item for item in items if item[2].get("media_type") == media_type]))
        for header, items in groups:
            if len(items) == 0:
                continue
            lines.append(f"{header}:")
            for source_id, index, record, path in items:
                is_selected = gen["current_gallery_source"] == source_id and gen[gallery_source(source_id).selected_key] == index
                selected_suffix = ""
                if is_selected and record.get("media_type") == "video":
                    selected_suffix = f" @ {float(gen.get('selected_video_time', 0.0) or 0.0):.3f}s"
                marker = "*" if is_selected else " "
                label = str(record.get("label", "") or os.path.basename(path)).strip()
                description = str(record.get("prompt_summary", "") or "").strip()
                if len(description) > 0 and description.casefold() != label.casefold():
                    label = f"{label} - {description}"
                lines.append(f"{marker} {record.get('media_id', '?')} [{index + 1}] {label}{selected_suffix}")
        if len(lines) == 0:
            return ["No media available."]
        return lines

    def _resolve_by_reference(self, reference: str, media_type: str = "any") -> tuple[str, int] | None:
        ref = str(reference or "").strip()
        if len(ref) == 0:
            return None
        normalized_media_type = media_registry.normalize_media_type(media_type)
        candidates = [
            (source_id, index, record, path)
            for source_id, index, record, path in self._iter_records()
            if normalized_media_type == "any" or str(record.get("media_type", "")).strip() == normalized_media_type
        ]
        if ref.isdigit():
            choice = int(ref) - 1
            if 0 <= choice < len(candidates):
                return candidates[choice][0], candidates[choice][1]
            return None
        record = media_registry.get_media_record(self._session, ref)
        if record is not None and (normalized_media_type == "any" or str(record.get("media_type", "")).strip() == normalized_media_type):
            for source_id, index, candidate, _path in candidates:
                if candidate.get("media_id") == record.get("media_id"):
                    return source_id, index
        normalized_ref = ref.lower()
        matches = []
        for source_id, index, record, path in candidates:
            haystack = " ".join(
                [
                    str(record.get("media_id", "") or ""),
                    str(record.get("label", "") or ""),
                    str(record.get("prompt_summary", "") or ""),
                    os.path.basename(path),
                ]
            ).lower()
            if normalized_ref in haystack:
                matches.append((source_id, index))
        if len(matches) == 1:
            return matches[0]
        return None

    def select(self, reference: str, media_type: str = "any") -> dict[str, Any] | None:
        resolved = self._resolve_by_reference(reference, media_type=media_type)
        if resolved is None:
            return None
        source_id, index = resolved
        self._select_index(index, source_id)
        file_list, file_settings_list = self._resolve_lists(source_id)
        settings = file_settings_list[index] if index < len(file_settings_list) else None
        return self._register(str(file_list[index] or ""), settings)

    def selected_summary(self) -> str:
        gen = self._gen()
        source = gallery_source(gen.get("current_gallery_source", VIDEO.id))
        file_list, file_settings_list = self._resolve_lists(source.id)
        choice = gen[source.selected_key]
        if choice is None or choice < 0 or choice >= len(file_list):
            return "No media is currently selected."
        record = self._register(str(file_list[choice] or ""), file_settings_list[choice] if choice < len(file_settings_list) else None)
        if record is None:
            return "No media is currently selected."
        summary = f"Selected {record.get('media_id')}: {record.get('label', os.path.basename(str(file_list[choice] or '')))}"
        if record.get("media_type") == "video":
            frame_text = self.selected_frame_summary()
            summary += f" @ {float(gen.get('selected_video_time', 0.0) or 0.0):.3f}s"
            if frame_text.startswith("Selected frame: "):
                summary += f" ({frame_text[len('Selected frame: '):]})"
        return summary

    def set_selected_time(self, seconds: float) -> str:
        gen = self._gen()
        file_list = gen["file_list"]
        choice = gen["selected"]
        if gen.get("current_gallery_source", "video") != "video" or choice is None or choice < 0 or choice >= len(file_list):
            return "No video is currently selected."
        path = str(file_list[choice] or "").strip()
        if self._detect_media_type(path) != "video":
            return "The selected media is not a video."
        gen["selected_video_time"] = max(0.0, float(seconds))
        return f"Selected video time set to {gen['selected_video_time']:.3f}s."

    def _get_selected_video_state(self) -> tuple[dict[str, Any] | None, str, int, str]:
        gen = self._gen()
        file_list = gen["file_list"]
        file_settings_list = gen["file_settings_list"]
        choice = gen["selected"]
        if gen.get("current_gallery_source", "video") != "video" or choice is None or choice < 0 or choice >= len(file_list):
            return None, "", 0, "No video is currently selected."
        path = str(file_list[choice] or "").strip()
        if self._detect_media_type(path) != "video":
            return None, "", 0, "The selected media is not a video."
        settings = file_settings_list[choice] if choice < len(file_settings_list) else None
        record = self._register(path, settings)
        if record is None:
            return None, "", 0, "No video is currently selected."
        return record, path, choice, ""

    def selected_video_summary(self) -> str:
        record, _path, _choice, error = self._get_selected_video_state()
        if len(error) > 0:
            return error
        return f"Selected video: {record.get('media_id')} ({record.get('label', '')})"

    def selected_frame_summary(self) -> str:
        record, path, _choice, error = self._get_selected_video_state()
        if len(error) > 0:
            return error
        fps, _width, _height, frame_count = get_video_info(path)
        if fps <= 0 or frame_count <= 0:
            return f"Unable to read frame info for {record.get('media_id')}."
        current_time = float(self._gen().get("selected_video_time", 0.0) or 0.0)
        frame_no = max(0, min(frame_count - 1, int(round(current_time * fps))))
        return f"Selected frame: {frame_no} on {record.get('media_id')} @ {current_time:.3f}s ({fps} fps, {frame_count} frames)"

    def set_selected_frame(self, frame_no: int) -> str:
        record, path, _choice, error = self._get_selected_video_state()
        if len(error) > 0:
            return error
        fps, _width, _height, frame_count = get_video_info(path)
        if fps <= 0 or frame_count <= 0:
            return f"Unable to read frame info for {record.get('media_id')}."
        frame_no = int(frame_no)
        if frame_no < 0 or frame_no >= frame_count:
            return f"Frame must be between 0 and {frame_count - 1}."
        seconds = frame_no / float(fps)
        self._gen()["selected_video_time"] = seconds
        return f"Selected frame set to {frame_no} on {record.get('media_id')} @ {seconds:.3f}s."

    def clear_media(self) -> None:
        gen = self._gen()
        for source in GALLERY_SOURCES:
            gen[source.files_key].clear()
            gen[source.settings_key].clear()
            gen[source.selected_key] = -1
            gen[source.last_selected_key] = True
        gen["current_gallery_source"] = VIDEO.id
        gen["selected_video_time"] = None
        self._session.media_registry.clear()
        with self._session.turn_lock:
            self._session.pending_chat_media.clear()
