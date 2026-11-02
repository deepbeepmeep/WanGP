# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""MCP server adapter for WanGP's in-process API."""

import argparse
import contextlib
import contextvars
import copy
import dataclasses
import difflib
import hashlib
import io
import json
import logging
import mimetypes
import os
import re
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from postprocessing import parameter_ui
from shared.mesh3d.glb import MESH_EXTENSIONS
from shared.utils.gallery_sources import GALLERY_SOURCES, WORLD_EXTENSIONS, source_for_kind
from shared.utils.gallery_media import disambiguate_gallery_media_ids, gallery_media_ids
from shared.utils.setting_names import settings_error, split_settings
from shared.mcp.config import DEFAULT_PORT, server_port

if TYPE_CHECKING:
    from shared.api import SessionJob


_MAX_STORED_EVENTS = 500
_TIMING_LOG = logging.getLogger("wangp.mcp.timing")
_REQUEST_ID = contextvars.ContextVar("wangp_mcp_request_id", default="inprocess")
_REQUEST_TIMING = contextvars.ContextVar("wangp_mcp_request_timing", default=None)
_MEDIA_TRANSFER_TTL_SECONDS = 600
_MAX_UPLOAD_BYTES = 8 * 1024**3
mimetypes.add_type("model/gltf-binary", ".glb")  # 3D outputs; Python's table lacks the glTF types
mimetypes.add_type("model/gltf+json", ".gltf")
_TRANSPORT_ALIASES = {
    "stdio": "stdio",
    "sse": "sse",
    "streamable-http": "streamable-http",
    "streamable_http": "streamable-http",
}
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
_AUDIO_EXTENSIONS = {".wav", ".mp3", ".aac", ".m4a", ".flac", ".ogg", ".opus"}
_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
_MESH_EXTENSIONS = set(MESH_EXTENSIONS)  # every mesh format the loader reads; the 3D gallery itself stores GLB and splat PLY
_AGENT_GUIDE_PATH = Path(__file__).resolve().parents[2] / "wangp-agent" / "SKILL.md"
_AGENT_INSTRUCTIONS_HEADER = ("WanGP MCP API 2.1: generate and edit images, video, audio and 3D assets with WanGP's models, post-processing and media tools. "
                              "These instructions are also readable as wangp://guides/agent; wangp://guides/workflows indexes workflow sections to read on demand.")
_AGENT_SKILLS_DIR = Path(__file__).resolve().parents[1] / "deepy" / "skills"
_DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"
_DEEPY_VISUAL_TOOL_IDS = {"gen_image", "edit_image", "gen_video", "gen_video_with_speech", "gen_video_with_refs"}
_DEEPY_VIDEO_TOOL_IDS = {"gen_video", "gen_video_with_speech", "gen_video_with_refs"}
_DEEPY_AUDIO_TOOL_IDS = {"gen_song", "gen_speech_from_description", "gen_speech_from_sample"}
_DEEPY_MODEL_DEF_STRING_LIMIT = 256
_TOOLBOX_ACTIONS = {
    "add_to_gallery",
    "create_color_frame",
    "create_mask",
    "image_channels",
    "inspect_media",
    "inspect_video",
    "inspect_audio",
    "extract_image",
    "extract_video",
    "extract_audio",
    "transcribe_media",
    "advanced_transcription",
    "mute_video",
    "remove_vocals",
    "replace_audio",
    "resize_crop",
    "side_by_side",
    "render_3d_views",
    "draw_svg",
    "merge_videos",
    "search_doc",
    "load_doc_section",
    "get_media_details",
}
_TOOLBOX_MEDIA_PARAMETERS = {
    "add_to_gallery": ("path", "paths"),
    "inspect_media": ("media_id", "media_ids"),
    "inspect_video": ("media_id",),
    "view_media": ("media_ids",),
    "inspect_audio": ("media_id",),
    "extract_image": ("media_id",),
    "extract_video": ("media_id",),
    "extract_audio": ("media_id",),
    "transcribe_media": ("media_id",),
    "advanced_transcription": ("media_id",),
    "mute_video": ("media_id",),
    "remove_vocals": ("media_id",),
    "replace_audio": ("video_id", "audio_id"),
    "remux_media": ("video_id", "audio_ids"),
    "resize_crop": ("media_id",),
    "side_by_side": ("media_ids",),
    "render_3d_views": ("media_id",),
    "merge_videos": ("media_ids",),
    "get_media_details": ("media_id",),
    "image_channels": ("media_id",),
    "create_mask": ("media_id",),
}
_POSTPROCESS_PATH_PARAMETERS = {
    "audio_media_id": "audio_path",
    "voice_sample_media_id": "voice_sample_path",
    "voice_sample2_media_id": "voice_sample2_path",
}
_MCP_MEDIA_SETTING_KEYS = {
    "image_start", "image_end", "image_refs", "image_guide", "image_mask",
    "video_guide", "video_guide2", "video_guide3", "video_mask", "video_source",
    "audio_guide", "audio_guide2", "audio_guide3", "audio_source",
    "replace_voice_sample", "replace_voice_sample2", "custom_guide", parameter_ui.IMAGES_SETTING,
}
NESTED_SETTINGS = {"custom_settings", *parameter_ui.SETTINGS.values()}  # the settings holding dictionaries
_GALLERY_LOCK = threading.RLock()


def _read_markdown_section(path: Path, start_heading: str, end_heading: str) -> str:
    text = path.read_text(encoding="utf-8")
    start = text.index(start_heading)
    end = text.find(end_heading, start + len(start_heading))
    return text[start:] if end < 0 else text[start:end].rstrip()


def _register_documentation_resources(mcp, file_access_policy=None, long_text_active: bool = False, api_version: str | int = 1) -> None:
    from shared.deepy.config import python_apps_level
    from shared.deepy.web_search import enabled as web_search_enabled

    from .skills import skill_metadata

    api_version = str(api_version)

    def document_reader(document_path: Path):
        def read_document() -> str:
            return document_path.read_text(encoding="utf-8")
        return read_document

    for path in sorted(_DOCS_DIR.glob("*.md")):
        resource_uri = f"wangp://docs/{path.stem.casefold()}"
        read_document = document_reader(path)
        read_document.__name__ = f"read_{path.stem.casefold()}_documentation"
        description = "WanGP generation settings: model selection, prompts, output dimensions, sampling and guidance, media inputs, acceleration and caching, post-processing, sliding windows, LoRAs, flags, and model API metadata." if path.stem.casefold() == "settings" else f"WanGP {path.stem} documentation."
        if api_version != "1":
            scope = next((line.removeprefix("> Applies to: ").strip() for line in read_document().splitlines() if line.startswith("> Applies to: ")), "")
            if scope:
                description = scope
            else:
                logging.getLogger(__name__).warning("Missing or empty '> Applies to:' scope in %s; using the default documentation description.", path)
        mcp.resource(resource_uri, name=path.stem.casefold(), title=path.stem.replace("_", " ").title(), description=description, mime_type="text/markdown")(read_document)

    for path in sorted(_AGENT_SKILLS_DIR.glob("*/SKILL.md")):
        skill_name = path.parent.name
        if skill_name in {"long-story-writing", "long-generation-prompts"} and not long_text_active:
            continue
        if skill_name in ("web-research", "lora-search") and not web_search_enabled():
            continue
        if skill_name == "web-apps" and python_apps_level() == "off" or skill_name == "progression-design" and python_apps_level() == "off" and not long_text_active:
            continue
        resource_uri = f"wangp://skills/{skill_name}"
        read_skill = document_reader(path)
        read_skill.__name__ = f"read_{skill_name.replace('-', '_')}_skill"
        description = skill_metadata(read_skill(), skill_name)["description"]
        mcp.resource(resource_uri, name=skill_name, title=skill_name.replace("-", " ").title(), description=description, mime_type="text/markdown")(read_skill)

    from shared.utils.plugins import get_deepy_plugin_skill, get_deepy_plugin_skills
    plugin_skills = get_deepy_plugin_skills()
    for skill in plugin_skills:  # the enabled plugins' general skills are listed with WanGP's own
        if skill.routing:
            read_skill = document_reader(Path(skill.path))
            read_skill.__name__ = f"read_{skill.name.replace('-', '_')}_skill"
            mcp.resource(skill.uri, name=skill.name, title=skill.name.replace("-", " ").title(), description=skill.description, mime_type="text/markdown")(read_skill)
    if any(not skill.routing for skill in plugin_skills):
        @mcp.resource("wangp://skills/{plugin}/{name}", name="plugin_toolbox_skill", description="A skill of an enabled plugin's toolbox: the toolbox discovery (wangp_plugins with plugin) names them.", mime_type="text/markdown")
        def read_plugin_toolbox_skill(plugin: str, name: str) -> str:
            skill = get_deepy_plugin_skill(f"wangp://skills/{plugin}/{name}")
            if skill is None:
                raise ValueError(f"Unknown plugin skill: wangp://skills/{plugin}/{name}.")
            return Path(skill.path).read_text(encoding="utf-8")

    @mcp.resource("wangp://docs/settings/prompt-flags", name="prompt_flags", title="WanGP Prompt-Type Flags", description="Exact image_prompt_type, video_prompt_type, and audio_prompt_type flag definitions.", mime_type="text/markdown")
    def read_prompt_flags() -> str:
        return _read_markdown_section(_DOCS_DIR / "SETTINGS.md", "### `image_prompt_type`", "### `prompt_enhancer`")


def _normalize_transport(value: str | None) -> str:
    transport = str(value or "stdio").strip().lower()
    try:
        return _TRANSPORT_ALIASES[transport]
    except KeyError as exc:
        raise RuntimeError(f"Unsupported MCP transport: {value}. Use stdio, sse, or streamable-http.") from exc


@contextlib.contextmanager
def _stdio_safe_startup_output(transport: str):
    if transport != "stdio":
        yield
        return
    target = sys.stderr if sys.stderr is not None else io.StringIO()
    with contextlib.redirect_stdout(target):
        yield


def _artifact_to_dict(artifact: Any) -> dict[str, Any]:
    return {
        "path": artifact.path,
        "media_type": artifact.media_type,
        "client_id": artifact.client_id,
        "hdr": artifact.hdr,
        "audio_sampling_rate": artifact.audio_sampling_rate,
        "fps": artifact.fps,
        "has_video_tensor_uint8": artifact.video_tensor_uint8 is not None,
        "has_video_tensor_hdr": artifact.video_tensor_hdr is not None,
        "has_audio_tensor": artifact.audio_tensor is not None,
        "has_flashvsr_continue_cache": artifact.flashvsr_continue_cache is not None,
    }


def _error_to_dict(error: Any) -> dict[str, Any]:
    lora_hint = " Compare each URL with the template's activated_loras or wangp_model(action=\"loras\"): a mistyped name cannot be downloaded." if "Loras files could not be downloaded" in str(error.message) else ""
    return {
        "message": error.message + lora_hint,
        "task_index": error.task_index,
        "task_id": error.task_id,
        "stage": error.stage,
        "cancelled": error.cancelled,
    }


def _result_to_dict(result: Any, file_store=None, job_id="") -> dict[str, Any]:
    return {
        "success": result.success,
        "cancelled": result.cancelled,
        "generated_files": list(result.generated_files),
        "errors": [_error_to_dict(error) for error in result.errors],
        "total_tasks": result.total_tasks,
        "successful_tasks": result.successful_tasks,
        "failed_tasks": result.failed_tasks,
        "artifacts": [{**_artifact_to_dict(artifact), **({"side_files": file_store.save_companions(artifact, f"{job_id}/{index}")} if file_store is not None and artifact.side_files else {})} for index, artifact in enumerate(result.artifacts)],
    }


def _event_to_dict(event: Any) -> dict[str, Any]:
    return {
        "kind": event.kind,
        "timestamp": event.timestamp,
        "data": _json_safe(event.data),
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    class_name = type(value).__name__
    if class_name == "GeneratedArtifact":
        return _artifact_to_dict(value)
    if class_name == "GenerationError":
        return _error_to_dict(value)
    if class_name == "GenerationResult":
        return _result_to_dict(value)
    if class_name == "SessionEvent":
        return _event_to_dict(value)
    if class_name == "StreamMessage":
        return {"stream": value.stream, "text": value.text}
    if class_name == "PreviewUpdate":
        return {
            "has_image_preview": value.image is not None,
            "phase": value.phase,
            "status": value.status,
            "progress": value.progress,
            "current_step": value.current_step,
            "total_steps": value.total_steps,
        }
    if dataclasses.is_dataclass(value):
        return {field.name: _json_safe(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted((_json_safe(item) for item in value), key=str)
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    if shape is not None or dtype is not None:
        return {"type": type(value).__name__, "shape": list(shape) if shape is not None else None, "dtype": str(dtype) if dtype is not None else None}
    return str(value)


def _gallery_media_type(path: str, gallery: str) -> str:
    if gallery in ("audio", "3d", "world"):
        return gallery
    suffix = Path(path).suffix.lower()
    if suffix in _IMAGE_EXTENSIONS:
        return "image"
    if suffix in _AUDIO_EXTENSIONS:
        return "audio"
    return "video"


def _gallery_history(session) -> dict[str, dict[str, Any]]:
    history = getattr(session, "_mcp_gallery_history", None)
    if history is None:
        history = {}
        session._mcp_gallery_history = history
    return history


def _gallery_path_exists(session, path: str) -> bool:
    candidate = Path(str(path or ""))
    if not candidate.is_absolute():
        candidate = Path(getattr(session, "_root", Path.cwd())) / candidate
    return candidate.is_file()


def _prune_deleted_gallery_files(session, source: Path, assistant_session) -> bool:
    """Drop vanished sources after an authorized Prime delete or move."""
    def missing(path):
        target = (session._root / path).resolve()
        return (target == source or source in target.parents) and not target.is_file()

    changed = False
    gen = session._state["gen"]
    for paths_key, settings_key, selected_key in ((source.files_key, source.settings_key, source.selected_key) for source in GALLERY_SOURCES):
        paths = gen.get(paths_key, [])
        kept = [index for index, path in enumerate(paths) if not missing(path)]
        if len(kept) == len(paths):
            continue
        settings = gen[settings_key]
        selected = gen.get(selected_key, -1)
        paths[:] = [paths[index] for index in kept]
        settings[:] = [settings[index] for index in kept]
        gen[selected_key] = kept.index(selected) if selected in kept else min(selected, len(kept) - 1)
        if selected_key == "selected" and selected not in kept:
            gen["selected_video_time"] = 0
        changed = True
    history = _gallery_history(session)
    for media_id, record in list(history.items()):
        if missing(record["path"]):
            del history[media_id]
            changed = True
    registry = assistant_session.media_registry
    kept_records = [record for record in registry if not missing(record["path"])]
    changed |= len(kept_records) != len(registry)
    registry[:] = kept_records
    return changed


def _gallery_records(session, media_type: str = "all", limit: int = 50, *, viewed: bool = False) -> list[dict[str, Any]]:
    """viewed registers only the selection of the gallery shown in WanGP: the user's, for an agent recording elsewhere."""
    requested_type = str(media_type or "all").strip().lower()
    if requested_type not in {"all", "image", "video", "audio", "3d", "world"}:
        raise ValueError("media_type must be all, image, video, audio, 3d, or world")
    limit = max(1, min(int(limit), 500))
    gen = session._state["gen"]
    current_records = []
    workspace_id = getattr(session, '_workspace_id', None)
    if workspace_id and not viewed:
        from shared.utils.workspace_storage import workspace_gallery
        gen = workspace_gallery(gen, workspace_id)
    gallery_defs = [(source.media_ids, list(gen.get(source.files_key, []) or []), list(gen.get(source.settings_key, []) or []), int(gen.get(source.selected_key, -1))) for source in GALLERY_SOURCES]
    for gallery, paths, settings_list, selected_index in gallery_defs:
        for index, path in enumerate(paths):
            if viewed and index != selected_index:
                continue
            resolved_path = str(path or "").strip()
            item_type = _gallery_media_type(resolved_path, gallery)
            settings = settings_list[index] if index < len(settings_list) and isinstance(settings_list[index], dict) else {}
            ids = gallery_media_ids(resolved_path, gallery, settings, root=session._root)
            record = {
                "media_id": ids[0],
                "gallery": gallery,
                "index": index,
                "path": resolved_path,
                "media_type": item_type,
                "selected": index == selected_index,
                "in_gallery": True,
                "settings": {**_json_safe(settings), "gallery_media_ids": ids},
            }
            if record["selected"] and gallery == "visual" and item_type == "video":
                record["current_time_seconds"] = gen.get("selected_video_time")
            current_records.append(record)
    with _GALLERY_LOCK:
        history = _gallery_history(session)
        for media_id, record in list(history.items()):
            if not _gallery_path_exists(session, record.get("path", "")):
                history.pop(media_id, None)
                continue
            if viewed:  # the agent's own media stay as listed; only the user's selection counts
                record["selected"] = False
                continue
            record.update({"index": None, "selected": False, "in_gallery": False})
            record.pop("current_time_seconds", None)
        combined_records = [*history.values(), *current_records]
        id_lists = disambiguate_gallery_media_ids([(record["path"], record["gallery"], record["settings"]) for record in combined_records], root=session._root)
        history.clear()
        for record, ids in zip(combined_records, id_lists):
            record["media_id"] = ids[0]
            record["settings"]["gallery_media_ids"] = ids
            for media_id in ids:
                history.pop(media_id, None)
            history[record["media_id"]] = copy.deepcopy(record)
        records = [copy.deepcopy(record) for record in history.values() if requested_type == "all" or record["media_type"] == requested_type]
    return records[-limit:]


def _compact_gallery_stats(media_type: str, path: str) -> dict[str, Any]:
    if media_type == "video":
        from shared.utils.video_decode import probe_video_stream_metadata

        stat = Path(path).stat()
        metadata = probe_video_stream_metadata(path, file_version=(stat.st_mtime_ns, stat.st_size))
        if metadata is None:
            return {}
        width, height = int(metadata.get("display_width", 0) or 0), int(metadata.get("display_height", 0) or 0)
        frame_count = int(metadata.get("frame_count", 0) or 0)
        fps = float(metadata.get("fps_float", 0) or 0)
        duration = float(metadata.get("duration", 0) or 0)
        if duration <= 0 and frame_count > 0 and fps > 0:
            duration = frame_count / fps
        stats = {}
        if width > 0 and height > 0:
            stats["resolution"] = f"{width}x{height}"
        if frame_count > 0:
            stats["frame_count"] = frame_count
        if fps > 0:
            stats["fps"] = int(fps) if fps.is_integer() else round(fps, 3)
        if duration > 0:
            stats["duration_seconds"] = int(duration) if duration.is_integer() else round(duration, 3)
        return stats

    if media_type == "3d":
        from shared.utils.mesh_outputs import asset_summary

        summary = asset_summary(path)
        return {key: value for key, value in summary.items() if key != "frame_count" or value > 1}

    if media_type == "world":
        from shared.utils.world_outputs import world_summary

        return world_summary(path, segments=False)

    from shared.deepy.filesystem import file_info

    metadata = file_info(path)
    stats = {}
    if media_type == "image" and metadata["status"] == "done":
        stats["resolution"] = metadata["resolution"]
        if metadata["frame_count"] > 1:
            stats["frame_count"] = metadata["frame_count"]
    duration = metadata.get("duration_seconds")
    if duration is not None and duration > 0:
        stats["duration_seconds"] = int(duration) if duration.is_integer() else round(duration, 3)
    return stats


def _compact_gallery_record(record: dict[str, Any]) -> dict[str, Any]:
    from shared.deepy.media_registry import summarize_prompt

    media_id = str(record.get("media_id", "") or "").strip()
    media_type = str(record.get("media_type", "") or "").strip()
    settings = record.get("settings") if isinstance(record.get("settings"), dict) else {}
    prompt = str(settings.get("prompt", "") or "").strip()
    client_id = str(settings.get("client_id", "") or "").strip().lower()
    result = {
        "media_id": media_id,
        "gallery": record.get("gallery", ""),
        "media_type": media_type,
        "filename": Path(str(record.get("path", "") or "")).name,
        # Media made by a media tool describe the operation, also once post-processed; they keep their source's prompt in their settings.
        "description": summarize_prompt(settings.get("comments") or prompt, media_type),
        "source": "MCP" if client_id.startswith(("ai_mcp", "mcp_")) else "Deepy" if client_id.startswith("ai") else "WanGP",
        "selected": bool(record.get("selected", False)),
        "in_gallery": bool(record.get("in_gallery", False)),
    }
    if record.get("current_time_seconds") is not None:
        result["current_time_seconds"] = record["current_time_seconds"]
    if settings.get("output_role"):
        result["role"] = settings["output_role"]
    if settings.get("client_id"):
        result["task_id"] = settings["client_id"]
    if "window_no" in settings:
        result["window_no"] = settings["window_no"]
    result.update(_compact_gallery_stats(media_type, str(record.get("path", "") or "")))
    return result


def _output_record(item, path=None):
    """A produced output, without the Gallery's live selection and player position, nor what its media_id and path already say (gallery, source,
    and the filename once the path is given)."""
    record = {key: value for key, value in item.items() if key not in ("selected", "in_gallery", "current_time_seconds", "gallery", "source")}
    if path is not None:
        record.pop("filename")
        record["path"] = path
    return record


def _gallery_item(session, media_id: str) -> dict[str, Any]:
    lookup = str(media_id or "").strip().lower()
    _gallery_records(session, limit=500)
    with _GALLERY_LOCK:
        record = next((record for record in _gallery_history(session).values() if lookup in record["settings"]["gallery_media_ids"]), None)
        if record is not None and _gallery_path_exists(session, record.get("path", "")):
            return copy.deepcopy(record)
        known = [identifier for record in _gallery_history(session).values() for identifier in record["settings"]["gallery_media_ids"]]
    known = [identifier for identifier in known if identifier != lookup]  # an id whose file is gone has no alternative
    close = sorted({identifier for identifier in known if identifier.startswith(lookup)}) or difflib.get_close_matches(lookup, known, n=2, cutoff=0.9)  # a truncated or mistyped id
    raise KeyError(f"Unknown WanGP media_id: {media_id}" + (f". Did you mean {' or '.join(close[:2])}?" if close and len(lookup) > 8 else ""))


def resolve_gallery_media_path(session, media_id: str) -> str:
    path = Path(_gallery_item(session, media_id)["path"])
    return str((path if path.is_absolute() else Path(session._root) / path).resolve())


def _extract_media_settings(session, path: str) -> dict[str, Any]:
    runtime = session._ensure_runtime()
    with contextlib.chdir(runtime.root):
        settings, _any_visual, _any_audio = runtime.module.get_settings_from_file(session._state, path, False, False, False)
    return settings if isinstance(settings, dict) else {}


def _media_settings(session, *, media_id: str | None = None, path: str | None = None, allow_read_file_system: bool = False, file_access_policy=None) -> dict[str, Any]:
    media_id = str(media_id or "").strip()
    path = str(path or "").strip()
    if bool(media_id) == bool(path):
        raise ValueError("Provide exactly one of media_id or path.")
    if media_id:
        if not _is_media_id(media_id):
            raise ValueError("media_id must be returned by wangp_list_gallery.")
        record = _gallery_item(session, media_id)
        resolved_path = _resolve_existing_path(record["path"], "Gallery media")
        settings = record.get("settings") if isinstance(record.get("settings"), dict) else {}
        if not settings:
            settings = _extract_media_settings(session, resolved_path)
        return {"status": "done", "source": "gallery", "media_id": media_id, "media_type": record["media_type"], "filename": Path(resolved_path).name, **_reusable_media_settings(settings)}
    if not allow_read_file_system:
        raise PermissionError("Direct filesystem paths are disabled for this MCP server. Use a media_id returned by wangp_list_gallery.")
    if _is_media_id(path):
        raise ValueError("Use media_id, not path, for Gallery media.")
    resolved_path = str(file_access_policy.require_read(path, file=True)) if file_access_policy is not None else _resolve_existing_path(path, "media path")
    media_type = _mcp_media_type(resolved_path)
    if not media_type:
        raise ValueError(f"Unsupported media file extension: {Path(resolved_path).suffix}")
    return {"status": "done", "source": "filesystem", "media_id": "", "path": resolved_path, "media_type": media_type, "filename": Path(resolved_path).name, **_reusable_media_settings(_extract_media_settings(session, resolved_path))}


def _reusable_media_settings(settings: dict[str, Any]) -> dict[str, Any]:
    settings, metadata = split_settings(settings)
    return {"settings": _json_safe(settings), **({"metadata": _json_safe(metadata)} if metadata else {})}


def _compact_model_metadata(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact = []
    for record in records:
        item = copy.deepcopy(record)
        item.pop("setting_values", None)
        compact.append(item)
    return compact


def _compact_deepy_model_metadata(record: dict[str, Any]) -> dict[str, Any]:
    compact = {key: copy.deepcopy(record[key]) for key in ("model_type", "name", "family", "family_label", "base_model_type", "finetune", "main_output", "outputs", "inputs") if record.get(key) not in (None, "", [], {})}
    compact["capabilities"] = [key for key, enabled in record.get("capabilities", {}).items() if enabled]
    compact.update({key: copy.deepcopy(record[key]) for key in ("accelerated", "size", "specialities", "matched_specialities", "unmatched_specialities", "word_matches", "substring_matches", "resolution_budget", "resolution_budget_tiled", "resolution_min", "max_width", "max_height", "lock_resolutions", "allowed_resolutions", "vae_block_size", "resolution_assessment") if key in record})
    if record.get("sliding_window") and "sliding_window" not in compact["capabilities"]:
        compact["capabilities"].append("sliding_window")
    compact["media_inputs"] = {kind: copy.deepcopy(values) if kind == "custom_file" else [key for key, enabled in values.items() if enabled] for kind, values in record.get("media_inputs", {}).items() if isinstance(values, dict) and any(values.values())}
    return compact


def _compact_deepy_model_schema(schema: dict[str, Any] | None) -> dict[str, Any] | None:
    if schema is None:
        return None
    metadata = schema["metadata"]
    compact = _compact_deepy_model_metadata(metadata)
    compact.update({key: metadata[key] for key in ("description", "fps", "frames_minimum", "frames_steps", "frames_maximum") if metadata.get(key) not in (None, "")})
    return {"metadata": compact}


# Loading, packaging and UI-enhancer declarations: listed by name in previewed root definitions, still readable by property.
_INTERNAL_DEFINITION_KEYS = {"URLs", "URLs2", "path", "architecture", "metadata", "profiles_dir", "text_encoder_URLs", "text_encoder_folder", "finetune_custom_urls",
                             "finetunes_infos", "finetunes_params", "dtype", "device_explicit", "cuda_graphs", "tiny_vae_architecture", "qkv_splitting", "qkv_layout"}


def _internal_declaration(key: str, value: Any) -> bool:
    return key in _INTERNAL_DEFINITION_KEYS or key.startswith(("_", "system_configs")) or "prompt_enhancer" in key or isinstance(value, str) and value.startswith(("http://", "https://"))


def _mcp_model_definition(model_def: dict[str, Any] | None, property_name: str | None = None, string_limit: int | None = None, help_detail: str | None = None) -> dict[str, Any] | None:
    """help_detail "agent" serves deepy_infos / deepy_prompt_infos under the canonical names, "full" the UI help alone; None keeps both declarations."""
    if model_def is None:
        return None
    result = copy.deepcopy(model_def)
    result.pop("settings", None)
    if help_detail is not None:
        for key in ("infos", "prompt_infos"):
            agent_text = result.pop(f"deepy_{key}", None)
            if help_detail == "agent" and agent_text is not None:
                result[key] = agent_text
    if property_name is not None:
        if property_name not in result:
            raise KeyError(f"Unknown model definition property: {property_name}")
        return {property_name: result[property_name]}
    if string_limit is not None:
        internal = sorted(key for key, value in result.items() if _internal_declaration(key, value))
        result = {key: value for key, value in result.items() if key not in internal}
        for key, value in result.items():
            if isinstance(value, str) and len(value) > string_limit:
                result[key] = f'{value[:string_limit]} [Truncated, query property "{key}" for full {len(value)} characters]'
        if internal:
            result["other_declarations"] = internal
    return result


def _deepy_ui_settings(session) -> dict[str, Any]:
    from shared.deepy import ui_settings as deepy_ui_settings

    assistant_session = session._state.get("assistant_session")
    live_settings = getattr(assistant_session, "tool_ui_settings", None)
    return deepy_ui_settings.normalize_assistant_tool_ui_settings(**live_settings) if isinstance(live_settings, dict) and live_settings else deepy_ui_settings.get_persisted_assistant_tool_ui_settings()


def _deepy_template_defaults(session) -> dict[str, str]:
    from shared.deepy import ui_settings as deepy_ui_settings

    settings = _deepy_ui_settings(session)
    return {tool: settings[key] for tool, key in deepy_ui_settings.TEMPLATE_TOOL_UI_KEY.items()}


def _deepy_template_catalog(session, tool_id: str | None = None) -> list[dict[str, Any]]:
    from shared.deepy import tool_settings as deepy_tool_settings

    requested_tool = str(tool_id or "").strip()
    if requested_tool and requested_tool not in deepy_tool_settings.GENERATION_TOOL_IDS:
        raise ValueError(f"tool_id must be one of: {', '.join(deepy_tool_settings.GENERATION_TOOL_IDS)}")
    defaults = _deepy_template_defaults(session)
    tool_ids = [requested_tool] if requested_tool else list(deepy_tool_settings.GENERATION_TOOL_IDS)
    catalog = []
    for one_tool_id in tool_ids:
        default_template = defaults[one_tool_id]
        templates = [
            {"template": variant, "label": label, "default": variant == default_template}
            for label, variant in deepy_tool_settings.list_tool_variant_choices(one_tool_id, current_variant=default_template)
        ]
        catalog.append({"tool_id": one_tool_id, "label": deepy_tool_settings.TOOL_DISPLAY_NAMES[one_tool_id], "default_template": default_template, "templates": templates})
    return catalog


def _deepy_general_properties(session) -> dict[str, Any]:
    settings = _deepy_ui_settings(session)
    return {key: settings[key] for key in ("use_template_properties", "width", "height", "num_frames", "audio_duration", "seed")}


def _apply_deepy_general_properties(tool_id: str, settings: dict[str, Any], properties: dict[str, Any]) -> dict[str, Any]:
    effective = dict(settings)
    effective["seed"] = properties["seed"]
    if tool_id in _DEEPY_VISUAL_TOOL_IDS:
        effective.pop("width", None)
        effective.pop("height", None)
        effective["resolution"] = f"{properties['width']}x{properties['height']}"
        if tool_id in _DEEPY_VIDEO_TOOL_IDS:
            effective.pop("num_frames", None)
            effective["video_length"] = properties["num_frames"]
    elif tool_id in _DEEPY_AUDIO_TOOL_IDS:
        effective.pop("audio_duration", None)
        effective["duration_seconds"] = properties["audio_duration"]
    return effective


def _strip_deepy_settings_metadata(settings: dict[str, Any]) -> dict[str, Any]:
    stripped = dict(settings)
    stripped.pop("settings_version", None)
    stripped.pop("type", None)
    stripped.pop("profile_priority", None)
    return stripped


def _strip_deepy_fixed_image_mode(session, settings: dict[str, Any], model_type: str | None = None) -> dict[str, Any]:
    stripped = dict(settings)
    metadata = session.get_model_metadata(str(model_type or stripped.get("model_type", "") or "")) or {}
    outputs = {str(output).casefold() for output in metadata.get("main_output", [])}
    if "image" in outputs and "video" not in outputs and stripped.get("image_mode") != 2:
        stripped.pop("image_mode", None)
    return stripped


def _deepy_prompt_mode(session, model_type: str, mode: Any = None) -> str:
    # A Deepy task is one generation: separate-request modes become one prompt,
    # or one sliding window per line/paragraph on sliding-window video models.
    if mode not in (None, "G", "PG"):
        return mode
    if (session.get_model_def(model_type) or {}).get("sliding_window", False):
        return "W" if mode == "G" else "PW"
    return "FG"


def _deepy_task_settings(session, settings: dict[str, Any], model_type: str | None = None) -> dict[str, Any]:
    stripped = _strip_deepy_fixed_image_mode(session, _strip_deepy_settings_metadata(settings), model_type)
    if "multi_prompts_gen_type" in stripped:
        stripped["multi_prompts_gen_type"] = _deepy_prompt_mode(session, str(model_type or stripped.get("model_type", "") or ""), stripped["multi_prompts_gen_type"])
    return stripped


def _relevant_deepy_general_properties(tool_id: str, general_properties: dict[str, Any]) -> dict[str, Any]:
    if tool_id in _DEEPY_VIDEO_TOOL_IDS:
        keys = ("width", "height", "num_frames", "seed")
    elif tool_id in _DEEPY_VISUAL_TOOL_IDS:
        keys = ("width", "height", "seed")
    elif tool_id in _DEEPY_AUDIO_TOOL_IDS:
        keys = ("audio_duration", "seed")
    else:
        keys = ()
    return {key: general_properties[key] for key in keys}


def _deepy_template_settings(session, tool_id: str, template: str) -> dict[str, Any]:
    from shared.deepy import tool_settings as deepy_tool_settings

    catalog = _deepy_template_catalog(session, tool_id=tool_id)[0]
    requested_template = str(template or "").strip()
    resolved_template = catalog["default_template"] if requested_template.casefold() == "default" else deepy_tool_settings.find_tool_variant(tool_id, requested_template, current_variant=catalog["default_template"])
    if resolved_template is None:
        raise ValueError(f"Unknown Deepy template '{template}' for tool '{tool_id}'.")
    template_settings = session.merge_settings_with_defaults(deepy_tool_settings.clone_tool_preset(tool_id, resolved_template))
    general_properties = _deepy_general_properties(session)
    effective_settings = session.prepare_settings_for_export(template_settings)
    if not general_properties["use_template_properties"]:
        effective_settings = _apply_deepy_general_properties(tool_id, effective_settings, general_properties)
    effective_settings = _deepy_task_settings(session, effective_settings)
    result = {
        "tool_id": tool_id,
        "template": resolved_template,
        "default": resolved_template == catalog["default_template"],
        "default_template": catalog["default_template"],
        "general_properties_active": not general_properties["use_template_properties"] and tool_id in _DEEPY_VISUAL_TOOL_IDS | _DEEPY_AUDIO_TOOL_IDS,
        "settings": _json_safe(effective_settings),
    }
    if not general_properties["use_template_properties"]:
        result["general_properties"] = _json_safe(_relevant_deepy_general_properties(tool_id, general_properties))
    return result


def _resolve_existing_path(value: Any, label: str) -> str:
    path = Path(str(value or "").strip()).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {path}")
    return str(path)


def _mcp_media_type(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in _IMAGE_EXTENSIONS:
        return "image"
    if suffix in _VIDEO_EXTENSIONS:
        return "video"
    if suffix in _AUDIO_EXTENSIONS:
        return "audio"
    if suffix in _MESH_EXTENSIONS:
        return "3d"
    if suffix in WORLD_EXTENSIONS:
        return "world"
    return ""


def _is_media_id(value: Any) -> bool:
    return str(value or "").strip().lower().startswith(tuple(source.media_ids + ":" for source in GALLERY_SOURCES))


def _resolve_mcp_media_reference(session, value: Any, label: str, allow_read_file_system: bool, file_access_policy=None) -> str:
    reference = str(value or "").strip()
    if not reference:
        raise ValueError(f"{label} is empty")
    if _is_media_id(reference):
        path = _gallery_item(session, reference)["path"]
        return _resolve_existing_path(path, label)
    if not allow_read_file_system:
        raise PermissionError(f"Direct filesystem paths are disabled for this MCP server. Use a media_id returned by wangp_list_gallery for {label}, or restart with filesystem reads enabled.")
    return str(file_access_policy.require_read(reference, file=True)) if file_access_policy is not None else _resolve_existing_path(reference, label)


def _resolve_mcp_media_input(session, media_id: str | None, path: str | None, allow_read_file_system: bool, file_access_policy=None) -> str:
    media_id, path = str(media_id or "").strip(), str(path or "").strip()
    if bool(media_id) == bool(path):
        raise ValueError("Provide exactly one of media_id or path.")
    if media_id and not _is_media_id(media_id):
        raise ValueError("media_id must be returned by wangp_list_gallery.")
    if path and _is_media_id(path):
        raise ValueError("Use media_id, not path, for Gallery media.")
    return _resolve_mcp_media_reference(session, media_id or path, "media_id" if media_id else "path", allow_read_file_system, file_access_policy)


def _resolve_mcp_media_value(session, value: Any, label: str, allow_read_file_system: bool, file_access_policy=None) -> Any:
    if value is None or isinstance(value, str) and not value.strip():
        return value
    if isinstance(value, (list, tuple)):
        return [_resolve_mcp_media_reference(session, item, label, allow_read_file_system, file_access_policy) for item in value]
    return _resolve_mcp_media_reference(session, value, label, allow_read_file_system, file_access_policy)


def _validate_generation_media(session, settings, model_type, path):
    """Reject media that settings cleanup would discard; MCP v2 only."""
    from shared.api import _declared_choice_values, _has_media_setting, apply_media_flag_defaults

    supplied = {key for key in _MCP_MEDIA_SETTING_KEYS & settings.keys() if _has_media_setting(settings[key])}
    if not supplied:
        return
    model_def = session.get_model_def(model_type)
    if model_def is None:
        raise ValueError(f"{path}: Unknown model_type: {model_type}. Nothing was submitted.")
    normalized = dict(settings)
    if normalized.get("model_type"):
        apply_media_flag_defaults(normalized, model_def)
    effective = dict(session.get_default_settings(model_type))
    effective.update({key: value for key, value in normalized.items() if value is not None or key not in effective})
    image_mode = int(effective.get("image_mode", 0) or 0)
    image_output = model_def.get("image_outputs", False) or image_mode > 0
    image_flags = str(effective.get("image_prompt_type", "") or "")
    video_flags = str(effective.get("video_prompt_type", "") or "")
    audio_flags = str(effective.get("audio_prompt_type", "") or "")
    allowed_image = model_def.get("image_prompt_types_allowed", "")
    custom_choices = model_def.get("guide_custom_choices_image") if image_mode > 0 else None
    if custom_choices is None:
        custom_choices = model_def.get("guide_custom_choices")
    guide_choices = _declared_choice_values(custom_choices) + _declared_choice_values(model_def.get("guide_preprocessing")) + _declared_choice_values(model_def.get("custom_video_selection"))
    if image_output and model_def.get("inpaint_support", False):
        guide_choices.append(model_def.get("inpaint_video_prompt_type", "VAG"))
    reference_choices = [value for value in _declared_choice_values(model_def.get("image_ref_choices")) + guide_choices if "I" in value]
    audio_choices = model_def.get("audio_prompt_type_sources")
    audio_values = _declared_choice_values(audio_choices)
    has_audio = model_def.get("any_audio_prompt", False) and not image_output
    has_control = any("V" in value for value in guide_choices)
    if image_mode > 0 and custom_choices is not None:
        selected = "".join(flag for flag in custom_choices["letters_filter"] if flag in video_flags)
        if selected and selected not in _declared_choice_values(custom_choices):
            video_flags = "".join(flag for flag in video_flags if flag not in custom_choices["letters_filter"])
    control_active = "V" in video_flags
    has_mask = any("A" in value for value in _declared_choice_values(model_def.get("mask_preprocessing")) + guide_choices) or model_def.get("inpaint_support", False)
    mask_active = control_active and "A" in video_flags and "U" not in video_flags

    def reject(key, reason):
        raise ValueError(f"{path}.{key}: {reason} (model {model_type}). Nothing was submitted.")

    checks = {
        "image_start": ("S" in allowed_image, "S" in image_flags, "image_prompt_type containing S"),
        "image_end": ("E" in allowed_image, "E" in image_flags and (model_def.get("end_frames_always_enabled", False) or any(flag in image_flags for flag in "SVL")), "an enabled end-image mode"),
        "video_source": ("V" in allowed_image and not image_output, "V" in image_flags, "image_prompt_type containing V"),
        # An image / 3D model with reference videos reads its video guide instead of a control image.
        "image_guide": (has_control and image_output and not model_def.get("reference_video_enabled", False), control_active, "an enabled Control Image mode"),
        "video_guide": (has_control and (not image_output or model_def.get("reference_video_enabled", False)), control_active, "an enabled Control or Reference Video mode"),
        "video_guide2": (any("+" in value for value in guide_choices) and not image_output, control_active and "+" in video_flags, "an enabled two-video mode"),
        "video_guide3": (any("*" in value for value in guide_choices) and not image_output, control_active and "*" in video_flags, "an enabled three-video mode"),
        "image_mask": (has_mask and image_output, mask_active, "an enabled Control Image mask mode"),
        "video_mask": (has_mask and not image_output, mask_active, "an enabled Control Video mask mode"),
        "audio_guide": (has_audio and (audio_choices is None or any("A" in value for value in audio_values)), "A" in audio_flags, "audio_prompt_type containing A"),
        "audio_guide2": (has_audio and (any("B" in value for value in audio_values) if audio_choices is not None else not model_def.get("one_speaker_only", False) and not model_def.get("audio_only", False)), "B" in audio_flags, "audio_prompt_type containing B"),
        "audio_guide3": (has_audio and audio_choices is not None and any("D" in value for value in audio_values), "D" in audio_flags, "audio_prompt_type containing D"),
        "custom_guide": (model_def.get("custom_guide") is not None, True, "a declared custom guide"),
    }
    for key in sorted(supplied & checks.keys()):
        supported, active, required = checks[key]
        if not supported:
            reference = key in ("image_start", "image_end", "image_guide") and reference_choices and "image_refs" not in supplied
            reject(key, "input is unsupported in this model/output mode" + ("; this model takes its picture in image_refs" if reference else ""))
        if not active:
            reject(key, f"input would be ignored; requires {required}")
    if "image_refs" in supplied:
        injection = "F" in video_flags
        choices = [value for value in reference_choices if ("F" in value) == injection]
        if not choices:
            injection_choices = [value for value in reference_choices if "F" in value]
            if injection_choices:
                reject("image_refs", f"only frame injection is supported: video_prompt_type={injection_choices[0]!r} with frames_positions; general reference-image conditioning is unsupported")
            reject("image_refs", "frame injection is unsupported" if injection else "reference-image conditioning is unsupported")
        if not any(set(value) <= set(video_flags) for value in choices):
            reject("image_refs", f"input would be ignored or uses an unsupported mode; video_prompt_type must include one of {choices}")
    for key, method_key, capability in (("audio_source", "postprocess_audio", "needs_audio_source"), ("replace_voice_sample", "replace_voice_method", "needs_voice_sample"), ("replace_voice_sample2", "replace_voice_method", "needs_voice_sample2")):
        if key in supplied:
            from postprocessing.audio_processors import method_metadata
            if image_output or not method_metadata(effective.get(method_key, ""))[capability]:
                reject(key, f"input requires a {method_key} processor that uses it")
    if parameter_ui.IMAGES_SETTING in supplied:  # read by the post-processors declaring an images parameter (a 3D refiner's or texturer's Object Image)
        methods = {slot: parameter_ui.method_of(slot, effective.get("spatial_upsampling" if slot == parameter_ui.SPATIAL else "temporal_upsampling", "")) for slot in parameter_ui.SETTINGS}
        if parameter_ui.images_parameter(*(parameter_ui.slot_parameters(slot, method) for slot, method in methods.items())) is None:
            reject(parameter_ui.IMAGES_SETTING, "input requires a spatial or temporal post-processor that uses reference images")


def _resolve_generation_media(session, source: dict[str, Any] | list[dict[str, Any]], allow_read_file_system: bool, file_access_policy=None):
    resolved = copy.deepcopy(source)

    def resolve_task(task: dict[str, Any]) -> None:
        if isinstance(task.get("tasks"), list):
            for child in task["tasks"]:
                if not isinstance(child, dict):
                    raise TypeError("Manifest tasks must be dictionaries")
                resolve_task(child)
            return
        settings = task.get("params") if isinstance(task.get("params"), dict) else task.get("settings") if isinstance(task.get("settings"), dict) else task
        for key in _MCP_MEDIA_SETTING_KEYS & settings.keys():
            settings[key] = _resolve_mcp_media_value(session, settings[key], key, allow_read_file_system, file_access_policy)
        if not allow_read_file_system:
            for lora in settings.get("activated_loras", []) or []:
                lora_text = str(lora or "").strip()
                lora_path = Path(lora_text)
                if lora_path.is_absolute() or ".." in lora_text.replace("\\", "/").split("/") or lora_text.lower().startswith("file:"):
                    raise PermissionError("Direct filesystem paths are disabled for activated_loras. Use identifiers returned by wangp_list_loras.")
        elif file_access_policy is not None:
            for lora in settings.get("activated_loras", []) or []:
                lora_text = str(lora or "").strip()
                lora_path = Path(lora_text)
                if lora_path.is_absolute() or ".." in lora_text.replace("\\", "/").split("/") or lora_text.lower().startswith("file:"):
                    file_access_policy.require_read(lora_text.removeprefix("file:"), file=True)

    if isinstance(resolved, list):
        for task in resolved:
            if not isinstance(task, dict):
                raise TypeError("Generation tasks must be dictionaries")
            resolve_task(task)
    else:
        resolve_task(resolved)
    return resolved


def _register_gallery_media(session, path: str) -> dict[str, Any]:
    from shared.utils.workspace_storage import background_workspace, host_for_gen, recording_gallery

    resolved_path = _resolve_existing_path(path, "uploaded media")
    workspace_id = getattr(session, '_workspace_id', None)
    host = host_for_gen(session._state['gen']) if workspace_id else None
    background = host is not None and host.background_agents
    with background_workspace(host, workspace_id) if background else contextlib.nullcontext():
        if host is not None:
            if not background:
                host.activate_workspace(workspace_id)
            resolved_path = host.workspaces.adopt(workspace_id, [resolved_path])[0]
        media_type = _mcp_media_type(resolved_path)
        if not media_type:
            raise ValueError(f"Unsupported media file extension: {Path(resolved_path).suffix}")
        if media_type == "image":
            from PIL import Image

            with Image.open(resolved_path) as image:
                image.verify()
        elif media_type == "video":
            from shared.utils.video_decode import probe_video_stream_metadata

            if probe_video_stream_metadata(resolved_path) is None:
                raise ValueError(f"Unable to read uploaded video: {Path(resolved_path).name}")
        elif media_type == "3d":
            from shared.utils.mesh_outputs import asset_stats

            asset_stats(resolved_path)
        elif media_type == "world":
            from shared.utils.world_outputs import read_manifest

            read_manifest(resolved_path)
        else:
            from shared.utils.audio_video import get_audio_file_sample_rate

            get_audio_file_sample_rate(resolved_path)
        runtime = session._ensure_runtime()
        with contextlib.chdir(runtime.root):
            settings, _any_visual, _any_audio = runtime.module.get_settings_from_file(session._state, resolved_path, False, False, False)
        settings = settings if isinstance(settings, dict) else {}
        gen = recording_gallery(session._state["gen"])
        with _GALLERY_LOCK:
            source = source_for_kind(media_type)
            paths, settings_list, selected_key = gen[source.files_key], gen[source.settings_key], source.selected_key
            # the media tools record configured paths, which resolve() may spell in another case on Windows
            index = next((index for index, path in enumerate(paths) if os.path.normcase(os.path.abspath(path)) == os.path.normcase(resolved_path)), None)
            if index is not None:
                resolved_path = paths[index]
                if index < len(settings_list) and settings and not settings_list[index]:
                    settings_list[index] = settings
            else:
                paths.append(resolved_path)
                settings_list.append(settings)
                index = len(paths) - 1
            if gen is session._state["gen"]:  # an agent recording in the background leaves the selection to the user
                gen[selected_key] = index
    return next(record for record in _gallery_records(session, limit=500) if record["path"] == resolved_path)


class _MediaTransferStore:
    def __init__(self, session, policy=None, *, http=True) -> None:
        self.session = session
        self.policy = policy
        self.http = http
        self._uploads: dict[str, dict[str, Any]] = {}
        self._downloads: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def _cleanup(self) -> None:
        cutoff = time.time() - _MEDIA_TRANSFER_TTL_SECONDS
        for store in (self._uploads, self._downloads):
            for token, record in list(store.items()):
                if record["created"] < cutoff:
                    store.pop(token, None)

    def create_upload(self, filename: str, *, gallery=True) -> dict[str, Any]:
        safe_name = Path(str(filename or "").strip()).name
        if safe_name in {"", ".", ".."} or ":" in safe_name or "\\" in safe_name:
            raise ValueError("filename must be a non-empty file name.")
        if gallery and not _mcp_media_type(safe_name):
            raise ValueError("filename must have a supported image, video, audio, or 3D extension")
        token = uuid.uuid4().hex
        with self._lock:
            self._cleanup()
            self._uploads[token] = {"created": time.time(), "filename": safe_name, "gallery": gallery}
        route = "gallery" if gallery else "files"
        return {"upload_url": f"/wangp_api/{route}/upload/{token}", "method": "PUT", "filename": safe_name, "expires_in_seconds": _MEDIA_TRANSFER_TTL_SECONDS, "max_bytes": _MAX_UPLOAD_BYTES}

    def consume_upload(self, token: str) -> dict[str, Any] | None:
        with self._lock:
            self._cleanup()
            return self._uploads.pop(str(token or ""), None)

    def create_download(self, media_id: str) -> dict[str, Any]:
        record = _gallery_item(self.session, media_id)
        path = _resolve_existing_path(record["path"], "Gallery media")
        token = uuid.uuid4().hex
        with self._lock:
            self._cleanup()
            self._downloads[token] = {"created": time.time(), "media_id": record["media_id"]}
        return {"media_id": record["media_id"], "download_url": f"/wangp_api/gallery/download/{token}", "filename": Path(path).name, "media_type": record["media_type"], "size": Path(path).stat().st_size, "expires_in_seconds": _MEDIA_TRANSFER_TTL_SECONDS}

    def consume_download(self, token: str) -> dict[str, Any] | None:
        with self._lock:
            self._cleanup()
            transfer = self._downloads.pop(str(token or ""), None)
        if transfer is None:
            return None
        if "path" in transfer:
            path, _gallery = _resolve_io_source(self.session, self.policy, transfer["source"], file=True)
            return {**transfer, "path": path}
        return _gallery_item(self.session, transfer["media_id"])

    def upload_path(self, filename: str, *, gallery=True) -> Path:
        if not gallery:
            path = self.policy.require_write(self.policy.workspace / "uploads" / uuid.uuid4().hex / filename)
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
        root = self.policy.output_roots[0] if getattr(self.session, '_workspace_id', None) else Path(self.session._output_dir) if self.session._output_dir is not None else self.session._root / "outputs"
        upload_root = (root / "mcp_uploads" / uuid.uuid4().hex).resolve()
        upload_root.mkdir(parents=True, exist_ok=True)
        return upload_root / Path(filename).name

    def create_file_download(self, source: str) -> dict[str, Any]:
        path, _gallery = _resolve_io_source(self.session, self.policy, source, file=True)
        token = uuid.uuid4().hex
        with self._lock:
            self._cleanup()
            self._downloads[token] = {"created": time.time(), "path": path, "source": source}
        size = Path(path).stat().st_size
        result = {"status": "done", "path": self.policy.virtualize_path(path), "filename": Path(path).name, "size_bytes": size, "mime_type": mimetypes.guess_type(path)[0] or "application/octet-stream", "expires_in_seconds": _MEDIA_TRANSFER_TTL_SECONDS}
        # The ordinary resource is bounded; large files remain readable in exact byte ranges over stdio.
        result.update(resource_uri=f"wangp://files/{token}/0/{min(size, 8 * 1024**2)}", resource_chunk_bytes=8 * 1024**2)
        if size > 8 * 1024**2:
            result["continuation"] = f"Read wangp://files/{token}/<offset>/<length> until size_bytes; length is at most 8388608 bytes. Offsets and lengths count decoded bytes."
        if self.http:
            result.update(download_url=f"/wangp_api/files/download/{token}", method="GET")
        return result

    def read_file_range(self, token: str, offset: int, length: int) -> bytes:
        if offset < 0 or not 0 <= length <= 8 * 1024**2:
            raise ValueError("offset must be non-negative and length must be 0..8388608 bytes.")
        with self._lock:
            self._cleanup()
            transfer = self._downloads.get(token)
        if transfer is None or "path" not in transfer:
            raise ValueError("File download expired or not found; request another download.")
        path, _gallery = _resolve_io_source(self.session, self.policy, transfer["source"], file=True)
        path = Path(path)
        with path.open("rb") as reader:
            if offset > path.stat().st_size:
                raise ValueError("offset is beyond the end of the file.")
            reader.seek(offset)
            return reader.read(length)

    def save_companions(self, artifact, identity):
        files = []
        for filename, content in artifact.side_files.items():
            path = self.policy.require_write(self.policy.workspace / "artifacts" / identity / Path(filename).name)
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                with path.open("xb") as writer:
                    writer.write(content)
            files.append({"filename": path.name, "path": self.policy.virtualize_path(path), "size_bytes": path.stat().st_size, "mime_type": mimetypes.guess_type(path.name)[0] or "application/octet-stream"})
        return files


def _mcp_postprocessing_processes(media_type: str, allow_read_file_system: bool = False, processes: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    from postprocessing import catalog as postprocessing_catalog

    processes = postprocessing_catalog.call_processes(postprocessing_catalog.query_processes(media_type) if processes is None else processes)
    for process in processes:
        for parameter in process.get("parameters", ()):
            original_name = str(parameter.get("name", "") or "")
            replacement = _POSTPROCESS_PATH_PARAMETERS.get(original_name)
            if replacement is None:
                if parameter.get("media_type") == "image":
                    parameter["description"] = str(parameter.get("description", "") or "").rstrip() + (" Values may be Gallery media IDs or authorized existing image paths." if allow_read_file_system else " Values must be Gallery media IDs returned by wangp_list_gallery.")
                continue
            parameter["name"] = replacement
            parameter["description"] = "Media ID for the required audio file" + (" or an authorized existing server path." if allow_read_file_system else ". Direct filesystem paths are disabled for this server.")
    return processes


def _build_mcp_postprocessing_task(source_path: str, process: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    from postprocessing import catalog as postprocessing_catalog

    process_id = str(process["id"])
    process_type = str(process["type"])
    task = {
        "prompt": str(process["label"]),
        "resolution": "",
        "image_mode": 1 if _mcp_media_type(source_path) == "image" else 0,
        "repeat_generation": 1,
        "seed": int(parameters.get("seed", -1)),
        "temporal_upsampling": "",
        "spatial_upsampling": "",
        "film_grain_intensity": 0,
        "film_grain_saturation": 0.5,
        "postprocess_audio": "",
        "postprocess_audio_prompt": str(parameters.get("prompt", "")),
        "postprocess_audio_neg_prompt": str(parameters.get("negative_prompt", "")),
        "replace_voice_method": "",
        "replace_voice_sample": None,
        "replace_voice_sample2": None,
    }
    if process_type == postprocessing_catalog.PROCESS_TYPE_SPATIAL_UPSAMPLING:
        process_value = postprocessing_catalog.build_process_value(process, parameters)
        if process_value is None:
            raise ValueError("The spatial upsampling parameters could not be converted to a valid process value.")
        task.update({"mode": "edit_postprocessing", "video_source": source_path, "spatial_upsampling": process_value})
    elif process_type == postprocessing_catalog.PROCESS_TYPE_TEMPORAL_UPSAMPLING:
        process_value = postprocessing_catalog.build_process_value(process, parameters)
        if process_value is None:
            raise ValueError("The temporal upsampling parameters could not be converted to a valid process value.")
        task.update({"mode": "edit_postprocessing", "video_source": source_path, "temporal_upsampling": process_value})
    elif process_type in {postprocessing_catalog.PROCESS_TYPE_SOUNDTRACK, postprocessing_catalog.PROCESS_TYPE_VOICE_REPLACEMENT}:
        task.update({"mode": "edit_remux", "video_source": source_path, "postprocess_audio": process_id})
    elif process_type == postprocessing_catalog.PROCESS_TYPE_AUDIO_EDIT:
        task.update({"mode": "edit_audio", "audio_source": source_path, "postprocess_audio": process_id})
    parameter_defs = {str(parameter["name"]): parameter for parameter in process.get("parameters", ())}
    spatial_parameter_names = set()
    if process_type in (postprocessing_catalog.PROCESS_TYPE_SPATIAL_UPSAMPLING, postprocessing_catalog.PROCESS_TYPE_TEMPORAL_UPSAMPLING):
        spatial_parameters = {name: value for name, value in parameters.items() if name not in ("multiplier", "seed")}
        spatial_parameter_names.update(spatial_parameters)
        task.update(postprocessing_catalog.task_parameters(process, spatial_parameters))
    path_task_keys = {"audio_path": "audio_source", "voice_sample_path": "replace_voice_sample", "voice_sample2_path": "replace_voice_sample2"}
    for parameter_name, task_key in path_task_keys.items():
        if parameter_name in parameters:
            task[task_key] = _resolve_existing_path(parameters[parameter_name], parameter_name)
    standard_parameters = {"multiplier", "seed", "prompt", "negative_prompt", *path_task_keys, *spatial_parameter_names}
    for name, value in parameters.items():
        if name not in standard_parameters:
            task[str(parameter_defs[name].get("setting", name))] = value
    return task


def _build_default_toolbox(session, assistant_session=None, *, native_inspection=False, client_id_prefix="ai"):
    from shared.deepy.engine import AssistantSessionState, DeepyZeroTools

    runtime = session._ensure_runtime()
    if assistant_session is None:
        assistant_session = session._state.get("assistant_session")
    if assistant_session is None:
        assistant_session = AssistantSessionState()
        session._state["assistant_session"] = assistant_session
    toolbox = DeepyZeroTools(session._state["gen"], runtime.module.get_processed_queue, lambda *_args, **_kwargs: None, session=assistant_session, get_output_filepath=runtime.module.get_output_filepath, record_file_metadata=runtime.module.record_file_metadata, get_server_config=lambda: runtime.module.server_config)
    toolbox.client_id_prefix = client_id_prefix
    if native_inspection:  # external agents: WanGP's Visual Inspector answers, or the agent views images itself (view_media)
        from .media import bind_visual_inspector

        bind_visual_inspector(toolbox, runtime, session._state)
    return toolbox


def _external_file_policy(session, allow_read_file_system):
    from shared.deepy.filesystem import build_file_access_policy

    config = session._ensure_runtime().module.server_config
    paths = {key: str(session._output_dir or config.get(key) or config.get("save_path") or session._root / "outputs") for key in ("save_path", "image_save_path", "audio_save_path", "mesh_save_path")}
    paths = {key: str((session._root / value).resolve()) for key, value in paths.items()}
    return build_file_access_policy(paths, unrestricted_read=bool(allow_read_file_system))


def _toolbox_discovery(toolbox, allow_read_file_system: bool = False) -> list[dict[str, Any]]:
    actions = []
    for schema in toolbox.get_tool_schemas():
        function = copy.deepcopy(schema.get("function", {}))
        name = str(function.get("name", "") or "")
        if name not in _TOOLBOX_ACTIONS:
            continue
        for parameter_name in _TOOLBOX_MEDIA_PARAMETERS.get(name, ()):
            parameter = function.get("parameters", {}).get("properties", {}).get(parameter_name)
            if isinstance(parameter, dict):
                plural = parameter.get("type") == "array"
                maximum = int(parameter.get("maxItems", 0) or 0)
                plural_description = f"One to {maximum} media IDs returned by wangp_list_gallery" if maximum > 0 else "Media IDs returned by wangp_list_gallery"
                parameter["description"] = f"{plural_description if plural else 'A media ID returned by wangp_list_gallery'}" + (" or authorized existing server paths." if allow_read_file_system and plural else " or an authorized existing server path." if allow_read_file_system else ". Direct filesystem paths are disabled for this server.")
        if name == "inspect_media":
            nested_media_id = function.get("parameters", {}).get("properties", {}).get("media_inputs", {}).get("items", {}).get("properties", {}).get("media_id")
            if isinstance(nested_media_id, dict):
                nested_media_id["description"] = "A media ID returned by wangp_list_gallery or an authorized existing server path." if allow_read_file_system else "A media ID returned by wangp_list_gallery. Direct filesystem paths are disabled for this server."
        actions.append(function)
    return sorted(actions, key=lambda action: str(action.get("name", "")))


def _resolve_toolbox_arguments(session, toolbox, action: str, arguments: dict[str, Any], allow_read_file_system: bool = False, file_access_policy=None) -> dict[str, Any]:
    from shared.deepy import media_registry

    resolved = dict(arguments or {})

    def resolve_media_id(value: Any, parameter_name: str) -> str:
        if not isinstance(value, str):
            raise ValueError(f"{parameter_name} must be a string.")
        value = value.strip()
        if not value or media_registry.get_media_record(toolbox.session, value) is not None:
            return value
        media_path = _resolve_mcp_media_reference(session, value, parameter_name, allow_read_file_system, file_access_policy)
        record = media_registry.register_media(toolbox.session, media_path, source="gallery" if _is_media_id(value) else "filesystem")
        if record is None:
            raise ValueError(f"Unsupported media file: {media_path}")
        return record["media_id"]

    for parameter_name in _TOOLBOX_MEDIA_PARAMETERS.get(action, ()):
        raw_value = resolved.get(parameter_name, None)
        if raw_value is None:
            continue
        if parameter_name in {"media_ids", "audio_ids", "paths"} and not isinstance(raw_value, list):
            raise ValueError(f"{parameter_name} must be an array.")
        values = raw_value if isinstance(raw_value, list) else [raw_value]
        resolved_values = []
        for index, raw_item in enumerate(values):
            item_parameter_name = f"{parameter_name}[{index}]" if isinstance(raw_value, list) else parameter_name
            resolved_values.append(resolve_media_id(raw_item, item_parameter_name))
        resolved[parameter_name] = resolved_values if isinstance(raw_value, list) else resolved_values[0]
    if action == "inspect_media" and resolved.get("media_inputs") is not None:
        raw_inputs = resolved["media_inputs"]
        if not isinstance(raw_inputs, list):
            raise ValueError("media_inputs must be an array.")
        resolved_inputs = []
        for index, raw_input in enumerate(raw_inputs):
            if not isinstance(raw_input, dict):
                raise ValueError(f"media_inputs[{index}] must be an object.")
            resolved_input = dict(raw_input)
            resolved_input["media_id"] = resolve_media_id(resolved_input.get("media_id"), f"media_inputs[{index}].media_id")
            resolved_inputs.append(resolved_input)
        resolved["media_inputs"] = resolved_inputs
    if action == "draw_svg" and isinstance(resolved.get("svg"), str):  # each href resolved under this server's media and file rules
        from shared.deepy import long_text as deepy_long_text, svg_drawing

        resolved["svg"] = deepy_long_text.resolve_prompt_file(resolved["svg"], file_access_policy, read_only=True)  # @file("...") markup, read under this server's file rules
        resolved.update({key: resolve_media_id(resolved[key], key) for key in ("video_id", "like_video_id") if resolved.get(key) is not None})  # descriptions kept: they carry the overlay rules
        resolved["svg"] = svg_drawing.rewrite(resolved["svg"], {reference: resolve_media_id(reference, f"svg href {reference}") for reference in svg_drawing.references(resolved["svg"])})
    if action == "image_channels" and resolved.get("channel_sources") is not None:
        raw_sources = resolved["channel_sources"]
        if not isinstance(raw_sources, dict):
            raise ValueError("channel_sources must be an object.")
        resolved["channel_sources"] = {
            channel: resolve_media_id(value, f"channel_sources.{channel}")
            for channel, value in raw_sources.items()
        }
    return resolved


def _register_toolbox_result_media(session, result: Any) -> Any:
    if not isinstance(result, dict):
        return result
    output_paths = []
    if isinstance(result.get("output_file"), str) and result["output_file"].strip():
        output_paths.append(result["output_file"])
    if isinstance(result.get("output_files"), list):
        output_paths.extend(path for path in result["output_files"] if isinstance(path, str) and path.strip())
    if isinstance(result.get("paths"), list):
        output_paths.extend(path for path in result["paths"] if isinstance(path, str) and path.strip())
    records = [_register_gallery_media(session, path) for path in output_paths if Path(path).is_file() and _mcp_media_type(path)]
    gallery_items = [_output_record(_compact_gallery_record(record), record["path"]) for record in records]
    if gallery_items:
        result = dict(result)
        result["gallery_items"] = gallery_items
    return result


def _resolve_io_source(session, file_access_policy, value: Any, *, file: bool = False) -> tuple[str, bool]:
    source = str(value or "").strip()
    if not source:
        raise ValueError("source is empty.")
    if _is_media_id(source):
        path = Path(_gallery_item(session, source)["path"]).expanduser().resolve()
        if file and not path.is_file() or not file and not path.exists():
            raise FileNotFoundError(f"Gallery media does not exist: {path}")
        return str(path), True
    return str(file_access_policy.require_read(source, file=file)), False


def _run_io_action(session, file_access_policy, action: str, arguments: dict[str, Any], downloads_enabled: bool) -> dict[str, Any]:
    from shared.deepy import filesystem

    definition = filesystem.IO_ACTIONS[action]
    if action == "read_text":
        allowed = set(definition["parameters"]["properties"])
        unknown = sorted(set(arguments) - allowed)
        if unknown:
            raise ValueError(f"Unsupported read_text argument: {unknown[0]}. Use start_line and end_line for a bounded read.")
    for parameter in definition["parameters"].get("required", []):
        if parameter not in arguments or arguments[parameter] is None:
            raise ValueError(f"{parameter} is required.")
    if action == "list":
        return file_access_policy.virtualize_result(filesystem.list_entries(file_access_policy, path=arguments.get("path", ""), pattern=arguments.get("pattern", "*"), recursive=arguments.get("recursive", False), limit=arguments.get("limit", 200), offset=arguments.get("offset", 0), media_type=arguments.get("media_type", "all")))
    if action == "info":
        path, gallery = _resolve_io_source(session, file_access_policy, arguments["source"])
        result = filesystem.file_info(path)
        if gallery:
            result["media_id"] = _gallery_item(session, arguments["source"])["media_id"]
        return file_access_policy.virtualize_result(result)
    if action == "read_text":
        return file_access_policy.virtualize_result(filesystem.read_text(file_access_policy, arguments["path"], start_line=arguments.get("start_line", 1), end_line=arguments.get("end_line"), encoding=arguments.get("encoding", "utf-8-sig")))
    if action == "search_text":
        return file_access_policy.virtualize_result(filesystem.search_text(file_access_policy, arguments["path"], arguments["query"], pattern=arguments.get("pattern", "*"), recursive=arguments.get("recursive", False), regex=arguments.get("regex", False), case_sensitive=arguments.get("case_sensitive", False), limit=arguments.get("limit", 100)))
    if action == "write_text":
        return file_access_policy.virtualize_result(filesystem.write_text(file_access_policy, arguments["path"], arguments["text"], mode=arguments.get("mode", "create"), encoding=arguments.get("encoding", "utf-8")))
    if action == "mkdir":
        return file_access_policy.virtualize_result(filesystem.make_directory(file_access_policy, arguments["path"]))
    if action == "copy":
        source, gallery = _resolve_io_source(session, file_access_policy, arguments["source"], file=True)
        return file_access_policy.virtualize_result(filesystem.copy_file(file_access_policy, source, arguments["destination"], overwrite=arguments.get("overwrite", False), source_authorized=True))
    if action == "move":
        return file_access_policy.virtualize_result(filesystem.move_path(file_access_policy, arguments["source"], arguments["destination"]))
    if action == "delete":
        return file_access_policy.virtualize_result(filesystem.delete_path(file_access_policy, arguments["path"], recursive=arguments.get("recursive", False)))
    if action == "zip":
        raw_sources = arguments["sources"]
        if not isinstance(raw_sources, list) or not raw_sources:
            raise ValueError("sources must be a non-empty array.")
        sources = []
        for source in raw_sources:
            path, _gallery = _resolve_io_source(session, file_access_policy, source)
            sources.append(path)
        result = filesystem.zip_files(file_access_policy, sources, destination=arguments.get("destination", ""), authorized_sources=set(sources))
        if downloads_enabled:
            from shared.gradio.downloads import register_file_download
            result["download"] = register_file_download(result["output_file"], "application/zip")
        return file_access_policy.virtualize_result(result)
    if action == "unzip":
        source, _gallery = _resolve_io_source(session, file_access_policy, arguments["source"], file=True)
        return file_access_policy.virtualize_result(filesystem.unzip_file(file_access_policy, source, destination=arguments.get("destination", ""), overwrite=arguments.get("overwrite", False), source_authorized=True))
    if action == "download":
        if not downloads_enabled:
            raise RuntimeError("Direct WanGP downloads are unavailable for this MCP transport.")
        path, _gallery = _resolve_io_source(session, file_access_policy, arguments["source"], file=True)
        from shared.gradio.downloads import register_file_download
        return file_access_policy.virtualize_result({"status": "done", "path": path, "filename": Path(path).name, "size_bytes": Path(path).stat().st_size, "download": register_file_download(path), "error": ""})
    raise ValueError(f"Unknown IO action: {action}")


_UI_PROMPT_COUNTER = re.compile(r"^Prompt \d+/\d+(?:, | - |$)")


def _compact_job_snapshot(snapshot: dict[str, Any], *, keep_webui: bool = False, keep_paths: bool = False) -> dict[str, Any]:
    """Agent view of a job snapshot: no browser-queue handshake (unless kept for Prime's queue trigger), timestamps, empty events or false cancel
    flag; a finished job gets a top-level status and drops its last progress, and its generated_files when Gallery records name every file."""
    done = snapshot.get("done")  # recovered receipts are read back from disk
    hidden = {"created_at", "updated_at"} | (set() if keep_webui and not done else {"webui_submission_ready", "webui_load_queue_token"})
    compact = {key: value for key, value in snapshot.items() if key not in hidden and not (key in ("events", "cancel_requested") and not value)}
    progress = compact.get("progress")
    if done:
        compact.pop("progress", None)
        if isinstance(snapshot.get("created_at"), (int, float)) and isinstance(snapshot.get("updated_at"), (int, float)):
            compact["elapsed_seconds"] = round(snapshot["updated_at"] - snapshot["created_at"], 1)  # queue wait included: the time the job took to deliver
        result = compact.get("result")
        if isinstance(result, dict):
            compact.setdefault("status", "done" if result.get("success") else "interrupted" if result.get("cancelled") else "error")
            if not keep_paths and {Path(path).name for path in result.get("generated_files", [])} <= {item["filename"] if "filename" in item else Path(item["path"]).name for item in result.get("gallery_items", [])}:
                compact["result"] = {key: value for key, value in result.items() if key != "generated_files"}
    elif isinstance(progress, dict) and isinstance(progress.get("status"), str):
        compact["progress"] = {**progress, "status": _UI_PROMPT_COUNTER.sub("", progress["status"])}  # the counter numbers the whole UI queue, not this job's tasks
    return compact


class _JobRecord:
    def __init__(self, job_id: str, job: "SessionJob", session, file_store=None, state=None) -> None:
        self.job_id = job_id
        self.job = job
        self.session = session
        self.file_store = file_store
        self.state = state
        self.created_at = time.time()
        self.updated_at = self.created_at
        self.events: list[dict[str, Any]] = []
        self.progress: dict[str, Any] | None = None
        self.result: dict[str, Any] | None = None
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._watcher = threading.Thread(target=self._watch, daemon=True, name=f"wangp-mcp-job-{job_id}")

    def start(self) -> None:
        self._watcher.start()

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def wait(self, timeout: float | None = None) -> None:
        if not self._done.wait(timeout):
            raise TimeoutError("WanGP job result timed out")

    def _watch(self) -> None:
        for event in self.job.events.iter(timeout=0.2):
            event_dict = _event_to_dict(event)
            with self._lock:
                self.events.append(event_dict)
                if len(self.events) > _MAX_STORED_EVENTS:
                    del self.events[: len(self.events) - _MAX_STORED_EVENTS]
                self.updated_at = time.time()
                if event.kind == "progress":
                    self.progress = event_dict["data"]
        # Only the watcher builds the result. Media analysis must not hold the
        # snapshot lock or run on a polling request's thread.
        try:
            result = self._completed_result(self.job.result(timeout=0))
        except Exception as error:
            result = {"error": str(error)}
        with self._lock:
            self.result = result
            self.updated_at = time.time()
            self._done.set()
        if self.state is not None:
            saved = self.snapshot(event_limit=0)
            saved["status"] = _compact_job_snapshot(saved)["status"]
            self.state.save_job(self.job_id, saved)

    def _completed_result(self, completed) -> dict[str, Any]:
        result = _result_to_dict(completed, self.file_store, self.job_id)
        generated_paths = {(self.session._root / path).resolve() for path in result["generated_files"]}
        result["gallery_items"] = []
        if generated_paths:
            for record in _gallery_records(self.session, limit=500):
                if (self.session._root / record["path"]).resolve() in generated_paths:
                    path = None if self.file_store is None or self.file_store.policy is None else self.file_store.policy.virtualize_path((self.session._root / record["path"]).resolve())
                    item = _output_record(_compact_gallery_record(record), path)
                    if item["media_type"] == "audio":  # songs often stop well before their requested length: say where, once per job
                        from shared.deepy.video_tools import quiet_ranges
                        quiet = quiet_ranges(str((self.session._root / record["path"]).resolve()))
                        if quiet and item.get("duration_seconds", 0) - quiet[-1][1] < 0.1 and quiet[-1][1] - quiet[-1][0] >= 1:
                            item["sound_ends_seconds"] = quiet[-1][0]
                    elif item["media_type"] == "video":  # a generated shot can cut away to another shot or go black: measured, not seen
                        from shared.deepy.video_tools import picture_checks
                        item.update(picture_checks(str((self.session._root / record["path"]).resolve())))
                        if "cut_times_seconds" in item:
                            item["note"] = ("The model cut to other shots at cut_times_seconds. Look at a frame just after each cut: if it no longer shows the requested "
                                            "subject (a cutaway to objects, hands or other people), the shot failed there: use the part before the cut or regenerate it shorter.")
                    result["gallery_items"].append(item)
        if self.file_store is not None:
            policy = self.file_store.policy
            for artifact in result["artifacts"]:
                for companion in artifact.get("side_files", []):
                    path = policy.resolve_path(companion["path"])
                    for alias, root in policy.mounts:
                        if root in policy.output_roots and path.is_relative_to(root):
                            companion["path"] = f"@{alias}/{path.relative_to(root).as_posix()}"
                            break
        return result

    def snapshot(self, event_limit: int = 20) -> dict[str, Any]:
        event_limit = 20 if event_limit is None else event_limit
        event_limit = max(0, min(int(event_limit), _MAX_STORED_EVENTS))
        with self._lock:
            events = copy.deepcopy(self.events[-event_limit:] if event_limit else [])
            result = copy.deepcopy(self.result)
            progress = copy.deepcopy(self.progress)
            updated_at = self.updated_at
        return {
            "job_id": self.job_id,
            "workspace_id": getattr(self.session, '_workspace_id', None),
            "agent_context": getattr(self.session, '_agent_context', None),
            "done": result is not None,
            "cancel_requested": self.job.cancel_requested,
            "webui_submission_ready": self.job.webui_submission_ready,
            "webui_load_queue_token": self.job.webui_load_queue_token if self.job.webui_submission_ready else "",
            "created_at": self.created_at,
            "updated_at": updated_at,
            "events": events,
            "progress": progress,
            "result": result,
        }


class _ToolJob:
    """A toolbox action running in the background, followed with get_job / wait_jobs like a generation job."""

    def __init__(self, action: str, run, cancel: threading.Event | None = None, progress: dict[str, Any] | None = None) -> None:
        self.job_id, self.action = uuid.uuid4().hex, action
        self.created_at = self.updated_at = time.time()
        self.result: dict[str, Any] | None = None
        self.error: str | None = None
        self._cancel, self._progress = cancel, progress  # a plugin action stops on cancel and reports its progress there
        self._done = threading.Event()
        threading.Thread(target=self._run, args=(run,), daemon=True, name=f"wangp-mcp-tool-{self.job_id}").start()

    @property
    def job(self):  # polled like a generation's SessionJob
        return self

    @property
    def done(self) -> bool:
        return self._done.is_set()

    def _run(self, run) -> None:
        try:
            self.result = run()
        except Exception as error:
            self.error = str(error)
        finally:
            self.updated_at = time.time()
            self._done.set()

    @property
    def cancel_requested(self) -> bool:
        return self._cancel is not None and self._cancel.is_set()

    def cancel(self) -> None:
        if self._cancel is None:
            raise ValueError(f"The {self.action} toolbox job cannot be cancelled; it runs to completion.")
        self._cancel.set()

    def snapshot(self, event_limit: int = 0) -> dict[str, Any]:
        done = self.done
        snapshot = {"job_id": self.job_id, "action": self.action, "done": done, "status": ("error" if self.error is not None else "done") if done else "running", "created_at": self.created_at, "updated_at": self.updated_at}
        if done:
            snapshot["result"] = {"error": self.error} if self.error is not None else self.result
        else:
            snapshot["progress"] = {"status": f"Running {self.action}", **(self._progress or {}), "elapsed_seconds": round(time.time() - self.created_at, 1)}
            if self.cancel_requested:
                snapshot["cancel_requested"] = True
        return snapshot


class _JobStore:
    def __init__(self, session) -> None:
        self._session = session
        self.file_store = None
        self._jobs: dict[str, _JobRecord] = {}
        self._lock = threading.Lock()
        self.state = None
        self._owner = uuid.uuid4().hex
        self._submission = threading.local()

    def submit(self, source: dict[str, Any] | list[dict[str, Any]]) -> _JobRecord:
        receipt = getattr(self._submission, "receipt", None)
        job_id = receipt["job_id"] if receipt else uuid.uuid4().hex
        if receipt:
            from .preflight import task_settings
            source = copy.deepcopy(source)
            task_ids = []
            for index, settings in enumerate(task_settings(source)):
                settings.setdefault("client_id", f"mcp_{job_id}_{index + 1}")
                task_ids.append(settings["client_id"])
            receipt.update(status="submitting", resolved_settings=source, client_ids=task_ids)
            self.state.save_job(job_id, receipt)
        job = self._session.submit(source)
        record = _JobRecord(job_id, job, self._session, self.file_store, self.state if receipt else None)
        with self._lock:
            self._jobs[record.job_id] = record
        record.start()
        return record

    def start_tool(self, action: str, run, *, cancel: threading.Event | None = None, progress: dict[str, Any] | None = None) -> _ToolJob:
        record = _ToolJob(action, run, cancel, progress)
        with self._lock:
            self._jobs[record.job_id] = record
        return record

    def get(self, job_id: str) -> _JobRecord:
        with self._lock:
            record = self._jobs.get(str(job_id or "").strip())
        if record is None:
            if self.state is not None:
                from .state import RecoveredJob
                saved = self.state.get_job(job_id)
                if saved is not None:
                    return RecoveredJob(saved, self._session, self._owner)
            raise KeyError(f"Unknown WanGP job_id: {job_id}. Jobs are not kept across a WanGP restart: the outputs of finished jobs are in the Gallery (wangp_list_gallery); submit unfinished ones again.")
        return record

    def once(self, request_id, payload, submit):
        if request_id is None:
            return submit()
        if not isinstance(request_id, str) or not request_id.strip() or len(request_id) > 200:
            raise ValueError("request_id must contain 1..200 characters.")
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        receipt = {"job_id": uuid.uuid4().hex, "request_id": request_id, "owner": self._owner, "status": "validating", "done": False, "result": None, "created_at": time.time()}
        receipt.update(workspace_id=getattr(self._session, '_workspace_id', None), agent_context=getattr(self._session, '_agent_context', None))
        with self.state.transaction() as db:
            row = db.execute("SELECT fingerprint, job_id FROM requests WHERE request_id=?", (request_id,)).fetchone()
            if row is None:
                db.execute("INSERT INTO requests VALUES (?, ?, ?, ?)", (request_id, fingerprint, receipt["job_id"], json.dumps(receipt)))
        if row is not None:
            if row[0] != fingerprint:
                raise ValueError("request_id was already used with a different payload. Nothing was submitted.")
            return dict(self.get(row[1]).snapshot(event_limit=0), request_id=request_id, reused=True)
        self._submission.receipt = receipt
        try:
            result = dict(submit(), request_id=request_id, reused=False)
            # The watcher may already have persisted completion. Never overwrite it with an earlier snapshot.
            with self.state.transaction() as db:
                saved = json.loads(db.execute("SELECT snapshot FROM requests WHERE job_id=?", (receipt["job_id"],)).fetchone()[0])
                if not saved.get("done"):
                    db.execute("UPDATE requests SET snapshot=? WHERE job_id=?", (json.dumps({**saved, **result}), receipt["job_id"]))
            return result
        except Exception:
            if receipt["status"] == "validating":
                with self.state.transaction() as db:
                    db.execute("DELETE FROM requests WHERE request_id=?", (request_id,))
            raise
        finally:
            del self._submission.receipt

    def existing(self, request_id, payload):
        """Check a post-processing retry before resolving its possibly removed input."""
        if request_id is None:
            return None
        with self.state.transaction() as db:
            row = db.execute("SELECT fingerprint, job_id FROM requests WHERE request_id=?", (request_id,)).fetchone()
        if row is None:
            return None
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if row[0] != fingerprint:
            raise ValueError("request_id was already used with a different payload. Nothing was submitted.")
        return dict(self.get(row[1]).snapshot(event_limit=0), request_id=request_id, reused=True)


def _config_file_from_arg(value: str | None) -> str | None:
    if value is None:
        return None
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        return str(path / "wgp_config.json")
    return str(path)


def build_server_for_session(session, settings: dict[str, Any] | None = None, toolbox=None, default_job_event_limit: int = 20, allow_read_file_system: bool = False, http_media_transfer: bool = False, compact_model_tools: bool = False, file_access_policy=None, io_downloads: bool = False, api_version: str | int = "2.1", allow_async: bool = False, defer_wait: bool = False, python_sandbox=None):
    api_version = str(api_version)
    if api_version not in {"1", "2", "2.1"}:
        raise ValueError("MCP API version must be 1, 2, or 2.1.")
    try:
        from mcp.server.fastmcp import FastMCP
    except Exception as exc:
        raise RuntimeError("WanGP MCP server requires the 'mcp' Python package. Install project requirements or run `pip install mcp`.") from exc

    jobs = _JobStore(session)
    agent_instructions = None
    plugin_instructions = ""
    if api_version == "2.1":
        from .plugin_toolbox import instructions as plugin_toolbox_instructions
        from .skills import SkillMCP

        plugin_instructions = plugin_toolbox_instructions()  # routing declared by the enabled plugins, for Prime and external agents alike
        if not compact_model_tools:  # Prime already carries these rules in its system prompt
            from shared.deepy import load_agent_instructions
            agent_instructions = "\n\n".join(part for part in (_AGENT_INSTRUCTIONS_HEADER, load_agent_instructions(), plugin_instructions) if part)
        mcp = SkillMCP("WanGP", instructions=agent_instructions, skills_root=session._root / "user_skills", **dict(settings or {}))
    else:
        mcp = FastMCP("WanGP", **dict(settings or {}))
    default_job_event_limit = max(0, min(int(default_job_event_limit), _MAX_STORED_EVENTS))
    if file_access_policy is None:
        from shared.deepy.filesystem import build_file_access_policy
        file_access_policy = build_file_access_policy({"save_path": str(session._root / "outputs")} if api_version == "2.1" else {}, unrestricted_read=bool(allow_read_file_system))
    if api_version == "2.1":
        from shared.deepy.prime_filesystem import PrimeFileAccessPolicy, build_prime_policy

        if not isinstance(file_access_policy, PrimeFileAccessPolicy):
            file_access_policy = build_prime_policy(file_access_policy, file_access_policy.output_roots[0] / "mcp_workspaces" / uuid.uuid4().hex)
    from shared.deepy import long_text as deepy_long_text

    long_text_active = deepy_long_text.long_text_tools_active(file_access_policy) and deepy_long_text.workspace_mount(file_access_policy) is not None
    _register_documentation_resources(mcp, file_access_policy, long_text_active=long_text_active, api_version=api_version)
    allow_read_file_system = file_access_policy.read_enabled
    if api_version == "2.1":
        from .state import MCPState
        jobs.state = MCPState(session._root, getattr(session, '_agent_context', None) or (str(file_access_policy.workspace) if compact_model_tools else "external"))
    transfer_store = _MediaTransferStore(session, file_access_policy, http=http_media_transfer) if http_media_transfer or api_version == "2.1" else None
    if api_version == "2.1":
        from .transfers import ResumableTransfers
        transfer_store = ResumableTransfers(session, file_access_policy, jobs.state, http=http_media_transfer)
    mcp._wangp_file_store = transfer_store if api_version == "2.1" else None
    mcp._wangp_file_policy = file_access_policy
    mcp._wangp_api_version = api_version
    jobs.file_store = mcp._wangp_file_store
    toolbox_instance = toolbox
    if api_version == "2.1" and toolbox_instance is not None:
        toolbox_instance.file_access_policy_override = file_access_policy
    implementations = {}

    def api_tool():
        def register(function):
            implementations[function.__name__] = function
            return mcp.tool()(function) if api_version == "1" else function
        return register

    if http_media_transfer:
        @mcp.custom_route("/wangp_api/gallery/upload/{token}", methods=["PUT"], include_in_schema=False)
        async def upload_gallery_media(request):
            if api_version == "2.1":
                store = mcp._wangp_agents.transfer_store(request.path_params['token']) if hasattr(mcp, '_wangp_agents') else transfer_store
                return await store.http_upload(request)
            from starlette.responses import JSONResponse

            transfer = transfer_store.consume_upload(request.path_params["token"])
            if transfer is None:
                return JSONResponse({"error": "Upload expired or not found."}, status_code=404)
            content_length = request.headers.get("content-length")
            if content_length is not None:
                try:
                    declared_length = int(content_length)
                except ValueError:
                    return JSONResponse({"error": "Invalid Content-Length header."}, status_code=400)
                if declared_length > _MAX_UPLOAD_BYTES:
                    return JSONResponse({"error": f"Upload exceeds the {_MAX_UPLOAD_BYTES}-byte limit."}, status_code=413)
            target = transfer_store.upload_path(transfer["filename"], gallery=transfer["gallery"])
            size = 0
            try:
                with target.open("xb") as writer:
                    async for chunk in request.stream():
                        size += len(chunk)
                        if size > _MAX_UPLOAD_BYTES:
                            raise ValueError(f"Upload exceeds the {_MAX_UPLOAD_BYTES}-byte limit.")
                        writer.write(chunk)
                record = _compact_gallery_record(_register_gallery_media(session, str(target))) if transfer["gallery"] else {"path": file_access_policy.virtualize_path(target), "filename": target.name, "mime_type": mimetypes.guess_type(target.name)[0] or "application/octet-stream"}
            except Exception as exc:
                target.unlink(missing_ok=True)
                try:
                    target.parent.rmdir()
                except OSError:
                    pass
                return JSONResponse({"error": str(exc)}, status_code=413 if size > _MAX_UPLOAD_BYTES else 400)
            return JSONResponse({"status": "uploaded", "size": size, **record})

        @mcp.custom_route("/wangp_api/gallery/download/{token}", methods=["GET"], include_in_schema=False)
        async def download_gallery_media(request):
            if api_version == "2.1":
                store = mcp._wangp_agents.transfer_store(request.path_params['token']) if hasattr(mcp, '_wangp_agents') else transfer_store
                return await store.http_download(request)
            from starlette.responses import FileResponse, JSONResponse

            record = transfer_store.consume_download(request.path_params["token"])
            if record is None:
                return JSONResponse({"error": "Download expired or not found."}, status_code=404)
            path = _resolve_existing_path(record["path"], "Gallery media")
            return FileResponse(path, filename=Path(path).name, media_type=mimetypes.guess_type(path)[0] or "application/octet-stream")

        if api_version == "2.1":
            mcp.custom_route("/wangp_api/files/upload/{token}", methods=["PUT", "GET"], include_in_schema=False)(upload_gallery_media)
            mcp.custom_route("/wangp_api/files/download/{token}", methods=["GET"], include_in_schema=False)(download_gallery_media)

    if api_version == "2.1":
        @mcp.resource("wangp://files/{token}/{offset}/{length}", name="file_bytes", description="Read exact byte ranges from an authorized download. Use the URI from create_file_download; length is at most 8388608 decoded bytes.", mime_type="application/octet-stream")
        def read_file_bytes(token: str, offset: int, length: int) -> bytes:
            store = mcp._wangp_agents.transfer_store(token) if hasattr(mcp, '_wangp_agents') else transfer_store
            return store.read_file_range(token, offset, length)

        @api_tool()
        def wangp_create_file_download(source: str, include_sha256: bool = False) -> dict[str, Any]:
            """Download any authorized file or Gallery ID. HTTP servers return a URL; all transports support the returned binary MCP resource, with byte ranges for large files."""
            return transfer_store.create_file_download(source, include_sha256=include_sha256)

        @api_tool()
        def wangp_create_file_upload(filename: str, size_bytes: int | None = None, sha256: str | None = None) -> dict[str, Any]:
            """Create a resumable upload to the workspace. Supply size_bytes and optional SHA-256; use upload_chunk/get_upload/finish_upload on every transport, or HTTP PUT with Upload-Offset. Tokens survive restarts for 24 hours."""
            return transfer_store.create_upload(filename, gallery=False, size_bytes=size_bytes, sha256=sha256)

        @api_tool()
        def wangp_upload_chunk(token: str, offset: int, data_base64: str) -> dict[str, Any]:
            """Commit up to 8 MiB of decoded bytes at offset. Identical chunk retries are safe; an overlap with different bytes is rejected."""
            return transfer_store.upload_chunk(token, offset, data_base64)

        @api_tool()
        def wangp_get_upload(token: str) -> dict[str, Any]:
            """Read the durable upload offset after reconnect/restart, or retrieve the completed file path."""
            return transfer_store.get_upload(token)

        @api_tool()
        def wangp_finish_upload(token: str) -> dict[str, Any]:
            """Verify declared size and optional SHA-256, then publish the upload by atomic rename. Safe to repeat."""
            return transfer_store.finish_upload(token)

    def get_toolbox():
        nonlocal toolbox_instance
        if toolbox_instance is None:
            toolbox_instance = _build_default_toolbox(session, native_inspection=api_version == "2.1" and not compact_model_tools, client_id_prefix="ai" if compact_model_tools else "ai_mcp")
            if api_version == "2.1":
                toolbox_instance.file_access_policy_override = file_access_policy
        return toolbox_instance

    if agent_instructions:
        @mcp.resource("wangp://guides/agent", name="agent", title="WanGP agent instructions", description="Tool-calling conventions, model choice concepts and rules, settings layering, prompt separators, media input roles, Gallery and workspace use; also sent as the server instructions.", mime_type="text/markdown")
        def agent_guide() -> str:
            return agent_instructions

    @mcp.prompt(name="wangp_agent", title="WanGP Agent Guide", description="Instructions for discovering WanGP models, building settings, running jobs, and handling media" + (": the server instructions, for clients that do not show them." if agent_instructions else "."))
    def wangp_agent_prompt() -> str:
        if agent_instructions:
            return agent_instructions
        if api_version != "1":
            guide = f"WanGP MCP API {api_version}. Use toolbox descriptions to choose a domain. Use declared shortcuts and known call recipes directly; otherwise discover actions, read the selected contract, then execute. Reuse known contracts. Specialized workflows are available in wangp://guides/workflows."
            return f"{guide}\n\n{plugin_instructions}" if plugin_instructions else guide
        return _AGENT_GUIDE_PATH.read_text(encoding="utf-8")

    def legacy_model_tool(function):
        return function if compact_model_tools or api_version != "1" else mcp.tool()(function)

    @api_tool()
    def wangp_models(query: str = "", filters: dict[str, Any] | None = None, limit: int = 10, offset: int = 0) -> dict[str, Any]:
        """Search compact models; filters accepts family, base_model_type, finetune, model_type, main_output, inputs, or name."""

        filters = dict(filters or {})
        allowed = {"family", "base_model_type", "finetune", "model_type", "main_output", "inputs", "name"}
        unknown = sorted(set(filters) - allowed)
        if unknown:
            raise ValueError(f"Unknown model filter: {unknown[0]}")
        filters["query"] = str(query or "").strip() or None
        matches = session.list_model_metadata(**filters)
        offset, limit = max(0, int(offset)), max(1, min(int(limit), 50))
        page = [_compact_deepy_model_metadata(record) for record in matches[offset:offset + limit]]
        return {"models": page, "total": len(matches), "returned": len(page), "offset": offset, "has_more": offset + len(page) < len(matches)}

    @api_tool()
    def wangp_model(model_type: str, view: Literal["schema", "definition", "defaults"] = "schema", property: str | None = None) -> dict[str, Any]:
        """Return one model's compact schema, definition, or generation defaults. Compact servers preview root strings longer than 256 characters; repeat view='definition' with the property named in the suffix to retrieve its full value."""

        if property is not None and view != "definition":
            raise ValueError("property is supported only with view='definition'")
        if view == "definition":
            result = _mcp_model_definition(session.get_model_def(model_type), property_name=property, string_limit=_DEEPY_MODEL_DEF_STRING_LIMIT if compact_model_tools else None)
        elif view == "defaults":
            result = _deepy_task_settings(session, session.get_exported_default_settings(model_type), model_type)
        else:
            result = _compact_deepy_model_schema(session.get_model_schema(model_type))
        if result is None:
            raise KeyError(f"Unknown model_type: {model_type}")
        return result

    @legacy_model_tool
    def wangp_list_models(family: str | None = None, base_model_type: str | None = None, finetune: str | None = None, model_type: str | None = None, main_output: str | None = None, inputs: str | None = None, name: str | None = None, query: str | None = None, limit: int = 10, offset: int = 0, include_availability: bool = False) -> list[dict[str, Any]]:
        """List at most 10 compact model records. All string filters accept case-insensitive * and ? globs; query searches names, ids, families, and descriptions. For generation without a user-specified model, use the corresponding default template instead of browsing."""

        return _compact_model_metadata(session.list_model_metadata(family=family, base_model_type=base_model_type, finetune=finetune, model_type=model_type, main_output=main_output, inputs=inputs, name=name, query=query, limit=max(1, min(int(limit), 10)), offset=max(0, int(offset)), include_availability=include_availability))

    @legacy_model_tool
    def wangp_search_models(query: str, family: str | None = None, base_model_type: str | None = None, finetune: str | None = None, main_output: str | None = None, inputs: str | None = None, limit: int = 10, offset: int = 0, include_availability: bool = False) -> dict[str, Any]:
        """Search models by user-facing name, model id, architecture, family, or description and return a bounded page with totals. Other string filters accept * and ? globs."""

        if not str(query or "").strip():
            raise ValueError("query is required")
        matches = session.list_model_metadata(family=family, base_model_type=base_model_type, finetune=finetune, main_output=main_output, inputs=inputs, query=query, include_availability=include_availability)
        offset = max(0, int(offset))
        limit = max(1, min(int(limit), 50))
        page = _compact_model_metadata(matches[offset:offset + limit])
        return {"matches": page, "total_matches": len(matches), "returned": len(page), "offset": offset, "has_more": offset + len(page) < len(matches)}

    @legacy_model_tool
    def wangp_list_model_defs(family: str | None = None, base_model_type: str | None = None, finetune: str | None = None, model_type: str | None = None, main_output: str | None = None, inputs: str | None = None, name: str | None = None, query: str | None = None, limit: int = 10, offset: int = 0) -> list[dict[str, Any]]:
        """List a bounded page of full WanGP model definitions. String filters accept * and ? globs. Prefer search, then fetch one selected model."""

        return session.list_model_defs(family=family, base_model_type=base_model_type, finetune=finetune, model_type=model_type, main_output=main_output, inputs=inputs, name=name, query=query, limit=max(1, min(int(limit), 20)), offset=max(0, int(offset)))

    @legacy_model_tool
    def wangp_get_model(model_type: str) -> dict[str, Any] | None:
        """Return one WanGP model definition with parameter declarations and capabilities. Embedded default values are omitted. Do not call this after a template query unless a required parameter remains absent from both template settings and the compact schema."""

        return _mcp_model_definition(session.get_model_def(model_type))

    @legacy_model_tool
    def wangp_get_model_metadata(model_type: str, include_availability: bool = False) -> dict[str, Any] | None:
        """Return one compact model metadata record."""

        return session.get_model_metadata(model_type, include_availability=include_availability)

    @legacy_model_tool
    def wangp_get_model_availability(model_type: str) -> dict[str, Any]:
        """Return local file availability for one model using the same status as the UI model selector."""

        return session.get_model_availability(model_type)

    @legacy_model_tool
    def wangp_list_model_availability(family: str | None = None, base_model_type: str | None = None, finetune: str | None = None, model_type: str | None = None, main_output: str | None = None, inputs: str | None = None, name: str | None = None, query: str | None = None, limit: int = 20, offset: int = 0) -> list[dict[str, Any]]:
        """List a bounded page of local model-file availability records using the model discovery filters."""

        return session.list_model_availability(family=family, base_model_type=base_model_type, finetune=finetune, model_type=model_type, main_output=main_output, inputs=inputs, name=name, query=query, limit=max(1, min(int(limit), 50)), offset=max(0, int(offset)))

    @legacy_model_tool
    def wangp_get_default_settings(model_type: str) -> dict[str, Any]:
        """Return pristine model defaults generated from WanGP and the model handler, filtered to relevant fields and without fixed metadata such as type or settings version. User-saved UI defaults are not included. Do not call this after a template query because template settings already include these model defaults."""

        return _deepy_task_settings(session, session.get_exported_default_settings(model_type), model_type)

    @api_tool()
    def wangp_model_settings(model_type: str, setting_id: str | None = None) -> dict[str, Any]:
        """List saved settings, accelerator profiles and presets for a model, or return one by id."""

        result = session.get_model_settings(model_type, setting_id, include_selection=True) if api_version != "1" else session.get_model_settings(model_type, setting_id)
        if setting_id is not None and isinstance(result.get("content"), dict):
            result["content"] = _strip_deepy_fixed_image_mode(session, result["content"], model_type)
        return result

    @api_tool()
    def wangp_list_loras(model_type: str, name: str | None = None) -> dict[str, Any]:
        """Recursively list locally available LoRAs for a model. Optional name accepts a case-insensitive * and ? glob over each subfolder-relative identifier. Returned identifiers can be copied directly into activated_loras and paired with loras_multipliers."""

        return session.list_loras(model_type, name=name)

    @legacy_model_tool
    def wangp_get_model_schema(model_type: str) -> dict[str, Any] | None:
        """Return a compact capability and usage summary for one model. After a template query, call this only to verify a required capability or limit that the template does not establish. Use wangp_get_default_settings only for raw non-template generation requests."""

        return session.get_model_schema(model_type)

    @api_tool()
    def wangp_list_gallery(media_type: str = "all", limit: int = 50, selected_only: bool = False) -> list[dict[str, Any]]:
        """List compact summaries of current and remembered session Gallery media, including actual video resolution, frame count, FPS, and duration from WanGP's cached media probe. Set selected_only to return only live UI selections. Use media_id with wangp_get_media_settings only when full generation settings are needed."""

        records = _gallery_records(session, media_type=media_type, limit=500 if selected_only else limit)
        if selected_only:
            records = [record for record in records if record["selected"]]
        return [_compact_gallery_record(record) for record in records[-max(1, min(int(limit), 500)):]]

    @api_tool()
    def wangp_get_media_settings(media_id: str | None = None, path: str | None = None) -> dict[str, Any]:
        """Return generation settings for one media file. Provide exactly one input: media_id for Gallery media, or path for a server file when filesystem reads are enabled."""

        return _media_settings(session, media_id=media_id, path=path, allow_read_file_system=allow_read_file_system, file_access_policy=file_access_policy)

    if long_text_active:
        @api_tool()
        def rg(arguments: str) -> dict[str, Any]:
            """Search authorized UTF-8 files with ripgrep. Pass supported rg options and one pattern, then `--` and optional @alias paths; omitted paths search the temporary workspace."""

            return deepy_long_text.run_rg(file_access_policy, arguments)

        @api_tool()
        def edit(file_path: str, old_string: str, new_string: str, replace_all: bool = False) -> dict[str, Any]:
            """Replace exact text in one authorized UTF-8 file. old_string must be unique unless replace_all is true; whitespace and line endings are literal."""

            return deepy_long_text.edit_text(file_access_policy, file_path, old_string, new_string, replace_all)

        @api_tool()
        def append_text(file_path: str, text: str) -> dict[str, Any]:
            """Append exact literal UTF-8 text to an authorized file, creating the file when it does not exist. No newline or prefix is added implicitly."""

            return deepy_long_text.append_text(file_access_policy, file_path, text)

    @api_tool()
    def wangp_io(action: str | None = None, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Discover or run filesystem utilities. Use @alias/path; plain paths use video outputs. Omit action for actions; pass action alone for its schema."""

        from shared.deepy.filesystem import available_io_actions

        action_name = str(action or "").strip()
        actions = available_io_actions(file_access_policy, downloads_enabled=io_downloads)
        if not action_name:
            compact = [{"name": candidate["name"], "description": candidate["description"]} for candidate in actions]
            roots = [root["path"] for root in file_access_policy.roots()] if file_access_policy.read_enabled else []
            return {"status": "discovery", "actions": compact, "count": len(compact), **({"roots": roots} if roots else {}), **({"read_everywhere": True} if file_access_policy.read_everywhere else {}), "next": "Call again with one action and omit arguments for its schema."}
        action_defs = [candidate for candidate in actions if candidate["name"] == action_name]
        if not action_defs:
            raise ValueError(f"IO action '{action_name}' is unavailable. Call without action to list allowed actions.")
        if arguments is None:
            return {"status": "schema", "action": action_defs[0]}
        return _run_io_action(session, file_access_policy, action_name, copy.deepcopy(arguments), io_downloads)

    @api_tool()
    def wangp_notify(message: str, title: str = "Deepy notification") -> dict[str, Any]:
        """Send a message through WanGP's configured notification destinations."""

        from shared.notifications import send_notification

        message = str(message or "").strip()
        if not message:
            raise ValueError("message is required")
        return send_notification(session._ensure_runtime().module.server_config, str(title or "Deepy notification").strip(), message)

    if http_media_transfer:
        @api_tool()
        def wangp_create_gallery_upload(filename: str) -> dict[str, Any]:
            """Create a short-lived HTTP PUT URL for uploading image, video, or audio media. A successful upload registers and selects the media in the appropriate WanGP Gallery."""

            return transfer_store.create_upload(filename)

        @api_tool()
        def wangp_create_gallery_download(media_id: str) -> dict[str, Any]:
            """Create a short-lived HTTP GET URL for downloading one Gallery item. Only registered Gallery media can be downloaded."""

            return transfer_store.create_download(media_id)

    @api_tool()
    def wangp_list_deepy_templates(tool_id: str | None = None) -> list[dict[str, Any]]:
        """List settings templates available in Deepy's Template Settings section. The current default for every tool is marked explicitly."""

        return _deepy_template_catalog(session, tool_id=tool_id)

    @api_tool()
    def wangp_get_deepy_template_settings(tool_id: str, template: str) -> dict[str, Any]:
        """Return merged, migrated, model-filtered, non-conflicting settings for a Deepy template. For any generation task or derived subtask without a user-specified model, call this with template='default' before model discovery and use the result directly. If general_properties is returned, apply those values after settings."""

        return _deepy_template_settings(session, tool_id, template)

    def prepare_postprocess(media_id=None, path=None, process=None, parameters=None):
        from postprocessing import catalog as postprocessing_catalog

        source_path = _resolve_mcp_media_input(session, media_id, path, allow_read_file_system, file_access_policy)
        media_type = _mcp_media_type(source_path)
        if not media_type:
            raise ValueError(f"Unsupported media file extension: {Path(source_path).suffix}")
        if media_type == "3d":
            from shared.utils.mesh_postprocessing import is_mesh_file
            if not is_mesh_file(source_path):
                raise ValueError("Gaussian splats can not be post-processed: only meshes can be refined or textured.")
        processes = postprocessing_catalog.query_processes(media_type)
        discovered_processes = _mcp_postprocessing_processes(media_type, allow_read_file_system, processes)
        process_id = str(process or "").strip()
        if not process_id:
            return None, {"status": "discovery", "path": source_path, "media_type": media_type, "processes": discovered_processes, "count": len(discovered_processes)}
        matches = [candidate for candidate in processes if candidate["id"] == process_id]
        if len(matches) != 1:
            raise ValueError(f"Post-processing process '{process_id}' is {'not available' if not matches else 'ambiguous'} for {media_type}.")
        process_def = matches[0]
        normalized_parameters, error = postprocessing_catalog.normalize_parameters(process_def, parameters)
        if error:
            raise ValueError(error)
        for parameter_name in set(_POSTPROCESS_PATH_PARAMETERS.values()) & normalized_parameters.keys():
            normalized_parameters[parameter_name] = _resolve_mcp_media_reference(session, normalized_parameters[parameter_name], parameter_name, allow_read_file_system, file_access_policy)
        for parameter_def in process_def.get("parameters", ()):
            parameter_name = str(parameter_def.get("name", "") or "")
            if parameter_def.get("media_type") == "image" and parameter_name in normalized_parameters:
                normalized_parameters[parameter_name] = _resolve_mcp_media_value(session, normalized_parameters[parameter_name], parameter_name, allow_read_file_system, file_access_policy)
        task = _build_mcp_postprocessing_task(source_path, process_def, normalized_parameters)
        return task, {"source_path": source_path, "media_type": media_type, "process": process_id, "process_label": process_def["label"], "parameters": normalized_parameters}

    @api_tool()
    def wangp_postprocess(media_id: str | None = None, path: str | None = None, process: str | None = None, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
        """Discover or run compatible post-processing. Provide exactly one input: media_id for Gallery media, or path for a server file when filesystem reads are enabled. Omit process for discovery."""

        task, details = prepare_postprocess(media_id, path, process, parameters)
        if task is None:
            return details
        snapshot = jobs.submit(task).snapshot(event_limit=20)
        snapshot.update(details)
        return snapshot

    @api_tool()
    def wangp_toolbox(action: str | None = None, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """Discover or run WanGP media utilities. Omit action for a compact action list; pass action without arguments for its schema; then pass arguments to execute. Media arguments accept media IDs returned by wangp_list_gallery; direct server paths require startup permission."""

        sandbox_toolbox = get_toolbox()
        action_name = str(action or "").strip()
        actions = _toolbox_discovery(sandbox_toolbox, allow_read_file_system)
        if api_version != "2.1":
            actions = [action for action in actions if action["name"] != "inspect_audio"]
        if not action_name:
            compact_actions = [{"name": candidate["name"], "description": candidate.get("description", "")} for candidate in actions]
            return {"status": "discovery", "actions": compact_actions, "count": len(compact_actions), "next": "Call again with one action and omit arguments to get its exact schema."}
        if action_name not in _TOOLBOX_ACTIONS:
            raise ValueError(f"Unknown toolbox action '{action_name}'. Call without action to discover available actions.")
        action_defs = [candidate for candidate in actions if candidate["name"] == action_name]
        if not action_defs:
            raise ValueError(f"Toolbox action '{action_name}' is unavailable in this WanGP runtime.")
        if arguments is None:
            return {"status": "schema", "action": action_defs[0]}
        resolved_arguments = _resolve_toolbox_arguments(session, sandbox_toolbox, action_name, copy.deepcopy(arguments), allow_read_file_system, file_access_policy)
        validation_error = sandbox_toolbox.validate_tool_call(action_name, resolved_arguments)
        if validation_error:
            raise ValueError(validation_error)
        return _register_toolbox_result_media(session, sandbox_toolbox.call(action_name, resolved_arguments))

    def prepare_generation(source):
        if not isinstance(source, (dict, list)):
            raise TypeError("source must be a settings dict, task dict, manifest dict, or task list")
        _gallery_records(session, limit=500)
        resolved_source = copy.deepcopy(source)
        if api_version != "1":
            source = resolved_source
            manifest = isinstance(source, dict) and "tasks" in source
            tasks = source["tasks"] if manifest else source if isinstance(source, list) else [source]
            root = "settings.tasks" if manifest else "settings"
            if not isinstance(tasks, list) or not tasks:
                raise ValueError(f"{root} must be a non-empty task list. Nothing was submitted.")
            for index, task in enumerate(tasks):
                path = f"{root}[{index}]" if manifest or isinstance(source, list) else root
                if not isinstance(task, dict):
                    raise ValueError(f"{path} must be a settings or task object. Nothing was submitted.")
                settings = task
                for key in ("params", "settings"):
                    if key in task:
                        settings, path = task[key], f"{path}.{key}"
                        break
                if not isinstance(settings, dict):
                    raise ValueError(f"{path} must be a settings object. Nothing was submitted.")
                defaults_call = f' See wangp_model(model_type="{settings.get("model_type") or "<model_type>"}", action="defaults", arguments={{}}) for setting names and value formats.'
                # nested values: the model's custom_settings and the post-processing slots' parameter dictionaries
                nested = [key for key, value in settings.items() if isinstance(value, dict) and key not in NESTED_SETTINGS or key == "resolution" and not isinstance(value, str)]
                if nested:
                    expected = f'must be a "WIDTHxHEIGHT" string such as "1280x720", not {json.dumps(settings["resolution"])}' if nested[0] == "resolution" else "must be a single value; only custom_settings and *_postprocessing_parameters hold objects"
                    raise ValueError(f"{path}.{nested[0]} {expected}.{defaults_call} Nothing was submitted.")
                if str(settings.get("mode", "") or "").startswith("edit_"):
                    continue
                model_key = "base_model_type" if "model_type" not in settings and "base_model_type" in settings else "model_type"
                model_type = settings.get(model_key)
                if not isinstance(model_type, str) or not model_type.strip():
                    template_hint = "" if api_version == "1" else ', or the settings must name a template ("template": "<task>")'
                    raise ValueError(f"{path}.{model_key} must be a non-empty string for generation{template_hint}. Nothing was submitted.")
                if session.get_model_metadata(model_type.strip()) is None:  # a retired model type becomes its new type (settings converted in place)
                    model_type = session._ensure_runtime().module.convert_retired_model_type(settings)
                if session.get_model_metadata(model_type.strip()) is None:
                    template_call = 'wangp_templates(task="...", template="default")' if api_version == "2.1" else 'wangp_deepy_templates(tool_id="...")'
                    known = [str(item.get("model_type") or "") for item in session.list_model_defs()]
                    close = [name for name in known if name.startswith(model_type.strip())][:3] or difflib.get_close_matches(model_type.strip(), known, n=3, cutoff=0.5)
                    raise ValueError(f'{path}.{model_key}: unknown model {model_type.strip()!r}.' + (f" Closest model types: {', '.join(close)}." if close else "") + f' Use {template_call} for the configured default model, or wangp_models(query="...") to find one. Nothing was submitted.')
                unknown_error = settings_error(settings)
                if unknown_error:
                    if "accelerator profile" in unknown_error:
                        defaults_call = f' Profiles: wangp_model(model_type="{model_type.strip()}", action="saved_settings", arguments={{}}).'
                    raise ValueError(f"{path}: {unknown_error}{defaults_call} Nothing was submitted.")
                _validate_generation_media(session, settings, model_type.strip(), path)
                custom_settings = settings.get("custom_settings")
                if custom_settings:
                    declared = [setting["id"] for setting in session._ensure_runtime().module.get_model_custom_settings(session.get_model_def(model_type.strip()))]
                    unknown = [str(key) for key in custom_settings if key not in declared] if isinstance(custom_settings, dict) else [""]
                    if unknown:
                        allowed = ", ".join(declared[:5]) + (", ..." if len(declared) > 5 else "") if declared else "none"
                        raise ValueError(f"{path}.custom_settings.{unknown[0]} is not a custom setting of {model_type.strip()} (declared: {allowed}); prompt, resolution and other settings go directly in the settings. Nothing was submitted.")
                if not str(settings.get("prompt") or "").strip():
                    model_def = session.get_model_def(model_type.strip())
                    if not model_def.get("prompt_optional"):
                        meaning = model_def.get("prompt_description")
                        raise ValueError(f"{path}.prompt is empty; give the complete prompt{f' (for this model: {meaning})' if meaning else ''}, otherwise WanGP uses the model's default prompt. Nothing was submitted.")
                    settings["prompt"] = ""  # it names the files only: empty rather than the model's default prompt
                properties = _deepy_general_properties(session)
                outputs = (session.get_model_metadata(model_type.strip()) or {}).get("main_output", [])
                defaults = {"seed": properties["seed"], "multi_prompts_gen_type": _deepy_prompt_mode(session, model_type.strip())}
                if ("image" in outputs or "video" in outputs) and "world" not in outputs:  # a world's size and length are its model's own
                    defaults["resolution"] = f"{properties['width']}x{properties['height']}"
                    if "video" in outputs and settings.get("image_mode", 0) not in (1, 2):
                        defaults["video_length"] = properties["num_frames"]
                elif "audio" in outputs:
                    defaults["duration_seconds"] = properties["audio_duration"]
                for key, value in defaults.items():
                    if settings.get(key) is None:
                        settings[key] = value
                if api_version == "2.1":
                    from .preflight import validate_declared_settings
                    validate_declared_settings(session, settings, lambda value, key: _resolve_mcp_media_value(session, value, key, allow_read_file_system, file_access_policy))
                # Agents write final prompts: WanGP's automatic enhancer mode must not rewrite them through a template's or model default's enhancer
                if compact_model_tools or api_version != "1" and "prompt_enhancer" not in settings and session.get_default_settings(model_type.strip()).get("prompt_enhancer"):
                    settings["prompt_enhancer"] = ""
        if api_version != "1" or long_text_active:
            resolved_source = deepy_long_text.resolve_prompt_references(resolved_source, file_access_policy, read_only=api_version != "1")
        return _resolve_generation_media(session, resolved_source, allow_read_file_system, file_access_policy)

    @api_tool()
    def wangp_generate(source: dict[str, Any] | list[dict[str, Any]], wait: bool = False, timeout_s: float | None = None, event_limit: int | None = None) -> dict[str, Any]:
        """Start a WanGP generation from a settings dict, task dict, or task list."""

        record = jobs.submit(prepare_generation(source))
        if wait:
            record.wait(timeout=timeout_s)
        return record.snapshot(event_limit=default_job_event_limit if event_limit is None else event_limit)

    @api_tool()
    def wangp_get_job(job_id: str, event_limit: int | None = None) -> dict[str, Any]:
        """Poll a WanGP generation job."""

        return jobs.get(job_id).snapshot(event_limit=default_job_event_limit if event_limit is None else event_limit)

    @api_tool()
    def wangp_cancel_job(job_id: str, event_limit: int | None = None) -> dict[str, Any]:
        """Request cancellation of a WanGP generation job."""

        record = jobs.get(job_id)
        if not isinstance(record, _JobRecord):
            record.cancel()
        else:
            record.job.cancel()
        return record.snapshot(event_limit=default_job_event_limit if event_limit is None else event_limit)

    if api_version == "2.1":
        from .preflight import preflight

        def wangp_validate(settings=None, media=None, process=None, parameters=None, detail="summary"):
            return preflight(session, prepare_generation, prepare_postprocess, settings, media, process, parameters, detail)

        implementations["wangp_validate"] = wangp_validate

    if api_version != "1":
        from .v2 import register_v2
        if api_version == "2.1" and python_sandbox is None and hasattr(toolbox, "publish_sandbox_media"):
            python_sandbox = _build_python_sandbox(session, toolbox, file_access_policy, lambda name, arguments: call_tool_inline(mcp, name, arguments))
        mcp._wangp_python_sandbox = python_sandbox
        mcp._wangp_web_search = _build_web_search(session, file_access_policy) if api_version == "2.1" else None
        register_v2(mcp, session, implementations, jobs, file_access_policy, get_toolbox, downloads=io_downloads, allow_async=allow_async, defer_wait=defer_wait, deepy_help=compact_model_tools, extended=api_version == "2.1", python_sandbox=python_sandbox, web_search=mcp._wangp_web_search)

    from shared.utils.plugins import get_deepy_prime_plugin_tools

    for definition in get_deepy_prime_plugin_tools():
        plugin_read_enabled = file_access_policy.mode in {"read", "read_write"} if api_version != "1" else allow_read_file_system
        if definition.requires_file_system and not plugin_read_enabled:
            continue
        if mcp._tool_manager.get_tool(definition.name) is not None:
            raise RuntimeError(f"Deepy Prime plugin '{definition.plugin_id}' cannot replace built-in MCP tool '{definition.name}'.")
        mcp.tool(name=definition.name, title=definition.display_name, description=definition.description)(definition.function)

    return mcp


def build_inprocess_server(session, toolbox=None, default_job_event_limit: int = 20, allow_read_file_system: bool = False, file_access_policy=None, api_version: str | int = "2.1", allow_async: bool = False, python_sandbox=None):
    return build_server_for_session(session, toolbox=toolbox, default_job_event_limit=default_job_event_limit, allow_read_file_system=allow_read_file_system, compact_model_tools=True, file_access_policy=file_access_policy, io_downloads=True, api_version=api_version, allow_async=allow_async, defer_wait=True, python_sandbox=python_sandbox)


def _build_web_search(session, file_access_policy):
    """The session's web toolbox (wangp_web_search) when the configuration enables it: downloads and cookies live in the web folder of the agent's documents workspace (without a workspace, search and open only); LoRA search and installation use the session's model catalog and LoRA folders."""
    from shared.deepy.config import get_civitai_api_key, get_huggingface_token
    from shared.deepy.web_search import WebSearch, enabled

    if not enabled():
        return None
    hub = {"catalog": session.list_model_metadata, "lora_dir": session.get_lora_dir, "civitai_key": get_civitai_api_key, "hf_token": get_huggingface_token} if hasattr(session, "get_lora_dir") else {}
    if file_access_policy.workspace is None:
        return WebSearch(**hub)
    return WebSearch(file_access_policy.workspace / "web", cookie_file=file_access_policy.workspace.parent / "web_cookies.json",  # beside the workspace, out of every readable folder
                     resolve_path=lambda reference: str(file_access_policy.require_read(reference, file=True)), virtualize=file_access_policy.virtualize_path, **hub)


def call_tool_inline(mcp, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Run one of this server's tools on the current thread with its argument validation (for sandbox scripts: the MCP request loop is busy running the script that asks)."""
    tool = mcp._tool_manager.get_tool(name)
    if tool is None:
        raise ValueError(f"Unknown tool: {name}")
    try:
        validated = tool.fn_metadata.arg_model.model_validate(arguments)
        result = tool.fn(**validated.model_dump_one_level())
    except Exception as error:
        return {"status": "error", "tool": name, "error": str(error)}
    if not isinstance(result, dict):
        return {"status": "done", "content": result}
    result.setdefault("status", "done")
    return result


def _build_python_sandbox(session, toolbox, file_access_policy, call_tool=None):
    """The session's Python sandbox (toolbox wangp_python) when the configuration enables it: its folder lies in the agent's documents workspace; call_tool(name, arguments) serves scripts' WanGP tool calls."""
    from shared.deepy import media_registry
    from shared.deepy.config import DEEPY_PYTHON_SANDBOX_DEFAULT, DEEPY_PYTHON_SANDBOX_KEY, get_deepy_config_value, normalize_deepy_python_sandbox, python_apps_level
    from shared.deepy.python_sandbox import PythonSandbox

    if normalize_deepy_python_sandbox(get_deepy_config_value(DEEPY_PYTHON_SANDBOX_KEY, DEEPY_PYTHON_SANDBOX_DEFAULT)) != "enabled" or file_access_policy.workspace is None:
        return None

    def resolve_path(reference):
        record = media_registry.get_media_record(toolbox.session, reference)
        if record is not None:
            return _resolve_existing_path(record["path"], "ref")
        return _resolve_mcp_media_reference(session, reference, "ref", file_access_policy.read_enabled, file_access_policy)

    def compact(record):
        return None if record is None else _output_record(_compact_gallery_record(record), record["path"])

    apps = None
    if python_apps_level() != "off":
        from shared.apps.dev import AppWorkshop
        apps = AppWorkshop(file_access_policy.workspace / "sandbox", resolve_path=resolve_path, publish_app=lambda package, description: compact(toolbox.publish_sandbox_app(package, description)), virtualize=file_access_policy.virtualize_path)
    return PythonSandbox(file_access_policy.workspace / "sandbox", resolve_path=resolve_path, publish_media=lambda path, description: compact(toolbox.publish_sandbox_media(path, description)),
                         publish_frames=lambda frames, fps, name, description: compact(toolbox.publish_sandbox_frames(frames, fps, name, description)), virtualize=file_access_policy.virtualize_path,
                         call_tool=call_tool, list_dir=lambda reference: file_access_policy.require_read(reference, directory=True), apps=apps)


def _http_settings(transport: str, host: str | None, port: int | None, oauth) -> dict[str, Any]:
    from shared.authentication.oauth import transport_security

    settings: dict[str, Any] = {key: value for key, value in (("host", host), ("port", port)) if value is not None}
    if transport == "streamable-http":
        settings.update(json_response=True, stateless_http=True)
    if oauth is not None:
        settings["transport_security"] = transport_security(oauth.issuer)
    return settings


def _in_thread(function):
    import anyio
    from mcp.server.lowlevel.server import request_ctx

    async def run(**kwargs):
        dispatched = time.perf_counter()
        context = request_ctx.get(None)
        request = context.request if context is not None else None
        # Stateful MCP sessions dispatch from a long-lived task. Use the actual
        # message's HTTP request, not that task's inherited initialization context.
        request_id, timing = request.scope.get("wangp_mcp_timing", ("-", None)) if request is not None else (_REQUEST_ID.get(), _REQUEST_TIMING.get())

        def invoke():
            started = time.perf_counter()
            try:
                return function(**kwargs)
            finally:
                finished = time.perf_counter()
                if timing is not None:
                    timing.update(queue=(started - dispatched) * 1000, handler=(finished - started) * 1000)
                level = logging.WARNING if finished - dispatched >= 1 else logging.DEBUG
                _TIMING_LOG.log(level, "request=%s tool=%s action=%s dispatched_at=%.3f queue_ms=%.1f handler_ms=%.1f", request_id, function.__name__, kwargs.get("action", ""), time.time() - (finished - dispatched), (started - dispatched) * 1000, (finished - started) * 1000)

        return await anyio.to_thread.run_sync(invoke)
    return run


def build_webui_server(service, args):
    """MCP v2 for the web UI process: agent jobs join the shared queue, gallery and progress shown on every page."""
    import uvicorn
    from shared.api import WanGPSession
    from shared.authentication.oauth import mcp_authorization
    from shared.authentication.tls import tls_options
    from shared.deepy.engine import AssistantSessionState

    transport = _normalize_transport(args.mcp_transport or "streamable-http")
    if transport == "stdio":
        raise ValueError("The Web UI serves MCP over HTTP: use --mcp-transport streamable-http or sse.")
    oauth = mcp_authorization(args, transport)
    config = service._deps.get_server_config()
    port = server_port(args.mcp_port if args.mcp_port is not None else config.get('mcp_port', DEFAULT_PORT))
    cert, key, _ = tls_options(args, port)  # --https-port only redirects the web UI port
    session = WanGPSession(webui_state=service._state, console_output=False, console_isatty=False, webui_append=True)
    session._gradio_webui_context = {"load_queue": service.load_inline_queue}
    # The toolbox gets its own media registry: Deepy's chat session is not shared with external agents.
    server = build_server_for_session(session, _http_settings(transport, args.mcp_host.strip() or None, port, oauth), toolbox=_build_default_toolbox(session, AssistantSessionState(), native_inspection=str(args.mcp_api_version) == "2.1", client_id_prefix="ai_mcp"), allow_read_file_system=args.mcp_allow_read_file_system, file_access_policy=_external_file_policy(session, args.mcp_allow_read_file_system) if str(args.mcp_api_version) == "2.1" else None, http_media_transfer=True, api_version=args.mcp_api_version, allow_async=True)
    if str(args.mcp_api_version) == '2.1':
        from .workspaces import AgentWorkspaces
        AgentWorkspaces(server, session, config, host=service, allow_read=args.mcp_allow_read_file_system, http=True).install()
    for tool in server._tool_manager.list_tools():  # an agent waiting for its job must not block the others
        if not tool.is_async:
            tool.fn, tool.is_async = _in_thread(tool.fn), True
    app = server.sse_app() if transport == "sse" else server.streamable_http_app()
    app = app if oauth is None else oauth.wrap(app)

    async def refresh_pages(scope, receive, send):  # gallery changes made by agents reach every open page
        if scope["type"] != "http":
            return await app(scope, receive, send)
        request_id = uuid.uuid4().hex[:12]
        token = _REQUEST_ID.set(request_id)
        timing = {}
        timing_token = _REQUEST_TIMING.set(timing)
        scope = {**scope, "wangp_mcp_timing": (request_id, timing)}
        started = time.perf_counter()
        arrived_at = time.time()
        first_response = None

        async def timed_send(message):
            nonlocal first_response
            if message["type"] == "http.response.start":
                first_response = time.perf_counter()
                metrics = {**timing, "mcp": (first_response - started) * 1000}
                message = {**message, "headers": [*message.get("headers", []), (b"x-wangp-request-id", request_id.encode("ascii")), (b"server-timing", ", ".join(f"{name};dur={duration:.1f}" for name, duration in metrics.items()).encode("ascii"))]}
            await send(message)

        try:
            await app(scope, receive, timed_send)
        finally:
            service.host_changed()
            elapsed = time.perf_counter() - started
            first_ms = None if first_response is None else (first_response - started) * 1000
            level = logging.WARNING if scope["method"] == "POST" and elapsed >= 1 else logging.DEBUG
            _TIMING_LOG.log(level, "request=%s method=%s arrived_at=%.3f response_start_ms=%s total_ms=%.1f", request_id, scope["method"], arrived_at, first_ms, elapsed * 1000)
            _REQUEST_ID.reset(token)
            _REQUEST_TIMING.reset(timing_token)

    host, port = server.settings.host, server.settings.port
    print(f"WanGP MCP server: {'https' if cert else 'http'}://{host}:{port}{server.settings.sse_path if transport == 'sse' else server.settings.streamable_http_path}")
    return uvicorn.Server(uvicorn.Config(refresh_pages, host=host, port=port, ssl_certfile=cert, ssl_keyfile=key, log_level="warning"))


def build_server(args: argparse.Namespace):
    from shared.api import init
    from shared.authentication import forwarded_options
    from shared.authentication.oauth import mcp_authorization
    from shared.authentication.tls import tls_options

    args.transport = _normalize_transport(getattr(args, "transport", "stdio"))
    forwarded_options(args)
    if args.auth or args.auth_password is not None:
        raise ValueError("MCP uses separate OAuth authentication. Use --mcp-auth and --mcp-auth-password.")
    args.oauth = mcp_authorization(args, args.transport)
    console_output = bool(args.console_output) and args.transport != "stdio"
    with _stdio_safe_startup_output(args.transport):
        session = init(
            root=args.root,
            config_path=_config_file_from_arg(args.config),
            output_dir=args.output_dir,
            cli_args=tuple(args.cli_arg or ()),
            console_output=console_output,
            console_isatty=False,
        )
    if args.transport != 'stdio':
        config = session._ensure_runtime().module.server_config
        args.port = server_port(args.port if args.port is not None else config.get('mcp_port', DEFAULT_PORT))
    args.tls = tls_options(args, args.port or DEFAULT_PORT)
    server = build_server_for_session(session, _http_settings(args.transport, args.host, args.port, args.oauth), default_job_event_limit=getattr(args, "job_event_limit", 20), allow_read_file_system=getattr(args, "allow_read_file_system", False), file_access_policy=_external_file_policy(session, args.allow_read_file_system) if str(args.mcp_api_version) == "2.1" else None, http_media_transfer=args.transport != "stdio", api_version=args.mcp_api_version, allow_async=args.mcp_async)
    if str(args.mcp_api_version) == '2.1':
        from .workspaces import AgentWorkspaces
        config = dict(session._ensure_runtime().module.server_config)
        override = session._ensure_runtime().module.args.workspaces_dir
        if override:
            config['workspaces_dir'] = override
        AgentWorkspaces(server, session, config, allow_read=args.allow_read_file_system, http=args.transport != 'stdio').install()
    return server


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WanGP as an MCP server.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2]), help="WanGP repository root.")
    parser.add_argument("--config", default=None, help="Path to wgp_config.json or a directory containing it.")
    parser.add_argument("--output-dir", default=None, help="Directory for generated media.")
    parser.add_argument("--cli-arg", action="append", default=[], help="Extra argument passed to wgp.py during runtime initialization. Repeat for multiple args.")
    parser.add_argument("--console-output", action="store_true", help="Mirror WanGP stdout/stderr to the MCP server console.")
    parser.add_argument("--transport", default="stdio", help="MCP transport: stdio, sse, or streamable-http.")
    parser.add_argument("--mcp-api-version", choices=("1", "2", "2.1"), default="2.1", help="WanGP API version; latest (2.1) by default. Pin 1 or 2 for earlier contracts.")
    parser.add_argument("--mcp-async", action="store_true", help="Enable optional asynchronous generation/post-processing in API v2/v2.1. Disabled by default; v1 keeps its historical wait behavior.")
    parser.add_argument("--host", default=None, help="Optional host for non-stdio transports.")
    parser.add_argument("--port", type=server_port, default=None, help="Override the Agents configuration MCP server port for HTTP transports (default: 8000).")
    parser.add_argument("--job-event-limit", type=int, default=20, help="Default number of recent progress events included in job snapshots; use 0 for terminal state/results only.")
    parser.add_argument("--allow-read-file-system", action="store_true", help="Allow MCP tool arguments to reference arbitrary server filesystem paths. Disabled by default; media IDs returned by wangp_list_gallery remain available.")
    from shared.authentication import add_arguments
    add_arguments(parser)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        server = build_server(args)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    return run_server(server, args)


def run_server(server, args: argparse.Namespace) -> int:
    transport = _normalize_transport(getattr(args, "transport", "stdio"))
    if transport == "stdio":
        server.run()
    else:
        import uvicorn
        from shared.authentication.tls import HTTPSRedirect, run_http_and_https
        app = server.sse_app() if transport == "sse" else server.streamable_http_app()
        if args.oauth is not None:
            app = args.oauth.wrap(app)
        cert, key, https_port = args.tls
        host, port = server.settings.host, server.settings.port
        if https_port is None:
            uvicorn.run(app, host=host, port=port, ssl_certfile=cert, ssl_keyfile=key)
        else:
            app = HTTPSRedirect(app, https_port)
            run_http_and_https(uvicorn.Config(app, host=host, port=port, lifespan="off", timeout_graceful_shutdown=5), uvicorn.Config(app, host=host, port=https_port, ssl_certfile=cert, ssl_keyfile=key, timeout_graceful_shutdown=5))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
