# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

from shared.utils.settings_bundle import is_wangp_settings_filename


PLUGIN_DIR = Path(__file__).resolve().parent
APP_ROOT_DIR = PLUGIN_DIR.parent.parent
APP_SETTINGS_DIR = APP_ROOT_DIR / "settings"
PROCESS_SETTINGS_DIR = PLUGIN_DIR / "settings"
MEDIAFLOW_SETTINGS_FILE = APP_SETTINGS_DIR / "mediaflow_settings.json"
LEGACY_PROCESS_FULL_VIDEO_SETTINGS_FILE = APP_SETTINGS_DIR / "process_full_video_settings.json"
PROCESS_FULL_VIDEO_SETTINGS_FILE = MEDIAFLOW_SETTINGS_FILE
MEDIAFLOW_SETTINGS_VERSION = 2
METADATA_VERSION_KEY = "metadata_version"
MEDIA_KIND_STORAGE_KEY = "media_kind"
BATCH_MODE_STORAGE_KEY = "batch_mode"
LAST_SETTINGS_STORAGE_KEY = "last_settings"
USER_SETTINGS_STORAGE_KEY = "user_settings"
PRESERVE_ARTIFACTS_STORAGE_KEY = "preserve_media_generator_artifacts"
PRESERVE_ARTIFACTS_DEFAULT = 10
PRESERVE_ARTIFACTS_MIN = 1
PRESERVE_ARTIFACTS_MAX = 20
PREVIEW_OUTPUT_STORAGE_KEY = "preview_media_generator_output"
PREVIEW_OUTPUT_DEFAULT = True
DEFAULT_MEDIA_KIND = "video"
DEFAULT_BATCH_MODE = "single"
VALID_MEDIA_KINDS = ("video", "image")
VALID_BATCH_MODES = ("single", "batch")
LAUNCH_DEFAULT_PROCESS_NAME = "Outpaint Video - LTX 2.3 Distilled 1.1"
IMAGE_LAUNCH_DEFAULT_PROCESS_NAME = "FlashVSR One Pass Image Upscaling"
USER_PROCESS_VALUE_PREFIX = "__user_settings__:"
MOVED_PROCESS_SETTINGS_PATHS = {
    "settings/Detailer - LTX 2 Distilled.json": "settings/video/Detailer - LTX 2 Distilled.json",
    "settings/Detailer - LTX 2.3 Distilled 1.0.json": "settings/video/Detailer - LTX 2.3 Distilled 1.0.json",
    "settings/Detailer - LTX 2.3 Distilled 1.1.json": "settings/video/Detailer - LTX 2.3 Distilled 1.1.json",
    "settings/FlashVSR One Pass Image Upscaling.json": "settings/image/FlashVSR One Pass Image Upscaling.json",
    "settings/FlashVSR One Pass Upscale.json": "settings/video/FlashVSR One Pass Upscale.json",
    "settings/FlashVSR Two Pass Image Upscaling.json": "settings/image/FlashVSR Two Pass Image Upscaling.json",
    "settings/FlashVSR Two Pass Upscale.json": "settings/video/FlashVSR Two Pass Upscale.json",
    "settings/Flux PiD Image Upscaling.json": "settings/image/Flux PiD Image Upscaling.json",
    "settings/Flux2 PiD Image Upscaling.json": "settings/image/Flux2 PiD Image Upscaling.json",
    "settings/HDR (Convert SDR to HDR) - LTX 2.3 Distilled 1.0.json": "settings/video/HDR (Convert SDR to HDR) - LTX 2.3 Distilled 1.0.json",
    "settings/HDR (Convert SDR to HDR) - LTX 2.3 Distilled 1.1.json": "settings/video/HDR (Convert SDR to HDR) - LTX 2.3 Distilled 1.1.json",
    "settings/Lanczos Upscale.json": "settings/video/Lanczos Upscale.json",
    "settings/Outpaint Video - LTX 2.3 Distilled 1.0.json": "settings/video/Outpaint Video - LTX 2.3 Distilled 1.0.json",
    "settings/Outpaint Video - LTX 2.3 Distilled 1.1.json": "settings/video/Outpaint Video - LTX 2.3 Distilled 1.1.json",
    "settings/Refocus (Remove Blur) - LTX 2.3 Distilled 1.0.json": "settings/video/Refocus (Remove Blur) - LTX 2.3 Distilled 1.0.json",
    "settings/Refocus (Remove Blur) - LTX 2.3 Distilled 1.1.json": "settings/video/Refocus (Remove Blur) - LTX 2.3 Distilled 1.1.json",
    "settings/Uncompress (Remove MPEG Compression Artifacts) - LTX 2.3 Distilled 1.0.json": "settings/video/Uncompress (Remove MPEG Compression Artifacts) - LTX 2.3 Distilled 1.0.json",
    "settings/Uncompress (Remove MPEG Compression Artifacts) - LTX 2.3 Distilled 1.1.json": "settings/video/Uncompress (Remove MPEG Compression Artifacts) - LTX 2.3 Distilled 1.1.json",
    "settings/Ungrade (Remove Stylized Color Grading) - LTX 2.3 Distilled 1.0.json": "settings/video/Ungrade (Remove Stylized Color Grading) - LTX 2.3 Distilled 1.0.json",
    "settings/Ungrade (Remove Stylized Color Grading) - LTX 2.3 Distilled 1.1.json": "settings/video/Ungrade (Remove Stylized Color Grading) - LTX 2.3 Distilled 1.1.json",
}


def normalize_media_kind(value, default: str = DEFAULT_MEDIA_KIND) -> str:
    text = str(value or "").strip().lower()
    if text in VALID_MEDIA_KINDS:
        return text
    return default if default in VALID_MEDIA_KINDS or default == "" else DEFAULT_MEDIA_KIND


def normalize_batch_mode(value, default: str = DEFAULT_BATCH_MODE) -> str:
    text = str(value or "").strip().lower()
    if text in VALID_BATCH_MODES:
        return text
    return default if default in VALID_BATCH_MODES or default == "" else DEFAULT_BATCH_MODE


def normalize_preserve_artifact_count(value) -> int:
    try:
        count = int(value)
    except (TypeError, ValueError):
        count = PRESERVE_ARTIFACTS_DEFAULT
    return min(PRESERVE_ARTIFACTS_MAX, max(PRESERVE_ARTIFACTS_MIN, count))


def normalize_preview_enabled(value) -> bool:
    if value is None:
        return PREVIEW_OUTPUT_DEFAULT
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"0", "false", "no", "off"}:
            return False
        if text in {"1", "true", "yes", "on"}:
            return True
    return bool(value)


def last_settings_key(media_kind: str, batch_mode: str) -> str:
    return f"{normalize_media_kind(media_kind)}/{normalize_batch_mode(batch_mode)}"


def process_settings_media_kind(settings: dict | None) -> str:
    if not isinstance(settings, dict):
        return DEFAULT_MEDIA_KIND
    try:
        return "image" if int(settings.get("image_mode") or 0) == 1 else "video"
    except (TypeError, ValueError):
        return DEFAULT_MEDIA_KIND


def process_definition_media_kind(process_definition: dict | None) -> str:
    settings = process_definition.get("settings") if isinstance(process_definition, dict) else None
    return process_settings_media_kind(settings)


def remap_process_settings_path(path) -> str:
    text = str(path or "").strip().replace("\\", "/")
    if len(text) == 0:
        return ""
    for old_path, new_path in MOVED_PROCESS_SETTINGS_PATHS.items():
        if text.casefold().endswith(old_path.casefold()):
            return str(PLUGIN_DIR / Path(new_path))
    return str(path)


def _process_settings_paths() -> list[Path]:
    if not PROCESS_SETTINGS_DIR.is_dir():
        return []
    paths: list[Path] = []
    for media_kind in VALID_MEDIA_KINDS:
        paths.extend(sorted((PROCESS_SETTINGS_DIR / media_kind).glob("*.json")))
    paths.extend(sorted(PROCESS_SETTINGS_DIR.glob("*.json")))
    return paths


def load_process_definitions() -> tuple[dict[str, dict], str | None]:
    if not PROCESS_SETTINGS_DIR.is_dir():
        return {}, f"Missing process settings folder: {PROCESS_SETTINGS_DIR}"
    process_definitions: dict[str, dict] = {}
    for settings_path in _process_settings_paths():
        try:
            raw_settings = json.loads(settings_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {}, f"Unable to read process setting file {settings_path.name}: {exc}"
        if not isinstance(raw_settings, dict):
            return {}, f"Process setting file {settings_path.name} must contain a JSON object."
        process_name = str(settings_path.stem).strip()
        info = raw_settings.pop("process", None)  # catalog metadata, kept out of generation settings
        model_type = str(raw_settings.get("model_type") or "").strip()
        system_handler = str(raw_settings.get("system_handler") or "").strip()
        if len(process_name) == 0:
            return {}, f"Process setting file {settings_path.name} has an empty filename stem."
        if len(model_type) == 0 and len(system_handler) == 0:
            return {}, f"Process setting file {settings_path.name} is missing model_type."
        if process_name in process_definitions:
            continue
        process_definitions[process_name] = {"settings": raw_settings, "path": remap_process_settings_path(settings_path), "media_kind": process_settings_media_kind(raw_settings), "info": info if isinstance(info, dict) else {}}
    if len(process_definitions) == 0:
        return {}, f"No process setting files were found in: {PROCESS_SETTINGS_DIR}"
    return process_definitions, None


PROCESS_DEFINITIONS, PROCESS_DEFINITIONS_ERROR = load_process_definitions()


def default_process_name(media_kind: str = DEFAULT_MEDIA_KIND) -> str:
    media_kind = normalize_media_kind(media_kind)
    launch_default_name = IMAGE_LAUNCH_DEFAULT_PROCESS_NAME if media_kind == "image" else LAUNCH_DEFAULT_PROCESS_NAME
    launch_default_definition = PROCESS_DEFINITIONS.get(launch_default_name)
    if launch_default_definition is not None and process_definition_media_kind(launch_default_definition) == media_kind:
        return launch_default_name
    for process_name, process_definition in PROCESS_DEFINITIONS.items():
        if process_definition_media_kind(process_definition) == media_kind:
            return process_name
    return next(iter(PROCESS_DEFINITIONS), "")


def default_model_type(media_kind: str = DEFAULT_MEDIA_KIND) -> str:
    process_name = default_process_name(media_kind)
    return str(PROCESS_DEFINITIONS.get(process_name, {}).get("settings", {}).get("model_type") or "")


DEFAULT_PROCESS_NAME = default_process_name(DEFAULT_MEDIA_KIND)
DEFAULT_MODEL_TYPE = default_model_type(DEFAULT_MEDIA_KIND)


def load_saved_mediaflow_settings() -> dict:
    if MEDIAFLOW_SETTINGS_FILE.is_file() and LEGACY_PROCESS_FULL_VIDEO_SETTINGS_FILE.is_file():
        try:
            LEGACY_PROCESS_FULL_VIDEO_SETTINGS_FILE.unlink()
        except OSError:
            pass
    settings_file = MEDIAFLOW_SETTINGS_FILE if MEDIAFLOW_SETTINGS_FILE.is_file() else LEGACY_PROCESS_FULL_VIDEO_SETTINGS_FILE
    if not settings_file.is_file():
        return {}
    try:
        raw_settings = json.loads(settings_file.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[MediaFlow] Warning: unable to read saved UI settings from {settings_file}: {exc}")
        return {}
    return raw_settings if isinstance(raw_settings, dict) else {}


def save_mediaflow_settings(settings: dict) -> None:
    MEDIAFLOW_SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    MEDIAFLOW_SETTINGS_FILE.write_text(json.dumps(settings, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    if LEGACY_PROCESS_FULL_VIDEO_SETTINGS_FILE.is_file():
        try:
            LEGACY_PROCESS_FULL_VIDEO_SETTINGS_FILE.unlink()
        except OSError:
            pass


def _legacy_ui_state(settings: dict) -> dict:
    excluded = {METADATA_VERSION_KEY, USER_SETTINGS_STORAGE_KEY, LAST_SETTINGS_STORAGE_KEY}
    legacy = {key: value for key, value in settings.items() if key not in excluded}
    if set(legacy) <= {MEDIA_KIND_STORAGE_KEY, BATCH_MODE_STORAGE_KEY}:
        return {}
    return legacy


def _dedupe_refs(raw_refs) -> list[str]:
    if isinstance(raw_refs, str):
        raw_refs = [raw_refs]
    if not isinstance(raw_refs, list):
        return []
    refs: list[str] = []
    seen: set[str] = set()
    for raw_ref in raw_refs:
        ref = normalize_user_settings_ref(raw_ref)
        if len(ref) == 0 or ref.casefold() in seen:
            continue
        refs.append(ref)
        seen.add(ref.casefold())
    return refs


def _empty_user_settings_by_media() -> dict[str, list[str]]:
    return {media_kind: [] for media_kind in VALID_MEDIA_KINDS}


def _normalize_user_settings_by_media(raw_user_settings, selected_media_kind: str, classify_user_ref: Callable[[str], str | None] | None = None) -> dict[str, list[str]]:
    refs_by_media = _empty_user_settings_by_media()
    if isinstance(raw_user_settings, dict):
        for media_kind in VALID_MEDIA_KINDS:
            refs_by_media[media_kind] = _dedupe_refs(raw_user_settings.get(media_kind, []))
        return refs_by_media
    for ref in _dedupe_refs(raw_user_settings):
        media_kind = ""
        if classify_user_ref is not None:
            media_kind = normalize_media_kind(classify_user_ref(ref), "")
        media_kind = media_kind or selected_media_kind
        refs_by_media[media_kind].append(ref)
    return refs_by_media


def _normalize_settings_slot(slot: dict | None, media_kind: str, batch_mode: str) -> dict:
    if not isinstance(slot, dict):
        slot = {}
    excluded = {METADATA_VERSION_KEY, USER_SETTINGS_STORAGE_KEY, LAST_SETTINGS_STORAGE_KEY}
    normalized = {key: value for key, value in slot.items() if key not in excluded}
    normalized[MEDIA_KIND_STORAGE_KEY] = normalize_media_kind(normalized.get(MEDIA_KIND_STORAGE_KEY) or media_kind)
    normalized[BATCH_MODE_STORAGE_KEY] = normalize_batch_mode(normalized.get(BATCH_MODE_STORAGE_KEY) or batch_mode)
    normalized[PRESERVE_ARTIFACTS_STORAGE_KEY] = normalize_preserve_artifact_count(normalized.get(PRESERVE_ARTIFACTS_STORAGE_KEY))
    normalized[PREVIEW_OUTPUT_STORAGE_KEY] = normalize_preview_enabled(normalized.get(PREVIEW_OUTPUT_STORAGE_KEY))
    return normalized


def _parse_last_settings_key(key: str) -> tuple[str, str]:
    text = str(key or "").strip().replace(":", "/")
    parts = [part for part in text.split("/") if part]
    if len(parts) >= 2:
        return normalize_media_kind(parts[0]), normalize_batch_mode(parts[1])
    return DEFAULT_MEDIA_KIND, DEFAULT_BATCH_MODE


def _normalize_last_settings(raw_last_settings) -> dict[str, dict]:
    normalized: dict[str, dict] = {}
    if not isinstance(raw_last_settings, dict):
        return normalized
    for raw_key, raw_slot in raw_last_settings.items():
        if str(raw_key or "").strip().lower() in VALID_MEDIA_KINDS and isinstance(raw_slot, dict):
            media_kind = normalize_media_kind(raw_key)
            for batch_mode, nested_slot in raw_slot.items():
                normalized[last_settings_key(media_kind, str(batch_mode))] = _normalize_settings_slot(nested_slot, media_kind, str(batch_mode))
            continue
        media_kind, batch_mode = _parse_last_settings_key(str(raw_key))
        normalized[last_settings_key(media_kind, batch_mode)] = _normalize_settings_slot(raw_slot, media_kind, batch_mode)
    return normalized


def media_kind_for_process_value(process_value: str, classify_user_ref: Callable[[str], str | None] | None = None) -> str:
    process_value = str(process_value or "").strip()
    process_definition = PROCESS_DEFINITIONS.get(process_value)
    if isinstance(process_definition, dict):
        return process_definition_media_kind(process_definition)
    if is_user_process_value(process_value) and classify_user_ref is not None:
        return normalize_media_kind(classify_user_ref(user_process_ref_from_value(process_value)), "")
    return ""


def migrate_mediaflow_settings(settings: dict | None, classify_user_ref: Callable[[str], str | None] | None = None) -> tuple[dict, bool]:
    settings = settings if isinstance(settings, dict) else {}
    selected_media_kind = normalize_media_kind(settings.get(MEDIA_KIND_STORAGE_KEY), DEFAULT_MEDIA_KIND)
    selected_batch_mode = normalize_batch_mode(settings.get(BATCH_MODE_STORAGE_KEY), DEFAULT_BATCH_MODE)
    result = {
        METADATA_VERSION_KEY: MEDIAFLOW_SETTINGS_VERSION,
        MEDIA_KIND_STORAGE_KEY: selected_media_kind,
        BATCH_MODE_STORAGE_KEY: selected_batch_mode,
        USER_SETTINGS_STORAGE_KEY: _normalize_user_settings_by_media(settings.get(USER_SETTINGS_STORAGE_KEY), selected_media_kind, classify_user_ref),
        LAST_SETTINGS_STORAGE_KEY: _normalize_last_settings(settings.get(LAST_SETTINGS_STORAGE_KEY)),
    }

    legacy_slot = _legacy_ui_state(settings)
    if len(legacy_slot) > 0:
        process_media_kind = media_kind_for_process_value(str(legacy_slot.get("process_name") or ""), classify_user_ref)
        slot_media_kind = normalize_media_kind(legacy_slot.get(MEDIA_KIND_STORAGE_KEY) or process_media_kind, selected_media_kind)
        slot_batch_mode = normalize_batch_mode(legacy_slot.get(BATCH_MODE_STORAGE_KEY), selected_batch_mode)
        legacy_slot = _normalize_settings_slot(legacy_slot, slot_media_kind, slot_batch_mode)
        result[LAST_SETTINGS_STORAGE_KEY][last_settings_key(slot_media_kind, slot_batch_mode)] = legacy_slot
        result[MEDIA_KIND_STORAGE_KEY] = slot_media_kind
        result[BATCH_MODE_STORAGE_KEY] = slot_batch_mode

    changed = settings.get(METADATA_VERSION_KEY) != MEDIAFLOW_SETTINGS_VERSION or result != settings
    return result, changed


def ensure_mediaflow_settings_migrated(settings: dict | None, classify_user_ref: Callable[[str], str | None] | None = None) -> dict:
    migrated, changed = migrate_mediaflow_settings(settings, classify_user_ref)
    if changed:
        save_mediaflow_settings(migrated)
    return migrated


def current_media_kind(settings: dict | None) -> str:
    migrated, _changed = migrate_mediaflow_settings(settings)
    return normalize_media_kind(migrated.get(MEDIA_KIND_STORAGE_KEY), DEFAULT_MEDIA_KIND)


def current_batch_mode(settings: dict | None) -> str:
    migrated, _changed = migrate_mediaflow_settings(settings)
    return normalize_batch_mode(migrated.get(BATCH_MODE_STORAGE_KEY), DEFAULT_BATCH_MODE)


def get_last_ui_settings(settings: dict | None, media_kind: str | None = None, batch_mode: str | None = None) -> dict:
    migrated, _changed = migrate_mediaflow_settings(settings)
    media_kind = normalize_media_kind(media_kind or migrated.get(MEDIA_KIND_STORAGE_KEY), DEFAULT_MEDIA_KIND)
    batch_mode = normalize_batch_mode(batch_mode or migrated.get(BATCH_MODE_STORAGE_KEY), DEFAULT_BATCH_MODE)
    slot = migrated.get(LAST_SETTINGS_STORAGE_KEY, {}).get(last_settings_key(media_kind, batch_mode), {})
    current = dict(slot) if isinstance(slot, dict) else {}
    current[MEDIA_KIND_STORAGE_KEY] = media_kind
    current[BATCH_MODE_STORAGE_KEY] = batch_mode
    return current


def load_saved_process_full_video_settings() -> dict:
    return get_last_ui_settings(load_saved_mediaflow_settings(), "video", "single")


def save_process_full_video_settings(settings: dict) -> None:
    save_mediaflow_ui_settings(settings, "video", normalize_batch_mode(settings.get(BATCH_MODE_STORAGE_KEY), DEFAULT_BATCH_MODE))


def normalize_user_settings_ref(value) -> str:
    text = str(value or "").strip().strip('"').replace("\\", "/")
    if len(text) == 0 or text.startswith(("/", "./", "../")):
        return ""
    if len(text) >= 2 and text[1] == ":":
        return ""
    parts = [part.strip() for part in text.split("/") if len(part.strip()) > 0]
    if len(parts) != 2:
        return ""
    base_model_type, filename = parts
    filename = Path(filename).name
    if not is_wangp_settings_filename(filename):
        return ""
    return f"{base_model_type}/{filename}"


def is_user_process_value(value) -> bool:
    return str(value or "").startswith(USER_PROCESS_VALUE_PREFIX)


def user_process_value(ref: str) -> str:
    normalized = normalize_user_settings_ref(ref)
    return f"{USER_PROCESS_VALUE_PREFIX}{normalized}" if len(normalized) > 0 else ""


def user_process_ref_from_value(value) -> str:
    text = str(value or "").strip()
    if not text.startswith(USER_PROCESS_VALUE_PREFIX):
        return ""
    return normalize_user_settings_ref(text[len(USER_PROCESS_VALUE_PREFIX):])


def get_saved_user_settings_refs(settings: dict | None, media_kind: str | None = DEFAULT_MEDIA_KIND) -> list[str]:
    raw_refs = settings.get(USER_SETTINGS_STORAGE_KEY, []) if isinstance(settings, dict) else []
    if isinstance(raw_refs, dict):
        if media_kind is None:
            refs: list[str] = []
            seen: set[str] = set()
            for kind in VALID_MEDIA_KINDS:
                for ref in _dedupe_refs(raw_refs.get(kind, [])):
                    if ref.casefold() not in seen:
                        refs.append(ref)
                        seen.add(ref.casefold())
            return refs
        return _dedupe_refs(raw_refs.get(normalize_media_kind(media_kind), []))
    return _dedupe_refs(raw_refs)


def store_user_settings_refs(refs: list[str], media_kind: str = DEFAULT_MEDIA_KIND) -> None:
    saved_settings = ensure_mediaflow_settings_migrated(load_saved_mediaflow_settings())
    media_kind = normalize_media_kind(media_kind)
    user_settings = _normalize_user_settings_by_media(saved_settings.get(USER_SETTINGS_STORAGE_KEY), media_kind)
    user_settings[media_kind] = _dedupe_refs(refs)
    saved_settings[USER_SETTINGS_STORAGE_KEY] = user_settings
    save_mediaflow_settings(saved_settings)


def save_mediaflow_ui_settings(settings: dict, media_kind: str | None = None, batch_mode: str | None = None) -> None:
    saved_settings = ensure_mediaflow_settings_migrated(load_saved_mediaflow_settings())
    media_kind = normalize_media_kind(media_kind or settings.get(MEDIA_KIND_STORAGE_KEY) or saved_settings.get(MEDIA_KIND_STORAGE_KEY), DEFAULT_MEDIA_KIND)
    batch_mode = normalize_batch_mode(batch_mode or settings.get(BATCH_MODE_STORAGE_KEY) or saved_settings.get(BATCH_MODE_STORAGE_KEY), DEFAULT_BATCH_MODE)
    saved_settings[MEDIA_KIND_STORAGE_KEY] = media_kind
    saved_settings[BATCH_MODE_STORAGE_KEY] = batch_mode
    saved_settings.setdefault(LAST_SETTINGS_STORAGE_KEY, {})[last_settings_key(media_kind, batch_mode)] = _normalize_settings_slot(settings, media_kind, batch_mode)
    save_mediaflow_settings(saved_settings)


def save_process_full_video_ui_settings(settings: dict) -> None:
    save_mediaflow_ui_settings(settings, "video", normalize_batch_mode(settings.get(BATCH_MODE_STORAGE_KEY), DEFAULT_BATCH_MODE))


def save_mediaflow_selection(media_kind: str, batch_mode: str, process_model_type: str, process_name: str) -> None:
    saved_settings = ensure_mediaflow_settings_migrated(load_saved_mediaflow_settings())
    media_kind = normalize_media_kind(media_kind)
    batch_mode = normalize_batch_mode(batch_mode)
    slot_key = last_settings_key(media_kind, batch_mode)
    slot = dict(saved_settings.get(LAST_SETTINGS_STORAGE_KEY, {}).get(slot_key, {}))
    slot[MEDIA_KIND_STORAGE_KEY] = media_kind
    slot[BATCH_MODE_STORAGE_KEY] = batch_mode
    slot["process_model_type"] = str(process_model_type or "").strip()
    slot["process_name"] = str(process_name or "").strip()
    saved_settings[MEDIA_KIND_STORAGE_KEY] = media_kind
    saved_settings[BATCH_MODE_STORAGE_KEY] = batch_mode
    saved_settings.setdefault(LAST_SETTINGS_STORAGE_KEY, {})[slot_key] = slot
    save_mediaflow_settings(saved_settings)


def save_process_full_video_selection(process_model_type: str, process_name: str) -> None:
    save_mediaflow_selection("video", "single", process_model_type, process_name)
