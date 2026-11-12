# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from threading import RLock
from typing import Any

VIEW_FOLDER = "views"  # a view is a small file in such a folder, named like a video, that names a part of another video: nothing is encoded
_VIEW_MARK = b'{"wangp_view"'
_VIRTUAL_MEDIA_SOURCES: dict[str, dict[str, dict[str, Any]]] = {}
_VIRTUAL_MEDIA_LOCK = RLock()


@dataclass(frozen=True)
class VirtualMediaSpec:
    source_path: str
    start_frame: int = 0
    end_frame: int | None = None
    audio_track_no: int | None = None
    extras: tuple[tuple[str, str], ...] = ()
    pad_frames: int = 0
    start_s: float | None = None  # a part in seconds: what an audio file has; a video's seconds are its frames at its real rate (in_frames)
    end_s: float | None = None

    def as_suffix_items(self) -> list[tuple[str, str]]:
        items: list[tuple[str, str]] = []
        if self.start_frame != 0:
            items.append(("start_frame", str(int(self.start_frame))))
        if self.end_frame is not None:
            items.append(("end_frame", str(int(self.end_frame))))
        if self.start_s is not None:
            items.append(("start_s", _seconds_text(self.start_s)))
        if self.end_s is not None:
            items.append(("end_s", _seconds_text(self.end_s)))
        if self.audio_track_no is not None:
            items.append(("audio_track_no", str(int(self.audio_track_no))))
        if self.pad_frames:
            items.append(("pad_frames", str(int(self.pad_frames))))
        items.extend(list(self.extras))
        return items


def bounced_frame(number: int, own_frames: int) -> int:
    """The frame shown at a position of a padded view: its own frames, then those frames played backwards and forwards. Nothing is encoded for it."""
    if number < own_frames or own_frames < 2:
        return min(number, own_frames - 1)
    position = (number - own_frames) % (2 * own_frames - 2)
    return own_frames - 2 - position if position <= own_frames - 2 else position - own_frames + 2


def write_view(path: Any, source_path: Any, start_frame: int, end_frame: int, pad_frames: int = 0) -> str:
    """Write a view: frames start_frame to end_frame (zero-based, inclusive) of a video, then pad_frames more of those frames played backwards and forwards, for a model that renders certain lengths only.
    The source is kept relative to the view where it can be, so that a moved production keeps its views."""
    path, source_path = os.path.abspath(path), os.path.abspath(source_path)
    try:
        source = os.path.relpath(source_path, os.path.dirname(path))
    except ValueError:  # another drive
        source = source_path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"wangp_view": 1, "source": source, "start_frame": int(start_frame), "end_frame": int(end_frame), **({"pad_frames": int(pad_frames)} if pad_frames else {})}, handle)
    return path


def read_view(path: Any) -> dict[str, Any] | None:
    """What a view names, with its source as a full path; None for any other file."""
    text = os.fspath(path) if isinstance(path, os.PathLike) else path
    if not isinstance(text, str) or VIEW_FOLDER not in text or "|" in text:
        return None
    try:
        if os.path.basename(os.path.dirname(os.path.abspath(text))) != VIEW_FOLDER or os.path.getsize(text) > 4096:
            return None
        with open(text, "rb") as handle:
            content = handle.read()
    except OSError:
        return None
    if not content.startswith(_VIEW_MARK):
        return None
    view = json.loads(content)
    return {**view, "source": os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(text)), view["source"]))}


def parse_virtual_media_path(value: Any) -> VirtualMediaSpec | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if "|" not in text:
        view = read_view(text)
        return None if view is None else VirtualMediaSpec(source_path=view["source"], start_frame=view["start_frame"], end_frame=view["end_frame"], pad_frames=view.get("pad_frames", 0))
    source_path, suffix = text.split("|", 1)
    source_path = source_path.strip()
    if len(source_path) == 0:
        return None
    values: dict[str, str] = {}
    extras: list[tuple[str, str]] = []
    for raw_item in suffix.split(","):
        item = raw_item.strip()
        if len(item) == 0:
            continue
        key, sep, raw_value = item.partition("=")
        key = key.strip().lower()
        if sep == "":
            extras.append((item, ""))
            continue
        value_text = raw_value.strip()
        if key in ("start_frame", "end_frame", "audio_track_no", "pad_frames", "start_s", "end_s"):
            values[key] = value_text
        else:
            extras.append((key, value_text))
    if {"start_frame", "end_frame"} & values.keys() and {"start_s", "end_s"} & values.keys():
        raise ValueError(f"Name a part in frames (start_frame, end_frame) or in seconds (start_s, end_s), not both: {text}")
    return VirtualMediaSpec(
        source_path=source_path,
        start_frame=_parse_int(values.get("start_frame"), 0),
        end_frame=_parse_optional_int(values.get("end_frame")),
        audio_track_no=_parse_optional_int(values.get("audio_track_no")),
        extras=tuple(extras),
        pad_frames=_parse_int(values.get("pad_frames"), 0),
        start_s=_parse_optional_seconds(values.get("start_s")),
        end_s=_parse_optional_seconds(values.get("end_s")),
    )


def in_frames(spec: VirtualMediaSpec, fps: float) -> VirtualMediaSpec:
    """A part named in seconds as frames of a video at fps, its real rate: from the frame at start_s to the last one before end_s."""
    if spec.start_s is None and spec.end_s is None:
        return spec
    return replace(spec, start_frame=0 if spec.start_s is None else round(spec.start_s * fps), end_frame=None if spec.end_s is None else round(spec.end_s * fps) - 1, start_s=None, end_s=None)


def strip_virtual_media_suffix(value: Any) -> Any:
    spec = parse_virtual_media_path(value)
    return spec.source_path if spec is not None else value


def media_source_exists(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    if get_virtual_media_vsource(value) is not None:
        return get_virtual_media_entry(value) is not None
    return os.path.isfile(strip_virtual_media_suffix(value))


def build_virtual_media_path(
    source_path: str,
    *,
    start_frame: int | None = None,
    end_frame: int | None = None,
    audio_track_no: int | None = None,
    extras: dict[str, Any] | None = None,
    pad_frames: int = 0,
    start_s: float | None = None,
    end_s: float | None = None,
) -> str:
    base_path = str(source_path or "").strip()
    if len(base_path) == 0:
        return base_path
    parts: list[str] = []
    if start_frame is not None:
        parts.append(f"start_frame={int(start_frame)}")
    if end_frame is not None:
        parts.append(f"end_frame={int(end_frame)}")
    if start_s is not None:
        parts.append(f"start_s={_seconds_text(start_s)}")
    if end_s is not None:
        parts.append(f"end_s={_seconds_text(end_s)}")
    if audio_track_no is not None:
        parts.append(f"audio_track_no={int(audio_track_no)}")
    if pad_frames:
        parts.append(f"pad_frames={int(pad_frames)}")
    for key, value in (extras or {}).items():
        key_text = str(key or "").strip()
        value_text = str(value or "").strip()
        if len(key_text) == 0 or len(value_text) == 0:
            continue
        parts.append(f"{key_text}={value_text}")
    return base_path if len(parts) == 0 else f"{base_path}|{','.join(parts)}"


def replace_virtual_media_source(value: Any, source_path: str) -> Any:
    spec = parse_virtual_media_path(value)
    if spec is None:
        return source_path
    extras = dict(spec.extras)
    return build_virtual_media_path(
        source_path,
        start_frame=spec.start_frame if spec.start_frame != 0 else None,
        end_frame=spec.end_frame,
        audio_track_no=spec.audio_track_no,
        extras=extras,
        pad_frames=spec.pad_frames,
        start_s=spec.start_s,
        end_s=spec.end_s,
    )


def padded_virtual_media(value: Any, own_frames: int, total_frames: int) -> str:
    """A virtual video read as total_frames long, the frames past its own ones being those played backwards and forwards."""
    spec = parse_virtual_media_path(value)
    if spec.start_s is not None or spec.end_s is not None:
        from .video_decode import probe_video_stream_metadata
        spec = in_frames(spec, probe_video_stream_metadata(spec.source_path)["fps_float"])
    return build_virtual_media_path(spec.source_path, start_frame=spec.start_frame, end_frame=spec.start_frame + own_frames - 1, audio_track_no=spec.audio_track_no, extras=dict(spec.extras), pad_frames=total_frames - own_frames)


def clamp_virtual_frame_range(spec: VirtualMediaSpec | None, total_frames: int) -> tuple[int, int | None]:
    if spec is None:
        return 0, None
    explicit_frame_count = _parse_int(_extras_dict(spec).get("frame_count"), 0)
    total_frames = explicit_frame_count if explicit_frame_count > 0 else int(total_frames or 0)
    if total_frames <= 0:
        return 0, None
    start_frame = _resolve_relative_frame_index(spec.start_frame, total_frames, default_to_end=False)
    end_frame = _resolve_relative_frame_index(spec.end_frame, total_frames, default_to_end=True)
    if end_frame is None:
        return start_frame, total_frames - 1
    end_frame = max(start_frame, end_frame)
    return start_frame, end_frame


def get_virtual_media_vsource(value: Any) -> str | None:
    spec = value if isinstance(value, VirtualMediaSpec) else parse_virtual_media_path(value)
    if spec is None:
        return None
    vsource = _extras_dict(spec).get("vsource", "").strip()
    return vsource or None


def get_virtual_media_entry(value: Any) -> dict[str, Any] | None:
    spec = value if isinstance(value, VirtualMediaSpec) else parse_virtual_media_path(value)
    if spec is None:
        return None
    vsource = get_virtual_media_vsource(spec)
    if vsource is None:
        return None
    with _VIRTUAL_MEDIA_LOCK:
        entry = _VIRTUAL_MEDIA_SOURCES.get(vsource, {}).get(str(spec.source_path or "").strip())
        return None if entry is None else dict(entry)


def store_virtual_video(vsource: str, name: str, tensor: Any, fps: float, *, hdr: bool = False) -> None:
    import torch

    tensor = tensor.detach().cpu().to(dtype=torch.float32).contiguous().clone()
    with _VIRTUAL_MEDIA_LOCK:
        _VIRTUAL_MEDIA_SOURCES.setdefault(str(vsource).strip(), {})[str(name).strip()] = {
            "kind": "video",
            "tensor": tensor,
            "fps": max(float(fps or 0.0), 1.0),
            "hdr": bool(hdr),
        }
    _clear_virtual_media_caches()


def get_virtual_video(value: Any, name: str | None = None) -> Any:
    entry = _get_virtual_media_entry_from_key(value, name)
    if entry is None or entry.get("kind") != "video":
        return None
    tensor = entry.get("tensor")
    return None if tensor is None else tensor.clone()


def store_virtual_image(vsource: str, name: str, image: Any) -> None:
    with _VIRTUAL_MEDIA_LOCK:
        _VIRTUAL_MEDIA_SOURCES.setdefault(str(vsource).strip(), {})[str(name).strip()] = {"kind": "image", "image": image.copy()}
    _clear_virtual_media_caches()


def get_virtual_image(value: Any, name: str | None = None) -> Any:
    entry = _get_virtual_media_entry_from_key(value, name)
    if entry is None or entry.get("kind") != "image":
        return None
    image = entry.get("image")
    return None if image is None else image.copy()


def clear_virtual_media_source(vsource: str) -> None:
    with _VIRTUAL_MEDIA_LOCK:
        _VIRTUAL_MEDIA_SOURCES.pop(str(vsource).strip(), None)
    _clear_virtual_media_caches()


def _parse_int(value: Any, default: int) -> int:
    try:
        return int(str(value or "").strip())
    except (TypeError, ValueError):
        return default


def _parse_optional_int(value: Any) -> int | None:
    text = str(value or "").strip()
    if len(text) == 0:
        return None
    try:
        return int(text)
    except (TypeError, ValueError):
        return None


def _parse_optional_seconds(value: Any) -> float | None:
    text = str(value or "").strip()
    if len(text) == 0:
        return None
    try:
        seconds = float(text)
    except ValueError:
        raise ValueError(f"Seconds must be a number, not {text!r}") from None
    if not seconds >= 0:
        raise ValueError(f"Seconds cannot be negative: {text}")
    return seconds


def _seconds_text(seconds: float) -> str:
    return f"{float(seconds):.6f}".rstrip("0").rstrip(".")


def _resolve_relative_frame_index(value: int | None, total_frames: int, *, default_to_end: bool) -> int | None:
    if total_frames <= 0:
        return None if value is None else 0
    if value is None:
        return total_frames - 1 if default_to_end else 0
    index = int(value)
    if index < 0:
        index = total_frames + index
    return max(0, min(index, total_frames - 1))


def _extras_dict(spec: VirtualMediaSpec) -> dict[str, str]:
    return {str(key or "").strip().lower(): str(value or "").strip() for key, value in spec.extras}


def _get_virtual_media_entry_from_key(value: Any, name: str | None = None) -> dict[str, Any] | None:
    if name is None:
        return get_virtual_media_entry(value)
    with _VIRTUAL_MEDIA_LOCK:
        entry = _VIRTUAL_MEDIA_SOURCES.get(str(value).strip(), {}).get(str(name).strip())
        return None if entry is None else dict(entry)


def _clear_virtual_media_caches() -> None:
    try:
        from . import video_decode as _video_decode

        _video_decode.probe_video_stream_metadata.cache_clear()
    except Exception:
        pass
    try:
        from . import utils as _utils

        _utils._get_video_info_cached.cache_clear()
        _utils._get_video_info_details_cached.cache_clear()
    except Exception:
        pass
