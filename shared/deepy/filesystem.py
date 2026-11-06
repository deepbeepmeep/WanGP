# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import glob
import hashlib
import mimetypes
import os
import re
import shutil
import stat
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import ffmpeg
from PIL import Image

from shared.deepy.image_channels import image_color_details, upright_size
from shared.deepy.media_registry import detect_media_type
from shared.utils.utils import get_video_info
from shared.utils.video_decode import resolve_media_binary
from shared.utils.virtual_media import read_view


TEXT_MAX_CHARS = 16000
SEARCH_MAX_FILE_BYTES = 32 * 1024 * 1024
_TEXT_ENCODINGS = {"utf-8", "utf-8-sig"}
_PATH_KEYS = {"destination", "generated_files", "output_file", "output_files", "path", "paths", "source_path"}


def _resolved_path(value: Any) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Path is empty.")
    return Path(text).expanduser().resolve()


def _unique_paths(values: list[Any]) -> tuple[Path, ...]:
    paths = []
    seen = set()
    for value in values:
        try:
            path = _resolved_path(value)
        except ValueError:
            continue
        key = os.path.normcase(str(path))
        if key not in seen:
            seen.add(key)
            paths.append(path)
    return tuple(paths)


def _inside(path: Path, root: Path) -> bool:
    try:
        return os.path.normcase(os.path.commonpath((str(path), str(root)))) == os.path.normcase(str(root))
    except ValueError:
        return False


def _alias_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "").strip()).strip("_-").lower()


def _automatic_alias(path: Path, used: set[str]) -> str:
    components = [_alias_component(part) for part in path.parts if part not in {path.anchor, path.drive, "\\", "/"}]
    components = [part for part in components if part]
    for count in range(1, len(components) + 1):
        alias = "_".join(components[-count:])
        if alias.casefold() not in used and re.fullmatch(r"outputs\d*", alias.casefold()) is None:
            return alias
    base = "_".join(components) or "root"
    if base.casefold() not in used and re.fullmatch(r"outputs\d*", base.casefold()) is None:
        return base
    index = 2
    while f"{base}{index if base == 'root' else f'_{index}'}".casefold() in used:
        index += 1
    return f"{base}{index if base == 'root' else f'_{index}'}"


@dataclass(frozen=True)
class FileAccessPolicy:
    mode: str
    output_roots: tuple[Path, ...]
    selected_roots: tuple[Path, ...] = ()
    read_everywhere: bool = False
    root_aliases: tuple[str, ...] = ()

    @property
    def read_enabled(self) -> bool:
        return self.mode in {"read", "read_write"}

    @property
    def write_enabled(self) -> bool:
        return self.mode == "read_write"

    @property
    def read_roots(self) -> tuple[Path, ...]:
        return _unique_paths([*self.output_roots, *self.selected_roots])

    @property
    def write_roots(self) -> tuple[Path, ...]:
        return self.read_roots if self.write_enabled else ()

    @property
    def aliases(self) -> tuple[str, ...]:
        if self.root_aliases:
            return self.root_aliases
        output_aliases = ["outputs" if index == 0 else f"outputs{index + 1}" for index in range(len(self.output_roots))]
        used = {alias.casefold() for alias in output_aliases}
        selected_aliases = []
        for root in self.read_roots[len(self.output_roots):]:
            alias = _automatic_alias(root, used)
            used.add(alias.casefold())
            selected_aliases.append(alias)
        return tuple([*output_aliases, *selected_aliases])

    @property
    def mounts(self) -> tuple[tuple[str, Path], ...]:
        return tuple(zip(self.aliases, self.read_roots))

    @property
    def virtualized(self) -> bool:
        return not self.read_everywhere

    def _virtual_target(self, text: str) -> Path | None:
        reference = text.replace("\\", "/")
        if reference.startswith("@"):
            reference = reference[1:]
        elif text.startswith("\\") and not text.startswith("\\\\"):
            reference = reference[1:]
        else:
            return None
        alias, separator, remainder = reference.partition("/")
        alias = re.sub(r"^workspace(\d*)$", r"documents\1", alias, flags=re.IGNORECASE)  # the documents root was named @workspace before
        root = next((path for name, path in self.mounts if name.casefold() == alias.casefold()), None)
        if root is None:
            raise ValueError(f"Unknown virtual filesystem root: {alias or '(empty)'}. List authorized roots without a path first.")
        parts = [part for part in remainder.split("/") if part] if separator else []
        return root.joinpath(*parts).resolve()

    def resolve_path(self, path: Any) -> Path:
        text = str(path or "").strip()
        if not text:
            raise ValueError("Path is empty.")
        virtual_target = self._virtual_target(text)
        if virtual_target is not None:
            return virtual_target
        target = Path(text).expanduser()
        absolute = target.is_absolute() or re.match(r"^[A-Za-z]:[\\/]", text) is not None or text.startswith(("\\\\", "/"))
        if absolute and self.virtualized and not isinstance(path, Path):
            raise PermissionError("Absolute filesystem paths require Read Everywhere; use a virtual root returned by the directory listing tool.")
        return (target if absolute else self.output_roots[0] / target).resolve()

    def virtualize_path(self, path: Any) -> str:
        target = _resolved_path(path)
        if not self.virtualized:
            return str(target)
        for alias, root in sorted(self.mounts, key=lambda item: len(str(item[1])), reverse=True):
            if not _inside(target, root):
                continue
            relative = target.relative_to(root).as_posix()
            return f"@{alias}/{relative}" if relative != "." else f"@{alias}"
        return target.name or "file"

    def _virtualize_text(self, value: str) -> str:
        text = str(value)
        for alias, root in sorted(self.mounts, key=lambda item: len(str(item[1])), reverse=True):
            for physical in {str(root), root.as_posix()}:
                prefix = physical.rstrip("\\/") or physical
                text = re.sub(rf"{re.escape(prefix)}(?=$|[\\/])", f"@{alias}", text, flags=re.IGNORECASE)
        return text

    def virtualize_result(self, value: Any, key: str = "") -> Any:
        if not self.virtualized:
            return value
        if isinstance(value, dict):
            return {child_key: self.virtualize_result(child_value, str(child_key).casefold()) for child_key, child_value in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.virtualize_result(item, key) for item in value]
        if isinstance(value, (str, os.PathLike)):
            text = str(value)
            absolute = Path(text).is_absolute() or re.match(r"^[A-Za-z]:[\\/]", text) is not None or text.startswith("\\\\")
            virtual = text.startswith(("@", "\\", "/"))
            path_key = key in _PATH_KEYS or key.endswith(("_path", "_paths")) or key.endswith(("_file", "_files", "_folder", "_folders", "_directory", "_directories")) and (absolute or virtual)
            if path_key or key in {"source", "sources"} and (absolute or virtual):
                try:
                    target = self.resolve_path(text) if virtual and not absolute else _resolved_path(text)
                    return self.virtualize_path(target) if any(_inside(target, root) for _alias, root in self.mounts) else None
                except (OSError, PermissionError, ValueError):
                    if absolute:
                        return None
            return self._virtualize_text(text)
        return value

    def can_read(self, path: Any) -> bool:
        if not self.read_enabled:
            return False
        try:
            target = self.resolve_path(path)
        except (OSError, PermissionError, ValueError):
            return False
        return self.read_everywhere or any(_inside(target, root) for root in self.read_roots)

    def can_write(self, path: Any) -> bool:
        if not self.write_enabled:
            return False
        try:
            target = self.resolve_path(path)
        except (OSError, PermissionError, ValueError):
            return False
        return any(_inside(target, root) for root in self.write_roots)

    def require_read(self, path: Any, *, file: bool = False, directory: bool = False) -> Path:
        target = self.resolve_path(path)
        if not self.can_read(target):
            raise PermissionError(f"Filesystem read is not authorized for: {self.virtualize_path(target)}")
        if file and not target.is_file():
            named = sorted(target.parent.glob(f"{glob.escape(target.name)}.*")) if not target.suffix and target.parent.is_dir() else []
            if not named and isinstance(path, str) and not path.startswith("@") and not Path(path).is_absolute():  # a relative path meant for another folder, e.g. a Python sandbox file ('scratch/a.png')
                named = [candidate for root in self.read_roots for candidate in (root / path, root / "sandbox" / path) if candidate.is_file()]
            raise FileNotFoundError(f"Path is not an existing file: {self.virtualize_path(target)}" + (f" (did you mean {', '.join(self.virtualize_path(item) for item in named[:3])}?)" if named else ""))
        if directory and not target.is_dir():
            raise FileNotFoundError(f"Path is not an existing directory: {self.virtualize_path(target)}")
        if not file and not directory and not target.exists():
            raise FileNotFoundError(f"Path does not exist: {self.virtualize_path(target)}")
        return target

    def require_write(self, path: Any) -> Path:
        target = self.resolve_path(path)
        if not self.can_write(target):
            raise PermissionError(f"Filesystem write is not authorized for: {self.virtualize_path(target)}")
        return target

    def create_only(self, path: Any) -> bool:
        return False

    def require_mutation(self, path: Any) -> Path:
        target = self.require_write(path)
        if target in self.write_roots:
            raise PermissionError(f"A filesystem root cannot be moved or deleted: {self.virtualize_path(target)}")
        return target

    def roots(self) -> list[dict[str, Any]]:
        return [{"path": self.virtualize_path(path), "exists": path.is_dir(), "writable": self.can_write(path)} for path in self.read_roots]

    def with_root(self, path: Any, alias: str) -> "FileAccessPolicy":
        root = _resolved_path(path)
        normalized_alias = _alias_component(alias)
        if not normalized_alias:
            raise ValueError("Filesystem root alias is empty.")
        if any(name.casefold() == normalized_alias.casefold() for name in self.aliases):
            raise ValueError(f"Filesystem root alias already exists: {normalized_alias}")
        return FileAccessPolicy(
            mode=self.mode,
            output_roots=self.output_roots,
            selected_roots=tuple([*self.selected_roots, root]),
            read_everywhere=self.read_everywhere,
            root_aliases=tuple([*self.aliases, normalized_alias]),
        )


def build_file_access_policy(server_config: dict[str, Any] | None, *, unrestricted_read: bool = False) -> FileAccessPolicy:
    from shared.deepy.config import (
        DEEPY_ALLOW_READ_FILE_SYSTEM_KEY,
        DEEPY_FILE_SYSTEM_ACCESS_DISABLED,
        DEEPY_FILE_SYSTEM_ACCESS_READ,
        DEEPY_FILE_SYSTEM_PATHS_DEFAULT,
        DEEPY_FILE_SYSTEM_PATHS_KEY,
        DEEPY_READ_EVERYWHERE_DEFAULT,
        DEEPY_READ_EVERYWHERE_KEY,
        normalize_deepy_file_system_access,
        parse_deepy_file_system_paths,
        normalize_deepy_read_everywhere,
    )

    config = dict(server_config or {})
    mode = DEEPY_FILE_SYSTEM_ACCESS_READ if unrestricted_read else normalize_deepy_file_system_access(config.get(DEEPY_ALLOW_READ_FILE_SYSTEM_KEY, False))
    video_output = config.get("save_path", "outputs") or "outputs"
    output_roots = _unique_paths([video_output, config.get("image_save_path", video_output) or video_output, config.get("audio_save_path", video_output) or video_output, config.get("mesh_save_path", video_output) or video_output])
    configured_roots = parse_deepy_file_system_paths(config.get(DEEPY_FILE_SYSTEM_PATHS_KEY, DEEPY_FILE_SYSTEM_PATHS_DEFAULT)) if mode != DEEPY_FILE_SYSTEM_ACCESS_DISABLED else []
    seen_paths = {os.path.normcase(str(path)) for path in output_roots}
    selected = []
    for value, alias in configured_roots:
        path = _resolved_path(value)
        key = os.path.normcase(str(path))
        if key not in seen_paths:
            seen_paths.add(key)
            selected.append((path, alias))
    output_aliases = ["outputs" if index == 0 else f"outputs{index + 1}" for index in range(len(output_roots))]
    used_aliases = {alias.casefold() for alias in output_aliases} | {alias.casefold() for _path, alias in selected if alias}
    selected_aliases = []
    for path, alias in selected:
        alias = alias or _automatic_alias(path, used_aliases)
        used_aliases.add(alias.casefold())
        selected_aliases.append(alias)
    selected_roots = tuple(path for path, _alias in selected)
    read_everywhere = unrestricted_read or mode != DEEPY_FILE_SYSTEM_ACCESS_DISABLED and normalize_deepy_read_everywhere(config.get(DEEPY_READ_EVERYWHERE_KEY, DEEPY_READ_EVERYWHERE_DEFAULT))
    return FileAccessPolicy(mode=mode, output_roots=output_roots, selected_roots=selected_roots, read_everywhere=read_everywhere, root_aliases=tuple([*output_aliases, *selected_aliases]))


def list_entries(policy: FileAccessPolicy, path: str = "", pattern: str = "*", recursive: bool = False, limit: int = 200, offset: int = 0, media_type: str = "all") -> dict[str, Any]:
    if not str(path or "").strip():
        roots = policy.roots()
        return {"status": "done", "roots": roots, "count": len(roots), "read_everywhere": bool(policy.read_everywhere), "error": ""}
    directory = policy.require_read(path, directory=True)
    pattern = str(pattern or "*").strip() or "*"
    limit = max(1, min(int(limit), 1000))
    offset = max(0, int(offset))
    media_type = str(media_type or "all").strip().lower()
    if media_type not in {"all", "image", "video", "audio", "3d", "world", "txt", "other"}:
        raise ValueError("media_type must be all, image, video, audio, 3d, world, txt, or other")
    candidates = directory.rglob(pattern) if recursive else iter(sorted(directory.glob(pattern), key=lambda entry: str(entry).casefold()))
    entries = []
    for item in candidates:
        if not policy.can_read(item):
            continue
        if media_type != "all":
            detected_type = detect_media_type(str(item)) if item.is_file() else ""
            if media_type == "txt":
                matches_type = item.suffix.lower() == ".txt"
            elif media_type == "other":
                matches_type = detected_type == "any" and item.suffix.lower() != ".txt"
            else:
                matches_type = detected_type == media_type
            if not matches_type:
                continue
        if len(entries) < offset:
            entries.append(None)
            continue
        stat = item.stat()
        entries.append({"name": item.name, "path": str(item.resolve()), "type": "directory" if item.is_dir() else "file", "size_bytes": None if item.is_dir() else stat.st_size, "modified": stat.st_mtime})
        if len(entries) >= offset + limit + 1:
            break
    visible = [entry for entry in entries[offset:offset + limit] if entry is not None]
    has_more = len(entries) > offset + limit
    return {"status": "done", "path": str(directory), "entries": visible, "count": len(visible), "offset": offset, "has_more": has_more, "next_offset": offset + len(visible) if has_more else None, "error": ""}


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _rate(value: Any) -> float | None:
    text = str(value or "").strip()
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        denominator_value = _float(denominator)
        return None if not denominator_value else float(numerator) / denominator_value
    return _float(text)


def _stream_summary(stream: dict[str, Any]) -> dict[str, Any]:
    tags = stream.get("tags", {})
    title = tags.get("title") or tags.get("handler_name", "")  # MP4 keeps track names as handler_name
    summary = {"type": stream["codec_type"], "codec": stream.get("codec_name")}
    if stream["codec_type"] == "audio":
        summary.update(sample_rate=_integer(stream.get("sample_rate")), channels=_integer(stream.get("channels")), layout=stream.get("channel_layout"))
    if tags.get("language", "und") != "und":
        summary["language"] = tags["language"]
    if title and not re.fullmatch(r"\w*Handler", title):  # FFmpeg's VideoHandler / SoundHandler defaults name nothing
        summary["title"] = title
    if stream["codec_type"] == "subtitle" and stream.get("disposition", {}).get("default"):
        summary["default"] = True
    return {key: value for key, value in summary.items() if value is not None}


def _probe_media(path: Path, media_type: str) -> dict[str, Any]:
    probe = ffmpeg.probe(str(path), cmd=resolve_media_binary("ffprobe") or "ffprobe")
    streams = probe.get("streams", [])
    format_info = probe.get("format", {})
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    duration = _float(format_info.get("duration"))
    result = {
        "status": "done", "path": str(path), "filename": path.name, "size_bytes": path.stat().st_size, "file_type": media_type,
        "duration_seconds": duration, "has_audio": bool(audio_streams), "audio_track_count": len(audio_streams), "error": "",
        "streams": [_stream_summary(stream) for stream in streams if stream.get("codec_type") in {"video", "audio", "subtitle"}],
    }
    if media_type == "video":
        video_stream = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
        fps = _rate(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate"))
        frame_count = _integer(video_stream.get("nb_frames"))
        if frame_count is None and fps is not None and duration is not None:
            frame_count = int(round(fps * duration))
        width, height = _integer(video_stream.get("width")), _integer(video_stream.get("height"))
        result.update({"width": width, "height": height, "resolution": None if width is None or height is None else f"{width}x{height}", "frame_count": frame_count, "fps": fps})
    else:
        audio_stream = audio_streams[0] if audio_streams else {}
        result.update({"sample_rate": _integer(audio_stream.get("sample_rate")), "channels": _integer(audio_stream.get("channels"))})
    return result


def _query_image(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        width, height = upright_size(image)
        color_details = image_color_details(image)
        frame_count = int(getattr(image, "n_frames", 1) or 1)
        duration = sum(float(image.seek(index) or image.info.get("duration", 0) or 0) for index in range(frame_count)) / 1000 if frame_count > 1 else None
    return {"status": "done", "path": str(path), "filename": path.name, "size_bytes": path.stat().st_size, "file_type": "image", "width": width, "height": height, "resolution": f"{width}x{height}", "frame_count": frame_count, "fps": None if not duration else frame_count / duration, "duration_seconds": duration, "has_audio": False, "audio_track_count": 0, **color_details, "error": ""}


def file_info(path: str, policy: FileAccessPolicy | None = None) -> dict[str, Any]:
    target = policy.require_read(path) if policy is not None else _resolved_path(path)
    if target.is_dir():
        stat = target.stat()
        return {"status": "done", "path": str(target), "filename": target.name, "file_type": "directory", "size_bytes": None, "modified": stat.st_mtime, "error": ""}
    if (view := read_view(target)) is not None:  # a part of a video: the source's streams, its own length, and where it comes from
        fps, _width, _height, frames = get_video_info(str(target))
        return {**_probe_media(Path(view["source"]), "video"), "path": str(target), "filename": target.name, "view": {"source": view["source"], "start_frame": view["start_frame"], "end_frame": view["end_frame"]},
                "frame_count": frames, "duration": round(frames / fps, 3) if fps else None, "duration_seconds": round(frames / fps, 3) if fps else None, "modified": target.stat().st_mtime}
    try:
        media_type = detect_media_type(str(target))
        if media_type == "3d":
            from shared.utils.mesh_outputs import asset_summary
            result = {"status": "done", "path": str(target), "filename": target.name, "size_bytes": target.stat().st_size, "file_type": "3d", **asset_summary(str(target)), "error": ""}
        elif media_type == "world":
            from shared.utils.world_outputs import world_summary
            result = {"status": "done", "path": str(target), "filename": target.name, "size_bytes": target.stat().st_size, "file_type": "world", **world_summary(str(target)), "error": ""}
        else:
            result = _query_image(target) if media_type == "image" else _probe_media(target, media_type) if media_type in {"video", "audio"} else {"status": "done", "path": str(target), "filename": target.name, "size_bytes": target.stat().st_size, "file_type": "file", "mime_type": mimetypes.guess_type(target.name)[0] or "application/octet-stream", "error": ""}
        result["modified"] = target.stat().st_mtime
        return result
    except (OSError, ValueError, ffmpeg.Error) as exc:
        return {"status": "error", "path": str(target), "size_bytes": target.stat().st_size, "error": str(exc) or "Unable to inspect file."}


ARCHIVE_ENTRIES_SHOWN = 50


def _query_archive(file_path: Path) -> dict[str, Any]:
    """A ZIP archive (an RGBA video's PNG frames, for example): its entries, the first ones by name."""
    with zipfile.ZipFile(file_path) as archive:
        names = archive.namelist()
    return {"status": "done", "path": str(file_path), "filename": file_path.name, "size_bytes": file_path.stat().st_size, "file_type": "archive",
            "entry_count": len(names), "entries": names[:ARCHIVE_ENTRIES_SHOWN], "error": ""}


def query_text_lines(path: str, start_line: int = 1) -> dict[str, Any]:
    """UTF-8 text from start_line (from 1) in whole lines, up to TEXT_MAX_CHARS characters; next_line when the file goes on."""
    file_path = _resolved_path(path)
    lines = file_path.read_text(encoding="utf-8").splitlines()
    shown, size = [], 0
    for line in lines[max(start_line, 1) - 1:]:
        if shown and size + len(line) + 1 > TEXT_MAX_CHARS:
            break
        shown.append(line)
        size += len(line) + 1
    end_line = max(start_line, 1) - 1 + len(shown)
    return {"status": "done", "path": str(file_path), "filename": file_path.name, "file_type": "text", "text": "\n".join(shown), "start_line": max(start_line, 1),
            "end_line": end_line, "line_count": len(lines), **({"next_line": end_line + 1} if end_line < len(lines) else {}), "error": ""}


def query_side_file(path: str, start_line: int = 1) -> dict[str, Any]:
    """A file saved beside a media item: a media file's details, an archive's entries, or text read by lines from start_line (any length)."""
    file_path = _resolved_path(path)
    if detect_media_type(str(file_path)) in {"image", "video", "audio"} or zipfile.is_zipfile(file_path):
        return query_file(path)
    try:
        return query_text_lines(path, start_line)
    except UnicodeError:  # a binary companion (a MIDI score, ...): what it is
        return {"status": "done", "path": str(file_path), "filename": file_path.name, "size_bytes": file_path.stat().st_size, "file_type": "file",
                "mime_type": mimetypes.guess_type(file_path.name)[0] or "application/octet-stream", "error": ""}


def query_file(path: str) -> dict[str, Any]:
    file_path = _resolved_path(path)
    if not file_path.is_file():
        return {"status": "error", "path": str(file_path), "error": "Path is not an existing file."}
    try:
        media_type = detect_media_type(str(file_path))
        if media_type == "image":
            return _query_image(file_path)
        if media_type in {"video", "audio"}:
            return _probe_media(file_path, media_type)
        if zipfile.is_zipfile(file_path):
            return _query_archive(file_path)
        with file_path.open("r", encoding="utf-8") as reader:
            text = reader.read(TEXT_MAX_CHARS + 1)
        if len(text) > TEXT_MAX_CHARS:
            return {"status": "error", "path": str(file_path), "size_bytes": file_path.stat().st_size, "error": f"Text file exceeds the {TEXT_MAX_CHARS:,}-character limit."}
        return {"status": "done", "path": str(file_path), "filename": file_path.name, "size_bytes": file_path.stat().st_size, "file_type": "text", "text": text, "character_count": len(text), "error": ""}
    except (OSError, UnicodeError, ValueError, ffmpeg.Error) as exc:
        return {"status": "error", "path": str(file_path), "size_bytes": file_path.stat().st_size, "error": str(exc) or "Unable to inspect file."}


def _encoding(value: str) -> str:
    encoding = str(value or "utf-8").strip().lower()
    if encoding not in _TEXT_ENCODINGS:
        raise ValueError("encoding must be utf-8 or utf-8-sig.")
    return encoding


def read_text(policy: FileAccessPolicy, path: str, start_line: int = 1, end_line: int | None = None, encoding: str = "utf-8-sig") -> dict[str, Any]:
    file_path = policy.require_read(path, file=True)
    start_line = int(start_line)
    end_line = None if end_line is None else int(end_line)
    if start_line < 1 or end_line is not None and end_line < start_line:
        raise ValueError("Use a 1-based start_line and an end_line greater than or equal to it.")
    chunks, character_count, last_line, next_line, truncated, eof = [], 0, start_line - 1, None, False, True
    with file_path.open("r", encoding=_encoding(encoding)) as reader:
        for line_no, line in enumerate(reader, 1):
            if line_no < start_line:
                continue
            if end_line is not None and line_no > end_line:
                next_line, eof = line_no, False
                break
            remaining = TEXT_MAX_CHARS - character_count
            if len(line) > remaining:
                had_content = bool(chunks)
                if remaining > 0:
                    chunks.append(line[:remaining])
                    last_line = line_no
                next_line = line_no if had_content else line_no + 1
                truncated, eof = True, False
                break
            chunks.append(line)
            character_count += len(line)
            last_line = line_no
    text = "".join(chunks)
    return {"status": "done", "path": str(file_path), "start_line": start_line, "end_line": last_line, "next_line": next_line, "truncated": truncated, "eof": eof, "character_count": len(text), "text": text, "error": ""}


def search_text(policy: FileAccessPolicy, path: str, query: str, pattern: str = "*", recursive: bool = False, regex: bool = False, case_sensitive: bool = False, limit: int = 100) -> dict[str, Any]:
    source = policy.require_read(path)
    query = str(query or "")
    if not query:
        raise ValueError("query is empty.")
    if len(query) > 500:
        raise ValueError("query exceeds 500 characters.")
    matcher = re.compile(query if regex else re.escape(query), 0 if case_sensitive else re.IGNORECASE)
    limit = max(1, min(int(limit), 500))
    files = [source] if source.is_file() else source.rglob(pattern or "*") if recursive else sorted(source.glob(pattern or "*"), key=lambda item: str(item).casefold())
    matches, skipped = [], 0
    for candidate in files:
        if not candidate.is_file() or not policy.can_read(candidate):
            continue
        if candidate.stat().st_size > SEARCH_MAX_FILE_BYTES:
            skipped += 1
            continue
        try:
            with candidate.open("r", encoding="utf-8-sig") as reader:
                for line_no, line in enumerate(reader, 1):
                    if matcher.search(line[:100000]) is None:
                        continue
                    matches.append({"path": str(candidate.resolve()), "line": line_no, "text": line.strip()[:500]})
                    if len(matches) >= limit:
                        return {"status": "done", "query": query, "matches": matches, "count": len(matches), "truncated": True, "skipped_files": skipped, "error": ""}
        except (OSError, UnicodeError):
            skipped += 1
    return {"status": "done", "query": query, "matches": matches, "count": len(matches), "truncated": False, "skipped_files": skipped, "error": ""}


def write_text(policy: FileAccessPolicy, path: str, text: str, mode: str = "create", encoding: str = "utf-8") -> dict[str, Any]:
    file_path = policy.require_write(path)
    mode = str(mode or "create").strip().lower()
    open_mode = {"create": "x", "overwrite": "w", "append": "a"}.get(mode)
    if open_mode is None:
        raise ValueError("mode must be create, overwrite, or append.")
    if policy.create_only(file_path):
        open_mode = "x"
    if open_mode == "x" and file_path.exists():
        raise FileExistsError(f"{path} already exists and cannot be replaced here: write a new file name." if policy.create_only(file_path) else f"{path} already exists: pass mode \"overwrite\" to replace it, or use append_text to add to it.")
    content = str(text)
    file_path.parent.mkdir(parents=True, exist_ok=True)  # inside the authorized write area, like the file itself
    with file_path.open(open_mode, encoding=_encoding(encoding), newline="") as writer:  # the text as given: no Windows newline translation
        writer.write(content)
    digest = hashlib.sha256()
    with file_path.open("rb") as reader:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            digest.update(chunk)
    headings, line_count = [], 0
    with file_path.open("r", encoding=_encoding(encoding)) as reader:
        for line in reader:
            line_count += 1
            if re.match(r"^#{1,6}\s+\S", line):
                headings.append(line.rstrip("\r\n"))
    lines_written = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
    return {"status": "done", "path": str(file_path), "filename": file_path.name, "mode": mode, "characters_written": len(content), "lines_written": lines_written, "line_count": line_count, "markdown_heading_count": len(headings), "first_markdown_heading": headings[0] if headings else None, "last_markdown_heading": headings[-1] if headings else None, "sha256": digest.hexdigest(), "size_bytes": file_path.stat().st_size, "error": ""}


def make_directory(policy: FileAccessPolicy, path: str) -> dict[str, Any]:
    directory = policy.require_write(path)
    existed = directory.is_dir()
    directory.mkdir(parents=True, exist_ok=True)
    return {"status": "done", "path": str(directory), "created": not existed, "error": ""}


def copy_file(policy: FileAccessPolicy, source: str, destination: str, overwrite: bool = False, source_authorized: bool = False) -> dict[str, Any]:
    source_path = _resolved_path(source) if source_authorized else policy.require_read(source, file=True)
    if not source_path.is_file():
        raise FileNotFoundError(f"Source is not an existing file: {source_path}")
    destination_path = policy.resolve_path(destination)
    if destination_path.is_dir():
        destination_path /= source_path.name
    destination_path = policy.require_write(destination_path)
    if not destination_path.parent.is_dir():
        raise FileNotFoundError(f"Destination directory does not exist: {destination_path.parent}")
    if destination_path.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {destination_path}")
    if policy.create_only(destination_path):
        with source_path.open("rb") as reader, destination_path.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
        shutil.copystat(source_path, destination_path)
    else:
        shutil.copy2(source_path, destination_path)
    return {"status": "done", "source": str(source_path), "path": str(destination_path), "filename": destination_path.name, "size_bytes": destination_path.stat().st_size, "error": ""}


def move_path(policy: FileAccessPolicy, source: str, destination: str) -> dict[str, Any]:
    source_path = policy.require_mutation(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source does not exist: {policy.virtualize_path(source_path)}")
    destination_path = policy.resolve_path(destination)
    if destination_path.is_dir():
        destination_path /= source_path.name
    destination_path = policy.require_write(destination_path)
    if source_path == destination_path:
        raise ValueError("Source and destination are the same path.")
    if destination_path.exists():
        raise FileExistsError(f"Destination already exists: {policy.virtualize_path(destination_path)}")
    if not destination_path.parent.is_dir():
        raise FileNotFoundError(f"Destination directory does not exist: {policy.virtualize_path(destination_path.parent)}")
    if source_path.is_dir() and _inside(destination_path, source_path):
        raise ValueError("A directory cannot be moved inside itself.")
    item_type = "directory" if source_path.is_dir() else "file"
    if policy.create_only(destination_path):
        if item_type == "directory":
            shutil.copytree(source_path, destination_path, symlinks=True, copy_function=lambda source, target: copy_file(policy, Path(source), Path(target))["path"])
            shutil.rmtree(source_path)
        else:
            copy_file(policy, source_path, destination_path)
            source_path.unlink()
    else:
        shutil.move(str(source_path), str(destination_path))
    return {"status": "done", "source": str(source_path), "path": str(destination_path), "filename": destination_path.name, "type": item_type, "size_bytes": destination_path.stat().st_size if destination_path.is_file() else None, "error": ""}


def delete_path(policy: FileAccessPolicy, path: str, recursive: bool = False) -> dict[str, Any]:
    target = policy.require_mutation(path)
    if not target.exists():
        raise FileNotFoundError(f"Path does not exist: {policy.virtualize_path(target)}")
    item_type = "directory" if target.is_dir() else "file"
    if target.is_dir():
        if any(target.iterdir()) and not recursive:
            raise OSError("Directory is not empty; set recursive=true to delete it and its contents.")
        if recursive:
            shutil.rmtree(target)
        else:
            target.rmdir()
    else:
        target.unlink()
    return {"status": "done", "path": str(target), "filename": target.name, "type": item_type, "recursive": bool(recursive and item_type == "directory"), "deleted": True, "error": ""}


def _available_zip_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 10000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Unable to choose an unused ZIP filename beside: {path}")


def _zip_name(sources: list[Path]) -> str:
    if len(sources) == 1:
        return f"{sources[0].stem if sources[0].is_file() else sources[0].name}.zip"
    parents = {source.parent for source in sources}
    return f"{next(iter(parents)).name or 'archive'}.zip" if len(parents) == 1 else f"archive_{time.strftime('%Y%m%d_%H%M%S')}.zip"


def zip_files(policy: FileAccessPolicy, sources: list[str], destination: str = "", authorized_sources: set[str] | None = None) -> dict[str, Any]:
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources must be a non-empty array.")
    authorized = {os.path.normcase(str(_resolved_path(path))) for path in authorized_sources or set()}
    source_paths = []
    for source in sources:
        physical = _resolved_path(source)
        source_paths.append(physical if os.path.normcase(str(physical)) in authorized else policy.require_read(source))
    if any(not source.exists() for source in source_paths):
        raise FileNotFoundError("One or more ZIP sources do not exist.")
    default_name = _zip_name(source_paths)
    if str(destination or "").strip():
        requested = policy.resolve_path(destination)
        target = requested / default_name if requested.is_dir() or requested.suffix.lower() != ".zip" else requested
    else:
        parents = {source.parent for source in source_paths}
        same_parent = next(iter(parents)) if len(parents) == 1 else None
        default_folder = same_parent if same_parent is not None and policy.can_write(same_parent / default_name) else policy.output_roots[0]
        target = default_folder / default_name
    target = policy.require_write(_available_zip_path(target))
    if not target.parent.is_dir():
        raise FileNotFoundError(f"ZIP destination directory does not exist: {target.parent}")
    temp_path = target.with_name(f".{target.name}.{uuid.uuid4().hex}.part")
    try:
        common_base = Path(os.path.commonpath([str(source.parent) for source in source_paths]))
    except ValueError:
        common_base = None
    names = set()
    try:
        with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
            for source in source_paths:
                files = (source,) if source.is_file() else (item for item in source.rglob("*") if item.is_file())
                for item in files:
                    item = _resolved_path(item) if os.path.normcase(str(_resolved_path(item))) in authorized else policy.require_read(item, file=True)
                    if item in {target, temp_path}:
                        continue
                    try:
                        archive_name = str(item.relative_to(common_base)).replace("\\", "/") if common_base is not None else item.name
                    except ValueError:
                        archive_name = item.name
                    base_name, suffix = archive_name.rsplit(".", 1) if "." in archive_name else (archive_name, "")
                    unique_name, index = archive_name, 2
                    while unique_name.casefold() in names:
                        unique_name = f"{base_name}_{index}{'.' + suffix if suffix else ''}"
                        index += 1
                    names.add(unique_name.casefold())
                    archive.write(item, unique_name)
        if policy.create_only(target):
            os.link(temp_path, target)
        else:
            os.replace(temp_path, target)
    finally:
        if temp_path.exists():
            temp_path.unlink()
    return {"status": "done", "output_file": str(target), "filename": target.name, "source_count": len(source_paths), "file_count": len(names), "size_bytes": target.stat().st_size, "error": ""}


def unzip_file(policy: FileAccessPolicy, source: str, destination: str = "", overwrite: bool = False, source_authorized: bool = False) -> dict[str, Any]:
    if not isinstance(overwrite, bool):
        raise ValueError("overwrite must be a boolean.")
    source_path = _resolved_path(source) if source_authorized else policy.require_read(source, file=True)
    if not source_path.is_file():
        raise FileNotFoundError(f"ZIP source is not an existing file: {source_path}")
    if not zipfile.is_zipfile(source_path):
        raise ValueError(f"Source is not a valid ZIP file: {policy.virtualize_path(source_path)}")
    if str(destination or "").strip():
        destination_path = policy.require_write(destination)
    else:
        beside_source = source_path.with_suffix("")
        destination_path = policy.require_write(beside_source if policy.can_write(beside_source) else policy.output_roots[0] / source_path.stem)
    if destination_path.exists() and not destination_path.is_dir():
        raise FileExistsError(f"ZIP destination is not a directory: {policy.virtualize_path(destination_path)}")

    plans = []
    overwritten_count = 0
    with zipfile.ZipFile(source_path, "r") as archive:
        seen = set()
        for member in archive.infolist():
            member_name = member.filename.replace("\\", "/")
            member_path = PurePosixPath(member_name)
            if not member_path.parts:
                continue
            if member_path.is_absolute() or ".." in member_path.parts or re.match(r"^[A-Za-z]:", member_name) or "\0" in member_name:
                raise ValueError(f"Unsafe ZIP member path: {member.filename}")
            if stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"ZIP symlink entries are not allowed: {member.filename}")
            if member.flag_bits & 1:
                raise ValueError(f"Encrypted ZIP entries are not supported: {member.filename}")
            target = destination_path.joinpath(*member_path.parts).resolve()
            if not _inside(target, destination_path):
                raise ValueError(f"ZIP member escapes the destination: {member.filename}")
            from shared.deepy.prime_filesystem import PrimeFileAccessPolicy
            if isinstance(policy, PrimeFileAccessPolicy):
                policy.require_write(target)
            if target == source_path:
                raise ValueError("A ZIP cannot overwrite its own source archive.")
            key = os.path.normcase(str(target))
            if key in seen:
                raise ValueError(f"ZIP contains duplicate destination paths: {member.filename}")
            seen.add(key)
            is_directory = member.is_dir() or member_name.endswith("/")
            if is_directory and target.exists() and not target.is_dir():
                raise FileExistsError(f"ZIP directory conflicts with an existing file: {policy.virtualize_path(target)}")
            if not is_directory and target.exists() and (target.is_dir() or not overwrite):
                raise FileExistsError(f"ZIP file destination already exists: {policy.virtualize_path(target)}")
            if not is_directory and target.is_file():
                overwritten_count += 1
            plans.append((member, target, is_directory))

        planned_files = {os.path.normcase(str(target)) for _member, target, is_directory in plans if not is_directory}
        for member, target, _is_directory in plans:
            for parent in target.parents:
                if parent == destination_path:
                    break
                if os.path.normcase(str(parent)) in planned_files:
                    raise ValueError(f"ZIP file/directory path conflict: {member.filename}")
                if parent.exists() and not parent.is_dir():
                    raise FileExistsError(f"ZIP parent path conflicts with an existing file: {policy.virtualize_path(parent)}")

        destination_path.mkdir(parents=True, exist_ok=True)
        for member, target, is_directory in plans:
            if is_directory:
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member, "r") as reader, target.open("xb" if policy.create_only(target) else "wb") as writer:
                shutil.copyfileobj(reader, writer)

    files = [member for member, _target, is_directory in plans if not is_directory]
    return {"status": "done", "source": str(source_path), "path": str(destination_path), "file_count": len(files), "size_bytes": sum(member.file_size for member in files), "overwritten_count": overwritten_count, "error": ""}


IO_ACTIONS = {
    "list": {"description": "List authorized roots or a page of directory entries. When has_more is true, repeat the same filters with offset set to next_offset.", "access": "read", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Directory; omit to list authorized roots."}, "pattern": {"type": "string", "default": "*", "description": "Filename glob."}, "media_type": {"type": "string", "enum": ["all", "image", "video", "audio", "txt", "other"], "default": "all", "description": "Implicit extension filter. Typed filters return files only; other excludes supported media and .txt files."}, "recursive": {"type": "boolean", "default": False}, "limit": {"type": "integer", "default": 200, "description": "Maximum entries to return; Deepy may shorten the page to fit its context-safe output budget."}, "offset": {"type": "integer", "default": 0, "description": "Entry offset; preserve the previous filters and use its next_offset to continue."}}}},
    "info": {"description": "Return file, directory, or media metadata.", "access": "always", "parameters": {"type": "object", "properties": {"source": {"type": "string", "description": "Authorized path or Gallery media id."}}, "required": ["source"]}},
    "read_text": {"description": "Read a 1-based line range from a UTF-8 text file. Use next_line to continue and eof to distinguish a completed range from the end of the file.", "access": "read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "start_line": {"type": "integer", "default": 1}, "end_line": {"type": "integer"}, "encoding": {"type": "string", "enum": ["utf-8", "utf-8-sig"], "default": "utf-8-sig"}}, "required": ["path"], "additionalProperties": False}},
    "search_text": {"description": "Search authorized UTF-8 files and return matching lines.", "access": "read", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "query": {"type": "string"}, "pattern": {"type": "string", "default": "*", "description": "Filename glob."}, "recursive": {"type": "boolean", "default": False}, "regex": {"type": "boolean", "default": False}, "case_sensitive": {"type": "boolean", "default": False}, "limit": {"type": "integer", "default": 100}}, "required": ["path", "query"]}},
    "write_text": {"description": "Create, overwrite, or append literal UTF-8 text already present in the request.", "access": "write", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "text": {"type": "string"}, "mode": {"type": "string", "enum": ["create", "overwrite", "append"], "default": "create"}, "encoding": {"type": "string", "enum": ["utf-8", "utf-8-sig"], "default": "utf-8"}}, "required": ["path", "text"]}},
    "mkdir": {"description": "Create an authorized directory.", "access": "write", "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}},
    "copy": {"description": "Copy an authorized file to a writable destination.", "access": "write", "parameters": {"type": "object", "properties": {"source": {"type": "string", "description": "Authorized path or Gallery media id."}, "destination": {"type": "string"}, "overwrite": {"type": "boolean", "default": False}}, "required": ["source", "destination"]}},
    "move": {"description": "Move an authorized writable file or directory to an unused destination.", "access": "write", "parameters": {"type": "object", "properties": {"source": {"type": "string"}, "destination": {"type": "string"}}, "required": ["source", "destination"]}},
    "delete": {"description": "Permanently delete an authorized writable file or directory; recursive is required for a non-empty directory.", "access": "write", "parameters": {"type": "object", "properties": {"path": {"type": "string"}, "recursive": {"type": "boolean", "default": False}}, "required": ["path"]}},
    "zip": {"description": "Create a persistent ZIP from authorized files or folders.", "access": "write", "parameters": {"type": "object", "properties": {"sources": {"type": "array", "items": {"type": "string"}, "description": "Authorized paths or Gallery media ids."}, "destination": {"type": "string", "description": "Optional folder/.zip path; plain paths use video outputs."}}, "required": ["sources"]}},
    "unzip": {"description": "Safely extract an authorized ZIP into a writable directory without replacing existing files by default.", "access": "write", "parameters": {"type": "object", "properties": {"source": {"type": "string", "description": "Authorized ZIP file path."}, "destination": {"type": "string", "description": "Optional extraction directory; defaults to a folder named after the ZIP beside it when writable, otherwise in video outputs."}, "overwrite": {"type": "boolean", "default": False}}, "required": ["source"]}},
    "download": {"description": "Create a session-long download link for an existing file.", "access": "always", "parameters": {"type": "object", "properties": {"source": {"type": "string", "description": "Authorized file path or Gallery media id."}}, "required": ["source"]}},
}


def available_io_actions(policy: FileAccessPolicy, downloads_enabled: bool = True) -> list[dict[str, Any]]:
    actions = []
    for name, definition in IO_ACTIONS.items():
        access = definition["access"]
        if name == "download" and not downloads_enabled or access == "read" and not policy.read_enabled or access == "write" and not policy.write_enabled:
            continue
        actions.append({"name": name, "description": definition["description"], "parameters": definition["parameters"]})
    return actions


__all__ = [
    "TEXT_MAX_CHARS", "FileAccessPolicy", "IO_ACTIONS", "available_io_actions", "build_file_access_policy", "copy_file", "delete_path", "file_info", "list_entries",
    "make_directory", "move_path", "query_file", "read_text", "search_text", "unzip_file", "write_text", "zip_files",
]
