# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Lightweight in-process API wrapper around WanGP generation."""

from __future__ import annotations

import contextlib
import copy
import fnmatch
import importlib
import io
import json
import math
import numpy as np
import os
import queue
import re
import sys
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

from PIL import Image
from shared.tinyvae.media import VideoPreview

from shared.utils.frame_scheduler import fit_video_length, normalize_output_frame_count
from shared.utils.gallery_sources import GALLERY_SOURCES, gallery_state_defaults
from shared.utils.process_locks import set_main_generation_running
from shared.utils.virtual_media import get_virtual_media_vsource, parse_virtual_media_path, replace_virtual_media_source

_RUNTIME_LOCK = threading.RLock()
_GENERATION_LOCK = threading.RLock()
_RUNTIME: "_WanGPRuntime | None" = None
_BANNER_PRINTED = False
_STATUS_STEP_PREFIX_RE = re.compile(r"^(?:prompt|sample|sliding window|window|chunk|task|step|phase|pass)\s+\d+\s*/\s*\d+\s*(?:,\s*)?", re.IGNORECASE)
_STATUS_INDEX_RE = re.compile(r"^\[\s*\d+\s*/\s*\d+\s*\]\s*")
_STATUS_TIME_ONLY_RE = re.compile(r"^[\d:.]+\s*[smh]?$", re.IGNORECASE)
_SECONDS_FRAME_COUNT_RE = re.compile(r"^\s*([0-9]+(?:\.[0-9]+)?)\s*s\s*$", re.IGNORECASE)


def _has_media_setting(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        return any(_has_media_setting(item) for item in value)
    return True


def _declared_choice_values(definition: Any) -> list[str]:
    if not isinstance(definition, dict):
        return []
    values = []
    for choice in definition.get("choices", ()) or ():
        value = choice[1] if isinstance(choice, (list, tuple)) and len(choice) > 1 else choice
        value = str(value or "")
        if value not in values:
            values.append(value)
    for value in definition.get("selection", ()) or ():
        value = str(value or "")
        if value not in values:
            values.append(value)
    return values


def _add_unique_flags(existing: Any, flags: str) -> str:
    value = str(existing or "")
    for flag in str(flags or ""):
        if flag not in value:
            value += flag
    return value


def apply_media_flag_defaults(settings: dict[str, Any], model_def: dict[str, Any]) -> dict[str, str]:
    inferred = {}
    image_prompt_type = str(settings.get("image_prompt_type", "") or "")
    allowed_image_flags = str(model_def.get("image_prompt_types_allowed", "") or "")
    if not any(flag in image_prompt_type for flag in "SVL"):
        if _has_media_setting(settings.get("video_source")) and "V" in allowed_image_flags:
            image_prompt_type = _add_unique_flags(image_prompt_type, "V")
        elif _has_media_setting(settings.get("image_start")) and "S" in allowed_image_flags:
            image_prompt_type = _add_unique_flags(image_prompt_type, "S")
    if _has_media_setting(settings.get("image_end")) and "E" in allowed_image_flags:
        image_prompt_type = _add_unique_flags(image_prompt_type, "E")
    if image_prompt_type != str(settings.get("image_prompt_type", "") or ""):
        settings["image_prompt_type"] = image_prompt_type
        inferred["image_prompt_type"] = image_prompt_type

    video_prompt_type = str(settings.get("video_prompt_type", "") or "")
    if _has_media_setting(settings.get("image_refs")) and "I" not in video_prompt_type:
        reference_mode = next((value for value in _declared_choice_values(model_def.get("image_ref_choices")) if "I" in value), "")
        if reference_mode:
            video_prompt_type = _add_unique_flags(video_prompt_type, reference_mode)
            settings["video_prompt_type"] = video_prompt_type
            inferred["video_prompt_type"] = video_prompt_type

    audio_prompt_type = str(settings.get("audio_prompt_type", "") or "")
    has_audio_guide = _has_media_setting(settings.get("audio_guide"))
    has_audio_guide2 = _has_media_setting(settings.get("audio_guide2"))
    has_audio_guide3 = _has_media_setting(settings.get("audio_guide3"))
    if has_audio_guide or has_audio_guide2 or has_audio_guide3:
        required_flag = "D" if has_audio_guide3 else "B" if has_audio_guide2 else "A"
        audio_mode = next((value for value in _declared_choice_values(model_def.get("audio_prompt_type_sources")) if required_flag in value), "")
        if audio_mode and required_flag not in audio_prompt_type:
            audio_prompt_type = _add_unique_flags(audio_prompt_type, audio_mode)
            settings["audio_prompt_type"] = audio_prompt_type
            inferred["audio_prompt_type"] = audio_prompt_type
    return inferred


def _model_frames_maximum(runtime_module: Any, model_type: str, model_def: dict[str, Any]) -> int:
    handler_defaults = {}
    runtime_module.get_model_handler(model_type).update_default_settings(model_def["architecture"], model_def, handler_defaults)
    sliding_window_size = handler_defaults.get("sliding_window_size")
    if sliding_window_size is not None:
        return sliding_window_size
    sliding_window_size = model_def.get("sliding_window_defaults", {}).get("window_default")
    if sliding_window_size is not None:
        return sliding_window_size
    return normalize_output_frame_count(97, model_def.get("frames_minimum", 5), model_def.get("frames_steps", 4), model_def.get("frames_offset", 1))


def apply_video_length_duration(settings: dict[str, Any], model_def: dict[str, Any]) -> dict[str, Any] | None:
    raw_value = settings.get("video_length")
    if not isinstance(raw_value, str) or not raw_value.strip().lower().endswith("s"):
        return None
    match = _SECONDS_FRAME_COUNT_RE.fullmatch(raw_value)
    if match is None:
        raise ValueError("video_length seconds must use a positive value such as '10s' or '5.5s'")
    seconds = float(match.group(1))
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("video_length seconds must be greater than zero")
    force_fps = settings.get("force_fps")
    fps = None
    if isinstance(force_fps, (int, float)) and not isinstance(force_fps, bool):
        fps = float(force_fps)
    elif isinstance(force_fps, str) and force_fps.strip():
        try:
            fps = float(force_fps)
        except ValueError:
            pass
    if fps is None:
        fps = float(model_def.get("fps", 16))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("A positive numeric force_fps or model fps is required when video_length uses seconds")
    requested_frames = int(round(seconds * fps))
    frames = fit_video_length(requested_frames, model_def.get("frames_minimum", 5), model_def.get("frames_steps", 4), model_def.get("frames_offset", 1))
    settings["video_length"] = frames
    return {"input": raw_value, "seconds": seconds, "fps": fps, "frames": frames}


def extract_status_phase_label(text: str | None) -> str:
    raw_text = str(text or "").strip()
    if len(raw_text) == 0:
        return ""
    parts = [part.strip() for part in raw_text.split("|") if len(part.strip()) > 0] or [raw_text]
    stripped_wrapper = False
    for part in parts:
        phase_text = part
        while True:
            cleaned = _STATUS_INDEX_RE.sub("", phase_text)
            cleaned = _STATUS_STEP_PREFIX_RE.sub("", cleaned)
            cleaned = cleaned.lstrip(" -:,")
            if cleaned == phase_text:
                break
            stripped_wrapper = True
            phase_text = cleaned.strip()
        if len(phase_text) > 0 and not _STATUS_TIME_ONLY_RE.fullmatch(phase_text):
            return phase_text
    return "" if stripped_wrapper else raw_text


@dataclass(frozen=True)
class StreamMessage:
    stream: str
    text: str


@dataclass(frozen=True)
class ProgressUpdate:
    phase: str
    status: str
    progress: int
    current_step: int | None
    total_steps: int | None
    raw_phase: str | None = None
    unit: str | None = None
    task_id: str | None = None
    task_index: int | None = None
    task_count: int | None = None
    window_index: int | None = None
    window_count: int | None = None


@dataclass(frozen=True)
class PreviewUpdate:
    image: Image.Image | None
    phase: str
    status: str
    progress: int
    current_step: int | None
    total_steps: int | None
    video: bytes | None = None


@dataclass(frozen=True)
class SessionEvent:
    kind: str
    data: Any = None
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class GeneratedArtifact:
    path: str | None
    media_type: str
    client_id: str = ""
    video_tensor_uint8: Any = None
    video_tensor_hdr: Any = None
    hdr: bool = False
    audio_tensor: Any = None
    audio_sampling_rate: int | None = None
    fps: float | None = None
    flashvsr_continue_cache: Any = None
    side_files: dict[str, bytes] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any], *, default_client_id: str = "") -> "GeneratedArtifact | None":
        if not isinstance(payload, dict):
            return None
        return cls(
            path=str(payload.get("path") or "") or None,
            media_type=str(payload.get("media_type") or "video"),
            client_id=str(payload.get("client_id") or default_client_id or "").strip(),
            video_tensor_uint8=payload.get("video_tensor_uint8"),
            video_tensor_hdr=payload.get("video_tensor_hdr"),
            hdr=bool(payload.get("hdr", False)),
            audio_tensor=payload.get("audio_tensor"),
            audio_sampling_rate=payload.get("audio_sampling_rate"),
            fps=payload.get("fps"),
            flashvsr_continue_cache=payload.get("flashvsr_continue_cache"),
            side_files=payload.get("side_files", {}),
        )


@dataclass(frozen=True)
class GenerationResult:
    success: bool
    generated_files: list[str]
    errors: list["GenerationError"]
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    artifacts: tuple[GeneratedArtifact, ...] = ()

    @property
    def cancelled(self) -> bool:
        return len(self.errors) > 0 and all(error.cancelled for error in self.errors)


@dataclass(frozen=True)
class GenerationError:
    message: str
    task_index: int | None = None
    task_id: Any = None
    stage: str | None = None

    def __str__(self) -> str:
        return self.message

    @property
    def cancelled(self) -> bool:
        stage = str(self.stage or "").strip().lower()
        if stage == "cancelled":
            return True
        return str(self.message or "").strip().lower() == "generation was cancelled"


def get_api_output_options(plugin_data: Any) -> tuple[bool, bool]:
    api_options = {} if not isinstance(plugin_data, dict) else plugin_data.get("api", {})
    if not isinstance(api_options, dict):
        return False, False
    return bool(api_options.get("return_video_uint8") or api_options.get("return_media")), bool(api_options.get("return_audio") or api_options.get("return_media"))


def _coerce_api_video_tensor_uint8(output_video_frames: Any) -> Any:
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and torch.is_tensor(output_video_frames):
        if output_video_frames.dtype == torch.uint8:
            return output_video_frames
        return output_video_frames.detach().cpu().float().clamp(-1, 1).add(1.0).mul(127.5).round().to(torch.uint8)
    if isinstance(output_video_frames, list) and len(output_video_frames) == 1 and torch is not None and torch.is_tensor(output_video_frames[0]):
        return _coerce_api_video_tensor_uint8(output_video_frames[0])
    if isinstance(output_video_frames, list) and torch is not None:
        tensors = [item for item in output_video_frames if torch.is_tensor(item)]
        if len(tensors) == len(output_video_frames) and tensors and all(item.dtype == torch.uint8 and item.ndim == 4 for item in tensors):
            return torch.cat(tensors, dim=1)
        if len(tensors) == len(output_video_frames) and tensors and all(item.dtype != torch.uint8 and item.ndim == 4 for item in tensors):
            return torch.cat([_coerce_api_video_tensor_uint8(item) for item in tensors], dim=1)
    return None


def _coerce_api_video_tensor_hdr(output_video_frames: Any) -> Any:
    try:
        import torch
    except Exception:
        torch = None
    if torch is not None and torch.is_tensor(output_video_frames):
        return output_video_frames if output_video_frames.dtype != torch.uint8 else None
    if isinstance(output_video_frames, list) and len(output_video_frames) == 1 and torch is not None and torch.is_tensor(output_video_frames[0]):
        return output_video_frames[0] if output_video_frames[0].dtype != torch.uint8 else None
    if isinstance(output_video_frames, list) and torch is not None:
        tensors = [item for item in output_video_frames if torch.is_tensor(item)]
        if len(tensors) == len(output_video_frames) and tensors and all(item.dtype != torch.uint8 and item.ndim == 4 for item in tensors):
            return torch.cat(tensors, dim=1)
    return None


def _coerce_api_audio_tensor(output_audio_data: Any) -> Any:
    return None if output_audio_data is None else np.asarray(output_audio_data, dtype=np.float32)


def build_api_output_artifact_payload(client_id: str, video_path: Any, media_type: str, output_video_frames: Any, output_audio_data: Any, output_audio_sampling_rate: Any, output_fps: Any, *, hdr: bool = False, flashvsr_continue_cache: Any = None, side_files: dict[str, bytes] | None = None) -> dict[str, Any] | None:
    client_id = str(client_id or "").strip()
    if len(client_id) == 0:
        return None
    output_path = str(video_path[0]) if isinstance(video_path, list) and len(video_path) > 0 else str(video_path or "")
    return {
        "client_id": client_id,
        "path": output_path,
        "media_type": str(media_type or "video"),
        "video_tensor_uint8": None if hdr else _coerce_api_video_tensor_uint8(output_video_frames),
        "video_tensor_hdr": _coerce_api_video_tensor_hdr(output_video_frames) if hdr else None,
        "hdr": bool(hdr),
        "audio_tensor": _coerce_api_audio_tensor(output_audio_data),
        "audio_sampling_rate": int(output_audio_sampling_rate) if output_audio_sampling_rate else None,
        "fps": float(output_fps) if output_fps else None,
        "flashvsr_continue_cache": flashvsr_continue_cache,
        "side_files": side_files if side_files is not None else {},
    }


def store_api_output_artifact(gen: dict[str, Any], client_id: str, video_path: Any, media_type: str, output_video_frames: Any, output_audio_data: Any, output_audio_sampling_rate: Any, output_fps: Any, *, hdr: bool = False, flashvsr_continue_cache: Any = None, side_files: dict[str, bytes] | None = None) -> bool:
    payload = build_api_output_artifact_payload(client_id, video_path, media_type, output_video_frames, output_audio_data, output_audio_sampling_rate, output_fps, hdr=hdr, flashvsr_continue_cache=flashvsr_continue_cache, side_files=side_files)
    if payload is None:
        return False
    gen.setdefault("api_output_artifacts", {})[payload["client_id"]] = payload
    return True


class SessionStream:
    def __init__(self) -> None:
        self._queue: queue.Queue[SessionEvent | object] = queue.Queue()
        self._closed = threading.Event()
        self._sentinel = object()

    def put(self, kind: str, data: Any = None) -> None:
        if self._closed.is_set():
            return
        self._queue.put(SessionEvent(kind=kind, data=data))

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        self._queue.put(self._sentinel)

    def clear(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        if self._closed.is_set():
            self._queue.put(self._sentinel)

    def get(self, timeout: float | None = None) -> SessionEvent | None:
        try:
            item = self._queue.get(timeout=timeout)
        except queue.Empty:
            return None
        if item is self._sentinel:
            return None
        return item

    def iter(self, timeout: float | None = None) -> Iterator[SessionEvent]:
        while True:
            event = self.get(timeout=timeout)
            if event is None:
                if self._closed.is_set():
                    break
                continue
            yield event

    @property
    def closed(self) -> bool:
        return self._closed.is_set()


class _OutputCapture(io.TextIOBase):
    def __init__(
        self,
        stream_name: str,
        emit_line,
        console: io.TextIOBase | None = None,
        *,
        console_isatty: bool = True,
    ) -> None:
        self._stream_name = stream_name
        self._emit_line = emit_line
        self._console = console
        self._console_isatty = bool(console_isatty)
        self._buffer = ""

    def writable(self) -> bool:
        return True

    @property
    def encoding(self) -> str:
        return str(getattr(self._console, "encoding", "utf-8"))

    def isatty(self) -> bool:
        return self._console_isatty

    def write(self, text: str) -> int:
        if not text:
            return 0
        if self._console is not None:
            self._console.write(text)
        self._buffer += text
        self._drain(False)
        return len(text)

    def flush(self) -> None:
        if self._console is not None:
            self._console.flush()
        self._drain(True)

    def _drain(self, flush_all: bool) -> None:
        while True:
            split_at = -1
            for delimiter in ("\r", "\n"):
                index = self._buffer.find(delimiter)
                if index >= 0 and (split_at < 0 or index < split_at):
                    split_at = index
            if split_at < 0:
                break
            line = self._buffer[:split_at]
            self._buffer = self._buffer[split_at + 1 :]
            if line:
                self._emit_line(self._stream_name, line)
        if flush_all and self._buffer:
            self._emit_line(self._stream_name, self._buffer)
            self._buffer = ""


@dataclass(frozen=True)
class _WanGPRuntime:
    module: Any
    root: Path
    config_path: Path
    cli_args: tuple[str, ...]


class SessionJob:
    def __init__(self, session: "WanGPSession") -> None:
        self._session = session
        self._callbacks: object | None = None
        self.events = SessionStream()
        self._done = threading.Event()
        self._cancel_requested = threading.Event()
        self._webui_submission_ready = threading.Event()
        self._webui_submission_or_done = threading.Event()
        self._thread: threading.Thread | None = None
        self._result: GenerationResult | None = None
        self._webui_manifest: list[dict[str, Any]] = []
        self._webui_client_ids: tuple[str, ...] = ()
        self._webui_load_queue_token = ""
        self._webui_owner_call_id = ""

    def _bind_thread(self, thread: threading.Thread) -> None:
        self._thread = thread

    def _bind_callbacks(self, callbacks: object | None) -> None:
        self._callbacks = callbacks

    def _set_result(self, result: GenerationResult) -> None:
        self._result = result
        self._done.set()
        self._webui_submission_or_done.set()

    def _set_webui_bridge(self, *, manifest: Sequence[dict[str, Any]], client_ids: Sequence[str], load_queue_token: str) -> None:
        self._webui_manifest = copy.deepcopy(list(manifest))
        self._webui_client_ids = tuple(str(client_id or "").strip() for client_id in client_ids if str(client_id or "").strip())
        self._webui_load_queue_token = str(load_queue_token or "").strip()

    def release_input_payload(self) -> None:
        self._webui_manifest = []

    def release_output_payload(self) -> None:
        self._result = None
        self.events.clear()

    def _mark_webui_submission_ready(self) -> None:
        self._webui_submission_ready.set()
        self._webui_submission_or_done.set()

    def _bind_webui_owner_call(self, call_id: str) -> None:
        self._webui_owner_call_id = str(call_id or "").strip()

    def cancel(self) -> None:
        self._cancel_requested.set()
        owner = getattr(self._session, "_gradio_session_proxy", None)
        capture = getattr(owner, "_capture_cancelled_job", None)
        if callable(capture):
            capture(self)

    def result(self, timeout: float | None = None) -> GenerationResult:
        if not self._done.wait(timeout=timeout):
            raise TimeoutError("WanGP session job timed out")
        return self._result or GenerationResult(
            success=False,
            generated_files=[],
            errors=[],
            total_tasks=0,
            successful_tasks=0,
            failed_tasks=0,
            artifacts=(),
        )

    def wait_for_webui_submission_or_completion(self, timeout: float | None = None) -> bool:
        self._webui_submission_or_done.wait(timeout=timeout)
        return self._webui_submission_ready.is_set()

    def join(self, timeout: float | None = None) -> GenerationResult:
        return self.result(timeout=timeout)

    @property
    def done(self) -> bool:
        return self._done.is_set()

    @property
    def cancel_requested(self) -> bool:
        return self._cancel_requested.is_set()

    @property
    def webui_manifest(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._webui_manifest)

    @property
    def webui_client_ids(self) -> tuple[str, ...]:
        return self._webui_client_ids

    @property
    def primary_client_id(self) -> str:
        return "" if not self._webui_client_ids else self._webui_client_ids[0]

    @property
    def webui_load_queue_token(self) -> str:
        return self._webui_load_queue_token

    @property
    def webui_submission_ready(self) -> bool:
        return self._webui_submission_ready.is_set()

    @property
    def webui_owner_call_id(self) -> str:
        return self._webui_owner_call_id


class WanGPSession:
    def __init__(
        self,
        *,
        root: str | os.PathLike[str] | None = None,
        config_path: str | os.PathLike[str] | None = None,
        output_dir: str | os.PathLike[str] | None = None,
        callbacks: object | None = None,
        cli_args: Sequence[str] = (),
        console_output: bool = True,
        console_isatty: bool = True,
        webui_state: dict[str, Any] | None = None,
        webui_append: bool = False,
    ) -> None:
        self._root = Path(root or Path(__file__).resolve().parents[1]).resolve()
        self._config_path = Path(config_path).resolve() if config_path is not None else (self._root / "wgp_config.json").resolve()
        self._output_dir = Path(output_dir).resolve() if output_dir is not None else None
        self._callbacks = callbacks
        self._cli_args = tuple(str(arg) for arg in cli_args)
        self._console_output = bool(console_output)
        self._console_isatty = bool(console_isatty)
        self._use_webui_queue = isinstance(webui_state, dict)
        # Like the UI's Add to Queue: jobs append in order and several may wait, instead of one priority job.
        self._webui_append = self._use_webui_queue and webui_append
        self._state = webui_state if isinstance(webui_state, dict) else self._create_headless_state()
        self._active_job: SessionJob | None = None
        self._job_lock = threading.Lock()
        self._attachment_keys: tuple[str, ...] | None = None

    def ensure_ready(self) -> "WanGPSession":
        self._ensure_runtime()
        return self

    def list_model_defs(self, *, family: str | Sequence[str] | None = None, base_model_type: str | Sequence[str] | None = None, finetune: bool | str | None = None, model_type: str | Sequence[str] | None = None, main_output: str | Sequence[str] | None = None, inputs: str | Sequence[str] | None = None, name: str | Sequence[str] | None = None, query: str | None = None, resolution: str | None = None, include_resolution_unknown: bool = False, limit: int | None = None, offset: int = 0) -> list[dict[str, Any]]:
        from shared.resolutions import assess_resolution, parse_resolution
        if resolution is not None:
            parse_resolution(resolution)
        runtime = self._ensure_runtime()
        with _pushd(runtime.root):
            records = _strip_model_def_callables(runtime.module.list_model_defs(family=family, base_model_type=base_model_type, finetune=finetune, model_type=model_type, main_output=main_output, inputs=inputs))
        name_patterns = [str(value).strip().casefold() for value in (name if isinstance(name, (list, tuple, set)) else [name]) if value is not None and str(value).strip()]
        query_text = str(query or "").strip().casefold()
        if name_patterns:
            records = [record for record in records if any(fnmatch.fnmatchcase(str(record.get("name", "") or "").casefold(), pattern) for pattern in name_patterns)]
        if query_text:
            records = [
                record for record in records
                if query_text in " ".join(
                    str(value or "").casefold()
                    for value in (
                        record.get("model_type"), record.get("name"), record.get("description"),
                        record.get("metadata", {}).get("family"), record.get("metadata", {}).get("family_label"), record.get("metadata", {}).get("base_model_type"),
                    )
                )
            ]
        if resolution is not None:
            selected = []
            for record in records:
                if not any(kind in ("image", "video") for kind in record["metadata"]["main_output"]):
                    continue
                assessment = assess_resolution(record, resolution)
                if assessment["status"] == "recommended" or include_resolution_unknown and assessment["status"] == "unknown":
                    record["metadata"]["resolution_assessment"] = assessment
                    selected.append(record)
            records = selected
        offset = max(0, int(offset or 0))
        if limit is None:
            return records[offset:]
        limit = max(1, min(int(limit), 500))
        return records[offset:offset + limit]

    def get_model_defs(self, **filters: Any) -> list[dict[str, Any]]:
        return self.list_model_defs(**filters)

    def list_model_metadata(self, include_availability: bool = False, include_selection: bool = False, **filters: Any) -> list[dict[str, Any]]:
        metadata_records = []
        for model_def in self.list_model_defs(**filters):
            metadata = copy.deepcopy(model_def.get("metadata", {}))
            metadata.setdefault("model_type", str(model_def.get("model_type") or ""))
            metadata["name"] = model_def.get("name", metadata.get("model_type", ""))
            if include_selection:
                metadata.update(self.get_model_selection_metadata(metadata["model_type"]))
            metadata_records.append(metadata)
        if include_availability:
            self._add_availability_to_metadata(metadata_records)
        return metadata_records

    def get_model_def(self, model_type: str) -> dict[str, Any] | None:
        runtime = self._ensure_runtime()
        with _pushd(runtime.root):
            model_def = runtime.module.get_model_def(model_type)
        if model_def is None:
            return None
        model_def = copy.deepcopy(model_def)
        model_def["model_type"] = str(model_type)
        return _strip_model_def_callables(model_def)

    def get_model_metadata(self, model_type: str, include_availability: bool = False) -> dict[str, Any] | None:
        model_def = self.get_model_def(model_type)
        if model_def is None:
            return None
        metadata = copy.deepcopy(model_def.get("metadata", {}))
        metadata.setdefault("model_type", str(model_type))
        metadata["name"] = model_def.get("name", metadata.get("model_type", ""))
        if include_availability:
            metadata["availability"] = self.get_model_availability(model_type)
        return metadata

    def get_default_settings(self, model_type: str) -> dict[str, Any]:
        if self.get_model_def(model_type) is None:
            raise ValueError(f"Unknown model_type: {model_type}")
        runtime = self._ensure_runtime()
        with _pushd(runtime.root):
            settings = copy.deepcopy(runtime.module.get_factory_settings(model_type))
        settings["model_type"] = str(model_type)
        return settings

    def get_model_selection_metadata(self, model_type: str) -> dict[str, Any]:
        runtime = self._ensure_runtime()
        with _pushd(runtime.root):
            model_def = runtime.module.get_model_def(model_type)
            selection = {key: copy.deepcopy(model_def[key]) for key in ("size", "specialities") if key in model_def}
            selection["accelerated"] = "native" if model_def.get("accelerated") == "native" else "profiles" if any(group == "accelerator_profiles" and paths for group, _, paths in runtime.module._get_builtin_lset_groups(model_type)) else "none"
            return selection

    def get_model_settings(self, model_type: str, setting_id: str | None = None, *, include_selection: bool = False) -> dict[str, Any]:
        runtime = self._ensure_runtime()
        with _pushd(runtime.root):
            model_def = runtime.module.get_model_def(model_type)
            if model_def is None:
                raise ValueError(f"Unknown model_type: {model_type}")
            entries = []
            kinds = {"accelerator_profiles": ("accelerator_profile", "accelerator profile"), "preset_settings": ("preset", "preset")}
            for group_id, _, paths in runtime.module._get_builtin_lset_groups(model_type):
                prefix, setting_type = kinds[group_id]
                entries += [{"id": f"{prefix}:{str(path).replace(chr(92), '/')}", "type": setting_type, "_path": runtime.module._builtin_lset_file_path(path)} for path in paths]
            lora_dir = Path(runtime.module.get_lora_dir(model_type))
            entries += [{"id": f"user_settings:{path.name}", "type": "user settings", "_path": str(path)} for path in sorted((*lora_dir.glob("*.json"), *lora_dir.glob("*.zip")), key=lambda path: path.name.casefold())]
            if include_selection:
                from shared.model_selection import prioritize_profiles
                entries = prioritize_profiles(entries)
            if setting_id is None:
                return {"model_type": str(model_type), "settings": [{key: value for key, value in entry.items() if key != "_path"} for entry in entries]}
            entry = next((entry for entry in entries if entry["id"] == setting_id), None)
            if entry is None:
                raise ValueError(f"Unknown setting_id for {model_type}: {setting_id}")
            path = Path(entry["_path"])
            if path.suffix.lower() == ".zip":
                with zipfile.ZipFile(path) as archive:
                    manifest = json.loads(archive.read("queue.json").decode("utf-8"))
                if not isinstance(manifest, list) or not manifest or not isinstance(manifest[0], dict):
                    raise ValueError(f"Invalid settings bundle: {path.name}")
                content = manifest[0].get("params", manifest[0])
            else:
                content = json.loads(path.read_text(encoding="utf-8"))
            if include_selection:
                content.pop("profile_priority", None)
            return {"model_type": str(model_type), "id": entry["id"], "type": entry["type"], "content": content}

    def merge_settings_with_defaults(self, settings: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(settings, dict):
            raise TypeError("settings must be a dictionary")
        model_type = str(settings.get("model_type", "") or "").strip()
        if not model_type:
            raise ValueError("settings must include model_type")
        runtime = self._ensure_runtime()
        with _pushd(runtime.root):
            if runtime.module.get_model_def(model_type) is None:
                raise ValueError(f"Unknown model_type: {model_type}")
            merged = copy.deepcopy(runtime.module.get_factory_settings(model_type))
            merged.update(copy.deepcopy(settings))
            merged.pop("profile_priority", None)
            runtime.module.clean_settings(model_type, merged)
            merged["settings_version"] = runtime.module.settings_version
        merged["model_type"] = model_type
        return merged

    def prepare_settings_for_export(self, settings: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(settings, dict):
            raise TypeError("settings must be a dictionary")
        model_type = str(settings.get("model_type", "") or "").strip()
        if not model_type:
            raise ValueError("settings must include model_type")
        runtime = self._ensure_runtime()
        with _pushd(runtime.root):
            if runtime.module.get_model_def(model_type) is None:
                raise ValueError(f"Unknown model_type: {model_type}")
            prepared = copy.deepcopy(settings)
            prepared["state"] = self._state
            return runtime.module.prepare_inputs_dict("metadata", prepared, model_type)

    def get_exported_default_settings(self, model_type: str) -> dict[str, Any]:
        return self.prepare_settings_for_export(self.get_default_settings(model_type))

    def get_lora_dir(self, model_type: str) -> str:
        """The LoRA folder of a model (raises for unknown models and models without LoRA support)."""
        runtime = self._ensure_runtime()
        with _pushd(runtime.root):
            if runtime.module.get_model_def(model_type) is None:
                raise ValueError(f"Unknown model_type: {model_type}")
            return str(runtime.module.get_lora_dir(model_type))

    def list_loras(self, model_type: str, name: str | Sequence[str] | None = None) -> dict[str, Any]:
        runtime = self._ensure_runtime()
        with _pushd(runtime.root):
            model_def = runtime.module.get_model_def(model_type)
            if model_def is None:
                raise ValueError(f"Unknown model_type: {model_type}")
            if model_def.get("no_lora", False):
                return {"model_type": str(model_type), "supported": False, "loras": [], "count": 0}
            lora_dir = runtime.module.get_lora_dir(model_type)
            loras = runtime.module.setup_loras(model_type, None, lora_dir, "", None)[0]
        name_patterns = [str(value).strip().casefold() for value in (name if isinstance(name, (list, tuple, set)) else [name]) if value is not None and str(value).strip()]
        if name_patterns:
            loras = [lora for lora in loras if any(fnmatch.fnmatchcase(str(lora).casefold(), pattern) for pattern in name_patterns)]
        return {"model_type": str(model_type), "supported": True, "loras": list(loras), "count": len(loras)}

    def get_model_availability(self, model_type: str) -> dict[str, Any]:
        if self.get_model_def(model_type) is None:
            raise ValueError(f"Unknown model_type: {model_type}")
        return self._get_model_availability_records([model_type])[0]

    def _get_model_availability_records(self, model_types: Sequence[str]) -> list[dict[str, Any]]:
        runtime = self._ensure_runtime()
        with _pushd(runtime.root):
            dropdown_deps = runtime.module._get_dropdown_deps()
            return [
                _model_availability_to_dict(model_type, runtime.module.model_dropdowns.get_model_download_status(dropdown_deps, model_type))
                for model_type in model_types
            ]

    def list_model_availability(self, **filters: Any) -> list[dict[str, Any]]:
        return self._get_model_availability_records([record["model_type"] for record in self.list_model_metadata(**filters)])

    def _add_availability_to_metadata(self, metadata_records: list[dict[str, Any]]) -> None:
        availability_records = self._get_model_availability_records([record["model_type"] for record in metadata_records])
        availability_by_type = {record["model_type"]: record for record in availability_records}
        for record in metadata_records:
            record["availability"] = availability_by_type[record["model_type"]]

    def get_model_schema(self, model_type: str) -> dict[str, Any] | None:
        model_def = self.get_model_def(model_type)
        if model_def is None:
            return None
        metadata = copy.deepcopy(model_def.get("metadata", {}))
        metadata.pop("setting_values", None)
        metadata.update({
            "name": str(model_def.get("name", model_type) or model_type),
            "model_type": str(model_type),
            "description": str(model_def.get("description", "") or ""),
            "fps": model_def.get("fps", None),
            "frames_minimum": model_def.get("frames_minimum", None),
            "frames_steps": model_def.get("frames_steps", None),
            "infos": str(model_def.get("infos", "") or ""),
            "prompt_infos": str(model_def.get("prompt_infos", "") or ""),
            "sliding_window": bool(model_def.get("sliding_window", False)),
            **({"prompt_optional": True} if model_def.get("prompt_optional") else {}),
        })
        if "video" in metadata.get("outputs", []):
            metadata["frames_maximum"] = _model_frames_maximum(self._ensure_runtime().module, model_type, model_def)
        return {"metadata": metadata}

    def submit(self, source: str | os.PathLike[str] | dict[str, Any] | list[dict[str, Any]], callbacks: object | None = None) -> SessionJob:
        tasks = self._normalize_source(source, caller_base_path=self._get_caller_base_path())
        return self._submit_tasks(tasks, callbacks=callbacks)

    def submit_task(self, settings: dict[str, Any], callbacks: object | None = None) -> SessionJob:
        caller_base_path = self._get_caller_base_path()
        task = self._normalize_task(settings, task_index=1)
        return self._submit_tasks([self._absolutize_task_paths(task, caller_base_path)], callbacks=callbacks)

    def submit_media_postprocessing(self, media_source: str | os.PathLike[str], *, temporal_upsampling: str = "", spatial_upsampling: str = "", spatial_postprocessing_parameters: dict[str, Any] | None = None, temporal_postprocessing_parameters: dict[str, Any] | None = None, postprocessing_reference_images: list[Any] | None = None, film_grain_intensity: float = 0, film_grain_saturation: float = 0.5, seed: int = -1, api_options: dict[str, Any] | None = None, return_media: bool = False, callbacks: object | None = None, **settings_overrides: Any) -> SessionJob:
        settings = build_media_postprocessing_settings(media_source, temporal_upsampling=temporal_upsampling, spatial_upsampling=spatial_upsampling, spatial_postprocessing_parameters=spatial_postprocessing_parameters, temporal_postprocessing_parameters=temporal_postprocessing_parameters, postprocessing_reference_images=postprocessing_reference_images, film_grain_intensity=film_grain_intensity, film_grain_saturation=film_grain_saturation, seed=seed, api_options=api_options, return_media=return_media, **settings_overrides)
        return self.submit_task(settings, callbacks=callbacks)

    def submit_audio_remux(self, video_source: str | os.PathLike[str], *, postprocess_audio: str, audio_source: str | os.PathLike[str] | None = None, postprocess_audio_prompt: str = "", postprocess_audio_neg_prompt: str = "", seed: int = -1, repeat_generation: int = 1, replace_voice_sample: str | os.PathLike[str] | None = None, replace_voice_sample2: str | os.PathLike[str] | None = None, api_options: dict[str, Any] | None = None, return_media: bool = False, callbacks: object | None = None, **settings_overrides: Any) -> SessionJob:
        settings = build_audio_remux_settings(video_source, postprocess_audio=postprocess_audio, audio_source=audio_source, postprocess_audio_prompt=postprocess_audio_prompt, postprocess_audio_neg_prompt=postprocess_audio_neg_prompt, seed=seed, repeat_generation=repeat_generation, replace_voice_sample=replace_voice_sample, replace_voice_sample2=replace_voice_sample2, api_options=api_options, return_media=return_media, **settings_overrides)
        return self.submit_task(settings, callbacks=callbacks)

    def submit_audio_postprocessing(self, audio_source: str | os.PathLike[str], *, postprocess_audio: str, replace_voice_sample: str | os.PathLike[str] | None = None, replace_voice_sample2: str | os.PathLike[str] | None = None, api_options: dict[str, Any] | None = None, return_media: bool = False, callbacks: object | None = None, **settings_overrides: Any) -> SessionJob:
        settings = build_audio_postprocessing_settings(audio_source, postprocess_audio=postprocess_audio, replace_voice_sample=replace_voice_sample, replace_voice_sample2=replace_voice_sample2, api_options=api_options, return_media=return_media, **settings_overrides)
        return self.submit_task(settings, callbacks=callbacks)

    def submit_manifest(self, settings_list: list[dict[str, Any]], callbacks: object | None = None) -> SessionJob:
        caller_base_path = self._get_caller_base_path()
        tasks = [
            self._absolutize_task_paths(self._normalize_task(settings, task_index=index + 1), caller_base_path)
            for index, settings in enumerate(settings_list)
        ]
        return self._submit_tasks(tasks, callbacks=callbacks)

    def run(self, source: str | os.PathLike[str] | dict[str, Any] | list[dict[str, Any]], callbacks: object | None = None) -> GenerationResult:
        return self.submit(source, callbacks=callbacks).result()

    def run_task(self, settings: dict[str, Any], callbacks: object | None = None) -> GenerationResult:
        return self.submit_task(settings, callbacks=callbacks).result()

    def run_manifest(self, settings_list: list[dict[str, Any]], callbacks: object | None = None) -> GenerationResult:
        return self.submit_manifest(settings_list, callbacks=callbacks).result()

    def run_media_postprocessing(self, media_source: str | os.PathLike[str], **kwargs: Any) -> GenerationResult:
        return self.submit_media_postprocessing(media_source, **kwargs).result()

    def run_audio_remux(self, video_source: str | os.PathLike[str], **kwargs: Any) -> GenerationResult:
        return self.submit_audio_remux(video_source, **kwargs).result()

    def run_audio_postprocessing(self, audio_source: str | os.PathLike[str], **kwargs: Any) -> GenerationResult:
        return self.submit_audio_postprocessing(audio_source, **kwargs).result()

    def close(self) -> None:
        if self._use_webui_queue:
            return
        runtime = self._ensure_runtime()
        with _GENERATION_LOCK, _pushd(runtime.root):
            runtime.module.release_model()

    def cancel(self) -> None:
        with self._job_lock:
            job = self._active_job
        if job is not None:
            job.cancel()

    def _queue_lock(self):
        return self._ensure_runtime().module.lock if self._use_webui_queue else self._job_lock

    def _workspace_gallery(self):
        from shared.utils.workspace_storage import host_for_gen
        from shared.utils.workspaces import load_gallery
        gen = self._state['gen']
        workspace_id = getattr(self, '_workspace_id', None)
        host = host_for_gen(gen)
        if workspace_id and host is not None and host.workspace_id != workspace_id:
            return load_gallery(host.workspaces.get(workspace_id), host._workspace_settings)
        return gen

    def list_queue(self) -> dict[str, Any]:
        """List all pending/running tasks in this session's generation queue, including UI tasks."""
        gen = self._state["gen"]
        with self._queue_lock():
            entries = []
            for index, task in enumerate(gen.get("queue", [])):
                queue_id = task.setdefault("_api_queue_id", uuid.uuid4().hex)
                params = self._get_task_settings(task)
                running = gen.get("api_active_queue_task") is task
                cancelling = running and bool(gen.get("abort", False))
                entries.append({
                    "queue_id": queue_id, "position": index + 1, "task_id": task.get("id"),
                    **{key: task[key] for key in ('workspace_id', 'mcp_agent_id') if key in task},
                    "client_id": str(params.get("client_id", "") or ""),
                    "model_type": str(params.get("model_type", "") or ""),
                    "prompt": str(params.get("prompt", "") or "")[:320],
                    "status": "cancelling" if cancelling else "running" if running else "queued",
                })
            running_count = sum(entry["status"] != "queued" for entry in entries)
            return {"tasks": entries, "total_count": len(entries), "queued_count": len(entries) - running_count, "running_count": running_count}

    def cancel_queue_task(self, queue_id: str) -> dict[str, Any]:
        """Cancel one task identified by list_queue(), leaving the rest of its batch queued."""
        if not isinstance(queue_id, str) or not queue_id.strip():
            raise ValueError("queue_id must be a non-empty ID returned by list_queue().")
        gen = self._state["gen"]
        with self._queue_lock():
            queue = gen.get("queue", [])
            index = next((i for i, task in enumerate(queue) if task.get("_api_queue_id") == queue_id), None)
            if index is None:
                raise KeyError(f"Unknown or finished queue_id: {queue_id}")
            task = queue[index]
            task["_api_queue_cancel_requested"] = True
            running = gen.get("api_active_queue_task") is task
            if running:
                self._request_cancel_unlocked(self._ensure_runtime().module)
            else:
                del queue[index]
                if self._use_webui_queue:
                    wgp = self._ensure_runtime().module
                    wgp.record_queue_error(self._state, [task], "Generation was cancelled", abort=True)
                    if "prompts_max" in gen:
                        gen["prompts_max"] = max(0, gen["prompts_max"] - 1)
                    # The UI autosave mirrors the live queue under the same lock.
                    wgp.global_queue_ref = queue[:]
            return {"queue_id": queue_id, "status": "cancelling" if running else "cancelled"}

    @property
    def active_job(self) -> SessionJob | None:
        with self._job_lock:
            return self._active_job

    @staticmethod
    def _create_headless_state() -> dict[str, Any]:
        return {
            "model_type": "",
            "edit_model_type": "",
            "gen": {
                "queue": [],
                "in_progress": False,
                **gallery_state_defaults(0),
                "prompt_no": 0,
                "prompts_max": 0,
                "repeat_no": 0,
                "total_generation": 1,
                "window_no": 0,
                "total_windows": 0,
                "progress_status": "",
                "process_status": "process:main",
                "api_output_artifacts": {},
            },
            "loras": [],
        }

    def _submit_tasks(self, tasks: list[dict[str, Any]], callbacks: object | None = None) -> SessionJob:
        with self._job_lock:
            if not self._webui_append and self._active_job is not None and not self._active_job.done:
                raise RuntimeError("WanGP session already has a generation in progress")
            job = SessionJob(self)
            self._bind_callbacks_to_job(job, callbacks)
            prepared_tasks = copy.deepcopy(tasks)
            from shared.utils.workspace_storage import capture_task
            for task in prepared_tasks:
                if getattr(self, '_workspace_id', None):
                    task['workspace_id'] = self._workspace_id
                if getattr(self, '_agent_context', None):
                    task['mcp_agent_id'] = self._agent_context
                capture_task(task, self._state)
            if self._output_dir is not None and any(task.get('workspace_id') for task in prepared_tasks):
                raise ValueError('An explicit output_dir cannot be combined with a workspace-bound queue. Use its original Workspaces Root.')
            client_ids = self._ensure_task_client_ids(prepared_tasks, priority=self._use_webui_queue and not self._webui_append)
            if self._use_webui_queue:
                prepared_tasks, manifest, load_queue_token = self._prepare_webui_bridge(prepared_tasks)
                job._set_webui_bridge(manifest=manifest, client_ids=client_ids, load_queue_token=load_queue_token)
            thread = threading.Thread(
                target=self._run_job,
                args=(job, prepared_tasks),
                daemon=True,
                name="wangp-session-job",
            )
            job._bind_thread(thread)
            self._active_job = job
            if not self._use_webui_queue:
                self._state["gen"]["queue"] = prepared_tasks[:]
            thread.start()
            return job

    def _bind_callbacks_to_job(self, job: SessionJob, callbacks: object | None = None) -> None:
        callback = self._callbacks if callbacks is None else callbacks
        job._bind_callbacks(callback)
        if callback is None:
            return
        binder = getattr(callback, "bind_job", None)
        if not callable(binder):
            return
        try:
            binder(session=self, job=job)
        except TypeError:
            binder(job)

    @staticmethod
    def _ensure_task_client_ids(tasks: list[dict[str, Any]], *, priority: bool = False) -> tuple[str, ...]:
        client_seed = uuid.uuid4().hex[:16]  # unique across sessions and jobs submitted within one clock tick
        client_ids: list[str] = []
        for index, task in enumerate(tasks, start=1):
            params = copy.deepcopy(WanGPSession._get_task_settings(task))
            client_id = str(params.get("client_id", "") or "").strip()
            if len(client_id) == 0:
                client_id = f"api_{client_seed}_{index}"
            params["client_id"] = client_id
            if priority:
                params["priority"] = True
            elif "priority" in params and not params["priority"]:
                params.pop("priority", None)
            task["params"] = params
            task.setdefault("plugin_data", {}).setdefault("api", {}).setdefault("return_side_files", True)
            client_ids.append(client_id)
        return tuple(client_ids)

    def _prepare_webui_bridge(self, tasks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
        manifest = []
        for index, task in enumerate(tasks, start=1):
            params = copy.deepcopy(self._get_task_settings(task))
            params["priority"] = not self._webui_append
            task["params"] = params
            manifest.append({
                "id": task.get("id", index),
                **({"workspace_id": task['workspace_id']} if task.get('workspace_id') else {}),
                **({"mcp_agent_id": task['mcp_agent_id']} if task.get('mcp_agent_id') else {}),
                "params": copy.deepcopy(params),
                "plugin_data": copy.deepcopy(task.get("plugin_data", {})),
            })
        return tasks, manifest, str(time.time_ns())

    def _run_job(self, job: SessionJob, tasks: list[dict[str, Any]]) -> None:
        if self._use_webui_queue:
            self._run_webui_job(job, tasks)
            return
        from shared.api_cli import run_cli_job

        run_cli_job(self, job, tasks)

    def _run_webui_job(self, job: SessionJob, tasks: list[dict[str, Any]]) -> None:
        from shared.api_webui import run_webui_job

        run_webui_job(self, job, tasks)

    def _build_progress_update(self, data: Any, *, include_state_fallback: bool = True, task_id: str | None = None, task_index: int | None = None, task_count: int | None = None) -> ProgressUpdate:
        current_step: int | None = None
        total_steps: int | None = None
        status = ""
        unit: str | None = None

        if isinstance(data, list) and data:
            head = data[0]
            if isinstance(head, tuple) and len(head) == 2:
                current_step = int(head[0])
                total_steps = int(head[1])
                status = str(data[1] if len(data) > 1 else "")
                if len(data) > 3:
                    unit = str(data[3])
            else:
                status = str(data[1] if len(data) > 1 else head)
        else:
            status = str(data or "")

        raw_phase = None
        if include_state_fallback:
            progress_phase = self._state["gen"].get("progress_phase")
            if isinstance(progress_phase, tuple) and progress_phase:
                raw_phase = extract_status_phase_label(progress_phase[0])
                if current_step is None and len(progress_phase) > 1 and "denoising" in raw_phase.lower():
                    try:
                        progress_step = int(progress_phase[1])
                    except (TypeError, ValueError):
                        progress_step = -1
                    try:
                        inference_steps = int(self._state["gen"].get("num_inference_steps") or 0)
                    except (TypeError, ValueError):
                        inference_steps = 0
                    if progress_step >= 0 and inference_steps > 0:
                        current_step = progress_step
                        total_steps = inference_steps
            if len(status) == 0:
                status = str(self._state["gen"].get("progress_status", "") or raw_phase or "")
        status_phase_label = extract_status_phase_label(status)
        if len(status_phase_label) > 0 and len(str(raw_phase or "").strip()) > 0 and current_step is None:
            normalized_status_phase = self._normalize_phase(status_phase_label)
            normalized_raw_phase = self._normalize_phase(raw_phase)
            if normalized_status_phase != normalized_raw_phase:
                raw_phase = None
        display_phase = raw_phase or status_phase_label
        phase = self._normalize_phase(display_phase or status)
        if not self._phase_supports_progress(phase):
            current_step = None
            total_steps = None
        progress = self._estimate_progress(phase, current_step, total_steps)
        return ProgressUpdate(
            phase=phase,
            status=status,
            progress=progress,
            current_step=current_step,
            total_steps=total_steps,
            raw_phase=display_phase or None,
            unit=unit,
            task_id=task_id,
            task_index=task_index,
            task_count=task_count,
            window_index=(self._state["gen"].get("window_no") or None) if phase != "loading_model" else None,
            window_count=(self._state["gen"].get("total_windows") or None) if phase != "loading_model" else None,
        )

    def _build_preview_update(self, wgp, tasks: list[dict[str, Any]], payload: Any) -> PreviewUpdate | None:
        progress = self._build_progress_update([0, self._state["gen"].get("progress_status", "")])
        model_type = ""
        queue_tasks = self._state["gen"].get("queue") or tasks
        if queue_tasks:
            model_type = str(self._get_task_settings(queue_tasks[0]).get("model_type", ""))
        image = wgp.generate_preview(model_type, payload) if model_type else None
        return PreviewUpdate(
            image=image.image if isinstance(image, VideoPreview) else image,
            video=image.video if isinstance(image, VideoPreview) else None,
            phase=progress.phase,
            status=progress.status,
            progress=progress.progress,
            current_step=progress.current_step,
            total_steps=progress.total_steps,
        )

    def _emit_stream(self, job: SessionJob, stream_name: str, line: str) -> None:
        message = StreamMessage(stream=stream_name, text=line)
        job.events.put("stream", message)
        self._emit_callback("on_stream", message, job=job)

    def _emit_callback(self, method_name: str, payload: Any, *, job: SessionJob | None = None) -> None:
        callback = self._callbacks if job is None or job._callbacks is None else job._callbacks
        if callback is None:
            return
        method = getattr(callback, method_name, None)
        if callable(method):
            method(payload)
        on_event = getattr(callback, "on_event", None)
        if callable(on_event):
            on_event(SessionEvent(kind=method_name.removeprefix("on_"), data=payload))

    def _configure_runtime(self, runtime: _WanGPRuntime) -> None:
        runtime.module.server_config["notification_sound_enabled"] = 0
        if self._output_dir is not None:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            runtime.module.server_config["save_path"] = str(self._output_dir)
            runtime.module.server_config["image_save_path"] = str(self._output_dir)
            runtime.module.server_config["audio_save_path"] = str(self._output_dir)
            runtime.module.server_config["mesh_save_path"] = str(self._output_dir)
            runtime.module.save_path = str(self._output_dir)
            runtime.module.image_save_path = str(self._output_dir)
            runtime.module.audio_save_path = str(self._output_dir)
            runtime.module.mesh_save_path = str(self._output_dir)
        for output_path in (
            runtime.module.save_path,
            runtime.module.image_save_path,
            runtime.module.audio_save_path,
            runtime.module.mesh_save_path,
        ):
            Path(output_path).mkdir(parents=True, exist_ok=True)

    def _prepare_state_for_run(self, tasks: list[dict[str, Any]]) -> None:
        gen = self._state["gen"]
        set_main_generation_running(self._state, True)
        gen["process_status"] = "process:main"
        gen["progress_status"] = ""
        gen["progress_phase"] = ("", -1)
        gen["abort"] = False
        gen["early_stop"] = False
        gen["early_stop_forwarded"] = False
        gen["preview"] = None
        gen["status"] = "Generating..."
        gen["in_progress"] = True
        gen.setdefault("api_output_artifacts", {})
        self._ensure_runtime().module.gen_in_progress = True

    def _reset_state_after_run(self) -> None:
        gen = self._state["gen"]
        with self._job_lock:
            gen["queue"] = []
            gen.pop("api_active_queue_task", None)
        set_main_generation_running(self._state, False)
        gen["process_status"] = "process:main"
        gen["progress_status"] = ""
        gen["progress_phase"] = ("", -1)
        gen["abort"] = False
        gen["early_stop"] = False
        gen["early_stop_forwarded"] = False
        gen.pop("in_progress", None)
        self._ensure_runtime().module.gen_in_progress = False

    def _gallery_counts(self) -> dict[str, int]:
        gen = self._state["gen"]
        return {source.id: len(gen[source.files_key]) for source in GALLERY_SOURCES}

    def _collect_outputs(self, base_counts: dict[str, int]) -> list[str]:
        gen = self._state["gen"]
        return [str(Path(path).resolve()) for source in GALLERY_SOURCES for path in gen[source.files_key][base_counts[source.id]:]]

    def _consume_output_artifact(self, client_id: str) -> GeneratedArtifact | None:
        gen = self._state["gen"]
        artifacts = gen.get("api_output_artifacts")
        if not isinstance(artifacts, dict):
            return None
        payload = artifacts.pop(str(client_id or "").strip(), None)
        return GeneratedArtifact.from_payload(payload, default_client_id=str(client_id or "").strip())

    def _peek_output_artifact(self, client_id: str) -> GeneratedArtifact | None:
        gen = self._state["gen"]
        artifacts = gen.get("api_output_artifacts")
        if not isinstance(artifacts, dict):
            return None
        payload = artifacts.get(str(client_id or "").strip(), None)
        return GeneratedArtifact.from_payload(payload, default_client_id=str(client_id or "").strip())

    def _consume_output_artifacts(self, tasks: Sequence[dict[str, Any]]) -> tuple[GeneratedArtifact, ...]:
        artifacts: list[GeneratedArtifact] = []
        for task in tasks:
            client_id = str(self._get_task_settings(task).get("client_id", "") or "").strip()
            if len(client_id) == 0:
                continue
            artifact = self._consume_output_artifact(client_id)
            if artifact is not None:
                artifacts.append(artifact)
        return tuple(artifacts)

    def _request_cancel_unlocked(self, wgp) -> None:
        gen = self._state["gen"]
        gen["resume"] = True
        gen["abort"] = True
        if wgp.wan_model is not None:
            wgp.wan_model._interrupt = True

    def _normalize_source(
        self,
        source: str | os.PathLike[str] | dict[str, Any] | list[dict[str, Any]],
        *,
        caller_base_path: Path,
    ) -> list[dict[str, Any]]:
        if isinstance(source, (str, os.PathLike)):
            return self._load_tasks_from_path(self._resolve_source_path(Path(source), caller_base_path), caller_base_path)
        if isinstance(source, list):
            return [
                self._absolutize_task_paths(self._normalize_task(task, task_index=index + 1), caller_base_path)
                for index, task in enumerate(source)
            ]
        if isinstance(source, dict):
            if isinstance(source.get("tasks"), list):
                tasks = source["tasks"]
                return [
                    self._absolutize_task_paths(self._normalize_task(task, task_index=index + 1), caller_base_path)
                    for index, task in enumerate(tasks)
                ]
            return [self._absolutize_task_paths(self._normalize_task(source, task_index=1), caller_base_path)]
        raise TypeError("WanGP session source must be a path, a settings dict, or a manifest list")

    def _normalize_task(self, task: dict[str, Any], *, task_index: int) -> dict[str, Any]:
        if not isinstance(task, dict):
            raise TypeError(f"Task {task_index} must be a dictionary")
        normalized = copy.deepcopy(task)
        if "settings" in normalized and "params" not in normalized:
            normalized["params"] = normalized.pop("settings")
        if "params" not in normalized:
            normalized = {"id": task_index, "params": normalized, "plugin_data": {}}
        normalized.setdefault("id", task_index)
        normalized.setdefault("plugin_data", {})
        normalized.setdefault("params", {})
        if not isinstance(normalized["plugin_data"], dict):
            normalized["plugin_data"] = {}
        settings = normalized["params"]
        if isinstance(settings, dict):
            api_options = settings.pop("_api", None)
            if isinstance(api_options, dict):
                normalized["plugin_data"]["api"] = copy.deepcopy(api_options)
            runtime = self._ensure_runtime()
            runtime_settings_version = getattr(runtime.module, "settings_version", None)
            if runtime_settings_version is not None:
                settings.setdefault("settings_version", runtime_settings_version)
            runtime.module.convert_retired_model_type(settings)  # a model type merged into others (3D stage models + post-processors)
            model_type = str(settings.get("model_type", "") or "").strip()
            model_def = runtime.module.get_model_def(model_type) if model_type else None
            if isinstance(model_def, dict):
                duration_conversion = apply_video_length_duration(settings, model_def)
                if duration_conversion is not None:
                    print(f"WanGP API converted video_length={duration_conversion['input']!r} to {duration_conversion['frames']} frames at {duration_conversion['fps']:g} fps for {model_type}")
                inferred_flags = apply_media_flag_defaults(settings, model_def)
                if inferred_flags:
                    print(f"WanGP API inferred media flags for {model_type}: {inferred_flags}")
            self._normalize_settings_values(settings)
            normalized.setdefault("prompt", settings.get("prompt", ""))
            normalized.setdefault("length", settings.get("video_length"))
            normalized.setdefault("steps", settings.get("num_inference_steps"))
            normalized.setdefault("repeats", settings.get("repeat_generation", 1))
        return normalized

    @staticmethod
    def _normalize_settings_values(settings: dict[str, Any]) -> None:
        video_length = settings.get("video_length")
        if isinstance(video_length, str):
            try:
                settings["video_length"] = int(video_length.strip())
            except ValueError as exc:
                raise ValueError("video_length must be an integer frame count or a duration such as '10s'") from exc
        force_fps = settings.get("force_fps")
        if isinstance(force_fps, (int, float)) and not isinstance(force_fps, bool):
            if isinstance(force_fps, float) and not force_fps.is_integer():
                settings["force_fps"] = str(force_fps)
            else:
                settings["force_fps"] = str(int(force_fps))

    @staticmethod
    def _get_task_settings(task: dict[str, Any]) -> dict[str, Any]:
        settings = task.get("params")
        if isinstance(settings, dict):
            return settings
        settings = task.get("settings")
        if isinstance(settings, dict):
            return settings
        return {}

    def _load_tasks_from_path(self, path: Path, caller_base_path: Path) -> list[dict[str, Any]]:
        runtime = self._ensure_runtime()
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() == ".json":
            return self._load_settings_json(path, caller_base_path)
        with _pushd(runtime.root):
            tasks, error = runtime.module._parse_queue_zip(str(path), self._state)
        if error:
            raise RuntimeError(error)
        return [self._normalize_task(task, task_index=index + 1) for index, task in enumerate(tasks)]

    def _load_settings_json(self, path: Path, caller_base_path: Path) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        if isinstance(payload, list):
            raw_tasks = payload
        elif isinstance(payload, dict) and isinstance(payload.get("tasks"), list):
            raw_tasks = payload["tasks"]
        elif isinstance(payload, dict):
            raw_tasks = [payload]
        else:
            raise RuntimeError("Settings file must contain a JSON object or a list of tasks")

        tasks = [self._normalize_task(task, task_index=index + 1) for index, task in enumerate(raw_tasks)]
        return [self._absolutize_task_paths(task, caller_base_path) for task in tasks]

    @staticmethod
    def _get_caller_base_path() -> Path:
        return Path.cwd().resolve()

    @staticmethod
    def _resolve_source_path(path: Path, caller_base_path: Path) -> Path:
        if path.is_absolute():
            return path.resolve()
        return (caller_base_path / path).resolve()

    def _absolutize_task_paths(self, task: dict[str, Any], caller_base_path: Path) -> dict[str, Any]:
        normalized = copy.deepcopy(task)
        settings = normalized.get("params")
        if not isinstance(settings, dict):
            return normalized
        for key in self._get_attachment_keys():
            if key not in settings:
                continue
            settings[key] = self._absolutize_setting_path(settings[key], caller_base_path)
        return normalized

    def _get_attachment_keys(self) -> tuple[str, ...]:
        if self._attachment_keys is None:
            runtime = self._ensure_runtime()
            keys = getattr(runtime.module, "ATTACHMENT_KEYS", ())
            self._attachment_keys = tuple(str(key) for key in keys)
        return self._attachment_keys

    def _absolutize_setting_path(self, value: Any, caller_base_path: Path) -> Any:
        if isinstance(value, list):
            return [self._absolutize_setting_path(item, caller_base_path) for item in value]
        if isinstance(value, os.PathLike):
            value = os.fspath(value)
        if not isinstance(value, str) or not value.strip():
            return value
        spec = parse_virtual_media_path(value)
        if spec is not None and get_virtual_media_vsource(spec) is not None:
            return value
        path = Path(spec.source_path if spec is not None else value)
        if path.is_absolute():
            resolved = str(path.resolve())
        else:
            resolved = str((caller_base_path / path).resolve())
        return replace_virtual_media_source(value, resolved) if spec is not None else resolved

    @staticmethod
    def _make_generation_error(
        error: Any,
        *,
        task_index: int | None = None,
        task_id: Any = None,
        stage: str | None = None,
    ) -> GenerationError:
        if isinstance(error, GenerationError):
            return error
        if isinstance(error, BaseException):
            message = str(error) or error.__class__.__name__
        else:
            message = str(error)
        return GenerationError(message=message, task_index=task_index, task_id=task_id, stage=stage)

    def _ensure_runtime(self) -> _WanGPRuntime:
        global _RUNTIME
        with _RUNTIME_LOCK:
            if _RUNTIME is not None:
                if _RUNTIME.root != self._root or _RUNTIME.config_path != self._config_path or _RUNTIME.cli_args != self._cli_args:
                    raise RuntimeError("WanGP runtime already loaded with different root/config/cli args")
                return _RUNTIME

            argv = ["wgp.py", *self._cli_args]
            default_config_path = (self._root / "wgp_config.json").resolve()
            if self._config_path.name != "wgp_config.json":
                raise ValueError("config_path must point to a file named 'wgp_config.json'")
            if self._config_path != default_config_path:
                self._config_path.parent.mkdir(parents=True, exist_ok=True)
                if "--config" not in argv:
                    argv.extend(["--config", str(self._config_path.parent)])

            if str(self._root) not in sys.path:
                sys.path.insert(0, str(self._root))

            with _pushd(self._root), _temporary_argv(argv):
                module = importlib.import_module("wgp")
                module_root = Path(module.__file__).resolve().parent
                if module_root != self._root:
                    raise RuntimeError(f"WanGP module already loaded from {module_root}, expected {self._root}")
                if not hasattr(module, "app"):
                    module.app = module.WAN2GPApplication()
                module.download_ffmpeg()

            _RUNTIME = _WanGPRuntime(
                module=module,
                root=self._root,
                config_path=self._config_path,
                cli_args=self._cli_args,
            )
            _print_banner_once(module, enabled=not self._use_webui_queue and self._console_output)
            return _RUNTIME

    @staticmethod
    def _normalize_phase(text: str | None) -> str:
        lowered = extract_status_phase_label(text).lower()
        if "denoising first pass" in lowered or "denoising 1st pass" in lowered:
            return "inference_stage_1"
        if "denoising second pass" in lowered or "denoising 2nd pass" in lowered:
            return "inference_stage_2"
        if "denoising third pass" in lowered or "denoising 3rd pass" in lowered:
            return "inference_stage_3"
        if "loading model" in lowered or lowered.startswith("loading"):
            return "loading_model"
        if "enhancing prompt" in lowered or "encoding prompt" in lowered or "encoding" in lowered:
            return "encoding_text"
        if "vae decoding" in lowered or "decoding" in lowered:
            return "decoding"
        if "saved" in lowered or "completed" in lowered or "output" in lowered:
            return "downloading_output"
        if "cancel" in lowered or "abort" in lowered:
            return "cancelled"
        return "inference"

    @staticmethod
    def _phase_supports_progress(phase: str | None) -> bool:
        return str(phase or "") in {"encoding_text", "inference", "inference_stage_1", "inference_stage_2", "inference_stage_3", "decoding"}

    @staticmethod
    def _estimate_progress(phase: str, current_step: int | None, total_steps: int | None) -> int:
        if total_steps is None or total_steps <= 0 or current_step is None:
            if phase == "loading_model":
                return 10
            if phase == "encoding_text":
                return 18
            if phase == "inference_stage_1":
                return 25
            if phase == "inference_stage_2":
                return 70
            if phase == "inference_stage_3":
                return 80
            if phase == "decoding":
                return 90
            if phase == "downloading_output":
                return 95
            if phase == "cancelled":
                return 0
            return 15
        ratio = max(0.0, min(1.0, current_step / total_steps))
        if phase == "loading_model":
            return min(15, 5 + int(ratio * 10))
        if phase == "encoding_text":
            return min(22, 12 + int(ratio * 10))
        if phase == "inference_stage_1":
            return min(68, 20 + int(ratio * 48))
        if phase == "inference_stage_2":
            return min(88, 68 + int(ratio * 20))
        if phase == "inference_stage_3":
            return min(89, 80 + int(ratio * 9))
        if phase == "decoding":
            return min(95, 85 + int(ratio * 10))
        if phase == "downloading_output":
            return min(98, 92 + int(ratio * 6))
        if phase == "cancelled":
            return 0
        return min(90, 20 + int(ratio * 65))


def build_media_postprocessing_settings(media_source: str | os.PathLike[str], *, temporal_upsampling: str = "", spatial_upsampling: str = "", spatial_postprocessing_parameters: dict[str, Any] | None = None, temporal_postprocessing_parameters: dict[str, Any] | None = None, postprocessing_reference_images: list[Any] | None = None, film_grain_intensity: float = 0, film_grain_saturation: float = 0.5, seed: int = -1, api_options: dict[str, Any] | None = None, return_media: bool = False, **settings_overrides: Any) -> dict[str, Any]:
    settings = {
        "mode": "edit_postprocessing",
        "prompt": "Media postprocessing",
        "image_mode": 0,
        "video_source": os.fspath(media_source),
        "temporal_upsampling": temporal_upsampling or "",
        "spatial_upsampling": spatial_upsampling or "",
        "spatial_postprocessing_parameters": dict(spatial_postprocessing_parameters or {}),
        "temporal_postprocessing_parameters": dict(temporal_postprocessing_parameters or {}),
        "postprocessing_reference_images": list(postprocessing_reference_images or []),
        "film_grain_intensity": film_grain_intensity,
        "film_grain_saturation": film_grain_saturation,
        "postprocess_audio": "",
        "repeat_generation": 1,
        "batch_size": 1,
        "seed": int(seed),
    }
    _apply_edit_settings_overrides(settings, settings_overrides, api_options, return_media)
    return settings


def build_audio_remux_settings(video_source: str | os.PathLike[str], *, postprocess_audio: str, audio_source: str | os.PathLike[str] | None = None, postprocess_audio_prompt: str = "", postprocess_audio_neg_prompt: str = "", seed: int = -1, repeat_generation: int = 1, replace_voice_sample: str | os.PathLike[str] | None = None, replace_voice_sample2: str | os.PathLike[str] | None = None, api_options: dict[str, Any] | None = None, return_media: bool = False, **settings_overrides: Any) -> dict[str, Any]:
    settings = {
        "mode": "edit_remux",
        "prompt": "Audio remuxing",
        "image_mode": 0,
        "video_source": os.fspath(video_source),
        "postprocess_audio": postprocess_audio or "",
        "postprocess_audio_prompt": postprocess_audio_prompt or "",
        "postprocess_audio_neg_prompt": postprocess_audio_neg_prompt or "",
        "seed": int(seed),
        "repeat_generation": int(repeat_generation),
        "audio_source": None if audio_source is None else os.fspath(audio_source),
        "replace_voice_sample": None if replace_voice_sample is None else os.fspath(replace_voice_sample),
        "replace_voice_sample2": None if replace_voice_sample2 is None else os.fspath(replace_voice_sample2),
        "temporal_upsampling": "",
        "spatial_upsampling": "",
        "film_grain_intensity": 0,
        "film_grain_saturation": 0.5,
        "batch_size": 1,
    }
    _apply_edit_settings_overrides(settings, settings_overrides, api_options, return_media)
    return settings


def build_audio_postprocessing_settings(audio_source: str | os.PathLike[str], *, postprocess_audio: str, replace_voice_sample: str | os.PathLike[str] | None = None, replace_voice_sample2: str | os.PathLike[str] | None = None, api_options: dict[str, Any] | None = None, return_media: bool = False, **settings_overrides: Any) -> dict[str, Any]:
    settings = {
        "mode": "edit_audio",
        "prompt": "Audio postprocessing",
        "image_mode": 0,
        "audio_source": os.fspath(audio_source),
        "postprocess_audio": postprocess_audio or "",
        "replace_voice_sample": None if replace_voice_sample is None else os.fspath(replace_voice_sample),
        "replace_voice_sample2": None if replace_voice_sample2 is None else os.fspath(replace_voice_sample2),
        "repeat_generation": 1,
        "batch_size": 1,
    }
    _apply_edit_settings_overrides(settings, settings_overrides, api_options, return_media)
    return settings


def _apply_edit_settings_overrides(settings: dict[str, Any], settings_overrides: dict[str, Any], api_options: dict[str, Any] | None, return_media: bool) -> None:
    override_api_options = settings_overrides.pop("_api", None)
    settings.update(settings_overrides)
    settings.pop("model_type", None)
    settings.pop("base_model_type", None)
    options = copy.deepcopy(api_options if api_options is not None else override_api_options)
    if not isinstance(options, dict):
        options = {}
    if return_media:
        options["return_media"] = True
    if options:
        settings["_api"] = options


def _strip_model_def_callables(value: Any) -> Any:
    if callable(value):
        return None
    if isinstance(value, dict):
        return {key: _strip_model_def_callables(item) for key, item in value.items() if not callable(item)}
    if isinstance(value, (list, tuple)):
        return [_strip_model_def_callables(item) for item in value if not callable(item)]
    return value


def _model_availability_to_dict(model_type: str, status: int) -> dict[str, Any]:
    from shared import model_dropdowns

    if status == model_dropdowns.MODEL_FILE_STATUS_EXPECTED:
        label, indicator = "available", "blue_square"
    elif status == model_dropdowns.MODEL_FILE_STATUS_PARTIAL:
        label, indicator = "partial", "yellow_square"
    else:
        label, indicator = "missing", "black_square"
    return {
        "model_type": str(model_type),
        "status": label,
        "status_code": int(status),
        "indicator": indicator,
        "available": status == 2,
    }


def init(
    *,
    root: str | os.PathLike[str] | None = None,
    config_path: str | os.PathLike[str] | None = None,
    output_dir: str | os.PathLike[str] | None = None,
    callbacks: object | None = None,
    cli_args: Sequence[str] = (),
    console_output: bool = True,
    console_isatty: bool = True,
    webui_state: dict[str, Any] | None = None,
) -> WanGPSession:
    """Create and eagerly initialize a reusable WanGP session."""

    return WanGPSession(
        root=root,
        config_path=config_path,
        output_dir=output_dir,
        callbacks=callbacks,
        cli_args=cli_args,
        console_output=console_output,
        console_isatty=console_isatty,
        webui_state=webui_state,
    ).ensure_ready()

def create_gradio_webui_session(plugin) -> Any:
    from shared.api_webui import create_gradio_webui_session as _create_gradio_webui_session

    return _create_gradio_webui_session(plugin, init_fn=init)


def create_gradio_progress_callbacks(progress) -> Any:
    from shared.api_webui import create_gradio_progress_callbacks as _create_gradio_progress_callbacks

    return _create_gradio_progress_callbacks(progress)


@contextlib.contextmanager
def _pushd(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


@contextlib.contextmanager
def _temporary_argv(argv: Sequence[str]) -> Iterator[None]:
    previous = list(sys.argv)
    sys.argv = list(argv)
    try:
        yield
    finally:
        sys.argv = previous


def _print_banner_once(module, *, enabled: bool = True) -> None:
    global _BANNER_PRINTED
    if not enabled:
        return
    if _BANNER_PRINTED:
        return
    _BANNER_PRINTED = True
    banner = f"Powered by WanGP v{module.WanGP_version} - a DeepBeepMeep Production\n"
    console = sys.__stdout__ if sys.__stdout__ is not None else sys.stdout
    if console is not None:
        console.write(banner)
        console.flush()
