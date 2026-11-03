# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import contextlib
import copy
import asyncio
import functools
import inspect
import os
import threading
import time
from pathlib import Path
from typing import Any, Sequence

from shared.api import GeneratedArtifact, GenerationError, GenerationResult, PreviewUpdate, SessionJob, WanGPSession
from shared.tinyvae.media import VideoPreview
from shared.utils.generation_state import set_generation_state

# Tasks that plugins submit through their API session join WanGP's queue on the server, as Deepy's, MCP's and the World explorer's do.
# False restores the older hand-off, where the plugin's browser page passes each submission to the Media Generator's queue events.
PLUGIN_QUEUE_ON_SERVER = True

_NO_YIELDED_RESULT = object()
_GRADIO_LOG_PATCH_LOCK = threading.Lock()
_ORIGINAL_GRADIO_LOG_MESSAGE = None
_WRAPPED_LOG_LOCAL = threading.local()
_INLINE_QUEUE_LOCK = threading.Lock()  # the inline queue holds one submission until WanGP loads it


def _buffered_gradio_log_message(message: str, title: str, level: str = "info", duration: float | None = 10, visible: bool = True):
    from gradio.context import LocalContext

    blocks = LocalContext.blocks.get()
    event_id = LocalContext.event_id.get()
    if blocks is not None and event_id is not None:
        return _ORIGINAL_GRADIO_LOG_MESSAGE(message, title=title, level=level, duration=duration, visible=visible)
    owner = getattr(_WRAPPED_LOG_LOCAL, "owner", None)
    call_id = str(getattr(_WRAPPED_LOG_LOCAL, "call_id", "") or "").strip()
    if owner is not None and len(call_id) > 0:
        state = owner._get_wrapped_call(call_id)
        if state is not None:
            state.queue_log_message(message=message, title=title, level=level, duration=duration, visible=visible)
            return
    return _ORIGINAL_GRADIO_LOG_MESSAGE(message, title=title, level=level, duration=duration, visible=visible)


def _ensure_gradio_log_message_patch() -> None:
    global _ORIGINAL_GRADIO_LOG_MESSAGE
    with _GRADIO_LOG_PATCH_LOCK:
        if _ORIGINAL_GRADIO_LOG_MESSAGE is not None:
            return
        import gradio.helpers as gr_helpers

        _ORIGINAL_GRADIO_LOG_MESSAGE = gr_helpers.log_message
        gr_helpers.log_message = _buffered_gradio_log_message


def _server_queue(state):
    """The WanGP service that admits a plugin's tasks itself, or None when its browser page hands them to the queue."""
    if not PLUGIN_QUEUE_ON_SERVER:
        return None
    from shared.deepy.hybrid import service_for
    return service_for(state)


def _normalize_queue_request(request):
    if request is None:
        return None
    try:
        from gradio import route_utils

        route_utils.get_api_call_path(request)
        return request
    except Exception:
        pass
    try:
        from fastapi import Request as FastAPIRequest
    except Exception:
        return request
    scope = dict(getattr(request, "scope", {}) or {})
    if scope.get("type") != "http":
        return request
    queue_path = f"{route_utils.API_PREFIX}/queue/join"
    scope["path"] = queue_path
    scope["raw_path"] = queue_path.encode("utf-8")
    scope["query_string"] = b""
    try:
        return FastAPIRequest(scope, request.receive)
    except Exception:
        return request


class GradioProgressCallbacks:
    def __init__(self, progress) -> None:
        self._progress = progress
        self._ratio = 0.0

    def on_status(self, status) -> None:
        status = str(status or "").strip()
        if status:
            self._progress(self._ratio, desc=status)

    def on_progress(self, update) -> None:
        self._ratio = max(0.0, min(1.0, float(getattr(update, "progress", 0)) / 100.0))
        self._progress(self._ratio, desc=str(getattr(update, "status", "") or "Generating..."))


class _WrappedCallState:
    def __init__(self, output_count: int) -> None:
        self.output_count = output_count
        self.done = threading.Event()
        self.result: Any = None
        self.has_result = False
        self.error: BaseException | None = None
        self.job: SessionJob | None = None
        self.abort_client_id = ""
        self._abort_client_ids: list[str] = []
        self._abort_client_ids_lock = threading.Lock()
        self.callback_context_ready = threading.Event()
        self.callback_context: dict[str, Any] | None = None
        self._followup_jobs: list[SessionJob] = []
        self._followup_lock = threading.Lock()
        self._followup_enabled = False
        self._primary_job_forwarded = False
        self._yielded_results: list[Any] = []
        self._yielded_results_lock = threading.Lock()
        self._log_messages: list[dict[str, Any]] = []
        self._log_messages_lock = threading.Lock()

    def set_result(self, result: Any) -> None:
        self.result = result
        self.has_result = True
        self.done.set()

    def set_completed(self) -> None:
        self.done.set()

    def set_error(self, error: BaseException) -> None:
        self.error = error
        self.done.set()

    def set_callback_context(self, context: dict[str, Any]) -> None:
        self.callback_context = dict(context)
        self.callback_context_ready.set()

    def enable_followup_queue_triggers(self) -> None:
        self._followup_enabled = True

    def add_followup_job(self, job: SessionJob) -> None:
        if not self._followup_enabled or _server_queue(job._session._state) is not None:  # admitted by the server: nothing to forward
            return
        with self._followup_lock:
            self._followup_jobs.append(job)

    def pop_ready_followup_load_queue_token(self) -> str:
        with self._followup_lock:
            for index, job in enumerate(self._followup_jobs):
                if job.webui_submission_ready:
                    self._followup_jobs.pop(index)
                    return job.webui_load_queue_token
        return ""

    def pop_primary_load_queue_token(self) -> str | None:
        """None until the job's tasks are submitted, then once the token its browser page forwards ("" when the server admitted them)."""
        if self._primary_job_forwarded or self.job is None or not self.job.webui_submission_ready:
            return None
        self._primary_job_forwarded = True
        self.enable_followup_queue_triggers()
        return "" if _server_queue(self.job._session._state) is not None else self.job.webui_load_queue_token

    def queue_abort_client_id(self, client_id: str) -> None:
        client_id = str(client_id or "").strip()
        if len(client_id) == 0:
            return
        with self._abort_client_ids_lock:
            self._abort_client_ids.append(client_id)

    def pop_abort_client_id(self) -> str:
        with self._abort_client_ids_lock:
            if not self._abort_client_ids:
                return ""
            client_id = self._abort_client_ids.pop(0)
        self.abort_client_id = client_id
        return client_id

    def push_yielded_result(self, result: Any) -> None:
        with self._yielded_results_lock:
            self._yielded_results.append(result)

    def pop_yielded_result(self) -> Any:
        with self._yielded_results_lock:
            if not self._yielded_results:
                return _NO_YIELDED_RESULT
            return self._yielded_results.pop(0)

    def queue_log_message(self, *, message: str, title: str, level: str, duration: float | None, visible: bool) -> None:
        with self._log_messages_lock:
            self._log_messages.append({"log": str(message or ""), "title": str(title or ""), "level": str(level or "info"), "duration": duration, "visible": bool(visible)})

    def pop_log_messages(self) -> list[dict[str, Any]]:
        with self._log_messages_lock:
            if not self._log_messages:
                return []
            messages = list(self._log_messages)
            self._log_messages.clear()
            return messages


class _BoundGradioCallbacks:
    def __init__(self, callbacks: object, state: _WrappedCallState, owner: "GradioWanGPSession") -> None:
        self._callbacks = callbacks
        self._state = state
        self._owner = owner

    def __getattr__(self, name: str) -> Any:
        target = getattr(self._callbacks, name)
        if not callable(target):
            return target

        @functools.wraps(target)
        def wrapped(*args, **kwargs):
            self._state.callback_context_ready.wait(timeout=30.0)
            context = self._state.callback_context
            if not isinstance(context, dict):
                return target(*args, **kwargs)
            with self._owner._push_callback_context(context):
                return target(*args, **kwargs)

        return wrapped


class WebUIQueueProbe:
    _POLL_INTERVAL_SECONDS = 0.2
    _MISSING_OUTPUT_TIMEOUT_SECONDS = 5.0
    _QUEUE_ADMISSION_SUSPEND_NOTICE_SECONDS = 10.0
    _INLINE_QUEUE_SLOT_TIMEOUT_SECONDS = 10.0
    _CANCEL_GRACE_SECONDS = 1.0

    def __init__(self, session: WanGPSession, runtime, tasks: list[dict[str, Any]], job: SessionJob) -> None:
        self._session = session
        self._runtime = runtime
        self._tasks = tasks
        self._job = job
        self._wgp = runtime.module
        self._gen = session._state["gen"]
        self._manifest = job.webui_manifest or self._build_manifest(tasks)
        self._client_ids: list[str] = []
        self._task_index_by_client_id: dict[str, int] = {}
        self._task_id_by_client_id: dict[str, Any] = {}
        self._outputs_by_client_id: dict[str, list[str]] = {}
        self._artifacts_by_client_id: dict[str, GeneratedArtifact] = {}
        self._errors_by_client_id: dict[str, GenerationError] = {}
        self._admitted_client_ids: set[str] = set()
        self._missing_output_since: dict[str, float] = {}
        self._last_status_text = ""
        self._last_active_client_id = ""
        self._last_progress_key: tuple[Any, ...] | None = None
        self._last_preview_key: tuple[Any, ...] | None = None
        self._active_progress_seed: Any = None
        self._cancel_issued = False
        self._cancel_requested_at: float | None = None
        self._cancel_dispatched_client_ids: set[str] = set()
        self._submitted_at = 0.0
        self._queue_wait_suspended = False
        self._logged_admitted_client_ids: set[str] = set()
        self._logged_missing_output_client_ids: set[str] = set()
        self._live_started_client_ids: set[str] = set()

        for index, task in enumerate(self._tasks, start=1):
            params = self._session._get_task_settings(task)
            client_id = str(params.get("client_id", "") or "").strip()
            if len(client_id) == 0:
                continue
            self._client_ids.append(client_id)
            self._task_index_by_client_id[client_id] = index
            self._task_id_by_client_id[client_id] = task.get("id")

    def run(self) -> GenerationResult:
        self._submit_inline_manifest()
        while not self._all_clients_finished():
            self._poll_once()
            if self._all_clients_finished():
                break
            time.sleep(self._POLL_INTERVAL_SECONDS)
        generated_files = [path for client_id in self._client_ids for path in self._outputs_by_client_id.get(client_id, [])]
        errors = [self._errors_by_client_id[client_id] for client_id in self._client_ids if client_id in self._errors_by_client_id]
        successful_tasks = sum(client_id in self._outputs_by_client_id for client_id in self._client_ids)
        failed_tasks = len(self._client_ids) - successful_tasks
        return GenerationResult(
            success=len(errors) == 0 and failed_tasks == 0,
            generated_files=generated_files,
            errors=errors,
            total_tasks=len(self._client_ids),
            successful_tasks=successful_tasks,
            failed_tasks=failed_tasks,
            artifacts=tuple(self._artifacts_by_client_id.get(client_id) for client_id in self._client_ids if client_id in self._artifacts_by_client_id),
        )

    def _submit_inline_manifest(self) -> None:
        self._reset_idle_state()
        with _INLINE_QUEUE_LOCK:
            self._wait_for_inline_queue_slot()
            if self._job.cancel_requested:
                for client_id in self._client_ids:
                    self._register_error(client_id, "Generation was cancelled", stage="cancelled")
                return
            self._gen.setdefault("queue_errors", {})
            self._gen["inline_queue"] = copy.deepcopy(self._manifest)
            self._job._mark_webui_submission_ready()
            print(f"WanGP API queued client_ids={self._client_ids}")
            gradio_context = getattr(self._session, "_gradio_webui_context", None)
            if not isinstance(gradio_context, dict) or not gradio_context.get("defer_load_queue_trigger", False):
                self._trigger_load_queue_event()
        self._submitted_at = time.time()
        self._publish("status", "Queued in WanGP...", "on_status")

    def _trigger_load_queue_event(self) -> None:
        gradio_context = getattr(self._session, "_gradio_webui_context", None)
        if not isinstance(gradio_context, dict):
            raise RuntimeError("WanGP WebUI queue submission requires an active Gradio session context.")
        if "load_queue" in gradio_context:  # in-process server: no browser request to replay
            return gradio_context["load_queue"]()
        fn_index = gradio_context.get("load_queue_fn_index")
        blocks = gradio_context.get("blocks")
        request = gradio_context.get("request")
        session_hash = gradio_context.get("session_hash")
        if not isinstance(fn_index, int) or blocks is None or request is None or not session_hash:
            raise RuntimeError("WanGP WebUI queue trigger is unavailable for the current Gradio session.")
        from gradio.data_classes import PredictBodyInternal

        request = _normalize_queue_request(request)
        if getattr(blocks._queue, "server_app", None) is None and getattr(blocks, "app", None) is not None:
            blocks._queue.set_server_app(blocks.app)
        body = PredictBodyInternal(session_hash=session_hash, fn_index=fn_index, data=[None, None], request=request)
        success, error_or_event_id = asyncio.run(blocks._queue.push(body=body, request=request, username=getattr(request, "username", None)))
        if not success:
            raise RuntimeError(str(error_or_event_id))

    def _wait_for_inline_queue_slot(self) -> None:
        deadline = time.time() + self._INLINE_QUEUE_SLOT_TIMEOUT_SECONDS
        while self._gen.get("inline_queue") is not None:
            if self._job.cancel_requested:
                return
            if time.time() >= deadline:
                raise RuntimeError("WanGP inline queue bridge is busy")
            time.sleep(0.05)

    def _reset_idle_state(self) -> None:
        if self._gen.get("in_progress", False) or list(self._gen.get("queue", []) or []):
            return
        self._gen["abort"] = False
        self._gen["resume"] = False
        self._gen["early_stop"] = False
        self._gen["early_stop_forwarded"] = False
        set_generation_state(self._gen, status="", status_display=False)
        self._gen["last_progress_args"] = None
        self._gen["progress_args"] = None
        self._gen["preview"] = None

    def _poll_once(self) -> None:
        queue_client_ids, active_client_id = self._get_queue_snapshot()
        for client_id in queue_client_ids:
            if client_id in self._client_ids:
                self._admitted_client_ids.add(client_id)
                if client_id not in self._logged_admitted_client_ids:
                    print(f"WanGP API admitted client_id={client_id}")
                    self._logged_admitted_client_ids.add(client_id)
        if self._job.cancel_requested:
            self._request_cancel()
        # Historical admissions do not mean the remaining requests have resumed.
        if self._queue_wait_suspended and any(client_id in self._client_ids for client_id in queue_client_ids):
            print("WanGP back in focus API queue resumed")
            self._queue_wait_suspended = False
        self._check_queue_errors()
        self._check_outputs(queue_client_ids)
        self._emit_live_updates(queue_client_ids, active_client_id)
        self._check_queue_admission_timeout()
        self._finalize_cancelled_clients(queue_client_ids)

    def _get_queue_snapshot(self) -> tuple[list[str], str]:
        queue_client_ids: list[str] = []
        active_client_id = ""
        first_queue_task = True
        for task in list(self._gen.get("queue", []) or []):
            if not isinstance(task, dict):
                continue
            params = self._session._get_task_settings(task)
            client_id = str(params.get("client_id", "") or "").strip()
            if first_queue_task:
                active_client_id = client_id
                first_queue_task = False
            if len(client_id) == 0:
                continue
            queue_client_ids.append(client_id)
        return queue_client_ids, active_client_id

    def _check_queue_errors(self) -> None:
        queue_errors = self._gen.get("queue_errors", {}) or {}
        for client_id in self._client_ids:
            if client_id in self._outputs_by_client_id or client_id in self._errors_by_client_id:
                continue
            error_tuple = queue_errors.get(client_id)
            if error_tuple is None:
                continue
            error_text = str(error_tuple[0] if len(error_tuple) > 0 else "WanGP queue error")
            aborted = bool(error_tuple[1]) if len(error_tuple) > 1 else False
            print(f"WanGP API queue error client_id={client_id} aborted={aborted} error={error_text}")
            if aborted:
                self._remove_queue_client_id(client_id)
                self._register_error(client_id, "Generation was cancelled", stage="cancelled")
            else:
                self._register_error(client_id, error_text or "WanGP queue error", stage="generation")

    def _check_outputs(self, queue_client_ids: list[str]) -> None:
        galleries = self._wgp.get_processed_queue(self._session._workspace_gallery())
        for client_id in self._client_ids:
            if client_id in self._outputs_by_client_id or client_id in self._errors_by_client_id:
                continue
            output_paths = self._find_outputs_for_client(client_id, galleries)
            pending_artifact = self._session._peek_output_artifact(client_id)
            if pending_artifact is not None and client_id not in queue_client_ids:
                if not queue_client_ids and self._gen.get("in_progress", False):
                    if client_id not in self._logged_missing_output_client_ids:
                        print(f"WanGP API delaying completion for client_id={client_id} until main queue settles")
                        self._logged_missing_output_client_ids.add(client_id)
                    continue
                artifact = self._session._consume_output_artifact(client_id)
                artifact_path = str(artifact.path if artifact is not None else "").strip()
                if artifact_path:
                    artifact_path = str((Path(artifact_path) if Path(artifact_path).is_absolute() else self._session._root / artifact_path).resolve())
                resolved_output_paths = list(output_paths)
                resolved_path_keys = {os.path.normcase(str(Path(path).resolve())) for path in resolved_output_paths}
                if artifact_path and os.path.normcase(artifact_path) not in resolved_path_keys:
                    resolved_output_paths.append(artifact_path)
                if not resolved_output_paths:
                    self._register_error(client_id, f"Generation produced an API artifact for client_id '{client_id}' without an output path.", stage="generation")
                    continue
                self._outputs_by_client_id[client_id] = resolved_output_paths
                if artifact is not None:
                    self._artifacts_by_client_id[client_id] = GeneratedArtifact(
                        path=artifact_path or resolved_output_paths[-1],
                        media_type=artifact.media_type,
                        client_id=artifact.client_id,
                        video_tensor_uint8=artifact.video_tensor_uint8,
                        video_tensor_hdr=artifact.video_tensor_hdr,
                        hdr=artifact.hdr,
                        audio_tensor=artifact.audio_tensor,
                        audio_sampling_rate=artifact.audio_sampling_rate,
                        fps=artifact.fps,
                        flashvsr_continue_cache=artifact.flashvsr_continue_cache,
                    )
                self._missing_output_since.pop(client_id, None)
                self._logged_missing_output_client_ids.discard(client_id)
                print(f"WanGP API completed client_id={client_id} via artifact paths={resolved_output_paths}")
                for output_path in resolved_output_paths:
                    self._publish("output", {"client_id": client_id, "path": output_path}, "on_output")
                continue
            if output_paths and client_id not in queue_client_ids:
                if not queue_client_ids and self._gen.get("in_progress", False):
                    if client_id not in self._logged_missing_output_client_ids:
                        print(f"WanGP API delaying gallery completion for client_id={client_id} until main queue settles")
                        self._logged_missing_output_client_ids.add(client_id)
                    continue
                self._outputs_by_client_id[client_id] = output_paths
                artifact = self._session._consume_output_artifact(client_id)
                if artifact is not None:
                    self._artifacts_by_client_id[client_id] = GeneratedArtifact(
                        path=str(artifact.path or output_paths[-1]),
                        media_type=artifact.media_type,
                        client_id=artifact.client_id,
                        video_tensor_uint8=artifact.video_tensor_uint8,
                        video_tensor_hdr=artifact.video_tensor_hdr,
                        hdr=artifact.hdr,
                        audio_tensor=artifact.audio_tensor,
                        audio_sampling_rate=artifact.audio_sampling_rate,
                        fps=artifact.fps,
                        flashvsr_continue_cache=artifact.flashvsr_continue_cache,
                    )
                self._missing_output_since.pop(client_id, None)
                self._logged_missing_output_client_ids.discard(client_id)
                print(f"WanGP API completed client_id={client_id} via gallery paths={output_paths}")
                for output_path in output_paths:
                    self._publish("output", {"client_id": client_id, "path": output_path}, "on_output")
                continue
            if client_id in queue_client_ids:
                self._missing_output_since.pop(client_id, None)
                self._logged_missing_output_client_ids.discard(client_id)
                continue
            if client_id not in self._admitted_client_ids:
                continue
            started_missing_at = self._missing_output_since.setdefault(client_id, time.time())
            if client_id not in self._logged_missing_output_client_ids:
                print(f"WanGP API waiting for output client_id={client_id} queue_empty={not queue_client_ids} artifact_ready={pending_artifact is not None}")
                self._logged_missing_output_client_ids.add(client_id)
            if time.time() - started_missing_at >= self._MISSING_OUTPUT_TIMEOUT_SECONDS:
                self._register_error(
                    client_id,
                    f"Generation finished queue processing but no output with client_id '{client_id}' was found in the gallery.",
                    stage="generation",
                )

    def _emit_live_updates(self, queue_client_ids: list[str], active_client_id: str) -> None:
        if active_client_id != self._last_active_client_id:
            self._last_active_client_id = active_client_id
            self._last_progress_key = None
            self._last_preview_key = None
            self._last_status_text = ""
            self._active_progress_seed = copy.deepcopy(self._gen.get("last_progress_args"))
        live_generation_running = bool(self._gen.get("in_progress", False))
        active_client_is_live = live_generation_running and active_client_id in self._client_ids and active_client_id not in self._outputs_by_client_id and active_client_id not in self._errors_by_client_id
        if active_client_is_live:
            self._live_started_client_ids.add(active_client_id)
            progress_args = self._gen.get("last_progress_args")
            progress_ready = progress_args != self._active_progress_seed
            progress_update = self._session._build_progress_update(progress_args, include_state_fallback=False, task_id=active_client_id, task_index=self._task_index_by_client_id[active_client_id], task_count=len(self._client_ids)) if progress_ready else None
            if progress_update is not None:
                if len(progress_update.status) > 0 and progress_update.status != self._last_status_text:
                    self._last_status_text = progress_update.status
                    self._publish("status", progress_update.status, "on_status")
                progress_key = (
                    active_client_id,
                    progress_update.phase,
                    progress_update.progress,
                    progress_update.current_step,
                    progress_update.total_steps,
                    progress_update.status,
                    progress_update.unit,
                    progress_update.window_index,
                    progress_update.window_count,
                )
                if progress_key != self._last_progress_key:
                    self._last_progress_key = progress_key
                    self._publish("progress", progress_update, "on_progress")
            preview_image = self._gen.get("preview")
            if preview_image is not None and progress_update is not None:
                preview_key = (active_client_id, id(preview_image), getattr(preview_image, "size", None), progress_update.progress)
                if preview_key != self._last_preview_key:
                    self._last_preview_key = preview_key
                    self._publish(
                        "preview",
                        PreviewUpdate(
                            image=preview_image.image if isinstance(preview_image, VideoPreview) else preview_image,
                            video=preview_image.video if isinstance(preview_image, VideoPreview) else None,
                            phase=progress_update.phase,
                            status=progress_update.status,
                            progress=progress_update.progress,
                            current_step=progress_update.current_step,
                            total_steps=progress_update.total_steps,
                        ),
                        "on_preview",
                    )
            return
        queued_client_ids = [
            client_id for client_id in queue_client_ids
            if client_id in self._client_ids and client_id not in self._outputs_by_client_id and client_id not in self._errors_by_client_id
        ]
        if queued_client_ids and any(client_id not in self._live_started_client_ids for client_id in queued_client_ids):
            status_text = "Waiting in WanGP queue..."
            if status_text != self._last_status_text:
                self._last_status_text = status_text
                self._publish("status", status_text, "on_status")

    def _check_queue_admission_timeout(self) -> None:
        pending_client_ids = [
            client_id
            for client_id in self._client_ids
            if client_id not in self._outputs_by_client_id and client_id not in self._errors_by_client_id and client_id not in self._admitted_client_ids
        ]
        if not pending_client_ids:
            self._queue_wait_suspended = False
            return
        if self._gen.get("in_progress", False) or list(self._gen.get("queue", []) or []):
            self._submitted_at = time.time()
            return
        if self._submitted_at <= 0 or time.time() - self._submitted_at < self._QUEUE_ADMISSION_SUSPEND_NOTICE_SECONDS or self._queue_wait_suspended:
            return
        print("WanGP API queue suspended while waiting for Media Generator to get browser focus")
        self._publish("status", "Waiting for WanGP Media Generator to get browser focus...", "on_status")
        self._queue_wait_suspended = True

    def _finalize_cancelled_clients(self, queue_client_ids: list[str]) -> None:
        if not self._cancel_issued or self._cancel_requested_at is None:
            return
        if time.time() - self._cancel_requested_at < self._CANCEL_GRACE_SECONDS:
            return
        for client_id in self._client_ids:
            if client_id in self._outputs_by_client_id or client_id in self._errors_by_client_id:
                continue
            if client_id in queue_client_ids or self._inline_queue_contains_client_id(client_id):
                continue
            self._register_error(client_id, "Generation was cancelled", stage="cancelled")

    def _request_cancel(self) -> None:
        dispatched_any = False
        for client_id in self._client_ids:
            if client_id in self._outputs_by_client_id or client_id in self._errors_by_client_id:
                continue
            if client_id in self._cancel_dispatched_client_ids:
                continue
            if self._remove_inline_queue_client_id(client_id):
                self._cancel_dispatched_client_ids.add(client_id)
                dispatched_any = True
                continue
            if client_id in self._admitted_client_ids:
                self._queue_abort_client_id(client_id)
                self._cancel_dispatched_client_ids.add(client_id)
                dispatched_any = True
        if dispatched_any:
            self._cancel_issued = True
            self._cancel_requested_at = time.time()

    def _queue_abort_client_id(self, client_id: str) -> None:
        self._wgp.abort_generation(self._session._state, client_id, notify=False)
        print(f"WanGP API dispatched abort for client_id={client_id}")

    def _remove_inline_queue_client_id(self, client_id: str) -> bool:
        inline_queue = self._gen.get("inline_queue")
        if inline_queue is None:
            return False

        def _matches(item: Any) -> bool:
            if not isinstance(item, dict):
                return False
            params = item.get("params")
            if isinstance(params, dict) and str(params.get("client_id", "") or "").strip() == client_id:
                return True
            return str(item.get("client_id", "") or "").strip() == client_id

        if _matches(inline_queue):
            self._gen.pop("inline_queue", None)
            return True
        if isinstance(inline_queue, list):
            remaining = [item for item in inline_queue if not _matches(item)]
            if len(remaining) != len(inline_queue):
                if remaining:
                    self._gen["inline_queue"] = remaining
                else:
                    self._gen.pop("inline_queue", None)
                return True
        return False

    def _remove_queue_client_id(self, client_id: str) -> bool:
        queue = self._gen.get("queue")
        if not isinstance(queue, list):
            return False
        remaining = []
        removed = False
        for item in list(queue):
            if self._inline_item_matches_client_id(item, client_id):
                removed = True
                continue
            remaining.append(item)
        if removed:
            queue[:] = remaining
            self._gen["queue"] = queue
        return removed

    def _inline_queue_contains_client_id(self, client_id: str) -> bool:
        inline_queue = self._gen.get("inline_queue")
        if inline_queue is None:
            return False
        if isinstance(inline_queue, list):
            return any(self._inline_item_matches_client_id(item, client_id) for item in inline_queue)
        return self._inline_item_matches_client_id(inline_queue, client_id)

    @staticmethod
    def _inline_item_matches_client_id(item: Any, client_id: str) -> bool:
        if not isinstance(item, dict):
            return False
        params = item.get("params")
        if isinstance(params, dict) and str(params.get("client_id", "") or "").strip() == client_id:
            return True
        return str(item.get("client_id", "") or "").strip() == client_id

    @staticmethod
    def _find_outputs_for_client(
        client_id: str,
        galleries: dict[str, tuple[Sequence[Any], Sequence[Any]]],
    ) -> list[str]:
        outputs = []
        for paths, settings_list in galleries.values():
            for path, settings in zip(paths or [], settings_list or []):
                if not isinstance(settings, dict):
                    continue
                if str(settings.get("client_id", "") or "").strip() == client_id:
                    outputs.append(str(Path(path).resolve()))
        return outputs

    def _register_error(self, client_id: str, message: str, *, stage: str) -> None:
        if client_id in self._errors_by_client_id or client_id in self._outputs_by_client_id:
            return
        failure = GenerationError(
            message=message,
            task_index=self._task_index_by_client_id.get(client_id),
            task_id=self._task_id_by_client_id.get(client_id),
            stage=stage,
        )
        self._errors_by_client_id[client_id] = failure
        self._publish("error", failure, "on_error")

    def _publish(self, kind: str, payload: Any, callback_name: str | None = None) -> None:
        self._job.events.put(kind, payload)
        if callback_name is not None:
            self._session._emit_callback(callback_name, payload, job=self._job)

    def _all_clients_finished(self) -> bool:
        completed_count = len(self._outputs_by_client_id) + len(self._errors_by_client_id)
        return completed_count >= len(self._client_ids)

    @staticmethod
    def _build_manifest(tasks: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        manifest = []
        for index, task in enumerate(tasks, start=1):
            params = copy.deepcopy(WanGPSession._get_task_settings(task))
            manifest.append({"id": task.get("id", index), "params": params, "plugin_data": copy.deepcopy(task.get("plugin_data", {}))})
        return manifest


def run_webui_job(session, job: SessionJob, tasks: list[dict[str, Any]]) -> None:
    try:
        runtime = session._ensure_runtime()
        job.events.put("started", {"tasks": len(tasks), "backend": "webui_queue"})
        result = WebUIQueueProbe(session, runtime, tasks, job).run()
        job.events.put("completed", result)
        session._emit_callback("on_complete", result, job=job)
        job._set_result(result)
    except BaseException as exc:
        failure = session._make_generation_error(exc, task_index=None, task_id=None, stage="runtime")
        result = GenerationResult(
            success=False,
            generated_files=[],
            errors=[failure],
            total_tasks=len(tasks),
            successful_tasks=0,
            failed_tasks=max(1, len(tasks)),
            artifacts=(),
        )
        job.events.put("error", failure)
        session._emit_callback("on_error", failure, job=job)
        job.events.put("completed", result)
        session._emit_callback("on_complete", result, job=job)
        job._set_result(result)
    finally:
        job.events.close()
        with session._job_lock:
            if session._active_job is job:
                session._active_job = None


class GradioWanGPSession:
    def __init__(self, *, init_fn, plugin=None, state_component: Any = None, session_kwargs: dict[str, Any] | None = None) -> None:
        self._init_fn = init_fn
        self._plugin = plugin
        self._state_component = state_component
        self._session_kwargs = dict(session_kwargs or {})
        self._session_kwargs.setdefault("console_output", False)
        self._session: WanGPSession | None = None
        self._defer_load_queue_trigger = False
        self._ui_local = threading.local()
        self._wrapped_calls: dict[str, _WrappedCallState] = {}
        self._wrapped_calls_lock = threading.Lock()
        self._ui_call_component = None

    @classmethod
    def for_plugin(cls, plugin, *, init_fn, session_kwargs: dict[str, Any] | None = None):
        plugin.request_component("state")
        return cls(init_fn=init_fn, plugin=plugin, session_kwargs=session_kwargs)

    def submit(self, source, callbacks: object | None = None) -> SessionJob:
        session = self._ensure_session()
        self._bind_gradio_context(session)
        job = session.submit(source, callbacks=self._wrap_callbacks_for_current_call(callbacks))
        self._capture_job_for_current_call(job)
        return job

    def list_model_defs(self, **filters: Any) -> list[dict[str, Any]]:
        return self._ensure_session().list_model_defs(**filters)

    def get_model_defs(self, **filters: Any) -> list[dict[str, Any]]:
        return self.list_model_defs(**filters)

    def list_model_metadata(self, **filters: Any) -> list[dict[str, Any]]:
        return self._ensure_session().list_model_metadata(**filters)

    def get_model_def(self, model_type: str) -> dict[str, Any] | None:
        return self._ensure_session().get_model_def(model_type)

    def get_model_metadata(self, model_type: str) -> dict[str, Any] | None:
        return self._ensure_session().get_model_metadata(model_type)

    def get_default_settings(self, model_type: str) -> dict[str, Any]:
        return self._ensure_session().get_default_settings(model_type)

    def merge_settings_with_defaults(self, settings: dict[str, Any]) -> dict[str, Any]:
        return self._ensure_session().merge_settings_with_defaults(settings)

    def prepare_settings_for_export(self, settings: dict[str, Any]) -> dict[str, Any]:
        return self._ensure_session().prepare_settings_for_export(settings)

    def get_exported_default_settings(self, model_type: str) -> dict[str, Any]:
        return self._ensure_session().get_exported_default_settings(model_type)

    def get_model_settings(self, model_type: str, setting_id: str | None = None) -> dict[str, Any]:
        return self._ensure_session().get_model_settings(model_type, setting_id)

    def list_loras(self, model_type: str, name: str | None = None) -> dict[str, Any]:
        return self._ensure_session().list_loras(model_type, name=name)

    def get_model_schema(self, model_type: str) -> dict[str, Any] | None:
        return self._ensure_session().get_model_schema(model_type)

    def submit_task(self, settings: dict[str, Any], callbacks: object | None = None) -> SessionJob:
        session = self._ensure_session()
        self._bind_gradio_context(session)
        job = session.submit_task(settings, callbacks=self._wrap_callbacks_for_current_call(callbacks))
        self._capture_job_for_current_call(job)
        return job

    def submit_media_postprocessing(self, media_source, *, callbacks: object | None = None, **kwargs: Any) -> SessionJob:
        session = self._ensure_session()
        self._bind_gradio_context(session)
        job = session.submit_media_postprocessing(media_source, callbacks=self._wrap_callbacks_for_current_call(callbacks), **kwargs)
        self._capture_job_for_current_call(job)
        return job

    def submit_audio_remux(self, video_source, *, callbacks: object | None = None, **kwargs: Any) -> SessionJob:
        session = self._ensure_session()
        self._bind_gradio_context(session)
        job = session.submit_audio_remux(video_source, callbacks=self._wrap_callbacks_for_current_call(callbacks), **kwargs)
        self._capture_job_for_current_call(job)
        return job

    def submit_audio_postprocessing(self, audio_source, *, callbacks: object | None = None, **kwargs: Any) -> SessionJob:
        session = self._ensure_session()
        self._bind_gradio_context(session)
        job = session.submit_audio_postprocessing(audio_source, callbacks=self._wrap_callbacks_for_current_call(callbacks), **kwargs)
        self._capture_job_for_current_call(job)
        return job

    def submit_manifest(self, settings_list: list[dict[str, Any]], callbacks: object | None = None) -> SessionJob:
        session = self._ensure_session()
        self._bind_gradio_context(session)
        job = session.submit_manifest(settings_list, callbacks=self._wrap_callbacks_for_current_call(callbacks))
        self._capture_job_for_current_call(job)
        return job

    def run(self, source, callbacks: object | None = None) -> GenerationResult:
        session = self._ensure_session()
        self._bind_gradio_context(session)
        return session.run(source, callbacks=self._wrap_callbacks_for_current_call(callbacks))

    def run_task(self, settings: dict[str, Any], callbacks: object | None = None) -> GenerationResult:
        session = self._ensure_session()
        self._bind_gradio_context(session)
        return session.run_task(settings, callbacks=self._wrap_callbacks_for_current_call(callbacks))

    def run_media_postprocessing(self, media_source, **kwargs: Any) -> GenerationResult:
        return self.submit_media_postprocessing(media_source, **kwargs).result()

    def run_audio_remux(self, video_source, **kwargs: Any) -> GenerationResult:
        return self.submit_audio_remux(video_source, **kwargs).result()

    def run_audio_postprocessing(self, audio_source, **kwargs: Any) -> GenerationResult:
        return self.submit_audio_postprocessing(audio_source, **kwargs).result()

    def run_manifest(self, settings_list: list[dict[str, Any]], callbacks: object | None = None) -> GenerationResult:
        session = self._ensure_session()
        self._bind_gradio_context(session)
        return session.run_manifest(settings_list, callbacks=self._wrap_callbacks_for_current_call(callbacks))

    def ensure_ready(self):
        self._ensure_session().ensure_ready()
        return self

    def close(self) -> None:
        if self._session is None:
            return
        self._session.close()
        self._session = None

    def cancel(self) -> None:
        if self._session is not None:
            self._session.cancel()

    @contextlib.contextmanager
    def plugin_ui_context(self):
        import gradio as gr

        original_click = gr.Button.click
        if self._ui_call_component is None:
            self._ui_call_component = gr.State("")

        @functools.wraps(original_click)
        def patched_click(button, *args, **kwargs):
            fn = kwargs.get("fn")
            if fn is None and args:
                fn = args[0]
            if not callable(fn):
                return original_click(button, *args, **kwargs)
            if not self._callback_uses_api_session(fn):
                return original_click(button, *args, **kwargs)
            return self._wrap_button_click(original_click, button, *args, **kwargs)

        gr.Button.click = patched_click
        try:
            yield
        finally:
            gr.Button.click = original_click

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._ensure_session(), name)

    def _wrap_button_click(self, original_click, button, *args, **kwargs):
        fn = kwargs.get("fn")
        if fn is None and args:
            fn = args[0]
        inputs = kwargs.get("inputs")
        if inputs is None and len(args) > 1:
            inputs = args[1]
        outputs = kwargs.get("outputs")
        if outputs is None and len(args) > 2:
            outputs = args[2]
        original_outputs = self._normalize_outputs(outputs)
        explicit_show_progress = kwargs.get("show_progress") if "show_progress" in kwargs else None
        explicit_progress_targets = self._normalize_outputs(kwargs.get("show_progress_on")) if "show_progress_on" in kwargs else None
        load_queue_trigger = self._resolve_main_bridge_component("wangp_main_load_queue_trigger")
        abort_client_id = self._resolve_main_bridge_component("wangp_main_abort_client_id")
        call_state = self._ui_call_component
        wrapped_start = self._make_wrapped_click_start(fn, len(original_outputs))
        kwargs["fn"] = wrapped_start
        kwargs["outputs"] = [*original_outputs, load_queue_trigger, abort_client_id, call_state]
        kwargs["show_progress"] = "hidden"
        args = ()
        dependency = original_click(button, *args, **kwargs)
        wait_outputs = [*original_outputs, load_queue_trigger, abort_client_id, call_state]
        progress_targets = explicit_progress_targets if explicit_progress_targets is not None else [component for component in original_outputs if hasattr(component, "_id")]
        def wait_wrapped_call(call_id):
            yield from self._wait_wrapped_call(call_id, len(original_outputs))

        then_kwargs = {"fn": wait_wrapped_call, "inputs": [call_state], "outputs": wait_outputs, "show_progress": explicit_show_progress or "full"}
        if progress_targets is not None and len(progress_targets) > 0:
            then_kwargs["show_progress_on"] = progress_targets
        dependency.then(
            **then_kwargs,
        )
        return dependency

    def _make_wrapped_click_start(self, fn, output_count: int):
        import gradio as gr

        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            call_id = str(time.time_ns())
            state = _WrappedCallState(output_count)
            self._remember_wrapped_call(call_id, state)
            bound_state = self._resolve_state()
            bound_context = self._capture_current_gradio_context()
            state.set_callback_context(bound_context)
            worker = threading.Thread(target=self._run_wrapped_click_worker, args=(call_id, fn, args, kwargs, bound_state, bound_context), daemon=True, name="wangp-plugin-click")
            worker.start()
            while True:
                yielded_result = state.pop_yielded_result()
                if yielded_result is not _NO_YIELDED_RESULT:
                    return [*self._normalize_callback_result(yielded_result, output_count), gr.skip(), gr.skip(), call_id]
                load_queue_token = state.pop_primary_load_queue_token()
                if load_queue_token is not None:  # the tasks are submitted: the next step follows the job
                    return [*self._blank_outputs(output_count), load_queue_token or gr.skip(), gr.skip(), call_id]
                if state.job is not None and not state.done.is_set():
                    state.job._webui_submission_ready.wait(timeout=0.05)
                    continue
                if not state.done.wait(timeout=0.05):
                    continue
                if state.error is not None:
                    self._forget_wrapped_call(call_id)
                    raise self._as_gradio_error(state.error)
                yielded_result = state.pop_yielded_result()
                if yielded_result is not _NO_YIELDED_RESULT:
                    return [*self._normalize_callback_result(yielded_result, output_count), gr.skip(), gr.skip(), call_id]
                if not state.has_result:
                    self._forget_wrapped_call(call_id)
                    return [*self._blank_outputs(output_count), gr.skip(), gr.skip(), ""]
                self._forget_wrapped_call(call_id)
                return [*self._normalize_callback_result(state.result, output_count), gr.skip(), gr.skip(), ""]

        wrapped.__signature__ = inspect.signature(fn)
        return wrapped

    def _wait_wrapped_call(self, call_id: str, output_count: int):
        import gradio as gr

        call_id = str(call_id or "").strip()
        if len(call_id) == 0:
            yield [*self._blank_outputs(output_count), gr.skip(), gr.skip(), ""]
            return
        state = self._get_wrapped_call(call_id)
        if state is None:
            yield [*self._blank_outputs(output_count), gr.skip(), gr.skip(), ""]
            return
        # The page may render only the last of updates that arrive together: the final update repeats each output's latest change.
        latest = self._blank_outputs(output_count)
        try:
            state.set_callback_context(self._capture_progress_callback_context())
            self._flush_buffered_log_messages(state)
            while True:
                self._flush_buffered_log_messages(state)
                load_queue_token = state.pop_primary_load_queue_token()
                if load_queue_token:
                    yield [*self._blank_outputs(output_count), load_queue_token, gr.skip(), call_id]
                    continue
                load_queue_token = state.pop_ready_followup_load_queue_token()
                if load_queue_token:
                    print(f"WanGP API forwarding follow-up load_queue_trigger token={load_queue_token}")
                    yield [*self._blank_outputs(output_count), load_queue_token, gr.skip(), call_id]
                    continue
                abort_client_id = state.pop_abort_client_id()
                if abort_client_id:
                    print(f"WanGP API forwarding abort_client_id={abort_client_id}")
                    yield [*self._blank_outputs(output_count), gr.skip(), abort_client_id, call_id]
                    continue
                yielded_result = state.pop_yielded_result()
                if yielded_result is not _NO_YIELDED_RESULT:
                    outputs = self._normalize_callback_result(yielded_result, output_count)
                    latest = list(map(self._merge_update, latest, outputs))
                    yield [*outputs, gr.skip(), gr.skip(), call_id]
                    continue
                if state.done.is_set():
                    if state.error is not None:
                        raise self._as_gradio_error(state.error)
                    if state.has_result:
                        latest = list(map(self._merge_update, latest, self._normalize_callback_result(state.result, output_count)))
                    yield [*latest, gr.skip(), gr.skip(), ""]
                    break
                time.sleep(0.05)
        finally:
            self._forget_wrapped_call(call_id)

    @staticmethod
    def _merge_update(previous: Any, update: Any) -> Any:
        """A gr.update() adds its properties to the output's previous change; any other value replaces it."""
        if not (isinstance(update, dict) and update.get("__type__") == "update"):
            return update
        if isinstance(previous, dict) and previous.get("__type__") == "update":
            return {**previous, **update}
        import gradio as gr

        return update if isinstance(previous, gr.blocks.Block) else {"__type__": "update", "value": previous, **update}

    @staticmethod
    def _flush_buffered_log_messages(state: _WrappedCallState) -> None:
        context = state.callback_context
        if not isinstance(context, dict):
            return
        blocks = context.get("blocks")
        event_id = context.get("event_id")
        if blocks is None or event_id is None:
            return
        for message in state.pop_log_messages():
            blocks._queue.log_message(event_id=event_id, **message)

    def _run_wrapped_click_worker(self, call_id: str, fn, args, kwargs, bound_state: dict[str, Any], bound_context: dict[str, Any]) -> None:
        state = self._get_wrapped_call(call_id)
        if state is None:
            return
        _ensure_gradio_log_message_patch()
        self._ui_local.call_id = call_id
        self._ui_local.defer_load_queue_trigger = True
        self._ui_local.bound_state = bound_state
        _WRAPPED_LOG_LOCAL.owner = self
        _WRAPPED_LOG_LOCAL.call_id = call_id
        try:
            exec_context = dict(bound_context)
            if isinstance(state.callback_context, dict):
                exec_context.update(state.callback_context)
            self._ui_local.bound_gradio_context = exec_context
            with self._push_callback_context(exec_context):
                result = fn(*args, **kwargs)
                if inspect.isgenerator(result):
                    iterator = iter(result)
                    while True:
                        try:
                            state.push_yielded_result(next(iterator))
                        except StopIteration as stop:
                            if stop.value is not None:
                                state.set_result(stop.value)
                            else:
                                state.set_completed()
                            break
                else:
                    state.set_result(result)
        except BaseException as exc:
            state.set_error(exc)
        finally:
            _WRAPPED_LOG_LOCAL.owner = None
            _WRAPPED_LOG_LOCAL.call_id = ""
            self._ui_local.call_id = ""
            self._ui_local.defer_load_queue_trigger = False
            self._ui_local.bound_state = None
            self._ui_local.bound_gradio_context = None

    def _capture_job_for_current_call(self, job: SessionJob) -> None:
        call_id = str(getattr(self._ui_local, "call_id", "") or "").strip()
        if len(call_id) == 0:
            return
        job._bind_webui_owner_call(call_id)
        state = self._get_wrapped_call(call_id)
        if state is not None:
            if state.job is None:
                state.job = job
            else:
                state.add_followup_job(job)

    def _capture_cancelled_job(self, job: SessionJob) -> None:
        return

    def _enqueue_abort_client_id(self, job: SessionJob, client_id: str) -> bool:
        call_id = str(getattr(job, "webui_owner_call_id", "") or getattr(self._ui_local, "call_id", "") or "").strip()
        if len(call_id) == 0:
            return False
        state = self._get_wrapped_call(call_id)
        if state is None:
            return False
        state.queue_abort_client_id(client_id)
        return True

    def _remember_wrapped_call(self, call_id: str, state: _WrappedCallState) -> None:
        with self._wrapped_calls_lock:
            self._wrapped_calls[call_id] = state

    def _get_wrapped_call(self, call_id: str) -> _WrappedCallState | None:
        with self._wrapped_calls_lock:
            return self._wrapped_calls.get(call_id)

    def _forget_wrapped_call(self, call_id: str) -> None:
        with self._wrapped_calls_lock:
            self._wrapped_calls.pop(call_id, None)

    def _callback_uses_api_session(self, fn) -> bool:
        candidates: list[Any] = [fn]
        owner_candidates: list[Any] = [fn]
        api_attr_names = {"_api", "_wangp_session"}
        code = getattr(fn, "__code__", None)
        referenced_names = set(getattr(code, "co_names", ())) | set(getattr(code, "co_freevars", ()))
        try:
            closure_vars = inspect.getclosurevars(fn)
        except Exception:
            closure_vars = None
        if closure_vars is not None:
            candidates.extend(closure_vars.nonlocals.values())
            owner_candidates.extend(closure_vars.nonlocals.values())
            candidates.extend(closure_vars.globals.values())
        for values in (getattr(fn, "__defaults__", None) or (), (getattr(fn, "__kwdefaults__", None) or {}).values()):
            candidates.extend(values)
            owner_candidates.extend(values)
        for cell in getattr(fn, "__closure__", ()) or ():
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            candidates.append(value)
            owner_candidates.append(value)
        for candidate in candidates:
            owner = candidate.__self__ if inspect.ismethod(candidate) else candidate
            if owner is self or owner is self._session:
                return True
            if isinstance(owner, (GradioWanGPSession, WanGPSession)):
                return True
        if not referenced_names.intersection(api_attr_names):
            return False
        if closure_vars is not None:
            owner_candidates.extend(closure_vars.globals.values())
        for candidate in owner_candidates:
            owner = candidate.__self__ if inspect.ismethod(candidate) else candidate
            try:
                attrs = vars(owner)
            except TypeError:
                attrs = {}
            for attr_name in ("_api", "_wangp_session"):
                session = attrs.get(attr_name)
                if session is self or session is self._session or isinstance(session, (GradioWanGPSession, WanGPSession)):
                    return True
        return False

    def _wrap_callbacks_for_current_call(self, callbacks: object | None) -> object | None:
        if callbacks is None:
            return None
        call_id = str(getattr(self._ui_local, "call_id", "") or "").strip()
        if len(call_id) == 0:
            return callbacks
        state = self._get_wrapped_call(call_id)
        if state is None:
            return callbacks
        if isinstance(callbacks, _BoundGradioCallbacks):
            return callbacks
        return _BoundGradioCallbacks(callbacks, state, self)

    @staticmethod
    def _normalize_outputs(outputs: Any) -> list[Any]:
        if outputs is None:
            return []
        if isinstance(outputs, (list, tuple)):
            return list(outputs)
        return [outputs]

    @staticmethod
    def _blank_outputs(output_count: int) -> list[Any]:
        import gradio as gr

        return [gr.skip()] * output_count

    @staticmethod
    def _normalize_callback_result(result: Any, output_count: int) -> list[Any]:
        import gradio as gr

        if output_count <= 0:
            return []
        if output_count == 1:
            return [result]
        if isinstance(result, tuple):
            normalized = list(result)
        elif isinstance(result, list):
            normalized = list(result)
        else:
            normalized = [result]
        if len(normalized) < output_count:
            normalized.extend([gr.skip()] * (output_count - len(normalized)))
        return normalized[:output_count]

    @staticmethod
    def _as_gradio_error(error: BaseException):
        import gradio as gr

        return error if isinstance(error, gr.Error) else gr.Error(str(error))

    def _ensure_session(self) -> WanGPSession:
        state = self._resolve_state()
        if self._session is None or self._session._state is not state:
            session_kwargs = copy.deepcopy(self._session_kwargs)
            session_kwargs["webui_state"] = state
            self._session = WanGPSession(**session_kwargs)
            self._session._gradio_session_proxy = self
        return self._session

    def _resolve_state(self) -> dict[str, Any]:
        bound_state = getattr(self._ui_local, "bound_state", None)
        if isinstance(bound_state, dict):
            return bound_state
        component = self._state_component
        if component is None and self._plugin is not None:
            component = getattr(self._plugin, "state", None)
        state = self._resolve_live_session_state(component)
        if not isinstance(state, dict):
            state = getattr(component, "value", None) if component is not None else None
        if not isinstance(state, dict):
            raise RuntimeError("WanGP WebUI session requires access to the live Gradio state component.")
        return state

    @staticmethod
    def _resolve_live_session_state(component: Any) -> dict[str, Any] | None:
        component_id = getattr(component, "_id", None)
        if component_id is None:
            return None
        try:
            from gradio.context import LocalContext
        except Exception:
            return None
        try:
            blocks = LocalContext.blocks.get(None)
            request = LocalContext.request.get(None)
        except LookupError:
            return None
        session_hash = getattr(request, "session_hash", None) if request is not None else None
        state_holder = getattr(blocks, "state_holder", None) if blocks is not None else None
        if not session_hash or state_holder is None:
            return None
        try:
            session_state = state_holder[session_hash]
            state = session_state[component_id]
        except Exception:
            return None
        return state if isinstance(state, dict) else None

    def _bind_gradio_context(self, session: WanGPSession) -> None:
        service = _server_queue(session._state)
        if service is not None:
            session._gradio_webui_context = {"load_queue": service.load_inline_queue}
            return
        bound_context = getattr(self._ui_local, "bound_gradio_context", None)
        if isinstance(bound_context, dict):
            session._gradio_webui_context = dict(bound_context)
            session._gradio_webui_context["defer_load_queue_trigger"] = self._defer_load_queue_trigger or bool(getattr(self._ui_local, "defer_load_queue_trigger", False))
            return
        try:
            from gradio.context import LocalContext
        except Exception:
            raise RuntimeError("WanGP WebUI session requires an active Gradio callback context.")
        try:
            blocks = LocalContext.blocks.get(None)
            request_wrapper = LocalContext.request.get(None)
        except LookupError as exc:
            raise RuntimeError("WanGP WebUI session requires an active Gradio callback context.") from exc
        session_hash = getattr(request_wrapper, "session_hash", None) if request_wrapper is not None else None
        request = getattr(request_wrapper, "request", request_wrapper)
        if blocks is None or request is None or not session_hash:
            raise RuntimeError("WanGP WebUI session requires a live Gradio request with a session hash.")
        session._gradio_webui_context = {
            "blocks": blocks,
            "request": request,
            "session_hash": session_hash,
            "load_queue_fn_index": self._resolve_trigger_fn_index(blocks, session_hash, "load_queue_action", "change"),
            "abort_fn_index": self._resolve_abort_fn_index(blocks, session_hash),
            "defer_load_queue_trigger": self._defer_load_queue_trigger or bool(getattr(self._ui_local, "defer_load_queue_trigger", False)),
        }

    def _set_defer_load_queue_trigger(self, value: bool) -> None:
        self._defer_load_queue_trigger = bool(value)

    def _capture_current_gradio_context(self) -> dict[str, Any]:
        try:
            from gradio.context import LocalContext
        except Exception:
            raise RuntimeError("WanGP WebUI session requires an active Gradio callback context.")
        try:
            blocks = LocalContext.blocks.get(None)
            blocks_config = LocalContext.blocks_config.get(None)
            renderable = LocalContext.renderable.get(None)
            render_block = LocalContext.render_block.get(None)
            in_event_listener = LocalContext.in_event_listener.get(False)
            event_id = LocalContext.event_id.get(None)
            request_wrapper = LocalContext.request.get(None)
            progress = LocalContext.progress.get(None)
        except LookupError as exc:
            raise RuntimeError("WanGP WebUI session requires an active Gradio callback context.") from exc
        session_hash = getattr(request_wrapper, "session_hash", None) if request_wrapper is not None else None
        request = getattr(request_wrapper, "request", request_wrapper)
        if blocks is None or request is None or not session_hash:
            raise RuntimeError("WanGP WebUI session requires a live Gradio request with a session hash.")
        return {
            "blocks": blocks,
            "blocks_config": blocks_config,
            "renderable": renderable,
            "render_block": render_block,
            "in_event_listener": in_event_listener,
            "event_id": event_id,
            "request": request,
            "progress": progress,
            "session_hash": session_hash,
            "load_queue_fn_index": self._resolve_trigger_fn_index(blocks, session_hash, "load_queue_action", "change"),
            "abort_fn_index": self._resolve_abort_fn_index(blocks, session_hash),
            "defer_load_queue_trigger": self._defer_load_queue_trigger or bool(getattr(self._ui_local, "defer_load_queue_trigger", False)),
        }

    @staticmethod
    def _capture_progress_callback_context() -> dict[str, Any]:
        try:
            from gradio.context import LocalContext
        except Exception as exc:
            raise RuntimeError("WanGP progress callbacks require an active Gradio callback context.") from exc
        return {
            "blocks": LocalContext.blocks.get(None),
            "blocks_config": LocalContext.blocks_config.get(None),
            "renderable": LocalContext.renderable.get(None),
            "render_block": LocalContext.render_block.get(None),
            "in_event_listener": LocalContext.in_event_listener.get(False),
            "event_id": LocalContext.event_id.get(None),
            "request": LocalContext.request.get(None),
            "progress": LocalContext.progress.get(None),
        }

    @staticmethod
    @contextlib.contextmanager
    def _push_callback_context(context: dict[str, Any]):
        try:
            from gradio.context import LocalContext
        except Exception:
            yield
            return
        tokens = []
        mapping = {
            LocalContext.blocks: context.get("blocks"),
            LocalContext.blocks_config: context.get("blocks_config"),
            LocalContext.renderable: context.get("renderable"),
            LocalContext.render_block: context.get("render_block"),
            LocalContext.in_event_listener: context.get("in_event_listener", False),
            LocalContext.event_id: context.get("event_id"),
            LocalContext.request: context.get("request"),
            LocalContext.progress: context.get("progress"),
        }
        try:
            for var, value in mapping.items():
                tokens.append((var, var.set(value)))
            yield
        finally:
            for var, token in reversed(tokens):
                var.reset(token)

    @staticmethod
    def _resolve_main_bridge_component(elem_id: str):
        try:
            from gradio.context import Context, get_blocks_context
        except Exception as exc:
            raise RuntimeError(f"WanGP WebUI bridge component '{elem_id}' is unavailable outside the Gradio build context.") from exc
        blocks_context = get_blocks_context()
        if blocks_context is None and getattr(Context, "root_block", None) is not None:
            blocks_context = Context.root_block.default_config
        blocks = getattr(blocks_context, "blocks", None)
        if not isinstance(blocks, dict):
            raise RuntimeError(f"WanGP WebUI bridge component '{elem_id}' was not found in the current Blocks tree.")
        for component in blocks.values():
            if getattr(component, "elem_id", None) == elem_id:
                return component
        raise RuntimeError(f"WanGP WebUI bridge component '{elem_id}' could not be resolved.")

    @staticmethod
    def _resolve_trigger_fn_index(blocks, session_hash: str, api_name: str, event_name: str) -> int:
        session_state = blocks.state_holder[session_hash]
        for block_fn in session_state.blocks_config.fns.values():
            targets = getattr(block_fn, "targets", ()) or ()
            if getattr(block_fn, "api_name", None) == api_name and any(target[1] == event_name for target in targets):
                return int(getattr(block_fn, "_id"))
        raise RuntimeError(f"WanGP WebUI trigger '{api_name}' was not found.")

    @staticmethod
    def _resolve_abort_fn_index(blocks, session_hash: str) -> int:
        session_state = blocks.state_holder[session_hash]
        for block_fn in session_state.blocks_config.fns.values():
            targets = getattr(block_fn, "targets", ()) or ()
            api_name = str(getattr(block_fn, "api_name", "") or "")
            if api_name.startswith("abort_generation") and any(target[1] == "change" for target in targets):
                return int(getattr(block_fn, "_id"))
        raise RuntimeError("WanGP WebUI abort trigger was not found.")


def create_gradio_webui_session(plugin, *, init_fn, session_kwargs: dict[str, Any] | None = None) -> GradioWanGPSession:
    return GradioWanGPSession.for_plugin(plugin, init_fn=init_fn, session_kwargs=session_kwargs)


def create_gradio_progress_callbacks(progress) -> GradioProgressCallbacks:
    return GradioProgressCallbacks(progress)
