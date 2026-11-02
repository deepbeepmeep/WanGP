# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import json
import os
import re
import secrets
import shutil
import threading
import time
import traceback
from dataclasses import dataclass
from typing import Any, Callable

import gradio as gr
from shared.utils.config_store import config_lock, write_config

from shared.deepy.config import (
    DEEPY_CONTEXT_TOKENS_DEFAULT,
    DEEPY_CONTEXT_TOKENS_KEY,
    DEEPY_ZERO_CUSTOM_SYSTEM_PROMPT_KEY,
    DEEPY_ENABLED_KEY,
    DEEPY_PRIME_CUSTOM_SYSTEM_PROMPT_KEY,
    DEEPY_PRIME_MCP_SERVERS_KEY,
    DEEPY_SESSION_GALLERY_MEDIA_MODE_DEFAULT,
    DEEPY_SESSION_GALLERY_MEDIA_MODE_KEY,
    DEEPY_MULTI_SESSION_DEFAULT,
    DEEPY_MULTI_SESSION_KEY,
    DEEPY_TYPE_KEY,
    DEEPY_TYPE_PRIME,
    DEEPY_VRAM_MODE_KEY,
    DEEPY_VRAM_MODE_UNLOAD,
    DEEPY_QWEN_ENHANCER_IDS,
    deepy_available,
    deepy_requirement_error,
    deepy_requirement_met,
    normalize_deepy_enabled,
    normalize_deepy_context_tokens,
    normalize_deepy_type,
    normalize_deepy_prime_mcp_servers,
    normalize_deepy_session_gallery_media_mode,
    normalize_deepy_multi_session,
    normalize_deepy_session_mode,
    DEEPY_MULTI_SESSION_DEDICATED,
    normalize_deepy_vram_mode,
    set_deepy_runtime_config,
)
from shared.deepy import PRIME_SYSTEM_PROMPT, ZERO_SYSTEM_PROMPT
from shared.deepy import ui_settings as deepy_ui_settings
from shared.deepy import media_registry, session_store
from shared.deepy.debug_bootstrap import deepy_log_scope
from shared.deepy.publication import DeepyPublicationQueue
from shared.deepy.engine import (
    AssistantEngine,
    AssistantRuntimeHooks,
    AssistantSessionState,
    begin_assistant_replay_turn,
    begin_assistant_turn,
    build_interruption_notice,
    clear_assistant_pause,
    clear_assistant_steering,
    clear_assistant_session,
    get_or_create_assistant_session,
    mark_assistant_turn_message,
    record_interruption_history,
    request_assistant_interrupt,
    request_assistant_pause,
    request_assistant_steering,
    request_assistant_reset,
    resume_assistant,
    set_assistant_debug,
    set_assistant_tool_ui_settings,
    DeepyZeroTools,
)
from shared.gradio import assistant_chat
from shared.utils.thread_utils import AsyncStream, async_run_in, promote_async_task, promote_async_tasks
from shared.remote_llm.config import is_remote_engine, resolve_role_engine
from shared.utils.process_locks import acquire_GPU_ressources, release_GPU_ressources, unregister_GPU_resident


_DEEPY_GPU_PROCESS_ID = "deepy"
_DEEPY_DISABLED_TEXT = "Deepy is disabled in Configuration > Deepy."
AGENT_VISION_PROCESS, AGENT_VISION_NAME = "agent_vision", "Agent Visual Inspection"
_AGENT_VISION_LOCK = threading.Lock()  # one GPU owner id: agents' questions take turns
_CHAT_BATCH_MAX_EVENTS = 16
_CHAT_BATCH_MAX_SECONDS = 0.05
DEEPY_STREAM_TRACE_ENV = "WAN2GP_DEEPY_STREAM_TRACE"
_DEEPY_STREAM_TRACE_ENABLED = str(os.environ.get(DEEPY_STREAM_TRACE_ENV, "0")).strip().lower() in {"1", "true", "yes", "y", "on"}


def _drain_chat_output_batch(output_queue, first_payload: str) -> str:
    payloads = [first_payload]
    time.sleep(_CHAT_BATCH_MAX_SECONDS)
    while len(payloads) < _CHAT_BATCH_MAX_EVENTS:
        next_item = output_queue.top()
        if not isinstance(next_item, tuple) or len(next_item) < 1 or next_item[0] != "chat_output":
            break
        _cmd, next_payload = output_queue.pop()
        payloads.append(next_payload)
    return assistant_chat.build_event_batch(payloads)


@dataclass(slots=True)
class DeepyDeps:
    get_server_config: Callable[[], dict[str, Any]]
    get_server_config_filename: Callable[[], str]
    get_verbose_level: Callable[[], int]
    resolve_prompt_enhancer_settings: Callable[..., tuple[Any, int]]
    get_state_model_type: Callable[[Any], str]
    get_model_def: Callable[[str], Any]
    ensure_prompt_enhancer_loaded: Callable[..., tuple[Any, Any]]
    unload_prompt_enhancer_runtime: Callable[[], None]
    get_image_caption_model: Callable[[], Any]
    get_image_caption_processor: Callable[[], Any]
    get_enhancer_offloadobj: Callable[[], Any]
    acquire_gpu: Callable[[Any], None]
    release_gpu: Callable[..., None]
    register_gpu_resident: Callable[..., None]
    clear_gpu_resident: Callable[[Any], None]
    get_new_refresh_id: Callable[[], Any]
    get_gen_info: Callable[[Any], dict[str, Any]]
    get_processed_queue: Callable[[dict[str, Any]], dict[str, tuple[list[Any], list[Any]]]]
    get_output_filepath: Callable[[str, bool, bool], str]
    record_file_metadata: Callable[..., Any]
    exec_prompt_enhancer_engine: Callable[..., Any]
    clear_queue_action: Callable[[Any], Any]


def _unload_prompt_enhancer_runtime(prompt_enhancer_image_caption_model, prompt_enhancer_llm_model) -> None:
    from shared.prompt_enhancer import unload_prompt_enhancer_models

    unload_prompt_enhancer_models(prompt_enhancer_image_caption_model, prompt_enhancer_llm_model)


class DeepyController:
    def __init__(self, deps: DeepyDeps):
        self._deps = deps
        self._active_assistant_session: Any | None = None
        self._queue_state_lock = threading.RLock()
        self._session_mode = normalize_deepy_session_mode(self._server_config().get(DEEPY_MULTI_SESSION_KEY, DEEPY_MULTI_SESSION_DEFAULT))
        self._multi_session_enabled = normalize_deepy_multi_session(self._session_mode)
        self.workspace_host = None
        self._multi_session_latched = False
        # Called with (submission_id, phase, info) when a submitted request starts, finishes or is cancelled.
        self.request_listener = None

    def get_verbose_level(self) -> int:
        try:
            return int(self._deps.get_verbose_level() or 0)
        except Exception:
            return 0

    def _debug_log(self, message: str) -> None:
        if self.get_verbose_level() >= 2:
            with deepy_log_scope(start_if_needed=True):
                print(f"[AssistantController] {message}")

    def _sync_debug_enabled(self) -> bool:
        try:
            debug_enabled = int(self._deps.get_verbose_level() or 0) >= 2
        except Exception:
            debug_enabled = False
        set_assistant_debug(debug_enabled)
        return debug_enabled

    def _server_config(self) -> dict[str, Any]:
        return self._deps.get_server_config() or {}

    def _persist_tool_ui_settings(self, normalized: dict[str, Any]) -> None:
        server_config = self._server_config()
        with config_lock:
            deepy_ui_settings.store_assistant_tool_ui_settings(server_config, normalized)
            self._write_server_config(server_config)
        gr.Info("New Deepy Setting Saved")

    def _write_server_config(self, server_config: dict[str, Any]) -> None:
        server_config_filename = str(self._deps.get_server_config_filename() or "").strip()
        with config_lock:
            set_deepy_runtime_config(server_config, server_config_filename)
            if len(server_config_filename) > 0:
                write_config(server_config, server_config_filename)

    def set_compact_actions(self, enabled: bool) -> None:
        with config_lock:
            server_config = self._server_config()
            server_config["deepy_compact_actions"] = enabled
            self._write_server_config(server_config)

    def _session_environment(self, session) -> dict[str, Any]:
        servers = normalize_deepy_prime_mcp_servers(self._server_config().get(DEEPY_PRIME_MCP_SERVERS_KEY, {})) if self.get_deepy_type() == DEEPY_TYPE_PRIME else {}
        return {
            "schema_version": 1,
            "skills": list(session.active_skills or []),
            "mcp_servers": [{"name": name, "transport": config["transport"], "config_reference": f"{DEEPY_PRIME_MCP_SERVERS_KEY}.{name}"} for name, config in servers.items()],
        }

    def get_session_ui_settings(self) -> dict[str, Any]:
        settings = deepy_ui_settings.get_persisted_assistant_session_ui_settings(self._server_config())
        enabled = normalize_deepy_multi_session(settings["multi_session"])
        settings["reset_mode"] = session_store.RESET_MODE_NEW if enabled else session_store.RESET_MODE_RESET
        if not enabled:
            settings["gallery_media_mode"] = session_store.GALLERY_MEDIA_LINK
        settings.update(effective_multi_session=self._multi_session_enabled, effective_session_mode=self._session_mode, restart_required=settings["multi_session"] != self._session_mode, unsaved_changes=False)
        return settings

    def update_session_ui_settings(self, state, *, multi_session, reset_mode, gallery_media_mode, persist=False) -> dict[str, Any]:
        workspace_locked = self.workspace_host._deepy_workspace_locked() if self.workspace_host is not None else False
        requested_multi_session = normalize_deepy_multi_session(multi_session)
        requested_mode = normalize_deepy_session_mode(multi_session)
        persisted = deepy_ui_settings.get_persisted_assistant_session_ui_settings(self._server_config())
        if not self._multi_session_latched:
            self._multi_session_enabled = requested_multi_session
            self._session_mode = requested_mode
        normalized_gallery_mode = normalize_deepy_session_gallery_media_mode(gallery_media_mode) if requested_multi_session else session_store.GALLERY_MEDIA_LINK
        normalized = {
            "multi_session": requested_mode,
            "effective_multi_session": self._multi_session_enabled,
            "effective_session_mode": self._session_mode,
            "restart_required": requested_mode != self._session_mode,
            "reset_mode": session_store.RESET_MODE_NEW if requested_multi_session else session_store.RESET_MODE_RESET,
            "gallery_media_mode": normalized_gallery_mode,
            "unsaved_changes": requested_mode != persisted["multi_session"] or (requested_multi_session and normalized_gallery_mode != persisted["gallery_media_mode"]),
        }
        session = get_or_create_assistant_session(state)
        session.gallery_media_mode = normalize_deepy_session_gallery_media_mode(gallery_media_mode) if self._multi_session_enabled else session_store.GALLERY_MEDIA_LINK
        if session.storage_session_id and not session.worker_active:
            session_store.schedule_autosave(session)
        if persist:
            server_config = self._server_config()
            with config_lock:
                deepy_ui_settings.store_assistant_session_ui_settings(server_config, multi_session=requested_mode, reset_mode=normalized["reset_mode"], gallery_media_mode=normalized["gallery_media_mode"])
                self._write_server_config(server_config)
            normalized["unsaved_changes"] = False
        if self.workspace_host is not None:
            self.workspace_host.publish_workspaces()
            if self.workspace_host._deepy_workspace_locked() != workspace_locked:
                self.workspace_host.publish('gallery', self.workspace_host.gallery_snapshot())
        return normalized

    def multi_session_enabled(self) -> bool:
        return self._multi_session_enabled

    def dedicated_workspace_enabled(self) -> bool:
        return self._session_mode == DEEPY_MULTI_SESSION_DEDICATED

    def _complete_session_reset(self, state, session, reset_mode: str) -> None:
        reset_mode = session_store.RESET_MODE_NEW if self._multi_session_enabled else session_store.RESET_MODE_RESET
        started = time.perf_counter()
        with session.turn_lock:
            locked = time.perf_counter()
            if session.storage_session_id:
                session_store.flush_session(session)
            saved = time.perf_counter()
            if reset_mode == session_store.RESET_MODE_NEW or not self._multi_session_enabled:
                session_store.start_new_session(session, save_current=False)
            rotated = time.perf_counter()
            self.release_vram(state, True, discard_runtime_snapshot=True, preserve_reset_base=True)
            released = time.perf_counter()
            if not self._multi_session_enabled:
                session_store.ensure_mono_session_workspace(session.chat_session_id)
            elif reset_mode == session_store.RESET_MODE_RESET and session.storage_session_id:
                session_store.reset_session_files(session)
            session.pending_reset_mode = ""
        finished = time.perf_counter()
        if finished - started >= 1:
            print(f"[Deepy][Session reset] total={finished-started:.3f}s lock={locked-started:.3f}s save={saved-locked:.3f}s rotate={rotated-saved:.3f}s release/reset runtime={released-rotated:.3f}s finalize={finished-released:.3f}s", flush=True)

    def _reset_foreign_active_session(self, session) -> bool:
        active_session = self._active_assistant_session
        if active_session is None or active_session is session:
            return False
        request_assistant_reset(active_session)
        assistant_chat.reset_session_chat(active_session)
        active_session.chat_html = ""
        return True

    @staticmethod
    def _find_next_queued_user_message_id(session) -> str:
        for record in list(session.chat_transcript or []):
            if not isinstance(record, dict):
                continue
            if str(record.get("role", "")).strip().lower() != "user":
                continue
            if not bool(record.get("queued", False)):
                continue
            message_id = str(record.get("id", "") or "").strip()
            if len(message_id) > 0:
                return message_id
        return ""

    def _cancel_next_queued_request(self, session) -> str:
        with self._queue_state_lock:
            if int(session.queued_job_count or 0) <= 0:
                return ""
            message_id = self._find_next_queued_user_message_id(session)
            if len(message_id) == 0:
                return ""
            user_text = assistant_chat.get_message_content(session, message_id)
            interruption_notice = build_interruption_notice(user_text)
            session.queued_job_count = max(0, int(session.queued_job_count or 0) - 1)
            session.queued_cancel_count = max(0, int(session.queued_cancel_count or 0)) + 1
            assistant_chat._find_message(session, message_id)["queued"] = False
            assistant_chat.set_message_badge(session, message_id, "Interrupted")
            record_interruption_history(session, user_text, interruption_notice)
            return user_text

    def _apply_queued_request_action(self, state, action_payload):
        session = get_or_create_assistant_session(state)
        try:
            payload = json.loads(str(action_payload or ""))
        except (TypeError, ValueError):
            return gr.update(), gr.update(), gr.update(), gr.update()
        if not isinstance(payload, dict):
            return gr.update(), gr.update(), gr.update(), gr.update()
        action = str(payload.get("action", "") or "").strip().lower()
        message_id = str(payload.get("message_id", "") or "").strip()
        text = str(payload.get("text", "") or "").strip()
        if action not in {"edit", "remove", "steer"} or len(message_id) == 0 or (action == "edit" and len(text) == 0):
            return gr.update(), gr.update(), gr.update(), gr.update()
        with self._queue_state_lock:
            message_id = assistant_chat.resolve_message_id(session, message_id)
            record = assistant_chat._find_message(session, message_id)
            if record is None or str(record.get("role", "")).strip() != "user" or str(record.get("badge", "")).strip() != "Queued":
                return gr.update(), gr.update(), gr.update(), gr.update()
            if action == "edit":
                chat_event = assistant_chat.set_user_message_content(session, message_id, text)
            elif action == "steer":
                task = session.queued_task_handles.get(message_id)
                if task is None or not promote_async_task("assistant", task):
                    return gr.update(), gr.update(), gr.update(), gr.update()
                status_text = "Steering accepted. Waiting for context compaction to finish..." if session.assistant_compaction_active else "Steering accepted. Applying this request at the current boundary..."
                chat_event = assistant_chat.steer_queued_message(session, message_id, status_text=status_text)
                if session.worker_active and isinstance(session.current_turn, dict):
                    request_assistant_steering(session)
            else:
                session.cancelled_queued_message_ids.add(message_id)
                session.queued_job_count = max(0, int(session.queued_job_count or 0) - 1)
                chat_event = assistant_chat.remove_message(session, message_id)
            control_queue = session.control_queue if session.control_queue is not None and (session.worker_active or session.queued_job_count > 0) else None
            if control_queue is not None and chat_event is not None:
                control_queue.push("chat_output", chat_event)
        self._debug_log(f"Queued request {action} user_message_id={message_id} queued_jobs={int(session.queued_job_count or 0)}")
        return chat_event if chat_event is not None and control_queue is None else gr.update(), gr.update(), gr.update(), gr.update()

    def _cancel_active_prime_job(self, session, action: str) -> str:
        cancel_active_job = getattr(session.prime_toolbox, "cancel_active_job", None)
        if not callable(cancel_active_job):
            return ""
        try:
            return str(cancel_active_job() or "").strip()
        except Exception as exc:
            self._debug_log(f"Immediate MCP job cancellation failed after {action}: {exc}")
            return ""

    def is_available(self) -> bool:
        return deepy_available(self._server_config())

    def requirement_error_text(self) -> str:
        server_config = self._server_config()
        if not normalize_deepy_enabled(server_config.get(DEEPY_ENABLED_KEY, 0)):
            return _DEEPY_DISABLED_TEXT
        if not deepy_requirement_met(server_config):
            return deepy_requirement_error(server_config)
        return ""

    def get_vram_mode(self) -> str:
        server_config = self._server_config()
        if not normalize_deepy_enabled(server_config.get(DEEPY_ENABLED_KEY, 0)) or is_remote_engine(resolve_role_engine(server_config, "deepy")):
            return DEEPY_VRAM_MODE_UNLOAD
        return normalize_deepy_vram_mode(server_config.get(DEEPY_VRAM_MODE_KEY, DEEPY_VRAM_MODE_UNLOAD))

    def get_deepy_type(self) -> str:
        return normalize_deepy_type(self._server_config().get(DEEPY_TYPE_KEY, ""))

    def _ensure_vision_loaded(self, override_profile=-1, **loading_kwargs):
        self._deps.ensure_prompt_enhancer_loaded(override_profile=override_profile, **loading_kwargs)
        image_caption_model = self._deps.get_image_caption_model()
        image_caption_processor = self._deps.get_image_caption_processor()
        if image_caption_model is None or image_caption_processor is None:
            raise gr.Error("Prompt enhancer vision runtime is not available.")
        return image_caption_model, image_caption_processor

    def _unload_weights(self) -> None:
        enhancer_offloadobj = self._deps.get_enhancer_offloadobj()
        if enhancer_offloadobj is not None:
            enhancer_offloadobj.unload_all()

    def answer_visual_query(self, state, media_record, question: str, frame_no: int | None = None, max_image_edge: int | None = None) -> dict[str, Any]:
        """The prompt enhancer's Qwen VL model answers an external agent's visual question directly from the images, as Deepy's inspection does.
        It borrows the GPU like any WanGP process: a running generation pauses at its next step; Deepy's VRAM mode decides whether the model stays loaded."""
        server_config = self._server_config()
        if server_config.get("enhancer_enabled", 0) not in DEEPY_QWEN_ENHANCER_IDS:  # the model the prompt enhancer loads
            raise ValueError("WanGP's Visual Inspector uses the prompt enhancer: set it to a Qwen3.5VL or Qwen3.8VL mode in Configuration > Extensions.")
        keep_loaded = self.get_vram_mode() != DEEPY_VRAM_MODE_UNLOAD
        engine = AssistantEngine(AssistantSessionState(), AssistantRuntimeHooks(
            acquire_gpu=lambda: acquire_GPU_ressources(state, AGENT_VISION_PROCESS, AGENT_VISION_NAME),
            release_gpu=lambda **_kwargs: release_GPU_ressources(state, AGENT_VISION_PROCESS, process_name=AGENT_VISION_NAME),
            register_gpu_resident=lambda *_args, **_kwargs: None,
            clear_gpu_resident=lambda: unregister_GPU_resident(state, AGENT_VISION_PROCESS),
            ensure_loaded=self._deps.ensure_prompt_enhancer_loaded,
            unload_runtime=self._deps.unload_prompt_enhancer_runtime,
            unload_weights=self._unload_weights,
            ensure_vision_loaded=self._ensure_vision_loaded,
            get_offload_manager=self._deps.get_enhancer_offloadobj,
        ), None, lambda *_args: None, debug_enabled=self._sync_debug_enabled(), thinking_enabled=False, vram_mode=self.get_vram_mode())
        with _AGENT_VISION_LOCK:
            try:
                return engine._run_visual_query(media_record, question, frame_no, max_image_edge)
            finally:
                if engine._gpu_acquired:
                    release_callback = self._build_preload_release_callback()
                    try:
                        if not keep_loaded:
                            release_callback()
                    finally:  # a loaded model is released by the next GPU owner, as Deepy's own
                        release_GPU_ressources(state, AGENT_VISION_PROCESS, keep_resident=keep_loaded, process_name=AGENT_VISION_NAME, release_vram_callback=release_callback if keep_loaded else None)

    def _build_preload_release_callback(self) -> Callable[[], None]:
        def _release_preloaded_runtime() -> None:
            try:
                self._deps.unload_prompt_enhancer_runtime()
            finally:
                self._unload_weights()

        return _release_preloaded_runtime

    def release_vram(self, state, clear_session_state = False, discard_runtime_snapshot = False, preserve_reset_base = False):
        session = get_or_create_assistant_session(state)
        if clear_session_state:
            with self._queue_state_lock:
                worker_active = bool(session.worker_active)
                if worker_active or session.queued_job_count > 0 or session.control_queue is not None:
                    session.discard_runtime_snapshot_on_release = bool(discard_runtime_snapshot)
                    request_assistant_reset(session)
            if worker_active:
                self._debug_log("Waiting for the active Deepy worker before clearing its runtime.")
                session.worker_idle_event.wait()
        release_callback = session.release_vram_callback
        session.release_vram_callback = None
        session.discard_runtime_snapshot_on_release = bool(discard_runtime_snapshot)
        self._deps.clear_gpu_resident(state)
        try:
            if callable(release_callback):
                release_callback()
        finally:
            if discard_runtime_snapshot:
                session.runtime_snapshot = None
                if len(session.rendered_token_ids) == 0:
                    session.pending_replay_reason = ""
            session.discard_runtime_snapshot_on_release = False
        if clear_session_state:
            if session.storage_session_id and session.storage_deepy_type != self.get_deepy_type():
                session_store.start_new_session(session)
            reset_to_base = session.reset_to_base_callback
            preserved_reset_base = bool(reset_to_base()) if preserve_reset_base and callable(reset_to_base) else False
            if not preserved_reset_base:
                clear_assistant_session(session)
            session.interrupt_requested = False
            session.drop_state_requested = False
            session.control_queue = None
            session.worker_idle_event.set()

    def preload_cli_runtime(self, state, override_profile=-1) -> dict[str, Any]:
        self._sync_debug_enabled()
        self._deps.clear_gpu_resident(state)
        self._deps.acquire_gpu(state)
        keep_resident = False
        warmed_vllm = False
        try:
            model, _tokenizer = self._deps.ensure_prompt_enhancer_loaded(override_profile=override_profile)
            from shared.prompt_enhancer import qwen35_text

            if qwen35_text._use_vllm_prompt_enhancer(model):
                model._prompt_enhancer_min_model_len_hint = normalize_deepy_context_tokens(self._server_config().get(DEEPY_CONTEXT_TOKENS_KEY, DEEPY_CONTEXT_TOKENS_DEFAULT))
                engine = qwen35_text._get_or_create_vllm_engine(model, usage_mode="assistant")
                engine.reserve_runtime(prompt_len=64, max_tokens=1, cfg_scale=1.0)
                engine._ensure_llm()
                llm = getattr(engine, "_llm", None)
                if llm is None:
                    raise RuntimeError("Assistant LM engine runtime is not available.")
                llm.model_runner.ensure_runtime_ready()
                engine.release_runtime_allocations()
                warmed_vllm = True
            keep_resident = True
            return {"status": "ready", "warmed_vllm": warmed_vllm}
        finally:
            self._deps.release_gpu(
                state,
                keep_resident=keep_resident,
                release_vram_callback=self._build_preload_release_callback() if keep_resident else None,
                force_release_on_acquire=True,
            )

    def update_tool_ui_settings(self, state, *, auto_cancel_queue_tasks=None, separate_requests_with_empty_line=None, use_template_properties=None, model_speed=None, model_size=None, width=None, height=None, num_frames=None, audio_duration=None, seed=None, persist=False, **variants):
        """variants: the template of each task, given as its ui_key (ui_settings.TEMPLATE_TOOL_UI_KEY)."""
        session = get_or_create_assistant_session(state)
        normalized = set_assistant_tool_ui_settings(
            session,
            auto_cancel_queue_tasks=auto_cancel_queue_tasks,
            separate_requests_with_empty_line=separate_requests_with_empty_line,
            use_template_properties=use_template_properties,
            model_speed=model_speed,
            model_size=model_size,
            width=width,
            height=height,
            num_frames=num_frames,
            audio_duration=audio_duration,
            seed=seed,
            **variants,
        )
        if persist:
            self._persist_tool_ui_settings(normalized)
        return normalized

    @staticmethod
    def _split_request_blocks(text: str) -> list[str]:
        normalized = str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        if len(normalized) == 0:
            return []
        requests = []
        current_lines = []
        for raw_line in normalized.split("\n"):
            stripped_line = raw_line.strip()
            if not stripped_line or re.fullmatch(r"[-=_]{3,}", stripped_line):
                if current_lines:
                    requests.append("\n".join(current_lines).strip())
                    current_lines = []
                continue
            current_lines.append(raw_line.rstrip())
        if current_lines:
            requests.append("\n".join(current_lines).strip())
        return [request for request in requests if len(request) > 0]

    def _expand_assistant_requests(self, session, ask_request: Any) -> list[str]:
        normalized_request = str(ask_request or "").strip()
        if len(normalized_request) == 0:
            return []
        tool_ui_settings = deepy_ui_settings.normalize_assistant_tool_ui_settings(**session.tool_ui_settings) if isinstance(session.tool_ui_settings, dict) and len(session.tool_ui_settings) > 0 else deepy_ui_settings.get_persisted_assistant_tool_ui_settings(self._server_config())
        if not tool_ui_settings["separate_requests_with_empty_line"]:
            return [normalized_request]
        return self._split_request_blocks(normalized_request) or [normalized_request]

    def _prepare_session_request_locked(self, session, ask_request: str) -> None:
        self._multi_session_latched = True
        # A plugin session keeps its own folder even when saved sessions are disabled.
        if not self._multi_session_enabled and not session.storage_session_id:
            session.safe_checkpoint_callback = None
            return
        default_gallery_mode = normalize_deepy_session_gallery_media_mode(self._server_config().get(DEEPY_SESSION_GALLERY_MEDIA_MODE_KEY, DEEPY_SESSION_GALLERY_MEDIA_MODE_DEFAULT))
        gallery_mode = session.gallery_media_mode if session.storage_session_id else default_gallery_mode
        session_store.ensure_session(session, ask_request, self.get_deepy_type(), gallery_mode, self._session_environment(session))

    def begin_direct_request(self, state, ask_request: str):
        session = get_or_create_assistant_session(state)
        with self._queue_state_lock:
            self._prepare_session_request_locked(session, ask_request)
            user_message_id, user_event = assistant_chat.add_user_message(session, ask_request, queued=False)
            begin_assistant_turn(session, user_message_id, ask_request)
            if self._multi_session_enabled:
                session_store.schedule_autosave(session)
        if self.workspace_host is not None:
            self.workspace_host.ensure_session_workspace()
        return user_message_id, user_event

    def _queue_assistant_request(self, state, session, output_queue, ask_request: str, queued_epoch: int, *, queued: bool, client_submission_id: str = "", assistant_badge: str = "", replay_action: dict[str, Any] | None = None, media: list[dict[str, str]] | None = None, author: str = "You"):
        raw_send_cmd = output_queue.push
        assistant_badge = str(assistant_badge or "").strip()
        replay = None if replay_action is None else dict(replay_action)

        def notify(phase, **info):
            if client_submission_id and self.request_listener is not None:
                self.request_listener(client_submission_id, phase, info)

        def send_cmd(cmd, data=None):
            if queued_epoch != session.chat_epoch and cmd in {"chat_output", "load_queue_trigger", "refresh_gallery", "error"}:
                return
            if _DEEPY_STREAM_TRACE_ENABLED and self.get_verbose_level() >= 2 and cmd == "chat_output" and isinstance(data, str):
                try:
                    event = json.loads(data).get("event", {})
                    event_type = str(event.get("type", ""))
                    if event_type in {"sync", "reset", "upsert_message", "remove_message", "upsert_block", "append_block_text", "replace_block_text", "finalize_block", "remove_block"}:
                        self._debug_log(f"Chat event type={event_type} sequence={event.get('sequence', '-')} sequence_start={event.get('sequence_start', '-')} revision={event.get('revision', '-')} message={event.get('message_id', event.get('message', {}).get('id', '-'))} block={event.get('block_id', '-')}")
                except Exception:
                    pass
            raw_send_cmd(cmd, data)

        with self._queue_state_lock:
            session.queued_job_count += 1
            if replay is None:
                self._prepare_session_request_locked(session, ask_request)
                user_message_id, _user_event = assistant_chat.add_user_message(session, ask_request, queued=queued, client_submission_id=client_submission_id, media=media, author=author)
                if assistant_badge:
                    user_record = assistant_chat._find_message(session, user_message_id)
                    user_record["assistant_badge"] = assistant_badge
                    user_record["badge"] = ""
            else:
                user_message_id = str(replay["user_message_id"])
            self._debug_log(f"Request enqueued user_message_id={user_message_id} queued={bool(queued)} queued_jobs={int(session.queued_job_count or 0)}")
            session_store.schedule_autosave(session)

        def queue_worker_func():
            with deepy_log_scope(start_if_needed=True):
                started_turn = False
                active_request = ask_request if replay is None else str(replay["user_text"])
                runtime_assistant_badge = "" if replay is None else str(replay["assistant_badge"])
                with self._queue_state_lock:
                    stale_request = queued_epoch != session.chat_epoch
                    targeted_cancelled = not stale_request and user_message_id in session.cancelled_queued_message_ids
                    cancelled_request = not stale_request and (targeted_cancelled or int(session.queued_cancel_count or 0) > 0)
                    if stale_request:
                        if session.control_queue is output_queue:
                            session.control_queue = None
                    elif cancelled_request:
                        session.queued_task_handles.pop(user_message_id, None)
                        if targeted_cancelled:
                            session.cancelled_queued_message_ids.discard(user_message_id)
                        else:
                            session.queued_cancel_count = max(0, int(session.queued_cancel_count or 0) - 1)
                            assistant_chat.set_message_badge(session, user_message_id, "Interrupted")
                        cancelled_sync = assistant_chat.build_sync_event(session)
                        cancelled_has_more_work = session.queued_job_count > 0
                    else:
                        session.queued_task_handles.pop(user_message_id, None)
                        if replay is None:
                            user_record = assistant_chat._find_message(session, user_message_id)
                            runtime_assistant_badge = str(user_record.get("assistant_badge", "") or "").strip()
                            active_request = assistant_chat.get_message_content(session, user_message_id)
                        session.queued_job_count = max(0, session.queued_job_count - 1)
                        session.interrupt_requested = False
                        clear_assistant_pause(session)
                        clear_assistant_steering(session)
                        session.control_queue = output_queue
                        session.worker_active = True
                        session.worker_idle_event.clear()
                        self._active_assistant_session = session
                        if replay is None:
                            begin_assistant_turn(session, user_message_id, active_request, assistant_badge=runtime_assistant_badge)
                        else:
                            begin_assistant_replay_turn(session, replay)
                        started_turn = True
                        starting_sync = None
                        if queued and replay is None:
                            assistant_chat._find_message(session, user_message_id)["queued"] = False
                            assistant_chat.set_message_badge(session, user_message_id, None)
                        if queued or replay is not None:
                            starting_status = {"visible": True, "kind": "queued" if runtime_assistant_badge else "loading", "text": "Steering accepted. Deepy is applying the new instructions..." if runtime_assistant_badge else "Starting Deepy..."}
                            if replay is not None:
                                starting_status = {"visible": True, "kind": "loading", "text": "Replaying the interrupted action..."}
                            starting_sync = assistant_chat.build_sync_event(session, status=starting_status)
                if stale_request:
                    self._debug_log(f"Worker skipped stale request user_message_id={user_message_id} queued_epoch={queued_epoch} chat_epoch={session.chat_epoch}")
                    notify("cancelled")
                    raw_send_cmd("exit", None)
                    return
                if cancelled_request:
                    self._debug_log(f"Worker cancelled queued request user_message_id={user_message_id}")
                    notify("cancelled")
                    raw_send_cmd("chat_output", cancelled_sync)
                    if cancelled_has_more_work:
                        raw_send_cmd("chat_output", assistant_chat.build_status_event("Queued behind the current assistant task.", kind="queued", session=session))
                    else:
                        if self.get_vram_mode() == DEEPY_VRAM_MODE_UNLOAD:
                            self.release_vram(state)
                        raw_send_cmd("chat_output", assistant_chat.build_status_event(None, visible=False, session=session))
                        raw_send_cmd("exit", None)
                    return
                self._debug_log(f"Worker starting user_message_id={user_message_id} queued_jobs={int(session.queued_job_count or 0)}")
                notify("started", user_message_id=user_message_id)
                if starting_sync is not None:
                    send_cmd("chat_output", starting_sync)
                my_tools = self.create_tools(state, send_cmd, session=session)
                try:
                    self._debug_log(f"Prompt enhancer dispatch starting user_message_id={user_message_id}")
                    if replay is None:
                        self._deps.exec_prompt_enhancer_engine(state, "", None, "AK", [active_request], None, None, False, False, 0, None, 3.5, send_cmd, my_tools)
                    else:
                        modes = "AK" if replay["thinking_enabled"] else "A"
                        self.run_assistant_prompt_turn(state, None, modes, [active_request], replay["seed"], send_cmd=send_cmd, tools=my_tools, replay_action=replay)
                except Exception as e:
                    user_action_required = bool(getattr(e, "user_action_required", False))
                    if not user_action_required:
                        traceback.print_exc()
                    error_turn_id = assistant_chat.create_assistant_turn(session)
                    if runtime_assistant_badge:
                        assistant_chat.set_message_badge(session, error_turn_id, runtime_assistant_badge)
                    mark_assistant_turn_message(session, error_turn_id)
                    error_event = assistant_chat.set_assistant_content(session, error_turn_id, str(e) if user_action_required else f"Assistant crashed: {e}")
                    if error_event is not None:
                        send_cmd("chat_output", error_event)
                    send_cmd("chat_output", assistant_chat.build_status_event(None, visible=False, session=session))
                finally:
                    session_store.schedule_autosave(session)
                    with self._queue_state_lock:
                        if self._active_assistant_session is session:
                            self._active_assistant_session = None
                        session.worker_active = False
                        pending_reset_mode = str(session.pending_reset_mode or "")
                        stale_turn = queued_epoch != session.chat_epoch
                        has_more_work = not pending_reset_mode and not stale_turn and session.queued_job_count > 0
                        if not has_more_work and session.control_queue is output_queue:
                            session.control_queue = None
                        clear_assistant_pause(session)
                        pending_steering = has_more_work and any(str(record.get("role", "")).strip() == "user" and bool(record.get("queued", False)) and str(record.get("assistant_badge", "")).strip() == "Steered" for record in session.chat_transcript)
                        final_status = None if not has_more_work else {"visible": True, "kind": "queued", "text": "Steering accepted. Deepy is applying the new instructions..." if pending_steering else "Queued behind the current assistant task."}
                        final_sync = assistant_chat.build_sync_event(session, status=final_status) if has_more_work else None
                        interrupted = bool(session.interrupt_requested)
                        session.interrupt_requested = False
                    if pending_reset_mode:
                        self._complete_session_reset(state, session, pending_reset_mode)
                        raw_send_cmd("chat_output", assistant_chat.build_reset_event(session))
                    elif stale_turn:
                        if started_turn:
                            raw_send_cmd("chat_output", assistant_chat.build_reset_event(session))
                    elif has_more_work:
                        raw_send_cmd("chat_output", final_sync)
                    else:
                        raw_send_cmd("chat_output", assistant_chat.build_status_event(None, visible=False, session=session))
                    self._debug_log(f"Worker finished user_message_id={user_message_id} stale={bool(stale_turn)} has_more_work={bool(has_more_work)} queued_jobs={int(session.queued_job_count or 0)}")
                    if not has_more_work:
                        if self.get_vram_mode() == DEEPY_VRAM_MODE_UNLOAD:
                            self.release_vram(state)
                        if not output_queue.wait_for_chat_publication():
                            self._debug_log(f"Timed out waiting for the final Deepy UI publication for user_message_id={user_message_id}.")
                        raw_send_cmd("exit", None)
                        self._debug_log(f"Terminal publication queued for user_message_id={user_message_id}: {output_queue.metrics()}")
                    session.worker_idle_event.set()
                    notify("finished", user_message_id=user_message_id, interrupted=interrupted or stale_turn or bool(pending_reset_mode))

        task_handle = async_run_in("assistant", queue_worker_func)
        if queued:
            session.queued_task_handles[user_message_id] = task_handle
        return user_message_id, task_handle

    def store_selected_video_time(self, state, current_time):
        gen = self._deps.get_gen_info(state)
        try:
            value = float(current_time)
        except Exception:
            value = None
        gen["selected_video_time"] = None if value is None or value < 0 else value

    def create_tools(self, state, send_cmd, session = None):
        active_session = get_or_create_assistant_session(state) if session is None else session
        if self.get_deepy_type() == DEEPY_TYPE_PRIME:
            from shared.deepy.prime_tools import DeepyPrimeTools

            if active_session.prime_toolbox is not None and self.workspace_host is not None:
                from shared.deepy.filesystem import build_file_access_policy
                workspace_id = self.workspace_host.deepy_workspace_id()
                config = self.workspace_host.workspace_output_config(workspace_id)
                expected_roots = build_file_access_policy(config).output_roots
                current = active_session.prime_toolbox
                if getattr(current._api_session, '_workspace_id', None) != workspace_id or current.file_access_policy.output_roots != expected_roots:
                    current.close()
                    active_session.prime_toolbox = None

            if active_session.prime_toolbox is None:
                gen = self._deps.get_gen_info(state)
                zero_tools = DeepyZeroTools(gen, self._deps.get_processed_queue, send_cmd, session=active_session, get_output_filepath=self._deps.get_output_filepath, record_file_metadata=self._deps.record_file_metadata, get_server_config=self._server_config)
                active_session.prime_toolbox = DeepyPrimeTools(state, send_cmd, active_session, zero_tools=zero_tools)
            else:
                active_session.prime_toolbox.bind_turn(state, send_cmd)
            return active_session.prime_toolbox
        if active_session.prime_toolbox is not None:
            active_session.prime_toolbox.close()
            active_session.prime_toolbox = None
        gen = self._deps.get_gen_info(state)
        tools = DeepyZeroTools(
            gen,
            self._deps.get_processed_queue,
            send_cmd,
            session=active_session,
            get_output_filepath=self._deps.get_output_filepath,
            record_file_metadata=self._deps.record_file_metadata,
            get_server_config=self._server_config,
        )
        tools.zero_catalog = True
        return tools

    def _build_session_system_prompt(self, session, tools) -> tuple[str, str]:
        host = getattr(self, 'workspace_host', None)
        server_config = host.workspace_output_config(host.deepy_workspace_id()) if host is not None else self._server_config()
        if self.get_deepy_type() == DEEPY_TYPE_PRIME:
            session.file_access_policy = tools.file_access_policy
            scope = "Reading may extend outside these roots; writing does not." if tools.file_access_policy.read_everywhere else "Use authorized @alias paths from IO discovery."
            access = f"Filesystem access outside the session workspace is configured as {tools.file_access_policy.mode}. {scope}"
            job = f"\n\n{session.plugin_instructions}" if session.plugin_instructions else ""  # a plugin job's own instructions, for its session only
            return f"{PRIME_SYSTEM_PROMPT}\n\n{tools.get_system_instructions()}\n\n{access}{job}", DEEPY_PRIME_CUSTOM_SYSTEM_PROMPT_KEY
        from shared.deepy.filesystem import build_file_access_policy
        from shared.deepy.long_text import add_session_workspace, long_text_system_instructions, long_text_tools_active

        workspace = session_store.session_workspace(session) if self._multi_session_enabled or session.storage_session_dir else session_store.ensure_mono_session_workspace(session.chat_session_id)
        file_access_policy = add_session_workspace(build_file_access_policy(server_config), session.chat_session_id, workspace)
        session.file_access_policy = file_access_policy
        system_prompt = ZERO_SYSTEM_PROMPT
        custom_system_prompt_key = DEEPY_ZERO_CUSTOM_SYSTEM_PROMPT_KEY
        if file_access_policy.read_enabled:
            access = "read/write" if file_access_policy.write_enabled else "read-only"
            scope = "Reading is allowed everywhere; writing remains limited to output and selected folders." if file_access_policy.read_everywhere else "Use @alias/path from list_files roots; plain paths use @outputs."
            file_access_instructions = f"Filesystem {access} access is enabled. {scope} Files are toolbox actions: list_files without a path lists the roots, with a path one folder (pages: repeat with cursor=next_cursor while has_more); rg searches recursively; query_file reads one file's metadata or text."
        else:
            file_access_instructions = "Filesystem access is disabled. Use Gallery/media ids rather than direct paths."
        system_prompt = f"{system_prompt}\n\n{file_access_instructions}"
        experimental_instructions = long_text_system_instructions(file_access_policy)
        if experimental_instructions:
            system_prompt = f"{system_prompt}\n\n{experimental_instructions}"
        return system_prompt, custom_system_prompt_key

    def _build_local_engine(self, state, session, tools, send_cmd, system_prompt: str, custom_system_prompt_key: str, *, thinking_enabled: bool, override_profile=-1) -> AssistantEngine:
        from shared.deepy.loading_progress import DeepyLoadingProgress

        loading = DeepyLoadingProgress(
            lambda text: send_cmd("chat_output", assistant_chat.build_status_event(text, kind="loading", session=session)),
            lambda: session.interrupt_requested,
        )
        return AssistantEngine(
            session,
            AssistantRuntimeHooks(
                acquire_gpu=lambda: self._deps.acquire_gpu(state),
                release_gpu=lambda keep_resident = False, release_vram_callback = None, force_release_on_acquire = True: self._deps.release_gpu(state, keep_resident=keep_resident, release_vram_callback=release_vram_callback, force_release_on_acquire=force_release_on_acquire),
                register_gpu_resident=lambda release_vram_callback = None, force_release_on_acquire = True: self._deps.register_gpu_resident(state, release_vram_callback=release_vram_callback, force_release_on_acquire=force_release_on_acquire),
                clear_gpu_resident=lambda: self._deps.clear_gpu_resident(state),
                ensure_loaded=lambda: self._deps.ensure_prompt_enhancer_loaded(override_profile=override_profile, progress=loading, gen=loading.gen),
                unload_runtime=self._deps.unload_prompt_enhancer_runtime,
                unload_weights=self._unload_weights,
                ensure_vision_loaded=lambda: self._ensure_vision_loaded(override_profile=override_profile, progress=loading, gen=loading.gen),
                get_offload_manager=self._deps.get_enhancer_offloadobj,
            ),
            tools,
            send_cmd,
            debug_enabled=self._sync_debug_enabled(),
            thinking_enabled=thinking_enabled,
            vram_mode=self.get_vram_mode(),
            system_prompt=system_prompt,
            custom_system_prompt_key=custom_system_prompt_key,
        )

    def run_assistant_prompt_turn(self, state, model_def, prompt_enhancer_modes, original_prompts, seed, override_profile=-1, send_cmd=None, tools=None, replay_action: dict[str, Any] | None = None) -> None:
        debug_enabled = self._sync_debug_enabled()
        server_config = self._server_config()
        if not normalize_deepy_enabled(server_config.get(DEEPY_ENABLED_KEY, 0)):
            raise gr.Error(_DEEPY_DISABLED_TEXT)
        if not deepy_requirement_met(server_config):
            raise gr.Error(deepy_requirement_error(server_config))
        if send_cmd is None or tools is None:
            raise gr.Error("Assistant mode requires a command stream and a tool registry.")
        enhancer_temperature = server_config.get("prompt_enhancer_temperature", 0.6)
        enhancer_top_p = server_config.get("prompt_enhancer_top_p", 0.9)
        randomize_seed = server_config.get("prompt_enhancer_randomize_seed", True)
        assistant_seed = replay_action["seed"] if replay_action is not None else (secrets.randbits(32) if randomize_seed else (seed if seed is not None and seed >= 0 else 0))
        session = get_or_create_assistant_session(state)
        assistant_model_def = model_def
        session.session_environment = self._session_environment(session)
        system_prompt, custom_system_prompt_key = self._build_session_system_prompt(session, tools)
        remote_engine = resolve_role_engine(server_config, "deepy")
        if is_remote_engine(remote_engine):
            if replay_action is not None:
                raise RuntimeError("Persistent action replay is only available for the local Deepy runtime.")
            from shared.deepy.config import normalize_deepy_custom_system_prompt
            from shared.remote_llm.deepy_runner import run_remote_deepy_turn

            custom_prompt = normalize_deepy_custom_system_prompt(server_config.get(custom_system_prompt_key, ""))
            system_context_getter = getattr(tools, "get_system_context", None)
            system_context = str(system_context_getter() or "").strip() if callable(system_context_getter) else ""
            remote_system_prompt = "\n\n".join(part for part in (system_prompt, custom_prompt, system_context) if part).strip()
            return run_remote_deepy_turn(server_config, session, original_prompts[0] if original_prompts else "", remote_system_prompt, tools, send_cmd)
        _assistant_instructions, assistant_max_new_tokens = self._deps.resolve_prompt_enhancer_settings("", assistant_model_def, prompt_enhancer_modes, is_image=False, text_encoder_max_tokens=1024)
        assistant_max_new_tokens = int(replay_action["max_new_tokens"]) if replay_action is not None else max(1024, int(assistant_max_new_tokens))
        thinking_enabled = bool(replay_action["thinking_enabled"]) if replay_action is not None else "K" in prompt_enhancer_modes
        assistant = self._build_local_engine(state, session, tools, send_cmd, system_prompt, custom_system_prompt_key, thinking_enabled=thinking_enabled, override_profile=override_profile)
        with deepy_log_scope(start_if_needed=debug_enabled):
            assistant.run_turn(
                original_prompts[0] if len(original_prompts) > 0 else "",
                max_new_tokens=assistant_max_new_tokens,
                seed=assistant_seed,
                do_sample=True,
                temperature=enhancer_temperature,
                top_p=enhancer_top_p,
                replay_action=replay_action,
            )

    def ask_ai(self, state, ask_request, client_submission_id: str = "", steering: bool = False):
        from shared.deepy.drivers import gradio_chat_updates
        yield from gradio_chat_updates(self.iter_commands(state, ask_request, client_submission_id, steering), lambda: self._deps.get_new_refresh_id())

    def iter_commands(self, state, ask_request, client_submission_id: str = "", steering: bool = False, *, media: list[dict[str, str]] | None = None, author: str = "You"):
        debug_enabled = self._sync_debug_enabled()
        submission_id = str(client_submission_id or "").strip()[:128]
        acknowledged_submission_ids = [submission_id] if submission_id else []

        session = get_or_create_assistant_session(state)
        foreign_session_reset = self._reset_foreign_active_session(session)
        # A request carrying its own media (plugins) stays one message.
        request_blocks = self._expand_assistant_requests(session, ask_request) if media is None else [str(ask_request or "").strip()]
        if len(request_blocks) == 0:
            if debug_enabled:
                self._debug_log("Request ignored because it was empty after normalization.")
            yield "chat_output", assistant_chat.build_sync_event(session, acknowledged_submission_ids=acknowledged_submission_ids)
            return
        if debug_enabled:
            self._debug_log(f"Request received blocks={len(request_blocks)} worker_active={bool(session.worker_active)} queued_jobs={int(session.queued_job_count or 0)} foreign_session_reset={bool(foreign_session_reset)}")
        if session.drop_state_requested:
            if debug_enabled:
                self._debug_log("Request held because a Deepy reset is pending.")
            status = {"visible": True, "kind": "queued", "text": "Resetting after the current work stops..."}
            yield "chat_output", assistant_chat.build_sync_event(session, status=status, acknowledged_submission_ids=acknowledged_submission_ids)
            return
        if not self.is_available():
            if debug_enabled:
                self._debug_log(f"Request rejected: {self.requirement_error_text()}")
            error_turn_id = assistant_chat.create_assistant_turn(session)
            assistant_chat.set_assistant_content(session, error_turn_id, self.requirement_error_text())
            yield "chat_output", assistant_chat.build_sync_event(session, acknowledged_submission_ids=acknowledged_submission_ids)
            return
        if self.workspace_host is not None:
            with self._queue_state_lock:
                self._prepare_session_request_locked(session, request_blocks[0])
            # The host lock must be acquired outside all queue-lock scopes.
            self.workspace_host.ensure_session_workspace()
        if steering:
            with self._queue_state_lock:
                steering_active = bool(session.worker_active and session.control_queue is not None)
                if steering_active:
                    output_queue = session.control_queue
                    queued_epoch = session.chat_epoch
                    with session.turn_lock:
                        checkpoint = session.current_turn
                        steered_current_turn = isinstance(checkpoint, dict)
                    steered_message_ids = []
                    steered_tasks = []
                    for index, request_block in enumerate(request_blocks):
                        message_id, task = self._queue_assistant_request(state, session, output_queue, request_block, queued_epoch, queued=True, client_submission_id=submission_id, assistant_badge="Steered" if steered_current_turn and index == 0 else "")
                        if steered_current_turn:
                            steered_message_ids.append(message_id)
                            steered_tasks.append(task)
                    if steered_current_turn:
                        if not promote_async_tasks("assistant", steered_tasks):
                            raise RuntimeError("New steering request batch was not present in the assistant task queue.")
                        if not request_assistant_steering(session):
                            raise RuntimeError("Active assistant turn disappeared while applying steering.")
                        if session.assistant_compaction_active:
                            steering_text = "Steering accepted. Waiting for context compaction to finish..."
                        elif session.assistant_action_active:
                            steering_text = "Steering accepted. Steering will apply once the current tool action is done."
                        elif session.assistant_thought_active:
                            steering_text = "Steering accepted. Waiting for the current thought to finish..."
                        else:
                            steering_text = "Steering accepted. Applying the new instructions at the current boundary..."
                        steering_sync = assistant_chat.steer_queued_messages(session, steered_message_ids, status_text=steering_text, acknowledged_submission_ids=acknowledged_submission_ids)
                        if steering_sync is None:
                            raise RuntimeError("New steering request batch could not be promoted in the assistant transcript.")
                    else:
                        steering_text = "Queued behind the current assistant task."
                        steering_sync = assistant_chat.build_sync_event(session, status={"visible": True, "kind": "queued", "text": steering_text}, acknowledged_submission_ids=acknowledged_submission_ids)
            if steering_active:
                self._debug_log(f"Steering requested worker_active=True active_turn={steered_current_turn} thought_active={bool(session.assistant_thought_active)} action_active={bool(session.assistant_action_active)} queued_jobs={int(session.queued_job_count or 0)}")
                output_queue.push("chat_output", steering_sync)
                yield "request_accepted", None
                return
        with self._queue_state_lock:
            existing_output_queue = session.control_queue
            enqueue_active = existing_output_queue is not None and (session.worker_active or session.queued_job_count > 0)
            if enqueue_active:
                queued_epoch = session.chat_epoch
                for request_block in request_blocks:
                    self._queue_assistant_request(state, session, existing_output_queue, request_block, queued_epoch, queued=True, client_submission_id=submission_id, media=media, author=author)
                queued_sync = assistant_chat.build_sync_event(session, status={"visible": True, "kind": "queued", "text": "Queued behind the current assistant task."}, acknowledged_submission_ids=acknowledged_submission_ids)
        if enqueue_active:
            existing_output_queue.push("chat_output", queued_sync)
            yield "request_accepted", None
            return
        com_stream = AsyncStream()
        com_stream.output_queue = DeepyPublicationQueue(lambda: assistant_chat.build_sync_event(session))
        output_queue = com_stream.output_queue
        with self._queue_state_lock:
            queued = foreign_session_reset or session.worker_active or session.queued_job_count > 0
            queued_epoch = session.chat_epoch
            session.control_queue = output_queue
            for index, request_block in enumerate(request_blocks):
                self._queue_assistant_request(state, session, output_queue, request_block, queued_epoch, queued=queued or index > 0, client_submission_id=submission_id, media=media, author=author)
            accepted_status = None if queued else {"visible": True, "kind": "loading", "text": "Starting Deepy..."}
            accepted_sync = assistant_chat.build_sync_event(session, status=accepted_status, acknowledged_submission_ids=acknowledged_submission_ids)
            output_queue.push("chat_output", accepted_sync)
            if queued or len(request_blocks) > 1:
                output_queue.push("chat_output", assistant_chat.build_status_event("Queued behind the current assistant task.", kind="queued", session=session))
        while True:
            cmd, data = com_stream.output_queue.next()
            if cmd == "console_output":
                print(data)
            elif cmd == "chat_output":
                payload = _drain_chat_output_batch(com_stream.output_queue, data)
                if debug_enabled and _DEEPY_STREAM_TRACE_ENABLED:
                    envelope = json.loads(payload)
                    published = envelope.get("batch", [envelope])
                    descriptors = [f"{item.get('event', {}).get('type', '?')}:{item.get('event', {}).get('sequence', '-')}" for item in published]
                    self._debug_log(f"Publishing chat batch: {descriptors}")
                try:
                    yield "chat_output", payload
                finally:
                    com_stream.output_queue.complete_publication()
            elif cmd == "load_queue_trigger":
                yield cmd, data
            elif cmd == "abort_client_id":
                yield cmd, data
            elif cmd == "refresh_gallery":
                yield cmd, data
            elif cmd == "error":
                error_turn_id = assistant_chat.create_assistant_turn(session)
                error_event = assistant_chat.set_assistant_content(session, error_turn_id, str(data or "Assistant error."))
                yield "chat_output", error_event
            elif cmd == "exit":
                self._debug_log(f"Publication metrics: {com_stream.output_queue.metrics()}")
                break

    def enqueue_ai_while_busy(self, state, ask_request, client_submission_id: str = ""):
        self._sync_debug_enabled()
        session = get_or_create_assistant_session(state)
        submission_id = str(client_submission_id or "").strip()[:128]
        acknowledged_submission_ids = [submission_id] if submission_id else []
        request_blocks = self._expand_assistant_requests(session, ask_request)
        if len(request_blocks) == 0:
            yield assistant_chat.build_sync_event(session, acknowledged_submission_ids=acknowledged_submission_ids), gr.update(), gr.update(value=""), gr.update(), gr.update()
            return
        if session.drop_state_requested:
            status = {"visible": True, "kind": "queued", "text": "Resetting after the current work stops..."}
            yield assistant_chat.build_sync_event(session, status=status, acknowledged_submission_ids=acknowledged_submission_ids), gr.update(), gr.update(value=""), gr.update(), gr.update()
            return
        if not self.is_available():
            error_turn_id = assistant_chat.create_assistant_turn(session)
            assistant_chat.set_assistant_content(session, error_turn_id, self.requirement_error_text())
            yield assistant_chat.build_sync_event(session, acknowledged_submission_ids=acknowledged_submission_ids), gr.update(), gr.update(value=""), gr.update(), gr.update()
            return
        with self._queue_state_lock:
            output_queue = session.control_queue
            enqueue_active = output_queue is not None and (session.worker_active or session.queued_job_count > 0)
            if enqueue_active:
                queued_epoch = session.chat_epoch
                for request_block in request_blocks:
                    self._queue_assistant_request(state, session, output_queue, request_block, queued_epoch, queued=True, client_submission_id=submission_id)
                status = {"visible": True, "kind": "queued", "text": "Queued behind the current assistant task."}
                queued_sync = assistant_chat.build_sync_event(session, status=status, acknowledged_submission_ids=acknowledged_submission_ids)
        if enqueue_active:
            output_queue.push("chat_output", queued_sync)
            yield gr.update(), gr.update(), gr.update(value=""), gr.update(), gr.update()
            return
        yield from self.ask_ai(state, ask_request, client_submission_id=submission_id)

    def resume_restored_action(self, state, command_callback=None):
        session = get_or_create_assistant_session(state)
        replay = session.pending_action_replay
        if replay is None:
            return
        com_stream = AsyncStream()
        com_stream.output_queue = DeepyPublicationQueue(lambda: assistant_chat.build_sync_event(session))
        output_queue = com_stream.output_queue
        with self._queue_state_lock:
            if session.worker_active or session.queued_job_count > 0:
                raise gr.Error("Deepy is already active.")
            queued_epoch = session.chat_epoch
            session.control_queue = output_queue
            self._queue_assistant_request(state, session, output_queue, str(replay["user_text"]), queued_epoch, queued=False, replay_action=replay)
        if callable(command_callback):
            while True:
                cmd, data = output_queue.next()
                if cmd == "chat_output":
                    payload = _drain_chat_output_batch(output_queue, data)
                    try:
                        command_callback(cmd, payload)
                    finally:
                        output_queue.complete_publication()
                else:
                    command_callback(cmd, data)
                if cmd == "exit":
                    break
            return
        while True:
            cmd, data = output_queue.next()
            if cmd == "console_output":
                print(data)
            elif cmd == "chat_output":
                payload = _drain_chat_output_batch(output_queue, data)
                try:
                    yield payload, gr.update(), gr.update(), gr.update(), gr.update()
                finally:
                    output_queue.complete_publication()
            elif cmd == "load_queue_trigger":
                yield gr.update(), str(time.time()) + "_" + str(self._deps.get_new_refresh_id()), gr.update(), gr.update(), gr.update()
            elif cmd == "abort_client_id":
                yield gr.update(), gr.update(), gr.update(), gr.update(), str(data or "")
            elif cmd == "refresh_gallery":
                yield gr.update(), gr.update(), gr.update(), str(time.time()) + "_" + str(self._deps.get_new_refresh_id()), gr.update()
            elif cmd == "error":
                error_turn_id = assistant_chat.create_assistant_turn(session)
                error_event = assistant_chat.set_assistant_content(session, error_turn_id, str(data or "Assistant error."))
                yield error_event if error_event is not None else gr.update(), gr.update(), gr.update(), gr.update(), gr.update()
            elif cmd == "exit":
                break

    def list_saved_sessions(self) -> list[dict[str, Any]]:
        return session_store.list_sessions(self.get_deepy_type())

    def _wait_for_turn_boundary(self, session) -> None:
        if not session.worker_active:
            return
        with self._queue_state_lock:
            request_assistant_interrupt(session, "session_switch")
        session.worker_idle_event.wait()

    def start_new_session(self, state) -> dict[str, Any]:
        if not self._multi_session_enabled:
            raise gr.Error("Enable multi-session mode before starting a saved session.")
        session = get_or_create_assistant_session(state)
        if session.worker_active:
            session.pending_reset_mode = session_store.RESET_MODE_NEW
            request_assistant_reset(session)
            return {"pending": True, "event": assistant_chat.build_status_event("Starting a new session after the current tool finishes...", kind="queued", session=session)}
        with session.turn_lock:
            if session.storage_session_id:
                session_store.flush_session(session)
            session_store.start_new_session(session, save_current=False)
            self.release_vram(state, True, discard_runtime_snapshot=True, preserve_reset_base=True)
        return {"pending": False, "event": assistant_chat.build_reset_event(session)}

    def park_conversation(self, state, parking_dir) -> dict[str, Any]:
        """Set the idle conversation aside for a plugin job; unpark_conversation brings it back."""
        session = get_or_create_assistant_session(state)
        with session.turn_lock:
            parked = {"chat_session_id": session.chat_session_id, "gallery_workspace_id": session.gallery_workspace_id}
            if session.storage_session_id:
                session_store.flush_session(session)
                parked["storage_id"] = session.storage_session_id
            elif session.chat_transcript:
                # Without saved sessions the conversation exists only in memory: keep it in a private folder meanwhile.
                session.storage_title = "Parked Deepy conversation"
                session_store.ensure_session(session, "", self.get_deepy_type(), session_store.GALLERY_MEDIA_LINK, self._session_environment(session), media_import=True, directory=parking_dir)
                session_store.flush_session(session)
                parked["parked_id"], parked["parked_dir"] = session.storage_session_id, str(parking_dir)
            session_store.start_new_session(session, save_current=False)
            session.gallery_workspace_id = ""
            self.release_vram(state, True, discard_runtime_snapshot=True, preserve_reset_base=True)
        return parked

    def open_external_session(self, state, directory, title: str | None, new_title: str) -> dict[str, Any]:
        """Make a session folder outside the catalog the conversation; an empty folder starts a new session."""
        session = get_or_create_assistant_session(state)
        storage_id = session_store.open_external_session(directory)
        with session.turn_lock:
            if storage_id is None:
                session.storage_title = title or new_title
                session_store.ensure_session(session, "", self.get_deepy_type(), session_store.GALLERY_MEDIA_LINK, self._session_environment(session), media_import=True, directory=directory)
                return {"event": assistant_chat.build_reset_event(session), "resumed": False, "missing_media": []}
            try:
                session_store.validate_session(storage_id, self.get_deepy_type())
                loaded = session_store.load_session(session, storage_id, self.get_deepy_type())
            except Exception:
                session_store.close_external_session(storage_id)
                raise
            if title:
                session.storage_title = title
            return {"event": assistant_chat.build_replay_batch(session, loaded["replay_commands"]), "resumed": True, "missing_media": loaded["missing_media"]}

    def unpark_conversation(self, state, parked: dict[str, Any]) -> dict[str, Any]:
        """Save and close the plugin session, then reload the conversation set aside by park_conversation."""
        session = get_or_create_assistant_session(state)
        with session.turn_lock:
            plugin_id = session.storage_session_id
            session_store.start_new_session(session)
            session_store.close_external_session(plugin_id)
            self.release_vram(state, True, discard_runtime_snapshot=True, preserve_reset_base=True)
            storage_id = parked.get("storage_id") or parked.get("parked_id")
            if storage_id is None:
                session.chat_session_id, session.gallery_workspace_id = parked["chat_session_id"], parked["gallery_workspace_id"]
                return {"event": assistant_chat.build_reset_event(session), "prefill": False}
            loaded = session_store.load_session(session, storage_id, self.get_deepy_type())
            if "parked_id" in parked:
                # Back to an unsaved conversation: drop the private folder, keep the content.
                session_store.start_new_session(session, save_current=False)
                session_store.close_external_session(storage_id)
                shutil.rmtree(parked["parked_dir"])
            session.chat_session_id, session.gallery_workspace_id = parked["chat_session_id"], parked["gallery_workspace_id"]
            return {"event": assistant_chat.build_replay_batch(session, loaded["replay_commands"]), "prefill": True}

    def prefill_restored_session_context(self, state, *, replay_pending: bool = False):
        session = get_or_create_assistant_session(state)
        with session.turn_lock:
            prefill_tokens = 0
            if session.pending_replay_reason and session.messages and not is_remote_engine(resolve_role_engine(self._server_config(), "deepy")):
                send_cmd = lambda _cmd, _data=None: None
                tools = self.create_tools(state, send_cmd, session=session)
                system_prompt, custom_system_prompt_key = self._build_session_system_prompt(session, tools)
                thinking_enabled = True if session.pending_action_replay is None else bool(session.pending_action_replay["thinking_enabled"])
                assistant = self._build_local_engine(state, session, tools, send_cmd, system_prompt, custom_system_prompt_key, thinking_enabled=thinking_enabled)
                with deepy_log_scope(start_if_needed=self._sync_debug_enabled()):
                    prefill_tokens = assistant.prefill_restored_context()
            else:
                session.pending_replay_reason = ""
            session_store.schedule_autosave(session)
            replay_stream = self.resume_restored_action(state) if replay_pending and session.pending_action_replay is not None else None
            return (prefill_tokens, replay_stream) if replay_pending else prefill_tokens

    def resume_saved_session(self, state, storage_id: str, *, defer_context_prefill: bool = False, workspace_host=None) -> dict[str, Any]:
        if not self._multi_session_enabled:
            raise gr.Error("Enable multi-session mode and restart WanGP before resuming a saved session.")
        session = get_or_create_assistant_session(state)
        storage_id = str(storage_id or "").strip()
        if not storage_id:
            raise gr.Error("Select a saved Deepy session.")
        if session.storage_session_id == storage_id and session.pending_reset_mode == session_store.RESET_MODE_NEW:
            session.worker_idle_event.wait()
        if session.storage_session_id == storage_id:
            if workspace_host is not None:
                workspace_host.ensure_session_workspace()
            with session.turn_lock:
                context_prefill_pending = bool(session.pending_replay_reason)
                action_replay_pending = session.pending_action_replay is not None
                event = assistant_chat.build_replay_batch(session, session.ui_replay_commands) if session.ui_replay_commands else assistant_chat.build_sync_event(session)
                prefill_tokens = 0 if defer_context_prefill or not context_prefill_pending else self.prefill_restored_session_context(state)
                return {"event": event, "active_id": storage_id, "injected": 0, "prefill_tokens": prefill_tokens, "warnings": [], "context_prefill_pending": context_prefill_pending and defer_context_prefill, "action_replay_pending": action_replay_pending}
        manifest = session_store.validate_session(storage_id, self.get_deepy_type())
        if workspace_host is not None:
            workspace_host.validate_session_workspace(manifest.get('gallery_workspace_id'))
        self._wait_for_turn_boundary(session)
        with session.turn_lock:
            if session.storage_session_id:
                session_store.flush_session(session)
            session_store.start_new_session(session, save_current=False)
            self.release_vram(state, True, discard_runtime_snapshot=True, preserve_reset_base=True)
            try:
                loaded = session_store.load_session(session, storage_id, self.get_deepy_type())
            except Exception:
                session_store.start_new_session(session, save_current=False)
                raise
            from shared.deepy.filesystem import build_file_access_policy

            if workspace_host is not None:
                workspace_host.restore_session_workspace(session.gallery_workspace_id)
            config = workspace_host.workspace_output_config(workspace_host.deepy_workspace_id()) if workspace_host is not None else self._server_config()
            media_registry.sync_context_media_paths(session, build_file_access_policy(config))
            gallery = session_store.inject_session_media(session, self._deps.get_gen_info(state))
            saved_mcp = {str(item.get("name", "")) for item in list(loaded["environment"].get("mcp_servers", []) or []) if isinstance(item, dict)}
            current_mcp = set(normalize_deepy_prime_mcp_servers(self._server_config().get(DEEPY_PRIME_MCP_SERVERS_KEY, {}))) if self.get_deepy_type() == DEEPY_TYPE_PRIME else set()
            warnings = [f"Missing configured MCP server: {name}" for name in sorted(saved_mcp - current_mcp)]
            warnings.extend(f"Missing media: {path}" for path in loaded["missing_media"])
            prefill_tokens = 0 if defer_context_prefill else self.prefill_restored_session_context(state)
            return {"event": assistant_chat.build_replay_batch(session, loaded["replay_commands"]), "active_id": storage_id, "injected": gallery["injected"], "prefill_tokens": prefill_tokens, "warnings": warnings, "context_prefill_pending": defer_context_prefill, "action_replay_pending": session.pending_action_replay is not None}

    def rename_saved_session(self, state, storage_id: str, title: str) -> dict[str, Any]:
        session = get_or_create_assistant_session(state)
        with self._queue_state_lock:
            result = session_store.rename_stored_session(storage_id, title, active_session=session)
        if self.workspace_host is not None:
            self.workspace_host.rename_session_workspace(storage_id, result['title'])
        return result

    def duplicate_saved_session(self, state, storage_id: str) -> dict[str, Any]:
        session = get_or_create_assistant_session(state)
        return session_store.duplicate_stored_session(storage_id, active_session=session)

    def export_saved_session(self, state, storage_id: str) -> str:
        session = get_or_create_assistant_session(state)
        return str(session_store.export_stored_session(storage_id, active_session=session))

    def import_saved_session(self, archive_path: str) -> dict[str, Any]:
        return session_store.import_session(archive_path)

    def delete_saved_session(self, state, storage_id: str) -> dict[str, Any]:
        session = get_or_create_assistant_session(state)
        storage_id = str(storage_id or "").strip()
        if session.worker_active and session.storage_session_id == storage_id:
            raise gr.Error("Stop Deepy before deleting its active session.")
        active = session.storage_session_id == storage_id
        if active:
            session_store.flush_session(session)
            self.release_vram(state, True, discard_runtime_snapshot=True)
        destination = session_store.delete_session(storage_id, active_session=session)
        if active:
            session_store.start_new_session(session, save_current=False)
        if self.workspace_host is not None:
            self.workspace_host.delete_session_workspace(storage_id)
        return {"event": assistant_chat.build_reset_event(session) if active else None, "trash_path": str(destination), "active_id": session.storage_session_id}

    def stop_ai(self, state, queued_action=""):
        normalized_action = str(queued_action or "").strip()
        if normalized_action == assistant_chat.PAUSE_TOGGLE_ACTION:
            return self._toggle_pause_ai(state)
        if normalized_action:
            return self._apply_queued_request_action(state, queued_action)
        session = get_or_create_assistant_session(state)
        with self._queue_state_lock:
            worker_active = session.worker_active
            interrupt_requested = session.interrupt_requested
            if worker_active and not interrupt_requested:
                with session.turn_lock:
                    request_assistant_interrupt(session)
        if worker_active:
            cancelled_user_text = self._cancel_next_queued_request(session) if interrupt_requested else ""
            if cancelled_user_text:
                return assistant_chat.build_sync_event(session), gr.update(), gr.update(), gr.update()
            cancelled_job_id = self._cancel_active_prime_job(session, "Stop")
            self._debug_log(f"Stop requested worker_active=True active_prime_mcp_job={cancelled_job_id or 'none'}")
            status_text = "Stopping generation..." if cancelled_job_id else "Interrupting the current assistant task..."
            return assistant_chat.build_status_event(status_text, kind="stop_pending", session=session), gr.update(), gr.update(), gr.update()
        cancelled_user_text = self._cancel_next_queued_request(session)
        if cancelled_user_text:
            chat_event = assistant_chat.build_sync_event(session)
            return chat_event, gr.update(), gr.update(), gr.update()
        return gr.update(), gr.update(), gr.update(), gr.update()

    def _toggle_pause_ai(self, state):
        session = get_or_create_assistant_session(state)
        with self._queue_state_lock:
            if session.paused:
                changed = resume_assistant(session)
                status = {"visible": True, "kind": "resuming", "text": "Resuming Deepy..."}
                action = "resume"
            elif session.pause_requested:
                changed = False
                status = {"visible": True, "kind": "pause_pending", "text": "Pausing Deepy..."}
                action = "pause-pending"
            else:
                changed = request_assistant_pause(session)
                status = {"visible": True, "kind": "pause_pending", "text": "Pausing Deepy..."}
                action = "pause"
        if not changed and action != "pause-pending":
            return gr.update(), gr.update(), gr.update(), gr.update()
        self._debug_log(f"Deepy {action} requested worker_active={bool(session.worker_active)} tool_active={bool(session.assistant_action_active)}")
        return assistant_chat.build_sync_event(session, status=status), gr.update(), gr.update(), gr.update()

    def reset_ai(self, state, reset_mode=None):
        session = get_or_create_assistant_session(state)
        reset_mode = session_store.RESET_MODE_NEW if self._multi_session_enabled else session_store.RESET_MODE_RESET
        if session.worker_active:
            session.pending_reset_mode = reset_mode
            action = "new session" if reset_mode == session_store.RESET_MODE_NEW else "reset"
            print(f"[Assistant] {action.title()} requested during an active turn; it will apply after the current work stops.")
            request_assistant_reset(session)
            session.chat_html = ""
            status_text = "Starting a new session after the current tool finishes..." if reset_mode == session_store.RESET_MODE_NEW else "Resetting after the current tool finishes..."
            return assistant_chat.build_status_event(status_text, kind="queued", session=session), gr.update(), gr.update(value=""), gr.update()
        self._complete_session_reset(state, session, reset_mode)
        session.chat_html = ""
        return assistant_chat.build_reset_event(session), gr.update(), gr.update(value=""), gr.update()

    def browser_session_started(self, state):
        session = get_or_create_assistant_session(state)
        if self._reset_foreign_active_session(session):
            return assistant_chat.build_reset_event(session), gr.update(), gr.update(value=""), gr.update()
        if not session.worker_active and session.queued_job_count <= 0:
            return gr.update(), gr.update(), gr.update(), gr.update()
        request_assistant_reset(session)
        session.chat_html = ""
        return assistant_chat.build_status_event("Resetting after the current work stops...", kind="queued", session=session), gr.update(), gr.update(value=""), gr.update()


def create_controller(**deps_kwargs) -> DeepyController:
    return DeepyController(DeepyDeps(**deps_kwargs))
