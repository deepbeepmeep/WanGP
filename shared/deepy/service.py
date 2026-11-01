# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Persistent Deepy execution. HTTP connections are subscribers, not workers."""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
import uuid
from collections import OrderedDict, deque

from shared.deepy import chat, media_registry, session_store, ui_settings
from shared.deepy.config import DEEPY_GALLERY_MEDIA_VIEWER_DEFAULT, DEEPY_GALLERY_MEDIA_VIEWER_KEY
from shared.deepy.drivers import AppGallery
from shared.deepy.errors import DeepyBusy, error_payload
from shared.deepy.engine import get_or_create_assistant_session
from shared.deepy.runtime import GenerationRuntime
from shared.utils.downloads import registered_download_path
from shared.utils.form_sync import FormRegistry, Saved
from shared.utils.gallery_sources import VIDEO
from shared.gradio.progress import WangpProgress
from functools import partial
from shared.deepy.workspaces import WorkspaceSupport
from shared.deepy.plugin_jobs import PluginJobSupport


class DeepyService(WorkspaceSupport, PluginJobSupport, GenerationRuntime):
    workspace_viewer = None

    def __init__(self, deps, *, state=None, gallery_factory=AppGallery):
        self._deps = deps
        self._state = self._build_state() if state is None else state
        self._session = get_or_create_assistant_session(self._state)
        self.gallery = self._gallery = gallery_factory(deps, self._state)
        self._active_generation_client_id = ""
        self._condition = threading.Condition(threading.RLock())
        self._mutation_lock = threading.RLock()
        self._events = deque(maxlen=256)
        self._revision = 0
        self._instance_id = uuid.uuid4().hex
        self._submissions = OrderedDict()
        self._restoration_submissions = deque()
        self._threads = set()
        self._generation_lock = threading.Lock()
        self._progress = None
        self._progress_bar = WangpProgress()
        self._state["gen"]["download_progress_callback"] = partial(self._generation_event, "download")
        self._generation_aborting = False
        self._closing = False
        self._restoration = None
        self._plugin_job = None
        self._plugin_waiters = deque()
        self._request_states = {}
        deps.controller.request_listener = self._plugin_request_event
        self.forms = FormRegistry(self.publish)
        self.forms.register('deepy-web', self._settings_form()['values'], read=lambda: self._settings_form()['values'], scope=lambda: self._session.chat_session_id, lock=self._mutation_lock)

    def _print(self, text=""):
        print(text)

    def publish(self, kind, data):
        if kind == 'gallery':
            self.save_workspace()
        with self._condition:
            self._revision += 1
            self._events.append({"id": self._revision, "type": kind, "data": data})
            self._condition.notify_all()

    def publish_error(self, error):
        self.publish('error', error_payload(error))

    def snapshot(self):
        # Rendering history or media must never hold the event-publication lock.
        with self._condition:
            cursor, progress, busy = self._revision, self._progress, bool(self._threads) or bool(self._restoration and self._restoration["pending"])
        # Do not let a snapshot of the previous workspace replace the file URL
        # index after another browser has already switched to the new workspace.
        with self._mutation_lock:
            gallery, workspaces = self.gallery_snapshot(), self.workspace_snapshot()
        return {"cursor": cursor, "instance": self._instance_id, "display_settings": self.display_settings(), "chat": chat.build_sync_event(self._session), "gallery": gallery, "workspaces": workspaces, "progress": progress, "busy": busy, "restoration": self._restoration, "sessions": self._deps.controller.list_saved_sessions(), "active_session_id": self._session.storage_session_id, "active_session_title": self._session.storage_title, "multi_session": self._deps.controller.multi_session_enabled(), "deepy_type": self._deps.controller.get_deepy_type()}

    def events_after(self, cursor, timeout=15, instance=None):
        # Counter values can overlap after a restart. A cursor only belongs to
        # the service instance that issued it, even if it is still in range.
        same_instance = instance is None or instance == self._instance_id
        with self._condition:
            if same_instance and cursor == self._revision and not self._closing:
                self._condition.wait(timeout=timeout)
            expired = not same_instance or cursor > self._revision or (self._events and cursor < self._events[0]["id"] - 1)
            if not expired:
                return [event for event in self._events if event["id"] > cursor]
        snapshot = self.snapshot()
        return [{"id": snapshot["cursor"], "type": "snapshot", "data": snapshot}]

    def publish_sessions(self):
        self.publish("sessions", {"sessions": self._deps.controller.list_saved_sessions(), "active_session_id": self._session.storage_session_id, "active_session_title": self._session.storage_title, "multi_session": self._deps.controller.multi_session_enabled()})

    def display_settings(self):
        config = self._deps.get_server_config()
        return {"compact_actions": bool(config.get("deepy_compact_actions", True)), "gallery_media_viewer": bool(config.get(DEEPY_GALLERY_MEDIA_VIEWER_KEY, DEEPY_GALLERY_MEDIA_VIEWER_DEFAULT))}

    def update_display_settings(self, values):
        if set(values) != {"compact_actions"} or type(values["compact_actions"]) is not bool:
            raise ValueError("compact_actions must be a boolean.")
        self._deps.controller.set_compact_actions(values["compact_actions"])
        result = self.display_settings()
        self.publish("display_settings", result)
        self.forms.refresh("deepy-gradio")
        return result

    def _settings_form(self):
        current = ui_settings.get_persisted_assistant_tool_ui_settings(self._deps.get_server_config())
        current.update(self._session.tool_ui_settings)
        return ui_settings.get_simplified_settings_form(current, prime=self._deps.controller.get_deepy_type() == "prime")

    def settings(self):
        with self._mutation_lock:
            result = self._settings_form()
            result['sync'] = self.forms.forms['deepy-web'].snapshot()
            return result

    def submit_settings(self, baseline, values):
        return self.forms.forms['deepy-web'].submit(baseline, values, lambda merged: Saved(self.update_settings(merged))).output

    def update_settings(self, values, *, simplified=True, persist=True):
        with self._mutation_lock:
            if simplified:
                ui_settings.validate_simplified_settings(values, self.settings())
            current = ui_settings.get_persisted_assistant_tool_ui_settings(self._deps.get_server_config())
            current.update(self._session.tool_ui_settings)
            current.update(values)
            self._deps.controller.update_tool_ui_settings(self._state, **current, persist=persist)
            result = self.settings()
        self.publish("settings", result)
        return result

    def submit(self, text, submission_id, *, steering=False, media=None, author="You"):
        with self._mutation_lock:
            if self._closing:
                raise ValueError("Deepy is shutting down.")
            if submission_id in self._submissions:
                if self._submissions[submission_id] != (text, steering):
                    raise ValueError("This submission id already belongs to another request.")
                return submission_id in self._restoration_submissions
            self._submissions[submission_id] = (text, steering)
            if len(self._submissions) > 4096:
                self._submissions.popitem(last=False)
            with self._condition:
                queued = bool(self._restoration_submissions or (self._restoration and self._restoration['pending']))
                if queued:
                    self._restoration_submissions.append(submission_id)
                self._start_worker(lambda: self._run_request(text, submission_id, steering, media, author), [submission_id])
            return queued

    def _start_worker(self, task, acknowledged_ids=()):
        worker = threading.Thread(target=self._run_worker, args=(task, acknowledged_ids), name="Deepy web request", daemon=True)
        self._threads.add(worker)
        worker.start()

    def _run_worker(self, task, acknowledged_ids):
        try:
            task()
        except Exception as exc:
            traceback.print_exc()
            self.publish_error(exc)
        finally:
            try:
                self.publish("chat", chat.build_sync_event(self._session, acknowledged_submission_ids=acknowledged_ids))
                self.publish("gallery", self.gallery_snapshot())
                self.save_session()
                self.publish_sessions()
            finally:
                with self._condition:
                    self._threads.discard(threading.current_thread())
                    if task == self._resume_session and self._restoration["pending"]:
                        self._finish_restoration()

    def _run_request(self, text, submission_id, steering, media=None, author="You"):
        with self._condition:
            self._condition.wait_for(lambda: self._closing or (not (self._restoration and self._restoration['pending']) and (not self._restoration_submissions or self._restoration_submissions[0] == submission_id)))
            queued = submission_id in self._restoration_submissions
        try:
            if self._closing:
                return
            session_id = self._session.storage_session_id
            for command, data in self._deps.controller.iter_commands(self._state, text, submission_id, steering, media=media, author=author):
                if queued:
                    # The controller has accepted this request; subsequent ones can
                    # enter its normal queue/steering path while this turn streams.
                    with self._condition:
                        self._restoration_submissions.remove(submission_id)
                        self._condition.notify_all()
                    queued = False
                self.command(command, data)
                if self._session.storage_session_id != session_id:
                    session_id = self._session.storage_session_id
                    self.publish_sessions()
                    self.publish_workspaces()
        finally:
            if queued:
                with self._condition:
                    self._restoration_submissions.remove(submission_id)
                    self._condition.notify_all()

    def _resume_session(self):
        try:
            self._deps.controller.prefill_restored_session_context(self._state)
        finally:
            self.publish("chat", chat.build_status_event(None, visible=False, session=self._session))
        def resumed_command(command, data=None):
            if self._restoration['pending']:
                # Release waiting submissions only after the replay owns its queue.
                self._finish_restoration()
            self.command(command, data)
        for _ in self._deps.controller.resume_restored_action(self._state, command_callback=resumed_command):
            pass

    def _finish_restoration(self):
        self._restoration = {**self._restoration, "pending": False}
        self.publish("restoration", self._restoration)

    def require_restored(self):
        if self._restoration is not None and self._restoration["pending"]:
            raise DeepyBusy(f"{self._restoration['text']}. Wait for restoration to finish.")

    def _run_generation_command(self, data):
        with self._generation_lock:
            self._process_inline_queue(data)

    def command(self, command, data=None):
        if command == "chat_output" and data is not None:
            self.publish("chat", data)
        elif command == "load_queue_trigger":
            # Keep draining Deepy's events while generation runs. A failed media
            # task must not close the iterator that also owns queued/steered turns.
            with self._mutation_lock:
                self._start_worker(lambda: self._run_generation_command(data))
        elif command == "refresh_gallery":
            self.gallery.sync_refresh_path(data)
            self.publish("gallery", self.gallery_snapshot())
        elif command == "abort_client_id":
            self._deps.callbacks.emit("abort_generation", self._state, str(data or ""))
        elif command == "error":
            self.publish_error(data)

    def _generation_event(self, command, data):
        with self._condition:
            if command in {"progress", "status", "download"}:
                if command == "progress":
                    self._progress_bar(*data)
                elif command == "status":
                    self._progress_bar.status(data)
                else:
                    self._progress_bar.set_download(data)
                self._progress = {"type": command, "data": data, "description": self._progress_bar.description, "aborting": self._generation_aborting, "html": self._progress_bar.render(compact=True, aborting=self._generation_aborting)}
                self.publish("progress", self._progress)
            elif command == "exit":
                self._generation_aborting = False
                self._progress_bar = WangpProgress()
                self._progress = None
                self.publish("progress", None)

    def control(self, action, payload):
        started = time.perf_counter()
        self.require_restored()
        if action in {"reset", "resume", "delete", "rename"} and self._plugin_job is not None:
            raise DeepyBusy(f"{self._plugin_job.owner} is using Deepy. Change conversation once it finishes.")
        controller = self._deps.controller
        if action == "abort":
            with self._mutation_lock:
                if self.generation_running and not self._generation_aborting:
                    self._generation_aborting = True
                    self._generation_event("status", "Aborting generation…")
                    self.command("abort_client_id", self._active_generation_client_id)
            return self.snapshot()
        elif action == "stop":
            result = controller.stop_ai(self._state)
            if self._active_generation_client_id.startswith('ai_'):
                self.command("abort_client_id", self._active_generation_client_id)
        elif action == "pause":
            result = controller._toggle_pause_ai(self._state)
        elif action == "queued":
            result = controller.stop_ai(self._state, json.dumps(payload))
        elif action == "rename":
            with self._mutation_lock:
                controller.rename_saved_session(self._state, payload['id'], payload['title'])
                self.publish_sessions()
            return self.snapshot()
        elif action == "delete":
            with self._mutation_lock:
                self.require_idle()
                if payload.get('confirmed') is not True or not payload.get('id'):
                    raise ValueError('Confirm deletion of a saved Deepy session.')
                deleted = controller.delete_saved_session(self._state, payload['id'])
                if deleted['event'] is not None:
                    self.publish('chat', deleted['event'])
                    self.publish('settings', self.settings())
                self.publish_sessions()
            return self.snapshot()
        elif action in {"reset", "resume"}:
            with self._mutation_lock:
                locked = time.perf_counter()
                self.require_idle()
                if action == "reset":
                    result = controller.reset_ai(self._state)
                    reset_finished = time.perf_counter()
                else:
                    title = next((item['title'] for item in controller.list_saved_sessions() if item['id'] == payload['id']), payload['id'])
                    self._restoration = {"id": self._revision + 1, "text": f"Loading Session {title}", "pending": True}
                    self.publish("restoration", self._restoration)
                    try:
                        resumed = controller.resume_saved_session(self._state, payload["id"], defer_context_prefill=True, **({'workspace_host': self} if self.workspaces is not None else {}))
                        self.publish("chat", resumed["event"])
                        self.publish("chat", chat.build_status_event(f"Loading Session {self._session.storage_title}", kind="session_loading", session=self._session))
                        self.publish("gallery", self.gallery_snapshot())
                        self.publish("settings", self.settings())
                        self.publish_sessions()
                        self.publish_workspaces()
                    except Exception:
                        self.publish("chat", chat.build_status_event(None, visible=False, session=self._session))
                        self._finish_restoration()
                        raise
                    # Publish restored views before prefill can finish and clear loading.
                    self._start_worker(self._resume_session)
                    return self.snapshot()
        else:
            raise ValueError("Unknown Deepy action.")
        for value in result:
            if isinstance(value, str):
                self.publish("chat", value)
        if action in {"reset", "resume"}:
            self.publish("gallery", self.gallery_snapshot())
            self.publish("settings", self.settings())
            self.publish_sessions()
            self.publish_workspaces()
        result = self.snapshot()
        finished = time.perf_counter()
        if action == "reset" and finished - started >= 1:
            print(f"[Deepy][New session] total={finished-started:.3f}s service lock={locked-started:.3f}s reset={reset_finished-locked:.3f}s publish/views={finished-reset_finished:.3f}s", flush=True)
        return result

    @property
    def generation_running(self):
        return bool(self._active_generation_client_id or self._progress is not None)

    def require_idle(self):
        if self.generation_running:
            raise DeepyBusy("A generation is in progress. Wait for it to finish before changing conversation.")
        if self._threads or self._session.worker_active or self._session.queued_job_count:
            raise DeepyBusy("Deepy is active in this conversation. Wait for it to finish before changing conversation.")

    def import_media(self, path, *, context='deepy', session_epoch=None, from_chat=False):
        with self._mutation_lock:
            if session_epoch is not None and session_epoch != self._session.chat_epoch:
                raise DeepyBusy('The Deepy session changed during upload. Please import the file again.')
            if context == 'deepy':
                self.prepare_media_workspace()
            if self.workspaces is not None:
                path = self.workspaces.adopt(self.workspace_id, [str(path)])[0]
            result = self.gallery.add_path(str(path))
            record = media_registry.mark_media_access(result["record"], "read")
            entry = {"media_id": record["settings"]["gallery_media_ids"][0], "title": record["label"], "media_type": record["media_type"], "path": record["path"]}
            with self._session.turn_lock:
                pending = self._session.pending_chat_media
                existing = next((item for item in pending if item['media_id'] == entry['media_id']), None)
                if existing is None:
                    pending.append(entry)
                    existing = entry
                if from_chat:
                    existing['chat_upload'] = True
            if from_chat:
                self.publish('chat', chat.build_pending_upload_event(self._session))
            self.publish("gallery", self.gallery_snapshot())
            self.save_session()
            return self.gallery.media_id(result["record"]["path"])

    def remove_chat_upload(self, chat_session_id, media_id=None):
        """Detach the given pending chat upload, or the last one."""
        with self._mutation_lock:
            self.require_restored()
            if chat_session_id != self._session.chat_session_id:
                raise DeepyBusy('The Deepy session changed. Please try again.')
            with self._session.turn_lock:
                pending = self._session.pending_chat_media
                for index in range(len(pending) - 1, -1, -1):
                    if pending[index].get('chat_upload') and media_id in (None, pending[index]['media_id']):
                        pending.pop(index)
                        break
            event = chat.build_pending_upload_event(self._session)
            self.publish('chat', event)
            self.save_session()
            return event

    def select_media(self, media_id):
        with self._mutation_lock:
            self.require_restored()
            if self.workspaces is not None and self._deepy_workspace_locked():
                owned = self.workspaces.for_session(self._session.storage_session_id)
                if owned is None:
                    raise ValueError('Media not found in this Deepy session.')
                if self.workspace_id != owned['id']:
                    self.require_idle()
                    self.ensure_session_workspace()
                    self.gallery_snapshot()
            if self.gallery.select(media_id) is None:
                raise ValueError("Media not found.")
            self.publish("gallery", self.gallery_snapshot())

    def locate_gallery_media(self, url):
        """Image/video gallery position of a chat media URL (None when its file is not in that gallery), for the gallery's full-screen viewer."""
        path = registered_download_path(url)
        if path is None:
            return None
        key = os.path.normcase(path)
        with self._mutation_lock:
            paths, _ = self.gallery._resolve_lists(VIDEO.id)
            index = next((index for index, item in enumerate(paths) if os.path.normcase(os.path.abspath(str(item))) == key), None)
            return None if index is None else {"workspace": self.workspace_id, "index": index, "path": paths[index]}

    def save_session(self):
        future = session_store.schedule_autosave(self._session)
        storage_id = self._session.storage_session_id
        if future is not None:
            def saved(result):
                if not self._closing and storage_id == self._session.storage_session_id and result.exception() is None:
                    self.publish_sessions()
            future.add_done_callback(saved)

    def close(self):
        self.save_workspace()
        with self._condition:
            self._closing = True
            self._condition.notify_all()
        session_store.flush_session(self._session)
