# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Deepy requests started by plugins.

A job waits until Deepy and the generation queue are idle, sets the user's conversation and workspace aside, runs the
plugin's request in its own session and workspace (shown in the UI meanwhile), then brings the user's back.
"""
from __future__ import annotations

import contextlib
import os
import queue
import re
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

from shared.api import ProgressUpdate, SessionEvent, WanGPSession, extract_status_phase_label
from shared.deepy import media_registry, session_store
from shared.deepy.filesystem import build_file_access_policy
from shared.utils.gallery_media import gallery_media_ids
from shared.utils.gallery_sources import GALLERY_SOURCES, MESH_IMPORT_EXTENSIONS, source_for_kind
from shared.utils.media_imports import persist_gallery_import
from shared.utils.media_settings import peek_settings
from shared.utils.mesh_outputs import import_mesh_file
from shared.utils.workspace_storage import STORAGE_KEY, background_gallery
from shared.utils.workspaces import load_gallery

WORKSPACE_MODES = ("isolated", "temporary", "current")
MEDIA_MODES = ("copy", "link")
TERMINAL_EVENTS = ("done", "cancelled", "error")
LIVE_INTERVAL = 1 / 3  # seconds between the updates of a thought being written or of the speed
_MEDIA_ID_RE = re.compile(r"^(?:" + "|".join(re.escape(source.media_ids) for source in GALLERY_SOURCES) + r"):[0-9a-f]{12}$")
_service = None
_open_scratch: set[Path] = set()


def attach_service(service) -> None:
    """The web UI's Deepy service, which plugin jobs drive."""
    global _service
    _service = service


def detach_service(service) -> None:
    global _service
    if _service is service:
        _service = None


class JobCancelled(Exception):
    pass


@dataclass(frozen=True)
class DeepyResult:
    status: str                                 # "done", "cancelled" or "error"
    text: str = ""                              # Deepy's reply to the request
    outputs: tuple[dict[str, Any], ...] = ()    # media Deepy wrote: id, path, kind, name
    session_dir: str | None = None
    error: str | None = None
    reply: str = ""                             # its last message alone: text holds all it said during the job

    @property
    def cancelled(self) -> bool:
        return self.status == "cancelled"


def _progress_update(event) -> ProgressUpdate:
    current_step = total_steps = unit = None
    status = event["description"]
    if event["type"] == "progress":
        head, data = event["data"][0], event["data"]
        if isinstance(head, tuple):
            current_step, total_steps = int(head[0]), int(head[1])
        status = str(data[1]) if len(data) > 1 else status
        unit = str(data[3]) if len(data) > 3 else None
    phase = WanGPSession._normalize_phase(status)
    return ProgressUpdate(phase=phase, status=status, progress=WanGPSession._estimate_progress(phase, current_step, total_steps), current_step=current_step, total_steps=total_steps, raw_phase=extract_status_phase_label(status) or None, unit=unit)


class DeepyJob:
    """A plugin's Deepy request. Iterate events(), read result(), then close() it (or use it as a context manager)."""

    def __init__(self, service, owner, prompt, paths, *, media_mode, session_dir, workspace, title, instructions=""):
        self.owner, self.prompt, self.media_mode, self.workspace, self.title = owner, prompt, media_mode, workspace, title
        self.instructions = instructions
        self.session_dir = session_dir
        self._service, self._paths = service, paths
        root = session_store.sessions_root() / ".plugin-jobs"
        _sweep_scratch(root)
        # Private folder for an ephemeral session (and its temporary workspace) and for a parked unsaved conversation.
        self._scratch = root / uuid.uuid4().hex
        _open_scratch.add(self._scratch)
        self._session_folder = session_dir or self._scratch / "session"
        self._submission_id = "plugin-" + uuid.uuid4().hex
        self._events = queue.Queue()
        self._cancel = threading.Event()
        self._done = threading.Event()
        self._result = None
        self._phase = "waiting"
        self._text, self._reply, self._tools, self._status, self._outputs = "", "", {}, None, {}
        self._thoughts, self._thought_due, self._stats, self._stats_due = {}, 0.0, None, 0.0
        self._known, self._steers = set(), 0
        self.chat_events = False  # a page showing Deepy's window sets it: "chat" events then bring what that window needs
        self._thread = threading.Thread(target=self._run, name=f"Deepy job ({owner})", daemon=True)
        self._thread.start()

    def events(self, timeout: float | None = None) -> Iterator[SessionEvent]:
        """Events until the job ends: queued, started, status, text, thought, stats, tool, progress, output, chat when `chat_events` is set, then done, cancelled or error."""
        while True:
            event = self._events.get(timeout=timeout)
            yield event
            if event.kind in TERMINAL_EVENTS:
                return

    def result(self, timeout: float | None = None) -> DeepyResult:
        if not self._done.wait(timeout):
            raise TimeoutError("The Deepy job is still running.")
        return self._result

    def done(self) -> bool:
        return self._done.is_set()

    def steer(self, text: str) -> None:
        """Send Deepy a message while it works on this job; it applies it at its next step, as a message typed in the Deepy chat does."""
        if self._phase != "started" or self._done.is_set():
            raise RuntimeError("This Deepy job is not running: send the message with a new run() on the same session_dir.")
        self._steers += 1
        self._service.submit(str(text), f"{self._submission_id}-steer-{self._steers}", steering=True, author=self.owner)

    def cancel(self) -> None:
        """Stop waiting for Deepy, or interrupt the request once it runs."""
        self._cancel.set()
        with self._service._condition:
            self._service._condition.notify_all()

    def close(self) -> None:
        """End the job (cancelling it if needed) and delete its private files, temporary workspace included."""
        if not self._done.is_set():
            self.cancel()
            self._done.wait()
        shutil.rmtree(self._scratch, ignore_errors=True)
        _open_scratch.discard(self._scratch)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _emit(self, kind, data=None):
        self._events.put(SessionEvent(kind, data))

    def _finish(self, status, error=None):
        self._result = DeepyResult(status, self._text, tuple(self._outputs.values()), None if self.session_dir is None else str(self.session_dir), error, self._reply)
        self._done.set()
        self._emit(status, self._result)

    def _run(self):
        try:
            interrupted = self._service._plugin_run(self)
        except JobCancelled:
            self._finish("cancelled")
        except Exception as error:
            traceback.print_exc()
            self._finish("error", str(error))
        else:
            self._finish("cancelled" if interrupted else "done")

    def _follow(self, media):
        """Submit the request, translate what Deepy does into events, and return whether it was interrupted."""
        service = self._service
        session = service._session
        with service._condition:
            cursor, instance = service._revision, service._instance_id
            service._request_states[self._submission_id] = {"phase": "submitted"}
        with session.turn_lock:
            known = self._known = {record["path_key"] for record in session.media_registry if record.get("path_key")}
        service.submit(self.prompt, self._submission_id, media=media, author=self.owner)
        stopping = False
        try:
            while True:
                if service._closing:
                    raise RuntimeError("WanGP is shutting down.")
                for event in service.events_after(cursor, timeout=1, instance=instance):
                    cursor = event["id"]
                    if event["type"] == "snapshot":
                        instance = event["data"]["instance"]
                    elif event["type"] == "progress" and event["data"] is not None and self._phase == "started":
                        self._emit("progress", _progress_update(event["data"]))
                    elif event["type"] == "chat" and self.chat_events and event["data"] is not None and self._phase == "started":
                        self._emit("chat", event["data"])
                with service._condition:
                    state = service._request_states[self._submission_id]
                    # Deepy answers some requests without running them (unavailable, resetting): nothing will start.
                    if state["phase"] == "submitted" and service._plugin_idle():
                        raise RuntimeError("Deepy did not run the request; its reply is in the Deepy chat.")
                if state["phase"] == "started":
                    self._phase = "started"
                self._scan(session, known)
                if state["phase"] == "cancelled":
                    return True
                if state["phase"] == "finished" and (not self._steers or service._plugin_idle()):
                    # Steering interrupts the turn to continue with the new message: that is not a stop, and the job follows that work until Deepy is done.
                    return self._cancel.is_set() if self._steers else bool(state["interrupted"])
                if self._cancel.is_set() and not stopping:
                    stopping = True
                    service.control("stop", {})
        finally:
            with service._condition:
                del service._request_states[self._submission_id]

    def _scan(self, session, known, final=False):
        with session.turn_lock:
            transcript = session.chat_transcript
            start = next((index for index in range(len(transcript) - 1, -1, -1) if transcript[index].get("client_submission_id") == self._submission_id), None)
            blocks = []
            if start is not None:
                # The session is the job's: messages steering it (from the plugin or the Deepy chat) join the job, and so does the work after them.
                for record in transcript[start + 1:]:
                    if record.get("role") != "user":
                        blocks.extend(dict(block) for block in record.get("blocks", []) if isinstance(block, dict))
            status, stats = session.chat_status, session.chat_stats or session.remote_usage_stats
            written = [(record["path_key"], record["path"], record["media_type"], record.get("label", "")) for record in session.media_registry if "write" in record.get("access", []) and record.get("path_key") not in known and record.get("path")]
        said = [block["text"] for block in blocks if block.get("type") == "markdown" and block.get("text")]
        text = "\n\n".join(said)
        if text != self._text:
            self._reply = said[-1]
            self._emit("text", {"text": text, "delta": text[len(self._text):] if text.startswith(self._text) else text, "last": self._reply})
            self._text = text
        now = float("inf") if final else time.monotonic()  # the last pass holds nothing back
        for block in blocks:
            if block.get("type") == "reasoning":
                thought, streaming, before = block["text"], bool(block.get("streaming")), self._thoughts.get(block["id"], ("", True))
                if (thought, streaming) != before and (not streaming or now >= self._thought_due):  # a finished thought is never held back
                    self._thoughts[block["id"]], self._thought_due = (thought, streaming), now + LIVE_INTERVAL
                    self._emit("thought", {"id": block["id"], "text": thought, "delta": thought[len(before[0]):] if thought.startswith(before[0]) else thought, "streaming": streaming})
            if block.get("type") == "tool" and self._tools.get(block["id"]) != (block["status"], block["status_text"]):
                self._tools[block["id"]] = (block["status"], block["status_text"])
                failure = str((block.get("result") or {}).get("error") or "")[:400] if block["status"] == "error" else ""  # what went wrong, for the plugin's page
                self._emit("tool", {**{key: block[key] for key in ("id", "name", "label", "status", "status_text")}, **({"error": failure} if failure else {})})
        if status is not None and status != self._status:
            self._emit("status", {"kind": status["kind"], "text": status["text"], **({"progress": status["progress"]} if "progress" in status else {})})
        self._status = status
        if stats and stats != self._stats and now >= self._stats_due:
            self._stats, self._stats_due = stats, now + LIVE_INTERVAL
            self._emit("stats", {key: value for key, value in stats.items() if key != "visible"})
        for path_key, path, kind, label in written:
            if path_key not in self._outputs:
                self._outputs[path_key] = {"id": gallery_media_ids(path, source_for_kind(kind).media_ids)[0], "path": path, "kind": kind, "name": Path(path).name, "label": label}
                self._emit("output", self._outputs[path_key])


def _sweep_scratch(root: Path) -> None:
    """Remove private job folders left by an earlier run that ended without closing its jobs."""
    if not root.is_dir():
        return
    for child in root.iterdir():
        if child not in _open_scratch and not any(session_store.session_folder_in_use(child / name) for name in ("session", "parked")):
            shutil.rmtree(child, ignore_errors=True)


class PluginJobSupport:
    """Service side of plugin jobs (mixed into DeepyService)."""

    def plugin_workspaces(self) -> list[dict[str, Any]]:
        with self._mutation_lock:
            return [{"storage_mode": "shared", **item, "shown": item["id"] == self.workspace_id} for item in self.workspaces.catalog()]

    def _plugin_media_item(self, path, source, workspace_id, selected=False):
        path = os.path.abspath(path)
        return {"id": gallery_media_ids(path, source.media_ids)[0], "path": path, "kind": self.gallery._detect_media_type(path), "name": Path(path).name, "selected": selected, "workspace": workspace_id}

    def _plugin_gallery_state(self, workspace_id):
        if workspace_id == self.workspace_id:
            return self._state["gen"]
        return background_gallery(self, workspace_id) or load_gallery(self.workspaces.get(workspace_id), self._workspace_settings)

    def plugin_gallery(self, workspace_id=None, kind=None) -> list[dict[str, Any]]:
        with self._mutation_lock:
            workspace_id = workspace_id or self.workspace_id
            gen = self._plugin_gallery_state(workspace_id)
            items = [self._plugin_media_item(path, source, workspace_id, index == gen[source.selected_key]) for source in GALLERY_SOURCES for index, path in enumerate(gen[source.files_key])]
        return [item for item in items if kind is None or item["kind"] == kind]

    def plugin_selected(self) -> dict[str, Any] | None:
        with self._mutation_lock:
            gen = self._state["gen"]
            source = next(source for source in GALLERY_SOURCES if source.id == gen["current_gallery_source"])
            index = gen[source.selected_key]
            return self._plugin_media_item(gen[source.files_key][index], source, self.workspace_id, True) if 0 <= index < len(gen[source.files_key]) else None

    def plugin_resolve_media(self, media) -> list[str]:
        """Gallery ids (shown workspace first, then the others), gallery items or file paths, as existing file paths."""
        paths = []
        for item in media:
            if isinstance(item, dict):
                item = item["path"]
            text = os.fspath(item)
            path = self._plugin_find_media_id(text) if _MEDIA_ID_RE.fullmatch(text) else os.path.abspath(text)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Media file not found: {path}")
            paths.append(path)
        return paths

    def _plugin_find_media_id(self, media_id):
        with self._mutation_lock:
            gen = self._state["gen"]
            for source in GALLERY_SOURCES:
                for index, path in enumerate(gen[source.files_key]):
                    if media_id in gallery_media_ids(path, source.media_ids, peek_settings(gen[source.settings_key], index)):
                        return os.path.abspath(path)
            for workspace_id in self.workspaces.records:
                gallery = self.workspaces.get(workspace_id)["gallery"]
                for source in GALLERY_SOURCES:
                    for path in gallery.get(source.files_key, []):
                        if media_id in gallery_media_ids(path, source.media_ids):
                            return os.path.abspath(path)
        raise ValueError(f"No gallery media has the id {media_id}.")

    def plugin_job_storage(self, workspace, session_dir) -> str:
        """Storage of the workspace a job would use: 'dedicated' (own folder) or 'shared' (files kept where they are)."""
        with self._mutation_lock:
            if workspace == "current":
                return self.workspaces.get(self.workspace_id).get("storage_mode", "shared")
            return "dedicated" if workspace == "temporary" or session_dir is not None else self._deps.get_server_config().get(STORAGE_KEY, "shared")

    def _plugin_request_event(self, submission_id, phase, info):
        with self._condition:
            if submission_id in self._request_states:
                self._request_states[submission_id] = {"phase": phase, **info}
                self._condition.notify_all()

    def _plugin_idle(self):
        session = self._session
        return not (self._threads or session.worker_active or session.queued_job_count or self.generation_running or (self._restoration and self._restoration["pending"]))

    def _plugin_run(self, job):
        parked = self._plugin_acquire(job)
        try:
            job._emit("started", {"session_dir": None if job.session_dir is None else str(job.session_dir), "workspace_id": self.workspace_id})
            if job._cancel.is_set():
                raise JobCancelled()
            with self._mutation_lock:
                media = self._plugin_attach(job)
            return job._follow(media)
        finally:
            with self._condition:
                while not self._plugin_idle():
                    self._condition.wait(timeout=1)
            if job._phase == "started":  # a late steering message can start one more turn: its reply and media belong to the job
                job._scan(self._session, job._known, final=True)
            self._plugin_restore(job, parked)

    def _plugin_acquire(self, job):
        with self._condition:
            self._plugin_waiters.append(job)
        try:
            queued = False
            while True:
                with self._condition:
                    while not (self._plugin_waiters[0] is job and self._plugin_job is None and self._plugin_idle()):
                        if job._cancel.is_set():
                            raise JobCancelled()
                        if not queued:
                            queued = True
                            job._emit("queued")
                        self._condition.wait(timeout=1)
                # Requests and generations start under the mutation lock: idle checked here stays idle.
                with self._mutation_lock:
                    if self._plugin_job is None and self._plugin_idle():
                        self._plugin_job = job
                        try:
                            return self._plugin_takeover(job)
                        except BaseException:
                            self._plugin_job = None
                            raise
        finally:
            with self._condition:
                self._plugin_waiters.remove(job)
                self._condition.notify_all()

    def _plugin_takeover(self, job):
        controller = self._deps.controller
        parked = controller.park_conversation(self._state, job._scratch / "parked")
        parked["workspace_id"] = self.workspace_id
        try:
            opened = controller.open_external_session(self._state, job._session_folder, job.title, job.title or session_store.automatic_title(job.prompt))
            session = self._session
            session.plugin_instructions = job.instructions  # Prime's system prompt carries them for this job only
            parked["lease"] = session.storage_session_id
            workspace_id, created = self._plugin_workspace(job, session, opened["resumed"])
            if created:
                parked["created_workspace"] = workspace_id
            self.workspaces.lease(workspace_id, session.storage_session_id)
            if workspace_id != self.workspace_id:
                self.save_workspace()
                self._apply_workspace(workspace_id)
            session.gallery_workspace_id = workspace_id
            if opened["resumed"]:
                media_registry.sync_context_media_paths(session, build_file_access_policy(self.workspace_output_config()))
                if created:
                    session_store.inject_session_media(session, self._state["gen"])
            session_store.schedule_autosave(session)
        except BaseException:
            self._plugin_restore(job, parked)
            raise
        self.publish("chat", opened["event"])
        self._plugin_publish_views()
        return parked

    def plugin_folder_workspace(self, name, folder):
        with self._mutation_lock:
            for workspace_id in self.workspaces.records:
                if self.workspaces.media_dir(workspace_id) == folder:
                    return workspace_id
            taken, unique, number = {record['name'].casefold() for record in self.workspaces.records.values()}, name, 1
            while unique.casefold() in taken:  # another folder of the same name, such as two productions with one title
                number += 1
                unique = f'{name[:112]} ({number})'
            workspace_id = self.workspaces.create(unique, folder=folder)
            self.workspaces.protect(workspace_id, True)  # a plugin's own folder is not archived for inactivity
        self.publish_workspaces()
        return workspace_id

    def _plugin_workspace(self, job, session, resumed):
        if job.workspace == "current":
            return self.workspace_id, False
        if job.workspace not in WORKSPACE_MODES:  # an existing workspace named by its id
            return job.workspace, False
        folder = job.session_dir / "gallery" if job.session_dir is not None else job._scratch / "session" / "gallery" if job.workspace == "temporary" else None
        # A resumed session finds its isolated workspace again, unless the user deleted it.
        if resumed and session.gallery_workspace_id in self.workspaces.records and self.workspaces.media_dir(session.gallery_workspace_id) == folder:
            return session.gallery_workspace_id, False
        self.workspaces.default_mode = self._deps.get_server_config().get(STORAGE_KEY, "shared")
        return self.workspaces.create(session.storage_title, folder=folder), True

    def _plugin_attach(self, job):
        """Bring the job's media into its gallery; they go with its message only."""
        dedicated = self.workspaces.media_dir(self.workspace_id) is not None
        if job.media_mode == "link" and dedicated:
            raise ValueError("Linked media need a workspace with shared storage: this workspace keeps its media in its own folder.")
        output = self.workspace_output_config()
        in_gallery = {os.path.normcase(os.path.abspath(path)) for source in GALLERY_SOURCES for path in self._state["gen"][source.files_key]}
        entries = []
        for path in job._paths:
            # Media already in the job's gallery are used as they are.
            if job.media_mode == "copy" and os.path.normcase(path) not in in_gallery:
                if Path(path).suffix.lower() in MESH_IMPORT_EXTENSIONS:
                    Path(output["mesh_save_path"]).mkdir(parents=True, exist_ok=True)
                    path, _reused = import_mesh_file(path, output["mesh_save_path"])
                elif dedicated:
                    path = self.workspaces.adopt(self.workspace_id, [path])[0]
                else:
                    Path(output["save_path"]).mkdir(parents=True, exist_ok=True)
                    path, _reused = persist_gallery_import(path, output["save_path"])
            record = media_registry.mark_media_access(self.gallery.add_path(str(path))["record"], "read")
            entries.append({"media_id": record["settings"]["gallery_media_ids"][0], "title": record["label"], "media_type": record["media_type"]})
        self.publish("gallery", self.gallery_snapshot())
        return entries

    def _plugin_unpark(self, parked):
        self._session.plugin_instructions = ""  # never into the user's conversation
        if "lease" in parked and parked["lease"] in self.workspaces.session_workspaces:
            self.workspaces.end_lease(parked["lease"])
        restored = self._deps.controller.unpark_conversation(self._state, parked)
        if self.workspace_id != parked["workspace_id"]:
            self._apply_workspace(parked["workspace_id"], associate=False)
        return restored

    def _plugin_restore(self, job, parked):
        with self._mutation_lock:
            self.save_workspace()
            try:
                restored = self._plugin_unpark(parked)
                if job.workspace == "temporary" and "created_workspace" in parked:
                    self.workspaces.delete(parked["created_workspace"])
            finally:
                self._plugin_job = None
            self.publish("chat", restored["event"])
            self._plugin_publish_views()
            if restored["prefill"]:
                self._restoration = {"id": self._revision + 1, "text": "Restoring Your Deepy Conversation", "pending": True}
                self.publish("restoration", self._restoration)
                self._start_worker(self._resume_session)

    def _plugin_publish_views(self):
        self.publish("gallery", self.gallery_snapshot())
        self.publish("settings", self.settings())
        self.publish_sessions()
        self.publish_workspaces()


class DeepyClient:
    """Deepy for one plugin: list gallery media and run requests (WAN2GPPlugin.deepy)."""

    def __init__(self, owner: str):
        if _service is None:
            raise RuntimeError("Deepy is available to plugins in the WanGP web UI, once it has started.")
        self._service, self.owner = _service, owner

    @property
    def unavailable_reason(self) -> str:
        """Why Deepy cannot run requests (disabled, or its model is not set up), or "" when it can."""
        controller = self._service._deps.controller
        return "" if controller.is_available() else controller.requirement_error_text()

    @property
    def deepy_type(self) -> str:
        """The Deepy version jobs run with: "zero" or "prime"."""
        return self._service._deps.controller.get_deepy_type()

    def workspaces(self) -> list[dict[str, Any]]:
        """Workspaces: id, name, storage_mode and whether the UI shows it."""
        return self._service.plugin_workspaces()

    def gallery(self, workspace: str | None = None, kind: str | None = None) -> list[dict[str, Any]]:
        """Media of a workspace (the shown one by default): id, path, kind (image, video, audio, 3d), name, selected, workspace."""
        return self._service.plugin_gallery(workspace, kind)

    def selected(self) -> dict[str, Any] | None:
        """The media selected in the shown gallery."""
        return self._service.plugin_selected()

    def run(self, prompt: str, media=(), *, media_mode: str = "copy", session_dir: str | os.PathLike | None = None, workspace: str | None = None, title: str | None = None,
            instructions: str | None = None) -> DeepyJob:
        """Send a request to Deepy in its own session and return the running job (see docs/PLUGINS.md). instructions: rules added to Deepy Prime's
        system prompt for this job's session only (they survive context compaction, unlike the request); Deepy Zero ignores them."""
        prompt = str(prompt or "").strip()
        if not prompt:
            raise ValueError("The Deepy request is empty.")
        if media_mode not in MEDIA_MODES:
            raise ValueError(f"media_mode must be one of {', '.join(MEDIA_MODES)}.")
        session_dir = None if session_dir is None else Path(session_dir).expanduser().resolve()
        workspace = workspace or ("isolated" if session_dir is not None else "temporary")
        if workspace not in WORKSPACE_MODES and workspace not in self._service.workspaces.records:
            raise ValueError(f"workspace must be one of {', '.join(WORKSPACE_MODES)} or the id of an existing workspace.")
        if workspace == "temporary" and session_dir is not None:
            raise ValueError("A session kept in session_dir needs a lasting workspace: use 'isolated' or 'current'.")
        if media_mode == "link" and self._service.plugin_job_storage(workspace, session_dir) == "dedicated":
            raise ValueError("Linked media need a workspace with shared storage; this job's workspace keeps its media in its own folder.")
        if self.unavailable_reason:
            raise RuntimeError(self.unavailable_reason)
        paths = self._service.plugin_resolve_media(list(media))
        return DeepyJob(self._service, self.owner, prompt, paths, media_mode=media_mode, session_dir=session_dir, workspace=workspace, title=title,
                        instructions=str(instructions or "").strip())

    def release_vram(self, timeout: float = 120.0) -> bool:
        """Take Deepy's model out of VRAM (its conversation stays, and comes back at its next request) once Deepy is idle: no job, request
        or generation running. Waits up to timeout seconds for that; returns whether it released."""
        service = self._service
        deadline = time.monotonic() + timeout
        with service._condition:
            while not (service._plugin_job is None and service._plugin_idle()):
                if time.monotonic() > deadline:
                    return False
                service._condition.wait(timeout=1)
        with service._mutation_lock:
            if service._plugin_job is not None or not service._plugin_idle():
                return False
            service._deps.controller.release_vram(service._state)
        return True

    @contextlib.contextmanager
    def turn(self, cancelled=lambda: False):
        """Deepy's turn, for a plugin's work on the GPU without Deepy (its own renders): it waits for the jobs and generations before it, in
        order with the jobs waiting, takes Deepy's model out of VRAM, and jobs started meanwhile wait until it ends. When cancelled() turns
        true while it waits, the body runs at once without the turn: check the cancel there."""
        service, holder = self._service, SimpleNamespace(owner=self.owner)
        with service._condition:
            service._plugin_waiters.append(holder)
        held = False
        try:
            while not held and not cancelled():
                with service._condition:
                    if not (service._plugin_waiters[0] is holder and service._plugin_job is None and service._plugin_idle()):
                        service._condition.wait(timeout=1)
                        continue
                with service._mutation_lock:  # as a job starts: idle checked here stays idle
                    if service._plugin_job is None and service._plugin_idle():
                        service._plugin_job, held = holder, True
                        service._deps.controller.release_vram(service._state)
        finally:
            with service._condition:
                service._plugin_waiters.remove(holder)
                service._condition.notify_all()
        try:
            yield
        finally:
            if held:
                with service._condition:
                    service._plugin_job = None
                    service._condition.notify_all()

    def ensure_workspace(self, name: str, folder: str | os.PathLike) -> str:
        """The id of the workspace keeping its media in folder, created (with this name) the first time."""
        return self._service.plugin_folder_workspace(name, Path(folder).expanduser().resolve())

    def web_session(self, workspace: str | None = None):
        """A WanGPSession whose tasks join the web UI's queue from any thread. With a workspace id, its generations record in that
        workspace without changing the gallery the user is viewing, as external agents' do."""
        from shared.api import WanGPSession
        service = self._service
        session = WanGPSession(webui_state=service._state, console_output=False, console_isatty=False, webui_append=True)
        session._gradio_webui_context = {"load_queue": service.load_inline_queue}
        if workspace is not None:
            service.workspaces.get(workspace)
            session._workspace_id, session._agent_context = workspace, f"plugin:{self.owner}"
        return session
