# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""One persistent Deepy service shared by Gradio and the standalone browser client."""
from __future__ import annotations

import json
import threading
import traceback
from contextlib import asynccontextmanager
from copy import deepcopy
from types import SimpleNamespace

from shared.deepy import plugin_jobs
from shared.deepy.drivers import GradioGallery
from shared.deepy.runtime import GenerationRuntime
from shared.deepy.service import DeepyService
from shared.utils.gallery_sources import GALLERY_SOURCES, SOURCES, STRIP_SOURCES, VIDEO
from shared.utils.generation_state import pause_state, set_generation_state
from shared.gradio.progress import ProgressUpdates, WangpProgress


class SharedState(dict):
    """Browser-local generation forms with shared queue, galleries and Deepy."""
    service = None
    shared_keys = {'gen', 'assistant_session', 'last_model_per_family', 'last_model_per_type', 'last_model_per_output_filter', 'last_resolution_per_group'}

    def __deepcopy__(self, memo):
        copied = type(self)()
        memo[id(self)] = copied
        copied.service = self.service
        for key, value in self.items():
            copied[key] = value if key in self.shared_keys else deepcopy(value, memo)
        return copied


def service_for(state):
    return state.service if isinstance(state, SharedState) else None


class HybridService(DeepyService):
    def __init__(self, deps, state, *, process_queue, finalize_queue, unload, gallery_lock=None, model_label=lambda: '', task_label=lambda task: ''):
        # Gradio historically initializes these lists on the first UI interaction.
        defaults = GenerationRuntime._build_state(SimpleNamespace(_deps=deps))['gen']
        defaults.pop('in_progress')
        for key, value in defaults.items():
            state['gen'].setdefault(key, value)
        self._host_dirty = threading.Event()
        self._host_changes = threading.Event()
        self._generation_updates = ProgressUpdates(self._host_dirty.set)
        self._host_signature = None
        self._gallery_signature = None
        self.gallery_revision = 0
        self.settings_revision = 0
        self.preview_revision = 0
        # Model lists mark downloaded models: rebuilt in every page only after a transfer completed.
        self.models_revision = 0
        self._published_models_revision = 0
        self._models_download = state['gen'].get('download_progress_completed')
        self._preview_image = None
        self._settings_signature = None
        self._catalog_revision = 0
        self._model_forms = {}
        self._model_forms_lock = threading.Lock()
        self._queue_worker = None
        self._queue_revision = 0
        self._gallery_update = None
        self.autoloaded_queue = False
        self._process_queue = process_queue
        self._finalize_queue = finalize_queue
        self._unload = unload
        self._model_label = model_label
        self._task_label = task_label
        self._generation_view = None
        self._generation_view_revision = 0
        self._progress_event = None
        self._generation_status = state['gen'].get('status')
        super().__init__(deps, state=state, gallery_factory=lambda deps, state: GradioGallery(deps, state, self.host_changed))
        self._progress_bar = WangpProgress(on_change=self._generation_updates.changed)
        state.service = self
        state['gen']['state_change_callback'] = self.generation_state_changed
        self._publish_generation_view(force=True)
        from shared.deepy.workspace_viewer import WorkspaceViewer
        self.gallery_lock = threading.RLock() if gallery_lock is None else gallery_lock
        self.workspace_viewer = WorkspaceViewer(self)
        self._host_worker = threading.Thread(target=self._publish_host_changes, name='Deepy UI synchronization', daemon=True)
        self._host_worker.start()
        self.host_changed()
        plugin_jobs.attach_service(self)

    def record_model_form(self, model_type, settings):
        snapshot = deepcopy(settings)
        with self._model_forms_lock:
            self._model_forms[model_type] = snapshot

    def load_model_form(self, model_type):
        with self._model_forms_lock:
            snapshot = self._model_forms.get(model_type)
        return deepcopy(snapshot) if snapshot is not None else None

    def host_changed(self):
        self._host_changes.set()
        self._host_dirty.set()

    def generation_state_changed(self):
        self._host_changes.set()
        self._generation_updates.changed(immediate=True)

    def _generation_event(self, command, data):
        # Callbacks only record progress and wake the shared publisher. Rendering
        # and fan-out never run on the inference/download thread.
        with self._condition:
            if command == 'exit':
                self._progress_event = None
                self._generation_updates.changed(immediate=True)
                self._generation_aborting = False
                return
            if command not in {'progress', 'status', 'download'}:
                return
            if self._progress_event is None:
                self._progress_bar = WangpProgress(on_change=self._generation_updates.changed)
            self._progress_event = {'type': command, 'data': data}
            if command == 'progress':
                self._progress_bar(*data)
            elif command == 'status':
                self._generation_status = self._state['gen'].get('status')
                self._progress_bar.status(data)
            else:
                self._progress_bar.set_download(data)

    def _publish_generation_view(self, force=False):
        gen = self._state['gen']
        with self._condition:
            progress = self._progress_bar
            status = gen.get('status')
            if status != self._generation_status or progress.position is None:
                self._generation_status = status
                progress.status(status or 'Ready')
            if not self._generation_updates.consume(force=force):
                return
            pending, paused = pause_state(gen)
            active, aborting = bool(gen.get('status_display')), bool(gen.get('abort')) or self._generation_aborting
            task = gen.get('api_active_queue_task')
            model_name = self._task_label(task) if active and task is not None else self._model_label()
            html = None if gen.get('model_preloading') else progress.render(active=active, aborting=aborting, hold_complete=True, model_name=model_name)
            view = {'html': html, 'pending': pending, 'paused': paused}
            current = None if self._progress_event is None else {**self._progress_event, 'description': progress.description, 'aborting': aborting, 'html': progress.render(compact=True, aborting=aborting, hold_complete=True)}
            if current != self._progress:
                self._progress = current
                self.publish('progress', current)
            self._generation_updates.hold(progress.completion_delay if not aborting and (html is not None or current is not None) else 0)
            if view != self._generation_view:
                self._generation_view = view
                self._generation_view_revision += 1
                self.publish('generation_state', self.generation_view())

    def generation_view(self):
        with self._condition:
            return {'instance': self._instance_id, 'revision': self._generation_view_revision, **self._generation_view}

    def select_gallery_view(self, payload):
        with self._mutation_lock:
            # A click sent before another page switched workspace is obsolete.
            if payload['workspace'] != self.workspace_id:
                return
            source = payload['source']
            if source not in SOURCES:
                raise ValueError('Invalid gallery source.')
            gen = self._state['gen']
            path = payload['path']
            if path is None:
                gen['current_gallery_source'] = source
                if source != VIDEO.id:
                    gen['selected_video_time'] = None
            else:
                paths, _ = self.gallery._resolve_lists(source)
                index = payload['index']
                if not isinstance(index, int) or index < 0 or index >= len(paths) or paths[index] != path:
                    return
                self.gallery._select_index(index, source)
            self.host_changed()

    def publish_workspaces(self):
        super().publish_workspaces()
        self.host_changed()

    def _publish_host_changes(self):
        while True:
            self._host_dirty.wait(self._generation_updates.delay())
            self._host_dirty.clear()
            if self._closing:
                return
            try:
                with self._mutation_lock:
                    gen = self._state['gen']
                    host_changed = self._host_changes.is_set()
                    if host_changed:
                        self._host_changes.clear()
                    self._publish_generation_view(force=host_changed)
                    if not host_changed:
                        continue
                    for source in GALLERY_SOURCES:
                        if gen[source.last_selected_key]:
                            gen[source.selected_key] = len(gen[source.files_key]) - 1
                    if gen.pop('refresh_tab', False):
                        gen['current_gallery_source'] = gen['last_gallery_source']
                    # (files of every gallery, selections, active gallery, settings identities, gallery size limit)
                    signature = (tuple(tuple(gen[source.files_key]) for source in GALLERY_SOURCES), tuple(gen[source.selected_key] for source in GALLERY_SOURCES), gen['current_gallery_source'], tuple(tuple(map(id, list.__iter__(gen[source.settings_key]))) for source in GALLERY_SOURCES), self._deps.get_server_config().get('clear_file_list', 5))
                    gallery_view = (signature, self.generation_running, self._gallery_update, tuple(map(id, gen['queue'])))  # the page refresh also renders the queue
                    settings = (json.dumps(self._session.tool_ui_settings, sort_keys=True), self._session.storage_session_id, self._session.storage_title, self._catalog_revision)
                    changed = False
                    preview = gen.get('preview')
                    if preview is not self._preview_image:
                        self._preview_image = preview
                        self.preview_revision += 1
                        changed = True
                    if signature != self._gallery_signature:
                        media_changed = self._gallery_signature is None or signature[0] != self._gallery_signature[0]
                        self._gallery_signature = signature
                        self.publish('gallery', self.gallery_snapshot())
                        if media_changed:
                            self.save_session()
                    if gallery_view != self._host_signature:
                        self._host_signature = gallery_view
                        self.gallery_revision += 1
                        changed = True
                    if settings != self._settings_signature:
                        self._settings_signature = settings
                        self.settings_revision += 1
                        changed = True
                    if self.models_revision != self._published_models_revision:
                        self._published_models_revision = self.models_revision
                        changed = True
                    if changed:
                        self.publish('host_view', self.host_view())
            except Exception:
                traceback.print_exc()
                self.publish_error('Could not synchronize the Gradio view.')

    def host_view(self):
        return {'gallery': self.gallery_revision, 'settings': self.settings_revision, 'preview': self.preview_revision, 'models': self.models_revision}

    def snapshot(self):
        result = super().snapshot()
        result['hybrid'] = True
        result['host_view'] = self.host_view()
        result['forms'] = self.forms.revisions()
        result['generation_state'] = self.generation_view()
        return result

    def command(self, command, data=None):
        super().command(command, data)
        if command in {'refresh_gallery', 'load_queue_trigger'}:
            self.host_changed()

    def _run_worker(self, task, acknowledged_ids):
        try:
            super()._run_worker(task, acknowledged_ids)
        finally:
            self.host_changed()

    def publish_sessions(self):
        self._catalog_revision += 1
        super().publish_sessions()
        self.host_changed()

    def update_settings(self, values, **kwargs):
        result = super().update_settings(values, **kwargs)
        self.host_changed()
        return result

    def control(self, action, payload):
        result = super().control(action, payload)
        self.host_changed()
        return result

    @property
    def generation_running(self):
        return self._queue_worker is not None

    def start_generation(self):
        with self._mutation_lock:
            if self._queue_worker is None and self._state['gen']['queue']:
                self._queue_worker = threading.Thread(target=self._run_generation, name='WanGP generation service', daemon=True)
                self._queue_worker.start()
            return self._queue_worker

    def _run_generation(self):
        queue = self._state['gen']['queue']
        before = self.gallery.counts()
        set_generation_state(self._state['gen'], status_display=True)
        self._state["gen"]["last_progress_args"] = None
        self._generation_event("status", "Preparing generation…")
        try:
            for updates in self._process_queue(self._state):
                with self._condition:
                    self._queue_revision += 1
                    # Only output-trigger timestamps request a gallery/details
                    # refresh. Preview frames yield an empty Text component here.
                    if isinstance(updates[1], (int, float)):
                        self._gallery_update = updates[1]
                    self._condition.notify_all()
                self.host_changed()
        except Exception as error:
            if getattr(error, 'print_exception', True):
                traceback.print_exc()
            # The service broadcasts the error once to every connected client.
            # Observers must not raise it again through their own Gradio requests.
            self.publish_error(error)
        finally:
            try:
                self._finalize_queue(self._state)
                self.gallery.sync_latest_generated(before)
                self._unload(self._state)
            finally:
                with self._condition:
                    self._queue_worker = None
                    self._active_generation_client_id = ''
                    self._condition.notify_all()
                self._generation_event('exit', None)
                self._refresh_models()  # files of failed or cancelled loads, and other assets
                self.host_changed()
        # Tasks loaded after this worker emptied its queue went to a new list it never read.
        if self._state['gen']['queue'] is not queue:
            self.start_generation()

    def generation_event(self, command, data):
        queue = self._state['gen']['queue']
        if queue:
            self._active_generation_client_id = str(queue[0]['params'].get('client_id', ''))
        if command in {'progress', 'status'}:
            self._generation_event(command, data)
        elif command == 'output':
            self.host_changed()
        elif command == 'refresh_models':
            self._refresh_models()

    def _refresh_models(self):
        download = self._state['gen'].get('download_progress_completed')
        if download is not self._models_download:
            self._models_download = download
            self.models_revision += 1
            self.host_changed()

    def load_inline_queue(self):
        """Admit API submissions into the live queue without a browser request or waiting for running work."""
        self._deps.load_queue_action(None, self._state, SimpleNamespace(target=1))
        self.start_generation()
        self.host_changed()  # every page lists the new tasks, also behind a running generation

    def _process_inline_queue(self, payload=None):
        self._deps.load_queue_action(None, self._state, SimpleNamespace(target=1))
        worker = self.start_generation()
        if worker is not None:
            worker.join()

    def gradio_generation(self):
        """Observe the server worker; cancelling a browser request cannot stop it."""
        import gradio as gr
        worker = self.start_generation()
        revision = -1
        while worker is not None:
            with self._condition:
                self._condition.wait_for(lambda: self._queue_revision != revision or self._queue_worker is not worker, timeout=1)
                changed = self._queue_revision != revision
                revision = self._queue_revision
                finished = self._queue_worker is not worker
            if changed:
                # Shared view revisions deliver galleries, previews and model lists
                # to every page. Do not refresh them a second time in the initiating page.
                yield gr.update(), gr.update(), gr.update()
            if finished:
                break

    def finalized_updates(self):
        import gradio as gr
        # The revisioned view refresh publishes current state to every browser.
        # A late Gradio callback must not replay an earlier generation's selection.
        # Gallery tabs, current tab, image/video gallery, 3 values per strip gallery, then 6 generation controls.
        return (gr.update(),) * (3 + 3 * len(STRIP_SOURCES) + 6)

    def close(self):
        plugin_jobs.detach_service(self)
        super().close()
        self._host_dirty.set()
        if threading.current_thread() is not self._host_worker:
            self._host_worker.join(5)

    @asynccontextmanager
    async def lifespan(self, app, *, base_url, auth=None, voice_language=None, https_port=None):
        from shared.deepy.server import create_app
        from shared.utils.plugin_navigation import mount_web_apps
        from starlette.concurrency import run_in_threadpool
        app.mount('/deepy', create_app(self, auth=auth, voice_language=voice_language, https_port=https_port))
        from shared.apps import server as apps_server
        apps_server.mount(app, auth.public_url if auth is not None else None)
        mount_web_apps(app, base_url)
        try:
            await run_in_threadpool(self._deps.load_queue_action, None, self._state, SimpleNamespace(target=None))
            self.start_generation()
            yield
        finally:
            self.close()


def launch_gradio(demo, service, args, **kwargs):
    """Use Gradio's supported lifespan hook to mount the shared Web application."""
    from shared.deepy.server import server_options
    if service.workspaces is not None:
        def allow_workspace_media(directory):
            if directory not in demo.allowed_paths:
                demo.allowed_paths.append(directory)
        service._deps.callbacks.handlers['allow_workspace_media'] = allow_workspace_media
        kwargs['allowed_paths'] = [*kwargs.get('allowed_paths', []), *(str(service.workspaces._directory(record)) for record in service.workspaces.records.values() if 'folder' in record)]
    if args.mcp_webui is None:
        args.mcp_webui = service._deps.get_server_config().get('mcp_webui', False)
    host, port, cert, key, https_port, auth = server_options(args)
    print("WanGP web authentication disabled." if auth.gate is None else "WanGP web authentication enabled.")
    scheme = 'https' if cert and https_port is None else 'http'
    print(f"Deepy Web app: {scheme}://{host}:{port}/deepy/")
    if args.mcp_webui:
        from shared.mcp.server import build_webui_server
        mcp_server = build_webui_server(service, args)

    @asynccontextmanager
    async def lifespan(app):
        demo.run_startup_events()
        await demo.run_extra_startup_events()
        app.startup_events_triggered = True
        base_url = auth.public_url or f'{"https" if cert else "http"}://{host}:{https_port or port}'
        async with service.lifespan(app, base_url=base_url, auth=auth, voice_language=args.deepy_voice_language, https_port=https_port):
            # Agents start after the saved queue has been restored.
            if args.mcp_webui:
                threading.Thread(target=mcp_server.run, name='WanGP MCP server', daemon=True).start()
            yield

    from starlette.middleware import Middleware
    from shared.authentication.web import GradioStartupMiddleware, WebAuthMiddleware
    from shared.authentication.tls import HTTPSRedirect
    from shared.gradio.proxy_public_url import PublicURLMiddleware
    from shared.apps.server import AppFrameGuard
    from shared.utils.plugin_navigation import AddonNavigationMiddleware
    middleware = [Middleware(AppFrameGuard), Middleware(WebAuthMiddleware, auth=auth), Middleware(AddonNavigationMiddleware)]
    if auth.public_url is not None:
        middleware.insert(0, Middleware(PublicURLMiddleware, public_url=auth.public_url))
    if https_port is not None:
        middleware.insert(0, Middleware(HTTPSRedirect, port=https_port))
    middleware.insert(0, Middleware(GradioStartupMiddleware))
    demo.launch(**kwargs, app_kwargs={"lifespan": lifespan, "middleware": middleware}, ssl_certfile=cert if https_port is None else None, ssl_keyfile=key if https_port is None else None, ssl_verify=False if cert else True, prevent_thread_lock=https_port is not None)
    if https_port is not None:
        import uvicorn
        print(f"Deepy HTTPS app: https://{host}:{https_port}/deepy/")
        try:
            uvicorn.run(demo.app, host=host, port=https_port, ssl_certfile=cert, ssl_keyfile=key, lifespan='off', timeout_graceful_shutdown=5)
        finally:
            demo.close()
