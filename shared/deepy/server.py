# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""HTTP API and standalone UI for a persistent DeepyService."""
from __future__ import annotations

import asyncio
import json
import os
import re
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from starlette.websockets import WebSocketState

from shared.deepy import chat, session_store
from shared.deepy.errors import error_payload
from shared.deepy.gallery import _AUDIO_EXTENSIONS, _IMAGE_EXTENSIONS, _VIDEO_EXTENSIONS
from shared.deepy.voice import mount_voice_routes, save_upload
from shared.utils.downloads import _install_routes_on_app, registered_download_path
from shared.utils.media_imports import persist_gallery_import
from shared.utils.gallery_sources import MESH_EXTENSIONS, MESH_IMPORT_EXTENSIONS
from shared.utils.mesh_outputs import import_mesh_file
from shared.authentication import password
from shared.authentication.web import WebAuthentication, WebAuthMiddleware
from shared.authentication.tls import HTTPSRedirect, run_http_and_https, tls_options

WEB = Path(__file__).with_name("web")


class Message(BaseModel):
    text: str = Field(min_length=1, max_length=100000)
    submission_id: str = Field(min_length=1, max_length=128)
    steering: bool = False


class Control(BaseModel):
    action: Literal["stop", "pause", "reset", "resume", "queued", "abort", "rename", "delete"]
    payload: dict = Field(default_factory=dict)


def create_app(service, *, token=None, auth=None, voice_language=None, https_port=None):
    from shared.utils.network_diagnostics import install_network_diagnostics
    install_network_diagnostics()

    @asynccontextmanager
    async def lifespan(app):
        yield
        service.close()

    app = FastAPI(title="Deepy", lifespan=lifespan)
    auth = auth if auth is not None else WebAuthentication(token)
    app.add_middleware(WebAuthMiddleware, auth=auth)
    if https_port is not None:
        app.add_middleware(HTTPSRedirect, port=https_port)

    @app.exception_handler(ValueError)
    async def invalid_value(request, error):
        payload = error_payload(error)
        return JSONResponse({"detail": payload['message'], "error": payload}, status_code=400)

    from gradio import Error as GradioError
    app.add_exception_handler(session_store.SessionStoreError, invalid_value)
    app.add_exception_handler(GradioError, invalid_value)

    def chat_page(name):
        shell = chat.render_shell_html(service._deps.controller.get_deepy_type())
        page = (WEB / name).read_text(encoding="utf-8").replace("<!-- CHAT -->", shell).replace("<!-- STATS -->", chat.render_stats_html())
        def version_asset(match):
            name = match[1]
            path = WEB.parents[1] / 'gradio' / name if name in {'form_sync.js', 'progress.css', 'image_zoom.js', 'image_zoom.css'} else WEB / name
            if name == 'mesh_viewer.js':
                path = WEB.parents[1] / 'gradio/strip_gallery.py'
            return f'assets/{name}?v={path.stat().st_mtime_ns}'
        page = re.sub(r'assets/([\w.-]+\.(?:js|css))', version_asset, page)
        if https_port is not None:
            page = page.replace("<body data-deepy-app>", f'<body data-deepy-app data-deepy-https-port="{int(https_port)}">')
        return HTMLResponse(page, headers={"Cache-Control": "no-cache"})

    @app.get("/", response_class=HTMLResponse)
    def index():
        return chat_page("app.html")

    @app.get("/window", response_class=HTMLResponse)
    def window():  # Deepy's chat window as a component, for a plugin's page
        return chat_page("window.html")

    @app.get("/assets/{name}")
    def asset(name: str):
        if name == 'mesh_viewer.js':
            from shared.gradio.strip_gallery import MeshGallery
            return Response(MeshGallery.get_viewer_javascript('assets/mesh/'), media_type='text/javascript')
        if name in {'form_sync.js', 'progress.css', 'image_zoom.js', 'image_zoom.css'}:
            return FileResponse(WEB.parents[1] / 'gradio' / name)
        if name == "icon.png":
            return FileResponse(WEB.parents[2] / "favicon.png")
        if name not in {"window.js", "window.css", "chat.js", "compact_actions.js", "chat.css", "voice.js", "app.js", "app.css", "gradio_transport.js", "manifest.webmanifest", "icon.svg", "transport.js", "hybrid_transport.js", "workspaces.js", "workspaces.css", "media_view.js", "app_player.js", "app_player.css"}:
            raise HTTPException(404)
        return FileResponse(WEB / name, media_type="application/manifest+json" if name == "manifest.webmanifest" else None)

    @app.get('/assets/mesh/{name:path}')
    def mesh_asset(name: str):
        if name not in {'model-viewer.min.js', 'spark/three.module.js', 'spark/three.core.js', 'spark/spark.module.min.js', 'spark/OrbitControls.js', 'spark/Pass.js'}:
            raise HTTPException(404)
        return FileResponse(WEB.parents[1] / 'gradio/vendor' / name, media_type='text/javascript')

    @app.get("/deepy_api/state")
    def state():
        return service.snapshot()

    if service.workspace_viewer is not None:
        from shared.deepy.workspace_viewer_api import mount_workspace_viewer
        mount_workspace_viewer(app, service)

    @app.post('/deepy_api/workspaces/{action}')
    def workspace_action(action: Literal['create', 'rename', 'delete', 'select'], payload: dict):
        return service.workspace_control(action, payload)

    @app.get("/deepy_api/media/{media_id}/info")
    def media_info(media_id: str):
        with service._mutation_lock:
            if media_id not in service.gallery._media_paths:
                raise HTTPException(404, "Media no longer in this workspace.")
            return {"html": service.gallery.media_info(media_id)}

    @app.get("/deepy_api/media/{media_id}/file")
    def media_file(media_id: str, download: bool = False):
        from shared.utils.http_disconnect import DisconnectAwareFileResponse
        try:
            path = Path(service.gallery.media_path(media_id))
        except KeyError:
            raise HTTPException(404, "Media no longer in this workspace.")
        if not path.is_file():
            raise HTTPException(404, "Media file not found.")
        return DisconnectAwareFileResponse(path, filename=path.name, content_disposition_type='attachment' if download else 'inline')

    @app.get("/deepy_api/media/{media_id}/thumbnail")
    async def media_thumbnail(media_id: str):
        from fastapi import Response
        from shared.deepy.thumbnails import render_thumbnail
        try:
            path = service.gallery.media_path(media_id)
            if service.gallery._detect_media_type(path) not in ('image', 'video', '3d'):
                raise HTTPException(400, "Media has no visual thumbnail.")
            data = await render_thumbnail(path)
        except (KeyError, FileNotFoundError):
            raise HTTPException(404, "Media file not found.")
        if not data:
            raise HTTPException(404, "Thumbnail not available.")
        return Response(data, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})

    @app.get("/deepy_api/display-settings")
    def display_settings():
        return service.display_settings()

    @app.post("/deepy_api/display-settings")
    def update_display_settings(body: dict):
        return service.update_display_settings(body)

    @app.get("/deepy_api/settings")
    def settings():
        return service.settings()

    @app.post("/deepy_api/settings")
    def update_settings(body: dict):
        return service.submit_settings(body['baseline'], body['values']) if 'baseline' in body else service.update_settings(body)

    async def event_batches(after, instance):
        while not service._closing:
            events = await run_in_threadpool(service.events_after, after, instance=instance)
            if events:
                after = events[-1]['id']
                if events[-1]['type'] == 'snapshot':
                    instance = events[-1]['data']['instance']
            yield events

    @app.get("/deepy_api/events")
    async def events(request: Request, after: int = 0, instance: str | None = None):
        async def stream():
            async for events in event_batches(after, instance):
                if await request.is_disconnected():
                    return
                if not events:
                    yield ": keepalive\n\n"
                for event in events:
                    yield "id: " + str(event['id']) + "\ndata: " + json.dumps(event, ensure_ascii=False) + "\n\n"
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/deepy_api/events/poll")
    async def poll_events(after: int = 0, instance: str | None = None):
        # Short requests avoid consuming the browser's connection pool when
        # Gradio and Deepy are open in multiple tabs behind a proxy.
        events = await run_in_threadpool(service.events_after, after, 0, instance)
        return JSONResponse(events, headers={"Cache-Control": "no-store"})

    @app.websocket('/deepy_api/events')
    async def websocket_events(socket: WebSocket, after: int = 0, instance: str | None = None):
        await socket.accept()

        async def send_events():
            async for events in event_batches(after, instance):
                for event in events:
                    # Buffered sends may never suspend. Let the receiver and
                    # Windows transport process a disconnect between messages.
                    await asyncio.sleep(0)
                    if socket.client_state == WebSocketState.DISCONNECTED:
                        return
                    try:
                        await socket.send_json(event)
                    except RuntimeError:
                        # Uvicorn can finish the response as the receive task observes
                        # a closed browser, before the pending sender is cancelled.
                        if socket.client_state == WebSocketState.DISCONNECTED:
                            return
                        raise
            await socket.close()

        async def disconnected():
            async for _ in socket.iter_text():
                pass

        sender = asyncio.create_task(send_events())
        receiver = asyncio.create_task(disconnected())
        try:
            done, _ = await asyncio.wait([sender, receiver], return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                with suppress(WebSocketDisconnect):
                    task.result()
        finally:
            for task in (sender, receiver):
                task.cancel()
            await asyncio.gather(sender, receiver, return_exceptions=True)

    @app.post("/deepy_api/messages", status_code=202)
    def message(body: Message):
        if not body.text.strip():
            raise HTTPException(400, "The request is empty.")
        queued = service.submit(body.text, body.submission_id, steering=body.steering)
        return {"submission_id": body.submission_id, "queued_for_restoration": queued}

    @app.post("/deepy_api/control")
    def control(body: Control):
        if body.action == "resume" and not isinstance(body.payload.get("id"), str):
            raise HTTPException(400, "A saved session id is required.")
        if body.action == "rename" and not all(isinstance(body.payload.get(key), str) for key in ('id', 'title')):
            raise HTTPException(400, "A session id and title are required.")
        return service.control(body.action, body.payload)

    @app.post("/deepy_api/media")
    async def upload(file: UploadFile = File(...), from_chat: bool = Form(False)):
        session_epoch = service._session.chat_epoch
        with service._mutation_lock:
            service.prepare_media_workspace()
            workspace_id = service.workspace_id
            output = Path(service.workspace_output_config(workspace_id)['save_path'])
        output.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix='.import-', dir=output) as directory:
            source = await save_upload(file, Path(directory), limit=1024 * 1024 * 1024, extensions=_IMAGE_EXTENSIONS | _VIDEO_EXTENSIONS | _AUDIO_EXTENSIONS | set(MESH_IMPORT_EXTENSIONS), preserve_filename=True)
            def add():
                with service._mutation_lock:
                    if service.workspace_id != workspace_id:
                        raise ValueError('The workspace changed during upload. Please import the file again.')
                    if Path(source).suffix.lower() in MESH_IMPORT_EXTENSIONS:
                        path, reused = import_mesh_file(source, service.workspace_output_config(workspace_id)['mesh_save_path'])
                    else:
                        path, reused = persist_gallery_import(source, output, move=True)
                    try:
                        return service.import_media(path, session_epoch=session_epoch, from_chat=from_chat)
                    except Exception:
                        if not reused:
                            Path(path).unlink(missing_ok=True)
                        raise
            media_id = await run_in_threadpool(add)
        with service._mutation_lock:
            return {"id": media_id, "gallery": service.gallery_snapshot()}

    @app.post('/deepy_api/media/unattach')
    def unattach(payload: dict):
        return {'event': service.remove_chat_upload(payload.get('chat_session_id'), payload.get('media_id'))}

    @app.get("/deepy_api/media/locate")
    def locate_media(url: str):
        return service.locate_gallery_media(url)

    @app.get("/deepy_api/media/mesh-viewer")
    def mesh_viewer(url: str):
        from shared.gradio.strip_gallery import MeshGallery
        path = registered_download_path(url)
        if path is None or Path(path).suffix.lower() not in MESH_EXTENSIONS:
            raise HTTPException(404, "3D asset not found.")
        return {"html": MeshGallery.viewer_html(path, url)}

    @app.post("/deepy_api/media/{media_id}/select")
    def select(media_id: str):
        with service._mutation_lock:
            service.select_media(media_id)
            return {"gallery": service.gallery_snapshot()}

    _install_routes_on_app(app)
    mount_voice_routes(app, get_service=lambda: service, language=voice_language)
    return app


def server_options(args):
    if not args.mcp_webui and (args.mcp_auth or args.mcp_auth_password is not None or args.mcp_auth_url is not None):
        raise ValueError("MCP OAuth options require --mcp or --mcp-webui with a network transport.")
    host = "0.0.0.0" if args.listen else args.server_name or os.getenv("SERVER_NAME", "localhost")
    port = int(args.server_port) or int(os.getenv("SERVER_PORT", "7860"))
    cert, key, https_port = tls_options(args, port)
    if args.public_url is not None and https_port is not None:
        raise ValueError('Use --public-url for proxy-managed HTTPS or --https-port for WanGP redirects, not both.')
    return host, port, cert, key, https_port, WebAuthentication(password(args), public_url=args.public_url)


def run_server(deps, args):
    import uvicorn
    from shared.deepy.service import DeepyService

    host, port, cert, key, https_port, auth = server_options(args)
    print(f"Deepy server: {'https' if cert and https_port is None else 'http'}://{host}:{port}")
    if https_port is not None:
        print(f"Deepy HTTPS: https://{host}:{https_port}")
    print("WanGP web authentication disabled." if auth.gate is None else "WanGP web authentication enabled.")
    service = DeepyService(deps)
    service.configure_workspaces(args.workspaces_dir or service._deps.get_server_config().get('workspaces_dir') or str(WEB.parents[2] / 'workspaces'))
    app = create_app(service, auth=auth, voice_language=args.deepy_voice_language, https_port=https_port)
    from shared.apps import server as apps_server
    apps_server.mount(app, auth.public_url)
    app.add_middleware(apps_server.AppFrameGuard)  # outermost: runs before the authentication
    if https_port is None:
        uvicorn.run(app, host=host, port=port, ssl_certfile=cert, ssl_keyfile=key)
    else:
        run_http_and_https(uvicorn.Config(app, host=host, port=port, lifespan="off", timeout_graceful_shutdown=5), uvicorn.Config(app, host=host, port=https_port, ssl_certfile=cert, ssl_keyfile=key, timeout_graceful_shutdown=5))
    return 0
