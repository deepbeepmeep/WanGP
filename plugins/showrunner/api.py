# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""The Showrunner web application: its page, its assets, a JSON API over the productions and a live event stream."""
from __future__ import annotations

import asyncio
import json
import re
import shutil
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from . import production as model, titles

WEB = Path(__file__).with_name("web")
GUIDE = Path(__file__).resolve().parents[2] / "docs" / "SHOWRUNNER.md"
ASSET = re.compile(r"(?:js|css)/[\w.-]+\.(?:js|css)|icons/[\w.-]+\.svg|[\w.-]+\.(?:svg|png|webmanifest)")


class Created(BaseModel):
    title: str = Field(default="Untitled production", max_length=200)
    kind: str = "free"
    brief: dict[str, Any] = Field(default_factory=dict)


class Operations(BaseModel):
    ops: list[dict[str, Any]]
    base_revision: int | None = None


class DirectorRequest(BaseModel):
    task: str
    message: str = ""
    shots: list[str] = Field(default_factory=list)


class TemplatePick(BaseModel):
    role: str
    template: str | None = None


class Archived(BaseModel):
    archived: bool = True


class Text(BaseModel):
    text: str = Field(default="", max_length=2000)
    subtitle: str = Field(default="", max_length=2000)
    style: str = "title"
    design: str = ""
    picture: str = ""
    start_s: float = 0.0
    duration_s: float = 3.0
    fade_s: float = 0.5


class TextChange(BaseModel):
    fields: dict[str, Any]


class BringBack(BaseModel):
    kind: str
    item: str


class Resume(BaseModel):
    message: str = Field(default="", max_length=20000)


class Steer(BaseModel):
    text: str = Field(min_length=1, max_length=20000)


def create_app(showrunner, *, title: str = "Showrunner") -> FastAPI:
    app = FastAPI(title=title, docs_url=None, redoc_url=None, openapi_url=None)
    store = showrunner.store

    def production(production_id: str):
        try:
            return store.get(production_id)
        except KeyError as error:
            raise HTTPException(404, str(error.args[0]))

    @app.exception_handler(model.OperationError)
    async def refused(request, error):
        return JSONResponse({"detail": str(error), "current": error.current}, status_code=409)

    @app.exception_handler(ValueError)
    async def invalid(request, error):
        return JSONResponse({"detail": str(error)}, status_code=400)

    @app.exception_handler(RuntimeError)
    async def unavailable(request, error):
        return JSONResponse({"detail": str(error)}, status_code=409)

    @app.exception_handler(PermissionError)
    async def forbidden(request, error):
        return JSONResponse({"detail": str(error)}, status_code=403)

    @app.get("/", response_class=HTMLResponse)
    def index():
        page = (WEB / "index.html").read_text(encoding="utf-8")
        # Stylesheets and icons get a version; ES modules must keep one URL each (an import of ./app.js would otherwise load a second app).
        page = re.sub(r'href="(assets/)([^"?]+)"', lambda match: f'href="{match[1]}{match[2]}?v={(WEB / match[2]).stat().st_mtime_ns}"' if (WEB / match[2]).is_file() else match[0], page)
        return HTMLResponse(page, headers={"Cache-Control": "no-cache"})

    @app.get("/assets/{name:path}")
    def asset(name: str):
        if not ASSET.fullmatch(name) or not (WEB / name).is_file():
            raise HTTPException(404)
        return FileResponse(WEB / name, headers={"Cache-Control": "no-cache"})

    @app.get("/api/state")
    def state():
        return {"productions": store.catalog(), "director": showrunner.host.director_info(), "kinds": model.KINDS, "aspects": model.ASPECTS, "transitions": model.TRANSITIONS,
                "text_styles": titles.STYLES}

    @app.get("/api/guide")
    def guide():
        """The guide's pages, read when the page opens it: the sections of docs/SHOWRUNNER.md (its introduction as Overview), as HTML."""
        import markdown
        intro, *parts = re.split(r"^## ", GUIDE.read_text(encoding="utf-8").split("\n---\n")[0], flags=re.MULTILINE)
        pages = [("Overview", intro.split("\n", 1)[1]), *(part.split("\n", 1) for part in parts)]
        return {"pages": [{"title": title.strip(), "html": markdown.markdown(body.strip(), extensions=["tables", "sane_lists"], output_format="html5")} for title, body in pages]}

    @app.get("/api/templates")
    def default_template_roles():
        return showrunner.default_template_roles()

    @app.post("/api/templates")
    def choose_default_template(request: TemplatePick):
        return showrunner.choose_default_template(request.role, request.template or None)

    @app.post("/api/productions")
    def create(request: Created):
        created = showrunner.create(request.title, request.kind if request.kind in model.KINDS else "free", **{key: value for key, value in request.brief.items() if key in model.BRIEF_FIELDS - {"title", "kind"}})
        return created.snapshot()

    @app.get("/api/productions/{production_id}")
    def read(production_id: str):
        return production(production_id).snapshot()

    @app.post("/api/productions/{production_id}/ops")
    def operate(production_id: str, request: Operations):
        target = production(production_id)
        results = target.apply(request.ops, by="user", base_revision=request.base_revision)
        return {"results": results, "doc": target.snapshot()}

    @app.post("/api/productions/{production_id}/undo")
    def undo(production_id: str):
        production(production_id)
        return showrunner.undo(production_id)

    @app.post("/api/productions/{production_id}/redo")
    def redo(production_id: str):
        production(production_id)
        return showrunner.undo(production_id, redo=True)

    @app.get("/api/productions/{production_id}/versions/{deliverable_id}")
    def version(production_id: str, deliverable_id: str, against: str = "current"):
        production(production_id)
        return showrunner.version(production_id, deliverable_id, against)

    @app.post("/api/productions/{production_id}/versions/{deliverable_id}/bring")
    def bring_back(production_id: str, deliverable_id: str, request: BringBack):
        production(production_id)
        return showrunner.bring_back(production_id, deliverable_id, request.kind, request.item)

    @app.post("/api/productions/{production_id}/versions/{deliverable_id}/restore")
    def restore_version(production_id: str, deliverable_id: str):
        production(production_id)
        return showrunner.restore_version(production_id, deliverable_id)

    @app.post("/api/productions/{production_id}/versions/{deliverable_id}/duplicate")
    def duplicate_version(production_id: str, deliverable_id: str):
        production(production_id)
        return showrunner.duplicate_version(production_id, deliverable_id)

    @app.post("/api/productions/{production_id}/duplicate")
    def duplicate(production_id: str):
        production(production_id)
        return store.duplicate(production_id).snapshot()

    @app.post("/api/productions/{production_id}/archive")
    def archive(production_id: str, request: Archived):
        production(production_id)
        store.archive(production_id, request.archived)
        return {"archived": request.archived}

    @app.delete("/api/productions/{production_id}")
    def trash(production_id: str):
        production(production_id)
        showrunner.director.stop(production_id)
        store.trash(production_id)
        return {"trashed": production_id}

    @app.post("/api/productions/{production_id}/upload")
    async def upload(production_id: str, file: UploadFile = File(...), role: str = Form("bin"), target: str | None = Form(None)):
        production(production_id)
        if role not in ("bin", "song", "reference", "cast", "take", "track", "subtitles"):
            raise HTTPException(400, "Unknown upload role.")
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename or "").suffix) as handle:
            shutil.copyfileobj(file.file, handle)
            temporary = Path(handle.name)
        try:
            return await run_in_threadpool(showrunner.upload, production_id, file.filename or temporary.name, temporary, role, target or None)
        finally:
            temporary.unlink(missing_ok=True)

    @app.get("/api/productions/{production_id}/file")
    def media_file(production_id: str, path: str, download: bool = False):
        target = production(production_id)
        file = target.resolve(path)
        if file is None or not file.is_file():
            raise HTTPException(404, "File not found.")
        if not target.owns(file):
            raise HTTPException(403, "This file does not belong to the production.")
        return FileResponse(file, filename=file.name if download else None, headers={"Cache-Control": "private, max-age=3600"})  # a download keeps its name

    @app.get("/api/productions/{production_id}/analysis")
    async def analysis(production_id: str, refresh: bool = False):
        production(production_id)
        return {"analysis": await run_in_threadpool(showrunner.analysis, production_id, refresh)}

    @app.get("/api/productions/{production_id}/waveform")
    async def waveform(production_id: str, path: str):
        production(production_id)
        return await run_in_threadpool(showrunner.waveform, production_id, path)

    @app.get("/api/productions/{production_id}/subtitles")
    async def subtitles(production_id: str, path: str):
        production(production_id)
        return {"vtt": await run_in_threadpool(showrunner.subtitles, production_id, path)}

    @app.get("/api/productions/{production_id}/rates")
    async def shoot_rates(production_id: str):
        production(production_id)
        return await run_in_threadpool(showrunner.shoot_rates, production_id)

    @app.get("/api/productions/{production_id}/director")
    def director_status(production_id: str):
        production(production_id)
        return showrunner.director.status(production_id)

    @app.post("/api/productions/{production_id}/director")
    def direct(production_id: str, request: DirectorRequest):
        production(production_id)
        return showrunner.director.start(production_id, request.task, request.message, request.shots)

    @app.post("/api/productions/{production_id}/director/steer")
    def steer(production_id: str, request: Steer):
        production(production_id)
        showrunner.director.steer(production_id, request.text)
        return showrunner.director.status(production_id)

    @app.post("/api/productions/{production_id}/director/pause")
    def pause_director(production_id: str, now: bool = False):
        production(production_id)
        return showrunner.director.pause(production_id, now=now)

    @app.post("/api/productions/{production_id}/director/resume")
    def resume_director(production_id: str, request: Resume):
        production(production_id)
        return showrunner.director.resume(production_id, request.message)

    @app.post("/api/productions/{production_id}/director/stop")
    def stop(production_id: str):
        production(production_id)
        showrunner.director.stop(production_id)
        return showrunner.director.status(production_id)

    @app.get("/api/productions/{production_id}/templates")
    def template_roles(production_id: str):
        production(production_id)
        return showrunner.template_roles(production_id)

    @app.post("/api/productions/{production_id}/templates")
    def choose_template(production_id: str, request: TemplatePick):
        production(production_id)
        return showrunner.choose_template(production_id, request.role, request.template or None)

    @app.post("/api/productions/{production_id}/text")
    async def add_text(production_id: str, request: Text):
        production(production_id)
        return await run_in_threadpool(showrunner.add_text, production_id, request.text, request.subtitle, request.style, request.start_s, request.duration_s, request.fade_s,
                                       design=request.design or None)

    @app.post("/api/productions/{production_id}/text/{overlay_id}")
    async def change_text(production_id: str, overlay_id: str, request: TextChange):
        production(production_id)
        return await run_in_threadpool(showrunner.update_text, production_id, overlay_id, request.fields)

    @app.post("/api/productions/{production_id}/shots/{shot_id}/card")
    async def card(production_id: str, shot_id: str, request: Text):
        production(production_id)
        return await run_in_threadpool(showrunner.add_card, production_id, shot_id, request.text, request.subtitle, request.style,
                                       design=request.design or None, picture=request.picture or None)

    @app.post("/api/productions/{production_id}/shots/{shot_id}/rerun")
    def rerun(production_id: str, shot_id: str):
        production(production_id)
        return showrunner.rerun(production_id, shot_id)

    @app.post("/api/productions/{production_id}/shots/{shot_id}/cancel")
    def cancel(production_id: str, shot_id: str):
        production(production_id)
        showrunner.cancel_rerun(production_id, shot_id)
        return {"shot": shot_id}

    @app.post("/api/productions/{production_id}/assemble")
    async def assemble(production_id: str):
        production(production_id)
        return await run_in_threadpool(showrunner.assemble, production_id)

    def subscribe(production):
        """The events of one production (or of all), queued for this event loop, and the call that stops them."""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue(maxsize=512)

        def listener(event):
            if production is None or event.get("production") in (None, production):
                loop.call_soon_threadsafe(lambda: queue.full() or queue.put_nowait(event))

        return queue, store.subscribe(listener)

    # The page reads its events on a WebSocket: an HTTP stream would hold one of the few connections a browser opens to a server,
    # which all WanGP tabs share. Scripts read the same events as a server-sent event stream.
    @app.websocket("/api/events")
    async def socket_events(socket: WebSocket, production: str | None = None):
        await socket.accept()
        queue, unsubscribe = subscribe(production)

        async def send_events():
            while True:
                event = await queue.get()
                await socket.send_text(json.dumps(event, default=str))

        async def disconnected():
            async for _ in socket.iter_text():
                pass

        sender, receiver = asyncio.create_task(send_events()), asyncio.create_task(disconnected())
        try:
            done, _ = await asyncio.wait([sender, receiver], return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                with suppress(WebSocketDisconnect, RuntimeError):  # RuntimeError: a send racing the browser's close
                    task.result()
        finally:
            unsubscribe()
            for task in (sender, receiver):
                task.cancel()
            await asyncio.gather(sender, receiver, return_exceptions=True)

    @app.get("/api/events")
    async def events(request: Request, production: str | None = None):
        queue, unsubscribe = subscribe(production)

        async def stream():
            try:
                yield "retry: 2000\n\n"
                while not await request.is_disconnected():
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=15)
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
                        continue
                    yield f"event: {event.get('type', 'message')}\ndata: {json.dumps(event, default=str)}\n\n"
            finally:
                unsubscribe()

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return app
