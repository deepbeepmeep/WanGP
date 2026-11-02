# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Native Gradio workspace viewer routes; not mounted by standalone Deepy."""
from typing import Literal
from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import File, HTTPException, Query, Response, UploadFile
from pydantic import BaseModel, Field, StrictBool, StrictInt
from starlette.concurrency import run_in_threadpool

from shared.deepy.gallery import _AUDIO_EXTENSIONS, _IMAGE_EXTENSIONS, _VIDEO_EXTENSIONS
from shared.deepy.voice import save_upload
from shared.utils.gallery_sources import MESH_IMPORT_EXTENSIONS, SOURCES
from shared.utils.media_imports import persist_gallery_import

GallerySourceId = Literal[tuple(SOURCES)]


class ViewerAction(BaseModel):
    workspace: str
    source: GallerySourceId
    revision: str
    keys: list[str] = Field(min_length=1)
    before: str | None = None
    edge: Literal['start'] | None = None
    target: str | None = None
    name: str = ''
    confirmed: bool = False


class ViewerPage(BaseModel):
    workspace: str
    source: GallerySourceId = 'video'
    page: int = Field(default=0, ge=0)
    selected: list[str] = Field(default_factory=list)
    initial: bool = False


class ViewerRetention(BaseModel):
    days: StrictInt


class ViewerProtection(BaseModel):
    workspace: str
    protected: StrictBool


class ViewerMigration(BaseModel):
    workspace: str
    mode: Literal['shared', 'dedicated']
    folder: str | None = None
    confirmed: StrictBool = False


def mount_workspace_viewer(app, service):
    viewer = service.workspace_viewer

    @app.post('/deepy_api/workspace_viewer/storage')
    def storage(payload: ViewerMigration):
        return viewer.migrate(**payload.model_dump())

    @app.get('/deepy_api/workspace_viewer')
    def page(workspace: str, source: GallerySourceId = 'video', page: int = Query(0, ge=0)):
        return viewer.page(workspace, source, page)

    @app.post('/deepy_api/workspace_viewer')
    def selected_page(payload: ViewerPage):
        return viewer.page(**payload.model_dump())

    @app.get('/deepy_api/workspace_viewer/retention')
    def retention():
        return viewer.retention()

    @app.post('/deepy_api/workspace_viewer/retention')
    def save_retention(payload: ViewerRetention):
        return viewer.retention(payload.days)

    @app.post('/deepy_api/workspace_viewer/protection')
    def protection(payload: ViewerProtection):
        return viewer.protect(payload.workspace, payload.protected)

    @app.get('/deepy_api/workspace_viewer/info')
    def info(workspace: str, source: GallerySourceId, key: str):
        return viewer.info(workspace, source, key)

    @app.get('/deepy_api/workspace_viewer/player')
    def player(workspace: str, source: GallerySourceId, key: str):
        return viewer.player(workspace, source, key)

    @app.get('/deepy_api/workspace_viewer/thumbnail')
    async def thumbnail(workspace: str, source: GallerySourceId, key: str):
        from shared.deepy.thumbnails import render_thumbnail
        path = await run_in_threadpool(viewer.thumbnail_path, workspace, source, key)
        if service.gallery._detect_media_type(path) not in ('image', 'video', '3d', 'world', 'app'):
            raise HTTPException(400, 'This media has no image preview.')
        try:
            image = await render_thumbnail(path)
        except FileNotFoundError:
            raise HTTPException(404, 'Media file not found.')
        if not image:
            raise HTTPException(404, 'Preview unavailable.')
        # The RAM cache also validates companions replaced without changing the asset.
        return Response(image, media_type='image/jpeg', headers={'Cache-Control': 'private, no-store'})

    @app.post('/deepy_api/workspace_viewer/import')
    async def upload(workspace: str, file: UploadFile = File(...)):
        # The target is captured before uploading: never import into a workspace
        # selected by another browser while this upload was in flight.
        output = Path(service.workspace_output_config(workspace)['save_path'])
        output.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix='.import-', dir=output) as temporary:
            source = await save_upload(file, Path(temporary), limit=1024 * 1024 * 1024, extensions=_IMAGE_EXTENSIONS | _VIDEO_EXTENSIONS | _AUDIO_EXTENSIONS | set(MESH_IMPORT_EXTENSIONS), preserve_filename=True)
            def add():
                with service._mutation_lock:
                    if workspace != service.workspace_id:
                        raise ValueError('The active workspace changed during upload. Please try again.')
                    if source.suffix.lower() in MESH_IMPORT_EXTENSIONS:
                        from shared.utils.mesh_outputs import import_mesh_file
                        path, reused = import_mesh_file(str(source), service.workspace_output_config(workspace)['mesh_save_path'])
                    else:
                        path, reused = persist_gallery_import(source, output, move=True)
                    try:
                        media_id = service.import_media(path, context='gallery')
                    except Exception:
                        if not reused:
                            Path(path).unlink(missing_ok=True)
                        raise
                    return {'id': media_id, 'reused': reused, 'filename': Path(path).name}
            return await run_in_threadpool(add)

    @app.post('/deepy_api/workspace_viewer/{action}')
    def action(action: Literal['reorder', 'eject', 'delete', 'copy', 'move', 'archive'], payload: ViewerAction):
        return viewer.action(action, payload.model_dump())
