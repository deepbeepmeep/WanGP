# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import json
import mimetypes
import os
import queue as queue_module
import re
import threading
import time
import uuid
from pathlib import Path
from urllib.parse import quote, unquote


# Inline media need a type the browser displays or plays; the OS tables (Windows registry) can lack or misname these.
for _mime_type, _extension in (("image/webp", ".webp"), ("audio/aac", ".aac"), ("audio/flac", ".flac"), ("model/gltf-binary", ".glb")):
    mimetypes.add_type(_mime_type, _extension)

_download_jobs = {}
_file_downloads = {}
_file_download_tokens = {}
_gallery_downloads = {}
_lost_downloads = set()  # saved links whose file restore_file_downloads did not find
_download_jobs_lock = threading.Lock()
_download_routes_installed = False
_download_original_create_app = None


def _cleanup_download_jobs(ttl_seconds=600):
    now = time.time()
    with _download_jobs_lock:
        for token, job in list(_download_jobs.items()):
            if now - job.get("created", now) > ttl_seconds:
                _download_jobs.pop(token, None)


def _content_disposition(filename):
    fallback = re.sub(r'[^A-Za-z0-9._ -]+', '_', filename).strip() or "download"
    return f"attachment; filename=\"{fallback}\"; filename*=UTF-8''{quote(filename)}"


def register_download(filename, mime_type, iterator_factory):
    _cleanup_download_jobs()
    token = uuid.uuid4().hex
    with _download_jobs_lock:
        _download_jobs[token] = {
            "created": time.time(),
            "filename": filename,
            "mime_type": mime_type,
            "iterator_factory": iterator_factory,
        }
    return json.dumps({"url": f"/wangp_api/download/{token}", "filename": filename})


def register_file_download(path, mime_type=None):
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"Download file does not exist: {file_path}")
    key = os.path.normcase(str(file_path))
    with _download_jobs_lock:
        token = _file_download_tokens.get(key)
        if token is None:
            token = uuid.uuid5(uuid.NAMESPACE_URL, key).hex
            _file_download_tokens[key] = token
            _file_downloads[token] = {"path": str(file_path), "filename": file_path.name, "mime_type": mime_type or mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"}
    return {"url": f"/wangp_api/download/{token}", "filename": file_path.name, "size_bytes": file_path.stat().st_size}


def restore_file_downloads(text, roots):
    """Registers again the files under roots that the download links in text name: a link saved before a restart (a conversation's
    cards) works once more. A file's token comes from its path, so the same file gets its old link back. Returns how many were found."""
    with _download_jobs_lock:
        missing = set(re.findall(r"/wangp_api/download/([0-9a-f]{32})", text)) - _file_downloads.keys() - _lost_downloads
    found = 0
    for root in roots:
        root = Path(root).expanduser().resolve()
        for path in root.rglob("*") if missing and root.is_dir() else ():
            if uuid.uuid5(uuid.NAMESPACE_URL, os.path.normcase(str(path))).hex in missing and path.is_file():
                missing.discard(register_file_download(path)["url"].rsplit("/", 1)[1])
                found += 1
    with _download_jobs_lock:
        _lost_downloads.update(missing)  # gone or elsewhere: not searched for again
    return found


def register_gallery_download(media_id, path, mime_type=None):
    media_id = str(media_id or "").strip().casefold()
    if re.fullmatch(r"(?:visual|audio):[a-f0-9]{12}", media_id) is None:
        raise ValueError(f"Invalid Gallery media id: {media_id}")
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(f"Gallery file does not exist: {file_path}")
    with _download_jobs_lock:
        _gallery_downloads[media_id] = {"path": str(file_path), "mime_type": mime_type or mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"}
    return {"url": f"/wangp_api/gallery/media/{quote(media_id, safe='')}", "filename": file_path.name, "size_bytes": file_path.stat().st_size}


def stream_bytes(data, chunk_size=1024 * 1024):
    for offset in range(0, len(data), chunk_size):
        yield data[offset:offset + chunk_size]


def stream_writer(write_fn):
    chunks = queue_module.Queue(maxsize=8)
    stop_event = threading.Event()
    sentinel = object()

    def push(item):
        while not stop_event.is_set():
            try:
                chunks.put(item, timeout=0.5)
                return True
            except queue_module.Full:
                pass
        return False

    class StreamingWriter:
        def write(self, data):
            if not data:
                return 0
            data = bytes(data)
            if push(data):
                return len(data)
            raise BrokenPipeError("Download stream closed")

        def flush(self):
            pass

    def produce():
        try:
            if write_fn(StreamingWriter()) is False:
                raise RuntimeError("Failed to create download stream")
        except Exception as e:
            push(e)
        finally:
            push(sentinel)

    thread = threading.Thread(target=produce, daemon=True)
    thread.start()
    try:
        while True:
            item = chunks.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        stop_event.set()


def _pop_download_job(token):
    with _download_jobs_lock:
        return _download_jobs.pop(token, None)


def _get_file_download(token):
    with _download_jobs_lock:
        return _file_downloads.get(token)


def _get_gallery_download(media_id):
    with _download_jobs_lock:
        return _gallery_downloads.get(str(media_id or "").strip().casefold())


def registered_download_path(url):
    """Local file behind a registered download or Gallery media URL, None when the URL is not registered here."""
    url = unquote(str(url or ""))
    for prefix, lookup in (("/wangp_api/download/", _get_file_download), ("/wangp_api/gallery/media/", _get_gallery_download)):
        if url.startswith(prefix):
            entry = lookup(url[len(prefix):])
            return None if entry is None else entry["path"]
    return None


def _install_routes_on_app(fastapi_app):
    from shared.utils.http_disconnect import install_http_disconnect_patch
    install_http_disconnect_patch()

    if getattr(fastapi_app, "_wangp_download_routes_installed", False):
        return

    @fastapi_app.get("/wangp_api/download/{token}")
    async def _wangp_download(token: str, download: bool = False):
        from fastapi import Response
        from fastapi.responses import StreamingResponse
        from shared.utils.http_disconnect import DisconnectAwareFileResponse
        job = _pop_download_job(token)
        if job is not None:
            headers = {"Content-Disposition": _content_disposition(job["filename"])}
            return StreamingResponse(job["iterator_factory"](), media_type=job["mime_type"], headers=headers)
        file_download = _get_file_download(token)
        if file_download is None or not Path(file_download["path"]).is_file():
            return Response("Download expired or not found", status_code=404)
        return DisconnectAwareFileResponse(file_download["path"], filename=file_download["filename"], media_type=file_download["mime_type"], content_disposition_type="attachment" if download else "inline")

    @fastapi_app.get("/wangp_api/gallery/media/{media_id}")
    async def _wangp_gallery_media(media_id: str):
        from fastapi import Response
        from shared.utils.http_disconnect import DisconnectAwareFileResponse
        gallery_download = _get_gallery_download(media_id)
        if gallery_download is None or not Path(gallery_download["path"]).is_file():
            return Response("Gallery media expired or not found", status_code=404)
        return DisconnectAwareFileResponse(gallery_download["path"], media_type=gallery_download["mime_type"])

    @fastapi_app.get("/wangp_api/download/{token}/thumbnail")
    async def _wangp_thumbnail(token: str):
        from fastapi import Response
        from shared.deepy.thumbnails import render_thumbnail
        file_download = _get_file_download(token)
        if file_download is None or not Path(file_download['path']).is_file():
            return Response('Preview not found', status_code=404)
        data = await render_thumbnail(file_download['path'])
        if not data:
            return Response('Preview unavailable', status_code=404)
        return Response(data, media_type='image/jpeg', headers={'Cache-Control': 'private, max-age=60'})

    fastapi_app._wangp_download_routes_installed = True


def install_routes():
    global _download_routes_installed, _download_original_create_app
    if _download_routes_installed:
        return
    from gradio.routes import App
    _download_original_create_app = App.create_app

    def _patched_create_app(*args, **kwargs):
        fastapi_app = _download_original_create_app(*args, **kwargs)
        _install_routes_on_app(fastapi_app)
        return fastapi_app

    App.create_app = staticmethod(_patched_create_app)
    _download_routes_installed = True


__all__ = ["install_routes", "register_download", "register_file_download", "register_gallery_download", "registered_download_path", "restore_file_downloads", "stream_bytes", "stream_writer"]
