# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""The /apps routes: web apps and document previews served isolated from WanGP.

Server code registers what may be opened (a development bundle, a .wgpapp package, a page or a document) and gets an id; the WanGP page,
signed in, exchanges that id for a launch token (POST /apps/api/launch) and frames /apps/r/<token>/<entry>. Every document served there
carries a CSP `sandbox` directive, so the browser gives it an opaque origin even when it is opened in a tab of its own: no WanGP cookie or
storage, no access to the page around it, no top navigation. Its fetch directives name only its own token path and the vendored libraries
(/apps/lib/, with the import map injected into each app page), so nothing reaches WanGP's other routes or the internet. The token path is the
only credential of these routes. AppFrameGuard refuses what an opaque frame sends to every other route, as a second layer."""
from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import re
import secrets
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

from shared.apps import bundles

STORE_BYTES = 5 << 20  # an app's saved data (high scores, saves, settings)
LOG_ENTRIES = 300
LAUNCH_IDLE_S = 6 * 3600
MAX_LAUNCHES = 200
DOCUMENT_BYTES = 4 << 20
DOCUMENT_SUFFIXES = {".md", ".markdown", ".txt", ".csv", ".tsv", ".json", ".svg"}  # previews without scripts
PAGE_SUFFIXES = {".html", ".htm"}
_MIME = {".html": "text/html; charset=utf-8", ".htm": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8", ".mjs": "text/javascript; charset=utf-8", ".css": "text/css; charset=utf-8",
         ".json": "application/json", ".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp", ".gif": "image/gif",
         ".avif": "image/avif", ".ico": "image/x-icon", ".glb": "model/gltf-binary", ".gltf": "model/gltf+json", ".bin": "application/octet-stream", ".obj": "text/plain; charset=utf-8",
         ".mtl": "text/plain; charset=utf-8", ".hdr": "application/octet-stream", ".ktx2": "image/ktx2", ".wasm": "application/wasm", ".mp3": "audio/mpeg", ".ogg": "audio/ogg",
         ".wav": "audio/wav", ".m4a": "audio/mp4", ".flac": "audio/flac", ".mp4": "video/mp4", ".webm": "video/webm", ".ttf": "font/ttf", ".otf": "font/otf", ".woff": "font/woff",
         ".woff2": "font/woff2", ".txt": "text/plain; charset=utf-8", ".csv": "text/plain; charset=utf-8", ".md": "text/plain; charset=utf-8"}
_SANDBOX = "sandbox allow-scripts allow-forms allow-modals allow-downloads allow-pointer-lock"
_HEAD = re.compile(r"<head\b[^>]*>", re.IGNORECASE)
_DOCTYPE = re.compile(r"^\s*<!doctype[^>]*>", re.IGNORECASE)
_CLASSIC_SCRIPT = re.compile(r"<script(?![^>]*\bcrossorigin\b)(?![^>]*\btype\s*=\s*['\"]?module)([^>]*\bsrc\s*=)", re.IGNORECASE)
_API = "__wangp__"


@dataclass
class AppEntry:
    kind: str  # "dev" (a bundle in a Deepy sandbox), "package" (a published .wgpapp), "page" (one HTML file), "document" (a preview without scripts)
    folder: Path
    entry: str
    name: str
    store: Path | None = None
    source: Path | None = None  # the .wgpapp of a package


@dataclass
class Launch:
    app_id: str
    seen: float = field(default_factory=time.monotonic)


_lock = threading.Lock()
_apps: dict[str, AppEntry] = {}
_launches: OrderedDict[str, Launch] = OrderedDict()
_logs: dict[str, deque] = {}
_stats: dict[str, dict[str, Any]] = {}


def _app_id(kind: str, path: Path) -> str:
    return kind[0] + hashlib.sha1(str(Path(path).resolve()).lower().encode("utf-8")).hexdigest()[:15]


def register_dev(folder: Path) -> str:
    """A development bundle <sandbox>/apps/<slug>: its saved data stays in the sandbox's tmp/ (scratch, never published or snapshotted)."""
    folder = Path(folder).resolve()
    manifest = bundles.read_manifest(folder)
    app_id = _app_id("dev", folder)
    store = folder.parents[1] / "tmp" / "app_data" / f"{manifest.get('id') or folder.name}.json"
    with _lock:
        _apps[app_id] = AppEntry("dev", folder, manifest["entry"], manifest["name"], store)
    return app_id


def register_package(package: Path) -> str:
    """A published .wgpapp: extracted once into the cache; its saved data is kept per app id, so new versions keep it."""
    package = Path(package).resolve()
    folder = bundles.unpack(package)
    manifest = bundles.read_manifest(folder)
    app_id = _app_id("package", package)
    store = bundles.DATA_DIR / "data" / re.sub(r"[^A-Za-z0-9_-]", "_", str(manifest.get("id") or folder.name)) / "store.json"
    with _lock:
        _apps[app_id] = AppEntry("package", folder, manifest["entry"], manifest["name"], store, package)
    return app_id


def register_file(path: Path) -> str:
    """A file to preview: an HTML page (scripts, served like an app from its folder) or a document rendered without scripts."""
    path = Path(path).resolve()
    suffix = path.suffix.lower()
    if suffix not in DOCUMENT_SUFFIXES | PAGE_SUFFIXES:
        raise ValueError(f"No preview for {suffix} files.")
    kind = "page" if suffix in PAGE_SUFFIXES else "document"
    app_id = _app_id(kind, path)
    with _lock:
        _apps[app_id] = AppEntry(kind, path.parent, path.name, path.name)
    return app_id


def app_entry(app_id: str) -> AppEntry:
    with _lock:
        entry = _apps.get(app_id)
    if entry is None:
        raise KeyError("This app is not available any more (WanGP restarted or the file moved): open it again from the chat or the Apps gallery.")
    return entry


def launch(app_id: str) -> dict[str, Any]:
    """A launch token for a registered app; apps and pages need App Development, documents are always previewable."""
    from shared.deepy.config import python_apps_level

    entry = app_entry(app_id)
    if entry.kind != "document" and python_apps_level() == "off":
        raise PermissionError("Web apps are disabled: Configuration > Deepy > Python > App Development (Advanced).")
    token = secrets.token_urlsafe(18)
    with _lock:
        _launches[token] = Launch(app_id)
        now = time.monotonic()
        for stale in [key for key, item in _launches.items() if now - item.seen > LAUNCH_IDLE_S]:
            del _launches[stale]
        while len(_launches) > MAX_LAUNCHES:
            _launches.popitem(last=False)
    manifest = {} if entry.kind in ("document", "page") else bundles.read_manifest(entry.folder)
    url = f"/apps/r/{token}/" + ("" if entry.kind == "document" else entry.entry)
    return {"token": token, "url": url, "kind": entry.kind, "name": manifest.get("name") or entry.name, "controls": manifest.get("controls", {}), "canvas": manifest.get("canvas", "responsive")}


def close(token: str) -> None:
    with _lock:
        _launches.pop(token, None)


def _launched(token: str) -> tuple[str, AppEntry]:
    with _lock:
        launch_record = _launches.get(token)
        if launch_record is not None:
            launch_record.seen = time.monotonic()
            _launches.move_to_end(token)
    if launch_record is None:
        raise KeyError("This app session ended: open the app again.")
    return launch_record.app_id, app_entry(launch_record.app_id)


# ---------------------------------------------------------------------------------------------------- logs and saved data
def add_logs(app_id: str, entries: list[dict[str, Any]], stats: dict[str, Any] | None = None) -> None:
    with _lock:
        log = _logs.setdefault(app_id, deque(maxlen=LOG_ENTRIES))
        for item in entries[:100]:
            log.append({"t": round(time.time(), 3), "level": str(item.get("level") or "log")[:12], "text": str(item.get("text") or "")[:2000]})
        if stats:
            _stats[app_id] = {**_stats.get(app_id, {}), **{key: stats[key] for key in ("fps", "renderer", "size") if key in stats}, "updated": round(time.time(), 3)}


def app_logs(app_id: str, since: float = 0.0) -> dict[str, Any]:
    """What the app reported from the browsers it ran in since `since` (epoch seconds): console lines, errors, frame rate and renderer."""
    with _lock:
        entries = [item for item in _logs.get(app_id, ()) if item["t"] > since]
        stats = dict(_stats.get(app_id, {}))
    return {"entries": entries, "errors": sum(1 for item in entries if item["level"] in ("error", "exception")), "stats": stats}


def read_store(entry: AppEntry) -> dict[str, str]:
    if entry.store is None or not entry.store.is_file():
        return {}
    try:
        data = json.loads(entry.store.read_text(encoding="utf-8"))
    except ValueError:
        return {}
    return {str(key): str(value) for key, value in data.items()} if isinstance(data, dict) else {}


def update_store(entry: AppEntry, change: dict[str, Any]) -> dict[str, str]:
    if entry.store is None:
        raise PermissionError("This preview has no saved data.")
    with _lock:
        data = {} if change.get("clear") else read_store(entry)
        for key in change.get("remove") or []:
            data.pop(str(key), None)
        for key, value in (change.get("set") or {}).items():
            data[str(key)] = str(value)
        encoded = json.dumps(data, ensure_ascii=False)
        if len(encoded.encode("utf-8")) > STORE_BYTES:
            raise ValueError(f"The app's saved data would exceed {STORE_BYTES >> 20} MB.")
        entry.store.parent.mkdir(parents=True, exist_ok=True)
        entry.store.write_text(encoded, encoding="utf-8")
    return data


# ---------------------------------------------------------------------------------------------------- documents served to the frame
def csp(netloc: str, token: str, scripts: bool) -> str:
    own = f"{netloc}/apps/r/{token}/"
    if not scripts:
        return f"sandbox; default-src 'none'; style-src 'unsafe-inline'; img-src {own} data:; base-uri 'none'; form-action 'none'"
    lib = f"{netloc}/apps/lib/"
    return (f"{_SANDBOX}; default-src 'none'; script-src {own} {lib} 'unsafe-inline' 'unsafe-eval' blob:; style-src {own} {lib} 'unsafe-inline'; "
            f"img-src {own} {lib} data: blob:; media-src {own} data: blob:; font-src {own} {lib} data:; worker-src {own} {lib} blob:; connect-src {own} {lib} data: blob:; "
            "base-uri 'none'; form-action 'none'")


def frame_headers(netloc: str, token: str, scripts: bool = True) -> dict[str, str]:
    # Every response is CORS-readable: the app's own document has an opaque origin, so its module scripts and fetches are cross-origin requests.
    return {"Content-Security-Policy": csp(netloc, token, scripts), "Access-Control-Allow-Origin": "*", "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer", "Cache-Control": "no-store"}


def inject_runtime(page: str, entry: AppEntry) -> str:
    """The import map, the saved data and the runtime script, ahead of any module script of the page."""
    manifest = {} if entry.kind == "page" else bundles.read_manifest(entry.folder)
    config = {"name": manifest.get("name") or entry.name, "store": read_store(entry), "persist": entry.store is not None, "controls": manifest.get("controls", {})}
    encoded = lambda value: json.dumps(value, ensure_ascii=False).replace("</", "<\\/")
    block = f'<script type="importmap">{encoded(bundles.import_map())}</script><script>window.__wangpApp={encoded(config)}</script><script src="/apps/lib/wangp-app.js?v=2" crossorigin></script>'
    page = _CLASSIC_SCRIPT.sub(r"<script crossorigin\1", page)  # the page's own origin is opaque: without CORS loading, errors of classic scripts read only "Script error."
    head = _HEAD.search(page)
    if head is not None:
        return page[:head.end()] + block + page[head.end():]
    doctype = _DOCTYPE.match(page)
    return page[:doctype.end()] + block + page[doctype.end():] if doctype else block + page


def render_document(path: Path) -> str:
    """A self-contained HTML rendering of a Markdown, text, CSV/TSV, JSON or SVG file, for a frame without scripts."""
    suffix = path.suffix.lower()
    if path.stat().st_size > DOCUMENT_BYTES:
        body = f"<p>{html.escape(path.name)} is {path.stat().st_size / 1048576:.1f} MB: too large to preview. Download it from the chat instead.</p>"
    elif suffix == ".svg":
        body = f'<img class="svg" src="{html.escape(path.name)}" alt="{html.escape(path.name)}">'
    else:
        text = path.read_text(encoding="utf-8", errors="replace")
        if suffix in (".md", ".markdown"):
            import markdown
            body = f'<article>{markdown.markdown(text, extensions=["tables", "fenced_code", "sane_lists"], output_format="html5")}</article>'
        elif suffix in (".csv", ".tsv"):
            rows = list(csv.reader(io.StringIO(text), delimiter="\t" if suffix == ".tsv" else (csv.Sniffer().sniff(text[:4096], ",;\t|").delimiter if text.strip() else ",")))
            shown = rows[:2001]
            head, rest = (shown[0], shown[1:]) if shown else ([], [])
            table = "<table><thead><tr>" + "".join(f"<th>{html.escape(cell)}</th>" for cell in head) + "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>" for row in rest) + "</tbody></table>"
            body = table + (f"<p class='note'>{len(rows) - len(shown)} more rows not shown.</p>" if len(rows) > len(shown) else "")
        elif suffix == ".json":
            try:
                text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
            except ValueError:
                pass
            body = f"<pre>{html.escape(text)}</pre>"
        else:
            body = f"<pre>{html.escape(text)}</pre>"
    style = ("body{margin:0;padding:20px 24px;font:15px/1.55 system-ui,sans-serif;color:#1d2733;background:#fbfcfd}article{max-width:880px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.45 ui-monospace,Consolas,monospace}"
             "table{border-collapse:collapse;font-size:13px}th,td{border:1px solid #cfd8e3;padding:4px 8px;text-align:left;vertical-align:top}th{background:#eef3f8;position:sticky;top:0}"
             "img.svg{max-width:100%;height:auto;display:block;margin:auto}.note{color:#5a6b7d}code{font-family:ui-monospace,Consolas,monospace}"
             "@media (prefers-color-scheme: dark){body{color:#e4ebf2;background:#14191f}th{background:#1f2933}th,td{border-color:#334150}.note{color:#9aabbd}}")
    return f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(path.name)}</title><style>{style}</style></head><body>{body}</body></html>'


def bundle_file(folder: Path, relative: str) -> Path:
    """A file of folder named by a request path: never outside it, never a hidden file."""
    path = (folder / relative).resolve()
    if not path.is_relative_to(folder.resolve()) or any(part.startswith(".") for part in Path(relative).parts) or not path.is_file():
        raise FileNotFoundError(relative)
    return path


def lib_file(relative: str) -> Path:
    if relative == "wangp-app.js":
        return bundles.RUNTIME_SCRIPT
    return bundle_file(bundles.LIB_DIR, relative)


def media_type(path: Path) -> str:
    return _MIME.get(path.suffix.lower(), "application/octet-stream")


# ---------------------------------------------------------------------------------------------------- routes
def create_app(public_url: str | None = None):
    """The ASGI app mounted at /apps."""
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    def netloc(request: Request) -> str:
        return urlsplit(public_url).netloc if public_url else request.url.netloc

    @app.post("/api/launch")
    async def launch_route(request: Request):  # reached only through WanGP's own (signed-in, same-origin) page
        payload = await request.json()
        try:
            return launch(str(payload.get("app_id") or ""))
        except KeyError as error:
            raise HTTPException(404, str(error.args[0])) from None
        except PermissionError as error:
            raise HTTPException(403, str(error)) from None

    @app.post("/api/close")
    async def close_route(request: Request):
        close(str((await request.json()).get("token") or ""))
        return {"status": "done"}

    @app.get("/lib/{relative:path}")
    def lib_route(relative: str):
        try:
            path = lib_file(relative)
        except FileNotFoundError:
            raise HTTPException(404, "Not a vendored library file.") from None
        return FileResponse(path, media_type=media_type(path), headers={"Access-Control-Allow-Origin": "*", "Cache-Control": "public, max-age=3600", "X-Content-Type-Options": "nosniff"})

    @app.options("/r/{token}/{relative:path}")
    def preflight(token: str, relative: str):
        return Response(status_code=204, headers={"Access-Control-Allow-Origin": "*", "Access-Control-Allow-Methods": "GET, POST", "Access-Control-Allow-Headers": "Content-Type", "Access-Control-Max-Age": "600"})

    @app.get(f"/r/{{token}}/{_API}/store")
    def store_read(token: str, request: Request):
        try:
            _app_id, entry = _launched(token)
        except KeyError as error:
            raise HTTPException(404, str(error.args[0])) from None
        return JSONResponse(read_store(entry), headers=frame_headers(netloc(request), token))

    @app.post(f"/r/{{token}}/{_API}/store")
    async def store_write(token: str, request: Request):
        try:
            _app_id, entry = _launched(token)
            data = update_store(entry, await request.json())
        except KeyError as error:
            raise HTTPException(404, str(error.args[0])) from None
        except (PermissionError, ValueError) as error:
            return JSONResponse({"error": str(error)}, status_code=400, headers=frame_headers(netloc(request), token))
        return JSONResponse({"keys": len(data)}, headers=frame_headers(netloc(request), token))

    @app.post(f"/r/{{token}}/{_API}/log")
    async def log_write(token: str, request: Request):
        try:
            app_id, _entry = _launched(token)
        except KeyError as error:
            raise HTTPException(404, str(error.args[0])) from None
        payload = await request.json()
        add_logs(app_id, [item for item in payload.get("entries") or [] if isinstance(item, dict)], payload.get("stats") if isinstance(payload.get("stats"), dict) else None)
        return Response(status_code=204, headers=frame_headers(netloc(request), token))

    @app.get("/r/{token}/{relative:path}")
    def frame_route(token: str, relative: str, request: Request):
        try:
            _app_id, entry = _launched(token)
        except KeyError as error:
            raise HTTPException(404, str(error.args[0])) from None
        if entry.kind == "document":
            target = entry.folder / entry.entry
            if relative == "":
                return HTMLResponse(render_document(target), headers=frame_headers(netloc(request), token, scripts=False))
            if relative == entry.entry and target.suffix.lower() == ".svg":
                return FileResponse(target, media_type="image/svg+xml", headers=frame_headers(netloc(request), token, scripts=False))
            raise HTTPException(404, "Not part of this document.")
        try:
            path = bundle_file(entry.folder, relative or entry.entry)
        except FileNotFoundError:
            raise HTTPException(404, f"{relative} is not in the app.") from None
        headers = frame_headers(netloc(request), token)
        if path.suffix.lower() in PAGE_SUFFIXES:
            return HTMLResponse(inject_runtime(path.read_text(encoding="utf-8", errors="replace"), entry), headers=headers)
        return FileResponse(path, media_type=media_type(path), headers=headers)

    return app


def mount(app, public_url: str | None = None) -> None:
    app.mount("/apps", create_app(public_url))


class AppFrameGuard:
    """Refuses, on every route outside /apps/r/ and /apps/lib/, what an opaque-origin frame sends: requests with `Origin: null` (its fetches,
    form posts and WebSockets) and cross-site no-cors loads or frame navigations, except GET/HEAD navigations to the UI entry pages.
    Top-level navigations from other sites and non-browser clients (MCP, the CLI, scripts), which send no such headers, pass."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] in ("http", "websocket") and not scope["path"].startswith(("/apps/r/", "/apps/lib/")):
            headers = dict(scope.get("headers") or ())
            cross_site = headers.get(b"sec-fetch-site") == b"cross-site"
            ui_navigation = (scope["type"] == "http" and scope["method"] in ("GET", "HEAD") and scope["path"] in ("/", "/deepy", "/deepy/")
                             and headers.get(b"sec-fetch-mode") == b"navigate" and headers.get(b"sec-fetch-dest") in (b"iframe", b"frame"))
            blocked_frame = headers.get(b"sec-fetch-dest") in (b"iframe", b"frame", b"embed", b"object") and not ui_navigation
            if headers.get(b"origin") == b"null" or cross_site and (headers.get(b"sec-fetch-mode") == b"no-cors" or blocked_frame):
                if scope["type"] == "websocket":
                    return await send({"type": "websocket.close", "code": 1008})
                from starlette.responses import PlainTextResponse
                return await PlainTextResponse("Refused: isolated app frames and other sites cannot use WanGP's routes.", status_code=403)(scope, receive, send)
        return await self.app(scope, receive, send)
