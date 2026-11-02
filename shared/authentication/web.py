# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Password-only login, process-wide throttling and shared browser sessions."""
import hashlib
import html
import math
import secrets
import threading
import time
from urllib.parse import urlencode, urlsplit

from starlette.concurrency import run_in_threadpool
from starlette.requests import HTTPConnection, Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse

COOKIE = "wangp_session"
SESSION_SECONDS = 24 * 60 * 60
PAGE_HEADERS = {"Cache-Control": "no-store", "Referrer-Policy": "no-referrer", "X-Frame-Options": "DENY", "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'"}


class PasswordGate:
    def __init__(self, password):
        self.salt = secrets.token_bytes(16)
        self.digest = self._hash(password)
        self.lock = threading.Lock()
        self.failures = 0
        self.next_attempt = 0

    def _hash(self, value):
        return hashlib.scrypt(value.encode("utf-8"), salt=self.salt, n=16384, r=8, p=1)

    def check(self, value):
        if not self.lock.acquire(blocking=False):
            return False, 1
        try:
            remaining = math.ceil(self.next_attempt - time.monotonic())
            if remaining > 0:
                return False, remaining
            valid = isinstance(value, str) and len(value.encode("utf-8")) <= 1024 and secrets.compare_digest(self._hash(value), self.digest)
            if valid:
                self.failures = 0
                self.next_attempt = 0
                return True, 0
            self.failures += 1
            delay = 600 if self.failures >= 20 else max(0, self.failures - 4) * 30
            self.next_attempt = time.monotonic() + delay
            return False, delay
        finally:
            self.lock.release()


def _browser_origin(connection, public_url=None):
    if public_url is not None:
        return public_url
    scheme = {"ws": "http", "wss": "https"}.get(connection.url.scheme, connection.url.scheme)
    host = connection.url.netloc
    origin = connection.headers.get("origin")
    # A proxy can terminate or add TLS without changing the public Host. Accept
    # either web scheme only for that exact authority, including an explicit port.
    # Use the accepted browser scheme for login cookies, not the proxy's scheme.
    if scheme in {"http", "https"} and origin in {f"http://{host}", f"https://{host}"}:
        return origin
    return f"{scheme}://{host}"


def same_origin(connection, public_url=None):
    origin = connection.headers.get("origin")
    return origin is None or origin == _browser_origin(connection, public_url)


def login_page(title, description, action, fields, *, error="", status=200, retry=0, button="Sign In"):
    hidden = "".join(f'<input type="hidden" name="{html.escape(k, quote=True)}" value="{html.escape(v, quote=True)}">' for k, v in fields.items())
    page = f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title>
<style>body{{font:16px system-ui;background:#15171c;color:#f1f3f8;margin:0;display:grid;min-height:100vh;place-items:center}}main{{width:min(420px,85vw);padding:32px;background:#222630;border-radius:16px}}h1{{font-size:24px}}p{{line-height:1.5;overflow-wrap:anywhere}}input,button{{box-sizing:border-box;width:100%;padding:12px;margin-top:12px;border-radius:8px;border:1px solid #778}}button{{background:#a7c7ff;color:#101828;cursor:pointer}}.error{{color:#ffb7b7}}</style></head>
<body><main><h1>{html.escape(title)}</h1><p>{html.escape(description)}</p><form method="post" action="{html.escape(action, quote=True)}">{hidden}<label>Passphrase<input name="password" type="password" required autocomplete="current-password" maxlength="1024" autofocus></label><button>{html.escape(button)}</button><p class="error" role="alert">{html.escape(error)}</p></form></main></body></html>'''
    headers = {**PAGE_HEADERS, "Referrer-Policy": "same-origin"}
    if retry:
        headers["Retry-After"] = str(retry)
    return HTMLResponse(page, status_code=status, headers=headers)


async def small_form(request):
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > 16384:
            raise ValueError("Login request is too large.")
    from urllib.parse import parse_qs
    return {k: v[-1] for k, v in parse_qs(data.decode("utf-8"), keep_blank_values=True).items()}


class WebAuthentication:
    def __init__(self, password, public_url=None):
        from shared.authentication import parse_public_url
        self.public_url = parse_public_url(public_url) if public_url is not None else None
        self.gate = PasswordGate(password) if password is not None else None
        self.sessions = {}
        self.lock = threading.Lock()
        self.csrf_secret = secrets.token_urlsafe(32)

    def authenticated(self, connection):
        if self.gate is None:
            return True
        with self.lock:
            return self.sessions.get(connection.cookies.get(COOKIE, ""), 0) > time.monotonic()

    async def login(self, request):
        target = request.query_params.get("next", "/")
        parsed = urlsplit(target)
        if not target.startswith("/") or target.startswith("//") or "\\" in target or parsed.netloc or parsed.scheme or any(ord(c) < 32 for c in target):
            target = "/"
        action = "/auth/login?" + urlencode({"next": target})
        error, status, retry = "", 200, 0
        if request.method == "POST":
            try:
                form = await small_form(request)
            except (ValueError, UnicodeError):
                return JSONResponse({"detail": "Invalid login request."}, status_code=400)
            if not same_origin(request, self.public_url) or not secrets.compare_digest(form.get("csrf", "").encode("utf-8"), self.csrf_secret.encode("utf-8")):
                return JSONResponse({"detail": "Cross-origin request rejected."}, status_code=403)
            valid, retry = (True, 0) if self.gate is None else await run_in_threadpool(self.gate.check, form.get("password", ""))
            if valid:
                response = RedirectResponse(target, status_code=303, headers={"Cache-Control": "no-store"})
                session = secrets.token_urlsafe(32)
                with self.lock:
                    now = time.monotonic()
                    self.sessions = {key: expiry for key, expiry in self.sessions.items() if expiry > now}
                    if len(self.sessions) >= 4096:
                        return JSONResponse({"detail": "Too many active sessions."}, status_code=503)
                    self.sessions[session] = now + SESSION_SECONDS
                response.set_cookie(COOKIE, session, httponly=True, secure=_browser_origin(request, self.public_url).startswith("https://"), samesite="strict", path="/", max_age=SESSION_SECONDS)
                return response
            status = 429 if retry else 401
            error = f"Try again in {retry} seconds." if retry else "Incorrect passphrase."
        return login_page("Sign In to WanGP", "One login gives access to Gradio and Deepy.", action, {"csrf": self.csrf_secret}, error=error, status=status, retry=retry)


class WebAuthMiddleware:
    def __init__(self, app, auth):
        self.app, self.auth = app, auth

    async def __call__(self, scope, receive, send):
        if scope["type"] not in {"http", "websocket"}:
            return await self.app(scope, receive, send)
        connection = HTTPConnection(scope)
        path = scope["path"]
        if path.startswith(("/apps/r/", "/apps/lib/")):  # isolated app frames carry no cookie: their launch token path is the credential (shared.apps.server)
            return await self.app(scope, receive, send)
        if scope["type"] == "http" and path == "/auth/login" and scope["method"] in {"GET", "POST"}:
            return await (await self.auth.login(Request(scope, receive)))(scope, receive, send)
        unsafe = scope["type"] == "websocket" or scope["method"] not in {"GET", "HEAD", "OPTIONS"}
        if not self.auth.authenticated(connection) or (unsafe and not same_origin(connection, self.auth.public_url)):
            if scope["type"] == "websocket":
                return await send({"type": "websocket.close", "code": 1008})
            if unsafe and not same_origin(connection, self.auth.public_url):
                response = JSONResponse({"detail": "Cross-origin request rejected."}, status_code=403)
            elif scope["method"] in {"GET", "HEAD"} and "text/html" in connection.headers.get("accept", ""):
                target = path + ("?" + scope["query_string"].decode("latin1") if scope["query_string"] else "")
                response = RedirectResponse("/auth/login?" + urlencode({"next": target}), status_code=303)
            else:
                response = JSONResponse({"detail": "Sign in to WanGP."}, status_code=401)
            return await response(scope, receive, send)
        if path == "/auth/logout" and scope["type"] == "http" and scope["method"] == "POST":
            with self.auth.lock:
                self.auth.sessions.pop(connection.cookies.get(COOKIE, ""), None)
            response = RedirectResponse("/auth/login", status_code=303)
            response.delete_cookie(COOKIE, path="/")
            return await response(scope, receive, send)
        return await self.app(scope, receive, send)


class GradioStartupMiddleware:
    """Acknowledge Gradio's launch probe; queue startup runs in the lifespan.

    This public response performs no work and exposes no application state.
    The actual startup endpoint remains inaccessible through the auth gate.
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] == "GET" and scope["path"] == "/gradio_api/startup-events":
            return await JSONResponse(True)(scope, receive, send)
        return await self.app(scope, receive, send)
