# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Public Plugin URLs and shared controls for their existing Gradio tabs."""
from html import escape
import json
import re
from urllib.parse import parse_qsl

from tqdm import tqdm

from shared.gradio.app_icons import icon
from shared.utils.plugins import get_plugin_navigation, get_plugin_web_apps
from shared.utils.bundled_apps import FEATURED_APPS, VISITED_KEY

_CONFIG_SCRIPT = re.compile(r'(window\.gradio_config\s*=\s*)(.*?)(;</script>)', re.DOTALL)


def _script_json(value):
    return json.dumps(value, ensure_ascii=True, separators=(",", ":")).replace("<", "\\u003c")


def _launch_page(body, navigation, fullscreen, base):
    """Keep the initial interface hidden until native tab activation completes."""
    page = body.decode("utf-8")
    page = page.replace('<head>', f'<head><base href="{escape(base, quote=True)}">', 1)
    match = _CONFIG_SCRIPT.search(page)
    if not match:
        return page.encode("utf-8")
    config = json.loads(match[2])
    components = config.get("components", [])
    tabs = next((c for c in components if "wangp-native-app-tabs" in (c.get("props", {}).get("elem_classes") or [])), None)
    panel = next((c for c in components if c["type"] == "tabitem" and c.get("props", {}).get("id") == navigation.tab_id), None)
    unavailable = not navigation.enabled or tabs is None or panel is None
    fullscreen = fullscreen and not unavailable
    intro = navigation.plugin_id in FEATURED_APPS and navigation.plugin_id not in (navigation.config or {}).get(VISITED_KEY, [])
    launch = dict(plugin=navigation.plugin_id, target=navigation.tab_id, fullscreen=fullscreen, intro=intro, base=base, unavailable=unavailable)
    bootstrap = '<style>.wangp-app-launch .gradio-container { visibility: hidden; }</style>'
    bootstrap += f'<script>window.WanGPAppLaunch={_script_json(launch)};' + ('document.documentElement.classList.add("wangp-app-launch");' if not unavailable else '') + '</script>'
    return page.replace('</head>', bootstrap + '</head>', 1).encode("utf-8")


def toolbar_html(navigation, label):
    controls = []
    if navigation.fullscreen:
        controls.append(f'<a href=".{escape(navigation.path, quote=True)}/" data-addon-fullscreen title="Open Full Screen">{icon("expand")}<span>Full Screen</span></a>')
        controls.append(f'<a href="./" data-addon-return title="Back to WanGP">{icon("back")}<span>WanGP</span></a>')
    return f'<div class="wangp-addon-toolbar" data-addon-target="{escape(navigation.tab_id, quote=True)}"><strong>{escape(label)}</strong><div>{"".join(controls)}</div></div>'


class AddonNavigationMiddleware:
    """Serve the selected app at its folder URL, keeping Gradio's APIs at the root."""
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["method"] in ("GET", "HEAD"):
            root = scope.get("root_path", "").rstrip("/")
            path = scope["path"].removeprefix(root).rstrip("/")
            query = dict(parse_qsl(scope.get("query_string", b"").decode("utf-8"), keep_blank_values=True))
            folder = next((entry for entry in get_plugin_navigation() if entry.path == path), None)
            navigation = folder or next((entry for entry in get_plugin_navigation() if not path and entry.plugin_id == query.get("app")), None)
            if navigation:
                embedded = query.get("embedded") == "1"
                web_page = any(entry.path == path and entry.plugin_id == navigation.plugin_id for entry in get_plugin_web_apps())
                if not (embedded and web_page):
                    fullscreen = navigation.fullscreen and (query.get("fullscreen", "1" if folder else "0") == "1")
                    base = root + "/"
                    rewritten = {**scope, "path": base, "raw_path": base.encode("utf-8"), "headers": [(k, v) for k, v in scope["headers"] if k.lower() != b"accept-encoding"]}
                    start, chunks = None, []
                    async def serve_page(message):
                        nonlocal start
                        if message["type"] == "http.response.start":
                            start = message
                        elif message["type"] == "http.response.body":
                            chunks.append(message.get("body", b""))
                            if not message.get("more_body", False):
                                body = b"".join(chunks)
                                headers = dict(start["headers"])
                                response_headers = start["headers"]
                                if start["status"] == 200 and b"text/html" in headers.get(b"content-type", b"") and body:
                                    body = _launch_page(body, navigation, fullscreen, base)
                                    response_headers = [(k, v) for k, v in start["headers"] if k.lower() not in (b"content-length", b"cache-control", b"etag")]
                                    response_headers += [(b"content-length", str(len(body)).encode("ascii")), (b"cache-control", b"no-store")]
                                await send({**start, "headers": response_headers})
                                await send({"type": "http.response.body", "body": body})
                        else:
                            await send(message)
                    return await self.app(rewritten, receive, serve_page)
        return await self.app(scope, receive, send)


def mount_web_apps(app, base_url):
    """Called once by the shared server after the Plugin manager has loaded apps."""
    import traceback
    public_apps = {}
    for web_app in get_plugin_web_apps():
        try:
            app.mount(web_app.path, web_app.factory())
            public_apps[web_app.plugin_id] = web_app
        except Exception:
            print(f"[Plugins] Could not start the web app of Plugin '{web_app.plugin_id}':")
            traceback.print_exc()
    public_apps.update({entry.plugin_id: entry for entry in get_plugin_navigation() if entry.enabled})
    for entry in public_apps.values():
        tqdm.write(f"{entry.name} App: {base_url.rstrip('/')}{entry.path}/")
