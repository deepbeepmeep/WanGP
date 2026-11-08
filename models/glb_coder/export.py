# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Run the model's HTML in a disposable offline Chromium sandbox and collect its binary glTF export."""
import asyncio
import base64
import re
import struct
import sys
from pathlib import Path
from urllib.parse import urlsplit

ORIGIN = "https://glb-coder.invalid"
THREE = Path(__file__).resolve().parents[2] / "shared/apps/lib/three"
MAX_GLB_BYTES = 32 << 20
THREE_EXPORTER = THREE / "addons/exporters/GLTFExporter.js"
IMPORT_MAP = '<script type="importmap">{"imports":{"three":"/three/three.module.js","three/addons/":"/three/addons/"}}</script>'
CSP = ("sandbox allow-scripts; default-src 'none'; script-src 'unsafe-inline' " + ORIGIN +
       "; style-src 'unsafe-inline'; img-src data: blob:; connect-src 'none'; base-uri 'none'; form-action 'none'; frame-src 'none'")

# Read the export Blob before the generated page revokes its URL. Never accept a file download or use the page's filename.
SEEDED_RANDOM = """(() => {
    let state = SEED >>> 0;
    Math.random = () => { state = (Math.imul(1664525, state) + 1013904223) >>> 0; return state / 4294967296; };
})();"""
CAPTURE = SEEDED_RANDOM + """\n(() => {
    const create = URL.createObjectURL.bind(URL);
    let finish;
    Object.defineProperty(window, '__glbExport', {value: new Promise(resolve => { finish = resolve; })});
    Object.defineProperty(window, '__glbExportError', {value: err => finish({error: String(err).slice(0, 500)})});
    URL.createObjectURL = blob => {
        if (blob instanceof Blob && blob.size >= 12) {
            blob.slice(0, 4).text().then(async magic => {
                if (magic !== 'glTF') return;
                if (blob.size > MAX_BYTES) { finish('too-large'); return; }
                const reader = new FileReader();
                reader.onload = () => finish(reader.result.split(',')[1]);
                reader.readAsDataURL(blob);
            });
        }
        return create(blob);
    };
})();"""

# Generated pages sometimes pass a group as their clip list. Three.js stores the group's actual clips in Object3D.animations.
# Normalize this at the sandbox's exporter boundary, preserving the tracks, keyframes and geometry supplied by the page.
EXPORT_ANIMATIONS = """
const parseAsset = GLTFExporter.prototype.parse;
GLTFExporter.prototype.parse = function(input, done, error, options) {
    if (!Array.isArray(input)) {
        const clips = options?.animations ?? input.animations;
        if (Array.isArray(clips)) {
            options = {...options, animations: clips.flatMap(clip => clip?.isObject3D ? clip.animations : [clip])};
        }
    }
    const failed = err => {
        window.__glbExportError?.(err);
        if (error) error(err);
    };
    return parseAsset.call(this, input, done, failed, options);
};
"""


class SceneExportError(ValueError):
    """An error in model-produced content that a source edit could resolve."""


def _page(text, *, source_seed=None):
    start = re.search(r"<!doctype\s+html\b|<html\b", text, re.I)
    end = text.lower().rfind("</html>")
    if not start or end < start.start():
        raise SceneExportError("GLB 3D Coder did not finish a complete HTML scene.")
    html = text[start.start():end + len("</html>")]
    # The trained examples commonly supply their own Three.js CDN map. Serve the installed modules at one origin.
    html = re.sub(r"""<script\b[^>]*\btype\s*=\s*["']importmap["'][^>]*>.*?</script\s*>""", "", html, flags=re.I | re.S)
    html = re.sub(r"""https?://(?:cdn\.jsdelivr\.net/npm/|unpkg\.com/)three(?:@[\w.^~-]+)?/(?:build/three\.module(?:\.min)?\.js)""",
                  ORIGIN + "/three/three.module.js", html)
    html = re.sub(r"""https?://(?:cdn\.jsdelivr\.net/npm/|unpkg\.com/)three(?:@[\w.^~-]+)?/examples/jsm/""", ORIGIN + "/three/addons/", html)
    first_script = re.search(r"<script\b", html, re.I)
    if first_script is None:
        raise SceneExportError("The generated scene contains no Three.js code.")
    prefix = IMPORT_MAP
    if source_seed is not None:
        prefix += "\n<script>" + SEEDED_RANDOM.replace("SEED", str(int(source_seed) & 0xffffffff)) + "</script>"
        prefix += """\n<script type="module">import {GLTFExporter} from 'three/addons/exporters/GLTFExporter.js';""" + EXPORT_ANIMATIONS + "</script>\n"
    html = html[:first_script.start()] + prefix + html[first_script.start():]
    if source_seed is not None:
        html = html.replace(ORIGIN + "/three/", "./three/").replace('"/three/', '"./three/')
        html = "<!-- To run locally, serve this HTML beside a copy of WanGP's shared/apps/lib/three directory. -->\n" + html
    return html


def _validate_glb(data):
    from shared.mesh3d.glb import _split_glb

    if len(data) < 28 or data[:4] != b"glTF" or struct.unpack_from("<I", data, 8)[0] != len(data):
        raise ValueError("The generated export is not a complete binary GLB.")
    doc, binary, _ = _split_glb(data)
    if binary is None or not doc.get("meshes") or len(doc.get("buffers", [])) != 1:
        raise ValueError("The exported GLB contains no embedded mesh.")
    if any("uri" in item for key in ("images", "buffers") for item in doc.get(key, [])):
        raise ValueError("The generated GLB must embed all of its resources.")
    # The preview flattens scene instances. Reject cycles, shared children and excessive expanded geometry before that traversal.
    nodes = doc["nodes"]
    seen, vertices, indices = set(), 0, 0
    pending = list(doc["scenes"][doc.get("scene", 0)]["nodes"])
    while pending:
        index = pending.pop()
        if not 0 <= index < len(nodes) or index in seen:
            raise ValueError("Invalid node hierarchy in generated GLB.")
        seen.add(index)
        node = nodes[index]
        pending.extend(node.get("children", []))
        if "mesh" in node:
            for part in doc["meshes"][node["mesh"]]["primitives"]:
                if "POSITION" not in part.get("attributes", {}):
                    raise ValueError("A mesh has a geometry without vertex positions (no position attribute): give every exported mesh a BufferGeometry with positions.")
                count = doc["accessors"][part["attributes"]["POSITION"]]["count"]
                vertices += count
                indices += doc["accessors"][part["indices"]]["count"] if "indices" in part else count
        if len(seen) > 16384 or vertices > 2_000_000 or indices > 6_000_000:
            raise ValueError("The generated scene is too large for a low-poly asset.")
    if not vertices or any("bufferView" not in a or not 0 <= a["count"] <= 6_000_000 for a in doc["accessors"]):
        raise ValueError("Invalid mesh accessors in generated GLB.")


async def _export(html, seed, check_abort):
    from playwright.async_api import TimeoutError as PlaywrightTimeout, async_playwright
    from shared.deepy.web_screenshot import system_browser

    async def serve(route):
        url = urlsplit(route.request.url)
        if route.request.method == "GET" and url.netloc == "glb-coder.invalid":
            if url.path == "/scene" and route.request.is_navigation_request():
                return await route.fulfill(body=html, content_type="text/html", headers={"Content-Security-Policy": CSP})
            if url.path.startswith("/three/") and url.path.endswith(".js"):
                file = (THREE / url.path[len("/three/"):]).resolve()
                if THREE in file.parents and file.is_file():
                    response = {"body": file.read_text(encoding="utf-8") + EXPORT_ANIMATIONS} if file == THREE_EXPORTER else {"path": file}
                    return await route.fulfill(**response, content_type="text/javascript", headers={"Access-Control-Allow-Origin": "*"})
        await route.abort("blockedbyclient")

    async with async_playwright() as pw:
        executable = system_browser()
        if executable is None and not Path(pw.chromium.executable_path).is_file():
            raise RuntimeError("GLB export needs Edge, Chrome or Chromium. Install a browser or run python -m playwright install chromium.")
        browser = await pw.chromium.launch(executable_path=executable, chromium_sandbox=True, headless=True,
                                           args=["--disable-gpu", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                                                 "--proxy-server=http://127.0.0.1:9", "--proxy-bypass-list=<-loopback>",
                                                 "--force-webrtc-ip-handling-policy=disable_non_proxied_udp"])
        try:
            # The CSP sandbox has an opaque origin, so service workers, storage and cookies are inaccessible.
            context = await browser.new_context(accept_downloads=False)
            await context.route("**/*", serve)
            await context.add_init_script(script=CAPTURE.replace("SEED", str(int(seed) & 0xffffffff)).replace("MAX_BYTES", str(MAX_GLB_BYTES)))
            page = await context.new_page()
            errors = []
            page.on("pageerror", lambda exc: errors.append((getattr(exc, "stack", None) or str(exc))[:800]) if not errors else None)
            console_errors = []
            page.on("console", lambda msg: console_errors.append(msg.text[:400]) if msg.type == "error" and len(console_errors) < 6 else None)

            async def collect():
                await page.goto(ORIGIN + "/scene", wait_until="load")
                button = page.get_by_role("button", name=re.compile(r"export|download|glb", re.I)).first
                await button.click()
                encoded = await page.evaluate("""async limit => {
                    const data = await window.__glbExport;
                    if (typeof data?.error === 'string') return {error: data.error.slice(0, 500)};
                    return typeof data === 'string' && data.length <= limit ? data : 'too-large';
                }""", 4 * ((MAX_GLB_BYTES + 2) // 3))
                if encoded == "too-large":
                    raise SceneExportError("The generated GLB exceeded the 32 MiB export limit.")
                if isinstance(encoded, dict):
                    raise SceneExportError(f"GLTFExporter failed: {encoded['error']}")
                return base64.b64decode(encoded, validate=True)

            job = asyncio.create_task(collect())
            try:
                deadline = asyncio.get_running_loop().time() + 35
                while not job.done():
                    await asyncio.wait([job], timeout=1 / 3)
                    check_abort()
                    if errors:
                        raise SceneExportError(f"The generated Three.js page failed: {errors[0]}")
                    if asyncio.get_running_loop().time() >= deadline:
                        raise SceneExportError("The generated page did not export a GLB within 35 seconds. " + " ".join(console_errors))
                return job.result()
            except PlaywrightTimeout as exc:
                raise SceneExportError("Timed out waiting for the generated page or its Export GLB button. " + " ".join(console_errors)) from exc
            finally:
                job.cancel()
                await asyncio.gather(job, return_exceptions=True)
        finally:
            await browser.close()


def export_glb(text, seed, check_abort):
    html = _page(text)
    loop = asyncio.ProactorEventLoop() if sys.platform == "win32" else asyncio.new_event_loop()
    try:
        data = loop.run_until_complete(_export(html, seed, check_abort))
    finally:
        loop.close()
    try:
        _validate_glb(data)
    except (ValueError, KeyError, IndexError, TypeError, struct.error) as exc:
        raise SceneExportError(f"Invalid exported GLB: {exc}") from exc
    return data
