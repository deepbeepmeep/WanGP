# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""A headless run of a web app for Deepy's app_check: the installed Edge or Chrome (Playwright, a fresh profile) opens the app exactly as the
player serves it (same CSP, runtime and import map) from a private address that never touches the network: every request the page makes is
answered from the bundle and the vendored libraries by request routing, anything else is refused and reported. The run records console lines,
uncaught errors, failed loads, the WebGL renderer, the frame rate, the app's state probe (window.wangpState) after each step, screenshots, a
motion strip over a few seconds and a short outline of the page."""
from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from shared.apps import bundles, server

HOST = "app.wangp.invalid"
ORIGIN = f"https://{HOST}"
TOKEN = "check"
MAX_STEPS = 40
MAX_RECORD_S = 10.0
RUN_TIMEOUT_S = 120
SETTLE_S = 1.5
VIEWPORTS = {"desktop": {"width": 1280, "height": 720}, "phone": {"width": 390, "height": 844, "is_mobile": True, "has_touch": True}}
_OUTLINE = """() => {
  const rows = [];
  for (const node of document.querySelectorAll('h1,h2,h3,p,button,label,input,select,textarea,canvas,video,a,[role=button],[data-score],.score,#score,#hud')) {
    const box = node.getBoundingClientRect();
    if (box.width < 2 || box.height < 2 || getComputedStyle(node).visibility === 'hidden') continue;
    const outside = box.right < 0 || box.bottom < 0 || box.left > innerWidth || box.top > innerHeight;
    const text = (node.innerText || node.value || node.getAttribute('aria-label') || '').trim().replace(/\\s+/g, ' ').slice(0, 60);
    rows.push(`${node.tagName.toLowerCase()}${node.id ? '#' + node.id : ''} ${Math.round(box.width)}x${Math.round(box.height)} at ${Math.round(box.left)},${Math.round(box.top)}${outside ? ' (outside the window)' : ''}${text ? ': ' + text : ''}`);
    if (rows.length >= 40) break;
  }
  const wide = document.documentElement.scrollWidth > innerWidth + 2, tall = document.documentElement.scrollHeight > innerHeight + 2;
  return {elements: rows, scrolls: wide || tall ? `the page scrolls (${document.documentElement.scrollWidth}x${document.documentElement.scrollHeight} in a ${innerWidth}x${innerHeight} window)` : ''};
}"""
_FRAME_RATE = """() => new Promise(done => { let count = 0; const start = performance.now();
  const step = () => { count += 1; if (performance.now() - start < 1000) requestAnimationFrame(step); else done(Math.round(count * 10000 / (performance.now() - start)) / 10); };
  requestAnimationFrame(step); setTimeout(() => done(Math.round(count * 10000 / (performance.now() - start)) / 10), 2500); })"""
_STATE = "() => { try { return typeof window.wangpState === 'function' ? JSON.parse(JSON.stringify(window.wangpState())) : null; } catch (error) { return {probe_error: String(error)}; } }"


def run(folder: Path, out_dir: Path, *, steps: list[dict[str, Any]] | None = None, record: dict[str, Any] | None = None, viewport: str = "desktop") -> dict[str, Any]:
    """Run the bundle in folder headless; screenshots and the motion strip go to out_dir. Raises RuntimeError when no browser can run it."""
    from shared.deepy import web_screenshot

    if not web_screenshot.available():
        raise RuntimeError("app_check needs Playwright (pip install playwright) and Edge or Chrome installed.")
    if viewport not in VIEWPORTS:
        raise ValueError(f"viewport must be one of: {', '.join(VIEWPORTS)}.")
    if steps is not None and (not isinstance(steps, list) or len(steps) > MAX_STEPS):
        raise ValueError(f"steps must be a list of at most {MAX_STEPS} actions.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.png"):  # earlier checks' screenshots stay readable a while: results are often inspected after the next check
        if time.time() - old.stat().st_mtime > 1800:
            old.unlink()
    outcome: dict[str, Any] = {}

    def work() -> None:
        loop = asyncio.ProactorEventLoop() if sys.platform == "win32" else asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            outcome["result"] = loop.run_until_complete(asyncio.wait_for(_run(Path(folder), out_dir, steps or [], record, viewport, web_screenshot.system_browser()), RUN_TIMEOUT_S))
        except BaseException as error:
            outcome["error"] = error
        finally:
            loop.close()

    worker = threading.Thread(target=work, name="wangp-app-check", daemon=True)
    worker.start()
    worker.join()
    error = outcome.get("error")
    if isinstance(error, (ValueError, asyncio.TimeoutError, TimeoutError)):
        raise RuntimeError(f"The app check did not finish within {RUN_TIMEOUT_S} s." if not isinstance(error, ValueError) else str(error)) from error
    if error is not None:
        raise RuntimeError(f"The browser could not run the app: {type(error).__name__}: {error}".rstrip(": ")) from error
    return outcome["result"]


async def _run(folder: Path, out_dir: Path, steps: list[dict[str, Any]], record: dict[str, Any] | None, viewport: str, executable: str | None) -> dict[str, Any]:
    from playwright.async_api import async_playwright
    from playwright.async_api import Error as PlaywrightError

    manifest = bundles.read_manifest(folder)
    entry = server.AppEntry("dev", folder, manifest["entry"], manifest["name"])  # no saved data: the check starts fresh and keeps nothing
    netloc = HOST
    console: list[str] = []
    errors: list[str] = []
    missing: list[str] = []
    refused: list[str] = []
    reported: dict[str, Any] = {}
    store: dict[str, str] = {}

    async def answer(route, request):
        url = urlsplit(request.url)
        path = unquote(url.path)
        if url.netloc != HOST:
            refused.append(request.url[:200])
            return await route.abort("blockedbyclient")
        if path.startswith("/apps/lib/"):
            try:
                file = server.lib_file(path[len("/apps/lib/"):])
            except FileNotFoundError:
                missing.append(path)
                return await route.fulfill(status=404, body="")
            return await route.fulfill(path=str(file), headers={"Access-Control-Allow-Origin": "*", "Content-Type": server.media_type(file)})
        prefix = f"/apps/r/{TOKEN}/"
        if not path.startswith(prefix):
            refused.append(f"{path} (an absolute path outside the app)")
            return await route.fulfill(status=404, body="")
        relative = path[len(prefix):]
        headers = server.frame_headers(netloc, TOKEN)
        if relative.startswith("__wangp__/"):
            try:
                payload = json.loads(request.post_data or "{}")
            except ValueError:
                payload = {}
            if relative.endswith("log"):
                for item in payload.get("entries") or []:
                    if item.get("level") in ("error", "exception"):
                        errors.append(str(item.get("text"))[:500])
                reported.update(payload.get("stats") or {})
            elif relative.endswith("store") and request.method == "POST":
                store.update({str(key): str(value) for key, value in (payload.get("set") or {}).items()})
            return await route.fulfill(status=204, headers=headers)
        try:
            file = server.bundle_file(folder, relative or entry.entry)
        except FileNotFoundError:
            missing.append(relative)
            return await route.fulfill(status=404, headers=headers, body="")
        if file.suffix.lower() in server.PAGE_SUFFIXES:
            return await route.fulfill(status=200, headers={**headers, "Content-Type": "text/html; charset=utf-8"}, body=server.inject_runtime(file.read_text(encoding="utf-8", errors="replace"), entry))
        return await route.fulfill(status=200, headers={**headers, "Content-Type": server.media_type(file)}, path=str(file))

    async with async_playwright() as playwright:
        args = ["--autoplay-policy=no-user-gesture-required", "--proxy-server=http://127.0.0.1:9", "--proxy-bypass-list=<-loopback>"]  # nothing un-routed leaves
        browser = await playwright.chromium.launch(executable_path=executable, headless=True, args=args)
        try:
            context = await browser.new_context(viewport={key: value for key, value in VIEWPORTS[viewport].items() if key in ("width", "height")}, is_mobile=VIEWPORTS[viewport].get("is_mobile", False),
                                                has_touch=VIEWPORTS[viewport].get("has_touch", False), service_workers="block", accept_downloads=False)
            await context.route("**/*", answer)
            page = await context.new_page()
            page.on("console", lambda message: console.append(f"{message.type}: {message.text}"[:500]))  # uncaught errors arrive through the runtime's reports, with their location
            started = time.monotonic()
            await page.goto(f"{ORIGIN}/apps/r/{TOKEN}/{manifest['entry']}", wait_until="load", timeout=30000)
            await page.wait_for_timeout(SETTLE_S * 1000)
            load_s = round(time.monotonic() - started, 2)
            shots = []

            async def shot(name: str) -> dict[str, Any]:
                target = out_dir / f"{len(shots) + 1:02d}_{_safe(name)}.png"
                await page.screenshot(path=str(target))
                item = {"name": name, "path": str(target), **_picture_facts(target)}
                shots.append(item)
                return item

            first = await shot("start")
            fps = await page.evaluate(_FRAME_RATE)
            loaded = await page.evaluate(_STATE)
            done_steps = []
            for index, step in enumerate(steps, 1):
                outcome: dict[str, Any] = {"step": index, **{key: value for key, value in step.items() if key in ("key", "click", "type", "mouse", "drag", "wait_s", "shot", "hold_s", "repeat")}}
                try:
                    await _perform(page, step, shot)
                except (PlaywrightError, ValueError, TypeError) as error:
                    outcome["error"] = str(error).splitlines()[0][:300]
                outcome["state"] = await page.evaluate(_STATE)
                done_steps.append(outcome)
            motion = None
            if record is not None:
                motion = await _record(page, out_dir, float(record.get("seconds", 3)), float(record.get("fps", 4)))
            state = await page.evaluate(_STATE) if steps or record is not None else loaded  # the state the steps led to, which is what a check verifies
            last = shots[-1] if len(shots) > 1 else await shot("end")
            outline = await page.evaluate(_OUTLINE)
            await page.wait_for_timeout(1200)  # the runtime's last reports
        finally:
            await browser.close()
    findings = []
    if missing:
        findings.append({"level": "error", "message": "Files the app asked for and that do not exist: " + ", ".join(sorted(set(missing))[:10])})
    if refused:
        findings.append({"level": "error", "message": "Refused requests (internet or absolute paths): " + ", ".join(sorted(set(refused))[:10])})
    blocked = [line.split(": ", 1)[-1] for line in console if "Content Security Policy" in line]
    if blocked:
        findings.append({"level": "error", "message": f"Blocked by the app's isolation (nothing loads from outside the app folder and the vendored libraries): {blocked[0]}"})
    if errors:
        findings.append({"level": "error", "message": f"{len(errors)} uncaught error(s); the first: {errors[0]}"})
    if first["blank"] and (last is None or last["blank"]):
        findings.append({"level": "error", "message": f"The page shows one flat colour ({first['color']}): nothing is drawn (a black three.js canvas usually means no light, a camera looking away, or an error before the first render)."})
    if motion is not None and motion["still"]:
        findings.append({"level": "warning", "message": "Nothing moved during the recording: check that the animation loop runs (or that this app is meant to be still)."})
    if outline.get("scrolls"):
        findings.append({"level": "warning", "message": f"Layout: {outline['scrolls']}; a game or animation should fit the window (100vw/100vh, overflow hidden)."})
    if loaded is None:
        findings.append({"level": "info", "message": "No window.wangpState(): define it to return the app's key values (score, level, player position) so that checks can verify logic without pictures."})
    if fps is not None and fps < 20:
        findings.append({"level": "warning", "message": f"{fps} frames per second in the check browser ({reported.get('renderer') or 'renderer unknown'}): too slow to play smoothly if the renderer is a real GPU."})
    for step in done_steps:
        if step.get("error"):
            findings.append({"level": "error", "message": f"Step {step['step']} failed: {step['error']}"})
    return {"load_s": load_s, "renderer": reported.get("renderer"), "fps": fps, "state": state, **({"state_at_load": loaded} if loaded != state else {}), "steps": done_steps, "screenshots": [{key: value for key, value in item.items() if key in ("name", "path", "blank")} for item in shots],
            "motion": motion, "outline": outline.get("elements", []), "console": console[-30:], "errors": errors[:10], "saved_data": store, "findings": findings}


KEY_NAMES = {"right": "ArrowRight", "left": "ArrowLeft", "up": "ArrowUp", "down": "ArrowDown", "space": "Space", "spacebar": "Space", " ": "Space", "enter": "Enter",
             "return": "Enter", "esc": "Escape", "escape": "Escape", "shift": "Shift", "ctrl": "Control", "control": "Control", "alt": "Alt", "tab": "Tab",
             "backspace": "Backspace", "delete": "Delete"}  # names agents write, as the browser's key names


def _plain_step(step: dict[str, Any]) -> dict[str, Any]:
    """A step in the form of browser automation tools ({"action": "click", "x": 10, "y": 20}, {"action": "wait", "ms": 500},
    {"action": "mouse", "arguments": {"point": [10, 20]}}, {"action": "drag", "from": [..], "to": [..]}) as the plain form ({"click": [10, 20]},
    {"wait_s": 0.5}, {"mouse": [10, 20]}, {"drag": [[..], [..]]})."""
    kind = {"move": "mouse", "hover": "mouse", "press": "key", "screenshot": "shot", "sleep": "wait"}.get(step["action"], step["action"])
    rest = {**{key: value for key, value in step.items() if key not in ("action", "arguments")}, **(step["arguments"] if isinstance(step.get("arguments"), dict) else {})}
    point = [rest["x"], rest["y"]] if "x" in rest and "y" in rest else next((rest[key] for key in ("point", "position", "at", "xy", "coordinates") if key in rest), None)
    if kind in ("click", "mouse") and point is not None:
        return {kind: point}
    if kind == "drag" and "from" in rest and "to" in rest:
        return {"drag": [rest["from"], rest["to"]]}
    if kind == "wait":
        return {"wait_s": next((rest[key] for key in ("seconds", "s", "wait_s", "duration") if key in rest), rest.get("ms", 1000) / 1000)}
    if kind in rest:
        return rest
    value = next((rest.pop(key) for key in ("selector", "text", "value", "name", "key") if key in rest), None)
    if value is None:
        raise ValueError(f"Unknown step {step}: use key (with hold_s), click (selector, text=Label or [x, y]), type, mouse [x, y], drag [[x1, y1], [x2, y2]], wait_s or shot.")
    return {kind: value, **rest}


PLAIN_NAMES = {"wait": "wait_s", "sleep": "wait_s", "press": "key", "hover": "mouse", "move": "mouse", "screenshot": "shot"}  # plain steps' other names


async def _perform(page, step: dict[str, Any], shot) -> None:
    step = _plain_step(step) if "action" in step else {PLAIN_NAMES.get(key, key): value for key, value in step.items()}
    if "wait_ms" in step:
        step = {**{key: value for key, value in step.items() if key != "wait_ms"}, "wait_s": float(step["wait_ms"]) / 1000}
    for index in range(max(1, min(int(step.get("repeat") or 1), 50))):
        if index:
            await page.wait_for_timeout(max(0.0, min(float(step.get("every_s") or 0), 2.0)) * 1000)
        await _perform_once(page, step, shot)


async def _perform_once(page, step: dict[str, Any], shot) -> None:
    if "wait_s" in step:
        await page.wait_for_timeout(max(0.0, min(float(step["wait_s"]), 10.0)) * 1000)
    elif "key" in step:
        key = KEY_NAMES.get(str(step["key"]).strip().lower(), str(step["key"]))
        if step.get("hold_s"):
            await page.keyboard.down(key)
            await page.wait_for_timeout(max(0.0, min(float(step["hold_s"]), 10.0)) * 1000)
            await page.keyboard.up(key)
        else:
            await page.keyboard.press(key)
    elif "click" in step:
        target = step["click"]
        if isinstance(target, (list, tuple)):
            await page.mouse.click(float(target[0]), float(target[1]))
        else:
            from playwright.async_api import TimeoutError as PlaywrightTimeoutError

            try:
                await page.click(str(target), timeout=5000)
            except PlaywrightTimeoutError:
                raise ValueError(f"no visible element matches {target} after 5 s: it is hidden, covered or not created (an error above may have stopped the app)") from None
    elif "type" in step:
        await page.keyboard.type(str(step["type"]), delay=20)
    elif "mouse" in step:
        await page.mouse.move(float(step["mouse"][0]), float(step["mouse"][1]), steps=8)
    elif "drag" in step:
        (x1, y1), (x2, y2) = step["drag"]
        await page.mouse.move(float(x1), float(y1))
        await page.mouse.down()
        await page.mouse.move(float(x2), float(y2), steps=16)
        await page.mouse.up()
    elif "shot" in step:
        await shot(str(step["shot"]))
        return
    else:
        raise ValueError("Unknown step: use key (with hold_s), click (selector, text=Label or [x, y]), type, mouse [x, y], drag [[x1, y1], [x2, y2]], wait_s or shot.")
    if "shot" in step:
        await shot(str(step["shot"]))


async def _record(page, out_dir: Path, seconds: float, fps: float) -> dict[str, Any]:
    """Frames over a few seconds, joined into one labelled strip image, with how much each frame differs from the previous one."""
    from PIL import Image, ImageChops, ImageDraw, ImageStat

    seconds, fps = max(0.5, min(seconds, MAX_RECORD_S)), max(1.0, min(fps, 8.0))
    count = max(2, int(seconds * fps))
    frames, times = [], []
    start = time.monotonic()
    for index in range(count):
        target = out_dir / f"frame_{index:02d}.png"
        await page.screenshot(path=str(target))
        times.append(round(time.monotonic() - start, 2))
        frames.append(target)
        delay = start + (index + 1) / fps - time.monotonic()
        if delay > 0:
            await page.wait_for_timeout(delay * 1000)
    images = [Image.open(path).convert("RGB") for path in frames]
    changes = [round(sum(ImageStat.Stat(ImageChops.difference(a, b)).mean) / 3, 2) for a, b in zip(images, images[1:])]
    columns = min(4, len(images))
    width = 480
    height = round(images[0].height * width / images[0].width)
    sheet = Image.new("RGB", (columns * width, ((len(images) + columns - 1) // columns) * (height + 22)), (24, 26, 30))
    draw = ImageDraw.Draw(sheet)
    for index, (image, moment) in enumerate(zip(images, times)):
        x, y = index % columns * width, index // columns * (height + 22)
        sheet.paste(image.resize((width, height)), (x, y + 22))
        draw.text((x + 6, y + 4), f"{moment:.2f} s", fill=(235, 238, 242))
    strip = out_dir / "motion.png"
    sheet.save(strip)
    for path in frames:
        path.unlink()
    return {"strip": str(strip), "frames": len(images), "times_s": times, "change_per_frame": changes, "still": max(changes or [0]) < 0.3}


def _picture_facts(path: Path) -> dict[str, Any]:
    from PIL import Image, ImageChops, ImageStat

    with Image.open(path) as image:
        small = image.convert("RGB").resize((320, 180), Image.BOX)
    color = tuple(round(value) for value in ImageStat.Stat(small.resize((1, 1), Image.BOX)).mean)
    distance = ImageChops.difference(small, Image.new("RGB", small.size, color)).convert("L")
    drawn = sum(distance.point(lambda value: 255 if value > 24 else 0).histogram()[255:])  # pixels clearly unlike the dominant colour: a few words are enough
    return {"blank": drawn < 4, "color": "#" + "".join(f"{value:02x}" for value in color)}


def _safe(name: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in name)[:40] or "shot"
