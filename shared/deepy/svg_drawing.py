# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""SVG drawings for Deepy's Draw SVG tool, rasterized by resvg.

Only the drawing itself is trusted to resvg, which would otherwise read local files named in href attributes: every
reference outside the drawing (an href that is not #id or an embedded raster image) is resolved by the caller through
its media and file access rules, then embedded as a data URI. url() may only point inside the drawing, and DOCTYPE /
ENTITY declarations are rejected.
"""
from __future__ import annotations

import base64
import copy
import io
import json
import math
import os
import re
import struct
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

from PIL import Image, ImageColor

from shared.deepy.image_channels import exif_orientation, upright

SVG_NS, XLINK_NS = "http://www.w3.org/2000/svg", "http://www.w3.org/1999/xlink"
HREFS = ("href", f"{{{XLINK_NS}}}href")
MAX_SVG_CHARS, MAX_EMBEDDED_CHARS, MAX_SIDE, MAX_REFERENCES = 2_000_000, 64_000_000, 4096, 32  # markup and embedded base64 are measured separately
EMBEDDED_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif"}
_EMBEDDED = re.compile(r"data:image/(png|jpeg|gif);base64,", re.IGNORECASE)
_PAYLOAD = re.compile(r"(?<=;base64,)[A-Za-z0-9+/=\s]+")
_EXTERNAL_URL = re.compile(r"url\(\s*['\"]?(?!#)", re.IGNORECASE)
_LENGTH = re.compile(r"^\s*([0-9.]+)\s*(px)?\s*$")
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


def _parse(svg: str) -> ET.Element:
    if not isinstance(svg, str) or not svg.strip():
        raise ValueError("svg must be a non-empty SVG document.")
    embedded = sum(match.end() - match.start() for match in _PAYLOAD.finditer(svg))
    if len(svg) - embedded > MAX_SVG_CHARS:
        raise ValueError(f"The SVG markup exceeds {MAX_SVG_CHARS:,} characters (embedded image data is not counted).")
    if embedded > MAX_EMBEDDED_CHARS:
        raise ValueError(f"The embedded images total {embedded * 3 / 4e6:.1f} MB, above {MAX_EMBEDDED_CHARS * 3 // 4_000_000} MB: reference smaller images, for instance resized with resize_crop.")
    if re.search(r"<!\s*(DOCTYPE|ENTITY)", svg, re.IGNORECASE):
        raise ValueError("The SVG must not contain DOCTYPE or ENTITY declarations.")
    if _EXTERNAL_URL.search(svg):
        raise ValueError("url() may only reference elements of the drawing, as url(#id) (gradients, patterns, filters, clip paths, masks).")
    try:
        root = ET.fromstring(svg)
    except ET.ParseError as error:
        hint = ' A prefix such as xlink: needs its declaration on the <svg> element (xmlns:xlink="http://www.w3.org/1999/xlink"), or use plain href.' if "unbound prefix" in str(error) else ""
        raise ValueError(f"Invalid SVG markup: {error}.{hint}") from error
    if root.tag not in ("svg", f"{{{SVG_NS}}}svg"):
        raise ValueError("The root element must be <svg>.")
    return root


def references(svg: str) -> list[str]:
    """Distinct external references of a drawing, in document order: every href that is neither #id nor an embedded
    PNG / JPEG / GIF data URI."""
    found = []
    for element in _parse(svg).iter():
        for name in HREFS:
            value = (element.get(name) or "").strip()
            if value and not value.startswith("#") and not _EMBEDDED.match(value) and value not in found:
                found.append(value)
    if len(found) > MAX_REFERENCES:
        raise ValueError(f"A drawing can reference at most {MAX_REFERENCES} different images; a 3D turntable is render_3d_views with turntable_s.")
    return found


def rewrite(svg: str, replacements: dict[str, str]) -> str:
    """The drawing with each external reference replaced (exact href values)."""
    root = _parse(svg)
    for element in root.iter():
        for name in HREFS:
            value = (element.get(name) or "").strip()
            if value in replacements:
                element.set(name, replacements[value])
    return ET.tostring(root, encoding="unicode")


def data_uri(path: str) -> str:
    """An image file as a data URI: PNG, JPEG and GIF as they are, other formats and EXIF-rotated photos (resvg ignores the
    orientation) converted to upright PNG (first frame)."""
    mime = EMBEDDED_TYPES.get(Path(path).suffix.lower())
    with Image.open(path) as image:
        if mime is not None and exif_orientation(image) == 1:
            data = Path(path).read_bytes()
        else:
            buffer = io.BytesIO()
            upright(image).convert("RGBA").save(buffer, "PNG")
            data, mime = buffer.getvalue(), "image/png"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _intrinsic_size(root: ET.Element) -> tuple[float, float] | None:
    lengths = [_LENGTH.match(root.get(name) or "") for name in ("width", "height")]
    if all(lengths):
        return float(lengths[0].group(1)), float(lengths[1].group(1))
    box = (root.get("viewBox") or "").replace(",", " ").split()
    return (float(box[2]), float(box[3])) if len(box) == 4 else None


def rasterize(svg: str, width: int | None = None, height: int | None = None, background: str | None = None) -> Image.Image:
    """RGBA image of a self-contained drawing (no external reference left), at most MAX_SIDE pixels per side: width /
    height scale the drawing (one of them keeps its aspect ratio), background is a CSS color (transparent by default)."""
    import resvg_py

    if references(svg):
        raise ValueError("The drawing still has external references.")
    if any(value is not None and not 1 <= value <= MAX_SIDE for value in (width, height)):
        raise ValueError(f"width and height must be between 1 and {MAX_SIDE} pixels.")
    zoom = None
    if width is None and height is None and (size := _intrinsic_size(_parse(svg))) and max(size) > MAX_SIDE:
        zoom = MAX_SIDE / max(size)
    image = Image.open(io.BytesIO(bytes(resvg_py.svg_to_bytes(svg_string=svg, width=width, height=height, zoom=zoom, background=background))))
    image.load()
    if max(image.size) > MAX_SIDE:
        raise ValueError(f"The drawing renders at {image.width}x{image.height}; set width or height (at most {MAX_SIDE}).")
    return image.convert("RGBA")


# ------------------------------------------------------------------------------------------------------- animation
# resvg draws still images: WanGP evaluates the SVG animation elements at each frame time, then draws that frame.
ANIMATIONS = ("animate", "animateTransform", "set")
_CLOCK = re.compile(r"^\s*(-?[\d.]+)\s*(ms|s|min|h)?\s*$")
_NUMBER = re.compile(r"-?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _name(element: ET.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _seconds(value: str | None, default: float) -> float:
    if value is None or not value.strip():
        return default
    if value.strip() == "indefinite":
        return math.inf
    match = _CLOCK.match(value)
    if match is None:
        raise ValueError(f"Unsupported animation time {value!r}: use seconds such as 1.5s or 500ms.")
    return float(match[1]) * {"ms": 0.001, "s": 1, "min": 60, "h": 3600}[match[2] or "s"]


def _interpolate(start: str, end: str, fraction: float) -> str:
    """Colours blend; numbers (and lists of numbers such as "10 20") move linearly; anything else switches at the end."""
    try:
        first, last = ImageColor.getrgb(start), ImageColor.getrgb(end)
        return "#" + "".join(f"{round(a + (b - a) * fraction):02x}" for a, b in zip(first[:3], last[:3]))
    except ValueError:
        pass
    first, last = _NUMBER.findall(start), _NUMBER.findall(end)
    if first and len(first) == len(last):
        moved = iter(f"{float(a) + (float(b) - float(a)) * fraction:g}" for a, b in zip(first, last))
        return _NUMBER.sub(lambda _match: next(moved), start)
    return end if fraction >= 1 else start


def _ease(fraction: float, x1: float, y1: float, x2: float, y2: float) -> float:
    """keySplines easing: the cubic Bezier from (0,0) to (1,1) with control points (x1,y1) and (x2,y2)."""
    low, high = 0.0, 1.0
    for _ in range(40):
        middle = (low + high) / 2
        x = 3 * (1 - middle) ** 2 * middle * x1 + 3 * (1 - middle) * middle ** 2 * x2 + middle ** 3
        low, high = (middle, high) if x < fraction else (low, middle)
    t = (low + high) / 2
    return 3 * (1 - t) ** 2 * t * y1 + 3 * (1 - t) * t ** 2 * y2 + t ** 3


def _value_at(animation: ET.Element, time: float, base: str) -> str | None:
    """The animated value at document time, or None when the animation has no effect then (before begin, or ended without fill="freeze")."""
    local = time - _seconds(animation.get("begin"), 0.0)
    if _name(animation) == "set":
        duration, values = _seconds(animation.get("dur"), math.inf), [animation.get("to", base)]
    else:
        duration = _seconds(animation.get("dur"), math.nan)
        if not duration > 0 or math.isinf(duration):
            raise ValueError(f"<{_name(animation)}> needs a positive dur such as dur=\"2s\".")
        values = [value.strip() for value in animation.get("values").split(";")] if animation.get("values") else [animation.get("from", base), animation.get("to", base)]
    repeat = animation.get("repeatCount", "1")
    active = duration * (math.inf if repeat == "indefinite" else float(repeat))
    if local < 0 or local >= active and animation.get("fill") != "freeze":
        return None
    progress = 1.0 if local >= active else (local % duration) / duration if math.isfinite(duration) else 0.0
    if len(values) == 1:
        return values[0]
    times = [float(value) for value in animation.get("keyTimes").split(";")] if animation.get("keyTimes") else None
    if animation.get("calcMode") == "discrete":
        times = times or [index / len(values) for index in range(len(values))]
        return values[max(index for index, start in enumerate(times) if start <= progress)]
    times = times or [index / (len(values) - 1) for index in range(len(values))]
    segment = min(max(index for index, start in enumerate(times) if start <= progress), len(values) - 2)
    fraction = min(max((progress - times[segment]) / max(times[segment + 1] - times[segment], 1e-9), 0.0), 1.0)
    if animation.get("calcMode") == "spline":
        splines = [[float(number) for number in _NUMBER.findall(spline)] for spline in animation.get("keySplines", "").split(";")]
        fraction = _ease(fraction, *splines[segment])
    return _interpolate(values[segment], values[segment + 1], fraction)


def _set_property(element: ET.Element, name: str, value: str) -> None:
    style = element.get("style") or ""
    declaration = re.compile(rf"((?:^|;)\s*{re.escape(name)}\s*:)[^;]*")
    if declaration.search(style):  # a style declaration would override the attribute
        element.set("style", declaration.sub(lambda match: match.group(1) + value, style))
    else:
        element.set(name, value)


def is_animated(svg: str) -> bool:
    return any(_name(element) in ANIMATIONS for element in _parse(svg).iter())


def frame_at(root: ET.Element, time: float) -> str:
    """The SVG of a self-contained drawing at a time in seconds, with its animations applied and removed."""
    root = copy.deepcopy(root)
    ids = {element.get("id"): element for element in root.iter() if element.get("id")}
    parents = {child: parent for parent in root.iter() for child in parent}
    for animation in [element for element in root.iter() if _name(element) in ANIMATIONS]:
        reference = next((animation.get(name) for name in HREFS if animation.get(name)), None)
        target = ids[reference[1:]] if reference and reference.startswith("#") else parents[animation]
        if _name(animation) == "animateTransform":
            value = _value_at(animation, time, "0")
            if value is not None:
                transform = f"{animation.get('type', 'translate')}({value})"
                target.set("transform", f"{target.get('transform', '')} {transform}".strip() if animation.get("additive") == "sum" else transform)
        else:
            name = animation.get("attributeName")
            if not name:
                raise ValueError(f"<{_name(animation)}> needs attributeName.")
            value = _value_at(animation, time, target.get(name, ""))
            if value is not None:
                _set_property(target, name, value)
        parents[animation].remove(animation)
    return ET.tostring(root, encoding="unicode")


PARALLEL_FRAMES = 24  # animated clips with at least this many distinct frames render on worker processes
_ROOT = Path(__file__).resolve().parents[2]


def _render(svg: str, options: dict, mode: str):
    import resvg_py

    return Image.open(io.BytesIO(bytes(resvg_py.svg_to_bytes(svg_string=svg, **options)))).convert(mode)


def _runs(texts):
    """(text, repeat) for each run of identical frames: a still part renders once."""
    previous, repeat = None, 0
    for text in texts:
        if text == previous:
            repeat += 1
            continue
        if previous is not None:
            yield previous, repeat
        previous, repeat = text, 1
    yield previous, repeat


def _render_parallel(runs, options: dict, mode: str, workers: int):
    """Frames of the runs in order, rendered by worker processes (svg_frame_worker) with one frame in flight each, so that no pipe can block."""
    processes = [subprocess.Popen([sys.executable, "-m", "shared.deepy.svg_frame_worker"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, cwd=_ROOT) for _ in range(workers)]

    def send(process, text):
        payload = text.encode("utf-8")
        process.stdin.write(struct.pack("<I", len(payload)) + payload)
        process.stdin.flush()

    try:
        for process in processes:
            process.stdin.write(json.dumps({"options": options, "mode": mode}).encode("utf-8") + b"\n")
        in_flight = deque()
        for process, (text, repeat) in zip(processes, runs):
            send(process, text)
            in_flight.append((process, repeat))
        while in_flight:
            process, repeat = in_flight.popleft()
            header = process.stdout.read(8)
            if len(header) < 8:
                raise RuntimeError("A drawing worker process stopped unexpectedly.")
            frame_width, frame_height = struct.unpack("<II", header)
            if frame_width == 0:
                raise ValueError(process.stdout.read(frame_height).decode("utf-8"))
            frame = Image.frombytes(mode, (frame_width, frame_height), process.stdout.read(frame_width * frame_height * len(mode)))
            if (following := next(runs, None)) is not None:
                send(process, following[0])
                in_flight.append((process, following[1]))
            yield frame, repeat
    finally:
        for process in processes:
            process.kill()
            process.wait()


def render_frames(svg: str, duration: float, fps: float, width: int | None = None, height: int | None = None, background: str | None = "black", mode: str = "RGB"):
    """RGB (or RGBA) frames of a self-contained drawing over duration seconds. Identical consecutive frames render once (a drawing without
    animation renders a single frame); long animations render on several worker processes at once."""
    if references(svg):
        raise ValueError("The drawing still has external references.")
    root = _parse(svg)
    animated = any(_name(element) in ANIMATIONS for element in root.iter())
    zoom = None
    if width is None and height is None and (size := _intrinsic_size(root)) and max(size) > MAX_SIDE:
        zoom = MAX_SIDE / max(size)
    options = {"width": width, "height": height, "zoom": zoom, "background": background}
    count = max(1, round(duration * fps))
    runs = _runs(frame_at(root, index / fps) for index in range(count)) if animated else iter([(svg, count)])
    rendered = _render_parallel(runs, options, mode, min(12, max(1, (os.cpu_count() or 2) // 2))) if animated and count >= PARALLEL_FRAMES else ((_render(text, options, mode), repeat) for text, repeat in runs)
    for frame, repeat in rendered:
        if max(frame.size) > MAX_SIDE:
            raise ValueError(f"The drawing renders at {frame.width}x{frame.height}; set width or height (at most {MAX_SIDE}).")
        for _ in range(repeat):
            yield frame
