# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Deepy's Python sandbox worker: a persistent interpreter that runs the assistant's code in a separate, contained process.

Started by shared.deepy.python_sandbox as `python -I -B -u -X utf8 python_worker.py <sandbox root> <json config>`: isolated mode, so
neither the environment nor the current directory can inject modules, and this file imports only the standard library and the
allowed third-party libraries, never WanGP.

Protocol (JSON lines, UTF-8): the host writes {"id", "op": "run", "code"} or {"id", "op": "ping"}; the worker answers with the same id.
While code runs, the worker may ask the host for something ({"rpc": {"op": "path", ...}}) and reads the host's {"value"} or
{"error"} reply before continuing. print() output is captured, so the pipe carries protocol only.

Containment inside this process (the host adds OS limits: separate process, job object or rlimits, low integrity on Windows):
the assistant's code gets its own builtins with a guarded __import__ (allowlist below; library-internal imports stay unguarded so
numpy and PIL work), guarded open / os functions (writes only inside the sandbox folder, reads inside it, in the interpreter
installation, in the font folders and in paths granted by wangp.path), no process or network functions. Importing this module has
no side effects; main() applies the guards."""
import ast
import builtins
import glob
import io
import itertools
import json
import linecache
import os
import sys
import time
import traceback
import types

MAX_SVG_SIDE = 4096
ALLOWED_MODULES = {
    # computation and data
    "math", "cmath", "random", "statistics", "fractions", "decimal", "numbers", "itertools", "functools", "operator", "collections",
    "heapq", "bisect", "array", "copy", "enum", "dataclasses", "typing", "types", "abc", "contextlib", "pprint", "reprlib", "warnings", "inspect",
    # text
    "string", "re", "textwrap", "unicodedata", "difflib", "json", "csv", "struct", "base64", "binascii", "hashlib", "hmac", "zlib", "gzip",
    "bz2", "lzma", "zipfile", "tarfile", "io", "html", "xml", "uuid", "secrets", "colorsys", "wave", "aifc", "sunau", "audioop", "mimetypes",
    # time and files (os and io functions are guarded)
    "time", "datetime", "calendar", "zoneinfo", "locale", "os", "pathlib", "glob", "fnmatch", "tempfile", "shutil", "stat", "filecmp",
    # third-party, CPU only
    "numpy", "PIL", "scipy", "skimage", "cv2", "matplotlib", "mpl_toolkits", "soundfile", "sympy", "pandas", "resvg_py", "pypdf", "wangp",
}
LIBRARY_SUMMARY = "standard library (math, random, re, json, csv, textwrap, datetime, pathlib, zipfile, wave, colorsys, ...), numpy, PIL, scipy, skimage, cv2, matplotlib (Agg), soundfile, sympy, pandas, pypdf (PdfReader: text of a PDF)"
_REMOVED_OS = ("system", "popen", "execl", "execle", "execlp", "execlpe", "execv", "execve", "execvp", "execvpe", "spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv",
               "spawnve", "spawnvp", "spawnvpe", "posix_spawn", "posix_spawnp", "fork", "forkpty", "kill", "killpg", "startfile", "_exit", "abort",
               "chdir", "fchdir", "chroot", "setuid", "setgid", "seteuid", "setegid")  # environment changes stay inside this process: scipy and cv2 set variables at import
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC

_IN = sys.stdin.buffer
_OUT = sys.stdout.buffer
_real_import = builtins.__import__
_real_open = builtins.open
_os_originals = {}
ROOT = ""
ROOT_KEY = ""
READ_PREFIXES: list[str] = []
GRANTED: set[str] = set()
CONFIG: dict = {}


# ------------------------------------------------------------------------------------------------------ guards
def _key(path) -> str:
    path = os.fsdecode(path) if isinstance(path, (bytes, bytearray)) else os.fspath(path)
    return os.path.normcase(os.path.realpath(os.path.join(os.getcwd(), path)))


def _under(key: str, prefix: str) -> bool:
    return key == prefix or key.startswith(prefix.rstrip(os.sep) + os.sep)


_ALIASES: dict[str, str] = {}
_ALIAS_ROOTS: set[str] = set()  # local folders of the authorized roots seen so far (a write under one is pointed to wangp.io)


def _own(path):
    """The sandbox's own @alias path (@documents/sandbox/apps/x, as tool results print it, or /documents/sandbox/apps/x without its @) is its
    local path, readable and writable like a relative one."""
    alias, text = CONFIG.get("alias"), path.replace("\\", "/") if isinstance(path, str) else ""
    for prefix in (alias, "/" + alias.lstrip("@")) if alias and text else ():
        if text.startswith(prefix + "/"):
            return os.path.join(ROOT, text[len(prefix) + 1:])
    return path


def _tool_path(path, new=False):
    """A relative path of the sandbox (apps/x, out/a.png), as relative paths mean here, in the @alias form WanGP's tools take (they read a relative
    path from the outputs folder). Only a path that exists in the sandbox, or a new one (new=True) in a folder of it: media ids stay as they are."""
    if isinstance(path, str) and CONFIG.get("alias") and not path.startswith("@") and not os.path.isabs(path) and (os.path.exists(path) or new and os.path.isdir(os.path.dirname(path))):
        return CONFIG["alias"] + "/" + path.replace("\\", "/").removeprefix("./")
    return path


def _real(path):
    """An authorized @alias/... path (what wangp.listdir and the other tools return) becomes its local path, resolved once by the host and readable afterwards; other paths are returned as they are."""
    path = _own(path)
    if isinstance(path, str) and path.startswith("@") and len(path) > 1:
        if path not in _ALIASES:
            found = _rpc({"op": "realpath", "ref": path})
            key = os.path.normcase(os.path.realpath(found["path"]))
            if found.get("dir"):
                READ_PREFIXES.append(key)
                _ALIAS_ROOTS.add(key)
            else:
                GRANTED.add(key)
            _ALIASES[path] = found["path"]
        return _ALIASES[path]
    return path


def _outside_write(path, what: str) -> PermissionError:
    """Writing into an authorized folder outside the sandbox (an @alias path or a listed file's local path) goes through wangp.io."""
    return PermissionError(f"The sandbox cannot change {os.fspath(path)!s} directly ({what}): authorized folders outside the sandbox change through wangp.io in the same script, "
                           "e.g. wangp.io('mkdir', path='@test/new'), then wangp.io('move', source=e['path'], destination='@test/new/' + e['name']) for each file (copy the same way). "
                           "New files are saved inside the sandbox (out/x.png), then copied there.")


def _check(path, writing: bool, what: str) -> None:
    if isinstance(path, int):
        return
    path = _own(path)
    if writing and isinstance(path, str) and path.startswith("@"):
        raise _outside_write(path, what)
    key = _key(path)
    if _under(key, ROOT_KEY):
        return
    readable = key in GRANTED or any(_under(key, prefix) for prefix in READ_PREFIXES)
    if not writing and (readable or key == os.path.normcase(os.devnull)):
        return
    if writing and (key in GRANTED or any(_under(key, prefix) for prefix in _ALIAS_ROOTS)):
        raise _outside_write(path, what)
    verb = "write" if writing else "read"
    raise PermissionError(f"The sandbox cannot {verb} {os.fspath(path)!s} ({what}). Write inside the sandbox folder, which is the working directory (relative paths such as 'out/x.png' or 'tmp/frames'); read authorized files by their @alias path (open('@test/a.txt'), PIL.Image.open('@outputs/x.png')) or Gallery media through wangp.path(media_id). Copying or moving files between folders is wangp.io('copy' / 'move', ...).")


def _guarded_open(file, mode="r", *args, **kwargs):
    file = _own(file)
    if isinstance(file, str) and file.startswith("@") and any(flag in str(mode) for flag in "wax+"):
        raise PermissionError(f"The sandbox cannot write {file}: save the file inside the sandbox (out/result.csv, scratch/x.png), then copy it there with wangp.io('copy', source='@documents/sandbox/out/result.csv', destination={file!r}).")
    file = _real(file)
    _check(file, any(flag in str(mode) for flag in "wax+"), "open")
    return _real_open(file, mode, *args, **kwargs)


class _GuardedFileIO(io.FileIO):
    """io.FileIO opens files without builtins.open: it gets the same check."""

    def __init__(self, file, mode="r", closefd=True, opener=None):
        file = _real(file)
        _check(file, any(flag in str(mode) for flag in "wax+"), "FileIO")
        super().__init__(file, mode, closefd, opener)


def _guard_os():
    def blocked(name):
        def refuse(*_args, **_kwargs):
            raise PermissionError(f"os.{name} is not available in the sandbox.")
        refuse.__name__ = name
        return refuse

    for name in _REMOVED_OS:
        if hasattr(os, name):
            setattr(os, name, blocked(name))

    def guard(name, writing, *, both=False):
        original = _os_originals.setdefault(name, getattr(os, name))

        def guarded(path=".", *args, **kwargs):
            if writing and isinstance(path, str) and path.startswith("@"):
                raise _outside_write(path, f"os.{name}")
            path = _real(path)
            _check(path, writing, f"os.{name}")
            if both and args:
                _check(args[0], True, f"os.{name}")
            return original(path, *args, **kwargs)
        guarded.__name__ = name
        setattr(os, name, guarded)

    def guarded_os_open(path, flags, *args, **kwargs):
        _check(path, bool(flags & _WRITE_FLAGS), "os.open")
        return _os_originals["open"](path, flags, *args, **kwargs)

    _os_originals["open"] = os.open
    os.open = guarded_os_open
    if hasattr(os, "add_dll_directory"):  # scipy, cv2 and skimage register their own DLL folders at import: inside the interpreter installation only
        original_add_dll_directory = os.add_dll_directory

        def guarded_add_dll_directory(path):
            if not any(_under(_key(path), prefix) for prefix in READ_PREFIXES):
                raise PermissionError(f"os.add_dll_directory is limited to the interpreter installation in the sandbox ({os.fspath(path)}).")
            return original_add_dll_directory(path)
        os.add_dll_directory = guarded_add_dll_directory
    for name in ("listdir", "scandir"):
        guard(name, False)
    for name in ("stat", "lstat"):  # os.path.exists / getsize / isfile on an @alias path: resolved, not checked (libraries stat files everywhere)
        original_stat = _os_originals.setdefault(name, getattr(os, name))

        def aliased(path, *args, _original=original_stat, **kwargs):
            return _original(_real(path), *args, **kwargs)
        aliased.__name__ = name
        setattr(os, name, aliased)
    for name in ("remove", "unlink", "rmdir", "mkdir", "truncate", "chmod", "utime", "symlink"):
        if hasattr(os, name):
            guard(name, True)
    for name in ("rename", "replace", "link"):
        if hasattr(os, name):
            guard(name, True, both=True)


def _guard_network():
    import socket

    def refuse(*_args, **_kwargs):
        raise PermissionError("Network access is not available in the sandbox.")

    class BlockedSocket:
        def __init__(self, *args, **kwargs):
            refuse()

    socket.socket = socket.SocketType = BlockedSocket
    for name in ("create_connection", "create_server", "getaddrinfo", "gethostbyname", "gethostbyname_ex", "gethostbyaddr", "socketpair", "fromfd", "fromshare"):
        if hasattr(socket, name):
            setattr(socket, name, refuse)


def _blockwise_ogg(soundfile):
    """soundfile.write of a long OGG (Vorbis, Opus) in one call crashes libsndfile 1.2, and the sandbox process with it: OGG goes in blocks."""
    write = soundfile.write

    def blockwise(file, data, samplerate, subtype=None, endian=None, format=None, closefd=True, **kwargs):
        kind = format or (os.path.splitext(os.fspath(file))[1][1:] if isinstance(file, (str, os.PathLike)) else "")
        if str(kind).upper() not in ("OGG", "OGA", "OPUS"):
            return write(file, data, samplerate, subtype, endian, format, closefd, **kwargs)
        import numpy

        data = numpy.asarray(data)
        with soundfile.SoundFile(file, "w", samplerate, 1 if data.ndim == 1 else data.shape[1], subtype, endian, format, closefd, **kwargs) as handle:
            for start in range(0, len(data), 65536):
                handle.write(data[start:start + 65536])

    soundfile.write = blockwise


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level:
        raise ImportError("Relative imports are not available in the sandbox.")
    top = name.partition(".")[0]
    if top not in ALLOWED_MODULES:
        raise ImportError(f"'{top}' is not available in the sandbox. Available: {LIBRARY_SUMMARY}. No processes, raw network, GPU or ffmpeg: generate media with WanGP's models, cut or mix with its toolbox, download with wangp.download(url).")
    module = _real_import(name, globals, locals, fromlist, level)
    if top == "soundfile" and not getattr(sys.modules["soundfile"], "_wangp_blockwise", False):
        _blockwise_ogg(sys.modules["soundfile"])
        sys.modules["soundfile"]._wangp_blockwise = True
    return module


def _safe_builtins() -> dict:
    safe = dict(vars(builtins))
    for name in ("input", "breakpoint", "exit", "quit", "help", "copyright", "credits", "license"):
        safe.pop(name, None)
    safe["open"] = _guarded_open
    safe["__import__"] = _guarded_import
    return safe


# ------------------------------------------------------------------------------------------------------ host calls
class _StopRun(BaseException):
    """The user pressed Stop: raised at the script's next tool call, print or progress report. A BaseException, so that the script's own
    `except Exception` handlers do not swallow it; the interpreter and its variables survive."""


def _check_stop() -> None:
    if os.path.exists(os.path.join(ROOT, "tmp", ".stop")):
        raise _StopRun()


def _notify(payload: dict) -> None:
    """A one-way message to the host (live progress, recent output lines): no reply, the script continues at once."""
    _OUT.write(json.dumps({"notify": payload}, ensure_ascii=False, default=str).encode("utf-8") + b"\n")
    _OUT.flush()


class _LiveOutput(io.StringIO):
    """Captured print() output that also sends its last complete lines to the host, at most every NOTIFY_INTERVAL_S, so a long run shows what it prints."""

    NOTIFY_INTERVAL_S = 0.5

    def __init__(self):
        super().__init__()
        self._last = 0.0

    def write(self, text):
        written = super().write(text)
        if "\n" in text:
            _check_stop()
        if "\n" in text and time.monotonic() - self._last >= self.NOTIFY_INTERVAL_S:
            self._last = time.monotonic()
            value = self.getvalue()
            lines = [line[:200] for line in value[max(0, len(value) - 6000):].splitlines() if line.strip()][-20:]
            if lines:
                _notify({"op": "stdout", "lines": lines})
        return written


def _rpc(payload: dict):
    _check_stop()
    _OUT.write(json.dumps({"rpc": payload}, ensure_ascii=False).encode("utf-8") + b"\n")
    _OUT.flush()
    line = _IN.readline()
    if not line:
        raise RuntimeError("The sandbox host closed the connection.")
    reply = json.loads(line)
    if reply.get("stop"):
        raise _StopRun()
    if "error" in reply:
        raise getattr(builtins, str(reply.get("error_type", "")), RuntimeError)(reply["error"])
    return reply["value"]


def _external_references(markup: str) -> list[str]:
    import re
    import xml.etree.ElementTree as ET

    found = []
    url = re.compile(r"url\(\s*['\"]?\s*(?!#|data:)([^)'\"]+)")
    for element in ET.fromstring(markup).iter():
        for name, value in element.attrib.items():
            if name.endswith("href") and not value.startswith(("data:", "#")):
                found.append(value)
            found.extend(url.findall(value))
        if element.text:
            found.extend(url.findall(element.text))
    return found


def _build_wangp_module() -> types.ModuleType:
    module = types.ModuleType("wangp", "WanGP helpers inside the Python sandbox: path, video_frames, font, svg, publish_video, call, toolbox, inspect, generate, listdir, io, progress, out, root.")
    module.root = "."  # the sandbox folder is the working directory: relative paths, not virtual @documents paths
    module.out = "out"

    def path(ref: str) -> str:
        """Local path of a Gallery media id (visual:..., audio:..., 3d:...) or an authorized @alias/file, readable from Python afterwards.
        Files only: the sandbox folder itself is the working directory, so its content is addressed with relative paths ('out/x.png')."""
        ref = str(ref)
        if not ref.startswith("@") and not os.path.isabs(ref) and os.path.isfile(ref):  # the sandbox's own file ('out/x.png'): already readable here
            return os.path.abspath(ref)
        if ref.startswith(("@documents/sandbox", "@documents\\sandbox")) or ref in (".", "out", "tmp"):
            raise ValueError("The sandbox folder is the working directory: use relative paths such as 'out/clip.png' or 'tmp/frames/*.png' directly; wangp.path is for Gallery media ids and other authorized files.")
        value = _rpc({"op": "path", "ref": ref})
        GRANTED.add(_key(value))
        return value

    def video_frames(ref: str, start: int = 0, stop: int | None = None, step: int = 1):
        """The frames of a video or a view (@views/...) one by one, as RGB numpy arrays, numbered from its own first frame: start to stop (excluded), every step. Nothing is encoded: for a view it is the
        quick way in, as wangp.path cuts a real clip the first time."""
        import cv2

        part = _rpc({"op": "part", "ref": str(ref)})
        GRANTED.add(_key(part["path"]))
        first = part["start_frame"] + int(start)
        last = None if part["end_frame"] is None and stop is None else part["end_frame"] if stop is None else part["start_frame"] + int(stop) - 1 if part["end_frame"] is None else min(part["end_frame"], part["start_frame"] + int(stop) - 1)
        capture = cv2.VideoCapture(part["path"])
        try:
            for _ in range(first):  # read on, not seek: exact to the frame on every file
                capture.grab()
            index = first
            while (last is None or index <= last) and capture.grab():
                if (index - first) % int(step) == 0:
                    good, frame = capture.retrieve()
                    if not good:
                        return
                    yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                index += 1
        finally:
            capture.release()

    def svg(markup: str, width: int | None = None, height: int | None = None, background: str | None = None):
        """Rasterize self-contained SVG markup (no external file references) to an RGBA PIL image; width/height scale it."""
        import resvg_py
        from PIL import Image

        if references := _external_references(markup):
            raise ValueError(f"The SVG references external files ({references[0]}); embed images as data: URIs or draw them with PIL.")
        if any(value is not None and not 1 <= int(value) <= MAX_SVG_SIDE for value in (width, height)):
            raise ValueError(f"width and height must be between 1 and {MAX_SVG_SIDE} pixels.")
        import io  # the module, not the wangp.io helper defined beside this function
        image = Image.open(io.BytesIO(bytes(resvg_py.svg_to_bytes(svg_string=markup, width=width, height=height, background=background))))
        image.load()
        if max(image.size) > MAX_SVG_SIDE:
            raise ValueError(f"The drawing renders at {image.width}x{image.height}; set width or height (at most {MAX_SVG_SIDE}).")
        return image.convert("RGBA")

    font_index = {}

    def font(size: int, family: str = "sans", bold: bool = False, italic: bool = False):
        """A system TrueType font as PIL ImageFont: size in pixels and family sans, serif, mono, condensed or display (either order)."""
        from PIL import ImageFont

        if isinstance(size, str) and not size.strip().isdigit():
            size, family = family, size  # font("display", 96) reads naturally too
        if not font_index:
            for folder in CONFIG.get("font_dirs", []):
                for root_dir, _dirs, files in os.walk(folder):
                    for file in files:
                        font_index.setdefault(file.lower(), os.path.join(root_dir, file))
        candidates = CONFIG.get("fonts", {}).get(family)
        if not candidates:
            raise ValueError(f"Unknown font family {family!r}; choose one of {', '.join(CONFIG.get('fonts', {}))}.")
        style = ("bold_italic" if bold and italic else "bold" if bold else "italic" if italic else "regular")
        for name in candidates.get(style, []) + candidates.get("regular", []):
            if (found := font_index.get(name.lower())) is not None:
                GRANTED.add(_key(found))
                return ImageFont.truetype(found, int(size))
        raise ValueError(f"No {family} font file found among {candidates}; the font folders are {CONFIG.get('font_dirs', [])}.")

    def text_mesh(text: str, height: float | None = None, depth: float = 0.1, family: str = "sans", bold: bool = False, italic: bool = False, width: float | None = None, align: str = "center", pixels: int = 64):
        """Extruded 3D text as (vertices Nx3, faces Mx3, normals Mx3) numpy arrays, centred on the origin, reading along +X, facing +Z (front at
        z = depth / 2). Lines are separated by '\\n'. The block fits inside height and/or width (1 unit tall when neither is given): a label for a
        box face is text_mesh(label, height=box_h * 0.7, width=box_w * 0.9). Drawn with the sandbox's fonts and extruded from the pixels, so holes, accents and every font work
        and normals point outward; pixels = the height of one line in the drawing (64 for labels, 128 for large titles)."""
        from PIL import Image, ImageDraw

        face = font(int(pixels), family, bold=bold, italic=italic)
        spacing = int(pixels) // 4
        left, top, right, bottom = ImageDraw.Draw(Image.new("L", (1, 1))).multiline_textbbox((0, 0), str(text), font=face, align=align, spacing=spacing)
        picture = Image.new("L", (int(right - left) + 5, int(bottom - top) + 5), 0)
        ImageDraw.Draw(picture).multiline_text((2 - left, 2 - top), str(text), font=face, fill=255, align=align, spacing=spacing)
        return extrude(picture, height=height, width=width, depth=depth)

    def extrude_parts(image, colors: int = 2, height: float | None = None, width: float | None = None, depth: float = 0.1, max_pixels: int = 400, min_share: float = 0.01):
        """A coloured flat design (a logo, an icon, a flag) as one extruded part per colour: a list of (vertices, faces, normals, rgb) with rgb in
        0..1, sharing one scale and centre so the parts fit together. image is a PIL image, an image path or Gallery id; transparent pixels are
        empty, or, without transparency, the background colour found in the corners. The design is reduced to `colors` flat colours (gradients
        fall to their nearest colour); parts smaller than min_share of the design are dropped."""
        import numpy as np
        from PIL import Image

        picture = Image.open(path(image)) if isinstance(image, str) else image
        picture = picture.convert("RGBA")
        if max(picture.size) > max_pixels:
            picture = picture.resize((max(1, round(picture.width * max_pixels / max(picture.size))), max(1, round(picture.height * max_pixels / max(picture.size)))), Image.LANCZOS)
        pixels = np.asarray(picture)
        rgb = pixels[..., :3].astype(np.int32)
        solid = pixels[..., 3] >= 128
        if solid.all():  # no transparency: the corners' most common colour is the background
            corners = [tuple(rgb[y, x]) for y in (0, -1) for x in (0, -1)]
            background = np.array(max(set(corners), key=corners.count))
            solid = np.abs(rgb - background).sum(axis=-1) > 60
        if not solid.any():
            raise ValueError("The design has no visible pixels to extrude.")
        palette = Image.fromarray(rgb[solid].reshape(-1, 1, 3).astype(np.uint8)).quantize(colors=int(colors), method=Image.Quantize.MEDIANCUT)
        labels = np.full(solid.shape, -1)
        labels[solid] = np.asarray(palette).reshape(-1)
        swatches = np.asarray(palette.getpalette()[: 3 * int(colors)]).reshape(-1, 3) / 255.0
        parts = []
        for index in range(len(swatches)):
            mask = labels == index
            if mask.sum() >= min_share * solid.sum():
                vertices, faces, normals = extrude(mask, height=height, width=width, depth=depth)
                parts.append((vertices, faces, normals, tuple(float(value) for value in swatches[index])))
        return parts

    def extrude(shape, height: float | None = None, width: float | None = None, depth: float = 0.1, threshold: int = 128, max_pixels: int = 400):
        """Extrude a flat shape into 3D, as (vertices Nx3, faces Mx3, normals Mx3) numpy arrays centred on the origin and facing +Z: shape is a PIL
        image, an image path or Gallery id, or a 2D boolean array; its bright pixels (or opaque ones, for an image with transparency) become solid.
        For a dark logo on white, pass PIL.ImageOps.invert(image.convert('L')) or a mask of your own. Fits inside height and/or width (1 unit tall
        when neither is given); the drawing is reduced to max_pixels on its long side first, which is plenty for logos and keeps the mesh light."""
        import numpy as np
        from PIL import Image

        if isinstance(shape, str):
            shape = Image.open(path(shape))
        if isinstance(shape, Image.Image):
            shape = shape.getchannel("A") if "A" in shape.getbands() else shape.convert("L")
            if max(shape.size) > max_pixels:
                shape = shape.resize((max(1, round(shape.width * max_pixels / max(shape.size))), max(1, round(shape.height * max_pixels / max(shape.size)))), Image.LANCZOS)
            mask = np.asarray(shape) >= threshold
        else:
            mask = np.asarray(shape, dtype=bool)
        if not mask.any():
            raise ValueError("Nothing to extrude: no pixel of the shape is above the threshold (a dark shape on white needs PIL.ImageOps.invert first).")
        rows, cols = mask.shape
        scale = min([limit / size for limit, size in ((height, rows), (width, cols)) if limit is not None] or [1.0 / rows])  # units per pixel
        z0, z1 = -depth / 2, depth / 2
        X = lambda px: (px - cols / 2) * scale
        Y = lambda py: (rows / 2 - py) * scale
        vertices, faces, normals = [], [], []

        def quad(a, b, c, d, normal):  # counter-clockwise seen from the side the normal points to
            start = len(vertices)
            vertices.extend((a, b, c, d))
            faces.extend(((start, start + 1, start + 2), (start, start + 2, start + 3)))
            normals.extend((normal, normal))

        def runs(line):
            edges = np.flatnonzero(np.diff(np.concatenate(([0], line.astype(np.int8), [0]))))
            return zip(edges[::2], edges[1::2])

        opened: dict = {}  # caps: the filled runs of each row, merged downwards into rectangles
        for y in range(rows + 1):
            current = set(runs(mask[y])) if y < rows else set()
            for run in [run for run in opened if run not in current]:
                (x0, x1), y0 = run, opened.pop(run)
                quad((X(x0), Y(y), z1), (X(x1), Y(y), z1), (X(x1), Y(y0), z1), (X(x0), Y(y0), z1), (0, 0, 1))
                quad((X(x0), Y(y0), z0), (X(x1), Y(y0), z0), (X(x1), Y(y), z0), (X(x0), Y(y), z0), (0, 0, -1))
            for run in current:
                opened.setdefault(run, y)
        padded = np.pad(mask, 1)
        filled = padded[1:-1, 1:-1]
        for y, line in enumerate(filled & ~padded[:-2, 1:-1]):  # walls along the pixel outline, merged along each line
            for a, b in runs(line):
                quad((X(a), Y(y), z1), (X(b), Y(y), z1), (X(b), Y(y), z0), (X(a), Y(y), z0), (0, 1, 0))
        for y, line in enumerate(filled & ~padded[2:, 1:-1]):
            for a, b in runs(line):
                quad((X(a), Y(y + 1), z0), (X(b), Y(y + 1), z0), (X(b), Y(y + 1), z1), (X(a), Y(y + 1), z1), (0, -1, 0))
        for x, line in enumerate((filled & ~padded[1:-1, :-2]).T):
            for a, b in runs(line):
                quad((X(x), Y(b), z1), (X(x), Y(a), z1), (X(x), Y(a), z0), (X(x), Y(b), z0), (-1, 0, 0))
        for x, line in enumerate((filled & ~padded[1:-1, 2:]).T):
            for a, b in runs(line):
                quad((X(x + 1), Y(b), z0), (X(x + 1), Y(a), z0), (X(x + 1), Y(a), z1), (X(x + 1), Y(b), z1), (1, 0, 0))
        return np.asarray(vertices, np.float32), np.asarray(faces, np.int64), np.asarray(normals, np.float32)

    def publish_video(frames, fps: float = 24, name: str = "clip", audio=None) -> dict:
        """Encode frames (a glob, a list of image paths, or PIL images / numpy arrays of one size) into a Gallery video clip. audio (a video or
        audio media id or path) gives it a soundtrack: the sound of a source video whose frames were redrawn, music, narration."""
        from PIL import Image

        if isinstance(frames, str):
            paths = sorted(glob.glob(frames, recursive=True))
        else:
            paths = []
            folder = None
            for index, frame in enumerate(frames):
                if isinstance(frame, (str, os.PathLike)):
                    paths.append(os.fspath(frame))
                    continue
                if folder is None:
                    folder = os.path.join(ROOT, "tmp", f"frames_{int(time.time() * 1000)}")
                    os.makedirs(folder, exist_ok=True)
                image = frame if isinstance(frame, Image.Image) else Image.fromarray(frame)
                target = os.path.join(folder, f"{index:06d}.png")
                image.convert("RGB").save(target)
                paths.append(target)
        if not paths:
            raise ValueError("publish_video needs at least one frame.")
        item = _rpc({"op": "video", "frames": [os.path.abspath(path) for path in paths], "fps": float(fps), "name": str(name)})
        if audio is None or not item.get("media_id"):
            return item
        sound = toolbox("extract_audio", media_id=str(audio), name=f"{name}_audio")  # a video's soundtrack, or an audio file as it is
        if sound.get("status") == "error" or not sound.get("media_id"):
            return {**item, "audio_error": sound.get("error") or "no audio could be taken from the source"}
        return toolbox("remux_media", video_id=item["media_id"], audio_ids=[sound["media_id"]], mode="replace", name=str(name))

    def call(tool: str, action: str | None = None, arguments: dict | None = None, **kwargs) -> dict:
        """Run one of WanGP's tools exactly as Deepy would and return its result dict: wangp_toolbox (media utilities and inspections), wangp_io
        (file actions), wangp_generate (settings=), wangp_postprocess, templates, model search, the Gallery listing. Short names work ('io',
        'toolbox'); call('io', 'move', {'source': ..., 'destination': ...}) equals call('wangp_io', action='move', arguments={...}) and
        call('io', 'move', source=..., destination=...). Relative sandbox paths inside arguments ('out/a.png') are accepted. Each call costs
        what the tool costs; time spent inside WanGP's tools does not count toward the run's timeout_s."""
        tool = str(tool) if str(tool).startswith("wangp_") else "wangp_" + str(tool)
        if isinstance(action, dict):  # the whole tool payload: call('io', {'action': 'info', 'arguments': {...}})
            payload = {**action, **kwargs}
        elif action is not None:
            payload = {"action": str(action), "arguments": dict(arguments if arguments is not None else kwargs)}
        else:
            payload = dict(kwargs) if arguments is None else {**kwargs, "arguments": arguments}
        return _rpc({"op": "tool", "tool": tool, "arguments": payload})

    def toolbox(action: str, **arguments) -> dict:
        """wangp_toolbox action: toolbox('inspect_media', media_id='@documents/photos/a.jpg', question='...'), toolbox('resize_crop', ...)."""
        if set(arguments) == {"arguments"} and isinstance(arguments["arguments"], dict):  # toolbox(action, arguments={...}) is the same call
            arguments = arguments["arguments"]
        return call("wangp_toolbox", action=str(action), arguments=arguments)

    def inspect(media=None, question: str = "", **arguments) -> dict:
        """Ask the visual inspector a question about an image, a video frame or a 3D asset (a Gallery media id, an authorized @alias path or a sandbox file).
        To compare several visuals in one answer, omit media and pass media_inputs=[{"media_id": ref}, {"media_id": video, "time_s": 2.0}, ...];
        with media given, media_inputs items without a media_id are frames of it: inspect(clip, q, media_inputs=[{"time_s": 1}, {"time_s": 4}])."""
        if media is not None and arguments.get("media_inputs"):
            arguments["media_inputs"] = [{"media_id": str(media), **item} if "media_id" not in item else item for item in arguments["media_inputs"]]
        elif media is not None:
            arguments["media_ids" if isinstance(media, (list, tuple)) else "media_id"] = [str(item) for item in media] if isinstance(media, (list, tuple)) else str(media)
        return toolbox("inspect_media", question=str(question), **arguments)

    def generate(settings: dict, **arguments) -> dict:
        """wangp_generate with a settings object (a template name plus overrides, or full settings); waits for the result like Deepy does."""
        return call("wangp_generate", settings=dict(settings), **arguments)

    def web(action: str, arguments: dict | None = None, **kwargs) -> dict:
        """wangp_web_search from a script, when the user enabled web search: search, news, images, videos, open, download, screenshot, cookies (no clear), lora_search."""
        return call("wangp_web_search", action=str(action), arguments={**(arguments or {}), **kwargs})

    def download(url: str, **arguments) -> dict:
        """Download a page, image, video or file with wangp_web_search (same guards as Deepy's own downloads): the result's path (@documents/web/...)
        opens with open() or wangp.path(). Pass the address from a variable rather than retyping it."""
        return web("download", url=str(url), **arguments)

    def listdir(ref: str, pattern: str = "*", recursive: bool = False) -> list:
        """Files of an authorized folder as dicts: path (@alias path, for tools and open()), file (local path, for any library such as cv2),
        name, bytes, modified (epoch seconds), kind (image, video, audio, 3d, other) and type 'file'. pattern filters names, recursive walks subfolders."""
        entries = _rpc({"op": "list", "ref": _tool_path(ref), "pattern": str(pattern), "recursive": bool(recursive)})
        for entry in entries:
            key = os.path.normcase(os.path.realpath(entry["file"]))
            (READ_PREFIXES.append if entry["type"] == "dir" else GRANTED.add)(key)
            _ALIAS_ROOTS.add(os.path.dirname(key))
        return entries

    def io(action: str, arguments: dict | None = None, **kwargs) -> dict:
        """wangp_io file actions from a script: list, info, read_text, search_text, mkdir, copy, move, zip (no delete). io('move', source='@test/a.png', destination='@test/dups/a.png') or io('move', {...})."""
        arguments = {**(arguments or {}), **kwargs}
        return call("wangp_io", action=str(action), arguments={key: _tool_path(value, new=key == "destination") if key in ("path", "source", "destination") else value for key, value in arguments.items()})

    def plugins(plugin: str | None = None) -> dict:
        """The toolboxes of enabled plugins: plugins() lists them, plugins('media_flow') lists that plugin's actions with their call signatures."""
        return _rpc({"op": "tool", "tool": "wangp_plugins", "arguments": {} if plugin is None else {"plugin": str(plugin)}})

    def plugin(plugin: str, action: str, arguments: dict | None = None, **kwargs) -> dict:
        """Run a plugin action and wait for its result: plugin('media_flow', 'run', process='refocus', source='@documents/clip.mp4')."""
        return _rpc({"op": "tool", "tool": "wangp_plugins", "arguments": {"plugin": str(plugin), "action": str(action), "arguments": {**(arguments or {}), **kwargs}}})

    progress_state = {"sent": 0.0}

    def progress(done, total=None, text: str = "") -> None:
        """Report a loop's progress to the user: wangp.progress(i, len(items), name) as item i starts shows a progress bar from the first item, with the item
        running and the time left; wangp.progress(len(items), len(items)) after the loop completes it."""
        _check_stop()
        if isinstance(done, str):  # a status alone: wangp.progress("Inspecting shot 3")
            done, text = 0, done
        done, total = int(done), None if total is None else int(total)
        finished = total is not None and done >= total
        if finished or time.monotonic() - progress_state["sent"] >= 0.2:
            progress_state["sent"] = time.monotonic()
            _notify({"op": "progress", "done": done, "total": total, "text": str(text)[:120]})

    def app_rpc(action: str, **arguments):
        return _rpc({"op": "app", "action": action, "arguments": arguments})

    def app_new(name: str, template: str = "three-scene", description: str = "") -> dict:
        """Create the web app apps/<slug of name> from a template: three-scene, three-game, canvas-2d, phaser-game, physics-toy, audio-visualizer, blank."""
        return app_rpc("app_new", name=str(name), template=str(template), description=str(description))

    def app_check(app: str, steps: list | None = None, record: dict | None = None, viewport: str = "desktop") -> dict:
        """Run the web app headless: findings, frame rate, state probe per step, screenshots, and a motion strip with record={'seconds': 3, 'fps': 4}."""
        return app_rpc("app_check", app=str(app), steps=steps, record=record, viewport=str(viewport))

    def app_logs(app: str, since: float = 0.0) -> dict:
        """What the user's browser reported while the app was played: console lines, errors, frame rate."""
        return app_rpc("app_logs", app=str(app), since=float(since))

    def app_publish(app: str, note: str = "") -> dict:
        """Publish the web app to the Apps gallery as a new version."""
        return app_rpc("app_publish", app=str(app), note=str(note))

    def import_asset(media: str, app: str, path: str, max_size: int | None = None, transparent: bool = False) -> dict:
        """Copy a Gallery media id, @alias path or sandbox file into apps/<app>/<path>, converted to the format of path's extension (.webp/.png/.jpg
        resized to max_size pixels on the longer side, transparent=True cutting the subject out of its background, .ogg/.mp3/.m4a/.wav, .mp4/.webm,
        .glb): returns {file, bytes, size}."""
        return app_rpc("import_asset", media=str(media), app=str(app), path=str(path), max_size=None if max_size is None else int(max_size), transparent=bool(transparent))

    def progression(path: str) -> dict:
        """Search every order of play of a story, quest or puzzle written as facts and actions (apps/<app>/assets/story.json): shortest path to
        each ending, endings never reached, dead ends, states that lose the main ending, facts never needed, actions never possible."""
        return app_rpc("progression", path=str(path))

    module.app = types.SimpleNamespace(new=app_new, check=app_check, logs=app_logs, publish=app_publish, import_asset=import_asset, progression=progression)
    module.video_frames = video_frames
    module.path, module.svg, module.font, module.text_mesh, module.extrude, module.extrude_parts, module.publish_video = path, svg, font, text_mesh, extrude, extrude_parts, publish_video
    module.call, module.toolbox, module.inspect, module.generate, module.listdir, module.progress, module.io = call, toolbox, inspect, generate, listdir, progress, io
    module.web, module.download, module.plugins, module.plugin = web, download, plugins, plugin
    module.__all__ = ["path", "video_frames", "svg", "font", "text_mesh", "extrude", "extrude_parts", "publish_video", "call", "toolbox", "inspect", "generate", "listdir", "io", "web", "download", "plugins", "plugin", "progress", "app", "out", "root"]
    return module


# ------------------------------------------------------------------------------------------------------ execution
def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    head, tail = limit * 3 // 4, limit // 4
    return f"{text[:head]}\n... [{len(text) - head - tail:,} characters omitted: print counts, summaries or the first lines, not whole lists] ...\n{text[-tail:]}"


_TOOL_NAMES = {"inspect_media": "wangp.inspect(media, question)", "generate": "wangp.generate(settings)", "wangp_generate": "wangp.generate(settings)", "listdir": "wangp.listdir(folder)",
               "download": "wangp.download(url)", "wangp_web_search": "wangp.web(action, ...)", "web_search": "wangp.web(action, ...)",
               "import_asset": "wangp.app.import_asset(media, app, path, max_size=None, transparent=False)", "app_check": "wangp.app.check(app, steps)",
               "app_new": "wangp.app.new(name, template)", "app_publish": "wangp.app.publish(app)"}


def _format_error(exc: BaseException) -> tuple[str, list[str]]:
    message = f"{type(exc).__name__}: {exc}"
    name = getattr(exc, "name", None) if isinstance(exc, NameError) else None
    if name in _TOOL_NAMES:
        message += f". WanGP's tools are reached through the wangp module here: {_TOOL_NAMES[name]}"
    elif name and (name.startswith("wangp_") or name in ("view_media", "render_3d_views", "transcribe_media", "inspect_audio", "check_lip_sync", "detect_video_cuts", "plan_video_windows", "check_video_edit", "extract_audio", "remux_media", "merge_videos", "resize_crop")):
        message += f". WanGP's tools are reached through the wangp module here: wangp.toolbox({name.removeprefix('wangp_')!r}, ...) or wangp.call({name!r}, ...)"
    elif not isinstance(exc, PermissionError) and ("'@" in message or '"@' in message):  # open() knows @alias paths; libraries opening files in C do not
        message += ". Libraries that open files themselves (soundfile, cv2, librosa) need the local path: pass wangp.path('@...') instead of the @alias path"
    if isinstance(exc, SyntaxError):
        return message, [f"line {exc.lineno}: {(exc.text or '').strip()}"] if exc.lineno else []
    frames = traceback.extract_tb(exc.__traceback__)
    user = [frame for frame in frames if frame.filename.startswith("<deepy")]
    shown = (user or frames)[-4:]
    return message, [f"line {frame.lineno}: {frame.line}" if frame.filename.startswith("<deepy") else f"{os.path.basename(frame.filename)}:{frame.lineno} in {frame.name}" for frame in shown]


_RUNS = itertools.count(1)


def run_code(code: str, namespace: dict) -> dict:
    out, err = _LiveOutput(), io.StringIO()
    saved = sys.stdout, sys.stderr, sys.stdin
    sys.stdout, sys.stderr, sys.stdin = out, err, io.StringIO()
    result = {"status": "done"}
    started = time.perf_counter()
    name = f"<deepy-{next(_RUNS)}>"  # one name per run: functions defined in earlier runs keep their own source (inspect.getsource, tracebacks)
    linecache.cache[name] = (len(code), None, code.splitlines(keepends=True), name)
    try:
        tree = ast.parse(code, name, "exec")
        trailing = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
        exec(compile(tree, name, "exec"), namespace)
        if trailing is not None:
            value = eval(compile(ast.Expression(trailing.value), name, "eval"), namespace)
            if value is not None:
                result["result"] = _truncate(repr(value), CONFIG.get("max_result", 2000))
    except _StopRun:
        result.update(status="interrupted", error='The user pressed Stop: the script stopped at its next tool call, print or progress report. This is not an error of the script or of the tools. Its variables are kept in the interpreter, so the results gathered so far (lists, dicts) can be read or saved in the next run; files are kept too.')
    except SystemExit:
        result.update(status="error", error="SystemExit: exit() is not available in the sandbox; return normally.")
    except BaseException as exc:  # the assistant's errors, reported, never fatal for the worker
        message, lines = _format_error(exc)
        result.update(status="error", error=message, traceback=lines)
    finally:
        sys.stdout, sys.stderr, sys.stdin = saved
    result["elapsed_s"] = round(time.perf_counter() - started, 3)
    for name, stream in (("stdout", out), ("stderr", err)):
        if text := stream.getvalue():
            result[name] = _truncate(text, CONFIG.get("max_output", 8000))
    return result


def facts() -> dict:
    from importlib import metadata

    versions = {}
    for name, package in (("numpy", "numpy"), ("PIL", "pillow"), ("scipy", "scipy"), ("skimage", "scikit-image"), ("cv2", "opencv-python"), ("matplotlib", "matplotlib"), ("soundfile", "soundfile"), ("sympy", "sympy"), ("pandas", "pandas"), ("pypdf", "pypdf")):
        try:
            versions[name] = metadata.version(package)
        except metadata.PackageNotFoundError:
            pass
    return {"python": sys.version.split()[0], "libraries": versions, "pid": os.getpid()}


def _upright_open(open_image):
    """PIL.Image.open showing images as the Gallery does: a photo's EXIF orientation is applied to its pixels (Pillow's own lookup,
    which decodes a PNG at once to find metadata after its image data)."""
    import PIL.ImageOps

    def upright_open(fp, *args, **kwargs):
        image = open_image(fp, *args, **kwargs)
        if image.getexif().get(0x0112, 1) != 1:
            PIL.ImageOps.exif_transpose(image, in_place=True)
        return image

    return upright_open


def main() -> None:
    global ROOT, ROOT_KEY, CONFIG, READ_PREFIXES
    ROOT = os.path.realpath(sys.argv[1])
    CONFIG = json.loads(sys.argv[2])
    os.chdir(ROOT)
    ROOT_KEY = os.path.normcase(ROOT)
    READ_PREFIXES = [os.path.normcase(os.path.realpath(prefix)) for prefix in {sys.prefix, sys.base_prefix, sys.exec_prefix, *CONFIG.get("read_dirs", []), *CONFIG.get("font_dirs", [])}]
    for folder in ("out", "tmp"):
        os.makedirs(os.path.join(ROOT, folder), exist_ok=True)
    import numpy  # noqa: F401  pre-imported so that their own imports happen before the guards and the first run stays fast
    import PIL.Image
    import warnings
    PIL.Image.open = _upright_open(PIL.Image.open)
    for category in (DeprecationWarning, PendingDeprecationWarning, FutureWarning):  # library deprecations are noise in the assistant's context
        warnings.simplefilter("ignore", category)

    builtins.open = io.open = _guarded_open
    io.FileIO = _GuardedFileIO
    _guard_os()
    _guard_network()
    wangp = _build_wangp_module()
    sys.modules["wangp"] = wangp
    if hasattr(wangp, "app"):
        sys.modules["wangp.app"] = wangp.app  # import wangp.app / from wangp.app import check, as scripts write them
    sys.argv = ["<deepy>"]
    namespace = {"__name__": "__main__", "__builtins__": _safe_builtins(), "wangp": wangp}
    while line := _IN.readline():
        request = json.loads(line)
        if request.get("op") == "ping":
            response = {"status": "done", **facts()}
        elif request.get("op") == "run":
            response = run_code(str(request.get("code", "")), namespace)
        else:
            response = {"status": "error", "error": f"Unknown worker request: {request.get('op')!r}"}
        response["id"] = request.get("id")
        _OUT.write(json.dumps(response, ensure_ascii=False, default=str).encode("utf-8") + b"\n")
        _OUT.flush()


if __name__ == "__main__":
    main()
