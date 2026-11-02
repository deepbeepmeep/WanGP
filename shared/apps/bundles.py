# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Web app bundles.

A bundle is a folder: app.json (the manifest), an HTML entry, the app's ES modules and its assets. Bare imports resolve through the import map
WanGP injects (vendored libraries under /apps/lib/), everything else is relative to the bundle. Development bundles live in a Deepy session's
sandbox (apps/<slug>/); a published app is a .wgpapp file (a zip of the bundle with thumbnail.png and source.json) in the outputs, with its
thumbnail beside it (<stem>.png) for the Apps gallery."""
from __future__ import annotations

import hashlib
import html.parser
import json
import os
import re
import shutil
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any

PACKAGE = Path(__file__).resolve().parent
LIB_DIR = PACKAGE / "lib"  # vendored libraries, served at /apps/lib/
RUNTIME_SCRIPT = PACKAGE / "runtime" / "wangp-app.js"  # injected into every app page, served at /apps/lib/wangp-app.js
TEMPLATES_DIR = PACKAGE / "templates"
DATA_DIR = PACKAGE.parents[1] / "settings" / "apps"  # unpacked packages (cache/) and the saved data of published apps (data/)
MANIFEST = "app.json"
EXTENSION = ".wgpapp"
MAX_BUNDLE_BYTES = 200 << 20
MAX_FILES = 2000
TEMPLATES = {
    "three-scene": "a three.js scene with orbit controls, lights and an animation loop",
    "three-game": "a three.js game: keyboard movement, a score, a restart key",
    "canvas-2d": "a 2D canvas animation or game loop without a library",
    "phaser-game": "a 2D game with Phaser (scenes, arcade physics, input, sprites)",
    "physics-toy": "three.js rendering with cannon-es physics: objects to throw and stack",
    "audio-visualizer": "Web Audio analysis drawn on a canvas, started by a click",
    "blank": "an empty page with the runtime hooks",
}
_SLUG = re.compile(r"[^a-z0-9]+")
_IMPORTS = re.compile(r"""(?:\bimport\s*(?:[\w*{}\s,$]+?\s*from\s*)?|\bexport\s*(?:[\w*{}\s,$]+?\s*)from\s*|\bimport\s*\(\s*)(['"])([^'"\n]+)\1""")
_EXTERNAL = re.compile(r"""(['"`])((?:https?:)?//[^'"`\s]+)\1""")


def slugify(name: str) -> str:
    slug = _SLUG.sub("-", str(name or "").lower()).strip("-")[:48].strip("-")
    if not slug:
        raise ValueError("An app needs a name with letters or digits.")
    return slug


def import_map() -> dict[str, Any]:
    return json.loads((LIB_DIR / "importmap.json").read_text(encoding="utf-8"))


def read_manifest(folder: Path) -> dict[str, Any]:
    """The manifest with its defaults; raises ValueError for a missing or malformed app.json."""
    path = Path(folder) / MANIFEST
    if not path.is_file():
        raise ValueError(f"{MANIFEST} is missing in {Path(folder).name}.")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise ValueError(f"{MANIFEST} is not valid JSON: {error}") from None
    if not isinstance(manifest, dict) or not str(manifest.get("name") or "").strip():
        raise ValueError(f"{MANIFEST} must be an object with a name.")
    defaults = {"kind": "web", "entry": "index.html", "description": "", "version": 1, "controls": {}, "canvas": "responsive", "audio": False, "pointer_lock": False, "gamepad": False}
    manifest = {**defaults, **manifest}
    if not isinstance(manifest["controls"], dict):
        raise ValueError(f"{MANIFEST}: controls must map keys to actions, e.g. {{\"WASD\": \"move\"}}.")
    entry = str(manifest["entry"])
    if entry.startswith(("/", "\\")) or ".." in Path(entry).parts:
        raise ValueError(f"{MANIFEST}: entry must be a file inside the app folder.")
    return manifest


def write_manifest(folder: Path, manifest: dict[str, Any]) -> None:
    (Path(folder) / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def create(apps_root: Path, name: str, template: str, description: str = "") -> Path:
    """A new bundle apps_root/<slug> copied from a template, its manifest named."""
    if template not in TEMPLATES:
        raise ValueError(f"Unknown template {template!r}; choose one of: {', '.join(TEMPLATES)}.")
    folder = Path(apps_root) / slugify(name)
    if folder.exists():
        raise ValueError(f"An app named {folder.name} already exists: edit its files, or pick another name.")
    shutil.copytree(TEMPLATES_DIR / template, folder)
    manifest = read_manifest(folder)
    manifest.update(name=str(name).strip(), id=uuid.uuid4().hex[:12], description=str(description or manifest["description"]), version=1)
    write_manifest(folder, manifest)
    return folder


def bundle_files(folder: Path) -> list[Path]:
    """The files of a bundle: everything but hidden files and folders."""
    folder = Path(folder)
    files = []
    for root, dirs, names in os.walk(folder):
        dirs[:] = sorted(name for name in dirs if not name.startswith((".", "__")))
        files.extend(Path(root) / name for name in sorted(names) if not name.startswith("."))
    return files


def signature(folder: Path) -> str:
    """Changes with any file of the bundle (the preview reloads when it does)."""
    digest = hashlib.sha1()
    for path in bundle_files(folder):
        stat = path.stat()
        digest.update(f"{path.relative_to(folder).as_posix()}:{stat.st_size}:{stat.st_mtime_ns};".encode())
    return digest.hexdigest()[:12]


class _References(html.parser.HTMLParser):
    """URLs an HTML page loads (scripts, styles, images, media) and the text of its inline module scripts."""

    def __init__(self):
        super().__init__()
        self.urls: list[str] = []
        self.inline_modules: list[str] = []
        self._in_module = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for key in ("src", "href", "poster"):
            value = attrs.get(key)
            if value and not (tag == "a" and key == "href"):
                self.urls.append(value)
        self._in_module = tag == "script" and attrs.get("type") == "module" and not attrs.get("src")

    def handle_endtag(self, tag):
        if tag == "script":
            self._in_module = False

    def handle_data(self, data):
        if self._in_module:
            self.inline_modules.append(data)


def _resolve(reference: str, importer: Path, folder: Path, imports: dict[str, str], module: bool) -> tuple[str, str]:
    """('ok' | 'error', message) for a module specifier (module=True) or a URL an HTML page loads."""
    if reference.startswith(("data:", "blob:", "#")):
        return "ok", ""
    if reference.startswith(("http:", "https:", "//")):
        return "error", f"{reference} is on the internet: apps load nothing from outside WanGP. Use the import map names ({', '.join(sorted(imports))}) or files in the app folder."
    if reference.startswith("/apps/lib/"):
        known = reference == "/apps/lib/wangp-app.js" or (LIB_DIR / reference[len("/apps/lib/"):]).is_file()
        return ("ok", "") if known else ("error", f"{reference} is not one of the vendored libraries.")
    if reference.startswith("/"):
        return "error", f"{reference} is an absolute path: an app is served under its own address, so use a path relative to the app folder (assets/hero.png)."
    if module and not reference.startswith(("./", "../")):  # a bare specifier: only the import map resolves it
        if reference in imports or any(key.endswith("/") and reference.startswith(key) and (LIB_DIR / (imports[key][len("/apps/lib/"):] + reference[len(key):])).is_file() for key in imports):
            return "ok", ""
        return "error", f"'{reference}' is not in the import map (a relative module needs ./). Available: {', '.join(sorted(imports))}."
    clean = reference.split("?")[0].split("#")[0]
    if (importer.parent / clean).is_file() or (folder / clean).is_file():
        return "ok", ""
    return "error", f"{reference} does not exist in the app folder."


def static_check(folder: Path) -> dict[str, Any]:
    """Findings about a bundle without running it: the manifest, the module graph and asset references, the size."""
    folder = Path(folder)
    findings: list[dict[str, str]] = []
    try:
        manifest = read_manifest(folder)
    except ValueError as error:
        return {"manifest": None, "findings": [{"level": "error", "file": MANIFEST, "message": str(error)}], "files": 0, "bytes": 0}
    files = bundle_files(folder)
    size = sum(path.stat().st_size for path in files)
    if size > MAX_BUNDLE_BYTES or len(files) > MAX_FILES:
        findings.append({"level": "error", "file": "", "message": f"The app holds {len(files)} files, {size / 1048576:.1f} MB: the limit is {MAX_FILES} files and {MAX_BUNDLE_BYTES >> 20} MB. Resize textures and compress audio (wangp.app.import_asset does both)."})
    entry = folder / manifest["entry"]
    if not entry.is_file():
        findings.append({"level": "error", "file": manifest["entry"], "message": "The entry page named in app.json does not exist."})
    imports = import_map()["imports"]

    def report(level, path, message):
        if not any(item["message"] == message and item["file"] == path for item in findings):
            findings.append({"level": level, "file": path, "message": message})

    for path in files:
        relative = path.relative_to(folder).as_posix()
        suffix = path.suffix.lower()
        if suffix not in {".html", ".htm", ".js", ".mjs", ".css"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        modules = [text] if suffix in {".js", ".mjs"} else []
        if suffix in {".html", ".htm"}:
            parser = _References()
            parser.feed(text)
            modules.extend(parser.inline_modules)
            for url in parser.urls:
                level, message = _resolve(url, path, folder, imports, module=False)
                if level != "ok":
                    report(level, relative, message)
        for module in modules:
            for match in _IMPORTS.finditer(module):
                level, message = _resolve(match.group(2), path, folder, imports, module=True)
                if level != "ok":
                    report(level, relative, message)
        for match in _EXTERNAL.finditer(text):
            url = match.group(2)
            if not url.startswith(("http://www.w3.org/", "https://www.w3.org/")) and not any(url in item["message"] for item in findings):  # SVG/XHTML namespace URIs; imports already reported
                report("warning", relative, f"{url}: apps cannot reach the internet, so a request to it will fail.")
    return {"manifest": manifest, "findings": findings, "files": len(files), "bytes": size}


# ---------------------------------------------------------------------------------------------------- packages
def pack(folder: Path, target: Path, thumbnail: Path | None, source: dict[str, Any]) -> Path:
    """Write the bundle as a .wgpapp at target (with thumbnail.png and source.json inside) and its thumbnail beside it as <stem>.png."""
    folder, target = Path(folder), Path(target)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in bundle_files(folder):
            archive.write(path, path.relative_to(folder).as_posix())
        if thumbnail is not None:
            archive.write(thumbnail, "thumbnail.png")
        archive.writestr("source.json", json.dumps({**source, "packed": time.strftime("%Y-%m-%d %H:%M:%S")}, ensure_ascii=False, indent=2))
    if thumbnail is not None:
        shutil.copyfile(thumbnail, target.with_suffix(".png"))
    return target


_UNPACKED: dict[tuple[str, int, int], Path] = {}


def unpack(package: Path) -> Path:
    """The folder of a .wgpapp, extracted once into the cache (keyed by its content), read only for the app."""
    package = Path(package)
    stat = package.stat()
    known = _UNPACKED.get((str(package), stat.st_size, stat.st_mtime_ns))
    if known is not None and (known / MANIFEST).is_file():
        return known
    folder = _unpack(package)
    _UNPACKED[(str(package), stat.st_size, stat.st_mtime_ns)] = folder
    return folder


def _unpack(package: Path) -> Path:
    digest = hashlib.sha1()
    with open(package, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    folder = DATA_DIR / "cache" / digest.hexdigest()[:16]
    if (folder / MANIFEST).is_file():
        return folder
    staging = folder.with_name(folder.name + ".part")
    shutil.rmtree(staging, ignore_errors=True)
    with zipfile.ZipFile(package) as archive:
        members = archive.infolist()
        if sum(member.file_size for member in members) > MAX_BUNDLE_BYTES or len(members) > MAX_FILES + 2:
            raise ValueError(f"{package.name} is larger than an app may be ({MAX_BUNDLE_BYTES >> 20} MB, {MAX_FILES} files).")
        for member in members:
            name = Path(member.filename)
            if member.filename.startswith(("/", "\\")) or name.is_absolute() or ".." in name.parts or ":" in member.filename:
                raise ValueError(f"{package.name} contains an unsafe path: {member.filename}")
        staging.mkdir(parents=True)
        archive.extractall(staging)
    read_manifest(staging)
    staging.replace(folder)
    return folder


def package_manifest(package: Path) -> dict[str, Any]:
    """The manifest and source record of a .wgpapp, read without extracting it."""
    with zipfile.ZipFile(package) as archive:
        manifest = json.loads(archive.read(MANIFEST))
        source = json.loads(archive.read("source.json")) if "source.json" in archive.namelist() else {}
    return {**manifest, "source": source}
