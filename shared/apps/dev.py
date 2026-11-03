# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Deepy's app workshop: the web-app actions of wangp_python (app_new, app_open, app_check, app_logs, app_publish) and the wangp.app helpers of its
scripts. Apps are developed in the session's sandbox (apps/<slug>/, edited with wangp_io's text actions), checked headless, previewed in the
chat and published to the Apps gallery as .wgpapp packages."""
from __future__ import annotations

import json
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from shared.apps import bundles, progression, server

ASSET_IMAGE = {".png", ".jpg", ".jpeg", ".webp"}
ASSET_AUDIO = {".ogg": ["-c:a", "libvorbis", "-q:a", "5"], ".mp3": ["-c:a", "libmp3lame", "-q:a", "3"], ".m4a": ["-c:a", "aac", "-b:a", "160k"], ".wav": ["-c:a", "pcm_s16le"]}
ASSET_VIDEO = {".mp4": ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", "-movflags", "+faststart", "-c:a", "aac"], ".webm": ["-c:v", "libvpx-vp9", "-crf", "34", "-b:v", "0", "-c:a", "libopus"]}
ASSET_MESH = {".glb"}
PACKAGE_ONLY = {"thumbnail.png", "source.json", "wangp_settings.json"}  # added to a bundle when it is published, not part of the app

RULES = [
    "A web app is a folder apps/<name>/ in the sandbox: app.json (name, description, controls shown to the player such as {\"WASD\": \"move\", \"Space\": \"jump\"}, canvas), index.html and ES modules (main.js) plus assets/. Create it with app_new from a template, then work on its files with wangp_io: read_text, edit (replaces an exact passage: the way to change code), write_text (a new module) and append_text (a long file in parts), with the paths app_new returns (@documents/sandbox/apps/<name>/main.js). A call holds a few thousand tokens at most: keep modules under about 150 lines. Apps run in the user's browser, not in Python.",
    "Imports: the bare names of the import map (three, three/addons/..., cannon-es, phaser, @tweenjs/tween.js, simplex-noise, tone) and relative modules (./world.js). Nothing loads from the internet (no CDN, no fetch of URLs, no web fonts): the app is isolated and offline. Assets are files in the app folder, loaded with relative paths (assets/hero.webp).",
    "Make assets with WanGP: generate pictures, textures, sprites, 3D models (GLB) and music with wangp.generate / the toolbox, then wangp.app.import_asset(media, app, 'assets/name.webp', max_size=1024) copies them into the app resized and in a web format (WebP/PNG pictures, OGG/MP3 audio, MP4/WebM video, GLB). Procedural data (levels, terrain, beat maps) is computed in Python and written as JSON files under assets/.",
    "Keep the templates' hooks: the page fills the window (no scroll), the loop clamps its time step, window.wangpState() returns the app's key values (score, level, positions), and controls are listed in app.json. localStorage works and keeps its data per app (high scores, saves).",
    "Check every change with app_check (app: the folder name): it runs the app headless and returns errors, missing files, the frame rate, the state probe after each step, screenshots and, with record, a motion strip over a few seconds (inspect it with wangp_toolbox inspect_media to judge animation). steps drive input: {\"key\": \"ArrowRight\", \"hold_s\": 1}, {\"click\": \"#start\"} or {\"click\": [x, y]}, {\"mouse\": [x, y]}, {\"drag\": [[x1, y1], [x2, y2]]}, {\"wait_s\": 1}, {\"shot\": \"name\"}; {\"type\": \"...\"} only for a text field.",
    "The chat shows the app with a Play button after each run that changes it; app_logs returns what the user's browser reported while playing (errors, frame rate). app_publish adds it to the Apps gallery as a new version once it works. To change an app of the Apps gallery that is not in the sandbox (another conversation's, an imported one), app_open copies it into apps/<name>/ first; its next publication is its next version.",
]
APP = {"type": "string", "minLength": 1, "description": "The app: its folder name (app in app_new's result) or its title."}
ACTIONS = {
    "app_new": {
        "description": "Create a web app (an animation, a game, a three.js scene) in the sandbox from a template; then wangp_io's read_text, edit, write_text and append_text work on its files.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "minLength": 1, "description": "The app's title; its folder is apps/<slug of the name>."},
            "template": {"type": "string", "enum": list(bundles.TEMPLATES), "default": "three-scene", "description": "; ".join(f"{key}: {value}" for key, value in bundles.TEMPLATES.items())},
            "description": {"type": "string", "default": "", "description": "One sentence shown in the Apps gallery."},
        }, "required": ["name"], "additionalProperties": False},
        "example": {"action": "app_new", "arguments": {"name": "Orbit Garden", "template": "three-scene", "description": "Planets with generated textures orbiting a glowing sun."}},
    },
    "app_open": {
        "description": "Copy a published web app (an Apps gallery media id or a .wgpapp path) into the sandbox to change it; app_publish then adds its next version to the Apps gallery.",
        "parameters": {"type": "object", "properties": {"media": {"type": "string", "minLength": 1, "description": "The app: its Apps gallery media id or the path of its .wgpapp file."}}, "required": ["media"], "additionalProperties": False},
        "example": {"action": "app_open", "arguments": {"media": "app_3"}},
    },
    "app_check": {
        "description": "Run a web app headless: errors, missing files, frame rate, state probe after each input step, screenshots and an optional motion strip.",
        "parameters": {"type": "object", "properties": {
            "app": APP,
            "steps": {"type": "array", "items": {"type": "object"}, "maxItems": 40, "description": "Input to play: {key, hold_s}, {click: selector | 'text=Label' | [x, y]}, {type}, {mouse: [x, y]}, {drag: [[x1, y1], [x2, y2]]}, {wait_s}, {shot: name}; a step done several times takes repeat (<= 50) and every_s, the pause between them: {key: 'Space', repeat: 20, every_s: 0.3}."},
            "record": {"type": ["object", "boolean"], "description": "{seconds (<= 10), fps (<= 8)}, or true for 3 seconds at 4 fps: a strip of frames to judge motion with wangp_toolbox inspect_media."},
            "viewport": {"type": "string", "enum": ["desktop", "phone"], "default": "desktop"},
        }, "required": ["app"], "additionalProperties": False},
        "example": {"action": "app_check", "arguments": {"app": "orbit-garden", "steps": [{"key": "ArrowLeft", "hold_s": 1}, {"shot": "turned"}], "record": {"seconds": 3, "fps": 4}}},
    },
    "app_logs": {
        "description": "What the app reported from the user's browser while it was played: console lines, errors, frame rate, renderer.",
        "parameters": {"type": "object", "properties": {"app": APP, "since": {"type": "number", "default": 0, "description": "Only entries after this epoch time (the previous result's now)."}}, "required": ["app"], "additionalProperties": False},
    },
    "import_asset": {
        "description": "Copy a picture, 3D model, sound or video made with WanGP into an app's folder, resized and converted to the format of path's extension (also wangp.app.import_asset in scripts).",
        "parameters": {"type": "object", "properties": {
            "media": {"type": "string", "minLength": 1, "description": "A Gallery media id, an @alias path or a sandbox file."},
            "app": APP,
            "path": {"type": "string", "minLength": 1, "description": "Target inside the app folder; the extension sets the format: assets/x.webp|.png|.jpg, .ogg|.mp3|.m4a|.wav, .mp4|.webm, .glb."},
            "max_size": {"type": "integer", "minimum": 16, "description": "Pictures and video: longest side in pixels; .glb models: their textures' (1024 by default)."},
            "transparent": {"type": "boolean", "default": False, "description": "Pictures to .png or .webp: remove the background around the subject (a sprite, a character, an object on a plain background) and crop to it."},
        }, "required": ["media", "app", "path"], "additionalProperties": False},
        "example": {"action": "import_asset", "arguments": {"media": "visual:IMAGE", "app": "orbit-garden", "path": "assets/fox.png", "max_size": 256, "transparent": True}},
    },
    "app_publish": {
        "description": "Publish a working web app to the Apps gallery as a new version (a .wgpapp file the user keeps, plays full screen and can share).",
        "parameters": {"type": "object", "properties": {"app": APP, "note": {"type": "string", "default": "", "description": "What this version changes."}}, "required": ["app"], "additionalProperties": False},
    },
}
HELPERS = {
    "wangp.app.import_asset(media, app, path, max_size=None, transparent=False)": "copies a Gallery media id, @alias path or sandbox file into apps/<app>/<path> in the format of path's extension (.webp/.png/.jpg pictures resized to max_size pixels on the longer side, transparent=True cutting the subject out of its background, .ogg/.mp3/.m4a/.wav audio, .mp4/.webm video, .glb 3D models with their textures re-encoded and reduced to max_size, 1024 by default) and returns {file, bytes, size}",
    "wangp.app.check(app, steps=None, record=None) / wangp.app.new(name, template) / wangp.app.logs(app) / wangp.app.publish(app, note)": "the app actions from a script",
    "wangp.app.progression(path)": "searches every order of play of a story, quest or puzzle written as facts and actions (the progression-design skill's format): shortest path to each ending, endings never reached, dead ends, facts never needed, actions never possible",
}


_REMBG = []


def cut_out(picture):
    """The subject of a picture with its background made transparent (WanGP's background remover, on the CPU), cropped to it with a margin."""
    from rembg import remove

    from shared.utils.utils import new_rembg_session

    if not _REMBG:
        _REMBG.append(new_rembg_session())
    cut = remove(picture.convert("RGB"), session=_REMBG[0], alpha_matting=True)
    box = cut.getchannel("A").point(lambda value: 255 if value > 8 else 0).getbbox()
    if box is None:
        raise ValueError("No subject was found in the picture: nothing would remain once its background is removed.")
    margin = max(2, round(0.02 * max(cut.size)))
    return cut.crop((max(0, box[0] - margin), max(0, box[1] - margin), min(cut.width, box[2] + margin), min(cut.height, box[3] + margin)))


class AppWorkshop:
    """The apps of one sandbox. resolve_path(ref) gives the local path of a Gallery media id or authorized file; publish_app(package, description)
    records a .wgpapp in the outputs and the Apps gallery and returns its compact Gallery record; virtualize(path) gives the assistant's form of a path."""

    def __init__(self, sandbox_root: Path, *, resolve_path: Callable[[str], str], publish_app: Callable[[str, str], dict | None], virtualize: Callable[[str], str] = str):
        self.sandbox = Path(sandbox_root).resolve()
        self.root = self.sandbox / "apps"
        self._resolve_path, self._publish_app, self._virtualize = resolve_path, publish_app, virtualize

    def describe(self) -> dict[str, Any]:
        return {"templates": bundles.TEMPLATES, "import_map": sorted(bundles.import_map()["imports"]), "apps": self.names(), "rules": RULES, "helpers": HELPERS}

    def names(self) -> list[str]:
        return sorted(path.parent.name for path in self.root.glob(f"*/{bundles.MANIFEST}")) if self.root.is_dir() else []

    def folder(self, name: str) -> Path:
        folder = self.root / bundles.slugify(name)
        if not (folder / bundles.MANIFEST).is_file():
            known = self.names()
            raise ValueError(f"No app named {folder.name}. " + (f"Apps: {', '.join(known)}." if known else "There is none yet: app_new creates one."))
        return folder

    def run_action(self, action: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if action != "app_new" and "name" in arguments and "app" not in arguments:  # every result names the app "app"; "name" is the title of app_new
            arguments = {"app": arguments.pop("name"), **arguments}
        try:
            result = getattr(self, action)(**arguments)
            return {"status": "done", **result} if action == "import_asset" else result
        except (ValueError, RuntimeError, FileNotFoundError, PermissionError) as error:
            return {"status": "error", "error": str(error)}

    # ---- actions
    def app_new(self, name: str, template: str = "three-scene", description: str = "") -> dict[str, Any]:
        folder = bundles.create(self.root, name, template, description)
        files = [self._virtualize(str(path)) for path in bundles.bundle_files(folder)]  # the paths wangp_io takes
        return {"status": "done", "app": folder.name, "folder": self._virtualize(str(folder)), "files": files, "apps": [self.preview(folder)],
                "next": f"wangp_io read_text {self._virtualize(str(folder / 'main.js'))}, then change it with wangp_io edit (write_text replaces a starter file whole) and add modules with write_text (under about 150 lines per call: a longer call is cut off before it runs), bring assets made with WanGP in with import_asset, then app_check."}

    def is_starter(self, path: str) -> bool:
        """A file of an app that app_new copied from its template and nobody changed since: replacing it loses nothing."""
        file = Path(path).resolve()
        if not file.is_relative_to(self.root) or not file.is_file() or len(file.relative_to(self.root).parts) < 2:
            return False
        relative, data = Path(*file.relative_to(self.root).parts[1:]), file.read_bytes()
        return any((source := bundles.TEMPLATES_DIR / template / relative).is_file() and source.read_bytes() == data for template in bundles.TEMPLATES)

    def manifest_identity(self, path: str) -> dict[str, Any] | None:
        """The id and version of an app's app.json at path, read before a write that could drop them."""
        file = Path(path).resolve()
        if file.name != bundles.MANIFEST or file.parent.parent != self.root or not file.is_file():
            return None
        try:
            manifest = json.loads(file.read_text(encoding="utf-8"))
        except ValueError:
            return None
        return {key: manifest[key] for key in ("id", "version") if key in manifest} if isinstance(manifest, dict) else None

    def restore_identity(self, path: str, identity: dict[str, Any] | None) -> None:
        """Puts back the id and version a rewrite of app.json left out: they tie the app to its published versions (values it sets stay)."""
        try:
            manifest = json.loads(Path(path).read_text(encoding="utf-8")) if identity else None
        except (ValueError, OSError):
            return
        if isinstance(manifest, dict) and set(identity) - set(manifest):
            Path(path).write_text(json.dumps({**manifest, **{key: value for key, value in identity.items() if key not in manifest}}, ensure_ascii=False, indent=2), encoding="utf-8")

    def app_open(self, media: str) -> dict[str, Any]:
        package = Path(self._resolve_path(str(media)))
        if package.suffix.lower() != bundles.EXTENSION:
            raise ValueError(f"{media} is not a web app ({bundles.EXTENSION} file of the Apps gallery).")
        manifest = bundles.package_manifest(package)
        folder = self.root / bundles.slugify(manifest["name"])
        if folder.exists():
            existing = bundles.read_manifest(folder)
            if existing.get("id") != manifest.get("id"):
                raise ValueError(f"Another app already uses apps/{folder.name} in the sandbox: rename that one's folder first.")
            return {"status": "done", "app": folder.name, "folder": self._virtualize(str(folder)), "version": existing["version"], "apps": [self.preview(folder)],
                    "next": f"{manifest['name']} is already in the sandbox, maybe with changes newer than the gallery's: edit its files, then app_publish."}
        source = bundles.unpack(package)
        shutil.copytree(source, folder, ignore=lambda root, names: PACKAGE_ONLY & set(names) if Path(root) == source else ())
        # the next version follows the latest publication of this app (same manifest id) beside the package, not only the one opened
        published = [bundles.package_manifest(path) for path in package.parent.glob(f"*{bundles.EXTENSION}")]
        version = max(int(item["version"]) for item in published if item.get("id") == manifest.get("id")) + 1
        bundles.write_manifest(folder, {**bundles.read_manifest(folder), "version": version})
        files = [self._virtualize(str(path)) for path in bundles.bundle_files(folder)]
        return {"status": "done", "app": folder.name, "folder": self._virtualize(str(folder)), "files": files, "version": version, "apps": [self.preview(folder)],
                "next": f"Edit its files with wangp_io (read_text, edit, write_text), app_check, then app_publish: it becomes version {version}."}

    def progression(self, path: str) -> dict[str, Any]:
        file = self.sandbox / path if (self.sandbox / path).is_file() else Path(self._resolve_path(str(path)))
        result = progression.check(json.loads(file.read_text(encoding="utf-8")))
        if result is None:
            raise ValueError(f"{path} holds no actions with requires or gives: write the facts-and-actions format of the progression-design skill.")
        return result

    def app_check(self, app: str, steps: list | None = None, record: dict | bool | None = None, viewport: str = "desktop") -> dict[str, Any]:
        from shared.apps import check

        folder = self.folder(app)
        static = bundles.static_check(folder)
        result: dict[str, Any] = {"status": "done", "app": folder.name, "files": static["files"], "megabytes": round(static["bytes"] / 1048576, 2), "findings": static["findings"]}
        if any(item["level"] == "error" and item["file"] in (bundles.MANIFEST, (static["manifest"] or {}).get("entry")) for item in static["findings"]):
            result["status"] = "issues"
            return result
        run = check.run(folder, self.sandbox / "tmp" / "app_checks" / folder.name, steps=steps, record={} if record is True else record or None, viewport=viewport)
        for shot in run["screenshots"]:
            shot["path"] = self._virtualize(shot["path"])
        if run.get("motion"):
            run["motion"]["strip"] = self._virtualize(run["motion"]["strip"])
        story = folder / "assets" / "story.json"
        result["findings"] = static["findings"] + run.pop("findings") + (progression.findings(story) if story.is_file() else [])
        result.update(run)
        first = next((Path(item["path"]) for item in run["screenshots"] if not item["blank"]), None)
        if first is not None:
            local = self.sandbox / "tmp" / "app_checks" / folder.name / Path(first).name
            if local.is_file():
                shutil.copyfile(local, self.sandbox / "tmp" / "app_checks" / f"{folder.name}.thumbnail.png")
        if any(item["level"] == "error" for item in result["findings"]):
            result["status"] = "issues"
        result["next"] = "Fix the error findings and check again." if result["status"] == "issues" else "Look at the screenshots or the motion strip with wangp_toolbox inspect_media (wangp.inspect in a script) when appearance matters; app_publish when it is ready."
        return result

    def app_logs(self, app: str, since: float = 0.0) -> dict[str, Any]:
        folder = self.folder(app)
        logs = server.app_logs(server.register_dev(folder), float(since or 0))
        return {"status": "done", "app": folder.name, **logs, "now": round(time.time(), 3), "note": "Entries come from the user's browser while the app was played in the chat, the preview or the gallery." if logs["entries"] or logs["stats"] else "No report yet: the app has not been played since WanGP started."}

    def app_publish(self, app: str, note: str = "") -> dict[str, Any]:
        folder = self.folder(app)
        static = bundles.static_check(folder)
        errors = [item for item in static["findings"] if item["level"] == "error"]
        if errors:
            return {"status": "error", "error": "Fix these before publishing: " + "; ".join(f"{item['file']}: {item['message']}" for item in errors[:5])}
        thumbnail = self.sandbox / "tmp" / "app_checks" / f"{folder.name}.thumbnail.png"
        if not thumbnail.is_file():
            self.app_check(app)
        manifest = static["manifest"]
        if not manifest.get("id"):  # an app.json rewritten without its id (from a script): the app gets one again before it is packed
            manifest = {**manifest, "id": uuid.uuid4().hex[:12]}
            bundles.write_manifest(folder, manifest)
        packages = self.sandbox / "tmp" / "app_packages"
        packages.mkdir(parents=True, exist_ok=True)
        package = bundles.pack(folder, packages / f"{folder.name}{bundles.EXTENSION}", thumbnail if thumbnail.is_file() else None,
                               {"app": folder.name, "version": manifest["version"], "note": str(note or ""), "made_with": "Deepy"})
        description = manifest["description"] or manifest["name"]
        item = self._publish_app(str(package), f"{manifest['name']} v{manifest['version']}: {description}" + (f" ({note})" if note else ""))
        if item is None:
            raise RuntimeError("WanGP could not publish the app.")
        bundles.write_manifest(folder, {**manifest, "version": int(manifest["version"]) + 1})  # the next publication is the next version
        return {"status": "done", "app": folder.name, "version": manifest["version"], "outputs": [item], "output_files": [item["path"]] if item.get("path") else [],
                "next": "The app is in the Apps gallery; keep editing apps/" + folder.name + " for the next version."}

    def preview(self, folder: Path) -> dict[str, Any]:
        """The chat's Play card of a development app (rendered from this record by the chat)."""
        manifest = bundles.read_manifest(folder)
        thumbnail = self.sandbox / "tmp" / "app_checks" / f"{folder.name}.thumbnail.png"
        return {"app": folder.name, "name": manifest["name"], "folder": self._virtualize(str(folder)), "version": bundles.signature(folder), "thumbnail": self._virtualize(str(thumbnail)) if thumbnail.is_file() else ""}

    def changed(self, paths: list[str]) -> list[dict[str, Any]]:
        """Play cards of the apps whose files a run changed."""
        names = sorted({Path(path).relative_to(self.root).parts[0] for path in paths if Path(path).is_relative_to(self.root) and len(Path(path).relative_to(self.root).parts) > 1})
        return [self.preview(self.root / name) for name in names if (self.root / name / bundles.MANIFEST).is_file() and not self.manifest_problem(self.root / name / bundles.MANIFEST)]

    def manifest_problem(self, path: str | Path) -> str | None:
        """Why the app.json at path does not load (an app being rewritten has no Play card until it does), or None."""
        try:
            bundles.read_manifest(Path(path).parent)
        except ValueError as error:
            return str(error)
        return None

    # ---- script helpers
    def import_asset(self, media: str, app: str, path: str, max_size: int | None = None, transparent: bool = False) -> dict[str, Any]:
        folder = self.folder(app)
        target = (folder / str(path)).resolve()
        if not target.is_relative_to(folder) or target.name.startswith("."):
            raise ValueError(f"{path} must be a path inside the app folder, such as assets/hero.webp.")
        candidate = self.sandbox / str(media)
        source = candidate.resolve() if not str(media).startswith("@") and candidate.is_file() else Path(self._resolve_path(str(media)))
        suffix = target.suffix.lower()
        target.parent.mkdir(parents=True, exist_ok=True)
        facts: dict[str, Any] = {}
        if transparent and suffix not in (".png", ".webp"):
            raise ValueError("transparent=True needs a .png or .webp path: those keep the alpha channel.")
        if suffix in ASSET_IMAGE:
            from PIL import Image

            with Image.open(source) as image:
                image.load()
                picture = image.convert("RGBA" if suffix in (".png", ".webp") and ("A" in image.getbands() or "transparency" in image.info) else "RGB")
            if transparent:
                picture = cut_out(picture)
            if max_size:
                picture.thumbnail((int(max_size), int(max_size)), Image.LANCZOS)
            picture.save(target, **({"quality": 90} if suffix in (".jpg", ".jpeg", ".webp") else {"optimize": True}))
            facts["size"] = list(picture.size)
        elif suffix in ASSET_AUDIO or suffix in ASSET_VIDEO:
            from shared.utils.video_decode import resolve_media_binary

            scale = ["-vf", f"scale='min({int(max_size)},iw)':-2"] if max_size and suffix in ASSET_VIDEO else []
            command = [resolve_media_binary("ffmpeg"), "-v", "error", "-y", "-i", str(source), *(["-vn"] if suffix in ASSET_AUDIO else scale), *(ASSET_AUDIO.get(suffix) or ASSET_VIDEO[suffix]), str(target)]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=600)
            if completed.returncode != 0:
                raise RuntimeError(f"ffmpeg could not convert {source.name}: {completed.stderr.strip()[-400:]}")
        elif suffix in ASSET_MESH:
            from shared.mesh3d.glb import glb_compress_images, load_mesh, mesh_to_glb

            data = source.read_bytes() if source.suffix.lower() == ".glb" else mesh_to_glb(load_mesh(str(source)))
            target.write_bytes(glb_compress_images(data, int(max_size or 1024)))  # generated models carry 2048 PNG textures: most of their size
        else:
            raise ValueError(f"import_asset writes pictures ({', '.join(sorted(ASSET_IMAGE))}), audio ({', '.join(ASSET_AUDIO)}), video ({', '.join(ASSET_VIDEO)}) or .glb; for data files write them directly.")
        return {"file": target.relative_to(folder).as_posix(), "app": folder.name, "bytes": target.stat().st_size, **facts}

    def script_call(self, action: str, arguments: dict[str, Any]) -> Any:
        """The wangp.app helpers of a script: the actions above, import_asset and progression."""
        if action not in ACTIONS and action != "progression":
            raise ValueError(f"Unknown app helper {action!r}.")
        if action != "app_new" and "name" in arguments and "app" not in arguments:
            arguments = {"app": arguments.pop("name"), **arguments}
        result = getattr(self, action)(**arguments)
        if result.get("status", "done") == "error":
            raise RuntimeError(result["error"])
        return result
