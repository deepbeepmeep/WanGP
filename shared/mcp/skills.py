# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Persistent, shared Markdown skills exposed through ordinary MCP resources, in the Agent Skills format: one folder per skill holding its
SKILL.md (YAML name and description, then the method) and optional files the method names by relative path."""
import hashlib
import os
import re
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from urllib.parse import unquote

import yaml
from filelock import FileLock
from tqdm import tqdm

USER_SKILL_PREFIX = "wangp://skills/user/"
USER_SKILLS_DIR = Path(__file__).resolve().parents[2] / "user_skills"  # the installation's folder; a session may name another root
MAX_SKILL_BYTES = 128 * 1024
MAX_SKILL_FILES = 100
MAX_SKILL_FOLDER_BYTES = 16 * 1024 * 1024


def _front_matter(content):
    match = re.fullmatch(r"---\r?\n(.*?)\r?\n---\r?\n(.+)", content, flags=re.DOTALL)
    if match is None or not match[2].strip():
        raise ValueError("A skill needs YAML front matter between --- lines and a nonempty Markdown body.")
    try:
        metadata = yaml.safe_load(match[1])
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid skill front matter: {exc}") from exc
    return metadata if isinstance(metadata, dict) else {}


def skill_metadata(content, name):
    metadata = _front_matter(content)
    if metadata.get("name") != name:
        raise ValueError("Skill front matter must contain name matching the skill's directory name.")
    description = metadata.get("description")
    if not isinstance(description, str) or not description.strip() or len(description) > 1024:
        raise ValueError("Skill description must be nonempty text of at most 1024 characters.")
    return {"name": name, "description": description.strip()}


def skill_index(rows):
    """Each usable skill by URI and description, as the Agent Skills format loads them at startup, for the agents' instructions; empty without any."""
    lines = [f"- {row['uri']}: {' '.join(row['description'].split())}" for row in rows if "problem" not in row]
    return "\n".join(["The user's own skills add methods. Read a skill before a task its description covers; a file it names is read at the skill's URI followed by /<path>:", *lines]) if lines else ""


def _archive_files(source):
    """The files of one skill folder packed in a .zip, by path relative to the folder: SKILL.md at the archive's root or in one top folder."""
    with zipfile.ZipFile(source) as archive:
        entries = [info for info in archive.infolist() if not info.is_dir() and not info.filename.startswith("__MACOSX/")]
        manifests = [info.filename for info in entries if re.fullmatch(r"(?:[^/]+/)?SKILL\.md", info.filename)]
        if "SKILL.md" in manifests:
            base = ""
        elif len(manifests) == 1:
            base = manifests[0].removesuffix("SKILL.md")
        else:
            raise ValueError("The archive must hold one skill: its SKILL.md at the root or in one top folder." if manifests else "The archive holds no SKILL.md at its root or in a top folder.")
        files, total = {}, 0
        for info in entries:
            if not info.filename.startswith(base):
                continue
            relative = info.filename[len(base):]
            if any(part in ("", ".", "..") or ":" in part or "\\" in part for part in relative.split("/")):
                raise ValueError(f"Unsafe path in the archive: {info.filename}")
            total += info.file_size
            if len(files) == MAX_SKILL_FILES or total > MAX_SKILL_FOLDER_BYTES:
                raise ValueError(f"A skill cannot hold more than {MAX_SKILL_FILES} files or {MAX_SKILL_FOLDER_BYTES >> 20} MiB.")
            files[relative] = archive.read(info)
    return files


class SkillStore:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self._lock = FileLock(str(self.root / ".write.lock"), timeout=10)

    def path(self, name):
        if len(name) > 64 or re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name) is None or Path(name).is_reserved():
            raise ValueError("Skill name must be a lowercase hyphenated name, at most 64 characters, not a reserved filename.")
        path = (self.root / name / "SKILL.md").resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Skill path must stay inside the user-skills directory.")
        return path

    def read_document(self, name):
        """Read an editable snapshot, including a document whose front matter needs repairing."""
        path = self.path(name)
        with self._lock:
            with path.open("rb") as reader:
                data = reader.read(MAX_SKILL_BYTES + 1)
        if len(data) > MAX_SKILL_BYTES:
            raise ValueError("A skill cannot exceed 128 KiB.")
        content = data.decode("utf-8-sig")
        return {"revision": hashlib.sha256(data).hexdigest(), "content": content}

    def read(self, name):
        document = self.read_document(name)
        return {**skill_metadata(document["content"], name), "uri": USER_SKILL_PREFIX + name, **document}

    def file(self, name, relative):
        """A text file of a skill's folder, named by its path relative to the folder as the skill's method names it."""
        folder = self.path(name).parent
        path = (folder / relative).resolve()
        if not path.is_relative_to(folder) or not path.is_file():
            raise ValueError(f"The {name} skill has no file {relative}.")
        with path.open("rb") as reader:
            data = reader.read(MAX_SKILL_BYTES + 1)
        if len(data) > MAX_SKILL_BYTES:
            raise ValueError("A skill file cannot exceed 128 KiB.")
        try:
            return data.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise ValueError(f"{relative} is not a UTF-8 text file: agents read only the text files of a skill.") from None

    def scan(self):
        """Every folder of the user-skills directory: a skill's record, or its folder name and the problem that keeps agents from using it."""
        rows = []
        for folder in sorted(path for path in self.root.iterdir() if path.is_dir() and not path.name.startswith(".")) if self.root.is_dir() else ():
            try:
                rows.append(self.read(folder.name))
            except FileNotFoundError:
                rows.append({"name": folder.name, "problem": "The folder holds no SKILL.md."})
            except (OSError, ValueError) as exc:
                rows.append({"name": folder.name, "problem": str(exc)})
        return rows

    def skills(self):
        return [row for row in self.scan() if "problem" not in row]

    def instructions(self):
        """The user's skills for the agents' instructions, reporting in the console the folders they cannot use."""
        rows = self.scan()
        for row in rows:
            if "problem" in row:
                tqdm.write(f"[Skills] User skill folder '{row['name']}' is not used: {row['problem']}")
        return skill_index(rows)

    def save(self, name: str, content: str, expected_revision: str | None = None):
        path = self.path(name)
        metadata = skill_metadata(content, name)
        data = content.encode("utf-8")
        if len(data) > MAX_SKILL_BYTES:
            raise ValueError("A skill cannot exceed 128 KiB.")
        self.root.mkdir(parents=True, exist_ok=True)
        with self._lock:
            revision = self.read_document(name)["revision"] if path.exists() else None
            if expected_revision != revision:
                raise ValueError("Skill revision conflict. Read the skill with mcp_resource, then update using its revision; omit expected_revision only when creating a new skill.")
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, temporary = tempfile.mkstemp(prefix=".skill-", suffix=".tmp", dir=path.parent)
            try:
                with os.fdopen(fd, "wb") as writer:
                    writer.write(data)
                    writer.flush()
                    os.fsync(writer.fileno())
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        return {"status": "done", **metadata, "uri": USER_SKILL_PREFIX + name, "revision": hashlib.sha256(data).hexdigest()}

    def install(self, source, replace=False):
        """Install a skill from its SKILL.md or a .zip of its folder, as copying the folder into the user-skills directory would."""
        source = Path(source)
        files = _archive_files(source) if source.suffix.lower() == ".zip" else {"SKILL.md": source.read_bytes()}
        if len(files["SKILL.md"]) > MAX_SKILL_BYTES:
            raise ValueError("A skill cannot exceed 128 KiB.")
        content = files["SKILL.md"].decode("utf-8-sig")
        name = str(_front_matter(content).get("name", ""))
        target = self.path(name).parent
        skill_metadata(content, name)
        self.root.mkdir(parents=True, exist_ok=True)
        with self._lock:
            if target.exists() and not replace:
                raise ValueError(f"A skill named {name} is already installed: tick Replace to install this one instead.")
            staging = Path(tempfile.mkdtemp(prefix=".import-", dir=self.root))
            try:
                for relative, data in files.items():
                    (staging / relative).parent.mkdir(parents=True, exist_ok=True)
                    (staging / relative).write_bytes(data)
                retired = self.root / f".replaced-{name}-{time.time_ns()}" if target.exists() else None
                if retired is not None:
                    os.replace(target, retired)
                try:
                    os.replace(staging, target)
                except OSError:
                    if retired is not None:
                        os.replace(retired, target)
                    raise
                if retired is not None:
                    shutil.rmtree(retired, ignore_errors=True)
            finally:
                shutil.rmtree(staging, ignore_errors=True)
        return self.read(name)

    def remove(self, name):
        """Delete a skill's folder, usable or not."""
        folder = (self.root / name).resolve()
        if folder.parent != self.root or not folder.is_dir():
            raise ValueError(f"No skill folder named {name}.")
        with self._lock:
            shutil.rmtree(folder)

    def archive(self, name):
        """Move a skill and its companions out of discovery and the skill resource namespace."""
        folder = (self.root / name).resolve()
        archived = (self.root / ".archive").resolve()
        if name != folder.name or folder.name.startswith(".") or folder.parent != self.root or not folder.is_dir():
            raise ValueError(f"No skill folder named {name}.")
        if archived != self.root / ".archive":
            raise ValueError("The skill archive must be a real folder inside the user-skills directory.")
        with self._lock:
            archived.mkdir(exist_ok=True)
            target = archived / f"{folder.name}-{time.time_ns()}"
            os.replace(folder, target)
        return target.relative_to(self.root).as_posix()



def skill_mcp(*args, skills, **kwargs):
    """A FastMCP server that also lists and serves the store's skills. The MCP package loads here, not with the store (the settings use it)."""
    from mcp.server.fastmcp import FastMCP
    from mcp.server.lowlevel.helper_types import ReadResourceContents
    from mcp.types import Resource

    class SkillMCP(FastMCP):
        async def list_resources(self):
            return [*await super().list_resources(), *(Resource(uri=item["uri"], name=item["name"], description=item["description"], mimeType="text/markdown") for item in skills.skills())]

        async def read_resource(self, uri):
            if str(uri).startswith(USER_SKILL_PREFIX):
                name, _, relative = unquote(str(uri)[len(USER_SKILL_PREFIX):]).partition("/")
                if relative:
                    return [ReadResourceContents(content=skills.file(name, relative), mime_type="text/markdown" if relative.lower().endswith(".md") else "text/plain")]
                record = skills.read(name)
                return [ReadResourceContents(content=record["content"], mime_type="text/markdown", meta={"revision": record["revision"]})]
            return await super().read_resource(uri)

    server = SkillMCP(*args, **kwargs)
    server.skills = skills
    return server
