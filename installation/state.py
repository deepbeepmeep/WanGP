"""Durable installation metadata, path boundaries and cross-process leases."""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tempfile
from functools import lru_cache


class InstallError(RuntimeError):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_hash(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _stream_hash(stream)


def _stream_hash(stream):
    result = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        result.update(chunk)
    return result.hexdigest()


@lru_cache(maxsize=1)
def windows_long_paths_enabled():
    if os.name != "nt":
        return True
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem") as key:
            return bool(winreg.QueryValueEx(key, "LongPathsEnabled")[0])
    except OSError:
        return False


def plain_path(path):
    path = Path(os.path.abspath(path))
    if os.name == "nt" and not str(path).startswith("\\\\?\\") and len(str(path)) >= 248 and not windows_long_paths_enabled():
        raise InstallError("path_too_long", f"Windows long paths are disabled. Choose a shorter installation folder, such as E:\\WanGP. Path: {path}")
    for part in (*reversed(path.parents), path):
        try:
            info = part.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise InstallError("unsafe_path", f"Links and reparse points are not allowed here: {part}")
    return path


def relative_path(value):
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise InstallError("unsafe_path", f"Invalid release path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or str(path) != value or any(part in ("", ".", "..") or part.endswith((" ", ".")) for part in path.parts):
        raise InstallError("unsafe_path", f"Invalid release path: {value!r}")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if any(part.split(".")[0].upper() in reserved or any(ord(char) < 32 for char in part) for part in path.parts):
        raise InstallError("unsafe_path", f"Reserved release path: {value!r}")
    return path


def inside(root, relative):
    root = plain_path(root)
    path = plain_path(root.joinpath(*relative_path(relative).parts))
    if not path.is_relative_to(root):
        raise InstallError("unsafe_path", f"Path leaves its declared root: {relative}")
    return path


def sync_directory(path):
    if os.name != "nt":
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def atomic_bytes(path, content):
    path = plain_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".manager-write-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        plain_path(path)
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path, data):
    atomic_bytes(path, canonical_json(data) + b"\n")


def atomic_copy(source, destination, expected_hash):
    source, destination = plain_path(source), plain_path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".manager-write-", dir=destination.parent)
    try:
        hasher = hashlib.sha256()
        with os.fdopen(descriptor, "wb") as outgoing, open(source, "rb") as incoming:
            for chunk in iter(lambda: incoming.read(1024 * 1024), b""):
                hasher.update(chunk)
                outgoing.write(chunk)
            outgoing.flush()
            os.fsync(outgoing.fileno())
        if hasher.hexdigest() != expected_hash:
            raise InstallError("changed_input", f"File changed while copying: {source}")
        os.chmod(temporary, stat.S_IMODE(source.stat().st_mode))
        plain_path(destination)
        os.replace(temporary, destination)
        sync_directory(destination.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def read_json(path):
    with open(plain_path(path), encoding="utf-8") as stream:
        return json.load(stream)


class InstallationLock:
    """The same exclusive lease covers maintenance and a managed server lifetime."""

    def __init__(self, root):
        self.root = plain_path(root)
        self.stream = None

    def __enter__(self):
        path = inside(self.root, ".manager/installation.lock")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = open(path, "a+b")
        self.stream.seek(0, os.SEEK_END)
        if self.stream.tell() == 0:
            self.stream.write(b"\0")
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            self.stream = None
            raise InstallError("installation_busy", f"Installation is running or being maintained: {self.root}") from exc
        return self

    def __exit__(self, *exception):
        self.stream.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()
