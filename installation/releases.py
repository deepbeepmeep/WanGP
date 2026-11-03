"""Explicit local release bundles with immutable source inventories."""

import re
import time
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

from .state import InstallError, atomic_copy, atomic_json, digest, file_hash, inside, plain_path, read_json, relative_path


PROTECTED_ROOTS = {".git", ".manager", "settings", "finetunes", "workspaces", "deepy_sessions", "user_skills", "ckpts", "storage", "samples", "tmp", "cache", "computecache", "gradio_outputs", "mask_outputs", "ffmpeg_bins"}
PROTECTED_FILES = {"wgp_config.json", "envs.json", "plugins_local.json", "loras_url_cache.json", "loras_url_cache_v2.json", "resolutions.json", ".env", "scripts/args.txt", "queue.zip", "error_queue.zip"}


def source_path(value):
    path = relative_path(value)
    root = path.parts[0].lower()
    if root in PROTECTED_ROOTS or root.startswith(("outputs", "loras")) or value.lower() in PROTECTED_FILES or path.suffix.lower() in (".safetensors", ".ckpt", ".pt", ".pth", ".pyc") or any(part in ("__pycache__", "_runtime") or part.startswith(".manager-write-") for part in path.parts):
        raise InstallError("protected_file", f"User data cannot be owned by a source release: {value}")
    return path


def validate_manifest(manifest):
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("release_id"), str) or not manifest["release_id"]:
        raise InstallError("invalid_release", "Release requires schema version 1 and a release ID")
    if not isinstance(manifest.get("files"), dict) or not manifest["files"]:
        raise InstallError("invalid_release", "Release requires an explicit file inventory")
    names = set()
    for filename, expected in manifest["files"].items():
        source_path(filename)
        if filename.casefold() in names or not isinstance(expected, str) or not re.fullmatch("[0-9a-f]{64}", expected):
            raise InstallError("invalid_release", f"Invalid or case-colliding release file: {filename}")
        names.add(filename.casefold())
    for name in names:
        if any(parent.as_posix() in names for parent in Path(name).parents if str(parent) != "."):
            raise InstallError("invalid_release", f"Release file/directory collision: {name}")
    if not isinstance(manifest.get("config_schema"), int) or not isinstance(manifest.get("compatible_config_schemas"), list) or manifest["config_schema"] not in manifest["compatible_config_schemas"]:
        raise InstallError("invalid_release", "Release must declare its supported configuration schemas")
    if manifest.get("manifest_sha256") != digest({key: value for key, value in manifest.items() if key != "manifest_sha256"}):
        raise InstallError("changed_release", "Release manifest hash mismatch")
    return manifest


def source_manifest(inventory, release_id, *, config_schema=1, compatible_config_schemas=None):
    """Describe source bytes already inventoried by the reader or copier."""
    manifest = {"schema_version": 1, "release_id": release_id, "config_schema": config_schema, "compatible_config_schemas": compatible_config_schemas or [config_schema], "files": dict(inventory), "trust": "explicit_local_source"}
    manifest["manifest_sha256"] = digest(manifest)
    return validate_manifest(manifest)


def create_release(source, files, destination, release_id, *, config_schema=1, compatible_config_schemas=None):
    source, destination = plain_path(source), plain_path(destination)
    if not isinstance(files, list) or not files:
        raise InstallError("invalid_release", "Release files must be a non-empty JSON list")
    if destination.is_relative_to(source):
        raise InstallError("unsafe_path", "A release bundle must be outside its source directory")
    inventory = {}
    for filename in files:
        source_path(filename)
        if filename in inventory:
            raise InstallError("invalid_release", f"Duplicate release file: {filename}")
        inventory[filename] = file_hash(inside(source, filename))
        inside(destination / "source", filename)
    manifest = source_manifest(inventory, release_id, config_schema=config_schema, compatible_config_schemas=compatible_config_schemas)
    destination.mkdir(parents=True, exist_ok=False)
    for filename, expected in sorted(inventory.items()):
        atomic_copy(inside(source, filename), inside(destination / "source", filename), expected)
    atomic_json(destination / "release.json", manifest)
    return manifest


def load_release(directory, *, emit=None, check_cancelled=None):
    directory = plain_path(directory)
    manifest = validate_manifest(read_json(directory / "release.json"))
    check_cancelled = check_cancelled or (lambda: None)
    last_event = 0
    def progress(completed):
        nonlocal last_event
        now = time.monotonic()
        if emit and (completed in (0, len(manifest["files"])) or now - last_event >= 1 / 3):
            emit({"type": "progress", "title": "Verifying WanGP Source", "message": f"{completed:,} / {len(manifest['files']):,} files verified", "completed": completed, "total": len(manifest["files"])})
            last_event = now
    def verify(item):
        check_cancelled()
        filename, expected = item
        if file_hash(inside(directory / "source", filename)) != expected:
            raise InstallError("changed_release", f"Release source hash mismatch: {filename}")
    check_cancelled()
    progress(0)
    items = iter(manifest["files"].items())
    if len(manifest["files"]) < 64:
        for completed, item in enumerate(items, 1):
            verify(item)
            progress(completed)
    else:
        # Bounded read-only concurrency amortizes per-file OS/scanner latency.
        # Keep path/reparse checks and content hashes for every file; do not
        # replace validation with timestamp or size comparisons.
        completed = 0
        with ThreadPoolExecutor(max_workers=4, thread_name_prefix="source-check") as executor:
            while batch := list(islice(items, 64)):
                for _ in executor.map(verify, batch):
                    check_cancelled()
                    completed += 1
                    progress(completed)
    check_cancelled()
    return manifest


def cache_release(directory, destination):
    manifest = load_release(directory)
    destination = plain_path(destination)
    if destination.exists():
        if (destination / "release.json").exists() and load_release(destination) != manifest:
            raise InstallError("changed_release", "Cached release differs from its source")
    else:
        destination.mkdir(parents=True, exist_ok=False)
    for filename, expected in manifest["files"].items():
        target = inside(destination / "source", filename)
        if target.exists():
            if file_hash(target) != expected:
                raise InstallError("changed_release", f"Partial cached source differs: {filename}")
        else:
            atomic_copy(inside(Path(directory) / "source", filename), target, expected)
    atomic_json(destination / "release.json", manifest)
    return manifest
