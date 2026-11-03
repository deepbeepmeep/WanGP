"""Read the existing installer catalog without importing the application."""

import copy
import hashlib
import json
import re
import shlex
import sys
from pathlib import Path


class CatalogError(ValueError):
    pass


def os_key(platform=None):
    platform = sys.platform if platform is None else platform
    if platform in ("win32", "win"):
        return "win"
    if platform in ("darwin", "macos"):
        return "macos"
    if platform.startswith("linux"):
        return "linux"
    raise CatalogError(f"Unsupported platform: {platform}")


def resolve_command(entry, platform=None):
    return entry.get(os_key(platform)) if isinstance(entry, dict) else entry


def validate_catalog(data):
    if not isinstance(data, dict):
        raise CatalogError("Catalog must be an object")
    components = data.get("components")
    profiles = data.get("gpu_profiles")
    if not isinstance(components, dict) or not isinstance(profiles, dict) or not profiles:
        raise CatalogError("Catalog requires components and non-empty gpu_profiles objects")
    for group, entries in components.items():
        if not isinstance(entries, dict):
            raise CatalogError(f"components.{group} must be an object")
        for key, entry in entries.items():
            location = f"components.{group}.{key}"
            if not isinstance(entry, dict) or not isinstance(entry.get("label"), str):
                raise CatalogError(f"{location} requires a label")
            if group == "python":
                if not isinstance(entry.get("ver"), str) or not re.fullmatch(r"\d+\.\d+\.\d+", entry["ver"]):
                    raise CatalogError(f"{location}.ver must be an exact Python version")
                continue
            command = entry.get("cmd")
            if isinstance(command, dict):
                if not command or set(command) - {"win", "linux", "macos"}:
                    raise CatalogError(f"{location}.cmd has unsupported platform keys")
                values = command.values()
            else:
                values = [command]
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise CatalogError(f"{location}.cmd requires non-empty command strings")
    for key, profile in profiles.items():
        if not isinstance(profile, dict):
            raise CatalogError(f"gpu_profiles.{key} must be an object")
        platforms = profile.get("platforms", ["win", "linux", "macos"])
        if not isinstance(platforms, list) or not platforms or any(platform not in ("win", "linux", "macos") for platform in platforms):
            raise CatalogError(f"gpu_profiles.{key}.platforms must list supported platform IDs")
        for required in ("python", "torch"):
            if not isinstance(profile.get(required), str):
                raise CatalogError(f"gpu_profiles.{key}.{required} requires a component ID")
        for group in ("python", "torch", "triton", "sage", "sparge", "flash", "kernels"):
            value = profile.get(group)
            if group == "kernels":
                if not isinstance(value, list):
                    raise CatalogError(f"gpu_profiles.{key}.kernels must be a list")
                keys = value
            else:
                keys = [] if value is None else [value]
            for component in keys:
                if not isinstance(component, str) or component not in components.get(group, {}):
                    raise CatalogError(f"gpu_profiles.{key}.{group}: unknown component {component!r}")
    return data


def load_config(path="setup_config.json"):
    path = Path(path)
    try:
        return validate_catalog(json.loads(path.read_text(encoding="utf-8-sig")))
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogError(f"Cannot read {path}: {exc}") from exc


def catalog_digest(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def select_profile(data, key):
    try:
        return copy.deepcopy(data["gpu_profiles"][key])
    except KeyError:
        raise CatalogError(f"Unknown hardware profile {key!r}; choose from {', '.join(data['gpu_profiles'])}") from None


def kernel_variant(key, torch_key, python_key, amd_targets=()):
    """The existing installer's wheel-family routing, shared with planning."""
    if key in ("wgpkernels", "wgpkernels_cu128", "wgpkernels_rocm10"):
        key = {"cu130": "wgpkernels", "cu128": "wgpkernels_cu128", "rocm10": "wgpkernels_rocm10"}.get(torch_key)
    elif key in ("gguf", "gguf_cu128", "gguf_rocm10"):
        key = {("cu130", "3.11"): "gguf", ("cu128", "3.10"): "gguf_cu128", ("rocm10", "3.12"): "gguf_rocm10"}.get((torch_key, python_key))
    if key in ("wgpkernels_rocm10", "gguf_rocm10") and not set(amd_targets).intersection(("gfx1200", "gfx1201")):
        return None
    return key


def component_commands(data, group, key, platform=None, python_key=None, amd_targets=()):
    """Translate the catalog's supported pip recipes into argv and environment."""
    try:
        entry = data["components"][group][key]
    except KeyError:
        raise CatalogError(f"Unknown component: {group}.{key}") from None
    command = resolve_command(entry["cmd"], platform)
    if command is None:
        return []
    if any(not re.fullmatch(r"gfx[0-9a-f]+", target) for target in amd_targets):
        raise CatalogError("AMD targets must be gfx architecture IDs, for example gfx1201")
    if "{device}" in command and not amd_targets:
        raise CatalogError(f"{group}.{key} requires explicit AMD targets, for example --amd-target gfx1201")
    command = command.replace("{py}", "cp" + (python_key or "").replace(".", ""))
    command = command.replace("{device}", ",".join(f"device-{target}" for target in amd_targets))
    try:
        tokens = shlex.split(command)
    except ValueError as exc:
        raise CatalogError(f"Invalid recipe for {group}.{key}: {exc}") from exc
    commands, args, environment = [], [], {}
    for token in [*tokens, "&&"]:
        if token == "&&":
            if not args:
                raise CatalogError(f"Empty recipe command for {group}.{key}")
            if args[:2] == ["pip", "install"]:
                args = args[2:]
            elif args[0] == "{pip}":
                args = args[1:]
            if not args:
                raise CatalogError(f"Empty package list for {group}.{key}")
            commands.append({"args": args, "environment": environment})
            args, environment = [], {}
        elif not args and token == "{pip}":
            args.append(token)
        elif not args and re.fullmatch(r"[A-Z_][A-Z_0-9]*=.*", token):
            name, value = token.split("=", 1)
            if name != "FLASH_ATTENTION_TRITON_AMD_ENABLE":
                raise CatalogError(f"Unsupported recipe environment variable: {name}")
            environment[name] = value
        elif token in ("|", "||", ";", ">", ">>", "<") or "{pip}" in token or re.search(r"\{[a-z_]+\}", token):
            raise CatalogError(f"Unsupported recipe syntax in {group}.{key}: {token}")
        else:
            args.append(token)
    return commands
