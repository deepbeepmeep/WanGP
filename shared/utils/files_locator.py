# Copyright (c) 2025-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations
from functools import lru_cache
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Iterable, List, Optional, Union

default_checkpoints_paths = ["ckpts", "."]

_checkpoints_paths = default_checkpoints_paths


def _is_probable_url(value: str) -> bool:
    return "://" in str(value) or str(value).startswith(("mailto:", "urn:"))


def _absolute_normalized_path(path: Union[str, os.PathLike]) -> str:
    return os.path.abspath(os.path.normpath(os.path.expanduser(os.fspath(path))))


def _checkpoint_roots() -> list[str]:
    roots = []
    seen = set()
    for root in _checkpoints_paths:
        normalized = _absolute_normalized_path(root)
        key = normalized.casefold()
        if key not in seen:
            roots.append(normalized)
            seen.add(key)
    return roots


def _is_under_root(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([path, root]).casefold() == root.casefold()
    except ValueError:
        return False


def compress_path(path: Union[str, os.PathLike]) -> str:
    """Store checkpoint-root paths as relative paths; leave URLs unchanged."""
    if path is None:
        return ""
    value = os.fspath(path).strip()
    if not value or _is_probable_url(value) or value.startswith("="):
        return value
    if not os.path.isabs(value):
        normalized_relative = os.path.normpath(value)
        if is_relative_down_path(normalized_relative):
            return normalized_relative.replace("\\", "/")
        normalized = _absolute_normalized_path(value)
    else:
        normalized = _absolute_normalized_path(value)
    for root in sorted(_checkpoint_roots(), key=len, reverse=True):
        if not _is_under_root(normalized, root):
            continue
        relative = os.path.relpath(normalized, root)
        if relative and relative != ".":
            return relative.replace("\\", "/")
    return normalized


def uncompress_path(path: Union[str, os.PathLike]) -> str:
    """Return an absolute local path for checkpoint-relative values; leave URLs unchanged."""
    if path is None:
        return ""
    value = os.fspath(path).strip()
    if not value or _is_probable_url(value) or value.startswith("="):
        return value
    if os.path.isabs(value):
        return _absolute_normalized_path(value)
    if not is_relative_down_path(value):
        return _absolute_normalized_path(value)
    located = locate_file(value, error_if_none=False)
    if located is not None:
        return _absolute_normalized_path(located)
    roots = _checkpoint_roots()
    return _absolute_normalized_path(os.path.join(roots[0] if roots else ".", value))


def compress_paths(paths):
    if isinstance(paths, (list, tuple)):
        return [compress_path(path) for path in paths]
    return compress_path(paths)


def uncompress_paths(paths):
    if isinstance(paths, (list, tuple)):
        return [uncompress_path(path) for path in paths]
    return uncompress_path(paths)

@lru_cache(maxsize=4096)
def _is_relative_down_path_cached(path: str) -> bool:
    if len(path) == 0 or "\x00" in path:
        return False
    windows_path = PureWindowsPath(path)
    posix_path = PurePosixPath(path)
    if windows_path.drive or windows_path.root or posix_path.root:
        return False
    if ".." in windows_path.parts or ".." in posix_path.parts:
        return False
    return any(part not in ("", ".") for part in path.replace("\\", "/").split("/"))

def is_relative_down_path(path: Union[str, os.PathLike]) -> bool:
    """Return True for relative paths that cannot escape a base folder."""
    try:
        path = os.fspath(path).strip()
    except TypeError:
        return False
    if not isinstance(path, str):
        return False
    return _is_relative_down_path_cached(path)

def clean_relative_path(path, trigger_error = True):
    if path=="" or path is None: return path
    if is_relative_down_path(path): return path
    if not trigger_error: return ""
    raise Exception(f"Unsafe relative path found : '{path}'")

def set_checkpoints_paths(checkpoints_paths):
    global _checkpoints_paths
    _checkpoints_paths = [path.strip() for path in checkpoints_paths if len(path.strip()) > 0 ]
    if len(checkpoints_paths) == 0:
        _checkpoints_paths = default_checkpoints_paths

def _normalize_force_path(force_path):
    if force_path is not None and isinstance(force_path, list) and len(force_path):
        force_path = force_path[0]
    if force_path is None:
        return None
    force_path = os.fspath(force_path).strip()
    if len(force_path) == 0:
        return None
    normalized = os.path.normpath(force_path)
    return None if normalized in ("", ".") else normalized

def extract_alternate_path(url, lora_dir = None):
    if not url.startswith("http"):
        if "|" in url:
            raise f"local path {url} can't contain a '|'"
        return url
    
    path_parts = url.split("|")
    new_url = os.path.basename(path_parts[0]) 
    if len(path_parts) == 1: return new_url
    if len(path_parts) != 2: raise f"Invalid path {url}"
    alternate_path = clean_relative_path(path_parts[1])
    if alternate_path == "%lora_dir":
        if lora_dir is None:
            raise Exception(f"Unable to compute %lora_dir in {url}, no lora_dir was provided")
        alternate_path = os.path.abspath(lora_dir)
    return os.path.join(alternate_path, new_url) 

def get_download_location(file_name = None, force_path= None, lora_dir = None):
    if file_name is not None:
        file_name = extract_alternate_path(file_name, lora_dir)
        if os.path.isabs(file_name): return file_name
    if force_path is not None and isinstance(force_path, list) and len(force_path): force_path = force_path[0]
    if file_name is not None:
        if force_path is None:
            return os.path.join(_checkpoints_paths[0], file_name)
        else:
            return os.path.join(_checkpoints_paths[0], force_path, file_name)
    else:
        if force_path is None:
            return _checkpoints_paths[0]
        else:
            return os.path.join(_checkpoints_paths[0], force_path,)

def get_smart_download_root(force_path = None):
    force_path = _normalize_force_path(force_path)
    if force_path is None:
        return _checkpoints_paths[0]
    if os.path.isabs(force_path):
        return force_path
    for folder in _checkpoints_paths:
        candidate = os.path.join(folder, force_path)
        if os.path.isdir(candidate):
            return folder
    return _checkpoints_paths[0]

def get_smart_download_location(file_name = None, force_path = None):
    if file_name is not None:
        file_name = extract_alternate_path(file_name)
        if os.path.isabs(file_name):
            return file_name
    force_path = _normalize_force_path(force_path)
    if force_path is None:
        return get_download_location(file_name)
    if os.path.isabs(force_path):
        return force_path if file_name is None else os.path.join(force_path, file_name)
    root = get_smart_download_root(force_path)
    base_path = os.path.join(root, force_path)
    return base_path if file_name is None else os.path.join(base_path, file_name)

def locate_folder(folder_name, error_if_none = True):
    searched_locations = []
    if os.path.isabs(folder_name):
        if os.path.isdir(folder_name): return folder_name
        searched_locations.append(folder_name)
    else:
        for folder in _checkpoints_paths:
            path = os.path.join(folder, folder_name)
            if os.path.isdir(path):
                return path
            searched_locations.append(os.path.abspath(path))
    if error_if_none: raise Exception(f"Unable to locate folder '{folder_name}', tried {searched_locations}")    
    return None


def installed_copies(relative_path):
    """All installed copies of one declared checkpoint path, including lower-priority roots."""
    candidates = [relative_path] if os.path.isabs(relative_path) else [os.path.join(root, relative_path) for root in _checkpoints_paths]
    return [candidate for candidate in candidates if os.path.isfile(candidate)]


def locate_file(file_name, create_path_if_none = False, error_if_none = True, extra_paths = None):
    if file_name.startswith("http"):
        file_name = os.path.basename(file_name)
    searched_locations = []
    if os.path.isabs(file_name):
        if os.path.isfile(file_name): return file_name
        searched_locations.append(file_name)
    else:
        for folder in _checkpoints_paths + ([] if extra_paths is None else extra_paths):
            path = os.path.join(folder, file_name)
            if os.path.isfile(path):
                return path
            searched_locations.append(os.path.abspath(path))
    
    if create_path_if_none:
        return get_download_location(file_name)
    if error_if_none: raise Exception(f"Unable to locate file '{file_name}', tried {searched_locations}")
    return None

def get_local_model_filename(model_filename, use_locator = True, extra_paths = None, lora_dir = None):
    local_model_filename = extract_alternate_path(model_filename, lora_dir)
    if use_locator:
        if extra_paths is not None and not os.path.isabs(local_model_filename):
            if not isinstance(extra_paths, list): extra_paths = [extra_paths]
            for path in extra_paths:
                filename = locate_file(os.path.join(path, local_model_filename), error_if_none= False)
                if filename is not None: return filename
        local_model_filename = locate_file(local_model_filename, error_if_none= False )
    return local_model_filename
