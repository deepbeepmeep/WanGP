from __future__ import annotations
import os
from pathlib import Path
from typing import Iterable, List, Optional, Union

default_checkpoints_paths = ["ckpts", "."]

# --- LOCAL-ONLY MODE (no downloads, only local models) ---
import re as _re
_local_only = os.environ.get("WGP_LOCAL_ONLY", "").strip().lower() in ("1", "true", "yes")

def set_local_only(v):
    global _local_only
    _local_only = bool(v)

def is_local_only():
    return _local_only

def load_local_paths_from_file(paths_file="models.txt"):
    """Extract model root folders (e.g. D:\\ComfyUiCuda\\models) from a models.txt file."""
    roots = []
    if not os.path.isfile(paths_file):
        return roots
    with open(paths_file, encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            m = _re.match(r"^([A-Za-z]:[\\/].+?[\\/](?:checkpoints|diffusion_models|loras|vae|text_encoders|LLM|unet|model_patches|controlnet|clip|vae_approx))[\\/]", line, _re.I)
            if m and m.group(1) not in roots:
                roots.append(m.group(1))
    return roots
# ----------------------------------------------------------

_checkpoints_paths = default_checkpoints_paths

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
    alternate_path = path_parts[1]
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
    if file_name is not None and os.path.isabs(file_name):
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
