# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""
Packaging utilities for Wan2GP Generation Requests.

Packages GenerationRequests and their referenced media attachments (start images,
control videos, masks, audio tracks, etc.) into self-contained ZIP archives
ready for remote cloud workers (Modal, RunPod, etc.) and standalone execution.
"""
from __future__ import annotations

import base64
import copy
import io
import json
import os
import shutil
import tempfile
import zipfile
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from PIL import Image
except ImportError:
    Image = None

from shared.execution.types import ATTACHMENT_KEYS, EPHEMERAL_KEYS, GenerationRequest


def unwrap_attachment_item(item: Any) -> Any:
    """Unwrap complex Gradio or dictionary file references into local path or Image."""
    if isinstance(item, (tuple, list)) and len(item) > 0:
        item = item[0]
    if isinstance(item, dict):
        item = item.get("path") or item.get("name") or item.get("orig_name") or item.get("url") or item
    elif not isinstance(item, (str, bytes)):
        # Preserve PIL Image or file-like objects
        if Image is not None and isinstance(item, Image.Image):
            return item
        if hasattr(item, "path") and item.path:
            return item.path
        if hasattr(item, "name") and item.name:
            return item.name
        item = getattr(item, "path", None) or getattr(item, "name", None) or item
    return item


def _serialize_request_into_zip(
    request: GenerationRequest,
    resolved_id: Union[str, int],
    zf: zipfile.ZipFile,
    file_paths_in_zip: Dict[Any, str],
) -> Dict[str, Any]:
    """Helper to package media attachments of a single request into a zip."""
    params = request.to_params(include_runtime_state=False)

    all_attachment_keys = list(ATTACHMENT_KEYS)
    for k in request.inputs.keys():
        if k not in all_attachment_keys:
            all_attachment_keys.append(k)

    for key in all_attachment_keys:
        val = params.get(key)
        if val is None:
            continue

        is_list = isinstance(val, list)
        items = val if is_list else [val]

        processed_rel_names = []
        for item_idx, item in enumerate(items):
            item = unwrap_attachment_item(item)
            if item is None:
                continue

            # 1. PIL Image
            if Image is not None and isinstance(item, Image.Image):
                item_key = id(item)
                if item_key in file_paths_in_zip:
                    processed_rel_names.append(file_paths_in_zip[item_key])
                    continue
                arcname = f"task{resolved_id}_{key}_{item_idx}.png"
                img_buf = io.BytesIO()
                item.save(img_buf, format="PNG")
                zf.writestr(arcname, img_buf.getvalue())
                file_paths_in_zip[item_key] = arcname
                processed_rel_names.append(arcname)

            # 2. Local file on filesystem
            elif isinstance(item, str) and os.path.isfile(item):
                if item in file_paths_in_zip:
                    processed_rel_names.append(file_paths_in_zip[item])
                    continue
                _, ext = os.path.splitext(item)
                if not ext:
                    ext = ".bin"
                arcname = f"task{resolved_id}_{key}_{item_idx}{ext}"
                zf.write(item, arcname=arcname)
                file_paths_in_zip[item] = arcname
                processed_rel_names.append(arcname)

            # 3. Base64 encoded string data
            elif isinstance(item, str) and item.startswith("data:"):
                try:
                    header, b64_data = item.split(",", 1)
                    ext = ".bin"
                    if "image/" in header:
                        ext = "." + header.split("image/")[1].split(";")[0]
                    elif "video/" in header:
                        ext = "." + header.split("video/")[1].split(";")[0]
                    elif "audio/" in header:
                        ext = "." + header.split("audio/")[1].split(";")[0]
                    raw_bytes = base64.b64decode(b64_data)
                    arcname = f"task{resolved_id}_{key}_{item_idx}{ext}"
                    zf.writestr(arcname, raw_bytes)
                    processed_rel_names.append(arcname)
                except Exception:
                    processed_rel_names.append(item)

            # 4. Raw bytes
            elif isinstance(item, bytes):
                arcname = f"task{resolved_id}_{key}_{item_idx}.bin"
                zf.writestr(arcname, item)
                processed_rel_names.append(arcname)

            # 5. String that is already a relative filename, virtual source, etc.
            elif isinstance(item, str):
                processed_rel_names.append(item)

        if processed_rel_names:
            params[key] = processed_rel_names if is_list else processed_rel_names[0]

    # Clean runtime / non-setting / ephemeral keys
    disallowed = set(EPHEMERAL_KEYS) | {
        "state", "start_image_labels", "end_image_labels",
        "start_image_data_base64", "end_image_data_base64",
        "start_image_data", "end_image_data", "lset_name",
        "gpu_type", "engine", "version", "provider",
        "inputs", "plugin_data", "metadata", "execution_mode"
    }
    if params.get("resolution"):
        disallowed.add("width")
        disallowed.add("height")

    for bad_k in disallowed:
        params.pop(bad_k, None)

    params.setdefault("settings_version", 2.83)

    return {
        "id": resolved_id,
        "params": params,
        "plugin_data": request.plugin_data or {},
        "execution_mode": "cloud",
    }


def package_generation_request_to_zip(
    request: GenerationRequest,
    task_id: Optional[Union[str, int]] = None,
) -> bytes:
    """
    Package a single GenerationRequest and all media attachments into a binary ZIP bundle.
    """
    resolved_id = task_id or request.task_id or 1
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        file_paths_in_zip: Dict[Any, str] = {}
        manifest = [_serialize_request_into_zip(request, resolved_id, zf, file_paths_in_zip)]
        zf.writestr("queue.json", json.dumps(manifest, indent=2))
    return zip_buffer.getvalue()


def package_batch_requests_to_zip(
    requests: List[GenerationRequest],
) -> bytes:
    """
    Package multiple GenerationRequests into a single multi-task batch ZIP bundle.
    All tasks run sequentially on 1 warm GPU container without cold starts or model reloading.
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        file_paths_in_zip: Dict[Any, str] = {}
        manifests = []
        for idx, req in enumerate(requests, start=1):
            manifest_item = _serialize_request_into_zip(req, idx, zf, file_paths_in_zip)
            manifests.append(manifest_item)
        zf.writestr("queue.json", json.dumps(manifests, indent=2))
    return zip_buffer.getvalue()


def package_tasks_to_batch_zip(
    tasks: List[Dict[str, Any]],
    state: Any = None,
) -> bytes:
    """
    Package a list of Wan2GP queue task dicts into a single multi-task batch ZIP bundle.
    """
    requests = []
    for idx, t in enumerate(tasks, start=1):
        req = GenerationRequest.from_task(
            task=t,
            state=state,
            plugin_data=t.get("plugin_data"),
            execution_mode="cloud",
        )
        req.task_id = idx
        requests.append(req)
    return package_batch_requests_to_zip(requests)
