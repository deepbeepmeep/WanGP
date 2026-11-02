# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import json
import os
import time
from contextlib import nullcontext
from datetime import datetime
from typing import Any, Callable

from shared.utils.audio_metadata import save_audio_metadata
from shared.utils.audio_video import save_image_metadata
from shared.utils.gallery_sources import APP, AUDIO, MESH, VIDEO, WORLD, source_for_path
from shared.utils.video_metadata import save_video_metadata
from shared.utils.workspace_storage import recording_gallery


def _ensure_creation_metadata(configs: Any) -> Any:
    if not isinstance(configs, dict):
        return configs
    saved_configs = configs.copy()
    if "creation_date" in saved_configs and "creation_timestamp" in saved_configs:
        return saved_configs
    recorded_time = time.time()
    if "creation_date" not in saved_configs:
        saved_configs["creation_date"] = datetime.fromtimestamp(recorded_time).isoformat(timespec="seconds")
    if "creation_timestamp" not in saved_configs:
        saved_configs["creation_timestamp"] = int(recorded_time)
    return saved_configs


def record_file_metadata(video_path: str | list[str], configs: Any, is_image: bool, audio_only: bool, gen: dict[str, Any], *, get_processed_queue: Callable[[dict[str, Any]], dict[str, tuple[list[Any], list[Any]]]], metadata_choice: str = "metadata", embedded_images: Any = None, replace_last_file: bool = False, lock: Any = None, verbose_level: int = 0, write_metadata: bool = True, output_role: str = "final") -> None:
    galleries = get_processed_queue(gen)
    paths = [video_path] if not isinstance(video_path, list) else video_path
    queue_lock = lock if lock is not None else nullcontext()
    for no, path in enumerate(paths):
        source = AUDIO if audio_only else source_for_path(path)
        previous_path = None
        saved_configs = _ensure_creation_metadata(configs)
        if configs is not None and write_metadata:
            saved_configs["output_role"] = output_role
            if metadata_choice in ("json", "json_metadata"):
                with open(os.path.splitext(path)[0] + ".json", "w") as f:
                    json.dump(saved_configs, f, indent=4)
            if metadata_choice in ("metadata", "json_metadata"):
                if audio_only:
                    save_audio_metadata(path, saved_configs)
                elif source is WORLD:
                    from shared.utils.world_outputs import save_world_metadata
                    save_world_metadata(path, saved_configs)
                elif source is APP:
                    from shared.utils.app_outputs import save_app_metadata
                    save_app_metadata(path, saved_configs)
                elif source is MESH:
                    from shared.utils.mesh_outputs import save_mesh_metadata
                    save_mesh_metadata(path, saved_configs, embedded_images)
                elif is_image:
                    save_image_metadata(path, saved_configs)
                else:
                    save_video_metadata(path, saved_configs, embedded_images, allow_inplace_update=True, verbose_level=verbose_level)
        if verbose_level > 0:
            action = "saved to" if write_metadata else "added to gallery from"
            if audio_only:
                print(f"Audio file {action} Path: {path}")
            elif source is MESH:
                print(f"3D asset {action} Path: {path}")
            elif source is WORLD:
                print(f"World {action} Path: {path}")
            elif source is APP:
                print(f"App {action} Path: {path}")
            elif is_image:
                print(f"Image file {action} Path: {path}")
            else:
                print(f"Video file {action} Path: {path}")
        with queue_lock:
            file_list, file_settings_list = galleries[source.id]
            if replace_last_file and source is VIDEO and not is_image and no == 0 and len(file_list) > 0:
                previous_path = file_list[-1]
                file_list[-1] = path
                file_settings_list[-1] = saved_configs
            else:
                file_list.append(path)
                file_settings_list.append(saved_configs)
            recording_gallery(gen)["last_gallery_source"] = source.id
        if previous_path is not None and previous_path != path:
            if metadata_choice in ("json", "json_metadata"):
                previous_json_path = os.path.splitext(previous_path)[0] + ".json"
                if os.path.isfile(previous_json_path):
                    os.remove(previous_json_path)
            if os.path.isfile(previous_path):
                os.remove(previous_path)
