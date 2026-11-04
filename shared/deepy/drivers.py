# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Adapters between Deepy's commands and its host interfaces."""
from __future__ import annotations

import time
import hashlib
import os
from pathlib import Path

from shared.deepy.gallery import Gallery
from shared.deepy.media_registry import _label_from_settings
from shared.utils.gallery_sources import GALLERY_SOURCES, gallery_source
from shared.utils.media_settings import peek_settings
from shared.utils.mesh_outputs import splat_viewer_options


def gradio_chat_updates(commands, refresh_id):
    import gradio as gr

    first_publication = True
    for command, data in commands:
        outputs = [gr.update() for _ in range(5)]
        if command == "chat_output":
            outputs[0] = data if data is not None else gr.update()
            if first_publication:
                outputs[2] = gr.update(value="")
                first_publication = False
        elif command == "request_accepted":
            outputs[2] = gr.update(value="")
        elif command in {"load_queue_trigger", "refresh_gallery"}:
            outputs[1 if command == "load_queue_trigger" else 3] = str(time.time()) + "_" + str(refresh_id())
        elif command == "abort_client_id":
            outputs[4] = str(data or "")
        yield tuple(outputs)


class AppGallery(Gallery):
    """The API operates on the same media data as the CLI and Deepy tools."""

    def __init__(self, deps, state):
        super().__init__(deps, state)
        self._media_paths = {}
        self._snapshot_gen = self._gen()

    def media_info(self, media_id):
        source_id, index, path = self._media_paths[media_id]
        settings = self._snapshot_gen[gallery_source(source_id).settings_key]
        return self._deps.callbacks.emit_first("format_media_info", path, settings[index])

    def media_path(self, media_id):
        return self._media_paths[media_id][2]

    @staticmethod
    def media_id(path):
        return 'file_' + hashlib.sha256(os.path.normcase(os.path.abspath(path)).encode()).hexdigest()[:24]

    def select(self, reference, media_type="any"):
        if reference not in self._media_paths:
            if reference.startswith('file_'):
                return None
            return super().select(reference, media_type)
        source_id, index, path = self._media_paths[reference]
        self._select_index(index, source_id)
        return self._register(path, self._resolve_lists(source_id)[1][index])

    def snapshot(self, gen=None):
        gen = self._gen() if gen is None else gen
        labels = {record['path_key']: record['label'] for record in self._session.media_registry}
        items, paths = [], {}
        for source in GALLERY_SOURCES:
            files, settings = gen[source.files_key], gen[source.settings_key]
            for index, path in enumerate(files):
                try:
                    stat = os.stat(path)
                except FileNotFoundError:
                    continue
                media_id = self.media_id(path)
                paths[media_id] = (source.id, index, path)
                kind = self._detect_media_type(path)
                base = 'deepy_api/media/' + media_id
                version = f'?v={stat.st_mtime_ns}-{stat.st_size}'
                item = {'id': media_id, 'name': Path(path).name, 'kind': kind, 'url': base + '/file' + version, 'size': stat.st_size, 'selected': index == gen[source.selected_key], 'active': gen['current_gallery_source'] == source.id}
                if kind in ('image', 'video', '3d'):
                    item['thumbnail'] = base + '/thumbnail' + version
                if kind == '3d':
                    item['format'] = Path(path).suffix.lower().lstrip('.')
                    if item['format'] == 'ply':
                        item.update(splat_viewer_options(path))
                cached = peek_settings(settings, index)
                item['name'] = _label_from_settings(cached, path) or labels.get(os.path.normcase(os.path.abspath(path)), item['name'])
                if cached:
                    item['summary'] = {key: cached[key] for key in ('resolution', 'duration', 'fps', 'model_type') if key in cached}
                items.append(item)
        self._media_paths = paths
        self._snapshot_gen = gen
        return items


class GradioGallery(AppGallery):
    """An in-process driver for a live WebUI state; rendering stays in Gradio.

    The refresh callback requests a view update, never starts generation. This
    allows the same API to attach to the WebUI when hybrid mode is enabled.
    """

    def __init__(self, deps, state, refresh):
        super().__init__(deps, state)
        self._refresh = refresh

    def add_path(self, raw_path, preferred_type="any"):
        result = super().add_path(raw_path, preferred_type)
        self._refresh()
        return result

    def select(self, reference, media_type="any"):
        result = super().select(reference, media_type)
        self._refresh()
        return result

    def sync_refresh_path(self, payload):
        super().sync_refresh_path(payload)
        self._refresh()

    def sync_latest_generated(self, before_counts):
        super().sync_latest_generated(before_counts)
        self._refresh()
