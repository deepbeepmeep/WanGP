# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Paged native workspace management, independent of Gradio component state."""
import hashlib
import json
import os
import zipfile
from urllib.parse import urlencode
from collections import Counter
from pathlib import Path

from shared.utils.downloads import register_download, register_file_download, stream_writer
from shared.utils.gallery_sources import GALLERY_SOURCES, SOURCES
from shared.utils.gallery_view import gallery_offset
from shared.utils.config_store import update_config
from shared.utils.workspaces import ARCHIVE_AFTER_DAYS_KEY, ARCHIVE_PERIODS, media_available
from shared.utils.workspace_storage import bundle_files


def retain_entries(gallery, prefix, indices, *, inplace=False):
    paths = gallery[prefix + 'file_list']
    selected = gallery[prefix + 'selected']
    retained = [paths[i] for i in indices]
    if inplace:
        paths[:] = retained
    else:
        gallery[prefix + 'file_list'] = retained
    settings_key = prefix + 'file_settings_list'
    if settings_key in gallery:
        # Reordering/ejecting entries must not read their media metadata.
        retained_settings = type(gallery[settings_key])()
        retained_settings.extend(list.__getitem__(gallery[settings_key], i) for i in indices)
        if inplace:
            gallery[settings_key][:] = list.__iter__(retained_settings)
        else:
            gallery[settings_key] = retained_settings
    gallery[prefix + 'selected'] = indices.index(selected) if selected in indices else min(selected, len(indices) - 1)
    gallery[prefix + 'last_selected'] = gallery[prefix + 'selected'] == len(indices) - 1


class WorkspaceViewer:
    page_size = 60

    def __init__(self, service):
        self.service = service
        self._cache = {}

    def retention(self, days=None):
        with self.service._mutation_lock:
            config = self.service._deps.get_server_config()
            if days is not None:
                if days not in {value for _, value in ARCHIVE_PERIODS}:
                    raise ValueError('Choose a supported workspace archive period.')
                filename = self.service._deps.controller._deps.get_server_config_filename()
                update_config(config, filename, {ARCHIVE_AFTER_DAYS_KEY: days})
            return {'days': config.get(ARCHIVE_AFTER_DAYS_KEY, 0), 'choices': ARCHIVE_PERIODS}

    def protect(self, workspace, protected):
        with self.service._mutation_lock:
            if workspace != self.service.workspace_id:
                raise ValueError('The active workspace changed. Please try again.')
            self.service.workspaces.protect(workspace, protected)
            self.service.publish_workspaces()
            return {'archive_protected': protected}

    def _catalog(self, source):
        if source not in SOURCES:
            raise ValueError('Choose one of the workspace galleries.')
        prefix = SOURCES[source].prefix
        gen = self.service._state['gen']
        paths = gen[prefix + 'file_list']
        available = [index for index, path in enumerate(paths) if media_available(path)]
        if len(available) != len(paths):
            retain_entries(gen, prefix, available, inplace=True)
        signature = (self.service.workspace_id, tuple(paths), self.service._deps.get_server_config()['clear_file_list'])
        cached = self._cache.get(source)
        if cached is None or cached['signature'] != signature:
            occurrences = Counter()
            entries = []
            for index, path in enumerate(paths):
                occurrences[path] += 1
                key = hashlib.sha256((path + '\0' + str(occurrences[path])).encode()).hexdigest()[:24]
                entries.append({'key': key, 'index': index, 'path': path})
            cached = {'signature': signature, 'revision': hashlib.sha256(repr(signature).encode()).hexdigest(), 'entries': entries, 'by_key': {entry['key']: entry for entry in entries}, 'prefix': prefix}
            self._cache[source] = cached
        return cached

    def page(self, workspace, source='video', page=0, selected=(), initial=False):
        with self.service._mutation_lock, self.service.gallery_lock:
            if workspace != self.service.workspace_id:
                raise ValueError('The active workspace changed. Reopen the workspace viewer.')
            gen = self.service._state['gen']
            if initial:
                source = gen['current_gallery_source']
            catalog = self._catalog(source)
            count = len(catalog['entries'])
            if initial:
                index = gen[catalog['prefix'] + 'selected']
                selected = [catalog['entries'][index]['key']] if 0 <= index < count else []
                page = index // self.page_size if selected else 0
            pages = max(1, (count + self.page_size - 1) // self.page_size)
            page = max(0, min(page, pages - 1))
            offset = gallery_offset(count, catalog['signature'][2])
            items = []
            for entry in catalog['entries'][page * self.page_size:(page + 1) * self.page_size]:
                path = entry['path']
                kind = self.service.gallery._detect_media_type(path)
                try:
                    stamp = os.stat(path).st_mtime_ns
                    download = register_file_download(path)
                except OSError:
                    download = None
                item = {'key': entry['key'], 'index': entry['index'], 'name': Path(path).name, 'kind': kind, 'visible': entry['index'] >= offset, 'url': download['url'] if download else None, 'size': download['size_bytes'] if download else None}
                if download and kind in ('image', 'video', '3d', 'world', 'app'):
                    item['thumbnail'] = 'deepy_api/workspace_viewer/thumbnail?' + urlencode({'workspace': workspace, 'source': source, 'key': entry['key'], 'v': stamp})
                items.append(item)
            record = self.service.workspaces.get(workspace)
            sources = [{'id': gallery.id, 'label': gallery.label, 'accept': gallery.accept} for gallery in GALLERY_SOURCES]
            directory = self.service.workspaces.media_dir(workspace)
            return {'workspace': workspace, 'storage_mode': record.get('storage_mode', 'shared'), 'folder': str(directory) if directory else None, 'sources': sources, 'name': record['name'], 'last_activity': record['last_activity'], 'archive_protected': record['archive_protected'], 'source': source, 'revision': catalog['revision'], 'page': page, 'pages': pages, 'total': count, 'visible': count - offset, 'items': items, 'selected': [key for key in selected if key in catalog['by_key']]}

    def migrate(self, workspace, mode, *, folder=None, confirmed=False):
        service = self.service
        with service._mutation_lock:
            if workspace != service.workspace_id:
                raise ValueError('The active workspace changed. Please try again.')
            service.require_idle()
            if service._state['gen'].get('queue'):
                raise ValueError('Finish or clear the generation queue before converting workspace storage.')
            preview = service.workspaces.migration_preview(workspace, mode, folder=folder)
            if preview['missing']:
                if confirmed:
                    raise ValueError('Restore missing media before converting this workspace.')
                return preview
            service.save_workspace()
            if not confirmed:
                return service.workspaces.migration_preview(workspace, mode, folder=folder)
            result = service.workspaces.migrate(workspace, mode, folder=folder)
            service._apply_workspace(workspace, associate=False)
            service.publish_workspaces()
            service.publish('gallery', service.gallery_snapshot())
            service.host_changed()
            return result

    def _entry(self, workspace, source, key):
        if workspace != self.service.workspace_id:
            raise ValueError('The active workspace changed. Reopen the workspace viewer.')
        entry = self._catalog(source)['by_key'].get(key)
        if entry is None:
            raise ValueError('This media is no longer in the workspace.')
        return entry

    def info(self, workspace, source, key):
        with self.service._mutation_lock:
            entry = self._entry(workspace, source, key)
            settings = self.service._state['gen'][SOURCES[source].settings_key][entry['index']]
            return {'html': self.service._deps.callbacks.emit_first('format_media_info', entry['path'], settings)}

    def thumbnail_path(self, workspace, source, key):
        with self.service._mutation_lock:
            return self._entry(workspace, source, key)['path']

    def player(self, workspace, source, key):
        from shared.gradio.strip_gallery import AppGallery, MeshGallery, WorldGallery, gradio_file_url

        path = self.thumbnail_path(workspace, source, key)
        kind = self.service.gallery._detect_media_type(path)
        if kind == '3d':
            html = MeshGallery.viewer_html(path, gradio_file_url(path))
        elif kind == 'world':
            html = WorldGallery.viewer_html(path)
        elif kind == 'app':
            html = AppGallery.viewer_html(path)
        else:
            raise ValueError('This media uses the browser media player.')
        return {'html': html}

    def action(self, action, payload):
        service = self.service
        with service._mutation_lock, service.gallery_lock:
            workspace, source = payload['workspace'], payload['source']
            if workspace != service.workspace_id:
                raise ValueError('The active workspace changed. Reopen the workspace viewer.')
            catalog = self._catalog(source)
            if payload['revision'] != catalog['revision']:
                raise ValueError('The gallery changed on another page. Refresh your selection and try again.')
            keys = payload['keys']
            if not isinstance(keys, list) or not keys or any(not isinstance(key, str) or key not in catalog['by_key'] for key in keys):
                raise ValueError('Select media from this workspace first.')
            chosen = set(keys)
            entries = [entry for entry in catalog['entries'] if entry['key'] in chosen]
            prefix, gen = catalog['prefix'], service._state['gen']
            if action == 'archive':
                name = payload.get('name', '').strip()
                if not name or len(name) > 120 or any(char in name for char in '/\\:*?"<>|') or name in ('.', '..'):
                    raise ValueError('Enter a valid ZIP filename, without a folder path.')
                paths = [entry['path'] for entry in entries]
                def write_zip(output):
                    names = set()
                    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_STORED) as archive:
                        for path in paths:
                            bundle = bundle_files(path)
                            basename = Path(path).name
                            stem, suffix, index = Path(basename).stem, Path(basename).suffix, 1
                            while any((Path(basename).stem + part).casefold() in names for part in bundle):
                                index += 1
                                basename = f'{stem} ({index}){suffix}'
                            for part, member in bundle.items():
                                member_name = Path(basename).stem + part
                                names.add(member_name.casefold())
                                archive.write(member, member_name)
                            companions = [part for part in bundle if part != Path(path).suffix]
                            if companions:
                                ledger_name = basename + '.wangp-files.json'
                                names.add(ledger_name.casefold())
                                archive.writestr(ledger_name, json.dumps(companions))
                return json.loads(register_download(name if name.lower().endswith('.zip') else name + '.zip', 'application/zip', lambda: stream_writer(write_zip)))
            if action in ('copy', 'move'):
                service.require_idle()
                target = payload.get('target')
                if target == workspace:
                    raise ValueError('Choose another workspace.')
                with service.workspaces.lock:
                    record = service.workspaces.get(target)
                    saved = dict(record['gallery'])
                    saved[prefix + 'file_list'] = list(saved.get(prefix + 'file_list', [])) + [entry['path'] for entry in entries]
                    saved[prefix + 'selected'] = len(saved[prefix + 'file_list']) - 1
                    saved[prefix + 'last_selected'] = True
                    saved['current_gallery_source'] = source
                    service.workspaces.save(target, saved)
                service.publish_workspaces()
                if action == 'copy':
                    return {'count': len(entries)}
            if action not in ('reorder', 'eject', 'delete', 'move'):
                raise ValueError('Unknown workspace media action.')
            if action == 'delete':
                service.require_idle()
                if payload.get('confirmed') is not True:
                    raise ValueError('Confirm permanent file deletion first.')
                # A file may be referenced more than once or by several workspaces.
                removed, failures = set(), []
                for entry in entries:
                    path = os.path.abspath(entry['path'])
                    try:
                        main = Path(path).resolve()
                        companions = [member for member in bundle_files(path).values() if member != main]
                        for member in [*companions, main]:
                            member.unlink(missing_ok=True)
                        Path(path + '.wangp-files.json').unlink(missing_ok=True)
                        removed.add(os.path.normcase(path))
                    except OSError as exc:
                        failures.append(f'{Path(path).name}: {exc.strerror}')
                for group in (gallery.prefix for gallery in GALLERY_SOURCES):
                    retain_entries(gen, group, [i for i, path in enumerate(gen[group + 'file_list']) if os.path.normcase(os.path.abspath(path)) not in removed], inplace=True)
                with service.workspaces.lock:
                    for target in service.workspaces.catalog():
                        saved = dict(service.workspaces.get(target['id'])['gallery'])
                        for group in (gallery.prefix for gallery in GALLERY_SOURCES):
                            paths = saved.get(group + 'file_list', [])
                            indices = [i for i, path in enumerate(paths) if not isinstance(path, str) or os.path.normcase(os.path.abspath(path)) not in removed]
                            if len(indices) != len(paths):
                                retain_entries(saved, group, indices)
                        service.workspaces.save(target['id'], saved, retain_missing=False)
            else:
                indices = [entry['index'] for entry in catalog['entries'] if entry['key'] not in chosen]
                if action == 'reorder':
                    before = payload.get('before')
                    if before is not None and before not in catalog['by_key']:
                        raise ValueError('The drop target no longer exists.')
                    if before in chosen:
                        return {'count': 0}
                    position = 0 if payload.get('edge') == 'start' else len(indices) if before is None else indices.index(catalog['by_key'][before]['index'])
                    indices[position:position] = [entry['index'] for entry in entries]
                retain_entries(gen, prefix, indices, inplace=True)
                if action == 'reorder':
                    # The main gallery shows the newest bounded window. A
                    # reordered selection can leave it; keep its preview open
                    # on the nearest remaining visible item in that case.
                    offset = gallery_offset(len(indices), service._deps.get_server_config().get('clear_file_list', 5))
                    if 0 <= gen[prefix + 'selected'] < offset:
                        gen[prefix + 'selected'] = offset
                        gen[prefix + 'last_selected'] = offset == len(indices) - 1
                failures = []
            gen['selected_video_time'] = None
            service.publish('gallery', service.gallery_snapshot())
            service.host_changed()
            return {'count': len(entries), 'errors': failures}
