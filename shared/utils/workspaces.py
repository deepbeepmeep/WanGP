# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Persistent galleries with shared references or independently owned media."""
import os
import threading
import time
import uuid
import copy
from collections import Counter
from pathlib import Path

from shared.utils.config_store import read_config, write_config
from shared.utils.gallery_sources import GALLERY_SOURCES, SOURCES
from shared.utils.media_settings import MediaSettings
from shared.utils.workspace_storage import MODES, bundle_files, copy_bundle, owned_path
from shared.utils.readable_paths import available_path, safe_name


ARCHIVE_AFTER_DAYS_KEY = 'workspace_archive_after_days'
ARCHIVE_PERIODS = [
    ('Disabled', 0), ('1 week', 7), ('2 weeks', 14), ('3 weeks', 21),
    *[(f'{months} month' + ('s' if months > 1 else ''), months * 30) for months in range(1, 12)], ('1 year', 365),
]


def migrate_session_references(root):
    """Convert active and archived gallery references without reading media or chat history."""
    from shared.deepy.session_store import SessionReferences
    references = SessionReferences()
    root = Path(root)
    blocked = set()
    for parent in (root, root / 'archives'):
        for path in parent.glob('*.json'):
            if not path.is_file():
                continue
            record = read_config(path)
            if record.get('storage_mode', 'shared') == 'shared':
                _convert_references(path, record, references)
            if 'folder' in record:
                folder = Path(record['folder'])
                folder = folder if folder.is_absolute() else root / folder
                for identity, (directory, _manifest) in references.records.items():
                    if folder.resolve().is_relative_to(directory):
                        blocked.add(identity)
    return blocked


def _convert_references(path, record, references):
    converted = WorkspaceStore._map_paths(record, references.encode)
    if converted != record:
        stat = path.stat()
        write_config(converted, path)
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    return converted


def media_available(path):
    return isinstance(path, str) and os.path.isfile(path)


def _media_key(path):
    return (path['deepy_session_id'], path['path']) if isinstance(path, dict) else path


class WorkspaceStore:
    def __init__(self, root, default_mode='shared'):
        if default_mode not in MODES:
            raise ValueError('Choose shared or dedicated workspace storage.')
        self.default_mode = default_mode
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.records = {}
        self._paths = {}
        self.session_workspaces = {}
        from shared.deepy.session_store import SessionReferences
        references = self._references = SessionReferences()
        for path in sorted(self.root.glob('*.json')):
            if not path.is_file():
                continue
            record = read_config(path)
            if not isinstance(record, dict) or not isinstance(record.get('id'), str) or not record['id'] or not isinstance(record.get('name'), str) or not isinstance(record.get('gallery'), dict):
                raise ValueError(f'Invalid workspace: {path.name}')
            if record['id'] in self.records:
                raise ValueError(f'Duplicate workspace ID: {path.name}')
            # Older manifests have no content-activity timestamp. Seed it from
            # their last saved date without rewriting every file at startup.
            record.setdefault('last_activity', path.stat().st_mtime)
            record.setdefault('archive_protected', False)
            if record.get('storage_mode', 'shared') not in MODES:
                raise ValueError(f'Invalid workspace storage: {path.name}')
            if record.get('storage_mode') == 'dedicated':
                directory = self._directory(record)
                resolved = directory.resolve()  # once for the workspace's files
                record = self._map_paths(record, lambda value: str(owned_path(directory, value, resolved)))
            else:
                converted = _convert_references(path, record, references)
                record = converted
            if not isinstance(record['last_activity'], (int, float)) or not isinstance(record['archive_protected'], bool):
                raise ValueError(f'Invalid workspace activity or protection: {path.name}')
            self.records[record['id']] = record
            self._paths[record['id']] = path
            owner = record.get('deepy_session_id')
            if owner is not None:
                if not isinstance(owner, str) or not owner or owner in self.session_workspaces:
                    raise ValueError(f'Invalid Deepy workspace ownership: {path.name}')
                self.session_workspaces[owner] = record['id']
        for identity, record in self.records.items():
            self._paths[identity] = self._rename_manifest(self._paths[identity], record)
        for path in sorted((self.root / 'archives').glob('*.json')):
            if path.is_file():
                record = read_config(path)
                if record.get('storage_mode', 'shared') == 'shared':
                    record = _convert_references(path, record, references)
                self._rename_manifest(path, record)
        if not self.records:
            self.create('Default')

    @staticmethod
    def _rename_manifest(path, record):
        # Keep an allocated numbered name stable, even when earlier numbers become free.
        stem = safe_name(record['name'])
        number = path.stem.removeprefix(stem + ' (').removesuffix(')')
        if path.stem == stem or (path.stem == f'{stem} ({number})' and number.isdecimal() and int(number) >= 2):
            return path
        destination = available_path(path.parent, record['name'], suffix='.json', exclude=path)
        path.rename(destination)
        return destination

    def _new_folder(self, record):
        others = [*self.records.values(), *(read_config(path) for path in (self.root / 'archives').glob('*.json') if path.is_file())]
        reserved = ['archives', '.agents']
        for other in others:
            if 'folder' in other:
                directory = self._directory(other)
                if directory.is_relative_to(self.root):
                    reserved.append(directory.relative_to(self.root).parts[0])
        return available_path(self.root, record['name'], reserved=reserved).name

    def catalog(self):
        with self.lock:
            records = sorted(self.records.values(), key=lambda record: (-record['last_activity'], record['name'].casefold()))
            return [{key: record[key] for key in ('id', 'name', 'deepy_session_id', 'mcp_agent_id', 'storage_mode') if key in record} for record in records]

    def _directory(self, record):
        folder = record['folder']
        if not isinstance(folder, str) or not folder:
            raise ValueError('Invalid workspace folder.')
        path = Path(folder)
        if not path.is_absolute():
            if len(path.parts) != 1 or folder in ('.', '..') or path.drive or any(char in folder for char in '/\\:'):
                raise ValueError('Invalid workspace folder.')
            path = self.root / folder
        directory = path.resolve()
        if directory != Path(os.path.abspath(path)):
            raise ValueError('A workspace media folder cannot be a symbolic link or junction.')
        if directory == Path(directory.anchor) or self.root.is_relative_to(directory) or any(directory.is_relative_to(self.root / name) for name in ('archives', '.agents')):
            raise ValueError('Choose a media folder outside the workspace metadata and archive folders.')
        return directory

    def media_dir(self, workspace_id):
        record = self.get(workspace_id)
        return self._directory(record) if record.get('storage_mode', 'shared') == 'dedicated' else None

    @staticmethod
    def _map_paths(record, transform):
        result = {**record, 'gallery': dict(record['gallery'])}
        for source in GALLERY_SOURCES:
            if source.files_key in result['gallery']:
                result['gallery'][source.files_key] = [transform(path) for path in result['gallery'][source.files_key]]
        return result

    def adopt(self, workspace_id, paths, *, created=None):
        directory = self.media_dir(workspace_id)
        if directory is None:
            return [os.path.abspath(path) for path in paths]
        with self.lock:
            copied, result = [], []
            try:
                for path in paths:
                    destination, files = copy_bundle(path, directory)
                    copied.extend(files)
                    result.append(destination)
                if created is not None:
                    created.extend(copied)
                return result
            except BaseException:
                for path in reversed(copied):
                    path.unlink(missing_ok=True)
                raise

    def _name(self, name, excluding=None, owner=None):
        if not isinstance(name, str) or not name.strip() or len(name.strip()) > 120:
            raise ValueError('Enter a workspace name between 1 and 120 characters.')
        name = name.strip()
        if not owner and any(record['id'] != excluding and not (record.get('deepy_session_id') or record.get('mcp_agent_id')) and record['name'].casefold() == name.casefold() for record in self.records.values()):
            raise ValueError('A workspace with this name already exists.')
        return name

    def get(self, workspace_id):
        if not isinstance(workspace_id, str) or workspace_id not in self.records:
            raise ValueError('This workspace no longer exists.')
        record = self.records[workspace_id]
        if any(isinstance(path, dict) for source in GALLERY_SOURCES for path in record['gallery'].get(source.files_key, [])):
            from shared.deepy.session_store import SessionReferences
            record = self._map_paths(record, SessionReferences().path)
        return record

    def _write(self, record):
        saved = record
        if record.get('storage_mode') == 'dedicated':
            directory = self._directory(record)
            saved = self._map_paths(record, lambda value: Path(value).resolve().relative_to(directory).as_posix())
        else:
            from shared.deepy.session_store import SessionReferences
            references = SessionReferences()
            # Retain known identities if a session was removed while this store was open.
            references.roots = {**self._references.roots, **references.roots}
            self._references = references
            saved = self._map_paths(record, references.encode)
        previous = self._paths.get(record['id'])
        path = previous or available_path(self.root, record['name'], suffix='.json')
        if previous is not None and self.records[record['id']]['name'] != record['name']:
            path = self._rename_manifest(previous, record)
        try:
            write_config(saved, path)
        except BaseException:
            if previous is not None and path.name != previous.name:
                path.rename(previous)
            raise
        self._paths[record['id']] = path
        self.records[record['id']] = saved if record.get('storage_mode', 'shared') == 'shared' else record

    def for_session(self, session_id):
        with self.lock:
            workspace_id = self.session_workspaces.get(session_id)
            return self.get(workspace_id) if workspace_id else None

    def create(self, name, *, deepy_session_id=None, mcp_agent_id=None, storage_mode=None, folder=None):
        """A given absolute folder (a plugin's Deepy session folder) makes the workspace dedicated and may already hold its media."""
        with self.lock:
            if deepy_session_id and deepy_session_id in self.session_workspaces:
                raise ValueError('This Deepy session already has a dedicated workspace.')
            mode = 'dedicated' if folder is not None else storage_mode or self.default_mode
            if mode not in MODES:
                raise ValueError('Choose shared or dedicated workspace storage.')
            record = {'id': uuid.uuid4().hex, 'name': self._name(name, owner=deepy_session_id or mcp_agent_id), 'gallery': {}, 'last_activity': time.time(), 'archive_protected': False}
            if mode == 'dedicated':
                record.update(storage_mode=mode, folder=self._new_folder(record) if folder is None else str(folder))
                self._directory(record).mkdir(parents=True, exist_ok=folder is not None)
            if mcp_agent_id:
                record['mcp_agent_id'] = mcp_agent_id
            if deepy_session_id:
                record['deepy_session_id'] = deepy_session_id
            self._write(record)
            if deepy_session_id:
                self.session_workspaces[deepy_session_id] = record['id']
            return record['id']

    def lease(self, workspace_id, deepy_session_id):
        """Assign a workspace to a Deepy session for a plugin job without recording it as its owner."""
        with self.lock:
            self.get(workspace_id)
            if deepy_session_id in self.session_workspaces:
                raise ValueError('This Deepy session already has a workspace.')
            self.session_workspaces[deepy_session_id] = workspace_id

    def end_lease(self, deepy_session_id):
        with self.lock:
            del self.session_workspaces[deepy_session_id]

    def rename(self, workspace_id, name, *, deepy_session_id=None):
        with self.lock:
            record = self.get(workspace_id)
            if record.get('deepy_session_id') and record['deepy_session_id'] != deepy_session_id:
                raise ValueError('Rename this workspace through its Deepy session.')
            self._write({**record, 'name': self._name(name, workspace_id, record.get('deepy_session_id') or record.get('mcp_agent_id'))})

    def delete(self, workspace_id, *, deepy_session_id=None):
        with self.lock:
            record = self.get(workspace_id)
            owner = record.get('deepy_session_id')
            if owner and owner != deepy_session_id:
                raise ValueError('Delete this workspace through its Deepy session.')
            self._paths[workspace_id].unlink()
            del self._paths[workspace_id]
            del self.records[workspace_id]
            if owner:
                del self.session_workspaces[owner]
            if not self.records:
                self.create('Default')

    def save(self, workspace_id, gallery, *, retain_missing=True):
        with self.lock:
            record = self.get(workspace_id)
            created = []
            try:
                gallery = copy.deepcopy(gallery)
                if retain_missing:
                    # Opening/saving a filtered gallery must not erase unavailable media.
                    for source in GALLERY_SOURCES:
                        if source.files_key not in gallery and source.files_key not in record['gallery']:
                            continue
                        paths = gallery.setdefault(source.files_key, [])
                        selected = gallery.get(source.selected_key, -1)
                        counts, previous_counts = Counter(map(_media_key, paths)), Counter()
                        for index, path in enumerate(record['gallery'].get(source.files_key, [])):
                            key = _media_key(path)
                            previous_counts[key] += 1
                            if not media_available(path) and counts[key] < previous_counts[key]:
                                position = min(index, len(paths))
                                paths.insert(position, path)
                                counts[key] += 1
                                if position <= selected:
                                    selected += 1
                        if source.selected_key in gallery:
                            gallery[source.selected_key] = selected
                if record.get('storage_mode') == 'dedicated':
                    for source in GALLERY_SOURCES:
                        if source.files_key in gallery:
                            gallery[source.files_key] = [self.adopt(workspace_id, [path], created=created)[0] if media_available(path) else path for path in gallery[source.files_key]]
                if record['gallery'] != gallery:
                    changed = any(record['gallery'].get(source.files_key, []) != gallery.get(source.files_key, []) for source in GALLERY_SOURCES)
                    self._write({**record, 'gallery': gallery, 'last_activity': time.time() if changed else record['last_activity']})
                    return changed
            except BaseException:
                for path in reversed(created):
                    path.unlink(missing_ok=True)
                raise

    def migration_preview(self, workspace_id, mode, *, folder=None):
        if mode not in MODES:
            raise ValueError('Choose shared or dedicated workspace storage.')
        record = self.get(workspace_id)
        if folder is not None and (mode != 'dedicated' or not isinstance(folder, str) or not Path(folder).is_absolute()):
            raise ValueError('Enter an absolute server folder path for dedicated storage.')
        directory = None
        if mode == 'dedicated':
            directory = self._directory({**record, 'folder': folder if folder is not None else record.get('folder') or self._new_folder(record)})
            if directory.exists() and not directory.is_dir():
                raise ValueError('The target path is not a folder.')
            others = [*self.records.values(), *(read_config(path) for path in (self.root / 'archives').glob('*.json') if path.is_file())]
            for other in others:
                if other['id'] != workspace_id and 'folder' in other:
                    other_dir = self._directory(other)
                    if directory.is_relative_to(other_dir) or other_dir.is_relative_to(directory):
                        raise ValueError('The target overlaps another workspace folder. Choose a separate folder.')
            current = self.media_dir(workspace_id)
            if current and current != directory and (directory.is_relative_to(current) or current.is_relative_to(directory)):
                raise ValueError('The new folder cannot contain or be inside the current workspace folder.')
        paths = list({_media_key(path): path for source in GALLERY_SOURCES for path in record['gallery'].get(source.files_key, [])}.values())
        missing = [path for path in paths if not media_available(path)]
        files = {file for path in paths if path not in missing for file in bundle_files(path).values()}
        return {'workspace': workspace_id, 'mode': mode, 'current_mode': record.get('storage_mode', 'shared'), 'folder': str(directory) if directory else None, 'media_count': len(paths), 'file_count': len(files), 'bytes': sum(file.stat().st_size for file in files), 'missing': [f"Deepy {path['deepy_session_id']}: {path['path']}" if isinstance(path, dict) else path for path in missing], 'keeps_sources': True}

    def migrate(self, workspace_id, mode, *, folder=None):
        with self.lock:
            preview = self.migration_preview(workspace_id, mode, folder=folder)
            if preview['missing']:
                raise ValueError('Restore missing media before converting this workspace.')
            if preview['current_mode'] == mode and (mode == 'shared' or self.media_dir(workspace_id) == Path(preview['folder'])):
                return preview
            record = copy.deepcopy(self.get(workspace_id))
            record['storage_mode'] = mode
            created = []
            try:
                if mode == 'dedicated':
                    if folder is not None:
                        record['folder'] = preview['folder']
                    else:
                        record.setdefault('folder', Path(preview['folder']).name)
                    directory = self._directory(record)
                    directory.mkdir(parents=True, exist_ok=True)
                    mapping = {}
                    for source in GALLERY_SOURCES:
                        for path in record['gallery'].get(source.files_key, []):
                            if path not in mapping:
                                mapping[path], files = copy_bundle(path, directory)
                                created.extend(files)
                    record = self._map_paths(record, mapping.__getitem__)
                self._write(record)
            except BaseException:
                for path in reversed(created):
                    path.unlink(missing_ok=True)
                raise
            return preview

    def protect(self, workspace_id, protected):
        with self.lock:
            record = self.get(workspace_id)
            if record['archive_protected'] != protected:
                self._write({**record, 'archive_protected': protected})

    def expired(self, days):
        if type(days) is not int or days not in {value for _, value in ARCHIVE_PERIODS}:
            raise ValueError('Choose a supported workspace archive period.')
        if not days:
            return []
        cutoff = time.time() - days * 86400
        with self.lock:
            return [record['id'] for record in self.records.values() if not record['archive_protected'] and record['last_activity'] <= cutoff]

    def archive(self, workspace_id):
        with self.lock:
            record = self.get(workspace_id)
            archives = self.root / 'archives'
            archives.mkdir(exist_ok=True)
            if any(read_config(path)['id'] == workspace_id for path in archives.glob('*.json') if path.is_file()):
                raise ValueError(f'An archive already exists for workspace {record["name"]}.')
            destination = available_path(archives, record['name'], suffix='.json')
            self._paths[workspace_id].rename(destination)
            del self._paths[workspace_id]
            del self.records[workspace_id]
            if record.get('deepy_session_id'):
                del self.session_workspaces[record['deepy_session_id']]


def capture_gallery(gen):
    gallery = {key: gen[key] for key in ('current_gallery_source', 'selected_video_time')}
    for source in GALLERY_SOURCES:
        gallery.update({source.files_key: [os.path.abspath(path) for path in gen[source.files_key]], source.selected_key: gen[source.selected_key], source.last_selected_key: gen[source.last_selected_key]})
    return gallery


def load_gallery(record, read_settings):
    from shared.deepy.session_store import SessionReferences
    references = None
    saved = record['gallery']
    result = {'current_gallery_source': saved.get('current_gallery_source', 'video'), 'selected_video_time': saved.get('selected_video_time')}
    if result['current_gallery_source'] not in SOURCES:
        raise ValueError('Invalid workspace gallery tab.')
    # A workspace saved before a gallery existed has no entry for it.
    for prefix in (source.prefix for source in GALLERY_SOURCES):
        paths = saved.get(prefix + 'file_list', [])
        if not isinstance(paths, list):
            raise ValueError('Invalid workspace media paths.')
        if any(isinstance(path, dict) for path in paths):
            references = references or SessionReferences()
            paths = [references.path(path) for path in paths]
        elif any(not isinstance(path, str) for path in paths):
            raise ValueError('Invalid workspace media paths.')
        selected = saved.get(prefix + 'selected', -1)
        if not isinstance(selected, int):
            raise ValueError('Invalid workspace media selection.')
        selected_path = paths[selected] if 0 <= selected < len(paths) else None
        existing = [path for path in paths if media_available(path)]
        result[prefix + 'file_list'] = existing
        result[prefix + 'file_settings_list'] = MediaSettings(existing, read_settings)
        result[prefix + 'selected'] = existing.index(selected_path) if selected_path in existing else len(existing) - 1
        result[prefix + 'last_selected'] = bool(saved.get(prefix + 'last_selected', True))
    return result
