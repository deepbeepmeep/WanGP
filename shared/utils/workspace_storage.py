# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Independent media bundles and execution-local workspace destinations."""
import hashlib
import json
import os
import shutil
import tempfile
import threading
import weakref
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

from shared.utils.gallery_sources import GALLERY_SOURCES

STORAGE_KEY = 'workspace_storage_mode'
ROOT_KEY = 'workspaces_dir'
MODES = ('shared', 'dedicated')
OUTPUT_KEYS = ('save_path', 'image_save_path', 'audio_save_path', 'mesh_save_path')
GALLERY_KEYS = ('current_gallery_source', 'selected_video_time', *(key for source in GALLERY_SOURCES for key in (source.files_key, source.settings_key, source.selected_key, source.last_selected_key)))
_hosts = weakref.WeakValueDictionary()
_execution = ContextVar('wangp_workspace', default=None)
_background = ContextVar('wangp_workspace_background', default=None)
_copy_lock = threading.RLock()
_live_lock = threading.Lock()
_live = {}  # (host id, workspace id) -> [gallery, users] while operations record outside the visible gallery


def register_host(host):
    _hosts[id(host._state['gen'])] = host


def host_for_gen(gen):
    return _hosts.get(id(gen))


def workspace_gallery(gen, workspace_id):
    """Read an assigned gallery without changing the visible/executing workspace."""
    from shared.utils.workspaces import load_gallery
    host = host_for_gen(gen)
    if host is None:
        return gen
    if (live := background_gallery(host, workspace_id)) is not None:
        return live
    if callable(reader := getattr(host, 'read_workspace_gallery', None)):
        return reader(workspace_id)
    if host.workspace_id != workspace_id:
        return load_gallery(host.workspaces.get(workspace_id), host._workspace_settings)
    return gen


def current_binding(state=None):
    binding = _execution.get()
    if binding is not None:
        return binding
    host = host_for_gen(state['gen']) if state is not None else None
    return (host, host.workspace_id) if host is not None else None


@contextmanager
def execution_workspace(host, workspace_id):
    host.workspaces.get(workspace_id)
    token = _execution.set((host, workspace_id))
    try:
        yield
    finally:
        _execution.reset(token)


def background_gallery(host, workspace_id):
    entry = _live.get((id(host), workspace_id))
    return None if entry is None else entry[0]


def recording_gallery(gen):
    """Gallery receiving new outputs: the assigned workspace's own lists during a background operation."""
    background = _background.get()
    return background[1] if background is not None and background[0] is gen else gen


@contextmanager
def background_workspace(host, workspace_id):
    """Execute in an assigned workspace while the host keeps showing the gallery its user selected.

    Concurrent operations on one workspace share its lists, and so does the host when its user opens
    that workspace meanwhile; outputs stay in this workspace if the user switches away.
    """
    from shared.utils.workspaces import load_gallery
    key = id(host), workspace_id
    with host._mutation_lock, _live_lock:
        entry = _live.get(key)
        if entry is None:
            gen = host._state['gen']
            gallery = {name: gen[name] for name in GALLERY_KEYS} if host.workspace_id == workspace_id else load_gallery(host.workspaces.get(workspace_id), host._workspace_settings)
            entry = _live[key] = [gallery, 0]
        entry[1] += 1
    token = _background.set((host._state['gen'], entry[0]))
    try:
        with execution_workspace(host, workspace_id):
            yield
    finally:
        _background.reset(token)
        try:
            host.save_background(workspace_id, entry[0])
        finally:
            with _live_lock:
                entry[1] -= 1
                if not entry[1]:
                    del _live[key]


def output_config(config, *, state=None, binding=None):
    binding = binding or current_binding(state)
    if binding is None:
        return config
    host, workspace_id = binding
    directory = host.workspaces.media_dir(workspace_id)
    return {**config, **dict.fromkeys(OUTPUT_KEYS, str(directory))} if directory is not None else config


def capture_task(task, state):
    binding = current_binding(state)
    if binding is not None:
        task.setdefault('workspace_id', binding[1])
    return task


class HeadlessWorkspaceHost:
    """Keep saved queue destinations usable in CLI and Python API execution."""
    def __init__(self, state, root, read_settings, *, output_override=False):
        from shared.utils.workspaces import WorkspaceStore
        from shared.utils.media_settings import MediaSettingsCache
        if output_override:
            raise ValueError('A workspace-bound queue cannot use --output-dir. Use its original Workspaces Root.')
        self._state, self.workspaces = state, WorkspaceStore(root)
        self._workspace_settings = MediaSettingsCache(read_settings)
        self.workspace_id = None
        state['_workspace_host'] = self
        register_host(self)

    background_agents = False  # no user views this gallery

    def activate_workspace(self, workspace_id, *, execution=False):
        from shared.utils.workspaces import load_gallery
        record = self.workspaces.get(workspace_id)
        if self.workspace_id != workspace_id:
            self.save_workspace()
            self._state['gen'].update(load_gallery(record, self._workspace_settings))
            self.workspace_id = workspace_id

    def save_workspace(self):
        from shared.utils.workspaces import capture_gallery
        if self.workspace_id:
            self.workspaces.save(self.workspace_id, capture_gallery(self._state['gen']))


def workspace_generation(function=None, *, headless=None):
    if function is None:
        return lambda function: workspace_generation(function, headless=headless)
    def prepare(task, state):
        host = host_for_gen(state['gen'])
        if host is None and task.get('workspace_id') and headless is not None:
            host = headless(state)
        if host is None:
            if task.get('workspace_id'):
                raise ValueError('This queue needs its workspace store. Open it in WanGP with the original Workspaces Root.')
            return None
        workspace_id = task.get('workspace_id') or host.workspace_id
        # Agent jobs record in their own workspace and leave the host's user on the one they view.
        task['workspace_background'] = bool(task.get('mcp_agent_id')) and host.background_agents
        if not task['workspace_background']:
            host.activate_workspace(workspace_id, execution=True)
        return host, workspace_id

    @wraps(function)
    def run(task, *args, **kwargs):
        host = host_for_gen(kwargs['state']['gen'])
        with getattr(host, '_execution_lock', nullcontext()):
            binding = prepare(task, kwargs['state'])
            if binding is None:
                return function(task, *args, **kwargs)
            host, workspace_id = binding
            if task['workspace_background']:
                with background_workspace(host, workspace_id):
                    return function(task, *args, **kwargs)
            with execution_workspace(host, workspace_id):
                try:
                    return function(task, *args, **kwargs)
                finally:
                    host.save_workspace()
    run.prepare_workspace = prepare
    return run


def owned_path(root, relative, resolved_root=None):
    """resolved_root: root.resolve(), when a caller maps many paths of one folder."""
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts or ':' in str(relative):
        raise ValueError('Invalid relative workspace path.')
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(resolved_root or root.resolve()):
        raise ValueError('Workspace path escapes its media folder.')
    return resolved


def bundle_files(path):
    """Return suffix -> file for known companions; GLB and splat PLY are self-contained."""
    path = Path(path).resolve()
    files = {path.suffix: path}
    for suffix in ('.json', '.png') if path.suffix.lower() in ('.glb', '.ply') else ('.json',):
        companion = path.with_suffix(suffix)
        if companion != path and companion.is_file():
            files[suffix] = companion
    ledger = path.with_name(path.name + '.wangp-files.json')
    if ledger.is_file():
        suffixes = json.loads(ledger.read_text(encoding='utf-8'))
        if not isinstance(suffixes, list):
            raise ValueError(f'Invalid companion index: {ledger.name}')
        for suffix in suffixes:
            if not isinstance(suffix, str) or not suffix.startswith(('.', '_')) or any(c in suffix for c in '/\\:') or suffix in ('.', '..'):
                raise ValueError(f'Invalid companion suffix: {suffix!r}')
            companion = path.with_name(path.stem + suffix)
            if not companion.is_file():
                raise FileNotFoundError(f'Missing companion: {companion}')
            files[suffix] = companion
    return files


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').digest()


def copy_bundle(source, directory):
    """Copy a complete bundle, returning its main path and files created by this call."""
    source, directory = Path(source).resolve(), Path(directory).resolve()
    if source.is_relative_to(directory):
        return str(source), []
    files = bundle_files(source)
    directory.mkdir(parents=True, exist_ok=True)
    created = []
    with _copy_lock:
        index = 0
        while True:
            stem = source.stem + (f' ({index})' if index else '')
            targets = {suffix: directory / (stem + suffix) for suffix in files}
            ledger = directory / (stem + source.suffix + '.wangp-files.json')
            existing = [target.exists() for target in targets.values()]
            companions = [suffix for suffix in files if suffix != source.suffix]
            try:
                same_ledger = (not companions and not ledger.exists()) or (ledger.is_file() and json.loads(ledger.read_text(encoding='utf-8')) == companions)
            except (UnicodeError, json.JSONDecodeError):
                same_ledger = False
            if same_ledger and all(existing) and all(target.is_file() and digest(files[suffix]) == digest(target) for suffix, target in targets.items()):
                return str(targets[source.suffix]), []
            if not any(existing) and not ledger.exists():
                break
            index += 1
        try:
            with tempfile.TemporaryDirectory(prefix='.copy-', dir=directory) as temporary:
                for suffix, original in files.items():
                    target = targets[suffix]
                    staged = Path(temporary) / target.name
                    shutil.copy2(original, staged)
                    if digest(original) != digest(staged):
                        raise OSError(f'Copy verification failed: {original.name}')
                    with target.open('xb'):
                        pass
                    created.append(target)
                    staged.replace(target)
                if companions:
                    with ledger.open('x', encoding='utf-8') as stream:
                        created.append(ledger)
                        json.dump(companions, stream)
            return str(targets[source.suffix]), created
        except BaseException:
            for target in reversed(created):
                target.unlink(missing_ok=True)
            raise
