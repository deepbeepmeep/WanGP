# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Persistent agent contexts over the shared workspace and queue contracts."""
import inspect
import re
import threading
import uuid
import weakref
from pathlib import Path
from typing import Any, Literal

from shared.utils.config_store import read_config, write_config
from shared.utils.media_settings import MediaSettingsCache
from shared.utils.workspace_storage import OUTPUT_KEYS, background_workspace, execution_workspace, output_config, register_host
from shared.utils.workspaces import WorkspaceStore, capture_gallery, load_gallery

VISIBILITY_KEY, VISIBILITY_MODES = 'mcp_workspace_visibility', ('own', 'own_and_user', 'all')
SCOPED_TOOLS = {'wangp_generate', 'wangp_validate', 'wangp_postprocess', 'wangp_toolbox', 'wangp_list_gallery', 'wangp_io', 'wangp_session', 'wangp_plugins'}


class StandaloneWorkspaceHost:
    def __init__(self, session, store, config):
        self._state, self.workspaces, self.config = session._state, store, config
        self._mutation_lock = self._execution_lock = store.lock
        self.workspace_id = None
        self._session = session
        runtime = session._ensure_runtime()
        self._workspace_settings = MediaSettingsCache(lambda path: runtime.module.get_settings_from_file(self._state, path, False, False, False)[0])
        register_host(self)

    def require_idle(self):
        if self._session._active_job is not None and not self._session._active_job.done:
            raise ValueError('This action records media in the agent workspace, which waits for the running job of this agent_context. Retry after it finishes (wangp_session wait_jobs); inspection, gallery reads and IO are not blocked.')

    background_agents = False  # each context has its own gallery, which no user views

    def save_workspace(self):
        if self.workspace_id:
            self.workspaces.save(self.workspace_id, capture_gallery(self._state['gen']))

    def activate_workspace(self, workspace_id, *, execution=False):
        with self._mutation_lock:
            if not execution:
                self.require_idle()
            if self.workspace_id != workspace_id:
                self.save_workspace()
            # Another context can explicitly attach to this same collection.
            # Refresh at each operation boundary rather than overwriting its work
            # with this context's older in-memory gallery.
            self._state['gen'].update(load_gallery(self.workspaces.get(workspace_id), self._workspace_settings))
            self.workspace_id = workspace_id

    def deepy_workspace_id(self):
        return self._session._workspace_id

    def read_workspace_gallery(self, workspace_id):
        if self.workspace_id == workspace_id and self._session._active_job is not None and not self._session._active_job.done:
            return self._state['gen']
        return load_gallery(self.workspaces.get(workspace_id), self._workspace_settings)


class AgentWorkspaces:
    def __init__(self, server, session, config, *, host=None, allow_read=False, http=False):
        self.server, self.base_session, self.config = server, session, config
        self.host, self.allow_read, self.http = host, allow_read, http
        self.store = host.workspaces if host else WorkspaceStore(session._root / (config.get('workspaces_dir') or 'workspaces'), config.get('workspace_storage_mode', 'shared'))
        mode = config.get('mcp_workspace_mode', 'auto')
        if mode not in ('auto', 'per_agent', 'shared'):
            raise ValueError('Invalid MCP workspace assignment mode.')
        self.automatic = mode == 'per_agent' or mode == 'auto' and config.get('workspace_storage_mode', 'shared') == 'dedicated'
        self.contexts = {}
        self.shared = None
        self.versions = {}
        self.connections = weakref.WeakKeyDictionary()
        self.lock = threading.RLock()
        self.records_root = self.store.root / '.agents'
        self.records_root.mkdir(exist_ok=True)

    def connection(self):
        context = self.server.get_context()
        try:
            return context.session
        except ValueError:
            return None

    def visibility(self):
        mode = (self.host._deps.get_server_config() if self.host else self.config).get(VISIBILITY_KEY, 'own_and_user')
        if mode not in VISIBILITY_MODES:
            raise ValueError('Invalid MCP workspace visibility.')
        return mode

    def visible(self, workspace, record):
        """Agents see their own workspaces, the user's ordinary ones unless restricted, and Deepy conversations or other agents' workspaces only when allowed."""
        mode = self.visibility()
        if record is not None and workspace['id'] == record['workspace_id'] or record is not None and workspace.get('mcp_agent_id') == record['id']:
            return True
        return mode == 'all' or mode == 'own_and_user' and not workspace.get('mcp_agent_id') and not workspace.get('deepy_session_id')

    def viewed_workspace(self, record):
        """The workspace shown in WanGP when this agent may see it: selected_only reads the user's selection there."""
        workspace = self.store.get(self.host.workspace_id)
        return workspace if self.visible(workspace, record) else None

    def create(self, name='MCP Agent', workspace_id=None, storage_mode=None):
        identity = uuid.uuid4().hex
        if name == 'MCP Agent':
            name += ' ' + identity[:6]
        with self.lock:
            if workspace_id is None:
                self.store.default_mode = self.config.get('workspace_storage_mode', 'shared')
                workspace_id = self.store.create(name, mcp_agent_id=identity, storage_mode=storage_mode)
            else:
                self.store.get(workspace_id)
            record = {'id': identity, 'workspace_id': workspace_id}
            write_config(record, self.records_root / (identity + '.json'))
            if self.host:
                self.host.publish_workspaces()
            return record

    def resolve(self, identity=None, *, create=True):
        with self.lock:
            connection = self.connection()
            identity = identity or (self.connections.get(connection) if connection is not None else None)
            if identity is None:
                if not self.automatic:
                    if self.host is not None and self.shared is None and create:
                        # Shared agents keep the workspace shown when they first connected, whatever the user views later.
                        self.shared = self.create(workspace_id=self.host.workspace_id)
                    return self.shared
                if not create:
                    return None
                record = self.create()
                identity = record['id']
                if connection is not None:
                    self.connections[connection] = identity
            if not isinstance(identity, str) or re.fullmatch(r'[0-9a-f]{32}', identity) is None:
                raise ValueError('Invalid agent_context. Use wangp_workspace to create or resume a context.')
            path = self.records_root / (identity + '.json')
            if not path.is_file():
                raise ValueError('Unknown agent_context. Query workspaces or create a new context.')
            record = read_config(path)
            self.store.get(record['workspace_id'])
            return record

    def describe(self, record):
        workspace = self.store.get(record['workspace_id'])
        result = {'agent_context': record['id'], 'workspace_id': workspace['id'], 'name': workspace['name'], 'storage_mode': workspace.get('storage_mode', 'shared'),
                  'next': "Pass agent_context on later calls to resume this context or to run several agents over one connection. Media you create goes to this workspace without changing the user's view; to work in another workspace the user names, attach to it and use that agent_context."}
        result['folders'] = {f'@{alias}': str(root) for alias, root in self.runtime(record)[2]._wangp_file_policy.mounts}
        if record is self.shared:
            result['assignment'] = ('shared: every agent calling without agent_context uses this context, bound to the workspace shown in WanGP when the first agent connected. '
                                    'Use action=create for a workspace of your own, or attach to the one the user names.')
        return result

    def runtime(self, record):
        from shared.api import WanGPSession
        from shared.deepy.engine import AssistantSessionState
        from shared.deepy.filesystem import build_file_access_policy
        from shared.deepy.prime_filesystem import build_prime_policy
        from .server import build_server_for_session, _build_default_toolbox

        with self.lock:
            identity, workspace_id = record['id'], record['workspace_id']
            cached = self.contexts.get(identity)
            workspace = self.store.get(workspace_id)
            version = (workspace.get('storage_mode', 'shared'), workspace.get('folder'))
            if cached is not None and self.versions[identity] == version:
                return cached
            if cached is not None:
                cached[2]._wangp_pages.close()
            base = self.base_session
            if base._output_dir is not None:
                raise ValueError('Explicit --output-dir cannot be combined with an agent workspace. Restart without the override.')
            session = WanGPSession(root=base._root, config_path=base._config_path, cli_args=base._cli_args, webui_state=base._state if self.host else None, webui_append=bool(self.host), console_output=False, console_isatty=False)
            session._workspace_id, session._agent_context = workspace_id, identity
            host = self.host or StandaloneWorkspaceHost(session, self.store, self.config)
            if self.host:
                session._gradio_webui_context = {'load_queue': self.host.load_inline_queue}
                session._selection_source = lambda: self.viewed_workspace(record)
            config = output_config(self.config, binding=(host, workspace_id))
            # External MCP keeps its own grants; Deepy's selected folders and
            # filesystem mode must not broaden a connected agent's permissions.
            paths = {key: str((base._root / (config.get(key) or config.get('save_path') or 'outputs')).resolve()) for key in OUTPUT_KEYS}
            policy = build_file_access_policy(paths, unrestricted_read=self.allow_read)
            documents = self.store.root / '.agents' / identity / 'documents'
            policy = build_prime_policy(policy, documents)
            toolbox = _build_default_toolbox(session, AssistantSessionState(), native_inspection=True, client_id_prefix="ai_mcp")
            toolbox.workspace_id = workspace_id
            inner = build_server_for_session(session, toolbox=toolbox, file_access_policy=policy, api_version='2.1', http_media_transfer=self.http, allow_async=True)
            if cached is not None:
                # A storage conversion changes IO scopes, but completed jobs
                # remain queryable through their original operation records.
                inner._wangp_jobs._jobs = cached[2]._wangp_jobs._jobs
                inner._wangp_jobs._owner = cached[2]._wangp_jobs._owner
            cached = (session, host, inner)
            self.contexts[identity] = cached
            self.versions[identity] = version
            return cached

    @staticmethod
    def visible_action(name, arguments):
        if name != 'wangp_toolbox' or arguments.get('action') is None or arguments.get('arguments') is None:
            return False
        action = arguments['action']
        if action in ('media_settings', 'inspect_media', 'inspect_video', 'view_media', 'check_lip_sync', 'detect_video_cuts', 'plan_video_windows', 'check_video_edit'):
            return False
        return action != 'inspect_audio' or bool(arguments['arguments'].get('preview'))

    def invoke(self, name, arguments, record):
        session, host, inner = self.runtime(record)
        workspace_id, call = record['workspace_id'], inner._tool_manager.get_tool(name).fn
        if name == 'wangp_toolbox' and arguments.get('wait') is False and self.visible_action(name, arguments):
            # The job thread records its media under the same workspace scope as a waited call.
            job = inner._wangp_jobs.start_tool(arguments['action'], lambda: self.invoke(name, {**arguments, 'wait': True}, record))
            return inner._tool_manager.get_tool('wangp_session').fn(action='get_job', arguments={'job_id': job.job_id})
        if not self.visible_action(name, arguments):
            with execution_workspace(host, workspace_id):
                return call(**arguments)
        if host.background_agents:
            with background_workspace(host, workspace_id):
                return call(**arguments)
        with host._mutation_lock:
            host.activate_workspace(workspace_id)
            with execution_workspace(host, workspace_id):
                result = call(**arguments)
                host.save_workspace()
                return result

    def transfer_store(self, token):
        # The untrusted routing suffix selects a context; that context then verifies
        # the grant, scope, expiry and exact file through its normal API.
        identity = token.partition('.')[2]
        if not identity:
            return self.server._wangp_file_store
        context = self.resolve(identity)
        return self.runtime(context)[2]._wangp_file_store

    def install(self):
        from .server import _in_thread

        router = self
        originals = {tool.name: tool for tool in self.server._tool_manager.list_tools() if tool.name in SCOPED_TOOLS}
        for name, tool in originals.items():
            def make_proxy(name, tool):
                def invoke(**kwargs):
                    identity = kwargs.pop('agent_context', None)
                    record = router.resolve(identity)
                    if record is None:
                        return tool.fn(**kwargs)
                    return router.invoke(name, kwargs, record)
                invoke.__name__ = name
                proxy = _in_thread(invoke)
                signature = inspect.signature(tool.fn)
                proxy.__signature__ = signature.replace(parameters=[*signature.parameters.values(), inspect.Parameter('agent_context', inspect.Parameter.KEYWORD_ONLY, default=None, annotation=str | None)])
                proxy.__name__ = name
                return proxy
            self.server._tool_manager.remove_tool(name)
            self.server.add_tool(make_proxy(name, tool), name=name, description=tool.description + ' Optional agent_context resumes an assigned workspace; obtain it with wangp_workspace. Shared-user workspace and queue permissions apply.')
            proxy_tool = self.server._tool_manager.get_tool(name)
            proxy_tool.parameters = {**tool.parameters, 'properties': {**tool.parameters['properties'], 'agent_context': {'type': 'string', 'description': 'Context returned by wangp_workspace; reuse after reconnecting or for separate logical agents.'}}}
            proxy_tool.fn_metadata.arg_model.model_config['extra'] = 'forbid'
            proxy_tool.fn_metadata.arg_model.model_rebuild(force=True)

        @self.server.tool()
        def wangp_workspace(action: Literal['current', 'list', 'create', 'attach'] = 'current', name: str = 'MCP Agent', workspace_id: str | None = None, agent_context: str | None = None, query: str = '', limit: int = 50, offset: int = 0, storage_mode: Literal['shared', 'dedicated'] | None = None) -> dict[str, Any]:
            """Agent workspaces: current returns/resumes your assignment; list queries the workspaces you may use (your own, and the user's unless the user restricted it); create starts an independent context (storage_mode: dedicated keeps its media in its own folder, shared in the common output folders; the configured default otherwise); attach creates a context for an existing workspace_id. Pass the returned agent_context on media, IO, job and session calls, especially after reconnecting or when multiplexing agents. Current shared-user permissions permit discovery/attachment; per-user visibility and cancellation rights are not implemented."""
            if action == 'list':
                if not 1 <= limit <= 200 or offset < 0:
                    raise ValueError('limit must be 1..200 and offset non-negative.')
                caller = router.resolve(agent_context, create=False)
                records = [item for item in router.store.catalog() if query.casefold() in item['name'].casefold() and router.visible(item, caller)]
                page = [{**item, 'storage_mode': item.get('storage_mode', 'shared'), 'owner_kind': 'deepy_session' if item.get('deepy_session_id') else 'mcp_agent' if item.get('mcp_agent_id') else 'manual'} for item in records[offset:offset + limit]]
                return {'workspaces': page, 'total': len(records), 'next_offset': offset + limit if offset + limit < len(records) else None, 'visibility': router.visibility()}
            if action in ('create', 'attach'):
                if action == 'attach' and not workspace_id:
                    raise ValueError('attach requires workspace_id from action=list.')
                if action == 'attach' and not router.visible(router.store.get(workspace_id), router.resolve(agent_context, create=False)):
                    raise ValueError('Unknown workspace_id, or a workspace agents may not use (Configuration > Agents > Workspaces Agents Can See).')
                if action == 'create' and workspace_id:
                    raise ValueError('Use attach to select an existing workspace.')
                if action == 'attach' and storage_mode:
                    raise ValueError('storage_mode applies to create; an attached workspace keeps its own storage.')
                record = router.create(name, workspace_id, storage_mode)
            elif action == 'current':
                record = router.resolve(agent_context)
                if record is None:
                    return {'assignment': 'shared', 'automatic': router.automatic, 'next': 'Use action=create or action=attach to assign a workspace.'}
            else:
                raise ValueError('action must be current, list, create or attach.')
            return router.describe(record)
        self.server._wangp_agents = self
        if self.automatic:
            self.server.settings.stateless_http = False
        return self.server
