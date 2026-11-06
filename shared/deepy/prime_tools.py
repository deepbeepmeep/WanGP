# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import json
import logging
import os
import queue
import re
import shutil
import threading
import time
from concurrent.futures import Future
from contextlib import AsyncExitStack, suppress
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

import anyio
from anyio.from_thread import start_blocking_portal

from shared.api import WanGPSession
from shared.deepy import media_registry
from shared.deepy.config import DEEPY_AUTO_CANCEL_QUEUE_TASKS_DEFAULT, DEEPY_AUTO_CANCEL_QUEUE_TASKS_KEY, DEEPY_CONTEXT_TOKENS_DEFAULT, DEEPY_CONTEXT_TOKENS_KEY, DEEPY_MCP_AUTO_DISCOVER_PATHS_DEFAULT, DEEPY_MCP_AUTO_DISCOVER_PATHS_KEY, DEEPY_PRIME_MCP_SERVERS_KEY, get_deepy_config_value, get_deepy_runtime_config, normalize_deepy_auto_cancel_queue_tasks, normalize_deepy_context_tokens, normalize_deepy_mcp_auto_discover_paths, normalize_deepy_prime_mcp_servers
from shared.gradio import assistant_chat
from shared.mcp.server import _compact_job_snapshot, build_inprocess_server, resolve_gallery_media_path
from shared.mcp.resources import RESOURCE_ALIASES, RESOURCE_DESCRIPTION, query_resources, resource_record, _extract_markdown_sections, _select_markdown_sections, _search_markdown_sections, _resource_read_blocks


REMOTE_JOB_WAIT_S = 600  # a remote engine's generation or post-processing call returns by then, its job still running


def _resolve_external_stdio_command(value: str, allow_changed_path_search: bool = False) -> str:
    command = os.path.expanduser(os.path.expandvars(str(value or "").strip()))
    if not command or os.path.isfile(command):
        return command
    if resolved := shutil.which(command):
        return resolved
    if not allow_changed_path_search:
        return command
    path = Path(command)
    try:
        runtime_root = path.parents[2]
    except IndexError:
        return command
    if path.parent.name.casefold() != "bin" or not runtime_root.is_dir():
        return command
    candidates = [candidate for candidate in runtime_root.glob(f"*/bin/{path.name}") if candidate.is_file()]
    return str(max(candidates, key=lambda candidate: candidate.stat().st_mtime)) if candidates else command


def _relocate_external_stdio_env(configured_command: str, resolved_command: str, configured_env: dict[str, Any]) -> dict[str, str]:
    environment = {str(key): str(value) for key, value in configured_env.items()}
    configured_path = Path(os.path.expanduser(os.path.expandvars(configured_command)))
    resolved_path = Path(resolved_command)
    if not configured_path.is_absolute() or configured_path.parent.name.casefold() != "bin" or not resolved_path.is_absolute():
        return environment
    configured_bin = str(configured_path.parent)
    resolved_bin = str(resolved_path.parent)
    if configured_bin == resolved_bin:
        return environment
    pattern = re.compile(re.escape(configured_bin), re.IGNORECASE if os.name == "nt" else 0)
    return {key: pattern.sub(lambda _match: resolved_bin, value) for key, value in environment.items()}


def _mcp_connection_error_message(error: BaseException) -> str:
    pending, leaves = [error], []
    while pending:
        current = pending.pop(0)
        nested = list(getattr(current, "exceptions", ()) or ())
        if nested:
            pending[:0] = nested
            continue
        message = str(current).strip()
        leaves.append(f"{type(current).__name__}: {message}" if message else type(current).__name__)
    return "; ".join(dict.fromkeys(leaves)) or type(error).__name__


class DeepyPrimeTools:
    _POLL_INTERVAL_SECONDS = 0.2
    _plugin_tools_by_name = {}
    allow_async = False  # remote engines start jobs in the background (wait=false) and wait for them in bounded calls
    _queue_on_server = False  # their jobs join the queue whether or not a turn is still running to forward them

    def __init__(self, state: dict[str, Any], send_cmd: Callable[..., None], assistant_session, zero_tools=None, allow_async: bool = False) -> None:
        self.allow_async = allow_async
        self.state = state
        self.send_cmd = send_cmd
        self.assistant_session = assistant_session
        self._zero_tools = zero_tools
        from shared.deepy.filesystem import build_file_access_policy
        from shared.deepy.prime_filesystem import build_prime_policy
        from shared.deepy import session_store

        base_policy = build_file_access_policy(zero_tools._server_config() if zero_tools is not None else get_deepy_runtime_config())
        self._api_session = WanGPSession(webui_state=state, console_output=False, console_isatty=False, webui_concurrent=allow_async)
        from shared.utils.workspace_storage import host_for_gen
        host = host_for_gen(state['gen'])
        if host is not None:
            self._api_session._workspace_id = host.deepy_workspace_id()
        from shared.deepy.hybrid import service_for
        service = service_for(state) if allow_async else None
        self._queue_on_server = service is not None
        self._api_session._gradio_webui_context = {"load_queue": service.load_inline_queue} if service is not None else {"defer_load_queue_trigger": True}

        workspace = session_store.session_workspace(assistant_session)
        if workspace is None:
            workspace = session_store.ensure_mono_session_workspace(assistant_session.chat_session_id)
        self.file_access_policy = build_prime_policy(base_policy, workspace)
        if zero_tools is not None:
            zero_tools.file_access_policy_override = self.file_access_policy
        self.assistant_session.file_access_policy = self.file_access_policy
        self.allow_read_file_system = self.file_access_policy.read_enabled
        from shared.utils.plugins import get_deepy_prime_plugin_tools

        self._plugin_tools_by_name = {definition.name: definition for definition in get_deepy_prime_plugin_tools() if not definition.requires_file_system or base_policy.read_enabled}
        self._tool_progress_callback: Callable[..., None] | None = None
        self._server = build_inprocess_server(self._api_session, toolbox=zero_tools, default_job_event_limit=0, file_access_policy=self.file_access_policy, api_version="2.1", allow_async=allow_async)
        self._server._wangp_wait_interrupted = self._is_interrupted  # the chat Stop button ends a wait for jobs
        if self._server._wangp_python_sandbox is not None:
            sandbox = self._server._wangp_python_sandbox
            sandbox.interrupted = self._is_interrupted  # the chat Stop button kills a running script
            sandbox.call_tool = lambda name, arguments: self.call(name, arguments, inline=True)  # scripts call WanGP's tools with Prime's own bookkeeping
            sandbox.progress = self._report_script_progress
        if getattr(self._server, "_wangp_web_search", None) is not None:
            self._server._wangp_web_search.progress = self._report_script_progress  # downloads and LoRA installs show a progress bar like scripts
        plugins = getattr(self._server, "_wangp_plugins", None)
        if plugins is not None:  # plugin actions stop with the chat Stop button, report on their card, call WanGP's tools re-entrantly and start the browser queue
            plugins.interrupted = self._is_interrupted
            plugins.progress = self._report_plugin_progress
            plugins.call_tool = lambda name, arguments: self.call(name, arguments, inline=True, cascade=True)
            plugins.trigger_queue = lambda job: self.send_cmd("load_queue_trigger", {"job_id": "", "token": job.webui_load_queue_token})
        self._script_progress_target: Callable[..., None] | None = None  # the transcript entry of the running script, while one of its tool calls reports elsewhere
        self._plugin_progress: tuple[str, dict[str, Any]] | None = None  # the last progress a running plugin action reported on its card
        self._inline_calls = 0  # tool calls of a running script or plugin action: their generations report their steps
        self.pause_runtime: Callable[..., None] | None = None  # set by the engine running this toolbox: releases Deepy's GPU before a script's GPU tool
        self.run_script_tool: Callable[..., dict[str, Any]] | None = None  # set by the engine too: gives a script's tool call its own transcript card
        self._background_jobs = {}
        self._external_servers = normalize_deepy_prime_mcp_servers(get_deepy_config_value(DEEPY_PRIME_MCP_SERVERS_KEY, {}))
        self._auto_discover_mcp_paths = normalize_deepy_mcp_auto_discover_paths(get_deepy_config_value(DEEPY_MCP_AUTO_DISCOVER_PATHS_KEY, DEEPY_MCP_AUTO_DISCOVER_PATHS_DEFAULT))
        self._external_server_errors: dict[str, str] = {}
        self._request_queue: queue.Queue[Any] = queue.Queue()
        self._ready_event = threading.Event()
        self._worker_error: BaseException | None = None
        self._portal_context = start_blocking_portal(name="deepy-prime-mcp")
        self._portal = self._portal_context.__enter__()
        self._worker_future = self._portal.start_task_soon(self._connection_worker)
        if not self._ready_event.wait(timeout=60):
            self._request_queue.put(None)
            self._portal_context.__exit__(None, None, None)
            raise TimeoutError("Deepy Prime MCP server connections did not initialize in time.")
        if self._worker_error is not None:
            self._portal_context.__exit__(None, None, None)
            raise RuntimeError(f"Deepy Prime MCP server initialization failed: {self._worker_error}") from self._worker_error
        self._tool_defs_by_name = {tool["function"]["name"]: tool for tool in self._tool_defs}
        self._active_job_id = ""
        self._active_job_cancel_requested = False
        self._active_session_job = None
        self._remote_llm = False

    def bind_turn(self, state: dict[str, Any], send_cmd: Callable[..., None]) -> None:
        if self.state is not state:
            raise RuntimeError("Deepy Prime cannot move between WanGP browser sessions.")
        self.send_cmd = send_cmd
        if self._zero_tools is not None:
            self._zero_tools.send_cmd = send_cmd

    def close(self) -> None:
        if self._active_job_id:
            try:
                if self._active_session_job is not None:
                    self._active_session_job.cancel()
                else:
                    self._server._wangp_jobs.get(self._active_job_id).job.cancel()
            except Exception:
                pass
            self._active_job_id = ""
            self._active_session_job = None
        for job in list(self._background_jobs.values()):
            with suppress(ValueError):  # a toolbox job that cannot be cancelled runs to completion
                job.cancel()
        if self._server._wangp_python_sandbox is not None:
            self._server._wangp_python_sandbox.close()
        self._server._wangp_pages.close()
        self._request_queue.put(None)
        self._worker_future.result(timeout=30)
        self._portal_context.__exit__(None, None, None)
        self._api_session.close()

    def cancel_active_job(self) -> str:
        job_id = str(self._active_job_id or "").strip()
        background_id = ""
        auto_abort = normalize_deepy_auto_cancel_queue_tasks(get_deepy_config_value(DEEPY_AUTO_CANCEL_QUEUE_TASKS_KEY, DEEPY_AUTO_CANCEL_QUEUE_TASKS_DEFAULT))
        for running_id, job in list(self._background_jobs.items()) if auto_abort else ():
            if not job.done and not job.cancel_requested:
                with suppress(ValueError):  # a toolbox job that cannot be cancelled runs to completion
                    job.cancel()
                    background_id = running_id
        if not job_id or self._active_job_cancel_requested:
            return background_id
        self._active_job_cancel_requested = True
        try:
            if self._active_session_job is not None:
                self._active_session_job.cancel()
            else:
                self._server._wangp_jobs.get(job_id).job.cancel()
        except Exception:
            self._active_job_cancel_requested = False
            raise
        return job_id

    def bind_runtime_tools(self, vision_query_callback=None, tool_progress_callback=None, vision_is_remote: bool = False) -> None:
        self._tool_progress_callback = tool_progress_callback
        self._remote_llm = bool(vision_is_remote)
        if self._zero_tools is not None:
            self._zero_tools.bind_runtime_tools(vision_query_callback=vision_query_callback, tool_progress_callback=tool_progress_callback, vision_is_remote=vision_is_remote)

    @staticmethod
    def _safe_tool_name(value: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "").strip()).strip("_")
        return normalized or "tool"

    async def _connection_worker(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.sse import sse_client
        from mcp.client.stdio import stdio_client
        from mcp.client.streamable_http import streamablehttp_client

        from mcp.shared.memory import create_connected_server_and_client_session

        logger = logging.getLogger("mcp.server.lowlevel.server")
        previous_level = logger.level
        logger.setLevel(logging.WARNING)
        try:
            async with AsyncExitStack() as stack:
                clients = {"wangp": await stack.enter_async_context(create_connected_server_and_client_session(self._server._mcp_server))}
                for server_name, config in self._external_servers.items():
                    server_stack = AsyncExitStack()
                    await server_stack.__aenter__()
                    try:
                        transport = config["transport"]
                        if transport == "stdio":
                            configured_command = str(config["command"])
                            command = _resolve_external_stdio_command(configured_command, self._auto_discover_mcp_paths)
                            env = None
                            if isinstance(config.get("env"), dict):
                                configured_env = _relocate_external_stdio_env(configured_command, command, config["env"]) if self._auto_discover_mcp_paths and command != configured_command else {str(key): str(value) for key, value in config["env"].items()}
                                env = {**os.environ, **configured_env}
                            if command != configured_command:
                                print(f"[DeepyPrimeTools] External MCP server '{server_name}' resolved updated command: {command}")
                            streams = await server_stack.enter_async_context(stdio_client(StdioServerParameters(command=command, args=[str(arg) for arg in list(config.get("args", []) or [])], env=env, cwd=config.get("cwd"))))
                        elif transport == "sse":
                            streams = await server_stack.enter_async_context(sse_client(str(config["url"]), headers=dict(config.get("headers", {}) or {}), timeout=float(config.get("timeout_seconds", 30)), sse_read_timeout=float(config.get("read_timeout_seconds", 300))))
                        else:
                            streams = await server_stack.enter_async_context(streamablehttp_client(str(config["url"]), headers={str(key): str(value) for key, value in dict(config.get("headers", {}) or {}).items()}, timeout=float(config.get("timeout_seconds", 30)), sse_read_timeout=float(config.get("read_timeout_seconds", 300))))
                        client = await server_stack.enter_async_context(ClientSession(streams[0], streams[1], read_timeout_seconds=timedelta(seconds=float(config.get("read_timeout_seconds", 300)))))
                        await client.initialize()
                    except Exception as error:
                        message = _mcp_connection_error_message(error)
                        try:
                            await server_stack.aclose()
                        except Exception as cleanup_error:
                            message = f"{message}; cleanup: {_mcp_connection_error_message(cleanup_error)}"
                        self._external_server_errors[server_name] = message
                        print(f"[DeepyPrimeTools] Skipping unavailable external MCP server '{server_name}': {message}")
                        continue
                    stack.push_async_callback(server_stack.aclose)
                    clients[server_name] = client

                self._tool_defs = []
                self._tool_routes = {}
                self._resource_defs = []
                self._server_instructions = ""
                prompts = await clients["wangp"].list_prompts()
                if any(str(prompt.name) == "wangp_agent" for prompt in prompts.prompts):
                    prompt_result = await clients["wangp"].get_prompt("wangp_agent")
                    self._server_instructions = "\n\n".join(str(getattr(message.content, "text", "") or "").strip() for message in prompt_result.messages if getattr(message.content, "type", "") == "text").strip()
                for server_name, client in clients.items():
                    cursor = None
                    try:
                        while True:
                            listed_resources = await client.list_resources(cursor=cursor)
                            for resource in listed_resources.resources:
                                self._resource_defs.append(resource_record(resource, server_name))
                            cursor = getattr(listed_resources, "nextCursor", None)
                            if not cursor:
                                break
                    except Exception:
                        pass
                    try:
                        listed_tools = await client.list_tools()
                    except Exception as error:
                        if server_name == "wangp":
                            raise
                        message = _mcp_connection_error_message(error)
                        self._external_server_errors[server_name] = message
                        print(f"[DeepyPrimeTools] Ignoring tools from unavailable external MCP server '{server_name}': {message}")
                        continue
                    for tool in listed_tools.tools:
                        if server_name == "wangp" and tool.name == "mcp_resource":
                            continue  # Prime federates the same resource browser across all connected servers.
                        exposed_name = str(tool.name) if server_name == "wangp" else f"mcp_{self._safe_tool_name(server_name)}_{self._safe_tool_name(tool.name)}"
                        if exposed_name in self._tool_routes:
                            raise RuntimeError(f"Duplicate Deepy Prime MCP tool name: {exposed_name}")
                        self._tool_routes[exposed_name] = (server_name, str(tool.name))
                        description = str(tool.description or "")
                        if server_name != "wangp":
                            description = f"External MCP server '{server_name}'. {description}".strip()
                        self._tool_defs.append({"type": "function", "function": {"name": exposed_name, "description": description, "parameters": dict(tool.inputSchema or {"type": "object", "properties": {}})}})
                self._tool_defs.append({"type": "function", "function": {"name": "mcp_resource", "description": RESOURCE_DESCRIPTION, "parameters": {"type": "object", "properties": {"server": {"type": "string"}, "uri": {"type": "string"}, "query": {"type": "string"}, "section": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 100}, "cursor": {"type": "string"}}, "additionalProperties": False}}})
                self._ready_event.set()

                while True:
                    request = await anyio.to_thread.run_sync(self._request_queue.get)
                    if request is None:
                        break
                    tool_name, arguments, future = request
                    try:
                        if tool_name == "mcp_resource":
                            local_resources = (await clients["wangp"].list_resources()).resources
                            self._resource_defs = [item for item in self._resource_defs if item["server"] != "wangp"] + [resource_record(item) for item in local_resources]
                            async def read_resource(server_name, uri):
                                response = await clients[server_name].read_resource(uri)
                                return [{"uri": str(item.uri), "mime_type": str(item.mimeType or ""), **({"text": item.text} if hasattr(item, "text") else {"blob": item.blob})} for item in response.contents]

                            budget = min(16000, normalize_deepy_context_tokens(get_deepy_config_value(DEEPY_CONTEXT_TOKENS_KEY, DEEPY_CONTEXT_TOKENS_DEFAULT)) // 2)
                            future.set_result(await query_resources(self._server._wangp_pages, self._resource_defs, read_resource, arguments, servers=clients, unavailable_servers=self._external_server_errors, read_page_chars=budget))
                            continue
                        server_name, original_name = self._tool_routes[tool_name]
                        future.set_result(await clients[server_name].call_tool(original_name, arguments))
                    except BaseException as exc:
                        future.set_exception(exc)
        except BaseException as exc:
            self._worker_error = exc
            self._ready_event.set()
            raise
        finally:
            logger.setLevel(previous_level)

    def _call_mcp_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        self._remember_gallery_input_references(tool_name, arguments)
        future = Future()
        self._request_queue.put((tool_name, dict(arguments or {}), future))
        result = future.result()
        if isinstance(result, dict):
            normalized = self._enforce_output_budget(tool_name, self.file_access_policy.virtualize_result(result))
            self._remember_gallery_download_references(tool_name, normalized)
            return normalized
        content_text = "\n".join(str(getattr(item, "text", "") or "") for item in result.content if getattr(item, "type", "") == "text").strip()
        if result.isError:
            return self.file_access_policy.virtualize_result({"status": "error", "tool": tool_name, "error": content_text or f"MCP tool '{tool_name}' failed."})
        payload = result.structuredContent
        if payload is None and content_text:
            try:
                payload = json.loads(content_text)
            except json.JSONDecodeError:
                payload = content_text
        if isinstance(payload, dict):
            normalized = dict(payload)
            normalized.setdefault("status", "running" if normalized.get("done") is False else "done")
        else:
            normalized = {"status": "done", "content": payload}
        normalized = self._enforce_output_budget(tool_name, self.file_access_policy.virtualize_result(normalized))
        self._remember_gallery_download_references(tool_name, normalized)
        return normalized

    def _remember_gallery_input_references(self, tool_name: str, arguments: Any) -> None:
        route = self._tool_routes.get(tool_name)
        if route is None or route[0] != "wangp":
            return
        media_ids = []

        def collect(value: Any) -> None:
            if isinstance(value, dict):
                for child_value in value.values():
                    collect(child_value)
            elif isinstance(value, (list, tuple)):
                for child_value in value:
                    collect(child_value)
            elif isinstance(value, str) and re.fullmatch(r"(?:visual|audio):[a-f0-9]{12}", value.strip(), re.IGNORECASE):
                media_ids.append(value.strip().casefold())

        collect(arguments)
        for media_id in dict.fromkeys(media_ids):
            try:
                path = resolve_gallery_media_path(self._api_session, media_id)
            except (FileNotFoundError, KeyError, OSError, ValueError):
                continue
            self.assistant_session.gallery_download_registry[media_id] = path
            media_registry.record_media_access(self.assistant_session, path, source="wangp", access="read")

    def _remember_generated_media(self, result: Any) -> None:
        output_paths = []

        def collect(value: Any, key: str = "") -> None:
            if isinstance(value, dict):
                for child_key, child_value in value.items():
                    collect(child_value, str(child_key).casefold())
            elif isinstance(value, (list, tuple)):
                for child_value in value:
                    collect(child_value, key)
            elif key in {"output_file", "generated_files"} and isinstance(value, str):
                output_paths.append(value.strip())

        collect(self.file_access_policy.virtualize_result(result))
        for output_path in dict.fromkeys(path for path in output_paths if path):
            try:
                path = self.file_access_policy.resolve_path(output_path)
            except (FileNotFoundError, OSError, PermissionError, ValueError):
                continue
            if not path.is_file():
                continue
            media_registry.record_media_access(self.assistant_session, str(path), source="deepy", access="write")

    def _remember_gallery_download_references(self, tool_name: str, result: Any) -> None:
        route = self._tool_routes.get(tool_name)
        if route is None or route[0] != "wangp":
            return
        media_ids = []

        def collect(value: Any, key: str = "") -> None:
            if isinstance(value, dict):
                for child_key, child_value in value.items():
                    collect(child_value, str(child_key).casefold())
            elif isinstance(value, (list, tuple)):
                for child_value in value:
                    collect(child_value, key)
            elif key in {"media_id", "media_ids"} and isinstance(value, str) and re.fullmatch(r"(?:visual|audio):[a-f0-9]{12}", value.strip(), re.IGNORECASE):
                media_ids.append(value.strip().casefold())

        collect(result)
        for media_id in dict.fromkeys(media_ids):
            try:
                self.assistant_session.gallery_download_registry[media_id] = resolve_gallery_media_path(self._api_session, media_id)
            except (FileNotFoundError, KeyError, OSError, ValueError):
                continue

    @staticmethod
    def _enforce_output_budget(tool_name: str, result: dict[str, Any]) -> dict[str, Any]:
        serialized = json.dumps(result, ensure_ascii=False, separators=(",", ":"), default=str)
        max_chars = max(8000, min(normalize_deepy_context_tokens(get_deepy_config_value(DEEPY_CONTEXT_TOKENS_KEY, DEEPY_CONTEXT_TOKENS_DEFAULT)), 100000))
        if len(serialized) <= max_chars:
            return result
        if tool_name == "wangp_io" and isinstance(result.get("entries"), list) and result["entries"]:
            page = dict(result)
            entries = page["entries"] = []
            offset = int(page["offset"])
            page.update(count=0, has_more=True, next_offset=offset)
            for entry in result["entries"]:
                entries.append(entry)
                page.update(count=len(entries), next_offset=offset + len(entries))
                if len(json.dumps(page, ensure_ascii=False, separators=(",", ":"), default=str)) > max_chars:
                    entries.pop()
                    page.update(count=len(entries), next_offset=offset + len(entries))
                    break
            page["has_more"] = bool(result.get("has_more")) or len(entries) < len(result["entries"])
            if not page["has_more"]:
                page["next_offset"] = None
            if entries and len(json.dumps(page, ensure_ascii=False, separators=(",", ":"), default=str)) <= max_chars:
                return page
        return {
            "status": "error",
            "tool": tool_name,
            "error": f"MCP tool result exceeded Deepy's context-safe output budget ({len(serialized):,} > {max_chars:,} characters).",
            "hint": "Use query, wildcard filters, pagination, or a more specific item lookup before retrying.",
        }

    def _is_interrupted(self) -> bool:
        return bool(self.assistant_session.interrupt_requested)

    def _update_tool_progress(self, status: str, status_text: str, result: dict[str, Any]) -> None:
        if callable(self._tool_progress_callback):
            self._tool_progress_callback(status=status, status_text=status_text, result=self.file_access_policy.virtualize_result(result))

    def _report_script_progress(self, status_text: str, summary: dict[str, Any]) -> None:
        """Live status of a running Python script on its own transcript entry (its tool calls report to the script, not to that entry)."""
        target = self._script_progress_target or self._tool_progress_callback
        if callable(target):
            target(status="running", status_text=status_text, result=self.file_access_policy.virtualize_result(summary))

    def _report_plugin_progress(self, status_text: str, summary: dict[str, Any]) -> None:
        self._plugin_progress = (status_text, summary)
        self._update_tool_progress("running", status_text, summary)

    def _cascade_plugin_progress(self, result: dict[str, Any] | None) -> None:
        """A generation run by a plugin action moves the action's own bar: done of total items plus the running item's fraction."""
        inner = (result or {}).get("progress") if isinstance(result, dict) else None
        if not isinstance(inner, dict) or not inner.get("total"):
            return
        status_text, summary = self._plugin_progress or ("Running", {"status": "running", "live": True})
        outer = dict(summary.get("progress") or {"done": 0, "total": 1, "text": ""})
        total = float(outer.get("total") or 1)
        fraction = max(0.0, min(1.0, float(inner.get("done") or 0) / float(inner["total"])))
        outer["done"] = round(min(total, float(outer.get("done") or 0) + fraction), 3)
        outer["total"] = total
        self._report_script_progress(status_text, {**summary, "progress": outer})

    @staticmethod
    def _finalize_generation_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
        result = snapshot.get("result")
        if isinstance(result, dict):
            snapshot["status"] = "done" if result.get("success") else ("interrupted" if result.get("cancelled") else "error")
            generated_files = list(result.get("generated_files", []) or [])
            if generated_files:
                snapshot["output_file"] = str(generated_files[-1])
        return _compact_job_snapshot(snapshot, keep_paths=True)  # the chat attaches generated_files / output_file

    def _wait_for_generation(self, initial: dict[str, Any], timeout_s: float | None = None, event_limit: int = 0) -> dict[str, Any]:
        job_id = str(initial.get("job_id", "") or "").strip()
        if not job_id:
            return initial
        self._active_job_id = job_id
        self._active_job_cancel_requested = False
        queue_triggered = self._queue_on_server
        cancel_requested = False
        snapshot = initial
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        from shared.deepy.media_pause import MediaToolPause
        job = self._server._wangp_jobs.get(job_id).job
        reported = 0.0
        try:
            with MediaToolPause(self.assistant_session, self.state["gen"], job.webui_client_ids, self.send_cmd) as media_pause:
                while True:
                    media_pause.poll()
                    live = snapshot.get("progress") if self._inline_calls else None
                    if isinstance(live, dict) and live.get("progress") is not None and time.monotonic() - reported >= 0.34 and not snapshot.get("done"):
                        # A script's or plugin action's generation shows its steps (the status bar's second line, and the action's bar).
                        reported = time.monotonic()
                        text = str(live.get("status") or live.get("phase") or "Running")[:120]
                        self._update_tool_progress("running", text, {"status": "running", "job_id": job_id, "progress": {"done": round(float(live["progress"]), 1), "total": 100, "text": text}})
                    if snapshot.get("webui_submission_ready") and not queue_triggered and not (cancel_requested or job.cancel_requested or self._is_interrupted()):  # a stopped job never reaches the queue
                        self.send_cmd("load_queue_trigger", {"job_id": job_id, "token": snapshot.get("webui_load_queue_token", "")})
                        queue_triggered = True
                        self._update_tool_progress("running", "Running", {"status": "running", "job_id": job_id})
                    if snapshot.get("done"):
                        return self._finalize_generation_snapshot(snapshot)
                    if self._is_interrupted() and not cancel_requested:
                        if not self._active_job_cancel_requested:
                            self._server._wangp_jobs.get(job_id).job.cancel()
                            self._active_job_cancel_requested = True
                        cancel_requested = True
                        self._update_tool_progress("running", "Stopping generation", {"status": "running", "job_id": job_id})
                        continue
                    if deadline is not None and not self.assistant_session.pause_requested and time.monotonic() >= deadline:
                        self._watch_background_job(job_id, trigger_queue=not queue_triggered)
                        return _compact_job_snapshot({**snapshot, "status": "timeout", "waiting_timed_out": True})
                    time.sleep(self._POLL_INTERVAL_SECONDS)
                    snapshot = self._server._wangp_jobs.get(job_id).snapshot(event_limit=event_limit)
        finally:
            self._active_job_id = ""
            self._active_job_cancel_requested = False

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return list(self._tool_defs)

    def get_system_instructions(self) -> str:
        return str(self._server_instructions or "").strip()

    def get_remote_execution_instructions(self) -> str:
        return (f"wangp_generate and wangp_postprocess wait for their job for up to {REMOTE_JOB_WAIT_S} seconds, or timeout_s when shorter. A job still running then returns status timeout with its job_id and keeps running: "
                "follow it with wangp_session wait_jobs or get_job, and never submit it again. With wait=false they return the job_id at once, so you can start several jobs and keep working; wangp_toolbox and wangp_plugins accept wait=false too. "
                "Before your final answer, wait for the jobs whose results you still need and report the rest as running.")

    def get_runtime_context(self) -> str:
        from shared.deepy.ui_settings import normalize_assistant_tool_ui_settings

        settings = normalize_assistant_tool_ui_settings(**self.assistant_session.tool_ui_settings)
        signature = f"{settings['model_speed']}:{settings['model_size']}"
        if signature == self.assistant_session.model_selection_runtime_signature:
            return ""
        self.assistant_session.model_selection_runtime_signature = signature
        return f"Current model selection preferences: speed={settings['model_speed']}, size={settings['model_size']}. These replace earlier standing preferences. Model search applies them automatically; explicit user choices still win and configured templates remain the normal choice."

    def get_system_context(self) -> str:
        from shared.deepy import ui_settings as deepy_ui_settings

        live_settings = self.assistant_session.tool_ui_settings
        settings = deepy_ui_settings.normalize_assistant_tool_ui_settings(**live_settings) if isinstance(live_settings, dict) and live_settings else deepy_ui_settings.get_persisted_assistant_tool_ui_settings()
        lines = [
            "Deepy Settings / General Properties determine whether dimensions, frame count or audio duration, and seed come from templates or from standing defaults.",
            f"Use properties defined in template settings files: {bool(settings['use_template_properties'])}.",
        ]
        if settings["use_template_properties"]:  # named as settings: "width" and "height" are not setting names
            lines.append("When using a Deepy template, retain its resolution, video_length, duration_seconds and seed unless the user or the task states otherwise (a production brief's format, for instance).")
        else:
            lines.extend([
                f"Default resolution: \"{int(settings['width'])}x{int(settings['height'])}\".",
                f"Default video_length: {int(settings['num_frames'])} frames.",
                f"Default audio duration_seconds: {int(settings['audio_duration'])}.",
                f"Default seed: {int(settings['seed'])}.",
            ])
            lines.append("Override compatible template/model defaults with the resolution, video_length, duration_seconds and seed above when the user does not specify them.")
        lines.append("A supplied speech or song that must be heard whole sets video_length instead: its duration in seconds, such as \"10.2s\", which WanGP fits to the model's frame steps.")
        lines.append("The generate tool fills omitted dimensions, duration, and seed from Deepy's standing defaults before model factory defaults; pass the template values explicitly to preserve them.")
        return "\n".join(lines)

    def get_tool_display_name(self, tool_name: str) -> str:
        normalized_name = str(tool_name or "").strip()
        plugin_tool = self._plugin_tools_by_name.get(normalized_name)
        if plugin_tool is not None:
            return plugin_tool.display_name
        if normalized_name == "wangp_notify":
            return "Send Notification"
        if normalized_name.startswith("wangp_"):
            normalized_name = normalized_name[len("wangp_"):]
        return normalized_name.replace("_", " ").strip().title()

    @staticmethod
    def get_tool_stream_label_fields(tool_name: str) -> tuple[str, ...]:
        return {
            "wangp_generate": ("model_type",),
            "wangp_postprocess": ("media",),
            "wangp_model": ("model_type",),
            "wangp_models": ("action",),
            "wangp_deepy_templates": ("tool_id",),
            "wangp_templates": ("task", "template"),
            "wangp_session": ("action",),
            "wangp_toolbox": ("action",),
            "wangp_io": ("action",),
        }.get(str(tool_name or "").strip(), ())

    @staticmethod
    def _generation_settings(source: Any) -> list[dict[str, Any]]:
        settings = []

        def collect(value: Any) -> None:
            if isinstance(value, list):
                for item in value:
                    collect(item)
                return
            if not isinstance(value, dict):
                return
            if isinstance(value.get("tasks"), list):
                collect(value["tasks"])
                return
            settings.append(value["params"] if isinstance(value.get("params"), dict) else value["settings"] if isinstance(value.get("settings"), dict) else value)

        collect(source)
        return settings

    def _model_label_metadata(self, model_type: str) -> tuple[str, dict[str, Any]]:
        model_type = str(model_type or "").strip()
        if not model_type:
            return "", {}
        metadata = self._api_session.get_model_metadata(model_type) or {}
        return str(metadata.get("name", "") or model_type.replace("_", " ").replace("-", " ").title()).strip(), metadata

    def _template_label_settings(self, template: Any) -> dict[str, Any]:
        from shared.mcp.server import _deepy_template_settings

        task, _, name = str(template).partition("/")
        try:
            return _deepy_template_settings(self._api_session, task, name.strip() or "default")["settings"]
        except (KeyError, ValueError):  # an unknown task or template: the call itself reports it, the label stays generic
            return {}

    def _generation_label_context(self, source: Any) -> tuple[str, str]:
        tasks = self._generation_settings(source)
        model_names = []
        media_kinds = []
        for settings in tasks:
            if settings.get("template") and not settings.get("model_type"):  # a template reference: name the model the template runs
                settings = {**self._template_label_settings(settings["template"]), **settings}
            model_name, metadata = self._model_label_metadata(settings.get("model_type", ""))
            if model_name and model_name not in model_names:
                model_names.append(model_name)
            try:
                image_mode = int(settings.get("image_mode", 0) or 0)
            except (TypeError, ValueError):
                image_mode = 0
            outputs = metadata.get("main_output", [])
            outputs = [outputs] if isinstance(outputs, str) else list(outputs or [])
            normalized_outputs = {str(output or "").strip().casefold() for output in outputs}
            if "image_mode" in settings:
                media_kinds.append("Image" if image_mode > 0 else "Video" if "video" in normalized_outputs else "Audio" if normalized_outputs == {"audio"} else "Media")
            else:
                media_kinds.append("Image" if normalized_outputs == {"image"} else "Video" if normalized_outputs == {"video"} else "Audio" if normalized_outputs == {"audio"} else "Media")
        count = len(tasks)
        if count == 0:
            media_label = "Media"
        elif len(set(media_kinds)) > 1:
            media_label = f"{count} Media Items"
        else:
            kind = media_kinds[0]
            media_label = kind if count == 1 else f"{count} Media Items" if kind == "Media" else f"{count} {kind}s"
        return media_label, " and ".join(model_names)

    def get_tool_transcript_label(self, tool_name: str, arguments: dict[str, Any] | None = None) -> str:
        arguments = dict(arguments or {})
        if tool_name not in self._tool_defs_by_name:
            return f"Unknown Tool - {self.get_tool_display_name(tool_name)}"
        if self._zero_tools is not None:
            arguments = self._zero_tools.resolve_tool_label_arguments(arguments)
        finish = assistant_chat._finish_tool_call_label
        short = assistant_chat._short_tool_label_value
        humanize = assistant_chat._humanize_tool_value
        if tool_name == "wangp_python":
            action = str(arguments.get("action", "") or "").strip()
            action_arguments = arguments.get("arguments")
            if not action and isinstance(action_arguments, dict) and isinstance(action_arguments.get("code"), str):
                action = "run"  # code alone is a run
            if not action:
                return "Explore Python Sandbox"
            action = {"execute": "run", "exec": "run", "run_code": "run", "run_python": "run", "python": "run", "code": "run", "reset": "restart", "reboot": "restart", "list_sandbox": "files"}.get(action, action)
            label = {"run": "Python", "restart": "Restart Python Sandbox", "files": "List Sandbox Files"}.get(action, f"Python {humanize(action)}")
            if action_arguments is None:
                return f"Get {label} Schema"
            if action == "run" and isinstance(action_arguments, dict):
                first = next((line.strip() for line in str(action_arguments.get("code", "") or "").splitlines() if line.strip()), "")
                title = " ".join(first.lstrip("# ").split())  # a title, not a path: keep "videos in @outputs/aaa" whole
                if first.startswith("#") and title:
                    return finish(f"Python: {title if len(title) <= 72 else title[:71].rstrip() + '…'}")
            return label
        if tool_name == "wangp_plugins":
            from shared.mcp.plugin_toolbox import transcript_label

            return transcript_label(arguments, humanize)
        if tool_name == "wangp_validate" and isinstance(arguments.get("settings"), list) and len(arguments["settings"]) > 1:
            return f"Validate {len(arguments['settings'])} Tasks"
        if tool_name == "wangp_web_search":
            action = str(arguments.get("action", "") or "").strip()
            action_arguments = arguments.get("arguments")
            if not action:
                return "Explore Web Search"
            label = {"search": "Search the Web", "news": "Search the News", "images": "Search Images", "videos": "Search Videos", "open": "Open Web Page", "download": "Download from the Web", "screenshot": "Screenshot Web Page", "cookies": "Manage Web Cookies", "lora_search": "Search LoRAs", "lora_install": "Install LoRA"}.get(action, f"Web {humanize(action)}")
            if action_arguments is None:
                return f"Get {label} Schema"
            if isinstance(action_arguments, dict):
                if action in ("open", "download") and isinstance(action_arguments.get("url"), str):
                    return finish(f"{label}: {short(action_arguments['url'].split('//', 1)[-1].split('/', 1)[0], 72)}")
                if action == "lora_install" and isinstance(action_arguments.get("id"), str):
                    return finish(f"{label}: {short(action_arguments['id'], 72)}")
                if isinstance(action_arguments.get("query"), str):
                    return finish(f"{label}: {short(action_arguments['query'], 72)}")
            return label
        if tool_name in {"wangp_toolbox", "wangp_io"}:
            action = str(arguments.get("action", "") or "").strip()
            action_arguments = arguments.get("arguments")
            if tool_name == "wangp_io":
                if action == "info" and isinstance(action_arguments, dict):
                    if isinstance(action_arguments.get("paths"), list):
                        return f"Get File Information for {len(action_arguments['paths'])} Files"
                    path = str(action_arguments.get("path", action_arguments.get("source", action_arguments.get("file_path", ""))))
                    if path.startswith(("visual:", "audio:")):
                        return finish(f"Get Media Information for {short(path)}")
                if action in {"rg", "edit", "append_text"}:
                    label = {"rg": "Search Files", "edit": "Edit Text", "append_text": "Append Text"}[action]
                    if not isinstance(action_arguments, dict):
                        return f"Get {label} Schema"
                    if action == "rg":
                        return finish(f"{label}: {action_arguments.get('command', action_arguments.get('arguments', ''))}")
                    return finish(f"{label} in {short(action_arguments.get('path', action_arguments.get('file_path')))}")
                return assistant_chat.build_io_tool_call_label(action, action_arguments if "arguments" in arguments else None)
            if not action:
                return "List Toolbox Content"
            if action == "media_settings":
                return "Get Media Settings Schema" if action_arguments is None else assistant_chat.build_tool_call_label("wangp_get_media_settings", action_arguments)
            if action_arguments is None:
                action_label = self._zero_tools.get_tool_transcript_label(action, {}) if self._zero_tools is not None else action.replace("_", " ").title()
                return f"Get {action_label} Schema"
            if self._zero_tools is not None:
                return self._zero_tools.get_tool_transcript_label(action, action_arguments)
        model_label = ""
        media_label = ""
        if tool_name == "wangp_generate":
            settings = arguments.get("settings")
            if settings is None and isinstance(arguments.get("arguments"), dict):
                settings = arguments["arguments"].get("source")
            if settings is None and (arguments.get("model_type") or arguments.get("template")):
                settings = arguments  # members already streamed from an incomplete settings object
            if settings is None:
                return self.get_tool_display_name(tool_name)
            media_label, model_label = self._generation_label_context(settings)
        elif tool_name == "wangp_templates":
            task = arguments.get("task")
            template = arguments.get("template")
            if template is not None:
                return finish(f"Read {short(template)} Template for {humanize(task or '')}")
            return finish(f"Browse Templates for {humanize(task)}" if task else "Browse Template Tasks")
        elif tool_name in {"wangp_deepy_templates", "wangp_list_gallery"} and "action" not in arguments:
            if tool_name == "wangp_deepy_templates":
                tool_name = "wangp_list_deepy_templates" if arguments.get("cursor") else "wangp_get_deepy_template_settings"
                arguments = {"template": "default", **arguments}
        elif tool_name in {"wangp_model", "wangp_models", "wangp_deepy_templates", "wangp_list_gallery", "wangp_postprocess", "wangp_session"}:
            if tool_name == "wangp_models" and arguments.get("query") is not None:
                return finish(f"Find Models for {short(arguments['query'])}")
            action = arguments.get("action")
            nested = arguments.get("arguments")
            describing = nested is None and tool_name not in {"wangp_model", "wangp_models"}
            nested = dict(nested or {})
            target = ""
            if tool_name == "wangp_model":
                model_label, _metadata = self._model_label_metadata(arguments.get("model_type", ""))
                target = short(model_label)
            elif tool_name == "wangp_postprocess":
                target = short(arguments.get("media"))
            if not action:
                return finish(f"Explore {self.get_tool_display_name(tool_name)}{f' for {target}' if target else ''}")
            if describing:
                return finish(f"Get {humanize(action)} Schema{f' for {target}' if target else ''}")
            if tool_name == "wangp_model":
                if action in {"capabilities", "definition", "defaults"}:
                    subject = short(nested.get("property")) if action == "definition" else ""
                    return finish(f"Read {subject or humanize(action)} for {target}")
                if action == "profiles" and "setting_id" not in nested:
                    return finish(f"Read Accelerator Profiles for {target}")
                tool_name = {"saved_settings": "wangp_model_settings", "user_settings": "wangp_model_settings", "presets": "wangp_model_settings", "profiles": "wangp_model_settings", "loras": "wangp_list_loras"}.get(action, tool_name)
                nested["model_type"] = arguments["model_type"]
            elif tool_name == "wangp_deepy_templates":
                tool_name = {"deepy_templates": "wangp_list_deepy_templates", "deepy_template_settings": "wangp_get_deepy_template_settings"}.get(action, tool_name)
            elif tool_name == "wangp_session":
                tool_name = f"wangp_{action}"
            elif tool_name == "wangp_postprocess":
                return finish(f"Run {humanize(action)} on {target}")
            arguments = nested
        elif tool_name == "mcp_resource" and not arguments.get("uri") and arguments.get("query"):
            return finish(f"Find Documentation for {short(arguments['query'], 72)}")
        elif arguments.get("model_type"):
            model_label, _metadata = self._model_label_metadata(arguments["model_type"])
        return assistant_chat.build_tool_call_label(tool_name, arguments, base_label=self.get_tool_display_name(tool_name), model_label=model_label, media_label=media_label)

    def get_tool_template_filename(self, tool_name: str) -> str:
        return ""

    def get_tool_variant(self, tool_name: str) -> str:
        return ""

    def get_tool_policy(self, tool_name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        call_arguments = dict(arguments or {})
        plugin_tool = self._plugin_tools_by_name.get(tool_name)
        if plugin_tool is not None:
            return {"pause_runtime": plugin_tool.pause_runtime, "pause_reason": plugin_tool.pause_reason}
        if tool_name == "wangp_plugins":
            from shared.mcp.plugin_toolbox import call_policy

            return call_policy(call_arguments)
        if tool_name == "wangp_generate":
            return {"pause_runtime": isinstance(call_arguments.get("settings"), (dict, list, str)), "pause_reason": "tool"}
        if tool_name == "wangp_postprocess":
            return {"pause_runtime": bool(call_arguments.get("action")) and isinstance(call_arguments.get("arguments"), dict), "pause_reason": "tool"}
        if tool_name == "wangp_toolbox":
            action = str(call_arguments.get("action", "") or "").strip()
            if not action or call_arguments.get("arguments") is None:
                return {"pause_runtime": False, "pause_reason": "tool"}
            return {"pause_runtime": False, "pause_reason": "tool"} if action == "media_settings" else self._zero_tools.get_tool_policy(action, call_arguments["arguments"])
        return {"pause_runtime": False, "pause_reason": "tool"}

    def validate_tool_call(self, tool_name: str, arguments: dict[str, Any]) -> str:
        schema = self._tool_defs_by_name.get(str(tool_name or "").strip())
        if schema is None:
            return f"Unknown MCP tool: {tool_name}"
        parameters = schema["function"].get("parameters", {})
        properties = parameters.get("properties", {})
        toolbox = "action" in properties and "arguments" in properties
        inner = arguments.get("arguments")
        if toolbox and isinstance(inner, dict) and inner.get("action") == arguments.get("action") is not None and set(inner) == {"action", "arguments"} and isinstance(inner["arguments"], dict):
            arguments["arguments"] = inner = inner["arguments"]  # the whole call repeated inside arguments under the same action
        if toolbox and arguments.get("action") is not None and isinstance(inner, dict) and "arguments" in inner and inner["arguments"] is None:
            arguments["arguments"] = inner = None  # the action's contract, asked for with its null written one level too deep
        if toolbox and arguments.get("action") is None and isinstance(inner, dict):
            if set(arguments) == {"arguments"} and set(inner) & set(properties) - {"action", "arguments"} and set(inner) <= set(properties):
                arguments.pop("arguments")  # the whole call nested in arguments with its other top-level parameters (wangp_plugins' plugin)
                arguments.update(inner)
                inner = arguments.get("arguments")
            if isinstance(inner, dict) and set(inner) == {"arguments"} and isinstance(inner["arguments"], dict):
                arguments["arguments"] = inner = inner["arguments"]  # arguments nested twice
            if isinstance(inner, dict) and isinstance(inner.get("action"), str) and set(inner) <= {"action", "arguments"}:
                arguments.update(action=inner["action"], arguments=inner.get("arguments", {}))  # the whole call nested in arguments: unwrap it
            elif isinstance(inner, dict) and isinstance(inner.get("action"), str):
                arguments["action"] = inner.pop("action")  # the action written among its own arguments
        unknown = set(arguments) - set(properties)
        if set(properties) == {"action", "arguments"} and isinstance(arguments.get("arguments"), dict) and unknown and not unknown & set(arguments["arguments"]):
            for name in unknown:  # action parameters written beside action and arguments, checked there by the action's schema (a flat call stays refused)
                arguments["arguments"][name] = arguments.pop(name)
            unknown = set()
        if not self.allow_async and tool_name in {"wangp_generate", "wangp_postprocess"}:
            unknown -= {"wait", "timeout_s", "event_limit"}
        unknown = sorted(unknown)
        if unknown and (toolbox or parameters.get("additionalProperties") is False):
            hint = ' Put action at the top level and action parameters inside arguments: {"action":"<action>","arguments":{...}}.' if toolbox else " Pass the declared parameters directly, without action or arguments." if {"action", "arguments"} & set(unknown) else " Use only the declared parameters."
            if tool_name == "wangp_generate" and "settings" not in unknown:
                hint = ' Put every generation setting inside settings: {"settings": {"model_type": "...", "prompt": "...", ...}}.'
            if tool_name == "wangp_toolbox" and arguments.get("action") == "inspect_media" and "media" in unknown:
                hint = ' Put media_id (one visual) or media_ids (a list) inside arguments alongside question. Read the contract with {"action":"inspect_media","arguments":null}.'
            named = arguments.get("tool")
            if "tool" in unknown and isinstance(named, str) and named != tool_name and named in self._tool_defs_by_name:  # the tool it meant, written as a parameter of this one
                hint = f' {named} is a tool of its own: call {named} with {{"action": "{arguments.get("action") or "<action>"}", "arguments": {{...}}}}, not {tool_name}.'
            return f"Unexpected top-level parameters: {', '.join(unknown)}. Nothing was executed." + hint
        for name, parameter in parameters.get("properties", {}).items():
            value = arguments.get(name)
            kinds = {kind for option in (parameter, *parameter.get("anyOf", [])) for kind in (option.get("type") if isinstance(option.get("type"), list) else [option.get("type")])}
            if not isinstance(value, str) or not kinds & {"object", "array"}:
                continue
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError as error:
                if value.lstrip()[:1] in ("{", "["):  # malformed JSON object/array: point at the fault instead of a generic type error
                    return f"{name} is not valid JSON: {error.msg} at character {error.pos}, near {value[max(0, error.pos - 40):error.pos + 20]!r}. Nothing was executed; fix that spot and repeat the call with {name} as an object."
                continue
            if isinstance(decoded, dict) and "object" in kinds or isinstance(decoded, list) and "array" in kinds:
                arguments[name] = decoded
        if tool_name == "wangp_python" and not arguments.get("action") and isinstance(arguments.get("arguments"), dict) and isinstance(arguments["arguments"].get("code"), str):
            arguments["action"] = "run"  # code alone can only mean a run
        if tool_name == "wangp_web_search" and not arguments.get("action") and isinstance(arguments.get("arguments"), dict):
            if isinstance(arguments["arguments"].get("url"), str):
                arguments["action"] = "open"
            elif isinstance(arguments["arguments"].get("query"), str):
                arguments["action"] = "search"
        if toolbox and arguments.get("arguments") is not None:
            nested = arguments["arguments"]
            if not isinstance(nested, dict):
                hint = ' For rg, use {"action":"rg","arguments":{"command":"-n pattern -- @documents/file.txt"}}.' if tool_name == "wangp_io" and arguments.get("action") == "rg" else ""
                return 'arguments must be an object: {"action":"<action>","arguments":{...}}.' + hint
            if not arguments.get("action") and nested:
                if tool_name == "wangp_plugins":  # name the plugin's call as it should be written, with the caller's own plugin
                    plugin = json.dumps(arguments.get("plugin") or "<plugin>")
                    return (f'No action was named. Write the call flat, the action beside plugin and arguments: {{"plugin": {plugin}, "action": "<action>", "arguments": {{...}}}}; '
                            f'{{"plugin": {plugin}}} alone lists its actions. Nothing was executed.')
                likely = next((hint for key, hint in (("old_string", "edit"), ("text", "write_text (mode 'overwrite' replaces a file) or append_text"), ("command", "rg")) if key in nested), "")
                return ('No action was named: write it at the top level, beside arguments: {"action":"<action>","arguments":{...}}' + (f"; with these arguments: {likely}" if likely and tool_name == "wangp_io" else "")
                        + ". Nothing was executed.")
        for parameter_name in parameters.get("required", []) or []:
            if parameter_name not in arguments or arguments[parameter_name] is None:
                return f"{parameter_name} is required."
        return ""

    def extract_tool_calls(self, raw_text: str) -> list[dict[str, Any]]:
        from shared.prompt_enhancer.qwen35_assistant_runtime import extract_tool_calls

        parameters = {name: set(schema["function"]["parameters"].get("properties", {})) for name, schema in self._tool_defs_by_name.items()}
        return extract_tool_calls(raw_text, tool_parameters=parameters, preserve_unknown_parameters=True)

    def infer_tool_calls(self, raw_text: str) -> list[dict[str, Any]]:
        from shared.deepy.engine import DeepyZeroTools

        return DeepyZeroTools.infer_tool_calls(self, raw_text)

    def _watch_background_job(self, job_id, *, trigger_queue=True):
        if job_id in self._background_jobs:
            return
        record = self._server._wangp_jobs.get(job_id)
        self._background_jobs[job_id] = record.job

        def finish():
            try:
                if record.job.wait_for_webui_submission_or_completion() and trigger_queue and not self._queue_on_server and not record.job.done and not record.job.cancel_requested:
                    self.send_cmd("load_queue_trigger", {"job_id": job_id, "token": record.job.webui_load_queue_token})
                record.wait()
                self._remember_generated_media(record.snapshot(event_limit=0))
            finally:
                self._background_jobs.pop(job_id, None)

        threading.Thread(target=finish, name=f"deepy-job-{job_id}", daemon=True).start()

    def call(self, tool_name: str, arguments: dict[str, Any], *, inline: bool = False, cascade: bool = False) -> dict[str, Any]:
        """Execute a tool as the model would. inline=True serves a sandbox script: the call is validated here and the WanGP tool runs on this thread,
        because the MCP request loop is busy running the script that asks; it gets its own transcript card below the script's (run_script_tool,
        set by the engine), its progress also feeds the script's live summary, and everything after execution (generation waits, media bookkeeping) is shared."""
        if not inline:
            if tool_name == "wangp_plugins":
                self._plugin_progress = None  # a new action starts its card's progress afresh
            result = self._call(tool_name, arguments)
            self._hint_script_loop(tool_name, arguments, result)
            self._hint_repeated_failure(tool_name, arguments, result)
            return result
        arguments = dict(arguments)
        if tool_name == "wangp_web_search" and tool_name not in self._tool_defs_by_name:  # say why, or the model searches the docs for another way online
            return {"status": "error", "tool": tool_name, "error": "Web search is turned off, so scripts cannot reach the web: tell the user it can be enabled in Configuration > Deepy (Enable Web Search). There is no other way online."}
        problem = self.validate_tool_call(tool_name, arguments)
        route = self._tool_routes.get(tool_name)
        if problem or route is None or route[0] != "wangp":
            return {"status": "error", "tool": tool_name, "error": problem or f"{tool_name} is not one of WanGP's own tools."}
        policy = self.get_tool_policy(tool_name, arguments)
        if policy["pause_runtime"] and self.pause_runtime is not None:  # Deepy still holds the GPU while its script runs: a generation would wait for it forever
            self.pause_runtime(pause_reason=policy["pause_reason"])  # saves Deepy's state and unloads it; "vision" keeps the text model when it fits
        sandbox = self._server._wangp_python_sandbox

        def run(progress: Callable[..., None]) -> dict[str, Any]:
            def inner(status=None, status_text=None, result=None):
                sandbox.inner_status(str(status_text or ""))
                progress(status, status_text, result)
                if cascade:  # cascade=True: a plugin action's tool call, whose generation steps also move the action's bar
                    self._cascade_plugin_progress(result)

            saved = self._tool_progress_callback, getattr(self._zero_tools, "_tool_progress_callback", None)
            self._script_progress_target, self._tool_progress_callback = saved[0], inner
            self._inline_calls += 1
            web = getattr(self._server, "_wangp_web_search", None)
            if web is not None:  # a download's progress goes to its own card, not to the script's
                web.progress = lambda status_text, summary: inner("running", status_text, self.file_access_policy.virtualize_result(summary))
            if self._zero_tools is not None:
                self._zero_tools._tool_progress_callback = inner
                self._zero_tools._set_status = lambda text=None, kind="working": progress(None, text, None)  # the status bar keeps the script on its first line, the tool on its second
            try:
                return self._call(tool_name, arguments, route=route)
            finally:
                self._inline_calls -= 1
                self._tool_progress_callback, self._script_progress_target = saved[0], None
                if web is not None:
                    web.progress = self._report_script_progress
                if self._zero_tools is not None:
                    self._zero_tools._tool_progress_callback = saved[1]
                    self._zero_tools.__dict__.pop("_set_status", None)

        return run(lambda *_args: None) if self.run_script_tool is None else self.run_script_tool(tool_name, arguments, run)

    def _call(self, tool_name: str, arguments: dict[str, Any], route: tuple[str, str] | None = None) -> dict[str, Any]:
        ignored = []
        if not self.allow_async and tool_name in {"wangp_generate", "wangp_postprocess"}:
            arguments = dict(arguments)
            if isinstance(arguments.get("arguments"), dict):
                arguments["arguments"] = dict(arguments["arguments"])
            for values in (arguments, arguments.get("arguments")):
                if isinstance(values, dict):
                    for name in ("wait", "timeout_s", "event_limit"):
                        if name in values:
                            values.pop(name)
                            if name not in ignored:
                                ignored.append(name)
        if route is not None:  # inline: run the WanGP tool on this thread
            from shared.mcp.server import call_tool_inline

            self._remember_gallery_input_references(tool_name, arguments)
            result = self.file_access_policy.virtualize_result(call_tool_inline(self._server, route[1], arguments))
            self._remember_gallery_download_references(tool_name, result)
        else:
            result = self._call_mcp_tool(tool_name, arguments)
        if tool_name in {"wangp_generate", "wangp_postprocess"} and result.get("job_id") and not result.get("recovered"):
            from shared.mcp.v2 import JOB_CONTROLS

            # job controls sit at the top level; post-processing calls may still carry them inside arguments
            action_arguments = arguments if tool_name == "wangp_generate" else {**{key: value for key, value in arguments["arguments"].items() if key in JOB_CONTROLS}, **{key: arguments[key] for key in JOB_CONTROLS if key in arguments}}
            if action_arguments.get("wait", True):
                timeout_s = action_arguments.get("timeout_s")
                if self.allow_async and route is None:  # scripts and plugin actions wait for their own calls to end
                    timeout_s = REMOTE_JOB_WAIT_S if timeout_s is None else min(timeout_s, REMOTE_JOB_WAIT_S)
                result = self._wait_for_generation(result, timeout_s, action_arguments.get("event_limit", 0))
                if result.get("waiting_timed_out"):
                    result["info"] = "The job keeps running: follow it with wangp_session wait_jobs. Do not submit it again."
            else:
                self._watch_background_job(result["job_id"])
        elif tool_name in {"wangp_toolbox", "wangp_plugins"} and isinstance(result, dict) and result.get("job_id") and not result.get("done"):  # stops with the chat Stop button
            self._background_jobs = {job_id: job for job_id, job in self._background_jobs.items() if not job.done}
            self._background_jobs[result["job_id"]] = self._server._wangp_jobs.get(result["job_id"]).job
        if ignored:
            result["info"] = f"Ignored {', '.join(ignored)}: WanGP manages waiting for generation and post-processing."
        self._remember_generated_media(result)
        if self.allow_async and tool_name in {"wangp_generate", "wangp_postprocess", "wangp_session"}:  # the browser-queue handshake of running jobs is not for the model
            for job in (result, *result.get("jobs", [])):
                job.pop("webui_submission_ready", None)
                job.pop("webui_load_queue_token", None)
        if tool_name in {"wangp_generate", "wangp_postprocess"}:
            from shared.mcp.v2 import public_media_result
            result = public_media_result(result)
        return self.file_access_policy.virtualize_result(result)

    def _hint_script_loop(self, tool_name: str, arguments: dict[str, Any], result: dict[str, Any]) -> None:
        """The model's fourth successful call in a row of one toolbox or file action says once that a Python script loops over the remaining items in
        one call. Failed calls do not count: retries of one call are not items of a batch."""
        if isinstance(result, dict) and result.get("status") == "error":
            return
        action = arguments.get("action") if isinstance(arguments.get("arguments"), dict) else None
        key = (tool_name, action)
        count = self._action_streak[1] + 1 if getattr(self, "_action_streak", (None, 0))[0] == key else 1
        self._action_streak = (key, count)
        if count == 4 and action and tool_name in {"wangp_toolbox", "wangp_io"} and "wangp_python" in self._tool_defs_by_name and isinstance(result, dict):
            helper = "wangp.io" if tool_name == "wangp_io" else "wangp.toolbox"
            result["hint"] = (str(result.get("hint") or "") + f" Fourth {action} call in a row: for the remaining items, one wangp_python script can loop over them ({helper}({action!r}, ...) per item, wangp.progress) and return only the summary, instead of one call each.").strip()

    def _hint_repeated_failure(self, tool_name: str, arguments: dict[str, Any], result: dict[str, Any]) -> None:
        """A failed call identical to the previous failed one: smaller models copy their last call instead of changing it, so the result says so
        and names what the call holds (a missing argument is then visible)."""
        failed = isinstance(result, dict) and result.get("status") == "error"
        signature = (tool_name, json.dumps(arguments, sort_keys=True, default=str)) if failed else None
        repeated = failed and signature == getattr(self, "_last_failed_call", None)
        self._last_failed_call = signature
        if repeated:
            sent = arguments["arguments"] if isinstance(arguments.get("arguments"), dict) else arguments
            result["hint"] = (str(result.get("hint") or "") + f" This call is identical to your previous one, which failed the same way: change the call itself before retrying. It holds only {', '.join(sorted(sent)) or 'nothing'}.").strip()

    def _get_selected_media_record_from_source(self, source: str, requested_media_type: str = "all"):
        return None, None
