# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import os
import base64
import json
import mimetypes
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

import requests

from shared.llm_io import llm_io_enabled, log_llm_io, media_descriptor

from .base import BackendEvent, EventCallback, StopCallback, ToolCallback
from .usage import opencode_usage_data
from shared.mcp.bridge import MCPHttpBridge, build_tool_proxy


OPENCODE_PROGRESS_INSTRUCTIONS = """For work with multiple meaningful steps, send concise user-facing progress updates as short text messages before and between tool calls. Keep progress updates separate from the final answer, and do not reveal hidden reasoning."""
OPENCODE_MCP_SERVER = "wangp"  # OpenCode declares MCP tools as <server>_<tool>
OPENCODE_TOOL_NAMES_INSTRUCTIONS = f"""Tools that these instructions name without the {OPENCODE_MCP_SERVER}_ prefix are declared with it: call mcp_resource as {OPENCODE_MCP_SERVER}_mcp_resource."""
OPENCODE_RETRY_LIMIT_S = 300  # a provider retry further away than this (a usage limit) ends the turn with its message
OPENCODE_TOOL_TIMEOUT_MS = 24 * 3600 * 1000  # OpenCode fails an MCP call after 60 s by default; WanGP bounds job waits itself and scripts and plugin actions can take hours


def _resolve_opencode_executable(configured: str) -> str:
    configured = str(configured or "opencode").strip() or "opencode"
    if configured.lower() != "opencode":
        return shutil.which(configured) or configured
    candidates = []
    if os.name == "nt":
        appdata = os.environ.get("APPDATA", "")
        if appdata:
            candidates.extend((os.path.join(appdata, "npm", "opencode.cmd"), os.path.join(appdata, "npm", "opencode.exe")))
        candidates.extend(filter(None, (shutil.which("opencode.cmd"), shutil.which("opencode.exe"), shutil.which("opencode"))))
    else:
        candidates.append(shutil.which("opencode"))
    return next((candidate for candidate in candidates if candidate and os.path.isfile(candidate)), configured)


def _opencode_launch_command(executable: str, port: int) -> list[str]:
    args = [executable, "serve", "--hostname", "127.0.0.1", "--port", str(port)]
    if os.name == "nt" and Path(executable).suffix.lower() in {".cmd", ".bat"}:
        return [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/s", "/c", subprocess.list2cmdline(args)]
    return args


def _opencode_tools(tools: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """The tools under the names to register: OpenCode's server prefix then gives back the names that prompts and tool results use. Also returns the name WanGP knows each one by."""
    prefix = f"{OPENCODE_MCP_SERVER}_"
    names = {str(dict(definition.get("function", {}) or {}).get("name", "") or "").strip() for definition in tools}
    renamed, originals = [], {}
    for definition in tools:
        function = dict(definition.get("function", {}) or {})
        name = str(function.get("name", "") or "").strip()
        if name:
            exposed = f"plugin_{name}" if prefix + name in names else name.removeprefix(prefix)  # a plugin's tool named like one of WanGP's would be declared under its name
            originals[exposed] = name
            renamed.append({**definition, "function": {**function, "name": exposed}})
    return renamed, originals


class OpenCodeBackend:
    engine = "opencode"

    def __init__(self, profile: dict[str, Any], toolbox: Any | None = None) -> None:
        self.profile = dict(profile or {})
        self.base_url = str(self.profile.get("base_url", "http://127.0.0.1:4096") or "").rstrip("/")
        self.toolbox = toolbox
        self._process: subprocess.Popen[str] | None = None
        self._session_id = ""
        self._abort = False
        self._bridge: MCPHttpBridge | None = None
        self._call_tool: ToolCallback | None = None
        self._model_catalog = list(self.profile.get("model_catalog", []) or [])
        self._temp_dir = tempfile.TemporaryDirectory(prefix="wangp-opencode-", ignore_cleanup_errors=True)

    @staticmethod
    def _auth():
        password = os.environ.get("OPENCODE_SERVER_PASSWORD", "")
        return (os.environ.get("OPENCODE_SERVER_USERNAME", "opencode"), password) if password else None

    def _request(self, method: str, path: str, **kwargs):
        request_payload = {"method": method, "path": path}
        if "params" in kwargs:
            request_payload["query"] = kwargs["params"]
        if "json" in kwargs:
            request_payload["body"] = kwargs["json"]
        log_llm_io("OUT", self.engine, "http", request_payload)
        try:
            response = requests.request(method, self.base_url + path, timeout=kwargs.pop("timeout", 300), auth=self._auth(), **kwargs)
            response.raise_for_status()
            if not response.content:
                log_llm_io("IN", self.engine, "http", {"method": method, "path": path, "status": response.status_code, "body": ""})
                return {}
            result = response.json()
            log_llm_io("IN", self.engine, "http", {"method": method, "path": path, "status": response.status_code, "body": result})
            return result
        except requests.RequestException as exc:
            log_llm_io("IN", self.engine, "http-error", {"method": method, "path": path, "error": str(exc)})
            raise RuntimeError(f"OpenCode server request failed at {self.base_url}{path}: {exc}") from exc

    @staticmethod
    def _utf8_sse_lines(response):
        response.encoding = "utf-8"
        return response.iter_lines(decode_unicode=True)

    def _ensure_server(self) -> None:
        try:
            self._request("GET", "/global/health", timeout=2)
            return
        except RuntimeError:
            pass
        parsed = urlparse(self.base_url)
        if (parsed.hostname or "").lower() not in {"127.0.0.1", "localhost", "::1"}:
            raise RuntimeError(f"OpenCode server is not reachable at {self.base_url}.")
        configured = str(self.profile.get("executable", "opencode") or "opencode").strip()
        executable = _resolve_opencode_executable(configured)
        child_env = os.environ.copy()
        if config := str(self.profile.get("config", "") or "").strip():
            child_env["OPENCODE_CONFIG_CONTENT"] = config
        try:
            self._process = subprocess.Popen(_opencode_launch_command(executable, parsed.port or 4096), cwd=self._temp_dir.name, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=child_env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0)
        except (FileNotFoundError, PermissionError, OSError) as exc:
            raise RuntimeError(f"OpenCode is not reachable and could not be started with '{configured} serve'. Install/configure OpenCode or correct its executable and server URL. {exc}") from exc
        for _ in range(50):
            time.sleep(0.1)
            try:
                self._request("GET", "/global/health", timeout=1)
                return
            except RuntimeError:
                continue
        raise RuntimeError("OpenCode server did not become ready.")

    def list_models(self) -> list[dict[str, Any]]:
        self._ensure_server()
        response = self._request("GET", "/config/providers", timeout=30)
        providers = response.get("providers", []) if isinstance(response, dict) else []
        defaults = response.get("default", {}) if isinstance(response, dict) else {}
        if not isinstance(providers, list) or not isinstance(defaults, dict):
            raise RuntimeError("OpenCode returned an invalid provider catalog.")
        catalog = []
        for provider in providers:
            if not isinstance(provider, dict):
                continue
            provider_id = str(provider.get("id", "") or "").strip()
            provider_name = str(provider.get("name", provider_id) or provider_id).strip()
            models = provider.get("models", {})
            if not provider_id or not isinstance(models, dict):
                continue
            for model_id, model_data in models.items():
                model_data = model_data if isinstance(model_data, dict) else {}
                model_id = str(model_data.get("id", model_id) or "").strip()
                if not model_id:
                    continue
                variants = model_data.get("variants", {})
                variants = list(variants) if isinstance(variants, dict) else []
                limits = model_data.get("limit", {})
                limits = limits if isinstance(limits, dict) else {}
                try:
                    context_window = max(0, int(limits.get("context", 0) or 0))
                except (TypeError, ValueError):
                    context_window = 0
                catalog.append({"provider": provider_id, "provider_name": provider_name, "model": model_id, "display_name": str(model_data.get("name", model_id) or model_id).strip(), "is_default": defaults.get(provider_id) == model_id, "context_window": context_window, "reasoning_efforts": variants})
        if not catalog:
            raise RuntimeError("OpenCode did not report any available models. Configure or authenticate a provider in OpenCode first.")
        self._model_catalog = catalog
        return catalog

    def _ensure_session(self, system_prompt: str, tools: Sequence[dict[str, Any]]) -> None:
        self._ensure_server()
        if self._session_id:
            return
        result = self._request("POST", "/session", json={"title": "WanGP Deepy"})
        self._session_id = str(result.get("id", "") or "")
        if not self._session_id:
            raise RuntimeError("OpenCode did not return a session id.")
        if self.toolbox is not None:
            exposed, originals = _opencode_tools(tools)
            proxy = build_tool_proxy(exposed, lambda name, arguments: self._call_tool(originals[name], arguments) if self._call_tool is not None else {"status": "error", "tool": originals[name], "error": "WanGP tool bridge is not active."})
            self._bridge = MCPHttpBridge(proxy)
            url = self._bridge.start()
            try:
                self._request("POST", "/mcp", json={"name": OPENCODE_MCP_SERVER, "config": {"type": "remote", "url": url, "oauth": False, "codemode": False, "timeout": OPENCODE_TOOL_TIMEOUT_MS}}, timeout=15)
            except Exception:
                self._bridge.close()
                self._bridge = None
                raise

    def _model(self) -> dict[str, str] | None:
        provider, model = str(self.profile.get("provider", "") or "").strip(), str(self.profile.get("model", "") or "").strip()
        return {"providerID": provider, "modelID": model} if provider and model else None

    def _context_window(self, info: dict[str, Any] | None = None) -> int:
        info = info if isinstance(info, dict) else {}
        provider = str(info.get("providerID", self.profile.get("provider", "")) or "").strip()
        model = str(info.get("modelID", self.profile.get("model", "")) or "").strip()
        for entry in self._model_catalog:
            if isinstance(entry, dict) and entry.get("provider") == provider and entry.get("model") == model:
                try:
                    return max(0, int(entry.get("context_window", 0) or 0))
                except (TypeError, ValueError):
                    return 0
        return 0

    def _usage(self, response: dict[str, Any]) -> dict[str, Any]:
        usage = opencode_usage_data(response)
        if not usage:
            return {}
        info = response.get("info", response)
        info = info if isinstance(info, dict) else {}
        tokens = info.get("tokens", {})
        tokens = tokens if isinstance(tokens, dict) else {}
        try:
            usage["context_tokens"] = max(0, int(tokens.get("total", usage["total_tokens"]) or usage["total_tokens"]))
        except (TypeError, ValueError):
            usage["context_tokens"] = usage["total_tokens"]
        usage["context_window"] = self._context_window(info)
        return usage

    @staticmethod
    def _answer_text(payload: Any) -> str:
        parts = payload.get("parts", []) if isinstance(payload, dict) else []
        return "".join(str(part.get("text", "") or "") for part in parts if isinstance(part, dict) and part.get("type") == "text").strip()

    def run_turn(self, text: str, *, system_prompt: str, tools: Sequence[dict[str, Any]], images: Sequence[str], on_event: EventCallback, call_tool: ToolCallback, should_stop: StopCallback) -> str:
        self._call_tool = call_tool
        self._ensure_session(system_prompt, tools)
        if not self._model_catalog:
            self.list_models()
        self._abort = False
        body: dict[str, Any] = {
            "parts": [{"type": "text", "text": str(text or "")}],
            "system": "\n\n".join(part for part in (str(system_prompt or "").strip(), OPENCODE_TOOL_NAMES_INSTRUCTIONS if tools else "", OPENCODE_PROGRESS_INSTRUCTIONS) if part),
            "tools": {"*": False, f"{OPENCODE_MCP_SERVER}_*": True},  # OpenCode's own tools stay off, including the ones later versions add
        }
        if llm_io_enabled():
            log_llm_io("OUT", self.engine, "turn-context", {"system_prompt": system_prompt, "user_text": str(text or ""), "images": [media_descriptor(path) for path in images], "tools": list(tools), "model": self._model() or "Automatic"})
        if model := self._model():
            body["model"] = model
        if effort := str(self.profile.get("reasoning_effort", "") or "").strip():
            body["variant"] = effort
        for path in images:
            mime = mimetypes.guess_type(str(path))[0] or "image/png"
            with open(path, "rb") as reader:
                url = f"data:{mime};base64,{base64.b64encode(reader.read()).decode('ascii')}"
            body["parts"].append({"type": "file", "url": url, "mime": mime})
        result_box: dict[str, Any] = {}
        error_box: list[BaseException] = []
        event_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        stream_stop = threading.Event()
        stream_ready = threading.Event()
        stream_response: list[Any] = []

        def read_events():
            try:
                response = requests.get(self.base_url + "/event", headers={"Accept": "text/event-stream"}, stream=True, timeout=(10, None), auth=self._auth())
                response.raise_for_status()
                stream_response.append(response)
                for raw_line in self._utf8_sse_lines(response):
                    if stream_stop.is_set():
                        break
                    line = str(raw_line or "")
                    if not line.startswith("data:"):
                        continue
                    try:
                        event = json.loads(line[5:].strip())
                    except Exception:
                        continue
                    log_llm_io("IN", self.engine, "sse-event", event)
                    stream_ready.set()
                    event_queue.put(event)
            except Exception:
                stream_ready.set()

        def request_turn():
            try:
                result_box.update(self._request("POST", f"/session/{self._session_id}/message", json=body, timeout=(10, None)))  # the answer arrives when the whole turn ends
            except BaseException as exc:
                error_box.append(exc)

        event_worker = threading.Thread(target=read_events, name="wangp-opencode-events", daemon=True)
        event_worker.start()
        stream_ready.wait(timeout=3)
        worker = threading.Thread(target=request_turn, name="wangp-opencode-turn", daemon=True)
        worker.start()
        message_roles: dict[str, str] = {}
        part_messages: dict[str, str] = {}
        part_types: dict[str, str] = {}
        snapshots: dict[str, str] = {}
        pending: dict[str, list[str]] = {}
        last_usage: dict[str, Any] = {}
        provider_error = ""

        def emit_usage(response: dict[str, Any]) -> None:
            nonlocal last_usage
            usage = self._usage(response)
            if usage and usage != last_usage:
                last_usage = usage
                on_event(BackendEvent("usage", data=usage))

        def emit(kind: str, delta: str, part_id: str) -> None:
            if not delta:
                return
            snapshots[part_id] = snapshots.get(part_id, "") + delta
            on_event(BackendEvent(kind, delta, {"item_id": part_id, "summary_index": 0} if kind == "reasoning_delta" else {"item_id": part_id}))

        def consume(event: dict[str, Any]) -> None:
            nonlocal provider_error
            event_type = str(event.get("type", "") or "")
            properties = dict(event.get("properties", {}) or {})
            part = dict(properties.get("part", {}) or {})
            session_id = str(part.get("sessionID", properties.get("sessionID", "")) or "")
            if session_id and session_id != self._session_id:
                return
            if event_type == "message.updated":
                info = dict(properties.get("info", {}) or {})
                message_id = str(info.get("id", "") or "")
                if message_id:
                    message_roles[message_id] = str(info.get("role", "") or "")
                if info.get("role") == "assistant":
                    emit_usage({"info": info})
            elif event_type == "message.part.updated":
                part_id = str(part.get("id", "") or "")
                part_type = str(part.get("type", "") or "")
                message_id = str(part.get("messageID", "") or "")
                if not part_id or part_type not in {"text", "reasoning"}:
                    return
                part_messages[part_id] = message_id
                part_types[part_id] = part_type
                if message_roles.get(message_id) != "assistant":
                    pending.pop(part_id, None)
                    return
                for delta in pending.pop(part_id, []):
                    emit("reasoning_delta" if part_type == "reasoning" else "commentary_delta", delta, part_id)
                full_text = str(part.get("text", "") or "")
                previous = snapshots.get(part_id, "")
                if full_text.startswith(previous):
                    missing = full_text[len(previous):]
                    emit("reasoning_delta" if part_type == "reasoning" else "commentary_delta", missing, part_id)
                    if not missing and properties.get("delta"):
                        emit("reasoning_delta" if part_type == "reasoning" else "commentary_delta", str(properties["delta"]), part_id)
                elif full_text and full_text != previous:
                    snapshots[part_id] = full_text
                    if part_type == "text":
                        on_event(BackendEvent("commentary_replace", full_text, {"item_id": part_id}))
            elif event_type == "message.part.delta":
                part_id = str(properties.get("partID", "") or "")
                message_id = str(properties.get("messageID", part_messages.get(part_id, "")) or "")
                delta = str(properties.get("delta", "") or "")
                part_type = part_types.get(part_id, "")
                if message_roles.get(message_id) != "assistant":
                    return
                if part_type:
                    emit("reasoning_delta" if part_type == "reasoning" else "commentary_delta", delta, part_id)
                elif part_id and delta:
                    pending.setdefault(part_id, []).append(delta)
            elif event_type in {"session.compacted", "session.compaction"}:
                event_session_id = str(properties.get("sessionID", "") or "")
                if not event_session_id or event_session_id == self._session_id:
                    on_event(BackendEvent("compaction", data={"provider": "opencode"}))
            elif event_type == "session.status":
                status = dict(properties.get("status", {}) or {})
                if status.get("type") == "retry":
                    message = str(status.get("message", "") or "The provider did not answer.")
                    on_event(BackendEvent("provider_retry", message))
                    if float(status.get("next", 0) or 0) / 1000 - time.time() > OPENCODE_RETRY_LIMIT_S:
                        provider_error = message
                        self.interrupt()

        while worker.is_alive():
            try:
                consume(event_queue.get(timeout=0.1))
            except queue.Empty:
                pass
            if should_stop() and not self._abort:
                self.interrupt()
        worker.join()
        while True:
            try:
                consume(event_queue.get_nowait())
            except queue.Empty:
                break
        stream_stop.set()
        if stream_response:
            stream_response[0].close()
        event_worker.join(timeout=2)
        if provider_error:
            self._call_tool = None
            raise RuntimeError(f"OpenCode provider: {provider_error}")
        if error_box and not self._abort:
            self._call_tool = None
            raise error_box[0]
        answer = self._answer_text(result_box)
        final_parts = [part for part in result_box.get("parts", []) if isinstance(part, dict) and part.get("type") == "text"]
        if final_parts:
            for index, part in enumerate(final_parts):
                final_text = str(part.get("text", "") or "")
                if final_text:
                    on_event(BackendEvent("commentary_promote", final_text, {"item_id": str(part.get("id", "") or f"opencode_final_{index}")}))
        elif answer:
            on_event(BackendEvent("text_delta", answer))
        emit_usage(result_box)
        self._call_tool = None
        return answer

    def one_shot(self, text: str, *, system_prompt: str, images: Sequence[str], max_output_tokens: int) -> str:
        return self.run_turn(text, system_prompt=system_prompt, tools=[], images=images, on_event=lambda _event: None, call_tool=lambda _name, _args: {}, should_stop=lambda: False)

    def interrupt(self) -> None:
        self._abort = True
        if self._session_id:
            try:
                self._request("POST", f"/session/{self._session_id}/abort", timeout=10)
            except Exception:
                pass

    def close(self) -> None:
        self._call_tool = None
        if self._bridge is not None:
            self._bridge.close()
            self._bridge = None
        process, self._process = self._process, None
        if process is not None and process.poll() is None:
            if os.name == "nt":  # a .cmd launcher owns OpenCode: terminating cmd.exe alone leaves the server running
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                process.terminate()
