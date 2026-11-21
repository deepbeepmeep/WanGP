# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import os
import base64
import getpass
import json
import mimetypes
import queue
import secrets
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
OPENCODE_START_TIMEOUT_S = 15
# OpenCode 2: its own tools stay off, the WanGP server's are allowed (the last matching rule wins)
OPENCODE2_PERMISSIONS = [{"action": "*", "resource": "*", "effect": "deny"}, {"action": f"{OPENCODE_MCP_SERVER}_*", "resource": "*", "effect": "allow"}]
OPENCODE2_TURN_END = {"session.execution.succeeded", "session.execution.failed", "session.execution.interrupted"}


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


def _wangp_server_password(create: bool = False) -> str:
    """The password of the OpenCode 2 servers WanGP starts (OpenCode 2 always asks for one), kept for the user so that a server outliving
    WanGP still answers it and every OpenCode backend of WanGP reaches it. OpenCode 1 does not read OPENCODE_PASSWORD."""
    path = Path(tempfile.gettempdir()) / f"wangp-opencode-{getpass.getuser()}.key"
    try:
        return path.read_text(encoding="ascii").strip()
    except OSError:
        if not create:
            return ""
    password = secrets.token_urlsafe(24)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w", encoding="ascii") as writer:
        writer.write(password)
    return password


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
    """Deepy through an OpenCode server: OpenCode 1 (/global/health, /session/{id}/message answering the whole turn) or OpenCode 2 (/api/...,
    the prompt admitted at once and the turn followed through session events)."""
    engine = "opencode"

    def __init__(self, profile: dict[str, Any], toolbox: Any | None = None) -> None:
        self.profile = dict(profile or {})
        self.base_url = str(self.profile.get("base_url", "http://127.0.0.1:4096") or "").rstrip("/")
        self.toolbox = toolbox
        self._process: subprocess.Popen[str] | None = None
        self._api = 1  # the server's API, read by _ensure_server
        self._session_id = ""
        self._instructions = ""  # the system text last given to an OpenCode 2 session
        self._abort = False
        self._bridge: MCPHttpBridge | None = None
        self._call_tool: ToolCallback | None = None
        self._model_catalog = list(self.profile.get("model_catalog", []) or [])
        self._temp_dir = tempfile.TemporaryDirectory(prefix="wangp-opencode-", ignore_cleanup_errors=True)

    def _auth(self, api: int | None = None):
        if (api or self._api) == 2:  # its user is always opencode
            password = os.environ.get("OPENCODE_PASSWORD", "") or os.environ.get("OPENCODE_SERVER_PASSWORD", "") or _wangp_server_password()
            return ("opencode", password) if password else None
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

    def _server_api(self, timeout: float) -> int | None:
        """The API of the OpenCode server at base_url: 2 (it answers /api/info), 1 (/global/health), None when no server answers."""
        for api, path in ((2, "/api/info"), (1, "/global/health")):
            try:
                response = requests.get(self.base_url + path, timeout=timeout, auth=self._auth(api))
            except requests.RequestException:
                return None
            if response.status_code == 401:
                raise RuntimeError(f"The OpenCode server at {self.base_url} asks for a password. Start WanGP with the server's password in OPENCODE_PASSWORD (OpenCode 2) or OPENCODE_SERVER_PASSWORD (OpenCode 1), or stop that server: WanGP then starts its own.")
            try:
                payload = response.json() if response.ok else None
            except ValueError:  # OpenCode 2 answers /global/health with its web page
                payload = None
            if isinstance(payload, dict) and payload.get("version" if api == 2 else "healthy"):
                return api
        return None

    def _ensure_server(self) -> None:
        if api := self._server_api(2):
            self._api = api
            return
        parsed = urlparse(self.base_url)
        if (parsed.hostname or "").lower() not in {"127.0.0.1", "localhost", "::1"}:
            raise RuntimeError(f"OpenCode server is not reachable at {self.base_url}.")
        configured = str(self.profile.get("executable", "opencode") or "opencode").strip()
        executable = _resolve_opencode_executable(configured)
        child_env = os.environ.copy()
        if config := str(self.profile.get("config", "") or "").strip():
            child_env["OPENCODE_CONFIG_CONTENT"] = config
        if not (child_env.get("OPENCODE_PASSWORD") or child_env.get("OPENCODE_SERVER_PASSWORD")):
            child_env["OPENCODE_PASSWORD"] = _wangp_server_password(create=True)  # else OpenCode 2 picks a random one that WanGP cannot know
        try:
            self._process = subprocess.Popen(_opencode_launch_command(executable, parsed.port or 4096), cwd=self._temp_dir.name, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=child_env, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0)
        except (FileNotFoundError, PermissionError, OSError) as exc:
            raise RuntimeError(f"OpenCode is not reachable and could not be started with '{configured} serve'. Install/configure OpenCode or correct its executable and server URL. {exc}") from exc
        deadline = time.time() + OPENCODE_START_TIMEOUT_S
        while time.time() < deadline:
            time.sleep(0.1)
            if api := self._server_api(1):
                self._api = api
                return
        raise RuntimeError("OpenCode server did not become ready.")

    def list_models(self) -> list[dict[str, Any]]:
        self._ensure_server()
        if self._api == 2:
            return self._list_models_v2()
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

    def _list_models_v2(self) -> list[dict[str, Any]]:
        names = {str(provider.get("id", "")): str(provider.get("name", "") or provider.get("id", "")) for provider in self._request("GET", "/api/provider", timeout=30).get("data", [])}
        default = self._request("GET", "/api/model/default", timeout=30).get("data") or {}
        catalog = []
        for model in self._request("GET", "/api/model", timeout=30).get("data", []):
            if model.get("type", "chat") != "chat" or not model.get("enabled", True):
                continue
            provider_id, model_id = str(model.get("providerID", "") or ""), str(model.get("id", "") or "")
            catalog.append({"provider": provider_id, "provider_name": names.get(provider_id, provider_id), "model": model_id, "display_name": str(model.get("name", "") or model_id),
                            "is_default": (provider_id, model_id) == (default.get("providerID"), default.get("id")), "context_window": max(0, int((model.get("limit") or {}).get("context", 0) or 0)),
                            "reasoning_efforts": [str(variant.get("id", "")) for variant in model.get("variants") or [] if variant.get("id")]})
        if not catalog:
            raise RuntimeError("OpenCode did not report any available models. Configure or authenticate a provider in OpenCode first.")
        self._model_catalog = catalog
        return catalog

    def _start_bridge(self, tools: Sequence[dict[str, Any]]) -> str:
        exposed, originals = _opencode_tools(tools)
        proxy = build_tool_proxy(exposed, lambda name, arguments: self._call_tool(originals[name], arguments) if self._call_tool is not None else {"status": "error", "tool": originals[name], "error": "WanGP tool bridge is not active."})
        self._bridge = MCPHttpBridge(proxy)
        return self._bridge.start()

    def _ensure_session(self, system_prompt: str, tools: Sequence[dict[str, Any]]) -> None:
        self._ensure_server()
        if self._session_id:
            return
        if self._api == 2:
            return self._ensure_session_v2(tools)
        result = self._request("POST", "/session", json={"title": "WanGP Deepy"})
        self._session_id = str(result.get("id", "") or "")
        if not self._session_id:
            raise RuntimeError("OpenCode did not return a session id.")
        if self.toolbox is not None:
            url = self._start_bridge(tools)
            try:
                self._request("POST", "/mcp", json={"name": OPENCODE_MCP_SERVER, "config": {"type": "remote", "url": url, "oauth": False, "codemode": False, "timeout": OPENCODE_TOOL_TIMEOUT_MS}}, timeout=15)
            except Exception:
                self._bridge.close()
                self._bridge = None
                raise

    def _ensure_session_v2(self, tools: Sequence[dict[str, Any]]) -> None:
        body: dict[str, Any] = {"permissions": OPENCODE2_PERMISSIONS if self.toolbox is not None and tools else OPENCODE2_PERMISSIONS[:1]}  # a session without tools is denied WanGP's too: the server shares them between sessions
        if model := self._model():
            body["model"] = {"id": model["modelID"], "providerID": model["providerID"], **({"variant": effort} if (effort := str(self.profile.get("reasoning_effort", "") or "").strip()) else {})}
        self._session_id = str((self._request("POST", "/api/session", json=body).get("data") or {}).get("id", "") or "")
        if not self._session_id:
            raise RuntimeError("OpenCode did not return a session id.")
        self._instructions = ""
        self._request("PATCH", f"/api/session/{self._session_id}", json={"title": "WanGP Deepy"})  # named: OpenCode does not ask the model for a title
        if self.toolbox is not None:
            url = self._start_bridge(tools)
            try:
                self._request("PUT", f"/api/experimental/mcp/{OPENCODE_MCP_SERVER}", json={"config": {"type": "remote", "url": url, "oauth": False, "codemode": False, "timeout": {"execution": OPENCODE_TOOL_TIMEOUT_MS}}}, timeout=15)
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

    @staticmethod
    def _system_text(system_prompt: str, tools: Sequence[dict[str, Any]]) -> str:
        return "\n\n".join(part for part in (str(system_prompt or "").strip(), OPENCODE_TOOL_NAMES_INSTRUCTIONS if tools else "", OPENCODE_PROGRESS_INSTRUCTIONS) if part)

    @staticmethod
    def _image_urls(images: Sequence[str]) -> list[tuple[str, str]]:
        urls = []
        for path in images:
            mime = mimetypes.guess_type(str(path))[0] or "image/png"
            with open(path, "rb") as reader:
                urls.append((f"data:{mime};base64,{base64.b64encode(reader.read()).decode('ascii')}", mime))
        return urls

    def run_turn(self, text: str, *, system_prompt: str, tools: Sequence[dict[str, Any]], images: Sequence[str], on_event: EventCallback, call_tool: ToolCallback, should_stop: StopCallback) -> str:
        self._call_tool = call_tool
        self._ensure_session(system_prompt, tools)
        if not self._model_catalog:
            self.list_models()
        self._abort = False
        if llm_io_enabled():
            log_llm_io("OUT", self.engine, "turn-context", {"system_prompt": system_prompt, "user_text": str(text or ""), "images": [media_descriptor(path) for path in images], "tools": list(tools), "model": self._model() or "Automatic"})
        if self._api == 2:
            return self._run_turn_v2(text, self._system_text(system_prompt, tools), images, on_event, should_stop)
        body: dict[str, Any] = {
            "parts": [{"type": "text", "text": str(text or "")}],
            "system": self._system_text(system_prompt, tools),
            "tools": {"*": False, f"{OPENCODE_MCP_SERVER}_*": bool(tools)},  # OpenCode's own tools stay off, including the ones later versions add; WanGP's too in a turn without tools (the server shares them between sessions)
        }
        if model := self._model():
            body["model"] = model
        if effort := str(self.profile.get("reasoning_effort", "") or "").strip():
            body["variant"] = effort
        body["parts"].extend({"type": "file", "url": url, "mime": mime} for url, mime in self._image_urls(images))
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

    def _run_turn_v2(self, text: str, system: str, images: Sequence[str], on_event: EventCallback, should_stop: StopCallback) -> str:
        """OpenCode 2 admits the prompt at once: the turn's text, steps and end arrive as session events on /api/event."""
        session = self._session_id
        if system != self._instructions:  # the system text of the session (OpenCode puts it in its system prompt from the next step)
            self._request("PUT", f"/api/experimental/session/{session}/instructions/entries/{OPENCODE_MCP_SERVER}", json={"value": system}, timeout=30)
            self._instructions = system
        body: dict[str, Any] = {"text": str(text or "")}
        if images:
            body["files"] = [{"uri": url, "name": Path(path).name} for path, (url, _mime) in zip(images, self._image_urls(images))]
        event_queue: queue.Queue[dict[str, Any]] = queue.Queue()
        stream_ready, stream_stop, stream_closed = threading.Event(), threading.Event(), threading.Event()

        def read_events():
            # the stream is closed here, at its next line (a heartbeat every 15 s at most): closed from another thread, the close waits for it
            try:
                with requests.get(self.base_url + "/api/event", headers={"Accept": "text/event-stream"}, stream=True, timeout=(10, None), auth=self._auth()) as response:
                    response.raise_for_status()
                    response.encoding = "utf-8"
                    # events as they arrive: read by blocks, the small ones (server.connected, the end of the turn) would wait for later traffic
                    for raw_line in response.iter_lines(chunk_size=None, decode_unicode=True):
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
                        stream_ready.set()  # the stream opens with server.connected
                        event_queue.put(event)
            except Exception:
                pass
            finally:
                stream_ready.set()
                stream_closed.set()

        event_worker = threading.Thread(target=read_events, name="wangp-opencode-events", daemon=True)
        event_worker.start()
        stream_ready.wait(timeout=10)
        if stream_closed.is_set() and event_queue.empty():
            raise RuntimeError(f"OpenCode's event stream could not be opened at {self.base_url}/api/event.")
        texts: dict[str, str] = {}  # "<assistant message>:<ordinal>" -> its text so far, in arrival order
        step_models: dict[str, dict[str, Any]] = {}
        last_usage: dict[str, Any] = {}
        final_message = last_step = provider_error = failure = ended = ""
        try:
            self._request("POST", f"/api/session/{session}/prompt", json=body, timeout=60)
            while not ended:
                if should_stop() and not self._abort:
                    self.interrupt()
                try:
                    event = event_queue.get(timeout=0.1)
                except queue.Empty:
                    if stream_closed.is_set():
                        raise RuntimeError("OpenCode's event stream closed before the end of the turn.")
                    continue
                kind, data = str(event.get("type", "") or ""), dict(event.get("data", {}) or {})
                if data.get("sessionID") != session:
                    continue
                message = str(data.get("assistantMessageID", "") or "")
                key = f"{message}:{data.get('ordinal', 0)}"
                if kind == "session.text.delta" and data.get("delta"):
                    texts[key] = texts.get(key, "") + str(data["delta"])
                    on_event(BackendEvent("commentary_delta", str(data["delta"]), {"item_id": key}))
                elif kind == "session.text.ended":
                    full, previous = str(data.get("text", "") or ""), texts.get(key, "")
                    texts[key] = full
                    if full.startswith(previous) and full != previous:
                        on_event(BackendEvent("commentary_delta", full[len(previous):], {"item_id": key}))
                    elif full and not full.startswith(previous):
                        on_event(BackendEvent("commentary_replace", full, {"item_id": key}))
                elif kind == "session.reasoning.delta" and data.get("delta"):
                    on_event(BackendEvent("reasoning_delta", str(data["delta"]), {"item_id": f"{message}:reasoning:{data.get('ordinal', 0)}", "summary_index": 0}))
                elif kind == "session.step.started":
                    step_models[message] = dict(data.get("model", {}) or {})
                    last_step = message
                elif kind == "session.step.ended":
                    model = step_models.get(message, {})
                    usage = self._usage({"info": {"tokens": data.get("tokens", {}), "providerID": model.get("providerID", ""), "modelID": model.get("id", "")}})
                    if usage and usage != last_usage:
                        last_usage = usage
                        on_event(BackendEvent("usage", data=usage))
                    if data.get("finish") != "tool-calls":
                        final_message = message
                elif kind == "session.compaction.ended":
                    on_event(BackendEvent("compaction", data={"provider": "opencode"}))
                elif kind == "session.retry.scheduled":
                    reason = str((data.get("error", {}) or {}).get("message", "") or "The provider did not answer.")
                    on_event(BackendEvent("provider_retry", reason))
                    if float(data.get("at", 0) or 0) / 1000 - time.time() > OPENCODE_RETRY_LIMIT_S:
                        provider_error = reason
                        self.interrupt()
                elif kind in OPENCODE2_TURN_END:
                    ended = kind
                    failure = str((data.get("error", {}) or {}).get("message", "") or data.get("reason", "") or "")
        finally:
            stream_stop.set()
            self._call_tool = None
        if provider_error:
            raise RuntimeError(f"OpenCode provider: {provider_error}")
        if ended == "session.execution.failed":
            raise RuntimeError(f"OpenCode: {failure or 'the turn failed'}")
        if ended == "session.execution.interrupted":
            if not self._abort:
                raise RuntimeError(f"OpenCode stopped the turn ({failure}).")
            final_message = final_message or last_step  # stopped: what the step had written, as OpenCode 1 answers
        final_parts = [(key, part) for key, part in texts.items() if final_message and key.startswith(final_message + ":") and part]
        for key, part in final_parts:
            on_event(BackendEvent("commentary_promote", part, {"item_id": key}))
        return "".join(part for _key, part in final_parts).strip()

    def one_shot(self, text: str, *, system_prompt: str, images: Sequence[str], max_output_tokens: int, should_stop: StopCallback | None = None) -> str:
        return self.run_turn(text, system_prompt=system_prompt, tools=[], images=images, on_event=lambda _event: None, call_tool=lambda _name, _args: {}, should_stop=should_stop or (lambda: False))

    def interrupt(self) -> None:
        self._abort = True
        if self._session_id:
            try:
                self._request("POST", f"/api/session/{self._session_id}/interrupt" if self._api == 2 else f"/session/{self._session_id}/abort", timeout=10)
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
