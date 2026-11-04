# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Sequence
from urllib.parse import urlparse

from shared.llm_io import log_llm_io

from .base import BackendEvent, EventCallback, StopCallback, ToolCallback
from .usage import codex_usage_data


CODEX_CLI_DOCS_URL = "https://learn.chatgpt.com/docs/codex/cli"
CODEX_PROGRESS_INSTRUCTIONS = """For work with multiple meaningful steps, send concise user-facing progress updates in commentary messages before and between major steps. Keep commentary separate from the final answer, and do not use it for hidden reasoning."""


class CodexSetupRequired(RuntimeError):
    user_action_required = True
    preserve_backend = False


class CodexAuthenticationRequired(CodexSetupRequired):
    preserve_backend = True

    def __init__(self, auth_url: str) -> None:
        self.auth_url = auth_url
        super().__init__(f"Codex needs your ChatGPT sign-in. [Open the secure Codex sign-in window]({auth_url})\n\nAfter sign-in succeeds, return here and send the request again. WanGP does not receive or store your password or tokens.")


def _resolve_codex_executable(configured: str) -> str:
    configured = str(configured or "codex").strip() or "codex"
    if configured.lower() != "codex":
        return shutil.which(configured) or configured
    candidates = []
    if os.name == "nt":
        # Prefer the standalone native CLI on PATH over an older npm shell wrapper.
        candidates.append(shutil.which("codex.exe"))
        localappdata = os.environ.get("LOCALAPPDATA", "")
        if localappdata:
            app_bin = Path(localappdata, "OpenAI", "Codex", "bin")
            app_binaries = list(app_bin.glob("*/codex.exe")) + list(app_bin.glob("codex.exe"))
            candidates.extend(str(path) for path in sorted(app_binaries, key=lambda path: path.stat().st_mtime, reverse=True))
        appdata = os.environ.get("APPDATA", "")
        userprofile = os.environ.get("USERPROFILE", "")
        if appdata:
            candidates.extend((os.path.join(appdata, "npm", "codex.cmd"), os.path.join(appdata, "npm", "codex.exe")))
        if userprofile:
            candidates.extend((os.path.join(userprofile, ".codex", "bin", "codex.exe"), os.path.join(userprofile, ".local", "bin", "codex.exe")))
            extension_binaries = []
            for editor_dir in (".vscode", ".vscode-insiders", ".cursor", ".windsurf"):
                extensions_dir = Path(userprofile, editor_dir, "extensions")
                if extensions_dir.is_dir():
                    extension_binaries.extend(extensions_dir.glob("openai.chatgpt-*\\bin\\windows-*\\codex.exe"))
            candidates.extend(str(path) for path in sorted(extension_binaries, key=lambda path: path.stat().st_mtime, reverse=True))
        candidates.extend(filter(None, (shutil.which("codex.cmd"), shutil.which("codex.exe"), shutil.which("codex"))))
    else:
        candidates.append(shutil.which("codex"))
    for candidate in candidates:
        if candidate and os.path.isfile(candidate) and "\\program files\\windowsapps\\" not in os.path.abspath(candidate).lower():
            return candidate
    return configured


def _codex_launch_command(executable: str) -> list[str]:
    if os.name == "nt" and Path(executable).suffix.lower() in {".cmd", ".bat"}:
        return [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/s", "/c", subprocess.list2cmdline([executable, "app-server"])]
    return [executable, "app-server"]


def _valid_codex_auth_url(value: Any) -> str:
    auth_url = str(value or "").strip()
    parsed = urlparse(auth_url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not (host == "openai.com" or host.endswith(".openai.com") or host == "chatgpt.com" or host.endswith(".chatgpt.com")):
        raise RuntimeError("Codex App Server returned an unexpected sign-in URL.")
    return auth_url


class CodexBackend:
    engine = "codex"

    def __init__(self, profile: dict[str, Any]) -> None:
        self.profile = dict(profile or {})
        self._process: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._responses: dict[int, queue.Queue[dict[str, Any]]] = {}
        self._response_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._next_id = 1
        self._thread_id = ""
        self._turn_id = ""
        self._turn_done = threading.Event()
        self._turn_error = ""
        self._text = ""
        self._agent_message_phases: dict[str, str] = {}
        self._on_event: EventCallback | None = None
        self._call_tool: ToolCallback | None = None
        self._login_id = ""
        self._login_url = ""
        self._usage_total: dict[str, int] = {}
        self._turn_usage_start: dict[str, int] = {}
        self._temp_dir = tempfile.TemporaryDirectory(prefix="wangp-codex-")

    def _start(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._ensure_authenticated()
            return
        self._thread_id = ""
        self._turn_id = ""
        self._usage_total.clear()
        self._turn_usage_start.clear()
        configured = str(self.profile.get("executable", "codex") or "codex").strip()
        executable = _resolve_codex_executable(configured)
        try:
            self._process = subprocess.Popen(
                _codex_launch_command(executable), cwd=self._temp_dir.name,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
            )
        except (FileNotFoundError, PermissionError, OSError) as exc:
            raise CodexSetupRequired(f"Codex CLI is not available to WanGP. WanGP checks native CLIs on PATH, the Codex app's standalone CLI, npm installations and compatible Codex VS Code extension bundles automatically. If none is available, [install the Codex CLI]({CODEX_CLI_DOCS_URL}) and retry. With npm, run `npm install -g @openai/codex`; alternatively set a full executable path in Configuration.\n\nExecutables inside the protected Microsoft Store WindowsApps directory cannot be launched by WanGP; use a standalone CLI exposed by the app or installed separately. Details: {exc}") from exc
        self._reader = threading.Thread(target=self._read_loop, name="wangp-codex-app-server", daemon=True)
        self._reader.start()
        self._request("initialize", {"clientInfo": {"name": "WanGP", "title": "WanGP Deepy", "version": "1"}, "capabilities": {"experimentalApi": True}})
        self._notify("initialized", {})
        self._ensure_authenticated()

    def _ensure_authenticated(self) -> None:
        account_state = self._request("account/read", {"refreshToken": False}, timeout=15)
        if account_state.get("account") is not None or not bool(account_state.get("requiresOpenaiAuth", False)):
            self._login_id = ""
            self._login_url = ""
            return
        if not self._login_url:
            login = self._request("account/login/start", {"type": "chatgpt", "useHostedLoginSuccessPage": True, "appBrand": "codex"}, timeout=15)
            self._login_id = str(login.get("loginId", "") or "")
            self._login_url = _valid_codex_auth_url(login.get("authUrl"))
        raise CodexAuthenticationRequired(self._login_url)

    def _write(self, payload: dict[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None or process.poll() is not None:
            raise RuntimeError("Codex App Server stopped unexpectedly.")
        log_llm_io("OUT", self.engine, "app-server", payload)
        with self._write_lock:
            process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            process.stdin.flush()

    def _request(self, method: str, params: dict[str, Any], timeout: float = 60) -> dict[str, Any]:
        request_id = self._next_id
        self._next_id += 1
        response_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
        with self._response_lock:
            self._responses[request_id] = response_queue
        self._write({"id": request_id, "method": method, "params": params})
        try:
            response = response_queue.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError(f"Codex App Server did not answer {method}.") from exc
        finally:
            with self._response_lock:
                self._responses.pop(request_id, None)
        if response.get("error"):
            error = response["error"]
            message = error.get("message", error) if isinstance(error, dict) else error
            raise RuntimeError(f"Codex {method} failed: {message}")
        return dict(response.get("result", {}) or {})

    def _notify(self, method: str, params: dict[str, Any]) -> None:
        self._write({"method": method, "params": params})

    @staticmethod
    def _dynamic_tools(tools: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for tool in tools:
            function = dict(tool.get("function", {}) or {})
            name = str(function.get("name", "") or "").strip()
            if name:
                result.append({"name": name, "description": str(function.get("description", "") or ""), "inputSchema": dict(function.get("parameters", {}) or {"type": "object", "properties": {}})})
        return result

    def _ensure_thread(self, system_prompt: str, tools: Sequence[dict[str, Any]]) -> None:
        if self._thread_id:
            return
        params: dict[str, Any] = {
            "cwd": str(Path(self._temp_dir.name).resolve()),
            "approvalPolicy": "never",
            "sandbox": "read-only",
            "baseInstructions": "\n\n".join(part for part in (str(system_prompt or "").strip(), CODEX_PROGRESS_INSTRUCTIONS) if part),
            "dynamicTools": self._dynamic_tools(tools),
        }
        if self.profile.get("model"):
            params["model"] = self.profile["model"]
        result = self._request("thread/start", params)
        thread = result.get("thread", result)
        self._thread_id = str(thread.get("id", "") if isinstance(thread, dict) else "")
        if not self._thread_id:
            raise RuntimeError("Codex App Server did not return a thread id.")

    def list_models(self) -> list[dict[str, Any]]:
        self._start()
        catalog = []
        cursor = None
        seen = set()
        while True:
            params: dict[str, Any] = {"limit": 100, "includeHidden": False}
            if cursor:
                params["cursor"] = cursor
            result = self._request("model/list", params)
            for entry in result.get("data", []) or []:
                if not isinstance(entry, dict):
                    continue
                model = str(entry.get("model", entry.get("id", "")) or "").strip()
                if not model or model in seen:
                    continue
                seen.add(model)
                effort_entries = entry.get("supportedReasoningEfforts", []) or []
                efforts = [str(item.get("reasoningEffort", "") or "").strip() for item in effort_entries if isinstance(item, dict) and str(item.get("reasoningEffort", "") or "").strip()]
                catalog.append({"model": model, "display_name": str(entry.get("displayName", model) or model).strip(), "is_default": bool(entry.get("isDefault", False)), "default_reasoning_effort": str(entry.get("defaultReasoningEffort", "") or "").strip(), "reasoning_efforts": efforts})
            cursor = str(result.get("nextCursor", "") or "").strip()
            if not cursor:
                break
        if not catalog:
            raise RuntimeError("Codex App Server returned no picker-visible models.")
        return catalog

    def _read_loop(self) -> None:
        process = self._process
        if process is None or process.stdout is None:
            return
        for line in process.stdout:
            try:
                payload = json.loads(line)
            except Exception:
                continue
            log_llm_io("IN", self.engine, "app-server", payload)
            if "id" in payload and "method" not in payload:
                with self._response_lock:
                    target = self._responses.get(payload.get("id"))
                if target is not None:
                    target.put(payload)
                continue
            method = str(payload.get("method", "") or "")
            params = dict(payload.get("params", {}) or {})
            if method == "item/tool/call" and "id" in payload:
                threading.Thread(target=self._handle_tool_request, args=(payload["id"], params), name="wangp-codex-tool", daemon=True).start()  # this loop keeps reading: Stop gets its answer during a long tool
            elif method == "account/login/completed" and (not self._login_id or str(params.get("loginId", "") or "") == self._login_id):
                self._login_id = ""
                self._login_url = ""
            elif method in {"item/started", "item/completed"}:
                item = params.get("item", {})
                if isinstance(item, dict) and item.get("type") == "agentMessage":
                    item_id = str(item.get("id", "") or "")
                    phase = str(item.get("phase", "") or "")
                    if item_id and phase:
                        self._agent_message_phases[item_id] = phase
                    text = str(item.get("text", "") or "")
                    if method == "item/completed" and item_id and text:
                        if phase == "commentary":
                            if self._on_event:
                                self._on_event(BackendEvent("commentary_replace", text, {"item_id": item_id}))
                        elif phase == "final_answer":
                            self._text = text
                            if self._on_event:
                                self._on_event(BackendEvent("text_replace", text, {"item_id": item_id}))
                elif isinstance(item, dict) and item.get("type") == "contextCompaction" and method == "item/completed" and self._on_event:
                    self._on_event(BackendEvent("compaction", data={"provider": "codex"}))
            elif method in {"item/agentMessage/delta", "item/message/delta"}:
                delta = str(params.get("delta", "") or "")
                item_id = str(params.get("itemId", "") or "")
                phase = self._agent_message_phases.get(item_id, "final_answer")
                if phase == "commentary":
                    if self._on_event and delta:
                        self._on_event(BackendEvent("commentary_delta", delta, {"item_id": item_id}))
                else:
                    self._text += delta
                    if self._on_event and delta:
                        self._on_event(BackendEvent("text_delta", delta, {"item_id": item_id}))
            elif method in {"item/reasoning/summaryTextDelta", "item/reasoning/delta"}:
                delta = str(params.get("delta", "") or "")
                if self._on_event and delta:
                    self._on_event(BackendEvent("reasoning_delta", delta, {
                        "item_id": str(params.get("itemId", "") or ""),
                        "summary_index": params.get("summaryIndex"),
                    }))
            elif method == "thread/tokenUsage/updated":
                usage, self._usage_total = codex_usage_data(params.get("tokenUsage"), self._turn_usage_start)
                if self._on_event:
                    self._on_event(BackendEvent("usage", data=usage))
            elif method == "turn/completed":
                turn = params.get("turn", params)
                status = str(turn.get("status", "completed") if isinstance(turn, dict) else "completed")
                if status not in {"completed", "complete", "cancelled", "interrupted"}:
                    self._turn_error = str(turn.get("error", status) if isinstance(turn, dict) else status)
                self._turn_done.set()
            elif method in {"turn/failed", "error"}:
                self._turn_error = str(params.get("message", params.get("error", "Codex turn failed.")))
                self._turn_done.set()
        if not self._turn_done.is_set():
            self._turn_error = "Codex App Server exited during the turn."
            self._turn_done.set()
        with self._response_lock:
            pending_responses = list(self._responses.values())
        for response_queue in pending_responses:
            try:
                response_queue.put_nowait({"error": {"message": "Codex App Server exited before responding."}})
            except queue.Full:
                pass

    def _handle_tool_request(self, request_id: Any, params: dict[str, Any]) -> None:
        try:
            if self._call_tool is None:
                raise RuntimeError("WanGP tool bridge is not active.")
            result = self._call_tool(str(params.get("tool", "") or ""), dict(params.get("arguments", {}) or {}))
            content = json.dumps(result, ensure_ascii=False)
            response = {"id": request_id, "result": {"contentItems": [{"type": "inputText", "text": content}], "success": str(result.get("status", "")).lower() not in {"error", "failed", "interrupted"}}}
        except Exception as exc:
            response = {"id": request_id, "result": {"contentItems": [{"type": "inputText", "text": json.dumps({"status": "error", "error": str(exc)})}], "success": False}}
        self._write(response)

    def run_turn(self, text: str, *, system_prompt: str, tools: Sequence[dict[str, Any]], images: Sequence[str], on_event: EventCallback, call_tool: ToolCallback, should_stop: StopCallback) -> str:
        self._start()
        self._ensure_thread(system_prompt, tools)
        self._turn_done.clear()
        self._turn_error = ""
        self._text = ""
        self._turn_usage_start = dict(self._usage_total)
        self._agent_message_phases.clear()
        self._on_event = on_event
        self._call_tool = call_tool
        input_items: list[dict[str, Any]] = [{"type": "text", "text": str(text or "")}]
        input_items.extend({"type": "localImage", "path": str(Path(path).resolve())} for path in images)
        params: dict[str, Any] = {"threadId": self._thread_id, "input": input_items, "summary": "concise"}
        if self.profile.get("reasoning_effort"):
            params["effort"] = self.profile["reasoning_effort"]
        result = self._request("turn/start", params)
        turn = result.get("turn", result)
        self._turn_id = str(turn.get("id", "") if isinstance(turn, dict) else "")
        interrupted = False
        while not self._turn_done.wait(0.1):
            if should_stop() and not interrupted:
                interrupted = True
                self.interrupt()
        self._on_event = None
        self._call_tool = None
        if self._turn_error and not interrupted:
            raise RuntimeError(self._turn_error)
        return self._text.strip()

    def one_shot(self, text: str, *, system_prompt: str, images: Sequence[str], max_output_tokens: int) -> str:
        return self.run_turn(text, system_prompt=system_prompt, tools=[], images=images, on_event=lambda _event: None, call_tool=lambda _name, _args: {}, should_stop=lambda: False)

    def interrupt(self) -> None:
        if self._thread_id and self._turn_id:
            try:
                self._request("turn/interrupt", {"threadId": self._thread_id, "turnId": self._turn_id}, timeout=10)
            except Exception:
                pass

    def close(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            if process.stdin is not None:
                process.stdin.close()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    # A .cmd launcher owns Node and Codex children; killing only
                    # cmd.exe leaves those children holding the working directory.
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=subprocess.CREATE_NO_WINDOW)
                else:
                    process.kill()
                process.wait(timeout=3)
            if self._reader is not None:
                self._reader.join(timeout=3)
            if process.stdout is not None:
                process.stdout.close()
        self._temp_dir.cleanup()
