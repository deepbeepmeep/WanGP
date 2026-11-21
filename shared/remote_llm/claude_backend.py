# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import asyncio
import base64
from contextlib import suppress
import json
import mimetypes
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

from shared.llm_io import llm_io_enabled, log_llm_io, media_descriptor

from .base import BackendEvent, EventCallback, StopCallback, ToolCallback
from .usage import aggregate_usage_data, claude_context_window, claude_usage_data


CLAUDE_AUTH_DOCS_URL = "https://code.claude.com/docs/en/authentication"
CLAUDE_SDK_INSTALL_SPEC = "claude-agent-sdk==0.1.66"
CLAUDE_PROGRESS_INSTRUCTIONS = """For work with multiple meaningful steps, send concise user-facing progress updates as short text messages before and between tool calls. Keep progress updates separate from the final answer, and do not reveal hidden reasoning."""


def _restore_mcp_content_contract(server_config: dict[str, Any], sdk_tools: Sequence[Any]) -> None:
    server = server_config.get("instance")
    if server is None:
        return
    from mcp.types import TextContent

    tool_map = {tool_def.name: tool_def for tool_def in sdk_tools}

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]):
        result = await tool_map[name].handler(arguments)
        return [TextContent(type="text", text=item["text"]) for item in result.get("content", []) if item.get("type") == "text"]


class ClaudeSetupRequired(RuntimeError):
    user_action_required = True
    preserve_backend = False


class ClaudeAuthenticationRequired(ClaudeSetupRequired):
    preserve_backend = True

    def __init__(self) -> None:
        super().__init__(f"Claude Code needs sign-in. Run `claude auth login --claudeai` using the configured Claude executable, then retry. [Authentication help]({CLAUDE_AUTH_DOCS_URL})\n\nWanGP does not receive or store your password or tokens.")


def _resolve_claude_executable(configured: str) -> str:
    configured = str(configured or "claude").strip() or "claude"
    if configured.lower() != "claude":
        return shutil.which(configured) or configured
    candidates: list[str] = []
    if os.name == "nt":
        userprofile = os.environ.get("USERPROFILE", "")
        if userprofile:
            extension_binaries: list[Path] = []
            for editor_dir in (".vscode", ".vscode-insiders", ".cursor", ".windsurf"):
                extensions_dir = Path(userprofile, editor_dir, "extensions")
                if extensions_dir.is_dir():
                    extension_binaries.extend(extensions_dir.glob("anthropic.claude-code-*-win32-*\\resources\\native-binary\\claude.exe"))
            candidates.extend(str(path) for path in sorted(extension_binaries, key=lambda path: path.stat().st_mtime, reverse=True))
        candidates.extend(filter(None, (shutil.which("claude.exe"), shutil.which("claude.cmd"), shutil.which("claude"))))
    else:
        candidates.append(shutil.which("claude") or "")
    for candidate in candidates:
        if candidate and os.path.isfile(candidate):
            return candidate
    return ""


class ClaudeBackend:
    engine = "claude"

    def __init__(self, profile: dict[str, Any]) -> None:
        self.profile = dict(profile or {})
        self._session_id = ""
        self._resolved_model = ""
        self._cancelled = False
        self._client = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._temp_dir = tempfile.TemporaryDirectory(prefix="wangp-claude-")

    @staticmethod
    def _run_async(awaitable):
        if os.name != "nt":
            return asyncio.run(awaitable)
        policy_type = getattr(asyncio, "WindowsProactorEventLoopPolicy", None)
        loop = policy_type().new_event_loop() if policy_type is not None else asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(awaitable)
        finally:
            with suppress(Exception):
                loop.run_until_complete(loop.shutdown_asyncgens())
            asyncio.set_event_loop(None)
            loop.close()

    def _sdk(self):
        try:
            import claude_agent_sdk as sdk
        except ImportError as exc:
            detected = _resolve_claude_executable(str(self.profile.get("executable", "claude") or "claude"))
            detected_note = " WanGP found your Claude Code executable and will reuse it once the SDK bridge is installed." if detected else ""
            raise ClaudeSetupRequired(f"Claude Agent SDK is not installed.{detected_note} Run `pip install {CLAUDE_SDK_INSTALL_SPEC}` in WanGP's Python environment, then authenticate once with Claude Code. Do not install the unpinned latest SDK because it replaces WanGP's MCP/Pydantic stack. [Authentication help]({CLAUDE_AUTH_DOCS_URL})") from exc
        if str(getattr(sdk, "__version__", "") or "") != CLAUDE_SDK_INSTALL_SPEC.rsplit("==", 1)[-1]:
            raise ClaudeSetupRequired(f"WanGP requires {CLAUDE_SDK_INSTALL_SPEC} for compatible thought summaries and MCP tools. Run `pip install --upgrade {CLAUDE_SDK_INSTALL_SPEC}` in WanGP's Python environment.")
        return sdk

    def _options(self, sdk, system_prompt: str, mcp_servers: dict[str, Any], allowed_tools: list[str]) -> Any:
        option_fields = getattr(sdk.ClaudeAgentOptions, "__dataclass_fields__", {})
        options_kwargs: dict[str, Any] = {
            "system_prompt": "\n\n".join(part for part in (str(system_prompt or "").strip(), CLAUDE_PROGRESS_INSTRUCTIONS) if part),
            "mcp_servers": mcp_servers,
            "tools": [],
            "allowed_tools": allowed_tools,
            "disallowed_tools": ["Bash", "Edit", "Write", "Read", "Glob", "Grep", "WebFetch", "WebSearch", "NotebookEdit", "Task"],
            "include_partial_messages": True,
            "max_turns": 100,
            "cwd": self._temp_dir.name,
            "setting_sources": [],
            "thinking": {"type": "adaptive", "display": "summarized"},
        }
        if "strict_mcp_config" in option_fields:
            options_kwargs["strict_mcp_config"] = True
        if self.profile.get("model"):
            options_kwargs["model"] = self.profile["model"]
        if self.profile.get("reasoning_effort"):
            options_kwargs["effort"] = self.profile["reasoning_effort"]
        executable = _resolve_claude_executable(str(self.profile.get("executable", "claude") or "claude"))
        if executable:
            options_kwargs["cli_path"] = executable
        if self._session_id:
            options_kwargs["resume"] = self._session_id
        if option_fields and "thinking" not in option_fields:
            raise ClaudeSetupRequired(f"Claude thought summaries require a compatible Claude Agent SDK. Run `pip install {CLAUDE_SDK_INSTALL_SPEC}` in WanGP's Python environment.")
        if options_kwargs.get("effort") and option_fields and "effort" not in option_fields:
            raise ClaudeSetupRequired(f"Claude reasoning effort requires a compatible Claude Agent SDK. Run `pip install {CLAUDE_SDK_INSTALL_SPEC}` in WanGP's Python environment.")
        return sdk.ClaudeAgentOptions(**options_kwargs)

    async def _run(self, text: str, system_prompt: str, tools: Sequence[dict[str, Any]], images: Sequence[str], on_event: EventCallback, call_tool: ToolCallback, should_stop: StopCallback) -> str:
        sdk = self._sdk()
        sdk_tools = []
        allowed_tools = []
        for definition in tools:
            function = dict(definition.get("function", {}) or {})
            name = str(function.get("name", "") or "").strip()
            if not name:
                continue

            async def handler(arguments, tool_name=name):
                log_llm_io("IN", self.engine, "tool-call", {"tool": tool_name, "arguments": dict(arguments or {})})
                result = await asyncio.to_thread(call_tool, tool_name, dict(arguments or {}))
                log_llm_io("OUT", self.engine, "tool-result", {"tool": tool_name, "result": result})
                return {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]}

            sdk_tools.append(sdk.tool(name, str(function.get("description", "") or ""), dict(function.get("parameters", {}) or {}))(handler))
            allowed_tools.append(f"mcp__wangp__{name}")
        mcp_servers = {"wangp": sdk.create_sdk_mcp_server(name="wangp", version="1", tools=sdk_tools)} if sdk_tools else {}
        if mcp_servers:
            _restore_mcp_content_contract(mcp_servers["wangp"], sdk_tools)
        prompt: Any = str(text or "")
        if images:
            async def multimodal_prompt():
                content: list[dict[str, Any]] = [{"type": "text", "text": str(text or "")}]
                for path in images:
                    media_type = mimetypes.guess_type(str(path))[0] or "image/png"
                    with open(path, "rb") as reader:
                        encoded = base64.b64encode(reader.read()).decode("ascii")
                    content.append({"type": "image", "source": {"type": "base64", "media_type": media_type, "data": encoded}})
                yield {"type": "user", "message": {"role": "user", "content": content}}

            prompt = multimodal_prompt()
        answer_parts: list[str] = []
        current_message_id = ""
        current_block_types: dict[int, str] = {}
        streamed_text_ids: set[str] = set()
        streamed_text_parts: dict[str, list[str]] = {}
        streamed_reasoning_ids: set[str] = set()
        tool_requests: dict[str, dict[str, Any]] = {}
        pending_text_messages: list[tuple[str, str]] = []
        streamed_usage_by_message: dict[str, dict[str, Any]] = {}
        last_context_tokens = 0
        message_no = 0

        def matching_pending(block_text: str) -> tuple[str, str] | None:
            normalized = " ".join(str(block_text or "").split()).casefold()
            return next(((item_id, text) for item_id, text in pending_text_messages if " ".join(text.split()).casefold() == normalized), None)

        def promote_pending_text() -> None:
            for item_id, block_text in pending_text_messages:
                answer_parts.append(block_text)
                on_event(BackendEvent("commentary_promote", block_text, {"item_id": item_id}))
            pending_text_messages.clear()

        self._loop = asyncio.get_running_loop()
        options = self._options(sdk, system_prompt, mcp_servers, allowed_tools)
        if llm_io_enabled():
            log_llm_io("OUT", self.engine, "query", {
                "system_prompt": "\n\n".join(part for part in (str(system_prompt or "").strip(), CLAUDE_PROGRESS_INSTRUCTIONS) if part),
                "user_text": str(text or ""),
                "images": [media_descriptor(path) for path in images],
                "tools": list(tools),
                "model": str(self.profile.get("model", "") or "Automatic (Claude account default)"),
                "reasoning_effort": str(self.profile.get("reasoning_effort", "") or "Automatic"),
                "resumed_session": bool(self._session_id),
            })
        async with sdk.ClaudeSDKClient(options=options) as client:
            self._client = client
            await client.query(prompt)

            async def watch_stop():
                while not self._cancelled and not should_stop():
                    await asyncio.sleep(0.1)
                if self._cancelled or should_stop():
                    await client.interrupt()

            watcher = asyncio.create_task(watch_stop())
            try:
                async for message in client.receive_response():
                    log_llm_io("IN", self.engine, "sdk-message", message)
                    session_id = str(getattr(message, "session_id", "") or "")
                    if session_id:
                        self._session_id = session_id
                    event = getattr(message, "event", None)
                    if isinstance(event, dict):
                        event_type = str(event.get("type", "") or "")
                        if event_type == "message_start":
                            message_no += 1
                            raw_message = dict(event.get("message", {}) or {})
                            current_message_id = str(raw_message.get("id", "") or "").strip() or f"message-{message_no}"
                            current_block_types.clear()
                        elif event_type == "message_stop":
                            current_message_id = ""
                            current_block_types.clear()
                        elif event_type == "message_delta":
                            usage = claude_usage_data(event.get("usage"))
                            if usage:
                                usage_id = current_message_id or f"message-{message_no}"
                                streamed_usage_by_message[usage_id] = usage
                                last_context_tokens = usage["total_tokens"]
                                aggregate = aggregate_usage_data(streamed_usage_by_message)
                                context_window = claude_context_window(self._resolved_model or self.profile.get("model", ""))
                                if context_window:
                                    aggregate.update({"context_tokens": last_context_tokens, "context_window": context_window})
                                on_event(BackendEvent("usage", data=aggregate))
                        elif event_type == "content_block_start":
                            index = int(event.get("index", 0) or 0)
                            block = dict(event.get("content_block", {}) or {})
                            block_type = str(block.get("type", "") or "")
                            current_block_types[index] = block_type
                            if block_type == "thinking":
                                on_event(BackendEvent("reasoning_start"))
                            elif block_type == "tool_use":
                                tool_use_id = str(block.get("id", "") or "")
                                tool_requests[tool_use_id] = {"name": str(block.get("name", "") or ""), "input": dict(block.get("input", {}) or {})}
                                on_event(BackendEvent("tool_request_start", data={"id": tool_use_id, "name": tool_requests[tool_use_id]["name"]}))
                            initial_text = str(block.get("thinking", "") if block_type == "thinking" else block.get("text", "") if block_type == "text" else "")
                            if initial_text:
                                item_id = f"{current_message_id}:{block_type}:{index}"
                                if block_type == "thinking":
                                    streamed_reasoning_ids.add(item_id)
                                    on_event(BackendEvent("reasoning_delta", initial_text, {"item_id": item_id, "summary_index": index}))
                                elif block_type == "text":
                                    streamed_text_ids.add(item_id)
                                    streamed_text_parts.setdefault(item_id, []).append(initial_text)
                                    on_event(BackendEvent("commentary_delta", initial_text, {"item_id": item_id}))
                        elif event_type == "content_block_delta":
                            index = int(event.get("index", 0) or 0)
                            delta = dict(event.get("delta", {}) or {})
                            delta_type = str(delta.get("type", "") or "")
                            block_type = current_block_types.get(index, "thinking" if "thinking" in delta_type else "text" if delta_type == "text_delta" else "")
                            delta_text = str(delta.get("thinking", "") if block_type == "thinking" else delta.get("text", "") if block_type == "text" else "")
                            if delta_text:
                                item_id = f"{current_message_id}:{block_type}:{index}"
                                if block_type == "thinking":
                                    streamed_reasoning_ids.add(item_id)
                                    on_event(BackendEvent("reasoning_delta", delta_text, {"item_id": item_id, "summary_index": index}))
                                elif block_type == "text":
                                    streamed_text_ids.add(item_id)
                                    streamed_text_parts.setdefault(item_id, []).append(delta_text)
                                    on_event(BackendEvent("commentary_delta", delta_text, {"item_id": item_id}))
                        continue
                    if "systemmessage" in type(message).__name__.lower():
                        subtype = str(getattr(message, "subtype", "") or "").lower()
                        system_data = getattr(message, "data", {})
                        system_data = dict(system_data) if isinstance(system_data, dict) else {}
                        if subtype == "init":
                            self._resolved_model = str(system_data.get("model", "") or "").strip()
                        elif subtype == "compact_boundary":
                            on_event(BackendEvent("compaction", data={"provider": "claude", **system_data}))
                        continue
                    if "resultmessage" in type(message).__name__.lower():
                        promote_pending_text()
                        usage = claude_usage_data(getattr(message, "usage", None))
                        if usage:
                            context_window = claude_context_window(self._resolved_model or self.profile.get("model", ""))
                            if context_window:
                                usage.update({"context_tokens": last_context_tokens or usage["total_tokens"], "context_window": context_window})
                            on_event(BackendEvent("usage", data=usage))
                        continue
                    blocks = list(getattr(message, "content", []) or [])
                    if not blocks:
                        continue
                    message_id = current_message_id or str(getattr(message, "message_id", "") or getattr(message, "uuid", "") or f"message-{message_no + 1}")
                    has_tool = any("tooluse" in type(block).__name__.lower() for block in blocks) or str(getattr(message, "stop_reason", "") or "") == "tool_use"
                    for index, block in enumerate(blocks):
                        block_type = type(block).__name__.lower()
                        if "thinking" in block_type:
                            block_text = str(getattr(block, "thinking", "") or "")
                            item_id = f"{message_id}:thinking:{index}"
                            if block_text and item_id not in streamed_reasoning_ids:
                                on_event(BackendEvent("reasoning_delta", block_text, {"item_id": item_id, "summary_index": index}))
                        elif "text" in block_type:
                            block_text = str(getattr(block, "text", "") or "")
                            if not block_text:
                                continue
                            item_id = f"{message_id}:text:{index}"
                            normalized_block = " ".join(block_text.split()).casefold()
                            current_stream_ids = [stream_id for stream_id in streamed_text_parts if stream_id.startswith(f"{message_id}:text:")]
                            stream_matches = [stream_id for stream_id in current_stream_ids if " ".join("".join(streamed_text_parts[stream_id]).split()).casefold() == normalized_block]
                            if stream_matches or current_stream_ids:
                                item_id = (stream_matches or current_stream_ids)[-1]
                            duplicate = matching_pending(block_text)
                            if duplicate is not None:
                                if item_id != duplicate[0]:
                                    on_event(BackendEvent("commentary_remove", data={"item_id": item_id}))
                                continue
                            if has_tool:
                                on_event(BackendEvent("commentary_replace", block_text, {"item_id": item_id}))
                            else:
                                if item_id not in streamed_text_ids:
                                    on_event(BackendEvent("commentary_delta", block_text, {"item_id": item_id}))
                                pending_text_messages.append((item_id, block_text))
                        elif "tooluse" in block_type:
                            tool_use_id = str(getattr(block, "id", "") or "")
                            tool_requests[tool_use_id] = {"name": str(getattr(block, "name", "") or ""), "input": dict(getattr(block, "input", {}) or {})}
                        elif "toolresult" in block_type:
                            tool_use_id = str(getattr(block, "tool_use_id", "") or "")
                            request = tool_requests.pop(tool_use_id, {})
                            if bool(getattr(block, "is_error", False)):
                                content = getattr(block, "content", "")
                                error = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
                                on_event(BackendEvent("tool_request_error", error, {"id": tool_use_id, "name": str(request.get("name", "") or ""), "input": dict(request.get("input", {}) or {})}))
                    if has_tool:
                        # A completed text-only AssistantMessage immediately before
                        # tool use is a progress report, not part of the final answer.
                        pending_text_messages.clear()
                promote_pending_text()
            finally:
                watcher.cancel()
                with suppress(asyncio.CancelledError):
                    await watcher
                self._client = None
                self._loop = None
        answer = "".join(answer_parts).strip()
        normalized_answer = answer.casefold()
        if any(marker in normalized_answer for marker in ("failed to authenticate", "oauth session expired", "not logged in", "please run /login")):
            raise ClaudeAuthenticationRequired()
        return answer

    @staticmethod
    def _model_catalog(server_info: dict[str, Any]) -> list[dict[str, Any]]:
        catalog = []
        seen = set()
        for entry in list(server_info.get("models", []) or []):
            item = entry if isinstance(entry, dict) else {"value": entry}
            model = str(item.get("value", item.get("model", item.get("id", ""))) or "").strip()
            if not model or model == "default" or model in seen or item.get("isAvailable") is False:
                continue
            seen.add(model)
            effort_source = item.get("supportedEffortLevels", [])
            effort_source = effort_source if isinstance(effort_source, list) else []
            efforts = []
            for effort_entry in effort_source:
                effort = str(effort_entry.get("value", effort_entry.get("effort", "")) if isinstance(effort_entry, dict) else effort_entry or "").strip()
                if effort and effort not in efforts:
                    efforts.append(effort)
            if item.get("supportsEffort") and not efforts:
                efforts = ["low", "medium", "high", "xhigh", "max"]
            catalog.append({"model": model, "display_name": str(item.get("displayName", model) or model).strip(), "is_default": bool(item.get("isDefault", False)), "default_reasoning_effort": str(item.get("defaultEffort", "") or "").strip(), "reasoning_efforts": efforts})
        if not catalog:
            raise RuntimeError("Claude Code returned no picker-visible models.")
        return catalog

    async def _list_models(self) -> list[dict[str, Any]]:
        sdk = self._sdk()
        options = self._options(sdk, "", {}, [])
        async with sdk.ClaudeSDKClient(options=options) as client:
            server_info = await client.get_server_info()
        return self._model_catalog(dict(server_info or {}))

    def list_models(self) -> list[dict[str, Any]]:
        return self._run_async(self._list_models())

    def run_turn(self, text: str, *, system_prompt: str, tools: Sequence[dict[str, Any]], images: Sequence[str], on_event: EventCallback, call_tool: ToolCallback, should_stop: StopCallback) -> str:
        self._cancelled = False
        return self._run_async(self._run(text, system_prompt, tools, images, on_event, call_tool, should_stop))

    def one_shot(self, text: str, *, system_prompt: str, images: Sequence[str], max_output_tokens: int, should_stop: StopCallback | None = None) -> str:
        return self.run_turn(text, system_prompt=system_prompt, tools=[], images=images, on_event=lambda _event: None, call_tool=lambda _name, _args: {}, should_stop=should_stop or (lambda: False))

    def interrupt(self) -> None:
        self._cancelled = True
        if self._client is not None and self._loop is not None and self._loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(self._client.interrupt(), self._loop)
            except Exception:
                pass

    def close(self) -> None:
        self.interrupt()
        self._temp_dir.cleanup()
