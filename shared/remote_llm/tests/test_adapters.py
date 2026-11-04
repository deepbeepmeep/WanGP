# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import asyncio
import io
import json
import tempfile
import threading
import time
import unittest
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from shared.cli_args import parse_wgp_args
from shared.llm_io import configure_llm_io, get_llm_io_path, log_llm_io
from shared.deepy.engine import AssistantSessionState, begin_assistant_turn, clear_assistant_session
from shared.deepy.filesystem import FileAccessPolicy
from shared.gradio import assistant_chat
from shared.remote_llm.codex_backend import CodexAuthenticationRequired, CodexBackend, _codex_launch_command, _resolve_codex_executable
from shared.remote_llm.claude_backend import CLAUDE_PROGRESS_INSTRUCTIONS, ClaudeAuthenticationRequired, ClaudeBackend, _resolve_claude_executable, _restore_mcp_content_contract
from shared.remote_llm.images import temporary_image_paths
from shared.mcp.bridge import build_tool_proxy
from shared.remote_llm.opencode_backend import OpenCodeBackend
from shared.remote_llm.base import BackendEvent
from shared.remote_llm.deepy_runner import REMOTE_HISTORY_MAX_CHARS, _restored_history, _visual_query, run_remote_deepy_turn
from shared.remote_llm.usage import build_remote_usage_stats, claude_context_window
from shared.prompt_enhancer.qwen35_assistant_runtime import extract_tool_calls


class _ClosableBackend:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _FakeBackend(_ClosableBackend):
    def __init__(self):
        super().__init__()
        self.system_prompt = ""

    def run_turn(self, _text, *, on_event, call_tool, **_kwargs):
        self.system_prompt = str(_kwargs.get("system_prompt", "") or "")
        on_event(BackendEvent("text_delta", "I will generate. "))
        call_tool("wangp_generate", {"prompt": "sunset"})
        on_event(BackendEvent("text_delta", "Done."))
        return "I will generate. Done."

    def interrupt(self):
        pass


class _AuthBackend(_FakeBackend):
    def run_turn(self, *_args, **_kwargs):
        raise CodexAuthenticationRequired("https://chatgpt.com/codex/auth")


class _ReasoningBackend(_ClosableBackend):
    def run_turn(self, _text, *, on_event, **_kwargs):
        on_event(BackendEvent("reasoning_delta", "Check both products.", {"item_id": "reasoning-1", "summary_index": 0}))
        on_event(BackendEvent("reasoning_delta", " Compare the results.", {"item_id": "reasoning-1", "summary_index": 1}))
        on_event(BackendEvent("text_delta", "The second is larger."))
        return "The second is larger."

    def interrupt(self):
        pass


class _HiddenReasoningToolBackend(_ClosableBackend):
    def run_turn(self, _text, *, on_event, call_tool, **_kwargs):
        on_event(BackendEvent("reasoning_start"))
        on_event(BackendEvent("tool_request_start", data={"name": "wangp_generate"}))
        call_tool("wangp_generate", {"prompt": "sunset"})
        on_event(BackendEvent("reasoning_start"))
        on_event(BackendEvent("text_delta", "Done."))
        return "Done."

    def interrupt(self):
        pass


class _RejectedToolBackend(_ClosableBackend):
    def run_turn(self, _text, *, on_event, **_kwargs):
        on_event(BackendEvent("tool_request_start", data={"id": "tool-1", "name": "mcp__wangp__wangp_artifact"}))
        on_event(BackendEvent("tool_request_error", "InputValidationError: JSON parse failed (42 bytes)", {"id": "tool-1", "name": "mcp__wangp__wangp_artifact", "input": {"__unparsedToolInput": {"raw": "{\"action\":\"commit", "len": 42}}}))
        return ""

    def interrupt(self):
        pass


class _CommentaryBackend(_ClosableBackend):
    def run_turn(self, _text, *, on_event, **_kwargs):
        on_event(BackendEvent("commentary_delta", "Checking the available settings", {"item_id": "update-1"}))
        on_event(BackendEvent("commentary_replace", "Checking the available settings.", {"item_id": "update-1"}))
        on_event(BackendEvent("commentary_delta", "The settings are compatible.", {"item_id": "update-2"}))
        on_event(BackendEvent("text_delta", "Everything is ready."))
        return "Everything is ready."

    def interrupt(self):
        pass


class _PromotedCommentaryBackend(_ClosableBackend):
    def run_turn(self, _text, *, on_event, **_kwargs):
        on_event(BackendEvent("commentary_delta", "Final answer", {"item_id": "answer-1"}))
        on_event(BackendEvent("commentary_promote", "Final answer.", {"item_id": "answer-1"}))
        return "Final answer."

    def interrupt(self):
        pass


class _RemovingCommentaryBackend(_ClosableBackend):
    def run_turn(self, _text, *, on_event, **_kwargs):
        report = "I'll gather the default settings first."
        on_event(BackendEvent("commentary_delta", report, {"item_id": "update-1"}))
        on_event(BackendEvent("commentary_delta", report, {"item_id": "update-2"}))
        on_event(BackendEvent("commentary_remove", data={"item_id": "update-2"}))
        on_event(BackendEvent("text_delta", "Done."))
        return "Done."

    def interrupt(self):
        pass


class _UsageCompactionBackend(_ClosableBackend):
    def run_turn(self, _text, *, on_event, **_kwargs):
        on_event(BackendEvent("compaction", data={"provider": "claude", "pre_tokens": 50000}))
        on_event(BackendEvent("text_delta", "Continued."))
        on_event(BackendEvent("usage", data={"input_tokens": 1200, "cached_input_tokens": 1000, "output_tokens": 80, "total_tokens": 1280}))
        return "Continued."

    def interrupt(self):
        pass


class _FakeToolbox:
    def bind_runtime_tools(self, **_kwargs):
        pass

    def get_tool_schemas(self):
        return [{"type": "function", "function": {"name": "wangp_generate", "description": "Generate", "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}}}}}]

    def get_remote_execution_instructions(self):
        return "Use direct long-running tool calls."

    def validate_tool_call(self, _name, _arguments):
        return ""

    def get_tool_transcript_label(self, _name, _arguments):
        return "Generate - sunset"

    def get_tool_display_name(self, name):
        return name.removeprefix("wangp_").replace("_", " ").title()

    def call(self, name, _arguments):
        return {"status": "done", "tool": name}


class RemoteLLMAdapterTests(unittest.TestCase):
    def test_local_tool_parser_keeps_xml_like_prompt_tags_inside_source(self):
        source = {"model_type": "minimax_h3", "prompt": "<Subject 1> speaks. <d>[English] Hello.</d> <action>waves</action> <wait>true</wait>"}
        raw = "\n".join(("<tool_call>", "<function=wangp_generate>", f"<parameter=source>{json.dumps(source)}</parameter>", "</function>", "</tool_call>"))
        self.assertEqual(extract_tool_calls(raw, {"wangp_generate": {"source", "wait"}}), [{"name": "wangp_generate", "arguments": {"source": source}}])

    def test_local_tool_parser_does_not_promote_unwrapped_prompt_tags(self):
        raw = "<tool_call><function=wangp_generate><d>[English] Hello.</d><parameter=wait>true</parameter></function></tool_call>"
        self.assertEqual(extract_tool_calls(raw, {"wangp_generate": {"source", "wait"}}), [{"name": "wangp_generate", "arguments": {"wait": True}}])

    def test_local_tool_parser_accepts_known_generic_parameters_only(self):
        source = {"model_type": "minimax_h3", "prompt": "<d>[English] Hello.</d>"}
        raw = f"<tool_call><function=wangp_generate><source>{json.dumps(source)}</source><d>wrong</d><wait>true</wait></function></tool_call>"
        self.assertEqual(extract_tool_calls(raw, {"wangp_generate": {"source", "wait"}}), [{"name": "wangp_generate", "arguments": {"source": source, "wait": True}}])

    def test_plain_text_llm_io_log_uses_readable_directions_and_numeric_token_ids(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = configure_llm_io(temp_dir)
            try:
                log_llm_io("OUT", "local-deepy", "generation", {"system_prompt": "First line\nSecond line", "input_token_ids": [151644, 872], "media": {"type": "image", "data": "x" * 1000}})
                log_llm_io("IN", "local-deepy", "generation", {"text": "Answer", "stop_token": {"id": 151645, "name": "<|im_end|>"}})
                transcript = path.read_text(encoding="utf-8")
            finally:
                configure_llm_io(None)
        self.assertIn("[OUT → LLM]", transcript)
        self.assertIn("[IN ← LLM]", transcript)
        self.assertIn("system_prompt: |\n  First line\n  Second line", transcript)
        self.assertIn("- 151644", transcript)
        self.assertIn("name: <|im_end|>", transcript)
        self.assertNotIn("\"system_prompt\"", transcript)
        self.assertNotIn("x" * 1000, transcript)

    def test_llm_io_cli_option_creates_the_transcript_in_the_requested_folder(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                args = parse_wgp_args("wgp_config.json", ["--llm-io", temp_dir])
                path = get_llm_io_path()
                self.assertEqual(args.llm_io, temp_dir)
                self.assertIsNotNone(path)
                self.assertEqual(path.parent, Path(temp_dir).resolve())
                self.assertTrue(path.is_file())
            finally:
                configure_llm_io(None)

    def test_remote_visual_query_uses_each_video_frame_selector(self):
        backend = Mock()
        backend.one_shot.return_value = "Compared."
        frames = (0, 360, 720, 1080, 1440)
        with tempfile.TemporaryDirectory() as temp_dir:
            video = Path(temp_dir) / "source.mp4"
            video.touch()
            records = [{"media_id": "video_1", "media_type": "video", "path": str(video), "label": "Source", "frame_no": frame, "time_seconds": None} for frame in frames]
            with patch("shared.remote_llm.deepy_runner.resolve_role_engine", return_value="codex"), patch("shared.remote_llm.deepy_runner.is_remote_engine", return_value=True), patch("shared.remote_llm.deepy_runner.create_backend", return_value=backend), patch("shared.remote_llm.deepy_runner.deepy_video_tools.resolve_video_frame_no", side_effect=lambda _path, frame_no=None, time_seconds=None: int(frame_no)), patch("shared.remote_llm.deepy_runner.deepy_vision.decode_inspection_video_frames", side_effect=lambda _path, indices, max_edge=None: [Image.new("RGB", (4, 4), (frame % 255, 0, 0)) for frame in indices]):
                result = _visual_query({}, records, "Compare these frames.")

        self.assertEqual([item["frame_no"] for item in result["media"]], list(frames))
        self.assertEqual(result["answer"], "Compared.")
        backend.one_shot.assert_called_once()
        backend.close.assert_called_once()

    def test_remote_visual_query_resizes_images_before_submission(self):
        submitted_sizes = []
        backend = Mock()

        def one_shot(_question, *, images, **_kwargs):
            for path in images:
                with Image.open(path) as image:
                    submitted_sizes.append(image.size)
            return "Inspected."

        backend.one_shot.side_effect = one_shot
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "wide.png"
            Image.new("RGB", (2400, 1200), "red").save(source)
            record = {"media_id": "image_1", "media_type": "image", "path": str(source), "label": "Wide"}
            with patch("shared.remote_llm.deepy_runner.resolve_role_engine", return_value="codex"), patch("shared.remote_llm.deepy_runner.is_remote_engine", return_value=True), patch("shared.remote_llm.deepy_runner.create_backend", return_value=backend):
                result = _visual_query({}, record, "What is shown?")

        self.assertEqual(result["answer"], "Inspected.")
        self.assertEqual(submitted_sizes, [(1024, 512)])
        self.assertIn("Visual 1: image Wide.", backend.one_shot.call_args.args[0])

    def test_remote_visual_query_exposes_only_the_virtual_media_path(self):
        backend = Mock()
        backend.one_shot.return_value = "Inspected."
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "private" / "wide.png"
            source.parent.mkdir()
            Image.new("RGB", (8, 8), "red").save(source)
            policy = FileAccessPolicy(mode="read", output_roots=(Path(temp_dir).resolve(),))
            record = {"media_id": "image_1", "media_type": "image", "path": str(source), "label": f"Source {source}"}
            with patch("shared.remote_llm.deepy_runner.resolve_role_engine", return_value="codex"), patch("shared.remote_llm.deepy_runner.is_remote_engine", return_value=True), patch("shared.remote_llm.deepy_runner.create_backend", return_value=backend):
                result = _visual_query({}, record, "What is shown?", file_access_policy=policy)

        submitted_question = backend.one_shot.call_args.args[0]
        self.assertIn("@outputs/private/wide.png", submitted_question)
        self.assertNotIn(str(source), submitted_question)
        self.assertEqual(result["path"], "@outputs/private/wide.png")

    def test_remote_visual_query_uses_media_id_outside_virtual_roots(self):
        backend = Mock()
        backend.one_shot.return_value = "Inspected."
        with tempfile.TemporaryDirectory() as output_dir, tempfile.TemporaryDirectory() as gallery_dir:
            source = Path(gallery_dir) / "upload.png"
            Image.new("RGB", (8, 8), "red").save(source)
            policy = FileAccessPolicy(mode="read", output_roots=(Path(output_dir).resolve(),))
            record = {"media_id": "image_7", "media_type": "image", "path": str(source), "label": "Upload"}
            with patch("shared.remote_llm.deepy_runner.resolve_role_engine", return_value="codex"), patch("shared.remote_llm.deepy_runner.is_remote_engine", return_value=True), patch("shared.remote_llm.deepy_runner.create_backend", return_value=backend):
                result = _visual_query({}, record, "What is shown?", file_access_policy=policy)

        submitted_question = backend.one_shot.call_args.args[0]
        self.assertIn("image_7", submitted_question)
        self.assertNotIn(str(source), submitted_question)
        self.assertEqual(result["path"], "image_7")

    def test_remote_visual_query_leaves_active_event_loop_before_starting_backend(self):
        backend = Mock()

        def one_shot(*_args, **_kwargs):
            with self.assertRaises(RuntimeError):
                asyncio.get_running_loop()
            return "Inspected outside the MCP event loop."

        backend.one_shot.side_effect = one_shot
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "image.png"
            Image.new("RGB", (8, 8), "blue").save(source)
            record = {"media_id": "image_1", "media_type": "image", "path": str(source), "label": "Image"}

            async def inspect_from_mcp_loop():
                with patch("shared.remote_llm.deepy_runner.resolve_role_engine", return_value="claude"), patch("shared.remote_llm.deepy_runner.is_remote_engine", return_value=True), patch("shared.remote_llm.deepy_runner.create_backend", return_value=backend):
                    return _visual_query({}, record, "What is shown?")

            result = asyncio.run(inspect_from_mcp_loop())

        self.assertEqual(result["answer"], "Inspected outside the MCP event loop.")
        backend.close.assert_called_once()

    def test_codex_dynamic_tool_schema_uses_app_server_shape(self):
        tools = [{"type": "function", "function": {"name": "wangp_generate", "description": "Generate", "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}}}}}]
        self.assertEqual(CodexBackend._dynamic_tools(tools), [{"name": "wangp_generate", "description": "Generate", "inputSchema": tools[0]["function"]["parameters"]}])

    def test_codex_auth_flow_returns_secure_browser_url(self):
        backend = CodexBackend({})
        backend._request = Mock(side_effect=[{"account": None, "requiresOpenaiAuth": True}, {"loginId": "login-1", "authUrl": "https://chatgpt.com/codex/auth?state=one"}])
        try:
            with self.assertRaises(CodexAuthenticationRequired) as raised:
                backend._ensure_authenticated()
            self.assertEqual(raised.exception.auth_url, "https://chatgpt.com/codex/auth?state=one")
            self.assertEqual(backend._request.call_args_list[1].args, ("account/login/start", {"type": "chatgpt", "useHostedLoginSuccessPage": True, "appBrand": "codex"}))
        finally:
            backend.close()

    def test_codex_opts_into_experimental_dynamic_tools_api(self):
        backend = CodexBackend({})
        backend._request = Mock(return_value={})
        backend._notify = Mock()
        backend._ensure_authenticated = Mock()
        process = Mock()
        try:
            with patch("shared.remote_llm.codex_backend.subprocess.Popen", return_value=process), patch("shared.remote_llm.codex_backend.threading.Thread"):
                backend._start()
            initialize = backend._request.call_args_list[0]
            self.assertEqual(initialize.args[0], "initialize")
            self.assertTrue(initialize.args[1]["capabilities"]["experimentalApi"])
        finally:
            backend.close()

    def test_codex_streams_readable_reasoning_summaries(self):
        backend = CodexBackend({})
        events = []
        backend._on_event = events.append
        backend._process = Mock(stdout=io.StringIO(
            json.dumps({"method": "item/reasoning/summaryTextDelta", "params": {"itemId": "reasoning-1", "summaryIndex": 2, "delta": "Checking the result."}}) + "\n"
        ))
        try:
            backend._read_loop()
            self.assertEqual(events, [BackendEvent("reasoning_delta", "Checking the result.", {"item_id": "reasoning-1", "summary_index": 2})])
        finally:
            backend._process = None
            backend.close()

    def test_codex_keeps_commentary_separate_from_final_answer(self):
        backend = CodexBackend({})
        events = []
        backend._on_event = events.append
        messages = [
            {"method": "item/started", "params": {"item": {"type": "agentMessage", "id": "update-1", "text": "", "phase": "commentary"}}},
            {"method": "item/agentMessage/delta", "params": {"itemId": "update-1", "delta": "Checking"}},
            {"method": "item/completed", "params": {"item": {"type": "agentMessage", "id": "update-1", "text": "Checking inputs.", "phase": "commentary"}}},
            {"method": "item/started", "params": {"item": {"type": "agentMessage", "id": "final-1", "text": "", "phase": "final_answer"}}},
            {"method": "item/agentMessage/delta", "params": {"itemId": "final-1", "delta": "Done"}},
            {"method": "item/completed", "params": {"item": {"type": "agentMessage", "id": "final-1", "text": "Done.", "phase": "final_answer"}}},
            {"method": "turn/completed", "params": {"turn": {"status": "completed"}}},
        ]
        backend._process = Mock(stdout=io.StringIO("".join(json.dumps(message) + "\n" for message in messages)))
        try:
            backend._read_loop()
            self.assertEqual([event.kind for event in events], ["commentary_delta", "commentary_replace", "text_delta", "text_replace"])
            self.assertEqual(events[0].data, {"item_id": "update-1"})
            self.assertEqual(backend._text, "Done.")
        finally:
            backend._process = None
            backend.close()

    def test_codex_reports_turn_usage_and_context_compaction(self):
        backend = CodexBackend({})
        events = []
        backend._on_event = events.append
        backend._turn_usage_start = {
            "input_tokens": 1000,
            "cached_input_tokens": 700,
            "cache_write_input_tokens": 0,
            "output_tokens": 100,
            "reasoning_output_tokens": 20,
            "total_tokens": 1100,
        }
        usage = {
            "last": {"inputTokens": 550, "cachedInputTokens": 500, "cacheWriteInputTokens": 0, "outputTokens": 50, "reasoningOutputTokens": 30, "totalTokens": 600},
            "total": {"inputTokens": 1300, "cachedInputTokens": 950, "cacheWriteInputTokens": 0, "outputTokens": 180, "reasoningOutputTokens": 60, "totalTokens": 1480},
            "modelContextWindow": 200000,
        }
        messages = [
            {"method": "item/completed", "params": {"item": {"type": "contextCompaction", "id": "compact-1"}}},
            {"method": "thread/tokenUsage/updated", "params": {"threadId": "thread-1", "turnId": "turn-1", "tokenUsage": usage}},
            {"method": "turn/completed", "params": {"turn": {"status": "completed"}}},
        ]
        backend._process = Mock(stdout=io.StringIO("".join(json.dumps(message) + "\n" for message in messages)))
        try:
            backend._read_loop()
            self.assertEqual(events[0], BackendEvent("compaction", data={"provider": "codex"}))
            self.assertEqual(events[1], BackendEvent("usage", data={
                "input_tokens": 300,
                "cached_input_tokens": 250,
                "cache_write_input_tokens": 0,
                "output_tokens": 80,
                "reasoning_output_tokens": 40,
                "total_tokens": 380,
                "context_tokens": 600,
                "context_window": 200000,
            }))
        finally:
            backend._process = None
            backend.close()

    def test_remote_usage_hides_cumulative_breakdown_by_default(self):
        stats = build_remote_usage_stats({
            "input_tokens": 28777,
            "cached_input_tokens": 25344,
            "output_tokens": 178,
            "reasoning_output_tokens": 111,
            "total_tokens": 28955,
            "context_tokens": 28955,
            "context_window": 258400,
        })
        self.assertEqual(stats["text"], "context 28,955 / 258,400 tk")

    def test_claude_context_window_uses_resolved_model_limit(self):
        self.assertEqual(claude_context_window("claude-sonnet-5"), 1_000_000)
        self.assertEqual(claude_context_window("claude-opus-4-6"), 1_000_000)
        self.assertEqual(claude_context_window("claude-haiku-4-5-20251001"), 200_000)
        self.assertEqual(claude_context_window(""), 0)

    def test_claude_mcp_compat_returns_unwrapped_content(self):
        class Server:
            def call_tool(self):
                def register(handler):
                    self.handler = handler
                    return handler
                return register

        async def handler(_arguments):
            return {"content": [{"type": "text", "text": "done"}]}

        server = Server()
        _restore_mcp_content_contract({"instance": server}, [SimpleNamespace(name="echo", handler=handler)])
        result = asyncio.run(server.handler("echo", {}))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].text, "done")

    def test_claude_surfaces_predispatch_tool_json_error(self):
        class ClaudeAgentOptions:
            __dataclass_fields__ = {"thinking": None, "effort": None}

            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class ToolUseBlock:
            id = "tool-1"
            name = "mcp__wangp__wangp_artifact"
            input = {"__unparsedToolInput": {"raw": "{\"action\":\"commit", "len": 42}}

        class ToolResultBlock:
            tool_use_id = "tool-1"
            content = "InputValidationError: JSON parse failed (42 bytes)"
            is_error = True

        class AssistantMessage:
            content = [ToolUseBlock()]
            session_id = "session-1"
            message_id = "message-1"
            stop_reason = "tool_use"

        class UserMessage:
            content = [ToolResultBlock()]
            session_id = "session-1"
            message_id = "message-2"
            stop_reason = ""

        class ClaudeSDKClient:
            def __init__(self, options):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def query(self, _prompt):
                pass

            async def interrupt(self):
                pass

            async def receive_response(self):
                yield AssistantMessage()
                yield UserMessage()

        sdk = SimpleNamespace(ClaudeAgentOptions=ClaudeAgentOptions, ClaudeSDKClient=ClaudeSDKClient)
        backend = ClaudeBackend({})
        events = []
        try:
            with patch.object(backend, "_sdk", return_value=sdk):
                backend.run_turn("hello", system_prompt="system", tools=[], images=[], on_event=events.append, call_tool=lambda _name, _args: {}, should_stop=lambda: False)
            error = next(event for event in events if event.kind == "tool_request_error")
            self.assertEqual(error.text, "InputValidationError: JSON parse failed (42 bytes)")
            self.assertEqual(error.data["input"]["__unparsedToolInput"]["raw"], "{\"action\":\"commit")
        finally:
            backend.close()

    def test_codex_requests_concise_reasoning_summaries(self):
        backend = CodexBackend({})
        backend._start = Mock()
        backend._ensure_thread = Mock()
        backend._thread_id = "thread-1"
        captured = {}

        def request(method, params, timeout=60):
            captured.update({"method": method, "params": params, "timeout": timeout})
            backend._turn_done.set()
            return {"turn": {"id": "turn-1"}}

        backend._request = request
        try:
            backend.run_turn("hello", system_prompt="system", tools=[], images=[], on_event=lambda _event: None, call_tool=lambda _name, _args: {}, should_stop=lambda: False)
            self.assertEqual(captured["method"], "turn/start")
            self.assertEqual(captured["params"]["summary"], "concise")
        finally:
            backend.close()

    def test_codex_lists_picker_visible_models_with_pagination(self):
        backend = CodexBackend({})
        backend._start = Mock()
        backend._request = Mock(side_effect=[
            {"data": [{"id": "gpt-a", "model": "gpt-a", "displayName": "GPT A", "isDefault": True, "defaultReasoningEffort": "low", "supportedReasoningEfforts": [{"reasoningEffort": "low"}, {"reasoningEffort": "high"}]}], "nextCursor": "next"},
            {"data": [{"id": "gpt-b", "model": "gpt-b", "displayName": "GPT B", "isDefault": False, "defaultReasoningEffort": "medium", "supportedReasoningEfforts": [{"reasoningEffort": "medium"}]}], "nextCursor": None},
        ])
        try:
            self.assertEqual(backend.list_models(), [
                {"model": "gpt-a", "display_name": "GPT A", "is_default": True, "default_reasoning_effort": "low", "reasoning_efforts": ["low", "high"]},
                {"model": "gpt-b", "display_name": "GPT B", "is_default": False, "default_reasoning_effort": "medium", "reasoning_efforts": ["medium"]},
            ])
            self.assertEqual(backend._request.call_args_list[0].args, ("model/list", {"limit": 100, "includeHidden": False}))
            self.assertEqual(backend._request.call_args_list[1].args, ("model/list", {"limit": 100, "includeHidden": False, "cursor": "next"}))
        finally:
            backend.close()

    def test_claude_streams_summarized_thoughts_progress_and_final_answer(self):
        class ClaudeAgentOptions:
            __dataclass_fields__ = {"thinking": None, "effort": None}

            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class StreamEvent:
            def __init__(self, event):
                self.event = event
                self.session_id = "session-1"

        class ThinkingBlock:
            def __init__(self, thinking):
                self.thinking = thinking

        class TextBlock:
            def __init__(self, text):
                self.text = text

        class ToolUseBlock:
            pass

        class AssistantMessage:
            def __init__(self, content, stop_reason):
                self.content = content
                self.stop_reason = stop_reason
                self.session_id = "session-1"
                self.message_id = ""

        messages = [
            StreamEvent({"type": "message_start", "message": {"id": "message-1"}}),
            StreamEvent({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}}),
            StreamEvent({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Checking the inputs."}}),
            StreamEvent({"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}),
            StreamEvent({"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Checking inputs"}}),
            StreamEvent({"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "tool-1", "name": "mcp__wangp__wangp_generate", "input": {}}}),
            AssistantMessage([ThinkingBlock("Checking the inputs."), TextBlock("Checking inputs."), ToolUseBlock()], "tool_use"),
            StreamEvent({"type": "message_start", "message": {"id": "message-2"}}),
            StreamEvent({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            StreamEvent({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Done"}}),
            AssistantMessage([TextBlock("Done.")], "end_turn"),
        ]

        class ClaudeSDKClient:
            options = None

            def __init__(self, options):
                ClaudeSDKClient.options = options

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def query(self, _prompt):
                pass

            async def interrupt(self):
                pass

            async def receive_response(self):
                for message in messages:
                    yield message

        sdk = SimpleNamespace(ClaudeAgentOptions=ClaudeAgentOptions, ClaudeSDKClient=ClaudeSDKClient)
        backend = ClaudeBackend({"model": "sonnet", "reasoning_effort": "high"})
        events = []
        try:
            with patch.object(backend, "_sdk", return_value=sdk):
                answer = backend.run_turn("hello", system_prompt="system", tools=[], images=[], on_event=events.append, call_tool=lambda _name, _args: {}, should_stop=lambda: False)
            self.assertEqual(answer, "Done.")
            self.assertEqual([event.kind for event in events], ["reasoning_start", "reasoning_delta", "commentary_delta", "tool_request_start", "commentary_replace", "commentary_delta", "commentary_promote"])
            self.assertEqual(events[1].text, "Checking the inputs.")
            self.assertEqual(events[3].data, {"id": "tool-1", "name": "mcp__wangp__wangp_generate"})
            self.assertEqual(ClaudeSDKClient.options.kwargs["thinking"], {"type": "adaptive", "display": "summarized"})
            self.assertEqual(ClaudeSDKClient.options.kwargs["effort"], "high")
            self.assertNotIn("strict_mcp_config", ClaudeSDKClient.options.kwargs)
            self.assertIn(CLAUDE_PROGRESS_INSTRUCTIONS, ClaudeSDKClient.options.kwargs["system_prompt"])
        finally:
            backend.close()

    def test_claude_lists_models_from_sdk_initialization(self):
        catalog = ClaudeBackend._model_catalog({"models": [
            {"value": "default", "displayName": "Default"},
            {"value": "sonnet", "displayName": "Sonnet", "supportsEffort": True, "supportedEffortLevels": ["low", "high"], "defaultEffort": "high"},
            {"value": "opus", "displayName": "Opus", "supportsEffort": True, "supportedEffortLevels": [{"value": "high"}, {"value": "max"}]},
        ]})
        self.assertEqual(catalog, [
            {"model": "sonnet", "display_name": "Sonnet", "is_default": False, "default_reasoning_effort": "high", "reasoning_efforts": ["low", "high"]},
            {"model": "opus", "display_name": "Opus", "is_default": False, "default_reasoning_effort": "", "reasoning_efforts": ["high", "max"]},
        ])

    def test_claude_keeps_separate_pre_tool_text_as_commentary(self):
        class ClaudeAgentOptions:
            __dataclass_fields__ = {"thinking": object(), "effort": object()}

            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class StreamEvent:
            session_id = "session-1"

            def __init__(self, event):
                self.event = event

        class TextBlock:
            def __init__(self, text):
                self.text = text

        class ToolUseBlock:
            pass

        class AssistantMessage:
            session_id = "session-1"
            message_id = ""

            def __init__(self, content, stop_reason):
                self.content = content
                self.stop_reason = stop_reason

        messages = [
            StreamEvent({"type": "message_start", "message": {"id": "message-1"}}),
            StreamEvent({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            StreamEvent({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Checking the settings."}}),
            AssistantMessage([TextBlock("Checking the settings.")], "end_turn"),
            AssistantMessage([ToolUseBlock()], "tool_use"),
            StreamEvent({"type": "message_start", "message": {"id": "message-2"}}),
            StreamEvent({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            StreamEvent({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "Done"}}),
            AssistantMessage([TextBlock("Done.")], "end_turn"),
        ]

        class ClaudeSDKClient:
            def __init__(self, options):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def query(self, _prompt):
                pass

            async def interrupt(self):
                pass

            async def receive_response(self):
                for message in messages:
                    yield message

        sdk = SimpleNamespace(ClaudeAgentOptions=ClaudeAgentOptions, ClaudeSDKClient=ClaudeSDKClient)
        backend = ClaudeBackend({})
        events = []
        try:
            with patch.object(backend, "_sdk", return_value=sdk):
                answer = backend.run_turn("hello", system_prompt="system", tools=[], images=[], on_event=events.append, call_tool=lambda _name, _args: {}, should_stop=lambda: False)
            self.assertEqual(answer, "Done.")
            self.assertEqual([event.kind for event in events], ["commentary_delta", "commentary_delta", "commentary_promote"])
            self.assertEqual([event.text for event in events], ["Checking the settings.", "Done", "Done."])
        finally:
            backend.close()

    def test_claude_removes_repeated_pre_tool_report_from_a_second_sdk_message(self):
        class ClaudeAgentOptions:
            __dataclass_fields__ = {"thinking": object(), "effort": object()}

            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class StreamEvent:
            session_id = "session-1"

            def __init__(self, event):
                self.event = event

        class TextBlock:
            def __init__(self, text):
                self.text = text

        class ThinkingBlock:
            thinking = ""

        class ToolUseBlock:
            pass

        class AssistantMessage:
            session_id = "session-1"
            message_id = ""

            def __init__(self, content, stop_reason):
                self.content = content
                self.stop_reason = stop_reason

        report = "I'll gather the default settings first."
        messages = [
            StreamEvent({"type": "message_start", "message": {"id": "message-1"}}),
            StreamEvent({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}}),
            AssistantMessage([ThinkingBlock()], ""),
            StreamEvent({"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}),
            StreamEvent({"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": report}}),
            AssistantMessage([TextBlock(report)], "end_turn"),
            StreamEvent({"type": "message_start", "message": {"id": "message-2"}}),
            StreamEvent({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}}),
            StreamEvent({"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}),
            StreamEvent({"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": report}}),
            AssistantMessage([TextBlock(report)], "end_turn"),
            AssistantMessage([ToolUseBlock()], "tool_use"),
            StreamEvent({"type": "message_stop"}),
            StreamEvent({"type": "message_start", "message": {"id": "message-3"}}),
            StreamEvent({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}}),
            AssistantMessage([ThinkingBlock()], ""),
            StreamEvent({"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}),
            StreamEvent({"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Done"}}),
            AssistantMessage([TextBlock("Done.")], "end_turn"),
            StreamEvent({"type": "message_stop"}),
        ]

        class ClaudeSDKClient:
            def __init__(self, options):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def query(self, _prompt):
                pass

            async def interrupt(self):
                pass

            async def receive_response(self):
                for message in messages:
                    yield message

        backend = ClaudeBackend({})
        events = []
        try:
            sdk = SimpleNamespace(ClaudeAgentOptions=ClaudeAgentOptions, ClaudeSDKClient=ClaudeSDKClient)
            with patch.object(backend, "_sdk", return_value=sdk):
                answer = backend.run_turn("hello", system_prompt="system", tools=[], images=[], on_event=events.append, call_tool=lambda _name, _args: {}, should_stop=lambda: False)
            self.assertEqual(answer, "Done.")
            self.assertEqual([event.kind for event in events], ["reasoning_start", "commentary_delta", "reasoning_start", "commentary_delta", "commentary_remove", "reasoning_start", "commentary_delta", "commentary_promote"])
            self.assertEqual(events[4].data, {"item_id": "message-2:text:1"})
        finally:
            backend.close()

    def test_claude_reports_usage_and_compaction_boundary(self):
        class ClaudeAgentOptions:
            __dataclass_fields__ = {"thinking": object(), "effort": object()}

            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class SystemMessage:
            subtype = "compact_boundary"
            data = {"pre_tokens": 42000}
            session_id = "session-1"

        class TextBlock:
            def __init__(self, text):
                self.text = text

        class AssistantMessage:
            content = [TextBlock("Done.")]
            stop_reason = "end_turn"
            session_id = "session-1"
            message_id = "message-1"

        class ResultMessage:
            content = []
            session_id = "session-1"
            usage = {"input_tokens": 200, "cache_read_input_tokens": 800, "cache_creation_input_tokens": 50, "output_tokens": 75}

        class ClaudeSDKClient:
            def __init__(self, options):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def query(self, _prompt):
                pass

            async def interrupt(self):
                pass

            async def receive_response(self):
                for message in (SystemMessage(), AssistantMessage(), ResultMessage()):
                    yield message

        sdk = SimpleNamespace(ClaudeAgentOptions=ClaudeAgentOptions, ClaudeSDKClient=ClaudeSDKClient)
        backend = ClaudeBackend({})
        events = []
        try:
            with patch.object(backend, "_sdk", return_value=sdk):
                answer = backend.run_turn("hello", system_prompt="system", tools=[], images=[], on_event=events.append, call_tool=lambda _name, _args: {}, should_stop=lambda: False)
            self.assertEqual(answer, "Done.")
            self.assertEqual(events[0], BackendEvent("compaction", data={"provider": "claude", "pre_tokens": 42000}))
            self.assertEqual(events[-1], BackendEvent("usage", data={
                "input_tokens": 1050,
                "cached_input_tokens": 800,
                "cache_write_input_tokens": 50,
                "output_tokens": 75,
                "reasoning_output_tokens": 0,
                "total_tokens": 1125,
                "context_tokens": 0,
                "context_window": 0,
            }))
        finally:
            backend.close()

    def test_claude_updates_usage_after_each_streamed_model_action(self):
        class ClaudeAgentOptions:
            __dataclass_fields__ = {"thinking": object(), "effort": object()}

            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class StreamEvent:
            session_id = "session-1"

            def __init__(self, event):
                self.event = event

        class ResultMessage:
            content = []
            session_id = "session-1"
            usage = {"input_tokens": 5, "cache_read_input_tokens": 220, "cache_creation_input_tokens": 30, "output_tokens": 70, "output_tokens_details": {"thinking_tokens": 5}}

        class SystemMessage:
            subtype = "init"
            data = {"model": "claude-sonnet-5"}
            session_id = "session-1"

        messages = [
            SystemMessage(),
            StreamEvent({"type": "message_start", "message": {"id": "message-1"}}),
            StreamEvent({"type": "message_delta", "usage": {"input_tokens": 2, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 10, "output_tokens": 30, "output_tokens_details": {"thinking_tokens": 5}}}),
            StreamEvent({"type": "message_stop"}),
            StreamEvent({"type": "message_start", "message": {"id": "message-2"}}),
            StreamEvent({"type": "message_delta", "usage": {"input_tokens": 3, "cache_read_input_tokens": 120, "cache_creation_input_tokens": 20, "output_tokens": 40, "output_tokens_details": {"thinking_tokens": 0}}}),
            StreamEvent({"type": "message_stop"}),
            ResultMessage(),
        ]

        class ClaudeSDKClient:
            def __init__(self, options):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def query(self, _prompt):
                pass

            async def interrupt(self):
                pass

            async def receive_response(self):
                for message in messages:
                    yield message

        backend = ClaudeBackend({})
        events = []
        try:
            sdk = SimpleNamespace(ClaudeAgentOptions=ClaudeAgentOptions, ClaudeSDKClient=ClaudeSDKClient)
            with patch.object(backend, "_sdk", return_value=sdk):
                backend.run_turn("hello", system_prompt="system", tools=[], images=[], on_event=events.append, call_tool=lambda _name, _args: {}, should_stop=lambda: False)
            usage_events = [event.data for event in events if event.kind == "usage"]
            self.assertEqual(usage_events[0], {"input_tokens": 112, "cached_input_tokens": 100, "cache_write_input_tokens": 10, "output_tokens": 30, "reasoning_output_tokens": 5, "total_tokens": 142, "context_tokens": 142, "context_window": 1_000_000})
            self.assertEqual(usage_events[1], {"input_tokens": 255, "cached_input_tokens": 220, "cache_write_input_tokens": 30, "output_tokens": 70, "reasoning_output_tokens": 5, "total_tokens": 325, "context_tokens": 183, "context_window": 1_000_000})
            self.assertEqual(usage_events[2]["reasoning_output_tokens"], 5)
            self.assertEqual(usage_events[2]["total_tokens"], 325)
            self.assertEqual(usage_events[2]["context_tokens"], 183)
            self.assertEqual(build_remote_usage_stats(usage_events[2])["text"], "context 183 / 1,000,000 tk")
        finally:
            backend.close()

    def test_claude_stop_watcher_remains_responsive_during_tool_call(self):
        class ClaudeAgentOptions:
            __dataclass_fields__ = {"thinking": object(), "effort": object()}

            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class ResultMessage:
            content = []
            session_id = "session-1"
            usage = {}

        handlers = {}
        stop_requested = threading.Event()
        interrupt_received = threading.Event()
        release_tool = threading.Event()

        def tool(name, _description, _schema):
            def register(handler):
                handlers[name] = handler
                return handler
            return register

        def create_sdk_mcp_server(**kwargs):
            return kwargs

        class ClaudeSDKClient:
            def __init__(self, options):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def query(self, _prompt):
                pass

            async def interrupt(self):
                interrupt_received.set()
                release_tool.set()

            async def receive_response(self):
                task = asyncio.create_task(handlers["slow_tool"]({}))
                while not interrupt_received.is_set():
                    await asyncio.sleep(0.01)
                await task
                yield ResultMessage()

        def call_tool(_name, _arguments):
            stop_requested.set()
            release_tool.wait(2)
            return {"status": "interrupted"}

        sdk = SimpleNamespace(ClaudeAgentOptions=ClaudeAgentOptions, ClaudeSDKClient=ClaudeSDKClient, tool=tool, create_sdk_mcp_server=create_sdk_mcp_server)
        backend = ClaudeBackend({})
        started = time.monotonic()
        try:
            with patch.object(backend, "_sdk", return_value=sdk):
                backend.run_turn("hello", system_prompt="system", tools=[{"function": {"name": "slow_tool", "description": "Slow", "parameters": {"type": "object"}}}], images=[], on_event=lambda _event: None, call_tool=call_tool, should_stop=stop_requested.is_set)
            self.assertTrue(interrupt_received.is_set())
            self.assertLess(time.monotonic() - started, 1.0)
        finally:
            release_tool.set()
            backend.close()

    def test_claude_authentication_failure_is_user_action(self):
        class ClaudeAgentOptions:
            __dataclass_fields__ = {"thinking": object(), "effort": object()}

            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class TextBlock:
            def __init__(self, text):
                self.text = text

        class AssistantMessage:
            content = [TextBlock("Failed to authenticate: OAuth session expired and could not be refreshed")]
            session_id = ""
            stop_reason = ""

        class ClaudeSDKClient:
            def __init__(self, options):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_args):
                return False

            async def query(self, _prompt):
                pass

            async def interrupt(self):
                pass

            async def receive_response(self):
                yield AssistantMessage()

        sdk = SimpleNamespace(ClaudeAgentOptions=ClaudeAgentOptions, ClaudeSDKClient=ClaudeSDKClient)
        backend = ClaudeBackend({})
        try:
            with patch.object(backend, "_sdk", return_value=sdk), self.assertRaises(ClaudeAuthenticationRequired) as raised:
                backend.run_turn("hello", system_prompt="system", tools=[], images=[], on_event=lambda _event: None, call_tool=lambda _name, _args: {}, should_stop=lambda: False)
            self.assertTrue(raised.exception.user_action_required)
            self.assertTrue(raised.exception.preserve_backend)
        finally:
            backend.close()

    @unittest.skipUnless(os.name == "nt", "Windows npm wrapper behavior")
    def test_codex_resolver_prefers_native_path_cli_and_respects_explicit_wrapper(self):
        with tempfile.TemporaryDirectory() as root:
            native = Path(root, "codex.exe")
            native.touch()
            wrapper = Path(root, "npm", "codex.cmd")
            wrapper.parent.mkdir()
            wrapper.touch()
            with patch.dict(os.environ, {"APPDATA": root}), patch("shared.remote_llm.codex_backend.shutil.which", side_effect=lambda name: str(native) if name == "codex.exe" else None):
                self.assertEqual(_resolve_codex_executable("codex"), str(native))
                self.assertEqual(_resolve_codex_executable(str(wrapper)), str(wrapper))

    @unittest.skipUnless(os.name == "nt", "Windows subprocess tree cleanup")
    def test_codex_resolver_finds_app_cli_without_app_on_path(self):
        with tempfile.TemporaryDirectory() as root:
            app_bin = Path(root, "OpenAI", "Codex", "bin")
            native = app_bin / "current-build" / "codex.exe"
            native.parent.mkdir(parents=True)
            native.touch()
            old = app_bin / "codex.exe"
            old.touch()
            os.utime(old, (1, 1))
            with patch.dict(os.environ, {"LOCALAPPDATA": root}), patch("shared.remote_llm.codex_backend.shutil.which", return_value=None):
                self.assertEqual(_resolve_codex_executable("codex"), str(native))

    @unittest.skipUnless(os.name == "nt", "Windows subprocess tree cleanup")
    def test_codex_close_stops_children_before_removing_working_directory(self):
        import psutil

        backend = CodexBackend({})
        directory = Path(backend._temp_dir.name)
        script = "import subprocess, sys, time; child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); print(child.pid, flush=True); time.sleep(60)"
        backend._process = subprocess.Popen([sys.executable, "-c", script], cwd=directory, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
        child_pid = int(backend._process.stdout.readline())
        backend.close()
        self.assertFalse(psutil.pid_exists(child_pid))
        self.assertFalse(directory.exists())
        backend.close()

    @unittest.skipUnless(os.name == "nt", "Windows npm wrapper behavior")
    def test_codex_resolver_prefers_global_npm_wrapper_and_uses_cmd(self):
        with tempfile.TemporaryDirectory() as root:
            wrapper = Path(root, "npm", "codex.cmd")
            wrapper.parent.mkdir()
            wrapper.touch()
            with patch.dict(os.environ, {"APPDATA": root, "LOCALAPPDATA": root}), patch("shared.remote_llm.codex_backend.shutil.which", return_value=r"C:\Program Files\WindowsApps\OpenAI.Codex\codex.exe"):
                self.assertEqual(_resolve_codex_executable("codex"), str(wrapper))
            self.assertIn("cmd", Path(_codex_launch_command(str(wrapper))[0]).name.lower())

    @unittest.skipUnless(os.name == "nt", "Windows VS Code extension behavior")
    def test_codex_resolver_uses_vscode_extension_bundle(self):
        with tempfile.TemporaryDirectory() as root:
            executable = Path(root, ".vscode", "extensions", "openai.chatgpt-1.2.3-win32-x64", "bin", "windows-x86_64", "codex.exe")
            executable.parent.mkdir(parents=True)
            executable.touch()
            with patch.dict(os.environ, {"APPDATA": str(Path(root, "appdata")), "USERPROFILE": root, "LOCALAPPDATA": root}), patch("shared.remote_llm.codex_backend.shutil.which", return_value=r"C:\Program Files\WindowsApps\OpenAI.Codex\codex.exe"):
                self.assertEqual(_resolve_codex_executable("codex"), str(executable))

    @unittest.skipUnless(os.name == "nt", "Windows VS Code extension behavior")
    def test_claude_resolver_uses_vscode_extension_bundle(self):
        with tempfile.TemporaryDirectory() as root:
            executable = Path(root, ".vscode", "extensions", "anthropic.claude-code-2.1.238-win32-x64", "resources", "native-binary", "claude.exe")
            executable.parent.mkdir(parents=True)
            executable.touch()
            with patch.dict(os.environ, {"USERPROFILE": root}), patch("shared.remote_llm.claude_backend.shutil.which", return_value=None):
                self.assertEqual(_resolve_claude_executable("claude"), str(executable))

    def test_claude_options_use_auto_detected_executable(self):
        class ClaudeAgentOptions:
            __dataclass_fields__ = {"thinking": object(), "effort": object()}

            def __init__(self, **kwargs):
                self.kwargs = kwargs

        backend = ClaudeBackend({"executable": "claude"})
        try:
            with patch("shared.remote_llm.claude_backend._resolve_claude_executable", return_value=r"C:\\Claude\\claude.exe"):
                options = backend._options(SimpleNamespace(ClaudeAgentOptions=ClaudeAgentOptions), "system", {}, [])
            self.assertEqual(options.kwargs["cli_path"], r"C:\\Claude\\claude.exe")
        finally:
            backend.close()

    def test_assistant_external_links_open_in_new_window(self):
        rendered = assistant_chat._markdown_to_html("[Sign in](https://chatgpt.com/codex/auth)")
        self.assertIn('target="_blank"', rendered)
        self.assertIn('rel="noopener noreferrer"', rendered)

    def test_opencode_extracts_text_parts(self):
        payload = {"parts": [{"type": "text", "text": "hello "}, {"type": "tool", "text": "ignored"}, {"type": "text", "text": "world"}]}
        self.assertEqual(OpenCodeBackend._answer_text(payload), "hello world")

    def test_opencode_lists_models_with_variants_defaults_and_context_limits(self):
        backend = OpenCodeBackend({"base_url": "http://127.0.0.1:4096"})
        response = {"providers": [{"id": "openai", "name": "OpenAI", "models": {"gpt-codex": {"id": "gpt-codex", "name": "GPT Codex", "limit": {"context": 200000}, "variants": {"low": {}, "high": {}}}}}], "default": {"openai": "gpt-codex"}}
        with patch.object(backend, "_ensure_server"), patch.object(backend, "_request", return_value=response):
            catalog = backend.list_models()
        self.assertEqual(catalog, [{"provider": "openai", "provider_name": "OpenAI", "model": "gpt-codex", "display_name": "GPT Codex", "is_default": True, "context_window": 200000, "reasoning_efforts": ["low", "high"]}])
        self.assertEqual(backend._context_window({"providerID": "openai", "modelID": "gpt-codex"}), 200000)

    def test_opencode_turn_uses_selected_reasoning_variant(self):
        backend = OpenCodeBackend({"provider": "openai", "model": "gpt-codex", "reasoning_effort": "high", "model_catalog": [{"provider": "openai", "model": "gpt-codex", "context_window": 200000}]})
        captured = {}
        backend._ensure_session = lambda *_args: None
        backend._request = lambda method, path, **kwargs: captured.update(kwargs.get("json", {})) or {"info": {"providerID": "openai", "modelID": "gpt-codex", "tokens": {"input": 2, "output": 1, "reasoning": 0, "cache": {"read": 0, "write": 0}}}, "parts": [{"id": "part-final", "type": "text", "text": "done"}]}
        with patch("shared.remote_llm.opencode_backend.requests.get", side_effect=RuntimeError("no event stream")):
            answer = backend.run_turn("go", system_prompt="system", tools=[], images=[], on_event=lambda _event: None, call_tool=lambda *_args: {}, should_stop=lambda: False)
        self.assertEqual(answer, "done")
        self.assertEqual(captured["variant"], "high")
        self.assertEqual(captured["model"], {"providerID": "openai", "modelID": "gpt-codex"})

    def test_opencode_mcp_proxy_preserves_schema_and_callback(self):
        calls = []
        schema = {"type": "object", "properties": {"prompt": {"type": "string"}, "steps": {"type": "integer"}}, "required": ["prompt"]}
        server = build_tool_proxy([{"type": "function", "function": {"name": "wangp_generate", "description": "Generate", "parameters": schema}}], lambda name, arguments: calls.append((name, arguments)) or {"status": "done"})
        registered = server._tool_manager.get_tool("wangp_generate")
        self.assertEqual(registered.parameters, schema)
        result = asyncio.run(server._tool_manager.call_tool("wangp_generate", {"prompt": "sunset", "steps": 8}))
        self.assertEqual(result, {"status": "done"})
        self.assertEqual(calls, [("wangp_generate", {"prompt": "sunset", "steps": 8})])
        calls.clear()
        asyncio.run(server._tool_manager.call_tool("wangp_generate", {"prompt": "sunset"}))
        self.assertEqual(calls, [("wangp_generate", {"prompt": "sunset"})])  # a parameter left out does not reach WanGP as null
        calls.clear()
        asyncio.run(server._tool_manager.call_tool("wangp_generate", {"prompt": "sunset", "steps": None}))
        self.assertEqual(calls, [("wangp_generate", {"prompt": "sunset", "steps": None})])  # an explicit null does

    def test_temporary_image_paths_are_removed(self):
        with temporary_image_paths([Image.new("RGB", (4, 4), "red")]) as paths:
            self.assertEqual(len(paths), 1)
            created = Path(paths[0])
            self.assertTrue(created.is_file())
        self.assertFalse(created.exists())

    def test_session_clear_closes_external_backends(self):
        session = AssistantSessionState()
        backend = _ClosableBackend()
        session.remote_backends["test"] = backend
        clear_assistant_session(session)
        self.assertTrue(backend.closed)
        self.assertEqual(session.remote_backends, {})

    def test_remote_deepy_uses_existing_turn_and_prime_tool_path(self):
        session = AssistantSessionState()
        user_id, _event = assistant_chat.add_user_message(session, "make a sunset")
        begin_assistant_turn(session, user_id, "make a sunset")
        sent = []
        backend = _FakeBackend()
        config = {"llm_engines": {"deepy": "codex"}}
        with patch("shared.remote_llm.deepy_runner.create_backend", return_value=backend):
            run_remote_deepy_turn(config, session, "make a sunset", "system", _FakeToolbox(), lambda command, payload=None: sent.append((command, payload)))
        self.assertEqual([message["role"] for message in session.messages], ["user", "assistant", "tool", "assistant"])
        self.assertEqual(session.messages[-1]["content"], "I will generate. Done.")
        self.assertTrue(backend.system_prompt.endswith("Use direct long-running tool calls."))
        self.assertIsNone(session.current_turn)
        self.assertTrue(any(command == "chat_output" for command, _payload in sent))
        assistant_record = next(record for record in session.chat_transcript if record["role"] == "assistant")
        self.assertEqual([block["type"] for block in assistant_record["blocks"]], ["markdown", "tool", "markdown"])
        self.assertEqual([block["text"] for block in assistant_record["blocks"] if block["type"] == "markdown"], ["I will generate.", "Done."])
        status_events = [json.loads(payload)["event"].get("status") for command, payload in sent if command == "chat_output" and json.loads(payload)["event"]["type"] == "status"]
        self.assertIn({"visible": True, "kind": "tool", "text": "Generate - sunset..."}, status_events)

    def test_remote_engine_receives_the_turns_it_has_not_seen(self):
        class Recording(_ClosableBackend):
            def __init__(self):
                super().__init__()
                self.prompts = []

            def run_turn(self, text, **_kwargs):
                self.prompts.append(text)
                return "answer"

            def interrupt(self):
                pass

        session = AssistantSessionState()
        codex, claude = Recording(), Recording()

        def turn(engine, text):
            user_id, _event = assistant_chat.add_user_message(session, text)
            begin_assistant_turn(session, user_id, text)
            run_remote_deepy_turn({"llm_engines": {"deepy": engine}}, session, text, "system", _FakeToolbox(), lambda *_args: None)

        with patch("shared.remote_llm.deepy_runner.create_backend", side_effect=[codex, claude]):
            turn("codex", "first")
            turn("codex", "second")
            turn("claude", "third")
            turn("codex", "fourth")
        self.assertNotIn("wangp_conversation_history", codex.prompts[0])
        self.assertNotIn("wangp_conversation_history", codex.prompts[1])
        history = claude.prompts[0].split("</wangp_conversation_history>")[0]
        self.assertIn("first", history)
        self.assertIn("second", history)
        history = codex.prompts[2].split("</wangp_conversation_history>")[0]
        self.assertIn("third", history)  # what the other engine did, and nothing it already holds
        self.assertNotIn("second", history)

    def test_restored_history_keeps_the_latest_messages_whole(self):
        results = [{"role": "tool", "tool_call_id": str(number), "content": f"result {number} " + "x" * 100_000} for number in range(10)]
        messages = [{"role": "user", "content": "first words"}, *results, {"role": "assistant", "content": "latest words"}]
        history, cut = _restored_history(messages)
        self.assertTrue(cut)
        self.assertLessEqual(len(json.dumps(history, ensure_ascii=False)), REMOTE_HISTORY_MAX_CHARS)
        self.assertEqual(history[0]["content"], "first words")
        self.assertEqual(history[-1]["content"], "latest words")
        self.assertIn("characters shortened", history[1]["content"])
        self.assertTrue(history[1]["content"].startswith("result 0"))
        self.assertEqual(history[-2]["content"], results[-1]["content"])
        self.assertEqual(_restored_history(messages[:2]), ([{"role": "user", "content": "first words"}, {"role": "tool", "tool_call_id": "0", "content": results[0]["content"]}], False))
        words = [{"role": "user", "content": f"{number} " + "y" * 100_000} for number in range(10)]  # nothing left to shorten: the oldest go
        history, cut = _restored_history(words)
        self.assertTrue(cut)
        self.assertLessEqual(len(json.dumps(history, ensure_ascii=False)), REMOTE_HISTORY_MAX_CHARS)
        self.assertEqual(history[-1]["content"], words[-1]["content"])

    def test_remote_deepy_keeps_codex_alive_while_browser_login_completes(self):
        session = AssistantSessionState()
        user_id, _event = assistant_chat.add_user_message(session, "make a sunset")
        begin_assistant_turn(session, user_id, "make a sunset")
        backend = _AuthBackend()
        with patch("shared.remote_llm.deepy_runner.create_backend", return_value=backend), self.assertRaises(CodexAuthenticationRequired):
            run_remote_deepy_turn({"llm_engines": {"deepy": "codex"}}, session, "make a sunset", "system", _FakeToolbox(), lambda *_args: None)
        self.assertIs(session.remote_backends["codex"], backend)
        self.assertFalse(backend.closed)

    def test_remote_turn_drains_a_render_before_restoring_the_conversation(self):
        for failed in (False, True):
            with self.subTest(failed=failed):
                session = AssistantSessionState()
                user_id, _ = assistant_chat.add_user_message(session, "Render two shots")
                begin_assistant_turn(session, user_id, "Render two shots")
                between_shots = threading.Event()
                observed = []

                def render(name, arguments):
                    # A paused render needs both the stop signal and the session lock to exit.
                    with session.turn_lock:
                        session.pause_requested = True
                        session.pause_resume_event.clear()
                    between_shots.set()
                    session.pause_resume_event.wait(3)
                    with session.turn_lock:
                        observed.append((session.interrupt_requested, session.current_turn is not None))
                    return {"status": "interrupted", "tool": name}

                def end_early(prompt, *, call_tool, **kwargs):
                    worker = threading.Thread(target=call_tool, args=("wangp_generate", {}), daemon=True)
                    worker.start()
                    self.addCleanup(worker.join, 3)
                    self.assertTrue(between_shots.wait(3))
                    if failed:
                        raise ConnectionError("provider disconnected")
                    return ""

                backend, toolbox = Mock(), _FakeToolbox()
                backend.run_turn.side_effect = end_early
                toolbox.call = Mock(side_effect=render)
                session.remote_backends["opencode"] = backend
                args = ({"llm_engines": {"deepy": "opencode"}}, session, "Render two shots", "system", toolbox, lambda *_: None)
                if failed:
                    with self.assertRaisesRegex(ConnectionError, "provider disconnected"):
                        run_remote_deepy_turn(*args)
                else:
                    run_remote_deepy_turn(*args)
                self.assertEqual(observed, [(True, True)])
                self.assertIsNone(session.current_turn)
                self.assertIn("ended its turn", session.interruption_notice)
                backend.interrupt.assert_not_called()

                # An old provider callback must stay closed after the next turn starts.
                session.interrupt_requested = False
                begin_assistant_turn(session, user_id, "Next turn")
                late = backend.run_turn.call_args.kwargs["call_tool"]("wangp_generate", {})
                self.assertTrue(late["cancelled"])
                toolbox.call.assert_called_once()

    def test_remote_deepy_streams_reasoning_and_progress_status(self):
        session = AssistantSessionState()
        user_id, _event = assistant_chat.add_user_message(session, "compare products")
        begin_assistant_turn(session, user_id, "compare products")
        sent = []
        backend = _ReasoningBackend()
        with patch("shared.remote_llm.deepy_runner.create_backend", return_value=backend):
            run_remote_deepy_turn({"llm_engines": {"deepy": "codex"}}, session, "compare products", "system", _FakeToolbox(), lambda command, payload=None: sent.append((command, payload)))
        assistant_record = next(record for record in session.chat_transcript if record["role"] == "assistant")
        reasoning_blocks = [block for block in assistant_record["blocks"] if block["type"] == "reasoning"]
        self.assertEqual([block["text"] for block in reasoning_blocks], ["Check both products.\n\nCompare the results."])
        status_events = [json.loads(payload)["event"].get("status") for command, payload in sent if command == "chat_output" and json.loads(payload)["event"]["type"] == "status"]
        self.assertIn({"visible": True, "kind": "loading", "text": "Waiting for Codex..."}, status_events)
        self.assertIn({"visible": True, "kind": "thinking", "text": "Codex is thinking..."}, status_events)
        self.assertIn({"visible": True, "kind": "status", "text": "Codex is responding..."}, status_events)
        self.assertIsNone(status_events[-1])

    def test_remote_deepy_separates_hidden_reasoning_from_tool_lifecycle(self):
        session = AssistantSessionState()
        user_id, _event = assistant_chat.add_user_message(session, "make a sunset")
        begin_assistant_turn(session, user_id, "make a sunset")
        sent = []
        with patch("shared.remote_llm.deepy_runner.create_backend", return_value=_HiddenReasoningToolBackend()):
            run_remote_deepy_turn({"llm_engines": {"deepy": "claude"}}, session, "make a sunset", "system", _FakeToolbox(), lambda command, payload=None: sent.append((command, payload)))
        assistant_record = next(record for record in session.chat_transcript if record["role"] == "assistant")
        tool_record = next(block for block in assistant_record["blocks"] if block["type"] == "tool")
        self.assertEqual(tool_record["status"], "done")
        self.assertFalse(any(block["type"] == "reasoning" for block in assistant_record["blocks"]))
        statuses = [json.loads(payload)["event"].get("status") for command, payload in sent if command == "chat_output" and json.loads(payload)["event"]["type"] == "status"]
        labels = [status["text"] if status else None for status in statuses]
        self.assertLess(labels.index("Claude Code is thinking..."), labels.index("Claude Code is preparing a tool request..."))
        tool_index = labels.index("Generate - sunset...")
        waiting_index = labels.index("Waiting for Claude Code...", tool_index)
        self.assertLess(tool_index, waiting_index)
        self.assertEqual(labels[waiting_index + 1], "Claude Code is thinking...")

    def test_remote_deepy_displays_rejected_tool_with_raw_input(self):
        session = AssistantSessionState()
        user_id, _event = assistant_chat.add_user_message(session, "write the chapter")
        begin_assistant_turn(session, user_id, "write the chapter")
        with patch("shared.remote_llm.deepy_runner.create_backend", return_value=_RejectedToolBackend()):
            run_remote_deepy_turn({"llm_engines": {"deepy": "claude"}}, session, "write the chapter", "system", _FakeToolbox(), lambda _command, _payload=None: None)
        assistant_record = next(record for record in session.chat_transcript if record["role"] == "assistant")
        tool_record = next(block for block in assistant_record["blocks"] if block["type"] == "tool")
        self.assertEqual(tool_record["name"], "wangp_artifact")
        self.assertEqual(tool_record["status"], "error")
        self.assertEqual(tool_record["arguments"]["__unparsedToolInput"]["raw"], "{\"action\":\"commit")
        self.assertIn("JSON parse failed", tool_record["result"]["error"])

    def test_remote_deepy_renders_commentary_inside_deepy_turn(self):
        session = AssistantSessionState()
        user_id, _event = assistant_chat.add_user_message(session, "check settings")
        begin_assistant_turn(session, user_id, "check settings")
        backend = _CommentaryBackend()
        with patch("shared.remote_llm.deepy_runner.create_backend", return_value=backend):
            sent = []
            run_remote_deepy_turn({"llm_engines": {"deepy": "codex"}}, session, "check settings", "system", _FakeToolbox(), lambda command, payload=None: sent.append((command, payload)))
        assistant_records = [record for record in session.chat_transcript if record["role"] == "assistant"]
        self.assertEqual(len(assistant_records), 1)
        self.assertEqual(assistant_records[0]["author"], "Deepy")
        self.assertEqual([block["text"] for block in assistant_records[0]["blocks"] if block["type"] == "markdown"], ["Checking the available settings.", "The settings are compatible.", "Everything is ready."])
        status_events = [json.loads(payload)["event"].get("status") for command, payload in sent if command == "chat_output" and json.loads(payload)["event"]["type"] == "status"]
        self.assertIn({"visible": True, "kind": "thinking", "text": "Codex is thinking..."}, status_events)
        self.assertNotIn({"visible": True, "kind": "status", "text": "Codex is reporting progress..."}, status_events)
        self.assertEqual(session.messages[-1], {"role": "assistant", "content": "Everything is ready."})

    def test_remote_deepy_removes_repeated_live_commentary_block(self):
        session = AssistantSessionState()
        user_id, _event = assistant_chat.add_user_message(session, "check settings")
        begin_assistant_turn(session, user_id, "check settings")
        with patch("shared.remote_llm.deepy_runner.create_backend", return_value=_RemovingCommentaryBackend()):
            run_remote_deepy_turn({"llm_engines": {"deepy": "claude"}}, session, "check settings", "system", _FakeToolbox(), lambda *_args: None)
        assistant_record = next(record for record in session.chat_transcript if record["role"] == "assistant")
        self.assertEqual([block["text"] for block in assistant_record["blocks"] if block["type"] == "markdown"], ["I'll gather the default settings first.", "Done."])

    def test_remote_deepy_uses_claude_statuses_and_promotes_streamed_final_text(self):
        session = AssistantSessionState()
        user_id, _event = assistant_chat.add_user_message(session, "answer")
        begin_assistant_turn(session, user_id, "answer")
        sent = []
        with patch("shared.remote_llm.deepy_runner.create_backend", return_value=_PromotedCommentaryBackend()):
            run_remote_deepy_turn({"llm_engines": {"deepy": "claude"}}, session, "answer", "system", _FakeToolbox(), lambda command, payload=None: sent.append((command, payload)))
        assistant_record = next(record for record in session.chat_transcript if record["role"] == "assistant")
        self.assertEqual([block["text"] for block in assistant_record["blocks"] if block["type"] == "markdown"], ["Final answer."])
        self.assertEqual(session.messages[-1], {"role": "assistant", "content": "Final answer."})
        status_events = [json.loads(payload)["event"].get("status") for command, payload in sent if command == "chat_output" and json.loads(payload)["event"]["type"] == "status"]
        self.assertIn({"visible": True, "kind": "loading", "text": "Waiting for Claude Code..."}, status_events)
        self.assertIn({"visible": True, "kind": "thinking", "text": "Claude Code is thinking..."}, status_events)
        self.assertIn({"visible": True, "kind": "status", "text": "Claude Code is responding..."}, status_events)

    def test_remote_deepy_renders_compaction_marker_and_usage_footer(self):
        session = AssistantSessionState()
        previous_stats = {"text": "previous turn", "tooltip": "previous usage"}
        session.remote_usage_stats = previous_stats
        user_id, _event = assistant_chat.add_user_message(session, "continue")
        begin_assistant_turn(session, user_id, "continue")
        sent = []
        with patch("shared.remote_llm.deepy_runner.create_backend", return_value=_UsageCompactionBackend()), patch("shared.remote_llm.usage.SHOW_REMOTE_LLM_CUMULATIVE_USAGE", True):
            run_remote_deepy_turn({"llm_engines": {"deepy": "claude"}}, session, "continue", "system", _FakeToolbox(), lambda command, payload=None: sent.append((command, payload)))
        assistant_record = next(record for record in session.chat_transcript if record["role"] == "assistant")
        context_blocks = [block for block in assistant_record["blocks"] if block["type"] == "context_summary"]
        self.assertEqual(len(context_blocks), 1)
        self.assertIn("not exposed to WanGP", context_blocks[0]["text"])
        self.assertIn("50,000 tokens", context_blocks[0]["text"])
        self.assertEqual(session.remote_usage_stats["text"], "in 1,200 (1,000 cached) | out 80 | turn 1,280 tk")
        stats_events = [json.loads(payload)["event"]["stats"] for command, payload in sent if command == "chat_output" and json.loads(payload)["event"]["type"] == "stats"]
        self.assertEqual(stats_events, [session.remote_usage_stats])
        self.assertEqual(json.loads(assistant_chat.build_sync_event(session))["event"]["stats"], session.remote_usage_stats)


if __name__ == "__main__":
    unittest.main()
