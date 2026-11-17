# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import json
import threading
import time
import unittest
from unittest.mock import patch

from shared.remote_llm.opencode_backend import OpenCodeBackend, _opencode_tools


class OpenCodeBackendTests(unittest.TestCase):
    def test_tools_register_under_names_that_opencode_prefixes_back(self):
        tools = [{"type": "function", "function": {"name": name, "description": name, "parameters": {}}} for name in ("wangp_generate", "mcp_resource")]
        exposed, originals = _opencode_tools(tools)
        self.assertEqual([tool["function"]["name"] for tool in exposed], ["generate", "mcp_resource"])
        self.assertEqual(originals, {"generate": "wangp_generate", "mcp_resource": "mcp_resource"})
        tools.append({"type": "function", "function": {"name": "generate", "description": "A plugin's tool", "parameters": {}}})
        exposed, originals = _opencode_tools(tools)
        self.assertEqual(originals, {"generate": "wangp_generate", "mcp_resource": "mcp_resource", "plugin_generate": "generate"})

    def test_distant_provider_retry_ends_the_turn_with_its_message(self):
        backend = OpenCodeBackend({"model_catalog": [{"provider": "opencode", "model": "free"}]})
        backend._session_id = "session-1"
        backend._ensure_session = lambda *_args: None
        aborted = threading.Event()
        retries = []

        def request(_method, path, **_kwargs):
            if path.endswith("/message"):
                aborted.wait(timeout=5)
                return {}
            aborted.set()
            return True

        class Events:
            encoding = ""

            def raise_for_status(self):
                pass

            def close(self):
                pass

            def iter_lines(self, *, decode_unicode=False):
                yield "data: " + json.dumps({"type": "session.status", "properties": {"sessionID": "session-1", "status": {"type": "retry", "attempt": 1, "message": "Free usage exceeded", "next": (time.time() + 3600) * 1000}}})

        backend._request = request
        with patch("shared.remote_llm.opencode_backend.requests.get", return_value=Events()), self.assertRaisesRegex(RuntimeError, "Free usage exceeded"):
            backend.run_turn("continue", system_prompt="system", tools=[], images=[], on_event=lambda event: retries.append(event.text) if event.kind == "provider_retry" else None, call_tool=lambda *_args: {}, should_stop=lambda: False)
        self.assertTrue(aborted.is_set())
        self.assertEqual(retries, ["Free usage exceeded"])

    def test_opencode_2_turn_follows_the_session_events(self):
        backend = OpenCodeBackend({"provider": "fake", "model": "scripted", "model_catalog": [{"provider": "fake", "model": "scripted", "context_window": 32000}]})
        backend._api, backend._session_id = 2, "ses_1"
        backend._ensure_session = lambda *_args: None
        calls = []

        def request(method, path, **kwargs):
            calls.append((method, path, kwargs.get("json")))
            return {}

        def event(kind, **data):
            return "data: " + json.dumps({"type": kind, "data": {"sessionID": "ses_1", **data}})

        stream = [event("server.connected"), event("session.step.started", assistantMessageID="msg_a", model={"id": "scripted", "providerID": "fake"}),
                  event("session.text.delta", assistantMessageID="msg_a", ordinal=0, delta="Calling a tool."), event("session.step.ended", assistantMessageID="msg_a", finish="tool-calls", tokens={"input": 90, "output": 5, "reasoning": 0, "cache": {"read": 0, "write": 0}}),
                  event("session.step.started", assistantMessageID="msg_b", model={"id": "scripted", "providerID": "fake"}), event("session.text.delta", assistantMessageID="msg_b", ordinal=0, delta="Done: "),
                  event("session.text.ended", assistantMessageID="msg_b", ordinal=0, text="Done: café."), event("session.text.delta", sessionID="ses_other", assistantMessageID="msg_x", ordinal=0, delta="another session"),
                  event("session.step.ended", assistantMessageID="msg_b", finish="stop", tokens={"input": 120, "output": 12, "reasoning": 0, "cache": {"read": 0, "write": 0}}), event("session.execution.succeeded")]

        class Events:
            encoding = ""

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def raise_for_status(self):
                pass

            def iter_lines(self, *, chunk_size=512, decode_unicode=False):
                yield from stream

        backend._request = request
        events = []
        with patch("shared.remote_llm.opencode_backend.requests.get", return_value=Events()):
            answer = backend.run_turn("go", system_prompt="system", tools=[], images=[], on_event=events.append, call_tool=lambda *_args: {}, should_stop=lambda: False)
        self.assertEqual(answer, "Done: café.")
        self.assertEqual([(event.kind, event.text) for event in events if event.kind == "commentary_promote"], [("commentary_promote", "Done: café.")])
        self.assertEqual("".join(event.text for event in events if event.kind == "commentary_delta"), "Calling a tool.Done: café.")
        self.assertEqual([event.data["context_window"] for event in events if event.kind == "usage"], [32000, 32000])
        self.assertEqual([path for _method, path, _body in calls], ["/api/experimental/session/ses_1/instructions/entries/wangp", "/api/session/ses_1/prompt"])
        self.assertIn("system", calls[0][2]["value"])
        self.assertEqual(calls[1][2], {"text": "go"})

    def test_sse_is_decoded_as_utf8(self):
        class Response:
            encoding = "ISO-8859-1"

            def iter_lines(self, *, decode_unicode=False):
                payload = "data: I’ll generate café images.".encode("utf-8")
                yield payload.decode(self.encoding) if decode_unicode else payload

        response = Response()
        self.assertEqual(list(OpenCodeBackend._utf8_sse_lines(response)), ["data: I’ll generate café images."])
        self.assertEqual(response.encoding, "utf-8")

    def test_stop_aborts_without_reverting_session_history(self):
        backend = OpenCodeBackend({"model_catalog": [{"provider": "openai", "model": "gpt"}]})
        backend._session_id = "session-1"
        backend._ensure_session = lambda *_args: None
        aborted = threading.Event()
        calls = []

        def request(method, path, **kwargs):
            calls.append((method, path, kwargs.get("json")))
            if path.endswith("/message"):
                aborted.wait(timeout=2)
                return {}
            if path.endswith("/abort"):
                aborted.set()
                return True
            raise AssertionError(f"Unexpected OpenCode request: {method} {path}")

        backend._request = request
        with patch("shared.remote_llm.opencode_backend.requests.get", side_effect=RuntimeError("no event stream")):
            backend.run_turn("continue", system_prompt="system", tools=[], images=[], on_event=lambda _event: None, call_tool=lambda *_args: {}, should_stop=lambda: True)

        message_body = next(body for _method, path, body in calls if path.endswith("/message"))
        self.assertNotIn("messageID", message_body)
        self.assertTrue(any(path.endswith("/abort") for _method, path, _body in calls))
        self.assertFalse(any(path.endswith("/revert") for _method, path, _body in calls))
        self.assertEqual(backend._session_id, "session-1")


if __name__ == "__main__":
    unittest.main()
