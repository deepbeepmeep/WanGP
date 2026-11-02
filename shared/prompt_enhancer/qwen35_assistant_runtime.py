# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import json
import os
import re
import statistics
import time
from collections import deque
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import torch

from shared.llm_engines.lmgp import SamplingParams
from shared.llm_engines.lmgp.engine.block_manager import BlockManager
from shared.llm_engines.lmgp.layers.kv_formats import overlay_slots as _kv_overlay_slots
from shared.llm_engines.lmgp.engine.sequence import Sequence, SequenceStatus
from shared.prompt_enhancer import qwen35_text
from shared.prompt_enhancer.streaming import ThrottledStreamEmitter


_TRAILING_STOP_RE = re.compile(r"(?:<\|im_end\|>\s*|<\|endoftext\|>\s*|</s>\s*)+$", flags=re.IGNORECASE)
_FUNCTION_TAG_RE = re.compile(r"<function(?:=|\s+name=)([^\s>]+)[^>]*>(.*?)</function>", flags=re.DOTALL | re.IGNORECASE)
_FUNCTION_START_RE = re.compile(r"<function(?:=|\s+name=)([^\s>]+)[^>]*>", flags=re.IGNORECASE)
_PARAM_TAG_RE = re.compile(r"<parameter(?:=|\s+name=)([^\s>]+)[^>]*>(.*?)</parameter>", flags=re.DOTALL | re.IGNORECASE)
_GENERIC_PARAM_TAG_RE = re.compile(r"<([A-Za-z_][A-Za-z0-9_]*)>\s*(.*?)\s*</(?:parameter|\1)>", flags=re.DOTALL | re.IGNORECASE)
_JSON_PARSE_FAILED = object()
_ASSISTANT_PREFILL_CHUNK_TOKENS = 1024
ASSISTANT_THOUGHT_BUDGET_TOKENS = 4096
ASSISTANT_STATEMENT_BUDGET_TOKENS = 4096
ASSISTANT_TOOL_BATCH_BUDGET_TOKENS = 4096
ASSISTANT_ACTION_BUDGET_SMALL_CONTEXT_TOKENS = 16386
ASSISTANT_ACTION_BUDGET_BASE_CONTEXT_TOKENS = 32000
ASSISTANT_ACTION_BUDGET_MEDIUM_CONTEXT_TOKENS = 48000
ASSISTANT_ACTION_BUDGET_LARGE_CONTEXT_TOKENS = 64000
ASSISTANT_ACTION_BUDGET_SMALL_TOKENS = 2048
ASSISTANT_ACTION_BUDGET_MEDIUM_TOKENS = 6144
ASSISTANT_ACTION_BUDGET_LARGE_TOKENS = 8192
DEEPY_TELEMETRY_ENV = "WAN2GP_DEEPY_TELEMETRY"
_DEEPY_TELEMETRY_ENABLED = str(os.environ.get(DEEPY_TELEMETRY_ENV, "0")).strip().lower() in {"1", "true", "yes", "y", "on"}


def assistant_thought_budget_update(budget_tokens: int) -> str:
    return f"""<wangp_runtime_update>
The preceding thought reached its budget of {int(budget_tokens)} tokens. Continue more directly: answer, call a tool, or start a fresh thought only if needed. Reuse established conclusions and avoid repeating exploration.
</wangp_runtime_update>"""


ASSISTANT_THOUGHT_BUDGET_UPDATE = assistant_thought_budget_update(ASSISTANT_THOUGHT_BUDGET_TOKENS)


def assistant_action_budget_tokens(context_window_tokens: int) -> int:
    context_window_tokens = int(context_window_tokens)
    if context_window_tokens <= ASSISTANT_ACTION_BUDGET_SMALL_CONTEXT_TOKENS:
        return ASSISTANT_ACTION_BUDGET_SMALL_TOKENS
    if context_window_tokens < ASSISTANT_ACTION_BUDGET_BASE_CONTEXT_TOKENS:
        context_growth = context_window_tokens - ASSISTANT_ACTION_BUDGET_SMALL_CONTEXT_TOKENS
        budget_growth = ASSISTANT_THOUGHT_BUDGET_TOKENS - ASSISTANT_ACTION_BUDGET_SMALL_TOKENS
        return ASSISTANT_ACTION_BUDGET_SMALL_TOKENS + budget_growth * context_growth // (ASSISTANT_ACTION_BUDGET_BASE_CONTEXT_TOKENS - ASSISTANT_ACTION_BUDGET_SMALL_CONTEXT_TOKENS)
    if context_window_tokens >= ASSISTANT_ACTION_BUDGET_LARGE_CONTEXT_TOKENS:
        return ASSISTANT_ACTION_BUDGET_LARGE_TOKENS * context_window_tokens // ASSISTANT_ACTION_BUDGET_LARGE_CONTEXT_TOKENS
    if context_window_tokens >= ASSISTANT_ACTION_BUDGET_MEDIUM_CONTEXT_TOKENS:
        return ASSISTANT_ACTION_BUDGET_MEDIUM_TOKENS
    return ASSISTANT_THOUGHT_BUDGET_TOKENS


def _tool_call_markers(text: str) -> list[tuple[int, int, bool]]:
    source = str(text or "")
    markers = []
    inside_tool = False
    json_payload = False
    in_string = False
    escaped = False
    index = 0
    while index < len(source):
        char = source[index]
        if inside_tool and in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if inside_tool and json_payload and char == '"':
            in_string = True
            index += 1
            continue
        if char == "<":
            if inside_tool and not json_payload:
                # Tagged parameter values are plain text, including quotes and literal tool markers.
                parameter = _PARAM_TAG_RE.match(source, index)
                if parameter is not None:
                    index = parameter.end()
                    continue
            match = re.match(r"<\s*(/?)\s*tool_call\s*>", source[index:], flags=re.IGNORECASE)
            if match is not None:
                closing = bool(match.group(1))
                markers.append((index, index + match.end(), closing))
                inside_tool = not closing
                json_payload = inside_tool and source[index + match.end():].lstrip().startswith(("{", "["))
                in_string = False
                escaped = False
                index += match.end()
                continue
        index += 1
    return markers


def _tool_call_spans(text: str) -> list[tuple[int, int, str]]:
    source = str(text or "")
    spans = []
    open_marker = None
    for start, end, closing in _tool_call_markers(source):
        if not closing:
            if open_marker is not None:
                return []
            open_marker = (start, end)
            continue
        if open_marker is None:
            return []
        open_start, payload_start = open_marker
        spans.append((open_start, end, source[payload_start:start].strip()))
        open_marker = None
    return [] if open_marker is not None else spans


@dataclass(slots=True)
class AssistantDecodeResult:
    raw_text: str
    stop_reason: str
    token_count: int
    stop_token_id: int | None = None
    phase: str = ""


@dataclass(slots=True)
class AssistantActionState:
    phase: str
    limit: int
    generated_tokens: int = 0

    @property
    def remaining_tokens(self) -> int:
        return max(0, int(self.limit) - int(self.generated_tokens))


def render_assistant_messages(tokenizer, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, add_generation_prompt: bool, thinking_enabled: bool) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=bool(add_generation_prompt),
        tokenize=True,
        enable_thinking=bool(thinking_enabled),
    )
    if torch.is_tensor(rendered):
        rendered = rendered.tolist()
    return [int(token_id) for token_id in rendered]


def render_text_user_turn_suffix(tokenizer, user_content: str, thinking_enabled: bool) -> list[int]:
    user_content = str(user_content or "").strip()
    if len(user_content) == 0:
        return []
    suffix = f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n"
    suffix += "<think>\n" if bool(thinking_enabled) else "<think>\n\n</think>\n\n"
    token_ids = tokenizer.encode(suffix, add_special_tokens=False)
    if torch.is_tensor(token_ids):
        token_ids = token_ids.tolist()
    return [int(token_id) for token_id in token_ids]


def render_tool_turn_suffix(tokenizer, tool_contents: list[str], thinking_enabled: bool) -> list[int]:
    normalized_contents = [str(content or "").strip() for content in list(tool_contents or []) if len(str(content or "").strip()) > 0]
    if len(normalized_contents) == 0:
        return []
    suffix = "<|im_end|>\n<|im_start|>user"
    for tool_content in normalized_contents:
        suffix += f"\n<tool_response>\n{tool_content}\n</tool_response>"
    suffix += "<|im_end|>\n<|im_start|>assistant\n"
    suffix += "<think>\n" if bool(thinking_enabled) else "<think>\n\n</think>\n\n"
    token_ids = tokenizer.encode(suffix, add_special_tokens=False)
    if torch.is_tensor(token_ids):
        token_ids = token_ids.tolist()
    return [int(token_id) for token_id in token_ids]


def render_assistant_text_suffix(tokenizer, assistant_content: str, thinking_enabled: bool, prompt_open: bool) -> list[int]:
    assistant_content = str(assistant_content or "").strip()
    if len(assistant_content) == 0:
        return []
    if prompt_open:
        if bool(thinking_enabled) and assistant_content.startswith("<think>"):
            suffix = assistant_content[len("<think>") :]
            suffix = suffix[2:] if suffix.startswith("\r\n") else suffix[1:] if suffix.startswith("\n") else suffix
            suffix += "<|im_end|>\n"
        else:
            suffix = ("</think>\n\n" if bool(thinking_enabled) else "") + assistant_content + "<|im_end|>\n"
    else:
        suffix = f"<|im_start|>assistant\n<think>\n\n</think>\n\n{assistant_content}<|im_end|>\n"
    token_ids = tokenizer.encode(suffix, add_special_tokens=False)
    if torch.is_tensor(token_ids):
        token_ids = token_ids.tolist()
    return [int(token_id) for token_id in token_ids]


def strip_tool_blocks(raw_text: str) -> str:
    text = str(raw_text or "")
    spans = _tool_call_spans(text)
    if len(spans) == 0:
        return text.strip()
    parts = []
    cursor = 0
    for start, end, _payload in spans:
        parts.append(text[cursor:start])
        cursor = end
    parts.append(text[cursor:])
    return "\n".join(parts).strip()


def strip_trailing_stop_markup(raw_text: str, *, keep_trailing_newlines: bool = False) -> str:
    text = _TRAILING_STOP_RE.sub("", str(raw_text or ""))
    return text.rstrip(" \t") if keep_trailing_newlines else text.rstrip()


def _clean_tag_name(name: str) -> str:
    name = str(name or "").strip()
    if len(name) >= 2 and name[0] in ("'", '"') and name[-1] == name[0]:
        return name[1:-1].strip()
    return name


def _load_json_with_missing_closers(source: str):
    text = str(source or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as original_error:
        if not text or text[0] not in "{[" or original_error.pos != len(text):
            return _JSON_PARSE_FAILED
    expected_closers = []
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            expected_closers.append("}")
        elif char == "[":
            expected_closers.append("]")
        elif char in "}]":
            if not expected_closers or expected_closers.pop() != char:
                return _JSON_PARSE_FAILED
    if in_string or not expected_closers or len(expected_closers) > 3:
        return _JSON_PARSE_FAILED
    try:
        return json.loads(text + "".join(reversed(expected_closers)))
    except json.JSONDecodeError:
        return _JSON_PARSE_FAILED


def _parse_tagged_tool_call(payload: str, allow_incomplete_function: bool = False, tool_parameters: dict[str, set[str]] | None = None, *, preserve_unknown_parameters: bool = False) -> dict[str, Any] | None:
    function_match = _FUNCTION_TAG_RE.search(str(payload or ""))
    function_body = ""
    matched_closed_function = function_match is not None
    if function_match is not None:
        name = _clean_tag_name(function_match.group(1))
        function_body = function_match.group(2)
    else:
        function_start_match = _FUNCTION_START_RE.search(str(payload or ""))
        if function_start_match is None:
            return None
        name = _clean_tag_name(function_start_match.group(1))
        function_body = str(payload or "")[function_start_match.end():]
    if len(name) == 0:
        return None
    allowed_parameters = None if tool_parameters is None else tool_parameters.get(name, set())
    arguments = {}
    for match in _PARAM_TAG_RE.finditer(function_body):
        param_name, param_value = match.groups()
        clean_name = _clean_tag_name(param_name)
        clean_value = str(param_value or "").strip()
        if len(clean_name) == 0 or not preserve_unknown_parameters and allowed_parameters is not None and clean_name not in allowed_parameters:
            continue
        parsed_value = _load_json_with_missing_closers(clean_value)
        arguments[clean_name] = clean_value if parsed_value is _JSON_PARSE_FAILED else parsed_value
    if allowed_parameters is not None:
        generic_body = _PARAM_TAG_RE.sub("", function_body)
        for param_name, param_value in _GENERIC_PARAM_TAG_RE.findall(generic_body):
            clean_name = _clean_tag_name(param_name)
            clean_value = str(param_value or "").strip()
            if not clean_name or (not preserve_unknown_parameters and clean_name not in allowed_parameters) or clean_name in arguments:
                continue
            parsed_value = _load_json_with_missing_closers(clean_value)
            arguments[clean_name] = clean_value if parsed_value is _JSON_PARSE_FAILED else parsed_value
    if not matched_closed_function and (not allow_incomplete_function or len(arguments) == 0):
        return None
    return {"name": name, "arguments": arguments}


def _normalize_tool_call_dict(parsed: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(parsed, dict):
        return None
    name = str(parsed.get("name", "")).strip()
    arguments = parsed.get("arguments", {})
    if not isinstance(arguments, dict):
        return None
    if len(name) == 0:
        return None
    return {"name": name, "arguments": arguments}


RUNAWAY_MIN_REPEATS, RUNAWAY_MIN_CHARS, RUNAWAY_MAX_PERIOD = 40, 3000, 400  # a tool call ending in this much of one fragment has stopped progressing


def repeated_tail(text: str) -> tuple[str, int] | None:
    """The fragment the end of a text keeps repeating and its count, once the repetition is long enough to be a runaway generation."""
    text = text[-RUNAWAY_MIN_REPEATS * RUNAWAY_MAX_PERIOD:]
    for period in range(2, RUNAWAY_MAX_PERIOD + 1):
        unit, count = text[-period:], 1
        while text[-(count + 1) * period:-count * period] == unit:
            count += 1
        if count >= RUNAWAY_MIN_REPEATS and count * period >= RUNAWAY_MIN_CHARS:
            return unit, count
    return None


def validate_tool_call_structure(raw_text: str) -> str:
    text = str(raw_text or "")
    markers = _tool_call_markers(text)
    if len(markers) == 0:
        return ""
    depth = 0
    for _start, _end, closing in markers:
        if closing:
            if depth == 0:
                return "Tool call markup contains an unmatched closing tag."
            depth -= 1
        else:
            if depth > 0:
                return "Tool call markup contains a nested tool call."
            depth += 1
    if depth != 0:
        return "Tool call markup is incomplete."
    spans = _tool_call_spans(text)
    if len(spans) != sum(not closing for _start, _end, closing in markers):
        return "Tool call markup is malformed."
    for _start, _end, payload in spans:
        if len(payload) == 0:
            return "Tool call payload is empty."
        parsed = _load_json_with_missing_closers(payload)
        if parsed is _JSON_PARSE_FAILED:
            parsed = _parse_tagged_tool_call(payload)
        if _normalize_tool_call_dict(parsed) is None:
            return "Tool call payload must contain a name and an arguments object."
    return ""


def extract_incomplete_tool_name(raw_text: str) -> str:
    text = str(raw_text or "")
    open_markers = [(start, end) for start, end, closing in _tool_call_markers(text) if not closing]
    candidate = text if len(open_markers) == 0 else text[open_markers[-1][1]:]
    json_name = re.search(r"[\"']name[\"']\s*:\s*[\"']([^\"']+)", candidate, flags=re.IGNORECASE)
    if json_name is not None:
        return _clean_tag_name(json_name.group(1))
    function_start = _FUNCTION_START_RE.search(candidate)
    return "" if function_start is None else _clean_tag_name(function_start.group(1))


def _decode_completed_object_members(source: str, open_brace_index: int) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    values = {}
    index = int(open_brace_index) + 1
    while index < len(source):
        while index < len(source) and (source[index].isspace() or source[index] == ","):
            index += 1
        if index >= len(source) or source[index] == "}":
            break
        try:
            key, consumed = decoder.raw_decode(source[index:])
        except json.JSONDecodeError:
            break
        if not isinstance(key, str):
            break
        index += consumed
        while index < len(source) and source[index].isspace():
            index += 1
        if index >= len(source) or source[index] != ":":
            break
        index += 1
        while index < len(source) and source[index].isspace():
            index += 1
        try:
            value, consumed = decoder.raw_decode(source[index:])
        except json.JSONDecodeError:
            break
        value_end = index + consumed
        delimiter = value_end
        while delimiter < len(source) and source[delimiter].isspace():
            delimiter += 1
        if delimiter >= len(source) or source[delimiter] not in ",}":
            break
        values[key] = value
        index = delimiter
    return values


def extract_incomplete_tool_arguments(raw_text: str) -> dict[str, Any]:
    """Return fully decoded leading arguments without repairing incomplete JSON."""

    text = str(raw_text or "")
    open_markers = [(start, end) for start, end, closing in _tool_call_markers(text) if not closing]
    candidate = text if len(open_markers) == 0 else text[open_markers[-1][1]:]
    arguments_match = re.search(r'["\']arguments["\']\s*:\s*\{', candidate, flags=re.IGNORECASE)
    if arguments_match is not None:
        return _decode_completed_object_members(candidate, arguments_match.end() - 1)
    function_start = _FUNCTION_START_RE.search(candidate)
    if function_start is None:
        return {}
    open_brace_index = candidate.find("{", function_start.end())
    return {} if open_brace_index < 0 else _decode_completed_object_members(candidate, open_brace_index)


def _extract_bare_json_tool_call(text: str, tool_names=None) -> tuple[dict[str, Any] | None, tuple[int, int] | None]:
    decoder = json.JSONDecoder()
    source_text = str(text or "")
    for start_idx, ch in enumerate(source_text):
        if ch not in "{[":
            continue
        try:
            parsed, end_idx = decoder.raw_decode(source_text[start_idx:])
        except Exception:
            continue
        # An untagged object with a "name" is ordinary data (a drafted answer, a
        # quoted setting) unless it is the whole text or carries its arguments.
        whole_text = start_idx == 0 and len(source_text[end_idx:].strip()) == 0
        if not whole_text and not (isinstance(parsed, dict) and isinstance(parsed.get("arguments"), dict)):
            continue
        tool_call = _normalize_tool_call_dict(parsed)
        if tool_call is None or tool_names is not None and tool_call["name"] not in tool_names:
            continue
        return tool_call, (start_idx, start_idx + end_idx)
    return None, None


def _extract_inline_tool_call(text: str, allow_incomplete_function: bool = False, tool_parameters: dict[str, set[str]] | None = None, *, preserve_unknown_parameters: bool = False) -> tuple[dict[str, Any] | None, tuple[int, int] | None]:
    candidate = strip_trailing_stop_markup(str(text or "")).strip()
    if len(candidate) == 0:
        return None, None
    tagged_tool_call = _parse_tagged_tool_call(candidate, allow_incomplete_function=allow_incomplete_function, tool_parameters=tool_parameters, preserve_unknown_parameters=preserve_unknown_parameters)
    if tagged_tool_call is not None:
        return tagged_tool_call, (0, len(candidate))
    return _extract_bare_json_tool_call(candidate, None if tool_parameters is None else set(tool_parameters))


def extract_tool_calls(raw_text: str, tool_parameters: dict[str, set[str]] | None = None, *, preserve_unknown_parameters: bool = False) -> list[dict[str, Any]]:
    tool_calls = []
    source_text = str(raw_text or "")
    if _tool_call_markers(source_text) and validate_tool_call_structure(source_text):
        return []
    for _start, _end, payload in _tool_call_spans(source_text):
        if len(payload) == 0:
            continue
        parsed = _load_json_with_missing_closers(payload)
        if parsed is _JSON_PARSE_FAILED:
            parsed = _parse_tagged_tool_call(payload, tool_parameters=tool_parameters, preserve_unknown_parameters=preserve_unknown_parameters)
        tool_call = _normalize_tool_call_dict(parsed)
        if tool_call is None:
            continue
        tool_calls.append(tool_call)
    if len(tool_calls) > 0:
        return tool_calls
    # Untagged calls count only in the answer, never inside reasoning.
    _thinking_text, answer_text = qwen35_text._split_generated_text(source_text)
    inline_tool_call, _inline_span = _extract_inline_tool_call(answer_text, allow_incomplete_function=True, tool_parameters=tool_parameters, preserve_unknown_parameters=preserve_unknown_parameters)
    if inline_tool_call is not None:
        tool_calls.append(inline_tool_call)
    return tool_calls


def strip_inline_tool_call_text(raw_text: str) -> str:
    text = strip_trailing_stop_markup(str(raw_text or ""))
    inline_tool_call, inline_span = _extract_inline_tool_call(text)
    if inline_tool_call is None or inline_span is None:
        return text
    start_idx, end_idx = inline_span
    stripped_text = (text[:start_idx] + text[end_idx:]).strip()
    return stripped_text


def has_complete_tool_call(raw_text: str, allow_untagged: bool = True) -> bool:
    text = str(raw_text or "")
    if validate_tool_call_structure(text):
        return False
    for _start, _end, payload in _tool_call_spans(text):
        if len(payload) == 0:
            continue
        parsed = _load_json_with_missing_closers(payload)
        if parsed is _JSON_PARSE_FAILED:
            parsed = _parse_tagged_tool_call(payload)
        if _normalize_tool_call_dict(parsed) is not None:
            return True
    if not allow_untagged:
        return False
    # Untagged calls count only in the answer, never inside reasoning.
    _thinking_text, answer_text = qwen35_text._split_generated_text(text)
    return _extract_inline_tool_call(answer_text)[0] is not None


class Qwen35AssistantRuntime:
    def __init__(self, model, debug_enabled: bool = False):
        self.model = model
        self.tokenizer = getattr(model, "_prompt_enhancer_tokenizer", None)
        if self.tokenizer is None:
            raise RuntimeError("Prompt enhancer tokenizer is missing for assistant runtime.")
        self.debug_enabled = bool(debug_enabled)
        self.telemetry_enabled = self.debug_enabled and _DEEPY_TELEMETRY_ENABLED
        self._runtime_extra_tokens = getattr(model, "_prompt_enhancer_thinking_extra_tokens", 0)
        self._assistant_presence_state = None
        self._repetition_rules = None
        self._nvml_handle = None
        self._nvml_module = None
        self._nvml_failed = False
        self._nvml_clock_event_counters = None
        self._nvml_process_names = {}
        self._last_decode_window = None

    def _log(self, message: str) -> None:
        if self.debug_enabled:
            print(f"[AssistantRuntime] {message}")

    def _gpu_telemetry(self) -> str:
        if self._nvml_failed:
            return "gpu=unavailable"
        try:
            if self._nvml_handle is None:
                import pynvml

                pynvml.nvmlInit()
                self._nvml_module = pynvml
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(torch.cuda.current_device())
            nvml = self._nvml_module
            utilization = nvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
            temperature = nvml.nvmlDeviceGetTemperature(self._nvml_handle, nvml.NVML_TEMPERATURE_GPU)
            sm_clock = nvml.nvmlDeviceGetClockInfo(self._nvml_handle, nvml.NVML_CLOCK_SM)
            max_sm_clock = nvml.nvmlDeviceGetMaxClockInfo(self._nvml_handle, nvml.NVML_CLOCK_SM)
            memory_clock = nvml.nvmlDeviceGetClockInfo(self._nvml_handle, nvml.NVML_CLOCK_MEM)
            pstate = nvml.nvmlDeviceGetPerformanceState(self._nvml_handle)
            power = nvml.nvmlDeviceGetPowerUsage(self._nvml_handle) / 1000.0
            power_limit = nvml.nvmlDeviceGetPowerManagementLimit(self._nvml_handle) / 1000.0
            memory = nvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            throttle_mask = int(nvml.nvmlDeviceGetCurrentClocksThrottleReasons(self._nvml_handle))
            throttle_names = []
            reason_bits = (
                (0x001, "idle"),
                (0x002, "app_clock"),
                (0x004, "power"),
                (0x008, "hw_slowdown"),
                (0x010, "sync_boost"),
                (0x020, "thermal"),
                (0x040, "hw_thermal"),
                (0x080, "power_brake"),
                (0x100, "display_clock"),
                (0x200, "board_limit"),
                (0x400, "reliability"),
            )
            for reason_bit, label in reason_bits:
                if throttle_mask & reason_bit:
                    throttle_names.append(label)
            known_mask = sum(reason_bit for reason_bit, _label in reason_bits)
            unknown_mask = throttle_mask & ~known_mask
            if unknown_mask:
                throttle_names.append(f"unknown:{unknown_mask:#x}")
            throttle = ",".join(throttle_names) if throttle_names else "none"
            clock_event_summary = ""
            try:
                counter_fields = (
                    (int(nvml.NVML_FI_DEV_PERF_POLICY_POWER), "power"),
                    (int(nvml.NVML_FI_DEV_PERF_POLICY_THERMAL), "thermal"),
                    (int(nvml.NVML_FI_DEV_PERF_POLICY_SYNC_BOOST), "sync_boost"),
                    (int(nvml.NVML_FI_DEV_PERF_POLICY_BOARD_LIMIT), "board_limit"),
                    (int(nvml.NVML_FI_DEV_PERF_POLICY_LOW_UTILIZATION), "low_util"),
                    (int(nvml.NVML_FI_DEV_PERF_POLICY_RELIABILITY), "reliability"),
                )
                field_values = nvml.nvmlDeviceGetFieldValues(self._nvml_handle, [field_id for field_id, _label in counter_fields])
                counters = {label: int(field.value.ullVal) for field, (_field_id, label) in zip(field_values, counter_fields) if int(field.nvmlReturn) == int(nvml.NVML_SUCCESS)}
                previous_counters = self._nvml_clock_event_counters
                self._nvml_clock_event_counters = counters
                if previous_counters is not None:
                    deltas = [f"{label}:{max(0, value - previous_counters.get(label, value)) / 1000.0:.0f}" for label, value in counters.items() if value > previous_counters.get(label, value)]
                    clock_event_summary = f" clock_event_ms={','.join(deltas) if deltas else 'none'}"
            except Exception:
                self._nvml_clock_event_counters = None
            process_summary = ""
            try:
                processes = {}
                for query in (nvml.nvmlDeviceGetComputeRunningProcesses, nvml.nvmlDeviceGetGraphicsRunningProcesses):
                    for process in query(self._nvml_handle):
                        processes[int(process.pid)] = process
                process_names = []
                for pid in sorted(processes):
                    name = self._nvml_process_names.get(pid)
                    if name is None:
                        try:
                            name = nvml.nvmlSystemGetProcessName(pid)
                            if isinstance(name, bytes):
                                name = name.decode(errors="replace")
                            name = str(name).replace("\\", "/").rsplit("/", 1)[-1]
                        except Exception:
                            name = "unknown"
                        self._nvml_process_names[pid] = name
                    process_names.append(f"{name}:{pid}{'*' if pid == os.getpid() else ''}")
                process_summary = f" gpu_processes=[{','.join(process_names)}]"
            except Exception:
                pass
            return (
                f"gpu={int(utilization.gpu)}% memctl={int(utilization.memory)}% "
                f"clocks={int(sm_clock)}/{int(max_sm_clock)}sm,{int(memory_clock)}memMHz pstate=P{int(pstate)} "
                f"temp={int(temperature)}C power={power:.1f}/{power_limit:.0f}W "
                f"device_mem={int(memory.used) / (1 << 20):.0f}/{int(memory.total) / (1 << 20):.0f}MiB "
                f"throttle={throttle}({throttle_mask:#x}){clock_event_summary}{process_summary}"
            )
        except Exception as exc:
            self._nvml_failed = True
            return f"gpu=unavailable:{type(exc).__name__}"

    def _start_decode_telemetry(self, seq, label: str) -> dict | None:
        if not self.telemetry_enabled:
            return None
        runner = self._get_live_llm().model_runner
        set_mtp_profile = getattr(runner, "set_mtp_stage_profile_enabled", None)
        if callable(set_mtp_profile):
            set_mtp_profile(bool(getattr(self.model, "_prompt_enhancer_speculative_decoding", False)))
        now = time.perf_counter()
        allocator_retries = int(torch.cuda.memory_stats().get("num_alloc_retries", 0))
        return {
            "label": label,
            "last_at": now,
            "thread_cpu_at": time.thread_time(),
            "process_cpu_at": time.process_time(),
            "allocator_retries": allocator_retries,
            "mtp_profile_samples": 0,
            "emitted": 0,
            "calls": 0,
            "runner_seconds": 0.0,
            "scheduler_seconds": 0.0,
            "postprocess_seconds": 0.0,
            "decode_seconds": 0.0,
            "parse_seconds": 0.0,
            "stream_seconds": 0.0,
            "stream_checks": 0,
            "stream_callbacks": 0,
            "speculative": self._speculative_telemetry(seq),
        }

    def _log_mtp_stage_profile(self, label: str) -> None:
        if not self.telemetry_enabled:
            return
        runner = self._get_live_llm().model_runner
        get_samples = getattr(runner, "mtp_stage_profile_samples", None)
        if not callable(get_samples):
            return
        samples = get_samples()
        if not samples:
            return

        def percentiles(group: str) -> str:
            names = sorted({name for sample in samples for name in sample[group]})
            values = []
            for name in names:
                ordered = sorted(float(sample[group][name]) for sample in samples if name in sample[group])
                p50 = ordered[len(ordered) // 2]
                p90 = ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))]
                values.append(f"{name}={p50:.3f}/{p90:.3f}")
            return " ".join(values)

        outcomes = {}
        distributions = {}
        for sample in samples:
            key = f"a{sample['accepted_count']}e{sample['emitted_count']}c{sample['commit_start']}"
            outcomes[key] = outcomes.get(key, 0) + 1
            for distribution in sample.get("distributions", ()):
                values = distributions.setdefault(distribution["role"], {"vocab": distribution["vocab"], "min_p": [], "top_p": [], "excluded": []})
                values["min_p"].append(distribution["min_p_survivors"])
                values["top_p"].append(distribution["top_p_survivors"])
                values["excluded"].append(distribution["excluded_mass"])
        context_min = min(sample["context_tokens"] for sample in samples)
        context_max = max(sample["context_tokens"] for sample in samples)
        distribution_summary = " ".join(
            f"{role}=v{values['vocab']} min_p:{int(statistics.median(values['min_p']))} top_p:{int(statistics.median(values['top_p']))} excluded:{statistics.median(values['excluded']):.4f}"
            for role, values in distributions.items()
        )
        self._log(
            f"MTP stage telemetry {label}: samples={len(samples)} context={context_min}-{context_max} "
            f"outcomes={outcomes} cpu_ms_p50/p90=[{percentiles('cpu_ms')}] "
            f"cuda_ms_p50/p90=[{percentiles('gpu_ms')}] distributions=[{distribution_summary}]"
        )

    def _update_decode_telemetry(self, state: dict | None, seq, emitted_tokens: int = 0, runner_seconds: float = 0.0, scheduler_seconds: float = 0.0, postprocess_seconds: float = 0.0, decode_seconds: float = 0.0, parse_seconds: float = 0.0, stream_seconds: float = 0.0, stream_checked: bool = False, stream_callback: bool = False, force: bool = False) -> None:
        if state is None:
            return
        state["emitted"] += int(emitted_tokens)
        state["calls"] += int(runner_seconds > 0.0)
        state["runner_seconds"] += float(runner_seconds)
        state["scheduler_seconds"] += float(scheduler_seconds)
        state["postprocess_seconds"] += float(postprocess_seconds)
        state["decode_seconds"] += float(decode_seconds)
        state["parse_seconds"] += float(parse_seconds)
        state["stream_seconds"] += float(stream_seconds)
        state["stream_checks"] += int(stream_checked)
        state["stream_callbacks"] += int(stream_callback)
        now = time.perf_counter()
        elapsed = now - state["last_at"]
        if not force and elapsed < 2.0:
            return
        if state["calls"] == 0 and state["emitted"] == 0:
            return
        current = self._speculative_telemetry(seq)
        speculative = state["speculative"]
        mtp = "mtp=disabled"
        passes = 0
        if current is not None and speculative is not None:
            drafted = current["drafted"] - speculative["drafted"]
            accepted = current["accepted"] - speculative["accepted"]
            passes = current["target_passes"] - speculative["target_passes"]
            drafted_by_position = [current_value - previous_value for current_value, previous_value in zip(current["drafted_by_position"], speculative["drafted_by_position"])]
            accepted_by_position = [current_value - previous_value for current_value, previous_value in zip(current["accepted_by_position"], speculative["accepted_by_position"])]
            position_acceptance = ",".join(f"{accepted_count}/{drafted_count}" for accepted_count, drafted_count in zip(accepted_by_position, drafted_by_position))
            mtp = (
                f"mtp_accept={accepted}/{drafted}({100.0 * accepted / drafted if drafted else 0.0:.1f}%) "
                f"mtp_pos=[{position_acceptance}] tok/pass={state['emitted'] / passes if passes else 0.0:.3f} "
                f"mtp_cache={current['mtp_cache_tokens']} sync_delta={current['sync_delta']:+d} "
                f"pending={current['pending']} draft={current['draft']}"
            )
        throughput = state["emitted"] / elapsed if elapsed > 0.0 else 0.0
        passes_per_second = passes / elapsed if elapsed > 0.0 else 0.0
        tokens_per_pass = state["emitted"] / passes if passes else 0.0
        runner_ms_per_pass = 1000.0 * state["runner_seconds"] / state["calls"] if state["calls"] else 0.0
        accounted = state["runner_seconds"] + state["scheduler_seconds"] + state["postprocess_seconds"] + state["decode_seconds"] + state["parse_seconds"] + state["stream_seconds"]
        host_ms_per_pass = 1000.0 * max(0.0, elapsed - state["runner_seconds"]) / state["calls"] if state["calls"] else 0.0
        thread_cpu_now = time.thread_time()
        process_cpu_now = time.process_time()
        thread_cpu = thread_cpu_now - state["thread_cpu_at"]
        process_cpu = process_cpu_now - state["process_cpu_at"]
        allocator_stats = torch.cuda.memory_stats()
        allocator_retries = int(allocator_stats.get("num_alloc_retries", 0))
        allocator_retry_delta = allocator_retries - state["allocator_retries"]
        runner = self._get_live_llm().model_runner
        get_stage_samples = getattr(runner, "mtp_stage_profile_samples", None)
        stage_samples = get_stage_samples() if callable(get_stage_samples) else []
        new_stage_samples = stage_samples[state["mtp_profile_samples"]:]
        state["mtp_profile_samples"] = len(stage_samples)
        stage_summary = "mtp_cuda=none"
        if new_stage_samples:
            stage_names = ("draft", "verify_setup", "verify", "sampling", "commit", "mtp_advance")
            sample_summaries = []
            for sample in new_stage_samples:
                outcome = f"a{sample['accepted_count']}e{sample['emitted_count']}c{sample['commit_start']}"
                values = " ".join(f"{stage_name}:{float(sample['gpu_ms'][stage_name]):.3f}" for stage_name in stage_names if stage_name in sample["gpu_ms"])
                distributions = ",".join(
                    f"{distribution['role']}:{distribution['min_p_survivors']}/{distribution['top_p_survivors']}/{distribution['vocab']}"
                    for distribution in sample.get("distributions", ())
                )
                sample_summaries.append(f"{outcome}[{values} dist={distributions}]")
            stage_summary = f"mtp_cuda={' '.join(sample_summaries)}"
        gpu_telemetry_started = time.perf_counter()
        gpu_telemetry = self._gpu_telemetry()
        gpu_telemetry_ms = 1000.0 * (time.perf_counter() - gpu_telemetry_started)
        window = {
            "label": state["label"],
            "throughput": throughput,
            "passes_per_second": passes_per_second,
            "tokens_per_pass": tokens_per_pass,
            "runner_ms_per_pass": runner_ms_per_pass,
            "host_ms_per_pass": host_ms_per_pass,
            "context": len(seq.token_ids),
        }
        previous_window = self._last_decode_window
        perf_delta = "perf_delta=initial"
        if previous_window is not None:
            def relative_delta(name: str) -> float:
                previous = float(previous_window[name])
                return 100.0 * (float(window[name]) / previous - 1.0) if previous > 0.0 else 0.0

            throughput_delta = relative_delta("throughput")
            passes_delta = relative_delta("passes_per_second")
            tokens_per_pass_delta = relative_delta("tokens_per_pass")
            runner_delta = relative_delta("runner_ms_per_pass")
            host_delta = relative_delta("host_ms_per_pass")
            cause = "stable"
            if throughput_delta <= -10.0:
                if tokens_per_pass_delta <= -7.5 and passes_delta > -7.5:
                    cause = "mtp_acceptance"
                elif runner_delta >= 7.5:
                    cause = "gpu_runner"
                elif host_delta >= 25.0 and window["host_ms_per_pass"] - previous_window["host_ms_per_pass"] >= 0.25:
                    cause = "host_gap"
                else:
                    cause = "mixed"
            perf_delta = (
                f"perf_delta=from:{previous_window['label']} tok/s:{throughput_delta:+.1f}% "
                f"passes/s:{passes_delta:+.1f}% tok/pass:{tokens_per_pass_delta:+.1f}% "
                f"runner/pass:{runner_delta:+.1f}% host/pass:{host_delta:+.1f}% "
                f"context:{window['context'] - previous_window['context']:+d} cause:{cause}"
            )
        self._last_decode_window = window
        self._log(
            f"Decode telemetry {state['label']}: emitted={state['emitted']} calls={state['calls']} "
            f"elapsed={elapsed:.3f}s throughput={throughput:.2f} tok/s "
            f"passes={passes} passes/s={passes_per_second:.2f} runner_ms/pass={runner_ms_per_pass:.3f} host_ms/pass={host_ms_per_pass:.3f} "
            f"runner={state['runner_seconds']:.3f}s scheduler={state['scheduler_seconds']:.3f}s "
            f"postprocess={state['postprocess_seconds']:.3f}s decode={state['decode_seconds']:.3f}s "
            f"parse={state['parse_seconds']:.3f}s stream={state['stream_seconds']:.3f}s "
            f"callbacks={state['stream_callbacks']}/{state['stream_checks']} other={max(0.0, elapsed - accounted):.3f}s "
            f"cpu_thread={thread_cpu:.3f}s cpu_other={max(0.0, process_cpu - thread_cpu):.3f}s "
            f"cuda_mem={torch.cuda.memory_allocated() / (1 << 20):.0f}/{torch.cuda.memory_reserved() / (1 << 20):.0f}MiB alloc_retries=+{allocator_retry_delta} "
            f"context={len(seq.token_ids)} {mtp} {perf_delta} {stage_summary} {gpu_telemetry} telemetry_ms={gpu_telemetry_ms:.3f}"
        )
        state.update(
            last_at=time.perf_counter(),
            thread_cpu_at=time.thread_time(),
            process_cpu_at=time.process_time(),
            allocator_retries=allocator_retries,
            emitted=0,
            calls=0,
            runner_seconds=0.0,
            scheduler_seconds=0.0,
            postprocess_seconds=0.0,
            decode_seconds=0.0,
            parse_seconds=0.0,
            stream_seconds=0.0,
            stream_checks=0,
            stream_callbacks=0,
            speculative=current,
        )

    def _speculative_telemetry(self, seq, baseline: dict | None = None, label: str = "generation") -> dict | None:
        if not self.telemetry_enabled or not bool(getattr(self.model, "_prompt_enhancer_speculative_decoding", False)):
            return None
        runner = self._get_live_llm().model_runner
        current = runner.speculative_telemetry(seq.seq_id, len(seq.token_ids))
        if baseline is None:
            return current
        drafted = current["drafted"] - baseline["drafted"]
        accepted = current["accepted"] - baseline["accepted"]
        target_passes = current["target_passes"] - baseline["target_passes"]
        emitted = current["emitted_tokens"] - baseline["emitted_tokens"]
        drafted_by_position = [value - old for value, old in zip(current["drafted_by_position"], baseline["drafted_by_position"])]
        accepted_by_position = [value - old for value, old in zip(current["accepted_by_position"], baseline["accepted_by_position"])]
        position_rates = [f"{accepted_count}/{drafted_count}" for accepted_count, drafted_count in zip(accepted_by_position, drafted_by_position)]
        acceptance = 100.0 * accepted / drafted if drafted else 0.0
        tokens_per_pass = emitted / target_passes if target_passes else 0.0
        self._log(
            f"MTP telemetry {label}: accepted={accepted}/{drafted} ({acceptance:.1f}%) "
            f"by_position={position_rates} emitted={emitted} target_passes={target_passes} "
            f"tokens_per_pass={tokens_per_pass:.3f} sequence={current['sequence_tokens']} "
            f"mtp_cache={current['mtp_cache_tokens']} sync_delta={current['sync_delta']:+d} "
            f"pending={current['pending']} draft={current['draft']}"
        )
        return current

    def _log_speculative_alignment(self, seq, label: str) -> None:
        telemetry = self._speculative_telemetry(seq)
        if telemetry is not None:
            self._log(
                f"MTP state {label}: sequence={telemetry['sequence_tokens']} "
                f"mtp_cache={telemetry['mtp_cache_tokens']} sync_delta={telemetry['sync_delta']:+d} "
                f"pending={telemetry['pending']} draft={telemetry['draft']}"
            )

    @staticmethod
    def _format_preview(text: str, limit: int = 120) -> str:
        normalized = str(text or "").replace("\r", "\\r").replace("\n", "\\n")
        if len(normalized) <= limit:
            return normalized
        return f"{normalized[:limit]}..."

    @staticmethod
    def _describe_tensor(value: Any) -> str:
        if value is None:
            return "None"
        if not torch.is_tensor(value):
            return type(value).__name__
        try:
            ptr = int(value.data_ptr())
        except Exception:
            ptr = None
        return f"shape={tuple(int(x) for x in value.shape)} dtype={str(value.dtype).replace('torch.', '')} device={value.device} ptr={ptr}"

    def _describe_engine_state(self, engine) -> str:
        llm = None if engine is None else getattr(engine, "_llm", None)
        runner = None if llm is None else getattr(llm, "model_runner", None)
        live_model_len = 0 if llm is None else int(getattr(llm.config, "max_model_len", 0) or 0)
        runtime_ready = None if runner is None else getattr(runner, "_runtime_ready", None)
        runtime_signature = None if runner is None else getattr(runner, "_runtime_signature", None)
        kv_cache = None if runner is None else getattr(runner, "kv_cache", None)
        return (
            f"engine_id={id(engine) if engine is not None else None} "
            f"llm_id={id(llm) if llm is not None else None} "
            f"hints=(model_len={getattr(engine, '_max_model_len_hint', None)}, seqs={getattr(engine, '_max_num_seqs_hint', None)}, batched={getattr(engine, '_max_num_batched_tokens_hint', None)}) "
            f"live_model_len={live_model_len} runtime_ready={runtime_ready} runtime_signature={runtime_signature} "
            f"kv_cache={self._describe_tensor(kv_cache)}"
        )

    def _get_engine(self, max_context_tokens: int, max_new_tokens: int, usage_mode: str = "assistant", min_model_len: int | None = None):
        engine = qwen35_text._get_or_create_vllm_engine(self.model, usage_mode=usage_mode)
        desired_model_len, desired_num_seqs, desired_num_batched_tokens = engine._compute_runtime_hints(prompt_len=max_context_tokens, max_tokens=max_new_tokens, cfg_scale=1.0)
        min_model_len = engine._get_min_model_len_hint() if usage_mode == "assistant" else int(min_model_len or qwen35_text.QWEN35_PROMPT_MIN_MODEL_LEN)
        desired_model_len = max(desired_model_len, min_model_len)
        desired_num_batched_tokens = max(desired_num_batched_tokens, desired_model_len * desired_num_seqs)
        live_llm = getattr(engine, "_llm", None)
        self._log(
            "Requesting assistant engine "
            f"context={int(max_context_tokens)} max_new={int(max_new_tokens)} desired=(model_len={desired_model_len}, seqs={desired_num_seqs}, batched={desired_num_batched_tokens}) "
            f"live_before={self._describe_engine_state(engine)}"
        )
        if live_llm is not None and (
            int(getattr(live_llm.config, "max_model_len", 0) or 0) != desired_model_len
            or int(getattr(engine, "_max_num_seqs_hint", 0) or 0) != desired_num_seqs
            or int(getattr(engine, "_max_num_batched_tokens_hint", 0) or 0) != desired_num_batched_tokens
        ):
            self._log("Closing assistant engine before reserve because live runtime hints do not match the requested embedded decode.")
            engine.close()
            engine._max_model_len_hint = None
            engine._max_num_seqs_hint = None
            engine._max_num_batched_tokens_hint = None
        engine.reserve_runtime(prompt_len=max_context_tokens, max_tokens=max_new_tokens, cfg_scale=1.0, min_model_len=min_model_len)
        engine._ensure_llm()
        if engine._llm is None:
            raise RuntimeError("Assistant LM engine runtime is not available.")
        self._log(f"Assistant engine ready after reserve: {self._describe_engine_state(engine)}")
        return engine

    def _get_linear_state_modules(self):
        return [module for module in self._get_live_llm().model_runner.model.blk if module.layer_type == "linear_attention"]

    def _snapshot_linear_states(self):
        sources = [tensor.detach().as_subclass(torch.Tensor) for module in self._get_linear_state_modules() for tensor in (module.conv_state_buffer, module.recurrent_state_buffer)]
        if not sources:
            return []
        stream = torch.cuda.current_stream(sources[0].device) if sources[0].is_cuda else None
        saved = [torch.empty_like(source, device="cpu", pin_memory=stream is not None) for source in sources]
        try:
            for target, source in zip(saved, sources):
                target.copy_(source, non_blocking=stream is not None)
        finally:
            # Pinned destinations plus same-stream ordering: finish every transfer before
            # a caller can read the snapshot, mutate the source, or release the runtime.
            if stream is not None:
                stream.synchronize()
        return [{"conv": saved[index], "recurrent": saved[index + 1]} for index in range(0, len(saved), 2)]

    def _get_live_llm(self):
        engine = getattr(self.model, "_prompt_enhancer_vllm_engine", None)
        llm = None if engine is None else getattr(engine, "_llm", None)
        if llm is None:
            raise RuntimeError("Assistant runtime is not initialized.")
        return llm

    def get_max_model_len(self) -> int:
        return int(getattr(self._get_live_llm().config, "max_model_len", 0) or 0)

    def snapshot_sampling_state(self) -> tuple[bool, torch.Tensor | None]:
        runner = self._get_live_llm().model_runner
        generator = getattr(runner, "_sampling_generator", None)
        return generator is not None, None if generator is None else generator.get_state().detach().to("cpu").clone()

    def restore_sampling_state(self, snapshot: tuple[bool, torch.Tensor | None]) -> None:
        enabled, state = snapshot
        runner = self._get_live_llm().model_runner
        if not enabled:
            runner._sampling_generator = None
            return
        generator = torch.Generator(device=runner._get_runtime_device())
        generator.set_state(state)
        runner._sampling_generator = generator

    def snapshot_action_replay_state(self) -> dict[str, Any]:
        sampling_enabled, sampling_state = self.snapshot_sampling_state()
        presence = self._assistant_presence_state
        return {
            "sampling_enabled": sampling_enabled,
            "sampling_state": [] if sampling_state is None else sampling_state.tolist(),
            "presence": None if presence is None else {"penalty": presence.penalty, "seen_token_ids": sorted(int(token_id) for token_id in presence._seen_token_ids)},
        }

    def restore_action_replay_state(self, state: dict[str, Any]) -> None:
        sampling_enabled = bool(state["sampling_enabled"])
        sampling_values = list(state["sampling_state"])
        sampling_state = torch.tensor(sampling_values, dtype=torch.uint8) if sampling_enabled else None
        self.restore_sampling_state((sampling_enabled, sampling_state))
        presence = state["presence"]
        if presence is None:
            self._assistant_presence_state = None
            return
        self._assistant_presence_state = qwen35_text._PresencePenaltyState(presence["penalty"])
        for token_id in presence["seen_token_ids"]:
            self._assistant_presence_state.update(int(token_id))

    def _ensure_clean_runtime(self, max_context_tokens: int, max_new_tokens: int, seed: int | None = None):
        engine = self._get_engine(max_context_tokens=max_context_tokens, max_new_tokens=max_new_tokens)
        llm = engine._llm
        llm.reset()
        llm.model_runner.ensure_runtime_ready()
        if llm.config.num_kvcache_blocks > 0 and len(llm.scheduler.block_manager.blocks) != llm.config.num_kvcache_blocks:
            llm.scheduler.block_manager = BlockManager(llm.config.num_kvcache_blocks, llm.config.kvcache_block_size, _kv_overlay_slots(llm.config))
        llm.model_runner.reset_generation_state()
        llm.model_runner.call("set_sampling_seed", None if seed is None else int(seed))
        llm.scheduler.waiting.clear()
        llm.scheduler.running.clear()
        return engine, llm

    def _build_sampling_params(self, max_new_tokens: int, seed: int | None, do_sample: bool, temperature: float | None, top_p: float | None, top_k: int | None, thinking_enabled: bool, available_tokens: int | None = None, suppress_token_ids: tuple[int, ...] = (), apply_repetition_penalty: bool = True, max_thinking_tokens: int | None = None):
        requested_new_tokens = max(1, int(max_new_tokens))
        resolved_available_tokens = None if available_tokens is None else max(0, int(available_tokens))
        effective_new_tokens = requested_new_tokens if resolved_available_tokens is None else min(requested_new_tokens, resolved_available_tokens)
        requested_runtime_extra = qwen35_text._resolve_prompt_runtime_extra_tokens(self.model, thinking_enabled=thinking_enabled)
        if resolved_available_tokens is None:
            effective_runtime_extra = int(requested_runtime_extra)
        else:
            effective_runtime_extra = min(int(requested_runtime_extra), max(0, resolved_available_tokens - effective_new_tokens))
        logits_bias = qwen35_text._build_suppressed_token_logits_bias(self.model, thinking_enabled=thinking_enabled)
        logits_processor, logits_processor_update_state = qwen35_text._build_prompt_logits_processor(
            self.model,
            thinking_enabled=thinking_enabled,
            max_thinking_tokens_override=(effective_runtime_extra if max_thinking_tokens is None else max_thinking_tokens) if thinking_enabled else None,
            suppress_token_ids=suppress_token_ids,
        )
        temp, normalized_top_p, normalized_top_k = qwen35_text._normalize_vllm_sampling(
            do_sample=bool(do_sample),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        return SamplingParams(
            temperature=temp,
            max_tokens=effective_new_tokens + effective_runtime_extra,
            cfg_scale=1.0,
            top_k=normalized_top_k,
            top_p=normalized_top_p,
            min_p=qwen35_text._resolve_prompt_min_p(self.model),
            repetition_penalty=qwen35_text._resolve_prompt_repetition_penalty(self.model) if apply_repetition_penalty else 1.0,
            predictive_penalty=qwen35_text._resolve_predictive_penalty_enabled(self.model),
            ignore_eos=True,
            logits_processor=logits_processor,
            logits_processor_update_state=logits_processor_update_state,
            logits_bias=logits_bias,
            seed=None if seed is None else int(seed),
        ), {
            "requested_new_tokens": requested_new_tokens,
            "effective_new_tokens": effective_new_tokens,
            "requested_runtime_extra": int(requested_runtime_extra),
            "effective_runtime_extra": int(effective_runtime_extra),
            "available_tokens": None if resolved_available_tokens is None else int(resolved_available_tokens),
        }

    def _get_active_sequence(self):
        try:
            llm = self._get_live_llm()
        except Exception:
            return None
        if llm.scheduler.running:
            return llm.scheduler.running[0]
        if llm.scheduler.waiting:
            return llm.scheduler.waiting[0]
        return None

    def _seal_sequence(self, seq: Sequence) -> None:
        seq.num_prompt_tokens = seq.num_tokens
        seq.logits_processor = None
        seq.logits_processor_update_state = None
        seq.ignore_eos = True
        self._get_live_llm().scheduler.block_manager.normalize_tail_after_prefill(seq)

    def _prefill_context(self, token_ids: list[int], seed: int | None = None) -> Sequence:
        normalized_token_ids = [int(token_id) for token_id in token_ids]
        if len(normalized_token_ids) == 0:
            raise ValueError("Cannot prefill assistant context with an empty token sequence.")
        _engine, llm = self._ensure_clean_runtime(max_context_tokens=len(normalized_token_ids), max_new_tokens=1, seed=seed)
        speculative = bool(getattr(self.model, "_prompt_enhancer_speculative_decoding", False))
        # Without MTP, the first decode step processes the final context token,
        # so the scheduled prefill leaves it for the suffix path below.
        initial_token_ids = normalized_token_ids[:_ASSISTANT_PREFILL_CHUNK_TOKENS if speculative else min(_ASSISTANT_PREFILL_CHUNK_TOKENS, len(normalized_token_ids) - 1)]
        seq = Sequence(initial_token_ids, SamplingParams(max_tokens=1, ignore_eos=True))
        seq._assistant_context_id = uuid4().hex
        llm.scheduler.add(seq)
        scheduled, is_prefill = llm.scheduler.schedule()
        if not scheduled or not is_prefill:
            raise RuntimeError("Assistant context prefill did not schedule a prefill batch.")
        if speculative:
            llm.model_runner.call("prefill_mtp_only", scheduled)
        else:
            llm.model_runner.call("run", scheduled, is_prefill)
        seq = scheduled[0]
        seq = self._chunk_prefill_suffix(seq, normalized_token_ids[len(initial_token_ids):], processed_tokens=len(initial_token_ids))
        self._seal_sequence(seq)
        self._log_speculative_alignment(seq, "after context prefill")
        self._log(f"Primed assistant context with {len(normalized_token_ids)} tokens.")
        return seq

    def _chunk_prefill_suffix(self, seq: Sequence, token_ids: list[int], chunk_tokens: int = _ASSISTANT_PREFILL_CHUNK_TOKENS, processed_tokens: int | None = None) -> Sequence:
        suffix = [int(token_id) for token_id in list(token_ids or [])]
        if suffix and seq.token_ids and suffix[0] == int(seq.token_ids[-1]) == self._single_token_id("<|im_end|>"):
            suffix = suffix[1:]  # a response that sampled its own end of turn (after parallel tool calls) keeps it once, as the chat format writes it
        if len(suffix) == 0:
            return seq
        llm = self._get_live_llm()
        speculative = bool(getattr(self.model, "_prompt_enhancer_speculative_decoding", False))
        # Without MTP, decode processes the last token (a sampled token or the
        # final context token): prefill resumes at it and stops before the new one.
        processed = int(seq.num_tokens) - 1 if processed_tokens is None else processed_tokens
        original_processor = seq.logits_processor
        original_update = seq.logits_processor_update_state
        original_max_tokens = seq.max_tokens
        seq.logits_processor = None
        seq.logits_processor_update_state = None
        seq.ignore_eos = True
        seq.max_tokens = max(int(original_max_tokens), int(seq.num_completion_tokens or 0) + len(suffix) + 8)
        try:
            total_suffix_tokens = len(suffix)
            total_chunks = (total_suffix_tokens + chunk_tokens - 1) // chunk_tokens
            for chunk_index, chunk_start in enumerate(range(0, total_suffix_tokens, chunk_tokens), start=1):
                chunk_started_at = time.perf_counter()
                chunk = suffix[chunk_start : chunk_start + chunk_tokens]
                old_num_tokens = int(seq.num_tokens)
                seq.token_ids.extend(chunk)
                seq.last_token = int(seq.token_ids[-1])
                seq.num_tokens = len(seq.token_ids)
                if not llm.scheduler.block_manager.can_prompt_append(seq, old_num_tokens):
                    del seq.token_ids[old_num_tokens:]
                    seq.num_tokens = old_num_tokens
                    seq.last_token = int(seq.token_ids[-1]) if seq.token_ids else 0
                    seq.num_cached_tokens = min(int(getattr(seq, "num_cached_tokens", old_num_tokens) or 0), old_num_tokens)
                    raise RuntimeError("Assistant chunk prefill exceeded the available KV cache blocks.")
                llm.scheduler.block_manager.begin_prompt_append(seq, old_num_tokens)
                if speculative:
                    seq.num_cached_tokens = old_num_tokens
                    llm.model_runner.call("prefill_mtp_suffix", [seq], old_num_tokens)
                elif seq.num_tokens - 1 > processed:
                    full_token_ids = seq.token_ids
                    seq.num_cached_tokens, seq.num_tokens, seq.token_ids = processed, len(full_token_ids) - 1, full_token_ids[:-1]
                    try:
                        llm.model_runner.call("prefill_only", [seq])
                    finally:
                        seq.num_tokens, seq.token_ids = len(full_token_ids), full_token_ids
                llm.scheduler.block_manager.finalize_prompt_append(seq, old_num_tokens)
                processed = seq.num_tokens - 1
                seq.num_cached_tokens = seq.num_tokens if speculative else processed
                self._log(
                    f"Chunk-prefilled assistant suffix chunk {chunk_index}/{total_chunks} "
                    f"with {len(chunk)} tokens in {time.perf_counter() - chunk_started_at:.3f}s (context={int(seq.num_tokens)})."
                )
                self._log_speculative_alignment(seq, f"after suffix chunk {chunk_index}/{total_chunks}")
        finally:
            seq.logits_processor = original_processor
            seq.logits_processor_update_state = original_update
            seq.max_tokens = original_max_tokens
        return seq

    def prime_context(self, token_ids: list[int], seed: int | None = None) -> Sequence:
        return self._prefill_context(token_ids, seed=seed)

    def restore_or_replay(self, snapshot: dict[str, Any] | None, fallback_tokens: list[int], seed: int | None = None) -> tuple[str, str]:
        replay_reason = "no exact runtime snapshot was available"
        if snapshot:
            try:
                self.restore_snapshot(snapshot)
                return "restored", "exact KV snapshot restored"
            except torch.OutOfMemoryError:
                raise
            except Exception as exc:
                self._log(f"Exact restore failed, falling back to prefill: {exc}")
                replay_reason = f"exact KV restore failed: {exc}"
        # Leave the exception handler before replay so its traceback releases GPU temporaries.
        self.prime_context(fallback_tokens, seed=seed)
        return "prefilled", replay_reason

    def extend_context(self, target_token_ids: list[int]) -> str:
        seq = self._get_active_sequence()
        if seq is None:
            raise RuntimeError("Assistant context is not initialized.")
        current_token_ids = [int(token_id) for token_id in seq.token_ids]
        if target_token_ids[: len(current_token_ids)] != current_token_ids:
            raise RuntimeError("Assistant context target does not extend the active runtime prefix.")
        suffix = [int(token_id) for token_id in target_token_ids[len(current_token_ids) :]]
        if suffix:
            seq = self._chunk_prefill_suffix(seq, suffix)
        self._seal_sequence(seq)
        return "chunk_prefilled" if suffix else "extended"

    def append_suffix(self, suffix_token_ids: list[int]) -> str:
        seq = self._get_active_sequence()
        if seq is None:
            raise RuntimeError("Assistant context is not initialized.")
        suffix = [int(token_id) for token_id in suffix_token_ids]
        if suffix:
            seq = self._chunk_prefill_suffix(seq, suffix)
        self._seal_sequence(seq)
        return "chunk_prefilled" if suffix else "extended"

    def append_completion_suffix(self, suffix_token_ids: list[int]) -> str:
        seq = self._get_active_sequence()
        if seq is None:
            raise RuntimeError("Assistant context is not initialized.")
        prompt_token_count = int(seq.num_prompt_tokens)
        suffix = [int(token_id) for token_id in suffix_token_ids]
        if suffix:
            seq = self._chunk_prefill_suffix(seq, suffix)
        self._seal_sequence(seq)
        seq.num_prompt_tokens = prompt_token_count
        return "chunk_prefilled" if suffix else "extended"

    def generate_embedded_answer(
        self,
        prompt_token_ids: list[int],
        prompt_embeds,
        prompt_position_ids,
        position_offset: int,
        *,
        max_new_tokens: int,
        seed: int | None,
        do_sample: bool,
        temperature: float | None,
        top_p: float | None,
        top_k: int | None,
        min_model_len: int | None = None,
        restore_snapshot: dict[str, Any] | None = None,
    ) -> str:
        self._log(
            "Embedded decode request "
            f"prompt_tokens={len(prompt_token_ids)} prompt_embeds={self._describe_tensor(prompt_embeds)} "
            f"prompt_position_ids={self._describe_tensor(prompt_position_ids)} position_offset={int(position_offset or 0)} "
            f"max_new={int(max_new_tokens)} seed={seed} do_sample={bool(do_sample)}"
        )
        temp, normalized_top_p, normalized_top_k = qwen35_text._normalize_vllm_sampling(do_sample=bool(do_sample), temperature=temperature, top_p=top_p, top_k=top_k)
        live_engine = getattr(self.model, "_prompt_enhancer_vllm_engine", None)
        if live_engine is not None and live_engine.has_live_context() and live_engine._llm.can_aside(len(prompt_token_ids) + int(max_new_tokens)):
            # beside the live context, in the free pages of its cache: its pages stay in place, only its per-sequence state is set aside
            self._log("Embedded decode beside the live assistant context.")
            with live_engine._llm.aside():
                response = live_engine.generate_embedded(prompt_token_ids=[int(token_id) for token_id in prompt_token_ids], prompt_embeds=prompt_embeds, prompt_position_ids=prompt_position_ids,
                                                         max_tokens=int(max_new_tokens), temperature=temp, top_p=normalized_top_p, top_k=normalized_top_k, cfg_scale=1.0, seed=seed,
                                                         use_tqdm=True, release_vram_after=False, ignore_eos=False, position_offset=int(position_offset or 0))
            cleaned = qwen35_text._clean_generated_text("" if response is None else response.get("text", ""))
            self._log(f"Embedded decode response preview={self._format_preview(cleaned)}")
            return cleaned
        snapshot = self.snapshot_context() or restore_snapshot
        engine = self._get_engine(
            max_context_tokens=len(prompt_token_ids),
            max_new_tokens=max_new_tokens,
            usage_mode="assistant" if snapshot is not None else "multimodal",
            min_model_len=min_model_len,
        )
        try:
            temp, normalized_top_p, normalized_top_k = qwen35_text._normalize_vllm_sampling(
                do_sample=bool(do_sample),
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            response = engine.generate_embedded(
                prompt_token_ids=[int(token_id) for token_id in prompt_token_ids],
                prompt_embeds=prompt_embeds,
                prompt_position_ids=prompt_position_ids,
                max_tokens=int(max_new_tokens),
                temperature=temp,
                top_p=normalized_top_p,
                top_k=normalized_top_k,
                cfg_scale=1.0,
                seed=seed,
                use_tqdm=True,
                release_vram_after=False,
                ignore_eos=False,
                position_offset=int(position_offset or 0),
            )
            cleaned = qwen35_text._clean_generated_text("" if response is None else response.get("text", ""))
            self._log(f"Embedded decode response preview={self._format_preview(cleaned)}")
            return cleaned
        finally:
            if snapshot is not None:
                self._log("Embedded decode finished; restoring assistant snapshot.")
                self.restore_snapshot(snapshot)
            else:
                self._log("Embedded decode finished without an active assistant snapshot; releasing multimodal runtime allocations.")
                engine.release_runtime_allocations()

    def start_generation_segment(self, max_new_tokens: int, seed: int | None, do_sample: bool, temperature: float | None, top_p: float | None, top_k: int | None, thinking_enabled: bool, continue_existing_completion: bool = False, suppress_token_ids: tuple[int, ...] = (), apply_repetition_penalty: bool = True, resume_segment: bool = False, max_total_tokens: int | None = None, max_thinking_tokens: int | None = None) -> tuple[Sequence, int]:
        seq = self._get_active_sequence()
        if seq is None:
            raise RuntimeError("Assistant context is not initialized.")
        llm = self._get_live_llm()
        available_tokens = max(0, int(llm.config.max_model_len) - int(seq.num_tokens))
        if max_total_tokens is not None:
            available_tokens = min(available_tokens, max_total_tokens)
        sampling_params, budget_info = self._build_sampling_params(
            max_new_tokens=max_new_tokens,
            seed=seed,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            thinking_enabled=thinking_enabled,
            available_tokens=available_tokens,
            suppress_token_ids=suppress_token_ids,
            apply_repetition_penalty=apply_repetition_penalty,
            max_thinking_tokens=max_thinking_tokens,
        )
        if budget_info["effective_new_tokens"] != budget_info["requested_new_tokens"] or budget_info["effective_runtime_extra"] != budget_info["requested_runtime_extra"]:
            self._log(
                "Adjusted assistant segment budget to fit available context: "
                f"available={budget_info['available_tokens']} "
                f"new={budget_info['effective_new_tokens']}/{budget_info['requested_new_tokens']} "
                f"thinking_extra={budget_info['effective_runtime_extra']}/{budget_info['requested_runtime_extra']}."
            )
        existing_completion_tokens = int(seq.num_completion_tokens) if continue_existing_completion else 0
        if not continue_existing_completion:
            seq.num_prompt_tokens = seq.num_tokens
        seq.max_tokens = existing_completion_tokens + int(sampling_params.max_tokens)
        seq.temperature = sampling_params.temperature
        seq.ignore_eos = True
        seq.top_k = sampling_params.top_k
        seq.top_p = sampling_params.top_p
        seq.min_p = sampling_params.min_p
        seq.cfg_scale = sampling_params.cfg_scale
        seq.repetition_penalty = sampling_params.repetition_penalty
        seq.predictive_penalty = sampling_params.predictive_penalty
        seq.repetition_penalty_start = seq.num_prompt_tokens
        if seq.repetition_penalty != 1.0:  # compaction summaries copy media ids and file names too
            self._set_repetition_rules(seq)
        seq.logits_processor = sampling_params.logits_processor
        seq.logits_processor_update_state = sampling_params.logits_processor_update_state
        seq.logits_bias = sampling_params.logits_bias
        if resume_segment and callable(seq.logits_processor_update_state):
            for token_id in seq.completion_token_ids:
                seq.logits_processor_update_state(token_id)
        if not resume_segment:
            llm.model_runner.call("set_sampling_seed", sampling_params.seed)
        return seq, int(sampling_params.max_tokens)

    def action_budget(self, phase: str) -> int:
        normalized_phase = str(phase or "").strip().lower()
        if normalized_phase not in {"thought", "statement", "tool"}:
            raise ValueError(f"Unknown assistant action phase: {phase}")
        return assistant_action_budget_tokens(self.get_max_model_len())

    def _set_repetition_rules(self, seq: Sequence) -> None:
        """The repetition penalty keeps Deepy's prose and reasoning from looping on words; ids, paths and code are copied, not repeated. Only words
        are penalized: a space then letters, or letters of a script without ASCII (CJK). Symbols, digits and the pieces glued to them in ids and
        paths (outputs, _seed, the h of 16h37) never are, and inside code fences, backticks and double quotes the penalty is suspended
        (shared.llm_engines.lmgp.engine.model_runner). The vocabulary is classified once."""
        if self._repetition_rules is None:
            texts = self.tokenizer.batch_decode([[token_id] for token_id in range(len(self.tokenizer))], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            exempt = frozenset(token_id for token_id, text in enumerate(texts) if not (text.strip().isalpha() and (text[:1] == " " or not text.isascii())))
            marks = {token_id: text for token_id, text in enumerate(texts) if "`" in text or '"' in text or "\n" in text}
            self._repetition_rules = exempt, marks
        seq.repetition_exempt_ids, seq.repetition_span_marks = self._repetition_rules

    def _install_action_processors(self, seq: Sequence, phase: str, phase_limit: int, continuing_response: bool) -> None:
        penalty_enabled = phase in {"thought", "statement"}
        if penalty_enabled and (not continuing_response or self._assistant_presence_state is None):
            self._assistant_presence_state = qwen35_text._PresencePenaltyState(qwen35_text._resolve_prompt_presence_penalty(self.model))
        presence_state = self._assistant_presence_state if penalty_enabled else qwen35_text._PresencePenaltyState(None)
        thinking_state = None
        if phase == "thought":
            thinking_state = qwen35_text._ThinkingBudgetState(
                getattr(self.model, "_prompt_enhancer_close_think_token_id", None),
                phase_limit,
                getattr(self.model, "_prompt_enhancer_stop_token_ids", ()),
            )
        if not presence_state.enabled() and (thinking_state is None or not thinking_state.enabled()):
            seq.logits_processor = None
            seq.logits_processor_update_state = None
            return

        def logits_processor(_input_ids, logits):
            presence_state.apply_(logits)
            if thinking_state is not None:
                thinking_state.apply_(logits)
            return logits

        if thinking_state is None or not thinking_state.enabled():
            logits_processor_without_penalty = None
        else:
            def logits_processor_without_penalty(_input_ids, logits):
                return thinking_state.apply_(logits)
        logits_processor._without_penalty = logits_processor_without_penalty
        logits_processor._requires_input_ids = False
        logits_processor._supports_partial_vocab = lambda: thinking_state is None or not thinking_state.in_thinking or thinking_state.generated_thinking_tokens < thinking_state.max_thinking_tokens
        if logits_processor_without_penalty is not None:
            logits_processor_without_penalty._is_token_mask = True
            logits_processor_without_penalty._requires_input_ids = False
            logits_processor_without_penalty._supports_partial_vocab = logits_processor._supports_partial_vocab

        if not presence_state.enabled():
            # Only the thinking budget remains: the GPU verifier evaluates it
            # for every hypothetical accepted prefix, as for prompt generation.
            logits_processor._speculative_batch_rules = lambda: {
                "suppressed": (),
                "thinking_stops": thinking_state.stop_token_ids,
                "thinking": (thinking_state.close_think_token_id, thinking_state.max_thinking_tokens - thinking_state.generated_thinking_tokens, int(thinking_state.in_thinking)),
            }

        def update_state(token_id: int):
            presence_state.update(token_id)
            if thinking_state is not None:
                thinking_state.update(token_id)

        seq.logits_processor = logits_processor
        seq.logits_processor_update_state = update_state

    def start_generation_action(self, phase: str, seed: int | None, do_sample: bool, temperature: float | None, top_p: float | None, top_k: int | None, thinking_enabled: bool, continuing_response: bool = False, apply_repetition_penalty: bool = True, remaining_action_tokens: int | None = None, resume_action: bool = False, allowed_token_ids: tuple[int, ...] | None = None) -> tuple[Sequence, AssistantActionState]:
        phase = str(phase or "").strip().lower()
        phase_limit = self.action_budget(phase)
        generation_limit = phase_limit if remaining_action_tokens is None else min(phase_limit, max(0, int(remaining_action_tokens)))
        seq = self._get_active_sequence()
        if seq is None:
            raise RuntimeError("Assistant context is not initialized.")
        llm = self._get_live_llm()
        available_tokens = max(0, int(llm.config.max_model_len) - int(seq.num_tokens))
        if available_tokens < generation_limit:
            raise RuntimeError(f"Assistant {phase} action requires {generation_limit} reserved tokens but only {available_tokens} remain.")
        temp, normalized_top_p, normalized_top_k = qwen35_text._normalize_vllm_sampling(do_sample=bool(do_sample), temperature=temperature, top_p=top_p, top_k=qwen35_text._resolve_prompt_top_k(self.model, top_k))
        existing_completion_tokens = int(seq.num_completion_tokens) if continuing_response else 0
        if not continuing_response:
            seq.num_prompt_tokens = seq.num_tokens
        seq.max_tokens = existing_completion_tokens + generation_limit + 1
        seq.temperature = temp
        seq.ignore_eos = True
        seq.top_k = normalized_top_k
        seq.top_p = normalized_top_p
        seq.min_p = qwen35_text._resolve_prompt_min_p(self.model)
        seq.cfg_scale = 1.0
        # Tool calls are code and JSON, whose brackets, quotes and names repeat by design: penalizing the tokens a call already used tips close choices
        # to wrong ones (os.makedirs/assets for os.makedirs(assets). Thinking and prose keep it, as they keep the presence penalty.
        seq.repetition_penalty = qwen35_text._resolve_prompt_repetition_penalty(self.model) if apply_repetition_penalty and phase in {"thought", "statement"} else 1.0
        seq.predictive_penalty = qwen35_text._resolve_predictive_penalty_enabled(self.model)
        if seq.repetition_penalty != 1.0:
            self._set_repetition_rules(seq)
        if not resume_action:
            seq.repetition_penalty_start = seq.num_tokens
        seq.logits_bias = qwen35_text._build_suppressed_token_logits_bias(self.model, thinking_enabled=thinking_enabled) if allowed_token_ids is None else self._allowed_tokens_bias(tuple(allowed_token_ids))
        self._install_action_processors(seq, phase, generation_limit, continuing_response=continuing_response)
        if not continuing_response:
            llm.model_runner.call("set_sampling_seed", None if seed is None else int(seed))
        return seq, AssistantActionState(phase=phase, limit=generation_limit)

    def _append_action_suffix(self, text: str) -> None:
        token_ids = self.tokenizer.encode(str(text or ""), add_special_tokens=False)
        if torch.is_tensor(token_ids):
            token_ids = token_ids.tolist()
        if token_ids:
            self.append_completion_suffix([int(token_id) for token_id in token_ids])

    def tool_call_follows(self, seed: int | None, do_sample: bool, temperature: float | None, top_p: float | None, top_k: int | None, thinking_enabled: bool, stop_requested=None, apply_repetition_penalty: bool = True) -> bool:
        """After a complete tool call, whether the model opens another one in the same response (the chat format of parallel calls).
        One or two decode steps restricted to the format's choices - a newline then <tool_call>, or the end of the turn - so nothing
        is rewound and no state is snapshotted. A sampled end of turn stays where the format writes it; the tool results' suffix
        does not repeat it (_chunk_prefill_suffix)."""
        tool_open, turn_end, newline = (self._single_token_id(text) for text in ("<tool_call>", "<|im_end|>", "\n"))
        for allowed in ((newline, tool_open, turn_end), (tool_open, turn_end)):
            result = self.generate_action(phase="statement", seed=seed, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k, thinking_enabled=thinking_enabled,
                                          stop_requested=stop_requested, continuing_response=True, apply_repetition_penalty=apply_repetition_penalty, remaining_action_tokens=1, allowed_token_ids=allowed)
            if result.stop_reason != "statement_budget_exhausted":  # a newline uses up the one-token step: the second step decides
                return result.stop_reason == "tool_start"
        return False

    def _single_token_id(self, text: str) -> int:
        cache = self.__dict__.setdefault("_single_token_ids", {})
        if text not in cache:
            token_ids = self.tokenizer.encode(text, add_special_tokens=False)
            token_ids = token_ids.tolist() if torch.is_tensor(token_ids) else list(token_ids)
            if len(token_ids) != 1:
                raise RuntimeError(f"{text!r} is not a single token for this tokenizer.")
            cache[text] = int(token_ids[0])
        return cache[text]

    def _allowed_tokens_bias(self, token_ids: tuple[int, ...]):
        """A full-vocabulary bias that leaves only token_ids possible; cached, so that its device copy is made once."""
        cache = self.__dict__.setdefault("_allowed_tokens_biases", {})
        if token_ids not in cache:
            bias = torch.full((int(self.model.config.vocab_size),), float(qwen35_text.QWEN35_PROMPT_SUPPRESS_LOGITS_BIAS), dtype=torch.float32)
            bias[list(token_ids)] = 0
            cache[token_ids] = bias
        return cache[token_ids]

    def _close_exhausted_thought(self, budget_tokens: int) -> None:
        self._append_action_suffix(f"\n{assistant_thought_budget_update(budget_tokens)}\n</think>")

    def generate_action(self, phase: str, seed: int | None, do_sample: bool, temperature: float | None, top_p: float | None, top_k: int | None, thinking_enabled: bool, stop_requested=None, stream_callback=None, stream_interval_seconds: float = 1.0, continuing_response: bool = False, apply_repetition_penalty: bool = True, remaining_action_tokens: int | None = None, resume_action: bool = False, pause_requested=None, allowed_token_ids: tuple[int, ...] | None = None) -> AssistantDecodeResult:
        seq, action = self.start_generation_action(phase=phase, seed=seed, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k, thinking_enabled=thinking_enabled, continuing_response=continuing_response, apply_repetition_penalty=apply_repetition_penalty, remaining_action_tokens=remaining_action_tokens, resume_action=resume_action, allowed_token_ids=allowed_token_ids)
        stop_token_ids = {int(token_id) for token_id in getattr(self.model, "_prompt_enhancer_stop_token_ids", []) or [] if int(token_id) >= 0}
        speculative = bool(getattr(self.model, "_prompt_enhancer_speculative_decoding", False))
        boundary_markers = ["</think>"] if action.phase == "thought" else ["<tool_call>", *(["<think>"] if thinking_enabled else [])] if action.phase == "statement" else ["</tool_call>"]
        boundary_token_sequences = []
        for marker in boundary_markers:
            marker_token_ids = self.tokenizer.encode(marker, add_special_tokens=False)
            if torch.is_tensor(marker_token_ids):
                marker_token_ids = marker_token_ids.tolist()
            boundary_token_sequences.append(tuple(int(token_id) for token_id in marker_token_ids))
        if speculative:
            boundary_token_ids = set(stop_token_ids)
            for marker_token_ids in boundary_token_sequences:
                if len(marker_token_ids) == 1:
                    boundary_token_ids.add(int(marker_token_ids[0]))
            seq.speculative_stop_token_ids = boundary_token_ids
        stream_emitter = ThrottledStreamEmitter(stream_interval_seconds) if callable(stream_callback) else None
        raw_text = self.tokenizer.decode(seq.completion_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        baseline_close_think = len(re.findall(r"</think>", raw_text, flags=re.IGNORECASE))
        baseline_open_think = len(re.findall(r"<think>", raw_text, flags=re.IGNORECASE))
        baseline_open_tool = len(re.findall(r"<tool_call>", raw_text, flags=re.IGNORECASE))
        boundary_tail_size = max((len(token_ids) for token_ids in boundary_token_sequences), default=1) - 1
        boundary_tail_tokens = list(seq.completion_token_ids[-boundary_tail_size:]) if boundary_tail_size else []
        telemetry_baseline = self._speculative_telemetry(seq)
        decode_telemetry = self._start_decode_telemetry(seq, f"action={action.phase}")
        runaway_checked = action.generated_tokens

        def finish(stop_reason: str, stop_token_id: int | None = None, current_text: str | None = None) -> AssistantDecodeResult:
            decode_started = time.perf_counter() if decode_telemetry is not None and current_text is None else 0.0
            if current_text is None:
                current_text = self.tokenizer.decode(seq.completion_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            decode_seconds = time.perf_counter() - decode_started if decode_started else 0.0
            stream_started = time.perf_counter() if decode_telemetry is not None and stream_emitter is not None else 0.0
            stream_callback_called = False
            if stream_emitter is not None:
                stream_callback_called = stream_emitter.emit(stream_callback, raw_text=current_text, token_count=action.generated_tokens, stop_reason=stop_reason, is_final=True, force=True)
            stream_seconds = time.perf_counter() - stream_started if stream_started else 0.0
            self._update_decode_telemetry(decode_telemetry, seq, decode_seconds=decode_seconds, stream_seconds=stream_seconds, stream_checked=stream_emitter is not None, stream_callback=stream_callback_called, force=True)
            self._speculative_telemetry(seq, telemetry_baseline, f"action={action.phase} stop={stop_reason}")
            self._log_mtp_stage_profile(f"action={action.phase} stop={stop_reason}")
            return AssistantDecodeResult(raw_text=current_text, stop_reason=stop_reason, token_count=action.generated_tokens, stop_token_id=stop_token_id, phase=action.phase)

        while action.remaining_tokens > 0:
            if callable(stop_requested) and stop_requested():
                return finish("interrupted")
            if callable(pause_requested) and pause_requested():
                return finish("paused")
            llm = self._get_live_llm()
            if len(seq.token_ids) >= int(llm.config.max_model_len):
                return finish("context_limit")
            scheduler_started = time.perf_counter() if decode_telemetry is not None else 0.0
            try:
                scheduled, is_prefill = llm.scheduler.schedule()
            except AssertionError:
                if len(seq.token_ids) >= int(llm.config.max_model_len):
                    return finish("context_limit")
                raise
            scheduler_seconds = time.perf_counter() - scheduler_started if decode_telemetry is not None else 0.0
            if speculative:
                scheduled[0].speculative_max_emission = action.remaining_tokens
            runner_started = time.perf_counter() if decode_telemetry is not None else 0.0
            sampled_token_ids = llm.model_runner.call("run", scheduled, is_prefill)
            runner_seconds = time.perf_counter() - runner_started if decode_telemetry is not None else 0.0
            postprocess_started = time.perf_counter() if decode_telemetry is not None else 0.0
            emitted_tokens = llm.scheduler.postprocess(scheduled, sampled_token_ids)
            postprocess_seconds = time.perf_counter() - postprocess_started if decode_telemetry is not None else 0.0
            seq = scheduled[0]
            action.generated_tokens += emitted_tokens
            sampled_tokens = sampled_token_ids[0] if isinstance(sampled_token_ids[0], list) else [sampled_token_ids[0]]
            sampled_tokens = [int(token_id) for token_id in sampled_tokens]
            last_token_id = int(sampled_tokens[-1])
            combined_boundary_tokens = boundary_tail_tokens + sampled_tokens
            canonical_boundary = any(marker_tokens and any(combined_boundary_tokens[index:index + len(marker_tokens)] == list(marker_tokens) for index in range(len(combined_boundary_tokens) - len(marker_tokens) + 1)) for marker_tokens in boundary_token_sequences)
            boundary_tail_tokens = combined_boundary_tokens[-boundary_tail_size:] if boundary_tail_size else []
            decode_started = time.perf_counter() if decode_telemetry is not None else 0.0
            probe_text = "" if canonical_boundary else self.tokenizer.decode(sampled_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            boundary_candidate = canonical_boundary or ">" in probe_text or action.phase == "tool" and "}" in probe_text
            stream_due = stream_emitter is not None and stream_emitter.is_due()
            if boundary_candidate or stream_due:
                raw_text = self.tokenizer.decode(seq.completion_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            decode_seconds = time.perf_counter() - decode_started if decode_telemetry is not None else 0.0
            stream_started = time.perf_counter() if decode_telemetry is not None and stream_due else 0.0
            stream_callback_called = False
            if stream_due:
                stream_callback_called = stream_emitter.emit(stream_callback, raw_text=raw_text, token_count=action.generated_tokens, stop_reason=None, is_final=False)
            stream_seconds = time.perf_counter() - stream_started if stream_started else 0.0
            parse_started = time.perf_counter() if decode_telemetry is not None else 0.0
            stop_reason = None
            if boundary_candidate and action.phase == "tool" and not validate_tool_call_structure(raw_text) and has_complete_tool_call(raw_text):
                stop_reason = "tool_call"
            elif boundary_candidate and action.phase == "thought" and len(re.findall(r"</think>", raw_text, flags=re.IGNORECASE)) > baseline_close_think:
                stop_reason = "thought_complete"
            elif boundary_candidate and action.phase == "statement":
                if len(re.findall(r"<tool_call>", raw_text, flags=re.IGNORECASE)) > baseline_open_tool:
                    stop_reason = "tool_start"
                elif thinking_enabled and len(re.findall(r"<think>", raw_text, flags=re.IGNORECASE)) > baseline_open_think:
                    stop_reason = "thought_start"
            if stop_reason is None and any(int(token_id) in stop_token_ids for token_id in sampled_tokens):
                stop_reason = "tool_call" if action.phase == "tool" else "stop_token"
            if stop_reason is None and boundary_candidate and action.phase == "tool" and action.generated_tokens - runaway_checked >= 128:
                runaway_checked = action.generated_tokens
                stop_reason = "tool_repetition" if repeated_tail(raw_text) else None
            parse_seconds = time.perf_counter() - parse_started if decode_telemetry is not None else 0.0
            self._update_decode_telemetry(decode_telemetry, seq, emitted_tokens, runner_seconds, scheduler_seconds, postprocess_seconds, decode_seconds, parse_seconds, stream_seconds, stream_emitter is not None, stream_callback_called)
            if stop_reason is not None:
                return finish(stop_reason, last_token_id, raw_text)

        if action.phase == "thought":
            self._close_exhausted_thought(self.action_budget(action.phase))
            return finish("thought_budget_exhausted")
        if action.phase == "tool":
            raw_text = self.tokenizer.decode(seq.completion_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            if not validate_tool_call_structure(raw_text) and has_complete_tool_call(raw_text):
                return finish("tool_call", current_text=raw_text)
            return finish("tool_budget_exhausted", current_text=raw_text)
        return finish(f"{action.phase}_budget_exhausted")

    def generate_segment(self, max_new_tokens: int, seed: int | None, do_sample: bool, temperature: float | None, top_p: float | None, top_k: int | None, thinking_enabled: bool, stop_requested=None, stream_callback=None, stream_interval_seconds: float = 1.0, continue_existing_completion: bool = False, suppress_token_ids: tuple[int, ...] = (), apply_repetition_penalty: bool = True, resume_segment: bool = False, pause_requested=None, max_total_tokens: int | None = None, max_thinking_tokens: int | None = None) -> AssistantDecodeResult:
        seq, requested_segment_tokens = self.start_generation_segment(max_new_tokens=max_new_tokens, seed=seed, do_sample=do_sample, temperature=temperature, top_p=top_p, top_k=top_k, thinking_enabled=thinking_enabled, continue_existing_completion=continue_existing_completion, suppress_token_ids=suppress_token_ids, apply_repetition_penalty=apply_repetition_penalty, resume_segment=resume_segment, max_total_tokens=max_total_tokens, max_thinking_tokens=max_thinking_tokens)
        existing_completion_tokens = int(seq.num_completion_tokens)
        requested_segment_tokens = max(0, int(requested_segment_tokens))
        seq.max_tokens = max(int(seq.max_tokens or 0), existing_completion_tokens + requested_segment_tokens + 1)
        stop_token_ids = {int(token_id) for token_id in getattr(self.model, "_prompt_enhancer_stop_token_ids", []) or [] if int(token_id) >= 0}
        speculative = bool(getattr(self.model, "_prompt_enhancer_speculative_decoding", False))
        if speculative:
            seq.speculative_stop_token_ids = stop_token_ids
        boundary_token_ids = self.tokenizer.encode("</tool_call>", add_special_tokens=False)
        if torch.is_tensor(boundary_token_ids):
            boundary_token_ids = boundary_token_ids.tolist()
        boundary_token_ids = tuple(int(token_id) for token_id in boundary_token_ids)
        boundary_tail_size = max(0, len(boundary_token_ids) - 1)
        boundary_tail_tokens = list(seq.completion_token_ids[-boundary_tail_size:]) if boundary_tail_size else []
        stream_emitter = ThrottledStreamEmitter(stream_interval_seconds) if callable(stream_callback) else None
        raw_text = ""
        generated_tokens = 0
        telemetry_baseline = self._speculative_telemetry(seq)
        decode_telemetry = self._start_decode_telemetry(seq, "segment")

        def finish(stop_reason: str, stop_token_id: int | None = None, token_count: int | None = None, current_text: str | None = None) -> AssistantDecodeResult:
            nonlocal raw_text
            decode_started = time.perf_counter() if decode_telemetry is not None and current_text is None else 0.0
            if current_text is None:
                current_text = self.tokenizer.decode(seq.completion_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            raw_text = current_text
            decode_seconds = time.perf_counter() - decode_started if decode_started else 0.0
            stream_started = time.perf_counter() if decode_telemetry is not None and stream_emitter is not None else 0.0
            stream_callback_called = False
            if stream_emitter is not None:
                stream_callback_called = stream_emitter.emit(stream_callback, raw_text=raw_text, token_count=generated_tokens if token_count is None else token_count, stop_reason=stop_reason, is_final=True, force=True)
            stream_seconds = time.perf_counter() - stream_started if stream_started else 0.0
            self._update_decode_telemetry(decode_telemetry, seq, decode_seconds=decode_seconds, stream_seconds=stream_seconds, stream_checked=stream_emitter is not None, stream_callback=stream_callback_called, force=True)
            self._speculative_telemetry(seq, telemetry_baseline, f"segment stop={stop_reason}")
            self._log_mtp_stage_profile(f"segment stop={stop_reason}")
            return AssistantDecodeResult(raw_text=raw_text, stop_reason=stop_reason, token_count=generated_tokens if token_count is None else token_count, stop_token_id=stop_token_id)

        while generated_tokens < requested_segment_tokens:
            if callable(stop_requested) and stop_requested():
                return finish("interrupted")
            if callable(pause_requested) and pause_requested():
                return finish("paused")
            llm = self._get_live_llm()
            if len(seq.token_ids) >= int(llm.config.max_model_len):
                return finish("context_limit")
            scheduler_started = time.perf_counter() if decode_telemetry is not None else 0.0
            try:
                scheduled, is_prefill = llm.scheduler.schedule()
            except AssertionError:
                if len(seq.token_ids) >= int(llm.config.max_model_len):
                    return finish("context_limit")
                raise
            scheduler_seconds = time.perf_counter() - scheduler_started if decode_telemetry is not None else 0.0
            if speculative:
                scheduled[0].speculative_max_emission = requested_segment_tokens - generated_tokens
            runner_started = time.perf_counter() if decode_telemetry is not None else 0.0
            sampled_token_ids = llm.model_runner.call("run", scheduled, is_prefill)
            runner_seconds = time.perf_counter() - runner_started if decode_telemetry is not None else 0.0
            postprocess_started = time.perf_counter() if decode_telemetry is not None else 0.0
            emitted_tokens = llm.scheduler.postprocess(scheduled, sampled_token_ids)
            postprocess_seconds = time.perf_counter() - postprocess_started if decode_telemetry is not None else 0.0
            seq = scheduled[0]
            generated_tokens += emitted_tokens
            sampled_tokens = sampled_token_ids[0] if isinstance(sampled_token_ids[0], list) else [sampled_token_ids[0]]
            sampled_tokens = [int(token_id) for token_id in sampled_tokens]
            last_token_id = int(sampled_tokens[-1])
            combined_boundary_tokens = boundary_tail_tokens + sampled_tokens
            canonical_boundary = bool(boundary_token_ids) and any(combined_boundary_tokens[index:index + len(boundary_token_ids)] == list(boundary_token_ids) for index in range(len(combined_boundary_tokens) - len(boundary_token_ids) + 1))
            boundary_tail_tokens = combined_boundary_tokens[-boundary_tail_size:] if boundary_tail_size else []
            decode_started = time.perf_counter() if decode_telemetry is not None else 0.0
            probe_text = "" if canonical_boundary else self.tokenizer.decode(sampled_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            boundary_candidate = canonical_boundary or ">" in probe_text or "}" in probe_text
            stream_due = stream_emitter is not None and stream_emitter.is_due()
            if boundary_candidate or stream_due:
                raw_text = self.tokenizer.decode(seq.completion_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            decode_seconds = time.perf_counter() - decode_started if decode_telemetry is not None else 0.0
            stream_started = time.perf_counter() if decode_telemetry is not None and stream_due else 0.0
            stream_callback_called = False
            if stream_due:
                stream_callback_called = stream_emitter.emit(stream_callback, raw_text=raw_text, token_count=generated_tokens, stop_reason=None, is_final=False)
            stream_seconds = time.perf_counter() - stream_started if stream_started else 0.0
            parse_started = time.perf_counter() if decode_telemetry is not None else 0.0
            # Segments (compaction summaries) may quote earlier calls or settings
            # objects: only a tagged call ends them early.
            stop_reason = "tool_call" if boundary_candidate and has_complete_tool_call(raw_text, allow_untagged=False) else "stop_token" if last_token_id in stop_token_ids else None
            parse_seconds = time.perf_counter() - parse_started if decode_telemetry is not None else 0.0
            self._update_decode_telemetry(decode_telemetry, seq, emitted_tokens, runner_seconds, scheduler_seconds, postprocess_seconds, decode_seconds, parse_seconds, stream_seconds, stream_emitter is not None, stream_callback_called)
            if stop_reason is not None:
                return finish(stop_reason, last_token_id, current_text=raw_text)
        seq.max_tokens = existing_completion_tokens + requested_segment_tokens
        raw_text = self.tokenizer.decode(seq.completion_token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        return finish("max_tokens", token_count=requested_segment_tokens, current_text=raw_text)

    def snapshot_context(self, previous: dict[str, Any] | None = None) -> dict[str, Any] | None:
        seq = self._get_active_sequence()
        if seq is None:
            self._log("Snapshot requested but no active assistant sequence is available.")
            return None
        llm = self._get_live_llm()
        runner = llm.model_runner
        from shared.llm_engines.snapshot_cache import snapshot_cache

        # The final token may still be uncached. Only complete preceding pages are immutable.
        reuse_tokens = 0
        if previous is not None and previous["context_id"] == seq._assistant_context_id:
            saved_tokens = previous["sequence"]["token_ids"]
            if seq.token_ids[:len(saved_tokens)] == saved_tokens:
                reuse_tokens = max(0, len(saved_tokens) - 1)
        reuse_blocks = reuse_tokens // llm.config.kvcache_block_size
        if reuse_blocks and previous["sequence"]["block_table"][:reuse_blocks] != seq.block_table[:reuse_blocks]:
            reuse_tokens = reuse_blocks = 0
        ranges = []
        for block_id in seq.block_table:
            if ranges and ranges[-1][1] == block_id:
                ranges[-1] = (ranges[-1][0], block_id + 1)
            else:
                ranges.append((block_id, block_id + 1))
        snapshot = {
            "context_id": seq._assistant_context_id,
            "max_model_len_hint": getattr(getattr(self.model, "_prompt_enhancer_vllm_engine", None), "_max_model_len_hint", None),
            "max_num_seqs_hint": getattr(getattr(self.model, "_prompt_enhancer_vllm_engine", None), "_max_num_seqs_hint", None),
            "max_num_batched_tokens_hint": getattr(getattr(self.model, "_prompt_enhancer_vllm_engine", None), "_max_num_batched_tokens_hint", None),
            "runner_max_model_len": int(llm.config.max_model_len),
            "sequence": {
                "token_ids": [int(token_id) for token_id in seq.token_ids],
                "num_prompt_tokens": int(seq.num_prompt_tokens),
                "num_cached_tokens": int(seq.num_cached_tokens),
                "block_table": [int(block_id) for block_id in seq.block_table],
                "status": seq.status.name,
                "max_tokens": int(seq.max_tokens),
                "temperature": float(seq.temperature),
                "ignore_eos": bool(seq.ignore_eos),
                "top_k": None if seq.top_k is None else int(seq.top_k),
                "top_p": None if seq.top_p is None else float(seq.top_p),
                "min_p": None if seq.min_p is None else float(seq.min_p),
                "repetition_penalty": None if seq.repetition_penalty is None else float(seq.repetition_penalty),
                "predictive_penalty": bool(seq.predictive_penalty),
                "repetition_penalty_start": int(seq.repetition_penalty_start),
                "overlay_slot": int(seq.overlay_slot),
                "ring_high": int(seq.ring_high),
            },
            "block_manager": {
                "block_size": int(llm.scheduler.block_manager.block_size),
                "blocks": [
                    {
                        "ref_count": int(block.ref_count),
                        "hash": int(block.hash),
                        "token_ids": [int(token_id) for token_id in block.token_ids],
                    }
                    for block in llm.scheduler.block_manager.blocks
                ],
                "hash_to_block_id": {int(hash_key): int(block_id) for hash_key, block_id in llm.scheduler.block_manager.hash_to_block_id.items()},
                "free_block_ids": [int(block_id) for block_id in llm.scheduler.block_manager.free_block_ids],
                "used_block_ids": [int(block_id) for block_id in llm.scheduler.block_manager.used_block_ids],
                "free_overlay_slots": [int(slot) for slot in llm.scheduler.block_manager.free_overlay_slots],
            },
            "kv_tensors": {name: snapshot_cache(tensor, ranges, axis=axis, previous=None if previous is None else previous["kv_tensors"][name], reuse=reuse_blocks)
                           for name, (tensor, axis) in runner.kv_snapshot_items().items()},
            "kv_overlay": runner.snapshot_overlay(seq),
            "linear_states": self._snapshot_linear_states(),
            "speculative_state": runner.snapshot_speculative_state(seq.seq_id, previous=None if previous is None else previous["speculative_state"], reuse_tokens=reuse_tokens) if bool(getattr(self.model, "_prompt_enhancer_speculative_decoding", False)) else None,
            "generation_state": {
                "sampling": self.snapshot_sampling_state(),
                "presence": None if self._assistant_presence_state is None else {
                    "penalty": self._assistant_presence_state.penalty,
                    "seen_token_ids": sorted(int(token_id) for token_id in self._assistant_presence_state._seen_token_ids),
                },
            },
        }
        self._log(
            f"Snapshotted assistant context with {len(seq.token_ids)} tokens. "
            f"saved_hints=(model_len={snapshot['max_model_len_hint']}, seqs={snapshot['max_num_seqs_hint']}, batched={snapshot['max_num_batched_tokens_hint']}) "
            f"runner_state={self._describe_engine_state(getattr(self.model, '_prompt_enhancer_vllm_engine', None))}"
        )
        return snapshot

    def snapshot_rewind_state(self, context_snapshot: dict[str, Any] | None = None) -> dict[str, Any] | None:
        seq = self._get_active_sequence()
        if seq is None:
            return None
        llm = self._get_live_llm()
        runner = llm.model_runner
        speculative_state = None
        if bool(getattr(self.model, "_prompt_enhancer_speculative_decoding", False)):
            if context_snapshot is None:
                speculative_state = runner.snapshot_speculative_rewind_state(seq.seq_id)
            else:
                saved = context_snapshot["speculative_state"]
                speculative_state = {"mtp_cache_length": saved["mtp_cache"]["seq_length"], "draft": saved["draft"], "pending": saved["pending"]}
        return {
            "context_id": seq._assistant_context_id,
            "token_ids": [int(token_id) for token_id in seq.token_ids],
            "num_prompt_tokens": int(seq.num_prompt_tokens),
            "num_cached_tokens": int(seq.num_cached_tokens),
            "block_table": [int(block_id) for block_id in seq.block_table],
            "linear_states": context_snapshot["linear_states"] if context_snapshot is not None else self._snapshot_linear_states(),
            "speculative_state": speculative_state,
        }

    def restore_rewind_state(self, snapshot: dict[str, Any]) -> None:
        llm = self._get_live_llm()
        runner = llm.model_runner
        seq = self._get_active_sequence()
        # Full snapshots restore the same context into a potentially different GPU
        # allocation. A new prefill gets a different identity, even for equal tokens.
        if seq is None or seq._assistant_context_id != snapshot["context_id"]:
            raise RuntimeError("Assistant semantic-boundary checkpoint does not belong to the active context.")
        token_ids = [int(token_id) for token_id in snapshot["token_ids"]]
        if list(seq.token_ids[:len(token_ids)]) != token_ids:
            raise RuntimeError("Assistant live context no longer contains the semantic-boundary prefix.")
        block_manager = llm.scheduler.block_manager
        kept_block_table = [int(block_id) for block_id in snapshot["block_table"]]
        for block_id in reversed(seq.block_table[len(kept_block_table):]):
            block = block_manager.blocks[int(block_id)]
            block.ref_count -= 1
            if block.ref_count == 0:
                if block.hash != -1 and block_manager.hash_to_block_id.get(block.hash) == block.block_id:
                    del block_manager.hash_to_block_id[block.hash]
                block_manager._deallocate_block(block.block_id)
        seq.token_ids = token_ids
        seq.last_token = token_ids[-1]
        seq.num_tokens = len(token_ids)
        seq.num_prompt_tokens = min(int(snapshot["num_prompt_tokens"]), len(token_ids))
        seq.num_cached_tokens = min(int(snapshot["num_cached_tokens"]), len(token_ids))
        seq.block_table = kept_block_table
        if kept_block_table:
            tail = block_manager.blocks[kept_block_table[-1]]
            if tail.hash != -1 and block_manager.hash_to_block_id.get(tail.hash) == tail.block_id:
                del block_manager.hash_to_block_id[tail.hash]
            tail.hash = -1
            tail.token_ids = []
        linear_modules = self._get_linear_state_modules()
        linear_states = snapshot["linear_states"]
        if len(linear_modules) != len(linear_states):
            raise RuntimeError("Assistant semantic-boundary recurrent state layout changed.")
        with torch.inference_mode():
            for module, saved_state in zip(linear_modules, linear_states):
                module.conv_state_buffer.copy_(saved_state["conv"])
                module.recurrent_state_buffer.copy_(saved_state["recurrent"])
        speculative_state = snapshot.get("speculative_state")
        if speculative_state is not None:
            runner.restore_speculative_rewind_state(seq.seq_id, speculative_state)
        torch.cuda.synchronize()

    def restore_snapshot(self, snapshot: dict[str, Any]) -> None:
        engine = qwen35_text._get_or_create_vllm_engine(self.model, usage_mode="assistant")
        saved_model_len = int(snapshot.get("max_model_len_hint", 0) or snapshot.get("runner_max_model_len", 0) or 0)
        saved_num_seqs = int(snapshot.get("max_num_seqs_hint", 0) or 1)
        saved_num_batched_tokens = int(snapshot.get("max_num_batched_tokens_hint", 0) or 0)
        saved_model_len = max(saved_model_len, engine._get_min_model_len_hint())
        saved_num_seqs = max(1, saved_num_seqs)
        saved_num_batched_tokens = max(saved_num_batched_tokens, saved_model_len * saved_num_seqs)
        live_llm = getattr(engine, "_llm", None)
        self._log(
            "Restoring assistant snapshot "
            f"saved=(model_len={saved_model_len}, seqs={saved_num_seqs}, batched={saved_num_batched_tokens}) "
            f"live_before={self._describe_engine_state(engine)}"
        )
        if live_llm is not None and (
            int(getattr(live_llm.config, "max_model_len", 0) or 0) != saved_model_len
            or int(getattr(engine, "_max_num_seqs_hint", 0) or 0) != saved_num_seqs
            or int(getattr(engine, "_max_num_batched_tokens_hint", 0) or 0) != saved_num_batched_tokens
        ):
            self._log("Closing assistant engine before restore because live runtime hints do not match the saved snapshot.")
            engine.close()
        engine._max_model_len_hint = saved_model_len
        engine._max_num_seqs_hint = saved_num_seqs
        engine._max_num_batched_tokens_hint = saved_num_batched_tokens
        engine._ensure_llm()
        llm = engine._llm
        llm.reset()
        llm.model_runner.ensure_runtime_ready()
        runner = llm.model_runner
        if not hasattr(runner, "kv_cache"):
            raise RuntimeError("Assistant runtime has no KV cache to restore into.")
        saved_tensors = snapshot.get("kv_tensors") or {}
        live_tensors = runner.kv_snapshot_items()
        if set(saved_tensors) != set(live_tensors) or any(tuple(saved_tensors[name]["shape"]) != tuple(tensor.shape) for name, (tensor, _axis) in live_tensors.items()):
            saved_shapes = {name: saved["shape"] for name, saved in saved_tensors.items()}
            live_shapes = {name: tuple(int(x) for x in tensor.shape) for name, (tensor, _axis) in live_tensors.items()}
            self._log(f"Assistant KV cache snapshot mismatch saved_shape={saved_shapes} live_shape={live_shapes}")
            raise RuntimeError("Assistant KV cache snapshot shape does not match current runtime.")
        with torch.inference_mode():
            from shared.llm_engines.snapshot_cache import restore_cache

            for name, (tensor, _axis) in live_tensors.items():
                restore_cache(tensor, saved_tensors[name])
        linear_modules = self._get_linear_state_modules()
        linear_states = snapshot.get("linear_states", [])
        if len(linear_modules) != len(linear_states):
            raise RuntimeError("Assistant linear-state snapshot does not match runtime layer count.")
        with torch.inference_mode():
            for module, saved_state in zip(linear_modules, linear_states):
                saved_conv = saved_state["conv"]
                saved_recurrent = saved_state["recurrent"]
                if tuple(saved_conv.shape) != tuple(module.conv_state_buffer.shape) or tuple(saved_recurrent.shape) != tuple(module.recurrent_state_buffer.shape):
                    raise RuntimeError("Assistant linear-state snapshot tensor shape mismatch.")
                module.conv_state_buffer.copy_(saved_conv)
                module.recurrent_state_buffer.copy_(saved_recurrent)
        saved_block_manager = snapshot["block_manager"]
        llm.scheduler.block_manager = BlockManager(len(saved_block_manager["blocks"]), int(saved_block_manager["block_size"]), _kv_overlay_slots(llm.config))
        for block, saved_block in zip(llm.scheduler.block_manager.blocks, saved_block_manager["blocks"]):
            block.ref_count = int(saved_block["ref_count"])
            block.hash = int(saved_block["hash"])
            block.token_ids = [int(token_id) for token_id in saved_block["token_ids"]]
        llm.scheduler.block_manager.hash_to_block_id = {int(hash_key): int(block_id) for hash_key, block_id in saved_block_manager["hash_to_block_id"].items()}
        llm.scheduler.block_manager.free_block_ids = deque(int(block_id) for block_id in saved_block_manager["free_block_ids"])
        llm.scheduler.block_manager.used_block_ids = set(int(block_id) for block_id in saved_block_manager["used_block_ids"])
        llm.scheduler.block_manager.free_overlay_slots = deque(int(slot) for slot in saved_block_manager.get("free_overlay_slots", []))
        saved_seq = snapshot["sequence"]
        restored_seq = Sequence([int(token_id) for token_id in saved_seq["token_ids"]], SamplingParams(max_tokens=int(saved_seq["max_tokens"]), ignore_eos=bool(saved_seq["ignore_eos"])))
        restored_seq._assistant_context_id = snapshot["context_id"]
        restored_seq.num_prompt_tokens = int(saved_seq["num_prompt_tokens"])
        restored_seq.num_cached_tokens = int(saved_seq["num_cached_tokens"])
        restored_seq.block_table = [int(block_id) for block_id in saved_seq["block_table"]]
        restored_seq.status = SequenceStatus[saved_seq["status"]]
        restored_seq.max_tokens = int(saved_seq["max_tokens"])
        restored_seq.temperature = float(saved_seq["temperature"])
        restored_seq.ignore_eos = bool(saved_seq["ignore_eos"])
        restored_seq.top_k = None if saved_seq["top_k"] is None else int(saved_seq["top_k"])
        restored_seq.top_p = None if saved_seq["top_p"] is None else float(saved_seq["top_p"])
        restored_seq.min_p = None if saved_seq["min_p"] is None else float(saved_seq["min_p"])
        restored_seq.repetition_penalty = None if saved_seq["repetition_penalty"] is None else float(saved_seq["repetition_penalty"])
        restored_seq.predictive_penalty = bool(saved_seq["predictive_penalty"])
        restored_seq.repetition_penalty_start = int(saved_seq["repetition_penalty_start"])
        restored_seq.overlay_slot = int(saved_seq.get("overlay_slot", -1))
        restored_seq.ring_high = int(saved_seq.get("ring_high", 0))
        with torch.inference_mode():
            runner.restore_overlay(restored_seq, snapshot.get("kv_overlay"))
        restored_seq.logits_processor = None
        restored_seq.logits_processor_update_state = None
        llm.scheduler.waiting.clear()
        llm.scheduler.running.clear()
        if restored_seq.status == SequenceStatus.WAITING:
            llm.scheduler.waiting.append(restored_seq)
        else:
            restored_seq.status = SequenceStatus.RUNNING
            llm.scheduler.running.append(restored_seq)
        if bool(getattr(self.model, "_prompt_enhancer_speculative_decoding", False)):
            speculative_state = snapshot.get("speculative_state")
            if speculative_state is None:
                raise RuntimeError("Assistant snapshot does not contain predictive decoder state.")
            runner.restore_speculative_state(restored_seq.seq_id, speculative_state)
            self._log_speculative_alignment(restored_seq, "after snapshot restore")
        generation_state = snapshot["generation_state"]
        self.restore_sampling_state(generation_state["sampling"])
        saved_presence = generation_state["presence"]
        if saved_presence is None:
            self._assistant_presence_state = None
        else:
            self._assistant_presence_state = qwen35_text._PresencePenaltyState(saved_presence["penalty"])
            for token_id in saved_presence["seen_token_ids"]:
                self._assistant_presence_state.update(token_id)
        llm.scheduler.block_manager.normalize_tail_after_prefill(restored_seq)
        self._log(
            f"Restored assistant context with {len(restored_seq.token_ids)} tokens. "
            f"runner_state={self._describe_engine_state(engine)}"
        )
