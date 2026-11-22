# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Chat formats of the local assistant models: how a conversation becomes tokens and how sampled tokens become a turn.

Deepy and the prompt enhancer keep one text form of a model turn, whatever the model: reasoning between <think> and </think>, the answer as
plain text, each tool call as <tool_call>{"name": ..., "arguments": {...}}</tool_call> and <|im_end|> for the end of the turn. A format
writes prompts in the model's own tokens and reads sampled tokens back into that text, so that sessions, transcripts, compaction and the
tool-call parsers are the same for every model. The default format is that text itself (ChatML, rendered by the tokenizer's template)."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from types import SimpleNamespace
from typing import Any


def _encode(tokenizer, text: str) -> list[int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return [int(token_id) for token_id in (token_ids.tolist() if hasattr(token_ids, "tolist") else token_ids)]


def _decode(tokenizer, token_ids) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


class ChatFormat:
    """ChatML with <think> reasoning and <tool_call> JSON calls, rendered by the tokenizer's own chat template."""

    name = "qwen"
    turn_end = "<|im_end|>"  # the token a model samples to end its turn
    user_header = "<|im_start|>user\n"  # what a rendered user turn starts with
    tool_open, tool_close = "<tool_call>", "</tool_call>"
    think_close = "</think>"  # the token the thinking budget forces
    stop_tokens = ("<|im_end|>", "<|endoftext|>")
    thinking_tokens = ("<think>", "</think>")  # suppressed while thinking is off
    image_token = "<|image_pad|>"
    arguments_key = "arguments"
    # after a complete tool call, the tokens each next decode step may choose from: another call or the end of the turn
    follow_up_steps = (("\n", "<tool_call>", "<|im_end|>"), ("<tool_call>", "<|im_end|>"))

    def render_messages(self, tokenizer, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None, add_generation_prompt: bool, thinking_enabled: bool) -> list[int]:
        rendered = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=bool(add_generation_prompt), tokenize=True, enable_thinking=bool(thinking_enabled))
        return [int(token_id) for token_id in (rendered.tolist() if hasattr(rendered, "tolist") else rendered)]

    def render_prompt(self, tokenizer, messages: list[dict[str, Any]], thinking_enabled: bool) -> str:
        """The text of a one-shot prompt (prompt enhancer), ending where the model writes."""
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=bool(thinking_enabled)).rstrip() + "\n"

    def _generation_prefix(self, thinking_enabled: bool) -> str:
        return "<|im_start|>assistant\n" + ("<think>\n" if thinking_enabled else "<think>\n\n</think>\n\n")

    def render_user_turn(self, tokenizer, user_content: str, thinking_enabled: bool) -> list[int]:
        user_content = str(user_content or "").strip()
        if len(user_content) == 0:
            return []
        return _encode(tokenizer, f"<|im_start|>user\n{user_content}<|im_end|>\n" + self._generation_prefix(bool(thinking_enabled)))

    def render_tool_turn(self, tokenizer, tool_contents: list[str], thinking_enabled: bool, tool_names: list[str] | None = None) -> list[int]:
        """Closes the assistant turn that called the tools, then their results and the next generation prefix."""
        contents = [str(content or "").strip() for content in list(tool_contents or []) if len(str(content or "").strip()) > 0]
        if len(contents) == 0:
            return []
        suffix = "<|im_end|>\n<|im_start|>user" + "".join(f"\n<tool_response>\n{content}\n</tool_response>" for content in contents) + "<|im_end|>\n"
        return _encode(tokenizer, suffix + self._generation_prefix(bool(thinking_enabled)))

    def render_assistant_text(self, tokenizer, assistant_content: str, thinking_enabled: bool, prompt_open: bool) -> list[int]:
        """An assistant turn written by the runtime: after the generation prefix (prompt_open) or as a whole turn."""
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
        return _encode(tokenizer, suffix)

    def render_exchange(self, tokenizer, user_content: str, assistant_content: str) -> list[int]:
        return _encode(tokenizer, f"<|im_start|>user\n{user_content}<|im_end|>\n<|im_start|>assistant\n{assistant_content}<|im_end|>\n")

    def boundary_markers(self, phase: str, thinking_enabled: bool) -> list[str]:
        """The tokens that end an action: a thought, a statement (a tool call or a new thought starts) or a tool call."""
        return ["</think>"] if phase == "thought" else ["<tool_call>", *(["<think>"] if thinking_enabled else [])] if phase == "statement" else ["</tool_call>"]

    def encode_action_text(self, tokenizer, text: str) -> list[int]:
        """Text the runtime appends inside a turn being generated (a runtime update, a closed or reopened thought)."""
        return _encode(tokenizer, str(text or ""))

    def decode(self, tokenizer, token_ids, prompt_tail=()) -> str:
        """Sampled tokens as turn text; prompt_tail = the last prompt tokens, which say what the generation prefix left open."""
        return _decode(tokenizer, token_ids)

    def image_run(self, image_tokens: int) -> str:
        return "<|vision_start|>" + self.image_token * int(image_tokens) + "<|vision_end|>"

    def system_prompt(self, text: str) -> str:
        return text

    def tool_error(self, text: str) -> str:
        return text

    def scaffold_tokens(self, tokenizer, thinking_enabled: bool) -> dict[str, int]:
        """Tokens of the fixed parts: a user turn with its generation prefix, a tool result, the first one closing the calling turn."""
        user = len(self.render_user_turn(tokenizer, "x", thinking_enabled)) - len(_encode(tokenizer, "x"))
        one, two = (len(self.render_tool_turn(tokenizer, ["x"] * count, thinking_enabled, ["x"] * count)) for count in (1, 2))
        return {"user_turn": user, "tool_result": two - one - len(_encode(tokenizer, "x")), "tool_turn": 2 * one - two}

    def parse_turn(self, tokenizer, token_ids, prompt_tail=(), tool_parameters: dict[str, set[str]] | None = None) -> SimpleNamespace:
        """A sampled turn, complete or not: its thoughts, its visible text and its complete tool calls, as Deepy reads them."""
        from shared.prompt_enhancer import qwen35_text
        from shared.prompt_enhancer.qwen35_assistant_runtime import extract_tool_calls, strip_tool_blocks, strip_trailing_stop_markup, validate_tool_call_structure

        raw_text = self.decode(tokenizer, token_ids, prompt_tail)
        thinking, text = qwen35_text._split_generated_parts(strip_trailing_stop_markup(strip_tool_blocks(raw_text)))
        return SimpleNamespace(raw_text=raw_text, thinking=thinking, text=text, tool_calls=extract_tool_calls(raw_text, tool_parameters), tool_error=validate_tool_call_structure(raw_text),
                               ended=raw_text.rstrip().endswith(ChatFormat.turn_end))


@lru_cache(maxsize=4)
def _inkling_ids(tokenizer) -> SimpleNamespace:
    names = ("message_model", "content_text", "content_thinking", "content_invoke_tool_json", "end_message", "content_model_end_sampling")
    ids = SimpleNamespace(**{name: int(tokenizer.convert_tokens_to_ids(f"<|{name}|>")) for name in names})
    ids.kinds = {ids.content_thinking: "thinking", ids.content_text: "text", ids.content_invoke_tool_json: "tool"}
    # the other content tokens are kept as written
    ids.kinds.update({int(tokenizer.convert_tokens_to_ids(token)): token for token in ("<|content_image|>", "<|content_tool_error|>", "<|content_xml|>", "<|content_invoke_tool_text|>")})
    return ids


_THINK_BLOCK_RE = re.compile(r"\s*<think>\n?(.*?)\n?</think>\s*", flags=re.DOTALL)
_ARGS_KEY_RE = re.compile(r'\A(\s*\{\s*"name"\s*:\s*"(?:[^"\\]|\\.)*"\s*,\s*)"args"(\s*:)')


class Inkling2Format(ChatFormat):
    """Messages opened by a role token and a content token and closed by <|end_message|>. A model turn is a sequence of messages (thinking,
    text, one per tool call) that <|content_model_end_sampling|> ends. Rendering follows the model's chat_template.jinja, with text contents
    trimmed as the runtime's own turn suffixes are."""

    name = "inkling2"
    turn_end = "<|content_model_end_sampling|>"
    user_header = "<|message_user|>"
    tool_open, tool_close = "<|content_invoke_tool_json|>", "<|end_message|>"
    think_close = "<|end_message|>"
    stop_tokens = ("<|content_model_end_sampling|>", "<|endoftext|>")
    thinking_tokens = ("<|content_thinking|>",)
    image_token = "<|unused_200054|>"
    arguments_key = "args"
    follow_up_steps = (("<|message_model|>", "<|content_model_end_sampling|>"),)
    # The Think toggle as the template's reasoning effort. Off: no reasoning asked (and its content token suppressed). On: "medium" (0.7) and
    # not the template's default "high" (0.9): Deepy reasons before each of many small steps and every thought stays in a bounded context.
    reasoning_effort = {False: "none", True: "medium"}
    _effort_levels = {"none": "0", "minimal": "0.1", "low": "0.2", "medium": "0.7", "high": "0.9", "xhigh": "0.99", "max": "0.99"}

    def _generation_prefix(self, thinking_enabled: bool) -> str:
        # The template adds nothing: the model opens its own messages. With thinking on, the first one is opened for it, so that a turn starts
        # inside a thought as with the other format.
        return "<|message_model|><|content_thinking|>" if thinking_enabled else ""

    @staticmethod
    def _split_reasoning(content: str) -> tuple[str, str]:
        match = _THINK_BLOCK_RE.match(content)
        return ("", content) if match is None else (match.group(1), content[match.end():])

    def _assistant(self, message: dict[str, Any], thought_open: bool = False) -> str:
        """A model turn in history: its reasoning (reasoning_content, or a leading <think> block of the content), its text and its calls."""
        header = "<|message_model|>" + str(message.get("name") or "")
        content, tool_calls = message.get("content"), message.get("tool_calls") or []
        reasoning = str(message.get("reasoning_content") or "")
        if isinstance(content, str) and not reasoning:
            reasoning, content = self._split_reasoning(content)
        out = ""
        if thought_open:
            out += f"{reasoning}<|end_message|>"
        elif reasoning:
            out += f"{header}<|content_thinking|>{reasoning}<|end_message|>"
        if isinstance(content, str) and (content or not (tool_calls or reasoning or thought_open)):
            out += f"{header}<|content_text|>{content}<|end_message|>"
        for tool_call in tool_calls:
            function = tool_call["function"]
            arguments = ",".join(f"{json.dumps(key, ensure_ascii=False)}:{json.dumps(value, ensure_ascii=False, separators=(',', ':'))}" for key, value in sorted((function.get("arguments") or {}).items()))
            out += f'<|message_model|>{function["name"]}<|content_invoke_tool_json|>{{"name":{json.dumps(function["name"], ensure_ascii=False)},"args":{{{arguments}}}}}<|end_message|>'
        return out + "<|content_model_end_sampling|>"

    def _content(self, header: str, content) -> str:
        if isinstance(content, str):
            return f"{header}<|content_text|>{content.strip()}<|end_message|>"
        out = ""
        for part in content or []:
            if isinstance(part, str) or part.get("type", "text") in ("text", "input_text"):
                out += f"{header}<|content_text|>{part if isinstance(part, str) else part.get('text', '')}<|end_message|>"
            elif part["type"] in ("image", "input_image", "image_url"):
                out += f"{header}{self.image_run(part.get('image_tokens', 1))}<|end_message|>"
            else:
                raise ValueError(f"Unsupported content part type: {part['type']}")
        return out

    def render_text(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, add_generation_prompt: bool = False, thinking_enabled: bool = False) -> str:
        out = ""
        if tools:
            specs = []
            for tool in tools:
                function = tool.get("function", tool)
                specs.append({"description": function.get("description") or "", "name": function["name"], "parameters": function.get("parameters") or {}, "type": tool.get("type") or "function"})
            out += "<|message_system|>tool_declare<|content_xml|>" + json.dumps(specs, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "<|end_message|>"
        effort = f"<|message_system|><|content_text|>Thinking effort level: {self._effort_levels[self.reasoning_effort[bool(thinking_enabled)]]}<|end_message|>"
        effort_emitted = False
        call_names: dict[str, str] = {}  # tool results are named after their call: by id, else in the order of the calls
        pending_names: list[str] = []
        for message in messages:
            role = message["role"]
            if role != "system" and not effort_emitted:
                out, effort_emitted = out + effort, True
            if role == "assistant":
                pending_names = [tool_call["function"]["name"] for tool_call in message.get("tool_calls") or []]
                call_names.update({tool_call["id"]: tool_call["function"]["name"] for tool_call in message.get("tool_calls") or [] if tool_call.get("id")})
                out += self._assistant({**message, "content": message["content"].strip()} if isinstance(message.get("content"), str) else message)
                continue
            header = {"system": "<|message_system|>", "user": "<|message_user|>", "tool": "<|message_tool|>"}[role]
            if role == "tool":
                positional = pending_names.pop(0) if pending_names else ""
                header += str(message.get("name") or call_names.get(message.get("tool_call_id")) or ("" if message.get("tool_call_id") else positional))
            else:
                pending_names = []
            out += self._content(header, message.get("content"))
        if not effort_emitted:
            out += effort
        return out + (self._generation_prefix(bool(thinking_enabled)) if add_generation_prompt else "")

    def render_messages(self, tokenizer, messages, tools, add_generation_prompt, thinking_enabled) -> list[int]:
        return _encode(tokenizer, self.render_text(messages, tools, add_generation_prompt, thinking_enabled))

    def render_prompt(self, tokenizer, messages, thinking_enabled) -> str:
        # A one-shot prompt declares no tool and its answer is a text message: without reasoning, that message is opened here. Left to open
        # it, the model sometimes writes its answer as a tool call (seen with "name: value" user inputs, its reasoning suppressed).
        return self.render_text(messages, None, True, thinking_enabled) + ("" if thinking_enabled else "<|message_model|><|content_text|>")

    def render_user_turn(self, tokenizer, user_content, thinking_enabled) -> list[int]:
        user_content = str(user_content or "").strip()
        if len(user_content) == 0:
            return []
        return _encode(tokenizer, f"<|message_user|><|content_text|>{user_content}<|end_message|>" + self._generation_prefix(bool(thinking_enabled)))

    def render_tool_turn(self, tokenizer, tool_contents, thinking_enabled, tool_names=None) -> list[int]:
        tool_names = list(tool_names or []) + [""] * len(tool_contents or [])
        results = [(name, str(content or "").strip()) for name, content in zip(tool_names, tool_contents or []) if len(str(content or "").strip()) > 0]
        if len(results) == 0:
            return []
        suffix = "<|content_model_end_sampling|>" + "".join(f"<|message_tool|>{name}<|content_text|>{content}<|end_message|>" for name, content in results)
        return _encode(tokenizer, suffix + self._generation_prefix(bool(thinking_enabled)))

    def render_assistant_text(self, tokenizer, assistant_content, thinking_enabled, prompt_open) -> list[int]:
        assistant_content = str(assistant_content or "").strip()
        if len(assistant_content) == 0:
            return []
        return _encode(tokenizer, self._assistant({"content": assistant_content}, thought_open=bool(prompt_open and thinking_enabled)))

    def render_exchange(self, tokenizer, user_content, assistant_content) -> list[int]:
        return _encode(tokenizer, f"<|message_user|><|content_text|>{user_content}<|end_message|>" + self._assistant({"content": assistant_content}))

    def boundary_markers(self, phase, thinking_enabled) -> list[str]:
        return ["<|content_invoke_tool_json|>", *(["<|content_thinking|>"] if thinking_enabled else [])] if phase == "statement" else ["<|end_message|>"]

    def encode_action_text(self, tokenizer, text) -> list[int]:
        text = re.sub(r"\n*<think>\n?", "<|message_model|><|content_thinking|>", str(text or ""))
        return _encode(tokenizer, re.sub(r"\n?</think>\n*", "<|end_message|>", text))

    def image_run(self, image_tokens: int) -> str:
        return "<|content_image|>" + self.image_token * int(image_tokens)

    def system_prompt(self, text: str) -> str:
        return text.replace(" in Qwen tool-calling format", "")

    def tool_error(self, text: str) -> str:
        return text.replace("an arguments object", "an args object")

    @staticmethod
    def _opened_by_prompt(tokenizer, prompt_tail) -> bool:
        return len(prompt_tail) > 0 and int(prompt_tail[-1]) in _inkling_ids(tokenizer).kinds

    def messages(self, tokenizer, token_ids, prompt_tail=()) -> list[list]:
        """The messages of a sampled turn as [kind, text, closed, name], kind = "thinking", "text", "tool", "header" (a message whose content
        token is not sampled yet: its text is the name written so far), "end" (the end of the turn) or a content token this parser does not know."""
        ids = _inkling_ids(tokenizer)
        parts = [[ids.kinds[int(prompt_tail[-1])], "", False, ""]] if self._opened_by_prompt(tokenizer, prompt_tail) else []
        run: list[int] = []
        header = False

        def flush() -> str:
            text = _decode(tokenizer, run) if run else ""
            run.clear()
            return text

        def close_text() -> None:  # tokens outside a message, or the content of the open one
            text = flush()
            if parts and not parts[-1][2] and parts[-1][0] not in ("header", "end"):
                parts[-1][1] += text
            elif text:
                parts.append(["text", text, True, ""])

        for token_id in token_ids:
            token_id = int(token_id)
            if header and token_id in ids.kinds:
                parts.append([ids.kinds[token_id], "", False, flush()])
                header = False
            elif token_id == ids.message_model:
                close_text()
                header = True
            elif token_id == ids.end_message:
                close_text()
                if parts:
                    parts[-1][2] = True
                header = False
            elif token_id == ids.content_model_end_sampling:
                close_text()
                parts.append(["end", "", True, ""])
                header = False
            else:
                run.append(token_id)
        if header:
            parts.append(["header", flush(), False, ""])
        else:
            close_text()
        return parts

    @staticmethod
    def _call_text(payload: str, closed: bool, name: str) -> str:
        """A call's JSON with the key the turn text uses for its arguments; a complete call written another way is rebuilt."""
        converted, count = _ARGS_KEY_RE.subn(r'\1"arguments"\2', payload)
        if count or not closed:
            return converted
        try:
            parsed = json.loads(payload)
        except ValueError:
            return payload
        if not isinstance(parsed, dict) or "arguments" in parsed or not isinstance(parsed.get("args"), dict):
            return payload
        return json.dumps({"name": parsed.get("name", name), "arguments": parsed["args"]}, ensure_ascii=False)

    def decode(self, tokenizer, token_ids, prompt_tail=()) -> str:
        parts = self.messages(tokenizer, token_ids, prompt_tail)
        opened = self._opened_by_prompt(tokenizer, prompt_tail)
        text = ""
        for index, (kind, body, closed, name) in enumerate(parts):
            previous = parts[index - 1][0] if index else ""
            separator = "" if not text or text.endswith("\n\n") else "\n" if previous == kind == "tool" else "\n\n"
            if kind == "thinking":
                text += ("" if index == 0 and opened else separator + "<think>\n") + body + ("\n</think>\n\n" if closed else "")
            elif kind == "text":
                text += (separator if body and previous in ("text", "tool") else "") + body
            elif kind == "tool":
                text += separator + "<tool_call>\n" + self._call_text(body, closed, name) + ("\n</tool_call>" if closed else "")
            elif kind == "header":
                # after a tool call, the message being opened is read as the next call until its content token says otherwise
                text += "\n<tool_call>\n" if previous == "tool" else ""
            elif kind == "end":
                text += ChatFormat.turn_end
            else:
                text += kind + body
        return text


QWEN = ChatFormat()
FORMATS = {chat_format.name: chat_format for chat_format in (QWEN, Inkling2Format())}


def of(tokenizer) -> ChatFormat:
    """The format a tokenizer writes: the loader of a model that does not use the default one names it in tokenizer.chat_format."""
    return FORMATS[getattr(tokenizer, "chat_format", QWEN.name)]


def decode_response(tokenizer, response: dict | None, prompt_token_ids) -> str:
    """The text of an engine response to a one-shot prompt: its own decoding in the default format, else the format's reading of its tokens."""
    if response is None:
        return ""
    chat_format = of(tokenizer)
    return response.get("text", "") if chat_format is QWEN else chat_format.decode(tokenizer, response["token_ids"], prompt_token_ids[-1:])


def decode_completion(tokenizer, seq) -> str:
    """The response a sequence is generating, as turn text."""
    return of(tokenizer).decode(tokenizer, seq.completion_token_ids, seq.token_ids[max(0, seq.num_prompt_tokens - 1):seq.num_prompt_tokens])
