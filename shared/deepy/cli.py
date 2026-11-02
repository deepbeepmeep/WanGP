# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import json
import os
import re
import sys
import time
import traceback
from threading import Event, Thread
from typing import Any
import ctypes

try:
    import msvcrt
except Exception:  # pragma: no cover
    msvcrt = None

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.key_binding import KeyBindings
except Exception:  # pragma: no cover
    PromptSession = None
    KeyBindings = None

from shared.deepy import chat as assistant_chat, tool_settings as deepy_tool_settings, ui_settings as deepy_ui_settings
from shared.deepy.engine import get_or_create_assistant_session
from shared.deepy.gallery import Gallery as _VirtualGallery
from shared.deepy.runtime import GenerationRuntime, RuntimeCallbacks as DeepyCliCallbacks, RuntimeDeps as DeepyCliDeps


_USER32 = getattr(ctypes, "windll", None)
_USER32 = None if _USER32 is None else getattr(_USER32, "user32", None)
_VK_CONTROL = 0x11
_VK_LCONTROL = 0xA2
_VK_RCONTROL = 0xA3
_VK_S = 0x53
_DEEPY_LOGO = (
    "    ____                       ",
    "   / __ \\___  ___  ____  __  __",
    "  / / / / _ \\/ _ \\/ __ \\/ / / /",
    " / /_/ /  __/  __/ /_/ / /_/ / ",
    "/_____/\\___/\\___/ .___/\\__, /  ",
    "               /_/    /____/   ",
)
_TOOL_ALIASES = {
    "gen_image": "gen_image",
    "image": "gen_image",
    "image_gen": "gen_image",
    "generate_image": "gen_image",
    "edit_image": "edit_image",
    "edit": "edit_image",
    "image_edit": "edit_image",
    "gen_video": "gen_video",
    "video": "gen_video",
    "media_gen": "gen_video",
    "generate_media": "gen_video",
    "gen_video_with_speech": "gen_video_with_speech",
    "video_with_speech": "gen_video_with_speech",
    "talking_video": "gen_video_with_speech",
    "speech_video": "gen_video_with_speech",
    "gen_song": "gen_song",
    "song": "gen_song",
    "music": "gen_song",
    "gen_speech_from_description": "gen_speech_from_description",
    "speech_from_description": "gen_speech_from_description",
    "voice_description": "gen_speech_from_description",
    "gen_speech_from_sample": "gen_speech_from_sample",
    "speech_from_sample": "gen_speech_from_sample",
    "voice_clone": "gen_speech_from_sample",
    "gen_world": "gen_world",
    "world": "gen_world",
    "gen_3d": "gen_3d",
    "3d": "gen_3d",
    "texture_3d": "texture_3d",
    "texture": "texture_3d",
}


def _reconfigure_stdio() -> None:
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


class DeepyCliSession(GenerationRuntime):
    def __init__(self, deps: DeepyCliDeps):
        self._deps = deps
        self._state = self._build_state()
        self._session = get_or_create_assistant_session(self._state)
        self._gallery = _VirtualGallery(deps, self._state)
        self._last_status_text = ""
        self._assistant_live_print_state: dict[str, dict[str, str]] = {}
        self._interactive = False
        self._prompt_session = None
        self._turn_active = False
        self._turn_stop_requested = False
        self._turn_hotkey_stop: Event | None = None
        self._turn_hotkey_thread: Thread | None = None
        self._active_generation_client_id = ""


    def _print(self, text: str = "") -> None:
        print(text, flush=True)

    def _supports_terminal_formatting(self) -> bool:
        stream = getattr(sys, "stdout", None)
        return bool(getattr(stream, "isatty", lambda: False)())

    def _style_terminal_text(self, text: str, code: str) -> str:
        rendered = str(text or "")
        if not self._supports_terminal_formatting() or len(rendered) == 0:
            return rendered
        return f"\033[{code}m{rendered}\033[0m"

    def _italicize_terminal_block(self, text: str) -> str:
        lines = []
        for line in str(text or "").split("\n"):
            lines.append(self._style_terminal_text(line, "3") if len(line) > 0 else "")
        return "\n".join(lines)

    def _render_terminal_markdown(self, text: str) -> str:
        rendered_lines = []
        in_code_block = False
        for raw_line in str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            stripped = raw_line.strip()
            if stripped.startswith("```"):
                in_code_block = not in_code_block
                continue
            if in_code_block:
                rendered_lines.append(self._style_terminal_text(raw_line, "96"))
                continue
            line = raw_line
            heading_match = re.match(r"^\s{0,3}(#{1,6})\s+(.*)$", line)
            if heading_match is not None:
                heading_text = heading_match.group(2).strip()
                rendered_lines.append(self._style_terminal_text(heading_text, "1"))
                continue
            line = re.sub(r"\*\*([^*\n]+)\*\*", lambda match: self._style_terminal_text(match.group(1), "1"), line)
            line = re.sub(r"`([^`\n]+)`", lambda match: self._style_terminal_text(match.group(1), "96"), line)
            rendered_lines.append(line)
        return "\n".join(rendered_lines)

    def _print_status(self, text: str | None) -> None:
        text = str(text or "").strip()
        if len(text) == 0 or text == self._last_status_text:
            return
        self._last_status_text = text
        self._print(f"[Deepy] {text}")

    def _reset_status(self) -> None:
        self._last_status_text = ""

    def _emit_cli_callback(self, name: str, *args, first: bool = False, **kwargs) -> Any:
        callbacks = self._deps.callbacks if isinstance(self._deps.callbacks, DeepyCliCallbacks) else None
        if callbacks is None:
            return None if first else []
        try:
            return callbacks.emit_first(name, *args, **kwargs) if first else callbacks.emit(name, *args, **kwargs)
        except Exception as exc:
            self._print(f"[ERROR] Deepy CLI callback '{name}' failed: {exc}")
            return None if first else []

    def _abort_generation(self, client_id: str = "") -> None:
        self._emit_cli_callback("abort_generation", self._state, str(client_id or "").strip(), first=True)

    def _request_stop_current_turn(self) -> bool:
        if not self._turn_active:
            return False
        self._session.interrupt_requested = True
        if not self._turn_stop_requested:
            self._turn_stop_requested = True
            self._print("[Deepy] Stop requested.")
            self._emit_cli_callback("stop_requested", self, str(self._active_generation_client_id or "").strip())
        client_id = str(self._active_generation_client_id or "").strip()
        if len(client_id) > 0 or self._state["gen"].get("in_progress", False):
            self._abort_generation(client_id)
        return True

    def _ctrl_s_pressed(self) -> bool:
        if _USER32 is None:
            return False
        try:
            ctrl_down = any(
                bool(_USER32.GetAsyncKeyState(vk_code) & 0x8000)
                for vk_code in (_VK_CONTROL, _VK_LCONTROL, _VK_RCONTROL)
            )
            return ctrl_down and bool(_USER32.GetAsyncKeyState(_VK_S) & 0x8000)
        except Exception:
            return False

    def _monitor_turn_shortcuts(self, stop_event: Event) -> None:
        if not self._interactive or msvcrt is None:
            return
        ctrl_s_active = False
        while not stop_event.is_set():
            hotkey_down = self._ctrl_s_pressed()
            if hotkey_down and not ctrl_s_active:
                ctrl_s_active = True
                self._request_stop_current_turn()
            elif not hotkey_down:
                ctrl_s_active = False
            try:
                if not msvcrt.kbhit():
                    time.sleep(0.05)
                    continue
                key = msvcrt.getwch()
            except Exception:
                return
            if key in {"\x00", "\xe0"}:
                try:
                    if msvcrt.kbhit():
                        msvcrt.getwch()
                except Exception:
                    return
                continue
            if key == "\x13":
                self._request_stop_current_turn()

    def _start_turn_shortcut_monitor(self) -> None:
        self._turn_hotkey_stop = None
        self._turn_hotkey_thread = None
        if not self._interactive or msvcrt is None:
            return
        stop_event = Event()
        thread = Thread(target=self._monitor_turn_shortcuts, args=(stop_event,), daemon=True, name="DeepyCliStopHotkey")
        self._turn_hotkey_stop = stop_event
        self._turn_hotkey_thread = thread
        thread.start()

    def _stop_turn_shortcut_monitor(self) -> None:
        stop_event = self._turn_hotkey_stop
        thread = self._turn_hotkey_thread
        self._turn_hotkey_stop = None
        self._turn_hotkey_thread = None
        if stop_event is not None:
            stop_event.set()
        if thread is not None:
            thread.join(timeout=0.2)

    def _assistant_print_state(self, message_id: str) -> dict[str, str]:
        return self._assistant_live_print_state.setdefault(str(message_id or "").strip(), {"reasoning": "", "content": ""})

    def _print_assistant_block(self, text: str, *, italic: bool = False, show_prefix: bool = True) -> None:
        rendered = self._render_terminal_markdown(text)
        if italic:
            rendered = self._italicize_terminal_block(rendered)
        if not show_prefix:
            self._print(rendered)
        elif "\n" not in rendered:
            self._print(f"Deepy> {rendered}")
        else:
            self._print("Deepy>")
            self._print(rendered)

    def _print_assistant_delta(self, message_id: str, text: str, *, field: str, italic: bool = False, show_prefix: bool = True) -> bool:
        normalized_text = str(text or "").strip()
        if len(normalized_text) == 0:
            return False
        state = self._assistant_print_state(message_id)
        previous = str(state.get(field, "") or "")
        if normalized_text == previous:
            return False
        if len(previous) > 0 and normalized_text.startswith(previous):
            delta = normalized_text[len(previous):].lstrip("\n")
        else:
            delta = normalized_text
        state[field] = normalized_text
        if len(delta.strip()) == 0:
            return False
        self._print_assistant_block(delta, italic=italic, show_prefix=show_prefix)
        return True

    def _flush_live_assistant_reasoning(self) -> None:
        if int(getattr(self._deps.controller, "get_verbose_level", lambda: 0)() or 0) > 1:
            return
        for record in self._session.chat_transcript:
            if str(record.get("role", "")).strip() != "assistant":
                continue
            message_id = str(record.get("id", "") or "").strip()
            reasoning_text = assistant_chat.get_message_reasoning_content(self._session, message_id).strip()
            self._print_assistant_delta(message_id, reasoning_text, field="reasoning", italic=True, show_prefix=False)

    def _iter_tool_blocks(self, record: dict[str, Any]) -> list[dict[str, Any]]:
        blocks = record.get("blocks", None)
        if isinstance(blocks, list):
            return [block for block in blocks if isinstance(block, dict) and str(block.get("type", "")).strip().lower() == "tool"]
        legacy_tools = record.get("tools", None)
        if isinstance(legacy_tools, list):
            return [dict(block or {}, type="tool") for block in legacy_tools if isinstance(block, dict)]
        return []

    def _print_tool_outputs(self, record: dict[str, Any]) -> bool:
        printed_any = False
        for block in self._iter_tool_blocks(record):
            result = block.get("result", None)
            if not isinstance(result, dict):
                continue
            output_file = str(result.get("output_file", "") or "").strip()
            if len(output_file) == 0:
                continue
            tool_label = str(block.get("label", "") or block.get("name", "") or "Tool").strip()
            media_id = str(result.get("media_id", "") or "").strip()
            prefix = f"{tool_label} -> {media_id}" if len(media_id) > 0 else tool_label
            self._print(f"Deepy> {prefix}: {os.path.abspath(output_file)}")
            printed_any = True
        return printed_any

    def _consume_chat_payload(self, payload: str) -> None:
        payload_text = str(payload or "").strip()
        if len(payload_text) == 0:
            return
        try:
            envelope = json.loads(payload_text)
        except Exception:
            return
        batch = envelope.get("batch", None)
        if isinstance(batch, list):
            for item in batch:
                self._consume_chat_payload(json.dumps(item, ensure_ascii=False))
            return
        event = envelope.get("event", envelope)
        if not isinstance(event, dict):
            return
        if event.get("type") == "status":
            status = event.get("status", None)
            if isinstance(status, dict) and status.get("visible", False) and len(str(status.get("text", "")).strip()) > 0:
                status_text = str(status.get("text", "")).strip()
                if status_text.startswith("Using "):
                    self._flush_live_assistant_reasoning()
                self._print_status(status.get("text", ""))
            else:
                self._reset_status()




    def _send_cmd(self, cmd: str, data: Any = None) -> None:
        if cmd == "chat_output":
            self._consume_chat_payload(str(data or ""))
        elif cmd == "load_queue_trigger":
            self._process_inline_queue(data)
        elif cmd == "refresh_gallery":
            self._gallery.sync_refresh_path(data)
        elif cmd == "abort_client_id":
            self._abort_generation(str(data or ""))
        elif cmd == "error":
            self._print(f"[ERROR] {data}")

    def _reset_conversation(self) -> None:
        self._deps.controller.reset_ai(self._state)
        self._assistant_live_print_state.clear()
        self._reset_status()

    def _saved_sessions(self) -> list[dict[str, Any]]:
        return list(self._deps.controller.list_saved_sessions() or [])

    def _resolve_saved_session(self, reference: str) -> dict[str, Any] | None:
        sessions = self._saved_sessions()
        value = str(reference or "").strip()
        if value.isdigit() and 1 <= int(value) <= len(sessions):
            return sessions[int(value) - 1]
        folded = value.casefold()
        exact = [item for item in sessions if str(item.get("id", "")).casefold() == folded or str(item.get("title", "")).casefold() == folded]
        if len(exact) == 1:
            return exact[0]
        partial = [item for item in sessions if folded and folded in str(item.get("title", "")).casefold()]
        return partial[0] if len(partial) == 1 else None

    def _list_saved_sessions(self) -> None:
        sessions = self._saved_sessions()
        if not sessions:
            self._print("No saved Deepy sessions.")
            return
        active_id = str(self._session.storage_session_id or "")
        for index, item in enumerate(sessions, 1):
            marker = " *" if str(item.get("id", "")) == active_id else ""
            self._print(f"  {index}. {item.get('title', 'Deepy session')} [{item.get('id', '')}]{marker}")

    def _resume_saved_session(self, reference: str) -> None:
        if not self._deps.controller.multi_session_enabled():
            self._print("[ERROR] Enable multi-session mode in Deepy Settings and restart WanGP first.")
            return
        selected = self._resolve_saved_session(reference)
        if selected is None:
            self._print("[ERROR] Select a unique session by number, exact title, or session id. Use /sessions to list them.")
            return
        try:
            result = self._deps.controller.resume_saved_session(self._state, str(selected.get("id", "")))
        except Exception as exc:
            self._print(f"[ERROR] {exc}")
            return
        self._assistant_live_print_state.clear()
        self._reset_status()
        self._print(f"Resumed {selected.get('title', 'Deepy session')} ({result['prefill_tokens']:,} context tokens; {result['injected']} Gallery item(s) injected).")
        for warning in result.get("warnings", []):
            self._print(f"[WARN] {warning}")
        if self._session.pending_action_replay is not None:
            transcript_start = len(self._session.chat_transcript)
            self._print("Replaying the interrupted action from its beginning...")
            for _update in self._deps.controller.resume_restored_action(self._state, command_callback=self._send_cmd):
                pass
            self._print_new_assistant_messages(transcript_start)

    def _start_new_saved_session(self) -> None:
        if not self._deps.controller.multi_session_enabled():
            self._print("[ERROR] Enable multi-session mode in Deepy Settings and restart WanGP first.")
            return
        try:
            self._deps.controller.start_new_session(self._state)
        except Exception as exc:
            self._print(f"[ERROR] {exc}")
            return
        self._assistant_live_print_state.clear()
        self._reset_status()
        self._print("New Deepy session ready. Its folder will be created with the first request.")

    def _print_new_assistant_messages(self, transcript_start: int) -> None:
        printed_any = False
        show_reasoning = int(getattr(self._deps.controller, "get_verbose_level", lambda: 0)() or 0) <= 1
        for record in self._session.chat_transcript[transcript_start:]:
            if str(record.get("role", "")).strip() != "assistant":
                continue
            message_id = str(record.get("id", "") or "").strip()
            reasoning_text = assistant_chat.get_message_reasoning_content(self._session, message_id).strip() if show_reasoning else ""
            text = assistant_chat.get_message_content(self._session, message_id).strip()
            if self._print_assistant_delta(message_id, reasoning_text, field="reasoning", italic=True, show_prefix=False):
                printed_any = True
                if len(text) > 0:
                    self._print()
            if self._print_assistant_delta(message_id, text, field="content", italic=False):
                printed_any = True
            if self._print_tool_outputs(record):
                printed_any = True
        if not printed_any:
            self._print("Deepy> (no final response)")

    def _get_tool_ui_settings(self) -> dict[str, Any]:
        if isinstance(self._session.tool_ui_settings, dict) and len(self._session.tool_ui_settings) > 0:
            return deepy_ui_settings.normalize_assistant_tool_ui_settings(**self._session.tool_ui_settings)
        return deepy_ui_settings.normalize_assistant_tool_ui_settings()

    def _update_tool_ui_settings(self, **changes) -> dict[str, Any]:
        settings = self._get_tool_ui_settings()
        settings.update(changes)
        self._deps.controller.update_tool_ui_settings(self._state, **settings)
        return self._get_tool_ui_settings()

    def _resolve_tool_name(self, value: str) -> str | None:
        return _TOOL_ALIASES.get(str(value or "").strip().lower())

    def _parse_dimensions(self, text: str) -> tuple[int, int] | None:
        cleaned = str(text or "").strip().lower().replace(",", " ").replace("x", " ")
        parts = [part for part in cleaned.split() if len(part) > 0]
        if len(parts) != 2:
            return None
        try:
            return int(parts[0]), int(parts[1])
        except Exception:
            return None

    def _format_template_value(self, value: str) -> str:
        resolved = str(value or "").strip()
        if len(resolved) == 0:
            return "(unset)"
        if os.path.isfile(resolved):
            return os.path.abspath(resolved)
        return resolved

    def _tool_settings_lines(self) -> list[str]:
        settings = self._get_tool_ui_settings()
        return [
            f"Template properties: {'on' if settings['use_template_properties'] else 'off'}",
            f"Default size: {settings['width']}x{settings['height']}",
            f"Video frames: {settings['num_frames']}",
            f"Audio duration: {settings['audio_duration']} seconds",
            f"Seed: {settings['seed']}",
            *(f"{tool} template: {self._format_template_value(settings[key])}" for tool, key in deepy_ui_settings.TEMPLATE_TOOL_UI_KEY.items()),
        ]

    def _print_tool_settings(self) -> None:
        for line in self._tool_settings_lines():
            self._print(line)

    def _build_prompt_session(self, input=None, output=None):
        if PromptSession is None or KeyBindings is None:
            return None
        bindings = KeyBindings()

        @bindings.add("enter")
        def _submit(event):
            event.current_buffer.validate_and_handle()

        @bindings.add("c-j")
        def _newline_ctrl_j(event):
            event.current_buffer.insert_text("\n")

        @bindings.add("escape", "c-j")
        def _newline_ctrl_enter_windows(event):
            event.current_buffer.insert_text("\n")

        @bindings.add("escape", "enter")
        def _newline_escape_enter(event):
            event.current_buffer.insert_text("\n")

        return PromptSession(
            input=input,
            output=output,
            multiline=True,
            key_bindings=bindings,
            prompt_continuation=lambda width, _line_no, _wrap_count: "... ".rjust(width),
        )

    def _read_line(self) -> str:
        if not self._interactive:
            return input("")
        if self._prompt_session is None:
            self._prompt_session = self._build_prompt_session()
        if self._prompt_session is None:
            return input("deepy> ")
        return self._prompt_session.prompt("deepy> ")

    def ask(self, text: str) -> None:
        prompt = str(text or "").strip()
        if len(prompt) == 0:
            return
        if not self._deps.controller.is_available():
            self._print(self._deps.controller.requirement_error_text())
            return
        transcript_start = len(self._session.chat_transcript)
        self._deps.controller.begin_direct_request(self._state, prompt)
        self._reset_status()
        tools = self._deps.controller.create_tools(self._state, self._send_cmd, session=self._session)
        completed = False
        self._turn_active = True
        self._turn_stop_requested = False
        self._active_generation_client_id = ""
        self._session.worker_active = True
        self._session.interrupt_requested = False
        self._emit_cli_callback("turn_started", self, prompt)
        self._start_turn_shortcut_monitor()
        try:
            self._deps.controller.run_assistant_prompt_turn(self._state, None, "AK", [prompt], 0, override_profile=3.5, send_cmd=self._send_cmd, tools=tools)
            completed = True
        except Exception as exc:
            self._print(f"[ERROR] {exc}")
        finally:
            self._stop_turn_shortcut_monitor()
            self._turn_active = False
            self._turn_stop_requested = False
            self._active_generation_client_id = ""
            self._session.worker_active = False
            self._session.interrupt_requested = False
            self._reset_status()
            self._emit_cli_callback("turn_finished", self, prompt, completed=completed)
        if completed:
            self._print_new_assistant_messages(transcript_start)

    def _handle_add_command(self, value: str, preferred_type: str = "any") -> None:
        try:
            result = self._gallery.add_path(value, preferred_type=preferred_type)
        except Exception as exc:
            self._print(f"[ERROR] {exc}")
            return
        record = result.get("record", None)
        action = "Added" if result.get("added", False) else "Selected"
        if isinstance(record, dict):
            self._print(f"{action} {record.get('media_id')}: {record.get('label', '')}")
        else:
            self._print(f"{action} file.")

    def _handle_command(self, line: str) -> bool:
        command, _, rest = str(line or "").strip().partition(" ")
        argument = rest.strip()
        command = command.lower()
        if command in {"/quit", "/exit"}:
            return False
        if command == "/help":
            self._print("Commands:")
            self._print("  /add <path>     Add and select an image, video, or audio file")
            self._print("  /image <path>   Add and select an image file")
            self._print("  /video <path>   Add and select a video file")
            self._print("  /audio <path>   Add and select an audio file")
            self._print("  /list [scope]   List known media; scope: all, media, image, video, audio")
            self._print("  /media [scope]  Alias for /list")
            self._print("  /select <ref>   Select media by id, index, or name fragment")
            self._print("  /select-video <media_id>  Select a video by media id")
            self._print("  /time <secs>    Set the selected video playback time")
            self._print("  /frame [index]  Show or set the selected video frame (0-based)")
            self._print("  /selected       Show the selected media")
            self._print("  /selected-video Show the selected video media id")
            self._print("  /settings       Show current CLI generation settings")
            self._print("  /size [WxH]     Show or set default generation size and disable template properties")
            self._print("  /frames [count] Show or set default gen_video frame count and disable template properties")
            self._print("  /duration [seconds]  Show or set default audio duration and disable template properties")
            self._print("  /seed [value]   Show or set default generation seed and disable template properties")
            self._print("  /template <tool> <variant>  Set the preset for any Deepy generation tool")
            self._print("  /templates [tool]  List available preset variants")
            self._print("  /template-props [on|off]  Show or toggle template resolution/frame properties")
            self._print("  /sessions       List saved sessions (* marks the active session)")
            self._print("  /resume <ref>   Resume a saved session by number, title, or id")
            self._print("  /new            Start a new persistent session")
            self._print("  /reset          Clear the Deepy conversation but keep media")
            self._print("  /clear-media    Remove all virtual gallery media")
            self._print("  /quit           Exit the session")
            if self._interactive:
                self._print("Prompt entry:")
                self._print("  Enter           Send the current prompt")
                self._print("  Ctrl+Enter      Insert a newline on Windows terminals that expose it")
                self._print("  Ctrl+J          Insert a newline fallback")
                self._print("  Alt+Enter       Insert a newline")
                self._print("  Ctrl+S          Stop the current Deepy turn while it is running")
                self._print("  Shift+Enter     Not available here; the console reports it as plain Enter")
            return True
        if command == "/add":
            self._handle_add_command(argument, "any")
            return True
        if command == "/image":
            self._handle_add_command(argument, "image")
            return True
        if command == "/video":
            self._handle_add_command(argument, "video")
            return True
        if command == "/audio":
            self._handle_add_command(argument, "audio")
            return True
        if command in {"/list", "/media"}:
            for one_line in self._gallery.list_lines(argument or "all"):
                self._print(one_line)
            return True
        if command == "/select":
            record = self._gallery.select(argument)
            if record is None:
                self._print("Unable to resolve media selection.")
            else:
                self._print(f"Selected {record.get('media_id')}: {record.get('label', '')}")
            return True
        if command == "/select-video":
            record = self._gallery.select(argument, media_type="video")
            if record is None:
                self._print("Unable to resolve video selection.")
            else:
                self._print(f"Selected video {record.get('media_id')}: {record.get('label', '')}")
            return True
        if command == "/time":
            try:
                seconds = float(argument)
            except Exception:
                self._print("Provide a numeric time in seconds.")
            else:
                self._print(self._gallery.set_selected_time(seconds))
            return True
        if command == "/frame":
            if len(argument) == 0:
                self._print(self._gallery.selected_frame_summary())
                return True
            try:
                frame_no = int(argument)
            except Exception:
                self._print("Provide a numeric frame index.")
            else:
                self._print(self._gallery.set_selected_frame(frame_no))
            return True
        if command == "/selected":
            self._print(self._gallery.selected_summary())
            return True
        if command == "/selected-video":
            self._print(self._gallery.selected_video_summary())
            return True
        if command == "/settings":
            self._print_tool_settings()
            return True
        if command in {"/size", "/resolution"}:
            if len(argument) == 0:
                settings = self._get_tool_ui_settings()
                self._print(f"Default size: {settings['width']}x{settings['height']} (template properties {'on' if settings['use_template_properties'] else 'off'})")
                return True
            dimensions = self._parse_dimensions(argument)
            if dimensions is None:
                self._print("Use /size <width>x<height>.")
                return True
            try:
                settings = self._update_tool_ui_settings(width=dimensions[0], height=dimensions[1], use_template_properties=False)
            except Exception as exc:
                self._print(f"[ERROR] {exc}")
            else:
                self._print(f"Default size set to {settings['width']}x{settings['height']}. Template properties disabled.")
            return True
        if command == "/frames":
            if len(argument) == 0:
                settings = self._get_tool_ui_settings()
                self._print(f"Video frames: {settings['num_frames']} (template properties {'on' if settings['use_template_properties'] else 'off'})")
                return True
            try:
                frame_count = int(argument)
            except Exception:
                self._print("Use /frames <count>.")
                return True
            try:
                settings = self._update_tool_ui_settings(num_frames=frame_count, use_template_properties=False)
            except Exception as exc:
                self._print(f"[ERROR] {exc}")
            else:
                self._print(f"Default gen_video frame count set to {settings['num_frames']}. Template properties disabled.")
            return True
        if command in {"/duration", "/audio-duration"}:
            if len(argument) == 0:
                settings = self._get_tool_ui_settings()
                self._print(f"Audio duration: {settings['audio_duration']} seconds (template properties {'on' if settings['use_template_properties'] else 'off'})")
                return True
            try:
                duration = int(argument)
            except Exception:
                self._print("Use /duration <seconds>.")
                return True
            try:
                settings = self._update_tool_ui_settings(audio_duration=duration, use_template_properties=False)
            except Exception as exc:
                self._print(f"[ERROR] {exc}")
            else:
                self._print(f"Default audio duration set to {settings['audio_duration']} seconds. Template properties disabled.")
            return True
        if command == "/seed":
            if len(argument) == 0:
                settings = self._get_tool_ui_settings()
                self._print(f"Seed: {settings['seed']} (template properties {'on' if settings['use_template_properties'] else 'off'})")
                return True
            try:
                seed = int(argument)
            except Exception:
                self._print("Use /seed <value>. Use -1 for random.")
                return True
            try:
                settings = self._update_tool_ui_settings(seed=seed, use_template_properties=False)
            except Exception as exc:
                self._print(f"[ERROR] {exc}")
            else:
                self._print(f"Default seed set to {settings['seed']}. Template properties disabled.")
            return True
        if command in {"/template", "/preset"}:
            tool_name, _, tool_value = argument.partition(" ")
            resolved_tool = self._resolve_tool_name(tool_name)
            if resolved_tool is None or len(tool_value.strip()) == 0:
                self._print(f"Use /template <{'|'.join(deepy_tool_settings.GENERATION_TOOL_IDS)}> <variant>.")
                return True
            try:
                key = deepy_ui_settings.TEMPLATE_TOOL_UI_KEY[resolved_tool]
                value = self._update_tool_ui_settings(**{key: tool_value.strip()})[key]
            except Exception as exc:
                self._print(f"[ERROR] {exc}")
            else:
                self._print(f"{resolved_tool} template set to {self._format_template_value(value)}")
            return True
        if command == "/templates":
            if len(argument) == 0:
                for tool_name in deepy_tool_settings.GENERATION_TOOL_IDS:
                    variants = deepy_tool_settings.list_tool_variants(tool_name)
                    suffix = ", ".join(variants) if variants else "(none)"
                    self._print(f"{tool_name}: {suffix}")
                return True
            resolved_tool = self._resolve_tool_name(argument)
            if resolved_tool is None:
                self._print(f"Use /templates [{'|'.join(deepy_tool_settings.GENERATION_TOOL_IDS)}].")
                return True
            variants = deepy_tool_settings.list_tool_variants(resolved_tool)
            self._print(f"{resolved_tool}: {', '.join(variants) if variants else '(none)'}")
            return True
        if command == "/template-props":
            if len(argument) == 0:
                settings = self._get_tool_ui_settings()
                self._print(f"Template properties are {'on' if settings['use_template_properties'] else 'off'}.")
                return True
            normalized = argument.strip().lower()
            if normalized not in {"on", "off"}:
                self._print("Use /template-props on|off.")
                return True
            try:
                settings = self._update_tool_ui_settings(use_template_properties=normalized == "on")
            except Exception as exc:
                self._print(f"[ERROR] {exc}")
            else:
                self._print(f"Template properties {'enabled' if settings['use_template_properties'] else 'disabled'}.")
            return True
        if command == "/sessions":
            self._list_saved_sessions()
            return True
        if command == "/resume":
            self._resume_saved_session(argument)
            return True
        if command == "/new":
            self._start_new_saved_session()
            return True
        if command == "/reset":
            self._reset_conversation()
            settings = self._deps.controller.get_session_ui_settings()
            started_new = bool(settings["effective_multi_session"] and settings["reset_mode"] == "new_session")
            self._print("New Deepy session ready." if started_new else "Deepy conversation reset.")
            return True
        if command == "/clear-media":
            self._gallery.clear_media()
            self._print("Virtual galleries cleared.")
            return True
        self._print(f"Unknown command: {command}. Use /help.")
        return True

    def run(self) -> int:
        if not self._deps.controller.is_available():
            self._print(self._deps.controller.requirement_error_text())
            return 1
        self._interactive = bool(getattr(__import__("sys").stdin, "isatty", lambda: False)())
        for line in _DEEPY_LOGO:
            self._print(line)
        preload_runtime = getattr(self._deps.controller, "preload_cli_runtime", None)
        if callable(preload_runtime):
            self._print("[Deepy] Preloading runtime...")
            try:
                preload_result = preload_runtime(self._state, override_profile=3.5)
            except Exception as exc:
                self._print(f"[ERROR] Deepy preload failed: {exc}")
                return 1
            warmed_vllm = bool((preload_result or {}).get("warmed_vllm", False)) if isinstance(preload_result, dict) else False
            self._print("[Deepy] Prompt enhancer and vLLM are ready." if warmed_vllm else "[Deepy] Prompt enhancer is ready.")
        self._print("Deepy CLI session. Use /help for commands.")
        if self._interactive and PromptSession is not None:
            self._print("Multiline input: Enter sends, Ctrl+Enter or Alt+Enter inserts a newline, Ctrl+S stops the active turn.")
        while True:
            try:
                line = self._read_line()
            except EOFError:
                self._print()
                return 0
            except KeyboardInterrupt:
                self._print()
                return 130
            line = str(line or "").strip()
            if len(line) == 0:
                continue
            if line.startswith("/"):
                if not self._handle_command(line):
                    return 0
                continue
            self.ask(line)


def run_deepy_cli_session(deps: DeepyCliDeps) -> int:
    _reconfigure_stdio()
    return DeepyCliSession(deps).run()


__all__ = ["DeepyCliCallbacks", "DeepyCliDeps", "DeepyCliSession", "run_deepy_cli_session"]
