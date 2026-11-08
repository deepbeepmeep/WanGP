# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import html
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import gradio as gr

from shared.deepy import tool_settings as deepy_tool_settings
from shared.deepy.config import DEEPY_TYPE_KEY, DEEPY_MULTI_SESSION_CHOICES, DEEPY_MULTI_SESSION_SELECTABLE, get_deepy_config_value, normalize_deepy_type, normalize_deepy_multi_session, normalize_deepy_session_mode
from shared.deepy import ui_settings as deepy_ui_settings
from shared.deepy import session_store
from shared.gradio import assistant_chat


_TEMPLATE_TOOL_LAYOUT = deepy_ui_settings.TEMPLATE_TOOL_LAYOUT
_TEMPLATE_TOOL_ORDER = tuple(tool_name for row in _TEMPLATE_TOOL_LAYOUT for tool_name in row)
_TEMPLATE_TOOL_UI_KEY = deepy_ui_settings.TEMPLATE_TOOL_UI_KEY
_TEMPLATE_ADD_SELECTION_ERROR = "Select your saved settings in the preset list at the top of the generator first (type a name and press Save there to create them), then press + again"
_TEMPLATE_DELETE_BUILTIN_ERROR = "You cant delete a Built in Template"
_TEMPLATE_CAPTURE_JS = """() => {
  const selection = window.WAC && typeof window.WAC.getWanGpSettingsSelection === 'function'
    ? window.WAC.getWanGpSettingsSelection()
    : { value: '', label: '' };
  return [selection.value || '', selection.label || ''];
}"""


@dataclass(slots=True)
class DeepyTemplateToolControl:
    tool_name: str
    dropdown: Any
    add_btn: Any
    delete_btn: Any


@dataclass(slots=True)
class DeepyChatUI:
    dock: Any
    launcher_host: Any
    panel: Any
    settings_launcher_host: Any
    settings_save_btn: Any
    settings_panel: Any
    html_output: Any
    chat_event: Any
    sync_btn: Any
    submission_id: Any
    busy_queue_request: Any
    busy_queue_submission_id: Any
    busy_queue_btn: Any
    steer_request: Any
    steer_submission_id: Any
    steer_btn: Any
    queued_action_input: Any
    queued_action_btn: Any
    welcome_session_input: Any
    welcome_session_btn: Any
    session_resume_state: Any
    session_prefill_btn: Any
    session_refresh_btn: Any
    stats_output: Any
    pause_btn: Any
    stop_btn: Any
    request: Any
    ask_btn: Any
    reset_btn: Any
    multi_session: Any
    multi_session_options: Any
    session_workspace_picker: Any
    mono_session_options: Any
    session_dropdown: Any
    session_resume_btn: Any
    session_rename_btn: Any
    session_duplicate_btn: Any
    session_export_btn: Any
    session_export_file: Any
    session_import_file: Any
    session_delete_btn: Any
    session_gallery_media_mode: Any
    session_reset_mode: Any
    session_status: Any
    compact_actions: Any
    auto_cancel_queue_tasks: Any
    separate_requests_with_empty_line: Any
    use_template_properties: Any
    model_speed: Any
    model_size: Any
    override_height: Any
    override_width: Any
    override_num_frames: Any
    override_audio_duration: Any
    override_seed: Any
    template_controls: tuple[DeepyTemplateToolControl, ...]
    template_selection_history: Any
    template_modal_state: Any
    captured_lset_value: Any
    captured_lset_label: Any
    template_modal: Any
    template_modal_title: Any
    template_modal_body: Any
    template_modal_input: Any
    template_modal_yes_btn: Any
    template_modal_no_btn: Any
    template_modal_close_btn: Any


@dataclass(slots=True)
class DeepyChatHandlers:
    prepare_request_context: Callable[[Any, Any, Any], Any]
    update_tool_ui_settings: Callable[..., Any]
    store_selected_video_time: Callable[[Any, Any], Any]
    ask_ai: Callable[[Any, str], Any]
    enqueue_ai: Callable[[Any, str], Any]
    stop_ai: Callable[..., Any]
    reset_ai: Callable[..., Any]
    get_session_ui_settings: Callable[[], Any]
    update_session_ui_settings: Callable[..., Any]
    list_saved_sessions: Callable[[], Any]
    resume_saved_session: Callable[[Any, str], Any]
    prefill_restored_session_context: Callable[..., Any]
    rename_saved_session: Callable[[Any, str, str], Any]
    duplicate_saved_session: Callable[[Any, str], Any]
    export_saved_session: Callable[[Any, str], Any]
    import_saved_session: Callable[[str], Any]
    delete_saved_session: Callable[[Any, str], Any]


def _tool_values_from_inputs(*current_templates: Any) -> dict[str, Any]:
    """{task: template} of the template dropdowns' values (template_controls order: _TEMPLATE_TOOL_ORDER)."""
    return dict(zip(_TEMPLATE_TOOL_ORDER, current_templates))


def _tool_values_from_ui_settings(tool_ui_state: dict[str, Any]) -> dict[str, Any]:
    return {tool_name: tool_ui_state[_TEMPLATE_TOOL_UI_KEY[tool_name]] for tool_name in _TEMPLATE_TOOL_ORDER}


def _normalize_tool_variant(tool_name: str, value: Any) -> str:
    resolved = deepy_tool_settings.find_tool_variant(tool_name, value)
    if resolved is not None:
        return resolved
    try:
        fallback = deepy_tool_settings.get_default_tool_variant(tool_name)
    except Exception:
        variants = deepy_tool_settings.list_tool_variants(tool_name)
        fallback = variants[0] if len(variants) > 0 else ""
    return str(fallback or "")


def _build_template_selection_history(tool_values: dict[str, Any]) -> dict[str, dict[str, str]]:
    history: dict[str, dict[str, str]] = {}
    for tool_name in _TEMPLATE_TOOL_ORDER:
        current = _normalize_tool_variant(tool_name, tool_values.get(tool_name))
        history[tool_name] = {"current": current, "previous": current}
    return history


def _normalize_template_selection_history(history: Any, tool_values: dict[str, Any]) -> dict[str, dict[str, str]]:
    raw_history = history if isinstance(history, dict) else {}
    normalized: dict[str, dict[str, str]] = {}
    for tool_name in _TEMPLATE_TOOL_ORDER:
        current = _normalize_tool_variant(tool_name, tool_values.get(tool_name))
        previous = None
        record = raw_history.get(tool_name)
        if isinstance(record, dict):
            previous = deepy_tool_settings.find_tool_variant(tool_name, record.get("previous"))
        if previous is None:
            previous = current
        normalized[tool_name] = {"current": current, "previous": str(previous or current or "")}
    return normalized


def _modal_title_html(tool_name: str) -> str:
    display_name = deepy_tool_settings.TOOL_DISPLAY_NAMES.get(tool_name, tool_name.replace("_", " ").title())
    return (
        "<div class='chat__template-modal-titlebar'>"
        f"<div class='chat__template-modal-heading'>{html.escape(display_name)} Tool</div>"
        "</div>"
    )


def _session_modal_title_html(title: str) -> str:
    return (
        "<div class='chat__template-modal-titlebar'>"
        f"<div class='chat__template-modal-heading'>{html.escape(title)}</div>"
        "</div>"
    )


def _settings_title_html() -> str:
    return (
        "<div class='chat__template-modal-titlebar'>"
        "<div class='chat__template-modal-heading'>Deepy Settings</div>"
        "</div>"
    )


def _tool_display_name(tool_name: str) -> str:
    return deepy_tool_settings.TOOL_DISPLAY_NAMES.get(tool_name, tool_name.replace("_", " ").title())


def _modal_context_html(label: str, value: str) -> str:
    return (
        "<div class='chat__template-modal-context'>"
        f"<div class='chat__template-modal-context-label'>{html.escape(label)}</div>"
        f"<div class='chat__template-modal-context-value'>{html.escape(value)}</div>"
        "</div>"
    )


def _wangp_settings_placeholders() -> set[str]:
    get_new_preset_msg = getattr(sys.modules.get("__main__"), "get_new_preset_msg", None)
    if not callable(get_new_preset_msg):
        return set()
    placeholders: set[str] = set()
    for advanced in (True, False):
        label = str(get_new_preset_msg(advanced) or "").strip()
        if len(label) > 0:
            placeholders.add(label)
    return placeholders


def _current_wangp_settings_context_html(value: str) -> str:
    selected = str(value or "").strip()
    if len(selected) == 0 or selected in _wangp_settings_placeholders():
        return ""
    return _modal_context_html("Current WanGP Settings", selected)


def _modal_message_html(message: str, *, tone: str = "info") -> str:
    tone_class = {"info": "is-info", "warning": "is-warning", "error": "is-error"}.get(str(tone or "").strip().lower(), "is-info")
    return f"<div class='chat__template-modal-message {tone_class}'>{html.escape(message)}</div>"


def _closed_template_modal() -> tuple[dict[str, Any], Any, Any, Any, Any, Any, Any, Any]:
    return ({}, gr.update(visible=False), gr.update(value=""), gr.update(value=""), gr.update(value="", visible=False), gr.update(value="Yes", visible=False), gr.update(value="No", visible=False), gr.update(value="Close", visible=False))


def _open_template_modal(modal_state: dict[str, Any], title_html: str, body_html: str, *, input_visible: bool = False, input_value: str = "", input_label: str = "", yes_visible: bool = False, yes_label: str = "Yes", no_visible: bool = False, no_label: str = "No", close_visible: bool = True, close_label: str = "Close") -> tuple[dict[str, Any], Any, Any, Any, Any, Any, Any, Any]:
    return (
        dict(modal_state or {}),
        gr.update(visible=True),
        gr.update(value=title_html),
        gr.update(value=body_html),
        gr.update(value=input_value, label=input_label, visible=input_visible),
        gr.update(value=yes_label, visible=yes_visible),
        gr.update(value=no_label, visible=no_visible),
        gr.update(value=close_label, visible=close_visible),
    )


def _template_dropdown_updates(tool_values: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, str]]:
    refreshed = deepy_ui_settings.refresh_template_selector_state(tool_values)
    selected_values: dict[str, str] = {}
    dropdown_updates: list[Any] = []
    for tool_name in _TEMPLATE_TOOL_ORDER:
        selected_value = deepy_tool_settings.find_tool_variant(tool_name, refreshed[tool_name]["selected"])
        if selected_value is None:
            selected_value = _normalize_tool_variant(tool_name, tool_values.get(tool_name))
        selected_values[tool_name] = str(selected_value or "")
        dropdown_updates.append(gr.update(choices=refreshed[tool_name]["choices"], value=selected_values[tool_name]))
    return tuple(dropdown_updates), selected_values


def build_deepy_chat_ui(*, deepy_visible: bool) -> DeepyChatUI:
    template_selector_state = deepy_ui_settings.get_template_selector_state()
    tool_ui_state = deepy_ui_settings.get_persisted_assistant_tool_ui_settings()
    session_ui_state = deepy_ui_settings.get_persisted_assistant_session_ui_settings()
    deepy_type = normalize_deepy_type(get_deepy_config_value(DEEPY_TYPE_KEY, ""))
    saved_sessions = session_store.list_sessions(deepy_type)
    session_choices = [(str(item.get("title", "") or "Deepy session"), str(item.get("id", "") or "")) for item in saved_sessions] or [("No saved sessions", "")]
    session_value = session_choices[0][1]
    multi_session_enabled = normalize_deepy_multi_session(session_ui_state["multi_session"])
    session_ui_state["reset_mode"] = session_store.RESET_MODE_NEW if multi_session_enabled else session_store.RESET_MODE_RESET
    if not multi_session_enabled:
        session_ui_state["gallery_media_mode"] = session_store.GALLERY_MEDIA_LINK
    reset_label = "New" if multi_session_enabled else "Reset"
    initial_tool_values = _tool_values_from_ui_settings(tool_ui_state)
    template_controls: list[DeepyTemplateToolControl] = []
    controls_by_tool: dict[str, DeepyTemplateToolControl] = {}
    with gr.Column(elem_id=assistant_chat.DOCK_ID) as dock:
        launcher_host = gr.HTML(assistant_chat.render_launcher_html() if deepy_visible else "", elem_id=assistant_chat.LAUNCHER_HOST_ID, visible=deepy_visible)
        with gr.Column(elem_id=assistant_chat.PANEL_ID, visible=deepy_visible) as panel:
            settings_launcher_host = gr.HTML(assistant_chat.render_settings_launcher_html(), elem_id=assistant_chat.SETTINGS_LAUNCHER_HOST_ID)
            html_output = gr.HTML(assistant_chat.render_shell_html(deepy_type, saved_sessions, multi_session_enabled=multi_session_enabled), elem_id=assistant_chat.CHAT_BLOCK_ID)
            chat_event = gr.Text(value="", interactive=False, visible=False, elem_id=assistant_chat.CHAT_EVENT_ID)
            sync_btn = gr.Button("Synchronize Deepy Chat", visible=False, elem_id=assistant_chat.SYNC_BUTTON_ID)
            submission_id = gr.Text(value="", interactive=False, visible=False, elem_id=assistant_chat.SUBMISSION_ID)
            busy_queue_request = gr.Text(value="", interactive=False, visible=False, elem_id=assistant_chat.BUSY_QUEUE_INPUT_ID)
            busy_queue_submission_id = gr.Text(value="", interactive=False, visible=False, elem_id=assistant_chat.BUSY_QUEUE_SUBMISSION_ID)
            busy_queue_btn = gr.Button("Queue Busy Request", visible=False, elem_id=assistant_chat.BUSY_QUEUE_BUTTON_ID)
            steer_request = gr.Text(value="", interactive=False, visible=False, elem_id=assistant_chat.STEER_INPUT_ID)
            steer_submission_id = gr.Text(value="", interactive=False, visible=False, elem_id=assistant_chat.STEER_SUBMISSION_ID)
            steer_btn = gr.Button("Steer", visible=False, elem_id=assistant_chat.STEER_BUTTON_ID)
            queued_action_input = gr.Text(value="", interactive=False, visible=False, elem_id=assistant_chat.QUEUED_ACTION_INPUT_ID)
            queued_action_btn = gr.Button("Update Queued Request", visible=False, elem_id=assistant_chat.QUEUED_ACTION_BUTTON_ID)
            welcome_session_input = gr.Text(value="", interactive=True, visible=False, elem_id=assistant_chat.WELCOME_SESSION_INPUT_ID)
            welcome_session_btn = gr.Button("Resume Welcome Session", visible=False, elem_id=assistant_chat.WELCOME_SESSION_BUTTON_ID)
            session_resume_state = gr.State({})
            session_prefill_btn = gr.Button("Resume Session Context", visible=False, elem_id=assistant_chat.SESSION_PREFILL_BUTTON_ID)
            session_refresh_btn = gr.Button("Refresh Saved Sessions", visible=False, elem_id=assistant_chat.SESSION_REFRESH_BUTTON_ID)
            pause_btn = gr.Button("Pause / Resume", elem_id=assistant_chat.PAUSE_BRIDGE_ID)
            stop_btn = gr.Button("Stop", elem_id=assistant_chat.STOP_BRIDGE_ID)
            with gr.Row(elem_id=assistant_chat.CONTROLS_ID, elem_classes=["chat__gradio-controls"]):
                request = gr.Text(value="", label="Request", scale=3, show_label=False, elem_id=assistant_chat.REQUEST_ID)
                ask_btn = gr.Button("Ask", scale=1, min_width=10, elem_id=assistant_chat.ASK_BUTTON_ID)
                reset_btn = gr.Button(reset_label, scale=1, min_width=10, elem_id=assistant_chat.RESET_BUTTON_ID)
            stats_output = gr.HTML(assistant_chat.render_stats_html(), elem_id=assistant_chat.STATS_BLOCK_ID)
            with gr.Column(elem_id=assistant_chat.SETTINGS_PANEL_ID) as settings_panel:
                with gr.Column(elem_classes=["chat__template-modal-card", "chat__settings-card"]):
                    gr.HTML(_settings_title_html())
                    with gr.Column(elem_classes=["chat__settings-scroll"]):
                        with gr.Tabs():
                            with gr.Tab("Generation Properties"):
                                compact_actions = gr.Checkbox(value=bool(get_deepy_config_value("deepy_compact_actions", True)), label="Compacted View of Thoughts and Actions", elem_id="assistant_chat_compact_actions")
                                separate_requests_with_empty_line = gr.Checkbox(
                                    value=tool_ui_state["separate_requests_with_empty_line"],
                                    label="Separate Different Requests with an Empty Line",
                                )
                                auto_cancel_queue_tasks = gr.Checkbox(
                                    value=tool_ui_state["auto_cancel_queue_tasks"],
                                    label="Auto-abort or remove Deepy-started generation on Stop/Reset.",
                                )
                                use_template_properties = gr.Dropdown(
                                    choices=deepy_ui_settings.PROPERTY_MODE_CHOICES,
                                    value=tool_ui_state["use_template_properties"],
                                    label="Default Dimensions / Durations / Seed",
                                    elem_classes=["chat__template-tool-dropdown"],
                                )
                                property_controls = {}
                                for row in ((0, 1), (2, 3), (4,)):
                                    with gr.Row():
                                        for index in row:
                                            field = deepy_ui_settings.GENERATION_PROPERTY_FIELDS[index]
                                            property_controls[field["key"]] = gr.Slider(field["minimum"], field["maximum"], value=tool_ui_state[field["key"]], step=field["step"], label=field["label"], interactive=not tool_ui_state["use_template_properties"], show_reset_button=False, elem_classes=["chat__generation-property"])
                                override_width, override_height, override_num_frames, override_audio_duration, override_seed = (property_controls[field["key"]] for field in deepy_ui_settings.GENERATION_PROPERTY_FIELDS)
                            with gr.Tab("Templates Settings used by Tools"):
                                with gr.Row(visible=deepy_type == "prime"):
                                    model_speed, model_size = [gr.Dropdown(choices=field["choices"], value=tool_ui_state[field["key"]], label=field["label"], info=f"{field['label']} helps Deepy choose between models.") for field in deepy_ui_settings.MODEL_SELECTION_FIELDS]
                                with gr.Column(elem_classes=["chat__template-tool-grid"]):
                                    gr.Markdown("Please Match here Prerecorded Models Settings to each Generation Tool used by Deepy.")
                                    for tool_pair in _TEMPLATE_TOOL_LAYOUT:
                                        agent_tasks = not set(tool_pair).isdisjoint(deepy_tool_settings.AGENT_TOOL_IDS)  # Deepy Zero has no tool for them
                                        with gr.Row(elem_classes=["chat__template-tool-grid-row"], visible=deepy_type == "prime" or not agent_tasks):
                                            for tool_name in tool_pair:
                                                with gr.Column(elem_classes=["chat__template-tool-card"]):
                                                    with gr.Row(elem_classes=["chat__template-tool-row"]):
                                                        dropdown = gr.Dropdown(
                                                            choices=template_selector_state[tool_name]["choices"],
                                                            value=tool_ui_state[_TEMPLATE_TOOL_UI_KEY[tool_name]],
                                                            label=deepy_tool_settings.TOOL_DISPLAY_NAMES[tool_name],
                                                            elem_classes=["chat__template-tool-dropdown"],
                                                        )
                                                        with gr.Column(scale=0, min_width=34, elem_classes=["chat__template-tool-actions"]):
                                                            add_btn = gr.Button("\u2795", size="sm", min_width=1, elem_classes=["chat__template-tool-icon-btn"])
                                                            delete_btn = gr.Button("\U0001F5D1\uFE0F", size="sm", min_width=1, elem_classes=["chat__template-tool-icon-btn", "chat__template-tool-icon-btn--danger"])
                                                control = DeepyTemplateToolControl(tool_name=tool_name, dropdown=dropdown, add_btn=add_btn, delete_btn=delete_btn)
                                                controls_by_tool[tool_name] = control
                                                template_controls.append(control)
                            with gr.Tab("Sessions"):
                                multi_session = gr.Dropdown(choices=DEEPY_MULTI_SESSION_CHOICES, value=normalize_deepy_session_mode(session_ui_state["multi_session"]), label="Multisessions")
                                with gr.Column(visible=multi_session_enabled) as multi_session_options:
                                    gr.Markdown("Persistent sessions are created only when the first request is sent. Continuous saves run at safe action boundaries.")
                                    session_workspace_picker = gr.HTML('<div id="deepy-session-workspaces"></div>', visible=normalize_deepy_session_mode(session_ui_state["multi_session"]) == DEEPY_MULTI_SESSION_SELECTABLE)
                                    with gr.Row(elem_classes=["chat__session-selector"]):
                                        session_dropdown = gr.Dropdown(choices=session_choices, value=session_value, label="Saved Sessions", interactive=multi_session_enabled, elem_id="deepy_session_dropdown")
                                    with gr.Row(equal_height=True, elem_classes=["chat__session-action-buttons"]):
                                        session_resume_btn = gr.Button("↩️", scale=1, size="sm", min_width=1, interactive=multi_session_enabled, elem_id=assistant_chat.SESSION_RESUME_BUTTON_ID, elem_classes=["chat__template-tool-icon-btn"])
                                        session_rename_btn = gr.Button("✏️", scale=1, size="sm", min_width=1, interactive=multi_session_enabled, elem_id=assistant_chat.SESSION_RENAME_BUTTON_ID, elem_classes=["chat__template-tool-icon-btn"])
                                        session_duplicate_btn = gr.Button("⧉", scale=1, size="sm", min_width=1, interactive=multi_session_enabled, elem_id=assistant_chat.SESSION_DUPLICATE_BUTTON_ID, elem_classes=["chat__template-tool-icon-btn"])
                                        session_export_btn = gr.Button("📦", scale=1, size="sm", min_width=1, interactive=multi_session_enabled, elem_id=assistant_chat.SESSION_EXPORT_BUTTON_ID, elem_classes=["chat__template-tool-icon-btn"])
                                        session_import_file = gr.UploadButton("📥", file_types=[".zip"], type="filepath", scale=1, size="sm", min_width=1, interactive=multi_session_enabled, elem_id=assistant_chat.SESSION_IMPORT_BUTTON_ID, elem_classes=["chat__template-tool-icon-btn"])
                                        session_delete_btn = gr.Button("🗑️", scale=1, size="sm", min_width=1, interactive=multi_session_enabled, elem_id=assistant_chat.SESSION_DELETE_BUTTON_ID, elem_classes=["chat__template-tool-icon-btn", "chat__template-tool-icon-btn--danger"])
                                    session_export_file = gr.File(label="Exported Session Archive", visible=False, interactive=False)
                                    session_gallery_media_mode = gr.Dropdown(choices=[("Keep links to Gallery files", "link"), ("Copy Gallery files into each session", "copy")], value=session_ui_state["gallery_media_mode"], label="Gallery Media", interactive=multi_session_enabled)
                                with gr.Column(visible=not multi_session_enabled) as mono_session_options:
                                    session_reset_mode = gr.Dropdown(choices=[("Reset Current Session", session_store.RESET_MODE_RESET)], value=session_store.RESET_MODE_RESET, label="Reset Button", interactive=False)
                                session_status = gr.Markdown("")
                        with gr.Row(elem_classes=["chat__settings-actions"]):
                            settings_save_btn = gr.Button("Save Deepy Settings", variant="primary", elem_id=assistant_chat.SAVE_SETTINGS_BUTTON_ID)
                template_selection_history = gr.State(_build_template_selection_history(initial_tool_values))
                template_modal_state = gr.State({})
                captured_lset_value = gr.Text(value="", interactive=False, visible=False)
                captured_lset_label = gr.Text(value="", interactive=False, visible=False)
                with gr.Group(visible=False, elem_classes=["chat__template-modal-wrap"]) as template_modal:
                    with gr.Column(elem_classes=["chat__template-modal-card"]):
                        template_modal_title = gr.HTML("")
                        template_modal_body = gr.HTML("")
                        template_modal_input = gr.Textbox(value="", visible=False, elem_classes=["chat__template-modal-input"])
                        with gr.Row(elem_classes=["chat__template-modal-actions"]):
                            template_modal_yes_btn = gr.Button("Yes", size="sm", visible=False, elem_classes=["chat__template-modal-btn", "chat__template-modal-btn--primary"])
                            template_modal_no_btn = gr.Button("No", size="sm", visible=False, elem_classes=["chat__template-modal-btn"])
                            template_modal_close_btn = gr.Button("Close", size="sm", visible=False, elem_classes=["chat__template-modal-btn"])
    return DeepyChatUI(
        dock=dock,
        launcher_host=launcher_host,
        panel=panel,
        settings_launcher_host=settings_launcher_host,
        settings_save_btn=settings_save_btn,
        settings_panel=settings_panel,
        html_output=html_output,
        chat_event=chat_event,
        sync_btn=sync_btn,
        submission_id=submission_id,
        busy_queue_request=busy_queue_request,
        busy_queue_submission_id=busy_queue_submission_id,
        busy_queue_btn=busy_queue_btn,
        steer_request=steer_request,
        steer_submission_id=steer_submission_id,
        steer_btn=steer_btn,
        queued_action_input=queued_action_input,
        queued_action_btn=queued_action_btn,
        welcome_session_input=welcome_session_input,
        welcome_session_btn=welcome_session_btn,
        session_resume_state=session_resume_state,
        session_prefill_btn=session_prefill_btn,
        session_refresh_btn=session_refresh_btn,
        stats_output=stats_output,
        pause_btn=pause_btn,
        stop_btn=stop_btn,
        request=request,
        ask_btn=ask_btn,
        reset_btn=reset_btn,
        multi_session=multi_session,
        multi_session_options=multi_session_options,
        session_workspace_picker=session_workspace_picker,
        mono_session_options=mono_session_options,
        session_dropdown=session_dropdown,
        session_resume_btn=session_resume_btn,
        session_rename_btn=session_rename_btn,
        session_duplicate_btn=session_duplicate_btn,
        session_export_btn=session_export_btn,
        session_export_file=session_export_file,
        session_import_file=session_import_file,
        session_delete_btn=session_delete_btn,
        session_gallery_media_mode=session_gallery_media_mode,
        session_reset_mode=session_reset_mode,
        session_status=session_status,
        compact_actions=compact_actions,
        auto_cancel_queue_tasks=auto_cancel_queue_tasks,
        separate_requests_with_empty_line=separate_requests_with_empty_line,
        use_template_properties=use_template_properties,
        model_speed=model_speed,
        model_size=model_size,
        override_height=override_height,
        override_width=override_width,
        override_num_frames=override_num_frames,
        override_audio_duration=override_audio_duration,
        override_seed=override_seed,
        template_controls=tuple(template_controls),
        template_selection_history=template_selection_history,
        template_modal_state=template_modal_state,
        captured_lset_value=captured_lset_value,
        captured_lset_label=captured_lset_label,
        template_modal=template_modal,
        template_modal_title=template_modal_title,
        template_modal_body=template_modal_body,
        template_modal_input=template_modal_input,
        template_modal_yes_btn=template_modal_yes_btn,
        template_modal_no_btn=template_modal_no_btn,
        template_modal_close_btn=template_modal_close_btn,
    )


def bind_deepy_chat_ui(
    ui: DeepyChatUI,
    *,
    state: Any,
    output: Any,
    last_choice: Any,
    selected_video_time_input: Any,
    load_queue_trigger: Any,
    output_trigger: Any,
    abort_client_id: Any,
    handlers: DeepyChatHandlers,
) -> None:
    from shared.deepy.hybrid import service_for
    from shared.deepy.hybrid_ui import bind_handlers, bind_settings_sync, control_updates
    hybrid = service_for(state.value)
    if hybrid is not None:
        bind_handlers(hybrid, handlers)
    template_modal_outputs = [
        ui.template_modal_state,
        ui.template_modal,
        ui.template_modal_title,
        ui.template_modal_body,
        ui.template_modal_input,
        ui.template_modal_yes_btn,
        ui.template_modal_no_btn,
        ui.template_modal_close_btn,
    ]
    template_dropdown_inputs = template_dropdown_outputs = [control.dropdown for control in ui.template_controls]  # _TEMPLATE_TOOL_ORDER

    def toggle_override_controls(use_template_properties):
        interactive = not deepy_ui_settings.normalize_assistant_use_template_properties(use_template_properties)
        return gr.update(interactive=interactive), gr.update(interactive=interactive), gr.update(interactive=interactive), gr.update(interactive=interactive), gr.update(interactive=interactive)

    def recover_chat(state_value):
        from shared.deepy.engine import get_or_create_assistant_session

        session = get_or_create_assistant_session(state_value)
        output_queue = session.control_queue
        if output_queue is not None:
            with output_queue.condition:
                if session.control_queue is output_queue:
                    output_queue.push("chat_output", assistant_chat.build_sync_event(session))
                    return gr.update()
        return assistant_chat.build_sync_event(session)

    def _session_catalog(active_id="", active_session=None):
        sessions = handlers.list_saved_sessions()
        live_id = str(getattr(active_session, "storage_session_id", "") or "")
        if live_id and all(str(item.get("id", "") or "") != live_id for item in sessions):
            sessions.insert(0, session_store.session_metadata(active_session))
        choices = [(str(item.get("title", "") or "Deepy session"), str(item.get("id", "") or "")) for item in sessions] or [("No saved sessions", "")]
        selected = str(active_id or "") if any(value == str(active_id or "") for _label, value in choices) else choices[0][1]
        return sessions, gr.update(choices=choices, value=selected)

    def _session_catalog_event(active_id="", enabled=True, active_session=None):
        sessions, dropdown_update = _session_catalog(active_id, active_session)
        return assistant_chat.build_session_catalog_event(sessions, active_id, enabled), dropdown_update

    def _with_session_catalog_updates(state_value, responses):
        from shared.deepy.engine import get_or_create_assistant_session

        session = get_or_create_assistant_session(state_value)
        published_id = None
        for response in responses:
            current_id = str(session.storage_session_id or "")
            if published_id is None and not current_id:
                published_id = ""
                yield *response, gr.update()
                continue
            if current_id == published_id:
                yield *response, gr.update()
                continue
            published_id = current_id
            catalog_event, dropdown_update = _session_catalog_event(current_id, handlers.get_session_ui_settings()["effective_multi_session"], session)
            yield assistant_chat.build_event_batch([response[0], catalog_event]), *response[1:], dropdown_update

    def refresh_saved_sessions(state_value):
        from shared.deepy.engine import get_or_create_assistant_session

        session = get_or_create_assistant_session(state_value)
        return _session_catalog_event(session.storage_session_id, handlers.get_session_ui_settings()["effective_multi_session"], session)

    def _session_preference_updates(settings):
        effective = bool(settings["effective_multi_session"])
        requested = normalize_deepy_multi_session(settings["multi_session"])
        enabled = effective and requested
        reset_label = "New" if effective else "Reset"
        mode_status = "Multi-session mode is active." if effective else "Single-session mode is active; Reset clears the current temporary conversation."
        if settings["restart_required"]:
            status = "This change is not saved. Click Save Deepy Settings, then restart WanGP for it to take effect." if settings["unsaved_changes"] else "Deepy session settings are saved. Restart WanGP for this mode change to take effect."
        else:
            status = f"{mode_status} Changes are not saved yet; click Save Deepy Settings." if settings["unsaved_changes"] else f"{mode_status} Session settings are saved."
        return (
            gr.update(value=status),
            gr.update(value=reset_label),
            gr.update(visible=requested),
            gr.update(visible=not requested),
            *(gr.update(interactive=enabled) for _component in range(7)),
            gr.update(value=settings["gallery_media_mode"], interactive=enabled),
            gr.update(value=session_store.RESET_MODE_RESET, interactive=False),
            gr.update(visible=normalize_deepy_session_mode(settings["multi_session"]) == DEEPY_MULTI_SESSION_SELECTABLE),
        )

    def update_session_preferences(state_value, multi_session, reset_mode, gallery_media_mode, persist=False):
        settings = handlers.update_session_ui_settings(state_value, multi_session=multi_session, reset_mode=reset_mode, gallery_media_mode=gallery_media_mode, persist=persist)
        from shared.deepy.engine import get_or_create_assistant_session

        active_id = get_or_create_assistant_session(state_value).storage_session_id
        catalog_event, _dropdown_update = _session_catalog_event(active_id, settings["effective_multi_session"])
        return catalog_event, *_session_preference_updates(settings)

    def restore_saved_session_cards(state_value, storage_id):
        if hybrid is not None:
            from shared.deepy.errors import DeepyBusy
            try:
                hybrid.control('resume', {'id': storage_id})
            except DeepyBusy as exc:
                gr.Info(str(exc))
                return gr.update(), gr.update(), gr.update(), gr.update(), {}, gr.update()
            return gr.update(), gr.update(), gr.update(), gr.update(value=''), {}, gr.update()
        from shared.deepy.engine import get_or_create_assistant_session

        session = get_or_create_assistant_session(state_value)
        try:
            result = handlers.resume_saved_session(state_value, storage_id, defer_context_prefill=True)
        except Exception as exc:
            reason = str(exc).strip() or exc.__class__.__name__
            warning = f"Deepy session could not be restored: {reason}"
            gr.Warning(warning)
            catalog_event, dropdown_update = _session_catalog_event(session.storage_session_id, handlers.get_session_ui_settings()["effective_multi_session"], session)
            return catalog_event, dropdown_update, gr.update(value=warning), gr.update(), {}, gr.update()
        catalog_event, dropdown_update = _session_catalog_event(result["active_id"], True)
        request_id = str(time.time_ns())
        context_pending = bool(result["context_prefill_pending"])
        action_pending = bool(result["action_replay_pending"])
        needs_finish = context_pending or action_pending
        progress_text = "Session cards restored. Loading context..." if context_pending else "Session cards restored. Replaying interrupted action..."
        loading_event = assistant_chat.build_status_event(progress_text, kind="session_loading", session=session) if needs_finish else None
        resume_ready_event = assistant_chat.build_session_resume_ready_event(request_id, session) if needs_finish else None
        event = assistant_chat.build_event_batch([result["event"], catalog_event, loading_event, resume_ready_event])
        phase = {"request_id": request_id, "context_prefill_pending": context_pending, "action_replay_pending": action_pending, "injected": int(result["injected"]), "warnings": list(result["warnings"])}
        status = progress_text if needs_finish else "Session cards restored."
        refresh = str(time.time()) if phase["injected"] else gr.update()
        return event, dropdown_update, gr.update(value=status), gr.update(value=""), phase, refresh

    def finish_saved_session_resume(state_value, phase):
        phase = dict(phase or {})
        if not phase.get("context_prefill_pending") and not phase.get("action_replay_pending"):
            yield gr.update(), gr.update(value="Session resumed; context was already prepared."), gr.update(), gr.update(), gr.update(), gr.update()
            return
        from shared.deepy.engine import get_or_create_assistant_session

        session = get_or_create_assistant_session(state_value)
        try:
            prefill_tokens, replay_stream = handlers.prefill_restored_session_context(state_value, replay_pending=True)
        except Exception as exc:
            reason = str(exc).strip() or exc.__class__.__name__
            warning = f"Deepy session could not be restored: {reason}"
            gr.Warning(warning)
            yield assistant_chat.build_status_event(None, visible=False, session=session), gr.update(value=warning), gr.update(), gr.update(), gr.update(), gr.update()
            return
        details = f"Session resumed; {prefill_tokens:,} context tokens prepared and {int(phase.get('injected', 0))} Gallery item(s) injected."
        if phase.get("warnings"):
            details += "\n\n" + "\n".join(f"- {warning}" for warning in phase["warnings"])
        if replay_stream is None:
            yield assistant_chat.build_status_event(None, visible=False, session=session), gr.update(value=details), gr.update(), gr.update(), gr.update(), gr.update()
            return
        yield assistant_chat.build_status_event("Context prepared. Replaying the interrupted action...", kind="loading", session=session), gr.update(value=details), gr.update(), gr.update(), gr.update(), gr.update()
        for chat_event, load_update, request_update, output_update, abort_update in replay_stream:
            yield chat_event, gr.update(), load_update, request_update, output_update, abort_update
        yield gr.update(), gr.update(value=details + " The interrupted action was replayed from its beginning."), gr.update(), gr.update(), gr.update(), gr.update()

    def rename_saved_session(state_value, storage_id, title):
        if not str(storage_id or "").strip() or not str(title or "").strip():
            return gr.update(), gr.update(), gr.update()
        manifest = handlers.rename_saved_session(state_value, storage_id, title)
        catalog_event, dropdown_update = _session_catalog_event(manifest["id"], handlers.get_session_ui_settings()["effective_multi_session"])
        return catalog_event, dropdown_update, gr.update(value=f"Session renamed to **{manifest['title']}**.")

    def duplicate_saved_session(state_value, storage_id):
        if not str(storage_id or "").strip():
            return gr.update(), gr.update(), gr.update()
        manifest = handlers.duplicate_saved_session(state_value, storage_id)
        catalog_event, dropdown_update = _session_catalog_event(manifest["id"], handlers.get_session_ui_settings()["effective_multi_session"])
        return catalog_event, dropdown_update, gr.update(value=f"Duplicated as **{manifest['title']}**.")

    def export_saved_session(state_value, storage_id):
        if not str(storage_id or "").strip():
            raise gr.Error("Select a saved Deepy session to export.")
        path = handlers.export_saved_session(state_value, storage_id)
        return gr.update(value=path, visible=True), gr.update(value="Session archive is ready.")

    def _import_path(value):
        if isinstance(value, dict):
            return str(value.get("path") or value.get("name") or "")
        return str(getattr(value, "name", value) or "")

    def import_saved_session(archive_value):
        manifest = handlers.import_saved_session(_import_path(archive_value))
        catalog_event, dropdown_update = _session_catalog_event(manifest["id"], handlers.get_session_ui_settings()["effective_multi_session"])
        return catalog_event, dropdown_update, gr.update(value=f"Imported **{manifest['title']}**."), gr.update(value=None)

    def delete_saved_session(state_value, storage_id):
        if not str(storage_id or "").strip():
            return gr.update(), gr.update(), gr.update()
        from shared.deepy.errors import DeepyBusy
        try:
            result = handlers.delete_saved_session(state_value, storage_id)
        except DeepyBusy as exc:
            gr.Info(str(exc))
            return gr.update(), gr.update(), gr.update()
        catalog_event, dropdown_update = _session_catalog_event(result["active_id"], handlers.get_session_ui_settings()["effective_multi_session"])
        events = [catalog_event] if result["event"] is None else [result["event"], catalog_event]
        return assistant_chat.build_event_batch(events), dropdown_update, gr.update(value=f"Session moved to the recoverable trash folder: `{result['trash_path']}`")

    def _selected_session_title(storage_id):
        selected_id = str(storage_id or "").strip()
        for item in handlers.list_saved_sessions():
            if str(item.get("id", "") or "") == selected_id:
                return str(item.get("title", "") or "Deepy session")
        return selected_id

    def _missing_session_modal(action_title):
        body_html = _modal_message_html("Select a saved Deepy session first.", tone="error")
        return _open_template_modal({}, _session_modal_title_html(action_title), body_html, close_visible=True)

    def open_rename_session_modal(storage_id):
        selected_id = str(storage_id or "").strip()
        if not selected_id:
            return _missing_session_modal("Rename Session")
        title = _selected_session_title(selected_id)
        body_html = _modal_context_html("Selected Deepy Session", title)
        body_html += _modal_message_html("Enter the new name for this session.")
        return _open_template_modal({"action": "session_rename", "storage_id": selected_id}, _session_modal_title_html("Rename Session"), body_html, input_visible=True, input_value=title, input_label="Session name", yes_visible=True, yes_label="Rename", no_visible=True, no_label="Cancel", close_visible=False)

    def open_duplicate_session_modal(storage_id):
        selected_id = str(storage_id or "").strip()
        if not selected_id:
            return _missing_session_modal("Duplicate Session")
        title = _selected_session_title(selected_id)
        body_html = _modal_context_html("Selected Deepy Session", title)
        body_html += _modal_message_html("Create an independent copy of this saved session?")
        return _open_template_modal({"action": "session_duplicate", "storage_id": selected_id}, _session_modal_title_html("Duplicate Session"), body_html, yes_visible=True, yes_label="Duplicate", no_visible=True, no_label="Cancel", close_visible=False)

    def open_delete_session_modal(storage_id):
        selected_id = str(storage_id or "").strip()
        if not selected_id:
            return _missing_session_modal("Delete Session")
        title = _selected_session_title(selected_id)
        body_html = _modal_context_html("Selected Deepy Session", title)
        body_html += _modal_message_html("Delete this session? It will be moved to the recoverable trash folder.", tone="warning")
        return _open_template_modal({"action": "session_delete", "storage_id": selected_id}, _session_modal_title_html("Delete Session"), body_html, yes_visible=True, yes_label="Delete", no_visible=True, no_label="Cancel", close_visible=False)

    def track_template_selection(tool_name, selection_history, *current_templates):
        raw_history = selection_history if isinstance(selection_history, dict) else {}
        previous_current = None
        record = raw_history.get(tool_name)
        if isinstance(record, dict):
            previous_current = deepy_tool_settings.find_tool_variant(tool_name, record.get("current"))
        tool_values = _tool_values_from_inputs(*current_templates)
        normalized_history = _normalize_template_selection_history(selection_history, tool_values)
        current_value = normalized_history[tool_name]["current"]
        if previous_current is not None and previous_current != current_value:
            normalized_history[tool_name]["previous"] = previous_current
        return normalized_history

    def close_template_modal():
        return _closed_template_modal()

    def ask_ai_with_ui_settings(
        state_value,
        output_value,
        last_choice_value,
        ask_request,
        client_submission_id,
        auto_cancel_queue_tasks,
        separate_requests_with_empty_line,
        use_template_properties,
        model_speed,
        model_size,
        override_height,
        override_width,
        override_num_frames,
        override_audio_duration,
        override_seed,
        *current_templates,
    ):
        if hybrid is not None:
            hybrid.submit(ask_request, client_submission_id, steering=False)
            yield gr.update(), gr.update(), gr.update(value=""), gr.update(), gr.update(), gr.update()
            return
        handlers.prepare_request_context(state_value, output_value, last_choice_value)
        update_session_ui_settings(
            state_value,
            auto_cancel_queue_tasks,
            separate_requests_with_empty_line,
            use_template_properties,
            model_speed,
            model_size,
            override_height,
            override_width,
            override_num_frames,
            override_audio_duration,
            override_seed,
            *current_templates,
        )
        yield from _with_session_catalog_updates(state_value, handlers.ask_ai(state_value, ask_request, client_submission_id=client_submission_id))

    def enqueue_ai_with_ui_settings(
        state_value,
        output_value,
        last_choice_value,
        ask_request,
        client_submission_id,
        auto_cancel_queue_tasks,
        separate_requests_with_empty_line,
        use_template_properties,
        model_speed,
        model_size,
        override_height,
        override_width,
        override_num_frames,
        override_audio_duration,
        override_seed,
        *current_templates,
    ):
        if hybrid is not None:
            hybrid.submit(ask_request, client_submission_id, steering=False)
            yield gr.update(), gr.update(), gr.update(value=""), gr.update(), gr.update(), gr.update()
            return
        handlers.prepare_request_context(state_value, output_value, last_choice_value)
        update_session_ui_settings(
            state_value,
            auto_cancel_queue_tasks,
            separate_requests_with_empty_line,
            use_template_properties,
            model_speed,
            model_size,
            override_height,
            override_width,
            override_num_frames,
            override_audio_duration,
            override_seed,
            *current_templates,
        )
        yield from _with_session_catalog_updates(state_value, handlers.enqueue_ai(state_value, ask_request, client_submission_id=client_submission_id))

    def steer_ai_with_ui_settings(
        state_value,
        output_value,
        last_choice_value,
        ask_request,
        client_submission_id,
        auto_cancel_queue_tasks,
        separate_requests_with_empty_line,
        use_template_properties,
        model_speed,
        model_size,
        override_height,
        override_width,
        override_num_frames,
        override_audio_duration,
        override_seed,
        *current_templates,
    ):
        if hybrid is not None:
            hybrid.submit(ask_request, client_submission_id, steering=True)
            yield gr.update(), gr.update(), gr.update(value=""), gr.update(), gr.update(), gr.update()
            return
        handlers.prepare_request_context(state_value, output_value, last_choice_value)
        update_session_ui_settings(
            state_value,
            auto_cancel_queue_tasks,
            separate_requests_with_empty_line,
            use_template_properties,
            model_speed,
            model_size,
            override_height,
            override_width,
            override_num_frames,
            override_audio_duration,
            override_seed,
            *current_templates,
        )
        yield from _with_session_catalog_updates(state_value, handlers.ask_ai(state_value, ask_request, client_submission_id=client_submission_id, steering=True))

    def _apply_ui_settings(
        state_value,
        auto_cancel_queue_tasks,
        separate_requests_with_empty_line,
        use_template_properties,
        model_speed,
        model_size,
        override_height,
        override_width,
        override_num_frames,
        override_audio_duration,
        override_seed,
        *current_templates,
        persist,
    ):
        return handlers.update_tool_ui_settings(
            state_value,
            auto_cancel_queue_tasks=auto_cancel_queue_tasks,
            separate_requests_with_empty_line=separate_requests_with_empty_line,
            use_template_properties=use_template_properties,
            model_speed=model_speed,
            model_size=model_size,
            width=override_width,
            height=override_height,
            num_frames=override_num_frames,
            audio_duration=override_audio_duration,
            seed=override_seed,
            **{_TEMPLATE_TOOL_UI_KEY[tool]: template for tool, template in zip(_TEMPLATE_TOOL_ORDER, current_templates)},
            persist=persist,
        )

    def update_session_ui_settings(
        state_value,
        auto_cancel_queue_tasks,
        separate_requests_with_empty_line,
        use_template_properties,
        model_speed,
        model_size,
        override_height,
        override_width,
        override_num_frames,
        override_audio_duration,
        override_seed,
        *current_templates,
    ):
        return _apply_ui_settings(
            state_value,
            auto_cancel_queue_tasks,
            separate_requests_with_empty_line,
            use_template_properties,
            model_speed,
            model_size,
            override_height,
            override_width,
            override_num_frames,
            override_audio_duration,
            override_seed,
            *current_templates,
            persist=False,
        )

    def persist_ui_settings(
        state_value,
        compact_actions,
        auto_cancel_queue_tasks,
        separate_requests_with_empty_line,
        use_template_properties,
        model_speed,
        model_size,
        override_height,
        override_width,
        override_num_frames,
        override_audio_duration,
        override_seed,
        *values,
    ):
        *current_templates, multi_session, session_reset_mode, session_gallery_media_mode = values
        if hybrid is not None:
            hybrid.update_display_settings({"compact_actions": compact_actions})
        _apply_ui_settings(
            state_value,
            auto_cancel_queue_tasks,
            separate_requests_with_empty_line,
            use_template_properties,
            model_speed,
            model_size,
            override_height,
            override_width,
            override_num_frames,
            override_audio_duration,
            override_seed,
            *current_templates,
            persist=True,
        )
        return update_session_preferences(state_value, multi_session, session_reset_mode, session_gallery_media_mode, persist=True)

    def stop_ai_with_ui(state_value):
        return control_updates(hybrid, "stop") if hybrid is not None else handlers.stop_ai(state_value)

    def pause_ai_with_ui(state_value):
        return control_updates(hybrid, "pause") if hybrid is not None else handlers.stop_ai(state_value, queued_action=assistant_chat.PAUSE_TOGGLE_ACTION)

    def queued_request_action_with_ui(state_value, action_payload):
        if hybrid is not None:
            import json
            return control_updates(hybrid, "queued", json.loads(action_payload))
        return handlers.stop_ai(state_value, queued_action=action_payload)

    def reset_ai_with_ui(state_value, reset_mode):
        if hybrid is not None:
            return *control_updates(hybrid, "reset"), gr.update()
        from shared.deepy.engine import get_or_create_assistant_session

        result = handlers.reset_ai(state_value, reset_mode=reset_mode)
        session = get_or_create_assistant_session(state_value)
        catalog_event, dropdown_update = _session_catalog_event(session.storage_session_id, handlers.get_session_ui_settings()["effective_multi_session"], session)
        return assistant_chat.build_event_batch([result[0], catalog_event]), *result[1:], dropdown_update

    def open_add_template_modal(tool_name, state_value, lset_value, lset_label, current_variant):
        selected_label = str(Path(str(lset_value or "").strip()).name or str(lset_label or "").strip() or "Nothing selected").strip()
        title_html = _modal_title_html(tool_name)
        source_path = deepy_tool_settings.resolve_wangp_settings_file(state_value, lset_value)
        if source_path is None:
            body_html = _current_wangp_settings_context_html(selected_label)
            body_html += _modal_message_html(_TEMPLATE_ADD_SELECTION_ERROR, tone="error")
            return _open_template_modal({}, title_html, body_html, close_visible=True)
        try:
            validation_error = deepy_tool_settings.validate_wangp_settings_for_tool(tool_name, source_path)
        except Exception as exc:
            validation_error = str(exc)
        source_label = source_path.stem
        if validation_error is not None and len(str(validation_error).strip()) > 0:
            body_html = _modal_context_html("Selected WanGP Settings", source_label)
            body_html += _modal_message_html(str(validation_error).strip(), tone="error")
            return _open_template_modal({}, title_html, body_html, close_visible=True)
        linked_variant = deepy_tool_settings.build_linked_tool_variant(state_value, source_path)
        body_html = _modal_context_html("Selected WanGP Settings", source_label)
        body_html += _modal_message_html(f'You are about to link Tool {_tool_display_name(tool_name)} to Settings "{source_label}". Are you sure ?', tone="info")
        modal_state = {
            "action": "add",
            "tool_name": tool_name,
            "variant_name": linked_variant,
            "source_path": source_label,
            "previous_variant": deepy_tool_settings.find_tool_variant(tool_name, current_variant) or "",
        }
        return _open_template_modal(modal_state, title_html, body_html, yes_visible=True, no_visible=True, close_visible=False)

    def open_delete_template_modal(tool_name, current_variant):
        title_html = _modal_title_html(tool_name)
        selected_variant = str(current_variant or "").strip()
        selected_label = str(Path(selected_variant).name or selected_variant or "Nothing selected").strip()
        body_html = _modal_context_html("Selected Deepy Template", selected_label)
        if not deepy_tool_settings.is_linked_tool_variant(selected_variant):
            body_html += _modal_message_html(_TEMPLATE_DELETE_BUILTIN_ERROR, tone="error")
            return _open_template_modal({}, title_html, body_html, close_visible=True)
        body_html += _modal_message_html(f"You are about to remove the link to {selected_label}. Are you sure ?", tone="warning")
        modal_state = {"action": "delete", "tool_name": tool_name, "variant_name": selected_variant}
        return _open_template_modal(modal_state, title_html, body_html, yes_visible=True, no_visible=True, close_visible=False)

    def confirm_template_modal_action(template_modal_state, selection_history, *current_templates):
        tool_values = _tool_values_from_inputs(*current_templates)
        normalized_history = _normalize_template_selection_history(selection_history, tool_values)
        modal_state = template_modal_state if isinstance(template_modal_state, dict) else {}
        action = str(modal_state.get("action", "")).strip().lower()
        tool_name = str(modal_state.get("tool_name", "")).strip()
        if action not in {"add", "delete"} or tool_name not in _TEMPLATE_TOOL_ORDER:
            dropdown_updates, selected_values = _template_dropdown_updates(tool_values)
            normalized_history = _normalize_template_selection_history(normalized_history, selected_values)
            return (*dropdown_updates, normalized_history, *_closed_template_modal())
        try:
            if action == "add":
                previous_variant = deepy_tool_settings.find_tool_variant(tool_name, modal_state.get("previous_variant")) or normalized_history[tool_name]["current"]
                new_variant = str(modal_state.get("variant_name", "")).strip()
                tool_values[tool_name] = new_variant
                dropdown_updates, selected_values = _template_dropdown_updates(tool_values)
                normalized_history = _normalize_template_selection_history(normalized_history, selected_values)
                normalized_history[tool_name]["current"] = selected_values[tool_name]
                normalized_history[tool_name]["previous"] = deepy_tool_settings.find_tool_variant(tool_name, previous_variant) or selected_values[tool_name]
                return (*dropdown_updates, normalized_history, *_closed_template_modal())
            restored_variant = deepy_tool_settings.find_tool_variant(tool_name, normalized_history[tool_name]["previous"])
            if restored_variant is None:
                restored_variant = _normalize_tool_variant(tool_name, "")
            tool_values[tool_name] = restored_variant
            dropdown_updates, selected_values = _template_dropdown_updates(tool_values)
            normalized_history = _normalize_template_selection_history(normalized_history, selected_values)
            normalized_history[tool_name]["current"] = selected_values[tool_name]
            normalized_history[tool_name]["previous"] = selected_values[tool_name]
            return (*dropdown_updates, normalized_history, *_closed_template_modal())
        except Exception as exc:
            dropdown_noops = tuple(gr.update() for _ in _TEMPLATE_TOOL_ORDER)
            if action == "add":
                context_label = str(Path(str(modal_state.get("source_path", "")).strip()).name or str(modal_state.get("variant_name", "")).strip() or "Unknown").strip()
                body_html = _modal_context_html("Selected WanGP Settings", context_label)
            else:
                body_html = _modal_context_html("Selected Deepy Template", str(modal_state.get("variant_name", "")).strip() or "Unknown")
            body_html += _modal_message_html(str(exc), tone="error")
            modal_updates = _open_template_modal({}, _modal_title_html(tool_name), body_html, close_visible=True)
            return (*dropdown_noops, normalized_history, *modal_updates)

    def confirm_modal_action(modal_state_value, modal_input, state_value, selection_history, *current_templates):
        modal_state = modal_state_value if isinstance(modal_state_value, dict) else {}
        action = str(modal_state.get("action", "")).strip().lower()
        if not action.startswith("session_"):
            template_result = confirm_template_modal_action(modal_state, selection_history, *current_templates)
            split = len(_TEMPLATE_TOOL_ORDER) + 1
            return (*template_result[:split], gr.update(), gr.update(), gr.update(), *template_result[split:])
        tool_values = _tool_values_from_inputs(*current_templates)
        normalized_history = _normalize_template_selection_history(selection_history, tool_values)
        dropdown_noops = tuple(gr.update() for _ in _TEMPLATE_TOOL_ORDER)
        storage_id = str(modal_state.get("storage_id", "")).strip()
        title_map = {"session_rename": "Rename Session", "session_duplicate": "Duplicate Session", "session_delete": "Delete Session"}
        try:
            if action == "session_rename":
                if not str(modal_input or "").strip():
                    raise session_store.SessionStoreError("Session name cannot be empty.")
                session_updates = rename_saved_session(state_value, storage_id, modal_input)
            elif action == "session_duplicate":
                session_updates = duplicate_saved_session(state_value, storage_id)
            elif action == "session_delete":
                session_updates = delete_saved_session(state_value, storage_id)
            else:
                raise session_store.SessionStoreError("Unknown Deepy session action.")
            return (*dropdown_noops, normalized_history, *session_updates, *_closed_template_modal())
        except Exception as exc:
            action_title = title_map.get(action, "Session Action")
            body_html = _modal_context_html("Selected Deepy Session", _selected_session_title(storage_id)) if storage_id else ""
            body_html += _modal_message_html(str(exc).strip() or exc.__class__.__name__, tone="error")
            modal_updates = _open_template_modal({}, _session_modal_title_html(action_title), body_html, close_visible=True)
            return (*dropdown_noops, normalized_history, gr.update(), gr.update(), gr.update(value=str(exc)), *modal_updates)

    ui.use_template_properties.change(
        fn=toggle_override_controls,
        inputs=[ui.use_template_properties],
        outputs=[ui.override_height, ui.override_width, ui.override_num_frames, ui.override_audio_duration, ui.override_seed],
        show_progress="hidden",
        queue=False,
    )
    for control in ui.template_controls:
        control.dropdown.change(
            fn=track_template_selection,
            inputs=[gr.State(control.tool_name), ui.template_selection_history, *template_dropdown_inputs],
            outputs=[ui.template_selection_history],
            show_progress="hidden",
            queue=False,
        )
        control.add_btn.click(
            fn=open_add_template_modal,
            inputs=[gr.State(control.tool_name), state, ui.captured_lset_value, ui.captured_lset_label, control.dropdown],
            outputs=template_modal_outputs,
            js="""(toolName, stateValue, _capturedValue, _capturedLabel, currentVariant) => {
                const selection = window.WAC && typeof window.WAC.getWanGpSettingsSelection === 'function'
                    ? window.WAC.getWanGpSettingsSelection()
                    : { value: '', label: '' };
                return [toolName, stateValue, selection.value || '', selection.label || '', currentVariant];
            }""",
            show_progress="hidden",
            queue=False,
        )
        control.delete_btn.click(
            fn=open_delete_template_modal,
            inputs=[gr.State(control.tool_name), control.dropdown],
            outputs=template_modal_outputs,
            show_progress="hidden",
            queue=False,
        )
    ui.template_modal_no_btn.click(fn=close_template_modal, inputs=[], outputs=template_modal_outputs, show_progress="hidden", queue=False)
    ui.template_modal_close_btn.click(fn=close_template_modal, inputs=[], outputs=template_modal_outputs, show_progress="hidden", queue=False)
    ui.template_modal_yes_btn.click(
        fn=confirm_modal_action,
        inputs=[ui.template_modal_state, ui.template_modal_input, state, ui.template_selection_history, *template_dropdown_inputs],
        outputs=[*template_dropdown_outputs, ui.template_selection_history, ui.chat_event, ui.session_dropdown, ui.session_status, *template_modal_outputs],
        show_progress="hidden",
        queue=False,
    )
    selected_video_time_input.change(
        fn=handlers.store_selected_video_time,
        inputs=[state, selected_video_time_input],
        outputs=None,
        show_progress="hidden",
        queue=False,
    )
    session_preference_inputs = [state, ui.multi_session, ui.session_reset_mode, ui.session_gallery_media_mode]
    session_preference_outputs = [
        ui.chat_event,
        ui.session_status,
        ui.reset_btn,
        ui.multi_session_options,
        ui.mono_session_options,
        ui.session_dropdown,
        ui.session_resume_btn,
        ui.session_rename_btn,
        ui.session_duplicate_btn,
        ui.session_export_btn,
        ui.session_delete_btn,
        ui.session_import_file,
        ui.session_gallery_media_mode,
        ui.session_reset_mode,
        ui.session_workspace_picker,
    ]
    for component in (ui.multi_session, ui.session_reset_mode, ui.session_gallery_media_mode):
        (component.input if hybrid is not None else component.change)(fn=update_session_preferences, inputs=session_preference_inputs, outputs=session_preference_outputs, show_progress="hidden", queue=False)

    if hybrid is not None:
        from types import SimpleNamespace
        preference_view = SimpleNamespace(fn=lambda requested, mode: _session_preference_updates({**handlers.get_session_ui_settings(), 'multi_session': requested, 'gallery_media_mode': mode}), inputs=[ui.multi_session, ui.session_gallery_media_mode], outputs=session_preference_outputs[1:])
        with ui.settings_panel:
            settings_form = bind_settings_sync(hybrid, state, ui, _template_dropdown_updates, _session_catalog_event, preference_view)

    def wire_session_resume(button, storage_input):
        button.click(fn=restore_saved_session_cards, inputs=[state, storage_input], outputs=[ui.chat_event, ui.session_dropdown, ui.session_status, ui.request, ui.session_resume_state, output_trigger], show_progress="hidden")

    wire_session_resume(ui.session_resume_btn, ui.session_dropdown)
    wire_session_resume(ui.welcome_session_btn, ui.welcome_session_input)
    ui.session_prefill_btn.click(fn=finish_saved_session_resume, inputs=[state, ui.session_resume_state], outputs=[ui.chat_event, ui.session_status, load_queue_trigger, ui.request, output_trigger, abort_client_id], show_progress="hidden")
    ui.session_refresh_btn.click(fn=refresh_saved_sessions, inputs=[state], outputs=[ui.chat_event, ui.session_dropdown], show_progress="hidden", queue=False)
    ui.session_rename_btn.click(fn=open_rename_session_modal, inputs=[ui.session_dropdown], outputs=template_modal_outputs, show_progress="hidden", queue=False)
    ui.session_duplicate_btn.click(fn=open_duplicate_session_modal, inputs=[ui.session_dropdown], outputs=template_modal_outputs, show_progress="hidden", queue=False)
    ui.session_export_btn.click(fn=export_saved_session, inputs=[state, ui.session_dropdown], outputs=[ui.session_export_file, ui.session_status], show_progress="hidden")
    ui.session_import_file.upload(fn=import_saved_session, inputs=[ui.session_import_file], outputs=[ui.chat_event, ui.session_dropdown, ui.session_status, ui.session_import_file], show_progress="hidden")
    ui.session_delete_btn.click(fn=open_delete_session_modal, inputs=[ui.session_dropdown], outputs=template_modal_outputs, show_progress="hidden", queue=False)
    def save_settings(*args):
        from shared.utils.form_sync import Saved
        return Saved(persist_ui_settings(*args))

    save_settings_event = (lambda **kwargs: settings_form.bind_save(ui.settings_save_btn.click, save_settings, **kwargs)) if hybrid is not None else (lambda **kwargs: ui.settings_save_btn.click(fn=persist_ui_settings, **kwargs))
    save_settings_event(
        inputs=[
            state,
            ui.compact_actions,
            ui.auto_cancel_queue_tasks,
            ui.separate_requests_with_empty_line,
            ui.use_template_properties,
            ui.model_speed,
            ui.model_size,
            ui.override_height,
            ui.override_width,
            ui.override_num_frames,
            ui.override_audio_duration,
            ui.override_seed,
            *template_dropdown_inputs,
            ui.multi_session,
            ui.session_reset_mode,
            ui.session_gallery_media_mode,
        ],
        outputs=session_preference_outputs,
        show_progress="hidden",
    )
    ui.ask_btn.click(
        fn=ask_ai_with_ui_settings,
        inputs=[
            state,
            output,
            last_choice,
            ui.request,
            ui.submission_id,
            ui.auto_cancel_queue_tasks,
            ui.separate_requests_with_empty_line,
            ui.use_template_properties,
            ui.model_speed,
            ui.model_size,
            ui.override_height,
            ui.override_width,
            ui.override_num_frames,
            ui.override_audio_duration,
            ui.override_seed,
            *template_dropdown_inputs,
        ],
        outputs=[ui.chat_event, load_queue_trigger, ui.request, output_trigger, abort_client_id, ui.session_dropdown],
        show_progress="hidden",
        trigger_mode="multiple",
    )
    ui.busy_queue_btn.click(
        fn=enqueue_ai_with_ui_settings,
        inputs=[
            state,
            output,
            last_choice,
            ui.busy_queue_request,
            ui.busy_queue_submission_id,
            ui.auto_cancel_queue_tasks,
            ui.separate_requests_with_empty_line,
            ui.use_template_properties,
            ui.model_speed,
            ui.model_size,
            ui.override_height,
            ui.override_width,
            ui.override_num_frames,
            ui.override_audio_duration,
            ui.override_seed,
            *template_dropdown_inputs,
        ],
        outputs=[ui.chat_event, load_queue_trigger, ui.request, output_trigger, abort_client_id, ui.session_dropdown],
        show_progress="hidden",
        trigger_mode="multiple",
    )
    ui.steer_btn.click(
        fn=steer_ai_with_ui_settings,
        inputs=[
            state,
            output,
            last_choice,
            ui.steer_request,
            ui.steer_submission_id,
            ui.auto_cancel_queue_tasks,
            ui.separate_requests_with_empty_line,
            ui.use_template_properties,
            ui.model_speed,
            ui.model_size,
            ui.override_height,
            ui.override_width,
            ui.override_num_frames,
            ui.override_audio_duration,
            ui.override_seed,
            *template_dropdown_inputs,
        ],
        outputs=[ui.chat_event, load_queue_trigger, ui.request, output_trigger, abort_client_id, ui.session_dropdown],
        show_progress="hidden",
        trigger_mode="multiple",
    )
    ui.queued_action_btn.click(fn=queued_request_action_with_ui, inputs=[state, ui.queued_action_input], outputs=[ui.chat_event, load_queue_trigger, ui.request, abort_client_id], show_progress="hidden", queue=False, trigger_mode="multiple")
    ui.sync_btn.click(fn=recover_chat, inputs=[state], outputs=[ui.chat_event], show_progress="hidden", queue=False, trigger_mode="multiple")
    ui.pause_btn.click(fn=pause_ai_with_ui, inputs=[state], outputs=[ui.chat_event, load_queue_trigger, ui.request, abort_client_id], show_progress="hidden", queue=False, trigger_mode="multiple")
    ui.stop_btn.click(fn=stop_ai_with_ui, inputs=[state], outputs=[ui.chat_event, load_queue_trigger, ui.request, abort_client_id], show_progress="hidden", queue=False, trigger_mode="multiple")
    ui.reset_btn.click(fn=reset_ai_with_ui, inputs=[state, ui.session_reset_mode], outputs=[ui.chat_event, load_queue_trigger, ui.request, abort_client_id, ui.session_dropdown], show_progress="hidden")


__all__ = ["DeepyChatHandlers", "DeepyChatUI", "DeepyTemplateToolControl", "bind_deepy_chat_ui", "build_deepy_chat_ui"]
