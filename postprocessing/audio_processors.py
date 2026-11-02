# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Audio post-processor plugin API.

Audio processors are interchangeable handlers for post-generation audio work.
Each handler declares methods through ``query_audio_processor_def()`` and may
expose config controls under ``wgp_config["audio_processors"][config_key]``.
Definitions may also expose an optional ``description`` plus optional
``method_descriptions`` and ``method_parameters`` mappings for reusable
discovery interfaces. Existing handlers without these fields remain valid.
Discovery tests the existing optional ``enabled()`` method first. When it is
absent, handlers may expose a ``status`` property containing ``"enabled"`` or
``"disabled"``. Discovery reports ``"unknown"`` only when neither contract
provides a valid status. Disabled handlers may expose ``reason_disabled``. This
instance property is separate from the definition's existing per-method
``status`` progress labels.
Handlers may implement ``method_available(method)`` and ``method_visible(method)``.
Both default to True: hidden methods remain usable and are added to menus only
when restoring their selection; unavailable methods are excluded even then.
The shared Postprocessor VRAM Usage policy controls profile 0. Dispatch retains at most one
audio processor handler and releases it before another handler runs.

Plugin authors can register processors from ``plugin_info.json`` with:

```json
{
  "audio_processors": ["my_plugin.audio.MyProcessor"]
}
```

Relative paths are resolved by WanGP's plugin manager, matching spatial
upsampler plugin registration.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import importlib
from typing import Any, Callable

from shared.utils import offload_registry
from . import memory as processor_memory
from .processor_status import PROCESSOR_STATUS_DISABLED, PROCESSOR_STATUS_ENABLED, PROCESSOR_STATUS_UNKNOWN, handler_reason_disabled, handler_status, handler_method_available, handler_method_in_choices


AUDIO_PROCESSOR_TYPE_SOUNDTRACK = "soundtrack"
AUDIO_PROCESSOR_TYPE_VOICE_REPLACEMENT = "voice_replacement"
AUDIO_PROCESSOR_TYPE_AUDIO_EDIT = "audio_edit"
AUDIO_PROCESSOR_LABEL_CONTEXT_LATE_POSTPROCESSING = "late_postprocessing"
AUDIO_PROCESSOR_CONFIG_KEY = "audio_processors"
PERSISTENCE_CONFIG_KEY = "persistence"
PERSIST_UNLOAD = 1
PERSIST_RAM = 2
PERSIST_AUTO = 3  # Legacy config migration value.

MMAUDIO_METHOD = "mmaudio"
CUSTOM_SOUNDTRACK_METHOD = "custom"
REMOVE_BACKGROUND_METHOD = "remove_background"
REMOVE_VOCALS_METHOD = "remove_vocals"
SEEDVC_ONE_SPEAKER_METHOD = "seedvc_one_speaker"
SEEDVC_TWO_SPEAKERS_METHOD = "seedvc_two_speakers"
LEGACY_SEEDVC_METHODS = {
    "seedvc": SEEDVC_ONE_SPEAKER_METHOD,
    "seedvc2": SEEDVC_TWO_SPEAKERS_METHOD,
}
LEGACY_REPLACE_VOICE_ONE_SPEAKER_FLAG = "Y"
LEGACY_REPLACE_VOICE_TWO_SPEAKERS_FLAG = "Z"
LEGACY_REPLACE_VOICE_FLAGS = LEGACY_REPLACE_VOICE_ONE_SPEAKER_FLAG + LEGACY_REPLACE_VOICE_TWO_SPEAKERS_FLAG


audio_processor_handlers = [
    "postprocessing.custom_soundtrack.audio_processor.CustomSoundtrackProcessor",
    "postprocessing.mmaudio.audio_processor.MMAudioProcessor",
    "postprocessing.prismaudio.audio_processor.PrismAudioProcessor",
    "postprocessing.seedvc_processor.audio_processor.SeedVCProcessor",
    "postprocessing.audio_background_removal.audio_processor.BackgroundRemovalProcessor",
]
_audio_processor_handlers: list[Any] = []
_registered_audio_processor_handler_paths: set[str] = set()
_active_audio_processor_handler: Any | None = None
_audio_processor_server_config: dict[str, Any] | None = None


@dataclass
class AudioProcessorConfigBinding:
    handler: Any | None
    config_key: str
    controls: list[tuple[str, Any]]


CONTROL_AUDIO_METHOD = "control"


def normalize_method(method) -> str:
    method = str(method or "").strip()
    return LEGACY_SEEDVC_METHODS.get(method, method)


def _remove_flag_letters(value, letters: str) -> str:
    text = str(value or "")
    for letter in letters:
        text = text.replace(letter, "")
    return text


def _attachment_has_path_values(value) -> bool:
    if value is None:
        return False
    if isinstance(value, (list, tuple, set)):
        return any(_attachment_has_path_values(item) for item in value)
    if isinstance(value, dict):
        return any(_attachment_has_path_values(item) for item in value.values())
    if isinstance(value, str):
        return len(value.strip()) > 0
    return bool(value)


def fix_settings(ui_defaults: dict[str, Any], settings_version: float, attachment_has_path_values: Callable[[Any], bool] | None = None):
    audio_prompt_type = ui_defaults.get("audio_prompt_type", None)
    legacy_audio_prompt_type = audio_prompt_type or ""
    if audio_prompt_type is not None:
        audio_prompt_type = _remove_flag_letters(audio_prompt_type, "R" + LEGACY_REPLACE_VOICE_FLAGS)
        ui_defaults["audio_prompt_type"] = audio_prompt_type
    legacy_mmaudio_setting = ui_defaults.pop("MMAudio_setting", None)
    has_path_values = attachment_has_path_values or _attachment_has_path_values
    if settings_version < 2.59 or "postprocess_audio" not in ui_defaults:
        postprocess_audio = ui_defaults.get("postprocess_audio", "") or ""
        if len(postprocess_audio) == 0:
            if legacy_mmaudio_setting:
                postprocess_audio = MMAUDIO_METHOD
            elif "R" in legacy_audio_prompt_type:
                postprocess_audio = "control"
            elif has_path_values(ui_defaults.get("audio_source", None)):
                postprocess_audio = CUSTOM_SOUNDTRACK_METHOD
        ui_defaults["postprocess_audio"] = normalize_method(postprocess_audio)
    else:
        ui_defaults["postprocess_audio"] = normalize_method(ui_defaults.get("postprocess_audio", "") or "")
    if settings_version < 2.62:
        renamed_settings = {
            "MMAudio_prompt": "postprocess_audio_prompt",
            "MMAudio_neg_prompt": "postprocess_audio_neg_prompt",
        }
        for old_name, new_name in renamed_settings.items():
            if old_name in ui_defaults:
                ui_defaults.setdefault(new_name, ui_defaults[old_name])
                del ui_defaults[old_name]
    if settings_version < 2.63:
        for old_name, new_name in (("seedvc_voice_sample", "replace_voice_sample"), ("seedvc_voice_sample2", "replace_voice_sample2")):
            if old_name in ui_defaults:
                ui_defaults.setdefault(new_name, ui_defaults[old_name])
                del ui_defaults[old_name]
        replace_voice_method = normalize_method(ui_defaults.get("replace_voice_method", "") or "")
        if len(replace_voice_method) == 0:
            if LEGACY_REPLACE_VOICE_TWO_SPEAKERS_FLAG in legacy_audio_prompt_type:
                replace_voice_method = SEEDVC_TWO_SPEAKERS_METHOD
            elif LEGACY_REPLACE_VOICE_ONE_SPEAKER_FLAG in legacy_audio_prompt_type:
                replace_voice_method = SEEDVC_ONE_SPEAKER_METHOD
        ui_defaults["replace_voice_method"] = replace_voice_method
    else:
        ui_defaults["replace_voice_method"] = normalize_method(ui_defaults.get("replace_voice_method", "") or "")
    ui_defaults.setdefault("replace_voice_sample", None)
    ui_defaults.setdefault("replace_voice_sample2", None)
    return audio_prompt_type


def register_audio_processor(handler) -> None:
    if handler not in _audio_processor_handlers:
        _audio_processor_handlers.append(handler)


def _load_processor_class(path: str):
    module_path, class_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)


def _method_choices(handler_def: dict[str, Any]) -> list[tuple[str, str]]:
    return [(label, method) for label, method in handler_def.get("methods", [])]


def _method_labels(handler_def: dict[str, Any]) -> dict[str, str]:
    return {method: label for label, method in _method_choices(handler_def)}


def _method_label(handler_def: dict[str, Any], method: str, label_context: str | None = None) -> str:
    if label_context:
        context_labels = handler_def.get("method_context_labels", {})
        if isinstance(context_labels, dict):
            labels = context_labels.get(label_context, {})
            if isinstance(labels, dict) and method in labels:
                return labels[method]
    return _method_labels(handler_def).get(method, method)


def _config_key_from_handler_def(handler_def: dict[str, Any], fallback_name: str) -> str:
    config_key = str(handler_def.get("config_key", "") or "").strip()
    if config_key:
        return config_key
    method_choices = _method_choices(handler_def)
    if method_choices:
        return method_choices[0][1]
    return fallback_name.lower()


def default_config_sections(handler_modules: list[str] | None = None) -> dict[str, Any]:
    sections = {}
    for path in audio_processor_handlers if handler_modules is None else handler_modules:
        handler_cls = _load_processor_class(str(path or "").strip())
        if not hasattr(handler_cls, "default_config"):
            continue
        config = dict(handler_cls.default_config())
        config.pop(PERSISTENCE_CONFIG_KEY, None)
        if not config:
            continue
        handler_def = handler_cls.query_audio_processor_def()
        sections[_config_key_from_handler_def(handler_def, handler_cls.__name__)] = config
    return sections


def register_audio_processors(server_config, files_locator, handler_modules: list[str] | None = None) -> None:
    global _audio_processor_server_config

    _audio_processor_server_config = server_config
    modules = audio_processor_handlers if handler_modules is None else handler_modules
    for path in modules:
        path = str(path or "").strip()
        if not path or path in _registered_audio_processor_handler_paths:
            continue
        register_audio_processor(_load_processor_class(path)(server_config, files_locator))
        _registered_audio_processor_handler_paths.add(path)


def processor_handlers(processor_type: str | None = None, enabled_only: bool = False) -> list[Any]:
    handlers = []
    for handler in _audio_processor_handlers:
        handler_def = handler.query_audio_processor_def()
        if processor_type is not None and not any(_method_has_type(handler_def, method, processor_type) for _, method in _method_choices(handler_def)):
            continue
        if enabled_only and not handler_enabled(handler):
            continue
        handlers.append(handler)
    return handlers


def handler_enabled(handler) -> bool:
    return not hasattr(handler, "enabled") or handler.enabled()


def _handler_name(handler) -> str:
    return str(handler.query_audio_processor_def()["name"])


def _release_audio_processor_handler(handler) -> None:
    global _active_audio_processor_handler

    released = offload_registry.release_all([_handler_name(handler)])
    if not released and hasattr(handler, "release_vram"):
        handler.release_vram()
    if _active_audio_processor_handler is handler:
        _active_audio_processor_handler = None


def _activate_audio_processor(handler) -> None:
    global _active_audio_processor_handler

    if _active_audio_processor_handler is handler:
        return
    if _active_audio_processor_handler is not None:
        _release_audio_processor_handler(_active_audio_processor_handler)
    _active_audio_processor_handler = handler


def find_processor(method) -> Any | None:
    method = normalize_method(method)
    if not method:
        return None
    for handler in _audio_processor_handlers:
        if method in [key for _, key in _method_choices(handler.query_audio_processor_def())]:
            return handler
    return None


def method_label(method) -> str:
    method = normalize_method(method)
    handler = find_processor(method)
    if handler is None:
        return method
    return _method_label(handler.query_audio_processor_def(), method)


def _handler_pos(handler_def: dict[str, Any]) -> float:
    try:
        return float(handler_def.get("pos", 1000))
    except (TypeError, ValueError):
        return 1000


def _method_pos(handler_def: dict[str, Any], method: str) -> float:
    method_pos = handler_def.get("method_pos", {})
    if isinstance(method_pos, dict) and method in method_pos:
        try:
            return float(method_pos[method])
        except (TypeError, ValueError):
            pass
    return _handler_pos(handler_def)


def _method_types(handler_def: dict[str, Any], method: str) -> tuple[str, ...]:
    method_types = handler_def.get("method_types", {})
    if isinstance(method_types, dict) and method in method_types:
        value = method_types[method]
        return tuple(value) if isinstance(value, (list, tuple, set)) else (str(value),)
    value = handler_def.get("processor_types", ())
    return tuple(value) if isinstance(value, (list, tuple, set)) else (str(value),)


def _method_has_type(handler_def: dict[str, Any], method: str, processor_type: str) -> bool:
    return processor_type in _method_types(handler_def, method)


def _method_value(handler_def: dict[str, Any], key: str, method: str, default=None):
    values = handler_def.get(key, default)
    if isinstance(values, dict):
        return values.get(method, default)
    return values


def method_metadata(method) -> dict[str, Any]:
    method = normalize_method(method)
    handler = find_processor(method)
    metadata = {
        "method": method,
        "label": method,
        "status": "",
        "types": (),
        "needs_prompt": False,
        "needs_negative_prompt": False,
        "needs_audio_source": False,
        "needs_voice_sample": False,
        "needs_voice_sample2": False,
        "supports_repeat": False,
        "speaker_count": 0,
        "handler": handler,
    }
    if handler is None:
        return metadata
    handler_def = handler.query_audio_processor_def()
    metadata.update({
        "label": _method_label(handler_def, method),
        "status": str(_method_value(handler_def, "status", method, method) or ""),
        "types": _method_types(handler_def, method),
    })
    for key in ("needs_prompt", "needs_negative_prompt", "needs_audio_source", "needs_voice_sample", "needs_voice_sample2", "supports_repeat"):
        metadata[key] = bool(_method_value(handler_def, key, method, False))
    try:
        metadata["speaker_count"] = int(_method_value(handler_def, "speaker_count", method, 0) or 0)
    except (TypeError, ValueError):
        metadata["speaker_count"] = 0
    return metadata


def format_method_label(method, label_context: str | None = None) -> str:
    method = normalize_method(method)
    if not method:
        return ""
    handler = find_processor(method)
    if handler is None:
        return method
    if hasattr(handler, "format_method_label"):
        return str(handler.format_method_label(method, label_context=label_context) or method)
    return _method_label(handler.query_audio_processor_def(), method, label_context)


def method_has_type(method, processor_type: str) -> bool:
    return processor_type in method_metadata(method)["types"]


def _choice_value(choices: list[tuple[str, str]], value, default="") -> str:
    value = normalize_method(value)
    available = {method for _, method in choices}
    return value if value in available else default


def _gr_update(**kwargs):
    import gradio as gr

    return gr.update(**kwargs)


def soundtrack_choices(*, include_none: bool = True, include_control: bool = False, include_voice: bool = False, enabled_only: bool = True, label_context: str | None = None, voice_label_context: str | None = None, selected_method: str = "", include_hidden: bool = False) -> list[tuple[str, str]]:
    choices = query_method_choices(AUDIO_PROCESSOR_TYPE_SOUNDTRACK, include_none=include_none, enabled_only=enabled_only, label_context=label_context, selected_method=selected_method, include_hidden=include_hidden)
    if include_control:
        choices.append(("Control Video Audio Track (Reuse Control Video Audio Track)", CONTROL_AUDIO_METHOD))
    if include_voice:
        choices += query_method_choices(AUDIO_PROCESSOR_TYPE_VOICE_REPLACEMENT, include_none=False, enabled_only=enabled_only, label_context=voice_label_context or label_context, selected_method=selected_method, include_hidden=include_hidden)
    return choices


def voice_replacement_choices(*, include_none: bool = True, enabled_only: bool = True, label_context: str | None = None, selected_method: str = "", include_hidden: bool = False) -> list[tuple[str, str]]:
    return query_method_choices(AUDIO_PROCESSOR_TYPE_VOICE_REPLACEMENT, include_none=include_none, enabled_only=enabled_only, label_context=label_context, selected_method=selected_method, include_hidden=include_hidden)


def audio_edit_choices(*, include_none: bool = False, enabled_only: bool = True, selected_method: str = "", include_hidden: bool = False) -> list[tuple[str, str]]:
    return query_method_choices(AUDIO_PROCESSOR_TYPE_AUDIO_EDIT, include_none=include_none, enabled_only=enabled_only, label_context=AUDIO_PROCESSOR_LABEL_CONTEXT_LATE_POSTPROCESSING, selected_method=selected_method, include_hidden=include_hidden)


def has_audio_source_soundtrack(*, enabled_only: bool = True) -> bool:
    return any(method_metadata(method)["needs_audio_source"] for _, method in query_method_choices(AUDIO_PROCESSOR_TYPE_SOUNDTRACK, include_none=False, enabled_only=enabled_only))


def soundtrack_refresh_updates(method):
    metadata = method_metadata(method)
    return (
        _gr_update(visible=metadata["needs_prompt"] or metadata["needs_negative_prompt"]),
        _gr_update(visible=normalize_method(method) == CONTROL_AUDIO_METHOD),
        _gr_update(visible=metadata["needs_audio_source"]),
    )


def voice_replacement_refresh_updates(method):
    metadata = method_metadata(method)
    return _gr_update(visible=metadata["needs_voice_sample"]), _gr_update(visible=metadata["needs_voice_sample2"])


def audio_edit_refresh_updates(method):
    metadata = method_metadata(method)
    return _gr_update(visible=metadata["needs_voice_sample"]), _gr_update(visible=metadata["needs_voice_sample2"])


def late_remux_refresh_updates(method):
    metadata = method_metadata(method)
    return (
        _gr_update(visible=metadata["needs_prompt"] or metadata["needs_negative_prompt"]),
        _gr_update(visible=metadata["needs_audio_source"]),
        _gr_update(visible=metadata["needs_voice_sample"]),
        _gr_update(visible=metadata["needs_voice_sample2"]),
    )


def create_generation_audio_ui(gr, ui_get, ui_defaults, *, any_control_video: bool = False, update_form: bool = False):
    postprocess_choices = soundtrack_choices(include_none=True, include_control=any_control_video, selected_method=normalize_method(ui_get("postprocess_audio")))
    postprocess_value = _choice_value(postprocess_choices, ui_get("postprocess_audio"), "")
    postprocess_metadata = method_metadata(postprocess_value)
    postprocess_audio = gr.Dropdown(choices=postprocess_choices, value=postprocess_value, visible=True, scale=1, label="Postprocess Remux Audio")
    with gr.Column(visible=postprocess_metadata["needs_prompt"] or postprocess_metadata["needs_negative_prompt"]) as postprocess_audio_prompt_col:
        with gr.Row():
            postprocess_audio_prompt = gr.Text(ui_get("postprocess_audio_prompt"), label="Prompt")
            postprocess_audio_neg_prompt = gr.Text(ui_get("postprocess_audio_neg_prompt"), label="Negative Prompt")
    with gr.Column(visible=postprocess_value == CONTROL_AUDIO_METHOD) as postprocess_audio_control_col:
        gr.Markdown("<B>Reuse the Control Video audio track</B>")
    with gr.Column(visible=postprocess_metadata["needs_audio_source"]) as postprocess_audio_source_col:
        audio_source = gr.Audio(value=ui_defaults.get("audio_source", None), type="filepath", label="Soundtrack", show_download_button=True)

    replace_voice_choices = voice_replacement_choices(include_none=True, selected_method=normalize_method(ui_get("replace_voice_method")))
    replace_voice_value = _choice_value(replace_voice_choices, ui_get("replace_voice_method"), "")
    replace_voice_metadata = method_metadata(replace_voice_value)
    with gr.Column(visible=len(replace_voice_choices) > 1) as replace_voice_col:
        replace_voice_method = gr.Dropdown(choices=replace_voice_choices, value=replace_voice_value, label="Replace Voice")
        with gr.Row(visible=replace_voice_metadata["needs_voice_sample"]) as replace_voice_sample_row:
            replace_voice_sample = gr.Audio(value=ui_defaults.get("replace_voice_sample", None), type="filepath", label="Voice Sample #1", show_download_button=True)
        with gr.Row(visible=replace_voice_metadata["needs_voice_sample2"]) as replace_voice_sample2_row:
            replace_voice_sample2 = gr.Audio(value=ui_defaults.get("replace_voice_sample2", None), type="filepath", label="Voice Sample #2", show_download_button=True)

    if not update_form:
        def refresh_soundtrack(method):
            choices = soundtrack_choices(include_control=any_control_video, selected_method=normalize_method(method))
            value = _choice_value(choices, method)
            return gr.update(choices=choices, value=value), *soundtrack_refresh_updates(value)

        def refresh_voice(method):
            choices = voice_replacement_choices(selected_method=normalize_method(method))
            value = _choice_value(choices, method)
            return gr.update(choices=choices, value=value), *voice_replacement_refresh_updates(value)

        postprocess_audio.input(fn=refresh_soundtrack, inputs=[postprocess_audio], outputs=[postprocess_audio, postprocess_audio_prompt_col, postprocess_audio_control_col, postprocess_audio_source_col])
        replace_voice_method.input(fn=refresh_voice, inputs=[replace_voice_method], outputs=[replace_voice_method, replace_voice_sample_row, replace_voice_sample2_row])

    return {
        "postprocess_audio": postprocess_audio,
        "postprocess_audio_prompt_col": postprocess_audio_prompt_col,
        "postprocess_audio_prompt": postprocess_audio_prompt,
        "postprocess_audio_neg_prompt": postprocess_audio_neg_prompt,
        "postprocess_audio_control_col": postprocess_audio_control_col,
        "postprocess_audio_source_col": postprocess_audio_source_col,
        "audio_source": audio_source,
        "replace_voice_col": replace_voice_col,
        "replace_voice_method": replace_voice_method,
        "replace_voice_sample_row": replace_voice_sample_row,
        "replace_voice_sample": replace_voice_sample,
        "replace_voice_sample2_row": replace_voice_sample2_row,
        "replace_voice_sample2": replace_voice_sample2,
    }


def late_audio_edit_menu(selected=None):
    choices = audio_edit_choices(selected_method=normalize_method(selected))
    return choices, _choice_value(choices, selected, choices[0][1] if choices else None)


def late_remux_menu(selected=None):
    choices = soundtrack_choices(include_none=False, include_control=False, include_voice=True, voice_label_context=AUDIO_PROCESSOR_LABEL_CONTEXT_LATE_POSTPROCESSING, selected_method=normalize_method(selected))
    default = CUSTOM_SOUNDTRACK_METHOD if CUSTOM_SOUNDTRACK_METHOD in {method for _, method in choices} else (choices[0][1] if choices else None)
    return choices, _choice_value(choices, selected, default)


def create_late_audio_edit_ui(gr, *, default_visibility_false: dict[str, Any], update_form: bool = False):
    choices, value = late_audio_edit_menu()
    metadata = method_metadata(value)
    postprocess_audio = gr.Dropdown(choices=choices, visible=True, scale=1, label="Audio Action", show_label=False, elem_classes="postprocess", **({} if update_form else {"value": value}))
    with gr.Row(**({} if update_form else {"visible": metadata["needs_voice_sample"]})) as replace_voice_sample_row:
        replace_voice_sample = gr.Audio(label="Voice Sample #1", type="filepath", show_download_button=True)
    with gr.Row(**({} if update_form else {"visible": metadata["needs_voice_sample2"]})) as replace_voice_sample2_row:
        replace_voice_sample2 = gr.Audio(label="Voice Sample #2", type="filepath", show_download_button=True)
    if not update_form:
        postprocess_audio.change(fn=audio_edit_refresh_updates, inputs=[postprocess_audio], outputs=[replace_voice_sample_row, replace_voice_sample2_row])
    return {
        "postprocess_audio": postprocess_audio,
        "replace_voice_sample_row": replace_voice_sample_row,
        "replace_voice_sample": replace_voice_sample,
        "replace_voice_sample2_row": replace_voice_sample2_row,
        "replace_voice_sample2": replace_voice_sample2,
    }


def create_late_remux_ui(gr, *, update_form: bool = False, default_visibility_false: dict[str, Any]):
    choices, default_value = late_remux_menu()
    value = "" if update_form else default_value
    metadata = method_metadata(value)
    with gr.Column(visible=True) as postprocess_audio_col:
        with gr.Row():
            postprocess_audio = gr.Dropdown(choices=choices, visible=True, scale=1, label="Audio Action", show_label=False, elem_classes="postprocess", **({} if update_form else {"value": value}))
        with gr.Column(**({} if update_form else {"visible": metadata["needs_prompt"] or metadata["needs_negative_prompt"]})) as postprocess_audio_prompt_row:
            with gr.Row():
                postprocess_audio_prompt = gr.Text("", label="Prompt", elem_classes="postprocess")
                postprocess_audio_neg_prompt = gr.Text("", label="Negative Prompt", elem_classes="postprocess")
            postprocess_audio_seed = gr.Slider(-1, 999999999, value=-1, step=1, label="Seed (-1 for random)", show_reset_button=False)
            repeat_generation = gr.Slider(1, 25.0, value=1, step=1, label="Number of Sample Videos to Generate", show_reset_button=False)
    with gr.Row(**({} if update_form else {"visible": metadata["needs_audio_source"]})) as audio_source_row:
        audio_source = gr.Audio(label="Soundtrack", type="filepath", show_download_button=True)
    with gr.Row(**({} if update_form else {"visible": metadata["needs_voice_sample"]})) as replace_voice_sample_row:
        replace_voice_sample = gr.Audio(label="Voice Sample #1", type="filepath", show_download_button=True)
    with gr.Row(**({} if update_form else {"visible": metadata["needs_voice_sample2"]})) as replace_voice_sample2_row:
        replace_voice_sample2 = gr.Audio(label="Voice Sample #2", type="filepath", show_download_button=True)
    if not update_form:
        postprocess_audio.change(fn=late_remux_refresh_updates, inputs=[postprocess_audio], outputs=[postprocess_audio_prompt_row, audio_source_row, replace_voice_sample_row, replace_voice_sample2_row])
    return {
        "postprocess_audio_col": postprocess_audio_col,
        "postprocess_audio": postprocess_audio,
        "postprocess_audio_prompt_row": postprocess_audio_prompt_row,
        "postprocess_audio_prompt": postprocess_audio_prompt,
        "postprocess_audio_neg_prompt": postprocess_audio_neg_prompt,
        "postprocess_audio_seed": postprocess_audio_seed,
        "repeat_generation": repeat_generation,
        "audio_source_row": audio_source_row,
        "audio_source": audio_source,
        "replace_voice_sample_row": replace_voice_sample_row,
        "replace_voice_sample": replace_voice_sample,
        "replace_voice_sample2_row": replace_voice_sample2_row,
        "replace_voice_sample2": replace_voice_sample2,
    }


def query_method_choices(processor_type: str, *, include_none: bool = True, enabled_only: bool = False, label_context: str | None = None, selected_method: str = "", include_hidden: bool = False) -> list[tuple[str, str]]:
    choices = [("None", "")] if include_none else []
    for handler in processor_handlers(processor_type, enabled_only=enabled_only):
        handler_def = handler.query_audio_processor_def()
        for label, method in _method_choices(handler_def):
            if _method_has_type(handler_def, method, processor_type) and handler_method_in_choices(handler, method, selected_method=normalize_method(selected_method), include_hidden=include_hidden):
                choice_label = _method_label(handler_def, method, label_context)
                if hasattr(handler, "format_choice_label"):
                    choice_label = handler.format_choice_label(choice_label, method)
                choices.append((_method_pos(handler_def, method), str(choice_label or "").casefold(), str(method or ""), choice_label, method))
    return choices[:1] + [(label, method) for _, _, _, label, method in sorted(choices[1:])] if include_none else [(label, method) for _, _, _, label, method in sorted(choices)]


def validate_method(method, processor_type: str | None = None, **kwargs) -> str:
    method = normalize_method(method)
    handler = find_processor(method)
    if handler is None:
        return "You must choose a valid audio processing method" if method else "You must choose at least one audio processing method"
    if processor_type is not None and not _method_has_type(handler.query_audio_processor_def(), method, processor_type):
        return f"{method_label(method)} is not available for this audio processing stage"
    if not handler_method_available(handler, method):
        return f"{method_label(method)} is not available"
    if hasattr(handler, "validate_method"):
        error = handler.validate_method(method, **kwargs)
        if error:
            return error
    return ""


def download_for_method(method, process_files: Callable[..., Any], **kwargs) -> bool:
    method = normalize_method(method)
    handler = find_processor(method)
    return bool(handler is not None and hasattr(handler, "download") and handler.download(method, process_files, **kwargs))


def query_download_defs(enabled_only: bool = True) -> list[dict[str, Any]]:
    defs = []
    for handler in _audio_processor_handlers:
        if hasattr(handler, "query_download_defs"):
            defs.extend(handler.query_download_defs(enabled_only=enabled_only))
    return [one for one in defs if one]


def _dispatch_processor(handler, operation: str, method, *args, **kwargs):
    _activate_audio_processor(handler)
    kwargs["profile"] = kwargs["profile_no"] = 0
    if kwargs.get("init_pipe") is None:
        kwargs["init_pipe"] = partial(processor_memory.init_pipe, server_config=_audio_processor_server_config)
    persistent = persistent_models()
    try:
        return getattr(handler, operation)(method, *args, **kwargs)
    finally:
        if persistent:
            offload_registry.unload_vram([_handler_name(handler)])
        else:
            _release_audio_processor_handler(handler)


def generate_soundtrack(method, **kwargs) -> str:
    method = normalize_method(method)
    handler = find_processor(method)
    if handler is None or not hasattr(handler, "generate_soundtrack"):
        raise RuntimeError(f"No soundtrack audio processor registered for '{method}'")
    return _dispatch_processor(handler, "generate_soundtrack", method, **kwargs)


def replace_voice_tracks(method, audio_tracks: list[str], **kwargs) -> tuple[list[str], list[str]]:
    method = normalize_method(method)
    handler = find_processor(method)
    if handler is None or not hasattr(handler, "replace_voice_tracks"):
        raise RuntimeError(f"No voice replacement audio processor registered for '{method}'")
    return _dispatch_processor(handler, "replace_voice_tracks", method, audio_tracks, **kwargs)


def process_audio_file(method, **kwargs) -> str:
    method = normalize_method(method)
    handler = find_processor(method)
    if handler is None or not hasattr(handler, "process_audio_file"):
        raise RuntimeError(f"No audio edit processor registered for '{method}'")
    return _dispatch_processor(handler, "process_audio_file", method, **kwargs)


def config_key_for_handler(handler) -> str:
    handler_def = handler.query_audio_processor_def()
    config_key = str(handler_def.get("config_key", "") or "").strip()
    if config_key:
        return config_key
    method_choices = _method_choices(handler_def)
    if method_choices:
        return method_choices[0][1]
    return handler.__class__.__name__.lower()


def _nested_configs(server_config: dict[str, Any]) -> dict[str, Any]:
    configs = server_config.get(AUDIO_PROCESSOR_CONFIG_KEY, {})
    if not isinstance(configs, dict):
        configs = {}
        server_config[AUDIO_PROCESSOR_CONFIG_KEY] = configs
    else:
        server_config.setdefault(AUDIO_PROCESSOR_CONFIG_KEY, configs)
    return configs


def normalize_persistence(value) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = PERSIST_UNLOAD
    return value if value in (PERSIST_UNLOAD, PERSIST_RAM, PERSIST_AUTO) else PERSIST_UNLOAD


def persistence(server_config: dict[str, Any] | None = None) -> int:
    config = _audio_processor_server_config if server_config is None else server_config
    if config is None:
        raise RuntimeError("Audio processors are not registered")
    return normalize_persistence(_nested_configs(config).get(PERSISTENCE_CONFIG_KEY, PERSIST_AUTO))


def persistent_models(server_config: dict[str, Any] | None = None) -> bool:
    return True


def automatic_memory(server_config: dict[str, Any] | None = None) -> bool:
    """Postprocessors use the shared profile 0 VRAM policy."""
    return True


def write_persistence(server_config: dict[str, Any], value) -> int:
    value = normalize_persistence(value)
    _nested_configs(server_config)[PERSISTENCE_CONFIG_KEY] = value
    return value


def _default_config(handler) -> dict[str, Any]:
    config = dict(handler.default_config()) if hasattr(handler, "default_config") else {}
    config.pop(PERSISTENCE_CONFIG_KEY, None)
    return config


def _normalize_config_section(handler, config: dict[str, Any]) -> dict[str, Any]:
    config = dict(handler.normalize_config_section(config)) if hasattr(handler, "normalize_config_section") else dict(config)
    config.pop(PERSISTENCE_CONFIG_KEY, None)
    return config


def read_config_section(server_config: dict[str, Any], handler) -> dict[str, Any]:
    configs = server_config.get(AUDIO_PROCESSOR_CONFIG_KEY, {})
    config_key = config_key_for_handler(handler)
    nested = configs.get(config_key, {}) if isinstance(configs, dict) else {}
    nested = nested if isinstance(nested, dict) else {}
    values = {**_default_config(handler), **dict(nested or {})}
    return _normalize_config_section(handler, values)


def write_config_section(server_config: dict[str, Any], handler, config: dict[str, Any]) -> dict[str, Any]:
    config = _normalize_config_section(handler, {**_default_config(handler), **dict(config or {})})
    _nested_configs(server_config)[config_key_for_handler(handler)] = config
    return config


def migrate_audio_processor_config(server_config: dict[str, Any]) -> bool:
    before = repr(server_config.get(AUDIO_PROCESSOR_CONFIG_KEY, None))
    _nested_configs(server_config).pop(PERSISTENCE_CONFIG_KEY, None)
    for handler in _audio_processor_handlers:
        if _default_config(handler):
            write_config_section(server_config, handler, read_config_section(server_config, handler))
    after = repr(server_config.get(AUDIO_PROCESSOR_CONFIG_KEY, None))
    return before != after


def create_config_ui(gr, server_config: dict[str, Any], *, lock_config: bool = False) -> list[AudioProcessorConfigBinding]:
    bindings = []
    handlers = [handler for handler in _audio_processor_handlers if hasattr(handler, "create_config_ui")]
    if not handlers:
        return bindings
    with gr.Tabs():
        for handler in handlers:
            config = write_config_section(server_config, handler, read_config_section(server_config, handler))
            with gr.Tab(_handler_name(handler)) as tab:
                controls = handler.create_config_ui(gr, config, lock_config=lock_config)
            tab.visible = bool(controls)
            if controls:
                bindings.append(AudioProcessorConfigBinding(handler, config_key_for_handler(handler), list(controls)))
    return bindings


def config_components(bindings: list[AudioProcessorConfigBinding]) -> list[Any]:
    return [component for binding in bindings for _, component in binding.controls]


def collect_config_update(bindings: list[AudioProcessorConfigBinding], values) -> dict[str, dict[str, Any]]:
    values = list(values or [])
    updates, index = {}, 0
    for binding in bindings:
        config = {}
        for field, _ in binding.controls:
            if index >= len(values):
                raise ValueError("Audio processor config UI values do not match registered controls")
            config[field] = values[index]
            index += 1
        updates[binding.config_key] = _normalize_config_section(binding.handler, {**_default_config(binding.handler), **config})
    if index != len(values):
        raise ValueError("Audio processor config UI values do not match registered controls")
    return updates


def validate_config_update_messages(bindings: list[AudioProcessorConfigBinding], updates: dict[str, dict[str, Any]]) -> list[str]:
    messages = []
    for binding in bindings:
        if binding.config_key in updates and hasattr(binding.handler, "validate_config_section"):
            message = binding.handler.validate_config_section(updates[binding.config_key])
            if isinstance(message, str) and message:
                messages.append(message)
            elif isinstance(message, list):
                messages += [text for text in message if text]
    return messages


def apply_config_update(server_config: dict[str, Any], bindings: list[AudioProcessorConfigBinding], updates: dict[str, dict[str, Any]]) -> None:
    for binding in bindings:
        if binding.config_key in updates:
            write_config_section(server_config, binding.handler, updates[binding.config_key])


def release_changed_config_processors(old_config: dict[str, Any], new_config: dict[str, Any], changed_keys) -> None:
    changed_keys = set(changed_keys)
    memory_changed = processor_memory.config_values(old_config) != processor_memory.config_values(new_config)
    released_handler = _active_audio_processor_handler if memory_changed else None
    if released_handler is not None:
        _release_audio_processor_handler(released_handler)
    for handler in _audio_processor_handlers:
        if not hasattr(handler, "release_vram"):
            continue
        old_section = read_config_section(old_config, handler)
        new_section = read_config_section(new_config, handler)
        if hasattr(handler, "config_requires_release"):
            should_release = handler.config_requires_release(old_section, new_section, changed_keys)
        else:
            should_release = old_section != new_section
        if (should_release or memory_changed) and handler is not released_handler:
            _release_audio_processor_handler(handler)
