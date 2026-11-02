# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Temporal upsampler plugin API.

Temporal upsamplers are interchangeable handlers for frame interpolation.
Handlers expose methods and supported multipliers through
``query_temporal_upsampler_def()`` and may expose config controls under
``wgp_config["temporal_upsamplers"][config_key]``.
Definitions may also expose an optional ``description`` plus optional
``method_descriptions`` and ``method_parameters`` mappings for reusable
discovery interfaces. Existing handlers without these fields remain valid.

The slot also holds the texturers of 3D assets: a definition declares the media kinds it processes in ``media``
(default ``("video",)``; ``("3d",)`` for a texturer) and its ``postprocessing_category`` (``interpolator`` by default,
``texturer``). A texturer implements ``texture(mesh, temporal_upsampling, **kwargs)`` (called through ``texture_mesh``)
and returns the textured ``shared.mesh3d.Mesh``. ``method_parameters`` are declared as in the spatial slot: their values
travel in the ``temporal_postprocessing_parameters`` setting under their ids, the UI shows them below the slot's selector
(``postprocessing.parameter_ui``), and the processor receives them under their ``setting`` name.
Discovery tests the existing optional ``enabled()`` method first. When it is
absent, handlers may expose a ``status`` property containing ``"enabled"`` or
``"disabled"``. Discovery reports ``"unknown"`` only when neither contract
provides a valid status. Disabled handlers may expose ``reason_disabled``.
Handlers may implement ``method_available(method)`` and ``method_visible(method)``.
Both default to True: hidden methods remain usable and are added to menus only
when restoring their selection; unavailable methods are excluded even then.

Plugin authors can register processors from ``plugin_info.json`` with:

```json
{
  "temporal_upsampler_handlers": ["my_plugin.temporal.MyTemporalUpsampler"]
}
```
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import hashlib
import importlib
from typing import Any

from shared.attention import attention_shared_state
from shared.utils import offload_registry
from . import memory as processor_memory
from . import parameter_ui
from .processor_status import PROCESSOR_STATUS_DISABLED, PROCESSOR_STATUS_ENABLED, PROCESSOR_STATUS_UNKNOWN, handler_reason_disabled, handler_status, handler_method_available, handler_method_in_choices
from .model_context import compatible_loaded_model


TEMPORAL_UPSAMPLER_CONFIG_KEY = "temporal_upsamplers"
MULTIPLIER_SEPARATOR = "*"
PARAMETER_UI_POSTPROCESSING = "postprocessing"
PARAMETER_UI_LATE_POSTPROCESSING = "late_postprocessing"
MEDIA_VIDEO = "video"
MEDIA_3D = "3d"
CATEGORY_INTERPOLATOR = "interpolator"
CATEGORY_TEXTURER = "texturer"

temporal_upsampler_handlers = [
    "postprocessing.rife.temporal_upsampler.RifeTemporalUpsampler",
    "postprocessing.dlss5.temporal_upsampler.DLSSGTemporalUpsampler",
    "postprocessing.hunyuan3d_paint.texturer.Hunyuan3DPaintTexturer",
    "postprocessing.supratex.texturer.SupraTexTexturer",
    "postprocessing.trellis2_texture.texturer.Trellis2Texturer",
]
_temporal_upsampler_handlers: list[Any] = []
_temporal_upsampler_server_config: dict[str, Any] | None = None
_registered_temporal_upsampler_handler_paths: set[str] = set()
_kept_texturer: Any = None  # the texturer whose models stay loaded: one at a time, as the spatial slot keeps one processor


@dataclass
class TemporalUpsamplerConfigBinding:
    handler: Any
    config_key: str
    controls: list[tuple[str, Any]]


def format_multiplier(scale: float) -> str:
    scale = float(scale)
    return str(int(scale)) if scale.is_integer() else f"{scale:g}"


def format_multiplier_label(scale: float) -> str:
    return f"x{format_multiplier(scale)}"


def format_multiplier_value(method: str, scale: float) -> str:
    return f"{str(method or '').strip().lower()}{MULTIPLIER_SEPARATOR}{format_multiplier(scale)}"


def parse_multiplier_suffix(value, method: str, default_scale: float) -> float | None:
    text = str(value or "").strip().lower()
    method = str(method or "").strip().lower()
    if not text.startswith(method):
        return None
    suffix = text[len(method):]
    if suffix.startswith(MULTIPLIER_SEPARATOR):
        suffix = suffix[len(MULTIPLIER_SEPARATOR):]
        if not suffix:
            return None
    elif MULTIPLIER_SEPARATOR in suffix:
        return None
    try:
        return float(suffix or default_scale)
    except ValueError:
        return None


def format_method_scale_label(label: str, scale: float) -> str:
    return f"{str(label or '').removesuffix(' Upsampler')} {format_multiplier_label(scale)}"


def register_temporal_upsampler(handler) -> None:
    if handler not in _temporal_upsampler_handlers:
        _temporal_upsampler_handlers.append(handler)


def _load_upsampler_class(path: str):
    module_path, class_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)


def _method_choices(handler_def: dict[str, Any]) -> list[tuple[str, str]]:
    return [(label, method) for label, method in handler_def.get("methods", [])]


def _method_labels(handler_def: dict[str, Any]) -> dict[str, str]:
    return {method: label for label, method in _method_choices(handler_def)}


def _config_key_from_handler_def(handler_def: dict[str, Any], fallback_name: str) -> str:
    config_key = str(handler_def.get("config_key", "") or "").strip()
    if config_key:
        return config_key
    method_choices = _method_choices(handler_def)
    if method_choices:
        return method_choices[0][1]
    return fallback_name.lower()


def default_config_sections(handler_modules: list[str] | None = None) -> dict[str, dict[str, Any]]:
    sections = {}
    for path in temporal_upsampler_handlers if handler_modules is None else handler_modules:
        handler_cls = _load_upsampler_class(str(path or "").strip())
        if not hasattr(handler_cls, "default_config"):
            continue
        config = dict(handler_cls.default_config())
        if not config:
            continue
        handler_def = handler_cls.query_temporal_upsampler_def()
        sections[_config_key_from_handler_def(handler_def, handler_cls.__name__)] = config
    return sections


def register_temporal_upsamplers(server_config, files_locator, handler_modules: list[str] | None = None) -> None:
    global _temporal_upsampler_server_config
    _temporal_upsampler_server_config = server_config
    modules = temporal_upsampler_handlers if handler_modules is None else handler_modules
    for path in modules:
        path = str(path or "").strip()
        if not path or path in _registered_temporal_upsampler_handler_paths:
            continue
        register_temporal_upsampler(_load_upsampler_class(path)(server_config, files_locator))
        _registered_temporal_upsampler_handler_paths.add(path)


def registered_temporal_upsamplers(enabled_only: bool = False) -> list[Any]:
    return [handler for handler in _temporal_upsampler_handlers if not enabled_only or handler_enabled(handler)]


def handler_enabled(handler) -> bool:
    return not hasattr(handler, "enabled") or handler.enabled()


def handler_media(handler) -> tuple[str, ...]:
    return tuple(handler.query_temporal_upsampler_def().get("media", (MEDIA_VIDEO,)))


def method_category(method) -> str:
    handler = find_temporal_upsampler_by_method(method)
    return CATEGORY_INTERPOLATOR if handler is None else str(handler.query_temporal_upsampler_def().get("postprocessing_category", CATEGORY_INTERPOLATOR))


def method_parameters(method, *, ui_context: str | None = None) -> list[dict[str, Any]]:
    handler = find_temporal_upsampler_by_method(method)
    parameters = {} if handler is None else handler.query_temporal_upsampler_def().get("method_parameters", {})
    values = parameters.get(str(method or "").strip(), ()) if isinstance(parameters, dict) else ()
    output = []
    for parameter in values if isinstance(values, (list, tuple)) else ():
        if not isinstance(parameter, dict) or not str(parameter.get("name", "") or "").strip():
            continue
        contexts = parameter.get("ui", ())
        contexts = (contexts,) if isinstance(contexts, str) else tuple(contexts)
        if ui_context is None or ui_context in contexts:
            output.append(dict(parameter))
    return output


def runtime_parameter_kwargs(temporal_upsampling, parameter_values) -> dict[str, Any]:
    split = split_temporal_upsampling_value(temporal_upsampling)
    return {} if split is None else parameter_ui.runtime_kwargs(method_parameters(split[0]), parameter_values)


def slot_label(media_kind: str) -> str:
    return "Texturer" if media_kind == MEDIA_3D else "Temporal Upsampling"


def query_temporal_upsampler_defs(enabled_only: bool = False) -> list[dict[str, Any]]:
    return [handler.query_temporal_upsampler_def() for handler in registered_temporal_upsamplers(enabled_only)]


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


def find_temporal_upsampler(temporal_upsampling) -> Any | None:
    if not str(temporal_upsampling or "").strip():
        return None
    return next((handler for handler in _temporal_upsampler_handlers if handler.is_upsampling(temporal_upsampling)), None)


def find_temporal_upsampler_by_method(method) -> Any | None:
    method = str(method or "").strip()
    if not method:
        return None
    for handler in _temporal_upsampler_handlers:
        if method in [key for _, key in _method_choices(handler.query_temporal_upsampler_def())]:
            return handler
    return None


def require_temporal_upsampler_by_method(method) -> Any:
    handler = find_temporal_upsampler_by_method(method)
    if handler is None:
        raise RuntimeError(f"No temporal upsampler registered for method '{method}'")
    return handler


def method_multipliers(method) -> tuple[float, ...]:
    handler = find_temporal_upsampler_by_method(method)
    if handler is None:
        return ()
    return tuple(handler.query_temporal_upsampler_def().get("multipliers", {}).get(str(method), ()))


def multiplier_choices_for_method(method) -> list[tuple[str, float]]:
    return [(format_multiplier_label(scale), scale) for scale in method_multipliers(method)]


def _default_multiplier_from_def(handler_def: dict[str, Any], method: str) -> float | None:
    multipliers = tuple(handler_def.get("multipliers", {}).get(method, ()))
    if not multipliers:
        return None
    default_value = handler_def.get("default_temporal_upsampling", "")
    handler = find_temporal_upsampler(default_value)
    if handler is not None:
        split = handler.split_value(default_value)
        if split is not None and split[0] == method and split[1] in multipliers:
            return split[1]
    return multipliers[0]


def default_multiplier_for_method(method) -> float:
    handler = find_temporal_upsampler_by_method(method)
    if handler is None:
        return 2.0
    return _default_multiplier_from_def(handler.query_temporal_upsampler_def(), str(method or "").strip()) or 2.0


def normalize_multiplier_for_method(method, scale) -> float:
    multipliers = method_multipliers(method)
    if not multipliers:
        return 2.0
    try:
        scale = float(scale)
    except (TypeError, ValueError):
        scale = default_multiplier_for_method(method)
    return scale if scale in multipliers else default_multiplier_for_method(method)


def split_temporal_upsampling_value(value) -> tuple[str, float] | None:
    handler = find_temporal_upsampler(value)
    return None if handler is None else handler.split_value(value)


def build_temporal_upsampling_value(method, scale) -> str | None:
    handler = find_temporal_upsampler_by_method(method)
    return None if handler is None else handler.build_value(method, scale)


def normalize_temporal_upsampling_value(value) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    split = split_temporal_upsampling_value(text)
    if split is None:
        return text
    handler = find_temporal_upsampler_by_method(split[0])
    normalized = None if handler is None else handler.build_value(*split)
    return normalized or text


def format_temporal_upsampling_label(value) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    handler = find_temporal_upsampler(text)
    if handler is None:
        return text
    split = handler.split_value(text)
    if split is None:
        return text
    method, scale = split
    label = _method_labels(handler.query_temporal_upsampler_def()).get(method)
    if not label:
        return text
    return format_method_scale_label(label, scale) if method_multipliers(method) else label


def normalize_temporal_upsampling_state(method, scale) -> tuple[list[tuple[str, float]], float | None, str]:
    method = str(method or "").strip()
    if not method:
        return [], None, ""
    multiplier_choices = multiplier_choices_for_method(method)
    scale = normalize_multiplier_for_method(method, scale) if multiplier_choices else None  # texturers: no multiplier
    return multiplier_choices, scale, build_temporal_upsampling_value(method, scale) or ""


def normalize_temporal_upsampling_value_for_method(method, current_value) -> tuple[list[tuple[str, float]], float | None, str]:
    split = split_temporal_upsampling_value(current_value)
    return normalize_temporal_upsampling_state(method, 2.0 if split is None else split[1])


def query_method_choices(enabled_only: bool = False, *, selected_method: str = "", include_hidden: bool = False, media_kind: str = MEDIA_VIDEO) -> list[tuple[str, str]]:
    choices = []
    for handler in registered_temporal_upsamplers(enabled_only):
        handler_def = handler.query_temporal_upsampler_def()
        if media_kind not in handler_media(handler):
            continue
        choices += [(_method_pos(handler_def, method), str(label or "").casefold(), str(method or ""), label, method) for label, method in _method_choices(handler_def) if handler_method_in_choices(handler, method, selected_method=selected_method, include_hidden=include_hidden)]
    return [(label, method) for _, _, _, label, method in sorted(choices)]


def temporal_help_markdown(method_choices, *, field_help=None, media_kind: str = MEDIA_VIDEO) -> str:
    intro = ("Texturers paint the PBR materials of a 3D asset (base colour, metallic, roughness) from its reference image." if media_kind == MEDIA_3D else
             getattr(field_help, "TEMPORAL_UPSAMPLER_HELP_INTRO", "Temporal upsamplers increase frame rate by generating intermediate frames."))
    sections = [intro]
    handler_choices = {}
    for label, method in method_choices:
        handler = find_temporal_upsampler_by_method(method)
        if handler is not None:
            handler_choices.setdefault(handler, []).append((label, method))
    for handler, choices in handler_choices.items():
        handler_def = handler.query_temporal_upsampler_def()
        name = str(handler_def.get("name", choices[0][0]))
        description = str(handler_def.get("description", "") or "").strip()
        lines = [f"### {name}"]
        if description:
            lines.append(description)
        method_descriptions = handler_def.get("method_descriptions", {})
        if isinstance(method_descriptions, dict):
            details = [(str(label), str(method_descriptions.get(method, "") or "").strip()) for label, method in choices]
            lines += [f"- **{label}:** {detail}" for label, detail in details if detail and detail != description]
        sections.append("\n\n".join(lines))
    return "\n\n".join(sections)


def temporal_help_popup(method_choices, *, field_help=None, media_kind: str = MEDIA_VIDEO) -> tuple[str, str]:
    return ("Texturer" if media_kind == MEDIA_3D else "Temporal Upsampler"), temporal_help_markdown(method_choices, field_help=field_help, media_kind=media_kind)


def temporal_help_id(method_choices) -> str:
    signature = "|".join(str(method) for _label, method in method_choices)
    return f"temporal_upsampling_{hashlib.sha1(signature.encode('utf-8')).hexdigest()[:12]}"


def dropdown_state(temporal_upsampling, *, late_postprocessing: bool = False, media_kind: str = MEDIA_VIDEO) -> dict[str, Any]:
    method, scale = split_temporal_upsampling_value(temporal_upsampling) or ("", 2.0)
    method_choices = [("None", "")] + query_method_choices(enabled_only=not late_postprocessing, selected_method=method, media_kind=media_kind)
    if method not in {value for _, value in method_choices}:
        method = ""
    multiplier_choices, scale, value = normalize_temporal_upsampling_state(method, scale)
    return {"method": method, "scale": scale, "value": value, "method_choices": method_choices, "multiplier_choices": multiplier_choices}


def late_postprocessing_ui_state(temporal_upsampling, *, media_kind: str) -> dict[str, Any]:
    """The state of the late panel's slot for the selected media, with its choices' context (the slot's context state)."""
    context = {"late_postprocessing": True, "media_kind": media_kind}
    return {**dropdown_state(temporal_upsampling, **context), "context": context}


def create_generation_temporal_ui(gr, temporal_upsampling, *, visible: bool = True, late_postprocessing: bool = False, elem_classes=None, update_form: bool = False, field_help=None,
                                  media_kind: str = MEDIA_VIDEO, help_target_id: str | None = None) -> dict[str, Any]:
    """The temporal slot's selectors; its processors' parameters are in the shared inputs of parameter_ui. Its help (returned in
    extra_components, refreshed with the form) follows the media kind: Temporal Upsampler, or Texturer for 3D assets. So does the context
    of its choices, also in media_outputs for the late panel: the slot's events are bound once, for the first model."""
    context = {"late_postprocessing": late_postprocessing, "media_kind": media_kind}
    state = dropdown_state(temporal_upsampling if visible else "", **context)
    help_components = []
    with gr.Row():
        method = gr.Dropdown(choices=state["method_choices"], value=state["method"], visible=visible, scale=3, label=slot_label(media_kind), elem_id=help_target_id, elem_classes=elem_classes)
        if field_help is not None:
            help_title, help_markdown = temporal_help_popup(state["method_choices"], field_help=field_help, media_kind=media_kind)
            help_id = temporal_help_id(state["method_choices"])
            help_components = [gr.update(value=field_help.render_marker(help_target_id, help_id, title=help_title, markdown=help_markdown)) if update_form else field_help.bind(method, help_id, title=help_title, markdown=help_markdown)]
        multiplier = gr.Dropdown(choices=state["multiplier_choices"], value=state["scale"], visible=visible and bool(state["multiplier_choices"]), scale=1, label="Multiplier", elem_classes=elem_classes)
    value = gr.Textbox(value=state["value"], visible=False, elem_classes=elem_classes)

    context_component = context if update_form else gr.State(context)

    def refresh_method(method, value, context):
        _, _, value = normalize_temporal_upsampling_value_for_method(method, value)
        state = dropdown_state(value, **context)
        return gr.update(choices=state["method_choices"], value=state["method"]), gr.update(choices=state["multiplier_choices"], value=state["scale"], visible=bool(state["multiplier_choices"])), state["value"]

    def refresh_multiplier(method, scale):
        _, scale, value = normalize_temporal_upsampling_state(method, scale)
        return gr.update(value=scale, visible=bool(method and method_multipliers(method))), value

    if not update_form:
        gr.on(triggers=[method.input], fn=refresh_method, inputs=[method, value, context_component], outputs=[method, multiplier, value], show_progress="hidden")
        gr.on(triggers=[multiplier.input], fn=refresh_multiplier, inputs=[method, multiplier], outputs=[multiplier, value], show_progress="hidden")
    return {"value": value, "method": method, "method_value": state["method"], "multiplier": multiplier, "extra_components": [*help_components, context_component],
            "media_outputs": [*help_components, context_component]}


def validate_temporal_upsampling(temporal_upsampling, *, source_is_image: bool = False, media_kind: str | None = None) -> str:
    """media_kind: the processed media when it is not a video (``3d``)."""
    temporal_upsampling = str(temporal_upsampling or "").strip()
    if not temporal_upsampling:
        return ""
    handler = find_temporal_upsampler(temporal_upsampling)
    if handler is None:
        return f"No temporal upsampler registered for '{temporal_upsampling}'"
    method, _ = handler.split_value(temporal_upsampling)
    if not handler_method_available(handler, method):
        return f"Temporal upsampler '{method}' is not available"
    if (media_kind or MEDIA_VIDEO) not in handler_media(handler):
        return f"{handler.query_temporal_upsampler_def()['name']} does not process {'3D assets' if media_kind == MEDIA_3D else 'videos'}"
    if hasattr(handler, "validate_upsampling"):
        return handler.validate_upsampling(temporal_upsampling, source_is_image=source_is_image)
    return "Temporal Upsampling can not be used with an Image" if source_is_image else ""


def temporal_upsample(temporal_upsampling, sample, previous_last_frame, fps, *, main_offloadobj=None, loaded_model_context=None, **kwargs):
    handler = find_temporal_upsampler(temporal_upsampling)
    if handler is None:
        if str(temporal_upsampling or "").strip():
            raise ValueError(f"No temporal upsampler registered for '{temporal_upsampling}'")
        return sample, previous_last_frame, fps
    kwargs["profile"] = 0
    if kwargs.get("init_pipe") is None:
        kwargs["init_pipe"] = partial(processor_memory.init_pipe, server_config=_temporal_upsampler_server_config)
    name = handler.query_temporal_upsampler_def()["name"]
    persistent = handler.persistent_models() if hasattr(handler, "persistent_models") else False
    borrowed_context = compatible_loaded_model(handler, temporal_upsampling, loaded_model_context)
    if borrowed_context is None and hasattr(handler, "download"):
        from shared.utils.download import process_files_def
        handler.download(kwargs.get("process_files") or process_files_def, temporal_upsampling=temporal_upsampling)
    with attention_shared_state():
        try:
            core_offloadobj = loaded_model_context.offloadobj if loaded_model_context is not None else main_offloadobj
            if core_offloadobj is not None:
                core_offloadobj.unload_all()
            if borrowed_context is not None and hasattr(handler, "release_private_runtime"):
                handler.release_private_runtime()
            elif borrowed_context is None and hasattr(handler, "load_upsampler"):
                handler.load_upsampler(temporal_upsampling, **kwargs)
            return handler.temporal_upsample(temporal_upsampling, sample, previous_last_frame, fps, loaded_model_context=borrowed_context, **kwargs)
        finally:
            if borrowed_context is not None:
                borrowed_context.offloadobj.unload_all()
            elif persistent:
                offload_registry.unload_vram([name])
            else:
                offload_registry.release_all([name])


def texture_mesh(temporal_upsampling, mesh, *, main_offloadobj=None, loaded_model_context=None, parameter_values=None, **kwargs):
    """The mesh (shared.mesh3d.Mesh) textured by the texturer of temporal_upsampling, with its declared parameters from
    parameter_values (the slot's runtime values, parameter_ui.runtime_values) and kwargs (reference_image, seed,
    abort_callback, progress_callback...)."""
    global _kept_texturer
    handler = find_temporal_upsampler(temporal_upsampling)
    if handler is None:
        raise ValueError(f"No texturer registered for '{temporal_upsampling}'")
    kwargs["profile"] = 0
    if kwargs.get("init_pipe") is None:
        kwargs["init_pipe"] = partial(processor_memory.init_pipe, server_config=_temporal_upsampler_server_config)
    kwargs.update(runtime_parameter_kwargs(temporal_upsampling, parameter_values))
    name = handler.query_temporal_upsampler_def()["name"]
    persistent = handler.persistent_models() if hasattr(handler, "persistent_models") else False
    if persistent:
        if _kept_texturer not in (None, handler):
            offload_registry.release_all([_kept_texturer.query_temporal_upsampler_def()["name"]])
        _kept_texturer = handler
    borrowed_context = compatible_loaded_model(handler, temporal_upsampling, loaded_model_context)
    if borrowed_context is None and hasattr(handler, "download"):
        from shared.utils.download import process_files_def
        handler.download(kwargs.get("process_files") or process_files_def, temporal_upsampling=temporal_upsampling)
    with attention_shared_state():
        try:
            core_offloadobj = loaded_model_context.offloadobj if loaded_model_context is not None else main_offloadobj
            if core_offloadobj is not None:
                core_offloadobj.unload_all()
            if borrowed_context is None and hasattr(handler, "load_upsampler"):
                handler.load_upsampler(temporal_upsampling, **kwargs)
            return handler.texture(mesh, temporal_upsampling, loaded_model_context=borrowed_context, **kwargs)
        finally:
            if borrowed_context is not None:
                borrowed_context.offloadobj.unload_all()
            elif persistent:
                offload_registry.unload_vram([name])
            else:
                offload_registry.release_all([name])


def download_for_value(temporal_upsampling, process_files, **kwargs):
    handler = find_temporal_upsampler(temporal_upsampling)
    if handler is None or not hasattr(handler, "download"):
        return False
    return handler.download(process_files, temporal_upsampling=temporal_upsampling, **kwargs)


def query_download_defs(enabled_only: bool = True) -> list[dict[str, Any]]:
    defs = []
    for handler in registered_temporal_upsamplers(enabled_only):
        if hasattr(handler, "query_download_defs"):
            defs.extend(handler.query_download_defs(enabled_only=enabled_only))
        elif hasattr(handler, "query_download_def"):
            download_def = handler.query_download_def(enabled_only=enabled_only)
            if download_def is not None:
                defs.append(download_def)
    return defs


def config_key_for_handler(handler) -> str:
    handler_def = handler.query_temporal_upsampler_def()
    return _config_key_from_handler_def(handler_def, handler.__class__.__name__)


def _nested_configs(server_config: dict[str, Any]) -> dict[str, Any]:
    configs = server_config.get(TEMPORAL_UPSAMPLER_CONFIG_KEY, {})
    if not isinstance(configs, dict):
        configs = {}
        server_config[TEMPORAL_UPSAMPLER_CONFIG_KEY] = configs
    else:
        server_config.setdefault(TEMPORAL_UPSAMPLER_CONFIG_KEY, configs)
    return configs


def _default_config(handler) -> dict[str, Any]:
    return dict(handler.default_config()) if hasattr(handler, "default_config") else {}


def _normalize_config_section(handler, config: dict[str, Any]) -> dict[str, Any]:
    return dict(handler.normalize_config_section(config)) if hasattr(handler, "normalize_config_section") else dict(config)


def read_config_section(server_config: dict[str, Any], handler) -> dict[str, Any]:
    configs = server_config.get(TEMPORAL_UPSAMPLER_CONFIG_KEY, {})
    nested = configs.get(config_key_for_handler(handler), {}) if isinstance(configs, dict) else {}
    values = {**_default_config(handler), **(nested if isinstance(nested, dict) else {})}
    return _normalize_config_section(handler, values)


def write_config_section(server_config: dict[str, Any], handler, config: dict[str, Any]) -> dict[str, Any]:
    config = _normalize_config_section(handler, {**_default_config(handler), **dict(config or {})})
    _nested_configs(server_config)[config_key_for_handler(handler)] = config
    return config


def migrate_temporal_upsampler_config(server_config: dict[str, Any]) -> bool:
    before = repr(server_config.get(TEMPORAL_UPSAMPLER_CONFIG_KEY, None))
    for handler in _temporal_upsampler_handlers:
        if _default_config(handler):
            write_config_section(server_config, handler, read_config_section(server_config, handler))
    return before != repr(server_config.get(TEMPORAL_UPSAMPLER_CONFIG_KEY, None))


def create_config_ui(gr, server_config: dict[str, Any], *, lock_config: bool = False) -> list[TemporalUpsamplerConfigBinding]:
    bindings = []
    handlers = [handler for handler in _temporal_upsampler_handlers if hasattr(handler, "create_config_ui")]
    if not handlers:
        return bindings
    with gr.Tabs():
        for handler in handlers:
            config = write_config_section(server_config, handler, read_config_section(server_config, handler))
            with gr.Tab(handler.query_temporal_upsampler_def()["name"]) as tab:
                controls = handler.create_config_ui(gr, config, lock_config=lock_config)
            tab.visible = bool(controls)
            if controls:
                bindings.append(TemporalUpsamplerConfigBinding(handler, config_key_for_handler(handler), list(controls)))
    return bindings


def config_components(bindings: list[TemporalUpsamplerConfigBinding]) -> list[Any]:
    return [component for binding in bindings for _, component in binding.controls]


def collect_config_update(bindings: list[TemporalUpsamplerConfigBinding], values) -> dict[str, dict[str, Any]]:
    values = list(values or [])
    updates, index = {}, 0
    for binding in bindings:
        config = {}
        for field, _ in binding.controls:
            if index >= len(values):
                raise ValueError("Temporal upsampler config UI values do not match registered controls")
            config[field] = values[index]
            index += 1
        updates[binding.config_key] = _normalize_config_section(binding.handler, {**_default_config(binding.handler), **config})
    if index != len(values):
        raise ValueError("Temporal upsampler config UI values do not match registered controls")
    return updates


def validate_config_update_messages(bindings: list[TemporalUpsamplerConfigBinding], updates: dict[str, dict[str, Any]]) -> list[str]:
    messages = []
    for binding in bindings:
        if binding.config_key in updates and hasattr(binding.handler, "validate_config_section"):
            message = binding.handler.validate_config_section(updates[binding.config_key])
            if isinstance(message, str) and message:
                messages.append(message)
            elif isinstance(message, list):
                messages += [text for text in message if text]
    return messages


def apply_config_update(server_config: dict[str, Any], bindings: list[TemporalUpsamplerConfigBinding], updates: dict[str, dict[str, Any]]) -> None:
    for binding in bindings:
        if binding.config_key in updates:
            write_config_section(server_config, binding.handler, updates[binding.config_key])


def release_changed_config_temporal_upsamplers(old_config: dict[str, Any], new_config: dict[str, Any], changed_keys) -> None:
    changed_keys = set(changed_keys)
    memory_changed = processor_memory.config_values(old_config) != processor_memory.config_values(new_config)
    for handler in _temporal_upsampler_handlers:
        if not hasattr(handler, "release_vram"):
            continue
        old_section = read_config_section(old_config, handler)
        new_section = read_config_section(new_config, handler)
        if hasattr(handler, "config_requires_release"):
            should_release = handler.config_requires_release(old_section, new_section, changed_keys)
        else:
            should_release = old_section != new_section
        if should_release or memory_changed:
            handler.release_vram()


class SimpleScaleSuffixMixin:
    """Value helpers writing '<method>*<multiplier>' while accepting the legacy concatenated form."""

    def _method_keys(self):
        return [key for _, key in _method_choices(self.query_temporal_upsampler_def())]

    def split_value(self, value):
        text = str(value or "").strip().lower()
        for method in sorted(self._method_keys(), key=len, reverse=True):
            if text.startswith(method):
                scale = parse_multiplier_suffix(text, method, 2.0)
                return None if scale is None else (method, scale)
        return None

    def build_value(self, method, scale):
        method = str(method or "").strip().lower()
        if method not in self._method_keys():
            return None
        multipliers = self.query_temporal_upsampler_def().get("multipliers", {}).get(method, ())
        scale = float(scale or 0)
        if scale not in multipliers:
            scale = _default_multiplier_from_def(self.query_temporal_upsampler_def(), method) or 0
        return format_multiplier_value(method, scale)

    def is_upsampling(self, value) -> bool:
        return self.split_value(value) is not None
