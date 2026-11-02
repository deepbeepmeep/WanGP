# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Spatial upsampler plugin API.

Every spatial upsampler (built-in or extension) class is listed in
``spatial_upsampler_handlers`` and declares its capabilities through
``query_upsampler_def()``:

```python
{
    "name": "FlashVSR",                      # display name
    "upsampler_types": ("postprocessing",),  # "postprocessing" and/or "vae"
    "media": ("video", "image"),             # media kinds the upsampler can process ("video", "image", "audio", "3d")
    "profile": "video",                      # memory profile kind: video, image, or audio
    "config_key": "flashvsr",                # optional subkey under wgp_config["spatial_upsamplers"]
    "pos": 20,                               # default dropdown order for this handler's methods
    "method_pos": {"flashvsr": 20},          # optional per-method order; independent of multiplier
    "methods": [("FlashVSR", "flashvsr")],   # interchangeable post-processing methods (label, method key)
    "vae_methods": [],                       # VAE methods (label, method key); model-pipeline integration
    "multipliers": {"flashvsr": (2.0, 4.0)}, # optional; omit when a refiner has no scale
    "default_spatial_upsampling": "flashvsr*2",
    "postprocessing_category": "upsampler",   # "upsampler" or "refiner"
    "source_audio_conditioning": False,        # request a decoded source-audio input without changing final remux audio
    "description": "Restore detail while spatially upscaling media.", # processor-owned help/discovery description
    "method_descriptions": {"flashvsr": "..."}, # optional descriptions per method
    "method_parameters": {"flashvsr": [{      # optional runtime/UI/assistant descriptors
        "name": "flashvsr_strength",          # id in spatial_postprocessing_parameters
        "setting": "strength",                # upscale() keyword; defaults to name
        "type": "number",
        "component": "slider",
        "ui": ("postprocessing", "late_postprocessing"),
        "required": False,
        "default": 0.5,
        "minimum": 0.0,
        "maximum": 1.0,
        "description": "Restoration strength.",
    }]},
}
```

Decoded-media handlers declare category ``upsampler`` or ``refiner``. Methods
without declared multipliers serialize as their bare method key and do not show
a Scale dropdown. Handler descriptions power both the selector's dynamic help
and postprocessing discovery. Parameter values travel in the
``spatial_postprocessing_parameters`` setting under their ids, the UI shows them
below the slot's selector (``postprocessing.parameter_ui``: sliders, dropdowns,
a text input), and an images parameter reads the reference images input shared
with the temporal slot (``postprocessing_reference_images``). Assistant
discovery keeps only call-relevant parameter fields and omits UI/runtime
presentation metadata.

Handlers must also implement:
- ``is_upsampling(value)``: does this handler own this ``spatial_upsampling`` value?
- ``split_value(value)`` -> ``(method, scale)`` or ``None``
- ``build_value(method, scale)`` -> ``spatial_upsampling`` value or ``None``
- ``validate_upsampling(value, image_mode)`` -> error text ("" when valid)

``wgp.py`` calls ``register_spatial_upsamplers(...)`` once and then accesses
handlers through this API instead of keeping per-upsampler globals.

Post-processing ("postprocessing" type) handlers are interchangeable and must
additionally implement ``upscale(sample, value, **kwargs)`` and may implement
``load_upsampler(value, **kwargs)``, ``download(...)``, ``enabled()`` and
``release_vram()``. They are automatically offered for late post-processing of
existing media. VAE ("vae") handlers are plugged into model pipelines through
the generic VAE upsampler hooks below; model defs declare support.

Discovery tests the existing optional ``enabled()`` method first. When it is
absent, handlers may expose a ``status`` property containing ``"enabled"`` or
``"disabled"``. Discovery reports ``"unknown"`` only when neither contract
provides a valid status. Disabled handlers may expose ``reason_disabled``.
Handlers may implement ``method_available(method)`` and ``method_visible(method)``.
Both default to True: hidden methods remain usable and are added to menus only
when restoring their selection; unavailable methods are excluded even then.

Handlers may also expose Config-tab controls with ``create_config_ui(...)`` and
normalize their own nested section under ``wgp_config["spatial_upsamplers"]``.
The shared Postprocessor VRAM Usage policy controls profile 0. At most one spatial
upsampler handler is retained: dispatching another handler releases the previous
one before loading the new selection. Handlers still own incompatible variant
switches within one handler (for example PiD version/backbone changes).

Upsamplers that allocate their own mmgp offload object must register it in
``shared.utils.offload_registry`` so its resources can be tracked and released
centrally (WanGP unload tool).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from contextlib import contextmanager
from contextvars import ContextVar
import hashlib
import importlib
import inspect
import sys
from typing import Any

from shared.attention import attention_shared_state
from shared.utils import offload_registry
from . import memory as processor_memory
from . import parameter_ui
from .model_context import compatible_loaded_model
from .processor_status import PROCESSOR_STATUS_DISABLED, PROCESSOR_STATUS_ENABLED, PROCESSOR_STATUS_UNKNOWN, handler_reason_disabled, handler_status, handler_method_in_choices

# Backward compatibility for external plugins written against the old module name.
sys.modules.setdefault("postprocessing.upsamplers", sys.modules[__name__])

UPSAMPLER_TYPE_POSTPROCESSING = "postprocessing"
UPSAMPLER_TYPE_VAE = "vae"
POSTPROCESSING_CATEGORY_UPSAMPLER = "upsampler"
POSTPROCESSING_CATEGORY_REFINER = "refiner"
POSTPROCESSING_CATEGORIES = (POSTPROCESSING_CATEGORY_UPSAMPLER, POSTPROCESSING_CATEGORY_REFINER)
PARAMETER_UI_POSTPROCESSING = "postprocessing"
PARAMETER_UI_LATE_POSTPROCESSING = "late_postprocessing"
PARAMETER_UI_MEDIA_FLOW = "media_flow"
UPSAMPLER_PROFILE_VIDEO = "video"
UPSAMPLER_PROFILE_IMAGE = "image"
UPSAMPLER_PROFILE_AUDIO = "audio"
UPSAMPLER_PROFILE_3D = "3d"  # meshes: the spatial slot holds 3D refiners
UPSAMPLER_CONFIG_KEY = "spatial_upsamplers"
PERSISTENCE_CONFIG_KEY = "persistence"
PERSIST_UNLOAD = 1
PERSIST_RAM = 2
PERSIST_AUTO = 3  # Legacy config migration value.
MULTIPLIER_SEPARATOR = "*"

spatial_upsampler_handlers = [
    "postprocessing.lanczos.wgp_bridge.LanczosUpsampler",
    "postprocessing.dlss5.spatial_upsampler.DLSS5SpatialUpsampler",
    "postprocessing.flashvsr.wgp_bridge.FlashVSRBridge",
    "postprocessing.seedvr2.wgp_bridge.SeedVR2Bridge",
    "postprocessing.pid.wgp_bridge.PiDBridge",
    "postprocessing.h3_face_refiner.wgp_bridge.H3FaceRefinerBridge",
    "postprocessing.chain_of_zoom.wgp_bridge.ChainOfZoomBridge",
    "postprocessing.ltx2_upsampler.wgp_bridge.LTXVideoUpsamplerBridge",
    "postprocessing.hi3d_refiner.wgp_bridge.Hi3DRefinerBridge",
    "postprocessing.spatial_upsamplers.WanVaeUpsampler",
    "models.minimax_h3.vae_upsampler.MiniMaxH3VaeUpsampler",
]
_upsampler_handlers: list[Any] = []
_registered_upsampler_handler_paths: set[str] = set()
_active_upsampler_handler: Any | None = None
_upsampler_server_config: dict[str, Any] | None = None


@dataclass
class UpsamplerConfigBinding:
    handler: Any | None
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


def format_method_label(label: str) -> str:
    return str(label or "").removesuffix(" Upsampler")


def format_method_scale_label(label: str, scale: float) -> str:
    return f"{format_method_label(label)} {format_multiplier_label(scale)}"


def register_upsampler(handler) -> None:
    if handler not in _upsampler_handlers:
        _upsampler_handlers.append(handler)


def _load_upsampler_class(path: str):
    module_path, class_name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)


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
    for path in spatial_upsampler_handlers if handler_modules is None else handler_modules:
        handler_cls = _load_upsampler_class(str(path or "").strip())
        if not hasattr(handler_cls, "default_config"):
            continue
        config = dict(handler_cls.default_config())
        config.pop(PERSISTENCE_CONFIG_KEY, None)
        if not config:
            continue
        handler_def = handler_cls.query_upsampler_def()
        sections[_config_key_from_handler_def(handler_def, handler_cls.__name__)] = config
    return sections


def register_spatial_upsamplers(server_config, files_locator, handler_modules: list[str] | None = None) -> None:
    global _upsampler_server_config

    _upsampler_server_config = server_config
    modules = spatial_upsampler_handlers if handler_modules is None else handler_modules
    for path in modules:
        path = str(path or "").strip()
        if not path or path in _registered_upsampler_handler_paths:
            continue
        register_upsampler(_load_upsampler_class(path)(server_config, files_locator))
        _registered_upsampler_handler_paths.add(path)


_ui_queries = ContextVar("spatial_upsampler_ui_queries", default=None)


@contextmanager
def ui_query_scope():
    """Reuse discovery within one form build, never across pages or refreshes."""
    token = _ui_queries.set({})
    try:
        yield
    finally:
        _ui_queries.reset(token)


def _ui_query(handler, kind, query):
    cache = _ui_queries.get()
    if cache is None:
        return query()
    key = (id(handler), kind)
    if key not in cache:
        cache[key] = query()
    return cache[key]


def _handler_def(handler):
    return _ui_query(handler, "definition", handler.query_upsampler_def)


def upsampler_handlers(upsampler_type: str | None = None, enabled_only: bool = False) -> list[Any]:
    handlers = []
    for handler in _upsampler_handlers:
        if upsampler_type is not None and upsampler_type not in _handler_def(handler).get("upsampler_types", ()):
            continue
        if enabled_only and not handler_enabled(handler):
            continue
        handlers.append(handler)
    return handlers


def handler_enabled(handler) -> bool:
    return not hasattr(handler, "enabled") or _ui_query(handler, "enabled", handler.enabled)


def _handler_name(handler) -> str:
    return str(_handler_def(handler)["name"])


def _release_upsampler_handler(handler) -> None:
    global _active_upsampler_handler

    released = offload_registry.release_all([_handler_name(handler)])
    if not released and hasattr(handler, "release_vram"):
        handler.release_vram()
    if _active_upsampler_handler is handler:
        _active_upsampler_handler = None


def _activate_upsampler(handler) -> None:
    global _active_upsampler_handler

    if _active_upsampler_handler is handler:
        return
    if _active_upsampler_handler is not None:
        _release_upsampler_handler(_active_upsampler_handler)
    _active_upsampler_handler = handler


def query_upsampler_defs(upsampler_type: str | None = None, enabled_only: bool = False) -> list[dict[str, Any]]:
    return [_handler_def(handler) for handler in upsampler_handlers(upsampler_type, enabled_only)]


def _method_choices(handler_def: dict[str, Any]) -> list[tuple[str, str]]:
    return handler_def.get("methods", []) + handler_def.get("vae_methods", [])


def _method_labels(handler_def: dict[str, Any]) -> dict[str, str]:
    return {key: label for label, key in _method_choices(handler_def)}


def method_definition(method) -> tuple[Any | None, dict[str, Any], str]:
    handler = find_upsampler_by_method(method)
    if handler is None:
        return None, {}, str(method or "").strip()
    return handler, _handler_def(handler), str(method or "").strip()


def method_description(method) -> str:
    handler, handler_def, method = method_definition(method)
    if handler is None:
        return ""
    descriptions = handler_def.get("method_descriptions", {})
    if isinstance(descriptions, dict) and method in descriptions:
        description = str(descriptions[method] or "").strip()
        if description:
            return description
    return str(handler_def.get("description", "") or "").strip()


def method_category(method) -> str:
    _handler, handler_def, _method = method_definition(method)
    category = str(handler_def.get("postprocessing_category", POSTPROCESSING_CATEGORY_UPSAMPLER) or "").strip().lower()
    return category if category in POSTPROCESSING_CATEGORIES else POSTPROCESSING_CATEGORY_UPSAMPLER


def method_progress_label(method, default="Upsampling - Starting") -> str:
    _handler, handler_def, _method = method_definition(method)
    return str(handler_def.get("progress_label", default) or default)


def method_parameters(method, *, ui_context: str | None = None) -> list[dict[str, Any]]:
    _handler, handler_def, method = method_definition(method)
    parameters = handler_def.get("method_parameters", {})
    values = parameters.get(method, ()) if isinstance(parameters, dict) else ()
    if not isinstance(values, (list, tuple)):
        return []
    output = []
    for parameter in values:
        if not isinstance(parameter, dict) or not str(parameter.get("name", "") or "").strip():
            continue
        contexts = parameter.get("ui", ())
        contexts = (contexts,) if isinstance(contexts, str) else tuple(contexts)
        if ui_context is not None and ui_context not in contexts:
            continue
        output.append(dict(parameter))
    return output


def method_uses_setting(method, setting: str) -> bool:
    return any(str(parameter.get("setting", parameter["name"])) == setting for parameter in method_parameters(method))


def runtime_parameter_kwargs(spatial_upsampling, parameter_values) -> dict[str, Any]:
    split = split_upsampling_value(spatial_upsampling)
    return {} if split is None else parameter_ui.runtime_kwargs(method_parameters(split[0]), parameter_values)


def _handler_method_label(handler, label: str, method: str) -> str:
    return handler.format_method_label(label, method) if hasattr(handler, "format_method_label") else label


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


def find_upsampler(spatial_upsampling) -> Any | None:
    if not str(spatial_upsampling or "").strip():
        return None
    return next((handler for handler in _upsampler_handlers if handler.is_upsampling(spatial_upsampling)), None)


def find_postprocessing_upsampler(spatial_upsampling) -> Any | None:
    handler = find_upsampler(spatial_upsampling)
    if handler is None:
        return None
    method = handler.split_value(spatial_upsampling)[0]
    return handler if method in [key for _, key in _handler_def(handler).get("methods", [])] else None


def resolve_late_postprocessing_prompt(spatial_upsampling, prompt) -> str:
    prompt = str(prompt or "").strip()
    if prompt:
        return prompt
    handler = find_postprocessing_upsampler(spatial_upsampling)
    return "" if handler is None else str(_handler_def(handler).get("default_prompt", "")).strip()


def find_vae_upsampler(spatial_upsampling) -> Any | None:
    handler = find_upsampler(spatial_upsampling)
    if handler is None:
        return None
    method = handler.split_value(spatial_upsampling)[0]
    return handler if method in [key for _, key in _handler_def(handler).get("vae_methods", [])] else None


def is_vae_upsampling(spatial_upsampling) -> bool:
    return find_vae_upsampler(spatial_upsampling) is not None


def query_download_defs(enabled_only: bool = True) -> list[dict[str, Any]]:
    """Inventory optional assets through their owners, without downloading them."""
    import inspect

    definitions = []
    for handler in _upsampler_handlers:
        if enabled_only and handler_status(handler) == PROCESSOR_STATUS_DISABLED:
            continue
        query = getattr(handler, "query_download_defs", None) or getattr(handler, "query_download_def", None)
        if query is None:
            continue
        kwargs = {"enabled_only": enabled_only} if "enabled_only" in inspect.signature(query).parameters else {}
        result = query(**kwargs)
        if result is not None:
            definitions.extend(result if isinstance(result, list) else [result])
    return definitions


def download_for_value(spatial_upsampling, process_files, **kwargs):
    handler = find_upsampler(spatial_upsampling)
    if handler is None or not hasattr(handler, "download"):
        return False
    kwargs.setdefault("status_text", f"Downloading {_handler_def(handler)['name']} Model Files")
    return handler.download(process_files, spatial_upsampling=spatial_upsampling, **kwargs)


_progress_compat_warnings = set()


def _unit_compatible_progress(progress_callback):
    """Adapt plugin wrappers written for the older progress_callback(phase, current, total) by dropping the optional unit."""
    try:
        inspect.signature(progress_callback).bind(None, None, None, None)
        return progress_callback
    except TypeError:
        pass
    parts = (inspect.getsourcefile(progress_callback) or "unknown").replace("\\", "/").split("/")
    owner = next((f"Plugin '{parts[i + 1]}'" for i, part in enumerate(parts[:-1]) if part.lower() == "plugins"), "/".join(parts))
    if owner not in _progress_compat_warnings:
        _progress_compat_warnings.add(owner)
        print(f"[Upsamplers] Warning: {owner} wraps the post-processing progress callback without its optional 'unit' argument; progress units will not be shown. Please update it.")
    return lambda phase, current=None, total=None, unit=None: progress_callback(phase, current, total)


def upscale_postprocessing(handler, sample, spatial_upsampling, *, main_offloadobj=None, loaded_model_context=None, parameter_values=None, **kwargs):
    """parameter_values: the slot's runtime values (parameter_ui.runtime_values), passed to the processor as its declared parameters (under
    their setting names)."""
    if kwargs.get("progress_callback") is not None:
        kwargs["progress_callback"] = _unit_compatible_progress(kwargs["progress_callback"])
    _activate_upsampler(handler)
    persistent = persistent_models()
    kwargs["profile"] = 0
    if kwargs.get("init_pipe") is None:
        kwargs["init_pipe"] = partial(processor_memory.init_pipe, server_config=_upsampler_server_config)
    name = _handler_def(handler)["name"]
    kwargs.update(runtime_parameter_kwargs(spatial_upsampling, parameter_values))
    borrowed_context = compatible_loaded_model(handler, spatial_upsampling, loaded_model_context, **kwargs)
    with attention_shared_state():
        try:
            core_offloadobj = loaded_model_context.offloadobj if loaded_model_context is not None else main_offloadobj
            if core_offloadobj is not None:
                core_offloadobj.unload_all()
            if borrowed_context is not None and hasattr(handler, "release_private_runtime"):
                handler.release_private_runtime()
            elif borrowed_context is None and hasattr(handler, "load_upsampler"):
                handler.load_upsampler(spatial_upsampling, **kwargs)
            return handler.upscale(sample, spatial_upsampling, loaded_model_context=borrowed_context, **kwargs)
        finally:
            if borrowed_context is not None:
                borrowed_context.offloadobj.unload_all()
                _release_upsampler_handler(handler)
            elif persistent:
                offload_registry.unload_vram([name])
            else:
                _release_upsampler_handler(handler)


def handler_media(handler) -> tuple[str, ...]:
    return tuple(_handler_def(handler).get("media", (UPSAMPLER_PROFILE_VIDEO, UPSAMPLER_PROFILE_IMAGE)))


def validate_postprocessing_spatial_upsampling(spatial_upsampling, image_mode: int, *, media_kind: str | None = None) -> str:
    """media_kind: the processed media when it is neither a video nor an image (``3d``); image_mode tells those apart otherwise."""
    if is_vae_upsampling(spatial_upsampling):
        return "VAE Spatial Upsampling is only available during generation"
    edit_upsampler = find_postprocessing_upsampler(spatial_upsampling)
    if len(spatial_upsampling) > 0 and edit_upsampler is None:
        return f"No spatial upsampler registered for '{spatial_upsampling}'"
    kind = media_kind or (UPSAMPLER_PROFILE_IMAGE if image_mode else UPSAMPLER_PROFILE_VIDEO)
    if edit_upsampler is not None and kind not in handler_media(edit_upsampler):
        return f"{_handler_def(edit_upsampler)['name']} does not process {'3D assets' if kind == UPSAMPLER_PROFILE_3D else kind + 's'}"
    if edit_upsampler is not None:
        error = edit_upsampler.validate_upsampling(spatial_upsampling, image_mode)
        if error:
            return error
    return ""


def _handler_supports_model_vae_method(handler, method: str, model_type, model_def, image_mode: int) -> bool:
    return hasattr(handler, "supports_model_vae_method") and handler.supports_model_vae_method(method, model_type, model_def, image_mode)


def query_model_vae_method_choices(model_type, model_def, image_mode: int) -> list[tuple[str, str]]:
    choices = []
    for handler in upsampler_handlers(UPSAMPLER_TYPE_VAE):
        handler_def = _handler_def(handler)
        for label, method in handler_def.get("vae_methods", []):
            if _handler_supports_model_vae_method(handler, method, model_type, model_def, image_mode):
                choices.append((_method_pos(handler_def, method), str(label or "").casefold(), str(method or ""), _handler_method_label(handler, label, method), method))
    return [(label, method) for _, _, _, label, method in sorted(choices)]


def validate_model_vae_upsampling(spatial_upsampling, image_mode: int, model_type, model_def, medium: str) -> str:
    handler = find_vae_upsampler(spatial_upsampling)
    if handler is None:
        return ""
    if hasattr(handler, "validate_model_vae_upsampling"):
        return handler.validate_model_vae_upsampling(spatial_upsampling, image_mode, model_type, model_def, medium)
    method = handler.split_value(spatial_upsampling)[0]
    return "" if _handler_supports_model_vae_method(handler, method, model_type, model_def, image_mode) else f"{format_upsampling_label(spatial_upsampling)} is not available for {medium}"


def model_load_vae_upsampling_value(spatial_upsampling, model_type, model_def, image_mode: int) -> str | None:
    handler = find_vae_upsampler(spatial_upsampling)
    if handler is None or not hasattr(handler, "model_load_upsampling_value"):
        return None
    return handler.model_load_upsampling_value(spatial_upsampling, model_type, model_def, image_mode)


def loaded_model_vae_upsampling_value(model) -> str | None:
    for handler in upsampler_handlers(UPSAMPLER_TYPE_VAE):
        if hasattr(handler, "loaded_model_vae_upsampling_value"):
            value = handler.loaded_model_vae_upsampling_value(model)
            if value is not None:
                return value
    return None


def model_load_kwargs_for_vae_upsampling(spatial_upsampling, model_type, model_def, image_mode: int) -> dict[str, Any]:
    handler = find_vae_upsampler(spatial_upsampling)
    if handler is not None:
        _activate_upsampler(handler)
    if handler is None or not hasattr(handler, "model_load_kwargs_for_vae_upsampling"):
        return {}
    return handler.model_load_kwargs_for_vae_upsampling(spatial_upsampling, model_type, model_def, image_mode)


def post_model_process_vae_upsampling(sample, spatial_upsampling):
    handler = find_vae_upsampler(spatial_upsampling)
    if handler is None or not hasattr(handler, "post_model_process_vae_upsampling"):
        return sample
    return handler.post_model_process_vae_upsampling(sample, spatial_upsampling)


def has_post_model_process_vae_upsampling(spatial_upsampling) -> bool:
    handler = find_vae_upsampler(spatial_upsampling)
    return handler is not None and hasattr(handler, "post_model_process_vae_upsampling")


def resize_x2_vae_output(sample, scale):
    """Brings the output of a VAE decoder that always upsamples x2 to the selected multiplier."""
    if scale == 2.0:
        return sample
    from PIL import Image
    from postprocessing.lanczos import resize_lanczos_spatial

    height, width = sample.shape[-2:]
    return resize_lanczos_spatial(sample, None, size=(round(height * scale / 2), round(width * scale / 2)), method=Image.Resampling.BICUBIC)


def prepare_vae_upsampler(handler, spatial_upsampling, **kwargs):
    if handler is None or not hasattr(handler, "prepare_vae_upsampler"):
        return None
    _activate_upsampler(handler)
    return handler.prepare_vae_upsampler(spatial_upsampling, **kwargs)


def release_vae_upsampler(handler, session) -> None:
    if handler is not None and session is not None and not persistent_models():
        _release_upsampler_handler(handler)


def find_upsampler_by_method(method) -> Any | None:
    method = str(method or "").strip()
    if not method:
        return None
    for handler in _upsampler_handlers:
        handler_def = _handler_def(handler)
        if method in [key for _, key in _method_choices(handler_def)]:
            return handler
    return None


def require_upsampler_by_method(method) -> Any:
    handler = find_upsampler_by_method(method)
    if handler is None:
        raise RuntimeError(f"No spatial upsampler registered for method '{method}'")
    return handler


def method_multipliers(method) -> tuple[float, ...]:
    handler = find_upsampler_by_method(method)
    if handler is None:
        return ()
    return tuple(_handler_def(handler).get("multipliers", {}).get(str(method), ()))


def ratio_choices_for_method(method) -> list[tuple[str, float]]:
    return [(format_multiplier_label(scale), scale) for scale in method_multipliers(method)]


def _default_multiplier_from_def(handler_def: dict[str, Any], method: str) -> float | None:
    multipliers = tuple(handler_def.get("multipliers", {}).get(method, ()))
    if not multipliers:
        return None
    default_value = handler_def.get("default_spatial_upsampling", "")
    handler = find_upsampler(default_value)
    if handler is not None:
        split = handler.split_value(default_value)
        if split is not None and split[0] == method and split[1] in multipliers:
            return split[1]
    return multipliers[0]


def default_multiplier_for_method(method) -> float:
    handler = find_upsampler_by_method(method)
    if handler is None:
        return 2.0
    handler_def = _handler_def(handler)
    if not tuple(handler_def.get("multipliers", {}).get(str(method or "").strip(), ())):
        return 1.0
    return _default_multiplier_from_def(handler_def, str(method or "").strip()) or 2.0


def normalize_multiplier_for_method(method, scale) -> float:
    multipliers = method_multipliers(method)
    if not multipliers:
        return default_multiplier_for_method(method)
    try:
        scale = float(scale)
    except (TypeError, ValueError):
        scale = default_multiplier_for_method(method)
    return scale if scale in multipliers else default_multiplier_for_method(method)


def split_upsampling_value(value) -> tuple[str, float] | None:
    handler = find_upsampler(value)
    return None if handler is None else handler.split_value(value)


def build_upsampling_value(method, scale) -> str | None:
    handler = find_upsampler_by_method(method)
    return None if handler is None else handler.build_value(method, scale)


def normalize_upsampling_value(value) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    split = split_upsampling_value(text)
    if split is None:
        return text
    handler = find_upsampler_by_method(split[0])
    normalized = None if handler is None else handler.build_value(*split)
    return normalized or text


def format_upsampling_label(value) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    handler = find_upsampler(text)
    if handler is None:
        return text
    split = handler.split_value(text)
    if split is None:
        return text
    method, scale = split
    label = _method_labels(_handler_def(handler)).get(method)
    if label:
        label = _handler_method_label(handler, label, method)
    return (format_method_scale_label(label, scale) if method_multipliers(method) else label) if label else text


def normalize_upsampling_state(method, scale) -> tuple[list[tuple[str, float]], float | None, str]:
    method = str(method or "").strip()
    ratio_choices = ratio_choices_for_method(method)
    if not method:
        ratio_choices = ratio_choices_for_method("lanczos")
    scale = normalize_multiplier_for_method(method or "lanczos", scale) if ratio_choices else None
    return ratio_choices, scale, "" if not method else build_upsampling_value(method, scale) or ""


def normalize_upsampling_value_for_method(method, current_value) -> tuple[list[tuple[str, float]], float, str]:
    split = split_upsampling_value(current_value)
    return normalize_upsampling_state(method, 2.0 if split is None else split[1])


def _method_choice_sort_key(choice: tuple[str, str]) -> tuple[float, str, str]:
    label, method = choice
    handler = find_upsampler_by_method(method)
    position = 1000 if handler is None else _method_pos(_handler_def(handler), method)
    return position, str(label or "").casefold(), str(method or "")


def media_kind_of(image_outputs: bool = False, media_kind: str | None = None) -> str:
    return media_kind or (UPSAMPLER_PROFILE_IMAGE if image_outputs else UPSAMPLER_PROFILE_VIDEO)


def query_postprocessing_method_choices(image_outputs: bool = False, late_postprocessing: bool = False, *, selected_method: str = "", media_kind: str | None = None) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    video_post_choices, image_post_choices = [], []
    kind = media_kind_of(image_outputs, media_kind)
    for handler in upsampler_handlers(UPSAMPLER_TYPE_POSTPROCESSING):
        handler_def = _handler_def(handler)
        method_keys = [key for _, key in handler_def.get("methods", [])]
        if "lanczos" not in method_keys and not (late_postprocessing or handler_enabled(handler)):
            continue
        media = handler_def.get("media", (UPSAMPLER_PROFILE_VIDEO, UPSAMPLER_PROFILE_IMAGE))
        if kind not in media:
            continue
        choices = [(_method_pos(handler_def, method), str(label or "").casefold(), str(method or ""), _handler_method_label(handler, label, method), method) for label, method in handler_def.get("methods", []) if handler_method_in_choices(handler, method, selected_method=selected_method)]
        if UPSAMPLER_PROFILE_VIDEO in media:
            video_post_choices += choices
        else:
            image_post_choices += choices
    return [(label, method) for _, _, _, label, method in sorted(video_post_choices)], [(label, method) for _, _, _, label, method in sorted(image_post_choices)]


def dropdown_state(spatial_upsampling, *, image_outputs: bool = False, late_postprocessing: bool = False, vae_choices: list[tuple[str, str]] | None = None, excluded_methods: set[str] | None = None, exclude_method_fn=None, media_kind: str | None = None) -> dict[str, Any]:
    method, scale = split_upsampling_value(spatial_upsampling) or ("", 2.0)
    video_post_choices, image_post_choices = query_postprocessing_method_choices(image_outputs=image_outputs, late_postprocessing=late_postprocessing, selected_method=method, media_kind=media_kind)
    excluded_methods = excluded_methods or set()
    video_post_choices = [choice for choice in video_post_choices if choice[1] not in excluded_methods and not (exclude_method_fn and exclude_method_fn(choice[1]))]
    image_post_choices = [choice for choice in image_post_choices if choice[1] not in excluded_methods and not (exclude_method_fn and exclude_method_fn(choice[1]))]
    method_choices = [("None", "")] + sorted(video_post_choices + image_post_choices + list(vae_choices or []), key=_method_choice_sort_key)
    if method not in {value for _, value in method_choices}:
        method = ""
    ratio_choices, scale, value = normalize_upsampling_state(method, scale)
    return {"method": method, "scale": scale, "value": value, "method_choices": method_choices, "ratio_choices": ratio_choices}


def late_postprocessing_ui_state(spatial_upsampling, *, image_outputs: bool, media_kind: str | None = None) -> dict[str, Any]:
    """The state of the late panel's slot for the selected media, with its choices' context (the slot's context state)."""
    context = {"image_outputs": image_outputs, "late_postprocessing": True, "media_kind": media_kind}
    return {**dropdown_state(spatial_upsampling, **context), "context": context}


def spatial_help_markdown(method_choices, *, media_profile: str, field_help=None) -> str:
    intro = ("3D refiners add fine geometric detail to a mesh from its reference image, keeping its position and scale." if media_profile == UPSAMPLER_PROFILE_3D else
             getattr(field_help, "SPATIAL_UPSAMPLER_HELP_INTRO", "Spatial upsamplers increase resolution. Visual refiners improve targeted decoded content without necessarily changing its dimensions."))
    sections = [intro]
    handler_choices = {}
    for label, method in method_choices:
        handler = find_upsampler_by_method(method)
        if handler is not None:
            handler_choices.setdefault(handler, []).append((label, method))
    for handler, choices in handler_choices.items():
        handler_def = _handler_def(handler)
        name = str(handler_def.get("name", choices[0][0]))
        category = slot_label(media_profile) if media_profile == UPSAMPLER_PROFILE_3D else "Visual Refiner" if str(handler_def.get("postprocessing_category", POSTPROCESSING_CATEGORY_UPSAMPLER)).lower() == POSTPROCESSING_CATEGORY_REFINER else "Spatial Upsampler"
        description = str(handler_def.get("description", "") or "").strip()
        media_descriptions = handler_def.get("media_descriptions", {})
        media_description = str(media_descriptions.get(media_profile, "") or "").strip() if isinstance(media_descriptions, dict) else ""
        lines = [f"### {name} — {category}"]
        if description:
            lines.append(description)
        if media_description and media_description != description:
            lines.append(media_description)
        method_descriptions = handler_def.get("method_descriptions", {})
        if isinstance(method_descriptions, dict):
            details = [(str(label), str(method_descriptions.get(method, "") or "").strip()) for label, method in choices]
            details = [(label, detail) for label, detail in details if detail and detail not in (description, media_description)]
            lines += [f"- **{label}:** {detail}" for label, detail in details]
        sections.append("\n\n".join(lines))
    return "\n\n".join(sections)


def spatial_help_popup(method_choices, *, media_profile: str, field_help=None) -> tuple[str, str]:
    return slot_label(media_profile), spatial_help_markdown(method_choices, media_profile=media_profile, field_help=field_help)


def spatial_help_id(method_choices, *, media_profile: str) -> str:
    signature = f"{media_profile}|" + "|".join(str(method) for _label, method in method_choices)
    return f"spatial_upsampling_{hashlib.sha1(signature.encode('utf-8')).hexdigest()[:12]}"


def slot_label(media_kind: str) -> str:
    return "3D Refiner" if media_kind == UPSAMPLER_PROFILE_3D else "Spatial Upsampler / Visual Refiner"


def create_generation_spatial_ui(gr, spatial_upsampling, *, image_outputs: bool = False, late_postprocessing: bool = False, vae_choices: list[tuple[str, str]] | None = None, excluded_methods: set[str] | None = None, exclude_method_fn=None, elem_classes=None, update_form: bool = False, field_help=None, help_target_id: str | None = None, media_kind: str | None = None) -> dict[str, Any]:
    """The spatial slot's selectors; its processors' parameters are in the shared inputs of parameter_ui. The context of its choices
    (returned in extra_components and media_outputs) follows the model shown (form refresh) or the selected media (late panel): the
    slot's events are bound once, for the first model."""
    method, scale = split_upsampling_value(spatial_upsampling) or ("", 2.0)
    context = {"image_outputs": image_outputs, "late_postprocessing": late_postprocessing, "vae_choices": vae_choices, "excluded_methods": excluded_methods, "exclude_method_fn": exclude_method_fn, "media_kind": media_kind}
    state = dropdown_state(build_upsampling_value(method, scale) or "", **context)
    media_profile = media_kind_of(image_outputs, media_kind)
    with gr.Row():
        method_component = gr.Dropdown(choices=state["method_choices"], value=state["method"], visible=True, scale=3, label=slot_label(media_profile), elem_id=help_target_id, elem_classes=elem_classes)
        if field_help is not None:
            help_title, help_markdown = spatial_help_popup(state["method_choices"], media_profile=media_profile, field_help=field_help)
            help_id = spatial_help_id(state["method_choices"], media_profile=media_profile)
            help_component = gr.update(value=field_help.render_marker(help_target_id, help_id, title=help_title, markdown=help_markdown)) if update_form else field_help.bind(method_component, help_id, title=help_title, markdown=help_markdown)
        ratio_component = gr.Dropdown(choices=state["ratio_choices"], value=state["scale"] if state["ratio_choices"] else None, visible=bool(state["method"] and state["ratio_choices"]), scale=1, label="Scale", elem_classes=elem_classes)
    value_component = gr.Textbox(value=state["value"], visible=False, elem_classes=elem_classes)
    if field_help is None:
        help_component = gr.Markdown(value=spatial_help_markdown(state["method_choices"], media_profile=media_profile), visible=True, elem_classes=elem_classes)

    context_component = context if update_form else gr.State(context)

    def refresh_method(method, value, context):
        _, _, value = normalize_upsampling_value_for_method(method, value)
        method_state = dropdown_state(value, **context)
        method, ratio_choices, scale, value = (method_state[key] for key in ("method", "ratio_choices", "scale", "value"))
        return (gr.update(choices=method_state["method_choices"], value=method), gr.update(choices=ratio_choices, value=scale if ratio_choices else None, visible=bool(method and ratio_choices)),
                value)

    def refresh_ratio(method, scale):
        _, scale, value = normalize_upsampling_state(method, scale)
        return gr.update(value=scale, visible=bool(method and method_multipliers(method))), value

    if not update_form:
        gr.on(triggers=[method_component.input], fn=refresh_method, inputs=[method_component, value_component, context_component], outputs=[method_component, ratio_component, value_component], show_progress="hidden")
        gr.on(triggers=[ratio_component.input], fn=refresh_ratio, inputs=[method_component, ratio_component], outputs=[ratio_component, value_component], show_progress="hidden")
    return {"value": value_component, "method": method_component, "method_value": state["method"], "ratio": ratio_component, "help": help_component, "help_target_id": help_target_id or method_component.elem_id,
            "extra_components": [help_component, context_component], "media_outputs": [help_component, context_component]}


def query_postprocessing_upsampling_choices(include_name: bool = True, enabled_only: bool = False, image_outputs: bool | None = None, *, include_hidden: bool = False) -> list[tuple[str, str]]:
    """Flat (label, value) choices covering every method x multiplier of the post-processing upsamplers."""
    choices = []
    for handler in upsampler_handlers(UPSAMPLER_TYPE_POSTPROCESSING, enabled_only):
        handler_def = _handler_def(handler)
        if image_outputs is not None:
            media = handler_def.get("media", ("video", "image"))
            if ("image" if image_outputs else "video") not in media:
                continue
        multipliers = handler_def.get("multipliers", {})
        for label, method in handler_def.get("methods", []):
            if not handler_method_in_choices(handler, method, include_hidden=include_hidden):
                continue
            declared_multipliers = tuple(multipliers.get(method, ()))
            for scale in declared_multipliers or (1.0,):
                value = handler.build_value(method, scale)
                if value is not None:
                    display_label = format_method_scale_label(label, scale) if declared_multipliers else str(label)
                    choices.append((_method_pos(handler_def, method), str(label or "").casefold(), float(scale), str(value or ""), display_label if include_name else (format_multiplier_label(scale) if declared_multipliers else str(label)), value))
    return [(label, value) for _, _, _, _, label, value in sorted(choices)]


def profile_type_for_handler(handler) -> str:
    handler_def = _handler_def(handler)
    profile = str(handler_def.get("profile", "") or "").strip().lower()
    if profile in (UPSAMPLER_PROFILE_VIDEO, UPSAMPLER_PROFILE_IMAGE, UPSAMPLER_PROFILE_AUDIO):
        return profile
    if getattr(handler, "uses_image_profile", False):
        return UPSAMPLER_PROFILE_IMAGE
    media = handler_def.get("media", ())
    if media == (UPSAMPLER_PROFILE_IMAGE,):
        return UPSAMPLER_PROFILE_IMAGE
    if media == (UPSAMPLER_PROFILE_AUDIO,):
        return UPSAMPLER_PROFILE_AUDIO
    return UPSAMPLER_PROFILE_VIDEO


def config_key_for_handler(handler) -> str:
    handler_def = _handler_def(handler)
    config_key = str(handler_def.get("config_key", "") or "").strip()
    if config_key:
        return config_key
    method_choices = _method_choices(handler_def)
    if method_choices:
        return method_choices[0][1]
    return handler.__class__.__name__.lower()


def _nested_configs(server_config: dict[str, Any]) -> dict[str, Any]:
    configs = server_config.get(UPSAMPLER_CONFIG_KEY, {})
    if not isinstance(configs, dict):
        configs = {}
        server_config[UPSAMPLER_CONFIG_KEY] = configs
    else:
        server_config.setdefault(UPSAMPLER_CONFIG_KEY, configs)
    return configs


def normalize_persistence(value) -> int:
    try:
        value = int(value)
    except (TypeError, ValueError):
        value = PERSIST_UNLOAD
    return value if value in (PERSIST_UNLOAD, PERSIST_RAM, PERSIST_AUTO) else PERSIST_UNLOAD


def persistence(server_config: dict[str, Any] | None = None) -> int:
    config = _upsampler_server_config if server_config is None else server_config
    if config is None:
        raise RuntimeError("Spatial upsamplers are not registered")
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


def _legacy_config(handler, server_config: dict[str, Any]) -> dict[str, Any]:
    return dict(handler.legacy_config(server_config)) if hasattr(handler, "legacy_config") else {}


def _legacy_config_keys(handler) -> tuple[str, ...]:
    return tuple(handler.legacy_config_keys()) if hasattr(handler, "legacy_config_keys") else ()


def _normalize_config_section(handler, config: dict[str, Any]) -> dict[str, Any]:
    config = dict(handler.normalize_config_section(config)) if hasattr(handler, "normalize_config_section") else dict(config)
    config.pop(PERSISTENCE_CONFIG_KEY, None)
    return config


def read_config_section(server_config: dict[str, Any], handler, *, prefer_legacy: bool = False) -> dict[str, Any]:
    configs = server_config.get(UPSAMPLER_CONFIG_KEY, {})
    config_key = config_key_for_handler(handler)
    nested = configs.get(config_key, {}) if isinstance(configs, dict) else {}
    nested = nested if isinstance(nested, dict) else {}
    nested_exists = isinstance(configs, dict) and config_key in configs
    legacy = _legacy_config(handler, server_config) if prefer_legacy or not nested_exists else {}
    values = {**_default_config(handler), **dict(nested or {}), **legacy}
    return _normalize_config_section(handler, values)


def read_config_section_by_key(server_config: dict[str, Any], config_key: str) -> dict[str, Any]:
    handler = next((candidate for candidate in _upsampler_handlers if config_key_for_handler(candidate) == config_key), None)
    if handler is None:
        configs = server_config.get(UPSAMPLER_CONFIG_KEY, {})
        return dict(configs.get(config_key, {})) if isinstance(configs, dict) else {}
    return read_config_section(server_config, handler)


def write_config_section(server_config: dict[str, Any], handler, config: dict[str, Any]) -> dict[str, Any]:
    config = _normalize_config_section(handler, {**_default_config(handler), **dict(config or {})})
    _nested_configs(server_config)[config_key_for_handler(handler)] = config
    return config


def migrate_upsampler_config(server_config: dict[str, Any], *, prefer_legacy: bool = False, apply_pre_1_1_defaults: bool = False) -> bool:
    legacy_keys = tuple(dict.fromkeys(key for handler in _upsampler_handlers for key in _legacy_config_keys(handler)))
    before = repr((server_config.get(UPSAMPLER_CONFIG_KEY, None), {key: server_config.get(key) for key in legacy_keys if key in server_config}))
    _nested_configs(server_config).pop(PERSISTENCE_CONFIG_KEY, None)
    for handler in _upsampler_handlers:
        if _default_config(handler) or hasattr(handler, "legacy_config"):
            config = write_config_section(server_config, handler, read_config_section(server_config, handler, prefer_legacy=prefer_legacy))
            if apply_pre_1_1_defaults and hasattr(handler, "apply_pre_1_1_defaults") and handler.apply_pre_1_1_defaults(config):
                write_config_section(server_config, handler, config)
    for key in legacy_keys:
        server_config.pop(key, None)
    after = repr((server_config.get(UPSAMPLER_CONFIG_KEY, None), {key: server_config.get(key) for key in legacy_keys if key in server_config}))
    return before != after


def config_for_method(method, server_config: dict[str, Any] | None = None) -> dict[str, Any]:
    handler = require_upsampler_by_method(method)
    return read_config_section(handler.server_config if server_config is None else server_config, handler)


def create_config_ui(gr, server_config: dict[str, Any], *, lock_config: bool = False) -> list[UpsamplerConfigBinding]:
    bindings = []
    handlers = [handler for handler in _upsampler_handlers if hasattr(handler, "create_config_ui")]
    if not handlers:
        return bindings
    with gr.Tabs():
        for handler in handlers:
            config = write_config_section(server_config, handler, read_config_section(server_config, handler))
            with gr.Tab(_handler_def(handler)["name"]) as tab:
                controls = handler.create_config_ui(gr, config, lock_config=lock_config)
            tab.visible = bool(controls)
            if controls:
                bindings.append(UpsamplerConfigBinding(handler, config_key_for_handler(handler), list(controls)))
    return bindings


def config_components(bindings: list[UpsamplerConfigBinding]) -> list[Any]:
    return [component for binding in bindings for _, component in binding.controls]


def collect_config_update(bindings: list[UpsamplerConfigBinding], values) -> dict[str, dict[str, Any]]:
    values = list(values or [])
    updates, index = {}, 0
    for binding in bindings:
        config = {}
        for field, _ in binding.controls:
            if index >= len(values):
                raise ValueError("Spatial upsampler config UI values do not match registered controls")
            config[field] = values[index]
            index += 1
        updates[binding.config_key] = _normalize_config_section(binding.handler, {**_default_config(binding.handler), **config})
    if index != len(values):
        raise ValueError("Spatial upsampler config UI values do not match registered controls")
    return updates


def validate_config_update_messages(bindings: list[UpsamplerConfigBinding], updates: dict[str, dict[str, Any]]) -> list[str]:
    messages = []
    for binding in bindings:
        if binding.config_key in updates and hasattr(binding.handler, "validate_config_section"):
            message = binding.handler.validate_config_section(updates[binding.config_key])
            if isinstance(message, str) and message:
                messages.append(message)
            elif isinstance(message, list):
                messages += [text for text in message if text]
    return messages


def apply_config_update(server_config: dict[str, Any], bindings: list[UpsamplerConfigBinding], updates: dict[str, dict[str, Any]]) -> None:
    for binding in bindings:
        if binding.config_key in updates:
            write_config_section(server_config, binding.handler, updates[binding.config_key])


def release_changed_config_upsamplers(old_config: dict[str, Any], new_config: dict[str, Any], changed_keys) -> None:
    changed_keys = set(changed_keys)
    memory_changed = processor_memory.config_values(old_config) != processor_memory.config_values(new_config)
    released_handler = _active_upsampler_handler if memory_changed else None
    if released_handler is not None:
        _release_upsampler_handler(released_handler)
    for handler in _upsampler_handlers:
        if not hasattr(handler, "release_vram"):
            continue
        old_section = read_config_section(old_config, handler)
        new_section = read_config_section(new_config, handler)
        if hasattr(handler, "config_requires_release"):
            should_release = handler.config_requires_release(old_section, new_section, changed_keys)
        else:
            should_release = old_section != new_section
        if (should_release or memory_changed) and handler is not released_handler:
            _release_upsampler_handler(handler)


class SimpleScaleSuffixMixin:
    """Value helpers writing '<method>*<multiplier>' while accepting the legacy concatenated form."""

    def _method_keys(self):
        handler_def = self.query_upsampler_def()
        return [key for _, key in _method_choices(handler_def)]

    def split_value(self, value):
        text = str(value or "").strip().lower()
        for method in sorted(self._method_keys(), key=len, reverse=True):
            if text.startswith(method):
                suffix = text[len(method):]
                multipliers = tuple(self.query_upsampler_def().get("multipliers", {}).get(method, ()))
                if not multipliers:
                    return (method, 1.0) if not suffix else None
                scale = parse_multiplier_suffix(text, method, 2.0)
                return None if scale is None else (method, scale)
        return None

    def build_value(self, method, scale):
        method = str(method or "").strip().lower()
        if method not in self._method_keys():
            return None
        multipliers = self.query_upsampler_def().get("multipliers", {}).get(method, ())
        if not multipliers:
            return method
        scale = float(scale or 0)
        if scale not in multipliers:
            scale = _default_multiplier_from_def(self.query_upsampler_def(), method) or 0
        return format_multiplier_value(method, scale)

    def is_upsampling(self, value) -> bool:
        return self.split_value(value) is not None


class WanVaeUpsampler(SimpleScaleSuffixMixin):
    """Capability declaration for the Wan VAE 2x upscaling decoder.

    VAE upsamplers are plugged directly into model pipelines: models declare
    support through their model def and the upsampler API decides whether a main
    model reload or an external VAE upsampler runtime is needed.
    """

    def __init__(self, server_config=None, files_locator=None):
        pass

    def query_upsampler_def(self) -> dict[str, Any]:
        return {
            "name": "VAE Upscaling",
            "upsampler_types": (UPSAMPLER_TYPE_VAE,),
            "media": ("video", "image"),
            "profile": UPSAMPLER_PROFILE_VIDEO,
            "config_key": "vae",
            "pos": 30,
            "method_pos": {"vae": 30},
            "methods": [],
            "vae_methods": [("VAE Upscaling", "vae")],
            "multipliers": {"vae": (1.0, 1.5, 2.0)},
            "default_spatial_upsampling": "vae*2",
            "postprocessing_category": POSTPROCESSING_CATEGORY_UPSAMPLER,
            "description": "Runs through the compatible generation model's existing VAE path, so it adds no separate decoded-media pass and has little extra VRAM impact. It can create more detail than Lanczos.",
        }

    def validate_upsampling(self, spatial_upsampling, image_mode: int) -> str:
        split = self.split_value(spatial_upsampling)
        return "" if split is not None and split[1] in self.query_upsampler_def()["multipliers"]["vae"] else "VAE Spatial Upsampling only supports x1, x1.5 and x2"

    def supports_model_vae_method(self, method, model_type, model_def, image_mode: int) -> bool:
        return method == "vae" and image_mode in model_def.get("vae_upsampler", [])

    def validate_model_vae_upsampling(self, spatial_upsampling, image_mode: int, model_type, model_def, medium: str) -> str:
        error = self.validate_upsampling(spatial_upsampling, image_mode)
        if error:
            return error
        return "" if self.supports_model_vae_method("vae", model_type, model_def, image_mode) else f"VAE Spatial Upsampling is not available for {medium}"

    def model_load_upsampling_value(self, spatial_upsampling, model_type, model_def, image_mode: int) -> str | None:
        return spatial_upsampling if self.supports_model_vae_method("vae", model_type, model_def, image_mode) else None

    def loaded_model_vae_upsampling_value(self, model) -> str | None:
        return None if model is None or not hasattr(model, "vae") or not hasattr(model.vae, "upsampling_set") else model.vae.upsampling_set

    def model_load_kwargs_for_vae_upsampling(self, spatial_upsampling, model_type, model_def, image_mode: int) -> dict[str, Any]:
        return {"VAE_upsampling": spatial_upsampling}

    def post_model_process_vae_upsampling(self, sample, spatial_upsampling):
        return resize_x2_vae_output(sample, self.split_value(spatial_upsampling)[1])
