# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


_GetModelDef = Callable[[str], dict[str, Any] | None]
_Resolver = Callable[[dict[str, Any] | None, dict[str, Any]], Any]
_MODEL_DEF_PROVIDER: _GetModelDef | None = None
_CUSTOM_SETTINGS_MAX = 5


@dataclass(frozen=True)
class SettingDef:
    key: str
    label: str
    type: str
    min: int | float | None = None
    max: int | float | None = None
    step: int | float | None = None
    visible: bool = True
    custom: bool = False


@dataclass(frozen=True)
class ContainerDef:
    key: str
    visible: bool = True


@dataclass(frozen=True)
class _SettingSpec:
    key: str
    label: str | _Resolver
    type: str | _Resolver
    min: int | float | None | _Resolver = None
    max: int | float | None | _Resolver = None
    step: int | float | None | _Resolver = None
    visible: bool | _Resolver = True
    active_when: bool | _Resolver = True
    custom: bool = False
    containers: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ContainerSpec:
    key: str
    visible: bool | _Resolver = True


_SETTING_SPECS: dict[str, _SettingSpec] = {}
_SETTING_ORDER: list[str] = []
_CONTAINER_SPECS: dict[str, _ContainerSpec] = {}


def configure(*, get_model_def: _GetModelDef | None = None) -> None:
    global _MODEL_DEF_PROVIDER
    if get_model_def is not None:
        _MODEL_DEF_PROVIDER = get_model_def


def _normalize_type(value: Any) -> str:
    kind = str(value or "number").strip().lower()
    if kind in {"int", "integer"}:
        return "integer"
    if kind in {"float", "number"}:
        return "number"
    return "string"


def _resolve(value: Any, model_def: dict[str, Any] | None, context: dict[str, Any]) -> Any:
    return value(model_def, context) if callable(value) else value


def _normalize_number(value: Any) -> int | float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        parsed = float(str(value).strip())
    except Exception:
        return None
    return int(parsed) if parsed.is_integer() else parsed


def _format_number(value: int | float | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return f"{value:g}" if isinstance(value, float) else str(value)


def _trim_label(label: str, fallback: str) -> str:
    label = " ".join(str(label or "").strip().split())
    trimmed = label.split("(", 1)[0].strip()
    return trimmed or fallback


def _resolve_model_def(configs: dict[str, Any] | None, model_def: dict[str, Any] | None = None) -> dict[str, Any] | None:
    if isinstance(model_def, dict):
        return model_def
    if not isinstance(configs, dict) or not callable(_MODEL_DEF_PROVIDER):
        return None
    model_type = str(configs.get("model_type", "") or "").strip()
    return _MODEL_DEF_PROVIDER(model_type) if len(model_type) > 0 else None


def _with_data_context(data: dict[str, Any] | None, **context: Any) -> dict[str, Any]:
    merged = dict(context)
    if isinstance(data, dict):
        merged.setdefault("settings", data)
        model_type = str(data.get("model_type", "") or "").strip()
        if len(model_type) > 0 and "model_type" not in merged:
            merged["model_type"] = model_type
    return merged


def _get_context_get_max_frames(context: dict[str, Any]) -> Callable[[int], int]:
    callback = context.get("get_max_frames", None)
    return callback if callable(callback) else (lambda frames: int(frames))


def _context_model_type(context: dict[str, Any]) -> str:
    return str(context.get("model_type", "") or "").strip().lower()


def _context_model_type_contains(context: dict[str, Any], *parts: str) -> bool:
    model_type = _context_model_type(context)
    return any(part.lower() in model_type for part in parts)


def _context_setting(context: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in context:
        return context[key]
    settings = context.get("settings", None)
    return settings.get(key, default) if isinstance(settings, dict) else default


def _find_custom_setting(model_def: dict[str, Any] | None, setting_id: str) -> dict[str, Any] | None:
    custom_settings = (model_def or {}).get("custom_settings", [])
    if not isinstance(custom_settings, list):
        return None
    for setting in custom_settings[:_CUSTOM_SETTINGS_MAX]:
        if not isinstance(setting, dict):
            continue
        if str(setting.get("id", "") or "").strip() == setting_id:
            return setting
    return None


def _custom_label(setting_id: str, fallback: str) -> _Resolver:
    def resolver(model_def: dict[str, Any] | None, _context: dict[str, Any]) -> str:
        setting = _find_custom_setting(model_def, setting_id)
        if not isinstance(setting, dict):
            return fallback
        return str(setting.get("label", setting.get("name", fallback)) or fallback)
    return resolver


def _custom_type(setting_id: str, fallback: str) -> _Resolver:
    def resolver(model_def: dict[str, Any] | None, _context: dict[str, Any]) -> str:
        setting = _find_custom_setting(model_def, setting_id)
        return _normalize_type(setting.get("type", fallback) if isinstance(setting, dict) else fallback)
    return resolver


def _custom_bound(setting_id: str, bound: str, fallback: int | float | None) -> _Resolver:
    def resolver(model_def: dict[str, Any] | None, _context: dict[str, Any]) -> int | float | None:
        setting = _find_custom_setting(model_def, setting_id)
        if not isinstance(setting, dict):
            return fallback
        return _normalize_number(setting.get(bound, fallback))
    return resolver


def _show_if_flag(flag_name: str) -> _Resolver:
    def resolver(model_def: dict[str, Any] | None, _context: dict[str, Any]) -> bool:
        if model_def is None:
            return True
        return bool((model_def or {}).get(flag_name, False))
    return resolver


def _show_if(predicate: Callable[[dict[str, Any]], bool]) -> _Resolver:
    def resolver(model_def: dict[str, Any] | None, _context: dict[str, Any]) -> bool:
        if model_def is None:
            return True
        return bool(predicate(model_def or {}))
    return resolver


def _show_if_guidance_phase(phase: int) -> _Resolver:
    return _show_if(lambda model_def: int(model_def.get("guidance_max_phases", 0) or 0) >= phase and int(model_def.get("visible_phases", phase) or 0) >= phase)


def _active_if_guidance_phase(phase: int) -> _Resolver:
    def resolver(model_def: dict[str, Any] | None, context: dict[str, Any]) -> bool:
        if bool((model_def or {}).get("lock_guidance_phases", False)):
            return int((model_def or {}).get("guidance_max_phases", 0) or 0) >= phase
        value = _context_setting(context, "guidance_phases", None)
        if value is None:
            return True
        return int(value) >= phase
    return resolver


def _show_if_custom(setting_id: str) -> _Resolver:
    return _show_if(lambda model_def: _find_custom_setting(model_def, setting_id) is not None)


def _show_if_text(key: str) -> _Resolver:
    return _show_if(lambda model_def: len(str(model_def.get(key, "") or "").strip()) > 0)


def _show_if_not_none(key: str) -> _Resolver:
    return _show_if(lambda model_def: model_def.get(key, None) is not None)


def _show_if_int_at_least(key: str, minimum: int) -> _Resolver:
    return _show_if(lambda model_def: int(model_def.get(key, 0) or 0) >= minimum)


def _switch_threshold_label(model_def: dict[str, Any] | None, context: dict[str, Any]) -> str:
    custom_label = str(_model_setting_config(model_def, "switch_threshold").get("label", "") or "").strip()
    if custom_label:
        return custom_label
    guidance_phases = 1
    try:
        guidance_phases = int(context.get("guidance_phases", 1) or 0)
    except Exception:
        pass
    if guidance_phases >= 3:
        return "Phase 1-2"
    return "Model / Guidance Switch Threshold" if bool((model_def or {}).get("multiple_submodels", False)) else "Guidance Switch Threshold"


def _model_label(key: str, fallback: str) -> _Resolver:
    def resolver(model_def: dict[str, Any] | None, _context: dict[str, Any]) -> str:
        text = str((model_def or {}).get(key, "") or "").strip()
        return text or fallback
    return resolver


def _model_setting_config(model_def: dict[str, Any] | None, key: str) -> dict[str, Any]:
    return (model_def or {}).get(key, {})


def _model_setting_label(key: str, fallback: str) -> _Resolver:
    def resolver(model_def: dict[str, Any] | None, _context: dict[str, Any]) -> str:
        text = _model_setting_config(model_def, key).get("label", "").strip()
        return text or fallback
    return resolver


def _model_setting_bound(key: str, bound: str, fallback: int | float | None) -> _Resolver:
    def resolver(model_def: dict[str, Any] | None, _context: dict[str, Any]) -> int | float | None:
        return _normalize_number(_model_setting_config(model_def, key).get(bound, fallback))
    return resolver


def _model_setting_type(key: str, fallback: str) -> _Resolver:
    def resolver(model_def: dict[str, Any] | None, _context: dict[str, Any]) -> str:
        return _normalize_type(_model_setting_config(model_def, key).get("type", fallback))
    return resolver


def _show_if_model_setting_label(key: str) -> _Resolver:
    return _show_if(lambda model_def: len(_model_setting_config(model_def, key).get("label", "").strip()) > 0)


def _control_net_label(index: int) -> _Resolver:
    def resolver(model_def: dict[str, Any] | None, _context: dict[str, Any]) -> str:
        name = str((model_def or {}).get("control_net_weight_name", "") or "").strip() or "Control Net"
        size = int((model_def or {}).get("control_net_weight_size", 0) or 0)
        suffix = "" if size <= 1 and index == 1 else f" #{index}"
        return f"{name} Weight{suffix}"
    return resolver


def _control_net_alt_label(model_def: dict[str, Any] | None, _context: dict[str, Any]) -> str:
    name = str((model_def or {}).get("control_net_weight_alt_name", "") or "").strip()
    return f"{name} Weight" if len(name) > 0 else "Control Net Alt Weight"


def _audio_scale_label(model_def: dict[str, Any] | None, _context: dict[str, Any]) -> str:
    return str((model_def or {}).get("audio_scale_name", "") or "").strip() or "Audio Scale"


def _guide_setting_visible(model_def: dict[str, Any]) -> bool:
    return model_def.get("guide_custom_choices", None) is not None or model_def.get("guide_custom_choices_image", None) is not None or model_def.get("guide_preprocessing", None) is not None


def _guidance_row_visible(model_def: dict[str, Any]) -> bool:
    return int(model_def.get("guidance_max_phases", 0) or 0) >= 1 and int(model_def.get("visible_phases", 1) or 0) >= 1


def _guidance_phases_row_visible(model_def: dict[str, Any]) -> bool:
    return int(model_def.get("guidance_max_phases", 0) or 0) >= 2 and (int(model_def.get("visible_phases", 2) or 0) >= 2 or bool(model_def.get("switch_threshold")) or bool(model_def.get("inference_steps2")))


def _embedded_guidance_row_visible(model_def: dict[str, Any]) -> bool:
    return bool(model_def.get("embedded_guidance", False)) or bool(model_def.get("audio_guidance", False)) or model_def.get("alt_guidance", None) is not None or model_def.get("alt_scale", None) is not None


def _masking_visible(model_def: dict[str, Any]) -> bool:
    return _guide_setting_visible(model_def) or bool(model_def.get("mask_strength_always_enabled", False))


def _sliding_window_visible(model_def: dict[str, Any]) -> bool:
    return bool(model_def.get("sliding_window", False))


def _sample_solver_row_visible(model_def: dict[str, Any]) -> bool:
    return model_def.get("sample_solvers", None) is not None or bool(model_def.get("flow_shift", False)) or bool(model_def.get("custom_sigmas", False))


def _control_net_visible(index: int) -> _Resolver:
    return _show_if(lambda model_def: int(model_def.get("control_net_weight_size", 0) or 0) >= index and len(str(model_def.get("control_net_weight_name", "") or "").strip()) > 0)


def _control_net_weights_row_visible(model_def: dict[str, Any]) -> bool:
    return int(model_def.get("control_net_weight_size", 0) or 0) >= 1 or len(str(model_def.get("control_net_weight_alt_name", "") or "").strip()) > 0 or len(str(model_def.get("audio_scale_name", "") or "").strip()) > 0


def _sliding_window_defaults(model_def: dict[str, Any] | None) -> dict[str, Any]:
    defaults = (model_def or {}).get("sliding_window_defaults", {})
    return defaults if isinstance(defaults, dict) else {}


def _sliding_window_size_min(model_def: dict[str, Any] | None, _context: dict[str, Any]) -> int:
    defaults = _sliding_window_defaults(model_def)
    value = defaults.get("window_min", 5)
    try:
        return int(value)
    except Exception:
        return 5


def _sliding_window_size_max(model_def: dict[str, Any] | None, context: dict[str, Any]) -> int:
    defaults = _sliding_window_defaults(model_def)
    raw_max = defaults.get("window_max", 257)
    try:
        raw_max = int(raw_max)
    except Exception:
        raw_max = 257
    return int(_get_context_get_max_frames(context)(raw_max))


def _sliding_window_size_step(model_def: dict[str, Any] | None, _context: dict[str, Any]) -> int:
    defaults = _sliding_window_defaults(model_def)
    raw_step = defaults.get("window_step", (model_def or {}).get("frames_steps", 4))
    try:
        return int(raw_step)
    except Exception:
        return 4


def _sliding_window_overlap_bound(bound: str, fallback: int) -> _Resolver:
    def resolver(model_def: dict[str, Any] | None, _context: dict[str, Any]) -> int:
        defaults = _sliding_window_defaults(model_def)
        try:
            return int(defaults.get(bound, fallback))
        except Exception:
            return fallback
    return resolver


def _temperature_visible(model_def: dict[str, Any]) -> bool:
    return bool(model_def.get("audio_only", False)) and bool(model_def.get("temperature", True))


def _sliding_window_overlap_noise_visible(model_def: dict[str, Any] | None, context: dict[str, Any]) -> bool:
    return bool((model_def or {}).get("vace_class", False)) or _context_model_type_contains(context, "sky_df", "diffusion_forcing")


def _sliding_window_discard_last_frames_visible(model_def: dict[str, Any] | None, context: dict[str, Any]) -> bool:
    return bool((model_def or {}).get("sliding_window", False)) and not _context_model_type_contains(context, "sky_df", "diffusion_forcing")


def _add_container(key: str, visible: bool | _Resolver = True) -> None:
    _CONTAINER_SPECS[key] = _ContainerSpec(key, visible=visible)


def _add_setting(
    key: str,
    label: str | _Resolver,
    type: str | _Resolver,
    *,
    min: int | float | None | _Resolver = None,
    max: int | float | None | _Resolver = None,
    step: int | float | None | _Resolver = None,
    visible: bool | _Resolver = True,
    active_when: bool | _Resolver = True,
    custom: bool = False,
    containers: tuple[str, ...] | list[str] = (),
) -> None:
    _SETTING_SPECS[key] = _SettingSpec(key, label, type, min=min, max=max, step=step, visible=visible, active_when=active_when, custom=custom, containers=tuple(containers))
    _SETTING_ORDER.append(key)


def get_container_def(key: str, model_def: dict[str, Any] | None = None, **context: Any) -> ContainerDef:
    spec = _CONTAINER_SPECS[key]
    return ContainerDef(key=spec.key, visible=bool(_resolve(spec.visible, model_def, context)))


def get_def(key: str, model_def: dict[str, Any] | None = None, **context: Any) -> SettingDef:
    spec = _SETTING_SPECS[key]
    visible = bool(_resolve(spec.visible, model_def, context)) and bool(_resolve(spec.active_when, model_def, context))
    for container_key in spec.containers:
        visible = visible and get_container_def(container_key, model_def, **context).visible
    return SettingDef(
        key=spec.key,
        label=str(_resolve(spec.label, model_def, context) or spec.key),
        type=_normalize_type(_resolve(spec.type, model_def, context)),
        min=_normalize_number(_resolve(spec.min, model_def, context)),
        max=_normalize_number(_resolve(spec.max, model_def, context)),
        step=_normalize_number(_resolve(spec.step, model_def, context)),
        visible=visible,
        custom=bool(spec.custom),
    )


def iter_defs(model_def: dict[str, Any] | None = None, *, only_visible: bool = False, keys: list[str] | tuple[str, ...] | None = None, **context: Any) -> dict[str, SettingDef]:
    names = _SETTING_ORDER if keys is None else keys
    resolved = {key: get_def(key, model_def, **context) for key in names if key in _SETTING_SPECS}
    return {key: one for key, one in resolved.items() if one.visible} if only_visible else resolved


def validate_setting_value(label: str, value: Any, setting_type: str, min_value: int | float | None = None, max_value: int | float | None = None) -> str | None:
    setting_type = _normalize_type(setting_type)
    if value is None or isinstance(value, str) and len(value.strip()) == 0:
        return None
    if setting_type == "string":
        return None
    if isinstance(value, bool):
        kind = "integer" if setting_type == "integer" else "number"
        return f"{label} must be a {kind}."
    if setting_type == "integer":
        try:
            parsed = int(value)
        except Exception:
            try:
                parsed_float = float(value)
            except Exception:
                return f"{label} must be an integer."
            if not parsed_float.is_integer():
                return f"{label} must be an integer."
            parsed = int(parsed_float)
    else:
        try:
            parsed = float(value)
        except Exception:
            return f"{label} must be a number."
    if min_value is not None and parsed < min_value:
        return f"{label} must be at least {_format_number(min_value)}."
    if max_value is not None and parsed > max_value:
        return f"{label} must be at most {_format_number(max_value)}."
    return None


def validate_resolved_value(setting_def: SettingDef, value: Any) -> str | None:
    return validate_setting_value(_trim_label(setting_def.label, setting_def.key), value, setting_def.type, setting_def.min, setting_def.max)


def validate_inputs(inputs: dict[str, Any] | None, model_def: dict[str, Any] | None = None, **context: Any) -> str:
    if not isinstance(inputs, dict):
        return ""
    context = _with_data_context(inputs, **context)
    custom_settings = inputs.get("custom_settings", None)
    custom_settings = custom_settings if isinstance(custom_settings, dict) else {}
    for key, setting_def in iter_defs(model_def, only_visible=True, **context).items():
        raw_value = custom_settings.get(key, None) if setting_def.custom else inputs.get(key, None)
        error = validate_resolved_value(setting_def, raw_value)
        if error is not None:
            return error
    return ""


def get_info(configs: dict[str, Any] | None, model_def: dict[str, Any] | None = None, **context: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(configs, dict):
        return {}
    context = _with_data_context(configs, **context)
    model_def = _resolve_model_def(configs, model_def)
    if not isinstance(model_def, dict):
        return {}
    resolved_defs = iter_defs(model_def, only_visible=True, guidance_phases=configs.get("guidance_phases", 1), **context)
    custom_settings = configs.get("custom_settings", None)
    custom_settings = custom_settings if isinstance(custom_settings, dict) else {}
    info: dict[str, dict[str, Any]] = {}
    for key in _SETTING_ORDER:
        setting_def = resolved_defs.get(key, None)
        if setting_def is None:
            continue
        raw_value = custom_settings.get(key, None) if setting_def.custom else configs.get(key, None)
        if raw_value is None or isinstance(raw_value, str) and len(raw_value.strip()) == 0:
            continue
        label = _trim_label(setting_def.label, setting_def.key)
        if label in info and str(info[label].get("key", "")) != key:
            label = f"{label} [{key}]"
        info[label] = {
            "key": key,
            "value": raw_value,
            "type": setting_def.type,
            "custom": setting_def.custom,
            "min": setting_def.min,
            "max": setting_def.max,
            "step": setting_def.step,
        }
    return info


def get_summary_label(key: str, model_def: dict[str, Any] | None = None, fallback: str | None = None, **context: Any) -> str:
    setting_def = get_def(key, model_def, **context)
    name = _model_setting_config(model_def, key).get("name", "").strip()
    return name or _trim_label(setting_def.label, fallback or setting_def.key)


_add_container("guidance_row", visible=_show_if(_guidance_row_visible))
_add_container("guidance_phases_row", visible=_show_if(_guidance_phases_row_visible))
_add_container("embedded_guidance_row", visible=_show_if(_embedded_guidance_row_visible))
_add_container("temperature_row", visible=_show_if(_temperature_visible))
_add_container("sample_solver_row", visible=_show_if(_sample_solver_row_visible))
_add_container("control_net_weights_row", visible=_show_if(_control_net_weights_row_visible))
_add_container("NAG_col", visible=_show_if_flag("NAG"))
_add_container("top_pk_row", visible=_show_if(lambda model_def: bool(model_def.get("top_p_slider", False)) or bool(model_def.get("top_k_slider", False))))
_add_container("motion_amplitude_col", visible=_show_if_flag("motion_amplitude"))


_add_setting("guidance_scale", "Guidance (CFG)", "number", min=1.0, max=20.0, step=0.1, visible=_show_if_guidance_phase(1), containers=("guidance_row",))
_add_setting("guidance2_scale", "Guidance2 (CFG)", "number", min=1.0, max=20.0, step=0.1, visible=_show_if_guidance_phase(2), containers=("guidance_row",))
_add_setting("guidance3_scale", "Guidance3 (CFG)", "number", min=1.0, max=20.0, step=0.1, visible=_show_if_guidance_phase(3), containers=("guidance_row",))
_add_setting("switch_threshold", _switch_threshold_label, _model_setting_type("switch_threshold", "integer"), min=_model_setting_bound("switch_threshold", "min", 0), max=_model_setting_bound("switch_threshold", "max", 1000), step=_model_setting_bound("switch_threshold", "step", 1), visible=_show_if_int_at_least("guidance_max_phases", 2), active_when=_active_if_guidance_phase(2), containers=("guidance_phases_row",))
_add_setting("num_inference_steps2", "Phase 2 Steps", "integer", min=1, max=100, step=1, visible=_show_if(lambda model_def: model_def.get("inference_steps2", False) and model_def.get("guidance_max_phases", 0) >= 2), active_when=_active_if_guidance_phase(2), containers=("guidance_phases_row",))
_add_setting("custom_sigmas", "Custom Sigmas", "string", visible=_show_if_flag("custom_sigmas"), containers=("sample_solver_row",))
_add_setting("switch_threshold2", "Phase 2-3", "integer", min=0, max=1000, step=1, visible=_show_if_int_at_least("guidance_max_phases", 3), containers=("guidance_phases_row",))
_add_setting("alt_guidance_scale", _model_label("alt_guidance", "Alternate Guidance"), "number", min=1.0, max=20.0, step=0.5, visible=_show_if_not_none("alt_guidance"), containers=("embedded_guidance_row",))
_add_setting("alt_scale", _model_label("alt_scale", "Alt Scale"), "number", min=0.0, max=1.0, step=0.05, visible=_show_if_not_none("alt_scale"), containers=("embedded_guidance_row",))
_add_setting("audio_guidance_scale", "Audio Guidance", "number", min=1.0, max=20.0, step=0.5, visible=_show_if_flag("audio_guidance"), containers=("embedded_guidance_row",))
_add_setting("audio_scale", _audio_scale_label, "number", min=0.0, max=1.0, step=0.01, visible=_show_if_text("audio_scale_name"), containers=("control_net_weights_row",))
_add_setting("flow_shift", "Shift Scale", "number", min=1.0, max=25.0, step=0.1, visible=_show_if_flag("flow_shift"), containers=("sample_solver_row",))
_add_setting("embedded_guidance_scale", "Embedded Guidance Scale", "number", min=1.0, max=20.0, step=0.5, visible=_show_if_flag("embedded_guidance"), containers=("embedded_guidance_row",))
_add_setting("denoising_strength", _model_setting_label("denoising_strength", "Denoising Strength (the Lower the Closer to the Control Image/Video)"), "number", min=0.0, max=1.0, step=0.01, visible=_show_if(_guide_setting_visible))
_add_setting("masking_strength", _model_setting_label("masking_strength", "Masking Strength (the Lower the More Freedom for Unmasked Area)"), "number", min=0.0, max=1.0, step=0.01, visible=_show_if(_masking_visible))
_add_setting("control_net_weight", _control_net_label(1), "number", min=0.0, max=2.0, step=0.01, visible=_control_net_visible(1), containers=("control_net_weights_row",))
_add_setting("control_net_weight2", _control_net_label(2), "number", min=0.0, max=2.0, step=0.01, visible=_control_net_visible(2), containers=("control_net_weights_row",))
_add_setting("control_net_weight_alt", _control_net_alt_label, "number", min=0.0, max=2.0, step=0.01, visible=_show_if_text("control_net_weight_alt_name"), containers=("control_net_weights_row",))
_add_setting("motion_amplitude", "Motion Amplitude", "number", min=1.0, max=1.4, step=0.01, visible=_show_if_flag("motion_amplitude"), containers=("motion_amplitude_col",))
_add_setting("mask_expand", "Expand / Shrink Mask Area", "integer", min=-10, max=50, step=1, visible=_show_if(_guide_setting_visible))
_add_setting("input_video_strength", _model_setting_label("input_video_strength", "Input Video Strength"), "number", min=0.0, max=1.0, step=0.01, visible=_show_if_model_setting_label("input_video_strength"))
_add_setting("attention_sparsity", _model_setting_label("attention_sparsity", "Attention Sparsity"), "number", min=_model_setting_bound("attention_sparsity", "start", 0.0), max=_model_setting_bound("attention_sparsity", "end", 4.0), step=_model_setting_bound("attention_sparsity", "inc", 0.05), visible=_show_if(lambda model_def: any(mode.get("supports_sparsity", False) for mode in model_def.get("custom_attention_modes", {}).values())))
_add_setting("image_refs_relative_size", _model_setting_label("image_refs_relative_size", "Rescale Internaly Image Ref (% in relation to Output Video) to change Output Composition"), "integer", min=_model_setting_bound("image_refs_relative_size", "min", 20), max=_model_setting_bound("image_refs_relative_size", "max", 100), step=_model_setting_bound("image_refs_relative_size", "step", 1), visible=_show_if_flag("any_image_refs_relative_size"))
_add_setting("sliding_window_size", "Sliding Window Size", "integer", min=_sliding_window_size_min, max=_sliding_window_size_max, step=_sliding_window_size_step, visible=_show_if(_sliding_window_visible))
_add_setting("sliding_window_overlap", "Windows Frames Overlap (needed to maintain continuity between windows, a higher value will require more windows)", "integer", min=_sliding_window_overlap_bound("overlap_min", 1), max=_sliding_window_overlap_bound("overlap_max", 97), step=_sliding_window_overlap_bound("overlap_step", 4), visible=_show_if(_sliding_window_visible))
_add_setting("sub_parallel_window_size", "Sub Parallel Window Size (0 = disabled)", "integer", min=0, max=_sliding_window_size_max, step=_sliding_window_size_step, visible=_show_if_flag("sub_parallel_windows"))
_add_setting("sub_parallel_window_overlap", "Sub Parallel Window Overlap", "integer", min=_sliding_window_overlap_bound("overlap_min", 1), max=_sliding_window_overlap_bound("overlap_max", 97), step=_sliding_window_overlap_bound("overlap_step", 4), visible=_show_if_flag("sub_parallel_windows"))
_add_setting("sliding_window_color_correction_strength", "Color Correction Strength (match colors of new window with previous one, 0 = disabled)", "number", min=0.0, max=1.0, step=0.01, visible=_show_if_flag("color_correction"))
_add_setting("sliding_window_overlap_noise", "Noise to be added to overlapped frames to reduce blur effect", "integer", min=0, max=150, step=1, visible=_sliding_window_overlap_noise_visible)
_add_setting("sliding_window_discard_last_frames", "Discard Last Frames of a Window (that may have bad quality)", "integer", min=0, max=20, step=_sliding_window_overlap_bound("discard_step", 4), visible=_sliding_window_discard_last_frames_visible)
_add_setting("sliding_window_trim_first_frames", "Trim First Frames in First Window or if there is no Overlap Frames", "integer", min=0, max=10, step=1, visible=_show_if(_sliding_window_visible))
_add_setting("temperature", "Temperature", "number", min=0.1, max=_model_setting_bound("temperature_slider", "end", 1.5), step=0.01, visible=_show_if(_temperature_visible), containers=("temperature_row",))
_add_setting("top_p", "Top-p", "number", min=0.0, max=1.0, step=0.01, visible=_show_if_flag("top_p_slider"), containers=("top_pk_row",))
_add_setting("top_k", "Top-k (0 = disabled)", "integer", min=0, max=100, step=1, visible=_show_if_flag("top_k_slider"), containers=("top_pk_row",))
_add_setting("NAG_scale", _model_setting_label("NAG_scale", "NAG Scale"), "number", min=_model_setting_bound("NAG_scale", "min", 1.0), max=_model_setting_bound("NAG_scale", "max", 20.0), step=_model_setting_bound("NAG_scale", "step", 0.01), containers=("NAG_col",))
_add_setting("NAG_tau", _model_setting_label("NAG_tau", "NAG Tau"), "number", min=_model_setting_bound("NAG_tau", "min", 1.0), max=_model_setting_bound("NAG_tau", "max", 5.0), step=_model_setting_bound("NAG_tau", "step", 0.01), containers=("NAG_col",))
_add_setting("NAG_alpha", _model_setting_label("NAG_alpha", "NAG Alpha"), "number", min=_model_setting_bound("NAG_alpha", "min", 0.0), max=_model_setting_bound("NAG_alpha", "max", 2.0), step=_model_setting_bound("NAG_alpha", "step", 0.01), containers=("NAG_col",))
_add_setting("pace", _custom_label("pace", "Pace"), _custom_type("pace", "number"), min=_custom_bound("pace", "min", 0.2), max=_custom_bound("pace", "max", 1.0), custom=True, visible=_show_if_custom("pace"))
_add_setting("exaggeration", _custom_label("exaggeration", "Exaggeration"), _custom_type("exaggeration", "number"), min=_custom_bound("exaggeration", "min", 0.25), max=_custom_bound("exaggeration", "max", 2.0), custom=True, visible=_show_if_custom("exaggeration"))


__all__ = [
    "ContainerDef",
    "SettingDef",
    "configure",
    "get_container_def",
    "get_def",
    "get_info",
    "get_summary_label",
    "iter_defs",
    "validate_inputs",
    "validate_resolved_value",
    "validate_setting_value",
]
