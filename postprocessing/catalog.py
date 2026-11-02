# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Reusable discovery metadata for registered WanGP post-processors."""

from __future__ import annotations

import copy
import math
from typing import Any

from postprocessing import audio_processors, parameter_ui, spatial_upsamplers, temporal_upsamplers
from postprocessing.processor_status import PROCESSOR_STATUS_DISABLED, PROCESSOR_STATUS_ENABLED, PROCESSOR_STATUS_UNKNOWN, handler_method_available


PROCESS_TYPE_SPATIAL_UPSAMPLING = "spatial_upsampling"
PROCESS_TYPE_TEMPORAL_UPSAMPLING = "temporal_upsampling"
PROCESS_TYPE_SOUNDTRACK = "soundtrack"
PROCESS_TYPE_VOICE_REPLACEMENT = "voice_replacement"
PROCESS_TYPE_AUDIO_EDIT = "audio_edit"

_CALL_PARAMETER_FIELDS = ("name", "type", "description", "required", "default", "enum", "minimum", "maximum", "media_type")


def _method_description(handler_def: dict[str, Any], method: str, fallback: str) -> str:
    descriptions = handler_def.get("method_descriptions", {})
    if isinstance(descriptions, dict) and method in descriptions:
        return str(descriptions[method] or "").strip() or fallback
    return str(handler_def.get("description", "") or "").strip() or fallback


def _method_parameters(handler_def: dict[str, Any], method: str) -> list[dict[str, Any]]:
    parameters = handler_def.get("method_parameters", {})
    if not isinstance(parameters, dict):
        return []
    method_parameters = parameters.get(method, ())
    if not isinstance(method_parameters, (list, tuple)):
        return []
    normalized = []
    for parameter in method_parameters:
        if not isinstance(parameter, dict) or not str(parameter.get("name", "") or "").strip():
            continue
        parameter = copy.deepcopy(parameter)
        parameter.setdefault("type", "string")
        parameter.setdefault("required", True)
        parameter.setdefault("description", str(parameter["name"]).replace("_", " ").capitalize() + ".")
        normalized.append(parameter)
    return normalized


def _merge_parameters(inferred: list[dict[str, Any]], declared: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = {str(parameter["name"]): copy.deepcopy(parameter) for parameter in inferred}
    order = list(merged)
    for parameter in declared:
        name = str(parameter["name"])
        if name not in merged:
            order.append(name)
            merged[name] = {}
        merged[name].update(parameter)
    return [merged[name] for name in order]


def call_parameters(parameters) -> list[dict[str, Any]]:
    """Return only parameter metadata an assistant needs to make the call."""

    return [{key: copy.deepcopy(parameter[key]) for key in _CALL_PARAMETER_FIELDS if key in parameter} for parameter in parameters]


def call_processes(processes) -> list[dict[str, Any]]:
    """Remove handler runtime and UI metadata from assistant discovery results."""

    output = copy.deepcopy(processes)
    for process in output:
        process["parameters"] = call_parameters(process.get("parameters", ()))
    return output


def _multiplier_parameter(multipliers) -> dict[str, Any]:
    return {
        "name": "multiplier",
        "type": "number",
        "description": "Output scale multiplier.",
        "required": True,
        "enum": list(multipliers),
    }


def _process(process_id: str, label: str, description: str, process_type: str, media: tuple[str, ...], parameters: list[dict[str, Any]], category: str | None = None, *, status: str, reason_disabled: str) -> dict[str, Any]:
    process = {
        "id": str(process_id),
        "label": str(label),
        "description": str(description),
        "type": str(process_type),
        "media": list(media),
        "parameters": parameters,
        "status": status,
    }
    if category:
        process["category"] = str(category)
    if reason_disabled:
        process["reason_disabled"] = reason_disabled
    return process


def _spatial_processes(media_type: str, enabled_only: bool) -> list[dict[str, Any]]:
    processes = []
    for handler in spatial_upsamplers.upsampler_handlers(spatial_upsamplers.UPSAMPLER_TYPE_POSTPROCESSING, enabled_only):
        handler_def = handler.query_upsampler_def()
        status = spatial_upsamplers.handler_status(handler)
        reason_disabled = spatial_upsamplers.handler_reason_disabled(handler)
        media = tuple(handler_def.get("media", ("video", "image")))
        if media_type not in media:
            continue
        multipliers_by_method = handler_def.get("multipliers", {})
        for label, method in handler_def.get("methods", ()):
            if not handler_method_available(handler, method):
                continue
            display_label = handler.format_method_label(label, method) if hasattr(handler, "format_method_label") else label
            multipliers = tuple(multipliers_by_method.get(method, ()))
            inferred = [_multiplier_parameter(multipliers)] if multipliers else []
            description = _method_description(handler_def, method, f"Spatially upscale the {media_type} with {display_label}.")
            category = spatial_upsamplers.method_category(method)
            processes.append((_method_position(handler_def, method), _process(method, display_label, description, PROCESS_TYPE_SPATIAL_UPSAMPLING, media, _merge_parameters(inferred, _method_parameters(handler_def, method)), category, status=status, reason_disabled=reason_disabled)))
    return [process for _, process in sorted(processes, key=lambda item: (item[0], item[1]["label"].casefold(), item[1]["id"]))]


def _temporal_processes(media_type: str, enabled_only: bool) -> list[dict[str, Any]]:
    processes = []
    for handler in temporal_upsamplers.registered_temporal_upsamplers(enabled_only):
        handler_def = handler.query_temporal_upsampler_def()
        media = temporal_upsamplers.handler_media(handler)
        if media_type not in media:
            continue
        status = temporal_upsamplers.handler_status(handler)
        reason_disabled = temporal_upsamplers.handler_reason_disabled(handler)
        multipliers_by_method = handler_def.get("multipliers", {})
        for label, method in handler_def.get("methods", ()):
            if not handler_method_available(handler, method):
                continue
            multipliers = tuple(multipliers_by_method.get(method, ()))
            inferred = [_multiplier_parameter(multipliers)] if multipliers else []
            fallback = f"Paint the PBR textures of the 3D asset with {label}." if media_type == "3d" else f"Increase the video frame rate with {label} frame interpolation."
            description = _method_description(handler_def, method, fallback)
            category = temporal_upsamplers.method_category(method) if media_type == "3d" else None
            processes.append((_method_position(handler_def, method), _process(method, label, description, PROCESS_TYPE_TEMPORAL_UPSAMPLING, media, _merge_parameters(inferred, _method_parameters(handler_def, method)), category, status=status, reason_disabled=reason_disabled)))
    return [process for _, process in sorted(processes, key=lambda item: (item[0], item[1]["label"].casefold(), item[1]["id"]))]


def _audio_parameters(metadata: dict[str, Any], process_type: str) -> list[dict[str, Any]]:
    parameters = []
    if metadata["needs_prompt"]:
        parameters.append({"name": "prompt", "type": "string", "description": "Optional prompt guiding the generated soundtrack.", "required": False})
    if metadata["needs_negative_prompt"]:
        parameters.append({"name": "negative_prompt", "type": "string", "description": "Optional negative prompt for the generated soundtrack.", "required": False})
    if metadata["needs_audio_source"]:
        parameters.append({"name": "audio_media_id", "type": "string", "description": "Media id of the audio file to use as the soundtrack.", "required": True})
    if metadata["needs_voice_sample"]:
        parameters.append({"name": "voice_sample_media_id", "type": "string", "description": "Media id of the target voice sample.", "required": True})
    if metadata["needs_voice_sample2"]:
        parameters.append({"name": "voice_sample2_media_id", "type": "string", "description": "Media id of the second target voice sample.", "required": True})
    if process_type == PROCESS_TYPE_SOUNDTRACK and not metadata["needs_audio_source"]:
        parameters.append({"name": "seed", "type": "integer", "description": "Optional soundtrack generation seed; -1 selects a random seed.", "required": False, "default": -1})
    return parameters


def _audio_processes(media_type: str, enabled_only: bool) -> list[dict[str, Any]]:
    if media_type == "audio":
        requested_types = (audio_processors.AUDIO_PROCESSOR_TYPE_AUDIO_EDIT,)
    elif media_type == "video":
        requested_types = (audio_processors.AUDIO_PROCESSOR_TYPE_SOUNDTRACK, audio_processors.AUDIO_PROCESSOR_TYPE_VOICE_REPLACEMENT)
    else:
        return []
    process_types = {
        audio_processors.AUDIO_PROCESSOR_TYPE_SOUNDTRACK: PROCESS_TYPE_SOUNDTRACK,
        audio_processors.AUDIO_PROCESSOR_TYPE_VOICE_REPLACEMENT: PROCESS_TYPE_VOICE_REPLACEMENT,
        audio_processors.AUDIO_PROCESSOR_TYPE_AUDIO_EDIT: PROCESS_TYPE_AUDIO_EDIT,
    }
    processes = []
    seen = set()
    for processor_type in requested_types:
        for handler in audio_processors.processor_handlers(processor_type, enabled_only):
            handler_def = handler.query_audio_processor_def()
            status = audio_processors.handler_status(handler)
            reason_disabled = audio_processors.handler_reason_disabled(handler)
            for label, method in handler_def.get("methods", ()):
                if not handler_method_available(handler, method):
                    continue
                metadata = audio_processors.method_metadata(method)
                if processor_type not in metadata["types"] or method in seen:
                    continue
                seen.add(method)
                process_type = process_types[processor_type]
                description = _method_description(handler_def, method, f"Apply {label} to the {media_type}.")
                parameters = _merge_parameters(_audio_parameters(metadata, process_type), _method_parameters(handler_def, method))
                processes.append((_method_position(handler_def, method), _process(method, audio_processors.format_method_label(method, audio_processors.AUDIO_PROCESSOR_LABEL_CONTEXT_LATE_POSTPROCESSING), description, process_type, (media_type,), parameters, status=status, reason_disabled=reason_disabled)))
    return [process for _, process in sorted(processes, key=lambda item: (item[0], item[1]["label"].casefold(), item[1]["id"]))]


def _method_position(handler_def: dict[str, Any], method: str) -> float:
    positions = handler_def.get("method_pos", {})
    value = positions.get(method, handler_def.get("pos", 1000)) if isinstance(positions, dict) else handler_def.get("pos", 1000)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 1000


def query_processes(media_type: str, *, enabled_only: bool = True) -> list[dict[str, Any]]:
    """Return JSON-serializable post-processing operations for a media kind."""

    media_type = str(media_type or "").strip().lower()
    if media_type not in {"image", "video", "audio", "3d"}:
        return []
    return _spatial_processes(media_type, enabled_only) + _temporal_processes(media_type, enabled_only) + _audio_processes(media_type, enabled_only)


def find_processes(media_type: str, process_id: str, *, enabled_only: bool = True) -> list[dict[str, Any]]:
    process_id = str(process_id or "").strip()
    return [process for process in query_processes(media_type, enabled_only=enabled_only) if process["id"] == process_id]


def build_process_value(process: dict[str, Any], parameters: dict[str, Any]) -> str | None:
    process_type = str(process.get("type", "") or "")
    process_id = str(process.get("id", "") or "")
    if process_type == PROCESS_TYPE_SPATIAL_UPSAMPLING:
        return spatial_upsamplers.build_upsampling_value(process_id, parameters.get("multiplier"))
    if process_type == PROCESS_TYPE_TEMPORAL_UPSAMPLING:
        return temporal_upsamplers.build_temporal_upsampling_value(process_id, parameters.get("multiplier"))
    return process_id


def task_parameters(process: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    """Task keys of the processor parameters of a spatial or temporal process (its multiplier and seed excepted): its slot's dictionary,
    and the reference images input shared by both slots."""
    process_type = str(process.get("type", "") or "")
    if process_type not in (PROCESS_TYPE_SPATIAL_UPSAMPLING, PROCESS_TYPE_TEMPORAL_UPSAMPLING):
        return {}
    images = {str(parameter["name"]) for parameter in process.get("parameters", ()) if parameter_ui.parameter_component_type(parameter) == "images"}
    values = {name: value for name, value in parameters.items() if name not in ("multiplier", "seed", *images)}
    slot = parameter_ui.SPATIAL if process_type == PROCESS_TYPE_SPATIAL_UPSAMPLING else parameter_ui.TEMPORAL
    task = {parameter_ui.SETTINGS[slot]: values} if values else {}
    task.update({parameter_ui.IMAGES_SETTING: parameters[name] for name in images if name in parameters})
    return task


def normalize_parameters(process: dict[str, Any], parameters: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str]:
    """Validate and normalize a process parameter object against discovery metadata."""

    if parameters is None:
        parameters = {}
    if not isinstance(parameters, dict):
        return None, "parameters must be an object."
    parameter_defs = {str(parameter["name"]): parameter for parameter in process.get("parameters", ())}
    unknown = sorted(set(parameters) - set(parameter_defs))
    if unknown:
        return None, f"Unsupported parameter{'s' if len(unknown) != 1 else ''}: {', '.join(unknown)}."
    normalized = {}
    for name, parameter_def in parameter_defs.items():
        if name not in parameters:
            if parameter_def.get("required", True):
                return None, f"{name} is required."
            if "default" in parameter_def:
                normalized[name] = copy.deepcopy(parameter_def["default"])
            continue
        value, error = _normalize_parameter(name, parameters[name], parameter_def)
        if error:
            return None, error
        normalized[name] = value
    return normalized, ""


def _normalize_parameter(name: str, value: Any, parameter_def: dict[str, Any]) -> tuple[Any, str]:
    parameter_type = str(parameter_def.get("type", "string") or "string")
    try:
        if parameter_type == "number":
            if isinstance(value, bool):
                raise ValueError
            value = float(value)
            if not math.isfinite(value):
                raise ValueError
        elif parameter_type == "integer":
            numeric_value = float(value)
            if isinstance(value, bool) or not math.isfinite(numeric_value) or numeric_value != int(numeric_value):
                raise ValueError
            value = int(numeric_value)
        elif parameter_type == "boolean":
            if not isinstance(value, bool):
                raise ValueError
        elif parameter_type == "string":
            if not isinstance(value, str):
                raise ValueError
            value = value.strip()
            if parameter_def.get("required", True) and not value:
                return None, f"{name} is empty."
        elif parameter_type == "object":
            if not isinstance(value, dict):
                raise ValueError
        elif parameter_type == "array":
            if not isinstance(value, list):
                raise ValueError
        else:
            return None, f"{name} uses unsupported parameter type {parameter_type}."
    except (TypeError, ValueError, OverflowError):
        return None, f"{name} must be a {parameter_type}."
    allowed = parameter_def.get("enum", None)
    if isinstance(allowed, (list, tuple)) and value not in allowed:
        return None, f"{name} must be one of: {', '.join(str(item) for item in allowed)}."
    minimum, maximum = parameter_def.get("minimum"), parameter_def.get("maximum")
    if minimum is not None and value < minimum:
        return None, f"{name} must be at least {minimum}."
    if maximum is not None and value > maximum:
        return None, f"{name} must be at most {maximum}."
    return value, ""
