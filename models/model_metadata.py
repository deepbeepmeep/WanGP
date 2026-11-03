# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import copy
import fnmatch
import os
import re
from .help_shared import normalize_specialities


METADATA_KEY = "metadata"


_IMAGE_PROMPT_LABELS = {
    "": "Text/new generation",
    "T": "Text/new generation",
    "S": "Start image",
    "E": "End image",
    "V": "Continue from source video",
    "L": "Continue from last generated video",
}


def get_model_family(base_model_type, model_def, model_types_handlers, families_infos, for_ui=False):
    if for_ui:
        model_family = model_def.get("group", None)
        if model_family is not None and model_family in families_infos:
            return model_family
    handler = model_types_handlers.get(base_model_type, None)
    if handler is None:
        return "unknown"
    return handler.query_model_family()


def _model_def_finetune(model_def):
    return os.path.basename(os.path.dirname(str(model_def.get("path", "") or ""))).casefold() == "finetunes"


def _choice_values(choice_def):
    if not isinstance(choice_def, dict):
        return []
    values = []
    for choice in choice_def.get("choices", []) or []:
        if isinstance(choice, (list, tuple)):
            values.append(str(choice[1] if len(choice) > 1 else choice[0] if len(choice) > 0 else "") or "")
        else:
            values.append(str(choice or ""))
    values.extend(str(value or "") for value in choice_def.get("selection", []) or [])
    default = choice_def.get("default", None)
    if default is not None:
        values.append(str(default or ""))
    return values


def _choice_values_contain(choice_def, letters):
    return any(any(letter in value for letter in letters) for value in _choice_values(choice_def))


def _normalized_choices(choice_def):
    if not isinstance(choice_def, dict):
        return []
    labels = choice_def.get("labels", {}) or {}
    choices = []
    seen = set()

    def add_choice(label, value):
        value = "" if value is None else str(value)
        if value in seen:
            return
        seen.add(value)
        choices.append({"label": str(label or value or "None"), "value": value})

    for choice in choice_def.get("choices", []) or []:
        if isinstance(choice, (list, tuple)):
            label = choice[0] if len(choice) > 0 else ""
            value = choice[1] if len(choice) > 1 else label
            add_choice(label, value)
        else:
            add_choice(labels.get(choice, choice), choice)
    for key, label in labels.items():
        add_choice(label, key.replace("V", "").replace("P", ""))
    for value in choice_def.get("selection", []) or []:
        add_choice(labels.get(value, value), value)
    return choices


def normalize_choice_def(choice_def):
    if not isinstance(choice_def, dict):
        return None
    keys = ("default", "label", "letters_filter", "visible", "show_label", "scale", "trigger", "type")
    normalized = {key: choice_def[key] for key in keys if key in choice_def}
    normalized["choices"] = _normalized_choices(choice_def)
    return normalized


def infer_main_outputs(model_def):
    if model_def.get("world_outputs", False):  # interactive worlds, rendered as videos
        return ["world", "video"]
    if model_def.get("audio_only", False):
        return ["audio"]
    if model_def.get("mesh_outputs", False):
        return ["3d"]
    if model_def.get("image_outputs", False):
        return ["image"]
    if model_def.get("v2i_switch_supported", False) or model_def.get("inpaint_support", False):
        return ["image", "video"]
    return ["video"]


def infer_outputs(model_def):
    outputs = infer_main_outputs(model_def)
    if model_def.get("returns_audio", False) and "audio" not in outputs:
        outputs.append("audio")
    # other kinds of output a model can also give, such as "text" side files (a transcript) or an optional "video"
    outputs += [output for output in model_def.get("extra_outputs", []) if output not in outputs]
    return outputs


def infer_inputs(model_def):
    inputs = ["text"]
    image_prompt_types_allowed = str(model_def.get("image_prompt_types_allowed", "") or "")
    image_refs = model_def.get("image_ref_choices", None)
    alt_guide_refs = model_def.get("guide_custom_choices", None)
    guide_preprocessing = model_def.get("guide_preprocessing", None)
    custom_video_selection = model_def.get("custom_video_selection", None)
    image_outputs = bool(model_def.get("image_outputs", False))

    if model_def.get("any_audio_prompt", False):
        inputs.append("audio")
    if "S" in image_prompt_types_allowed or "E" in image_prompt_types_allowed or model_def.get("end_frames_always_enabled", False) or model_def.get("inpaint_support", False) or _choice_values_contain(image_refs, "IKF") or _choice_values_contain(alt_guide_refs, "IKF"):
        inputs.append("image")
    # The video guide of an image / 3D model with reference videos is a video input, not a control image.
    if image_outputs and not model_def.get("reference_video_enabled", False) and (_choice_values_contain(guide_preprocessing, "V") or _choice_values_contain(alt_guide_refs, "V")) and "image" not in inputs:
        inputs.append("image")
    if model_def.get("reference_video_enabled", False) or "V" in image_prompt_types_allowed or "L" in image_prompt_types_allowed or (not image_outputs and (_choice_values_contain(guide_preprocessing, "V") or _choice_values_contain(alt_guide_refs, "V") or _choice_values_contain(custom_video_selection, "V"))):
        inputs.append("video")
    if custom_guide_kinds(model_def):
        inputs.extend(custom_guide_kinds(model_def))
    return list(dict.fromkeys(inputs))


def custom_guide_kinds(model_def):
    """Media kinds of the custom_guide file input ("3d", "world", ...; "file" for other formats such as MIDI or scores)."""
    from shared.utils.gallery_sources import MESH_IMPORT_EXTENSIONS, WORLD_EXTENSIONS
    custom_guide = model_def.get("custom_guide")
    if custom_guide is None:
        return []
    kinds = ["3d" if file_type.lower() in MESH_IMPORT_EXTENSIONS else "world" if file_type.lower() in WORLD_EXTENSIONS else "file" for file_type in custom_guide.get("file_types", [])]
    return list(dict.fromkeys(kinds or ["file"]))


def infer_media_inputs(model_def):
    image_prompt_types_allowed = str(model_def.get("image_prompt_types_allowed", "") or "")
    image_refs = model_def.get("image_ref_choices", None)
    guide_preprocessing = model_def.get("guide_preprocessing", None)
    guide_custom_choices = model_def.get("guide_custom_choices", None)
    custom_video_selection = model_def.get("custom_video_selection", None)
    reference_choices = [value for value in _choice_values(image_refs) + _choice_values(guide_custom_choices) if "F" not in value]
    has_reference = any("I" in value for value in reference_choices)
    single_reference = bool(model_def.get("one_image_ref_needed", False) or model_def.get("one_image_ref_only", False))
    has_control = _choice_values_contain(guide_preprocessing, "V") or _choice_values_contain(guide_custom_choices, "V") or _choice_values_contain(custom_video_selection, "V")
    image_outputs = bool(model_def.get("image_outputs", False))
    image_control = image_outputs and not model_def.get("reference_video_enabled", False)
    custom_guide = model_def.get("custom_guide")
    return {
        **({"custom_file": {"name": custom_guide.get("name", "Custom Guide"), "kinds": custom_guide_kinds(model_def), "file_types": list(custom_guide.get("file_types", [])), "required": bool(custom_guide.get("required", False)),
                           **{key: custom_guide[key] for key in ("model_mode", "model_mode_not", "video_prompt_type", "video_prompt_type_not", "audio_prompt_type", "audio_prompt_type_not") if key in custom_guide}}} if custom_guide is not None else {}),
        "image": {
            "start": "S" in image_prompt_types_allowed,
            "end": "E" in image_prompt_types_allowed or bool(model_def.get("end_frames_always_enabled", False)),
            "reference": has_reference,
            "single_reference": has_reference and single_reference,
            "multiple_references": has_reference and not single_reference,
            "background": any("K" in value for value in reference_choices),
            "injected_frames": _choice_values_contain(image_refs, "F") or _choice_values_contain(guide_custom_choices, "F"),
            "control": image_control and has_control,
            "mask": image_outputs and (_choice_values_contain(model_def.get("mask_preprocessing", None), "A") or bool(model_def.get("inpaint_support", False))),
        },
        "video": {
            "reference": bool(model_def.get("reference_video_enabled", False)),
            "continue": "V" in image_prompt_types_allowed,
            "last": "L" in image_prompt_types_allowed,
            "control": (not image_outputs) and has_control,
            "mask": (not image_outputs) and _choice_values_contain(model_def.get("mask_preprocessing", None), "A"),
        },
        "audio": {
            "prompt": bool(model_def.get("any_audio_prompt", False)),
            "reference": bool(model_def.get("reference_audio_enabled", False)),
            # A control audio drives the output and is kept as its soundtrack.
            "control": bool(model_def.get("output_audio_is_input_audio", False)) or _choice_values_contain(model_def.get("audio_prompt_type_sources", None), "S"),
            # the soundtrack of a video (audio_prompt_type "K": the Control Video's first audio track) as the audio input
            "soundtrack": _choice_values_contain(model_def.get("audio_prompt_type_sources", None), "K")
                          or guide_custom_choices is not None and "K" in "".join(guide_custom_choices.get("audio_letters", {}).values()),
            "output": bool(model_def.get("audio_only", False) or model_def.get("returns_audio", False)),
        },
    }


def infer_capabilities(model_def, main_outputs, outputs, inputs, media_inputs):
    image_inputs = media_inputs["image"]
    video_inputs = media_inputs["video"]
    audio_inputs = media_inputs["audio"]
    return {
        "text_to_video": "video" in main_outputs and "text" in inputs,
        "image_to_video": "video" in main_outputs and image_inputs["start"],
        "video_to_video": "video" in main_outputs and (video_inputs["continue"] or video_inputs["control"] or video_inputs["reference"]),
        "text_to_image": "image" in main_outputs and "text" in inputs and not model_def.get("one_image_ref_needed", False),
        "image_to_image": "image" in main_outputs and (image_inputs["start"] or image_inputs["reference"] or image_inputs["control"]),
        "text_to_audio": "audio" in main_outputs and "text" in inputs,
        "audio_to_audio": "audio" in main_outputs and audio_inputs["prompt"],
        "audio_to_video": "video" in main_outputs and audio_inputs["prompt"],
        "audio_output": "audio" in outputs,
        "inpainting": bool(model_def.get("inpaint_support", False) or image_inputs["mask"] or video_inputs["mask"]),
        "outpainting": bool(model_def.get("video_guide_outpainting", False)),
        "reference_images": image_inputs["reference"],
        "reference_videos": video_inputs["reference"],
        "background_image": image_inputs["background"],
        "injected_frames": image_inputs["injected_frames"],
        "control_image": image_inputs["control"],
        "control_video": video_inputs["control"],
        "video_continuation": video_inputs["continue"],
        "sliding_window": bool(model_def.get("sliding_window", False)),
        "lora": not bool(model_def.get("no_lora", False)),
        # 3D capabilities are only listed for 3D models, which keeps the other models' metadata unchanged.
        **({"image_to_3d": image_inputs["reference"], "text_to_3d": "text" in inputs and not model_def.get("one_image_ref_needed", False), "video_to_3d": video_inputs["reference"], "3d_to_3d": "3d" in inputs} if "3d" in main_outputs else {}),
        **({"text_to_world": "text" in inputs, "image_to_world": image_inputs["start"]} if "world" in main_outputs else {}),
        # Specialized workflows can narrow capabilities inferred from shared controls.
        **model_def.get("capability_overrides", {}),
    }


def infer_setting_values(model_def):
    from shared.prompt_enhancer.labels import resolve_definition

    image_prompt_types_allowed = str(model_def.get("image_prompt_types_allowed", "") or "")
    image_prompt_choices = [{"label": _IMAGE_PROMPT_LABELS.get("", ""), "value": ""}]
    for letter in image_prompt_types_allowed:
        if letter == "T":
            continue
        image_prompt_choices.append({"label": _IMAGE_PROMPT_LABELS.get(letter, letter), "value": letter})
    return {
        "image_prompt_type": {
            "allowed": image_prompt_types_allowed,
            "choices": image_prompt_choices,
        },
        "video_prompt_type": {
            "guide_preprocessing": normalize_choice_def(model_def.get("guide_preprocessing", None)),
            "mask_preprocessing": normalize_choice_def(model_def.get("mask_preprocessing", None)),
            "guide_custom_choices": normalize_choice_def(model_def.get("guide_custom_choices", None)),
            "image_ref_choices": normalize_choice_def(model_def.get("image_ref_choices", None)),
            "custom_video_selection": normalize_choice_def(model_def.get("custom_video_selection", None)),
            "forced": str(model_def.get("set_video_prompt_type", "") or ""),
        },
        "audio_prompt_type": {
            "sources": normalize_choice_def(model_def.get("audio_prompt_type_sources", None)),
            "custom_option": model_def.get("audio_prompt_type_custom_option", None),
        },
        "model_mode": normalize_choice_def(model_def.get("model_modes", None)),
        "sample_solver": normalize_choice_def({"choices": model_def.get("sample_solvers", [])}) if model_def.get("sample_solvers", None) is not None else None,
        "prompt_enhancer": normalize_choice_def(resolve_definition(model_def)),
    }


def store_metadata(model_type, model_def, model_types_handlers, families_infos):
    from shared.resolutions import resolution_metadata
    if "specialities" in model_def:
        model_def["specialities"] = normalize_specialities(model_def["specialities"])
    base_model_type = model_def.get("architecture", None) or model_type
    family = get_model_family(base_model_type, model_def, model_types_handlers, families_infos, for_ui=True)
    family_label = families_infos.get(family, families_infos.get("unknown", (100, "Unknown")))[1]
    main_outputs = infer_main_outputs(model_def)
    outputs = infer_outputs(model_def)
    inputs = infer_inputs(model_def)
    media_inputs = infer_media_inputs(model_def)
    model_def[METADATA_KEY] = {
        "model_type": model_type,
        "family": family,
        "family_label": family_label,
        "base_model_type": base_model_type,
        "finetune": _model_def_finetune(model_def),
        "main_output": main_outputs,
        "outputs": outputs,
        "inputs": inputs,
        "media_inputs": media_inputs,
        "capabilities": infer_capabilities(model_def, main_outputs, outputs, inputs, media_inputs),
        "setting_values": infer_setting_values(model_def),
        **(resolution_metadata(model_def) if any(output in ("image", "video") for output in main_outputs) else {}),
        **{key: copy.deepcopy(model_def[key]) for key in ("accelerated", "size", "specialities") if key in model_def},
    }
    return model_def


def _normalize_filter_values(value, *, split_string=True):
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        values = [str(one).strip() for one in value if str(one).strip()]
    elif split_string and isinstance(value, str):
        values = [one.strip() for one in re.split(r"[,|/]", value) if one.strip()]
    else:
        values = [str(value).strip()] if str(value).strip() else []
    return values or None


def _normalize_bool_filter(value):
    if value is None or isinstance(value, str) and len(value.strip()) == 0:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().casefold()
    if text in ("1", "true", "yes", "y", "on"):
        return True
    if text in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"Invalid finetune filter value: {value}")


def _metadata_matches_filter(value, expected):
    expected = _normalize_filter_values(expected)
    if expected is None:
        return True
    if isinstance(value, (list, tuple, set)):
        actual = [str(one).casefold() for one in value]
        return any(fnmatch.fnmatchcase(one, pattern.casefold()) for one in actual for pattern in expected)
    actual = str(value or "").casefold()
    return any(fnmatch.fnmatchcase(actual, pattern.casefold()) for pattern in expected)


def list_model_defs(models_def, *, family=None, base_model_type=None, finetune=None, model_type=None, main_output=None, inputs=None):
    finetune_filter = _normalize_bool_filter(finetune)
    requested_outputs = _normalize_filter_values(main_output)
    requested_inputs = _normalize_filter_values(inputs)
    requested_model_types = _normalize_filter_values(model_type, split_string=False)
    records = []
    for one_model_type, model_def in models_def.items():
        metadata = model_def.get(METADATA_KEY, {})
        if not _metadata_matches_filter(one_model_type, requested_model_types):
            continue
        if not _metadata_matches_filter(metadata.get("family"), family) and not _metadata_matches_filter(metadata.get("family_label"), family):
            continue
        if not _metadata_matches_filter(metadata.get("base_model_type"), base_model_type):
            continue
        if finetune_filter is not None and bool(metadata.get("finetune", False)) != finetune_filter:
            continue
        if requested_outputs is not None and not _metadata_matches_filter(metadata.get("main_output", []), requested_outputs):
            continue
        if requested_inputs is not None and not _metadata_matches_filter(metadata.get("inputs", []), requested_inputs):
            continue
        record = copy.deepcopy(model_def)
        record["model_type"] = one_model_type
        records.append(record)
    return records
