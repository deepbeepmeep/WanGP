# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import os
import time
from collections import defaultdict
from dataclasses import dataclass

import gradio as gr


MODEL_FILE_STATUS_MISSING = 0
MODEL_FILE_STATUS_PARTIAL = 1
MODEL_FILE_STATUS_EXPECTED = 2
MODEL_STATUS_PREFIXES = {
    MODEL_FILE_STATUS_MISSING: "\u2B1B",
    MODEL_FILE_STATUS_EXPECTED: "\U0001F7E6",
    MODEL_FILE_STATUS_PARTIAL: "\U0001F7E8",
}
MODEL_OUTPUT_FILTER_CONFIG_KEY = "model_output_filter"
SHOW_ALL_MODEL_FILTER_CONFIG_KEY = "show_all_model_filter"
INVERT_MODEL_SELECTION_CONFIG_KEY = "invert_model_selection"
MODEL_OUTPUT_FILTER_ALL = "all"
MODEL_OUTPUT_FILTER_VIDEO = "video"
MODEL_OUTPUT_FILTER_IMAGE = "image"
MODEL_OUTPUT_FILTER_AUDIO = "audio"
MODEL_OUTPUT_FILTER_3D = "3d"
MODEL_OUTPUT_FILTER_WORLD = "world"
MODEL_OUTPUT_FILTERS = {MODEL_OUTPUT_FILTER_ALL, MODEL_OUTPUT_FILTER_VIDEO, MODEL_OUTPUT_FILTER_IMAGE, MODEL_OUTPUT_FILTER_AUDIO, MODEL_OUTPUT_FILTER_3D, MODEL_OUTPUT_FILTER_WORLD}
LAST_MODEL_PER_OUTPUT_FILTER_CONFIG_KEY = "last_model_per_output_filter"
MODEL_SELECTOR_DEBUG = False
_model_selector_debug_seq = 0


def debug_model_selector_event(event, **values):
    if not MODEL_SELECTOR_DEBUG:
        return
    global _model_selector_debug_seq
    _model_selector_debug_seq += 1
    fields = []
    for key, value in values.items():
        text = str(value).replace("\n", "\\n")
        if len(text) > 160:
            text = text[:157] + "..."
        fields.append(f"{key}={text}")
    print(f"[model-selector-debug #{_model_selector_debug_seq:04d} {time.time():.3f}] {event}: {' '.join(fields)}", flush=True)


@dataclass
class DropdownDeps:
    transformer_types: list
    displayed_model_types: list
    transformer_type: str
    three_levels_hierarchy: bool
    families_infos: dict
    server_config: dict
    transformer_quantization: str
    transformer_dtype_policy: str
    text_encoder_quantization: str
    get_model_def: callable
    get_model_recursive_prop: callable
    get_model_filename: callable
    get_local_model_filename: callable
    get_lora_dir: callable
    get_lora_local_path: callable
    get_parent_model_type: callable
    get_base_model_type: callable
    get_model_family: callable
    get_model_name: callable
    get_transformer_dtype: callable
    list_model_defs: callable = None


def compact_name(family_name, model_name):
    suffix = model_name[len(family_name):]
    if model_name.startswith(family_name) and not suffix[:1].isalpha():  # a whole word: "Animate" leaves "AnimateAnyMesh" whole
        version = family_name.rsplit(" ", 1)[-1]
        if suffix.startswith(".") and version.isdigit():
            return version + suffix
        return suffix.strip()
    return model_name


def decorate_model_dropdown_label(label, status):
    if not isinstance(label, str):
        return label
    prefix = MODEL_STATUS_PREFIXES.get(status, "")
    return f"{prefix} {label}" if len(prefix) > 0 else label


def decorate_dropdown_choices_with_status(choices, status_map):
    decorated = []
    for choice in choices:
        if not isinstance(choice, tuple) or len(choice) < 2:
            decorated.append(choice)
            continue
        label, value = choice[0], choice[1]
        status = status_map.get(value, MODEL_FILE_STATUS_MISSING)
        decorated.append((decorate_model_dropdown_label(label, status), value, *choice[2:]))
    return decorated


def is_finetune_model(deps, model_type):
    model_def = deps.get_model_def(model_type) or {}
    return os.path.basename(os.path.dirname(str(model_def.get("path", "") or ""))).casefold() == "finetunes"


def decorate_finetune_dropdown_choices(deps, choices):
    decorated = []
    for choice in choices:
        if not isinstance(choice, tuple) or len(choice) < 2:
            decorated.append(choice)
            continue
        label, value = choice[0], choice[1]
        if is_finetune_model(deps, value) and isinstance(label, str) and not label.rstrip().endswith("*"):
            label = label.rstrip() + " *"
        decorated.append((label, value, *choice[2:]))
    return decorated


# Selectable model entries: "1:<family>" a whole family, "2:<parent_model_type>" a model with all its finetunes, a bare model type that model alone.
# Family and model entries are expanded when read, so they include finetunes added after the selection was saved.
def parse_model_selection(entry):
    level, sep, key = entry.partition(":")
    return (int(level), key) if sep and level in ("1", "2") else (3, entry)


def expand_model_selection(selection, model_types, get_parent_model_type, get_model_family, inverted=False):
    expanded = []
    for entry in selection:
        level, key = parse_model_selection(entry)
        if level == 3:
            expanded.append(key)
        else:
            # a family: its group in the selector, or the handler's family (3D models are grouped by goal since 2026-10-29)
            expanded += [model_type for model_type in model_types if (key in (get_model_family(model_type, for_ui=True), get_model_family(model_type)) if level == 1 else get_parent_model_type(model_type) == key)]
    if inverted:
        excluded = set(expanded)
        return [model_type for model_type in model_types if model_type not in excluded]
    return list(dict.fromkeys(expanded))


def get_dropdown_model_types(deps):
    inverted = bool(deps.server_config.get(INVERT_MODEL_SELECTION_CONFIG_KEY, False))
    dropdown_types = expand_model_selection(deps.transformer_types, deps.displayed_model_types, deps.get_parent_model_type, deps.get_model_family, inverted=inverted) if len(deps.transformer_types) > 0 or inverted else list(deps.displayed_model_types)
    if not inverted and deps.transformer_type not in dropdown_types:
        dropdown_types.append(deps.transformer_type)
    return list(dict.fromkeys(dropdown_types))


def normalize_model_output_filter(model_filter):
    model_filter = str(model_filter or MODEL_OUTPUT_FILTER_VIDEO).strip().casefold()
    return model_filter if model_filter in MODEL_OUTPUT_FILTERS else MODEL_OUTPUT_FILTER_VIDEO


def get_model_output_filter(deps=None, state=None):
    if isinstance(state, dict) and MODEL_OUTPUT_FILTER_CONFIG_KEY in state:
        selected_filter = normalize_model_output_filter(state[MODEL_OUTPUT_FILTER_CONFIG_KEY])
    else:
        selected_filter = normalize_model_output_filter(deps.server_config.get(MODEL_OUTPUT_FILTER_CONFIG_KEY) if deps is not None else None)
    if selected_filter == MODEL_OUTPUT_FILTER_ALL and deps is not None and not deps.server_config.get(SHOW_ALL_MODEL_FILTER_CONFIG_KEY, False):
        return MODEL_OUTPUT_FILTER_VIDEO
    return selected_filter


def store_model_output_filter(server_config, state, model_type=None):
    selected_filter = get_model_output_filter(state=state)
    server_config[MODEL_OUTPUT_FILTER_CONFIG_KEY] = selected_filter
    if model_type is None or selected_filter == MODEL_OUTPUT_FILTER_ALL:
        return
    last_model_per_output_filter = server_config.get(LAST_MODEL_PER_OUTPUT_FILTER_CONFIG_KEY, {})
    last_model_per_output_filter[selected_filter] = model_type
    server_config[LAST_MODEL_PER_OUTPUT_FILTER_CONFIG_KEY] = last_model_per_output_filter
    if isinstance(state, dict):
        state[LAST_MODEL_PER_OUTPUT_FILTER_CONFIG_KEY] = last_model_per_output_filter


def _record_main_outputs(record):
    outputs = record.get("metadata", {}).get("main_output", [])
    if isinstance(outputs, str):
        return [outputs]
    return list(outputs or [])


def _get_model_defs_by_type(deps, dropdown_types):
    # Internal callers only read metadata for exact IDs; the public bulk query
    # deep-copies entire definitions, including settings, for external callers.
    records = {}
    for model_type in dropdown_types:
        model_def = deps.get_model_def(model_type)
        if model_def is not None:
            records[model_type] = model_def
    return records


def _get_model_types_for_output_filter(deps, model_filter, dropdown_types=None):
    model_filter = normalize_model_output_filter(model_filter)
    if model_filter == MODEL_OUTPUT_FILTER_ALL:
        return None
    dropdown_types = get_dropdown_model_types(deps) if dropdown_types is None else dropdown_types
    records_by_type = _get_model_defs_by_type(deps, dropdown_types)
    family_model_types = defaultdict(list)
    for model_type in dropdown_types:
        if model_type in records_by_type:
            family_model_types[deps.get_model_family(model_type, for_ui=True)].append(model_type)
    allowed_families = set()
    for family, model_types in family_model_types.items():
        family_outputs = [_record_main_outputs(records_by_type[model_type]) for model_type in model_types]
        if model_filter == MODEL_OUTPUT_FILTER_VIDEO:  # world models render videos but have their own filter
            if any(MODEL_OUTPUT_FILTER_VIDEO in outputs and MODEL_OUTPUT_FILTER_WORLD not in outputs for outputs in family_outputs):
                allowed_families.add(family)
        elif model_filter == MODEL_OUTPUT_FILTER_WORLD:
            if any(MODEL_OUTPUT_FILTER_WORLD in outputs for outputs in family_outputs):
                allowed_families.add(family)
        elif all(outputs == [model_filter] for outputs in family_outputs):
            allowed_families.add(family)
    return {model_type for model_type in dropdown_types if deps.get_model_family(model_type, for_ui=True) in allowed_families}


def available_output_filters(deps):
    """Output filters offered in the model selector: the 3D and World filters only once such a model is available."""
    filters = [MODEL_OUTPUT_FILTER_VIDEO, MODEL_OUTPUT_FILTER_IMAGE, MODEL_OUTPUT_FILTER_AUDIO]
    if deps.server_config.get(SHOW_ALL_MODEL_FILTER_CONFIG_KEY, False):
        filters.insert(0, MODEL_OUTPUT_FILTER_ALL)
    outputs = [_record_main_outputs(record) for record in _get_model_defs_by_type(deps, get_dropdown_model_types(deps)).values()]
    if any(output == [MODEL_OUTPUT_FILTER_3D] for output in outputs):
        filters.append(MODEL_OUTPUT_FILTER_3D)
    if any(MODEL_OUTPUT_FILTER_WORLD in output for output in outputs):
        filters.append(MODEL_OUTPUT_FILTER_WORLD)
    return filters


def model_type_matches_output_filter(deps, model_type, model_filter):
    allowed_model_types = _get_model_types_for_output_filter(deps, model_filter)
    return allowed_model_types is None or model_type in allowed_model_types


def get_filtered_dropdown_model_types(deps, model_filter=None, dropdown_types=None):
    dropdown_types = get_dropdown_model_types(deps) if dropdown_types is None else list(dict.fromkeys(dropdown_types))
    allowed_model_types = _get_model_types_for_output_filter(deps, get_model_output_filter(deps) if model_filter is None else model_filter, dropdown_types)
    if allowed_model_types is None:
        return dropdown_types
    return [model_type for model_type in dropdown_types if model_type in allowed_model_types]


def _main_outputs_for_model_type(deps, model_type):
    record = _get_model_defs_by_type(deps, [model_type]).get(model_type, {})
    return _record_main_outputs(record)


def _model_output_filter_from_outputs(outputs):
    if MODEL_OUTPUT_FILTER_WORLD in outputs:
        return MODEL_OUTPUT_FILTER_WORLD
    if MODEL_OUTPUT_FILTER_VIDEO in outputs:
        return MODEL_OUTPUT_FILTER_VIDEO
    if outputs == [MODEL_OUTPUT_FILTER_AUDIO]:
        return MODEL_OUTPUT_FILTER_AUDIO
    if outputs == [MODEL_OUTPUT_FILTER_3D]:
        return MODEL_OUTPUT_FILTER_3D
    if MODEL_OUTPUT_FILTER_IMAGE in outputs:
        return MODEL_OUTPUT_FILTER_IMAGE
    if MODEL_OUTPUT_FILTER_AUDIO in outputs:
        return MODEL_OUTPUT_FILTER_AUDIO
    return MODEL_OUTPUT_FILTER_VIDEO


def get_model_output_filter_for_model_type(deps, model_type):
    return _model_output_filter_from_outputs(_main_outputs_for_model_type(deps, model_type))


def reconcile_model_output_filter_for_model_type(deps, state, model_type):
    current_filter = get_model_output_filter(deps, state)
    selected_filter = current_filter if current_filter == MODEL_OUTPUT_FILTER_ALL or model_type_matches_output_filter(deps, model_type, current_filter) else get_model_output_filter_for_model_type(deps, model_type)
    previous_filter = normalize_model_output_filter(state.get(MODEL_OUTPUT_FILTER_CONFIG_KEY, current_filter)) if isinstance(state, dict) else current_filter
    if isinstance(state, dict):
        state[MODEL_OUTPUT_FILTER_CONFIG_KEY] = selected_filter
    return selected_filter, selected_filter != previous_filter


def _first_model_type_for_filter(deps, state, model_filter, dropdown_types):
    previous_model_type = (state or {}).get(LAST_MODEL_PER_OUTPUT_FILTER_CONFIG_KEY, {}).get(model_filter, "")
    if previous_model_type in dropdown_types:
        return previous_model_type
    family_ids = sorted(
        {deps.get_model_family(model_type, for_ui=True) for model_type in dropdown_types if deps.get_model_family(model_type, for_ui=True) in deps.families_infos},
        key=lambda family: deps.families_infos[family][0],
    )
    if len(family_ids) == 0:
        return ""
    family = family_ids[0]
    family_name = deps.families_infos[family][1]
    family_model_types = sorted(
        [model_type for model_type in dropdown_types if deps.get_model_family(model_type, for_ui=True) == family],
        key=lambda model_type: compact_name(family_name, deps.get_model_name(model_type)).casefold(),
    )
    last_model_per_family = (state or {}).get("last_model_per_family", {})
    previous_model_type = last_model_per_family.get(family, "")
    return previous_model_type if previous_model_type in family_model_types else family_model_types[0]


def select_model_for_output_filter(deps, state, current_model_type, model_filter=None, dropdown_types=None):
    model_filter = get_model_output_filter(deps, state) if model_filter is None else normalize_model_output_filter(model_filter)
    dropdown_types = get_filtered_dropdown_model_types(deps, model_filter, dropdown_types)
    if current_model_type in dropdown_types:
        return current_model_type
    if len(dropdown_types) == 0:
        return current_model_type
    return _first_model_type_for_filter(deps, state, model_filter, dropdown_types)


def get_family_dropdown_model_types(deps, current_model_family, dropdown_types=None):
    dropdown_types = get_dropdown_model_types(deps) if dropdown_types is None else dropdown_types
    if current_model_family is None:
        return dropdown_types
    return [model_type for model_type in dropdown_types if deps.get_model_family(model_type, for_ui=True) == current_model_family]


def _get_module_files_for_status(deps, model_type, quantization, dtype_policy):
    transformer_dtype = deps.get_transformer_dtype(model_type, dtype_policy)
    modules = deps.get_model_recursive_prop(model_type, "modules", return_list=True)
    modules = [deps.get_model_recursive_prop(module, "modules", sub_prop_name="_list", return_list=True) if isinstance(module, str) else module for module in modules]
    module_files = []
    for module_type in modules:
        if isinstance(module_type, dict):
            URLs1 = module_type.get("URLs", None)
            if URLs1 is None:
                return None
            module_files.append(deps.get_model_filename(model_type, quantization, transformer_dtype, URLs=URLs1))
            URLs2 = module_type.get("URLs2", None)
            if URLs2 is None:
                return None
            module_files.append(deps.get_model_filename(model_type, quantization, transformer_dtype, URLs=URLs2))
        else:
            module_files.append(deps.get_model_filename(model_type, quantization, transformer_dtype, module_type=module_type))
    return module_files


def _get_status_quantization_and_dtype(deps):
    quantization = deps.server_config.get("transformer_quantization", deps.transformer_quantization)
    dtype_policy = deps.server_config.get("transformer_dtype_policy", deps.transformer_dtype_policy)
    return quantization, dtype_policy


def _append_expected_file_entry(entries, seen, filename, extra_paths=None):
    if not isinstance(filename, str) or len(filename) == 0:
        return
    if extra_paths is None:
        extra_list = []
    elif isinstance(extra_paths, list):
        extra_list = [path for path in extra_paths if isinstance(path, str) and len(path) > 0]
    else:
        extra_list = [extra_paths] if isinstance(extra_paths, str) and len(extra_paths) > 0 else []
    key = (filename.casefold(), tuple(path.casefold() for path in extra_list))
    if key in seen:
        return
    seen.add(key)
    entries.append({"filename": filename, "extra_paths": extra_list if len(extra_list) > 0 else None})


def _append_expected_local_path_entry(entries, seen, local_path):
    if not isinstance(local_path, str) or len(local_path) == 0:
        return
    path_key = local_path.casefold()
    if path_key in seen:
        return
    seen.add(path_key)
    entries.append({"path": local_path})


def get_expected_core_file_entries_for_status(deps, model_type):
    model_def = deps.get_model_def(model_type)
    if model_def is None:
        return []
    quantization, dtype_policy = _get_status_quantization_and_dtype(deps)
    entries = []
    seen = set()

    expected_filename = deps.get_model_filename(model_type, quantization=quantization, dtype_policy=dtype_policy)
    _append_expected_file_entry(entries, seen, expected_filename)
    if isinstance(model_def, dict) and "URLs2" in model_def:
        expected_filename2 = deps.get_model_filename(model_type, quantization=quantization, dtype_policy=dtype_policy, submodel_no=2)
        _append_expected_file_entry(entries, seen, expected_filename2)

    module_files = _get_module_files_for_status(deps, model_type, quantization, dtype_policy)
    if isinstance(module_files, list):
        for filename in module_files:
            _append_expected_file_entry(entries, seen, filename)

    text_encoder_URLs = deps.get_model_recursive_prop(model_type, "text_encoder_URLs", return_list=True)
    if text_encoder_URLs is not None:
        text_encoder_filename = deps.get_model_filename(model_type=model_type, quantization=deps.text_encoder_quantization, dtype_policy=dtype_policy, URLs=text_encoder_URLs)
        text_encoder_folder = model_def.get("text_encoder_folder", None)
        _append_expected_file_entry(entries, seen, text_encoder_filename, extra_paths=text_encoder_folder)
    return entries


def get_missing_core_file_entries_for_status(deps, model_type):
    missing_entries = []
    for entry in get_expected_core_file_entries_for_status(deps, model_type):
        filename = entry.get("filename", "")
        extra_paths = entry.get("extra_paths", None)
        if deps.get_local_model_filename(filename, extra_paths=extra_paths) is None:
            missing_entries.append(entry)
    return missing_entries


def get_expected_secondary_file_entries_for_status(deps, model_type):
    model_def = deps.get_model_def(model_type)
    if model_def is None:
        return []
    entries = []
    seen = set()

    preload_urls = deps.get_model_recursive_prop(model_type, "preload_URLs", return_list=True)
    if preload_urls is None:
        preload_urls = []
    if not isinstance(preload_urls, list):
        preload_urls = [preload_urls]
    for url in preload_urls:
        if isinstance(url, str) and len(url) > 0:
            _append_expected_file_entry(entries, seen, url)

    vae_urls = model_def.get("VAE_URLs", [])
    if vae_urls is None:
        vae_urls = []
    if not isinstance(vae_urls, list):
        vae_urls = [vae_urls]
    for url in vae_urls:
        if isinstance(url, str) and len(url) > 0:
            _append_expected_file_entry(entries, seen, url)

    model_loras = deps.get_model_recursive_prop(model_type, "loras", return_list=True)
    if model_loras is None:
        model_loras = []
    if not isinstance(model_loras, list):
        model_loras = [model_loras]
    lora_dir = deps.get_lora_dir(model_type)
    for url in model_loras:
        if not isinstance(url, str) or len(url) == 0:
            continue
        _append_expected_local_path_entry(entries, seen, deps.get_lora_local_path(lora_dir, url))

    return entries


def has_secondary_model_files_for_status(deps, model_type, quantization, dtype_policy):
    model_def = deps.get_model_def(model_type)
    if model_def is None:
        return True

    text_encoder_URLs = deps.get_model_recursive_prop(model_type, "text_encoder_URLs", return_list=True)
    if text_encoder_URLs is not None:
        text_encoder_filename = deps.get_model_filename(model_type=model_type, quantization=deps.text_encoder_quantization, dtype_policy=dtype_policy, URLs=text_encoder_URLs)
        if isinstance(text_encoder_filename, str) and len(text_encoder_filename) > 0:
            text_encoder_folder = model_def.get("text_encoder_folder", None)
            if deps.get_local_model_filename(text_encoder_filename, extra_paths=text_encoder_folder) is None:
                return False
    lora_dir = deps.get_lora_dir(model_type)
    for prop, recursive in (("preload_URLs", True), ("VAE_URLs", False)):
        if recursive:
            urls = deps.get_model_recursive_prop(model_type, prop, return_list=True)
        else:
            urls = model_def.get(prop, [])
        if urls is None:
            continue
        if not isinstance(urls, list):
            urls = [urls]
        for url in urls:
            if not isinstance(url, str) or len(url) == 0:
                continue
            if deps.get_local_model_filename(url, lora_dir=lora_dir) is None:
                return False

    model_loras = deps.get_model_recursive_prop(model_type, "loras", return_list=True)
    if model_loras is None:
        model_loras = []
    if not isinstance(model_loras, list):
        model_loras = [model_loras]
    lora_dir = deps.get_lora_dir(model_type)
    for url in model_loras:
        if not isinstance(url, str) or len(url) == 0:
            continue
        if not os.path.isfile(deps.get_lora_local_path(lora_dir, url)):
            return False

    module_files = _get_module_files_for_status(deps, model_type, quantization, dtype_policy)
    if module_files is None:
        return False
    for filename in module_files:
        if not isinstance(filename, str) or len(filename) == 0:
            continue
        if deps.get_local_model_filename(filename) is None:
            return False
    return True


def get_model_download_status(deps, model_type):
    quantization, dtype_policy = _get_status_quantization_and_dtype(deps)
    model_def = deps.get_model_def(model_type)
    expected_filenames = []
    expected_filename = deps.get_model_filename(model_type, quantization=quantization, dtype_policy=dtype_policy)
    if isinstance(expected_filename, str) and len(expected_filename) > 0:
        expected_filenames.append(expected_filename)
    if isinstance(model_def, dict) and "URLs2" in model_def:
        expected_filename2 = deps.get_model_filename(model_type, quantization=quantization, dtype_policy=dtype_policy, submodel_no=2)
        if isinstance(expected_filename2, str) and len(expected_filename2) > 0:
            expected_filenames.append(expected_filename2)

    expected_exists = []
    for filename in expected_filenames:
        expected_exists.append(deps.get_local_model_filename(filename) is not None)

    if len(expected_exists) > 0 and all(expected_exists):
        if not has_secondary_model_files_for_status(deps, model_type, quantization, dtype_policy):
            return MODEL_FILE_STATUS_PARTIAL
        return MODEL_FILE_STATUS_EXPECTED

    if any(expected_exists):
        return MODEL_FILE_STATUS_PARTIAL

    candidate_urls = []
    for prop in ("URLs", "URLs2"):
        urls = deps.get_model_recursive_prop(model_type, prop, return_list=True)
        if not isinstance(urls, list):
            urls = [urls] if urls else []
        candidate_urls += urls

    checked_candidates = set()
    expected_set = {name.casefold() for name in expected_filenames if isinstance(name, str) and len(name) > 0}
    for candidate in candidate_urls:
        if not isinstance(candidate, str) or len(candidate) == 0:
            continue
        candidate_key = candidate.casefold()
        if candidate_key in checked_candidates:
            continue
        checked_candidates.add(candidate_key)
        if candidate_key in expected_set:
            continue
        if deps.get_local_model_filename(candidate) is not None:
            return MODEL_FILE_STATUS_PARTIAL
    return MODEL_FILE_STATUS_MISSING


def get_model_download_status_maps(deps, dropdown_types=None):
    direct_status_map = {}
    dropdown_types = get_dropdown_model_types(deps) if dropdown_types is None else dropdown_types
    parent_to_children = defaultdict(list)

    for model_type in dropdown_types:
        if deps.get_model_def(model_type) is None:
            continue
        status = get_model_download_status(deps, model_type)
        direct_status_map[model_type] = status
        parent_model_type = deps.get_parent_model_type(model_type)
        if parent_model_type is not None:
            parent_to_children[parent_model_type].append(model_type)

    aggregated_parent_status_map = dict(direct_status_map)
    for parent_model_type, children in parent_to_children.items():
        child_statuses = [direct_status_map.get(child, MODEL_FILE_STATUS_MISSING) for child in children]
        if len(child_statuses) == 0:
            continue
        parent_status = MODEL_FILE_STATUS_MISSING
        if any(status == MODEL_FILE_STATUS_EXPECTED for status in child_statuses):
            parent_status = MODEL_FILE_STATUS_EXPECTED
        elif any(status == MODEL_FILE_STATUS_PARTIAL for status in child_statuses):
            parent_status = MODEL_FILE_STATUS_PARTIAL
        aggregated_parent_status_map[parent_model_type] = max(aggregated_parent_status_map.get(parent_model_type, MODEL_FILE_STATUS_MISSING), parent_status)
    return direct_status_map, aggregated_parent_status_map


def get_model_download_status_map(deps, dropdown_types=None):
    return get_model_download_status_maps(deps, dropdown_types)[1]


def create_models_hierarchy(rows):
    """
    rows: list of (model_name, model_id, parent_model_id)
    returns:
      parents_list: list[(parent_header, parent_id)]
      children_dict: dict[parent_id] -> list[(child_display_name, child_id)]
    """
    toks = lambda s: [t for t in s.split() if t]
    norm = lambda s: " ".join(s.split()).casefold()

    groups, parents, order = defaultdict(list), {}, []
    for name, mid, pmid in rows:
        groups[pmid].append((name, mid))
        if mid == pmid and pmid not in parents:
            parents[pmid] = name
            order.append(pmid)

    parents_list, children_dict = [], {}

    for pid in order:
        p_name = parents[pid]
        p_tok = toks(p_name)
        p_low = [w.casefold() for w in p_tok]
        n = len(p_low)
        p_last = p_low[-1]
        p_set = set(p_low)

        kids = []
        for name, mid in groups.get(pid, []):
            ot = toks(name)
            lt = [w.casefold() for w in ot]
            st = set(lt)
            kids.append((name, mid, ot, lt, st))

        outliers = {mid for _, mid, _, _, st in kids if mid != pid and p_set.isdisjoint(st)}

        prefix_non = []
        for name, mid, ot, lt, st in kids:
            if mid == pid or (mid not in outliers and lt and lt[0] == p_low[0]):
                prefix_non.append((ot, lt))

        def lcp_len(a, b):
            i = 0
            m = min(len(a), len(b))
            while i < m and a[i] == b[i]:
                i += 1
            return i

        L = n if len(prefix_non) <= 1 else min(lcp_len(lt, p_low) for _, lt in prefix_non)
        if L == 0 and len(prefix_non) > 1:
            L = n

        shares_last = any(mid != pid and mid not in outliers and lt and lt[-1] == p_last for _, mid, _, lt, _ in kids)
        header_tokens_disp = p_tok[:L] + ([p_tok[-1]] if shares_last and L < n else [])
        header = " ".join(header_tokens_disp)
        header_has_last = (L == n) or (shares_last and L < n)

        prefix_low = p_low[:L]

        def startswith_prefix(lt):
            if L == 0 or len(lt) < L:
                return False
            for i in range(L):
                if lt[i] != prefix_low[i]:
                    return False
            return True

        def base_rem(ot, lt):
            return ot[L:] if startswith_prefix(lt) else ot[:]

        def trim_rem(rem, lt):
            out = rem[:]
            if header_has_last and lt and lt[-1] == p_last and out and out[-1].casefold() == p_last:
                out = out[:-1]
            return out

        kid_infos = []
        for name, mid, ot, lt, _ in kids:
            rem_core = base_rem(ot, lt) if mid not in outliers else ot[:]
            kid_infos.append({
                "name": name,
                "mid": mid,
                "ot": ot,
                "lt": lt,
                "outlier": mid in outliers,
                "rem_core": rem_core,
                "rem_trim": trim_rem(rem_core, lt) if mid not in outliers else ot[:],
                "rem_set": {w.casefold() for w in rem_core} if mid not in outliers else set(),
                "rem_trim_set": {w.casefold() for w in (trim_rem(rem_core, lt) if mid not in outliers else ot[:])} if mid not in outliers else set(),
            })

        default_info = next(info for info in kid_infos if info["mid"] == pid)

        def disp(info):
            if info["outlier"]:
                return info["name"]
            if info["mid"] == pid:
                rem = info["rem_trim"]
            else:
                rem = info["rem_trim"]
            s = " ".join(rem).strip()
            return s if s else "Default"

        entries = [(disp(default_info), pid)]
        for info in kid_infos:
            if info["mid"] == pid:
                continue
            entries.append((disp(info), info["mid"]))

        p_full = norm(p_name)
        full_by_mid = {mid: name for name, mid, *_ in kids}
        num = 2
        numbered = [entries[0]]
        for dname, mid in entries[1:]:
            if dname == "Default" and norm(full_by_mid[mid]) == p_full:
                numbered.append((f"Default #{num}", mid))
                num += 1
            else:
                numbered.append((dname, mid))

        parents_list.append((header, pid))
        children_dict[pid] = numbered

    for pid in groups.keys():
        if pid in parents:
            continue
        first_name = groups[pid][0][0]
        parents_list.append((first_name, pid))
        children_dict[pid] = [(name, mid) for name, mid in groups[pid]]

    parents_list = sorted(parents_list, key=lambda c: c[0])
    return parents_list, children_dict


def create_models_selector_hierarchy(deps, dropdown_types=None):
    dropdown_types = get_filtered_dropdown_model_types(deps) if dropdown_types is None else list(dict.fromkeys(dropdown_types))
    family_model_types = defaultdict(list)
    for model_type in dropdown_types:
        family = deps.get_model_family(model_type, for_ui=True)
        if family in deps.families_infos:
            family_model_types[family].append(model_type)

    tree = {"folders": [], "items": []}
    sorted_families = sorted(family_model_types, key=lambda family: deps.families_infos[family][0])
    for family in sorted_families:
        family_name = deps.families_infos[family][1]
        rows = [
            (compact_name(family_name, deps.get_model_name(model_type)), model_type, deps.get_parent_model_type(model_type))
            for model_type in family_model_types[family]
        ]
        rows.sort(key=lambda row: row[0].casefold())
        parent_choices, children_by_parent = create_models_hierarchy(rows)
        family_folder = {"name": family_name, "path": family_name, "value": f"1:{family}", "folders": [], "items": []}
        for parent_name, parent_model_type in parent_choices:
            parent_path = f"{family_name}/{parent_name}"
            parent_folder = {"name": parent_name, "path": parent_path, "value": f"2:{parent_model_type}", "folders": [], "items": []}
            for child_name, child_model_type in children_by_parent.get(parent_model_type, []):
                if is_finetune_model(deps, child_model_type) and not child_name.rstrip().endswith("*"):
                    child_name = child_name.rstrip() + " *"
                parent_folder["items"].append({"name": child_name, "value": child_model_type})
            family_folder["folders"].append(parent_folder)
        tree["folders"].append(family_folder)
    return tree


def get_sorted_dropdown(deps, dropdown_types, current_model_family, current_model_type, three_levels=True):
    models_families = [deps.get_model_family(t, for_ui=True) for t in dropdown_types]
    families = {}
    for family in models_families:
        if family not in families:
            families[family] = 1

    families_orders = [deps.families_infos[family][0] for family in families]
    families_labels = [deps.families_infos[family][1] for family in families]
    sorted_familes = [info[1:] for info in sorted(zip(families_orders, families_labels, families), key=lambda c: c[0])]
    if current_model_family is None:
        dropdown_choices = [(deps.families_infos[family][0], deps.get_model_name(model_type), model_type) for model_type, family in zip(dropdown_types, models_families)]
    else:
        dropdown_choices = [(deps.families_infos[family][0], compact_name(deps.families_infos[family][1], deps.get_model_name(model_type)), model_type) for model_type, family in zip(dropdown_types, models_families) if family == current_model_family]
    dropdown_choices = sorted(dropdown_choices, key=lambda c: (c[0], c[1]))
    if three_levels:
        dropdown_choices = [(*model[1:], deps.get_parent_model_type(model[2])) for model in dropdown_choices]
        sorted_choices, finetunes_dict = create_models_hierarchy(dropdown_choices)
        return sorted_familes, sorted_choices, finetunes_dict[deps.get_parent_model_type(current_model_type)]
    dropdown_types_list = list({deps.get_base_model_type(model[2]) for model in dropdown_choices})
    dropdown_choices = [model[1:] for model in dropdown_choices]
    return sorted_familes, dropdown_types_list, dropdown_choices


def generate_dropdown_model_list(deps, current_model_type, state=None, select_visible_model=False):
    model_filter = get_model_output_filter(deps, state)
    selectable_dropdown_types = get_dropdown_model_types(deps)
    dropdown_types = get_filtered_dropdown_model_types(deps, model_filter, selectable_dropdown_types)
    if select_visible_model and deps.server_config.get(INVERT_MODEL_SELECTION_CONFIG_KEY, False):
        if not selectable_dropdown_types:
            return (
                gr.Dropdown(choices=[], value=None, show_label=False, scale=2 if deps.three_levels_hierarchy else 1, elem_id="family_list", min_width=50, interactive=True),
                gr.Dropdown(choices=[], value=None, show_label=False, scale=6, elem_id="model_base_types_list", visible=deps.three_levels_hierarchy, interactive=True),
                gr.Dropdown(choices=[], value=None, show_label=False, scale=3, visible=not deps.three_levels_hierarchy, elem_id="model_list", interactive=True),
            )
        if current_model_type not in selectable_dropdown_types:
            current_model_type = (dropdown_types or selectable_dropdown_types)[0]
    if current_model_type not in dropdown_types:
        dropdown_types.append(current_model_type)
    current_model_family = deps.get_model_family(current_model_type, for_ui=True)
    sorted_familes, sorted_models, sorted_finetunes = get_sorted_dropdown(deps, dropdown_types, current_model_family, current_model_type, three_levels=deps.three_levels_hierarchy)
    status_model_types = get_family_dropdown_model_types(deps, current_model_family, dropdown_types)
    if current_model_type not in status_model_types:
        status_model_types.append(current_model_type)
    direct_status_map, aggregated_parent_status_map = get_model_download_status_maps(deps, status_model_types)
    sorted_models = decorate_dropdown_choices_with_status(sorted_models, aggregated_parent_status_map)
    if deps.three_levels_hierarchy:
        sorted_finetunes = decorate_finetune_dropdown_choices(deps, sorted_finetunes)
    sorted_finetunes = decorate_dropdown_choices_with_status(sorted_finetunes, direct_status_map)

    dropdown_families = gr.Dropdown(choices=sorted_familes, value=current_model_family, show_label=False, scale=2 if deps.three_levels_hierarchy else 1, elem_id="family_list", min_width=50, interactive=True)
    dropdown_models = gr.Dropdown(choices=sorted_models, value=deps.get_parent_model_type(current_model_type) if deps.three_levels_hierarchy else deps.get_base_model_type(current_model_type), show_label=False, scale=3 if len(sorted_finetunes) > 1 else 6, elem_id="model_base_types_list", visible=deps.three_levels_hierarchy, interactive=True)
    dropdown_finetunes = gr.Dropdown(choices=sorted_finetunes, value=current_model_type, show_label=False, scale=3, visible=len(sorted_finetunes) > 1 or not deps.three_levels_hierarchy, elem_id="model_list", interactive=True)
    return dropdown_families, dropdown_models, dropdown_finetunes


def change_model_family(deps, state, current_model_family):
    dropdown_types = get_filtered_dropdown_model_types(deps, get_model_output_filter(deps, state))
    current_family_name = deps.families_infos[current_model_family][1]
    models_families = [deps.get_model_family(t, for_ui=True) for t in dropdown_types]
    dropdown_choices = [(compact_name(current_family_name, deps.get_model_name(model_type)), model_type) for model_type, family in zip(dropdown_types, models_families) if family == current_model_family]
    dropdown_choices = sorted(dropdown_choices, key=lambda c: c[0])
    family_dropdown_types = [choice[1] for choice in dropdown_choices]
    direct_status_map, aggregated_parent_status_map = get_model_download_status_maps(deps, family_dropdown_types)
    last_model_per_family = state.get("last_model_per_family", {})
    model_type = last_model_per_family.get(current_model_family, "")
    if len(model_type) == "" or model_type not in [choice[1] for choice in dropdown_choices]:
        model_type = dropdown_choices[0][1]

    if deps.three_levels_hierarchy:
        parent_model_type = deps.get_parent_model_type(model_type)
        dropdown_choices = [(*tup, deps.get_parent_model_type(tup[1])) for tup in dropdown_choices]
        dropdown_base_types_choices, finetunes_dict = create_models_hierarchy(dropdown_choices)
        finetunes_dict[parent_model_type] = decorate_finetune_dropdown_choices(deps, finetunes_dict[parent_model_type])
        dropdown_choices = decorate_dropdown_choices_with_status(finetunes_dict[parent_model_type], direct_status_map)
        dropdown_base_types_choices = decorate_dropdown_choices_with_status(dropdown_base_types_choices, aggregated_parent_status_map)
        model_finetunes_visible = len(dropdown_choices) > 1
    else:
        parent_model_type = deps.get_base_model_type(model_type)
        model_finetunes_visible = True
        dropdown_base_types_choices = list({deps.get_base_model_type(model[1]) for model in dropdown_choices})
        dropdown_choices = decorate_dropdown_choices_with_status(dropdown_choices, direct_status_map)

    return gr.Dropdown(choices=dropdown_base_types_choices, value=parent_model_type, scale=3 if model_finetunes_visible else 6), gr.Dropdown(choices=dropdown_choices, value=model_type, visible=model_finetunes_visible)


def change_model_base_types(deps, state, current_model_family, model_base_type_choice):
    if not deps.three_levels_hierarchy:
        return gr.update()
    dropdown_types = get_filtered_dropdown_model_types(deps, get_model_output_filter(deps, state))
    current_family_name = deps.families_infos[current_model_family][1]
    dropdown_choices = [(compact_name(current_family_name, deps.get_model_name(model_type)), model_type, model_base_type_choice) for model_type in dropdown_types if deps.get_parent_model_type(model_type) == model_base_type_choice and deps.get_model_family(model_type, for_ui=True) == current_model_family]
    dropdown_choices = sorted(dropdown_choices, key=lambda c: c[0])
    _, finetunes_dict = create_models_hierarchy(dropdown_choices)
    base_dropdown_types = [choice[1] for choice in dropdown_choices]
    direct_status_map, _ = get_model_download_status_maps(deps, base_dropdown_types)
    finetunes_dict[model_base_type_choice] = decorate_finetune_dropdown_choices(deps, finetunes_dict[model_base_type_choice])
    dropdown_choices = decorate_dropdown_choices_with_status(finetunes_dict[model_base_type_choice], direct_status_map)
    model_finetunes_visible = len(dropdown_choices) > 1
    last_model_per_type = state.get("last_model_per_type", {})
    model_type = last_model_per_type.get(model_base_type_choice, "")
    if len(model_type) == "" or model_type not in [choice[1] for choice in dropdown_choices]:
        model_type = dropdown_choices[0][1]
    return gr.update(scale=3 if model_finetunes_visible else 6), gr.Dropdown(choices=dropdown_choices, value=model_type, visible=model_finetunes_visible)
