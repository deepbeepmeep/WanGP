# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import json
import math
import os
import re

from shared.match_archi import match_nvidia_architecture


RESOLUTION_FILE = "resolutions.json"
# Include the block-aligned 1080p presets (1920x1088).
DEFAULT_RESOLUTION_BUDGET = "1920x1088"

DEFAULT_RESOLUTION_CHOICES_4K = [
    # 16.8 MP
    ("4096x4096 (1:1)", "4096x4096"),
    ("5440x3072 (16:9)", "5440x3072"),
    ("3072x5440 (9:16)", "3072x5440"),
    # 4K
    ("3840x2176 (16:9)", "3840x2176"),
    ("2176x3840 (9:16)", "2176x3840"),
    ("2880x2880 (1:1)", "2880x2880"),
    ("3840x1664 (21:9)", "3840x1664"),
    ("1664x3840 (9:21)", "1664x3840"),
    ("2048x2048 (1:1)", "2048x2048"),
    # 1440p
    ("1920x1920 (1:1)", "1920x1920"),
    ("2560x1440 (16:9)", "2560x1440"),
    ("1440x2560 (9:16)", "1440x2560"),
    ("1920x1440 (4:3)", "1920x1440"),
    ("1440x1920 (3:4)", "1440x1920"),
    ("2160x1440 (3:2)", "2160x1440"),
    ("1440x2160 (2:3)", "1440x2160"),
    ("1456x1456 (1:1)", "1456x1456"),
    ("2688x1152 (21:9)", "2688x1152"),
    ("1152x2688 (9:21)", "1152x2688"),
]

DEFAULT_RESOLUTION_CHOICES = [
    # 1080p
    ("1920x1088 (16:9)", "1920x1088"),
    ("1088x1920 (9:16)", "1088x1920"),
    ("1440x1440 (1:1)", "1440x1440"),
    ("1536x1024 (3:2)", "1536x1024"),
    ("1024x1536 (2:3)", "1024x1536"),
    ("1920x832 (21:9)", "1920x832"),
    ("832x1920 (9:21)", "832x1920"),
    ("2048x768 (8:3)", "2048x768"),
    ("1024x1792 (4:7)", "1024x1792"),
    ("1088x1088 (1:1)", "1088x1088"),
    # 720p
    ("1024x1024 (1:1)", "1024x1024"),
    ("1280x720 (16:9)", "1280x720"),
    ("720x1280 (9:16)", "720x1280"),
    ("1600x400 (4:1)", "1600x400"),
    ("1280x544 (21:9)", "1280x544"),
    ("544x1280 (9:21)", "544x1280"),
    ("1104x832 (4:3)", "1104x832"),
    ("832x1104 (3:4)", "832x1104"),
    ("960x960 (1:1)", "960x960"),
    # 540p
    ("960x544 (16:9)", "960x544"),
    ("544x960 (9:16)", "544x960"),
    # 480p
    ("832x624 (4:3)", "832x624"),
    ("624x832 (3:4)", "624x832"),
    ("720x720 (1:1)", "720x720"),
    ("832x480 (16:9)", "832x480"),
    ("480x832 (9:16)", "480x832"),
    # 384p
    ("672x384 (16:9)", "672x384"),
    ("384x672 (9:16)", "384x672"),
    ("512x512 (1:1)", "512x512"),
    # 320p
    ("576x320 (16:9)", "576x320"),
    ("320x576 (9:16)", "320x576"),
    ("448x448 (1:1)", "448x448"),
    # 256p
    ("448x256 (7:4)", "448x256"),
    ("256x448 (4:7)", "256x448"),
    ("320x320 (1:1)", "320x320"),
]

# Frozen for legacy category expressions, whose aliases and equality ranges must
# not change when the display groups below acquire a new boundary.
LEGACY_GROUP_THRESHOLDS = {
    "256p": 448 * 256,
    "320p": 448 * 448,
    "384p": 512 * 512,
    "480p": 832 * 624,
    "540p": 960 * 544,
    "720p": 1024 * 1024,
    "1080p": 1920 * 1088,
    "1440p": 2560 * 1440,
    "2160p": 3840 * 2176,
    "4096p": 4096 * 4096,
}

GROUP_THRESHOLDS = {**LEGACY_GROUP_THRESHOLDS}
GROUP_THRESHOLDS.pop("4096p")
GROUP_THRESHOLDS.pop("1440p")
GROUP_THRESHOLDS["2048sq"] = 2048 * 2048
GROUP_THRESHOLDS["4096sq"] = 4096 * 4096
GROUP_THRESHOLDS = dict(sorted(GROUP_THRESHOLDS.items(), key=lambda item: item[1]))
GROUP_THRESHOLDS["larger"] = math.inf
GROUP_LABELS = {"2048sq": "4.2 MP", "2160p": "4K UHD", "4096sq": "16.8 MP", "larger": "Custom"}

GROUP_TIERS = {
    "256p": 256,
    "320p": 320,
    "384p": 384,
    "480p": 480,
    "540p": 540,
    "720p": 720,
    "1080p": 1080,
    "1440p": 1440,
    "2160p": 2160,
    "4096p": 4096,
    "2048sq": 2048,
    "4096sq": 4096,
}

CATEGORY_ALIASES = {
    "2k": 1440,
    "4k": 2160,
}

_custom_resolutions = None


def is_resolution_value(value):
    return isinstance(value, str) and re.fullmatch(r"[1-9]\d*x[1-9]\d*", value.strip().lower()) is not None


def parse_resolution(value):
    if not is_resolution_value(value):
        raise ValueError('resolution must be a positive "WIDTHxHEIGHT" string.')
    width, height = value.strip().lower().split("x", 1)
    return int(width), int(height)


def normalize_resolution_choices(resolution_choices, source_name, printer=print):
    if resolution_choices is None:
        return None
    if not isinstance(resolution_choices, list):
        printer(f'"{source_name}" should be a list of 2 elements lists ["Label","WxH"]')
        return None

    normalized = []
    for tup in resolution_choices:
        if not isinstance(tup, (list, tuple)) or len(tup) != 2 or not isinstance(tup[0], str) or not isinstance(tup[1], str):
            printer(f'"{source_name}" contains an invalid list of two elements: {tup}')
            return None
        if not is_resolution_value(tup[1]):
            printer(f'"{source_name}" contains a resolution value that is not in the format "WxH": {tup[1]}')
            return None
        normalized.append((tup[0], tup[1].lower()))
    return normalized


def load_custom_resolution_choices(resolution_file=RESOLUTION_FILE, printer=print):
    global _custom_resolutions
    if _custom_resolutions is not None:
        return _custom_resolutions
    if not os.path.isfile(resolution_file):
        return []

    try:
        with open(resolution_file, "r", encoding="utf-8") as f:
            resolution_choices = json.load(f)
    except Exception as e:
        printer(f'Invalid "{resolution_file}" : {e}')
        resolution_choices = None
    normalized = normalize_resolution_choices(resolution_choices, resolution_file, printer)
    if normalized is None:
        return []
    _custom_resolutions = normalized
    return _custom_resolutions


def reset_custom_resolution_cache():
    global _custom_resolutions
    _custom_resolutions = None


def dedupe_resolution_choices(resolution_choices):
    seen = set()
    deduped = []
    for label, resolution in resolution_choices:
        if resolution in seen:
            continue
        seen.add(resolution)
        deduped.append((label, resolution))
    return deduped


def normalize_block_size(block_size):
    block_size = int(block_size)
    if block_size < 1:
        raise ValueError(f"Invalid resolution block size: {block_size}")
    return block_size


def align_dimension_to_block(value, block_size):
    if block_size <= 1:
        return value
    return max(block_size, value // block_size * block_size)


def align_resolution_value(resolution, block_size):
    block_size = normalize_block_size(block_size)
    width, height = parse_resolution(resolution)
    return f"{align_dimension_to_block(width, block_size)}x{align_dimension_to_block(height, block_size)}"


def align_resolution_label(label, resolution, aligned_resolution):
    if resolution == aligned_resolution:
        return label
    exact_pattern = re.compile(re.escape(resolution), re.IGNORECASE)
    if exact_pattern.search(label):
        return exact_pattern.sub(aligned_resolution, label, count=1)
    return re.sub(r"\d+x\d+", aligned_resolution, label, count=1, flags=re.IGNORECASE)


def align_resolution_choices(resolution_choices, block_size):
    block_size = normalize_block_size(block_size)
    if block_size <= 1:
        return resolution_choices
    return dedupe_resolution_choices(
        (align_resolution_label(label, resolution, aligned_resolution), aligned_resolution)
        for label, resolution in resolution_choices
        for aligned_resolution in [align_resolution_value(resolution, block_size)]
    )


def builtin_resolution_choices(include_4k=False):
    return list(DEFAULT_RESOLUTION_CHOICES_4K if include_4k else []) + list(DEFAULT_RESOLUTION_CHOICES)


def all_global_resolution_choices():
    return dedupe_resolution_choices(builtin_resolution_choices(include_4k=True) + load_custom_resolution_choices())


def default_global_resolution_choices(enable_4k_resolutions=False):
    return dedupe_resolution_choices(builtin_resolution_choices(include_4k=enable_4k_resolutions) + load_custom_resolution_choices())


def keep_resolution_on_model_switch_enabled(value):
    if isinstance(value, str):
        return value.strip().casefold() not in {"0", "false", "no", "off"}
    return value not in (False, 0)


def categorize_resolution(resolution_str):
    width, height = parse_resolution(resolution_str)
    pixel_count = width * height

    for group, threshold in GROUP_THRESHOLDS.items():
        if pixel_count <= threshold:
            return group
    return next(reversed(GROUP_THRESHOLDS))


def _category_tier(category):
    key = str(category).strip().lower()
    if key in GROUP_TIERS:
        return GROUP_TIERS[key]
    if key in CATEGORY_ALIASES:
        return CATEGORY_ALIASES[key]
    if key.endswith("p"):
        key = key[:-1]
    if not key.isdigit():
        return None
    tier = int(key)
    return tier if tier in GROUP_TIERS.values() else None


def _normalize_category_expression(expression):
    def replace_token(match):
        tier = _category_tier(match.group(0))
        return str(tier) if tier is not None else match.group(0)

    return re.sub(r"[a-z0-9]+", replace_token, str(expression).strip().lower())


def normalize_category_expressions(category_expressions):
    if category_expressions is None:
        return []
    if isinstance(category_expressions, str):
        return [category_expressions]
    if isinstance(category_expressions, (list, tuple)):
        return [str(expression) for expression in category_expressions]
    return []


def category_allowed(category, category_expressions):
    expressions = normalize_category_expressions(category_expressions)
    if not expressions:
        return True
    if category == "larger":
        return False
    tier = GROUP_TIERS[category]
    return any(match_nvidia_architecture({_normalize_category_expression(expression): True}, tier) for expression in expressions)


def filter_resolution_choices_by_categories(resolution_choices, category_expressions):
    expressions = normalize_category_expressions(category_expressions)
    if not expressions:
        return list(resolution_choices)
    result = []
    for choice in resolution_choices:
        width, height = parse_resolution(choice[1])
        for expression in expressions:
            thresholds = GROUP_THRESHOLDS if re.search(r"\b(?:2048|4096)sq\b", expression, re.I) else LEGACY_GROUP_THRESHOLDS
            group = next((group for group, area in thresholds.items() if width * height <= area), "larger")
            if category_allowed(group, [expression]):
                result.append(choice)
                break
    return result


def closest_resolution(target_resolution, resolution_choices):
    if not resolution_choices:
        return target_resolution
    if not is_resolution_value(target_resolution or ""):
        return resolution_choices[0][1]

    target_width, target_height = parse_resolution(target_resolution)
    target_ratio = target_width / target_height
    target_pixels = target_width * target_height
    target_group = categorize_resolution(target_resolution)
    group_order = list(GROUP_THRESHOLDS)
    target_group_index = group_order.index(target_group)
    grouped_choices = {}
    for choice in resolution_choices:
        grouped_choices.setdefault(categorize_resolution(choice[1]), []).append(choice)
    closest_group = min(grouped_choices, key=lambda group: (abs(group_order.index(group) - target_group_index), abs(GROUP_THRESHOLDS[group] - GROUP_THRESHOLDS[target_group])))

    def score(choice):
        width, height = parse_resolution(choice[1])
        ratio_score = abs(math.log((width / height) / target_ratio))
        pixel_score = abs((width * height) - target_pixels) / target_pixels
        return ratio_score, pixel_score

    return min(grouped_choices[closest_group], key=score)[1]


def resolve_resolution_choices(current_resolution_choice, model_def, enable_4k_resolutions=False, block_size=None):
    model_def = migrate_resolution_definition(model_def)
    custom = normalize_resolution_choices(model_def.get("resolutions"), "model.resolutions") or []
    locked = model_def.get("lock_resolutions", False)
    if locked:
        resolution_choices = custom
    elif enable_4k_resolutions:
        resolution_choices = dedupe_resolution_choices(default_global_resolution_choices(True) + custom)
    elif available_resolution_budgets(model_def):
        max_pixels = max(available_resolution_budgets(model_def).values(), default=0)
        resolution_choices = [choice for choice in builtin_resolution_choices(include_4k=True) if math.prod(parse_resolution(choice[1])) <= max_pixels]
        resolution_choices = dedupe_resolution_choices(resolution_choices + custom + load_custom_resolution_choices())
    else:
        resolution_choices = dedupe_resolution_choices(default_global_resolution_choices(enable_4k_resolutions) + custom)
    resolution_choices = align_resolution_choices(resolution_choices, model_def.get("vae_block_size", 16) if block_size is None else block_size)
    minimum = (resolution_pixels(model_def.get("resolution_min")) or 0) if locked or not enable_4k_resolutions else 0
    resolution_choices = [choice for choice in resolution_choices if math.prod(parse_resolution(choice[1])) >= minimum and not resolution_dimension_error(model_def, choice[1])]

    if is_resolution_value(current_resolution_choice):
        exceeds_limits = bool(resolution_dimension_error(model_def, current_resolution_choice))
        if not locked and not exceeds_limits:
            current_resolution_choice = align_resolution_value(current_resolution_choice, model_def.get("vae_block_size", 16) if block_size is None else block_size)
        if not any(current_resolution_choice == resolution[1] for resolution in resolution_choices):
            budgets = available_resolution_budgets(model_def)
            if locked or exceeds_limits:
                current_resolution_choice = None
            elif model_def.get("resolution_budget") is not None and not enable_4k_resolutions and budgets and resolution_choices and math.prod(parse_resolution(current_resolution_choice)) > max(budgets.values()):
                current_resolution_choice = closest_resolution(current_resolution_choice, resolution_choices)  # a remembered or saved size above the budget the model declares is not offered
            else:
                resolution_choices.append((f"{current_resolution_choice} (Current)", current_resolution_choice))
    elif resolution_choices:
        current_resolution_choice = resolution_choices[0][1]
    else:
        current_resolution_choice = None

    return resolution_choices, current_resolution_choice


def resolve_model_switch_resolution(source_resolution, target_model_def, enable_4k_resolutions=False, block_size=None):
    if not is_resolution_value(source_resolution or ""):
        return None
    return resolve_resolution_choices(source_resolution, target_model_def, enable_4k_resolutions, block_size)[1]


def group_resolution_choices(resolution_choices, selected_resolution):
    grouped_resolutions = {}
    for resolution in resolution_choices:
        group = categorize_resolution(resolution[1])
        grouped_resolutions.setdefault(group, []).append(resolution)

    if not grouped_resolutions:
        return [], [], None

    available_groups = [group for group in GROUP_THRESHOLDS if group in grouped_resolutions]
    available_groups.reverse()

    if selected_resolution is None and available_groups:
        selected_group = available_groups[0]
    else:
        selected_group = categorize_resolution(selected_resolution)
        if selected_group not in grouped_resolutions and available_groups:
            selected_group = available_groups[0]

    return available_groups, grouped_resolutions.get(selected_group, []), selected_group


def group_choices(resolution_choices, selected_group):
    return [resolution for resolution in resolution_choices if categorize_resolution(resolution[1]) == selected_group]


def remember_last_resolution(last_resolution_per_group, resolution):
    last_resolution_per_group[categorize_resolution(resolution)] = resolution
    return last_resolution_per_group


def migrate_resolution_preferences(config):
    """Rekey remembered groups by dimensions, never rewrite saved generation sizes."""
    remembered = {}
    for resolution in config.get("last_resolution_per_group", {}).values():
        if is_resolution_value(resolution):
            remember_last_resolution(remembered, resolution)
    if is_resolution_value(config.get("last_resolution_choice")):
        remember_last_resolution(remembered, config["last_resolution_choice"])
    updates = {}
    if remembered != config.get("last_resolution_per_group", {}):
        updates["last_resolution_per_group"] = remembered
    if config.get("resolution_picker_mode") not in ("presets", "mp"):
        updates["resolution_picker_mode"] = "presets"
    return updates


def resolution_pixels(value):
    """Parse the public area shorthand; WxH does not constrain aspect ratio."""
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    if is_resolution_value(value):
        return math.prod(parse_resolution(value))
    raise ValueError('Resolution bounds must be positive integer pixels or "WIDTHxHEIGHT".')


def migrate_resolution_definition(model_def, legacy_lists=False):
    """Expand old menu rules without declaring their ceiling as model capability."""
    result = dict(model_def)
    rules = result.pop("resolutions_categories", None)
    if rules is not None:
        custom = normalize_resolution_choices(result.get("resolutions"), "model.resolutions") or []
        result["resolutions"] = dedupe_resolution_choices(custom + filter_resolution_choices_by_categories(builtin_resolution_choices(include_4k=True), rules))
        result.setdefault("lock_resolutions", False)
    elif legacy_lists and result.get("resolutions") and "lock_resolutions" not in result:
        result["lock_resolutions"] = True
    budget = result.get("resolution_budget")
    if isinstance(budget, dict):
        ordinary = budget.get("two_phases", budget.get("default"))
        result.pop("resolution_budget")
        if ordinary is not None:
            result["resolution_budget"] = ordinary
        if "two_phases_tiled" in budget:
            result.setdefault("resolution_budget_tiled", budget["two_phases_tiled"])
    return result


def fix_resolution_settings(settings):
    if settings.get("settings_version", 0) < 2.80 and "model" in settings:
        settings["model"] = migrate_resolution_definition(settings["model"], legacy_lists=True)



def locked_resolution_values(model_def):
    choices = normalize_resolution_choices(model_def.get("resolutions"), "model.resolutions") or []
    minimum = resolution_pixels(model_def.get("resolution_min")) or 0
    return [value for _, value in align_resolution_choices(choices, model_def.get("vae_block_size", 16)) if math.prod(parse_resolution(value)) >= minimum and not resolution_dimension_error(model_def, value)]


def resolution_dimension_error(model_def, resolution):
    dimensions = zip(("width", "height"), parse_resolution(resolution))
    exceeded = [f"maximum {axis} of {model_def['max_' + axis]} pixels" for axis, value in dimensions if model_def.get("max_" + axis) is not None and value > model_def["max_" + axis]]
    return f"Resolution {resolution} exceeds this model's {' and '.join(exceeded)}." if exceeded else ""


def validate_resolution_definition(model_def):
    for key in ("max_width", "max_height"):
        value = model_def.get(key)
        if value is not None and (type(value) is not int or value < normalize_block_size(model_def.get("vae_block_size", 16))):
            raise ValueError(f"{key} must be an integer at least as large as vae_block_size, or null for no limit.")
    for key in ("resolution_budget", "resolution_budget_tiled", "resolution_min"):
        resolution_pixels(model_def.get(key))
    minimum = resolution_pixels(model_def.get("resolution_min")) or 0
    for key in ("resolution_budget", "resolution_budget_tiled"):
        maximum = resolution_pixels(model_def.get(key) or (DEFAULT_RESOLUTION_BUDGET if key == "resolution_budget" else None))
        if maximum is not None and minimum > maximum:
            raise ValueError(f"resolution_min exceeds {key}.")
    if model_def.get("lock_resolutions") and not locked_resolution_values(model_def):
        raise ValueError("Lock Resolutions requires at least one custom resolution within the dimension limits and at or above the minimum.")


def validate_resolution(model_def, resolution):
    error = resolution_dimension_error(model_def, resolution)
    if error:
        raise ValueError(error)
    if model_def.get("lock_resolutions"):
        allowed = locked_resolution_values(model_def)
        if resolution.lower() not in allowed:
            raise ValueError("Resolution is locked for this model. Choose one of: " + ", ".join(allowed))


def validate_agent_resolution(model_def, resolution, guidance_phases=1, video_prompt_type=""):
    """Agent requests must fit the selected workflow; UI overrides are separate."""
    if model_def.get("audio_only") or model_def.get("mesh_outputs"):
        return
    validate_resolution(model_def, resolution)
    pixels = math.prod(parse_resolution(resolution))
    minimum = resolution_pixels(model_def.get("resolution_min")) or 0
    mode = resolution_mode(model_def, guidance_phases, video_prompt_type)
    maximum = available_resolution_budgets(model_def).get(mode, resolution_budget(model_def)["default"])
    if pixels < minimum or pixels > maximum:
        assessment = assess_resolution(model_def, resolution, guidance_phases, video_prompt_type)
        hint = f" Use settings {json.dumps(assessment['settings'])} for this size." if assessment.get("settings") else " Choose another resolution or model."
        raise ValueError(f"Resolution {resolution} is outside the accepted range for {RESOLUTION_MODE_LABELS[mode]}: {minimum:,} to {maximum:,} pixels ({minimum / 1_000_000:.2f}–{maximum / 1_000_000:.2f} MP)." + hint)


def resolution_budget(model_def):
    """Normalized recommendations; unlocked custom presets do not imply a ceiling."""
    result = {"default": resolution_pixels(model_def.get("resolution_budget")) or resolution_pixels(DEFAULT_RESOLUTION_BUDGET)}
    for key, mode in (("resolution_budget", "default"), ("resolution_budget_tiled", "two_phases_tiled")):
        value = resolution_pixels(model_def.get(key))
        if value is not None:
            result[mode] = value
    return result


def available_resolution_budgets(model_def):
    budget = resolution_budget(model_def)
    result = {}
    max_phases = model_def.get("guidance_max_phases", 0)
    locked = model_def.get("lock_guidance_phases", False)
    if "default" in budget and not (locked and max_phases == 2):
        result["default"] = budget["default"]
    if max_phases >= 2 and (not locked or max_phases == 2):
        two = budget.get("default")
        if two is not None:
            result["two_phases"] = two
        if model_def.get("phase_2_spatial_tiling", False):
            tiled = budget.get("two_phases_tiled", two)
            if tiled is not None:
                result["two_phases_tiled"] = tiled
    return result


def resolution_mode(model_def, guidance_phases=1, video_prompt_type=""):
    if model_def.get("lock_guidance_phases", False):
        guidance_phases = "2~" if guidance_phases == "2~" and model_def.get("guidance_max_phases") == 2 else model_def.get("guidance_max_phases", 0)
    if model_def.get("guidance_max_phases", 0) >= 2 and guidance_phases in (2, "2", "2~"):
        if model_def.get("phase_2_spatial_tiling", False) and (guidance_phases == "2~" or "~" in video_prompt_type):
            return "two_phases_tiled"
        return "two_phases"
    return "default"


def normalize_phase_2_tiling_selection(model_def, guidance_phases, video_prompt_type):
    if guidance_phases == "2~":
        guidance_phases = 2
        if model_def.get("phase_2_spatial_tiling", False):
            if "~" not in video_prompt_type:
                video_prompt_type += "~"
        else:
            video_prompt_type = video_prompt_type.replace("~", "")
    elif guidance_phases != 2 or not model_def.get("phase_2_spatial_tiling", False):
        video_prompt_type = video_prompt_type.replace("~", "")
    return guidance_phases, video_prompt_type


def assess_resolution(model_def, resolution, guidance_phases=1, video_prompt_type=""):
    """Assess an exact canvas without replacing it with a preset or MP target."""
    width, height = parse_resolution(resolution)
    requested = f"{width}x{height}"
    budgets = available_resolution_budgets(model_def)
    current = resolution_mode(model_def, guidance_phases, video_prompt_type)
    minimum = resolution_pixels(model_def.get("resolution_min")) or 0
    locked = model_def.get("lock_resolutions", False)
    allowed = locked_resolution_values(model_def) if locked else []
    matching = [mode for mode, budget in budgets.items() if minimum <= width * height <= budget]
    selected = current if current in matching else next(iter(matching), None)
    result = {"requested": requested, "aligned": align_resolution_value(requested, model_def.get("vae_block_size", 16)), "status": "recommended" if matching else "above_recommendation" if budgets else "unknown"}
    dimension_error = resolution_dimension_error(model_def, requested)
    if dimension_error:
        return {**result, "status": "not_allowed", "reason": dimension_error}
    if locked and requested not in allowed:
        return {**result, "status": "not_allowed", "allowed_resolutions": allowed}
    if width * height < minimum:
        return {**result, "status": "below_recommendation"}
    if locked and not budgets:
        result["status"] = "recommended"
    if selected is not None:
        result["mode"] = selected
        if selected != current:
            result["settings"] = {"guidance_phases": {"default": 1, "two_phases": 2, "two_phases_tiled": "2~"}[selected]}
    return result


def resolution_metadata(model_def):
    budgets = available_resolution_budgets(model_def)
    result = {"resolution_budget": resolution_budget(model_def)["default"], "vae_block_size": model_def.get("vae_block_size", 16)}
    result.update({key: model_def[key] for key in ("max_width", "max_height") if model_def.get(key) is not None})
    if "two_phases_tiled" in budgets:
        result["resolution_budget_tiled"] = budgets["two_phases_tiled"]
    if model_def.get("resolution_min") is not None:
        result["resolution_min"] = resolution_pixels(model_def["resolution_min"])
    if model_def.get("lock_resolutions"):
        result.update(lock_resolutions=True, allowed_resolutions=locked_resolution_values(model_def))
    return result


RESOLUTION_MODE_LABELS = {"default": "Default Mode", "two_phases": "Two Phases", "two_phases_tiled": "Two Phases with Tiling"}
ASPECT_RATIOS = ["1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16", "21:9", "9:21", "4:1", "1:4", "8:3", "3:8", "4:7", "7:4"]


def megapixels_to_resolution(megapixels, aspect_ratio, block_size, model_def=None):
    ratio_width, ratio_height = map(int, aspect_ratio.split(":"))
    if not math.isfinite(megapixels) or megapixels <= 0 or min(ratio_width, ratio_height) <= 0:
        raise ValueError("Megapixels and aspect ratio must be positive.")
    block = normalize_block_size(block_size)
    area, ratio = megapixels * 1_000_000, ratio_width / ratio_height
    if model_def is not None:
        area = min([area] + [align_dimension_to_block(model_def[key], block) ** 2 * factor for key, factor in (("max_width", 1 / ratio), ("max_height", ratio)) if model_def.get(key) is not None])
    width, height = (max(block, math.floor(value / block + 0.5) * block) for value in (math.sqrt(area * ratio), math.sqrt(area / ratio)))
    return f"{width}x{height}"


def megapixel_controls(resolution, resolution_choices, model_def, enable_4k_resolutions=False):
    width, height = parse_resolution(resolution)
    divisor = math.gcd(width, height)
    ratio = f"{width // divisor}:{height // divisor}"
    ratios = [(value, value) for value in ASPECT_RATIOS]
    # Keep familiar nominal preset ratios despite their block-aligned dimensions.
    label = next((label for label, value in resolution_choices if value == resolution), "")
    nominal = re.search(r"\((\d+:\d+)\)", label)
    if nominal and nominal[1] in ASPECT_RATIOS:
        ratio = nominal[1]
    elif ratio not in ASPECT_RATIOS:
        # A rounded canvas can still represent a familiar ratio. Allow only the
        # half-block error introduced by rounding each requested dimension.
        common = {value: int(value.split(":")[0]) / int(value.split(":")[1]) for value in ASPECT_RATIOS}
        nearest = min(common, key=lambda value: abs(math.log((width / height) / common[value])))
        half_block = normalize_block_size(model_def.get("vae_block_size", 16)) / 2
        if (width - half_block) / (height + half_block) <= common[nearest] <= (width + half_block) / (height - half_block):
            ratio = nearest
    if ratio not in ASPECT_RATIOS:
        ratios.append((ratio, ratio))
    mp = width * height / 1_000_000
    maximum = max([mp, 0.1] + [math.prod(parse_resolution(value)) / 1_000_000 for _, value in resolution_choices] + [value / 1_000_000 for value in available_resolution_budgets(model_def).values()])
    minimum = (100000 if enable_4k_resolutions and not model_def.get("lock_resolutions") else resolution_pixels(model_def.get("resolution_min")) or 100000) / 1_000_000
    maximum = max(minimum + 0.05, math.ceil(maximum / 0.05) * 0.05)
    limit_resolution = megapixels_to_resolution(maximum, ratio, model_def.get("vae_block_size", 16), model_def)
    if any(model_def.get(key) is not None for key in ("max_width", "max_height")):
        maximum = min(maximum, math.prod(parse_resolution(limit_resolution)) / 1_000_000)
    return {"ratio": ratio, "ratios": ratios, "value": min(mp, maximum), "minimum": min(minimum, mp, maximum), "maximum": maximum}


def resolution_advice(model_def, resolution, guidance_phases=1, video_prompt_type=""):
    if model_def.get("audio_only") or model_def.get("mesh_outputs") or not is_resolution_value(resolution):
        return "", ""
    assessment = assess_resolution(model_def, resolution, guidance_phases, video_prompt_type)
    budgets = available_resolution_budgets(model_def)
    current = resolution_mode(model_def, guidance_phases, video_prompt_type)
    budget = budgets.get(current)
    summary = f"Recommended Maximum: {budget / 1_000_000:.2f} MP ({RESOLUTION_MODE_LABELS[current]})." if budget else ""
    warning = ""
    if assessment["aligned"] != assessment["requested"]:
        warning = f"Canvas aligned from {assessment['requested']} to {assessment['aligned']}. "
    # UI advice is about the canvas that will actually be generated.
    if assessment["status"] != "not_allowed":
        assessment = assess_resolution(model_def, assessment["aligned"], guidance_phases, video_prompt_type)
    if assessment["status"] == "above_recommendation":
        warning += f"This size exceeds the recommended maximum of {budget / 1_000_000:.2f} MP; quality may degrade."
    elif assessment["status"] == "below_recommendation":
        warning += "This size is below the model's recommended minimum."
    elif assessment["status"] == "not_allowed":
        warning += assessment.get("reason", "Choose a resolution from this model's locked list.")
    elif assessment.get("mode", current) != current:
        warning += f"This size exceeds the current mode's recommended maximum of {budget / 1_000_000:.2f} MP. {RESOLUTION_MODE_LABELS[assessment['mode']]} is recommended for this size."
    return summary, warning.strip()
