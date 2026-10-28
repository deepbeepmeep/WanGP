# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from shared.utils.config_store import write_config
from decimal import Decimal, InvalidOperation

from postprocessing.mmaudio import MMAUDIO_DEFAULT_MODE
from postprocessing.mmaudio.audio_processor import MMAudioProcessor
from postprocessing.seedvc_processor.audio_processor import SeedVCProcessor
from shared.deepy.config import DEEPY_CONTEXT_TOKENS_KEY, DEEPY_CONTEXT_TOKENS_MIN, DEEPY_ENABLED_KEY, DEEPY_TEMPLATE_CONFIG_MIGRATIONS


LEGACY_EXTENSIONS_DEFAULTS_MIGRATED_KEY = "_extensions_defaults_migrated"
EXTENSIONS_DEFAULTS_VERSION_KEY = "extensions_defaults_version"
EXTENSIONS_DEFAULTS_TARGET_VERSION = Decimal("1.30")
EXTENSIONS_DEFAULTS_TARGET_VERSION_TEXT = str(EXTENSIONS_DEFAULTS_TARGET_VERSION)
INSTALLED_REMOTE_PLUGINS_KEY = "installed_remote_plugins"

SEEDVC_MODE_CHOICES = [("v1.0 Speech", 1), ("v1.0 Singing / F0 44k", 2), ("v2 Speech", 3)]
PROMPT_ENHANCER_CHOICES = [
    ("Florence 2 (image captioning) + LLama 3.2 3B (text generation)", 1),
    ("Florence 2 (image captioning) + Llama Joy 8B (uncensored, richer)", 2),
    ("Qwen3.5VL Abliterated 4B (recommended, captioning + uncensored text enhancement, vllm accelerated if available)", 3),
    ("Qwen3.5VL Abliterated 9B (captioning + uncensored high end text enhancement, vllm accelerated if available)", 4),
    ("Qwen3.8VL Uncensored 27B by Jonathan Coletti (highest quality, choose GGUF Q2, Q3 or Q4 below)", 5),
    ("Qwen3.8VL Uncensored 9B (distilled from Qwen3.8, Deepy Prime in about 6.5 GB VRAM, choose GGUF Q4 or Q8 below)", 6),
]

SEEDVC_DEFAULT_MODE = 2
PROMPT_ENHANCER_LOW_VRAM_DEFAULT_MODE = PROMPT_ENHANCER_CHOICES[0][1]
PROMPT_ENHANCER_HIGH_VRAM_DEFAULT_MODE = 3
PROMPT_ENHANCER_QWEN_MIN_VRAM_GB = 10
DEEPY_DEFAULT_ENABLED = 1
MEDIAFLOW_PLUGIN_ID = "media_flow"
PLUGIN_ID_MIGRATIONS = {
    "wan2gp-about": "about",
    "wan2gp-configuration": "configuration",
    "wan2gp-downloads": "downloads",
    "wan2gp-guides": "guides",
    "wan2gp-media-flow": MEDIAFLOW_PLUGIN_ID,
    "wan2gp-mediaflow": MEDIAFLOW_PLUGIN_ID,
    "media-flow": MEDIAFLOW_PLUGIN_ID,
    "wan2gp-models-manager": "models_manager",
    "models-manager": "models_manager",
    "wan2gp-motion-designer": "motion_designer",
    "motion-designer": "motion_designer",
    "wan2gp-plugin-manager": "plugin_manager",
    "plugin-manager": "plugin_manager",
    "wan2gp-process-full-video": MEDIAFLOW_PLUGIN_ID,
    "wan2gp-sample": "sample",
    "wan2gp-video-mask-creator": "video_mask_creator",
    "video-mask-creator": "video_mask_creator",
}
PROTECTED_PLUGIN_IDS = {
    "about",
    "agent_orchestrator",
    "configuration",
    "downloads",
    "guides",
    "media_flow",
    "models_manager",
    "motion_designer",
    "plugin_manager",
    "sample",
    "video_mask_creator",
}


def _to_int(value, default=0):
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


def _is_off(value) -> bool:
    return _to_int(value, 0) == 0


def _cuda_vram_gb() -> float:
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024 ** 3)
    except Exception:
        return 0.0


def get_prompt_enhancer_default_mode() -> int:
    return PROMPT_ENHANCER_HIGH_VRAM_DEFAULT_MODE if _cuda_vram_gb() >= PROMPT_ENHANCER_QWEN_MIN_VRAM_GB else PROMPT_ENHANCER_LOW_VRAM_DEFAULT_MODE


def _write_config(config, config_filename):
    if not config_filename:
        return
    write_config(config, config_filename)


def _set_missing_persistence(config, key):
    if key not in config or _to_int(config.get(key), 1) not in (1, 2):
        config[key] = 1


def _migrate_shared_processor_persistence(server_config) -> bool:
    from postprocessing import audio_processors as audio_processor_api
    from postprocessing import spatial_upsamplers as upsampler_api

    legacy_keys = ("flashvsr_persistence", "pid_persistence", "coz_persistence", "mmaudio_persistence", "seedvc_persistence")
    before = repr((server_config.get(upsampler_api.UPSAMPLER_CONFIG_KEY), server_config.get(audio_processor_api.AUDIO_PROCESSOR_CONFIG_KEY), {key: server_config.get(key) for key in legacy_keys if key in server_config}))
    spatial_sections = server_config.get(upsampler_api.UPSAMPLER_CONFIG_KEY)
    if not isinstance(spatial_sections, dict):
        spatial_sections = {}
        server_config[upsampler_api.UPSAMPLER_CONFIG_KEY] = spatial_sections
    flashvsr_section = spatial_sections.get("flashvsr", {})
    flashvsr_persistence = flashvsr_section.get("persistence", server_config.get("flashvsr_persistence", upsampler_api.PERSIST_UNLOAD)) if isinstance(flashvsr_section, dict) else server_config.get("flashvsr_persistence", upsampler_api.PERSIST_UNLOAD)
    spatial_sections[upsampler_api.PERSISTENCE_CONFIG_KEY] = upsampler_api.normalize_persistence(spatial_sections.get(upsampler_api.PERSISTENCE_CONFIG_KEY, flashvsr_persistence))
    for section in spatial_sections.values():
        if isinstance(section, dict):
            section.pop("persistence", None)
    for key in ("flashvsr_persistence", "pid_persistence", "coz_persistence"):
        server_config.pop(key, None)

    audio_sections = server_config.get(audio_processor_api.AUDIO_PROCESSOR_CONFIG_KEY)
    if not isinstance(audio_sections, dict):
        audio_sections = {}
        server_config[audio_processor_api.AUDIO_PROCESSOR_CONFIG_KEY] = audio_sections
    mmaudio_section = audio_sections.get("mmaudio", {})
    legacy_mmaudio_persistence = server_config.get("mmaudio_persistence", 2 if _to_int(server_config.get("mmaudio_enabled", 0), 0) == 2 else audio_processor_api.PERSIST_UNLOAD)
    mmaudio_persistence = mmaudio_section.get("persistence", legacy_mmaudio_persistence) if isinstance(mmaudio_section, dict) else legacy_mmaudio_persistence
    audio_sections[audio_processor_api.PERSISTENCE_CONFIG_KEY] = audio_processor_api.normalize_persistence(audio_sections.get(audio_processor_api.PERSISTENCE_CONFIG_KEY, mmaudio_persistence))
    for section in audio_sections.values():
        if isinstance(section, dict):
            section.pop("persistence", None)
    for key in ("mmaudio_persistence", "seedvc_persistence"):
        server_config.pop(key, None)
    after = repr((server_config.get(upsampler_api.UPSAMPLER_CONFIG_KEY), server_config.get(audio_processor_api.AUDIO_PROCESSOR_CONFIG_KEY), {key: server_config.get(key) for key in legacy_keys if key in server_config}))
    return before != after


def _migrate_audio_processors_config(server_config, version: Decimal) -> bool:
    from postprocessing import audio_processors as audio_processor_api

    changed = False
    sections = server_config.get(audio_processor_api.AUDIO_PROCESSOR_CONFIG_KEY, {})
    if not isinstance(sections, dict):
        sections = {}
        server_config[audio_processor_api.AUDIO_PROCESSOR_CONFIG_KEY] = sections
        changed = True
    else:
        server_config.setdefault(audio_processor_api.AUDIO_PROCESSOR_CONFIG_KEY, sections)
    legacy_keys = ("mmaudio_mode", "mmaudio_persistence", "mmaudio_enabled", "seedvc_mode", "seedvc_persistence")
    if any(key in server_config for key in legacy_keys):
        mmaudio_mode = server_config.get("mmaudio_mode", None)
        if mmaudio_mode is None:
            mmaudio_mode = 0 if _is_off(server_config.get("mmaudio_enabled", 0)) else MMAUDIO_DEFAULT_MODE
        if version < Decimal("1.1") and _is_off(mmaudio_mode):
            mmaudio_mode = MMAUDIO_DEFAULT_MODE
        sections["mmaudio"] = MMAudioProcessor.normalize_config_section({"mode": mmaudio_mode})

        seedvc_mode = server_config.get("seedvc_mode", SEEDVC_DEFAULT_MODE if version < Decimal("1.11") else 0)
        if version < Decimal("1.11") and _is_off(seedvc_mode):
            seedvc_mode = SEEDVC_DEFAULT_MODE
        sections["seedvc"] = SeedVCProcessor.normalize_config_section({"mode": seedvc_mode})
        for key in legacy_keys:
            if key in server_config:
                del server_config[key]
                changed = True
        changed = True
    changed = audio_processor_api.migrate_audio_processor_config(server_config) or changed
    return changed


def _migrate_temporal_upsamplers_config(server_config) -> bool:
    from postprocessing import temporal_upsamplers as temporal_upsampler_api

    changed = False
    sections = server_config.get(temporal_upsampler_api.TEMPORAL_UPSAMPLER_CONFIG_KEY, {})
    if not isinstance(sections, dict):
        sections = {}
        server_config[temporal_upsampler_api.TEMPORAL_UPSAMPLER_CONFIG_KEY] = sections
        changed = True
    else:
        server_config.setdefault(temporal_upsampler_api.TEMPORAL_UPSAMPLER_CONFIG_KEY, sections)
    if "rife_version" in server_config:
        del server_config["rife_version"]
        changed = True
    if "rife" in sections:
        del sections["rife"]
        changed = True
    changed = temporal_upsampler_api.migrate_temporal_upsampler_config(server_config) or changed
    return changed


def _migrate_deepy_template_names(server_config) -> bool:
    changed = False
    for key, migrations in DEEPY_TEMPLATE_CONFIG_MIGRATIONS.items():
        old_value = server_config.get(key)
        new_value = migrations.get(old_value)
        if new_value is not None:
            server_config[key] = new_value
            changed = True
    return changed


def _extension_defaults_version(config) -> Decimal:
    version = config.get(EXTENSIONS_DEFAULTS_VERSION_KEY, None)
    if version is not None:
        try:
            return Decimal(str(version))
        except (InvalidOperation, ValueError):
            return Decimal("1.0")
    return Decimal("1.1") if config.get(LEGACY_EXTENSIONS_DEFAULTS_MIGRATED_KEY, False) else Decimal("1.0")


def _migrate_int8_kernels(server_config) -> bool:
    changed = False
    if "int8_kernels" not in server_config:
        server_config["int8_kernels"] = "auto" if _to_int(server_config.get("enable_int8_kernels", 1), 0) == 1 else "disabled"
        changed = True
    if "enable_int8_kernels" in server_config:
        del server_config["enable_int8_kernels"]
        changed = True
    return changed


def migrate_extension_defaults(server_config, server_config_filename="", *, workspace_root=None) -> bool:
    if not isinstance(server_config, dict):
        return False

    version = _extension_defaults_version(server_config)
    prompt_enhancer_default_mode = get_prompt_enhancer_default_mode()
    changed = False

    if version < Decimal("1.1"):
        mmaudio_mode = server_config.get("mmaudio_mode", None)
        if mmaudio_mode is None:
            mmaudio_mode = 0 if _is_off(server_config.get("mmaudio_enabled", 0)) else MMAUDIO_DEFAULT_MODE
        if _is_off(mmaudio_mode):
            server_config["mmaudio_mode"] = MMAUDIO_DEFAULT_MODE
            changed = True
        else:
            server_config["mmaudio_mode"] = _to_int(mmaudio_mode, MMAUDIO_DEFAULT_MODE)
        _set_missing_persistence(server_config, "mmaudio_persistence")
        server_config["mmaudio_enabled"] = server_config["mmaudio_persistence"]

        if _is_off(server_config.get("seedvc_mode", 0)):
            server_config["seedvc_mode"] = SEEDVC_DEFAULT_MODE
            changed = True
        _set_missing_persistence(server_config, "seedvc_persistence")

        if _is_off(server_config.get("enhancer_enabled", 0)):
            server_config["enhancer_enabled"] = prompt_enhancer_default_mode
            changed = True

        if _is_off(server_config.get(DEEPY_ENABLED_KEY, 0)):
            server_config[DEEPY_ENABLED_KEY] = DEEPY_DEFAULT_ENABLED
            changed = True

    if version < Decimal("1.11"):
        if server_config.get("seedvc_mode") != SEEDVC_DEFAULT_MODE:
            server_config["seedvc_mode"] = SEEDVC_DEFAULT_MODE
            changed = True
        _set_missing_persistence(server_config, "seedvc_persistence")

    if version < Decimal("1.18"):
        changed = _migrate_shared_processor_persistence(server_config) or changed

    if version < Decimal("1.13"):
        from postprocessing import spatial_upsamplers as upsampler_api

        changed = upsampler_api.migrate_upsampler_config(server_config, prefer_legacy=True, apply_pre_1_1_defaults=version < Decimal("1.1")) or changed

    if version < Decimal("1.19"):
        from postprocessing import spatial_upsamplers as upsampler_api

        changed = upsampler_api.migrate_upsampler_config(server_config) or changed

    if version < Decimal("1.20"):
        changed = _migrate_deepy_template_names(server_config) or changed

    if version < Decimal("1.21"):
        changed = _migrate_int8_kernels(server_config) or changed

    if version < Decimal("1.22") and "kernel_precision" not in server_config:
        server_config["kernel_precision"] = "fast"
        changed = True

    if version < Decimal("1.23"):
        from shared.prompt_enhancer.config import PROMPT_ENHANCER_SPECULATIVE_DECODING_KEY, PROMPT_ENHANCER_SPECULATIVE_DECODING_DEFAULT, split_speculative_decoding, speculative_decoding_config
        key = PROMPT_ENHANCER_SPECULATIVE_DECODING_KEY
        server_config[key] = speculative_decoding_config(*split_speculative_decoding(server_config.get(key, PROMPT_ENHANCER_SPECULATIVE_DECODING_DEFAULT)))
        changed = True

    if version < Decimal("1.25"): # a VRAM preload per kind of output, Default (0) or Manual (a value set), Dynamic chosen later
        preload = server_config.pop("preload_in_VRAM", 0)
        server_config.pop("auto_preload_in_VRAM", None)
        for output_type in ("video", "image", "audio"):
            value = _to_int(server_config.setdefault(f"{output_type}_preload_in_VRAM", preload))
            server_config[f"{output_type}_preload_mode"] = "manual" if value > 0 else "default"
        changed = True

    if version < Decimal("1.29") and server_config.get("compile", "") == "":
        # Compilation off is "disabled" (1.26); 1.29 also corrects the configs that got an empty value again after 1.26, which
        # the Configuration tab could not save.
        server_config["compile"] = "disabled"
        changed = True

    if version < Decimal("1.30") and DEEPY_CONTEXT_TOKENS_KEY in server_config and _to_int(server_config[DEEPY_CONTEXT_TOKENS_KEY]) < DEEPY_CONTEXT_TOKENS_MIN:
        server_config[DEEPY_CONTEXT_TOKENS_KEY] = DEEPY_CONTEXT_TOKENS_MIN  # Deepy context window: 20,480 tokens at least
        changed = True

    if workspace_root is not None and version < Decimal('1.28'):
        # Also correct 1.27's overly broad skip of stale session lock files.
        from shared.deepy.session_store import migrate_session_folders
        from shared.utils.workspaces import migrate_session_references
        blocked = migrate_session_references(workspace_root)
        migrate_session_folders(blocked=blocked)

    changed = _migrate_audio_processors_config(server_config, version) or changed
    changed = _migrate_temporal_upsamplers_config(server_config) or changed
    from postprocessing.memory import migrate_config as migrate_postprocessor_memory
    changed = migrate_postprocessor_memory(server_config) or changed

    if server_config.pop(LEGACY_EXTENSIONS_DEFAULTS_MIGRATED_KEY, None) is not None:
        changed = True
    if server_config.get(EXTENSIONS_DEFAULTS_VERSION_KEY) != EXTENSIONS_DEFAULTS_TARGET_VERSION_TEXT:
        server_config[EXTENSIONS_DEFAULTS_VERSION_KEY] = EXTENSIONS_DEFAULTS_TARGET_VERSION_TEXT
        changed = True
    changed = migrate_bundled_plugin_ids(server_config) or changed
    changed = migrate_installed_remote_plugins(server_config) or changed

    if changed:
        _write_config(server_config, server_config_filename)
    return changed


def _migrate_plugin_id_list(server_config, key) -> bool:
    plugin_ids = server_config.get(key, [])
    if not isinstance(plugin_ids, list):
        return False
    changed = False
    migrated = []
    seen = set()
    for plugin_id in plugin_ids:
        plugin_id = str(plugin_id or "").strip()
        plugin_id = PLUGIN_ID_MIGRATIONS.get(plugin_id, plugin_id)
        if not plugin_id or plugin_id in seen:
            changed = True
            continue
        seen.add(plugin_id)
        migrated.append(plugin_id)
    if migrated != plugin_ids:
        server_config[key] = migrated
        changed = True
    return changed


def migrate_bundled_plugin_ids(server_config, server_config_filename="") -> bool:
    if not isinstance(server_config, dict):
        return False
    changed = _migrate_plugin_id_list(server_config, "enabled_plugins")
    changed = _migrate_plugin_id_list(server_config, "pending_plugin_deletions") or changed
    if changed:
        _write_config(server_config, server_config_filename)
    return changed


def migrate_mediaflow_plugin_id(server_config, server_config_filename="") -> bool:
    return migrate_bundled_plugin_ids(server_config, server_config_filename)


def migrate_installed_remote_plugins(server_config, server_config_filename="") -> bool:
    if not isinstance(server_config, dict):
        return False
    changed = _migrate_plugin_id_list(server_config, INSTALLED_REMOTE_PLUGINS_KEY)
    if INSTALLED_REMOTE_PLUGINS_KEY not in server_config:
        server_config[INSTALLED_REMOTE_PLUGINS_KEY] = [plugin_id for plugin_id in server_config.get("enabled_plugins", []) if plugin_id not in PROTECTED_PLUGIN_IDS]
        changed = True
    if changed:
        _write_config(server_config, server_config_filename)
    return changed


def enabled_choice_value(value, choices, default):
    allowed = {choice_value for _, choice_value in choices}
    value = _to_int(value, default)
    return value if value in allowed else default
