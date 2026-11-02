# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from packaging.version import InvalidVersion, Version
from shared.utils.config_store import config_lock, read_config


DEEPY_ENABLED_KEY = "deepy_enabled"
DEEPY_TYPE_KEY = "deepy_type"
DEEPY_VRAM_MODE_KEY = "deepy_vram_mode"
DEEPY_VOICE_LANGUAGE_KEY = "deepy_voice_language"
DEEPY_MODEL_SPEED_KEY = "deepy_model_speed"
DEEPY_MODEL_SIZE_KEY = "deepy_model_size"
DEEPY_TOOL_GEN_IMAGE_KEY = "deepy_tool_gen_image"
DEEPY_TOOL_EDIT_IMAGE_KEY = "deepy_tool_edit_image"
DEEPY_TOOL_GEN_VIDEO_KEY = "deepy_tool_gen_video"
DEEPY_TOOL_GEN_VIDEO_WITH_SPEECH_KEY = "deepy_tool_gen_video_with_speech"
DEEPY_TOOL_GEN_SONG_KEY = "deepy_tool_gen_song"
DEEPY_TOOL_GEN_VIDEO_WITH_REFS_KEY = "deepy_tool_gen_video_with_refs"
DEEPY_TOOL_GEN_SPEECH_FROM_DESCRIPTION_KEY = "deepy_tool_gen_speech_from_description"
DEEPY_TOOL_GEN_SPEECH_FROM_SAMPLE_KEY = "deepy_tool_gen_speech_from_sample"
DEEPY_TOOL_GEN_WORLD_KEY = "deepy_tool_gen_world"
DEEPY_TOOL_GEN_3D_KEY = "deepy_tool_gen_3d"
DEEPY_TOOL_TEXTURE_3D_KEY = "deepy_tool_texture_3d"
DEEPY_CONTEXT_TOKENS_KEY = "deepy_context_tokens"
DEEPY_KV_CACHE_QUANTIZATION_KEY = "deepy_kv_cache_quantization"
DEEPY_COMPACTION_TYPE_KEY = "deepy_compaction_type"
DEEPY_COMPACTION_THINKING_KEY = "deepy_compaction_thinking"
DEEPY_REPETITION_PENALTY_KEY = "deepy_repetition_penalty"
DEEPY_ZERO_CUSTOM_SYSTEM_PROMPT_KEY = "deepy_zero_custom_system_prompt"
DEEPY_PRIME_CUSTOM_SYSTEM_PROMPT_KEY = "deepy_prime_custom_system_prompt"
DEEPY_PRIME_MCP_SERVERS_KEY = "deepy_prime_mcp_servers"
DEEPY_MCP_AUTO_DISCOVER_PATHS_KEY = "deepy_mcp_auto_discover_paths"
DEEPY_ALLOW_READ_FILE_SYSTEM_KEY = "deepy_allow_read_file_system"
DEEPY_FILE_SYSTEM_PATHS_KEY = "deepy_file_system_paths"
DEEPY_READ_EVERYWHERE_KEY = "deepy_read_everywhere"
DEEPY_AUTO_CANCEL_QUEUE_TASKS_KEY = "deepy_auto_cancel_queue_tasks"
DEEPY_SEPARATE_REQUESTS_WITH_EMPTY_LINE_KEY = "deepy_separate_requests_with_empty_line"
DEEPY_SESSION_RESET_MODE_KEY = "deepy_session_reset_mode"
DEEPY_SESSION_GALLERY_MEDIA_MODE_KEY = "deepy_session_gallery_media_mode"
DEEPY_MULTI_SESSION_KEY = "deepy_multi_session"
DEEPY_GALLERY_MEDIA_VIEWER_KEY = "deepy_gallery_media_viewer"  # chat images and videos open in the gallery's full-screen viewer instead of a new browser tab
DEEPY_GALLERY_MEDIA_VIEWER_DEFAULT = True
DEEPY_TEMPLATE_CONFIG_MIGRATIONS = {
    DEEPY_TOOL_GEN_IMAGE_KEY: {
        "Qwen Image 2.1 Viggle Turbo 8 Steps": "Qwen Image 2.1 Viggle Turbo v0.2.1 6 Steps",
    },
    DEEPY_TOOL_EDIT_IMAGE_KEY: {
        "Qwen Image 2.1 Viggle Turbo 8 Steps": "Qwen Image 2.1 Viggle Turbo v0.2.1 6 Steps",
    },
    DEEPY_TOOL_GEN_VIDEO_KEY: {
        "MiniMax H3 FL2VA Turbo Lightx2v 8 Steps": "MiniMax H3 FL2VA Pruned Turbo Lightx2v 8 Steps",
        "Pika H3 Flex FL2VA 6 Steps": "Pika H3 Flex FL2VA Pruned 6 Steps",  # Pika templates moved to the pruned models (2026-10-31)
    },
    DEEPY_TOOL_GEN_VIDEO_WITH_SPEECH_KEY: {
        "MiniMax H3 FL2VA Turbo Lightx2v 8 Steps With Sound": "MiniMax H3 FL2VA Pruned Turbo Lightx2v 8 Steps With Sound",
        "Pika H3 Flex Ref2VA 6 Steps With Audio References": "Pika H3 Flex Ref2VA Pruned 6 Steps With Audio References",
    },
    DEEPY_TOOL_GEN_VIDEO_WITH_REFS_KEY: {
        "Pika H3 Flex Ref2VA 6 Steps": "Pika H3 Flex Ref2VA Pruned 6 Steps",
    },
    DEEPY_TOOL_GEN_3D_KEY: {
        "TRELLIS.2 4B": "TRELLIS.2 + SUPRA-Tex PBR 4B + 2.1B",  # TRELLIS.2 split into a Shape model and a Texturer (2026-10-29)
    },
}

DEEPY_VRAM_MODE_UNLOAD = "unload"
DEEPY_VRAM_MODE_ALWAYS_LOADED = "always_loaded"
DEEPY_VRAM_MODE_UNLOAD_ON_REQUEST = "unload_on_request"
DEEPY_TYPE_ZERO = "zero"
DEEPY_TYPE_PRIME = "prime"
DEEPY_TYPE_DISABLED = "disabled"
DEEPY_TYPE_DEFAULT = DEEPY_TYPE_ZERO
DEEPY_PRIME_GUIDANCE_DEFAULT = "Use the current Speed and Model Size preferences for choices left unspecified by the user when selecting beyond configured templates."
_DEEPY_PRIME_GUIDANCE_OLD_DEFAULT = "When several models can satisfy the request, prefer the highest-quality base or full model unless the user explicitly prioritizes speed or names another model."
DEEPY_COMPACTION_TYPE_DISCARD = "discard"
DEEPY_COMPACTION_TYPE_SUMMARIZE = "summarize"
DEEPY_COMPACTION_CHOICE_THINKING = "summarize_thinking"
DEEPY_DEFAULT_GEN_IMAGE = "Qwen Image 2.1 Turbo 8 Steps"
DEEPY_DEFAULT_EDIT_IMAGE = "Qwen Image 2.1 Turbo 8 Steps"
DEEPY_DEFAULT_GEN_VIDEO = "MiniMax H3 FL2VA Pruned PDD 8 Steps"
DEEPY_DEFAULT_GEN_VIDEO_WITH_SPEECH = "MiniMax H3 FL2VA Pruned PDD 8 Steps With Sound"
DEEPY_DEFAULT_GEN_SONG = "ACE-Step 1.5 Turbo LM 1.7B"
DEEPY_DEFAULT_GEN_VIDEO_WITH_REFS = "MiniMax H3 Ref2VA Pruned PDD 8 Steps"
DEEPY_DEFAULT_GEN_SPEECH_FROM_DESCRIPTION = "Qwen3 1.7B"
DEEPY_DEFAULT_GEN_SPEECH_FROM_SAMPLE = "Index TTS 2"
DEEPY_DEFAULT_GEN_WORLD = "Hy-Nomas World Model 8B"
DEEPY_DEFAULT_GEN_3D = "TRELLIS.2 + SUPRA-Tex PBR 4B + 2.1B"
DEEPY_DEFAULT_TEXTURE_3D = "SUPRA-Tex"
DEEPY_CONTEXT_TOKENS_MIN = 20480  # Deepy Zero's system prompt and tools alone take about 12k tokens
DEEPY_CONTEXT_TOKENS_MAX = 256000
DEEPY_CONTEXT_TOKENS_DEFAULT = 20480
DEEPY_COMPACTION_SUMMARIZE_MIN_TOKENS = 32000
DEEPY_COMPACTION_THINKING_MIN_TOKENS = 48000
DEEPY_COMPACTION_TYPE_DEFAULT = DEEPY_COMPACTION_TYPE_DISCARD
DEEPY_COMPACTION_THINKING_DEFAULT = False
DEEPY_REPETITION_PENALTY_DEFAULT = True
DEEPY_KV_CACHE_QUANTIZATION_AUTO = "auto"
DEEPY_KV_CACHE_QUANTIZATION_DEFAULT = DEEPY_KV_CACHE_QUANTIZATION_AUTO
DEEPY_KV_CACHE_QUANTIZATION_MIN_GGUF_KERNELS_VERSION = "1.0.13"
DEEPY_ALLOW_READ_FILE_SYSTEM_DEFAULT = False
DEEPY_FILE_SYSTEM_ACCESS_DISABLED = "disabled"
DEEPY_FILE_SYSTEM_ACCESS_READ = "read"
DEEPY_FILE_SYSTEM_ACCESS_READ_WRITE = "read_write"
DEEPY_FILE_SYSTEM_ACCESS_DEFAULT = DEEPY_FILE_SYSTEM_ACCESS_DISABLED
DEEPY_FILE_SYSTEM_PATHS_DEFAULT: list[str] = []
DEEPY_READ_EVERYWHERE_DEFAULT = False
DEEPY_PYTHON_SANDBOX_KEY = "deepy_python_sandbox"
DEEPY_PYTHON_SANDBOX_DEFAULT = "enabled"
PYTHON_APPS_KEY = "python_apps"  # shared config group of App Development (Advanced): {"development": "off" | "apps" | "plugins"}
PYTHON_APPS_LEVELS = ("off", "apps", "plugins")
DEEPY_WEB_SEARCH_KEY = "deepy_web_search"
DEEPY_WEB_SEARCH_DEFAULT = "disabled"
DEEPY_ADVANCED_TRANSCRIPTION_KEY = "deepy_advanced_transcription"  # the transcription model of Prime's advanced_transcription toolbox action, or disabled
DEEPY_ADVANCED_TRANSCRIPTION_DEFAULT = "disabled"
DEEPY_ADVANCED_TRANSCRIPTION_CHOICES = [
    ("Use Whisper Only", "disabled"),
    ("Nemotron INT8 Experts", "nemotron_transcribe"),
    ("Nemotron Q4 Experts (Recommended)", "nemotron_transcribe_q4"),
]
DEEPY_TRANSCRIPTION_TERMS_KEY = "deepy_transcription_terms"  # names and jargon added to every advanced transcription (Nemotron context terms)
CIVITAI_API_KEY_KEY = "civitai_api_key"  # shared config: Civitai downloads of LoRAs through Deepy's web toolbox (searching needs none); plaintext only when not stored securely
CIVITAI_API_KEY_SECURE_KEY = "civitai_api_key_secure"  # the key lives in the OS credential manager under the credential id below
CIVITAI_API_KEY_CREDENTIAL_ID_KEY = "civitai_api_key_credential_id"
HUGGINGFACE_TOKEN_KEY = "huggingface_token"  # optional: gated or private Hugging Face repositories for Deepy's LoRA hub; public ones need none
HUGGINGFACE_TOKEN_SECURE_KEY = "huggingface_token_secure"
HUGGINGFACE_TOKEN_CREDENTIAL_ID_KEY = "huggingface_token_credential_id"
SECRET_CONFIG_KEYS = {"civitai": (CIVITAI_API_KEY_KEY, CIVITAI_API_KEY_SECURE_KEY, CIVITAI_API_KEY_CREDENTIAL_ID_KEY), "huggingface": (HUGGINGFACE_TOKEN_KEY, HUGGINGFACE_TOKEN_SECURE_KEY, HUGGINGFACE_TOKEN_CREDENTIAL_ID_KEY)}
DEEPY_MCP_AUTO_DISCOVER_PATHS_DEFAULT = False
DEEPY_AUTO_CANCEL_QUEUE_TASKS_DEFAULT = True
DEEPY_SEPARATE_REQUESTS_WITH_EMPTY_LINE_DEFAULT = True
DEEPY_SESSION_RESET_MODE_DEFAULT = "new_session"
DEEPY_SESSION_GALLERY_MEDIA_MODE_DEFAULT = "link"
DEEPY_MULTI_SESSION_DISABLED = "disabled"
DEEPY_MULTI_SESSION_SELECTABLE = "selectable"
DEEPY_MULTI_SESSION_DEDICATED = "dedicated"
DEEPY_MULTI_SESSION_DEFAULT = DEEPY_MULTI_SESSION_DISABLED
DEEPY_MULTI_SESSION_CHOICES = [
    ("Disabled", DEEPY_MULTI_SESSION_DISABLED),
    ("Multisessions with dedicated Workspace", DEEPY_MULTI_SESSION_DEDICATED),
    ("Multisessions with selectable Workspace", DEEPY_MULTI_SESSION_SELECTABLE),
]
DEEPY_CONFIG_FILENAME = "wgp_config.json"

# Experimental, intentionally not user-configurable yet. The tools remain gated by
# Deepy's existing read/write filesystem permission in addition to this switch.
DEEPY_LONG_TEXT_TOOLS_EXPERIMENT = True

DEEPY_QWEN_ENHANCER_IDS = {3, 4, 5, 6}
_DEEPY_QWEN_VARIANT_LABELS = {3: "Qwen3.5-4B", 4: "Qwen3.5-9B", 5: "Qwen3.8-27B", 6: "Qwen3.8-9B"}
_DEEPY_QWEN_KV_CACHE_SPECS = {
    3: {"num_kv_cache_layers": 8, "num_key_value_heads": 4, "head_dim": 256, "dtype_bytes": 2, "kvcache_block_size": 256},
    4: {"num_kv_cache_layers": 8, "num_key_value_heads": 4, "head_dim": 256, "dtype_bytes": 2, "kvcache_block_size": 256},
    5: {"num_kv_cache_layers": 16, "num_key_value_heads": 4, "head_dim": 256, "dtype_bytes": 2, "kvcache_block_size": 256},
    6: {"num_kv_cache_layers": 8, "num_key_value_heads": 4, "head_dim": 256, "dtype_bytes": 2, "kvcache_block_size": 256},
}
_DEEPY_DEFAULT_GEN_IMAGE_ALIASES = {"Z_Image_Turbo": "Z Image Turbo", **DEEPY_TEMPLATE_CONFIG_MIGRATIONS[DEEPY_TOOL_GEN_IMAGE_KEY]}
_DEEPY_DEFAULT_EDIT_IMAGE_ALIASES = {"Qwen_Edit": DEEPY_DEFAULT_EDIT_IMAGE, **DEEPY_TEMPLATE_CONFIG_MIGRATIONS[DEEPY_TOOL_EDIT_IMAGE_KEY]}
_DEEPY_DEFAULT_GEN_VIDEO_ALIASES = {
    "ltx2_22B_distilled": "LTX-2 2.3 Distilled 1.0",
    "LTX-2 2.3 Distilled": "LTX-2 2.3 Distilled 1.0",
    **DEEPY_TEMPLATE_CONFIG_MIGRATIONS[DEEPY_TOOL_GEN_VIDEO_KEY],
}
_DEEPY_GEN_VIDEO_WITH_SPEECH_ALIASES = {
    "LTX-2.3 Distilled With Sound": "LTX-2.3 Distilled 1.0 With Sound",
    **DEEPY_TEMPLATE_CONFIG_MIGRATIONS[DEEPY_TOOL_GEN_VIDEO_WITH_SPEECH_KEY],
}
_DEEPY_RUNTIME_CONFIG: dict[str, Any] | None = None
_DEEPY_RUNTIME_CONFIG_FILENAME = ""


def normalize_deepy_voice_language(value: Any) -> str:
    language = str(value or "auto").strip().lower()
    if language == "auto":
        return language
    from whisper.tokenizer import LANGUAGES, TO_LANGUAGE_CODE

    language = TO_LANGUAGE_CODE.get(language, language)
    if language not in LANGUAGES:
        raise ValueError(f"Unsupported Whisper language: {value}")
    return language


def deepy_voice_language_choices() -> list[tuple[str, str]]:
    from whisper.tokenizer import LANGUAGES

    return [("Auto", "auto"), *sorted((name.title(), code) for code, name in LANGUAGES.items())]


def normalize_deepy_enabled(value: Any) -> int:
    return 1 if int(value or 0) != 0 else 0


def normalize_deepy_type(value: Any) -> str:
    return DEEPY_TYPE_PRIME if str(value or "").strip().lower() == DEEPY_TYPE_PRIME else DEEPY_TYPE_ZERO


def deepy_mode_from_config(enabled: Any, deepy_type: Any) -> str:
    return normalize_deepy_type(deepy_type) if normalize_deepy_enabled(enabled) else DEEPY_TYPE_DISABLED


def split_deepy_mode(value: Any) -> tuple[int, str]:
    text = str(value or "").strip().lower()
    if text == DEEPY_TYPE_PRIME:
        return 1, DEEPY_TYPE_PRIME
    if text == DEEPY_TYPE_ZERO:
        return 1, DEEPY_TYPE_ZERO
    return 0, DEEPY_TYPE_ZERO


def normalize_deepy_vram_mode(value: Any) -> str:
    text = str(value or "").strip()
    if text == DEEPY_VRAM_MODE_ALWAYS_LOADED:
        return DEEPY_VRAM_MODE_ALWAYS_LOADED
    if text == DEEPY_VRAM_MODE_UNLOAD_ON_REQUEST:
        return DEEPY_VRAM_MODE_UNLOAD_ON_REQUEST
    return DEEPY_VRAM_MODE_UNLOAD


def _normalize_deepy_variant(value: Any, aliases: dict[str, str], default: str) -> str:
    text = str(value or "").strip()
    return aliases.get(text, text) or default


def normalize_deepy_tool_gen_image(value: Any) -> str:
    return _normalize_deepy_variant(value, _DEEPY_DEFAULT_GEN_IMAGE_ALIASES, DEEPY_DEFAULT_GEN_IMAGE)


def normalize_deepy_tool_edit_image(value: Any) -> str:
    return _normalize_deepy_variant(value, _DEEPY_DEFAULT_EDIT_IMAGE_ALIASES, DEEPY_DEFAULT_EDIT_IMAGE)


def normalize_deepy_tool_gen_video(value: Any) -> str:
    return _normalize_deepy_variant(value, _DEEPY_DEFAULT_GEN_VIDEO_ALIASES, DEEPY_DEFAULT_GEN_VIDEO)


def normalize_deepy_tool_gen_video_with_speech(value: Any) -> str:
    return _normalize_deepy_variant(value, _DEEPY_GEN_VIDEO_WITH_SPEECH_ALIASES, DEEPY_DEFAULT_GEN_VIDEO_WITH_SPEECH)


def normalize_deepy_tool_gen_video_with_refs(value: Any) -> str:
    return _normalize_deepy_variant(value, DEEPY_TEMPLATE_CONFIG_MIGRATIONS[DEEPY_TOOL_GEN_VIDEO_WITH_REFS_KEY], DEEPY_DEFAULT_GEN_VIDEO_WITH_REFS)


def normalize_deepy_tool_gen_song(value: Any) -> str:
    return _normalize_deepy_variant(value, {}, DEEPY_DEFAULT_GEN_SONG)


def normalize_deepy_tool_gen_speech_from_description(value: Any) -> str:
    return _normalize_deepy_variant(value, {}, DEEPY_DEFAULT_GEN_SPEECH_FROM_DESCRIPTION)


def normalize_deepy_tool_gen_speech_from_sample(value: Any) -> str:
    return _normalize_deepy_variant(value, {}, DEEPY_DEFAULT_GEN_SPEECH_FROM_SAMPLE)


def normalize_deepy_tool_gen_world(value: Any) -> str:
    return _normalize_deepy_variant(value, {}, DEEPY_DEFAULT_GEN_WORLD)


def normalize_deepy_tool_gen_3d(value: Any) -> str:
    return _normalize_deepy_variant(value, DEEPY_TEMPLATE_CONFIG_MIGRATIONS[DEEPY_TOOL_GEN_3D_KEY], DEEPY_DEFAULT_GEN_3D)


def normalize_deepy_tool_texture_3d(value: Any) -> str:
    return _normalize_deepy_variant(value, {}, DEEPY_DEFAULT_TEXTURE_3D)


def normalize_deepy_context_tokens(value: Any) -> int:
    try:
        tokens = int(value or DEEPY_CONTEXT_TOKENS_DEFAULT)
    except Exception:
        tokens = DEEPY_CONTEXT_TOKENS_DEFAULT
    return max(DEEPY_CONTEXT_TOKENS_MIN, min(DEEPY_CONTEXT_TOKENS_MAX, tokens))


def normalize_deepy_kv_cache_quantization(value: Any) -> str:
    """"" (BF16), "auto", "int8", or a mixed layout "body[:exact]" in its canonical form (shared.llm_engines.lmgp.layers.kv_formats)."""
    text = str(DEEPY_KV_CACHE_QUANTIZATION_DEFAULT if value is None else value).strip().lower()
    if text == DEEPY_KV_CACHE_QUANTIZATION_AUTO:
        return DEEPY_KV_CACHE_QUANTIZATION_AUTO
    if text == "":
        return ""
    from shared.llm_engines.lmgp.layers.kv_formats import is_layout_value, parse_layout
    return parse_layout(text).value if is_layout_value(text) else ""


def split_deepy_kv_cache_quantization(value: Any) -> tuple[str, str]:
    """(body choice, exact-overlay choice) of a config value for the two dropdowns; the exact part is the default for non-mixed values."""
    from shared.llm_engines.lmgp.layers.kv_formats import DEFAULT_EXACT
    body, _, exact = normalize_deepy_kv_cache_quantization(value).partition(":")
    return body, exact or DEFAULT_EXACT


def join_deepy_kv_cache_quantization(body: Any, exact: Any) -> str:
    body = str(body or "").strip().lower()
    if body in ("", "int8", DEEPY_KV_CACHE_QUANTIZATION_AUTO):
        return normalize_deepy_kv_cache_quantization(body)
    return normalize_deepy_kv_cache_quantization(f"{body}:{str(exact or '').strip().lower()}")


def deepy_kv_cache_quantization_is_mixed(value: Any) -> bool:
    return normalize_deepy_kv_cache_quantization(value) not in ("", "int8", DEEPY_KV_CACHE_QUANTIZATION_AUTO)


def describe_deepy_kv_cache_quantization(mode: str) -> str:
    """Short label of a resolved mode: BF16, INT8, or the body formats of a mixed layout."""
    if not mode:
        return "BF16"
    from shared.llm_engines.lmgp.layers.kv_formats import parse_layout
    layout = parse_layout(mode)
    names = {"q8": "INT8", "q6_0": "Q6", "q4_1": "Q4.1", "q4_0": "Q4"}
    return names[layout.k.name] if layout.k == layout.v else f"{names[layout.k.name]}/{names[layout.v.name]}"


@lru_cache(maxsize=1)
def get_gguf_kernels_version() -> str:
    try:
        return version("llamacpp-gguf-cuda")
    except PackageNotFoundError:
        return ""


def resolve_deepy_kv_cache_quantization(value: Any) -> tuple[str, str]:
    mode = normalize_deepy_kv_cache_quantization(value)
    if mode != DEEPY_KV_CACHE_QUANTIZATION_AUTO:
        return mode, ""
    installed_version = get_gguf_kernels_version()
    try:
        supported = bool(installed_version) and Version(installed_version) >= Version(DEEPY_KV_CACHE_QUANTIZATION_MIN_GGUF_KERNELS_VERSION)
    except InvalidVersion:
        supported = False
    if supported:
        return "int8", f"Fast KV Cache quantization enabled automatically because GGUF kernels {installed_version} were found."
    found = f" {installed_version}" if installed_version else ""
    return "", f"Fast KV Cache quantization disabled automatically because GGUF kernels{found} do not meet the {DEEPY_KV_CACHE_QUANTIZATION_MIN_GGUF_KERNELS_VERSION}+ requirement."


def normalize_deepy_compaction_type(value: Any) -> str:
    return DEEPY_COMPACTION_TYPE_SUMMARIZE if str(value or "").strip().lower() in {DEEPY_COMPACTION_TYPE_SUMMARIZE, DEEPY_COMPACTION_CHOICE_THINKING} else DEEPY_COMPACTION_TYPE_DISCARD


def normalize_deepy_compaction_thinking(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "on", "yes"}
    return bool(value)


def normalize_deepy_repetition_penalty(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", "disabled"}
    return bool(value)


def deepy_compaction_min_tokens(compaction_type: Any, compaction_thinking: Any = False) -> int:
    thinking = str(compaction_type or "").strip().lower() == DEEPY_COMPACTION_CHOICE_THINKING or normalize_deepy_compaction_thinking(compaction_thinking)
    return DEEPY_COMPACTION_THINKING_MIN_TOKENS if thinking else DEEPY_COMPACTION_SUMMARIZE_MIN_TOKENS


def validate_deepy_compaction_config(compaction_type: Any, context_tokens: Any, *, compaction_thinking: Any = False) -> str:
    normalized_type = normalize_deepy_compaction_type(compaction_type)
    normalized_tokens = normalize_deepy_context_tokens(context_tokens)
    minimum_tokens = deepy_compaction_min_tokens(compaction_type, compaction_thinking)
    if normalized_type == DEEPY_COMPACTION_TYPE_SUMMARIZE and normalized_tokens < minimum_tokens:
        mode = "Summarize with Thinking" if minimum_tokens == DEEPY_COMPACTION_THINKING_MIN_TOKENS else "Summarize"
        raise ValueError(f"Deepy {mode} compaction requires at least {minimum_tokens:,} context tokens.")
    return normalized_type


def validate_deepy_version_config(deepy_type: Any, compaction_type: Any, context_tokens: Any, enhancer_enabled: Any, *, compaction_thinking: Any = False) -> tuple[str, str, int]:
    normalized_type = normalize_deepy_type(deepy_type)
    normalized_compaction = normalize_deepy_compaction_type(compaction_type)
    normalized_tokens = normalize_deepy_context_tokens(context_tokens)
    try:
        enhancer_no = int(enhancer_enabled or 0)
    except (TypeError, ValueError):
        enhancer_no = 0
    if normalized_type == DEEPY_TYPE_PRIME:
        if enhancer_no not in (5, 6):
            raise ValueError("Deepy Prime requires a Qwen3.8 VL model.")
        if normalized_compaction != DEEPY_COMPACTION_TYPE_SUMMARIZE:
            raise ValueError("Deepy Prime requires Summarize compaction.")
    validate_deepy_compaction_config(compaction_type, normalized_tokens, compaction_thinking=compaction_thinking)
    return normalized_type, normalized_compaction, normalized_tokens


def normalize_deepy_custom_system_prompt(value: Any) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    return text


def normalize_deepy_prime_guidance(value: Any) -> str:
    text = normalize_deepy_custom_system_prompt(value)
    return DEEPY_PRIME_GUIDANCE_DEFAULT if not text or text == _DEEPY_PRIME_GUIDANCE_OLD_DEFAULT else text


def normalize_deepy_prime_mcp_servers(value: Any) -> dict[str, dict[str, Any]]:
    if isinstance(value, str):
        value = json.loads(value or "{}")
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("Deepy Prime external MCP servers must be a JSON object.")
    normalized = {}
    for raw_name, raw_config in value.items():
        name = str(raw_name or "").strip()
        if not name or not isinstance(raw_config, dict):
            raise ValueError("Each Deepy Prime external MCP server needs a non-empty name and a configuration object.")
        config = dict(raw_config)
        transport = str(config.get("transport", "stdio") or "stdio").strip().lower().replace("_", "-")
        if transport not in {"stdio", "sse", "streamable-http"}:
            raise ValueError(f"Deepy Prime MCP server '{name}' has an unsupported transport: {transport}.")
        if transport == "stdio" and not str(config.get("command", "") or "").strip():
            raise ValueError(f"Deepy Prime MCP stdio server '{name}' requires command.")
        if transport != "stdio" and not str(config.get("url", "") or "").strip():
            raise ValueError(f"Deepy Prime MCP server '{name}' requires url.")
        config["transport"] = transport
        normalized[name] = config
    return normalized


def normalize_deepy_auto_cancel_queue_tasks(value: Any) -> bool:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "0", "false", "off", "no"}:
            return False
        if text in {"1", "true", "on", "yes"}:
            return True
    return bool(value)


def normalize_deepy_file_system_access(value: Any) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in {DEEPY_FILE_SYSTEM_ACCESS_DISABLED, "", "0", "false", "off", "no"}:
            return DEEPY_FILE_SYSTEM_ACCESS_DISABLED
        if normalized in {DEEPY_FILE_SYSTEM_ACCESS_READ_WRITE, "readwrite", "write", "rw", "2"}:
            return DEEPY_FILE_SYSTEM_ACCESS_READ_WRITE
        if normalized in {DEEPY_FILE_SYSTEM_ACCESS_READ, "1", "true", "on", "yes"}:
            return DEEPY_FILE_SYSTEM_ACCESS_READ
        return DEEPY_FILE_SYSTEM_ACCESS_DISABLED
    return DEEPY_FILE_SYSTEM_ACCESS_READ if bool(value) else DEEPY_FILE_SYSTEM_ACCESS_DISABLED


def normalize_deepy_allow_read_file_system(value: Any) -> bool:
    return normalize_deepy_file_system_access(value) != DEEPY_FILE_SYSTEM_ACCESS_DISABLED


def normalize_deepy_file_system_paths(value: Any) -> list[str]:
    values = value.splitlines() if isinstance(value, str) else value if isinstance(value, (list, tuple)) else []
    paths = []
    seen = set()
    for item in values:
        path = str(item or "").strip()
        key = path.replace("\\", "/").casefold()
        if not path or key in seen:
            continue
        seen.add(key)
        paths.append(path)
    return paths


def parse_deepy_file_system_paths(value: Any) -> list[tuple[str, str]]:
    parsed = []
    aliases = set()
    for entry in normalize_deepy_file_system_paths(value):
        if entry[0] in {'"', "'"}:
            quote = entry[0]
            end = entry.find(quote, 1)
            if end < 0:
                raise ValueError(f"Additional filesystem folder has an unterminated quote: {entry}")
            tail = entry[end + 1:]
            if tail and not tail[0].isspace():
                raise ValueError(f"Filesystem alias must be separated from its quoted path by a space: {entry}")
            path, remainder = entry[1:end].strip(), tail.strip()
            if remainder and any(character.isspace() for character in remainder):
                raise ValueError(f"Filesystem alias must be one word: {remainder}")
            alias = remainder
        else:
            parts = entry.rsplit(None, 1)
            path, alias = (parts[0], parts[1]) if len(parts) == 2 else (entry, "")
        if not path:
            raise ValueError("Additional filesystem folder path is empty.")
        if alias:
            if re.fullmatch(r"[A-Za-z0-9_-]+", alias) is None:
                raise ValueError(f"Filesystem alias may contain only letters, numbers, '_' and '-': {alias}")
            key = alias.casefold()
            if re.fullmatch(r"outputs\d*", key):
                raise ValueError(f"Filesystem alias is reserved for WanGP output folders: {alias}")
            if key in aliases:
                raise ValueError(f"Filesystem alias is used more than once: {alias}")
            aliases.add(key)
        parsed.append((path, alias))
    return parsed


def normalize_deepy_read_everywhere(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "on", "yes"}
    return bool(value)


def normalize_deepy_python_sandbox(value: Any) -> str:
    if isinstance(value, bool):
        return "enabled" if value else "disabled"
    return "disabled" if str(value or "").strip().lower() in {"disabled", "off", "false", "0", "no"} else "enabled"


def normalize_python_apps(value: Any) -> dict[str, Any]:
    group = dict(value) if isinstance(value, dict) else {}
    level = str(group.get("development") or "off").strip().lower()
    group["development"] = level if level in PYTHON_APPS_LEVELS else "off"
    return group


def python_apps_level() -> str:
    """App Development (Advanced): "off", "apps" (web apps) or "plugins"; always "off" while the Python sandbox is disabled."""
    if normalize_deepy_python_sandbox(get_deepy_config_value(DEEPY_PYTHON_SANDBOX_KEY, DEEPY_PYTHON_SANDBOX_DEFAULT)) != "enabled":
        return "off"
    return normalize_python_apps(get_deepy_config_value(PYTHON_APPS_KEY))["development"]


def normalize_deepy_web_search(value: Any) -> str:
    if isinstance(value, bool):
        return "enabled" if value else "disabled"
    return "enabled" if str(value or "").strip().lower() in {"enabled", "on", "true", "1", "yes"} else "disabled"


def normalize_deepy_advanced_transcription(value: Any) -> str:
    value = str(value or "").strip()
    return value if value in {choice for _label, choice in DEEPY_ADVANCED_TRANSCRIPTION_CHOICES} else DEEPY_ADVANCED_TRANSCRIPTION_DEFAULT


def advanced_transcription_model() -> str:
    """The model type of Prime's advanced transcription, or "" while the user has not enabled it."""
    model_type = normalize_deepy_advanced_transcription(get_deepy_config_value(DEEPY_ADVANCED_TRANSCRIPTION_KEY, DEEPY_ADVANCED_TRANSCRIPTION_DEFAULT))
    return "" if model_type == DEEPY_ADVANCED_TRANSCRIPTION_DEFAULT else model_type


def advanced_transcription_terms() -> str:
    """The user's names and jargon for every advanced transcription, comma- or line-separated."""
    return str(get_deepy_config_value(DEEPY_TRANSCRIPTION_TERMS_KEY, "") or "").strip()


def normalize_deepy_mcp_auto_discover_paths(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "on", "yes"}
    return bool(value)


def normalize_deepy_separate_requests_with_empty_line(value: Any) -> bool:
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "0", "false", "off", "no"}:
            return False
        if text in {"1", "true", "on", "yes"}:
            return True
    return bool(value)


def normalize_deepy_session_reset_mode(value: Any) -> str:
    return "reset_session" if str(value or "").strip().lower() == "reset_session" else "new_session"


def normalize_deepy_session_gallery_media_mode(value: Any) -> str:
    return "copy" if str(value or "").strip().lower() == "copy" else "link"


def normalize_deepy_multi_session(value: Any) -> bool:
    return normalize_deepy_session_mode(value) != DEEPY_MULTI_SESSION_DISABLED


def normalize_deepy_session_mode(value: Any) -> str:
    if isinstance(value, str):
        value = value.strip().lower()
        if value in {DEEPY_MULTI_SESSION_DISABLED, DEEPY_MULTI_SESSION_SELECTABLE, DEEPY_MULTI_SESSION_DEDICATED}:
            return value
        value = value in {"1", "true", "on", "yes"}
    return DEEPY_MULTI_SESSION_SELECTABLE if value else DEEPY_MULTI_SESSION_DISABLED


def estimate_deepy_kv_cache_mb(enhancer_enabled: Any, context_tokens: Any, kv_cache_quantization: Any = "") -> tuple[str | None, int | None]:
    try:
        enhancer_no = int(enhancer_enabled or 0)
    except Exception:
        enhancer_no = 0
    spec = _DEEPY_QWEN_KV_CACHE_SPECS.get(enhancer_no)
    if spec is None:
        return None, None
    tokens = normalize_deepy_context_tokens(context_tokens)
    block_size = max(1, int(spec.get("kvcache_block_size", 256) or 256))
    num_blocks = (tokens + block_size - 1) // block_size
    layers, heads, head_dim = int(spec["num_kv_cache_layers"]), int(spec["num_key_value_heads"]), int(spec["head_dim"])
    mode = resolve_deepy_kv_cache_quantization(kv_cache_quantization)[0]
    if mode:
        from shared.llm_engines.lmgp.layers.kv_formats import parse_layout
        layout = parse_layout(mode)
        total_bytes = num_blocks * block_size * layout.body_bytes_per_token(layers, heads, head_dim) + layout.overlay_bytes(layers, heads, head_dim, block_size, 1)
    else:
        total_bytes = 2 * layers * num_blocks * block_size * heads * head_dim * int(spec["dtype_bytes"])
    return _DEEPY_QWEN_VARIANT_LABELS.get(enhancer_no, "Qwen3.5"), int(round(total_bytes / (1024 * 1024)))


def format_deepy_context_tokens_label(enhancer_enabled: Any, context_tokens: Any, kv_cache_quantization: Any = "") -> str:
    requested_quantization = normalize_deepy_kv_cache_quantization(kv_cache_quantization)
    quantization = resolve_deepy_kv_cache_quantization(requested_quantization)[0]
    variant_label, cache_mb = estimate_deepy_kv_cache_mb(enhancer_enabled, context_tokens, requested_quantization)
    if variant_label is None or cache_mb is None:
        return "Context Window Tokens (KV cache: N/A)"
    auto_label = " Auto" if requested_quantization == DEEPY_KV_CACHE_QUANTIZATION_AUTO else ""
    return f"Context Window Tokens (KV cache ~{cache_mb} MB {describe_deepy_kv_cache_quantization(quantization)}{auto_label} on {variant_label})"


def normalize_deepy_runtime_config(server_config: dict[str, Any] | None) -> dict[str, Any]:
    from shared.remote_llm.config import is_remote_engine, local_enhancer_id, resolve_role_engine

    runtime_config = {**get_deepy_default_runtime_config(), **(server_config or {})}
    runtime_config[DEEPY_ENABLED_KEY] = normalize_deepy_enabled(runtime_config.get(DEEPY_ENABLED_KEY, 0))
    runtime_config[DEEPY_VOICE_LANGUAGE_KEY] = normalize_deepy_voice_language(runtime_config.get(DEEPY_VOICE_LANGUAGE_KEY, "auto"))
    runtime_config[DEEPY_TYPE_KEY] = normalize_deepy_type(runtime_config.get(DEEPY_TYPE_KEY, DEEPY_TYPE_DEFAULT))
    runtime_config[DEEPY_TOOL_GEN_IMAGE_KEY] = normalize_deepy_tool_gen_image(runtime_config.get(DEEPY_TOOL_GEN_IMAGE_KEY, DEEPY_DEFAULT_GEN_IMAGE))
    runtime_config[DEEPY_TOOL_EDIT_IMAGE_KEY] = normalize_deepy_tool_edit_image(runtime_config.get(DEEPY_TOOL_EDIT_IMAGE_KEY, DEEPY_DEFAULT_EDIT_IMAGE))
    runtime_config[DEEPY_TOOL_GEN_VIDEO_KEY] = normalize_deepy_tool_gen_video(runtime_config.get(DEEPY_TOOL_GEN_VIDEO_KEY, DEEPY_DEFAULT_GEN_VIDEO))
    runtime_config[DEEPY_TOOL_GEN_VIDEO_WITH_SPEECH_KEY] = normalize_deepy_tool_gen_video_with_speech(runtime_config.get(DEEPY_TOOL_GEN_VIDEO_WITH_SPEECH_KEY, DEEPY_DEFAULT_GEN_VIDEO_WITH_SPEECH))
    runtime_config[DEEPY_TOOL_GEN_SONG_KEY] = normalize_deepy_tool_gen_song(runtime_config.get(DEEPY_TOOL_GEN_SONG_KEY, DEEPY_DEFAULT_GEN_SONG))
    runtime_config[DEEPY_TOOL_GEN_VIDEO_WITH_REFS_KEY] = normalize_deepy_tool_gen_video_with_refs(runtime_config.get(DEEPY_TOOL_GEN_VIDEO_WITH_REFS_KEY, DEEPY_DEFAULT_GEN_VIDEO_WITH_REFS))
    runtime_config[DEEPY_TOOL_GEN_SPEECH_FROM_DESCRIPTION_KEY] = normalize_deepy_tool_gen_speech_from_description(runtime_config.get(DEEPY_TOOL_GEN_SPEECH_FROM_DESCRIPTION_KEY, DEEPY_DEFAULT_GEN_SPEECH_FROM_DESCRIPTION))
    runtime_config[DEEPY_TOOL_GEN_SPEECH_FROM_SAMPLE_KEY] = normalize_deepy_tool_gen_speech_from_sample(runtime_config.get(DEEPY_TOOL_GEN_SPEECH_FROM_SAMPLE_KEY, DEEPY_DEFAULT_GEN_SPEECH_FROM_SAMPLE))
    runtime_config[DEEPY_TOOL_GEN_WORLD_KEY] = normalize_deepy_tool_gen_world(runtime_config.get(DEEPY_TOOL_GEN_WORLD_KEY, DEEPY_DEFAULT_GEN_WORLD))
    runtime_config[DEEPY_TOOL_GEN_3D_KEY] = normalize_deepy_tool_gen_3d(runtime_config.get(DEEPY_TOOL_GEN_3D_KEY, DEEPY_DEFAULT_GEN_3D))
    runtime_config[DEEPY_TOOL_TEXTURE_3D_KEY] = normalize_deepy_tool_texture_3d(runtime_config.get(DEEPY_TOOL_TEXTURE_3D_KEY, DEEPY_DEFAULT_TEXTURE_3D))
    runtime_config[DEEPY_AUTO_CANCEL_QUEUE_TASKS_KEY] = normalize_deepy_auto_cancel_queue_tasks(runtime_config.get(DEEPY_AUTO_CANCEL_QUEUE_TASKS_KEY, DEEPY_AUTO_CANCEL_QUEUE_TASKS_DEFAULT))
    runtime_config[DEEPY_SEPARATE_REQUESTS_WITH_EMPTY_LINE_KEY] = normalize_deepy_separate_requests_with_empty_line(runtime_config.get(DEEPY_SEPARATE_REQUESTS_WITH_EMPTY_LINE_KEY, DEEPY_SEPARATE_REQUESTS_WITH_EMPTY_LINE_DEFAULT))
    runtime_config[DEEPY_SESSION_RESET_MODE_KEY] = normalize_deepy_session_reset_mode(runtime_config.get(DEEPY_SESSION_RESET_MODE_KEY, DEEPY_SESSION_RESET_MODE_DEFAULT))
    runtime_config[DEEPY_SESSION_GALLERY_MEDIA_MODE_KEY] = normalize_deepy_session_gallery_media_mode(runtime_config.get(DEEPY_SESSION_GALLERY_MEDIA_MODE_KEY, DEEPY_SESSION_GALLERY_MEDIA_MODE_DEFAULT))
    runtime_config[DEEPY_MULTI_SESSION_KEY] = normalize_deepy_session_mode(runtime_config.get(DEEPY_MULTI_SESSION_KEY, DEEPY_MULTI_SESSION_DEFAULT))
    engine = resolve_role_engine(runtime_config, "deepy")
    remote = is_remote_engine(engine)
    qwen_local = not remote and local_enhancer_id(engine) in DEEPY_QWEN_ENHANCER_IDS
    if qwen_local:
        runtime_config[DEEPY_REPETITION_PENALTY_KEY] = normalize_deepy_repetition_penalty(runtime_config.get(DEEPY_REPETITION_PENALTY_KEY, DEEPY_REPETITION_PENALTY_DEFAULT))
        runtime_config[DEEPY_KV_CACHE_QUANTIZATION_KEY] = normalize_deepy_kv_cache_quantization(runtime_config.get(DEEPY_KV_CACHE_QUANTIZATION_KEY, DEEPY_KV_CACHE_QUANTIZATION_DEFAULT))
    if runtime_config[DEEPY_ENABLED_KEY] and (qwen_local or remote):
        runtime_config[DEEPY_ZERO_CUSTOM_SYSTEM_PROMPT_KEY] = normalize_deepy_custom_system_prompt(runtime_config.get(DEEPY_ZERO_CUSTOM_SYSTEM_PROMPT_KEY, ""))
        runtime_config[DEEPY_PRIME_CUSTOM_SYSTEM_PROMPT_KEY] = normalize_deepy_prime_guidance(runtime_config.get(DEEPY_PRIME_CUSTOM_SYSTEM_PROMPT_KEY, DEEPY_PRIME_GUIDANCE_DEFAULT))
        runtime_config[DEEPY_PRIME_MCP_SERVERS_KEY] = normalize_deepy_prime_mcp_servers(runtime_config.get(DEEPY_PRIME_MCP_SERVERS_KEY, {}))
        runtime_config[DEEPY_MCP_AUTO_DISCOVER_PATHS_KEY] = normalize_deepy_mcp_auto_discover_paths(runtime_config.get(DEEPY_MCP_AUTO_DISCOVER_PATHS_KEY, DEEPY_MCP_AUTO_DISCOVER_PATHS_DEFAULT))
        runtime_config[DEEPY_ALLOW_READ_FILE_SYSTEM_KEY] = normalize_deepy_file_system_access(runtime_config.get(DEEPY_ALLOW_READ_FILE_SYSTEM_KEY, DEEPY_FILE_SYSTEM_ACCESS_DEFAULT))
        if not remote:
            runtime_config[DEEPY_VRAM_MODE_KEY] = normalize_deepy_vram_mode(runtime_config.get(DEEPY_VRAM_MODE_KEY, DEEPY_VRAM_MODE_UNLOAD))
        if qwen_local:
            runtime_config[DEEPY_CONTEXT_TOKENS_KEY] = normalize_deepy_context_tokens(runtime_config.get(DEEPY_CONTEXT_TOKENS_KEY, DEEPY_CONTEXT_TOKENS_DEFAULT))
            runtime_config[DEEPY_COMPACTION_TYPE_KEY] = normalize_deepy_compaction_type(runtime_config.get(DEEPY_COMPACTION_TYPE_KEY, DEEPY_COMPACTION_TYPE_DEFAULT))
            runtime_config[DEEPY_COMPACTION_THINKING_KEY] = normalize_deepy_compaction_thinking(runtime_config.get(DEEPY_COMPACTION_THINKING_KEY, DEEPY_COMPACTION_THINKING_DEFAULT))
        if runtime_config[DEEPY_ALLOW_READ_FILE_SYSTEM_KEY] != DEEPY_FILE_SYSTEM_ACCESS_DISABLED:
            runtime_config[DEEPY_FILE_SYSTEM_PATHS_KEY] = normalize_deepy_file_system_paths(runtime_config.get(DEEPY_FILE_SYSTEM_PATHS_KEY, DEEPY_FILE_SYSTEM_PATHS_DEFAULT))
            runtime_config[DEEPY_READ_EVERYWHERE_KEY] = normalize_deepy_read_everywhere(runtime_config.get(DEEPY_READ_EVERYWHERE_KEY, DEEPY_READ_EVERYWHERE_DEFAULT))
    return runtime_config


def get_deepy_default_runtime_config() -> dict[str, Any]:
    return {
        DEEPY_ENABLED_KEY: 0,
        DEEPY_TYPE_KEY: DEEPY_TYPE_DEFAULT,
        DEEPY_VRAM_MODE_KEY: DEEPY_VRAM_MODE_UNLOAD,
        DEEPY_VOICE_LANGUAGE_KEY: "auto",
        DEEPY_TOOL_EDIT_IMAGE_KEY: DEEPY_DEFAULT_EDIT_IMAGE,
        DEEPY_TOOL_GEN_IMAGE_KEY: DEEPY_DEFAULT_GEN_IMAGE,
        DEEPY_TOOL_GEN_VIDEO_KEY: DEEPY_DEFAULT_GEN_VIDEO,
        DEEPY_TOOL_GEN_VIDEO_WITH_SPEECH_KEY: DEEPY_DEFAULT_GEN_VIDEO_WITH_SPEECH,
        DEEPY_TOOL_GEN_SONG_KEY: DEEPY_DEFAULT_GEN_SONG,
        DEEPY_TOOL_GEN_VIDEO_WITH_REFS_KEY: DEEPY_DEFAULT_GEN_VIDEO_WITH_REFS,
        DEEPY_TOOL_GEN_SPEECH_FROM_DESCRIPTION_KEY: DEEPY_DEFAULT_GEN_SPEECH_FROM_DESCRIPTION,
        DEEPY_TOOL_GEN_SPEECH_FROM_SAMPLE_KEY: DEEPY_DEFAULT_GEN_SPEECH_FROM_SAMPLE,
        DEEPY_TOOL_GEN_WORLD_KEY: DEEPY_DEFAULT_GEN_WORLD,
        DEEPY_TOOL_GEN_3D_KEY: DEEPY_DEFAULT_GEN_3D,
        DEEPY_TOOL_TEXTURE_3D_KEY: DEEPY_DEFAULT_TEXTURE_3D,
        DEEPY_CONTEXT_TOKENS_KEY: DEEPY_CONTEXT_TOKENS_DEFAULT,
        DEEPY_KV_CACHE_QUANTIZATION_KEY: DEEPY_KV_CACHE_QUANTIZATION_DEFAULT,
        DEEPY_COMPACTION_TYPE_KEY: DEEPY_COMPACTION_TYPE_DEFAULT,
        DEEPY_COMPACTION_THINKING_KEY: DEEPY_COMPACTION_THINKING_DEFAULT,
        DEEPY_REPETITION_PENALTY_KEY: DEEPY_REPETITION_PENALTY_DEFAULT,
        DEEPY_ZERO_CUSTOM_SYSTEM_PROMPT_KEY: "",
        DEEPY_PRIME_CUSTOM_SYSTEM_PROMPT_KEY: DEEPY_PRIME_GUIDANCE_DEFAULT,
        DEEPY_MODEL_SPEED_KEY: "fast",
        DEEPY_MODEL_SIZE_KEY: "smaller",
        DEEPY_PRIME_MCP_SERVERS_KEY: {},
        DEEPY_MCP_AUTO_DISCOVER_PATHS_KEY: DEEPY_MCP_AUTO_DISCOVER_PATHS_DEFAULT,
        DEEPY_ALLOW_READ_FILE_SYSTEM_KEY: DEEPY_ALLOW_READ_FILE_SYSTEM_DEFAULT,
        DEEPY_FILE_SYSTEM_PATHS_KEY: list(DEEPY_FILE_SYSTEM_PATHS_DEFAULT),
        DEEPY_READ_EVERYWHERE_KEY: DEEPY_READ_EVERYWHERE_DEFAULT,
        DEEPY_PYTHON_SANDBOX_KEY: DEEPY_PYTHON_SANDBOX_DEFAULT,
        DEEPY_AUTO_CANCEL_QUEUE_TASKS_KEY: DEEPY_AUTO_CANCEL_QUEUE_TASKS_DEFAULT,
        DEEPY_SEPARATE_REQUESTS_WITH_EMPTY_LINE_KEY: DEEPY_SEPARATE_REQUESTS_WITH_EMPTY_LINE_DEFAULT,
        DEEPY_SESSION_RESET_MODE_KEY: DEEPY_SESSION_RESET_MODE_DEFAULT,
        DEEPY_SESSION_GALLERY_MEDIA_MODE_KEY: DEEPY_SESSION_GALLERY_MEDIA_MODE_DEFAULT,
        DEEPY_MULTI_SESSION_KEY: DEEPY_MULTI_SESSION_DEFAULT,
    }


def set_deepy_runtime_config(server_config: dict[str, Any] | None, server_config_filename: str = "") -> dict[str, Any]:
    global _DEEPY_RUNTIME_CONFIG, _DEEPY_RUNTIME_CONFIG_FILENAME
    with config_lock:
        normalized = normalize_deepy_runtime_config(server_config)
        if isinstance(server_config, dict):
            server_config.update(normalized)
        _DEEPY_RUNTIME_CONFIG = normalized
        _DEEPY_RUNTIME_CONFIG_FILENAME = str(server_config_filename or "").strip()
        return normalized


def get_deepy_runtime_config() -> dict[str, Any]:
    global _DEEPY_RUNTIME_CONFIG
    if isinstance(_DEEPY_RUNTIME_CONFIG, dict) and len(_DEEPY_RUNTIME_CONFIG) > 0:
        return _DEEPY_RUNTIME_CONFIG
    candidate_paths = []
    if len(_DEEPY_RUNTIME_CONFIG_FILENAME) > 0:
        candidate_paths.append(Path(_DEEPY_RUNTIME_CONFIG_FILENAME))
    candidate_paths.append(Path.cwd() / DEEPY_CONFIG_FILENAME)
    for path in candidate_paths:
        if not path.is_file():
            continue
        try:
            loaded = read_config(path)
        except Exception:
            continue
        if isinstance(loaded, dict):
            return set_deepy_runtime_config(loaded, str(path))
    _DEEPY_RUNTIME_CONFIG = {}
    return _DEEPY_RUNTIME_CONFIG


def get_deepy_config_value(key: str, default: Any = None) -> Any:
    return get_deepy_runtime_config().get(key, default)


def configured_secret(config: dict[str, Any], name: str) -> str:
    """The secret (SECRET_CONFIG_KEYS name) a configuration holds: plaintext, or read from the OS credential manager through its credential id (raises SecureStorageError when that store is unavailable)."""
    value_key, _secure_key, id_key = SECRET_CONFIG_KEYS[name]
    plaintext = str(config.get(value_key, "") or "").strip()
    credential_id = str(config.get(id_key, "") or "").strip()
    if plaintext or not credential_id:
        return plaintext
    from shared.utils.credentials import load_secret

    return load_secret(credential_id).strip()


def prepare_secret_update(config: dict[str, Any], name: str, value: Any, secure: Any) -> dict[str, Any]:
    """Configuration keys for a saved secret: in the OS credential manager when secure (plaintext blanked), else in wgp_config.json; a cleared value or a move to plaintext removes the stored credential."""
    from uuid import uuid4

    from shared.utils.credentials import availability_error, delete_secret, save_secret

    value_key, secure_key, id_key = SECRET_CONFIG_KEYS[name]
    value, secure = str(value or "").strip(), bool(secure)
    credential_id = str(config.get(id_key, "") or "").strip()
    if secure and value:
        credential_id = credential_id or uuid4().hex
        save_secret(credential_id, value)
    elif credential_id:
        if not availability_error():
            delete_secret(credential_id)
        credential_id = ""
    return {value_key: "" if secure else value, secure_key: secure, id_key: credential_id}


def configured_civitai_api_key(config: dict[str, Any]) -> str:
    return configured_secret(config, "civitai")


def prepare_civitai_api_key_update(config: dict[str, Any], key: Any, secure: Any) -> dict[str, Any]:
    return prepare_secret_update(config, "civitai", key, secure)


_SECRET_WARNED: set[str] = set()


def _runtime_secret(name: str, label: str) -> str:
    try:
        return configured_secret(get_deepy_runtime_config(), name)
    except Exception as error:  # SecureStorageError: the secret stays unusable until the credential service is back
        if name not in _SECRET_WARNED:
            _SECRET_WARNED.add(name)
            from tqdm import tqdm
            tqdm.write(f"[Deepy] The {label} could not be read: {error}")
        return ""


def get_civitai_api_key() -> str:
    """The Civitai API key for runtime use: CIVITAI_API_KEY in the environment, else the configured one ('' with one console warning when the credential store is unavailable)."""
    return os.environ.get("CIVITAI_API_KEY", "").strip() or _runtime_secret("civitai", "Civitai API key")


def get_huggingface_token() -> str:
    """The Hugging Face token for Deepy's LoRA hub: HF_TOKEN in the environment, else the configured one, else the login cached by the Hugging Face CLI; '' when none (public repositories need none)."""
    token = os.environ.get("HF_TOKEN", "").strip() or os.environ.get("HUGGING_FACE_HUB_TOKEN", "").strip() or _runtime_secret("huggingface", "Hugging Face token")
    if token:
        return token
    try:
        from huggingface_hub import get_token
        return (get_token() or "").strip()
    except Exception:
        return ""


def deepy_requirement_met(server_config: dict[str, Any] | None) -> bool:
    return len(deepy_requirement_error(server_config)) == 0


def deepy_requirement_error(server_config: dict[str, Any] | None) -> str:
    runtime_config = server_config or {}
    from shared.remote_llm.config import is_remote_engine, local_enhancer_id, resolve_role_engine

    deepy_engine = resolve_role_engine(runtime_config, "deepy")
    if is_remote_engine(deepy_engine):
        if normalize_deepy_type(runtime_config.get(DEEPY_TYPE_KEY, DEEPY_TYPE_DEFAULT)) != DEEPY_TYPE_PRIME:
            return "External LLM engines require Deepy Prime because only Deepy Prime exposes WanGP's MCP tools."
        return ""
    try:
        enhancer_enabled = local_enhancer_id(deepy_engine, runtime_config.get("enhancer_enabled", 0))
    except Exception:
        enhancer_enabled = 0
    if enhancer_enabled not in DEEPY_QWEN_ENHANCER_IDS:
        return "Deepy requires Prompt Enhancer to be set to a Qwen3.5VL Abliterated or Qwen3.8VL Uncensored mode in the Extensions tab."
    try:
        validate_deepy_version_config(runtime_config.get(DEEPY_TYPE_KEY, DEEPY_TYPE_DEFAULT), runtime_config.get(DEEPY_COMPACTION_TYPE_KEY, DEEPY_COMPACTION_TYPE_DEFAULT), runtime_config.get(DEEPY_CONTEXT_TOKENS_KEY, DEEPY_CONTEXT_TOKENS_DEFAULT), enhancer_enabled, compaction_thinking=runtime_config.get(DEEPY_COMPACTION_THINKING_KEY, DEEPY_COMPACTION_THINKING_DEFAULT))
    except ValueError as exc:
        return str(exc)
    return ""


def deepy_enabled(server_config: dict[str, Any] | None) -> bool:
    runtime_config = server_config or {}
    return normalize_deepy_enabled(runtime_config.get(DEEPY_ENABLED_KEY, 0)) == 1


def deepy_available(server_config: dict[str, Any] | None) -> bool:
    return deepy_enabled(server_config) and deepy_requirement_met(server_config)


def deepy_requirement_message(server_config: dict[str, Any] | None) -> str:
    runtime_config = server_config or {}
    requirement_error = deepy_requirement_error(runtime_config)
    if len(requirement_error) == 0:
        from shared.remote_llm.config import ENGINE_LABELS, is_remote_engine, resolve_role_engine

        version = "Deepy Prime" if normalize_deepy_type(runtime_config.get(DEEPY_TYPE_KEY, DEEPY_TYPE_DEFAULT)) == DEEPY_TYPE_PRIME else "Deepy Zero"
        engine = resolve_role_engine(runtime_config, "deepy")
        detail = f" using {ENGINE_LABELS.get(engine, engine.title())} externally" if is_remote_engine(engine) else " with the current Prompt Enhancer and context configuration"
        return (
            "<div style='color:#1b6d44; font-weight:600;'>"
            f"{version} is available{detail}."
            "</div>"
        )
    return (
        "<div style='color:#8a4a14; font-weight:600;'>"
        f"{requirement_error}"
        "</div>"
    )
