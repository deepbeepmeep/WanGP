# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlparse


LLM_CONFIG_KEY = "llm_engines"

ENGINE_LOCAL_1 = "local_florence_llama32"
ENGINE_LOCAL_2 = "local_florence_llamajoy"
ENGINE_QWEN35_4B = "qwen35_4b"
ENGINE_QWEN35_9B = "qwen35_9b"
ENGINE_QWEN38_27B = "qwen38_27b"
ENGINE_QWEN38_9B = "qwen38_9b"
ENGINE_QWEN38_FLASH_NEXT = "qwen38_flash_next"
ENGINE_CODEX = "codex"
ENGINE_CLAUDE = "claude"
ENGINE_OPENCODE = "opencode"
ENGINE_SAME_AS_DEEPY = "same_as_deepy"
ENGINE_AUTO = "auto"
ENGINE_DISABLED = "disabled"

CODEX_DEFAULT_MODEL_CATALOG = [
    {"model": "gpt-5.6-sol", "display_name": "GPT-5.6-Sol", "is_default": True, "default_reasoning_effort": "low", "reasoning_efforts": ["low", "medium", "high", "xhigh", "max", "ultra"]},
    {"model": "gpt-5.6-terra", "display_name": "GPT-5.6-Terra", "is_default": False, "default_reasoning_effort": "medium", "reasoning_efforts": ["low", "medium", "high", "xhigh", "max", "ultra"]},
    {"model": "gpt-5.6-luna", "display_name": "GPT-5.6-Luna", "is_default": False, "default_reasoning_effort": "medium", "reasoning_efforts": ["low", "medium", "high", "xhigh", "max"]},
    {"model": "gpt-5.5", "display_name": "GPT-5.5", "is_default": False, "default_reasoning_effort": "medium", "reasoning_efforts": ["low", "medium", "high", "xhigh"]},
    {"model": "gpt-5.4", "display_name": "GPT-5.4", "is_default": False, "default_reasoning_effort": "medium", "reasoning_efforts": ["low", "medium", "high", "xhigh"]},
    {"model": "gpt-5.4-mini", "display_name": "GPT-5.4-Mini", "is_default": False, "default_reasoning_effort": "medium", "reasoning_efforts": ["low", "medium", "high", "xhigh"]},
    {"model": "gpt-5.3-codex-spark", "display_name": "GPT-5.3-Codex-Spark", "is_default": False, "default_reasoning_effort": "high", "reasoning_efforts": ["low", "medium", "high", "xhigh"]},
]
CODEX_REASONING_EFFORT_LABELS = {"none": "None", "minimal": "Minimal", "low": "Low", "medium": "Medium", "high": "High", "xhigh": "Extra high", "max": "Max", "ultra": "Ultra"}

CLAUDE_DEFAULT_MODEL_CATALOG = [
    {"model": "best", "display_name": "Best available", "is_default": False, "default_reasoning_effort": "", "reasoning_efforts": ["low", "medium", "high", "xhigh", "max"]},
    {"model": "fable", "display_name": "Fable", "is_default": False, "default_reasoning_effort": "", "reasoning_efforts": ["low", "medium", "high", "xhigh", "max"]},
    {"model": "sonnet", "display_name": "Latest Sonnet", "is_default": False, "default_reasoning_effort": "", "reasoning_efforts": ["low", "medium", "high", "xhigh", "max"]},
    {"model": "opus", "display_name": "Latest Opus", "is_default": False, "default_reasoning_effort": "", "reasoning_efforts": ["low", "medium", "high", "xhigh", "max"]},
    {"model": "haiku", "display_name": "Latest Haiku", "is_default": False, "default_reasoning_effort": "", "reasoning_efforts": ["low", "medium", "high", "xhigh", "max"]},
    {"model": "sonnet[1m]", "display_name": "Sonnet (1M context)", "is_default": False, "default_reasoning_effort": "", "reasoning_efforts": ["low", "medium", "high", "xhigh", "max"]},
    {"model": "opus[1m]", "display_name": "Opus (1M context)", "is_default": False, "default_reasoning_effort": "", "reasoning_efforts": ["low", "medium", "high", "xhigh", "max"]},
    {"model": "opusplan", "display_name": "Opus for planning, Sonnet for execution", "is_default": False, "default_reasoning_effort": "", "reasoning_efforts": ["low", "medium", "high", "xhigh", "max"]},
]
CLAUDE_REASONING_EFFORT_LABELS = {"low": "Low", "medium": "Medium", "high": "High", "xhigh": "Extra high", "max": "Max"}
OPENCODE_REASONING_EFFORT_LABELS = {"none": "None", "minimal": "Minimal", "low": "Low", "medium": "Medium", "high": "High", "xhigh": "Extra high", "max": "Max"}

CODEX_AUTOMATIC_MODEL_LABEL = "Automatic (Codex recommended default)"
CLAUDE_AUTOMATIC_MODEL_LABEL = "Automatic (Claude account default)"
OPENCODE_AUTOMATIC_PROVIDER_LABEL = "Automatic (OpenCode default)"
OPENCODE_AUTOMATIC_MODEL_LABEL = "Automatic (provider default)"
AUTOMATIC_REASONING_EFFORT_LABEL = "Automatic (model default)"

REMOTE_ENGINES = frozenset({ENGINE_CODEX, ENGINE_CLAUDE, ENGINE_OPENCODE})
ENGINE_LABELS = {ENGINE_CODEX: "Codex", ENGINE_CLAUDE: "Claude Code", ENGINE_OPENCODE: "OpenCode"}
LOCAL_ENGINE_TO_ENHANCER_ID = {
    ENGINE_LOCAL_1: 1,
    ENGINE_LOCAL_2: 2,
    ENGINE_QWEN35_4B: 3,
    ENGINE_QWEN35_9B: 4,
    ENGINE_QWEN38_27B: 5,
    ENGINE_QWEN38_9B: 6,
    ENGINE_QWEN38_FLASH_NEXT: 7,
}
ENHANCER_ID_TO_LOCAL_ENGINE = {value: key for key, value in LOCAL_ENGINE_TO_ENHANCER_ID.items()}
QWEN38_FLASH_NEXT_VISIBLE = False  # Qwen3.8 Flash-Next among the Prompt Enhancer / Deepy engines and models; False hides it

LLM_ENGINE_CHOICES = [
    ("Florence 2 + Llama 3.2 3B (local)", ENGINE_LOCAL_1),
    ("Florence 2 + Llama Joy 8B (local)", ENGINE_LOCAL_2),
    ("Qwen3.5 VL Abliterated 4B (local)", ENGINE_QWEN35_4B),
    ("Qwen3.5 VL Abliterated 9B (local)", ENGINE_QWEN35_9B),
    ("Qwen3.8 VL Uncensored 27B (local)", ENGINE_QWEN38_27B),
    *([("Qwen3.8 VL Flash-Next 125B (local, 80 GB download, 64 GB of RAM)", ENGINE_QWEN38_FLASH_NEXT)] if QWEN38_FLASH_NEXT_VISIBLE else []),
    ("Codex (external)", ENGINE_CODEX),
    ("Claude Code (external)", ENGINE_CLAUDE),
    ("OpenCode / universal providers (external)", ENGINE_OPENCODE),
]
DEEPY_ENGINE_CHOICES = LLM_ENGINE_CHOICES
PROMPT_ENGINE_CHOICES = [("Same as Deepy", ENGINE_SAME_AS_DEEPY), ("Codex", ENGINE_CODEX), ("Claude Code", ENGINE_CLAUDE), ("OpenCode", ENGINE_OPENCODE)]
VISUAL_INSPECTOR_CHOICES = [("Auto", ENGINE_AUTO), ("Same as Deepy", ENGINE_SAME_AS_DEEPY), ("Disabled", ENGINE_DISABLED)]


@dataclass(frozen=True, slots=True)
class EngineCapabilities:
    execution_location: str
    runtime_owner: str
    requires_internet: bool
    data_leaves_machine: bool
    uses_local_gpu: bool
    supports_streaming: bool
    supports_sessions: bool
    supports_interrupt: bool
    supports_steering: bool
    supports_mcp: bool
    supports_images: bool
    manages_compaction: bool


REMOTE_CAPABILITIES = EngineCapabilities(
    execution_location="remote",
    runtime_owner="external",
    requires_internet=True,
    data_leaves_machine=True,
    uses_local_gpu=False,
    supports_streaming=True,
    supports_sessions=True,
    supports_interrupt=True,
    supports_steering=True,
    supports_mcp=True,
    supports_images=True,
    manages_compaction=True,
)
LOCAL_CAPABILITIES = EngineCapabilities(
    execution_location="wangp_local",
    runtime_owner="wangp",
    requires_internet=False,
    data_leaves_machine=False,
    uses_local_gpu=True,
    supports_streaming=True,
    supports_sessions=True,
    supports_interrupt=True,
    supports_steering=True,
    supports_mcp=False,
    supports_images=True,
    manages_compaction=False,
)

_DEFAULT_CONFIG = {
    "deepy": ENGINE_QWEN35_4B,
    "prompt_enhancer": ENGINE_SAME_AS_DEEPY,
    "visual_inspector": ENGINE_AUTO,
    "profiles": {
        ENGINE_CODEX: {"executable": "codex", "model": "", "reasoning_effort": "", "model_catalog": CODEX_DEFAULT_MODEL_CATALOG},
        ENGINE_CLAUDE: {"executable": "claude", "model": "", "reasoning_effort": "", "model_catalog": CLAUDE_DEFAULT_MODEL_CATALOG},
        ENGINE_OPENCODE: {"executable": "opencode", "base_url": "http://127.0.0.1:4096", "provider": "", "model": "", "reasoning_effort": "", "config": "", "model_catalog": []},
    },
}


def is_remote_engine(engine: Any) -> bool:
    return str(engine or "").strip().lower() in REMOTE_ENGINES


def local_enhancer_id(engine: Any, fallback: int = 3) -> int:
    return LOCAL_ENGINE_TO_ENHANCER_ID.get(str(engine or "").strip().lower(), int(fallback))


def engine_from_legacy_enhancer(value: Any) -> str:
    try:
        enhancer_id = int(value or 0)
    except (TypeError, ValueError):
        enhancer_id = 0
    return ENHANCER_ID_TO_LOCAL_ENGINE.get(enhancer_id, ENGINE_QWEN35_4B)


def _normalize_engine(value: Any, default: str, *, allow_same: bool = False, allow_auto: bool = False, allow_disabled: bool = False) -> str:
    engine = str(value or "").strip().lower()
    allowed = set(LOCAL_ENGINE_TO_ENHANCER_ID) | set(REMOTE_ENGINES)
    if allow_same:
        allowed.add(ENGINE_SAME_AS_DEEPY)
    if allow_auto:
        allowed.add(ENGINE_AUTO)
    if allow_disabled:
        allowed.add(ENGINE_DISABLED)
    return engine if engine in allowed else default


def _profile_text(profile: dict[str, Any], key: str, default: str = "") -> str:
    return str(profile.get(key, default) or default).strip()


def normalize_codex_model_catalog(value: Any) -> list[dict[str, Any]]:
    source = value if isinstance(value, list) else CODEX_DEFAULT_MODEL_CATALOG
    catalog = []
    seen = set()
    for entry in source:
        item = entry if isinstance(entry, dict) else {"model": entry}
        model = str(item.get("model", item.get("id", "")) or "").strip()
        if not model or model in seen:
            continue
        seen.add(model)
        effort_source = item.get("reasoning_efforts", item.get("supportedReasoningEfforts", []))
        effort_source = effort_source if isinstance(effort_source, list) else []
        efforts = []
        for effort_entry in effort_source:
            effort = str(effort_entry.get("reasoningEffort", "") if isinstance(effort_entry, dict) else effort_entry or "").strip()
            if effort and effort not in efforts:
                efforts.append(effort)
        default_effort = str(item.get("default_reasoning_effort", item.get("defaultReasoningEffort", "")) or "").strip()
        catalog.append({"model": model, "display_name": str(item.get("display_name", item.get("displayName", model)) or model).strip(), "is_default": bool(item.get("is_default", item.get("isDefault", False))), "default_reasoning_effort": default_effort, "reasoning_efforts": efforts})
    return catalog or deepcopy(CODEX_DEFAULT_MODEL_CATALOG)


def _normalize_automatic_choice(value: Any, label: str) -> str:
    selected = str(value or "").strip()
    if selected in {label, f"{label} (currently selected)"}:
        return ""
    return selected


def normalize_codex_model_selection(value: Any) -> str:
    return _normalize_automatic_choice(value, CODEX_AUTOMATIC_MODEL_LABEL)


def normalize_claude_model_selection(value: Any) -> str:
    return _normalize_automatic_choice(value, CLAUDE_AUTOMATIC_MODEL_LABEL)


def normalize_reasoning_effort_selection(value: Any) -> str:
    return _normalize_automatic_choice(value, AUTOMATIC_REASONING_EFFORT_LABEL)


def normalize_opencode_provider_selection(value: Any) -> str:
    return _normalize_automatic_choice(value, OPENCODE_AUTOMATIC_PROVIDER_LABEL)


def normalize_opencode_model_selection(value: Any) -> str:
    return _normalize_automatic_choice(value, OPENCODE_AUTOMATIC_MODEL_LABEL)


def codex_model_choices(catalog: Any, selected_model: Any = "") -> list[tuple[str, str]]:
    selected = normalize_codex_model_selection(selected_model)
    choices = [(CODEX_AUTOMATIC_MODEL_LABEL, "")]
    known = set()
    for entry in normalize_codex_model_catalog(catalog):
        model = entry["model"]
        label = entry["display_name"]
        if entry["is_default"]:
            label = f"{label} (recommended)"
        choices.append((label, model))
        known.add(model)
    if selected and selected not in known:
        choices.append((f"{selected} (currently selected)", selected))
    return choices


def codex_reasoning_effort_choices(catalog: Any, model: Any = "", selected_effort: Any = "") -> list[tuple[str, str]]:
    normalized = normalize_codex_model_catalog(catalog)
    selected_model = str(model or "").strip()
    selected = normalize_reasoning_effort_selection(selected_effort)
    entry = next((item for item in normalized if item["model"] == selected_model), None)
    if entry is None:
        entry = next((item for item in normalized if item["is_default"]), normalized[0])
    default_effort = entry["default_reasoning_effort"]
    choices = [(AUTOMATIC_REASONING_EFFORT_LABEL, "")]
    for effort in entry["reasoning_efforts"]:
        label = CODEX_REASONING_EFFORT_LABELS.get(effort, effort)
        choices.append((f"{label} (model default)" if effort == default_effort else label, effort))
    if selected and selected not in entry["reasoning_efforts"]:
        choices.append((f"{CODEX_REASONING_EFFORT_LABELS.get(selected, selected)} (currently selected)", selected))
    return choices


def normalize_claude_model_catalog(value: Any) -> list[dict[str, Any]]:
    source = value if isinstance(value, list) else CLAUDE_DEFAULT_MODEL_CATALOG
    catalog = []
    seen = set()
    for entry in source:
        item = entry if isinstance(entry, dict) else {"model": entry}
        model = str(item.get("model", item.get("value", item.get("id", ""))) or "").strip()
        if not model or model == "default" or model in seen:
            continue
        seen.add(model)
        effort_source = item.get("reasoning_efforts", item.get("supportedEffortLevels", []))
        effort_source = effort_source if isinstance(effort_source, list) else []
        efforts = []
        for effort_entry in effort_source:
            effort = str(effort_entry.get("value", effort_entry.get("effort", "")) if isinstance(effort_entry, dict) else effort_entry or "").strip()
            if effort and effort not in efforts:
                efforts.append(effort)
        catalog.append({"model": model, "display_name": str(item.get("display_name", item.get("displayName", model)) or model).strip(), "is_default": bool(item.get("is_default", item.get("isDefault", False))), "default_reasoning_effort": str(item.get("default_reasoning_effort", item.get("defaultEffort", "")) or "").strip(), "reasoning_efforts": efforts})
    return catalog or deepcopy(CLAUDE_DEFAULT_MODEL_CATALOG)


def claude_model_choices(catalog: Any, selected_model: Any = "") -> list[tuple[str, str]]:
    selected = normalize_claude_model_selection(selected_model)
    choices = [(CLAUDE_AUTOMATIC_MODEL_LABEL, "")]
    known = set()
    for entry in normalize_claude_model_catalog(catalog):
        model = entry["model"]
        choices.append((entry["display_name"], model))
        known.add(model)
    if selected and selected not in known:
        choices.append((f"{selected} (currently selected)", selected))
    return choices


def claude_reasoning_effort_choices(catalog: Any, model: Any = "", selected_effort: Any = "") -> list[tuple[str, str]]:
    normalized = normalize_claude_model_catalog(catalog)
    selected_model = str(model or "").strip()
    selected = normalize_reasoning_effort_selection(selected_effort)
    entry = next((item for item in normalized if item["model"] == selected_model), None)
    efforts = entry["reasoning_efforts"] if entry and entry["reasoning_efforts"] else ["low", "medium", "high", "xhigh", "max"]
    default_effort = entry["default_reasoning_effort"] if entry else ""
    choices = [(AUTOMATIC_REASONING_EFFORT_LABEL, "")]
    for effort in efforts:
        label = CLAUDE_REASONING_EFFORT_LABELS.get(effort, effort)
        choices.append((f"{label} (model default)" if effort == default_effort else label, effort))
    if selected and selected not in efforts:
        choices.append((f"{CLAUDE_REASONING_EFFORT_LABELS.get(selected, selected)} (currently selected)", selected))
    return choices


def normalize_opencode_model_catalog(value: Any) -> list[dict[str, Any]]:
    source = value if isinstance(value, list) else []
    catalog = []
    seen = set()
    for entry in source:
        item = entry if isinstance(entry, dict) else {}
        provider = str(item.get("provider", item.get("providerID", "")) or "").strip()
        model = str(item.get("model", item.get("id", "")) or "").strip()
        if not provider or not model or (provider, model) in seen:
            continue
        seen.add((provider, model))
        effort_source = item.get("reasoning_efforts", item.get("variants", []))
        if isinstance(effort_source, dict):
            effort_source = list(effort_source)
        effort_source = effort_source if isinstance(effort_source, list) else []
        efforts = []
        for effort_entry in effort_source:
            effort = str(effort_entry.get("value", effort_entry.get("name", "")) if isinstance(effort_entry, dict) else effort_entry or "").strip()
            if effort and effort not in efforts:
                efforts.append(effort)
        try:
            context_window = max(0, int(item.get("context_window", item.get("contextWindow", 0)) or 0))
        except (TypeError, ValueError):
            context_window = 0
        catalog.append({"provider": provider, "provider_name": str(item.get("provider_name", item.get("providerName", provider)) or provider).strip(), "model": model, "display_name": str(item.get("display_name", item.get("displayName", model)) or model).strip(), "is_default": bool(item.get("is_default", item.get("isDefault", False))), "context_window": context_window, "reasoning_efforts": efforts})
    return catalog


def opencode_provider_choices(catalog: Any, selected_provider: Any = "") -> list[tuple[str, str]]:
    selected = normalize_opencode_provider_selection(selected_provider)
    choices = [(OPENCODE_AUTOMATIC_PROVIDER_LABEL, "")]
    known = set()
    for entry in normalize_opencode_model_catalog(catalog):
        provider = entry["provider"]
        if provider not in known:
            choices.append((entry["provider_name"], provider))
            known.add(provider)
    if selected and selected not in known:
        choices.append((f"{selected} (currently selected)", selected))
    return choices


def opencode_model_choices(catalog: Any, provider: Any = "", selected_model: Any = "") -> list[tuple[str, str]]:
    selected_provider = normalize_opencode_provider_selection(provider)
    selected = normalize_opencode_model_selection(selected_model)
    choices = [(OPENCODE_AUTOMATIC_MODEL_LABEL, "")]
    known = set()
    for entry in normalize_opencode_model_catalog(catalog):
        if selected_provider and entry["provider"] != selected_provider:
            continue
        model = entry["model"]
        label = entry["display_name"] if selected_provider else f"{entry['provider_name']} / {entry['display_name']}"
        if entry["is_default"]:
            label = f"{label} (default)"
        choices.append((label, model))
        known.add(model)
    if selected and selected not in known:
        choices.append((f"{selected} (currently selected)", selected))
    return choices


def opencode_reasoning_effort_choices(catalog: Any, provider: Any = "", model: Any = "", selected_effort: Any = "") -> list[tuple[str, str]]:
    selected_provider = normalize_opencode_provider_selection(provider)
    selected_model = normalize_opencode_model_selection(model)
    selected = normalize_reasoning_effort_selection(selected_effort)
    entry = next((item for item in normalize_opencode_model_catalog(catalog) if item["provider"] == selected_provider and item["model"] == selected_model), None)
    efforts = entry["reasoning_efforts"] if entry else []
    choices = [(AUTOMATIC_REASONING_EFFORT_LABEL, "")]
    choices.extend((OPENCODE_REASONING_EFFORT_LABELS.get(effort, effort), effort) for effort in efforts)
    if selected and selected not in efforts:
        choices.append((f"{OPENCODE_REASONING_EFFORT_LABELS.get(selected, selected)} (currently selected)", selected))
    return choices


def normalize_llm_config(server_config: dict[str, Any] | None) -> dict[str, Any]:
    server_config = server_config or {}
    source = server_config.get(LLM_CONFIG_KEY, {})
    source = source if isinstance(source, dict) else {}
    default_deepy = engine_from_legacy_enhancer(server_config.get("enhancer_enabled", 3))
    normalized = deepcopy(_DEFAULT_CONFIG)
    normalized["deepy"] = _normalize_engine(source.get("deepy"), default_deepy)
    normalized["prompt_enhancer"] = ENGINE_SAME_AS_DEEPY
    normalized["visual_inspector"] = ENGINE_AUTO
    source_profiles = source.get("profiles", {})
    source_profiles = source_profiles if isinstance(source_profiles, dict) else {}
    codex = source_profiles.get(ENGINE_CODEX, {})
    claude = source_profiles.get(ENGINE_CLAUDE, {})
    opencode = source_profiles.get(ENGINE_OPENCODE, {})
    codex = codex if isinstance(codex, dict) else {}
    claude = claude if isinstance(claude, dict) else {}
    opencode = opencode if isinstance(opencode, dict) else {}
    normalized["profiles"][ENGINE_CODEX] = {"executable": _profile_text(codex, "executable", "codex"), "model": normalize_codex_model_selection(_profile_text(codex, "model")), "reasoning_effort": normalize_reasoning_effort_selection(_profile_text(codex, "reasoning_effort")), "model_catalog": normalize_codex_model_catalog(codex.get("model_catalog"))}
    normalized["profiles"][ENGINE_CLAUDE] = {"executable": _profile_text(claude, "executable", "claude"), "model": normalize_claude_model_selection(_profile_text(claude, "model")), "reasoning_effort": normalize_reasoning_effort_selection(_profile_text(claude, "reasoning_effort")), "model_catalog": normalize_claude_model_catalog(claude.get("model_catalog"))}
    normalized["profiles"][ENGINE_OPENCODE] = {"executable": _profile_text(opencode, "executable", "opencode"), "base_url": _profile_text(opencode, "base_url", "http://127.0.0.1:4096").rstrip("/"), "provider": normalize_opencode_provider_selection(_profile_text(opencode, "provider")), "model": normalize_opencode_model_selection(_profile_text(opencode, "model")), "reasoning_effort": normalize_reasoning_effort_selection(_profile_text(opencode, "reasoning_effort")), "config": _profile_text(opencode, "config"), "model_catalog": normalize_opencode_model_catalog(opencode.get("model_catalog"))}
    return normalized


def resolve_role_engine(server_config: dict[str, Any] | None, role: str) -> str:
    config = normalize_llm_config(server_config)
    return str(config["deepy"])


def engine_capabilities(server_config: dict[str, Any] | None, engine: str) -> EngineCapabilities:
    if not is_remote_engine(engine):
        return LOCAL_CAPABILITIES
    capabilities = REMOTE_CAPABILITIES
    if engine == ENGINE_OPENCODE:
        profile = normalize_llm_config(server_config)["profiles"][ENGINE_OPENCODE]
        host = (urlparse(profile["base_url"]).hostname or "").lower()
        if host in {"127.0.0.1", "localhost", "::1"} and str(profile.get("provider", "")).strip().lower() in {"local", "ollama", "lmstudio", "lm-studio"}:
            return EngineCapabilities(**{**asdict(capabilities), "execution_location": "local_external", "requires_internet": False, "data_leaves_machine": False})
    return capabilities


def selected_remote_engines(server_config: dict[str, Any] | None) -> set[str]:
    engine = resolve_role_engine(server_config, "deepy")
    return {engine} if is_remote_engine(engine) else set()


def validate_llm_config(server_config: dict[str, Any] | None, *, deepy_enabled: Any, deepy_type: Any) -> dict[str, Any]:
    normalized = normalize_llm_config(server_config)
    remote_engines = selected_remote_engines({**dict(server_config or {}), LLM_CONFIG_KEY: normalized})
    if remote_engines and (int(deepy_enabled or 0) != 1 or str(deepy_type or "").strip().lower() != "prime"):
        names = ", ".join(sorted(ENGINE_LABELS.get(engine, engine.title()) for engine in remote_engines))
        raise ValueError(f"{names} requires Deepy Prime because only Deepy Prime exposes WanGP's MCP tools. Select Deepy Prime before saving.")
    if ENGINE_CODEX in remote_engines:
        from .codex_config import validate_codex_profile
        validate_codex_profile(normalized["profiles"][ENGINE_CODEX])
    if ENGINE_CLAUDE in remote_engines:
        from .claude_config import validate_claude_profile
        validate_claude_profile(normalized["profiles"][ENGINE_CLAUDE])
    if ENGINE_OPENCODE in remote_engines:
        from .opencode_config import validate_opencode_profile
        validate_opencode_profile(normalized["profiles"][ENGINE_OPENCODE])
    return normalized


def privacy_warning(server_config: dict[str, Any] | None) -> str:
    engine = resolve_role_engine(server_config, "deepy")
    providers = [ENGINE_LABELS.get(engine, engine.title())] if engine in REMOTE_ENGINES and engine_capabilities(server_config, engine).data_leaves_machine else []
    if not providers:
        return ""
    return f"**Internet connection required. Prompts, attached images, uploaded media frames, and relevant WanGP tool results may be sent to {', '.join(providers)}. Privacy, retention, and processing are governed by those providers and are not guaranteed by WanGP.**"
