# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

from typing import Any

from packaging.version import InvalidVersion, Version

from shared.deepy.config import (
    DEEPY_COMPACTION_SUMMARIZE_MIN_TOKENS,
    DEEPY_COMPACTION_TYPE_KEY,
    DEEPY_COMPACTION_TYPE_SUMMARIZE,
    DEEPY_CONTEXT_TOKENS_DEFAULT,
    DEEPY_CONTEXT_TOKENS_KEY,
    DEEPY_ENABLED_KEY,
    DEEPY_KV_CACHE_QUANTIZATION_DEFAULT,
    DEEPY_KV_CACHE_QUANTIZATION_KEY,
    DEEPY_TYPE_KEY,
    DEEPY_TYPE_PRIME,
    deepy_mode_from_config,
    estimate_deepy_kv_cache_mb,
    get_gguf_kernels_version,
    normalize_deepy_context_tokens,
)
from shared.prompt_enhancer.config import (
    OTHER_PROGRAMS_VRAM_GIB,
    PROMPT_ENHANCER_SPECULATIVE_DECODING_AUTO,
    PROMPT_ENHANCER_SPECULATIVE_DECODING_DEFAULT,
    PROMPT_ENHANCER_SPECULATIVE_DECODING_KEY,
    resolve_prompt_enhancer_speculative_decoding,
    speculative_decoding_runtime,
)
from shared.remote_llm.config import ENGINE_QWEN38_27B, LLM_CONFIG_KEY, local_enhancer_id, normalize_llm_config


DEEPY_PRIME_Q3_MIN_VRAM_GB = 16
DEEPY_PRIME_Q4_MIN_VRAM_GB = 24
QWEN38_QUANTIZATIONS = {"gguf_ptq1": "Bonsai Q1", "gguf_q2": "Q2", "gguf_q3": "Q3", "gguf": "Q4"}  # the most compressed first
BONSAI_MIN_GGUF_KERNELS_VERSION = "1.0.22"
DEEPY_FULL_CONTEXT_TOKENS = 64000  # a production's conversation is long, and below this Deepy keeps condensing it
DEEPY_FULL_CONTEXT_MIN_VRAM_GB = 10  # below, Bonsai no longer fits in VRAM with this context
QWEN38_VRAM_GIB = {"gguf_ptq1": 6.3, "gguf_q2": 9.8, "gguf_q3": 12.1, "gguf": 15.8}  # each version without its KV cache: the weights the GPU holds, activations and margins
MTP_WEIGHTS_VRAM_GIB = 0.42
MTP_KV_VRAM_GIB_PER_TOKEN = 4 / 2**20  # its own BF16 KV cache
BLOCK_DRAFTER_VRAM_GIB = 3.0  # DSpark, DFlash2 and DSpine at 32K context: weights, their window-capped cache (DSpark's grows 20 KiB per token) and buffers


def detected_nvidia_vram_gb() -> float:
    try:
        import torch

        if not torch.cuda.is_available() or not getattr(torch.version, "cuda", None) or getattr(torch.version, "hip", None):
            return 0.0
        return torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024 ** 3)
    except Exception:
        return 0.0


def reported_vram(vram_gb: float | None = None) -> int:
    return int((detected_nvidia_vram_gb() if vram_gb is None else vram_gb) + 0.5)


def bonsai_kernels_installed() -> bool:
    try:
        return Version(get_gguf_kernels_version()) >= Version(BONSAI_MIN_GGUF_KERNELS_VERSION)
    except InvalidVersion:
        return False


def deepy_prime_hardware_profile(vram_gb: float | None = None) -> tuple[str, str, int] | None:
    """Deepy Prime on the Qwen3.8 27B version that suits this GPU: (label, quantization, VRAM). None without an NVIDIA GPU, or below 16 GB without Bonsai's kernels, as Q2 would not fit there."""
    reported_vram_gb = reported_vram(vram_gb)
    if reported_vram_gb >= DEEPY_PRIME_Q4_MIN_VRAM_GB:
        return "Q4", "gguf", reported_vram_gb
    if reported_vram_gb >= DEEPY_PRIME_Q3_MIN_VRAM_GB:
        return "Q3", "gguf_q3", reported_vram_gb
    return ("Bonsai Q1", "gguf_ptq1", reported_vram_gb) if reported_vram_gb > 0 and bonsai_kernels_installed() else None


def qwen38_recommended_quantization(vram_gb: float | None = None) -> str | None:
    """The Qwen3.8 27B version that suits this GPU, None when no NVIDIA GPU tells its VRAM. Bonsai runs only on its kernels: Q2 is the lightest without them."""
    reported_vram_gb = reported_vram(vram_gb)
    if reported_vram_gb <= 0:
        return None
    profile = deepy_prime_hardware_profile(reported_vram_gb)
    return profile[1] if profile else "gguf_q2"


def apply_first_launch_deepy_prime_defaults(server_config: dict[str, Any], vram_gb: float | None = None) -> str:
    """The first launch starts on what Configuration recommends, with nothing to configure: Deepy Prime on the Qwen3.8 27B version that suits this GPU, so that Showrunner works at once."""
    profile = deepy_prime_hardware_profile(vram_gb)
    if profile is None:
        return ""
    label, quantization, reported_vram_gb = profile
    llm_config = normalize_llm_config(server_config)
    llm_config["deepy"] = ENGINE_QWEN38_27B
    server_config.update({
        DEEPY_ENABLED_KEY: 1,
        DEEPY_TYPE_KEY: DEEPY_TYPE_PRIME,
        DEEPY_CONTEXT_TOKENS_KEY: DEEPY_FULL_CONTEXT_TOKENS if reported_vram_gb >= DEEPY_FULL_CONTEXT_MIN_VRAM_GB else DEEPY_COMPACTION_SUMMARIZE_MIN_TOKENS,
        DEEPY_COMPACTION_TYPE_KEY: DEEPY_COMPACTION_TYPE_SUMMARIZE,
        "enhancer_enabled": local_enhancer_id(ENGINE_QWEN38_27B),
        "prompt_enhancer_quantization": quantization,
        LLM_CONFIG_KEY: llm_config,
    })
    return label


def qwen38_auto_mtp(server_config: dict[str, Any], quantization: str, reported_vram_gb: int) -> tuple[bool, float, float, float]:
    """What Auto would decide for this GPU, version, context and KV cache: (it takes MTP, VRAM the version needs, VRAM MTP adds, VRAM available), in GiB."""
    enhancer = local_enhancer_id(ENGINE_QWEN38_27B)
    tokens = server_config.get(DEEPY_CONTEXT_TOKENS_KEY, DEEPY_CONTEXT_TOKENS_DEFAULT)
    needed = QWEN38_VRAM_GIB[quantization] + estimate_deepy_kv_cache_mb(enhancer, tokens, server_config.get(DEEPY_KV_CACHE_QUANTIZATION_KEY, DEEPY_KV_CACHE_QUANTIZATION_DEFAULT))[1] / 1024
    extra, available = MTP_WEIGHTS_VRAM_GIB + normalize_deepy_context_tokens(tokens) * MTP_KV_VRAM_GIB_PER_TOKEN, reported_vram_gb - OTHER_PROGRAMS_VRAM_GIB
    fit = lambda: (int(needed * 2**30), int(extra * 2**30), int(available * 2**30))
    return speculative_decoding_runtime(resolve_prompt_enhancer_speculative_decoding(enhancer, "auto", reported_vram_gb, qwen_backend=quantization, fit=fit)[0])[0] != 0, needed, extra, available


def performance_warnings(server_config: dict[str, Any], vram_gb: float | None = None) -> list[tuple[str, str, str]]:
    """What the settings of the local Qwen3.8 27B cost on this GPU, as (icon, title, text): a version less compressed than the one that suits it; a speculative decoding chosen by hand that pushes the model out of VRAM
    (MTP where Auto would keep it off for this version, a block drafter that does not fit); MTP disabled while this version fits in VRAM with it."""
    reported_vram_gb = reported_vram(vram_gb)
    chosen = server_config.get("prompt_enhancer_quantization")
    if reported_vram_gb <= 0 or normalize_llm_config(server_config)["deepy"] != ENGINE_QWEN38_27B or chosen not in QWEN38_QUANTIZATIONS:
        return []
    warnings, recommended, order = [], qwen38_recommended_quantization(reported_vram_gb), list(QWEN38_QUANTIZATIONS)
    if order.index(chosen) > order.index(recommended):
        warnings.append(("⚠️", f"Qwen3.8 {QWEN38_QUANTIZATIONS[chosen]} may perform poorly on your GPU", f"with {reported_vram_gb} GB of VRAM, part of the model may not fit and Deepy and the Prompt Enhancer can become very slow. For this GPU, select {QWEN38_QUANTIZATIONS[recommended]} under Qwen LLM Quantization"))
    mode = speculative_decoding_runtime(server_config.get(PROMPT_ENHANCER_SPECULATIVE_DECODING_KEY, PROMPT_ENHANCER_SPECULATIVE_DECODING_DEFAULT))[0]
    if mode == PROMPT_ENHANCER_SPECULATIVE_DECODING_AUTO:
        return warnings
    auto_takes_mtp, needed, extra, available = qwen38_auto_mtp(server_config, chosen, reported_vram_gb)
    if mode == 0:
        if auto_takes_mtp and needed + extra <= available:
            warnings.append(("💡", "Deepy could run about twice as fast on your GPU", f"with {reported_vram_gb} GB of VRAM, Qwen3.8 {QWEN38_QUANTIZATIONS[chosen]} fits in VRAM with MTP, which is disabled. Set Speculative Decoding to Auto"))
    elif not auto_takes_mtp if mode == 1 else needed + BLOCK_DRAFTER_VRAM_GIB > available:
        warnings.append(("⚠️", "Speculative Decoding may slow Deepy down on your GPU", f"with {reported_vram_gb} GB of VRAM, it pushes part of Qwen3.8 {QWEN38_QUANTIZATIONS[chosen]} out of VRAM. Set Speculative Decoding to Auto"))
    return warnings


def deepy_performance_warning(server_config: dict[str, Any], vram_gb: float | None = None, icons: bool = False) -> str:
    """The warnings as plain text, for Showrunner's notice and, with their icons, Deepy's chat."""
    return " ".join(f"{icon + ' ' if icons else ''}{title}: {text} in Configuration > Prompt Enhancer / Deepy." for icon, title, text in performance_warnings(server_config, vram_gb))


def deepy_onboarding_notice(server_config: dict[str, Any], vram_gb: float | None = None) -> str:
    warnings = [f"{icon} **{title}** — {text} below." for icon, title, text in performance_warnings(server_config, vram_gb)]
    return "\n\n".join(message for message in (*warnings, deepy_prime_upgrade_message(server_config, vram_gb)) if message)


def deepy_prime_upgrade_message(server_config: dict[str, Any], vram_gb: float | None = None) -> str:
    profile = deepy_prime_hardware_profile(vram_gb)
    if profile is None or deepy_mode_from_config(server_config.get(DEEPY_ENABLED_KEY, 0), server_config.get(DEEPY_TYPE_KEY)) == DEEPY_TYPE_PRIME:
        return ""
    label, _quantization, reported_vram_gb = profile
    select = "Deepy Prime" if normalize_llm_config(server_config)["deepy"] == ENGINE_QWEN38_27B else "the Qwen3.8 VL Uncensored 27B engine and Deepy Prime"
    return f"✨ **Switch to Deepy Prime {label}** — your NVIDIA GPU ({reported_vram_gb} GB VRAM) is eligible. Gain Showrunner, multi-step planning, image/video/audio tool chaining, and end-to-end handling of long creations. Select {select} below to switch."
