# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

from typing import Any

from packaging.version import InvalidVersion, Version

from shared.deepy.config import (
    DEEPY_COMPACTION_SUMMARIZE_MIN_TOKENS,
    DEEPY_COMPACTION_TYPE_KEY,
    DEEPY_COMPACTION_TYPE_SUMMARIZE,
    DEEPY_CONTEXT_TOKENS_KEY,
    DEEPY_ENABLED_KEY,
    DEEPY_TYPE_KEY,
    DEEPY_TYPE_PRIME,
    deepy_mode_from_config,
    get_gguf_kernels_version,
)
from shared.remote_llm.config import ENGINE_QWEN38_27B, LLM_CONFIG_KEY, local_enhancer_id, normalize_llm_config


DEEPY_PRIME_Q3_MIN_VRAM_GB = 16
DEEPY_PRIME_Q4_MIN_VRAM_GB = 24
QWEN38_QUANTIZATIONS = {"gguf_ptq1": "Bonsai Q1", "gguf_q2": "Q2", "gguf_q3": "Q3", "gguf": "Q4"}  # the most compressed first
BONSAI_MIN_GGUF_KERNELS_VERSION = "1.0.22"


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


def deepy_prime_hardware_profile(vram_gb: float | None = None) -> tuple[str, str, int] | None:
    reported_vram_gb = reported_vram(vram_gb)
    if reported_vram_gb >= DEEPY_PRIME_Q4_MIN_VRAM_GB:
        return "Q4", "gguf", reported_vram_gb
    if reported_vram_gb >= DEEPY_PRIME_Q3_MIN_VRAM_GB:
        return "Q3", "gguf_q3", reported_vram_gb
    return None


def bonsai_kernels_installed() -> bool:
    try:
        return Version(get_gguf_kernels_version()) >= Version(BONSAI_MIN_GGUF_KERNELS_VERSION)
    except InvalidVersion:
        return False


def qwen38_recommended_quantization(vram_gb: float | None = None) -> str | None:
    """The Qwen3.8 27B version that suits this GPU, None when no NVIDIA GPU tells its VRAM. Bonsai runs only on its kernels: Q2 is the lightest without them."""
    reported_vram_gb = reported_vram(vram_gb)
    if reported_vram_gb <= 0:
        return None
    profile = deepy_prime_hardware_profile(reported_vram_gb)
    return profile[1] if profile else "gguf_ptq1" if bonsai_kernels_installed() else "gguf_q2"


def apply_first_launch_deepy_prime_defaults(server_config: dict[str, Any], vram_gb: float | None = None) -> str:
    """The first launch starts on what Configuration recommends: Qwen3.8 27B in the version that suits this GPU, with Deepy Prime from 16 GB of VRAM. Nothing changes without an NVIDIA GPU, nor
    below 16 GB without Bonsai's kernels: Q2 would not fit there."""
    quantization = qwen38_recommended_quantization(vram_gb)
    if quantization is None or quantization == "gguf_q2":
        return ""
    llm_config = normalize_llm_config(server_config)
    llm_config["deepy"] = ENGINE_QWEN38_27B
    server_config.update({"enhancer_enabled": local_enhancer_id(ENGINE_QWEN38_27B), "prompt_enhancer_quantization": quantization, LLM_CONFIG_KEY: llm_config})
    if deepy_prime_hardware_profile(vram_gb) is not None:
        server_config.update({
            DEEPY_ENABLED_KEY: 1,
            DEEPY_TYPE_KEY: DEEPY_TYPE_PRIME,
            DEEPY_CONTEXT_TOKENS_KEY: DEEPY_COMPACTION_SUMMARIZE_MIN_TOKENS,
            DEEPY_COMPACTION_TYPE_KEY: DEEPY_COMPACTION_TYPE_SUMMARIZE,
        })
    return QWEN38_QUANTIZATIONS[quantization]


def qwen38_quantization_warning(server_config: dict[str, Any], vram_gb: float | None = None) -> str:
    """A warning when the local Qwen3.8 27B runs a version less compressed than the one that suits this GPU."""
    recommended = qwen38_recommended_quantization(vram_gb)
    chosen = server_config.get("prompt_enhancer_quantization")
    if recommended is None or normalize_llm_config(server_config)["deepy"] != ENGINE_QWEN38_27B or chosen not in QWEN38_QUANTIZATIONS:
        return ""
    order = list(QWEN38_QUANTIZATIONS)
    if order.index(chosen) <= order.index(recommended):
        return ""
    return (f"⚠️ **Qwen3.8 {QWEN38_QUANTIZATIONS[chosen]} may perform poorly on your GPU** — with {reported_vram(vram_gb)} GB of VRAM, part of the model may not fit and Deepy and the Prompt Enhancer "
            f"can become very slow. For this GPU, select **{QWEN38_QUANTIZATIONS[recommended]}** under Qwen LLM Quantization below.")


def deepy_onboarding_notice(server_config: dict[str, Any], vram_gb: float | None = None) -> str:
    return "\n\n".join(message for message in (qwen38_quantization_warning(server_config, vram_gb), deepy_prime_upgrade_message(server_config, vram_gb)) if message)


def deepy_prime_upgrade_message(server_config: dict[str, Any], vram_gb: float | None = None) -> str:
    profile = deepy_prime_hardware_profile(vram_gb)
    if profile is None or deepy_mode_from_config(server_config.get(DEEPY_ENABLED_KEY, 0), server_config.get(DEEPY_TYPE_KEY)) == DEEPY_TYPE_PRIME:
        return ""
    label, _quantization, reported_vram_gb = profile
    return f"✨ **Switch to Deepy Prime {label}** — your NVIDIA GPU ({reported_vram_gb} GB VRAM) is eligible. Gain multi-step planning, image/video/audio tool chaining, and end-to-end handling of long creations. Select Deepy Prime below to switch."
