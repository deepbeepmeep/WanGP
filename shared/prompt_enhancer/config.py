# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

from typing import Any


PROMPT_ENHANCER_SPECULATIVE_DECODING_KEY = "prompt_enhancer_speculative_decoding"
PROMPT_ENHANCER_SPECULATIVE_DECODING_AUTO = 2
PROMPT_ENHANCER_SPECULATIVE_DECODING_DEFAULT = PROMPT_ENHANCER_SPECULATIVE_DECODING_AUTO
PROMPT_ENHANCER_SPECULATIVE_DECODING_IDS = frozenset((4, 5, 6))
PROMPT_ENHANCER_SPECULATIVE_DECODING_VRAM_GB = {4: 12, 5: 24, 6: 8}  # Auto when the model's fit in VRAM is unknown (resolve_..., fit)
MTP_AUTO_SLACK = 512 << 20  # Auto: MTP still pays when it leaves this much of the model out of VRAM (with the model in VRAM it decodes about twice as fast)
PROMPT_ENHANCER_SPECULATIVE_DRAFT_COUNTS = (2, 3, 4)
BLOCK_DRAFT_METHODS = ("dspark", "dflash2")
SPECULATIVE_METHOD_LABELS = {"auto": "Auto", "disabled": "Disabled", "mtp": "MTP", "dspark": "DSpark", "dflash2": "DFlash2"}
SPECULATIVE_MAX_TOKENS = {"mtp": 8, "dspark": 7, "dflash2": 7}
SPECULATIVE_DEFAULT_TOKENS = {"mtp": 2, "dspark": 7, "dflash2": 5}
PROMPT_ENHANCER_SPECULATIVE_DECODING_CHOICES = [
    ("Auto (MTP)", PROMPT_ENHANCER_SPECULATIVE_DECODING_AUTO),
    ("Disabled", 0),
    *[(f"MTP with {count} draft tokens", 1 if count == 2 else count) for count in PROMPT_ENHANCER_SPECULATIVE_DRAFT_COUNTS],
]


def normalize_prompt_enhancer_speculative_decoding(value: Any) -> int | str | dict:
    if isinstance(value, dict):
        return speculative_decoding_config(value.get("method", "auto"), value.get("tokens"))
    if isinstance(value, str):
        text = value.strip().lower()
        if text in BLOCK_DRAFT_METHODS:
            return text
        if text in ("auto", "2"):
            return PROMPT_ENHANCER_SPECULATIVE_DECODING_AUTO
        if text in {str(count) for count in PROMPT_ENHANCER_SPECULATIVE_DRAFT_COUNTS if count > 2}:
            return int(text)
        return 1 if text in {"1", "true", "yes", "on"} else 0
    try:
        mode = int(value)
        return mode if mode == PROMPT_ENHANCER_SPECULATIVE_DECODING_AUTO or mode in PROMPT_ENHANCER_SPECULATIVE_DRAFT_COUNTS else (1 if bool(value) else 0)
    except (TypeError, ValueError):
        return PROMPT_ENHANCER_SPECULATIVE_DECODING_DEFAULT


def speculative_decoding_config(method, tokens=None):
    method = str(method).lower()
    if method not in SPECULATIVE_METHOD_LABELS:
        raise ValueError(f"Unknown speculative decoding method: {method}")
    if method in SPECULATIVE_MAX_TOKENS:
        tokens = SPECULATIVE_DEFAULT_TOKENS[method] if tokens is None else int(tokens)
        if not 1 <= tokens <= SPECULATIVE_MAX_TOKENS[method]:
            raise ValueError(f"{SPECULATIVE_METHOD_LABELS[method]} supports 1 to {SPECULATIVE_MAX_TOKENS[method]} draft tokens.")
    else:
        tokens = None
    return {"method": method, "tokens": tokens}


def split_speculative_decoding(value):
    value = normalize_prompt_enhancer_speculative_decoding(value)
    if isinstance(value, dict):
        return value["method"], value["tokens"]
    if value in BLOCK_DRAFT_METHODS:
        return value, SPECULATIVE_DEFAULT_TOKENS[value]
    if value == 0:
        return "disabled", None
    if value == PROMPT_ENHANCER_SPECULATIVE_DECODING_AUTO:
        return "auto", None
    return "mtp", 2 if value == 1 else value


def speculative_decoding_runtime(value):
    method, tokens = split_speculative_decoding(value)
    return {"auto": 2, "disabled": 0, "mtp": 1}.get(method, method), tokens


def block_drafters_supported() -> bool:
    """DSpark and DFlash2 run on Triton and FlashAttention 2 (the LLM engine uses each one wherever it is supported)."""
    from shared.llm_engines.lmgp.vllm_support import resolve_prompt_enhancer_lm_features
    features = resolve_prompt_enhancer_lm_features()
    return features.triton and features.flash


def speculative_decoding_ui_state(enhancer_enabled, quantization, value, tokens=None):
    methods = ["auto", "disabled"]
    if prompt_enhancer_supports_speculative_decoding(enhancer_enabled):
        methods.append("mtp")
    if int(enhancer_enabled or 0) == 5 and quantization in ("gguf", "gguf_q3", "gguf_q2", "gguf_ptq1") and block_drafters_supported():
        methods.extend(BLOCK_DRAFT_METHODS)
    if isinstance(value, str) and value in SPECULATIVE_METHOD_LABELS:
        method, count = value, tokens
    else:
        method, count = split_speculative_decoding(value)
    if method not in methods:
        method = "auto"
    maximum = SPECULATIVE_MAX_TOKENS.get(method, 0)
    if method in BLOCK_DRAFT_METHODS:
        from .block_draft import block_draft_spec
        maximum = min(maximum, block_draft_spec(method, bonsai=quantization == "gguf_ptq1")["drafts"])
    count = min(maximum, max(1, int(count or SPECULATIVE_DEFAULT_TOKENS[method]))) if maximum else None
    # Additional residency near 32K context, including draft weights/cache/state.
    # These are approximate ranges, not a capacity check (see BONSAI_VRAM.md).
    costs = {
        "auto": "0 to ~1 GiB" if "mtp" in methods else "0 GiB",
        "disabled": "0 GiB",
        "mtp": "~0.5–1 GiB",
        "dspark": "~4–5 GiB",
        "dflash2": "~4–5 GiB",
    }
    return [(f"{SPECULATIVE_METHOD_LABELS[item]} [+{costs[item]} VRAM]", item) for item in methods], method, list(range(1, maximum + 1)), count


def prompt_enhancer_supports_speculative_decoding(enhancer_enabled: Any) -> bool:
    try:
        return int(enhancer_enabled or 0) in PROMPT_ENHANCER_SPECULATIVE_DECODING_IDS
    except (TypeError, ValueError):
        return False


def validate_prompt_enhancer_speculative_decoding(enhancer_enabled: Any, value: Any) -> int | str | dict:
    enabled = normalize_prompt_enhancer_speculative_decoding(value)
    runtime_mode, _ = speculative_decoding_runtime(enabled)
    if runtime_mode in BLOCK_DRAFT_METHODS and int(enhancer_enabled or 0) != 5:
        raise ValueError("DSpark and DFlash2 require Qwen3.8-27B or Bonsai 2 27B.")
    if runtime_mode not in (0, PROMPT_ENHANCER_SPECULATIVE_DECODING_AUTO) and not prompt_enhancer_supports_speculative_decoding(enhancer_enabled):
        if int(enhancer_enabled or 0) == 3:
            raise ValueError("Speculative decoding is not available with Qwen3.5-4B.")
        raise ValueError("Speculative decoding requires the Qwen3.5-9B, Qwen3.8-9B or Qwen3.8-27B prompt enhancer.")
    return enabled


def resolve_prompt_enhancer_speculative_decoding(enhancer_enabled: Any, value: Any, total_vram_gb: float | None = None, *, qwen_backend: str = "", fit=None) -> tuple[int | str | dict, str]:
    """The speculative decoding a setting runs, and the message explaining an automatic choice. fit: a function giving (VRAM the model
    needs without MTP, VRAM MTP adds, VRAM available) in bytes, or None if unknown. Auto then takes MTP unless it would push a model that
    fits in VRAM without it out of VRAM: measured on Qwen3.8 27B IQ2_M, MTP decodes 2.1x faster when both fit (15 GiB) and 1.5-1.7x when
    the weights stream either way (an 11 GB RTX 2080 Ti, a VRAM-limited RTX 4090 after a video), but 20% slower when the model fits
    only without it (11 GiB). Without fit, a fixed VRAM size per model decides."""
    mode = validate_prompt_enhancer_speculative_decoding(enhancer_enabled, value)
    if speculative_decoding_runtime(mode)[0] != PROMPT_ENHANCER_SPECULATIVE_DECODING_AUTO:
        return mode, ""
    try:
        enhancer_no = int(enhancer_enabled or 0)
    except (TypeError, ValueError):
        enhancer_no = 0
    required_vram_gb = PROMPT_ENHANCER_SPECULATIVE_DECODING_VRAM_GB.get(enhancer_no)
    if required_vram_gb is None:
        return 0, "Speculative Decoding disabled automatically because the selected model does not support it."
    if total_vram_gb is None:
        try:
            import torch

            total_vram_gb = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / (1024 ** 3)
        except Exception:
            total_vram_gb = 0
    if enhancer_no == 5 and qwen_backend == "gguf_ptq1":
        detected_vram = max(0.0, float(total_vram_gb))
        if detected_vram > 10:
            return speculative_decoding_config("mtp", 2), f"Speculative Decoding enabled automatically for Bonsai PTQ1: {detected_vram:.2f} GiB VRAM detected (more than 10 GiB; 2 MTP tokens)."
        return 0, f"Speculative Decoding disabled automatically for Bonsai PTQ1: {detected_vram:.2f} GiB VRAM detected (10 GiB or less)."
    detected_vram_gb = max(0, int(float(total_vram_gb) + 0.5))
    model_label = {4: "Qwen3.5-9B", 6: "Qwen3.8-9B"}.get(enhancer_no, "Qwen3.8-27B")
    if float(total_vram_gb) <= 8:  # 8 GB GPUs: the drafter, its caches and graphs would take room from the model's blocks
        return 0, f"Speculative Decoding disabled automatically for {model_label}: {max(0.0, float(total_vram_gb)):.2f} GiB VRAM detected (8 GiB or less)."
    sizes = None if fit is None else fit()
    if sizes is not None:
        needed, extra, available = sizes
        gib = lambda nbytes: f"{nbytes / 2**30:.1f}"
        if needed + extra - available <= MTP_AUTO_SLACK:
            return 1, f"Speculative Decoding enabled automatically for {model_label}: with MTP it needs about {gib(needed + extra)} GiB of VRAM, {gib(available)} GiB available."
        if needed - available <= MTP_AUTO_SLACK:
            return 0, (f"Speculative Decoding disabled automatically for {model_label}: it fits or nearly fits in VRAM without MTP (about {gib(needed)} GiB needed, "
                       f"{gib(available)} GiB available), not with it (+{gib(extra)} GiB would stream part of the model at every pass).")
        return 1, (f"Speculative Decoding enabled automatically for {model_label}: part of the model streams from RAM either way (about {gib(needed)} GiB needed, "
                   f"{gib(available)} GiB available), and MTP decodes several tokens per pass over it.")
    if detected_vram_gb >= required_vram_gb:
        return 1, f"Speculative Decoding enabled automatically for {model_label}: {detected_vram_gb} GB VRAM detected (minimum {required_vram_gb} GB)."
    return 0, f"Speculative Decoding disabled automatically for {model_label}: {detected_vram_gb} GB VRAM detected (minimum {required_vram_gb} GB)."
