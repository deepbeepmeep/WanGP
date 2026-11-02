"""
Service for sending messages to Deepy (the built-in Qwen3.5 VL LLM).

Deepy uses Qwen35AssistantRuntime (vLLM-backed) for text generation.
This service follows the same pattern: it loads the prompt enhancer runtime,
instantiates Qwen35AssistantRuntime, and calls generate_action().

Lazy-loads the runtime on first use and caches it for subsequent requests.
Unloads VRAM after each call so subsequent heavy models fit.

NOTE: Qwen3.5-based Deepy requires CUDA. On Mac (MPS) or CPU-only
systems, Deepy is unavailable and the service will raise a clear error.
"""

from __future__ import annotations

# Force single GPU to avoid multi-GPU device-mismatch issues
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import secrets
import threading
import base64
from pathlib import Path
from typing import Optional

import torch

# Lazy-loaded runtime
_deepy_lock = threading.RLock()
_deepy_runtime = None  # PromptEnhancerRuntime
_deepy_model_loaded: Optional[int] = None  # tracks which enhancer_enabled is loaded
_deepy_download_in_progress: bool = False
_deepy_models_on_gpu: bool = False


def _resolve_project_root() -> Path:
    """Resolve the project root from the current file location."""
    return Path(__file__).resolve().parents[2]


def _ensure_deepy_assets(enhancer_enabled: int):
    """Download prompt enhancer model assets if missing."""
    global _deepy_download_in_progress

    from shared.prompt_enhancer import download_prompt_enhancer_assets

    _deepy_download_in_progress = True
    try:
        download_prompt_enhancer_assets(enhancer_enabled=enhancer_enabled)
    finally:
        _deepy_download_in_progress = False


def _fix_qwen_device_placement(runtime):
    """Materialise all GGUF source tensors to CUDA for the Qwen LLM model."""
    llm = getattr(runtime, "llm_model", None)
    if llm is None:
        return

    from shared.qtypes.gguf import materialize_module_source_tensors

    try:
        materialize_module_source_tensors(llm)
        print("[DEEPY] Materialised GGUF source tensors to CUDA")
    except Exception as exc:
        print(f"[DEEPY] Failed to materialise GGUF tensors: {exc}")
        try:
            runtime.llm_model = llm.to("cuda")
            print("[DEEPY] Fallback: moved entire LLM model to CUDA")
        except Exception as exc2:
            print(f"[DEEPY] Fallback also failed: {exc2}")


def _load_deepy_runtime(enhancer_enabled: int):
    """Load (or reload) the prompt enhancer runtime for the given model."""
    global _deepy_runtime, _deepy_model_loaded

    from shared.prompt_enhancer.loader import load_prompt_enhancer_runtime
    from shared.utils.download import process_files_def

    print(f"[DEEPY] Loading enhancer runtime (enabled={enhancer_enabled})...")

    runtime = load_prompt_enhancer_runtime(
        process_files_def,
        enhancer_enabled=enhancer_enabled,
    )

    _materialize_deepy_models(runtime, enhancer_enabled)

    _deepy_runtime = runtime
    _deepy_model_loaded = enhancer_enabled
    print(f"[DEEPY] Enhancer runtime loaded successfully")


def _materialize_deepy_models(runtime, enhancer_enabled: int):
    """Move enhancer models onto CUDA if they are currently on CPU."""
    global _deepy_models_on_gpu
    if _deepy_models_on_gpu:
        return
    if enhancer_enabled in (1, 2):
        for obj_name in ("llm_model", "image_caption_model"):
            obj = getattr(runtime, obj_name, None)
            if obj is not None:
                try:
                    obj = obj.to("cuda")
                    setattr(runtime, obj_name, obj)
                    print(f"[DEEPY] Moved {obj_name} to CUDA")
                except Exception as exc:
                    print(f"[DEEPY] Failed to move {obj_name}: {exc}")
    elif enhancer_enabled in (3, 4):
        _fix_qwen_device_placement(runtime)
        vl = getattr(runtime, "image_caption_model", None)
        if vl is not None:
            try:
                from shared.qtypes.gguf import materialize_module_source_tensors
                materialize_module_source_tensors(vl)
            except Exception:
                pass
    _deepy_models_on_gpu = True


def unload_deepy_runtime():
    """Offload the cached prompt-enhancer models off the GPU after use."""
    global _deepy_runtime, _deepy_models_on_gpu
    runtime = _deepy_runtime
    if runtime is None:
        return
    for obj_name in ("llm_model", "image_caption_model"):
        obj = getattr(runtime, obj_name, None)
        if obj is None:
            continue
        try:
            unload = getattr(obj, "unload", None)
            if callable(unload):
                unload()
        except Exception as exc:
            print(f"[DEEPY] unload() failed for {obj_name}: {exc}")
        try:
            obj.to("cpu")
        except Exception as exc:
            print(f"[DEEPY] Failed to move {obj_name} to CPU: {exc}")
    _deepy_models_on_gpu = False
    import gc
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    print("[DEEPY] Enhancer models offloaded from GPU")


def get_deepy(enhancer_enabled: int):
    """Get the prompt enhancer runtime (for Deepy), loading if necessary."""
    global _deepy_runtime, _deepy_model_loaded

    if _deepy_runtime is None or _deepy_model_loaded != enhancer_enabled:
        with _deepy_lock:
            if _deepy_runtime is None or _deepy_model_loaded != enhancer_enabled:
                _ensure_deepy_assets(enhancer_enabled)
                _load_deepy_runtime(enhancer_enabled)

    return _deepy_runtime


def check_deepy_hardware(enhancer_enabled: int):
    """Raise RuntimeError if the model's required hardware is not available."""
    if enhancer_enabled in (3, 4):  # Qwen3.5 models
        if not torch.cuda.is_available():
            device_hint = "MPS (Apple Silicon)" if torch.backends.mps.is_available() else "CPU-only"
            raise RuntimeError(
                f"Deepy (Qwen3.5) requires CUDA (NVIDIA GPU). "
                f"Detected: {device_hint}."
            )


def _build_deepy_prompt(prompt: str, image: Optional[bytes] = None) -> str:
    """Build a conversational prompt for the Qwen3.5 VL model.

    For image input, creates a base64 data URI prefix that signals
    the model to process the image alongside the text.
    """
    parts = []

    if image is not None:
        b64 = base64.b64encode(image).decode("ascii")
        parts.append(f"image: data:image/png;base64,{b64}")

    parts.append(prompt)
    return "\n\n".join(parts)


def run_deepy_message(
    prompt: str,
    image: Optional[bytes] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: Optional[int] = None,
    enhancer_enabled: int = 4,  # Default: Qwen3.5-9B Abliterated
) -> tuple[str, int]:
    """Send a message to Deepy and return the response.

    Uses Qwen35AssistantRuntime (vLLM-backed) just like Deepy does internally.
    Must be called while holding _deepy_lock.
    """
    if seed is None or seed < 0:
        seed = secrets.randbits(32)

    from shared.deepy import ZERO_SYSTEM_PROMPT

    # Build the prompt
    system_prompt = ZERO_SYSTEM_PROMPT
    user_prompt = _build_deepy_prompt(prompt, image)

    # Get the prompt enhancer runtime (already loaded by caller)
    runtime = get_deepy(enhancer_enabled)
    _materialize_deepy_models(runtime, enhancer_enabled)

    # Ensure the project root is in sys.path for imports
    import sys
    project_root = _resolve_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Import Qwen35AssistantRuntime — this is what Deepy uses for text generation
    from shared.prompt_enhancer.qwen35_assistant_runtime import (
        Qwen35AssistantRuntime,
        Qwen35AssistantSessionState,
    )

    # Build a minimal session state for the conversation
    session_state = Qwen35AssistantSessionState(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    # Create the runtime (wraps the Qwen3.5 model, uses vLLM for generation)
    assistant_runtime = Qwen35AssistantRuntime(
        model=runtime.llm_model,
        tokenizer=runtime.llm_tokenizer,
    )

    # Generate using vLLM (same as Deepy does)
    result = assistant_runtime.generate_action(
        phase="statement",
        seed=seed,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        top_k=None,
        thinking_enabled=False,
        max_new_tokens=max_new_tokens,
    )

    # Clean up response
    response = result.raw_text.strip()
    for marker in ["Assistant:", "<|assistant|>", "assistant:"]:
        if response.startswith(marker):
            response = response[len(marker):].strip()

    # Free VRAM after use so subsequent heavy models fit
    unload_deepy_runtime()

    return response, seed
