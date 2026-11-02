# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Tuple

import torch
from mmgp import offload
from safetensors import safe_open
from shared.utils import files_locator as fl
from shared.utils.cancellation import check_cancelled

from .florence2 import Florence2Config, Florence2ForConditionalGeneration, Florence2Processor
from .florence2.image_processing_florence2 import Florence2ImageProcessorLite

from transformers import AutoTokenizer, BartTokenizer, BartTokenizerFast

from .assets import (
    FLORENCE2_FILES,
    FLORENCE2_FOLDER,
    LLAMA32_FILES,
    LLAMA32_FOLDER,
    LLAMAJOY_FILES,
    LLAMAJOY_FOLDER,
    PROMPT_ENHANCER_REPO,
)
from shared.deepy.config import resolve_deepy_kv_cache_quantization
from .config import resolve_prompt_enhancer_speculative_decoding, speculative_decoding_runtime


@dataclass(slots=True)
class PromptEnhancerRuntime:
    image_caption_model: Any = None
    image_caption_processor: Any = None
    llm_model: Any = None
    llm_tokenizer: Any = None
    pipe_models: dict[str, Any] = field(default_factory=dict)
    budgets: dict[str, int] = field(default_factory=dict)
    co_tenants: dict[str, list[str]] = field(default_factory=dict)


def ensure_prompt_enhancer_assets(process_files_def, enhancer_enabled: int, qwen_backend: str = "quanto_int8", speculative_decoding: bool = False):
    enhancer_enabled = int(enhancer_enabled)
    if enhancer_enabled == 1:
        process_files_def(
            repoId=PROMPT_ENHANCER_REPO,
            sourceFolderList=[FLORENCE2_FOLDER, LLAMA32_FOLDER],
            fileList=[
                FLORENCE2_FILES,
                LLAMA32_FILES,
            ],
        )
        return
    if enhancer_enabled == 2:
        process_files_def(
            repoId=PROMPT_ENHANCER_REPO,
            sourceFolderList=[FLORENCE2_FOLDER, LLAMAJOY_FOLDER],
            fileList=[
                FLORENCE2_FILES,
                LLAMAJOY_FILES,
            ],
        )
        return
    if enhancer_enabled in (3, 4, 5, 6):
        from .qwen35_vl import ensure_qwen35_prompt_enhancer_assets, get_qwen35_prompt_enhancer_variant

        # Auto: the drafter's files whenever it may run (the choice is made at load, from the model's fit in VRAM)
        speculative_decoding = resolve_prompt_enhancer_speculative_decoding(enhancer_enabled, speculative_decoding, qwen_backend=qwen_backend, fit=lambda: (0, 0, 1))[0]
        speculative_decoding, _ = speculative_decoding_runtime(speculative_decoding)
        ensure_qwen35_prompt_enhancer_assets(process_files_def, backend=qwen_backend, variant=get_qwen35_prompt_enhancer_variant(enhancer_enabled), speculative_decoding=speculative_decoding)


def download_prompt_enhancer_assets(enhancer_enabled: int, qwen_backend: str = "quanto_int8", speculative_decoding: bool = False, send_cmd=None, progress=None, status_text="Downloading Prompt Enhancer model files...", gen=None):
    enhancer_enabled = int(enhancer_enabled)
    if enhancer_enabled <= 0:
        return False

    from shared.utils.download import download_def_missing_files, process_files_def_if_needed

    downloaded = False
    status_sent = False

    def process_download_def(**download_def):
        nonlocal downloaded, status_sent
        has_missing_files = len(download_def_missing_files(download_def)) > 0
        download_status_text = None
        if has_missing_files and not status_sent:
            if progress is not None:
                progress(0, status_text)
            download_status_text = status_text
            status_sent = True
        downloaded = process_files_def_if_needed(download_def, send_cmd=send_cmd, status_text=download_status_text, gen=gen) or downloaded

    ensure_prompt_enhancer_assets(process_download_def, enhancer_enabled=enhancer_enabled, qwen_backend=qwen_backend, speculative_decoding=speculative_decoding)
    return downloaded


def unload_prompt_enhancer_models(*models):
    seen = set()
    for model in models:
        if model is None:
            continue
        model_id = id(model)
        if model_id in seen:
            continue
        seen.add(model_id)
        unload = getattr(model, "unload", None)
        if callable(unload):
            unload()


def _set_pad_token_from_tokenizer(model, tokenizer):
    model.generation_config.pad_token = tokenizer.eos_token
    if model.generation_config.pad_token_id is None:
        eos_token_id = model.generation_config.eos_token_id
        model.generation_config.pad_token_id = eos_token_id[0] if isinstance(eos_token_id, list) else eos_token_id


def _load_llama32_prompt_enhancer():
    llm_model = offload.fast_load_transformers_model(
        fl.locate_file(f"{LLAMA32_FOLDER}/Llama3_2_quanto_bf16_int8.safetensors"),
        defaultConfigPath=fl.locate_file(f"{LLAMA32_FOLDER}/config.json", error_if_none=False),
        configKwargs={"attn_implementation": "sdpa", "hidden_act": "silu"},
        writable_tensors=False,
    )
    llm_model._validate_model_kwargs = lambda *_args, **_kwargs: None
    llm_model._offload_hooks = ["generate"]
    llm_tokenizer = AutoTokenizer.from_pretrained(fl.locate_folder(LLAMA32_FOLDER))
    _set_pad_token_from_tokenizer(llm_model, llm_tokenizer)
    llm_model.eval()
    return llm_model, llm_tokenizer, 5000


def _load_joycaption_prompt_enhancer():
    def preprocess_sd(sd, quant_map=None, tied_map=None):
        rules = {"model.language_model": "model", "model.vision_tower": None, "model.multi_modal_projector": None}
        return tuple(offload.map_state_dict([sd, quant_map, tied_map], rules))

    llm_model = offload.fast_load_transformers_model(
        fl.locate_file(f"{LLAMAJOY_FOLDER}/llama_joycaption_quanto_bf16_int8.safetensors"),
        forcedConfigPath=fl.locate_file(f"{LLAMAJOY_FOLDER}/llama_config.json", error_if_none=False),
        configKwargs={"attn_implementation": "sdpa", "hidden_act": "silu"},
        preprocess_sd=preprocess_sd,
        writable_tensors=False,
    )
    llm_tokenizer = AutoTokenizer.from_pretrained(fl.locate_folder(LLAMAJOY_FOLDER))
    _set_pad_token_from_tokenizer(llm_model, llm_tokenizer)
    llm_model.eval()
    return llm_model, llm_tokenizer, 10000


def load_prompt_enhancer_runtime(process_files_def, enhancer_enabled: int, qwen_backend: str = "quanto_int8", speculative_decoding: bool = False, deepy_kv_cache_quantization: str = "") -> PromptEnhancerRuntime:
    enhancer_enabled = int(enhancer_enabled)
    runtime = PromptEnhancerRuntime()
    if enhancer_enabled <= 0:
        return runtime

    ensure_prompt_enhancer_assets(process_files_def, enhancer_enabled=enhancer_enabled, qwen_backend=qwen_backend, speculative_decoding=speculative_decoding)

    if enhancer_enabled in (3, 4, 5, 6):
        deepy_kv_cache_quantization, kv_cache_message = resolve_deepy_kv_cache_quantization(deepy_kv_cache_quantization)
        if kv_cache_message:
            print(f"[Prompt Enhancer / Deepy] {kv_cache_message}")
        from .qwen35_text import load_qwen35_text_prompt_enhancer, qwen35_mtp_fit
        from .qwen35_vl import (
            enhancer_quantization_GGUF,
            enhancer_quantization_GGUF_Q3,
            enhancer_quantization_GGUF_Q2,
            enhancer_quantization_GGUF_PTQ1,
            enhancer_quantization_GGUF_Q8,
            enhancer_quantization_QUANTO_INT8,
            alias_qwen35_text_embedding_for_mmgp,
            get_qwen35_assets_dir_name,
            get_qwen35_prompt_enhancer_variant,
            get_qwen35_quantization,
            get_qwen35_text_gguf_path,
            get_qwen35_variant_spec,
            load_qwen35_vl_prompt_enhancer,
        )

        qwen35_variant = get_qwen35_prompt_enhancer_variant(enhancer_enabled)
        spec = get_qwen35_variant_spec(qwen35_variant)
        quantization = get_qwen35_quantization(qwen_backend or enhancer_quantization_QUANTO_INT8, variant=qwen35_variant)
        backend = enhancer_quantization_GGUF if quantization in (enhancer_quantization_GGUF_Q3, enhancer_quantization_GGUF_Q2, enhancer_quantization_GGUF_PTQ1, enhancer_quantization_GGUF_Q8) else quantization
        assets_dir_name = get_qwen35_assets_dir_name(qwen35_variant)
        assets_dir = fl.locate_folder(assets_dir_name, error_if_none=False) or fl.get_download_location(assets_dir_name)
        if backend == enhancer_quantization_GGUF:
            from shared.qtypes.gguf import get_gguf_compute_dtype
            default_dtype = get_gguf_compute_dtype()
        else:
            default_dtype = torch.bfloat16
        text_model_path = get_qwen35_text_gguf_path(assets_dir, variant=qwen35_variant, backend=quantization) if backend == enhancer_quantization_GGUF else None
        fit = lambda: qwen35_mtp_fit(assets_dir, qwen35_variant, backend, text_model_path, deepy_kv_cache_quantization)
        speculative_decoding, speculative_decoding_message = resolve_prompt_enhancer_speculative_decoding(enhancer_enabled, speculative_decoding, qwen_backend=qwen_backend, fit=fit)
        speculative_decoding, speculative_tokens = speculative_decoding_runtime(speculative_decoding)
        if speculative_decoding_message:
            print(f"[Prompt Enhancer / Deepy] {speculative_decoding_message}")
        runtime.llm_model = load_qwen35_text_prompt_enhancer(
            model_path=text_model_path,
            assets_dir=assets_dir,
            default_dtype=default_dtype,
            backend=backend,
            attn_implementation="sdpa",
            variant=qwen35_variant,
            speculative_decoding=speculative_decoding,
            kv_cache_int8=deepy_kv_cache_quantization != "",
            kv_cache_layout=deepy_kv_cache_quantization,
        )
        if speculative_tokens is not None:
            speculative_tokens = min(speculative_tokens, getattr(runtime.llm_model, "_block_draft_max_tokens", speculative_tokens))
            if backend == enhancer_quantization_GGUF:
                # 8 drafts verify 9 rows, beyond the 2-8 row GGUF decode kernels: the verification falls to MMQ (36.5 ms a pass instead
                # of 26 for 4 drafts on an RTX 4090, Qwen3.8 27B Q4_K_M) for no more emitted tokens
                speculative_tokens = min(speculative_tokens, 7)
            runtime.llm_model._prompt_enhancer_speculative_tokens = speculative_tokens
            runtime.llm_model._prompt_enhancer_speculative_sampling_tokens = speculative_tokens
        runtime.llm_tokenizer = getattr(runtime.llm_model, "_prompt_enhancer_tokenizer", None)
        runtime.llm_model.eval()
        caption_embedding_model = alias_qwen35_text_embedding_for_mmgp(runtime.llm_model)
        runtime.image_caption_model, vision_tower_model = load_qwen35_vl_prompt_enhancer(
            assets_dir=assets_dir,
            attn_implementation="sdpa",
            text_model=runtime.llm_model,
            input_embedding_model=caption_embedding_model,
            backend=backend,
            variant=qwen35_variant,
        )
        runtime.image_caption_processor = getattr(runtime.image_caption_model, "_prompt_enhancer_processor", None)
        runtime.image_caption_model.eval()
        runtime.pipe_models["vision_tower_model"] = vision_tower_model
        runtime.pipe_models["embedding_model"] = caption_embedding_model
        runtime.llm_model._mm_pinned_pool = True  # the prompt enhancer's and Deepy Prime's language model decodes token by token: its blocks enter MMGP's pinned pool
        runtime.pipe_models["llm_model"] = runtime.llm_model
        runtime.budgets["vision_tower_model"] = spec.get("vision_budget", 3000)
        runtime.budgets["embedding_model"] = spec.get("embedding_budget", 2000)
        runtime.budgets["llm_model"] = spec.get("llm_budget", 10000)
        runtime.co_tenants["vision_tower_model"] = ["embedding_model"]
        runtime.co_tenants["embedding_model"] = ["vision_tower_model"]
        return runtime

    runtime.image_caption_model, runtime.image_caption_processor = load_florence2(fl.locate_folder(FLORENCE2_FOLDER), attn_implementation="sdpa")
    runtime.image_caption_model._model_dtype = torch.float
    runtime.image_caption_model.eval()
    runtime.pipe_models["image_caption_model"] = runtime.image_caption_model
    if enhancer_enabled == 1:
        runtime.llm_model, runtime.llm_tokenizer, budget = _load_llama32_prompt_enhancer()
    else:
        runtime.llm_model, runtime.llm_tokenizer, budget = _load_joycaption_prompt_enhancer()
    runtime.llm_model._mm_pinned_pool = True  # decodes token by token: its blocks enter MMGP's pinned pool
    runtime.pipe_models["llm_model"] = runtime.llm_model
    runtime.budgets["llm_model"] = budget
    return runtime


def _load_state_dict(weights_path: Path) -> dict:
    check_cancelled()
    if weights_path.suffix == ".safetensors":
        state_dict = {}
        with safe_open(str(weights_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                check_cancelled()
                state_dict[key] = f.get_tensor(key)
        return state_dict
    return torch.load(str(weights_path), map_location="cpu")


def _resolve_weights_path(model_path: Path) -> Path:
    # Prefer fp32 weights for stability/quality when available.
    preferred = model_path / "xmodel.safetensors"
    if preferred.exists():
        return preferred
    fallback = model_path / "model.safetensors"
    if fallback.exists():
        return fallback
    fallback = model_path / "pytorch_model.bin"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"No Florence2 weights found in {model_path} (expected model.safetensors/xmodel.safetensors/pytorch_model.bin)"
    )


def load_florence2(
    model_dir: str,
    attn_implementation: str = "sdpa",
) -> Tuple[Florence2ForConditionalGeneration, Florence2Processor]:
    model_path = Path(model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Florence2 folder not found: {model_path}")

    config = Florence2Config.from_pretrained(str(model_path))
    if attn_implementation:
        config._attn_implementation = attn_implementation
    weights_path = _resolve_weights_path(model_path)
    state_dict = _load_state_dict(weights_path)

    check_cancelled()
    model = Florence2ForConditionalGeneration(config)
    check_cancelled()
    load_info = model.load_state_dict(state_dict, strict=False)
    del state_dict
    check_cancelled()
    if load_info.missing_keys:
        allowed_missing = {
            "language_model.model.encoder.embed_tokens.weight",
            "language_model.model.decoder.embed_tokens.weight",
        }
        extra_missing = [k for k in load_info.missing_keys if k not in allowed_missing]
        if extra_missing:
            print(f"Florence2 missing keys: {extra_missing}")
    if load_info.unexpected_keys:
        print(f"Florence2 unexpected keys: {len(load_info.unexpected_keys)}")
    model.eval()

    image_processor = Florence2ImageProcessorLite.from_preprocessor_config(model_path)
    tokenizer = None
    tokenizer_errors = []
    for tok_cls in (BartTokenizerFast, BartTokenizer):
        try:
            tokenizer = tok_cls.from_pretrained(str(model_path))
            break
        except Exception as exc:
            tokenizer_errors.append(exc)
    if tokenizer is None:
        raise RuntimeError(f"Unable to load Florence2 tokenizer: {tokenizer_errors}")
    try:
        processor = Florence2Processor(image_processor=image_processor, tokenizer=tokenizer)
    except TypeError as exc:
        if "CLIPImageProcessor" not in str(exc):
            raise
        try:
            from transformers import CLIPImageProcessor
        except Exception:
            from transformers.models.clip import CLIPImageProcessor
        image_processor = CLIPImageProcessor.from_pretrained(str(model_path))
        processor = Florence2Processor(image_processor=image_processor, tokenizer=tokenizer)

    return model, processor
