# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import contextlib
import gc
import json
import os
import re
import sys
import types
from collections import OrderedDict

import numpy as np
import torch
from mmgp import offload, runtime
from tqdm.auto import tqdm

from shared.llm_io import known_token_ids, llm_io_enabled, log_llm_io
from shared.deepy.config import DEEPY_CONTEXT_TOKENS_KEY, DEEPY_REPETITION_PENALTY_DEFAULT, DEEPY_REPETITION_PENALTY_KEY, get_deepy_config_value, normalize_deepy_context_tokens, normalize_deepy_repetition_penalty
from shared.utils import files_locator as fl
from shared.llm_engines.lmgp import SamplingParams
from shared.llm_engines.lmgp.models.qwen3_5 import Qwen3_5ForCausalLM, clear_qwen35_runtime_caches
from shared.llm_engines.lmgp.layers.attention import reset_attention_backend_logs
from shared.llm_engines.lmgp.utils.context import reset_context
from shared.llm_engines.lmgp.vllm_support import TextEngine, resolve_prompt_enhancer_lm_features
from shared.qtypes.gguf import GGUFFirstRowsLinear, GGUFWeightTensor, materialize_module_source_tensors
try:
    from optimum.quanto.tensor.weights.qbytes import WeightQBytesTensor
except Exception:  # pragma: no cover
    WeightQBytesTensor = None

from .qwen3_5 import register_qwen35_config
from .qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from .qwen35_vl import (
    enhancer_quantization_GGUF,
    enhancer_quantization_QUANTO_INT8,
    _collect_suppressed_token_ids,
    _collect_stop_token_ids,
    _resolve_qwen35_checkpoint_file,
    _load_qwen35_tokenizer,
    _resolve_qwen35_asset_file,
    _resolve_qwen35_assets_dir,
    get_qwen35_variant_spec,
    get_qwen35_text_gguf_path,
)


# Debug only: comma-separated features the LLM engine must not use (graphs, triton, flash), e.g. WGP_LLM_DISABLE=graphs to compare with eager runs
QWEN35_LLM_DISABLE_ENV = "WGP_LLM_DISABLE"
QWEN35_LLM_PLANNED_CONTEXT = 32768  # tokens of KV cache planned beside the LLM's weights (MMGP cpuCompute pinning share)
QWEN35_LLM_MAX_PRELOAD_MB = 0  # tests of partial residency: most VRAM the LLM's weights may keep (base, blocks, slots of the streamed ones); 0: all that fit
QWEN35_LLM_PRELOAD_MARGIN_MB = 256  # VRAM the dynamic preload leaves free beyond the LLM's measured peaks, half of MMGP's default: its decode passes replay in CUDA graphs and vary less (16 GB cards near the edge: 35 -> 65 tk/s); None: MMGP's default
# the input token table, unless the head shares it, in reserved RAM read by the GPU in place (MMGP runtime.map_in_ram): out of VRAM, same outputs
QWEN35_LLM_MAPPED_TOKEN_TABLE = True
QWEN35_LLM_ACTIVATIONS_MB = 512  # speculative decoding Auto: VRAM the LLM's passes take beside its weights and KV cache (about 330 MB measured for the 27B)

QWEN35_GGUF_LLAMACPP_ENV = "WGP_GGUF_LLAMACPP_CUDA"
QWEN35_PROMPT_MIN_NEW_TOKENS = 4
QWEN35_PROMPT_MIN_MODEL_LEN = 8000
QWEN35_PROMPT_DEFAULT_TOP_K = 20
QWEN35_PROMPT_DEFAULT_MIN_P_GGUF = 0.05
QWEN35_PENALTY_MODE = "repetition"  # "none", "presence", or "repetition"
QWEN35_PREDICTIVE_PENALTY_ENABLED = False
QWEN35_PROMPT_PRESENCE_PENALTY = 1.5
QWEN35_PROMPT_REPETITION_PENALTY = 1.05
QWEN35_PROMPT_SUPPRESS_LOGITS_BIAS = -1e4
QWEN35_PROMPT_ENABLE_THINKING = False
QWEN35_PROMPT_THINKING_EXTRA_TOKENS = 3000
QWEN35_PROMPT_THINKING_MAX_TOKENS = 2000
# Applied only to variants that ship a native MTP block.
QWEN35_PROMPT_ENABLE_SPECULATIVE_DECODING = False
QWEN35_PROMPT_SPECULATIVE_TOKENS = 2
QWEN35_PROMPT_SPECULATIVE_SAMPLING_TOKENS = 2


def _env_enabled(name: str, default: bool = True) -> bool:
    raw = str(os.environ.get(name, "1" if default else "0")).strip().lower()
    return raw in ("1", "true", "yes", "y", "on")


def _is_mps_available() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def _resolve_gguf_model_path(model_path: str | None, assets_dir: str, variant: str | None = None) -> str:
    if model_path is not None:
        resolved = fl.locate_file(model_path, error_if_none=False) or model_path
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"Missing Qwen3.5 GGUF checkpoint: {resolved}")
        return resolved

    exact_path = get_qwen35_text_gguf_path(assets_dir, variant=variant)
    if os.path.isfile(exact_path):
        return exact_path
    raise FileNotFoundError(f"Missing expected Qwen3.5 GGUF checkpoint: {exact_path}")


def get_qwen35_text_assets_dir(assets_dir: str, variant: str | None = None) -> str:
    return _resolve_qwen35_assets_dir(assets_dir, variant=variant)


def get_qwen35_text_quanto_int8_path(assets_dir: str, variant: str | None = None) -> str:
    filename = get_qwen35_variant_spec(variant)["text_int8_filename"]
    return _resolve_qwen35_checkpoint_file(assets_dir, filename, variant=variant, error_if_none=False)


def _add_qwen35_mtp_shared_weights(state_dict, quantization_map=None, tied_weights_map=None):
    for source, target in (("token_embd", "mtp.embed_tokens"), ("output", "mtp.lm_head")):
        for name in tuple(state_dict):
            if name == source or name.startswith(source + "."):
                state_dict[target + name[len(source):]] = state_dict[name]
        if quantization_map is not None and source in quantization_map:
            quantization_map[target] = dict(quantization_map[source])
    return state_dict, quantization_map, tied_weights_map


def _resolve_gguf_linear_attention_layout_from_filename(model_path: str) -> tuple[bool, bool, bool]:
    filename = os.path.basename(str(model_path or "")).strip().lower().replace("_", "-")
    if filename in {
        "qwen3.5-9b-abliterated-text-q4-k-m-bis.gguf",
        "qwen3.8-27b-uncensored-q4-k-m.gguf",
        "qwen3.8-27b-uncensored-nomtp-iq3-s.gguf",
        "qwen3.8-27b-uncensored-iq2-m.gguf",
        "qwen3.8-9b-uncensored-heretic-q4-k-m.gguf",
        "qwen3.8-9b-uncensored-heretic-q8-0.gguf",
    }:
        return True, True, False
    if filename in {
        "qwen3.5-9b-abliterated-text-q4-k-m.gguf",
        "qwen3.5-4b-abliterated-text-q4-k-m.gguf",
    }:
        return False, True, False
    return False, False, True


def _qwen35_kv_bytes(config, kv_cache_layout: str, tokens: int) -> int:
    # the KV cache of the full-attention layers for tokens: BF16, or the body and exact overlays of a Deepy layout
    layers = sum(layer_type == "full_attention" for layer_type in config.layer_types)
    kv_heads, head_dim = int(config.num_key_value_heads), int(getattr(config, "head_dim", int(config.hidden_size) // int(config.num_attention_heads)))
    if not kv_cache_layout:
        return layers * 2 * kv_heads * head_dim * 2 * tokens
    from shared.llm_engines.lmgp.layers.kv_formats import parse_layout
    layout = parse_layout(kv_cache_layout)
    return layout.body_bytes_per_token(layers, kv_heads, head_dim) * tokens + layout.overlay_bytes(layers, kv_heads, head_dim, 256, 1)


def _qwen35_mtp_filename(spec, backend, model_path, prism_metadata):
    # the MTP drafter's own checkpoint, None when the text checkpoint holds it
    q3_filename = spec.get("text_gguf_q3_filename")
    uses_separate_q3_mtp = backend == enhancer_quantization_GGUF and q3_filename and os.path.basename(str(model_path or "")).lower() == q3_filename.lower()
    return spec["text_gguf_ptq1_mtp_filename"] if prism_metadata else spec.get("text_gguf_q3_mtp_filename" if uses_separate_q3_mtp else "text_mtp_filename")


def qwen35_mtp_fit(assets_dir: str | None, variant: str | None, backend: str, model_path: str | None, kv_cache_layout: str):
    """For speculative decoding Auto: (VRAM the text model needs without MTP, VRAM the MTP drafter adds, VRAM this process may hold), in
    bytes. The weights are the GGUF tensors the GPU holds (the token table is read in place from RAM), beside the KV cache of Deepy's
    context, the activations and the margins of MMGP's dynamic preload; the drafter adds its layer and its own BF16 KV cache. None when the
    checkpoint is not GGUF or not downloaded yet."""
    from mmgp import allocator, auto_preload, memory_pressure
    from shared.qtypes import gguf
    from shared.qtypes.prism import get_prism_metadata
    if backend != enhancer_quantization_GGUF:
        return None
    assets_dir = _resolve_qwen35_assets_dir(assets_dir, variant=variant)
    spec = get_qwen35_variant_spec(variant)
    try:
        model_path = _resolve_gguf_model_path(model_path, assets_dir, variant=variant)
    except FileNotFoundError:
        return None
    config_path = _resolve_qwen35_asset_file(assets_dir, "config.json", error_if_none=False) or os.path.join(assets_dir, "config.json")
    if not os.path.isfile(config_path):
        return None
    config = _load_text_config(config_path)
    drafter = lambda name: name.startswith(f"blk.{int(config.num_hidden_layers)}.") or name.startswith("mtp.")  # the MTP layer follows the model's
    tensors = [(info.name, info.n_bytes) for info in gguf._gguf_get_index(model_path).tensor_infos]
    in_ram = "token_embd.weight" if QWEN35_LLM_MAPPED_TOKEN_TABLE and not spec.get("tie_word_embeddings", False) else None
    weights = sum(nbytes for name, nbytes in tensors if not drafter(name) and name != in_ram)
    mtp = sum(nbytes for name, nbytes in tensors if drafter(name))
    mtp_filename = _qwen35_mtp_filename(spec, backend, model_path, get_prism_metadata(model_path))
    if mtp_filename:
        mtp_path = _resolve_qwen35_checkpoint_file(assets_dir, mtp_filename, variant=variant)
        if not mtp_path or not os.path.isfile(mtp_path):
            return None
        mtp += sum(info.n_bytes for info in gguf._gguf_get_index(mtp_path).tensor_infos if info.name != "token_embd.weight")
    tokens = normalize_deepy_context_tokens(get_deepy_config_value(DEEPY_CONTEXT_TOKENS_KEY))
    kv_heads, head_dim = int(config.num_key_value_heads), int(getattr(config, "head_dim", int(config.hidden_size) // int(config.num_attention_heads)))
    work = _qwen35_kv_bytes(config, kv_cache_layout, tokens) + (QWEN35_LLM_ACTIVATIONS_MB << 20)
    margin = (auto_preload.MARGIN if QWEN35_LLM_PRELOAD_MARGIN_MB is None else QWEN35_LLM_PRELOAD_MARGIN_MB << 20) + memory_pressure.HEADROOM + int(auto_preload.MARGIN_SHARE * work)
    if allocator.active == "vmm":
        available = allocator.stats()["reserved"] + allocator.room()
    else:
        available = torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved()
    return weights + work + margin, mtp + 2 * kv_heads * head_dim * 2 * tokens, available


def _map_qwen35_token_table(model) -> None:
    """MMGP keeps the input token table (and the MTP drafter's alias of it) in reserved RAM mapped in the GPU's address space: the lookups,
    inside the engine's graphs and with token ids sampled on the GPU too, read their rows over the bus and dequantize them with the same
    kernels, bit for bit, the table out of VRAM. A head tied to the table reads all of it at each pass: it stays in VRAM."""
    raw = lambda tensor: tensor._data if hasattr(tensor, "_data") else tensor
    if raw(model.output.weight).data_ptr() == raw(model.token_embd.weight).data_ptr():
        return
    runtime.map_in_ram(model.token_embd, *([] if model.mtp is None else [model.mtp.embed_tokens]))


def _load_text_config(config_path: str) -> Qwen3_5TextConfig:
    with open(config_path, "r", encoding="utf-8") as reader:
        config = json.load(reader)
    text_config = config.get("text_config", config)
    return Qwen3_5TextConfig(**text_config)

def _ensure_tied_output_weight(new_sd, tied_map):
    if "token_embd.weight" not in new_sd:
        return tied_map
    new_sd.pop("output.weight", None)
    tied_map = dict(tied_map or {})
    tied_list = list(tied_map.get("token_embd.weight", []))
    if "output.weight" not in tied_list:
        tied_list.append("output.weight")
    tied_map["token_embd.weight"] = tied_list
    return tied_map


def _build_qwen35_gguf_preprocess_sd(tie_output_to_embeddings: bool = False, num_hidden_layers: int | None = None, enable_mtp: bool = False):
    def preprocess_sd(sd, quant_map=None, tied_map=None):
        new_sd = OrderedDict()
        for name, tensor in sd.items():
            if name.startswith("mtp."):
                if enable_mtp:
                    new_sd[name] = tensor
                continue
            if name.startswith("v."):
                continue
            block_match = re.match(r"^blk\.(\d+)\.", name)
            if block_match and num_hidden_layers is not None and int(block_match.group(1)) >= num_hidden_layers:
                if not enable_mtp or int(block_match.group(1)) != num_hidden_layers:
                    continue
                suffix = name[block_match.end():]
                mtp_names = {
                    "nextn.eh_proj.weight": "mtp.fc.weight",
                    "nextn.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
                    "nextn.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
                    "nextn.shared_head_norm.weight": "mtp.norm.weight",
                }
                name = mtp_names.get(suffix, f"mtp.block.{suffix}")
            if name.endswith(".ssm_dt.bias"):
                name = name[:-5]
            if name.endswith(".ssm_conv1d.weight"):
                tensor = tensor.unsqueeze(1)
            new_sd[name] = tensor
        if tie_output_to_embeddings:
            tied_map = _ensure_tied_output_weight(new_sd, tied_map)
        return new_sd, quant_map, tied_map

    return preprocess_sd


def _resolve_quanto_log_ssm_a(model_path: str, spec: dict) -> bool:
    filename = os.path.basename(str(model_path or "")).strip().lower()
    log_ssm_a_filenames = {str(name).strip().lower() for name in spec.get("text_int8_log_ssm_a_filenames", [])}
    selected_filename = str(spec.get("text_int8_filename", "")).strip().lower()
    return filename in log_ssm_a_filenames or (bool(spec.get("text_int8_log_ssm_a", False)) and filename == selected_filename)


def _normalize_generated_text(text: str, *, keep_trailing_newlines: bool = False) -> str:
    text = str(text or "")
    text = text.replace("<|im_end|>", "").replace("<|im_start|>", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines, fence = [], ""
    for line in text.split("\n"):
        marker = re.match(r" {0,3}(`{3,}|~{3,})", line)
        if fence and not (marker and marker.group(1)[0] == fence[0] and len(marker.group(1)) >= len(fence) and not line.strip().strip(fence[0])):
            lines.append(line)  # code keeps its indentation and spacing
            continue
        fence = "" if fence else (marker.group(1) if marker else "")
        lines.append(re.sub(r"[ \t]+", " ", line).strip())
    text = "\n".join(lines)
    return text.lstrip() if keep_trailing_newlines else text.strip()


def _clean_answer_text(text: str, *, keep_trailing_newlines: bool = False) -> str:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"^(?:\s*</think>\s*)+", "", text, flags=re.IGNORECASE)
    text = text.replace("<think>", "\n").replace("</think>", "\n")
    text = re.sub(r"^\s*assistant\s*:?\s*", "", text, flags=re.IGNORECASE)
    parts = re.split(
        r"(?:<\|im_start\|>\s*(?:assistant|user|system|tool)\b|<tool_call>\s*|<tool_response>\s*|<tools>\s*|\n\s*(?:assistant|user|system|tool)\s*:)",
        text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )
    text = parts[0]
    cleaned_lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.lower() == "code interpreter":
            break
        cleaned_lines.append(line)
    return _normalize_generated_text("\n".join(cleaned_lines), keep_trailing_newlines=keep_trailing_newlines and len(parts) == 1)


def _split_generated_parts(text: str, *, keep_trailing_newlines: bool = False) -> tuple[list[str], str]:
    text = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
    think_chunks = []
    answer_parts = []
    cursor = 0
    first_open = re.search(r"<think>", text, flags=re.IGNORECASE)
    first_close = re.search(r"</think>", text, flags=re.IGNORECASE)
    if first_close is not None and (first_open is None or first_close.start() < first_open.start()):
        leading_reasoning = _normalize_generated_text(text[:first_close.start()].replace("<think>", "\n"))
        if len(leading_reasoning) > 0:
            think_chunks.append(leading_reasoning)
        cursor = first_close.end()
    remaining = text[cursor:]
    explicit_cursor = 0
    for match in re.finditer(r"<think>\s*(.*?)\s*</think>", remaining, flags=re.DOTALL | re.IGNORECASE):
        answer_parts.append(remaining[explicit_cursor:match.start()])
        if len(match.group(1).strip()) > 0:
            think_chunks.append(match.group(1))
        explicit_cursor = match.end()
    trailing = remaining[explicit_cursor:]
    unmatched_open = re.search(r"<think>\s*(.*)\Z", trailing, flags=re.DOTALL | re.IGNORECASE)
    if unmatched_open is not None:
        answer_parts.append(trailing[:unmatched_open.start()])
        if len(unmatched_open.group(1).strip()) > 0:
            think_chunks.append(unmatched_open.group(1))
    else:
        answer_parts.append(trailing)
    answer_text = "\n".join(answer_parts)
    if len(think_chunks) <= 1:
        close_matches = list(re.finditer(r"</think>", answer_text, flags=re.IGNORECASE))
        if close_matches:
            trailing_text = answer_text[close_matches[-1].end():]
            trailing_preview = re.sub(r"(?:<\|im_end\|>\s*|</s>\s*)+$", "", trailing_text, flags=re.IGNORECASE).lstrip()
            if len(trailing_preview) == 0 or trailing_preview.lower().startswith("<tool_call>"):
                recovered_reasoning = _normalize_generated_text(answer_text[:close_matches[-1].start()].replace("<think>", "\n"))
                if len(recovered_reasoning) > 0:
                    think_chunks.append(recovered_reasoning)
                answer_text = trailing_text
    if len(think_chunks) == 0:
        timeline_match = re.search(r"(?mi)^\(at\s+[0-9]+(?:\.[0-9]+)?\s+seconds?\s*:", text)
        if timeline_match is not None:
            leading_text = _normalize_generated_text(text[:timeline_match.start()])
            if leading_text.lower().startswith("thinking process"):
                think_chunks.append(leading_text)
                answer_text = text[timeline_match.start():]
    return [normalized for chunk in think_chunks if len(normalized := _normalize_generated_text(chunk)) > 0], _clean_answer_text(answer_text, keep_trailing_newlines=keep_trailing_newlines)


def _split_generated_text(text: str, *, keep_trailing_newlines: bool = False) -> tuple[str, str]:
    think_chunks, answer_text = _split_generated_parts(text, keep_trailing_newlines=keep_trailing_newlines)
    return _normalize_generated_text("\n\n".join(think_chunks)), answer_text


def _clean_generated_text(text: str) -> str:
    _thinking_text, answer_text = _split_generated_text(text)
    return answer_text


def _prompt_enhancer_thinking_enabled(model, thinking_enabled: bool | None = None) -> bool:
    if thinking_enabled is not None:
        return bool(thinking_enabled)
    return bool(getattr(model, "_prompt_enhancer_enable_thinking", QWEN35_PROMPT_ENABLE_THINKING))


def set_qwen35_prompt_enhancer_thinking(model, enabled: bool) -> None:
    model._prompt_enhancer_enable_thinking = bool(enabled)
    tokenizer = getattr(model, "_prompt_enhancer_tokenizer", None)
    if tokenizer is None:
        return
    base_suppress_token_ids = tuple(_collect_suppressed_token_ids(tokenizer))
    model._prompt_enhancer_base_suppress_token_ids = base_suppress_token_ids
    model._prompt_enhancer_suppress_token_ids = [] if model._prompt_enhancer_enable_thinking else list(base_suppress_token_ids)
    model._prompt_enhancer_suppress_logits_bias = None


def _resolve_prompt_runtime_extra_tokens(model, thinking_enabled: bool | None = None) -> int:
    if not _prompt_enhancer_thinking_enabled(model, thinking_enabled=thinking_enabled):
        return 0
    extra_tokens = getattr(model, "_prompt_enhancer_thinking_extra_tokens", QWEN35_PROMPT_THINKING_EXTRA_TOKENS)
    try:
        return max(0, int(extra_tokens))
    except Exception:
        return 0


def _print_thinking_process(message_index: int, total_messages: int, thinking_text: str) -> None:
    if len(thinking_text) == 0:
        return
    print(f"[Qwen3.5VL][Thinking {message_index + 1}/{total_messages}]")
    try:
        print(thinking_text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
        safe_text = thinking_text.encode(encoding, errors="replace").decode(encoding, errors="replace")
        sys.stdout.write(safe_text + "\n")
        sys.stdout.flush()


class _PresencePenaltyState:
    def __init__(self, presence_penalty: float | None):
        if presence_penalty is None or float(presence_penalty) <= 0:
            self.penalty = None
        else:
            self.penalty = float(presence_penalty)
        self._seen_token_ids = set()
        self._bias_cache = {}

    def enabled(self) -> bool:
        return self.penalty is not None

    def update(self, token_id: int) -> None:
        if self.penalty is None:
            return
        token_id = int(token_id)
        if token_id < 0 or token_id in self._seen_token_ids:
            return
        self._seen_token_ids.add(token_id)
        for (vocab_size, _, _), bias in self._bias_cache.items():
            if token_id < vocab_size:
                bias[token_id] = bias.new_tensor(-self.penalty)

    def apply_(self, logits: torch.Tensor) -> torch.Tensor:
        if self.penalty is None or len(self._seen_token_ids) == 0:
            return logits
        cache_key = (logits.shape[-1], logits.device, logits.dtype)
        bias = self._bias_cache.get(cache_key)
        if bias is None:
            bias = logits.new_zeros(logits.shape[-1])
            valid_token_ids = [token_id for token_id in self._seen_token_ids if token_id < logits.shape[-1]]
            if len(valid_token_ids) > 0:
                bias.index_fill_(0, torch.tensor(valid_token_ids, device=logits.device, dtype=torch.long), bias.new_tensor(-self.penalty))
            self._bias_cache[cache_key] = bias
        logits.add_(bias)
        return logits


class _ThinkingBudgetState:
    def __init__(self, close_think_token_id: int | None, max_thinking_tokens: int | None, stop_token_ids: list[int] | tuple[int, ...] | None = None):
        try:
            close_think_token_id = int(close_think_token_id)
        except (TypeError, ValueError):
            close_think_token_id = -1
        try:
            max_thinking_tokens = int(max_thinking_tokens)
        except (TypeError, ValueError):
            max_thinking_tokens = 0
        self.close_think_token_id = close_think_token_id
        self.max_thinking_tokens = max_thinking_tokens
        self.in_thinking = close_think_token_id >= 0 and max_thinking_tokens > 0
        self.generated_thinking_tokens = 0
        self.stop_token_ids = tuple(
            token_id
            for token_id in {int(token_id) for token_id in tuple(stop_token_ids or ()) if int(token_id) >= 0}
            if token_id != self.close_think_token_id
        )
        self._stop_index_cache = {}
        self._close_index_cache = {}

    @staticmethod
    def _token_index(logits: torch.Tensor, token_ids: tuple[int, ...], cache: dict) -> torch.Tensor:
        cache_key = (logits.shape[-1], logits.device)
        index = cache.get(cache_key)
        if index is None:
            index = torch.tensor([token_id for token_id in token_ids if token_id < logits.shape[-1]], dtype=torch.long, device=logits.device)
            cache[cache_key] = index
        return index

    def enabled(self) -> bool:
        return self.in_thinking

    def update(self, token_id: int) -> None:
        if not self.in_thinking:
            return
        token_id = int(token_id)
        if token_id == self.close_think_token_id:
            self.in_thinking = False
            return
        self.generated_thinking_tokens += 1

    def apply_(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.in_thinking:
            return logits
        if self.generated_thinking_tokens < self.max_thinking_tokens:
            if self.stop_token_ids:
                blocked_ids = self._token_index(logits, self.stop_token_ids, self._stop_index_cache)
                if blocked_ids.numel():
                    logits.index_fill_(-1, blocked_ids, float("-inf"))
            return logits
        logits.fill_(float("-inf"))
        close_id = self._token_index(logits, (self.close_think_token_id,), self._close_index_cache)
        logits.index_fill_(-1, close_id, 0)
        return logits




def _build_chat_prompt(tokenizer, message, enable_thinking: bool = False):
    text = tokenizer.apply_chat_template(
        message,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=bool(enable_thinking),
    )
    return text.rstrip() + "\n"


def _resolve_prompt_penalty_mode(model) -> str:
    mode = str(getattr(model, "_prompt_enhancer_penalty_mode", QWEN35_PENALTY_MODE)).strip().lower()
    if mode not in {"none", "presence", "repetition"}:
        raise ValueError(f"Unknown Qwen3.5 penalty mode: {mode}")
    return mode


def _resolve_prompt_presence_penalty(model) -> float | None:
    if _resolve_prompt_penalty_mode(model) != "presence":
        return None
    presence_penalty = getattr(model, "_prompt_enhancer_presence_penalty", QWEN35_PROMPT_PRESENCE_PENALTY)
    if presence_penalty is None:
        return None
    presence_penalty = float(presence_penalty)
    return presence_penalty if presence_penalty > 0 else None


def _resolve_prompt_repetition_penalty(model) -> float:
    if _resolve_prompt_penalty_mode(model) != "repetition":
        return 1.0
    repetition_penalty = float(getattr(model, "_prompt_enhancer_repetition_penalty", QWEN35_PROMPT_REPETITION_PENALTY))
    return repetition_penalty if repetition_penalty > 0 else 1.0


def _resolve_predictive_penalty_enabled(model) -> bool:
    return bool(getattr(model, "_prompt_enhancer_predictive_penalty_enabled", QWEN35_PREDICTIVE_PENALTY_ENABLED))


def _build_presence_penalty_logits_processor(presence_penalty: float | None):
    state = _PresencePenaltyState(presence_penalty)
    if not state.enabled():
        return None, None

    def logits_processor(_input_ids, logits):
        state.apply_(logits)
        return logits

    logits_processor._requires_input_ids = False

    def update_state(token_id: int):
        state.update(token_id)

    return logits_processor, update_state


def _build_prompt_logits_processor(model, thinking_enabled: bool | None = None, max_thinking_tokens_override: int | None = None, suppress_token_ids: tuple[int, ...] = ()):
    processors = []
    processors_without_penalty = []
    update_callbacks = []
    thinking_state = None

    suppressed_ids = tuple(dict.fromkeys(int(token_id) for token_id in suppress_token_ids if int(token_id) >= 0))
    if suppressed_ids:
        suppressed_index_cache = {}

        def suppress_tokens_logits_processor(_input_ids, logits):
            cache_key = (logits.shape[-1], logits.device)
            valid_ids = suppressed_index_cache.get(cache_key)
            if valid_ids is None:
                valid_ids = torch.tensor([token_id for token_id in suppressed_ids if token_id < logits.shape[-1]], dtype=torch.long, device=logits.device)
                suppressed_index_cache[cache_key] = valid_ids
            if valid_ids.numel():
                logits.index_fill_(-1, valid_ids, float("-inf"))
            return logits

        suppress_tokens_logits_processor._is_token_mask = True
        processors.append(suppress_tokens_logits_processor)
        processors_without_penalty.append(suppress_tokens_logits_processor)

    presence_processor, presence_update_state = _build_presence_penalty_logits_processor(_resolve_prompt_presence_penalty(model))
    if presence_processor is not None:
        processors.append(presence_processor)
    if presence_update_state is not None:
        update_callbacks.append(presence_update_state)

    if _prompt_enhancer_thinking_enabled(model, thinking_enabled=thinking_enabled):
        thinking_max_tokens = max_thinking_tokens_override if max_thinking_tokens_override is not None else getattr(model, "_prompt_enhancer_thinking_max_tokens", QWEN35_PROMPT_THINKING_MAX_TOKENS)
        thinking_state = _ThinkingBudgetState(
            getattr(model, "_prompt_enhancer_close_think_token_id", None),
            thinking_max_tokens,
            getattr(model, "_prompt_enhancer_stop_token_ids", ()),
        )
        if thinking_state.enabled():
            def thinking_logits_processor(_input_ids, logits):
                return thinking_state.apply_(logits)

            thinking_logits_processor._is_token_mask = True
            processors.append(thinking_logits_processor)
            processors_without_penalty.append(thinking_logits_processor)
            update_callbacks.append(thinking_state.update)

    if not processors:
        return None, None

    if len(processors) == 1:
        logits_processor = processors[0]
    else:
        def logits_processor(input_ids, logits):
            for processor in processors:
                logits = processor(input_ids, logits)
            return logits

    if not processors_without_penalty:
        logits_processor_without_penalty = None
    elif len(processors_without_penalty) == 1:
        logits_processor_without_penalty = processors_without_penalty[0]
    else:
        def logits_processor_without_penalty(input_ids, logits):
            for processor in processors_without_penalty:
                logits = processor(input_ids, logits)
            return logits
    logits_processor._without_penalty = logits_processor_without_penalty
    logits_processor._is_token_mask = all(getattr(processor, "_is_token_mask", False) for processor in processors)
    logits_processor._requires_input_ids = False
    logits_processor._supports_partial_vocab = lambda: thinking_state is None or not thinking_state.in_thinking or thinking_state.generated_thinking_tokens < thinking_state.max_thinking_tokens
    if logits_processor_without_penalty is not None:
        logits_processor_without_penalty._is_token_mask = True
        logits_processor_without_penalty._requires_input_ids = False
        logits_processor_without_penalty._supports_partial_vocab = logits_processor._supports_partial_vocab

    if presence_processor is None:
        # A verifier can evaluate these rules against each hypothetical accepted
        # prefix on the GPU. Python state is updated only for committed tokens.
        def speculative_batch_rules():
            return {
                "suppressed": suppressed_ids,
                "thinking_stops": () if thinking_state is None else thinking_state.stop_token_ids,
                "thinking": (-1, 0, 0) if thinking_state is None else (
                    thinking_state.close_think_token_id,
                    thinking_state.max_thinking_tokens - thinking_state.generated_thinking_tokens,
                    int(thinking_state.in_thinking),
                ),
            }
        logits_processor._speculative_batch_rules = speculative_batch_rules

    if len(update_callbacks) == 1:
        update_state = update_callbacks[0]
    else:
        def update_state(token_id: int):
            for callback in update_callbacks:
                callback(token_id)

    return logits_processor, update_state


def _build_suppressed_token_logits_bias(model, thinking_enabled: bool | None = None):
    if _prompt_enhancer_thinking_enabled(model, thinking_enabled=thinking_enabled):
        return None
    suppress_token_ids = tuple(int(token_id) for token_id in tuple(getattr(model, "_prompt_enhancer_base_suppress_token_ids", ()) or ()) if int(token_id) >= 0)
    vocab_size = int(getattr(getattr(model, "config", None), "vocab_size", 0) or 0)
    if vocab_size <= 0 or len(suppress_token_ids) == 0:
        return None
    valid_token_ids = tuple(token_id for token_id in suppress_token_ids if token_id < vocab_size)
    if len(valid_token_ids) == 0:
        return None
    cache = getattr(model, "_prompt_enhancer_suppress_logits_bias_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        model._prompt_enhancer_suppress_logits_bias_cache = cache
    cache_key = (vocab_size, valid_token_ids)
    cached_bias = cache.get(cache_key)
    if torch.is_tensor(cached_bias):
        return cached_bias
    bias = torch.zeros(vocab_size, dtype=torch.float32)
    bias[torch.tensor(valid_token_ids, dtype=torch.long)] = float(QWEN35_PROMPT_SUPPRESS_LOGITS_BIAS)
    cache[cache_key] = bias
    return bias


def _normalize_vllm_sampling(do_sample, temperature, top_p, top_k):
    if not do_sample:
        return 1.0, None, 1
    if temperature is None or float(temperature) <= 0:
        return 1.0, None, 1
    normalized_top_p = top_p if top_p is not None and 0.0 < float(top_p) < 1.0 else None
    normalized_top_k = int(top_k) if top_k is not None and int(top_k) > 0 else None
    return float(temperature), normalized_top_p, normalized_top_k


def _resolve_prompt_top_k(model, top_k):
    if top_k is not None:
        resolved_top_k = int(top_k)
        return resolved_top_k if resolved_top_k > 0 else None
    default_top_k = getattr(model, "_prompt_enhancer_default_top_k", QWEN35_PROMPT_DEFAULT_TOP_K)
    if default_top_k is None:
        return None
    default_top_k = int(default_top_k)
    return default_top_k if default_top_k > 0 else None


def _resolve_prompt_min_p(model):
    min_p = getattr(model, "_prompt_enhancer_default_min_p", None)
    if min_p is None:
        return None
    min_p = float(min_p)
    return min_p if min_p > 0.0 else None


def _resolve_llm_features(backend: str) -> dict:
    """What the LLM engine uses: the LM decoder optimizations the installation supports, each on its own (resolve_lm_features: CUDA
    graphs, MMGP's, the weights stay under its management; the Triton kernels: fused norms, activations, cache writes, GDN, sampling;
    FlashAttention 2), and the GGUF CUDA kernels."""
    lm = resolve_prompt_enhancer_lm_features()
    if backend == enhancer_quantization_GGUF:
        from shared.qtypes.gguf import _gguf_cuda_module
        native = _gguf_cuda_module()
        gguf_ok, gguf_reason = native is not None, "ok" if native is not None else ("disabled by " + QWEN35_GGUF_LLAMACPP_ENV if not _env_enabled(QWEN35_GGUF_LLAMACPP_ENV) else "not installed")
    else:
        gguf_ok, gguf_reason = False, "not a GGUF checkpoint"
    return dict(cuda=torch.cuda.is_available() and not _is_mps_available(), graphs=lm.graphs, graphs_reason=lm.graphs_reason, triton=lm.triton,
                triton_reason=lm.triton_reason, flash=lm.flash, flash_reason=lm.flash_reason, gguf=gguf_ok, gguf_reason=gguf_reason,
                gguf_version=getattr(native, "__version__", "") if gguf_ok else "")


def _qwen35_preload_workload(args, kwargs) -> str:
    """The two workloads of the dynamic preload (block arguments: [hidden states, residual], ...): the decoding passes (a token, or a
    speculative verification of a few) and the prompt chunks, whose activations take more VRAM; each keeps its own measure."""
    hidden_states = args[0][0]
    return "decode" if hidden_states.shape[-2] <= 64 else "prefill"


def _describe_llm_features(features: dict, kv_cache_int8: bool, speculative: str, kv_cache_layout: str = "int8") -> str:
    def item(name, used, reason=""):
        return name if used else f"no {name} ({reason})"
    parts = [item("CUDA graphs", features["graphs"], features["graphs_reason"]),
             item("Triton fused kernels", features["triton"], features["triton_reason"])]
    if kv_cache_int8:
        from shared.deepy.config import describe_deepy_kv_cache_quantization
        from shared.llm_engines.lmgp.layers.kv_formats import parse_layout
        layout = parse_layout(kv_cache_layout or "int8")
        exact = f" with INT8 sink and last {layout.recent_tokens} tokens" if layout.overlay else ""
        parts.append(f"{describe_deepy_kv_cache_quantization(layout.value)} KV cache{exact}" + (" (Triton paged attention)" if features["triton"] else " (dequantized attention)"))
    else:
        parts.append(item("FlashAttention 2", features["flash"], features["flash_reason"]))
    if features["gguf"]:
        parts.append(f"GGUF CUDA kernels {features['gguf_version']}".rstrip())
    elif features["gguf_reason"] != "not a GGUF checkpoint":
        parts.append(f"no GGUF CUDA kernels ({features['gguf_reason']})")
    if speculative:
        parts.append(speculative)
    return ", ".join(parts)


def _use_vllm_prompt_enhancer(model) -> bool:
    return bool(getattr(model, "_prompt_enhancer_use_vllm", False)) and torch.cuda.is_available()


def _use_legacy_cuda_runner_prompt_enhancer(model) -> bool:
    return bool(getattr(model, "_prompt_enhancer_use_legacy_cuda_runner", False)) and (torch.cuda.is_available() or _is_mps_available())


def _get_assistant_graph_pool_handle(model, usage_mode: str | None, enable_cudagraph: bool):
    if usage_mode not in ("assistant", "multimodal") or not enable_cudagraph or not torch.cuda.is_available():
        return None
    handle = getattr(model, "_prompt_enhancer_assistant_graph_pool_handle", None)
    if handle is None:
        try:
            handle = torch.cuda.graph_pool_handle()
        except Exception:
            handle = None
        model._prompt_enhancer_assistant_graph_pool_handle = handle
    return handle


def _get_or_create_vllm_engine(model, usage_mode: str | None = None):
    register_qwen35_config()
    engine = getattr(model, "_prompt_enhancer_vllm_engine", None)
    active_mode = getattr(model, "_prompt_enhancer_vllm_mode", None)
    if engine is not None and usage_mode is not None and active_mode not in (None, usage_mode):
        try:  # its close saves the live context of an owner away (Deepy parked)
            engine.close()
        except Exception:
            pass
        engine = None
        model._prompt_enhancer_vllm_engine = None
    if engine is not None:
        if usage_mode is not None:
            model._prompt_enhancer_vllm_mode = usage_mode
        return engine

    runtime_model_path = getattr(model, "_prompt_enhancer_vllm_model_path", None)
    tokenizer = getattr(model, "_prompt_enhancer_tokenizer", None)
    if not runtime_model_path:
        raise RuntimeError("Qwen3.5 prompt enhancer vLLM runtime path is not configured.")
    if tokenizer is None:
        raise RuntimeError("Qwen3.5 prompt enhancer tokenizer is not configured.")
    enable_cudagraph = bool(getattr(model, "_prompt_enhancer_enable_cudagraph", False))
    graph_pool_handle = _get_assistant_graph_pool_handle(model, usage_mode, enable_cudagraph)

    engine = TextEngine(
        model=model,
        model_path=runtime_model_path,
        tokenizer=tokenizer,
        enforce_eager=not enable_cudagraph,
        graph_pool_handle=graph_pool_handle,
        kv_cache_int8=bool(getattr(model, "_deepy_kv_cache_int8", False)),
        kv_cache_layout=str(getattr(model, "_deepy_kv_cache_layout", "") or "int8"),
    )
    model._prompt_enhancer_vllm_engine = engine
    model._prompt_enhancer_vllm_mode = usage_mode
    return engine


def _reset_vllm_sequence_state(model):
    for module in model.modules():
        reset_sequence_state = getattr(module, "reset_sequence_state", None)
        if callable(reset_sequence_state):
            reset_sequence_state()
            continue
        if hasattr(module, "conv_state"):
            module.conv_state = None
        if hasattr(module, "recurrent_state"):
            module.recurrent_state = None


def _generate_messages_vllm(
    self,
    messages,
    max_new_tokens,
    do_sample=True,
    temperature=None,
    top_p=None,
    top_k=None,
    seed=None,
    thinking_enabled: bool | None = None,
    stop_requested=None,
    stream_callback=None,
    enhancement_progress=None,
):
    from .qwen35_prompt_decode import generate_prompt

    reset_context()
    tokenizer = self._prompt_enhancer_tokenizer
    if len(messages) == 0:
        return []

    engine = getattr(self, "_prompt_enhancer_vllm_engine", None)
    # an assistant's context in the engine (Deepy kept loaded): each prompt runs beside it, in the free pages of its cache (LLMEngine.aside)
    beside = engine is not None and engine.has_live_context()
    if not beside:
        engine = _get_or_create_vllm_engine(self, usage_mode="text")
    outputs = []
    apply_repetition_penalty = normalize_deepy_repetition_penalty(get_deepy_config_value(DEEPY_REPETITION_PENALTY_KEY, DEEPY_REPETITION_PENALTY_DEFAULT))
    thinking_enabled = _prompt_enhancer_thinking_enabled(self, thinking_enabled=thinking_enabled)
    runtime_extra_tokens = _resolve_prompt_runtime_extra_tokens(self, thinking_enabled=thinking_enabled)
    progress_desc = "Qwen3.5 prompt enhancement"
    for idx, message in enumerate(tqdm(messages, total=len(messages), desc=progress_desc, dynamic_ncols=True, leave=False)):
        if enhancement_progress is not None:
            enhancement_progress.prompt(idx, len(messages), int(max_new_tokens) + runtime_extra_tokens)
            stream_callback = enhancement_progress.tokens
        prompt = _build_chat_prompt(tokenizer, message, enable_thinking=thinking_enabled)
        try:
            prompt_token_ids = [int(token_id) for token_id in tokenizer.encode(prompt)]
            prompt_len = len(prompt_token_ids)
        except Exception:
            prompt_token_ids = []
            prompt_len = 0
        generation_max_tokens = int(max_new_tokens) + runtime_extra_tokens
        if beside and not engine._llm.can_aside(prompt_len + generation_max_tokens):
            engine.save_context()  # no room beside it: its owner saves it to RAM (restored at its next use), the prompts run as alone
            beside = False
            engine = _get_or_create_vllm_engine(self, usage_mode="text")
        if not beside:
            # its own minimum: Deepy, which shares the model, leaves its context window in _prompt_enhancer_min_model_len_hint
            engine.reserve_runtime(prompt_len=prompt_len, max_tokens=generation_max_tokens, cfg_scale=1.0, min_model_len=QWEN35_PROMPT_MIN_MODEL_LEN)
            engine._ensure_llm()
        if engine._llm is None:
            raise RuntimeError("Qwen3.5 prompt enhancer vLLM runtime is not available.")
        temp, normalized_top_p, normalized_top_k = _normalize_vllm_sampling(
            do_sample=bool(do_sample),
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )
        logits_bias = _build_suppressed_token_logits_bias(self, thinking_enabled=thinking_enabled)
        sample_seed = None if seed is None else int(seed) + idx
        logits_processor, logits_processor_update_state = _build_prompt_logits_processor(self, thinking_enabled=thinking_enabled)
        sampling_params = SamplingParams(
            temperature=temp,
            max_tokens=generation_max_tokens,
            cfg_scale=1.0,
            top_k=normalized_top_k,
            top_p=normalized_top_p,
            min_p=_resolve_prompt_min_p(self),
            repetition_penalty=_resolve_prompt_repetition_penalty(self) if apply_repetition_penalty else 1.0,
            predictive_penalty=_resolve_predictive_penalty_enabled(self),
            ignore_eos=False,
            logits_processor=logits_processor,
            logits_processor_update_state=logits_processor_update_state,
            logits_bias=logits_bias,
            seed=sample_seed,
        )

        if llm_io_enabled():
            log_llm_io("OUT", "local-prompt-enhancer", "qwen-generation", {
                "prompt": prompt,
                "messages": message,
                "input_token_ids": prompt_token_ids,
                "known_token_ids": known_token_ids(tokenizer),
                "generation": {"max_new_tokens": generation_max_tokens, "temperature": temp, "top_p": normalized_top_p, "top_k": normalized_top_k, "seed": sample_seed, "thinking_enabled": thinking_enabled},
            }, prompt_number=idx + 1)

        try:
            with engine._llm.aside() if beside else contextlib.nullcontext():
                response = generate_prompt(
                    engine._llm,
                    prompt,
                    sampling_params=sampling_params,
                    stop_token_ids=self._prompt_enhancer_stop_token_ids,
                    close_think_token_id=self._prompt_enhancer_close_think_token_id if thinking_enabled else None,
                    max_thinking_tokens=self._prompt_enhancer_thinking_max_tokens,
                    stop_requested=stop_requested,
                    stream_callback=stream_callback,
                )
            raw_text = response["text"]
            log_llm_io("IN", "local-prompt-enhancer", "qwen-generation", response, prompt_number=idx + 1)
            # The template supplies the opening <think>; generated text starts
            # inside reasoning even when it contains no explicit opening tag.
            thinking_open = thinking_enabled
            for marker in re.finditer(r"</?think>", raw_text, flags=re.IGNORECASE):
                thinking_open = marker.group().lower() == "<think>"
            if thinking_open:
                raise RuntimeError("Prompt Enhancement Ended With Unfinished Thinking. Please Try Again.")
            if response["stop_reason"] != "stop_token":
                raise RuntimeError(f"Prompt Enhancement Did Not Finish ({response['stop_reason']}). Please Try Again.")
            thinking_text, answer_text = _split_generated_text(raw_text)
            if not answer_text.strip():
                raise RuntimeError("Prompt Enhancement Returned No Final Prompt. Please Try Again.")
            engine._last_failure_reason = ""
        except BaseException as exc:
            engine._last_failure_reason = str(exc)
            # Discard failed/cancelled runtime state before another enhancement (beside a live context, aside() already discarded the
            # prompt's own and gave the context back)
            if not beside:
                try:
                    _unload_prompt_enhancer_text_runtime(self)
                except Exception as cleanup_error:
                    tqdm.write(f"Prompt Enhancer Cleanup Failed: {cleanup_error}")
            raise
        finally:
            reset_context()

        if thinking_enabled:
            _print_thinking_process(idx, len(messages), thinking_text)
        outputs.append(answer_text)
    return outputs


def _generate_messages(
    self,
    messages,
    max_new_tokens,
    do_sample=True,
    temperature=None,
    top_p=None,
    top_k=None,
    seed=None,
    thinking_enabled: bool | None = None,
    stop_requested=None,
    stream_callback=None,
    enhancement_progress=None,
):
    top_k = _resolve_prompt_top_k(self, top_k)
    if _use_vllm_prompt_enhancer(self) or _use_legacy_cuda_runner_prompt_enhancer(self):
        return _generate_messages_vllm(
            self,
            messages,
            max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            thinking_enabled=thinking_enabled,
            stop_requested=stop_requested,
            stream_callback=stream_callback,
            enhancement_progress=enhancement_progress,
        )
    raise RuntimeError("Qwen3.5 prompt enhancer text runtime is not configured with an available decode engine.")


def _unload_prompt_enhancer_text_runtime(self):
    engine = getattr(self, "_prompt_enhancer_vllm_engine", None)
    if engine is not None:
        try:
            engine.close()
        finally:
            self._prompt_enhancer_vllm_engine = None
            self._prompt_enhancer_vllm_mode = None
    self._prompt_enhancer_assistant_graph_pool_handle = None
    try:
        clear_qwen35_runtime_caches()
    except Exception:
        pass
    reset_context()
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def _load_local_text_model(
    model_path: str,
    config_path: str,
    preprocess_sd=None,
    default_dtype: torch.dtype = torch.float16,
    safe_legacy_mode: bool = False,
    materialize_source_tensors: bool = True,
    enable_mtp: bool = False,
    modules=None,
    postprocess_sd=None,
    preserve_native_dtypes=False,
    flash_attention=None,
):
    config = _load_text_config(config_path)
    config._prompt_enhancer_safe_legacy = bool(safe_legacy_mode)
    if flash_attention is not None:
        config._prompt_enhancer_flash_attention = bool(flash_attention)
    config._prompt_enhancer_enable_mtp_speculative = bool(enable_mtp)
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(config)

    load_options = {}
    if preserve_native_dtypes:
        from shared.qtypes.prism import preserve_checkpoint_dtypes
        load_options["pre_load_callback"] = preserve_checkpoint_dtypes

    offload.load_model_data(
        model,
        model_path,
        preprocess_sd=preprocess_sd,
        postprocess_sd=postprocess_sd,
        modules=modules,
        writable_tensors=False,
        default_dtype=default_dtype,
        **load_options,
    )
    if materialize_source_tensors:
        materialize_module_source_tensors(model)
    return model


def _tie_qwen35_output_to_embeddings(model: torch.nn.Module) -> None:
    token_embd = getattr(model, "token_embd", None)
    output = getattr(model, "output", None)
    if token_embd is None or output is None or not hasattr(token_embd, "weight") or not hasattr(output, "weight"):
        raise RuntimeError("Cannot tie Qwen3.5 output head to token embeddings.")
    if output.weight is not token_embd.weight:
        output.weight = token_embd.weight


def _resolve_legacy_text_execution_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    if _is_mps_available():
        return torch.device("mps")
    raise RuntimeError("Qwen3.5 legacy prompt enhancement requires CUDA or MPS.")


def _configure_qwen35_text_model(
    model,
    model_dtype: torch.dtype | None = None,
    *,
    v_head_reordered: bool = False,
    ssm_param_reordered: bool = False,
    interleave_ssm_ab: bool = False,
    log_ssm_a: bool = False,
):
    for module in model.modules():
        if getattr(module, "layer_type", None) == "linear_attention" and hasattr(module, "_gguf_interleave_ssm_ab"):
            module._gguf_interleave_ssm_ab = bool(interleave_ssm_ab)
        if getattr(module, "layer_type", None) == "linear_attention" and hasattr(module, "_gguf_v_head_reordered"):
            module._gguf_v_head_reordered = bool(v_head_reordered)
        if getattr(module, "layer_type", None) == "linear_attention" and hasattr(module, "_gguf_ssm_param_reordered"):
            module._gguf_ssm_param_reordered = bool(ssm_param_reordered)
        if getattr(module, "layer_type", None) == "linear_attention" and hasattr(module, "_log_ssm_a"):
            module._log_ssm_a = bool(log_ssm_a)
    if model_dtype is not None:
        model.config.dtype = model_dtype
        model.config.torch_dtype = model_dtype


class _RowsSpan:
    # the memory of tensors that lie one after the other, for numpy, which keeps this object as the base of its array: the tensors stay alive
    def __init__(self, tensors, nbytes):
        self.tensors = tensors
        self.__array_interface__ = {"version": 3, "shape": (nbytes,), "typestr": "|u1", "data": (tensors[0].data_ptr(), False)}


def _cat_rows(tensors):
    # torch.cat along the rows, as a view when the tensors lie one after the other in memory (consecutive tensors of a checkpoint mapping,
    # such as ffn_gate and ffn_up in GGUF files): the fused weight then stays in the checkpoint file instead of private RAM, which the system
    # would write to the page file under memory pressure
    end = tensors[0].data_ptr()
    for tensor in tensors:
        if tensor.device.type != "cpu" or not tensor.is_contiguous() or tensor.dtype != tensors[0].dtype or tensor.data_ptr() != end:
            return torch.cat(tensors, dim=0)
        end += tensor.numel() * tensor.element_size()
    span = np.asarray(_RowsSpan(tensors, end - tensors[0].data_ptr()))
    return torch.from_numpy(span).view(tensors[0].dtype).view(-1, *tensors[0].shape[1:])


def _concat_linear_weights(weights):
    if len(weights) == 0:
        raise ValueError("At least one linear weight is required to build a fused projection.")
    first = weights[0]
    if isinstance(first, GGUFWeightTensor):
        tensor_type = getattr(first, "_tensor_type", None)
        dtype = getattr(first, "dtype", torch.float16)
        device = getattr(first, "device", torch.device("cpu"))
        if all(
            isinstance(weight, GGUFWeightTensor)
            and getattr(weight, "_tensor_type", None) == tensor_type
            and int(weight.shape[1]) == int(first.shape[1])
            for weight in weights
        ):
            raw = _cat_rows([weight._data for weight in weights])
            out_features = sum(int(weight.shape[0]) for weight in weights)
            in_features = int(first.shape[1])
            return GGUFWeightTensor.create(
                raw_tensor=raw,
                size=(out_features, in_features),
                stride=(in_features, 1),
                dtype=dtype,
                device=device,
                requires_grad=False,
                tensor_type=tensor_type,
                tensor_shape=(out_features, in_features),
            )
    if WeightQBytesTensor is not None and isinstance(first, WeightQBytesTensor):
        qtype = getattr(first, "qtype", getattr(first, "_qtype", None))
        axis = getattr(first, "axis", getattr(first, "_axis", None))
        activation_qtype = getattr(first, "activation_qtype", None)
        scale_ndim = int(first._scale.ndim)
        if all(
            isinstance(weight, WeightQBytesTensor)
            and getattr(weight, "qtype", getattr(weight, "_qtype", None)) == qtype
            and getattr(weight, "axis", getattr(weight, "_axis", None)) == axis
            and getattr(weight, "activation_qtype", None) == activation_qtype
            and int(weight.shape[1]) == int(first.shape[1])
            and int(weight._scale.ndim) == scale_ndim
            for weight in weights
        ):
            raw_data = _cat_rows([weight._data for weight in weights])
            raw_scale = _cat_rows([weight._scale for weight in weights])
            out_features = sum(int(weight.shape[0]) for weight in weights)
            in_features = int(first.shape[1])
            return WeightQBytesTensor.create(
                qtype,
                axis,
                size=(out_features, in_features),
                stride=(in_features, 1),
                data=raw_data,
                scale=raw_scale,
                activation_qtype=activation_qtype,
                requires_grad=False,
            )
    return _cat_rows([weight.detach() for weight in weights])


def _concat_linear_biases(biases):
    if len(biases) == 0 or any(bias is None for bias in biases):
        return None
    return _cat_rows([bias.detach() for bias in biases])


def _build_fused_column_linear(modules):
    modules = [module for module in modules if module is not None]
    if len(modules) <= 1:
        return modules[0] if modules else None
    template = modules[0]
    out_features = sum(int(module.weight.shape[0]) for module in modules)
    fused_kwargs = {}
    for attr_name, kwarg_name in (
        ("weight_qtype", "weights"),
        ("activation_qtype", "activations"),
        ("optimizer", "optimizer"),
        ("quantize_input", "quantize_input"),
    ):
        if hasattr(template, attr_name):
            fused_kwargs[kwarg_name] = getattr(template, attr_name)
    if hasattr(template, "_router_default_dtype"):
        fused_kwargs["dtype"] = template._router_default_dtype
    if hasattr(template, "weight") and torch.is_tensor(template.weight):
        fused_kwargs["device"] = torch.device("meta")
        fused_kwargs.setdefault("dtype", template.weight.dtype)
    try:
        fused = template.__class__(int(template.in_features), out_features, bias=all(module.bias is not None for module in modules), **fused_kwargs)
    except TypeError:
        fused = template.__class__(int(template.in_features), out_features, bias=all(module.bias is not None for module in modules))
    fused_weight = _concat_linear_weights([module.weight for module in modules])
    fused._parameters["weight"] = fused_weight
    fused._parameters["bias"] = _concat_linear_biases([module.bias for module in modules])
    if hasattr(fused, "qweight"):
        try:
            fused.qweight = fused_weight
        except Exception:
            pass
    for attr_name in ("weight_qtype", "activation_qtype", "optimizer", "input_scale", "output_scale", "_router_default_dtype", "_router_forward_impl", "_convrot_group_size", "_convrot_default_dtype"):
        if hasattr(template, attr_name):
            setattr(fused, attr_name, getattr(template, attr_name))
    if hasattr(template, "_gguf_default_dtype"):
        fused._gguf_default_dtype = template._gguf_default_dtype
    return fused


def _apply_qwen35_projection_fusions(model, *, prism=False, optimize_prism=True) -> None:
    blocks = list(getattr(model, "blk", ()))
    mtp = getattr(model, "mtp", None)
    if mtp is not None:
        blocks.append(mtp.block)
    use_optimized = not bool(getattr(model.config, "_prompt_enhancer_safe_legacy", False)) and torch.version.hip is None
    for block in blocks:
        block.ffn_down._fuse_silu_mul = use_optimized and not prism and isinstance(block.ffn_down.weight, GGUFWeightTensor)

        if getattr(block, "ffn_gate_up", None) is None and all(
            getattr(block, name, None) is not None for name in ("ffn_gate", "ffn_up")
        ):
            block.ffn_gate_up = _build_fused_column_linear([block.ffn_gate, block.ffn_up])
            block.ffn_gate = None
            block.ffn_up = None
        if prism and block is not getattr(mtp, "block", None):
            if not optimize_prism:
                continue
            # Fuse only projections with the same checkpoint basis and dtype.
            # Alpha/beta remain unrotated BF16, separate from packed QKV/gate.
            if block.layer_type == "linear_attention":
                block.attn_qkv_gate = _build_fused_column_linear([block.attn_qkv, block.attn_gate])
                block.ssm_ab = _build_fused_column_linear([block.ssm_alpha, block.ssm_beta])
                block.attn_qkv = block.attn_gate = block.ssm_alpha = block.ssm_beta = None
            else:
                block.attn_qkv_full = _build_fused_column_linear([block.attn_q, block.attn_k, block.attn_v])
                block.attn_q = block.attn_k = block.attn_v = None
            continue
        if getattr(block, "layer_type", None) != "linear_attention":
            continue
        # Real prompt-enhancer benchmarks only kept this fusion: it removes two extra
        # decode-time matmul launches from the hottest linear-attention block.
        if getattr(block, "attn_gate_ab", None) is None and all(
            getattr(block, name, None) is not None for name in ("attn_gate", "ssm_alpha", "ssm_beta")
        ):
            block.attn_gate_ab = _build_fused_column_linear([block.attn_gate, block.ssm_alpha, block.ssm_beta])
            block.attn_gate = None
            block.ssm_alpha = None
            block.ssm_beta = None

    for projection in model.modules():
        if isinstance(getattr(projection, "weight", None), GGUFWeightTensor):
            projection._use_optimized_kernels = use_optimized

def load_qwen35_text_prompt_enhancer(
    model_path: str | None = None,
    assets_dir: str | None = None,
    default_dtype: torch.dtype = torch.float16,
    backend: str = enhancer_quantization_QUANTO_INT8,
    attn_implementation: str = "sdpa",
    variant: str | None = None,
    speculative_decoding: bool = QWEN35_PROMPT_ENABLE_SPECULATIVE_DECODING,
    kv_cache_int8: bool = False,
    kv_cache_layout: str = "int8",
):
    del attn_implementation
    if assets_dir is None:
        raise ValueError("A local Qwen3.5 assets directory is required.")

    assets_dir = _resolve_qwen35_assets_dir(assets_dir, variant=variant)
    spec = get_qwen35_variant_spec(variant)
    text_assets_dir = assets_dir
    tokenizer_json = _resolve_qwen35_asset_file(assets_dir, "tokenizer.json", variant=variant, error_if_none=False) or os.path.join(assets_dir, "tokenizer.json")
    tokenizer_config = _resolve_qwen35_asset_file(assets_dir, "tokenizer_config.json", variant=variant, error_if_none=False) or os.path.join(assets_dir, "tokenizer_config.json")
    text_config_path = _resolve_qwen35_asset_file(text_assets_dir, "config.json", error_if_none=False) or os.path.join(text_assets_dir, "config.json")
    for required_file in (tokenizer_json, tokenizer_config, text_config_path):
        if not os.path.isfile(required_file):
            raise FileNotFoundError(f"Missing Qwen3.5 text prompt enhancer asset: {required_file}")

    tokenizer = _load_qwen35_tokenizer(assets_dir)
    chat_template_path = _resolve_qwen35_asset_file(text_assets_dir, "chat_template.jinja", error_if_none=False) or os.path.join(text_assets_dir, "chat_template.jinja")
    if os.path.isfile(chat_template_path):
        with open(chat_template_path, "r", encoding="utf-8") as reader:
            tokenizer.chat_template = reader.read()

    features = _resolve_llm_features(backend)  # the engine uses whatever the hardware and the installed packages allow
    safe_legacy_mode = not features["triton"]
    prism_metadata = {}
    if backend == enhancer_quantization_GGUF:
        from shared.qtypes.prism import get_prism_metadata
        model_path = _resolve_gguf_model_path(model_path, assets_dir, variant=variant)
        prism_metadata = get_prism_metadata(model_path)
    from .config import BLOCK_DRAFT_METHODS
    block_draft_method = speculative_decoding if speculative_decoding in BLOCK_DRAFT_METHODS else None
    if block_draft_method is not None and (variant != "27b" or not (features["triton"] and features["flash"])):
        raise ValueError("DSpark and DFlash2 require Qwen3.8/Bonsai 27B, Triton and FlashAttention 2.")
    enable_mtp = bool(speculative_decoding and block_draft_method is None and spec.get("supports_mtp", False))
    mtp_filename = _qwen35_mtp_filename(spec, backend, model_path, prism_metadata) if enable_mtp else None
    mtp_modules = [_resolve_qwen35_checkpoint_file(assets_dir, mtp_filename, variant=variant)] if mtp_filename else None
    postprocess_sd = _add_qwen35_mtp_shared_weights if enable_mtp else None
    if backend == enhancer_quantization_GGUF:
        model_path = _resolve_gguf_model_path(model_path, assets_dir, variant=variant)
        gguf_v_head_reordered, gguf_ssm_param_reordered, gguf_interleave_ssm_ab = _resolve_gguf_linear_attention_layout_from_filename(model_path)
        if prism_metadata:
            gguf_v_head_reordered, gguf_ssm_param_reordered, gguf_interleave_ssm_ab = True, True, False
        quanto_log_ssm_a = False
        preprocess_sd = _build_qwen35_gguf_preprocess_sd(
            tie_output_to_embeddings=bool(spec.get("tie_word_embeddings", False)),
            num_hidden_layers=_load_text_config(text_config_path).num_hidden_layers,
            enable_mtp=enable_mtp,
        )
        runtime_model_path = text_assets_dir
    else:
        gguf_v_head_reordered = False
        gguf_ssm_param_reordered = False
        gguf_interleave_ssm_ab = False
        if model_path is None:
            model_path = get_qwen35_text_quanto_int8_path(assets_dir, variant=variant)
        quanto_log_ssm_a = _resolve_quanto_log_ssm_a(model_path, spec)
        preprocess_sd = None
        runtime_model_path = text_assets_dir

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Qwen3.5 text checkpoint not found: {model_path}")
    print(f"[Qwen3.5VL][{spec['display_name']}][{backend}] Loading text checkpoint: {model_path}")
    if backend == enhancer_quantization_GGUF:
        print(
            "[Qwen3.5VL]"
            f"[{spec['display_name']}][gguf] Linear-attention layout flags: "
            f"v_head_reordered={bool(gguf_v_head_reordered)} "
            f"ssm_param_reordered={bool(gguf_ssm_param_reordered)} "
            f"interleave_ssm_ab={bool(gguf_interleave_ssm_ab)}"
        )
    elif quanto_log_ssm_a:
        print(f"[Qwen3.5VL][{spec['display_name']}][quanto_int8] SSM A parameters are stored in log form.")

    model = _load_local_text_model(
        model_path,
        text_config_path,
        preprocess_sd=preprocess_sd,
        default_dtype=default_dtype,
        safe_legacy_mode=safe_legacy_mode,
        materialize_source_tensors=backend != enhancer_quantization_GGUF,
        enable_mtp=enable_mtp,
        modules=mtp_modules,
        postprocess_sd=postprocess_sd,
        preserve_native_dtypes=bool(prism_metadata),
        flash_attention=features["flash"],
    )
    if backend == enhancer_quantization_QUANTO_INT8 and spec.get("text_int8_tie_word_embeddings", False):
        _tie_qwen35_output_to_embeddings(model)
    if backend == enhancer_quantization_GGUF:
        _configure_qwen35_text_model(
            model,
            default_dtype,
            v_head_reordered=gguf_v_head_reordered,
            ssm_param_reordered=gguf_ssm_param_reordered,
            interleave_ssm_ab=gguf_interleave_ssm_ab,
        )
    elif quanto_log_ssm_a:
        _configure_qwen35_text_model(model, log_ssm_a=True)
    _apply_qwen35_projection_fusions(model, prism=bool(prism_metadata))
    if prism_metadata:
        from shared.qtypes.prism import install_prism_transforms
        install_prism_transforms(model, prism_metadata)
        if features["triton"]:
            from shared.qtypes.prism import install_prism_decode
            install_prism_decode(model)
    # The sm80+ decode fusions were validated on the 27B GGUF; Turing's recurrence takes head counts, layouts and A / log A from the blocks
    # (27B, 9B, 4B; GGUF and Quanto INT8)
    if features["triton"] and (backend == enhancer_quantization_GGUF and variant == "27b" or torch.cuda.get_device_capability(0) < (8, 0)):
        from shared.kernels.qwen_gdn import install_gdn_decode
        install_gdn_decode(model)
    mtp_draft_vocab_size = spec.get("mtp_draft_vocab_size")
    # row views of GGUF weights: the 9B's quanto INT8 checkpoint keeps drafting over the whole vocabulary
    if enable_mtp and mtp_draft_vocab_size is not None and backend == enhancer_quantization_GGUF:
        if prism_metadata:
            from shared.qtypes.prism import PrismFirstRowsLinear
            model.mtp.lm_head = PrismFirstRowsLinear(model.mtp.lm_head, mtp_draft_vocab_size)
        else:
            model.mtp.lm_head = GGUFFirstRowsLinear(model.mtp.lm_head, mtp_draft_vocab_size)
        model.mtp.draft_vocab_size = mtp_draft_vocab_size

    model._prompt_enhancer_tokenizer = tokenizer
    model._prompt_enhancer_gguf_v_head_reordered = bool(gguf_v_head_reordered)
    model._prompt_enhancer_gguf_ssm_param_reordered = bool(gguf_ssm_param_reordered)
    model._prompt_enhancer_log_ssm_a = bool(quanto_log_ssm_a)
    model._prompt_enhancer_thinking_extra_tokens = QWEN35_PROMPT_THINKING_EXTRA_TOKENS
    model._prompt_enhancer_thinking_max_tokens = QWEN35_PROMPT_THINKING_MAX_TOKENS
    model._prompt_enhancer_close_think_token_id = tokenizer.convert_tokens_to_ids("</think>")
    model._prompt_enhancer_stop_token_ids = _collect_stop_token_ids(tokenizer, model.config)
    set_qwen35_prompt_enhancer_thinking(model, QWEN35_PROMPT_ENABLE_THINKING)
    model._prompt_enhancer_suppress_logits_bias = None
    model._prompt_enhancer_suppress_logits_bias_cache = {}
    model._prompt_enhancer_default_top_k = QWEN35_PROMPT_DEFAULT_TOP_K
    model._prompt_enhancer_default_min_p = QWEN35_PROMPT_DEFAULT_MIN_P_GGUF if backend == enhancer_quantization_GGUF else None
    model._prompt_enhancer_penalty_mode = QWEN35_PENALTY_MODE
    model._prompt_enhancer_presence_penalty = QWEN35_PROMPT_PRESENCE_PENALTY
    model._prompt_enhancer_repetition_penalty = QWEN35_PROMPT_REPETITION_PENALTY
    model._prompt_enhancer_predictive_penalty_enabled = QWEN35_PREDICTIVE_PENALTY_ENABLED
    model._prompt_enhancer_min_model_len_hint = QWEN35_PROMPT_MIN_MODEL_LEN
    model._prompt_enhancer_allow_extended_context = True
    model._prompt_enhancer_min_new_tokens = (
        QWEN35_PROMPT_MIN_NEW_TOKENS
        if backend == enhancer_quantization_GGUF and _env_enabled(QWEN35_GGUF_LLAMACPP_ENV, default=True)
        else 0
    )
    model._prompt_enhancer_use_vllm = False
    model._prompt_enhancer_use_legacy_cuda_runner = False
    model._prompt_enhancer_enable_cudagraph = features["graphs"]
    model._prompt_enhancer_allow_vllm_kernels = features["triton"]
    model._prompt_enhancer_llm_features = features
    model._prompt_enhancer_vllm_model_path = runtime_model_path
    model._prompt_enhancer_vllm_engine = None
    model._prompt_enhancer_vllm_mode = None
    model._prompt_enhancer_assistant_graph_pool_handle = None
    model._prompt_enhancer_safe_legacy = safe_legacy_mode
    model._prompt_enhancer_speculative_decoding = enable_mtp
    model._prompt_enhancer_speculative_method_logged = False
    model._deepy_kv_cache_int8 = bool(kv_cache_int8)
    model._deepy_kv_cache_layout = str(kv_cache_layout or "int8") if kv_cache_int8 else ""
    model._prompt_enhancer_speculative_tokens = spec.get("mtp_speculative_tokens", QWEN35_PROMPT_SPECULATIVE_TOKENS)
    model._prompt_enhancer_speculative_sampling_tokens = spec.get("mtp_speculative_sampling_tokens", QWEN35_PROMPT_SPECULATIVE_SAMPLING_TOKENS)
    model._prompt_enhancer_speculative_confidence = spec.get("mtp_speculative_confidence", 0.30)
    if block_draft_method is not None:
        from .block_draft import install_block_draft
        install_block_draft(model, block_draft_method, bonsai=bool(prism_metadata))
    model._prompt_enhancer_use_vllm = features["cuda"]  # the LM engine with each optimization where supported, else (Apple GPUs) the legacy runner
    model._prompt_enhancer_use_legacy_cuda_runner = not features["cuda"]
    # MMGP manages the weights: its dynamic preload keeps in VRAM all the blocks that fit beside the KV cache and the activations (all of
    # them at first: without its VRAM allocator, the budgets of the profile apply), with CUDA graphs its graphs replay the decoding passes
    model._lm_engine_mmgp = features["cuda"]
    model._lm_engine_mmgp_graphs = features["graphs"]
    if QWEN35_LLM_MAPPED_TOKEN_TABLE:
        _map_qwen35_token_table(model)
    from mmgp import allocator
    if allocator.active == "vmm":
        model._budget = 2 * os.path.getsize(model_path)  # beyond the model: every block preloaded
    # MMGP cpuCompute (decoding computes the blocks out of VRAM on the CPU): what the runtime needs beside the weights, for its pinning share
    kv_bytes = _qwen35_kv_bytes(model.config, model._deepy_kv_cache_layout if kv_cache_int8 else "", QWEN35_LLM_PLANNED_CONTEXT)
    model._mm_cpu_compute_reserve = kv_bytes + (2 << 30)  # KV cache, activations, vision tower
    if QWEN35_LLM_MAX_PRELOAD_MB:
        model._budget = model._auto_preload_max = QWEN35_LLM_MAX_PRELOAD_MB << 20
    model._mm_graph_keys = 32  # its decoding passes: one token, then 2 to 9 for the speculative verifications
    model._mm_graph_placement_key = True  # only MMGP moves its weights: the key of a whole-model graph is checked in constant time
    model._mm_graph_check_once = True  # its passes of any length read the same state: reads checked, cuBLAS use found, at the first capture
    model._auto_preload_workload = _qwen35_preload_workload
    if QWEN35_LLM_PRELOAD_MARGIN_MB is not None:
        model._auto_preload_margin = QWEN35_LLM_PRELOAD_MARGIN_MB << 20
    for block in model.blk:
        block._mm_graph_keys = 32
    reset_attention_backend_logs()
    speculative = f"{block_draft_method} drafts" if block_draft_method else ("native MTP" if enable_mtp else "")
    print(f"[Qwen3.5VL][{spec['display_name']}] LLM engine: {_describe_llm_features(features, kv_cache_int8, speculative, model._deepy_kv_cache_layout)}")
    model.generate_messages = types.MethodType(_generate_messages, model)
    model.unload = types.MethodType(_unload_prompt_enhancer_text_runtime, model)
    model._offload_hooks = ["forward"]
    model.eval()
    return model


load_qwen35_prompt_enhancer = load_qwen35_text_prompt_enhancer


__all__ = [
    "QWEN35_PROMPT_ENABLE_SPECULATIVE_DECODING",
    "QWEN35_PROMPT_SPECULATIVE_TOKENS",
    "get_qwen35_text_quanto_int8_path",
    "load_qwen35_prompt_enhancer",
    "load_qwen35_text_prompt_enhancer",
    "set_qwen35_prompt_enhancer_thinking",
]
