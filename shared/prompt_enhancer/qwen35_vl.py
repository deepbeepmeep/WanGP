# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import json
import os
import re
import types
from collections import OrderedDict
from contextlib import contextmanager, nullcontext

import torch

from mmgp import offload
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer, Qwen2TokenizerFast, Qwen2VLImageProcessorFast, Qwen2VLProcessor
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen2_vl.video_processing_qwen2_vl import Qwen2VLVideoProcessor

from shared.llm_io import known_token_ids, llm_io_enabled, log_llm_io, media_descriptor
from shared.llm_engines.lmgp.models.qwen3_5 import Qwen3_5DynamicCache
from shared.llm_engines.lmgp.utils.context import reset_context
from shared.qtypes.gguf import materialize_module_source_tensors
from shared.utils import files_locator as fl
from shared.utils.cancellation import cancellation_context, check_cancelled

from .assets import (
    QWEN35_4B_TEXT_GGUF_FILENAME,
    QWEN35_4B_TEXT_INT8_FILENAME,
    QWEN35_4B_VISION_FILENAME,
    QWEN35_ABLITERATED_REPO,
    QWEN35_ABLITERATED_TEXT_REQUIRED_FILES,
    QWEN35_TEXT_GGUF_FILENAME,
    QWEN35_TEXT_INT8_FILENAME,
    QWEN35_VARIANT_4B,
    QWEN35_VARIANT_9B,
    QWEN35_VARIANT_SPECS,
    QWEN35_VISION_FILENAME,
    QWEN38_VARIANT_27B,
    QWEN38_VARIANT_9B,
)
from .qwen3_5 import load_qwen35_model_class
from .progress import EnhancementProgress


UPSTREAM_MODELING_FILENAME = "modeling_qwen3_5.py"
enhancer_quantization_GGUF = "gguf"
enhancer_quantization_GGUF_Q3 = "gguf_q3"
enhancer_quantization_GGUF_Q2 = "gguf_q2"
enhancer_quantization_GGUF_PTQ1 = "gguf_ptq1"
enhancer_quantization_GGUF_Q8 = "gguf_q8"
enhancer_quantization_SAFETENSORS = "safetensors"
enhancer_quantization_QUANTO_INT8 = "quanto_int8"
QWEN35_GGUF_LLAMACPP_ENV = "WGP_GGUF_LLAMACPP_CUDA"
QWEN35_PROMPT_MIN_NEW_TOKENS = 4
QWEN35_IMAGE_CAPTION_MAX_EDGE = 1024
QWEN35_VARIANT_ALIASES = {
    "": QWEN35_VARIANT_9B,
    "9": QWEN35_VARIANT_9B,
    "9b": QWEN35_VARIANT_9B,
    "9b_abliterated": QWEN35_VARIANT_9B,
    "qwen3.5-9b": QWEN35_VARIANT_9B,
    "qwen3.5-9b abliterated": QWEN35_VARIANT_9B,
    "4": QWEN35_VARIANT_4B,
    "4b": QWEN35_VARIANT_4B,
    "4b_abliterated": QWEN35_VARIANT_4B,
    "qwen3.5-4b": QWEN35_VARIANT_4B,
    "qwen3.5-4b abliterated": QWEN35_VARIANT_4B,
    "qwen3.5-4b_abliterated": QWEN35_VARIANT_4B,
    "27": QWEN38_VARIANT_27B,
    "27b": QWEN38_VARIANT_27B,
    "qwen3.8-27b": QWEN38_VARIANT_27B,
    "qwen3.8-27b uncensored": QWEN38_VARIANT_27B,
    "qwen3.8-9b": QWEN38_VARIANT_9B,
    "qwen3.8-9b uncensored": QWEN38_VARIANT_9B,
}


def get_qwen35_variant_spec(variant: str | None = None) -> dict:
    normalized = QWEN35_VARIANT_ALIASES.get(str(variant or "").strip().lower(), str(variant or "").strip().lower())
    if normalized not in QWEN35_VARIANT_SPECS:
        raise ValueError(f"Unsupported Qwen3.5 prompt enhancer variant: {variant}")
    return QWEN35_VARIANT_SPECS[normalized]


def get_qwen35_assets_dir_name(variant: str | None = None) -> str:
    return get_qwen35_variant_spec(variant)["assets_dir_name"]


def get_qwen35_prompt_enhancer_variant(model_no) -> str:
    return {3: QWEN35_VARIANT_4B, 4: QWEN35_VARIANT_9B, 5: QWEN38_VARIANT_27B, 6: QWEN38_VARIANT_9B}[int(model_no)]


def get_qwen35_quantization(backend: str, variant: str | None = None) -> str:
    spec = get_qwen35_variant_spec(variant)
    if backend in (enhancer_quantization_GGUF_Q2, enhancer_quantization_GGUF_Q3, enhancer_quantization_GGUF_PTQ1, enhancer_quantization_GGUF_Q8):
        quantization = backend.rsplit("_", 1)[-1]
        if f"text_gguf_{quantization}_filename" not in spec:
            raise ValueError(f"{spec['display_name']} does not provide a GGUF {quantization.upper()} checkpoint.")
        return backend
    return spec.get("backend", backend)


def _get_qwen35_gguf_filename(spec: dict, backend: str) -> str:
    key = {
        enhancer_quantization_GGUF: "text_gguf_filename",
        enhancer_quantization_GGUF_Q3: "text_gguf_q3_filename",
        enhancer_quantization_GGUF_Q2: "text_gguf_q2_filename",
        enhancer_quantization_GGUF_PTQ1: "text_gguf_ptq1_filename",
        enhancer_quantization_GGUF_Q8: "text_gguf_q8_filename",
    }[backend]
    return spec[key]


def _resolve_qwen35_assets_dir(assets_dir: str | None, variant: str | None = None, error_if_none: bool = True) -> str | None:
    folder_name = get_qwen35_assets_dir_name(variant)
    candidate = folder_name if assets_dir is None else assets_dir
    resolved = fl.locate_folder(candidate, error_if_none=False)
    if resolved is None and assets_dir is not None and candidate != folder_name:
        resolved = fl.locate_folder(folder_name, error_if_none=False)
    if resolved is None and error_if_none:
        raise FileNotFoundError(f"Missing Qwen3.5 assets folder '{candidate}'")
    return resolved


def _resolve_qwen35_asset_file(
    assets_dir: str | None,
    filename: str,
    variant: str | None = None,
    error_if_none: bool = True,
) -> str | None:
    del variant
    if assets_dir is None:
        return None
    path = os.path.join(assets_dir, filename)
    if not os.path.isfile(path) and error_if_none:
        raise FileNotFoundError(f"Missing Qwen3.5 asset: {path}")
    return path if os.path.isfile(path) else None


def _resolve_qwen35_checkpoint_file(
    assets_dir: str | None,
    filename: str,
    variant: str | None = None,
    error_if_none: bool = True,
) -> str | None:
    del variant
    if assets_dir is None:
        return None
    resolved = fl.locate_file(filename, error_if_none=False, extra_paths=[assets_dir])
    if resolved is not None:
        return resolved
    fallback_path = os.path.join(assets_dir, filename)
    if error_if_none:
        raise FileNotFoundError(f"Missing Qwen3.5 checkpoint: {fallback_path}")
    return fallback_path


def get_qwen35_modeling_path() -> str:
    return os.path.join(os.path.dirname(__file__), "qwen3_5", UPSTREAM_MODELING_FILENAME)


def ensure_qwen35_prompt_enhancer_assets(process_files_def, backend: str = enhancer_quantization_QUANTO_INT8, variant: str | None = None, speculative_decoding: bool = False):
    from .block_draft import BLOCK_DRAFT_METHODS, ensure_block_draft_assets
    if speculative_decoding in BLOCK_DRAFT_METHODS:
        ensure_block_draft_assets(process_files_def, speculative_decoding, variant, backend)
        speculative_decoding = False
    spec = get_qwen35_variant_spec(variant)
    backend = get_qwen35_quantization(backend, variant=variant)
    repo_subfolder = spec.get("repo_subfolder", "")
    qwen35_shared_files = list(spec["root_files"])
    if spec["root_repo"] == spec.get("gguf_repo"):
        gguf_backend = backend in (enhancer_quantization_GGUF, enhancer_quantization_GGUF_Q3, enhancer_quantization_GGUF_Q2, enhancer_quantization_GGUF_PTQ1, enhancer_quantization_GGUF_Q8)
        checkpoint_filename = _get_qwen35_gguf_filename(spec, backend) if gguf_backend else spec["text_int8_filename"]
        qwen35_shared_files.append(spec["vision_filename"])
        checkpoint_folder = spec.get("gguf_repo_subfolder", repo_subfolder)
        if checkpoint_folder == repo_subfolder:
            qwen35_shared_files.append(checkpoint_filename)
        else:
            process_files_def(repoId=spec["gguf_repo"], sourceFolderList=[checkpoint_folder], targetFolderList=[spec["assets_dir_name"]], fileList=[[checkpoint_filename]])
        if speculative_decoding:
            mtp_filename = spec.get("text_gguf_q3_mtp_filename" if backend == enhancer_quantization_GGUF_Q3 else "text_mtp_filename")
            if mtp_filename:
                qwen35_shared_files.append(mtp_filename)
    download_def = {"repoId": spec["root_repo"], "sourceFolderList": [repo_subfolder], "fileList": [qwen35_shared_files]}
    if len(repo_subfolder) == 0:
        download_def["targetFolderList"] = [spec["assets_dir_name"]]
    process_files_def(**download_def)
    if spec["root_repo"] != spec.get("gguf_repo"):
        if backend not in (enhancer_quantization_GGUF, enhancer_quantization_GGUF_Q3, enhancer_quantization_GGUF_Q2, enhancer_quantization_GGUF_PTQ1, enhancer_quantization_GGUF_Q8):
            raise ValueError(f"{spec['display_name']} supports only the GGUF backend.")
        gguf_files = [spec["vision_filename"], _get_qwen35_gguf_filename(spec, backend)]
        if speculative_decoding and backend == enhancer_quantization_GGUF_Q3:
            gguf_files.append(spec["text_gguf_q3_mtp_filename"])
        if speculative_decoding and backend == enhancer_quantization_GGUF_PTQ1:
            gguf_files.append(spec["text_gguf_ptq1_mtp_filename"])
        process_files_def(repoId=spec["gguf_repo"], sourceFolderList=[spec.get("gguf_repo_subfolder", "")], fileList=[gguf_files])
    if spec.get("text_repo") and spec.get("text_required_files"):
        process_files_def(repoId=spec["text_repo"], sourceFolderList=[repo_subfolder], fileList=[list(spec["text_required_files"])])
    qwen35_modeling_path = get_qwen35_modeling_path()
    if not os.path.isfile(qwen35_modeling_path):
        raise FileNotFoundError(f"Missing repo-local Qwen3.5 modeling file: {qwen35_modeling_path}")


def _load_qwen35_tokenizer(assets_dir: str):
    tokenizer_config_path = _resolve_qwen35_asset_file(assets_dir, "tokenizer_config.json")
    tokenizer_class = None
    tokenizer_config = {}
    if tokenizer_config_path is not None:
        try:
            with open(tokenizer_config_path, "r", encoding="utf-8") as reader:
                tokenizer_config = json.load(reader)
                tokenizer_class = tokenizer_config.get("tokenizer_class")
        except Exception:
            tokenizer_class = None
    chat_template = _load_qwen35_chat_template(assets_dir)
    if tokenizer_class == "TokenizersBackend":
        tokenizer_path = _resolve_qwen35_asset_file(assets_dir, "tokenizer.json", error_if_none=True)
        tokenizer = Qwen2TokenizerFast(
            tokenizer_file=tokenizer_path,
            bos_token=tokenizer_config.get("bos_token"),
            eos_token=tokenizer_config.get("eos_token"),
            unk_token=tokenizer_config.get("unk_token"),
            pad_token=tokenizer_config.get("pad_token"),
            clean_up_tokenization_spaces=bool(tokenizer_config.get("clean_up_tokenization_spaces", False)),
            split_special_tokens=bool(tokenizer_config.get("split_special_tokens", False)),
            model_max_length=int(tokenizer_config.get("model_max_length", 1000000000000000019884624838656)),
        )
        tokenizer.name_or_path = assets_dir
        for token_name, token_value in tokenizer_config.get("model_specific_special_tokens", {}).items():
            if token_value is not None:
                setattr(tokenizer, token_name, token_value)
        for token_name in ("image_token", "video_token", "vision_bos_token", "vision_eos_token", "audio_bos_token", "audio_eos_token", "audio_token"):
            token_value = tokenizer_config.get(token_name)
            if token_value is not None:
                setattr(tokenizer, token_name, token_value)
        if chat_template is not None:
            tokenizer.chat_template = chat_template
        return tokenizer
    tokenizer = AutoTokenizer.from_pretrained(assets_dir, trust_remote_code=True)
    if chat_template is not None:
        tokenizer.chat_template = chat_template
    return tokenizer


def _load_qwen35_chat_template(assets_dir: str) -> str | None:
    chat_template_path = _resolve_qwen35_asset_file(assets_dir, "chat_template.jinja", error_if_none=False)
    if chat_template_path is None or not os.path.isfile(chat_template_path):
        return None
    with open(chat_template_path, "r", encoding="utf-8") as reader:
        return reader.read()


def _load_qwen35_image_processor(assets_dir: str):
    preprocessor_config_path = _resolve_qwen35_asset_file(assets_dir, "preprocessor_config.json", error_if_none=False)
    if preprocessor_config_path is not None:
        return Qwen2VLImageProcessorFast.from_pretrained(assets_dir)

    video_preprocessor_config_path = _resolve_qwen35_asset_file(assets_dir, "video_preprocessor_config.json", error_if_none=True)
    with open(video_preprocessor_config_path, "r", encoding="utf-8") as reader:
        config = json.load(reader)
    config["image_processor_type"] = "Qwen2VLImageProcessorFast"
    config.pop("processor_class", None)
    config.pop("video_processor_type", None)
    return Qwen2VLImageProcessorFast(**config)


def get_qwen35_text_gguf_path(assets_dir: str, variant: str | None = None, backend: str = enhancer_quantization_GGUF) -> str:
    spec = get_qwen35_variant_spec(variant)
    filename = _get_qwen35_gguf_filename(spec, get_qwen35_quantization(backend, variant=variant))
    return _resolve_qwen35_checkpoint_file(assets_dir, filename, variant=variant, error_if_none=False)


def _build_qwen35_vl_gguf_preprocess_sd(patch_shape):
    def preprocess_sd(sd, quant_map=None, tied_map=None):
        new_sd = OrderedDict()
        patch_slices = []
        for name, tensor in sd.items():
            if name in ("v.pos_embed.weight", "v.position_embd.weight"):
                target_name = "pos_embed.weight"
                target_tensor = tensor
            elif name in ("v.patch_embed.bias", "v.patch_embd.bias"):
                target_name = "patch_embed.proj.bias"
                target_tensor = tensor
            elif name == "v.patch_embed.weight":
                target_name = "patch_embed.proj.weight"
                target_tensor = tensor.reshape(*patch_shape)
            elif re.match(r"^v\.patch_embd\.weight(?:\.\d+)?$", name):
                patch_index = int(name.rsplit(".", 1)[-1]) if name.rsplit(".", 1)[-1].isdigit() else 0
                patch_slices.append((patch_index, tensor))
                continue
            elif name.startswith("v.merger."):
                target_name = name[2:]
                target_tensor = tensor
            elif name.startswith("v.post_ln."):
                target_name = "merger.norm." + name.removeprefix("v.post_ln.")
                target_tensor = tensor
            elif name.startswith("mm.0."):
                target_name = "merger.linear_fc1." + name.removeprefix("mm.0.")
                target_tensor = tensor
            elif name.startswith("mm.2."):
                target_name = "merger.linear_fc2." + name.removeprefix("mm.2.")
                target_tensor = tensor
            else:
                vision_match = re.match(r"^v\.blk\.(\d+)\.(.+)$", name)
                if not vision_match:
                    continue
                layer_no = vision_match.group(1)
                suffix = vision_match.group(2)
                prefix = f"blocks.{layer_no}."
                target_name = None
                target_tensor = tensor
                if suffix == "attn_qkv.weight" or suffix == "attn_qkv.bias":
                    projection_suffix = suffix.rsplit(".", 1)[-1]
                    for projection_name, projection_tensor in zip(("q_proj", "k_proj", "v_proj"), tensor.chunk(3, dim=0)):
                        new_sd[prefix + f"attn.{projection_name}.{projection_suffix}"] = projection_tensor
                    continue
                elif suffix == "attn_q.weight":
                    target_name = prefix + "attn.q_proj.weight"
                elif suffix == "attn_q.bias":
                    target_name = prefix + "attn.q_proj.bias"
                elif suffix == "attn_k.weight":
                    target_name = prefix + "attn.k_proj.weight"
                elif suffix == "attn_k.bias":
                    target_name = prefix + "attn.k_proj.bias"
                elif suffix == "attn_v.weight":
                    target_name = prefix + "attn.v_proj.weight"
                elif suffix == "attn_v.bias":
                    target_name = prefix + "attn.v_proj.bias"
                elif suffix == "attn_out.weight":
                    target_name = prefix + "attn.proj.weight"
                elif suffix == "attn_out.bias":
                    target_name = prefix + "attn.proj.bias"
                elif suffix.startswith("ffn_up."):
                    target_name = prefix + "mlp.linear_fc1." + suffix.removeprefix("ffn_up.")
                elif suffix.startswith("ffn_down."):
                    target_name = prefix + "mlp.linear_fc2." + suffix.removeprefix("ffn_down.")
                elif suffix.startswith("ln1."):
                    target_name = prefix + "norm1." + suffix.removeprefix("ln1.")
                elif suffix.startswith("ln2."):
                    target_name = prefix + "norm2." + suffix.removeprefix("ln2.")
                elif suffix.startswith("mlp.") or suffix.startswith("norm"):
                    target_name = prefix + suffix
                if target_name is None:
                    continue
            new_sd[target_name] = target_tensor
        if patch_slices:
            new_sd["patch_embed.proj.weight"] = torch.stack([tensor for _, tensor in sorted(patch_slices)], dim=2).reshape(*patch_shape)
        return new_sd, quant_map, tied_map

    return preprocess_sd


def _move_batch_to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _resolve_execution_device(self, model_inputs=None) -> torch.device:
    if model_inputs is not None:
        for value in model_inputs.values():
            if torch.is_tensor(value):
                return value.device
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    try:
        return self.device
    except Exception:
        return torch.device("cpu")


def _resize_image_for_caption(image: Image.Image) -> Image.Image:
    width, height = image.size
    max_edge = max(width, height)
    if max_edge <= QWEN35_IMAGE_CAPTION_MAX_EDGE:
        return image
    scale = QWEN35_IMAGE_CAPTION_MAX_EDGE / max_edge
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def alias_qwen35_text_embedding_for_mmgp(text_model: torch.nn.Module) -> torch.nn.Module:
    source_embedding = text_model.token_embd
    cls = source_embedding.__class__
    kwargs = {
        "padding_idx": source_embedding.padding_idx,
        "max_norm": source_embedding.max_norm,
        "norm_type": source_embedding.norm_type,
        "scale_grad_by_freq": source_embedding.scale_grad_by_freq,
        "sparse": source_embedding.sparse,
        "dtype": getattr(source_embedding, "_gguf_default_dtype", getattr(source_embedding.weight, "dtype", None)),
        "device": "meta",
    }
    if hasattr(source_embedding, "weight_qtype"):
        kwargs.update(
            weights=source_embedding.weight_qtype,
            activations=getattr(source_embedding, "activation_qtype", None),
            optimizer=getattr(source_embedding, "optimizer", None),
        )
    embedding_model = cls(source_embedding.num_embeddings, source_embedding.embedding_dim, **kwargs)
    embedding_model.weight = source_embedding.weight
    for name, buffer in source_embedding._buffers.items():
        embedding_model._buffers[name] = buffer
    if hasattr(source_embedding, "prism_transform"):
        from shared.qtypes.prism import PrismHadamard
        transform = source_embedding.prism_transform
        embedding_model.prism_transform = PrismHadamard(transform.signs, transform.inverse, transform.grouped_shape)
    if hasattr(source_embedding, "_gguf_default_dtype"):
        embedding_model._gguf_default_dtype = source_embedding._gguf_default_dtype
    embedding_model.eval()
    return embedding_model


class _SharedQwen35TextAdapter(torch.nn.Module):
    def __init__(self, text_model: torch.nn.Module, input_embedding_model: torch.nn.Module | None = None):
        super().__init__()
        object.__setattr__(self, "_shared_text_model", text_model)
        object.__setattr__(self, "_input_embedding_model", input_embedding_model)
        self.config = getattr(text_model, "config", None)

    @property
    def text_model(self):
        return object.__getattribute__(self, "_shared_text_model")

    @property
    def embed_tokens(self):
        input_embedding_model = object.__getattribute__(self, "_input_embedding_model")
        return input_embedding_model if input_embedding_model is not None else self.text_model.embed_tokens

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.text_model.token_embd = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        **kwargs,
    ):
        del kwargs
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds for the shared Qwen3.5 text adapter.")
        if past_key_values is None and (use_cache is None or bool(use_cache)):
            past_key_values = Qwen3_5DynamicCache(self.config)
        hidden_states = self.text_model(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values)


class _SharedQwen35LmHeadAdapter(torch.nn.Module):
    def __init__(self, lm_head: torch.nn.Module):
        super().__init__()
        object.__setattr__(self, "_shared_lm_head", lm_head)

    @property
    def lm_head(self):
        return object.__getattribute__(self, "_shared_lm_head")

    @property
    def weight(self):
        return self.lm_head.weight

    def forward(self, *args, **kwargs):
        return self.lm_head(*args, **kwargs)


class _PromptEnhancerQwen2VLProcessor(Qwen2VLProcessor):
    def __repr__(self):
        tokenizer = getattr(self, "tokenizer", None)
        image_processor = getattr(self, "image_processor", None)
        video_processor = getattr(self, "video_processor", None)
        name_or_path = getattr(tokenizer, "name_or_path", None) or getattr(self, "name_or_path", None)
        parts = []
        if name_or_path:
            parts.append(f"name_or_path={name_or_path!r}")
        if tokenizer is not None:
            parts.append(f"tokenizer={tokenizer.__class__.__name__}")
        if image_processor is not None:
            parts.append(f"image_processor={image_processor.__class__.__name__}")
        if video_processor is not None:
            parts.append(f"video_processor={video_processor.__class__.__name__}")
        return f"Qwen2VLProcessor({', '.join(parts)})"

    __str__ = __repr__


class _Qwen35CaptionWrapper(torch.nn.Module):
    def __init__(
        self,
        caption_runtime_model: torch.nn.Module,
        vision_tower_model: torch.nn.Module,
        text_model: torch.nn.Module,
        tokenizer,
        processor,
    ):
        super().__init__()
        object.__setattr__(self, "_caption_runtime_model", caption_runtime_model)
        object.__setattr__(self, "vision_tower_model", vision_tower_model)
        object.__setattr__(self, "model", caption_runtime_model.model)
        self.config = caption_runtime_model.config
        self.generation_config = caption_runtime_model.generation_config
        self._prompt_enhancer_tokenizer = tokenizer
        self._prompt_enhancer_processor = processor
        object.__setattr__(self, "_prompt_enhancer_text_model", text_model)
        self._prompt_enhancer_image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        self._prompt_enhancer_video_token_id = tokenizer.convert_tokens_to_ids("<|video_pad|>")

    def forward(self, *args, **kwargs):
        return self._caption_runtime_model(*args, **kwargs)

    def eval(self):
        self._caption_runtime_model.eval()
        self.vision_tower_model.eval()
        return super().eval()


def _clean_generated_text(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "\n", text, flags=re.DOTALL | re.IGNORECASE)
    text = text.replace("<think>", "\n").replace("</think>", "\n")
    text = text.replace("<|im_end|>", "").replace("<|im_start|>", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    return "\n".join(line for line in lines if line).strip()


def _iter_token_ids(token_ids):
    if token_ids is None:
        return
    if isinstance(token_ids, (list, tuple, set)):
        for token_id in token_ids:
            yield from _iter_token_ids(token_id)
        return
    try:
        token_id = int(token_ids)
    except (TypeError, ValueError):
        return
    if token_id >= 0:
        yield token_id


def _collect_stop_token_ids(tokenizer, *config_objects):
    stop_token_ids = []
    seen = set()

    def add(token_ids):
        for token_id in _iter_token_ids(token_ids):
            if token_id in seen:
                continue
            seen.add(token_id)
            stop_token_ids.append(token_id)

    add(getattr(tokenizer, "eos_token_id", None))
    for token in ("<|im_end|>", "<|endoftext|>"):
        add(tokenizer.convert_tokens_to_ids(token))

    for config_object in config_objects:
        if config_object is None:
            continue
        add(getattr(config_object, "eos_token_id", None))
        add(getattr(getattr(config_object, "text_config", None), "eos_token_id", None))

    return stop_token_ids


def _collect_suppressed_token_ids(tokenizer):
    suppressed = []
    seen = set()
    for token in ("<think>", "</think>"):
        token_id = tokenizer.convert_tokens_to_ids(token)
        try:
            token_id = int(token_id)
        except (TypeError, ValueError):
            continue
        if token_id < 0 or token_id in seen:
            continue
        seen.add(token_id)
        suppressed.append(token_id)
    return suppressed


def _resolve_stop_tokens_for_generation(self):
    stop_token_ids = getattr(self, "_prompt_enhancer_stop_token_ids", None)
    if stop_token_ids:
        return stop_token_ids[0] if len(stop_token_ids) == 1 else stop_token_ids
    return self.generation_config.eos_token_id


def _get_qwen35_text_runtime_helpers():
    from . import qwen35_text as qwen35_text_mod

    return qwen35_text_mod


def _apply_top_k_top_p(logits: torch.Tensor, top_k: int | None, top_p: float | None) -> torch.Tensor:
    logits = logits.clone()
    if top_k is not None and top_k > 0 and top_k < logits.shape[-1]:
        threshold = torch.topk(logits, int(top_k), dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
    if top_p is not None and 0.0 < float(top_p) < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_mask = cumulative_probs > float(top_p)
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(sorted_mask, float("-inf"))
        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)
    return logits


def _sample_next_token(
    logits: torch.Tensor,
    do_sample: bool,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    suppress_token_ids: set[int] | None = None,
) -> torch.Tensor:
    if suppress_token_ids:
        blocked = [token_id for token_id in suppress_token_ids if 0 <= token_id < logits.shape[-1]]
        if blocked:
            logits = logits.clone()
            logits[..., blocked] = float("-inf")
    if not do_sample:
        return torch.argmax(logits, dim=-1, keepdim=True)
    if temperature is None or float(temperature) <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    logits = logits / float(temperature)
    logits = _apply_top_k_top_p(logits, top_k=top_k, top_p=top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _trim_generated_ids(generated_ids: torch.Tensor, stop_token_ids: set[int]) -> list[list[int]]:
    rows = []
    for row in generated_ids.tolist():
        trimmed = []
        for token_id in row:
            if token_id in stop_token_ids:
                break
            trimmed.append(int(token_id))
        rows.append(trimmed)
    return rows


def _generate_and_decode(
    self,
    model_inputs,
    max_new_tokens,
    do_sample,
    temperature,
    top_p,
    top_k,
    seed,
    progress_desc: str | None = None,
):
    device = _resolve_execution_device(self, model_inputs)
    if seed is None:
        rng_context = nullcontext()
    else:
        devices = []
        if isinstance(device, torch.device) and device.type == "cuda":
            devices = [device.index or 0]
        rng_context = torch.random.fork_rng(devices=devices) if devices else torch.random.fork_rng()

    with rng_context, torch.inference_mode():
        if seed is not None:
            torch.manual_seed(int(seed))
            if isinstance(device, torch.device) and device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(int(seed))
        if hasattr(self, "model") and hasattr(self.model, "rope_deltas"):
            self.model.rope_deltas = None
        stop_token_ids = set(_iter_token_ids(_resolve_stop_tokens_for_generation(self)))
        suppress_token_ids = set(getattr(self, "_prompt_enhancer_suppress_token_ids", []) or [])
        current_inputs = {
            key: value
            for key, value in dict(model_inputs).items()
            if key not in {"use_cache", "return_dict", "output_attentions", "output_hidden_states"}
        }
        if current_inputs.get("pixel_values") is not None or current_inputs.get("pixel_values_videos") is not None:
            _prompt_token_ids, prompt_embeds, prompt_position_ids, _position_offset = _prepare_multimodal_vllm_prompt(self, current_inputs)
            current_inputs = {"inputs_embeds": prompt_embeds.unsqueeze(0)}
            if prompt_position_ids is not None:
                current_inputs["position_ids"] = prompt_position_ids.unsqueeze(1) if prompt_position_ids.ndim == 2 else prompt_position_ids
        generated_steps = []
        min_new_tokens = int(getattr(self, "_prompt_enhancer_min_new_tokens", 0) or 0)
        step_iter = range(int(max_new_tokens))
        if progress_desc:
            step_iter = tqdm(
                step_iter,
                total=int(max_new_tokens),
                desc=progress_desc,
                unit="tok",
                dynamic_ncols=True,
                leave=False,
            )
        for step in step_iter:
            forward_inputs = {
                key: value
                for key, value in current_inputs.items()
                if value is not None
            }
            outputs = self(
                **forward_inputs,
                use_cache=True,
                return_dict=True,
                logits_to_keep=1,
            )
            logits = outputs.logits[:, -1, :]
            step_suppress = suppress_token_ids | stop_token_ids if step < min_new_tokens else suppress_token_ids
            next_token = _sample_next_token(
                logits,
                do_sample=bool(do_sample),
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                suppress_token_ids=step_suppress,
            )
            generated_steps.append(next_token)
            if stop_token_ids and all(int(token_id) in stop_token_ids for token_id in next_token.view(-1).tolist()):
                break
            next_inputs_embeds = self._prompt_enhancer_text_model.embed_tokens(next_token)
            current_inputs = {
                "inputs_embeds": next_inputs_embeds,
                "past_key_values": outputs.past_key_values,
            }
        if generated_steps:
            generated_ids = torch.cat(generated_steps, dim=1)
        else:
            generated_ids = model_inputs["input_ids"][:, :0]
        decoded_ids = _trim_generated_ids(generated_ids, stop_token_ids)
        return [
            _clean_generated_text(text)
            for text in self._prompt_enhancer_tokenizer.batch_decode(
                decoded_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        ]


def _prepare_multimodal_vllm_prompt(self, model_inputs, image_features=None):
    runtime_model = self._caption_runtime_model
    model_inputs = _move_batch_to_device(model_inputs, _resolve_execution_device(self, model_inputs))
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs.get("attention_mask")
    image_grid_thw = model_inputs.get("image_grid_thw")
    video_grid_thw = model_inputs.get("video_grid_thw")
    mm_token_type_ids = model_inputs.get("mm_token_type_ids")
    with torch.inference_mode():
        if hasattr(runtime_model.model, "rope_deltas"):
            runtime_model.model.rope_deltas = None
        pixel_values = model_inputs.get("pixel_values")
        pixel_values_videos = model_inputs.get("pixel_values_videos")
        if pixel_values is not None:
            image_outputs = runtime_model.model.get_image_features(pixel_values, image_grid_thw, return_dict=True)
            image_features = image_outputs.pooler_output
        if pixel_values_videos is not None:
            video_outputs = runtime_model.model.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True)
        inputs_embeds = runtime_model.model.get_input_embeddings()(input_ids)
        if image_features is not None:
            image_embeds = torch.cat(image_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            if pixel_values is None:
                # Precomputed Deepy features replace whole token rows. Avoid expanding
                # the mask over hidden channels and materializing huge nonzero indices.
                inputs_embeds[input_ids == runtime_model.config.image_token_id] = image_embeds
                del image_embeds
            else:
                image_mask, _ = runtime_model.model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        if pixel_values_videos is not None:
            video_embeds = torch.cat(video_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = runtime_model.model.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)
        position_ids = runtime_model.model.compute_3d_position_ids(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            attention_mask=attention_mask,
            past_key_values=None,
            mm_token_type_ids=mm_token_type_ids,
        )
        active_mask = attention_mask[0].bool() if attention_mask is not None else torch.ones(input_ids.shape[1], device=input_ids.device, dtype=torch.bool)
        prompt_token_ids = [int(token_id) for token_id in input_ids[0][active_mask].tolist()]
        prompt_embeds = inputs_embeds[0, active_mask].contiguous()
        if position_ids is None:
            prompt_position_ids = None
        else:
            prompt_position_ids = position_ids[:, 0, active_mask].contiguous() if position_ids.ndim == 3 else position_ids[:, active_mask].contiguous()
        rope_deltas = getattr(runtime_model.model, "rope_deltas", None)
        position_offset = int(rope_deltas.reshape(-1)[0].item()) if torch.is_tensor(rope_deltas) and rope_deltas.numel() > 0 else 0
    return prompt_token_ids, prompt_embeds, prompt_position_ids, position_offset


TEXT_MODEL_ID = "llm_model"
VISION_MODEL_ID = "vision_tower_model"


@contextmanager
def vision_beside_text(caption_model, manager):
    """The vision tower loads beside the resident language model, whose weights stay (MMGP co-tenants), and the prompt's text embeddings
    read the language model's own token table (the separate embedding alias would load a second copy). Yields unload_vision, which
    releases the tower alone."""
    cotenants = manager.cotenants_map
    adapter = caption_model.model.language_model
    input_embedding = adapter._input_embedding_model

    def unload_vision():
        if VISION_MODEL_ID not in manager.active_models_ids:
            return
        torch.cuda.synchronize()
        # MMGP owns the CPU originals, including any streamed vision blocks.
        prefix = VISION_MODEL_ID + "/"
        for name in manager.blocks_of_modules:
            if name == VISION_MODEL_ID or name.startswith(prefix):
                manager.gpu_unload_blocks(VISION_MODEL_ID, None if name == VISION_MODEL_ID else name[len(prefix):])
        index = manager.active_models_ids.index(VISION_MODEL_ID)
        manager.active_models_ids.pop(index)
        manager.active_models.pop(index)
        torch.cuda.empty_cache()

    try:
        manager.cotenants_map = {**cotenants, TEXT_MODEL_ID: [*cotenants.get(TEXT_MODEL_ID, []), VISION_MODEL_ID], VISION_MODEL_ID: [*cotenants.get(VISION_MODEL_ID, []), TEXT_MODEL_ID]}
        object.__setattr__(adapter, "_input_embedding_model", caption_model._prompt_enhancer_text_model.token_embd)
        yield unload_vision
    finally:
        unload_vision()
        object.__setattr__(adapter, "_input_embedding_model", input_embedding)
        manager.cotenants_map = cotenants


def _captions_beside(self, offload_manager) -> bool:
    """An assistant's live context in the engine (Deepy kept loaded) and the MMGP dynamic preload: the captions run beside it (the tower
    beside the resident language model, the requests in the free pages of its cache) instead of closing it."""
    from mmgp import runtime as mmgp_runtime
    text_model = self._prompt_enhancer_text_model
    engine = getattr(text_model, "_prompt_enhancer_vllm_engine", None)
    return (offload_manager is not None and engine is not None and engine.has_live_context() and mmgp_runtime.dynamic_preload(text_model)
            and offload_manager.models.get(TEXT_MODEL_ID) is text_model and TEXT_MODEL_ID in offload_manager.active_models_ids)


def _generate_image_captions_vllm(self, images, *, image_contexts=None, offload_manager=None, enhancement_progress=None, beside=False):
    qwen35_text_mod = _get_qwen35_text_runtime_helpers()
    text_model = self._prompt_enhancer_text_model
    tokenizer = self._prompt_enhancer_tokenizer
    processor = self._prompt_enhancer_processor
    live = text_model._prompt_enhancer_vllm_engine if beside else None
    if not beside:
        # Finish every vision pass before loading the shared decoder. A previous
        # request's graphs must be closed before MMGP can change its weight storage.
        text_model.unload()
        if offload_manager is not None:
            offload_manager.unload_all()
    groups = [([image], None) for image in images] if image_contexts is None else [(context.images, context.labels) for context in image_contexts]
    progress = enhancement_progress or EnhancementProgress()
    image_count = len({id(image) for group, _ in groups for image in group})
    prepared = []
    features = {}
    outputs = []
    # beside: the tower may take the idle cache's VRAM if it runs short (its owner saves the context first)
    with vision_beside_text(self, offload_manager) if beside else nullcontext(), live.idle() if beside else nullcontext():
        for group_images, labels in tqdm(groups, desc="Encoding Prompt Images", dynamic_ncols=True, leave=False):
            check_cancelled()
            if not group_images:
                prepared.append(None)
                continue
            caption_tokens = 128 if labels is None else 128 * len(group_images) + 128
            content = []
            for index, image in enumerate(group_images):
                if labels is not None:
                    content.append({"type": "text", "text": labels[index]})
                content.append({"type": "image", "image": image})
            if labels is None:
                message = [{"role": "user", "content": content + [{"type": "text", "text": "Describe this image accurately in one concise paragraph, focusing on the main subject, setting, and notable objects. Output only the description."}]}]
            else:
                names = "; ".join(labels)
                instructions = f"Describe these labeled images accurately: {names}. Use each exact image name as a heading, followed by a concise description of its subject, setting and notable details. Start and end images anchor the opening and ending of this window; frame labels locate intermediate anchors within it. Reference images guide appearance. Distinguish the images and mention visible relationships without inventing motion or unseen events. Output only the labeled descriptions."
                headings = "\n".join(f"{label}: <visible details>" for label in labels)
                content.append({"type": "text", "text": f"Describe all {len(labels)} separate still images in the order shown. Use each heading below exactly once, without renaming, adding or omitting any heading. Describe only visible details; do not infer movement from still images.\n{headings}"})
                message = [{"role": "system", "content": instructions}, {"role": "user", "content": content}]
            text = processor.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            # PIL sources and pending pixel batches stay in RAM even if another
            # model left PyTorch's default device set to CUDA.
            with torch.device("cpu"):
                model_inputs = processor(
                    text=[text],
                    images=[_resize_image_for_caption(image) for image in group_images],
                    return_tensors="pt",
                    padding=True,
                    return_mm_token_type_ids=True,
                )
            # Bound vision activation memory to one image and reuse repeated window
            # anchors. Retain features/embeddings on CPU between phases.
            pixels = model_inputs.pop("pixel_values")
            grids = model_inputs["image_grid_thw"]
            image_features, first_patch = [], 0
            with torch.inference_mode():
                for image, grid in zip(group_images, grids.tolist()):
                    check_cancelled()
                    patch_count = grid[0] * grid[1] * grid[2]
                    key = (id(image), tuple(grid))
                    if key not in features:
                        with progress.vision(self.vision_tower_model.blocks, len(features), image_count):
                            output = self.model.get_image_features(pixels[first_patch:first_patch + patch_count], grids.new_tensor([grid]), return_dict=True)
                        features[key] = output.pooler_output[0].to("cpu")
                        del output
                    image_features.append(features[key])
                    first_patch += patch_count
                prompt_token_ids, prompt_embeds, prompt_position_ids, position_offset = _prepare_multimodal_vllm_prompt(self, model_inputs, image_features=image_features)
                prepared.append((prompt_token_ids, prompt_embeds.to("cpu"), None if prompt_position_ids is None else prompt_position_ids.to("cpu"), position_offset, caption_tokens, message, text))
            del pixels, image_features, model_inputs, prompt_embeds, prompt_position_ids
    features.clear()
    # MMGP may have released the idle cache meanwhile, or the requests may not fit beside the context: then as without it
    beside = beside and live.has_live_context() and live._llm.can_aside(max((len(request[0]) + request[4] for request in prepared if request is not None), default=0))
    if beside:
        engine = live
    else:
        if live is not None:
            live.save_context()  # no room beside it: its owner saves it to RAM (restored at its next use), the captions run as alone
        elif offload_manager is not None:
            offload_manager.unload_all()
        # Captioning and enhancement use the same cache policy and engine identity.
        engine = qwen35_text_mod._get_or_create_vllm_engine(text_model, usage_mode="text")
        for request in prepared:
            if request is not None:
                engine.reserve_runtime(prompt_len=len(request[0]), max_tokens=request[4], cfg_scale=1.0, min_model_len=qwen35_text_mod.QWEN35_PROMPT_MIN_MODEL_LEN)
    with live._llm.aside() if beside else nullcontext():
        for index, request in enumerate(prepared):
            check_cancelled()
            if request is None:
                outputs.append("")
                continue
            prompt_token_ids, prompt_embeds, prompt_position_ids, position_offset, caption_tokens, message, text = request
            progress.caption(index, len(groups), caption_tokens)
            engine._ensure_llm()
            if engine._llm is None:
                raise RuntimeError("Qwen3.5 caption vLLM runtime is not available.")
            temp, normalized_top_p, normalized_top_k = qwen35_text_mod._normalize_vllm_sampling(
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
            )
            if llm_io_enabled():
                log_llm_io("OUT", "local-image-captioner", "qwen-visual-generation", {
                    "prompt": text,
                    "messages": message,
                    "input_token_ids": prompt_token_ids,
                    "known_token_ids": known_token_ids(tokenizer),
                    "prompt_embeddings": prompt_embeds,
                    "prompt_position_ids": prompt_position_ids,
                    "position_offset": position_offset,
                    "generation": {"max_new_tokens": caption_tokens, "temperature": temp, "top_p": normalized_top_p, "top_k": normalized_top_k, "do_sample": False},
                })
            response = engine.generate_embedded(
                prompt_token_ids=prompt_token_ids,
                prompt_embeds=prompt_embeds,
                prompt_position_ids=prompt_position_ids,
                max_tokens=caption_tokens,
                temperature=temp,
                top_p=normalized_top_p,
                top_k=normalized_top_k,
                cfg_scale=1.0,
                seed=0,  # greedy: the seed settles exact ties, so an image's caption does not depend on the sampler's state
                use_tqdm=True,
                release_vram_after=False,
                ignore_eos=False,
                position_offset=position_offset,
                stream_callback=progress.tokens if enhancement_progress is not None else None,
            )
            raw_text = "" if response is None else response.get("text", "")
            log_llm_io("IN", "local-image-captioner", "qwen-visual-generation", {"text": raw_text, "response": response})
            outputs.append(_clean_generated_text(raw_text))
            reset_context()
    return outputs


def _generate_image_captions(self, images, *, image_contexts=None, offload_manager=None, stop_requested=None, enhancement_progress=None):
    qwen35_text_mod = _get_qwen35_text_runtime_helpers()
    if qwen35_text_mod._use_vllm_prompt_enhancer(self._prompt_enhancer_text_model) or qwen35_text_mod._use_legacy_cuda_runner_prompt_enhancer(self._prompt_enhancer_text_model):
        def check_stop():
            if stop_requested is not None and stop_requested():
                raise InterruptedError("Prompt Enhancement Cancelled")
        hooks = [block.register_forward_pre_hook(lambda *_: check_cancelled()) for block in self.vision_tower_model.blocks]
        beside = _captions_beside(self, offload_manager)
        try:
            with cancellation_context(check_stop) if stop_requested is not None else nullcontext():
                return _generate_image_captions_vllm(self, images, image_contexts=image_contexts, offload_manager=offload_manager, enhancement_progress=enhancement_progress, beside=beside)
        except BaseException:
            if not beside:  # beside, the tower is already released and the assistant's context stays
                self._prompt_enhancer_text_model.unload()
                if offload_manager is not None:
                    offload_manager.unload_all()
            raise
        finally:
            for hook in hooks:
                hook.remove()
    images = [_resize_image_for_caption(image) for image in images]
    outputs = []
    processor = self._prompt_enhancer_processor
    for image in images:
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": "Describe this image accurately in one concise paragraph, focusing on the main subject, setting, and notable objects. Output only the description.",
                    },
                ],
            }
        ]
        text = processor.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        model_inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
            return_mm_token_type_ids=True,
        )
        model_inputs = _move_batch_to_device(model_inputs, torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu"))
        if llm_io_enabled():
            log_llm_io("OUT", "local-image-captioner", "qwen-visual-generation", {
                "prompt": text,
                "messages": message,
                "image": media_descriptor(image),
                "input_token_ids": model_inputs["input_ids"].tolist(),
                "known_token_ids": known_token_ids(self._prompt_enhancer_tokenizer),
                "generation": {"max_new_tokens": 128, "do_sample": False},
            })
        decoded = _generate_and_decode(
            self,
            model_inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            seed=None,
            progress_desc="Qwen3.5 image description tokens",
        )
        log_llm_io("IN", "local-image-captioner", "qwen-visual-generation", {"text": decoded})
        outputs.extend(decoded)
    return outputs


def _unload_prompt_enhancer_vl_runtime(self):
    text_model = getattr(self, "_prompt_enhancer_text_model", None)
    unload = getattr(text_model, "unload", None)
    if callable(unload):
        unload()
    return None


def get_qwen35_vision_path(assets_dir: str, variant: str | None = None) -> str:
    filename = get_qwen35_variant_spec(variant)["vision_filename"]
    return _resolve_qwen35_checkpoint_file(assets_dir, filename, variant=variant, error_if_none=False)


def load_qwen35_vl_prompt_enhancer(
    model_path: str | None = None,
    assets_dir: str | None = None,
    attn_implementation: str = "sdpa",
    text_model: torch.nn.Module | None = None,
    input_embedding_model: torch.nn.Module | None = None,
    backend: str = enhancer_quantization_QUANTO_INT8,
    variant: str | None = None,
):
    assets_dir = _resolve_qwen35_assets_dir(assets_dir, variant=variant)
    if text_model is None:
        raise ValueError("A loaded Qwen3.5 text model is required to build the multimodal prompt enhancer.")
    legacy_safe_mode = bool(getattr(text_model, "_prompt_enhancer_safe_legacy", False))
    if legacy_safe_mode:
        attn_implementation = "sdpa"

    config_path = _resolve_qwen35_asset_file(assets_dir, "config.json", error_if_none=True)
    modeling_path = get_qwen35_modeling_path()
    for required_file in (
        config_path,
        _resolve_qwen35_asset_file(assets_dir, "tokenizer.json", error_if_none=True),
        _resolve_qwen35_asset_file(assets_dir, "tokenizer_config.json", error_if_none=True),
        _resolve_qwen35_asset_file(assets_dir, "video_preprocessor_config.json", error_if_none=True),
    ):
        if not os.path.isfile(required_file):
            raise FileNotFoundError(f"Missing Qwen3.5VL asset: {required_file}")
    if not os.path.isfile(modeling_path):
        raise FileNotFoundError(f"Missing repo-local Qwen3.5 modeling file: {modeling_path}")

    if model_path is None:
        model_path = get_qwen35_vision_path(assets_dir, variant=variant)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Qwen3.5VL checkpoint not found: {model_path}")

    model_class = load_qwen35_model_class(modeling_path, class_name="Qwen3_5ForConditionalGeneration")
    config = AutoConfig.from_pretrained(assets_dir, trust_remote_code=True)
    config._prompt_enhancer_safe_legacy = legacy_safe_mode
    config._attn_implementation = attn_implementation
    if hasattr(config, "text_config") and config.text_config is not None:
        config.text_config._prompt_enhancer_safe_legacy = legacy_safe_mode
        config.text_config._attn_implementation = attn_implementation
    if hasattr(config, "vision_config") and config.vision_config is not None:
        config.vision_config._attn_implementation = attn_implementation
    with torch.device("meta"):
        model = model_class(config)
    model.model.visual = model.model.visual.__class__._from_config(config.vision_config)
    model.model.language_model = _SharedQwen35TextAdapter(text_model, input_embedding_model)
    model.lm_head = _SharedQwen35LmHeadAdapter(text_model.lm_head)
    if str(model_path).lower().endswith(".gguf"):
        preprocess_sd = _build_qwen35_vl_gguf_preprocess_sd(tuple(model.model.visual.patch_embed.proj.weight.shape))
        offload.load_model_data(
            model.model.visual,
            model_path,
            preprocess_sd=preprocess_sd,
            writable_tensors=False,
            default_dtype=torch.bfloat16,
        )
        materialize_module_source_tensors(model.model.visual)
        model.model.visual.to(dtype=torch.float16)
    else:
        offload.load_model_data(
            model.model.visual,
            model_path,
            modelPrefix="model.visual",
            writable_tensors=False,
            default_dtype=torch.bfloat16,
        )

    tokenizer = _load_qwen35_tokenizer(assets_dir)
    image_processor = _load_qwen35_image_processor(assets_dir)
    video_processor = Qwen2VLVideoProcessor.from_pretrained(assets_dir)
    processor = _PromptEnhancerQwen2VLProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        chat_template=tokenizer.chat_template,
    )
    processor.chat_template = tokenizer.chat_template

    caption_model = _Qwen35CaptionWrapper(
        caption_runtime_model=model,
        vision_tower_model=model.model.visual,
        text_model=text_model,
        tokenizer=tokenizer,
        processor=processor,
    )
    caption_model.generate_image_captions = types.MethodType(_generate_image_captions, caption_model)
    caption_model.unload = types.MethodType(_unload_prompt_enhancer_vl_runtime, caption_model)

    stop_token_ids = _collect_stop_token_ids(
        tokenizer,
        model.generation_config,
        model.config,
    )
    if not stop_token_ids:
        raise RuntimeError("Could not determine Qwen3.5VL stop token ids.")
    caption_model._prompt_enhancer_stop_token_ids = stop_token_ids
    caption_model._prompt_enhancer_suppress_token_ids = _collect_suppressed_token_ids(tokenizer)
    raw_fast_env = str(os.environ.get(QWEN35_GGUF_LLAMACPP_ENV, "1")).strip().lower()
    caption_model._prompt_enhancer_min_new_tokens = (
        QWEN35_PROMPT_MIN_NEW_TOKENS
        if backend == enhancer_quantization_GGUF and raw_fast_env in ("1", "true", "yes", "y", "on")
        else 0
    )
    caption_model.generation_config.eos_token_id = stop_token_ids[0] if len(stop_token_ids) == 1 else stop_token_ids

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = stop_token_ids[0]
    caption_model.generation_config.pad_token_id = int(pad_token_id)

    if hasattr(caption_model.config, "text_config") and hasattr(caption_model.config.text_config, "eos_token_id"):
        caption_model.config.text_config.eos_token_id = stop_token_ids[0]
    if hasattr(caption_model.config, "text_config") and hasattr(caption_model.config.text_config, "pad_token_id"):
        caption_model.config.text_config.pad_token_id = int(pad_token_id)
    caption_model.eval()
    return caption_model, caption_model.vision_tower_model


__all__ = [
    "QWEN35_ABLITERATED_TEXT_REQUIRED_FILES",
    "QWEN35_4B_TEXT_INT8_FILENAME",
    "QWEN35_VARIANT_9B",
    "QWEN35_VARIANT_4B",
    "enhancer_quantization_GGUF",
    "enhancer_quantization_GGUF_Q3",
    "enhancer_quantization_GGUF_Q2",
    "enhancer_quantization_GGUF_PTQ1",
    "enhancer_quantization_GGUF_Q8",
    "enhancer_quantization_SAFETENSORS",
    "enhancer_quantization_QUANTO_INT8",
    "QWEN35_TEXT_GGUF_FILENAME",
    "QWEN35_VISION_FILENAME",
    "UPSTREAM_MODELING_FILENAME",
    "get_qwen35_prompt_enhancer_variant",
    "get_qwen35_quantization",
    "get_qwen35_assets_dir_name",
    "get_qwen35_modeling_path",
    "get_qwen35_variant_spec",
    "ensure_qwen35_prompt_enhancer_assets",
    "get_qwen35_text_gguf_path",
    "get_qwen35_vision_path",
    "load_qwen35_vl_prompt_enhancer",
]
