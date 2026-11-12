# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import os
from typing import Any, Callable

from postprocessing.spatial_upsamplers import format_multiplier_value, parse_multiplier_suffix
from postprocessing.flashvsr.sparse_backend_config import (
    SPARSE_BACKEND_AUTO,
    SPARSE_BACKEND_CHOICES,
    SPARSE_BACKEND_SPARGE,
    SPARSE_BACKEND_TRITON_SPARSE,
    normalize_sparse_backend,
)


class FlashVSRBridge:
    MODE_OFF = 0
    MODE_TINY = 1
    MODE_FULL = 2
    MODE_TINY_LONG = 3
    BACKEND_AUTO = SPARSE_BACKEND_AUTO
    BACKEND_TRITON_SPARSE = SPARSE_BACKEND_TRITON_SPARSE
    BACKEND_SPARGE = SPARSE_BACKEND_SPARGE
    TOPK_RATIO_DEFAULT = 0.0
    TOPK_RATIO_MAX = 4.0
    UPSAMPLING_VALUE_PREFIX = "flashvsr"
    UPSAMPLING_TWO_PASS_VALUE_PREFIX = "flashvsr2pass"
    UPSAMPLING_PLUS_VALUE_PREFIX = "flashvsrplus"
    UPSAMPLING_PLUS_TWO_PASS_VALUE_PREFIX = "flashvsrplus2pass"
    UPSAMPLING_RATIOS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
    MODE_CHOICES = [
        ("Tiny Decoder (Slightly Lower Quality, Faster Decoding, Needs Less RAM)", MODE_TINY),
        ("Full Wan VAE Decoder (Best Quality, Slower Decoding, Needs More RAM)", MODE_FULL),
    ]
    MODEL_V11 = "v1_1"
    MODEL_PLUS_TURBO = "plus_turbo"
    MODEL_CHOICES = [
        ("FlashVSR+ Turbo (Hybrid Attention: Faster At High Resolutions, Preference-Aligned Detail)", MODEL_PLUS_TURBO),
        ("FlashVSR v1.1", MODEL_V11),
    ]
    MODEL_DEFAULT = MODEL_PLUS_TURBO  # offer FlashVSR+ Turbo by default in the upsampling menus
    PLUS_MODELS = (MODEL_PLUS_TURBO,)
    METHOD_MODELS = {
        UPSAMPLING_VALUE_PREFIX: MODEL_V11,
        UPSAMPLING_TWO_PASS_VALUE_PREFIX: MODEL_V11,
        UPSAMPLING_PLUS_VALUE_PREFIX: MODEL_PLUS_TURBO,
        UPSAMPLING_PLUS_TWO_PASS_VALUE_PREFIX: MODEL_PLUS_TURBO,
    }
    QUANTIZATION_BF16 = "bf16"
    QUANTIZATION_INT8 = "int8"
    QUANTIZATION_CHOICES = [("BF16", QUANTIZATION_BF16), ("INT8 ConvRot (Half The Transformer Size, Faster With INT8 Kernels)", QUANTIZATION_INT8)]
    QUANTIZATION_DEFAULT = QUANTIZATION_BF16
    CACHE_WINDOWS_DEFAULT = 3
    CACHE_WINDOWS_MAX = 3

    # transformer and LQ projection file stems per model; the TCDecoder, the Wan VAE and the prompt tensor are shared by every model
    _MODEL_FILE_STEMS = {MODEL_V11: "FlashVSR_v1.1", MODEL_PLUS_TURBO: "FlashVSR_plus_turbo"}
    TRANSFORMER_FILENAME = "FlashVSR_v1.1_transformer_bf16.safetensors"
    LQ_PROJ_FILENAME = "FlashVSR_v1.1_lq_proj_bf16.safetensors"
    TCDECODER_FILENAME = "FlashVSR_v1.1_tcdecoder_bf16.safetensors"
    POSI_PROMPT_FILENAME = "FlashVSR_v1.1_posi_prompt_bf16.safetensors"
    VAE_FILENAME = "Wan2.1_VAE.safetensors"

    _VARIANTS = {
        MODE_TINY: "tiny",
        MODE_FULL: "full",
        MODE_TINY_LONG: "tiny-long",
    }

    def __init__(self, server_config: dict[str, Any], files_locator):
        self.server_config = server_config
        self.files_locator = files_locator

    @classmethod
    def transformer_filename(cls, model: str, quantization: str) -> str:
        return f"{cls._MODEL_FILE_STEMS[model]}_transformer_{'int8_convrot' if quantization == cls.QUANTIZATION_INT8 else 'bf16'}.safetensors"

    @classmethod
    def lq_proj_filename(cls, model: str) -> str:
        return f"{cls._MODEL_FILE_STEMS[model]}_lq_proj_bf16.safetensors"

    @classmethod
    def normalize_model(cls, value: Any) -> str:
        value = str(value or cls.MODEL_DEFAULT).strip().lower()
        return value if value in cls._MODEL_FILE_STEMS else cls.MODEL_DEFAULT

    @classmethod
    def normalize_quantization(cls, value: Any) -> str:
        value = str(value or cls.QUANTIZATION_DEFAULT).strip().lower()
        return value if value in (cls.QUANTIZATION_BF16, cls.QUANTIZATION_INT8) else cls.QUANTIZATION_DEFAULT

    @classmethod
    def normalize_cache_windows(cls, value: Any) -> int:
        try:
            value = int(value)
        except (TypeError, ValueError):
            value = cls.CACHE_WINDOWS_DEFAULT
        return max(1, min(cls.CACHE_WINDOWS_MAX, value))

    @classmethod
    def normalize_topk_ratio(cls, value: Any) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = cls.TOPK_RATIO_DEFAULT
        return max(0.0, min(cls.TOPK_RATIO_MAX, value))

    @classmethod
    def normalize_backend(cls, value: Any) -> str:
        return normalize_sparse_backend(value)

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return {"mode": cls.MODE_OFF, "model": cls.MODEL_DEFAULT, "quantization": cls.QUANTIZATION_DEFAULT, "backend": cls.BACKEND_AUTO, "topk_ratio": cls.TOPK_RATIO_DEFAULT, "cache_windows": cls.CACHE_WINDOWS_DEFAULT}

    @classmethod
    def legacy_config_keys(cls) -> tuple[str, ...]:
        return ("flashvsr_mode", "flashvsr_persistence", "flashvsr_backend", "flashvsr_topk_ratio")

    @classmethod
    def legacy_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        return {
            "mode": config.get("flashvsr_mode", cls.MODE_OFF),
            "backend": config.get("flashvsr_backend", cls.BACKEND_AUTO),
            "topk_ratio": config.get("flashvsr_topk_ratio", cls.TOPK_RATIO_DEFAULT),
        }

    @classmethod
    def normalize_config_section(cls, config: dict[str, Any]) -> dict[str, Any]:
        normalized = cls.default_config()
        normalized.update(config or {})
        try:
            normalized["mode"] = int(normalized.get("mode", cls.MODE_OFF))
        except (TypeError, ValueError):
            normalized["mode"] = cls.MODE_OFF
        if normalized["mode"] not in cls._VARIANTS and normalized["mode"] != cls.MODE_OFF:
            normalized["mode"] = cls.MODE_OFF
        normalized["model"] = cls.normalize_model(normalized.get("model", cls.MODEL_DEFAULT))
        normalized["quantization"] = cls.normalize_quantization(normalized.get("quantization", cls.QUANTIZATION_DEFAULT))
        normalized["backend"] = cls.normalize_backend(normalized.get("backend", cls.BACKEND_AUTO))
        normalized["topk_ratio"] = cls.normalize_topk_ratio(normalized.get("topk_ratio", cls.TOPK_RATIO_DEFAULT))
        normalized["cache_windows"] = cls.normalize_cache_windows(normalized.get("cache_windows", cls.CACHE_WINDOWS_DEFAULT))
        return normalized

    @classmethod
    def apply_pre_1_1_defaults(cls, config: dict[str, Any]) -> bool:
        if config["mode"] == cls.MODE_OFF:
            config["mode"] = cls.MODE_TINY
            return True
        return False

    def config(self, config: dict[str, Any] | None = None) -> dict[str, Any]:
        from postprocessing import spatial_upsamplers as upsampler_api

        return upsampler_api.read_config_section(self.server_config if config is None else config, self)

    def normalize_config(self, config: dict[str, Any] | None = None) -> int:
        from postprocessing import spatial_upsamplers as upsampler_api

        config = self.server_config if config is None else config
        section = upsampler_api.write_config_section(config, self, self.config(config))
        return section["mode"]

    def settings(self, config: dict[str, Any] | None = None) -> tuple[bool, str | None]:
        mode = self.normalize_config(config)
        return mode != self.MODE_OFF, self._VARIANTS.get(mode)

    def topk_ratio(self) -> float:
        return self.config()["topk_ratio"]

    def backend(self) -> str:
        return self.config()["backend"]

    def model(self) -> str:
        return self.config()["model"]

    def quantization(self) -> str:
        return self.config()["quantization"]

    def cache_windows(self) -> int:
        return self.config()["cache_windows"]

    def overlap_frames(self, spatial_upsampling=None) -> int:
        """The output frames of a chunk that the next source chunk regenerates (the Media Flow overlap and the continuation tail)."""
        from postprocessing.flashvsr.runtime import continue_cache_frames
        return continue_cache_frames(self.model() if spatial_upsampling is None else self.model_for_upsampling(spatial_upsampling))

    def enabled(self) -> bool:
        return self.settings()[0]

    @classmethod
    def format_ratio(cls, scale: float) -> str:
        scale = float(scale)
        return str(int(scale)) if scale.is_integer() else f"{scale:g}"

    @classmethod
    def format_ratio_label(cls, scale: float) -> str:
        return cls.format_ratio(scale)

    @classmethod
    def upsampling_value(cls, scale: float) -> str:
        return format_multiplier_value(cls.UPSAMPLING_VALUE_PREFIX, scale)

    @classmethod
    def upsampling_two_pass_value(cls, scale: float) -> str:
        return format_multiplier_value(cls.UPSAMPLING_TWO_PASS_VALUE_PREFIX, scale)

    @classmethod
    def scale_for_upsampling(cls, spatial_upsampling) -> float | None:
        split = cls.split_value(spatial_upsampling)
        return None if split is None else split[1]

    @classmethod
    def is_two_pass_upsampling(cls, spatial_upsampling) -> bool:
        split = cls.split_value(spatial_upsampling)
        return split is not None and split[0] in (cls.UPSAMPLING_TWO_PASS_VALUE_PREFIX, cls.UPSAMPLING_PLUS_TWO_PASS_VALUE_PREFIX)

    @classmethod
    def model_for_upsampling(cls, spatial_upsampling) -> str:
        split = cls.split_value(spatial_upsampling)
        if split is None:
            raise ValueError(f"Unknown FlashVSR upsampling mode: {spatial_upsampling}")
        return cls.METHOD_MODELS[split[0]]

    def method_visible(self, method: str) -> bool:
        return self.METHOD_MODELS[method] == self.model()

    def preferred_method(self, method: str) -> str:
        two_pass = self.is_two_pass_upsampling(method)
        return next(key for key in self.METHOD_MODELS if self.method_visible(key) and self.is_two_pass_upsampling(key) == two_pass)

    @classmethod
    def query_upsampler_def(cls) -> dict[str, Any]:
        return {
            "name": "FlashVSR",
            "upsampler_types": ("postprocessing",),
            "media": ("video", "image"),
            "profile": "video",
            "config_key": "flashvsr",
            "pos": 20,
            "method_pos": {method: 20 + index for index, method in enumerate(cls.METHOD_MODELS)},
            "methods": [("FlashVSR v1.1", cls.UPSAMPLING_VALUE_PREFIX), ("FlashVSR v1.1 Two Pass", cls.UPSAMPLING_TWO_PASS_VALUE_PREFIX),
                        ("FlashVSR+ Turbo", cls.UPSAMPLING_PLUS_VALUE_PREFIX), ("FlashVSR+ Turbo Two Pass", cls.UPSAMPLING_PLUS_TWO_PASS_VALUE_PREFIX)],
            "vae_methods": [],
            "multipliers": {method: cls.UPSAMPLING_RATIOS for method in cls.METHOD_MODELS},
            "default_spatial_upsampling": cls.upsampling_value(2.0),
            "postprocessing_category": "upsampler",
            "description": "Provides learned detail recovery and is optimized for modest VRAM use. Configuration > Extensions chooses which version is normally offered: FlashVSR v1.1 or FlashVSR+ Turbo (faster at high resolutions, preference-aligned detail). Extract Settings restores the version saved with the media even when it is hidden by the current configuration. The Full decoder's final VAE decode can be slow; Tiny is faster, and Two Pass may reduce horizontal banding.",
            "media_descriptions": {"video": "Its learned reconstruction is designed for good temporal stability."},
            "method_descriptions": {
                cls.UPSAMPLING_VALUE_PREFIX: "Restore detail and spatially upscale with FlashVSR v1.1. The Full decoder's final VAE decode can be slow; Tiny is faster and large outputs can consume substantial VRAM.",
                cls.UPSAMPLING_TWO_PASS_VALUE_PREFIX: "Apply two FlashVSR v1.1 restoration passes for stronger detail reconstruction and reduced horizontal banding, at approximately twice the processing cost.",
                cls.UPSAMPLING_PLUS_VALUE_PREFIX: "Restore detail and spatially upscale with FlashVSR+ Turbo, with faster processing at high resolutions and preference-aligned detail.",
                cls.UPSAMPLING_PLUS_TWO_PASS_VALUE_PREFIX: "Apply two FlashVSR+ Turbo restoration passes for stronger detail reconstruction and reduced horizontal banding, at approximately twice the processing cost.",
            },
        }

    @classmethod
    def split_value(cls, value) -> tuple[str, float] | None:
        text = str(value or "").strip().lower()
        for method in sorted(cls.METHOD_MODELS, key=len, reverse=True):
            scale = parse_multiplier_suffix(text, method, 2.0)
            if scale in cls.UPSAMPLING_RATIOS:
                return method, scale
        return None

    @classmethod
    def build_value(cls, method, scale) -> str | None:
        scale = float(scale or 2.0)
        if scale not in cls.UPSAMPLING_RATIOS:
            scale = 2.0
        return format_multiplier_value(method, scale) if method in cls.METHOD_MODELS else None

    def is_upsampling(self, spatial_upsampling) -> bool:
        return self.scale_for_upsampling(spatial_upsampling) is not None

    def query_chunking(self, spatial_upsampling):
        """How a long video goes through this upsampler a chunk at a time (postprocessing.chunking)."""
        from .process_handler import HANDLER
        return HANDLER

    def validate_upsampling(self, spatial_upsampling, image_mode: int) -> str:
        if not self.is_upsampling(spatial_upsampling):
            return ""
        if not self.enabled():
            return "FlashVSR Spatial Upsampling is disabled in Configuration > Extensions"
        return ""

    def create_config_ui(self, gr, config: dict[str, Any], *, lock_config: bool = False):
        with gr.Group():
            with gr.Row():
                mode = gr.Dropdown(choices=self.MODE_CHOICES, value=config["mode"], label="FlashVSR Spatial Upsampling (Needs Triton; SpargeAttn optional)", interactive=not lock_config)
            with gr.Row():
                model = gr.Dropdown(choices=self.MODEL_CHOICES, value=config["model"], label="Model", info="Chooses the version normally shown in upsampling menus. Extract Settings can restore the other version without changing this preference. FlashVSR+ Turbo always runs with memory profile 0 and processes 16 frames per step.", interactive=not lock_config)
                quantization = gr.Dropdown(choices=self.QUANTIZATION_CHOICES, value=config["quantization"], label="Transformer Quantization", interactive=not lock_config)
            with gr.Row():
                backend = gr.Dropdown(choices=SPARSE_BACKEND_CHOICES, value=config["backend"], label="Backend", interactive=not lock_config)
                topk_ratio = gr.Slider(0.0, self.TOPK_RATIO_MAX, value=config["topk_ratio"], step=0.05, label="FlashVSR Quality / Sparse Top-K Ratio (0 = Auto)", info="Higher keeps more sparse attention candidates and can improve quality at the cost of speed and memory. Auto: the released setting (FlashVSR+: 2.5 per 16-frame step).", interactive=not lock_config)
                cache_windows = gr.Slider(1, self.CACHE_WINDOWS_MAX, value=config["cache_windows"], step=1, label="FlashVSR+ Cache Windows", info="Previous 8-frame windows the FlashVSR+ global attention layers remember (3 = the released setting). Fewer windows save RAM at high resolutions with less temporal context.", interactive=not lock_config)
        return [("mode", mode), ("model", model), ("quantization", quantization), ("backend", backend), ("topk_ratio", topk_ratio), ("cache_windows", cache_windows)]

    def validate_config_section(self, config: dict[str, Any]):
        if config["mode"] <= 0:
            return ""
        try:
            from postprocessing.flashvsr.attention_backend import sparse_attention_requirement_message
            return sparse_attention_requirement_message(config["backend"]) or ""
        except Exception as exc:
            return f"FlashVSR sparse attention dependency check failed: {type(exc).__name__}: {exc}"

    def config_requires_release(self, old_config: dict[str, Any], new_config: dict[str, Any], changed_keys: set[str]) -> bool:
        return old_config != new_config or bool({"profile", "video_profile", "vae_config"} & changed_keys)

    def query_download_def(self, enabled_only: bool = True, spatial_upsampling=None) -> dict[str, Any] | None:
        if enabled_only and not self.enabled():
            return None
        _, variant = self.settings()
        model = self.model() if spatial_upsampling is None else self.model_for_upsampling(spatial_upsampling)
        return self._model_download_def(model, self.quantization(), variant)

    def query_checkpoint_inventory(self):
        for label, model in self.MODEL_CHOICES:
            for _, quant in self.QUANTIZATION_CHOICES:
                for mode in ("tiny", "full"):  # Tiny-Long uses the same files as Tiny.
                    yield {"name": label.split(" (")[0], "variant": f"{quant.upper()} {mode.title()}", "download_defs": [self._model_download_def(model, quant, mode)]}

    def _model_download_def(self, model, quant, variant):
        files = [self.transformer_filename(model, quant), self.lq_proj_filename(model), self.POSI_PROMPT_FILENAME]
        if variant != "full":
            files.append(self.TCDECODER_FILENAME)
        return {
            "repoId": "DeepBeepMeep/Wan2.1",
            "sourceFolderList": ["FlashVSR", ""] if variant == "full" else ["FlashVSR"],
            "fileList": [files, [self.VAE_FILENAME]] if variant == "full" else [files],
        }

    def _locate_flashvsr_file(self, filename: str) -> str:
        return self.files_locator.locate_file(os.path.join("FlashVSR", filename))

    def paths(self, variant: str, model: str | None = None, quantization: str | None = None):
        from postprocessing.flashvsr.runtime import FlashVSRPaths
        model = self.normalize_model(self.model() if model is None else model)
        quantization = self.normalize_quantization(self.quantization() if quantization is None else quantization)
        return FlashVSRPaths(
            transformer=self._locate_flashvsr_file(self.transformer_filename(model, quantization)),
            lq_proj=self._locate_flashvsr_file(self.lq_proj_filename(model)),
            posi_prompt=self._locate_flashvsr_file(self.POSI_PROMPT_FILENAME),
            tcdecoder=None if variant == "full" else self._locate_flashvsr_file(self.TCDECODER_FILENAME),
            vae=self.files_locator.locate_file(self.VAE_FILENAME) if variant == "full" else None,
        )

    def vae_tile_size(self, vae_config: int, output_height: int | None = None, output_width: int | None = None) -> int:
        import torch
        from models.wan.modules.vae import WanVAE

        device_mem_capacity = torch.cuda.get_device_properties(0).total_memory / 1048576 if torch.cuda.is_available() else 0
        return WanVAE.get_VAE_tile_size(vae_config, device_mem_capacity, False, output_height=output_height, output_width=output_width)

    def download(self, process_files: Callable[..., Any], send_cmd=None, status_text: str | None = None, spatial_upsampling=None) -> bool:
        from shared.utils.download import process_files_def_if_needed

        return process_files_def_if_needed(self.query_download_def(spatial_upsampling=spatial_upsampling), process_files=process_files, send_cmd=send_cmd, status_text=status_text)

    def load_upsampler(self, spatial_upsampling, *, process_files: Callable[..., Any], init_pipe: Callable[..., int], profile, progress_callback=None, **kwargs):
        scale = self.scale_for_upsampling(spatial_upsampling)
        if scale is None:
            raise ValueError(f"Unknown FlashVSR upsampling mode: {spatial_upsampling}")
        enabled, variant = self.settings()
        if not enabled:
            raise RuntimeError("FlashVSR spatial upsampling is disabled in Configuration > Extensions.")
        self.download(process_files, spatial_upsampling=spatial_upsampling)
        from postprocessing.flashvsr.attention_backend import set_sparse_backend
        set_sparse_backend(self.backend())
        from postprocessing.flashvsr.runtime import load_models

        model = self.model_for_upsampling(spatial_upsampling)
        load_models(self.paths(variant, model), variant=variant, init_pipe=init_pipe, profile=profile, model=model, progress_callback=progress_callback)

    def upscale(self, sample, spatial_upsampling, *, vae_config: int, seed=0, continue_cache=None, return_continue_cache=False, still_image=False, abort_callback=None, progress_callback=None, **kwargs):
        scale = self.scale_for_upsampling(spatial_upsampling)
        if scale is None:
            raise ValueError(f"Unknown FlashVSR upsampling mode: {spatial_upsampling}")
        from postprocessing.flashvsr.runtime import upscale_video

        output_height = int(sample.shape[-2] * scale)
        output_width = int(sample.shape[-1] * scale)
        flashvsr_tile_size = self.vae_tile_size(vae_config, output_height, output_width)
        return upscale_video(
            sample,
            scale,
            seed=seed,
            continue_cache=continue_cache,
            return_continue_cache=return_continue_cache,
            vae_tile_size=flashvsr_tile_size,
            topk_ratio=self.topk_ratio(),
            still_image=still_image,
            two_pass=self.is_two_pass_upsampling(spatial_upsampling),
            cache_windows=self.cache_windows(),
            abort_callback=abort_callback,
            progress_callback=progress_callback,
        )

    def release_vram(self) -> None:
        from postprocessing.flashvsr.runtime import release_models
        release_models()
