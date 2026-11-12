# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import os
from typing import Any, Callable

from postprocessing.spatial_upsamplers import SimpleScaleSuffixMixin, UPSAMPLER_PROFILE_VIDEO, UPSAMPLER_TYPE_POSTPROCESSING, format_multiplier_value


class SeedVR2Bridge(SimpleScaleSuffixMixin):
    UPSAMPLING_VALUE_PREFIX = "seedvr2"
    UPSAMPLING_7B_VALUE_PREFIX = "seedvr2_7b"
    UPSAMPLING_RATIOS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
    MODEL_3B = "3b"
    MODEL_7B = "7b"
    MODEL_CHOICES = [("SeedVR2 3B (BF16)", MODEL_3B), ("SeedVR2 7B (INT8 ConvRot: Larger, Closer To The Source On Real Footage)", MODEL_7B)]
    MODEL_DEFAULT = MODEL_3B
    METHOD_MODELS = {UPSAMPLING_VALUE_PREFIX: MODEL_3B, UPSAMPLING_7B_VALUE_PREFIX: MODEL_7B}
    TRANSFORMER_FILENAMES = {MODEL_3B: "SeedVR2_3B_bf16.safetensors", MODEL_7B: "SeedVR2_7B_int8_convrot.safetensors"}
    VAE_FILENAME = "SeedVR2_VAE_bf16.safetensors"
    POSITIVE_EMBEDDING_FILENAME = "SeedVR2_pos_emb_bf16.safetensors"
    AUTO_VAE_TILE_SIZE = 256
    MAX_VAE_TILE_SIZE = 512
    WINDOW_SIZE_AUTO = 0
    WINDOW_SIZE_UNLIMITED = -1
    WINDOW_SIZE_VALUES = (WINDOW_SIZE_AUTO, 60, 120, 180, 240, 300, *range(360, 1441, 120), WINDOW_SIZE_UNLIMITED)
    WINDOW_SIZE_CHOICES = [("Auto", WINDOW_SIZE_AUTO), *[(f"{value} frames", value) for value in WINDOW_SIZE_VALUES[1:-1]], ("Unlimited", WINDOW_SIZE_UNLIMITED)]
    MODEL_FRAME_STEP = 4
    WINDOW_OVERLAP_FRAMES = 3

    def __init__(self, server_config: dict[str, Any], files_locator):
        self.server_config = server_config
        self.files_locator = files_locator

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return {"model": cls.MODEL_DEFAULT, "window_size": cls.WINDOW_SIZE_AUTO}

    @classmethod
    def normalize_config_section(cls, config: dict[str, Any]) -> dict[str, Any]:
        normalized = cls.default_config()
        normalized.update(config or {})
        if normalized["model"] not in cls.TRANSFORMER_FILENAMES:
            normalized["model"] = cls.MODEL_DEFAULT
        try:
            window_size = int(normalized["window_size"])
        except (TypeError, ValueError):
            window_size = cls.WINDOW_SIZE_AUTO
        normalized["window_size"] = window_size if window_size in cls.WINDOW_SIZE_VALUES else cls.WINDOW_SIZE_AUTO
        return normalized

    @classmethod
    def resolve_window_size(cls, value, gpu_memory_gb: float | None = None) -> int:
        value = cls.normalize_config_section({"window_size": value})["window_size"]
        if value != cls.WINDOW_SIZE_AUTO:
            return value
        if gpu_memory_gb is None:
            import torch
            gpu_memory_gb = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1_000_000_000 if torch.cuda.is_available() else 0.0
        return 240 if gpu_memory_gb >= 24 else 120 if gpu_memory_gb >= 12 else 60

    @classmethod
    def resolve_model_window_size(cls, value, gpu_memory_gb: float | None = None) -> int:
        window_size = cls.resolve_window_size(value, gpu_memory_gb)
        return window_size if window_size == cls.WINDOW_SIZE_UNLIMITED else ((window_size - 1) // cls.MODEL_FRAME_STEP) * cls.MODEL_FRAME_STEP + 1

    def config(self) -> dict[str, Any]:
        from postprocessing import spatial_upsamplers as upsampler_api
        return upsampler_api.read_config_section(self.server_config, self)

    def window_frames(self) -> int:
        return self.resolve_model_window_size(self.config()["window_size"])

    def model(self) -> str:
        return self.config()["model"]

    def method(self) -> str:
        """The method shown in the upsampling menus: the model chosen in Configuration > Extensions."""
        return next(method for method, model in self.METHOD_MODELS.items() if model == self.model())

    def method_visible(self, method: str) -> bool:
        return self.METHOD_MODELS[method] == self.model()

    def preferred_method(self, method: str) -> str:
        return self.method()

    def model_for_upsampling(self, spatial_upsampling) -> str:
        split = self.split_value(spatial_upsampling)
        if split is None:
            raise ValueError(f"Unknown SeedVR2 upsampling mode: {spatial_upsampling}")
        return self.METHOD_MODELS[split[0]]

    @staticmethod
    def format_ratio(scale: float) -> str:
        scale = float(scale)
        return str(int(scale)) if scale.is_integer() else f"{scale:g}"

    @classmethod
    def upsampling_value(cls, scale: float, method: str = UPSAMPLING_VALUE_PREFIX) -> str:
        return format_multiplier_value(method, scale)

    @classmethod
    def query_upsampler_def(cls) -> dict[str, Any]:
        return {
            "name": "SeedVR2",
            "upsampler_types": (UPSAMPLER_TYPE_POSTPROCESSING,),
            "media": ("video", "image"),
            "profile": UPSAMPLER_PROFILE_VIDEO,
            "config_key": "seedvr2",
            "pos": 30,
            "method_pos": {cls.UPSAMPLING_VALUE_PREFIX: 30, cls.UPSAMPLING_7B_VALUE_PREFIX: 31},
            "methods": [("SeedVR2 3B", cls.UPSAMPLING_VALUE_PREFIX), ("SeedVR2 7B", cls.UPSAMPLING_7B_VALUE_PREFIX)],
            "vae_methods": [],
            "multipliers": {method: cls.UPSAMPLING_RATIOS for method in cls.METHOD_MODELS},
            "default_spatial_upsampling": cls.upsampling_value(2.0),
            "postprocessing_category": "upsampler",
            "description": "Restore detail and reduce compression or generation degradation while spatially upscaling with SeedVR2. Configuration > Extensions chooses which model is normally offered: SeedVR2 3B or SeedVR2 7B. Extract Settings restores the model saved with the media even when it is hidden by the current configuration.",
            "media_descriptions": {"video": "Overlapping model windows preserve temporal continuity; the window size is configurable in Extensions."},
            "method_descriptions": {
                cls.UPSAMPLING_VALUE_PREFIX: "Restore detail with SeedVR2 3B: preferred visually on AI-generated and real-world video in the authors' study, with lighter weights.",
                cls.UPSAMPLING_7B_VALUE_PREFIX: "Restore detail with SeedVR2 7B: stays closer to the source on degraded real footage; about 2.4x the parameters of 3B, downloaded in INT8, and slightly slower.",
            },
        }

    def enabled(self) -> bool:
        return True

    def create_config_ui(self, gr, config: dict[str, Any], *, lock_config: bool = False):
        with gr.Row():
            model = gr.Dropdown(choices=self.MODEL_CHOICES, value=config["model"], label="SeedVR2 Model", info="Chooses the model normally shown in upsampling menus. 3B looks better on AI-generated video; 7B stays closer to the source on real footage. Extract Settings can restore the other model without changing this preference.", interactive=not lock_config)
            window_size = gr.Dropdown(choices=self.WINDOW_SIZE_CHOICES, value=config["window_size"], label="SeedVR2 Window Size", info="Auto uses 240 frames on GPUs with at least 24 GB, 120 with at least 12 GB, otherwise 60. Finite values are adjusted down to the nearest 4n+1 model window.", interactive=not lock_config)
        return [("model", model), ("window_size", window_size)]

    def config_requires_release(self, old_config: dict[str, Any], new_config: dict[str, Any], changed_keys: set[str]) -> bool:
        return old_config["model"] != new_config["model"] or bool({"video_profile", "vae_config"} & changed_keys)

    def query_chunking(self, spatial_upsampling):
        """How a long video goes through this upsampler a chunk at a time (postprocessing.chunking)."""
        from .process_handler import HANDLER
        return HANDLER

    def validate_upsampling(self, spatial_upsampling, image_mode: int) -> str:
        split = self.split_value(spatial_upsampling)
        if split is None or split[0] not in self.METHOD_MODELS or split[1] not in self.UPSAMPLING_RATIOS:
            return "SeedVR2 only supports x1.0 through x4.0 in 0.5 increments"
        return ""

    def _filenames(self, model: str) -> tuple[str, str, str]:
        return self.TRANSFORMER_FILENAMES[model], self.VAE_FILENAME, self.POSITIVE_EMBEDDING_FILENAME

    def query_download_def(self, enabled_only: bool = True, spatial_upsampling=None) -> dict[str, Any]:
        model = self.model() if spatial_upsampling is None else self.model_for_upsampling(spatial_upsampling)
        return self._model_download_def(model)

    def query_checkpoint_inventory(self):
        return [{"name": "SeedVR2", "variant": model.upper(), "download_defs": [self._model_download_def(model)]} for model in self.TRANSFORMER_FILENAMES]

    def _model_download_def(self, model):
        return {
            "repoId": "DeepBeepMeep/MiniMax-H3",
            "sourceFolderList": ["SeedVR2"],
            "fileList": [list(self._filenames(model))],
        }

    def _locate(self, filename: str) -> str:
        return self.files_locator.locate_file(os.path.join("SeedVR2", filename))

    def paths(self, model: str):
        from .runtime import SeedVR2Paths
        transformer, vae, positive_embedding = self._filenames(model)
        return SeedVR2Paths(transformer=self._locate(transformer), vae=self._locate(vae), positive_embedding=self._locate(positive_embedding))

    def download(self, process_files: Callable[..., Any], send_cmd=None, status_text: str | None = None, spatial_upsampling=None) -> bool:
        model = self.model() if spatial_upsampling is None else self.model_for_upsampling(spatial_upsampling)
        if all(self.files_locator.locate_file(os.path.join("SeedVR2", filename), error_if_none=False) is not None for filename in self._filenames(model)):
            return False
        from shared.utils.download import send_download_status
        send_download_status(send_cmd, status_text)
        process_files(**self.query_download_def(spatial_upsampling=spatial_upsampling))
        return True

    def vae_tile_size(self, vae_config: int, output_height: int, output_width: int) -> int:
        import torch
        from models.wan.modules.vae import WanVAE
        device_mem_capacity = torch.cuda.get_device_properties(0).total_memory / 1048576 if torch.cuda.is_available() else 0
        return WanVAE.get_VAE_tile_size(vae_config, device_mem_capacity, False, output_height=output_height, output_width=output_width)

    def load_upsampler(self, spatial_upsampling, *, process_files: Callable[..., Any], init_pipe: Callable[..., int], profile, progress_callback=None, **kwargs):
        if self.validate_upsampling(spatial_upsampling, 0):
            raise ValueError(f"Unknown SeedVR2 upsampling mode: {spatial_upsampling}")
        self.download(process_files, spatial_upsampling=spatial_upsampling)
        from .runtime import load_models
        model = self.model_for_upsampling(spatial_upsampling)
        load_models(self.paths(model), model=model, init_pipe=init_pipe, profile=profile, progress_callback=progress_callback)

    def upscale(self, sample, spatial_upsampling, *, vae_config: int, vae_tile_size=None, seed=0, still_image=False, abort_callback=None, progress_callback=None, **kwargs):
        split = self.split_value(spatial_upsampling)
        if split is None:
            raise ValueError(f"Unknown SeedVR2 upsampling mode: {spatial_upsampling}")
        scale = split[1]
        output_height, output_width = int(sample.shape[-2] * scale), int(sample.shape[-1] * scale)
        vae_tile_size = int(self.vae_tile_size(vae_config, output_height, output_width) or self.AUTO_VAE_TILE_SIZE)
        vae_tile_size = min(vae_tile_size, self.AUTO_VAE_TILE_SIZE if int(vae_config) == 0 else self.MAX_VAE_TILE_SIZE)
        from .runtime import upscale_video
        if still_image:
            sample = sample.expand(-1, 5, -1, -1)
        output, cache = upscale_video(sample, scale, seed=seed, vae_tile_size=vae_tile_size, window_size=self.window_frames(), window_overlap=self.WINDOW_OVERLAP_FRAMES, abort_callback=abort_callback, progress_callback=progress_callback)
        if still_image and output is not None:
            output = output[:, 2:3]
        return output, cache

    def release_vram(self) -> None:
        from .runtime import release_models
        release_models()
