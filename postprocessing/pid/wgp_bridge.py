# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

from typing import Any, Callable

import torch

from postprocessing.spatial_upsamplers import format_multiplier_value
from postprocessing.pid.runtime import (
    PID_TEXT_ENCODER_FILES,
    PID_TEXT_ENCODER_FOLDER,
    PID_TILING_THRESHOLD_CHOICES,
    PID_TILING_THRESHOLD_DEFAULT,
    PID_VERSION_DEFAULT,
    PID_QWEN_VAE_UPSAMPLING_METHOD,
    PID_QWEN_POST_UPSAMPLING_METHOD,
    PID_FLUX_VAE_UPSAMPLING_METHOD,
    PID_FLUX2_VAE_UPSAMPLING_METHOD,
    PID_FLUX_VAE_UPSAMPLING_METHOD_V15,
    PID_FLUX2_VAE_UPSAMPLING_METHOD_V15,
    PID_FLUX_POST_UPSAMPLING_METHOD,
    PID_FLUX2_POST_UPSAMPLING_METHOD,
    PID_FLUX_POST_UPSAMPLING_METHOD_V15,
    PID_FLUX2_POST_UPSAMPLING_METHOD_V15,
    PID_VAE_UPSAMPLING_METHODS,
    get_pid_download_def,
    get_pid_upsampler,
    is_pid_upsampling,
    pid_checkpoint_types_for_tiling_threshold,
    pid_backbone_for_upsampling,
    pid_checkpoint_filename,
    pid_post_upsampling_choices,
    pid_spatial_block_size,
    pid_version_for_upsampling,
    pid_vae_filename,
    normalize_pid_tiling_threshold,
    normalize_pid_version_filter,
    split_pid_upsampling,
)


class PiDBridge:
    UPSAMPLING_RATIOS = (4.0,)
    UPSAMPLING_METHODS = (PID_FLUX_POST_UPSAMPLING_METHOD, PID_FLUX2_POST_UPSAMPLING_METHOD, PID_FLUX_POST_UPSAMPLING_METHOD_V15, PID_FLUX2_POST_UPSAMPLING_METHOD_V15, PID_QWEN_POST_UPSAMPLING_METHOD, PID_FLUX_VAE_UPSAMPLING_METHOD, PID_FLUX2_VAE_UPSAMPLING_METHOD, PID_FLUX_VAE_UPSAMPLING_METHOD_V15, PID_FLUX2_VAE_UPSAMPLING_METHOD_V15, PID_QWEN_VAE_UPSAMPLING_METHOD)
    batch_image_inputs = True
    uses_image_profile = True

    def __init__(self, server_config: dict[str, Any], files_locator):
        self.server_config = server_config
        self.files_locator = files_locator
        self._session = None

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return {"version": PID_VERSION_DEFAULT, "tiling_threshold": PID_TILING_THRESHOLD_DEFAULT}

    @classmethod
    def legacy_config_keys(cls) -> tuple[str, ...]:
        return ("pid_tiling_threshold", "pid_persistence")

    @classmethod
    def legacy_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        return {"tiling_threshold": config.get("pid_tiling_threshold", PID_TILING_THRESHOLD_DEFAULT)}

    @classmethod
    def normalize_config_section(cls, config: dict[str, Any]) -> dict[str, Any]:
        normalized = cls.default_config()
        normalized.update(config or {})
        normalized["version"] = normalize_pid_version_filter(normalized["version"])
        normalized["tiling_threshold"] = normalize_pid_tiling_threshold(normalized.get("tiling_threshold", PID_TILING_THRESHOLD_DEFAULT))
        return normalized

    def config(self) -> dict[str, Any]:
        from postprocessing import spatial_upsamplers as upsampler_api

        return upsampler_api.read_config_section(self.server_config, self)

    def persistent_models(self) -> bool:
        from postprocessing import spatial_upsamplers as upsampler_api

        return upsampler_api.persistent_models(self.server_config)

    def format_method_label(self, label: str, method: str) -> str:
        version = pid_version_for_upsampling(method)
        return str(label).replace(" PiD ", f" PiD v{'1.0' if version == '1' else version} ")

    def method_available(self, method: str) -> bool:
        version_filter = self.config()["version"]
        return version_filter == "both" or pid_version_for_upsampling(method) == version_filter

    @classmethod
    def query_upsampler_def(cls) -> dict[str, Any]:
        post_methods = pid_post_upsampling_choices()
        vae_methods = [("Flux VAE PiD Upsampler", PID_FLUX_VAE_UPSAMPLING_METHOD), ("Flux2 VAE PiD Upsampler", PID_FLUX2_VAE_UPSAMPLING_METHOD), ("Flux VAE PiD Upsampler", PID_FLUX_VAE_UPSAMPLING_METHOD_V15), ("Flux2 VAE PiD Upsampler", PID_FLUX2_VAE_UPSAMPLING_METHOD_V15), ("Qwen VAE PiD Upsampler", PID_QWEN_VAE_UPSAMPLING_METHOD)]
        return {
            "name": "PiD",
            "upsampler_types": ("postprocessing", "vae"),
            "media": ("image",),
            "profile": "image",
            "config_key": "pid",
            "pos": 40,
            "method_pos": {
                PID_FLUX_POST_UPSAMPLING_METHOD: 40,
                PID_FLUX2_POST_UPSAMPLING_METHOD: 41,
                PID_FLUX_POST_UPSAMPLING_METHOD_V15: 40,
                PID_FLUX2_POST_UPSAMPLING_METHOD_V15: 41,
                PID_QWEN_POST_UPSAMPLING_METHOD: 42,
                PID_FLUX_VAE_UPSAMPLING_METHOD: 40,
                PID_FLUX2_VAE_UPSAMPLING_METHOD: 41,
                PID_FLUX_VAE_UPSAMPLING_METHOD_V15: 40,
                PID_FLUX2_VAE_UPSAMPLING_METHOD_V15: 41,
                PID_QWEN_VAE_UPSAMPLING_METHOD: 42,
            },
            "methods": post_methods,
            "vae_methods": vae_methods,
            "multipliers": {method: cls.UPSAMPLING_RATIOS for method in cls.UPSAMPLING_METHODS},
            "default_spatial_upsampling": "flux_pid*4",
            "postprocessing_category": "upsampler",
            "description": "Uses a dedicated x4 diffusion upsampler for strong detail recovery.",
            "method_descriptions": {
                **{method: "VAE-encode the input before diffusion upscaling; tiling can reduce memory peaks on large targets." for _label, method in post_methods},
                **{method: "Reuse the compatible generation model's existing latent, avoiding a separate input encode." for _label, method in vae_methods},
            },
        }

    def is_upsampling(self, spatial_upsampling) -> bool:
        return is_pid_upsampling(spatial_upsampling)

    @classmethod
    def split_value(cls, value) -> tuple[str, float] | None:
        return split_pid_upsampling(value)

    @classmethod
    def build_value(cls, method, scale) -> str | None:
        method = str(method or "").strip().lower()
        scale = float(scale or 4.0)
        return format_multiplier_value(method, scale) if method in cls.UPSAMPLING_METHODS and scale == 4.0 else None

    def validate_upsampling(self, spatial_upsampling, image_mode: int) -> str:
        if not self.is_upsampling(spatial_upsampling):
            return ""
        if image_mode == 0:
            return "PiD Spatial Upsampling is only available for Images"
        return ""

    @classmethod
    def _model_vae_methods(cls, model_def: dict[str, Any]) -> dict[str, tuple[int, ...]]:
        methods = model_def.get("vae_upsamplers", {})
        out = {}
        for method, modes in methods.items() if isinstance(methods, dict) else ():
            if method not in PID_VAE_UPSAMPLING_METHODS:
                continue
            if isinstance(modes, dict):
                modes = modes.get("image_modes", ())
            elif isinstance(modes, int):
                modes = (modes,)
            out[method] = tuple(int(mode) for mode in modes)
        if PID_FLUX_VAE_UPSAMPLING_METHOD in out:
            out[PID_FLUX_VAE_UPSAMPLING_METHOD_V15] = out[PID_FLUX_VAE_UPSAMPLING_METHOD]
        if PID_FLUX2_VAE_UPSAMPLING_METHOD in out:
            out[PID_FLUX2_VAE_UPSAMPLING_METHOD_V15] = out[PID_FLUX2_VAE_UPSAMPLING_METHOD]
        return out

    def supports_model_vae_method(self, method, model_type, model_def, image_mode: int) -> bool:
        if not self.method_available(method):
            return False
        return int(image_mode) in self._model_vae_methods(model_def).get(str(method or "").strip().lower(), ())

    def validate_model_vae_upsampling(self, spatial_upsampling, image_mode: int, model_type, model_def, medium: str) -> str:
        split = self.split_value(spatial_upsampling)
        method = "" if split is None else split[0]
        if method not in PID_VAE_UPSAMPLING_METHODS:
            return f"Unknown PiD upsampling mode: {spatial_upsampling}"
        model_methods = self._model_vae_methods(model_def)
        if method not in model_methods:
            return "This model does not support the selected VAE PiD Upsampler"
        return "" if int(image_mode) in model_methods[method] else f"VAE PiD Upsampler is not available for {medium}"

    def create_config_ui(self, gr, config: dict[str, Any], *, lock_config: bool = False):
        with gr.Group():
            with gr.Row():
                version = gr.Dropdown(choices=[("PiD v1.5 only", "1.5"), ("PiD v1.0 only", "1"), ("Both PiD v1.5 and v1.0", "both")], value=config["version"], label="Visible PiD Versions", interactive=not lock_config)
                tiling_threshold = gr.Dropdown(choices=PID_TILING_THRESHOLD_CHOICES, value=config["tiling_threshold"], label="PiD Tiling Threshold", interactive=not lock_config)
        return [("version", version), ("tiling_threshold", tiling_threshold)]

    def validate_config_section(self, config: dict[str, Any]):
        return ""

    def config_requires_release(self, old_config: dict[str, Any], new_config: dict[str, Any], changed_keys: set[str]) -> bool:
        return old_config != new_config or bool({"image_profile", "attention_mode"} & changed_keys)

    def query_checkpoint_inventory(self):
        for label, method in pid_post_upsampling_choices():
            version = pid_version_for_upsampling(method)
            defs = get_pid_download_def(pid_backbone_for_upsampling(method), ckpt_type=("2kto4k",) if version == "1.5" else None, version=version)
            yield {"name": label, "variant": f"v{'1.0' if version == '1' else version}", "download_defs": defs}

    def _required_files(self, backbone, spatial_upsampling):
        version = pid_version_for_upsampling(spatial_upsampling)
        ckpt_types = ("2kto4k",) if version == "1.5" else pid_checkpoint_types_for_tiling_threshold(self.config()["tiling_threshold"])
        required = [pid_checkpoint_filename(backbone, ckpt_type, version) for ckpt_type in ckpt_types]
        required.append(pid_vae_filename(backbone))
        required.extend([f"{PID_TEXT_ENCODER_FOLDER}/{filename}" for filename in PID_TEXT_ENCODER_FILES])
        return required

    def download(self, process_files: Callable[..., Any], send_cmd=None, status_text: str | None = None, spatial_upsampling=None) -> bool:
        backbone = pid_backbone_for_upsampling(spatial_upsampling)
        if all(self.files_locator.locate_file(path, error_if_none=False) is not None for path in self._required_files(backbone, spatial_upsampling)):
            return False
        from shared.utils.download import send_download_status

        send_download_status(send_cmd, status_text)
        version = pid_version_for_upsampling(spatial_upsampling)
        ckpt_types = ("2kto4k",) if version == "1.5" else pid_checkpoint_types_for_tiling_threshold(self.config()["tiling_threshold"])
        for download_def in get_pid_download_def(backbone, ckpt_type=ckpt_types, include_vae=True, version=version):
            process_files(**download_def)
        return True

    def _prepare_sample(self, sample, device, dtype, backbone):
        frames = sample.transpose(0, 1).contiguous()
        block_size = pid_spatial_block_size(backbone)
        target_height = max(block_size, round(int(frames.shape[-2]) / block_size) * block_size)
        target_width = max(block_size, round(int(frames.shape[-1]) / block_size) * block_size)
        if frames.shape[-2:] != (target_height, target_width):
            frames = torch.nn.functional.interpolate(frames, size=(target_height, target_width), mode="bilinear", align_corners=False, antialias=True)
        if frames.dtype == torch.uint8:
            frames = frames.to(device=device, dtype=dtype).div_(127.5).sub_(1.0)
        else:
            frames = frames.to(device=device, dtype=dtype)
        return frames

    def load_upsampler(
        self,
        spatial_upsampling,
        *,
        process_files: Callable[..., Any],
        init_pipe: Callable[..., int],
        profile,
        **kwargs,
    ):
        if not self.is_upsampling(spatial_upsampling):
            raise ValueError(f"Unknown PiD upsampling mode: {spatial_upsampling}")
        self.download(process_files, spatial_upsampling=spatial_upsampling)
        backbone = pid_backbone_for_upsampling(spatial_upsampling)
        config = self.config()
        self._session = get_pid_upsampler(
            backbone,
            None,
            init_pipe=init_pipe,
            profile=profile,
            tiling_threshold=config["tiling_threshold"],
            attention_mode=self.server_config.get("attention_mode"),
            version=pid_version_for_upsampling(spatial_upsampling),
        )
        self._session.ensure_loaded()

    def prepare_vae_upsampler(
        self,
        spatial_upsampling,
        *,
        send_cmd,
        process_files: Callable[..., Any],
        init_pipe: Callable[..., int],
        profile,
        attention_mode=None,
        **kwargs,
    ):
        if not self.is_upsampling(spatial_upsampling):
            raise ValueError(f"Unknown PiD upsampling mode: {spatial_upsampling}")
        self.download(process_files, send_cmd=send_cmd, status_text="Downloading PiD upsampler model files...", spatial_upsampling=spatial_upsampling)
        config = self.config()
        self._session = get_pid_upsampler(
            pid_backbone_for_upsampling(spatial_upsampling),
            None,
            init_pipe=init_pipe,
            profile=profile,
            persistent_models=self.persistent_models(),
            tiling_threshold=config["tiling_threshold"],
            attention_mode=attention_mode,
            version=pid_version_for_upsampling(spatial_upsampling),
        )
        self._session.progress_label = "PiD Spatial Upsampling"
        return self._session

    def upscale(
        self,
        sample,
        spatial_upsampling,
        *,
        seed=0,
        abort_callback=None,
        progress_callback=None,
        **kwargs,
    ):
        if not self.is_upsampling(spatial_upsampling):
            raise ValueError(f"Unknown PiD upsampling mode: {spatial_upsampling}")
        session = self._session
        if session is None:
            raise RuntimeError("PiD upsampler is not loaded.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        images = self._prepare_sample(sample, device, session.dtype, session.backbone)
        output = session.decode(
            images,
            None,
            prompt="",
            seed=seed,
            vae_encode=True,
            abort_callback=abort_callback,
            progress_callback=progress_callback,
            cleanup=False,
        )
        images = None
        if output is None:
            return None, None
        output = output.to("cpu").transpose(0, 1).contiguous()
        return output, None

    def release_vram(self) -> None:
        from postprocessing.pid.runtime import release_models

        self._session = None
        release_models()
