# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import os
import sys
from typing import Any, Callable

from postprocessing.spatial_upsamplers import PARAMETER_UI_LATE_POSTPROCESSING, PARAMETER_UI_POSTPROCESSING, SimpleScaleSuffixMixin, UPSAMPLER_PROFILE_VIDEO, UPSAMPLER_TYPE_POSTPROCESSING

from .runtime import DEFAULT_REFINE_PROMPT, DEFAULT_WINDOW_FRAMES, MAX_WINDOW_FRAMES, MODEL_TYPES, REFINE_DEFAULT_WINDOW_FRAMES, REFINE_MAX_TILES_PER_CALL, REFINE_MAX_WINDOW_FRAMES, REFINE_METHOD, REFINE_MIN_WINDOW_FRAMES, REFINE_MULTIPLIERS, RUNTIME_NAME, TEMPORAL_STRIDE, WINDOW_OVERLAP_FRAMES, lora_urls, supported_spatial_size

DEFAULT_PROMPT = "high quality, detailed, sharp, natural textures"


class LTXVideoUpsamplerBridge(SimpleScaleSuffixMixin):
    METHODS = (("LTX 2.3 Pixel Spatial Upscaler", "ltx23"), ("LTX 2.5 Pixel Spatial Upscaler", "ltx25"), ("LTX 2.5 Detail Refiner", REFINE_METHOD))
    MULTIPLIERS = {"ltx23": (2.0,), "ltx25": (2.0,), REFINE_METHOD: REFINE_MULTIPLIERS}
    MIN_WINDOW_FRAMES = TEMPORAL_STRIDE + 1
    WINDOW_SIZE_VALUES = tuple(range(MIN_WINDOW_FRAMES, MAX_WINDOW_FRAMES + 1, TEMPORAL_STRIDE))
    WINDOW_OVERLAP_VALUES = tuple(range(1, MAX_WINDOW_FRAMES, TEMPORAL_STRIDE))
    REFINER_WINDOW_VALUES = tuple(range(REFINE_MIN_WINDOW_FRAMES, REFINE_MAX_WINDOW_FRAMES + 1, TEMPORAL_STRIDE))

    def __init__(self, server_config: dict[str, Any], files_locator):
        self.server_config = server_config
        self.files_locator = files_locator

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return {"window_size": DEFAULT_WINDOW_FRAMES, "window_overlap": WINDOW_OVERLAP_FRAMES, "refiner_window_size": REFINE_DEFAULT_WINDOW_FRAMES, "refiner_gentle": True, "refiner_tiles_per_call": 0}

    @classmethod
    def normalize_config_section(cls, config: dict[str, Any]) -> dict[str, Any]:
        normalized = cls.default_config()
        normalized.update(config or {})
        normalized.pop("version", None)
        try:
            window_size = int(normalized["window_size"])
        except (TypeError, ValueError):
            window_size = DEFAULT_WINDOW_FRAMES
        try:
            window_overlap = int(normalized["window_overlap"])
        except (TypeError, ValueError):
            window_overlap = WINDOW_OVERLAP_FRAMES
        try:
            refiner_window_size = int(normalized["refiner_window_size"])
        except (TypeError, ValueError):
            refiner_window_size = REFINE_DEFAULT_WINDOW_FRAMES
        try:
            refiner_tiles_per_call = int(normalized["refiner_tiles_per_call"])
        except (TypeError, ValueError):
            refiner_tiles_per_call = 0
        normalized["window_size"] = window_size if window_size in cls.WINDOW_SIZE_VALUES else DEFAULT_WINDOW_FRAMES
        normalized["window_overlap"] = window_overlap if window_overlap in cls.WINDOW_OVERLAP_VALUES else WINDOW_OVERLAP_FRAMES
        if normalized["window_overlap"] >= normalized["window_size"]:
            normalized["window_overlap"] = normalized["window_size"] - TEMPORAL_STRIDE
        normalized["refiner_window_size"] = refiner_window_size if refiner_window_size in cls.REFINER_WINDOW_VALUES else REFINE_DEFAULT_WINDOW_FRAMES
        normalized["refiner_gentle"] = bool(normalized["refiner_gentle"])
        normalized["refiner_tiles_per_call"] = refiner_tiles_per_call if 0 <= refiner_tiles_per_call <= REFINE_MAX_TILES_PER_CALL else 0
        return normalized

    def config(self) -> dict[str, Any]:
        from postprocessing import spatial_upsamplers as upsampler_api
        return upsampler_api.read_config_section(self.server_config, self)

    def create_config_ui(self, gr, config: dict[str, Any], *, lock_config: bool = False):
        with gr.Group():
            with gr.Row():
                window_size = gr.Slider(self.MIN_WINDOW_FRAMES, MAX_WINDOW_FRAMES, value=config["window_size"], step=TEMPORAL_STRIDE, label="LTX Upsampler Sliding Window Size", info="Frames processed per LTX refinement window. Larger windows use more VRAM.", interactive=not lock_config)
                window_overlap = gr.Slider(1, config["window_size"] - TEMPORAL_STRIDE, value=config["window_overlap"], step=TEMPORAL_STRIDE, label="LTX Upsampler Sliding Window Overlap", info="Frames shared and crossfaded between consecutive windows.", interactive=not lock_config)
            with gr.Row():
                refiner_window_size = gr.Slider(REFINE_MIN_WINDOW_FRAMES, REFINE_MAX_WINDOW_FRAMES, value=config["refiner_window_size"], step=TEMPORAL_STRIDE, label="LTX 2.5 Detail Refiner Temporal Window", info="Frames refined together in each temporal window. 97 matches the Refine Details training clips; shorter windows use less VRAM.", interactive=not lock_config)
                refiner_gentle = gr.Checkbox(value=config["refiner_gentle"], label="LTX 2.5 Detail Refiner Gentle Mode", info="Starts from the source video and runs 3 steps: faster and closer to the source. Disable to rebuild detail from noise in 8 steps.", interactive=not lock_config)
            with gr.Row():
                refiner_tiles_per_call = gr.Slider(0, REFINE_MAX_TILES_PER_CALL, value=config["refiner_tiles_per_call"], step=1, label="LTX 2.5 Detail Refiner Tiles per Model Call", info="0 = Auto (uses up to a quarter of free VRAM, at most 2 GB). Lower values cap VRAM; the result is identical, only speed changes.", interactive=not lock_config)

        def update_overlap_limit(new_window_size, current_overlap):
            maximum = int(new_window_size) - TEMPORAL_STRIDE
            return gr.update(maximum=maximum, value=min(int(current_overlap), maximum))

        window_size.change(update_overlap_limit, inputs=[window_size, window_overlap], outputs=window_overlap)
        return [("window_size", window_size), ("window_overlap", window_overlap), ("refiner_window_size", refiner_window_size), ("refiner_gentle", refiner_gentle), ("refiner_tiles_per_call", refiner_tiles_per_call)]

    @classmethod
    def query_upsampler_def(cls) -> dict[str, Any]:
        return {
            "name": RUNTIME_NAME,
            "upsampler_types": (UPSAMPLER_TYPE_POSTPROCESSING,),
            "media": ("video",),
            "profile": UPSAMPLER_PROFILE_VIDEO,
            "config_key": "ltx2",
            "pos": 45,
            "method_pos": {"ltx23": 45, "ltx25": 46, REFINE_METHOD: 47},
            "methods": list(cls.METHODS),
            "vae_methods": [],
            "multipliers": cls.MULTIPLIERS,
            "default_spatial_upsampling": "ltx23*2",
            "default_prompt": DEFAULT_PROMPT,
            "source_audio_conditioning": True,
            "postprocessing_category": "upsampler",
            "description": "Spatially upscale video x2 with the selected LTX 2.3 or 2.5 model, preserving motion and conditioning on source audio. Long videos use deterministic overlapping windows; window size and overlap are configurable in Extensions.",
            "method_descriptions": {
                REFINE_METHOD: "Rebuild fine texture, edges and grain in soft, compressed, upscaled or generated video with the LTX 2.5 Refine Details IC-LoRA, keeping framing, exposure and color. The source is resized with Lanczos, then refined in overlapping 1024x576 tiles fused at every step, so x1 (detail only), x1.5 and x2 outputs up to 8K can be processed. Best for 1080p to 4K. Slow: 4K needs 49 tiles per step. Temporal window and gentle mode are configurable in Extensions.",
            },
            "method_parameters": {REFINE_METHOD: [
                {"name": "refine_prompt", "setting": "refine_prompt", "type": "string", "component": "textbox", "ui": (PARAMETER_UI_POSTPROCESSING, PARAMETER_UI_LATE_POSTPROCESSING), "required": False, "default": DEFAULT_REFINE_PROMPT, "label": "Refiner Style Prompt", "description": "Describe rendering only: sharpness, texture, grain, light. Every tile receives the whole prompt, so naming subjects can paint them into unrelated parts of the frame.", "lines": 2},
            ]},
        }

    def enabled(self) -> bool:
        return True

    def validate_upsampling(self, spatial_upsampling, image_mode: int) -> str:
        split = self.split_value(spatial_upsampling)
        if image_mode:
            return "LTX video upsampling is available for videos only"
        if split is not None and split[0] == REFINE_METHOD:
            return "" if split[1] in REFINE_MULTIPLIERS else "LTX 2.5 Detail Refiner supports x1, x1.5 and x2"
        if split is None or split[0] not in MODEL_TYPES or split[1] != 2.0:
            return "LTX video upsampling only supports x2"
        return ""

    def download(self, process_files: Callable[..., Any], send_cmd=None, status_text: str | None = None, spatial_upsampling=None) -> bool:
        split = self.split_value(spatial_upsampling)
        if split is None:
            return False
        wgp = sys.modules.get("wgp")
        if wgp is None:
            return False
        from shared.utils.download import send_download_status

        send_download_status(send_cmd, status_text)
        model_type = MODEL_TYPES[split[0]]
        model_def = wgp.get_model_def(model_type)
        transformer = wgp.get_model_filename(model_type, wgp.transformer_quantization, wgp.transformer_dtype_policy, model_def=model_def)
        wgp.download_models(transformer, model_type, 0, -1, model_def=model_def)
        text_encoder_urls = wgp.get_model_recursive_prop(model_type, "text_encoder_URLs", return_list=True, model_def=model_def)
        text_encoder = wgp.get_model_filename(model_type, wgp.text_encoder_quantization, wgp.transformer_dtype_policy, URLs=text_encoder_urls)
        wgp.download_models(text_encoder, model_type, 2, -1, force_path=model_def["text_encoder_folder"], model_def=model_def)
        lora_dir = wgp.get_lora_dir(model_type)
        os.makedirs(lora_dir, exist_ok=True)
        for url in lora_urls(wgp, split[0]):
            path = wgp.get_lora_local_path(lora_dir, url)
            if not os.path.isfile(path):
                wgp.download_file(url, path)
        return True

    def load_upsampler(self, spatial_upsampling, *, process_files: Callable[..., Any], **kwargs):
        error = self.validate_upsampling(spatial_upsampling, 0)
        if error:
            raise ValueError(error)
        self.download(process_files, spatial_upsampling=spatial_upsampling)
        from .runtime import load_model

        load_model(self.split_value(spatial_upsampling)[0])

    def upscale(self, sample, spatial_upsampling, *, vae_config: int, vae_tile_size=None, seed=0, fps=24.0, frame_offset=0, prompt="", negative_prompt="", refine_prompt="", continue_cache=None, return_continue_cache=False, audio_waveform=None, audio_sample_rate=0, source_audio_path=None, still_image=False, abort_callback=None, progress_callback=None, **kwargs):
        if still_image:
            raise ValueError("LTX video upsampling is available for videos only")
        split = self.split_value(spatial_upsampling)
        if split is None:
            raise ValueError(f"Unknown LTX video upsampling mode: {spatial_upsampling}")
        if split[0] == REFINE_METHOD:
            from .runtime import refine_video

            config = self.config()
            return refine_video(sample, scale=split[1], prompt=str(refine_prompt or "").strip() or DEFAULT_REFINE_PROMPT, seed=seed, fps=fps, window_frames=config["refiner_window_size"], window_overlap=config["window_overlap"], gentle=config["refiner_gentle"], vae_config=vae_config, tiles_per_call=config["refiner_tiles_per_call"], frame_offset=frame_offset, continue_cache=continue_cache, return_continue_cache=return_continue_cache, abort_callback=abort_callback, progress_callback=progress_callback)
        from .runtime import RUNTIME, upscale_video

        if vae_tile_size is None:
            vae_tile_size = RUNTIME.vae_tile_size(vae_config, supported_spatial_size(sample.shape[-2]) * 2, supported_spatial_size(sample.shape[-1]) * 2)
        config = self.config()
        return upscale_video(sample, prompt=prompt, negative_prompt=negative_prompt, audio_waveform=audio_waveform, audio_sample_rate=audio_sample_rate, source_audio_path=source_audio_path, seed=seed, fps=fps, window_size=config["window_size"], window_overlap=config["window_overlap"], frame_offset=frame_offset, vae_tile_size=vae_tile_size, abort_callback=abort_callback, progress_callback=progress_callback)

    def release_vram(self) -> None:
        from .runtime import release_model

        release_model()
