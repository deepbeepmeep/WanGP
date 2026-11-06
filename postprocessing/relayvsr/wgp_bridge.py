# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import os
from typing import Any, Callable

from postprocessing.spatial_upsamplers import SimpleScaleSuffixMixin, UPSAMPLER_PROFILE_VIDEO, UPSAMPLER_TYPE_POSTPROCESSING, format_multiplier_value

METHOD = "relayvsr"
WEIGHTS_REPO = "DeepBeepMeep/Wan2.1"
FOLDER = "RelayVSR"
FILE_PREFIXES = {4: "RelayVSR", 2: "RelayVSR_X2"}  # the native x4 checkpoints and the native x2 ones (keyframe models in INT8 ConvRot only)


class RelayVSRBridge(SimpleScaleSuffixMixin):
    UPSAMPLING_RATIOS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
    X2_MODEL_CHOICES = [("RelayVSR X2: Native x2, More Than Twice As Fast", "x2"), ("RelayVSR x4, Resized Down", "x4")]

    def __init__(self, server_config: dict[str, Any], files_locator):
        self.server_config = server_config
        self.files_locator = files_locator

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return {"x2_model": "x2"}

    @classmethod
    def normalize_config_section(cls, config: dict[str, Any]) -> dict[str, Any]:
        normalized = cls.default_config()
        normalized.update({key: value for key, value in (config or {}).items() if key in normalized})
        if normalized["x2_model"] not in {value for _, value in cls.X2_MODEL_CHOICES}:
            normalized["x2_model"] = "x2"
        return normalized

    def config(self) -> dict[str, Any]:
        from postprocessing import spatial_upsamplers as upsampler_api
        return upsampler_api.read_config_section(self.server_config, self)

    @classmethod
    def query_upsampler_def(cls) -> dict[str, Any]:
        return {
            "name": "RelayVSR",
            "upsampler_types": (UPSAMPLER_TYPE_POSTPROCESSING,),
            "media": ("video", "image"),
            "profile": UPSAMPLER_PROFILE_VIDEO,
            "config_key": METHOD,
            "pos": 25,
            "methods": [("RelayVSR", METHOD)],
            "vae_methods": [],
            "multipliers": {METHOD: cls.UPSAMPLING_RATIOS},
            "default_spatial_upsampling": format_multiplier_value(METHOD, 4.0),
            "postprocessing_category": "upsampler",
            "description": "Fast generative upscaling that restores detail on real-world and generated video: every 15 frames a one-step Wan 2.2 5B model generates a detailed keyframe, and a small decoder carries its detail to every frame, several times faster than FlashVSR+ Turbo. Its x4 model serves ratios above x2 and the RelayVSR X2 model ratios up to x2 (more than twice as fast as the x4 model there, a Configuration choice); other ratios are resized from their output. Its memory grows with the size of the source, not with the length of the video, and stays in RAM with the Minimum VRAM usage (otherwise what does not fit in VRAM): it suits sources up to about 540p at x4, larger ones at x2.",
            "media_descriptions": {"video": "Frames are processed in a stream of 16-frame windows sharing their keyframes, so long videos need no more memory than short ones."},
        }

    def enabled(self) -> bool:
        return True

    def create_config_ui(self, gr, config: dict[str, Any], *, lock_config: bool = False):
        with gr.Row():
            x2_model = gr.Dropdown(choices=self.X2_MODEL_CHOICES, value=config["x2_model"], label="RelayVSR Model For Upscaling Up To x2", info="RelayVSR X2 upscales natively by 2, with a quarter of the x4 model's work and memory for the same source; the x4 model upscales by 4, then the result is resized down. Ratios above x2 always use the x4 model. Each model downloads on its first use.", interactive=not lock_config)
        return [("x2_model", x2_model)]

    def validate_upsampling(self, spatial_upsampling, image_mode: int) -> str:
        split = self.split_value(spatial_upsampling)
        if split is None or split[0] != METHOD or split[1] not in self.UPSAMPLING_RATIOS:
            return "RelayVSR only supports x1.0 through x4.0 in 0.5 increments"
        return ""

    def _native(self, spatial_upsampling) -> int:
        """The checkpoints' native scale for a value: x2 for ratios up to 2 when the configuration selects RelayVSR X2, x4 otherwise."""
        return 2 if self.split_value(spatial_upsampling)[1] <= 2 and self.config()["x2_model"] == "x2" else 4

    def _filenames(self, native: int) -> tuple[str, str]:
        return f"{FILE_PREFIXES[native]}_keyframes_int8_convrot.safetensors", f"{FILE_PREFIXES[native]}_decoder_bf16.safetensors"

    def _download_def(self, native: int) -> dict[str, Any]:
        return {"repoId": WEIGHTS_REPO, "sourceFolderList": [FOLDER], "fileList": [list(self._filenames(native))]}

    def query_download_def(self, enabled_only: bool = True, spatial_upsampling=None):
        if spatial_upsampling is not None:
            return self._download_def(self._native(spatial_upsampling))
        return [self._download_def(native) for native in ((4, 2) if self.config()["x2_model"] == "x2" else (4,))]

    def query_checkpoint_inventory(self):
        return [{"name": name, "variant": "INT8", "download_defs": [self._download_def(native)]} for native, name in ((4, "RelayVSR"), (2, "RelayVSR X2"))]

    def download(self, process_files: Callable[..., Any], send_cmd=None, status_text: str | None = None, spatial_upsampling=None) -> bool:
        from shared.utils.download import process_files_def_if_needed
        return process_files_def_if_needed(self.query_download_def(spatial_upsampling=spatial_upsampling), process_files=process_files, send_cmd=send_cmd, status_text=status_text)

    def load_upsampler(self, spatial_upsampling, *, process_files: Callable[..., Any], init_pipe: Callable[..., int], profile, **kwargs):
        if self.validate_upsampling(spatial_upsampling, 0):
            raise ValueError(f"Unknown RelayVSR upsampling mode: {spatial_upsampling}")
        self.download(process_files, spatial_upsampling=spatial_upsampling)
        from .runtime import RelayVSRPaths, load_models
        native = self._native(spatial_upsampling)
        keyframes, decoder = (self.files_locator.locate_file(os.path.join(FOLDER, filename)) for filename in self._filenames(native))
        load_models(RelayVSRPaths(keyframes=keyframes, decoder=decoder, scale=native), init_pipe=init_pipe, profile=profile)

    def upscale(self, sample, spatial_upsampling, *, seed=0, frame_offset=0, continue_cache=None, return_continue_cache=False, abort_callback=None, progress_callback=None, **kwargs):
        """continue_cache / return_continue_cache: the stream a Media Flow chunk hands to the next one (runtime.CarriedStream), or a
        runtime.StreamReplay for a chunk that no stream continues; frame_offset: the sample's first frame in its source."""
        split = self.split_value(spatial_upsampling)
        if split is None:
            raise ValueError(f"Unknown RelayVSR upsampling mode: {spatial_upsampling}")
        from .runtime import upscale_video
        return upscale_video(sample, split[1], seed=seed, frame_offset=frame_offset, continue_cache=continue_cache, return_continue_cache=return_continue_cache, abort_callback=abort_callback, progress_callback=progress_callback)

    def release_vram(self) -> None:
        from .runtime import release_models
        release_models()
