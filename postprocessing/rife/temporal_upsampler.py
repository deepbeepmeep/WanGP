from __future__ import annotations

from typing import Any

import torch

from postprocessing import temporal_upsamplers as temporal_upsampler_api


RIFE_V4_FILENAME = "rife4.26.pkl"


class RifeTemporalUpsampler(temporal_upsampler_api.SimpleScaleSuffixMixin):
    METHOD = "rife"

    def __init__(self, server_config=None, files_locator=None):
        self.files_locator = files_locator

    @classmethod
    def query_temporal_upsampler_def(cls) -> dict[str, Any]:
        return {
            "name": "RIFE v4.26",
            "config_key": "rife",
            "pos": 10,
            "method_pos": {cls.METHOD: 10},
            "methods": [("RIFE v4.26", cls.METHOD)],
            "multipliers": {cls.METHOD: (2.0, 3.0, 4.0)},
            "default_temporal_upsampling": "rife*2",
            "description": "Fast RIFE v4.26 neural frame interpolation based on learned motion estimation, with smooth x2, x3, and x4 output. Arbitrary timesteps provide exact intermediate positions for every multiplier.",
        }

    def query_download_def(self, **_kwargs) -> dict[str, Any]:
        return {"repoId": "DeepBeepMeep/Wan2.1", "sourceFolderList": [""], "fileList": [[RIFE_V4_FILENAME]]}

    def query_download_defs(self, **_kwargs) -> list[dict[str, Any]]:
        return [self.query_download_def()]

    def download(self, process_files, *, send_cmd=None, status_text: str | None = None, **_kwargs):
        from shared.utils.download import download_def_missing_files, send_download_status

        download_def = self.query_download_def()
        if not download_def_missing_files(download_def):
            return False
        send_download_status(send_cmd, status_text or "Downloading RIFE v4.26 temporal upsampling model...")
        process_files(**download_def)
        return True

    def query_chunking(self, temporal_upsampling):
        """How a long video goes through this upsampler a chunk at a time (postprocessing.chunking)."""
        from postprocessing.system_upsampler_handler import RIFE_HANDLER
        return RIFE_HANDLER

    def validate_upsampling(self, temporal_upsampling, *, source_is_image: bool = False) -> str:
        split = self.split_value(temporal_upsampling)
        if split is None or split[1] not in self.query_temporal_upsampler_def()["multipliers"][self.METHOD]:
            return f"Unknown temporal upsampling mode: {temporal_upsampling}"
        return "Temporal Upsampling can not be used with an Image" if source_is_image else ""

    def temporal_upsample(self, temporal_upsampling, sample, previous_last_frame, fps, *, processing_device="cuda", to_uint8_callback=None, abort_callback=None, progress_callback=None, **kwargs):
        split = self.split_value(temporal_upsampling)
        if split is None:
            return sample, previous_last_frame, fps
        if split[1] not in self.query_temporal_upsampler_def()["multipliers"][self.METHOD]:
            raise ValueError(f"Unknown temporal upsampling mode: {temporal_upsampling}")
        multiplier = int(split[1])
        if multiplier <= 1:
            return sample, previous_last_frame, fps
        rife_model_path = self.files_locator.locate_file(RIFE_V4_FILENAME)
        from postprocessing.rife.inference import temporal_interpolation

        output = temporal_interpolation(rife_model_path, sample, multiplier, device=processing_device, previous_last_frame=previous_last_frame, abort_callback=abort_callback, progress_callback=progress_callback)
        if output is None:
            return None, previous_last_frame, fps * multiplier
        # Keep a single source frame, not a view retaining the complete video.
        last_frame = (sample[:, -1:] if previous_last_frame is not None else output[:, -1:]).to(device="cpu").clone()
        if previous_last_frame is not None:
            output = output[:, 1:]
        return output, last_frame, fps * multiplier
