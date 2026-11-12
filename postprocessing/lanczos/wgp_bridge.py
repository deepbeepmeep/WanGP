# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from postprocessing.spatial_upsamplers import SimpleScaleSuffixMixin, UPSAMPLER_PROFILE_VIDEO, UPSAMPLER_TYPE_POSTPROCESSING
from shared.utils.utils import get_default_workers, process_images_multithread, resize_lanczos


def resize_lanczos_spatial(sample, scale, method=None, size=None):
    h, w = sample.shape[-2:]
    h, w = size if size is not None else (int(round(h * scale / 16) * 16), int(round(w * scale / 16) * 16))
    frames_to_upsample = [sample[:, i] for i in range(sample.shape[1])]
    if sample.dtype == torch.uint8:
        resample = Image.Resampling.LANCZOS if method is None else method

        def upsample_frames(frame):
            np_frame = frame.permute(1, 2, 0).cpu().numpy()
            if np_frame.shape[2] == 1:
                np_frame = np_frame[:, :, 0]
            img = Image.fromarray(np_frame)
            img = img.resize((w, h), resample=resample)
            out = np.array(img)
            if out.ndim == 2:
                out = out[:, :, None]
            return torch.from_numpy(out).permute(2, 0, 1).to(torch.uint8).unsqueeze(1)
    else:
        def upsample_frames(frame):
            return resize_lanczos(frame, h, w, method).unsqueeze(1)
    return torch.cat(process_images_multithread(upsample_frames, frames_to_upsample, "upsample", wrap_in_list=False, max_workers=get_default_workers(), in_place=True), dim=1)


class LanczosUpsampler(SimpleScaleSuffixMixin):
    """Built-in CPU Lanczos resampling."""

    MULTIPLIERS = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)
    batch_image_inputs = True

    def __init__(self, server_config=None, files_locator=None):
        pass

    def query_upsampler_def(self) -> dict:
        return {
            "name": "Lanczos",
            "upsampler_types": (UPSAMPLER_TYPE_POSTPROCESSING,),
            "media": ("video", "image"),
            "profile": UPSAMPLER_PROFILE_VIDEO,
            "config_key": "lanczos",
            "pos": 10,
            "method_pos": {"lanczos": 10},
            "methods": [("Lanczos", "lanczos")],
            "vae_methods": [],
            "multipliers": {"lanczos": self.MULTIPLIERS},
            "default_spatial_upsampling": "lanczos*2",
            "postprocessing_category": "upsampler",
            "description": "A very fast CPU resize with negligible VRAM use. It is sharper than a basic resize, but cannot reconstruct lost detail and may add ringing or aliasing.",
        }

    def query_chunking(self, spatial_upsampling):
        """How a long video goes through this upsampler a chunk at a time (postprocessing.chunking)."""
        from postprocessing.chunking import FRAME_BY_FRAME
        return FRAME_BY_FRAME

    def validate_upsampling(self, spatial_upsampling, image_mode: int) -> str:
        return ""

    def upscale(self, sample, spatial_upsampling, **kwargs):
        scale = self.split_value(spatial_upsampling)[1]
        return resize_lanczos_spatial(sample, scale), None
