# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import gradio as gr

from shared.utils import latent_io


@dataclass(frozen=True)
class LatentProcessHandler:
    """Runs a job from a saved pre-decode latent companion file (``*_latent.pt``).

    'decode' re-runs only the VAE and writes the latent back to pixels;
    multi-window videos (the all-latent sidecar) are decoded window by
    window and re-assembled with the exact trims of the original job,
    reproducing the video unchanged.
    """

    action: str
    model_type: str
    model_label: str
    prompt: str

    chunk_size_seconds = 86400.0
    frame_step = 1
    overlap_frames = 0
    hide_sliding_window_overlap = True
    hide_output_resolution = True
    hide_prompt = True

    def _require_latent_entry(self, source_path: str) -> tuple[str, dict]:
        latent_path = latent_io.find_latent_file(source_path)
        if latent_path is None:
            raise gr.Error("No saved latent companion file was found next to this media. Latent processes only work on media generated with the 'Save Latents' option enabled.")
        entry = latent_io.latent_payload(latent_io.load_latent_file(latent_path))
        if entry is None:
            raise gr.Error(f"{latent_path.split('/')[-1]} is not a WanGP latent companion file.")
        return str(latent_path), entry

    def build_queue_settings(self, process_settings: dict, *, source_path: str, start_frame: int, frame_count: int, target_control: str, seed: int, continue_cache: Any, audio_track_no: int | None = None) -> dict:
        latent_path, entry = self._require_latent_entry(source_path)
        if self.action != "decode":
            raise gr.Error(f"Unsupported latent process action: {self.action}")
        return {
            "mode": "edit_latent",
            "model_type": self.model_type,
            "image_mode": 0,
            "video_source": source_path,
            "_api": {
                "return_media": True,
                "suppress_source_audio": True,
                "suppress_metadata_images": True,
                "latent_action": "decode",
                "latent_source": latent_path,
            },
        }

    def supports_continue_cache(self) -> bool:
        return False

    def supports_continue_cache_for_target(self, value: str | None) -> bool:
        return False


LATENT_DECODE_HANDLER = LatentProcessHandler("decode", "__latent_decode", "Latent Decode", "")
