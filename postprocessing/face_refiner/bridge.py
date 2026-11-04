# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""The post-processor every face refiner is: tracks the faces of a video, has an engine restore each track's crops, pastes them back, with what
refiners running a model of their own share: PrivateRuntime, model_step_callback. A refiner declares its method, parameters and files, and opens
an engine with:
- prepare(tracked_faces): [(transform, strengths, data)] per track of track_faces, data being the engine's own, or None when aborted;
- window_count(frames): the model windows of a segment of that many frames, for the progress;
- refine_segment(segment): the restored crops, uint8 (3, frames, height, width), or None when aborted;
- close(): at the end, also when aborted or failing."""
from __future__ import annotations

import gc
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable

import torch

from postprocessing.spatial_upsamplers import PARAMETER_UI_LATE_POSTPROCESSING, PARAMETER_UI_MEDIA_FLOW, PARAMETER_UI_POSTPROCESSING, SimpleScaleSuffixMixin
from shared.utils import offload_registry

ASSET_REPO = "DeepBeepMeep/MiniMax-H3"
ASSET_FOLDER = "buffalo_l"
DETECTOR_FILE = "face_yolov8m.pt"
FALLBACK_DETECTOR_FILE = "person_yolov8m-seg.safetensors"
DETECTOR_PATH = f"{ASSET_FOLDER}/{DETECTOR_FILE}"
FALLBACK_DETECTOR_PATH = f"{ASSET_FOLDER}/{FALLBACK_DETECTOR_FILE}"
INSIGHTFACE_FILES = ("det_10g.onnx", "2d106det.onnx", "w600k_r50.onnx")
ASSET_DOWNLOAD_DEF = {"repoId": ASSET_REPO, "sourceFolderList": [ASSET_FOLDER], "fileList": [[DETECTOR_FILE, FALLBACK_DETECTOR_FILE, *INSIGHTFACE_FILES]]}
UNDETECTED_FRAMES_CHOICES = [("Fade out", "fade_out"), ("Skip", "skip"), ("Composite anyway", "composite_anyway")]


def face_progress_callback(callback, face_index: int | None = None, face_count: int = 1):
    if not callable(callback):
        return callback

    def report(phase, current=None, total=None):
        phase = str(phase)
        status = "Face Refiner"
        if face_index is not None and face_count > 1:
            status += f", Face {face_index + 1}/{face_count}"
        window_pos = phase.find("Window ")
        if window_pos >= 0:
            window, separator, phase = phase[window_pos:].partition(" - ")
            status += f", {window}"
            phase = phase if separator else ""
        if phase:
            status += f" - {phase}"
        callback(status, current, total)

    return report


def model_step_callback(pipeline, abort_callback, progress_callback, title, total, label=""):
    """The callback of a WanGP pipeline during a refinement: aborts it, reports its phases with their steps."""

    def step_callback(step_idx=-1, _latent=None, _force_refresh=False, read_state=False, **kwargs):
        if callable(abort_callback) and abort_callback():
            pipeline._interrupt = True
            return
        if read_state:
            return
        phase = kwargs.get("progress_title", title)
        if callable(progress_callback):
            progress_callback(f"{label} - {phase}" if label else phase, int(step_idx) + 1, kwargs.get("override_num_inference_steps", int(total)))

    step_callback.checkpoint = lambda: step_callback(read_state=True)
    return step_callback


class PrivateRuntime:
    """A model a refiner loads for itself, registered for WanGP's unload tool under the refiner's name."""

    def __init__(self, name):
        self.name = name
        self.model = self.offloadobj = None

    def adopt(self, model, offloadobj):
        self.model, self.offloadobj = model, offloadobj
        offload_registry.register_offloadobj(self.name, offloadobj, self.release)

    def _before_release(self):
        pass

    def release(self) -> None:
        if self.offloadobj is not None:
            offload_registry.unregister_offloadobj(self.name, self.offloadobj)
            self._before_release()
            self.offloadobj.release()
        self.model = self.offloadobj = None
        gc.collect()

    def unload_all(self) -> None:
        if self.offloadobj is not None:
            self.offloadobj.unload_all()


@dataclass
class FaceRefineJob:
    """One video: what an engine needs beside the tracks."""
    wgp: Any
    detector_path: str
    config: dict[str, Any]
    frames: torch.Tensor
    prompt: str
    seed: int
    fps: float
    frame_offset: int
    strength: float
    audio_waveform: Any
    audio_sample_rate: int
    source_audio_path: str | None
    image_refs_relative_size: float
    vae_tile_size: Any
    vae_config: int
    loaded_model_context: Any
    insightface_model_dir: str
    abort_callback: Callable | None
    progress_callback: Callable | None
    parameters: dict[str, Any]  # the refiner's own declared parameters


@dataclass
class FaceSegment:
    """A continuous presence of a face in a track: the frames start:stop of the video."""
    track_index: int
    transform: dict[str, Any]
    strengths: torch.Tensor
    data: Any
    start: int
    stop: int
    load_crops: Callable
    progress_callback: Callable | None
    window_index_offset: int
    window_count_total: int


def face_count_parameter(description="Choose an exact maximum from 1 to 5, or 0 for automatic selection. Runtime increases approximately linearly with the selected face count."):
    return {"name": "face_count", "setting": "face_count", "type": "integer", "component": "slider", "ui": (PARAMETER_UI_POSTPROCESSING, PARAMETER_UI_LATE_POSTPROCESSING, PARAMETER_UI_MEDIA_FLOW), "required": False, "default": 1, "minimum": 0, "maximum": 5, "step": 1, "label": "Faces to Refine", "label_long": "Faces to Refine - 0 selects automatically, up to 5; more faces take longer", "description": description}


def face_strength_parameter(default, label_long, description):
    return {"name": "face_strength", "setting": "strength", "type": "number", "component": "slider", "ui": (PARAMETER_UI_POSTPROCESSING, PARAMETER_UI_LATE_POSTPROCESSING, PARAMETER_UI_MEDIA_FLOW), "required": False, "default": default, "minimum": 0.0, "maximum": 1.0, "step": 0.05, "label": "Face Refinement Strength", "label_long": label_long, "description": description}


def face_reference_images_parameter():
    return {"name": "reference_images", "setting": "reference_images", "type": "array", "component": "images", "ui": (PARAMETER_UI_LATE_POSTPROCESSING,), "required": False, "default": [], "label": "Optional Face Reference Images", "description": "Optional reference-image media ids. InsightFace matches them to detected identities irrespective of order; unmatched tracks use an automatically selected source frame.", "multiple": True, "media_type": "image"}


class FaceRefinerBridge(SimpleScaleSuffixMixin):
    METHOD = ""
    NAME = ""
    STRENGTH_DEFAULT = 0.75
    ENGINE_CONFIG: dict[str, Any] = {}
    CONFIGURABLE_FIELDS = ("undetected_frames",)

    def __init__(self, server_config: dict[str, Any], files_locator):
        self.server_config = server_config
        self.files_locator = files_locator

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return {
            "detector_confidence": 0.25,
            "crop_factor": 1.6,
            "canvas_mode": "auto_no_downscale",
            "canvas_width": 512,
            "canvas_height": 512,
            "canvas_tiers": None,
            "smooth_window": 21,
            "size_smooth_window": 51,
            "smooth_method": "gaussian",
            "size_mode": "per_frame",
            "identity_threshold": 0.28,
            "auto_min_face_height": 32,
            "auto_min_presence": 0.2,
            "fallback_detector": FALLBACK_DETECTOR_FILE,
            "fallback_head_frac": 0.5,
            "paste_region": "face_only",
            "mask_dilation": 24,
            "feather": 24,
            "feather_fraction": None,
            "colour_match": 1.0,
            "blend": 1.0,
            "undetected_frames": "fade_out",
            "feather_scales_with_crop": False,
            **cls.ENGINE_CONFIG,
        }

    @classmethod
    def normalize_config_section(cls, config: dict[str, Any]) -> dict[str, Any]:
        values = cls.default_config()
        config = dict(config or {})
        values.update({key: config[key] for key in cls.CONFIGURABLE_FIELDS if key in config})
        if values["undetected_frames"] not in ("fade_out", "skip", "composite_anyway"):
            values["undetected_frames"] = "fade_out"
        return values

    def config(self) -> dict[str, Any]:
        from postprocessing import spatial_upsamplers as upsampler_api

        return upsampler_api.read_config_section(self.server_config, self)

    def enabled(self) -> bool:
        return True

    def validate_upsampling(self, spatial_upsampling, image_mode: int) -> str:
        split = self.split_value(spatial_upsampling)
        if image_mode:
            return f"{self.NAME} is available for videos only"
        if split is None or split[0] != self.METHOD:
            return f"Invalid {self.NAME} selection"
        return ""

    def _download_extras(self, wgp):
        pass

    def download(self, process_files: Callable[..., Any], send_cmd=None, status_text: str | None = None, spatial_upsampling=None) -> bool:
        """query_download_defs, then what a refiner finds another way."""
        wgp = sys.modules.get("wgp")
        if wgp is None:
            return False
        from shared.utils.download import send_download_status

        send_download_status(send_cmd, status_text)
        for download_def in self.query_download_defs():
            process_files(**download_def)
        self._download_extras(wgp)
        return True

    def load_upsampler(self, spatial_upsampling, *, process_files: Callable[..., Any], profile, **kwargs):
        error = self.validate_upsampling(spatial_upsampling, 0)
        if error:
            raise ValueError(error)
        self.download(process_files, spatial_upsampling=spatial_upsampling)

    def refines(self, config: dict[str, Any], strength: float) -> bool:
        return config["blend"] != 0.0 and strength != 0.0

    def stitch_blend(self, config: dict[str, Any], strength: float) -> float:
        return config["blend"]

    def open_engine(self, job: FaceRefineJob):
        raise NotImplementedError

    def upscale(self, sample, spatial_upsampling, *, loaded_model_context=None, prompt="", seed=0, fps=24.0, frame_offset=0,
                audio_waveform=None, audio_sample_rate=0, source_audio_path=None, vae_tile_size=None, vae_config=0, still_image=False,
                reference_images=None, image_refs_relative_size=100.0, face_count=1, strength=None,
                abort_callback=None, progress_callback=None, profile=-1, **kwargs):
        if still_image:
            raise ValueError(f"{self.NAME} is available for videos only")
        error = self.validate_upsampling(spatial_upsampling, 0)
        if error:
            raise ValueError(error)
        wgp = sys.modules["wgp"]
        from .face import crop_face_track, frames_to_sample, sample_to_frames, stitch, track_faces

        config = self.config()
        strength = max(0.0, min(1.0, float(self.STRENGTH_DEFAULT if strength is None else strength)))
        if not self.refines(config, strength):
            return sample, None
        raw_progress_callback = progress_callback
        progress_callback = face_progress_callback(raw_progress_callback)
        frames, source_format = sample_to_frames(sample)
        reference_images = wgp.clean_image_list(reference_images or []) or []
        detector_path = self.files_locator.locate_file(DETECTOR_PATH)
        fallback_detector_path = None if config["fallback_detector"] == "none" else self.files_locator.locate_file(FALLBACK_DETECTOR_PATH)
        insightface_model_path = self.files_locator.locate_file(f"{ASSET_FOLDER}/{INSIGHTFACE_FILES[0]}")
        insightface_model_dir = os.path.dirname(os.path.abspath(insightface_model_path))
        tracked_faces = track_faces(frames, detector_path, face_count=max(0, min(5, int(face_count))), reference_images=reference_images,
                                    confidence=config["detector_confidence"], crop_factor=config["crop_factor"],
                                    canvas_width=config["canvas_width"], canvas_height=config["canvas_height"], canvas_mode=config["canvas_mode"], canvas_tiers=config["canvas_tiers"],
                                    smooth_window=config["smooth_window"], size_smooth_window=config["size_smooth_window"], smooth_method=config["smooth_method"],
                                    size_mode=config["size_mode"], identity_threshold=config["identity_threshold"], auto_min_face_height=config["auto_min_face_height"],
                                    auto_min_presence=config["auto_min_presence"], fallback_detector_path=fallback_detector_path, insightface_model_dir=insightface_model_dir,
                                    fallback_head_frac=config["fallback_head_frac"], strength_small_face=strength,
                                    strength_large_face=strength, strength_smooth_frames=1, abort_callback=abort_callback,
                                    progress_callback=progress_callback)
        if tracked_faces is None:
            return None, None
        if not tracked_faces:
            print("[FaceRefine] Auto selection found no relevant face tracks; leaving the source video unchanged")
            return sample, None
        if callable(progress_callback):
            progress_callback("Loading Model")
        job = FaceRefineJob(wgp=wgp, detector_path=detector_path, config=config, frames=frames, prompt=prompt, seed=int(seed), fps=fps, frame_offset=frame_offset, strength=strength,
                            audio_waveform=audio_waveform, audio_sample_rate=audio_sample_rate, source_audio_path=source_audio_path, image_refs_relative_size=image_refs_relative_size,
                            vae_tile_size=vae_tile_size, vae_config=vae_config, loaded_model_context=loaded_model_context, insightface_model_dir=insightface_model_dir,
                            abort_callback=abort_callback, progress_callback=progress_callback, parameters=kwargs)
        engine = self.open_engine(job)
        try:
            tracks = engine.prepare(tracked_faces)
            if tracks is None:
                return None, None
            source_frames = frames
            output = frames.clone()
            for track_index, (transform, strengths, data) in enumerate(tracks):
                track_progress_callback = face_progress_callback(raw_progress_callback, track_index, len(tracks))
                segments = transform["segments"]
                segment_window_counts = [engine.window_count(stop - start) for start, stop in segments]
                total_segment_windows = sum(segment_window_counts)
                window_index_offset = 0
                excluded = len(transform["active"]) - sum(stop - start for start, stop in segments)
                print(f"[FaceRefine] Face track {track_index + 1}: refining {len(segments)} presence segment(s); excluded {excluded} absent frame(s)")
                for segment_index, (start, stop) in enumerate(segments, start=1):
                    print(f"[FaceRefine] Face track {track_index + 1}, segment {segment_index}/{len(segments)}: source frames {start + 1}-{stop}")

                    def load_crops(window_start, window_stop, *, uint8_storage=False):
                        return crop_face_track(source_frames, transform, start + window_start, start + window_stop, uint8_storage=uint8_storage)

                    refined = engine.refine_segment(FaceSegment(track_index, transform, strengths, data, start, stop, load_crops, track_progress_callback, window_index_offset, total_segment_windows))
                    if refined is None:
                        return None, None
                    window_index_offset += segment_window_counts[segment_index - 1]
                    segment_transform = transform.copy()
                    for key in ("boxes", "face_rect", "weights", "detected", "active"):
                        segment_transform[key] = transform[key][start:stop]
                    segment_transform["frames"] = stop - start
                    stitch_progress_callback = track_progress_callback
                    if total_segment_windows > 1 and callable(track_progress_callback):
                        stitch_progress_callback = lambda phase, current=None, total=None: track_progress_callback(f"Window {window_index_offset}/{total_segment_windows} - {phase}", current, total)
                    stitched = stitch(output[start:stop], refined, segment_transform, paste_region=config["paste_region"], mask_dilation=config["mask_dilation"],
                                      feather=config["feather"], feather_fraction=config["feather_fraction"], colour_match=config["colour_match"], blend=self.stitch_blend(config, strength),
                                      undetected_frames=config["undetected_frames"], feather_scales_with_crop=config["feather_scales_with_crop"],
                                      abort_callback=abort_callback, progress_callback=stitch_progress_callback)
                    if stitched is None:
                        return None, None
                    refined = stitched = None
                if track_index + 1 == len(tracks):
                    source_frames = job.frames = frames = None
        finally:
            engine.close()
        return frames_to_sample(output, source_format, sample.dtype), None

    def release_vram(self) -> None:
        self.release_private_runtime()


__all__ = ["ASSET_DOWNLOAD_DEF", "ASSET_FOLDER", "DETECTOR_FILE", "DETECTOR_PATH", "FALLBACK_DETECTOR_FILE", "FALLBACK_DETECTOR_PATH", "FaceRefineJob", "FaceRefinerBridge", "FaceSegment", "INSIGHTFACE_FILES", "PrivateRuntime",
           "UNDETECTED_FRAMES_CHOICES", "face_count_parameter", "face_progress_callback", "face_reference_images_parameter", "face_strength_parameter", "model_step_callback"]
