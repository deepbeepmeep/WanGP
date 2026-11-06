# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from mmgp import offload
from postprocessing.face_refiner.bridge import ASSET_DOWNLOAD_DEF, FaceRefineJob, FaceRefinerBridge, FaceSegment, PrivateRuntime, UNDETECTED_FRAMES_CHOICES, face_count_parameter, face_reference_images_parameter, face_strength_parameter, model_step_callback
from postprocessing.spatial_upsamplers import PARAMETER_UI_LATE_POSTPROCESSING, PARAMETER_UI_MEDIA_FLOW, PARAMETER_UI_POSTPROCESSING, POSTPROCESSING_CATEGORY_REFINER, UPSAMPLER_PROFILE_VIDEO, UPSAMPLER_TYPE_POSTPROCESSING
from shared.hubert import HUBERT_FOLDER
from shared.utils import files_locator as fl
from shared.utils.media_control import controlled_model_loading, loading_callback

METHOD = "taomatefvsr"
RUNTIME_NAME = "TaoMate-FVSR Face Refiner"
WEIGHTS_REPO = "DeepBeepMeep/Wan2.2"
TRANSFORMER_FILE = "TaoMate-FVSR_{}.safetensors"
AUDIO_ENCODER_FILE = "hubert_large_ll60k_13layers_{}.safetensors"
AUDIO_ENCODER_ASSETS = ("config.json", "preprocessor_config.json")
VAE_FILE = "Wan2.2_VAE.safetensors"
WORKING_SIZES = {"auto": (512, 1024), "512": (512,), "1024": (1024,)}
WORKING_SIZE_CHOICES = [("Auto: 512, 1024 For Large Faces", "auto"), ("512: Faster", "512"), ("1024: Slower, Sharper On Large Faces", "1024")]
FOLLOW_AUDIO_LARGEST, FOLLOW_AUDIO_ALL, FOLLOW_AUDIO_NONE = "largest", "all", "none"


def weights_format(quantization: str) -> str:
    """The checkpoint of a shared quantization setting: there is no FP8 file, FP8 takes INT8's, as for the models."""
    return "int8_convrot" if quantization in ("int8", "fp8") else "bf16"


def model_download_defs(transformer_format: str, audio_encoder_format: str) -> list[dict[str, Any]]:
    return [{"repoId": WEIGHTS_REPO, "sourceFolderList": ["", HUBERT_FOLDER], "fileList": [[TRANSFORMER_FILE.format(transformer_format), VAE_FILE], [AUDIO_ENCODER_FILE.format(audio_encoder_format), *AUDIO_ENCODER_ASSETS]]}, ASSET_DOWNLOAD_DEF]


class TaoMateRuntime(PrivateRuntime):
    formats = None

    @controlled_model_loading
    def load(self, formats: tuple[str, str]) -> None:
        if self.model is not None and self.formats == formats:
            return
        self.release()
        import wgp
        from models.wan.taomate.refine import TaoMatePipeline

        tqdm.write("[FaceRefine] Loading the private TaoMate-FVSR runtime")
        transformer, audio_encoder = fl.locate_file(TRANSFORMER_FILE.format(formats[0])), fl.locate_file(os.path.join(HUBERT_FOLDER, AUDIO_ENCODER_FILE.format(formats[1])))
        with wgp.model_unload_guard(), offload.loading_context(loading_callback(), {transformer: "transformer", audio_encoder: "text_encoder"}):
            torch.set_default_device(None)  # cpu, without the DeviceContext mode that routes every torch call through Python
            pipeline = TaoMatePipeline(transformer, audio_encoder, fl.locate_file(VAE_FILE))
            pipe = {"transformer": pipeline.model, "vae": pipeline.vae.model, "text_encoder": pipeline.audio_encoder.model}
            kwargs = {"coTenantsMap": {}}
            profile = wgp.init_pipe(pipe, kwargs, 0, extension=True)
            if "prefetch_window" in wgp.server_config:
                kwargs["prefetchWindow"] = wgp.server_config["prefetch_window"]
            if wgp.server_config.get("smart_memory_pinning", True):
                kwargs["smartPinning"] = 0
            offloadobj = offload.profile(pipe, profile_no=profile, quantizeTransformer=False, vram_safety_coefficient=wgp.args.vram_safety_coefficient, convertWeightsFloatTo=pipeline.dtype, loading_callback=loading_callback(), **kwargs)
        self.formats = formats
        self.adopt(pipeline, offloadobj)


RUNTIME = TaoMateRuntime(RUNTIME_NAME)


def restored_reference(wgp, method, crop, native_size, seed, abort_callback, progress_callback):
    """A reference crop (3, size, size) uint8 restored at the canvas size by an image-capable upsampler, from its native resolution."""
    from postprocessing import spatial_upsamplers as upsampler_api

    size = crop.shape[-1]
    ratios = upsampler_api.method_multipliers(method)
    ratio = max((value for value in ratios if size / value >= native_size * 0.98), default=min(ratios))
    side = round(size / ratio / 8) * 8
    small = F.interpolate(crop.unsqueeze(0).float(), size=(side, side), mode="area", antialias=False)[0].round_().byte().unsqueeze(1)
    print(f"[FaceRefine] Restoring the reference with {method}: {side}x{side} x{ratio:g}")
    restored = wgp.perform_spatial_upsampling(small, upsampler_api.build_upsampling_value(method, ratio), still_image=True, seed=seed, abort_callback=abort_callback, progress_callback=progress_callback)
    if restored is None:
        return None
    restored = restored[:, 0].float() if restored.dtype == torch.uint8 else restored[:, 0].float().add_(1.0).mul_(127.5)
    return F.interpolate(restored.unsqueeze(0), size=(size, size), mode="bicubic", antialias=True)[0].clamp_(0.0, 255.0).round_().byte()


class TaoMateEngine:
    """Restores a track from its audio and its reference, chosen as H3 Face Refiner does: the picture supplied for this person, else the track's best frame."""

    def __init__(self, job: FaceRefineJob, formats: tuple[str, str]):
        from shared.hubert import audio_16k_mono

        self.job = job
        self.formats = formats
        self.pipeline = None
        self.pcm = audio_16k_mono(job.source_audio_path) if job.source_audio_path else None if job.audio_waveform is None else audio_16k_mono(waveform=job.audio_waveform, sample_rate=job.audio_sample_rate)
        if self.pcm is None:
            print("[FaceRefine] The video has no audio: restoring its faces without audio guidance")
        self.tile_size = self._tile_size(job.vae_config)

    @staticmethod
    def _tile_size(vae_config) -> int:
        from models.wan.modules.vae2_2 import Wan2_2_VAE

        return Wan2_2_VAE.get_VAE_tile_size(vae_config, torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 1048576, False)

    def _track_reference(self, track_index, transform, matched_reference):
        """The track's reference, uint8 (3, size, size), restored when a restorer is selected unless a supplied picture is already sharp at this size; None when aborted."""
        from postprocessing.face_refiner.face import best_reference_crop, reference_image_crop

        job, size = self.job, transform["canvas"][0]
        supplied = None if matched_reference is None else reference_image_crop(matched_reference, job.detector_path, job.config["crop_factor"], size)
        if supplied is not None:
            crop, native = supplied
            print(f"[FaceRefine] Face track {track_index + 1}: using the supplied reference image, {native:.0f}px around the face")
            if native >= size:
                return crop
        else:
            if matched_reference is not None:
                print(f"[FaceRefine] Face track {track_index + 1}: no face detected in the supplied reference image; using the track's best frame")
            frame, crop = best_reference_crop(job.frames, transform, job.insightface_model_dir)
            native = transform["boxes"][frame][3]
            print(f"[FaceRefine] Face track {track_index + 1}: using stabilized single-face crop from pose-aware source frame {frame + 1} as reference, {native:.0f}px")
        method = job.config["reference_restorer"]
        return crop if method == "none" else restored_reference(job.wgp, method, crop, max(1, round(native)), job.seed, job.abort_callback, job.progress_callback)

    def prepare(self, tracked_faces):
        """The reference of each track, restored before the model loads."""
        job, tracks = self.job, []
        follow_audio = job.parameters.get("follow_audio", FOLLOW_AUDIO_LARGEST)
        heights = [np.mean([box[3] - box[1] for box in transform["raw_face_boxes"] if box is not None]) for transform, _strengths, _matched_reference in tracked_faces]
        for track_index, (transform, strengths, matched_reference) in enumerate(tracked_faces):
            with_audio = follow_audio == FOLLOW_AUDIO_ALL or follow_audio == FOLLOW_AUDIO_LARGEST and heights[track_index] == max(heights)
            print(f"[FaceRefine] Face track {track_index + 1}: {'follows the soundtrack' if with_audio and self.pcm is not None else 'restored without the soundtrack'}")
            reference = self._track_reference(track_index, transform, matched_reference)
            if reference is None:
                return None
            tracks.append((transform, strengths, (reference, with_audio)))
        RUNTIME.load(self.formats)
        self.pipeline = RUNTIME.model
        return tracks

    def window_count(self, frames: int) -> int:
        from models.wan.taomate.refine import window_plans

        return len(window_plans(frames))

    def refine_segment(self, segment: FaceSegment):
        job, config, pipeline = self.job, self.job.config, self.pipeline
        progress = segment.progress_callback

        def status(phase):
            if callable(progress):
                progress(phase)

        pipeline._interrupt = False
        reference, with_audio = segment.data
        try:
            return pipeline.refine(segment.stop - segment.start, lambda start, stop: segment.load_crops(start, stop, uint8_storage=True), reference, self.pcm if with_audio else None,
                                   first_frame=job.frame_offset + segment.start, fps=job.fps, size=segment.transform["canvas"][0], seed=job.seed + segment.track_index,
                                   steps=config["steps"], noise=config["noise"], shift=config["flow_shift"], audio_cfg=config["audio_guidance"], ref_cfg=config["reference_guidance"], tile_size=self.tile_size,
                                   window_offset=segment.window_index_offset, window_total=segment.window_count_total, callback=model_step_callback(pipeline, job.abort_callback, progress, "Denoising", config["steps"]), set_progress_status=status)
        finally:
            pipeline._interrupt = False

    def close(self):
        RUNTIME.unload_all()  # nothing is loaded yet when aborted while the references are restored


class TaoMateFVSRBridge(FaceRefinerBridge):
    METHOD = METHOD
    NAME = RUNTIME_NAME
    STRENGTH_DEFAULT = 1.0
    CONFIGURABLE_FIELDS = ("undetected_frames", "working_size", "reference_restorer")
    ENGINE_CONFIG = {
        "reference_restorer": "seedvr2",
        "crop_factor": 1.47,  # upstream's crops, measured with this detector: 1.42 to 1.53 times the face height, centered on the face
        "paste_region": "full_crop",  # the crop inside soft borders, as upstream's paste mask: the jaw follows the audio beyond the face's box
        "feather_fraction": 0.16,
        "working_size": "auto",
        "steps": 10,
        "noise": 0.8,
        "flow_shift": 5.0,
        "audio_guidance": 4.0,
        "reference_guidance": 1.7,
    }

    @classmethod
    def normalize_config_section(cls, config: dict[str, Any]) -> dict[str, Any]:
        values = super().normalize_config_section(config)
        if values["working_size"] not in WORKING_SIZES:
            values["working_size"] = "auto"
        values["canvas_tiers"] = WORKING_SIZES[values["working_size"]]
        return values

    @classmethod
    def query_upsampler_def(cls) -> dict[str, Any]:
        return {
            "name": RUNTIME_NAME,
            "upsampler_types": (UPSAMPLER_TYPE_POSTPROCESSING,),
            "media": ("video",),
            "profile": UPSAMPLER_PROFILE_VIDEO,
            "config_key": "taomate_fvsr",
            "pos": 10_002,
            "methods": [("TaoMate-FVSR Face Refiner", METHOD)],
            "vae_methods": [],
            "default_spatial_upsampling": METHOD,
            "postprocessing_category": POSTPROCESSING_CATEGORY_REFINER,
            "progress_label": "Face Refiner",
            "source_audio_conditioning": True,
            "description": "Restore speaking and singing faces from the video's soundtrack: sharper lips and teeth that follow the audio, more facial detail. Keeps the rest of the video and its resolution. Choose it for close-ups or when a mouth looks wrong; it pushes the mouth toward the audio much harder than H3 Face Refiner, so the mouth can differ more from the source.",
            "media_descriptions": {"video": "**Pros**\n\n- Sharpest lips and teeth, and the most skin and eye detail, on close-ups.\n- On close-ups the upper face and identity stay close to the source.\n- The one to try when a mouth looks wrong.\n- You choose which faces follow the soundtrack, so listeners can stay quiet.\n- Faster than H3 Face Refiner on a single close-up.\n\n**Cons**\n\n- The mouth can differ from the source: closed lips may open, and a face following the soundtrack mouths its words even if that person is only listening.\n- On small or distant faces, mouths open wider and the face is less faithful than with H3 Face Refiner.\n- Takes no prompt.\n- Damage in a face's best frame can remain unless you supply a clean picture of that person.\n- Slower than H3 Face Refiner with several faces, and uses a little more VRAM. Faces large in the frame take several times longer.\n- Downloads SeedVR2 as well, to restore its reference picture."},
            "method_parameters": {METHOD: [
                face_count_parameter(),
                face_strength_parameter(1.0, "Face Refinement Strength - lower blends the original face back", "How much of the restored face replaces the original one. It blends the result over the source: below 1 the original mouth can show through."),
                {"name": "follow_audio", "setting": "follow_audio", "type": "string", "component": "dropdown", "ui": (PARAMETER_UI_POSTPROCESSING, PARAMETER_UI_LATE_POSTPROCESSING, PARAMETER_UI_MEDIA_FLOW), "required": False, "default": FOLLOW_AUDIO_LARGEST,
                 "choices": [("Largest Face", FOLLOW_AUDIO_LARGEST), ("All Faces", FOLLOW_AUDIO_ALL), ("No Face", FOLLOW_AUDIO_NONE)], "enum": [FOLLOW_AUDIO_LARGEST, FOLLOW_AUDIO_ALL, FOLLOW_AUDIO_NONE], "label": "Faces Following the Soundtrack",
                 "description": "Which refined faces have their mouth driven by the soundtrack. A face following it speaks its words even if that person is listening, so keep Largest Face when other people are in the shot and choose All Faces only when they all sing or speak together. The others are restored from the picture alone, which also suits music without vocals."},
                face_reference_images_parameter(),
            ]},
        }

    def create_config_ui(self, gr, config: dict[str, Any], *, lock_config: bool = False):
        from postprocessing import spatial_upsamplers as upsampler_api

        restorers = [("None: The Reference As It Is", "none")]
        for definition in upsampler_api.query_upsampler_defs(UPSAMPLER_TYPE_POSTPROCESSING, enabled_only=True):
            if "image" in definition.get("media", ()):  # every method, those the upsampler does not currently offer included: they stay valid
                restorers += [(label, method) for label, method in definition["methods"] if method != "lanczos" and definition.get("multipliers", {}).get(method)]
        with gr.Group():
            with gr.Row():
                undetected_frames = gr.Dropdown(choices=UNDETECTED_FRAMES_CHOICES, value=config["undetected_frames"], label="TaoMate-FVSR Undetected Frames", interactive=not lock_config)
                working_size = gr.Dropdown(choices=WORKING_SIZE_CHOICES, value=config["working_size"], label="TaoMate-FVSR Working Size", info="The model restores square crops of this size, then pastes them at the face's size. 1024 takes several times longer and more VRAM.", interactive=not lock_config)
                reference_restorer = gr.Dropdown(choices=restorers, value=config["reference_restorer"], label="TaoMate-FVSR Reference Restorer", info="Restores the best frame of each face, or a small supplied reference: the model takes its skin, eyes and hair detail from it. Downloaded with the refiner and run once per face.", interactive=not lock_config)
        controls = locals().copy()
        return [(name, controls[name]) for name in self.CONFIGURABLE_FIELDS]

    def stitch_blend(self, config: dict[str, Any], strength: float) -> float:
        return config["blend"] * strength

    def _formats(self) -> tuple[str, str]:
        return weights_format(self.server_config.get("transformer_quantization", "int8")), weights_format(self.server_config.get("text_encoder_quantization", "int8"))

    def query_checkpoint_inventory(self):
        return [{"variant": label, "download_defs": model_download_defs(weights, weights)} for label, weights in (("INT8", "int8_convrot"), ("BF16", "bf16"))]  # the restorer lists its own files

    def query_download_defs(self) -> list[dict[str, Any]]:
        """The refiner's files for the shared quantization settings and those of the selected reference restorer, queried from its owner: all there before any GPU work."""
        from postprocessing import spatial_upsamplers as upsampler_api

        method = self.config()["reference_restorer"]
        restorer = [] if method == "none" else upsampler_api.download_defs_for_value(upsampler_api.build_upsampling_value(method, upsampler_api.default_multiplier_for_method(method)))
        return model_download_defs(*self._formats()) + restorer

    def open_engine(self, job: FaceRefineJob):
        return TaoMateEngine(job, self._formats())

    def release_private_runtime(self) -> None:
        RUNTIME.release()


__all__ = ["TaoMateFVSRBridge"]
