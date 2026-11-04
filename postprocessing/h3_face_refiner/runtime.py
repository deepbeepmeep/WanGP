from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from tqdm import tqdm

from postprocessing.face_refiner.bridge import PrivateRuntime, model_step_callback
from shared.utils.audio_video import slice_audio_window


RUNTIME_NAME = "H3 Face Refiner"
DEFAULT_MODEL_TYPE = "minimax_h3_ref2va_pruned"
DEFAULT_WINDOW_FRAMES = 362
DEFAULT_WINDOW_OVERLAP = 18
MAX_WINDOW_FRAMES = 362
TEMPORAL_STRIDE = 17
FRAME_OFFSET = 5
MIN_WINDOW_FRAMES = FRAME_OFFSET + TEMPORAL_STRIDE
DEFAULT_AUDIO_SAMPLE_RATE = 32000
TEXT_ENCODER_CONFIG = "gguf_q4_k_m"


def _report(progress_callback, phase, current=None, total=None):
    if callable(progress_callback):
        progress_callback(phase, current, total)


def window_starts(frame_count: int, window_size: int, window_overlap: int) -> tuple[int, ...]:
    if window_size > MAX_WINDOW_FRAMES or (window_size - FRAME_OFFSET) % TEMPORAL_STRIDE:
        raise ValueError(f"H3 Face Refiner window size must be 5 + 17n and no larger than {MAX_WINDOW_FRAMES}")
    if window_overlap >= window_size:
        raise ValueError("H3 Face Refiner window overlap must be smaller than its window size")
    if frame_count <= window_size:
        return (0,)
    step = window_size - window_overlap
    window_count = (frame_count - window_overlap + step - 1) // step
    return tuple(int(start) for start in np.linspace(0, frame_count - window_size, window_count).round())


def pad_window(video: torch.Tensor, strengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    frame_count = int(video.shape[1])
    padded_count = max(MIN_WINDOW_FRAMES, ((frame_count - FRAME_OFFSET + TEMPORAL_STRIDE - 1) // TEMPORAL_STRIDE) * TEMPORAL_STRIDE + FRAME_OFFSET)
    if padded_count == frame_count:
        return video, strengths
    return (torch.cat((video, video[:, -1:].expand(-1, padded_count - frame_count, -1, -1)), dim=1),
            torch.cat((strengths, strengths[-1:].expand(padded_count - frame_count))))


def crossfade_frames(previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    weights = torch.linspace(0.0, torch.pi, previous.shape[1], device=previous.device, dtype=torch.float32).cos_().mul_(-0.5).add_(0.5).view(1, -1, 1, 1)
    return previous.float().lerp_(current.float(), weights).to(previous.dtype)


def _uint8_to_signed(video: torch.Tensor) -> torch.Tensor:
    output = torch.empty(video.shape, dtype=torch.float32, device="cpu")
    for index in range(video.shape[1]):
        output[:, index].copy_(video[:, index].float().div_(127.5).sub_(1.0))
    return output


def _signed_to_uint8(video: torch.Tensor) -> torch.Tensor:
    output = torch.empty(video.shape, dtype=torch.uint8, device="cpu")
    for index in range(video.shape[1]):
        frame = video[:, index].to(dtype=torch.float32, device="cpu", copy=True).clamp_(-1.0, 1.0).add_(1.0).mul_(127.5).round_().byte()
        output[:, index].copy_(frame)
    return output


def slice_audio_for_window(audio_waveform, audio_sample_rate: int, source_audio_path: str | None, start: int, frame_count: int, padded_count: int, fps: float):
    if source_audio_path:
        audio, sample_rate = slice_audio_window(source_audio_path, start, frame_count, fps)
    else:
        sample_rate = int(audio_sample_rate or DEFAULT_AUDIO_SAMPLE_RATE)
        if audio_waveform is None:
            return np.zeros((int(round(padded_count * sample_rate / fps)), 1), dtype=np.float32), sample_rate
        audio = audio_waveform.detach().cpu().numpy() if torch.is_tensor(audio_waveform) else np.asarray(audio_waveform)
        if audio.ndim == 1:
            audio = audio[:, None]
        start_sample = int(round(start * sample_rate / fps))
        stop_sample = start_sample + int(round(frame_count * sample_rate / fps))
        audio = audio[start_sample:stop_sample]
    target_samples = int(round(padded_count * sample_rate / fps))
    if audio.shape[0] < target_samples:
        audio = np.pad(audio, ((0, target_samples - audio.shape[0]), (0, 0)))
    return np.asarray(audio[:target_samples], dtype=np.float32), sample_rate


@dataclass(frozen=True)
class LoraSnapshot:
    paths: tuple[str, ...]
    active: tuple[str, ...]
    multipliers: tuple[object, ...]
    step: int


def _snapshot_loras(transformer) -> LoraSnapshot:
    adapters = getattr(transformer, "_loras_adapters", None)
    active = tuple(getattr(transformer, "_loras_active_adapters", None) or ())
    scaling = getattr(transformer, "_loras_scaling", None) or {}
    return LoraSnapshot(tuple(adapters.values()) if adapters else (), active,
                        tuple(copy.deepcopy(scaling[adapter]) for adapter in active), int(getattr(transformer, "_lora_step_no", 0)))


def _load_loras(transformer, paths, multipliers, model_type):
    from mmgp import offload
    import wgp

    offload.load_loras_into_model(transformer, list(paths), list(multipliers), activate_all_loras=True,
                                  preprocess_sd=wgp.get_loras_preprocessor(transformer, model_type), pinnedLora=False,
                                  maxReservedLoras=wgp.server_config.get("max_reserved_loras", -1),
                                  split_linear_modules_map=getattr(transformer, "split_linear_modules_map", None))
    if transformer._loras_errors:
        raise RuntimeError("Unable to load H3 Face Refiner LoRA: " + ", ".join(message for _, message in transformer._loras_errors))


def _restore_loras(transformer, snapshot: LoraSnapshot, model_type):
    from mmgp import offload

    if not snapshot.paths:
        offload.unload_loras_from_model(transformer)
        offload.set_step_no_for_lora(transformer, snapshot.step)
        return
    _load_loras(transformer, snapshot.paths, [1.0] * len(snapshot.paths), model_type)
    offload.activate_loras(transformer, list(snapshot.active), list(snapshot.multipliers))
    offload.set_step_no_for_lora(transformer, snapshot.step)


@contextmanager
def turbo_lora(pipeline, turbo_path: str | None, model_type: str, multiplier: float = 0.75):
    if turbo_path is None:
        yield
        return
    transformer, _ = pipeline.get_trans_lora()
    snapshot = _snapshot_loras(transformer)
    cache = transformer.cache
    try:
        _load_loras(transformer, [turbo_path], [float(multiplier)], model_type)
        transformer.cache = None
        yield
    finally:
        transformer.cache = cache
        _restore_loras(transformer, snapshot, model_type)


class H3FaceRefinerRuntime(PrivateRuntime):
    def __init__(self):
        super().__init__(RUNTIME_NAME)
        self.shared_offloadobj = None
        self.profile = None

    def load(self, shared_context=None) -> None:
        import wgp

        profile = 0
        shared_offloadobj = None if shared_context is None else shared_context.offloadobj
        if self.model is not None and self.profile == profile and self.shared_offloadobj is shared_offloadobj:
            return
        self.release()
        if not torch.cuda.is_available():
            raise RuntimeError("H3 Face Refiner requires CUDA")

        try:
            model_kwargs = {"disable_pinning": True}
            if shared_context is not None:
                model_kwargs.update(shared_h3_pipeline=shared_context.model, shared_h3_offloadobj=shared_context.offloadobj)
            tqdm.write(f"[H3FaceRefine] Loading private H3 runtime with postprocessor memory profile {profile}; GGUF Q4_K_M text encoder; RAM pinning disabled")
            self.model, self.offloadobj = wgp.load_models(DEFAULT_MODEL_TYPE, override_profile=profile, output_type="video", config_id=TEXT_ENCODER_CONFIG, track_as_main=False, extension=True, **model_kwargs)
            self.shared_offloadobj = shared_offloadobj
            if shared_context is not None:
                self.model.set_offload_handoff(shared_context.offloadobj, self.offloadobj)
                print(f"[H3FaceRefine] Reusing text encoder, video VAE, audio VAE, and latent upscaler from loaded H3 model '{shared_context.model_type}'; loading only pruned Ref2VA denoiser")
            self.profile = profile
            self.adopt(self.model, self.offloadobj)
        except Exception:
            self.release()
            raise

    def _before_release(self):
        if self.shared_offloadobj is not None:
            self.model.detach_shared_components()

    def release(self) -> None:
        if self.shared_offloadobj is not None:
            self.shared_offloadobj.unload_all()
        super().release()
        self.shared_offloadobj = self.profile = None

    def unload_all(self) -> None:
        super().unload_all()
        if self.shared_offloadobj is not None:
            self.shared_offloadobj.unload_all()


RUNTIME = H3FaceRefinerRuntime()


@torch.inference_mode()
def refine_video(video: torch.Tensor | None, strengths: torch.Tensor, *, pipeline, model_type: str, turbo_path: str | None, turbo_multiplier: float,
                 prompt: str, denoising_strength: float, sampling_steps: int, shift: float, sample_solver: str,
                 seed: int, fps: float, window_size: int, window_overlap: int, frame_offset: int = 0,
                 audio_waveform=None, audio_sample_rate: int = 0, source_audio_path: str | None = None,
                 reference_images=None, image_refs_relative_size: float = 100.0,
                 vae_tile_size=None, abort_callback=None, progress_callback=None, frame_count=None, video_window_loader=None,
                 window_index_offset: int = 0, window_count_total: int | None = None):
    frame_count = int(video.shape[1]) if video_window_loader is None else int(frame_count)
    starts = window_starts(frame_count, window_size, window_overlap)
    output = None
    filled_stop = 0
    with turbo_lora(pipeline, turbo_path, model_type, turbo_multiplier):
        for window_index, start in enumerate(tqdm(starts, desc="H3 face refinement windows", unit="window"), start=1):
            if callable(abort_callback) and abort_callback():
                return None
            stop = min(start + window_size, frame_count)
            input_window = video[:, start:stop] if video_window_loader is None else video_window_loader(start, stop)
            if input_window.dtype == torch.uint8:
                input_window = _uint8_to_signed(input_window)
            input_window, strength_window = pad_window(input_window, strengths[start:stop])
            audio_window, window_audio_rate = slice_audio_for_window(audio_waveform, audio_sample_rate, source_audio_path, frame_offset + start, stop - start, int(input_window.shape[1]), fps)
            total_windows = len(starts) if window_count_total is None else int(window_count_total)
            label = f"Window {int(window_index_offset) + window_index}/{total_windows}" if total_windows > 1 else ""

            def status_callback(phase):
                _report(progress_callback, f"{label} - {phase}" if label else phase)

            step_callback = model_step_callback(pipeline, abort_callback, progress_callback, "H3 Face Refinement", sampling_steps, label)

            if hasattr(pipeline, "_interrupt"):
                pipeline._interrupt = False
            try:
                window_output = pipeline.refine_video(input_window, prompt=prompt, strengths=strength_window,
                                                      denoising_strength=denoising_strength, sampling_steps=sampling_steps,
                                                      shift=shift, sample_solver=sample_solver,
                                                      seed=int(seed) + int(frame_offset) + start, fps=fps,
                                                      reference_images=reference_images,
                                                      image_refs_relative_size=image_refs_relative_size,
                                                      VAE_tile_size=vae_tile_size, audio_waveform=audio_window,
                                                      audio_sample_rate=window_audio_rate, callback=step_callback,
                                                      set_progress_status=status_callback)
            finally:
                if hasattr(pipeline, "_interrupt"):
                    pipeline._interrupt = False
            input_window = strength_window = audio_window = None
            if window_output is None:
                return None
            window_output = window_output[:, :stop - start].cpu()
            if output is None:
                output = torch.empty((window_output.shape[0], frame_count, *window_output.shape[2:]), dtype=torch.uint8, device="cpu")
                output[:, :stop].copy_(_signed_to_uint8(window_output))
            else:
                overlap = min(filled_stop - start, int(window_output.shape[1]))
                if overlap:
                    previous = _uint8_to_signed(output[:, start:start + overlap])
                    output[:, start:start + overlap].copy_(_signed_to_uint8(crossfade_frames(previous, window_output[:, :overlap])))
                output[:, start + overlap:stop].copy_(_signed_to_uint8(window_output[:, overlap:]))
            filled_stop = max(filled_stop, stop)
    _report(progress_callback, "H3 face refinement complete")
    return output


def load_model(shared_context=None) -> None:
    RUNTIME.load(shared_context)


def release_model() -> None:
    RUNTIME.release()


__all__ = ["DEFAULT_MODEL_TYPE", "DEFAULT_WINDOW_FRAMES", "DEFAULT_WINDOW_OVERLAP", "MAX_WINDOW_FRAMES", "RUNTIME", "RUNTIME_NAME",
           "TEMPORAL_STRIDE", "load_model", "refine_video", "release_model", "window_starts"]
