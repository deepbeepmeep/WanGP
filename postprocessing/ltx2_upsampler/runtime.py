from __future__ import annotations

import gc

import numpy as np
import torch
from tqdm import tqdm

from postprocessing.lanczos.wgp_bridge import resize_lanczos_spatial
from shared.utils import offload_registry
from shared.utils.audio_video import slice_audio_window


RUNTIME_NAME = "LTX-2 Video Upsampler"
DEFAULT_WINDOW_FRAMES = 81
MAX_WINDOW_FRAMES = 481
WINDOW_OVERLAP_FRAMES = 17
TEMPORAL_STRIDE = 8
SPATIAL_BLOCK_SIZE = 32
REFINE_METHOD = "ltx25_refine"
MODEL_TYPES = {"ltx23": "ltx2_22B", "ltx25": "ltx2_25_22B_distilled", REFINE_METHOD: "ltx2_25_22B_distilled"}
FAKE_MODEL_TYPES = {"ltx23": "ltx2_upsampler_23", "ltx25": "ltx2_upsampler_25", REFINE_METHOD: "ltx2_upsampler_25"}
LORA_KEYS = {"ltx23": ("ltx2_lora_distilled_1_1", "ltx2_lora_pixel_spatial_upscaler"), "ltx25": ("ltx2_lora_pixel_spatial_upscaler",), REFINE_METHOD: ("ltx2_lora_refine_details",)}
LORA_MULTIPLIERS = {"ltx23": (0.5, 1.0), "ltx25": (1.0,), REFINE_METHOD: (1.0,)}
AUDIO_SAMPLE_RATE = 16000
# Refine Details IC-LoRA: trained on 1024x576 (or 576x1024) crops of 97 frames.
REFINE_TILE_SIZE = (576, 1024)
REFINE_DEFAULT_WINDOW_FRAMES = 97
REFINE_MIN_WINDOW_FRAMES = 49
REFINE_MAX_WINDOW_FRAMES = 121
REFINE_TAIL_PADDING_FRAMES = 8
REFINE_MULTIPLIERS = (1.0, 1.5, 2.0)
REFINE_MAX_OUTPUT_RESOLUTION = 4320
REFINE_LANCZOS_FRAMES = 16
REFINE_MAX_TILES_PER_CALL = 16
REFINE_CARRY_KEY = "ltx25_refine_carry_latents"
REFINE_FUSION_KEY = "ltx25_refine_fusion_x0"
REFINE_CARRY_START_KEY = "ltx25_refine_carry_start_frame"
REFINE_RESUME_KEY = "ltx25_refine_resume"
# Media Flow chunk continuity (evaluation switch): "conditioning" starts each chunk from the re-encoded previous refined
# frames; "fusion" blends each chunk's first overlap latents with the previous chunk's per-step predictions (no clear gain
# over conditioning on dense texture, 1.57x vs 1.24x compute); "crossfade" refines chunks independently and crossfades them.
REFINE_MEDIA_FLOW_CHUNK_MODE = "conditioning"
# Conditioning only: True crossfades the previous chunk's overlap frames into the new chunk's; False keeps the previous
# chunk's frames and switches at the end of the overlap.
REFINE_MEDIA_FLOW_CONDITIONING_CROSSFADE = False
# True: like ComfyUI, stretch the source to the nearest multiple of 32 and output that size (1920x1080 -> 1920x1088).
# False: pad to the next multiple of 32 with replicated edges and crop back to the exact requested size.
REFINE_STRETCH_TO_CANVAS = False
DEFAULT_REFINE_PROMPT = "sharp photographic detail, crisp natural texture, fine surface detail, clean edges, natural film grain, high resolution footage"


def lora_urls(wgp, variant: str) -> tuple[str, ...]:
    model_def = wgp.get_model_def(MODEL_TYPES[variant])
    return tuple(model_def[key] for key in LORA_KEYS[variant])


def _report(progress_callback, phase: str, current: int | None = None, total: int | None = None, unit: str | None = None):
    if callable(progress_callback):
        progress_callback(phase, current, total, *([unit] if unit else []))


def window_starts(frame_count: int, window_size: int = DEFAULT_WINDOW_FRAMES, window_overlap: int = WINDOW_OVERLAP_FRAMES) -> tuple[int, ...]:
    if window_size > MAX_WINDOW_FRAMES:
        raise ValueError(f"LTX upsampler window size cannot exceed {MAX_WINDOW_FRAMES} frames")
    if window_overlap >= window_size:
        raise ValueError("LTX upsampler window overlap must be smaller than its window size")
    if frame_count <= window_size:
        return (0,)
    return tuple(range(0, frame_count - window_overlap, window_size - window_overlap))


def supported_spatial_size(size: int) -> int:
    return max(SPATIAL_BLOCK_SIZE, round(int(size) / SPATIAL_BLOCK_SIZE) * SPATIAL_BLOCK_SIZE)


def refine_output_size(height: int, width: int, scale: float) -> tuple[int, int]:
    return max(2, round(height * scale / 2) * 2), max(2, round(width * scale / 2) * 2)


def refine_canvas_size(size: int) -> int:
    return -(-int(size) // SPATIAL_BLOCK_SIZE) * SPATIAL_BLOCK_SIZE


def refine_stretched_size(size: int) -> int:
    """ComfyUI sizing: nearest multiple of 32, ties round up."""
    return max(SPATIAL_BLOCK_SIZE, int(size / SPATIAL_BLOCK_SIZE + 0.5) * SPATIAL_BLOCK_SIZE)


def refine_padded_frames(frame_count: int) -> int:
    """Pad by 8 frames (the last frames of an extent lose detail) up to the LTX 8n+1 cadence."""
    return (int(frame_count) + REFINE_TAIL_PADDING_FRAMES - 1 + TEMPORAL_STRIDE - 1) // TEMPORAL_STRIDE * TEMPORAL_STRIDE + 1


def refine_window_frames(padded_frames: int, window_frames: int) -> int:
    """A clip slightly longer than the window stays one extent (up to 121 frames), as in ComfyUI."""
    return padded_frames if padded_frames <= max(window_frames, min(window_frames + 3 * TEMPORAL_STRIDE, REFINE_MAX_WINDOW_FRAMES)) else window_frames


def refine_overlap_frames(window_overlap: int, chunk_frames: int) -> int:
    """Media Flow chunk overlap, also the number of refined frames carried to the next chunk: 8n+1, at most half the chunk."""
    return max(0, min(int(window_overlap), (int(chunk_frames) - 2) // (2 * TEMPORAL_STRIDE) * TEMPORAL_STRIDE + 1, int(chunk_frames) - 1))


def refine_fusion_overlap_latents(window_overlap: int, chunk_frames: int) -> int:
    """Latents shared by consecutive fusion chunks (8 frames each). The output switches chunks mid-overlap, so the overlap
    setting counts the frames on each side of the switch (17 -> 4 latents); at least 1, at most half the chunk."""
    return max(1, min(2 * ((int(window_overlap) - 1) // TEMPORAL_STRIDE), int(chunk_frames) // (2 * TEMPORAL_STRIDE)))


def refine_scale_error(source_height: int, source_width: int, scale: float) -> str:
    """Outputs are limited to 8K; the error names the largest multiplier that still fits."""
    if scale * min(source_height, source_width) <= REFINE_MAX_OUTPUT_RESOLUTION:
        return ""
    fitting = [value for value in REFINE_MULTIPLIERS if value * min(source_height, source_width) <= REFINE_MAX_OUTPUT_RESOLUTION]
    suggestion = f"select x{max(fitting):g} or lower" if fitting else "this source is too large"
    return f"LTX 2.5 Detail Refiner outputs at most {REFINE_MAX_OUTPUT_RESOLUTION}p: for {source_width}x{source_height}, {suggestion}."


def pad_window(video: torch.Tensor) -> torch.Tensor:
    frame_count = int(video.shape[1])
    padded_frame_count = max(TEMPORAL_STRIDE + 1, ((frame_count - 1 + TEMPORAL_STRIDE - 1) // TEMPORAL_STRIDE) * TEMPORAL_STRIDE + 1)
    if padded_frame_count == frame_count:
        return video
    return torch.cat((video, video[:, -1:].expand(-1, padded_frame_count - frame_count, -1, -1)), dim=1)


def crossfade_frames(previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    weights = torch.linspace(0.0, torch.pi, previous.shape[1], device=previous.device, dtype=torch.float32).cos_().mul_(-0.5).add_(0.5).view(1, -1, 1, 1)
    return previous.float().lerp_(current.float(), weights).round_().to(previous.dtype)


def slice_audio_for_window(audio_waveform, audio_sample_rate: int, source_audio_path: str | None, start: int, frame_count: int, padded_frame_count: int, fps: float):
    if source_audio_path:
        audio, sample_rate = slice_audio_window(source_audio_path, start, frame_count, fps)
    else:
        sample_rate = int(audio_sample_rate or AUDIO_SAMPLE_RATE)
        if audio_waveform is None:
            audio = np.zeros((int(round(frame_count * sample_rate / fps)), 1), dtype=np.float32)
        else:
            audio = audio_waveform.detach().cpu().numpy() if torch.is_tensor(audio_waveform) else np.asarray(audio_waveform)
            if audio.ndim == 1:
                audio = audio[:, None]
            start_sample = int(round(start * sample_rate / fps))
            stop_sample = start_sample + int(round(frame_count * sample_rate / fps))
            audio = audio[start_sample:stop_sample]
    target_samples = int(round(padded_frame_count * sample_rate / fps))
    if audio.shape[0] < target_samples:
        audio = np.pad(audio, ((0, target_samples - audio.shape[0]), (0, 0)))
    return np.asarray(audio[:target_samples], dtype=np.float32), sample_rate


class LTXUpsamplerRuntime:
    def __init__(self):
        self.variant = None
        self.model = None
        self.offloadobj = None

    def load(self, variant: str) -> None:
        if self.model is not None and self.variant == variant:
            return
        if variant not in MODEL_TYPES:
            raise ValueError(f"Unknown LTX upsampler variant: {variant}")
        import wgp

        if self.model is not None and FAKE_MODEL_TYPES[self.variant] == FAKE_MODEL_TYPES[variant]:
            self.variant = variant
            try:
                self._load_loras(wgp, variant)
            except Exception:
                self.release()
                raise
            return
        self.release()
        if not torch.cuda.is_available():
            raise RuntimeError("LTX video upsampling requires CUDA")
        self.variant = variant
        try:
            self.model, self.offloadobj = wgp.load_models(
                MODEL_TYPES[variant],
                override_profile=0,
                output_type="video",
                runtime_model_type=FAKE_MODEL_TYPES[variant],
                track_as_main=False,
                extension=True,
            )
            self._load_loras(wgp, variant)
            offload_registry.register_offloadobj(RUNTIME_NAME, self.offloadobj, self.release)
        except Exception:
            self.release()
            raise

    def _load_loras(self, wgp, variant: str) -> None:
        from mmgp import offload

        lora_dir = wgp.get_lora_dir(MODEL_TYPES[variant])
        lora_paths = [wgp.get_lora_local_path(lora_dir, url) for url in lora_urls(wgp, variant)]
        transformer, _ = self.model.get_trans_lora()
        offload.load_loras_into_model(
            transformer,
            lora_paths,
            LORA_MULTIPLIERS[variant],
            activate_all_loras=True,
            preprocess_sd=wgp.get_loras_preprocessor(transformer, FAKE_MODEL_TYPES[variant]),
            pinnedLora=False,
            maxReservedLoras=wgp.server_config.get("max_reserved_loras", -1),
            split_linear_modules_map=getattr(transformer, "split_linear_modules_map", None),
        )
        if transformer._loras_errors:
            raise RuntimeError(f"Error while loading LTX {variant[3:]} upsampler LoRAs: " + ", ".join(message for _, message in transformer._loras_errors))

    def release(self) -> None:
        if self.offloadobj is not None:
            offload_registry.unregister_offloadobj(RUNTIME_NAME, self.offloadobj)
            self.offloadobj.release()
        self.variant = self.model = self.offloadobj = None
        gc.collect()

    def vae_tile_size(self, vae_config: int, output_height: int, output_width: int):
        device_memory = torch.cuda.get_device_properties(0).total_memory / 1048576
        mixed_precision = self.model.VAE_dtype == torch.float32
        return self.model.vae.get_VAE_tile_size(vae_config, device_memory, mixed_precision, output_height=output_height, output_width=output_width)

    @torch.inference_mode()
    def upscale(
        self,
        sample: torch.Tensor,
        *,
        prompt: str,
        negative_prompt: str,
        seed: int,
        fps: float,
        window_size: int = DEFAULT_WINDOW_FRAMES,
        window_overlap: int = WINDOW_OVERLAP_FRAMES,
        frame_offset: int = 0,
        audio_waveform=None,
        audio_sample_rate: int = 0,
        source_audio_path: str | None = None,
        vae_tile_size=None,
        abort_callback=None,
        progress_callback=None,
    ):
        if self.model is None:
            raise RuntimeError("LTX video upsampler model is not loaded")
        frame_count = int(sample.shape[1])
        target_height, target_width = supported_spatial_size(sample.shape[-2]), supported_spatial_size(sample.shape[-1])
        starts = window_starts(frame_count, window_size, window_overlap)
        output = None
        for window_index, start in enumerate(tqdm(starts, desc=f"LTX {self.variant[3:]} upsampler windows", unit="window"), start=1):
            if callable(abort_callback) and abort_callback():
                return None, None
            stop = min(start + window_size, frame_count)
            input_window = pad_window(sample[:, start:stop])
            if input_window.shape[-2:] != (target_height, target_width):
                input_window = torch.nn.functional.interpolate(input_window.permute(1, 0, 2, 3), size=(target_height, target_width), mode="bilinear", align_corners=False, antialias=True).permute(1, 0, 2, 3).contiguous()
            audio_window, window_audio_sample_rate = slice_audio_for_window(audio_waveform, audio_sample_rate, source_audio_path, start, stop - start, int(input_window.shape[1]), fps)
            window_label = f"Window {window_index} / {len(starts)}" if len(starts) > 1 else ""

            def status_callback(phase, label=window_label):
                _report(progress_callback, f"{label} - {phase}" if label else phase)

            step_count = 8

            def step_callback(step_idx=-1, _latent=None, _force_refresh=False, read_state=False, **kwargs):
                if callable(abort_callback) and abort_callback():
                    self.model._interrupt = True
                    return
                if read_state:
                    return
                phase = kwargs.get("progress_title", "Distilled Refinement")
                phase = f"{window_label} - {phase}" if window_label else phase
                _report(progress_callback, phase, int(step_idx) + 1, kwargs.get("override_num_inference_steps", step_count))

            step_callback.checkpoint = lambda: step_callback(read_state=True)
            self.model._interrupt = False
            window_output = self.model.upscale_video(
                input_window,
                prompt=prompt,
                negative_prompt=negative_prompt,
                audio_waveform=audio_window,
                audio_sample_rate=window_audio_sample_rate,
                upsampler_variant=self.variant,
                seed=seed,
                fps=fps,
                pixel_frame_offset=int(frame_offset) + start,
                VAE_tile_size=vae_tile_size,
                callback=step_callback,
                set_progress_status=status_callback,
                interrupt_check=abort_callback,
            )
            input_window = audio_window = None
            if window_output is None:
                return None, None
            window_output = window_output[:, : stop - start]
            if output is None:
                output = window_output.new_empty(window_output.shape[0], frame_count, *window_output.shape[2:])
                output[:, :stop].copy_(window_output)
            else:
                overlap = min(window_overlap, int(window_output.shape[1]))
                output[:, start:start + overlap].copy_(crossfade_frames(output[:, start:start + overlap], window_output[:, :overlap]))
                output[:, start + overlap:stop].copy_(window_output[:, overlap:])
            window_output = None
        _report(progress_callback, f"LTX {self.variant[3:]} upsampling complete")
        return output, None


    @torch.inference_mode()
    def refine(
        self,
        sample: torch.Tensor,
        *,
        scale: float,
        prompt: str,
        seed: int,
        fps: float,
        window_frames: int,
        window_overlap: int,
        gentle: bool,
        vae_config: int,
        tiles_per_call: int = 0,
        frame_offset: int = 0,
        continue_cache=None,
        return_continue_cache: bool = False,
        abort_callback=None,
        progress_callback=None,
    ):
        if self.model is None:
            raise RuntimeError("LTX video upsampler model is not loaded")
        _, frame_count, source_height, source_width = map(int, sample.shape)
        error = refine_scale_error(source_height, source_width, scale)
        if error:
            raise ValueError(error)
        output_height, output_width = refine_output_size(source_height, source_width, scale)
        if REFINE_STRETCH_TO_CANVAS:
            output_height, output_width = refine_stretched_size(output_height), refine_stretched_size(output_width)
        height, width = refine_canvas_size(output_height), refine_canvas_size(output_width)
        tile_height, tile_width = REFINE_TILE_SIZE[::-1] if height > width else REFINE_TILE_SIZE
        vae_tile_size = self.vae_tile_size(vae_config, height, width)
        extent = MAX_WINDOW_FRAMES - REFINE_TAIL_PADDING_FRAMES
        starts = window_starts(frame_count, extent, window_overlap) if frame_count > extent else (0,)
        mode = REFINE_MEDIA_FLOW_CHUNK_MODE if return_continue_cache else ""
        # Fusion chunks start with a copy of their first frame (dropped after decoding): every real frame then sits on
        # the 8-frame latent grid shared by consecutive chunks, whether the run is fresh or resumed.
        lead = int(mode == "fusion")
        fusion_previous = continue_cache[REFINE_FUSION_KEY] if continue_cache is not None and mode == "fusion" else None
        # Fusion exports its last 2n latents: the next chunk overlaps the last n, a resumed run (which restarts n latents
        # before the last written frame) all 2n.
        fusion_latents = refine_fusion_overlap_latents(window_overlap, frame_count) if mode == "fusion" else 0
        # Overlap conditioning: the next chunk starts carry_frames (8n+1) frames before this one ends; those refined
        # frames are re-encoded as a fresh clip that conditions its start.
        conditioning_latent = continue_cache[REFINE_CARRY_KEY] if continue_cache is not None and mode == "conditioning" else None
        carry_frames = refine_overlap_frames(window_overlap, frame_count) if mode == "conditioning" and len(starts) == 1 else 0
        output = carry = None
        for window_index, start in enumerate(starts, start=1):
            if callable(abort_callback) and abort_callback():
                return None, None
            stop = min(start + extent, frame_count)
            source = sample[:, start:stop]
            source_frames = stop - start
            padded_frames = refine_padded_frames(source_frames + lead)
            last_latent = (source_frames + lead - 1) // TEMPORAL_STRIDE
            window_label = f"Window {window_index} / {len(starts)}" if len(starts) > 1 else ""

            def load_window(first: int, count: int) -> torch.Tensor:
                pixels = torch.empty((1, 3, count, height, width), dtype=torch.uint8, device="cpu")
                for chunk_start in range(0, count, REFINE_LANCZOS_FRAMES):
                    chunk_count = min(REFINE_LANCZOS_FRAMES, count - chunk_start)
                    indices = torch.arange(first + chunk_start - lead, first + chunk_start + chunk_count - lead, device=source.device).clamp_(0, source_frames - 1)
                    frames = source[:, indices]
                    if frames.dtype != torch.uint8:
                        frames = frames.float().add_(1.0).mul_(127.5).clamp_(0.0, 255.0).round_().to(torch.uint8)
                    if frames.shape[-2:] != (output_height, output_width):
                        frames = resize_lanczos_spatial(frames, 1.0, size=(output_height, output_width))
                    target = pixels[0, :, chunk_start : chunk_start + chunk_count]
                    target[:, :, :output_height, :output_width] = frames
                    frames = None
                    target[:, :, output_height:, :output_width] = target[:, :, output_height - 1 : output_height, :output_width]
                    target[:, :, :, output_width:] = target[:, :, :, output_width - 1 : output_width]
                return pixels

            def status_callback(phase, label=window_label):
                _report(progress_callback, f"{label} - {phase}" if label else phase)

            def step_callback(step_idx=-1, _latent=None, _force_refresh=False, read_state=False, label=window_label, **kwargs):
                if callable(abort_callback) and abort_callback():
                    self.model._interrupt = True
                    return
                if read_state:
                    return
                phase = kwargs.get("progress_title", "Detail Refinement")
                _report(progress_callback, f"{label} - {phase}" if label else phase, int(step_idx) + 1, kwargs.get("override_num_inference_steps"), kwargs.get("progress_unit"))

            step_callback.checkpoint = lambda callback=step_callback: callback(read_state=True)
            self.model._interrupt = False
            refined = self.model.refine_video(
                load_window,
                padded_frames,
                height,
                width,
                source_frames + lead,
                output_height,
                output_width,
                prompt=prompt,
                seed=seed,
                fps=fps,
                window_frames=refine_window_frames(padded_frames, window_frames),
                tile_height=tile_height,
                tile_width=tile_width,
                gentle=gentle,
                tiles_per_call=tiles_per_call,
                conditioning_latent=conditioning_latent if window_index == 1 else None,
                carry_frames=carry_frames,
                fusion_previous=fusion_previous if window_index == 1 else None,
                fusion_export=slice(last_latent - 2 * fusion_latents + 1, last_latent + 1) if fusion_latents else None,
                pixel_frame_offset=int(frame_offset) + start,
                VAE_tile_size=vae_tile_size,
                callback=step_callback,
                set_progress_status=status_callback,
                interrupt_check=abort_callback,
            )
            source = None
            if refined is None:
                return None, None
            window_output, carry_latents = refined
            window_output = window_output[:, lead:]
            refined = None
            if carry_latents is not None and mode == "fusion":
                next_start = int(frame_offset) + frame_count - fusion_latents * TEMPORAL_STRIDE
                carry = {REFINE_FUSION_KEY: carry_latents[:, fusion_latents:], REFINE_CARRY_START_KEY: next_start, REFINE_RESUME_KEY: {REFINE_FUSION_KEY: carry_latents, REFINE_CARRY_START_KEY: next_start - fusion_latents * TEMPORAL_STRIDE}}
            elif carry_latents is not None:
                # The next chunk starts at the carried frames; a resumed run restarts carry_frames before the
                # last written frame, so it needs the frames preceding the (unwritten) tail.
                next_start = int(frame_offset) + frame_count - carry_frames
                carry = {REFINE_CARRY_KEY: carry_latents[0], REFINE_CARRY_START_KEY: next_start, REFINE_RESUME_KEY: {REFINE_CARRY_KEY: carry_latents[1], REFINE_CARRY_START_KEY: next_start - carry_frames}}
            if len(starts) == 1:
                output = window_output
            elif output is None:
                output = window_output.new_empty(window_output.shape[0], frame_count, *window_output.shape[2:])
                output[:, :stop].copy_(window_output)
            else:
                overlap = min(window_overlap, int(window_output.shape[1]))
                output[:, start:start + overlap].copy_(crossfade_frames(output[:, start:start + overlap], window_output[:, :overlap]))
                output[:, start + overlap:stop].copy_(window_output[:, overlap:])
            window_output = None
        _report(progress_callback, "LTX 2.5 Detail Refinement Complete")
        return output, carry


RUNTIME = LTXUpsamplerRuntime()


def load_model(variant: str) -> None:
    RUNTIME.load(variant)


def upscale_video(sample: torch.Tensor, **kwargs):
    return RUNTIME.upscale(sample, **kwargs)


def refine_video(sample: torch.Tensor, **kwargs):
    return RUNTIME.refine(sample, **kwargs)


def release_model() -> None:
    RUNTIME.release()
