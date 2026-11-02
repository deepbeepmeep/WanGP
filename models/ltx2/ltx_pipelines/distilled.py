from shared.utils.phase_progress import control_video_encoding
from shared.utils.tiled_fusion import TiledFusionPlan
import logging
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import replace

import torch

from ..ltx_core.components.diffusion_steps import EulerAncestralDiffusionStep, EulerCFGPPDiffusionStep, EulerDiffusionStep
from ..ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ..ltx_core.components.noisers import GaussianNoiser
from ..ltx_core.components.protocols import DiffusionStepProtocol
from ..ltx_core.conditioning import AudioConditionByLatent, AudioConditionByReferenceLatent
from ..ltx_core.loader import LoraPathStrengthAndSDOps
from ..ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ..ltx_core.model.upsampler import upsample_video
from ..ltx_core.model.video_vae import TilingConfig, encode_video as vae_encode_video, get_video_chunks_number
from ..ltx_core.model.video_vae import decode_video_to_tensor as vae_decode_video_to_tensor
from ..ltx_core.text_encoders.gemma import encode_text, postprocess_text_embeddings, resolve_text_connectors
from ..ltx_core.tools import VideoLatentTools
from ..ltx_core.types import AudioLatentShape, LatentState, VideoPixelShape
from shared.prompt_relay import encode_prompt_relay
from .utils import ModelLedger
from .utils.args import default_2_stage_distilled_arg_parser
from .utils.constants import (
    AUDIO_SAMPLE_RATE,
    DEFAULT_NEGATIVE_PROMPT,
    DISTILLED_8_STEPS_STAGE_2_SIGMA_VALUES,
    DISTILLED_SIGMA_VALUES,
    LTX23_USE_DISTILLED_8_STEPS_STAGE_2_SIGMAS,
    STAGE_2_DISTILLED_SIGMA_VALUES,
)
from .utils.spatial_tiling import spatially_tiled_denoising_func
from .utils.helpers import (
    assert_resolution,
    bind_interrupt_check,
    cleanup_memory,
    denoise_audio_video,
    euler_denoising_loop,
    generate_enhanced_prompt,
    get_device,
    image_conditionings_by_adding_guiding_latent,
    image_conditionings_by_replacing_latent,
    latent_conditionings_by_latent_sequence,
    multi_modal_guider_denoising_func,
    noise_video_state,
    paired_reference_conditionings_by_latents,
    prepare_mask_injection,
    simple_denoising_func,
    video_conditionings_by_frozen_video,
    video_conditionings_by_control_video,
)
from .utils.tiled_fusion import tiled_fusion_denoising_func
from .utils.types import PipelineComponents
from ..editanything import build_editanything_reference_conditioning
from shared.utils.loras_mutipliers import update_loras_slists
from shared.utils.self_refiner import create_self_refiner_handler, normalize_self_refiner_plan
from shared.utils.text_encoder_cache import TextEncoderCache

device = get_device()
_BENCH_TRANSFORMER_ENV = "WAN2GP_LTX2_BENCH_TRANSFORMER"
LTX25_UPSAMPLER_INJECT_FIRST_FRAME = False
# Tiles evaluated per transformer call (joint pass): each block's weights are streamed once per group.
# The group only adds one hidden state per tile, bounded by this share of free VRAM; outputs are identical.
REFINE_TILE_BATCH_MAX_BYTES = 2 << 30
REFINE_TILE_BATCH_FREE_RATIO = 0.25
# Canvas, blend buffer and window references stay in VRAM only while they need less than this share of free VRAM.
REFINE_CANVAS_FREE_RATIO = 0.1
# Carried Media Flow overlap latents are VAE round trips (softer): half strength lets the model refine them
# again, which removes the detail overshoot after the join while keeping chunks continuous.
REFINE_CARRY_STRENGTH = 0.5


def _pad_frames_to_canvas(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """uint8 ``(F, h, w, 3)`` frames to ``(1, 3, F, height, width)`` with replicated bottom / right edges."""
    count, frame_height, frame_width, _ = frames.shape
    canvas = frames.new_empty((1, 3, count, height, width))
    canvas[0, :, :, :frame_height, :frame_width] = frames.permute(3, 0, 1, 2)
    canvas[0, :, :, frame_height:, :frame_width] = canvas[0, :, :, frame_height - 1 : frame_height, :frame_width]
    canvas[0, :, :, :, frame_width:] = canvas[0, :, :, :, frame_width - 1 : frame_width]
    return canvas


class _FixedNoiseNoiser:
    def __init__(self, noise: torch.Tensor):
        self.noise = noise

    def __call__(self, latent_state: LatentState, noise_scale: float = 1.0) -> LatentState:
        noise = torch.zeros_like(latent_state.latent) if self.noise is None else self.noise.to(device=latent_state.latent.device, dtype=latent_state.latent.dtype)
        self.noise = None
        reference_tokens = int(latent_state.latent.shape[1] - noise.shape[1])
        if reference_tokens:
            noise = torch.cat((noise, noise.new_zeros(noise.shape[0], reference_tokens, noise.shape[2])), dim=1)
        scaled_mask = latent_state.denoise_mask * noise_scale
        latent = noise * scaled_mask + latent_state.latent * (1 - scaled_mask)
        return replace(latent_state, latent=latent.to(latent_state.latent.dtype))


def _indexed_noise_tokens(shape: torch.Size, seed: int, latent_frame_offset: int, dtype: torch.dtype) -> torch.Tensor:
    batch, channels, frames, height, width = map(int, shape)
    noise_frames = []
    modulus = 2**63 - 1
    for frame_no in range(frames):
        global_frame_no = int(latent_frame_offset) + frame_no
        frame_seed = (int(seed) + 0x9E3779B97F4A7C15 * (global_frame_no + 1)) % modulus
        generator = torch.Generator(device="cpu").manual_seed(frame_seed)
        noise_frames.append(torch.randn((batch, channels, height, width), generator=generator, device="cpu", dtype=torch.float32).to(dtype))
    noise = torch.stack(noise_frames, dim=2)
    return noise.permute(0, 2, 3, 4, 1).reshape(batch, -1, channels)


def _env_flag(name: str, default: str = "0") -> bool:
    val = os.environ.get(name, default)
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def _align_seq_len(tensor: torch.Tensor | None, target_len: int) -> torch.Tensor | None:
    if tensor is None:
        return tensor
    seq_dim = 0 if tensor.dim() == 2 else 1
    cur_len = tensor.shape[seq_dim]
    if cur_len == target_len:
        return tensor
    if cur_len < target_len:
        pad_len = target_len - cur_len
        if seq_dim == 0:
            pad = tensor[-1:].repeat(pad_len, 1)
            return torch.cat([tensor, pad], dim=0)
        pad = tensor[:, -1:, :].repeat(1, pad_len, 1)
        return torch.cat([tensor, pad], dim=1)
    return tensor.narrow(seq_dim, 0, target_len)


class _TransformerBenchWrapper:
    def __init__(self, module, enabled: bool = False) -> None:
        self._module = module
        self._enabled = bool(enabled)
        self._cuda_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._cpu_total_ms = 0.0
        self._cpu_calls = 0

    def __getattr__(self, name):
        return getattr(self._module, name)

    def __call__(self, *args, **kwargs):
        if not self._enabled:
            return self._module(*args, **kwargs)
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            out = self._module(*args, **kwargs)
            end.record()
            self._cuda_events.append((start, end))
            return out

        t0 = time.perf_counter()
        out = self._module(*args, **kwargs)
        self._cpu_total_ms += (time.perf_counter() - t0) * 1000.0
        self._cpu_calls += 1
        return out

    def consume(self) -> tuple[float, int]:
        if not self._enabled:
            return 0.0, 0
        if torch.cuda.is_available():
            if not self._cuda_events:
                return 0.0, 0
            torch.cuda.synchronize()
            total_ms = 0.0
            for start, end in self._cuda_events:
                total_ms += float(start.elapsed_time(end))
            calls = len(self._cuda_events)
            self._cuda_events.clear()
            return total_ms, calls

        total_ms = self._cpu_total_ms
        calls = self._cpu_calls
        self._cpu_total_ms = 0.0
        self._cpu_calls = 0
        return total_ms, calls


class DistilledPipeline:
    """
    Two-stage distilled video generation pipeline.
    Stage 1 generates video at half resolution, then Stage 2 performs the
    distilled x2 upscale/refine pass to reach the requested output resolution.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        gemma_root: str | None = None,
        spatial_upsampler_path: str | None = None,
        loras: list[LoraPathStrengthAndSDOps] | None = None,
        device: torch.device = device,
        fp8transformer: bool = False,
        model_device: torch.device | None = None,
        models: object | None = None,
    ):
        self.device = device
        self.dtype = torch.bfloat16
        self.models = models

        if self.models is None:
            if checkpoint_path is None or gemma_root is None or spatial_upsampler_path is None:
                raise ValueError("checkpoint_path, gemma_root, and spatial_upsampler_path are required.")
            self.model_ledger = ModelLedger(
                dtype=self.dtype,
                device=model_device or device,
                checkpoint_path=checkpoint_path,
                spatial_upsampler_path=spatial_upsampler_path,
                gemma_root_path=gemma_root,
                loras=loras or [],
                fp8transformer=fp8transformer,
            )
        else:
            self.model_ledger = None

        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )
        self.text_encoder_cache = TextEncoderCache()
        self._joyai_echo = None

    def _get_model(self, name: str):
        if self.models is not None:
            return getattr(self.models, name)
        if self.model_ledger is None:
            raise ValueError(f"Missing model source for '{name}'.")
        return getattr(self.model_ledger, name)()

    @contextmanager
    def joyai_echo_context(self, joyai_echo):
        previous = self._joyai_echo
        self._joyai_echo = joyai_echo
        try:
            yield
        finally:
            self._joyai_echo = previous

    def _joyai_echo_reference_conditionings(self, joyai_echo, suffix: str, dtype):
        paired_audio = bool(joyai_echo.get(f"paired_audio{suffix}"))
        video_latent = joyai_echo.get(f"video_latent{suffix}")
        audio_latent = joyai_echo.get(f"audio_latent{suffix}")
        audio_segment_lengths = joyai_echo.get(f"audio_segment_lengths{suffix}")
        return paired_reference_conditionings_by_latents(
            video_latent=video_latent,
            audio_latent=audio_latent,
            audio_segment_lengths=audio_segment_lengths,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            video_downscale_factor=int(joyai_echo.get("downscale_factor", 1)),
            use_paired_cross_attention_mask=paired_audio,
            audio_target_to_reference=not paired_audio,
        )

    @torch.inference_mode()
    def upscale_video(
        self,
        source_video: torch.Tensor,
        prompt: str,
        negative_prompt: str,
        audio_latent: torch.Tensor,
        upsampler_variant: str,
        seed: int,
        frame_rate: float,
        sigma_values: tuple[float, ...],
        latent_frame_offset: int = 0,
        pixel_frame_offset: int = 0,
        tiling_config: TilingConfig | None = None,
        callback: Callable[..., None] | None = None,
        set_progress_status: Callable[[str], None] | None = None,
        interrupt_check: Callable[[], bool] | None = None,
    ) -> torch.Tensor | None:
        """Encode, spatially upscale, and lightly refine one source-video window."""
        if interrupt_check is not None and interrupt_check():
            return None
        dtype = torch.bfloat16
        prompt = prompt.strip()
        negative_prompt = negative_prompt.strip() or DEFAULT_NEGATIVE_PROMPT
        use_cfg_pp = upsampler_variant == "ltx23"
        use_ancestral_sampler = upsampler_variant == "ltx25"
        if not use_cfg_pp and not use_ancestral_sampler:
            raise ValueError(f"Unknown LTX upsampler variant: {upsampler_variant}")

        if set_progress_status is not None:
            set_progress_status("Encoding prompt")
        text_encoder = self._get_model("text_encoder")
        feature_extractor, video_connector, audio_connector = resolve_text_connectors(text_encoder, None)
        encode_fn = lambda prompts: postprocess_text_embeddings(
            encode_text(text_encoder, prompts=prompts),
            feature_extractor,
            video_connector,
            audio_connector,
        )
        contexts = self.text_encoder_cache.encode(encode_fn, [prompt, negative_prompt] if use_cfg_pp else [prompt], device=self.device, parallel=True)
        video_context, audio_context = contexts[0]
        negative_video_context, negative_audio_context = contexts[1] if use_cfg_pp else (None, None)
        del text_encoder, contexts
        cleanup_memory()

        if interrupt_check is not None and interrupt_check():
            return None
        if set_progress_status is not None:
            set_progress_status("VAE encoding")
        video_encoder = self._get_model("video_encoder")
        with control_video_encoding():
            source_latent = vae_encode_video(source_video, video_encoder, tiling_config).to(device=self.device, dtype=dtype)

        if interrupt_check is not None and interrupt_check():
            return None
        output_shape = VideoPixelShape(
            batch=int(source_video.shape[0]),
            frames=int(source_video.shape[2]),
            height=int(source_video.shape[3]) * 2,
            width=int(source_video.shape[4]) * 2,
            fps=float(frame_rate),
        )
        conditionings, _, _ = paired_reference_conditionings_by_latents(
            video_latent=source_latent,
            audio_latent=None,
            audio_segment_lengths=None,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            video_downscale_factor=2,
        )
        if use_ancestral_sampler and LTX25_UPSAMPLER_INJECT_FIRST_FRAME:
            first_frame = source_video[0, :, 0].permute(1, 2, 0)
            conditionings += image_conditionings_by_replacing_latent(
                images=[(first_frame, 0, 0.7, "lanczos")],
                height=int(output_shape.height),
                width=int(output_shape.width),
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
                tiling_config=tiling_config,
            )
        target_audio_frames = AudioLatentShape.from_video_pixel_shape(output_shape).frames
        if audio_latent.shape[2] < target_audio_frames:
            audio_latent = torch.nn.functional.pad(audio_latent, (0, 0, 0, target_audio_frames - audio_latent.shape[2]), mode="replicate")
        audio_latent = audio_latent[:, :, :target_audio_frames].to(device=self.device, dtype=dtype)
        audio_conditionings = [AudioConditionByLatent(audio_latent, 1.0), AudioConditionByReferenceLatent(audio_latent)]
        target_latent_shape = source_latent.shape[:3] + (int(source_latent.shape[3]) * 2, int(source_latent.shape[4]) * 2)
        source_video = source_latent = audio_latent = None
        if interrupt_check is not None and interrupt_check():
            return None
        fixed_noise = _indexed_noise_tokens(target_latent_shape, seed, latent_frame_offset, dtype)
        noiser = _FixedNoiseNoiser(fixed_noise)
        sigmas = torch.tensor(sigma_values, device=self.device, dtype=torch.float32)
        transformer = self._get_model("transformer")
        bind_interrupt_check(transformer, interrupt_check)

        denoise_fn = (
            multi_modal_guider_denoising_func(
                video_guider=MultiModalGuider(MultiModalGuiderParams(cfg_scale=1.0), negative_context=negative_video_context),
                audio_guider=MultiModalGuider(MultiModalGuiderParams(cfg_scale=1.0), negative_context=negative_audio_context),
                v_context_p=video_context,
                a_context_p=audio_context,
                transformer=transformer,
                force_unconditional=True,
                capture_unconditional=True,
            )
            if use_cfg_pp
            else simple_denoising_func(video_context=video_context, audio_context=audio_context, transformer=transformer)
        )
        ancestral_noise_generator = torch.Generator(device=self.device).manual_seed(int(seed) + 10000) if use_ancestral_sampler else None

        def denoising_loop(sigmas, video_state, audio_state, stepper, preview_tools=None):
            if callback is not None:
                callback(-1, None, True, override_num_inference_steps=len(sigmas) - 1)
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=denoise_fn,
                interrupt_check=interrupt_check,
                callback=callback,
                preview_tools=preview_tools,
                transformer=transformer,
                ancestral_noise_generator=ancestral_noise_generator,
            )

        if set_progress_status is not None:
            set_progress_status("Distilled refinement")
        stepper = EulerCFGPPDiffusionStep() if use_cfg_pp else EulerAncestralDiffusionStep()
        video_state, _ = denoise_audio_video(
            output_shape=output_shape,
            conditionings=conditionings,
            noiser=noiser,
            sigmas=sigmas,
            stepper=stepper,
            denoising_loop_fn=denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            audio_conditionings=audio_conditionings,
            noise_scale=float(sigmas[0]),
            freeze_audio=True,
            video_position_offset=float(pixel_frame_offset) / float(frame_rate),
        )
        fixed_noise = noiser = None
        if video_state is None or interrupt_check is not None and interrupt_check():
            return None
        del transformer, video_encoder, video_context, audio_context, negative_video_context, negative_audio_context
        cleanup_memory()

        if set_progress_status is not None:
            set_progress_status("VAE decoding")
        latent = [video_state.latent]
        video_state = None
        return vae_decode_video_to_tensor(
            latent,
            self._get_model("video_decoder"),
            tiling_config,
            expected_frames=int(output_shape.frames),
            expected_height=int(output_shape.height),
            expected_width=int(output_shape.width),
            interrupt_check=interrupt_check,
        )

    @torch.inference_mode()
    def refine_video(
        self,
        load_window: Callable[[int, int], torch.Tensor],
        frame_count: int,
        height: int,
        width: int,
        output_frames: int,
        output_height: int,
        output_width: int,
        prompt: str,
        seed: int,
        frame_rate: float,
        window_frames: int,
        tile_height: int,
        tile_width: int,
        gentle: bool,
        tiles_per_call: int = 0,
        conditioning_latent: torch.Tensor | None = None,
        carry_frames: int = 0,
        fusion_previous: torch.Tensor | None = None,
        fusion_export: slice | None = None,
        latent_frame_offset: int = 0,
        tiling_config: TilingConfig | None = None,
        callback: Callable[..., None] | None = None,
        set_progress_status: Callable[[str], None] | None = None,
        interrupt_check: Callable[[], bool] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        """Refine Details IC-LoRA with per-step tiled fusion on a ``frame_count`` x ``height`` x ``width`` canvas.

        ``load_window(start, count)`` returns uint8 ``(1, 3, count, height, width)`` source frames already
        resized to the canvas; every temporal window is encoded as its own fresh reference clip.
        ``tiles_per_call`` 0 sizes transformer tile groups from free VRAM. ``conditioning_latent``
        ``(1, C, k, H, W)`` conditions the first latent frames (previous Media Flow chunk). Returns
        the decoded uint8 frames and, when ``carry_frames`` > 0, CPU latents of the last ``carry_frames``
        refined frames (next chunk) and of the ``carry_frames`` before them (resumed run), each re-encoded
        as a fresh clip. Cross-chunk fusion instead blends ``fusion_previous`` ``(steps, n, H, W, C)`` into latents
        ``1..n`` at every step and returns this canvas' per-step predictions for the ``fusion_export`` latents.
        """
        dtype = torch.bfloat16
        components = self.pipeline_components
        if set_progress_status is not None:
            set_progress_status("Encoding Text Prompt")
        text_encoder = self._get_model("text_encoder")
        feature_extractor, video_connector, audio_connector = resolve_text_connectors(text_encoder, None)
        encode_fn = lambda prompts: postprocess_text_embeddings(encode_text(text_encoder, prompts=prompts), feature_extractor, video_connector, audio_connector)
        video_context, _ = self.text_encoder_cache.encode(encode_fn, [prompt.strip()], device=self.device, parallel=True)[0]
        del text_encoder
        cleanup_memory()

        scale = components.video_scale_factors
        build_plan = lambda device: TiledFusionPlan.build((frame_count - 1) // scale.time + 1, height // scale.height, width // scale.width, tile_height // scale.height, tile_width // scale.width, (window_frames - 1) // scale.time + 1, device=device)
        plan = build_plan(self.device)
        channels = components.video_latent_channels
        storage_bytes = plan.frames * plan.height * plan.width * channels * 6 + len(plan.windows) * plan.tile_frames * plan.height * plan.width * channels * 2
        if storage_bytes > torch.cuda.mem_get_info(self.device)[0] * REFINE_CANVAS_FREE_RATIO:
            plan = build_plan(torch.device("cpu"))
        storage = plan.weights.device
        video_encoder = self._get_model("video_encoder")
        references = []
        for window, t0 in enumerate(plan.windows, start=1):
            if interrupt_check is not None and interrupt_check():
                return None
            if set_progress_status is not None:
                set_progress_status(f"Encoding Reference Video {window}/{len(plan.windows)}" if len(plan.windows) > 1 else "Encoding Reference Video")
            pixels = load_window(t0 * scale.time, (plan.tile_frames - 1) * scale.time + 1)
            with control_video_encoding():
                latent = vae_encode_video(pixels, video_encoder, tiling_config, device=self.device, dtype=dtype)
            pixels = None
            references.append(latent.to(device=storage, dtype=dtype).permute(0, 2, 3, 4, 1).contiguous())
            latent = None
        del video_encoder
        initial_latent = None
        if gentle:
            initial_latent = references[0] if len(references) == 1 else plan.blend_windows(references).to(dtype)
            initial_latent = initial_latent.permute(0, 4, 1, 2, 3).contiguous()
        conditionings = [] if conditioning_latent is None else latent_conditionings_by_latent_sequence(conditioning_latent.to(device=storage, dtype=dtype), strength=REFINE_CARRY_STRENGTH, start_index=0)
        sigmas = torch.tensor(STAGE_2_DISTILLED_SIGMA_VALUES if gentle else DISTILLED_SIGMA_VALUES, device=storage, dtype=torch.float32)
        if fusion_previous is not None and fusion_previous.shape[0] != len(sigmas) - 1:
            print("[LTX 2.5 Detail Refiner] Continuation data was refined with another step count: refining this chunk without it.")
            fusion_previous = None
        exported = [] if fusion_export is not None else None
        latent_shape = (1, channels, plan.frames, plan.height, plan.width)
        noiser = _FixedNoiseNoiser(_indexed_noise_tokens(latent_shape, seed, latent_frame_offset, dtype))
        transformer = self._get_model("transformer")
        bind_interrupt_check(transformer, interrupt_check)
        if tiles_per_call <= 0:
            tile_bytes = 2 * plan.tile_frames * plan.tile_height * plan.tile_width * transformer.velocity_model.inner_dim * torch.finfo(dtype).bits // 8
            tiles_per_call = int(min(REFINE_TILE_BATCH_MAX_BYTES, torch.cuda.mem_get_info(self.device)[0] * REFINE_TILE_BATCH_FREE_RATIO) // tile_bytes)
        tiles_per_call = max(1, min(plan.tiles_per_step, tiles_per_call))

        denoise_fn = tiled_fusion_denoising_func(transformer=transformer, video_context=video_context, plan=plan, references=references, patchifier=components.video_patchifier, scale_factors=scale, fps=frame_rate, device=self.device, tiles_per_call=tiles_per_call, callback=callback, previous_x0=fusion_previous, export_x0=exported, export_latents=fusion_export)

        def denoising_loop(sigmas, video_state, audio_state, stepper):
            return euler_denoising_loop(sigmas=sigmas, video_state=video_state, audio_state=audio_state, stepper=stepper, denoise_fn=denoise_fn, interrupt_check=interrupt_check, transformer=transformer)

        if set_progress_status is not None:
            set_progress_status("Detail Refinement")
        video_state, _ = denoise_audio_video(
            output_shape=VideoPixelShape(batch=1, frames=frame_count, height=height, width=width, fps=float(frame_rate)),
            conditionings=conditionings,
            noiser=noiser,
            sigmas=sigmas,
            stepper=EulerDiffusionStep(),
            denoising_loop_fn=denoising_loop,
            components=components,
            dtype=dtype,
            device=storage,
            noise_scale=float(sigmas[0]),
            initial_video_latent=initial_latent,
            skip_audio=True,
        )
        references = initial_latent = noiser = denoise_fn = conditionings = None
        if video_state is None or interrupt_check is not None and interrupt_check():
            return None
        del transformer, video_context
        cleanup_memory()

        if set_progress_status is not None:
            set_progress_status("VAE Decoding")
        latent = [video_state.latent.to(self.device)]
        video_state = None
        decoded = vae_decode_video_to_tensor(latent, self._get_model("video_decoder"), tiling_config, expected_frames=output_frames, expected_height=output_height, expected_width=output_width, interrupt_check=interrupt_check)
        if decoded is None:
            return None
        carry_latents = None if exported is None else torch.stack(exported)
        if carry_frames > 0:
            if set_progress_status is not None:
                set_progress_status("Encoding Overlap Frames")
            carry_latents = tuple(vae_encode_video(_pad_frames_to_canvas(frames, height, width), self._get_model("video_encoder"), tiling_config, device=self.device, dtype=dtype).cpu() for frames in (decoded[-carry_frames:], decoded[-2 * carry_frames : -carry_frames]))
        return decoded, carry_latents

    def __call__(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[tuple[str, int, float]],
        prompt_relay_frame_offset: int = 0,
        prompt_relay_epsilon: float = 1e-3,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        custom_sigmas=None,
        guiding_images: list[tuple] | None = None,
        guiding_images_stage2: list[tuple] | None = None,
        images_stage2: list[tuple[str, int, float]] | None = None,
        alt_guidance_scale: float = 1.0,
        audio_cfg_guidance_scale: float = 1.0,
        NAG_scale: float = 1.0,
        NAG_tau: float = 3.5,
        NAG_alpha: float = 0.5,
        video_conditioning: list[tuple[str, float]] | None = None,
        video_conditioning_downscale_factor: int = 1,
        video_conditioning_stage2: list[tuple[str, float]] | None = None,
        latent_conditioning_stage2: torch.Tensor | None = None,
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
        audio_conditionings: list | None = None,
        audio_conditionings_stage2: list | None = None,
        audio_identity_guidance_scale: float = 0.0,
        callback: Callable[..., None] | None = None,
        set_progress_status: Callable[[str], None] | None = None,
        interrupt_check: Callable[[], bool] | None = None,
        loras_slists: dict | None = None,
        text_connectors: dict | None = None,
        masking_source: dict | None = None,
        masking_strength: float | None = None,
        return_latent_slice: slice | None = None,
        hdr_transform: str | None = None,
        precomputed_contexts: tuple[torch.Tensor, torch.Tensor] | None = None,
        skip_audio: bool = False,
        continuous_conditioning_and_guide: bool = False,
        skip_stage_2: bool = False,
        tiled_stage_2: bool = False,
        frozen_video_conditioning: torch.Tensor | None = None,
        frozen_output_video: torch.Tensor | None = None,
        self_refiner_setting: int = 0,
        self_refiner_plan: str = "",
        self_refiner_f_uncertainty: float = 0.1,
        self_refiner_certain_percentage: float = 0.999,
        self_refiner_max_plans: int = 1,
        editanything_ref_images=None,
        ltx2_22B_class: bool = False,
        use_ancestral_sampler: bool = False,
        layout_to_render: bool = False,
    ) -> tuple[Iterator[torch.Tensor], torch.Tensor]:
        joyai_echo = self._joyai_echo or {}
        return_joyai_memory = bool(joyai_echo.get("return_latents"))
        assert_resolution(height=height, width=width, is_two_stage=not skip_stage_2)
        alt_guidance_scale = 1.0
        negative_prompt = negative_prompt or DEFAULT_NEGATIVE_PROMPT

        generator = torch.Generator(device=self.device).manual_seed(seed)
        mask_generator = torch.Generator(device=self.device).manual_seed(int(seed) + 1)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerAncestralDiffusionStep() if use_ancestral_sampler else EulerDiffusionStep()
        ancestral_noise_generator = torch.Generator(device=self.device).manual_seed(int(seed) + 10000) if use_ancestral_sampler else None
        self_refiner_handler = None
        self_refiner_handler_audio = None
        self_refiner_handler_stage2 = None
        self_refiner_handler_audio_stage2 = None
        if self_refiner_setting and self_refiner_setting > 0:
            plans, _ = normalize_self_refiner_plan(self_refiner_plan or "", max_plans=self_refiner_max_plans)
            plan_stage1 = plans[0] if plans else []
            plan_stage2 = plans[1] if len(plans) > 1 else []
            self_refiner_handler = create_self_refiner_handler(
                plan_stage1,
                self_refiner_f_uncertainty,
                self_refiner_setting,
                self_refiner_certain_percentage,
                channel_dim=-1,
            )
            self_refiner_handler_audio = create_self_refiner_handler(
                plan_stage1,
                self_refiner_f_uncertainty,
                self_refiner_setting,
                self_refiner_certain_percentage,
                channel_dim=-1,
            )
            if plan_stage2:
                self_refiner_handler_stage2 = create_self_refiner_handler(
                    plan_stage2,
                    self_refiner_f_uncertainty,
                    self_refiner_setting,
                    self_refiner_certain_percentage,
                    channel_dim=-1,
                )
                self_refiner_handler_audio_stage2 = create_self_refiner_handler(
                    plan_stage2,
                    self_refiner_f_uncertainty,
                    self_refiner_setting,
                    self_refiner_certain_percentage,
                    channel_dim=-1,
                )
        dtype = torch.bfloat16

        enable_audio_text_nag = False
        video_NAG = None
        audio_NAG = None
        if precomputed_contexts is not None:
            video_context, audio_context = precomputed_contexts
            video_context = video_context.to(device=self.device, dtype=dtype)
            audio_context = None if skip_audio else audio_context.to(device=self.device, dtype=dtype)
            contexts = [(video_context, audio_context)]
            NAG_scale = 1.0
            video_context_mask_builder = None
            audio_context_mask_builder = None
        else:
            text_encoder = self._get_model("text_encoder")
            if enhance_prompt:
                prompt = generate_enhanced_prompt(text_encoder, prompt, images[0][0] if len(images) > 0 else None)
            feature_extractor, video_connector, audio_connector = resolve_text_connectors(
                text_encoder, text_connectors
            )
            encode_fn = lambda prompts: postprocess_text_embeddings(
                encode_text(text_encoder, prompts=prompts),
                feature_extractor,
                video_connector,
                audio_connector,
            )
            encode_fn_with_masks = lambda prompts: postprocess_text_embeddings(
                encode_text(text_encoder, prompts=prompts),
                feature_extractor,
                video_connector,
                audio_connector,
                return_attention_masks=True,
            )
            relay_conditioning = encode_prompt_relay(prompt, encode_fn_with_masks, self.text_encoder_cache, self.device, num_frames, frame_rate, text_encoder.tokenizer, visible_frame_offset=prompt_relay_frame_offset, epsilon=prompt_relay_epsilon)
            if relay_conditioning is not None:
                video_context = relay_conditioning.video_context
                audio_context = None if skip_audio else relay_conditioning.audio_context
                video_context_mask_builder = relay_conditioning.video_mask_builder
                audio_context_mask_builder = None if skip_audio else relay_conditioning.audio_mask_builder
                if float(NAG_scale) > 1.0:
                    context_n = self.text_encoder_cache.encode(encode_fn, [negative_prompt], device=self.device, parallel=True)[0]
                    contexts = [(video_context, audio_context), context_n]
                else:
                    contexts = [(video_context, audio_context)]
            elif float(NAG_scale) > 1.0:
                contexts = self.text_encoder_cache.encode(
                    encode_fn,
                    [prompt, negative_prompt],
                    device=self.device,
                    parallel=True,
                )
                video_context_mask_builder = None
                audio_context_mask_builder = None
            else:
                contexts = self.text_encoder_cache.encode(encode_fn, [prompt], device=self.device, parallel=True)
                video_context_mask_builder = None
                audio_context_mask_builder = None

            torch.cuda.synchronize()
            del text_encoder
            cleanup_memory()
        audio_context_n = None
        if float(NAG_scale) > 1.0:
            (video_context, audio_context), (video_context_n, audio_context_nag) = contexts
            video_pos_len = video_context.shape[0] if video_context.dim() == 2 else video_context.shape[1]
            video_context_n = _align_seq_len(video_context_n, video_pos_len)
            video_cat_dim = 0 if video_context.dim() == 2 else 1
            video_context = torch.cat([video_context, video_context_n], dim=video_cat_dim)
            video_NAG = {
                "scale": float(NAG_scale),
                "tau": float(NAG_tau),
                "alpha": float(NAG_alpha),
                "cap_embed_len": int(video_pos_len),
                "enable_audio_text_nag": enable_audio_text_nag,
            }
            if enable_audio_text_nag:
                audio_pos_len = audio_context.shape[0] if audio_context.dim() == 2 else audio_context.shape[1]
                audio_context_nag = _align_seq_len(audio_context_nag, audio_pos_len)
                audio_cat_dim = 0 if audio_context.dim() == 2 else 1
                audio_context = torch.cat([audio_context, audio_context_nag], dim=audio_cat_dim)
                audio_NAG = {
                    "scale": float(NAG_scale),
                    "tau": float(NAG_tau),
                    "alpha": float(NAG_alpha),
                    "cap_embed_len": int(audio_pos_len),
                    "enable_audio_text_nag": enable_audio_text_nag,
                }
        else:
            video_context, audio_context = contexts[0]

        # Stage 1: Initial low resolution video generation.
        bench_transformer = _env_flag(_BENCH_TRANSFORMER_ENV, "0")
        skip_stage_2 = bool(skip_stage_2)
        stage_1_pass_no = 0 if skip_stage_2 else 1
        if interrupt_check is not None and interrupt_check():
            return None, None
        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width if skip_stage_2 else width // 2,
            height=height if skip_stage_2 else height // 2,
            fps=frame_rate,
        )
        memory_video_conditionings, memory_audio_conditionings, cross_attention_mask_builder = self._joyai_echo_reference_conditionings(joyai_echo, "", dtype)
        video_encoder = self._get_model("video_encoder")
        transformer = _TransformerBenchWrapper(self._get_model("transformer"), enabled=bench_transformer)
        bind_interrupt_check(transformer, interrupt_check)
        stage_1_ref_conditionings, stage_1_ref_context, stage_1_ref_adaln = build_editanything_reference_conditioning(
            transformer,
            editanything_ref_images,
            height=stage_1_output_shape.height,
            width=stage_1_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
            tiling_config=tiling_config,
        )
        # DISTILLED_SIGMA_VALUES = [0.421875, 0]
        stage_1_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(self.device)
        if custom_sigmas and custom_sigmas[0] is not None:
            stage_1_sigmas = torch.tensor([*custom_sigmas[0], 0.0], dtype=torch.float32, device=self.device)
        pass_no = stage_1_pass_no
        if loras_slists is not None:
            stage_1_steps = len(stage_1_sigmas) - 1
            update_loras_slists(
                transformer,
                loras_slists,
                stage_1_steps,
                phase_switch_step=stage_1_steps,
                phase_switch_step2=stage_1_steps,
            )

        if set_progress_status is not None:
            set_progress_status("VAE Encoding")

        def denoising_loop_stage1(
            sigmas: torch.Tensor,
            video_state: LatentState,
            audio_state: LatentState,
            stepper: DiffusionStepProtocol,
            preview_tools: VideoLatentTools | None = None,
            mask_context=None,
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=video_context,
                    audio_context=audio_context,
                    transformer=transformer,  # noqa: F821
                    video_nag=video_NAG,
                    audio_nag=audio_NAG,
                    alt_guidance_scale=alt_guidance_scale,
                    audio_context_n=audio_context_n,
                    audio_guidance_scale=audio_cfg_guidance_scale,
                    audio_identity_guidance_scale=audio_identity_guidance_scale,
                    skip_audio_to_video=frozen_video_conditioning is not None,
                    ref_context=stage_1_ref_context,
                    ref_adaln=stage_1_ref_adaln,
                    video_context_mask_builder=video_context_mask_builder,
                    audio_context_mask_builder=audio_context_mask_builder,
                    cross_attention_mask_builder=cross_attention_mask_builder,
                ),
                mask_context=mask_context,
                interrupt_check=interrupt_check,
                callback=callback,
                preview_tools=preview_tools,
                pass_no=stage_1_pass_no,
                transformer=transformer,
                self_refiner_handler=self_refiner_handler,
                self_refiner_handler_audio=self_refiner_handler_audio,
                ancestral_noise_generator=ancestral_noise_generator,
                self_refiner_generator=generator,
            )

        if interrupt_check is not None and interrupt_check():
            return None, None
        if frozen_video_conditioning is not None:
            stage_1_conditionings = video_conditionings_by_frozen_video(
                video=frozen_video_conditioning,
                height=stage_1_output_shape.height,
                width=stage_1_output_shape.width,
                num_frames=num_frames,
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
                tiling_config=tiling_config,
            )
        else:
            stage_1_conditionings = image_conditionings_by_replacing_latent(
                images=images,
                height=stage_1_output_shape.height,
                width=stage_1_output_shape.width,
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
                tiling_config=tiling_config,
            )
        stage_1_conditionings += stage_1_ref_conditionings
        if frozen_video_conditioning is None and guiding_images:
            stage_1_conditionings += image_conditionings_by_adding_guiding_latent(
                images=guiding_images,
                height=stage_1_output_shape.height,
                width=stage_1_output_shape.width,
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
                tiling_config=tiling_config,
            )
        if frozen_video_conditioning is None and video_conditioning:
            stage_1_conditionings += video_conditionings_by_control_video(
                video_conditioning=video_conditioning,
                height=stage_1_output_shape.height,
                width=stage_1_output_shape.width,
                num_frames=num_frames,
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
                downscale_factor=video_conditioning_downscale_factor,
                tiling_config=tiling_config,
                continuous_conditioning_and_guide=continuous_conditioning_and_guide,
            )
        stage_1_conditionings += memory_video_conditionings
        stage_1_audio_conditionings = list(audio_conditionings or []) + memory_audio_conditionings

        mask_context = prepare_mask_injection(
            masking_source=masking_source,
            masking_strength=masking_strength,
            output_shape=stage_1_output_shape,
            video_encoder=video_encoder,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            tiling_config=tiling_config,
            generator=mask_generator,
            num_steps=len(stage_1_sigmas) - 1,
        )
        if callback is not None:
            callback(-1, None, True, override_num_inference_steps=len(stage_1_sigmas) - 1, pass_no=pass_no)
        video_state, audio_state = denoise_audio_video(
            output_shape=stage_1_output_shape,
            conditionings=stage_1_conditionings,
            audio_conditionings=stage_1_audio_conditionings,
            noiser=noiser,
            sigmas=stage_1_sigmas,
            stepper=stepper,
            denoising_loop_fn=denoising_loop_stage1,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            noise_scale=stage_1_sigmas[0],
            mask_context=mask_context,
            skip_audio=skip_audio,
        )
        stage1_transformer_ms = 0.0
        stage1_transformer_calls = 0
        if bench_transformer:
            stage1_transformer_ms, stage1_transformer_calls = transformer.consume()
            print(
                "[WAN2GP][LTX2][bench] transformer pass1: "
                f"{stage1_transformer_ms / 1000.0:.3f}s ({stage1_transformer_calls} calls)"
            )
        if video_state is None or (not skip_audio and audio_state is None):
            return None, None
        if interrupt_check is not None and interrupt_check():
            return None, None
        stage_1_memory_latents = None
        if return_joyai_memory:
            stage_1_memory_latents = {
                "video": video_state.latent.detach().cpu(),
                "audio": audio_state.latent.detach().cpu() if audio_state is not None else None,
            }
        if skip_stage_2:
            if bench_transformer:
                print(
                    "[WAN2GP][LTX2][bench] transformer total: "
                    f"{stage1_transformer_ms / 1000.0:.3f}s ({stage1_transformer_calls} calls)"
                )
            torch.cuda.synchronize()
            del transformer
            del video_encoder
            cleanup_memory()
            latent_slice = None
            if return_latent_slice is not None:
                latent_slice = video_state.latent[:, :, return_latent_slice].detach().to("cpu")
            if frozen_output_video is None:
                video_latent = [video_state.latent]
                video_state = None
                decoded_video = vae_decode_video_to_tensor(
                    video_latent,
                    self._get_model("video_decoder"),
                    tiling_config,
                    expected_frames=int(stage_1_output_shape.frames),
                    expected_height=int(stage_1_output_shape.height),
                    expected_width=int(stage_1_output_shape.width),
                    interrupt_check=interrupt_check,
                    hdr_transform=hdr_transform,
                    output_dtype=torch.float16 if hdr_transform is not None else None,
                    generator=generator,
                )
            else:
                decoded_video = frozen_output_video[:, :num_frames, :height, :width].permute(1, 2, 3, 0)
            decoded_audio = None
            if audio_state is not None:
                decoded_audio = vae_decode_audio(
                    audio_state.latent, self._get_model("audio_decoder"), self._get_model("vocoder")
                )
            if return_joyai_memory:
                return decoded_video, decoded_audio, latent_slice, {"phase1": stage_1_memory_latents}
            if latent_slice is not None:
                return decoded_video, decoded_audio, latent_slice
            return decoded_video, decoded_audio

        stepper = EulerAncestralDiffusionStep() if layout_to_render else EulerDiffusionStep()
        stage_2_sigma_values = DISTILLED_8_STEPS_STAGE_2_SIGMA_VALUES if not layout_to_render and LTX23_USE_DISTILLED_8_STEPS_STAGE_2_SIGMAS and ltx2_22B_class else STAGE_2_DISTILLED_SIGMA_VALUES
        stage_2_sigmas = torch.Tensor(stage_2_sigma_values).to(self.device)
        if custom_sigmas and len(custom_sigmas) > 1 and custom_sigmas[1] is not None:
            stage_2_sigmas = torch.tensor([*custom_sigmas[1], 0.0], dtype=torch.float32, device=self.device)
        upscaled_video_latent = upsample_video(
            latent=video_state.latent[:1],
            video_encoder=video_encoder,
            upsampler=self._get_model("spatial_upsampler"),
        )

        torch.cuda.synchronize()
        cleanup_memory()

        pass_no = 2
        stage_2_ref_context = stage_2_ref_adaln = None
        stage_2_ref_conditionings = []
        if loras_slists is not None:
            stage_2_steps = len(stage_2_sigmas) - 1
            update_loras_slists(
                transformer,
                loras_slists,
                stage_2_steps,
                phase_switch_step=0,
                phase_switch_step2=stage_2_steps,
            )
        if set_progress_status is not None:
            set_progress_status("VAE Encoding")

        stage_2_memory_video_conditionings, stage_2_memory_audio_conditionings, stage_2_cross_attention_mask_builder = self._joyai_echo_reference_conditionings(joyai_echo, "_stage2", dtype)

        def denoising_loop_stage2(
            sigmas: torch.Tensor,
            video_state: LatentState,
            audio_state: LatentState,
            stepper: DiffusionStepProtocol,
            preview_tools: VideoLatentTools | None = None,
            mask_context=None,
        ) -> tuple[LatentState, LatentState]:
            denoise_fn = simple_denoising_func(
                video_context=video_context,
                audio_context=audio_context,
                transformer=transformer,  # noqa: F821
                video_nag=video_NAG,
                audio_nag=audio_NAG,
                alt_guidance_scale=alt_guidance_scale,
                skip_audio_to_video=frozen_video_conditioning is not None,
                ref_context=stage_2_ref_context,
                ref_adaln=stage_2_ref_adaln,
                video_context_mask_builder=video_context_mask_builder,
                audio_context_mask_builder=audio_context_mask_builder,
                cross_attention_mask_builder=stage_2_cross_attention_mask_builder,
            )
            if tiled_stage_2:
                denoise_fn = spatially_tiled_denoising_func(denoise_fn, stage_2_output_shape, self.pipeline_components, interrupt_check, callback)
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=denoise_fn,
                mask_context=mask_context,
                interrupt_check=interrupt_check,
                callback=callback,
                preview_tools=preview_tools,
                pass_no=2,
                transformer=transformer,
                self_refiner_handler=self_refiner_handler_stage2,
                self_refiner_handler_audio=self_refiner_handler_audio_stage2,
                self_refiner_generator=generator,
                ancestral_noise_generator=ancestral_noise_generator if layout_to_render else None,
            )

        stage_2_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width,
            height=height,
            fps=frame_rate,
        )
        stage_2_ref_conditionings, stage_2_ref_context, stage_2_ref_adaln = build_editanything_reference_conditioning(
            transformer,
            editanything_ref_images,
            height=stage_2_output_shape.height,
            width=stage_2_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
            tiling_config=tiling_config,
        )
        if interrupt_check is not None and interrupt_check():
            return None, None
        if frozen_video_conditioning is not None:
            stage_2_conditionings = video_conditionings_by_frozen_video(
                video=frozen_video_conditioning,
                height=stage_2_output_shape.height,
                width=stage_2_output_shape.width,
                num_frames=num_frames,
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
                tiling_config=tiling_config,
            )
        else:
            stage_2_conditionings = image_conditionings_by_replacing_latent(
                images=images_stage2 if images_stage2 is not None else images,
                height=stage_2_output_shape.height,
                width=stage_2_output_shape.width,
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
                tiling_config=tiling_config,
            )
        stage_2_conditionings += stage_2_ref_conditionings
        if frozen_video_conditioning is None and guiding_images_stage2:
            stage_2_conditionings += image_conditionings_by_adding_guiding_latent(
                images=guiding_images_stage2,
                height=stage_2_output_shape.height,
                width=stage_2_output_shape.width,
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
                tiling_config=tiling_config,
            )
        if frozen_video_conditioning is None and latent_conditioning_stage2 is not None:
            stage_2_conditionings += latent_conditionings_by_latent_sequence(
                latent_conditioning_stage2,
                strength=1.0,
                start_index=0,
            )
        if frozen_video_conditioning is None and video_conditioning_stage2:
            stage_2_conditionings += video_conditionings_by_control_video(
                video_conditioning=video_conditioning_stage2,
                height=stage_2_output_shape.height,
                width=stage_2_output_shape.width,
                num_frames=num_frames,
                video_encoder=video_encoder,
                dtype=dtype,
                device=self.device,
                downscale_factor=video_conditioning_downscale_factor,
                tiling_config=tiling_config,
                continuous_conditioning_and_guide=continuous_conditioning_and_guide,
            )
        stage_2_conditionings += stage_2_memory_video_conditionings
        mask_context = prepare_mask_injection(
            masking_source=masking_source,
            masking_strength=masking_strength,
            output_shape=stage_2_output_shape,
            video_encoder=video_encoder,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            tiling_config=tiling_config,
            generator=mask_generator,
            num_steps=len(stage_2_sigmas) - 1,
        )
        if callback is not None:
            callback(-1, None, True, override_num_inference_steps=len(stage_2_sigmas) - 1, pass_no=pass_no)
        freeze_audio_stage2 = audio_identity_guidance_scale > 0.0 or bool(joyai_echo.get("freeze_stage2_audio", False))
        stage_2_audio_conditionings = audio_conditionings if audio_conditionings_stage2 is None else audio_conditionings_stage2
        stage_2_audio_conditionings = list(stage_2_audio_conditionings or []) + stage_2_memory_audio_conditionings
        video_state, audio_state = denoise_audio_video(
            output_shape=stage_2_output_shape,
            conditionings=stage_2_conditionings,
            audio_conditionings=stage_2_audio_conditionings,
            noiser=noiser,
            sigmas=stage_2_sigmas,
            stepper=stepper,
            denoising_loop_fn=denoising_loop_stage2,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            noise_scale=stage_2_sigmas[0],
            audio_noise_scale=0.0 if freeze_audio_stage2 else stage_2_sigmas[0],
            initial_video_latent=upscaled_video_latent,
            initial_audio_latent=audio_state.latent if audio_state is not None else None,
            mask_context=mask_context,
            freeze_audio=freeze_audio_stage2,
            skip_audio=skip_audio,
        )
        if bench_transformer:
            stage2_transformer_ms, stage2_transformer_calls = transformer.consume()
            total_transformer_ms = stage1_transformer_ms + stage2_transformer_ms
            total_transformer_calls = stage1_transformer_calls + stage2_transformer_calls
            print(
                "[WAN2GP][LTX2][bench] transformer pass2: "
                f"{stage2_transformer_ms / 1000.0:.3f}s ({stage2_transformer_calls} calls)"
            )
            print(
                "[WAN2GP][LTX2][bench] transformer total: "
                f"{total_transformer_ms / 1000.0:.3f}s ({total_transformer_calls} calls)"
            )
        if video_state is None or (not skip_audio and audio_state is None):
            return None, None
        if interrupt_check is not None and interrupt_check():
            return None, None

        torch.cuda.synchronize()
        del transformer
        del video_encoder
        cleanup_memory()

        latent_slice = None
        if return_latent_slice is not None:
            latent_slice = video_state.latent[:, :, return_latent_slice].detach().to("cpu")
        phase_2_memory_latents = None
        if return_joyai_memory:
            phase_2_memory_latents = {
                "video": video_state.latent.detach().cpu(),
                "audio": audio_state.latent.detach().cpu() if audio_state is not None else None,
            }
        if frozen_output_video is None:
            video_latent = [video_state.latent]
            video_state = None
            decoded_video = vae_decode_video_to_tensor(
                video_latent,
                self._get_model("video_decoder"),
                tiling_config,
                expected_frames=int(stage_2_output_shape.frames),
                expected_height=int(stage_2_output_shape.height),
                expected_width=int(stage_2_output_shape.width),
                interrupt_check=interrupt_check,
                hdr_transform=hdr_transform,
                output_dtype=torch.float16 if hdr_transform is not None else None,
                generator=generator,
            )
        else:
            decoded_video = frozen_output_video[:, :num_frames, :height, :width].permute(1, 2, 3, 0)
        decoded_audio = None
        if audio_state is not None:
            decoded_audio = vae_decode_audio(
                audio_state.latent, self._get_model("audio_decoder"), self._get_model("vocoder")
            )
        if return_joyai_memory:
            memory_latents = {
                "phase1": stage_1_memory_latents,
                "phase2": phase_2_memory_latents,
            }
            return decoded_video, decoded_audio, latent_slice, memory_latents
        if latent_slice is not None:
            return decoded_video, decoded_audio, latent_slice
        return decoded_video, decoded_audio


@torch.inference_mode()
def main() -> None:
    from .utils.media_io import encode_video

    logging.getLogger().setLevel(logging.INFO)
    parser = default_2_stage_distilled_arg_parser()
    args = parser.parse_args()
    pipeline = DistilledPipeline(
        checkpoint_path=args.checkpoint_path,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=args.lora,
        fp8transformer=args.enable_fp8,
    )
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
    video, audio = pipeline(
        prompt=args.prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        images=args.images,
        tiling_config=tiling_config,
        enhance_prompt=args.enhance_prompt,
    )

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )


if __name__ == "__main__":
    main()
