# SPDX-FileCopyrightText: Copyright (c) 2025 Lightricks. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for WanGP.
"""LTX-2.5 SDR-to-HDR IC-LoRA conversion.

The SDR clip is moved to the ACEScct working space and appended as a frozen full-length reference. Every segment
border (seam) also gets a 1-frame SDR guide held at 95% and an empty generated keyframe slot. A single stage of 8
distilled Euler steps runs without CFG on the precomputed scene embedding; the generated slots then feed the
keyframe-aware NAD decode. The decoded ACEScct frames are returned as scene-linear Rec.709.
"""

from collections.abc import Callable

import torch

from shared.utils.hdr import sdr_to_vae_range
from shared.utils.loras_mutipliers import update_loras_slists
from shared.utils.phase_progress import control_video_encoding
from shared.utils.utils import guide_to_float

from ..ltx_core.components.diffusion_steps import EulerDiffusionStep
from ..ltx_core.components.noisers import GaussianNoiser
from ..ltx_core.conditioning import VideoConditionByKeyframeIndex, VideoConditionByReferenceLatent, VideoGeneratedKeyframeSlots
from ..ltx_core.model.video_vae import TilingConfig, encode_video as vae_encode_video
from ..ltx_core.model.video_vae import decode_video_to_tensor as vae_decode_video_to_tensor
from ..ltx_core.model.video_vae.keyframes import DecodeKeyframes
from ..ltx_core.types import VideoPixelShape
from .utils.constants import DISTILLED_SIGMA_VALUES
from .utils.helpers import bind_interrupt_check, cleanup_memory, denoise_audio_video, euler_denoising_loop, image_conditionings_by_replacing_latent, simple_denoising_func
from .utils.types import PipelineComponents

HDR_TRANSFORM = "acescct"
SEGMENT_CANDIDATES = (24, 32)
KEYFRAME_STRENGTH = 0.95
MAX_CONDITIONING_FPS = 30.0
_CONVERSION_CHUNK_FRAMES = 16


def seam_pixel_frames(num_frames: int) -> list[int]:
    """Borders of the 24 or 32-frame segments (whichever pads ``num_frames - 1`` least, ties to 32) that fall inside the clip."""
    content = num_frames - 1
    segment = min(SEGMENT_CANDIDATES, key=lambda size: ((-content) % size, -size))
    return [frame for frame in range(segment, content + (-content) % segment + 1, segment) if frame < num_frames]


def _sdr_to_working_space(video: torch.Tensor, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """``(C, F, H, W)`` sRGB SDR in [-1, 1] to ``(1, C, F, H, W)`` ACEScct in the VAE range on ``device``, converted by frame chunks."""
    out = torch.empty((1, *video.shape), dtype=dtype, device=device)
    for start in range(0, video.shape[1], _CONVERSION_CHUNK_FRAMES):
        out[0, :, start:start + _CONVERSION_CHUNK_FRAMES] = sdr_to_vae_range(guide_to_float(video[:, start:start + _CONVERSION_CHUNK_FRAMES]).to(device), transform=HDR_TRANSFORM, channel_dim=0)
    return out


@torch.inference_mode()
def convert_sdr_to_hdr(
    transformer,
    video_encoder,
    video_decoder,
    components: PipelineComponents,
    device: torch.device,
    control_video: torch.Tensor,
    prefix_frames_count: int,
    prefix_images: list[tuple],
    scene_context: torch.Tensor,
    num_frames: int,
    height: int,
    width: int,
    frame_rate: float,
    seed: int,
    tiling_config: TilingConfig | None = None,
    loras_slists: dict | None = None,
    callback: Callable[..., None] | None = None,
    set_progress_status: Callable[[str], None] | None = None,
    interrupt_check: Callable[[], bool] | None = None,
    custom_sigmas=None,
) -> torch.Tensor | None:
    """Convert an SDR control video ``(C, F, H, W)`` in [-1, 1] into scene-linear Rec.709 frames ``(F, H, W, 3)``.

    With ``prefix_frames_count`` (sliding windows, continued videos) the control video starts one frame before the
    first new frame and ``prefix_images`` already hold the working-space prefix frames, as in the other LTX-2 modes.
    """
    dtype = torch.bfloat16
    if set_progress_status is not None:
        set_progress_status("Encoding Control Video")
    sdr_video = _sdr_to_working_space(control_video[:, :num_frames], dtype, device)
    with control_video_encoding():
        reference = vae_encode_video(sdr_video, video_encoder, tiling_config, device=device, dtype=dtype)
    conditionings = image_conditionings_by_replacing_latent(images=prefix_images, height=height, width=width, video_encoder=video_encoder, dtype=dtype, device=device, tiling_config=tiling_config)
    conditionings.append(VideoConditionByReferenceLatent(latent=reference.to(device=device, dtype=dtype), strength=1.0, frame_idx=-prefix_frames_count))
    reference = None
    # Control frame ``c >= 1`` covers output frame ``prefix + c - 1`` when a prefix is conditioned, frame ``c`` otherwise.
    control_offset = prefix_frames_count - 1 if prefix_frames_count > 0 else 0
    seams = [frame for frame in seam_pixel_frames(num_frames) if frame >= prefix_frames_count and frame - control_offset < sdr_video.shape[2]]
    for frame in seams:
        if interrupt_check is not None and interrupt_check():
            return None
        guide = vae_encode_video(sdr_video[:, :, frame - control_offset:frame - control_offset + 1], video_encoder, tiling_config, device=device, dtype=dtype)
        conditionings.append(VideoConditionByKeyframeIndex(keyframes=guide.to(device=device, dtype=dtype), frame_idx=frame, strength=KEYFRAME_STRENGTH, num_pixel_frames=1))
    sdr_video = guide = None
    if seams:
        conditionings.append(VideoGeneratedKeyframeSlots(seams))
    if interrupt_check is not None and interrupt_check():
        return None

    sigmas = torch.tensor(DISTILLED_SIGMA_VALUES, device=device, dtype=torch.float32)
    if custom_sigmas and custom_sigmas[0] is not None:
        sigmas = torch.tensor([*custom_sigmas[0], 0.0], device=device, dtype=torch.float32)
    bind_interrupt_check(transformer, interrupt_check)
    if loras_slists is not None:
        steps = len(sigmas) - 1
        update_loras_slists(transformer, loras_slists, steps, phase_switch_step=steps, phase_switch_step2=steps)
    denoise_fn = simple_denoising_func(video_context=scene_context.to(device=device, dtype=dtype), audio_context=None, transformer=transformer)

    def denoising_loop(sigmas, video_state, audio_state, stepper, preview_tools=None):
        return euler_denoising_loop(sigmas=sigmas, video_state=video_state, audio_state=audio_state, stepper=stepper, denoise_fn=denoise_fn, interrupt_check=interrupt_check, callback=callback, preview_tools=preview_tools, transformer=transformer)

    if set_progress_status is not None:
        set_progress_status("Converting SDR to HDR")
    if callback is not None:
        callback(-1, None, True, override_num_inference_steps=len(sigmas) - 1)
    # The IC-LoRA was trained with a RoPE clock of at most 30 fps; only the positions change, not the output frame rate.
    video_state, _ = denoise_audio_video(
        output_shape=VideoPixelShape(batch=1, frames=num_frames, height=height, width=width, fps=min(float(frame_rate), MAX_CONDITIONING_FPS)),
        conditionings=conditionings,
        noiser=GaussianNoiser(generator=torch.Generator(device=device).manual_seed(seed)),
        sigmas=sigmas,
        stepper=EulerDiffusionStep(),
        denoising_loop_fn=denoising_loop,
        components=components,
        dtype=dtype,
        device=device,
        noise_scale=float(sigmas[0]),
        skip_audio=True,
    )
    conditionings = denoise_fn = None
    if video_state is None or interrupt_check is not None and interrupt_check():
        return None
    keyframes = None
    if video_state.generated_keyframes is not None and getattr(video_decoder, "is_diffusion_decoder", False):
        keyframes = DecodeKeyframes(video_state.generated_keyframes, video_state.generated_keyframe_layout.pixel_frame_indices)
    cleanup_memory()

    if set_progress_status is not None:
        set_progress_status("VAE Decoding")
    latent = [video_state.latent]
    video_state = None
    return vae_decode_video_to_tensor(latent, video_decoder, tiling_config, expected_frames=num_frames, expected_height=height, expected_width=width, interrupt_check=interrupt_check, hdr_transform=HDR_TRANSFORM, output_dtype=torch.float16, generator=torch.Generator(device=device).manual_seed(seed), keyframes=keyframes)
