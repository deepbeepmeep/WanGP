from shared.utils.phase_progress import control_video_encoding
import gc
import inspect
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import partial

import torch
import torch.nn.functional as F
from tqdm import tqdm

from mmgp import offload

from ...ltx_core.components.diffusion_steps import Res2sDiffusionStep
from ...ltx_core.components.noisers import Noiser
from ...ltx_core.components.protocols import DiffusionStepProtocol, GuiderProtocol
from ...ltx_core.conditioning import (
    AudioConditionByAppendedReferenceLatent,
    ConditioningItem,
    VideoConditionByKeyframeIndex,
    VideoConditionByLatentIndex,
    VideoConditionByReferenceLatent,
)
from ...ltx_core.guidance.perturbations import (
    BatchedPerturbationConfig,
    Perturbation,
    PerturbationConfig,
    PerturbationType,
)
from ...ltx_core.components.guiders import MultiModalGuider
from ...ltx_core.model.transformer import Modality, X0Model
from ...ltx_core.model.video_vae import VideoEncoder, TilingConfig, encode_video as vae_encode_video
from ...ltx_core.text_encoders.gemma import GemmaTextEncoderModelBase
from ...ltx_core.tools import AudioLatentTools, LatentTools, VideoLatentTools
from ...ltx_core.types import AudioLatentShape, LatentState, TimestepCompressionPlan, VideoLatentShape, VideoPixelShape
from ...ltx_core.utils import to_denoised, to_velocity
from .media_io import decode_image, load_image_conditioning, load_video_conditioning, resize_aspect_ratio_preserving
from .res2s import get_res2s_coefficients
from .types import (
    DenoisingFunc,
    DenoisingLoopFunc,
    PipelineComponents,
)
from shared.utils.self_refiner import run_refinement_loop_multi


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cleanup_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def image_conditionings_by_replacing_latent(
    images: list[tuple],
    height: int,
    width: int,
    video_encoder: VideoEncoder,
    dtype: torch.dtype,
    device: torch.device,
    tiling_config: TilingConfig | None = None,
) -> list[ConditioningItem]:
    conditionings = []
    for image_entry in images:
        if len(image_entry) == 4:
            image_path, frame_idx, strength, resample = image_entry
        else:
            image_path, frame_idx, strength = image_entry
            resample = None
        image = load_image_conditioning(
            image_path=image_path,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            resample=resample,
        )
        encoded_image = vae_encode_video(image, video_encoder, tiling_config)
        conditionings.append(
            VideoConditionByLatentIndex(
                latent=encoded_image,
                strength=strength,
                latent_idx=frame_idx,
            )
        )

    return conditionings


def image_conditionings_by_adding_guiding_latent(
    images: list[tuple],
    height: int,
    width: int,
    video_encoder: VideoEncoder,
    dtype: torch.dtype,
    device: torch.device,
    tiling_config: TilingConfig | None = None,
) -> list[ConditioningItem]:
    conditionings = []
    for image_entry in images:
        literal_frame_index = len(image_entry) == 5 and image_entry[4]
        if len(image_entry) >= 4:
            image_path, frame_idx, strength, resample = image_entry[:4]
        else:
            image_path, frame_idx, strength = image_entry
            resample = None
        image = load_image_conditioning(
            image_path=image_path,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            resample=resample,
        )
        encoded_image = vae_encode_video(image, video_encoder, tiling_config)
        conditionings.append(
            VideoConditionByKeyframeIndex(keyframes=encoded_image, frame_idx=frame_idx, strength=strength, num_pixel_frames=1 if literal_frame_index else None, literal_frame_index=literal_frame_index)
        )
    return conditionings


def prefix_video_latent_conditioning(
    prefix: torch.Tensor,
    height: int,
    width: int,
    num_frames: int,
    video_encoder: VideoEncoder,
    dtype: torch.dtype,
    device: torch.device,
    stage_1: bool = False,
) -> VideoConditionByLatentIndex | None:
    """A saved latent run as a frozen prefix at the window's first latent.

    ``prefix`` is the sidecar's kept latents on the full-resolution latent
    grid (``[c, t, h, w]`` or batched ``[1, c, t, h, w]``). ``height`` /
    ``width`` are this stage's own output canvas: stage 2's is the
    full-resolution canvas (the prefix grid must match it exactly and is
    injected as-is) and stage 1's is the half-resolution canvas, whose
    latent grid is half the prefix's, so the prefix is bilinearly
    half-downsampled (the stage-2 upsampler carries the same content back
    up). The injected tokens carry a zero denoise mask, so they stay clean
    at every denoise step of the stage (the same mechanism that freezes a
    start image). Returns ``None`` — the caller keeps its pixel-image
    conditioning — when the prefix does not match the window's latent grid.
    """
    if prefix is None:
        return None
    p = prefix
    if p.dim() == 4:
        p = p.unsqueeze(0)
    if p.dim() != 5:
        return None
    z = int(getattr(video_encoder, "latent_channels", 0) or 0)
    t = (int(num_frames) - 1) // 8 + 1
    h = int(height) // 32
    w = int(width) // 32
    spatial_scale = 2 if stage_1 else 1
    # The prefix covers the first P latents of the window (P <= t); the
    # remaining latents are freshly denoised on from it.
    if z <= 0 or p.shape[1] != z or not 1 <= p.shape[2] <= t:
        return None
    if h <= 0 or w <= 0 or p.shape[3] != spatial_scale * h or p.shape[4] != spatial_scale * w:
        return None
    if stage_1:
        b, c, pt, h2, w2 = p.shape
        p = (
            F.interpolate(
                p.permute(0, 2, 1, 3, 4).reshape(b * pt, c, h2, w2),
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            .reshape(b, pt, c, h, w)
            .permute(0, 2, 1, 3, 4)
            .contiguous()
        )
    return VideoConditionByLatentIndex(
        latent=p.to(device=device, dtype=dtype),
        strength=1.0,
        latent_idx=0,
    )


def video_conditionings_by_keyframe(
    video_conditioning: list[tuple],
    height: int,
    width: int,
    num_frames: int,
    video_encoder: VideoEncoder,
    dtype: torch.dtype,
    device: torch.device,
    tiling_config: TilingConfig | None = None,
    continuous_conditioning_and_guide: bool = False,
) -> list[ConditioningItem]:
    conditionings = []
    for entry in video_conditioning:
        if len(entry) == 2:
            video_path, strength = entry
            frame_idx = 0
        elif len(entry) == 3:
            video_path, frame_idx, strength = entry
        else:
            raise ValueError("Video conditioning entries must be (video, strength) or (video, frame_idx, strength).")
        video = load_video_conditioning(
            video_path=video_path,
            height=height,
            width=width,
            frame_cap=num_frames,
            dtype=dtype,
            device=device,
        )
        # remove_prepend = False
        # if frame_idx < 0:
        #     remove_prepend = True
        #     frame_idx = -frame_idx
        # if frame_idx < 0:
        #     encoded_video = vae_encode_video(video, video_encoder, tiling_config)
        #     encoded_video = encoded_video[:, :, 1:]
        #     frame_idx = -frame_idx + 1
        # else:
        #     encoded_video = vae_encode_video(video, video_encoder, tiling_config)

        encoded_video = vae_encode_video(video, video_encoder, tiling_config, device=device)
        if continuous_conditioning_and_guide and frame_idx < 0:
            split_frame = -int(frame_idx)
            latent_stride = int(getattr(getattr(video_encoder, "video_downscale_factors", None), "time", 8))
            split_latent = _pixel_to_latent_index(split_frame, latent_stride)
            if split_latent > 0:
                conditionings += latent_conditionings_by_latent_sequence(
                    encoded_video[:, :, :split_latent], strength=strength, start_index=0
                )
            if split_latent < encoded_video.shape[2]:
                conditionings.append(
                    VideoConditionByKeyframeIndex(
                        keyframes=encoded_video[:, :, split_latent:],
                        frame_idx=split_frame,
                        strength=strength,
                    )
                )
            continue
        cond = VideoConditionByKeyframeIndex(
            keyframes=encoded_video,
            frame_idx=frame_idx,
            strength=strength,
        )
        conditionings.append(cond)

    return conditionings


def video_conditionings_by_reference_latent(
    video_conditioning: list[tuple],
    height: int,
    width: int,
    num_frames: int,
    video_encoder: VideoEncoder,
    dtype: torch.dtype,
    device: torch.device,
    downscale_factor: int = 1,
    tiling_config: TilingConfig | None = None,
) -> list[ConditioningItem]:
    scale = max(1, int(downscale_factor))
    if scale > 1 and (height % scale != 0 or width % scale != 0):
        raise ValueError(f"Output dimensions ({height}x{width}) must be divisible by reference downscale factor {scale}.")

    ref_height = height // scale
    ref_width = width // scale
    conditionings = []
    for entry in video_conditioning:
        if len(entry) == 2:
            video_path, strength = entry
            frame_idx = 0
        elif len(entry) == 3:
            video_path, frame_idx, strength = entry
        else:
            raise ValueError("Video conditioning entries must be (video, strength) or (video, frame_idx, strength).")
        video = load_video_conditioning(
            video_path=video_path,
            height=ref_height,
            width=ref_width,
            frame_cap=num_frames,
            dtype=dtype,
            device=device,
        )
        encoded_video = vae_encode_video(video, video_encoder, tiling_config, device=device)
        conditionings.append(
            VideoConditionByReferenceLatent(
                latent=encoded_video,
                frame_idx=frame_idx,
                strength=strength,
                downscale_factor=scale,
            )
        )
    return conditionings


def video_conditionings_by_frozen_video(
    video: torch.Tensor,
    height: int,
    width: int,
    num_frames: int,
    video_encoder: VideoEncoder,
    dtype: torch.dtype,
    device: torch.device,
    tiling_config: TilingConfig | None = None,
) -> list[ConditioningItem]:
    video = load_video_conditioning(
        video_path=video,
        height=height,
        width=width,
        frame_cap=num_frames,
        dtype=dtype,
        device=device,
    )
    encoded_video = vae_encode_video(video, video_encoder, tiling_config, device=device)
    return latent_conditionings_by_latent_sequence(encoded_video, strength=1.0, start_index=0)


def video_conditionings_by_control_video(
    video_conditioning: list[tuple],
    height: int,
    width: int,
    num_frames: int,
    video_encoder: VideoEncoder,
    dtype: torch.dtype,
    device: torch.device,
    downscale_factor: int = 1,
    tiling_config: TilingConfig | None = None,
    continuous_conditioning_and_guide: bool = False,
) -> list[ConditioningItem]:
    from ...msr import MSRReferenceImages

    if isinstance(video_conditioning, MSRReferenceImages):
        return video_conditioning.encode(height, width, video_encoder, dtype, device, tiling_config)
    with control_video_encoding():
        if int(downscale_factor or 1) > 1:
            return video_conditionings_by_reference_latent(
                video_conditioning=video_conditioning,
                height=height,
                width=width,
                num_frames=num_frames,
                video_encoder=video_encoder,
                dtype=dtype,
                device=device,
                downscale_factor=downscale_factor,
                tiling_config=tiling_config,
            )
        return video_conditionings_by_keyframe(
            video_conditioning=video_conditioning,
            height=height,
            width=width,
            num_frames=num_frames,
            video_encoder=video_encoder,
            dtype=dtype,
            device=device,
            tiling_config=tiling_config,
            continuous_conditioning_and_guide=continuous_conditioning_and_guide,
        )


def latent_conditionings_by_latent_sequence(
    latents: torch.Tensor,
    strength: float = 1.0,
    start_index: int = 0,
) -> list[ConditioningItem]:
    if latents.dim() == 4:
        latents = latents.unsqueeze(0)
    if latents.dim() != 5:
        raise ValueError(f"Expected latent tensor with 5 dimensions; got {latents.shape}.")
    if latents.shape[2] == 0:
        return []
    conditionings = []
    for latent_idx in range(latents.shape[2]):
        conditionings.append(
            VideoConditionByLatentIndex(
                latent=latents[:, :, latent_idx : latent_idx + 1],
                strength=strength,
                latent_idx=start_index + latent_idx,
            )
        )
    return conditionings


@dataclass(frozen=True)
class MaskInjection:
    mask_tokens: torch.Tensor
    source_tokens: torch.Tensor
    noise_tokens: torch.Tensor
    token_slice: slice
    masked_steps: int


@dataclass(frozen=True)
class AVCrossAttentionMasks:
    video_cross_attention_mask: torch.Tensor | None = None
    audio_cross_attention_mask: torch.Tensor | None = None


CrossAttentionMaskBuilder = Callable[[LatentState, LatentState | None], AVCrossAttentionMasks | None]


def _slot_ranges(total_seq_len: int, num_slots: int) -> list[tuple[int, int]]:
    if total_seq_len <= 0 or num_slots <= 0:
        return []
    ranges = []
    start = 0
    for slot_idx in range(num_slots):
        end = round((slot_idx + 1) * total_seq_len / num_slots)
        if end > start:
            ranges.append((start, end))
        start = end
    return ranges


def _slot_ranges_from_lengths(lengths: tuple[int, ...] | None, *, total_seq_len: int, num_slots: int) -> list[tuple[int, int]]:
    if not lengths or len(lengths) != num_slots:
        return _slot_ranges(total_seq_len, num_slots)
    ranges = []
    start = 0
    for raw_length in lengths:
        length = max(0, int(raw_length))
        end = min(start + length, total_seq_len)
        if end > start:
            ranges.append((start, end))
        start = end
    if start != total_seq_len:
        return _slot_ranges(total_seq_len, num_slots)
    return ranges


def _build_paired_tail_cross_mask(
    *,
    batch_size: int,
    query_prefix_seq_len: int,
    query_memory_seq_len: int,
    kv_prefix_seq_len: int,
    kv_memory_seq_len: int,
    num_memory_slots: int,
    device: torch.device,
    query_segment_lengths: tuple[tuple[int, ...], ...] | None = None,
    kv_segment_lengths: tuple[tuple[int, ...], ...] | None = None,
) -> torch.Tensor:
    query_total_seq_len = query_prefix_seq_len + query_memory_seq_len
    kv_total_seq_len = kv_prefix_seq_len + kv_memory_seq_len
    mask = torch.zeros(batch_size, query_total_seq_len, kv_total_seq_len, dtype=torch.bool, device=device)
    if query_prefix_seq_len > 0 and kv_prefix_seq_len > 0:
        mask[:, :query_prefix_seq_len, :kv_prefix_seq_len] = True
    for batch_idx in range(batch_size):
        query_lengths = query_segment_lengths[batch_idx] if query_segment_lengths is not None and batch_idx < len(query_segment_lengths) else None
        kv_lengths = kv_segment_lengths[batch_idx] if kv_segment_lengths is not None and batch_idx < len(kv_segment_lengths) else None
        query_ranges = _slot_ranges_from_lengths(query_lengths, total_seq_len=query_memory_seq_len, num_slots=num_memory_slots)
        kv_ranges = _slot_ranges_from_lengths(kv_lengths, total_seq_len=kv_memory_seq_len, num_slots=num_memory_slots)
        for (q_start, q_end), (k_start, k_end) in zip(query_ranges, kv_ranges, strict=False):
            mask[batch_idx, query_prefix_seq_len + q_start : query_prefix_seq_len + q_end, kv_prefix_seq_len + k_start : kv_prefix_seq_len + k_end] = True
    return mask


def build_paired_tail_cross_attention_mask_builder(
    *,
    video_memory_seq_len: int,
    audio_memory_seq_len: int,
    num_memory_slots: int,
    audio_segment_lengths: tuple[tuple[int, ...], ...] | None = None,
) -> CrossAttentionMaskBuilder | None:
    if video_memory_seq_len <= 0 or audio_memory_seq_len <= 0 or num_memory_slots <= 0:
        return None

    def build(video_state: LatentState, audio_state: LatentState | None) -> AVCrossAttentionMasks | None:
        if audio_state is None:
            return None
        video_total = int(video_state.latent.shape[1])
        audio_total = int(audio_state.latent.shape[1])
        if video_total <= video_memory_seq_len or audio_total <= audio_memory_seq_len:
            return None
        batch_size = int(video_state.latent.shape[0])
        device = video_state.latent.device
        video_prefix = video_total - int(video_memory_seq_len)
        audio_prefix = audio_total - int(audio_memory_seq_len)
        video_cross_attention_mask = _build_paired_tail_cross_mask(
            batch_size=batch_size,
            query_prefix_seq_len=video_prefix,
            query_memory_seq_len=int(video_memory_seq_len),
            kv_prefix_seq_len=audio_prefix,
            kv_memory_seq_len=int(audio_memory_seq_len),
            num_memory_slots=int(num_memory_slots),
            device=device,
            kv_segment_lengths=audio_segment_lengths,
        )
        audio_cross_attention_mask = _build_paired_tail_cross_mask(
            batch_size=batch_size,
            query_prefix_seq_len=audio_prefix,
            query_memory_seq_len=int(audio_memory_seq_len),
            kv_prefix_seq_len=video_prefix,
            kv_memory_seq_len=int(video_memory_seq_len),
            num_memory_slots=int(num_memory_slots),
            device=device,
            query_segment_lengths=audio_segment_lengths,
        )
        return AVCrossAttentionMasks(
            video_cross_attention_mask=video_cross_attention_mask,
            audio_cross_attention_mask=audio_cross_attention_mask,
        )

    return build


def paired_reference_conditionings_by_latents(
    *,
    video_latent: torch.Tensor | None,
    audio_latent: torch.Tensor | None,
    audio_segment_lengths: tuple[tuple[int, ...], ...] | None,
    components: PipelineComponents,
    dtype: torch.dtype,
    device: torch.device,
    video_downscale_factor: int = 1,
    use_paired_cross_attention_mask: bool = False,
    audio_target_to_reference: bool = True,
) -> tuple[list[ConditioningItem], list[ConditioningItem], CrossAttentionMaskBuilder | None]:
    video_conditionings: list[ConditioningItem] = []
    audio_conditionings: list[ConditioningItem] = []
    cross_attention_mask_builder = None
    video_memory_seq_len = audio_memory_seq_len = num_memory_slots = 0
    if video_latent is not None:
        video_latent = video_latent.to(device=device, dtype=dtype)
        video_shape = VideoLatentShape.from_torch_shape(video_latent.shape)
        num_memory_slots = int(video_shape.frames)
        video_memory_seq_len = int(components.video_patchifier.get_token_count(video_shape))
        video_conditionings.append(VideoConditionByReferenceLatent(latent=video_latent, strength=1.0, frame_idx=0, downscale_factor=video_downscale_factor))
    if audio_latent is not None:
        audio_latent = audio_latent.to(device=device, dtype=dtype)
        audio_shape = AudioLatentShape.from_torch_shape(audio_latent.shape)
        audio_memory_seq_len = int(components.audio_patchifier.get_token_count(audio_shape))
        audio_conditionings.append(AudioConditionByAppendedReferenceLatent(latent=audio_latent, strength=1.0, target_to_reference=audio_target_to_reference))
    if use_paired_cross_attention_mask:
        cross_attention_mask_builder = build_paired_tail_cross_attention_mask_builder(
            video_memory_seq_len=video_memory_seq_len,
            audio_memory_seq_len=audio_memory_seq_len,
            num_memory_slots=num_memory_slots,
            audio_segment_lengths=audio_segment_lengths,
        )
    return video_conditionings, audio_conditionings, cross_attention_mask_builder


def _pixel_to_latent_index(frame_idx: int, stride: int) -> int:
    if frame_idx <= 0:
        return 0
    return (frame_idx - 1) // stride + 1


def _coerce_mask_tensor(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim == 5:
        if mask.shape[1] in (1, 3, 4):
            return mask[:, :1]
        if mask.shape[-1] in (1, 3, 4):
            return mask.permute(0, 4, 1, 2, 3)[:, :1]
    elif mask.ndim == 4:
        if mask.shape[0] in (1, 3, 4):
            return mask.unsqueeze(0)[:, :1]
        if mask.shape[-1] in (1, 3, 4):
            return mask.permute(3, 0, 1, 2).unsqueeze(0)[:, :1]
        return mask.unsqueeze(1)
    elif mask.ndim == 3:
        if mask.shape[-1] in (1, 3, 4):
            return mask.permute(2, 0, 1).unsqueeze(0).unsqueeze(2)[:, :1]
        if mask.shape[0] in (1, 3, 4):
            return mask.unsqueeze(0).unsqueeze(2)[:, :1]
        return mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 2:
        return mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    raise ValueError(f"Unsupported mask tensor shape: {tuple(mask.shape)}")


def _normalize_mask_values(mask: torch.Tensor) -> torch.Tensor:
    mask = mask.float()
    if mask.min() < 0.0:
        mask = (mask + 1.0) * 0.5
    elif mask.max() > 1.0:
        mask = mask / 255.0
    return mask.clamp(0.0, 1.0)


def _resize_mask_spatial(mask: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if mask.shape[3] == height and mask.shape[4] == width:
        return mask
    return F.interpolate(mask, size=(mask.shape[2], height, width), mode="nearest")


def _mask_to_latents(mask: torch.Tensor, target_frames: int, target_h: int, target_w: int) -> torch.Tensor:
    if target_frames <= 0 or mask.shape[2] == 0:
        raise ValueError("Mask has no frames to map into latent space.")
    if mask.shape[2] == 1:
        mask = F.interpolate(mask, size=(1, target_h, target_w), mode="nearest")
        if target_frames > 1:
            mask = mask.expand(-1, -1, target_frames, -1, -1)
        return mask
    if target_frames == 1:
        return F.interpolate(mask[:, :, :1], size=(1, target_h, target_w), mode="nearest")
    first = F.interpolate(mask[:, :, :1], size=(1, target_h, target_w), mode="nearest")
    rest = mask[:, :, 1:]
    if rest.shape[2] == 0:
        rest = torch.ones(
            (mask.shape[0], 1, target_frames - 1, target_h, target_w),
            device=mask.device,
            dtype=mask.dtype,
        )
    else:
        rest = F.interpolate(rest, size=(target_frames - 1, target_h, target_w), mode="nearest")
    return torch.cat([first, rest], dim=2)


def prepare_mask_injection(  # noqa: PLR0913
    masking_source: dict | None,
    masking_strength: float | None,
    output_shape: VideoPixelShape,
    video_encoder: VideoEncoder,
    components: PipelineComponents,
    dtype: torch.dtype,
    device: torch.device,
    tiling_config: TilingConfig | None,
    generator: torch.Generator,
    num_steps: int,
) -> MaskInjection | None:
    if masking_source is None:
        return None
    try:
        strength = float(masking_strength or 0.0)
    except (TypeError, ValueError):
        return None
    strength = max(0.0, min(1.0, strength))
    if strength <= 0.0 or num_steps <= 0:
        return None
    masked_steps = min(num_steps, int(math.ceil(num_steps * strength)))
    if masked_steps <= 0:
        return None

    video = masking_source.get("video")
    mask = masking_source.get("mask")
    if video is None or mask is None:
        return None
    start_frame = int(masking_source.get("start_frame") or 0)

    video_tensor = load_video_conditioning(
        video_path=video,
        height=output_shape.height,
        width=output_shape.width,
        frame_cap=None,
        dtype=dtype,
        device=device,
    )

    mask_tensor = _coerce_mask_tensor(mask).to(device=device)
    mask_tensor = _normalize_mask_values(mask_tensor)
    if mask_tensor.shape[0] != video_tensor.shape[0]:
        if mask_tensor.shape[0] == 1:
            mask_tensor = mask_tensor.expand(video_tensor.shape[0], -1, -1, -1, -1)
        else:
            return None
    mask_tensor = _resize_mask_spatial(mask_tensor, output_shape.height, output_shape.width)
    if mask_tensor.shape[2] < video_tensor.shape[2]:
        pad_frames = video_tensor.shape[2] - mask_tensor.shape[2]
        pad = torch.ones(
            (mask_tensor.shape[0], 1, pad_frames, mask_tensor.shape[3], mask_tensor.shape[4]),
            device=mask_tensor.device,
            dtype=mask_tensor.dtype,
        )
        mask_tensor = torch.cat([mask_tensor, pad], dim=2)
    elif mask_tensor.shape[2] > video_tensor.shape[2]:
        mask_tensor = mask_tensor[:, :, : video_tensor.shape[2]]
    if video_tensor.shape[2] == 0 or mask_tensor.shape[2] == 0:
        return None

    source_latents = vae_encode_video(video_tensor, video_encoder, tiling_config, device=device).to(device=device, dtype=dtype)
    try:
        mask_latents = _mask_to_latents(
            mask_tensor, source_latents.shape[2], source_latents.shape[3], source_latents.shape[4]
        )
    except ValueError:
        return None
    mask_latents = (mask_latents >= 0.5).to(dtype)

    output_latent_shape = VideoLatentShape.from_pixel_shape(
        shape=output_shape,
        latent_channels=components.video_latent_channels,
        scale_factors=components.video_scale_factors,
    )
    start_latent = _pixel_to_latent_index(start_frame, components.video_scale_factors.time)
    if start_latent >= output_latent_shape.frames:
        return None
    available_frames = output_latent_shape.frames - start_latent
    control_frames = min(source_latents.shape[2], available_frames)
    if control_frames <= 0:
        return None
    source_latents = source_latents[:, :, :control_frames]
    mask_latents = mask_latents[:, :, :control_frames]

    source_tokens = components.video_patchifier.patchify(source_latents)
    mask_tokens = components.video_patchifier.patchify(mask_latents).to(dtype=source_tokens.dtype)
    noise_tokens = torch.randn(
        source_tokens.shape,
        device=source_tokens.device,
        dtype=source_tokens.dtype,
        generator=generator,
    )

    patch_t, patch_h, patch_w = components.video_patchifier.patch_size
    if patch_t != 1:
        raise ValueError("Mask injection expects temporal patch size of 1.")
    tokens_per_frame = (output_latent_shape.height // patch_h) * (output_latent_shape.width // patch_w)
    token_offset = start_latent * tokens_per_frame
    token_count = control_frames * tokens_per_frame
    token_slice = slice(token_offset, token_offset + token_count)

    return MaskInjection(
        mask_tokens=mask_tokens,
        source_tokens=source_tokens,
        noise_tokens=noise_tokens,
        token_slice=token_slice,
        masked_steps=masked_steps,
    )


def _apply_mask_injection(
    video_state: LatentState,
    sigmas: torch.Tensor,
    step_idx: int,
    mask_context: MaskInjection,
) -> None:
    if step_idx >= mask_context.masked_steps:
        return
    sigma_next = sigmas[step_idx + 1].to(mask_context.source_tokens.dtype)
    token_slice = mask_context.token_slice
    current = video_state.latent[:, token_slice]
    noisy_source = mask_context.noise_tokens * sigma_next + (1 - sigma_next) * mask_context.source_tokens
    video_state.latent[:, token_slice] = noisy_source * (1 - mask_context.mask_tokens) + mask_context.mask_tokens * current


def euler_denoising_loop(
    sigmas: torch.Tensor,
    video_state: LatentState,
    audio_state: LatentState,
    stepper: DiffusionStepProtocol,
    denoise_fn: DenoisingFunc,
    *,
    mask_context: MaskInjection | None = None,
    interrupt_check: Callable[[], bool] | None = None,
    callback: Callable[..., None] | None = None,
    preview_tools: VideoLatentTools | None = None,
    pass_no: int = 0,
    transformer=None,
    self_refiner_handler=None,
    self_refiner_handler_audio=None,
    self_refiner_generator: torch.Generator | None = None,
    ancestral_noise_generator: torch.Generator | None = None,
) -> tuple[LatentState | None, LatentState | None]:
    """
    Perform the joint audio-video denoising loop over a diffusion schedule.
    This function iterates over all but the final value in ``sigmas`` and, at
    each diffusion step, calls ``denoise_fn`` to obtain denoised video and
    audio latents. The denoised latents are post-processed with their
    respective denoise masks and clean latents, then passed to ``stepper`` to
    advance the noisy latents one step along the diffusion schedule.
    ### Parameters
    sigmas:
        A 1D tensor of noise levels (diffusion sigmas) defining the sampling
        schedule. All steps except the last element are iterated over.
    video_state:
        The current video :class:`LatentState`, containing the noisy latent,
        its clean reference latent, and the denoising mask.
    audio_state:
        The current audio :class:`LatentState`, analogous to ``video_state``
        but for the audio modality.
    stepper:
        An implementation of :class:`DiffusionStepProtocol` that updates a
        latent given the current latent, its denoised estimate, the full
        ``sigmas`` schedule, and the current step index.
    denoise_fn:
        A callable implementing :class:`DenoisingFunc`. It is invoked as
        ``denoise_fn(video_state, audio_state, sigmas, step_index)`` and must
        return a tuple ``(denoised_video, denoised_audio)``, where each element
        is a tensor with the same shape as the corresponding latent.
    ### Returns
    tuple[LatentState, LatentState]
        A pair ``(video_state, audio_state)`` containing the final video and
        audio latent states after completing the denoising loop.
    """
    prewarm = getattr(denoise_fn, "_prewarm", None)
    cleanup = getattr(denoise_fn, "_cleanup", None)
    if callable(prewarm):
        prewarm(video_state, audio_state, sigmas)

    def advance(state: LatentState, denoised: torch.Tensor, step_idx: int, unconditional_denoised: torch.Tensor | None = None) -> torch.Tensor:
        if getattr(stepper, "uses_unconditional", False):
            if unconditional_denoised is None:
                raise RuntimeError("CFG++ diffusion requires an unconditional denoised prediction")
            return stepper.step(state.latent, denoised, sigmas, step_idx, unconditional_denoised_sample=unconditional_denoised)
        if ancestral_noise_generator is None:
            return stepper.step(state.latent, denoised, sigmas, step_idx)
        noise = torch.randn(state.latent.shape, generator=ancestral_noise_generator, device=state.latent.device, dtype=state.latent.dtype)
        latent = stepper.step(state.latent, denoised, sigmas, step_idx, noise=noise)
        return post_process_latent(latent, state.denoise_mask, state.clean_latent)

    try:
        for step_idx, _ in enumerate(tqdm(sigmas[:-1])):
            if interrupt_check is not None and interrupt_check():
                return None, None

            offload.set_step_no_for_lora(transformer, step_idx)
            denoised_video = denoised_audio = None
            denoised_video, denoised_audio = denoise_fn(video_state, audio_state, sigmas, step_idx)
            if denoised_video is None or (audio_state is not None and denoised_audio is None):
                return None, None

            denoised_video = post_process_latent(denoised_video, video_state.denoise_mask, video_state.clean_latent)
            if audio_state is not None:
                denoised_audio = post_process_latent(denoised_audio, audio_state.denoise_mask, audio_state.clean_latent)
            unconditional_video = unconditional_audio = None
            if getattr(stepper, "uses_unconditional", False):
                unconditional_video, unconditional_audio = denoise_fn.unconditional_predictions
                unconditional_video = post_process_latent(unconditional_video, video_state.denoise_mask, video_state.clean_latent)
                if audio_state is not None:
                    unconditional_audio = post_process_latent(unconditional_audio, audio_state.denoise_mask, audio_state.clean_latent)

            refiner_steps = 0
            if self_refiner_handler is not None:
                self_refiner_handler.reset_buffer()
                refiner_steps = self_refiner_handler.get_anneal_steps(step_idx)
            if self_refiner_handler_audio is not None:
                self_refiner_handler_audio.reset_buffer()
                if refiner_steps == 0:
                    refiner_steps = self_refiner_handler_audio.get_anneal_steps(step_idx)

            use_audio_refiner = (
                self_refiner_handler is not None
                and self_refiner_handler_audio is not None
                and denoised_audio is not None
                and audio_state is not None
            )

            if use_audio_refiner and refiner_steps > 1:
                current_sigma = float(sigmas[step_idx].item()) if torch.is_tensor(sigmas[step_idx]) else float(sigmas[step_idx])
                next_sigma = float(sigmas[step_idx + 1].item()) if step_idx + 1 < len(sigmas) else 0.0
                refine_failed = False

                def denoise_multi(latents_list):
                    nonlocal refine_failed
                    temp_video_state = replace(video_state, latent=latents_list[0])
                    temp_audio_state = replace(audio_state, latent=latents_list[1])
                    denoised_video_loop, denoised_audio_loop = denoise_fn(temp_video_state, temp_audio_state, sigmas, step_idx)
                    if denoised_video_loop is None and denoised_audio_loop is None:
                        refine_failed = True
                        return [denoised_video, denoised_audio]
                    if denoised_video_loop is None or denoised_audio_loop is None:
                        refine_failed = True
                        return [denoised_video, denoised_audio]
                    denoised_video_loop = post_process_latent(
                        denoised_video_loop, temp_video_state.denoise_mask, temp_video_state.clean_latent
                    )
                    denoised_audio_loop = post_process_latent(
                        denoised_audio_loop, temp_audio_state.denoise_mask, temp_audio_state.clean_latent
                    )
                    return [denoised_video_loop, denoised_audio_loop]

                def step_func(noise_preds, latents_list):
                    latents_next_video = stepper.step(latents_list[0], noise_preds[0], sigmas, step_idx)
                    latents_next_audio = stepper.step(latents_list[1], noise_preds[1], sigmas, step_idx)
                    return [latents_next_video, latents_next_audio], noise_preds

                refined_latents = run_refinement_loop_multi(
                    handlers=[self_refiner_handler, self_refiner_handler_audio],
                    latents_list=[video_state.latent, audio_state.latent],
                    noise_pred_list=[denoised_video, denoised_audio],
                    current_sigma=current_sigma,
                    next_sigma=next_sigma,
                    m_steps=refiner_steps,
                    denoise_func=denoise_multi,
                    step_func=step_func,
                    generators=[self_refiner_generator, self_refiner_generator],
                    devices=[video_state.latent.device, audio_state.latent.device],
                    noise_masks=[video_state.denoise_mask, audio_state.denoise_mask],
                )
                if refine_failed or refined_latents is None or any(latent is None for latent in refined_latents):
                    return None, None
                video_state = replace(video_state, latent=refined_latents[0])
                audio_state = replace(audio_state, latent=refined_latents[1])
            elif self_refiner_handler is not None and refiner_steps > 1:
                current_sigma = float(sigmas[step_idx].item()) if torch.is_tensor(sigmas[step_idx]) else float(sigmas[step_idx])
                next_sigma = float(sigmas[step_idx + 1].item()) if step_idx + 1 < len(sigmas) else 0.0
                denoised_audio_final = denoised_audio
                refine_failed = False

                def denoise_video(latents_in):
                    nonlocal denoised_audio_final, refine_failed
                    temp_video_state = replace(video_state, latent=latents_in)
                    denoised_video_loop, denoised_audio_loop = denoise_fn(temp_video_state, audio_state, sigmas, step_idx)
                    if denoised_video_loop is None and denoised_audio_loop is None:
                        refine_failed = True
                        return denoised_video
                    if denoised_video_loop is None:
                        refine_failed = True
                        return denoised_video
                    denoised_video_loop = post_process_latent(
                        denoised_video_loop, temp_video_state.denoise_mask, temp_video_state.clean_latent
                    )
                    if denoised_audio_loop is not None:
                        denoised_audio_loop = post_process_latent(
                            denoised_audio_loop, audio_state.denoise_mask, audio_state.clean_latent
                        )
                        denoised_audio_final = denoised_audio_loop
                    return denoised_video_loop

                def step_func(n_pred_in, latents_in):
                    latents_next = stepper.step(latents_in, n_pred_in, sigmas, step_idx)
                    return latents_next, n_pred_in

                latents_refined = self_refiner_handler.run_refinement_loop(
                    latents=video_state.latent,
                    noise_pred=denoised_video,
                    current_sigma=current_sigma,
                    next_sigma=next_sigma,
                    m_steps=refiner_steps,
                    denoise_func=denoise_video,
                    step_func=step_func,
                    generator=self_refiner_generator,
                    device=video_state.latent.device,
                    noise_mask=video_state.denoise_mask,
                )
                if refine_failed or latents_refined is None:
                    return None, None
                video_state = replace(video_state, latent=latents_refined)
                if audio_state is not None:
                    audio_state = replace(audio_state, latent=stepper.step(audio_state.latent, denoised_audio_final, sigmas, step_idx))
            else:
                video_state = replace(video_state, latent=advance(video_state, denoised_video, step_idx, unconditional_video))
                if audio_state is not None:
                    audio_state = replace(audio_state, latent=advance(audio_state, denoised_audio, step_idx, unconditional_audio))

            if mask_context is not None:
                _apply_mask_injection(video_state, sigmas, step_idx, mask_context)
            preview_state = replace(video_state, latent=denoised_video) if getattr(callback, "tiny_vae", False) else video_state
            _invoke_callback(callback, step_idx, pass_no, preview_state, preview_tools)
            del preview_state

        return video_state, audio_state
    finally:
        if callable(cleanup):
            cleanup()


def gradient_estimating_euler_denoising_loop(
    sigmas: torch.Tensor,
    video_state: LatentState,
    audio_state: LatentState,
    stepper: DiffusionStepProtocol,
    denoise_fn: DenoisingFunc,
    ge_gamma: float = 2.0,
    *,
    mask_context: MaskInjection | None = None,
    interrupt_check: Callable[[], bool] | None = None,
    callback: Callable[..., None] | None = None,
    preview_tools: VideoLatentTools | None = None,
    pass_no: int = 0,
) -> tuple[LatentState | None, LatentState | None]:
    """
    Perform the joint audio-video denoising loop using gradient-estimation sampling.
    This function is similar to :func:`euler_denoising_loop`, but applies
    gradient estimation to improve the denoised estimates by tracking velocity
    changes across steps. See the referenced function for detailed parameter
    documentation.
    ### Parameters
    ge_gamma:
        Gradient estimation coefficient controlling the velocity correction term.
        Default is 2.0. Paper: https://openreview.net/pdf?id=o2ND9v0CeK
    sigmas, video_state, audio_state, stepper, denoise_fn:
        See :func:`euler_denoising_loop` for parameter descriptions.
    ### Returns
    tuple[LatentState, LatentState]
        See :func:`euler_denoising_loop` for return value description.
    """

    previous_audio_velocity = None
    previous_video_velocity = None
    cleanup = getattr(denoise_fn, "_cleanup", None)

    def update_velocity_and_sample(
        noisy_sample: torch.Tensor, denoised_sample: torch.Tensor, sigma: float, previous_velocity: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        current_velocity = to_velocity(noisy_sample, sigma, denoised_sample)
        if previous_velocity is not None:
            delta_v = current_velocity - previous_velocity
            total_velocity = ge_gamma * delta_v + previous_velocity
            denoised_sample = to_denoised(noisy_sample, total_velocity, sigma)
        return current_velocity, denoised_sample

    try:
        for step_idx, _ in enumerate(tqdm(sigmas[:-1])):
            if interrupt_check is not None and interrupt_check():
                return None, None
            denoised_video, denoised_audio = denoise_fn(video_state, audio_state, sigmas, step_idx)
            if denoised_video is None or denoised_audio is None:
                return None, None

            denoised_video = post_process_latent(denoised_video, video_state.denoise_mask, video_state.clean_latent)
            denoised_audio = post_process_latent(denoised_audio, audio_state.denoise_mask, audio_state.clean_latent)

            if sigmas[step_idx + 1] == 0:
                _invoke_callback(
                    callback,
                    step_idx,
                    pass_no,
                    replace(video_state, latent=denoised_video),
                    preview_tools,
                )
                return replace(video_state, latent=denoised_video), replace(audio_state, latent=denoised_audio)

            previous_video_velocity, denoised_video = update_velocity_and_sample(
                video_state.latent, denoised_video, sigmas[step_idx], previous_video_velocity
            )
            previous_audio_velocity, denoised_audio = update_velocity_and_sample(
                audio_state.latent, denoised_audio, sigmas[step_idx], previous_audio_velocity
            )

            video_state = replace(video_state, latent=stepper.step(video_state.latent, denoised_video, sigmas, step_idx))
            audio_state = replace(audio_state, latent=stepper.step(audio_state.latent, denoised_audio, sigmas, step_idx))
            if mask_context is not None:
                _apply_mask_injection(video_state, sigmas, step_idx, mask_context)
            preview_state = replace(video_state, latent=denoised_video) if getattr(callback, "tiny_vae", False) else video_state
            _invoke_callback(callback, step_idx, pass_no, preview_state, preview_tools)
            del preview_state

        return video_state, audio_state
    finally:
        if callable(cleanup):
            cleanup()


def noise_video_state(
    output_shape: VideoPixelShape,
    noiser: Noiser,
    conditionings: list[ConditioningItem],
    components: PipelineComponents,
    dtype: torch.dtype,
    device: torch.device,
    noise_scale: float = 1.0,
    initial_latent: torch.Tensor | None = None,
) -> tuple[LatentState, VideoLatentTools]:
    """Initialize and noise a video latent state for the diffusion pipeline.
    Creates a video latent state from the output shape, applies conditionings,
    and adds noise using the provided noiser. Returns the noised state and
    video latent tools for further processing. If initial_latent is provided, it will be used to create the initial
    state, otherwise an empty initial state will be created.
    """
    video_latent_shape = VideoLatentShape.from_pixel_shape(
        shape=output_shape,
        latent_channels=components.video_latent_channels,
        scale_factors=components.video_scale_factors,
    )
    video_tools = VideoLatentTools(components.video_patchifier, video_latent_shape, output_shape.fps)
    video_state = create_noised_state(
        tools=video_tools,
        conditionings=conditionings,
        noiser=noiser,
        dtype=dtype,
        device=device,
        noise_scale=noise_scale,
        initial_latent=initial_latent,
    )

    return video_state, video_tools


def bind_interrupt_check(transformer: object, interrupt_check: Callable[[], bool] | None) -> None:
    if interrupt_check is None or transformer is None:
        return
    target = getattr(transformer, "velocity_model", transformer)
    if hasattr(target, "interrupt_check"):
        target.interrupt_check = interrupt_check


def noise_audio_state(
    output_shape: VideoPixelShape,
    noiser: Noiser,
    conditionings: list[ConditioningItem],
    components: PipelineComponents,
    dtype: torch.dtype,
    device: torch.device,
    noise_scale: float = 1.0,
    initial_latent: torch.Tensor | None = None,
) -> tuple[LatentState, AudioLatentTools]:
    """Initialize and noise an audio latent state for the diffusion pipeline.
    Creates an audio latent state from the output shape, applies conditionings,
    and adds noise using the provided noiser. Returns the noised state and
    audio latent tools for further processing. If initial_latent is provided, it will be used to create the initial
    state, otherwise an empty initial state will be created.
    """
    audio_latent_shape = AudioLatentShape.from_video_pixel_shape(output_shape)
    audio_tools = AudioLatentTools(components.audio_patchifier, audio_latent_shape)
    audio_state = create_noised_state(
        tools=audio_tools,
        conditionings=conditionings,
        noiser=noiser,
        dtype=dtype,
        device=device,
        noise_scale=noise_scale,
        initial_latent=initial_latent,
    )

    return audio_state, audio_tools


def create_noised_state(
    tools: LatentTools,
    conditionings: list[ConditioningItem],
    noiser: Noiser,
    dtype: torch.dtype,
    device: torch.device,
    noise_scale: float = 1.0,
    initial_latent: torch.Tensor | None = None,
) -> LatentState:
    """Create a noised latent state from empty state, conditionings, and noiser.
    Creates an empty latent state, applies conditionings, and then adds noise
    using the provided noiser. Returns the final noised state ready for diffusion.
    """
    state = tools.create_initial_state(device, dtype, initial_latent)
    state = state_with_conditionings(state, conditionings, tools)
    state = noiser(state, noise_scale)

    return state


def state_with_conditionings(
    latent_state: LatentState, conditioning_items: list[ConditioningItem], latent_tools: LatentTools
) -> LatentState:
    """Apply a list of conditionings to a latent state.
    Iterates through the conditioning items and applies each one to the latent
    state in sequence. Returns the modified state with all conditionings applied.
    """
    for conditioning in conditioning_items:
        latent_state = conditioning.apply_to(latent_state=latent_state, latent_tools=latent_tools)

    return latent_state


def post_process_latent(denoised: torch.Tensor, denoise_mask: torch.Tensor, clean: torch.Tensor) -> torch.Tensor:
    """Blend denoised output with clean state based on mask."""
    return (denoised * denoise_mask + clean.float() * (1 - denoise_mask)).to(denoised.dtype)


def modality_from_latent_state(
    state: LatentState,
    context: torch.Tensor,
    sigma: float | torch.Tensor,
    enabled: bool = True,
    nag: dict | None = None,
    step_index: int | None = None,
    sigma_schedule: torch.Tensor | None = None,
    ref_context: torch.Tensor | None = None,
    ref_adaln: torch.Tensor | None = None,
    context_mask_builder: Callable[[LatentState, torch.Tensor | None, torch.Tensor], torch.Tensor | None] | None = None,
    cross_attention_mask: torch.Tensor | None = None,
) -> Modality:
    """Create a Modality from a latent state.
    Constructs a Modality object with the latent state's data, timesteps derived
    from the denoise mask and sigma, positions, and the provided context.
    """
    runtime_cache = state.runtime_cache
    if runtime_cache.timestep_plan is None:
        runtime_cache.timestep_plan = build_timestep_compression_plan(state.denoise_mask, state.positions)
    timesteps, frame_indices = timesteps_from_mask(state.denoise_mask, sigma, plan=runtime_cache.timestep_plan)
    sigma_tensor = sigma if torch.is_tensor(sigma) else torch.tensor(sigma, device=state.latent.device)
    sigma_tensor = sigma_tensor.to(device=state.latent.device, dtype=state.latent.dtype)
    if sigma_tensor.ndim == 0:
        sigma_tensor = sigma_tensor.expand(state.latent.shape[0])
    elif sigma_tensor.ndim == 1 and sigma_tensor.shape[0] == 1:
        sigma_tensor = sigma_tensor.expand(state.latent.shape[0])
    elif sigma_tensor.ndim > 1:
        sigma_tensor = sigma_tensor.reshape(state.latent.shape[0], -1)[:, 0]
    context_mask = context_mask_builder(state, frame_indices, context) if context_mask_builder is not None else None
    return Modality(
        enabled=enabled,
        latent=state.latent,
        sigma=sigma_tensor,
        timesteps=timesteps,
        positions=state.positions,
        context=context,
        ref_context=ref_context,
        ref_adaln=ref_adaln,
        nag=nag,
        context_mask=context_mask,
        attention_mask=state.attention_mask,
        cross_attention_mask=cross_attention_mask,
        frame_indices=frame_indices,
        keyframes_mask=state.keyframes_mask,
        runtime_cache=runtime_cache,
        step_index=step_index,
        sigma_schedule=sigma_schedule,
    )


def _get_batch_size(video_state: LatentState | None, audio_state: LatentState | None) -> int:
    if video_state is not None:
        return int(video_state.latent.shape[0])
    if audio_state is not None:
        return int(audio_state.latent.shape[0])
    return 1


def _cross_attn_perturbations(batch_size: int) -> BatchedPerturbationConfig:
    perts = [
        PerturbationConfig(
            [
                Perturbation(PerturbationType.SKIP_A2V_CROSS_ATTN, None),
                Perturbation(PerturbationType.SKIP_V2A_CROSS_ATTN, None),
            ]
        )
        for _ in range(batch_size)
    ]
    return BatchedPerturbationConfig(perts)


def _skip_audio_to_video_perturbations(batch_size: int) -> BatchedPerturbationConfig:
    perts = [
        PerturbationConfig([Perturbation(PerturbationType.SKIP_A2V_CROSS_ATTN, None)])
        for _ in range(batch_size)
    ]
    return BatchedPerturbationConfig(perts)


def _merge_perturbations(batch_size: int, *configs: BatchedPerturbationConfig | None) -> BatchedPerturbationConfig | None:
    merged = []
    any_perturbation = False
    for batch_idx in range(batch_size):
        perturbations = []
        for config in configs:
            if config is None or batch_idx >= len(config.perturbations):
                continue
            items = config.perturbations[batch_idx].perturbations or []
            perturbations.extend(items)
        any_perturbation = any_perturbation or bool(perturbations)
        merged.append(PerturbationConfig(perturbations))
    return BatchedPerturbationConfig(merged) if any_perturbation else None


PERTURBATION_perturbation = 1
PERTURBATION_SKIP_SELF_ATTENTION = 2


def _normalize_perturbation_layers(perturbation_layers) -> list[int] | None:
    if perturbation_layers is None:
        return None
    if isinstance(perturbation_layers, (list, tuple)):
        return [int(layer) for layer in perturbation_layers if str(layer).strip() != ""]
    if isinstance(perturbation_layers, (int, float)):
        return [int(perturbation_layers)]
    if isinstance(perturbation_layers, str):
        return [int(layer.strip()) for layer in perturbation_layers.split(",") if layer.strip()]
    return None


def _legacy_perturbation_layer_configs(
    batch_size: int, perturbation_layers: list[int] | None
) -> BatchedPerturbationConfig:
    perts = [
        PerturbationConfig(
            [
                Perturbation(PerturbationType.SKIP_VIDEO_SELF_ATTN, perturbation_layers),
                Perturbation(PerturbationType.SKIP_AUDIO_SELF_ATTN, perturbation_layers),
                Perturbation(PerturbationType.SKIP_A2V_CROSS_ATTN, perturbation_layers),
                Perturbation(PerturbationType.SKIP_V2A_CROSS_ATTN, perturbation_layers),
            ]
        )
        for _ in range(batch_size)
    ]
    return BatchedPerturbationConfig(perts)


def _self_attn_perturbation_configs(
    batch_size: int, perturbation_layers: list[int] | None
) -> BatchedPerturbationConfig:
    perts = [
        PerturbationConfig(
            [
                Perturbation(PerturbationType.SKIP_VIDEO_SELF_ATTN, perturbation_layers),
                Perturbation(PerturbationType.SKIP_AUDIO_SELF_ATTN, perturbation_layers),
            ]
        )
        for _ in range(batch_size)
    ]
    return BatchedPerturbationConfig(perts)


def _perturbation_active(
    step_index: int,
    sigmas: torch.Tensor,
    perturbation_start: float,
    perturbation_end: float,
) -> bool:
    total_steps = max(len(sigmas) - 1, 1)
    start_step = int(perturbation_start * total_steps)
    end_step = int(perturbation_end * total_steps)
    return start_step <= step_index < end_step


def _rescale_prediction(cond: torch.Tensor | None, pred: torch.Tensor | None, rescale_scale: float) -> torch.Tensor | None:
    if cond is None or pred is None or math.isclose(rescale_scale, 0.0):
        return pred
    pred_std = pred.std().clamp_min(1e-6)
    factor = cond.std() / pred_std
    factor = rescale_scale * factor + (1.0 - rescale_scale)
    return pred * factor


def build_timestep_compression_plan(
    denoise_mask: torch.Tensor,
    positions: torch.Tensor | None = None,
) -> TimestepCompressionPlan:
    if positions is None or positions.ndim < 4 or positions.shape[1] != 3:
        return TimestepCompressionPlan()

    token_mask = denoise_mask
    if token_mask.ndim > 2:
        token_mask = token_mask.mean(dim=-1)

    batch_size = token_mask.shape[0]
    frame_times = positions[:, 0, :, 0]
    run_lengths = None
    quantum = 0
    frame_masks = []
    frame_indices = []
    token_count = frame_times.shape[1]
    for b in range(batch_size):
        changes = torch.nonzero(frame_times[b, 1:] != frame_times[b, :-1]).flatten() + 1
        bounds = torch.cat([changes.new_zeros(1), changes, changes.new_full((1,), token_count)])
        lengths = (bounds[1:] - bounds[:-1]).tolist()
        if run_lengths is None:
            run_lengths = lengths
            quantum = run_lengths[0]
            for length in run_lengths[1:]:
                quantum = math.gcd(quantum, length)
        elif lengths != run_lengths:
            return TimestepCompressionPlan()

        group_values = []
        group_indices = []
        group_idx = 0
        for start, end, length in zip(bounds[:-1].tolist(), bounds[1:].tolist(), run_lengths):
            run_mask = token_mask[b, start:end]
            if not torch.allclose(run_mask, run_mask[:1], atol=1e-6):
                return TimestepCompressionPlan()
            repeat = length // quantum
            group_values.append(run_mask[:1].expand(repeat))
            group_indices.append(
                torch.arange(group_idx, group_idx + repeat, device=token_mask.device).repeat_interleave(quantum)
            )
            group_idx += repeat
        frame_masks.append(torch.cat(group_values))
        frame_indices.append(torch.cat(group_indices))

    return TimestepCompressionPlan(frame_mask=torch.stack(frame_masks, dim=0), frame_indices=torch.stack(frame_indices, dim=0))


def timesteps_from_mask(
    denoise_mask: torch.Tensor,
    sigma: float | torch.Tensor,
    positions: torch.Tensor | None = None,
    plan: TimestepCompressionPlan | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute timesteps from a denoise mask and sigma value."""
    if plan is None:
        plan = build_timestep_compression_plan(denoise_mask, positions)
    if plan.frame_mask is None or plan.frame_indices is None:
        return denoise_mask * sigma, None
    return plan.frame_mask * sigma, plan.frame_indices


def _prepare_conditioning_context(
    transformer: X0Model | None,
    state: LatentState,
    context: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    is_audio: bool,
):
    if transformer is None:
        return context
    velocity_model = getattr(transformer, "velocity_model", transformer)
    preprocessor_name = "audio_args_preprocessor" if is_audio else "video_args_preprocessor"
    preprocessor = getattr(velocity_model, preprocessor_name, None)
    if preprocessor is None or not hasattr(preprocessor, "build_prepared_conditioning"):
        return context
    seed_modality = modality_from_latent_state(state, context, sigmas[0], step_index=0, sigma_schedule=sigmas)
    return preprocessor.build_prepared_conditioning(seed_modality)


def _clear_phase_timestep_embedders(transformer: X0Model | None) -> None:
    if transformer is None:
        return
    velocity_model = getattr(transformer, "velocity_model", transformer)
    for preprocessor_name in ("video_args_preprocessor", "audio_args_preprocessor"):
        preprocessor = getattr(velocity_model, preprocessor_name, None)
        clear_fn = None if preprocessor is None else getattr(preprocessor, "clear_phase_timestep_embedders", None)
        if callable(clear_fn):
            clear_fn()


def _get_audio_reference_token_count(audio_state: LatentState) -> int:
    positions = audio_state.positions
    if positions is None or positions.ndim < 4 or positions.shape[1] < 1:
        return 0
    ref_mask = positions[:, 0, :, 1] < 0
    if ref_mask.ndim != 2 or not torch.any(ref_mask):
        return 0
    counts = ref_mask.sum(dim=1)
    if not torch.all(counts == counts[:1]):
        return 0
    ref_tokens = int(counts[0].item())
    total_tokens = int(ref_mask.shape[1])
    if ref_tokens <= 0 or ref_tokens >= total_tokens:
        return 0
    if not torch.all(ref_mask[:, :ref_tokens]) or torch.any(ref_mask[:, ref_tokens:]):
        return 0
    return ref_tokens


def _slice_audio_target_state(audio_state: LatentState, ref_audio_tokens: int) -> LatentState | None:
    ref_audio_tokens = int(ref_audio_tokens)
    total_tokens = int(audio_state.latent.shape[1])
    if ref_audio_tokens <= 0 or ref_audio_tokens >= total_tokens:
        return None
    return LatentState(
        latent=audio_state.latent[:, ref_audio_tokens:],
        denoise_mask=audio_state.denoise_mask[:, ref_audio_tokens:],
        positions=audio_state.positions[:, :, ref_audio_tokens:],
        clean_latent=audio_state.clean_latent[:, ref_audio_tokens:],
    )


def simple_denoising_func(
    video_context: torch.Tensor,
    audio_context: torch.Tensor | None,
    transformer: X0Model,
    video_nag: dict | None = None,
    audio_nag: dict | None = None,
    alt_guidance_scale: float = 1.0,
    audio_context_n: torch.Tensor | None = None,
    audio_guidance_scale: float = 1.0,
    audio_identity_guidance_scale: float = 0.0,
    manage_lora_step: bool = True,
    skip_audio_to_video: bool = False,
    ref_context: torch.Tensor | None = None,
    ref_adaln: torch.Tensor | None = None,
    video_context_mask_builder: Callable[[LatentState, torch.Tensor | None, torch.Tensor], torch.Tensor | None] | None = None,
    audio_context_mask_builder: Callable[[LatentState, torch.Tensor | None, torch.Tensor], torch.Tensor | None] | None = None,
    cross_attention_mask_builder: CrossAttentionMaskBuilder | None = None,
) -> DenoisingFunc:
    prepared_video_context = prepared_audio_context = None
    prepared_audio_context_n = prepared_audio_context_id = None

    def _prewarm(video_state: LatentState, audio_state: LatentState, sigmas: torch.Tensor) -> None:
        nonlocal prepared_video_context, prepared_audio_context
        if prepared_video_context is None:
            prepared_video_context = _prepare_conditioning_context(
                transformer, video_state, video_context, sigmas, is_audio=False
            )
        if audio_state is not None and audio_context is not None and prepared_audio_context is None:
            prepared_audio_context = _prepare_conditioning_context(
                transformer, audio_state, audio_context, sigmas, is_audio=True
            )

    def _cleanup() -> None:
        nonlocal prepared_video_context, prepared_audio_context, prepared_audio_context_n, prepared_audio_context_id
        prepared_video_context = None
        prepared_audio_context = None
        prepared_audio_context_n = None
        prepared_audio_context_id = None
        _clear_phase_timestep_embedders(transformer)

    def simple_denoising_step(
        video_state: LatentState, audio_state: LatentState, sigmas: torch.Tensor, step_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal prepared_audio_context_n, prepared_audio_context_id
        _prewarm(video_state, audio_state, sigmas)
        sigma = sigmas[step_index]
        cross_masks = cross_attention_mask_builder(video_state, audio_state) if cross_attention_mask_builder is not None else None
        video_cross_attention_mask = audio_cross_attention_mask = None
        if cross_masks is not None:
            video_cross_attention_mask = cross_masks.video_cross_attention_mask
            audio_cross_attention_mask = cross_masks.audio_cross_attention_mask
        pos_video = modality_from_latent_state(
            video_state,
            prepared_video_context,
            sigma,
            nag=video_nag,
            step_index=step_index,
            sigma_schedule=sigmas,
            ref_context=ref_context,
            ref_adaln=ref_adaln,
            context_mask_builder=video_context_mask_builder,
            cross_attention_mask=video_cross_attention_mask,
        )
        pos_audio = None
        if audio_state is not None and prepared_audio_context is not None:
            pos_audio = modality_from_latent_state(
                audio_state,
                prepared_audio_context,
                sigma,
                nag=audio_nag,
                step_index=step_index,
                sigma_schedule=sigmas,
                context_mask_builder=audio_context_mask_builder,
                cross_attention_mask=audio_cross_attention_mask,
            )

        if transformer is not None and manage_lora_step:
            offload.set_step_no_for_lora(transformer, step_index)
        use_alt = not math.isclose(alt_guidance_scale, 1.0)
        use_audio_cfg = (
            audio_state is not None
            and audio_context_n is not None
            and not math.isclose(audio_guidance_scale, 1.0)
        )
        ref_audio_tokens = _get_audio_reference_token_count(audio_state) if audio_state is not None and audio_identity_guidance_scale > 0 else 0
        id_audio_state = _slice_audio_target_state(audio_state, ref_audio_tokens) if ref_audio_tokens > 0 else None
        use_id = id_audio_state is not None
        if use_audio_cfg and prepared_audio_context_n is None:
            prepared_audio_context_n = _prepare_conditioning_context(
                transformer, audio_state, audio_context_n, sigmas, is_audio=True
            )
        if use_id and prepared_audio_context_id is None:
            prepared_audio_context_id = _prepare_conditioning_context(
                transformer, id_audio_state, audio_context, sigmas, is_audio=True
            )
        batch_size = _get_batch_size(video_state, audio_state)
        a2v_perturbations = _skip_audio_to_video_perturbations(batch_size) if skip_audio_to_video else None
        if not use_alt and not use_audio_cfg and not use_id:
            denoised_video, denoised_audio = transformer(video=pos_video, audio=pos_audio, perturbations=a2v_perturbations)
            if denoised_video is None and denoised_audio is None:
                return None, None
            return denoised_video, denoised_audio

        video_list = [pos_video]
        audio_list = [pos_audio]
        perturbations = [a2v_perturbations]
        neg_index = None
        alt_index = None
        id_index = None
        if use_audio_cfg:
            neg_index = len(video_list)
            video_list.append(
                modality_from_latent_state(
                    video_state,
                    prepared_video_context,
                    sigma,
                    nag=video_nag,
                    step_index=step_index,
                    sigma_schedule=sigmas,
                    context_mask_builder=video_context_mask_builder,
                    cross_attention_mask=video_cross_attention_mask,
                )
            )
            audio_list.append(
                modality_from_latent_state(
                    audio_state,
                    prepared_audio_context_n,
                    sigma,
                    nag=audio_nag,
                    step_index=step_index,
                    sigma_schedule=sigmas,
                    cross_attention_mask=audio_cross_attention_mask,
                )
            )
            perturbations.append(a2v_perturbations)
        if use_alt:
            alt_index = len(video_list)
            video_list.append(
                modality_from_latent_state(
                    video_state,
                    prepared_video_context,
                    sigma,
                    nag=video_nag,
                    step_index=step_index,
                    sigma_schedule=sigmas,
                    context_mask_builder=video_context_mask_builder,
                    cross_attention_mask=video_cross_attention_mask,
                )
            )
            audio_list.append(
                modality_from_latent_state(
                    audio_state,
                    prepared_audio_context,
                    sigma,
                    nag=audio_nag,
                    step_index=step_index,
                    sigma_schedule=sigmas,
                    context_mask_builder=audio_context_mask_builder,
                    cross_attention_mask=audio_cross_attention_mask,
                )
            )
            perturbations.append(_cross_attn_perturbations(batch_size))
        if use_id:
            id_index = len(video_list)
            video_list.append(
                modality_from_latent_state(
                    video_state,
                    prepared_video_context,
                    sigma,
                    nag=video_nag,
                    step_index=step_index,
                    sigma_schedule=sigmas,
                    context_mask_builder=video_context_mask_builder,
                    cross_attention_mask=video_cross_attention_mask,
                )
            )
            audio_list.append(
                modality_from_latent_state(
                    id_audio_state,
                    prepared_audio_context_id,
                    sigma,
                    nag=audio_nag,
                    step_index=step_index,
                    sigma_schedule=sigmas,
                    context_mask_builder=audio_context_mask_builder,
                )
            )
            perturbations.append(a2v_perturbations)

        denoised_video_list, denoised_audio_list = transformer(
            video=video_list,
            audio=audio_list,
            perturbations=perturbations,
        )
        if denoised_video_list is None and denoised_audio_list is None:
            return None, None
        pos_denoised_video = denoised_video_list[0]
        pos_denoised_audio = denoised_audio_list[0]
        if pos_denoised_video is None and pos_denoised_audio is None:
            return None, None
        denoised_video = pos_denoised_video
        denoised_audio = pos_denoised_audio
        if use_audio_cfg and neg_index is not None and denoised_audio is not None:
            neg_denoised_audio = denoised_audio_list[neg_index]
            if neg_denoised_audio is not None:
                denoised_audio = denoised_audio + (audio_guidance_scale - 1.0) * (
                    pos_denoised_audio - neg_denoised_audio
                )
        if use_alt and alt_index is not None:
            alt_denoised_video = denoised_video_list[alt_index]
            alt_denoised_audio = denoised_audio_list[alt_index]
            if denoised_video is not None and alt_denoised_video is not None:
                denoised_video = denoised_video + (alt_guidance_scale - 1.0) * (
                    pos_denoised_video - alt_denoised_video
                )
            if denoised_audio is not None and alt_denoised_audio is not None:
                denoised_audio = denoised_audio + (alt_guidance_scale - 1.0) * (
                    pos_denoised_audio - alt_denoised_audio
                )
        if use_id and id_index is not None and denoised_audio is not None:
            id_denoised_audio = denoised_audio_list[id_index]
            target_audio_tokens = int(pos_denoised_audio.shape[1]) - int(ref_audio_tokens)
            if (
                id_denoised_audio is not None
                and target_audio_tokens > 0
                and id_denoised_audio.shape[1] == target_audio_tokens
            ):
                denoised_audio = denoised_audio.clone()
                denoised_audio[:, ref_audio_tokens:] = denoised_audio[:, ref_audio_tokens:] + audio_identity_guidance_scale * (
                    pos_denoised_audio[:, ref_audio_tokens:] - id_denoised_audio
                )
        return denoised_video, denoised_audio

    simple_denoising_step._prewarm = _prewarm
    simple_denoising_step._cleanup = _cleanup
    return simple_denoising_step


def guider_denoising_func(
    video_guider: GuiderProtocol,
    audio_guider: GuiderProtocol,
    v_context_p: torch.Tensor,
    v_context_n: torch.Tensor,
    a_context_p: torch.Tensor,
    a_context_n: torch.Tensor,
    transformer: X0Model,
    alt_guidance_scale: float = 1.0,
    alt_scale: float = 0.0,
    perturbation_switch: int = 0,
    perturbation_layers: list[int] | None = None,
    perturbation_start: float = 0.0,
    perturbation_end: float = 1.0,
    audio_identity_guidance_scale: float = 0.0,
    ref_context: torch.Tensor | None = None,
    ref_adaln: torch.Tensor | None = None,
    v_context_p_mask_builder: Callable[[LatentState, torch.Tensor | None, torch.Tensor], torch.Tensor | None] | None = None,
    a_context_p_mask_builder: Callable[[LatentState, torch.Tensor | None, torch.Tensor], torch.Tensor | None] | None = None,
    skip_audio_to_video: bool = False,
) -> DenoisingFunc:
    perturb_all_layers = perturbation_layers is None
    perturbation_layers_norm = _normalize_perturbation_layers(perturbation_layers)
    prepared_v_context_p = prepared_v_context_n = None
    prepared_a_context_p = prepared_a_context_n = prepared_a_context_id = None

    def _prewarm(video_state: LatentState, audio_state: LatentState, sigmas: torch.Tensor) -> None:
        nonlocal prepared_v_context_p, prepared_v_context_n, prepared_a_context_p, prepared_a_context_n
        if prepared_v_context_p is None:
            prepared_v_context_p = _prepare_conditioning_context(transformer, video_state, v_context_p, sigmas, is_audio=False)
        if prepared_a_context_p is None:
            prepared_a_context_p = _prepare_conditioning_context(transformer, audio_state, a_context_p, sigmas, is_audio=True)

    def _cleanup() -> None:
        nonlocal prepared_v_context_p, prepared_v_context_n, prepared_a_context_p, prepared_a_context_n, prepared_a_context_id
        prepared_v_context_p = prepared_v_context_n = None
        prepared_a_context_p = prepared_a_context_n = prepared_a_context_id = None
        _clear_phase_timestep_embedders(transformer)

    def guider_denoising_step(
        video_state: LatentState, audio_state: LatentState, sigmas: torch.Tensor, step_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal prepared_v_context_n, prepared_a_context_n, prepared_a_context_id
        _prewarm(video_state, audio_state, sigmas)
        sigma = sigmas[step_index]
        pos_video = modality_from_latent_state(
            video_state,
            prepared_v_context_p,
            sigma,
            step_index=step_index,
            sigma_schedule=sigmas,
            ref_context=ref_context,
            ref_adaln=ref_adaln,
            context_mask_builder=v_context_p_mask_builder,
        )
        pos_audio = modality_from_latent_state(
            audio_state,
            prepared_a_context_p,
            sigma,
            step_index=step_index,
            sigma_schedule=sigmas,
            context_mask_builder=a_context_p_mask_builder,
        )

        if transformer is not None:
            offload.set_step_no_for_lora(transformer, step_index)
        use_video_cfg = video_guider.enabled()
        use_audio_cfg = audio_guider.enabled()
        use_cfg = use_video_cfg or use_audio_cfg
        use_alt = not math.isclose(alt_guidance_scale, 1.0)
        ref_audio_tokens = _get_audio_reference_token_count(audio_state) if audio_identity_guidance_scale > 0 else 0
        id_audio_state = _slice_audio_target_state(audio_state, ref_audio_tokens) if ref_audio_tokens > 0 else None
        use_id = id_audio_state is not None
        if use_id and prepared_a_context_id is None:
            prepared_a_context_id = _prepare_conditioning_context(
                transformer, id_audio_state, a_context_p, sigmas, is_audio=True
            )
        use_perturbation = _perturbation_active(step_index, sigmas, perturbation_start, perturbation_end)
        has_perturbation_layers = perturb_all_layers or bool(perturbation_layers_norm)
        use_legacy_perturbation = (
            perturbation_switch == PERTURBATION_perturbation and use_cfg and use_perturbation and has_perturbation_layers
        )
        use_stg = (
            perturbation_switch == PERTURBATION_SKIP_SELF_ATTENTION and use_perturbation and has_perturbation_layers
        )
        selected_layers = None if perturb_all_layers else perturbation_layers_norm
        batch_size = _get_batch_size(video_state, audio_state)
        a2v_perturbations = _skip_audio_to_video_perturbations(batch_size) if skip_audio_to_video else None
        if use_cfg or use_alt or use_stg or use_id:
            video_list = [pos_video]
            audio_list = [pos_audio]
            perturbations: list[BatchedPerturbationConfig | None] = [a2v_perturbations]
            neg_index = None
            stg_index = None
            alt_index = None
            id_index = None

            if use_cfg:
                if use_video_cfg and prepared_v_context_n is None:
                    prepared_v_context_n = _prepare_conditioning_context(
                        transformer, video_state, v_context_n, sigmas, is_audio=False
                    )
                if use_audio_cfg and prepared_a_context_n is None:
                    prepared_a_context_n = _prepare_conditioning_context(
                        transformer, audio_state, a_context_n, sigmas, is_audio=True
                    )
                neg_video_context = prepared_v_context_n if use_video_cfg else prepared_v_context_p
                neg_video_mask_builder = None if use_video_cfg else v_context_p_mask_builder
                neg_audio_context = prepared_a_context_n if use_audio_cfg else prepared_a_context_p
                neg_audio_mask_builder = None if use_audio_cfg else a_context_p_mask_builder
                neg_index = len(video_list)
                video_list.append(
                    modality_from_latent_state(
                        video_state,
                        neg_video_context,
                        sigma,
                        step_index=step_index,
                        sigma_schedule=sigmas,
                        context_mask_builder=neg_video_mask_builder,
                    )
                )
                audio_list.append(
                    modality_from_latent_state(
                        audio_state,
                        neg_audio_context,
                        sigma,
                        step_index=step_index,
                        sigma_schedule=sigmas,
                        context_mask_builder=neg_audio_mask_builder,
                    )
                )
                perturbations.append(
                    _merge_perturbations(
                        batch_size,
                        a2v_perturbations,
                        _legacy_perturbation_layer_configs(batch_size, selected_layers) if use_legacy_perturbation else None,
                    )
                )

            if use_stg:
                stg_index = len(video_list)
                video_list.append(
                    modality_from_latent_state(
                        video_state,
                        prepared_v_context_p,
                        sigma,
                        step_index=step_index,
                        sigma_schedule=sigmas,
                        context_mask_builder=v_context_p_mask_builder,
                    )
                )
                audio_list.append(
                    modality_from_latent_state(
                        audio_state,
                        prepared_a_context_p,
                        sigma,
                        step_index=step_index,
                        sigma_schedule=sigmas,
                        context_mask_builder=a_context_p_mask_builder,
                    )
                )
                perturbations.append(
                    _merge_perturbations(batch_size, a2v_perturbations, _self_attn_perturbation_configs(batch_size, selected_layers))
                )

            if use_alt:
                alt_index = len(video_list)
                video_list.append(
                    modality_from_latent_state(
                        video_state,
                        prepared_v_context_p,
                        sigma,
                        step_index=step_index,
                        sigma_schedule=sigmas,
                        context_mask_builder=v_context_p_mask_builder,
                    )
                )
                audio_list.append(
                    modality_from_latent_state(
                        audio_state,
                        prepared_a_context_p,
                        sigma,
                        step_index=step_index,
                        sigma_schedule=sigmas,
                        context_mask_builder=a_context_p_mask_builder,
                    )
                )
                perturbations.append(_merge_perturbations(batch_size, a2v_perturbations, _cross_attn_perturbations(batch_size)))

            if use_id:
                id_index = len(video_list)
                video_list.append(
                    modality_from_latent_state(
                        video_state,
                        prepared_v_context_p,
                        sigma,
                        step_index=step_index,
                        sigma_schedule=sigmas,
                        context_mask_builder=v_context_p_mask_builder,
                    )
                )
                audio_list.append(
                    modality_from_latent_state(
                        id_audio_state,
                        prepared_a_context_id,
                        sigma,
                        step_index=step_index,
                        sigma_schedule=sigmas,
                        context_mask_builder=a_context_p_mask_builder,
                    )
                )
                perturbations.append(a2v_perturbations)

            denoised_video_list, denoised_audio_list = transformer(
                video=video_list,
                audio=audio_list,
                perturbations=perturbations,
            )
            if denoised_video_list is None and denoised_audio_list is None:
                return None, None
            pos_denoised_video = denoised_video_list[0]
            pos_denoised_audio = denoised_audio_list[0]
            if pos_denoised_video is None and pos_denoised_audio is None:
                return None, None

            denoised_video = pos_denoised_video
            denoised_audio = pos_denoised_audio

            if use_cfg and neg_index is not None:
                neg_denoised_video = denoised_video_list[neg_index]
                neg_denoised_audio = denoised_audio_list[neg_index]
                if denoised_video is not None and neg_denoised_video is not None and use_video_cfg:
                    denoised_video = denoised_video + video_guider.delta(pos_denoised_video, neg_denoised_video)
                if denoised_audio is not None and neg_denoised_audio is not None and use_audio_cfg:
                    denoised_audio = denoised_audio + audio_guider.delta(pos_denoised_audio, neg_denoised_audio)

            if use_stg and stg_index is not None:
                stg_denoised_video = denoised_video_list[stg_index]
                stg_denoised_audio = denoised_audio_list[stg_index]
                if denoised_video is not None and stg_denoised_video is not None:
                    denoised_video = denoised_video + (pos_denoised_video - stg_denoised_video)
                if denoised_audio is not None and stg_denoised_audio is not None:
                    denoised_audio = denoised_audio + (pos_denoised_audio - stg_denoised_audio)

            if use_alt and alt_index is not None:
                alt_denoised_video = denoised_video_list[alt_index]
                alt_denoised_audio = denoised_audio_list[alt_index]
                if denoised_video is not None and alt_denoised_video is not None:
                    denoised_video = denoised_video + (alt_guidance_scale - 1.0) * (
                        pos_denoised_video - alt_denoised_video
                    )
                if denoised_audio is not None and alt_denoised_audio is not None:
                    denoised_audio = denoised_audio + (alt_guidance_scale - 1.0) * (
                        pos_denoised_audio - alt_denoised_audio
                    )
            if use_id and id_index is not None and denoised_audio is not None:
                id_denoised_audio = denoised_audio_list[id_index]
                target_audio_tokens = int(pos_denoised_audio.shape[1]) - int(ref_audio_tokens)
                if (
                    id_denoised_audio is not None
                    and target_audio_tokens > 0
                    and id_denoised_audio.shape[1] == target_audio_tokens
                ):
                    denoised_audio = denoised_audio.clone()
                    denoised_audio[:, ref_audio_tokens:] = denoised_audio[:, ref_audio_tokens:] + audio_identity_guidance_scale * (
                        pos_denoised_audio[:, ref_audio_tokens:] - id_denoised_audio
                    )
        else:
            denoised_video, denoised_audio = transformer(video=pos_video, audio=pos_audio, perturbations=a2v_perturbations)
            if denoised_video is None and denoised_audio is None:
                return None, None
            pos_denoised_video = denoised_video
            pos_denoised_audio = denoised_audio

        denoised_video = _rescale_prediction(pos_denoised_video, denoised_video, alt_scale)
        denoised_audio = _rescale_prediction(pos_denoised_audio, denoised_audio, alt_scale)

        pos_video = pos_audio = None
        return denoised_video, denoised_audio

    guider_denoising_step._prewarm = _prewarm
    guider_denoising_step._cleanup = _cleanup
    return guider_denoising_step


def _channelwise_normalize(x: torch.Tensor) -> torch.Tensor:
    return x.sub_(x.mean(dim=(-2, -1), keepdim=True)).div_(x.std(dim=(-2, -1), keepdim=True).clamp_min_(1e-6))


def _get_new_noise(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    noise = torch.randn(x.shape, generator=generator, dtype=torch.float64, device=x.device)
    noise = (noise - noise.mean()) / noise.std().clamp_min(1e-6)
    return _channelwise_normalize(noise)


def _inject_res2s_sde_noise(
    state: LatentState,
    sample: torch.Tensor,
    denoised_sample: torch.Tensor,
    step_noise_generator: torch.Generator,
    stepper: Res2sDiffusionStep,
    sigmas: torch.Tensor,
    step_idx: int,
    *,
    legacy_mode: bool = True,
) -> torch.Tensor:
    new_noise = _get_new_noise(state.latent, step_noise_generator)
    noise_sigmas = sigmas
    noise_step_idx = step_idx
    if not legacy_mode:
        timesteps, _ = timesteps_from_mask(state.denoise_mask.double(), sigmas[step_idx].double())
        next_timesteps, _ = timesteps_from_mask(state.denoise_mask.double(), sigmas[step_idx + 1].double())
        noise_sigmas = torch.stack([timesteps, next_timesteps])
        noise_step_idx = 0
    x_next = stepper.step(sample=sample, denoised_sample=denoised_sample, sigmas=noise_sigmas, step_index=noise_step_idx, noise=new_noise)
    if legacy_mode:
        x_next = post_process_latent(x_next, state.denoise_mask, state.clean_latent)
    return x_next


def multi_modal_guider_denoising_func(
    video_guider: MultiModalGuider,
    audio_guider: MultiModalGuider,
    v_context_p: torch.Tensor,
    a_context_p: torch.Tensor,
    transformer: X0Model,
    *,
    audio_identity_guidance_scale: float = 0.0,
    last_denoised_video: torch.Tensor | None = None,
    last_denoised_audio: torch.Tensor | None = None,
    ref_context: torch.Tensor | None = None,
    ref_adaln: torch.Tensor | None = None,
    v_context_p_mask_builder: Callable[[LatentState, torch.Tensor | None, torch.Tensor], torch.Tensor | None] | None = None,
    a_context_p_mask_builder: Callable[[LatentState, torch.Tensor | None, torch.Tensor], torch.Tensor | None] | None = None,
    skip_audio_to_video: bool = False,
    force_unconditional: bool = False,
    capture_unconditional: bool = False,
) -> DenoisingFunc:
    prepared_v_context_p = prepared_v_context_n = None
    prepared_a_context_p = prepared_a_context_n = prepared_a_context_id = None

    def _prewarm(video_state: LatentState, audio_state: LatentState, sigmas: torch.Tensor) -> None:
        nonlocal prepared_v_context_p, prepared_a_context_p
        if prepared_v_context_p is None:
            prepared_v_context_p = _prepare_conditioning_context(transformer, video_state, v_context_p, sigmas, is_audio=False)
        if prepared_a_context_p is None:
            prepared_a_context_p = _prepare_conditioning_context(transformer, audio_state, a_context_p, sigmas, is_audio=True)

    def _cleanup() -> None:
        nonlocal prepared_v_context_p, prepared_v_context_n, prepared_a_context_p, prepared_a_context_n, prepared_a_context_id
        prepared_v_context_p = prepared_v_context_n = None
        prepared_a_context_p = prepared_a_context_n = prepared_a_context_id = None
        _clear_phase_timestep_embedders(transformer)

    def guider_denoising_step(
        video_state: LatentState, audio_state: LatentState, sigmas: torch.Tensor, step_index: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        nonlocal last_denoised_video, last_denoised_audio, prepared_v_context_n, prepared_a_context_n, prepared_a_context_id
        _prewarm(video_state, audio_state, sigmas)

        skip_video = video_guider.should_skip_step(step_index)
        skip_audio = audio_guider.should_skip_step(step_index)
        if skip_video and skip_audio and last_denoised_video is not None and last_denoised_audio is not None:
            return last_denoised_video, last_denoised_audio

        sigma = sigmas[step_index]
        pos_video = modality_from_latent_state(
            video_state,
            prepared_v_context_p,
            sigma,
            enabled=not skip_video,
            step_index=step_index,
            sigma_schedule=sigmas,
            ref_context=ref_context,
            ref_adaln=ref_adaln,
            context_mask_builder=v_context_p_mask_builder,
        )
        pos_audio = modality_from_latent_state(
            audio_state,
            prepared_a_context_p,
            sigma,
            enabled=not skip_audio,
            step_index=step_index,
            sigma_schedule=sigmas,
            context_mask_builder=a_context_p_mask_builder,
        )

        use_video_cfg = force_unconditional or video_guider.do_unconditional_generation()
        use_audio_cfg = force_unconditional or audio_guider.do_unconditional_generation()
        use_cfg = use_video_cfg or use_audio_cfg
        use_video_stg = video_guider.do_perturbed_generation()
        use_audio_stg = audio_guider.do_perturbed_generation()
        use_stg = use_video_stg or use_audio_stg
        use_video_modality = video_guider.do_isolated_modality_generation()
        use_audio_modality = audio_guider.do_isolated_modality_generation()
        use_modality = use_video_modality or use_audio_modality
        ref_audio_tokens = _get_audio_reference_token_count(audio_state) if audio_identity_guidance_scale > 0 else 0
        id_audio_state = _slice_audio_target_state(audio_state, ref_audio_tokens) if ref_audio_tokens > 0 else None
        use_id = id_audio_state is not None
        if use_id and prepared_a_context_id is None:
            prepared_a_context_id = _prepare_conditioning_context(transformer, id_audio_state, a_context_p, sigmas, is_audio=True)

        if use_cfg or use_stg or use_modality or use_id:
            batch_size = _get_batch_size(video_state, audio_state)
            a2v_perturbations = _skip_audio_to_video_perturbations(batch_size) if skip_audio_to_video else None
            video_list = [pos_video]
            audio_list = [pos_audio]
            perturbations: list[BatchedPerturbationConfig | None] = [a2v_perturbations]
            neg_index = None
            stg_index = None
            modality_index = None
            id_index = None

            if use_cfg:
                if use_video_cfg and video_guider.negative_context is None:
                    raise ValueError("Negative video context is required for HQ unconditional denoising.")
                if use_audio_cfg and audio_guider.negative_context is None:
                    raise ValueError("Negative audio context is required for HQ unconditional denoising.")
                if use_video_cfg and prepared_v_context_n is None:
                    prepared_v_context_n = _prepare_conditioning_context(
                        transformer, video_state, video_guider.negative_context, sigmas, is_audio=False
                    )
                if use_audio_cfg and prepared_a_context_n is None:
                    prepared_a_context_n = _prepare_conditioning_context(
                        transformer, audio_state, audio_guider.negative_context, sigmas, is_audio=True
                    )
                neg_index = len(video_list)
                neg_video_mask_builder = None if use_video_cfg else v_context_p_mask_builder
                neg_audio_mask_builder = None if use_audio_cfg else a_context_p_mask_builder
                video_list.append(
                    modality_from_latent_state(
                        video_state,
                        prepared_v_context_n if use_video_cfg else prepared_v_context_p,
                        sigma,
                        step_index=step_index,
                        sigma_schedule=sigmas,
                        context_mask_builder=neg_video_mask_builder,
                    )
                )
                audio_list.append(
                    modality_from_latent_state(
                        audio_state,
                        prepared_a_context_n if use_audio_cfg else prepared_a_context_p,
                        sigma,
                        step_index=step_index,
                        sigma_schedule=sigmas,
                        context_mask_builder=neg_audio_mask_builder,
                    )
                )
                perturbations.append(a2v_perturbations)

            if use_stg:
                stg_index = len(video_list)
                video_list.append(
                    modality_from_latent_state(
                        video_state,
                        prepared_v_context_p,
                        sigma,
                        step_index=step_index,
                        sigma_schedule=sigmas,
                        context_mask_builder=v_context_p_mask_builder,
                    )
                )
                audio_list.append(
                    modality_from_latent_state(
                        audio_state,
                        prepared_a_context_p,
                        sigma,
                        step_index=step_index,
                        sigma_schedule=sigmas,
                        context_mask_builder=a_context_p_mask_builder,
                    )
                )
                stg_perturbations = []
                if use_video_stg:
                    stg_perturbations.append(
                        Perturbation(type=PerturbationType.SKIP_VIDEO_SELF_ATTN, blocks=video_guider.params.stg_blocks or None)
                    )
                if use_audio_stg:
                    stg_perturbations.append(
                        Perturbation(type=PerturbationType.SKIP_AUDIO_SELF_ATTN, blocks=audio_guider.params.stg_blocks or None)
                    )
                stg_config = BatchedPerturbationConfig([PerturbationConfig(stg_perturbations) for _ in range(batch_size)])
                perturbations.append(_merge_perturbations(batch_size, a2v_perturbations, stg_config))

            if use_modality:
                modality_index = len(video_list)
                video_list.append(
                    modality_from_latent_state(
                        video_state,
                        prepared_v_context_p,
                        sigma,
                        step_index=step_index,
                        sigma_schedule=sigmas,
                        context_mask_builder=v_context_p_mask_builder,
                    )
                )
                audio_list.append(
                    modality_from_latent_state(
                        audio_state,
                        prepared_a_context_p,
                        sigma,
                        step_index=step_index,
                        sigma_schedule=sigmas,
                        context_mask_builder=a_context_p_mask_builder,
                    )
                )
                perturbations.append(_merge_perturbations(batch_size, a2v_perturbations, _cross_attn_perturbations(batch_size)))

            if use_id:
                id_index = len(video_list)
                video_list.append(
                    modality_from_latent_state(
                        video_state,
                        prepared_v_context_p,
                        sigma,
                        step_index=step_index,
                        sigma_schedule=sigmas,
                        context_mask_builder=v_context_p_mask_builder,
                    )
                )
                audio_list.append(
                    modality_from_latent_state(
                        id_audio_state,
                        prepared_a_context_id,
                        sigma,
                        step_index=step_index,
                        sigma_schedule=sigmas,
                        context_mask_builder=a_context_p_mask_builder,
                    )
                )
                perturbations.append(a2v_perturbations)

            denoised_video_list, denoised_audio_list = transformer(
                video=video_list,
                audio=audio_list,
                perturbations=perturbations,
            )
            if denoised_video_list is None and denoised_audio_list is None:
                return None, None
            pos_denoised_video = denoised_video_list[0]
            pos_denoised_audio = denoised_audio_list[0]
            if pos_denoised_video is None and pos_denoised_audio is None:
                return None, None

            denoised_video = pos_denoised_video
            denoised_audio = pos_denoised_audio
            neg_denoised_video = pos_denoised_video
            neg_denoised_audio = pos_denoised_audio
            ptb_denoised_video = pos_denoised_video
            ptb_denoised_audio = pos_denoised_audio
            mod_denoised_video = pos_denoised_video
            mod_denoised_audio = pos_denoised_audio

            if use_cfg and neg_index is not None:
                neg_denoised_video = denoised_video_list[neg_index]
                neg_denoised_audio = denoised_audio_list[neg_index]
                if use_video_cfg and pos_denoised_video is not None and neg_denoised_video is None:
                    return None, None
                if use_audio_cfg and pos_denoised_audio is not None and neg_denoised_audio is None:
                    return None, None
                neg_denoised_video = pos_denoised_video if neg_denoised_video is None else neg_denoised_video
                neg_denoised_audio = pos_denoised_audio if neg_denoised_audio is None else neg_denoised_audio

            if use_stg and stg_index is not None:
                ptb_denoised_video = denoised_video_list[stg_index]
                ptb_denoised_audio = denoised_audio_list[stg_index]
                if use_video_stg and pos_denoised_video is not None and ptb_denoised_video is None:
                    return None, None
                if use_audio_stg and pos_denoised_audio is not None and ptb_denoised_audio is None:
                    return None, None
                ptb_denoised_video = pos_denoised_video if ptb_denoised_video is None else ptb_denoised_video
                ptb_denoised_audio = pos_denoised_audio if ptb_denoised_audio is None else ptb_denoised_audio

            if use_modality and modality_index is not None:
                mod_denoised_video = denoised_video_list[modality_index]
                mod_denoised_audio = denoised_audio_list[modality_index]
                if use_video_modality and pos_denoised_video is not None and mod_denoised_video is None:
                    return None, None
                if use_audio_modality and pos_denoised_audio is not None and mod_denoised_audio is None:
                    return None, None
                mod_denoised_video = pos_denoised_video if mod_denoised_video is None else mod_denoised_video
                mod_denoised_audio = pos_denoised_audio if mod_denoised_audio is None else mod_denoised_audio

            if not skip_video and pos_denoised_video is not None:
                denoised_video = video_guider.calculate(
                    pos_denoised_video, neg_denoised_video, ptb_denoised_video, mod_denoised_video
                )
            elif skip_video and last_denoised_video is not None:
                denoised_video = last_denoised_video

            if not skip_audio and pos_denoised_audio is not None:
                denoised_audio = audio_guider.calculate(
                    pos_denoised_audio, neg_denoised_audio, ptb_denoised_audio, mod_denoised_audio
                )
            elif skip_audio and last_denoised_audio is not None:
                denoised_audio = last_denoised_audio

            if use_id and id_index is not None and denoised_audio is not None and pos_denoised_audio is not None:
                id_denoised_audio = denoised_audio_list[id_index]
                if id_denoised_audio is None:
                    return None, None
                target_audio_tokens = int(pos_denoised_audio.shape[1]) - int(ref_audio_tokens)
                if target_audio_tokens > 0 and id_denoised_audio.shape[1] == target_audio_tokens:
                    denoised_audio = denoised_audio.clone()
                    denoised_audio[:, ref_audio_tokens:] = denoised_audio[:, ref_audio_tokens:] + audio_identity_guidance_scale * (
                        pos_denoised_audio[:, ref_audio_tokens:] - id_denoised_audio
                    )
        else:
            batch_size = _get_batch_size(video_state, audio_state)
            a2v_perturbations = _skip_audio_to_video_perturbations(batch_size) if skip_audio_to_video else None
            denoised_video, denoised_audio = transformer(video=pos_video, audio=pos_audio, perturbations=a2v_perturbations)
            if denoised_video is None and denoised_audio is None:
                return None, None
            if skip_video and last_denoised_video is not None:
                denoised_video = last_denoised_video
            if skip_audio and last_denoised_audio is not None:
                denoised_audio = last_denoised_audio

        last_denoised_video = denoised_video
        last_denoised_audio = denoised_audio
        if capture_unconditional:
            guider_denoising_step.unconditional_predictions = (neg_denoised_video, neg_denoised_audio) if use_cfg else (None, None)
        return denoised_video, denoised_audio

    guider_denoising_step.unconditional_predictions = (None, None)
    guider_denoising_step._prewarm = _prewarm
    guider_denoising_step._cleanup = _cleanup
    return guider_denoising_step


def res2s_audio_video_denoising_loop(
    sigmas: torch.Tensor,
    video_state: LatentState,
    audio_state: LatentState,
    stepper: DiffusionStepProtocol,
    denoise_fn: DenoisingFunc,
    *,
    noise_seed: int = -1,
    noise_seed_substep: int | None = None,
    bongmath: bool = True,
    bongmath_max_iter: int = 100,
    legacy_mode: bool = True,
    mask_context: MaskInjection | None = None,
    interrupt_check: Callable[[], bool] | None = None,
    callback: Callable[..., None] | None = None,
    preview_tools: VideoLatentTools | None = None,
    pass_no: int = 0,
    transformer=None,
) -> tuple[LatentState | None, LatentState | None]:
    if not isinstance(stepper, Res2sDiffusionStep):
        raise ValueError("stepper must be an instance of Res2sDiffusionStep")

    prewarm = getattr(denoise_fn, "_prewarm", None)
    cleanup = getattr(denoise_fn, "_cleanup", None)
    if callable(prewarm):
        prewarm(video_state, audio_state, sigmas)

    try:
        if noise_seed_substep is None:
            noise_seed_substep = noise_seed + 10000
        step_noise_generator = torch.Generator(device=video_state.latent.device).manual_seed(noise_seed)
        substep_noise_generator = torch.Generator(device=video_state.latent.device).manual_seed(noise_seed_substep)
        step_noise_injecting_fn = partial(
            _inject_res2s_sde_noise,
            step_noise_generator=step_noise_generator,
            stepper=stepper,
            legacy_mode=legacy_mode,
        )
        substep_noise_injecting_fn = partial(
            _inject_res2s_sde_noise,
            step_noise_generator=substep_noise_generator,
            stepper=stepper,
            legacy_mode=legacy_mode,
        )

        n_full_steps = len(sigmas) - 1
        if sigmas[-1] == 0:
            sigmas = torch.cat([sigmas[:-1], torch.tensor([0.0011, 0.0], device=sigmas.device)], dim=0)
        hs = -torch.log(sigmas[1:].double().cpu() / sigmas[:-1].double().cpu())
        phi_cache = {}

        for step_idx in tqdm(range(n_full_steps)):
            if interrupt_check is not None and interrupt_check():
                return None, None
            if transformer is not None:
                offload.set_step_no_for_lora(transformer, step_idx)

            sigma = sigmas[step_idx].double()
            sigma_next = sigmas[step_idx + 1].double()
            x_anchor_video = video_state.latent.clone().double()
            x_anchor_audio = audio_state.latent.clone().double()

            denoised_video_1, denoised_audio_1 = denoise_fn(video_state, audio_state, sigmas, step_idx)
            if denoised_video_1 is None or denoised_audio_1 is None:
                return None, None
            denoised_video_1 = post_process_latent(denoised_video_1, video_state.denoise_mask, video_state.clean_latent)
            denoised_audio_1 = post_process_latent(denoised_audio_1, audio_state.denoise_mask, audio_state.clean_latent)

            h = hs[step_idx].item()
            a21, b1, b2 = get_res2s_coefficients(h, phi_cache)
            sub_sigma = torch.sqrt(sigma * sigma_next)

            eps_1_video = denoised_video_1.double() - x_anchor_video
            eps_1_audio = denoised_audio_1.double() - x_anchor_audio
            x_mid_video = x_anchor_video + h * a21 * eps_1_video
            x_mid_audio = x_anchor_audio + h * a21 * eps_1_audio

            x_mid_video = substep_noise_injecting_fn(
                state=video_state,
                sample=x_anchor_video,
                denoised_sample=x_mid_video,
                sigmas=torch.stack([sigma, sub_sigma]),
                step_idx=0,
            )
            x_mid_audio = substep_noise_injecting_fn(
                state=audio_state,
                sample=x_anchor_audio,
                denoised_sample=x_mid_audio,
                sigmas=torch.stack([sigma, sub_sigma]),
                step_idx=0,
            )

            if bongmath and h < 0.5 and sigma > 0.03:
                for _ in range(bongmath_max_iter):
                    x_anchor_video = x_mid_video - h * a21 * eps_1_video
                    eps_1_video = denoised_video_1.double() - x_anchor_video
                    x_anchor_audio = x_mid_audio - h * a21 * eps_1_audio
                    eps_1_audio = denoised_audio_1.double() - x_anchor_audio

            if interrupt_check is not None and interrupt_check():
                return None, None
            if transformer is not None:
                offload.set_step_no_for_lora(transformer, step_idx)

            mid_video_state = replace(video_state, latent=x_mid_video.to(video_state.latent.dtype))
            mid_audio_state = replace(audio_state, latent=x_mid_audio.to(audio_state.latent.dtype))
            denoised_video_2, denoised_audio_2 = denoise_fn(
                video_state=mid_video_state,
                audio_state=mid_audio_state,
                sigmas=torch.stack([sub_sigma]).to(sigmas.device),
                step_index=0,
            )
            if denoised_video_2 is None or denoised_audio_2 is None:
                return None, None
            denoised_video_2 = post_process_latent(denoised_video_2, video_state.denoise_mask, video_state.clean_latent)
            denoised_audio_2 = post_process_latent(denoised_audio_2, audio_state.denoise_mask, audio_state.clean_latent)

            eps_2_video = denoised_video_2.double() - x_anchor_video
            eps_2_audio = denoised_audio_2.double() - x_anchor_audio
            x_next_video = x_anchor_video + h * (b1 * eps_1_video + b2 * eps_2_video)
            x_next_audio = x_anchor_audio + h * (b1 * eps_1_audio + b2 * eps_2_audio)

            x_next_video = step_noise_injecting_fn(
                state=video_state,
                sample=x_anchor_video,
                denoised_sample=x_next_video,
                sigmas=sigmas,
                step_idx=step_idx,
            )
            x_next_audio = step_noise_injecting_fn(
                state=audio_state,
                sample=x_anchor_audio,
                denoised_sample=x_next_audio,
                sigmas=sigmas,
                step_idx=step_idx,
            )

            video_state = replace(video_state, latent=x_next_video.to(video_state.latent.dtype))
            audio_state = replace(audio_state, latent=x_next_audio.to(audio_state.latent.dtype))
            if mask_context is not None:
                _apply_mask_injection(video_state, sigmas, step_idx, mask_context)
            preview_state = replace(video_state, latent=denoised_video_2) if getattr(callback, "tiny_vae", False) else video_state
            _invoke_callback(callback, step_idx, pass_no, preview_state, preview_tools)
            del preview_state

        if sigmas[-1] == 0:
            if interrupt_check is not None and interrupt_check():
                return None, None
            if transformer is not None:
                offload.set_step_no_for_lora(transformer, max(n_full_steps - 1, 0))
            denoised_video_1, denoised_audio_1 = denoise_fn(video_state, audio_state, sigmas, n_full_steps)
            if denoised_video_1 is None or denoised_audio_1 is None:
                return None, None
            denoised_video_1 = post_process_latent(denoised_video_1, video_state.denoise_mask, video_state.clean_latent)
            denoised_audio_1 = post_process_latent(denoised_audio_1, audio_state.denoise_mask, audio_state.clean_latent)
            video_state = replace(video_state, latent=denoised_video_1.to(video_state.latent.dtype))
            audio_state = replace(audio_state, latent=denoised_audio_1.to(audio_state.latent.dtype))

        return video_state, audio_state
    finally:
        if callable(cleanup):
            cleanup()


def denoise_audio_video(  # noqa: PLR0913
    output_shape: VideoPixelShape,
    conditionings: list[ConditioningItem],
    noiser: Noiser,
    sigmas: torch.Tensor,
    stepper: DiffusionStepProtocol,
    denoising_loop_fn: DenoisingLoopFunc,
    components: PipelineComponents,
    dtype: torch.dtype,
    device: torch.device,
    audio_conditionings: list[ConditioningItem] | None = None,
    noise_scale: float = 1.0,
    audio_noise_scale: float | None = None,
    initial_video_latent: torch.Tensor | None = None,
    initial_audio_latent: torch.Tensor | None = None,
    mask_context: MaskInjection | None = None,
    freeze_audio: bool = False,
    skip_audio: bool = False,
    video_position_offset: float = 0.0,
) -> tuple[LatentState | None, LatentState | None]:
    video_state, video_tools = noise_video_state(
        output_shape=output_shape,
        noiser=noiser,
        conditionings=conditionings,
        components=components,
        dtype=dtype,
        device=device,
        noise_scale=noise_scale,
        initial_latent=initial_video_latent,
    )
    audio_state = audio_tools = None
    if not skip_audio:
        audio_state, audio_tools = noise_audio_state(
            output_shape=output_shape,
            noiser=noiser,
            conditionings=audio_conditionings or [],
            components=components,
            dtype=dtype,
            device=device,
            noise_scale=noise_scale if audio_noise_scale is None else audio_noise_scale,
            initial_latent=initial_audio_latent,
        )
    if freeze_audio and audio_state is not None:
        audio_state = replace(audio_state, denoise_mask=torch.zeros_like(audio_state.denoise_mask))
    if video_position_offset:
        positions = video_state.positions.clone()
        positions[:, 0].add_(float(video_position_offset))
        video_state = replace(video_state, positions=positions)

    loop_kwargs = {}
    if "preview_tools" in inspect.signature(denoising_loop_fn).parameters:
        loop_kwargs["preview_tools"] = video_tools
    if "mask_context" in inspect.signature(denoising_loop_fn).parameters:
        loop_kwargs["mask_context"] = mask_context
    video_state, audio_state = denoising_loop_fn(
        sigmas,
        video_state,
        audio_state,
        stepper,
        **loop_kwargs,
    )

    if video_state is None or (not skip_audio and audio_state is None):
        return None, None

    video_state = video_tools.clear_conditioning(video_state)
    video_state = video_tools.unpatchify(video_state)
    if audio_state is not None and audio_tools is not None:
        audio_state = audio_tools.clear_conditioning(audio_state)
        audio_state = audio_tools.unpatchify(audio_state)

    return video_state, audio_state


def _invoke_callback(
    callback: Callable[..., None] | None,
    step_idx: int,
    pass_no: int,
    video_state: LatentState | None,
    preview_tools: VideoLatentTools | None,
) -> None:
    if callback is None or video_state is None:
        return
    preview_latents = None
    if preview_tools is not None:
        preview_state = preview_tools.clear_conditioning(video_state)
        preview_state = preview_tools.unpatchify(preview_state)
        preview_latents = preview_state.latent[0].detach()
    callback(step_idx, preview_latents, False, pass_no=pass_no)


_UNICODE_REPLACEMENTS = str.maketrans("\u2018\u2019\u201c\u201d\u2014\u2013\u00a0\u2032\u2212", "''\"\"-- '-")


def clean_response(text: str) -> str:
    """Clean a response from curly quotes and leading non-letter characters which Gemma tends to insert."""
    text = text.translate(_UNICODE_REPLACEMENTS)

    # Remove leading non-letter characters
    for i, char in enumerate(text):
        if char.isalpha():
            return text[i:]
    return text


def generate_enhanced_prompt(
    text_encoder: GemmaTextEncoderModelBase,
    prompt: str,
    image_path: str | None = None,
    image_long_side: int = 896,
    seed: int = 42,
) -> str:
    """Generate an enhanced prompt from a text encoder and a prompt."""
    image = None
    if image_path:
        image = decode_image(image_path=image_path)
        image = torch.tensor(image)
        image = resize_aspect_ratio_preserving(image, image_long_side).to(torch.uint8)
        prompt = text_encoder.enhance_i2v(prompt, image, seed=seed)
    else:
        prompt = text_encoder.enhance_t2v(prompt, seed=seed)
    logging.info(f"Enhanced prompt: {prompt}")
    return clean_response(prompt)


def assert_resolution(
    height: int,
    width: int,
    is_two_stage: bool,
) -> None:
    """Assert that the resolution is divisible by the required divisor."""
    divisor = 64 if is_two_stage else 32
    pipeline_label = "two-stage" if is_two_stage else "one-stage"
    if height % divisor != 0 or width % divisor != 0:
        raise ValueError(
            f"Resolution ({height}x{width}) is not divisible by {divisor}. "
            f"For {pipeline_label} pipelines, "
            f"height and width must be multiples of {divisor}."
        )
