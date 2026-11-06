# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
# Inference flow adapted from FlashVSR (https://github.com/OpenImagingLab/FlashVSR) and FlashVSR+
# (https://github.com/OpenImagingLab/FlashVSR_Plus), Copyright (c) their authors, Apache License 2.0 (see docs/third_party_licenses/Apache-2.0.txt).
from __future__ import annotations
from shared.utils.media_control import controlled_model_loading, loading_callback
from shared.utils.phase_progress import generation_progress

import gc
import math
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from einops import rearrange
from safetensors.torch import load_file
from tqdm import tqdm

from mmgp import offload
from models.wan.modules.vae import WanVAE
from postprocessing.memory import minimal_vram
from shared.utils import offload_registry
from .attention_backend import log_sparse_backend, require_sparge_attention
from .stream_cache import StreamCache
from .tcdecoder import build_tcdecoder, tile_plan
from .utils import Causal_LQ4x_Proj
from .wan_video_dit import WanModel, precompute_freqs_cis_3d, structural_block_candidates, structural_block_mask


FLASHVSR_VARIANT_TINY_LONG = "tiny-long"
FLASHVSR_VARIANT_TINY = "tiny"
FLASHVSR_VARIANT_FULL = "full"

FLASHVSR_MODEL_V11 = "v1_1"
FLASHVSR_MODEL_PLUS_TURBO = "plus_turbo"
FLASHVSR_PLUS_MODELS = (FLASHVSR_MODEL_PLUS_TURBO,)

FLASHVSR_TOPK_RATIO = 0.0  # 0 = auto area-scaled ratio; >0 = fixed sparse attention ratio.
FLASHVSR_FULL_MIN_AUTO_TOPK_RATIO = 1.5
FLASHVSR_KV_CACHE_WINDOWS = 1  # Stream cache windows kept between denoise chunks; each window is two latent frames.
FLASHVSR_CONTINUE_CACHE_FRAMES = 11

# FlashVSR+ (released inference configuration, inference/presets.py and models/config.py upstream): 4 latent frames per step, the draft
# top-k budget `spatial^2 * 2.5 - 1` ranked over the 2-window chunk, 11x11 local candidates and a 3-window cache for the draft top-k
# layers; Turbo's hybrid attention replaces every layer but 0, 6, 12, 18, 24 by a local layer: 3x3 candidates over the previous window
# and the current chunk, a 1-window cache.
FLASHVSR_PLUS_CHUNK_LATENTS = 4
FLASHVSR_PLUS_TOPK_RATIO = 2.5
FLASHVSR_PLUS_TOPK_GROUP_WINDOWS = 2
# Upstream ranks (query window, key window) pairs of 128 x 128 tokens; WanGP's sparse kernels rank (query window, key half-window)
# pairs of 128 x 64 tokens, so the same count attends half the tokens (the v1.1 port's convention). The Plus models keep upstream's
# attended volume: twice the count on the half-window map (parity harness: latents within 2.6-3.1 % of upstream against 16 % with
# the v1.1 convention, upstream's own BF16 rounding being 2.4 %).
FLASHVSR_PLUS_HALF_WINDOW_BUDGET = 2
FLASHVSR_PLUS_LOCAL_RANGE = 11
FLASHVSR_PLUS_CACHE_WINDOWS = 3
FLASHVSR_PLUS_LOCAL_LAYER_RANGE = 3
FLASHVSR_PLUS_LOCAL_LAYER_CACHE_WINDOWS = 1
FLASHVSR_PLUS_LOCAL_LAYERS = {FLASHVSR_MODEL_PLUS_TURBO: frozenset(layer for layer in range(30) if layer % 6)}
FLASHVSR_PLUS_CONTINUE_CACHE_FRAMES = 19  # ((frames - 1) mod 16) + 4 at most: the output frames of the last chunk, conditioned on repeated frames
FLASHVSR_COTENANTS_MAP = {"lq_proj": ["transformer"]}
FLASHVSR_SAVE_STILL_IMAGE_DEBUG_VIDEO = False
FLASHVSR_DISABLE_STILL_IMAGE_OPTIMIZATIONS = False
FLASHVSR_STILL_IMAGE_SHIFT_CORRECTION = True
FLASHVSR_STILL_IMAGE_SHIFT_CORRECTION_INPUT_SHIFT = None  # None = half-period output phase shift.
FLASHVSR_STILL_IMAGE_SHIFT_CORRECTION_PERIOD = 16
FLASHVSR_STILL_IMAGE_RETURN_WARMED_FRAME = True
FLASHVSR_STILL_IMAGE_SHIFT_BLEND = 0.5
FLASHVSR_SHIFT_BLEND_WORKING_MEMORY_MB = 512
FLASHVSR_COLOR_FIX_WORKING_MEMORY_MB = 512
FLASHVSR_STILL_IMAGE_DEBUG_VIDEO_PATH = "flashvsr_still_image_debug.mp4"
FLASHVSR_STILL_IMAGE_DEBUG_VIDEO_FPS = 4

WAN_1_3B_CONFIG = {
    "has_image_input": False,
    "patch_size": (1, 2, 2),
    "in_dim": 16,
    "dim": 1536,
    "ffn_dim": 8960,
    "freq_dim": 256,
    "text_dim": 4096,
    "out_dim": 16,
    "num_heads": 12,
    "num_layers": 30,
    "eps": 1e-6,
}


@contextmanager
def _default_dtype(dtype: torch.dtype):
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous_dtype)


@dataclass
class FlashVSRPaths:
    transformer: str
    lq_proj: str
    posi_prompt: str
    tcdecoder: str | None = None
    vae: str | None = None


def _preprocess_transformer_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    converter = WanModel.state_dict_converter()
    state_dict, _ = converter.from_civitai(state_dict)
    return state_dict


def _sinusoidal_embedding_1d(dim: int, position: torch.Tensor) -> torch.Tensor:
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(10000, -torch.arange(dim // 2, dtype=torch.float64, device=position.device).div(dim // 2)))
    return torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1).to(position.dtype)


def _next_conditioning_frame_count(frame_count: int) -> int:
    padded = max(25, frame_count + 4)
    remainder = padded % 8
    if remainder != 1:
        padded += (1 - remainder) % 8
    return padded


def _aligned_output_size(height: int, width: int, scale: float) -> tuple[int, int]:
    target_h = max(1, int(height * scale))
    target_w = max(1, int(width * scale))
    return max(128, math.ceil(target_h / 128) * 128), max(128, math.ceil(target_w / 128) * 128)


def _conditioning_sizes(sample: torch.Tensor, scale: float) -> tuple[int, int, int, int]:
    _, frames, height, width = sample.shape
    output_height = max(1, int(height * scale))
    output_width = max(1, int(width * scale))
    padded_output_height, padded_output_width = _aligned_output_size(height, width, scale)
    pad_h = padded_output_height - output_height
    pad_w = padded_output_width - output_width
    if pad_h or pad_w:
        print(f"[FlashVSR] Edge padding output canvas {output_width}x{output_height} -> {padded_output_width}x{padded_output_height}; final crop restores {output_width}x{output_height}")
    return output_height, output_width, padded_output_height, padded_output_width


def _prepare_conditioning_range(sample: torch.Tensor, start: int, end: int, output_height: int, output_width: int, padded_output_height: int, padded_output_width: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    frames = int(sample.shape[1])
    pad_h = padded_output_height - output_height
    pad_w = padded_output_width - output_width
    frame_indices = [min(max(frame_idx, 0), frames - 1) for frame_idx in range(start, end)]
    lq = sample[:, frame_indices]
    if lq.dtype == torch.uint8:
        lq = lq.float().div_(127.5).sub_(1.0)
    else:
        lq = lq.detach().float().clamp_(-1.0, 1.0)
    lq = F.interpolate(lq.permute(1, 0, 2, 3).contiguous(), size=(output_height, output_width), mode="bicubic", align_corners=False)
    if pad_h or pad_w:
        lq = F.pad(lq, (0, pad_w, 0, pad_h), mode="replicate")
    return lq.clamp_(-1.0, 1.0).to(dtype=dtype).permute(1, 0, 2, 3).contiguous()


def _pad_conditioning_frames(lq_video: torch.Tensor, target_frames: int) -> torch.Tensor:
    missing = target_frames - lq_video.shape[2]
    if missing <= 0:
        return lq_video[:, :, :target_frames]
    tail = lq_video[:, :, -1:].repeat(1, 1, missing, 1, 1)
    return torch.cat([lq_video, tail], dim=2)


def _crop_output_frames(frames: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if frames.shape[-2:] == (height, width):
        return frames
    return frames[..., :height, :width].contiguous()


def _shift_spatial_replicate(tensor: torch.Tensor, shift_y: int, shift_x: int) -> torch.Tensor:
    if shift_y == 0 and shift_x == 0:
        return tensor.clone()
    height, width = tensor.shape[-2:]
    shift_y = max(1 - height, min(height - 1, int(shift_y)))
    shift_x = max(1 - width, min(width - 1, int(shift_x)))
    crop = tensor[..., max(0, -shift_y):height - max(0, shift_y), max(0, -shift_x):width - max(0, shift_x)]
    return F.pad(crop, (max(0, shift_x), max(0, -shift_x), max(0, shift_y), max(0, -shift_y)), mode="replicate")


def _apply_still_image_shift_correction(base: torch.Tensor, shifted: torch.Tensor, shift_y: int, shift_x: int) -> torch.Tensor:
    working_bytes_per_frame = base[:, :1].numel() * (shifted.element_size() + 2 * torch.finfo(torch.float32).bits // 8)
    chunk_frames = max(1, min(base.shape[1], FLASHVSR_SHIFT_BLEND_WORKING_MEMORY_MB * 1024**2 // working_bytes_per_frame))
    for start in range(0, base.shape[1], chunk_frames):
        end = min(start + chunk_frames, base.shape[1])
        base_chunk = base[:, start:end]
        shifted_chunk = _shift_spatial_replicate(shifted[:, start:end], shift_y, shift_x)
        base_float = base_chunk.to(dtype=torch.float32, copy=True)
        shifted_float = shifted_chunk.to(dtype=torch.float32)
        base_float.lerp_(shifted_float, float(FLASHVSR_STILL_IMAGE_SHIFT_BLEND))
        if base.dtype == torch.uint8:
            base_float.round_().clamp_(0, 255)
        else:
            base_float.clamp_(-1.0, 1.0)
        base_chunk.copy_(base_float)
        del base_chunk, shifted_chunk, base_float, shifted_float
    return base


def _shift_continue_cache(continue_cache: Any, shift_y: int, shift_x: int) -> Any:
    if not isinstance(continue_cache, dict):
        return continue_cache
    tail = continue_cache.get("tail_frames")
    if not torch.is_tensor(tail) or tail.ndim != 4:
        return continue_cache
    shifted_cache = dict(continue_cache)
    shifted_cache["tail_frames"] = _shift_spatial_replicate(tail, shift_y, shift_x)
    return shifted_cache


def _two_pass_shifted_continue_cache(continue_cache: Any, shift_y: int, shift_x: int) -> Any:
    if not isinstance(continue_cache, dict):
        return continue_cache
    tail = continue_cache.get("tail_frames_shifted")
    if not torch.is_tensor(tail) or tail.ndim != 4:
        return _shift_continue_cache(continue_cache, shift_y, shift_x)
    shifted_cache = dict(continue_cache)
    shifted_cache["tail_frames"] = tail
    return shifted_cache


def _make_two_pass_continue_cache(base_cache: Any, shifted_cache: Any, shift_y: int, shift_x: int, out_shift_y: int, out_shift_x: int) -> Any:
    if not isinstance(base_cache, dict):
        return base_cache
    cache = dict(base_cache)
    shifted_tail = shifted_cache.get("tail_frames") if isinstance(shifted_cache, dict) else None
    if torch.is_tensor(shifted_tail) and shifted_tail.ndim == 4:
        cache["tail_frames_shifted"] = shifted_tail.contiguous()
        cache.update({"two_pass": True, "shift_y": shift_y, "shift_x": shift_x, "out_shift_y": out_shift_y, "out_shift_x": out_shift_x})
    return cache


def _select_still_image_frame(frames: torch.Tensor, frame_index: int) -> torch.Tensor:
    return frames[:, :, frame_index:frame_index + 1].contiguous() if frames.ndim == 5 else frames[:, frame_index:frame_index + 1].contiguous()


def _decoded_frames_to_cpu(frames: torch.Tensor, frame_count: int, height: int, width: int) -> torch.Tensor:
    frames = frames.detach()[0, :, :frame_count, :height, :width]
    if frames.device.type == "cpu" and frames.dtype in (torch.float32, torch.uint8) and frames.is_contiguous():
        return frames
    if frames.dtype == torch.uint8:
        return frames.cpu().contiguous()
    frames_cpu = torch.empty(tuple(frames.shape), dtype=torch.float32, device="cpu")
    frames_cpu.copy_(frames)
    return frames_cpu


def _normalized_video_to_cpu_uint8(frames: torch.Tensor) -> torch.Tensor:
    if frames.dtype == torch.uint8:
        return frames.detach().cpu().contiguous()
    return frames.detach().cpu().float().clamp_(-1.0, 1.0).add_(1.0).mul_(127.5).round_().clamp_(0, 255).to(torch.uint8).contiguous()


def _drop_module_tensors(module: torch.nn.Module | None) -> None:
    if module is None:
        return
    for submodule in module.modules():
        submodule._parameters.clear()
        submodule._buffers.clear()


def _save_still_image_debug_video(frames: torch.Tensor) -> None:
    if not FLASHVSR_SAVE_STILL_IMAGE_DEBUG_VIDEO:
        return
    path = os.path.abspath(FLASHVSR_STILL_IMAGE_DEBUG_VIDEO_PATH)
    try:
        from shared.utils.audio_video import save_video
        debug_frames = frames.detach().cpu()
        save_video(tensor=debug_frames, save_file=path, fps=FLASHVSR_STILL_IMAGE_DEBUG_VIDEO_FPS, nrow=1, normalize=True, value_range=(-1, 1), codec_type="libx264_8", container="mp4")
        print(f"[FlashVSR] Still image debug video saved to {path} ({int(debug_frames.shape[2])} frames)")
        del debug_frames
    except Exception as exc:
        print(f"[FlashVSR] Failed to save still image debug video: {exc}")


def _nested_tensors_to(value: Any, device: torch.device | str, dtype: torch.dtype | None = None) -> Any:
    if torch.is_tensor(value):
        return value.detach().to(device=device, dtype=dtype or value.dtype)
    if isinstance(value, list):
        return [_nested_tensors_to(item, device, dtype) for item in value]
    return value


def _report_progress(progress_callback, phase: str, current_step: int | None = None, total_steps: int | None = None, unit: str | None = None) -> None:
    if callable(progress_callback):
        progress_callback(phase, current_step, total_steps, *([unit] if unit else []))


def _abort_requested(abort_callback) -> bool:
    return callable(abort_callback) and abort_callback()


def _apply_continue_cache(frames: torch.Tensor, continue_cache: Any) -> torch.Tensor:
    if not isinstance(continue_cache, dict):
        return frames
    tail = continue_cache.get("tail_frames")
    if not torch.is_tensor(tail) or tail.ndim != 4:
        return frames
    if tail.shape[0] != frames.shape[0] or tail.shape[-2:] != frames.shape[-2:]:
        return frames
    overlap = min(int(tail.shape[1]), int(frames.shape[1]))
    if overlap <= 0:
        return frames
    if frames.dtype == torch.uint8:
        if tail.dtype != torch.uint8:
            tail = tail.float().clamp(-1.0, 1.0).add(1.0).mul_(127.5).round_().clamp_(0, 255).to(torch.uint8)
        frames[:, :overlap].copy_(tail[:, -overlap:].to(device=frames.device))
        return frames
    if tail.dtype == torch.uint8:
        tail = tail.to(device=frames.device, dtype=frames.dtype).div(127.5).sub(1.0)
    else:
        tail = tail.to(device=frames.device, dtype=frames.dtype)
    frames[:, :overlap].copy_(tail[:, -overlap:])
    return frames


def _make_continue_cache(frames: torch.Tensor, scale: float, variant: str, overlap_frames: int = FLASHVSR_CONTINUE_CACHE_FRAMES) -> dict[str, Any]:
    tail_len = min(overlap_frames, frames.shape[1])
    tail = frames[:, -tail_len:].detach().cpu()
    if tail.dtype != torch.uint8:
        tail = tail.float().clamp(-1.0, 1.0).add(1.0).mul_(127.5).round_().clamp_(0, 255).to(torch.uint8)
    return {"tail_frames": tail.contiguous(), "scale": scale, "variant": variant}


def _wavelet_color_fix(frames: torch.Tensor, lq_video: torch.Tensor) -> torch.Tensor:
    if frames.shape != lq_video[:, :, :frames.shape[2]].shape:
        return frames
    for start in range(0, frames.shape[2], 4):
        end = min(start + 4, frames.shape[2])
        frame_chunk = frames[:, :, start:end]
        lq_chunk = lq_video[:, :, start:end].to(device=frames.device, dtype=frames.dtype)
        mean_frames = frame_chunk.mean(dim=(3, 4), keepdim=True)
        std_frames = frame_chunk.std(dim=(3, 4), keepdim=True).clamp_min_(1e-5)
        mean_lq = lq_chunk.mean(dim=(3, 4), keepdim=True)
        std_lq = lq_chunk.std(dim=(3, 4), keepdim=True).clamp_min_(1e-5)
        frame_chunk.sub_(mean_frames).div_(std_frames).mul_(std_lq).add_(mean_lq).clamp_(-1.0, 1.0)
    return frames


def _wavelet_color_fix_from_sample(frames: torch.Tensor, sample: torch.Tensor, *, device: torch.device, abort_callback=None, progress_callback=None) -> torch.Tensor | None:
    """Per frame and channel, the output's mean and standard deviation matched to the source's. frames (1, 3, T, H, W) stay in CPU RAM
    and are written in place; the work goes to the GPU by chunks of frames of one channel (contiguous slices: one transfer each way per
    chunk, converted on the GPU, FP32 statistics), the chunk sized by FLASHVSR_COLOR_FIX_WORKING_MEMORY_MB."""
    total = min(int(frames.shape[2]), int(sample.shape[1]))
    frame_bytes = frames[0, 0, 0].numel() * (5 + frames.element_size())  # the FP32 copy, the staged chunk and its converted result
    chunk = max(1, min(total, FLASHVSR_COLOR_FIX_WORKING_MEMORY_MB * 1024**2 // frame_bytes))
    _report_progress(progress_callback, "Color Correction", 0, total, "frames")
    last_progress = time.monotonic()
    with tqdm(total=total, desc="Color Correction", unit="frames", mininterval=1 / 3) as progress:
        for start in range(0, total, chunk):
            if _abort_requested(abort_callback):
                return None
            end = min(start + chunk, total)
            for channel in range(frames.shape[1]):
                out = frames[0, channel, start:end]
                out_float = out.to(device=device).to(torch.float32)  # transferred as stored, converted on the GPU
                lq = sample[channel, start:end].to(device=device).to(torch.float32, copy=True)  # never the caller's tensor: the scaling below is in place
                if sample.dtype != torch.uint8 and frames.dtype == torch.uint8:
                    lq.clamp_(-1.0, 1.0).add_(1.0).mul_(127.5)
                elif sample.dtype == torch.uint8 and frames.dtype != torch.uint8:
                    lq.div_(127.5).sub_(1.0)
                elif sample.dtype != torch.uint8:
                    lq.clamp_(-1.0, 1.0)
                mean_frames = out_float.mean(dim=(1, 2), keepdim=True)
                std_frames = out_float.std(dim=(1, 2), keepdim=True).clamp_min_(1e-5)
                mean_lq = lq.mean(dim=(1, 2), keepdim=True)
                std_lq = lq.std(dim=(1, 2), keepdim=True).clamp_min_(1e-5)
                out_float.sub_(mean_frames).div_(std_frames).mul_(std_lq).add_(mean_lq)
                out.copy_(out_float.round_().clamp_(0, 255).to(torch.uint8) if frames.dtype == torch.uint8 else out_float.clamp_(-1.0, 1.0))
                del out_float, lq, mean_frames, std_frames, mean_lq, std_lq
            progress.update(end - start)
            now = time.monotonic()
            if end == total or now - last_progress >= 1 / 3:
                _report_progress(progress_callback, "Color Correction", end, total, "frames")
                last_progress = now
    return None if _abort_requested(abort_callback) else frames


@dataclass
class StreamStep:
    latents: tuple[int, int]  # the latent frames denoised by this step
    lq_clips: list[tuple[int, int]]  # the source frame ranges fed to the LQ projection, in order (the first chunk starts with its 1-frame warm-up)
    lq_end: int  # the end of the decoder's source frame range for this step (its start is the previous step's end)
    rope_start: int  # the temporal RoPE position of the step's first latent frame


@dataclass
class StreamSchedule:
    steps: list[StreamStep]
    latent_frames: int  # latent frames of the whole run
    still_frames: int  # frames decoded by the first step, the last of which is the still-image output


def _v11_schedule(input_frames: int, still_image: bool, optimize_still_image: bool) -> StreamSchedule:
    """FlashVSR v1.1: 6 latent frames (7 LQ clips) at the first step, then 2 per step, 8 source frames per step; a still image denoises 2."""
    num_frames = _next_conditioning_frame_count(input_frames)
    first_latents = 2 if optimize_still_image else 6
    first_lq_steps = first_latents + 1 if optimize_still_image else 7
    steps = []
    for process_idx in range((num_frames - 1) // 8 - 2):
        if process_idx == 0:
            steps.append(StreamStep((0, first_latents), [(max(0, inner * 4 - 3), (inner + 1) * 4 - 3) for inner in range(first_lq_steps)], 1 if optimize_still_image else 21, 0))
        else:
            steps.append(StreamStep((4 + process_idx * 2, 6 + process_idx * 2), [(process_idx * 8 + 17 + inner * 4, process_idx * 8 + 21 + inner * 4) for inner in range(2)], process_idx * 8 + 21, 4 + process_idx * 2))
    return StreamSchedule(steps, first_latents if still_image else (num_frames - 1) // 4, (first_latents - 1) * 4 + 1)


def _plus_schedule(input_frames: int) -> StreamSchedule:
    """FlashVSR+: chunks of 4 latent frames, the source padded by repetition to 4n+1 frames and to whole chunks (the conditioning ranges
    clamp to the last frame); the first step decodes 13 frames, the others 16."""
    latents = (max(1, input_frames) - 1 + 3) // 4 + 1
    chunks = (latents + FLASHVSR_PLUS_CHUNK_LATENTS - 1) // FLASHVSR_PLUS_CHUNK_LATENTS
    steps = []
    for process_idx in range(chunks):
        first_latent = process_idx * FLASHVSR_PLUS_CHUNK_LATENTS
        clips = [(0, 1)] if process_idx == 0 else []
        clips += [(1 + 4 * latent, 5 + 4 * latent) for latent in range(first_latent, first_latent + FLASHVSR_PLUS_CHUNK_LATENTS)]
        steps.append(StreamStep((first_latent, first_latent + FLASHVSR_PLUS_CHUNK_LATENTS), clips, 4 * (first_latent + FLASHVSR_PLUS_CHUNK_LATENTS) - 3, first_latent))
    return StreamSchedule(steps, chunks * FLASHVSR_PLUS_CHUNK_LATENTS, 4 * FLASHVSR_PLUS_CHUNK_LATENTS - 3)


def _plus_local_layers_structural(block_h: int, block_w: int, seqlen: int, cached_windows: int, topk: int) -> bool:
    """Whether the draft top-k budget covers every candidate of the local layers, so their block map is structural (no scoring)."""
    _, candidates, entries = structural_block_candidates(block_h, block_w, seqlen, cached_windows, FLASHVSR_PLUS_LOCAL_LAYER_RANGE)
    return candidates <= min(entries - 1, topk)


def _plus_layer_configs(dit: WanModel, model: str, block_h: int, block_w: int, seqlen: int, chunk_index: int, cache_windows: int, topk: int, device) -> list[dict]:
    """Per block: the local range, cached windows and top-k grouping of the FlashVSR+ model; the local layers share one structural
    block map per step when the budget covers their candidates (section 5 of the assessment), the scored 3x3 draft otherwise."""
    local_layers = FLASHVSR_PLUS_LOCAL_LAYERS[model]
    groups = seqlen // FLASHVSR_PLUS_TOPK_GROUP_WINDOWS
    draft = {"local_range": FLASHVSR_PLUS_LOCAL_RANGE, "kv_len": cache_windows, "topk_groups": groups, "block_mask": None}
    cached = 0 if chunk_index == 0 else FLASHVSR_PLUS_LOCAL_LAYER_CACHE_WINDOWS
    mask = structural_block_mask(block_h, block_w, seqlen, cached, FLASHVSR_PLUS_LOCAL_LAYER_RANGE, dit.blocks[0].self_attn.num_heads, device) if _plus_local_layers_structural(block_h, block_w, seqlen, cached, topk) else None
    local = {"local_range": FLASHVSR_PLUS_LOCAL_LAYER_RANGE, "kv_len": FLASHVSR_PLUS_LOCAL_LAYER_CACHE_WINDOWS, "topk_groups": groups, "block_mask": mask}
    return [local if block_id in local_layers else draft for block_id in range(len(dit.blocks))]


def _denoise_stream_chunk(
    dit: WanModel,
    x: torch.Tensor,
    context: torch.Tensor | None,
    lq_layer_chunks: list[list[torch.Tensor | None]],
    stream_cache: StreamCache,
    chunk_index: int,
    timestep_embed: torch.Tensor,
    timestep_mod: torch.Tensor,
    *,
    topk_ratio: float = 2.0,
    kv_ratio: float = FLASHVSR_KV_CACHE_WINDOWS,
    local_range: int = 9,
    cache_next: bool = True,
    allow_short_start: bool = False,
    rope_start: int | None = None,
    plus_model: str | None = None,
    abort_callback=None,
) -> torch.Tensor | None:
    x, (frames, height, width) = dit.patchify(x)
    win = (2, 8, 8)
    seqlen = frames // win[0]
    window_size = win[0] * height * width // 128
    window_topk = int(window_size * window_size * topk_ratio) - 1  # upstream's budget in (query window, key window) pairs
    topk = int(window_size * window_size * topk_ratio * FLASHVSR_PLUS_HALF_WINDOW_BUDGET) - 1 if plus_model else window_topk
    kv_len = max(1, int(kv_ratio))
    if rope_start is None:
        rope_start = 0 if chunk_index == 0 else 4 + chunk_index * 2
    freqs_t = dit.freqs[0][rope_start:rope_start + frames]
    freqs = tuple((freq.real.to(device=x.device, dtype=x.dtype), freq.imag.to(device=x.device, dtype=x.dtype)) for freq in (freqs_t, dit.freqs[1][:height], dit.freqs[2][:width]))
    layer_configs = _plus_layer_configs(dit, plus_model, height // 8, width // 8, seqlen, chunk_index, kv_len, window_topk, x.device) if plus_model else None
    dit._stream_storing = stream_cache.storing
    for block_id, block in enumerate(dit.blocks):
        layer = layer_configs[block_id] if layer_configs else {"local_range": local_range, "kv_len": kv_len, "topk_groups": None, "block_mask": None}
        if _abort_requested(abort_callback):
            return None
        if block_id < len(lq_layer_chunks[0]):
            offset = 0
            for chunk in lq_layer_chunks:
                lq = chunk[block_id].to(x.device, dtype=x.dtype)
                next_offset = offset + lq.shape[1]
                x[:, offset:next_offset].add_(lq)
                offset = next_offset
                chunk[block_id] = None
                del lq
        x_ref = [x]
        x = None
        x = block(
            x_ref, context, timestep_mod, freqs, frames, height, width, seqlen, topk,
            block_id=block_id, kv_len=layer["kv_len"], is_stream=True,
            stream_cache=stream_cache, local_range=layer["local_range"], cache_next=cache_next, allow_short_start=allow_short_start,
            topk_groups=layer["topk_groups"], block_mask=layer["block_mask"],
        )
        x_ref.clear()
        if _abort_requested(abort_callback):
            return None
    x = dit.head([x], timestep_embed)
    return dit.unpatchify([x], (frames, height, width))


class FlashVSRRuntime:
    def __init__(self) -> None:
        self.variant: str | None = None
        self.dtype = torch.bfloat16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dit: WanModel | None = None
        self.lq_proj: Causal_LQ4x_Proj | None = None
        self.tcdecoder: torch.nn.Module | None = None
        self.vae: WanVAE | None = None
        self.offloadobj = None
        self.prompt_context: torch.Tensor | None = None
        self.timestep: torch.Tensor | None = None
        self.timestep_embed: torch.Tensor | None = None
        self.timestep_mod: torch.Tensor | None = None
        self.profile = None
        self.model: str = FLASHVSR_MODEL_V11
        self.transformer_path: str | None = None
        self._interrupt = False  # read by the shared phase progress of the Wan VAE decode (generation_progress)

    @generation_progress
    def _decode_with_vae(self, latents: torch.Tensor, tile_size: int, *, callback=None, set_progress_status=None, **decode_kwargs) -> torch.Tensor:
        """The Wan VAE decode under the shared phase progress: its tile events reach `callback` with the title, total and unit."""
        return self.vae.decode_to_cpu_uint8([latents], tile_size, **decode_kwargs)[0]

    @controlled_model_loading
    def load(self, paths: FlashVSRPaths, variant: str, profile, init_pipe, model: str = FLASHVSR_MODEL_V11) -> None:
        require_sparge_attention()
        variant = variant or FLASHVSR_VARIANT_TINY_LONG
        profile = 0
        if self.dit is not None and self.variant == variant and self.profile == profile and self.model == model and self.transformer_path == paths.transformer:
            return
        self.release()
        self.variant = variant
        self.profile = profile
        self.model = model
        self.transformer_path = paths.transformer
        with init_empty_weights(include_buffers=True), _default_dtype(self.dtype):
            self.dit = WanModel(**WAN_1_3B_CONFIG).eval()
            self.lq_proj = Causal_LQ4x_Proj(in_dim=3, out_dim=1536, layer_num=1).eval()
        self.dit._offload_hooks = ["reinit_cross_kv"]
        self.lq_proj._offload_hooks = ["stream_forward"]
        offload.load_model_data(self.dit, paths.transformer, writable_tensors=False, preprocess_sd=_preprocess_transformer_state_dict, default_dtype=self.dtype, ignore_unused_weights=True, verboseLevel=-1)
        self._save_quantized_if_requested(paths.transformer)
        self.dit.freqs = precompute_freqs_cis_3d(WAN_1_3B_CONFIG["dim"] // WAN_1_3B_CONFIG["num_heads"])
        offload.load_model_data(self.lq_proj, paths.lq_proj, writable_tensors=False, default_dtype=self.dtype, verboseLevel=-1)
        self.dit.requires_grad_(False)
        self.lq_proj.requires_grad_(False)
        self.prompt_context = load_file(paths.posi_prompt, device="cpu")["context"].to(self.dtype)
        pipe = {"transformer": self.dit, "lq_proj": self.lq_proj}
        if variant in (FLASHVSR_VARIANT_TINY, FLASHVSR_VARIANT_TINY_LONG):
            self.tcdecoder = build_tcdecoder(new_channels=[512, 256, 128, 128], device="cpu", dtype=self.dtype, new_latent_channels=16 + 768).eval()
            self.tcdecoder._offload_hooks = ["decode_video"]
            offload.load_model_data(self.tcdecoder, paths.tcdecoder, writable_tensors=False, default_dtype=self.dtype, ignore_unused_weights=True, verboseLevel=-1)
            self.tcdecoder.requires_grad_(False)
            pipe["tcdecoder"] = self.tcdecoder
        else:
            self.vae = WanVAE(vae_pth=paths.vae, dtype=self.dtype, upsampler_factor=1, device="cpu")
            self.vae.device = self.device
            self.vae.model.requires_grad_(False)
            pipe["vae"] = self.vae.model
        kwargs = {"coTenantsMap": FLASHVSR_COTENANTS_MAP}
        profile_no = init_pipe(pipe, kwargs, profile)
        self.cache_in_ram = minimal_vram(kwargs)  # Postprocessor VRAM Usage: Minimum keeps the stream cache in RAM
        kwargs["pinnedMemory"] = False
        self.offloadobj = offload.profile(pipe, loading_callback=loading_callback(), profile_no=profile_no, quantizeTransformer=False, convertWeightsFloatTo=self.dtype, verboseLevel=-1, **kwargs)
        offload_registry.register_offloadobj("FlashVSR", self.offloadobj, self.release)
        log_sparse_backend()

    def _save_quantized_if_requested(self, source_path: str) -> None:
        """`--save-quantized` (with `--convrot` for INT8 ConvRot, the default format; Quanto otherwise): the transformer's quantized
        file written beside its BF16 source, once, through the shared converters."""
        args = getattr(sys.modules.get("wgp"), "args", None)
        if args is None or not getattr(args, "save_quantized", False) or "_bf16" not in os.path.basename(source_path):
            return
        convrot = bool(getattr(args, "convrot", False))
        target = os.path.join(os.path.dirname(source_path), os.path.basename(source_path).replace("_bf16", "_int8_convrot" if convrot else "_quanto_bf16_int8"))
        if os.path.isfile(target):
            print(f"[FlashVSR] Quantized transformer '{os.path.basename(target)}' already exists")
            return
        if convrot:
            from shared.convert.convrot import save_convrot_model
            save_convrot_model(self.dit, source_path, target, verboseLevel=1)
        else:
            offload.save_model(self.dit, target, do_quantize=True)
        print(f"[FlashVSR] Quantized transformer saved to '{target}'")

    def _prepare_run_state(self) -> None:
        if self.device.type != "cuda":
            raise RuntimeError("FlashVSR requires CUDA.")
        context = self.prompt_context.to(self.device, dtype=self.dtype)
        self.dit.reinit_cross_kv(context)
        self.timestep = torch.tensor([1000.0], device=self.device, dtype=self.dtype)
        self.timestep_embed = self.dit.time_embedding(_sinusoidal_embedding_1d(self.dit.freq_dim, self.timestep))
        self.timestep_mod = self.dit.time_projection(self.timestep_embed).unflatten(1, (6, self.dit.dim))

    def _clear_runtime_caches(self) -> None:
        if self.dit is not None:
            self.dit.clear_cross_kv()
        if self.lq_proj is not None:
            self.lq_proj.clear_cache()
        if self.tcdecoder is not None:
            self.tcdecoder.clean_mem()
        if self.vae is not None:
            self.vae.model.clear_cache()
        self.timestep = None
        self.timestep_embed = None
        self.timestep_mod = None

    def _unload_mmgp(self) -> None:
        self._clear_runtime_caches()
        if self.offloadobj is not None:
            self.offloadobj.unload_all()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _decode_tcdecoder(self, latents: torch.Tensor, sample: torch.Tensor, lq_start: int, lq_end: int, output_height: int, output_width: int, padded_output_height: int, padded_output_width: int, tile_size: int, tile_mems: dict[tuple[int, int], Any] | None, abort_callback=None, progress_callback=None, progress_step: int | None = None, progress_total: int | None = None) -> tuple[torch.Tensor | None, dict[tuple[int, int], Any] | None]:
        if self.tcdecoder is None:
            raise RuntimeError("FlashVSR tiny variants require TCDecoder.")
        _report_progress(progress_callback, "TCDecoder Decoding", progress_step, progress_total)
        tile_size = int(tile_size or 0)
        cur_lq = _prepare_conditioning_range(sample, lq_start, lq_end, output_height, output_width, padded_output_height, padded_output_width, dtype=self.dtype).unsqueeze(0)
        if tile_size <= 0 or (padded_output_height <= tile_size and padded_output_width <= tile_size):
            cur_lq = cur_lq.to(self.device, dtype=self.dtype)
            frames = self.tcdecoder.decode_video(latents.transpose(1, 2), parallel=False, show_progress_bar=False, cond=cur_lq).transpose(1, 2).mul_(2).sub_(1)
            del cur_lq
            _report_progress(progress_callback, "TCDecoder Decoding", progress_step + 1 if progress_step is not None else None, progress_total)
            return _normalized_video_to_cpu_uint8(frames), tile_mems

        # exact tiles (tile_plan): each decoder module computes only what the next ones need; per tile the MemBlock memory of its regions
        latent_tile = max(1, tile_size // 8)
        latent_height = padded_output_height // 8
        latent_width = padded_output_width // 8
        tile_mems = {} if tile_mems is None else tile_mems
        frames_out = None
        for latent_y0 in range(0, latent_height, latent_tile):
            latent_y1 = min(latent_y0 + latent_tile, latent_height)
            write_y0, write_y1 = latent_y0 * 8, min(latent_y1 * 8, output_height)
            if write_y1 <= write_y0:
                continue
            for latent_x0 in range(0, latent_width, latent_tile):
                if _abort_requested(abort_callback):
                    del cur_lq
                    return None, tile_mems
                latent_x1 = min(latent_x0 + latent_tile, latent_width)
                write_x0, write_x1 = latent_x0 * 8, min(latent_x1 * 8, output_width)
                if write_x1 <= write_x0:
                    continue
                ((y0, y1), (x0, x1)), crops = tile_plan(self.tcdecoder.decoder, (latent_height, latent_width), ((write_y0, write_y1), (write_x0, write_x1)))
                tile_key = (latent_y0, latent_x0)
                saved_mem = tile_mems.get(tile_key)
                if saved_mem is None:
                    self.tcdecoder.clean_mem()
                else:
                    self.tcdecoder.mem = _nested_tensors_to(saved_mem, self.device, self.dtype)
                cur_lq_tile = cur_lq[:, :, :, y0 * 8:y1 * 8, x0 * 8:x1 * 8].contiguous().to(self.device, dtype=self.dtype)
                cur_latents = latents[:, :, :, y0:y1, x0:x1].to(self.device, dtype=self.dtype)
                tile_frames = self.tcdecoder.decode_video(cur_latents.transpose(1, 2), parallel=False, show_progress_bar=False, cond=cur_lq_tile, crops=crops).transpose(1, 2).mul_(2).sub_(1)
                tile_mems[tile_key] = _nested_tensors_to(self.tcdecoder.mem, "cpu")
                self.tcdecoder.clean_mem()
                if frames_out is None:
                    frames_out = torch.empty((tile_frames.shape[0], tile_frames.shape[1], tile_frames.shape[2], output_height, output_width), dtype=torch.uint8, device="cpu")
                tile_cpu = _normalized_video_to_cpu_uint8(tile_frames)
                frames_out[:, :, :, write_y0:write_y1, write_x0:write_x1].copy_(tile_cpu)
                del cur_lq_tile, cur_latents, tile_frames, tile_cpu
        del cur_lq
        _report_progress(progress_callback, "TCDecoder Decoding", progress_step + 1 if progress_step is not None else None, progress_total)
        return frames_out, tile_mems

    def release(self) -> None:
        self._clear_runtime_caches()
        if self.offloadobj is not None:
            offload_registry.unregister_offloadobj("FlashVSR", self.offloadobj)
            self.offloadobj.release()
            self.offloadobj = None
        _drop_module_tensors(self.dit)
        _drop_module_tensors(self.lq_proj)
        _drop_module_tensors(self.tcdecoder)
        _drop_module_tensors(self.vae.model if self.vae is not None else None)
        self.dit = None
        self.lq_proj = None
        self.tcdecoder = None
        self.vae = None
        self.prompt_context = None
        self.timestep = None
        self.timestep_embed = None
        self.timestep_mod = None
        self.variant = None
        self.profile = None
        self.model = FLASHVSR_MODEL_V11
        self.transformer_path = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def upscale(
        self,
        sample: torch.Tensor,
        scale: float,
        *,
        seed: int = 0,
        continue_cache: Any = None,
        return_continue_cache: bool = False,
        vae_tile_size: int | None = None,
        topk_ratio: float = FLASHVSR_TOPK_RATIO,
        still_image: bool = False,
        cache_windows: int = FLASHVSR_PLUS_CACHE_WINDOWS,
        abort_callback=None,
        progress_callback=None,
    ) -> tuple[torch.Tensor | None, dict[str, Any] | None]:
        try:
            return self._upscale_impl(sample, scale, seed=seed, continue_cache=continue_cache, return_continue_cache=return_continue_cache, vae_tile_size=vae_tile_size, topk_ratio=topk_ratio, still_image=still_image, cache_windows=cache_windows, abort_callback=abort_callback, progress_callback=progress_callback)
        finally:
            self._clear_runtime_caches()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @torch.inference_mode()
    def _upscale_impl(
        self,
        sample: torch.Tensor,
        scale: float,
        *,
        seed: int = 0,
        continue_cache: Any = None,
        return_continue_cache: bool = False,
        vae_tile_size: int | None = None,
        topk_ratio: float = FLASHVSR_TOPK_RATIO,
        still_image: bool = False,
        cache_windows: int = FLASHVSR_PLUS_CACHE_WINDOWS,
        abort_callback=None,
        progress_callback=None,
    ) -> tuple[torch.Tensor | None, dict[str, Any] | None]:
        if self.dit is None or self.lq_proj is None:
            raise RuntimeError("FlashVSR models are not loaded.")
        def abort_result():
            return None, None

        input_frames = sample.shape[1]
        plus = self.model in FLASHVSR_PLUS_MODELS
        output_height, output_width, padded_output_height, padded_output_width = _conditioning_sizes(sample, scale)
        configured_topk_ratio = max(0.0, min(4.0, float(topk_ratio or 0.0)))
        if configured_topk_ratio > 0:
            topk_ratio = configured_topk_ratio
            print(f"[FlashVSR] Sparse top-k ratio fixed to {topk_ratio:.3f}")
        elif plus:
            topk_ratio = FLASHVSR_PLUS_TOPK_RATIO
        else:
            raw_topk_ratio = min(2.0, 2.0 * 768 * 1280 / max(int(padded_output_height) * int(padded_output_width), 1))
            topk_ratio = max(raw_topk_ratio, FLASHVSR_FULL_MIN_AUTO_TOPK_RATIO)
            if topk_ratio != raw_topk_ratio:
                print(f"[FlashVSR] Sparse top-k ratio adjusted to {topk_ratio:.3f} for {padded_output_width}x{padded_output_height} (minimum; raw auto {raw_topk_ratio:.3f})")
            elif topk_ratio < 2.0:
                print(f"[FlashVSR] Sparse top-k ratio adjusted to {topk_ratio:.3f} for {padded_output_width}x{padded_output_height}")
        self._prepare_run_state()
        self.lq_proj.clear_cache()
        if self.tcdecoder is not None:
            self.tcdecoder.clean_mem()
        if self.vae is not None:
            self.vae.model.clear_cache()
        cache_windows = max(1, min(FLASHVSR_PLUS_CACHE_WINDOWS, int(cache_windows))) if plus else max(1, int(FLASHVSR_KV_CACHE_WINDOWS))
        print(f"[FlashVSR] Stream KV cache windows: {cache_windows}")
        tcdecoder_tile_size = int(vae_tile_size or 0) if self.tcdecoder is not None else 0
        tcdecoder_tile_mems = None
        if self.tcdecoder is not None:
            if tcdecoder_tile_size > 0 and (padded_output_height > tcdecoder_tile_size or padded_output_width > tcdecoder_tile_size):
                print(f"[FlashVSR] TCDecoder spatial tiling policy: tile_size={tcdecoder_tile_size}px, exact halos per stage")
                tcdecoder_tile_mems = {}
            else:
                print("[FlashVSR] TCDecoder spatial tiling policy: tile_size=0px")
        generator = torch.Generator(device="cpu").manual_seed(0 if seed is None or seed < 0 else int(seed))
        still_image = bool(still_image and input_frames == 1)
        self.lq_proj.shift_start_prefix = still_image
        optimize_still_image = still_image and not FLASHVSR_DISABLE_STILL_IMAGE_OPTIMIZATIONS
        schedule = _plus_schedule(input_frames) if plus else _v11_schedule(input_frames, still_image, optimize_still_image)
        first_chunk_latent_frames = schedule.steps[0].latents[1]
        still_output_frame = schedule.still_frames - 1 if still_image and FLASHVSR_STILL_IMAGE_RETURN_WARMED_FRAME else 0
        if plus:
            if still_image:
                print(f"[FlashVSR] Still image mode: denoising one {first_chunk_latent_frames}-latent chunk; returning decoded frame {still_output_frame}")
            block_h, block_w = padded_output_height // 128, padded_output_width // 128
            topk = int(block_h * block_w * block_h * block_w * topk_ratio) - 1
            structural = all(_plus_local_layers_structural(block_h, block_w, FLASHVSR_PLUS_CHUNK_LATENTS // 2, cached, topk) for cached in (0, FLASHVSR_PLUS_LOCAL_LAYER_CACHE_WINDOWS))
            local_layers = len(FLASHVSR_PLUS_LOCAL_LAYERS[self.model])
            print(f"[FlashVSR] FlashVSR+ hybrid attention: {local_layers} local layers ({FLASHVSR_PLUS_LOCAL_LAYER_RANGE}x{FLASHVSR_PLUS_LOCAL_LAYER_RANGE} windows, {FLASHVSR_PLUS_LOCAL_LAYER_CACHE_WINDOWS} cached window, {'structural map' if structural else 'scored draft: small canvas'}), {len(self.dit.blocks) - local_layers} draft top-k layers ({FLASHVSR_PLUS_LOCAL_RANGE}x{FLASHVSR_PLUS_LOCAL_RANGE} windows, {cache_windows} cached windows)")
        elif optimize_still_image:
            print(f"[FlashVSR] Still image mode: denoising {first_chunk_latent_frames} startup latent frames instead of 6; returning decoded frame {still_output_frame}")
        elif still_image and FLASHVSR_DISABLE_STILL_IMAGE_OPTIMIZATIONS:
            print(f"[FlashVSR] Still image debug mode: image optimizations disabled; denoising original 6 startup latent frames; returning decoded frame {still_output_frame}")
        latents = torch.empty((1, 16, schedule.latent_frames, padded_output_height // 8, padded_output_width // 8), device="cpu", dtype=self.dtype)
        latents.normal_(generator=generator)
        process_total = len(schedule.steps)
        local_layers = FLASHVSR_PLUS_LOCAL_LAYERS[self.model] if plus else ()
        attention = self.dit.blocks[0].self_attn
        windows = [FLASHVSR_PLUS_LOCAL_LAYER_CACHE_WINDOWS if block in local_layers else cache_windows for block in range(len(self.dit.blocks))]
        stream_cache = StreamCache(self.dit.blocks, windows, 2 * (padded_output_height // 16) * (padded_output_width // 16), attention.num_heads, attention.head_dim, self.dtype, self.device, self.cache_in_ram)
        frames_out = None
        frames_cursor = 0
        lq_pre_idx = 0

        def decode_step(step: StreamStep, process_idx: int, cur_latents: torch.Tensor) -> bool:
            """The step's latents through the TCDecoder, conditioned on the source frames since the previous step, into frames_out."""
            nonlocal frames_out, frames_cursor, lq_pre_idx, tcdecoder_tile_mems
            save_still_debug_video = still_image and frames_cursor == 0 and FLASHVSR_SAVE_STILL_IMAGE_DEBUG_VIDEO
            decode_lq_cur_idx = schedule.still_frames if still_image and frames_cursor == 0 else step.lq_end
            cur_frames, tcdecoder_tile_mems = self._decode_tcdecoder(cur_latents, sample, lq_pre_idx, decode_lq_cur_idx, output_height, output_width, padded_output_height, padded_output_width, tcdecoder_tile_size, tcdecoder_tile_mems, abort_callback=abort_callback, progress_callback=progress_callback, progress_step=process_idx, progress_total=process_total)
            if cur_frames is None:
                return False
            cur_frames = _crop_output_frames(cur_frames.detach().cpu(), output_height, output_width)
            if save_still_debug_video:
                _save_still_image_debug_video(cur_frames)
            if still_image and frames_cursor == 0:
                cur_frames = _select_still_image_frame(cur_frames, still_output_frame)
            copy_frames = min(int(cur_frames.shape[2]), input_frames - frames_cursor)
            if copy_frames > 0:
                if frames_out is None:
                    frames_out = torch.empty((cur_frames.shape[0], cur_frames.shape[1], input_frames, output_height, output_width), dtype=torch.uint8, device="cpu")
                frames_chunk = _normalized_video_to_cpu_uint8(cur_frames[:, :, :copy_frames])
                frames_out[:, :, frames_cursor:frames_cursor + copy_frames].copy_(frames_chunk)
                frames_cursor += copy_frames
                del frames_chunk
            lq_pre_idx = step.lq_end
            del cur_frames
            return True

        _report_progress(progress_callback, "Denoising", 0, process_total)
        try:
            for process_idx, step in enumerate(tqdm(schedule.steps, desc="FlashVSR")):
                if _abort_requested(abort_callback):
                    return abort_result()
                lq_layer_chunks = []
                for lq_start, lq_end in step.lq_clips:
                    if _abort_requested(abort_callback):
                        return abort_result()
                    lq_chunk = _prepare_conditioning_range(sample, lq_start, lq_end, output_height, output_width, padded_output_height, padded_output_width, dtype=self.dtype).unsqueeze(0).to(self.device, dtype=self.dtype)
                    lq_list = [lq_chunk]
                    del lq_chunk
                    cur = self.lq_proj.stream_forward(lq_list)
                    if cur is not None:
                        lq_layer_chunks.append(cur)
                    del cur
                latent_start, latent_end = step.latents
                cur_latents = latents[:, :, latent_start:latent_end].to(self.device, dtype=self.dtype)

                noise_pred = _denoise_stream_chunk(
                    self.dit, cur_latents, None, lq_layer_chunks, stream_cache, process_idx,
                    self.timestep_embed, self.timestep_mod, topk_ratio=topk_ratio, kv_ratio=cache_windows, cache_next=process_idx + 1 < process_total, allow_short_start=optimize_still_image and process_idx == 0,
                    rope_start=step.rope_start, plus_model=self.model if plus else None, abort_callback=abort_callback,
                )
                if noise_pred is None:
                    return abort_result()
                cur_latents = cur_latents - noise_pred
                _report_progress(progress_callback, "Denoising", process_idx + 1, process_total)
                if self.variant == FLASHVSR_VARIANT_TINY_LONG:
                    if not decode_step(step, process_idx, cur_latents):
                        return abort_result()
                else:
                    latents[:, :, latent_start:latent_end].copy_(cur_latents.detach().cpu())
                lq_layer_chunks = None
        finally:
            stream_cache.release()
        self.lq_proj.clear_cache()
        self.dit.clear_cross_kv()
        gc.collect()
        if self.variant == FLASHVSR_VARIANT_TINY_LONG:
            frames = frames_out
        else:
            if self.variant == FLASHVSR_VARIANT_TINY:
                if _abort_requested(abort_callback):
                    return abort_result()
                self.tcdecoder.clean_mem()
                frames_out = None
                frames_cursor = 0
                lq_pre_idx = 0
                for decode_idx, step in enumerate(schedule.steps):
                    if _abort_requested(abort_callback):
                        return abort_result()
                    latent_start, latent_end = step.latents
                    cur_latents = latents[:, :, latent_start:latent_end].to(self.device, dtype=self.dtype)
                    if not decode_step(step, decode_idx, cur_latents):
                        return abort_result()
                    del cur_latents
                frames = frames_out
            else:
                if _abort_requested(abort_callback):
                    return abort_result()
                _report_progress(progress_callback, "VAE Decoding")
                if self.vae is None:
                    raise RuntimeError("FlashVSR full variant requires the Wan VAE.")
                vae_tile_size = int(vae_tile_size or 0)
                print(f"[FlashVSR] Wan VAE tiling policy: tile_size={vae_tile_size}px")
                save_still_debug_video = still_image and FLASHVSR_SAVE_STILL_IMAGE_DEBUG_VIDEO
                decode_latents = latents[0, :, :first_chunk_latent_frames].contiguous() if still_image else latents[0]
                self._interrupt = False
                bars = []

                def vae_progress(step_idx=-1, _latent=None, _force_refresh=False, read_state=False, **kwargs):
                    if _abort_requested(abort_callback):
                        self._interrupt = True
                    total = kwargs.get("override_num_inference_steps")
                    if read_state or not total:
                        return
                    if not bars:
                        bars.append(tqdm(total=total, desc=kwargs.get("progress_title", "VAE Decoding"), unit=kwargs.get("progress_unit", "tiles"), mininterval=1 / 3))
                    bars[0].n = int(step_idx) + 1
                    bars[0].refresh()
                    _report_progress(progress_callback, kwargs.get("progress_title", "VAE Decoding"), int(step_idx) + 1, total, kwargs.get("progress_unit"))

                try:
                    frames = self._decode_with_vae(decode_latents, vae_tile_size, callback=vae_progress, target_frames=None if save_still_debug_video else 1 if still_image else input_frames, target_height=output_height, target_width=output_width, frame_start=0 if save_still_debug_video or not still_image else still_output_frame)
                finally:
                    for bar in bars:
                        bar.close()
                if frames is None:
                    return abort_result()
                if save_still_debug_video:
                    _save_still_image_debug_video(frames)
                if still_image:
                    frames = _select_still_image_frame(frames, still_output_frame if save_still_debug_video else 0)
        if self.tcdecoder is not None:
            self.tcdecoder.clean_mem()
        if self.vae is not None:
            self.vae.model.clear_cache()
        latents = frames_out = tcdecoder_tile_mems = None
        noise_pred = cur_latents = lq_layer_chunks = None
        lq_chunk = cur = cur_lq = cur_frames = None
        if torch.is_tensor(frames) and frames.dtype == torch.uint8 and frames.ndim == 4:
            if frames.shape[1:] != (input_frames, output_height, output_width):
                frames = frames[:, :input_frames, :output_height, :output_width].contiguous()
        else:
            decoded_frames = frames
            frames = _decoded_frames_to_cpu(decoded_frames, input_frames, output_height, output_width)
            del decoded_frames
        gc.collect()
        if _wavelet_color_fix_from_sample(frames.unsqueeze(0), sample, device=self.device, abort_callback=abort_callback, progress_callback=progress_callback) is None:
            return abort_result()
        if frames.dtype != torch.uint8:
            frames.clamp_(-1.0, 1.0)
        frames = _apply_continue_cache(frames, continue_cache)
        cache = _make_continue_cache(frames, scale, self.variant, FLASHVSR_PLUS_CONTINUE_CACHE_FRAMES if plus else FLASHVSR_CONTINUE_CACHE_FRAMES) if return_continue_cache else None
        sample = None
        return frames, cache


_RUNTIME = FlashVSRRuntime()


def continue_cache_frames(model: str) -> int:
    """The output frames of a chunk that the next source chunk regenerates (Media Flow overlap, continuation cache tail)."""
    return FLASHVSR_PLUS_CONTINUE_CACHE_FRAMES if model in FLASHVSR_PLUS_MODELS else FLASHVSR_CONTINUE_CACHE_FRAMES


def load_models(paths: FlashVSRPaths, variant: str = FLASHVSR_VARIANT_TINY_LONG, *, init_pipe, profile, model: str = FLASHVSR_MODEL_V11, progress_callback=None) -> None:
    _report_progress(progress_callback, "Caching")
    _RUNTIME.load(paths, variant, profile=profile, init_pipe=init_pipe, model=model)


@torch.inference_mode()
def upscale_video(
    sample: torch.Tensor,
    scale: float,
    *,
    seed: int = 0,
    continue_cache: Any = None,
    return_continue_cache: bool = False,
    vae_tile_size: int | None = None,
    topk_ratio: float = FLASHVSR_TOPK_RATIO,
    still_image: bool = False,
    two_pass: bool = False,
    cache_windows: int = FLASHVSR_PLUS_CACHE_WINDOWS,
    abort_callback=None,
    progress_callback=None,
) -> tuple[torch.Tensor | None, dict[str, Any] | None]:
    shift_correction = bool(
        FLASHVSR_STILL_IMAGE_SHIFT_CORRECTION
        and two_pass
    )
    run = dict(seed=seed, return_continue_cache=return_continue_cache, vae_tile_size=vae_tile_size, topk_ratio=topk_ratio, still_image=still_image, cache_windows=cache_windows, abort_callback=abort_callback, progress_callback=progress_callback)
    if shift_correction:
        shift_y, shift_x = FLASHVSR_STILL_IMAGE_SHIFT_CORRECTION_INPUT_SHIFT or (max(1, int(round(FLASHVSR_STILL_IMAGE_SHIFT_CORRECTION_PERIOD * 0.5 / scale))), 0)
        out_shift_y, out_shift_x = int(round(shift_y * scale)), int(round(shift_x * scale))
        print(f"[FlashVSR] x{scale:g} shifted two-pass blend: extra shifted pass ({shift_y}px input / {out_shift_y}px output), blend={FLASHVSR_STILL_IMAGE_SHIFT_BLEND:g}")
        base = base_cache = shifted_sample = shifted_continue_cache = shifted = shifted_cache = None
        try:
            base, base_cache = _RUNTIME.upscale(sample, scale, continue_cache=continue_cache, **run)
            if base is None:
                result = (None, None)
            else:
                shifted_sample = _shift_spatial_replicate(sample, shift_y, shift_x)
                shifted_continue_cache = _two_pass_shifted_continue_cache(continue_cache, out_shift_y, out_shift_x)
                shifted, shifted_cache = _RUNTIME.upscale(shifted_sample, scale, continue_cache=shifted_continue_cache, **run)
                result = (None, None) if shifted is None else (_apply_still_image_shift_correction(base, shifted, -out_shift_y, -out_shift_x), _make_two_pass_continue_cache(base_cache, shifted_cache, shift_y, shift_x, out_shift_y, out_shift_x))
        finally:
            base = base_cache = shifted_sample = shifted_continue_cache = shifted = shifted_cache = None
    else:
        result = _RUNTIME.upscale(sample, scale, continue_cache=continue_cache, **run)
    return result


def release_models() -> None:
    _RUNTIME.release()
