"""WanGP inference pipeline for MiniMax H3."""
from shared.utils.media_control import inference_checkpoint

import functools
import hashlib
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as audio_F
from PIL import Image
from tqdm import tqdm

from mmgp import offload
from shared.utils.loras_mutipliers import extract_loras_side, parse_loras_multipliers, update_loras_slists
from shared.utils.text_encoder_cache import TextEncoderCache
from shared.utils.phase_progress import control_video_encoding, generation_progress
from shared.utils.utils import guide_mask_to_float, guide_to_float
from shared.utils.frame_scheduler import SLASH_BLOCK_RE, floor_frame_count, normalize_frame_count, normalize_overlap
from shared.utils.utils import convert_video_tensor_to_uint8_chunked
from .constants import (H3_AUDIO_REFINEMENT_DENOISE, H3_AUDIO_REFINEMENT_SETTING, H3_AUDIO_REFINEMENT_STEPS, H3_CONTROL_LATENT_CONTINUATION,
                        H3_PHASE_2_NOISE_LEVEL_START_DEFAULT, H3_REGENERATE_LORA, h3_grouped_masking_enabled)
from .dialogue import H3_DIALOGUE_GENERATION, generate_dialogue, is_dialogue_prompt
from .excerpts import H3_AUDIO_EXCERPTS_SETTING, H3_VIDEO_EXCERPTS_SETTING, parse_excerpts, reference_video_frame_limit
from .first_block_cache import MiniMaxH3FirstBlockCache
from .interrupt import GenerationInterrupted
from .pdd import pdd_sampling_plans, pdd_sampling_plans_for_sigmas
from .pika import PIKA_SOLVER, PIKA_STEPS, PIKA_VIDEO_SHIFT, noise as pika_noise, renoise_ as pika_renoise_, sigma_schedule as pika_sigma_schedule
from .spectrum import MiniMaxH3Spectrum
from .transformer import VISUAL_COND_TIMESTEP, pack_audio, patchify_video, unpack_audio
from .video_vae import LATENTS_MEAN, LATENTS_STD
from .veda.attention import generation_context as veda_generation_context


AUDIO_SAMPLE_RATE = 32000
AUDIO_LATENT_FPS = 40
# Allow audio repair before the final overlap frame; False restores the original continuation behavior.
H3_EXTENDED_AUDIO_REPAIR = True
# Audio latents generated past the last video frame and returned as a tail (model def "audio_tail_ms"): the decoder's last (edge)
# latent and the AAC end of a saved file then lie after the video, so the audio under every frame is complete and needs no repair.
# 5 latents leave at least 112 ms: an AAC export drops up to one frame (32 ms at 32 kHz) and alters the two frames before it.
H3_AUDIO_TAIL_LATENTS = 5
SOL_ATTN_TAU_END = 0.8
H3_TWO_PHASE_SCALE = 2.0
H3_PHASE_2_SIGMAS = (H3_PHASE_2_NOISE_LEVEL_START_DEFAULT, 0.6316, 0.3158, 0.0)
H3_PHASE_2_SAMPLE_SOLVER = "euler"
# Keep phase-1 audio as frozen conditioning; True restores Regenerate's silent conditioning.
H3_REGENERATE_SILENT_AUDIO = False
H3_PHASE_2_TILE_COUNT = 4
H3_PHASE_2_TILE_OVERLAP_RATIO = 0.25
H3_PHASE_2_TILE_ALIGNMENT = 32
H3_PHASE_2_TILING_FLAG = "~"
H3_TURBO_LORA_KEY = "minimax_h3_lora_turbo"
# Per-window control videos (Depth / In-Context Guide): True = aligned guide, packed as a keyframe clip on the target's own
# RoPE time/space grid (guide frame i on output frame i) and not shown to the prompt encoder; False = reference video placed
# on its own time span before the target and presented to the prompt encoder as <Video 1>
H3_ALIGNED_CONTROL_VIDEO = False
VDN_TURBO_LORA = "minimax_h3_vdn_turbo_8step_lora_bf16.safetensors"


def _return_none_on_interrupt(method):
    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except GenerationInterrupted:
            return None

    return wrapped


def video_latent_frames(frame_count):
    if frame_count == 1:
        return 1
    frame_count = normalize_frame_count(max(5, int(frame_count)), 5, 17, 5)
    return 2 + ((frame_count - 5) // 17) * 5


def _as_video(tensor):
    if tensor is None:
        return None
    if tensor.ndim == 3:
        return tensor.unsqueeze(1)
    if tensor.ndim != 4:
        raise ValueError(f"Expected a CTHW tensor, got shape {tuple(tensor.shape)}")
    return tensor


def _resize_video(video, height, width):
    video = _as_video(video)
    if video is None or video.shape[-2:] == (height, width):
        return video
    if video.dtype == torch.uint8:  # a uint8 control video (model_def "uint8_guides"): resized by chunks of frames from WanGP's float values
        output = torch.empty((video.shape[0], video.shape[1], height, width), dtype=torch.float32, device=video.device)
        for start in range(0, video.shape[1], 16):
            output[:, start:start + 16] = _resize_video(guide_to_float(video[:, start:start + 16]), height, width)
        return output
    return F.interpolate(video.permute(1, 0, 2, 3), size=(height, width), mode="bicubic", align_corners=False, antialias=True).permute(1, 0, 2, 3)


def _spatial_tiles(length, alignment=H3_PHASE_2_TILE_ALIGNMENT):
    tile_length = min(length, math.ceil(length / (2.0 - H3_PHASE_2_TILE_OVERLAP_RATIO) / alignment) * alignment)
    return [(0, tile_length), (length - tile_length, tile_length)]


def _crop_spatial_tile(tensor, top, left, tile_height, tile_width):
    tile = tensor[..., top:min(top + tile_height, tensor.shape[-2]), left:min(left + tile_width, tensor.shape[-1])]
    pad_height, pad_width = tile_height - tile.shape[-2], tile_width - tile.shape[-1]
    return F.pad(tile, (0, pad_width, 0, pad_height), mode="replicate") if pad_height or pad_width else tile


def _phase_2_tile_weights(height, width, top_overlap=0, bottom_overlap=0,
                          left_overlap=0, right_overlap=0):
    height_weights = torch.ones(height, dtype=torch.float32, device="cpu")
    width_weights = torch.ones(width, dtype=torch.float32, device="cpu")
    if top_overlap:
        height_weights[:top_overlap] = torch.linspace(0.0, math.pi / 2.0, top_overlap, device="cpu").sin_().square_()
    if bottom_overlap:
        height_weights[-bottom_overlap:] = torch.linspace(0.0, math.pi / 2.0, bottom_overlap, device="cpu").cos_().square_()
    if left_overlap:
        width_weights[:left_overlap] = torch.linspace(0.0, math.pi / 2.0, left_overlap, device="cpu").sin_().square_()
    if right_overlap:
        width_weights[-right_overlap:] = torch.linspace(0.0, math.pi / 2.0, right_overlap, device="cpu").cos_().square_()
    return (height_weights[:, None] * width_weights[None]).view(1, 1, 1, height, width)


def _video_to_uint8_cpu(video, max_buffer_mb=64):
    output = torch.empty(video.shape, dtype=torch.uint8, device="cpu")
    frame_bytes = max(1, video.shape[0] * video.shape[-2] * video.shape[-1] * (video.element_size() + 1))
    chunk_frames = max(1, int(max_buffer_mb * 1024 * 1024) // frame_bytes)
    for start in range(0, video.shape[1], chunk_frames):
        end = min(video.shape[1], start + chunk_frames)
        pixels = video[:, start:end].add(1.0).mul(127.5).round_().clamp_(0, 255).to(torch.uint8)
        output[:, start:end].copy_(pixels.to(device="cpu", non_blocking=False))
    return output


def _build_frozen_control_video(input_frames, input_video, frame_num, prefix_frames_count):
    control = _as_video(input_frames)
    if control is None:
        raise ValueError("MiniMax H3 audio-from-control-video mode requires a Control Video")
    prefix_count = min(int(prefix_frames_count), input_video.shape[1]) if input_video is not None else 0
    output_frames = min(int(frame_num), prefix_count + control.shape[1])
    if output_frames < 5:
        raise ValueError("MiniMax H3 audio-from-control-video mode requires at least 5 Control Video frames")
    output_frames = floor_frame_count(output_frames, 5, 17, 5)
    pieces = []
    if prefix_count:
        pieces.append(input_video[:, :min(prefix_count, output_frames)])
    remaining = output_frames - sum(piece.shape[1] for piece in pieces)
    if remaining:
        control = control[:, -remaining:] if pieces and control.shape[1] > remaining else control[:, :remaining]
        pieces.append(guide_to_float(control))
    return torch.cat(pieces, dim=1) if len(pieces) > 1 else pieces[0]


def _qwen_frames(video):
    return guide_to_float(video).permute(1, 2, 3, 0).add(1.0).mul_(0.5).clamp_(0.0, 1.0)


def _fit_audio_samples(audio, sample_count):
    return F.pad(audio[..., :sample_count], (0, max(0, sample_count - audio.shape[-1])))


def _resize_video_mask(mask, latent_shape, clip_length, temporal_ratio, binarize=True):
    latent_t, latent_h, latent_w = latent_shape
    mask = guide_mask_to_float(mask[:1]).unsqueeze(0).float()
    pad_frames = (-mask.shape[2]) % clip_length
    if pad_frames:
        mask = F.pad(mask, (0, 0, 0, 0, 0, pad_frames), mode="replicate")
    offsets = [0, *range(1, clip_length, temporal_ratio)]
    intervals = list(zip(offsets, [*offsets[1:], clip_length]))
    mask = torch.stack([
        mask[:, :, clip_start + start:clip_start + stop].amax(dim=2)
        for clip_start in range(0, mask.shape[2], clip_length)
        for start, stop in intervals
    ][:latent_t], dim=2)
    if not binarize and mask.shape[-2:] == (1, 1):
        return torch.ceil(mask.clamp_(0.0, 1.0) * 256.0).div_(256.0)
    if binarize:
        mask = mask.ge(0.5).float()
        mask = F.adaptive_max_pool3d(mask, (latent_t, latent_h, latent_w))
        return mask.gt(0.5).float()
    mask = F.interpolate(mask, size=(latent_t, latent_h, latent_w), mode="nearest")
    return mask.clamp_(0.0, 1.0)


def _snap_video_mask_to_patch_cells(mask, patch_size=(1, 2, 2)):
    patch_t, patch_h, patch_w = patch_size
    cells = F.max_pool3d(mask, kernel_size=patch_size, stride=patch_size)
    return cells.repeat_interleave(patch_t, 2).repeat_interleave(patch_h, 3).repeat_interleave(patch_w, 4)


def _set_grouped_video_rows(payload, editable_mask, latent_shape, device):
    for key in ("target_video_order", "target_video_inverse_order", "target_video_fixed_rows", "target_video_mask_active"):
        payload.pop(key, None)
    if editable_mask is None:
        return False
    latent_t, latent_h, latent_w = latent_shape
    editable_rows = torch.ones(latent_t * (latent_h // 2) * (latent_w // 2), dtype=torch.bool, device=device)
    source_rows = editable_mask[0, 0, :, ::2, ::2].flatten().to(device=device, dtype=torch.bool)
    editable_rows[:source_rows.numel()] = source_rows
    fixed = (~editable_rows).nonzero().flatten()
    order = torch.cat((fixed, editable_rows.nonzero().flatten()))
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(order.numel(), device=device)
    payload["target_video_order"] = order
    payload["target_video_inverse_order"] = inverse
    payload["target_video_fixed_rows"] = fixed.numel()
    payload["target_video_mask_active"] = False
    return True


def _outpainting_frame_location(video, outpainting_dims):
    if outpainting_dims is None:
        return None
    if not isinstance(outpainting_dims, (list, tuple)) or len(outpainting_dims) != 4:
        raise ValueError("MiniMax H3 outpainting_dims must contain top, bottom, left, and right margins")
    dims = [max(0.0, float(value)) for value in outpainting_dims]
    if not any(dims):
        return None
    from shared.utils.utils import get_outpainting_frame_location

    height, width = video.shape[-2:]
    return get_outpainting_frame_location(height, width, dims, 1, quantize_margins=32)


def _build_outpainting_mask(video, outpainting_dims):
    location = _outpainting_frame_location(video, outpainting_dims)
    if location is None:
        return None
    inner_height, inner_width, top, left = location
    height, width = video.shape[-2:]
    mask = torch.ones((1, video.shape[1], height, width), dtype=torch.float32, device=video.device)
    mask[:, :, top:top + inner_height, left:left + inner_width] = 0.0
    return mask


def _encode_video_source(vae, video, device, outpainting_dims=None):
    location = _outpainting_frame_location(video, outpainting_dims)
    if location is None:
        return vae.encode(video.unsqueeze(0), device)
    inner_height, inner_width, top, left = location
    source = video[..., top:top + inner_height, left:left + inner_width]
    source_latents = vae.encode(source.unsqueeze(0), device)
    ratio = vae.spatial_compression_ratio
    latents = source_latents.new_zeros((*source_latents.shape[:-2], math.ceil(video.shape[-2] / ratio), math.ceil(video.shape[-1] / ratio)))
    latent_top, latent_left = top // ratio, left // ratio
    latents[..., latent_top:latent_top + source_latents.shape[-2], latent_left:latent_left + source_latents.shape[-1]].copy_(source_latents)
    return latents


def _uniform_latent_frame_sigma_schedule(sigmas, strength, latent_frames):
    return (sigmas.float() * strength).view(-1, 1, 1, 1, 1, 1).expand(-1, 1, 1, latent_frames, 1, 1)


def _reinject_video_source(video, source, noise, editable_mask, sigma, buffer, preserved_sigma=None):
    torch.lerp(source, noise, sigma if preserved_sigma is None else preserved_sigma, out=buffer)
    if editable_mask is None:
        video.copy_(buffer)
    else:
        video.lerp_(buffer, 1.0 - editable_mask)


def _blend_video_source(video, source, editable_mask):
    target = video[:, :, :source.shape[2]]
    target.lerp_(source.to(target), 1.0 - editable_mask.to(target))


def _masking_step_mask(editable_mask, step, denoising_start_step, mask_end_step):
    return editable_mask if editable_mask is not None and denoising_start_step <= step < mask_end_step else None


def _er_sde_step(x, denoised, sigma, sigma_next, previous_sigma, previous_previous_sigma,
                 previous_denoised, previous_denoised_d, noise, stage_used):
    active = sigma > 0
    safe_sigma = torch.where(active, sigma, torch.full_like(sigma, 0.5))
    er_lambda_s = safe_sigma / (1.0 - safe_sigma)
    er_lambda_t = sigma_next / (1.0 - sigma_next)
    alpha_t = 1.0 - sigma_next

    def noise_scaler(value):
        return value * ((value ** 0.3).exp() + 10.0)

    ratio = noise_scaler(er_lambda_t) / noise_scaler(er_lambda_s)
    updated = (alpha_t / (1.0 - sigma)) * ratio * x + alpha_t * (1.0 - ratio) * denoised
    denoised_d = None
    if stage_used >= 2:
        safe_previous_sigma = torch.where(active, previous_sigma, torch.full_like(previous_sigma, 0.5))
        previous_lambda = safe_previous_sigma / (1.0 - safe_previous_sigma)
        dt = er_lambda_t - er_lambda_s
        lambda_step_size = -dt / 200.0
        lambda_positions = er_lambda_t.unsqueeze(-1) + torch.arange(200, dtype=torch.float32, device=x.device) * lambda_step_size.unsqueeze(-1)
        scaled_positions = noise_scaler(lambda_positions)
        integral = torch.sum(1.0 / scaled_positions, dim=-1) * lambda_step_size
        denoised_d = (denoised - previous_denoised) / (er_lambda_s - previous_lambda)
        updated.add_(denoised_d * alpha_t * (dt + integral * noise_scaler(er_lambda_t)))
        if stage_used >= 3:
            safe_previous_previous_sigma = torch.where(active, previous_previous_sigma, torch.full_like(previous_previous_sigma, 0.5))
            previous_previous_lambda = safe_previous_previous_sigma / (1.0 - safe_previous_previous_sigma)
            integral_u = torch.sum((lambda_positions - er_lambda_s.unsqueeze(-1)) / scaled_positions, dim=-1) * lambda_step_size
            denoised_u = (denoised_d - previous_denoised_d) / ((er_lambda_s - previous_previous_lambda) / 2.0)
            updated.add_(denoised_u * alpha_t * (dt**2 / 2.0 + integral_u * noise_scaler(er_lambda_t)))
    noise_scale = (er_lambda_t**2 - er_lambda_s**2 * ratio**2).sqrt().nan_to_num(nan=0.0)
    updated.add_(noise * alpha_t * noise_scale)
    updated = torch.where(active, updated, x)
    if denoised_d is not None:
        denoised_d = torch.where(active, denoised_d, torch.zeros_like(denoised_d))
    return updated, denoised_d


def _res_multistep_coefficients(sigmas):
    """Precompute deterministic second-order RES weights (arXiv:2308.02157)."""
    values = [float(sigma) for sigma in sigmas]
    coefficients = []
    old_sigma_down = None
    for index, (sigma, sigma_next) in enumerate(zip(values, values[1:])):
        if old_sigma_down is None or sigma_next == 0.0:
            ratio = sigma_next / sigma
            coefficients.append((ratio, 1.0 - ratio, 0.0))
        else:
            t = -math.log(sigma)
            h = -math.log(sigma_next) - t
            c2 = (-math.log(values[index - 1]) + math.log(old_sigma_down)) / h
            phi1 = math.expm1(-h) / -h
            phi2 = (phi1 - 1.0) / -h
            coefficients.append((math.exp(-h), h * (phi1 - phi2 / c2), h * phi2 / c2))
        old_sigma_down = sigma_next
    return coefficients


def _res_multistep_update(sample, denoised, old_denoised, coefficients):
    sample_coefficient, denoised_coefficient, old_denoised_coefficient = coefficients
    sample.mul_(sample_coefficient).add_(denoised, alpha=denoised_coefficient)
    if old_denoised_coefficient:
        sample.add_(old_denoised, alpha=old_denoised_coefficient)


def _map_shifted_sigma(sigma, source_shift, target_shift):
    base_sigma = sigma / (source_shift + sigma * (1.0 - source_shift))
    return target_shift * base_sigma / (1.0 + (target_shift - 1.0) * base_sigma)


def _resolve_canvas(width, height, short_edge, max_pixels=None):
    ratio = width / height
    if ratio >= 1.0:
        target_w, target_h = short_edge * ratio, float(short_edge)
    else:
        target_w, target_h = float(short_edge), short_edge / ratio
    if max_pixels is not None and target_w * target_h > max_pixels:
        scale = math.sqrt(max_pixels / (target_w * target_h))
        target_w, target_h = target_w * scale, target_h * scale
    return max(32, round(target_h / 32) * 32), max(32, round(target_w / 32) * 32)


def _to_pil(frame):
    if isinstance(frame, Image.Image):
        return frame.convert("RGB")
    if torch.is_tensor(frame):
        if frame.ndim == 4:
            frame = frame[:, 0]
        array = frame.permute(1, 2, 0).add(1.0).mul_(127.5).round_().clamp_(0, 255).byte().cpu().numpy()
        return Image.fromarray(array)
    return Image.fromarray(np.asarray(frame).astype(np.uint8)).convert("RGB")


def _pil_to_video(image):
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().div_(127.5).sub_(1.0).unsqueeze(1)


class MiniMaxH3Pipeline:
    refinement_api = "masked_video_sigma_v1"

    def __init__(self, transformer, text_encoder, video_vae, audio_vae, latent_upscaler=None, reference_mode=False, audio_only=False, dtype=torch.bfloat16, fixed_prompt=None):
        self.transformer = transformer
        self.text_encoder = text_encoder
        self.fixed_prompt = fixed_prompt
        self.vae = video_vae
        if video_vae.upsampling:
            # Masked refiners require a decoder that preserves the input dimensions.
            self.refinement_api = None
        self.video_encoder = torch.nn.ModuleDict({"encoder": video_vae.encoder, "quant_conv": video_vae.quant_conv})
        self.video_decoder = torch.nn.ModuleDict({"post_quant_conv": video_vae.post_quant_conv, "decoder": video_vae.decoder})
        # These are profiled as separate MMGP models. Preserve the dtype selected
        # while loading the VAE instead of applying the transformer's global dtype.
        self.video_encoder._convertWeightsFloatTo = None
        self.video_decoder._convertWeightsFloatTo = None
        if torch.cuda.is_available() and torch.cuda.get_device_properties(None).total_memory >= 10 * 1024**3:
            self.video_decoder._budget = 0
        self.audio_vae = audio_vae
        self.latent_upscaler = latent_upscaler
        self._phase_2_loras = None
        self.reference_mode = bool(reference_mode)
        self.ptd = False
        self.audio_only = bool(audio_only)
        self.dialogue_whisper = None
        self.dtype = dtype
        self.text_encoder_cache = TextEncoderCache()
        self._shared_offloadobj = None
        self._private_offloadobj = None
        self._interrupt = False
        self._early_stop = False

    def set_offload_handoff(self, shared_offloadobj, private_offloadobj):
        self._shared_offloadobj = shared_offloadobj
        self._private_offloadobj = private_offloadobj

    def _use_shared_components(self):
        if self._private_offloadobj is not None:
            self._private_offloadobj.unload_all()

    def _use_transformer(self):
        if self._shared_offloadobj is not None:
            self._shared_offloadobj.unload_all()

    def detach_shared_components(self):
        self.text_encoder = self.vae = self.video_encoder = self.video_decoder = self.audio_vae = self.latent_upscaler = None
        self._shared_offloadobj = self._private_offloadobj = None

    @property
    def _interrupt(self):
        return getattr(self, "_abort", False)

    @_interrupt.setter
    def _interrupt(self, value):
        self._abort = bool(value)
        for name in ("transformer", "text_encoder", "vae", "audio_vae", "latent_upscaler", "dialogue_whisper"):
            if hasattr(self, name):
                module = getattr(self, name)
                if module is not None:
                    module._interrupt = self._abort

    @property
    def device(self):
        return torch.device("cuda" if torch.cuda.is_available() else next(self.transformer.parameters()).device)

    def get_loras_transformer(self, _get_model_recursive_prop, model_type, model_def, guidance_phases, activated_loras,
                              loras_multipliers="", transformer_loras_filenames=(), num_inference_steps=1, prompts=(), **_kwargs):
        self._phase_2_loras = None
        if int(guidance_phases) > 1 and not model_def.get("pdd", False):
            accelerator_loras, _ = extract_loras_side(activated_loras, loras_multipliers)
            self._phase_2_loras = (len(transformer_loras_filenames), len(accelerator_loras))
        if model_def.get("vdn", False):
            selected_loras = {os.path.basename(str(lora).split("|", 1)[0]).lower() for lora in activated_loras}
            selected_vdn = VDN_TURBO_LORA.lower() in selected_loras
            if selected_vdn and self._phase_2_loras is None:
                print(f"Default system '{VDN_TURBO_LORA}' LoRA and corresponding multiplier will be ignored as the user has provided it")
                return [], []
            preload_urls = _get_model_recursive_prop(model_type, "preload_URLs", return_list=True)
            turbo_lora = next(url.split("|", 1)[0] for url in preload_urls if os.path.basename(url.split("|", 1)[0]).lower() == VDN_TURBO_LORA.lower())
            return [turbo_lora], ["0;1" if selected_vdn else "1;1"]
        if self._phase_2_loras is None:
            return [], []
        # Window overrides can turn the accelerator off later, requiring the default again.
        window_overrides = any("loras_mult" in match.group(1).lower() for prompt in prompts for match in SLASH_BLOCK_RE.finditer(prompt))
        if accelerator_loras and not window_overrides:
            _, slists, error = parse_loras_multipliers(loras_multipliers, len(activated_loras), num_inference_steps, nb_phases=guidance_phases)
            if error:
                raise ValueError(f"Error parsing Loras: {error}")
            if any(value != 0 for multiplier in slists["phase2"][:len(accelerator_loras)] for value in (multiplier if isinstance(multiplier, list) else [multiplier])):
                self._phase_2_loras = None
                return [], []
        return [model_def[H3_TURBO_LORA_KEY]], ["0;1"]

    def get_trans_lora(self):
        return self.transformer, None

    def _check_abort(self):
        inference_checkpoint()
        if self._interrupt:
            raise GenerationInterrupted

    def _early_stop_requested(self):
        return bool(self._early_stop)

    def request_early_stop(self):
        self._early_stop = True

    def _set_interrupt_state(self):
        self.transformer._interrupt = self._interrupt
        if self.text_encoder is not None:
            self.text_encoder._interrupt = self._interrupt
        self.vae._interrupt = self._interrupt
        self.audio_vae._interrupt = self._interrupt
        if self.latent_upscaler is not None:
            self.latent_upscaler._interrupt = self._interrupt
        if self.dialogue_whisper is not None:
            self.dialogue_whisper._interrupt = self._interrupt

    @staticmethod
    def _update_tensor_digest(digest, tensor):
        tensor = tensor.detach()
        if tensor.device.type != "cpu":
            tensor = tensor.to("cpu")
        tensor = tensor.contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(memoryview(tensor.view(torch.uint8).reshape(-1).numpy()))

    def _prompt_cache_key(self, prompt, presentation):
        digest = hashlib.sha256()
        for item in presentation:
            digest.update(item["type"].encode())
            if "frames" in item:
                self._update_tensor_digest(digest, item["frames"])
            if "timestamps" in item:
                digest.update(repr(tuple(item["timestamps"])).encode())
        return "minimax_h3", str(self.dtype), prompt, digest.digest()

    def _encode_prompt(self, prompt, presentation):
        if self.fixed_prompt is not None:
            return self.fixed_prompt.prompt_embeds.to(device=self.device, dtype=self.dtype), self.fixed_prompt.text_token_tags

        def encode_fn(prompts):
            return [self.text_encoder.encode(prompts[0], presentation, self.device, self.dtype)]

        cache_key = self._prompt_cache_key(prompt, presentation)
        return self.text_encoder_cache.encode(encode_fn, prompt, device=self.device, cache_keys=cache_key)[0]

    def _configure_tiling(self, tile_size):
        if tile_size == 0:  # vae_config == 1: tiling explicitly disabled
            self.vae.use_tiling = False
            return
        tile = tile_size or 256
        self.vae.enable_tiling(tile_sample_min_height=tile, tile_sample_min_width=tile)

    def _encode_video(self, video, keep_all_latents=False):
        self._check_abort()
        return self.vae.encode_condition(video.unsqueeze(0), self.device, keep_all_latents=keep_all_latents).cpu()

    def _waveform(self, waveform, sample_rate):
        if waveform is None:
            return None
        audio = torch.as_tensor(waveform, dtype=torch.float32, device="cpu")
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        elif audio.ndim == 2:
            audio = audio.transpose(0, 1)
        else:
            raise ValueError(f"Expected mono or sample-major audio, got shape {tuple(audio.shape)}")
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)
        elif audio.shape[0] != 2:
            audio = audio[:2]
        sample_rate = int(sample_rate or AUDIO_SAMPLE_RATE)
        if sample_rate != AUDIO_SAMPLE_RATE:
            audio = audio_F.resample(audio, sample_rate, AUDIO_SAMPLE_RATE)
        return audio.unsqueeze(0)

    def _load_audio_reference(self, path):
        import soundfile as sf

        audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        return self._waveform(audio, sample_rate)

    def _load_video_excerpts(self, path, positions, fps, height, width):
        from shared.utils.utils import get_resampled_video_transparent, get_video_info

        source_fps, _, _, source_frames = get_video_info(path)
        excerpts = []
        for start, duration in parse_excerpts(positions, fps, source_frames / source_fps, "Reference Positions", video_frames=True):
            self._check_abort()
            frames = get_resampled_video_transparent(path, int(start * fps), round(duration * fps), fps)
            excerpts.append(torch.stack([_resize_video(frame.permute(2, 0, 1).float().div_(127.5).sub_(1.0).unsqueeze(1), height, width)[:, 0] for frame in frames], dim=1))
        return excerpts

    def _prepare_audio_references(self, sources):
        import soundfile as sf

        sources = [source for source in sources if source is not None]
        durations = [source.shape[-1] / AUDIO_SAMPLE_RATE if torch.is_tensor(source) else sf.info(source).duration for source in sources]
        max_duration = 15 / len(sources) if sources and sum(durations) > 15 else None
        waveforms = []
        for source in sources:
            if torch.is_tensor(source):
                waveform = source
            else:
                info = sf.info(source)
                frames = -1 if max_duration is None else round(max_duration * info.samplerate)
                audio, sample_rate = sf.read(source, frames=frames, dtype="float32", always_2d=True)
                waveform = self._waveform(audio, sample_rate)
            if max_duration is not None:
                waveform = waveform[..., :round(max_duration * AUDIO_SAMPLE_RATE)]
            waveforms.append(waveform)
        return waveforms

    def _encode_audio(self, waveform):
        self._check_abort()
        return self.audio_vae.encode(waveform.to(device=self.device, dtype=torch.float32)).cpu()

    def _add_image_condition(self, image, frame_index, presentation, visual_latents, keyframes, anchor=None, timestep=None):
        video = _as_video(image)
        if video is None:
            return
        video = video[:, :1]
        latent = self._encode_video(video)
        presentation.append({"type": "image", "frames": _qwen_frames(video.clone())})
        visual_latents.append(latent)
        keyframe = {"anchor": anchor or ("first" if frame_index == 0 else "last"), "latent_frame_count": latent.shape[2]}
        if anchor == "frame":
            keyframe["frame_index"] = int(frame_index)
        if timestep is not None:
            keyframe["timestep"] = timestep
        keyframes.append(keyframe)

    def _add_video_history(self, video, visual_latents, keyframes):
        with control_video_encoding():  # show VAE tile progress for videos, never for images
            latent = self._encode_video(video, keep_all_latents=True)
        visual_latents.append(latent)
        keyframes.append({"anchor": "history", "latent_frame_count": latent.shape[2]})

    def _add_audio_condition(self, latent, anchor, audio_latents, audio_keyframes):
        audio_latents.append(latent)
        audio_keyframes.append({"anchor": anchor, "latent_frame_count": latent.shape[-1]})

    def _prepare_image_reference(self, image, target_width, target_height, image_refs_relative_size):
        if image is None:
            return None
        image = _to_pil(image)
        ratio = image.width / image.height
        pixel_budget = target_width * target_height * image_refs_relative_size / 100
        target_h, target_w = _resolve_canvas(*image.size, math.sqrt(pixel_budget / max(ratio, 1 / ratio)))
        if image.size != (target_w, target_h):
            image = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
        return _pil_to_video(image)

    def _add_image_reference(self, image, target_width, target_height, image_refs_relative_size, presentation, visual_latents, refs):
        video = self._prepare_image_reference(image, target_width, target_height, image_refs_relative_size)
        if video is None:
            return
        latent = self._encode_video(video)
        if self.fixed_prompt is None:
            presentation.append({"type": "image", "frames": _qwen_frames(video.clone())})
        visual_latents.append(latent)
        refs.append({"kind": "image", "latent_h": latent.shape[-2], "latent_w": latent.shape[-1]})

    @torch.inference_mode()
    def prewarm_refinement_prompt(self, prompt, reference_images, width, height, image_refs_relative_size=100.0):
        """Populate the prompt/reference cache without loading or running either VAE."""
        self._use_shared_components()
        self._check_abort()
        presentation = []
        for image in reference_images or ():
            video = self._prepare_image_reference(image, width, height, image_refs_relative_size)
            if video is not None:
                presentation.append({"type": "image", "frames": _qwen_frames(video)})
        context, _ = self._encode_prompt(prompt, presentation)
        self._check_abort()
        context = presentation = None

    def _add_video_reference(self, video, soundtrack, fps, presentation, visual_latents, audio_latents, refs):
        video = _as_video(video)
        if video is None:
            return
        if video.shape[1] < 5 or (video.shape[1] - 5) % 17:
            raise ValueError(f"MiniMax H3 reference videos must contain 17n+5 preprocessed frames, got {video.shape[1]}")
        with control_video_encoding():
            latent = self._encode_video(video)
        audio_latent = self._encode_audio(soundtrack) if soundtrack is not None else None
        if audio_latent is not None:
            presentation.append({"type": "audio"})
            audio_latents.append(audio_latent)
        if self.fixed_prompt is None:
            sample_indices, cursor = [], 0.0
            while round(cursor) < video.shape[1]:
                if not sample_indices or round(cursor) > sample_indices[-1]:
                    sample_indices.append(round(cursor))
                cursor += fps / 2
            presentation.append({"type": "video", "frames": _qwen_frames(video[:, sample_indices].clone()),
                                 "timestamps": [index / fps for index in sample_indices]})
        visual_latents.append(latent)
        refs.append({"kind": "video_audio" if audio_latent is not None else "video", "latent_t": latent.shape[2],
                     "latent_h": latent.shape[-2], "latent_w": latent.shape[-1],
                     "ref_audio_t": 0 if audio_latent is None else audio_latent.shape[-1]})

    def _add_audio_reference(self, waveform, presentation, audio_latents, refs):
        if waveform is None:
            return
        latent = self._encode_audio(waveform)
        presentation.append({"type": "audio"})
        audio_latents.append(latent)
        refs.append({"kind": "audio", "ref_audio_t": latent.shape[-1]})

    def _prepare_control_anchors(self, frame_anchors, frame_count, height, width, prefix_latents=None):
        """Encode known frames on the target's native temporal grid, without appended keyframe rows."""
        latents = {}
        clip_length, temporal_ratio = self.vae.config.clip_length, self.vae.temporal_compression_ratio
        latents_per_clip = 1 + (clip_length - 1) // temporal_ratio
        for frame_index, frames in frame_anchors:
            self._check_abort()
            if frame_index == 0 and prefix_latents is not None:
                encoded = prefix_latents
            else:
                with control_video_encoding(frames.shape[1] > 1):
                    encoded = _encode_video_source(self.vae, _resize_video(frames.float(), height, width), self.device).float().cpu()
            # Each 17-frame VAE chunk represents [1, 4, 4, 4, 4] pixel frames.
            latent_index = frame_index // clip_length * latents_per_clip + math.ceil((frame_index % clip_length) / temporal_ratio)
            latent_count = (frames.shape[1] - 1) // clip_length * latents_per_clip + 1
            if H3_CONTROL_LATENT_CONTINUATION:
                latent_count += math.ceil(((frames.shape[1] - 1) % clip_length) / temporal_ratio)
            for offset in range(min(latent_count, video_latent_frames(frame_count) - latent_index)):
                latents[latent_index + offset] = encoded[:, :, offset:offset + 1]
        indices = sorted(latents)
        return torch.tensor(indices, dtype=torch.long, device="cpu"), torch.cat([latents[index] for index in indices], dim=2)

    def _prepare_control_rows(self, control_frames, masks, frame_anchors, frame_count, height, width, condition_latents, inpaint_only, anchor_indices=None, anchor_latents=None):
        """Rows for the VideoX-Fun ControlNet-Union branch: [control latents | visibility map | masked-video latents].
        Explicit target-frame anchors are also visible in the inpaint condition."""
        patch = self.transformer.patch_size

        def fit(video):
            video = _resize_video(video[:, :frame_count].float(), height, width)
            return torch.cat((video, video[:, -1:].expand(-1, frame_count - video.shape[1], -1, -1)), dim=1) if video.shape[1] < frame_count else video

        def encode(video):
            self._check_abort()
            with control_video_encoding():
                return _encode_video_source(self.vae, video, self.device).float().cpu()

        frames = fit(guide_to_float(control_frames[:, :frame_count]))
        mask = None if masks is None else fit(guide_mask_to_float(masks[:1, :frame_count])).gt(0.5)
        control_rows = inpaint_rows = None
        if not inpaint_only:
            control_rows = patchify_video(encode(frames if mask is None else torch.where(mask, frames, -1.0)), patch)
        if mask is not None or frame_anchors:
            regenerate = torch.ones_like(frames[:1], dtype=torch.bool) if mask is None else mask
            # post_norm holes are zero after ImageNet normalization. Build the disposable masked input once;
            # anchors can replace its pixels directly without cloning the complete RGB source video first.
            neutral = self.vae.pixel_mean.view(3, 1, 1, 1).float().cpu().mul(2.0).sub(1.0)
            content = torch.where(regenerate, neutral, frames)
            if frame_anchors:
                for frame_index, known in frame_anchors:
                    known = _resize_video(known.float(), height, width)
                    regenerate[:, frame_index:frame_index + known.shape[1]] = False
                    content[:, frame_index:frame_index + known.shape[1]] = known
                known = None
            masked_latents = encode(content)
            content = None
            visibility = F.interpolate((~regenerate).float().unsqueeze(0), size=masked_latents.shape[2:], mode="trilinear", align_corners=False)
            if anchor_indices is not None:
                # Resampling 124 pixel frames to 37 latents loses a one-frame anchor. Reuse the exact target anchors;
                # ordinary spatial inpainting keeps the upstream interpolation and masked-video encoding above.
                visibility.index_fill_(2, anchor_indices, 1.0)
                masked_latents.index_copy_(2, anchor_indices, anchor_latents.to(masked_latents))
            inpaint_rows = torch.cat((patchify_video(visibility, patch), patchify_video(masked_latents, patch)), dim=1)
            regenerate = masked_latents = visibility = None
        frames = mask = None
        rows_count = (control_rows if control_rows is not None else inpaint_rows).shape[0]
        latent_columns = self.transformer.latents_dim * math.prod(patch)
        mask_columns = math.prod(patch)
        target_rows = torch.cat((control_rows if control_rows is not None else torch.zeros(rows_count, latent_columns, device="cpu"),
                                 inpaint_rows if inpaint_rows is not None else torch.zeros(rows_count, mask_columns + latent_columns, device="cpu")), dim=1)
        # Keyframe / continuation rows carry no control condition (the transformer does not inject skips into them either)
        condition_count = sum(latent.shape[2] * (latent.shape[-2] // patch[1]) * (latent.shape[-1] // patch[2]) for latent in condition_latents)
        condition_rows = torch.zeros(condition_count, target_rows.shape[1], device="cpu")
        return torch.cat((condition_rows, target_rows)).to(self.device)

    def _prepare_condition_rows(self, visual_latents, audio_latents, generator, keyframes=(), pika=False):
        video_rows = []
        for index, latent in enumerate(visual_latents):
            if pika:
                latent = latent.to(device=generator.device, dtype=torch.float32 if self.reference_mode else self.dtype)
            rows = patchify_video(latent.float(), self.transformer.patch_size)
            noise = (pika_noise(latent.shape, generator) if pika else
                     torch.randn(latent.shape, generator=generator, dtype=torch.float32, device="cpu"))
            noise = patchify_video(noise, self.transformer.patch_size)
            timestep = keyframes[index].get("timestep", VISUAL_COND_TIMESTEP) if index < len(keyframes) else VISUAL_COND_TIMESTEP
            video_rows.append(rows.mul(timestep).add_(noise.mul_(1.0 - timestep)) if pika else rows.mul(timestep).add_(noise, alpha=1.0 - timestep))
        return (torch.cat(video_rows) if video_rows else None,
                torch.cat([pack_audio(latent.float()) for latent in audio_latents]) if audio_latents else None)

    @_return_none_on_interrupt
    @generation_progress
    @veda_generation_context
    @torch.inference_mode()
    def generate(self, input_prompt, image_start=None, image_end=None, image_end_frame_position=None, input_frames=None, input_frames2=None, input_frames3=None, input_ref_images=None,
                 frames_to_inject=None, frames_relative_positions_list=None, image_refs_relative_size=100, input_video_strength=1.0,
                 input_masks=None, outpainting_dims=None, denoising_strength=1.0, masking_strength=1.0,
                 input_video=None, input_waveform=None, input_waveform_sample_rate=None, audio_conditioning_end=None,
                 video_guide=None, audio_guide=None, audio_guide2=None, audio_guide3=None, prefix_frames_count=0,
                 frame_num=124, height=768, width=1344, shift=12.0, sampling_steps=30, sampling_steps2=3, seed=0,
                 callback=None, VAE_tile_size=None, audio_prompt_type="", video_prompt_type="", fps=24,
                 sample_solver="euler", attention_sparsity=1.0, custom_sigmas=None,
                 guide_phases=1, switch_threshold=H3_PHASE_2_NOISE_LEVEL_START_DEFAULT, loras_slists=None, loras_selected=None, set_progress_status=None,
                 starting_sigma=None, preserve_input_mask_values=False, refinement_mode=False,
                 custom_settings=None, duration_seconds=None, verbose_level=0, dialogue_segment=False, image_mode=0,
                 context_scale=None, overlapped_latents=None, window_start_frame_no=0, **kwargs):
        if self.audio_only and H3_DIALOGUE_GENERATION and not dialogue_segment and is_dialogue_prompt(input_prompt):
            self._early_stop = False
            return generate_dialogue(
                self, input_prompt, audio_guide=audio_guide, audio_guide2=audio_guide2, audio_guide3=audio_guide3, input_waveform=input_waveform,
                input_waveform_sample_rate=input_waveform_sample_rate, audio_prompt_type=audio_prompt_type,
                duration_seconds=duration_seconds, sampling_steps=sampling_steps, seed=seed, shift=shift, callback=callback,
                VAE_tile_size=VAE_tile_size, fps=fps, sample_solver=sample_solver, attention_sparsity=attention_sparsity,
                loras_slists=loras_slists, loras_selected=loras_selected, custom_settings=custom_settings,
                set_progress_status=set_progress_status, verbose_level=verbose_level)
        self._use_shared_components()
        phase_1_sigmas = custom_sigmas[0] if custom_sigmas else None
        phase_2_sigmas = custom_sigmas[1] if custom_sigmas and len(custom_sigmas) > 1 else None
        if phase_1_sigmas is not None:
            sampling_steps = len(phase_1_sigmas)
        if phase_2_sigmas is not None:
            sampling_steps2 = len(phase_2_sigmas)
        grouped_masked_denoising = h3_grouped_masking_enabled(custom_settings) and not self.transformer.control_layers  # control rows follow the natural target order
        image_outputs = image_mode > 0 and not self.audio_only
        # Image saving uses 1 FPS, but H3's joint latents retain their native 24 FPS timing.
        fps = 24.0 if image_outputs else float(fps)
        if fps <= 0:
            raise ValueError("MiniMax H3 requires a positive output frame rate")
        self._set_interrupt_state()
        self._check_abort()
        self._configure_tiling(VAE_tile_size)
        if sample_solver not in ("euler", "er_sde", "res_multistep", "ralston_2s", PIKA_SOLVER):
            raise ValueError(f"Unsupported MiniMax H3 sampler {sample_solver!r}")
        pika = sample_solver == PIKA_SOLVER
        if pika:
            if int(sampling_steps) != PIKA_STEPS or float(shift) != PIKA_VIDEO_SHIFT or int(guide_phases) != 1:
                raise ValueError("Pika H3 Flex requires 6 steps, flow shift 12 and one generation phase")
            if phase_1_sigmas is not None or phase_2_sigmas is not None or starting_sigma is not None:
                raise ValueError("Pika H3 Flex uses its fixed schedule; custom sigmas and partial-schedule refinement are unsupported")
            if self.transformer.cache is not None:
                raise ValueError("Pika H3 Flex requires No Skipping")
        pdd = self.transformer.pdd_num_steps is not None
        if pdd:
            pdd_steps = int(self.transformer.pdd_num_steps) // int(self.transformer.pdd_block_size)
            if sample_solver != "euler":
                raise ValueError("MiniMax H3 PDD requires the Euler sampler")
            if int(sampling_steps) != pdd_steps:
                raise ValueError(f"MiniMax H3 PDD requires exactly {pdd_steps} inference steps")
            if starting_sigma is not None:
                raise ValueError("MiniMax H3 PDD does not support partial-schedule refinement")
        if sample_solver == "ralston_2s" and self.transformer.cache is not None and self.transformer.cache.cache_type == "spectrum":
            raise ValueError("MiniMax H3 Ralston 2S does not support Spectrum Feature Forecasting; use No Skipping or First Block Cache")
        if int(sampling_steps) < 1:
            raise ValueError("MiniMax H3 requires at least one inference step")
        if self.audio_only:
            frame_num = round(float(duration_seconds) * fps)
            height = width = 32
            guide_phases = 1
        frame_num = 1 if image_outputs else normalize_frame_count(int(frame_num), 5, 17, 5)
        if image_outputs:
            prefix_frames_count = 0
        audio_from_control_video = not image_outputs and not self.reference_mode and "2" in (audio_prompt_type or "")
        # The disabled experiment aligns target overlap with the previous VAE grid (5 + 17n).
        # Retain support for legacy 1 + 17n prefixes when encoding an external continuation.
        overlap_offset = 5 if H3_CONTROL_LATENT_CONTINUATION and self.transformer.control_layers and int(prefix_frames_count or 0) % 17 == 5 else 1
        prefix_frames_count, overlap_error = normalize_overlap(int(prefix_frames_count or 0), 17, overlap_offset)
        if overlap_error:
            raise ValueError(overlap_error)
        continuation = _as_video(input_video) if input_video is not None and prefix_frames_count > 0 else None
        continuation_count = min(prefix_frames_count, continuation.shape[1]) if continuation is not None and image_start is None else 0
        if continuation_count and continuation_count < prefix_frames_count:
            continuation_count = floor_frame_count(continuation_count, 1, 17, overlap_offset if continuation_count >= overlap_offset else 1)
        # ControlNet continuation keeps the overlap on the target timeline, both as fixed video latents
        # and as visible pixels in its inpainting input. Native H3 history alone loses that appearance constraint.
        control_target_anchors = bool(self.transformer.control_layers) and not audio_from_control_video and not refinement_mode
        control_known_frames = None
        if control_target_anchors and continuation_count:
            control_known_frames, continuation_count = continuation[:, -continuation_count:], 0
        frozen_control_video = (_build_frozen_control_video(input_frames, continuation, frame_num, continuation_count)
                                if audio_from_control_video else None)
        if frozen_control_video is not None:
            frame_num = frozen_control_video.shape[1]
        history_frames = continuation[:, -continuation_count:-1] if continuation_count > 1 else None
        history_count = 0 if history_frames is None else history_frames.shape[1]
        audio_start_seconds = history_count / fps
        if audio_conditioning_end is not None and audio_conditioning_end < audio_start_seconds:
            if not H3_EXTENDED_AUDIO_REPAIR:
                raise ValueError("H3 native continuation regenerates only the last overlap frame's audio; the source soundtrack ends too early to repair this join.")
            # Regenerate the missing history tail on the audio grid while retaining every copied video frame.
            audio_start_seconds = max(0, math.floor(audio_conditioning_end * AUDIO_LATENT_FPS) / AUDIO_LATENT_FPS)
        audio_time_offset = (audio_start_seconds - history_count / fps) * AUDIO_LATENT_FPS
        target_frames = frame_num - history_count
        aligned_target_frames = 1 if image_outputs else normalize_frame_count(target_frames, 5, 17, 5)
        if target_frames <= 0:
            raise ValueError("Sliding-window overlap leaves no frames for H3 to generate")
        refinement_mode = bool(refinement_mode)
        fun_control = bool(self.transformer.control_layers) and input_frames is not None
        raw_control_outpainting = fun_control and outpainting_dims is not None and not any(letter in (video_prompt_type or "") for letter in "PDESCM")
        control_video = refinement_mode or "G" in (video_prompt_type or "")
        # target frame 0 of a control video is the last overlap frame of the previous window (or the start image): kept, not taken from the control video
        control_start_frame = ((continuation[:, -1:] if continuation_count else _as_video(image_start)[:, :1] if image_start is not None else None)
                               if control_video and not audio_from_control_video else None)
        if control_start_frame is not None and input_masks is not None:
            input_masks = input_masks.clone()
            input_masks[:, history_count] = 0
        outpainting_mask = (_build_outpainting_mask(input_frames, outpainting_dims)
                            if (control_video or fun_control) and not audio_from_control_video and input_frames is not None else None)
        if raw_control_outpainting and input_masks is None:
            # Raw preprocessing keeps the source picture; only the expanded borders are editable.
            input_masks = torch.zeros_like(input_frames[:1]) if outpainting_mask is None else outpainting_mask
        elif outpainting_mask is not None:
            input_masks = outpainting_mask if input_masks is None else torch.maximum(guide_mask_to_float(input_masks).float(), outpainting_mask.to(device=input_masks.device))
        video_to_video = control_video and not audio_from_control_video and (float(denoising_strength) < 1.0 or input_masks is not None)
        control_source = None
        if video_to_video:
            control_source = _as_video(input_frames)[:, history_count:history_count + aligned_target_frames]
            if control_start_frame is not None:
                control_frames = guide_to_float(control_source[:, 1:])
                control_source = torch.cat((_resize_video(control_start_frame, *control_source.shape[-2:]).to(control_frames), control_frames), dim=1)
                control_frames = None
        control_start_frame = None
        if grouped_masked_denoising and video_to_video and input_masks is not None and not preserve_input_mask_values and offload.shared_state.get("_attention") == "sol":
            raise ValueError("MiniMax H3 Grouped Rows mask denoising is not compatible with Sol Attention; select Shared Timestep or another attention mode")

        waveform = self._waveform(input_waveform, input_waveform_sample_rate)
        history_waveform = None
        target_audio_condition = None
        target_video_condition = None
        frozen_target_video = None if frozen_control_video is None else frozen_control_video[:, history_count:]
        target_height, target_width = int(height), int(width)
        two_phase = int(guide_phases) > 1 and frozen_target_video is None
        tiled_phase_2 = two_phase and H3_PHASE_2_TILING_FLAG in (video_prompt_type or "")
        if two_phase:
            if self.latent_upscaler is None:
                raise RuntimeError("MiniMax H3 two-phase generation requires the latent upscaler")
            height = max(32, round(target_height / H3_TWO_PHASE_SCALE / 32) * 32)
            width = max(32, round(target_width / H3_TWO_PHASE_SCALE / 32) * 32)

        def activate_lora_phase(phase, steps):
            if loras_slists is None:
                return
            phase_switch_step = steps if phase == 1 else 0
            update_loras_slists(self.transformer, loras_slists, steps, phase_switch_step=phase_switch_step, phase_switch_step2=steps)
            offload.set_step_no_for_lora(self.transformer, 0)

        def apply_phase_2_lora_policy():
            if loras_slists is None:
                return
            if pdd:
                for index, lora in enumerate(loras_selected or ()):
                    name = os.path.basename(str(lora).split("|", 1)[0])
                    if "turbo" not in name.lower():
                        continue
                    print(f"MiniMax H3 PDD phase 2: disabled Turbo LoRA '{name}' so it does not stack with the PDD accelerator")
                    loras_slists["phase2"][index] = 0.0
                    loras_slists["shared"][index] = False
                return
            if self._phase_2_loras is None:
                return
            default_index, accelerator_count = self._phase_2_loras
            phase_2_multipliers = loras_slists["phase2"][default_index + 1:default_index + 1 + accelerator_count]
            override = any(value != 0 for multiplier in phase_2_multipliers for value in (multiplier if isinstance(multiplier, list) else [multiplier]))
            loras_slists["phase2"][default_index] = 0.0 if override else 1.0
            loras_slists["shared"][default_index] = False

        activate_lora_phase(1, int(sampling_steps))
        if self.transformer.control_layers and two_phase:
            raise ValueError("MiniMax H3 ControlNet-Union supports one-phase generation only")
        video_references = "V" in (video_prompt_type or "") and "G" not in (video_prompt_type or "") and not fun_control  # Ref2VA reference videos, or an in-context control video (VU) on any H3 model
        aligned_control = (_as_video(input_frames)[:, history_count:history_count + aligned_target_frames]
                           if H3_ALIGNED_CONTROL_VIDEO and video_references and "-" not in (video_prompt_type or "") and self.fixed_prompt is None else None)
        # Injected frames on the window's first or end frame act as its start / end image (same prompt order as the enhancer's labels);
        # frames inside the overlap or on a frame already owned by the start / continuation frame are dropped
        injected_frames = [(image, int(position) - history_count) for image, position in zip(frames_to_inject or (), frames_relative_positions_list or ())]
        # Below full strength, end images and injected frames become noised conditions that the video eases into instead of jumping to them
        soft_anchor_timestep = float(input_video_strength) if float(input_video_strength) < VISUAL_COND_TIMESTEP else None
        has_start_frame = control_known_frames is not None or continuation_count > 0 or image_start is not None and not audio_from_control_video
        injected_start = None if has_start_frame else next((image for image, index in injected_frames if index == 0), None)
        end_indices = {target_frames - 1} | ({int(image_end_frame_position) - history_count} if image_end_frame_position is not None else set())
        injected_end = None if image_end is not None and not audio_from_control_video else next(((image, index) for image, index in injected_frames if index in end_indices), None)
        injected_frames = [(image, index) for image, index in injected_frames
                           if 0 < index < target_frames and (control_known_frames is None or index >= control_known_frames.shape[1])
                           and (injected_end is None or index != injected_end[1])]
        control_frame_anchors = [] if control_known_frames is None else [(0, control_known_frames)]

        def prepare_keyframes(stage_height, stage_width, stage_presentation, tile_origin=None):
            stage_latents, stage_keyframes = [], []

            def add_image_condition(video, frame_index, anchor=None, timestep=None):
                if control_target_anchors:
                    video = _as_video(video)[:, :1]
                    control_frame_anchors.append((frame_index, video))
                    stage_presentation.append({"type": "image", "frames": _qwen_frames(video.clone())})
                else:
                    self._add_image_condition(video, frame_index, stage_presentation, stage_latents, stage_keyframes, anchor=anchor, timestep=timestep)

            def prepare_stage_video(source):
                if tile_origin is None:
                    return _resize_video(source, stage_height, stage_width)
                source = _resize_video(source, target_height, target_width)
                return _crop_spatial_tile(source, tile_origin[0], tile_origin[1], stage_height, stage_width)

            def prepare_injected_frame(image):
                image = _to_pil(image)
                if tile_origin is not None:
                    return prepare_stage_video(_pil_to_video(image))
                return _pil_to_video(image if image.size == (stage_width, stage_height) else image.resize((stage_width, stage_height), Image.Resampling.LANCZOS))

            if continuation_count:
                if history_frames is not None:
                    self._add_video_history(prepare_stage_video(history_frames), stage_latents, stage_keyframes)
                add_image_condition(prepare_stage_video(continuation[:, -1:]), 0)
            elif image_start is not None and not audio_from_control_video:
                add_image_condition(prepare_stage_video(image_start), 0)
            elif injected_start is not None:
                add_image_condition(prepare_injected_frame(injected_start), 0)
            if image_end is not None and not audio_from_control_video:
                if image_end_frame_position is None:
                    add_image_condition(prepare_stage_video(image_end), aligned_target_frames - 1, timestep=soft_anchor_timestep)
                else:
                    add_image_condition(prepare_stage_video(image_end), int(image_end_frame_position) - history_count, anchor="frame", timestep=soft_anchor_timestep)
            elif injected_end is not None:
                add_image_condition(prepare_injected_frame(injected_end[0]), injected_end[1], anchor="frame", timestep=soft_anchor_timestep)
            for image, frame_index in injected_frames:
                add_image_condition(prepare_injected_frame(image), frame_index, anchor="frame", timestep=soft_anchor_timestep)
            if aligned_control is not None:
                with control_video_encoding():
                    latent = self._encode_video(prepare_stage_video(aligned_control))
                stage_latents.append(latent)
                stage_keyframes.append({"anchor": "first", "latent_frame_count": latent.shape[2]})
            return stage_latents, stage_keyframes

        presentation, audio_latents, refs, audio_keyframes = [], [], [], []
        visual_latents, keyframes = prepare_keyframes(height, width, presentation)
        if not self.reference_mode and (input_ref_images or input_frames is not None and not (control_video or audio_from_control_video or video_references or fun_control) or input_frames2 is not None):
            raise ValueError("Image, video, and audio references require the Ref2VA checkpoint")
        if continuation_count:
            if waveform is not None:
                overlap_samples = round(continuation_count / fps * AUDIO_SAMPLE_RATE)
                continuation_waveform = _fit_audio_samples(waveform[0], overlap_samples).unsqueeze(0)
                history_samples = round(audio_start_seconds * AUDIO_SAMPLE_RATE)
                history_waveform = continuation_waveform[..., :history_samples] if history_samples else None
                continuation_audio = self._encode_audio(continuation_waveform)
                boundary_latents = (continuation_audio.shape[-1] if not history_count else
                                    min(continuation_audio.shape[-1], max(1, round(AUDIO_LATENT_FPS / fps))))
                history_latents = continuation_audio.shape[-1] - boundary_latents
                if audio_time_offset < 0:
                    history_latents = round(audio_start_seconds * AUDIO_LATENT_FPS)
                if audio_conditioning_end is not None:
                    condition_tokens = max(0, math.floor(audio_conditioning_end * AUDIO_LATENT_FPS + 1e-9))
                    boundary_tokens = max(0, math.floor((audio_conditioning_end - audio_start_seconds) * AUDIO_LATENT_FPS + 1e-9))
                    condition_tokens = min(condition_tokens, history_latents + boundary_tokens)
                    continuation_audio = continuation_audio[..., :condition_tokens]
                if history_latents:
                    self._add_audio_condition(continuation_audio[..., :history_latents], "history", audio_latents, audio_keyframes)
                if continuation_audio.shape[-1] > history_latents:
                    self._add_audio_condition(continuation_audio[..., history_latents:], "first", audio_latents, audio_keyframes)
        if frozen_target_video is not None:
            with control_video_encoding(audio_from_control_video):
                target_video_condition = self._encode_video(_resize_video(frozen_target_video, height, width), keep_all_latents=True)
        if self.reference_mode and self.fixed_prompt is None:
            if image_outputs and video_references:
                input_ref_images = [*(input_ref_images or []), guide_to_float(input_frames)]
                video_references = False
            for image in input_ref_images or []:
                self._add_image_reference(image, width, height, image_refs_relative_size, presentation, visual_latents, refs)

        video_sources = []
        if video_references and aligned_control is None:
            video_sources.append(input_frames)
            if "+" in (video_prompt_type or ""):
                video_sources.append(input_frames2)
            if "*" in (video_prompt_type or ""):
                video_sources.append(input_frames3)
        video_sources = [_as_video(source) for source in video_sources]
        if self.reference_mode and self.fixed_prompt is None and "1" in (video_prompt_type or ""):  # excerpts replace the whole reference video
            video_sources = self._load_video_excerpts(video_guide, custom_settings[H3_VIDEO_EXCERPTS_SETTING], fps, height, width)
        if self.fixed_prompt is not None:
            if any(source is None for source in video_sources):
                print("Viggle: no control video frames available for this window; continuing without control-video motion guidance.")
                video_sources = [source for source in video_sources if source is not None]
            video_sources = [source[:, history_count:] for source in video_sources]
        reference_videos = "-" in (video_prompt_type or "")  # without "-" the videos are per-window control slices, sized by the window rather than the 15s reference budget
        if reference_videos and len(video_sources) > 1 and sum(video.shape[1] for video in video_sources) > reference_video_frame_limit(1, fps):  # share the budget evenly, like audio references
            max_frames = reference_video_frame_limit(len(video_sources), fps)
            video_sources = [video[:, :max_frames] for video in video_sources]
        if reference_videos and sum(video.shape[1] for video in video_sources) > reference_video_frame_limit(1, fps):
            raise ValueError(f"MiniMax H3 reference videos must total at most {reference_video_frame_limit(1, fps)} frames (found {sum(video.shape[1] for video in video_sources)})")
        soundtrack = "S" in (audio_prompt_type or "")  # the audio is the soundtrack: condition on it rather than reference it
        soundtrack_excerpts = "1" in (audio_prompt_type or "")  # K1: soundtrack excerpts become independent audio references
        soundtrack_sources = (audio_guide, audio_guide2, audio_guide3) if self.fixed_prompt is None and "K" in (audio_prompt_type or "") and not soundtrack and not soundtrack_excerpts else (None, None, None)
        soundtracks = [self._load_audio_reference(soundtrack_sources[index]) if soundtrack_sources[index] is not None else None for index in range(len(video_sources))]
        soundtracks = [None if track is None else track[..., :round(video.shape[1] / fps * AUDIO_SAMPLE_RATE)] for track, video in zip(soundtracks, video_sources)]  # each soundtrack matches its video
        for index, source in enumerate(video_sources):
            self._add_video_reference(_resize_video(source, height, width), soundtracks[index], fps, presentation, visual_latents, audio_latents, refs)
        if self.fixed_prompt is not None:
            self._add_image_reference(input_ref_images[0], width, height, 100, presentation, visual_latents, refs)
        reference_sources = []
        if self.reference_mode and self.fixed_prompt is None and not refinement_mode and "A" in (audio_prompt_type or "") and not soundtrack:
            reference_sources.append(audio_guide if audio_guide is not None else waveform)
        if self.reference_mode and self.fixed_prompt is None and not refinement_mode and "B" in (audio_prompt_type or ""):
            reference_sources.append(audio_guide2)
        if self.reference_mode and self.fixed_prompt is None and not refinement_mode and "D" in (audio_prompt_type or ""):
            reference_sources.append(audio_guide3)
        if self.reference_mode and self.fixed_prompt is None and not refinement_mode and "K" in (audio_prompt_type or "") and soundtrack_excerpts:
            import soundfile as sf
            from shared.utils.utils import get_video_info

            source_fps, _, _, source_frames = get_video_info(video_guide)
            sample_rate = sf.info(audio_guide).samplerate
            for start, duration in parse_excerpts(custom_settings[H3_AUDIO_EXCERPTS_SETTING], fps, source_frames / source_fps, "Audio Reference Positions"):
                audio, _ = sf.read(audio_guide, start=round(start * sample_rate), frames=round(duration * sample_rate), dtype="float32", always_2d=True)  # read only the excerpt
                reference_sources.append(self._waveform(audio, sample_rate))
        for reference_audio in self._prepare_audio_references(reference_sources):
            self._add_audio_reference(reference_audio, presentation, audio_latents, refs)
        if (refinement_mode or soundtrack or not self.reference_mode or self.fixed_prompt is not None) and any(flag in (audio_prompt_type or "") for flag in "AK") and waveform is not None:
            condition_start = round(audio_start_seconds * AUDIO_SAMPLE_RATE)
            condition_samples = round((target_frames / fps - audio_time_offset / AUDIO_LATENT_FPS) * AUDIO_SAMPLE_RATE)
            condition_waveform = waveform[..., condition_start:condition_start + condition_samples]
            if condition_waveform.shape[-1]:
                target_audio_condition = self._encode_audio(condition_waveform)
        elif control_known_frames is not None and waveform is not None:
            target_audio_condition = self._encode_audio(waveform[..., :round(control_known_frames.shape[1] / fps * AUDIO_SAMPLE_RATE)])
        if target_audio_condition is not None and audio_conditioning_end is not None:
            condition_tokens = max(0, math.floor((audio_conditioning_end - audio_start_seconds) * AUDIO_LATENT_FPS + 1e-9))
            target_audio_condition = target_audio_condition[..., :condition_tokens]
        if self.reference_mode:
            visual_ref_count = sum(ref["kind"] in ("image", "video", "video_audio") for ref in refs)
            audio_ref_count = sum(ref["kind"] in ("audio", "video_audio") for ref in refs)
            if pika and not visual_ref_count:
                raise ValueError("Pika H3 Flex Ref2VA requires at least one image or video reference")
            if not pika and not self.audio_only and audio_ref_count > visual_ref_count:
                raise ValueError(f"MiniMax H3 requires at least as many image and video references as audio references (found {visual_ref_count} visual and {audio_ref_count} audio)")
            standalone_audio_count = sum(ref["kind"] == "audio" for ref in refs)
            if len(refs) > 12 or sum(ref["kind"] == "image" for ref in refs) > 9 or sum(ref["kind"] in ("video", "video_audio") for ref in refs) > 3 or (standalone_audio_count if pika else audio_ref_count) > 3:
                raise ValueError("MiniMax H3 supports at most 12 references: 9 images, 3 videos, and 3 audio clips")

        source_latents = editable_mask = None
        if video_to_video:
            if set_progress_status is not None:
                set_progress_status("Encoding H3 Control Video")
            self._check_abort()
            source_video = _resize_video(control_source, height, width)
            with control_video_encoding(control_video):
                source_latents = _encode_video_source(self.vae, source_video, self.device, outpainting_dims).cpu()
            self._check_abort()
            if input_masks is not None:
                source_mask = input_masks[:, history_count:history_count + source_video.shape[1]]
                editable_mask = _resize_video_mask(source_mask, source_latents.shape[-3:], self.vae.config.clip_length, self.vae.temporal_compression_ratio, binarize=not preserve_input_mask_values)
                if not preserve_input_mask_values:
                    editable_mask = _snap_video_mask_to_patch_cells(editable_mask, self.transformer.patch_size)
            source_video = source_mask = None
        control_anchor_indices = control_anchor_latents = control_anchor_noised = None
        control_anchor_frames = ()
        if control_frame_anchors:
            if set_progress_status is not None:
                set_progress_status("Encoding H3 Known Frames")
            prefix_latents = None
            if H3_CONTROL_LATENT_CONTINUATION and control_known_frames is not None and overlapped_latents is not None:
                offset = window_start_frame_no - overlapped_latents["start_frame"]
                if offset < 0 or offset % self.vae.config.clip_length:
                    raise ValueError("H3 ControlNet latent continuation requires overlaps of 5 + 17n frames")
                latent_offset = offset // self.vae.config.clip_length * (1 + (self.vae.config.clip_length - 1) // self.vae.temporal_compression_ratio)
                prefix_latents = overlapped_latents["latents"][:, :, latent_offset:]
            control_anchor_indices, control_anchor_latents = self._prepare_control_anchors(control_frame_anchors, aligned_target_frames, height, width, prefix_latents)
            control_anchor_frames = tuple(control_anchor_indices.tolist())
            prefix_latents = None
        overlapped_latents = None

        control_rows = None
        if fun_control:
            if set_progress_status is not None:
                set_progress_status("Encoding H3 Control Video")
            window_masks = None if input_masks is None else input_masks[:, history_count:history_count + aligned_target_frames]
            control_rows = self._prepare_control_rows(input_frames[:, history_count:history_count + aligned_target_frames], window_masks, control_frame_anchors,
                                                      aligned_target_frames, height, width, visual_latents, "M" in (video_prompt_type or "") or raw_control_outpainting,
                                                      anchor_indices=control_anchor_indices, anchor_latents=control_anchor_latents)
            window_masks = None

        if set_progress_status is not None:
            set_progress_status("Encoding H3 prompt and references")
        context, text_tags = self._encode_prompt(input_prompt, presentation)
        presentation = None  # the frames shown to the text encoder are no longer needed (phase 2 builds its own)
        self._check_abort()
        self._use_transformer()
        # PTD changes the token-refiner adapters at every denoising step. Keep the
        # encoder output here; forward refines it after the shared LoRA step is set.
        if not self.ptd:
            context = self.transformer.preprocess_text_embeds(context)

        latent_t = video_latent_frames(aligned_target_frames)
        latent_h, latent_w = math.ceil(height / 16), math.ceil(width / 16)
        audio_tail_latents = 0 if pika or self.audio_only or image_outputs else H3_AUDIO_TAIL_LATENTS
        audio_t = round(aligned_target_frames / fps * AUDIO_LATENT_FPS - audio_time_offset) + audio_tail_latents
        generator = torch.Generator(device=self.device if pika else "cpu").manual_seed(int(seed))
        condition_generator = torch.Generator(device=self.device).manual_seed(int(seed) + 7919) if pika else generator
        cond_video_rows, cond_audio_rows = self._prepare_condition_rows(visual_latents, audio_latents, condition_generator, keyframes, pika=pika)
        target_video_condition_frames = 0 if target_video_condition is None else target_video_condition.shape[2]
        video = ((pika_noise((1, 24, latent_t, latent_h, latent_w), generator) if pika else
                  torch.randn((1, 24, latent_t, latent_h, latent_w), generator=generator, dtype=torch.float32, device="cpu"))
                 if target_video_condition is None else target_video_condition).to(self.device)
        target_video_condition = None
        if control_anchor_indices is not None:
            control_anchor_indices = control_anchor_indices.to(video.device)
            control_anchor_latents = control_anchor_latents.to(video)
            control_anchor_noised = torch.lerp(control_anchor_latents, video.index_select(2, control_anchor_indices), 1.0 - VISUAL_COND_TIMESTEP)
        if source_latents is not None:
            source_latents = source_latents[:, :, :latent_t].to(device=video.device, dtype=video.dtype)
            source_noise = video[:, :, :source_latents.shape[2]].clone()
            source_buffer = torch.empty_like(source_latents)
            editable_mask = editable_mask[:, :, :source_latents.shape[2]].to(video) if editable_mask is not None else None
        else:
            source_noise = source_buffer = None
        audio = (pika_noise((1, 32, 2, audio_t), generator, audio=True) if pika else
                 unpack_audio(torch.randn((audio_t * 2, 32), generator=generator, dtype=torch.float32, device="cpu")).to(self.device))
        target_audio_condition_latents = 0 if target_audio_condition is None else min(audio_t, target_audio_condition.shape[-1])
        if target_audio_condition_latents:
            audio[..., :target_audio_condition_latents].copy_(target_audio_condition[..., :target_audio_condition_latents].to(audio))
        target_audio_condition = None
        payload = {"keyframes": keyframes or None, "audio_keyframes": audio_keyframes or None,
                   "refs": refs or None, "cond_video_rows": cond_video_rows,
                   "cond_audio_rows": cond_audio_rows, "frame_count": aligned_target_frames, "text_token_tags": text_tags,
                   "fps": fps, "audio_time_offset": audio_time_offset, "target_audio_condition_latents": target_audio_condition_latents,
                   "target_video_condition_frames": target_video_condition_frames,
                   "attention_sparsity": float(attention_sparsity)}
        if control_rows is not None:
            payload.update(control_rows=control_rows, control_scale=float(context_scale[0]), control_anchor_frames=control_anchor_frames)
        control_rows = None

        fasth3 = getattr(self.transformer, "fasth3_inference", None)
        grid_max_t = 1.0
        if fasth3 is not None:  # a FastH3 student runs its trained contract: its video shift, a PDD fine grid on [0, grid_max_t]
            if starting_sigma is not None:
                raise ValueError("MiniMax H3 FastH3 does not support partial-schedule refinement")
            if not pdd or int(sampling_steps) != len(fasth3["pdd_step_indices"]) - 1 or sample_solver != "euler":
                raise ValueError(f"MiniMax H3 FastH3 requires exactly {len(fasth3['pdd_step_indices']) - 1} PDD inference steps with the Euler sampler")
            shift, grid_max_t = float(fasth3["video_scheduler_shift"]), float(fasth3.get("grid_max_t", 1.0))
        if starting_sigma is None:
            base_sigmas = torch.linspace(1.0, 0.0, int(sampling_steps) + 1, dtype=torch.float32, device=self.device)
        else:
            starting_sigma = float(starting_sigma)
            if not 0.0 < starting_sigma <= 1.0:
                raise ValueError("MiniMax H3 starting_sigma must be in (0, 1]")
            base_start = starting_sigma / (float(shift) - starting_sigma * (float(shift) - 1.0))
            base_sigmas = torch.linspace(base_start, 0.0, int(sampling_steps) + 1, dtype=torch.float32, device=self.device)
        sigmas_video = torch.unique_consecutive(float(shift) * base_sigmas / (1.0 + (float(shift) - 1.0) * base_sigmas))
        sigmas_audio = torch.unique_consecutive(3.0 * base_sigmas / (1.0 + 2.0 * base_sigmas))
        if pika:
            sigmas_video, sigmas_audio = pika_sigma_schedule(self.device)
        if phase_1_sigmas is not None:
            sigmas_video = torch.tensor([*phase_1_sigmas, 0.0], dtype=torch.float32, device=self.device)
            sigmas_audio = _map_shifted_sigma(sigmas_video, float(shift), 3.0)
            video[:, :, target_video_condition_frames:].mul_(sigmas_video[0])
            audio[..., target_audio_condition_latents:].mul_(sigmas_audio[0])
        if sigmas_video.shape != sigmas_audio.shape:
            raise ValueError("The selected H3 flow shift collapses a different number of video and audio schedule points")
        if grid_max_t != 1.0:  # the whole grid scaled to grid_max_t: the first noise level is grid_max_t, so is the initial noise
            sigmas_video, sigmas_audio = sigmas_video * grid_max_t, sigmas_audio * grid_max_t
            video.mul_(grid_max_t)
            audio.mul_(grid_max_t)
        refinement_sigmas_video = None
        if source_latents is not None and starting_sigma is not None and preserve_input_mask_values:
            if editable_mask is None or editable_mask.shape[-2:] != (1, 1):
                raise ValueError("MiniMax H3 refinement requires a spatially uniform strength mask")
            latent_strengths = editable_mask[0, 0, :source_latents.shape[2], 0, 0]
            refinement_strength = latent_strengths[0]
            if not bool((latent_strengths == refinement_strength).all()):
                raise ValueError("MiniMax H3 refinement requires one uniform strength for the whole video")
            refinement_sigmas_video = _uniform_latent_frame_sigma_schedule(sigmas_video, refinement_strength, source_latents.shape[2])
            print(f"MiniMax H3 refinement: uniform strength {float(refinement_strength):.2f}, initial sigma {float(refinement_sigmas_video[0, 0, 0, 0, 0, 0]):.3f}")
        if source_latents is not None and starting_sigma is not None:
            source_video = video[:, :, :source_latents.shape[2]]
            torch.lerp(source_latents, source_noise, refinement_sigmas_video[0] if refinement_sigmas_video is not None else sigmas_video[0], out=source_buffer)
            if refinement_sigmas_video is not None or editable_mask is None or float(masking_strength) <= 0.0:
                source_video.copy_(source_buffer)
            elif preserve_input_mask_values:
                preserved = torch.lerp(source_latents, source_noise, 1.0 - VISUAL_COND_TIMESTEP)
                source_video.copy_(preserved).lerp_(source_buffer, editable_mask)
                preserved = None
            else:
                source_video.copy_(source_latents).lerp_(source_buffer, editable_mask)
        tau_start = float(attention_sparsity)
        cache = self.transformer.cache
        audio_scale = float(shift) / 3.0

        def run_denoising(stage_sigmas_video, stage_sigmas_audio, description, denoising_extra="", pass_no=-1, freeze_audio=False, stage_solver=None, use_cache=True, effective_sigmas_video=None):
            nonlocal video, audio
            stage_solver = stage_solver or sample_solver
            audio_on_video_schedule = stage_solver in ("res_multistep", "ralston_2s")
            model_steps = stage_sigmas_video.numel() - 1
            if pdd:
                stage_pdd_video_plans = pdd_sampling_plans_for_sigmas(stage_sigmas_video, shift, self.transformer.pdd_num_steps, grid_max_t)
                stage_pdd_audio_plans = (pdd_sampling_plans(3.0, self.transformer.pdd_num_steps, self.transformer.pdd_block_size, grid_max_t)[-1:].expand(model_steps, -1)
                                         if freeze_audio else pdd_sampling_plans_for_sigmas(stage_sigmas_audio, 3.0, self.transformer.pdd_num_steps, grid_max_t))
            else:
                stage_pdd_video_plans = stage_pdd_audio_plans = None
            tau_denominator = max(1, model_steps - 1)
            denoising_start_step = 0 if starting_sigma is not None else int(round(model_steps * (1.0 - float(denoising_strength)), 4))
            mask_end_step = min(model_steps, denoising_start_step + math.ceil(model_steps * float(masking_strength))) if editable_mask is not None else 0
            effective_res_sigmas = effective_sigmas_video[:, 0, 0, 0, 0, 0] if effective_sigmas_video is not None else stage_sigmas_video
            res_coefficients = _res_multistep_coefficients(effective_res_sigmas) if stage_solver == "res_multistep" else None
            audio_res_coefficients = _res_multistep_coefficients(stage_sigmas_video) if effective_sigmas_video is not None and stage_solver == "res_multistep" else res_coefficients
            if stage_solver == "ralston_2s":
                ralston_sigmas_video = torch.lerp(stage_sigmas_video[:-1], stage_sigmas_video[1:], 2.0 / 3.0)
                ralston_effective_sigmas_video = torch.lerp(effective_sigmas_video[:-1], effective_sigmas_video[1:], 2.0 / 3.0) if effective_sigmas_video is not None else None
                ralston_sigmas_audio = torch.zeros_like(ralston_sigmas_video) if freeze_audio else _map_shifted_sigma(ralston_sigmas_video, float(shift), 3.0)
            else:
                ralston_sigmas_video = ralston_effective_sigmas_video = ralston_sigmas_audio = None
            active_cache = cache if use_cache else None
            spectrum = MiniMaxH3Spectrum(active_cache, stage_sigmas_video[:-1], stage_solver) if active_cache is not None and active_cache.cache_type == "spectrum" else None
            first_block_cache = MiniMaxH3FirstBlockCache(active_cache) if active_cache is not None and active_cache.cache_type == "first_block" else None
            offline_spectrum = spectrum is not None and spectrum.full_anchor_cache
            grouped_masking = _set_grouped_video_rows(payload, editable_mask if grouped_masked_denoising and not preserve_input_mask_values and mask_end_step > denoising_start_step else None,
                                                       video.shape[-3:], video.device)

            def condition_control_video(value, sigma):
                if control_anchor_indices is None or target_video_condition_frames:
                    return sigma.flatten()
                value.index_copy_(2, control_anchor_indices, control_anchor_noised)
                frame_sigmas = sigma.flatten().expand(1, value.shape[2]).clone()
                frame_sigmas[:, control_anchor_indices] = 1.0 - VISUAL_COND_TIMESTEP
                return frame_sigmas

            def denoise_pass(pass_description, pass_extra):
                nonlocal video, audio
                old_video_denoised = old_audio_denoised = None
                old_video_denoised_d = old_audio_denoised_d = None
                er_generator = torch.Generator(device=video.device).manual_seed(int(seed)) if stage_solver == "er_sde" else None
                renoise_generator = torch.Generator(device=video.device).manual_seed(int(seed) + 1) if stage_solver == PIKA_SOLVER else None
                for step in tqdm(range(model_steps), desc=pass_description):
                    self._set_interrupt_state()
                    self._check_abort()
                    payload["attention_sparsity"] = tau_start + (SOL_ATTN_TAU_END - tau_start) * step / tau_denominator
                    offload.set_step_no_for_lora(self.transformer, step)
                    if pdd:
                        self.transformer.final_layer.video_out.set_plan(stage_pdd_video_plans[step:step + 1])
                        self.transformer.final_layer.audio_out.set_plan(stage_pdd_audio_plans[step:step + 1])
                    if spectrum is not None:
                        spectrum.begin_step(step)
                    if first_block_cache is not None:
                        first_block_cache.begin_step(step)
                    audio_tail = audio[..., target_audio_condition_latents:]
                    if audio_on_video_schedule and audio_tail.shape[-1]:
                        audio_tail.mul_(stage_sigmas_audio[step] / stage_sigmas_video[step])
                    sigma_video = effective_sigmas_video[step] if effective_sigmas_video is not None else stage_sigmas_video[step]
                    sigma_video_next = effective_sigmas_video[step + 1] if effective_sigmas_video is not None else stage_sigmas_video[step + 1]
                    previous_sigma_video = (effective_sigmas_video[step - 1] if effective_sigmas_video is not None else stage_sigmas_video[step - 1]) if step else None
                    previous_previous_sigma_video = (effective_sigmas_video[step - 2] if effective_sigmas_video is not None else stage_sigmas_video[step - 2]) if step > 1 else None
                    step_editable_mask = _masking_step_mask(editable_mask, step, denoising_start_step, mask_end_step)
                    payload["target_video_mask_active"] = grouped_masking and step_editable_mask is not None
                    if payload["target_video_mask_active"] and effective_sigmas_video is None:
                        _reinject_video_source(video[:, :, :source_latents.shape[2]], source_latents, source_noise, step_editable_mask,
                                               sigma_video, source_buffer, 1.0 - VISUAL_COND_TIMESTEP)
                    model_sigma_video = condition_control_video(video, sigma_video)
                    video_velocity, audio_velocity = self.transformer(video, audio, model_sigma_video, stage_sigmas_audio[step:step + 1], context, payload, spectrum=spectrum, first_block_cache=first_block_cache)
                    if spectrum is not None:
                        spectrum.finish_step()
                    if stage_solver == "er_sde":
                        video_denoised = video_velocity.float().mul_(sigma_video).add_(video)
                        if effective_sigmas_video is None and source_latents is not None and step_editable_mask is not None:
                            _blend_video_source(video_denoised, source_latents, step_editable_mask)
                        audio_denoised = None
                        if audio_tail.shape[-1]:
                            audio_denoised = audio_velocity[..., target_audio_condition_latents:].float().mul_(stage_sigmas_audio[step]).add_(audio_tail)
                        if not bool(sigma_video_next.any()):
                            video.copy_(video_denoised)
                            if audio_denoised is not None:
                                audio_tail.copy_(audio_denoised)
                        else:
                            video_er_noise = torch.randn(video.shape, dtype=video.dtype, device=video.device, generator=er_generator)
                            audio_er_noise = torch.randn(audio.shape, dtype=audio.dtype, device=audio.device, generator=er_generator)
                            video, video_denoised_d = _er_sde_step(video, video_denoised, sigma_video, sigma_video_next,
                                                                  previous_sigma_video, previous_previous_sigma_video,
                                                                  old_video_denoised, old_video_denoised_d, video_er_noise, min(3, step + 1))
                            if audio_denoised is not None:
                                updated_audio, audio_denoised_d = _er_sde_step(audio_tail, audio_denoised, stage_sigmas_audio[step], stage_sigmas_audio[step + 1],
                                                                               stage_sigmas_audio[step - 1] if step else None,
                                                                               stage_sigmas_audio[step - 2] if step > 1 else None,
                                                                               old_audio_denoised, old_audio_denoised_d,
                                                                               audio_er_noise[..., target_audio_condition_latents:], min(3, step + 1))
                                audio_tail.copy_(updated_audio)
                                old_audio_denoised_d = audio_denoised_d
                            video_er_noise = audio_er_noise = None
                            old_video_denoised_d = video_denoised_d
                        old_video_denoised, old_audio_denoised = video_denoised, audio_denoised
                    elif stage_solver == PIKA_SOLVER:
                        final = step == model_steps - 1
                        if not target_video_condition_frames:
                            velocity_list, video_velocity = [video_velocity], None
                            pika_renoise_(video, velocity_list, sigma_video, sigma_video_next, renoise_generator, final=final)
                        if audio_tail.shape[-1] and not freeze_audio:
                            velocity_list, audio_velocity = [audio_velocity[..., target_audio_condition_latents:]], None
                            pika_renoise_(audio_tail, velocity_list, stage_sigmas_audio[step], stage_sigmas_audio[step + 1], renoise_generator, final=final, audio=True)
                    elif stage_solver == "euler":
                        video_ratio = sigma_video_next / sigma_video
                        if not target_video_condition_frames:
                            video_denoised = video_velocity.mul_(sigma_video).add_(video)
                            if effective_sigmas_video is None and source_latents is not None and step_editable_mask is not None:
                                _blend_video_source(video_denoised, source_latents, step_editable_mask)
                            if effective_sigmas_video is None:
                                video.mul_(video_ratio).add_(video_denoised, alpha=1.0 - video_ratio)
                            else:
                                video.mul_(video_ratio).add_(video_denoised * (1.0 - video_ratio))
                        if audio_tail.shape[-1]:
                            audio_ratio = stage_sigmas_audio[step + 1] / stage_sigmas_audio[step]
                            audio_velocity_tail = audio_velocity[..., target_audio_condition_latents:]
                            audio_velocity_tail.mul_(stage_sigmas_audio[step]).add_(audio_tail)
                            audio_tail.mul_(audio_ratio).add_(audio_velocity_tail, alpha=1.0 - audio_ratio)
                    elif stage_solver == "res_multistep":
                        coefficients = res_coefficients[step]
                        if not target_video_condition_frames:
                            video_denoised = video_velocity.mul_(sigma_video).add_(video)
                            if effective_sigmas_video is None and source_latents is not None and step_editable_mask is not None:
                                _blend_video_source(video_denoised, source_latents, step_editable_mask)
                            _res_multistep_update(video, video_denoised, old_video_denoised, coefficients)
                            old_video_denoised = video_denoised
                        if audio_tail.shape[-1]:
                            audio_velocity_tail = audio_velocity[..., target_audio_condition_latents:]
                            audio_denoised = audio_velocity_tail.mul_(stage_sigmas_audio[step]).add_(audio_tail).mul_(audio_scale)
                            audio_tail.mul_(stage_sigmas_video[step] / stage_sigmas_audio[step])
                            _res_multistep_update(audio_tail, audio_denoised, old_audio_denoised, audio_res_coefficients[step])
                            old_audio_denoised = audio_denoised
                    else:
                        sigma_audio = stage_sigmas_audio[step]
                        stage_sigma_video = ralston_effective_sigmas_video[step] if ralston_effective_sigmas_video is not None else ralston_sigmas_video[step]
                        stage_sigma_audio = ralston_sigmas_audio[step]
                        step_size = sigma_video - sigma_video_next
                        stage_size = sigma_video - stage_sigma_video
                        carrier_sigma_video = stage_sigmas_video[step]
                        carrier_stage_sigma_video = ralston_sigmas_video[step]
                        carrier_step_size = carrier_sigma_video - stage_sigmas_video[step + 1]
                        carrier_stage_size = carrier_sigma_video - carrier_stage_sigma_video
                        if effective_sigmas_video is None and source_latents is not None and step_editable_mask is not None:
                            video_velocity.mul_(sigma_video).add_(video)
                            _blend_video_source(video_velocity, source_latents, step_editable_mask)
                            video_velocity.sub_(video).div_(sigma_video)
                        stage_video = video if target_video_condition_frames else video_velocity.to(dtype=video.dtype, copy=True).mul_(stage_size).add_(video)
                        if effective_sigmas_video is None and source_latents is not None and not target_video_condition_frames and (step < denoising_start_step or step < mask_end_step):
                            stage_source_video = stage_video[:, :, :source_latents.shape[2]]
                            stage_source_mask = None if step < denoising_start_step else editable_mask
                            _reinject_video_source(stage_source_video, source_latents, source_noise, stage_source_mask, stage_sigma_video, source_buffer,
                                                   1.0 - VISUAL_COND_TIMESTEP if preserve_input_mask_values or payload["target_video_mask_active"] else None)
                        if audio_tail.shape[-1]:
                            audio_velocity_tail = audio_velocity[..., target_audio_condition_latents:]
                            audio_velocity_tail.mul_(1.0 + (audio_scale - 1.0) * sigma_audio).add_(audio_tail, alpha=audio_scale - 1.0)
                            audio_tail.mul_(carrier_sigma_video / sigma_audio)
                            stage_audio = audio.clone()
                            stage_audio_tail = stage_audio[..., target_audio_condition_latents:]
                            stage_audio_tail.add_(audio_velocity_tail.clone().mul_(carrier_stage_size)).mul_(stage_sigma_audio / carrier_stage_sigma_video)
                        else:
                            stage_audio = audio
                        self._check_abort()
                        payload["attention_sparsity"] = tau_start + (SOL_ATTN_TAU_END - tau_start) * min(step + 2.0 / 3.0, tau_denominator) / tau_denominator
                        stage_model_sigma_video = condition_control_video(stage_video, stage_sigma_video)
                        stage_video_velocity, stage_audio_velocity = self.transformer(stage_video, stage_audio, stage_model_sigma_video, stage_sigma_audio.view(1), context, payload, first_block_cache=first_block_cache)
                        self._check_abort()
                        if not target_video_condition_frames:
                            stage_video_velocity.mul_(stage_sigma_video).add_(stage_video)
                            if effective_sigmas_video is None and source_latents is not None and step_editable_mask is not None:
                                _blend_video_source(stage_video_velocity, source_latents, step_editable_mask)
                            stage_video_velocity.sub_(video).div_(sigma_video)
                            video_velocity.mul_(0.25).add_(stage_video_velocity, alpha=0.75).mul_(step_size)
                            video.add_(video_velocity)
                        if audio_tail.shape[-1]:
                            stage_audio_velocity_tail = stage_audio_velocity[..., target_audio_condition_latents:]
                            stage_audio_velocity_tail.mul_(1.0 + (audio_scale - 1.0) * stage_sigma_audio).add_(stage_audio_tail, alpha=audio_scale - 1.0)
                            stage_audio_tail.mul_(carrier_stage_sigma_video / stage_sigma_audio)
                            stage_audio_velocity_tail.mul_(carrier_stage_sigma_video).add_(stage_audio_tail).sub_(audio_tail).div_(carrier_sigma_video)
                            audio_velocity_tail.mul_(0.25).add_(stage_audio_velocity_tail, alpha=0.75).mul_(carrier_step_size)
                            audio_tail.add_(audio_velocity_tail)
                        stage_video = stage_audio = stage_video_velocity = stage_audio_velocity = None
                    final_masked_er_step = preserve_input_mask_values and stage_solver == "er_sde" and not bool(sigma_video_next.any())
                    if effective_sigmas_video is None and source_latents is not None and not final_masked_er_step and (step < denoising_start_step or step < mask_end_step):
                        source_video = video[:, :, :source_latents.shape[2]]
                        source_mask = None if step < denoising_start_step else editable_mask
                        keep_grouped_rows_fixed = grouped_masking and denoising_start_step <= step and step + 1 < mask_end_step
                        _reinject_video_source(source_video, source_latents, source_noise, source_mask, stage_sigmas_video[step + 1], source_buffer,
                                               1.0 - VISUAL_COND_TIMESTEP if preserve_input_mask_values or keep_grouped_rows_fixed else None)
                    video_velocity = audio_velocity = video_denoised = audio_velocity_tail = None
                    if control_anchor_indices is not None and not target_video_condition_frames:
                        video.index_copy_(2, control_anchor_indices, control_anchor_latents if step == model_steps - 1 else control_anchor_noised)
                    if callback is not None:
                        preview = video[0].detach() if not self.audio_only and (not offline_spectrum or spectrum.replaying) else None
                        callback(step, preview, False, denoising_extra=pass_extra, **({"pass_no": pass_no} if pass_no >= 0 else {}))

            try:
                if callback is not None:
                    callback(-1, None, True, override_num_inference_steps=model_steps, denoising_extra=denoising_extra, **({"pass_no": pass_no} if pass_no >= 0 else {}))
                if not offline_spectrum:
                    denoise_pass(description, denoising_extra)
                    return
                initial_video = video.detach().to(device="cpu", copy=True, non_blocking=False)
                initial_audio = audio.detach().to(device="cpu", copy=True, non_blocking=False)
                denoise_pass(f"{description} Spectrum anchor capture", denoising_extra)
                spectrum.complete_capture(self._check_abort)
                spectrum.start_replay()
                video = initial_video.to(self.device)
                audio = initial_audio.to(self.device)
                replay_extra = f"{denoising_extra} Spectrum smoothing replay".strip()
                if set_progress_status is not None:
                    set_progress_status(replay_extra)
                if callback is not None:
                    callback(-1, None, True, override_num_inference_steps=model_steps, denoising_extra=replay_extra, **({"pass_no": pass_no} if pass_no >= 0 else {}))
                denoise_pass(f"{description} Spectrum replay", replay_extra)
            finally:
                if spectrum is not None:
                    spectrum.reset()
                if first_block_cache is not None:
                    first_block_cache.reset()
                if audio_on_video_schedule:
                    audio[..., target_audio_condition_latents:].div_(audio_scale)

        phase_1_extra = "Phase 1/2 Low Resolution" if two_phase else ""
        if set_progress_status is not None:
            set_progress_status(phase_1_extra or "Denoising")
        run_denoising(sigmas_video, sigmas_audio, "H3 phase 1 denoising" if two_phase else "H3 denoising", phase_1_extra, 1 if two_phase else -1,
                      effective_sigmas_video=refinement_sigmas_video)

        decoded_video = None
        if two_phase:
            self._use_shared_components()
            if set_progress_status is not None:
                set_progress_status("Upscaling H3 latent between phases")

            def upscaler_progress(_phase, current_step, total_steps):
                if set_progress_status is not None:
                    set_progress_status(f"Upscaling H3 latent between phases ({int(current_step) + 1}/{int(total_steps)})")

            target_latent_h, target_latent_w = math.ceil(target_height / 16), math.ceil(target_width / 16)
            effective_scale = (target_latent_h / video.shape[-2] + target_latent_w / video.shape[-1]) / 2.0
            mean = video.new_tensor(LATENTS_MEAN, dtype=torch.bfloat16).view(1, -1, 1, 1, 1)
            std = video.new_tensor(LATENTS_STD, dtype=torch.bfloat16).view(1, -1, 1, 1, 1)
            normalized_video = (video.to(torch.bfloat16) - mean) / std
            video = self.latent_upscaler(normalized_video, effective_scale, target_size=(latent_t, target_latent_h, target_latent_w), abort_callback=lambda: self._interrupt, progress_callback=upscaler_progress)
            video = (video * std + mean).float()
            normalized_video = mean = std = None
            self._check_abort()

            apply_phase_2_lora_policy()
            activate_lora_phase(2, int(sampling_steps2))
            saved_phase_1_audio = None
            regenerate = False
            for index, lora in enumerate(loras_selected or ()):
                if os.path.basename(str(lora).split("|", 1)[0]).lower() != H3_REGENERATE_LORA:
                    continue
                multiplier = loras_slists["phase2"][index]
                regenerate = any(value != 0 for value in (multiplier if isinstance(multiplier, list) else [multiplier]))
                if regenerate:
                    break
            if regenerate and H3_REGENERATE_SILENT_AUDIO:
                if set_progress_status is not None:
                    set_progress_status("Encoding Silent Audio for Phase 2")
                saved_phase_1_audio = audio
                silence = torch.zeros((1, 2, audio_t * (AUDIO_SAMPLE_RATE // AUDIO_LATENT_FPS)), dtype=torch.float32, device="cpu")
                audio = self._encode_audio(silence).to(saved_phase_1_audio)
                del silence
            phase_2_reference_presentation, phase_2_reference_latents = [], []
            phase_2_refs = []
            if self.reference_mode:
                for image in input_ref_images or []:
                    self._add_image_reference(image, target_width, target_height, image_refs_relative_size, phase_2_reference_presentation, phase_2_reference_latents, phase_2_refs)
                for index, source in enumerate(video_sources):
                    if soundtracks[index] is not None:
                        phase_2_reference_presentation.append({"type": "audio"})
                    self._add_video_reference(_resize_video(source, target_height, target_width), None, fps, phase_2_reference_presentation, phase_2_reference_latents, [], phase_2_refs)
                visual_refs = [ref for ref in refs if ref["kind"] != "audio"]
                for phase_2_ref, original_ref in zip(phase_2_refs, visual_refs):
                    phase_2_ref["kind"] = original_ref["kind"]
                    phase_2_ref["ref_audio_t"] = original_ref.get("ref_audio_t", 0)
                for audio_ref in (ref for ref in refs if ref["kind"] == "audio"):
                    phase_2_reference_presentation.append({"type": "audio"})
                    phase_2_refs.append(audio_ref)
            phase_2_presentation = []
            phase_2_visual_latents, keyframes = prepare_keyframes(target_height, target_width, phase_2_presentation)
            phase_2_keyframe_count = len(keyframes)
            phase_2_presentation.extend(phase_2_reference_presentation)
            phase_2_visual_latents.extend(phase_2_reference_latents)
            if set_progress_status is not None:
                set_progress_status("Encoding H3 phase 2 prompt and references")
            context, text_tags = self._encode_prompt(input_prompt, phase_2_presentation)
            self._check_abort()
            phase_2_generator = torch.Generator(device="cpu").manual_seed(int(seed))
            phase_2_sigmas_video = (torch.tensor((float(switch_threshold), *H3_PHASE_2_SIGMAS[1:]), dtype=torch.float32, device=self.device)
                                   if int(sampling_steps2) == len(H3_PHASE_2_SIGMAS) - 1
                                   else torch.linspace(float(switch_threshold), 0.0, int(sampling_steps2) + 1, dtype=torch.float32, device=self.device))
            if phase_2_sigmas is not None:
                phase_2_sigmas_video = torch.tensor([*phase_2_sigmas, 0.0], dtype=torch.float32, device=self.device)
            target_audio_condition_latents = audio_t
            target_video_condition_frames = 0

            if tiled_phase_2:
                phase_2_keyframe_conditions = []
                for latent, keyframe in zip(phase_2_visual_latents[:phase_2_keyframe_count], keyframes):
                    timestep = keyframe.get("timestep", VISUAL_COND_TIMESTEP)
                    condition = latent.float().mul_(timestep)
                    condition.add_(torch.randn(latent.shape, generator=phase_2_generator, dtype=torch.float32, device="cpu"), alpha=1.0 - timestep)
                    phase_2_keyframe_conditions.append(condition)
                phase_2_reference_rows, _ = self._prepare_condition_rows(phase_2_visual_latents[phase_2_keyframe_count:], [], phase_2_generator)
                phase_2_visual_latents = None
                phase_2_base_video = video.detach().to(device="cpu", copy=True, non_blocking=False)
                phase_2_noise = torch.randn(video.shape, generator=torch.Generator(device="cpu").manual_seed(int(seed)), dtype=torch.float32, device="cpu")
                phase_2_latent_canvas = torch.lerp(phase_2_base_video, phase_2_noise, float(phase_2_sigmas_video[0]))
                video = phase_2_base_video = None
                row_tiles = _spatial_tiles(target_height)
                column_tiles = _spatial_tiles(target_width)
                tile_count = H3_PHASE_2_TILE_COUNT
                if video_to_video:
                    phase_2_source_video = _resize_video(control_source, target_height, target_width)
                    with control_video_encoding(control_video):
                        phase_2_source_latents = _encode_video_source(self.vae, phase_2_source_video, self.device, outpainting_dims)
                    phase_2_source_latents = phase_2_source_latents[:, :, :latent_t].to(device="cpu", dtype=phase_2_latent_canvas.dtype, non_blocking=False)
                    if input_masks is not None:
                        phase_2_source_mask = input_masks[:, history_count:history_count + phase_2_source_video.shape[1]]
                        phase_2_editable_mask = _resize_video_mask(phase_2_source_mask, phase_2_source_latents.shape[-3:], self.vae.config.clip_length,
                                                                   self.vae.temporal_compression_ratio, binarize=not preserve_input_mask_values)
                        if not preserve_input_mask_values:
                            phase_2_editable_mask = _snap_video_mask_to_patch_cells(phase_2_editable_mask, self.transformer.patch_size)
                    else:
                        phase_2_editable_mask = None
                    phase_2_source_video = phase_2_source_mask = None
                else:
                    phase_2_source_latents = phase_2_editable_mask = None
                if phase_2_source_latents is not None:
                    phase_2_source_noise = phase_2_noise[:, :, :phase_2_source_latents.shape[2]]
                    phase_2_source_buffer = torch.empty_like(phase_2_source_latents)
                    torch.lerp(phase_2_source_latents, phase_2_source_noise, float(phase_2_sigmas_video[0]), out=phase_2_source_buffer)
                    if phase_2_editable_mask is not None and float(masking_strength) > 0.0:
                        phase_2_latent_canvas[:, :, :phase_2_source_latents.shape[2]].lerp_(phase_2_source_buffer, 1.0 - phase_2_editable_mask)
                else:
                    phase_2_source_noise = phase_2_source_buffer = None

                phase_2_tiles = []
                phase_2_weight_sum = torch.zeros((1, 1, 1, target_latent_h, target_latent_w), dtype=torch.float32, device="cpu")
                for row, (top, tile_height) in enumerate(row_tiles):
                    top_overlap = 0 if not row else row_tiles[row - 1][0] + row_tiles[row - 1][1] - top
                    bottom_overlap = 0 if row + 1 == len(row_tiles) else top + tile_height - row_tiles[row + 1][0]
                    for column, (left, tile_width) in enumerate(column_tiles):
                        left_overlap = 0 if not column else column_tiles[column - 1][0] + column_tiles[column - 1][1] - left
                        right_overlap = 0 if column + 1 == len(column_tiles) else left + tile_width - column_tiles[column + 1][0]
                        latent_top, latent_left = top // 16, left // 16
                        latent_height, latent_width = tile_height // 16, tile_width // 16
                        tile_condition_rows = ([patchify_video(_crop_spatial_tile(condition, latent_top, latent_left, latent_height, latent_width), self.transformer.patch_size)
                                                for condition in phase_2_keyframe_conditions])
                        tile_condition_rows = torch.cat(tile_condition_rows) if tile_condition_rows else None
                        weights = _phase_2_tile_weights(latent_height, latent_width, top_overlap // 16, bottom_overlap // 16,
                                                        left_overlap // 16, right_overlap // 16)
                        phase_2_weight_sum[..., latent_top:latent_top + latent_height, latent_left:latent_left + latent_width].add_(weights)
                        phase_2_tiles.append((latent_top, latent_left, latent_height, latent_width, weights,
                                              tile_condition_rows, (target_latent_h, target_latent_w, latent_top, latent_left)))

                source_latents = source_noise = source_buffer = editable_mask = None
                self._use_transformer()
                context = self.transformer.preprocess_text_embeds(context)
                payload["keyframes"] = keyframes or None
                payload["refs"] = phase_2_refs or None
                payload["text_token_tags"] = text_tags
                payload["target_audio_condition_latents"] = audio_t
                payload["target_video_condition_frames"] = 0
                phase_extra = "Phase 2/2 Tiled"
                model_steps = phase_2_sigmas_video.numel() - 1
                phase_2_pdd_video_plans = pdd_sampling_plans_for_sigmas(phase_2_sigmas_video, shift, self.transformer.pdd_num_steps, grid_max_t) if pdd else None
                phase_2_pdd_audio_plan = pdd_sampling_plans(3.0, self.transformer.pdd_num_steps, self.transformer.pdd_block_size, grid_max_t)[-1:] if pdd else None
                progress_steps = model_steps * tile_count
                denoising_start_step = 0 if starting_sigma is not None else int(round(model_steps * (1.0 - float(denoising_strength)), 4))
                mask_end_step = min(model_steps, denoising_start_step + math.ceil(model_steps * float(masking_strength))) if phase_2_editable_mask is not None else 0
                grouped_masking = grouped_masked_denoising and phase_2_editable_mask is not None and not preserve_input_mask_values and mask_end_step > denoising_start_step
                if not grouped_masking:
                    _set_grouped_video_rows(payload, None, phase_2_latent_canvas.shape[-3:], self.device)
                if set_progress_status is not None:
                    set_progress_status(phase_extra)
                if callback is not None:
                    callback(-1, None, True, override_num_inference_steps=progress_steps, denoising_extra=phase_extra, pass_no=2)
                for step in tqdm(range(model_steps), desc="H3 phase 2 tiled denoising"):
                    self._set_interrupt_state()
                    self._check_abort()
                    payload["attention_sparsity"] = tau_start + (SOL_ATTN_TAU_END - tau_start) * step / max(1, model_steps - 1)
                    offload.set_step_no_for_lora(self.transformer, step)
                    if pdd:
                        self.transformer.final_layer.video_out.set_plan(phase_2_pdd_video_plans[step:step + 1])
                        self.transformer.final_layer.audio_out.set_plan(phase_2_pdd_audio_plan)
                    sigma_video, sigma_video_next = phase_2_sigmas_video[step], phase_2_sigmas_video[step + 1]
                    step_editable_mask = _masking_step_mask(phase_2_editable_mask, step, denoising_start_step, mask_end_step)
                    grouped_mask_active = grouped_masking and step_editable_mask is not None
                    if grouped_mask_active:
                        _reinject_video_source(phase_2_latent_canvas[:, :, :phase_2_source_latents.shape[2]], phase_2_source_latents,
                                               phase_2_source_noise, step_editable_mask, sigma_video, phase_2_source_buffer,
                                               1.0 - VISUAL_COND_TIMESTEP)
                    phase_2_accumulator = torch.zeros_like(phase_2_latent_canvas)
                    for tile_no, (latent_top, latent_left, latent_height, latent_width, weights, tile_condition_rows, spatial_context) in enumerate(phase_2_tiles, 1):
                        self._check_abort()
                        progress_step = step * tile_count + tile_no
                        condition_rows = [rows for rows in (tile_condition_rows, phase_2_reference_rows) if rows is not None]
                        payload["cond_video_rows"] = torch.cat(condition_rows) if len(condition_rows) > 1 else (condition_rows[0] if condition_rows else None)
                        payload["target_spatial_context"] = spatial_context
                        for key in ("layout_signature", "layout", "rope"):
                            payload.pop(key, None)
                        tile_video = _crop_spatial_tile(phase_2_latent_canvas, latent_top, latent_left, latent_height, latent_width).to(self.device)
                        if grouped_masking:
                            tile_editable_mask = _crop_spatial_tile(phase_2_editable_mask, latent_top, latent_left, latent_height, latent_width).to(self.device)
                            _set_grouped_video_rows(payload, tile_editable_mask, tile_video.shape[-3:], tile_video.device)
                            payload["target_video_mask_active"] = grouped_mask_active
                        tile_velocity, audio_velocity = self.transformer(tile_video, audio, sigma_video.view(1), phase_2_sigmas_video.new_zeros(1), context, payload)
                        tile_denoised = tile_velocity.float().mul_(sigma_video).add_(tile_video)
                        tile_prediction = tile_denoised.detach().to(device="cpu", non_blocking=False)
                        tile_prediction.mul_(weights)
                        phase_2_accumulator[..., latent_top:latent_top + latent_height, latent_left:latent_left + latent_width].add_(tile_prediction)
                        tile_video = tile_velocity = tile_denoised = tile_prediction = tile_editable_mask = audio_velocity = condition_rows = None
                        self._check_abort()
                        if callback is not None and tile_no < tile_count:
                            callback(progress_step - 1, None, False, denoising_extra=phase_extra, pass_no=2)
                    phase_2_accumulator.div_(phase_2_weight_sum)
                    if phase_2_source_latents is not None and step_editable_mask is not None:
                        _blend_video_source(phase_2_accumulator, phase_2_source_latents, step_editable_mask)
                    video_ratio = float(sigma_video_next / sigma_video)
                    phase_2_latent_canvas.mul_(video_ratio).add_(phase_2_accumulator, alpha=1.0 - video_ratio)
                    if phase_2_source_latents is not None and (step < denoising_start_step or step < mask_end_step):
                        phase_2_source_mask = None if step < denoising_start_step else phase_2_editable_mask
                        keep_grouped_rows_fixed = grouped_masking and denoising_start_step <= step and step + 1 < mask_end_step
                        _reinject_video_source(phase_2_latent_canvas[:, :, :phase_2_source_latents.shape[2]], phase_2_source_latents,
                                               phase_2_source_noise, phase_2_source_mask, float(sigma_video_next), phase_2_source_buffer,
                                               1.0 - VISUAL_COND_TIMESTEP if preserve_input_mask_values or keep_grouped_rows_fixed else None)
                    phase_2_accumulator = None
                    if callback is not None:
                        callback(progress_step - 1, phase_2_latent_canvas[0].detach(), False, denoising_extra=phase_extra, pass_no=2)

                video = phase_2_latent_canvas.to(self.device)
                payload.pop("target_spatial_context", None)
                for key in ("layout_signature", "layout", "rope"):
                    payload.pop(key, None)
                phase_2_noise = phase_2_latent_canvas = phase_2_keyframe_conditions = phase_2_reference_rows = None
                phase_2_source_latents = phase_2_source_noise = phase_2_source_buffer = phase_2_editable_mask = None
                phase_2_tiles = phase_2_weight_sum = None
            else:
                self._use_transformer()
                context = self.transformer.preprocess_text_embeds(context)
                payload["cond_video_rows"], _ = self._prepare_condition_rows(phase_2_visual_latents, [], phase_2_generator, keyframes)
                payload["keyframes"] = keyframes or None
                payload["refs"] = phase_2_refs or None
                payload["text_token_tags"] = text_tags
                payload["target_audio_condition_latents"] = audio_t
                payload["target_video_condition_frames"] = 0
                for key in ("layout_signature", "layout", "rope"):
                    payload.pop(key, None)
                presentation, visual_latents, refs = phase_2_presentation, phase_2_visual_latents, phase_2_refs
                phase_2_noise = torch.randn(video.shape, generator=torch.Generator(device="cpu").manual_seed(int(seed)), dtype=torch.float32, device="cpu").to(self.device)
                video = torch.lerp(video, phase_2_noise, phase_2_sigmas_video[0])
                if video_to_video:
                    self._use_shared_components()
                    source_video = _resize_video(control_source, target_height, target_width)
                    with control_video_encoding(control_video):
                        source_latents = _encode_video_source(self.vae, source_video, self.device, outpainting_dims).cpu()[:, :, :latent_t].to(video)
                    source_noise = phase_2_noise[:, :, :source_latents.shape[2]]
                    source_buffer = torch.empty_like(source_latents)
                    if input_masks is not None:
                        source_mask = input_masks[:, history_count:history_count + source_video.shape[1]]
                        editable_mask = _resize_video_mask(source_mask, source_latents.shape[-3:], self.vae.config.clip_length, self.vae.temporal_compression_ratio, binarize=not preserve_input_mask_values).to(video)
                        if not preserve_input_mask_values:
                            editable_mask = _snap_video_mask_to_patch_cells(editable_mask, self.transformer.patch_size)
                    self._use_transformer()
                else:
                    source_latents = source_noise = source_buffer = editable_mask = None
                phase_2_noise = source_video = source_mask = None
                if set_progress_status is not None:
                    set_progress_status("Phase 2/2 High Resolution")
                run_denoising(phase_2_sigmas_video, torch.zeros_like(phase_2_sigmas_video), "H3 phase 2 denoising", "Phase 2/2 High Resolution", 2,
                              freeze_audio=True, stage_solver=H3_PHASE_2_SAMPLE_SOLVER, use_cache=False)
            phase_2_presentation = phase_2_visual_latents = phase_2_reference_presentation = phase_2_reference_latents = phase_2_refs = None
            if saved_phase_1_audio is not None:
                audio = saved_phase_1_audio
                saved_phase_1_audio = None

        audio_refinement = "none" if self.audio_only or image_outputs else (custom_settings or {}).get(H3_AUDIO_REFINEMENT_SETTING, "none")
        if pdd or soundtrack or not self.reference_mode and any(flag in (audio_prompt_type or "") for flag in "AK"):
            audio_refinement = "none"
        if audio_refinement != "none":
            refinement_steps = H3_AUDIO_REFINEMENT_STEPS
            full_steps = round(refinement_steps / H3_AUDIO_REFINEMENT_DENOISE)
            refinement_base_sigmas = torch.linspace(1.0, 0.0, full_steps + 1, dtype=torch.float32, device=self.device)[-refinement_steps - 1:]
            refinement_sigmas_video = float(shift) * refinement_base_sigmas / (1.0 + (float(shift) - 1.0) * refinement_base_sigmas)
            refinement_sigmas_audio = 3.0 * refinement_base_sigmas / (1.0 + 2.0 * refinement_base_sigmas)
            audio_noise = torch.randn(audio.shape, generator=torch.Generator(device="cpu").manual_seed(int(seed)), dtype=torch.float32, device="cpu").to(audio)
            audio.lerp_(audio_noise, refinement_sigmas_audio[0])
            audio_noise = None
            if set_progress_status is not None:
                set_progress_status(f"Audio Refinement Extra Phase ({refinement_steps} steps, denoising {H3_AUDIO_REFINEMENT_DENOISE:g})")
            self._use_shared_components()
            context, text_tags = self._encode_prompt(input_prompt, [])
            self._check_abort()
            self._use_transformer()
            context = self.transformer.preprocess_text_embeds(context)
            target_video_condition_frames = video.shape[2]
            target_audio_condition_latents = 0
            source_latents = source_noise = source_buffer = editable_mask = None
            payload = {"keyframes": None, "audio_keyframes": None, "refs": None, "cond_video_rows": None,
                       "cond_audio_rows": None, "frame_count": aligned_target_frames, "text_token_tags": text_tags,
                       "fps": fps, "audio_time_offset": audio_time_offset, "target_audio_condition_latents": 0,
                       "target_video_condition_frames": target_video_condition_frames,
                       "attention_sparsity": float(attention_sparsity)}
            active_loras = list(getattr(self.transformer, "_loras_active_adapters", ()))
            lora_scaling = dict(getattr(self.transformer, "_loras_scaling", {}) or {})
            lora_step = getattr(self.transformer, "_lora_step_no", 0)
            try:
                offload.activate_loras(self.transformer, [])
                run_denoising(refinement_sigmas_video, refinement_sigmas_audio, "Audio Refinement Extra Phase",
                              "Audio Refinement Extra Phase", 3 if two_phase else 2, stage_solver="euler", use_cache=False)
            finally:
                offload.activate_loras(self.transformer, active_loras, [lora_scaling[name] for name in active_loras])
                offload.set_step_no_for_lora(self.transformer, lora_step)

        if set_progress_status is not None:
            set_progress_status("VAE Decoding of Image" if image_outputs else "Decoding H3 Stereo Audio" if self.audio_only or decoded_video is not None or frozen_target_video is not None else "VAE Decoding of Video and Audio")
        self._check_abort()
        self._use_shared_components()
        context = payload = presentation = visual_latents = audio_latents = refs = keyframes = audio_keyframes = control_source = source_latents = source_noise = source_buffer = editable_mask = None
        # One bounded CPU window; WGP owns its lifetime and clears it between tasks.
        # The shared return_latent_slice is expressed in frame-step units, not H3's VAE units.
        latent_slice = {"latents": video.detach().cpu(), "start_frame": window_start_frame_no} if H3_CONTROL_LATENT_CONTINUATION and control_target_anchors and not image_outputs else None
        control_anchor_indices = control_anchor_latents = control_anchor_noised = None
        # videos are decoded to uint8 frames, which only reach the RAM converted, with the rounding WGP (truncate) or the tiled phase 2 (round)
        # gave them; images and refinements (the H3 Face Refiner blends signed windows) keep a float video
        uint8_rounding = "round" if tiled_phase_2 else None if image_outputs or refinement_mode else "truncate"
        to_uint8 = _video_to_uint8_cpu if tiled_phase_2 else convert_video_tensor_to_uint8_chunked
        if not self.audio_only and decoded_video is None:
            if frozen_target_video is None:
                video = video.to(self.vae._model_dtype)
                decoded_video = self.vae.decode(video, uint8_rounding)[0, :, :target_frames]
                if uint8_rounding is None:
                    decoded_video = decoded_video.clamp_(-1.0, 1.0)
            else:
                decoded_video = frozen_target_video[:, :target_frames].cpu()
        video = None
        for frame_index, known in control_frame_anchors:  # restore exact supplied pixels after the VAE round trip
            known = _resize_video(known.float(), *decoded_video.shape[-2:])
            decoded_video[:, frame_index:frame_index + known.shape[1]] = (to_uint8(known) if decoded_video.dtype == torch.uint8 else known).to(decoded_video)
        if image_outputs:
            audio = None
            self._check_abort()
            return {"x": decoded_video}
        if set_progress_status is not None:
            set_progress_status("Decoding H3 Stereo Audio")
        decoded_audio = self.audio_vae.decode(audio)[0]
        audio = None
        target_samples = round((target_frames / fps - audio_time_offset / AUDIO_LATENT_FPS) * AUDIO_SAMPLE_RATE)
        tail_samples = round(audio_tail_latents * AUDIO_SAMPLE_RATE / AUDIO_LATENT_FPS)
        decoded_audio = _fit_audio_samples(decoded_audio, target_samples) if self.audio_only else decoded_audio[..., :target_samples + tail_samples]

        output_prefix = history_frames
        if output_prefix is not None:
            if (two_phase or self.vae.upsampling) and output_prefix.shape[-2:] != decoded_video.shape[-2:]:
                output_prefix = _resize_video(output_prefix, decoded_video.shape[-2], decoded_video.shape[-1])
            if decoded_video.dtype == torch.uint8 and output_prefix.dtype != torch.uint8:
                output_prefix = output_prefix.clamp(-1.0, 1.0).add_(1.0).mul_(127.5).round_().to(torch.uint8) if tiled_phase_2 else to_uint8(output_prefix)
            decoded_video = torch.cat((output_prefix.to(decoded_video), decoded_video), dim=1)
            if history_waveform is not None:
                prefix_samples = round(audio_start_seconds * AUDIO_SAMPLE_RATE)
                prefix_audio = _fit_audio_samples(history_waveform[0].to(decoded_audio), prefix_samples)
            else:
                prefix_samples = round(audio_start_seconds * AUDIO_SAMPLE_RATE)
                prefix_audio = torch.zeros((2, prefix_samples), dtype=decoded_audio.dtype, device=decoded_audio.device)
            decoded_audio = torch.cat((prefix_audio, decoded_audio), dim=1)

        total_samples = round(frame_num / fps * AUDIO_SAMPLE_RATE)
        decoded_audio = decoded_audio[:, :total_samples + tail_samples].transpose(0, 1).float().cpu().numpy()
        if self.audio_only:
            return {"x": torch.from_numpy(decoded_audio.T.copy()), "audio_sampling_rate": AUDIO_SAMPLE_RATE,
                    "overridden_inputs": {"resolution": "32x32", "video_length": frame_num, "duration_seconds": round(frame_num / fps, 3)}}
        samples = {"x": decoded_video, "audio": decoded_audio, "audio_sampling_rate": AUDIO_SAMPLE_RATE}
        if latent_slice is not None:
            samples["latent_slice"] = latent_slice
        return samples

    def refine_video(self, video, *, prompt, strengths, denoising_strength=0.45, sampling_steps=4, shift=12.0,
                     seed=0, fps=24.0, sample_solver="euler", VAE_tile_size=None, audio_waveform=None,
                     audio_sample_rate=0, reference_images=None, image_refs_relative_size=100.0,
                     callback=None, set_progress_status=None):
        video = _as_video(video)
        if video is None:
            raise ValueError("MiniMax H3 refinement requires a source video")
        strengths = torch.as_tensor(strengths, dtype=torch.float32).flatten()
        if strengths.numel() != video.shape[1]:
            raise ValueError(f"MiniMax H3 refinement received {strengths.numel()} strengths for {video.shape[1]} frames")
        if not bool((strengths == strengths[0]).all()):
            raise ValueError("MiniMax H3 refinement requires one uniform strength for the whole video")
        if not 0.0 <= float(denoising_strength) <= 1.0:
            raise ValueError("MiniMax H3 refinement denoising strength must be between 0 and 1")
        strength_mask = strengths.clamp(0.0, 1.0).view(1, -1, 1, 1)
        if float(denoising_strength) == 0.0 or not bool(strength_mask.any()):
            return video
        total_steps = max(int(sampling_steps), int(int(sampling_steps) / float(denoising_strength)))
        base_start = float(sampling_steps) / total_steps
        start = float(shift) * base_start / (1.0 + (float(shift) - 1.0) * base_start)
        result = self.generate(
            input_prompt=prompt,
            input_frames=video,
            input_ref_images=reference_images,
            image_refs_relative_size=image_refs_relative_size,
            input_masks=strength_mask,
            denoising_strength=1.0,
            masking_strength=1.0,
            input_waveform=audio_waveform,
            input_waveform_sample_rate=audio_sample_rate,
            frame_num=video.shape[1],
            height=video.shape[-2],
            width=video.shape[-1],
            shift=shift,
            sampling_steps=sampling_steps,
            seed=seed,
            callback=callback,
            VAE_tile_size=VAE_tile_size,
            audio_prompt_type="A" if audio_waveform is not None else "",
            fps=fps,
            sample_solver=sample_solver,
            guide_phases=1,
            starting_sigma=start,
            preserve_input_mask_values=True,
            refinement_mode=True,
            set_progress_status=set_progress_status,
        )
        return None if result is None else result["x"]


__all__ = ["MiniMaxH3Pipeline", "video_latent_frames"]
