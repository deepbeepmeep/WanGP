# Adapted from TaoMate-FVSR (https://github.com/TaoLiveAIGC/TaoMate-FVSR, Apache-2.0, see LICENSE in this folder): the window geometry, the
# guidance and the overlap replay of its inference. Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Restores a track of square face crops by windows of 77 frames. Each window holds 20 latent frames: slots 0 and 1 anchor the reference, slots 2 to
19 cover 72 frames. Successive windows move by 56 frames: slots 2 to 5 replay the previous window's slots 16 to 19 at every noise level, slot 5 of the
reference canvas holds the last frame already restored, and one VAE decoder runs across the windows.

Upstream anchors on the first frame of the video, restored. Here the reference can be any picture framed like the crops, such as the track's best frame:
its latents fill the anchor slots too, which later windows already hold away from the frames they restore. A first frame that is small or damaged would
otherwise be copied by the model, since it takes facial detail from the reference. Given the first frame itself, this is upstream's sequence."""
import os

import numpy as np
import torch
from tqdm import tqdm

from mmgp import offload
from shared.hubert import FEATURES_PER_SECOND, SAMPLE_RATE, HubertEncoder
from shared.utils.phase_progress import control_video_encoding, generation_progress, vae_decoding_progress, vae_encoding_progress

from ..modules.model import WanModel
from ..modules.posemb_layers import get_rotary_pos_embed, set_rope_freqs_dtype, set_use_fp32_rope_freqs
from ..modules.vae2_2 import Wan2_2_VAE
from . import AUDIO_FEATURES, AUDIO_FRAMES, AUDIO_WINDOW

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")
LATENT_CHANNELS = 48
WINDOW_FRAMES = 77
BOOTSTRAP_FRAMES = 5
PAYLOAD_FRAMES = 72
WINDOW_STRIDE = 56
OVERLAP_FRAMES = 16
REPLAY_SLOTS = slice(2, 6)
TRACE_SLOTS = slice(16, 20)
PREVIOUS_FRAME_SLOT = 5
ANCHOR = -1  # in encoded_frames: the reference picture


def window_plans(frame_count):
    """What each window reads and delivers: the source frames of its 77 positions, its first latent frame, the frames it delivers, the slots to decode."""
    plans, start, delivered = [], 0, 0
    while delivered < frame_count:
        first = not plans
        requested = [0] * BOOTSTRAP_FRAMES + list(range(PAYLOAD_FRAMES)) if first else list(range(start - BOOTSTRAP_FRAMES, start + PAYLOAD_FRAMES))
        stop = min(start + PAYLOAD_FRAMES, frame_count)
        begin = 0 if first else start + OVERLAP_FRAMES
        decode_start = 0 if first else 6
        plans.append({"frames": [min(index, frame_count - 1) for index in requested], "latent_start": start // 4, "delivered": (begin, stop), "discard": BOOTSTRAP_FRAMES if first else 0,
                      "decode": (decode_start, AUDIO_FRAMES if first else decode_start + (stop - begin + 3) // 4)})
        delivered, start = stop, start + WINDOW_STRIDE
    return plans


def encoded_frames(plans, frame_count):
    """The pictures of the whole latent sequence: the reference repeated for the anchor slots, then the source frames up to the last window's end."""
    return [ANCHOR] * BOOTSTRAP_FRAMES + [min(index, frame_count - 1) for index in range(plans[-1]["latent_start"] * 4 + PAYLOAD_FRAMES)]


def schedule(steps, shift):
    """The shifted 1000-entry Wan schedule with rounded boundaries: sigmas and the next sigma of each step, on the CPU as trained."""
    base = torch.linspace(1.0, 0.0, 1001, device="cpu")[:-1]
    base = shift * base / (1 + (shift - 1) * base)
    sigmas = base[torch.linspace(0, 1000, steps + 1, device="cpu").round().long()[:-1]]
    return sigmas, torch.cat([sigmas[1:], sigmas.new_zeros(1)])


def window_pcm(pcm, frames, first_frame, fps):
    """The audio of a window's positions, each its source frame's slice of pcm (float32, 16 kHz), zeros beyond the end of the audio: (samples, feature index of each position)."""
    parts, lengths = [], []
    for index in frames:
        start, stop = (round((first_frame + index + offset) * SAMPLE_RATE / fps) for offset in (0, 1))
        part = pcm[max(0, start):max(0, stop)]
        parts.append(np.pad(part, (0, stop - start - len(part))))
        lengths.append(stop - start)
    centers = np.cumsum([0, *lengths[:-1]]) * FEATURES_PER_SECOND / SAMPLE_RATE
    return np.concatenate(parts), np.round(centers).astype(np.int64)


def audio_windows(hidden_states, centers):
    """The eight hidden states of each latent frame (frames, window, features) from HuBERT's (count, layers, width): the first slot looks back from the first position,
    the others forward from the first position they cover, the indices clamped to the available states."""
    flat = hidden_states.flatten(1)
    windows = []
    for slot in range(AUDIO_FRAMES):
        center = int(centers[0 if slot == 0 else (slot - 1) * 4 + 1])
        indices = torch.arange(center - 6, center + 2, device=flat.device) if slot == 0 else torch.arange(center, center + 8, device=flat.device)
        windows.append(flat.index_select(0, indices.clamp_(0, flat.shape[0] - 1)))
    return torch.stack(windows)


def _to_vae_input(frames, device):
    """uint8 (3, T, H, W) to [-1, 1] on the GPU."""
    return frames.to(device=device, dtype=torch.float32, non_blocking=True).mul_(2.0 / 255.0).sub_(1.0)


def _encode_sequence(vae, frames, load_frames, reference, frame_size, tile_size, device):
    """The latents (1, channels, count, height, width) of the pictures at the indices `frames` (ANCHOR: the reference), in RAM, read and encoded by chunks."""
    stream = vae.encode_stream(frame_size, tile_size)
    chunks = [frames[:1], *(frames[start:start + 4] for start in range(1, len(frames), 4))]
    latents = []
    with control_video_encoding(), vae_encoding_progress(len(chunks) * max(1, _stream_tiles(stream)), vae.model.encoder, cleanup=vae.model.clear_cache):
        for chunk in chunks:
            if chunk[0] == ANCHOR:
                pixels = reference.unsqueeze(1).expand(-1, len(chunk), -1, -1)
            else:
                low, high = min(chunk), max(chunk) + 1
                pixels = load_frames(low, high)[:, [index - low for index in chunk]]
            latents.append(vae.stream_encode(_to_vae_input(pixels, device), stream).cpu())
            pixels = None
    return torch.stack(latents, dim=0).squeeze(2).permute(1, 0, 2, 3).unsqueeze(0)


def _stream_tiles(stream):
    return 1 if stream.exact is None else sum(len(tiles) for _, _, _, tiles in stream.exact.work)


def _branches(has_audio, audio_cfg, ref_cfg):
    """(reference, audio) of each model pass: the positive reference then the negative one, each with the positive audio then the negative one."""
    audios = (True, False) if has_audio and audio_cfg != 1 else (has_audio,)
    return [(reference, audio) for reference in ((True, False) if ref_cfg != 1 else (True,)) for audio in audios]


def _guided(outputs, audio_passes, audio_cfg, ref_cfg):
    """The guided velocity: audio guidance inside each reference branch, then reference guidance. outputs is cleared."""
    results = []
    for start in range(0, len(outputs), audio_passes):
        positive = outputs[start]
        if audio_passes == 2:
            negative = outputs[start + 1]
            outputs[start + 1] = None
            positive = negative.lerp_(positive, audio_cfg)
        outputs[start] = None
        results.append(positive)
    outputs.clear()
    return results[0] if len(results) == 1 else results[1].lerp_(results[0], ref_cfg)


def refine_track(pipeline, frame_count, load_frames, reference, pcm, *, first_frame, fps, size, callback, set_status, seed=42, steps=10, noise=0.8, shift=5.0, audio_cfg=4.0, ref_cfg=1.7, tile_size=0, window_offset=0, window_total=None):
    """Restored uint8 crops (3, frames, size, size), or None when aborted. load_frames(start, stop): uint8 crops (3, T, size, size) in RAM; reference: uint8 (3, size, size), framed like
    the crops; pcm: float32 mono at 16 kHz of the whole audio track or None, whose frame first_frame starts this track."""
    device, dtype, model, vae = pipeline.device, pipeline.dtype, pipeline.model, pipeline.vae
    frame_size = (size, size)
    plans = window_plans(frame_count)
    latent_size = size // 16
    has_audio = pcm is not None

    set_status("Processing Audio")
    hidden_windows = None
    if has_audio:
        hidden_windows = []
        for plan in plans:
            samples, centers = window_pcm(pcm, plan["frames"], first_frame, fps)
            hidden_windows.append(audio_windows(pipeline.audio_encoder(samples), centers).to("cpu", dtype))
            if pipeline._interrupt:
                return None

    set_status("Encoding Video")
    global_latents = _encode_sequence(vae, encoded_frames(plans, frame_count), load_frames, reference, frame_size, tile_size, device)
    if pipeline._interrupt:
        return None
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    noise_latents = torch.randn((1, LATENT_CHANNELS, AUDIO_FRAMES, latent_size, latent_size), generator=generator, device="cpu")
    if global_latents.shape[2] > AUDIO_FRAMES:
        tail = torch.randn((1, LATENT_CHANNELS, global_latents.shape[2] - AUDIO_FRAMES, latent_size, latent_size), generator=generator, device="cpu")
        noise_latents, tail = torch.cat([noise_latents, tail], dim=2), None

    reference_input = _to_vae_input(reference.unsqueeze(1), device)
    identity = vae.encode([reference_input], tile_size)[0]
    bootstrap = vae.encode([reference_input.expand(-1, BOOTSTRAP_FRAMES, -1, -1)], tile_size)[0][:, 1:2]
    reference_input = None
    canvas = torch.zeros((1, LATENT_CHANNELS, AUDIO_FRAMES, latent_size, latent_size), dtype=torch.float32, device=device)
    canvas[0, :, :1] = identity
    canvas[0, :, 1:2] = bootstrap
    identity = bootstrap = None

    sigmas, following = schedule(steps, shift)
    freqs = get_rotary_pos_embed((AUDIO_FRAMES, latent_size, latent_size), enable_RIFLEx=False, device=device)
    branches = _branches(has_audio, audio_cfg, ref_cfg)
    audio_passes = 2 if has_audio and audio_cfg != 1 else 1
    silence = model.project_audio(torch.zeros((AUDIO_FRAMES, AUDIO_WINDOW, AUDIO_FEATURES), dtype=dtype, device=device))
    decoder = vae.decode_stream(frame_size, tile_size)
    output = torch.empty((3, frame_count, size, size), dtype=torch.uint8, device="cpu")
    previous_trace = last_frame = None
    offload.shared_state["_chipmunk"] = offload.shared_state["_radial"] = False
    window_total = len(plans) if window_total is None else window_total

    for number, plan in enumerate(plans):
        status = f"Window {window_offset + number + 1}/{window_total} - " if window_total > 1 else ""
        title = f"{status}Denoising"
        if number:
            canvas[:, :, PREVIOUS_FRAME_SLOT:PREVIOUS_FRAME_SLOT + 1] = vae.encode([last_frame], tile_size)[0].unsqueeze(0)
        audio_tokens = silence if not has_audio else model.project_audio(hidden_windows[number].to(device=device, dtype=dtype))
        hidden_windows = hidden_windows if not has_audio else [None if i == number else tensor for i, tensor in enumerate(hidden_windows)]
        start = plan["latent_start"]
        latents = global_latents[:, :, start:start + AUDIO_FRAMES].to(device=device, dtype=torch.float32)
        if start:
            latents[:, :, :2] = global_latents[:, :, :2].to(device=device, dtype=torch.float32)
        state = latents.lerp_(noise_latents[:, :, start:start + AUDIO_FRAMES].to(device=device, dtype=torch.float32), noise)
        latents = None
        trace = [state[:, :, TRACE_SLOTS].cpu()]
        callback(-1, None, True, override_num_inference_steps=steps, progress_title=title)
        for step in tqdm(range(steps), desc="TaoMate-FVSR denoising", leave=False):
            if number:
                state[:, :, REPLAY_SLOTS] = previous_trace[step].to(device)
            timestep = torch.full((1,), float(sigmas[step]) * 1000.0, dtype=dtype, device=device)  # rounded to the dtype the model was evaluated with
            outputs = model([state] * len(branches), timestep, [None] * len(branches), freqs=freqs, pipeline=pipeline, callback=callback, current_step_no=step, real_step_no=step, max_steps=steps,
                            taomate_reference=[canvas if with_reference else None for with_reference, _ in branches], taomate_audio=[audio_tokens if with_audio else silence for _, with_audio in branches])
            if pipeline._interrupt or outputs[0] is None:
                return None
            velocity = _guided(outputs, audio_passes, audio_cfg, ref_cfg)
            state.add_(velocity, alpha=float(following[step] - sigmas[step]))
            velocity = None
            if number:
                state[:, :, REPLAY_SLOTS] = previous_trace[step + 1].to(device)
            trace.append(state[:, :, TRACE_SLOTS].cpu())
            callback(step, None, False, override_num_inference_steps=steps, progress_title=title)
        if not torch.isfinite(state).all():
            raise RuntimeError("TaoMate-FVSR produced non-finite latents")
        previous_trace, trace = trace, None
        generated, state, audio_tokens = state.cpu(), None, None
        begin, stop = plan["delivered"]
        pixels = []
        with vae_decoding_progress((plan["decode"][1] - plan["decode"][0]) * _stream_tiles(decoder), vae.model.decoder):
            for slot in range(*plan["decode"]):
                frames = vae.stream_decode(generated[0, :, slot:slot + 1], decoder)
                pixels.append(frames.float().add_(1.0).mul_(127.5).round_().clamp_(0.0, 255.0).to("cpu", torch.uint8))
                frames = None
        pixels = torch.cat(pixels, dim=1)[:, plan["discard"]:plan["discard"] + stop - begin]
        output[:, begin:stop] = pixels
        last_frame = _to_vae_input(pixels[:, -1:], device)
        generated = pixels = None
    return output


class TaoMatePipeline:
    """The model, its VAE and its audio encoder, at the precision of their prepared checkpoints, for a post-processor restoring tracks."""
    _interrupt = False

    def __init__(self, transformer_path, audio_encoder_path, vae_path, dtype=torch.bfloat16, vae_dtype=torch.float16):
        self.device = torch.device("cuda")
        self.dtype = dtype
        set_use_fp32_rope_freqs(True)
        set_rope_freqs_dtype(dtype)
        self.audio_encoder = HubertEncoder(audio_encoder_path)
        self.vae = Wan2_2_VAE(vae_pth=vae_path, dtype=vae_dtype, upsampler_factor=1, device="cpu")
        self.vae.device = self.device
        self.model = offload.fast_load_transformers_model([transformer_path], modelClass=WanModel, defaultConfigPath=CONFIG_PATH, forcedConfigPath=CONFIG_PATH, writable_tensors=False, default_dtype=dtype,
                                                          preprocess_sd=lambda state: WanModel.preprocess_sd_with_dtype(dtype, state), pre_load_callback=lambda model: model.lock_layers_dtypes(dtype))
        self.model.apply_post_init_changes()

    @generation_progress
    @torch.inference_mode()
    def refine(self, frame_count, load_frames, reference, pcm, *, callback=None, set_progress_status=None, **kwargs):
        return refine_track(self, frame_count, load_frames, reference, pcm, callback=callback, set_status=set_progress_status, **kwargs)


__all__ = ["TaoMatePipeline", "WINDOW_FRAMES", "refine_track", "window_plans"]
