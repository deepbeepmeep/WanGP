import json
import math
import os

import torch
from accelerate import init_empty_weights
from mmgp import offload
from tqdm import tqdm

from models.hyvideo.vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from postprocessing.mmaudio.ext.autoencoder import AutoEncoderModule
from shared.utils import files_locator as fl
from shared.utils.phase_progress import check_abort, generation_progress, set_phase_status
from .piflow import DXPolicy, policy_rollout, segment_schedule, shift_timesteps
from .text_encoder import Kandinsky6TextEncoder
from .transformer import Kandinsky6Transformer

VIDEO_VAE_FILE = "hunyuan_video_VAE_fp32.safetensors"  # shared with Hunyuan Video and Kandinsky 5, loaded in FP16: upstream decodes in FP16
VIDEO_VAE_CONFIG_FILE = "hunyuan_video_VAE_config.json"
AUDIO_VAE_FILE = "mmaudio/v1-44.pth"
CLIP_FOLDER = "clip_vit_large_patch14"
AUDIO_SAMPLE_RATE = 44100
AUDIO_DOWNSAMPLE = 1024
VAE_TILE_KEYS = ("tile_sample_min_tsize", "tile_latent_min_tsize", "tile_sample_min_size", "tile_latent_min_size", "tile_overlap_factor")


def _piflow_step(prediction, state, sigma, segment, tau_src, tau_dst, piflow):
    """Rolls a video or audio latent over one segment with the DX policy of the DiT grid prediction (upstream piflow_denoise_loop):
    the latent becomes FP32 after the first step, as upstream."""
    shape = (1,) * state.dim()
    sigma = sigma.reshape(shape)
    policy = DXPolicy(prediction, state, sigma, segment.reshape(shape), piflow["shift"], piflow["eps"])
    return policy_rollout(state, sigma, tau_src.reshape(shape), tau_dst.reshape(shape), piflow["num_policy_substeps"], policy)


class Kandinsky6VideoVAE(AutoencoderKLCausal3D):
    def get_VAE_tile_size(self, vae_config, device_mem_capacity, mixed_precision):
        return super().get_VAE_tile_size(vae_config, device_mem_capacity, False)  # always the FP16 VAE: the 16-bit tile budgets


class model_factory:
    def __init__(self, model_filename, model_type, base_model_type, text_encoder_filename, save_quantized=False):
        self.device = torch.device("cuda")
        with open(os.path.join(os.path.dirname(__file__), "configs", f"{base_model_type}.json"), encoding="utf-8") as reader:
            self.config = json.load(reader)
        model_filename = model_filename[0] if isinstance(model_filename, (list, tuple)) else model_filename
        with init_empty_weights():
            transformer = Kandinsky6Transformer(**self.config["transformer"])
        transformer.lock_checkpoint_fp32_modules()
        # BF16 is the checkpoint dtype of every unlocked tensor, and the compute dtype of the quantized layers
        offload.load_model_data(transformer, model_filename, writable_tensors=False, default_dtype=torch.bfloat16)
        self.transformer = transformer.eval().requires_grad_(False)
        if save_quantized:
            from wgp import save_quantized_model
            save_quantized_model(transformer, model_type, model_filename, torch.bfloat16, None)
        self.text_encoder = Kandinsky6TextEncoder(text_encoder_filename, fl.locate_folder(CLIP_FOLDER))
        self.vae = offload.fast_load_transformers_model(fl.locate_file(VIDEO_VAE_FILE), writable_tensors=False, modelClass=Kandinsky6VideoVAE,
                                                        defaultConfigPath=fl.locate_file(VIDEO_VAE_CONFIG_FILE), default_dtype=torch.float16).eval()
        self.audio_vae = AutoEncoderModule(vae_ckpt_path=fl.locate_file(AUDIO_VAE_FILE), mode="44k", need_vae_encoder=False)
        # The Kandinsky 6 checkpoints store these MMAudio weights already weight-normalized and normalize them again when loading.
        self.audio_vae.vae.remove_weight_norm()
        self.audio_vae._convertWeightsFloatTo = None
        self._interrupt = False

    def _set_vae_tiling(self, tile_config):
        if tile_config is None:
            self.vae.disable_tiling()
            return
        for key in VAE_TILE_KEYS:
            setattr(self.vae, key, tile_config[key])
        self.vae.enable_tiling()

    def _encode_start_image(self, image_start, seed):
        image = image_start.to(device=self.device, dtype=self.vae.dtype)[None, :, None]
        generator = torch.Generator(device=self.device).manual_seed(seed + 2)
        latent = self.vae.encode(image).latent_dist.sample(generator) * self.vae.config.scaling_factor
        return latent.permute(0, 2, 3, 4, 1)  # (1, 1, height, width, channels)

    @generation_progress
    @torch.inference_mode()
    def generate(self, input_prompt, seed, sampling_steps, frame_num, height, width, fps, image_start=None, callback=None, VAE_tile_size=None, **kwargs):
        device, transformer = self.device, self.transformer
        self._set_vae_tiling(VAE_tile_size)
        latent_frames, latent_height, latent_width = (frame_num - 1) // 4 + 1, height // 8, width // 8
        reference = None if image_start is None else self._encode_start_image(image_start, seed)
        text = self.text_encoder.encode(input_prompt, device)
        check_abort()

        generator = torch.Generator(device=device).manual_seed(seed)
        video = torch.randn(latent_frames, latent_height, latent_width, transformer.in_visual_dim, device=device, dtype=torch.bfloat16, generator=generator)[None]
        audio_tokens = math.ceil(((latent_frames - 1) * 4 + 1) / fps * AUDIO_SAMPLE_RATE / AUDIO_DOWNSAMPLE)
        generator = torch.Generator(device=device).manual_seed(seed + 1)
        audio = torch.randn(audio_tokens, transformer.in_audio_dim, device=device, dtype=torch.bfloat16, generator=generator)[None]
        mask = torch.zeros((*video.shape[:-1], 1), device=device, dtype=torch.bfloat16)
        token_types = None
        if reference is not None:
            # the start image is a clean extra latent frame (token type 1) appended after the generated ones, masked in the conditioning
            video = torch.cat([video, reference.to(torch.bfloat16)], dim=1)
            mask = torch.cat([mask, torch.ones_like(mask[:, :1])], dim=1)
            token_types = torch.cat([torch.zeros(latent_frames, dtype=torch.long, device=device), torch.ones(1, dtype=torch.long, device=device)])[None]
        patch = transformer.patch_size
        ropes = transformer.rope_tables(latent_frames, latent_height // patch[1], latent_width // patch[2], text["text_embeds"].shape[1], audio_tokens,
                                        self.config["rope_scale_factor"], device, reference_frame=reference is not None)

        piflow = self.config["piflow"]
        shift, eps = piflow["shift"], piflow["eps"]
        raw_t_src = torch.ones(1, device=device, dtype=torch.float32)
        if callback is not None:
            callback(-1, None, True, override_num_inference_steps=sampling_steps)
        for step, segment_size in enumerate(tqdm(segment_schedule(sampling_steps, piflow["final_step_size_scale"], eps))):
            tau_src = raw_t_src
            tau_dst = (tau_src - segment_size).clamp(min=eps)
            sigma = shift_timesteps(tau_src, shift)
            if reference is not None:
                video[:, -1] = reference.to(video.dtype)
            model_input = torch.cat([video, torch.zeros_like(video), mask.to(video.dtype)], dim=-1)
            pred_video, pred_audio = transformer(model_input, audio, text["text_embeds"], text["pooled_embed"], sigma * 1000, ropes, token_types)
            del model_input
            segment = torch.full((1,), float(segment_size), device=device, dtype=sigma.dtype)
            # The last PiFlow grid entry predicts x0 at the current source time.
            preview = video.float() - sigma * pred_video[:, -1].float() if getattr(callback, "tiny_vae", False) else None
            video = _piflow_step(pred_video, video, sigma, segment, tau_src, tau_dst, piflow)
            del pred_video
            audio = _piflow_step(pred_audio, audio, sigma, segment, tau_src, tau_dst, piflow)
            del pred_audio
            raw_t_src = tau_dst
            if callback is not None:
                callback(step, (preview if preview is not None else video)[0, :latent_frames].permute(3, 0, 1, 2), False)
            del preview
            check_abort()
        del text, ropes

        latents = (video[:, :latent_frames] / self.vae.config.scaling_factor).permute(0, 4, 1, 2, 3).to(self.vae.dtype)
        del video
        frames = self.vae.decode(latents).sample[0].clamp_(-1, 1)
        del latents
        set_phase_status("Decoding Audio")
        mel = self.audio_vae.decode((audio / self.config["audio_scaling_factor"]).transpose(1, 2).float())
        waveform = self.audio_vae.vocode(mel).flatten().clamp_(-1, 1).cpu().numpy()
        waveform = waveform[:round(frames.shape[1] / fps * AUDIO_SAMPLE_RATE)]
        return {"x": frames, "audio": waveform, "audio_sampling_rate": AUDIO_SAMPLE_RATE}
