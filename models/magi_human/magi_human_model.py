from shared.utils.phase_progress import text_encoding_progress, generation_progress, set_phase_status
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.functional as taF
from accelerate import init_empty_weights
from diffusers.video_processor import VideoProcessor
from mmgp import offload
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers.models.t5gemma import T5GemmaEncoderModel

from shared.utils import files_locator as fl
from shared.utils.loras_mutipliers import update_loras_slists
from shared.utils.text_encoder_cache import TextEncoderCache
from shared.utils.utils import calculate_new_dimensions
from models.wan.modules.vae2_2 import Wan2_2_VAE

_UPSTREAM_ROOT = Path(__file__).resolve().parent / "upstream"
if str(_UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM_ROOT))

from inference.model.dit.dit_module import DiTModel, MagiAbortRequested  # noqa: E402
from inference.model.sa_audio import SAAudioFeatureExtractor  # noqa: E402
from inference.model.turbo_vaed import get_turbo_vaed  # noqa: E402
from inference.pipeline.data_proxy import MagiDataProxy  # noqa: E402
from inference.pipeline.scheduler_unipc import FlowUniPCMultistepScheduler  # noqa: E402
from inference.pipeline.video_process import load_audio_and_encode, resample_audio_sinc, resizecrop  # noqa: E402


MODEL_CONFIG = {
    "num_layers": 40,
    "hidden_size": 5120,
    "head_dim": 128,
    "num_query_groups": 8,
    "video_in_channels": 48 * 4,
    "audio_in_channels": 64,
    "text_in_channels": 3584,
    "checkpoint_qk_layernorm_rope": False,
    "params_dtype": torch.bfloat16,
    "tread_config": {"selection_rate": 0.5, "start_layer_idx": 2, "end_layer_idx": 25},
    "mm_layers": [0, 1, 2, 3, 36, 37, 38, 39],
    "local_attn_layers": [],
    "enable_attn_gating": True,
    "activation_type": "swiglu7",
    "gelu7_layers": [0, 1, 2, 3],
    "post_norm_layers": [],
}

DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, "
    "worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, "
    "deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, "
    "walking backwards, low quality, worst quality, poor quality, noise, background noise, hiss, hum, buzz, crackle, static, "
    "compression artifacts, MP3 artifacts, digital clipping, distortion, muffled, muddy, unclear, echo, reverb, room echo, "
    "over-reverberated, hollow sound, distant, washed out, harsh, shrill, piercing, grating, tinny, thin sound, boomy, bass-heavy, "
    "flat EQ, over-compressed, abrupt cut, jarring transition, sudden silence, looping artifact, music, instrumental, sirens, alarms, "
    "crowd noise, unrelated sound effects, chaotic, disorganized, messy, cheap sound, emotionless, flat delivery, deadpan, lifeless, "
    "apathetic, robotic, mechanical, monotone, flat intonation, undynamic, boring, reading from a script, AI voice, synthetic, "
    "text-to-speech, TTS, insincere, fake emotion, exaggerated, overly dramatic, melodramatic, cheesy, cringey, hesitant, unconfident, "
    "tired, weak voice, stuttering, stammering, mumbling, slurred speech, mispronounced, bad articulation, lisp, vocal fry, creaky voice, "
    "mouth clicks, lip smacks, wet mouth sounds, heavy breathing, audible inhales, plosives, p-pops, coughing, clearing throat, sneezing, "
    "speaking too fast, rushed, speaking too slow, dragged out, unnatural pauses, awkward silence, choppy, disjointed, multiple speakers, "
    "two voices, background talking, out of tune, off-key, autotune artifacts"
)


@dataclass
class _EvalInput:
    x_t: torch.Tensor
    audio_x_t: torch.Tensor
    audio_feat_len: list[int]
    txt_feat: torch.Tensor
    txt_feat_len: list[int]


class _ConfigObject:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _ZeroSNRDDPMDiscretization:
    def __init__(
        self,
        linear_start=0.00085,
        linear_end=0.0120,
        num_timesteps=1000,
        shift_scale=1.0,
        keep_start=False,
        post_shift=False,
    ):
        if keep_start and not post_shift:
            linear_start = linear_start / (shift_scale + (1 - shift_scale) * linear_start)
        self.num_timesteps = num_timesteps
        betas = torch.linspace(linear_start**0.5, linear_end**0.5, num_timesteps, dtype=torch.float64).square().cpu().numpy()
        alphas_cumprod = np.cumprod(1.0 - betas, axis=0)
        if not post_shift:
            alphas_cumprod = alphas_cumprod / (shift_scale + (1 - shift_scale) * alphas_cumprod)
        self.alphas_cumprod = alphas_cumprod
        self.post_shift = post_shift
        self.shift_scale = shift_scale

    def get_sigmas(self, n=None, device="cpu"):
        n = self.num_timesteps if n is None else int(n)
        if n < self.num_timesteps:
            timesteps = np.linspace(self.num_timesteps - 1, 0, n, endpoint=False).astype(int)[::-1]
            alphas_cumprod = self.alphas_cumprod[timesteps]
        elif n == self.num_timesteps:
            alphas_cumprod = self.alphas_cumprod
        else:
            raise ValueError
        alphas_cumprod = torch.tensor(alphas_cumprod, dtype=torch.float32, device=device)
        alphas_cumprod_sqrt = alphas_cumprod.sqrt()
        alphas_cumprod_sqrt_0 = alphas_cumprod_sqrt[0].clone()
        alphas_cumprod_sqrt_T = alphas_cumprod_sqrt[-1].clone()
        alphas_cumprod_sqrt -= alphas_cumprod_sqrt_T
        alphas_cumprod_sqrt *= alphas_cumprod_sqrt_0 / (alphas_cumprod_sqrt_0 - alphas_cumprod_sqrt_T)
        if self.post_shift:
            alphas_cumprod_sqrt = (alphas_cumprod_sqrt**2 / (self.shift_scale + (1 - self.shift_scale) * alphas_cumprod_sqrt**2)) ** 0.5
        return torch.flip(alphas_cumprod_sqrt, dims=(0,))

    def __call__(self, n=None, do_append_zero=True, device="cpu", flip=False):
        sigmas = self.get_sigmas(n=n, device=device)
        if do_append_zero:
            sigmas = torch.cat([sigmas, sigmas.new_zeros((1,))])
        return sigmas if not flip else torch.flip(sigmas, dims=(0,))


class MagiHumanTextEncoder:
    def __init__(self, checkpoint_path: str, tokenizer_path: str, dtype: torch.dtype):
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path
        self.dtype = dtype
        self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        model_prefix = "encoder" if "quanto" in os.path.basename(checkpoint_path).lower() else "model.encoder"
        self.model = offload.fast_load_transformers_model(
            checkpoint_path,
            writable_tensors=False,
            modelClass=T5GemmaEncoderModel,
            defaultConfigPath=os.path.join(tokenizer_path, "config.json"),
            modelPrefix=model_prefix,
            configKwargs={"is_encoder_decoder": False},
            default_dtype=dtype,
        )
        self.model.eval().requires_grad_(False)

    @torch.inference_mode()
    def encode(self, prompts):
        if isinstance(prompts, str):
            prompts = [prompts]
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(self.device)
        with text_encoding_progress(self.model.encoder.layers, prompt_count=len(prompts)):
            outputs = self.model(**inputs)
        return outputs.last_hidden_state.to(self.dtype)


class MagiHumanModel:
    def __init__(
        self,
        model_filename,
        model_type,
        base_model_type,
        model_def,
        text_encoder_filename=None,
        quantizeTransformer=False,
        dtype=torch.bfloat16,
        VAE_dtype=torch.float32,
        mixed_precision_transformer=False,
        save_quantized=False,
        **kwargs,
    ):
        self.device = torch.device("cuda")
        self.dtype = dtype
        self.VAE_dtype = VAE_dtype
        self.model_def = model_def or {}
        self.model_type = model_type
        self.base_model_type = base_model_type
        self._interrupt = False

        text_encoder_folder = self.model_def.get("text_encoder_folder", "t5gemma-9b-9b-ul2")
        text_encoder_path = text_encoder_filename or fl.locate_file(
            os.path.join(text_encoder_folder, "t5gemma-9b-9b-ul2_bf16.safetensors")
        )
        tokenizer_path = fl.locate_folder(text_encoder_folder)
        self.text_encoder = MagiHumanTextEncoder(text_encoder_path, tokenizer_path, dtype)

        transformer_paths = list(model_filename) if isinstance(model_filename, (list, tuple)) else [model_filename]
        transformer_path = transformer_paths[0]
        self.transformer_config_path = fl.locate_file(self.model_def.get("config_file", "models/magi_human/configs/magi_human_distill.json"))
        model_cfg = self._build_model_config()
        self.transformer = self._load_transformer(transformer_path, model_cfg, quantizeTransformer, save_quantized, dtype)
        self.transformer2 = None
        if len(transformer_paths) > 1:
            self.transformer2 = self._load_transformer(transformer_paths[1], model_cfg, quantizeTransformer, save_quantized, dtype)

        if save_quantized:
            from wgp import save_quantized_model

            save_quantized_model(self.transformer, model_type, transformer_path, dtype, self.transformer_config_path)
            if self.transformer2 is not None:
                save_quantized_model(self.transformer2, model_type, transformer_paths[1], dtype, self.transformer_config_path, submodel_no=2)

        self.data_proxy = self._build_data_proxy("v2")
        self.sr_data_proxy = self._build_data_proxy("v1")
        self.fps = int(self.model_def.get("fps", 25))
        self.vae_stride = (4, 16, 16)
        self.latent_channels = 48
        self.audio_channels = 64
        self.text_target_length = 640
        self.text_encoder_cache = TextEncoderCache()
        self._default_negative_prompt_embeds = None
        self._default_negative_prompt_len = None
        self._turbo_vae_decode_max_pixels = 832 * 480
        self.sr_num_inference_steps = int(self.model_def.get("sr_num_inference_steps", 5))
        self.sr_cfg_number = int(self.model_def.get("sr_cfg_number", 1))
        self.sr_noise_value = int(self.model_def.get("sr_noise_value", 220))
        self.sr_video_txt_guidance_scale = float(self.model_def.get("sr_video_txt_guidance_scale", 3.5))
        self.use_cfg_trick = bool(self.model_def.get("use_cfg_trick", True))
        self.cfg_trick_start_frame = int(self.model_def.get("cfg_trick_start_frame", 13))
        self.cfg_trick_value = float(self.model_def.get("cfg_trick_value", 2.0))
        self.using_sde_flag = bool(self.model_def.get("using_sde_flag", False))
        self.sr_audio_noise_scale = float(self.model_def.get("sr_audio_noise_scale", 0.7))
        self._sr_sigmas = _ZeroSNRDDPMDiscretization()(1000, do_append_zero=False, flip=True)
        self.video_processor = VideoProcessor(vae_scale_factor=16)

        self.vae = Wan2_2_VAE(vae_pth=fl.locate_file("Wan2.2_VAE.safetensors"), dtype=VAE_dtype, device="cpu")
        self.vae.device = self.device
        self.audio_vae = SAAudioFeatureExtractor(device="cpu", model_path=fl.locate_folder("stable-audio-open-1.0"))
        self.turbo_vae = get_turbo_vaed(
            fl.locate_file("turbo_vae/TurboV3-Wan22-TinyShallow_7_7.json"),
            fl.locate_file("turbo_vae/TurboV3-Wan22-TinyShallow_7_7.safetensors"),
            device="cpu",
            weight_dtype=VAE_dtype,
        )

    def _build_model_config(self):
        cfg = dict(MODEL_CONFIG)
        cfg["num_heads_q"] = cfg["hidden_size"] // cfg["head_dim"]
        cfg["num_heads_kv"] = cfg["num_query_groups"]
        return _ConfigObject(**cfg)

    def _build_data_proxy(self, coords_style: str):
        return MagiDataProxy(
            _ConfigObject(
                t_patch_size=1,
                patch_size=2,
                frame_receptive_field=11,
                spatial_rope_interpolation="extra",
                ref_audio_offset=1000,
                text_offset=0,
                coords_style=coords_style,
            )
        )

    def _load_transformer(self, checkpoint_path: str, model_cfg, quantizeTransformer: bool, save_quantized: bool, dtype: torch.dtype):
        with init_empty_weights():
            transformer = DiTModel(model_cfg)
        offload.load_model_data(
            transformer,
            checkpoint_path,
            do_quantize=quantizeTransformer and not save_quantized,
            writable_tensors=False,
            default_dtype=dtype,
        )
        transformer.eval().requires_grad_(False)
        transformer._interrupt_check = self._interrupt_requested
        transformer.block._interrupt_check = self._interrupt_requested
        return transformer

    def _interrupt_requested(self):
        return bool(self._interrupt)

    def request_early_stop(self):
        self._interrupt = True

    def _run_transformer(self, transformer, *inputs):
        try:
            return transformer(*inputs)
        except MagiAbortRequested:
            self._interrupt = True
            return None

    def prepare_preview_payload(self, latents, preview_meta=None):
        if not torch.is_tensor(latents):
            return None
        return {"latents": latents.float()}

    def _pad_or_trim(self, tensor: torch.Tensor, target_size: int):
        current_size = tensor.shape[1]
        if current_size < target_size:
            pad = torch.zeros((tensor.shape[0], target_size - current_size, tensor.shape[2]), device=tensor.device, dtype=tensor.dtype)
            return torch.cat([tensor, pad], dim=1), current_size
        return tensor[:, :target_size], target_size

    def _pad_or_trim_prompt(self, tensor: torch.Tensor, target_size: int):
        current_size = tensor.shape[0]
        if current_size < target_size:
            pad = torch.zeros((target_size - current_size, tensor.shape[1]), device=tensor.device, dtype=tensor.dtype)
            return torch.cat([tensor, pad], dim=0), current_size
        return tensor[:target_size], target_size

    def _normalize_prompt_batch(self, prompt, batch_size: int):
        if isinstance(prompt, str):
            return [prompt] * batch_size
        prompt = list(prompt)
        if len(prompt) == 1 and batch_size > 1:
            return prompt * batch_size
        if len(prompt) != batch_size:
            raise ValueError("Prompt batch size does not match Magi Human batch size.")
        return prompt

    def _encode_prompt(self, prompt, batch_size: int):
        prompts = self._normalize_prompt_batch(prompt, batch_size)
        encode_fn = lambda prompt_batch: list(self.text_encoder.encode(prompt_batch))
        encoded_prompts = self.text_encoder_cache.encode(encode_fn, prompts, device=self.device, parallel=True)
        embeddings = []
        prompt_lens = []
        for encoded_prompt in encoded_prompts:
            padded_prompt, prompt_len = self._pad_or_trim_prompt(encoded_prompt, self.text_target_length)
            embeddings.append(padded_prompt)
            prompt_lens.append(int(prompt_len))
        stacked = torch.stack(embeddings, dim=0).to(self.device)
        unique_lens = set(prompt_lens)
        return stacked, prompt_lens[0] if len(unique_lens) == 1 else prompt_lens

    def _get_uncond_prompt(self, negative_prompt: str):
        if negative_prompt:
            embeds, prompt_len = self._encode_prompt(negative_prompt, 1)
            return embeds, [int(prompt_len)]
        if self._default_negative_prompt_embeds is None:
            embeds, prompt_len = self._encode_prompt(DEFAULT_NEGATIVE_PROMPT, 1)
            self._default_negative_prompt_embeds = embeds
            self._default_negative_prompt_len = [int(prompt_len)]
        return self._default_negative_prompt_embeds, self._default_negative_prompt_len

    def _coerce_image_batch(self, image, batch_size: int):
        if image is None:
            return None
        if isinstance(image, list):
            image = image[0] if image else None
        if image is None:
            return None
        if not torch.is_tensor(image):
            raise ValueError("Magi Human expects image tensors prepared by WanGP.")
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.dim() == 5 and image.shape[2] == 1:
            image = image.squeeze(2)
        if image.dim() != 4:
            raise ValueError(f"Unexpected Magi Human image shape: {tuple(image.shape)}")
        image = image.to(device=self.device, dtype=self.VAE_dtype)
        if image.shape[0] == 1 and batch_size > 1:
            image = image.repeat(batch_size, 1, 1, 1)
        return image

    def _coerce_last_video_frame_batch(self, video, batch_size: int):
        if video is None:
            return None
        if not torch.is_tensor(video):
            raise ValueError("Magi Human expects video tensors prepared by WanGP.")
        if video.dim() == 4:
            return self._coerce_image_batch(video[:, -1], batch_size)
        if video.dim() == 5:
            return self._coerce_image_batch(video[:, :, -1], batch_size)
        raise ValueError(f"Unexpected Magi Human video shape: {tuple(video.shape)}")

    def _encode_image_latent(self, image_batch: torch.Tensor, height: int, width: int, tile_size: int):
        latents = []
        for image in image_batch:
            image_uint8 = image.float().clamp(-1, 1).add(1).mul(127.5).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            image_pil = resizecrop(Image.fromarray(image_uint8), height, width)
            image_tensor = self.video_processor.preprocess(image_pil, height=height, width=width)[0].to(device=self.device, dtype=self.VAE_dtype)
            latents.append(self.vae.encode([image_tensor.unsqueeze(1)], tile_size=tile_size)[0])
        return torch.stack(latents, dim=0).to(device=self.device, dtype=torch.float32)

    def _resolve_vae_tile_size(self, VAE_tile_size, height: int, width: int):
        if isinstance(VAE_tile_size, dict):
            tile_size = int(VAE_tile_size.get("tile_sample_min_size", VAE_tile_size.get("tile_latent_min_size", 0)) or 0)
        elif isinstance(VAE_tile_size, (list, tuple)):
            if not VAE_tile_size:
                tile_size = 0
            elif len(VAE_tile_size) >= 2 and isinstance(VAE_tile_size[0], bool):
                tile_size = int(VAE_tile_size[1] if VAE_tile_size[0] else 0)
            else:
                tile_size = int(VAE_tile_size[-1] or 0)
        else:
            tile_size = int(VAE_tile_size or 0)

        if height * width >= 1280 * 704:
            print(
                f"[Magi][VAE] resolved tile_size={tile_size} for {width}x{height} "
                f"(input={VAE_tile_size})",
                flush=True,
            )
        return tile_size

    def _load_audio_latent(self, audio_path: str, frame_num: int, fps: int):
        seconds = max(1, math.ceil((frame_num - 1) / max(fps, 1)))
        audio_latent = load_audio_and_encode(self.audio_vae, audio_path, seconds=seconds).permute(0, 2, 1).to(torch.float32)
        return self._finalize_audio_latent(audio_latent, frame_num)

    def _finalize_audio_latent(self, audio_latent: torch.Tensor, frame_num: int):
        if audio_latent.shape[1] < frame_num:
            pad = torch.zeros((audio_latent.shape[0], frame_num - audio_latent.shape[1], audio_latent.shape[2]), device=audio_latent.device, dtype=audio_latent.dtype)
            audio_latent = torch.cat([audio_latent, pad], dim=1)
        return audio_latent[:, :frame_num].to(self.device)

    def _encode_audio_waveform_latent(self, input_waveform, input_waveform_sample_rate: int, frame_num: int, fps: int):
        if input_waveform_sample_rate is None or int(input_waveform_sample_rate) <= 0:
            raise ValueError("Magi Human requires a valid input_waveform_sample_rate when input_waveform is provided.")
        waveform = torch.from_numpy(input_waveform) if isinstance(input_waveform, np.ndarray) else input_waveform
        if not torch.is_tensor(waveform):
            raise ValueError("Magi Human expects input_waveform as a numpy array or torch tensor.")
        waveform = waveform.to(dtype=torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        elif waveform.ndim == 2:
            waveform = waveform.T
        elif waveform.ndim == 3 and waveform.shape[0] == 1:
            waveform = waveform.squeeze(0).T
        if waveform.ndim != 2:
            raise ValueError(f"Unexpected Magi Human waveform shape: {tuple(waveform.shape)}")
        target_sample_rate = 51200
        if int(input_waveform_sample_rate) != target_sample_rate:
            waveform = taF.resample(waveform, int(input_waveform_sample_rate), target_sample_rate)
        seconds = max(1, math.ceil((frame_num - 1) / max(fps, 1)))
        waveform = waveform[:, : min(waveform.shape[-1], int(seconds * target_sample_rate))]
        if waveform.shape[0] == 1:
            waveform = waveform.expand(2, -1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2]
        encode_device = next(self.audio_vae.vae_model.parameters()).device
        waveform = waveform.to(device=encode_device)
        audio_latent = self.audio_vae.encode(waveform).permute(0, 2, 1).to(torch.float32)
        return self._finalize_audio_latent(audio_latent, frame_num)

    def _decode_video(self, latents: torch.Tensor, tile_size: int):
        turbo_dtype = next(self.turbo_vae.parameters()).dtype
        sample_h = int(latents.shape[-2]) * self.vae_stride[1]
        sample_w = int(latents.shape[-1]) * self.vae_stride[2]
        output_offload = int(tile_size) > 0 or sample_h * sample_w > self._turbo_vae_decode_max_pixels
        temporal_chunk_size = 0
        # if sample_h * sample_w >= 1920 * 1088:
        #     temporal_chunk_size = 3
        # elif sample_h * sample_w >= 1280 * 704 and latents.shape[2] > self.turbo_vae.step_size:
        #     temporal_chunk_size = 5
        # print(
        #     f"[Magi][VAE] decode {sample_w}x{sample_h} latent_shape={tuple(latents[:1].shape)} "
        #     f"tile_size={tile_size} temporal_chunk_size={temporal_chunk_size} output_offload={output_offload}",
        #     flush=True,
        # )
        decoded = self.turbo_vae.decode(
            latents[:1].to(device=self.device, dtype=turbo_dtype),
            output_offload=output_offload,
            tile_size=tile_size,
            temporal_chunk_size=temporal_chunk_size,
        )
        with torch.inference_mode(False):
            return decoded[0].float().clamp(-1, 1).clone()

    def _decode_audio(self, latent_audio: torch.Tensor):
        set_phase_status("Audio VAE Decoding")
        audio_dtype = next(self.audio_vae.vae_model.parameters()).dtype
        audio_output = self.audio_vae.decode(latent_audio.squeeze(0).T.to(audio_dtype))
        audio_output = audio_output.float().squeeze(0).T.detach().cpu().numpy()
        return resample_audio_sinc(audio_output, 441 / 512)

    def _prepare_proxy_input(self, eval_input: _EvalInput, data_proxy: MagiDataProxy):
        packed = data_proxy.process_input(eval_input)
        return packed, dict(data_proxy._saved_data)

    def _process_proxy_output(self, pred: torch.Tensor, saved_state: dict, data_proxy: MagiDataProxy):
        data_proxy._saved_data = dict(saved_state)
        return data_proxy.process_output(pred)

    def _resolve_base_phase_dimensions(self, height: int, width: int):
        # return 256, 448
        phase_height, phase_width = calculate_new_dimensions(256, 448, height, width, 0, block_size=self.model_def.get("vae_block_size", 32))
        return int(phase_height), int(phase_width)

    def _run_diffusion_phase(
        self,
        transformer,
        data_proxy,
        latent_video,
        latent_audio,
        image_latent,
        prompt_embeds,
        prompt_lens,
        sampling_steps,
        shift,
        guide_scale,
        audio_cfg_scale,
        use_audio_guide,
        callback,
        pass_no,
        total_passes,
        joint_pass,
        uncond_prompt_embeds=None,
        uncond_prompt_lens=None,
        update_audio=True,
        use_sr_model=False,
        sr_cfg_scale=None,
    ):
        video_scheduler = FlowUniPCMultistepScheduler()
        audio_scheduler = FlowUniPCMultistepScheduler()
        video_scheduler.set_timesteps(int(sampling_steps), device=self.device, shift=float(shift))
        audio_scheduler.set_timesteps(int(sampling_steps), device=self.device, shift=float(shift))
        timesteps = video_scheduler.timesteps
        cfg_number = int(self.sr_cfg_number if use_sr_model else 2)
        if not use_sr_model:
            cfg_number = 1 if float(guide_scale) == 1.0 and (not update_audio or float(audio_cfg_scale) == 1.0) else 2
        sr_guidance = None
        if use_sr_model and cfg_number == 2:
            latent_frames = latent_video.shape[2]
            sr_guidance = torch.full((latent_video.shape[0], 1, latent_frames, 1, 1), float(self.sr_video_txt_guidance_scale), device=self.device, dtype=latent_video.dtype)
            if self.use_cfg_trick:
                sr_guidance[:, :, : self.cfg_trick_start_frame] = min(self.cfg_trick_value, self.sr_video_txt_guidance_scale)
            if sr_cfg_scale is not None:
                sr_guidance.mul_(float(sr_cfg_scale) / float(self.sr_video_txt_guidance_scale))
        if callback is not None:
            callback(-1, None, True, override_num_inference_steps=len(timesteps), pass_no=pass_no)
        progress_bar = tqdm(timesteps, desc=f"Phase {pass_no}/{total_passes}" if total_passes > 1 else None)
        for step_idx, t in enumerate(progress_bar):
            if self._interrupt:
                return None, None
            latent_video[:, :, :1] = image_latent[:, :, :1]
            eval_input = _EvalInput(
                x_t=latent_video,
                audio_x_t=latent_audio,
                audio_feat_len=[latent_audio.shape[1]] * latent_audio.shape[0],
                txt_feat=prompt_embeds,
                txt_feat_len=prompt_lens,
            )
            packed_inputs, packed_state = self._prepare_proxy_input(eval_input, data_proxy)
            preview_latents = None
            if cfg_number == 2:
                uncond_eval_input = _EvalInput(
                    x_t=latent_video,
                    audio_x_t=latent_audio,
                    audio_feat_len=[latent_audio.shape[1]] * latent_audio.shape[0],
                    txt_feat=uncond_prompt_embeds,
                    txt_feat_len=uncond_prompt_lens,
                )
                uncond_inputs, uncond_state = self._prepare_proxy_input(uncond_eval_input, data_proxy)
                if joint_pass:
                    pred_pair = self._run_transformer(transformer, *[[packed_inputs[i], uncond_inputs[i]] for i in range(len(packed_inputs))])
                    if pred_pair is None:
                        return None, None
                    pred_cond, pred_uncond = pred_pair
                else:
                    pred_cond = self._run_transformer(transformer, *packed_inputs)
                    if pred_cond is None:
                        return None, None
                    pred_uncond = self._run_transformer(transformer, *uncond_inputs)
                    if pred_uncond is None:
                        return None, None
                pred_video, pred_audio = self._process_proxy_output(pred_cond, packed_state, data_proxy)
                pred_video_uncond, pred_audio_uncond = self._process_proxy_output(pred_uncond, uncond_state, data_proxy)
                current_video_guidance = sr_guidance if use_sr_model else float(guide_scale if t > 500 else 2.0)
                pred_video = pred_video_uncond + current_video_guidance * (pred_video - pred_video_uncond)
                pred_audio = pred_audio_uncond + float(audio_cfg_scale) * (pred_audio - pred_audio_uncond)
                latent_video = video_scheduler.step(pred_video, t, latent_video, return_dict=False)[0]
                if update_audio and not use_audio_guide:
                    latent_audio = audio_scheduler.step(pred_audio, t, latent_audio, return_dict=False)[0]
            else:
                pred = self._run_transformer(transformer, *packed_inputs)
                if pred is None:
                    return None, None
                pred_video, pred_audio = self._process_proxy_output(pred, packed_state, data_proxy)
                if use_sr_model:
                    latent_video = video_scheduler.step(pred_video, t, latent_video, return_dict=False)[0]
                else:
                    if callback is not None and getattr(callback, "tiny_vae", False):
                        # DDIM uses full-precision sigmas, not the integer model timestep.
                        preview_latents = latent_video - video_scheduler.sigmas[step_idx] * pred_video
                    latent_video = video_scheduler.step_ddim(pred_video, step_idx, latent_video)
                if update_audio and not use_audio_guide:
                    latent_audio = audio_scheduler.step_ddim(pred_audio, step_idx, latent_audio)
            if callback is not None:
                if getattr(callback, "tiny_vae", False) and (cfg_number == 2 or use_sr_model):
                    preview_latents = video_scheduler.model_outputs[-1]
                callback(step_idx, (preview_latents if preview_latents is not None else latent_video)[0].detach(), pass_no=pass_no)
            del preview_latents
        latent_video[:, :, :1] = image_latent[:, :, :1]
        return latent_video, latent_audio

    @torch.inference_mode()
    @generation_progress
    def generate(
        self,
        seed=None,
        input_prompt="",
        alt_prompt="",
        n_prompt="",
        sampling_steps=8,
        input_ref_images=None,
        input_frames=None,
        input_frames2=None,
        input_masks=None,
        input_masks2=None,
        input_video=None,
        image_start=None,
        image_end=None,
        frame_num=101,
        batch_size=1,
        height=256,
        width=448,
        guide_scale=1.0,
        guide2_scale=1.0,
        guide3_scale=1.0,
        switch_threshold=0.0,
        switch2_threshold=0.0,
        guide_phases=1,
        model_switch_phase=0,
        embedded_guidance_scale=0.0,
        shift=5.0,
        sample_solver="unipc",
        denoising_strength=1.0,
        masking_strength=1.0,
        callback=None,
        VAE_tile_size=None,
        joint_pass=True,
        audio_cfg_scale=1.0,
        prefix_video=None,
        prefix_frames_count=0,
        input_video_strength=1.0,
        input_waveform=None,
        input_waveform_sample_rate=None,
        audio_guide=None,
        audio_guide2=None,
        audio_prompt_type="",
        fps=None,
        offloadobj=None,
        set_header_text=None,
        loras_slists=None,
        set_progress_status=None,
        **kwargs,
    ):
        self._interrupt = False
        if seed is None or seed == -1:
            seed = torch.seed() % (2**32 - 1)
        if fps is None or fps <= 0:
            fps = self.fps
        if batch_size < 1:
            raise ValueError("Magi Human batch_size must be positive.")
        if batch_size != 1:
            raise ValueError("Magi Human currently supports batch_size=1 only.")

        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        prompt = alt_prompt or input_prompt
        prompt_embeds, prompt_len = self._encode_prompt(prompt, batch_size)
        if isinstance(prompt_len, int):
            prompt_lens = [prompt_len] * batch_size
        else:
            prompt_lens = [int(prompt_len)] * batch_size
        if self.base_model_type == "magi_human_distill":
            guide_scale = 1.0
            audio_cfg_scale = 1.0
        need_cfg = float(guide_scale) != 1.0 or float(audio_cfg_scale) != 1.0 or (self.transformer2 is not None and self.sr_cfg_number == 2)
        if need_cfg:
            uncond_prompt_embeds, uncond_prompt_lens = self._get_uncond_prompt(n_prompt)
        else:
            uncond_prompt_embeds, uncond_prompt_lens = None, None
        image_batch = input_video[:, -1].unsqueeze(0)
        # image_batch = self._coerce_last_video_frame_batch(input_video, batch_size)
        # if image_batch is None:
        #     image_batch = self._coerce_last_video_frame_batch(prefix_video, batch_size)
        # if image_batch is None:
        #     source_image = image_start
        #     if source_image is None and input_ref_images is not None:
        #         source_image = input_ref_images[0] if isinstance(input_ref_images, list) and input_ref_images else input_ref_images
        #     image_batch = self._coerce_image_batch(source_image, batch_size)
        # if image_batch is None:
        #     raise ValueError("Magi Human requires a start image.")

        total_passes = 2 if self.transformer2 is not None else 1
        phase1_height, phase1_width = self._resolve_base_phase_dimensions(height, width) if self.transformer2 is not None else (height, width)
        tile_size = self._resolve_vae_tile_size(VAE_tile_size, phase1_height, phase1_width)
        image_latent = self._encode_image_latent(image_batch, phase1_height, phase1_width, tile_size)

        latent_frames = (frame_num - 1) // self.vae_stride[0] + 1
        latent_h = phase1_height // self.vae_stride[1]
        latent_w = phase1_width // self.vae_stride[2]
        latent_video = torch.randn(
            (batch_size, self.latent_channels, latent_frames, latent_h, latent_w),
            generator=generator,
            device=self.device,
            dtype=torch.float32,
        )
        latent_video[:, :, :1] = image_latent[:, :, :1]

        use_audio_guide = bool("A" in (audio_prompt_type or "") and (input_waveform is not None or audio_guide))
        if "A" in (audio_prompt_type or "") and input_waveform is not None:
            latent_audio = self._encode_audio_waveform_latent(input_waveform, input_waveform_sample_rate, frame_num, fps)
        elif use_audio_guide:
            latent_audio = self._load_audio_latent(audio_guide, frame_num, fps)
        else:
            latent_audio = torch.randn((batch_size, frame_num, self.audio_channels), generator=generator, device=self.device, dtype=torch.float32)

        if loras_slists is not None:
            update_loras_slists(self.transformer, loras_slists, int(sampling_steps), phase_switch_step=int(sampling_steps), phase_switch_step2=int(sampling_steps))
        latent_video, latent_audio = self._run_diffusion_phase(
            self.transformer,
            self.data_proxy,
            latent_video,
            latent_audio,
            image_latent,
            prompt_embeds,
            prompt_lens,
            sampling_steps,
            shift,
            guide_scale,
            audio_cfg_scale,
            use_audio_guide,
            callback,
            1,
            total_passes,
            joint_pass,
            uncond_prompt_embeds=uncond_prompt_embeds,
            uncond_prompt_lens=uncond_prompt_lens,
            update_audio=True,
            use_sr_model=False,
        )
        if latent_video is None or latent_audio is None:
            return None

        if self._interrupt:
            return None

        final_video_latent = latent_video
        final_audio_latent = latent_audio
        final_tile_size = tile_size
        if self.transformer2 is not None:
            final_tile_size = self._resolve_vae_tile_size(VAE_tile_size, height, width)
            sr_image_latent = self._encode_image_latent(image_batch, height, width, final_tile_size)
            sr_latent_h = height // self.vae_stride[1]
            sr_latent_w = width // self.vae_stride[2]
            sr_video_latent = F.interpolate(latent_video, size=(latent_frames, sr_latent_h, sr_latent_w), mode="trilinear", align_corners=True)
            if self.sr_noise_value > 0:
                sigma = self._sr_sigmas.to(sr_video_latent.device)[self.sr_noise_value]
                sr_video_latent = sr_video_latent * sigma + torch.randn_like(sr_video_latent) * (1 - sigma**2).sqrt()
            sr_audio_latent = torch.randn_like(latent_audio) * self.sr_audio_noise_scale + latent_audio * (1 - self.sr_audio_noise_scale)
            if loras_slists is not None:
                update_loras_slists(self.transformer2, loras_slists, self.sr_num_inference_steps, phase_switch_step=0, phase_switch_step2=self.sr_num_inference_steps)
            final_video_latent, _ = self._run_diffusion_phase(
                self.transformer2,
                self.sr_data_proxy,
                sr_video_latent,
                sr_audio_latent,
                sr_image_latent,
                prompt_embeds,
                prompt_lens,
                self.sr_num_inference_steps,
                shift,
                1.0,
                1.0,
                True,
                callback,
                2,
                total_passes,
                False,
                uncond_prompt_embeds=uncond_prompt_embeds,
                uncond_prompt_lens=uncond_prompt_lens,
                update_audio=False,
                use_sr_model=True,
            )
            if final_video_latent is None:
                return None

        video = self._decode_video(final_video_latent, final_tile_size)
        audio = self._decode_audio(final_audio_latent[:1])
        return {"x": video, "audio": audio, "audio_sampling_rate": 44100}
