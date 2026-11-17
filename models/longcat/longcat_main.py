from shared.utils.phase_progress import text_encoding_progress, generation_progress
import json
import math
import os
from functools import wraps

import numpy as np
import torch
from accelerate import init_empty_weights
from mmgp import offload, runtime
from tqdm import tqdm
import librosa
import pyloudnorm as pyln
import scipy.signal as ss
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, Wav2Vec2FeatureExtractor, WhisperModel

from shared.utils import files_locator as fl
from ..wan.modules.t5 import T5EncoderModel
from .modules.longcat_video_dit import LongCatVideoTransformer3DModel
from .modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel
from ..wan.modules.vae import WanVAE
from .modules.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from .audio_process.wav2vec2 import Wav2Vec2ModelWrapper
from shared.utils.text_encoder_cache import TextEncoderCache
from shared import attention_kit
from shared.utils.loras_mutipliers import update_loras_slists, expand_slist


def release_execution_caches(method):
    @wraps(method)
    def run(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        finally:
            self._clear_runtime_caches()
            if self._interrupt:
                self._audio_chunks.clear()
                self._reference_latent = None
    return run


def _load_json_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.pop("_class_name", None)
    cfg.pop("_diffusers_version", None)
    cfg.pop("architectures", None)
    cfg.pop("model_max_length", None)
    return cfg


def optimized_scale(positive_flat, negative_flat):
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
    squared_norm = torch.sum(negative_flat ** 2, dim=1, keepdim=True) + 1e-8
    return dot_product / squared_norm


LONGCAT_AVATAR_TYPES = {"longcat_avatar", "longcat_avatar_v1_5"}


class LongCatModel:
    def __init__(
        self,
        checkpoint_dir,
        model_filename=None,
        model_type=None,
        model_def=None,
        base_model_type=None,
        text_encoder_filename=None,
        quantizeTransformer=False,
        save_quantized=False,
        dtype=torch.bfloat16,
        VAE_dtype=torch.float32,
        mixed_precision_transformer=False,
        **kwargs,
    ):
        self.device = torch.device("cuda")
        self.dtype = dtype
        self.VAE_dtype = VAE_dtype
        self.model_def = model_def or {}
        self.base_model_type = base_model_type
        self.is_avatar = base_model_type in LONGCAT_AVATAR_TYPES
        self.is_avatar_v1_5 = base_model_type == "longcat_avatar_v1_5"
        self.is_avatar_v1_6 = self.model_def.get("avatar_version") == "1.6"
        self.audio_encoder_name = None
        self.sparse_attention_enabled = bool(self.model_def.get("sparse_attention", False))
        self._interrupt = False
        self._reference_image = None
        self._reference_latent = None
        self._kv_cache = None
        self._audio_chunks = {}

        text_encoder_path = text_encoder_filename
        if text_encoder_path is None:
            raise ValueError("Missing resolved LongCat text encoder weights path.")
        text_encoder_folder = self.model_def.get("text_encoder_folder")
        if text_encoder_folder:
            tokenizer_path = fl.locate_folder(text_encoder_folder)
        else:
            tokenizer_path = os.path.dirname(text_encoder_path)
        self.text_encoder = T5EncoderModel(
            text_len=512,
            dtype=dtype,
            device=torch.device("cpu"),
            checkpoint_path=text_encoder_path,
            tokenizer_path=tokenizer_path,
        )
        self.text_encoder_cache = TextEncoderCache()

        transformer_config_path = self.model_def.get("transformer_config")
        if not transformer_config_path:
            transformer_config_path = (
                "models/longcat/configs/longcat_avatar.json"
                if self.is_avatar
                else "models/longcat/configs/longcat_video.json"
            )
        transformer_cfg = _load_json_config(transformer_config_path)
        if self.sparse_attention_enabled:
            transformer_cfg["enable_bsa"] = True
            sparse_params = self.model_def.get("sparse_attention_params")
            if isinstance(sparse_params, dict) and sparse_params:
                bsa_params = dict(transformer_cfg.get("bsa_params") or {})
                bsa_params.update(sparse_params)
                transformer_cfg["bsa_params"] = bsa_params
        transformer_cls = (
            LongCatVideoAvatarTransformer3DModel
            if self.is_avatar
            else LongCatVideoTransformer3DModel
        )
        with init_empty_weights(include_buffers=True):
            transformer = transformer_cls(**transformer_cfg)
            from .modules.prefix_attention import split_map
            transformer.split_linear_modules_map = split_map(transformer.config.hidden_size)
        model_path = model_filename[0] if isinstance(model_filename, (list, tuple)) else model_filename
        if model_path is None:
            raise ValueError("Missing LongCat transformer weights path.")
        def prepare_projections(state_dict, quantization_map):
            # Let the shared handlers choose whether to split checkpoint tensors
            # or loaded modules. Both layouts occur in released LongCat files.
            from mmgp.quant_router import split_fused_weights
            state_dict, quantization_map = split_fused_weights(
                state_dict, quantization_map, transformer.split_linear_modules_map,
                default_dtype=torch.bfloat16)
            with init_empty_weights(include_buffers=True):
                for index, block in enumerate(transformer.blocks):
                    if any(key.startswith(f"blocks.{index}.attn.q.") for key in state_dict):
                        offload.split_linear_modules(block, transformer.split_linear_modules_map)
            return state_dict, quantization_map

        offload.load_model_data(transformer, model_path, writable_tensors=False, default_dtype=torch.bfloat16,
                               postprocess_sd=prepare_projections)
        offload.split_linear_modules(transformer, transformer.split_linear_modules_map)
        transformer._model_dtype = dtype
        transformer.final_layer._lock_dtype = torch.float32  # holds no weights itself: keeps MMGP from rounding its FP32 AdaLN input to BF16
        transformer.eval().requires_grad_(False)
        self.transformer = transformer
        if save_quantized:
            from wgp import save_quantized_model

            save_quantized_model(transformer, model_type, model_path, dtype, transformer_config_path)

        vae_weights = self.model_def.get("vae_URL")
        if vae_weights:
            vae_weights = fl.locate_file(vae_weights)
        else:
            for candidate in ["Wan2.1_VAE_bf16.safetensors", "Wan2.1_VAE.safetensors", "longcat_vae_bf16.safetensors"]:
                vae_weights = fl.locate_file(candidate, error_if_none=False)
                if vae_weights:
                    break
        if not vae_weights:
            raise FileNotFoundError("Unable to locate a compatible VAE weights file for LongCat.")
        # the Wan 2.1 VAE of the Wan models: latents normalized by the class, exact tiles, decoded frames written to RAM as bytes
        self.vae = WanVAE(vae_pth=vae_weights, dtype=VAE_dtype, device="cpu")
        self.vae.device = self.device  # (its normalization tensors; the weights stay in RAM, managed by MMGP)
        self.vae.model.requires_grad_(False)

        scheduler_cfg = _load_json_config(self.model_def.get("scheduler_config", "models/longcat/configs/longcat_scheduler.json"))
        self.scheduler = FlowMatchEulerDiscreteScheduler(**scheduler_cfg)
        self.num_timesteps = 1000
        self.num_distill_sample_steps = int(self.model_def.get("num_distill_sample_steps", 8 if self.is_avatar_v1_5 else 50))

        if self.is_avatar:
            if self.is_avatar_v1_5:
                whisper_folder_name = self.model_def.get("audio_encoder_folder", "whisper-large-v3")
                whisper_folder = fl.locate_folder(whisper_folder_name)
                whisper_model_path = fl.locate_file(os.path.join(whisper_folder_name, self.model_def.get("audio_encoder_file", "model.safetensors")))
                whisper_config_path = fl.locate_file(os.path.join(whisper_folder_name, "config.json"))
                fl.locate_file(os.path.join(whisper_folder_name, "generation_config.json"))
                fl.locate_file(os.path.join(whisper_folder_name, "preprocessor_config.json"))
                self.audio_encoder_name = "whisper-large-v3"
                self.audio_encoder = offload.fast_load_transformers_model(
                    whisper_model_path,
                    modelClass=WhisperModel,
                    defaultConfigPath=whisper_config_path,
                    modelPrefix="model",
                    writable_tensors=False,
                    default_dtype=dtype,
                    ignore_unused_weights=True,
                    pre_load_callback=lambda model: delattr(model, "decoder"),
                )
                self.audio_encoder._model_dtype = dtype
                self.audio_encoder.eval().requires_grad_(False)
                self.audio_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_folder, local_files_only=True)
            else:
                wav2vec_folder = fl.locate_folder("chinese-wav2vec2-base")
                self.audio_encoder_name = "wav2vec2"
                self.audio_encoder = Wav2Vec2ModelWrapper(wav2vec_folder)
                self.audio_encoder.eval().requires_grad_(False)
                if hasattr(self.audio_encoder, "feature_extractor"):
                    self.audio_encoder.feature_extractor._freeze_parameters()
                self.audio_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_folder, local_files_only=True)
        else:
            self.audio_encoder = None
            self.audio_feature_extractor = None

        self.vae_scale_factor_temporal, self.vae_scale_factor_spatial = 4, 8
        self.transformer._interrupt_check = lambda: self._interrupt

    def get_loras_transformer(self, get_model_recursive_prop, **kwargs):
        if not self.is_avatar_v1_5:
            return [], []
        sample_solver = kwargs.get("sample_solver") or self.model_def.get("sample_solver", "distill")
        lora_url = self.model_def.get("distill_lora_URL") if sample_solver == "distill" else None
        return ([lora_url], ["1.0"]) if lora_url else ([], [])

    def _clear_runtime_caches(self):
        if self._kv_cache is not None:
            self._kv_cache.clear()
            self._kv_cache = None
        clear = getattr(self.transformer, "clear_runtime_caches", None)
        if clear is not None:
            clear()

    def prepare_preview_payload(self, latents, preview_meta=None):
        if not torch.is_tensor(latents):
            return None
        return {"latents": latents.float()}

    def _validate_sparse_attention(self, latents):
        if not self.sparse_attention_enabled:
            return
        bsa_params = getattr(self.transformer.config, "bsa_params", None) or {}
        chunk_shape_q = bsa_params.get("chunk_3d_shape_q")
        chunk_shape_k = bsa_params.get("chunk_3d_shape_k")
        chunk_shape = chunk_shape_q or chunk_shape_k
        if not chunk_shape or latents.dim() != 5:
            self.transformer.disable_bsa()
            self.sparse_attention_enabled = False
            tqdm.write("Sparse attention disabled: missing BSA parameters.")
            return
        attn_mode = offload.shared_state.get("_attention", "auto")
        require_grid_divisible = False
        if attn_mode == "flash":
            require_grid_divisible = True
        elif attn_mode == "auto":
            try:
                from shared.attention import flash_attn_bsa_3d
            except Exception:
                flash_attn_bsa_3d = None
            require_grid_divisible = flash_attn_bsa_3d is not None

        patch_t, patch_h, patch_w = self.transformer.config.patch_size
        n_t = latents.shape[2] // patch_t
        n_h = latents.shape[3] // patch_h
        n_w = latents.shape[4] // patch_w
        cp_split_hw = getattr(self.transformer.config, "cp_split_hw", None)
        if cp_split_hw:
            if n_h % cp_split_hw[0] != 0 or n_w % cp_split_hw[1] != 0:
                self.transformer.disable_bsa()
                self.sparse_attention_enabled = False
                tqdm.write("Sparse attention disabled: cp_split_hw does not divide token grid.")
                return
        if require_grid_divisible:
            shape_q = chunk_shape_q or chunk_shape
            shape_k = chunk_shape_k or chunk_shape
            if (
                n_t % shape_q[0] != 0
                or n_h % shape_q[1] != 0
                or n_w % shape_q[2] != 0
                or n_t % shape_k[0] != 0
                or n_h % shape_k[1] != 0
                or n_w % shape_k[2] != 0
            ):
                self.transformer.disable_bsa()
                self.sparse_attention_enabled = False
                tqdm.write("Sparse attention disabled: flash BSA needs token grid divisible by chunk shape.")

    def _encode_prompt(
        self,
        prompt,
        negative_prompt,
        num_videos_per_prompt=1,
        max_length=512,
        device=None,
        dtype=None,
    ):
        device = device or self.device
        dtype = dtype or self.dtype
        def encode_fn(prompts):
            ids, mask = self.text_encoder.tokenizer(
                prompts,
                return_mask=True,
                add_special_tokens=True,
            )
            ids = ids.to(device)
            mask = mask.to(device)
            with text_encoding_progress(self.text_encoder.model.blocks, prompt_count=len(prompts)):
                prompt_embeds = self.text_encoder.model(ids, mask).to(dtype)
            return list(zip(prompt_embeds, mask))
        prompt_list = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt_list)
        neg_list = [] if negative_prompt is None else [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
        if len(neg_list) == 1 and batch_size > 1:
            neg_list = neg_list * batch_size
        contexts = self.text_encoder_cache.encode(encode_fn, prompt_list + neg_list, device=device)
        prompt_contexts, negative_contexts = contexts[:batch_size], contexts[batch_size:]
        prompt_embeds = torch.stack([ctx[0] for ctx in prompt_contexts], dim=0)
        mask = torch.stack([ctx[1] for ctx in prompt_contexts], dim=0)
        seq_len = prompt_embeds.shape[1]
        prompt_embeds = prompt_embeds.unsqueeze(1)
        if num_videos_per_prompt > 1:
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1, 1)
            prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, 1, seq_len, -1)
            mask = mask.repeat(num_videos_per_prompt, 1)

        neg_embeds = None
        neg_mask = None
        if negative_prompt is not None:
            neg_list = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            if len(neg_list) == 1 and batch_size > 1:
                neg_list = neg_list * batch_size
            neg_contexts = negative_contexts
            neg_embeds = torch.stack([ctx[0] for ctx in neg_contexts], dim=0)
            neg_mask = torch.stack([ctx[1] for ctx in neg_contexts], dim=0)
            neg_embeds = neg_embeds.unsqueeze(1)
            if num_videos_per_prompt > 1:
                neg_embeds = neg_embeds.repeat(1, num_videos_per_prompt, 1, 1)
                neg_embeds = neg_embeds.view(batch_size * num_videos_per_prompt, 1, seq_len, -1)
                neg_mask = neg_mask.repeat(num_videos_per_prompt, 1)

        return prompt_embeds, mask, neg_embeds, neg_mask

    def _prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        num_frames,
        dtype,
        device,
        generator,
        latents=None,
        image=None,
        video=None,
        num_cond_frames=0,
        tile_size=0,
    ):
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        if latents is None:
            shape = (
                batch_size,
                num_channels_latents,
                num_latent_frames,
                int(height) // self.vae_scale_factor_spatial,
                int(width) // self.vae_scale_factor_spatial,
            )
            latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        num_cond_latents = 0
        if image is not None or video is not None:
            inputs = [image[i].unsqueeze(1) if image is not None else video[i][:, -num_cond_frames:] for i in range(batch_size)]
            cond_latents = torch.stack(self.vae.encode(inputs, tile_size)).to(dtype)
            num_cond_latents = 1 + (num_cond_frames - 1) // self.vae_scale_factor_temporal
            latents[:, :, :num_cond_latents] = cond_latents[:, :, :num_cond_latents]
        return latents, num_cond_latents

    def _loudness_norm(self, audio_array, sr=16000, lufs=-23, threshold=100):
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio_array)
        if not np.isfinite(loudness) or loudness > threshold:
            return audio_array
        return pyln.normalize.loudness(audio_array, loudness, lufs)

    def _add_noise_floor(self, audio_array, noise_level=0.0001):
        noise = np.random.normal(0, noise_level, size=audio_array.shape)
        return audio_array + noise

    def _smooth_transients(self, audio_array, sr=16000):
        b, a = ss.butter(3, 3000 / (sr / 2))
        return ss.lfilter(b, a, audio_array)

    @staticmethod
    def _interpolate_audio_state(audio_state, target_len):
        if audio_state.shape[0] == target_len:
            return audio_state
        audio_state = audio_state.transpose(0, 1).unsqueeze(0).float()
        audio_state = F.interpolate(audio_state, size=target_len, mode="linear", align_corners=False)
        return audio_state.squeeze(0).transpose(0, 1)

    @torch.no_grad()
    def _get_audio_embedding_wav2vec(self, speech_array, fps=32, device="cpu", sample_rate=16000):
        audio_duration = len(speech_array) / sample_rate
        video_length = audio_duration * fps

        speech_array = self._loudness_norm(speech_array, sample_rate)
        speech_array = self._add_noise_floor(speech_array)
        speech_array = self._smooth_transients(speech_array, sample_rate)

        audio_feature = np.squeeze(
            self.audio_feature_extractor(speech_array, sampling_rate=sample_rate).input_values
        )
        audio_feature = np.nan_to_num(audio_feature, nan=0.0, posinf=0.0, neginf=0.0)
        audio_feature = torch.from_numpy(audio_feature).float().to(device=device)
        audio_feature = audio_feature.unsqueeze(0)

        embeddings = self.audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)
        audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
        audio_emb = audio_emb.permute(1, 0, 2).contiguous()
        audio_emb = torch.nan_to_num(audio_emb, nan=0.0, posinf=0.0, neginf=0.0)
        return audio_emb

    @torch.no_grad()
    def _get_audio_embedding_whisper(self, speech_array, fps=25, device=None, sample_rate=16000):
        device = device or self.device
        audio_duration = len(speech_array) / sample_rate
        video_length = max(int(round(audio_duration * fps)), 1)
        encoder_length = max(int(math.ceil(audio_duration * 50)), 1)
        speech_array = self._loudness_norm(speech_array, sample_rate)
        speech_array = self._add_noise_floor(speech_array)
        speech_array = self._smooth_transients(speech_array, sample_rate)

        layer_groups = [(0, 8), (8, 16), (16, 24), (24, 32), (32, 33)]
        audio_chunks = [[] for _ in layer_groups]
        mel_chunk = 750 * 640
        for start in range(0, len(speech_array), mel_chunk):
            chunk = speech_array[start : start + mel_chunk]
            features = self.audio_feature_extractor(
                [chunk],
                return_tensors="pt",
                return_attention_mask=True,
                sampling_rate=sample_rate,
            )
            input_features = features.input_features.to(device=device, dtype=self.dtype)
            attention_mask = getattr(features, "attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device=device)
            input_features = self.audio_encoder._mask_input_features(input_features, attention_mask=attention_mask)
            encoder_dtype = getattr(self.audio_encoder.encoder, "dtype", self.dtype)
            outputs = self.audio_encoder.encoder(
                input_features.to(encoder_dtype),
                head_mask=None,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )
            chunk_states = outputs.hidden_states
            for group_idx, (layer_start, layer_end) in enumerate(layer_groups):
                layer_state = torch.stack(chunk_states[layer_start:layer_end], dim=0).mean(dim=0).squeeze(0)
                audio_chunks[group_idx].append(layer_state.detach().to("cpu"))
            del outputs, chunk_states, input_features, attention_mask, features

        audio_layers = []
        for chunks in audio_chunks:
            layer_state = torch.cat(chunks, dim=0)[:encoder_length]
            layer_state = self._interpolate_audio_state(layer_state, video_length).to(dtype=self.dtype)
            audio_layers.append(layer_state)
        audio_emb = torch.stack(audio_layers, dim=1).contiguous()
        return torch.nan_to_num(audio_emb, nan=0.0, posinf=0.0, neginf=0.0)

    @torch.no_grad()
    def _get_audio_embedding(self, speech_array, fps=32, device="cpu", sample_rate=16000):
        if self.is_avatar_v1_5:
            return self._get_audio_embedding_whisper(speech_array, fps=fps, device=device, sample_rate=sample_rate)
        return self._get_audio_embedding_wav2vec(speech_array, fps=fps, device=device, sample_rate=sample_rate)

    def _build_audio_windows(self, audio_path, frame_num, fps, window_start_frame_no, audio_stride):
        if self.is_avatar_v1_6:
            from .audio_process.whisper_windows import audio_windows
            return audio_windows(self, audio_path, frame_num, fps, window_start_frame_no)
        speech_array, sr = librosa.load(audio_path, sr=16000)
        target_len = int((window_start_frame_no + frame_num) / fps * sr)
        if len(speech_array) < target_len:
            pad = target_len - len(speech_array)
            speech_array = np.pad(speech_array, (0, pad), mode="constant")

        audio_device = self.device if self.is_avatar_v1_5 else "cpu"
        full_audio_emb = self._get_audio_embedding(speech_array, fps=fps * audio_stride, device=audio_device, sample_rate=sr)
        if torch.isnan(full_audio_emb).any():
            raise ValueError("Audio embedding contains NaNs.")

        audio_start_idx = window_start_frame_no * audio_stride
        audio_end_idx = audio_start_idx + audio_stride * frame_num
        window = self.transformer.audio_window if hasattr(self.transformer, "audio_window") else 5
        offsets = torch.arange(window, device=full_audio_emb.device) - window // 2
        centers = torch.arange(audio_start_idx, audio_end_idx, audio_stride, device=full_audio_emb.device).unsqueeze(1) + offsets.unsqueeze(0)
        centers = torch.clamp(centers, min=0, max=full_audio_emb.shape[0] - 1)
        audio_emb = full_audio_emb[centers][None, ...].to("cpu")
        del full_audio_emb
        return audio_emb

    def _build_ref_target_masks(self, height, width, speakers_bboxes=None):
        if not speakers_bboxes:
            speakers_bboxes = {"person1": [5, 10, 45, 90], "person2": [55, 10, 95, 90]}
        human_masks = []
        background_mask = torch.zeros([height, width], device="cpu")
        for _, person_bbox in speakers_bboxes.items():
            y_min, x_min, y_max, x_max = person_bbox
            x_min, y_min, x_max, y_max = max(x_min, 5), max(y_min, 5), min(x_max, 95), min(y_max, 95)
            x_min, y_min, x_max, y_max = (
                int(height * x_min / 100),
                int(width * y_min / 100),
                int(height * x_max / 100),
                int(width * y_max / 100),
            )
            human_mask = torch.zeros([height, width], device="cpu")
            human_mask[int(x_min) : int(x_max), int(y_min) : int(y_max)] = 1
            background_mask += human_mask
            human_masks.append(human_mask)
        background_mask = torch.where(background_mask > 0, torch.tensor(0, device="cpu"), torch.tensor(1, device="cpu"))
        human_masks.append(background_mask)
        return torch.stack(human_masks, dim=0)

    def get_timesteps_sigmas(self, sampling_steps, use_distill=False):
        if use_distill:
            distill_indices = torch.arange(1, self.num_distill_sample_steps + 1, dtype=torch.float32, device=self.device)
            distill_indices = (distill_indices * (self.num_timesteps // self.num_distill_sample_steps)).round().long()
            if self.is_avatar_v1_5:
                distill_indices = self.num_timesteps - distill_indices
                sigmas = torch.flip(torch.linspace(0, 1, self.num_timesteps, device=self.device), [0])
                sigmas = torch.flip(sigmas[distill_indices], [0]).float()
                if sampling_steps != self.num_distill_sample_steps:
                    inference_indices = np.linspace(0, self.num_distill_sample_steps, num=sampling_steps, endpoint=False)
                    sigmas = sigmas[np.floor(inference_indices).astype(np.int64)]
            else:
                inference_indices = np.linspace(0, self.num_distill_sample_steps, num=sampling_steps, endpoint=False)
                inference_indices = np.floor(inference_indices).astype(np.int64)
                sigmas = torch.flip(distill_indices, [0])[inference_indices].float() / self.num_timesteps
        else:
            sigmas = torch.linspace(1, 0.001, sampling_steps, dtype=torch.float32, device=self.device)
        return sigmas.to(dtype=torch.float32, device="cpu")

    @torch.inference_mode()
    @generation_progress
    @release_execution_caches
    def generate(
        self,
        seed=None,
        input_prompt="",
        n_prompt="",
        sampling_steps=50,
        input_ref_images=None,
        input_video=None,
        frame_num=93,
        batch_size=1,
        height=480,
        width=832,
        guide_scale=4.0,
        audio_cfg_scale=None,
        joint_pass=False,
        VAE_tile_size=None,
        prefix_frames_count=0,
        callback=None,
        cfg_star_switch=False,
        cfg_zero_step=-1,
        audio_guide=None,
        audio_guide2=None,
        fps=None,
        window_start_frame_no=0,
        sample_solver=None,
        reference_image_enabled=None,
        ref_img_index=None,
        mask_frame_range=None,
        overlapped_latents=None,
        return_latent_slice=None,
        speakers_bboxes=None,
        window_no=None,
        offloadobj=None,
        set_progress_status=None,
        custom_settings=None,
        loras_slists=None,
        **kwargs,
    ):
        if self._interrupt:
            return None

        if self.is_avatar_v1_6 and sampling_steps != 8:
            raise ValueError("LongCat Avatar 1.6 requires eight distilled denoising steps.")

        if window_no in (None, 1):
            self._reference_image = None
            self._reference_latent = None
            self._audio_chunks.clear()

        if seed is None or seed == -1:
            seed = torch.seed() % (2**32 - 1)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        if fps is None or fps == 0:
            fps = self.model_def.get("fps", 15 if not self.is_avatar else 16)

        if frame_num % self.vae_scale_factor_temporal != 1:
            frame_num = frame_num // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1
        frame_num = max(frame_num, 1)

        sample_solver = self.model_def.get("sample_solver", "auto") if sample_solver is None else sample_solver
        if sample_solver in (None, ""):
            sample_solver = "default"

        prompt_embeds, prompt_mask, neg_embeds, neg_mask = self._encode_prompt(
            input_prompt,
            n_prompt if n_prompt is not None else "",
            device=self.device,
            dtype=self.dtype,
        )

        any_guidance = guide_scale is not None and guide_scale > 1
        if self.is_avatar:
            if audio_cfg_scale is None:
                audio_cfg_scale = 1.0
            any_guidance = any_guidance or audio_cfg_scale > 1

        if reference_image_enabled is None:
            reference_image_enabled = self.model_def.get("reference_image_enabled", True)
        reference_image_enabled = self.is_avatar and bool(reference_image_enabled)
        reference_features_enabled = reference_image_enabled

        ref_img_index = self.model_def.get("ref_img_index", 10) if ref_img_index is None else ref_img_index
        mask_frame_range = self.model_def.get("mask_frame_range", 3) if mask_frame_range is None else mask_frame_range
        if not reference_features_enabled:
            ref_img_index = None
            mask_frame_range = None

        ref_image = None
        if reference_image_enabled:
            if input_ref_images is not None:
                ref_list = input_ref_images if isinstance(input_ref_images, list) else [input_ref_images]
                if len(ref_list) > 0:
                    ref_image = ref_list[0]
            if window_no in (None, 1):
                if ref_image is not None:
                    self._reference_image = (
                        ref_image.detach().to("cpu") if torch.is_tensor(ref_image) else ref_image
                    )
                else:
                    self._reference_image = None
            if ref_image is None and self._reference_image is not None:
                ref_image = self._reference_image

        cond_video = None
        num_cond_frames = 0
        if input_video is not None:
            cond_video = input_video
            num_cond_frames = max(int(prefix_frames_count or 0), 0)

        VAE_tile_size = VAE_tile_size or 0

        if cond_video is not None:
            cond_video = cond_video.to(device=self.device, dtype=self.VAE_dtype)
            if cond_video.dim() == 4:
                cond_video = cond_video.unsqueeze(0)
            cond_video_frames = cond_video.shape[2]
            if num_cond_frames <= 0:
                cond_video = None
                num_cond_frames = 0
            else:
                num_cond_frames = min(num_cond_frames, cond_video_frames)

        if sample_solver not in ("auto", "default", "enhance_hf", "distill"):
            raise ValueError(f"Unsupported scheduler '{sample_solver}' for LongCat.")
        if self.model_def.get("distill_only", False) and sample_solver != "distill":
            raise ValueError("This LongCat Avatar model requires the distilled scheduler.")
        use_distill = sample_solver == "distill"
        enhance_hf = sample_solver == "enhance_hf"
        if sample_solver == "auto":
            enhance_hf = cond_video is not None and num_cond_frames > 1
        if use_distill and enhance_hf:
            raise ValueError("distill and enhance_hf schedules cannot both be enabled.")

        image_cond = None
        ref_latent = None
        num_ref_latents = 0
        if reference_image_enabled and ref_image is not None:
            if not torch.is_tensor(ref_image):
                ref_image = torch.from_numpy(np.array(ref_image)).float().div_(127.5).sub_(1.).movedim(-1, 0)
            ref_image = ref_image.to(device=self.device, dtype=self.VAE_dtype)
            if ref_image.dim() == 3:
                ref_image = ref_image.unsqueeze(0)
            if ref_image.dim() == 5 and ref_image.shape[2] == 1:
                ref_image = ref_image.squeeze(2)
            if ref_image.dim() != 4:
                raise ValueError("reference image must be CHW or BCHW for LongCat.")
            if ref_image.shape[0] == 1 and batch_size > 1:
                ref_image = ref_image.repeat(batch_size, 1, 1, 1)
            elif ref_image.shape[0] != batch_size:
                raise ValueError("reference image batch size does not match prompts.")
            if cond_video is None:
                image_cond = ref_image
            else:
                ref_latent = (self._reference_latent.to(self.device) if self._reference_latent is not None else
                              torch.stack(self.vae.encode([image.unsqueeze(1) for image in ref_image], VAE_tile_size)))
                num_ref_latents = 1

        if torch.is_tensor(overlapped_latents):
            if overlapped_latents.dim() == 4:
                overlapped_latents = overlapped_latents.unsqueeze(0)
            if overlapped_latents.dim() != 5:
                overlapped_latents = None
        else:
            overlapped_latents = None

        cond_image_frames = 1 if image_cond is not None else num_cond_frames
        expected_num_cond_latents = (
            1 + (cond_image_frames - 1) // self.vae_scale_factor_temporal if cond_image_frames > 0 else 0
        )
        use_overlap_latents = (
            overlapped_latents is not None and expected_num_cond_latents > 0 and image_cond is None
        )
        if use_overlap_latents:
            lat_h = int(height) // self.vae_scale_factor_spatial
            lat_w = int(width) // self.vae_scale_factor_spatial
            if (
                overlapped_latents.shape[1] != self.transformer.config.in_channels
                or overlapped_latents.shape[3] != lat_h
                or overlapped_latents.shape[4] != lat_w
            ):
                use_overlap_latents = False

        if use_overlap_latents:
            num_latent_frames = (frame_num - 1) // self.vae_scale_factor_temporal + 1
            shape = (
                batch_size,
                self.transformer.config.in_channels,
                num_latent_frames,
                lat_h,
                lat_w,
            )
            latents = torch.randn(shape, generator=generator, device=self.device, dtype=torch.float32)

            overlap_latents = overlapped_latents.to(device=self.device, dtype=torch.float32)
            if overlap_latents.shape[0] == 1 and batch_size > 1:
                overlap_latents = overlap_latents.repeat(batch_size, 1, 1, 1, 1)
            if overlap_latents.shape[2] > expected_num_cond_latents:
                overlap_latents = overlap_latents[:, :, -expected_num_cond_latents:]

            cond_latents = None
            if cond_video is not None and overlap_latents.shape[2] < expected_num_cond_latents:
                cond_latents = torch.stack(self.vae.encode([cond_video[i][:, -cond_image_frames:] for i in range(batch_size)], VAE_tile_size))
                overlap_len = min(overlap_latents.shape[2], cond_latents.shape[2])
                if overlap_len > 0:
                    cond_latents[:, :, -overlap_len:] = overlap_latents[:, :, -overlap_len:]
            else:
                cond_latents = overlap_latents

            num_cond_latents = min(cond_latents.shape[2], num_latent_frames) if cond_latents is not None else 0
            if num_cond_latents > 0:
                latents[:, :, :num_cond_latents] = cond_latents[:, :, -num_cond_latents:]
        else:
            latents, num_cond_latents = self._prepare_latents(
                batch_size=batch_size,
                num_channels_latents=self.transformer.config.in_channels,
                height=height,
                width=width,
                num_frames=frame_num,
                dtype=torch.float32,
                device=self.device,
                generator=generator,
                latents=None,
                image=image_cond,
                video=None if image_cond is not None else cond_video,
                num_cond_frames=cond_image_frames,
                tile_size=VAE_tile_size,
            )
        if reference_image_enabled and ref_latent is None and self.is_avatar and num_cond_latents > 1:
            ref_latent = (self._reference_latent.to(self.device) if self._reference_latent is not None else
                          latents[:, :, :1].clone())
            num_ref_latents = 1
        if ref_latent is not None:
            num_cond_latents += num_ref_latents
            latents = torch.cat([ref_latent, latents], dim=2)
        self._validate_sparse_attention(latents)

        sigmas = self.get_timesteps_sigmas(sampling_steps, use_distill=use_distill)
        self.scheduler.set_timesteps(sampling_steps, sigmas=sigmas, device=self.device)
        timesteps = self.scheduler.timesteps
        if enhance_hf:
            num_tail_uniform_steps = max(3, min(15, int(len(timesteps) * 0.2)))
            tail_uniform_start = float(timesteps.max()) * 0.5
            tail_uniform_end = 0
            timesteps_uniform_tail = list(
                np.linspace(
                    tail_uniform_start,
                    tail_uniform_end,
                    num_tail_uniform_steps,
                    dtype=np.float32,
                    endpoint=(tail_uniform_end != 0),
                )
            )
            timesteps_uniform_tail = [
                torch.tensor(t, device=self.device, dtype=torch.float32).unsqueeze(0)
                for t in timesteps_uniform_tail
            ]
            filtered_timesteps = [
                timestep.unsqueeze(0).to(self.device) for timestep in timesteps if timestep > tail_uniform_start
            ]
            timesteps = torch.cat(filtered_timesteps + timesteps_uniform_tail)
            self.scheduler.timesteps = timesteps
            self.scheduler.sigmas = torch.cat(
                [timesteps / self.num_timesteps, torch.zeros(1, device=timesteps.device)]
            )

        audio_emb = None
        ref_target_masks = None
        if self.is_avatar:
            if audio_guide is None:
                raise ValueError("Audio guide is required for LongCat Avatar.")
            audio_stride = int(self.model_def.get("audio_stride", 1 if self.is_avatar_v1_5 else 2))
            audio_emb = self._build_audio_windows(
                audio_guide, frame_num, fps, window_start_frame_no, audio_stride
            )
            if audio_guide2 is not None or self.model_def.get("multi_speakers_only", False):
                if audio_guide2 is None:
                    raise ValueError("Second audio guide is required for LongCat Avatar Multi.")
                audio_emb2 = self._build_audio_windows(
                    audio_guide2, frame_num, fps, window_start_frame_no, audio_stride
                )
                audio_emb = torch.cat([audio_emb, audio_emb2], dim=0)
                ref_target_masks = self._build_ref_target_masks(height, width, speakers_bboxes)
            if ref_target_masks is not None:
                ref_target_masks = ref_target_masks.to(self.device)
            audio_emb = audio_emb.to(self.device, dtype=self.dtype)

        latents = latents.to(self.device, dtype=self.dtype)
        prompt_embeds = prompt_embeds.to(self.device)
        prompt_mask = prompt_mask.to(self.device)
        if neg_embeds is None:
            neg_embeds = prompt_embeds
            neg_mask = prompt_mask
        else:
            neg_embeds = neg_embeds.to(self.device)
            neg_mask = neg_mask.to(self.device)

        ref_kwargs = {}
        if self.is_avatar and num_ref_latents > 0:
            ref_kwargs = {
                "num_ref_latents": num_ref_latents,
                "ref_img_index": ref_img_index,
                "mask_frame_range": mask_frame_range,
            }

        cache_mode = (custom_settings or {}).get("longcat_kv_cache", "Disabled")
        if cache_mode != "Disabled" and num_cond_latents:
            self._kv_cache = attention_kit.PrefixCache(
                self.transformer.blocks, quantize=True,
                native=ref_target_masks is None and not (mask_frame_range or 0),
                ram=cache_mode in ("INT8_RAM", "Q4_RAM"), name="LongCat KV Cache",
                bits=4 if cache_mode in ("Q4", "Q4_RAM") else 8)
        schedules = []
        if loras_slists is not None:
            update_loras_slists(self.transformer, loras_slists, len(timesteps))
            schedules = [expand_slist(loras_slists, n, len(timesteps), len(timesteps), len(timesteps))
                         for n in range(len(loras_slists["phase1"]))]
        last_adapter_scales = None

        def denoise(**inputs):
            if not self.is_avatar:
                inputs.pop("audio_embs", None)
                inputs.pop("ref_target_masks", None)
            cache = self._kv_cache
            if cache is None:
                return self.transformer(**inputs)
            hidden = inputs["hidden_states"]
            steps = inputs["timestep"]
            joint = isinstance(hidden, list)
            inputs["hidden_states"] = [value[:, :, num_cond_latents:] for value in hidden] if joint else hidden[:, :, num_cond_latents:]
            inputs["timestep"] = [value[:, num_cond_latents:] for value in steps] if joint else steps[:, num_cond_latents:]
            inputs["kv_cache_dict"] = cache
            predictions = self.transformer(**inputs)
            if predictions is None:
                return None
            def restore(value, original):
                if value is None:
                    return None
                result = value.new_zeros(original.shape)
                result[:, :, num_cond_latents:] = value
                return result
            return [restore(value, original) for value, original in zip(predictions, hidden)] if joint else restore(predictions, hidden)

        if callback is not None:
            callback(-1, None, True, override_num_inference_steps=len(timesteps))

        with tqdm(total=len(timesteps), desc="Denoising") as progress_bar:
            for i, t in enumerate(timesteps):
                if self._interrupt:
                    return None
                offload.set_step_no_for_lora(self.transformer, i)
                adapter_scales = tuple(value[i] if isinstance(value, list) else value for value in schedules)
                if self._kv_cache is not None:
                    if adapter_scales != last_adapter_scales:
                        self._kv_cache.clear(keep_ring=True)
                    if self._kv_cache.empty():
                        self.transformer(
                            hidden_states=latents[:, :, :num_cond_latents],
                            timestep=latents.new_zeros(latents.shape[0], num_cond_latents),
                            encoder_hidden_states=prompt_embeds, encoder_attention_mask=prompt_mask,
                            num_cond_latents=num_cond_latents, return_kv=True, skip_crs_attn=True,
                            kv_cache_dict=self._kv_cache, **ref_kwargs)
                        if self._interrupt:
                            return None
                    last_adapter_scales = adapter_scales
                def _aborted(outputs):
                    if outputs is None:
                        return True
                    if isinstance(outputs, (list, tuple)):
                        return any(item is None for item in outputs)
                    return outputs is None

                timestep = t.expand(latents.shape[0]).to(self.dtype)
                if num_cond_latents > 0:
                    timestep = timestep[:, None].expand(-1, latents.shape[2]).clone()
                    timestep[:, :num_cond_latents] = 0

                predictions = 3 if self.is_avatar and audio_emb is not None else 2
                boost = joint_pass and any_guidance and runtime.joint_pass_useful(self.transformer, math.prod(latents.shape[2:]) // math.prod(self.transformer.patch_size), self.transformer.blocks[0].hidden_size, predictions)
                if self.is_avatar and audio_emb is not None and any_guidance:
                    audio_cond = audio_emb.to(self.device, dtype=self.dtype)
                    audio_uncond = torch.zeros_like(audio_cond)
                    x_list = [latents, latents, latents]
                    ctx_list = [prompt_embeds, neg_embeds, neg_embeds]
                    mask_list = [prompt_mask, neg_mask, neg_mask]
                    audio_list = [audio_cond, audio_cond, audio_uncond]
                    ref_list = [ref_target_masks, ref_target_masks, ref_target_masks]
                    if boost:
                        outputs = denoise(
                            hidden_states=x_list,
                            timestep=[timestep] * len(x_list),
                            encoder_hidden_states=ctx_list,
                            encoder_attention_mask=mask_list,
                            num_cond_latents=[num_cond_latents] * len(x_list),
                            audio_embs=audio_list,
                            ref_target_masks=ref_list,
                            **ref_kwargs,
                        )
                        if _aborted(outputs):
                            return None
                    else:
                        outputs = []
                        for x_i, ctx_i, mask_i, audio_i, ref_i in zip(
                            x_list, ctx_list, mask_list, audio_list, ref_list
                        ):
                            output = denoise(
                                hidden_states=x_i,
                                timestep=timestep,
                                encoder_hidden_states=ctx_i,
                                encoder_attention_mask=mask_i,
                                num_cond_latents=num_cond_latents,
                                audio_embs=audio_i,
                                ref_target_masks=ref_i,
                                **ref_kwargs,
                            )
                            if _aborted(output):
                                return None
                            outputs.append(output)
                    noise_pred_cond, noise_pred_uncond_text, noise_pred_uncond = outputs
                    noise_pred = (
                        noise_pred_uncond
                        + guide_scale * (noise_pred_cond - noise_pred_uncond_text)
                        + audio_cfg_scale * (noise_pred_uncond_text - noise_pred_uncond)
                    )
                elif any_guidance:
                    x_list = [latents, latents]
                    ctx_list = [prompt_embeds, neg_embeds]
                    mask_list = [prompt_mask, neg_mask]
                    if boost:
                        outputs = denoise(
                            hidden_states=x_list,
                            timestep=[timestep] * len(x_list),
                            encoder_hidden_states=ctx_list,
                            encoder_attention_mask=mask_list,
                            num_cond_latents=[num_cond_latents] * len(x_list),
                            **ref_kwargs,
                        )
                        if _aborted(outputs):
                            return None
                    else:
                        outputs = []
                        for x_i, ctx_i, mask_i in zip(x_list, ctx_list, mask_list):
                            output = denoise(
                                hidden_states=x_i,
                                timestep=timestep,
                                encoder_hidden_states=ctx_i,
                                encoder_attention_mask=mask_i,
                                num_cond_latents=num_cond_latents,
                                **ref_kwargs,
                            )
                            if _aborted(output):
                                return None
                            outputs.append(output)
                    noise_pred_cond, noise_pred_uncond = outputs
                    if cfg_star_switch:
                        positive_flat = noise_pred_cond.view(latents.shape[0], -1)
                        negative_flat = noise_pred_uncond.view(latents.shape[0], -1)
                        st_star = optimized_scale(positive_flat, negative_flat).view(latents.shape[0], 1, 1, 1)
                        if cfg_zero_step >= 0 and i <= cfg_zero_step:
                            noise_pred = noise_pred_cond * 0.0
                        else:
                            noise_pred_uncond = noise_pred_uncond * st_star
                            noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
                    else:
                        noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)
                else:
                    noise_pred = denoise(
                        hidden_states=latents,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        encoder_attention_mask=prompt_mask,
                        num_cond_latents=num_cond_latents,
                        audio_embs=audio_emb if self.is_avatar else None,
                        ref_target_masks=ref_target_masks if self.is_avatar else None,
                        **ref_kwargs,
                    )
                    if _aborted(noise_pred):
                        return None

                noise_pred = -noise_pred

                if num_cond_latents > 0:
                    latents[:, :, num_cond_latents:] = self.scheduler.step(
                        noise_pred[:, :, num_cond_latents:],
                        t,
                        latents[:, :, num_cond_latents:],
                        return_dict=False,
                    )[0]
                else:
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback is not None:
                    callback(i, latents[0, :, num_ref_latents:])
                progress_bar.update()

        if num_ref_latents > 0:
            latents = latents[:, :, num_ref_latents:]
            num_cond_latents -= num_ref_latents

        if reference_image_enabled and self._reference_latent is None:
            # Keep an explicit reference across windows, even when a different start
            # image or source video supplies the opening frame. With reference-only
            # input, the first clean frame already contains its encoded image.
            anchor = ref_latent if ref_latent is not None else latents[:, :, :1]
            self._reference_latent = anchor.detach().to("cpu").clone()
            del anchor

        latent_slice = None
        if return_latent_slice is not None:
            latent_slice = latents[:, :, return_latent_slice].detach().to("cpu")

        self._clear_runtime_caches()
        video = self.vae.decode_to_cpu_uint8([latents[0]], VAE_tile_size)[0]  # bytes in RAM, frame chunk by frame chunk

        if latent_slice is not None:
            return {"x": video, "latent_slice": latent_slice}
        return video
