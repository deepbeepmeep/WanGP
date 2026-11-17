import os
from dataclasses import replace
from typing import Optional

import torch
import torchaudio
from accelerate import init_empty_weights
from mmgp import offload
from mmgp import offload as mmgp_offload
from tqdm import tqdm

from shared.utils.text_encoder_cache import TextEncoderCache

from .ltx2 import _VAEContainer, _load_config_from_checkpoint, _make_sd_postprocess, _make_vae_postprocess
from .ltx_core.components.diffusion_steps import EulerDiffusionStep
from .ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from .ltx_core.components.noisers import GaussianNoiser
from .ltx_core.components.patchifiers import AudioPatchifier
from .ltx_core.conditioning import AudioConditionByReferenceLatent
from .ltx_core.guidance.perturbations import BatchedPerturbationConfig, Perturbation, PerturbationConfig, PerturbationType
from .ltx_core.model.audio_vae import (
    VOCODER_COMFY_KEYS_FILTER,
    AudioDecoderConfigurator,
    AudioEncoderConfigurator,
    AudioProcessor,
    VocoderConfigurator,
    decode_audio,
)
from .ltx_core.model.transformer import LTXAudioOnlyModelConfigurator, LTXV_MODEL_COMFY_RENAMING_MAP, X0Model
from .ltx_core.text_encoders.gemma import (
    TEXT_EMBEDDING_PROJECTION_KEY_OPS,
    TEXT_EMBEDDINGS_CONNECTOR_KEY_OPS,
    GemmaTextEmbeddingsConnectorModelConfigurator,
    build_gemma_text_encoder,
    encode_text,
    postprocess_text_embeddings,
    resolve_text_connectors,
)
from .ltx_core.text_encoders.gemma.feature_extractor import GemmaFeaturesExtractorProjLinear
from .ltx_core.tools import AudioLatentTools
from .ltx_core.types import AudioLatentShape, VideoPixelShape
from .ltx_pipelines.utils.helpers import (
    _prepare_conditioning_context,
    modality_from_latent_state,
    post_process_latent,
    state_with_conditionings,
)


def ltx_audio_tts_model_device_dtype(module: torch.nn.Module, default_device: torch.device, default_dtype: torch.dtype):
    param = next(module.parameters(), None)
    if param is None:
        return default_device, default_dtype
    return param.device, param.dtype


def ltx_audio_tts_duration_to_frames(duration: float, fps: float) -> int:
    return ((int(duration * fps) + 7) // 8) * 8 + 1


class LTXAudioTTSPipelineBase:
    def __init__(
        self,
        model_weights_path: str,
        gemma_path: str,
        *,
        audio_vae_path: str | None = None,
        vocoder_path: str | None = None,
        text_projection_path: str | None = None,
        text_connector_path: str | None = None,
        audio_components_path: str | None = None,
        config_path: str | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or torch.bfloat16
        self._interrupt = False
        self._early_stop = False
        self.audio_patchifier = AudioPatchifier(patch_size=1)
        self.text_encoder_cache = TextEncoderCache()
        self._init_models(
            model_weights_path=model_weights_path,
            audio_vae_path=audio_vae_path,
            vocoder_path=vocoder_path,
            text_projection_path=text_projection_path,
            text_connector_path=text_connector_path,
            audio_components_path=audio_components_path,
            gemma_path=gemma_path,
            config_path=config_path,
        )

    def _component_path(self, specific_path: str | None, audio_components_path: str | None) -> str:
        path = specific_path or audio_components_path
        if not path:
            raise ValueError("Missing LTX audio TTS component checkpoint path.")
        return path

    def _load_component(
        self,
        model: torch.nn.Module,
        path: str,
        sd_ops=None,
        *,
        postprocess=None,
        ignore_unused_weights: bool = False,
        ignore_missing_keys: bool = False,
    ) -> torch.nn.Module:
        if postprocess is None and sd_ops is not None:
            postprocess = _make_sd_postprocess(sd_ops)
        mmgp_offload.load_model_data(
            model,
            path,
            postprocess_sd=postprocess,
            default_dtype=self.dtype,
            writable_tensors=False,
            ignore_missing_keys=ignore_missing_keys,
            ignore_unused_weights=ignore_unused_weights,
        )
        model.eval().requires_grad_(False)
        return model

    def _init_models(
        self,
        *,
        model_weights_path: str,
        audio_vae_path: str | None,
        vocoder_path: str | None,
        text_projection_path: str | None,
        text_connector_path: str | None,
        audio_components_path: str | None,
        gemma_path: str,
        config_path: str | None,
    ) -> None:
        base_config = _load_config_from_checkpoint(model_weights_path, fallback_config_path=config_path)
        if not base_config:
            raise ValueError("Missing LTX audio TTS transformer config.")
        component_config_path = audio_components_path or audio_vae_path or vocoder_path or text_projection_path or text_connector_path
        pipeline_config = _load_config_from_checkpoint(component_config_path, fallback_config_path=config_path) or base_config

        with init_empty_weights():
            velocity_model = LTXAudioOnlyModelConfigurator.from_config(base_config)
        velocity_model = self._load_component(velocity_model, model_weights_path, LTXV_MODEL_COMFY_RENAMING_MAP, ignore_unused_weights=True)
        self.model = X0Model(velocity_model)
        self.model.eval().requires_grad_(False)

        audio_vae_source = self._component_path(audio_vae_path, audio_components_path)
        shared_audio_vae = audio_vae_path is None and audio_components_path is not None
        with init_empty_weights():
            audio_encoder = AudioEncoderConfigurator.from_config(pipeline_config)
            audio_decoder = AudioDecoderConfigurator.from_config(pipeline_config)
            if hasattr(audio_encoder, "mid") and hasattr(audio_encoder.mid, "attn_1"):
                audio_encoder.mid.attn_1 = torch.nn.Identity()
            audio_vae = _VAEContainer(audio_encoder, audio_decoder)
        audio_vae = self._load_component(audio_vae, audio_vae_source, postprocess=_make_vae_postprocess("audio_vae."), ignore_unused_weights=shared_audio_vae)
        self.audio_encoder = audio_vae.encoder
        self.audio_decoder = audio_vae.decoder

        vocoder_source = self._component_path(vocoder_path, audio_components_path)
        with init_empty_weights():
            vocoder = VocoderConfigurator.from_config(pipeline_config)
        self.vocoder = self._load_component(vocoder, vocoder_source, VOCODER_COMFY_KEYS_FILTER, ignore_unused_weights=vocoder_path is None and audio_components_path is not None)

        ddconfig = pipeline_config.get("audio_vae", {}).get("model", {}).get("params", {}).get("ddconfig", {})
        if "mel_bins" in ddconfig:
            self.audio_encoder.mel_bins = int(ddconfig["mel_bins"])

        text_projection_source = self._component_path(text_projection_path, audio_components_path)
        with init_empty_weights():
            text_embedding_projection = GemmaFeaturesExtractorProjLinear.from_config(pipeline_config)
        self.text_embedding_projection = self._load_component(text_embedding_projection, text_projection_source, TEXT_EMBEDDING_PROJECTION_KEY_OPS, ignore_unused_weights=text_projection_path is None and audio_components_path is not None)

        text_connector_source = self._component_path(text_connector_path, audio_components_path)
        with init_empty_weights():
            text_embeddings_connector = GemmaTextEmbeddingsConnectorModelConfigurator.from_config(pipeline_config)
        self.text_embeddings_connector = self._load_component(text_embeddings_connector, text_connector_source, TEXT_EMBEDDINGS_CONNECTOR_KEY_OPS, ignore_unused_weights=text_connector_path is None and audio_components_path is not None)
        self.video_embeddings_connector = self.text_embeddings_connector.video_embeddings_connector
        self.audio_embeddings_connector = self.text_embeddings_connector.audio_embeddings_connector

        self.text_encoder = build_gemma_text_encoder(gemma_path, default_dtype=self.dtype)
        self.text_encoder.eval().requires_grad_(False)
        self._text_connectors = {
            "feature_extractor_linear": self.text_embedding_projection,
            "embeddings_connector": self.video_embeddings_connector,
            "audio_embeddings_connector": self.audio_embeddings_connector,
        }

    def get_trans_lora(self):
        return self.model, None

    def get_loras_transformer(self, get_model_recursive_prop, **kwargs):
        return [], []

    def abort(self):
        self._interrupt = True

    def _early_stop_requested(self) -> bool:
        return bool(self._early_stop)

    def request_early_stop(self) -> None:
        self._early_stop = True

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        feature_extractor, video_connector, audio_connector = resolve_text_connectors(self.text_encoder, self._text_connectors)
        encode_fn = lambda prompts: postprocess_text_embeddings(
            encode_text(self.text_encoder, prompts=prompts),
            feature_extractor,
            video_connector,
            audio_connector,
        )
        (_, audio_context) = self.text_encoder_cache.encode(encode_fn, [prompt], device=self.device, parallel=True)[0]
        return audio_context.to(device=self.device, dtype=self.dtype)

    def _encode_prompts(self, prompts: list[str]) -> list[torch.Tensor]:
        feature_extractor, video_connector, audio_connector = resolve_text_connectors(self.text_encoder, self._text_connectors)
        encode_fn = lambda batch: postprocess_text_embeddings(
            encode_text(self.text_encoder, prompts=batch),
            feature_extractor,
            video_connector,
            audio_connector,
        )
        contexts = self.text_encoder_cache.encode(encode_fn, prompts, device=self.device, parallel=True)
        return [audio_context.to(device=self.device, dtype=self.dtype) for _, audio_context in contexts]

    def _waveform_from_input(self, input_waveform, input_waveform_sample_rate, audio_guide: str | None):
        if input_waveform is not None:
            waveform = torch.as_tensor(input_waveform, dtype=torch.float32)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            elif waveform.ndim == 2:
                waveform = waveform.T
            return waveform, int(input_waveform_sample_rate)
        if not audio_guide:
            return None, 0
        waveform, sample_rate = torchaudio.load(os.fspath(audio_guide))
        return waveform.float(), int(sample_rate)

    def _coerce_reference_channels(self, waveform: torch.Tensor) -> torch.Tensor:
        target_channels = int(getattr(self.audio_encoder, "in_channels", waveform.shape[1]))
        if waveform.shape[1] == target_channels:
            return waveform
        if waveform.shape[1] == 1 and target_channels > 1:
            return waveform.repeat(1, target_channels, 1)
        if target_channels == 1:
            return waveform.mean(dim=1, keepdim=True)
        waveform = waveform[:, :target_channels, :]
        if waveform.shape[1] < target_channels:
            pad_shape = (waveform.shape[0], target_channels - waveform.shape[1], waveform.shape[2])
            waveform = torch.cat([waveform, torch.zeros(pad_shape, dtype=waveform.dtype)], dim=1)
        return waveform

    def _encode_reference_waveform(self, waveform: torch.Tensor, sample_rate: int, max_seconds: float | None = None, normalize_peak: float | None = None):
        waveform = waveform.unsqueeze(0)
        waveform = self._coerce_reference_channels(waveform)
        if max_seconds is not None:
            max_samples = int(round(float(sample_rate) * float(max_seconds)))
            waveform = waveform[:, :, :max_samples]
        waveform = waveform.to(dtype=torch.float32)
        if normalize_peak is not None:
            peak = waveform.abs().max()
            if peak > 0:
                waveform = waveform * (float(normalize_peak) / peak)
        audio_processor = AudioProcessor(
            sample_rate=self.audio_encoder.sample_rate,
            mel_bins=self.audio_encoder.mel_bins,
            mel_hop_length=self.audio_encoder.mel_hop_length,
            n_fft=self.audio_encoder.n_fft,
            device=self.device,
        ).to(waveform.device)
        mel = audio_processor.waveform_to_mel(waveform, sample_rate)
        audio_device, audio_dtype = ltx_audio_tts_model_device_dtype(self.audio_encoder, self.device, self.dtype)
        mel = mel.to(device=audio_device, dtype=audio_dtype)
        with torch.inference_mode():
            ref_latent = self.audio_encoder(mel)
        return ref_latent.to(device=self.device, dtype=self.dtype)

    @staticmethod
    def _reference_tail_waveform(waveform: torch.Tensor, sample_rate: int, tail_seconds: float) -> torch.Tensor:
        tail_samples = int(round(float(tail_seconds) * int(sample_rate)))
        return waveform[:, -tail_samples:] if waveform.shape[-1] > tail_samples else waveform

    def _encode_reference(self, input_waveform, input_waveform_sample_rate, audio_guide: str | None, *, tail_seconds: float, max_seconds: float | None = None):
        waveform, sample_rate = self._waveform_from_input(input_waveform, input_waveform_sample_rate, audio_guide)
        if waveform is None or sample_rate <= 0:
            return None
        return self._encode_reference_waveform(self._reference_tail_waveform(waveform, sample_rate, tail_seconds), sample_rate, max_seconds=max_seconds)

    def _encode_tail_reference(self, audio: torch.Tensor, sample_rate: int, tail_seconds: float):
        channels_first = audio.detach().cpu().float()
        if channels_first.ndim == 3:
            channels_first = channels_first.squeeze(0)
        if channels_first.ndim == 1:
            channels_first = channels_first.unsqueeze(0)
        return self._encode_reference_waveform(channels_first[:, -int(float(tail_seconds) * sample_rate):], sample_rate)

    def _callback_start(self, callback, total_steps: int, status_extra: str = "") -> None:
        if callback is not None:
            callback(-1, None, True, override_num_inference_steps=total_steps, pass_no=0, denoising_extra=status_extra)

    def _callback_step(self, callback, step_idx: int, status_extra: str = "") -> None:
        if callback is not None:
            callback(step_idx, None, False, pass_no=0, denoising_extra=status_extra)

    @staticmethod
    def _custom_float(custom_settings, key: str, default: float) -> float:
        if not isinstance(custom_settings, dict):
            return default
        raw_value = custom_settings.get(key, default)
        if raw_value is None or raw_value == "":
            return default
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return default
        return value if value > 0 else default

    @staticmethod
    def _custom_nonnegative_float(custom_settings, key: str, default: float) -> float:
        if not isinstance(custom_settings, dict):
            return default
        raw_value = custom_settings.get(key, default)
        if raw_value is None or raw_value == "":
            return default
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            return default
        return value if value >= 0 else default

    @staticmethod
    def _custom_int(custom_settings, key: str, default: int) -> int:
        if not isinstance(custom_settings, dict):
            return default
        raw_value = custom_settings.get(key, default)
        if raw_value is None or raw_value == "":
            return default
        try:
            value = int(raw_value)
        except (TypeError, ValueError):
            return default
        return value if value > 0 else default

    def _build_audio_state(self, duration: float, fps: float, sigmas: torch.Tensor, seed: int, ref_latent=None, reference_conditioner=AudioConditionByReferenceLatent):
        generator = torch.Generator(device=self.device).manual_seed(int(seed))
        noiser = GaussianNoiser(generator=generator)
        pixel_shape = VideoPixelShape(batch=1, frames=ltx_audio_tts_duration_to_frames(duration, fps), width=64, height=64, fps=fps)
        audio_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
        audio_tools = AudioLatentTools(self.audio_patchifier, audio_shape)
        audio_state = audio_tools.create_initial_state(self.device, self.dtype)
        conditionings = [reference_conditioner(ref_latent)] if ref_latent is not None else []
        audio_state = state_with_conditionings(audio_state, conditionings, audio_tools)
        return noiser(audio_state), audio_tools

    def _audio_stg_perturbations(self, batch_size: int, stg_blocks: list[int] | None):
        return BatchedPerturbationConfig([PerturbationConfig([Perturbation(type=PerturbationType.SKIP_AUDIO_SELF_ATTN, blocks=stg_blocks or None)]) for _ in range(batch_size)])

    @torch.inference_mode()
    def _generate_audio_euler(
        self,
        audio_context: torch.Tensor,
        sigmas: torch.Tensor,
        audio_state,
        audio_tools: AudioLatentTools,
        *,
        audio_context_n: torch.Tensor | None = None,
        cfg_scale: float = 1.0,
        stg_scale: float = 0.0,
        stg_blocks: list[int] | None = None,
        rescale_scale: float = 0.0,
        callback=None,
        status_extra: str = "",
        set_progress_status=None,
    ):
        stepper = EulerDiffusionStep()
        sigmas = sigmas.to(device=self.device, dtype=torch.float32)
        total_steps = len(sigmas) - 1
        if set_progress_status is not None:
            set_progress_status(f"Denoising | {status_extra}" if status_extra else "Denoising")
        self._callback_start(callback, total_steps, status_extra)

        velocity_model = getattr(self.model, "velocity_model", self.model)
        velocity_model.interrupt_check = lambda: bool(self._interrupt or self._early_stop_requested())
        prepared_audio_context = _prepare_conditioning_context(self.model, audio_state, audio_context, sigmas, is_audio=True)
        prepared_audio_context_n = None
        use_cfg = audio_context_n is not None and abs(float(cfg_scale) - 1.0) > 1e-6
        use_stg = abs(float(stg_scale)) > 1e-6
        guider = MultiModalGuider(MultiModalGuiderParams(cfg_scale=float(cfg_scale), stg_scale=float(stg_scale), stg_blocks=stg_blocks or [], rescale_scale=float(rescale_scale), modality_scale=1.0), negative_context=audio_context_n)
        try:
            if use_cfg:
                prepared_audio_context_n = _prepare_conditioning_context(self.model, audio_state, audio_context_n, sigmas, is_audio=True)
            for step_idx, _ in enumerate(tqdm(sigmas[:-1])):
                if self._interrupt or self._early_stop_requested():
                    return None
                offload.set_step_no_for_lora(self.model, step_idx)
                sigma = sigmas[step_idx]
                pos_audio = modality_from_latent_state(audio_state, prepared_audio_context, sigma, step_index=step_idx, sigma_schedule=sigmas)

                if use_cfg or use_stg:
                    audio_list = [pos_audio]
                    perturbations = [None]
                    neg_index = stg_index = None
                    if use_cfg:
                        neg_index = len(audio_list)
                        audio_list.append(modality_from_latent_state(audio_state, prepared_audio_context_n, sigma, step_index=step_idx, sigma_schedule=sigmas))
                        perturbations.append(None)
                    if use_stg:
                        stg_index = len(audio_list)
                        audio_list.append(modality_from_latent_state(audio_state, prepared_audio_context, sigma, step_index=step_idx, sigma_schedule=sigmas))
                        perturbations.append(self._audio_stg_perturbations(audio_state.latent.shape[0], stg_blocks))
                    _, denoised_audio_list = self.model(video=None, audio=audio_list, perturbations=perturbations)
                    if denoised_audio_list is None:
                        return None
                    pos_denoised_audio = denoised_audio_list[0]
                    if pos_denoised_audio is None:
                        return None
                    neg_denoised_audio = denoised_audio_list[neg_index] if neg_index is not None else pos_denoised_audio
                    stg_denoised_audio = denoised_audio_list[stg_index] if stg_index is not None else pos_denoised_audio
                    if neg_denoised_audio is None or stg_denoised_audio is None:
                        return None
                    denoised_audio = guider.calculate(pos_denoised_audio, neg_denoised_audio, stg_denoised_audio, pos_denoised_audio)
                else:
                    _, denoised_audio = self.model(video=None, audio=pos_audio, perturbations=None)
                    if denoised_audio is None:
                        return None

                denoised_audio = post_process_latent(denoised_audio, audio_state.denoise_mask, audio_state.clean_latent)
                if float(sigmas[step_idx + 1].item()) == 0.0:
                    audio_state = replace(audio_state, latent=denoised_audio)
                else:
                    audio_state = replace(audio_state, latent=stepper.step(audio_state.latent, denoised_audio, sigmas, step_idx))
                self._callback_step(callback, step_idx, status_extra)
        finally:
            velocity_model.interrupt_check = None

        if self._interrupt or self._early_stop_requested():
            return None
        audio_state = audio_tools.clear_conditioning(audio_state)
        audio_state = audio_tools.unpatchify(audio_state)
        return audio_state

    def _decode_audio_state(self, audio_state, set_progress_status=None, status_extra: str = ""):
        if set_progress_status is not None:
            set_progress_status(f"VAE Decoding | {status_extra}" if status_extra else "VAE Decoding")
        return decode_audio(audio_state.latent, self.audio_decoder, self.vocoder).detach().cpu().float()
