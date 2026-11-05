from __future__ import annotations

from shared.utils.phase_progress import vae_decoding_progress, text_encoding_progress, generation_progress
import math
import os

import torch
from accelerate import init_empty_weights
from transformers import AutoTokenizer
from transformers.masking_utils import create_causal_mask
from tqdm import tqdm

from mmgp import offload, runtime
from shared.utils import files_locator as fl

from models.flux.modules.autoencoder_flux2 import AutoencoderKLFlux2, AutoEncoderParamsFlux2
from .constants import IMAGE_POSITION_OFFSET, LLM_TOKEN_INDICATOR, OUTPUT_IMAGE_INDICATOR, QWEN3_VL_ACTIVATION_LAYERS, SEQUENCE_PADDING_INDICATOR
from .latent_norm import get_latent_norm
from .modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer, get_linear_split_map
from .qwen3_vl_configuration import Qwen3VLConfig, register_qwen3_vl_config
from .qwen3_vl_transformers import Qwen3VLTextModel
from .sampler_configs import PRESETS
from .scheduler import get_schedule_for_resolution, make_step_intervals

_DEFAULT_PRESET = "V4_DEFAULT_20"
_TRANSFORMER_WRAPPER_PREFIX = "model.diffusion_model."
_SAMPLE_SOLVERS = {"euler", "res_2m", "res_2s"}


def _res_phi(order: int, neg_h: float) -> float:
    if order == 1:
        return 1.0 + neg_h * (0.5 + neg_h * (1.0 / 6.0 + neg_h / 24.0)) if abs(neg_h) < 1e-4 else math.expm1(neg_h) / neg_h
    if order == 2:
        return 0.5 + neg_h * (1.0 / 6.0 + neg_h * (1.0 / 24.0 + neg_h / 120.0)) if abs(neg_h) < 1e-3 else (math.expm1(neg_h) - neg_h) / (neg_h * neg_h)
    raise ValueError(f"Unsupported RES phi order {order}")


def _res_2s_coefficients(h: float, c2: float = 0.5) -> tuple[float, float, float]:
    f1 = _res_phi(1, -h)
    f2 = _res_phi(2, -h)
    a21 = c2 * _res_phi(1, -h * c2)
    b2 = f2 / c2
    return a21, f1 - b2, b2


def _res_2m_coefficients(h: float, h_prev: float) -> tuple[float, float]:
    c2 = -h_prev / h
    f1 = _res_phi(1, -h)
    f2 = _res_phi(2, -h)
    b2 = f2 / c2
    return f1 - b2, b2


def _phase_label(step_idx: int, guide_phases: int, phase_switch_step: int, phase_switch_step2: int) -> str:
    if guide_phases <= 1:
        return ""
    phase_no = 3 if guide_phases >= 3 and step_idx >= phase_switch_step2 else 2 if guide_phases >= 2 and step_idx >= phase_switch_step else 1
    return f"Phase {phase_no}/{guide_phases}"


def _phase_steps_description(num_steps: int, guide_phases: int, phase_switch_step: int, phase_switch_step2: int) -> str:
    if guide_phases <= 1:
        return ""
    phase_switch_step = min(max(int(phase_switch_step), 0), num_steps)
    phase_switch_step2 = min(max(int(phase_switch_step2), phase_switch_step), num_steps)
    description = "Denoising Steps:"
    description += " Phase 1 = None" if phase_switch_step == 0 else f" Phase 1 = 1:{phase_switch_step}"
    if guide_phases >= 2:
        description += ", Phase 2 = None" if phase_switch_step == phase_switch_step2 else f", Phase 2 = {phase_switch_step + 1}:{phase_switch_step2}"
    if guide_phases >= 3 and phase_switch_step2 < num_steps:
        description += f", Phase 3 = {phase_switch_step2 + 1}:{num_steps}"
    return description


def _time_snr_shift(shift: float, t: float) -> float:
    if shift == 1.0:
        return t
    return shift * t / (1.0 + (shift - 1.0) * t)


def _flow_model_timestep(t: float, shift: float) -> float:
    return 1.0 - _time_snr_shift(shift, 1.0 - t)


def _custom_float(custom_settings: dict, key: str, default: float) -> float:
    value = custom_settings.get(key, default)
    return float(value)


def _apply_ideogram_lora_branches(conditional_transformer, unconditional_transformer, loras_slists, num_steps: int, phase_switch_step: int, phase_switch_step2: int) -> None:
    if loras_slists is None:
        return
    from shared.utils.loras_mutipliers import update_loras_slists

    update_loras_slists(conditional_transformer, loras_slists.get("cond", loras_slists), num_steps, phase_switch_step=phase_switch_step, phase_switch_step2=phase_switch_step2)
    if unconditional_transformer is not None:
        update_loras_slists(unconditional_transformer, loras_slists.get("uncond", loras_slists), num_steps, phase_switch_step=phase_switch_step, phase_switch_step2=phase_switch_step2)


def _strip_transformer_wrapper(state_dict, quantization_map=None, tied_weights_map=None):
    if not any(key.startswith(_TRANSFORMER_WRAPPER_PREFIX) for key in state_dict):
        return state_dict, quantization_map, tied_weights_map

    def strip_mapping(mapping):
        if mapping is None:
            return None
        prefix_len = len(_TRANSFORMER_WRAPPER_PREFIX)
        return {key[prefix_len:]: value for key, value in mapping.items() if key.startswith(_TRANSFORMER_WRAPPER_PREFIX)}

    return strip_mapping(state_dict), strip_mapping(quantization_map), strip_mapping(tied_weights_map)


def _load_transformer(filename: str, dtype: torch.dtype) -> Ideogram4Transformer:
    config = Ideogram4Config()
    split_map = get_linear_split_map(config.emb_dim)
    with init_empty_weights(include_buffers=True):
        transformer = Ideogram4Transformer(config)
    transformer.rotary_emb.reset_inv_freq()
    offload.load_model_data(transformer, filename, writable_tensors=False, default_dtype=dtype, fused_split_map=split_map, preprocess_sd=_strip_transformer_wrapper)
    transformer.split_linear_modules_map = split_map
    transformer.eval().requires_grad_(False)
    return transformer


class Ideogram4TextEncoder(torch.nn.Module):
    def __init__(self, config: Qwen3VLConfig) -> None:
        super().__init__()
        self.language_model = Qwen3VLTextModel(config.text_config)


def _load_text_encoder(filename: str, config_path: str, dtype: torch.dtype) -> Ideogram4TextEncoder:
    register_qwen3_vl_config()
    config = Qwen3VLConfig.from_json_file(config_path)
    with init_empty_weights(include_buffers=True):
        text_encoder = Ideogram4TextEncoder(config)
    text_encoder.language_model.rotary_emb.reset_inv_freq()
    offload.load_model_data(text_encoder.language_model, filename, modelPrefix="language_model", writable_tensors=False, default_dtype=dtype)
    runtime.lookup_on_cpu(text_encoder.language_model.embed_tokens)  # the 1.2 GB token table stays in RAM: mmgp looks up the prompt's rows on the CPU
    text_encoder.eval().requires_grad_(False)
    return text_encoder


def _load_autoencoder(filename: str, dtype: torch.dtype) -> AutoencoderKLFlux2:
    with init_empty_weights(include_buffers=True):
        autoencoder = AutoencoderKLFlux2(AutoEncoderParamsFlux2())
    offload.load_model_data(autoencoder, filename, writable_tensors=False, default_dtype=None)
    autoencoder.eval().requires_grad_(False)
    return autoencoder


class Ideogram4WanPipeline:
    def __init__(self, conditional_transformer, unconditional_transformer, text_encoder, text_tokenizer, autoencoder, dtype=torch.bfloat16) -> None:
        self.conditional_transformer = conditional_transformer
        self.unconditional_transformer = unconditional_transformer
        self.text_encoder = text_encoder
        self.text_tokenizer = text_tokenizer
        self.autoencoder = autoencoder
        self.dtype = dtype
        self.patch_size = 2
        self.ae_scale_factor = 8
        self.max_text_tokens = 2048
        self._interrupt = False
        shift, scale = get_latent_norm()
        self.latent_shift = shift
        self.latent_scale = scale

    @property
    def device(self) -> torch.device:
        return next(self.conditional_transformer.parameters()).device

    @property
    def runtime_device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else self.device)

    def _tokenize(self, prompt: str) -> tuple[torch.Tensor, int]:
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        text = self.text_tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        encoded = self.text_tokenizer(text, return_tensors="pt", add_special_tokens=False)
        token_ids = encoded["input_ids"][0]
        num_text_tokens = int(token_ids.shape[0])
        if num_text_tokens > self.max_text_tokens:
            raise ValueError(f"prompt has {num_text_tokens} tokens, exceeds max_text_tokens={self.max_text_tokens}")
        return token_ids, num_text_tokens

    def _build_inputs(self, prompts: list[str], height: int, width: int) -> dict[str, torch.Tensor | int]:
        tokenized = [self._tokenize(p) for p in prompts]
        batch_size = len(prompts)
        patch = self.patch_size * self.ae_scale_factor
        if height % patch != 0 or width % patch != 0:
            raise ValueError(f"height/width must be divisible by {patch}")
        grid_h = height // patch
        grid_w = width // patch
        num_image_tokens = grid_h * grid_w
        max_text_tokens = max(num_text for _, num_text in tokenized)
        total_seq_len = max_text_tokens + num_image_tokens

        h_idx = torch.arange(grid_h, device="cpu").view(-1, 1).expand(grid_h, grid_w).reshape(-1)
        w_idx = torch.arange(grid_w, device="cpu").view(1, -1).expand(grid_h, grid_w).reshape(-1)
        t_idx = torch.zeros_like(h_idx)
        image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET

        token_ids = torch.zeros(batch_size, max_text_tokens, dtype=torch.long, device="cpu")
        text_position_ids = torch.zeros(batch_size, max_text_tokens, 3, dtype=torch.long, device="cpu")
        position_ids = torch.zeros(batch_size, total_seq_len, 3, dtype=torch.long, device="cpu")
        segment_ids = torch.full((batch_size, total_seq_len), SEQUENCE_PADDING_INDICATOR, dtype=torch.long, device="cpu")
        indicator = torch.zeros(batch_size, total_seq_len, dtype=torch.long, device="cpu")

        for batch_idx, (tokens, num_text) in enumerate(tokenized):
            pad_len = max_text_tokens - num_text
            total_unpadded = num_text + num_image_tokens
            offset = pad_len
            token_ids[batch_idx, offset:offset + num_text] = tokens
            text_pos = torch.arange(num_text, device="cpu")
            text_pos_3d = torch.stack([text_pos, text_pos, text_pos], dim=1)
            text_position_ids[batch_idx, offset:offset + num_text] = text_pos_3d
            position_ids[batch_idx, offset:offset + num_text] = text_pos_3d
            position_ids[batch_idx, offset + num_text:] = image_pos
            indicator[batch_idx, offset:offset + num_text] = LLM_TOKEN_INDICATOR
            indicator[batch_idx, offset + num_text:] = OUTPUT_IMAGE_INDICATOR
            segment_ids[batch_idx, offset:offset + total_unpadded] = 1

        device = self.runtime_device
        return {
            "token_ids": token_ids.to(device),
            "text_position_ids": text_position_ids.to(device),
            "position_ids": position_ids.to(device),
            # a key mask only with padding (prompts of different lengths): all True otherwise, and attention without a mask is faster
            "segment_ids": segment_ids.to(device) if (segment_ids == SEQUENCE_PADDING_INDICATOR).any() else None,
            "indicator": indicator.to(device),
            "num_image_tokens": num_image_tokens,
            "grid_h": grid_h,
            "grid_w": grid_w,
            "max_text_tokens": max_text_tokens,
        }

    def _encode_text(self, token_ids: torch.Tensor, text_position_ids: torch.Tensor, indicator: torch.Tensor) -> torch.Tensor | None:
        attention_mask = (indicator == LLM_TOKEN_INDICATOR).to(torch.long)
        pos_2d = text_position_ids[..., 0].contiguous()
        language_model = self.text_encoder.language_model
        language_model._interrupt = self._interrupt
        with torch.inference_mode():
            inputs_embeds = language_model.embed_tokens(token_ids)
            position_ids = pos_2d[None, ...].expand(4, pos_2d.shape[0], -1)
            text_position_ids = position_ids[0]
            mrope_position_ids = position_ids[1:]
            cache_position = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device)
            causal_mask = create_causal_mask(
                config=language_model.config,
                input_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=None,
                position_ids=text_position_ids,
            )
            position_embeddings = language_model.rotary_emb(inputs_embeds, mrope_position_ids)
            tap_layers = set(QWEN3_VL_ACTIVATION_LAYERS)
            captured = {}
            hidden_states = inputs_embeds
            with text_encoding_progress(language_model.layers, prompt_count=token_ids.shape[0]):
                for layer_idx, decoder_layer in enumerate(language_model.layers):
                    hidden_states = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=text_position_ids,
                        past_key_values=None,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
                    if layer_idx in tap_layers:
                        captured[layer_idx] = hidden_states.clone()
                    if self._interrupt:
                        return None
        del hidden_states, inputs_embeds, position_embeddings, causal_mask, position_ids, mrope_position_ids, cache_position, text_position_ids
        first = captured[QWEN3_VL_ACTIVATION_LAYERS[0]]
        batch_size, seq_len, hidden_size = first.shape
        stacked = first.new_empty(batch_size, seq_len, hidden_size, len(QWEN3_VL_ACTIVATION_LAYERS))
        for capture_idx, layer_idx in enumerate(QWEN3_VL_ACTIVATION_LAYERS):
            hidden = captured.pop(layer_idx)
            stacked[..., capture_idx].copy_(hidden)
            del hidden
        stacked = stacked.reshape(batch_size, seq_len, hidden_size * len(QWEN3_VL_ACTIVATION_LAYERS))
        stacked.mul_(attention_mask.to(device=stacked.device, dtype=stacked.dtype).unsqueeze(-1))
        return stacked

    def _decode_image(self, z: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        z = self._unpack_vae_latents(z, grid_h, grid_w)
        vae_dtype = next(self.autoencoder.decoder.parameters()).dtype
        with vae_decoding_progress(1, self.autoencoder.decoder):
            return self.autoencoder.decoder(z.to(vae_dtype)).float().clamp(-1.0, 1.0)

    def _decode(self, z: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        return self._decode_image(z, grid_h, grid_w).cpu().transpose(0, 1)

    def _unpack_vae_latents(self, z: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        z = self._normalize_packed_latents(z)
        batch_size = z.shape[0]
        patch = self.patch_size
        ae_channels = z.shape[-1] // (patch * patch)
        z = z.view(batch_size, grid_h, grid_w, patch, patch, ae_channels)
        z = z.permute(0, 5, 1, 3, 2, 4).contiguous()
        return z.view(batch_size, ae_channels, grid_h * patch, grid_w * patch)

    def _normalize_packed_latents(self, z: torch.Tensor) -> torch.Tensor:
        latent_shift = self.latent_shift.to(z.device, z.dtype)
        latent_scale = self.latent_scale.to(z.device, z.dtype)
        return z * latent_scale + latent_shift

    def _pack_vae_upsampler_lq_latent(self, z: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        z = self._normalize_packed_latents(z)
        batch_size = z.shape[0]
        patch = self.patch_size
        ae_channels = z.shape[-1] // (patch * patch)
        z = z.view(batch_size, grid_h, grid_w, patch, patch, ae_channels).permute(0, 5, 3, 4, 1, 2).contiguous()
        z = z.view(batch_size, ae_channels * patch * patch, grid_h, grid_w)
        vae_mean = self.autoencoder.bn.running_mean.view(1, -1, 1, 1).to(device=z.device, dtype=z.dtype)
        vae_scale = torch.sqrt(self.autoencoder.bn.running_var.view(1, -1, 1, 1) + self.autoencoder.bn_eps).to(device=z.device, dtype=z.dtype)
        return z.sub(vae_mean).div(vae_scale)

    @torch.inference_mode()
    def __call__(
        self,
        prompts: str | list[str],
        *,
        height: int = 1024,
        width: int = 1024,
        num_steps: int = 20,
        guidance_scale: float = 7.0,
        guidance2_scale: float = 3.0,
        guidance3_scale: float = 3.0,
        guidance_schedule=None,
        mu: float = 0.0,
        std: float = 1.75,
        sample_solver: str = "euler",
        flow_shift: float = 1.0,
        guide_phases: int = 1,
        switch_threshold: int = 0,
        switch2_threshold: int = 0,
        loras_slists=None,
        seed: int | None = None,
        callback=None,
        vae_upsampler=None,
        set_progress_status=None,
        set_header_text=None,
    ) -> torch.Tensor | None:
        if isinstance(prompts, str):
            prompts = [prompts]
        sample_solver = (sample_solver or "euler").lower()
        if sample_solver not in _SAMPLE_SOLVERS:
            raise ValueError(f"Unsupported Ideogram 4 sampler '{sample_solver}'.")
        device = self.runtime_device
        schedule = get_schedule_for_resolution((height, width), known_mean=mu, std=std)
        step_intervals = make_step_intervals(num_steps, device)
        time_points = schedule(step_intervals).to(device)
        sigma_points = 1.0 - time_points
        phase_switch_step = num_steps
        phase_switch_step2 = num_steps
        if guidance_schedule is not None:
            gw_per_step = torch.as_tensor(guidance_schedule, dtype=torch.float32, device=device)
        else:
            gw_per_step = torch.full((num_steps,), float(guidance_scale), dtype=torch.float32, device=device)
            if int(guide_phases) >= 2 and int(switch_threshold) > 0:
                switch_sigma = float(switch_threshold) / 1000.0
                phase_switch_step = int((sigma_points[1:] > switch_sigma).sum().item())
                override_mask = sigma_points[1:] <= switch_sigma
                gw_per_step = torch.where(override_mask, torch.full_like(gw_per_step, float(guidance2_scale)), gw_per_step)
            if int(guide_phases) >= 3 and int(switch2_threshold) > 0:
                switch2_sigma = float(switch2_threshold) / 1000.0
                phase_switch_step2 = int((sigma_points[1:] > switch2_sigma).sum().item())
                override2_mask = sigma_points[1:] <= switch2_sigma
                gw_per_step = torch.where(override2_mask, torch.full_like(gw_per_step, float(guidance3_scale)), gw_per_step)

        _apply_ideogram_lora_branches(self.conditional_transformer, self.unconditional_transformer, loras_slists, num_steps, phase_switch_step, phase_switch_step2)
        phase_description = _phase_steps_description(num_steps, int(guide_phases), phase_switch_step, phase_switch_step2)
        if len(phase_description) > 0 and callable(set_header_text):
            set_header_text(phase_description)

        inputs = self._build_inputs(prompts, height=height, width=width)
        if self._interrupt:
            return None
        batch_size = len(prompts)
        num_image_tokens = inputs["num_image_tokens"]
        grid_h = inputs["grid_h"]
        grid_w = inputs["grid_w"]
        max_text_tokens = inputs["max_text_tokens"]
        latent_dim = self.conditional_transformer.config.in_channels

        llm_features = self._encode_text(
            inputs["token_ids"],
            inputs["text_position_ids"],
            inputs["indicator"][:, :max_text_tokens],
        )
        if llm_features is None or self._interrupt:
            return None

        if self.unconditional_transformer is not None:
            neg_position_ids = inputs["position_ids"][:, max_text_tokens:]
            neg_segment_ids = None  # image tokens are never padded
            neg_indicator = inputs["indicator"][:, max_text_tokens:]
            neg_llm_features = llm_features.new_empty(batch_size, 0, llm_features.shape[-1])

        generator = torch.Generator(device=device)
        if seed is not None and seed >= 0:
            generator.manual_seed(int(seed))
        z = torch.randn(batch_size, num_image_tokens, latent_dim, dtype=torch.float32, device=device, generator=generator)
        pos_z = torch.empty(batch_size, max_text_tokens + num_image_tokens, latent_dim, dtype=torch.float32, device=device)
        pos_z[:, :max_text_tokens].zero_()

        if callback is not None:
            callback(-1, None, True, override_num_inference_steps=num_steps, denoising_extra=_phase_label(0, int(guide_phases), phase_switch_step, phase_switch_step2))

        def predict_velocity(current_z: torch.Tensor, t_val: float, guidance: torch.Tensor) -> torch.Tensor | None:
            model_t_val = _flow_model_timestep(t_val, float(flow_shift))
            t = torch.full((batch_size,), model_t_val, dtype=torch.float32, device=device)
            pos_z[:, max_text_tokens:].copy_(current_z)
            pos_out = self.conditional_transformer(
                llm_features=llm_features,
                x=pos_z,
                t=t,
                position_ids=inputs["position_ids"],
                segment_ids=inputs["segment_ids"],
                indicator=inputs["indicator"],
            )
            if pos_out is None:
                return None
            pos_v = pos_out[:, max_text_tokens:]
            if self.unconditional_transformer is None:
                return pos_v

            neg_v = self.unconditional_transformer(
                llm_features=neg_llm_features,
                x=current_z,
                t=t,
                position_ids=neg_position_ids,
                segment_ids=neg_segment_ids,
                indicator=neg_indicator,
            )
            if neg_v is None:
                return None
            return guidance * pos_v + (1.0 - guidance) * neg_v

        prev_denoised = None
        prev_sigma = None
        for step_idx, i in enumerate(tqdm(range(num_steps - 1, -1, -1), total=num_steps, desc="Denoising")):
            if self._interrupt:
                return None
            t_val = float(time_points[i + 1].item())
            s_val = float(time_points[i].item())
            sigma = float(sigma_points[i + 1].item())
            sigma_down = float(sigma_points[i].item())
            guidance = gw_per_step[i]
            denoising_extra = _phase_label(step_idx, int(guide_phases), phase_switch_step, phase_switch_step2)
            v = predict_velocity(z, t_val, guidance)
            if v is None:
                return None

            h = -math.log(sigma_down / sigma) if sigma_down > 0.0 else 0.0
            denoised = z + v * sigma
            if sample_solver == "res_2m" and prev_denoised is not None and sigma_down > 0.0 and h < 1.0:
                b1, b2 = _res_2m_coefficients(h, -math.log(sigma / prev_sigma))
                z = z + h * (b1 * (denoised - z) + b2 * (prev_denoised - z))
            elif sample_solver in {"res_2s", "res_2m"} and sigma_down > 0.0 and (sample_solver == "res_2s" or sigma >= 0.1):
                a21, b1, b2 = _res_2s_coefficients(h)
                sub_sigma = sigma * math.exp(-0.5 * h)
                sub_z = z + h * a21 * (denoised - z)
                sub_v = predict_velocity(sub_z, 1.0 - sub_sigma, guidance)
                if sub_v is None:
                    return None
                sub_denoised = sub_z + sub_v * sub_sigma
                z = z + h * (b1 * (denoised - z) + b2 * (sub_denoised - z))
            else:
                z = z + v * (s_val - t_val)
            prev_denoised = denoised
            prev_sigma = sigma
            if callback is not None:
                preview_x0 = denoised if getattr(callback, "tiny_vae", False) else z
                preview = self._unpack_vae_latents(preview_x0[:1], grid_h, grid_w)[0].unsqueeze(1)
                callback(step_idx, preview, False, denoising_extra=denoising_extra)
                del preview, preview_x0

        if self._interrupt:
            return None
        if vae_upsampler is None:
            return self._decode(z, grid_h=grid_h, grid_w=grid_w)

        def _vae_upsampler_progress(_phase, current_step=None, total_steps=None):
            if callable(set_progress_status):
                progress_label = getattr(vae_upsampler, "progress_label", "VAE Spatial Upsampling")
                if current_step is None or total_steps is None:
                    set_progress_status(f"{progress_label} in Progress")
                else:
                    total_steps = int(total_steps)
                    step_no = min(int(current_step) + 1, total_steps)
                    set_progress_status(f"{progress_label} in Progress ({step_no}/{total_steps})")

        _vae_upsampler_progress(None)
        lq_image_ref = [self._decode_image(z, grid_h=grid_h, grid_w=grid_w)]
        lq_latent_ref = [self._pack_vae_upsampler_lq_latent(z, grid_h, grid_w)]
        image = vae_upsampler.decode_inputs(
            lq_image_ref,
            lq_latent_ref,
            prompt=prompts,
            seed=seed,
            abort_callback=lambda: self._interrupt,
            progress_callback=_vae_upsampler_progress,
        )
        if image is None:
            return None
        return image.cpu().transpose(0, 1)


class model_factory:
    def __init__(
        self,
        checkpoint_dir,
        model_filename=None,
        model_type=None,
        model_def=None,
        base_model_type=None,
        text_encoder_filename=None,
        dtype=torch.bfloat16,
        VAE_dtype=torch.float32,
        save_quantized=False,
        **kwargs,
    ):
        model_def = model_def or {}
        conditional_only = bool(model_def.get("conditional_transformer_only", False))
        min_transformers = 1 if conditional_only else 2
        if not isinstance(model_filename, (list, tuple)) or len(model_filename) < min_transformers:
            raise ValueError("Ideogram 4 requires a conditional transformer file." if conditional_only else "Ideogram 4 requires conditional and unconditional transformer files.")
        if text_encoder_filename is None:
            raise ValueError("Ideogram 4 requires a Qwen3-VL text encoder file.")

        self.model_type = model_type
        self.base_model_type = base_model_type
        self.model_def = model_def
        dtype = torch.bfloat16
        self.dtype = dtype

        text_encoder_folder = model_def.get("text_encoder_folder", "Qwen3-VL-8B-Instruct")
        tokenizer_path = os.path.dirname(fl.locate_file(os.path.join(text_encoder_folder, "tokenizer_config.json")))
        text_config_path = fl.locate_file(os.path.join(text_encoder_folder, "config.json"))
        vae_filename = fl.locate_file("flux2_vae.safetensors")

        self.conditional_transformer = _load_transformer(model_filename[0], dtype)
        self.unconditional_transformer = None if conditional_only else _load_transformer(model_filename[1], dtype)
        self.transformer = self.conditional_transformer
        self.model = self.conditional_transformer
        if self.unconditional_transformer is not None:
            self.transformer2 = self.unconditional_transformer
            self.model2 = self.unconditional_transformer
        if save_quantized:
            from wgp import save_quantized_model
            save_quantized_model(self.conditional_transformer, model_type, model_filename[0], dtype, None)
            if self.unconditional_transformer is not None:
                save_quantized_model(self.unconditional_transformer, model_type, model_filename[1], dtype, None, submodel_no=2)
        self.text_encoder = _load_text_encoder(text_encoder_filename, text_config_path, dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, extra_special_tokens={})
        self.autoencoder = _load_autoencoder(vae_filename, VAE_dtype)
        self.pipeline = Ideogram4WanPipeline(
            self.conditional_transformer,
            self.unconditional_transformer,
            self.text_encoder,
            self.tokenizer,
            self.autoencoder,
            dtype=VAE_dtype,
        )

    @generation_progress
    def generate(
        self,
        seed=None,
        input_prompt="",
        sample_solver="euler",
        width=1024,
        height=1024,
        sampling_steps=20,
        guide_scale=7.0,
        guide2_scale=3.0,
        guide3_scale=3.0,
        shift=1.0,
        switch_threshold=0,
        switch2_threshold=0,
        guide_phases=1,
        batch_size=1,
        model_mode=None,
        custom_settings=None,
        loras_slists=None,
        vae_upsampler=None,
        set_progress_status=None,
        set_header_text=None,
        callback=None,
        **kwargs,
    ):
        preset = PRESETS.get(model_mode)
        custom_settings = custom_settings if isinstance(custom_settings, dict) else {}
        num_steps = int(preset.num_steps if preset is not None else sampling_steps)
        mu = _custom_float(custom_settings, "ideogram_mu", preset.mu if preset is not None else 0.0)
        std = _custom_float(custom_settings, "ideogram_std", preset.std if preset is not None else 1.75)
        guidance_schedule = preset.guidance_schedule if preset is not None and len(custom_settings) == 0 else None

        prompts = [input_prompt] * int(batch_size)
        return self.pipeline(
            prompts,
            height=height,
            width=width,
            num_steps=num_steps,
            guidance_scale=guide_scale,
            guidance2_scale=guide2_scale,
            guidance3_scale=guide3_scale,
            guidance_schedule=guidance_schedule,
            mu=mu,
            std=std,
            sample_solver=sample_solver,
            flow_shift=float(shift),
            guide_phases=guide_phases,
            switch_threshold=switch_threshold,
            switch2_threshold=switch2_threshold,
            loras_slists=loras_slists,
            seed=seed,
            callback=callback,
            vae_upsampler=vae_upsampler,
            set_progress_status=set_progress_status,
            set_header_text=set_header_text,
        )

    @property
    def _interrupt(self):
        return getattr(self.pipeline, "_interrupt", False)

    @_interrupt.setter
    def _interrupt(self, value):
        if hasattr(self, "pipeline"):
            self.pipeline._interrupt = bool(value)
            self.conditional_transformer._interrupt = bool(value)
            if self.unconditional_transformer is not None:
                self.unconditional_transformer._interrupt = bool(value)
            if hasattr(self.text_encoder, "language_model"):
                self.text_encoder.language_model._interrupt = bool(value)
