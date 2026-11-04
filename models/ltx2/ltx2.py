import sys
import copy
import json
import math
import os
import re
import types
from contextlib import nullcontext
from typing import Callable, Iterator

import torch
import torchaudio
from mmgp import offload
from accelerate import init_empty_weights
from safetensors.torch import load_file
from shared.utils import files_locator as fl
from shared.utils.phase_progress import generation_progress
from shared.cuda_memory import mmgp_vram_allocator
from shared.utils.hdr import VIDEO_PROMPT_HDR_OUTPUT_FLAG, hdr_linear_to_vae_range, sdr_to_vae_range

from .ltx_core.conditioning import AudioConditionByLatent, AudioConditionByLatentPrefix, AudioConditionByReferenceLatent
from .ltx_core.model.audio_vae import (
    VOCODER_COMFY_KEYS_FILTER,
    AudioDecoderConfigurator,
    AudioEncoderConfigurator,
    AudioProcessor,
    VocoderConfigurator,
)
from .ltx_core.model.transformer import (
    LTXV_MODEL_COMFY_RENAMING_MAP,
    LTXModelConfigurator,
    X0Model,
)
from .ltx_core.model.transformer.sol_attention import SOL_ATTN_TAU_START_KEY
from .ltx_core.model.upsampler import LatentUpsamplerConfigurator
from .ltx_core.model.video_vae import VideoDecoderConfigurator, VideoEncoderConfigurator
from .ltx_core.model.video_vae.diffusion_video_decoder import DiffusionVideoDecoder
from .ltx_core.text_encoders.gemma import (
    AUDIO_EMBEDDINGS_CONNECTOR_KEY_OPS,
    GemmaTextEmbeddingsConnectorModelConfigurator,
    GemmaTextEmbeddingsConnectorModel,
    TEXT_EMBEDDING_PROJECTION_KEY_OPS,
    TEXT_EMBEDDINGS_CONNECTOR_KEY_OPS,
    VIDEO_EMBEDDINGS_CONNECTOR_KEY_OPS,
    build_gemma_text_encoder,
)
from .ltx_core.text_encoders.gemma.embeddings_connector import AudioEmbeddings1DConnectorConfigurator, Embeddings1DConnectorConfigurator
from .ltx_core.text_encoders.gemma.feature_extractor import GemmaFeaturesExtractorProjLinear
from .ltx_core.model.video_vae import SpatialTilingConfig, TemporalTilingConfig, TilingConfig
from .ltx_core.types import AudioLatentShape, VideoPixelShape
from .inpainting import (
    _apply_ltx2_mask_blend,
    _apply_ltx2_inpaint_preprocess_dilation,
    _build_outpainting_mask_cthw,
    _get_outpainting_inner_rect,
    _ltx2_inpainting_enabled,
    _merge_ltx2_masks,
    _normalize_outpainting_dims,
    _pad_ltx2_masked_control_video_tail,
)
from .lora_utils import is_ic_lora_filename, phase2_ic_lora_name
from .ltx2_runtime import (
    LTX2_INPAINTING_LAPLACIAN_BLEND,
    LTX2_INPAINTING_LAPLACIAN_MASK_LOW_RES_DILATION,
    LTX2_INPAINTING_PREPROCESS_MASK_DILATION,
    LTX2_INPAINTING_SANITIZE_LAPLACIAN_SOURCE,
    LTX2_OUTPAINTING_LAPLACIAN_BLEND,
    LTX2_OUTPAINTING_LAPLACIAN_MASK_LOW_RES_DILATION,
    LTX2_OUTPAINTING_METHOD,
    LTX2_PAD_MASKED_CONTROL_VIDEO_TAIL,
)
from .ltx_pipelines.distilled import DistilledPipeline
from .ltx_pipelines.hdr_ic_lora import convert_sdr_to_hdr
from .ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from .ltx_pipelines.utils.media_io import UInt8Guide
from .ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE, DEFAULT_NEGATIVE_PROMPT, DISTILLED_SIGMA_VALUES


_GEMMA_FOLDER = "gemma-3-12b-it-qat-q4_0-unquantized"
_SPATIAL_UPSCALER_FILENAME = "ltx-2-spatial-upscaler-x2-1.0.safetensors"
LTX2_USE_FP32_ROPE_FREQS = True
LTX2_ID_LORA_GUIDANCE_SCALE = 3.0
LTX2_ID_LORA_AUDIO_CFG_SCALE = 7.0
LTX2_ID_LORA_MAX_REFERENCE_SECONDS = 121.0 / 25.0
LTX2_OUTPAINT_GAMMA = 2.0
LTX2_DISABLE_STAGE2_WITH_CONTROL_VIDEO = True
LTX2_ENABLE_EMBEDDING_LORAS = False
LTX2_VAE_TEMPORAL_TILING_FPS = 24.0
LTX2_UPSCALE_SIGMAS = tuple(DISTILLED_SIGMA_VALUES)
LTX2_UPSCALE_MAX_FRAMES = 481
LTX2_EMBEDDING_LORA_PREFIXES = (
    "text_embedding_projection.",
    "feature_extractor_linear.",
    "text_embeddings_connector.",
    "embeddings_connector.",
    "video_embeddings_connector.",
    "audio_embeddings_connector.",
)
LTX2_COMFY_LORA_UNDERSCORED_NAMES = (
    "av_ca_audio_scale_shift_adaln_single",
    "av_ca_video_scale_shift_adaln_single",
    "av_ca_a2v_gate_adaln_single",
    "av_ca_v2a_gate_adaln_single",
    "audio_prompt_adaln_single",
    "audio_patchify_proj",
    "audio_to_video_attn",
    "prompt_adaln_single",
    "video_to_audio_attn",
    "audio_adaln_single",
    "transformer_blocks",
    "timestep_embedder",
    "audio_proj_out",
    "to_gate_logits",
    "patchify_proj",
    "adaln_single",
    "audio_attn1",
    "audio_attn2",
    "audio_ff",
    "linear_1",
    "linear_2",
    "proj_out",
    "k_norm",
    "q_norm",
    "to_out",
    "to_k",
    "to_q",
    "to_v",
)


_LTX2_MAIN_LORA_ARCHITECTURES = {"ltx2_22B", "ltx2_25_22B", "ltx2_25_22B_alpha_gen", "ltx2_25_22B_layout_to_render"}


def _ltx2_main_loras_compatible(base_model_type: str | None) -> bool:
    return base_model_type in _LTX2_MAIN_LORA_ARCHITECTURES


def _ltx2_main_ingredients_enabled(base_model_type: str | None, video_prompt_type: str) -> bool:
    video_prompt_type = video_prompt_type or ""
    return _ltx2_main_loras_compatible(base_model_type) and "I" in video_prompt_type and not any(letter in video_prompt_type for letter in f"KFVOPDEMA{VIDEO_PROMPT_HDR_OUTPUT_FLAG}")


def _normalize_config(config_value):
    if isinstance(config_value, dict):
        return config_value
    if isinstance(config_value, (bytes, bytearray, memoryview)):
        try:
            config_value = bytes(config_value).decode("utf-8")
        except Exception:
            return {}
    if isinstance(config_value, str):
        try:
            return json.loads(config_value)
        except json.JSONDecodeError:
            return {}
    return {}


def _is_editanything_model(model_def) -> bool:
    return bool((model_def or {}).get("ltx2_edit_anything", False))


def _load_config_from_checkpoint(path, fallback_config_path: str | None = None):
    from mmgp import quant_router

    if isinstance(path, (list, tuple)):
        if not path:
            return {}
        path = path[0]
    if not path:
        return {}

    def _read_config_metadata(one_path: str) -> dict:
        if not one_path:
            return {}
        _, metadata = quant_router.load_metadata_state_dict(one_path)
        if not metadata:
            return {}
        return _normalize_config(metadata.get("config"))

    config = _read_config_metadata(path)
    if config:
        return config
    if not fallback_config_path:
        return {}
    try:
        with open(fallback_config_path, "r", encoding="utf-8") as reader:
            return _normalize_config(json.load(reader))
    except Exception:
        return {}


def _strip_model_prefix(key: str) -> str:
    for prefix in ("model.", "velocity_model."):
        if key.startswith(prefix):
            return _strip_model_prefix(key[len(prefix) :])
    return key


def _apply_sd_ops(state_dict: dict, quantization_map: dict | None, sd_ops):
    if sd_ops is not None:
        has_match = False
        for key in state_dict.keys():
            key = _strip_model_prefix(key)
            if sd_ops.apply_to_key(key) is not None:
                has_match = True
                break
        if not has_match:
            new_sd = {_strip_model_prefix(k): v for k, v in state_dict.items()}
            new_qm = {}
            if quantization_map:
                new_qm = {_strip_model_prefix(k): v for k, v in quantization_map.items()}
            return new_sd, new_qm

    new_sd = {}
    for key, value in state_dict.items():
        key = _strip_model_prefix(key)
        if sd_ops is None:
            new_sd[key] = value
            continue
        else:
            new_key = sd_ops.apply_to_key(key)
            if new_key is None:
                continue
            new_pairs = sd_ops.apply_to_key_value(new_key, value)
        for pair in new_pairs:
            new_sd[pair.new_key] = pair.new_value

    new_qm = {}
    if quantization_map:
        for key, value in quantization_map.items():
            key = _strip_model_prefix(key)
            if sd_ops is None:
                new_key = key
            else:
                new_key = sd_ops.apply_to_key(key)
                if new_key is None:
                    continue
            new_qm[new_key] = value
    return new_sd, new_qm


def _make_sd_postprocess(sd_ops):
    def postprocess(state_dict, quantization_map):
        return _apply_sd_ops(state_dict, quantization_map, sd_ops)

    return postprocess


def _split_vae_state_dict(state_dict: dict, prefix: str):
    new_sd = {}
    for key, value in state_dict.items():
        key = _strip_model_prefix(key)
        if key.startswith(prefix):
            key = key[len(prefix) :]
        elif key.startswith(("encoder.", "decoder.", "per_channel_statistics.")):
            key = key
        else:
            continue
        if key.startswith("per_channel_statistics."):
            suffix = key[len("per_channel_statistics.") :]
            new_sd[f"encoder.per_channel_statistics.{suffix}"] = value.clone()
            new_sd[f"decoder.per_channel_statistics.{suffix}"] = value.clone()
        else:
            new_sd[key] = value

    return new_sd, {}


def _make_vae_postprocess(prefix: str):
    def postprocess(state_dict, quantization_map):
        return _split_vae_state_dict(state_dict, prefix)

    return postprocess


def _split_diffusion_vae_state_dict(state_dict: dict, prefix: str):
    new_sd = {}
    for key, value in state_dict.items():
        key = _strip_model_prefix(key)
        if key.startswith(prefix):
            key = key[len(prefix):]
        elif not key.startswith(("encoder.", "decoder.", "per_channel_statistics.")):
            continue
        if key.startswith("per_channel_statistics."):
            suffix = key[len("per_channel_statistics."):]
            new_sd[f"encoder.per_channel_statistics.{suffix}"] = value.clone()
            new_sd[f"decoder.per_channel_statistics.{suffix}"] = value.clone()
        elif key.startswith("decoder.t_embedder.mlp.0."):
            new_sd[key.replace("decoder.t_embedder.mlp.0.", "decoder.t_embedder.timestep_embedder.linear_1.")] = value
        elif key.startswith("decoder.t_embedder.mlp.2."):
            new_sd[key.replace("decoder.t_embedder.mlp.2.", "decoder.t_embedder.timestep_embedder.linear_2.")] = value
        elif ".attn.qkv." in key:
            prefix_key, suffix = key.split(".attn.qkv.", 1)
            q, k, v = value.chunk(3, dim=0)
            new_sd[f"{prefix_key}.attn.qkv.to_q.{suffix}"] = q
            new_sd[f"{prefix_key}.attn.qkv.to_k.{suffix}"] = k
            new_sd[f"{prefix_key}.attn.qkv.to_v.{suffix}"] = v
        else:
            new_sd[key] = value
    return new_sd, {}


def _make_diffusion_vae_postprocess(prefix: str):
    def postprocess(state_dict, quantization_map):
        return _split_diffusion_vae_state_dict(state_dict, prefix)

    return postprocess


def _diffusion_vae_encoder_config(config: dict) -> dict:
    encoder = config["vae"]["encoder"]
    return {
        "vae": {
            "dims": encoder["dims"],
            "in_channels": encoder["in_channels"],
            "latent_channels": encoder["out_channels"],
            "encoder_blocks": encoder["blocks"],
            "patch_size": encoder["patch_size"],
            "norm_layer": encoder["norm_layer"],
            "latent_log_var": encoder["latent_log_var"],
            "encoder_spatial_padding_mode": encoder["spatial_padding_mode"],
        }
    }


class _AudioVAEWrapper(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module) -> None:
        super().__init__()
        per_stats = getattr(decoder, "per_channel_statistics", None)
        if per_stats is not None:
            self.per_channel_statistics = per_stats
        self.decoder = decoder


class _VAEContainer(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder


class _ExternalConnectorWrapper:
    def __init__(self, module: torch.nn.Module) -> None:
        self._module = module

    def __call__(self, *args, **kwargs):
        return self._module(*args, **kwargs)


class LTX2SuperModel(torch.nn.Module):
    def __init__(self, ltx2_model: "LTX2") -> None:
        super().__init__()
        object.__setattr__(self, "_ltx2", ltx2_model)

        transformer = ltx2_model.model
        # MMGP registers LoRA slots on the X0Model, not its inner velocity model.
        self.model = transformer
        split_map = getattr(transformer, "split_linear_modules_map", None)
        if split_map is not None:
            self.split_linear_modules_map = split_map

        self.text_embedding_projection = ltx2_model.text_embedding_projection
        self.text_embeddings_connector = ltx2_model.text_embeddings_connector

    @property
    def _interrupt(self) -> bool:
        return self._ltx2._interrupt

    @_interrupt.setter
    def _interrupt(self, value: bool) -> None:
        self._ltx2._interrupt = value

    def forward(self, *args, **kwargs):
        return self._ltx2.model(*args, **kwargs)

    def generate(self, *args, **kwargs):
        return self._ltx2.generate(*args, **kwargs)

    def get_trans_lora(self):
        return self, None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._ltx2, name)


class _LTX2VAEHelper:
    def __init__(self, block_size: int = 64) -> None:
        self.block_size = block_size

    def get_VAE_tile_size(
        self,
        vae_config: int,
        device_mem_capacity: float,
        mixed_precision: bool,
        output_height: int | None = None,
        output_width: int | None = None,
    ) -> int | tuple[int, int]:
        if mmgp_vram_allocator():  # no allocator fragmentation to guard against: Default and Low VRAM levels
            return 256 if vae_config == 3 or device_mem_capacity < 8000 else 512
        if vae_config >= 4:
            vae_config = 0

        if vae_config == 0:
            if mixed_precision:
                device_mem_capacity = device_mem_capacity / 1.5
            if device_mem_capacity >= 24000:
                use_vae_config = 1
            elif device_mem_capacity >= 8000:
                use_vae_config = 2
            else:
                use_vae_config = 3
        else:
            use_vae_config = vae_config

        ref_size = output_height if output_height is not None else output_width
        if ref_size is not None and ref_size > 480:
            use_vae_config += 1

        spatial_tile_size = 128
        if use_vae_config <= 1:
            spatial_tile_size = 0
        elif use_vae_config == 2:
            spatial_tile_size = 512
        elif use_vae_config == 3:
            spatial_tile_size = 256

        return spatial_tile_size


def _attach_lora_preprocessor(transformer: torch.nn.Module) -> None:
    def preprocess_loras(self: torch.nn.Module, model_type: str, sd: dict) -> dict:
        if not sd:
            return sd
        module_names = getattr(self, "_lora_module_names", None)
        if module_names is None:
            module_names = {name for name, _ in self.named_modules()}
            self._lora_module_names = module_names

        def split_lora_key(lora_key: str) -> tuple[str | None, str]:
            if lora_key.endswith(".alpha"):
                return lora_key[: -len(".alpha")], ".alpha"
            if lora_key.endswith(".diff"):
                return lora_key[: -len(".diff")], ".diff"
            if lora_key.endswith(".diff_b"):
                return lora_key[: -len(".diff_b")], ".diff_b"
            if lora_key.endswith(".dora_scale"):
                return lora_key[: -len(".dora_scale")], ".dora_scale"
            pos = lora_key.rfind(".lora_")
            if pos > 0:
                return lora_key[:pos], lora_key[pos:]
            return None, ""

        new_sd = {}
        dropped_keys = []
        for key, value in sd.items():
            original_key = key
            if key.startswith("model."):
                key = key[len("model.") :]
            if key.startswith("diffusion_model."):
                key = key[len("diffusion_model.") :]
            if key.startswith("transformer."):
                key = key[len("transformer.") :]
            module_name, suffix = split_lora_key(key)
            if not module_name:
                dropped_keys.append(original_key)
                continue
            if module_name.startswith("lora_unet_"):
                module_name = module_name[len("lora_unet_") :]
                for name in LTX2_COMFY_LORA_UNDERSCORED_NAMES:
                    module_name = module_name.replace(name, name.replace("_", "\0"))
                module_name = module_name.replace("_", ".").replace("\0", "_")
            if not LTX2_ENABLE_EMBEDDING_LORAS and module_name.startswith(LTX2_EMBEDDING_LORA_PREFIXES):
                continue
            if module_name.startswith("embeddings_connector."):
                module_name = f"text_embeddings_connector.video_embeddings_connector.{module_name[len('embeddings_connector.') :]}"
            if module_name.startswith("video_embeddings_connector."):
                module_name = f"text_embeddings_connector.{module_name}"
            if module_name.startswith("audio_embeddings_connector."):
                module_name = f"text_embeddings_connector.{module_name}"
            if module_name.startswith("feature_extractor_linear."):
                module_name = f"text_embedding_projection.{module_name[len('feature_extractor_linear.') :]}"
            if module_name not in module_names:
                prefixed_name = f"model.{module_name}" if module_name.startswith("velocity_model.") else f"model.velocity_model.{module_name}"
                if module_name.endswith(".to_out") and f"{prefixed_name}.0" in module_names:
                    module_name = f"{prefixed_name}.0"
                elif prefixed_name in module_names:
                    module_name = prefixed_name
                else:
                    dropped_keys.append(original_key)
                    continue
            elif module_name.endswith(".to_out") and f"{module_name}.0" in module_names:
                module_name = f"{module_name}.0"
            new_sd[f"{module_name}{suffix}"] = value
        if dropped_keys:
            sample = ", ".join(dropped_keys[:8])
            if len(dropped_keys) > 8:
                sample += ", ..."
            raise ValueError(
                f"LTX2 LoRA preprocessing dropped {len(dropped_keys)} unmatched keys for model '{model_type}': {sample}"
            )
        return new_sd

    transformer.preprocess_loras = types.MethodType(preprocess_loras, transformer)


def _coerce_image_list(image_value):
    if isinstance(image_value, list):
        return image_value[0] if image_value else None
    return image_value


def _duplicate_ref_image_as_video(ref_image, frame_count: int = 9):
    if ref_image is None:
        return None
    frame_count = max(1, int(frame_count))
    if isinstance(ref_image, (list, tuple)):
        ref_image = ref_image[0] if ref_image else None
        if ref_image is None:
            return None
    if torch.is_tensor(ref_image):
        image = ref_image.detach()
        if image.ndim == 3:
            if image.shape[0] in (1, 3, 4):
                return image.unsqueeze(1).repeat(1, frame_count, 1, 1)
            return image.unsqueeze(0).repeat(frame_count, 1, 1, 1)
        if image.ndim == 4:
            if image.shape[0] in (1, 3, 4):
                return image[:, :1].repeat(1, frame_count, 1, 1)
            if image.shape[-1] in (1, 3, 4):
                return image[:1].repeat(frame_count, 1, 1, 1)
        return image

    import numpy as np
    from PIL import Image

    if isinstance(ref_image, str):
        with Image.open(ref_image) as image:
            frame = np.array(image.convert("RGB"))
    else:
        frame = np.array(ref_image)[..., :3]
    return np.repeat(frame[None, ...], frame_count, axis=0)


def _msr_ref_image_to_frame(ref_image, width: int, height: int):
    import numpy as np
    from PIL import Image

    if isinstance(ref_image, str):
        with Image.open(ref_image) as image:
            pil_image = image.convert("RGB")
    elif torch.is_tensor(ref_image):
        image = ref_image.detach().cpu()
        if image.ndim == 4:
            image = image[0]
        if image.ndim == 3 and image.shape[0] in (1, 3, 4):
            image = image.permute(1, 2, 0)
        frame = image.numpy()
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = frame * 255.0
            frame = frame.clip(0, 255).astype(np.uint8)
        pil_image = Image.fromarray(frame[..., :3]).convert("RGB")
    else:
        frame = np.array(ref_image)
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = frame * 255.0
            frame = frame.clip(0, 255).astype(np.uint8)
        pil_image = Image.fromarray(frame[..., :3]).convert("RGB")
    return np.array(pil_image.resize((int(width), int(height)), Image.Resampling.LANCZOS))


def _is_msr_v2_model_def(model_def) -> bool:
    values = [str((model_def or {}).get("name", ""))]
    loras = (model_def or {}).get("loras", [])
    values.extend(str(value) for value in (loras if isinstance(loras, (list, tuple)) else [loras]))
    return any("msr-v2" in value.lower() or "msr ref v2" in value.lower() for value in values)


def _resolve_msr_reference_frame_count(model_def, custom_settings, ref_images, background_first: bool) -> int:
    default = int((model_def or {}).get("ltx2_msr_frame_count", 41))
    if not _is_msr_v2_model_def(model_def):
        return default
    raw_value = (custom_settings or {}).get("msr_reference_frames", "") if isinstance(custom_settings, dict) else ""
    try:
        selected = int(raw_value or 0)
    except (TypeError, ValueError):
        selected = 0
    if selected in (17, 25, 33, 41, 49, 57, 65):
        return selected
    ref_count = len(ref_images) if isinstance(ref_images, (list, tuple)) else 1
    subject_count = max(1, ref_count - 1 if background_first else ref_count)
    return min(65, 16 * subject_count + 1)


def _build_msr_v2_reference_frames(frames, frame_count: int):
    subject_frames = frames[:-1]
    background_frame = frames[-1]
    video_frames = [background_frame] * frame_count
    subject_budget = max(0, (frame_count - 1) // 8)

    if subject_budget >= len(subject_frames):
        counts = [1] * len(subject_frames)
        extra = subject_budget - len(subject_frames)
        priority = [0, *range(1, len(subject_frames)), 0]
        for index in priority:
            if extra <= 0:
                break
            counts[index] += 1
            extra -= 1
        index = 0
        while extra > 0:
            counts[index % len(counts)] += 1
            extra -= 1
            index += 1
        cursor = 0
        for frame, count in zip(subject_frames, counts):
            start = 0 if cursor == 0 else 1 + (cursor - 1) * 8
            end = (cursor + count - 1) * 8
            cursor += count
            for frame_index in range(start, min(frame_count - 1, end) + 1):
                video_frames[frame_index] = frame
    else:
        subject_frame_count = max(1, (subject_budget - 1) * 8 + 1)
        for index, frame in enumerate(subject_frames):
            start = int(index * subject_frame_count / len(subject_frames))
            end = int((index + 1) * subject_frame_count / len(subject_frames)) - 1
            if index == len(subject_frames) - 1:
                end = subject_frame_count - 1
            for frame_index in range(start, min(frame_count - 1, max(start, end)) + 1):
                video_frames[frame_index] = frame

    return video_frames


def _build_msr_reference_video(
    ref_images,
    frame_count: int,
    width: int,
    height: int,
    background_first: bool,
    msr_v2: bool = False,
):
    import numpy as np

    ref_images = list(ref_images) if isinstance(ref_images, (list, tuple)) else [ref_images]
    if background_first:
        if not 2 <= len(ref_images) <= 5:
            raise ValueError("LTX2 MSR Background + Subjects mode requires 2 to 5 reference images, with the background image first.")
        # WanGP uses the first K+I reference as the ratio/background anchor; MSR expects it last in the pseudo-video.
        ref_images = ref_images[1:] + ref_images[:1]
    elif not 1 <= len(ref_images) <= 4:
        raise ValueError("LTX2 MSR Subjects / Objects only mode requires 1 to 4 reference images.")
    frames = [_msr_ref_image_to_frame(ref_image, width, height) for ref_image in ref_images]
    if msr_v2 and background_first:
        video_frames = _build_msr_v2_reference_frames(frames, frame_count)
    else:
        base_count = frame_count // len(frames)
        remainder = frame_count % len(frames)
        video_frames = []
        for index, frame in enumerate(frames):
            video_frames.extend([frame] * (base_count + (1 if index < remainder else 0)))
    return np.stack(video_frames, axis=0)


def _to_latent_index(frame_idx: int, stride: int) -> int:
    frame_idx = int(frame_idx)
    stride = int(stride)
    if frame_idx <= 0:
        return 0
    # Causal LTX VAEs keep pixel frame 0 in its own latent slot.
    return (frame_idx - 1) // stride + 1


def _normalize_tiling_size(tile_size: int) -> int:
    tile_size = int(tile_size)
    if tile_size <= 0:
        return 0
    tile_size = max(64, tile_size)
    if tile_size % 32 != 0:
        tile_size = int(math.ceil(tile_size / 32) * 32)
    return tile_size


def _normalize_temporal_tiling_size(tile_frames: int) -> int:
    tile_frames = int(tile_frames)
    if tile_frames <= 0:
        return 0
    tile_frames = max(16, tile_frames)
    if tile_frames % 8 != 0:
        tile_frames = int(math.ceil(tile_frames / 8) * 8)
    return tile_frames


def _normalize_temporal_overlap(overlap_frames: int, tile_frames: int) -> int:
    overlap_frames = max(0, int(overlap_frames))
    if overlap_frames % 8 != 0:
        overlap_frames = int(round(overlap_frames / 8) * 8)
    overlap_frames = max(0, min(overlap_frames, max(0, tile_frames - 8)))
    return overlap_frames


def _build_tiling_config(tile_size: int | tuple | list | None, fps: float | None) -> TilingConfig | None:
    temporal_tiling_divisor = 1
    spatial_config = None
    if isinstance(tile_size, (tuple, list)):
        if len(tile_size) == 0:
            tile_size = None
        else:
            if len(tile_size) > 1:
                temporal_tiling_divisor = max(1, int(tile_size[0] or 1))
            tile_size = tile_size[-1]
    if tile_size is not None:
        tile_size = _normalize_tiling_size(tile_size)
        if tile_size > 0:
            overlap = max(0, tile_size // 4)
            overlap = int(math.floor(overlap / 32) * 32)
            if overlap >= tile_size:
                overlap = max(0, tile_size - 32)
            spatial_config = SpatialTilingConfig(tile_size_in_pixels=tile_size, tile_overlap_in_pixels=overlap)

    temporal_config = None
    if fps is not None and fps > 0:
        temporal_tiling_divisor = max(1, temporal_tiling_divisor)
        tile_frames = _normalize_temporal_tiling_size(int(math.ceil(LTX2_VAE_TEMPORAL_TILING_FPS * 5.0 / temporal_tiling_divisor)))
        if tile_frames > 0:
            overlap_frames = int(round(tile_frames * 3 / 8))
            overlap_frames = _normalize_temporal_overlap(overlap_frames, tile_frames)
            temporal_config = TemporalTilingConfig(
                tile_size_in_frames=tile_frames,
                tile_overlap_in_frames=overlap_frames,
            )

    if spatial_config is None and temporal_config is None:
        return None
    return TilingConfig(spatial_config=spatial_config, temporal_config=temporal_config)


def _infer_ic_lora_downscale_factor(loras_selected) -> int | None:
    factors = []
    for lora_path in loras_selected or []:
        name = os.path.basename(str(lora_path)).lower()
        if not is_ic_lora_filename(name):
            continue
        match = re.search(r"-ref([0-9]+(?:\.[0-9]+)?)", name)
        if not match:
            factors.append(1)
            continue
        ref_ratio = float(match.group(1))
        if ref_ratio <= 0:
            factors.append(1)
            continue
        factors.append(max(1, int(round(1.0 / ref_ratio))))
    if not factors:
        return None
    return min(factors)


def _collect_video_chunks(
    video: Iterator[torch.Tensor] | torch.Tensor,
    interrupt_check: Callable[[], bool] | None = None,
    expected_frames: int | None = None,
    expected_height: int | None = None,
    expected_width: int | None = None,
) -> torch.Tensor | None:
    iterator = None
    if video is None:
        return None
    try:
        if torch.is_tensor(video):
            frames = video
            if expected_height is not None or expected_width is not None:
                frames = frames[:, :expected_height, :expected_width]
            return frames.permute(3, 0, 1, 2)
        else:
            iterator = iter(video)
            video_tensor = None
            write_pos = 0
            for chunk in iterator:
                if interrupt_check is not None and interrupt_check():
                    return None
                if chunk is None:
                    continue
                chunk = chunk if torch.is_tensor(chunk) else torch.tensor(chunk)
                if expected_height is not None or expected_width is not None:
                    chunk = chunk[:, :expected_height, :expected_width]
                if video_tensor is None:
                    channels = int(chunk.shape[-1])
                    frame_capacity = int(expected_frames) if expected_frames is not None and expected_frames > 0 else int(chunk.shape[0])
                    video_tensor = torch.empty(
                        (channels, frame_capacity, chunk.shape[1], chunk.shape[2]),
                        dtype=chunk.dtype,
                        device=chunk.device,
                    )
                frame_count = min(int(chunk.shape[0]), int(video_tensor.shape[1] - write_pos))
                if frame_count <= 0:
                    break
                video_tensor[:, write_pos : write_pos + frame_count].copy_(chunk[:frame_count].permute(3, 0, 1, 2))
                write_pos += frame_count
            if video_tensor is None:
                return None
            return video_tensor[:, :write_pos]
    finally:
        if iterator is not None:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
    # frames = frames.to(dtype=torch.float32).div_(127.5).sub_(1.0)
    # return frames.permute(3, 0, 1, 2).contiguous()


def _build_frozen_control_video(
    input_frames: torch.Tensor | None,
    input_video: torch.Tensor | None,
    frame_num: int,
    prefix_frames_count: int,
    latent_stride: int = 8,
) -> torch.Tensor:
    if input_frames is None:
        raise ValueError("LTX2 audio-from-control-video mode requires a raw Control Video.")
    requested_frames = int(frame_num)
    prefix_frames = 0
    if input_video is not None and prefix_frames_count > 0:
        prefix_frames = min(int(prefix_frames_count), int(input_video.shape[1]))
    target_frames = min(requested_frames, prefix_frames + int(input_frames.shape[1]))
    target_frames = ((target_frames - 1) // int(latent_stride)) * int(latent_stride) + 1
    pieces = []
    remaining_frames = target_frames
    if prefix_frames > 0:
        prefix = input_video[:, : min(prefix_frames, target_frames)]
        pieces.append(prefix)
        remaining_frames -= int(prefix.shape[1])
    if remaining_frames > 0:
        tail = input_frames
        if tail.shape[1] > remaining_frames:
            tail = tail[:, -remaining_frames:] if pieces else tail[:, :remaining_frames]
        pieces.append(tail)
    if not pieces:
        raise ValueError("LTX2 audio-from-control-video mode received no Control Video frames.")
    frozen_video = torch.cat(pieces, dim=1) if len(pieces) > 1 else pieces[0]
    return frozen_video[:, :target_frames]


def _apply_gamma_to_media(media_tensor: torch.Tensor | None, gamma: float) -> bool:
    if media_tensor is None or not torch.is_tensor(media_tensor) or media_tensor.dim() < 2 or gamma <= 0 or media_tensor.numel() == 0:
        return False
    exponent = 1.0 / float(gamma)
    if media_tensor.dtype == torch.uint8:
        corrected = media_tensor.to(dtype=torch.float32).div_(255.0).clamp_(0.0, 1.0).pow_(exponent)
        media_tensor.copy_(corrected.mul_(255.0).round_().clamp_(0.0, 255.0).to(dtype=torch.uint8))
        return True
    corrected = media_tensor.to(dtype=torch.float32).add_(1.0).mul_(0.5).clamp_(0.0, 1.0).pow_(exponent)
    media_tensor.copy_(corrected.mul_(2.0).sub_(1.0).to(dtype=media_tensor.dtype))
    return True


def _apply_gamma_to_video_rect(video_tensor: torch.Tensor | None, rect: tuple[int, int, int, int] | None, gamma: float) -> bool:
    if video_tensor is None or not torch.is_tensor(video_tensor) or rect is None or video_tensor.dim() < 4:
        return False
    top, bottom, left, right = rect
    region = video_tensor[..., top:bottom, left:right]
    return _apply_gamma_to_media(region, gamma)


class LTX2:
    def __init__(
        self,
        model_filename,
        model_type: str,
        base_model_type: str,
        model_def: dict,
        dtype: torch.dtype = torch.bfloat16,
        VAE_dtype: torch.dtype = torch.float32,
        text_encoder_filename: str | None = None,
        text_encoder_filepath = None,
        checkpoint_paths: dict | None = None,
    ) -> None:
        self.device = torch.device("cuda")
        self.dtype = dtype
        self.VAE_dtype = VAE_dtype
        self.base_model_type = base_model_type
        self.model_def = model_def
        self._interrupt = False
        self._hdr_scene_context = None
        self.vae = _LTX2VAEHelper()
        from .ltx_core.model.transformer import rope as rope_utils

        self.use_fp32_rope_freqs = bool(model_def.get("ltx2_rope_freqs_fp32", LTX2_USE_FP32_ROPE_FREQS))
        rope_utils.set_use_fp32_rope_freqs(self.use_fp32_rope_freqs)

        if isinstance(model_filename, (list, tuple)):
            if not model_filename:
                raise ValueError("Missing LTX-2 checkpoint path.")
            transformer_path = list(model_filename)
        else:
            transformer_path = model_filename
        component_paths = checkpoint_paths or {}
        if component_paths:
            transformer_path = component_paths.get("transformer")
            if not transformer_path:
                raise ValueError("Missing transformer path in checkpoint_paths.")

        gemma_root = text_encoder_filepath if text_encoder_filename is None else text_encoder_filename
        if not gemma_root:
            raise ValueError("Missing Gemma text encoder path.")
        pixel_spatial_upsampler = model_type.startswith("ltx2_upsampler_")
        if component_paths:
            spatial_upsampler_path = component_paths.get("spatial_upsampler")
        else:
            spatial_upsampler_path = None
        if not pixel_spatial_upsampler and not spatial_upsampler_path:
            spatial_upsampler_name = model_def.get("ltx2_spatial_upscaler_file", _SPATIAL_UPSCALER_FILENAME)
            spatial_upsampler_path = fl.locate_file(spatial_upsampler_name)

        # Internal FP8 handling is disabled; mmgp manages quantization/dtypes.
        pipeline_kind = "distilled" if model_type.startswith("ltx2_upsampler_") else model_def.get("ltx2_pipeline", "two_stage")

        pipeline_models = self._init_models(
            transformer_path=transformer_path,
            component_paths=component_paths,
            gemma_root=gemma_root,
            gemma_side_files_root=text_encoder_filepath,
            spatial_upsampler_path=spatial_upsampler_path,
        )

        if pipeline_kind == "distilled":
            self.pipeline = DistilledPipeline(
                device=self.device,
                models=pipeline_models,
            )
        else:
            self.pipeline = TI2VidTwoStagesPipeline(
                device=self.device,
                stage_1_models=pipeline_models,
                stage_2_models=pipeline_models,
            )
        self._build_diffuser_model()

    def _init_models(
        self,
        transformer_path,
        component_paths: dict,
        gemma_root: str,
        gemma_side_files_root: str | None,
        spatial_upsampler_path: str | None,
    ):
        from mmgp import offload as mmgp_offload

        fallback_config_path = component_paths.get("model_config") if component_paths else None
        base_config = _load_config_from_checkpoint(transformer_path, fallback_config_path=fallback_config_path)
        if not base_config:
            raise ValueError("Missing config in transformer checkpoint.")

        def _component_path(key: str):
            if component_paths:
                path = component_paths.get(key)
                if not path:
                    raise ValueError(f"Missing '{key}' path in checkpoint_paths.")
                return path
            return transformer_path

        def _component_config(path):
            config = _load_config_from_checkpoint(path, fallback_config_path=fallback_config_path)
            return config or base_config

        def _load_component(model, path, sd_ops=None, postprocess=None, ignore_unused_weights=False):
            if postprocess is None and sd_ops is not None:
                postprocess = _make_sd_postprocess(sd_ops)
            mmgp_offload.load_model_data(
                model,
                path,
                postprocess_sd=postprocess,
                default_dtype=self.dtype,
                writable_tensors=False,
                ignore_missing_keys=False,
                ignore_unused_weights=ignore_unused_weights,
            )
            model.eval().requires_grad_(False)
            return model

        transformer_sd_ops = LTXV_MODEL_COMFY_RENAMING_MAP
        with init_empty_weights():
            velocity_model = LTXModelConfigurator.from_config(base_config)
        velocity_model = _load_component(velocity_model, transformer_path, transformer_sd_ops, ignore_unused_weights=True)
        transformer_modules = component_paths.get("transformer_modules") if component_paths else None
        if transformer_modules:
            from .editanything import install_editanything_modules

            install_editanything_modules(velocity_model, transformer_modules, self.model_def)
        transformer = X0Model(velocity_model)
        transformer.eval().requires_grad_(False)
        VAE_URLs = self.model_def.get("VAE_URLs", None)
        video_vae_path =  fl.locate_file(VAE_URLs[0]) if VAE_URLs is not None and len(VAE_URLs) else _component_path("video_vae")
        video_config_path = component_paths.get("video_vae_config") if component_paths else None
        if video_config_path:
            with open(video_config_path, "r", encoding="utf-8") as reader:
                video_config = copy.deepcopy(json.load(reader))
        else:
            video_config = copy.deepcopy(_component_config(video_vae_path))
        diffusion_vae = video_config.get("vae", {}).get("_class_name") == "CausalDiffusionVAE"
        if diffusion_vae:
            print("[WanGP][LTX2] Loading NAD Diffusion Decoder.")
        elif self.model_def.get("ltx2_pruna_vae", False):
            print("[WanGP][LTX2] Loading PrunaAI VAE (up to x2 faster).")
        video_config_vae = video_config.setdefault("vae", {})
        video_config_vae["spatial_padding_mode"] = "reflect"
        video_config_vae["encoder_spatial_padding_mode"] = "reflect"
        video_config_vae["decoder_spatial_padding_mode"] = "reflect"
        # print("[LTX2 VAE Config] forcing encoder/decoder spatial_padding_mode=reflect")
        with init_empty_weights():
            video_encoder = VideoEncoderConfigurator.from_config(_diffusion_vae_encoder_config(video_config) if diffusion_vae else video_config)
            video_decoder = DiffusionVideoDecoder.from_config(video_config) if diffusion_vae else VideoDecoderConfigurator.from_config(video_config)
            video_vae = _VAEContainer(video_encoder, video_decoder)
        vae_postprocess = _make_diffusion_vae_postprocess("vae.") if diffusion_vae else _make_vae_postprocess("vae.")
        video_vae = _load_component(video_vae, video_vae_path, postprocess=vae_postprocess, ignore_unused_weights=True)
        video_encoder = video_vae.encoder
        video_decoder = video_vae.decoder

        audio_vae_path = _component_path("audio_vae")
        audio_config = _component_config(audio_vae_path)
        with init_empty_weights():
            audio_encoder = AudioEncoderConfigurator.from_config(audio_config)
            audio_decoder = AudioDecoderConfigurator.from_config(audio_config)
            audio_vae = _VAEContainer(audio_encoder, audio_decoder)
        audio_vae = _load_component(audio_vae, audio_vae_path, postprocess=_make_vae_postprocess("audio_vae."))
        audio_encoder = audio_vae.encoder
        audio_decoder = audio_vae.decoder

        vocoder_path = _component_path("vocoder")
        vocoder_config = _component_config(vocoder_path)
        with init_empty_weights():
            vocoder = VocoderConfigurator.from_config(vocoder_config)
        vocoder = _load_component(vocoder, vocoder_path, VOCODER_COMFY_KEYS_FILTER)

        text_projection_path = _component_path("text_embedding_projection")
        text_projection_config = _component_config(text_projection_path)
        with init_empty_weights():
            text_embedding_projection = GemmaFeaturesExtractorProjLinear.from_config(text_projection_config)
        text_embedding_projection = _load_component( text_embedding_projection, text_projection_path, TEXT_EMBEDDING_PROJECTION_KEY_OPS )

        self.split_text_connectors = "video_embeddings_connector" in component_paths
        if self.split_text_connectors:
            video_connector_path = _component_path("video_embeddings_connector")
            video_connector_config = _component_config(video_connector_path)
            with init_empty_weights():
                video_embeddings_connector = Embeddings1DConnectorConfigurator.from_config(video_connector_config)
            video_embeddings_connector = _load_component(video_embeddings_connector, video_connector_path, VIDEO_EMBEDDINGS_CONNECTOR_KEY_OPS)

            audio_connector_path = _component_path("audio_embeddings_connector")
            audio_connector_config = _component_config(audio_connector_path)
            with init_empty_weights():
                audio_embeddings_connector = AudioEmbeddings1DConnectorConfigurator.from_config(audio_connector_config)
            audio_embeddings_connector = _load_component(audio_embeddings_connector, audio_connector_path, AUDIO_EMBEDDINGS_CONNECTOR_KEY_OPS)
            text_embeddings_connector = GemmaTextEmbeddingsConnectorModel(video_embeddings_connector, audio_embeddings_connector)
        else:
            text_connector_path = _component_path("text_embeddings_connector")
            text_connector_config = _component_config(text_connector_path)
            with init_empty_weights():
                text_embeddings_connector = GemmaTextEmbeddingsConnectorModelConfigurator.from_config(text_connector_config)
            text_embeddings_connector = _load_component(text_embeddings_connector, text_connector_path, TEXT_EMBEDDINGS_CONNECTOR_KEY_OPS)

        text_encoder = build_gemma_text_encoder(
            gemma_root,
            default_dtype=self.dtype,
            side_files_root=gemma_side_files_root,
        )
        text_encoder.eval().requires_grad_(False)

        spatial_upsampler = None
        if spatial_upsampler_path:
            upsampler_config = _load_config_from_checkpoint(spatial_upsampler_path)
            with init_empty_weights():
                spatial_upsampler = LatentUpsamplerConfigurator.from_config(upsampler_config)
            spatial_upsampler = _load_component(spatial_upsampler, spatial_upsampler_path, None)

        self.text_encoder = text_encoder
        self.text_embedding_projection = text_embedding_projection
        self.text_embeddings_connector = text_embeddings_connector
        self.video_embeddings_connector = text_embeddings_connector.video_embeddings_connector
        self.audio_embeddings_connector = text_embeddings_connector.audio_embeddings_connector
        self.video_encoder = video_encoder
        self.video_decoder = video_decoder
        self.audio_encoder = audio_encoder
        self.audio_decoder = audio_decoder
        self.vocoder = vocoder
        self.spatial_upsampler = spatial_upsampler
        self.model = transformer
        self.model2 = None

        return types.SimpleNamespace(
            text_encoder=self.text_encoder,
            text_embedding_projection=self.text_embedding_projection,
            text_embeddings_connector=self.text_embeddings_connector,
            video_encoder=self.video_encoder,
            video_decoder=self.video_decoder,
            audio_encoder=self.audio_encoder,
            audio_decoder=self.audio_decoder,
            vocoder=self.vocoder,
            spatial_upsampler=self.spatial_upsampler,
            transformer=self.model,
        )

    def _load_hdr_scene_context(self, lora_dir: str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        cached = self._hdr_scene_context
        if cached is not None:
            return cached
        path = fl.locate_file(self.model_def.get("ltx2_hdr_scene_embeddings_file", ""), error_if_none=False)
        tensors = load_file(path, device="cpu")
        # LTX-2.3 stores (1, L, D) contexts, LTX-2.5 (L, D).
        self._hdr_scene_context = tuple(tensors[name].detach().cpu().reshape(1, *tensors[name].shape[-2:]) for name in ("video_context", "audio_context"))
        return self._hdr_scene_context

    def _detach_text_encoder_connectors(self) -> None:
        text_encoder = getattr(self, "text_encoder", None)
        if text_encoder is None:
            return
        connectors = {}
        feature_extractor = getattr(self, "text_embedding_projection", None)
        video_connector = getattr(self, "video_embeddings_connector", None)
        audio_connector = getattr(self, "audio_embeddings_connector", None)
        if feature_extractor is not None:
            connectors["feature_extractor_linear"] = feature_extractor
        if video_connector is not None:
            connectors["embeddings_connector"] = video_connector
        if audio_connector is not None:
            connectors["audio_embeddings_connector"] = audio_connector
        if not connectors:
            return
        for name, module in connectors.items():
            if name in text_encoder._modules:
                del text_encoder._modules[name]
            setattr(text_encoder, name, _ExternalConnectorWrapper(module))
        self._text_connectors = connectors

    def _build_diffuser_model(self) -> None:
        self._detach_text_encoder_connectors()
        self.diffuser_model = LTX2SuperModel(self)
        _attach_lora_preprocessor(self.diffuser_model)


    def get_trans_lora(self):
        trans = getattr(self, "diffuser_model", None)
        if trans is None:
            trans = self.model
        return trans, None

    @generation_progress
    def upscale_video(
        self,
        sample: torch.Tensor,
        prompt: str,
        negative_prompt: str,
        audio_waveform,
        audio_sample_rate: int,
        upsampler_variant: str,
        seed: int,
        fps: float,
        pixel_frame_offset: int = 0,
        VAE_tile_size=None,
        callback=None,
        set_progress_status=None,
        interrupt_check=None,
    ) -> torch.Tensor | None:
        if not isinstance(self.pipeline, DistilledPipeline):
            raise RuntimeError("LTX video upsampling requires a distilled checkpoint")
        channels, frame_count, source_height, source_width = map(int, sample.shape)
        if channels != 3 or frame_count > LTX2_UPSCALE_MAX_FRAMES:
            raise ValueError(f"LTX video upsampling expects RGB windows of at most {LTX2_UPSCALE_MAX_FRAMES} frames")
        was_uint8 = sample.dtype == torch.uint8
        source = sample.to(dtype=torch.float32)
        source = source.div_(127.5).sub_(1.0) if was_uint8 else source.clamp_(-1.0, 1.0)
        source = source.unsqueeze(0)
        source = source.to(device=self.device, dtype=torch.bfloat16)
        tiling_config = _build_tiling_config(VAE_tile_size, fps)
        if set_progress_status is not None:
            set_progress_status("Audio VAE encoding")
        waveform = torch.as_tensor(audio_waveform, dtype=torch.float32)
        waveform = waveform.unsqueeze(1) if waveform.ndim == 1 else waveform
        waveform = waveform.T.unsqueeze(0)
        audio_encoder = self.pipeline._get_model("audio_encoder")
        target_channels = int(audio_encoder.in_channels)
        if waveform.shape[1] != target_channels:
            if waveform.shape[1] == 1:
                waveform = waveform.repeat(1, target_channels, 1)
            elif target_channels == 1:
                waveform = waveform.mean(dim=1, keepdim=True)
            else:
                waveform = waveform[:, :target_channels]
                if waveform.shape[1] < target_channels:
                    waveform = torch.cat((waveform, waveform.new_zeros(waveform.shape[0], target_channels - waveform.shape[1], waveform.shape[2])), dim=1)
        audio_processor = AudioProcessor(sample_rate=audio_encoder.sample_rate, mel_bins=audio_encoder.mel_bins, mel_hop_length=audio_encoder.mel_hop_length, n_fft=audio_encoder.n_fft, device=self.device)
        mel = audio_processor.waveform_to_mel(waveform, int(audio_sample_rate))
        audio_params = next(audio_encoder.parameters())
        audio_latent = audio_encoder(mel.to(device=audio_params.device, dtype=audio_params.dtype))
        decoded = self.pipeline.upscale_video(
            source_video=source,
            prompt=prompt,
            negative_prompt=negative_prompt,
            audio_latent=audio_latent,
            upsampler_variant=upsampler_variant,
            seed=seed,
            frame_rate=fps,
            sigma_values=LTX2_UPSCALE_SIGMAS,
            latent_frame_offset=int(pixel_frame_offset) // 8,
            pixel_frame_offset=pixel_frame_offset,
            tiling_config=tiling_config,
            callback=callback,
            set_progress_status=set_progress_status,
            interrupt_check=interrupt_check,
        )
        if decoded is None:
            return None
        decoded = decoded[:frame_count, :source_height * 2, :source_width * 2]
        return decoded.permute(3, 0, 1, 2).contiguous()

    @generation_progress
    def refine_video(
        self,
        load_window,
        frame_count: int,
        height: int,
        width: int,
        output_frames: int,
        output_height: int,
        output_width: int,
        prompt: str,
        seed: int,
        fps: float,
        window_frames: int,
        tile_height: int,
        tile_width: int,
        gentle: bool,
        tiles_per_call: int = 0,
        conditioning_latent: torch.Tensor | None = None,
        carry_frames: int = 0,
        fusion_previous: torch.Tensor | None = None,
        fusion_export: slice | None = None,
        pixel_frame_offset: int = 0,
        VAE_tile_size=None,
        callback=None,
        set_progress_status=None,
        interrupt_check=None,
    ) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        if not isinstance(self.pipeline, DistilledPipeline):
            raise RuntimeError("LTX detail refinement requires a distilled checkpoint")
        if frame_count > LTX2_UPSCALE_MAX_FRAMES:
            raise ValueError(f"LTX detail refinement expects at most {LTX2_UPSCALE_MAX_FRAMES} frames per call")
        decoded = self.pipeline.refine_video(
            load_window=load_window,
            frame_count=frame_count,
            height=height,
            width=width,
            output_frames=output_frames,
            output_height=output_height,
            output_width=output_width,
            prompt=prompt,
            seed=seed,
            frame_rate=fps,
            window_frames=window_frames,
            tile_height=tile_height,
            tile_width=tile_width,
            gentle=gentle,
            tiles_per_call=tiles_per_call,
            conditioning_latent=conditioning_latent,
            carry_frames=carry_frames,
            fusion_previous=fusion_previous,
            fusion_export=fusion_export,
            latent_frame_offset=int(pixel_frame_offset) // 8,
            tiling_config=_build_tiling_config(VAE_tile_size, fps),
            callback=callback,
            set_progress_status=set_progress_status,
            interrupt_check=interrupt_check,
        )
        return None if decoded is None else (decoded[0].permute(3, 0, 1, 2), decoded[1])

    def get_loras_transformer(self, get_model_recursive_prop, model_type, video_prompt_type, base_model_type=None, model_def = None, lora_dir = None, sample_solver = None, **kwargs):
        control_map = {
            "O": "pose_align",
            "P": "pose",
            "D": "depth",
            "E": "canny",
        }
        from shared.utils.utils import get_outpainting_dims

        loras = []
        loras_mult = []
        guidance_phases = max(1, int(kwargs["guidance_phases"]))
        audio_prompt_type = kwargs["audio_prompt_type"]
        outpainting_ratio = kwargs["video_guide_outpainting_ratio"].strip()
        outpainting_setting = str(kwargs["video_guide_outpainting"])
        pipeline_kind = model_def.get("ltx2_pipeline", "two_stage")
        resolved_base_model_type = base_model_type
        sample_solver = (sample_solver or "").lower()
        selected_loras = {os.path.basename(lora).lower() for lora in kwargs.get("activated_loras", [])}
        preload_urls = get_model_recursive_prop(model_type, "preload_URLs", return_list=True)
        if isinstance(preload_urls, str):
            preload_urls = [preload_urls]

        def _get_preload_lora_url(signature):
            matched_url = None
            for entry in preload_urls:
                if isinstance(entry, str) and entry.endswith("|%lora_dir"):
                    source_url = entry.split("|", 1)[0]
                    if signature in os.path.basename(source_url).lower():
                        matched_url = source_url
            return matched_url

        def _append_system_lora(name, multiplier, signature):
            signature = signature.lower()
            url = _get_preload_lora_url(signature) or model_def.get(f"ltx2_lora_{name}", "")
            if not url:
                return
            if any(signature in os.path.basename(lora).lower() for lora in loras):
                return
            for lora in selected_loras:
                if signature in lora:
                    print(f"Default system '{signature}' lora and corresponding multiplier will be ignored as User has provided its own lora ({lora})")
                    return
            loras.append(url)
            loras_mult.append(multiplier)

        distilled_samplers = {"distilled_8_steps", "distilled_8_steps_ancestral"}
        hdr_enabled = _ltx2_main_loras_compatible(resolved_base_model_type) and VIDEO_PROMPT_HDR_OUTPUT_FLAG in video_prompt_type
        hdr_conversion = hdr_enabled and model_def["ltx2_hdr_transform"] == "acescct"
        if pipeline_kind != "distilled" and hdr_conversion:
            # The LTX-2.5 SDR-to-HDR IC-LoRA runs on the distilled model: one phase of the 8 distilled steps.
            _append_system_lora("distilled", 1.0, "distilled-lora")
        elif pipeline_kind != "distilled" and (guidance_phases > 1 or sample_solver in distilled_samplers | {"res2s"}):
            use_hq_sampler = sample_solver == "res2s"
            use_distilled_8_steps = sample_solver in distilled_samplers
            use_id_lora = "1" in audio_prompt_type
            if guidance_phases == 1 and use_hq_sampler:
                mult = 0.2
            elif guidance_phases == 1 and use_distilled_8_steps:
                mult = 0.5
            elif use_hq_sampler:
                mult = "0.25;0.5"
            elif use_id_lora:
                mult = "0;0.8"
            elif use_distilled_8_steps:
                mult = "0.5;0.5"
            else:
                mult = "0;1"
            distilled_lora_name = "distilled_1_1" if resolved_base_model_type == "ltx2_22B" and sample_solver in distilled_samplers | {"res2s"} else "distilled"
            _append_system_lora(distilled_lora_name, mult, "distilled-lora")
        if hdr_enabled:
            _append_system_lora("hdr", 1.0, "ic-lora-sdr-to-hdr" if hdr_conversion else "ic-lora-hdr")
        if any(letter in video_prompt_type for letter in control_map):
            _append_system_lora("union_control", 1.0, "union-control")
        any_outpainting = get_outpainting_dims(outpainting_setting, outpainting_ratio) is not None
        if _ltx2_main_loras_compatible(resolved_base_model_type) and any_outpainting and LTX2_OUTPAINTING_METHOD == 1:
            _append_system_lora("outpaint", 1.0, "ic-lora-outpaint")
        if _ltx2_main_loras_compatible(resolved_base_model_type) and (_ltx2_inpainting_enabled(video_prompt_type) or any_outpainting and LTX2_OUTPAINTING_METHOD == 2):
            _append_system_lora("inpaint", 1.0, "in-outpainting")
        if not (model_def.get("ltx2_msr", False) or model_def.get("ltx2_layout_to_render", False)) and _ltx2_main_ingredients_enabled(resolved_base_model_type, video_prompt_type):
            _append_system_lora("ingredients", 1.4, "ic-lora-ingredients")
        if "1" in audio_prompt_type:
            _append_system_lora("id", 1.0 if guidance_phases == 1 else "1;0", "id-lora-celebvhq")
        return loras, loras_mult

    @generation_progress
    @torch.inference_mode()
    def generate(
        self,
        input_prompt: str,
        n_prompt: str | None = None,
        image_start=None,
        image_end=None,
        image_end_frame_position: int | None = None,
        sampling_steps: int = 40,
        guide_scale: float = 4.0,
        alt_guide_scale: float = 1.0,
        input_video=None,
        prefix_frames_count: int = 0,
        window_no: int = 1,
        input_frames=None,
        input_frames2=None,
        frames_to_inject = None,
        input_masks=None,
        input_masks2=None,
        frames_relative_positions_list=None,
        masking_strength: float | None = None,
        input_video_strength: float | None = None,
        return_latent_slice=None,
        video_prompt_type: str = "",
        audio_prompt_type: str = "",
        denoising_strength: float | None = None,
        cfg_star_switch: int = 0,
        apg_switch: int = 0,
        perturbation_switch: int = 0,
        perturbation_layers: list[int] | None = None,
        perturbation_start: float = 0.0,
        perturbation_end: float = 1.0,
        audio_cfg_scale: float | None = None,
        alt_scale: float = 0.0,
        sample_solver: str = "",
        custom_sigmas=None,
        NAG_scale: float = 1.0,
        NAG_tau: float = 3.5,
        NAG_alpha: float = 0.5,
        self_refiner_setting: int = 0,
        self_refiner_plan: str = "",
        self_refiner_f_uncertainty: float = 0.1,
        self_refiner_certain_percentage: float = 0.999,
        loras_slists=None,
        loras_selected=None,
        text_connectors=None,
        input_ref_images=None,
        input_waveform=None,
        input_waveform_sample_rate=None,
        audio_conditioning_end=None,
        audio_scale: float | None = None,
        outpainting_dims: list[int] | None = None,
        frame_num: int = 121,
        image_mode: int = 0,
        height: int = 1024,
        width: int = 1536,
        fps: float = 24.0,
        seed: int = 0,
        callback=None,
        set_progress_status=None,
        VAE_tile_size=None,
        guide_phases= 1,
        custom_settings=None,
        video_guide=None,
        frame_window_options=None,
        gen_state=None,
        input_video_is_hdr: bool = False,
        lora_dir: str | None = None,
        **kwargs,
    ):
        if self._interrupt:
            return None
        # Hand the "Start Tau" (Attention Sparsity) setting to the Sol-Attn policy.
        attention_sparsity = kwargs.get("attention_sparsity")
        if attention_sparsity not in (None, ""):
            try:
                offload.shared_state[SOL_ATTN_TAU_START_KEY] = float(attention_sparsity)
            except (TypeError, ValueError):
                offload.shared_state.pop(SOL_ATTN_TAU_START_KEY, None)
        else:
            offload.shared_state.pop(SOL_ATTN_TAU_START_KEY, None)
        joyai_context = None
        joyai_memory_bank = None
        joyai_store_mem_selectors = []
        if self.model_def.get("joyai_echo", False):
            from .joyai_echo import prepare_joyai_echo_context
            joyai_context, joyai_memory_bank, joyai_store_mem_selectors, clear_joyai_control_inputs = prepare_joyai_echo_context(self, gen_state, custom_settings, video_guide, frame_window_options, fps, video_prompt_type, height, width, guide_phases, VAE_tile_size, window_no)
            if clear_joyai_control_inputs:
                input_frames = input_frames2 = input_masks = input_masks2 = None
                video_prompt_type = ""
        distill = self.model_def.get("ltx2_pipeline", "two_stage") == "distilled"
        editanything = _is_editanything_model(self.model_def)
        msr = self.model_def.get("ltx2_msr", False)
        alpha_gen = self.model_def.get("ltx2_alpha_gen", False)
        layout_to_render = self.model_def.get("ltx2_layout_to_render", False)
        alpha_source = None
        if alpha_gen:
            alpha_source = input_frames
            height, width = input_frames.shape[-2:]
            if max(height, width) > 1920 or min(height, width) > 1088:
                raise ValueError("Alpha Gen supports source videos up to 1920Ã—1088. Resize the source before loading it.")
            frame_num = min(frame_num, input_frames.shape[1])
            input_prompt = ""
            guide_phases = 1
            denoising_strength, masking_strength = 1.0, 0.0
            input_frames = torch.nn.functional.pad(input_frames, (0, -width % 32, 0, -height % 32), mode="replicate")
        ltx25 = self.base_model_type in {"ltx2_25_22B", "ltx2_25_22B_msr", "ltx2_25_22B_alpha_gen", "ltx2_25_22B_layout_to_render"}
        main_ingredients = not (msr or layout_to_render) and _ltx2_main_ingredients_enabled(self.base_model_type, video_prompt_type)
        output_frame_num = frame_num
        msr_frame_count = 0
        if msr and "I" in video_prompt_type and input_ref_images is not None:
            msr_frame_count = _resolve_msr_reference_frame_count(
                self.model_def,
                custom_settings,
                input_ref_images,
                "K" in video_prompt_type,
            )
            if not ltx25:
                frame_num = max(frame_num, msr_frame_count)

        hdr_enabled = _ltx2_main_loras_compatible(self.base_model_type) and VIDEO_PROMPT_HDR_OUTPUT_FLAG in video_prompt_type
        hdr_transform = self.model_def["ltx2_hdr_transform"] if hdr_enabled else None
        input_video_is_hdr = bool(input_video_is_hdr)
        hdr_scene_context = self._load_hdr_scene_context(lora_dir) if hdr_enabled else None
        if hdr_enabled or alpha_gen:
            NAG_scale = 1.0
            audio_prompt_type = ""
            input_waveform = None
        audio_from_control_video = "2" in audio_prompt_type
        image_start = _coerce_image_list(image_start)
        image_end = _coerce_image_list(image_end)
        if frames_to_inject is None:
            frames_to_inject = []
        if frames_relative_positions_list is None:
            frames_relative_positions_list = []
        elif isinstance(frames_relative_positions_list, (list, tuple)):
            frames_relative_positions_list = list(frames_relative_positions_list)
        else:
            frames_relative_positions_list = [frames_relative_positions_list]
        if image_start is None:
            new_frames_to_inject = []
            new_frames_relative_positions_list = []
            for frame_to_inject, frame_relative_position in zip(frames_to_inject,frames_relative_positions_list):
                if frame_relative_position == 0:
                    image_start = frame_to_inject
                else:
                    new_frames_to_inject.append(frame_to_inject)
                    new_frames_relative_positions_list.append(frame_relative_position)
            frames_to_inject = new_frames_to_inject 
            frames_relative_positions_list = new_frames_relative_positions_list

        outpainting_dims = _normalize_outpainting_dims(outpainting_dims)
        any_outpainting = outpainting_dims is not None and "V" in video_prompt_type
        ltx2_inpainting = _ltx2_inpainting_enabled(video_prompt_type)
        new_outpainting = any_outpainting and LTX2_OUTPAINTING_METHOD == 2
        self_refiner_max_plans = self.model_def.get("self_refiner_max_plans", 1)
        requested_outpaint_gamma_roundtrip = _ltx2_main_loras_compatible(self.base_model_type) and any_outpainting and LTX2_OUTPAINTING_METHOD == 1
        if hdr_enabled:
            requested_outpaint_gamma_roundtrip = False
        if any_outpainting and LTX2_OUTPAINTING_METHOD == 1:
            guide_phases = 1
        if new_outpainting:
            input_masks = _merge_ltx2_masks(input_masks, _build_outpainting_mask_cthw(input_frames, outpainting_dims))
            input_masks2 = _merge_ltx2_masks(input_masks2, _build_outpainting_mask_cthw(input_frames2, outpainting_dims))
        if ltx2_inpainting:
            input_frames, input_masks = _apply_ltx2_inpaint_preprocess_dilation(input_frames, input_masks, LTX2_INPAINTING_PREPROCESS_MASK_DILATION)
            input_frames2, input_masks2 = _apply_ltx2_inpaint_preprocess_dilation(input_frames2, input_masks2, LTX2_INPAINTING_PREPROCESS_MASK_DILATION)
        use_outpaint_gamma_roundtrip = False
        latent_stride = 8
        if hasattr(self.pipeline, "pipeline_components"):
            scale_factors = getattr(self.pipeline.pipeline_components, "video_scale_factors", None)
            if scale_factors is not None:
                latent_stride = int(getattr(scale_factors, "time", scale_factors[0]))
        if image_mode > 0 and "V" in video_prompt_type and any(letter in video_prompt_type for letter in "PODE") and ((int(frame_num) - 1) // latent_stride + 1) <= 1:
            frame_num = latent_stride + 1
            print(f"[WAN2GP][LTX2] Expanding image pose/depth/edge control from one latent to two latents ({frame_num} frames) to allow denoised image generation.")

        input_video_strength = max(0.0, min(1.0, input_video_strength))
        if requested_outpaint_gamma_roundtrip:
            conditioning_gamma_applied = _apply_gamma_to_media(image_start, LTX2_OUTPAINT_GAMMA)
            conditioning_gamma_applied = _apply_gamma_to_media(image_end, LTX2_OUTPAINT_GAMMA) or conditioning_gamma_applied
            if torch.is_tensor(input_video) and prefix_frames_count > 0:
                conditioning_gamma_applied = _apply_gamma_to_media(input_video[:, :prefix_frames_count], LTX2_OUTPAINT_GAMMA) or conditioning_gamma_applied
            for ref_image in frames_to_inject:
                conditioning_gamma_applied = _apply_gamma_to_media(ref_image, LTX2_OUTPAINT_GAMMA) or conditioning_gamma_applied
            if conditioning_gamma_applied:
                print("[WAN2GP][LTX2] Applying full-frame gamma preprocessing for outpainting IC-LoRA conditioning images.")
                use_outpaint_gamma_roundtrip = True
        if "G" not in video_prompt_type:
            denoising_strength = 1.0
            masking_strength = 0.0
        if ltx2_inpainting or new_outpainting:
            masking_strength = 0.0
        if hdr_enabled and torch.is_tensor(input_video) and (input_video_is_hdr or hdr_transform == "acescct"):
            to_working_space = hdr_linear_to_vae_range if input_video_is_hdr else sdr_to_vae_range
            input_video = to_working_space(input_video, transform=hdr_transform, channel_dim=0).to(dtype=input_video.dtype)
        control_strength = denoising_strength
        ic_lora_downscale_factor = None
        ic_lora_downscale_factor = _infer_ic_lora_downscale_factor(loras_selected)
        video_conditioning_downscale_factor = ic_lora_downscale_factor or 1
        if video_conditioning_downscale_factor > 1 and ((int(frame_num) - 1) // latent_stride + 1) <= 1:
            print("[WAN2GP][LTX2] Disabling downscaled control conditioning for single-latent-frame generation.")
            video_conditioning_downscale_factor = 1
         # merge_conditioning_and_guide = False
        has_prefix_frames = input_video is not None 
        is_start_image_only = image_start is not None and (not has_prefix_frames or prefix_frames_count <= 1)
        merge_conditioning_and_guide = continuous_conditioning_and_guide = False
        video_conditioning = None
        frozen_control_video = None
        masking_source = None
        if input_frames is not None or input_frames2 is not None:
            if audio_from_control_video:
                frozen_control_video = _build_frozen_control_video(input_frames, input_video, frame_num, prefix_frames_count, latent_stride)
                frame_num = int(frozen_control_video.shape[1])
            else:
                # continuous_conditioning_and_guide = has_prefix_frames and (ic_lora_downscale_factor or 1) == 1 and not is_start_image_only
                # merge_conditioning_and_guide = has_prefix_frames and any_outpainting
                continuous_conditioning_and_guide = has_prefix_frames and any_outpainting
                skip_first_guide_latent = has_prefix_frames and (not is_start_image_only) and not (merge_conditioning_and_guide or continuous_conditioning_and_guide)
                if requested_outpaint_gamma_roundtrip:
                    control_tensor = input_frames if input_frames is not None else input_frames2
                    control_rect = None if control_tensor is None else _get_outpainting_inner_rect(control_tensor.shape[-2], control_tensor.shape[-1], outpainting_dims)
                    if control_rect is not None and _apply_gamma_to_video_rect(control_tensor, control_rect, LTX2_OUTPAINT_GAMMA):
                        print("[WAN2GP][LTX2] Applying preserved-area gamma preprocessing for outpainting IC-LoRA control video.")
                        use_outpaint_gamma_roundtrip = True

                control_start_frame = prefix_frames_count
                if merge_conditioning_and_guide or continuous_conditioning_and_guide:
                    if prefix_frames_count == 1:
                        input_frames[:, 0] = input_video[:, 0]
                        if new_outpainting and input_masks is not None:
                            input_masks[:, 0] = 0
                    else:
                        input_frames = torch.concat( [input_video[:, :prefix_frames_count],  input_frames[:, 1:]], dim=1)
                        if new_outpainting and input_masks is not None:
                            input_masks = torch.concat([torch.zeros_like(input_video[:1, :prefix_frames_count], dtype=input_masks.dtype, device=input_masks.device), input_masks[:, 1:]], dim=1)
                    if continuous_conditioning_and_guide:
                        control_start_frame = -prefix_frames_count
                    else:
                        prefix_frames_count = 0
                        control_start_frame =  0
                    input_video = None
                elif skip_first_guide_latent:
                    control_start_frame = -prefix_frames_count

                if LTX2_PAD_MASKED_CONTROL_VIDEO_TAIL and new_outpainting:
                    target_control_frames = int(frame_num) - int(control_start_frame) if control_start_frame > 0 else int(frame_num)
                    input_frames, input_masks = _pad_ltx2_masked_control_video_tail(input_frames, input_masks, target_control_frames)
                    input_frames2, input_masks2 = _pad_ltx2_masked_control_video_tail(input_frames2, input_masks2, target_control_frames)

                conditioning_entries = []
                for control_frames in (input_frames, input_frames2):
                    if control_frames is not None:
                        conditioning_entries.append((UInt8Guide(control_frames) if control_frames.dtype == torch.uint8 else control_frames, control_start_frame, control_strength))
                if conditioning_entries:
                    video_conditioning = conditioning_entries
                if masking_strength > 0.0:
                    if input_masks is not None and input_frames is not None:
                        masking_source = {
                            "video": UInt8Guide(input_frames) if input_frames.dtype == torch.uint8 else input_frames,
                            "mask": input_masks,
                            "start_frame": control_start_frame,
                        }
                    elif input_masks2 is not None and input_frames2 is not None:
                        masking_source = {
                            "video": UInt8Guide(input_frames2) if input_frames2.dtype == torch.uint8 else input_frames2,
                            "mask": input_masks2,
                            "start_frame": control_start_frame,
                        }

        force_stage2_ref_video_conditioning = False
        if msr and "I" in video_prompt_type and input_ref_images is not None:
            if ltx25:
                from .msr import MSRReferenceImages, load_slot_embeddings

                slots = load_slot_embeddings(fl.locate_file(self.model_def["ltx2_msr_slot_embeddings_file"]))
                refs = list(input_ref_images) if isinstance(input_ref_images, (list, tuple)) else [input_ref_images]
                if "K" in video_prompt_type:
                    refs = refs[1:] + refs[:1]
                frames = [_msr_ref_image_to_frame(ref, int(width), int(height)) for ref in refs]
                video_conditioning = MSRReferenceImages(frames, slots, msr_frame_count, control_strength)
            else:
                ref_video = _build_msr_reference_video(
                    input_ref_images,
                    msr_frame_count,
                    int(width),
                    int(height),
                    "K" in video_prompt_type,
                    _is_msr_v2_model_def(self.model_def),
                )
                if video_conditioning is None:
                    video_conditioning = []
                video_conditioning.append((ref_video, 0, control_strength))
            force_stage2_ref_video_conditioning = True
        elif main_ingredients and "I" in video_prompt_type and input_ref_images is not None:
            ref_frame_count = int(frame_num)
            ref_video = _duplicate_ref_image_as_video(input_ref_images, ref_frame_count)
            if ref_video is not None:
                if video_conditioning is None:
                    video_conditioning = []
                video_conditioning.append((ref_video, 0, control_strength))
                force_stage2_ref_video_conditioning = True
        elif not (editanything or layout_to_render) and "I" in video_prompt_type and "F" not in video_prompt_type and "K" not in video_prompt_type and input_ref_images is not None:
            ref_frame_count = self.model_def.get("ltx2_ic_lora_ref_video_frames", 1)
            ref_video = _duplicate_ref_image_as_video(input_ref_images, ref_frame_count)
            if ref_video is not None:
                if video_conditioning is None:
                    video_conditioning = []
                video_conditioning.append((ref_video, 0, control_strength))

        latent_conditioning_stage2 = None

        images = []
        guiding_images = []
        guiding_images_stage2 = []
        images_stage2 = []
        if layout_to_render:
            # The appearance is an appended keyframe at -1, never a replacement
            # of the first generated latent. Both passes receive the same guide.
            entry = (input_ref_images[0], -1, input_video_strength, "lanczos", True)
            guiding_images.append(entry)
            guiding_images_stage2.append(entry)
            force_stage2_ref_video_conditioning = True
        stage2_override = False

        def _append_prefix_entries(target_list, extra_list=None):
            if input_video is None or is_start_image_only:
                return
            frame_count = min(prefix_frames_count, input_video.shape[1])
            if frame_count <= 0:
                return
            entry = (input_video[:, :frame_count].permute(1, 2, 3, 0), 0, input_video_strength)
            target_list.append(entry)
            if extra_list is not None:
                extra_list.append(entry)

        def _append_injected_ref_entries(target_list, extra_list=None):
            injected_ref_count = min(len(frames_to_inject), len(frames_relative_positions_list))
            for ref_image, frame_idx in zip(frames_to_inject[:injected_ref_count], frames_relative_positions_list[:injected_ref_count]):
                entry = (ref_image, int(frame_idx), input_video_strength, "lanczos")
                target_list.append(entry)
                if extra_list is not None:
                    extra_list.append(entry)

        if image_start is None:
            _append_prefix_entries(images, images_stage2)
        else:
            entry = (image_start, _to_latent_index(0, latent_stride), input_video_strength, "lanczos")
            images.append(entry)
            images_stage2.append(entry)

        if image_end is not None:
            entry = (image_end, output_frame_num - 1 if image_end_frame_position is None else int(image_end_frame_position), input_video_strength)
            guiding_images.append(entry)
            guiding_images_stage2.append(entry)

        _append_injected_ref_entries(guiding_images, guiding_images_stage2)

        tiling_config = _build_tiling_config(VAE_tile_size, fps)
        def interrupt_check():
            if callback is not None:
                callback(-1, None, False, True)
            return self._interrupt
        text_connectors = text_connectors or getattr(self, "_text_connectors", None)
        editanything_ref_images = input_ref_images if editanything else None

        audio_conditionings = None
        audio_conditionings_stage2 = None
        audio_identity_guidance_scale = 0.0
        if input_waveform is not None:
            if audio_scale is None:
                audio_scale = 1.0
            audio_strength = max(0.0, min(1.0, float(audio_scale)))
            if audio_strength > 0.0:
                if self._interrupt:
                    return None
                waveform, waveform_sample_rate =  torch.from_numpy(input_waveform), input_waveform_sample_rate
                if self._interrupt:
                    return None
                if waveform.ndim == 1:
                    waveform = waveform.unsqueeze(0).unsqueeze(0)
                elif waveform.ndim == 2:
                    waveform = waveform.T.unsqueeze(0)
                target_channels = int(getattr(self.audio_encoder, "in_channels", waveform.shape[1]))
                if target_channels <= 0:
                    target_channels = waveform.shape[1]
                if waveform.shape[1] != target_channels:
                    if waveform.shape[1] == 1 and target_channels > 1:
                        waveform = waveform.repeat(1, target_channels, 1)
                    elif target_channels == 1:
                        waveform = waveform.mean(dim=1, keepdim=True)
                    else:
                        waveform = waveform[:, :target_channels, :]
                        if waveform.shape[1] < target_channels:
                            pad_channels = target_channels - waveform.shape[1]
                            pad = waveform.new_zeros((waveform.shape[0], pad_channels, waveform.shape[2]))
                            waveform = torch.cat([waveform, pad], dim=1)

                waveform = waveform.to(device="cpu", dtype=torch.float32)
                if "1" in audio_prompt_type:
                    max_samples = int(round(float(waveform_sample_rate) * LTX2_ID_LORA_MAX_REFERENCE_SECONDS))
                    waveform = waveform[:, :, :max_samples]
                audio_processor = AudioProcessor(
                    sample_rate=self.audio_encoder.sample_rate,
                    mel_bins=self.audio_encoder.mel_bins,
                    mel_hop_length=self.audio_encoder.mel_hop_length,
                    n_fft=self.audio_encoder.n_fft,
                    device=self.device,
                )
                skip_audio_conditioning = False
                waveform_sample_rate = int(waveform_sample_rate or 0)
                input_samples = int(waveform.shape[-1])
                if "1" not in audio_prompt_type and audio_processor.waveform_too_short_for_mel(waveform, waveform_sample_rate):
                    print(f"[WAN2GP][LTX2] Audio conditioning is too short for mel encoding ({input_samples} samples at {waveform_sample_rate} Hz); disabling it so audio frames are denoised.")
                    skip_audio_conditioning = True
                if not skip_audio_conditioning:
                    audio_processor = audio_processor.to(waveform.device)
                    mel = audio_processor.waveform_to_mel(waveform, waveform_sample_rate)
                    if self._interrupt:
                        return None
                    audio_params = next(self.audio_encoder.parameters(), None)
                    audio_device = audio_params.device if audio_params is not None else self.device
                    audio_dtype = audio_params.dtype if audio_params is not None else self.dtype
                    mel = mel.to(device=audio_device, dtype=audio_dtype)
                    with torch.inference_mode():
                        audio_latent = self.audio_encoder(mel)
                    if self._interrupt:
                        return None
                    audio_downsample = getattr(
                        getattr(self.audio_encoder, "patchifier", None),
                        "audio_latent_downsample_factor",
                        4,
                    )
                    audio_latent = audio_latent.to(device=self.device, dtype=self.dtype)
                    if audio_conditioning_end is not None and "1" not in audio_prompt_type:
                        # Keep the reliable prefix fixed; the remaining target tokens retain their noise.
                        token_seconds = self.audio_encoder.mel_hop_length * audio_downsample / self.audio_encoder.sample_rate
                        prefix_tokens = max(0, math.floor(audio_conditioning_end / token_seconds + 1e-9))
                        audio_latent = audio_latent[:, :, :prefix_tokens, :]
                    if "1" in audio_prompt_type:
                        audio_conditionings = [AudioConditionByReferenceLatent(audio_latent)]
                        audio_conditionings_stage2 = []
                        audio_identity_guidance_scale = LTX2_ID_LORA_GUIDANCE_SCALE
                    else:
                        target_shape = AudioLatentShape.from_video_pixel_shape(
                            VideoPixelShape(
                                batch=audio_latent.shape[0],
                                frames=int(frame_num),
                                width=1,
                                height=1,
                                fps=float(fps),
                            ),
                            channels=audio_latent.shape[1],
                            mel_bins=audio_latent.shape[3],
                            sample_rate=self.audio_encoder.sample_rate,
                            hop_length=self.audio_encoder.mel_hop_length,
                            audio_latent_downsample_factor=audio_downsample,
                        )
                        target_frames = target_shape.frames
                        if audio_latent.shape[2] == 0:
                            audio_conditionings = None
                        elif audio_latent.shape[2] < target_frames:
                            audio_conditionings = [AudioConditionByLatentPrefix(audio_latent)]
                        else:
                            if audio_latent.shape[2] > target_frames:
                                audio_latent = audio_latent[:, :, :target_frames, :]
                            audio_conditionings = [AudioConditionByLatent(audio_latent, audio_strength)]

        target_height = int(height)
        target_width = int(width)
        resolution_divisor = 32 if alpha_gen else 64
        if target_height % resolution_divisor != 0:
            target_height = int(math.ceil(target_height / resolution_divisor) * resolution_divisor)
        if target_width % resolution_divisor != 0:
            target_width = int(math.ceil(target_width / resolution_divisor) * resolution_divisor)

        if latent_conditioning_stage2 is not None:
            expected_lat_h = target_height // 32
            expected_lat_w = target_width // 32
            if (
                latent_conditioning_stage2.shape[3] != expected_lat_h
                or latent_conditioning_stage2.shape[4] != expected_lat_w
            ):
                latent_conditioning_stage2 = None
            else:
                latent_conditioning_stage2 = latent_conditioning_stage2.to(device=self.device, dtype=self.dtype)

        video_conditioning_stage2 = None
        negative_prompt = n_prompt if n_prompt else DEFAULT_NEGATIVE_PROMPT
        skip_stage_2 = guide_phases <= 1
        phase2_ic_lora = phase2_ic_lora_name(loras_selected, loras_slists, force_phase2_control=editanything, force_name="EditAnything") if video_conditioning else None
        if video_conditioning and (phase2_ic_lora is not None or (force_stage2_ref_video_conditioning and not skip_stage_2)):
            video_conditioning_stage2 = video_conditioning
        if audio_cfg_scale is None:
            effective_audio_cfg_scale = LTX2_ID_LORA_AUDIO_CFG_SCALE if "1" in audio_prompt_type else float(guide_scale)
        else:
            effective_audio_cfg_scale = float(audio_cfg_scale)
        if "1" in audio_prompt_type and effective_audio_cfg_scale <= 1.0:
            effective_audio_cfg_scale = LTX2_ID_LORA_AUDIO_CFG_SCALE
        sample_solver = sample_solver.lower()
        prompt_relay_epsilon = 1e-3
        if isinstance(custom_settings, dict):
            try:
                prompt_relay_epsilon = float(custom_settings.get("prompt_relay_epsilon", prompt_relay_epsilon))
            except (TypeError, ValueError):
                pass
        if not 0 < prompt_relay_epsilon < 1:
            prompt_relay_epsilon = 1e-3
        prompt_relay_frame_offset = 0
        if int(window_no or 1) > 1 or (input_video is not None and not is_start_image_only):
            prompt_relay_frame_offset = max(0, int(prefix_frames_count or 0))
        ltx2_22B_class = self.model_def.get("ltx2_22B_class", False)

        def run_ltx2_pipeline(**pipeline_kwargs):
            pipeline_kwargs["custom_sigmas"] = custom_sigmas
            pipeline_context = self.pipeline.joyai_echo_context(joyai_context) if joyai_context is not None else nullcontext()
            with pipeline_context:
                return self.pipeline(**pipeline_kwargs)

        if hdr_transform == "acescct":
            pipeline_output = convert_sdr_to_hdr(self.model, self.video_encoder, self.video_decoder, self.pipeline.pipeline_components, self.device, input_frames, prefix_frames_count if images else 0, images, hdr_scene_context[0], int(frame_num), target_height, target_width, float(fps), int(seed), tiling_config=tiling_config, loras_slists=loras_slists, callback=callback, set_progress_status=set_progress_status, interrupt_check=interrupt_check, custom_sigmas=custom_sigmas), None
        elif isinstance(self.pipeline, TI2VidTwoStagesPipeline):
            pipeline_output = run_ltx2_pipeline(
                prompt=input_prompt,
                negative_prompt=negative_prompt,
                seed=int(seed),
                height=target_height,
                width=target_width,
                num_frames=int(frame_num),
                frame_rate=float(fps),
                prompt_relay_frame_offset=prompt_relay_frame_offset,
                prompt_relay_epsilon=prompt_relay_epsilon,
                num_inference_steps=int(sampling_steps),
                cfg_guidance_scale=float(guide_scale),
                audio_cfg_guidance_scale=effective_audio_cfg_scale,
                cfg_star_switch=cfg_star_switch,
                apg_switch=apg_switch,
                perturbation_switch=perturbation_switch,
                perturbation_layers=perturbation_layers,
                perturbation_start=perturbation_start,
                perturbation_end=perturbation_end,
                alt_guidance_scale=float(alt_guide_scale),
                alt_scale=float(alt_scale),
                sample_solver=sample_solver,
                images=images,
                guiding_images=guiding_images or None,
                guiding_images_stage2=guiding_images_stage2 or None,
                images_stage2=images_stage2 if stage2_override else None,
                video_conditioning=video_conditioning,
                video_conditioning_downscale_factor=video_conditioning_downscale_factor,
                video_conditioning_stage2=video_conditioning_stage2,
                latent_conditioning_stage2=latent_conditioning_stage2,
                tiling_config=tiling_config,
                enhance_prompt=False,
                audio_conditionings=audio_conditionings,
                audio_conditionings_stage2=audio_conditionings_stage2,
                audio_identity_guidance_scale=audio_identity_guidance_scale,
                callback=callback,
                set_progress_status=set_progress_status,
                interrupt_check=interrupt_check,
                loras_slists=loras_slists,
                text_connectors=text_connectors,
                masking_source=masking_source,
                masking_strength=masking_strength,
                return_latent_slice=return_latent_slice,
                continuous_conditioning_and_guide=continuous_conditioning_and_guide,
                skip_stage_2=skip_stage_2,
                frozen_video_conditioning=frozen_control_video,
                frozen_output_video=frozen_control_video,
                self_refiner_setting=self_refiner_setting,
                self_refiner_plan=self_refiner_plan,
                self_refiner_f_uncertainty=self_refiner_f_uncertainty,
                self_refiner_certain_percentage=self_refiner_certain_percentage,
                self_refiner_max_plans=self_refiner_max_plans,
                editanything_ref_images=editanything_ref_images,
                ltx2_22B_class=ltx2_22B_class,
                hdr_transform=hdr_transform,
                skip_audio=hdr_enabled or alpha_gen,
                tiled_stage_2="~" in video_prompt_type and not skip_stage_2,
            )
        else:
            distilled_kwargs = {}
            if distill:
                distilled_kwargs.update(
                    {
                        "NAG_scale": float(NAG_scale),
                        "NAG_tau": float(NAG_tau),
                        "NAG_alpha": float(NAG_alpha),
                    }
                )
            pipeline_output = run_ltx2_pipeline(
                prompt=input_prompt,
                negative_prompt=negative_prompt,
                seed=int(seed),
                height=target_height,
                width=target_width,
                num_frames=int(frame_num),
                frame_rate=float(fps),
                prompt_relay_frame_offset=prompt_relay_frame_offset,
                prompt_relay_epsilon=prompt_relay_epsilon,
                images=images,
                guiding_images=guiding_images or None,
                guiding_images_stage2=guiding_images_stage2 or None,
                images_stage2=images_stage2 if stage2_override else None,
                alt_guidance_scale=float(alt_guide_scale),
                audio_cfg_guidance_scale=effective_audio_cfg_scale,
                video_conditioning=video_conditioning,
                video_conditioning_downscale_factor=video_conditioning_downscale_factor,
                video_conditioning_stage2=video_conditioning_stage2,
                latent_conditioning_stage2=latent_conditioning_stage2,
                tiling_config=tiling_config,
                enhance_prompt=False,
                audio_conditionings=audio_conditionings,
                audio_conditionings_stage2=audio_conditionings_stage2,
                audio_identity_guidance_scale=audio_identity_guidance_scale,
                callback=callback,
                set_progress_status=set_progress_status,
                interrupt_check=interrupt_check,
                loras_slists=loras_slists,
                text_connectors=text_connectors,
                masking_source=masking_source,
                masking_strength=masking_strength,
                return_latent_slice=return_latent_slice,
                hdr_transform=hdr_transform,
                precomputed_contexts=hdr_scene_context,
                skip_audio=hdr_enabled or alpha_gen,
                tiled_stage_2="~" in video_prompt_type and not skip_stage_2,
                continuous_conditioning_and_guide=continuous_conditioning_and_guide,
                skip_stage_2=skip_stage_2,
                frozen_video_conditioning=frozen_control_video,
                frozen_output_video=frozen_control_video,
                self_refiner_setting=self_refiner_setting,
                self_refiner_plan=self_refiner_plan,
                self_refiner_f_uncertainty=self_refiner_f_uncertainty,
                self_refiner_certain_percentage=self_refiner_certain_percentage,
                self_refiner_max_plans=self_refiner_max_plans,
                editanything_ref_images=editanything_ref_images,
                ltx2_22B_class=ltx2_22B_class,
                use_ancestral_sampler=distill and ltx25,
                layout_to_render=layout_to_render,
                **distilled_kwargs,
            )

        latent_slice = None
        memory_latents = None
        if isinstance(pipeline_output, tuple) and len(pipeline_output) == 4:
            video, audio, latent_slice, memory_latents = pipeline_output
        elif isinstance(pipeline_output, tuple) and len(pipeline_output) == 3:
            video, audio, latent_slice = pipeline_output
        else:
            video, audio = pipeline_output

        if video is None or (audio is None and not (hdr_enabled or alpha_gen)):
            return None

        if self._interrupt:
            return None
        video_tensor = _collect_video_chunks(
            video,
            interrupt_check=interrupt_check,
            expected_frames=int(frame_num),
            expected_height=int(height),
            expected_width=int(width),
        )
        if video_tensor is None:
            return None

        video_tensor = video_tensor[:, :output_frame_num, :height, :width]
        if alpha_gen:
            from .vfx import alpha_result
            if set_progress_status is not None:
                set_progress_status("Exporting Alpha Matte and RGBA Video")
            return alpha_result(video_tensor, alpha_source, input_masks, fps, sys.modules["wgp"].server_config.get("rgba_video_output", "png_zip"), lambda: self._interrupt)
        if use_outpaint_gamma_roundtrip:
            if torch.is_inference(video_tensor):
                raise RuntimeError("LTX2 decoded video output is still an inference tensor; decode_video_to_tensor must allocate the output buffer outside inference mode.")
            exponent = float(LTX2_OUTPAINT_GAMMA)
            if video_tensor.dtype == torch.uint8:
                corrected = video_tensor.to(dtype=torch.float32).div_(255.0).clamp_(0.0, 1.0).pow_(exponent)
                video_tensor.copy_(corrected.mul_(255.0).round_().clamp_(0.0, 255.0).to(dtype=torch.uint8))
            else:
                corrected = video_tensor.to(dtype=torch.float32).add_(1.0).mul_(0.5).clamp_(0.0, 1.0).pow_(exponent)
                video_tensor.copy_(corrected.mul_(2.0).sub_(1.0).to(dtype=video_tensor.dtype))
        blend_inpainting = ltx2_inpainting and LTX2_INPAINTING_LAPLACIAN_BLEND
        blend_outpainting = new_outpainting and LTX2_OUTPAINTING_LAPLACIAN_BLEND
        if blend_inpainting or blend_outpainting:
            blend_source = input_frames if input_frames is not None else input_frames2
            blend_mask = input_masks if input_masks is not None else input_masks2
            mask_low_res_dilation = LTX2_OUTPAINTING_LAPLACIAN_MASK_LOW_RES_DILATION if blend_outpainting else LTX2_INPAINTING_LAPLACIAN_MASK_LOW_RES_DILATION
            video_tensor = _apply_ltx2_mask_blend(video_tensor, blend_source, blend_mask, output_frame_num, height, width, mask_low_res_dilation=mask_low_res_dilation, sanitize_masked_source=blend_inpainting and LTX2_INPAINTING_SANITIZE_LAPLACIAN_SOURCE)
        if image_mode > 0:
            video_tensor = video_tensor[:, :1]
        audio_np = None if image_mode > 0 or hdr_enabled else audio.detach().float().cpu().numpy() if audio is not None else None
        if audio_np is not None and audio_np.ndim == 2:
            if audio_np.shape[0] in (1, 2) and audio_np.shape[1] > audio_np.shape[0]:
                audio_np = audio_np.T
        output_audio_sampling_rate = int(getattr(self.vocoder, "output_sampling_rate", AUDIO_SAMPLE_RATE))
        result = {
            "x": video_tensor,
            "audio": audio_np,
            "audio_sampling_rate": output_audio_sampling_rate,
        }
        if hdr_enabled:
            result["hdr"] = True
            result["hdr_format"] = "linear_srgb"
            result["hdr_transform"] = hdr_transform
        if latent_slice is not None:
            result["latent_slice"] = latent_slice
        if memory_latents is not None:
            result["_memory_latents"] = memory_latents
        if joyai_memory_bank is not None:
            from .joyai_echo import record_joyai_echo_memory
            result = record_joyai_echo_memory(self, result, joyai_memory_bank, joyai_store_mem_selectors, prefix_frames_count, frame_num, fps, window_no)
        return result
