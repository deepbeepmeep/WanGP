"""Model construction and checkpoint loading for MiniMax H3."""

import os
import re
from functools import partial

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from tqdm import tqdm

from mmgp import offload, quant_router, runtime
from shared.utils import files_locator as fl

from .audio_vae import MiniMaxH3AudioVAE
from .convrot_layout import get_convrot_layout
from .latent_upscaler import MiniMaxH3LatentUpscaler
from .pipeline import MiniMaxH3Pipeline
from .text_encoder import MiniMaxH3TextEncoder, load_h3_qwen_config
from .transformer import MiniMaxH3Model, get_linear_split_map
from .video_vae import MiniMaxH3VideoVAE, get_video_vae_linear_split_map, group_qkv_rows
from .viggle import load_fixed_prompt
from .vae_upsampler import X2_VAE_FILE, X2_VAE_INT8_FILE, X2_VAE_VALUE


VIDEO_VAE_FILE = "MiniMax-H3-video_vae_fp16.safetensors"
VIDEO_VAE_FP8MIX_FILE = "minimax_h3_video_vae_fp8mix.safetensors"
VIDEO_VAE_INT8_FILE = "minimax_h3/minimax_h3_video_vae_int8_convrot.safetensors"
VIDEO_VAE_LIGHT_FILE = "minimax_h3/minimax_h3_lightvae_lite_bf16.safetensors"
VIDEO_VAE_LIGHT_INT8_FILE = "minimax_h3/minimax_h3_lightvae_lite_int8_convrot.safetensors"
VIDEO_VAE_SPLIT_INT8 = True  # comparison switch: the INT8 VAEs split their q / k / v projections too (decoder attention through the attention kit)
AUDIO_VAE_FILE = "MiniMax-H3-audio_vae_fp32.safetensors"
LATENT_UPSCALER_FOLDER = "minimax_h3"
LATENT_UPSCALER_FILE = "minimax_h3_latent_upscaler_3d_bf16.safetensors"
TEXT_ENCODER_FOLDER = "Qwen3-VL-32B-Instruct"
ADALN_CURVE_DIM = 8
NUM_TRANSFORMER_BLOCKS = 50


def _hybrid_ref2va_blocks(filename):
    filenames = filename if isinstance(filename, (list, tuple)) else [filename]
    ranges = set()
    for path in filenames:
        match = re.search(r"hybrid[_-]fl2va[_-]ref2va[_-]b(\d+)-(\d+)", os.path.basename(path), re.IGNORECASE)
        if match:
            ranges.add(tuple(map(int, match.groups())))
    if not ranges:
        return None
    if len(ranges) != 1:
        raise ValueError(f"MiniMax H3 checkpoint files specify conflicting hybrid block ranges: {sorted(ranges)}")
    start, end = ranges.pop()
    if not 0 <= start <= end < NUM_TRANSFORMER_BLOCKS:
        raise ValueError(f"Invalid MiniMax H3 hybrid Ref2VA block range b{start}-{end}; expected blocks 0-{NUM_TRANSFORMER_BLOCKS - 1}")
    return start, end


def _strip_wrappers(state_dict, quantization_map=None, tied_weights_map=None, pdd=False, ref2va_adaln_t_table=None):
    prefixes = ("model.diffusion_model.", "diffusion_model.")

    def strip(mapping):
        if mapping is None:
            return None
        result = {}
        for key, value in mapping.items():
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    break
            if key.startswith("transformer_blocks.") and ".attn.linear_attention." in key:
                key = "blocks." + key[len("transformer_blocks."):].replace(".attn.linear_attention.", ".attn.vdn.linear_attention.")
            elif key.startswith("transformer_blocks.") and ".attn.softmax_gate." in key:
                key = "blocks." + key[len("transformer_blocks."):].replace(".attn.softmax_gate.", ".attn.vdn.softmax_gate.")
            elif key.startswith("transformer_blocks.") and ".attn.to_out_linear." in key:
                key = "blocks." + key[len("transformer_blocks."):].replace(".attn.to_out_linear.", ".attn.vdn.to_out_linear.")
            if pdd and (key.startswith("final_layer.video_out.") or key.startswith("final_layer.audio_out.") or
                        key in ("final_layer.video_out", "final_layer.audio_out")):
                continue
            if pdd and key.startswith("pdd_heads."):
                key = "final_layer." + key[len("pdd_heads."):]
            result[key] = value
        return result

    state_dict = strip(state_dict)
    if pdd:  # stacked PDD heads [num_steps, out, in]: head 0 is the base projection, the others are stored relative to it (FP32)
        for key in [f"final_layer.{head}.{suffix}" for head in ("video_out", "audio_out")
                    for suffix in ("weight", "bias") if f"final_layer.{head}.{suffix}" in state_dict]:
            head, suffix = key.rsplit(".", 1)
            stacked = state_dict.pop(key)
            base = stacked[0].float()
            state_dict[f"{head}.linear.{suffix}"], state_dict[f"{head}.{'heads' if suffix == 'weight' else 'head_biases'}"] = base, stacked[1:].float() - base
    if ref2va_adaln_t_table is not None:
        state_dict["ref2va_adaln_t_table"] = ref2va_adaln_t_table
    return state_dict, strip(quantization_map), strip(tied_weights_map)


def _strip_text_encoder_wrapper(state_dict, quantization_map=None, tied_weights_map=None):
    root = "language_model" if "model.embed_tokens.weight" in state_dict else ""
    return tuple(offload.map_state_dict([state_dict, quantization_map, tied_weights_map], rules={"model": root}))


def probe_h3_checkpoint(filename):
    """Inspect tensor headers before allocating the network."""
    checkpoint_path = filename[0] if isinstance(filename, (list, tuple)) else filename
    hybrid_ref2va_blocks = _hybrid_ref2va_blocks(filename)
    state_dict, _ = quant_router.load_metadata_state_dict(checkpoint_path)
    normalized = {}
    for key, value in state_dict.items():
        for prefix in ("model.diffusion_model.", "diffusion_model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        normalized[key] = value
    separate_qkv = any(key.endswith(".attn.q_proj.weight") for key in normalized)
    table = next((tensor for key, tensor in normalized.items() if key == "adaln_t_table"), None)
    if table is None:
        if hybrid_ref2va_blocks is not None:
            raise ValueError("MiniMax H3 FL2VA/Ref2VA hybrid checkpoints require compressed AdaLN tables")
        return {"compressed_modulation": False, "adaln_curve_grid": None, "time_embed_dim": 2688, "adaln_dtype": None, "separate_qkv": separate_qkv}
    if not table.dtype.is_floating_point:
        raise ValueError(f"H3 AdaLN curve table must use an unquantized floating-point GGUF qtype, got {table.dtype}")
    if len(table.shape) != 2 or table.shape[0] < 2:
        raise ValueError(f"Invalid H3 AdaLN curve table shape: {tuple(table.shape)}")
    rank = int(table.shape[1])
    if rank != ADALN_CURVE_DIM:
        print(
            f"MiniMax H3 pruned checkpoint '{checkpoint_path}' uses non-official AdaLN rank {rank}; the official rank is {ADALN_CURVE_DIM}. "
            "This file may be incompatible with MiniMax H3 LoRAs. Delete it and retry so WanGP can automatically download the official replacement."
        )
    return {"compressed_modulation": True, "adaln_curve_grid": int(table.shape[0]), "time_embed_dim": rank, "adaln_dtype": table.dtype,
            "hybrid_ref2va_blocks": hybrid_ref2va_blocks, "separate_qkv": separate_qkv}


def probe_h3_control_module(filenames, time_embed_dim):
    """Return the control layers and input channels of an attached ControlNet-Union module."""
    layers, control_in_dim = set(), 0
    for path in filenames:
        state_dict, _ = quant_router.load_metadata_state_dict(path)
        for key, value in state_dict.items():
            if key.startswith("blocks.") and key.endswith(".control.adaln_proj.linear.weight"):
                layers.add(int(key.split(".")[1]))
                if value.shape[1] != time_embed_dim:
                    raise ValueError(f"MiniMax H3 control module '{os.path.basename(path)}' uses AdaLN width {value.shape[1]} but the transformer uses {time_embed_dim}; "
                                     "pair pruned control modules with pruned transformers and full modules with full transformers")
            elif key == "control_patch_proj.weight":
                control_in_dim = value.shape[1] // 4
    if layers and (0 not in layers or not control_in_dim):
        raise ValueError("Incomplete MiniMax H3 control module: it must include the layer-0 control block and control_patch_proj")
    return tuple(sorted(layers)), control_in_dim


def _resample_adaln_table(table, rows, dtype):
    if table.shape[0] != rows:
        position = torch.linspace(0, table.shape[0] - 1, rows, device=table.device)
        lower = position.floor().long().clamp(max=table.shape[0] - 2)
        table = torch.lerp(table[lower], table[lower + 1], (position - lower).unsqueeze(1))
    return table.to(dtype=dtype)


def _qkv_layout(filenames, qkv_layout):
    """Quantized H3 checkpoints (ConvRot, NVFP4, W4A8, GGUF) store every fused qkv grouped, their BF16 layers included; unquantized ones
    use the official interleaved rows unless the model definition says otherwise (finetunes)."""
    for path in filenames:
        if path.lower().endswith(".gguf"):
            return "grouped"
        with safe_open(path, "pt") as reader:
            metadata = reader.metadata() or {}
            if metadata.get("quantization_format") or metadata.get("_quantization_metadata") or any(key.endswith((".comfy_quant", ".weight_s_rel")) for key in reader.keys()):
                return "grouped"
    return qkv_layout


def _load_transformer(filename, dtype, qkv_splitting=True, qkv_layout="interleaved", pdd=False,
                      pdd_num_steps=None, pdd_block_size=None, vdn=False, veda=False, vsa=False):
    checkpoint = probe_h3_checkpoint(filename)
    # Checkpoints with separate projections require split modules even with the Lower RAM preference.
    qkv_splitting = qkv_splitting or checkpoint["separate_qkv"]
    hybrid_ref2va_blocks = checkpoint.get("hybrid_ref2va_blocks")
    filenames = list(filename) if isinstance(filename, (list, tuple)) else [filename]
    from .lora_tables import detect_checkpoint

    lora_affine_package = detect_checkpoint(filenames[0])
    if lora_affine_package is not None:
        tqdm.write(f"MiniMax H3: detected AdaLN map '{lora_affine_package.name}'.")
    ref2va_adaln_t_table = None
    if hybrid_ref2va_blocks is not None:
        from .lora_affine import _load_affine_package
        ref2va_adaln_t_table = _resample_adaln_table(_load_affine_package("ref2va", checkpoint["time_embed_dim"])[0],
                                                     checkpoint["adaln_curve_grid"], checkpoint["adaln_dtype"])
        start, end = hybrid_ref2va_blocks
        print(f"MiniMax H3: detected FL2VA/Ref2VA hybrid checkpoint; blocks {start}-{end} use the Ref2VA AdaLN table, "
              "and all other blocks plus the final layer use the FL2VA table.")
    if pdd and (int(pdd_num_steps) < 1 or int(pdd_block_size) < 1 or int(pdd_num_steps) % int(pdd_block_size)):
        raise ValueError(f"Invalid MiniMax H3 PDD grid={pdd_num_steps}, block={pdd_block_size}")
    control_layers, control_in_dim = probe_h3_control_module(filenames[1:], checkpoint["time_embed_dim"])
    if veda:
        from .veda.attention import VedaAttention, VedaPolicy, read_metadata
        veda_policy = VedaPolicy(read_metadata(filenames[-1]))
    with init_empty_weights(include_buffers=True):
        transformer = MiniMaxH3Model(adaln_curve_grid=checkpoint["adaln_curve_grid"], time_embed_dim=checkpoint["time_embed_dim"], adaln_dtype=checkpoint["adaln_dtype"],
                                     hybrid_ref2va_blocks=hybrid_ref2va_blocks, pdd_num_steps=pdd_num_steps,
                                     pdd_block_size=pdd_block_size, vdn=vdn, vsa=vsa, control_layers=control_layers,
                                     control_in_dim=control_in_dim, dtype=dtype, device="meta")
        if veda:
            transformer.veda_attention = veda_policy
            for layer, block in enumerate(transformer.blocks):
                block.attn.veda = VedaAttention(veda_policy, layer, device="meta")
        if vsa:
            from .vsa.attention import VSAAttention, VSAPolicy, read_fasth3_inference
            fasth3_inference = read_fasth3_inference(filenames[0])
            transformer.vsa_attention = VSAPolicy(fasth3_inference)
            for layer, block in enumerate(transformer.blocks):
                block.attn.vsa = VSAAttention(transformer.vsa_attention, layer)
    qkv_layout = _qkv_layout(filenames, qkv_layout)
    split_map = get_linear_split_map(transformer.attention_inner_size, qkv_layout=qkv_layout) if qkv_splitting and not any(path.lower().endswith(".gguf") for path in filenames) else None
    if split_map is not None:
        offload.split_linear_modules(transformer, split_map)
    else:
        for module in transformer.modules():
            if hasattr(module, "qkv_interleaved"):
                module.qkv_interleaved = qkv_layout == "interleaved"
    transformer.requires_grad_(False)
    preprocess_sd = partial(_strip_wrappers, pdd=pdd, ref2va_adaln_t_table=ref2va_adaln_t_table)
    offload.load_model_data(transformer, filenames, writable_tensors=False, default_dtype=dtype, preprocess_sd=preprocess_sd,
                            fused_split_map=split_map)
    transformer.eval().requires_grad_(False)
    transformer.h3_checkpoint_info = checkpoint
    transformer.h3_lora_affine_package = lora_affine_package
    transformer.h3_lora_map_checked = hybrid_ref2va_blocks is None
    transformer.split_linear_modules_map = split_map
    transformer.fasth3_inference = fasth3_inference if vsa else None  # the trained schedule and sparsity contract of a FastH3 student
    return transformer


def _load_text_encoder(filename, dtype):
    config_path = fl.locate_file(os.path.join(TEXT_ENCODER_FOLDER, "config.json"))
    config = load_h3_qwen_config(config_path)
    with init_empty_weights(include_buffers=True):
        text_encoder = MiniMaxH3TextEncoder(config)
    text_encoder.visual.rotary_pos_emb.reset_inv_freq()
    text_encoder.language_model.rotary_emb.reset_inv_freq()
    text_encoder.requires_grad_(False)
    offload.load_model_data(text_encoder, filename, writable_tensors=False, default_dtype=dtype,
                            preprocess_sd=_strip_text_encoder_wrapper, ignore_unused_weights=True)
    runtime.lookup_on_cpu(text_encoder.language_model.embed_tokens)  # the 1.5 GB token table stays in RAM: mmgp looks up the prompt's rows on the CPU
    text_encoder.set_tokenizer(os.path.dirname(config_path))
    text_encoder.eval().requires_grad_(False)
    return text_encoder


def _load_video_vae(filename, qkv_splitting=True, upsampling=False):
    lightvae = filename in (VIDEO_VAE_LIGHT_FILE, VIDEO_VAE_LIGHT_INT8_FILE)
    filename = fl.locate_file(filename)
    dtype = quant_router.load_metadata_state_dict(filename)[0]["encoder.conv_in.weight"].dtype
    dtype = torch.float16 if dtype == torch.float32 else dtype  # checkpoint-native FP16/BF16; FP32 wastes VRAM for little precision
    print(f"Loading MiniMax H3 Video VAE '{filename}'...")
    with init_empty_weights(include_buffers=False):
        vae = MiniMaxH3VideoVAE(upsampling=upsampling, decoder_num_layers=12 if lightvae else 36)
    split_map = get_video_vae_linear_split_map() if qkv_splitting else None
    if split_map is not None:
        offload.split_linear_modules(vae, split_map)
    def preprocess(state_dict):
        if split_map is not None:  # q / k / v rows grouped, as MMGP splits them (quantized ones included)
            group_qkv_rows(state_dict)
        for name in ("latents_mean", "latents_std"):
            if upsampling:
                state_dict["_" + name] = state_dict.pop(name)
            else:
                state_dict.pop(name, None)
        if upsampling:
            # B32 consumes source RGB encoder features and is not part of latent decoding.
            for name in [key for key in state_dict if key.startswith("detail_b32.")]:
                del state_dict[name]
        return state_dict

    # X2's learned tensors are FP16. Give quantized layers the same compute dtype
    # instead of inheriting FP32 from the empty model's constructor.
    offload.load_model_data(vae, filename, writable_tensors=False, default_dtype=None if upsampling else dtype, fp32_dtype=dtype, preprocess_sd=preprocess,
                            fused_split_map=split_map)
    vae.split_linear_modules_map = split_map
    vae._model_dtype = dtype
    vae.upsampling_set = X2_VAE_VALUE if upsampling else None
    vae.eval().requires_grad_(False)
    return vae


def _load_audio_vae(filename):
    filename = fl.locate_file(filename)
    with init_empty_weights(include_buffers=False):
        vae = MiniMaxH3AudioVAE()

    def preprocess(state_dict):
        state_dict.pop("latents_mean", None)
        state_dict.pop("latents_std", None)
        for key in list(state_dict):
            if key.endswith(".weight_g"):
                state_dict[key.removesuffix("weight_g") + "parametrizations.weight.original0"] = state_dict.pop(key)
            elif key.endswith(".weight_v"):
                state_dict[key.removesuffix("weight_v") + "parametrizations.weight.original1"] = state_dict.pop(key)
        return state_dict

    offload.load_model_data(vae, filename, writable_tensors=False, default_dtype=torch.float32, preprocess_sd=preprocess)
    vae.eval().requires_grad_(False)
    return vae


def _load_latent_upscaler(filename):
    filename = fl.locate_file(filename)
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with init_empty_weights(include_buffers=True):
            upscaler = MiniMaxH3LatentUpscaler()
    finally:
        torch.set_default_dtype(previous_dtype)
    offload.load_model_data(upscaler, filename, writable_tensors=False, default_dtype=torch.bfloat16)
    return upscaler.eval().requires_grad_(False)


def model_factory(model_filename, text_encoder_filename, qkv_splitting, dtype=torch.bfloat16, save_quantized=False,
                  model_type="minimax_h3_fl2va", reference_mode=False, video_vae_filename=VIDEO_VAE_FILE,
                  audio_vae_filename=AUDIO_VAE_FILE, latent_upscaler_filename=os.path.join(LATENT_UPSCALER_FOLDER, LATENT_UPSCALER_FILE),
                  shared_h3_pipeline=None, qkv_layout="interleaved", pdd=False, pdd_num_steps=None, pdd_block_size=None, vdn=False, audio_only=False,
                  fixed_prompt_filename=None, veda=False, vsa=False):
    transformer = _load_transformer(model_filename, dtype, qkv_splitting, qkv_layout, pdd, pdd_num_steps, pdd_block_size, vdn, veda, vsa)
    if shared_h3_pipeline is None:
        text_encoder = _load_text_encoder(text_encoder_filename, dtype) if fixed_prompt_filename is None else None
        video_vae_qkv_splitting = qkv_splitting and video_vae_filename in (VIDEO_VAE_FILE, VIDEO_VAE_LIGHT_FILE, X2_VAE_FILE) + (
            (VIDEO_VAE_INT8_FILE, VIDEO_VAE_LIGHT_INT8_FILE, X2_VAE_INT8_FILE) if VIDEO_VAE_SPLIT_INT8 else ())
        video_vae = _load_video_vae(video_vae_filename, video_vae_qkv_splitting, upsampling=video_vae_filename in (X2_VAE_FILE, X2_VAE_INT8_FILE))
        audio_vae = _load_audio_vae(audio_vae_filename)
        latent_upscaler = _load_latent_upscaler(latent_upscaler_filename) if fixed_prompt_filename is None else None
    else:
        text_encoder = shared_h3_pipeline.text_encoder
        video_vae = shared_h3_pipeline.vae
        audio_vae = shared_h3_pipeline.audio_vae
        latent_upscaler = shared_h3_pipeline.latent_upscaler
    fixed_prompt = load_fixed_prompt(fixed_prompt_filename) if fixed_prompt_filename is not None else None
    pipeline = MiniMaxH3Pipeline(transformer, text_encoder, video_vae, audio_vae, latent_upscaler=latent_upscaler, reference_mode=reference_mode, audio_only=audio_only, dtype=dtype, fixed_prompt=fixed_prompt)
    if save_quantized:
        from wgp import save_quantized_model
        save_quantized_model(transformer, model_type, model_filename[0], dtype, None,
                             convrot_layout=get_convrot_layout(transformer))
    return pipeline


__all__ = ["ADALN_CURVE_DIM", "AUDIO_VAE_FILE", "LATENT_UPSCALER_FILE", "LATENT_UPSCALER_FOLDER", "TEXT_ENCODER_FOLDER",
           "VIDEO_VAE_FILE", "VIDEO_VAE_FP8MIX_FILE", "VIDEO_VAE_INT8_FILE", "VIDEO_VAE_LIGHT_FILE", "VIDEO_VAE_LIGHT_INT8_FILE", "model_factory", "probe_h3_checkpoint"]
