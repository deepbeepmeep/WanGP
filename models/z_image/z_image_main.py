from shared.utils.phase_progress import generation_progress
import json
import os
from functools import lru_cache

import torch
from accelerate import init_empty_weights
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import logging
from mmgp import offload
from shared.utils import files_locator as fl
from shared.utils.gguf_mapping import has_standard_gguf_tensor_names, remap_state_dict_triplet
from transformers import AutoConfig, AutoTokenizer, Qwen3ForCausalLM

from .autoencoder_kl import AutoencoderKL
from .pipeline_z_image import ZImagePipeline
from .z_image_transformer2d import ZImageTransformer2DModel


logger = logging.get_logger(__name__)

_HF_TOKEN_EMBEDDING_KEY = "model.embed_tokens.weight"
_HF_LM_HEAD_KEY = "lm_head.weight"


def _qwen3_gguf_mapping_needed(state_dict) -> bool:
    if not state_dict or _HF_TOKEN_EMBEDDING_KEY in state_dict:
        return False
    return has_standard_gguf_tensor_names(state_dict)


@lru_cache(maxsize=4)
def _qwen3_gguf_load_info(config_path: str):
    config_path = os.path.abspath(config_path)
    config = AutoConfig.from_pretrained(os.path.dirname(config_path), local_files_only=True)
    with init_empty_weights():
        model = Qwen3ForCausalLM(config)
    from transformers.modeling_gguf_pytorch_utils import get_gguf_hf_weights_map

    return get_gguf_hf_weights_map(model), bool(getattr(config, "tie_word_embeddings", False))


def _build_qwen3_state_dict_preprocessor(config_path: str):
    config_path = os.path.abspath(config_path)

    def preprocess_state_dict(state_dict, quantization_map=None, tied_weights_map=None):
        if not _qwen3_gguf_mapping_needed(state_dict):
            return state_dict, quantization_map, tied_weights_map

        name_map, tie_word_embeddings = _qwen3_gguf_load_info(config_path)
        mapped = remap_state_dict_triplet(
            state_dict,
            quantization_map,
            tied_weights_map,
            name_map,
            keep_unmapped=False,
        )
        mapped_state_dict, mapped_quantization_map, mapped_tied_weights_map = mapped
        if _HF_TOKEN_EMBEDDING_KEY not in mapped_state_dict:
            return state_dict, quantization_map, tied_weights_map

        if tie_word_embeddings and _HF_LM_HEAD_KEY not in mapped_state_dict:
            if mapped_tied_weights_map is None:
                mapped_tied_weights_map = {}
            tied_names = mapped_tied_weights_map.setdefault(_HF_TOKEN_EMBEDDING_KEY, [])
            if _HF_LM_HEAD_KEY not in tied_names:
                tied_names.append(_HF_LM_HEAD_KEY)

        return mapped_state_dict, mapped_quantization_map, mapped_tied_weights_map

    return preprocess_state_dict

def _fix_qwen_fp8_sd(sd):
    # Find the actual key for embed tokens in the VL checkpoint
    embed_key = next((k for k in sd.keys() if "embed_tokens.weight" in k), None)
    
    if embed_key is not None:
        # 1. Provide it at the root (standard behavior)
        sd["lm_head.weight"] = sd[embed_key]
        
        # 2. Provide it at common Vision-Language nested levels
        sd["language_model.lm_head.weight"] = sd[embed_key]
        sd["model.language_model.lm_head.weight"] = sd[embed_key]
        
        # 3. Provide it dynamically based on however the checkpoint prefixes the embed tokens
        prefix_a = embed_key.replace("model.embed_tokens.weight", "")
        prefix_b = embed_key.replace("embed_tokens.weight", "")
        
        if prefix_a:
            sd[prefix_a + "lm_head.weight"] = sd[embed_key]
        if prefix_b:
            sd[prefix_b + "lm_head.weight"] = sd[embed_key]
            
    return sd

def conv_state_dict(sd: dict) -> dict:
    if "x_embedder.weight" not in sd and "model.diffusion_model.x_embedder.weight" not in sd:
        return sd

    inverse_replace = {
        "final_layer.": "all_final_layer.2-1.",
        "x_embedder.": "all_x_embedder.2-1.",
        ".attention.out.bias": ".attention.to_out.0.bias",
        ".attention.k_norm.weight": ".attention.norm_k.weight",
        ".attention.q_norm.weight": ".attention.norm_q.weight",
        ".attention.out.weight": ".attention.to_out.0.weight",
    }

    out_sd: dict[str, torch.Tensor] = {}

    for key, tensor in sd.items():
        key = key.replace("model.diffusion_model.", "")
        
        new_key = key
        for ori_sub, orig_sub in inverse_replace.items():
            new_key = new_key.replace(ori_sub, orig_sub)
        out_sd[new_key] = tensor

    return out_sd
    
_ZIMAGE_FUSED_SPLIT_MAP = {
    "attention.to_qkv": {"mapped_modules": ("attention.to_q", "attention.to_k", "attention.to_v")},
    "attention.qkv": {"mapped_modules": ("attention.to_q", "attention.to_k", "attention.to_v")},
    "feed_forward.net.0.proj": {"mapped_modules": ("feed_forward.w3", "feed_forward.w1")},
    "feed_forward.net.2": {"mapped_modules": ("feed_forward.w2",)},
}


class model_factory:
    def __init__(
        self,
        checkpoint_dir,
        model_filename=None,
        model_type=None,
        model_def=None,
        base_model_type=None,
        text_encoder_filename=None,
        quantizeTransformer=False,
        dtype=torch.bfloat16,
        VAE_dtype=torch.float32,
        mixed_precision_transformer=False,
        save_quantized=False,
        is_control=False,
        **kwargs,
    ):
        model_def = model_def or {}
        source =  model_def.get("source", None)
        module_source =  model_def.get("module_source", None)


        # model_filename can be a string or list of files (transformer + modules)
        transformer_filename = model_filename[0] if isinstance(model_filename, (list, tuple)) else model_filename
        if transformer_filename is None:
            raise ValueError("No transformer filename provided for Z-Image.")

        self.base_model_type = base_model_type
        self.is_control = is_control
        self.model_def = model_def

        default_transformer_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", f"{base_model_type}.json")

        def preprocess_sd(state_dict):
            return conv_state_dict(state_dict)

        model_class = ZImageTransformer2DModel

        # the loader converts the unquantized weights to dtype and gives it to the quantized formats as their compute dtype, without
        # dequantizing them (a transformer.to(dtype) would)
        kwargs_light= { "writable_tensors": False, "default_dtype": dtype, "preprocess_sd": preprocess_sd, "fused_split_map": _ZIMAGE_FUSED_SPLIT_MAP }
        # model_filename contains all files to load (transformer + modules merged by loader)
        import json
        import accelerate
        with open(default_transformer_config, "r") as f:
            config = json.load(f)
        config.pop("_class_name", None)
        config.pop("_diffusers_version", None)

        with accelerate.init_empty_weights():
            transformer = model_class(**config)

        if source is not None:
            offload.load_model_data(transformer, fl.locate_file(source), **kwargs_light)
        elif module_source is not None:
            offload.load_model_data(transformer, model_filename[:1] + [fl.locate_file(module_source)], **kwargs_light)
        else:
            offload.load_model_data(transformer, model_filename, **kwargs_light)

        from wgp import save_model
        from mmgp.safetensors2 import torch_load_file

        if module_source is not None:
            save_model(transformer, model_type, dtype, None, is_module=True, filter=list(torch_load_file(fl.locate_file(module_source))), module_source_no=1)

        if not source is None:
            save_model(transformer, model_type, dtype, None, submodel_no= 1)

        if save_quantized:
            from wgp import save_quantized_model
            save_quantized_model(transformer, model_type, transformer_filename, dtype, default_transformer_config)

        # Text encoder

        # text_encoder = Qwen3ForCausalLM.from_pretrained(os.path.dirname(text_encoder_filename), trust_remote_code=True)
        # text_encoder.to(torch.bfloat16)
        # offload.save_model(text_encoder, "c:/temp/qwnen3_bf16_.safetensors")

        text_encoder_folder = model_def.get("text_encoder_folder")
        if text_encoder_folder:
            text_encoder_path = fl.locate_folder(text_encoder_folder)
        else:
            text_encoder_path = os.path.dirname(text_encoder_filename)
        text_encoder_config = os.path.join(text_encoder_path, "config.json")

        qwen3_preprocessor = _build_qwen3_state_dict_preprocessor(text_encoder_config)

        def unified_preprocessor(sd, qm=None, twm=None):
            if qwen3_preprocessor is not None:
                sd, qm, twm = qwen3_preprocessor(sd, qm, twm)
            sd = _fix_qwen_fp8_sd(sd)
            return sd, qm, twm

        text_encoder = offload.fast_load_transformers_model(
            text_encoder_filename,
            writable_tensors=False,
            modelClass=Qwen3ForCausalLM,
            defaultConfigPath=text_encoder_config,
            preprocess_sd=unified_preprocessor,
        )

        # Resolve SDPA dtype mismatch for GGUF text encoders
        text_encoder.to(dtype)
        
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(text_encoder_path, trust_remote_code=True)

        # VAE
        vae_filename = fl.locate_file("ZImageTurbo_VAE_bf16.safetensors")
        vae_config_path = fl.locate_file("ZImageTurbo_VAE_bf16_config.json") 

        vae = offload.fast_load_transformers_model(
            vae_filename,
            writable_tensors=False,
            modelClass=AutoencoderKL,
            defaultConfigPath=vae_config_path,
            default_dtype=None,
        )

        # Scheduler
        with open(fl.locate_file("ZImageTurbo_scheduler_config.json"), "r", encoding="utf-8") as f:
            scheduler_config = json.load(f)

        scheduler = FlowMatchEulerDiscreteScheduler(**scheduler_config)

        self.pipeline = ZImagePipeline(
            scheduler=scheduler, vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, transformer=transformer
        )
        self.transformer = transformer
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.vae = vae
        self.scheduler = scheduler

    @generation_progress
    def generate(
        self,
        seed: int | None = None,
        input_prompt: str = "",
        n_prompt: str | None = None,
        sampling_steps: int = 20,
        sample_solver: str = "default",
        width: int = 1024,
        height: int = 1024,
        guide_scale: float = 0.0,
        batch_size: int = 1,
        callback=None,
        max_sequence_length: int = 512,
        VAE_tile_size=None,
        cfg_normalization: bool = False,
        cfg_truncation: float = 1.0,
        input_frames=None,
        input_masks=None,
        context_scale: float = [0],
        input_ref_images = None,
        NAG_scale: float = 1.0,
        NAG_tau: float = 3.5,
        NAG_alpha: float = 0.5,
        loras_slists=None,
        vae_upsampler=None,
        set_progress_status=None,
        **kwargs,
    ):
        generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        if seed is None or seed < 0:
            generator.seed()
        else:
            generator.manual_seed(int(seed))

        if VAE_tile_size is not None and hasattr(self.vae, "use_tiling"):
            if isinstance(VAE_tile_size, int):
                tiling = VAE_tile_size > 0
                tile_size = max(VAE_tile_size, 0)
            else:
                tiling = bool(VAE_tile_size[0])
                tile_size = VAE_tile_size[1] if len(VAE_tile_size) > 1 else 0
            self.vae.use_tiling = tiling
            self.vae.tile_latent_min_height = tile_size
            self.vae.tile_latent_min_width = tile_size

        unified_solver = self.model_def.get("unified_solver", False)
        if unified_solver:
            sample_solver = "unified"
        elif not sample_solver:
            sample_solver = "default"

        if self.model_def.get("guidance_max_phases", 0) < 1:
            guide_scale = 0

        def _vae_upsampler_progress(_phase, current_step=None, total_steps=None):
            if callable(set_progress_status):
                progress_label = getattr(vae_upsampler, "progress_label", "VAE Spatial Upsampling")
                if current_step is None or total_steps is None:
                    set_progress_status(f"{progress_label} in Progress")
                else:
                    total_steps = int(total_steps)
                    step_no = min(int(current_step) + 1, total_steps)
                    set_progress_status(f"{progress_label} in Progress ({step_no}/{total_steps})")

        images = self.pipeline(
            prompt=input_prompt,
            negative_prompt=n_prompt,
            num_inference_steps=sampling_steps,
            sample_solver=sample_solver,
            guidance_scale=guide_scale,
            num_images_per_prompt=batch_size,
            generator=generator,
            height=height,
            width=width,
            max_sequence_length=max_sequence_length,
            callback_on_step_end=None,
            output_type="pt",
            return_dict=True,
            cfg_normalization=cfg_normalization,
            cfg_truncation=cfg_truncation,
            callback=callback,
            pipeline=self.pipeline,
            control_image=input_frames,
            inpaint_mask=input_masks,
            control_context_scale=None if context_scale is None else context_scale[0],
            input_ref_images= input_ref_images,
            NAG_scale=NAG_scale,
            NAG_tau=NAG_tau,
            NAG_alpha=NAG_alpha,
            loras_slists=loras_slists,
            vae_upsampler=vae_upsampler,
            vae_upsampler_seed=seed,
            vae_upsampler_progress_callback=_vae_upsampler_progress,
        )

        if images is None:
            return None

        return images.transpose(0, 1)

    def get_loras_transformer(self, *args, **kwargs):
        return [], []

    @property
    def _interrupt(self):
        return getattr(self.pipeline, "_interrupt", False)

    @_interrupt.setter
    def _interrupt(self, value):
        if hasattr(self, "pipeline"):
            self.pipeline._interrupt = value
