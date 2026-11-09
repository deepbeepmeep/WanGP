from shared.utils.phase_progress import generation_progress
import os
import json
import torch
from accelerate import init_empty_weights
from omegaconf import OmegaConf
from mmgp import offload
from shared.utils import files_locator as fl
from shared.utils.utils import convert_image_to_tensor, convert_tensor_to_image
from PIL import Image

from .kandinsky.models.dit import get_dit
from .kandinsky.models.text_embedders import get_text_embedder
from models.hyvideo.vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from .kandinsky.pipeline import Kandinsky5Pipeline


def _resolve_repo_path(path):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(base_dir, path)


def _infer_task(base_model_type):
    if not base_model_type:
        return "t2v"
    base = base_model_type.lower()
    if "i2v" in base:
        return "i2v"
    if "t2v" in base:
        return "t2v"
    if "i2i" in base:
        return "i2i"
    if "t2i" in base:
        return "t2i"
    return "t2v"


def _get_config_path(base_model_type):
    if not base_model_type:
        raise ValueError("Missing base_model_type for Kandinsky config resolution.")
    return os.path.join("models", "kandinsky5", "configs", f"{base_model_type}.yaml")


def _select_qwen_checkpoint(path_hint=None, folder_hint=None):
    if path_hint is not None:
        return path_hint
    if folder_hint:
        return fl.locate_folder(folder_hint)
    candidates = [
        os.path.join("Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-7B-Instruct_bf16.safetensors"),
        os.path.join("Qwen2.5-VL-7B-Instruct", "Qwen2.5-VL-7B-Instruct_quanto_bf16_int8.safetensors"),
    ]
    for candidate in candidates:
        resolved = fl.locate_file(candidate, error_if_none=False)
        if resolved is not None:
            return resolved
    return fl.locate_folder("Qwen2.5-VL-7B-Instruct")


def _preprocess_dit_state_dict(state_dict):
    prefixes = ["model.diffusion_model.", "diffusion_model.", "model."]
    for prefix in prefixes:
        if any(key.startswith(prefix) for key in state_dict):
            return {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}
    return state_dict


def _apply_vae_tile_config(vae, tile_config):
    if tile_config is None:
        if hasattr(vae, "disable_tiling"):
            vae.disable_tiling()
        return
    if hasattr(vae, "apply_tile_config"):
        vae.apply_tile_config(tile_config)
        return
    for key in (
        "tile_sample_min_tsize",
        "tile_latent_min_tsize",
        "tile_sample_min_size",
        "tile_latent_min_size",
        "tile_overlap_factor",
    ):
        if key in tile_config and hasattr(vae, key):
            setattr(vae, key, tile_config[key])
    if hasattr(vae, "enable_tiling"):
        vae.enable_tiling()


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
        save_quantized=False,
        dtype=torch.bfloat16,
        VAE_dtype=torch.float32,
        mixed_precision_transformer=False,
        **kwargs,
    ):
        self.device = torch.device("cuda")
        load_device = torch.device("cpu")
        self.dtype = dtype
        self.VAE_dtype = VAE_dtype
        self.model_def = model_def or {}
        self.base_model_type = base_model_type or self.model_def.get("architecture") or model_type
        self.mode = _infer_task(self.base_model_type)

        config_path = _resolve_repo_path(_get_config_path(self.base_model_type))
        conf = OmegaConf.load(config_path)
        overrides = self.model_def.get("k5_config_overrides")
        if overrides:
            conf = OmegaConf.merge(conf, OmegaConf.create(overrides))

        if isinstance(model_filename, (list, tuple)):
            model_filename = model_filename[0]
        conf.model.checkpoint_path = model_filename

        text_encoder_folder = self.model_def.get("text_encoder_folder")
        qwen_path = _select_qwen_checkpoint(text_encoder_filename, text_encoder_folder)
        conf.model.text_embedder.qwen.checkpoint_path = qwen_path
        conf.model.text_embedder.clip.checkpoint_path = fl.locate_folder("clip_vit_large_patch14")

        vae_filename = fl.locate_file("hunyuan_video_VAE_fp32.safetensors")
        vae_config = fl.locate_file("hunyuan_video_VAE_config.json")
        conf.model.vae.checkpoint_path = vae_filename
        conf.model.vae.name = "hunyuan"
        conf.model.vae.config_path = vae_config

        with init_empty_weights():
            dit = get_dit(conf.model.dit_params)
        # These hold no weights themselves: their FP32 locks keep MMGP from rounding the float32 timesteps and time embedding to BF16
        for module in (dit, dit.time_embeddings, dit.out_layer):
            module._lock_dtype = torch.float32
        offload.load_model_data(
            dit,
            model_filename,
            writable_tensors=False,
            default_dtype=torch.bfloat16,
            preprocess_sd=_preprocess_dit_state_dict,
        )
        offload.change_dtype(dit, dtype, True)
        dit.eval().requires_grad_(False)
        if save_quantized:
            from wgp import save_quantized_model
            save_quantized_model(dit, model_type, model_filename, dtype, None)

        quantized_qwen = isinstance(qwen_path, str) and "quanto" in qwen_path.lower()
        text_embedder = get_text_embedder(
            conf.model.text_embedder, device=load_device, quantized_qwen=quantized_qwen
        )

        vae = offload.fast_load_transformers_model(
            vae_filename,
            writable_tensors=False,
            modelClass=AutoencoderKLCausal3D,
            defaultConfigPath=vae_config,
            default_dtype=VAE_dtype,
        )
        vae = vae.to(dtype=VAE_dtype, device=load_device).eval()
        vae._model_dtype = VAE_dtype

        device_map = {"dit": self.device, "vae": self.device, "text_embedder": self.device}
        self.pipeline = Kandinsky5Pipeline(
            mode=self.mode,
            device_map=device_map,
            dit=dit,
            text_embedder=text_embedder,
            vae=vae,
            conf=conf,
        )
        self._interrupt = False

        self.transformer = dit
        self.text_embedder = text_embedder
        self.vae = vae

    @generation_progress
    def generate(
        self,
        seed: int | None = None,
        input_prompt: str = "",
        n_prompt: str = "",
        sampling_steps: int = 30,
        guide_scale: float = 5.0,
        frame_num: int = 81,
        width: int = 768,
        height: int = 512,
        image_start=None,
        image_mode: int = 0,
        shift: float = 10.0,
        callback=None,
        progress: bool | None = None,
        joint_pass: bool = False,
        VAE_tile_size: dict | None = None,
        set_progress_status=None,
        **kwargs,
    ):
        if seed is not None and seed < 0:
            seed = None
        if progress is None:
            progress = callback is None

        negative_caption = n_prompt or ""
        scheduler_scale = shift

        if VAE_tile_size is not None:
            _apply_vae_tile_config(self.vae, VAE_tile_size)
            self.vae._use_vae_tiling = True
        else:
            _apply_vae_tile_config(self.vae, None)
            self.vae._use_vae_tiling = False

        image_pil = None
        if image_start is not None:
            if torch.is_tensor(image_start):
                image_pil = convert_tensor_to_image(image_start)
            elif isinstance(image_start, Image.Image):
                image_pil = image_start
            else:
                raise ValueError(f"Unsupported image_start type: {type(image_start)}")

        time_length = 0 if image_mode > 0 else 1
        outputs = self.pipeline(
            text=input_prompt,
            image=image_pil,
            time_length=time_length,
            width=width,
            height=height,
            frame_num=frame_num,
            seed=seed,
            num_steps=sampling_steps,
            guidance_weight=guide_scale,
            scheduler_scale=scheduler_scale,
            negative_caption=negative_caption,
            expand_prompts=False,
            save_path=None,
            progress=progress,
            callback=callback,
            joint_pass=joint_pass,
        )
        if outputs is None:
            return None

        if isinstance(outputs, list):
            frames = [convert_image_to_tensor(frame) for frame in outputs]
            frames = torch.stack(frames, dim=0)
            video = frames.permute(1, 0, 2, 3)
        else:
            if outputs.dim() == 5:
                video = outputs[0].float().div_(127.5).sub_(1.0)
            else:
                video = outputs.float().div_(127.5).sub_(1.0)
        return video

    def get_loras_transformer(self, *args, **kwargs):
        return [], []

    @property
    def _interrupt(self):
        if hasattr(self, "pipeline"):
            return self.pipeline._interrupt
        return False

    @_interrupt.setter
    def _interrupt(self, value):
        if hasattr(self, "pipeline"):
            self.pipeline._interrupt = value
