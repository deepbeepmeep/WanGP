# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from accelerate import init_empty_weights
from diffusers import FlowMatchEulerDiscreteScheduler
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen2VLImageProcessor

from mmgp import offload, runtime, cuda_graphs
from models.flux3.vae import Flux3VAE
from models.ideogram4.qwen3_vl_configuration import Qwen3VLConfig
from shared.utils import files_locator as fl
from shared.utils.phase_progress import generation_progress, GenerationAborted, set_phase_status
from shared.utils.text_encoder_cache import TextEncoderCache
from shared.utils.utils import convert_tensor_to_image, guide_to_float
from shared.utils.loras_mutipliers import update_loras_slists
from .caption import Caption, VISION_MARKER
from .encoder import FiboEncoder, FiboProcessor, PROMPT_TEMPLATE
from .text_images import draw_text_image
from .assets import load_fonts
from .transformer import BriaFibo2Transformer2DModel

NEGATIVE_EDIT = '{"short_description":"","objects":[],"background_setting":"","lighting":{},"aesthetics":{},"photographic_characteristics":{},"style_medium":"","context":"","artistic_style":""}'
NEGATIVE_T2I = '{"short_description":"","objects":[{"description":"the main subject","texture":"smooth, featureless, plastic-like, airbrushed, no fine surface detail","appearance_details":"soft blurred edges, simplified shapes, no fine detail","skin_tone_and_texture":"waxy airbrushed skin, poreless, plastic smooth"}],"background_setting":"a plain seamless studio backdrop, perfectly smooth gradient, blurred and featureless, no visible texture","lighting":{},"aesthetics":{"artifacts":["oversaturated colours","neon colour cast","overexposed highlights","lens flare","slight blur","motion blur","haze"]},"photographic_characteristics":{"depth_of_field":"shallow, heavy bokeh obscuring most of the frame","focus":"soft, slightly out of focus, lacking fine texture"},"style_medium":"","context":"","artistic_style":""}'


def pack(x):
    b, c, h, w = x.shape
    return x.reshape(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, c * 4, h // 2, w // 2)


def unpack(x):
    b, c, h, w = x.shape
    return x.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, h * 2, w * 2)


class FiboVAE(Flux3VAE):
    def __init__(self):
        super().__init__()
        self.bn._lock_dtype = torch.float32
        self.encoder._lock_dtype = self.decoder._lock_dtype = torch.float16
        self._convertWeightsFloatTo = None

    def normalize(self, z):
        z = z.float()
        mean = self.bn.running_mean.reshape(1, -1, 1, 1)
        scale = (self.bn.running_var + self.bn_eps).rsqrt().reshape(1, -1, 1, 1)
        return (z - mean) * scale

    def pre_decode(self, z):
        mean = self.bn.running_mean.reshape(1, -1, 1, 1)
        scale = (self.bn.running_var + self.bn_eps).rsqrt().reshape(1, -1, 1, 1)
        return unpack(z.float() / scale + mean).to(torch.float16)


class FiboPipeline:
    def __init__(self, transformer, encoder, processor, vae, edit_only=False):
        self.transformer, self.text_encoder, self.processor, self.vae = transformer, encoder, processor, vae
        self.edit_only = edit_only
        self.text_encoder_cache = TextEncoderCache(max_size_mb=768)
        self.preview_mean = vae.bn.running_mean.detach().float().cpu().reshape(1, 128, 1, 1)
        self.preview_scale = (vae.bn.running_var.detach().float().cpu() + vae.bn_eps).rsqrt().reshape(1, 128, 1, 1)
        self._interrupt = False

    @property
    def _interrupt(self):
        return self.transformer._interrupt

    @_interrupt.setter
    def _interrupt(self, value):
        self.transformer._interrupt = self.transformer.perceiver._interrupt = value
        self.text_encoder._interrupt = self.text_encoder.language_model._interrupt = self.text_encoder.visual._interrupt = value

    def abort(self):
        self._interrupt = True

    def encode(self, prompt, images, device):
        keys = tuple((image.size, hashlib.sha256(image.tobytes()).digest()) for image in images)
        def compute(items):
            values = []
            for text in items:
                with torch.device("cpu"):
                    inputs = self.processor(text=[PROMPT_TEMPLATE.format(text)], images=images or None, padding="longest",
                                            max_length=4096, truncation=True, return_tensors="pt")
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    values.append(self.text_encoder(inputs))
                if self._interrupt:
                    raise GenerationAborted
            return values
        return self.text_encoder_cache.encode(compute, [prompt], device=device, cache_keys=(prompt, keys))[0]

    def preview(self, latents):
        value = pack(latents)
        value = value / self.preview_scale.to(value) + self.preview_mean.to(value)
        return unpack(value).transpose(0, 1)

    @generation_progress
    @torch.inference_mode()
    def generate(self, input_prompt="", n_prompt="", seed=0, sampling_steps=50, width=1024, height=1024, guide_scale=5.0,
                 batch_size=1, input_ref_images=None, input_frames=None, input_masks=None, video_prompt_type="",
                 callback=None, VAE_tile_size=None, set_progress_status=None, loras_slists=None, **kwargs):
        device = runtime.execution_device(self.transformer)
        text = "\n".join(line for line in input_prompt.splitlines() if not line.lstrip().startswith("#"))
        references = list(input_ref_images or []) if "I" in video_prompt_type else []
        if input_frames is not None:
            references.insert(0, convert_tensor_to_image(guide_to_float(input_frames)) if torch.is_tensor(input_frames) else input_frames)
        if self.edit_only:
            if not references:
                raise ValueError("Fibo 2.0 Edit Turbo requires a source image. Use Dev for text-only generation.")
            guide_scale = 1.0
        references = [(convert_tensor_to_image(image) if torch.is_tensor(image) else image).convert("RGB") for image in references]
        if input_masks is not None:
            mask = convert_tensor_to_image(input_masks, mask_levels=True) if torch.is_tensor(input_masks) else input_masks
            binary = mask.convert("L").point(lambda value: 255 if value >= 128 else 0)
            references[0] = Image.composite(Image.new("RGB", references[0].size, (128, 128, 128)), references[0], binary)
        caption, _ = Caption._parse_prompt(text)
        text_images = []
        if isinstance(caption, dict):
            entries = [entry["text"] for entry in caption.get("text_render", [])
                       if isinstance(entry, dict) and isinstance(entry.get("text"), str) and entry["text"]]
            if entries:
                fonts = load_fonts()
                text_images = [draw_text_image(entry, fonts) for entry in entries]
        vision_images = []
        for image in references:
            thumbnail = image.copy()
            thumbnail.thumbnail((768, 768))
            vision_images.append(thumbnail)
        base_prompt = None
        if references:
            prompt = Caption._get_edit_prompt(text, len(references), bool(text_images))
        else:
            if isinstance(caption, dict):
                if text_images:
                    base_prompt = json.dumps(Caption._mark_text_images(caption.copy(), ""), separators=(",", ":"), ensure_ascii=False)
                    caption = Caption._mark_text_images(caption, VISION_MARKER)
                prompt = json.dumps(caption, separators=(",", ":"), ensure_ascii=False)
            else:
                prompt = caption or ""
        set_phase_status("Preparing Text Conditioning")
        positive = self.encode(prompt, vision_images + text_images, device)
        base = self.encode(base_prompt, [], device) if base_prompt is not None else positive
        negative = self.encode(n_prompt or (NEGATIVE_EDIT if references else NEGATIVE_T2I), [], device) if guide_scale > 1 else None
        set_phase_status("Preparing Image Conditioning")
        tiled, tile_size = VAE_tile_size if VAE_tile_size is not None else (True, 1024)
        contexts = []
        for image in references:
            pixels = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).unsqueeze(0).float()
            pixels = (pixels / 127.5 - 1).to(runtime.execution_device(self.vae))
            with torch.autocast("cuda", dtype=torch.float16):
                contexts.append(unpack(self.vae(pixels, encode=True, tile_size=tile_size if tiled else 0)).to(device, dtype=torch.bfloat16))
            del pixels
            if self._interrupt:
                raise GenerationAborted
        if batch_size > 1:
            positive, base = positive.repeat(batch_size, 1, 1, 1), base.repeat(batch_size, 1, 1, 1)
            negative = None if negative is None else negative.repeat(batch_size, 1, 1, 1)
            contexts = [value.repeat(batch_size, 1, 1, 1) for value in contexts]
        set_phase_status("Preparing Timesteps")
        with torch.device("cpu"):
            scheduler = FlowMatchEulerDiscreteScheduler(use_dynamic_shifting=True, shift=3.0)
        tokens = (height // 16) * (width // 16)
        mu = (1.15 - 0.5) / (4096 - 256) * (tokens - 256) + 0.5
        scheduler.set_timesteps(sigmas=np.linspace(1.0, 1 / sampling_steps, sampling_steps), device=device, mu=mu)
        latents = torch.cat([torch.randn((1, 32, height // 8, width // 8), generator=torch.Generator(device="cpu").manual_seed(seed + i),
                                       device="cpu", dtype=torch.float32) for i in range(batch_size)]).to(device)
        # Read branching scalars once on CPU, outside graph capture and the hot loop.
        times = scheduler.timesteps.cpu().tolist()
        update_loras_slists(self.transformer, loras_slists, sampling_steps)
        if callback is not None:
            callback(-1, None, True)
        with torch.autocast("cuda", dtype=torch.bfloat16), cuda_graphs.scope():
            for step, t in enumerate(tqdm(scheduler.timesteps, desc="Denoising", mininterval=1/3)):
                if self._interrupt:
                    raise GenerationAborted
                offload.set_step_no_for_lora(self.transformer, step)
                model_input = latents.to(torch.bfloat16)
                timestep = t.expand(batch_size) / 1000
                embedding = positive if times[step] / 1000 <= 0.9 else base
                velocity = self.transformer(model_input, timestep, embedding, context_latents=contexts)[0].float()
                if self._interrupt:
                    raise GenerationAborted
                if negative is not None:
                    unconditional = self.transformer(model_input, timestep, negative, context_latents=contexts)[0].float()
                    velocity = unconditional + guide_scale * (velocity - unconditional)
                    del unconditional
                preview_x0 = None
                if callback is not None and getattr(callback, "tiny_vae", False):
                    if scheduler.step_index is None:
                        scheduler._init_step_index(t)
                    preview_x0 = latents.float() - scheduler.sigmas[scheduler.step_index].to(latents.device) * velocity
                latents = scheduler.step(velocity, t, latents, return_dict=False)[0]
                del velocity, model_input
                if callback is not None:
                    callback(step, self.preview(preview_x0 if preview_x0 is not None else latents), False)
                del preview_x0
        del positive, negative, base, contexts, embedding
        set_phase_status("VAE Decoding")
        with torch.autocast("cuda", dtype=torch.float16):
            images = [self.vae(pack(item.unsqueeze(0)).to(runtime.execution_device(self.vae)), tile_size=tile_size if tiled else 0)
                      for item in latents]
        return torch.cat(images).transpose(0, 1)


def load_pipeline(model_filename, text_encoder_filename, model_def, model_type, save_quantized):
    folder = model_def["text_encoder_folder"]
    assets = {name: fl.locate_file(f"{folder}/{name}") for name in
              ("fibo2_encoder_config.json", "fibo2_processor_config.json", "tokenizer.json", "tokenizer_config.json", "chat_template.jinja")}
    config_path = Path(__file__).with_name(f"{model_def['architecture']}.json")
    config = json.loads(config_path.read_text())
    encoder_config = Qwen3VLConfig.from_json_file(assets["fibo2_encoder_config.json"])
    with init_empty_weights(include_buffers=True):
        transformer = BriaFibo2Transformer2DModel(**config)
        encoder = FiboEncoder(encoder_config)
        vae = FiboVAE()
    encoder.visual.rotary_pos_emb.reset_inv_freq()
    encoder.language_model.rotary_emb.reset_inv_freq()
    offload.load_model_data(transformer, model_filename, writable_tensors=False, default_dtype=torch.bfloat16)
    offload.load_model_data(encoder, text_encoder_filename, writable_tensors=False, default_dtype=torch.bfloat16,
                           fused_split_map=encoder.split_linear_modules_map)
    runtime.lookup_on_cpu(encoder.language_model.embed_tokens)  # the 0.7 GB token table stays in RAM: mmgp looks up the prompt's rows on the CPU
    offload.load_model_data(vae, fl.locate_file("fibo2/flux2_vae_fp16.safetensors"), writable_tensors=False, default_dtype=None)
    for module in (transformer, encoder, vae):
        module.eval().requires_grad_(False)
    tokenizer = AutoTokenizer.from_pretrained(str(Path(assets["tokenizer.json"]).parent), local_files_only=True, extra_special_tokens={})
    image_config = json.loads(Path(assets["fibo2_processor_config.json"]).read_text())["image_processor"]
    image_processor = Qwen2VLImageProcessor(**image_config)
    processor = FiboProcessor(image_processor, tokenizer)
    if save_quantized:
        from wgp import save_quantized_model
        save_quantized_model(transformer, model_type, model_filename[0], torch.bfloat16, str(config_path),
                             convrot_prefixes=("layers.", "noise_refiner.", "perceiver.layers."))
    return FiboPipeline(transformer, encoder, processor, vae, edit_only=model_def["architecture"] == "fibo2_edit_turbo")
