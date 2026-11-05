# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import json
from pathlib import Path

import numpy as np
import torch
from accelerate import init_empty_weights
from mmgp import offload, runtime
from tqdm import tqdm
from transformers import AutoTokenizer

from shared.utils import files_locator as fl
from shared.utils.phase_progress import generation_progress, set_phase_status, GenerationAborted
from shared.utils.text_encoder_cache import TextEncoderCache
from .transformer import DiT, rope_embed
from .text_encoder import TextEncoder, Qwen3VLConfig, encode_prompt
from .vae import Flux3VAE


def image_ids(height, width, device, reference=0):
    rows, cols = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
    ids = torch.zeros((height * width, 4), device=device, dtype=torch.float32)
    ids[:, 1], ids[:, 2] = rows.flatten() * 0.5, cols.flatten() * 0.5
    ids[:, 3] = -reference
    return ids


class Flux3Pipeline:
    def __init__(self, transformer, text_encoder, tokenizer, vae):
        self.transformer, self.text_encoder = transformer, text_encoder
        self.tokenizer, self.vae = tokenizer, vae
        self.text_encoder_cache = TextEncoderCache()
        self.preview_mean = vae.bn.running_mean.detach().float().cpu().reshape(1, 128, 1, 1)
        self.preview_std = (vae.bn.running_var.detach().float().cpu() + vae.bn_eps).sqrt().reshape(1, 128, 1, 1)
        self._interrupt = False

    def abort(self):
        self._interrupt = True

    @property
    def _interrupt(self):
        return self.transformer._interrupt

    @_interrupt.setter
    def _interrupt(self, value):
        self.transformer._interrupt = self.text_encoder._interrupt = value

    @generation_progress
    @torch.inference_mode()
    def generate(self, input_prompt='', n_prompt='', seed=0, sampling_steps=16, width=1024, height=1024, guide_scale=1.0,
                 batch_size=1, input_ref_images=None, callback=None, VAE_tile_size=None,
                 set_progress_status=None, **kwargs):
        self.transformer._interrupt = self.text_encoder._interrupt = self._interrupt
        device = runtime.execution_device(self.transformer)
        tiled, tile_size = VAE_tile_size if VAE_tile_size is not None else (True, 1024)
        tile_size = tile_size if tiled else 0
        references, reference_ids = [], []
        set_phase_status('Preparing Image Conditioning')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            for index, image in enumerate(input_ref_images or []):
                pixels = torch.from_numpy(np.array(image.convert('RGB'), copy=True)).permute(2, 0, 1).unsqueeze(0)
                pixels = pixels.to(runtime.execution_device(self.vae), dtype=torch.float32).div_(127.5).sub_(1).bfloat16()
                z = self.vae(pixels, encode=True, tile_size=tile_size)
                references.append(z[0].permute(1, 2, 0).reshape(-1, 128).to(device))
                reference_ids.append(image_ids(z.shape[-2], z.shape[-1], device, index + 1))
                del pixels, z
                if self._interrupt:
                    raise GenerationAborted
        reference = torch.cat(references) if references else None
        ref_ids = torch.cat(reference_ids) if references else None
        del references, reference_ids
        use_cfg = not self.transformer.flow_map and guide_scale != 1
        prompts = [n_prompt or '', input_prompt] if use_cfg else [input_prompt]
        texts = self.text_encoder_cache.encode(lambda items: [encode_prompt(self.text_encoder, self.tokenizer, p) for p in items], prompts, device=device)
        set_phase_status('Preparing Timesteps')
        h, w = height // 16, width // 16
        ids = image_ids(h, w, device)
        frequencies = {'image': [rope_embed(ids)] * len(texts), 'text': [], 'single': []}
        if reference is not None:
            frequencies['image_cond'] = [rope_embed(ref_ids)] * len(texts)
        for text in texts:
            txt_ids = torch.zeros((text.shape[0], 4), device=device, dtype=torch.float32)
            txt_ids[:, 3] = torch.arange(text.shape[0], device=device)
            frequencies['text'].append(rope_embed(txt_ids))
            frequencies['single'].append(rope_embed(torch.cat([ids] + ([ref_ids] if ref_ids is not None else []) + [txt_ids])))
        # The released schedule and random generator run on CPU, independent of
        # PyTorch's default device; Euler state remains FP32 between steps.
        schedule = torch.linspace(1, 0, sampling_steps + 1, device='cpu')
        schedule = 16.0 * schedule / (1 + 15.0 * schedule)
        schedule_device = schedule.to(device)
        results = []
        for image_no in range(batch_size):
            generator = torch.Generator(device='cpu').manual_seed(seed + image_no)
            latents = torch.randn((h * w, 128), device='cpu', generator=generator).bfloat16().float().to(device)
            if callback is not None:
                callback(-1, None, True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                for step in tqdm(range(sampling_steps), desc='Denoising', mininterval=1/3):
                    if self._interrupt:
                        raise GenerationAborted
                    predictions = self.transformer(latents, texts, reference, frequencies, schedule_device[step], schedule_device[step + 1] if self.transformer.flow_map else None)
                    velocity = predictions[0] + guide_scale * (predictions[1] - predictions[0]) if use_cfg else predictions[0]
                    preview_x0 = latents - schedule[step].item() * velocity if callback is not None and getattr(callback, "tiny_vae", False) else None
                    latents = latents + (schedule[step + 1] - schedule[step]).item() * velocity
                    if callback is not None:
                        preview = (preview_x0 if preview_x0 is not None else latents).reshape(h, w, 128).permute(2, 0, 1).unsqueeze(0)
                        preview = preview * self.preview_std.to(preview) + self.preview_mean.to(preview)
                        preview = preview.reshape(1, 32, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(1, 32, h * 2, w * 2).transpose(0, 1)
                        callback(step, preview, False)
                        del preview
                    del preview_x0
            z = latents.reshape(h, w, 128).permute(2, 0, 1).unsqueeze(0)
            set_phase_status('VAE Decoding')
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pixels = self.vae(z.to(runtime.execution_device(self.vae), dtype=torch.bfloat16), tile_size=tile_size)
            results.append(pixels[0])
        return torch.stack(results, dim=1)


def load_pipeline(model_filename, text_encoder_filename, base_model_type, model_def, model_type, save_quantized, dtype):
    config_path = Path(__file__).with_name(base_model_type + '.json')
    config = json.loads(config_path.read_text())
    folder = model_def['text_encoder_folder']
    assets = {name: fl.locate_file(f'{folder}/{name}') for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja')}
    encoder_config = Qwen3VLConfig.from_json_file(assets['config.json'])
    tokenizer = AutoTokenizer.from_pretrained(str(Path(assets['tokenizer_config.json']).parent), local_files_only=True, padding_side='right', extra_special_tokens={})
    with init_empty_weights(include_buffers=True):
        transformer = DiT(config, flow_map=base_model_type == 'flux3_klein')
        encoder = TextEncoder(encoder_config)
        vae = Flux3VAE()
    with torch.device('cpu'):
        encoder.language_model.rotary_emb.reset_inv_freq()
    offload.load_model_data(transformer, model_filename, writable_tensors=False, default_dtype=torch.bfloat16, fused_split_map=transformer.split_linear_modules_map)
    offload.load_model_data(encoder, text_encoder_filename, writable_tensors=False, default_dtype=torch.bfloat16, ignore_unused_weights=True)  # the shared Qwen3-VL-4B file: language_model.* used, visual.* ignored
    runtime.lookup_on_cpu(encoder.language_model.embed_tokens)  # the 0.7 GB token table stays in RAM: mmgp looks up the prompt's rows on the CPU
    offload.load_model_data(vae, fl.locate_file('flux3/flux2-vae.safetensors'), writable_tensors=False, default_dtype=torch.bfloat16)
    transformer.eval()
    encoder.eval()
    vae.eval()
    if save_quantized:
        from wgp import save_quantized_model
        save_quantized_model(transformer, model_type, model_filename[0], dtype, str(config_path))
    return Flux3Pipeline(transformer, encoder, tokenizer, vae)
