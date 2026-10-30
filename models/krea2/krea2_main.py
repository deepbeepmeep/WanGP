from shared.utils.phase_progress import text_encoding_prompts, text_encoding_progress, generation_progress
import json
import math
import os
from functools import lru_cache

import numpy as np
import torch
from accelerate import init_empty_weights
from einops import rearrange, repeat
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, Qwen2VLImageProcessorFast, Qwen2VLProcessor
from transformers.processing_utils import ProcessorMixin

from mmgp import offload, runtime
from shared.utils import files_locator as fl
from shared.utils.gguf_mapping import has_standard_gguf_tensor_names, remap_state_dict_triplet
from shared.utils.text_encoder_cache import TextEncoderCache

from models.ideogram4.qwen3_vl_configuration import Qwen3VLConfig, register_qwen3_vl_config
from models.ideogram4.qwen3_vl_transformers import Qwen3VLModel, Qwen3VLTextModel, Qwen3VLVisionModel
from models.qwen.vae_variants import load_vae

from .krea2_mmdit import SingleStreamDiT, config_from_diffusers


_TEXT_ENCODER_SELECT_LAYERS = (2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35)
_DEFAULT_NEGATIVE_PROMPT = ""
_TRANSFORMER_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "krea2_transformer_config.json")
_TRANSFORMER_STATE_DICT_PREFIX = "model.diffusion_model."


def _load_json(path):
    with open(path, "r", encoding="utf-8") as reader:
        return json.load(reader)


def preprocess_sd(state_dict):
    if not any(key.startswith(_TRANSFORMER_STATE_DICT_PREFIX) for key in state_dict):
        return state_dict
    prefix_len = len(_TRANSFORMER_STATE_DICT_PREFIX)
    return {key[prefix_len:] if key.startswith(_TRANSFORMER_STATE_DICT_PREFIX) else key: value for key, value in state_dict.items()}


def _timesteps(seq_len, steps, x1, x2, device, y1=0.5, y2=1.15, sigma=1.0, mu=None):
    ts = torch.linspace(1, 0, steps + 1, device=device)
    if mu is None:
        slope = (y2 - y1) / (x2 - x1)
        mu = slope * seq_len + (y1 - slope * x1)
    ts = math.exp(mu) / (math.exp(mu) + (1.0 / ts - 1.0) ** sigma)
    return ts.tolist()


def _prepare(img, txtlen, patch, txtmask):
    b, _, h, w = img.shape
    h_, w_ = h // patch, w // patch
    imgids = torch.zeros((h_, w_, 3), device=img.device)
    imgids[..., 1] = torch.arange(h_, device=img.device)[:, None]
    imgids[..., 2] = torch.arange(w_, device=img.device)[None, :]
    imgpos = repeat(imgids, "h w three -> b (h w) three", b=b, three=3)
    imgmask = torch.ones(b, h_ * w_, device=img.device, dtype=torch.bool)
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)
    txtpos = torch.zeros(b, txtlen, 3, device=img.device)
    mask = torch.cat((txtmask, imgmask), dim=1)
    pos = torch.cat((txtpos, imgpos), dim=1)
    return img, pos, mask


def _pack_image_latents(latents, patch):
    return rearrange(latents, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)


class Krea2TextEncoder(torch.nn.Module):
    def __init__(self, config, with_vision=False):
        super().__init__()
        self.config = config
        if with_vision:
            self.visual = Qwen3VLVisionModel._from_config(config.vision_config)
        self.language_model = Qwen3VLTextModel(config.text_config)

    get_rope_index = Qwen3VLModel.get_rope_index


class Krea2Qwen3VLProcessor(Qwen2VLProcessor):
    attributes = ["image_processor", "tokenizer"]

    def __init__(self, image_processor, tokenizer):
        self.image_token = "<|image_pad|>"
        self.video_token = "<|video_pad|>"
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)
        ProcessorMixin.__init__(self, image_processor, tokenizer, chat_template=getattr(tokenizer, "chat_template", None))


class Qwen3VLConditioner(torch.nn.Module):
    def __init__(self, text_encoder, tokenizer, processor, max_length=512, select_layers=_TEXT_ENCODER_SELECT_LAYERS):
        super().__init__()
        self.qwen = text_encoder
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_length = max_length
        self.select_layers = select_layers
        self.prompt_template_encode_prefix = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n"
        self.prompt_template_encode_suffix = "<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_encode_start_idx = 34
        self.prompt_template_encode_suffix_start_idx = 5

    def _tokenize(self, text: list[str], device, images=None):
        prefix_idx = self.prompt_template_encode_start_idx
        target_device = torch.device(device)
        vision = "" if images is None else "<|vision_start|><|image_pad|><|vision_end|>" * len(images)
        prefixed_text = [self.prompt_template_encode_prefix + vision + item for item in text]
        suffix_text = [self.prompt_template_encode_suffix] * len(text)
        # Tokenizers create PyTorch tensors via the global default device; pin that choice here so MMGP
        # offload state cannot make token tensors bounce through CPU with an unsafe async copy.
        with torch.device(target_device):
            suffix_inputs = self.processor(text=suffix_text, truncation=True, return_tensors="pt").to(target_device)
            if images is None:
                inputs = self.tokenizer(prefixed_text, truncation=True, return_length=False, return_overflowing_tokens=False, padding="max_length", max_length=self.max_length + prefix_idx - self.prompt_template_encode_suffix_start_idx, return_tensors="pt").to(target_device)
            else:
                inputs = self.processor(text=prefixed_text, images=images * len(text), padding="longest", return_tensors="pt").to(target_device)
        suffix_ids = suffix_inputs["input_ids"]
        suffix_mask = suffix_inputs["attention_mask"].bool()
        input_ids = torch.cat([inputs["input_ids"], suffix_ids], dim=1)
        mask = torch.cat([inputs["attention_mask"].bool(), suffix_mask], dim=1)
        position_ids = mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(mask == 0, 1)
        return input_ids, mask, position_ids, prefix_idx, inputs

    @torch.inference_mode()
    def forward(self, text: list[str], device, images=None):
        self.qwen.language_model._interrupt = getattr(self, "_interrupt", False)
        if getattr(self, "_interrupt", False):
            return None, None
        input_ids, mask, position_ids, prefix_idx, inputs = self._tokenize(text, device=device, images=images)
        inputs_embeds = visual_pos_masks = deepstack_visual_embeds = None
        if images is not None:
            image_grid_thw = inputs["image_grid_thw"]
            image_embeds, deepstack_visual_embeds = self.qwen.visual(inputs["pixel_values"].to(self.qwen.visual.dtype), grid_thw=image_grid_thw)
            inputs_embeds = self.qwen.language_model.embed_tokens(input_ids)
            visual_pos_masks = input_ids == self.qwen.config.image_token_id
            inputs_embeds = inputs_embeds.masked_scatter(visual_pos_masks.unsqueeze(-1).expand_as(inputs_embeds), image_embeds.to(inputs_embeds.dtype))
            position_ids, _ = self.qwen.get_rope_index(input_ids, image_grid_thw=image_grid_thw, attention_mask=mask)
        selected_layers = [layer_idx - 1 for layer_idx in self.select_layers]
        with text_encoding_progress(self.qwen.language_model.layers, prompt_count=len(text)):
            states = self.qwen.language_model(input_ids=None if inputs_embeds is not None else input_ids, inputs_embeds=inputs_embeds, attention_mask=mask, position_ids=position_ids, use_cache=False, visual_pos_masks=visual_pos_masks, deepstack_visual_embeds=deepstack_visual_embeds, return_mid_results_layers=selected_layers)
        if states.last_hidden_state is None:
            return None, None
        mid_results = states.mid_results
        hiddens = torch.stack(mid_results, dim=2)
        states.mid_results = None
        del mid_results, states
        hiddens = hiddens[:, prefix_idx:]
        mask = mask[:, prefix_idx:]
        return hiddens, mask


class _TextEncodingInterrupted(Exception):
    pass


def _lora_schedules_are_static_for_modules(model, prefixes):
    scaling = getattr(model, "_loras_scaling", None)
    if not scaling:
        return True
    dynamic_adapters = {name for name, values in scaling.items() if isinstance(values, list) and any(value != values[0] for value in values[1:])}
    if not dynamic_adapters:
        return True
    shortcuts = getattr(model, "_loras_model_shortcuts", None)
    if not shortcuts:
        return True
    for module_name, loras_data in shortcuts.items():
        if module_name.startswith(prefixes) and any(adapter in loras_data for adapter in dynamic_adapters):
            return False
    return True


class Krea2Pipeline:
    def __init__(self, transformer, vae, encoder, dtype=torch.bfloat16):
        self.transformer = transformer
        self.vae = vae
        self.encoder = encoder
        # an entry keeps 12 encoder layers over the 512 padded tokens (~31 MB): the padding rows stay, as they still shape SageAttention's
        # key quantization; the default 100 MB would hold 3 prompts
        self.text_encoder_cache = TextEncoderCache(max_size_mb=512)
        self.dtype = dtype
        self.compression = 8
        self.channels = 16
        self._interrupt = False
        self.transformer._interrupt = False
        self.transformer.txtfusion._interrupt = False
        self.encoder._interrupt = False
        self.encoder.qwen.language_model._interrupt = False

    @property
    def runtime_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else next(self.transformer.parameters()).device)

    def _decode_latents_to_cpu_uint8(self, latents):
        latents = rearrange(latents, "b c h w -> b c 1 h w").to(self.vae.dtype)
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=latents.device).view(1, self.channels, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = torch.tensor(self.vae.config.latents_std, device=latents.device).view(1, self.channels, 1, 1, 1).to(latents.device, latents.dtype)
        latents = (latents * latents_std) + latents_mean
        return self.vae.decode_to_cpu_uint8(latents)[:, :, 0]

    def _encode_image_to_latents(self, image, width, height, device, dtype, fit=False, resize_to_target=True):
        from shared.utils.utils import convert_image_to_tensor

        image = image.convert("RGB")
        if fit:
            image_width, image_height = image.size
            scale = min(height / image_height, width / image_width)
            if image_height * scale >= height * 0.92 and image_width * scale >= width * 0.92:
                scale = max(height / image_height, width / image_width)
                crop_height = min(image_height, round(height / scale))
                crop_width = min(image_width, round(width / scale))
                top, left = (image_height - crop_height) // 2, (image_width - crop_width) // 2
                image = image.crop((left, top, left + crop_width, top + crop_height))
                fit_height, fit_width = height, width
            else:
                align = self.compression * self.transformer.config.patch
                fit_height = min(max(align, int(image_height * scale) // align * align), height)
                fit_width = min(max(align, int(image_width * scale) // align * align), width)
            image = image.resize((fit_width, fit_height), resample=Image.Resampling.BICUBIC)
        elif resize_to_target:
            image = image.resize((width, height), resample=Image.Resampling.LANCZOS)
        tensor = convert_image_to_tensor(image).unsqueeze(0).unsqueeze(2).to(device=device, dtype=self.vae.dtype)
        latents = self.vae.encode(tensor).latent_dist.mode()
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=latents.device).view(1, self.channels, 1, 1, 1).to(latents.device, latents.dtype)
        latents_std = torch.tensor(self.vae.config.latents_std, device=latents.device).view(1, self.channels, 1, 1, 1).to(latents.device, latents.dtype)
        latents = (latents - latents_mean) / latents_std
        return latents[:, :, 0].to(device=device, dtype=dtype)

    def _build_inpaint_mask(self, image_mask, width, height, align, device):
        def mask_tensor(size):
            mask_array = np.array(image_mask.convert("RGBA").resize(size, resample=Image.Resampling.NEAREST))
            alpha = mask_array[..., 3]
            channel = alpha if alpha.min() < 255 else mask_array[..., 0]
            return torch.from_numpy(channel.astype(np.float32)).div_(255.0).ge_(0.5).to(torch.float32)

        mask = mask_tensor((width // align, height // align)).unsqueeze(0)
        mask_rebuilt = mask_tensor((width, height)).unsqueeze(0).unsqueeze(0)
        return mask.reshape(1, -1, 1).to(device), mask_rebuilt

    def _image_to_cpu_uint8(self, image, width, height):
        from shared.utils.utils import convert_image_to_tensor

        image = image.convert("RGB").resize((width, height), resample=Image.Resampling.LANCZOS)
        return convert_image_to_tensor(image).add(1).mul(127.5).round().clamp(0, 255).to(torch.uint8).unsqueeze(0)

    def _encode_prompts(self, prompts, device, dtype, images=None):
        self.encoder._interrupt = self._interrupt
        self.encoder.qwen.language_model._interrupt = self._interrupt

        def encode_fn(prompt_batch):
            hiddens, masks = self.encoder(prompt_batch, device=device, images=images)
            if hiddens is None:
                raise _TextEncodingInterrupted
            return [(hiddens[i], masks[i]) for i in range(len(prompt_batch))]

        try:
            if images is None:
                cache_keys = [(self.encoder.max_length, tuple(self.encoder.select_layers), prompt) for prompt in prompts]
                encoded = self.text_encoder_cache.encode(encode_fn, prompts, device=device, cache_keys=cache_keys)
            else:
                encoded = encode_fn(prompts)
        except _TextEncodingInterrupted:
            return None, None
        hiddens = torch.stack([item[0] for item in encoded], dim=0).to(device=device, dtype=dtype, non_blocking=True)
        masks = torch.stack([item[1] for item in encoded], dim=0).to(device=device, non_blocking=True)
        return hiddens, masks

    @torch.inference_mode()
    def __call__(
        self,
        prompts,
        negative_prompts=None,
        width=1024,
        height=1024,
        steps=28,
        guidance=4.5,
        seed=0,
        y1=0.5,
        y2=1.15,
        mu=None,
        callback=None,
        loras_slists=None,
        source_image=None,
        source_crop=None,
        source_offset=None,
        image_mask=None,
        outpainting_mask=None,
        denoising_strength=1.0,
        masking_strength=1.0,
        model_mode=None,
        NAG_scale: float = 1.0,
        NAG_tau: float = 3.5,
        NAG_alpha: float = 0.5,
        reference_images=None,
        fit_all_references=False,
        reference_offsets=None,
        vae_upsampler=None,
        vae_upsampler_seed: int = 0,
        vae_upsampler_progress_callback=None,
    ):
        patch = self.transformer.config.patch
        align = self.compression * patch
        width, height = int(width), int(height)
        if width % align != 0 or height % align != 0:
            raise ValueError(f"Krea 2 width and height must be divisible by {align}; got {width}x{height}.")
        prompts = [prompts] if isinstance(prompts, str) else prompts
        negative_prompts = [_DEFAULT_NEGATIVE_PROMPT] * len(prompts) if negative_prompts is None else negative_prompts
        device = self.runtime_device
        dtype = self.dtype
        batch_size = len(prompts)
        noise = torch.empty(batch_size, self.channels, height // self.compression, width // self.compression, device=device, dtype=dtype)
        for i in range(batch_size):
            noise[i]= torch.randn(self.channels, height // self.compression, width // self.compression, device=device, dtype=dtype, generator=torch.Generator(device=device).manual_seed(int(seed) + i))
        edit = bool(reference_images)
        grounding_images = None
        if edit:
            grounding_images = []
            for image in reference_images:
                image = image.convert("RGB")
                if max(image.size) > 768:
                    scale = 768 / max(image.size)
                    image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
                grounding_images.append(image)
        pending_prompts = prompts + negative_prompts if guidance > 0 or float(NAG_scale) > 1.0 else prompts
        cache_keys = [(self.encoder.max_length, tuple(self.encoder.select_layers), p) for p in pending_prompts]
        prompt_count = len(pending_prompts) if grounding_images is not None else sum(k not in self.text_encoder_cache._entries for k in dict.fromkeys(cache_keys))
        with text_encoding_prompts(prompt_count):
            txt, txtmask = self._encode_prompts(prompts, device, dtype, images=grounding_images)
            if txt is None:
                return None
            cfg = guidance > 0
            true_cfg_scale = guidance + 1.0 if cfg else 1.0
            NAG = None
            nagtxt = nagtxtmask = None
            context_len = txt.shape[1]
            if float(NAG_scale) > 1.0 and not cfg:
                nagtxt, nagtxtmask = self._encode_prompts(negative_prompts, device, dtype, images=grounding_images)
                if nagtxt is None:
                    return None
                context_len = max(txt.shape[1], nagtxt.shape[1])
                txtmask = torch.cat((txtmask, txtmask.new_zeros(txtmask.shape[0], context_len - txtmask.shape[1])), dim=1)
                nagtxtmask = torch.cat((nagtxtmask, nagtxtmask.new_zeros(nagtxtmask.shape[0], context_len - nagtxtmask.shape[1])), dim=1)
                NAG = {"scale": float(NAG_scale), "tau": float(NAG_tau), "alpha": float(NAG_alpha), "cap_embed_len": context_len, "prefix_len": 0}
            x, pos, mask = _prepare(noise, context_len, patch, txtmask)
            if cfg:
                untxt, untxtmask = self._encode_prompts(negative_prompts, device, dtype, images=grounding_images)
                if untxt is None:
                    return None
                _, unpos, unmask = _prepare(noise, untxt.shape[1], patch, untxtmask)
        x1 = (256 // align) ** 2
        x2 = (1280 // align) ** 2
        ts = _timesteps(x.shape[1], steps, x1, x2, x.device, y1=y1, y2=y2, mu=mu)
        img = x
        reference_tokens = []
        if edit:
            target_grid_h, target_grid_w = height // align, width // align
            reference_positions = []
            reference_masks = []
            reference_offsets = [None] * len(reference_images) if reference_offsets is None else reference_offsets
            for frame_no, (image, reference_offset) in enumerate(zip(reference_images, reference_offsets), start=1):
                latents = self._encode_image_to_latents(image, width, height, device, dtype, fit=reference_offset is None and (fit_all_references or frame_no >= 2), resize_to_target=reference_offset is None)
                grid_h, grid_w = latents.shape[-2] // patch, latents.shape[-1] // patch
                reference_tokens.append(_pack_image_latents(latents, patch).expand(batch_size, -1, -1).contiguous())
                ref_pos = torch.zeros(batch_size, grid_h * grid_w, 3, device=device)
                ref_pos[..., 0] = frame_no
                offset_h, offset_w = ((target_grid_h - grid_h) // 2, (target_grid_w - grid_w) // 2) if reference_offset is None else reference_offset
                ref_pos[..., 1] = (torch.arange(grid_h, device=device) + offset_h).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
                ref_pos[..., 2] = (torch.arange(grid_w, device=device) + offset_w).view(1, -1).expand(grid_h, grid_w).reshape(-1)
                reference_positions.append(ref_pos)
                reference_masks.append(torch.ones(batch_size, grid_h * grid_w, device=device, dtype=torch.bool))
            pos = torch.cat([pos[:, :context_len]] + reference_positions + [pos[:, context_len:]], dim=1)
            mask = torch.cat([mask[:, :context_len]] + reference_masks + [mask[:, context_len:]], dim=1)
            if cfg:
                unpos = torch.cat([unpos[:, :untxt.shape[1]]] + reference_positions + [unpos[:, untxt.shape[1]:]], dim=1)
                unmask = torch.cat([unmask[:, :untxt.shape[1]]] + reference_masks + [unmask[:, untxt.shape[1]:]], dim=1)
        if NAG is not None:
            NAG["query_start"] = context_len + sum(tokens.shape[1] for tokens in reference_tokens)
            NAG["query_end"] = NAG["query_start"] + img.shape[1]
        self.transformer._interrupt = self._interrupt
        model_mode_int = None
        if model_mode is not None:
            try:
                model_mode_int = int(model_mode)
            except (TypeError, ValueError):
                model_mode_int = None
        lanpaint_proc = None
        original_image_latents = None
        image_mask_latents = None
        outpainting_mask_latents = None
        image_mask_rebuilt = None
        first_step = 0
        if source_image is not None and image_mask is not None:
            source_latents = self._encode_image_to_latents(source_image if source_crop is None else source_crop, width, height, device, dtype, resize_to_target=source_crop is None)
            if source_latents.shape[0] == 1 and batch_size > 1:
                source_latents = source_latents.expand(batch_size, -1, -1, -1).contiguous()
            if source_crop is not None:
                source_canvas_latents = torch.zeros_like(noise)
                source_top, source_left = source_offset
                source_canvas_latents[..., source_top:source_top + source_latents.shape[-2], source_left:source_left + source_latents.shape[-1]] = source_latents
                source_latents = source_canvas_latents
            original_image_latents = _pack_image_latents(source_latents, patch)
            image_mask_latents, image_mask_rebuilt = self._build_inpaint_mask(image_mask, width, height, align, device)
            if outpainting_mask is not None:
                outpainting_mask_latents, outpainting_mask_rebuilt = self._build_inpaint_mask(outpainting_mask, width, height, align, device)
                image_mask_latents = torch.maximum(image_mask_latents, outpainting_mask_latents)
                image_mask_rebuilt = torch.maximum(image_mask_rebuilt, outpainting_mask_rebuilt)
            randn = x.clone()
            if model_mode_int in (2, 3, 4, 5):
                from shared.lanpaint import LanPaint

                lanpaint_steps = {2: 2, 3: 5, 4: 10, 5: 15}.get(model_mode_int, 5)
                lanpaint_proc = LanPaint(NSteps=lanpaint_steps, Lambda=16.0, StepSize=0.2, Beta=1.0, Friction=15.0, IS_FLUX=False, IS_FLOW=True, overdamped_fallback=True)
                denoising_strength = 1.0
                masking_strength = 1.0
            if denoising_strength < 1.0:
                first_step = int(len(ts[:-1]) * (1.0 - denoising_strength))
            masked_steps = math.ceil(len(ts[:-1]) * masking_strength)
            latent_noise_factor = ts[first_step]
            if outpainting_mask_latents is None:
                img = original_image_latents * (1.0 - latent_noise_factor) + randn * latent_noise_factor
                ts = ts[first_step:]
        step_offset = 0 if outpainting_mask_latents is not None else first_step
        updated_steps = len(ts) - 1
        if callback is not None:
            callback(-1, None, True, override_num_inference_steps=updated_steps)
        from shared.utils.loras_mutipliers import update_loras_slists
        update_loras_slists(self.transformer, loras_slists, steps)
        context_static = _lora_schedules_are_static_for_modules(self.transformer, ("txtfusion.", "txtmlp."))
        timestep_static = _lora_schedules_are_static_for_modules(self.transformer, ("tmlp.", "tproj."))
        if context_static:
            offload.set_step_no_for_lora(self.transformer, 0)
            self.transformer._interrupt = self._interrupt
            txt_list = [txt]
            txt = None
            txt = self.transformer.prepare_context(txt_list, mask, context_len)
            if txt is None:
                return None
            if NAG is not None:
                nagtxt_list = [nagtxt]
                nagtxt = None
                nagtxt = self.transformer.prepare_context(nagtxt_list, nagtxtmask, context_len)
                if nagtxt is None:
                    return None
            if cfg:
                untxt_list = [untxt]
                untxt = None
                untxt = self.transformer.prepare_context(untxt_list, unmask)
                if untxt is None:
                    return None
        t_values = torch.tensor(ts[:-1], dtype=img.dtype, device=img.device)
        if timestep_static:
            offload.set_step_no_for_lora(self.transformer, 0)
            t_all, tvec_all = self.transformer.prepare_timestep(t_values)
            step_tensors = tuple((t_all[i : i + 1], tvec_all[i : i + 1]) for i in range(updated_steps))
        else:
            step_tensors = []
            for step_no, tcurr in enumerate(t_values):
                offload.set_step_no_for_lora(self.transformer, step_offset + step_no)
                step_tensors.append(self.transformer.prepare_timestep(tcurr[None]))
        torch.cuda.empty_cache()
        for i, (tcurr, tprev) in enumerate(tqdm(list(zip(ts[:-1], ts[1:])), total=updated_steps)):
            offload.set_step_no_for_lora(self.transformer, step_offset + i)
            self.transformer._interrupt = self._interrupt
            if self._interrupt:
                return None
            t, tvec = step_tensors[i]

            def run_model(latents, cfg_scale):
                model_latents = torch.cat(reference_tokens + [latents], dim=1) if edit else latents
                step_txt = txt if context_static else self.transformer.prepare_context(txt, mask, context_len)
                if step_txt is None:
                    return None, None
                step_nagtxt = None
                if NAG is not None:
                    step_nagtxt = nagtxt if context_static else self.transformer.prepare_context(nagtxt, nagtxtmask, context_len)
                    if step_nagtxt is None:
                        return None, None
                if cfg and cfg_scale > 1.0:
                    step_untxt = untxt if context_static else self.transformer.prepare_context(untxt, unmask)
                    if step_untxt is None:
                        return None, None
                    cond, uncond = self.transformer.forward_cfg(img=model_latents, context=step_txt, uncond_context=step_untxt, t=t, tvec=tvec, pos=pos, uncond_pos=unpos, mask=mask, uncond_mask=unmask, target_len=latents.shape[1])
                    if cond is None or uncond is None:
                        return None, None
                    if not torch.isfinite(cond).all() or not torch.isfinite(uncond).all():
                        raise RuntimeError("Krea 2 produced non-finite CFG denoiser predictions.")
                    return cond, uncond
                cond = self.transformer(img=model_latents, context=step_txt, t=t, tvec=tvec, pos=pos, mask=mask, NAG=NAG, neg_context=step_nagtxt, neg_mask=nagtxtmask, target_len=latents.shape[1])
                if cond is None:
                    return None, None
                if not torch.isfinite(cond).all():
                    raise RuntimeError("Krea 2 produced non-finite denoiser predictions.")
                return cond, None

            def cfg_predictions(cond, uncond, cfg_scale, _t):
                if cfg and cfg_scale > 1.0:
                    return uncond + cfg_scale * (cond - uncond)
                return cond

            if lanpaint_proc is not None and i < updated_steps - 1:
                lanpaint_mask = image_mask_latents.expand_as(img).contiguous()
                sigma = torch.full((img.shape[0],), tcurr, dtype=img.dtype, device=img.device)
                img = lanpaint_proc(run_model, cfg_predictions, true_cfg_scale, true_cfg_scale, img, original_image_latents, randn, sigma, lanpaint_mask, height=height, width=width, vae_scale_factor=self.compression)
                if img is None:
                    return None

            cond, uncond = run_model(img, true_cfg_scale)
            if cond is None:
                return None
            v = cfg_predictions(cond, uncond, true_cfg_scale, t)
            step_mask = None
            if image_mask_latents is not None:
                if outpainting_mask_latents is not None and i < first_step:
                    step_mask = outpainting_mask_latents
                elif outpainting_mask_latents is not None and i - first_step < masked_steps:
                    step_mask = image_mask_latents
                elif outpainting_mask_latents is None and i < masked_steps:
                    step_mask = image_mask_latents
            preview_x0 = img - tcurr * v if callback is not None and getattr(callback, "tiny_vae", False) else None
            if step_mask is not None and lanpaint_proc is None:
                v.mul_(step_mask)
            img = img + (tprev - tcurr) * v
            del cond, uncond, v
            if step_mask is not None:
                latent_noise_factor = tprev
                noisy_image = original_image_latents * (1.0 - latent_noise_factor) + randn * latent_noise_factor
                img = noisy_image * (1 - step_mask) + step_mask * img
                if preview_x0 is not None:
                    preview_x0 = original_image_latents * (1 - step_mask) + step_mask * preview_x0
            if callback is not None:
                preview = rearrange(preview_x0 if preview_x0 is not None else img, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=patch, pw=patch, h=height // align, w=width // align)
                callback(i, preview.transpose(0, 1), False, preview_meta=None)
                del preview
            del preview_x0
        if self._interrupt:
            return None
        latents = rearrange(img, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=patch, pw=patch, h=height // align, w=width // align)
        decoded = self._decode_latents_to_cpu_uint8(latents)
        if image_mask_rebuilt is not None and (lanpaint_proc is not None or masking_strength == 1):
            source_pixels = self._image_to_cpu_uint8(source_image, width, height).to(decoded.device, torch.float32)
            mask_pixels = image_mask_rebuilt.to(decoded.device, torch.float32)
            if lanpaint_proc is not None:
                from shared.lanpaint import blend_images_with_mask

                if outpainting_mask is not None:
                    blend_pad = 4
                    source_pixels = torch.nn.functional.pad(source_pixels, (blend_pad,) * 4, mode="replicate")
                    decoded = torch.nn.functional.pad(decoded.to(torch.float32), (blend_pad,) * 4, mode="replicate")
                    mask_pixels = torch.nn.functional.pad(mask_pixels, (blend_pad,) * 4, mode="replicate")
                    decoded = blend_images_with_mask(source_pixels, decoded, mask_pixels, blend_overlap=9)[..., blend_pad:-blend_pad, blend_pad:-blend_pad]
                else:
                    decoded = blend_images_with_mask(source_pixels, decoded, mask_pixels, blend_overlap=9)
                decoded = decoded.round().clamp(0, 255).to(torch.uint8)
            else:
                decoded = (source_pixels * (1 - mask_pixels) + decoded.to(torch.float32) * mask_pixels).round().clamp(0, 255).to(torch.uint8)
        if vae_upsampler is not None:
            if vae_upsampler_progress_callback is not None:
                vae_upsampler_progress_callback("VAE")
            decoded = vae_upsampler.decode_inputs([decoded], [latents], prompt=prompts, seed=vae_upsampler_seed, abort_callback=lambda: self._interrupt, progress_callback=vae_upsampler_progress_callback)
        return decoded


def _load_transformer(model_filename, config_path, dtype):
    config = config_from_diffusers(_load_json(config_path))
    with init_empty_weights(include_buffers=True):
        transformer = SingleStreamDiT(config)
    offload.load_model_data(transformer, model_filename, writable_tensors=False, preprocess_sd=preprocess_sd, default_dtype=dtype)
    transformer.eval().requires_grad_(False)
    return transformer


@lru_cache(maxsize=4)
def _krea2_qwen3_gguf_name_map(
    vocab_size,
    hidden_size,
    intermediate_size,
    num_hidden_layers,
    num_attention_heads,
    num_key_value_heads,
    head_dim,
):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from transformers.modeling_gguf_pytorch_utils import get_gguf_hf_weights_map

    config = Qwen3Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        tie_word_embeddings=False,
    )
    with init_empty_weights():
        model = Qwen3ForCausalLM(config)

    # Qwen3-VL uses the same text-layer layout as Qwen3, but its text-only
    # module omits the leading "model." and has no language-model head.
    return {
        gguf_name: hf_name.removeprefix("model.")
        for gguf_name, hf_name in get_gguf_hf_weights_map(model).items()
        if hf_name.startswith("model.")
    }


def _strip_language_model_name(name):
    prefix = "language_model."
    if name.startswith(prefix):
        return name[len(prefix):]
    marker = "." + prefix
    pos = name.find(marker)
    if pos >= 0:
        return name[pos + len(marker):]
    return None


def _strip_language_model_mapping(mapping):
    if mapping is None:
        return None
    stripped = mapping.__class__()
    for name, value in mapping.items():
        stripped_name = _strip_language_model_name(name)
        if stripped_name is None:
            continue
        if isinstance(value, list):
            value = [_strip_language_model_name(item) or item for item in value]
        stripped[stripped_name] = value
    return stripped


def _build_krea2_text_encoder_preprocessor(text_config):
    map_args = (
        text_config.vocab_size,
        text_config.hidden_size,
        text_config.intermediate_size,
        text_config.num_hidden_layers,
        text_config.num_attention_heads,
        text_config.num_key_value_heads,
        text_config.head_dim,
    )

    def preprocess_state_dict(state_dict, quantization_map=None, tied_weights_map=None):
        if "embed_tokens.weight" in state_dict:
            return state_dict, quantization_map, tied_weights_map

        if has_standard_gguf_tensor_names(state_dict):
            original = (state_dict, quantization_map, tied_weights_map)
            state_dict, quantization_map, tied_weights_map = remap_state_dict_triplet(
                state_dict,
                quantization_map,
                tied_weights_map,
                _krea2_qwen3_gguf_name_map(*map_args),
                keep_unmapped=False,
            )
            if "embed_tokens.weight" not in state_dict:
                return original
            return state_dict, quantization_map, tied_weights_map

        stripped_state_dict = _strip_language_model_mapping(state_dict)
        if not stripped_state_dict or "embed_tokens.weight" not in stripped_state_dict:
            return state_dict, quantization_map, tied_weights_map
        return (
            stripped_state_dict,
            _strip_language_model_mapping(quantization_map),
            _strip_language_model_mapping(tied_weights_map),
        )

    return preprocess_state_dict


def _load_text_encoder(text_encoder_filename, config_path, dtype, with_vision=False):
    register_qwen3_vl_config()
    config = Qwen3VLConfig.from_json_file(config_path)
    with init_empty_weights(include_buffers=True):
        text_encoder = Krea2TextEncoder(config, with_vision=with_vision)
    if with_vision:
        text_encoder.visual.rotary_pos_emb.reset_inv_freq()
    text_encoder.language_model.rotary_emb.reset_inv_freq()
    offload.load_model_data(
        text_encoder.language_model,
        text_encoder_filename,
        writable_tensors=False,
        default_dtype=dtype,
        preprocess_sd=_build_krea2_text_encoder_preprocessor(config.text_config),
    )
    runtime.lookup_on_cpu(text_encoder.language_model.embed_tokens)  # the 0.7 GB token table stays in RAM: mmgp looks up the prompt's rows on the CPU
    if with_vision:  # the shared checkpoint holds the vision tower beside the language model
        offload.load_model_data(
            text_encoder.visual,
            text_encoder_filename,
            modelPrefix="visual",
            writable_tensors=False,
            default_dtype=dtype,
        )
    text_encoder.eval().requires_grad_(False)
    return text_encoder


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
        VAE_upsampling=None,
        save_quantized=False,
        **kwargs,
    ):
        dtype = torch.bfloat16
        self.base_model_type = base_model_type
        self.model_def = model_def
        transformer_filename = model_filename[0] if isinstance(model_filename, (list, tuple)) else model_filename
        config_path = _TRANSFORMER_CONFIG_PATH
        transformer = _load_transformer(transformer_filename, config_path, dtype)
        if save_quantized:
            from wgp import save_quantized_model
            save_quantized_model(transformer, model_type, transformer_filename, dtype, config_path)
        text_encoder_folder = model_def["text_encoder_folder"]
        text_encoder_config_path = fl.locate_file(os.path.join(text_encoder_folder, "config.json"))
        edit = base_model_type in ("krea2_raw_edit", "krea2_turbo_edit")
        text_encoder = _load_text_encoder(text_encoder_filename, text_encoder_config_path, dtype, with_vision=edit)
        tokenizer_config = fl.locate_file(os.path.join(text_encoder_folder, "tokenizer_config.json"))
        fl.locate_file(os.path.join(text_encoder_folder, "tokenizer.json"))
        fl.locate_file(os.path.join(text_encoder_folder, "chat_template.jinja"))
        tokenizer_path = os.path.dirname(tokenizer_config)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, max_length=512, trust_remote_code=True, extra_special_tokens={})
        image_processor = Qwen2VLImageProcessorFast.from_pretrained(fl.locate_file(os.path.join(text_encoder_folder, "preprocessor_config.json")))  # located per file: a same-named folder in another checkpoint root must not shadow it
        processor = Krea2Qwen3VLProcessor(image_processor, tokenizer)
        vae = load_vae(model_def, VAE_upsampling)
        self.pipeline = Krea2Pipeline(transformer, vae, Qwen3VLConditioner(text_encoder, tokenizer, processor), dtype=dtype)
        self.transformer = transformer
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.vae = vae

    @generation_progress
    def generate(
        self,
        seed: int | None = None,
        input_prompt: str = "",
        n_prompt: str | None = None,
        sampling_steps: int = 28,
        width: int = 1024,
        height: int = 1024,
        guide_scale: float = 4.5,
        batch_size: int = 1,
        input_frames=None,
        input_masks=None,
        denoising_strength=1.0,
        masking_strength=1.0,
        model_mode=None,
        NAG_scale: float = 1.0,
        NAG_tau: float = 3.5,
        NAG_alpha: float = 0.5,
        callback=None,
        VAE_tile_size=None,
        loras_slists=None,
        input_ref_images=None,
        video_prompt_type="",
        outpainting_dims=None,
        vae_upsampler=None,
        set_progress_status=None,
        **kwargs,
    ):
        if VAE_tile_size is not None and hasattr(self.vae, "use_tiling"):
            if isinstance(VAE_tile_size, int):
                tiling = VAE_tile_size > 0
                tile_size = max(VAE_tile_size, 0)
            else:
                tiling = bool(VAE_tile_size[0])
                tile_size = VAE_tile_size[1] if len(VAE_tile_size) > 1 else 0
            if tiling:
                self.vae.enable_tiling(tile_sample_min_height=tile_size or None, tile_sample_min_width=tile_size or None)
            else:
                self.vae.disable_tiling()
        identity_edit = self.base_model_type in ("krea2_raw_edit", "krea2_turbo_edit")
        turbo = self.base_model_type in ("krea2_turbo", "krea2_turbo_edit")
        if turbo:
            guide_scale = 0
            kwargs_mu = 1.15
        else:
            kwargs_mu = None
        generator_seed = seed if seed is not None and seed >= 0 else torch.seed()
        prompts = [input_prompt] * int(batch_size)
        control_image = image_mask = None
        if input_frames is not None:
            from shared.utils.utils import convert_tensor_to_image

            control_image = convert_tensor_to_image(input_frames) if torch.is_tensor(input_frames) else input_frames
        if input_masks is not None:
            from shared.utils.utils import convert_tensor_to_image

            image_mask = convert_tensor_to_image(input_masks, mask_levels=True) if torch.is_tensor(input_masks) else input_masks
        reference_images = input_ref_images if "I" in video_prompt_type else None
        if reference_images is not None:
            from shared.utils.utils import convert_tensor_to_image

            reference_images = [convert_tensor_to_image(image) if torch.is_tensor(image) else image for image in reference_images]
        if identity_edit and control_image is not None:
            reference_images = [control_image] + (reference_images or [])
        reference_offsets = outpainting_mask = source_crop = source_offset = None
        if identity_edit and reference_images and outpainting_dims is not None:
            from shared.utils.utils import get_outpainting_frame_location

            align = self.pipeline.compression * self.transformer.config.patch
            source_height, source_width, margin_top, margin_left = get_outpainting_frame_location(height, width, outpainting_dims, 1, quantize_margins=align)
            reference_images[0] = reference_images[0].crop((margin_left, margin_top, margin_left + source_width, margin_top + source_height))
            reference_offsets = [(margin_top // align, margin_left // align)] + [None] * (len(reference_images) - 1)
            if control_image is not None and image_mask is not None and (source_height != height or source_width != width):
                source_mask = np.array(image_mask.convert("RGBA").crop((margin_left, margin_top, margin_left + source_width, margin_top + source_height)))
                source_mask = source_mask[..., 3] if source_mask[..., 3].min() < 255 else source_mask[..., 0]
                empty_source_mask = not np.any(source_mask >= 128)
                if empty_source_mask and str(model_mode) not in {"2", "3", "4", "5"}:
                    image_mask = None
                else:
                    source_crop = reference_images[0]
                    source_offset = margin_top // self.pipeline.compression, margin_left // self.pipeline.compression
                    outpainting_mask = Image.new("L", (width, height), 255)
                    margin_bottom, margin_right = height - margin_top - source_height, width - margin_left - source_width
                    outpainting_mask.paste(0, (margin_left + (align if margin_left else 0), margin_top + (align if margin_top else 0), margin_left + source_width - (align if margin_right else 0), margin_top + source_height - (align if margin_bottom else 0)))
                    if empty_source_mask:
                        image_mask = outpainting_mask
        def _vae_upsampler_progress(_phase, current_step=None, total_steps=None):
            if callable(set_progress_status):
                label = getattr(vae_upsampler, "progress_label", "VAE Spatial Upsampling")
                set_progress_status(f"{label} in Progress" if current_step is None or total_steps is None else f"{label} in Progress ({int(current_step) + 1}/{int(total_steps)})")

        images = self.pipeline(
            prompts,
            negative_prompts=[n_prompt or _DEFAULT_NEGATIVE_PROMPT] * len(prompts),
            width=width,
            height=height,
            steps=sampling_steps,
            guidance=guide_scale,
            seed=generator_seed,
            mu=kwargs_mu,
            callback=callback,
            loras_slists=loras_slists,
            source_image=control_image,
            source_crop=source_crop,
            source_offset=source_offset,
            image_mask=image_mask,
            outpainting_mask=outpainting_mask,
            denoising_strength=denoising_strength,
            masking_strength=masking_strength,
            model_mode=model_mode,
            NAG_scale=NAG_scale,
            NAG_tau=NAG_tau,
            NAG_alpha=NAG_alpha,
            reference_images=reference_images,
            fit_all_references="I" in video_prompt_type and "K" not in video_prompt_type,
            reference_offsets=reference_offsets,
            vae_upsampler=vae_upsampler,
            vae_upsampler_seed=generator_seed,
            vae_upsampler_progress_callback=_vae_upsampler_progress,
        )
        if images is None:
            return None
        return images.transpose(0, 1)

    @property
    def _interrupt(self):
        return getattr(self.pipeline, "_interrupt", False)

    @_interrupt.setter
    def _interrupt(self, value):
        if hasattr(self, "pipeline"):
            self.pipeline._interrupt = value
            self.pipeline.encoder._interrupt = value
            self.pipeline.encoder.qwen.language_model._interrupt = value
        if hasattr(self, "transformer"):
            self.transformer._interrupt = value
            self.transformer.txtfusion._interrupt = value
        if hasattr(self, "text_encoder"):
            self.text_encoder.language_model._interrupt = value
