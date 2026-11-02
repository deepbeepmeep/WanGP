"""Qwen Image 2.1 generation with WanGP progress and MMGP-owned residency."""
import json
import math
import hashlib

import numpy as np
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from tqdm import tqdm

from .processor import load_processor
from shared.utils.phase_progress import check_abort, generation_progress, set_phase_status, vae_decoding_progress
from shared.utils.utils import convert_tensor_to_image, get_outpainting_frame_location
from .prompt import QwenImage21Pipeline
from .transformer import QwenImage21KVCache
from .schedulers import QwenImage21TurboScheduler
from shared.utils.text_encoder_cache import TextEncoderCache


OUTPAINTING_ALIGNMENT = 32
# Developer-only choice; generation settings cannot override this choice.
OUTPAINTING_METHOD = "Red Canvas"  # Alternative: "Reference".


RED_OUTPAINTING_PROMPT = "Remove the red paddings on the sides and show what's behind them."
VIGGLE_SIGMAS = {
    4: (1.0, 0.75, 0.5, 0.25),
    5: (1.0, 0.875, 0.75, 0.5, 0.25),
    6: (1.0, 0.9375, 0.875, 0.75, 0.5, 0.25),
}
PRUNA_SIGMAS = {
    5: (1.0, 0.94, 6 / 7, 2 / 3, 0.4),
    8: (1.0, 14 / 15, 6 / 7, 10 / 13, 2 / 3, 6 / 11, 0.4, 2 / 9),
}


def reference_outpainting_offset(location, width, height):
    """Align the target RoPE grid with the source at its requested canvas position."""
    source_height, source_width, top, left = location
    target_height, target_width = height // 16, width // 16
    source_height, source_width = source_height // 16, source_width // 16
    return (target_height - target_height // 2 - top // 16 - (source_height - source_height // 2),
            target_width - target_width // 2 - left // 16 - (source_width - source_width // 2))


def reference_outpainting_instruction(location, width, height):
    source_height, source_width, top, left = location
    sides = [name for name, present in (("top", top > 0), ("bottom", top + source_height < height),
                                       ("left", left > 0), ("right", left + source_width < width)) if present]
    return f"Extend the scene in <image1> beyond its {', '.join(sides)} border{'s' if len(sides) > 1 else ''} to fill the larger canvas. Keep its existing content, lighting, and perspective."


def red_outpainting_canvas(source, width, height, location):
    from PIL import Image
    _, _, top, left = location
    canvas = Image.new("RGBA", (width, height), (255, 0, 0, 255))
    canvas.paste(source.convert("RGBA"), (left, top))
    return canvas


def interior_edit_mask(input_masks, location):
    """Keep painted source edits; outpainting margins are instruction-only."""
    if input_masks is None:
        return None
    mask = np.array(convert_tensor_to_image(input_masks, mask_levels=True).convert("L"), copy=True)
    h, w, top, left = location
    mask[:top] = 0
    mask[top+h:] = 0
    mask[:, :left] = 0
    mask[:, left+w:] = 0
    return torch.from_numpy(mask).unsqueeze(0).unsqueeze(0) if np.any(mask > 64) else None


def outpainting_overlap(location, width, height):
    """Aligned overlap, capped to retain a core even for small source images."""
    h, w, top, left = location
    def bands(size, before, after):
        count = int(before) + int(after)
        desired = min(256, max(128, (size // 4 // 32) * 32))
        available = ((size - 32) // max(count, 1) // 32) * 32
        band = min(desired, available)
        return (band if before else 0), (band if after else 0)
    upper, lower = bands(h, top > 0, top + h < height)
    leading, trailing = bands(w, left > 0, left + w < width)
    return upper, lower, leading, trailing


def restore_outpainting_source(output, source, location, overlap, edit_mask=None):
    """Restore original pixels with bounded CPU scratch and premultiplied alpha."""
    h, w, top, left = location
    upper, lower, leading, trailing = overlap
    channels = output.shape[1]
    source = np.asarray(source.convert("RGBA" if channels == 4 else "RGB"))
    def ramp(size, before, after):
        values = np.ones(size, dtype=np.float32)
        for band, reverse in ((before, False), (after, True)):
            if band:
                t = np.linspace(0, 1, band, dtype=np.float32)
                smooth = t * t * (3 - 2 * t)
                if reverse:
                    values[-band:] *= smooth[::-1]
                else:
                    values[:band] *= smooth
        return values
    vertical, horizontal = ramp(h, upper, lower), ramp(w, leading, trailing)
    mask = None if edit_mask is None else np.asarray(edit_mask.convert("L"))[top:top+h, left:left+w]
    for batch in range(output.shape[0]):
        for row in range(0, h, 32):
            check_abort()
            stop = min(row + 32, h)
            weight = vertical[row:stop, None] * horizontal[None, :]
            if mask is not None:
                # A white mask explicitly asks to regenerate these source pixels.
                weight *= 1 - mask[row:stop].astype(np.float32) / 255
            weight = weight[..., None]
            target = output[batch, :, top+row:top+stop, left:left+w].permute(1, 2, 0).numpy()
            original = source[row:stop]
            generated = target.astype(np.float32)
            if channels == 4:
                alpha_source = original[..., 3:4].astype(np.float32) / 255
                alpha_generated = generated[..., 3:4] / 255
                alpha = alpha_source * weight + alpha_generated * (1 - weight)
                rgb = (original[..., :3] * alpha_source * weight + generated[..., :3] * alpha_generated * (1 - weight)) / np.maximum(alpha, 1e-8)
                mixed = np.concatenate((rgb, alpha * 255), axis=-1)
            else:
                mixed = original * weight + generated * (1 - weight)
            target[:] = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)
            # Preserve all original bytes, including hidden RGB under zero alpha.
            np.copyto(target, original, where=weight == 1)
            # Zero-weight pixels must remain exact, including transparent RGB.
            np.copyto(target, generated.astype(np.uint8), where=weight == 0)


def crop_outpainting_source(images, width, height, outpainting_dims):
    """Remove the shared preprocessor's padded canvas before either encoder."""
    if outpainting_dims is None or not any(outpainting_dims):
        return images, None
    if not images:
        raise ValueError("Outpainting requires a control image or a main reference image.")
    if images[0].size != (width, height):
        raise ValueError("The outpainting source must match the prepared canvas dimensions.")
    source_height, source_width, top, left = get_outpainting_frame_location(
        height, width, outpainting_dims, 1, quantize_margins=OUTPAINTING_ALIGNMENT)
    location = source_height, source_width, top, left
    if any(value % OUTPAINTING_ALIGNMENT for value in location):
        raise ValueError("Outpainting source dimensions and offsets must align to 32 pixels.")
    if min(source_height, source_width) < OUTPAINTING_ALIGNMENT:
        raise ValueError("Outpainting must retain at least a 32-pixel source in each dimension.")
    source = images[0].crop((left, top, left + source_width, top + source_height))
    return [source, *images[1:]], location


def source_latent_canvas(source, width, height, location):
    """Place already normalized source latents; padding never enters the VAE."""
    if location is None:
        return source
    source_height, source_width, top, left = location
    if source.shape[-2:] != (source_height // 16, source_width // 16):
        raise ValueError("Encoded outpainting source does not match the aligned crop.")
    canvas = source.new_zeros((*source.shape[:-2], height // 16, width // 16))
    canvas[..., top // 16:(top + source_height) // 16, left // 16:(left + source_width) // 16].copy_(source)
    return canvas


class Qwen21Pipeline(QwenImage21Pipeline):
    def __init__(self, transformer, text_encoder, vae, processor, scheduler_config):
        self.transformer, self.text_encoder, self.vae = transformer, text_encoder, vae
        self.processor, self.tokenizer = processor, processor.tokenizer
        self._interrupt = False
        self.text_encoder_cache = TextEncoderCache()
        self.vae_scale_factor = 16
        with open(scheduler_config, encoding="utf-8") as reader:
            self.scheduler_config = json.load(reader)
        self.sys_prompt = "Comprehend and analyze the provided prompt."
        system = f"<|im_start|>system\n{self.sys_prompt}<|im_end|>\n"
        self.prompt_template_t2i = system + "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
        self.prompt_template_ti2i = system + "<|im_start|>user\n<image1><|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
        self._drop_idx = len(self.tokenizer.encode(system))
        self._img_token_id = self.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    def _vae_stats(self, latents):
        mean = latents.new_tensor(self.vae.config.latents_mean).view(1, -1, 1, 1, 1)
        std = latents.new_tensor(self.vae.config.latents_std).view(1, -1, 1, 1, 1)
        return mean, std

    def _encode_conditioning(self, prompts, images, device, batch_size):
        image_keys = []
        for image in images:
            check_abort()
            image_keys.append((image.size, hashlib.sha256(image.convert("RGBA").tobytes()).digest()))
        context = (id(self.text_encoder), getattr(self.text_encoder, "_model_dtype", None),
                   id(self.tokenizer), self.prompt_template_t2i, self.prompt_template_ti2i,
                   self._drop_idx, self._img_token_id, json.dumps(self.processor.config, sort_keys=True), tuple(image_keys))
        def encode(prompts_to_encode):
            check_abort()
            result = self.encode_prompt(prompt=prompts_to_encode[0], image=images or None,
                                        device=device, num_images_per_prompt=1)
            check_abort()
            return [result]
        check_abort()
        contexts = self.text_encoder_cache.encode(encode, prompts, device=device,
                                                  cache_keys=[(prompt, context) for prompt in prompts])
        check_abort()
        return [(embeds.repeat(batch_size, 1, 1),
                 None if mask is None else mask.repeat(batch_size, 1),
                 slots.repeat(batch_size, 1)) for embeds, mask, slots in contexts]

    @generation_progress
    @torch.inference_mode()
    def generate(self, input_prompt, seed=1, n_prompt=None, sampling_steps=40, sample_solver="default", input_ref_images=None, input_frames=None, input_masks=None, width=1024, height=1024, guide_scale=4.0, batch_size=1, joint_pass=True, VAE_tile_size=None, denoising_strength=1.0, masking_strength=1.0, model_mode=0, loras_slists=None, NAG_scale=1.0, NAG_tau=3.5, NAG_alpha=0.5, callback=None, set_progress_status=None, outpainting_dims=None, custom_settings=None, **kwargs):
        if sample_solver == "qwen21_turbo" and sampling_steps != len(QwenImage21TurboScheduler.sample_sigmas):
            raise ValueError("Qwen Image 2.1 Turbo scheduler requires 8 inference steps.")
        device = torch.device("cuda")
        output_channels = 4 if (custom_settings or {}).get("rgba", "Disabled") == "Enabled" else 3
        kv_cache_mode = (custom_settings or {}).get("qwen21_kv_cache", "Disabled")
        use_kv_cache = kv_cache_mode in ("Enabled", "INT8", "INT8_RAM")
        border_method = OUTPAINTING_METHOD
        if border_method not in ("Red Canvas", "Reference"):
            raise ValueError(f"Unknown outpainting border method: {border_method}")
        generator = torch.Generator(device=device).manual_seed(seed)
        images = list(input_ref_images) if input_ref_images else []
        if input_frames is not None:
            images.insert(0, convert_tensor_to_image(input_frames))
        if len(images) > 10:
            raise ValueError("Qwen Image 2.1 accepts at most 10 reference images.")
        if width % 32 or height % 32:
            raise ValueError("Qwen Image 2.1 dimensions must be multiples of 32.")
        images, source_location = crop_outpainting_source(images, width, height, outpainting_dims)
        instruction_outpainting = source_location is not None and border_method == "Red Canvas"
        reference_outpainting = source_location is not None and border_method == "Reference"
        target_rope_offset = reference_outpainting_offset(source_location, width, height) if reference_outpainting else (0, 0)
        if instruction_outpainting:
            images[0] = red_outpainting_canvas(images[0], width, height, source_location)
            input_masks = interior_edit_mask(input_masks, source_location)
            if RED_OUTPAINTING_PROMPT not in input_prompt:
                input_prompt = input_prompt.rstrip().rstrip(".") + ". " + RED_OUTPAINTING_PROMPT
        elif reference_outpainting:
            if input_masks is not None:
                raise ValueError("Reference outpainting cannot combine an edit mask; use Red Canvas for masked edits.")
            input_prompt = input_prompt.rstrip().rstrip(".") + ". " + reference_outpainting_instruction(source_location, width, height)
        reference_pixels = max((image.width * image.height for image in images), default=0)
        self.vae.configure_tiling(VAE_tile_size, width, height, batch_size,
                                  reference_pixels=reference_pixels)
        self.vae.use_slicing = True
        nag_parameters = (NAG_scale, NAG_tau, NAG_alpha) if NAG_scale > 1 and guide_scale <= 1 else None
        branches, caches = [], []
        try:
            prompts = [input_prompt, n_prompt or " "] if guide_scale > 1 or nag_parameters else [input_prompt]
            for embeds, mask, slots in self._encode_conditioning(prompts, images, device, batch_size):
                branches.append({"encoder_hidden_states": embeds, "encoder_hidden_states_mask": mask, "img_mask": slots})
            set_phase_status("Preparing Image Conditioning")
            condition_latents, shapes = [], []
            original = None
            for image_index, image in enumerate(images):
                check_abort()
                array = np.array(image.convert("RGBA"), copy=True)
                pixels = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).unsqueeze(2).to(device=device, dtype=self.vae.dtype).div_(127.5).sub_(1)
                encoded = self.vae.encode(pixels).latent_dist.mode()
                mean, std = self._vae_stats(encoded)
                encoded = (encoded - mean) / std
                if image_index == 0 and (input_masks is not None or reference_outpainting):
                    source_latents = source_latent_canvas(encoded, width, height, source_location if reference_outpainting else None)
                    original = source_latents.flatten(2).transpose(1, 2).expand(batch_size, -1, -1)
                shapes.append((1, encoded.shape[-2], encoded.shape[-1]))
                condition_latents.append(encoded.flatten(2).transpose(1, 2).expand(batch_size, -1, -1))
                del encoded, pixels, mean, std
            dtype = branches[0]["encoder_hidden_states"].dtype
            latents = torch.randn((batch_size, 64, height // 16, width // 16), device=device, dtype=dtype, generator=generator).flatten(2).transpose(1, 2)
            condition = torch.cat(condition_latents, dim=1).to(dtype) if condition_latents else None
            original = original.to(dtype) if original is not None else None
            latent_mask = noise_source = user_mask = reference_anchor_mask = None
            if input_masks is not None:
                from PIL import Image
                mask_image = convert_tensor_to_image(input_masks, mask_levels=True).convert("L")
                mask_image = mask_image.resize((width // 16, height // 16), Image.Resampling.LANCZOS)
                latent_mask = torch.from_numpy(np.array(mask_image, copy=True)).to(device=device).gt(64).to(dtype).reshape(1, -1, 1)
                user_mask = latent_mask
            if reference_outpainting:
                source_height, source_width, top, left = source_location
                upper, lower, leading, trailing = outpainting_overlap(source_location, width, height)
                reference_anchor_mask = latents.new_ones(1, height // 16, width // 16)
                reference_anchor_mask[:, (top + upper) // 16:(top + source_height - lower) // 16,
                                      (left + leading) // 16:(left + source_width - trailing) // 16] = 0
                reference_anchor_mask = reference_anchor_mask.reshape(1, -1, 1)
                noise_source = latents.clone()
            del condition_latents
            shapes.append((1, height // 16, width // 16))
            for branch in branches:
                slots = branch["img_mask"]
                branch["img_mask"] = torch.cat([slots, slots.new_ones(slots.shape[0], latents.shape[1] // 4)], dim=1)
                branch["img_shapes"] = [shapes] * batch_size
                branch["target_rope_offset"] = target_rope_offset
                # INT8_RAM: the layers in RAM, streamed to the GPU with the blocks (NAG reads its cache outside the blocks: kept in VRAM);
                # Enabled / INT8: in VRAM as far as it fits beside the weights, the other layers in RAM and streamed
                cache = QwenImage21KVCache(self.transformer.transformer_blocks, quantize=kv_cache_mode in ("INT8", "INT8_RAM"), native=nag_parameters is None,
                                           ram=kv_cache_mode == "INT8_RAM" and nag_parameters is None,
                                           dynamic=kv_cache_mode in ("Enabled", "INT8")) if use_kv_cache else None
                branch["kv_cache"] = cache
                if cache is not None:
                    caches.append(cache)
            del branch, embeds, mask, slots
            # Terminal stretching needs at least two sigma values. A single
            # step must go directly from full noise to zero without 0/0.
            from mmgp import offload
            if loras_slists is not None:
                from shared.utils.loras_mutipliers import update_loras_slists
                update_loras_slists(self.transformer, loras_slists, sampling_steps)
            viggle = sample_solver == "viggle_v02"
            pruna = sample_solver == "pruna"
            turbo = sample_solver == "qwen21_turbo"
            if viggle and sampling_steps not in VIGGLE_SIGMAS:
                raise ValueError("Viggle Turbo scheduler supports 4, 5 or 6 inference steps.")
            if pruna and sampling_steps not in PRUNA_SIGMAS:
                raise ValueError("Pruna scheduler supports 5 or 8 inference steps.")
            scheduler_overrides = ({"use_dynamic_shifting": False, "shift": 1.0, "shift_terminal": None} if pruna
                                   else {"shift_terminal": None} if sampling_steps == 1 or viggle else {})
            scheduler_class = QwenImage21TurboScheduler if turbo else FlowMatchEulerDiscreteScheduler
            scheduler = scheduler_class.from_config(
                self.scheduler_config, **scheduler_overrides)
            cfg = scheduler.config
            sigmas = (PRUNA_SIGMAS[sampling_steps] if pruna else VIGGLE_SIGMAS[sampling_steps] if viggle
                      else np.linspace(1, 1 / sampling_steps, sampling_steps))
            if turbo:
                scheduler.set_timesteps(sampling_steps, device=device)
            elif cfg.use_dynamic_shifting:
                slope = (cfg.max_shift - cfg.base_shift) / (cfg.max_image_seq_len - cfg.base_image_seq_len)
                mu = latents.shape[1] * slope + cfg.base_shift - slope * cfg.base_image_seq_len
                scheduler.set_timesteps(sampling_steps, device=device, sigmas=sigmas, mu=mu)
            else:
                scheduler.set_timesteps(sampling_steps, device=device, sigmas=sigmas)
            if viggle:
                print(f"Viggle Turbo Scheduler - Raw Sigmas: {list(sigmas)}; Applied Sigmas: {[round(value, 6) for value in scheduler.sigmas.tolist()]}; Terminal Shift: {scheduler.config.shift_terminal}")
            elif pruna:
                print(f"Pruna Scheduler - Sigmas: {[round(value, 6) for value in scheduler.sigmas.tolist()]}; Dynamic Shift: {cfg.use_dynamic_shifting}; Shift: {cfg.shift}; Terminal Shift: {cfg.shift_terminal}")
            elif turbo:
                print(f"Qwen Image 2.1 Turbo Scheduler - Sigmas: {[round(value, 6) for value in scheduler.sigmas.tolist()]}")
            first_step = 0
            lanpaint = None
            if latent_mask is not None:
                if model_mode in (2, 3, 4, 5) and not instruction_outpainting:
                    from shared.lanpaint import LanPaint
                    lanpaint = LanPaint(NSteps={2: 2, 3: 5, 4: 10, 5: 15}[model_mode])
                    denoising_strength = masking_strength = 1.0
                first_step = min(int(sampling_steps * (1 - denoising_strength)), sampling_steps - 1)
                noise_source = latents.clone()
                if not instruction_outpainting:
                    sigma = scheduler.sigmas[first_step]
                    latents = original * (1 - sigma) + noise_source * sigma
            # New margins always start from full noise, even when an inner
            # masked edit uses a lower denoising strength (as in Krea2).
            step_offset = 0 if instruction_outpainting else first_step
            scheduler.set_begin_index(step_offset)
            timesteps = scheduler.timesteps[step_offset:]
            masked_steps = math.ceil((len(scheduler.timesteps) - first_step) * masking_strength)
            # Prefix activations depend on adapter weights, even though their timestep is zero.
            dynamic_loras = loras_slists is not None and any(isinstance(value, list) for key in ("phase1", "phase2", "phase3") for value in loras_slists.get(key, []))
            set_phase_status("Denoising")
            if callback:
                callback(-1, None, True, override_num_inference_steps=len(timesteps))
            for i, timestep in enumerate(tqdm(timesteps, desc="Denoising")):
                check_abort()
                offload.set_step_no_for_lora(self.transformer, step_offset + i)
                if dynamic_loras:
                    for cache in caches:
                        cache.clear(keep_ring=True)
                def denoise(value, guidance):
                    model_input = torch.cat([condition, value], dim=1) if condition is not None else value
                    inputs = [{**branch, "hidden_states": model_input, "timestep": timestep.expand(batch_size).to(dtype) / 1000, "kv_cache_mode": None if branch["kv_cache"] is None else ("extract" if branch["kv_cache"].layer_caches[0].empty() else "cached"), "return_dict": False} for branch in branches]
                    predictions = self.transformer(branches=inputs, nag_parameters=nag_parameters) if nag_parameters or (joint_pass and len(inputs) > 1) else [self.transformer(**item)[0] for item in inputs]
                    return predictions[0][:, -value.shape[1]:], predictions[1][:, -value.shape[1]:] if len(predictions) > 1 else None
                def combine(positive, negative, guidance, time):
                    return positive if negative is None else negative + guidance * (positive - negative)
                if lanpaint is not None and i < len(timesteps) - 1:
                    from shared.lanpaint import _pack_latents, _unpack_latents
                    def pack(value):
                        return _pack_latents(value.transpose(1, 2).reshape(batch_size, 64, 1, height // 16, width // 16))
                    def unpack(value):
                        return _unpack_latents(value, height, width, 16).flatten(2).transpose(1, 2)
                    def packed_denoise(value, guidance):
                        positive, negative = denoise(unpack(value), guidance)
                        return pack(positive), None if negative is None else pack(negative)
                    latents = unpack(lanpaint(packed_denoise, combine, guide_scale, 1.0, pack(latents), pack(original), pack(noise_source), timestep / 1000, pack(latent_mask.expand(batch_size, -1, 64)), height=height, width=width, vae_scale_factor=16))
                positive, negative = denoise(latents, guide_scale)
                noise = combine(positive, negative, guide_scale, timestep)
                latents = scheduler.step(noise, timestep, latents, return_dict=False)[0]
                absolute_step = step_offset + i
                step_mask = latent_mask
                if instruction_outpainting:
                    # Delay only painted interior pixels until their requested
                    # noise level. The red margins always run the full schedule.
                    step_mask = 1 - user_mask if user_mask is not None and absolute_step < first_step else None
                if reference_anchor_mask is not None:
                    sigma = scheduler.sigmas[absolute_step + 1]
                    # Keep the source core on the same fixed-noise trajectory.
                    # Release a wide border band gradually for a coherent join.
                    step_mask = reference_anchor_mask.reshape(1, height // 16, width // 16).clone()
                    region = step_mask[:, top // 16:(top + source_height) // 16, left // 16:(left + source_width) // 16]
                    region.mul_((1 - sigma).pow(4))
                    # Cropped VAE edge features must be regenerated at every step.
                    if upper: region[:, :2, :] = 1
                    if lower: region[:, -2:, :] = 1
                    if leading: region[:, :, :2] = 1
                    if trailing: region[:, :, -2:] = 1
                    step_mask = step_mask.reshape(1, -1, 1)
                if step_mask is not None and (reference_anchor_mask is not None or absolute_step < first_step + masked_steps):
                    if reference_anchor_mask is None:
                        sigma = scheduler.sigmas[absolute_step + 1]
                    known = original * (1 - sigma) + noise_source * sigma
                    latents = known * (1 - step_mask) + latents * step_mask
                if callback:
                    callback(i, latents.transpose(1, 2).reshape(batch_size, 64, height // 16, width // 16).transpose(0, 1), False)
            for cache in caches:
                cache.clear()
            caches.clear()
            branches.clear()
            set_phase_status("VAE Decoding")
            latents = latents.transpose(1, 2).reshape(batch_size, 64, 1, height // 16, width // 16).to(self.vae.dtype)
            mean, std = self._vae_stats(latents)
            latents = latents * std + mean
            tile_count = 1
            if self.vae.use_tiling and (height > self.vae.tile_sample_min_height or width > self.vae.tile_sample_min_width):
                tile_count = math.ceil((height // 16) / (self.vae.tile_sample_stride_height // 16)) * math.ceil((width // 16) / (self.vae.tile_sample_stride_width // 16))
            print(f"VAE Decoding - {batch_size * tile_count} {'Tiles' if tile_count > 1 else 'Images'}")
            latent_holder = [latents]
            del latents, condition, positive, negative, noise, original, latent_mask, noise_source, reference_anchor_mask, step_mask, user_mask
            with vae_decoding_progress(batch_size * tile_count, self.vae.decoder, cleanup=self.vae.clear_cache):
                output = self.vae.decode_to_cpu_uint8(latent_holder, output_channels=output_channels)[:, :, 0]
            return output.transpose(0, 1)
        finally:
            for cache in caches:
                cache.clear()
            caches.clear()
            branches.clear()
            self.vae.clear_cache()
            self.transformer.pos_embed.freqs = [freq.cpu() for freq in self.transformer.pos_embed.freqs]
