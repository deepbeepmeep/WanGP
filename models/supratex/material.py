# SUPRA-Tex Image-to-Material Translation, inference port of supra3d/supra-tex (deployment/runtime/qwen/examples/qwen_image_21/pbr/
# infer.py, inference_utils.py, diffsynth/pipelines/qwen_image_21_pbr.py and supratex/frontend/qwen_image.py, MIT License) on WanGP's
# Qwen Image 2.1 port (models/qwen21): one endpoint prediction of the albedo and metallic / roughness latents of the object image by
# the Qwen Image 2.1 transformer with the SUPRA-Tex Material Adapter (attention LoRA and modality embeddings, Qwen Research License,
# see LICENSE_Qwen_Research.txt in the checkpoint folder: non-commercial use).
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import numpy as np
import torch

from shared.utils.phase_progress import check_abort, set_phase_status

PROMPT = ("Generate aligned albedo and metallic/roughness material maps for the object in the input image. "
          "Pack MRA with red=metallic, green=roughness, and blue=constant one placeholder. "
          "No observed ambient occlusion is available.")
RESOLUTION = 1024
NOISE_SEED = 42  # the released inference contract fixes the endpoint noise


class MaterialEstimator:
    """Albedo and MRA maps (metallic, roughness, coverage) of a square RGBA object image, RESOLUTION x RESOLUTION."""

    def __init__(self, qwen, embeddings, lora_path):
        self.qwen, self.embeddings, self.lora_path = qwen, embeddings, lora_path
        self.adapter_loaded = False

    def _load_adapter(self):
        """The adapter's LoRA in the Qwen transformer, once MMGP manages the model (its LoRA slots exist after profiling)."""
        if self.adapter_loaded:
            return
        from mmgp import offload

        transformer = self.qwen.transformer
        offload.load_loras_into_model(transformer, [self.lora_path], [1.0], activate_all_loras=True, preprocess_sd=lambda sd: transformer.preprocess_loras("qwen_image_21_7B", sd),
                                      split_linear_modules_map=getattr(transformer, "split_linear_modules_map", None))
        if transformer._loras_errors:
            raise RuntimeError(f"SUPRA-Tex material adapter: {transformer._loras_errors}")
        self.adapter_loaded = True

    @torch.inference_mode()
    def __call__(self, square, device):
        """square: PIL RGBA RESOLUTION^2 (straight colors). Returns albedo (3, R, R) and mra (3, R, R) FP32 in [0, 1] on device."""
        self._load_adapter()
        qwen = self.qwen
        embeds, mask, slots = qwen._encode_conditioning([PROMPT], [square], device, 1)[0]
        dtype = embeds.dtype
        set_phase_status("Estimating Materials")
        pixels = torch.from_numpy(np.array(square, copy=True)).permute(2, 0, 1)[None, :, None].to(device=device, dtype=qwen.vae.dtype).div_(127.5).sub_(1)
        qwen.vae.configure_tiling(None, RESOLUTION, RESOLUTION, 2)
        encoded = qwen.vae.encode(pixels).latent_dist.mode()
        mean, std = qwen._vae_stats(encoded)
        condition = ((encoded - mean) / std)[:, :, 0].flatten(2).transpose(1, 2).to(dtype)
        del pixels, encoded
        side = RESOLUTION // 16
        noise = torch.randn((1, 2, 64, side, side), generator=torch.Generator(device="cpu").manual_seed(NOISE_SEED), dtype=torch.float32, device="cpu").to(device=device, dtype=dtype)
        embeddings = self.embeddings.to(device=device, dtype=dtype)
        targets = [noise[:, index].flatten(2).transpose(1, 2) + embeddings[index] for index in range(2)]
        hidden = torch.cat([condition, *targets], dim=1)
        del condition, targets
        branch = {"hidden_states": hidden, "timestep": torch.ones(1, device=device, dtype=dtype), "encoder_hidden_states": embeds, "encoder_hidden_states_mask": mask,
                  "img_mask": torch.cat([slots, slots.new_ones(1, 2 * side * side // 4)], dim=1), "img_shapes": [[(1, side, side)] * 3],
                  "kv_cache": None, "kv_cache_mode": None, "return_dict": False, "num_target_images": 2}
        check_abort()
        velocity = qwen.transformer(branches=[branch])[0][:, -2 * side * side:]
        del branch, hidden
        latents = noise.float() - velocity.float().view(1, 2, side * side, 64).transpose(2, 3).reshape(1, 2, 64, side, side)
        del velocity, noise
        set_phase_status("Decoding Materials")
        latents = latents[0].unsqueeze(2).to(qwen.vae.dtype)
        mean, std = qwen._vae_stats(latents)
        maps = torch.cat([qwen.vae.decode((latents[index:index + 1] * std + mean)).sample for index in range(2)])[:, :3, 0]
        qwen.vae.clear_cache()
        maps = maps.float().add_(1).div_(2).mul_(255).round_().clamp_(0, 255).div_(255)  # 8-bit maps (upstream saves PNGs)
        coverage = torch.from_numpy(np.array(square.getchannel("A"), copy=True)).to(device).float().div_(255)
        albedo, mra = maps[0], maps[1]
        mra[2] = coverage
        return albedo, mra
