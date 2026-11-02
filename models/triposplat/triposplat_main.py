# TripoSplat for WanGP: one object image to 3D Gaussian splats, from VAST-AI-Research/TripoSplat triposplat.py (TripoSplatPipeline; MIT
# License, Copyright (c) 2026 VAST, see LICENSE in this folder). Modified by DeepBeepMeep for WanGP: model loading through MMGP, background
# removal with rembg's BiRefNet General session, WanGP's FLUX.2 VAE encoder and checkpoint, conditioning cache, joint-pass CFG,
# random draws from the generation's generator, glTF axes and the WanGP splat contract (INRIA PLY bytes and previews).
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import hashlib
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn
from accelerate import init_empty_weights
from mmgp import offload
from PIL import Image, ImageFilter
from tqdm import tqdm

from models.flux.modules.autoencoder_flux2 import AutoEncoderParamsFlux2, Encoder
from shared.mesh3d.render import splat_preview
from shared.mesh3d.splats import splats_to_ply
from shared.utils import files_locator as fl
from shared.utils.conv_bands import band_convs
from shared.utils.phase_progress import PhaseProgress, check_abort, generation_progress, set_phase_status, text_encoding_progress
from shared.utils.text_encoder_cache import TextEncoderCache
from .decoder import TripoSplatDecoder, fused_split_map as decoder_split_map
from .dinov3 import DINOv3ViT
from .flow import TripoSplatFlow, fused_split_map as flow_split_map
from .settings import DEFAULT_CUSTOM_SETTINGS

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "triposplat.json")
DECODER_FILE = "triposplat/triposplat_vae_decoder_fp16.safetensors"
IMAGE_ENCODER_FILE = "dinov3_vith16plus/dinov3_vith16plus_lvd1689m_bf16.safetensors"  # DINOv3 License (LICENSE_DINOv3.md beside it)
VAE_FILE = "flux2_vae.safetensors"  # shared with FLUX.2: TripoSplat's copy has the same encoder weights and latent statistics
IMAGE_MEAN, IMAGE_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
PREVIEW_SIZE = 512


class Flux2ImageEncoder(nn.Module):
    """FLUX.2 VAE encoder (WanGP's implementation) with TripoSplat's latent statistics (upstream Flux2VAEEncoder.encode, BatchNorm eps)."""

    def __init__(self, bn_eps):
        super().__init__()
        params = AutoEncoderParamsFlux2()
        self.encoder = Encoder(resolution=params.resolution, in_channels=params.in_channels, ch=params.ch, ch_mult=params.ch_mult,
                               num_res_blocks=params.num_res_blocks, z_channels=params.z_channels)
        self.bn = nn.BatchNorm2d(4 * params.z_channels, eps=bn_eps, affine=False)
        band_convs(self)

    def forward(self, pixels):
        """pixels (1, 3, H, W) FP32 in [0, 1]. Returns the posterior mean and log-variance (1, 32, H / 8, W / 8) and the per-channel mean
        and standard deviation of the patchified latents (128,), in the weight dtype (BF16, as upstream: its statistics too)."""
        dtype = self.encoder.conv_in.weight.dtype
        mean, logvar = self.encoder(pixels.to(dtype) * 2 - 1).chunk(2, dim=1)
        return mean, logvar, self.bn.running_mean.to(dtype), torch.sqrt(self.bn.running_var.to(dtype) + self.bn.eps)


def _load(module, filename, **kwargs):
    """Checkpoint-native dtypes, read-only mmap. MMGP keeps them (no FP32 conversion) and does not cast the FP32 tensors given to the
    model and to its top-level layers (timesteps, point coordinates, image features) to the model dtype."""
    offload.load_model_data(module, filename, writable_tensors=False, default_dtype=None, **kwargs)
    module._convertWeightsFloatTo = None
    for child in [module, *module.children()]:
        if not isinstance(child, nn.ModuleList):
            child._lock_dtype = None
    return module.eval().requires_grad_(False)


def _remove_background(image):
    """RGB image with the BiRefNet General matte as its alpha (the colours are kept, as upstream's BiRefNet cutout)."""
    from rembg import remove
    from preprocessing.rembg.assets import OBJECT_MODEL
    from shared.utils.utils import new_rembg_session

    rgba = image.copy()
    rgba.putalpha(remove(image, session=new_rembg_session(OBJECT_MODEL), only_mask=True))
    return rgba


def prepare_image(image, canvas, margin, erode_radius):
    """Upstream preprocess_image: the shorter side resized to canvas, the background removed unless the image has a real alpha channel,
    the matte eroded, a square of margin times the object size cropped around it, resized to canvas, and composited on black."""
    scale = canvas / min(image.size)
    image = image.resize((max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))), Image.LANCZOS)
    if not (image.mode == "RGBA" and image.getextrema()[3][0] < 255):
        set_phase_status("Removing Background")
        image = _remove_background(image.convert("RGB"))
    if erode_radius > 0:
        image.putalpha(image.getchannel(3).filter(ImageFilter.MinFilter(2 * erode_radius + 1)))
    ys, xs = np.nonzero(np.array(image.getchannel(3)))
    if len(xs) == 0:
        raise ValueError("No object was found in the Object Image: use a picture of one object, or an RGBA image with its alpha mask.")
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    half = max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 * margin
    image = image.crop([int(cx - half), int(cy - half), int(cx + half), int(cy + half)]).resize((canvas, canvas), Image.LANCZOS)
    composite = Image.new("RGB", (canvas, canvas), (0, 0, 0))
    composite.paste(image, mask=image.getchannel(3))
    return composite


def splats_to_gltf(splats):
    """Upstream axes (Z up, front towards -Y) to glTF (Y up, front towards +Z): a -90 degree turn about X of the means and rotations."""
    x, y, z = splats["means"].unbind(1)
    w, qx, qy, qz = splats["rotations"].unbind(1)
    c = math.sqrt(0.5)
    rotations = torch.stack([c * (w + qx), c * (qx - w), c * (qy + qz), c * (qz - qy)], dim=1)
    return dict(splats, means=torch.stack([x, z, -y], dim=1), rotations=rotations)


class model_factory:
    def __init__(self, model_filename, model_type, base_model_type, save_quantized=False):
        self.device = torch.device("cuda")
        self._interrupt = False
        with open(CONFIG_PATH, encoding="utf-8") as reader:
            self.config = json.load(reader)
        config = self.config
        with init_empty_weights():
            flow = TripoSplatFlow(**config["flow"])
        # FP16 checkpoint (INT8 linears compute in FP16); the fused q/k/v projections are split at load for the attention kit
        self.transformer = _load(flow, model_filename[0], fp32_dtype=torch.float16, fused_split_map=flow_split_map(config["flow"]["model_channels"]))
        if save_quantized:
            from wgp import save_quantized_model
            save_quantized_model(self.transformer, model_type, model_filename[0], torch.float16, None)
        with init_empty_weights():
            decoder = TripoSplatDecoder(config["octree"], config["gaussians"], config["octree_levels"])
            image_encoder = DINOv3ViT(**config["image_encoder"])
            vae = Flux2ImageEncoder(config["vae_bn_eps"])
        self.decoder = _load(decoder, fl.locate_file(DECODER_FILE), fused_split_map=decoder_split_map(config["gaussians"]["model_channels"]))
        self.image_encoder = _load(image_encoder, fl.locate_file(IMAGE_ENCODER_FILE))
        self.vae = _load(vae, fl.locate_file(VAE_FILE), preprocess_sd={"decoder": None})
        self.vae._convertWeightsFloatTo = torch.bfloat16  # FP32 checkpoint run in BF16, as upstream
        self.cache = TextEncoderCache()

    def pipe(self):
        return {"transformer": self.transformer, "image_encoder": self.image_encoder, "vae": self.vae, "decoder": self.decoder}

    def _encode_image(self, image):
        """DINOv3 features and FLUX.2 VAE posterior of the prepared object image, cached on the CPU by source image content."""
        config = self.config
        settings = (config["canvas_size"], config["crop_margin"], config["erode_radius"])
        digest = hashlib.sha1(image.tobytes() + repr((image.mode, image.size, settings)).encode()).hexdigest()

        def encode(_):
            prepared = prepare_image(image, *settings)
            check_abort()
            pixels = (torch.from_numpy(np.array(prepared)).permute(2, 0, 1)[None].float() / 255).to(self.device)
            mean = torch.tensor(IMAGE_MEAN, dtype=torch.float32, device=self.device).view(1, -1, 1, 1)
            std = torch.tensor(IMAGE_STD, dtype=torch.float32, device=self.device).view(1, -1, 1, 1)
            with text_encoding_progress(self.image_encoder.layer, title="Encoding Image", next_status="Encoding Image Latents"):
                features = self.image_encoder((pixels - mean) / std)
            posterior_mean, logvar, latent_mean, latent_std = self.vae(pixels)
            return [{"features": features, "mean": posterior_mean, "logvar": logvar, "latent_mean": latent_mean, "latent_std": latent_std}]

        return self.cache.encode(encode, ["image"], device=self.device, cache_keys=[digest])[0]

    def _vae_features(self, encoded, generator):
        """Upstream Flux2VAEEncoder.encode (deterministic=False): a posterior sample, patchified 2x2, normalized, (1, tokens, 128) FP32,
        after 5 zero tokens matching the DINOv3 class and register tokens."""
        mean, logvar = encoded["mean"], encoded["logvar"]
        latents = mean + torch.exp(0.5 * logvar) * torch.randn(mean.shape, dtype=mean.dtype, device=mean.device, generator=generator)
        batch, channels, height, width = latents.shape
        latents = latents.view(batch, channels, height // 2, 2, width // 2, 2).permute(0, 1, 3, 5, 2, 4).reshape(batch, channels * 4, height // 2, width // 2)
        latents = (latents - encoded["latent_mean"].view(1, -1, 1, 1)) / encoded["latent_std"].view(1, -1, 1, 1)
        latents = latents.to(torch.float32).flatten(2).transpose(1, 2)
        return torch.cat([latents.new_zeros(batch, 5, latents.shape[2]), latents], dim=1)

    def _sample(self, cond, uncond, generator, steps, guidance, shift, title, callback):
        """Upstream FlowEulerCfgSampler: Euler steps over the shifted schedule, CFG (guidance > 1) on the latent and camera velocities."""
        flow = self.transformer
        latent = torch.randn(1, flow.q_token_length, flow.in_channels, device=self.device, generator=generator)
        camera = torch.randn(1, 1, flow.cam_channels, device=self.device, generator=generator)
        schedule = np.linspace(1, 0, steps + 1)
        t_seq = shift * schedule / (1 + (shift - 1) * schedule)
        contexts = [cond] if guidance <= 1 else [cond, uncond]
        if callback is not None:
            callback(-1, None, True, override_num_inference_steps=steps, progress_title=title)
        for step, (t, t_prev) in enumerate(tqdm(list(zip(t_seq[:-1], t_seq[1:])), desc=title)):
            outputs = flow(latent, camera, torch.tensor([1000 * t], device=self.device, dtype=torch.float32), contexts)
            (v_latent, v_camera) = outputs[0]
            if len(outputs) > 1:
                v_latent = guidance * v_latent - (guidance - 1) * outputs[1][0]
                v_camera = guidance * v_camera - (guidance - 1) * outputs[1][1]
            del outputs
            latent, camera = latent - v_latent * (t - t_prev), camera - v_camera * (t - t_prev)
            del v_latent, v_camera
            if callback is not None:
                callback(step, None, False, override_num_inference_steps=steps, progress_title=title)
            check_abort()
        return latent

    @generation_progress
    @torch.inference_mode()
    def generate(self, seed, sampling_steps, guide_scale, batch_size, shift, custom_settings, original_input_ref_images, callback=None,
                 set_progress_status=None, **kwargs):
        num_gaussians = int((DEFAULT_CUSTOM_SETTINGS | custom_settings)["num_gaussians"])
        per_point = self.config["gaussians"]["num_gaussians"]
        encoded = self._encode_image(original_input_ref_images[0])
        flow = self.transformer
        uncond = flow.encode_context(torch.zeros_like(encoded["features"]), torch.zeros(1, encoded["features"].shape[1], flow.cond_embedder2.in_features,
                                     dtype=torch.float32, device=self.device)) if guide_scale > 1 else None
        previews, plys = [], []
        for sample_no in range(batch_size):
            suffix = f" {sample_no + 1}/{batch_size}" if batch_size > 1 else ""
            generator = torch.Generator(device=self.device).manual_seed(seed + sample_no)
            cond = flow.encode_context(encoded["features"], self._vae_features(encoded, generator))
            latent = self._sample(cond, uncond, generator, sampling_steps, guide_scale, shift, "Denoising Gaussians" + suffix, callback)
            del cond
            with PhaseProgress(self.config["octree_levels"] + len(self.decoder.gs.blocks), "Decoding Gaussians" + suffix, "blocks") as progress:
                splats = splats_to_gltf(self.decoder(latent, num_gaussians // per_point, generator, progress.advance))
            del latent
            set_phase_status("Rendering Preview")
            previews.append(splat_preview(splats, PREVIEW_SIZE).cpu())
            set_phase_status("Exporting Splats")
            plys.append(splats_to_ply(splats["means"], splats["scales"], splats["rotations"], splats["opacities"], splats["sh_dc"]))
            del splats
            check_abort()
        return {"x": torch.stack(previews, dim=1), "splats": plys}
