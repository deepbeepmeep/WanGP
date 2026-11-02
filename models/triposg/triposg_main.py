# TripoSG (image to mesh) and PartCrafter (image to part meshes) for WanGP, ported from VAST-AI-Research/TripoSG
# triposg/pipelines/pipeline_triposg.py, triposg/schedulers/scheduling_rectified_flow.py, scripts/inference_triposg.py and
# scripts/image_process.py (Copyright (c) 2025 VAST-AI-Research and contributors, MIT) and wgsxm/PartCrafter
# src/pipelines/pipeline_partcrafter.py and scripts/inference_partcrafter.py (Copyright (c) 2025 Yuchen Lin, MIT).
# Modified by DeepBeepMeep for WanGP: model loading through MMGP, WanGP generation contract (GLB bytes and previews), explicit devices,
# sequential guidance branches, the conditioning cached on the CPU, the shared mesh toolkit for surfaces, decimation, previews and GLB
# export, and WanGP's rembg u2net in place of BriaRMBG-1.4 (non-commercial) for photos without a usable alpha channel.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import hashlib
import json
import os

import cv2
import numpy as np
import torch
from accelerate import init_empty_weights
from mmgp import offload
from PIL import Image
from tqdm import tqdm

from models.hunyuan3d.dinov2 import Dinov2Model
from shared.mesh3d import Mesh, mesh_to_glb, meshes_to_glb
from shared.mesh3d.cleanup import decimate, merge_vertices
from shared.mesh3d.mesh import vertex_normals
from shared.mesh3d.render import preview
from shared.utils import files_locator as fl
from shared.utils.phase_progress import PhaseProgress, check_abort, generation_progress, set_phase_status, text_encoding_progress
from shared.utils.text_encoder_cache import TextEncoderCache
from .dit import TripoSGDiT
from .vae import TripoSGVAEDecoder, extract_mesh, flash_volume, hierarchical_volume

CONFIGS = os.path.join(os.path.dirname(__file__), "configs")
VAE_FILE, DINO_FILE = "triposg/triposg_vae_fp16.safetensors", "hunyuan3d/dinov2_large_fp16.safetensors"
PREVIEW_SIZE = 512
# PartCrafter's fixed part colours (src/utils/data_utils.py RGB) and one more, so that its 16 parts all differ
PART_COLORS = [(82, 170, 220), (215, 91, 78), (45, 136, 117), (247, 172, 83), (124, 121, 121), (127, 171, 209), (243, 152, 101), (145, 204, 192),
               (150, 59, 121), (181, 206, 78), (189, 119, 149), (199, 193, 222), (200, 151, 54), (236, 110, 102), (238, 182, 212), (96, 96, 168)]
DEFAULT_SETTINGS = {"num_parts": 4, "mesh_resolution": 512, "max_faces": 0}


def _load(module, filename):
    offload.load_model_data(module, filename, writable_tensors=False, default_dtype=torch.float16)
    return module.eval().requires_grad_(False)


def _image_digest(image, *parameters):
    digest = hashlib.sha1(image.tobytes())
    digest.update(repr((image.mode, image.size, parameters)).encode())
    return digest.hexdigest()


def _valid_alpha(alpha, min_ratio=0.01):
    """Upstream is_valid_alpha: at least 1% of the pixels in the lowest and in the highest of 20 alpha bins."""
    hist = cv2.calcHist([alpha], [0], None, [20], [0, 256])
    return hist[0] >= alpha.size * min_ratio and hist[-1] >= alpha.size * min_ratio


def _segmentation_alpha(rgb):
    """Upstream's photo path with rembg BiRefNet General in place of BriaRMBG-1.4: the soft mask thresholded (Otsu), components under 200 pixels
    removed, and the box of the largest remaining contour."""
    from rembg import remove
    from skimage.measure import label
    from skimage.morphology import remove_small_objects
    from preprocessing.rembg.assets import OBJECT_MODEL
    from shared.utils.utils import new_rembg_session

    session = new_rembg_session(OBJECT_MODEL)
    mask = np.array(remove(Image.fromarray(rgb), session=session, only_mask=True))
    del session
    _, mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cleaned = (remove_small_objects(label(mask), min_size=200) > 0).astype(np.uint8)
    if not cleaned.any():
        raise ValueError("No object was found in the Object Image: use a picture of one object, or an RGBA image with its alpha mask.")
    contours, _ = cv2.findContours(cleaned * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cleaned.astype(np.float32), cv2.boundingRect(max(contours, key=cv2.contourArea))


def object_image(image, padding_ratio=0.1, background=1.0):
    """Upstream load_image: images over 2000 pixels downscaled, the alpha channel (when it holds a cutout) or a segmentation mask
    composites the object over white, which is cropped to its box and padded to a square with a margin of padding_ratio."""
    has_alpha = image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info)
    array = np.array(image.convert("RGBA" if has_alpha else "RGB"))
    height, width = array.shape[:2]
    scale = 2000 / max(height, width)
    if scale < 1:
        array = cv2.resize(array, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    rgb = np.ascontiguousarray(array[..., :3])
    if has_alpha and _valid_alpha(np.ascontiguousarray(array[..., 3])):
        alpha = array[..., 3]
        rows, cols = np.where(alpha > 0)
        x, y, w, h = cols.min(), rows.min(), cols.max() - cols.min() + 1, rows.max() - rows.min() + 1
        alpha = torch.from_numpy(alpha)[None].float() / 255.0
    else:
        set_phase_status("Removing Background")
        alpha, (x, y, w, h) = _segmentation_alpha(rgb)
        alpha = torch.from_numpy(alpha)[None]
    pixels = torch.from_numpy(rgb).float().permute(2, 0, 1) / 255.0
    pixels = pixels * alpha + torch.full_like(pixels, background) * (1 - alpha)
    if w > h:
        pad_x = int(w * padding_ratio)
        pad_y = int(pad_x + (w - h) / 2)
    else:
        pad_y = int(h * padding_ratio)
        pad_x = int(pad_y + (h - w) / 2)
    pixels = torch.nn.functional.pad(pixels[:, y:y + h, x:x + w], (pad_x, pad_x, pad_y, pad_y), mode="constant", value=background)
    return Image.fromarray((pixels.permute(1, 2, 0).numpy() * 255).astype(np.uint8))


def flow_schedule(steps, device):
    """RectifiedFlowScheduler(shift=1).set_timesteps: sigmas 1 - i / steps (noise at 1) then 0, timesteps 1000 * sigma (FP32)."""
    sigmas = torch.from_numpy(np.array([1.0 - i / steps for i in range(steps)])).to(dtype=torch.float32, device=device)
    return sigmas * 1000, torch.cat([sigmas, torch.zeros(1, device=device)])


class model_factory:
    def __init__(self, model_filename, model_type, base_model_type, save_quantized=False):
        self.device = torch.device("cuda")
        self._interrupt = False
        self.base_model_type = base_model_type
        with open(os.path.join(CONFIGS, base_model_type + ".json"), encoding="utf-8") as reader:
            self.config = json.load(reader)
        filename = model_filename[0] if isinstance(model_filename, (list, tuple)) else model_filename
        with init_empty_weights():
            transformer = TripoSGDiT(**self.config["transformer"])
        self.transformer = _load(transformer, filename)
        if save_quantized:
            from wgp import save_quantized_model
            save_quantized_model(self.transformer, model_type, filename, torch.float16, None)
        with init_empty_weights():
            vae, image_encoder = TripoSGVAEDecoder(**self.config["vae"]), Dinov2Model(**self.config["image_encoder"])
        self.vae = _load(vae, fl.locate_file(VAE_FILE))
        self.image_encoder = _load(image_encoder, fl.locate_file(DINO_FILE))
        self.cache = TextEncoderCache()
        self._prepared = (None, None)  # (source image digest, object image): background removal of the last reference

    # ---------------------------------------------------------------- conditioning

    def _object_image(self, image):
        digest = _image_digest(image, self.config["padding_ratio"])
        if self._prepared[0] != digest:
            self._prepared = (digest, object_image(image, self.config["padding_ratio"]))
            check_abort()
        return self._prepared[1]

    def _condition(self, image):
        """DINOv2-L last hidden state (1, 257, 1024) of the object image (BitImageProcessor: shortest edge 256 bicubic, center crop 224),
        cached on the CPU."""
        from transformers import BitImageProcessor

        def encode(_):
            with text_encoding_progress(list(self.image_encoder.encoder.layer), next_status="Denoising Parts" if self.base_model_type == "partcrafter" else "Denoising Shape", title="Encoding Image"):
                pixels = BitImageProcessor(**self.config["image_processor"])(images=image, return_tensors="pt").pixel_values
                return [self.image_encoder(pixels.to(self.device, torch.float16))]

        return self.cache.encode(encode, ["image"], device=self.device, cache_keys=[("dinov2_large", _image_digest(image))])[0]

    # ---------------------------------------------------------------- shape

    def _denoise(self, latents, cond, steps, guidance_scale, parts, title, callback):
        """Upstream denoising loop: Euler steps of the rectified flow on the velocity predicted with classifier-free guidance against the
        zero-image embedding (guidance <= 1: conditional prediction only)."""
        timesteps, sigmas = flow_schedule(steps, self.device)
        guided = guidance_scale > 1
        contexts = [cond, None] if guided else [cond]
        part_ids = None
        if parts:  # upstream numbers the parts of its batch [uncond parts, cond parts]: the cond parts get the second half of the ids
            ids = torch.arange(2 * parts, device=self.device)
            part_ids = [ids[parts:], ids[:parts]] if guided else [ids[:parts]]
        if callback is not None:
            callback(-1, None, True, override_num_inference_steps=steps, progress_title=title)
        for i in tqdm(range(steps), desc=title):
            predictions = self.transformer(latents, timesteps[i].expand(latents.shape[0]), contexts, part_ids)
            if guided:
                noise_pred_image, noise_pred_uncond = predictions
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_image - noise_pred_uncond)
            else:
                noise_pred = predictions[0]
            del predictions
            latents = (latents.to(torch.float32) + (sigmas[i] - sigmas[i + 1]) * noise_pred).to(noise_pred.dtype)
            del noise_pred
            if callback is not None:
                callback(i, None, False, override_num_inference_steps=steps, progress_title=title)
            check_abort()
        return latents

    def _meshes(self, latents, resolution):
        """Surface of each latent set (meshes, tokens, channels), None for a volume without surface."""
        depth, bounds, meshes = int(resolution).bit_length() - 1, self.config["bounds"], []
        with PhaseProgress(len(latents), "Decoding Volume", "meshes") as progress:
            for latent in latents:
                keys_values = self.vae(latent[None])
                if self.config["volume"] == "flash":
                    grid, grid_bounds = flash_volume(self.vae, keys_values, octree_depth=depth, bounds=bounds)
                else:
                    grid, grid_bounds = hierarchical_volume(self.vae, keys_values, hierarchical_octree_depth=depth, bounds=bounds)
                del keys_values
                meshes.append(extract_mesh(grid, grid_bounds))
                del grid
                progress.advance()
        return meshes

    # ---------------------------------------------------------------- generation

    @generation_progress
    @torch.inference_mode()
    def generate(self, seed, sampling_steps, guide_scale, batch_size, custom_settings, original_input_ref_images, callback=None, set_progress_status=None,
                 **kwargs):
        settings = DEFAULT_SETTINGS | custom_settings
        resolution, max_faces = int(settings["mesh_resolution"]), int(settings["max_faces"])
        parts = int(settings["num_parts"]) if self.base_model_type == "partcrafter" else 0
        image = self._object_image(original_input_ref_images[0])
        cond = self._condition(image)
        previews, glbs = [], []
        for sample_no in range(batch_size):
            suffix = f" {sample_no + 1}/{batch_size}" if batch_size > 1 else ""
            generator = torch.Generator(device=self.device).manual_seed(seed + sample_no)  # upstream draws the noise on the GPU
            latents = torch.randn((max(parts, 1), self.config["num_tokens"], self.config["transformer"]["in_channels"]), generator=generator,
                                  device=self.device, dtype=torch.float16)
            title = ("Denoising Parts" if parts else "Denoising Shape") + suffix
            latents = self._denoise(latents, cond, sampling_steps, guide_scale, parts, title, callback)
            meshes = self._meshes(latents, resolution)
            del latents
            empty = [i for i, mesh in enumerate(meshes) if mesh is None]
            meshes = [mesh for mesh in meshes if mesh is not None]
            if not meshes:
                raise ValueError(f"{'PartCrafter' if parts else 'TripoSG'} produced an empty volume: try another seed or a clearer Object Image.")
            if empty:
                print(f"[PartCrafter] parts {empty} have no surface and are omitted")
            if max_faces > 0:
                set_phase_status("Reducing Mesh")
                total = sum(len(mesh.faces) for mesh in meshes)
                meshes = [decimate(merge_vertices(mesh), max(4, max_faces * len(mesh.faces) // total)) for mesh in meshes]
            set_phase_status("Exporting Mesh")
            meshes = [mesh.replace(normals=vertex_normals(mesh.vertices, mesh.faces)) for mesh in meshes]  # GLB normals, seconds on the CPU
            if parts:
                colors = [torch.tensor(PART_COLORS[i % len(PART_COLORS)], dtype=torch.float32, device=self.device) / 255 for i in range(len(meshes))]
                meshes = [mesh.replace(colors=color.repeat(len(mesh.vertices), 1)) for mesh, color in zip(meshes, colors)]
                glbs.append(meshes_to_glb([mesh.to("cpu") for mesh in meshes]))
                offsets = np.cumsum([0] + [len(mesh.vertices) for mesh in meshes[:-1]])
                merged = Mesh(torch.cat([mesh.vertices for mesh in meshes]), torch.cat([mesh.faces + int(offset) for mesh, offset in zip(meshes, offsets)]),
                              colors=torch.cat([mesh.colors for mesh in meshes]))
                previews.append(preview(merged, PREVIEW_SIZE).cpu())
            else:
                previews.append(preview(meshes[0], PREVIEW_SIZE).cpu())
                glbs.append(mesh_to_glb(meshes[0].to("cpu")))
            del meshes
            check_abort()
        return {"x": torch.stack(previews, dim=1), "meshes": glbs}
