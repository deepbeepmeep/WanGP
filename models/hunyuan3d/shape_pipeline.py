# Hunyuan3D 2.1 image-to-shape pipeline, ported from Tencent Hunyuan3D 2.1 hy3dshape/{pipelines.py, preprocessors.py, rembg.py,
# schedulers.py, models/conditioner.py, models/autoencoders/surface_extractors.py} and gradio_app.py (Copyright (C) 2025 Tencent;
# Tencent Hunyuan 3D 2.1 Community License, see models/hunyuan3d/LICENSE and Notice.txt).
# Modified by DeepBeepMeep for WanGP: one function per stage, explicit devices, sequential guidance branches, WanGP progress and
# abort checks, the conditioning cached on the CPU, and faces with undecoded (NaN) vertices removed from the extracted surface instead
# of being collapsed onto vertex 0 by trimesh.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import hashlib

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from shared.utils.phase_progress import check_abort

DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)


def remove_background(image):
    """Object image with its alpha: an RGBA input with transparency is kept, other images go through rembg with upstream's settings
    (BackgroundRemover: no alpha matting, transparent white background), with BiRefNet General instead of u2net (no cast shadows)."""
    if image.mode == "RGBA" and image.getextrema()[3][0] < 255:
        return image
    from rembg import remove
    from preprocessing.rembg.assets import OBJECT_MODEL
    from shared.utils.utils import new_rembg_session

    session = new_rembg_session(OBJECT_MODEL)
    rgba = remove(image.convert("RGB"), session=session, bgcolor=[255, 255, 255, 0])
    del session
    return rgba


def image_key(image, *parameters):
    """Cache key of an image and the parameters of its preprocessing."""
    digest = hashlib.sha1(image.tobytes())
    digest.update(repr((image.mode, image.size, parameters)).encode())
    return digest.hexdigest()


def recenter(image, border_ratio):
    """Upstream ImageProcessorV2.recenter: the object cropped from its alpha, scaled to (1 - border_ratio) of a square canvas, centered,
    composited on white. image: uint8 [H, W, 4]. Returns RGB and mask uint8."""
    mask = image[..., 3]
    size = max(image.shape[:2])
    result = np.zeros((size, size, 4), dtype=np.uint8)
    coords = np.nonzero(mask)
    x_min, x_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    h, w = x_max - x_min, y_max - y_min
    if h == 0 or w == 0:
        raise ValueError("The object image is empty after background removal")
    scale = int(size * (1 - border_ratio)) / max(h, w)
    h2, w2 = int(h * scale), int(w * scale)
    x2_min, y2_min = (size - h2) // 2, (size - w2) // 2
    result[x2_min:x2_min + h2, y2_min:y2_min + w2] = cv2.resize(image[x_min:x_max, y_min:y_max], (w2, h2), interpolation=cv2.INTER_AREA)
    alpha = result[..., 3:].astype(np.float32) / 255
    rgb = result[..., :3] * alpha + np.full((size, size, 3), 255, dtype=np.uint8) * (1 - alpha)
    return rgb.clip(0, 255).astype(np.uint8), (alpha * 255).clip(0, 255).astype(np.uint8)


def shape_image_tensor(rgba, size=512, border_ratio=0.15):
    """Upstream ImageProcessorV2 on an RGBA image: recentered, resized (bicubic), as a float tensor in [-1, 1] (1, 3, size, size)."""
    rgb, _ = recenter(np.asarray(rgba.convert("RGBA")), border_ratio)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_CUBIC)
    return (torch.tensor(rgb, device="cpu").float() / 255 * 2 - 1).permute(2, 0, 1)[None]


def dino_input(image, device, dtype, size=518):
    """Upstream DinoImageEncoder preprocessing of a [-1, 1] image: [0, 1], torchvision Resize (antialiased bilinear, computed in FP32)
    and Normalize in the encoder dtype."""
    image = ((image - (-1)) / (1 - (-1))).to(device, dtype=dtype)
    image = F.interpolate(image.float(), size=(size, size), mode="bilinear", align_corners=False, antialias=True).to(dtype)
    mean = torch.as_tensor(DINO_MEAN, dtype=dtype, device=device).view(-1, 1, 1)
    std = torch.as_tensor(DINO_STD, dtype=dtype, device=device).view(-1, 1, 1)
    return image.sub_(mean).div_(std)


def flow_sigmas(steps, device):
    """Upstream FlowMatchEulerDiscreteScheduler with sigmas linspace(0, 1, steps) (noise at 0, data at 1): float32 sigmas followed by
    a final 1."""
    sigmas = torch.from_numpy(np.linspace(0, 1, steps)).to(dtype=torch.float32, device=device)
    return torch.cat([sigmas, torch.ones(1, device=device)])


def sample_latents(transformer, cond, latents, steps, guidance_scale, callback=None, title="Denoising Shape"):
    """Upstream Hunyuan3DDiTFlowMatchingPipeline denoising loop: Euler steps on the velocity predicted with classifier-free guidance
    against the zero-image embedding (guidance 1: conditional prediction only)."""
    sigmas = flow_sigmas(steps, latents.device)
    timesteps = sigmas[:-1] * 1000
    contexts = [cond] if guidance_scale == 1 else [cond, None]
    if callback is not None:
        callback(-1, None, True, override_num_inference_steps=steps, progress_title=title)
    for i in tqdm(range(steps), desc=title):
        timestep = timesteps[i].expand(latents.shape[0]).to(latents.dtype) / 1000
        predictions = transformer(latents, timestep, contexts)
        if guidance_scale == 1:
            noise_pred = predictions[0]
        else:
            noise_pred_cond, noise_pred_uncond = predictions
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        del predictions
        latents = (latents.to(torch.float32) + (sigmas[i + 1] - sigmas[i]) * noise_pred).to(noise_pred.dtype)
        del noise_pred
        if callback is not None:
            callback(i, None, False, override_num_inference_steps=steps, progress_title=title)
        check_abort()
    return latents


def extract_surface(grid_logits, octree_resolution, bounds=1.01, mc_level=0.0):
    """Upstream MCSurfaceExtractor (scikit-image Lewiner marching cubes) followed by the face flip of export_to_trimesh. The grid holds
    NaN where FlashVDM did not decode: the faces touching the NaN vertices this creates are dropped (trimesh's processing collapsed them
    onto vertex 0) with the vertices left unused. Returns float32 vertices and int64 faces (numpy)."""
    from skimage import measure

    vertices, faces, _, _ = measure.marching_cubes(grid_logits.float().cpu().numpy(), mc_level, method="lewiner")
    bbox_min, bbox_size, grid_size = np.array([-bounds] * 3), np.array([2 * bounds] * 3), np.array([int(octree_resolution) + 1] * 3)
    vertices = (vertices / grid_size * bbox_size + bbox_min).astype(np.float32)
    faces = np.ascontiguousarray(faces[:, ::-1]).astype(np.int64)
    finite = np.isfinite(vertices).all(axis=1)
    faces = faces[finite[faces].all(axis=1)]
    used = np.zeros(len(vertices), dtype=bool)
    used[faces] = True
    remap = np.cumsum(used) - 1
    return vertices[used], remap[faces]
