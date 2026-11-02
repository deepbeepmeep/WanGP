# Hunyuan3D 2.1 for WanGP: image to shape (hy3dshape) and PBR texturing (hy3dpaint), from Tencent Hunyuan3D 2.1 gradio_app.py and
# textureGenPipeline.py (Copyright (C) 2025 Tencent; Tencent Hunyuan 3D 2.1 Community License, see models/hunyuan3d/LICENSE and
# Notice.txt). Modified by DeepBeepMeep for WanGP: model loading through MMGP, WanGP generation contract (GLB bytes and previews), shared
# mesh toolkit for decimation, UV unwrapping, previews and GLB export.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import json
import os

import numpy as np
import torch
from accelerate import init_empty_weights
from mmgp import offload
from PIL import Image
from torch import nn

from shared.mesh3d import Mesh, load_mesh, mesh_to_glb
from shared.mesh3d.cleanup import decimate, merge_vertices
from shared.mesh3d.render import preview
from shared.utils import files_locator as fl
from shared.utils.phase_progress import PhaseProgress, check_abort, generation_progress, set_phase_status, text_encoding_progress
from shared.utils.text_encoder_cache import TextEncoderCache
from .dinov2 import Dinov2Model
from .paint_unet import PaintUNet, ReferenceUNet
from .shape_dit import HunYuanDiTPlain
from .shape_pipeline import dino_input, extract_surface, image_key, remove_background, sample_latents, shape_image_tensor
from .shape_vae import ShapeVAEDecoder

CONFIGS = os.path.join(os.path.dirname(__file__), "configs")
PROJECT = "hunyuan3d"
SHAPE_VAE_FILE, DINO_LARGE_FILE = "hunyuan3d_21_shape_vae_fp16.safetensors", "dinov2_large_fp16.safetensors"
PAINT_VAE_FILE, DINO_GIANT_FILE, ESRGAN_FILE = "hunyuan3d_21_paint_vae_fp16.safetensors", "dinov2_giant_fp16.safetensors", "realesrgan_x4plus_fp16.safetensors"
PREVIEW_SIZE = 512


class PaintModels(nn.Module):
    """unet and unet_dual of upstream UNet2p5DConditionModel, the layout of the paint checkpoint."""

    def __init__(self):
        super().__init__()
        self.unet = PaintUNet()
        self.unet_dual = ReferenceUNet()


def _load(module, filename):
    offload.load_model_data(module, filename, writable_tensors=False, default_dtype=torch.float16)
    return module.eval().requires_grad_(False)


class model_factory:
    """hunyuan3d_21_shape: a mesh of the object image; hunyuan3d_21_paint: PBR textures of a supplied mesh (also the Hunyuan3D 2.1 Paint
    Texturer)."""

    def __init__(self, model_filename, model_type, base_model_type, save_quantized=False):
        self.device = torch.device("cuda")
        self._interrupt = False
        filename = model_filename[0] if isinstance(model_filename, (list, tuple)) else model_filename
        self.transformer = self.vae = self.image_encoder = None
        self.paint = self.paint_vae = self.paint_image_encoder = self.upscaler = None
        if base_model_type == "hunyuan3d_21_paint":
            self._load_paint(filename, model_type, save_quantized)
            self.transformer = self.paint.unet  # wgp's main model of the paint type
        else:
            self._load_shape(filename, model_type, save_quantized)
        self.cache = TextEncoderCache()

    def _load_shape(self, filename, model_type, save_quantized):
        with open(os.path.join(CONFIGS, "hunyuan3d_21_shape.json"), encoding="utf-8") as reader:
            self.config = json.load(reader)
        with init_empty_weights():
            transformer = HunYuanDiTPlain(**self.config["transformer"])
        self.transformer = _load(transformer, filename)
        if save_quantized:
            from wgp import save_quantized_model
            save_quantized_model(self.transformer, model_type, filename, torch.float16, None)
        with init_empty_weights():
            vae, image_encoder = ShapeVAEDecoder(**self.config["vae"]), Dinov2Model(**self.config["image_encoder"])
        self.vae = _load(vae, fl.locate_file(os.path.join(PROJECT, SHAPE_VAE_FILE)))
        self.image_encoder = _load(image_encoder, fl.locate_file(os.path.join(PROJECT, DINO_LARGE_FILE)))

    def _load_paint(self, filename, model_type, save_quantized):
        from models.z_image.autoencoder_kl import AutoencoderKL
        from .rrdbnet import RRDBNet

        with open(os.path.join(CONFIGS, "hunyuan3d_21.json"), encoding="utf-8") as reader:
            self.paint_config = json.load(reader)
        with init_empty_weights():
            paint = PaintModels()
        self.paint = _load(paint, filename)
        if save_quantized:
            from wgp import save_quantized_model
            save_quantized_model(self.paint, model_type, filename, torch.float16, None)
        with init_empty_weights():
            vae = AutoencoderKL(**self.paint_config["vae"])
            image_encoder, upscaler = Dinov2Model(**self.paint_config["image_encoder"]), RRDBNet()
        self.paint_vae = _load(vae, fl.locate_file(os.path.join(PROJECT, PAINT_VAE_FILE)))
        self.paint_image_encoder = _load(image_encoder, fl.locate_file(os.path.join(PROJECT, DINO_GIANT_FILE)))
        self.upscaler = _load(upscaler, fl.locate_file(os.path.join(PROJECT, ESRGAN_FILE)))

    # ---------------------------------------------------------------- conditioning (cached on the CPU)

    def _object_image(self, image):
        """Background removed (rembg u2net), cached by the input image."""
        key = ("rgba", image_key(image))
        rgba = self.cache.encode(lambda _: [torch.from_numpy(np.array(remove_background(image).convert("RGBA")))], ["rgba"], cache_keys=[key])[0]
        return Image.fromarray(rgba.numpy())

    def _shape_condition(self, rgba):
        """DINOv2-L features of the recentered object image (upstream ImageProcessorV2 then DinoImageEncoder)."""
        size, border = self.config["image_size"], self.config["border_ratio"]

        def encode(_):
            with text_encoding_progress(list(self.image_encoder.encoder.layer), next_status="Denoising Shape", title="Encoding Image"):
                pixels = dino_input(shape_image_tensor(rgba, size, border), self.device, torch.float16)
                return [self.image_encoder(pixels)]

        return self.cache.encode(encode, ["shape"], device=self.device, cache_keys=[("dino_large", image_key(rgba, size, border))])[0]

    def _paint_condition(self, reference):
        """DINOv2-giant features of the paint reference image (upstream Dino_v2: BitImageProcessor at 224 px)."""
        from transformers import BitImageProcessor

        def encode(_):
            with text_encoding_progress(list(self.paint_image_encoder.encoder.layer), next_status="Generating Texture Views", title="Encoding Image"):
                pixels = BitImageProcessor(**self.paint_config["image_processor"])(images=reference, return_tensors="pt").pixel_values
                return [self.paint_image_encoder(pixels.to(self.device).half())]

        return self.cache.encode(encode, ["paint"], device=self.device, cache_keys=[("dino_giant", image_key(reference))])[0]

    # ---------------------------------------------------------------- shape

    def _generate_shapes(self, rgba, seed, steps, guidance_scale, batch_size, octree_resolution, callback):
        cond = self._shape_condition(rgba)
        generator = torch.Generator(device="cpu").manual_seed(seed)  # upstream draws the noise on the CPU (randn_tensor)
        latents = torch.randn((batch_size, *self.vae.latent_shape), generator=generator, device="cpu", dtype=torch.float16).to(self.device)
        latents = sample_latents(self.transformer, cond.expand(batch_size, -1, -1), latents, steps, guidance_scale, callback)
        set_phase_status("Decoding Shape")
        decoded = self.vae((1.0 / self.vae.scale_factor) * latents)
        del latents, cond
        meshes = []
        for sample in decoded.split(1):
            grid = self.vae.decode_volume(sample, bounds=self.config["bounds"], num_chunks=self.config["num_chunks"], mc_level=self.config["mc_level"],
                                          octree_resolution=octree_resolution)
            set_phase_status("Extracting Surface")
            vertices, faces = extract_surface(grid, octree_resolution, self.config["bounds"], self.config["mc_level"])
            del grid
            meshes.append(Mesh(torch.from_numpy(vertices).to(self.device), torch.from_numpy(faces).to(self.device)))
            check_abort()
        return meshes

    # ---------------------------------------------------------------- texture

    def _paint_reference(self, rgba, view_size):
        """Upstream textureGenPipeline + forward_one: the object image squashed to 512 x 512, composited on white, resized to the view
        size (PIL bicubic)."""
        size = self.paint_config["reference_size"]
        image = rgba.resize((size, size))
        white = Image.new("RGB", image.size, (255, 255, 255))
        white.paste(image, mask=image.getchannel("A"))
        return white.convert("RGB").resize((view_size, view_size))

    def _texture_mesh(self, mesh, rgba, seed, guidance_scale, max_faces, views, view_size, texture_size, callback):
        from .paint_pipeline import RENDER_SIZE, BackgroundUnwrap, MeshRenderer, bake_views, inpaint, multiview_diffusion, select_views

        set_phase_status("Preparing Mesh")
        mesh = merge_vertices(Mesh(mesh.vertices.float(), mesh.faces))
        mesh = decimate(mesh, max_faces or self.paint_config["default_faces"])
        bake_size = min(2 * texture_size, 4096)
        unwrap = BackgroundUnwrap(mesh, max_faces=40000)  # bounded CPU regions, overlapped with view generation
        try:
            set_phase_status("Rendering Views")
            renderer = MeshRenderer(mesh.vertices, mesh.faces, None, bake_size, self.device)
            cameras = select_views(renderer, views)
            resize = lambda array: Image.fromarray(array).resize((view_size, view_size))
            normal_maps = [resize(renderer.render_normal(elev, azim)) for elev, azim, _ in cameras]
            position_maps = [resize(renderer.render_position(elev, azim)) for elev, azim, _ in cameras]
            del renderer
            check_abort()
            reference = self._paint_reference(rgba, view_size)
            dino = self._paint_condition(reference)
            albedo, mr = multiview_diffusion(self.paint.unet, self.paint.unet_dual, self.paint_vae, dino, reference, normal_maps, position_maps, view_size,
                                             self.paint_config["texture_steps"], guidance_scale, seed, callback, self.device)
            del dino, normal_maps, position_maps
            with PhaseProgress(len(albedo) + len(mr), "Upscaling Views", "views") as progress:
                upscaled = []
                for image in albedo + mr:
                    upscaled.append(Image.fromarray(self.upscaler.enhance(image, self.device)))
                    progress.advance()
            albedo, mr = upscaled[:len(albedo)], upscaled[len(albedo):]
            for images in (albedo, mr):  # upstream resizes only the first two views of each material to the render size, as here
                for i in range(2):
                    images[i] = images[i].resize((RENDER_SIZE, RENDER_SIZE))
            set_phase_status("Unwrapping UVs")
            mesh = unwrap.result(check_abort=check_abort, progress=True)
        finally:
            unwrap.close()
        uvs = torch.stack([mesh.uvs[:, 0], 1 - mesh.uvs[:, 1]], dim=1)  # upstream reads xatlas' uvs with v up (OBJ)
        renderer = MeshRenderer(mesh.vertices, mesh.faces, uvs, bake_size, self.device)
        (albedo_texture, mr_texture), mask = bake_views(renderer, albedo, mr, cameras)
        del albedo, mr
        set_phase_status("Inpainting Texture")
        albedo_texture, mr_texture = inpaint(renderer, (albedo_texture, mr_texture), mask, ((0, 1, 2), (0, 1)))  # metallic, roughness
        del renderer
        # upstream set_texture_mr (without force_set) stores the metallic-roughness texture through one more truncating uint8 round trip
        mr_texture = ((mr_texture / 255).astype(np.float32) * 255).astype(np.uint8)
        base_color, metallic, roughness = (_texture_map(t, bake_size, texture_size) for t in (albedo_texture, mr_texture[..., 0], mr_texture[..., 1]))
        metallic_roughness = np.stack([np.full_like(metallic, 255), roughness, metallic], axis=-1)
        return mesh.replace(uvs=uvs, base_color=torch.from_numpy(base_color).to(mesh.device), metallic_roughness=torch.from_numpy(metallic_roughness).to(mesh.device))

    @generation_progress
    @torch.inference_mode()
    def texture_mesh(self, mesh, image, seed=0, guidance_scale=3.0, max_faces=0, texture_views="8_768", texture_size=2048, callback=None, set_progress_status=None):
        """The Texturer entry (post-processing): PBR textures of mesh (shared.mesh3d.Mesh) painted from the object image (PIL); max_faces 0
        keeps the mesh's faces."""
        views, view_size = (int(value) for value in str(texture_views).split("_"))
        mesh = Mesh(mesh.vertices.float().to(self.device), mesh.faces.to(self.device))
        return self._texture_mesh(mesh, self._object_image(image), int(seed), float(guidance_scale), int(max_faces) or mesh.faces.shape[0], views, view_size, int(texture_size), callback)

    # ---------------------------------------------------------------- generation

    @generation_progress
    @torch.inference_mode()
    def generate(self, input_prompt, seed, sampling_steps, guide_scale, batch_size=1, original_input_ref_images=None, input_custom=None, custom_settings=None,
                 callback=None, set_progress_status=None, **kwargs):
        """Shape model: meshes of the object image. Paint model: PBR textures of the Input Mesh (input_custom), guide_scale being the
        texture guidance."""
        settings = custom_settings or {}
        set_phase_status("Removing Background")
        rgba = self._object_image(original_input_ref_images[0])
        check_abort()
        if self.paint is None:
            octree_resolution, max_faces = int(settings.get("octree_resolution", 256)), int(settings.get("max_faces", 0))
            meshes = self._generate_shapes(rgba, seed, sampling_steps, guide_scale, batch_size, octree_resolution, callback)
            if max_faces > 0:
                set_phase_status("Reducing Mesh")
                meshes = [decimate(merge_vertices(mesh), max_faces) for mesh in meshes]
        else:
            set_phase_status("Loading Mesh")
            source = load_mesh(input_custom, device=self.device)
            views, view_size = (int(v) for v in str(settings.get("texture_views", "8_768")).split("_"))
            max_faces, texture_size = int(settings.get("max_faces", 40000)), int(settings.get("texture_size", 2048))
            meshes = [self._texture_mesh(Mesh(source.vertices.float(), source.faces), rgba, seed + i, guide_scale, max_faces, views, view_size, texture_size, callback)
                      for i in range(batch_size)]
        set_phase_status("Exporting Mesh")
        previews, glbs = [], []
        for mesh in meshes:
            previews.append(preview(mesh, PREVIEW_SIZE).cpu())
            glbs.append(mesh_to_glb(mesh.to("cpu")))
            check_abort()
        return {"x": torch.stack(previews, dim=1), "meshes": glbs}


def _texture_map(texture, bake_size, texture_size):
    """Upstream save_mesh: the inpainted uint8 texture as float, halved with cv2.resize (bilinear) when baked at twice the output size,
    back to uint8 by truncation."""
    import cv2

    if bake_size == texture_size:
        return texture
    image = (texture / 255).astype(np.float32)
    image = cv2.resize(image, (image.shape[1] // 2, image.shape[0] // 2))
    return (image * 255).astype(np.uint8)
