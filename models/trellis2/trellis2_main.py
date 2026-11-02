# TRELLIS.2 pipelines, inference port of microsoft/TRELLIS.2 (MIT License, see LICENSE in this folder). Shape (trellis2_shape,
# trellis2/pipelines/trellis2_image_to_3d.py): sparse structure flow, shape flow (512, 1024 or a 512 -> 1024/1536 cascade), shape decoder,
# mesh extraction. Texture (trellis2_texture, trellis2/pipelines/trellis2_texturing.py): a supplied mesh encoded as shape latents (the
# flexible dual grid and shape encoder of WanGP's SUPRA-Tex port), texture flow, texture decoder, PBR baking in a UV atlas.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import hashlib
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights
from mmgp import offload
from PIL import Image
from tqdm import tqdm

from shared.utils import files_locator as fl
from shared.utils.phase_progress import PhaseProgress, check_abort, generation_progress, set_phase_status, text_encoding_progress
from shared.utils.text_encoder_cache import TextEncoderCache
from . import sampler
from .dinov3 import DINOv3ViT, image_to_pixels
from .dit import TrellisFlowDiT, fused_split_map
from .export import bake_textures, export_glb, finish_surface, remesh_surface, render_preview, to_gltf_mesh
from .fdg_mesh import decode_fdg, flexible_dual_grid_to_mesh
from .settings import DEFAULT_CUSTOM_SETTINGS, DEFAULT_TEXTURE_SETTINGS
from .sparse import SparseTensor
from .vae import SparseStructureDecoder, SparseUnetDecoder

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "trellis2.json")
# checkpoint prefixes without digits, so that MMGP detects the 30 blocks of each flow (quantization, offloading)
SHAPE_STAGES = ("ss_flow", "shape_lowres", "shape_highres")
TEXTURE_STAGES = ("tex_lowres", "tex_highres")
SS_DECODER_FILE = "trellis2/trellis2_ss_decoder_fp16.safetensors"
SHAPE_DECODER_FILE = "trellis2/trellis2_shape_decoder_fp16.safetensors"
TEXTURE_DECODER_FILE = "trellis2/trellis2_texture_decoder_fp16.safetensors"
SHAPE_ENCODER_FILE = "trellis2/trellis2_shape_encoder_fp16.safetensors"  # shared with SUPRA-Tex
IMAGE_ENCODER_FILE = "dinov3_vitl16/dinov3_vitl16_lvd1689m.safetensors"  # Meta's weights, shared folder, DINOv3 License beside them
PIPELINES = {0: "512", 1: "1024", 2: "1024_cascade", 3: "1536_cascade"}
PREVIEW_SIZE = 512
SHAPE_ARCH, TEXTURE_ARCH = "trellis2_shape", "trellis2_texture"
UNWRAP_REGION_FACES = 40000


def _keep_checkpoint_dtypes(model):
    """MMGP keeps the checkpoint dtypes (no FP32 conversion) and does not cast the FP32 tensors given to the model and to its top-level
    layers to the model dtype: upstream feeds FP32 latents, timesteps and features to the layers it computes in FP32."""
    model._convertWeightsFloatTo = None
    for module in [model, *model.children()]:
        if not isinstance(module, nn.ModuleList) and not hasattr(module, "_lock_dtype"):
            module._lock_dtype = None
    return model.eval().requires_grad_(False)


def _load_module(module, filename):
    """Checkpoint-native dtypes (default_dtype None), read-only mmap."""
    offload.load_model_data(module, filename, writable_tensors=False, default_dtype=None)
    return _keep_checkpoint_dtypes(module)


def preprocess_image(image, remove_background):
    """Upstream Trellis2ImageTo3DPipeline.preprocess_image: images larger than 1024 are downscaled, the alpha channel (or a background
    removal mask) crops a square around the object, and the object is premultiplied over black."""
    has_alpha = image.mode == "RGBA" and not np.all(np.array(image)[:, :, 3] == 255)
    scale = min(1, 1024 / max(image.size))
    if scale < 1:
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    output = image if has_alpha else remove_background(image.convert("RGB"))
    alpha = np.array(output)[:, :, 3]
    points = np.argwhere(alpha > 0.8 * 255)
    if len(points) == 0:
        raise ValueError("No object was found in the reference image: use an image of one object, or an RGBA image with its alpha mask.")
    bbox = np.min(points[:, 1]), np.min(points[:, 0]), np.max(points[:, 1]), np.max(points[:, 0])
    center = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    size = int(max(bbox[2] - bbox[0], bbox[3] - bbox[1]))
    output = np.array(output.crop((center[0] - size // 2, center[1] - size // 2, center[0] + size // 2, center[1] + size // 2))).astype(np.float32) / 255
    return Image.fromarray(((output[:, :, :3] * output[:, :, 3:4]) * 255).astype(np.uint8))


def _rembg(image):
    set_phase_status("Removing Background")
    from rembg import remove
    from preprocessing.rembg.assets import OBJECT_MODEL
    from shared.utils.utils import new_rembg_session

    return remove(image, session=new_rembg_session(OBJECT_MODEL))


class model_factory:
    """trellis2_shape: untextured meshes of the object image; trellis2_texture: PBR textures of a supplied mesh (also the TRELLIS.2 Texture
    Texturer)."""

    def __init__(self, model_filename, model_type, base_model_type, save_quantized=False):
        self.device = torch.device("cuda")
        with open(CONFIG_PATH, encoding="utf-8") as reader:
            self.config = json.load(reader)
        self.textured = base_model_type == TEXTURE_ARCH
        stages = TEXTURE_STAGES if self.textured else SHAPE_STAGES
        dit = self.config["dit"]
        with init_empty_weights():
            container = nn.ModuleDict({name: TrellisFlowDiT(**dit, **self.config["stages"][name]) for name in stages})
        # BF16 checkpoint (INT8 linears compute in BF16); the fused q/k/v and k/v projections are split at load for the attention kit
        offload.load_model_data(container, model_filename[0], writable_tensors=False, default_dtype=None, fp32_dtype=torch.bfloat16, fused_split_map=fused_split_map(dit["model_channels"]))
        container.eval().requires_grad_(False)
        if save_quantized:
            from wgp import save_quantized_model
            save_quantized_model(container, model_type, model_filename[0], torch.bfloat16, None)
        self.flows = {name: _keep_checkpoint_dtypes(flow) for name, flow in container.items()}
        self.transformer = self.flows[stages[0]]
        with init_empty_weights():
            image_encoder = DINOv3ViT(**self.config["image_encoder"])
            if self.textured:
                from models.supratex.encoder import SparseUnetEncoder
                shape_encoder, texture_decoder = SparseUnetEncoder(**self.config["shape_encoder"]), SparseUnetDecoder(**self.config["texture_decoder"])
            else:
                ss_decoder, shape_decoder = SparseStructureDecoder(**self.config["ss_decoder"]), SparseUnetDecoder(**self.config["shape_decoder"])
        if self.textured:
            self.shape_encoder = _load_module(shape_encoder, fl.locate_file(SHAPE_ENCODER_FILE))
            self.texture_decoder = _load_module(texture_decoder, fl.locate_file(TEXTURE_DECODER_FILE))
        else:
            ss_decoder.lock_checkpoint_fp32_modules()
            self.ss_decoder = _load_module(ss_decoder, fl.locate_file(SS_DECODER_FILE))
            self.shape_decoder = _load_module(shape_decoder, fl.locate_file(SHAPE_DECODER_FILE))
        self.image_encoder = _load_module(image_encoder, fl.locate_file(IMAGE_ENCODER_FILE))
        self.cache = TextEncoderCache()
        self._prepared = (None, None)  # (source image digest, preprocessed image): background removal of the last reference
        self._interrupt = False

    def pipe(self):
        if self.textured:
            return {"transformer": self.flows["tex_lowres"], "transformer2": self.flows["tex_highres"], "shape_encoder": self.shape_encoder,
                    "texture_decoder": self.texture_decoder, "image_encoder": self.image_encoder}
        return {"transformer": self.flows["ss_flow"], "transformer2": self.flows["shape_lowres"], "transformer3": self.flows["shape_highres"],
                "ss_decoder": self.ss_decoder, "shape_decoder": self.shape_decoder, "image_encoder": self.image_encoder}

    def _prepare_image(self, image):
        digest = hashlib.sha1(image.tobytes() + repr((image.mode, image.size)).encode()).hexdigest()
        if self._prepared[0] != digest:
            set_phase_status("Preparing Image")
            self._prepared = (digest, preprocess_image(image, _rembg))
            check_abort()
        return self._prepared[1]

    def _encode_image(self, image, sizes):
        """DINOv3 features (1, tokens, 1024) FP32 of the preprocessed image at each size, cached on the CPU by image content and size."""
        digest = hashlib.sha1(image.tobytes()).hexdigest()

        def encode(batch):
            with text_encoding_progress(self.image_encoder.layer, title="Encoding Image"):
                return [self.image_encoder(image_to_pixels(image, batch[0], self.device))]

        features = self.cache.encode(encode, list(sizes), device=self.device, cache_keys=[(digest, size) for size in sizes])
        return {size: feature.to(torch.bfloat16) for size, feature in zip(sizes, features)}

    def _denoise(self, flow, noise, cond, coords, params, title, callback, guidance=None, concat=None, dense=False):
        """Euler sampling of one flow over the tokens of one sample (dense structure grid or sparse voxels)."""
        steps = params["steps"]
        guidance = params["guidance_strength"] if guidance is None else guidance
        # upstream computes the dense grid phases once on the CPU (a model buffer), the sparse ones on the device
        phases = flow.phases(coords).to(self.device)
        channels = noise.shape[1]

        def model(x, t, contexts):
            tokens = x.view(1, channels, -1).transpose(1, 2).contiguous() if dense else x[None]
            if concat is not None:
                tokens = torch.cat([tokens, concat[None]], dim=-1)
            outputs = flow(tokens, t, contexts, phases)
            return [out.transpose(1, 2).reshape(noise.shape) if dense else out[0] for out in outputs]

        bar = tqdm(total=steps, desc=title)

        def on_step(step):
            bar.update(1)
            if callback is not None:
                callback(step, None, False, override_num_inference_steps=steps, progress_title=title)
            check_abort()

        if callback is not None:
            callback(-1, None, True, override_num_inference_steps=steps, progress_title=title)
        try:
            return sampler.sample(model, noise, cond, torch.zeros_like(cond), steps, params["rescale_t"], guidance, params["guidance_rescale"],
                                  params["guidance_interval"], self.config["sigma_min"], sampler.dense_std if dense else sampler.sparse_std, on_step)
        finally:
            bar.close()

    def _normalization(self, name):
        stats = self.config[name]
        return (torch.tensor(stats["mean"], dtype=torch.float32, device=self.device)[None],
                torch.tensor(stats["std"], dtype=torch.float32, device=self.device)[None])

    def _decode(self, decoder, x, title, subdivisions=None, coords_only=False):
        with PhaseProgress(decoder.block_count(coords_only), title, "blocks") as progress:
            return decoder([x], subdivisions, coords_only, progress=progress.advance)

    def _structure(self, cond, generator, steps, guidance, resolution, title, callback):
        flow = self.flows["ss_flow"]
        grid = flow.resolution
        noise = torch.randn(1, flow.in_channels, grid, grid, grid, generator=generator, device="cpu").to(self.device)
        axis = torch.arange(grid, device="cpu")
        coords = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)
        params = dict(self.config["samplers"]["structure"], steps=steps)
        latent = self._denoise(flow, noise, cond, coords, params, title, callback, guidance=guidance, dense=True)
        set_phase_status("Decoding Structure")
        occupied = self.ss_decoder(latent) > 0
        if resolution != occupied.shape[2]:
            ratio = occupied.shape[2] // resolution
            occupied = F.max_pool3d(occupied.float(), ratio, ratio, 0) > 0.5
        return torch.argwhere(occupied[0, 0]).int()

    def _shape(self, flow_name, cond, coords, generator, steps, guidance, title, callback):
        flow = self.flows[flow_name]
        noise = torch.randn(coords.shape[0], flow.in_channels, generator=generator, device="cpu").to(self.device)
        params = dict(self.config["samplers"]["shape"], steps=steps)
        slat = self._denoise(flow, noise, cond, coords, params, title, callback, guidance=guidance)
        mean, std = self._normalization("shape_normalization")
        return slat * std + mean

    @torch.inference_mode()
    def _shapes(self, image, seed, steps, guidance, batch_size, pipeline, max_faces, max_tokens, callback):
        """Meshes (glTF axes, shared.mesh3d Mesh on the CPU) of the preprocessed object image."""
        conds = self._encode_image(image, (512,) if pipeline == "512" else (512, 1024))
        meshes = []
        for sample_no in range(batch_size):
            suffix = f" {sample_no + 1}/{batch_size}" if batch_size > 1 else ""
            generator = torch.Generator(device="cpu").manual_seed(seed + sample_no)
            coords = self._structure(conds[512], generator, steps, guidance, 64 if pipeline == "1024" else 32, "Denoising Structure" + suffix, callback)
            if pipeline in ("512", "1024"):
                flow, cond, resolution = ("shape_lowres", conds[512], 512) if pipeline == "512" else ("shape_highres", conds[1024], 1024)
                shape_slat = self._shape(flow, cond, coords, generator, steps, guidance, "Denoising Shape" + suffix, callback)
            else:
                coarse = self._shape("shape_lowres", conds[512], coords, generator, steps, guidance, "Denoising Coarse Shape" + suffix, callback)
                fine_coords = self._decode(self.shape_decoder, SparseTensor(coarse, coords), "Upsampling Shape", coords_only=True)
                del coarse
                resolution = 1536 if pipeline == "1536_cascade" else 1024
                while True:
                    coords = ((fine_coords + 0.5) / 512 * (resolution // 16)).int().unique(dim=0)
                    if coords.shape[0] < max_tokens or resolution == 1024:
                        break
                    resolution -= 128
                del fine_coords
                shape_slat = self._shape("shape_highres", conds[1024], coords, generator, steps, guidance, "Denoising Shape" + suffix, callback)
            fields, _ = self._decode(self.shape_decoder, SparseTensor(shape_slat, coords), "Decoding Shape" + suffix)
            del shape_slat, coords
            set_phase_status("Extracting Surface")
            mesh = list(flexible_dual_grid_to_mesh(fields, *decode_fdg(fields.feats, self.config["shape_decoder"]["voxel_margin"]), resolution))
            del fields
            vertices, faces, _ = remesh_surface(mesh, resolution)
            set_phase_status("Simplifying Mesh")
            surface = finish_surface(vertices.cpu(), faces.cpu(), max_faces)
            del vertices, faces
            meshes.append(to_gltf_mesh(surface.vertices, surface.faces))
            check_abort()
        return meshes

    @torch.inference_mode()
    def _texture(self, mesh, image, seed, steps, guidance, resolution, max_faces, texture_size, keep_alpha, callback):
        """Textured copy (shared.mesh3d Mesh on the CPU, the input frame) of mesh, reduced to max_faces when positive: upstream's texturing
        pipeline, the mesh normalized into the unit cube (TRELLIS axes), encoded as shape latents at resolution^3 voxels, its texture
        latents sampled from the object image and decoded, then baked into a UV atlas."""
        from shared.mesh3d import BackgroundUnwrap, Mesh
        from shared.mesh3d.cleanup import decimate, merge_vertices
        from models.supratex import dual_grid, encoder

        cond = self._encode_image(self._prepare_image(image), (resolution,))[resolution]
        set_phase_status("Preparing Mesh")
        mesh = merge_vertices(Mesh(mesh.vertices.float().cpu(), mesh.faces.cpu()))
        if max_faces and mesh.faces.shape[0] > max_faces:
            mesh = decimate(mesh, max_faces)
        unwrap = BackgroundUnwrap(mesh, max_faces=UNWRAP_REGION_FACES)  # xatlas in a child process, overlapped with the GPU stages
        try:
            low, high = mesh.vertices.amin(0), mesh.vertices.amax(0)
            center, scale = ((low + high) / 2).to(self.device), 0.99999 / float((high - low).max())

            def to_cube(vertices):  # upstream preprocess_mesh: the bounding box centered in the unit cube, glTF Y up -> TRELLIS Z up
                vertices = (vertices.to(self.device) - center) * scale
                return torch.stack([vertices[:, 0], -vertices[:, 2], vertices[:, 1]], dim=1)

            set_phase_status("Encoding Shape")
            faces = mesh.faces.to(self.device)
            voxels, offsets, intersected = dual_grid.mesh_to_flexible_dual_grid((to_cube(mesh.vertices) + 0.5) * resolution, faces, resolution)
            del faces
            check_abort()
            shape, levels = self.shape_encoder([encoder.shape_input(voxels, offsets, intersected)])
            del voxels, offsets, intersected
            flow = self.flows["tex_lowres" if resolution == 512 else "tex_highres"]
            mean, std = self._normalization("shape_normalization")
            concat = (shape.feats - mean) / std
            generator = torch.Generator(device="cpu").manual_seed(seed)
            noise = torch.randn(len(shape), flow.in_channels - concat.shape[1], generator=generator, device="cpu").to(self.device)
            params = dict(self.config["samplers"]["texture"], steps=steps)
            latents = self._denoise(flow, noise, cond, shape.coords, params, "Denoising Texture", callback, guidance=guidance, concat=concat)
            del concat, noise
            mean, std = self._normalization("texture_normalization")
            subdivisions = encoder.decoder_subdivisions(levels, shape.coords)  # the decoder rebuilds exactly the encoded voxels, as upstream
            voxels, _ = self._decode(self.texture_decoder, SparseTensor(latents * std + mean, shape.coords), "Decoding Texture", subdivisions)
            voxels.feats.mul_(0.5).add_(0.5)
            del latents, shape, levels, subdivisions
            set_phase_status("Unwrapping UVs")
            unwrapped = unwrap.result(check_abort=check_abort, progress=True)
        finally:
            unwrap.close()
        cube = Mesh(to_cube(unwrapped.vertices), unwrapped.faces.to(self.device), uvs=unwrapped.uvs.to(self.device))
        _, faces, uvs, base_color, metallic_roughness = bake_textures(cube, voxels, voxels.feats, resolution, texture_size, 0.0)
        del voxels, cube
        # opaque by default, as the official exporter: the decoder's alpha would make the material BLEND
        return Mesh(unwrapped.vertices.cpu(), faces.cpu(), uvs=uvs.cpu(), base_color=base_color if keep_alpha else base_color[..., :3].contiguous(),
                    metallic_roughness=metallic_roughness)

    @generation_progress
    @torch.inference_mode()
    def texture_mesh(self, mesh, image, seed=0, guidance_scale=1.0, resolution=1024, max_faces=0, texture_size=2048, steps=12, callback=None, set_progress_status=None):
        """The Texturer entry (post-processing): PBR textures of mesh (shared.mesh3d Mesh) from the object image (PIL); max_faces 0 keeps the
        mesh's faces."""
        return self._texture(mesh, image, int(seed), int(steps), float(guidance_scale), int(resolution), int(max_faces), int(texture_size), False, callback)

    @generation_progress
    @torch.inference_mode()
    def generate(self, seed, sampling_steps, guide_scale, batch_size, model_mode, custom_settings, original_input_ref_images, input_custom=None,
                 callback=None, set_progress_status=None, **kwargs):
        """trellis2_shape: meshes of the object image; trellis2_texture: textures of the Input Mesh (input_custom)."""
        image = original_input_ref_images[0]
        if self.textured:
            from shared.mesh3d import load_mesh

            settings = DEFAULT_TEXTURE_SETTINGS | custom_settings
            set_phase_status("Loading Mesh")
            source = load_mesh(input_custom)
            meshes = [self._texture(source, image, seed + i, sampling_steps, guide_scale, int(model_mode), int(settings["max_faces"]), int(settings["texture_size"]),
                                    settings["alpha_mode"] == "blend", callback) for i in range(batch_size)]
        else:
            settings = DEFAULT_CUSTOM_SETTINGS | custom_settings
            meshes = self._shapes(self._prepare_image(image), seed, sampling_steps, guide_scale, batch_size, PIPELINES[int(model_mode)], int(settings["max_faces"]),
                                  int(settings["max_tokens"]), callback)
        set_phase_status("Exporting Mesh")
        previews, glbs = [], []
        for mesh in meshes:
            glbs.append(export_glb(mesh))
            previews.append(render_preview(mesh, PREVIEW_SIZE))
            check_abort()
        return {"x": torch.stack(previews, dim=1), "meshes": glbs}
