# SUPRA-Tex (supra3d/supra-tex, MIT License; material adapter under the Qwen Research License): PBR textures of a mesh from one object
# image. The image's albedo and metallic / roughness maps (Qwen Image 2.1 with the material adapter) are projected on the visible surface
# of the mesh's TRELLIS.2 voxels, encoded, completed over the whole surface by a 2.1B flow transformer, decoded and baked in a UV atlas.
# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import json
import math
import os

import numpy as np
import torch
from accelerate import init_empty_weights
from mmgp import offload, runtime
from PIL import Image
from safetensors.torch import load_file

from shared.mesh3d import Mesh, glb_read_extras, load_mesh, mesh_to_glb
from shared.mesh3d.cleanup import BackgroundUnwrap, decimate, merge_vertices
from shared.mesh3d.render import preview
from shared.utils.mesh_postprocessing import OBJECT_CAMERA
from shared.utils import files_locator as fl
from shared.utils.phase_progress import PhaseProgress, check_abort, generation_progress, set_phase_status
from shared.utils.text_encoder_cache import TextEncoderCache
from models.trellis2.dit import fused_split_map
from models.trellis2.export import bake_textures
from models.trellis2.sparse import SparseTensor
from models.trellis2.vae import SparseUnetDecoder
from . import completion, dual_grid, encoder, naf, observation, pixal3d
from .alignment import Placement, place_mesh
from .material import RESOLUTION as MATERIAL_RESOLUTION, MaterialEstimator

PROJECT = "supratex"
TEXTURE_ARCH, SHAPE_ARCH = "supratex_texture", "pixal3d"
NAF_FILE = "supratex/naf_bf16.safetensors"
CONFIG_FILE, PARTIAL_FILE = "supratex/completion_config.json", "supratex/completion_partial_normalization.json"
MATERIAL_LORA_FILE, MATERIAL_EMBEDDINGS_FILE = "supratex/supratex_material_lora_bf16.safetensors", "supratex/supratex_material_embeddings_bf16.safetensors"
SHAPE_ENCODER_FILE, TEXTURE_ENCODER_FILE = "trellis2/trellis2_shape_encoder_fp16.safetensors", "trellis2/trellis2_texture_encoder_fp16.safetensors"
TEXTURE_DECODER_FILE = "trellis2/trellis2_texture_decoder_fp16.safetensors"  # identical to the one of TRELLIS.2 (shared file)
ENCODER_CONFIG = {"model_channels": [64, 128, 256, 512, 1024], "latent_channels": 32, "num_blocks": [0, 4, 8, 16, 4]}
DECODER_CONFIG = {"out_channels": 6, "model_channels": [1024, 512, 256, 128, 64], "latent_channels": 32, "num_blocks": [4, 16, 8, 4, 0], "pred_subdiv": False}
PREVIEW_SIZE = 512
UNWRAP_REGION_FACES = 40000
COMPLETION_STEPS = 25  # upstream's texture sampler steps


def _keep_checkpoint_dtypes(model):
    """Checkpoint dtypes (no MMGP conversion); the top-level layers receive the FP32 tensors upstream feeds them."""
    model._convertWeightsFloatTo = None
    for module in [model, *model.children()]:
        if not isinstance(module, torch.nn.ModuleList) and not hasattr(module, "_lock_dtype"):
            module._lock_dtype = None
    return model.eval().requires_grad_(False)


def _load_module(module, filename, **kwargs):
    offload.load_model_data(module, filename, writable_tensors=False, default_dtype=None, **kwargs)
    return _keep_checkpoint_dtypes(module)


def object_square(image):
    """Upstream frontend preprocessing: at most 1024 px, background removed when the image has no transparency (rembg), a square
    crop centered on the object (alpha above 0.8) with 10% padding. Returns the straight RGBA square."""
    from models.hunyuan3d.shape_pipeline import remove_background

    scale = min(1.0, 1024.0 / max(image.size))
    if scale < 1:
        image = image.resize((int(image.width * scale), int(image.height * scale)), Image.Resampling.LANCZOS)
    rgba = remove_background(image).convert("RGBA")
    foreground = np.argwhere(np.asarray(rgba)[..., 3] > 0.8 * 255)
    if not foreground.size:
        raise ValueError("No object was found in the Object Image: use a picture of one object, or an RGBA image with its alpha mask.")
    (y0, x0), (y1, x1) = foreground.min(0), foreground.max(0)
    side = int(max(x1 - x0, y1 - y0) * 1.1)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return rgba.crop((cx - side // 2, cy - side // 2, cx + side // 2, cy + side // 2))


class model_factory:
    """supratex_texture: textures of a supplied mesh (also the SUPRA-Tex Texturer); pixal3d: a pixel-aligned mesh of the object image, its
    GLB carrying the object image's camera (OBJECT_CAMERA) for the texturer."""

    def __init__(self, model_filename, model_type, base_model_type, text_encoder_filename=None, save_quantized=False, **kwargs):
        self.device = torch.device("cuda")
        self.textured, self.geometry = base_model_type == TEXTURE_ARCH, base_model_type == SHAPE_ARCH
        self.qwen = self.pixal3d = None
        if self.textured:
            self._load_texturing(model_filename, model_type, text_encoder_filename, save_quantized, **kwargs)
        else:  # its flows are the model file, MoGe-2 the module after it
            self._load_geometry(model_filename[0], model_filename[-1], model_type, save_quantized)
        self.cache = TextEncoderCache()
        self._interrupt = False

    def _load_texturing(self, model_filename, model_type, text_encoder_filename, save_quantized, **kwargs):
        from models.qwen21.qwen21_handler import family_handler as qwen21

        with init_empty_weights():
            self.transformer = completion.CompletionTransformer()
            self.shape_encoder, self.texture_encoder = encoder.SparseUnetEncoder(6, **ENCODER_CONFIG), encoder.SparseUnetEncoder(6, **ENCODER_CONFIG)
            self.texture_decoder = SparseUnetDecoder(**DECODER_CONFIG)
        # BF16 checkpoint (FP32 normalization statistics: buffers); INT8 linears compute in BF16; the fused q/k/v projections are split
        # at load for the attention kit
        _load_module(self.transformer, model_filename[0], fp32_dtype=torch.bfloat16, fused_split_map=fused_split_map(self.transformer.model_channels))
        if save_quantized:
            from wgp import save_quantized_model
            save_quantized_model(self.transformer, model_type, model_filename[0], torch.bfloat16, None)
        _load_module(self.shape_encoder, fl.locate_file(SHAPE_ENCODER_FILE))
        _load_module(self.texture_encoder, fl.locate_file(TEXTURE_ENCODER_FILE))
        _load_module(self.texture_decoder, fl.locate_file(TEXTURE_DECODER_FILE))
        self.qwen, self.qwen_pipe = qwen21.load_model([model_filename[1]], "qwen_image_21_7B", "qwen_image_21_7B", {}, text_encoder_filename=text_encoder_filename, **kwargs)
        embeddings = load_file(fl.locate_file(MATERIAL_EMBEDDINGS_FILE))["modality_embeddings"]
        self.material = MaterialEstimator(self.qwen, embeddings, fl.locate_file(MATERIAL_LORA_FILE))
        with open(fl.locate_file(CONFIG_FILE), encoding="utf-8") as reader:
            dataset = json.load(reader)["dataset"]["args"]
        with open(fl.locate_file(PARTIAL_FILE), encoding="utf-8") as reader:
            partial = json.load(reader)
        tensor = lambda values: torch.tensor(values, dtype=torch.float32, device=self.device)
        self.shape_stats = tuple(tensor(dataset["shape_slat_normalization"][key]) for key in ("mean", "std"))
        self.pbr_stats = tuple(tensor(dataset["pbr_slat_normalization"][key]) for key in ("mean", "std"))
        self.partial_stats = (tensor(partial["mean"]), tensor(partial["std"]))

    def _load_geometry(self, flows_filename, moge_filename, model_type, save_quantized):
        from models.recon3d import moge2
        from models.recon3d.dinov2 import QKV_SPLIT_MAP
        from models.trellis2.dinov3 import DINOv3ViT
        from models.trellis2.trellis2_main import CONFIG_PATH, IMAGE_ENCODER_FILE, SHAPE_DECODER_FILE, SS_DECODER_FILE
        from models.trellis2.vae import SparseStructureDecoder

        with open(CONFIG_PATH, encoding="utf-8") as reader:
            self.trellis_config = json.load(reader)  # TRELLIS.2's decoders, DINOv3, samplers and shape statistics: shared by Pixal3D
        config = self.trellis_config
        with init_empty_weights():
            container = torch.nn.ModuleDict({name: pixal3d.Pixal3DFlow(**config["dit"], **stage) for name, stage in pixal3d.STAGES.items()})
            ss_decoder, shape_decoder = SparseStructureDecoder(**config["ss_decoder"]), SparseUnetDecoder(**config["shape_decoder"])
            image_encoder, naf_model, moge = DINOv3ViT(**config["image_encoder"]), naf.NAF(), moge2.MoGe2.from_config()
        # BF16 checkpoint (INT8 linears compute in BF16); the fused q/k/v and k/v projections are split at load for the attention kit
        offload.load_model_data(container, flows_filename, writable_tensors=False, default_dtype=None, fp32_dtype=torch.bfloat16, fused_split_map=fused_split_map(config["dit"]["model_channels"]))
        container.eval().requires_grad_(False)
        if save_quantized:
            from wgp import save_quantized_model
            save_quantized_model(container, model_type, flows_filename, torch.bfloat16, None)
        flows = {name: _keep_checkpoint_dtypes(flow) for name, flow in container.items()}
        ss_decoder.lock_checkpoint_fp32_modules()
        self.ss_decoder = _load_module(ss_decoder, fl.locate_file(SS_DECODER_FILE))
        self.shape_decoder = _load_module(shape_decoder, fl.locate_file(SHAPE_DECODER_FILE))
        self.image_encoder = _load_module(image_encoder, fl.locate_file(IMAGE_ENCODER_FILE))
        self.naf = _load_module(naf_model, fl.locate_file(NAF_FILE))
        self.moge = _load_module(moge, moge_filename, fused_split_map=QKV_SPLIT_MAP)
        self.flows = flows
        self.transformer = flows["ss_flow"]  # wgp's main model of the pixal3d type (TRELLIS.2 does the same)
        self.pixal3d = pixal3d.Pixal3DGeometry(flows, self.ss_decoder, self.shape_decoder, self.image_encoder, self.naf, config, self.device)

    def pipe(self):
        pipe = {}
        if self.textured:
            pipe.update({"transformer": self.transformer, "shape_encoder": self.shape_encoder, "texture_encoder": self.texture_encoder,
                         "texture_decoder": self.texture_decoder, "material_transformer": self.qwen_pipe["transformer"],
                         "text_encoder": self.qwen_pipe["text_encoder"], "vision_encoder": self.qwen_pipe["vision_encoder"], "vae": self.qwen_pipe["vae"]})
        if self.geometry:
            flows = {"transformer": self.flows["ss_flow"], "shape_flow": self.flows["shape_lowres"], "shape_flow2": self.flows["shape_highres"]}
            pipe.update(flows, ss_decoder=self.ss_decoder, shape_decoder=self.shape_decoder, image_encoder=self.image_encoder, naf=self.naf, moge=self.moge)
        return pipe

    # ---------------------------------------------------------------- stages

    def _materials(self, square):
        """Albedo and MRA maps (3, R, R) of the object square, cached on the CPU by image content."""
        import hashlib

        square = square if square.size == (MATERIAL_RESOLUTION,) * 2 else square.resize((MATERIAL_RESOLUTION,) * 2, Image.Resampling.LANCZOS)
        key = ("supratex_materials", hashlib.sha1(square.tobytes()).hexdigest())
        maps = self.cache.encode(lambda _: [torch.cat(self.material(square, self.device))], ["materials"], device=self.device, cache_keys=[key])[0]
        return square, maps[:3].float(), maps[3:].float()

    def _encode(self, canonical, faces):
        """Surface voxels (dual grid), shape latents and encoder levels of the placed mesh."""
        set_phase_status("Encoding Shape")
        grid = observation.RESOLUTION
        voxels, offsets, intersected = dual_grid.mesh_to_flexible_dual_grid((canonical + 0.5) * grid, faces, grid)
        check_abort()
        shape, levels = self.shape_encoder([encoder.shape_input(voxels, offsets, intersected)])
        del offsets, intersected
        return voxels.int(), shape, levels

    def _observe(self, surface, shape, placement, square, albedo, mra):
        """Observation message of the object image's visible surface voxels."""
        set_phase_status("Projecting the Object Image")
        alpha = torch.from_numpy(np.array(square.getchannel("A"), copy=True)).to(self.device)
        trusted = observation.trusted_mask(alpha)
        stats = {}
        visible_voxels = observation.visible_cells(surface, placement.camera_angle_x, placement.distance, trusted, stats)
        visible = SparseTensor(None, visible_voxels).find(surface) >= 0
        del visible_voxels
        uv = observation.project(surface, placement.camera_angle_x, placement.distance)
        encoded, _ = self.texture_encoder([observation.texture_input(surface, visible, albedo, mra, uv)])
        del uv
        message = observation.observation_message(encoded, surface, visible, shape.coords, self.partial_stats)
        check_abort()
        return message, stats

    def _complete(self, shape, message, seed, steps, callback):
        set_phase_status("Completing Texture")
        condition = (shape.feats - self.shape_stats[0]) / self.shape_stats[1]
        runtime.ensure_loaded(self.transformer)  # prepare() reads the input layers and statistics outside the hooked forward
        prepared = self.transformer.prepare(shape.coords, condition, message)
        noise = torch.randn((len(shape), 32), generator=torch.Generator(device=self.device).manual_seed(seed), device=self.device)
        title = "Completing Texture"
        step_callback = None if callback is None else lambda step, latents, force: callback(step, latents, force, override_num_inference_steps=steps, progress_title=title)
        sampled = completion.sample(self.transformer, noise, condition, prepared, steps=steps, callback=step_callback)
        del prepared, noise
        return sampled * self.pbr_stats[1] + self.pbr_stats[0]

    def _decode(self, latents, shape, levels):
        set_phase_status("Decoding Texture")
        flags = encoder.decoder_subdivisions(levels, shape.coords)
        decoded, _ = self.texture_decoder([SparseTensor(latents, shape.coords)], subdivisions=flags)
        return decoded.replace((decoded.feats * 0.5 + 0.5).clamp_(0, 1))

    @torch.inference_mode()
    def _texture(self, mesh, image, seed, max_faces, texture_size, steps, callback, placement=None):
        """Textured copy (shared.mesh3d Mesh on the CPU, the input frame) of mesh, reduced to max_faces when positive. placement: the
        mesh's known placement in the canonical frame (a Pixal3D mesh), else it is fitted to the image's silhouette."""
        set_phase_status("Preparing Object Image")
        square, albedo, mra = self._materials(object_square(image))
        set_phase_status("Preparing Mesh")
        mesh = merge_vertices(Mesh(mesh.vertices.float().cpu(), mesh.faces.cpu()))
        if max_faces and mesh.faces.shape[0] > max_faces:
            mesh = decimate(mesh, max_faces)
        unwrap = BackgroundUnwrap(mesh, max_faces=UNWRAP_REGION_FACES)  # xatlas in a child process, overlapped with the GPU stages
        try:
            vertices, faces = mesh.vertices.to(self.device), mesh.faces.to(self.device)
            if placement is None:
                set_phase_status("Placing the Mesh")
                alpha = torch.from_numpy(np.array(square.getchannel("A"), copy=True)).to(self.device)
                placement = place_mesh(vertices, faces, alpha)
            canonical = placement.apply(vertices)
            surface, shape, levels = self._encode(canonical, faces)
            del canonical
            message, stats = self._observe(surface, shape, placement, square, albedo, mra)
            from tqdm import tqdm
            fit = "pixel-aligned mesh" if math.isnan(placement.iou) else f"silhouette IoU {placement.iou:.3f}, yaw {placement.yaw:.1f}, pitch {placement.pitch:.1f}"
            tqdm.write(f"SUPRA-Tex: {fit}, field of view {math.degrees(placement.camera_angle_x):.1f} deg, "
                       f"{stats['admitted']} of {stats['trusted']} object pixels observed, {len(message)} of {len(shape)} cells")
            latents = self._complete(shape, message, seed, steps, callback)
            del message
            attrs = self._decode(latents, shape, levels)
            del latents, shape, levels, surface
            set_phase_status("Unwrapping UVs")
            unwrapped = unwrap.result(check_abort=check_abort, progress=True)
        finally:
            unwrap.close()
        set_phase_status("Baking Texture")
        canonical = Mesh(placement.apply(unwrapped.vertices.to(self.device)), unwrapped.faces.to(self.device), uvs=unwrapped.uvs.to(self.device))
        _, out_faces, uvs, base_color, metallic_roughness = bake_textures(canonical, attrs, attrs.feats, observation.RESOLUTION, texture_size, 0.0)
        del attrs, canonical
        # opaque, as TRELLIS.2's default exporter: the decoder's alpha (253-255 here) would make the material BLEND, and a viewer then shows
        # the faces behind through it
        return Mesh(unwrapped.vertices.cpu(), out_faces.cpu(), uvs=uvs.cpu(), base_color=base_color[..., :3].contiguous(), metallic_roughness=metallic_roughness)

    @generation_progress
    @torch.inference_mode()
    def texture_mesh(self, mesh, image, seed=0, max_faces=0, texture_size=2048, steps=COMPLETION_STEPS, object_camera=None, callback=None, set_progress_status=None):
        """The Texturer entry (post-processing): PBR textures of mesh (shared.mesh3d Mesh) completed from the object image (PIL); max_faces
        0 keeps the mesh's faces. object_camera: the camera of a Pixal3D mesh (its GLB's OBJECT_CAMERA), used instead of a fitted one."""
        return self._texture(mesh, image, int(seed), int(max_faces), int(texture_size), int(steps), callback, placement=_camera_placement(object_camera, self.device))

    @torch.inference_mode()
    def _geometry(self, square, seed, steps, guidance, max_faces, callback):
        """Pixal3D mesh (shared.mesh3d Mesh on the CPU) of the object square, in the canonical frame (the glTF frame: Y up, facing +Z),
        and its placement (the camera the image was seen with)."""
        from models.trellis2.export import finish_surface, remesh_surface
        from models.trellis2.fdg_mesh import decode_fdg, flexible_dual_grid_to_mesh

        rgba = np.asarray(square, dtype=np.float32) / 255
        condition = Image.fromarray((rgba[..., :3] * rgba[..., 3:4] * 255).clip(0, 255).astype(np.uint8))  # premultiplied over black
        set_phase_status("Estimating the Camera")
        camera_angle_x = self.camera_angle(condition)
        distance = 0.5 / math.tan(camera_angle_x / 2)  # the image spans the canonical cube's width at the origin
        generator = torch.Generator(device="cpu").manual_seed(seed)
        latents, coords, resolution = self.pixal3d.generate(condition, camera_angle_x, distance, generator, steps, guidance, callback)
        with PhaseProgress(self.shape_decoder.block_count(), "Decoding Shape", "blocks") as progress:
            fields, _ = self.shape_decoder([SparseTensor(latents, coords)], progress=progress.advance)
        del latents, coords
        set_phase_status("Extracting Surface")
        mesh = list(flexible_dual_grid_to_mesh(fields, *decode_fdg(fields.feats, self.trellis_config["shape_decoder"]["voxel_margin"]), resolution))
        del fields
        vertices, faces, _ = remesh_surface(mesh, resolution)
        set_phase_status("Simplifying Mesh")
        mesh = finish_surface(vertices.cpu(), faces.cpu(), max_faces)
        identity = torch.eye(3, device=self.device)
        return mesh, Placement(identity, torch.zeros(3, device=self.device), 1.0, torch.zeros(3, device=self.device), camera_angle_x, distance, float("nan"))

    def camera_angle(self, condition):
        """Horizontal field of view of the object image (MoGe-2), cached by image content."""
        import hashlib

        key = ("supratex_camera", hashlib.sha1(condition.tobytes()).hexdigest())
        return float(self.cache.encode(lambda _: [torch.tensor([pixal3d.camera_angle_from_moge(self.moge, condition, self.device)], device="cpu")], ["camera"], cache_keys=[key])[0][0])

    # ---------------------------------------------------------------- generation

    @generation_progress
    @torch.inference_mode()
    def generate(self, input_prompt, seed, sampling_steps=25, guide_scale=7.5, batch_size=1, original_input_ref_images=None, input_custom=None,
                 custom_settings=None, callback=None, set_progress_status=None, **kwargs):
        """supratex_texture: textures of the Input Mesh (input_custom); pixal3d: meshes of the object image, each GLB with its camera."""
        settings = custom_settings or {}
        max_faces, texture_size = int(settings.get("max_faces", 0)), int(settings.get("texture_size", 2048))
        image = original_input_ref_images[0]
        cameras = []
        if self.textured:
            set_phase_status("Loading Mesh")
            source = load_mesh(input_custom)
            camera = glb_read_extras(input_custom).get(OBJECT_CAMERA) if str(input_custom).lower().endswith(".glb") else None  # a Pixal3D mesh
            placement = _camera_placement(camera, self.device)
            meshes = [self._texture(source, image, seed + i, max_faces, texture_size, sampling_steps, callback, placement=placement) for i in range(batch_size)]
        else:
            set_phase_status("Preparing Object Image")
            square = object_square(image)
            meshes = []
            for i in range(batch_size):
                mesh, placement = self._geometry(square, seed + i, sampling_steps, guide_scale, max_faces, callback)
                meshes.append(mesh)
                cameras.append({OBJECT_CAMERA: {"camera_angle_x": placement.camera_angle_x, "distance": placement.distance}})
        set_phase_status("Exporting Mesh")
        previews, glbs = [], []
        for i, mesh in enumerate(meshes):
            previews.append(preview(mesh.to(self.device), PREVIEW_SIZE).cpu())
            glbs.append(mesh_to_glb(mesh, extras=cameras[i] if cameras else None))
            check_abort()
        return {"x": torch.stack(previews, dim=1), "meshes": glbs}


def _camera_placement(object_camera, device):
    """Placement of a Pixal3D mesh (in the canonical frame, seen with the object image's camera) from its GLB's OBJECT_CAMERA; None
    without one (the texturer then fits the placement to the image's silhouette)."""
    if not object_camera:
        return None
    identity, zeros = torch.eye(3, device=device), torch.zeros(3, device=device)
    return Placement(identity, zeros, 1.0, zeros.clone(), float(object_camera["camera_angle_x"]), float(object_camera["distance"]), float("nan"))
