# Hi3D v1.0 mesh refinement for WanGP, inference port of hi3d-official/Hi3Dv1.0 (sparc3d/pipelines/sparc3d_pipeline.py and
# sparc3d/schedulers: Hi3D Open License Agreement 1.0; see LICENSE, NOTICE and THIRD_PARTY_LICENSES.md in this folder).
# A coarse mesh and a picture of the same object in, a detailed geometry-only mesh out: up to three passes on structure grids of 64, 128
# and 192 voxels, each sampling 32-channel latents on the surface voxels of the current mesh (Heun flow matching, guidance against a
# zero condition while t >= 0.7), decoding them 8x finer and extracting the next mesh (extraction grids 512, 1024, 1536).
# Modified by DeepBeepMeep for WanGP: model loading through MMGP, the WanGP generation contract (GLB bytes and previews), explicit
# devices, the conditioning cached on the CPU, the guidance branches run block by block, the output kept in the input mesh's frame
# (upstream recentres it and scales it to a unit radius), and the shared mesh toolkit for cleanup, decimation, previews and GLB export.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import hashlib
import json
import os

import torch
from accelerate import init_empty_weights
from mmgp import offload
from tqdm import tqdm

from models.hunyuan3d.dinov2 import Dinov2Model
from shared.mesh3d import Mesh, load_mesh, mesh_to_glb
from shared.mesh3d.cleanup import decimate, remove_floaters
from shared.mesh3d.mesh import vertex_normals
from shared.mesh3d.render import preview
from shared.utils import files_locator as fl
from shared.utils.phase_progress import PhaseProgress, check_abort, generation_progress, set_phase_status, text_encoding_progress
from shared.utils.text_encoder_cache import TextEncoderCache
from .conditioner import CLIPVisionModel, clip_pixels, condition_tokens, dino_pixels, has_alpha, prepare_image
from .decoder import SparconvVAE
from .dit import Sparc3DTransformer, fused_split_map
from .extract import extract_surface
from .thin_parts import restore_thin_parts
from .voxelize import adaptive_voxels

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "configs", "hi3d.json")
VAE_FILE, CLIP_FILE, DINO_FILE = "hi3d/hi3d_vae_fp16.safetensors", "hi3d/clip_vit_large_patch14_vision_fp16.safetensors", "hunyuan3d/dinov2_large_fp16.safetensors"
DINO_CONFIG = {"hidden_size": 1024, "num_hidden_layers": 24, "num_attention_heads": 16, "image_size": 518, "patch_size": 14}
LEVELS = (512, 1024, 1536)
COMPONENT_FACE_RATIO = 0.005
PREVIEW_SIZE = 512
DEFAULT_SETTINGS = {"level": 1024, "max_faces": 0}


def _keep_dtypes(module):
    """MMGP keeps the checkpoint dtypes (no conversion to the shared default dtype)."""
    module._convertWeightsFloatTo = None
    return module.eval().requires_grad_(False)


def _load(module, filename, **kwargs):
    offload.load_model_data(module, filename, writable_tensors=False, default_dtype=None, **kwargs)
    return _keep_dtypes(module)


def _image_digest(image):
    digest = hashlib.sha1(image.tobytes())
    digest.update(repr((image.mode, image.size)).encode())
    return digest.hexdigest()


def flow_schedule(steps, tokens, sigma_min, time_rescale_factor, device):
    """Upstream FlowMatchSchedule.set_timesteps: base times rescaled twice (fixed factor, then a factor warped by the token count),
    sigmas from sigma_min to 1, model times inflated by a small churn factor (no noise added). Returns (inflated times, sigmas)."""
    def rational(t, factor):
        return factor * t / (1.0 + (factor - 1.0) * t)

    warp = max(1.0, min(5.0, (19.0 * tokens - 20000.0) / (3.0 * tokens + 60000.0)))
    warped = rational(rational(torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32, device=device), time_rescale_factor), warp)
    sigmas = sigma_min + (1.0 - sigma_min) * warped
    gammas = torch.where(warped[:-1] > 0.985, 0.04, 0.002)
    return sigmas[:-1] * (1.0 + gammas), sigmas


def guidance_scale(t):
    """Upstream effective guidance against the zero condition: 4 at the first time, 10 t^2 on [0.7, 1), none below 0.7."""
    return 4.0 if t >= 1.0 else 10.0 * t * t if t >= 0.7 else 1.0


class model_factory:
    def __init__(self, model_filename, model_type, base_model_type, save_quantized=False):
        self.device = torch.device("cuda")
        self._interrupt = False
        with open(CONFIG_PATH, encoding="utf-8") as reader:
            self.config = json.load(reader)
        config = self.config["transformer"]
        self.transformers = []
        for submodel_no, (filename, resolution) in enumerate(zip(model_filename, (64, 128)), start=1):
            with init_empty_weights():
                transformer = Sparc3DTransformer(resolution=resolution, **config)
            # BF16 checkpoint (INT8 linears compute in BF16); the fused q/k/v and k/v projections are split at load for the attention kit
            offload.load_model_data(transformer, filename, writable_tensors=False, default_dtype=None, fp32_dtype=torch.bfloat16,
                                    fused_split_map=fused_split_map(config["model_channels"]))
            if save_quantized:
                from wgp import save_quantized_model
                save_quantized_model(transformer, model_type, filename, torch.bfloat16, None, submodel_no=submodel_no)
            transformer.dtype = transformer.blocks[0].mlp.mlp[0].bias.dtype  # biases stay in the compute dtype (also in INT8 checkpoints)
            self.transformers.append(_keep_dtypes(transformer))
        self.transformer = self.transformers[0]
        with init_empty_weights():
            vae, dino, clip = SparconvVAE(**self.config["vae"]), Dinov2Model(**DINO_CONFIG), CLIPVisionModel()
        self.vae = _load(vae, fl.locate_file(VAE_FILE))
        self.image_encoder = _load(dino, fl.locate_file(DINO_FILE))
        self.image_encoder2 = _load(clip, fl.locate_file(CLIP_FILE))
        self.latent_mean = torch.tensor(self.config["latent_mean"], dtype=torch.float32, device=self.device)
        self.latent_std = torch.tensor(self.config["latent_std"], dtype=torch.float32, device=self.device)
        self.cache = TextEncoderCache()
        self._prepared = (None, None)  # (source image digest, prepared image)

    def pipe(self):
        return {"transformer": self.transformers[0], "transformer2": self.transformers[1], "vae": self.vae, "image_encoder": self.image_encoder,
                "image_encoder2": self.image_encoder2}

    # ---------------------------------------------------------------- conditioning

    def _object_image(self, image):
        """Upstream framing of the object; an image without transparency first goes through rembg BiRefNet General."""
        digest = _image_digest(image)
        if self._prepared[0] != digest:
            source = image
            if not (has_alpha(image) and image.convert("RGBA").getextrema()[3][0] < 255):
                set_phase_status("Removing Background")
                from rembg import remove
                from preprocessing.rembg.assets import OBJECT_MODEL
                from shared.utils.utils import new_rembg_session
                source = remove(image.convert("RGB"), session=new_rembg_session(OBJECT_MODEL))
            self._prepared = (digest, prepare_image(source))
            check_abort()
        return self._prepared[1]

    def _condition(self, image):
        """(1, 2917, 2048) FP32 DINOv2-L + CLIP ViT-L/14 tokens of the prepared image, cached on the CPU."""
        def encode(_):
            layers = list(self.image_encoder.encoder.layer) + list(self.image_encoder2.vision_model.encoder.layers)
            with text_encoding_progress(layers, next_status="Refining Geometry", title="Encoding Image"):
                dino = self.image_encoder(dino_pixels(image, self.device))
                clip = self.image_encoder2(clip_pixels(image, self.device))
                return [condition_tokens(dino, clip)]

        return self.cache.encode(encode, ["image"], device=self.device, cache_keys=[("hi3d_dinov2_clip", _image_digest(image))])[0]

    # ---------------------------------------------------------------- refinement

    def _sample(self, transformer, condition, voxels, steps, generator, title, callback):
        """One pass: Heun flow sampling of the latents (voxels, 32) on the voxels, denormalized to the VAE latent space."""
        plan = transformer.plan(voxels)
        dtype = transformer.dtype
        contexts = [condition.to(dtype)]
        negative = [None]  # upstream's all-zero condition
        sample = torch.randn((voxels.shape[0], 32), generator=generator, device="cpu", dtype=torch.float32).to(self.device)
        times, sigmas = flow_schedule(steps, voxels.shape[0], **self.config["scheduler"], device="cpu")

        def velocity(x, t):
            guided = guidance_scale(t)
            outputs = transformer(x, torch.full((1,), 1000.0 * t, dtype=torch.float32, device=self.device), contexts + (negative if t >= 0.7 else []), plan)
            if len(outputs) == 1:
                return outputs[0]
            positive, uncond = outputs
            return uncond.add_(positive.sub_(uncond), alpha=guided) if guided != 1.0 else positive

        if callback is not None:
            callback(-1, None, True, override_num_inference_steps=steps, progress_title=title)
        for i in tqdm(range(steps), desc=title):
            t, t_next = float(times[i]), float(sigmas[i + 1])
            first = velocity(sample, t)
            predicted = sample - (t - t_next) * first
            second = velocity(predicted, t_next)
            del predicted
            sample = sample - (t - t_next) * 0.5 * (first + second)
            del first, second
            if callback is not None:
                callback(i, None, False, override_num_inference_steps=steps, progress_title=title)
            check_abort()
        del plan
        return sample * self.latent_std + self.latent_mean

    def _decode(self, latents, voxels, frame):
        """Mesh (vertices in the input mesh's frame) of the decoded latents of a pass whose voxels were placed by frame."""
        with PhaseProgress(4, "Decoding Volume", "stages") as progress:
            attributes, coords = self.vae(latents, voxels, progress.advance)
        set_phase_status("Extracting Surface")
        vertices, faces = extract_surface(attributes, coords, frame.resolution * 8)
        del attributes, coords
        if faces.shape[0] == 0:
            raise ValueError("Hi3D produced an empty surface: check that the Object Image and the Input Mesh show the same object.")
        check_abort()
        return Mesh(frame.to_mesh(vertices / 8.0), faces)

    def refine(self, mesh, condition, level, steps, seed, callback=None, prefix=""):
        ladder = self.config["ladder"]
        passes = LEVELS.index(level) + 1
        generator = torch.Generator(device="cpu").manual_seed(seed)
        for k in range(passes):
            set_phase_status("Voxelizing Mesh")
            voxels, frame = adaptive_voxels(mesh.vertices.float(), mesh.faces, ladder["structure"][k], ladder["budgets"][k])
            if frame.resolution != ladder["structure"][k]:
                tqdm.write(f"[Hi3D] pass {k + 1}: {mesh.faces.shape[0]} faces exceed the {ladder['budgets'][k]} voxel budget at {ladder['structure'][k]}^3, "
                      f"refined on a {frame.resolution}^3 grid ({frame.resolution * 8}^3 surface grid)")
            del mesh
            transformer = self.transformers[0 if k == 0 else 1]
            latents = self._sample(transformer, condition, voxels, steps, generator, f"{prefix}Refining Geometry {LEVELS[k]} ({k + 1}/{passes})", callback)
            mesh = self._decode(latents, voxels, frame)
            del latents, voxels
        return mesh

    # ---------------------------------------------------------------- generation

    def _refined(self, source, condition, level, max_faces, steps, seed, callback=None, prefix=""):
        """The refined source mesh, cleaned up (upstream's floater removal), with the thin parts of source it lost (perforated or vanished
        sheets) restored from source, decimated to max_faces (0: the level's default) and with its smooth normals."""
        mesh = self.refine(source, condition, level, steps, seed, callback, prefix)
        set_phase_status("Reducing Mesh")
        mesh = remove_floaters(mesh, min_face_ratio=COMPONENT_FACE_RATIO)
        mesh, restored = restore_thin_parts(source, mesh)
        if restored:
            tqdm.write(f"[Hi3D] {restored} faces of thin parts lost by the refinement were restored from the input mesh")
        mesh = decimate(mesh, max_faces or self.config["ladder"]["faces"][LEVELS.index(level)], parts=2).to(self.device)
        check_abort()
        return mesh.replace(normals=vertex_normals(mesh.vertices, mesh.faces))

    @generation_progress
    @torch.inference_mode()
    def refine_mesh(self, mesh, image, level=1024, max_faces=0, steps=25, seed=0, callback=None, set_progress_status=None):
        """The 3D Refiner entry (post-processing): mesh (shared.mesh3d.Mesh) refined at level from the object image (PIL), in its own frame."""
        condition = self._condition(self._object_image(image))
        return self._refined(Mesh(mesh.vertices.float().to(self.device), mesh.faces.long().to(self.device)), condition, int(level), int(max_faces), int(steps), int(seed), callback)

    @generation_progress
    @torch.inference_mode()
    def generate(self, seed, sampling_steps, batch_size, custom_settings, original_input_ref_images, input_custom, callback=None, set_progress_status=None,
                 **kwargs):
        settings = DEFAULT_SETTINGS | custom_settings
        level, max_faces = int(settings["level"]), int(settings["max_faces"])
        image = self._object_image(original_input_ref_images[0])
        condition = self._condition(image)
        set_phase_status("Loading Mesh")
        source = load_mesh(input_custom, device=self.device)
        source = Mesh(source.vertices.float(), source.faces.long())
        previews, glbs = [], []
        for sample_no in range(batch_size):
            prefix = f"{sample_no + 1}/{batch_size} " if batch_size > 1 else ""
            mesh = self._refined(source, condition, level, max_faces, sampling_steps, seed + sample_no, callback, prefix)
            set_phase_status("Exporting Mesh")
            previews.append(preview(mesh, PREVIEW_SIZE).cpu())
            glbs.append(mesh_to_glb(mesh.to("cpu")))
            del mesh
            check_abort()
        return {"x": torch.stack(previews, dim=1), "meshes": glbs}
