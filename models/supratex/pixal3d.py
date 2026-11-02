# Pixal3D image-to-3D geometry (TencentARC/Pixal3D, MIT License; pixal3d/pipelines/pixal3d_image_to_3d.py, pixal3d/modules/
# projected_image_feature_extractor.py, pixal3d/modules/{,sparse/}attention/proj_attention.py; SUPRA-Tex's native geometry runner), on
# WanGP's TRELLIS.2 port (models/trellis2): the TRELLIS.2 shape cascade (sparse structure, shape at 512, shape at up to 1536) whose flows
# are conditioned on the object image through camera-aligned projection. Each voxel of the flow's grid is projected into the image with the
# canonical camera (field of view from MoGe-2), its DINOv3 patch features (and their NAF upsampling to 512 px for the shape flows) are
# sampled there and added through a per-block linear to the cross-attention output on the 5 global DINOv3 tokens. The generated mesh is
# pixel-aligned: the canonical frame (Y up, the camera on +Z) is the TRELLIS grid frame, so the mesh needs no placement before texturing.
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from shared.kernels import ops
from shared.utils.phase_progress import PhaseProgress, check_abort, set_phase_status
from models.trellis2 import sampler
from models.trellis2.dit import CrossAttention, TrellisFlowDiT, TransformerBlock
from models.trellis2.dinov3 import image_to_pixels
from models.trellis2.sparse import SparseTensor

STAGES = {"ss_flow": {"in_channels": 8, "out_channels": 8, "resolution": 16, "proj_in_channels": 1024},
          "shape_lowres": {"in_channels": 32, "out_channels": 32, "resolution": 32, "proj_in_channels": 2048},
          "shape_highres": {"in_channels": 32, "out_channels": 32, "resolution": 64, "proj_in_channels": 2048}}
GLOBAL_TOKENS = 5  # DINOv3 class token and registers
NAF_SIZE = 512
MAX_TOKENS = 49152  # Pixal3D's 1536 cascade lowers its resolution by 128 until the fine voxels fit this budget
SS_RESOLUTION = 32


class ProjCrossAttention(nn.Module):
    """SparseProjectAttention / ProjectAttention: the cross-attention on the global tokens plus a linear of the projected features."""

    def __init__(self, channels, ctx_channels, heads, proj_in_channels):
        super().__init__()
        self.cross_attn_block = CrossAttention(channels, ctx_channels, heads)
        self.proj_linear = nn.Linear(proj_in_channels, channels)


class ProjBlock(TransformerBlock):
    def __init__(self, channels, ctx_channels, heads, mlp_ratio, proj_in_channels):
        super().__init__(channels, ctx_channels, heads, mlp_ratio)
        self.cross_attn = ProjCrossAttention(channels, ctx_channels, heads, proj_in_channels)

    def forward(self, x, mod, context, rope):
        """context: (global tokens (1, 5, 1024), projected features (tokens, proj_in_channels)) BF16."""
        global_tokens, projected = context
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.modulation.float() + mod).to(mod.dtype).chunk(6, dim=1)
        rows = x.view(-1, x.shape[-1])
        spans = [(0, rows.shape[0], 0)]
        y = self.self_attn.attend([ops.norm_modulate([rows], self.norm1, shift_msa, scale_msa, spans).view(x.shape)], rope)
        ops.gated_linear(self.self_attn.to_out, [y.view(rows.shape)], rows, gate_msa, spans)
        cross = self.cross_attn.cross_attn_block
        y = cross.attend([self.norm2(x)], global_tokens)
        ops.linear_residual_(cross.to_out, [y.view(rows.shape)], rows)
        del y
        ops.linear_residual_(self.cross_attn.proj_linear, [projected], rows)
        ops.mlp([ops.norm_modulate([rows], self.norm3, shift_mlp, scale_mlp, spans)], self.mlp.mlp[0], self.mlp.mlp[2], "gelu", residual=rows, gate=gate_mlp[0])
        return x


class Pixal3DFlow(TrellisFlowDiT):
    """SparseStructureFlowModel / ElasticSLatFlowModel in Pixal3D's proj mode."""

    def __init__(self, in_channels, out_channels, proj_in_channels, model_channels=1536, cond_channels=1024, num_blocks=30, num_heads=12,
                 mlp_ratio=5.3334, rope_freq=(1.0, 10000.0), resolution=None):
        super().__init__(in_channels, out_channels, model_channels, cond_channels, num_blocks, num_heads, mlp_ratio, rope_freq, resolution)
        self.blocks = nn.ModuleList([ProjBlock(model_channels, cond_channels, num_heads, mlp_ratio, proj_in_channels) for _ in range(num_blocks)])


def project_features(features, coords, grid, camera_angle_x, distance, mesh_scale, image_size):
    """ProjGrid: features (1, C, h, w) of the image sampled (bilinear, border) at the projections of the voxels coords (N, 3) of a
    grid^3 grid spanning [-1, 1] (endpoints included), seen by Pixal3D's front camera. Returns (N, C) BF16."""
    points = (2 * coords.double() / (grid - 1) - 1) / (2 * mesh_scale)  # x right, y up, z towards the camera
    depth = distance - points[:, 2]
    focal = 16.0 / math.tan(camera_angle_x / 2.0) * image_size / 32.0
    pixels = torch.stack([focal * points[:, 0] / (depth + 1e-8) + image_size / 2, -focal * points[:, 1] / (depth + 1e-8) + image_size / 2], 1)
    grid_points = ((pixels + 0.5) / image_size * 2 - 1).float().view(1, -1, 1, 2)
    out = torch.empty((coords.shape[0], features.shape[1]), dtype=torch.bfloat16, device=features.device)
    for start in range(0, coords.shape[0], 1 << 15):
        part = grid_points[:, start:start + (1 << 15)]
        out[start:start + part.shape[1]] = F.grid_sample(features.float(), part, mode="bilinear", align_corners=False, padding_mode="border")[0, :, :, 0].T
    return out


def camera_angle_from_moge(moge, image, device):
    """Horizontal field of view (radians) of the object image (premultiplied over black) estimated by MoGe-2 at its highest detail."""
    from models.recon3d import moge2

    colors = torch.from_numpy(np.array(image.convert("RGB"), copy=True)).to(device).float().div_(255)
    low, high = moge.config["num_tokens_range"]
    with PhaseProgress(moge.block_count(), "Estimating the Camera", "blocks") as progress:
        output = moge(colors.permute(2, 0, 1)[None].contiguous(), high, progress.advance)
    intrinsics = moge2.point_map(output, colors.shape[:2])[4]
    return 2.0 * math.atan(colors.shape[1] / (2.0 * float(intrinsics[0, 0])))


class Pixal3DGeometry:
    """The shape cascade of one object image (preprocessed: a square crop premultiplied over black)."""

    def __init__(self, flows, ss_decoder, shape_decoder, image_encoder, naf, config, device):
        self.flows, self.ss_decoder, self.shape_decoder, self.image_encoder, self.naf = flows, ss_decoder, shape_decoder, image_encoder, naf
        self.config, self.device = config, device

    def _dino(self, image, size):
        """Global tokens (1, 5, 1024) BF16 and patch features (1, 1024, size / 16, size / 16) FP32 of the image at size px."""
        tokens = self.image_encoder(image_to_pixels(image, size, self.device))
        side = size // 16
        patches = tokens[:, GLOBAL_TOKENS:].transpose(1, 2).reshape(1, -1, side, side).float()
        return tokens[:, :GLOBAL_TOKENS].to(torch.bfloat16), patches

    def _naf(self, image, size, patches):
        pixels = torch.from_numpy(np.array(image.resize((size, size), Image.LANCZOS).convert("RGB"), copy=True)).to(self.device)
        pixels = pixels.permute(2, 0, 1)[None].float().div_(255)
        return self.naf(pixels, patches, NAF_SIZE)

    def _denoise(self, flow, noise, cond, coords, params, title, callback, dense=False):
        steps = params["steps"]
        phases = flow.phases(coords).to(self.device)
        channels = noise.shape[1]

        def model(x, t, contexts):
            tokens = x.view(1, channels, -1).transpose(1, 2).contiguous() if dense else x[None]
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
        negative = tuple(torch.zeros_like(part) for part in cond)
        try:
            return sampler.sample(model, noise, cond, negative, steps, params["rescale_t"], params["guidance_strength"], params["guidance_rescale"],
                                  params["guidance_interval"], self.config["sigma_min"], sampler.dense_std if dense else sampler.sparse_std, on_step)
        finally:
            bar.close()

    def generate(self, image, camera_angle_x, distance, generator, steps, guidance, callback=None, mesh_scale=1.0, max_tokens=MAX_TOKENS):
        """Shape latents (N, 32) FP32 (denormalized), their voxels (N, 3) and the surface resolution of the cascade."""
        set_phase_status("Encoding Image")
        global_512, patches_512 = self._dino(image, 512)
        samplers = self.config["samplers"]
        project = lambda features, coords, grid, size: project_features(features, coords, grid, camera_angle_x, distance, mesh_scale, size)
        # sparse structure: the dense 16^3 grid, DINOv3 features only
        flow = self.flows["ss_flow"]
        grid = flow.resolution
        axis = torch.arange(grid, device="cpu")
        dense = torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=-1).reshape(-1, 3)
        noise = torch.randn(1, flow.in_channels, grid, grid, grid, generator=generator, device="cpu").to(self.device)
        cond = (global_512, project(patches_512, dense.to(self.device), grid, 512))
        latent = self._denoise(flow, noise, cond, dense, dict(samplers["structure"], steps=steps, guidance_strength=guidance), "Denoising Structure", callback, dense=True)
        del cond, noise
        set_phase_status("Decoding Structure")
        occupied = self.ss_decoder(latent) > 0
        ratio = occupied.shape[2] // SS_RESOLUTION
        coords = torch.argwhere(F.max_pool3d(occupied.float(), ratio, ratio, 0)[0, 0] > 0.5).int()
        del latent, occupied
        # coarse shape at 512: DINOv3 and NAF features
        set_phase_status("Encoding Image")
        upsampled = self._naf(image, 512, patches_512)
        cond = (global_512, torch.cat([project(patches_512, coords, SS_RESOLUTION, 512), project(upsampled, coords, SS_RESOLUTION, 512)], 1))
        del upsampled
        mean, std = (torch.tensor(self.config["shape_normalization"][key], dtype=torch.float32, device=self.device)[None] for key in ("mean", "std"))
        flow = self.flows["shape_lowres"]
        noise = torch.randn(coords.shape[0], flow.in_channels, generator=generator, device="cpu").to(self.device)
        coarse = self._denoise(flow, noise, cond, coords, dict(samplers["shape"], steps=steps, guidance_strength=guidance), "Denoising Coarse Shape", callback) * std + mean
        del cond, noise
        with PhaseProgress(self.shape_decoder.block_count(True), "Upsampling Shape", "blocks") as progress:
            fine = self.shape_decoder([SparseTensor(coarse, coords)], None, True, progress=progress.advance)
        del coarse
        # fine shape: Pixal3D's quantization of the 512 voxels to the fine latent grid, at most max_tokens of them
        resolution = 1536
        while True:
            grid = resolution // 16
            coords = ((fine.float() + 0.5) / 512 * (grid - 1)).round().int().unique(dim=0)
            if coords.shape[0] < max_tokens or resolution == 1024:
                break
            resolution -= 128
        del fine
        set_phase_status("Encoding Image")
        global_1024, patches_1024 = self._dino(image, 1024)
        upsampled = self._naf(image, 1024, patches_1024)
        cond = (global_1024, torch.cat([project(patches_1024, coords, grid, 1024), project(upsampled, coords, grid, 1024)], 1))
        del upsampled, patches_1024
        flow = self.flows["shape_highres"]
        noise = torch.randn(coords.shape[0], flow.in_channels, generator=generator, device="cpu").to(self.device)
        latents = self._denoise(flow, noise, cond, coords, dict(samplers["shape"], steps=steps, guidance_strength=guidance), "Denoising Shape", callback) * std + mean
        return latents, coords, resolution
