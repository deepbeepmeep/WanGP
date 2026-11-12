import torch

from ...components.patchifiers import get_pixel_coords
from ...tools import VideoLatentTools
from ...types import LatentState, VideoLatentShape
from ..item import ConditioningItem


class VideoConditionByReferenceLatent(ConditioningItem):
    def __init__(self, latent: torch.Tensor, strength: float = 1.0, frame_idx: int = 0, downscale_factor: int = 1):
        self.latent = latent
        self.strength = strength
        self.frame_idx = int(frame_idx)
        self.downscale_factor = max(1, int(downscale_factor))

    def apply_to(self, latent_state: LatentState, latent_tools: VideoLatentTools) -> LatentState:
        tokens = latent_tools.patchifier.patchify(self.latent)
        latent_shape = VideoLatentShape.from_torch_shape(self.latent.shape)
        positions = get_pixel_coords(
            latent_coords=latent_tools.patchifier.get_patch_grid_bounds(output_shape=latent_shape, device=self.latent.device),
            scale_factors=latent_tools.scale_factors,
            causal_fix=latent_tools.causal_fix,
        ).to(dtype=torch.float32)

        frame_idx = self.frame_idx
        remove_prepend = frame_idx < 0
        if remove_prepend:
            frame_idx = -frame_idx
        positions[:, 0, ...] += frame_idx
        positions[:, 0, ...] /= latent_tools.fps
        if self.downscale_factor != 1:
            positions[:, 1, ...] *= self.downscale_factor
            positions[:, 2, ...] *= self.downscale_factor

        denoise_mask = torch.full(
            size=(*tokens.shape[:2], 1),
            fill_value=1.0 - self.strength,
            device=self.latent.device,
            dtype=self.latent.dtype,
        )
        if remove_prepend:
            frame_tokens = latent_tools.patchifier.get_token_count(latent_shape._replace(frames=1))
            tokens = tokens[:, frame_tokens:]
            denoise_mask = denoise_mask[:, frame_tokens:]
            positions = positions[:, :, frame_tokens:]
            positions[:, 0, ...] -= latent_tools.scale_factors.time / latent_tools.fps

        keyframes_mask = None
        if latent_state.keyframes_mask is not None:
            keyframes_mask = torch.cat([latent_state.keyframes_mask, torch.zeros_like(denoise_mask)], dim=1)

        return LatentState(
            latent=torch.cat([latent_state.latent, tokens], dim=1),
            denoise_mask=torch.cat([latent_state.denoise_mask, denoise_mask], dim=1),
            positions=torch.cat([latent_state.positions, positions], dim=2),
            clean_latent=torch.cat([latent_state.clean_latent, tokens], dim=1),
            keyframes_mask=keyframes_mask,
        )
