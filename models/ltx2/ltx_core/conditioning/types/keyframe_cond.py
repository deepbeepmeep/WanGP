import torch

from ...components.patchifiers import get_pixel_coords
from ..item import ConditioningItem
from ...tools import VideoLatentTools
from ...types import LatentState, VideoLatentShape


class VideoConditionByKeyframeIndex(ConditioningItem):
    """
    Conditions video generation on keyframe latents at a specific frame index.
    Appends keyframe tokens to the latent state with positions offset by frame_idx,
    and sets denoise strength according to the strength parameter.
    ``num_pixel_frames=1`` marks a standalone one-frame encode: its temporal span is narrowed to that single pixel frame.
    """

    def __init__(self, keyframes: torch.Tensor, frame_idx: int, strength: float, num_pixel_frames: int | None = None, literal_frame_index: bool = False):
        self.keyframes = keyframes
        self.frame_idx = frame_idx
        self.strength = strength
        self.num_pixel_frames = num_pixel_frames
        self.literal_frame_index = literal_frame_index

    def apply_to(
        self,
        latent_state: LatentState,
        latent_tools: VideoLatentTools,
    ) -> LatentState:
        tokens = latent_tools.patchifier.patchify(self.keyframes)
        latent_coords = latent_tools.patchifier.get_patch_grid_bounds(
            output_shape=VideoLatentShape.from_torch_shape(self.keyframes.shape),
            device=self.keyframes.device,
        )
        positions = get_pixel_coords(
            latent_coords=latent_coords,
            scale_factors=latent_tools.scale_factors,
            causal_fix=latent_tools.causal_fix,
        )
        # Negative continuation offsets retain their legacy prepend convention.
        # Layout appearance references use a literal -1 pixel-frame position.
        remove_prepend = self.frame_idx < 0 and not self.literal_frame_index
        frame_idx = -self.frame_idx if remove_prepend else self.frame_idx
        positions[:, 0, ...] += frame_idx
        if self.num_pixel_frames == 1:
            positions[:, 0, ..., 1:] = positions[:, 0, ..., :1] + 1
        positions = positions.to(dtype=torch.float32)
        positions[:, 0, ...] /= latent_tools.fps

        denoise_mask = torch.full(
            size=(*tokens.shape[:2], 1),
            fill_value=1.0 - self.strength,
            device=self.keyframes.device,
            dtype=self.keyframes.dtype,
        )
        if remove_prepend:
            latent_frame_tokens = self.keyframes.shape[-1] * self.keyframes.shape[-2]
            tokens = tokens[:, latent_frame_tokens:]
            denoise_mask = denoise_mask[:, latent_frame_tokens:]
            positions = positions[:, :, latent_frame_tokens:]
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
