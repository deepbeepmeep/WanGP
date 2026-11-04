from dataclasses import dataclass

import torch

from ...types import LatentStateRuntimeCache


@dataclass(frozen=True)
class PreparedConditioning:
    context: torch.Tensor
    projected_context: torch.Tensor | None = None


@dataclass(frozen=True)
class Modality:
    """
    Input data for a single modality (video or audio) in the transformer.
    Bundles the latent tokens, timestep embeddings, positional information,
    and text conditioning context for processing by the diffusion transformer.
    """

    latent: (
        torch.Tensor
    )  # Shape: (B, T, D) where B is the batch size, T is the number of tokens, and D is input dimension
    sigma: torch.Tensor  # Shape: (B,). Current sigma value for prompt/cross-attention AdaLN.
    timesteps: torch.Tensor  # Shape: (B, T) where T is the number of timesteps
    positions: (
        torch.Tensor
    )  # Shape: (B, 3, T) for video, where 3 is the number of dimensions and T is the number of tokens
    context: torch.Tensor | PreparedConditioning
    ref_context: torch.Tensor | None = None
    ref_adaln: torch.Tensor | None = None
    nag: dict | None = None
    enabled: bool = True
    context_mask: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    cross_attention_mask: torch.Tensor | None = None
    frame_indices: torch.Tensor | None = None
    keyframes_mask: torch.Tensor | None = None
    runtime_cache: LatentStateRuntimeCache | None = None
    step_index: int | None = None
    sigma_schedule: torch.Tensor | None = None
    # Leading tokens that carry conditioning (a reference video, a keyframe, a start image, a Multiple Subject Reference) and
    # so are kept in the exact set under Sol-Attn; None when the conditioned tokens are not one leading run.
    sol_sink_tokens: int | None = None
