"""LTX-2 adapter for per-step tiled latent fusion (see ``shared/utils/tiled_fusion.py``).

Every tile is evaluated as its own clip: tile target tokens followed by the co-located crop of its
temporal window's reference (IC-LoRA guide) latent, all with positions starting at zero. The x0
predictions of the target tokens are fused on the full canvas; the Euler step stays in the loop.
The canvas, its blend buffer and the references may live on CPU; only tile crops move to ``device``.
"""

from __future__ import annotations

import torch

from shared.utils.tiled_fusion import TiledFusionPlan

from ...ltx_core.components.patchifiers import VideoLatentPatchifier, get_pixel_coords
from ...ltx_core.types import LatentState, LatentStateRuntimeCache, SpatioTemporalScaleFactors, VideoLatentShape
from .helpers import _prepare_conditioning_context, modality_from_latent_state
from .types import DenoisingFunc


def _tile_template(plan: TiledFusionPlan, patchifier: VideoLatentPatchifier, scale_factors: SpatioTemporalScaleFactors, fps: float, channels: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Positions and keyframe marker shared by all tiles (``[target | reference]``), and the target token count."""
    shape = VideoLatentShape(batch=1, channels=channels, frames=plan.tile_frames, height=plan.tile_height, width=plan.tile_width)
    positions = get_pixel_coords(latent_coords=patchifier.get_patch_grid_bounds(output_shape=shape, device=device), scale_factors=scale_factors, causal_fix=True).float()
    positions[:, 0] /= fps
    tokens = patchifier.get_token_count(shape)
    keyframes_mask = torch.zeros((1, 2 * tokens, 1), device=device)
    keyframes_mask[:, : plan.tile_height * plan.tile_width] = 1.0
    return torch.cat((positions, positions), dim=2), keyframes_mask, tokens


def tiled_fusion_denoising_func(
    *,
    transformer,
    video_context: torch.Tensor,
    plan: TiledFusionPlan,
    references: list[torch.Tensor],
    patchifier: VideoLatentPatchifier,
    scale_factors: SpatioTemporalScaleFactors,
    fps: float,
    device: torch.device,
    tiles_per_call: int = 1,
    callback=None,
    previous_x0: torch.Tensor | None = None,
    export_x0: list | None = None,
    export_latents: slice | None = None,
) -> DenoisingFunc:
    """``references[w]`` is window ``w``'s reference latent, channels-last ``(1, tile_frames, H, W, C)``.

    Conditioned canvas frames (denoise mask < 1, e.g. Media Flow overlap conditioning) keep their mask in every tile.
    ``callback`` receives integer tile progress over all steps; the start event fires just before the first tile.
    Cross-chunk fusion: ``previous_x0`` ``(steps, n, H, W, C)`` holds the previous clip's fused predictions for latents
    ``1..n`` of this canvas (latent 0 is this clip's own start, skipped as in temporal windows) and is blended in with a
    linear ramp from the previous clip to this one; ``export_x0`` receives this canvas' ``export_latents`` predictions at
    every step.
    """
    channels = int(references[0].shape[-1])
    positions, keyframes_mask, tile_tokens = _tile_template(plan, patchifier, scale_factors, fps, channels, device)
    window_masks, runtime_caches = {}, {}
    prepared_context = None
    done_tiles = 0

    def tile_state(canvas_mask: torch.Tensor, window: int, latent: torch.Tensor) -> LatentState:
        # Conditioned frames always span whole frames, so every tile of a window shares one mask and one cache.
        if window not in window_masks:
            target_mask = plan.crop(canvas_mask, plan.windows[window], 0, 0).reshape(1, -1, 1).to(device=device, dtype=torch.float32)
            window_masks[window] = torch.cat((target_mask, torch.zeros_like(target_mask)), dim=1)
            runtime_caches[window] = LatentStateRuntimeCache()
        return LatentState(latent=latent, denoise_mask=window_masks[window], positions=positions, clean_latent=latent, keyframes_mask=keyframes_mask, runtime_cache=runtime_caches[window])

    def tile_latent(canvas: torch.Tensor, window: int, t0: int, y0: int, x0: int) -> torch.Tensor:
        target = plan.crop(canvas, t0, y0, x0).reshape(1, -1, channels).to(device)
        reference = plan.crop(references[window], 0, y0, x0).reshape(1, -1, channels).to(device)
        return torch.cat((target, reference), dim=1)

    def canvas_views(video_state: LatentState) -> tuple[torch.Tensor, torch.Tensor]:
        return video_state.latent.view(1, plan.frames, plan.height, plan.width, channels), video_state.denoise_mask.view(1, plan.frames, plan.height, plan.width, 1)

    def _prewarm(video_state: LatentState, audio_state: LatentState | None, sigmas: torch.Tensor) -> None:
        nonlocal prepared_context
        if prepared_context is None:
            canvas, canvas_mask = canvas_views(video_state)
            prepared_context = _prepare_conditioning_context(transformer, tile_state(canvas_mask, 0, tile_latent(canvas, 0, 0, 0, 0)), video_context, sigmas.to(device), is_audio=False)

    def _cleanup() -> None:
        nonlocal prepared_context
        prepared_context = None

    def tiled_fusion_step(video_state: LatentState, audio_state: LatentState | None, sigmas: torch.Tensor, step_index: int) -> tuple[torch.Tensor | None, None]:
        nonlocal done_tiles
        _prewarm(video_state, audio_state, sigmas)
        canvas, canvas_mask = canvas_views(video_state)
        accumulator = torch.zeros(canvas.shape, dtype=torch.float32, device=canvas.device)
        step_sigmas = sigmas.to(device)
        tiles = plan.tiles(step_index)
        total_tiles, title = (len(sigmas) - 1) * plan.tiles_per_step, f"Detail Refinement - Step {step_index + 1}/{len(sigmas) - 1}"
        if callback is not None and step_index == 0:
            done_tiles = 0
            callback(-1, None, True, override_num_inference_steps=total_tiles, progress_unit="tiles", progress_title=title)
        for start in range(0, len(tiles), tiles_per_call):
            group = tiles[start : start + tiles_per_call]
            modalities = [modality_from_latent_state(tile_state(canvas_mask, tile[0], tile_latent(canvas, *tile)), prepared_context, step_sigmas[step_index], step_index=step_index, sigma_schedule=step_sigmas) for tile in group]
            if len(group) == 1:
                denoised, _ = transformer(video=modalities[0], audio=None, perturbations=None)
                denoised = [denoised]
            else:
                denoised, _ = transformer(video=modalities, audio=[None] * len(group), perturbations=None)
            modalities = None
            if denoised is None or denoised[0] is None:
                return None, None
            for (window, _, y0, x0), tile in zip(group, denoised):
                plan.accumulate(accumulator, tile[:, :tile_tokens].to(accumulator.device).view(1, plan.tile_frames, plan.tile_height, plan.tile_width, channels), window, y0, x0)
            denoised = None
            done_tiles += len(group)
            if callback is not None:
                callback(done_tiles - 1, None, False, override_num_inference_steps=total_tiles, progress_unit="tiles", progress_title=title)
        denoised = plan.normalize(accumulator, step_index)
        accumulator = None
        if previous_x0 is not None:
            count = int(previous_x0.shape[1])
            weight = torch.arange(count, 0, -1, dtype=torch.float32, device=denoised.device).div_(count + 1).view(1, count, 1, 1, 1)
            denoised[:, 1 : count + 1].lerp_(previous_x0[step_index].to(denoised), weight)
        denoised = denoised.to(video_state.latent.dtype)
        if export_x0 is not None:
            export_x0.append(denoised[0, export_latents].cpu())
        return denoised.view_as(video_state.latent), None

    tiled_fusion_step._prewarm = _prewarm
    tiled_fusion_step._cleanup = _cleanup
    return tiled_fusion_step
