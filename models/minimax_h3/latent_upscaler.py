"""MiniMax H3 learned 3D latent upscaler.

Architecture compatible with LBH-123-AI/Minimax_h3_latent_Upscaler (Apache-2.0).
"""

import math

import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from shared.kernels import ops as kernels
from .interrupt import GenerationInterrupted


def _normalization(channels):
    return nn.GroupNorm(32, channels)


def _zero_module(module):
    for parameter in module.parameters():
        parameter.detach().zero_()
    return module


def _check_abort(abort_callback):
    if callable(abort_callback) and abort_callback():
        raise GenerationInterrupted


class ResidualBlock3D(nn.Module):
    def __init__(self, channels, embedding_channels=64, dropout=0.1):
        super().__init__()
        self.in_layers = nn.Sequential(_normalization(channels), nn.SiLU(inplace=True), nn.Conv3d(channels, channels, 3, padding=1))
        self.emb_layers = nn.Sequential(nn.SiLU(), nn.Linear(embedding_channels, 2 * channels))
        self.out_norm = _normalization(channels)
        self.out_layers = nn.Sequential(nn.SiLU(inplace=True), nn.Dropout(dropout), _zero_module(nn.Conv3d(channels, channels, 3, padding=1)))
        self.skip = nn.Identity()

    def forward(self, hidden_states, embedding):
        residual = hidden_states
        hidden_states = self.in_layers(hidden_states)
        scale, shift = self.emb_layers(embedding).to(hidden_states.dtype)[:, :, None, None, None].chunk(2, dim=1)
        hidden_states = self.out_norm(hidden_states)
        pending, hidden_states = [hidden_states], None
        hidden_states = kernels.scale_shift(pending, scale, shift)
        # Inference-only blocks reuse their residual after its final read.
        return residual.add_(self.out_layers(hidden_states))


class TemporalConv3D(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.norm = _normalization(channels)
        self.dwconv = nn.Conv3d(channels, channels, kernel_size=(kernel_size, 1, 1), padding=(kernel_size // 2, 0, 0), groups=channels)
        self.pwconv = _zero_module(nn.Conv3d(channels, channels, 1))

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = F.silu(self.norm(hidden_states), inplace=True)
        hidden_states = self.dwconv(hidden_states)
        return residual.add_(self.pwconv(hidden_states))


class MiniMaxH3LatentUpscaler(nn.Module):
    def __init__(self, in_channels=24, in_blocks=12, out_blocks=12, channels=512, dropout=0.1, temporal_every=2, temporal_kernel=5, temporal_chunk_size=16):
        super().__init__()
        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        self.embed = nn.Sequential(nn.Linear(1, 64), nn.SiLU(), nn.Linear(64, 64))
        self.in_blocks = self._make_blocks(in_blocks, channels, dropout, temporal_every, temporal_kernel)
        self.out_blocks = self._make_blocks(out_blocks, channels, dropout, temporal_every, temporal_kernel)
        self.norm_out = _normalization(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)
        self.temporal_chunk_size = int(temporal_chunk_size)

    @staticmethod
    def _make_blocks(count, channels, dropout, temporal_every, temporal_kernel):
        blocks = nn.ModuleList()
        for index in range(count):
            blocks.append(ResidualBlock3D(channels, dropout=dropout))
            if temporal_every > 0 and index % temporal_every == 0:
                blocks.append(TemporalConv3D(channels, temporal_kernel))
        return blocks

    def _forward_segment(self, latent, scale, target_size, abort_callback, advance):
        embedding = self.embed(latent.new_tensor([[float(scale) - 1.0]])).expand(latent.shape[0], -1)
        _check_abort(abort_callback)
        latent = self.conv_in(latent)
        advance()
        for block in self.in_blocks:
            _check_abort(abort_callback)
            latent = block(latent, embedding) if isinstance(block, ResidualBlock3D) else block(latent)
            advance()
        _check_abort(abort_callback)
        latent = F.interpolate(latent, size=target_size, mode="trilinear", align_corners=False)
        advance()
        for block in self.out_blocks:
            _check_abort(abort_callback)
            latent = block(latent, embedding) if isinstance(block, ResidualBlock3D) else block(latent)
            advance()
        _check_abort(abort_callback)
        latent = F.silu(self.norm_out(latent), inplace=True)
        latent = self.conv_out(latent)
        advance()
        return latent

    def forward(self, latent, scale, target_size=None, abort_callback=None, progress_callback=None):
        target_size = ((latent.shape[2], int(round(latent.shape[-2] * float(scale))), int(round(latent.shape[-1] * float(scale))))
                       if target_size is None else tuple(int(size) for size in target_size))
        target_time, target_height, target_width = target_size
        if target_time != latent.shape[2]:
            raise ValueError("MiniMax H3 latent upscaling must preserve the latent frame count")
        segment_count = max(1, math.ceil(latent.shape[2] / self.temporal_chunk_size))
        steps_per_segment = len(self.in_blocks) + len(self.out_blocks) + 3
        total_steps = segment_count * steps_per_segment
        step = 0

        with tqdm(total=total_steps, desc="H3 Latent Upscaling", unit="layer") as progress:
            def advance():
                nonlocal step
                if callable(progress_callback):
                    progress_callback("Latent Network", step, total_steps)
                step += 1
                progress.update()

            if segment_count == 1:
                return self._forward_segment(latent, scale, target_size, abort_callback, advance)

            overlap = next((block.dwconv.kernel_size[0] // 2 for block in self.in_blocks if isinstance(block, TemporalConv3D)), 0)
            output = latent.new_empty(latent.shape[0], latent.shape[1], latent.shape[2], target_height, target_width)
            for start in range(0, latent.shape[2], self.temporal_chunk_size):
                _check_abort(abort_callback)
                low = max(0, start - overlap)
                high = min(latent.shape[2], start + self.temporal_chunk_size + overlap)
                segment = self._forward_segment(latent[:, :, low:high], scale, (high - low, target_height, target_width), abort_callback, advance)
                source_start = start - low
                count = min(self.temporal_chunk_size, latent.shape[2] - start)
                output[:, :, start:start + count].copy_(segment[:, :, source_start:source_start + count])
                segment = None
            return output


__all__ = ["MiniMaxH3LatentUpscaler", "ResidualBlock3D", "TemporalConv3D"]
