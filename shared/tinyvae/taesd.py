# TAESD decoder architecture (https://github.com/madebyollin/taesd), Copyright (c) 2023 Ollin Boer Bohan,
# MIT License (see shared/tinyvae/LICENSES/taesd-MIT.txt). Modified for WanGP.
from __future__ import annotations

import contextlib
import math
from typing import Any

import torch
import torch.nn as nn

from shared.kernels import dispatch
from shared.utils.conv_bands import BAND_BYTES

try:
    from shared.kernels import band_conv_triton as _kernels
except (ImportError, OSError, RuntimeError):
    _kernels = None


def conv(n_in: int, n_out: int, **kwargs: Any) -> nn.Conv2d:
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kwargs)


class Clamp(nn.Module):
    def forward(self, value: Any) -> Any:
        return torch.tanh(value / 3) * 3


class Block(nn.Module):
    def __init__(self, n_in: int, n_out: int, use_midblock_gn: bool = False) -> None:
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()
        self.pool = None
        if use_midblock_gn:
            conv1x1, n_gn = lambda n_in, n_out: nn.Conv2d(n_in, n_out, 1, bias=False), n_in * 4
            self.pool = nn.Sequential(conv1x1(n_in, n_gn), nn.GroupNorm(4, n_gn), nn.ReLU(inplace=True), conv1x1(n_gn, n_in))

    def forward(self, value: Any) -> Any:
        if self.pool is not None:
            value = value + self.pool(value)
        return self.fuse(self.conv(value) + self.skip(value))


class Bands(nn.Sequential):
    """The layers stage by stage (a stage starts at each upsampling). A stage whose activations exceed BAND_BYTES runs by bands of its input
    rows, each with the halo of rows that the stage's 3x3 convolutions read: every output value has the same inputs as with the whole frame,
    and no full-size activation or cuDNN workspace is allocated (TAEH3 in 16 bits, a 2720x1536 frame: 1.5 GiB -> 0.26 GiB, as fast). A stage
    with a GroupNorm, whose statistics span the frame, runs whole. Activations are channels-last: 2.5 times faster 16-bit convolutions.
    Inside `kernels()`, on the GPUs where they beat cuDNN, the stages run on the band kernels (shared.kernels.band_conv_triton): the same
    bands, 1.7 times faster at 2720x1536 and 4.7 times at 832x480, where several frames then share each launch."""

    plans = None  # inside kernels(): per stage, its kernel operations, or None for the layers themselves

    def __init__(self, *layers: nn.Module) -> None:
        super().__init__(*layers)
        starts = [0] + [index for index, layer in enumerate(layers) if index and isinstance(layer, nn.Upsample)]
        self.stages = []  # (layers, upscale, halo in input rows or None for a stage that runs whole, channels of the widest activation)
        for first, last in zip(starts, starts[1:] + [len(layers)]):
            modules = [module for layer in layers[first:last] for module in layer.modules()]
            convs = [module for module in modules if isinstance(module, nn.Conv2d)]
            scale = 2 if isinstance(layers[first], nn.Upsample) else 1
            halo = None if any(isinstance(module, nn.GroupNorm) for module in modules) else math.ceil(sum(conv.kernel_size == (3, 3) for conv in convs) / scale)
            self.stages.append((layers[first:last], scale, halo, max(conv.out_channels for conv in convs)))

    @contextlib.contextmanager
    def kernels(self, latent: Any) -> Any:
        """Decodes of latents like `latent` (frames, channels, h, w) inside the block use the band kernels where they apply, their weights
        prepared once and released at the end. Yields how many frames to decode per call."""
        frames = 1
        if _kernels is not None and dispatch.choose("Tiny VAE Convolutions", owned_triton=_kernels.eligible(latent)):
            self.plans = [_kernels.plan(list(layers), fold=index == len(self.stages) - 1) for index, (layers, *_) in enumerate(self.stages)]
            if self.plans[-1] is not None:
                upscale = math.prod(scale for _, scale, *_ in self.stages)
                frames = _kernels.frames_together(self.plans[-1], latent.shape[-2] * upscale, latent.shape[-1] * upscale, latent.element_size())
        try:
            yield frames
        finally:
            self.plans = None

    def forward(self, value: Any) -> Any:
        value = value.contiguous(memory_format=torch.channels_last)
        for index, (layers, scale, halo, widest) in enumerate(self.stages):
            if self.plans is not None and self.plans[index] is not None:
                value = _kernels.run_stage(self.plans[index], value, BAND_BYTES)
                continue
            batch, _, height, width = value.shape
            rows = height if halo is None else max(1, BAND_BYTES // (batch * widest * scale * scale * width * value.element_size()))
            out = None
            for start in range(0, height, rows):
                stop = min(height, start + rows)
                low = max(0, start - (halo or 0))
                band = value[:, :, low:min(height, stop + (halo or 0))]
                for layer in layers:
                    band = layer(band)
                if rows >= height:
                    out = band
                    break
                if out is None:
                    out = torch.empty((batch, band.shape[1], scale * height, band.shape[-1]), dtype=band.dtype, device=band.device, memory_format=torch.channels_last)
                out[:, :, scale * start:scale * stop].copy_(band[:, :, scale * (start - low):scale * (stop - low)])
                del band
            value = out
        return value


def Decoder(latent_channels: int = 4, use_midblock_gn: bool = False) -> nn.Sequential:
    midblock_kwargs = dict(use_midblock_gn=use_midblock_gn)
    return Bands(
        Clamp(), conv(latent_channels, 64), nn.ReLU(),
        Block(64, 64, **midblock_kwargs), Block(64, 64, **midblock_kwargs), Block(64, 64, **midblock_kwargs), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), conv(64, 3),
    )
