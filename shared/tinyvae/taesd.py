# TAESD decoder architecture (https://github.com/madebyollin/taesd), Copyright (c) 2023 Ollin Boer Bohan,
# MIT License (see shared/tinyvae/LICENSES/taesd-MIT.txt). Modified for WanGP.
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


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


def Decoder(latent_channels: int = 4, use_midblock_gn: bool = False) -> nn.Sequential:
    midblock_kwargs = dict(use_midblock_gn=use_midblock_gn)
    return nn.Sequential(
        Clamp(), conv(latent_channels, 64), nn.ReLU(),
        Block(64, 64, **midblock_kwargs), Block(64, 64, **midblock_kwargs), Block(64, 64, **midblock_kwargs), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), conv(64, 3),
    )


def F16Decoder() -> nn.Sequential:
    return nn.Sequential(
        Clamp(), conv(64, 256), nn.ReLU(),
        Block(256, 256), Block(256, 256), Block(256, 256), nn.Upsample(scale_factor=2), conv(256, 128, bias=False),
        Block(128, 128), Block(128, 128), Block(128, 128), nn.Upsample(scale_factor=2), conv(128, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), conv(64, 16), nn.PixelShuffle(2),
    )
