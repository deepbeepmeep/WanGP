# SPDX-License-Identifier: Apache-2.0
# Pika H3 Flex sampling adapted from Pika-Labs/Pika-H3-Flex, revision
# 64d81b2e98aa8aa1cf078b07a6ef3aaeae1033da (core/sampling.py, evaluation.py).
# Copyright (c) 2026 Mellis, Inc. See PIKA_NOTICE.
# Modified for WanGP: native latent layouts, explicit RNGs and in-place updates.
"""The released six-evaluation Pika H3 Flex consistency sampler."""

import torch

PIKA_SOLVER = "pika_renoise"
PIKA_STEPS = 6
PIKA_VIDEO_SHIFT = 12.0


def sigma_schedule(device):
    # Upstream computes the shifted knots in FP64, then rounds once to FP32.
    base = torch.tensor((1000, 750, 500, 250, 100, 40, 0), dtype=torch.float64, device="cpu") / 1000.0
    return tuple((shift * base / (1.0 + (shift - 1.0) * base)).to(device=device, dtype=torch.float32) for shift in (12.0, 3.0))


def noise(shape, generator, audio=False):
    """Draw in the released layout before converting to WanGP's latent layout."""
    if audio:
        batch, channels, stereo, frames = shape
        value = torch.randn((batch, frames, stereo * channels), generator=generator, dtype=torch.float32, device=generator.device)
        return value.view(batch, frames, stereo, channels).permute(0, 3, 2, 1).contiguous()
    batch, channels, frames, height, width = shape
    return torch.randn((batch, frames, channels, height, width), generator=generator, dtype=torch.float32,
                       device=generator.device).permute(0, 2, 1, 3, 4).contiguous()


def renoise_(sample, velocity_list, sigma, sigma_next, generator, final=False, audio=False):
    """Consume the velocity, predict clean latents and add independent fresh noise."""
    prediction = velocity_list.pop().float().mul_(sigma).add_(sample)
    if final:
        sample.copy_(prediction)
    else:
        fresh = noise(sample.shape, generator, audio=audio).mul_(sigma_next)
        sample.copy_(prediction.mul_(1.0 - sigma_next).add_(fresh))
