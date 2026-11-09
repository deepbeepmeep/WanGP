# SPDX-License-Identifier: Apache-2.0
# DMAD re-noise sampling adapted from Yzmblog/DMAD, revision
# Sampling code adapted from https://github.com/Yzmblog/DMAD
# Copyright 2026 the DMAD authors. See DMAD_NOTICE.
# Modified for WanGP: native latent layouts, explicit RNGs and in-place updates.
"""The re-noise multistep rule that DMAD-distilled MiniMax H3 students are trained with."""

import torch

DMAD_SOLVER = "dmad_renoise"
DMAD_AUDIO_SHIFT = 2.0  # the students' audio grid; video keeps the selected flow shift (12 upstream)


def renoise_(sample, denoised, sigma_next, generator, final=False):
    """The next input is the clean prediction `denoised` (consumed) re-noised with fresh noise at `sigma_next`."""
    if not final:
        noise = torch.randn(denoised.shape, generator=generator, dtype=denoised.dtype, device=denoised.device)
        denoised.mul_(1.0 - sigma_next).add_(noise.mul_(sigma_next))
    sample.copy_(denoised)
