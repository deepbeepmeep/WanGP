# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Convolutions of large images by bands of rows (stride 1, kernel height 3 with padding 1, or 1 without), optionally after a nearest x2
upsampling.

Output rows s..e-1 read only the (upsampled) input rows s-1..e: each band convolves its rows plus a one-row halo, so every output value has
the same inputs and weights as with the full convolution. Neither the full-size upsampled input nor cuDNN's workspace for the full image is
allocated: at 2048x2048, a VAE decoder convolution takes a workspace of up to twice its output, which no allocator can do without (cuDNN does
not fall back to a smaller one when memory runs out). cuDNN may choose another kernel for a band's shape, so values can differ in rounding only.
"""
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

BAND_BYTES = 32 << 20  # against 128 MB: VAE encode / decode peaks 0.1-0.4 GiB lower at the 1024 tile, a few % of VAE time either way


def _upsample(x, mode):
    if mode == "nearest" and x.dtype in (torch.float16, torch.bfloat16):
        # Nearest only copies values. Autocast promotes it to FP32, although the following convolution rounds back to
        # 16-bit; avoid that full-size FP32 allocation and conversion without changing any sampled value.
        with torch.autocast(x.device.type, enabled=False):
            return F.interpolate(x, scale_factor=2.0, mode=mode)
    return F.interpolate(x, scale_factor=2.0, mode=mode)


def conv_bands(x, conv, upsample=False, mode="nearest", band_bytes=BAND_BYTES):
    """conv(x), or conv(F.interpolate(x, scale_factor=2, mode=mode)) with upsample, for x (batch, channels, height, width); conv (a module or a
    function) runs on each band, whose halo output rows are dropped."""
    n, c, h, w = x.shape
    scale = 2 if upsample else 1
    row_bytes = n * c * scale * scale * w * x.element_size()
    if row_bytes * h <= band_bytes:
        return conv(_upsample(x, mode) if upsample else x)
    rows = max(1, band_bytes // row_bytes - 2)
    out = None
    for start in range(0, h, rows):
        stop = min(h, start + rows)
        lo, hi = max(0, start - 1), min(h, stop + 1)
        band = _upsample(x[:, :, lo:hi], mode) if upsample else x[:, :, lo:hi]
        first = 1 if start > 0 else 0  # the halo row above the band's first output row
        band = band[:, :, scale * (start - lo) - first:scale * (stop - lo) + (stop < h)]
        band = conv(band)[:, :, first:first + scale * (stop - start)]
        if out is None:  # in the layout of the input (channels_last stays channels_last)
            layout = torch.channels_last if x.stride(1) == 1 else torch.contiguous_format
            out = torch.empty((n, band.shape[1], scale * h, band.shape[-1]), dtype=band.dtype, device=band.device, memory_format=layout)
        out[:, :, scale * start:scale * stop].copy_(band)
        del band
    return out


def downsample_bands(x, conv, band_bytes=BAND_BYTES):
    """3x3 stride-2 conv with the image VAE's right/bottom zero pad. x is borrowed; only a strip is padded at a time."""
    n, c, height, width = x.shape
    rows = max(1, band_bytes // (n * c * (width + 1) * x.element_size() * 2) - 1)
    if rows >= height // 2:
        return conv(F.pad(x, (0, 1, 0, 1)))
    out = None
    for start in range(0, height // 2, rows):
        stop = min(height // 2, start + rows)
        end = 2 * stop + 1
        band = F.pad(x[:, :, 2*start:min(height, end)], (0, 1, 0, max(0, end-height)))
        band = conv(band)
        if out is None:
            layout = torch.channels_last if x.stride(1) == 1 else torch.contiguous_format
            out = torch.empty((n, band.shape[1], height // 2, width // 2), dtype=band.dtype, device=band.device, memory_format=layout)
        out[:, :, start:stop].copy_(band)
        del band
    return out


def band_convs(module):
    """Runs every 3x3 convolution of module with stride 1 and padding 1 by bands of rows (when its input exceeds BAND_BYTES)."""
    for conv in module.modules():
        if type(conv) is nn.Conv2d and conv.kernel_size == (3, 3) and conv.stride == (1, 1) and conv.padding == (1, 1) and conv.dilation == (1, 1) and conv.padding_mode == "zeros":
            conv.forward = functools.partial(conv_bands, conv=functools.partial(nn.Conv2d.forward, conv))
