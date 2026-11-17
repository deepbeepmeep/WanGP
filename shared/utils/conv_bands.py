# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Convolutions of large images by bands of rows (stride 1, kernel height 3 with padding 1, or 1 without), optionally after a nearest x2
upsampling.

Output rows s..e-1 read only the (upsampled) input rows s-1..e: each band convolves its rows plus a one-row halo, so every output value has
the same inputs and weights as with the full convolution. Neither the full-size upsampled input nor cuDNN's workspace for the full image is
allocated: at 2048x2048, a VAE decoder convolution takes a workspace of up to twice its output, which no allocator can do without (cuDNN does
not fall back to a smaller one when memory runs out). cuDNN may choose another kernel for a band's shape, so values can differ in rounding only.

The convolutions that band_convs prepares also compute a window of output rows, written in place into a caller's tensor (banded_forward):
where the row-window kernel is the faster choice (shared.kernels.band_conv_triton, channels-last 16-bit activations), it reads the input in
place, with neither bands nor workspace nor halo rows; elsewhere the same rows come from the bands above.
"""
import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

from shared.kernels import dispatch

try:
    from shared.kernels import band_conv_triton as _kernels
except (ImportError, OSError, RuntimeError):
    _kernels = None

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
    if is_banded(conv) and band_bytes == BAND_BYTES:
        return conv(x, upsample=upsample, mode=mode)  # one call of the module: its own bands, or the row-window kernel
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


def conv2d_weights(x, weight, bias, conv, window=None, out=None, out_row=0, add=False):
    """conv(x) for a 3x3 convolution with stride 1 and zero padding 1 given by its tensors (weight (cout, cin, 3, 3), bias; conv: that
    convolution as a function of x), such as one frame of a causal 3D convolution. Where the row-window kernel is the faster choice, x
    is made channels-last for it: its output is too, and so are the activations computed from it, so that a decoder pays that copy once.
    Elsewhere conv runs by bands of rows. window = (first, count): only these output rows; out: written into its rows from out_row
    (added to them with add: conv(x) followed by add_) instead of a new tensor, as banded_forward does."""
    if _kernels is not None:
        layer = _kernels.Weights(weight, bias)
        if _kernels.conv2d_eligible(x, layer, out=out, layout=False) and dispatch.choose("Image VAE Convolutions", owned_triton=True) is not None:
            return _kernels.conv2d(x.contiguous(memory_format=torch.channels_last), layer, window=window, out=out, out_row=out_row, add=add)
    if window is None and out is None:
        return conv_bands(x, conv)
    first, count = (0, x.shape[2]) if window is None else window
    lo, hi = max(0, first - 1), min(x.shape[2], first + count + 1)  # the window's rows and their halo
    y = conv_bands(x[:, :, lo:hi], conv)[:, :, first - lo:first - lo + count]
    if out is None:
        return y
    target = out[:, :, out_row:out_row + count]
    target.add_(y) if add else target.copy_(y)
    return out


def band_rows(height, row_bytes, halo=0, band_bytes=BAND_BYTES):
    """Rows per band for an operation by bands of `height` rows of row_bytes each, a band (its halo rows included) within band_bytes:
    equal bands, so that the last one is no sliver (a few rows would fall below the size from which the row-window kernel is used)."""
    rows = max(1, band_bytes // max(1, row_bytes) - 2 * halo)
    return height if rows >= height else -(-height // -(-height // rows))


def kernel_layout(x):
    """x in the layout the row-window kernel reads (channels-last; channels_last_3d for frames) where that kernel runs x's dtype on its
    device, x itself elsewhere. For the small input of an encoder: its first convolution then writes a channels-last activation, instead
    of a full-size one being copied to that layout before the next convolution."""
    if _kernels is None or not _kernels.eligible(x):
        return x
    strides, step = [1] * x.dim(), x.shape[1]
    for axis in range(x.dim() - 1, 1, -1):
        strides[axis], step = step, step * x.shape[axis]
    strides[0] = step
    if x.stride() == tuple(strides):
        return x
    # a copy even when PyTorch calls x contiguous in that format (a picture permuted from height x width x channels, whose
    # one-picture axis has any stride): cuDNN would still write its first convolution in the other layout
    return x.clone(memory_format=torch.channels_last_3d if x.dim() == 5 else torch.channels_last)


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


def banded_forward(conv, x, upsample=False, mode="nearest", window=None, out=None, out_row=0, add=False):
    """The forward of a convolution prepared by band_convs: conv(x), or conv of x's x2 upsampling. window = (first, count): only these
    output rows. out: written into its rows from out_row (added to them with add: conv(x) followed by add_) instead of a new tensor."""
    if _kernels is not None and _kernels.conv2d_eligible(x, conv, upsample, mode, out) and dispatch.choose("Image VAE Convolutions", owned_triton=True) is not None:
        return _kernels.conv2d(x, conv, upsample=upsample, window=window, out=out, out_row=out_row, add=add)
    if window is not None:  # from the input rows the window reads (its halo included), not from the whole of x
        scale, (first, count) = 2 if upsample else 1, window
        lo = max(0, (first - 1) // scale)
        x, window = x[:, :, lo:min(x.shape[2], (first + count) // scale + 1)], (first - scale * lo, count)
    y = conv_bands(x, functools.partial(nn.Conv2d.forward, conv), upsample, mode)
    if window is not None:
        y = y[:, :, window[0]:window[0] + window[1]]
    if out is None:
        return y
    target = out[:, :, out_row:out_row + y.shape[2]]
    target.add_(y) if add else target.copy_(y)
    return out


def is_banded(conv):
    """Whether conv is a convolution prepared by band_convs (its forward takes upsample, mode, window, out, out_row and add), also
    under forwards that wrap it since (functools.update_wrapper's __wrapped__ chain: the weight-loading hooks of an offloaded model)."""
    forward = conv.forward if isinstance(conv, nn.Conv2d) else None
    while forward is not None and getattr(forward, "func", None) is not banded_forward:
        forward = getattr(forward, "__wrapped__", None)
    return forward is not None


def band_convs(module):
    """Runs every 3x3 convolution of module with stride 1 and padding 1 by bands of rows (when its input exceeds BAND_BYTES), or by the
    row-window kernel (banded_forward)."""
    for conv in module.modules():
        if type(conv) is nn.Conv2d and conv.kernel_size == (3, 3) and conv.stride == (1, 1) and conv.padding == (1, 1) and conv.dilation == (1, 1) and conv.padding_mode == "zeros":
            conv.forward = functools.partial(banded_forward, conv)
