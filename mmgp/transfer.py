# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Copies between the GPU and RAM of views with gaps, without a temporary of their size.

A view has gaps when its elements do not fill one range of memory: a range of an axis after others (the pages of a [layers, pages, ...]
cache, a window of frames of a [batch, channels, frames, ...] video), a step slice. PyTorch moves bytes between devices in one linear
transfer: a side with gaps, or two sides whose layouts differ, is first made contiguous in a temporary of the whole view on its own device
(the GPU for a GPU view, in both directions), a GPU copy of GBs for a long context's KV cache. Here a view is copied by parts: its
contiguous strips directly when they are large, otherwise chunks of at most STAGING_BYTES that PyTorch stages, so that no temporary
exceeds STAGING_BYTES whatever the layout. A whole tensor that is only permuted (no gaps) is moved by PyTorch as it is, in one transfer."""
import torch

STAGING_BYTES = 16 << 20  # the largest temporary of a copy: parts of a view below it are staged by PyTorch (fewer, larger transfers)


def _dense(tensor):
    """Its elements fill one range of memory, in some order of its axes: no gaps, no overlaps."""
    expected = 1
    for size, stride in sorted(((size, stride) for size, stride in zip(tensor.shape, tensor.stride()) if size != 1), key=lambda item: item[1]):
        if stride != expected:
            return False
        expected *= size
    return True


def copy_(dst, src, non_blocking=False):
    """dst.copy_(src) for tensors of the same shape, without a temporary above STAGING_BYTES."""
    size = dst.numel() * max(dst.element_size(), src.element_size())  # a temporary of either side (dtype conversions stage in the source's)
    if dst.device == src.device or size <= STAGING_BYTES or dst.stride() == src.stride() and dst.dtype == src.dtype and _dense(dst):
        return dst.copy_(src, non_blocking=non_blocking)  # no temporary, or one below STAGING_BYTES
    rows = dst.shape[0]
    step = STAGING_BYTES * rows // size  # rows of the first axis that fit in STAGING_BYTES together
    for start in range(0, rows, max(step, 1)):
        part = slice(start, start + step) if step > 1 else start  # a chunk that PyTorch stages, or a row copied by its own parts
        copy_(dst[part], src[part], non_blocking)
    return dst


def move(tensor, device, dtype=None, non_blocking=False):
    """tensor.to(device, dtype, copy=True) without a temporary above STAGING_BYTES (a dtype conversion converts the whole tensor in one
    otherwise): a tensor without gaps keeps its layout if its dtype is kept, the result is contiguous otherwise."""
    dtype = tensor.dtype if dtype is None else dtype
    if dtype == tensor.dtype and _dense(tensor):
        return tensor.to(device, copy=True, non_blocking=non_blocking)
    return copy_(torch.empty(tensor.shape, dtype=dtype, device=device), tensor, non_blocking)
