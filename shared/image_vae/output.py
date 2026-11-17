# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""What VAE decoders do with decoded tiles and chunks of frames, whatever the model: [-1, 1] values to bytes, on the GPU or in RAM,
and the blending of a tile's edge with the tile before it (tensors (batch, channels, frames, height, width))."""
import torch
from mmgp import transfer


def float_to_uint8(frames):
    """frames in [-1, 1] (rounded in place) as bytes on their own device."""
    return frames.clamp_(-1.0, 1.0).add_(1.0).mul_(127.5).round_().clamp_(0.0, 255.0).to(torch.uint8)


def float_to_video_bytes(frames):
    """frames in [-1, 1] as the bytes WanGP makes of a float video (shared.utils.utils.convert_video_tensor_to_uint8_chunked: in 16
    bits, truncated), on their own device, frames left as they are. A decoder that converts its pieces as it decodes them returns the
    video WanGP would have converted, without a float video in RAM nor its conversion on the CPU."""
    return (frames.to(torch.float16) + 1).mul_(127.5).clamp_(0, 255).to(torch.uint8)


def float_to_cpu_uint8(frames):
    """frames in [-1, 1] (rounded in place) as bytes in RAM, without a RAM copy in their float dtype nor a GPU one of a cropped view."""
    frames.clamp_(-1.0, 1.0).add_(1.0).mul_(127.5).round_().clamp_(0.0, 255.0)
    return transfer.move(frames, "cpu", torch.uint8)


def blend_edge_(head, tail, dim):
    """The first slices of a tile along dim (head, in place) blended with the last ones of the tile before it (tail, the same shape),
    with the values of the usual loops `b[i] = a[-extent + i] * (1 - i / extent) + b[i] * (i / extent)`, which multiply slice by slice
    by a Python number: PyTorch computes such a product of a 16-bit tensor in 32 bits (the weight not rounded to 16 bits first) and
    rounds it; the two products are then summed."""
    extent, dtype = head.shape[dim], head.dtype
    math = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    shape = [1] * head.ndim
    shape[dim] = extent
    ramp = torch.arange(extent, dtype=torch.float64, device=head.device) / extent
    head.copy_((tail.to(math) * (1 - ramp).to(math).view(shape)).to(dtype) + (head.to(math) * ramp.to(math).view(shape)).to(dtype))


def blend_v_edge_(top_edge, tile, blend_extent):
    """tile's first rows blended, in place, with the last rows of the tile above (top_edge)."""
    blend_extent = min(int(top_edge.shape[-2]), int(tile.shape[-2]), int(blend_extent))
    if blend_extent <= 0:
        return
    weights = torch.arange(blend_extent, device=tile.device, dtype=tile.dtype).div_(blend_extent).view(1, 1, 1, blend_extent, 1)
    edge = top_edge[:, :, :, -blend_extent:, :].to(device=tile.device, dtype=tile.dtype)
    edge.mul_(1.0 - weights)
    tile[:, :, :, :blend_extent, :].mul_(weights).add_(edge)


def blend_h_edge_(left_edge, tile, blend_extent):
    """tile's first columns blended, in place, with the last columns of the tile on its left (left_edge)."""
    blend_extent = min(int(left_edge.shape[-1]), int(tile.shape[-1]), int(blend_extent))
    if blend_extent <= 0:
        return
    weights = torch.arange(blend_extent, device=tile.device, dtype=tile.dtype).div_(blend_extent).view(1, 1, 1, 1, blend_extent)
    edge = left_edge[:, :, :, :, -blend_extent:].to(device=tile.device, dtype=tile.dtype)
    edge.mul_(1.0 - weights)
    tile[:, :, :, :, :blend_extent].mul_(weights).add_(edge)
