# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Pictures decoded or encoded by overlapping tiles, for VAEs whose layers need the whole of what they are given (GroupNorm statistics,
a global attention): a tile is a picture of its own, and tiles are blended where they overlap."""
import torch


def starts(length, size):
    """First positions of tiles of `size` along `length`, a quarter of each shared with the next; the last one ends at the edge."""
    stride = size * 3 // 4
    positions = list(range(0, max(length - size, 0) + 1, stride))
    if positions[-1] + size < length:
        positions.append(length - size)
    return positions


def count(height, width, size):
    """Number of tiles of `size` (0: no tiling) for a height x width input."""
    return len(starts(height, size)) * len(starts(width, size)) if size and (height > size or width > size) else 1


def halo_plan(steps, start, stop, length):
    """Plan along one spatial axis of an exact tile through a chain of local layers (no overlap, no blending): for the output range
    [start, stop) of the chain, the range of its input (axis `length` there) that the tile needs, and for each step the crop of its
    output that the later steps still need. Every step runs on a region wider than its share of the tile by the receptive field
    still ahead (its halo); a region clamped at an edge of the frame gets the layers' own padding there. steps, in forward order:
    ("same", reach): an output reads the inputs within `reach` of it (2 for a residual block of two 3-tap convolutions, 1 for a
    convolution); ("nearest2",): nearest x2 then a 3-tap convolution (output 2m reads inputs m - 1 and m, 2m + 1 reads m and m + 1);
    ("shuffle2",): a 3-tap convolution then a x2 pixel shuffle (outputs 2m and 2m + 1 read inputs m - 1 to m + 1); ("unshuffle2",):
    a 3-tap convolution then a x2 pixel unshuffle (output m reads inputs 2m - 1 to 2m + 2; its region starts and ends on even
    positions, as the pairs do); ("expand", factor): a pixel shuffle without a convolution (outputs factor * m to factor * m + factor - 1
    read input m alone).
    Returns ((low, high), crops), crops[i] = (first, last + 1) of step i's output."""
    lengths = [length]
    for step in steps:
        factor = step[1] if step[0] == "expand" else 1 if step[0] == "same" else 2
        lengths.append(lengths[-1] // 2 if step[0] == "unshuffle2" else lengths[-1] * factor)
    low, high, crops = start, stop, []
    for step, size in zip(reversed(steps), reversed(lengths[:-1])):
        needed_low, needed_high = low, high
        if step[0] == "same":
            low, high = max(needed_low - step[1], 0), min(needed_high + step[1], size)
            crops.append((needed_low - low, needed_high - low))
            continue
        if step[0] == "expand":
            factor = step[1]
            low, high = needed_low // factor, min(-(-needed_high // factor), size)
            crops.append((needed_low - factor * low, needed_high - factor * low))
            continue
        if step[0] == "unshuffle2":
            low, high = max(2 * needed_low - 2, 0), min(2 * needed_high + 2, size)
            crops.append((needed_low - low // 2, needed_high - low // 2))
            continue
        if step[0] == "nearest2":
            low, high = max((needed_low - 1) // 2, 0), min(needed_high // 2 + 1, size)
        else:
            low, high = max(needed_low // 2 - 1, 0), min((needed_high + 1) // 2 + 1, size)
        crops.append((needed_low - 2 * low, needed_high - 2 * low))
    return (low, high), crops[::-1]


def tiled(fn, x, size, factor):
    """fn(x) for x (batch, channels, height, width) by tiles of size x size (the whole of x when it fits one, or size is 0): fn runs on
    each tile, whose output is `factor` times as large (8 for the decoder of an 8x VAE, 1 / 8 for its encoder), and outputs are blended
    with linear ramps over the parts tiles share. The result is on fn's output device, in its dtype. x is borrowed."""
    height, width = x.shape[-2:]
    if count(height, width, size) == 1:
        return fn(x)
    overlap = max(1, int((size - size * 3 // 4) * factor))
    out = weights = None
    for row in starts(height, size):
        for col in starts(width, size):
            tile = fn(x[..., row:row + size, col:col + size])
            if out is None:
                out = torch.zeros((tile.shape[0], tile.shape[1], int(height * factor), int(width * factor)), dtype=torch.float32, device=tile.device)
                weights = torch.zeros((1, 1, *out.shape[-2:]), dtype=torch.float32, device=tile.device)
                dtype = tile.dtype
            h, w = tile.shape[-2:]
            ramp_h, ramp_w = torch.ones(h, device=tile.device), torch.ones(w, device=tile.device)
            if row:
                ramp_h[:overlap] *= torch.linspace(0, 1, overlap, device=tile.device)
            if row + size < height:
                ramp_h[-overlap:] *= torch.linspace(1, 0, overlap, device=tile.device)
            if col:
                ramp_w[:overlap] *= torch.linspace(0, 1, overlap, device=tile.device)
            if col + size < width:
                ramp_w[-overlap:] *= torch.linspace(1, 0, overlap, device=tile.device)
            weight = ramp_h[:, None] * ramp_w[None, :]
            y, z = int(row * factor), int(col * factor)
            out[..., y:y + h, z:z + w].add_(tile.float().mul_(weight))
            weights[..., y:y + h, z:z + w].add_(weight)
            del tile
    return out.div_(weights).to(dtype)
