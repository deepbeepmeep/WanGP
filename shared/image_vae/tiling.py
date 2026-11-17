# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""VAE tiling chosen from a VRAM budget.

The VAE Tiling setting (wgp_config "vae_config") is the VRAM that VAE encoding and decoding may use, the VAE's weights included:
- Auto (0, the default): what the GPU has left for WanGP when the generation starts, the VRAM of other programs deducted, less a
  safety margin (a tenth of it, 512 MB at least: fragmentation, what the estimates miss). The most VRAM, so the fastest, but the
  tiles follow whatever else runs: not the same from one day to the next.
- a fixed share of the card, 3/4 (1), 1/2 (2) or 1/3 (3): the same tiles on a given GPU whatever else runs (other programs are not
  looked at: choose a smaller share when they need VRAM).
The card is the one WanGP sees (the emulated one with --max-vram / --vram-stress). A VAE declares what it was measured to need
(COSTS below its class): choose_tile() estimates each tile size for the output's resolution and returns the largest that fits, or
0, no tiling, when the whole output fits.

Costs: {"weights": GiB, "decode": points, "encode": points}, points = [(megapixels, GiB above the weights), ...] in increasing
order, measured as peaks of allocated VRAM on tiles of that many pixels (or on whole outputs: a tile and a frame of the same size
cost the same), on the route that needs the most (with and without the Triton convolutions), plus 10 %. A video VAE's points are per
frame size: its chunks of frames have a fixed length, and decoded frames leave the GPU as they are produced."""
import torch
from mmgp import allocator

from shared.cuda_memory import mmgp_vram_allocator

GIB = 1 << 30
SHARES = {1: 3 / 4, 2: 1 / 2, 3: 1 / 3}  # any other value: Auto
AUTO_MARGIN, AUTO_MARGIN_SHARE = 512 << 20, 0.1  # Auto keeps the larger of both out of what is available


def available_vram(card):
    """Bytes of the card (its size in bytes, as WanGP sees it) that other programs leave to this process now: what they hold is
    deducted (MMGP VRAM allocator), else what is free plus what this process already reserved."""
    others = allocator.other_programs() if mmgp_vram_allocator() else None
    if others is None and torch.cuda.is_available():
        return min(card, torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved())
    return max(0, card - (others or 0))


def vram_budget(vae_config, device_mem_capacity):
    """Bytes of VRAM a VAE may use, weights included, for the VAE Tiling setting on a card of device_mem_capacity MB (the emulated
    card's with --max-vram): a fixed share of the card, or with Auto what other programs leave of it now less the safety margin."""
    card = device_mem_capacity * 2**20
    if vae_config in SHARES:
        return SHARES[vae_config] * card
    available = available_vram(card)
    return max(0, available - max(AUTO_MARGIN, AUTO_MARGIN_SHARE * available))


def estimate(points, pixels):
    """Bytes of VRAM for `pixels` from measured points [(megapixels, GiB), ...]: interpolated between two points, proportional to the
    pixels below the first and beyond the last."""
    megapixels = pixels / 1e6
    if megapixels <= points[0][0] or megapixels >= points[-1][0]:
        x, y = points[0] if megapixels <= points[0][0] else points[-1]
        return y * megapixels / x * GIB
    for (xa, ya), (xb, yb) in zip(points, points[1:]):
        if megapixels <= xb:
            return (ya + (yb - ya) * (megapixels - xa) / (xb - xa)) * GIB


def choose_tile(vae_config, device_mem_capacity, costs, height=None, width=None, tiles=(1024, 512, 256)):
    """The tile size in output pixels for a VAE of measured `costs`: 0 (no tiling) when the whole height x width output fits the
    VRAM budget of the setting, else the largest of `tiles` (decreasing) that does, the smallest when none does (RAM spilling then
    absorbs the excess). Unknown height or width: always a tile. Without the MMGP VRAM allocator (AMD GPUs, the PyTorch allocator
    chosen on the command line), one size smaller: PyTorch's allocator fragments, and nothing spills."""
    budget = vram_budget(vae_config, device_mem_capacity) - costs["weights"] * GIB
    area = height * width if height and width else None
    candidates = ([0] if area else []) + [tile for tile in tiles if area is None or tile * tile < area]
    need = lambda tile: max(estimate(costs[stage], area if tile == 0 else tile * tile) for stage in ("decode", "encode") if stage in costs)
    index = next((i for i, tile in enumerate(candidates) if need(tile) <= budget), len(candidates) - 1)
    if not mmgp_vram_allocator():
        index = min(index + 1, len(candidates) - 1)
    return candidates[index]


def largest_tile(vae_config, device_mem_capacity, costs, height=None, width=None, step=32, minimum=128, maximum=2048):
    """For a VAE whose tiles take any size (regions of tile x tile pixels of area, the points of `costs` at regions that fill them):
    0 (no tiling) when the whole height x width output fits the VRAM budget of the setting, else the side of the largest tile that
    does, a multiple of `step` (the budget is used, not the nearest preset below it: fewer regions, less time); `minimum` when
    nothing fits. Unknown height or width: always a tile, `maximum` at most. Without the MMGP VRAM allocator (AMD GPUs, the PyTorch
    allocator chosen on the command line), half the room above the weights: PyTorch's allocator fragments, and nothing spills."""
    room = vram_budget(vae_config, device_mem_capacity) - costs["weights"] * GIB
    if not mmgp_vram_allocator():
        room /= 2
    area = height * width if height and width else None
    need = lambda pixels: max(estimate(costs[stage], pixels) for stage in ("decode", "encode") if stage in costs)
    if area and need(area) <= room:
        return 0
    fits = [side for side in range(minimum, maximum + 1, step) if (area is None or side * side < area) and need(side * side) <= room]
    return fits[-1] if fits else minimum
