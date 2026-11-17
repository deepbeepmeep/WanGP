# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import torch
from models.flux.modules.autoencoder_flux2 import AutoencoderKLFlux2, AutoEncoderParamsFlux2
from shared.image_vae import tiling
from shared.utils.phase_progress import vae_decoding_progress, vae_encoding_progress


class Flux3VAE(AutoencoderKLFlux2):
    def __init__(self):
        super().__init__(AutoEncoderParamsFlux2())

    # measured VRAM (shared.image_vae.tiling): BF16 autocast, the larger of the routes with and without the Triton convolutions, + 10 %
    COSTS = {"weights": 0.16, "decode": [(1.05, 0.75), (4.2, 2.8)], "encode": [(1.05, 0.63), (4.2, 2.3)]}

    @staticmethod
    def get_VAE_tile_size(vae_config, device_mem_capacity, mixed_precision, output_height=None, output_width=None):
        """(tiling, tile size in pixels): the largest tile within the VAE Tiling setting's share of the VRAM, no tiling when the whole
        picture fits (tiling.choose_tile)."""
        # (tiles of a GroupNorm VAE each take their own statistics: 4 levels off the untiled picture on average at 2048 and 1024, more below)
        tile = tiling.choose_tile(vae_config, device_mem_capacity, Flux3VAE.COSTS, output_height, output_width, (2048, 1024, 512))
        return tile > 0, tile

    def forward(self, x, encode=False, tile_size=1024):
        # Sizes are sample pixels; normalized Flux latents are downsampled by 16.
        if encode:
            with vae_encoding_progress(1, self.encoder, enabled=False):
                return self._tiles(x, tile_size, True)
        return self._tiles(x, tile_size // 16 if tile_size else 0, False)

    def _tiles(self, x, size, encode):
        height, width = x.shape[-2:]
        size = size or max(height, width)
        stride = size * 3 // 4
        rows = list(range(0, max(height - size, 0) + 1, stride))
        cols = list(range(0, max(width - size, 0) + 1, stride))
        if rows[-1] + size < height:
            rows.append(height - size)
        if cols[-1] + size < width:
            cols.append(width - size)
        factor = 1 / 16 if encode else 16
        out_h, out_w = int(height * factor), int(width * factor)
        # Decoding keeps only one tile row in floating point. Completed rows go
        # straight to the final byte image; reference latents remain floating.
        buffer_h = out_h if encode else min(size, height) * 16
        result = torch.zeros((x.shape[0], 128 if encode else 3, buffer_h, out_w), device='cpu', dtype=torch.float32)
        weights = torch.zeros((1, 1, buffer_h, out_w), device='cpu', dtype=torch.float32)
        if not encode:
            pixels = torch.empty((x.shape[0], 3, out_h, out_w), device='cpu', dtype=torch.uint8)
        from contextlib import nullcontext
        progress = nullcontext() if encode else vae_decoding_progress(len(rows) * len(cols), self.decoder)
        with progress:
            for row_index, row in enumerate(rows):
                for col in cols:
                    tile = x[..., row:row + size, col:col + size]
                    if encode:
                        tile = super().encode(tile)
                    else:
                        tile = self.decoder(self.pre_decode(tile))
                    tile = tile.float().cpu()
                    h, w = tile.shape[-2:]
                    ramp_h = torch.ones(h, device='cpu')
                    ramp_w = torch.ones(w, device='cpu')
                    overlap = max(1, int((size - stride) * factor))
                    if row:
                        ramp_h[:overlap] *= torch.linspace(0, 1, overlap, device='cpu')
                    if row + size < height:
                        ramp_h[-overlap:] *= torch.linspace(1, 0, overlap, device='cpu')
                    if col:
                        ramp_w[:overlap] *= torch.linspace(0, 1, overlap, device='cpu')
                    if col + size < width:
                        ramp_w[-overlap:] *= torch.linspace(1, 0, overlap, device='cpu')
                    weight = ramp_h[:, None] * ramp_w[None, :]
                    y, z = int(row * factor) if encode else 0, int(col * factor)
                    result[..., y:y+h, z:z+w].add_(tile * weight)
                    weights[..., y:y+h, z:z+w].add_(weight)
                if not encode:
                    y = row * 16
                    stop = rows[row_index + 1] * 16 if row_index + 1 < len(rows) else out_h
                    count = stop - y
                    ready = result[..., :count, :].div_(weights[..., :count, :])
                    pixels[..., y:stop, :].copy_(ready.add_(1).mul_(127.5).clamp_(0, 255))
                    remaining = buffer_h - count
                    if remaining:
                        result[..., :remaining, :].copy_(result[..., count:, :].clone())
                        weights[..., :remaining, :].copy_(weights[..., count:, :].clone())
                    result[..., remaining:, :].zero_()
                    weights[..., remaining:, :].zero_()
        if not encode:
            return pixels
        result.div_(weights)
        return result.to(x.device, dtype=x.dtype)
