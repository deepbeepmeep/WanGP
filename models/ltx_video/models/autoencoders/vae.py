from shared.utils.phase_progress import vae_encoding_progress, vae_decoding_progress, set_phase_status
from shared.image_vae import tiling
from shared.image_vae.output import blend_edge_
from mmgp import transfer
from typing import Optional, Union

import torch
import inspect
import math
import torch.nn as nn
from diffusers import ConfigMixin, ModelMixin
from diffusers.models.autoencoders.vae import (
    DecoderOutput,
    DiagonalGaussianDistribution,
)
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from ...models.autoencoders.conv_nd_factory import make_conv_nd


class AutoencoderKLWrapper(ModelMixin, ConfigMixin):
    """Variational Autoencoder (VAE) model with KL loss.

    VAE from the paper Auto-Encoding Variational Bayes by Diederik P. Kingma and Max Welling.
    This model is a wrapper around an encoder and a decoder, and it adds a KL loss term to the reconstruction loss.

    Args:
        encoder (`nn.Module`):
            Encoder module.
        decoder (`nn.Module`):
            Decoder module.
        latent_channels (`int`, *optional*, defaults to 4):
            Number of latent channels.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        latent_channels: int = 4,
        dims: int = 2,
        sample_size=512,
        use_quant_conv: bool = True,
        normalize_latent_channels: bool = False,
    ):
        super().__init__()


        self.per_channel_statistics = nn.Module()
        std_of_means = torch.zeros( (128,), dtype= torch.bfloat16)

        # self.per_channel_statistics.register_buffer("std-of-means", std_of_means)
        # self.per_channel_statistics.register_buffer(
        #     "mean-of-means",
        #         torch.zeros_like(std_of_means)
        # )
        
        self.register_buffer("std_of_means", std_of_means)
        self.register_buffer(
            "mean_of_means",
                torch.zeros_like(std_of_means)
        )


        # pass init params to Encoder
        self.encoder = encoder
        self.use_quant_conv = use_quant_conv
        self.normalize_latent_channels = normalize_latent_channels

        # pass init params to Decoder
        quant_dims = 2 if dims == 2 else 3
        self.decoder = decoder
        if use_quant_conv:
            self.quant_conv = make_conv_nd(
                quant_dims, 2 * latent_channels, 2 * latent_channels, 1
            )
            self.post_quant_conv = make_conv_nd(
                quant_dims, latent_channels, latent_channels, 1
            )
        else:
            self.quant_conv = nn.Identity()
            self.post_quant_conv = nn.Identity()

        if normalize_latent_channels:
            if dims == 2:
                self.latent_norm_out = nn.BatchNorm2d(latent_channels, affine=False)
            else:
                self.latent_norm_out = nn.BatchNorm3d(latent_channels, affine=False)
        else:
            self.latent_norm_out = nn.Identity()
        self.use_z_tiling = False
        self.use_hw_tiling = False
        self.dims = dims
        self.z_sample_size = 1

        self.decoder_params = inspect.signature(self.decoder.forward).parameters

        # only relevant if vae tiling is enabled
        self.set_tiling_params(sample_size=sample_size, overlap_factor=0.25)

    # measured VRAM (shared.image_vae.tiling), BF16, 97 frames by temporal tiles of 4 latent frames, + 10 %: tiles of 256 and 512
    # pixels, then whole frames of 832x480, 1280x704 and 1920x1088 (3.5 GiB per megapixel, 2.3 times faster than tiles of 512 and
    # 4.6 times faster than tiles of 256). A tiled decode also holds the assembled video (1.2 GiB at 1920x1088 x 97).
    COSTS = {"weights": 2.32, "decode": [(0.0655, 0.67), (0.262, 1.12), (0.4, 1.55), (0.9, 3.5), (2.09, 8.1)], "encode": [(0.0655, 0.27), (0.262, 1.03), (0.4, 1.55), (0.9, 3.5), (2.09, 8.1)]}

    @staticmethod
    def get_VAE_tile_size(vae_config, device_mem_capacity, mixed_precision, output_height=None, output_width=None):
        """(latent frames per temporal tile, tile size in pixels): no spatial tiling (0) whenever the whole frames fit the VAE Tiling
        setting's VRAM budget, else the largest tile that does."""
        return 4, tiling.choose_tile(vae_config, device_mem_capacity, AutoencoderKLWrapper.COSTS, output_height, output_width, (512, 256))

    def set_tiling_params(self, sample_size: int = 512, overlap_factor: float = 0.25):
        self.tile_sample_min_size = sample_size
        num_blocks = len(self.encoder.down_blocks)
        # self.tile_latent_min_size = int(sample_size / (2 ** (num_blocks - 1)))
        self.tile_latent_min_size = int(sample_size / 32)
        self.tile_overlap_factor = overlap_factor

    def enable_z_tiling(self, z_sample_size: int = 4):
        r"""
        Enable tiling during VAE decoding.

        When this option is enabled, the VAE will split the input tensor in tiles to compute decoding in several
        steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.use_z_tiling = z_sample_size > 1
        self.z_sample_size = z_sample_size
        assert (
            z_sample_size % 4 == 0 or z_sample_size == 1
        ), f"z_sample_size must be a multiple of 4 or 1. Got {z_sample_size}."

    def disable_z_tiling(self):
        r"""
        Disable tiling during VAE decoding. If `use_tiling` was previously invoked, this method will go back to computing
        decoding in one step.
        """
        self.use_z_tiling = False

    def enable_hw_tiling(self):
        r"""
        Enable tiling during VAE decoding along the height and width dimension.
        """
        self.use_hw_tiling = True

    def disable_hw_tiling(self):
        r"""
        Disable tiling during VAE decoding along the height and width dimension.
        """
        self.use_hw_tiling = False

    def _hw_tiled_encode(self, x: torch.FloatTensor, return_dict: bool = True):
        overlap_size = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_latent_min_size * self.tile_overlap_factor)
        row_limit = self.tile_latent_min_size - blend_extent

        # Split the image into 512x512 tiles and encode them separately.
        rows = []
        for i in range(0, x.shape[3], overlap_size):
            row = []
            for j in range(0, x.shape[4], overlap_size):
                tile = x[
                    :,
                    :,
                    :,
                    i : i + self.tile_sample_min_size,
                    j : j + self.tile_sample_min_size,
                ]
                tile = self.encoder(tile)
                tile = self.quant_conv(tile)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=4))

        moments = torch.cat(result_rows, dim=3)
        return moments

    def blend_z(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[2], b.shape[2], blend_extent)
        for z in range(blend_extent):
            b[:, :, z, :, :] = a[:, :, -blend_extent + z, :, :] * (
                1 - z / blend_extent
            ) + b[:, :, z, :, :] * (z / blend_extent)
        return b

    def blend_v(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[3], b.shape[3], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (
                1 - y / blend_extent
            ) + b[:, :, :, y, :] * (y / blend_extent)
        return b

    def blend_h(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[4], b.shape[4], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (
                1 - x / blend_extent
            ) + b[:, :, :, :, x] * (x / blend_extent)
        return b

    def _hw_tiled_decode(self, z: torch.FloatTensor, target_shape, timestep = None):
        overlap_size = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
        blend_extent = int(self.tile_sample_min_size * self.tile_overlap_factor)
        row_limit = self.tile_sample_min_size - blend_extent
        tile_target_shape = (
            *target_shape[:3],
            self.tile_sample_min_size,
            self.tile_sample_min_size,
        )
        # Split z into overlapping 64x64 tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, z.shape[3], overlap_size):
            row = []
            for j in range(0, z.shape[4], overlap_size):
                tile = z[
                    :,
                    :,
                    :,
                    i : i + self.tile_latent_min_size,
                    j : j + self.tile_latent_min_size,
                ]
                tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile, target_shape=tile_target_shape, timestep = timestep)
                row.append(decoded)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=4))

        dec = torch.cat(result_rows, dim=3)
        return dec

    def encode(
        self, z: torch.FloatTensor, return_dict: bool = True
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        set_phase_status("VAE Encoding")
        temporal_tiling = self.use_z_tiling and z.shape[2] > (self.z_sample_size + 1) > 1
        tiles = len(range(0, z.shape[2], int(self.z_sample_size * 8 * 0.75))) if temporal_tiling else 1
        if self.use_hw_tiling and (temporal_tiling or z.shape[2] > 1):
            stride = int(self.tile_sample_min_size * (1 - self.tile_overlap_factor))
            tiles *= len(range(0, z.shape[-2], stride)) * len(range(0, z.shape[-1], stride))
        with vae_encoding_progress(tiles, self.encoder, enabled=z.shape[2] > 1):
            if self.use_z_tiling and z.shape[2] > (self.z_sample_size + 1) > 1:
                tile_latent_min_tsize = self.z_sample_size
                tile_sample_min_tsize = tile_latent_min_tsize *  8
                tile_overlap_factor = 0.25

                B, C, T, H, W = z.shape
                overlap_size = int(tile_sample_min_tsize * (1 - tile_overlap_factor))
                blend_extent = int(tile_latent_min_tsize * tile_overlap_factor)
                t_limit = tile_latent_min_tsize - blend_extent

                row = []
                for i in range(0, T, overlap_size):
                    tile = z[:, :, i: i + tile_sample_min_tsize + 1, :, :]
                    if self.use_hw_tiling:
                        tile = self._hw_tiled_encode(tile, return_dict)
                    else:
                        tile = self._encode(tile)
                    if i > 0:
                        tile = tile[:, :, 1:, :, :]
                    row.append(tile)
                result_row = []
                for i, tile in enumerate(row):
                    if i > 0:
                        tile = self.blend_z(row[i - 1], tile, blend_extent)
                        result_row.append(tile[:, :, :t_limit, :, :])
                    else:
                        result_row.append(tile[:, :, :t_limit + 1, :, :])

                moments = torch.cat(result_row, dim=2)


            else:
                moments = (
                    self._hw_tiled_encode(z, return_dict)
                    if self.use_hw_tiling and z.shape[2] > 1
                    else self._encode(z)
                )

            posterior = DiagonalGaussianDistribution(moments)
            if not return_dict:
                return (posterior,)

            return AutoencoderKLOutput(latent_dist=posterior)

    def _normalize_latent_channels(self, z: torch.FloatTensor) -> torch.FloatTensor:
        if isinstance(self.latent_norm_out, nn.BatchNorm3d):
            _, c, _, _, _ = z.shape
            z = torch.cat(
                [
                    self.latent_norm_out(z[:, : c // 2, :, :, :]),
                    z[:, c // 2 :, :, :, :],
                ],
                dim=1,
            )
        elif isinstance(self.latent_norm_out, nn.BatchNorm2d):
            raise NotImplementedError("BatchNorm2d not supported")
        return z

    def _unnormalize_latent_channels(self, z: torch.FloatTensor) -> torch.FloatTensor:
        if isinstance(self.latent_norm_out, nn.BatchNorm3d):
            running_mean = self.latent_norm_out.running_mean.view(1, -1, 1, 1, 1)
            running_var = self.latent_norm_out.running_var.view(1, -1, 1, 1, 1)
            eps = self.latent_norm_out.eps

            z = z * torch.sqrt(running_var + eps) + running_mean
        elif isinstance(self.latent_norm_out, nn.BatchNorm3d):
            raise NotImplementedError("BatchNorm2d not supported")
        return z

    def _encode(self, x: torch.FloatTensor) -> AutoencoderKLOutput:
        h = self.encoder(x)
        moments = self.quant_conv(h)
        moments = self._normalize_latent_channels(moments)
        return moments

    def _decode(
        self,
        z: torch.FloatTensor,
        target_shape=None,
        timestep: Optional[torch.Tensor] = None,
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        z = self._unnormalize_latent_channels(z)
        z = self.post_quant_conv(z)
        if "timestep" in self.decoder_params:
            dec = self.decoder(z, target_shape=target_shape, timestep=timestep)
        else:
            dec = self.decoder(z, target_shape=target_shape)
        return dec

    def _decode_to_ram(self, z, target_shape, timestep, to_ram):
        """The video decode() gives, in RAM in its final form (to_ram: a function of a decoded piece on the GPU, a dtype conversion or
        bytes): with temporal tiles, each tile's frames once blended with the tail frames kept from the tile before it (on the GPU,
        in FP16 as decode() blends them in RAM) go through to_ram and are written to their place without waiting for the GPU. RAM
        holds the video in its final form only (decode() keeps every decoded tile in RAM in FP16, blends them there with PyTorch's
        CPU operations, then concatenates them)."""
        if not (self.use_z_tiling and z.shape[2] > (self.z_sample_size + 1) > 1):
            decoded = self._hw_tiled_decode(z, target_shape, timestep) if self.use_hw_tiling else self._decode(z, target_shape=target_shape, timestep=timestep)
            return to_ram(decoded).cpu()
        latent_frames, frames_per_latent = self.z_sample_size, 8
        stride = int(latent_frames * 0.75)
        extent = int(latent_frames * frames_per_latent * 0.25)
        limit = latent_frames * frames_per_latent - extent
        starts = list(range(0, z.shape[2], stride))
        lengths = [(min(start + latent_frames + 1, z.shape[2]) - start - 1) * frames_per_latent + (1 if index == 0 else 0) for index, start in enumerate(starts)]
        kept = [min(length, limit + (1 if index == 0 else 0)) for index, length in enumerate(lengths)]
        out, tail, position = None, None, 0
        with transfer.RamWriter("decoded video", depth=64, max_bytes=256 << 20) as writer:
            for index, start in enumerate(starts):
                tile = z[:, :, start : start + latent_frames + 1]
                decoded = self._hw_tiled_decode(tile, target_shape, timestep) if self.use_hw_tiling else self._decode(tile, target_shape=target_shape, timestep=timestep)
                decoded = decoded.to(torch.float16)
                if index > 0:
                    decoded = decoded[:, :, 1:]
                    e = min(tail.shape[2], decoded.shape[2], extent)
                    blend_edge_(decoded[:, :, :e], tail[:, :, tail.shape[2] - e :], 2)
                tail = decoded[:, :, decoded.shape[2] - min(extent, decoded.shape[2]) :].clone()
                if kept[index]:  # (a last tile of one latent frame after the first adds no frame)
                    piece = to_ram(decoded[:, :, : kept[index]])
                    if out is None:
                        out = torch.empty((*piece.shape[:2], sum(kept), *piece.shape[3:]), dtype=piece.dtype, device="cpu")
                    writer.write(out[:, :, position : position + piece.shape[2]], piece)
                    position += piece.shape[2]
                    del piece
                del decoded
        return out

    def decode(
        self,
        z: torch.FloatTensor,
        return_dict: bool = True,
        target_shape=None,
        timestep: Optional[torch.Tensor] = None,
        to_ram=None,
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        """to_ram: a function of a decoded piece on the GPU (its final form); the video is then returned in RAM, written piece by piece
        as it is decoded (_decode_to_ram)."""
        tiles = 1
        if self.use_z_tiling and z.shape[2] > (self.z_sample_size + 1) > 1:
            tiles *= len(range(0, z.shape[2], int(self.z_sample_size * 0.75)))
        if self.use_hw_tiling:
            stride = int(self.tile_latent_min_size * (1 - self.tile_overlap_factor))
            tiles *= len(range(0, z.shape[-2], stride)) * len(range(0, z.shape[-1], stride))
        with vae_decoding_progress(tiles, self.decoder):
            assert target_shape is not None, "target_shape must be provided for decoding"
            if to_ram is not None:
                decoded = self._decode_to_ram(z, target_shape, timestep, to_ram)
                return (decoded,) if not return_dict else DecoderOutput(sample=decoded)
            if self.use_z_tiling and z.shape[2] > (self.z_sample_size + 1) > 1:
            # Split z into overlapping tiles and decode them separately.
                tile_latent_min_tsize = self.z_sample_size
                tile_sample_min_tsize = tile_latent_min_tsize *  8
                tile_overlap_factor = 0.25

                B, C, T, H, W = z.shape
                overlap_size = int(tile_latent_min_tsize * (1 - tile_overlap_factor))
                blend_extent = int(tile_sample_min_tsize * tile_overlap_factor)
                t_limit = tile_sample_min_tsize - blend_extent

                row = []
                for i in range(0, T, overlap_size):
                    tile = z[:, :, i: i + tile_latent_min_tsize + 1, :, :]
                    target_shape_split = list(target_shape)
                    target_shape_split[2] = tile.shape[2] * 8
                    if self.use_hw_tiling:
                        decoded = self._hw_tiled_decode(tile, target_shape, timestep)
                    else:
                        decoded = self._decode(tile, target_shape=target_shape, timestep=timestep)

                    if i > 0:
                        decoded = decoded[:, :, 1:, :, :]
                    row.append(decoded.to(torch.float16).cpu())
                    decoded = None
                result_row = []
                for i, tile in enumerate(row):
                    if i > 0:
                        tile = self.blend_z(row[i - 1], tile, blend_extent)
                        result_row.append(tile[:, :, :t_limit, :, :])
                    else:
                        result_row.append(tile[:, :, :t_limit + 1, :, :])

                dec = torch.cat(result_row, dim=2)
                if not return_dict:
                    return (dec,)

                return DecoderOutput(sample=dec)
            else:
                decoded = (
                    self._hw_tiled_decode(z, target_shape, timestep)
                    if self.use_hw_tiling
                    else self._decode(z, target_shape=target_shape, timestep=timestep)
                )

            if not return_dict:
                return (decoded,)

            return DecoderOutput(sample=decoded)

    def forward(
        self,
        sample: torch.FloatTensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.FloatTensor]:
        r"""
        Args:
            sample (`torch.FloatTensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`DecoderOutput`] instead of a plain tuple.
            generator (`torch.Generator`, *optional*):
                Generator used to sample from the posterior.
        """
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z, target_shape=sample.shape).sample

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)
