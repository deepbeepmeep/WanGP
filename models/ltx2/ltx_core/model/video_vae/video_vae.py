from dataclasses import replace
import itertools
from typing import Any, Callable, Iterator, List, Optional, Tuple

import torch
from mmgp import transfer
from shared.utils.phase_progress import vae_encoding_progress, PhaseProgress, check_abort
from einops import rearrange
from torch import nn

from ..common.normalization import PixelNorm
from ..transformer.timestep_embedding import PixArtAlphaCombinedTimestepSizeEmbeddings
from .convolution import make_conv_nd
from .streaming import DecoderStream, EncoderStream
from .enums import LogVarianceType, NormLayerType, PaddingModeType
from .ops import PerChannelStatistics, patchify, unpatchify
from .resnet import ResnetBlock3D, UNetMidBlock3D
from .sampling import DepthToSpaceUpsample, SpaceToDepthDownsample
from .tiling import (
    DEFAULT_MAPPING_OPERATION,
    DEFAULT_SPLIT_OPERATION,
    DimensionIntervals,
    MappingOperation,
    SplitOperation,
    Tile,
    TilingConfig,
    compute_trapezoidal_mask_1d,
    create_tiles,
)
from ...types import SpatioTemporalScaleFactors, VideoLatentShape, VideoPixelShape, VIDEO_SCALE_FACTORS
from shared.utils.hdr import vae_range_to_hdr_linear


def _make_encoder_block(
    block_name: str,
    block_config: dict[str, Any],
    in_channels: int,
    convolution_dimensions: int,
    norm_layer: NormLayerType,
    norm_num_groups: int,
    spatial_padding_mode: PaddingModeType,
) -> Tuple[nn.Module, int]:
    out_channels = in_channels

    if block_name == "res_x":
        block = UNetMidBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            num_layers=block_config["num_layers"],
            resnet_eps=1e-6,
            resnet_groups=norm_num_groups,
            norm_layer=norm_layer,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "res_x_y":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = ResnetBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            eps=1e-6,
            groups=norm_num_groups,
            norm_layer=norm_layer,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_time":
        block = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(2, 1, 1),
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_space":
        block = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(1, 2, 2),
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all":
        block = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(2, 2, 2),
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all_x_y":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=(2, 2, 2),
            causal=True,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(2, 2, 2),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_space_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(1, 2, 2),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_time_res":
        out_channels = in_channels * block_config.get("multiplier", 2)
        block = SpaceToDepthDownsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=(2, 1, 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    else:
        raise ValueError(f"unknown block: {block_name}")

    return block, out_channels


class VideoEncoder(nn.Module):
    _DEFAULT_NORM_NUM_GROUPS = 32
    """
    Variational Autoencoder Encoder. Encodes video frames into a latent representation.
    The encoder compresses the input video through a series of downsampling operations controlled by
    patch_size and encoder_blocks. The output is a normalized latent tensor with shape (B, 128, F', H', W').
    Compression Behavior:
        The total compression is determined by:
        1. Initial spatial compression via patchify: H -> H/4, W -> W/4 (patch_size=4)
        2. Sequential compression through encoder_blocks based on their stride patterns
        Compression blocks apply 2x compression in specified dimensions:
            - "compress_time" / "compress_time_res": temporal only
            - "compress_space" / "compress_space_res": spatial only (H and W)
            - "compress_all" / "compress_all_res": all dimensions (F, H, W)
            - "res_x" / "res_x_y": no compression
        Standard LTX Video configuration:
            - patch_size=4
            - encoder_blocks: 1x compress_space_res, 1x compress_time_res, 2x compress_all_res
            - Final dimensions: F' = 1 + (F-1)/8, H' = H/32, W' = W/32
            - Example: (B, 3, 33, 512, 512) -> (B, 128, 5, 16, 16)
            - Note: Input must have 1 + 8*k frames (e.g., 1, 9, 17, 25, 33...)
    Args:
        convolution_dimensions: The number of dimensions to use in convolutions (2D or 3D).
        in_channels: The number of input channels. For RGB images, this is 3.
        out_channels: The number of output channels (latent channels). For latent channels, this is 128.
        encoder_blocks: The list of blocks to construct the encoder. Each block is a tuple of (block_name, params)
                        where params is either an int (num_layers) or a dict with configuration.
        patch_size: The patch size for initial spatial compression. Should be a power of 2.
        norm_layer: The normalization layer to use. Can be either `group_norm` or `pixel_norm`.
        latent_log_var: The log variance mode. Can be either `per_channel`, `uniform`, `constant` or `none`.
    """

    def __init__(
        self,
        convolution_dimensions: int = 3,
        in_channels: int = 3,
        out_channels: int = 128,
        encoder_blocks: List[Tuple[str, int]] | List[Tuple[str, dict[str, Any]]] = [],  # noqa: B006
        patch_size: int = 4,
        norm_layer: NormLayerType = NormLayerType.PIXEL_NORM,
        latent_log_var: LogVarianceType = LogVarianceType.UNIFORM,
        encoder_spatial_padding_mode: PaddingModeType = PaddingModeType.ZEROS,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.norm_layer = norm_layer
        self.latent_channels = out_channels
        self.latent_log_var = latent_log_var
        self._norm_num_groups = self._DEFAULT_NORM_NUM_GROUPS

        # Per-channel statistics for normalizing latents
        self.per_channel_statistics = PerChannelStatistics(latent_channels=out_channels)

        in_channels = in_channels * patch_size**2
        feature_channels = out_channels

        self.conv_in = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=feature_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=encoder_spatial_padding_mode,
        )

        self.down_blocks = nn.ModuleList([])

        for block_name, block_params in encoder_blocks:
            # Convert int to dict format for uniform handling
            block_config = {"num_layers": block_params} if isinstance(block_params, int) else block_params

            block, feature_channels = _make_encoder_block(
                block_name=block_name,
                block_config=block_config,
                in_channels=feature_channels,
                convolution_dimensions=convolution_dimensions,
                norm_layer=norm_layer,
                norm_num_groups=self._norm_num_groups,
                spatial_padding_mode=encoder_spatial_padding_mode,
            )

            self.down_blocks.append(block)

        # out
        if norm_layer == NormLayerType.GROUP_NORM:
            self.conv_norm_out = nn.GroupNorm(num_channels=feature_channels, num_groups=self._norm_num_groups, eps=1e-6)
        elif norm_layer == NormLayerType.PIXEL_NORM:
            self.conv_norm_out = PixelNorm()

        self.conv_act = nn.SiLU()

        conv_out_channels = out_channels
        if latent_log_var == LogVarianceType.PER_CHANNEL:
            conv_out_channels *= 2
        elif latent_log_var in {LogVarianceType.UNIFORM, LogVarianceType.CONSTANT}:
            conv_out_channels += 1
        elif latent_log_var != LogVarianceType.NONE:
            raise ValueError(f"Invalid latent_log_var: {latent_log_var}")

        self.conv_out = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=feature_channels,
            out_channels=conv_out_channels,
            kernel_size=3,
            padding=1,
            causal=True,
            spatial_padding_mode=encoder_spatial_padding_mode,
        )

    def forward(self, sample: torch.Tensor, stream: EncoderStream | None = None) -> torch.Tensor:
        r"""
        Encode video frames into normalized latent representation.
        Args:
            sample: Input video (B, C, F, H, W). F must be 1 + 8*k (e.g., 1, 9, 17, 25, 33...).
            stream: the frames are the next chunk of this streamed encoding (see encode_video).
        Returns:
            Normalized latent means (B, 128, F', H', W') where F' = 1+(F-1)/8, H' = H/32, W' = W/32.
            Example: (B, 3, 33, 512, 512) -> (B, 128, 5, 16, 16).
        """
        if stream is not None:
            return stream.step(sample)
        # Validate frame count
        frames_count = sample.shape[2]
        if ((frames_count - 1) % 8) != 0:
            raise ValueError(
                "Invalid number of frames: Encode input must have 1 + 8 * x frames "
                "(e.g., 1, 9, 17, ...). Please check your input."
            )

        # Initial spatial compression: trade spatial resolution for channel depth
        # This reduces H,W by patch_size and increases channels, making convolutions more efficient
        # Example: (B, 3, F, 512, 512) -> (B, 48, F, 128, 128) with patch_size=4
        sample = patchify(sample, patch_size_hw=self.patch_size, patch_size_t=1)
        sample = self.conv_in(sample)

        for down_block in self.down_blocks:
            sample = down_block(sample)

        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return self.latent_means(sample)

    def latent_means(self, sample: torch.Tensor) -> torch.Tensor:
        """The normalized latent means of the output of conv_out."""
        if self.latent_log_var == LogVarianceType.UNIFORM:
            # Uniform Variance: model outputs N means and 1 shared log-variance channel.
            # We need to expand the single logvar to match the number of means channels
            # to create a format compatible with PER_CHANNEL (means + logvar, each with N channels).
            # Sample shape: (B, N+1, ...) where N = latent_channels (e.g., 128 means + 1 logvar = 129)
            # Target shape: (B, 2*N, ...) where first N are means, last N are logvar

            if sample.shape[1] < 2:
                raise ValueError(
                    f"Invalid channel count for UNIFORM mode: expected at least 2 channels "
                    f"(N means + 1 logvar), got {sample.shape[1]}"
                )

            # Extract means (first N channels) and logvar (last 1 channel)
            means = sample[:, :-1, ...]  # (B, N, ...)
            logvar = sample[:, -1:, ...]  # (B, 1, ...)

            # Repeat logvar N times to match means channels
            # Use expand/repeat pattern that works for both 4D and 5D tensors
            num_channels = means.shape[1]
            repeat_shape = [1, num_channels] + [1] * (sample.ndim - 2)
            repeated_logvar = logvar.repeat(*repeat_shape)  # (B, N, ...)

            # Concatenate to create (B, 2*N, ...) format: [means, repeated_logvar]
            sample = torch.cat([means, repeated_logvar], dim=1)
        elif self.latent_log_var == LogVarianceType.CONSTANT:
            sample = sample[:, :-1, ...]
            approx_ln_0 = -30  # this is the minimal clamp value in DiagonalGaussianDistribution objects
            sample = torch.cat(
                [sample, torch.ones_like(sample, device=sample.device) * approx_ln_0],
                dim=1,
            )

        # Split into means and logvar, then normalize means
        means, _ = torch.chunk(sample, 2, dim=1)
        return self.per_channel_statistics.normalize(means)


def _make_decoder_block(
    block_name: str,
    block_config: dict[str, Any],
    in_channels: int,
    convolution_dimensions: int,
    norm_layer: NormLayerType,
    timestep_conditioning: bool,
    norm_num_groups: int,
    spatial_padding_mode: PaddingModeType,
) -> Tuple[nn.Module, int]:
    out_channels = in_channels
    if block_name == "res_x":
        block = UNetMidBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            num_layers=block_config["num_layers"],
            resnet_eps=1e-6,
            resnet_groups=norm_num_groups,
            norm_layer=norm_layer,
            inject_noise=block_config.get("inject_noise", False),
            timestep_conditioning=timestep_conditioning,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "attn_res_x":
        block = UNetMidBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            num_layers=block_config["num_layers"],
            resnet_groups=norm_num_groups,
            norm_layer=norm_layer,
            inject_noise=block_config.get("inject_noise", False),
            timestep_conditioning=timestep_conditioning,
            attention_head_dim=block_config["attention_head_dim"],
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "res_x_y":
        out_channels = block_config.get("out_channels", in_channels // block_config.get("multiplier", 2))
        block = ResnetBlock3D(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            eps=1e-6,
            groups=norm_num_groups,
            norm_layer=norm_layer,
            inject_noise=block_config.get("inject_noise", False),
            timestep_conditioning=False,
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_time":
        out_channels = in_channels // block_config.get("multiplier", 1)
        block = DepthToSpaceUpsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            stride=(2, 1, 1),
            out_channels_reduction_factor=block_config.get("multiplier", 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_space":
        out_channels = in_channels // block_config.get("multiplier", 1)
        block = DepthToSpaceUpsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            stride=(1, 2, 2),
            out_channels_reduction_factor=block_config.get("multiplier", 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    elif block_name == "compress_all":
        out_channels = in_channels // block_config.get("multiplier", 1)
        block = DepthToSpaceUpsample(
            dims=convolution_dimensions,
            in_channels=in_channels,
            stride=(2, 2, 2),
            residual=block_config.get("residual", False),
            out_channels_reduction_factor=block_config.get("multiplier", 1),
            spatial_padding_mode=spatial_padding_mode,
        )
    else:
        raise ValueError(f"unknown layer: {block_name}")

    return block, out_channels


# Streamed decoding (tiled_decode) and encoding (encode_video). Latent area of a spatial tile: TILE_AREA squares of the tiling size, plus for
# the decoding FRAME_SHARE of the frame; calibrated so that each tiling size stayed under the VRAM of the former temporal tiles at 480p-1080p
# (in decoding it grew with the frame), then raised for the 256 level (720p decoded as one exact tile, fewer encoding tiles: +0.2-0.5 GB).
# Frames per piece of a decoder stage, per chunk of the encoder.
STREAM_TILE_AREA, STREAM_FRAME_SHARE, STREAM_PIECE_FRAMES = 5, 0.64, 4
ENCODE_TILE_AREA, ENCODE_CHUNK_FRAMES = 7.25, 4


def _stream_tiles(shape: tuple, tiling_config: TilingConfig | None, tile_area: float, frame_share: float, scale: int, device: torch.device) -> List[Tile]:
    """Spatial tiles of a latent of this shape for a streamed decoding or encoding: the fewest whose latent area, with the overlaps, fits
    tile_area squares of the tiling size plus frame_share of the frame; their output coordinates and masks in units of `scale`."""
    splitters, mappers = [DEFAULT_SPLIT_OPERATION] * len(shape), [DEFAULT_MAPPING_OPERATION] * len(shape)
    spatial = tiling_config.spatial_config if tiling_config is not None else None
    if spatial is not None:
        latent_scale = VIDEO_SCALE_FACTORS.height
        overlap, height, width = spatial.tile_overlap_in_pixels // latent_scale, shape[3], shape[4]

        def size(length: int, count: int) -> int:
            return length if count == 1 else -(-(length + (count - 1) * overlap) // count)

        def area(counts: tuple[int, int]) -> int:
            return size(height, counts[0]) * size(width, counts[1])

        budget = tile_area * (spatial.tile_size_in_pixels // latent_scale) ** 2 + frame_share * height * width
        counts = [(h, w) for h in range(1, height + 1) for w in range(1, width + 1) if (h == 1 or size(height, h) > overlap) and (w == 1 or size(width, w) > overlap)]
        fits = [c for c in counts if area(c) <= budget]
        counts = min(fits, key=lambda c: (c[0] * c[1], area(c))) if fits else min(counts, key=area)
        for axis, length, count in ((3, height, counts[0]), (4, width, counts[1])):
            if count > 1:
                splitters[axis] = split_in_spatial(size(length, count), overlap)
                mappers[axis] = to_mapping_operation(map_spatial_slice, scale, device)
    return create_tiles(shape, splitters, mappers)


class VideoDecoder(nn.Module):
    _DEFAULT_NORM_NUM_GROUPS = 32
    """
    Variational Autoencoder Decoder. Decodes latent representation into video frames.
    The decoder upsamples latents through a series of upsampling operations (inverse of encoder).
    Output dimensions: F = 8x(F'-1) + 1, H = 32xH', W = 32xW' for standard LTX Video configuration.
    Upsampling blocks expand dimensions by 2x in specified dimensions:
        - "compress_time": temporal only
        - "compress_space": spatial only (H and W)
        - "compress_all": all dimensions (F, H, W)
        - "res_x" / "res_x_y" / "attn_res_x": no upsampling
    Causal Mode:
        causal=False (standard): Symmetric padding, allows future frame dependencies.
        causal=True: Causal padding, each frame depends only on past/current frames.
        First frame removed after temporal upsampling in both modes. Output shape unchanged.
        Example: (B, 128, 5, 16, 16) -> (B, 3, 33, 512, 512) for both modes.
    Args:
        convolution_dimensions: The number of dimensions to use in convolutions (2D or 3D).
        in_channels: The number of input channels (latent channels). Default is 128.
        out_channels: The number of output channels. For RGB images, this is 3.
        decoder_blocks: The list of blocks to construct the decoder. Each block is a tuple of (block_name, params)
                        where params is either an int (num_layers) or a dict with configuration.
        patch_size: Final spatial expansion factor. For standard LTX Video, use 4 for 4x spatial expansion:
                    H -> Hx4, W -> Wx4. Should be a power of 2.
        norm_layer: The normalization layer to use. Can be either `group_norm` or `pixel_norm`.
        causal: Whether to use causal convolutions. For standard LTX Video, use False for symmetric padding.
                When True, uses causal padding (past/current frames only).
        timestep_conditioning: Whether to condition the decoder on timestep for denoising.
    """

    def __init__(
        self,
        convolution_dimensions: int = 3,
        in_channels: int = 128,
        out_channels: int = 3,
        decoder_blocks: List[Tuple[str, int | dict]] = [],  # noqa: B006
        patch_size: int = 4,
        norm_layer: NormLayerType = NormLayerType.PIXEL_NORM,
        causal: bool = False,
        timestep_conditioning: bool = False,
        decoder_spatial_padding_mode: PaddingModeType = PaddingModeType.REFLECT,
        base_channels: int = 128,
    ):
        super().__init__()

        # Spatiotemporal downscaling between decoded video space and VAE latents.
        # According to the LTXV paper, the standard configuration downsamples
        # video inputs by a factor of 8 in the temporal dimension and 32 in
        # each spatial dimension (height and width). This parameter determines how
        # many video frames and pixels correspond to a single latent cell.
        self.video_downscale_factors = SpatioTemporalScaleFactors(
            time=8,
            width=32,
            height=32,
        )

        self.patch_size = patch_size
        out_channels = out_channels * patch_size**2
        self.causal = causal
        self.timestep_conditioning = timestep_conditioning
        self._norm_num_groups = self._DEFAULT_NORM_NUM_GROUPS

        # Per-channel statistics for denormalizing latents
        self.per_channel_statistics = PerChannelStatistics(latent_channels=in_channels)

        # Noise and timestep parameters for decoder conditioning
        self.decode_noise_scale = 0.025
        self.decode_timestep = 0.05

        # LTX decoder starts from a wider trunk then reduces channels across upsampling blocks.
        feature_channels = base_channels * 8

        self.conv_in = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=in_channels,
            out_channels=feature_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            causal=True,
            spatial_padding_mode=decoder_spatial_padding_mode,
        )

        self.up_blocks = nn.ModuleList([])

        for block_name, block_params in list(reversed(decoder_blocks)):
            # Convert int to dict format for uniform handling
            block_config = {"num_layers": block_params} if isinstance(block_params, int) else block_params

            block, feature_channels = _make_decoder_block(
                block_name=block_name,
                block_config=block_config,
                in_channels=feature_channels,
                convolution_dimensions=convolution_dimensions,
                norm_layer=norm_layer,
                timestep_conditioning=timestep_conditioning,
                norm_num_groups=self._norm_num_groups,
                spatial_padding_mode=decoder_spatial_padding_mode,
            )

            self.up_blocks.append(block)

        if norm_layer == NormLayerType.GROUP_NORM:
            self.conv_norm_out = nn.GroupNorm(num_channels=feature_channels, num_groups=self._norm_num_groups, eps=1e-6)
        elif norm_layer == NormLayerType.PIXEL_NORM:
            self.conv_norm_out = PixelNorm()

        self.conv_act = nn.SiLU()
        self.conv_out = make_conv_nd(
            dims=convolution_dimensions,
            in_channels=feature_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            causal=True,
            spatial_padding_mode=decoder_spatial_padding_mode,
        )

        if timestep_conditioning:
            self.timestep_scale_multiplier = nn.Parameter(torch.tensor(1000.0))
            self.last_time_embedder = PixArtAlphaCombinedTimestepSizeEmbeddings(
                embedding_dim=feature_channels * 2, size_emb_dim=0
            )
            self.last_scale_shift_table = nn.Parameter(torch.empty(2, feature_channels))

    # def forward(self, sample: torch.Tensor, target_shape) -> torch.Tensor:
    def forward(
        self,
        sample: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        r"""
        Decode latent representation into video frames.
        Args:
            sample: Latent tensor (B, 128, F', H', W').
            timestep: Timestep for conditioning (if timestep_conditioning=True). Uses default 0.05 if None.
            generator: Random generator for deterministic noise injection (if inject_noise=True in blocks).
        Returns:
            Decoded video (B, 3, F, H, W) where F = 8x(F'-1) + 1, H = 32xH', W = 32xW'.
            Example: (B, 128, 5, 16, 16) -> (B, 3, 33, 512, 512).
            Note: First frame is removed after temporal upsampling regardless of causal mode.
            When causal=False, allows future frame dependencies in convolutions but maintains same output shape.
        """
        batch_size = sample.shape[0]

        # Add noise if timestep conditioning is enabled
        if self.timestep_conditioning:
            noise = (
                torch.randn(
                    sample.size(),
                    generator=generator,
                    dtype=sample.dtype,
                    device=sample.device,
                )
                * self.decode_noise_scale
            )

            sample = noise + (1.0 - self.decode_noise_scale) * sample

        # Denormalize latents
        sample = self.per_channel_statistics.un_normalize(sample)

        # Use default decode_timestep if timestep not provided
        if timestep is None and self.timestep_conditioning:
            timestep = torch.full((batch_size,), self.decode_timestep, device=sample.device, dtype=sample.dtype)

        sample = self.conv_in(sample, causal=self.causal)

        scaled_timestep = None
        if self.timestep_conditioning:
            if timestep is None:
                raise ValueError("'timestep' parameter must be provided when 'timestep_conditioning' is True")
            scaled_timestep = timestep * self.timestep_scale_multiplier.to(sample)

        for up_block in self.up_blocks:
            check_abort()
            if isinstance(up_block, UNetMidBlock3D):
                block_kwargs = {
                    "causal": self.causal,
                    "timestep": scaled_timestep if self.timestep_conditioning else None,
                    "generator": generator,
                }
                sample = up_block(sample, **block_kwargs)
            elif isinstance(up_block, ResnetBlock3D):
                sample = up_block(sample, causal=self.causal, generator=generator)
            else:
                sample = up_block(sample, causal=self.causal)

        sample = self.conv_norm_out(sample)

        if self.timestep_conditioning:
            embedded_timestep = self.last_time_embedder(
                timestep=scaled_timestep.flatten(),
                hidden_dtype=sample.dtype,
            )
            embedded_timestep = embedded_timestep.view(batch_size, embedded_timestep.shape[-1], 1, 1, 1)
            ada_values = self.last_scale_shift_table[None, ..., None, None, None].to(
                device=sample.device, dtype=sample.dtype
            ) + embedded_timestep.reshape(
                batch_size,
                2,
                -1,
                embedded_timestep.shape[-3],
                embedded_timestep.shape[-2],
                embedded_timestep.shape[-1],
            )
            shift, scale = ada_values.unbind(dim=1)
            sample = sample * (1 + scale) + shift

        sample = self.conv_act(sample)
        sample = self.conv_out(sample, causal=self.causal)

        # Final spatial expansion: reverse the initial patchify from encoder
        # Moves pixels from channels back to spatial dimensions
        # Example: (B, 48, F, 128, 128) -> (B, 3, F, 512, 512) with patch_size=4
        sample = unpatchify(sample, patch_size_hw=self.patch_size, patch_size_t=1)

        return sample
    ############## LTX2 Streaming Mode - Copyright DeepBeepMeep 2025-2026 #############
    _offload_hooks = ["stream_step"]  # MMGP loads the weights when a chunk is decoded, as for forward

    def stream_step(self, stream: DecoderStream, latent: torch.Tensor, last: bool) -> None:
        stream.step(latent, last)

    def tiled_decode(self, latent: torch.Tensor, tiling_config: TilingConfig | None, write: Callable[[int, slice, slice, torch.Tensor], None], interrupt_check: Callable[[], bool] | None = None) -> bool:
        """
        Decodes the latent spatial tile after spatial tile, each tile streamed latent frame after latent frame exactly as in the decoding of
        the whole video (see decode_stream: no temporal tiles), so that VRAM only holds the state of one tile. `write(first frame, rows,
        columns, frames (B, C, T, h, w))` receives the patches of the video: the parts of a tile that no other tile covers as they are
        decoded, the overlaps of the tiles blended (on the CPU) when their last tile is decoded. Returns False when interrupted.
        """
        scale = self.video_downscale_factors
        tiles = _stream_tiles(latent.shape, tiling_config, STREAM_TILE_AREA, STREAM_FRAME_SHARE, scale.height, latent.device)
        frames, height, width = latent.shape[2], latent.shape[3] * scale.height, latent.shape[4] * scale.width
        regions = [(tile.out_coords[3].indices(height)[:2], tile.out_coords[4].indices(width)[:2]) for tile in tiles]
        rows, cols = sorted({v for region in regions for v in region[0]}), sorted({v for region in regions for v in region[1]})
        cells = [((top, bottom), (left, right), [i for i, ((t, b), (l, r)) in enumerate(regions) if t <= top and bottom <= b and l <= left and right <= r])
                 for top, bottom in zip(rows, rows[1:]) for left, right in zip(cols, cols[1:])]
        masks = []  # blending weights of each tile, (h, w)
        for tile, ((top, bottom), (left, right)) in zip(tiles, regions):
            mask = torch.ones((bottom - top, right - left), device=latent.device)
            for axis, view in ((3, (-1, 1)), (4, (1, -1))):
                if tile.masks_1d[axis] is not None:
                    mask *= tile.masks_1d[axis].view(view)
            masks.append(mask)

        def local(index: int, top: int, bottom: int, left: int, right: int) -> tuple[slice, slice]:
            (tile_top, _), (tile_left, _) = regions[index]
            return slice(top - tile_top, bottom - tile_top), slice(left - tile_left, right - tile_left)

        weights = {}
        for (top, bottom), (left, right), owners in cells:
            if len(owners) > 1:
                weights[(top, left)] = sum(masks[i][local(i, top, bottom, left, right)] for i in owners).cpu()
        sums, position = {}, 0

        def sink(out: torch.Tensor) -> None:  # frames completed by the stream of the tile `index`, from frame `position`
            nonlocal position
            count = out.shape[2]
            for (top, bottom), (left, right), owners in mine:
                part = out[(..., *local(index, top, bottom, left, right))]
                if len(owners) == 1:
                    write(position, slice(top, bottom), slice(left, right), part)
                    continue
                part = part.mul(masks[index][local(index, top, bottom, left, right)].to(part.dtype)).float().cpu()
                total = sums.get((top, left))
                if total is None:
                    total = sums[(top, left)] = part.new_zeros((*part.shape[:2], (frames - 1) * scale.time + 1, *part.shape[3:]))
                total = total[:, :, position : position + count].add_(part)
                if index == owners[-1]:
                    write(position, slice(top, bottom), slice(left, right), total.div_(weights[(top, left)]))
            position += count

        with PhaseProgress(frames * len(tiles)) as progress:
            for index, tile in enumerate(tiles):
                stream, position, mine = DecoderStream(self, STREAM_PIECE_FRAMES, sink), 0, [cell for cell in cells if index in cell[2]]
                for latent_index in range(frames):
                    if interrupt_check is not None and interrupt_check():
                        return False
                    self.stream_step(stream, latent[tile.in_coords][:, :, latent_index : latent_index + 1], latent_index == frames - 1)
                    progress.advance()
                for (top, bottom), (left, right), owners in mine:
                    if index == owners[-1]:
                        sums.pop((top, left), None)
        return True


def decode_video(
    latent: torch.Tensor | list[torch.Tensor],
    video_decoder: VideoDecoder,
    tiling_config: TilingConfig | None = None,
    interrupt_check: Callable[[], bool] | None = None,
    generator: torch.Generator | None = None,
) -> Iterator[torch.Tensor]:
    """
    Decode a video latent tensor with the given decoder.
    Args:
        latent: Tensor [c, f, h, w]
        video_decoder: Decoder module.
        tiling_config: Optional tiling settings.
    Yields:
        The decoded video [f, h, w, c], uint8 in [0, 255].
    """
    video = decode_video_to_tensor(latent, video_decoder, tiling_config, interrupt_check=interrupt_check, generator=generator)
    if video is not None:
        yield video


def decode_video_to_tensor(
    latent: torch.Tensor | list[torch.Tensor],
    video_decoder: VideoDecoder,
    tiling_config: TilingConfig | None = None,
    expected_frames: int | None = None,
    expected_height: int | None = None,
    expected_width: int | None = None,
    interrupt_check: Callable[[], bool] | None = None,
    hdr_transform: str | None = None,
    output_dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
    keyframes=None,
) -> torch.Tensor | None:
    """``keyframes`` (:class:`DecodeKeyframes`) enables the keyframe-aware decode of the NAD diffusion decoder."""
    if isinstance(latent, list):
        latent_tensor = latent[0]
        latent.clear()
        latent = latent_tensor
    full_video_shape = VideoLatentShape.from_torch_shape(latent.shape).upscale(video_decoder.video_downscale_factors)
    frame_capacity = int(expected_frames) if expected_frames is not None else int(full_video_shape.frames)
    target_height = int(expected_height) if expected_height is not None else int(full_video_shape.height)
    target_width = int(expected_width) if expected_width is not None else int(full_video_shape.width)
    if frame_capacity <= 0 or target_height <= 0 or target_width <= 0:
        return None

    is_hdr = hdr_transform is not None
    tensor_dtype = output_dtype or (torch.float16 if is_hdr else torch.uint8)
    with torch.inference_mode(False):
        video_tensor = torch.empty((frame_capacity, target_height, target_width, 3), dtype=tensor_dtype, device="cpu")
    tiled_iterator = None
    decoded_video = None
    write_pos = 0
    is_diffusion_decoder = getattr(video_decoder, "is_diffusion_decoder", False)

    def write(position: int, rows: slice, cols: slice, frames: torch.Tensor) -> int:
        # patch of the video from frame `position`, cropped to the expected size
        frame_count = min(int(frames.shape[2]), frame_capacity - position)
        (top, bottom, _), (left, right, _) = rows.indices(target_height), cols.indices(target_width)
        if frame_count <= 0 or bottom <= top or right <= left:
            return frame_count
        frames = frames[:, :, :frame_count, : bottom - top, : right - left]
        if frames.dtype is torch.uint8:
            pass
        elif is_hdr:
            frames = vae_range_to_hdr_linear(frames, transform=hdr_transform, channel_dim=1).to(dtype=tensor_dtype)
        else:
            frames = frames.add_(1.0).mul_(127.5).clamp_(0.0, 255.0)
        transfer.copy_(video_tensor[position : position + frame_count, top:bottom, left:right], frames[0].permute(1, 2, 3, 0))
        return frame_count

    try:
        if tiling_config is not None and is_diffusion_decoder:
            tiled_decode_kwargs = {"generator": generator, "interrupt_check": interrupt_check, "copy_emitted_chunks": False, "output_uint8": tensor_dtype is torch.uint8, "keyframes": keyframes, "high_precision": is_hdr}
            decoder_input = [latent]
            latent = None
            tiled_iterator = video_decoder.tiled_decode(decoder_input, tiling_config, **tiled_decode_kwargs)
            for frames in tiled_iterator:
                if interrupt_check is not None and interrupt_check():
                    return None
                if frames is None:
                    continue
                frame_count = write(write_pos, slice(None), slice(None), frames)
                if frame_count <= 0:
                    break
                write_pos += frame_count
        elif tiling_config is not None:
            def write_patch(position: int, rows: slice, cols: slice, frames: torch.Tensor) -> None:
                nonlocal write_pos
                write_pos = max(write_pos, position + write(position, rows, cols, frames))

            if not video_decoder.tiled_decode(latent, tiling_config, write_patch, interrupt_check):
                return None
        else:
            if interrupt_check is not None and interrupt_check():
                return None
            if is_diffusion_decoder:
                decoder_input = [latent]
                latent = None
                decoded_video = video_decoder(decoder_input, generator=generator, interrupt_check=interrupt_check, keyframes=keyframes, high_precision=is_hdr)
            else:
                with PhaseProgress(1) as progress:
                    decoded_video = video_decoder(latent)
                    progress.advance()
            if interrupt_check is not None and interrupt_check():
                return None
            frame_count = min(int(decoded_video.shape[2]), frame_capacity)
            if frame_count <= 0:
                return None
            decoded_video = decoded_video[:, :, :frame_count, :target_height, :target_width]
            if is_hdr:
                decoded_video = vae_range_to_hdr_linear(decoded_video, transform=hdr_transform, channel_dim=1).to(dtype=tensor_dtype)
            else:
                decoded_video = decoded_video.add_(1.0).mul_(127.5).clamp_(0.0, 255.0)
            transfer.copy_(video_tensor[:frame_count], decoded_video[0].permute(1, 2, 3, 0))  # (to RAM, without a GPU copy of the video)
            write_pos = frame_count
        return video_tensor[:write_pos]
    finally:
        if tiled_iterator is not None:
            close = getattr(tiled_iterator, "close", None)
            if close is not None:
                close()
        decoded_video = None
        tiling_config = None
        video_decoder = None
        latent = None


def _default_intervals(length: int) -> DimensionIntervals:
    return DEFAULT_SPLIT_OPERATION(length)


def _pixel_frames_to_latent_frames(frames: int, scale: int) -> int:
    frames = int(frames)
    if frames <= 0:
        return 0
    scale = max(1, int(scale))
    return (frames - 1 + scale - 1) // scale + 1


def _build_temporal_intervals(
    length: int,
    tiling_config: TilingConfig | None,
    scale: int,
) -> DimensionIntervals:
    if (
        tiling_config is None
        or tiling_config.temporal_config is None
        or length <= 0
    ):
        return _default_intervals(length)
    cfg = tiling_config.temporal_config
    tile_size = max(1, _pixel_frames_to_latent_frames(cfg.tile_size_in_frames, scale))
    overlap = max(0, _pixel_frames_to_latent_frames(cfg.tile_overlap_in_frames, scale))
    overlap = min(overlap, max(tile_size - 1, 0))
    return split_in_temporal(tile_size, overlap)(length)


def _build_spatial_intervals(
    length: int,
    tiling_config: TilingConfig | None,
    factor: int,
    long_side: int,
) -> DimensionIntervals:
    if (
        tiling_config is None
        or tiling_config.spatial_config is None
        or length <= 0
    ):
        return _default_intervals(length)
    cfg = tiling_config.spatial_config
    size = cfg.tile_size_in_pixels // factor
    overlap = cfg.tile_overlap_in_pixels // factor
    lower_threshold = max(2, overlap + 1)
    tile_size = max(lower_threshold, round(size * length / max(1, long_side)))
    return split_in_spatial(tile_size, overlap)(length)


def _make_mask_1d(
    length: int,
    left_ramp: int,
    right_ramp: int,
    *,
    left_starts_from_0: bool,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if length <= 0:
        return torch.ones(0, device=device, dtype=dtype)
    return compute_trapezoidal_mask_1d(length, left_ramp, right_ramp, left_starts_from_0, device=device).to(dtype=dtype)


def _encoder_input(video: torch.Tensor, device: torch.device | None, dtype: torch.dtype | None) -> torch.Tensor:
    """Videos may stay on CPU; each encoded tile is moved to ``device`` on its own, and uint8 tiles are normalized to [-1, 1]."""
    if video.dtype != torch.uint8:
        return video.to(device=device)
    return video.to(device=device).to(dtype).div_(127.5).sub_(1.0)


def encode_video(
    video: torch.Tensor,
    video_encoder: VideoEncoder,
    tiling_config: TilingConfig | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """
    Encode a video tensor with the given encoder, optionally using spatial/temporal tiling.
    Args:
        video: Tensor [b, c, f, h, w], possibly on CPU: each tile is moved to ``device`` on its own. A uint8 video
            (0-255) is also converted per tile to ``dtype`` in [-1, 1], so the full-resolution float video never exists.
        video_encoder: Encoder module.
        tiling_config: Optional tiling settings.
    Returns:
        Encoded latent tensor [b, c_l, f_l, h_l, w_l].
    """
    if tiling_config is None or (
        tiling_config.spatial_config is None and tiling_config.temporal_config is None
    ):
        with vae_encoding_progress(1, video_encoder, enabled=video.shape[2] > 1):
            return video_encoder(_encoder_input(video, device, dtype))
    if video.shape[2] == 1 and tiling_config.spatial_config is not None:
        tiling_config = replace(tiling_config, spatial_config=None)
    # streamed (see streaming.py): spatial tile after spatial tile, by chunks of frames (the first frame alone); the tiles blended
    batch, _, frames, height, width = video.shape
    scale = VIDEO_SCALE_FACTORS
    if (frames - 1) % scale.time:
        raise ValueError("Invalid number of frames: Encode input must have 1 + 8 * x frames (e.g., 1, 9, 17, ...). Please check your input.")
    shape = (batch, getattr(video_encoder, "latent_channels", 128), 1 + (frames - 1) // scale.time, height // scale.height, width // scale.width)
    tiles = _stream_tiles(shape, tiling_config, ENCODE_TILE_AREA, 0, 1, device)
    chunks = [(0, 1)] + [(start, min(start + ENCODE_CHUNK_FRAMES, frames)) for start in range(1, frames, ENCODE_CHUNK_FRAMES)]
    output = weights = None
    with vae_encoding_progress(len(tiles) * len(chunks), video_encoder, enabled=frames > 1):
        for tile in tiles:
            stream, mask, index = EncoderStream(video_encoder), None, 0
            rows, cols = (slice(region.start * scale.height, region.stop * scale.height) for region in tile.in_coords[3:5])
            for start, stop in chunks:
                latent = video_encoder(_encoder_input(video[:, :, start:stop, rows, cols], device, dtype), stream=stream)
                if latent is None:
                    continue
                count, index = latent.shape[2], index + latent.shape[2]
                if output is None:
                    output = latent.new_zeros(shape)
                    weights = latent.new_zeros(shape[3:]) if len(tiles) > 1 else None
                if weights is None:
                    output[:, :, index - count : index] = latent
                    continue
                if mask is None:  # blending weights of the tile, (h, w)
                    mask = latent.new_ones(latent.shape[3:])
                    for axis, view in ((3, (-1, 1)), (4, (1, -1))):
                        if tile.masks_1d[axis] is not None:
                            mask *= tile.masks_1d[axis].view(view).to(device=mask.device, dtype=mask.dtype)
                    weights[tile.out_coords[3], tile.out_coords[4]] += mask
                output[:, :, index - count : index, tile.out_coords[3], tile.out_coords[4]] += latent * mask
    return output if weights is None else output / weights.clamp_min(1e-6)


def get_video_chunks_number(num_frames: int, tiling_config: TilingConfig | None = None) -> int:
    """
    Get the number of video chunks for a given number of frames and tiling configuration.
    Args:
        num_frames: Number of frames in the video.
        tiling_config: Tiling configuration.
    Returns:
        Number of video chunks.
    """
    if not tiling_config or not tiling_config.temporal_config:
        return 1
    cfg = tiling_config.temporal_config
    scale = VIDEO_SCALE_FACTORS.time
    latent_frames = _pixel_frames_to_latent_frames(num_frames, scale)
    tile_size = _pixel_frames_to_latent_frames(cfg.tile_size_in_frames, scale)
    overlap = _pixel_frames_to_latent_frames(cfg.tile_overlap_in_frames, scale)
    overlap = min(overlap, max(tile_size - 1, 0))
    if latent_frames <= tile_size:
        return 1
    intervals = split_in_temporal(tile_size, overlap)(latent_frames)
    return len(intervals.starts)


def split_in_spatial(size: int, overlap: int) -> SplitOperation:
    def split(dimension_size: int) -> DimensionIntervals:
        if dimension_size <= size:
            return DEFAULT_SPLIT_OPERATION(dimension_size)
        amount = (dimension_size + size - 2 * overlap - 1) // (size - overlap)
        starts = [i * (size - overlap) for i in range(amount)]
        ends = [start + size for start in starts]
        ends[-1] = dimension_size
        left_ramps = [0] + [overlap] * (amount - 1)
        right_ramps = [overlap] * (amount - 1) + [0]
        return DimensionIntervals(starts=starts, ends=ends, left_ramps=left_ramps, right_ramps=right_ramps)

    return split


def split_in_temporal(size: int, overlap: int) -> SplitOperation:
    non_causal_split = split_in_spatial(size, overlap)

    def split(dimension_size: int) -> DimensionIntervals:
        if dimension_size <= size:
            return DEFAULT_SPLIT_OPERATION(dimension_size)
        intervals = non_causal_split(dimension_size)
        starts = intervals.starts
        starts[1:] = [s - 1 for s in starts[1:]]
        left_ramps = intervals.left_ramps
        left_ramps[1:] = [r + 1 for r in left_ramps[1:]]
        return replace(intervals, starts=starts, left_ramps=left_ramps)

    return split


def to_mapping_operation(
    map_func: Callable[[int, int, int, int, int, torch.device], Tuple[slice, torch.Tensor]],
    scale: int,
    device: torch.device,
) -> MappingOperation:
    def map_op(intervals: DimensionIntervals) -> tuple[list[slice], list[torch.Tensor | None]]:
        output_slices: list[slice] = []
        masks_1d: list[torch.Tensor | None] = []
        number_of_slices = len(intervals.starts)
        for i in range(number_of_slices):
            start = intervals.starts[i]
            end = intervals.ends[i]
            left_ramp = intervals.left_ramps[i]
            right_ramp = intervals.right_ramps[i]
            output_slice, mask_1d = map_func(start, end, left_ramp, right_ramp, scale, device)
            output_slices.append(output_slice)
            masks_1d.append(mask_1d)
        return output_slices, masks_1d

    return map_op


def map_temporal_slice(begin: int, end: int, left_ramp: int, right_ramp: int, scale: int, device: torch.device) -> Tuple[slice, torch.Tensor]:
    start = begin * scale
    stop = 1 + (end - 1) * scale
    left_ramp = 1 + (left_ramp - 1) * scale
    right_ramp = right_ramp * scale

    return slice(start, stop), compute_trapezoidal_mask_1d(stop - start, left_ramp, right_ramp, True, device=device)


def map_spatial_slice(begin: int, end: int, left_ramp: int, right_ramp: int, scale: int, device: torch.device) -> Tuple[slice, torch.Tensor]:
    start = begin * scale
    stop = end * scale
    left_ramp = left_ramp * scale
    right_ramp = right_ramp * scale

    return slice(start, stop), compute_trapezoidal_mask_1d(stop - start, left_ramp, right_ramp, False, device=device)
