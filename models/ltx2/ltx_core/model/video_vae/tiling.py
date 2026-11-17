import itertools
from dataclasses import dataclass
from typing import Callable, List, NamedTuple, Tuple

import torch


def compute_trapezoidal_mask_1d(
    length: int,
    ramp_left: int,
    ramp_right: int,
    left_starts_from_0: bool = False,
    *,
    device: torch.device,
) -> torch.Tensor:
    """
    Generate a 1D trapezoidal blending mask with linear ramps.
    Args:
        length: Output length of the mask.
        ramp_left: Fade-in length on the left.
        ramp_right: Fade-out length on the right.
        left_starts_from_0: Whether the ramp starts from 0 or first non-zero value.
            Useful for temporal tiles where the first tile is causal.
        device: Device on which the mask is created.
    Returns:
        A 1D tensor of shape `(length,)` with values in [0, 1].
    """
    if length <= 0:
        raise ValueError("Mask length must be positive.")

    ramp_left = max(0, min(ramp_left, length))
    ramp_right = max(0, min(ramp_right, length))

    mask = torch.ones(length, device=device)

    if ramp_left > 0:
        interval_length = ramp_left + 1 if left_starts_from_0 else ramp_left + 2
        fade_in = torch.linspace(0.0, 1.0, interval_length, device=device)[:-1]
        if not left_starts_from_0:
            fade_in = fade_in[1:]
        mask[:ramp_left] *= fade_in

    if ramp_right > 0:
        fade_out = torch.linspace(1.0, 0.0, steps=ramp_right + 2, device=device)[1:-1]
        mask[-ramp_right:] *= fade_out

    return mask.clamp_(0, 1)


@dataclass(frozen=True)
class SpatialTilingConfig:
    """Configuration for dividing each frame into spatial tiles with optional overlap.
    Args:
        tile_size_in_pixels (int): Size of each tile in pixels. Must be at least 64 and divisible by 32.
        tile_overlap_in_pixels (int, optional): Overlap between tiles in pixels. Must be divisible by 32. Defaults to 0.
        decode_tile_latents / encode_tile_latents (int, optional): latent pixels a tile of the streamed decoding / encoding may cover,
            overlaps included (from the VAE Tiling VRAM budget). Defaults to the area derived from the tile size.
        diffusion_tile_in_pixels (int, optional): the NAD diffusion decoder's tile (from its own costs and the budget; 0: whole frames).
            Defaults to tile_size_in_pixels.
    """

    tile_size_in_pixels: int
    tile_overlap_in_pixels: int = 0
    decode_tile_latents: int | None = None
    encode_tile_latents: int | None = None
    diffusion_tile_in_pixels: int | None = None

    def __post_init__(self) -> None:
        if self.tile_size_in_pixels < 64:
            raise ValueError(f"tile_size_in_pixels must be at least 64, got {self.tile_size_in_pixels}")
        if self.tile_size_in_pixels % 32 != 0:
            raise ValueError(f"tile_size_in_pixels must be divisible by 32, got {self.tile_size_in_pixels}")
        if self.tile_overlap_in_pixels % 32 != 0:
            raise ValueError(f"tile_overlap_in_pixels must be divisible by 32, got {self.tile_overlap_in_pixels}")
        if self.tile_overlap_in_pixels >= self.tile_size_in_pixels:
            raise ValueError(
                f"Overlap must be less than tile size, got {self.tile_overlap_in_pixels} and {self.tile_size_in_pixels}"
            )


@dataclass(frozen=True)
class TemporalTilingConfig:
    """Configuration for dividing a video into temporal tiles (chunks of frames) with optional overlap.
    Args:
        tile_size_in_frames (int): Number of frames in each tile. Must be at least 16 and divisible by 8.
        tile_overlap_in_frames (int, optional): Number of overlapping frames between consecutive tiles.
            Must be divisible by 8. Defaults to 0.
    """

    tile_size_in_frames: int
    tile_overlap_in_frames: int = 0

    def __post_init__(self) -> None:
        if self.tile_size_in_frames < 16:
            raise ValueError(f"tile_size_in_frames must be at least 16, got {self.tile_size_in_frames}")
        if self.tile_size_in_frames % 8 != 0:
            raise ValueError(f"tile_size_in_frames must be divisible by 8, got {self.tile_size_in_frames}")
        if self.tile_overlap_in_frames % 8 != 0:
            raise ValueError(f"tile_overlap_in_frames must be divisible by 8, got {self.tile_overlap_in_frames}")
        if self.tile_overlap_in_frames >= self.tile_size_in_frames:
            raise ValueError(
                f"Overlap must be less than tile size, got {self.tile_overlap_in_frames} and {self.tile_size_in_frames}"
            )


@dataclass(frozen=True)
class TilingConfig:
    """Configuration for splitting video into tiles with optional overlap.
    Attributes:
        spatial_config: Configuration for splitting spatial dimensions into tiles.
        temporal_config: Configuration for splitting temporal dimension into tiles.
    """

    spatial_config: SpatialTilingConfig | None = None
    temporal_config: TemporalTilingConfig | None = None

    @classmethod
    def default(cls) -> "TilingConfig":
        return cls(
            spatial_config=SpatialTilingConfig(tile_size_in_pixels=512, tile_overlap_in_pixels=64),
            temporal_config=TemporalTilingConfig(tile_size_in_frames=64, tile_overlap_in_frames=24),
        )


@dataclass(frozen=True)
class DimensionIntervals:
    """Intervals which a single dimension of the latent space is split into.
    Each interval is defined by its start, end, left ramp, and right ramp.
    The start and end are the indices of the first and last element (exclusive) in the interval.
    Ramps are regions of the interval where the value of the mask tensor is
    interpolated between 0 and 1 for blending with neighboring intervals.
    The left ramp and right ramp values are the lengths of the left and right ramps.
    """

    starts: List[int]
    ends: List[int]
    left_ramps: List[int]
    right_ramps: List[int]


@dataclass(frozen=True)
class LatentIntervals:
    """Intervals which the latent tensor of given shape is split into.
    Each dimension of the latent space is split into intervals based on the length along said dimension.
    """

    original_shape: torch.Size
    dimension_intervals: Tuple[DimensionIntervals, ...]


# Operation to split a single dimension of the tensor into intervals based on the length along the dimension.
SplitOperation = Callable[[int], DimensionIntervals]
# Operation to map the intervals in input dimension to slices and masks along a corresponding output dimension.
MappingOperation = Callable[[DimensionIntervals], tuple[list[slice], list[torch.Tensor | None]]]


def default_split_operation(length: int) -> DimensionIntervals:
    return DimensionIntervals(starts=[0], ends=[length], left_ramps=[0], right_ramps=[0])


DEFAULT_SPLIT_OPERATION: SplitOperation = default_split_operation


def default_mapping_operation(
    _intervals: DimensionIntervals,
) -> tuple[list[slice], list[torch.Tensor | None]]:
    return [slice(0, None)], [None]


DEFAULT_MAPPING_OPERATION: MappingOperation = default_mapping_operation


class Tile(NamedTuple):
    """
    Represents a single tile.
    Attributes:
        in_coords:
            Tuple of slices specifying where to cut the tile from the INPUT tensor.
        out_coords:
            Tuple of slices specifying where this tile's OUTPUT should be placed in the reconstructed OUTPUT tensor.
        masks_1d:
            Per-dimension masks in OUTPUT units.
            These are used to create all-dimensional blending mask.
    Methods:
        blend_mask:
            Create a single N-D mask from the per-dimension masks.
    """

    in_coords: Tuple[slice, ...]
    out_coords: Tuple[slice, ...]
    masks_1d: Tuple[Tuple[torch.Tensor, ...]]

    @staticmethod
    def _format_slice(axis_slice: slice) -> str:
        start = "" if axis_slice.start is None else axis_slice.start
        stop = "" if axis_slice.stop is None else axis_slice.stop
        if axis_slice.step is None:
            return f"{start}:{stop}"
        return f"{start}:{stop}:{axis_slice.step}"

    @staticmethod
    def _format_mask(mask_1d: torch.Tensor | None) -> str:
        if mask_1d is None:
            return "None"
        return f"len={mask_1d.numel()}"

    def __repr__(self) -> str:
        in_coords = ",".join(self._format_slice(axis_slice) for axis_slice in self.in_coords)
        out_coords = ",".join(self._format_slice(axis_slice) for axis_slice in self.out_coords)
        masks = ",".join(self._format_mask(mask_1d) for mask_1d in self.masks_1d)
        return f"Tile(in=({in_coords}), out=({out_coords}), masks=({masks}))"

    @property
    def blend_mask(self) -> torch.Tensor:
        num_dims = len(self.out_coords)
        per_dimension_masks: List[torch.Tensor] = []

        for dim_idx in range(num_dims):
            mask_1d = self.masks_1d[dim_idx]
            view_shape = [1] * num_dims
            if mask_1d is None:
                # Broadcast mask along this dimension (length 1).
                one = torch.ones(1)

                view_shape[dim_idx] = 1
                per_dimension_masks.append(one.view(*view_shape))
                continue

            # Reshape (L,) -> (1, ..., L, ..., 1) so masks across dimensions broadcast-multiply.
            view_shape[dim_idx] = mask_1d.shape[0]
            per_dimension_masks.append(mask_1d.view(*view_shape))

        # Multiply per-dimension masks to form the full N-D mask (separable blending window).
        combined_mask = per_dimension_masks[0]
        for mask in per_dimension_masks[1:]:
            combined_mask = combined_mask * mask

        return combined_mask


def create_tiles_from_intervals_and_mappers(
    intervals: LatentIntervals,
    mappers: List[MappingOperation],
) -> List[Tile]:
    full_dim_input_slices = []
    full_dim_output_slices = []
    full_dim_masks_1d = []
    for axis_index in range(len(intervals.original_shape)):
        dimension_intervals = intervals.dimension_intervals[axis_index]
        starts = dimension_intervals.starts
        ends = dimension_intervals.ends
        input_slices = [slice(s, e) for s, e in zip(starts, ends, strict=True)]
        output_slices, masks_1d = mappers[axis_index](dimension_intervals)
        full_dim_input_slices.append(input_slices)
        full_dim_output_slices.append(output_slices)
        full_dim_masks_1d.append(masks_1d)

    tiles = []
    tile_in_coords = list(itertools.product(*full_dim_input_slices))
    tile_out_coords = list(itertools.product(*full_dim_output_slices))
    tile_mask_1ds = list(itertools.product(*full_dim_masks_1d))
    for in_coord, out_coord, mask_1d in zip(tile_in_coords, tile_out_coords, tile_mask_1ds, strict=True):
        tiles.append(
            Tile(
                in_coords=in_coord,
                out_coords=out_coord,
                masks_1d=mask_1d,
            )
        )
    return tiles


def create_tiles(
    latent_shape: torch.Size,
    splitters: List[SplitOperation],
    mappers: List[MappingOperation],
) -> List[Tile]:
    if len(splitters) != len(latent_shape):
        raise ValueError(
            f"Number of splitters must be equal to number of dimensions in latent shape, "
            f"got {len(splitters)} and {len(latent_shape)}"
        )
    if len(mappers) != len(latent_shape):
        raise ValueError(
            f"Number of mappers must be equal to number of dimensions in latent shape, "
            f"got {len(mappers)} and {len(latent_shape)}"
        )
    intervals = [splitter(length) for splitter, length in zip(splitters, latent_shape, strict=True)]
    latent_intervals = LatentIntervals(original_shape=latent_shape, dimension_intervals=tuple(intervals))
    return create_tiles_from_intervals_and_mappers(latent_intervals, mappers)
