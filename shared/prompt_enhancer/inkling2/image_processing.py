# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
# Pixel layout after Inkling2ImageProcessor of Hugging Face Transformers (pull request 51050, Apache License 2.0, Copyright 2026 the
# HuggingFace Team).
"""Images to the inputs of the Inkling 2 vision tower: one 40x40 pixel cell per image token."""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

CELL_SIZE = 40
MAX_IMAGE_TOKENS = 4096  # per picture
MAX_IMAGE_EDGE = 4096  # pixels, the model card's recommended range is 40 to 4096 per dimension
PAD_VALUE = -1.0  # on the 0-255 scale: padded pixels are just below black once normalized


class Inkling2ImageInputs(NamedTuple):
    pixel_patches: torch.Tensor  # [image tokens, 40, 40, 3] FP32 in [-1, 1] (padding slightly below), on the CPU
    grids: list[tuple[int, int]]  # (rows, columns) of cells per image
    token_counts: list[int]  # rows * columns: the number of image placeholder tokens to write for each image


def grid_of(width: int, height: int) -> tuple[int, int]:
    """(rows, columns) of the cells of a picture of this size."""
    return math.ceil(height / CELL_SIZE), math.ceil(width / CELL_SIZE)


def fit_size(width: int, height: int, max_tokens: int = MAX_IMAGE_TOKENS, max_edge: int = MAX_IMAGE_EDGE) -> tuple[int, int]:
    """Largest (width, height) of the same aspect ratio, never above the source, within max_tokens cells and max_edge pixels per side.
    A side is not shrunk below one cell; a source side already below one cell is kept (the cell is padded)."""
    scale = min(1.0, max_edge / max(width, height), math.sqrt(max_tokens * CELL_SIZE**2 / (width * height)))  # no feasible scale is above
    while True:
        new_width, new_height = (min(side, max(min(side, CELL_SIZE), round(side * scale))) for side in (width, height))
        rows, columns = grid_of(new_width, new_height)
        if rows * columns <= max_tokens or rows == columns == 1:
            return new_width, new_height
        scale = max(CELL_SIZE * (rows - 1) / height, CELL_SIZE * (columns - 1) / width)  # the next scale down that drops a row or a column


def _to_pil(image) -> Image.Image:
    if isinstance(image, Image.Image):
        return image
    if image.dtype != torch.uint8 or image.ndim != 3 or image.shape[0] not in (3, 4):
        raise ValueError(f"Inkling 2 images are PIL images or uint8 tensors [3 or 4, height, width], got {image.dtype} {tuple(image.shape)}.")
    return Image.fromarray(image.permute(1, 2, 0).cpu().numpy())


def to_rgb(image: Image.Image) -> torch.Tensor:
    """uint8 [3, height, width] on the CPU; transparency is composited on the model's gray checkerboard (8-pixel squares)."""
    pixels = torch.from_numpy(np.array(image.convert("RGBA" if image.has_transparency_data else "RGB"))).permute(2, 0, 1)
    if pixels.shape[0] == 3:
        return pixels
    height, width = pixels.shape[-2:]
    rows, columns = torch.arange(height, device="cpu")[:, None] // 8, torch.arange(width, device="cpu")[None, :] // 8
    background = torch.where((rows + columns) % 2 == 0, 204, 153)
    pixels = pixels.to(torch.int32)
    alpha = pixels[3:4]
    return ((pixels[:3] * alpha + background * (255 - alpha) + 127) // 255).to(torch.uint8)


def to_cells(pixels: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    """uint8 [3, height, width] -> FP32 cells [rows * columns, 40, 40, 3] (row-major), padded at the bottom and right, and (rows, columns)."""
    height, width = pixels.shape[-2:]
    rows, columns = grid_of(width, height)
    pixels = F.pad(pixels.to(torch.float32), (0, columns * CELL_SIZE - width, 0, rows * CELL_SIZE - height), value=PAD_VALUE)
    pixels = (pixels - 127.5) / 127.5
    return pixels.view(3, rows, CELL_SIZE, columns, CELL_SIZE).permute(1, 3, 2, 4, 0).flatten(0, 1), (rows, columns)


def preprocess_images(images, max_tokens: int = MAX_IMAGE_TOKENS, max_edge: int = MAX_IMAGE_EDGE) -> Inkling2ImageInputs:
    """PIL images or uint8 tensors [3 or 4, height, width] -> the tower's inputs. Pictures above max_tokens cells or max_edge pixels are
    reduced first (Lanczos, aspect ratio kept); none is enlarged. Orientation (EXIF) is the caller's."""
    patches, grids = [], []
    for image in images if isinstance(images, (list, tuple)) else [images]:
        image = _to_pil(image)
        size = fit_size(*image.size, max_tokens=max_tokens, max_edge=max_edge)
        if size != image.size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        cells, grid = to_cells(to_rgb(image))
        patches.append(cells)
        grids.append(grid)
    return Inkling2ImageInputs(torch.cat(patches), grids, [rows * columns for rows, columns in grids])
