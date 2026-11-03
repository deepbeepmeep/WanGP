# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Image orientation and color metadata, and channel operations for Deepy media tools."""

from __future__ import annotations

import struct
from pathlib import Path

from PIL import Image, ImageColor, ImageOps


CHANNELS = "RGBA"
EXIF_ORIENTATION = 0x0112
PNG_METADATA_CHUNKS = (b"eXIf", b"tEXt", b"zTXt", b"iTXt")  # where Pillow finds an orientation: EXIF, a raw EXIF profile or XMP
PNG_TAIL_BYTES = 128 << 10


def _png_metadata_after_data(path: str) -> bool:
    """Whether a PNG has metadata chunks after its image data. They sit just before its end: the last 128 KiB show them after the
    last IDAT chunk. A tail without one belongs to a file of few large chunks, whose headers are walked instead."""
    with open(path, "rb") as file:
        file.seek(max(0, file.seek(0, 2) - PNG_TAIL_BYTES))
        tail = file.read()
        if (data := tail.rfind(b"IDAT")) >= 0:
            return any(kind in tail[data:] for kind in PNG_METADATA_CHUNKS)  # a false match only costs Pillow's decode
        file.seek(8)
        seen_data = False
        while len(header := file.read(8)) == 8:
            length, kind = struct.unpack(">I4s", header)
            if kind == b"IEND":
                return False
            if kind == b"IDAT":
                seen_data = True
            elif seen_data and kind in PNG_METADATA_CHUNKS:
                return True
            file.seek(length + 4, 1)
    return False


def exif_orientation(image: Image.Image) -> int:
    """EXIF orientation of an opened image as Pillow, and so WanGP's generation, reads it (an EXIF block, a raw EXIF profile or XMP),
    without decoding its pixels: Pillow decodes a whole PNG to look for them after its image data, only needed when some are there."""
    if image.format == "PNG" and "exif" not in image.info and not _png_metadata_after_data(image.filename):
        return Image.Image.getexif(image).get(EXIF_ORIENTATION, 1)  # the chunks read at opening hold all the metadata
    return image.getexif().get(EXIF_ORIENTATION, 1)


def upright(image: Image.Image) -> Image.Image:
    """The image as the Gallery and WanGP's generation show it: a photo's EXIF orientation applied to its pixels, in place."""
    if exif_orientation(image) != 1:
        ImageOps.exif_transpose(image, in_place=True)
    return image


def upright_size(image: Image.Image) -> tuple[int, int]:
    """(width, height) as displayed, without decoding the pixels: EXIF orientations 5 to 8 turn the image a quarter."""
    return image.size[::-1] if exif_orientation(image) in (5, 6, 7, 8) else image.size


def image_color_details(image: Image.Image) -> dict:
    bands = list(image.getbands())
    has_alpha = "A" in bands or "transparency" in image.info
    alpha_extrema = None
    if has_alpha:
        alpha = image.getchannel("A") if "A" in bands else image.convert("RGBA").getchannel("A")
        alpha_extrema = [int(value) for value in alpha.getextrema()]
    return {
        "color_mode": image.mode,
        "bands": bands,
        "has_alpha": has_alpha,
        "has_transparent_pixels": bool(alpha_extrema and alpha_extrema[0] < 255),
        "alpha_extrema": alpha_extrema,
    }


def _load_image(path: str) -> Image.Image:
    with Image.open(Path(path)) as image:
        return upright(image).copy()


def _flatten(image: Image.Image, background: str | None) -> Image.Image:
    if not background:
        raise ValueError("background is required when converting an image with alpha to RGB.")
    try:
        color = ImageColor.getrgb(background)
    except ValueError as exc:
        raise ValueError("background must be a color name or hex color.") from exc
    backdrop = Image.new("RGBA", image.size, (*color[:3], 255))
    return Image.alpha_composite(backdrop, image.convert("RGBA")).convert("RGB")


def render_image_channels(
    operation: str,
    image_path: str | None = None,
    mode: str | None = None,
    channels: list[str] | None = None,
    channel_paths: dict[str, str] | None = None,
    background: str | None = None,
) -> list[tuple[str, Image.Image]]:
    """Return named PNG-ready images; callers choose paths and publish them."""
    operation = str(operation or "").strip().lower()
    if operation not in {"convert", "extract", "combine"}:
        raise ValueError("operation must be convert, extract, or combine.")
    if operation != "combine" and not image_path:
        raise ValueError("media_id is required for convert and extract.")
    if mode is not None and mode not in {"RGB", "RGBA"}:
        raise ValueError("mode must be RGB or RGBA.")
    if operation != "extract" and channels is not None:
        raise ValueError("channels is only used by extract.")
    if operation != "combine" and channel_paths:
        raise ValueError("channel_sources is only used by combine.")

    if operation == "convert":
        image = _load_image(image_path)
        target = mode or "RGBA"
        if target == "RGB" and image_color_details(image)["has_alpha"]:
            image = _flatten(image, background)
        else:
            image = image.convert(target)
        return [(f"converted_{target.lower()}", image)]

    if operation == "extract":
        image = _load_image(image_path)
        if image.mode not in {"RGB", "RGBA"}:
            raise ValueError("extract requires an RGB or RGBA source image.")
        if channels is not None and not isinstance(channels, list):
            raise ValueError("channels must be a list of R, G, B, or A.")
        selected = list(image.getbands()) if channels is None else channels
        if not selected or len(set(selected)) != len(selected) or any(channel not in image.getbands() for channel in selected):
            raise ValueError("channels must be distinct channels present in the source image.")
        return [(f"channel_{channel}", image.getchannel(channel)) for channel in selected]

    paths = channel_paths or {}
    if not isinstance(paths, dict) or any(channel not in CHANNELS for channel in paths):
        raise ValueError("channel_sources must map only R, G, B, or A to image IDs.")
    if not image_path and not all(channel in paths for channel in "RGB"):
        raise ValueError("combine needs media_id or R, G, and B channel sources.")
    base = _load_image(image_path) if image_path else None
    if base is not None and base.mode not in {"RGB", "RGBA"}:
        raise ValueError("combine requires an RGB or RGBA base image.")
    masks = {}
    for channel, path in paths.items():
        mask = _load_image(path)
        if mask.mode != "L":
            raise ValueError(f"channel_sources.{channel} must be an 8-bit grayscale image.")
        masks[channel] = mask
    size = base.size if base is not None else next(iter(masks.values())).size
    if any(mask.size != size for mask in masks.values()):
        raise ValueError("All channel sources and the base image must have the same dimensions.")
    target = mode or ("RGBA" if "A" in masks or base is not None and base.mode == "RGBA" else "RGB")
    if target == "RGB" and "A" in masks:
        raise ValueError("An A channel requires RGBA output.")
    if base is not None and base.mode == "RGBA" and target == "RGB":
        base = _flatten(base, background)
    elif base is not None:
        base = base.convert(target)
    base_bands = dict(zip(base.getbands(), base.split())) if base is not None else {}
    merged = Image.merge(target, tuple(masks.get(channel) or base_bands.get(channel) or Image.new("L", size, 255) for channel in target))
    return [(f"combined_{target.lower()}", merged)]
