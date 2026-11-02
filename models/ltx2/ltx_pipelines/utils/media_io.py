from __future__ import annotations

import math
from collections.abc import Generator, Iterator
from fractions import Fraction
from io import BytesIO
from typing import NamedTuple

import numpy as np
import torch
from einops import rearrange
from mmgp import transfer
from PIL import Image
from torch._prims_common import DeviceLikeType
from tqdm import tqdm

from shared.utils.utils import guide_to_float

from .constants import DEFAULT_IMAGE_CRF

# Control videos are prepared on the device by chunks of frames and kept in RAM; the VAE encoder uploads each tile.
CONDITIONING_CHUNK_FRAMES = 16


class UInt8Guide(NamedTuple):
    """A control video WanGP passed in uint8 (model_def "uint8_guides"). It is loaded by chunks through the [-1, 1] float values WanGP gives
    float control videos, so that the conditioning is identical without a float copy of the whole video."""
    video: torch.Tensor


def _get_av():
    import av
    return av


def resize_aspect_ratio_preserving(image: torch.Tensor, long_side: int) -> torch.Tensor:
    """
    Resize image preserving aspect ratio (filling target long side).
    Preserves the input dimensions order.
    Args:
        image: Input image tensor with shape (F (optional), H, W, C)
        long_side: Target long side size.
    Returns:
        Tensor with shape (F (optional), H, W, C) F = 1 if input is 3D, otherwise input shape[0]
    """
    height, width = image.shape[-3:2]
    max_side = max(height, width)
    scale = long_side / float(max_side)
    target_height = int(height * scale)
    target_width = int(width * scale)
    resized = resize_and_center_crop(image, target_height, target_width)
    # rearrange and remove batch dimension
    result = rearrange(resized, "b c f h w -> b f h w c")[0]
    # preserve input dimensions
    return result[0] if result.shape[0] == 1 else result


def _get_lanczos_resample() -> int:
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS


def _resize_and_center_crop_lanczos(tensor: torch.Tensor, height: int, width: int) -> torch.Tensor:
    if tensor.ndim == 3:
        frames = tensor.unsqueeze(0)
    elif tensor.ndim == 4:
        frames = tensor
    else:
        raise ValueError(f"Expected input with 3 or 4 dimensions; got shape {tensor.shape}.")

    _, src_h, src_w, _ = frames.shape
    if src_h == height and src_w == width:
        return rearrange(frames, "f h w c -> 1 c f h w")

    scale = max(height / src_h, width / src_w)
    new_h = math.ceil(src_h * scale)
    new_w = math.ceil(src_w * scale)
    frames = frames.detach().cpu()
    resample = _get_lanczos_resample()
    resized_frames = []
    for frame in frames:
        frame_np = frame.numpy()
        frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)
        pil_frame = Image.fromarray(frame_np)
        pil_frame = pil_frame.resize((new_w, new_h), resample=resample)
        crop_top = (new_h - height) // 2
        crop_left = (new_w - width) // 2
        pil_frame = pil_frame.crop((crop_left, crop_top, crop_left + width, crop_top + height))
        resized_frames.append(torch.from_numpy(np.array(pil_frame).astype(np.float32)))

    resized = torch.stack(resized_frames, dim=0)
    return rearrange(resized, "f h w c -> 1 c f h w")


def resize_and_center_crop(tensor: torch.Tensor, height: int, width: int, resample: str | None = None) -> torch.Tensor:
    """
    Resize tensor preserving aspect ratio (filling target), then center crop to exact dimensions.
    Args:
        latent: Input tensor with shape (H, W, C) or (F, H, W, C)
        height: Target height
        width: Target width
    Returns:
        Tensor with shape (1, C, 1, height, width) for 3D input or (1, C, F, height, width) for 4D input
    """
    if resample == "lanczos":
        return _resize_and_center_crop_lanczos(tensor, height, width)

    if tensor.ndim == 3:
        tensor = rearrange(tensor, "h w c -> 1 c h w")
    elif tensor.ndim == 4:
        tensor = rearrange(tensor, "f h w c -> f c h w")
    else:
        raise ValueError(f"Expected input with 3 or 4 dimensions; got shape {tensor.shape}.")

    _, _, src_h, src_w = tensor.shape
    if src_h == height and src_w == width:
        return rearrange(tensor, "f c h w -> 1 c f h w")

    scale = max(height / src_h, width / src_w)
    # Use ceil to avoid floating-point rounding causing new_h/new_w to be
    # slightly smaller than target, which would result in negative crop offsets.
    new_h = math.ceil(src_h * scale)
    new_w = math.ceil(src_w * scale)

    tensor = torch.nn.functional.interpolate(tensor, size=(new_h, new_w), mode="bilinear", align_corners=False)

    crop_top = (new_h - height) // 2
    crop_left = (new_w - width) // 2
    tensor = tensor[:, :, crop_top : crop_top + height, crop_left : crop_left + width]

    tensor = rearrange(tensor, "f c h w -> 1 c f h w")
    return tensor


def normalize_latent(latent: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return (latent / 127.5 - 1.0).to(device=device, dtype=dtype)


def _coerce_image_input(image_input: object) -> torch.Tensor:
    if isinstance(image_input, Image.Image):
        image = np.array(image_input)[..., :3]
        return torch.tensor(image, dtype=torch.float32, device="cpu")
    if isinstance(image_input, np.ndarray):
        return torch.tensor(image_input[..., :3], dtype=torch.float32, device="cpu")
    if torch.is_tensor(image_input):
        return image_input.detach().clone().to(dtype=torch.float32)
    raise TypeError(f"Unsupported image input type: {type(image_input)}")


def _normalize_image_tensor(image: torch.Tensor) -> torch.Tensor:
    image = image.float()
    if image.ndim == 3:
        if image.shape[-1] in (1, 3, 4):
            return image
        if image.shape[0] in (1, 3, 4):
            return image.movedim(0, -1)
    if image.ndim == 4 and image.shape[-1] in (1, 3, 4):
        return image
    raise ValueError(f"Unsupported image tensor shape: {tuple(image.shape)}")


def _coerce_video_input(video_input: object) -> torch.Tensor:
    if isinstance(video_input, Image.Image):
        image = np.array(video_input)[..., :3]
        return torch.tensor(image, dtype=torch.float32, device="cpu")
    if isinstance(video_input, np.ndarray):
        return torch.tensor(video_input, dtype=torch.float32, device="cpu")
    if torch.is_tensor(video_input):
        return video_input.detach().clone().to(dtype=torch.float32)
    raise TypeError(f"Unsupported video input type: {type(video_input)}")


def _normalize_video_tensor(video: torch.Tensor) -> torch.Tensor:
    if video.ndim == 3:
        if video.shape[-1] in (1, 3, 4):
            return video.unsqueeze(0)
        if video.shape[0] in (1, 3, 4):
            return video.permute(1, 2, 0).unsqueeze(0)
    if video.ndim == 4:
        if video.shape[-1] in (1, 3, 4):
            return video
        if video.shape[0] in (1, 3, 4):
            return video.permute(1, 2, 3, 0)
        if video.shape[1] in (1, 3, 4):
            return video.permute(0, 2, 3, 1)
    raise ValueError(f"Unsupported video tensor shape: {tuple(video.shape)}")

def _scale_to_255(image: torch.Tensor, value_range: tuple[float, float] | None = None) -> torch.Tensor:
    """``value_range`` is the ``(min, max)`` of the whole clip when ``image`` is one chunk of it."""
    if not torch.is_floating_point(image):
        return image.to(dtype=torch.float32)
    min_val, max_val = value_range or (float(image.min()), float(image.max()))
    if max_val <= 1.0 and min_val >= -1.0:
        image = (image + 1.0) * 127.5
    elif max_val <= 1.0 and min_val >= 0.0:
        image = image * 255.0
    return image


def load_image_conditioning(
    image_path: str | torch.Tensor | np.ndarray | Image.Image,
    height: int,
    width: int,
    dtype: torch.dtype,
    device: torch.device,
    resample: str | None = None,
) -> torch.Tensor:
    """
    Loads an image from a path or tensor and preprocesses it for conditioning.
    Note: The image is resized to the nearest multiple of 2 for compatibility with video codecs.
    """
    if isinstance(image_path, str):
        image = decode_image(image_path=image_path)
        image = preprocess(image=image)
        image = torch.tensor(image, dtype=torch.float32, device="cpu")
    else:
        image = _coerce_image_input(image_path)
    image = _normalize_image_tensor(image)
    image = _scale_to_255(image)
    image = image.to(device=device)
    image = resize_and_center_crop(image, height, width, resample=resample)
    image = normalize_latent(image, device, dtype)
    return image


def load_video_conditioning(
    video_path: str | torch.Tensor | np.ndarray | Image.Image,
    height: int,
    width: int,
    frame_cap: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Loads a video from a path or tensor and preprocesses it for conditioning.
    Note: The video is resized to the nearest multiple of 2 for compatibility with video codecs.
    Tensor, array and image inputs are prepared on ``device`` by chunks of frames and returned in RAM:
    pass ``device`` to ``encode_video`` so that each VAE tile is uploaded on its own.
    """
    if isinstance(video_path, str):
        frames = decode_video_from_file(path=video_path, frame_cap=frame_cap, device=device)
        result = None
        for f in frames:
            frame = resize_and_center_crop(f.to(torch.float32), height, width)
            frame = normalize_latent(frame, device, dtype)
            result = frame if result is None else torch.cat([result, frame], dim=2)
        return result

    uint8_guide = isinstance(video_path, UInt8Guide)
    if uint8_guide:
        video_path = video_path.video
    video = _normalize_video_tensor(video_path.detach() if torch.is_tensor(video_path) else _coerce_video_input(video_path))
    if frame_cap is not None and video.shape[0] > frame_cap:
        video = video[:frame_cap]
    value_range = (-1.0, 1.0) if uint8_guide else (float(video.min()), float(video.max()))
    result = None
    for start in range(0, video.shape[0], CONDITIONING_CHUNK_FRAMES):
        chunk = video[start : start + CONDITIONING_CHUNK_FRAMES]
        chunk = _scale_to_255((guide_to_float(chunk) if uint8_guide else chunk).to(device=device, dtype=torch.float32), value_range)
        chunk = normalize_latent(resize_and_center_crop(chunk, height, width), device, dtype)
        if result is None:
            result = torch.empty((*chunk.shape[:2], video.shape[0], *chunk.shape[3:]), dtype=dtype, device="cpu")
        transfer.copy_(result[:, :, start : start + chunk.shape[2]], chunk)
    return result


def decode_image(image_path: str) -> np.ndarray:
    image = Image.open(image_path)
    np_array = np.array(image)[..., :3]
    return np_array


def _write_audio(
    container: av.container.Container, audio_stream: av.audio.AudioStream, samples: torch.Tensor, audio_sample_rate: int
) -> None:
    av = _get_av()
    if samples.ndim == 1:
        samples = samples[:, None]

    if samples.shape[1] != 2 and samples.shape[0] == 2:
        samples = samples.T

    if samples.shape[1] != 2:
        raise ValueError(f"Expected samples with 2 channels; got shape {samples.shape}.")

    # Convert to int16 packed for ingestion; resampler converts to encoder fmt.
    if samples.dtype != torch.int16:
        samples = torch.clip(samples, -1.0, 1.0)
        samples = (samples * 32767.0).to(torch.int16)

    frame_in = av.AudioFrame.from_ndarray(
        samples.contiguous().reshape(1, -1).cpu().numpy(),
        format="s16",
        layout="stereo",
    )
    frame_in.sample_rate = audio_sample_rate

    _resample_audio(container, audio_stream, frame_in)


def _prepare_audio_stream(container: av.container.Container, audio_sample_rate: int) -> av.audio.AudioStream:
    """
    Prepare the audio stream for writing.
    """
    audio_stream = container.add_stream("aac", rate=audio_sample_rate)
    audio_stream.codec_context.sample_rate = audio_sample_rate
    audio_stream.codec_context.layout = "stereo"
    audio_stream.codec_context.time_base = Fraction(1, audio_sample_rate)
    return audio_stream


def _resample_audio(
    container: av.container.Container, audio_stream: av.audio.AudioStream, frame_in: av.AudioFrame
) -> None:
    av = _get_av()
    cc = audio_stream.codec_context

    # Use the encoder's format/layout/rate as the *target*
    target_format = cc.format or "fltp"  # AAC → usually fltp
    target_layout = cc.layout or "stereo"
    target_rate = cc.sample_rate or frame_in.sample_rate

    audio_resampler = av.audio.resampler.AudioResampler(
        format=target_format,
        layout=target_layout,
        rate=target_rate,
    )

    audio_next_pts = 0
    for rframe in audio_resampler.resample(frame_in):
        if rframe.pts is None:
            rframe.pts = audio_next_pts
        audio_next_pts += rframe.samples
        rframe.sample_rate = frame_in.sample_rate
        container.mux(audio_stream.encode(rframe))

    # flush audio encoder
    for packet in audio_stream.encode():
        container.mux(packet)


def encode_video(
    video: torch.Tensor | Iterator[torch.Tensor],
    fps: int,
    audio: torch.Tensor | None,
    audio_sample_rate: int | None,
    output_path: str,
    video_chunks_number: int,
) -> None:
    av = _get_av()
    if isinstance(video, torch.Tensor):
        video = iter([video])

    first_chunk = next(video)

    _, height, width, _ = first_chunk.shape

    container = av.open(output_path, mode="w")
    stream = container.add_stream("libx264", rate=int(fps))
    stream.width = width
    stream.height = height
    stream.pix_fmt = "yuv420p"

    if audio is not None:
        if audio_sample_rate is None:
            raise ValueError("audio_sample_rate is required when audio is provided")

        audio_stream = _prepare_audio_stream(container, audio_sample_rate)

    def all_tiles(
        first_chunk: torch.Tensor, tiles_generator: Generator[tuple[torch.Tensor, int], None, None]
    ) -> Generator[tuple[torch.Tensor, int], None, None]:
        yield first_chunk
        yield from tiles_generator

    for video_chunk in tqdm(all_tiles(first_chunk, video), total=video_chunks_number):
        video_chunk_cpu = video_chunk.to("cpu").numpy()
        for frame_array in video_chunk_cpu:
            frame = av.VideoFrame.from_ndarray(frame_array, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

    # Flush encoder
    for packet in stream.encode():
        container.mux(packet)

    if audio is not None:
        _write_audio(container, audio_stream, audio, audio_sample_rate)

    container.close()


def decode_audio_from_file(path: str, device: torch.device) -> torch.Tensor | None:
    av = _get_av()
    container = av.open(path)
    try:
        audio = []
        audio_stream = next(s for s in container.streams if s.type == "audio")
        for frame in container.decode(audio_stream):
            audio.append(torch.tensor(frame.to_ndarray(), dtype=torch.float32, device=device).unsqueeze(0))
        container.close()
        audio = torch.cat(audio)
    except StopIteration:
        audio = None
    finally:
        container.close()

    return audio


def decode_video_from_file(path: str, frame_cap: int, device: DeviceLikeType) -> Generator[torch.Tensor]:
    av = _get_av()
    container = av.open(path)
    try:
        video_stream = next(s for s in container.streams if s.type == "video")
        for frame in container.decode(video_stream):
            tensor = torch.tensor(frame.to_rgb().to_ndarray(), dtype=torch.uint8, device=device).unsqueeze(0)
            yield tensor
            frame_cap = frame_cap - 1
            if frame_cap == 0:
                break
    finally:
        container.close()


def encode_single_frame(output_file: str, image_array: np.ndarray, crf: float) -> None:
    av = _get_av()
    container = av.open(output_file, "w", format="mp4")
    try:
        stream = container.add_stream("libx264", rate=1, options={"crf": str(crf), "preset": "veryfast"})
        # Round to nearest multiple of 2 for compatibility with video codecs
        height = image_array.shape[0] // 2 * 2
        width = image_array.shape[1] // 2 * 2
        image_array = image_array[:height, :width]
        stream.height = height
        stream.width = width
        av_frame = av.VideoFrame.from_ndarray(image_array, format="rgb24").reformat(format="yuv420p")
        container.mux(stream.encode(av_frame))
        container.mux(stream.encode())
    finally:
        container.close()


def decode_single_frame(video_file: str) -> np.array:
    av = _get_av()
    container = av.open(video_file)
    try:
        stream = next(s for s in container.streams if s.type == "video")
        frame = next(container.decode(stream))
    finally:
        container.close()
    return frame.to_ndarray(format="rgb24")


def preprocess(image: np.array, crf: float = DEFAULT_IMAGE_CRF) -> np.array:
    if crf == 0:
        return image

    with BytesIO() as output_file:
        encode_single_frame(output_file, image, crf)
        video_bytes = output_file.getvalue()
    with BytesIO(video_bytes) as video_file:
        image_array = decode_single_frame(video_file)
    return image_array
