# Copyright (c) 2026 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import subprocess
from typing import Optional

import numpy as np
import torch
import whisper
from einops import rearrange
from PIL import Image
from scipy.signal import resample
from torch.nn import functional as F

from inference.utils import print_rank_0
from shared.utils.subprocess_stdin import child_kwargs


def merge_video_and_audio(video_path: str, audio_path: str, save_path: str):
    # Merge video with audio and keep the shortest stream.
    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-i",
        audio_path,
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-y",
        save_path,
        "-loglevel",
        "error",
    ]
    try:
        subprocess.run(cmd, check=True, **child_kwargs())
        os.remove(video_path)
        os.remove(audio_path)
    except subprocess.CalledProcessError as e:
        print_rank_0(f"ffmpeg failed: {e}")


def upsample_video(video_np: np.ndarray, width: int, height: int, upsample_mode: str = "bilinear") -> np.ndarray:
    """
    Upsample video NumPy array to specified resolution.

    This function assumes the input NumPy array is the result of VAE decoding,
    with data type uint8 and dimension order (T, H, W, C).

    Args:
        video_np (np.ndarray): Input video array with shape (T, H, W, C),
                               data type uint8.
        width (int): Target width.
        height (int): Target height.
        upsample_mode (str): Upsampling mode. Supports "bilinear", "nearest", "bicubic".
                            Defaults to "bilinear".

    Returns:
        np.ndarray: Upsampled video array with shape (T, height, width, C),
                    data type uint8.
    """
    assert upsample_mode in ["bilinear", "nearest", "bicubic"], "Supported upsample modes: bilinear, nearest, bicubic"
    
     # 1. Convert NumPy array to PyTorch tensor
    video_tensor = torch.from_numpy(video_np)

    # 2. Convert from uint8 to float32 and normalize to [0, 1]
    #    F.interpolate works better on floating point numbers
    if video_tensor.dtype == torch.uint8:
        video_tensor = video_tensor.float() / 255.0

    # 3. Adjust dimension order to match F.interpolate requirements (T, H, W, C) -> (C, T, H, W)
    video_tensor = rearrange(video_tensor, "t h w c -> c t h w")
    
    # 4. Use F.interpolate for upsampling
    #    Note: interpolate operates on spatial dimensions (H, W), so size=(height, width)
    upsampled_tensor = F.interpolate(
        video_tensor,
        size=(height, width),
        mode=upsample_mode,
        align_corners=False if upsample_mode in ["bilinear", "bicubic"] else None,
    )

    # 5. Adjust dimension order back (C, T, H, W) -> (T, H, W, C)
    upsampled_tensor = rearrange(upsampled_tensor, "c t h w -> t h w c")
    
    # 6. Convert data from [0, 1] range back to [0, 255] and convert to uint8
    upsampled_tensor = (upsampled_tensor.clamp(0, 1) * 255).byte()
    
    # 7. Convert PyTorch tensor back to NumPy array
    return upsampled_tensor.numpy()


def resizecrop(image: Image.Image, th: int, tw: int) -> Image.Image:
    w, h = image.size
    if w == tw and h == th:
        return image
    if h / w > th / tw:
        new_w = int(w)
        new_h = int(new_w * th / tw)
    else:
        new_h = int(h)
        new_w = int(new_h * tw / th)
    left = (w - new_w) / 2
    top = (h - new_h) / 2
    right = (w + new_w) / 2
    bottom = (h + new_h) / 2
    return image.crop((left, top, right, bottom))


def resample_audio_sinc(audio: torch.Tensor, time_stretching: float):
    print_rank_0(f"before resample audio: {audio.shape}")
    new_length = int(audio.shape[0] * time_stretching)
    audio = resample(audio, new_length)
    print_rank_0(f"after resample audio: {audio.shape}")
    return audio


def merge_overlapping_vae_features(audio_feats, overlap_ratio=0.5):
    if not audio_feats:
        return None
    if len(audio_feats) == 1:
        return audio_feats[0]

    batch_size, total_frames, feature_dim = audio_feats[0].shape
    overlap_frames = int(total_frames * overlap_ratio)
    step_frames = total_frames - overlap_frames
    final_length = (len(audio_feats) - 1) * step_frames + total_frames
    output_feat = torch.zeros(batch_size, final_length, feature_dim, device=audio_feats[0].device, dtype=audio_feats[0].dtype)

    for block_idx, current_feat in enumerate(audio_feats):
        output_start = block_idx * step_frames
        if block_idx == 0:
            output_feat[:, output_start : output_start + total_frames, :] = current_feat
            continue

        non_overlap_start = output_start + overlap_frames
        non_overlap_end = output_start + total_frames
        output_feat[:, non_overlap_start:non_overlap_end, :] = current_feat[:, overlap_frames:, :]

        for frame_idx in range(overlap_frames):
            output_pos = output_start + frame_idx
            prev_weight = (overlap_frames - frame_idx) / overlap_frames
            curr_weight = frame_idx / overlap_frames
            output_feat[:, output_pos, :] = (
                prev_weight * output_feat[:, output_pos, :] + curr_weight * current_feat[:, frame_idx, :]
            )
    return output_feat


def load_audio_and_encode(audio_vae: any, audio_path: str, seconds: Optional[int] = None) -> torch.Tensor:
    """Load and encode audio using the provided audio VAE."""
    sample_rate = 51200
    audio_chunk_duration = 29
    overlap_ratio = 0.5

    audio_full = whisper.load_audio(audio_path, sr=sample_rate)
    if seconds is not None:
        audio_full = audio_full[: min(int(seconds * sample_rate), audio_full.shape[0])]
    total_samples = audio_full.shape[0]

    window_size = int(audio_chunk_duration * sample_rate)
    step_size = int(window_size * (1 - overlap_ratio))
    if total_samples <= window_size:
        audio = torch.from_numpy(audio_full).cuda()
        audio = audio.unsqueeze(0).expand(2, -1)
        return audio_vae.vae_model.encode(audio)

    encoded_chunks = []
    latent_to_audio_ratio = None
    for offset_start in range(0, total_samples, step_size):
        offset_end = min(offset_start + window_size, total_samples)
        chunk = whisper.pad_or_trim(audio_full[offset_start:offset_end], length=window_size)
        chunk_tensor = torch.from_numpy(chunk).cuda().unsqueeze(0).expand(2, -1)
        encoded_chunk = audio_vae.vae_model.encode(chunk_tensor)

        if latent_to_audio_ratio is None:
            latent_to_audio_ratio = encoded_chunk.shape[-1] / window_size

        encoded_chunks.append(encoded_chunk.permute(0, 2, 1))
        if offset_end >= total_samples:
            break

    final_feat = merge_overlapping_vae_features(encoded_chunks, overlap_ratio=overlap_ratio).permute(0, 2, 1)
    final_target_len = math.ceil(total_samples * latent_to_audio_ratio)
    return final_feat[:, :, :final_target_len]
