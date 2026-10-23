import os
import json
import librosa
import binascii
import imageio
import subprocess
import numpy as np
import os.path as osp
from tqdm import tqdm
import pyloudnorm as pyln
from einops import rearrange
import scipy.signal as ss

import torch
import torch.nn.functional as F
import torchvision

import gc
from shared.utils.subprocess_stdin import child_kwargs

def torch_gc():
    gc.collect()


def linear_interpolation(features, seq_len):
    features = features.transpose(1, 2)
    output_features = F.interpolate(features, size=seq_len, align_corners=True, mode='linear')
    return output_features.transpose(1, 2)


def calculate_x_ref_attn_map(qk_list, ref_target_masks, attn_bias=None):

    # compute cross-reference attention maps between query features and reference key features.
    noise_q, ref_k = qk_list
    ref_k = ref_k.to(noise_q.dtype).to(noise_q.device)
    scale = 1.0 / noise_q.shape[-1] ** 0.5
    noise_q = noise_q * scale
    noise_q = noise_q.transpose(1, 2)
    ref_k = ref_k.transpose(1, 2)
    attn = noise_q @ ref_k.transpose(-2, -1)

    if attn_bias is not None:
        attn = attn + attn_bias

    x_ref_attn_map_source = attn.softmax(-1)

    x_ref_attn_maps = []
    ref_target_masks = ref_target_masks.to(noise_q.dtype)
    x_ref_attn_map_source = x_ref_attn_map_source.to(noise_q.dtype)

    for _, ref_target_mask in enumerate(ref_target_masks):
        ref_target_mask = ref_target_mask[None, None, None, ...]
        x_ref_attn_map = x_ref_attn_map_source.clone()
        x_ref_attn_map = x_ref_attn_map * ref_target_mask
        x_ref_attn_map = x_ref_attn_map.sum(-1) / ref_target_mask.sum() 
        x_ref_attn_map = x_ref_attn_map.permute(0, 2, 1) 
        x_ref_attn_map = x_ref_attn_map.mean(-1) 
        
        x_ref_attn_maps.append(x_ref_attn_map)
    
    qk_list[:] = []
    del attn
    del x_ref_attn_map_source

    return torch.concat(x_ref_attn_maps, dim=0)


def get_attn_map_with_target(noise_q, key, shape, ref_target_masks=None, split_num=2, cp_split_hw=None):
    
    N_t, N_h, N_w = shape
    x_seqlens = N_h * N_w
    ref_k = key[:, :x_seqlens]
    noise_q = noise_q.contiguous()

    _, seq_lens, heads, _ = noise_q.shape
    class_num, _ = ref_target_masks.shape
    x_ref_attn_maps = torch.zeros(class_num, seq_lens).to(noise_q.device).to(noise_q.dtype)

    split_chunk = heads // split_num
    
    # calculate attn map within each group and take the mean
    for i in range(split_num):
        qk_list = [
            noise_q[:, :, i * split_chunk:(i + 1) * split_chunk, :],
            ref_k[:, :, i * split_chunk:(i + 1) * split_chunk, :],
        ]
        x_ref_attn_maps_perhead = calculate_x_ref_attn_map(qk_list, ref_target_masks)
        x_ref_attn_maps += x_ref_attn_maps_perhead
    
    return x_ref_attn_maps / split_num


def rand_name(length=8, suffix=''):
    name = binascii.b2a_hex(os.urandom(length)).decode('utf-8')
    if suffix:
        if not suffix.startswith('.'):
            suffix = '.' + suffix
        name += suffix
    return name

def cache_video(tensor,
                save_file=None,
                fps=30,
                suffix='.mp4',
                nrow=8,
                normalize=True,
                value_range=(-1, 1),
                retry=5):
    
    # cache file
    cache_file = osp.join('/tmp', rand_name(
        suffix=suffix)) if save_file is None else save_file

    # save to cache
    error = None
    for _ in range(retry):
       
        # preprocess
        tensor = tensor.clamp(min(value_range), max(value_range))
        tensor = torch.stack([
                torchvision.utils.make_grid(
                    u, nrow=nrow, normalize=normalize, value_range=value_range)
                for u in tensor.unbind(2)
            ],
                                 dim=1).permute(1, 2, 3, 0)
        tensor = (tensor * 255).type(torch.uint8).cpu()

        # write video
        writer = imageio.get_writer(cache_file, fps=fps, codec='libx264', quality=10, ffmpeg_params=["-crf", "10"])
        for frame in tensor.numpy():
            writer.append_data(frame)
        writer.close()
        return cache_file

def get_audio_duration(audio_path):
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_entries", "format=duration",
        audio_path,
    ]
    out = subprocess.check_output(cmd, **child_kwargs())
    info = json.loads(out)
    return float(info["format"]["duration"])


def save_video_ffmpeg(gen_video_samples, save_path, audio_path, fps=25, quality=5, high_quality_save=False):

    def save_video(frames, save_path, fps, quality=9, ffmpeg_params=None):
        writer = imageio.get_writer(
            save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params
        )
        for frame in tqdm(frames, desc="Saving video"):
            frame = np.array(frame)
            writer.append_data(frame)
        writer.close()
    save_path_tmp = save_path + "-temp.mp4"
    
    os.makedirs(os.path.dirname(save_path_tmp), exist_ok=True)
    video_audio =  gen_video_samples.cpu().numpy()
    video_audio = np.clip(video_audio, 0, 255).astype(np.uint8)
    save_video(video_audio, save_path_tmp, fps=fps, quality=quality)

    # crop audio according to video length
    T, _, _, _ = gen_video_samples.shape
    duration = T / fps
    save_path_crop_audio = save_path + "-cropaudio.wav"
    final_command = [
        "ffmpeg",
        "-y",
        "-i",
        audio_path,
        "-t",
        f'{duration}',
        save_path_crop_audio,
    ]
    subprocess.run(final_command, check=True, **child_kwargs())

    # crop video according to audio length
    crop_audio_duration = get_audio_duration(save_path_crop_audio)
    save_path_crop_tmp = save_path + "-cropvideo.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-i", save_path_tmp,
        "-t", f"{crop_audio_duration}",
        "-c:v", "copy",
        "-c:a", "copy",
        save_path_crop_tmp,
    ]
    subprocess.run(cmd, check=True, **child_kwargs())

    # generate video with audio
    save_path = save_path + ".mp4"
    if high_quality_save:
        final_command = [
            "ffmpeg",
            "-y",
            "-i", save_path_crop_tmp,
            "-i", save_path_crop_audio,
            "-c:v", "libx264",
            "-crf", "0",
            "-preset", "veryslow", 
            "-c:a", "aac",
            "-shortest",
            save_path,
        ]
        subprocess.run(final_command, check=True, **child_kwargs())
    else:
        final_command = [
            "ffmpeg",
            "-y",
            "-i",
            save_path_crop_tmp,
            "-i",
            save_path_crop_audio,
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            save_path,
        ]
        subprocess.run(final_command, check=True, **child_kwargs())
        
    os.remove(save_path_tmp)
    os.remove(save_path_crop_tmp)
    os.remove(save_path_crop_audio)
