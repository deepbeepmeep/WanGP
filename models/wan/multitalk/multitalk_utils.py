import os
from einops import rearrange

import torch
import torch.nn as nn

from einops import rearrange, repeat
from functools import lru_cache
import imageio
import uuid
from tqdm import tqdm
import numpy as np
import subprocess
import soundfile as sf
import torchvision
import binascii
import os.path as osp
from skimage import color
from mmgp.offload import get_cache, clear_caches

VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
ASPECT_RATIO_627 = {
     '0.26': ([320, 1216], 1), '0.38': ([384, 1024], 1), '0.50': ([448, 896], 1), '0.67': ([512, 768], 1), 
     '0.82': ([576, 704], 1),  '1.00': ([640, 640], 1),  '1.22': ([704, 576], 1), '1.50': ([768, 512], 1), 
     '1.86': ([832, 448], 1),  '2.00': ([896, 448], 1),  '2.50': ([960, 384], 1), '2.83': ([1088, 384], 1), 
     '3.60': ([1152, 320], 1), '3.80': ([1216, 320], 1), '4.00': ([1280, 320], 1)}


ASPECT_RATIO_960 = {
     '0.22': ([448, 2048], 1), '0.29': ([512, 1792], 1), '0.36': ([576, 1600], 1), '0.45': ([640, 1408], 1), 
     '0.55': ([704, 1280], 1), '0.63': ([768, 1216], 1), '0.76': ([832, 1088], 1), '0.88': ([896, 1024], 1), 
     '1.00': ([960, 960], 1), '1.14': ([1024, 896], 1), '1.31': ([1088, 832], 1), '1.50': ([1152, 768], 1), 
     '1.58': ([1216, 768], 1), '1.82': ([1280, 704], 1), '1.91': ([1344, 704], 1), '2.20': ([1408, 640], 1), 
     '2.30': ([1472, 640], 1), '2.67': ([1536, 576], 1), '2.89': ([1664, 576], 1), '3.62': ([1856, 512], 1), 
     '3.75': ([1920, 512], 1)}



def torch_gc():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def normalize_and_scale(column, source_range, target_range, epsilon=1e-8):

    source_min, source_max = source_range
    new_min, new_max = target_range
 
    normalized = (column - source_min) / (source_max - source_min + epsilon)
    scaled = normalized * (new_max - new_min) + new_min
    return scaled


# @torch.compile
def calculate_x_ref_attn_map_per_head(visual_q, ref_k, ref_target_masks, ref_images_count, attn_bias=None):
    dtype = visual_q.dtype
    ref_k = ref_k.to(dtype).to(visual_q.device)
    scale = 1.0 / visual_q.shape[-1] ** 0.5
    visual_q = visual_q * scale
    visual_q = visual_q.transpose(1, 2)
    ref_k = ref_k.transpose(1, 2)
    visual_q_shape = visual_q.shape
    visual_q = visual_q.view(-1, visual_q_shape[-1] )
    number_chunks = visual_q_shape[-2]*ref_k.shape[-2] /  53090100 * 2
    chunk_size =  int(visual_q_shape[-2] / number_chunks)
    chunks =torch.split(visual_q, chunk_size)
    maps_lists = [ [] for _ in ref_target_masks]  
    for q_chunk  in chunks:
        attn = q_chunk @ ref_k.transpose(-2, -1)
        x_ref_attn_map_source = attn.softmax(-1) # B, H, x_seqlens, ref_seqlens
        del attn
        ref_target_masks = ref_target_masks.to(device=visual_q.device, dtype=dtype)
        x_ref_attn_map_source = x_ref_attn_map_source.to(dtype)

        for class_idx, ref_target_mask in enumerate(ref_target_masks):
            ref_target_mask = ref_target_mask[None, None, None, ...]
            x_ref_attnmap = x_ref_attn_map_source * ref_target_mask
            x_ref_attnmap = x_ref_attnmap.sum(-1) / ref_target_mask.sum() # B, H, x_seqlens, ref_seqlens --> B, H, x_seqlens
            maps_lists[class_idx].append(x_ref_attnmap)

        del x_ref_attn_map_source

    x_ref_attn_maps = []
    for class_idx, maps_list in enumerate(maps_lists):
        attn_map_fuse = torch.concat(maps_list, dim= -1)
        attn_map_fuse = attn_map_fuse.view(1, visual_q_shape[1], -1).squeeze(1)
        x_ref_attn_maps.append( attn_map_fuse )


    return torch.concat(x_ref_attn_maps, dim=0)

def calculate_x_ref_attn_map(visual_q, ref_k, ref_target_masks, ref_images_count):
    dtype = visual_q.dtype
    ref_k = ref_k.to(dtype).to(visual_q.device)
    scale = 1.0 / visual_q.shape[-1] ** 0.5
    visual_q = visual_q * scale
    visual_q = visual_q.transpose(1, 2)
    ref_k = ref_k.transpose(1, 2)
    attn = visual_q @ ref_k.transpose(-2, -1)

    x_ref_attn_map_source = attn.softmax(-1) # B, H, x_seqlens, ref_seqlens
    del attn
    x_ref_attn_maps = []
    ref_target_masks = ref_target_masks.to(device=visual_q.device, dtype=dtype)
    x_ref_attn_map_source = x_ref_attn_map_source.to(dtype)

    for class_idx, ref_target_mask in enumerate(ref_target_masks):
        ref_target_mask = ref_target_mask[None, None, None, ...]
        x_ref_attnmap = x_ref_attn_map_source * ref_target_mask
        x_ref_attnmap = x_ref_attnmap.sum(-1) / ref_target_mask.sum() # B, H, x_seqlens, ref_seqlens --> B, H, x_seqlens
        x_ref_attnmap = x_ref_attnmap.permute(0, 2, 1) # B, x_seqlens, H
        x_ref_attnmap = x_ref_attnmap.mean(-1) # B, x_seqlens       (mean of heads)
        x_ref_attn_maps.append(x_ref_attnmap)
    
    del x_ref_attn_map_source

    return torch.concat(x_ref_attn_maps, dim=0)

def get_attn_map_with_target(visual_q, ref_k, shape, ref_target_masks=None, split_num=10, ref_images_count = 0):
    """Args:
        query (torch.tensor): B M H K
        key (torch.tensor): B M H K
        shape (tuple): (N_t, N_h, N_w)
        ref_target_masks: [B, N_h * N_w]
    """

    N_t, N_h, N_w = shape
    
    x_seqlens = N_h * N_w
    if x_seqlens <= 1508:
        split_num = 10 # 540p
    else:
        split_num = 20 if x_seqlens <= 3600 else 40 # 720p / 1080p

    ref_k     = ref_k[:, :x_seqlens]
    if ref_images_count > 0 :
        visual_q_shape = visual_q.shape 
        visual_q = visual_q.reshape(visual_q_shape[0], N_t, -1)
        visual_q = visual_q[:, ref_images_count:]
        visual_q = visual_q.reshape(visual_q_shape[0], -1, *visual_q_shape[-2:])

    _, seq_lens, heads, _ = visual_q.shape
    class_num, _ = ref_target_masks.shape
    x_ref_attn_maps = torch.zeros(class_num, seq_lens, dtype=visual_q.dtype, device=visual_q.device)

    split_chunk = heads // split_num
    
    if split_chunk == 1:
        for i in range(split_num):
            x_ref_attn_maps_perhead = calculate_x_ref_attn_map_per_head(visual_q[:, :, i:(i+1), :], ref_k[:, :, i:(i+1), :], ref_target_masks, ref_images_count)
            x_ref_attn_maps += x_ref_attn_maps_perhead
    else:
        for i in range(split_num):
            x_ref_attn_maps_perhead = calculate_x_ref_attn_map(visual_q[:, :, i*split_chunk:(i+1)*split_chunk, :], ref_k[:, :, i*split_chunk:(i+1)*split_chunk, :], ref_target_masks, ref_images_count)
            x_ref_attn_maps += x_ref_attn_maps_perhead
    
    x_ref_attn_maps /= split_num
    return x_ref_attn_maps


def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")


class RotaryPositionalEmbedding1D(nn.Module):

    def __init__(self,
                 head_dim,
                 ):
        super().__init__()
        self.head_dim = head_dim
        self.base = 10000


    def precompute_freqs_cis_1d(self, pos_indices):

        freqs = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2)[: (self.head_dim // 2)].float() / self.head_dim))
        freqs = freqs.to(pos_indices.device)
        freqs = torch.einsum("..., f -> ... f", pos_indices.float(), freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)
        return freqs

    def forward(self, qlist, pos_indices, cache_entry = None):
        """1D RoPE.

        Args:
            query (torch.tensor): [B, head, seq, head_dim]
            pos_indices (torch.tensor): [seq,]
        Returns:
            query with the same shape as input.
        """
        xq= qlist[0]
        qlist.clear()
        cache = get_cache("multitalk_rope")
        freqs_cis= cache.get(cache_entry, None)
        if freqs_cis is None:
            freqs_cis = cache[cache_entry] = self.precompute_freqs_cis_1d(pos_indices)
        cos, sin = freqs_cis.cos().unsqueeze(0).unsqueeze(0), freqs_cis.sin().unsqueeze(0).unsqueeze(0)
        # cos, sin = rearrange(cos, 'n d -> 1 1 n d'), rearrange(sin, 'n d -> 1 1 n d')
        # real * cos - imag * sin
        # imag * cos + real * sin
        xq_dtype = xq.dtype
        xq_out = xq.to(torch.float)
        xq = None        
        xq_rot = rotate_half(xq_out)
        xq_out *= cos
        xq_rot *= sin
        xq_out += xq_rot
        del xq_rot
        xq_out = xq_out.to(xq_dtype)
        return xq_out 
    


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

def save_video_ffmpeg(gen_video_samples, save_path, vocal_audio_list, fps=25, quality=5, high_quality_save=False):
    
    def save_video(frames, save_path, fps, quality=9, ffmpeg_params=None):
        writer = imageio.get_writer(
            save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params
        )
        for frame in tqdm(frames, desc="Saving video"):
            frame = np.array(frame)
            writer.append_data(frame)
        writer.close()
    save_path_tmp = save_path + "-temp.mp4"

    if high_quality_save:
        cache_video(
                    tensor=gen_video_samples.unsqueeze(0),
                    save_file=save_path_tmp,
                    fps=fps,
                    nrow=1,
                    normalize=True,
                    value_range=(-1, 1)
                    )
    else:
        video_audio = (gen_video_samples+1)/2 # C T H W
        video_audio = video_audio.permute(1, 2, 3, 0).cpu().numpy()
        video_audio = np.clip(video_audio * 255, 0, 255).astype(np.uint8)  # to [0, 255]
        save_video(video_audio, save_path_tmp, fps=fps, quality=quality)


    # crop audio according to video length
    _, T, _, _ = gen_video_samples.shape
    duration = T / fps
    save_path_crop_audio = save_path + "-cropaudio.wav"
    final_command = [
        "ffmpeg",
        "-i",
        vocal_audio_list[0],
        "-t",
        f'{duration}',
        save_path_crop_audio,
    ]
    subprocess.run(final_command, check=True)

    save_path = save_path + ".mp4"
    if high_quality_save:
        final_command = [
            "ffmpeg",
            "-y",
            "-i", save_path_tmp,
            "-i", save_path_crop_audio,
            "-c:v", "libx264",
            "-crf", "0",
            "-preset", "veryslow",
            "-c:a", "aac", 
            "-shortest",
            save_path,
        ]
        subprocess.run(final_command, check=True)
        os.remove(save_path_tmp)
        os.remove(save_path_crop_audio)
    else:
        final_command = [
            "ffmpeg",
            "-y",
            "-i",
            save_path_tmp,
            "-i",
            save_path_crop_audio,
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            save_path,
        ]
        subprocess.run(final_command, check=True)
        os.remove(save_path_tmp)
        os.remove(save_path_crop_audio)


class MomentumBuffer:
    def __init__(self, momentum: float): 
        self.momentum = momentum 
        self.running_average = 0 
    
    def update(self, update_value: torch.Tensor): 
        new_average = self.momentum * self.running_average 
        self.running_average = update_value + new_average
    


def project( 
        v0: torch.Tensor, # [B, C, T, H, W] 
        v1: torch.Tensor, # [B, C, T, H, W] 
        ): 
    dtype = v0.dtype 
    v0, v1 = v0.double(), v1.double() 
    v1 = torch.nn.functional.normalize(v1, dim=[-1, -2, -3, -4]) 
    v0_parallel = (v0 * v1).sum(dim=[-1, -2, -3, -4], keepdim=True) * v1 
    v0_orthogonal = v0 - v0_parallel
    return v0_parallel.to(dtype), v0_orthogonal.to(dtype)


def adaptive_projected_guidance( 
          diff: torch.Tensor, # [B, C, T, H, W] 
          pred_cond: torch.Tensor, # [B, C, T, H, W] 
          momentum_buffer: MomentumBuffer = None, 
          eta: float = 0.0,
          norm_threshold: float = 55,
          ): 
    if momentum_buffer is not None: 
        momentum_buffer.update(diff) 
        diff = momentum_buffer.running_average
    if norm_threshold > 0: 
        ones = torch.ones_like(diff) 
        diff_norm = diff.norm(p=2, dim=[-1, -2, -3, -4], keepdim=True) 
        print(f"diff_norm: {diff_norm}")
        scale_factor = torch.minimum(ones, norm_threshold / diff_norm) 
        diff = diff * scale_factor 
    diff_parallel, diff_orthogonal = project(diff, pred_cond) 
    normalized_update = diff_orthogonal + eta * diff_parallel
    return normalized_update

def match_and_blend_colors(source_chunk: torch.Tensor, reference_image: torch.Tensor, strength: float) -> torch.Tensor:
    """
    Matches the color of a source video chunk to a reference image and blends with the original.

    Args:
        source_chunk (torch.Tensor): The video chunk to be color-corrected (B, C, T, H, W) in range [-1, 1].
                                     Assumes B=1 (batch size of 1).
        reference_image (torch.Tensor): The reference image (B, C, 1, H, W) in range [-1, 1].
                                        Assumes B=1 and T=1 (single reference frame).
        strength (float): The strength of the color correction (0.0 to 1.0).
                          0.0 means no correction, 1.0 means full correction.

    Returns:
        torch.Tensor: The color-corrected and blended video chunk.
    """
    # print(f"[match_and_blend_colors] Input source_chunk shape: {source_chunk.shape}, reference_image shape: {reference_image.shape}, strength: {strength}")

    if strength == 0.0:
        # print(f"[match_and_blend_colors] Strength is 0, returning original source_chunk.")
        return source_chunk

    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"Strength must be between 0.0 and 1.0, got {strength}")

    device = source_chunk.device
    dtype = source_chunk.dtype

    # Squeeze batch dimension, permute to T, H, W, C for skimage
    # Source: (1, C, T, H, W) -> (T, H, W, C)
    source_np = source_chunk.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()
    # Reference: (1, C, 1, H, W) -> (H, W, C)
    ref_np = reference_image.squeeze(0).squeeze(1).permute(1, 2, 0).cpu().numpy() # Squeeze T dimension as well

    # Normalize from [-1, 1] to [0, 1] for skimage
    source_np_01 = (source_np + 1.0) / 2.0
    ref_np_01 = (ref_np + 1.0) / 2.0

    # Clip to ensure values are strictly in [0, 1] after potential float precision issues
    source_np_01 = np.clip(source_np_01, 0.0, 1.0)
    ref_np_01 = np.clip(ref_np_01, 0.0, 1.0)

    # Convert reference to Lab
    try:
        ref_lab = color.rgb2lab(ref_np_01)
    except ValueError as e:
        # Handle potential errors if image data is not valid for conversion
        print(f"Warning: Could not convert reference image to Lab: {e}. Skipping color correction for this chunk.")
        return source_chunk


    corrected_frames_np_01 = []
    for i in range(source_np_01.shape[0]): # Iterate over time (T)
        source_frame_rgb_01 = source_np_01[i]
        
        try:
            source_lab = color.rgb2lab(source_frame_rgb_01)
        except ValueError as e:
            print(f"Warning: Could not convert source frame {i} to Lab: {e}. Using original frame.")
            corrected_frames_np_01.append(source_frame_rgb_01)
            continue

        corrected_lab_frame = source_lab.copy()

        # Perform color transfer for L, a, b channels
        for j in range(3): # L, a, b
            mean_src, std_src = source_lab[:, :, j].mean(), source_lab[:, :, j].std()
            mean_ref, std_ref = ref_lab[:, :, j].mean(), ref_lab[:, :, j].std()

            # Avoid division by zero if std_src is 0
            if std_src == 0:
                # If source channel has no variation, keep it as is, but shift by reference mean
                # This case is debatable, could also just copy source or target mean.
                # Shifting by target mean helps if source is flat but target isn't.
                corrected_lab_frame[:, :, j] = mean_ref 
            else:
                corrected_lab_frame[:, :, j] = (corrected_lab_frame[:, :, j] - mean_src) * (std_ref / std_src) + mean_ref
        
        try:
            fully_corrected_frame_rgb_01 = color.lab2rgb(corrected_lab_frame)
        except ValueError as e:
            print(f"Warning: Could not convert corrected frame {i} back to RGB: {e}. Using original frame.")
            corrected_frames_np_01.append(source_frame_rgb_01)
            continue
            
        # Clip again after lab2rgb as it can go slightly out of [0,1]
        fully_corrected_frame_rgb_01 = np.clip(fully_corrected_frame_rgb_01, 0.0, 1.0)

        # Blend with original source frame (in [0,1] RGB)
        blended_frame_rgb_01 = (1 - strength) * source_frame_rgb_01 + strength * fully_corrected_frame_rgb_01
        corrected_frames_np_01.append(blended_frame_rgb_01)

    corrected_chunk_np_01 = np.stack(corrected_frames_np_01, axis=0)

    # Convert back to [-1, 1]
    corrected_chunk_np_minus1_1 = (corrected_chunk_np_01 * 2.0) - 1.0

    # Permute back to (C, T, H, W), add batch dim, and convert to original torch.Tensor type and device
    # (T, H, W, C) -> (C, T, H, W)
    corrected_chunk_tensor = torch.from_numpy(corrected_chunk_np_minus1_1).permute(3, 0, 1, 2).unsqueeze(0)
    corrected_chunk_tensor = corrected_chunk_tensor.contiguous() # Ensure contiguous memory layout
    output_tensor = corrected_chunk_tensor.to(device=device, dtype=dtype)
    # print(f"[match_and_blend_colors] Output tensor shape: {output_tensor.shape}")
    return output_tensor


from skimage import color
from scipy import ndimage
from scipy.ndimage import binary_erosion, distance_transform_edt


def match_and_blend_colors_with_mask(
    source_chunk: torch.Tensor, 
    reference_video: torch.Tensor, 
    mask: torch.Tensor,
    strength: float,
    copy_mode: str = "corrected",  # "corrected", "reference", "source", "progressive_blend"
    source_border_distance: int = 10,
    reference_border_distance: int = 10
) -> torch.Tensor:
    """
    Matches the color of a source video chunk to a reference video using mask-based region sampling.

    Args:
        source_chunk (torch.Tensor): The video chunk to be color-corrected (B, C, T, H, W) in range [-1, 1].
                                     Assumes B=1 (batch size of 1).
        reference_video (torch.Tensor): The reference video (B, C, T, H, W) in range [-1, 1].
                                        Must have same temporal dimension as source_chunk.
        mask (torch.Tensor): Binary mask (B, 1, T, H, W) or (T, H, W) or (H, W) with values 0 and 1.
                            Color correction is applied to pixels where mask=1.
        strength (float): The strength of the color correction (0.0 to 1.0).
                          0.0 means no correction, 1.0 means full correction.
        copy_mode (str): What to do with mask=0 pixels: 
                        "corrected" (keep original), "reference", "source", 
                        "progressive_blend" (double-sided progressive blending near borders).
        source_border_distance (int): Distance in pixels from mask border to sample source video (mask=1 side).
        reference_border_distance (int): Distance in pixels from mask border to sample reference video (mask=0 side).
                                         For "progressive_blend" mode, this also defines the blending falloff distance.

    Returns:
        torch.Tensor: The color-corrected and blended video chunk.
        
    Notes:
        - Color statistics are sampled from border regions to determine source and reference tints
        - Progressive blending creates smooth double-sided transitions:
          * mask=1 side: 60% source + 40% reference at border → 100% source deeper in
          * mask=0 side: 60% reference + 40% source at border → 100% reference deeper in
    """
    
    if strength == 0.0:
        return source_chunk

    if not 0.0 <= strength <= 1.0:
        raise ValueError(f"Strength must be between 0.0 and 1.0, got {strength}")
    
    if copy_mode not in ["corrected", "reference", "source", "progressive_blend"]:
        raise ValueError(f"copy_mode must be 'corrected', 'reference', 'source', or 'progressive_blend', got {copy_mode}")

    device = source_chunk.device
    dtype = source_chunk.dtype
    B, C, T, H, W = source_chunk.shape

    # Handle different mask dimensions
    if mask.dim() == 2:  # (H, W)
        mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(B, 1, T, H, W)
    elif mask.dim() == 3:  # (T, H, W)
        mask = mask.unsqueeze(0).unsqueeze(0).expand(B, 1, T, H, W)
    elif mask.dim() == 4:  # (B, T, H, W) - missing channel dim
        mask = mask.unsqueeze(1)
    # mask should now be (B, 1, T, H, W)

    # Convert to numpy for processing
    source_np = source_chunk.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()  # (T, H, W, C)
    reference_np = reference_video.squeeze(0).permute(1, 2, 3, 0).cpu().numpy()  # (T, H, W, C)
    mask_np = mask.squeeze(0).squeeze(0).cpu().numpy()  # (T, H, W)

    # Normalize from [-1, 1] to [0, 1] for skimage
    source_np_01 = (source_np + 1.0) / 2.0
    reference_np_01 = (reference_np + 1.0) / 2.0
    
    # Clip to ensure values are in [0, 1]
    source_np_01 = np.clip(source_np_01, 0.0, 1.0)
    reference_np_01 = np.clip(reference_np_01, 0.0, 1.0)

    corrected_frames_np_01 = []
    
    for t in range(T):
        source_frame = source_np_01[t]  # (H, W, C)
        reference_frame = reference_np_01[t]  # (H, W, C)
        frame_mask = mask_np[t]  # (H, W)
        
        # Find mask borders and create distance maps
        border_regions = get_border_sampling_regions(frame_mask, source_border_distance, reference_border_distance)
        source_sample_region = border_regions['source_region']  # mask=1 side
        reference_sample_region = border_regions['reference_region']  # mask=0 side
        
        # Sample pixels for color statistics
        try:
            source_stats = compute_color_stats(source_frame, source_sample_region)
            reference_stats = compute_color_stats(reference_frame, reference_sample_region)
        except ValueError as e:
            print(f"Warning: Could not compute color statistics for frame {t}: {e}. Using original frame.")
            corrected_frames_np_01.append(source_frame)
            continue
        
        # Apply color correction to mask=1 area and handle mask=0 area based on copy_mode
        corrected_frame = apply_color_correction_with_mask(
            source_frame, frame_mask, source_stats, reference_stats, strength
        )
        
        # Handle mask=0 pixels based on copy_mode
        if copy_mode == "reference":
            corrected_frame = apply_copy_with_mask(corrected_frame, reference_frame, frame_mask, "reference")
        elif copy_mode == "source":
            corrected_frame = apply_copy_with_mask(corrected_frame, source_frame, frame_mask, "source")
        elif copy_mode == "progressive_blend":
            # Apply progressive blending in mask=1 border area (source side)
            corrected_frame = apply_progressive_blend_in_corrected_area(
                corrected_frame, reference_frame, frame_mask, 
                border_regions['source_region'], border_regions['source_distances'], 
                border_regions['reference_region'], source_border_distance
            )
            # Copy reference pixels to mask=0 area first
            corrected_frame = apply_copy_with_mask(corrected_frame, reference_frame, frame_mask, "reference")
            # Then apply progressive blending in mask=0 border area (reference side)
            corrected_frame = apply_progressive_blend_in_reference_area(
                corrected_frame, source_frame, frame_mask, 
                border_regions['reference_region'], border_regions['reference_distances'], 
                reference_border_distance
            )
            
        corrected_frames_np_01.append(corrected_frame)

    corrected_chunk_np_01 = np.stack(corrected_frames_np_01, axis=0)

    # Convert back to [-1, 1] and return to tensor format
    corrected_chunk_np_minus1_1 = (corrected_chunk_np_01 * 2.0) - 1.0
    corrected_chunk_tensor = torch.from_numpy(corrected_chunk_np_minus1_1).permute(3, 0, 1, 2).unsqueeze(0)
    corrected_chunk_tensor = corrected_chunk_tensor.contiguous()
    output_tensor = corrected_chunk_tensor.to(device=device, dtype=dtype)
    
    return output_tensor


def get_border_sampling_regions(mask, source_border_distance, reference_border_distance):
    """
    Create regions for sampling near mask borders with separate distances for source and reference.
    
    Args:
        mask: Binary mask (H, W) with 0s and 1s
        source_border_distance: Distance from border to include in source sampling (mask=1 side)
        reference_border_distance: Distance from border to include in reference sampling (mask=0 side)
        
    Returns:
        Dict with sampling regions and distance maps for blending
    """
    # Convert to boolean for safety
    mask_bool = mask.astype(bool)
    
    # Distance from mask=0 regions (distance into mask=1 areas from border)
    dist_from_mask0 = distance_transform_edt(mask_bool)
    
    # Distance from mask=1 regions (distance into mask=0 areas from border)  
    dist_from_mask1 = distance_transform_edt(~mask_bool)
    
    # Source region: mask=1 pixels within source_border_distance of mask=0 pixels
    source_region = mask_bool & (dist_from_mask0 <= source_border_distance)
    
    # Reference region: mask=0 pixels within reference_border_distance of mask=1 pixels
    reference_region = (~mask_bool) & (dist_from_mask1 <= reference_border_distance)
    
    return {
        'source_region': source_region,
        'reference_region': reference_region,
        'source_distances': dist_from_mask0,  # Distance into mask=1 from border
        'reference_distances': dist_from_mask1  # Distance into mask=0 from border
    }


def compute_color_stats(image, sample_region):
    """
    Compute color statistics (mean and std) for Lab channels in the sampling region.
    
    Args:
        image: RGB image (H, W, C) in range [0, 1]
        sample_region: Boolean mask (H, W) indicating pixels to sample
        
    Returns:
        Dict with 'mean' and 'std' for Lab components
    """
    if not np.any(sample_region):
        raise ValueError("No pixels in sampling region")
    
    # Convert to Lab
    try:
        image_lab = color.rgb2lab(image)
    except ValueError as e:
        raise ValueError(f"Could not convert image to Lab: {e}")
    
    # Extract pixels in sampling region
    sampled_pixels = image_lab[sample_region]  # (N, 3) where N is number of sampled pixels
    
    # Compute statistics for each Lab channel
    stats = {
        'mean': np.mean(sampled_pixels, axis=0),  # (3,) for L, a, b
        'std': np.std(sampled_pixels, axis=0)     # (3,) for L, a, b
    }
    
    return stats


def apply_color_correction_with_mask(source_frame, mask, source_stats, reference_stats, strength):
    """
    Apply color correction to pixels where mask=1.
    
    Args:
        source_frame: RGB image (H, W, C) in range [0, 1]
        mask: Binary mask (H, W)
        source_stats: Color statistics from source sampling region
        reference_stats: Color statistics from reference sampling region
        strength: Blending strength
        
    Returns:
        Corrected RGB image (H, W, C)
    """
    try:
        source_lab = color.rgb2lab(source_frame)
    except ValueError as e:
        print(f"Warning: Could not convert source frame to Lab: {e}. Using original frame.")
        return source_frame
    
    corrected_lab = source_lab.copy()
    correction_region = (mask == 1)  # Apply correction to mask=1 pixels
    
    # Apply color transfer to pixels where mask=1
    for c in range(3):  # L, a, b channels
        mean_src = source_stats['mean'][c]
        std_src = source_stats['std'][c]
        mean_ref = reference_stats['mean'][c]
        std_ref = reference_stats['std'][c]
        
        if std_src == 0:
            # Handle case where source channel has no variation
            corrected_lab[correction_region, c] = mean_ref
        else:
            # Standard color transfer formula
            corrected_lab[correction_region, c] = (
                (corrected_lab[correction_region, c] - mean_src) * (std_ref / std_src) + mean_ref
            )
    
    try:
        fully_corrected_rgb = color.lab2rgb(corrected_lab)
    except ValueError as e:
        print(f"Warning: Could not convert corrected frame back to RGB: {e}. Using original frame.")
        return source_frame
    
    # Clip to [0, 1]
    fully_corrected_rgb = np.clip(fully_corrected_rgb, 0.0, 1.0)
    
    # Blend with original (only in correction region)
    result = source_frame.copy()
    result[correction_region] = (
        (1 - strength) * source_frame[correction_region] + 
        strength * fully_corrected_rgb[correction_region]
    )
    
    return result


def apply_progressive_blend_in_corrected_area(corrected_frame, reference_frame, mask, source_region, source_distances, reference_region, source_border_distance):
    """
    Apply progressive blending in the corrected area (mask=1) near the border.
    
    Args:
        corrected_frame: RGB image (H, W, C) - the color-corrected source frame
        reference_frame: RGB image (H, W, C) - the reference frame
        mask: Binary mask (H, W)
        source_region: Boolean mask (H, W) indicating the source blending region (mask=1 near border)
        source_distances: Distance map (H, W) into mask=1 area from mask=0 border
        reference_region: Boolean mask (H, W) indicating the reference sampling region (mask=0 near border)
        source_border_distance: Maximum distance for source blending
        
    Returns:
        Blended RGB image (H, W, C)
        
            Notes:
        - Each source pixel blends with its closest reference border pixel (for speed)
        - At mask border: 60% source + 40% reference
        - Deeper into mask=1 area: 100% corrected source
    """
    result = corrected_frame.copy()
    
    # Blend in the source region (mask=1 pixels near border)
    blend_region = source_region
    
    if np.any(blend_region):
        # Find immediate border pixels (mask=0 pixels adjacent to mask=1 pixels)
        # This is much faster than using the entire reference region
        from scipy.ndimage import binary_dilation
        
        # Dilate mask=1 by 1 pixel, then find intersection with mask=0
        mask_1_dilated = binary_dilation(mask == 1, structure=np.ones((3, 3)))
        border_pixels = (mask == 0) & mask_1_dilated
        
        if np.any(border_pixels):
            # Find closest border pixel for each source pixel
            source_coords = np.column_stack(np.where(blend_region))  # (N, 2) - y, x coordinates
            border_coords = np.column_stack(np.where(border_pixels))  # (M, 2) - much smaller set!
            
            # For each source pixel, find closest border pixel
            from scipy.spatial.distance import cdist
            distances_matrix = cdist(source_coords, border_coords, metric='euclidean')
            closest_border_indices = np.argmin(distances_matrix, axis=1)
            
            # Normalize source distances for blending weights
            min_distance_in_region = np.min(source_distances[blend_region])
            max_distance_in_region = np.max(source_distances[blend_region])
            
            if max_distance_in_region > min_distance_in_region:
                # Calculate blend weights: 0.4 at border (60% source + 40% reference), 0.0 at max distance (100% source)
                source_dist_values = source_distances[blend_region]
                normalized_distances = (source_dist_values - min_distance_in_region) / (max_distance_in_region - min_distance_in_region)
                blend_weights = 0.4 * (1.0 - normalized_distances)  # Start with 40% reference influence at border
                
                # Apply blending with closest border pixels
                for i, (source_y, source_x) in enumerate(source_coords):
                    closest_border_idx = closest_border_indices[i]
                    border_y, border_x = border_coords[closest_border_idx]
                    
                    weight = blend_weights[i]
                    # Blend with closest border pixel
                    result[source_y, source_x] = (
                        (1.0 - weight) * corrected_frame[source_y, source_x] + 
                        weight * reference_frame[border_y, border_x]
                    )
    
    return result


def apply_progressive_blend_in_reference_area(reference_frame, source_frame, mask, reference_region, reference_distances, reference_border_distance):
    """
    Apply progressive blending in the reference area (mask=0) near the border.
    
    Args:
        reference_frame: RGB image (H, W, C) - the reference frame with copied reference pixels
        source_frame: RGB image (H, W, C) - the original source frame
        mask: Binary mask (H, W)
        reference_region: Boolean mask (H, W) indicating the reference blending region (mask=0 near border)
        reference_distances: Distance map (H, W) into mask=0 area from mask=1 border
        reference_border_distance: Maximum distance for reference blending
        
    Returns:
        Blended RGB image (H, W, C)
        
    Notes:
        - Each reference pixel blends with its closest source border pixel (for speed)
        - At mask border: 60% reference + 40% source
        - Deeper into mask=0 area: 100% reference
    """
    result = reference_frame.copy()
    
    # Blend in the reference region (mask=0 pixels near border)
    blend_region = reference_region
    
    if np.any(blend_region):
        # Find immediate border pixels (mask=1 pixels adjacent to mask=0 pixels)
        from scipy.ndimage import binary_dilation
        
        # Dilate mask=0 by 1 pixel, then find intersection with mask=1
        mask_0_dilated = binary_dilation(mask == 0, structure=np.ones((3, 3)))
        source_border_pixels = (mask == 1) & mask_0_dilated
        
        if np.any(source_border_pixels):
            # Find closest source border pixel for each reference pixel
            reference_coords = np.column_stack(np.where(blend_region))  # (N, 2) - y, x coordinates
            source_border_coords = np.column_stack(np.where(source_border_pixels))  # (M, 2)
            
            # For each reference pixel, find closest source border pixel
            from scipy.spatial.distance import cdist
            distances_matrix = cdist(reference_coords, source_border_coords, metric='euclidean')
            closest_source_indices = np.argmin(distances_matrix, axis=1)
            
            # Normalize reference distances for blending weights
            min_distance_in_region = np.min(reference_distances[blend_region])
            max_distance_in_region = np.max(reference_distances[blend_region])
            
            if max_distance_in_region > min_distance_in_region:
                # Calculate blend weights: 0.4 at border (60% reference + 40% source), 0.0 at max distance (100% reference)
                reference_dist_values = reference_distances[blend_region]
                normalized_distances = (reference_dist_values - min_distance_in_region) / (max_distance_in_region - min_distance_in_region)
                blend_weights = 0.4 * (1.0 - normalized_distances)  # Start with 40% source influence at border
                
                # Apply blending with closest source border pixels
                for i, (ref_y, ref_x) in enumerate(reference_coords):
                    closest_source_idx = closest_source_indices[i]
                    source_y, source_x = source_border_coords[closest_source_idx]
                    
                    weight = blend_weights[i]
                    # Blend: weight=0.4 means 60% reference + 40% source at border
                    result[ref_y, ref_x] = (
                        (1.0 - weight) * reference_frame[ref_y, ref_x] + 
                        weight * source_frame[source_y, source_x]
                    )
    
    return result


def apply_copy_with_mask(source_frame, reference_frame, mask, copy_source):
    """
    Copy pixels to mask=0 regions based on copy_source parameter.
    
    Args:
        source_frame: RGB image (H, W, C)
        reference_frame: RGB image (H, W, C)
        mask: Binary mask (H, W)
        copy_source: "reference" or "source"
        
    Returns:
        Combined RGB image (H, W, C)
    """
    result = source_frame.copy()
    mask_0_region = (mask == 0)
    
    if copy_source == "reference":
        result[mask_0_region] = reference_frame[mask_0_region]
    # If "source", we keep the original source pixels (no change needed)
    
    return result
