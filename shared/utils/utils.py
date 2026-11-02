# Copyright (c) 2025-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import argparse
import math
import os
import os.path as osp
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import cv2
import tempfile
import imageio
import torch
from PIL import Image
import numpy as np
from rembg import remove
import random
import ffmpeg
import os
import tempfile
import time
from functools import lru_cache
from mmgp import transfer
from . import files_locator as fl
from .video_decode import probe_video_stream_metadata, decode_video_frames_ffmpeg, get_video_summary_extras
from .virtual_media import get_virtual_image, parse_virtual_media_path, strip_virtual_media_suffix
from .gallery_sources import APP_EXTENSIONS, MESH_EXTENSIONS, MESH_IMPORT_EXTENSIONS, WORLD_EXTENSIONS


from PIL import Image
video_info_cache = []
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def has_video_file_extension(filename):
    filename = strip_virtual_media_suffix(filename)
    extension = os.path.splitext(filename)[-1].lower()
    return extension in [".mp4", ".mkv", ".avi", ".mov"]

def has_image_file_extension(filename):
    filename = strip_virtual_media_suffix(filename)
    extension = os.path.splitext(filename)[-1].lower()
    return extension in [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff", ".jfif", ".pjpeg"]

def has_audio_file_extension(filename):
    filename = strip_virtual_media_suffix(filename)
    extension = os.path.splitext(filename)[-1].lower()
    return extension in [".wav", ".mp3", ".aac", ".flac"]


def has_mesh_file_extension(filename):
    return os.path.splitext(strip_virtual_media_suffix(filename))[-1].lower() in MESH_EXTENSIONS

def has_world_file_extension(filename):
    return os.path.splitext(strip_virtual_media_suffix(filename))[-1].lower() in WORLD_EXTENSIONS

def has_app_file_extension(filename):
    return os.path.splitext(strip_virtual_media_suffix(filename))[-1].lower() in APP_EXTENSIONS

def get_media_kind(filename):
    """"image", "video", "audio", "3d", "world", "app" or None, from the file extension."""
    for kind, test in (("video", has_video_file_extension), ("image", has_image_file_extension), ("audio", has_audio_file_extension), ("3d", has_mesh_file_extension),
                       ("world", has_world_file_extension), ("app", has_app_file_extension)):
        if test(filename):
            return kind
    return None

def resample(video_fps, video_frames_count, max_target_frames_count, target_fps, start_target_frame ):
    import math

    video_frame_duration = 1 /round(video_fps, 0)
    target_frame_duration = 1 / round(target_fps, 0) 
    
    target_time = start_target_frame * target_frame_duration
    frame_no = math.ceil(target_time / video_frame_duration)  
    cur_time = frame_no * video_frame_duration
    frame_ids =[]
    while True:
        if max_target_frames_count != 0 and len(frame_ids) >= max_target_frames_count :
            break
        diff = round( (target_time -cur_time) / video_frame_duration , 5)
        add_frames_count = math.ceil( diff)
        frame_no += add_frames_count
        if frame_no >= video_frames_count:             
            break
        frame_ids.append(frame_no)
        cur_time += add_frames_count * video_frame_duration
        target_time += target_frame_duration
    frame_ids = frame_ids[:max_target_frames_count]
    return frame_ids

import os
from datetime import datetime

def get_file_creation_date(file_path):
    # On Windows
    if os.name == 'nt':
        return datetime.fromtimestamp(os.path.getctime(file_path))
    # On Unix/Linux/Mac (gets last status change, not creation)
    else:
        stat = os.stat(file_path)
    return datetime.fromtimestamp(stat.st_birthtime if hasattr(stat, 'st_birthtime') else stat.st_mtime)

def sanitize_file_name(file_name, rep =""):
    return "".join(rep if ord(char) < 32 or char in '/\\*:|?<>"' else char for char in file_name)

def truncate_for_filesystem(s, max_bytes=None):
    if max_bytes is None:
        max_bytes = 50 if os.name == 'nt'else 100

    if len(s.encode('utf-8')) <= max_bytes: return s
    l, r = 0, len(s)
    while l < r:
        m = (l + r + 1) // 2
        if len(s[:m].encode('utf-8')) <= max_bytes: l = m
        else: r = m - 1
    return s[:l]

def get_default_workers():
    return os.cpu_count()/ 2

def to_rgb_tensor(value, device="cpu", dtype=torch.float):
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=device, dtype=dtype)
    else:
        if isinstance(value, (list, tuple, np.ndarray)):
            vals = value
        else:
            vals = [value, value, value]
        tensor = torch.tensor(vals, device=device, dtype=dtype)
    if tensor.numel() == 1:
        tensor = tensor.repeat(3)
    elif tensor.numel() != 3:
        tensor = tensor.flatten()
        if tensor.numel() < 3:
            tensor = tensor.repeat(3)[:3]
        else:
            tensor = tensor[:3]
    return tensor.view(3, 1, 1)

def guide_to_float(video, memory_format=torch.preserve_format):
    """A control video in [-1, 1]. Models declaring model_def "uint8_guides" may receive uint8 control videos (0-255): they are converted
    with the operations WanGP applies to float guides, so the values are identical (the divisor is a tensor: CUDA divides by a scalar
    through its reciprocal, which changes some values). Float videos are returned as they are."""
    if video is None or video.dtype != torch.uint8:
        return video
    return video.to(torch.float32, memory_format=memory_format).div_(torch.tensor(127.5, device=video.device)).sub_(1.0)

def guide_to_dtype(video, dtype, chunk_frames=16):
    """A control video (C, T, H, W) in dtype with the values of guide_to_float, converted by chunks of frames: a uint8 guide does not go
    through a float32 copy of the whole video. Float videos are only cast."""
    if video.dtype != torch.uint8:
        return video.to(dtype)
    output = torch.empty_like(video, dtype=dtype)
    for start in range(0, video.shape[1], chunk_frames):
        output[:, start:start + chunk_frames] = guide_to_float(video[:, start:start + chunk_frames])
    return output

def guide_mask_to_float(mask, memory_format=torch.preserve_format):
    """A control mask in [0, 1]: uint8 masks (0 / 255, model_def "uint8_guides") as WanGP's float masks; float masks are returned as they are."""
    if mask is None or mask.dtype != torch.uint8:
        return mask
    return mask.to(torch.float32, memory_format=memory_format).div_(torch.tensor(255.0, device=mask.device))

def process_images_multithread(image_processor, items, process_type, wrap_in_list = True, max_workers: int = os.cpu_count()/ 2, in_place = False) :
    if not items:
       return []    

    import concurrent.futures
    start_time = time.time()
    # print(f"Preprocessus:{process_type} started")
    if process_type in ["prephase", "upsample"]: 
        if wrap_in_list :
            items_list = [ [img] for img in items]
        else:
            items_list = items
        if max_workers == 1:
            results = []
            for idx, item in enumerate(items):
                item = image_processor(item)
                results.append(item)
                if wrap_in_list: items_list[idx] = None
                if in_place: items[idx] = item[0] if wrap_in_list else item
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(image_processor, img): idx for idx, img in enumerate(items_list)}
                results = [None] * len(items_list)
                for future in concurrent.futures.as_completed(futures):
                    idx = futures[future]
                    results[idx] = future.result()
                    if wrap_in_list: items_list[idx] = None
                    if in_place: 
                        items[idx] = results[idx][0] if wrap_in_list else results[idx] 

        if wrap_in_list: 
            results = [ img[0] for img in results]
    else:
        results=  image_processor(items) 

    end_time = time.time()
    # print(f"duration:{end_time-start_time:.1f}")

    return results

def get_resampled_video_transparent(video_in, start_frame, max_frames, target_fps, bridge='torch'):
    base_video_in = strip_virtual_media_suffix(video_in) if isinstance(video_in, str) else video_in
    virtual_image = get_virtual_image(video_in) if isinstance(video_in, str) else None
    if virtual_image is not None:
        video_in = virtual_image
    elif isinstance(base_video_in, str) and has_image_file_extension(base_video_in):
        video_in = Image.open(base_video_in)
    if isinstance(video_in, Image.Image):
        frame = torch.from_numpy(np.array(video_in).astype(np.uint8)).unsqueeze(0)
        return frame if bridge == "torch" else frame.numpy()
    metadata = probe_video_stream_metadata(video_in)
    fps_float = metadata["fps_float"] if metadata is not None else 0.0
    if max_frames < 0:
        max_frames = int(max((metadata["frame_count"] / fps_float) * target_fps + max_frames, 0)) if metadata is not None and fps_float > 0 else 0
    return decode_video_frames_ffmpeg(video_in, start_frame, max_frames, target_fps=target_fps, bridge=bridge)


@lru_cache(maxsize=100)
def _get_video_info_cached(video_path):
    metadata = probe_video_stream_metadata(video_path)
    if metadata is not None:
        return metadata["fps"], metadata["display_width"], metadata["display_height"], metadata["frame_count"]
    global video_info_cache
    import cv2
    cap = cv2.VideoCapture(video_path)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, width, height, frame_count


def get_video_info(video_path):
    if isinstance(video_path, Image.Image):
        width, height = video_path.size
        return 1, width, height, 1
    return _get_video_info_cached(video_path)


@lru_cache(maxsize=100)
def _get_video_info_details_cached(video_path):
    metadata = probe_video_stream_metadata(video_path)
    if metadata is not None:
        return metadata
    fps, width, height, frame_count = get_video_info(video_path)
    return {
        "source_path": strip_virtual_media_suffix(video_path),
        "width": width,
        "height": height,
        "display_width": width,
        "display_height": height,
        "fps_float": float(fps),
        "fps": int(fps),
        "frame_count": int(frame_count),
        "duration": float(frame_count / fps) if fps > 0 else 0.0,
        "sample_aspect_ratio": "1:1",
        "display_aspect_ratio": "",
        "color_transfer": "",
        "color_primaries": "",
        "color_space": "",
        "color_range": "",
        "needs_sar_fix": False,
        "needs_tonemap": False,
    }


def get_video_info_details(video_path):
    if isinstance(video_path, Image.Image):
        width, height = video_path.size
        return {
            "source_path": "",
            "width": width,
            "height": height,
            "display_width": width,
            "display_height": height,
            "fps_float": 1.0,
            "fps": 1,
            "frame_count": 1,
            "duration": 1.0,
            "sample_aspect_ratio": "1:1",
            "display_aspect_ratio": "",
            "color_transfer": "",
            "color_primaries": "",
            "color_space": "",
            "color_range": "",
            "needs_sar_fix": False,
            "needs_tonemap": False,
        }
    return _get_video_info_details_cached(video_path)

def get_video_frame(file_name: str, frame_no: int, return_last_if_missing: bool = False, target_fps = None,  return_PIL = True) -> torch.Tensor:
    """Extract nth frame from video as PyTorch tensor normalized to [-1, 1]."""
    metadata = probe_video_stream_metadata(file_name)
    if metadata is not None and (metadata["needs_sar_fix"] or metadata["needs_tonemap"] or parse_virtual_media_path(file_name) is not None):
        fps_float = metadata["fps_float"] if metadata["fps_float"] > 0 else float(metadata["fps"] or 1)
        if target_fps is not None and float(target_fps) > 0:
            max_target_frames = int(round(metadata["frame_count"] / fps_float * float(target_fps))) if metadata["frame_count"] > 0 else 0
            if return_last_if_missing and max_target_frames > 0:
                frame_no = min(max(0, int(frame_no)), max_target_frames - 1)
            frames = decode_video_frames_ffmpeg(file_name, int(frame_no), 1, target_fps=float(target_fps), bridge="torch")
        else:
            if return_last_if_missing and metadata["frame_count"] > 0:
                frame_no = min(max(0, int(frame_no)), metadata["frame_count"] - 1)
            frames = decode_video_frames_ffmpeg(file_name, int(frame_no), 1, target_fps=None, bridge="torch")
        if frames.shape[0] == 0:
            raise ValueError(f"Failed to read frame {frame_no}")
        frame = frames[0]
        if return_PIL:
            if torch.is_tensor(frame) and frame.dtype != torch.uint8:
                from .hdr import linear_to_srgb
                frame = linear_to_srgb(frame.to(dtype=torch.float32)).mul(255.0).round_().clamp_(0.0, 255.0).to(torch.uint8)
            return Image.fromarray(frame.cpu().numpy() if torch.is_tensor(frame) else frame)
        return frame.permute(2, 0, 1).float().div_(127.5).sub_(1.0)
    virtual_spec = parse_virtual_media_path(file_name)
    base_file_name = strip_virtual_media_suffix(file_name)
    cap = cv2.VideoCapture(base_file_name)
    
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {file_name}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    virtual_start = 0
    if virtual_spec is not None and total_frames > 0:
        virtual_start = max(0, min(int(virtual_spec.start_frame), total_frames - 1))
        virtual_end = total_frames - 1 if virtual_spec.end_frame is None else max(virtual_start, min(int(virtual_spec.end_frame), total_frames - 1))
        total_frames = max(0, virtual_end - virtual_start + 1)
    if target_fps is not None:
        frame_no = round(target_fps * frame_no /fps)

    # Handle out of bounds
    if frame_no >= total_frames or frame_no < 0:
        if return_last_if_missing:
            frame_no = total_frames - 1
        else:
            cap.release()
            raise IndexError(f"Frame {frame_no} out of bounds (0-{total_frames-1})")
    
    # Get frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, virtual_start + frame_no)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        raise ValueError(f"Failed to read frame {frame_no}")
    
    # Convert BGR->RGB, reshape to (C,H,W), normalize to [-1,1]
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    if return_PIL:
          return Image.fromarray(frame)
    else:
        return (torch.from_numpy(frame).permute(2, 0, 1).float() / 127.5) - 1.0

def convert_image_to_video(image):
    if image is None:
        return None
    
    # Convert PIL/numpy image to OpenCV format if needed
    if isinstance(image, np.ndarray):
        # Gradio images are typically RGB, OpenCV expects BGR
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        # Handle PIL Image
        img_array = np.array(image)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    height, width = img_bgr.shape[:2]
    
    # Create temporary video file (auto-cleaned by Gradio)
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_video.name, fourcc, 30.0, (width, height))
        out.write(img_bgr)
        out.release()
        return temp_video.name
    
def resize_lanczos(img, h, w, method = None):
    img = (img + 1).float().mul_(127.5)
    img = Image.fromarray(np.clip(img.movedim(0, -1).cpu().numpy(), 0, 255).astype(np.uint8))
    img = img.resize((w,h), resample=Image.Resampling.LANCZOS if method is None else method) 
    img = torch.from_numpy(np.array(img).astype(np.float32)).movedim(-1, 0)
    img = img.div(127.5).sub_(1)
    return img

def resolve_rembg_home():
    rembg_home = fl.locate_folder("rembg", error_if_none=False)
    if rembg_home is None:
        rembg_home = fl.get_smart_download_location(force_path="rembg")
    return os.path.abspath(rembg_home)

REMBG_ON_CPU = True  # CPU: ~1.3 s per image like a warm GPU (alpha matting runs on the CPU anyway), no 10-15 s CUDA provider start, no VRAM


def new_rembg_session(model_name="u2net"):
    from preprocessing.rembg.assets import MODELS, ensure_assets
    model_path = ensure_assets(model_name=model_name) if model_name in MODELS else None  # checked at download: no rehash of the file
    os.environ["U2NET_HOME"] = resolve_rembg_home()
    import onnxruntime as ort
    from rembg.sessions import sessions_class
    # rembg 2.0.65 hard-codes its providers, so the session is built here. On the GPU, onnxruntime's default exhaustive
    # cuDNN search costs ~10 s on the first image; the heuristic choice gives the same masks.
    session_class = next(cls for cls in sessions_class if cls.name() == model_name)
    session = session_class.__new__(session_class)
    session.model_name = model_name
    cuda = not REMBG_ON_CPU and "CUDAExecutionProvider" in ort.get_available_providers()
    providers = [("CUDAExecutionProvider", {"cudnn_conv_algo_search": "HEURISTIC"}), "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
    session.inner_session = ort.InferenceSession(str(model_path or session_class.download_models()), sess_options=ort.SessionOptions(), providers=providers)
    return session

def remove_background(img, session=None):
    if session ==None:
        session = new_rembg_session() 
    img = Image.fromarray(np.clip(255. * img.movedim(0, -1).cpu().numpy(), 0, 255).astype(np.uint8))
    img = remove(img, session=session, alpha_matting = True, bgcolor=[255, 255, 255, 0]).convert('RGB')
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).movedim(-1, 0)


def convert_image_to_tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32)).div_(127.5).sub_(1.).movedim(-1, 0)

def convert_tensor_to_image(t, frame_no = 0, mask_levels = False):
    if len(t.shape) == 4:
        t = t[:, frame_no] 
    if t.shape[0]== 1:
        t = t.expand(3,-1,-1)
    if t.dtype == torch.uint8:
        return Image.fromarray(t.permute(1, 2, 0).cpu().numpy())
    if mask_levels:
        return Image.fromarray(t.clone().mul_(255).permute(1,2,0).to(torch.uint8).cpu().numpy())
    else:
        return Image.fromarray(t.clone().add_(1.).mul_(127.5).permute(1,2,0).to(torch.uint8).cpu().numpy())

def convert_video_tensor_to_uint8_chunked(video, value_range=(-1, 1), max_buffer_mb=256, output_device=None):
    # output_device: where the uint8 video goes (the video's device by default); a video on the GPU is converted there chunk by chunk,
    # so only the uint8 video reaches the RAM
    output_device = video.device if output_device is None else torch.device(output_device)
    if video.dtype == torch.uint8:
        return video.to(output_device)
    min_val, max_val = value_range
    scale = 255.0 / (max_val - min_val)
    output = torch.empty(video.shape, dtype=torch.uint8, device=output_device)
    time_dim = 2 if video.ndim == 5 else 1 if video.ndim >= 4 else 0
    frames = video.shape[time_dim]
    frame_elems = max(1, video.numel() // max(1, frames))
    chunk_frames = max(1, min(frames, int(max_buffer_mb * 1024 * 1024) // max(1, frame_elems * 2)))
    buffer_shape = list(video.shape)
    buffer_shape[time_dim] = chunk_frames
    buffer = torch.empty(buffer_shape, dtype=torch.float16, device=video.device)
    for start in range(0, frames, chunk_frames):
        size = min(chunk_frames, frames - start)
        work = buffer.narrow(time_dim, 0, size)
        work.copy_(video.narrow(time_dim, start, size))
        work.sub_(min_val).mul_(scale).clamp_(0, 255)
        transfer.copy_(output.narrow(time_dim, start, size), work)  # (to RAM, without temporaries of the chunk)
    return output

def save_image(tensor_image, name, frame_no = -1):
    convert_tensor_to_image(tensor_image, frame_no).save(name)

def parse_outpainting_ratio(outpainting_ratio):
    ratio = "" if outpainting_ratio is None else str(outpainting_ratio).strip()
    if len(ratio) == 0:
        return None
    ratio = ratio.split(":")
    if len(ratio) != 2:
        return None
    try:
        width_ratio, height_ratio = float(ratio[0]), float(ratio[1])
    except (TypeError, ValueError):
        return None
    return None if width_ratio <= 0 or height_ratio <= 0 else width_ratio / height_ratio


def get_outpainting_dims(video_guide_outpainting, video_guide_outpainting_ratio = ""):
    if video_guide_outpainting is None:
        return None
    video_guide_outpainting = str(video_guide_outpainting).strip()
    if video_guide_outpainting.startswith("#") :
        return None
    if video_guide_outpainting == "0 0 0 0" or len(video_guide_outpainting) == 0:
        return [0, 0, 0, 0] if len(video_guide_outpainting_ratio) else None
    outpainting_dims = video_guide_outpainting.split(" ")
    return None if len(outpainting_dims) != 4 else [int(v) for v in outpainting_dims]


def _split_outpainting_padding(total_padding, before_weight, after_weight):
    total_padding = max(0, int(total_padding))
    if total_padding == 0:
        return 0, 0
    before_weight = max(0.0, float(before_weight))
    after_weight = max(0.0, float(after_weight))
    if before_weight == after_weight:
        before_padding = total_padding // 2
    elif before_weight == 0:
        before_padding = 0
    elif after_weight == 0:
        before_padding = total_padding
    else:
        before_padding = round(total_padding * before_weight / (before_weight + after_weight))
    before_padding = max(0, min(total_padding, int(before_padding)))
    return before_padding, total_padding - before_padding


def resolve_outpainting_dims(frame_height, frame_width, outpainting_dims, outpainting_ratio = ""):
    target_ratio = parse_outpainting_ratio(outpainting_ratio)
    if outpainting_dims is None:
        return None if target_ratio is None else [0.0, 0.0, 0.0, 0.0]
    outpainting_top, outpainting_bottom, outpainting_left, outpainting_right = [max(0.0, float(v)) for v in outpainting_dims]
    if target_ratio is None or frame_height <= 0 or frame_width <= 0:
        return [outpainting_top, outpainting_bottom, outpainting_left, outpainting_right]

    source_ratio = frame_width / frame_height
    if source_ratio < target_ratio:
        total_padding = max(0, round(frame_height * target_ratio - frame_width))
        left_padding, right_padding = _split_outpainting_padding(total_padding, outpainting_left, outpainting_right)
        return [0.0, 0.0, 100.0 * left_padding / frame_width, 100.0 * right_padding / frame_width]
    if source_ratio > target_ratio:
        total_padding = max(0, round(frame_width / target_ratio - frame_height))
        top_padding, bottom_padding = _split_outpainting_padding(total_padding, outpainting_top, outpainting_bottom)
        return [100.0 * top_padding / frame_height, 100.0 * bottom_padding / frame_height, 0.0, 0.0]
    return [0.0, 0.0, 0.0, 0.0]


def get_outpainting_full_area_dimensions(frame_height,frame_width, outpainting_dims, outpainting_ratio = ""):
    outpainting_top, outpainting_bottom, outpainting_left, outpainting_right = resolve_outpainting_dims(frame_height, frame_width, outpainting_dims, outpainting_ratio)
    frame_height = int(frame_height * (100 + outpainting_top + outpainting_bottom) / 100)
    frame_width =  int(frame_width * (100 + outpainting_left + outpainting_right) / 100)
    return frame_height, frame_width  

def rgb_bw_to_rgba_mask(img, thresh=127):
    if img.mode=="RGBA": return img 
    arr = np.array(img.convert('L'))
    alpha = (arr > thresh).astype(np.uint8) * 255
    rgba = np.dstack([np.full_like(alpha, 255)] * 3 + [alpha])
    return Image.fromarray(rgba, 'RGBA')


def image_editor_layer_to_rgb_mask(layer):
    if isinstance(layer, Image.Image) and layer.mode == "RGBA":
        alpha = layer.getchannel("A")
        return Image.merge("RGB", (alpha, alpha, alpha))
    return layer


def _quantize_outpainting_axis(before, inner, total, quantum):
    quantum = int(quantum or 0)
    if quantum <= 1:
        return inner, before
    after = total - before - inner
    before = max(0, int(round(before / quantum) * quantum))
    after = max(0, int(round(after / quantum) * quantum))
    min_inner = quantum if total >= quantum else 1
    overflow = before + after + min_inner - total
    if overflow > 0:
        if after >= before:
            after = max(0, after - int(math.ceil(overflow / quantum) * quantum))
        else:
            before = max(0, before - int(math.ceil(overflow / quantum) * quantum))
    inner = total - before - after
    return inner, before

def  get_outpainting_frame_location(final_height, final_width,  outpainting_dims, block_size = 8, outpainting_ratio = "", source_height = None, source_width = None, quantize_margins = 0):
    if source_height is not None and source_width is not None:
        outpainting_dims = resolve_outpainting_dims(source_height, source_width, outpainting_dims, outpainting_ratio)
    outpainting_top, outpainting_bottom, outpainting_left, outpainting_right= outpainting_dims
    raw_height = int(final_height / ((100 + outpainting_top + outpainting_bottom) / 100))
    height = int(raw_height / block_size) * block_size
    extra_height = raw_height - height
          
    raw_width = int(final_width / ((100 + outpainting_left + outpainting_right) / 100)) 
    width = int(raw_width / block_size) * block_size
    extra_width = raw_width - width  
    margin_top = int(outpainting_top/(100 + outpainting_top + outpainting_bottom) * final_height)
    if extra_height != 0 and (outpainting_top + outpainting_bottom) != 0:
        margin_top += int(outpainting_top / (outpainting_top + outpainting_bottom) * extra_height)
    if (margin_top + height) > final_height or outpainting_bottom == 0: margin_top = final_height - height
    margin_left = int(outpainting_left/(100 + outpainting_left + outpainting_right) * final_width)
    if extra_width != 0 and (outpainting_left + outpainting_right) != 0:
        margin_left += int(outpainting_left / (outpainting_left + outpainting_right) * extra_width)
    if (margin_left + width) > final_width or outpainting_right == 0: margin_left = final_width - width
    if quantize_margins:
        height, margin_top = _quantize_outpainting_axis(margin_top, height, final_height, quantize_margins)
        width, margin_left = _quantize_outpainting_axis(margin_left, width, final_width, quantize_margins)
    return height, width, margin_top, margin_left

def rescale_and_crop(img, w, h):
    ow, oh = img.size
    target_ratio = w / h
    orig_ratio = ow / oh
    
    if orig_ratio > target_ratio:
        # Crop width first
        nw = int(oh * target_ratio)
        img = img.crop(((ow - nw) // 2, 0, (ow + nw) // 2, oh))
    else:
        # Crop height first
        nh = int(ow / target_ratio)
        img = img.crop((0, (oh - nh) // 2, ow, (oh + nh) // 2))
    
    return img.resize((w, h), Image.LANCZOS)

def resize_lanczos_frames(frames, target_h, target_w, crop=False, max_workers=None, in_place=True):
    frames = list(frames)

    def resize_frame(frame):
        arr = frame.detach().cpu().numpy() if torch.is_tensor(frame) else np.asarray(frame)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)
        img = rescale_and_crop(img, target_w, target_h) if crop else img.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)
        return torch.from_numpy(np.array(img, copy=True))

    return process_images_multithread(resize_frame, frames, "upsample", wrap_in_list=False, max_workers=max(1, int(max_workers or get_default_workers())), in_place=in_place)

def resize_lanczos_cthw(tensor, target_h, target_w, crop=False, max_workers=None):
    if tensor is None or tensor.shape[-2:] == (target_h, target_w):
        return tensor
    device, dtype = tensor.device, tensor.dtype
    frames = [((frame.permute(1, 2, 0) + 1.0) * 127.5).clamp(0, 255).to(torch.uint8) for frame in tensor.permute(1, 0, 2, 3).detach().cpu()]
    frames = resize_lanczos_frames(frames, target_h, target_w, crop=crop, max_workers=max_workers)
    frames = [frame.to(torch.float32).permute(2, 0, 1).div_(127.5).sub_(1.0) for frame in frames]
    return torch.stack(frames, dim=1).to(device=device, dtype=dtype).contiguous()

def expand_or_shrink_mask(mask, expand_scale, iterations=3):
    expand_scale = int(expand_scale or 0)
    if expand_scale == 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abs(expand_scale), abs(expand_scale)))
    return (cv2.dilate if expand_scale > 0 else cv2.erode)(mask, kernel, iterations=iterations)

def prepare_binary_mask_frame(mask, target_h=None, target_w=None, expand_scale=0, invert=False, threshold=127):
    mask = mask.detach().cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
    if mask.ndim == 3 and mask.shape[-1] == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    if target_h is not None and target_w is not None and mask.shape[:2] != (target_h, target_w):
        mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    _, mask = cv2.threshold(mask.astype(np.uint8, copy=False), threshold, 255, cv2.THRESH_BINARY)
    mask = expand_or_shrink_mask(mask, expand_scale)
    return (mask <= threshold if invert else mask > threshold).astype(np.float32)

def expand_colored_mask_cthw(mask, object_colors, expand_scale, background_color=None, max_workers=None):
    if mask is None or int(expand_scale or 0) == 0 or not object_colors:
        return mask
    device, dtype = mask.device, mask.dtype
    was_negative = float(mask.min()) < 0
    mask_u8 = ((mask.detach().cpu().float() + 1.0) * 127.5 if was_negative else mask.detach().cpu().float()).clamp(0, 255).to(torch.uint8)
    colors = np.asarray(object_colors, dtype=np.uint8).reshape(-1, 3)
    background = np.asarray([0, 0, 0] if background_color is None else background_color, dtype=np.uint8).reshape(1, 1, 3)

    def expand_frame(frame):
        frame = frame.permute(1, 2, 0).numpy()
        output = np.broadcast_to(background, frame.shape).copy()
        occupied = np.zeros(frame.shape[:2], dtype=bool)
        for color in colors:
            selector = np.where(color.reshape(1, 1, 3) >= 128, frame >= 128, frame < 128).all(axis=-1)
            selector = expand_or_shrink_mask(selector.astype(np.uint8) * 255, expand_scale) > 127
            selector &= ~occupied
            output[selector] = color
            occupied |= selector
        return torch.from_numpy(output)

    frames = process_images_multithread(expand_frame, [mask_u8[:, i] for i in range(mask_u8.shape[1])], "upsample", wrap_in_list=False, max_workers=max(1, int(max_workers or get_default_workers())), in_place=True)
    out = torch.stack(frames, dim=0).permute(3, 0, 1, 2).to(torch.float32)
    if was_negative:
        out = out.div_(127.5).sub_(1.0)
    return out.to(device=device, dtype=dtype).contiguous()

def calculate_new_dimensions(canvas_height, canvas_width, image_height, image_width, fit_into_canvas, block_size=16, *, max_width=None, max_height=None):
    if fit_into_canvas == None or fit_into_canvas == 2:
        # return image_height, image_width
        return canvas_height, canvas_width
    if fit_into_canvas == 1:
        scale1  = min(canvas_height / image_height, canvas_width / image_width)
        scale2  = min(canvas_width / image_height, canvas_height / image_width)
        scale = max(scale1, scale2) 
    else: #0 or #2 (crop)
        scale = (canvas_height * canvas_width / (image_height * image_width))**(1/2)

    for limit, dimension in ((max_width, image_width), (max_height, image_height)):
        if limit is not None:
            scale = min(scale, (limit // block_size * block_size) / dimension)
    new_height = round( image_height * scale / block_size) * block_size
    new_width = round( image_width * scale / block_size) * block_size
    if max_width is not None or max_height is not None:
        new_height, new_width = max(block_size, new_height), max(block_size, new_width)
    return new_height, new_width

def calculate_dimensions_and_resize_image(image, canvas_height, canvas_width, fit_into_canvas, fit_crop, block_size = 16):
    if fit_crop:
        image = rescale_and_crop(image, canvas_width, canvas_height)
        new_width, new_height = image.size  
    else:
        image_width, image_height = image.size
        new_height, new_width = calculate_new_dimensions(canvas_height, canvas_width, image_height, image_width, fit_into_canvas, block_size = block_size )
        image = image.resize((new_width, new_height), resample=Image.Resampling.LANCZOS) 
    return image, new_height, new_width

def resize_and_remove_background(img_list, budget_width, budget_height, rm_background, any_background_ref, fit_into_canvas = 0, block_size= 16, outpainting_dims = None, outpainting_ratio = "", background_ref_outpainted = True, inpaint_color = 127.5, return_tensor = False, ignore_last_refs = 0, background_removal_color =  [255, 255, 255], outpainting_quantize_margins = 0, keep_alpha = False, *, max_width=None, max_height=None, rembg_model="u2net"):
    if max_width is not None or max_height is not None:
        budget_height, budget_width = calculate_new_dimensions(budget_height, budget_width, budget_height, budget_width, 0, block_size=block_size, max_width=max_width, max_height=max_height)
    if rm_background:
        session = new_rembg_session(rembg_model)
        matting = rembg_model == "u2net"  # BiRefNet's matte is already soft

    output_list =[]
    output_mask_list =[]
    for i, img in enumerate(img_list if ignore_last_refs == 0 else img_list[:-ignore_last_refs]):
        width, height =  img.size 
        resized_mask = None
        if any_background_ref == 1 and i==0 or any_background_ref == 2:
            if outpainting_dims is not None and background_ref_outpainted:
                resized_image, resized_mask = fit_image_into_canvas(img, (budget_height, budget_width), inpaint_color, full_frame = True, outpainting_dims = outpainting_dims, outpainting_ratio = outpainting_ratio, return_mask= True, return_image= True, outpainting_quantize_margins = outpainting_quantize_margins)
            elif img.size != (budget_width, budget_height):
                resized_image= img.resize((budget_width, budget_height), resample=Image.Resampling.LANCZOS) 
            else:
                resized_image =img
        elif fit_into_canvas == 1:
            canvas_color = to_rgb_tensor(background_removal_color, device="cpu", dtype=torch.uint8).view(1, 1, 3).numpy()
            if img.mode == "RGBA":  # references kept with their alpha (image_refs_keep_alpha) get a transparent canvas
                canvas_color = np.concatenate([canvas_color, np.zeros_like(canvas_color[..., :1])], axis=-1)
            canvas = np.broadcast_to(canvas_color, (budget_height, budget_width, canvas_color.shape[-1])).copy()
            scale = min(budget_height / height, budget_width / width)
            new_height = int(height * scale)
            new_width = int(width * scale)
            resized_image= img.resize((new_width,new_height), resample=Image.Resampling.LANCZOS) 
            top = (budget_height - new_height) // 2
            left = (budget_width - new_width) // 2
            canvas[top:top + new_height, left:left + new_width] = np.array(resized_image)
            resized_image = Image.fromarray(canvas)
        else:
            new_height, new_width = calculate_new_dimensions(budget_height, budget_width, height, width, 0, block_size=block_size, max_width=max_width, max_height=max_height)
            resized_image= img.resize((new_width,new_height), resample=Image.Resampling.LANCZOS) 
        if rm_background  and not (any_background_ref and i==0 or any_background_ref == 2) :
            # resized_image = remove(resized_image, session=session, alpha_matting_erode_size = 1,alpha_matting_background_threshold = 70, alpha_foreground_background_threshold = 100, alpha_matting = True, bgcolor=[255, 255, 255, 0]).convert('RGB')
            if keep_alpha:  # models reading the alpha channel get the transparent cut-out; an image already transparent is kept as is
                if not (img.mode == "RGBA" and img.getextrema()[3][0] < 255):
                    resized_image = remove(resized_image, session=session, alpha_matting_erode_size = 1, alpha_matting = matting)
            else:
                resized_image = remove(resized_image, session=session, alpha_matting_erode_size = 1, alpha_matting = matting, bgcolor=background_removal_color + [0]).convert('RGB')
        if return_tensor:
            output_list.append(convert_image_to_tensor(resized_image).unsqueeze(1)) 
        else:
            output_list.append(resized_image) 
        output_mask_list.append(resized_mask)
    if ignore_last_refs:
        for img in img_list[-ignore_last_refs:]:
            output_list.append(convert_image_to_tensor(img).unsqueeze(1) if return_tensor else img) 
            output_mask_list.append(None)

    return output_list, output_mask_list

def fit_image_into_canvas(ref_img, image_size, canvas_tf_bg =127.5, device ="cpu", full_frame = False, outpainting_dims = None, outpainting_ratio = "", return_mask = False, return_image = False, outpainting_quantize_margins = 0):
    inpaint_color = to_rgb_tensor(canvas_tf_bg, device=device, dtype=torch.float) / 127.5 - 1
    inpaint_color = inpaint_color.unsqueeze(1)

    ref_width, ref_height = ref_img.size
    if (ref_height, ref_width) == image_size and outpainting_dims  == None:
        ref_img = TF.to_tensor(ref_img).sub_(0.5).div_(0.5).unsqueeze(1)
        canvas = torch.zeros_like(ref_img[:1]) if return_mask else None
    else:
        if outpainting_dims != None:
            final_height, final_width = image_size
            canvas_height, canvas_width, margin_top, margin_left = get_outpainting_frame_location(final_height, final_width, outpainting_dims, 1, outpainting_ratio, ref_height, ref_width, quantize_margins=outpainting_quantize_margins)
        else:
            canvas_height, canvas_width = image_size
        if full_frame:
            new_height = canvas_height
            new_width = canvas_width
            top = left = 0 
        else:
            # if fill_max  and (canvas_height - new_height) < 16:
            #     new_height = canvas_height
            # if fill_max  and (canvas_width - new_width) < 16:
            #     new_width = canvas_width
            scale = min(canvas_height / ref_height, canvas_width / ref_width)
            new_height = int(ref_height * scale)
            new_width = int(ref_width * scale)
            top = (canvas_height - new_height) // 2
            left = (canvas_width - new_width) // 2
        ref_img = ref_img.resize((new_width, new_height), resample=Image.Resampling.LANCZOS) 
        ref_img = TF.to_tensor(ref_img).sub_(0.5).div_(0.5).unsqueeze(1)
        if outpainting_dims != None:
            canvas = inpaint_color.expand(3, 1, final_height, final_width).clone()
            canvas[:, :, margin_top + top:margin_top + top + new_height, margin_left + left:margin_left + left + new_width] = ref_img 
        else:
            canvas = inpaint_color.expand(3, 1, canvas_height, canvas_width).clone()
            canvas[:, :, top:top + new_height, left:left + new_width] = ref_img 
        ref_img = canvas
        canvas = None
        if return_mask:
            if outpainting_dims != None:
                canvas = torch.ones((1, 1, final_height, final_width), dtype= torch.float, device=device) # [-1, 1]
                canvas[:, :, margin_top + top:margin_top + top + new_height, margin_left + left:margin_left + left + new_width] = 0
            else:
                canvas = torch.ones((1, 1, canvas_height, canvas_width), dtype= torch.float, device=device) # [-1, 1]
                canvas[:, :, top:top + new_height, left:left + new_width] = 0
            canvas = canvas.to(device)
    if return_image:
        return convert_tensor_to_image(ref_img), canvas

    return ref_img.to(device), canvas

def prepare_video_guide_and_mask( video_guides, video_masks, pre_video_guide, image_size, current_video_length = 81, latent_size = 4, any_mask = False, any_guide_padding = False, guide_inpaint_color = 127.5, keep_video_guide_frames = [],  inject_frames = [], outpainting_dims = None, outpainting_ratio = "", device ="cpu", outpainting_quantize_margins = 0, frame_offset = 1):
    src_videos, src_masks = [], []
    inpaint_color = to_rgb_tensor(guide_inpaint_color, device=device, dtype=torch.float)
    inpaint_color_compressed = (inpaint_color / 127.5 - 1).unsqueeze(1)
    prepend_count = pre_video_guide.shape[1] if pre_video_guide is not None else 0
    # uint8 guides and masks (model_def "uint8_guides") stay uint8 unless float frames are merged into them below (padding and frames
    # that are not kept are filled in uint8 when the inpainting color is made of whole values)
    whole_color = torch.equal(inpaint_color, inpaint_color.round())
    if pre_video_guide is not None and any(guide is not None and guide.dtype == torch.uint8 for guide in video_guides):
        # the frames prepended to uint8 guides are kept in uint8 when they are 8 bit frames (start image, previous window)
        pre_video_guide_uint8 = pre_video_guide.add(1.0).mul_(127.5).round_().clamp_(0, 255).to(torch.uint8)
        if torch.equal(guide_to_float(pre_video_guide_uint8), pre_video_guide):
            pre_video_guide = pre_video_guide_uint8
    float_edits = pre_video_guide is not None and pre_video_guide.dtype != torch.uint8 or not (whole_color or all(keep_video_guide_frames)) or any(frame is not None for frame in inject_frames)
    for guide_no, (cur_video_guide, cur_video_mask) in enumerate(zip(video_guides, video_masks)):
        padded = any_guide_padding and (cur_video_guide is None or prepend_count + cur_video_guide.shape[1] < current_video_length)
        src_video, src_mask = (guide_to_float(cur_video_guide), guide_mask_to_float(cur_video_mask)) if float_edits or padded and not whole_color else (cur_video_guide, cur_video_mask)
        if pre_video_guide is not None:
            pre_guide = pre_video_guide if src_video is None or src_video.dtype == pre_video_guide.dtype else guide_to_float(pre_video_guide)
            src_video = pre_guide if src_video is None else torch.cat( [pre_guide, src_video], dim=1)
            if any_mask:
                src_mask = torch.zeros_like(pre_guide[:1]) if src_mask is None else torch.cat( [torch.zeros_like(pre_guide[:1], dtype=src_mask.dtype), src_mask], dim=1)

        if any_guide_padding:
            if src_video is None:
                src_video = inpaint_color_compressed.expand(3, current_video_length, *image_size).clone()
            elif src_video.shape[1] < current_video_length:
                color = inpaint_color.to(torch.uint8).unsqueeze(1) if src_video.dtype == torch.uint8 else inpaint_color_compressed
                pad = color.to(src_video.device).expand(3, current_video_length - src_video.shape[1], *src_video.shape[-2:]).clone()
                src_video = torch.cat([src_video, pad], dim=1)
        elif src_video is not None:
            new_num_frames = src_video.shape[1] if src_video.shape[1] < frame_offset else (src_video.shape[1] - frame_offset) // latent_size * latent_size + frame_offset
            if new_num_frames < src_video.shape[1]:
                print(f"invalid number of control frames {src_video.shape[1]}, potentially {src_video.shape[1]-new_num_frames} frames will be lost")
            src_video = src_video[:, :new_num_frames]

        full_mask = 255 if src_video is not None and src_video.dtype == torch.uint8 else 1
        if any_mask and src_video is not None:
            if src_mask is None:
                src_mask = torch.full_like(src_video[:1], full_mask)
            elif src_mask.shape[1] < src_video.shape[1]:
                src_mask = torch.cat([src_mask, torch.full( (1, src_video.shape[1]- src_mask.shape[1], *src_mask.shape[-2:]  ), full_mask, dtype = src_video.dtype, device= src_video.device) ], dim=1)
            else:
                src_mask = src_mask[:, :src_video.shape[1]]                                        

        if src_video is not None :
            for k, keep in enumerate(keep_video_guide_frames):
                if not keep:
                    pos = prepend_count + k
                    src_video[:, pos:pos+1] = (inpaint_color.to(torch.uint8).unsqueeze(1) if src_video.dtype == torch.uint8 else inpaint_color_compressed).to(src_video.device)
                    if any_mask: src_mask[:, pos:pos+1] = full_mask

            for k, frame in enumerate(inject_frames):
                if frame != None:
                    pos = prepend_count + k
                    src_video[:, pos:pos+1], msk = fit_image_into_canvas(frame, image_size, guide_inpaint_color, device, True, outpainting_dims, outpainting_ratio, return_mask= any_mask, outpainting_quantize_margins=outpainting_quantize_margins)
                    if any_mask: src_mask[:, pos:pos+1] = msk
        src_videos.append(src_video)
        src_masks.append(src_mask)
    return src_videos, src_masks


