import torch, os, gc
from safetensors import safe_open
from contextlib import contextmanager
from einops import rearrange, repeat
import torch.nn as nn
import torch.nn.functional as F
import time
import hashlib
import types
from collections import deque
import numpy as np
from mmgp import transfer

CACHE_T = 2
FLASHVSR_LQ_PROJ_SPATIAL_TILING = True
FLASHVSR_LQ_PROJ_SPATIAL_TILE_SIZE = 128
FLASHVSR_LQ_PROJ_SPATIAL_TILE_HALO = 2
FLASHVSR_SHIFT_STILL_IMAGE_START_PREFIX = False
FLASHVSR_START_PREFIX_SHIFTS = ((2, 2), (1, 1), (0, 0))

def _cache_tail_cpu(x):
    return x[:, :, -CACHE_T:, :, :].detach().to(device="cpu", copy=True)


def _copy_token_tile_to_cpu(dst, src, y0, y1, x0, x1, full_h, full_w):
    src_cpu = src.detach().to("cpu")
    tile_h, tile_w = y1 - y0, x1 - x0
    frames = src.shape[1] // (tile_h * tile_w)
    for frame_idx in range(frames):
        src_frame = frame_idx * tile_h * tile_w
        dst_frame = frame_idx * full_h * full_w
        for row in range(tile_h):
            dst[:, dst_frame + (y0 + row) * full_w + x0:dst_frame + (y0 + row) * full_w + x1].copy_(src_cpu[:, src_frame + row * tile_w:src_frame + (row + 1) * tile_w])
    del src_cpu


def _crop_shifted_tile(frame, y0, y1, x0, x1, shift_y, shift_x):
    if shift_y == 0 and shift_x == 0:
        return frame[:, :, :, y0:y1, x0:x1]
    height, width = frame.shape[-2:]
    ys = torch.arange(y0 - shift_y, y1 - shift_y, device=frame.device).clamp_(0, height - 1)
    xs = torch.arange(x0 - shift_x, x1 - shift_x, device=frame.device).clamp_(0, width - 1)
    return frame.index_select(-2, ys).index_select(-1, xs)


def _linear_outputs_cpu(linear_layers, x):
    outputs = []
    for layer in linear_layers:
        y = layer(x)
        outputs.append(y.detach().to("cpu"))
        del y
    return outputs

@contextmanager
def init_weights_on_device(device = torch.device("meta"), include_buffers :bool = False):
    
    old_register_parameter = torch.nn.Module.register_parameter
    if include_buffers:
        old_register_buffer = torch.nn.Module.register_buffer
    
    def register_empty_parameter(module, name, param):
        old_register_parameter(module, name, param)
        if param is not None:
            param_cls = type(module._parameters[name])
            kwargs = module._parameters[name].__dict__
            kwargs["requires_grad"] = param.requires_grad
            module._parameters[name] = param_cls(module._parameters[name].to(device), **kwargs)

    def register_empty_buffer(module, name, buffer, persistent=True):
        old_register_buffer(module, name, buffer, persistent=persistent)
        if buffer is not None:
            module._buffers[name] = module._buffers[name].to(device)
            
    def patch_tensor_constructor(fn):
        def wrapper(*args, **kwargs):
            kwargs["device"] = device
            return fn(*args, **kwargs)

        return wrapper
    
    if include_buffers:
        tensor_constructors_to_patch = {
            torch_function_name: getattr(torch, torch_function_name)
            for torch_function_name in ["empty", "zeros", "ones", "full"]
        }
    else:
        tensor_constructors_to_patch = {}
    
    try:
        torch.nn.Module.register_parameter = register_empty_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = register_empty_buffer
        for torch_function_name in tensor_constructors_to_patch.keys():
            setattr(torch, torch_function_name, patch_tensor_constructor(getattr(torch, torch_function_name)))
        yield
    finally:
        torch.nn.Module.register_parameter = old_register_parameter
        if include_buffers:
            torch.nn.Module.register_buffer = old_register_buffer
        for torch_function_name, old_torch_function in tensor_constructors_to_patch.items():
            setattr(torch, torch_function_name, old_torch_function)

def load_state_dict_from_folder(file_path, torch_dtype=None):
    state_dict = {}
    for file_name in os.listdir(file_path):
        if "." in file_name and file_name.split(".")[-1] in [
            "safetensors", "bin", "ckpt", "pth", "pt"
        ]:
            state_dict.update(load_state_dict(os.path.join(file_path, file_name), torch_dtype=torch_dtype))
    return state_dict


def load_state_dict(file_path, torch_dtype=None):
    if file_path.endswith(".safetensors"):
        return load_state_dict_from_safetensors(file_path, torch_dtype=torch_dtype)
    else:
        return load_state_dict_from_bin(file_path, torch_dtype=torch_dtype)


def load_state_dict_from_safetensors(file_path, torch_dtype=None):
    state_dict = {}
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for k in f.keys():
            state_dict[k] = f.get_tensor(k)
            if torch_dtype is not None:
                state_dict[k] = state_dict[k].to(torch_dtype)
    return state_dict


def load_state_dict_from_bin(file_path, torch_dtype=None):
    state_dict = torch.load(file_path, map_location="cpu", weights_only=True)
    if torch_dtype is not None:
        for i in state_dict:
            if isinstance(state_dict[i], torch.Tensor):
                state_dict[i] = state_dict[i].to(torch_dtype)
    return state_dict


def search_for_embeddings(state_dict):
    embeddings = []
    for k in state_dict:
        if isinstance(state_dict[k], torch.Tensor):
            embeddings.append(state_dict[k])
        elif isinstance(state_dict[k], dict):
            embeddings += search_for_embeddings(state_dict[k])
    return embeddings


def search_parameter(param, state_dict):
    for name, param_ in state_dict.items():
        if param.numel() == param_.numel():
            if param.shape == param_.shape:
                if torch.dist(param, param_) < 1e-3:
                    return name
            else:
                if torch.dist(param.flatten(), param_.flatten()) < 1e-3:
                    return name
    return None


def build_rename_dict(source_state_dict, target_state_dict, split_qkv=False):
    matched_keys = set()
    with torch.no_grad():
        for name in source_state_dict:
            rename = search_parameter(source_state_dict[name], target_state_dict)
            if rename is not None:
                print(f'"{name}": "{rename}",')
                matched_keys.add(rename)
            elif split_qkv and len(source_state_dict[name].shape)>=1 and source_state_dict[name].shape[0]%3==0:
                length = source_state_dict[name].shape[0] // 3
                rename = []
                for i in range(3):
                    rename.append(search_parameter(source_state_dict[name][i*length: i*length+length], target_state_dict))
                if None not in rename:
                    print(f'"{name}": {rename},')
                    for rename_ in rename:
                        matched_keys.add(rename_)
    for name in target_state_dict:
        if name not in matched_keys:
            print("Cannot find", name, target_state_dict[name].shape)


def search_for_files(folder, extensions):
    files = []
    if os.path.isdir(folder):
        for file in sorted(os.listdir(folder)):
            files += search_for_files(os.path.join(folder, file), extensions)
    elif os.path.isfile(folder):
        for extension in extensions:
            if folder.endswith(extension):
                files.append(folder)
                break
    return files


def convert_state_dict_keys_to_single_str(state_dict, with_shape=True):
    keys = []
    for key, value in state_dict.items():
        if isinstance(key, str):
            if isinstance(value, torch.Tensor):
                if with_shape:
                    shape = "_".join(map(str, list(value.shape)))
                    keys.append(key + ":" + shape)
                keys.append(key)
            elif isinstance(value, dict):
                keys.append(key + "|" + convert_state_dict_keys_to_single_str(value, with_shape=with_shape))
    keys.sort()
    keys_str = ",".join(keys)
    return keys_str


def split_state_dict_with_prefix(state_dict):
    keys = sorted([key for key in state_dict if isinstance(key, str)])
    prefix_dict = {}
    for key in  keys:
        prefix = key if "." not in key else key.split(".")[0]
        if prefix not in prefix_dict:
            prefix_dict[prefix] = []
        prefix_dict[prefix].append(key)
    state_dicts = []
    for prefix, keys in prefix_dict.items():
        sub_state_dict = {key: state_dict[key] for key in keys}
        state_dicts.append(sub_state_dict)
    return state_dicts

def hash_state_dict_keys(state_dict, with_shape=True):
    keys_str = convert_state_dict_keys_to_single_str(state_dict, with_shape=with_shape)
    keys_str = keys_str.encode(encoding="UTF-8")
    return hashlib.md5(keys_str).hexdigest()

def clean_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if torch.mps.is_available():
        torch.mps.empty_cache()

def get_device_list():
    devs = []
    try:
        if hasattr(torch, "cuda") and hasattr(torch.cuda, "is_available") and torch.cuda.is_available():
            devs += [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    try:
        if hasattr(torch, "mps") and hasattr(torch.mps, "is_available") and torch.mps.is_available():
            devs += [f"mps:{i}" for i in range(torch.mps.device_count())]
    except Exception:
        pass
    return devs

class RMS_norm(nn.Module):
    
    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)
        
        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.
        
    def forward(self, x):
        return F.normalize(
            x, dim=(1 if self.channel_first else
                    -1)) * self.scale * self.gamma + self.bias
    
class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolusion.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)
        
    def forward(self, x_list, cache_x=None):
        x = x_list[0]
        x_list.clear()
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            # print(cache_x.shape, x.shape)
            cache_t = cache_x.shape[2]
            old_x = x
            x = old_x.new_empty(*old_x.shape[:2], cache_t + old_x.shape[2], *old_x.shape[3:])
            transfer.copy_(x[:, :, :cache_t, :, :], cache_x, non_blocking=True)  # (a cache in RAM: without a GPU copy of it)
            x[:, :, cache_t:, :, :].copy_(old_x)
            del cache_x, old_x
            padding[4] -= cache_t
            # print('cache!')
        x = F.pad(x, padding, mode='replicate') # mode='replicate'
        # print(x[0,0,:,0,0])
        
        return super().forward(x)
    
class PixelShuffle3d(nn.Module):
    def __init__(self, ff, hh, ww):
        super().__init__()
        self.ff = ff
        self.hh = hh
        self.ww = ww
        
    def forward(self, x):
        # x: (B, C, F, H, W)
        return rearrange(x, 
                         'b c (f ff) (h hh) (w ww) -> b (c ff hh ww) f h w',
                         ff=self.ff, hh=self.hh, ww=self.ww)
    
class Buffer_LQ4x_Proj(nn.Module):
    
    def __init__(self, in_dim, out_dim, layer_num=30):
        super().__init__()
        self.ff = 1
        self.hh = 16
        self.ww = 16
        self.hidden_dim1 = 2048
        self.hidden_dim2 = 3072
        self.layer_num = layer_num
        
        self.pixel_shuffle = PixelShuffle3d(self.ff, self.hh, self.ww)
        
        self.conv1 = CausalConv3d(in_dim*self.ff*self.hh*self.ww, self.hidden_dim1, (4, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)) # f -> f/2 h -> h w -> w
        self.norm1 = RMS_norm(self.hidden_dim1, images=False)
        self.act1 = nn.SiLU()
        
        self.conv2 = CausalConv3d(self.hidden_dim1, self.hidden_dim2, (4, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)) # f -> f/2 h -> h w -> w
        self.norm2 = RMS_norm(self.hidden_dim2, images=False)
        self.act2 = nn.SiLU()
        
        self.linear_layers = nn.ModuleList([nn.Linear(self.hidden_dim2, out_dim) for _ in range(layer_num)])
        
        self.clip_idx = 0
        
    def forward(self, video):
        self.clear_cache()
        # x: (B, C, F, H, W)
        
        t = video.shape[2]
        iter_ = 1 + (t - 1) // 4
        first_frame = video[:, :, :1, :, :].expand(-1, -1, 3, -1, -1)
        video = torch.cat([first_frame, video], dim=2)
        # print(video.shape)
        
        out_x = []
        for i in range(iter_):
            x = self.pixel_shuffle(video[:,:,i*4:(i+1)*4,:,:])
            cache1_x = _cache_tail_cpu(x)
            self.cache['conv1'] = cache1_x
            x = self.conv1(x, self.cache['conv1'])
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = _cache_tail_cpu(x)
            self.cache['conv2'] = cache2_x
            if i == 0:
                continue
            x = self.conv2(x, self.cache['conv2'])
            x = self.norm2(x)
            x = self.act2(x)
            out_x.append(x)
        out_x = torch.cat(out_x, dim = 2)
        # print(out_x.shape)
        out_x = rearrange(out_x, 'b c f h w -> b (f h w) c')
        outputs = []
        for i in range(self.layer_num):
            outputs.append(self.linear_layers[i](out_x))
        return outputs
    
    def clear_cache(self):
        self.cache = {}
        self.cache['conv1'] = None
        self.cache['conv2'] = None
        self.clip_idx = 0
        
    def stream_forward(self, video_clip):
        if self.clip_idx == 0:
            # self.clear_cache()
            first_frame = video_clip[:, :, :1, :, :].expand(-1, -1, 3, -1, -1)
            video_clip = torch.cat([first_frame, video_clip], dim=2)
            x = self.pixel_shuffle(video_clip)
            cache1_x = _cache_tail_cpu(x)
            self.cache['conv1'] = cache1_x
            x = self.conv1(x, self.cache['conv1'])
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = _cache_tail_cpu(x)
            self.cache['conv2'] = cache2_x
            self.clip_idx += 1
            return None
        else:
            x = self.pixel_shuffle(video_clip)
            cache1_x = _cache_tail_cpu(x)
            self.cache['conv1'] = cache1_x
            x = self.conv1(x, self.cache['conv1'])
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = _cache_tail_cpu(x)
            self.cache['conv2'] = cache2_x
            x = self.conv2(x, self.cache['conv2'])
            x = self.norm2(x)
            x = self.act2(x)
            out_x = rearrange(x, 'b c f h w -> b (f h w) c')
            del x
            outputs = _linear_outputs_cpu(self.linear_layers, out_x)
            del out_x
            self.clip_idx += 1
            return outputs

class Causal_LQ4x_Proj(nn.Module):
    
    def __init__(self, in_dim, out_dim, layer_num=30):
        super().__init__()
        self.ff = 1
        self.hh = 16
        self.ww = 16
        self.hidden_dim1 = 2048
        self.hidden_dim2 = 3072
        self.layer_num = layer_num
        
        self.pixel_shuffle = PixelShuffle3d(self.ff, self.hh, self.ww)
        
        self.conv1 = CausalConv3d(in_dim*self.ff*self.hh*self.ww, self.hidden_dim1, (4, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)) # f -> f/2 h -> h w -> w
        self.norm1 = RMS_norm(self.hidden_dim1, images=False)
        self.act1 = nn.SiLU()
        
        self.conv2 = CausalConv3d(self.hidden_dim1, self.hidden_dim2, (4, 3, 3), stride=(2, 1, 1), padding=(1, 1, 1)) # f -> f/2 h -> h w -> w
        self.norm2 = RMS_norm(self.hidden_dim2, images=False)
        self.act2 = nn.SiLU()
        
        self.linear_layers = nn.ModuleList([nn.Linear(self.hidden_dim2, out_dim) for _ in range(layer_num)])
        
        self.clip_idx = 0
        self.shift_start_prefix = False
        
    def forward(self, video):
        self.clear_cache()
        # x: (B, C, F, H, W)
        
        t = video.shape[2]
        iter_ = 1 + (t - 1) // 4
        first_frame = video[:, :, :1, :, :].expand(-1, -1, 3, -1, -1)
        video = torch.cat([first_frame, video], dim=2)
        # print(video.shape)
        
        out_x = []
        for i in range(iter_):
            x = self.pixel_shuffle(video[:,:,i*4:(i+1)*4,:,:])
            cache1_x = _cache_tail_cpu(x)
            x = self.conv1(x, self.cache['conv1'])
            self.cache['conv1'] = cache1_x
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = _cache_tail_cpu(x)
            if i == 0:
                self.cache['conv2'] = cache2_x
                continue
            x = self.conv2(x, self.cache['conv2'])
            self.cache['conv2'] = cache2_x
            x = self.norm2(x)
            x = self.act2(x)
            out_x.append(x)
        out_x = torch.cat(out_x, dim = 2)
        out_x = rearrange(out_x, 'b c f h w -> b (f h w) c')
        outputs = []
        for i in range(self.layer_num):
            outputs.append(self.linear_layers[i](out_x))
        return outputs
    
    def clear_cache(self):
        """Fully reset cache and clip index - use only at start of new video."""
        self.cache = {}
        self.cache['conv1'] = None
        self.cache['conv2'] = None
        self.clip_idx = 0
    
    def clean_mem(self):
        """Clean conv caches for memory but preserve clip_idx for streaming continuity."""
        # Don't reset clip_idx - it tracks streaming state across iterations
        # Only clear conv caches if they exist
        if hasattr(self, 'cache') and self.cache:
            if 'conv1' in self.cache and self.cache['conv1'] is not None:
                del self.cache['conv1']
                self.cache['conv1'] = None
            if 'conv2' in self.cache and self.cache['conv2'] is not None:
                del self.cache['conv2']
                self.cache['conv2'] = None

    def _start_prefix(self, first_frame, y0, y1, x0, x1):
        if not (self.shift_start_prefix and FLASHVSR_SHIFT_STILL_IMAGE_START_PREFIX):
            return first_frame[:, :, :, y0:y1, x0:x1].expand(-1, -1, 3, -1, -1)
        return torch.cat([_crop_shifted_tile(first_frame, y0, y1, x0, x1, shift_y, shift_x) for shift_y, shift_x in FLASHVSR_START_PREFIX_SHIFTS], dim=2)

    def _stream_forward_tiled(self, video_clip, need_output, prepend_first_frames=False):
        tile_size = int(FLASHVSR_LQ_PROJ_SPATIAL_TILE_SIZE)
        halo = int(FLASHVSR_LQ_PROJ_SPATIAL_TILE_HALO)
        height, width = video_clip.shape[-2] // self.hh, video_clip.shape[-1] // self.ww
        old_cache1, old_cache2 = self.cache['conv1'], self.cache['conv2']
        cache1_x = None
        cache2_x = None
        outputs = None
        for y0 in range(0, height, tile_size):
            y1 = min(y0 + tile_size, height)
            in_y0, in_y1 = max(0, y0 - halo), min(height, y1 + halo)
            conv2_y0, conv2_y1 = max(0, y0 - 1), min(height, y1 + 1)
            for x0 in range(0, width, tile_size):
                x1 = min(x0 + tile_size, width)
                in_x0, in_x1 = max(0, x0 - halo), min(width, x1 + halo)
                conv2_x0, conv2_x1 = max(0, x0 - 1), min(width, x1 + 1)
                py0, py1 = in_y0 * self.hh, in_y1 * self.hh
                px0, px1 = in_x0 * self.ww, in_x1 * self.ww
                video_tile = video_clip[:, :, :, py0:py1, px0:px1].contiguous()
                if prepend_first_frames:
                    first_frame = self._start_prefix(video_clip[:, :, :1], py0, py1, px0, px1)
                    video_tile = torch.cat([first_frame, video_tile], dim=2)
                    del first_frame
                x_tile = self.pixel_shuffle(video_tile)
                del video_tile
                tail = min(CACHE_T, x_tile.shape[2])
                if cache1_x is None:
                    cache1_x = torch.empty((x_tile.shape[0], x_tile.shape[1], tail, height, width), dtype=x_tile.dtype, device="cpu")
                transfer.copy_(cache1_x[:, :, :, y0:y1, x0:x1], x_tile[:, :, -tail:, y0 - in_y0:y1 - in_y0, x0 - in_x0:x1 - in_x0].detach())
                cache1_tile = None if old_cache1 is None else old_cache1[:, :, :, in_y0:in_y1, in_x0:in_x1]
                x_list = [x_tile]
                del x_tile
                tile = self.conv1(x_list, cache1_tile)
                tile = self.norm1(tile)
                tile = self.act1(tile)
                tail = min(CACHE_T, tile.shape[2])
                if cache2_x is None:
                    cache2_x = torch.empty((tile.shape[0], tile.shape[1], tail, height, width), dtype=tile.dtype, device="cpu")
                transfer.copy_(cache2_x[:, :, :, y0:y1, x0:x1], tile[:, :, -tail:, y0 - in_y0:y1 - in_y0, x0 - in_x0:x1 - in_x0].detach())
                if not need_output:
                    del tile
                    continue
                conv2_tile = tile[:, :, :, conv2_y0 - in_y0:conv2_y1 - in_y0, conv2_x0 - in_x0:conv2_x1 - in_x0].contiguous()
                del tile
                cache2_tile = None if old_cache2 is None else old_cache2[:, :, :, conv2_y0:conv2_y1, conv2_x0:conv2_x1]
                x_list = [conv2_tile]
                del conv2_tile
                tile = self.conv2(x_list, cache2_tile)
                tile = self.norm2(tile)
                tile = self.act2(tile)
                tile = tile[:, :, :, y0 - conv2_y0:y1 - conv2_y0, x0 - conv2_x0:x1 - conv2_x0].contiguous()
                out_x = rearrange(tile, 'b c f h w -> b (f h w) c')
                del tile
                if outputs is None:
                    outputs = [None] * self.layer_num
                for i, layer in enumerate(self.linear_layers):
                    y = layer(out_x)
                    if outputs[i] is None:
                        outputs[i] = torch.empty((y.shape[0], y.shape[1] // ((y1 - y0) * (x1 - x0)) * height * width, y.shape[2]), dtype=y.dtype, device="cpu")
                    _copy_token_tile_to_cpu(outputs[i], y, y0, y1, x0, x1, height, width)
                    del y
                del out_x
        self.cache['conv1'] = cache1_x
        self.cache['conv2'] = cache2_x
        del video_clip, old_cache1, old_cache2
        return outputs
        
    def stream_forward(self, video_clip_list):
        video_clip = video_clip_list[0]
        video_clip_list.clear()
        if self.clip_idx == 0:
            # self.clear_cache()
            if FLASHVSR_LQ_PROJ_SPATIAL_TILING:
                self._stream_forward_tiled(video_clip, False, prepend_first_frames=True)
                self.clip_idx += 1
                return None
            first_frame = self._start_prefix(video_clip[:, :, :1], 0, video_clip.shape[-2], 0, video_clip.shape[-1])
            video_clip = torch.cat([first_frame, video_clip], dim=2)
            del first_frame
            x = self.pixel_shuffle(video_clip)
            del video_clip
            cache1_x = _cache_tail_cpu(x)
            x_list = [x]
            del x
            x = self.conv1(x_list, self.cache['conv1'])
            self.cache['conv1'] = cache1_x
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = _cache_tail_cpu(x)
            self.cache['conv2'] = cache2_x
            self.clip_idx += 1
            return None
        else:
            if FLASHVSR_LQ_PROJ_SPATIAL_TILING:
                outputs = self._stream_forward_tiled(video_clip, True)
                self.clip_idx += 1
                return outputs
            x = self.pixel_shuffle(video_clip)
            del video_clip
            cache1_x = _cache_tail_cpu(x)
            x_list = [x]
            del x
            x = self.conv1(x_list, self.cache['conv1'])
            self.cache['conv1'] = cache1_x
            x = self.norm1(x)
            x = self.act1(x)
            cache2_x = _cache_tail_cpu(x)
            x_list = [x]
            del x
            x = self.conv2(x_list, self.cache['conv2'])
            self.cache['conv2'] = cache2_x
            x = self.norm2(x)
            x = self.act2(x)
            out_x = rearrange(x, 'b c f h w -> b (f h w) c')
            del x
            outputs = _linear_outputs_cpu(self.linear_layers, out_x)
            del out_x
            self.clip_idx += 1
            return outputs

class FrameStreamBuffer:
    def __init__(self, frame_generator: types.GeneratorType, buffer_size: int = 60, device='cpu', dtype=torch.float16):
        self.generator = frame_generator
        self.buffer_size = buffer_size
        self.device = device
        self.dtype = dtype
        
        self.buffer = deque()
        self.start_frame_index = 0
        self._fill_buffer(initial_fill_count=self.buffer_size)
        
    def _fill_buffer(self, initial_fill_count: int):
        try:
            for _ in range(initial_fill_count):
                frame = next(self.generator)
                self.buffer.append(frame)
        except StopIteration:
            pass
            
    def get_chunk(self, start: int, end: int) -> torch.Tensor:
        if start < self.start_frame_index:
            raise IndexError(f"Start frame {start} has already been discarded (current buffer starts at {self.start_frame_index})")

        while end > self.start_frame_index + len(self.buffer):
            try:
                self.buffer.append(next(self.generator))
            except StopIteration:
                if end > self.start_frame_index + len(self.buffer):
                    print(f"End frame {end} is out of range! It will be truncated to {self.start_frame_index + len(self.buffer)}")
                    end = self.start_frame_index + len(self.buffer)
                break

        while len(self.buffer) > self.buffer_size:
            self.buffer.popleft()
            self.start_frame_index += 1

        relative_start = start - self.start_frame_index
        relative_end = end - self.start_frame_index

        chunk_list = [self.buffer[i] for i in range(relative_start, relative_end)]
        if not chunk_list:
            C, H, W = self.buffer[0].shape
            return torch.empty((1, C, 0, H, W), device=self.device, dtype=self.dtype)

        chunk_tensor = torch.stack(chunk_list, dim=1) # (C, chunk_len, H, W)
        return chunk_tensor.unsqueeze(0).to(device=self.device) # (1, C, chunk_len, H, W)

class TensorAsBuffer:
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor
        
    def get_chunk(self, start: int, end: int) -> torch.Tensor:
        return self.tensor[:, :, start:end, :, :]

def tensor_to_imageio_frame(frame_tensor: torch.Tensor) -> np.ndarray:
    img_tensor = (frame_tensor + 1.0) / 2.0
    img_tensor_hwc = img_tensor.permute(1, 2, 0)
    img_tensor_hwc_u8 = (img_tensor_hwc * 255.0).clamp(0, 255).to(torch.uint8)
    img_np = img_tensor_hwc_u8.cpu().numpy()
    
    return img_np
