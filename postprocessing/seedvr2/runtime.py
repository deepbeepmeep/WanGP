from __future__ import annotations
from shared.utils.media_control import controlled_model_loading, loading_callback

import gc
from contextlib import contextmanager
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from mmgp import offload, transfer
from safetensors.torch import load_file
from tqdm import tqdm

from shared.utils import offload_registry
from .models.dit import NaDiT
from .models.dit.nablocks.attention.mmattn import qkv_split_map
from .models.video_vae_v3.modules import VideoAutoencoderKLWrapper
from .models.video_vae_v3.modules.causal_inflation_lib import InflatedCausalConv3d


_DIT_COMMON_CONFIG = {
    "vid_in_channels": 33,
    "vid_out_channels": 16,
    "txt_in_dim": 5120,
    "head_dim": 128,
    "expand_ratio": 4,
    "norm": "fusedrms",
    "norm_eps": 1e-5,
    "ada": "single",
    "qk_bias": False,
    "qk_norm": "fusedrms",
    "patch_size": (1, 2, 2),
    "block_type": "mmdit_sr",
    "window": (4, 3, 3),
    "attention_window_batch_size": 8,
}

DIT_CONFIGS = {
    "3b": {
        **_DIT_COMMON_CONFIG,
        "vid_dim": 2560,
        "vid_out_norm": "fusedrms",
        "txt_in_norm": "fusedln",
        "txt_dim": 2560,
        "emb_dim": 15360,
        "heads": 20,
        "num_layers": 32,
        "mm_layers": 10,
        "mlp_type": "swiglu",
        "window_method": [method for _ in range(16) for method in ("720pwin_by_size_bysize", "720pswin_by_size_bysize")],
        "rope_type": "mmrope3d",
        "rope_dim": 128,
    },
    # the original architecture: separate video and text weights in every block, GELU MLPs, video-only pixel RoPE, no output norm
    "7b": {
        **_DIT_COMMON_CONFIG,
        "vid_dim": 3072,
        "txt_dim": 3072,
        "emb_dim": 18432,
        "heads": 24,
        "num_layers": 36,
        "mm_layers": 36,
        "mlp_type": "normal",
        "window_method": [method for _ in range(18) for method in ("720pwin_by_size_bysize", "720pswin_by_size_bysize")],
        "rope_type": "rope3d",
        "rope_dim": 64,
        "vid_only_last_layer": False,
    },
}

VAE_CONFIG = {
    "act_fn": "silu",
    "block_out_channels": (128, 256, 512, 512),
    "down_block_types": ("DownEncoderBlock3D",) * 4,
    "up_block_types": ("UpDecoderBlock3D",) * 4,
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 16,
    "layers_per_block": 2,
    "norm_num_groups": 32,
    "slicing_sample_min_size": 4,
    "temporal_scale_num": 2,
    "inflation_mode": "pad",
    "spatial_downsample_factor": 8,
    "temporal_downsample_factor": 4,
    "use_quant_conv": False,
    "use_post_quant_conv": False,
    "freeze_encoder": False,
    "force_upcast": False,
}

VAE_TEMPORAL_TILE_SIZE = 8
OUTPUT_FRAME_BATCH_SIZE = 4


@contextmanager
def _default_dtype(dtype: torch.dtype):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


@dataclass(frozen=True)
class SeedVR2Paths:
    transformer: str
    vae: str
    positive_embedding: str


def _report(progress_callback, phase: str, current: int | None = None, total: int | None = None):
    if callable(progress_callback):
        progress_callback(phase, current, total)


def _pad_4n1(video: torch.Tensor) -> torch.Tensor:
    frames = video.shape[2]
    if frames % 4 == 1:
        return video
    count = ((frames - 1) // 4 + 1) * 4 + 1 - frames
    return torch.cat((video, video[:, :, -1:].expand(-1, -1, count, -1, -1)), dim=2)


def _resize_input(sample: torch.Tensor, height: int, width: int, device: torch.device) -> torch.Tensor:
    frames = sample.permute(1, 0, 2, 3).to(device=device, dtype=torch.float32)
    frames = frames.div_(255.0) if sample.dtype == torch.uint8 else frames.clamp_(-1.0, 1.0).add_(1.0).mul_(0.5)
    frames = F.interpolate(frames, size=(height, width), mode="bicubic", align_corners=False, antialias=True).clamp_(0.0, 1.0)
    return frames.mul_(2.0).sub_(1.0)


def _prepare_video(sample: torch.Tensor, height: int, width: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    video = _resize_input(sample, height, width, device).permute(1, 0, 2, 3).unsqueeze(0)
    pad_h, pad_w = (-height) % 16, (-width) % 16
    if pad_h or pad_w:
        video = F.pad(video, (0, pad_w, 0, pad_h), value=-1.0)
    return _pad_4n1(video).to(dtype)


def _wavelet_low(image: torch.Tensor) -> torch.Tensor:
    channels = image.shape[1]
    kernel = image.new_tensor(((0.0625, 0.125, 0.0625), (0.125, 0.25, 0.125), (0.0625, 0.125, 0.0625))).view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    for radius in (1, 2, 4, 8, 16):
        radius = min(radius, max(1, min(image.shape[-2:]) // 8))
        image = F.conv2d(F.pad(image, (radius,) * 4, mode="replicate"), kernel, groups=channels, dilation=radius)
    return image


def _color_fixed_frames(decoded: torch.Tensor, sample: torch.Tensor, height: int, width: int, device: torch.device, abort_callback=None) -> torch.Tensor:
    """The wavelet color fix against the input, then the uint8 frames, by chunks of frames on the GPU (decoded may be in RAM)."""
    frames = torch.empty(decoded.shape[1:], dtype=torch.uint8, device="cpu")
    for start in tqdm(range(0, decoded.shape[2], OUTPUT_FRAME_BATCH_SIZE), desc="SeedVR2 color", leave=False):
        if callable(abort_callback) and abort_callback():
            raise InterruptedError("SeedVR2 upscaling aborted")
        stop = min(start + OUTPUT_FRAME_BATCH_SIZE, decoded.shape[2])
        content = transfer.move(decoded[0, :, start:stop], device).permute(1, 0, 2, 3).float()
        style = _resize_input(sample[:, start:stop], height, width, device)
        count = stop - start
        low_input = torch.empty(count * 2, *content.shape[1:], device=device, dtype=torch.float32)
        low_input[:count].copy_(content)
        low_input[count:].copy_(style)
        style = None
        low_content, low_style = _wavelet_low(low_input).split(count)
        fixed = (content - low_content + low_style).clamp_(-1.0, 1.0).to(decoded.dtype)
        transfer.copy_(frames[:, start:stop], fixed.permute(1, 0, 2, 3).mul_(127.5).add_(127.5).round_().clamp_(0, 255).to(torch.uint8))
    return frames


def _window_starts(frame_count: int, window_size: int, overlap: int) -> tuple[int, ...]:
    if window_size < 0 or frame_count <= window_size:
        return (0,)
    if window_size <= overlap:
        raise ValueError("SeedVR2 window size must exceed its overlap")
    return tuple(range(0, frame_count - overlap, window_size - overlap))


def _crossfade_frames(previous: torch.Tensor, current: torch.Tensor) -> torch.Tensor:
    weights = torch.linspace(0.0, torch.pi, previous.shape[1], device=previous.device, dtype=torch.float32).cos_().mul_(-0.5).add_(0.5).view(1, -1, 1, 1)
    return previous.float().lerp_(current.float(), weights).round_().to(previous.dtype)


class SeedVR2Runtime:
    def __init__(self):
        self.dtype = torch.bfloat16
        self.device = torch.device("cuda")
        self.dit = None
        self.vae = None
        self.positive_embedding = None
        self.offloadobj = None
        self.profile = None
        self.model = None

    @controlled_model_loading
    def load(self, paths: SeedVR2Paths, *, model: str, init_pipe, profile) -> None:
        if self.dit is not None and self.profile == profile and self.model == model:
            return
        self.release()
        if not torch.cuda.is_available():
            raise RuntimeError("SeedVR2 requires CUDA.")
        self.profile, self.model = profile, model
        with init_empty_weights(include_buffers=True), _default_dtype(self.dtype):
            self.dit = NaDiT(**DIT_CONFIGS[model]).eval()
            self.vae = VideoAutoencoderKLWrapper(**VAE_CONFIG).eval()
        self.dit._offload_hooks = ["forward"]
        self.vae._offload_hooks = ["decode"]
        config = DIT_CONFIGS[model]
        offload.load_model_data(self.dit, paths.transformer, writable_tensors=False, default_dtype=self.dtype, fused_split_map=qkv_split_map(config["heads"], config["head_dim"]), verboseLevel=-1)
        offload.load_model_data(self.vae, paths.vae, writable_tensors=False, default_dtype=self.dtype, verboseLevel=-1)
        self.dit.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.vae.debug = None
        self.vae.set_causal_slicing(split_size=VAE_TEMPORAL_TILE_SIZE, memory_device="same")
        self.vae.set_memory_limit(conv_max_mem=0.25, norm_max_mem=0.5)
        self.positive_embedding = load_file(paths.positive_embedding, device="cpu")["embedding"].to(self.dtype)
        pipe = {"transformer": self.dit, "vae": self.vae}
        kwargs = {}
        profile_no = init_pipe(pipe, kwargs, profile)
        kwargs["pinnedMemory"] = False
        self.offloadobj = offload.profile(pipe, loading_callback=loading_callback(), profile_no=profile_no, quantizeTransformer=False, convertWeightsFloatTo=self.dtype, verboseLevel=-1, **kwargs)
        offload_registry.register_offloadobj("SeedVR2", self.offloadobj, self.release)

    def _clear_vae_memory(self):
        if self.vae is not None:
            for module in self.vae.modules():
                if isinstance(module, InflatedCausalConv3d):
                    module.memory = None

    def release(self) -> None:
        self._clear_vae_memory()
        if self.offloadobj is not None:
            offload_registry.unregister_offloadobj("SeedVR2", self.offloadobj)
            self.offloadobj.release()
            self.offloadobj = None
        self.dit = self.vae = self.positive_embedding = self.profile = self.model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @torch.inference_mode()
    def _upscale_window(self, sample: torch.Tensor, scale: float, *, seed: int = 0, vae_tile_size: int | None = None, abort_callback=None, progress_callback=None):
        if self.dit is None or self.vae is None:
            raise RuntimeError("SeedVR2 models are not loaded.")
        input_frames = sample.shape[1]
        output_height = max(1, int(sample.shape[-2] * scale))
        output_width = max(1, int(sample.shape[-1] * scale))
        tile_size = int(vae_tile_size or 512)
        tiled = True
        overlap = min(128, tile_size // 4)
        self.vae.encoder.abort_callback = abort_callback
        self.vae.decoder.abort_callback = abort_callback

        try:
            with tqdm(total=4, desc="SeedVR2") as progress:
                if callable(abort_callback) and abort_callback():
                    return None, None
                _report(progress_callback, "SeedVR2 preprocessing", 0, 1)
                conditioning = _prepare_video(sample, output_height, output_width, torch.device("cpu"), self.dtype)
                _report(progress_callback, "SeedVR2 preprocessing", 1, 1)
                progress.update()

                encode_progress = lambda current, total: _report(progress_callback, "SeedVR2 VAE encoding chunks", current, total)
                latent = self.vae.encode(conditioning, tiled=tiled, tile_size=(tile_size, tile_size), tile_overlap=(overlap, overlap), progress_callback=encode_progress).latent
                if latent.ndim == 4:
                    latent = latent.unsqueeze(2)
                latent = latent.mul_(0.9152).permute(0, 2, 3, 4, 1)[0].contiguous()
                conditioning = None
                progress.update()

                _report(progress_callback, "SeedVR2 denoising layers", 0, len(self.dit.blocks))
                generator = torch.Generator(device=self.device).manual_seed(int(seed))
                noise = torch.randn(latent.shape, generator=generator, device=self.device, dtype=self.dtype)
                dit_input = torch.empty(*latent.shape[:-1], 33, device=self.device, dtype=self.dtype)
                dit_input[..., :16].copy_(noise)
                dit_input[..., 16:32].copy_(latent)
                dit_input[..., 32].fill_(1)
                latent = None
                vid_shape = torch.tensor([noise.shape[:3]], device=self.device)
                text = self.positive_embedding.to(self.device)
                text_shape = torch.tensor([[text.shape[0]]], device=self.device)
                vid_list, txt_list = [dit_input.reshape(-1, 33)], [text]
                dit_input = text = None
                pred = self.dit(vid_list=vid_list, txt_list=txt_list, vid_shape=vid_shape, txt_shape=text_shape, timestep=torch.tensor([1000.0], device=self.device, dtype=self.dtype), abort_callback=abort_callback, progress_callback=progress_callback).vid_sample
                latent = noise.sub_(pred.reshape_as(noise))
                noise = pred = None
                progress.update()

                decode_progress = lambda current, total: _report(progress_callback, "SeedVR2 VAE decoding chunks", current, total)
                latent = latent.permute(3, 0, 1, 2).unsqueeze(0).div_(0.9152)
                decoded = self.vae.decode(latent, tiled=tiled, tile_size=(tile_size, tile_size), tile_overlap=(overlap, overlap), progress_callback=decode_progress).sample
                if decoded.ndim == 4:
                    decoded = decoded.unsqueeze(2)
                decoded = decoded[:, :, :input_frames, :output_height, :output_width]
                frames = _color_fixed_frames(decoded, sample, output_height, output_width, self.device, abort_callback)
                progress.update()
                return frames, None
        except InterruptedError:
            return None, None
        finally:
            self._clear_vae_memory()

    @torch.inference_mode()
    def upscale(self, sample: torch.Tensor, scale: float, *, seed: int = 0, vae_tile_size: int | None = None, window_size: int = -1, window_overlap: int = 3, abort_callback=None, progress_callback=None):
        if self.dit is None or self.vae is None:
            raise RuntimeError("SeedVR2 models are not loaded.")
        frame_count = int(sample.shape[1])
        starts = _window_starts(frame_count, int(window_size), int(window_overlap))
        output = None
        for window_index, start in enumerate(tqdm(starts, desc="SeedVR2 windows", unit="window", disable=len(starts) == 1), start=1):
            stop = frame_count if int(window_size) < 0 else min(start + int(window_size), frame_count)
            if len(starts) == 1:
                window_progress = progress_callback
            else:
                window_progress = lambda phase, current=None, total=None, index=window_index: _report(progress_callback, f"{phase} (Chunk {index} / {len(starts)})", current, total)
            window_output, _ = self._upscale_window(sample[:, start:stop], scale, seed=seed, vae_tile_size=vae_tile_size, abort_callback=abort_callback, progress_callback=window_progress)
            if window_output is None:
                return None, None
            if output is None:
                output = window_output.new_empty(window_output.shape[0], frame_count, *window_output.shape[2:])
                output[:, :stop].copy_(window_output)
            else:
                overlap = min(int(window_overlap), int(window_output.shape[1]))
                output[:, start:start + overlap].copy_(_crossfade_frames(output[:, start:start + overlap], window_output[:, :overlap]))
                output[:, start + overlap:stop].copy_(window_output[:, overlap:])
            window_output = None
        _report(progress_callback, "SeedVR2 complete")
        return output, None


_RUNTIME = SeedVR2Runtime()


def load_models(paths: SeedVR2Paths, *, model: str, init_pipe, profile, progress_callback=None):
    _report(progress_callback, "Caching SeedVR2")
    _RUNTIME.load(paths, model=model, init_pipe=init_pipe, profile=profile)


def upscale_video(sample, scale: float, **kwargs):
    return _RUNTIME.upscale(sample, scale, **kwargs)


def release_models():
    _RUNTIME.release()
