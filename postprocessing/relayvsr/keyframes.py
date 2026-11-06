# Adapted from RelayVSR (https://github.com/kopperx/RelayVSR, Apache-2.0, see LICENSE in this folder): the low-resolution projection of its
# keyframe model, its streaming self-attention over the last keyframes and its spatial window. Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""RelayVSR's keyframe model: Wan 2.2 5B generating in one step (t = 1000) the latent of a keyframe from its low-resolution frame (native x4
or x2 checkpoints: the x4 one reads the frame upscaled x4), under a fixed prompt. Each layer attends the keys of the last four keyframes (the current one included), kept in a ring, within a window of
22 x 40 tokens around each token, shifted inside the frame at its borders. Keys are rotated once at their keyframe's index: the offsets
between a query and its keys are those of the original model, which rotates them again at each keyframe."""
import functools

import torch
import torch.nn as nn
from mmgp import auto_preload, runtime

from shared import attention_kit
from shared.attention import pay_attention
from models.wan.modules.model import WanAttentionBlock, WanModel, frame_modulation, modulate_frames, sinusoidal_embedding_1d

from .kernels import shifted_window_attention

CONFIG = {"model_type": "ti2v2_2", "patch_size": (1, 2, 2), "text_len": 512, "in_dim": 48, "dim": 3072, "ffn_dim": 14336, "freq_dim": 256, "text_dim": 4096,
          "out_dim": 48, "num_heads": 24, "num_layers": 30, "eps": 1e-6}
LATENT_CHANNELS = 48
CONTEXT_FRAMES = 4  # keyframes attended: the current one and the three before it
WINDOW = (22, 40)  # tokens attended around each token (rows, columns)
TIMESTEP = 1000.0
ROPE_DIMS = (44, 42, 42)
SPATIAL_FACTOR = 32  # pixels of the upscaled frame per token
RING = True  # comparison switch: False keeps the keyframe memory in GPU tensors


def rope_table(index, height, width, device):
    """Wan's 3D RoPE (cos, sin) (height * width, 128), channel pairs repeated, at temporal position index: FP32 from float64 angles."""
    angles = []
    for dim, positions in zip(ROPE_DIMS, (torch.full((1,), float(index), dtype=torch.float64), torch.arange(height, dtype=torch.float64), torch.arange(width, dtype=torch.float64))):
        angles.append(torch.outer(positions, 1.0 / 10000.0 ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim)))
    t, h, w = angles
    grid = torch.cat([t[:, None, None].expand(1, height, width, -1), h[None, :, None].expand(1, height, width, -1), w[None, None].expand(1, height, width, -1)], dim=-1)
    grid = grid.reshape(height * width, -1).repeat_interleave(2, dim=-1)
    return grid.cos().to(device=device, dtype=torch.float32), grid.sin().to(device=device, dtype=torch.float32)


class ChannelRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, 1, 1))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        return (y * torch.rsqrt(y.square().mean(dim=1, keepdim=True) + self.eps) * self.weight).to(x.dtype)


class LowResolutionProjection(nn.Module):
    """The tokens added to the patch embeddings: the x4 frame unshuffled by 16 pixels, two convolutions (the second by stride 2) and a linear."""

    def __init__(self, dim=3072, scale=4, hidden1=768, hidden2=3072):
        super().__init__()
        self.scale = scale  # x4: the frame upscaled x4 in 16 x 16 patches, x2: the frame itself in 8 x 8 patches
        self.conv1 = nn.Conv2d(3 * (4 * scale) ** 2, hidden1, kernel_size=1)
        self.norm1 = ChannelRMSNorm(hidden1)
        self.conv2 = nn.Conv2d(hidden1, hidden2, kernel_size=3, stride=2, padding=1)
        self.norm2 = ChannelRMSNorm(hidden2)
        self.out_proj = nn.Linear(hidden2, dim)

    def forward(self, frame):
        if self.scale == 4:
            frame = nn.functional.interpolate(frame, scale_factor=4, mode="bilinear", align_corners=False)
        x = nn.functional.pixel_unshuffle(frame, 4 * self.scale)
        x = nn.functional.silu(self.norm1(self.conv1(x)), inplace=True)
        x = nn.functional.silu(self.norm2(self.conv2(x)), inplace=True)
        return self.out_proj(x.flatten(2).transpose(1, 2))


class KeyframeCache:
    """The keys / values of the last CONTEXT_FRAMES keyframes at every layer, normalized and rotated: (CONTEXT_FRAMES * tokens, heads,
    head_dim) per layer, the keyframe of index i in the rows of slot i % CONTEXT_FRAMES, in q8 (int8 with FP16 scales per 32 channels). A state ring of the model's blocks (mmgp.runtime.state_ring) holds it: in RAM with in_ram (a minimal ring: the GPU holds the
    layer running only); otherwise, with MMGP's VRAM allocator, in VRAM as far as the VRAM allows and the other layers in RAM (resident="dynamic"). The
    layers in RAM are fetched one block ahead with the weights. Each keyframe replaces the rows of its slot (the first one stores the layers,
    written into the tensors the ring reserves for them), which the fetches leave out. Without the ring (no in_ram, no VRAM allocator): GPU
    tensors."""

    def __init__(self, blocks, tokens, height, width, heads, head_dim, device, in_ram=False):
        self.tokens, self.height, self.width, self.index, self.device = tokens, height, width, 0, device
        shape = (CONTEXT_FRAMES * tokens, heads, head_dim)
        self.shapes = [(shape, torch.int8), ((*shape[:-1], head_dim // attention_kit.GROUP), torch.float16)] * 2
        name = "RelayVSR keyframe memory"
        self.ring = None if not RING else runtime.state_ring(blocks, name, minimal=True) if in_ram else runtime.state_ring(blocks, name, resident="dynamic") if runtime.dynamic_state(blocks) else None
        self.layers = [None if self.ring is not None else self._new() for _ in blocks]  # with the ring: the lists its reserve() returns

    def _new(self):
        return [torch.empty(shape, dtype=dtype, device=self.device) for shape, dtype in self.shapes]

    def _rows(self, index):
        slot = index % CONTEXT_FRAMES
        return slice(slot * self.tokens, (slot + 1) * self.tokens)

    @property
    def frames(self):
        return min(self.index + 1, CONTEXT_FRAMES)

    def attention_fn(self, layer):
        """The attention of the current keyframe's queries at this layer (attention_kit attention_fn): its keys / values stored in their slot, then attended."""
        def attend(qkv_list, group):
            q, k, v = qkv_list
            qkv_list.clear()
            stored = self.layers[layer]
            if stored is None:  # the first keyframe with the ring: the tensors it reserves for the layer, stored once the layer ran
                stored = self.layers[layer] = self.ring.reserve(layer, self.shapes)
            rows, used, heads = self._rows(self.index), slice(0, self.frames * self.tokens), slice(group.kv.start, group.kv.stop)
            attended = []
            for tensor, data, scales in zip((k, v), stored[0::2], stored[1::2]):
                quantized, quantized_scales = attention_kit.quantize_q8(tensor)
                data[rows, heads] = quantized[0]
                scales[rows, heads] = quantized_scales[0]
                del quantized, quantized_scales
                out = torch.empty((1, used.stop, len(group.kv), k.shape[-1]), dtype=k.dtype, device=k.device)
                attended.append(attention_kit.dequantize_q8_into(out, data[used, heads].unsqueeze(0), scales[used, heads].unsqueeze(0)))
            keys, values = attended
            del k, v, stored, attended
            if self.height <= WINDOW[0] and self.width <= WINDOW[1]:  # the window holds the whole frame
                return pay_attention([q, keys[:, :, :q.shape[2]].contiguous(), values[:, :, :q.shape[2]].contiguous()])
            return shifted_window_attention(q, keys, values, self.height, self.width, WINDOW, self.frames, 0)
        return attend

    def commit(self, layer):
        """After the layer's attention (during its block's pass): the ring stores it (first keyframe) or this keyframe's rows; the next
        keyframe's rows are left out of its fetches."""
        if self.ring is None:
            return
        rows, skip = self._rows(self.index), self._rows(self.index + 1)
        if self.index == 0:
            self.ring.store(layer, list(self.layers[layer]), current=True, skip=skip, rows=rows)
        else:
            self.ring.update(layer, rows, skip=skip)

    def release(self):
        self.layers = []
        if self.ring is not None:
            self.ring.release()
            self.ring = None


class RelayWanBlock(WanAttentionBlock):
    """A Wan 2.2 block whose self-attention reads the keyframe cache."""

    def forward(self, x, e, context, freqs, attention_fn, commit):
        """commit(): after the attention, the keyframe memory of the layer kept (its keys and values for the next keyframes)."""
        e = frame_modulation(self.modulation.weight, e)
        attention_dtype = self.self_attn.q.weight.dtype
        x_list = [modulate_frames(x, self.norm1, e[0], e[1]).to(attention_dtype)]
        attention = self.self_attn
        y = attention_kit.qkv_attention(x_list, attention.q, attention.k, attention.v, attention.num_heads, attention.head_dim,
                                        lambda query, key, group: attention._norm_rope_(query, key, group, freqs), norm_spans_heads=True, attention_fn=attention_fn)
        commit()
        x = attention._project(y.flatten(2), x, e[2][0])
        y = None
        y_list = [self.norm3(x).to(attention_dtype)]
        x += self.cross_attn(y_list, context, None).to(x.dtype)
        y_list = [modulate_frames(x, self.norm2, e[3], e[4]).to(attention_dtype)]
        from shared.kernels import ops as kernels
        kernels.mlp(y_list, self.ffn[0], self.ffn[2], "gelu", residual=x, segments=kernels.gate_segments(e[5], x.shape[0], x.shape[1]))
        return x


class RelayVSRKeyframeModel(WanModel):
    """Wan 2.2 5B with RelayVSR's low-resolution projection and fixed prompt embeddings; keyframe() is its entry point."""
    _offload_hooks = ["keyframe"]
    _storing = False  # the first keyframe runs: it stores the keyframe memory of every layer

    def _auto_preload_workload(self, args, kwargs):
        """Workload of MMGP's automatic VRAM preload: by input size, the first keyframe apart (the memory it stores would count as
        activations of every keyframe)."""
        return auto_preload.workload(args, kwargs), self._storing

    def __init__(self, scale=4):
        super().__init__(**CONFIG)
        for block in self.blocks:
            block.__class__ = RelayWanBlock
        self.scale = scale  # the checkpoint's native upscaling, 4 or 2
        self.lq_proj = LowResolutionProjection(self.dim, scale)
        self.prompt_embeds = nn.Parameter(torch.empty(1, self.text_len, self.text_dim), requires_grad=False)

    def new_cache(self, height, width, device, in_ram=False):
        """A KeyframeCache for upscaled frames of height x width pixels (multiples of SPATIAL_FACTOR); in_ram: kept in RAM (minimal VRAM)."""
        rows, columns = height // SPATIAL_FACTOR, width // SPATIAL_FACTOR
        return KeyframeCache(self.blocks, rows * columns, rows, columns, self.num_heads, self.dim // self.num_heads, device, in_ram)

    def keyframe(self, frame, noise, cache, abort=None):
        """The latent (1, 48, h, w) float32 of a keyframe: frame (1, 3, 32 h / scale, 32 w / scale) in [-1, 1], the low-resolution frame; noise
        (1, 48, 1, 2 h, 2 w) float32; cache: its KeyframeCache, which then holds this keyframe. None when aborted."""
        dtype = self.time_projection[1].weight.dtype
        x = self.patch_embedding(noise.to(dtype).to(self.patch_embedding.weight.dtype)).to(dtype)  # the noise as the model reads it, in its dtype
        rows, columns = x.shape[-2:]
        x = x.flatten(2).transpose(1, 2).contiguous()
        x += self.lq_proj(frame.to(self.lq_proj.conv1.weight.dtype)).to(dtype)
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, torch.full((1,), TIMESTEP, device=x.device)).to(dtype))
        e0 = self.time_projection(e).unflatten(1, (6, self.dim)).to(e.dtype)
        context = self.text_embedding(self.prompt_embeds.to(self.text_embedding[0].weight.dtype))
        freqs = rope_table(cache.index, rows, columns, x.device)
        self._storing = cache.index == 0
        for layer, block in enumerate(self.blocks):
            if abort is not None and abort():
                return None
            x = block(x, e0, context, freqs, cache.attention_fn(layer), functools.partial(cache.commit, layer))
        cache.index += 1
        x = self.head([x], e)
        x = x.view(rows, columns, 2, 2, LATENT_CHANNELS).permute(4, 0, 2, 1, 3).reshape(1, LATENT_CHANNELS, rows * 2, columns * 2)
        return noise[:, :, 0] - x.float()


__all__ = ["CONTEXT_FRAMES", "KeyframeCache", "LATENT_CHANNELS", "RelayVSRKeyframeModel", "SPATIAL_FACTOR"]
