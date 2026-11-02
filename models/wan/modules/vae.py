# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modifications Copyright (c) 2025-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import logging
import os
from mmgp import offload, transfer
import torch
from shared.utils.utils import guide_to_dtype
from shared.utils.phase_progress import vae_encoding_progress, PhaseProgress, check_abort, set_phase_status
import torch.cuda.amp as amp
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from shared.kernels import kernel_policy, vae_ops
from shared.cuda_memory import mmgp_vram_allocator

try:
    from . import vae_kernels as _kernels
except (ImportError, OSError, RuntimeError):
    _kernels = None

__all__ = [
    'WanVAE',
]

CACHE_T = 2


def _use_fast(x):
    """Channels-last activations with the Triton kernels: "fast" kernel precision, 16-bit VAE (FP32 VAEs keep the PyTorch path, whose
    cuDNN algorithms reserve less memory there), CUDA GPUs with Triton."""
    return _kernels is not None and x.is_cuda and x.dtype != torch.float32 and kernel_policy.allow_approximate() and _kernels.supported(x.device)


def _causal_input(x, conv, feat_cache, feat_idx, norm=None):
    """Fast path: the padded channels-last input of causal conv, silu(norm(x)) (norm called as a module, for the offload and abort hooks)
    or x, after the cached last frames of the previous chunk; caches its own last CACHE_T frames for the next chunk."""
    if feat_cache is None:
        return _kernels.causal_pad(x, conv._padding) if norm is None else norm(x, pad=conv._padding)
    idx = feat_idx[0]
    cache = feat_cache[idx]
    x, feat_cache[idx] = _kernels.causal_pad(x, conv._padding, cache, keep=CACHE_T) if norm is None else norm(x, pad=conv._padding, cache=cache, keep=CACHE_T)
    feat_idx[0] += 1
    return x


def _residual_fast(block, x, feat_cache, feat_idx):
    """ResidualBlock on channels-last activations: each norm + SiLU + padding in one kernel, the residual added by the last convolution."""
    norm1, _, conv1, norm2, _, _, conv2 = block.residual
    h = block.shortcut(x, padded=True) if isinstance(block.shortcut, nn.Conv3d) else x
    x = conv1(_causal_input(x, conv1, feat_cache, feat_idx, norm1), padded=True)
    return conv2(_causal_input(x, conv2, feat_cache, feat_idx, norm2), padded=True, residual=h)


def _resample_fast(module, x, feat_cache, feat_idx):
    """Resample on channels-last activations: frames as a (b t) c h w view, the nearest x2 upsampling read in place by the Triton
    convolution or made in the activation dtype (an exact copy, without the FP32 round trip), the causal time convolutions without padding
    copies."""
    b, c, t, h, w = x.shape
    if module.mode == 'upsample3d' and feat_cache is not None:
        idx = feat_idx[0]
        if feat_cache[idx] is None:
            feat_cache[idx] = 'Rep'
            feat_idx[0] += 1
        else:  # 'Rep': zero frames before the first chunk, as no cache
            x, feat_cache[idx] = _kernels.causal_pad(x, module.time_conv._padding, None if isinstance(feat_cache[idx], str) else feat_cache[idx], keep=CACHE_T)
            feat_idx[0] += 1
            x = module.time_conv(x, padded=True)  # (b, 2c, t, h, w): frame 2i from the first c channels of frame i, 2i + 1 from the others
            x = x.permute(0, 2, 3, 4, 1).unflatten(4, (2, c)).transpose(2, 4).transpose(3, 4).reshape(b, 2 * t, h, w, c).permute(0, 4, 1, 2, 3)
    t = x.shape[2]
    x = x.transpose(1, 2).flatten(0, 1)
    if module.mode.startswith('upsample'):
        upsample, conv = module.resample
        x = _kernels.conv(x, conv.weight, conv.bias, upsample=True) if _kernels.conv_eligible(x, conv, upsample=True) else conv(F.interpolate(x, scale_factor=upsample.scale_factor, mode=upsample.mode))
    else:
        x = module.resample(x)
    x = x.contiguous(memory_format=torch.channels_last).unflatten(0, (b, t)).transpose(1, 2)  # cuDNN may return one frame NCHW
    if module.mode == 'downsample3d' and feat_cache is not None:  # the first chunk passes; later ones start from the last frame before
        idx = feat_idx[0]
        cache_x = x[:, :, -1:].clone()
        if feat_cache[idx] is not None:
            x = module.time_conv(torch.cat([feat_cache[idx][:, :, -1:], x], 2), padded=True)
        feat_cache[idx] = cache_x
        feat_idx[0] += 1
    return x


def _attention_fast(block, x):
    """AttentionBlock on channels-last activations, as tokens (a view): keys and values of the frame, then the queries, attention,
    projection and residual for ~4096 tokens at a time, instead of the frame's q / k / v / output copies. Every query is computed alone
    (the 1x1 convolutions as matrix products round in another order)."""
    b, c, t, h, w = x.shape
    tokens = x.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c)
    normed = F.normalize(tokens, dim=-1).mul_(block.norm.scale).mul_(block.norm.gamma.view(c))  # RMS_norm, its bias is 0
    weight, bias = block.to_qkv.weight.view(3 * c, c), block.to_qkv.bias
    k, v = (F.linear(normed, weight[i * c:(i + 1) * c], bias[i * c:(i + 1) * c]).unsqueeze(1) for i in (1, 2))
    out = torch.empty_like(x)
    out_tokens = out.permute(0, 2, 3, 4, 1).reshape(b * t, h * w, c)
    for start in range(0, h * w, 4096):
        stop = min(h * w, start + 4096)
        y = F.scaled_dot_product_attention(F.linear(normed[:, start:stop], weight[:c], bias[:c]).unsqueeze(1), k, v).squeeze(1)
        out_tokens[:, start:stop] = F.linear(y, block.proj.weight.view(c, c), block.proj.bias).add_(tokens[:, start:stop])
    return out


def _kind(stage):
    """Region rule of a fast-path stage and its halo: 'same' (zero-padded 3x3 convolutions), 'up' (nearest x2 + 3x3 convolution), 'down'
    (right / bottom zero pad + 3x3 stride 2 convolution), 'block' (Wan 2.2 residual blocks: their stages) or 'global' (attention)."""
    if isinstance(stage, (nn.Conv3d, nn.Sequential)):  # first convolution, head
        return 'same', 1
    if hasattr(stage, 'residual'):
        return 'same', 2
    if hasattr(stage, 'to_qkv'):
        return 'global', 0
    if hasattr(stage, 'mode'):
        return ('up' if stage.mode.startswith('up') else 'down'), 1
    return 'block', 0


def _inner(block):
    return list(block.upsamples if hasattr(block, 'upsamples') else block.downsamples)


def _scales(stage):
    kind = _kind(stage)[0]
    return kind in ('up', 'down') or kind == 'block' and any(_scales(inner) for inner in _inner(stage))


def _out_size(stage, size):
    kind = _kind(stage)[0]
    if kind == 'block':
        for inner in _inner(stage):
            size = _out_size(inner, size)
        return size
    return 2 * size if kind == 'up' else (size - 2) // 2 + 1 if kind == 'down' else size


def _sizes(stages, size):
    """Frame size (h, w) at the input of each stage, and at the output of the last one."""
    sizes = [size]
    for stage in stages:
        sizes.append(tuple(_out_size(stage, s) for s in sizes[-1]))
    return sizes


def _plan(stages, size, region):
    """Regions for `region` ((rows), (cols)) at the output of stages run on frames of `size` (h, w): per stage (output region, inner plan
    of blocks, input region, input size), and the input region needed. Each zero-padded convolution corrupts one pixel next to a region
    border that is not a frame border, so the halos make every delivered value equal to the whole-frame one."""
    plan = []
    for stage, size in zip(reversed(stages), reversed(_sizes(stages, size)[:-1])):
        kind, halo = _kind(stage)
        if kind == 'block':
            inner_plan, need = _plan(_inner(stage), size, region)
        else:
            inner_plan, need = None, []
            for (lo, hi), s in zip(region, size):
                if kind == 'up':
                    lo, hi = max(lo - 1, 0) // 2, (min(hi + 1, 2 * s) - 1) // 2 + 1
                elif kind == 'down':
                    lo, hi = 2 * lo, min(2 * hi + 1, s)
                else:
                    lo, hi = max(lo - halo, 0), min(hi + halo, s)
                need.append((lo, hi))
            need = tuple(need)
        plan.append((region, inner_plan, need, size))
        region = need
    return plan[::-1], region


def _flat(stages, plan):
    """(kind, input region, input size) of every stage, blocks expanded."""
    for stage, (_, inner_plan, need, size) in zip(stages, plan):
        if inner_plan is None:
            yield _kind(stage)[0], need, size
        else:
            yield from _flat(_inner(stage), inner_plan)


def _crop(x, cover, region):
    """x, covering the region `cover`, cut to `region` (channels last)."""
    (r0, _), (c0, _) = cover
    (a, b), (c, d) = region
    if (a - r0, c - c0, b - a, d - c) == (0, 0, x.shape[-2], x.shape[-1]):
        return x
    return x[..., a - r0:b - r0, c - c0:d - c0].contiguous(memory_format=torch.channels_last_3d)


def _call(stage, x, feat_cache, feat_idx, first_chunk):
    if isinstance(stage, nn.Sequential):  # head: norm, SiLU, convolution
        return stage[2](_causal_input(x, stage[2], feat_cache, feat_idx, stage[0]), padded=True)
    if isinstance(stage, nn.Conv3d):
        return stage(_causal_input(x, stage, feat_cache, feat_idx), padded=True)
    if hasattr(stage, 'to_qkv'):
        return stage(x, fast=True)
    if hasattr(stage, 'upsamples'):
        return stage(x, feat_cache, feat_idx, first_chunk, fast=True)
    return stage(x, feat_cache, feat_idx, fast=True)


def _run_region(stages, plan, x, cover, feat_cache, feat_idx, first_chunk):
    """stages on x covering `cover`, each output cut to its planned region."""
    for stage, (region, inner_plan, _, _) in zip(stages, plan):
        check_abort()
        kind = _kind(stage)[0]
        if kind == 'block':  # Wan 2.2: main path region by region, the shortcut cut to the same region
            main, shortcut = _run_region(_inner(stage), inner_plan, x, cover, feat_cache, feat_idx, first_chunk), stage.avg_shortcut
            if hasattr(stage, 'upsamples'):
                if shortcut is not None:  # a pixel gather: only the input pixels of the region, not the main path's halo
                    source = tuple((lo // 2, (hi - 1) // 2 + 1) for lo, hi in region)
                    main.add_(_crop(shortcut(_crop(x, cover, source), first_chunk), tuple((2 * lo, 2 * hi) for lo, hi in source), region))
            else:
                main.add_(shortcut(_crop(x, cover, tuple((shortcut.factor_s * lo, shortcut.factor_s * hi) for lo, hi in region))))
            x = main
        else:
            y = _call(stage, x, feat_cache, feat_idx, first_chunk)
            out = cover if kind == 'same' else tuple((2 * lo, 2 * hi) for lo, hi in cover) if kind == 'up' else tuple((lo // 2, lo // 2 + n) for (lo, _), n in zip(cover, y.shape[-2:]))
            x = _crop(y, out, region)
        cover = region
    return x


def _run_part(net, x, part, feat_cache, feat_idx):
    """Stages start:stop of an Encoder3d / Decoder3d on the whole frame (plan None) or on the region `cover` of the frame (plan)."""
    start, stop, plan, cover, first_chunk = part
    stages = net._stages()[start:stop]
    if plan is not None:
        return _run_region(stages, plan, x, cover, feat_cache, feat_idx, first_chunk)
    for stage in stages:
        check_abort()
        x = _call(stage, x, feat_cache, feat_idx, first_chunk)
    return x


def _split(sizes, budget):
    """Non-overlapping ranges covering each axis in equal parts of at most budget."""
    out = []
    for size in sizes:
        count = -(-size // max(1, budget))
        bounds = [size * i // count for i in range(count + 1)]
        out.append(list(zip(bounds[:-1], bounds[1:])))
    return [(rows, cols) for rows in out[0] for cols in out[1]]


def _segments(net, size, splits, tile):
    """Exact tiling of net (Encoder3d / Decoder3d) for frames of `size` at its input: [(start, stop, tiles)] for the stage ranges between
    `splits`. The attention runs on the whole frame (tiles None), the other ranges tile after tile with halos, the regions (halos
    included) within `tile` pixels at their largest resolution (decoder output / encoder input), encoder ranges at lower resolution
    within tile x sqrt(their scale), ranges at the latent resolution within tile / 2 (more channels and causal caches per pixel).
    Consecutive whole-frame ranges are merged (no round trip through the CPU)."""
    stages, bounds, out, full = net._stages(), [0, *splits, len(net._stages())], [], size[0]
    for start, stop in zip(bounds[:-1], bounds[1:]):
        part = stages[start:stop]
        out_size = _sizes(part, size)[-1]
        if any(_kind(stage)[0] == 'global' for stage in part):
            tiles = None
        else:  # halo measured on a centered 64-pixel probe tile of a large frame
            factor = size[0] // out_size[0] if size[0] >= out_size[0] else 1 / (out_size[0] // size[0])
            big = (round(4096 * factor),) * 2 if factor >= 1 else (round(512 * factor),) * 2
            big_out = _sizes(part, big)[-1]
            plan, need = _plan(part, big, tuple(((s - 64) // 2, (s - 64) // 2 + 64) for s in big_out))
            if factor > 1:  # encoder: largest at the input
                budget = (tile * (size[0] / full) ** 0.5 - (need[0][1] - need[0][0] - 64 * factor)) // factor
            elif factor == 1:
                budget = tile // 2 - (need[0][1] - need[0][0] - 64)
            else:  # decoder: largest at the output, where up stages also hold their upsampled input; intermediate outputs: tile / 2
                widths = [(hi - lo) * (2 if kind == 'up' else 1) for kind, ((lo, hi), _), in_size in _flat(part, plan) if in_size[0] * (2 if kind == 'up' else 1) == big_out[0]]
                budget = (tile if stop == len(stages) else tile // 2) - (max(widths) - 64)
            tiles = _split(out_size, max(int(budget), 8))
            tiles = None if len(tiles) == 1 else tiles
        if tiles is None and out and out[-1][2] is None:
            out[-1] = (out[-1][0], stop, None)
        else:
            out.append((start, stop, tiles))
        size = out_size
    return out


def _exact_segments(net, size, tile, decoder):
    """Exact tiling of a decoder or an encoder for frames of `size` at its input: only the attention needs the whole frame; the stages
    at the latent resolution around it and the scaling ones are tiled (see _segments)."""
    stages = net._stages()
    attention = next(i for i, stage in enumerate(stages) if _kind(stage)[0] == 'global')
    if decoder:  # a first residual block with an upsampling (Wan 2.2) runs alone: its halo at the latent resolution is the largest
        first = next(i for i in range(attention, len(stages)) if _scales(stages[i]))
        return _segments(net, size, [attention, attention + 1, first] + ([first + 1] if _kind(stages[first])[0] == 'block' else []), tile)
    scaling = [i for i in range(attention) if _scales(stages[i])]
    segments = _segments(net, size, [scaling[-1] + 1, attention, attention + 1], tile)
    start, stop, tiles = segments[0]
    if tiles is not None and sum((b - a) * (d - c) for (a, b), (c, d) in (_plan(stages[start:stop], size, region)[1] for region in tiles)) > 2 * size[0] * size[1]:
        # tiles small for the encoder's halo (it grows by 2x per downsampling): a first range ending at quarter resolution, its outputs
        # kept on the CPU, has a 4x smaller halo at full resolution
        segments = _segments(net, size, [scaling[1] + 1, scaling[-1] + 1, attention, attention + 1], tile)
    return segments


def _exact_calls(segments, chunks):
    return chunks * sum(1 if tiles is None else len(tiles) for _, _, tiles in segments)


def _exact_tiled(net, chunks, segments, size, sink):
    """Tiled run of net (Encoder3d / Decoder3d, fast path) with the result of the whole-frame run: segment after segment over all chunks
    ([(input of the first stage, cached, first_chunk)]); whole-frame segments with one causal cache, tiled ones tile after tile, each tile
    with its own caches on its input region (halos included), delivering its tile only (no overlap, no blending). Outputs between
    segments wait on the CPU; the last segment's go to sink(chunk, frame offset, region, output)."""
    device, inputs = chunks[0][0].device, [x for x, _, _ in chunks]
    for number, (start, stop, tiles) in enumerate(segments):
        out_size, last = _sizes(net._stages()[start:stop], size)[-1], number == len(segments) - 1
        store = [None] * len(chunks)
        work = [(tuple((0, s) for s in out_size), None, tuple((0, s) for s in size))] if tiles is None else [(region, *_plan(net._stages()[start:stop], size, region)) for region in tiles]
        for region, plan, need in sorted(work, key=lambda item: -(item[2][0][1] - item[2][0][0]) * (item[2][1][1] - item[2][1][0])):  # largest first: the allocator reuses their blocks
            (a, b), (c, d) = need
            cache, offset = [None] * sum(isinstance(m, nn.Conv3d) for m in net.modules()), 0
            for i, (_, cached, first) in enumerate(chunks):
                x = inputs[i][..., a:b, c:d].to(device).contiguous(memory_format=torch.channels_last_3d)
                y = net(x, feat_cache=cache if cached else None, feat_idx=[0], part=(start, stop, plan, need, first))
                x = None
                if last:
                    sink(i, offset, region, y)
                else:
                    if store[i] is None:
                        store[i] = torch.empty((*y.shape[:3], *out_size), dtype=y.dtype, device="cpu", memory_format=torch.channels_last_3d)
                    store[i][..., region[0][0]:region[0][1], region[1][0]:region[1][1]].copy_(y)
                offset += y.shape[2]
                y = None
        inputs, size = store, out_size


def _exact_decode(model, z, segments, any_end_frame, sink):
    """Tiled decoding equal to the untiled one (fast path) of scaled latents z on the GPU, through _exact_tiled."""
    x = model.conv2(z)
    count = x.shape[2]
    _exact_tiled(model.decoder, [(x[:, :, i:i + 1], not (any_end_frame and i == count - 1), i == 0) for i in range(count)], segments, tuple(x.shape[-2:]), sink)


def _exact_encode(model, x, scale, tile_size, any_end_frame):
    """Tiled encoding equal to the untiled one (fast path) of x on the GPU (Wan 2.2: in patches): the encoder up to its last downsampling
    tile after tile with halos, the rest (attention included) on the whole latent frame."""
    count = 2 + (x.shape[2] - 2) // 4 if any_end_frame else 1 + (x.shape[2] - 1) // 4
    chunks = [(x[:, :, max(0, 4 * i - 3):4 * i + 1], True, False) for i in range(count)]
    if any_end_frame:
        chunks[-1] = (x[:, :, -1:], False, False)
    out_list = [None] * count

    def sink(i, offset, region, y):
        if out_list[i] is None:
            out_list[i] = y.new_empty((*y.shape[:3], x.shape[-2] // 8, x.shape[-1] // 8))
        out_list[i][..., region[0][0]:region[0][1], region[1][0]:region[1][1]] = y
    _exact_tiled(model.encoder, chunks, _exact_segments(model.encoder, tuple(x.shape[-2:]), tile_size, False), tuple(x.shape[-2:]), sink)
    mu = model.conv1(torch.cat(out_list, 2)).chunk(2, dim=1)[0]
    if isinstance(scale[0], torch.Tensor):
        return (mu - scale[0].view(1, model.z_dim, 1, 1, 1)) * scale[1].view(1, model.z_dim, 1, 1, 1)
    return (mu - scale[0]) * scale[1]


def _vae_float_to_cpu_uint8(frames):
    frames.clamp_(-1.0, 1.0).add_(1.0).mul_(127.5).round_().clamp_(0.0, 255.0)
    return transfer.move(frames, "cpu", torch.uint8)  # (without a RAM copy of the video in its float dtype, nor a GPU one of a cropped view)


def _blend_v_edge_(top_edge, tile, blend_extent):
    blend_extent = min(int(top_edge.shape[-2]), int(tile.shape[-2]), int(blend_extent))
    if blend_extent <= 0:
        return
    weights = torch.arange(blend_extent, device=tile.device, dtype=tile.dtype).div_(blend_extent).view(1, 1, 1, blend_extent, 1)
    edge = top_edge[:, :, :, -blend_extent:, :].to(device=tile.device, dtype=tile.dtype)
    edge.mul_(1.0 - weights)
    tile[:, :, :, :blend_extent, :].mul_(weights).add_(edge)


def _blend_h_edge_(left_edge, tile, blend_extent):
    blend_extent = min(int(left_edge.shape[-1]), int(tile.shape[-1]), int(blend_extent))
    if blend_extent <= 0:
        return
    weights = torch.arange(blend_extent, device=tile.device, dtype=tile.dtype).div_(blend_extent).view(1, 1, 1, 1, blend_extent)
    edge = left_edge[:, :, :, :, -blend_extent:].to(device=tile.device, dtype=tile.dtype)
    edge.mul_(1.0 - weights)
    tile[:, :, :, :, :blend_extent].mul_(weights).add_(edge)


class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolusion.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (self.padding[2], self.padding[2], self.padding[1],
                         self.padding[1], 2 * self.padding[0], 0)
        self.padding = (0, 0, 0)

    def forward(self, x, cache_x=None, padded=False, residual=None):
        if padded:  # fast path: x is already the padded channels-last input; residual is added to the output
            if _kernels.conv_eligible(x, self):
                return _kernels.conv(x, self.weight, self.bias, residual)
            x = F.conv3d(x, _kernels.channels_last_weight(self.weight), self.bias, self.stride)  # cuDNN, without PyTorch's slower weight layout copy
            return x if residual is None else x.add_(residual)
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
            cache_x = None
        x = F.pad(x, padding)
        try:
            out = super().forward(x)
            return out
        except RuntimeError as e:
            if "miopenStatus" in str(e):
                print("⚠️ MIOpen fallback: AMD gets upset when trying to work with large areas, and so CPU will be "
                      "used for this decoding (which is very slow). Consider using tiled VAE Decoding.")
                x_cpu = x.float().cpu()
                weight_cpu = self.weight.float().cpu()
                bias_cpu = self.bias.float().cpu() if self.bias is not None else None
                print(f"[Fallback] x shape: {x_cpu.shape}, weight shape: {weight_cpu.shape}")
                out = F.conv3d(x_cpu, weight_cpu, bias_cpu,
                               self.stride, (0, 0, 0),  # avoid double padding here
                               self.dilation, self.groups)
                out = out.to(x.device)
                if x.dtype in (torch.float16, torch.bfloat16):
                    out = out.half()
                if x.dtype != out.dtype:
                    out = out.to(x.dtype)
                return out
            raise


class RMS_norm(nn.Module):

    def __init__(self, dim, channel_first=True, images=True, bias=False):
        super().__init__()
        broadcastable_dims = (1, 1, 1) if not images else (1, 1)
        shape = (dim, *broadcastable_dims) if channel_first else (dim,)

        self.channel_first = channel_first
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.

    def forward(self, x, pad=None, cache=None, keep=0):
        if pad is not None:  # fast path: silu(norm(x)) as the padded input of the next causal convolution
            return _kernels.causal_pad(x, pad, cache, self, keep)
        dtype = x.dtype
        x = F.normalize(
            x, dim=(1 if self.channel_first else
                    -1)) * self.scale * self.gamma + self.bias
        x = x.to(dtype)
        return x 

class Upsample(nn.Upsample):

    def forward(self, x):
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().forward(x.float()).type_as(x)


class Resample(nn.Module):

    def __init__(self, dim, mode):
        assert mode in ('none', 'upsample2d', 'upsample3d', 'downsample2d',
                        'downsample3d')
        super().__init__()
        self.dim = dim
        self.mode = mode

        # layers
        if mode == 'upsample2d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
        elif mode == 'upsample3d':
            self.resample = nn.Sequential(
                Upsample(scale_factor=(2., 2.), mode='nearest-exact'),
                nn.Conv2d(dim, dim // 2, 3, padding=1))
            self.time_conv = CausalConv3d(
                dim, dim * 2, (3, 1, 1), padding=(1, 0, 0))

        elif mode == 'downsample2d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
        elif mode == 'downsample3d':
            self.resample = nn.Sequential(
                nn.ZeroPad2d((0, 1, 0, 1)),
                nn.Conv2d(dim, dim, 3, stride=(2, 2)))
            self.time_conv = CausalConv3d(
                dim, dim, (3, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))

        else:
            self.resample = nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0], fast=False):
        if fast:
            return _resample_fast(self, x, feat_cache, feat_idx)
        b, c, t, h, w = x.size()
        if self.mode == 'upsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = 'Rep'
                    feat_idx[0] += 1
                else:
                    clone = True
                    cache_x = x[:, :, -CACHE_T:, :, :]#.clone()
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] != 'Rep':
                        # cache last frame of last two chunk
                        clone = False
                        cache_x = torch.cat([
                            feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                                cache_x.device), cache_x
                        ],
                                            dim=2)
                    if cache_x.shape[2] < 2 and feat_cache[
                            idx] is not None and feat_cache[idx] == 'Rep':
                        clone = False
                        cache_x = torch.cat([
                            torch.zeros_like(cache_x).to(cache_x.device),
                            cache_x
                        ],
                                            dim=2)
                    if clone:
                        cache_x = cache_x.clone()
                    if feat_cache[idx] == 'Rep':
                        x = self.time_conv(x)
                    else:
                        x = self.time_conv(x, feat_cache[idx])
                    feat_cache[idx] = cache_x
                    feat_idx[0] += 1

                    x = x.reshape(b, 2, c, t, h, w)
                    x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]),
                                    3)
                    x = x.reshape(b, c, t * 2, h, w)
        t = x.shape[2]
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.resample(x)
        x = rearrange(x, '(b t) c h w -> b c t h w', t=t)

        if self.mode == 'downsample3d':
            if feat_cache is not None:
                idx = feat_idx[0]
                if feat_cache[idx] is None:
                    feat_cache[idx] = x #.to("cpu") #x.clone() yyyy
                    feat_idx[0] += 1
                else:

                    cache_x = x[:, :, -1:, :, :].clone()
                    # if cache_x.shape[2] < 2 and feat_cache[idx] is not None and feat_cache[idx]!='Rep':
                    #     # cache last frame of last two chunk
                    #     cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)

                    x = self.time_conv(
                        torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2))
                    feat_cache[idx] = cache_x#.to("cpu") #yyyyy
                    feat_idx[0] += 1
        return x

    def init_weight(self, conv):
        conv_weight = conv.weight
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        one_matrix = torch.eye(c1, c2)
        init_matrix = one_matrix
        nn.init.zeros_(conv_weight)
        #conv_weight.data[:,:,-1,1,1] = init_matrix * 0.5
        conv_weight.data[:, :, 1, 0, 0] = init_matrix  #* 0.5
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def init_weight2(self, conv):
        conv_weight = conv.weight.data
        nn.init.zeros_(conv_weight)
        c1, c2, t, h, w = conv_weight.size()
        init_matrix = torch.eye(c1 // 2, c2)
        #init_matrix = repeat(init_matrix, 'o ... -> (o 2) ...').permute(1,0,2).contiguous().reshape(c1,c2)
        conv_weight[:c1 // 2, :, -1, 0, 0] = init_matrix
        conv_weight[c1 // 2:, :, -1, 0, 0] = init_matrix
        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)


class ResidualBlock(nn.Module):

    def __init__(self, in_dim, out_dim, dropout=0.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # layers
        self.residual = nn.Sequential(
            RMS_norm(in_dim, images=False), nn.SiLU(),
            CausalConv3d(in_dim, out_dim, 3, padding=1),
            RMS_norm(out_dim, images=False), nn.SiLU(), nn.Dropout(dropout),
            CausalConv3d(out_dim, out_dim, 3, padding=1))
        self.shortcut = CausalConv3d(in_dim, out_dim, 1) \
            if in_dim != out_dim else nn.Identity()

    def forward(self, x, feat_cache=None, feat_idx=[0], fast=False):
        if fast:
            return _residual_fast(self, x, feat_cache, feat_idx)
        h = self.shortcut(x)
        dtype = x.dtype
        for layer in self.residual:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx]).to(dtype)
                feat_cache[idx] = cache_x#.to("cpu")
                feat_idx[0] += 1
            else:
                x = layer(x).to(dtype)
        return x + h


class AttentionBlock(nn.Module):
    """
    Causal self-attention with a single head.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # layers
        self.norm = RMS_norm(dim)
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        # zero out the last layer params
        nn.init.zeros_(self.proj.weight)

    def forward(self, x, fast=False):
        if fast:
            return _attention_fast(self, x)
        identity = x
        b, c, t, h, w = x.size()
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.norm(x)
        # compute query, key, value
        q, k, v = self.to_qkv(x).reshape(b * t, 1, c * 3,
                                         -1).permute(0, 1, 3,
                                                     2).contiguous().chunk(
                                                         3, dim=-1)

        # apply attention (by chunks of queries where SDPA has no memory-efficient kernel for the VAE's dtype)
        x = vae_ops.attention(q, k, v)
        x = x.squeeze(1).permute(0, 2, 1).reshape(b * t, c, h, w)

        # output
        x = self.proj(x)
        x = rearrange(x, '(b t) c h w-> b c t h w', t=t)
        return x + identity


class Encoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample

        # dimensions
        dims = [dim * u for u in [1] + dim_mult]
        scale = 1.0

        # init block
        self.conv1 = CausalConv3d(3, dims[0], 3, padding=1)

        # downsample blocks
        downsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            for _ in range(num_res_blocks):
                downsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    downsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # downsample block
            if i != len(dim_mult) - 1:
                mode = 'downsample3d' if temperal_downsample[
                    i] else 'downsample2d'
                downsamples.append(Resample(out_dim, mode=mode))
                scale /= 2.0
        self.downsamples = nn.Sequential(*downsamples)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(out_dim, out_dim, dropout), AttentionBlock(out_dim),
            ResidualBlock(out_dim, out_dim, dropout))

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, z_dim, 3, padding=1))

    def _stages(self):
        return [self.conv1, *self.downsamples, *self.middle, self.head]

    def _forward_fast(self, x, feat_cache, feat_idx):
        x = self.conv1(_causal_input(x, self.conv1, feat_cache, feat_idx), padded=True)
        for layer in self.downsamples:
            x = layer(x, feat_cache, feat_idx, fast=True)
        for layer in self.middle:
            x = layer(x, feat_cache, feat_idx, fast=True) if isinstance(layer, ResidualBlock) else layer(x, fast=True)
        x = self.head[2](_causal_input(x, self.head[2], feat_cache, feat_idx, self.head[0]), padded=True)
        return x.contiguous()

    def forward(self, x, feat_cache=None, feat_idx=[0], part=None):
        if part is not None:  # exact tiled encoding: some stages, on the whole frame or a region
            return _run_part(self, x, part, feat_cache, feat_idx)
        if _use_fast(x):
            return self._forward_fast(x, feat_cache, feat_idx)
        dtype = x.dtype
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                                    dim=2)
            x = self.conv1(x, feat_cache[idx]).to(dtype)
            feat_cache[idx] = cache_x
            del cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)


        ## downsamples
        for layer in self.downsamples:
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)


        ## middle
        for layer in self.middle:
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)


        ## head
        for layer in self.head:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                del cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)


        return x


class Decoder3d(nn.Module):

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_upsample=[False, True, True],
                 dropout=0.0,
                 upsampler_factor = 1,
                 ):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_upsample = temperal_upsample

        # dimensions
        dims = [dim * u for u in [dim_mult[-1]] + dim_mult[::-1]]
        scale = 1.0 / 2**(len(dim_mult) - 2)

        # init block
        self.conv1 = CausalConv3d(z_dim, dims[0], 3, padding=1)

        # middle blocks
        self.middle = nn.Sequential(
            ResidualBlock(dims[0], dims[0], dropout), AttentionBlock(dims[0]),
            ResidualBlock(dims[0], dims[0], dropout))

        # upsample blocks
        upsamples = []
        for i, (in_dim, out_dim) in enumerate(zip(dims[:-1], dims[1:])):
            # residual (+attention) blocks
            if i == 1 or i == 2 or i == 3:
                in_dim = in_dim // 2
            for _ in range(num_res_blocks + 1):
                upsamples.append(ResidualBlock(in_dim, out_dim, dropout))
                if scale in attn_scales:
                    upsamples.append(AttentionBlock(out_dim))
                in_dim = out_dim

            # upsample block
            if i != len(dim_mult) - 1:
                mode = 'upsample3d' if temperal_upsample[i] else 'upsample2d'
                upsamples.append(Resample(out_dim, mode=mode))
                scale *= 2.0
        self.upsamples = nn.Sequential(*upsamples)

        # output blocks
        self.head = nn.Sequential(
            RMS_norm(out_dim, images=False), nn.SiLU(),
            CausalConv3d(out_dim, 3 * int(upsampler_factor*upsampler_factor), 3, padding=1))

    def _stages(self):
        return [self.conv1, *self.middle, *self.upsamples, self.head]

    def _forward_fast(self, x, feat_cache, feat_idx):
        x = self.conv1(_causal_input(x, self.conv1, feat_cache, feat_idx), padded=True)
        for layer in self.middle:
            check_abort()
            x = layer(x, feat_cache, feat_idx, fast=True) if isinstance(layer, ResidualBlock) else layer(x, fast=True)
        for layer in self.upsamples:
            check_abort()
            x = layer(x, feat_cache, feat_idx, fast=True)
        check_abort()
        x = self.head[2](_causal_input(x, self.head[2], feat_cache, feat_idx, self.head[0]), padded=True)
        return x.contiguous()

    def forward(self, x, feat_cache=None, feat_idx=[0], part=None):
        if part is not None:  # exact tiled decoding: some stages, on the whole frame or a region
            return _run_part(self, x, part, feat_cache, feat_idx)
        if _use_fast(x):
            return self._forward_fast(x, feat_cache, feat_idx)
        ## conv1
        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([
                    feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                        cache_x.device), cache_x
                ],
                                    dim=2)
            x = self.conv1(x, feat_cache[idx])
            feat_cache[idx] = cache_x#.to("cpu")
            del cache_x
            feat_idx[0] += 1
        else:
            x = self.conv1(x)
        cache_x = None

        ## middle
        for layer in self.middle:
            check_abort()
            if isinstance(layer, ResidualBlock) and feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## upsamples
        for layer in self.upsamples:
            check_abort()
            if feat_cache is not None:
                x = layer(x, feat_cache, feat_idx)
            else:
                x = layer(x)

        ## head
        for layer in self.head:
            check_abort()
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :] .clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([
                        feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(
                            cache_x.device), cache_x
                    ],
                                        dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x#.to("cpu")
                del cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count


class WanVAE_(nn.Module):

    _offload_hooks = ['encode', 'decode']

    def __init__(self,
                 dim=128,
                 z_dim=4,
                 dim_mult=[1, 2, 4, 4],
                 num_res_blocks=2,
                 attn_scales=[],
                 temperal_downsample=[True, True, False],
                 upsampler_factor = 1,
                 dropout=0.0):
        super().__init__()
        self.dim = dim
        self.z_dim = z_dim
        self.dim_mult = dim_mult
        self.num_res_blocks = num_res_blocks
        self.attn_scales = attn_scales
        self.temperal_downsample = temperal_downsample
        self.temperal_upsample = temperal_downsample[::-1]
        self.upsampler_factor = upsampler_factor

        # modules
        self.encoder = Encoder3d(dim, z_dim * 2, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_downsample, dropout)
        self.conv1 = CausalConv3d(z_dim * 2, z_dim * 2, 1)
        self.conv2 = CausalConv3d(z_dim, z_dim, 1)
        self.decoder = Decoder3d(dim, z_dim, dim_mult, num_res_blocks,
                                 attn_scales, self.temperal_upsample, dropout, upsampler_factor)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var

    def encode(self, x, scale = None, any_end_frame = False):
        self.clear_cache()
        ## cache
        t = x.shape[2]
        if any_end_frame:
            iter_ = 2 + (t - 2) // 4
        else:
            iter_ = 1 + (t - 1) // 4
        ## 对encode输入的x，按时间拆分为1、4、4、4....
        out_list = []
        for i in range(iter_):
            self._enc_conv_idx = [0]
            if i == 0:
                out_list.append(self.encoder(
                    x[:, :, :1, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx))
            elif any_end_frame and i== iter_ -1:
                out_list.append(self.encoder(
                    x[:, :, -1:, :, :],
                    feat_cache= None,
                    feat_idx=self._enc_conv_idx))
            else:
                out_list.append(self.encoder(
                    x[:, :, 1 + 4 * (i - 1):1 + 4 * i, :, :],
                    feat_cache=self._enc_feat_map,
                    feat_idx=self._enc_conv_idx))

        self.clear_cache()
        out = torch.cat(out_list, 2)
        out_list = None

        mu, log_var = self.conv1(out).chunk(2, dim=1)
        if scale != None:
            if isinstance(scale[0], torch.Tensor):
                mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                    1, self.z_dim, 1, 1, 1)
            else:
                mu = (mu - scale[0]) * scale[1]
        return mu


    def decode(self, z, scale=None, any_end_frame = False):
        self.clear_cache()
        # z: [b,c,t,h,w]
        if scale != None:
            if isinstance(scale[0], torch.Tensor):
                z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(1, self.z_dim, 1, 1, 1)
            else:
                z = z / scale[1] + scale[0]
        iter_ = z.shape[2]
        x = self.conv2(z)
        out_list = []
        for i in range(iter_):
            self._conv_idx = [0]
            if i == 0:
                out_list.append(self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx))
            elif any_end_frame and i==iter_-1:
                out_list.append(self.decoder(
                    x[:, :, -1:, :, :],
                    feat_cache=None ,
                    feat_idx=self._conv_idx))
            else:
                out_list.append(self.decoder(
                    x[:, :, i:i + 1, :, :],
                    feat_cache=self._feat_map,
                    feat_idx=self._conv_idx))
        self.clear_cache()
        out = torch.cat(out_list, 2)

        if self.upsampler_factor > 1:
            out = F.pixel_shuffle(out.movedim(2, 1), upscale_factor=self.upsampler_factor).movedim(1, 2)  # pixel shuffle needs [..., C, H, W] format

        return out
    
    def blend_v(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (1 - y / blend_extent) + b[:, :, :, y, :] * (y / blend_extent)
        return b

    def blend_h(self, a: torch.Tensor, b: torch.Tensor, blend_extent: int) -> torch.Tensor:
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (1 - x / blend_extent) + b[:, :, :, :, x] * (x / blend_extent)
        return b
    
    def spatial_tiled_decode(self, z, scale, tile_size, any_end_frame= False):
        tile_sample_min_size = tile_size
        tile_latent_min_size = int(tile_sample_min_size / 8)
        tile_overlap_factor = 0.25

        # z: [b,c,t,h,w]

        if isinstance(scale[0], torch.Tensor):
            z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view( 1, self.z_dim, 1, 1, 1)
        else:
            z = z / scale[1] + scale[0]
        if _use_fast(z):  # tiles with halos: the untiled result
            f = self.upsampler_factor
            out = torch.empty((z.shape[0], 3, (z.shape[2] - 2) * 4 + 2 if any_end_frame else (z.shape[2] - 1) * 4 + 1, z.shape[-2] * 8 * f, z.shape[-1] * 8 * f), dtype=z.dtype, device=z.device)

            def sink(i, offset, region, y):
                y = F.pixel_shuffle(y.movedim(2, 1), upscale_factor=f).movedim(1, 2) if f > 1 else y
                out[:, :, offset:offset + y.shape[2], region[0][0] * f:region[0][1] * f, region[1][0] * f:region[1][1] * f] = y
            _exact_decode(self, z, _exact_segments(self.decoder, tuple(z.shape[-2:]), tile_size, True), any_end_frame, sink)
            return out

        overlap_size = int(tile_latent_min_size * (1 - tile_overlap_factor)) #8 0.75
        tile_sample_min_size *=  self.upsampler_factor
        blend_extent = int(tile_sample_min_size * tile_overlap_factor) #256 0.25
        row_limit = tile_sample_min_size - blend_extent

        # Split z into overlapping tiles and decode them separately.
        # The tiles have an overlap to avoid seams between tiles.
        rows = []
        for i in range(0, z.shape[-2], overlap_size):
            row = []
            for j in range(0, z.shape[-1], overlap_size):
                tile = z[:, :, :, i: i + tile_latent_min_size, j: j + tile_latent_min_size]
                decoded = self.decode(tile, any_end_frame= any_end_frame)
                row.append(decoded)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))

        return torch.cat(result_rows, dim=-2)

    def decode_tile_chunks(self, z, any_end_frame=False):
        self.clear_cache()
        x = self.conv2(z)
        frame_start = 0
        try:
            for i in range(x.shape[2]):
                self._conv_idx = [0]
                if i == 0:
                    tile = self.decoder(x[:, :, i:i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
                elif any_end_frame and i == x.shape[2] - 1:
                    tile = self.decoder(x[:, :, -1:, :, :], feat_cache=None, feat_idx=self._conv_idx)
                else:
                    tile = self.decoder(x[:, :, i:i + 1, :, :], feat_cache=self._feat_map, feat_idx=self._conv_idx)
                if self.upsampler_factor > 1:
                    tile = F.pixel_shuffle(tile.movedim(2, 1), upscale_factor=self.upsampler_factor).movedim(1, 2)
                yield frame_start, tile
                frame_start += int(tile.shape[2])
                del tile
        finally:
            del x
            self.clear_cache()

    def decode_to_cpu_uint8(self, z, scale=None, tile_size=0, target_frames=None, target_height=None, target_width=None, any_end_frame=False, device=None, frame_start=0):
        device = torch.device(device) if device is not None else (scale[0].device if scale is not None and isinstance(scale[0], torch.Tensor) else z.device)
        dtype = getattr(self, "_model_dtype", z.dtype)
        tile_size = int(tile_size or 0)
        latent_source = z.detach()
        if tile_size > 0 and latent_source.device.type != "cpu":
            latent_source = latent_source.to("cpu")
        decoded_frame_count = 0 if latent_source.shape[2] <= 0 else (int(latent_source.shape[2]) - 1) * 4 + 1
        frame_start = min(max(0, int(frame_start or 0)), decoded_frame_count)
        target_frames = decoded_frame_count - frame_start if target_frames is None else min(int(target_frames), decoded_frame_count - frame_start)
        target_end = frame_start + target_frames
        needed_latents = 0 if target_frames <= 0 else min(int(latent_source.shape[2]), (max(target_end, 1) - 1 + 3) // 4 + 1)
        latent_source = latent_source[:, :, :needed_latents]
        full_height = latent_source.shape[-2] * 8 * self.upsampler_factor
        full_width = latent_source.shape[-1] * 8 * self.upsampler_factor
        target_height = full_height if target_height is None else min(int(target_height), full_height)
        target_width = full_width if target_width is None else min(int(target_width), full_width)
        if target_frames <= 0:
            return torch.empty((latent_source.shape[0], 3, 0, target_height, target_width), dtype=torch.uint8, device="cpu")
        spatial_tiles = 1
        segments = _exact_segments(self.decoder, tuple(latent_source.shape[-2:]), tile_size, True) if tile_size > 0 and _use_fast(torch.empty(0, dtype=dtype, device=device)) else None
        if segments is not None:  # tiles with halos (the untiled result): whole-frame calls, then the tiles
            spatial_tiles = _exact_calls(segments, 1)
        elif tile_size > 0:
            stride = max(1, int(max(1, tile_size // 8) * 0.75))
            row_limit = max(1, tile_size * self.upsampler_factor - int(tile_size * self.upsampler_factor * 0.25))
            rows = min(len(range(0, latent_source.shape[-2], stride)), (target_height + row_limit - 1) // row_limit)
            cols = min(len(range(0, latent_source.shape[-1], stride)), (target_width + row_limit - 1) // row_limit)
            spatial_tiles = rows * cols
        with PhaseProgress(spatial_tiles * needed_latents) as progress, progress.track(self.decoder):
            try:
                if scale is not None and isinstance(scale[0], torch.Tensor):
                    scale = [u.to(device=device) for u in scale]
                if segments is not None:
                    z = latent_source.to(device=device, dtype=dtype)
                    if scale is not None:
                        z = z / scale[1].view(1, self.z_dim, 1, 1, 1) + scale[0].view(1, self.z_dim, 1, 1, 1) if isinstance(scale[0], torch.Tensor) else z / scale[1] + scale[0]
                    decoded = torch.empty((latent_source.shape[0], 3, target_frames, target_height, target_width), dtype=torch.uint8, device="cpu")
                    f = self.upsampler_factor

                    def sink(i, offset, region, y):
                        y = F.pixel_shuffle(y.movedim(2, 1), upscale_factor=f).movedim(1, 2) if f > 1 else y
                        (r0, r1), (c0, c1) = ((lo * f, min(hi * f, limit)) for (lo, hi), limit in zip(region, (target_height, target_width)))
                        a, b = max(offset, frame_start), min(offset + y.shape[2], target_end)
                        if a < b and r0 < r1 and c0 < c1:
                            decoded[:, :, a - frame_start:b - frame_start, r0:r1, c0:c1] = _vae_float_to_cpu_uint8(y[:, :, a - offset:b - offset, :r1 - r0, :c1 - c0])
                    _exact_decode(self, z, segments, any_end_frame, sink)
                    return decoded
                if tile_size <= 0:
                    z = latent_source.to(device=device, dtype=dtype)
                    frames = self.decode(z, scale, any_end_frame=any_end_frame)[:, :, frame_start:target_end, :target_height, :target_width]
                    decoded = _vae_float_to_cpu_uint8(frames)
                    del z, frames
                    return decoded

                tile_sample_min_size = tile_size
                tile_latent_min_size = max(1, int(tile_sample_min_size / 8))
                tile_overlap_factor = 0.25
                overlap_size = max(1, int(tile_latent_min_size * (1 - tile_overlap_factor)))
                tile_sample_min_size *= self.upsampler_factor
                blend_extent = int(tile_sample_min_size * tile_overlap_factor)
                row_limit = max(1, tile_sample_min_size - blend_extent)
                decoded = torch.empty((latent_source.shape[0], 3, target_frames, target_height, target_width), dtype=torch.uint8, device="cpu")
                previous_row_edges = []
                row_index = 0
                for latent_y in range(0, latent_source.shape[-2], overlap_size):
                    current_row_edges = []
                    left_edge = None
                    col_index = 0
                    write_y0 = row_index * row_limit
                    write_y1 = min(write_y0 + row_limit, target_height)
                    has_next_row = write_y1 < target_height
                    if write_y1 <= write_y0:
                        break
                    for latent_x in range(0, latent_source.shape[-1], overlap_size):
                        write_x0 = col_index * row_limit
                        write_x1 = min(write_x0 + row_limit, target_width)
                        has_next_col = write_x1 < target_width
                        if write_x1 <= write_x0:
                            break
                        tile_latents = latent_source[:, :, :, latent_y:latent_y + tile_latent_min_size, latent_x:latent_x + tile_latent_min_size].to(device=device, dtype=dtype)
                        if scale is not None:
                            if isinstance(scale[0], torch.Tensor):
                                tile_latents.div_(scale[1].view(1, self.z_dim, 1, 1, 1)).add_(scale[0].view(1, self.z_dim, 1, 1, 1))
                            else:
                                tile_latents.div_(scale[1]).add_(scale[0])
                        bottom_edge = None
                        right_edge = None
                        previous_edge = previous_row_edges[col_index] if row_index > 0 and col_index < len(previous_row_edges) else None
                        for tile_frame_start, tile in self.decode_tile_chunks(tile_latents, any_end_frame=any_end_frame):
                            if tile_frame_start >= target_end:
                                break
                            tile_frame_end = tile_frame_start + int(tile.shape[2])
                            copy_start, copy_end = max(tile_frame_start, frame_start), min(tile_frame_end, target_end)
                            if copy_start >= copy_end:
                                del tile
                                continue
                            out_start, out_end = copy_start - frame_start, copy_end - frame_start
                            tile = tile[:, :, copy_start - tile_frame_start:copy_end - tile_frame_start]
                            if previous_edge is not None:
                                _blend_v_edge_(previous_edge[:, :, out_start:out_end], tile, blend_extent)
                            if left_edge is not None:
                                _blend_h_edge_(left_edge[:, :, out_start:out_end], tile, blend_extent)
                            if has_next_row:
                                edge = tile[:, :, :, -min(blend_extent, tile.shape[-2]):, :].detach().cpu()
                                if bottom_edge is None:
                                    bottom_edge = torch.empty((edge.shape[0], edge.shape[1], target_frames, edge.shape[3], edge.shape[4]), dtype=edge.dtype, device="cpu")
                                bottom_edge[:, :, out_start:out_end].copy_(edge)
                                del edge
                            if has_next_col:
                                edge = tile[:, :, :, :, -min(blend_extent, tile.shape[-1]):].detach().cpu()
                                if right_edge is None:
                                    right_edge = torch.empty((edge.shape[0], edge.shape[1], target_frames, edge.shape[3], edge.shape[4]), dtype=edge.dtype, device="cpu")
                                right_edge[:, :, out_start:out_end].copy_(edge)
                                del edge
                            tile = tile[:, :, :, :write_y1 - write_y0, :write_x1 - write_x0]
                            decoded[:, :, out_start:out_end, write_y0:write_y0 + tile.shape[-2], write_x0:write_x0 + tile.shape[-1]].copy_(_vae_float_to_cpu_uint8(tile))
                            del tile
                        current_row_edges.append(bottom_edge)
                        left_edge = right_edge
                        del tile_latents, previous_edge
                        col_index += 1
                    left_edge = None
                    previous_row_edges = current_row_edges
                    row_index += 1
                return decoded

            finally:
                self.clear_cache()

    def spatial_tiled_encode(self, x, scale, tile_size, any_end_frame = False) :
        if _use_fast(x):  # tiles with halos: the untiled result
            return _exact_encode(self, x, scale, tile_size, any_end_frame)
        tile_sample_min_size = tile_size
        tile_latent_min_size = int(tile_sample_min_size / 8)
        tile_overlap_factor = 0.25

        overlap_size = int(tile_sample_min_size * (1 - tile_overlap_factor))
        blend_extent = int(tile_latent_min_size * tile_overlap_factor)
        row_limit = tile_latent_min_size - blend_extent

        # Split video into tiles and encode them separately.
        rows = []
        for i in range(0, x.shape[-2], overlap_size):
            row = []
            for j in range(0, x.shape[-1], overlap_size):
                tile = x[:, :, :, i: i + tile_sample_min_size, j: j + tile_sample_min_size]
                tile = self.encode(tile, any_end_frame= any_end_frame)
                row.append(tile)
            rows.append(row)
        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                # blend the above tile and the left tile
                # to the current tile and add the current tile to the result row
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_extent)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_extent)
                result_row.append(tile[:, :, :, :row_limit, :row_limit])
            result_rows.append(torch.cat(result_row, dim=-1))

        mu = torch.cat(result_rows, dim=-2)

        if isinstance(scale[0], torch.Tensor):
            mu = (mu - scale[0].view(1, self.z_dim, 1, 1, 1)) * scale[1].view(
                1, self.z_dim, 1, 1, 1)
        else:
            mu = (mu - scale[0]) * scale[1]

        return mu


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def sample(self, imgs, deterministic=False):
        mu, log_var = self.encode(imgs)
        if deterministic:
            return mu
        std = torch.exp(0.5 * log_var.clamp(-30.0, 20.0))
        return mu + std * torch.randn_like(std)

    def clear_cache(self):
        self._conv_num = count_conv3d(self.decoder)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        #cache encode
        self._enc_conv_num = count_conv3d(self.encoder)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


def _video_vae(pretrained_path=None, z_dim=None, device='cpu', preprocess_sd=None, **kwargs):
    """
    Autoencoder3d adapted from Stable Diffusion 1.x, 2.x and XL.
    """
    # params
    cfg = dict(
        dim=96,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0)
    cfg.update(**kwargs)

    # init model
    with torch.device('meta'):
        model = WanVAE_(**cfg)

    from mmgp import offload
    # load checkpoint
    logging.info(f'loading {pretrained_path}')
    # model.load_state_dict(
    #     torch.load(pretrained_path, map_location=device), assign=True)
    # offload.load_model_data(model, pretrained_path.replace(".pth", "_bf16.safetensors"), writable_tensors= False)    
    offload.load_model_data(model, pretrained_path, writable_tensors=False, default_dtype=None, preprocess_sd=preprocess_sd)
    return model


class WanVAE:

    def __init__(self,
                 z_dim=16,
                 vae_pth='cache/vae_step_411000.pth',
                 dtype=torch.float,
                 upsampler_factor = 1,
                 device="cuda",
                 preprocess_sd=None):
        self.dtype = dtype
        self.device = device
        self.z_dim = z_dim

        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=dtype, device=device)
        self.std = torch.tensor(std, dtype=dtype, device=device)
        self.scale = [self.mean, 1.0 / self.std]

        # init model
        self.model = _video_vae(
            pretrained_path=vae_pth,
            upsampler_factor = upsampler_factor,
            z_dim=z_dim,
            preprocess_sd=preprocess_sd,
        ).to(dtype).eval() #.requires_grad_(False).to(device)
        self.model._model_dtype = dtype

    @staticmethod
    def get_VAE_tile_size(vae_config, device_mem_capacity, mixed_precision, output_height=None, output_width=None):
        if mmgp_vram_allocator():  # no allocator fragmentation to guard against: Default and Low VRAM levels
            return 256 if vae_config == 3 or device_mem_capacity < 8000 else 512
        # VAE Tiling
        if vae_config == 0:
            if mixed_precision:
                device_mem_capacity = device_mem_capacity / 2
            if device_mem_capacity >= 24000:
                if output_height is not None and output_width is not None and int(output_height) * int(output_width) > 1920 * 1088:
                    use_vae_config = 2
                else:
                    use_vae_config = 1
            elif device_mem_capacity >= 16000:
                use_vae_config = 3
            elif device_mem_capacity >= 8000:
                use_vae_config = 4
            else:          
                use_vae_config = 5
        else:
            # Keep WGP's historical public presets; Wan inserts one internal tiers between presets 1 and 3.
            use_vae_config = vae_config + 2

        if use_vae_config == 1:
            VAE_tile_size = 0  
        elif use_vae_config == 2:
            VAE_tile_size = 1024 
        elif use_vae_config == 3:
            VAE_tile_size = 512 
        elif use_vae_config == 4:
            VAE_tile_size = 256  
        else: 
            VAE_tile_size = 128  

        return  VAE_tile_size

    def encode(self, videos, tile_size = 256, any_end_frame = False):
        """
        videos: A list of videos each with shape [C, T, H, W].
        """
        set_phase_status("VAE Encoding")
        tiles = 0
        for video in videos:
            temporal_tiles = 2 + (video.shape[1] - 2) // 4 if any_end_frame else 1 + (video.shape[1] - 1) // 4
            if tile_size > 0 and _use_fast(torch.empty(0, dtype=self.dtype, device=video.device)):
                temporal_tiles = _exact_calls(_exact_segments(self.model.encoder, tuple(video.shape[-2:]), tile_size, False), temporal_tiles)
            elif tile_size > 0:
                stride = int(tile_size * 0.75)
                temporal_tiles *= len(range(0, video.shape[-2], stride)) * len(range(0, video.shape[-1], stride))
            tiles += temporal_tiles
        with vae_encoding_progress(tiles, self.model.encoder, cleanup=self.model.clear_cache, enabled=any(video.shape[1] > 1 for video in videos)):
            scale = [u.to(device = self.device) for u in self.scale]
            # uint8 control videos (model_def "uint8_guides") get WanGP's float values, converted by chunks of frames
            if tile_size > 0:
                return [ self.model.spatial_tiled_encode(guide_to_dtype(u, self.dtype).unsqueeze(0), scale, tile_size, any_end_frame=any_end_frame).float().squeeze(0) for u in videos ]
            else:
                return [ self.model.encode(guide_to_dtype(u, self.dtype).unsqueeze(0), scale, any_end_frame=any_end_frame).float().squeeze(0) for u in videos ]


    def decode(self, zs, tile_size, any_end_frame = False):
        scale = [u.to(device = self.device) for u in self.scale]  
        if tile_size > 0:
            return [ self.model.spatial_tiled_decode(u.to(self.dtype).unsqueeze(0), scale, tile_size, any_end_frame=any_end_frame).clamp_(-1, 1).float().squeeze(0) for u in zs ]
        else:
            return [ self.model.decode(u.to(self.dtype).unsqueeze(0), scale, any_end_frame=any_end_frame).clamp_(-1, 1).float().squeeze(0) for u in zs ]

    def decode_to_cpu_uint8(self, zs, tile_size, target_frames=None, target_height=None, target_width=None, any_end_frame=False, frame_start=0):
        scale = [u.to(device=self.device) for u in self.scale]
        tile_size = int(tile_size or 0)
        return [
            self.model.decode_to_cpu_uint8(u.detach().to(device="cpu" if tile_size > 0 else u.device, dtype=self.dtype).unsqueeze(0), scale, tile_size, target_frames=target_frames, target_height=target_height, target_width=target_width, any_end_frame=any_end_frame, device=self.device, frame_start=frame_start).squeeze(0)
            for u in zs
        ]
