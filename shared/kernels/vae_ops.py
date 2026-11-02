# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Shared dispatch for video/audio activation kernels.

Streaming schedules, temporal history, and neighborhood topology stay with the
model. This module owns kernel eligibility and fallbacks, never those caches.
"""
import torch
import torch.nn.functional as F
from . import kernel_policy, ops, native_ops, video_kitchen as _kitchen
from .dispatch import choose, cuda_inference, take

try:
    from . import video_triton as _video
except (ImportError, OSError, RuntimeError):
    _video = None
try:
    from . import conv3d_triton as _conv, stream_triton as _stream
except (ImportError, OSError, RuntimeError):
    _conv = _stream = None
try:
    from .audio_triton import snake_beta_aa as _snake, snake_ as _snake_plain
except (ImportError, OSError, RuntimeError):
    _snake = _snake_plain = None
try:
    from . import group_norm_silu as _norm_silu
except (ImportError, OSError, RuntimeError):
    _norm_silu = None
try:
    from . import sparse_conv_triton as _sparse
except (ImportError, OSError, RuntimeError):
    _sparse = None

_AUDIO_DTYPES = (torch.bfloat16, torch.float16, torch.float32)

_DTYPES = (torch.bfloat16, torch.float16)
_STREAM_CONV_MIN_PIXELS = 8192  # smaller calls keep cuDNN's lower launch overhead


def prepare_rotary(cos, sin):
    return _kitchen.prepare_rotary(cos, sin)


def conv3d(x_list, conv, residual=None):
    x = take(x_list)
    if choose("Video Conv3D", kitchen=_kitchen.can_conv3d(x, conv)):
        return _kitchen.conv3d(x, conv, residual)
    x = F.conv3d(x, conv.weight, conv.bias, stride=conv.stride, padding=0, dilation=conv.dilation)
    return x if residual is None else x.add_(residual)


def frame_group_norm(x_list, norm, silu_pad=None):
    x = take(x_list)
    triton_pad = False
    if silu_pad is not None:
        left, right, top, bottom, front = silu_pad
        triton_pad = (_video is not None and cuda_inference(x, dtypes=_DTYPES)
                      and left == right == top == bottom and left < min(x.shape[-2:]))
    # Kitchen also prepares the channels-last layout for the next convolution.
    # Choosing native norm on its isolated time regresses the complete encoder.
    kitchen = silu_pad is not None and _kitchen.available(x, norm, silu_pad)
    family = choose("Frame GroupNorm + SiLU + Pad", kitchen=kitchen, owned_triton=triton_pad and not kitchen) if silu_pad is not None else None
    if family == "kitchen":
        return _kitchen.group_norm_silu_pad3d(x, norm, silu_pad)
    batch, channels, frames, height, width = x.shape
    x = x.permute(0, 2, 1, 3, 4).contiguous().view(batch * frames, channels, 1, height, width)
    x = F.group_norm(x, norm.num_groups, norm.weight, norm.bias, norm.eps)
    if silu_pad is not None:
        left, right, top, bottom, front = silu_pad
        if family is not None:
            return _video.silu_pad_frames(x, batch, frames, left, front)
    x = x.view(batch, frames, channels, height, width).permute(0, 2, 1, 3, 4).contiguous()
    if silu_pad is not None:
        x = F.silu(x)
        x = F.pad(x, (left, right, top, bottom, 0, 0), mode="reflect")
        x = F.pad(x, (0, 0, 0, 0, front, 0))
    return x


def rope_plan(x, dim_head, rotary):
    triton_rms = (_video is not None and rotary is not None and cuda_inference(x, dtypes=_DTYPES)
                  and _video.can_rms_rope(dim_head, rotary[0].shape[-1]))
    family = choose("Video RMSNorm + RoPE", owned_triton=triton_rms, kitchen=_kitchen.can_fuse_rms_rope(x, rotary))
    cast = bool(choose(owned_triton=_video is not None and _video.CAST_ROPE_SUPPORTED and cuda_inference(x)))
    return family, cast


def normalize_projection(x_list, norm, rotary, plan):
    """Consume a (batch, tokens, heads, head_dim) projection, normalizing each row independently."""
    x = take(x_list)
    family, cast = plan
    if family is not None:
        return x
    dtype = x.dtype
    if not cast:
        from shared.attention_kit import rows_
        # Keep FP32 norm arithmetic without full-projection FP32 temporaries.
        rows_(x, lambda part, first, last: norm(part.float()).to(dtype))
        return x
    x = norm(x.float())
    return _video.cast_rope(x, rotary)


def finish_rope_(q, k, rotary, eps, plan):
    family, cast = plan
    if family == "kitchen":
        return _kitchen.rms_rope(q, k, rotary, eps)
    if family is not None:
        _video.rms_rope_(q, k, rotary, eps)
    elif not cast and rotary is not None:
        cos, sin = (table.to(q.dtype) for table in rotary)
        dim = cos.shape[-1]
        # Both outputs read the original pair before either half is overwritten.
        for x in (q, k):
            first, second = x[..., :dim].chunk(2, dim=-1)
            half = first.shape[-1]
            a = first * cos[..., :half] - second * sin[..., :half]
            b = second * cos[..., half:] + first * sin[..., half:]
            first.copy_(a)
            second.copy_(b)
    return q, k


def swiglu(x_list):
    if not choose("Video SwiGLU", owned_triton=_video is not None and cuda_inference(x_list[0], dtypes=_DTYPES)):
        return ops.swiglu(x_list)
    x = take(x_list)
    if x.dtype == torch.float16:
        return _video.swiglu_rn(x)
    return _video.swiglu(x) if kernel_policy.allow_approximate() else _video.swiglu_(x)


def scaled_residual(x_list, scale, residual):
    x = take(x_list)
    if choose(owned_triton=_video is not None and cuda_inference(x, dtypes=_DTYPES)):
        return _video.residual_(x, scale, residual)
    return x.mul_(scale).add_(residual)


def silu_(x, chunk_bytes=256 << 20):
    """x * sigmoid(x) written into x by chunks: the same values, without two full-size temporaries."""
    flat, rows = x.view(-1), max(1, chunk_bytes // x.element_size())
    for start in range(0, flat.numel(), rows):
        part = flat[start:start + rows]
        part.mul_(torch.sigmoid(part))
    return x


def group_norm_silu(x, norm, dtype, chunk_bytes=256 << 20):
    """silu(norm(x)) of a 2-D GroupNorm (N, C, H, W) as computed under autocast: normalization and SiLU in FP32, rounded once to dtype;
    x is borrowed. The WanGP kernel allocates only the output (shared.kernels.group_norm_silu). PyTorch: by chunks of groups (the FP32
    copies autocast would make), each rounded right away, so that neither the FP32 copy of x nor the FP32 result is allocated in full."""
    valid = _norm_silu is not None and norm.affine and _norm_silu.eligible(x, norm.num_groups, dtype)
    if choose("GroupNorm + SiLU", owned_triton=valid) is not None:
        return _norm_silu.group_norm_silu(x, norm.num_groups, norm.weight, norm.bias, norm.eps, dtype)
    out = torch.empty_like(x, dtype=dtype)
    step = x.shape[1] // norm.num_groups
    groups = max(1, chunk_bytes // (x[:, :step].numel() * 4))
    for start in range(0, norm.num_groups, groups):
        c0, c1 = start * step, min(norm.num_groups, start + groups) * step
        out[:, c0:c1] = silu_(F.group_norm(x[:, c0:c1].float(), (c1 - c0) // step, norm.weight[c0:c1].float(), norm.bias[c0:c1].float(), norm.eps), chunk_bytes)
    return out


def image_norm_conv_supported(x, norm):
    """Whether whole-image statistics + bounded normalized strips preserve this autocast path."""
    valid = (torch.is_autocast_enabled(x.device.type) and _norm_silu is not None and norm.affine
             and _norm_silu.strips_eligible(x, norm.num_groups, torch.get_autocast_dtype(x.device.type)))
    return choose("Image Norm + Conv Strips", owned_triton=valid) is not None


IMAGE_BAND_BYTES = 32 << 20


def image_norm_conv_is_banded(x, band_bytes=IMAGE_BAND_BYTES):
    """Whether a 16-bit normalized image exceeds the strip budget; smaller images need no interleaved workspace."""
    return x.numel() * 2 > band_bytes


def image_norm_conv_bands(x, norm, conv, band_bytes=IMAGE_BAND_BYTES):
    """Yield (first,last,conv_band) with global GN + SiLU before a 3x3, stride-1, padding-1 convolution.
    x is borrowed. Eligibility is checked by the caller. All statistics are read before the first yield; a one-row halo on each
    side gives every convolution its original inputs. Call the module itself so MMGP still owns its weight loading and hooks.
    A caller overwriting x must retain its last written row until the following band has consumed that row's halo."""
    n, c, height, width = x.shape
    dtype = torch.get_autocast_dtype(x.device.type)
    row_bytes = n * c * width * 2  # eligible output types are both 16-bit
    affine = _norm_silu.image_affine(x, norm.num_groups, norm.weight, norm.bias, norm.eps)
    rows = height if height * row_bytes <= band_bytes else max(1, band_bytes // row_bytes - 2)
    for start in range(0, height, rows):
        stop = min(height, start + rows)
        lo, hi = max(0, start - 1), min(height, stop + 1)
        band = _norm_silu.image_rows(x, affine, lo, hi, dtype)
        band = conv(band)[:, :, start-lo:stop-lo]
        yield start, stop, band
        del band


def image_norm_conv(x_list, norm, conv, *, out=None, add=False):
    """Consume x_list through image_norm_conv_bands. out is borrowed, disjoint from x, optionally a residual updated in place."""
    x = take(x_list)
    for start, stop, band in image_norm_conv_bands(x, norm, conv):
        if out is None:
            if start == 0 and stop == x.shape[2]:
                return band  # the whole result already exists; no destination allocation or copy is needed
            out = torch.empty((x.shape[0], band.shape[1], x.shape[2], band.shape[3]), dtype=band.dtype,
                              device=band.device, memory_format=torch.channels_last)
        target = out[:, :, start:stop]
        target.add_(band) if add else target.copy_(band)
        del band
    return out


SPARSE_SCRATCH_BYTES = 256 << 20  # bound of the gathered neighbor rows (PyTorch) or of the neighbor map computed per launch (kernel)


def sparse_conv(feats, level, weight, bias, *, rowmap=None, affine=None, residual=None, out=None, rows=None):
    """Submanifold 3x3x3 convolution of feats (N, Ci) 16-bit over the voxels of level (len(level) rows; level.neighbors(start, stop):
    their neighbor rows (rows, 27) int32 in FlexGEMM's tap order, -1 for inactive sites; level.cache["neighbors"]: the whole map when
    kept), weight (Co, 3, 3, 3, Ci), bias (Co,) or None. rowmap (len(level),): the feats row of each voxel (features on a coarser level),
    None when feats are the level's rows. affine: (scale, shift) FP32 (Ci,) of a group norm applied with SiLU to the neighbor features (in
    FP32, rounded to the activation dtype; inactive sites stay zero). residual: (len(level), Co) or a callable (start, stop) -> rows,
    added to the rounded convolution. out: the destination (len(level), Co), else a new tensor. rows: (start, stop), the output rows
    of these voxels alone (out and a residual tensor then hold stop - start rows; a callable residual still takes voxel indices).
    feats, weights and residual are borrowed. The WanGP implicit GEMM (shared.kernels.sparse_conv_triton: no gathered copy, FP32
    accumulation over all taps), else the gathered rows and addmm by chunks of rows."""
    first, last = (0, len(level)) if rows is None else rows
    co = weight.shape[0]
    out = feats.new_empty((last - first, co)) if out is None else out
    flat = weight.reshape(co, -1)
    # the kernel walks the input channels by K steps of 32 or 64
    if choose("Sparse Conv3D", owned_triton=_sparse is not None and cuda_inference(feats, dtypes=_DTYPES) and feats.is_contiguous() and feats.shape[1] % 32 == 0):
        full = level.cache.get("neighbors")
        if full is not None and not callable(residual):
            return _sparse.conv_(feats, full, flat, bias, out, rowmap=rowmap, affine=affine, residual=residual, rows=last - first, neighbor_offset=first)
        # a chunk's neighbor map is looked up through int64 keys and positions (about 32 bytes per entry) unless the level keeps it whole
        step = max(1, SPARSE_SCRATCH_BYTES // ((27 * 32 if full is None else 0) + (co * feats.element_size() if callable(residual) else 0)))
        for start in range(first, last, step):
            stop = min(start + step, last)
            neighbors = level.neighbors(start, stop) if full is None else full
            part = residual(start, stop) if callable(residual) else residual
            _sparse.conv_(feats, neighbors, flat, bias, out, rowmap=rowmap, affine=affine, residual=part, rows=stop - start,
                          neighbor_offset=0 if full is None else start, out_offset=start - first, residual_offset=0 if callable(residual) else start - first)
            del neighbors, part
        return out
    channels = feats.shape[1]
    step = max(1, SPARSE_SCRATCH_BYTES // (27 * channels * (4 + feats.element_size() if affine is not None else feats.element_size())))
    for start in range(first, last, step):
        stop = min(start + step, last)
        neighbors = level.neighbors(start, stop)
        index = neighbors.clamp(min=0)
        cols = feats.index_select(0, (index if rowmap is None else rowmap[index]).view(-1)).view(stop - start, 27, channels)
        if affine is not None:
            cols = F.silu(cols.float().mul_(affine[0]).add_(affine[1]), inplace=True).to(feats.dtype)
        cols.masked_fill_((neighbors < 0).unsqueeze(-1), 0)
        rows_out = torch.addmm(bias, cols.view(stop - start, -1), flat.t()) if bias is not None else cols.view(stop - start, -1) @ flat.t()
        del cols, index, neighbors
        if residual is not None:
            rows_out.add_(residual(start, stop) if callable(residual) else residual[start - first:stop - first])
        out[start - first:stop - first] = rows_out
    return out


def channel_moments(x):
    """(2, C) FP64 sums of x and of x^2 per channel of x (N, C), x borrowed: statistics of a group norm over voxels. The WanGP kernel
    reads x once (FP64 partial sums of exact 16-bit products), PyTorch by chunks of rows (FP32 squares summed in FP64)."""
    if choose("Channel Moments", owned_triton=_sparse is not None and cuda_inference(x, dtypes=_DTYPES) and x.is_contiguous()):
        return _sparse.moments(x)
    sums = torch.zeros((2, x.shape[1]), dtype=torch.float64, device=x.device)
    rows = max(1, (64 << 20) // (x.shape[1] * 4))
    for start in range(0, x.shape[0], rows):
        part = x[start:start + rows].float()
        sums[0] += part.sum(0, dtype=torch.float64)
        sums[1] += part.square().sum(0, dtype=torch.float64)  # not in place: part is x itself for FP32 features
        del part
    return sums


def attention(q, k, v, chunk_bytes=64 << 20):
    """F.scaled_dot_product_attention(q, k, v) of (batch, heads, tokens, dim) tensors: one call where SDPA has a flash, memory-efficient or
    cuDNN kernel for them. Elsewhere (BF16 on Turing...) its math path materializes every score in FP32, 2.4 GiB for the single-head
    mid-block attention of a VAE's 1024 px tile: there, by chunks of queries whose scores take at most chunk_bytes, each attending all the
    keys (values differ by rounding only)."""
    sdpa = torch.backends.cuda
    params = sdpa.SDPAParams(q, k, v, None, 0.0, False, False)
    rows = max(1, chunk_bytes // (q.shape[0] * q.shape[1] * k.shape[-2] * 4))
    if rows >= q.shape[-2] or q.device.type != "cuda" or sdpa.can_use_flash_attention(params) or sdpa.can_use_efficient_attention(params) or sdpa.can_use_cudnn_attention(params):
        return F.scaled_dot_product_attention(q, k, v)
    out = q.new_empty(q.shape[:-1] + v.shape[-1:])
    for start in range(0, q.shape[-2], rows):
        out[..., start:start + rows, :] = F.scaled_dot_product_attention(q[..., start:start + rows, :], k, v)
    return out


def pixel_norm_silu(x_list, norm, act, *, channel_norm):
    x = take(x_list)
    valid = (cuda_inference(x, dtypes=_DTYPES, approximate=True) and x.ndim == 5 and x.numel() > 0 and channel_norm
             and isinstance(act, torch.nn.SiLU))
    family = choose("PixelNorm + SiLU", wgpkernels=valid and native_ops.available(x.device),
                    triton=valid and _stream is not None and _conv.supported(x.device))
    if family == "wgpkernels":
        return native_ops.implementation.pixel_norm_silu(x, norm.eps)
    if family == "triton":
        return _stream.norm_silu(x, norm.eps)
    return act(norm(x))


def temporal_conv(conv, parts):
    """Consume temporal parts without keeping their original tensors alive."""
    b, c, _, h, w = parts[0].shape
    t, x = sum(part.shape[2] for part in parts), parts[0]
    valid = (cuda_inference(x, dtypes=_DTYPES, approximate=True) and c % 32 == 0 and conv.out_channels % 32 == 0
             and conv.stride == (1, 1, 1) and b * (t - 2) * h * w >= _STREAM_CONV_MIN_PIXELS
             and conv.kernel_size == (3, 3, 3) and conv.dilation == (1, 1, 1)
             and conv.groups == 1 and conv.padding == (0, 1, 1) and conv.padding_mode in ("zeros", "reflect")
             and conv.weight.dtype == x.dtype and conv.weight.device == x.device
             and (conv.bias is None or (conv.bias.dtype == x.dtype and conv.bias.device == x.device)))
    triton_conv = valid and _conv is not None and conv.bias is not None and _conv.conv_supported(x.device)
    # Native CUDA tuning improves its tiles, but it still loses to Triton and
    # sometimes cuDNN. Keep it out of automatic dispatch until competitive.
    family = choose("Streamed Conv3D", triton=triton_conv)
    if family is None:
        x = torch.cat(parts, dim=2) if len(parts) > 1 else x
        parts.clear()
        return conv(x)
    padded = torch.empty((b, c, t, h + 2, w + 2), dtype=x.dtype, device=x.device, memory_format=torch.channels_last_3d)
    x, start = None, 0
    for part in parts:
        padded[:, :, start:start + part.shape[2], 1:-1, 1:-1] = part
        start += part.shape[2]
    del part
    parts.clear()
    if conv.padding_mode == "reflect":
        padded[..., 0, 1:-1], padded[..., -1, 1:-1] = padded[..., 2, 1:-1], padded[..., -3, 1:-1]
        padded[..., 0], padded[..., -1] = padded[..., 2], padded[..., -3]
    else:
        padded[..., 0, :], padded[..., -1, :], padded[..., 0], padded[..., -1] = 0, 0, 0, 0
    return _conv.conv(padded, conv.weight, conv.bias)


def snake_beta_aa(x_list, activation, upsample, downsample, *, fusable):
    x = take(x_list)
    # Retain this audio path's established FP32 eligibility in both precision
    # modes; it has its own end-to-end validation, unlike video fusions.
    if choose("SnakeBeta + Resampling", owned_triton=fusable and _snake is not None and x.is_cuda):
        return _snake(x, activation.alpha, activation.beta, upsample.filter, downsample.lowpass.filter)
    return downsample(activation(upsample(x)))


def snake_(x, alpha, beta, *, log_scale=True):
    """Snake activation in place on x (batch, channels, length): x += sin(a x)^2 / (b + 1e-9), a and b per channel from alpha and beta
    (shape (1, channels, 1) or (channels,)), stored in log scale (a = exp(alpha), b = exp(beta)) or directly. The Triton kernel computes
    in FP32 from the stored parameters and rounds once at the store, in place of PyTorch's five rounded kernels (exp, mul, sin,
    square, scale, add: the Oobleck VAE of ACE-Step 1.5), so it is at least as precise in every dtype (checked against FP64 on random
    and edge inputs, tests/test_snake_op.py); the PyTorch path reproduces the module's chain."""
    if choose("Snake", owned_triton=_snake_plain is not None and cuda_inference(x, dtypes=_AUDIO_DTYPES) and x.is_contiguous()
              and alpha.device == x.device and beta.device == x.device):
        return _snake_plain(x, alpha, beta, log_scale)
    a = torch.exp(alpha) if log_scale else alpha
    b = torch.exp(beta) if log_scale else beta
    wave = torch.mul(x, a.reshape(1, -1, 1))
    wave.sin_().mul_(wave).mul_((b.reshape(1, -1, 1) + 1e-9).reciprocal())
    return x.add_(wave)


def neighborhood_status():
    from packaging.version import InvalidVersion, Version
    try:
        import triton
    except (ImportError, OSError):
        return False, "not installed"
    try:
        version = Version(triton.__version__)
    except InvalidVersion:
        return False, f"unknown version {triton.__version__}"
    if version < Version("3.2.0"):
        return False, str(version)
    return torch.cuda.is_available(), str(version)


_NA_AVAILABLE, _NA_VERSION = neighborhood_status()


def neighborhood_attention(qkv_list, kernel_size, fallback, *, keyframe_times=None, slots=None):
    if choose("3D Neighborhood Attention", owned_triton=_NA_AVAILABLE and qkv_list[0].is_cuda):
        from . import neighborhood_triton
        if keyframe_times is None:
            return neighborhood_triton.na3d(qkv_list, kernel_size)
        time = qkv_list[0].shape[1]
        video_slots, keyframe_slots = slots
        return neighborhood_triton.joint_na3d(qkv_list, video_slots(keyframe_times, time), keyframe_slots(keyframe_times, time), kernel_size)
    if keyframe_times is None:
        return fallback(qkv_list, kernel_size)
    return fallback(qkv_list, keyframe_times, kernel_size)
