# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Triton kernels of the causal video VAEs of the Wan architecture (Wan 2.1 and 2.2; the first Qwen VAE has the same layers) on
channels-last activations ("fast" kernel precision).

causal_pad: the input of each causal convolution built in one pass: RMS norm over the channels + SiLU (or the plain input), the cached
last frames of the previous chunk in front, the zero padding, and the next chunk's cache. Replaces the norm (5 passes), SiLU, cache copy,
concatenation and padding passes of the PyTorch path. FP32 arithmetic in registers with that path's rounding points: the norm rounded to
the activation dtype (F.normalize), IEEE square root and division, the scale and gamma products each rounded, SiLU with expf and IEEE
division. Only the sum of squares runs in another order.

conv (shared/kernels/conv3d_triton.py, also used by the streamed LTX VAE): implicit-GEMM convolution (FP32 accumulation) of the padded
input, with the bias and the residual of the block added as PyTorch rounds them, and the decoder's nearest x2 upsampling + 3x3
convolution read from the low-resolution input (no upsampled copy). 1.1-1.5x faster than cuDNN on RTX 50 GPUs, where it is enabled; only
the accumulation order differs.

causal_pad compiles once per power-of-two range of channel counts, mode and activation dtype; sizes, strides and paddings are not
specialized, so a new resolution, video length or tile size compiles nothing. Launch configurations start from a static choice and are tuned once per kernel,
identity and GPU on scratch outputs of a large call."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from shared.kernels.triton_tuner import LaunchTuner
from shared.kernels.conv3d_triton import channels_last_weight, conv, conv_eligible, conv_supported, supported  # noqa: F401 (used through this module)

_TUNER = LaunchTuner("wan2gp_wan_vae_kernels", 7, "Wan VAE")
_TUNE_ELEMENTS = 1 << 24  # calls writing at least this many elements tune the configurations
_CL = torch.channels_last_3d
_EPS = {dtype: torch.tensor(1e-12, dtype=dtype, device="cpu").item() for dtype in (torch.float16, torch.bfloat16, torch.float32)}  # F.normalize eps


@triton.jit(do_not_specialize=["TOTAL", "T", "H", "W", "NC", "FRONT", "PH", "PW", "KEEP", "SB", "SC", "ST", "SH", "SW"])
def _causal_pad_kernel(X, CACHE, GAMMA, OUT, KEPT, TOTAL, T, H, W, NC, FRONT, PH, PW, KEEP, SB, SC, ST, SH, SW, SCALE, EPS,
                       C, BLOCK_C: tl.constexpr, BLOCK_P: tl.constexpr, STRIDED: tl.constexpr, NORM: tl.constexpr,
                       REPLICATE: tl.constexpr):
    """BLOCK_P pixels (all channels) of the padded output (batch, FRONT + T, H + 2 PH, W + 2 PW, C): silu(rms_norm(x)) (NORM) or x inside,
    the NC cached frames (batch, NC, H, W, C) just before the first frame, zeros elsewhere, or with REPLICATE the nearest pixel of the
    frame and the earliest frame there is (the first cached one, else x's first); the last KEEP frames without the spatial padding also
    go to KEPT (batch, KEEP, H, W, C), the cache of the next chunk. x is channels-last contiguous, or read with the strides SB..SW
    (STRIDED)."""
    p = tl.program_id(0).to(tl.int64) * BLOCK_P + tl.arange(0, BLOCK_P)
    valid = p < TOTAL
    w = p % (W + 2 * PW) - PW
    r = p // (W + 2 * PW)
    h = r % (H + 2 * PH) - PH
    r = r // (H + 2 * PH)
    t = r % (FRONT + T) - FRONT
    b = r // (FRONT + T)
    inside = valid & (h >= 0) & (h < H) & (w >= 0) & (w < W)
    kept_frame = t >= T - KEEP
    if REPLICATE:
        h = tl.minimum(tl.maximum(h, 0), H - 1)
        w = tl.minimum(tl.maximum(w, 0), W - 1)
        t = tl.maximum(t, -NC)
        from_x = valid & (t >= 0)
        from_cache = valid & (t < 0)
    else:
        from_x = inside & (t >= 0)
        from_cache = inside & (t < 0) & (t >= -NC)
    c = tl.arange(0, BLOCK_C)
    lanes = c < C
    if STRIDED:
        x = tl.load(X + (b * SB + t * ST + h * SH + w * SW)[:, None] + (c * SC)[None, :], from_x[:, None] & lanes[None, :], other=0.0)
    else:
        x = tl.load(X + ((((b * T + t) * H + h) * W + w) * C)[:, None] + c[None, :], from_x[:, None] & lanes[None, :], other=0.0)
    dtype = OUT.dtype.element_ty
    if NORM:  # F.normalize(x, dim=1) * scale * gamma, then SiLU, rounded where the PyTorch path rounds
        x = x.to(tl.float32)
        norm = tl.sqrt_rn(tl.sum(x * x, axis=1)).to(dtype).to(tl.float32)
        norm = tl.maximum(norm, EPS).to(dtype).to(tl.float32)
        x = tl.div_rn(x, norm[:, None]).to(dtype).to(tl.float32)
        x = (x * SCALE).to(dtype).to(tl.float32)
        x = (x * tl.load(GAMMA + c, lanes, other=0.0).to(tl.float32)[None, :]).to(dtype).to(tl.float32)
        x = tl.div_rn(x, 1.0 + libdevice.exp(-x))
    cached = tl.load(CACHE + ((((b * NC + t + NC) * H + h) * W + w) * C)[:, None] + c[None, :], from_cache[:, None] & lanes[None, :], other=0.0)
    out = tl.where(from_x[:, None], x.to(dtype), cached.to(dtype))
    tl.store(OUT + (p * C)[:, None] + c[None, :], out, valid[:, None] & lanes[None, :])
    keep = inside & kept_frame
    tl.store(KEPT + ((((b * KEEP + t - T + KEEP) * H + h) * W + w) * C)[:, None] + c[None, :], out, keep[:, None] & lanes[None, :])


def causal_pad(x, pad, cache=None, norm=None, keep=0, replicate=False):
    """Channels-last input of a causal convolution with padding pad = (left, right, top, bottom, front, 0) (CausalConv3d._padding) for
    x (batch, channels, frames, height, width): silu(norm(x)) with an RMS_norm module (whose gamma and scale it reads), else x, after the
    last frames of the previous chunk (cache, at most `front` frames) and zero padding. keep > 0: also returns the last `keep` frames of
    that input without the spatial padding (the next chunk's cache): the frames that exist, so a first chunk shorter than `keep` keeps
    its own frames and no frame of zeros before them (the next chunk reads zeros before a shorter cache anyway). replicate: replicate
    padding instead of zeros, in space and before the earliest frame (the first cached one, else the first of x)."""
    batch, channels, frames, height, width = x.shape
    left, _, top, _, front, _ = pad
    strided = not x.is_contiguous(memory_format=_CL)
    if cache is not None and not cache.is_contiguous(memory_format=_CL):
        cache = cache.contiguous(memory_format=_CL)
    cached = 0 if cache is None else cache.shape[2]
    keep = min(keep, cached + frames)
    out = torch.empty((batch, channels, front + frames, height + 2 * top, width + 2 * left), dtype=x.dtype, device=x.device, memory_format=_CL)
    kept = torch.empty((batch, channels, keep, height, width), dtype=x.dtype, device=x.device, memory_format=_CL) if keep else out
    total = out.numel() // channels
    block_c = triton.next_power_of_2(channels)
    gamma, scale, eps = (norm.gamma, norm.scale, _EPS[x.dtype]) if norm is not None else (x, 1.0, 0.0)

    def launch(target, pixels, config, keep=keep):
        elements, warps = config
        block_p = max(1, elements // block_c)
        _causal_pad_kernel[(triton.cdiv(pixels, block_p),)](x, x if cache is None else cache, gamma, target, kept, pixels, frames, height, width,
                                                            cached, front, top, left, keep, *x.stride(), scale, eps, channels, block_c,
                                                            block_p, strided, norm is not None, replicate, num_warps=warps)

    def scratch():
        pixels = min(total, 1 << 16)
        target = torch.empty(pixels * channels, dtype=x.dtype, device=x.device)
        return lambda config: launch(target, pixels, config, 0)

    # timed per channel count (keeps the tuned entries), compiled per range: the candidates' binaries serve every count of the range
    identity = f"{channels}|{x.dtype}|{'norm' if norm is not None else 'copy'}|{'strided' if strided else 'cl'}{'|replicate' if replicate else ''}"
    launch(out, total, _TUNER.config("causal_pad", identity, x.device, ((4096, 4), (2048, 4), (8192, 8), (4096, 8)), scratch, out.numel() >= _TUNE_ELEMENTS))
    return (out, kept) if keep else out


from shared.kernels.triton_compilation_log import install_triton_compilation_logger  # noqa: E402

install_triton_compilation_logger()
