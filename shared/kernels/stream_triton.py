# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Triton kernel of the streamed LTX VAE ("fast" kernel precision): PixelNorm + SiLU in one pass, written channels last for the Triton
convolution that follows. FP32 arithmetic in registers with the PyTorch path's rounding points (x ** 2, the mean, + eps, the square root
and the division each rounded to the activation dtype, SiLU with expf and IEEE division); only the sum of squares runs in another order.
One binary per power-of-two range of channel counts, dtype and input layout; sizes, strides and the channel count itself are not
specialized."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from shared.kernels.triton_tuner import LaunchTuner

_TUNER = LaunchTuner("wan2gp_ltx_vae_kernels", 2, "LTX VAE")
_TUNE_ELEMENTS = 1 << 24  # calls writing at least this many elements tune the configurations
_CL = torch.channels_last_3d


@triton.jit(do_not_specialize=["TOTAL", "T", "H", "W", "SB", "SC", "ST", "SH", "SW"],
            do_not_specialize_on_alignment=["TOTAL", "T", "H", "W", "SB", "SC", "ST", "SH", "SW"])
def _norm_silu_kernel(X, OUT, TOTAL, T, H, W, SB, SC, ST, SH, SW, EPS, INV_C, C, BLOCK_C: tl.constexpr, BLOCK_P: tl.constexpr):
    """BLOCK_P pixels (all channels) of the channels-last output (batch, T, H, W, C): silu(x / sqrt(mean(x ** 2) + EPS)), x read with the
    strides SB..SW."""
    p = tl.program_id(0).to(tl.int64) * BLOCK_P + tl.arange(0, BLOCK_P)
    w = p % W
    r = p // W
    h = r % H
    r = r // H
    t = r % T
    b = r // T
    c = tl.arange(0, BLOCK_C)
    mask = (p < TOTAL)[:, None] & (c < C)[None, :]
    x = tl.load(X + (b * SB + t * ST + h * SH + w * SW)[:, None] + (c * SC)[None, :], mask, other=0.0).to(tl.float32)
    dtype = OUT.dtype.element_ty
    mean = (tl.sum((x * x).to(dtype).to(tl.float32), axis=1) * INV_C).to(dtype).to(tl.float32)
    rms = tl.sqrt_rn((mean + EPS).to(dtype).to(tl.float32)).to(dtype).to(tl.float32)
    x = tl.div_rn(x, rms[:, None]).to(dtype).to(tl.float32)
    x = tl.div_rn(x, 1.0 + libdevice.exp(-x))
    tl.store(OUT + (p * C)[:, None] + c[None, :], x.to(dtype), mask)


def norm_silu(x, eps):
    """silu(PixelNorm(x)) over the channels of x (batch, channels, frames, height, width), any layout; the output channels last."""
    batch, channels, frames, height, width = x.shape
    out = torch.empty_like(x, memory_format=_CL)
    total = out.numel() // channels
    block_c = triton.next_power_of_2(channels)

    def launch(target, pixels, config):
        elements, warps = config
        block_p = max(1, elements // block_c)
        _norm_silu_kernel[(triton.cdiv(pixels, block_p),)](x, target, pixels, frames, height, width, *x.stride(), eps, 1.0 / channels, channels,
                                                           block_c, block_p, num_warps=warps)

    def scratch():
        pixels = min(total, 1 << 16)
        target = torch.empty(pixels * channels, dtype=x.dtype, device=x.device)
        return lambda config: launch(target, pixels, config)

    identity = f"{channels}|{x.dtype}|{'cl' if x.is_contiguous(memory_format=_CL) else 'strided'}"  # timed per count; compiled per range
    launch(out, total, _TUNER.config("norm_silu", identity, x.device, ((4096, 4), (2048, 4), (8192, 8), (4096, 8)), scratch, out.numel() >= _TUNE_ELEMENTS))
    return out


from shared.kernels.triton_compilation_log import install_triton_compilation_logger  # noqa: E402

install_triton_compilation_logger()
