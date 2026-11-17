# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Triton convolutions for decoders that run by bands of rows (the tiny VAE previews): implicit-GEMM 3x3 convolutions with FP32 accumulation
on channels-last 16-bit activations, which compute a window of output rows from the rows a band holds.

- `_conv_rows`: zero padding 1 at the picture's own borders only. Output row `ho` reads input rows ho + R0 - 1 .. ho + R0 + 1; rows and
  columns outside the input are zeros, and the caller hands every row an interior output reads: no halo row is computed to be dropped. The
  bias, a residual and the ReLU are applied as it stores, rounded as PyTorch's separate operations round them. With UP the input is the
  half-size picture, its nearest x2 upsampling read in place.
- `_upconv_rows`: the same convolution of a nearest x2 upsampling, folded: an output pixel of parity (py, px) is a 2x2 convolution of the
  half-size input whose weights are the sums of the 3x3 taps that read the same input pixel. 4 products instead of 9; the summed weights
  are rounded once more, which only the last stage of a decoder tolerates without a measurable error (see `plan`).

Against cuDNN on an RTX 5090 (FP16 and BF16, channels-last): 204-240 against 113-175 TFLOPS from 64 to 512 channels, the same error
against a 64-bit reference; other GPUs time both once and keep the faster (_faster). One binary per channel counts and flags: sizes,
offsets, strides and batch are not specialized. Outputs are written at a row offset of the caller's tensor (rows of a wider buffer too):
batches of frames need no row slices, and image VAEs get their rows in place (conv2d).

The taps and their K tiles are one runtime loop (loads pipelined across taps), and every offset is a product by the channel count: with
separate loops per tap or free element strides, Triton loses the vector loads and stores (measured 20-60 TFLOPS on large pictures)."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

from shared.kernels import dispatch
from shared.kernels.triton_tuner import LaunchTuner

_CONFIG = (128, 64, 32, 4, 3)  # BLOCK_M, BN, BLOCK_K, warps, stages (measured: 64-256 pixels, K 32-64, 4-8 warps; see _Conv for 128 outputs)
_DTYPES = (torch.float16, torch.bfloat16)
_GROUP_M = 8 if torch.version.hip is None else 0  # see _grid
FRAMES_BYTES = 256 << 20  # frames decoded together: their largest activation within this (small frames are launch-bound one at a time)


@triton.jit(do_not_specialize=["M", "HO", "W", "HI", "R0", "RB", "RR", "RR0", "YB", "YR", "Y0"])
def _conv_rows(X, WT, BIAS, RES, Y, M, HO, W, HI, R0, RB, RR, RR0, YB, YR, Y0, CIN: tl.constexpr, COUT: tl.constexpr, BIASED: tl.constexpr,
               RESIDUAL: tl.constexpr, RELU: tl.constexpr, UP: tl.constexpr, BLOCK_M: tl.constexpr, BN1: tl.constexpr, BN2: tl.constexpr,
               BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    """BLOCK_M output pixels x BN1 + BN2 output channels (two products on the same input tile when the channels per program are no power of
    two: 96 = 64 + 32) of Y (batch, rows, W, COUT), rows Y0 + ho for ho in [0, HO). X (batch, HI, W, CIN), or with UP (batch, HI, W / 2,
    CIN); WT (COUT, 3, 3, CIN); RES (batch, rows, W, COUT) read at rows RR0 + ho. Y and RES are dense along a row, with their own strides
    between rows and pictures (YR, YB, RR, RB: rows of a wider buffer shared with another tensor), counted in pixels of COUT elements:
    offsets stay provable multiples of COUT, which the vector loads and stores need."""
    BN: tl.constexpr = BN1 + BN2
    FULL: tl.constexpr = COUT % BN == 0
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BLOCK_M)
    num_n: tl.constexpr = (COUT + BN - 1) // BN
    if GROUP_M > 0:  # programs ordered by groups of GROUP_M pixel tiles x every channel tile: the group's inputs and weights stay in cache
        first_m = (pid // (GROUP_M * num_n)) * GROUP_M
        size_m = tl.minimum(num_m - first_m, GROUP_M)
        pid_m = first_m + (pid % (GROUP_M * num_n)) % size_m
        pid_n = (pid % (GROUP_M * num_n)) // size_m
    else:  # a grid of pixel tiles x channel tiles
        pid_m = pid
        pid_n = tl.program_id(1)
    m = pid_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    rows = m < M
    w = m % W
    r = m // W
    ho = r % HO
    b = r // HO
    n1 = pid_n * BN + tl.arange(0, BN1)
    n2 = pid_n * BN + BN1 + tl.arange(0, BN2 if BN2 > 0 else BN1)
    cols = n1 < COUT
    k = tl.arange(0, BLOCK_K)
    KB: tl.constexpr = CIN // BLOCK_K
    acc1 = tl.zeros((BLOCK_M, BN1), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_M, BN2 if BN2 > 0 else BN1), dtype=tl.float32)
    for step in range(0, 9 * KB):  # one loop over the taps and their K tiles: the loads of the next products are pipelined across taps
        tap = step // KB
        uh = ho + R0 + tap // 3 - 1
        uw = w + tap % 3 - 1
        if UP:
            inside = rows & (uh >= 0) & (uh < 2 * HI) & (uw >= 0) & (uw < W)
            row = ((b * HI + uh // 2) * (W // 2) + uw // 2) * CIN + (step % KB) * BLOCK_K
        else:
            inside = rows & (uh >= 0) & (uh < HI) & (uw >= 0) & (uw < W)
            row = ((b * HI + uh) * W + uw) * CIN + (step % KB) * BLOCK_K
        a = tl.load(X + row[:, None] + k[None, :], inside[:, None], other=0.0)
        if FULL:
            weights = tl.load(WT + (n1 * 9 * CIN)[:, None] + (step * BLOCK_K + k)[None, :])
        else:
            weights = tl.load(WT + (n1 * 9 * CIN)[:, None] + (step * BLOCK_K + k)[None, :], cols[:, None], other=0.0)
        acc1 = tl.dot(a, tl.trans(weights), acc1)
        if BN2 > 0:
            acc2 = tl.dot(a, tl.trans(tl.load(WT + (n2 * 9 * CIN)[:, None] + (step * BLOCK_K + k)[None, :])), acc2)
    dtype = Y.dtype.element_ty
    pixel = (b * YB + (ho + Y0) * YR + w) * COUT
    source = (b * RB + (ho + RR0) * RR + w) * COUT
    y = acc1.to(dtype)
    if BIASED:
        y = (y.to(tl.float32) + tl.load(BIAS + n1, cols, other=0.0).to(tl.float32)[None, :]).to(dtype)
    if RESIDUAL:
        y = (y.to(tl.float32) + tl.load(RES + source[:, None] + n1[None, :], rows[:, None] & cols[None, :], other=0.0).to(tl.float32)).to(dtype)
    if RELU:
        y = tl.maximum(y, 0.0)
    if FULL:
        tl.store(Y + pixel[:, None] + n1[None, :], y, rows[:, None])
    else:
        tl.store(Y + pixel[:, None] + n1[None, :], y, rows[:, None] & cols[None, :])
    if BN2 > 0:
        y = acc2.to(dtype)
        if BIASED:
            y = (y.to(tl.float32) + tl.load(BIAS + n2).to(tl.float32)[None, :]).to(dtype)
        if RESIDUAL:
            y = (y.to(tl.float32) + tl.load(RES + source[:, None] + n2[None, :], rows[:, None], other=0.0).to(tl.float32)).to(dtype)
        if RELU:
            y = tl.maximum(y, 0.0)
        tl.store(Y + pixel[:, None] + n2[None, :], y, rows[:, None])


@triton.jit(do_not_specialize=["M", "HO2", "WL", "HI", "I0", "YB", "YR", "Y0"])
def _upconv_rows(X, WT, Y, M, HO2, WL, HI, I0, YB, YR, Y0, CIN: tl.constexpr, COUT: tl.constexpr, RELU: tl.constexpr, BLOCK_M: tl.constexpr, BN: tl.constexpr,
                 BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
    """X (batch, HI, WL, CIN); Y (batch, rows, 2 WL, COUT) with the strides YB, YR between pictures and rows (in pixels): output rows Y0 + 2 ir + py of the row pairs ir in [0, HO2), which read X rows
    ir + I0 - 1 + py and ir + I0 + py. WT (COUT, 4 phases, 2, 2, CIN). One program: BLOCK_M input positions of one phase (program_id(1) =
    2 py + px) x BN output channels. No bias (the convolutions after an upsampling have none); RELU: the ReLU as it stores."""
    pid = tl.program_id(0)
    phase = tl.program_id(2)
    py = phase >> 1
    px = phase & 1
    num_m = tl.cdiv(M, BLOCK_M)
    num_n: tl.constexpr = COUT // BN
    if GROUP_M > 0:  # programs ordered by groups of GROUP_M pixel tiles x every channel tile: the group's inputs and weights stay in cache
        first_m = (pid // (GROUP_M * num_n)) * GROUP_M
        size_m = tl.minimum(num_m - first_m, GROUP_M)
        pid_m = first_m + (pid % (GROUP_M * num_n)) % size_m
        pid_n = (pid % (GROUP_M * num_n)) // size_m
    else:  # a grid of pixel tiles x channel tiles
        pid_m = pid
        pid_n = tl.program_id(1)
    m = pid_m.to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    rows = m < M
    j = m % WL
    r = m // WL
    ir = r % HO2
    b = r // HO2
    n = pid_n * BN + tl.arange(0, BN)
    k = tl.arange(0, BLOCK_K)
    KB: tl.constexpr = CIN // BLOCK_K
    acc = tl.zeros((BLOCK_M, BN), dtype=tl.float32)
    for step in range(0, 4 * KB):
        tap = step // KB
        li = ir + I0 + tap // 2 - 1 + py
        lj = j + tap % 2 - 1 + px
        inside = rows & (li >= 0) & (li < HI) & (lj >= 0) & (lj < WL)
        row = ((b * HI + li) * WL + lj) * CIN + (step % KB) * BLOCK_K
        a = tl.load(X + row[:, None] + k[None, :], inside[:, None], other=0.0)
        weights = tl.load(WT + (n * 16 * CIN)[:, None] + (phase * 4 * CIN + step * BLOCK_K + k)[None, :])
        acc = tl.dot(a, tl.trans(weights), acc)
    target = (b * YB + (Y0 + 2 * ir + py) * YR + 2 * j + px) * COUT
    y = acc.to(Y.dtype.element_ty)
    if RELU:
        y = tl.maximum(y, 0.0)
    tl.store(Y + target[:, None] + n[None, :], y, rows[:, None])


def _tiles(cout):
    """(BN1, BN2, pipelining stages): the output channels per program. Measured on an RTX 5090: 128 where it divides (4 stages), 64, and
    for the counts no power of two divides well a split tile, two products on one loaded input tile (96 = 64 + 32, 144 = 128 + 16,
    48 = 32 + 16) instead of tiles of 32 or 16 that reload the input 2 to 9 times as often. Fewer than 16 outputs: one masked tile of 16.
    AMD: single tiles of at most 64 (64 KiB of shared memory)."""
    if torch.version.hip is None:
        for size, bn1, bn2 in ((128, 128, 0), (64, 64, 0), (96, 64, 32), (144, 128, 16), (48, 32, 16)):
            if cout % size == 0:
                return bn1, bn2, 4 if bn1 == 128 and bn2 == 0 else _CONFIG[4]
    return next((tile for tile in (64, 32, 16) if cout % tile == 0), 16), 0, _CONFIG[4]


def _grid(pixel_tiles, channel_tiles, *rest):
    """One axis of programs ordered by groups of pixel tiles, or with Triton's AMD backend, which does not compile that ordering nor a
    division of the program index (its buffer-operations pass fails), pixel tiles x channel tiles."""
    return (pixel_tiles * channel_tiles, *rest) if _GROUP_M else (pixel_tiles, channel_tiles)


def fold_upsample_weight(weight):
    """(cout, cin, 3, 3) -> (cout, 4 phases, 2, 2, cin) in the weight's dtype: the taps of a 3x3 convolution of a nearest x2 upsampling that read
    the same input pixel, summed in FP32 (phase 2 py + px; even rows: tap 0 | taps 1 + 2, odd rows: taps 0 + 1 | tap 2; columns alike)."""
    w = weight.float()
    by_row = (torch.stack((w[:, :, 0], w[:, :, 1] + w[:, :, 2]), 2), torch.stack((w[:, :, 0] + w[:, :, 1], w[:, :, 2]), 2))  # (cout, cin, 2, 3) per py
    phases = [torch.stack((v[..., 0], v[..., 1] + v[..., 2]), -1) if px == 0 else torch.stack((v[..., 0] + v[..., 1], v[..., 2]), -1) for v in by_row for px in range(2)]
    return torch.stack(phases, 1).permute(0, 1, 3, 4, 2).contiguous().to(weight.dtype)


class _Conv:
    """A 3x3 convolution's weights prepared for the kernels: channels-last, the input channels padded to a multiple of 32 with zeros, the
    weights folded for an upsampled input (fold)."""

    def __init__(self, conv, fold=False, dtype=None):
        weight, bias = conv.weight, conv.bias
        if dtype is not None and weight.dtype != dtype:  # FP32 weights under autocast: rounded to the activation dtype, as autocast does
            weight, bias = weight.to(dtype), None if bias is None else bias.to(dtype)
        fold = fold and torch.version.hip is None  # Triton's AMD backend does not compile _upconv_rows: the nine taps read in place there
        self.cout, self.fold, self.bias = weight.shape[0], fold, bias
        self.block_k = 16 if weight.shape[1] % 32 == 16 else 32  # 16 or 48 input channels: K tiles of 16, no padded input
        self.cin = -(-weight.shape[1] // self.block_k) * self.block_k
        if self.cin != weight.shape[1]:
            weight = F.pad(weight, (0, 0, 0, 0, 0, self.cin - weight.shape[1]))
        self.weight = fold_upsample_weight(weight) if fold else weight.permute(0, 2, 3, 1).contiguous()
        self.bn1, self.bn2, self.stages = _tiles(self.cout)
        self.bn = next((tile for tile in (_CONFIG[1], 32) if self.cout % tile == 0), 16)  # of the folded kernel

    def rows(self, x, r0, out_rows, relu=False, residual=None, residual_r0=0, out=None, out_r0=0, up=False):
        """Output rows [0, out_rows) of the convolution of x (batch, cin, hi, w): output row ho reads x rows ho + r0 - 1 .. ho + r0 + 1, or
        with up the rows of x's x2 upsampling; plus the bias, the residual's rows residual_r0 + ho, the ReLU; written to out rows out_r0 + ho."""
        batch, _, hi, width = x.shape
        width *= 2 if up else 1
        if out is None:
            out = torch.empty((batch, self.cout, out_rows, width), dtype=x.dtype, device=x.device, memory_format=torch.channels_last)
        (block_m, _, _, warps, _), block_k, stages = _CONFIG, self.block_k, self.stages
        count, res = batch * out_rows * width, out if residual is None else residual
        _conv_rows[_grid(triton.cdiv(count, block_m), -(-self.cout // (self.bn1 + self.bn2)))](
            x, self.weight, self.weight if self.bias is None else self.bias, res, out, count, out_rows, width, hi, r0, res.stride(0) // self.cout,
            res.stride(2) // self.cout, residual_r0, out.stride(0) // self.cout, out.stride(2) // self.cout, out_r0, self.cin, self.cout, self.bias is not None, residual is not None, relu, up, block_m, self.bn1, self.bn2, block_k, _GROUP_M,
            num_warps=warps, num_stages=stages)
        return out

    def up_rows(self, x, i0, pairs, relu=False, out=None, out_r0=0):
        """Output row pairs [0, pairs) of the convolution of the nearest x2 upsampling of x: pair ir reads x rows ir + i0 - 1 .. ir + i0 + 1."""
        if not self.fold:
            return self.rows(x, 2 * i0, 2 * pairs, relu=relu, out=out, out_r0=out_r0, up=True)
        batch, _, hi, width = x.shape
        if out is None:
            out = torch.empty((batch, self.cout, 2 * pairs, 2 * width), dtype=x.dtype, device=x.device, memory_format=torch.channels_last)
        (block_m, _, _, warps, _), block_k, stages = _CONFIG, self.block_k, self.stages
        count = batch * pairs * width
        _upconv_rows[(*_grid(triton.cdiv(count, block_m), self.cout // self.bn, 1), 4)](
            x, self.weight, out, count, pairs, width, hi, i0, out.stride(0) // self.cout, out.stride(2) // self.cout, out_r0, self.cin, self.cout, relu, block_m, self.bn, block_k, _GROUP_M, num_warps=warps, num_stages=stages)
        return out


_TUNER = LaunchTuner("wan2gp_band_conv", 1, "VAE")
_RACE_PIXELS = 1 << 17  # pixels of the timed sample, and the smallest call that is timed


def _faster(x, conv, relu):
    """Whether the row-window kernel is the faster convolution on x's GPU. Compute capability 12.x: measured (1.3-1.8x cuDNN at every
    shape). Other GPUs: timed once against cuDNN per convolution shape, dtype and GPU, on a bounded sample of the call's rows written to
    scratch outputs, and kept in the tuning cache; until a call large enough to time arrives, cuDNN. conv None: a decoder stage, timed
    on a 64 -> 64 convolution with its ReLU."""
    found = dispatch.capability(x.device.index)
    if found is not None and found[0] == 12:
        return True
    stage = conv is None

    def scratch():
        sample, layer = x, conv
        if stage:  # nothing of a decoder is at hand: a 64 -> 64 convolution of zeros (the time does not depend on the values)
            layer = nn.Conv2d(64, 64, 3, padding=1, bias=False, device="meta")
            layer.weight = nn.Parameter(torch.zeros(layer.weight.shape, dtype=x.dtype, device=x.device), False)
            sample = torch.zeros((1, 64, 256, _RACE_PIXELS // 256), dtype=x.dtype, device=x.device).contiguous(memory_format=torch.channels_last)
        rows = min(sample.shape[2], max(8, _RACE_PIXELS // sample.shape[3]))
        sample, prepared = sample[:1, :, :rows], _Conv(layer, dtype=x.dtype)
        out = torch.empty((1, prepared.cout, rows, sample.shape[3]), dtype=x.dtype, device=x.device, memory_format=torch.channels_last)
        weight, bias = layer.weight.to(x.dtype), None if layer.bias is None else layer.bias.to(x.dtype)

        def launch(config):
            if config[0] == "triton":
                return prepared.rows(sample, 0, rows, relu=relu or stage, out=out)
            with torch.autocast(x.device.type, enabled=False):
                y = F.conv2d(sample, weight, bias, padding=1)
            return y.relu_() if relu or stage else y
        return launch

    identity = f"stage|{x.dtype}" if stage else f"{conv.in_channels}|{conv.out_channels}|{'relu' if relu else ''}|{x.dtype}"
    tunable = stage or x.shape[0] * x.shape[2] * x.shape[3] >= _RACE_PIXELS
    return _TUNER.config("backend", identity, x.device, (("cudnn",), ("triton",)), scratch, tunable)[0] == "triton"


def eligible(x, conv=None, relu=False):
    """The band kernels for the 16-bit activation x: no autograd nor compilation, a GPU that runs these Triton kernels (NVIDIA from
    the dtype's floor: FP16 on compute capability 7.5, BF16 on 8.0; AMD), and the kernel being the faster convolution there (_faster)."""
    return dispatch.cuda_inference(x, dtypes=_DTYPES, hip=True) and _faster(x, conv, relu)


class Weights:
    """A 3x3 convolution with stride 1 and zero padding 1 given by its tensors (weight (cout, cin, 3, 3), bias or None), for layers that
    are no nn.Conv2d: one time slice of a causal 3D convolution run on a single frame."""

    def __init__(self, weight, bias):
        self.weight, self.bias, (self.out_channels, self.in_channels) = weight, bias, weight.shape[:2]


def _plain(conv):
    if type(conv) is Weights:
        return conv.weight.shape[2:] == (3, 3)
    return (type(conv) is nn.Conv2d and conv.kernel_size == (3, 3) and conv.stride == (1, 1) and conv.padding == (1, 1) and conv.dilation == (1, 1)
            and conv.groups == 1 and conv.padding_mode == "zeros")


MIN_PIXELS = 8192  # fewer output pixels: cuDNN's lower launch cost wins (as conv3d_triton's convolutions)


def conv2d_eligible(x, conv, upsample=False, mode="nearest", out=None, layout=True):
    """The row-window kernel for conv (3x3, stride 1, zero padding 1, input channels by 16) on the channels-last 16-bit activation x, or
    on its x2 upsampling (nearest; nearest-exact picks the same pixels at this scale), with the weights on x's device; out, when the
    caller supplies the destination: of x's dtype and dense along each row (rows and pictures a whole number of pixels apart). Without
    layout: whatever x's layout, for a caller that makes it channels-last itself."""
    if out is not None and (out.dtype != x.dtype or out.stride(1) != 1 or out.stride(3) != out.shape[1] or out.stride(2) % out.shape[1] or out.stride(0) % out.shape[1]):
        return False
    return (x.dim() == 4 and _plain(conv) and conv.in_channels % 16 == 0 and (not layout or x.is_contiguous(memory_format=torch.channels_last))
            and conv.weight.device == x.device and (not upsample or mode in ("nearest", "nearest-exact"))
            and x.shape[0] * x.shape[2] * x.shape[3] * (4 if upsample else 1) >= MIN_PIXELS and eligible(x, conv))


def conv2d(x, conv, *, upsample=False, window=None, out=None, out_row=0, add=False):
    """conv(x), or conv of x's nearest x2 upsampling, for the channels-last 16-bit x (conv2d_eligible): the output rows window = (first,
    count), all of them by default, written into out from its row out_row (a new tensor by default) and, with add, added to what out
    holds there. Rounded as PyTorch rounds the separate operations: the convolution, plus the bias, plus out. Reads x in place: no
    workspace, no upsampled input, no halo row computed to be dropped."""
    first, count = (0, x.shape[2] * (2 if upsample else 1)) if window is None else window
    out = _Conv(conv, dtype=x.dtype).rows(x, first, count, residual=out if add else None, residual_r0=out_row, out=out, out_r0=out_row, up=upsample)
    torch.autograd.graph.increment_version(out)
    return out


def plan(layers, fold):
    """The layers of a decoder stage (an optional nearest x2 upsampling first) as kernel operations, None when a layer is not one of them:
    ('up', conv) | ('conv', conv, relu) | ('block', conv, conv, conv, shortcut) for a residual block (.conv = conv, ReLU, conv, ReLU, conv;
    .skip; .fuse = ReLU; no .pool) | ('module', layer) for a first layer applied to the whole input (the clamp of the latent).
    fold: the convolution after the upsampling uses the folded weights (the decoder's last stage: measured as precise as cuDNN there, while
    the rounding of the summed weights raises the error by 5 % when it enters two stages earlier)."""
    ops, index = [], 0
    if layers and not isinstance(layers[0], (nn.Upsample, nn.Conv2d)) and not hasattr(layers[0], "skip"):
        ops.append(("module", layers[0]))
        index = 1
    while index < len(layers):
        layer = layers[index]
        if isinstance(layer, nn.Upsample):
            following = layers[index + 1] if index + 1 < len(layers) else None
            if index or layer.mode != "nearest" or layer.scale_factor != 2 or following is None or not _plain(following) or following.bias is not None or following.out_channels % 32:
                return None
            ops.append(("up", _Conv(following, fold)))
            index += 2
        elif isinstance(layer, nn.Conv2d):
            relu = index + 1 < len(layers) and isinstance(layers[index + 1], nn.ReLU)
            if not _plain(layer) or layer.out_channels % 16 and layer.out_channels >= 16:
                return None
            ops.append(("conv", _Conv(layer), relu))
            index += 2 if relu else 1
        elif hasattr(layer, "skip") and hasattr(layer, "fuse") and getattr(layer, "pool", None) is None:
            convs = [layer.conv[position] for position in (0, 2, 4)]
            if len(layer.conv) != 5 or not all(_plain(conv) and conv.out_channels % 32 == 0 for conv in convs) or not isinstance(layer.fuse, nn.ReLU):
                return None
            ops.append(("block", *(_Conv(conv) for conv in convs), None if isinstance(layer.skip, nn.Identity) else layer.skip))
            index += 1
        else:
            return None
    return ops


def frames_together(ops, height, width, element_size):
    """How many frames of this size the stage `ops` may decode in one call: its largest activation of all of them within FRAMES_BYTES."""
    widest = max(op[1].cout for op in ops if op[0] != "module")
    return max(1, min(8, FRAMES_BYTES // (widest * height * width * element_size)))


def run_stage(ops, x, band_bytes):
    """The stage on x (batch, c, h, w): its whole output, channels-last, computed by bands of output rows whose largest activation fits
    band_bytes. Each operation computes only the rows that the band's output reads: three more rows on each side per residual block and one
    per convolution, from the last operation back, cut at the picture's borders."""
    if ops[0][0] == "module":
        x, ops = ops[0][1](x), ops[1:]
    scale = 2 if ops[0][0] == "up" else 1
    first = ops[0][1]
    if x.shape[1] != first.cin:  # input channels padded with zeros to the kernels' multiple of 32 (a latent)
        x = F.pad(x, (0, 0, 0, 0, 0, first.cin - x.shape[1]))
    x = x.contiguous(memory_format=torch.channels_last)
    batch, height, width = x.shape[0], x.shape[2] * scale, x.shape[3] * scale
    last = ops[-1]
    out = torch.empty((batch, last[3].cout if last[0] == "block" else last[1].cout, height, width), dtype=x.dtype, device=x.device, memory_format=torch.channels_last)
    rows = max(2, band_bytes // (batch * max(op[1].cout for op in ops) * width * x.element_size()) // 2 * 2)
    for start in range(0, height, rows):
        windows, low, high = [], start, min(height, start + rows)
        for op in reversed(ops):  # the rows each operation's output must hold
            if op[0] == "up":
                low, high = low - low % 2, high + high % 2
            windows.append((low, high))
            reach = 3 if op[0] == "block" else 1
            low, high = max(0, low - reach), min(height, high + reach)
        value, value_row = x, 0
        for position, (op, (low, high)) in enumerate(zip(ops, reversed(windows))):
            target, target_row = (out, low) if position == len(ops) - 1 else (None, 0)
            if op[0] == "up":
                result = op[1].up_rows(value, low // 2 - value_row, (high - low) // 2, out=target, out_r0=target_row)
            elif op[0] == "conv":
                result = op[1].rows(value, low - value_row, high - low, relu=op[2], out=target, out_r0=target_row)
            else:
                inner, outer = (max(0, low - 1), min(height, high + 1)), (max(0, low - 2), min(height, high + 2))
                a = op[1].rows(value, outer[0] - value_row, outer[1] - outer[0], relu=True)
                b = op[2].rows(a, inner[0] - outer[0], inner[1] - inner[0], relu=True)
                del a
                shortcut, shortcut_row = value, value_row
                if op[4] is not None:  # a 1x1 shortcut convolution, of the rows the sum reads
                    shortcut, shortcut_row = op[4](value[:, :, low - value_row:high - value_row]), low
                result = op[3].rows(b, low - inner[0], high - low, relu=True, residual=shortcut, residual_r0=low - shortcut_row, out=target, out_r0=target_row)
                del b, shortcut
            value, value_row = result, low
        del value, result
    torch.autograd.graph.increment_version(out)
    return out


from shared.kernels.triton_compilation_log import install_triton_compilation_logger  # noqa: E402

install_triton_compilation_logger()
