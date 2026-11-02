# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""silu(group_norm(x)) of 16-bit NCHW activations, the GroupNorm + SiLU before each convolution of image VAEs, as PyTorch computes it under
autocast (GroupNorm and SiLU in FP32, one rounding to the output dtype), without its FP32 temporaries: PyTorch writes an FP32 copy of x, the
FP32 normalization and the FP32 sigmoid (two full-size FP32 tensors per chunk of groups, about 44 bytes moved per element), where these
kernels read x three times and write the output once (8 bytes per element) and allocate only the output.

Triton (compute capability 7.5 and up; BF16 read and written as bits on every GPU, so Turing and the others give the same values):
- each program sums a span of a group (contiguous in NCHW) in FP32 accumulators that carry their rounding errors (two-sum), reduced in FP64,
  and takes the largest |x|: the group's mean is as close to the exact one as FP64 rounding of exact partial sums allows (PyTorch: FP32
  Welford). x is summed times 2^-64 (exact) so that no 16-bit value can overflow the accumulators;
- a second pass sums the squares of (x - m32) * 2^-e the same way, m32 the mean rounded to FP32 and 2^e the power of two above
  max|x| + |m32| (no overflow nor loss whatever the magnitude); the variance corrects the rounding of the mean exactly
  (E[(x - m32)^2] - (m - m32)^2): no cancellation however large the mean against the spread;
- a third pass gives each program one channel's span, with scale = rstd * weight and the shift as scalars: y = (x - m32) * scale + shift
  (computed as ((x - m32) * 2^-e) * (scale * 2^e), the same values for normal numbers), then y * sigmoid(y) with the precise exponential
  and IEEE division (as PyTorch's FP32 sigmoid), rounded once to nearest even.
Activations below MIN_BYTES keep the PyTorch route: the three launches cost more there (a floor of ~0.1 ms on an RTX 4090, ~0.43 ms on
an RTX 2080 Ti under Windows), measured against the chunked route at the Flux VAE shapes. Partial sums are combined in a fixed order:
deterministic for a given GPU and configuration. The launch configuration (elements per load,
warps, pipelining stages, programs per SM) is tuned once per GPU, channels per group and dtype, on scratch output (x is only read); the
lengths and spans are runtime values: a new resolution or tile size compiles nothing. The image-strip kernels below specialize only
on the model's channel/group widths and layout, sharing coalesced loads across adjacent groups while preserving the same arithmetic."""
import functools

import torch

from . import dispatch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
    from .bf16_bits import bf16_bits_to_fp32, fp32_to_bf16_bits
    from .triton_tuner import LaunchTuner
except ImportError:
    triton = None

MIN_BYTES = 8 << 20  # smaller activations: PyTorch (RTX 4090: kernel x1.13 at 4 MiB, x0.79 at 8 MiB; RTX 2080 Ti: x1.0 at 4, x0.5 at 16)
MAX_SPANS = 128  # partial sums per group (one masked load combines them)
# (elements per load, warps, stages, programs per SM); the first is the untuned default
_CANDIDATES = ((2048, 8, 3, 4), (1024, 4, 3, 4), (4096, 8, 2, 4), (2048, 8, 3, 8), (1024, 4, 3, 8), (4096, 8, 2, 2), (2048, 4, 4, 4))
_TUNE_MIN_BYTES = 32 << 20  # smaller activations use the default configuration (their timings would represent launch latency)
_TUNE_MAX_BYTES = 256 << 20  # the scratch output of a tuning is at most this (a channel slice of larger activations)

if triton is not None:
    _TUNER = LaunchTuner("wan2gp_group_norm_silu", 1, "VAE")

    @triton.jit
    def _load(X, index, mask, BITS: tl.constexpr):
        x = tl.load(X + index, mask, 0)
        return bf16_bits_to_fp32(x) if BITS else x.to(tl.float32)

    @triton.jit
    def _group_total(PARTIAL, group, SPANS):
        span = tl.arange(0, 128)  # MAX_SPANS
        return tl.sum(tl.load(PARTIAL + group * SPANS + span, span < SPANS, 0.0), axis=0)

    @triton.jit
    def _group_stats(SUMS, MAXES, group, SPANS, LENGTH):
        """The group's mean (FP64), its rounding to FP32 and the power of two 2^-e above max|x| + |m32| (FP32, with e)."""
        span = tl.arange(0, 128)  # MAX_SPANS
        mean = _group_total(SUMS, group, SPANS) * 18446744073709551616.0 / LENGTH.to(tl.float64)  # sums of x * 2^-64
        mean32 = mean.to(tl.float32)
        bound = tl.max(tl.load(MAXES + group * SPANS + span, span < SPANS, 0.0), axis=0).to(tl.float64) + tl.abs(mean32).to(tl.float64)
        e = tl.minimum(tl.maximum(((bound.to(tl.int64, bitcast=True) >> 52) & 0x7FF) - 1022, -120), 120).to(tl.int32)  # bound < 2^e
        return mean, mean32, ((127 - e) << 23).to(tl.float32, bitcast=True), e

    @triton.jit(do_not_specialize=["LENGTH", "SPAN", "SPANS"])
    def _sums_kernel(X, PARTIAL, SUMS, MAXES, LENGTH, SPAN, SPANS, CENTERED: tl.constexpr, BLOCK: tl.constexpr, STAGES: tl.constexpr,
                     BITS: tl.constexpr):
        # partial sums of a span of a group: of x * 2^-64 with its largest |x| into MAXES, or with CENTERED of ((x - m32) * 2^-e)^2
        span, group = tl.program_id(0), tl.program_id(1).to(tl.int64)
        start = span.to(tl.int64) * SPAN
        end = tl.minimum(start + SPAN, LENGTH)
        base = X + group * LENGTH
        if CENTERED:
            _, mean32, unit, _ = _group_stats(SUMS, MAXES, group, SPANS, LENGTH)
            shift = mean32 * unit
        else:
            unit, shift = 5.421010862427522e-20, 0.0  # 2^-64
        hi = tl.zeros((BLOCK,), dtype=tl.float32)
        lo = tl.zeros((BLOCK,), dtype=tl.float32)
        top = tl.zeros((BLOCK,), dtype=tl.float32)
        for offset in tl.range(0, SPAN, BLOCK, num_stages=STAGES):
            index = start + offset + tl.arange(0, BLOCK)
            inside = index < end
            x = _load(base, index, inside, BITS)
            if CENTERED:
                x = tl.where(inside, x * unit - shift, 0.0)
                x = x * x
            else:
                top = tl.maximum(top, tl.abs(x))
                x = x * unit
            total = hi + x  # two-sum (launched without FP fusion): lo collects the rounding error of each addition
            back = total - hi
            lo += (hi - (total - back)) + (x - back)
            hi = total
        tl.store(PARTIAL + group * SPANS + span, tl.sum(hi.to(tl.float64) + lo.to(tl.float64), axis=0))
        if not CENTERED:
            tl.store(MAXES + group * SPANS + span, tl.max(top, axis=0))

    @triton.jit(do_not_specialize=["HW", "CHUNK", "SPANS", "LENGTH", "CHANNELS", "GROUP_CHANNELS"])
    def _apply_kernel(X, Y, SUMS, SQUARES, MAXES, W, B, HW, CHUNK, SPANS, LENGTH, CHANNELS, GROUP_CHANNELS, EPS, BLOCK: tl.constexpr,
                      STAGES: tl.constexpr, BITS: tl.constexpr, OUT_BITS: tl.constexpr):
        chunk, row = tl.program_id(0), tl.program_id(1).to(tl.int64)  # row: n * C + c
        channel = row % CHANNELS
        group = (row // CHANNELS) * (CHANNELS // GROUP_CHANNELS) + channel // GROUP_CHANNELS
        mean, mean32, unit, e = _group_stats(SUMS, MAXES, group, SPANS, LENGTH)
        power = ((1023 + e).to(tl.int64) << 52).to(tl.float64, bitcast=True)  # 2^e, exact (|e| <= 120)
        error = mean - mean32.to(tl.float64)
        variance = tl.maximum(_group_total(SQUARES, group, SPANS) / LENGTH.to(tl.float64) * power * power - error * error, 0.0)
        scale = (tl.load(W + channel).to(tl.float64) / tl.sqrt(variance + EPS) * power).to(tl.float32)  # FP64 IEEE division and root
        shift, centre = tl.load(B + channel), mean32 * unit
        start = chunk.to(tl.int64) * CHUNK
        end = tl.minimum(start + CHUNK, HW)
        base = row * HW
        for offset in tl.range(0, CHUNK, BLOCK, num_stages=STAGES):
            index = start + offset + tl.arange(0, BLOCK)
            inside = index < end
            x = _load(X + base, index, inside, BITS)
            y = (x * unit - centre) * scale + shift
            y = y * tl.div_rn(1.0, 1.0 + libdevice.exp(-y))  # PyTorch's FP32 sigmoid, then the product
            tl.store(Y + base + index, fp32_to_bf16_bits(y) if OUT_BITS else y.to(Y.dtype.element_ty), inside)


@functools.cache
def _sms(index):
    return torch.cuda.get_device_properties(index).multi_processor_count


def _launch(source, target, weight, bias, batch, channels, group_channels, hw, eps, bits, out_bits, config, sms):
    block, warps, stages, per_sm = config
    groups, length = batch * (channels // group_channels), group_channels * hw
    spans = max(1, min(MAX_SPANS, triton.cdiv(length, block), per_sm * sms // groups))
    span = triton.cdiv(triton.cdiv(length, spans), block) * block
    spans = triton.cdiv(length, span)
    sums = torch.empty((3, groups, spans), dtype=torch.float64, device=source.device)  # sums, squares, maxima (FP32 in the third)
    maxes = sums[2].view(torch.float32)
    for partial, centered in ((sums[0], False), (sums[1], True)):
        _sums_kernel[(spans, groups)](source, partial, sums[0], maxes, length, span, spans, centered, block, stages, bits, num_warps=warps,
                                      enable_fp_fusion=False)
    chunks = max(1, min(triton.cdiv(hw, block), per_sm * sms // (batch * channels)))
    chunk = triton.cdiv(triton.cdiv(hw, chunks), block) * block
    _apply_kernel[(triton.cdiv(hw, chunk), batch * channels)](source, target, sums[0], sums[1], maxes, weight, bias, hw, chunk, spans, length,
                                                              channels, group_channels, eps, block, stages, bits, out_bits, num_warps=warps)


def eligible(x, num_groups, dtype):
    """Whether the kernels run silu(group_norm(x)) to dtype: contiguous NCHW BF16 or FP16 x of MIN_BYTES or more and output, whole groups."""
    return (triton is not None and x.dim() == 4 and x.is_contiguous() and x.numel() * x.element_size() >= MIN_BYTES and x.shape[1] % num_groups == 0
            and dtype in (torch.bfloat16, torch.float16) and dispatch.cuda_inference(x, dtypes=(torch.bfloat16, torch.float16), capability_floor=(7, 5)))


def group_norm_silu(x, num_groups, weight, bias, eps, dtype):
    """silu(group_norm(x.float(), num_groups, weight, bias, eps)) rounded to dtype, x borrowed (not modified); eligible(x, ...) first."""
    batch, channels, height, width = x.shape
    hw, group_channels = height * width, channels // num_groups
    out = torch.empty(x.shape, dtype=dtype, device=x.device)
    bits, out_bits = x.dtype == torch.bfloat16, dtype == torch.bfloat16
    source, target = x.view(torch.int16) if bits else x, out.view(torch.int16) if out_bits else out
    weight, bias = weight.to(device=x.device, dtype=torch.float32), bias.to(device=x.device, dtype=torch.float32)
    sms = _sms(x.device.index)
    size = x.numel() * x.element_size()

    def scratch():  # this call's x (read only), or its first sample's first channels (contiguous), and an output of its own
        part = source if size <= _TUNE_MAX_BYTES else source[:1]
        if part.numel() * part.element_size() > _TUNE_MAX_BYTES:
            part = part[:, :max(1, _TUNE_MAX_BYTES // (part.numel() * part.element_size() // num_groups)) * group_channels]
        sample = torch.empty(part.shape, dtype=target.dtype, device=x.device)
        return lambda config: _launch(part, sample, weight, bias, part.shape[0], part.shape[1], group_channels, hw, eps, bits, out_bits, config, sms)

    config = _TUNER.config("group_norm_silu", f"{group_channels}|{x.dtype}|{dtype}", x.device, _CANDIDATES, scratch, size >= _TUNE_MIN_BYTES)
    _launch(source, target, weight, bias, batch, channels, group_channels, hw, eps, bits, out_bits, tuple(config), sms)
    return out


if triton is not None:
    _IMAGE_TUNER = LaunchTuner("wan2gp_image_norm_strips", 5, "VAE")

    @triton.jit(do_not_specialize=["H", "W", "SN", "SH", "SPAN", "SPANS"])
    def _channel_sums_kernel(X, PARTIAL, SUMS, MAXES, H, W, SN, SH, SPAN, SPANS,
                             C: tl.constexpr, GC: tl.constexpr, TILE_C: tl.constexpr, CENTERED: tl.constexpr,
                             ROWS: tl.constexpr, BITS: tl.constexpr):
        # Adjacent groups share each coalesced channel load instead of independently striding over the image.
        span, tile = tl.program_id(0), tl.program_id(1).to(tl.int64)
        batch, first = tile // (C // TILE_C), tile % (C // TILE_C) * TILE_C
        groups = batch * (C // GC) + first // GC + tl.arange(0, TILE_C // GC)
        if CENTERED:
            s = tl.arange(0, 128)
            mean = tl.sum(tl.load(SUMS + groups[:, None] * SPANS + s[None, :], s[None, :] < SPANS, 0.0), 1)
            mean = mean * 18446744073709551616.0 / (GC * H * W).to(tl.float64)
            mean32 = mean.to(tl.float32)
            bound = tl.max(tl.load(MAXES + groups[:, None] * SPANS + s[None, :], s[None, :] < SPANS, 0.0), 1).to(tl.float64)
            bound += tl.abs(mean32).to(tl.float64)
            e = tl.minimum(tl.maximum(((bound.to(tl.int64, bitcast=True) >> 52) & 0x7FF) - 1022, -120), 120).to(tl.int32)
            unit = ((127 - e) << 23).to(tl.float32, bitcast=True)
            shift = mean32 * unit
        else:
            unit = tl.full((TILE_C // GC,), 5.421010862427522e-20, tl.float32)
            shift = tl.full((TILE_C // GC,), 0.0, tl.float32)
        hi = tl.zeros((ROWS, TILE_C // GC, GC), tl.float32)
        lo = tl.zeros((ROWS, TILE_C // GC, GC), tl.float32)
        top = tl.zeros((ROWS, TILE_C // GC, GC), tl.float32)
        start = span.to(tl.int64) * SPAN
        end = tl.minimum(start + SPAN, H * W)
        for offset in tl.range(0, SPAN, ROWS):
            pixel = start + offset + tl.arange(0, ROWS)
            channel = first + tl.arange(0, TILE_C)
            address = batch * SN + pixel[:, None] // W * SH + pixel[:, None] % W * C + channel[None, :]
            x = _load(X, address, pixel[:, None] < end, BITS).reshape(ROWS, TILE_C // GC, GC)
            if CENTERED:
                x = tl.where((pixel < end)[:, None, None], x * unit[None, :, None] - shift[None, :, None], 0.0)
                x = x * x
            else:
                top = tl.maximum(top, tl.abs(x))
                x = x * unit[None, :, None]
            total = hi + x
            back = total - hi
            lo += (hi - (total - back)) + (x - back)
            hi = total
        sums = tl.sum(tl.sum(hi.to(tl.float64) + lo.to(tl.float64), 2), 0)
        tl.store(PARTIAL + groups * SPANS + span, sums)
        if not CENTERED:
            tl.store(MAXES + groups * SPANS + span, tl.max(tl.max(top, 2), 0))

    @triton.jit(do_not_specialize=["LENGTH", "SPAN", "SPANS", "C", "GC", "H", "W", "SN", "SC", "SH", "SW"])
    def _strided_sums_kernel(X, PARTIAL, SUMS, MAXES, LENGTH, SPAN, SPANS, C, GC, H, W, SN, SC, SH, SW,
                             CENTERED: tl.constexpr, CL: tl.constexpr, BLOCK: tl.constexpr, BITS: tl.constexpr):
        span, group = tl.program_id(0), tl.program_id(1).to(tl.int64)
        batch, channel = group // (C // GC), group % (C // GC) * GC
        start = span.to(tl.int64) * SPAN
        end = tl.minimum(start + SPAN, LENGTH)
        if CENTERED:
            _, mean32, unit, _ = _group_stats(SUMS, MAXES, group, SPANS, LENGTH)
            shift = mean32 * unit
        else:
            unit, shift = 5.421010862427522e-20, 0.0
        hi = tl.zeros((BLOCK,), tl.float32)
        lo = tl.zeros((BLOCK,), tl.float32)
        top = tl.zeros((BLOCK,), tl.float32)
        for offset in tl.range(0, SPAN, BLOCK):
            index = start + offset + tl.arange(0, BLOCK)
            inside = index < end
            if CL:
                c, pixel = index % GC, index // GC
            else:
                c, pixel = index // (H * W), index % (H * W)
            address = batch * SN + (channel + c) * SC + pixel // W * SH + pixel % W * SW
            x = _load(X, address, inside, BITS)
            if CENTERED:
                x = tl.where(inside, x * unit - shift, 0.0)
                x = x * x
            else:
                top = tl.maximum(top, tl.abs(x))
                x = x * unit
            total = hi + x
            back = total - hi
            lo += (hi - (total - back)) + (x - back)
            hi = total
        tl.store(PARTIAL + group * SPANS + span, tl.sum(hi.to(tl.float64) + lo.to(tl.float64), 0))
        if not CENTERED:
            tl.store(MAXES + group * SPANS + span, tl.max(top, 0))

    @triton.jit(do_not_specialize=["SPANS", "LENGTH", "C", "GC"])
    def _affine_kernel(SUMS, SQUARES, MAXES, W, B, A, SPANS, LENGTH, C, GC, EPS):
        row = tl.program_id(0).to(tl.int64)
        channel = row % C
        group = row // C * (C // GC) + channel // GC
        mean, mean32, unit, e = _group_stats(SUMS, MAXES, group, SPANS, LENGTH)
        power = ((1023 + e).to(tl.int64) << 52).to(tl.float64, bitcast=True)
        error = mean - mean32.to(tl.float64)
        variance = tl.maximum(_group_total(SQUARES, group, SPANS) / LENGTH.to(tl.float64) * power * power - error * error, 0.0)
        scale = (tl.load(W + channel).to(tl.float64) / tl.sqrt(variance + EPS) * power).to(tl.float32)
        tl.store(A + row * 4, unit)
        tl.store(A + row * 4 + 1, mean32 * unit)
        tl.store(A + row * 4 + 2, scale)
        tl.store(A + row * 4 + 3, tl.load(B + channel))

    @triton.jit(do_not_specialize=["SIZE", "C", "W", "ROWS", "FIRST", "SN", "SC", "SH", "SW"])
    def _rows_kernel(X, Y, A, SIZE, C, W, ROWS, FIRST, SN, SC, SH, SW,
                     BITS: tl.constexpr, OUT_BITS: tl.constexpr, BLOCK: tl.constexpr):
        index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        valid = index < SIZE
        channel, pixel = index % C, index // C
        col, row = pixel % W, pixel // W
        batch, row = row // ROWS, row % ROWS + FIRST
        affine = A + (batch * C + channel) * 4
        unit = tl.load(affine, valid, 0)
        centre = tl.load(affine + 1, valid, 0)
        scale = tl.load(affine + 2, valid, 0)
        shift = tl.load(affine + 3, valid, 0)
        x = _load(X, batch * SN + channel * SC + row * SH + col * SW, valid, BITS)
        y = (x * unit - centre) * scale + shift
        y = y * tl.div_rn(1.0, 1.0 + libdevice.exp(-y))
        tl.store(Y + index, fp32_to_bf16_bits(y) if OUT_BITS else y.to(Y.dtype.element_ty), valid)

    @triton.jit(do_not_specialize=["SIZE", "W", "FIRST", "SN", "SH"])
    def _channel_rows_kernel(X, Y, A, SIZE, W, FIRST, SN, SH, C: tl.constexpr, GAPPED: tl.constexpr,
                             BITS: tl.constexpr, OUT_BITS: tl.constexpr, BLOCK: tl.constexpr):
        # NHWC rows: contiguous accesses, with only a row-gap correction for the reused hidden/residual workspace.
        # C is the model's channel width; image sizes, batch count and strides remain runtime values.
        batch = tl.program_id(1).to(tl.int64)
        index = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        valid = index < SIZE
        address = batch * SN + FIRST * SH + index
        if GAPPED:
            address += index // (W * C) * (SH - W * C)
        affine = A + (batch * C + index % C) * 4
        unit = tl.load(affine, valid, 0)
        centre = tl.load(affine + 1, valid, 0)
        scale = tl.load(affine + 2, valid, 0)
        shift = tl.load(affine + 3, valid, 0)
        x = _load(X, address, valid, BITS)
        y = (x * unit - centre) * scale + shift
        y = y * tl.div_rn(1.0, 1.0 + libdevice.exp(-y))
        tl.store(Y + batch * SIZE + index, fp32_to_bf16_bits(y) if OUT_BITS else y.to(Y.dtype.element_ty), valid)



def strips_eligible(x, num_groups, dtype):
    """Strided 16-bit image normalization; includes row-interleaved decoder workspaces."""
    return (triton is not None and x.ndim == 4 and x.numel() * x.element_size() >= MIN_BYTES
            and x.shape[1] % num_groups == 0 and all(s > 0 for s in x.stride())
            and dtype in (torch.bfloat16, torch.float16)
            and dispatch.cuda_inference(x, dtypes=(torch.bfloat16, torch.float16), capability_floor=(7, 5)))


def _image_affine_launch(x, num_groups, weight, bias, eps, config):
    n, c, h, w = x.shape
    gc = c // num_groups
    groups, length = n * num_groups, gc * h * w
    block, warps, per_sm = config
    coalesced = x.stride(1) == 1 and x.stride(3) == c and c % 32 == 0 and 32 % gc == 0
    channel_block = 32
    span_length, span_block, tiles = (h*w, block//channel_block, n*(c//channel_block)) if coalesced else (length, block, groups)
    spans = max(1, min(MAX_SPANS, triton.cdiv(span_length, span_block), per_sm * _sms(x.device.index) // tiles))
    span = triton.cdiv(triton.cdiv(span_length, spans), span_block) * span_block
    spans = triton.cdiv(span_length, span)
    sums = torch.empty((3, groups, spans), dtype=torch.float64, device=x.device)
    maxes = sums[2].view(torch.float32)
    bits = x.dtype == torch.bfloat16
    source = x.view(torch.int16) if bits else x
    for partial, centered in ((sums[0], False), (sums[1], True)):
        if coalesced:
            _channel_sums_kernel[(spans, tiles)](source, partial, sums[0], maxes, h, w, x.stride(0), x.stride(2), span, spans,
                                                c, gc, channel_block, centered, block//channel_block, bits, num_warps=warps, enable_fp_fusion=False)
        elif x.is_contiguous():
            _sums_kernel[(spans, groups)](source, partial, sums[0], maxes, length, span, spans, centered, block, 3, bits,
                                          num_warps=warps, enable_fp_fusion=False)
        else:
            _strided_sums_kernel[(spans, groups)](source, partial, sums[0], maxes, length, span, spans, c, gc, h, w, *x.stride(),
                                                 centered, x.stride(1) == 1, block, bits, num_warps=warps, enable_fp_fusion=False)
    affine = torch.empty((n, c, 4), dtype=torch.float32, device=x.device)
    _affine_kernel[(n * c,)](sums[0], sums[1], maxes, weight.to(x.device, torch.float32), bias.to(x.device, torch.float32),
                            affine, spans, length, c, gc, eps, num_warps=4)
    return affine


def image_affine(x, num_groups, weight, bias, eps):
    """Whole-image GroupNorm statistics in a compact (N,C,4) affine; x is borrowed."""
    def scratch():
        rows = max(1, (32 << 20) // (x.shape[1] * x.shape[3] * x.element_size()))
        part = x[:1, :, :rows].clone(memory_format=torch.preserve_format)
        return lambda config: _image_affine_launch(part, num_groups, weight, bias, eps, config)
    config = _IMAGE_TUNER.config("image_affine", f"{x.dtype}|{x.shape[1]}|{x.shape[1]//num_groups}|{x.stride(1)==1}|{x.is_contiguous()}", x.device,
                                 ((1024,4,4), (2048,8,4), (512,4,8)), scratch, x.numel()*x.element_size() >= MIN_BYTES)
    return _image_affine_launch(x, num_groups, weight, bias, eps, config)


def _image_rows_launch(x, affine, first, out, config):
    n, c, rows, w = out.shape
    block, warps = config
    bits, out_bits = x.dtype == torch.bfloat16, out.dtype == torch.bfloat16
    source, target = x.view(torch.int16) if bits else x, out.view(torch.int16) if out_bits else out
    if x.stride(1) == 1 and x.stride(3) == c:
        size = c * rows * w
        _channel_rows_kernel[(triton.cdiv(size, block), n)](source, target, affine, size, w, first, x.stride(0), x.stride(2),
                                                          c, x.stride(2) != w * c, bits, out_bits, block, num_warps=warps)
    else:
        _rows_kernel[(triton.cdiv(out.numel(), block),)](source, target, affine, out.numel(), c, w, rows, first, *x.stride(),
                                                       bits, out_bits, block, num_warps=warps)


def image_rows(x, affine, first, last, dtype):
    """Apply the saved global normalization and SiLU to rows [first,last), directly in channels-last layout."""
    n, c, _, w = x.shape
    out = torch.empty((n, c, last - first, w), dtype=dtype, device=x.device, memory_format=torch.channels_last)
    def scratch():
        rows = min(last-first, max(1, (8 << 20) // (c * w * 2)))
        original = x[:1, :, first:first+rows]
        if x.stride(1) == 1 and x.stride(3) == c:
            part = torch.empty_strided(original.shape, (rows*x.stride(2), 1, x.stride(2), c), dtype=x.dtype, device=x.device)
            part.copy_(original)  # retain the real row-gap specialization without retaining a full image
        else:
            part = original.clone(memory_format=torch.preserve_format)
        sample = torch.empty(part.shape, dtype=dtype, device=x.device, memory_format=torch.channels_last)
        return lambda config: _image_rows_launch(part, affine[:1], 0, sample, config)
    config = _IMAGE_TUNER.config("image_rows", f"{x.dtype}|{dtype}|{c}|{x.stride(1)==1}|{x.stride(2)!=w*c}", x.device,
                                 ((1024,4), (512,4), (2048,8)), scratch, out.numel()*out.element_size() >= MIN_BYTES)
    _image_rows_launch(x, affine, first, out, config)
    return out


if triton is not None:
    from shared.kernels.triton_compilation_log import install_triton_compilation_logger  # noqa: E402

    install_triton_compilation_logger()
