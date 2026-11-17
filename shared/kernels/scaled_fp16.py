# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""BF16 convolutions on FP16 tensor cores, for GPUs with FP16 but no BF16 tensor cores (validated on Turing, compute capability 7.5),
where PyTorch has no fast BF16 convolution (cuDNN has no BF16 kernels there: `slow_conv_dilated3d`, i.e. vol2col and a CUDA-core GEMM).

A BF16 value converts exactly to FP16 (8 significant bits into 11) while its magnitude stays in FP16's normal range. The weight is scaled
by a power of two (largest |w| in [0.5, 1)); the input by the power of two that keeps every input below 2^15 and every possible output
below 2^15 (largest |x| times the largest L1 norm of an output channel's weights), so nothing overflows; cuDNN accumulates in FP32; the
FP16 result is scaled back, the bias added in FP32 and rounded once to BF16, in place. Precision: values that end up below FP16's normal
range lose bits, and the result rounds to FP16 (11 bits) before BF16 (8 bits): LTX-2.5's VAE decoder, encoder and spatial upsampler
measured within +-3% of the PyTorch BF16 route's error against FP32, 4.8-6.8x faster. Used with every kernel precision (user decision,
2026-10-21: a few % more error than PyTorch's route is worth it where PyTorch has no fast BF16 convolution), an exception to the strict rule.

install_globally(device), called once by wgp.py: on an eligible GPU the stock _conv_forward of Conv1d / Conv2d / Conv3d (which
subclasses with causal paddings and caches call through super()) computes every BF16 CUDA call of the process this way; other dtypes,
devices, autograd, compilation and graph capture keep PyTorch's route. install(module, device) does the same for the layers of one
module (tests, standalone use); conv2d() is the functional entry for a layer computing a convolution itself."""
import functools

import torch
import torch.nn.functional as F

from .dispatch import choose, cuda_inference

try:
    import triton
    import triton.language as tl
    from .bf16_bits import bf16_bits_to_fp32, fp32_to_bf16_bits
except ImportError:
    triton = None

_BLOCK = 1024
_SHIFT_LIMIT = 120  # powers of two applied in two halves: exact in FP32 for every shift of input and weight
_CONVOLUTIONS = {torch.nn.Conv1d: F.conv1d, torch.nn.Conv2d: F.conv2d, torch.nn.Conv3d: F.conv3d}


@functools.cache
def _supported(index):
    # GeForce GTX 16 (TU116/TU117) is also 7.5 but has no tensor cores, and FP16 convolutions fail there: black images (WanGP issue 2637,
    # and the GTX 16 exception of other diffusion apps). Its BF16 convolutions stay on PyTorch's route.
    return torch.cuda.get_device_capability(index) == (7, 5) and "GTX 16" not in torch.cuda.get_device_name(index)


def supported(device) -> bool:
    """Validated on compute capability 7.5 only (Volta has the same FP16 tensor cores, untested); needs Triton for the conversions."""
    device = torch.device(device)
    return triton is not None and device.type == "cuda" and torch.version.hip is None and torch.cuda.is_available() and _supported(device.index)


def _functional(layer):
    """F.convNd of a Conv1d/2d/3d layer or of a subclass that keeps the stock _conv_forward (causal paddings and caches around it)."""
    for cls, fn in _CONVOLUTIONS.items():
        if isinstance(layer, cls) and type(layer)._conv_forward is cls._conv_forward:
            return fn
    return None


def install(module, device) -> int:
    """Route the BF16 calls of module's Conv1d/2d/3d layers (and subclasses with the stock _conv_forward) on an eligible GPU
    (supported(device)); returns the number of layers."""
    if not supported(device):
        return 0
    count = 0
    for layer in module.modules():
        if _functional(layer) is not None and "_conv_forward" not in layer.__dict__:
            layer._conv_forward = functools.partial(_conv_forward, layer, layer._conv_forward)
            count += 1
    return count


def _eligible(input, weight, bias):
    """The input to route (a BF16 autocast's FP32 input cast first, as autocast would), or None: BF16 CUDA input, weight and bias,
    inference, no compilation or capture, on a GPU of the route (also for the functional entries, which are not installed per device)."""
    if not input.is_cuda or not _supported(input.device.index):
        return None
    if (input.dtype == torch.float32 and weight.dtype == torch.bfloat16 and input.is_cuda and torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") == torch.bfloat16):
        input = input.to(torch.bfloat16)  # what the caller's BF16 autocast does before the convolution (FP32 norm outputs under autocast)
    if (not cuda_inference(input, capability_floor=(7, 5)) or weight.dtype != torch.bfloat16  # BF16 handled as bits
            or type(weight) not in (torch.Tensor, torch.nn.Parameter) or weight.device != input.device
            or (bias is not None and bias.dtype != torch.bfloat16) or torch.cuda.is_current_stream_capturing()):
        return None
    return input


def install_globally(device) -> bool:
    """Route the BF16 calls of every Conv1d / Conv2d / Conv3d of the process (their stock _conv_forward, which subclasses call through
    super()) on an eligible GPU (supported(device)); once per process, from wgp.py. False (nothing patched) elsewhere."""
    if not supported(device):
        return False
    for cls in _CONVOLUTIONS:
        if "_scaled_fp16_stock" not in cls.__dict__:
            cls._scaled_fp16_stock = cls._conv_forward
            cls._conv_forward = _global_conv_forward
    return True


def _route(layer, input, weight, bias):
    """layer's convolution on the route for an eligible call, else None."""
    routed = _eligible(input, weight, bias)
    if routed is None:
        return None
    choose("BF16 Convolution", owned_triton=True)
    fn = next(fn for cls, fn in _CONVOLUTIONS.items() if isinstance(layer, cls))
    return _scaled(fn, routed, weight, bias, layer.stride, layer.padding, layer.dilation, layer.groups, layer.padding_mode,
                   layer._reversed_padding_repeated_twice, layer, "_scaled_fp16_stats")


def _global_conv_forward(layer, input, weight, bias):
    out = _route(layer, input, weight, bias)
    return type(layer)._scaled_fp16_stock(layer, input, weight, bias) if out is None else out


def _conv_forward(layer, original, input, weight, bias):
    out = _route(layer, input, weight, bias)
    return original(input, weight, bias) if out is None else out


def conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, *, owner, key):
    """F.conv2d, on the scaled-FP16 route where _conv_forward would take it (a layer computing a 2-D convolution itself, for example
    with a slice of a 3-D weight); owner.__dict__[key] caches the bound of that weight (one measurement, as for a layer)."""
    routed = _eligible(input, weight, bias)
    if routed is None:
        return F.conv2d(input, weight, bias, stride, padding, dilation, groups)
    choose("BF16 Convolution", owned_triton=True)
    return _scaled(F.conv2d, routed, weight, bias, stride, padding, dilation, groups, "zeros", None, owner, key)


def _weight_shift(owner, weight, key="_scaled_fp16_stats"):
    """(power of two applied to the weight, largest L1 norm of an output channel's scaled weights): measured once per layer (or per
    owner and key), at its first call on the GPU (one synchronization)."""
    stats = owner.__dict__.get(key)
    if stats is None:
        magnitude = weight.abs()
        _, exponent = torch.frexp(magnitude.amax().float())
        l1 = magnitude.float().flatten(1).sum(1).amax()
        shift = max(-_SHIFT_LIMIT, min(_SHIFT_LIMIT, -int(exponent.item())))
        stats = (shift, float(l1.item()) * 2.0 ** shift * (1 + 2.0 ** -10))  # bound rounded up
        owner.__dict__[key] = stats
    return stats


def convolution(layer, input, weight, bias):
    """layer's convolution (its stride, padding, padding mode, dilation, groups) of BF16 CUDA input, weight and bias on FP16 tensor cores."""
    return _scaled(_functional(layer), input, weight, bias, layer.stride, layer.padding, layer.dilation, layer.groups, layer.padding_mode,
                   layer._reversed_padding_repeated_twice, layer, "_scaled_fp16_stats")


def _scaled(fn, input, weight, bias, stride, padding, dilation, groups, padding_mode, reversed_padding, owner, key):
    """fn (F.conv1d/2d/3d) of BF16 CUDA input, weight and bias on FP16 tensor cores; owner.__dict__[key] caches the bound of weight."""
    weight_shift, l1 = _weight_shift(owner, weight, key)
    x = input.contiguous()
    low, high = torch.aminmax(x)
    x16 = torch.empty(x.shape, dtype=torch.float16, device=x.device)
    shift = torch.empty(1, dtype=torch.int32, device=x.device)
    _to_fp16_kernel[(triton.cdiv(x.numel(), _BLOCK),)](x.view(torch.int16), x16, x.numel(), 0, low.view(torch.int16), high.view(torch.int16), l1,
                                                        shift, True, _BLOCK)
    x = None
    weight = weight.contiguous()
    w16 = torch.empty(weight.shape, dtype=torch.float16, device=weight.device)
    _to_fp16_kernel[(triton.cdiv(weight.numel(), _BLOCK),)](weight.view(torch.int16), w16, weight.numel(), weight_shift, None, None, 0.0,
                                                             None, False, _BLOCK)
    if padding_mode != "zeros":
        x16, padding = F.pad(x16, reversed_padding, mode=padding_mode), 0
    with torch.autocast(device_type="cuda", enabled=False):  # a caller's BF16 autocast would turn the FP16 convolution back into BF16
        y = fn(x16, w16, None, stride, padding, dilation, groups).contiguous()
    x16 = w16 = None
    channels, spatial = y.shape[1], y[0, 0].numel()
    _to_bf16_kernel[(triton.cdiv(y.numel(), _BLOCK),)](y, y.view(torch.int16), None if bias is None else bias.view(torch.int16), shift,
                                                        weight_shift, y.numel(), channels, spatial, bias is not None, _BLOCK)
    return y.view(torch.bfloat16)


if triton is not None:
    @triton.jit
    def _power_of_two(shift):
        return ((shift + 127) << 23).to(tl.float32, bitcast=True)

    @triton.jit
    def _exponent(x):
        return ((x.to(tl.uint32, bitcast=True) >> 23) & 0xFF).to(tl.int32) - 127

    @triton.jit(do_not_specialize=["n", "fixed_shift", "l1"])
    def _to_fp16_kernel(X, Y, n, fixed_shift, LOW, HIGH, l1, SHIFT, DYNAMIC: tl.constexpr, BLOCK: tl.constexpr):
        # BF16 bits -> FP16 times a power of two: fixed (a weight), or from the input's extremes and the weight's L1 bound (stored in SHIFT)
        offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        if DYNAMIC:
            largest = tl.maximum(tl.abs(bf16_bits_to_fp32(tl.load(LOW))), tl.abs(bf16_bits_to_fp32(tl.load(HIGH))))
            bound = largest * l1
            # 14 - exponent: inputs and outputs below 2^15; zero, infinite and NaN inputs keep a shift of 0
            shift = 14 - tl.maximum(_exponent(bound), _exponent(largest))
            shift = tl.where((largest > 0) & (bound < float("inf")), tl.minimum(tl.maximum(shift, -120), 120), 0)
            tl.store(SHIFT, shift, mask=tl.program_id(0) == 0)
        else:
            shift = fixed_shift
        x = bf16_bits_to_fp32(tl.load(X + offsets, mask, 0)) * _power_of_two(shift)
        tl.store(Y + offsets, x.to(tl.float16), mask)

    @triton.jit(do_not_specialize=["weight_shift", "n", "channels", "spatial"])
    def _to_bf16_kernel(Y16, Y_BITS, BIAS, SHIFT, weight_shift, n, channels, spatial, HAS_BIAS: tl.constexpr, BLOCK: tl.constexpr):
        # FP16 result scaled back (two exact halves of the total power of two), bias added in FP32, rounded once to BF16 in place
        offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n
        total = -(tl.load(SHIFT) + weight_shift)
        half = total // 2
        y = tl.load(Y16 + offsets, mask, 0).to(tl.float32) * _power_of_two(half) * _power_of_two(total - half)
        if HAS_BIAS:
            y += bf16_bits_to_fp32(tl.load(BIAS + (offsets // spatial) % channels, mask, 0))
        tl.store(Y_BITS + offsets, fp32_to_bf16_bits(y), mask)

    # Register after definitions to preserve Triton's line-number-sensitive cache keys.
    from shared.kernels.triton_compilation_log import install_triton_compilation_logger
    install_triton_compilation_logger()
