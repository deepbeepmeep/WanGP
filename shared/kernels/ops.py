# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Activation operations shared by denoisers.

Owned inputs use x_list through the dispatch boundary. In-place operations borrow
their destination; residuals, weights and conditioning are always borrowed. Math
contracts are explicit: split-half RMS/RoPE rounds after normalization, matrix
RoPE keeps FP32 until its final cast, and complex RoPE rounds before rotation.
Whole-head normalization and compact rotary layouts remain model supplied.
"""
import collections
import weakref

import torch
import torch.nn.functional as F
from mmgp import cuda_graphs, offload, quant_router
from . import int8_backend, kernel_policy, native_ops
from .dispatch import choose, cuda_inference, device_supported, packed_fma, take

try:
    from . import denoiser_triton as _triton
except (ImportError, OSError, RuntimeError):
    _triton = None
try:  # the same fusions reading and writing BF16 as bits, for GPUs below the BF16 floor (compute capability 7.5)
    from . import denoiser_bits_triton as _bits_triton
except (ImportError, OSError, RuntimeError):
    _bits_triton = None

try:  # the split-half norm + RoPE fused with the staged SageAttention 2 quantization of q and k
    from . import qk_quant_triton as _qk_quant
except (ImportError, OSError, RuntimeError):
    _qk_quant = None

try:
    from . import adaln_triton as _adaln_triton
except (ImportError, OSError):
    _adaln_triton = None

try:  # a LoRA's output pass, alone or with the SwiGLU that reads its output
    from . import lora_triton as _lora_triton
except (ImportError, OSError, RuntimeError):
    _lora_triton = None
_LORA_DTYPES = (torch.bfloat16, torch.float16)

try:
    from comfy_kitchen import rms_rope_split_half_ as _kitchen_rms_rope
except (ImportError, OSError, RuntimeError):
    _kitchen_rms_rope = None
except ValueError as exc:
    if not str(exc).startswith("infer_schema(func):"):
        raise
    _kitchen_rms_rope = None


def can_linear(layer, x, *, dtypes=(torch.bfloat16,), act=None):
    """Whether layer(x) can run through the INT8 backend's fused kernel, with the input operation act ('gelu' needs WGPKernels)."""
    return int8_backend.can_fuse_linear(layer, x, dtypes=dtypes, act=act)


def linear(layer, x_list, **fusion):
    """Use the shared qtype implementation and normal module hooks."""
    if not fusion:
        return layer(take(x_list))
    return int8_backend.linear_with_fusion(layer, take(x_list), **fusion)


def _lora_output(out, y, factor, scale, rank_first):
    if not (_lora_triton is not None and cuda_inference(out, dtypes=_LORA_DTYPES) and _lora_triton.supported(out, y, factor, rank_first)
            and choose("LoRA Output", owned_triton=True)):
        return False
    _lora_triton.add_(out, y, factor, scale, rank_first)
    return True


offload.register_lora_output(_lora_output)
if _triton is not None:  # the row stores of CUDA graph loops over row ranges (offload's RowSource preparation)
    from . import row_loop_triton as _row_loop_triton
    offload.register_row_store_at(_row_loop_triton.store_rows_at)
    offload.register_row_copy_at(_row_loop_triton.copy_rows_at)


def project_many(modules, x_list):
    x = take(x_list)
    if (not torch.compiler.is_compiling() and x.is_contiguous()
            and all(can_linear(module, x) for module in modules)):
        outputs = int8_backend.linear_multi(modules, x.reshape(-1, x.shape[-1]))
        return tuple(out.reshape(*x.shape[:-1], out.shape[-1]) for out in outputs)
    return tuple(module(x) for module in modules)


def _bf16_bits_eligible(x, *, approximate=False):
    """A BF16 activation on a CUDA GPU that runs FP16 kernels but is below the BF16 floor (Turing): the bits variants apply."""
    return (_bits_triton is not None and x.dtype == torch.bfloat16
            and cuda_inference(x, capability_floor=(7, 5), approximate=approximate)
            and not device_supported(x.device, torch.bfloat16))


def _consecutive_segments(segments, total):
    """Segments (start, stop, row) that tile the rows in equal spans with rows 0, 1, 2...: one modulation row per batch row or frame."""
    span = segments[0][1] - segments[0][0] if segments else 0
    return (span > 0 and span * len(segments) == total
            and all(start == index * span and stop == start + span and row == index for index, (start, stop, row) in enumerate(segments)))


def norm_modulate(x_list, norm, shift, scale, segments, *, dtype=None):
    """norm(x), cast to dtype when given (an FP32 modulation after a 16-bit norm), then per segment (start, stop, row) of rows
    x * (1 + scale[row]) + shift[row] with PyTorch's roundings in the modulated dtype. norm may be weightless. The Triton kernel is
    bit-identical to the eager path except on the packed-FMA architecture, where it counts as approximate."""
    x = take(x_list)
    weight = norm.weight
    if (dtype is None and weight is None and getattr(type(norm), "forward", None) is torch.nn.LayerNorm.forward and kernel_policy.allow_approximate()
            and _consecutive_segments(segments, x.shape[0])):
        # Faster approximate kernels allowed: a plain weightless LayerNorm (a subclass with its own forward, e.g. normalizing in another
        # dtype, keeps the exact path) with one modulation row per equal span of rows is one fused AdaLN call on (spans, rows, dim),
        # rounded once (one ulp apart from the exact path on a few elements per million).
        spans, rows = len(segments), segments[0][1] - segments[0][0]
        x_list, x = [x.view(spans, rows, x.shape[-1])], None
        return _layer_norm_modulate(x_list, shift[:spans].to(x_list[0].dtype).unsqueeze(1), scale[:spans].to(x_list[0].dtype).unsqueeze(1), norm.eps).view(spans * rows, -1)
    layout = x.is_contiguous() and (weight is None or weight.device == x.device and weight.dtype == x.dtype) and x.shape[-1] <= 8192
    fused = _triton is not None and cuda_inference(x, approximate=packed_fma(x.device.index)) and layout
    bits = (not fused and dtype is None and layout and _bf16_bits_eligible(x)  # bit-identical to the eager path
            and scale.dtype in (torch.float32, torch.bfloat16) and shift.dtype == scale.dtype)
    if choose("Segmented Norm Modulation", owned_triton=fused or bits):
        return _bits_triton.norm_modulate(x, norm, shift, scale, segments) if bits else _triton.norm_modulate(x, norm, shift, scale, segments, dtype)
    x = norm(x)
    if dtype is not None:
        x = x.to(dtype)
    shift, scale = shift.to(x.dtype), scale.to(x.dtype)
    for start, stop, row in segments:
        x[start:stop].mul_(1.0 + scale[row]).add_(shift[row])
    return x


def _fused_norm(norm, dtype):
    """(weight, eps, layer) of a norm shared.kernels.norm_quant_triton computes (RMSNorm; LayerNorm without bias, weightless or with
    LayerNorm's own forward), None for another."""
    if isinstance(norm, torch.nn.RMSNorm):
        return norm.weight, norm.eps if norm.eps is not None else torch.finfo(dtype).eps, False
    if isinstance(norm, torch.nn.LayerNorm) and norm.bias is None and (norm.weight is None or type(norm).forward is torch.nn.LayerNorm.forward):
        return norm.weight, norm.eps, True
    return None


def norm_modulate_rows(x, norm, shift, scale, segments, shape=None, dtype=None):
    """norm_modulate of rows x (rows, dim), borrowed (cast to dtype when given; shift / scale None: norm(x) alone), as an mmgp
    offload.RowSource of `shape` (x's by default) made by ranges of rows, for inputs only their projections read (an attention that makes
    them again by head group, an FFN chunk by chunk: attention_kit.extreme_low_vram()): never whole beside x, rows as from the whole
    (row-wise operations). Its INT8 / FP8 forms come in one pass from the residual where the norm is one shared.kernels.norm_quant_triton
    computes (RowSource.fill_prepared: the norm and modulation in FP32, rounded once)."""
    def fill(start, stop):
        if scale is None:
            rows = norm(x[start:stop])
        else:
            spans = [(max(first, start) - start, min(last, stop) - start, row) for first, last, row in segments if first < stop and last > start]
            rows = norm_modulate([x[start:stop]], norm, shift, scale, spans)
        return rows if dtype is None else rows.to(dtype)
    source = offload.RowSource(x.shape if shape is None else shape, x.dtype if dtype is None else dtype, x.device, fill)
    fused = _fused_norm(norm, x.dtype) if dtype in (None, x.dtype) else None
    if fused is not None:
        source.fill_prepared, source.fill_prepared_at = row_fills(x, fused[0], fused[1], scale, shift, segments, fused[2])
    return source


def row_fills(x, weight, eps, scale, shift, segments, layer=False):
    """shared.kernels.norm_quant_triton.row_fills (a RowSource's fill_prepared and fill_prepared_at) where Triton is installed, else (None, None)."""
    if _triton is None:
        return None, None
    from . import norm_quant_triton
    return norm_quant_triton.row_fills(x, weight, eps, scale, shift, segments, layer)


def rms_norm_(x, eps, weight=None):
    """x normalized in place over its last dimension by its RMS (times weight), in one Triton pass in FP32 rounded once (as precise as
    the norm then the weight in 16 bits, two roundings, or more); None where it cannot run (no Triton, a layout whose rows are not one
    stride apart, D above 16384): the caller keeps its own path."""
    width = x.shape[-1]
    if (_triton is None or not cuda_inference(x, dtypes=(torch.bfloat16, torch.float16)) or x.stride(-1) != 1 or width > 16384
            or weight is not None and (weight.device != x.device or weight.numel() != width)):
        return None
    try:
        rows = x.view(-1, width)
    except RuntimeError:
        return None
    if not choose("RMSNorm", owned_triton=True):
        return None
    from . import row_loop_triton
    row_loop_triton.rms_norm_rows_(rows, eps, None if weight is None else weight.contiguous())
    return x


def rms_norm_modulate_supported(rows, shift=None, scale=None):
    """Whether rms_norm_modulate runs for these rows (R, D) and modulation rows: Triton, a 16-bit CUDA tensor, D at most 16384."""
    return (_triton is not None and cuda_inference(rows, dtypes=(torch.bfloat16, torch.float16)) and rows.stride(-1) == 1 and rows.shape[-1] <= 16384
            and (shift is None or shift.stride(-1) == 1 and scale.stride(-1) == 1 and shift.device == rows.device)
            and bool(choose("RMSNorm + Modulation", owned_triton=True)))


def rms_norm_modulate(rows, eps, count, first=0, shift=None, scale=None, span=0, counter=None, step=0):
    """The weightless RMSNorm of `count` rows of rows (R, D) from first (+ counter * step, the counter on the device: a CUDA graph loop's),
    with shift / scale (segments, D) x * (1 + scale[s]) + shift[s] for the segment s = row // span: one Triton pass in FP32, rounded once
    (against FP64 at most the error of rms_norm then the modulation in 16 bits: tests/test_rms_norm_modulate_precision.py), a new tensor
    (count, D). Check rms_norm_modulate_supported first."""
    from . import row_loop_triton
    return row_loop_triton.rms_norm_modulate(rows, eps, count, first, shift, scale, span, counter, step)


def mean_square_scale_(rows, squares, eps, weight):
    """rows.mul_(squares.add(eps).rsqrt_()).mul_(weight) in one pass, bit-identical (an RMSNorm over all heads applied to a group of
    heads from statistics taken beforehand): rows (B, T, C), squares (B, T, 1) FP32, weight (C,)."""
    if (_triton is not None and cuda_inference(rows, dtypes=_LORA_DTYPES) and rows.ndim == 3 and rows.stride(2) == 1 and squares.dtype == torch.float32
            and squares.shape == (*rows.shape[:2], 1) and weight.shape == rows.shape[2:] and weight.is_contiguous() and choose("Mean-Square Scale", owned_triton=True)):
        return _triton.mean_square_scale_(rows, squares, eps, weight)
    return rows.mul_(squares.add(eps).rsqrt_()).mul_(weight)


def square_div_(x, count):
    """x.square_().div_(count) in one pass, bit-identical: x FP32 and contiguous (per-token mean squares from vector norms)."""
    if _triton is not None and cuda_inference(x, dtypes=(torch.float32,)) and x.is_contiguous() and choose("Square / Divide", owned_triton=True):
        return _triton.square_div_(x, count)
    return x.square_().div_(count)


def scale_shift(x_list, scale, shift=None, *, in_place=True, frames=False, fused_add=False):
    """Native dtype arithmetic; optionally broadcast modulation per frame."""
    x = take(x_list)
    broadcast = frames or (x.ndim == 3 and scale.ndim == 3 and scale.shape[1] in (1, x.shape[1]))
    layout = (broadcast and not fused_add and x.ndim == 3 and scale.ndim == 3 and shift is not None and scale.shape == shift.shape
              and x.is_contiguous() and all(t.dtype == x.dtype and t.device == x.device for t in (scale, shift))
              and scale.stride() == shift.stride() and scale.stride(2) == 1  # rows may be strided views (LTX: unbind of a (B, T, K, D) projection)
              and (scale.is_contiguous() or scale.stride(0) == scale.shape[1] * scale.stride(1))
              and scale.shape[0] == x.shape[0] and scale.shape[-1] == x.shape[-1]
              and scale.shape[1] > 0 and x.shape[1] % scale.shape[1] == 0
              and (scale.shape[1] == x.shape[1] and scale.is_contiguous() or scale.shape[0] * scale.shape[1] <= 65535))
    fused = layout and _triton is not None and cuda_inference(x, approximate=True)
    bits = layout and not fused and _bf16_bits_eligible(x, approximate=True)  # below the BF16 floor
    if choose("Frame Scale/Shift" if frames else "Scale/Shift", owned_triton=fused or bits):
        return (_bits_triton if bits else _triton).scale_shift(x, scale, shift, in_place)
    shape = x.shape
    if frames and scale.shape[1] != x.shape[1]:
        x = x.reshape(x.shape[0], scale.shape[1], -1, x.shape[-1])
        scale = scale.unsqueeze(2)
        shift = None if shift is None else shift.unsqueeze(2)
    if fused_add:
        return torch.addcmul(shift, 1 + scale, x, out=x if in_place else None).reshape(shape)
    x = x.mul_(1 + scale) if in_place else x * (1 + scale)
    if shift is not None:
        x = x.add_(shift) if in_place else x + shift
    return x.reshape(shape)


def norm_scale_shift(x_list, norm, scale, shift, *, in_place=True, fused_add=False):
    """Keep the supplied norm's reduction/affine contract, then modulate.

    Unlike fused AdaLN this retains the native norm reduction and its rounding.
    Residual inputs may be borrowed by passing [residual]; the norm allocates.
    """
    x = norm(take(x_list))
    x_list, x = [x], None
    return scale_shift(x_list, scale, shift, in_place=in_place, fused_add=fused_add)


def layer_norm_modulate(x_list, shift, scale, eps=1e-6, *, frames=False):
    """layer_norm(x) * (1 + scale) + shift over the last dimension; scale and shift broadcast over x, or, with frames, hold one row per
    group of tokens (batch, groups, dim) for x (batch, tokens, dim) with groups dividing the tokens (a modulation per latent frame)."""
    x = take(x_list)
    shape = x.shape
    if frames and scale.ndim == 3 and x.ndim == 3 and scale.shape[1] not in (1, x.shape[1]):
        groups = scale.shape[1]
        x = x.view(shape[0] * groups, shape[1] // groups, shape[2])
        scale, shift = scale.reshape(-1, 1, shape[2]), shift.reshape(-1, 1, shape[2])
    x_list, x = [x], None
    return _layer_norm_modulate(x_list, shift, scale, eps).view(shape)


def _layer_norm_modulate(x_list, shift, scale, eps):
    x = take(x_list)
    valid = (cuda_inference(x, dtypes=(torch.bfloat16, torch.float16), approximate=True)
             and x.ndim in (2, 3) and x.is_contiguous() and x.numel() > 0 and x.shape[-1] <= 8192
             and all(t.ndim > 0 and t.ndim <= x.ndim and t.dtype == x.dtype and t.device == x.device
                     and t.shape[-1] == x.shape[-1] and t.stride(-1) == 1
                     and all(a == 1 or a == b for a, b in zip(t.shape, x.shape[-t.ndim:])) for t in (shift, scale)))
    family = choose("AdaLN", wgpkernels=valid and native_ops.available(x.device),
                    triton=valid and _adaln_triton is not None)
    if family == "wgpkernels":
        return native_ops.implementation.adaln(x, scale, shift, eps)
    if family == "triton":
        return _adaln_triton.adaln(x, scale, shift, eps)
    x = F.layer_norm(x, (x.shape[-1],), eps=eps)
    # The normalized tensor is fresh: modulate it in place, unless a higher-precision modulation promotes the result.
    x_list, in_place, x = [x], scale.dtype == x.dtype, None
    return scale_shift(x_list, scale, shift, in_place=in_place)


def swiglu(x_list, *, packed=True, in_place=True):
    """Native SiLU rounding, followed by native multiplication, without concat.

    Packed input is gate|value. Separate projections pass [gate, value]; both
    are consumed. No approximate kernel is promoted to a new model contract.
    """
    if packed:
        x = take(x_list)
        if in_place and _lora_triton is not None and x.ndim == 2 and x.stride(1) == 1 and cuda_inference(x, dtypes=_LORA_DTYPES) and choose("SwiGLU", owned_triton=True):
            return _lora_triton.swiglu_(x)
        gate, value = x.chunk(2, -1)
    else:
        gate, value = x_list
        x_list.clear()
    return F.silu(gate, inplace=in_place).mul_(value)


def _swiglu_of_linear(layer, x):
    """silu(gate) * value of layer(x), its LoRA added in the same pass over the rows; None when the layer has no such LoRA."""
    if _lora_triton is None or not cuda_inference(x, dtypes=_LORA_DTYPES) or not choose("LoRA Output", owned_triton=True):
        return None
    split = offload.linear_output_and_lora(layer, x)
    if split is None:
        return None
    out, (y, factor, scale, rank_first) = split
    if _lora_triton.supported(out, y, factor, rank_first):
        return _lora_triton.swiglu_(out, y, factor, scale, rank_first)
    out.addmm_(y, factor if rank_first else factor.T, beta=1, alpha=scale)
    return swiglu([out])


LORA_RESIDUAL_FUSION = True  # comparison switch


_lora_residual_kernel = weakref.WeakKeyDictionary()  # layer -> whether its last _lora_linear_residual_ took lora_triton.residual_ (graph loops follow it)


def _lora_linear_residual_(layer, x, residual, scale):
    """residual += scale * layer(x) (scale: None or one value per output) for a layer whose LoRAs add one low-rank product: its
    output pass and the residual update in one pass over the rows, with the roundings of layer(x) followed by add_ / addcmul_.
    False where it does not apply: the caller runs them."""
    if (not LORA_RESIDUAL_FUSION or _lora_triton is None or residual.dtype != x.dtype or residual.ndim != 2 or residual.stride(1) != 1
            or not cuda_inference(x, dtypes=_LORA_DTYPES) or not choose("LoRA Output", owned_triton=True)):
        return False
    split = offload.linear_output_and_lora(layer, x)
    if split is None:
        _lora_residual_kernel[layer] = False
        return False
    out, (y, factor, lora_scale, rank_first) = split
    _lora_residual_kernel[layer] = _lora_triton.supported(out, y, factor, rank_first)
    if _lora_residual_kernel[layer]:
        _lora_triton.residual_(residual, out, y, factor, lora_scale, rank_first, scale)
        return True
    out.addmm_(y, factor if rank_first else factor.T, beta=1, alpha=lora_scale)
    residual.add_(out) if scale is None else residual.addcmul_(out, scale)
    return True


def gated_mlp(x_list, gate_proj, up_proj, down_proj, activation):
    """Separate projections with native activation rounding and module hooks.

    Evaluate activation before up_proj so its source and result do not coexist
    with the up projection. All three projections keep their MMGP/LoRA hooks.
    """
    x = take(x_list)
    gate = activation(gate_proj(x))
    gate.mul_(up_proj(x))
    del x
    gate_list, gate = [gate], None
    return linear(down_proj, gate_list)


def situ(x_list, *, packed=True):
    """Bounded SiLU gate and saturated value, preserving native rounding.

    Packed input is value|gate; separate projections pass [gate, value].
    """
    if packed:
        value, gate = take(x_list).chunk(2, -1)
    else:
        gate, value = x_list
        x_list.clear()
    return (4.0 * torch.tanh(gate / 4.0) * torch.sigmoid(gate)) * (25.0 * torch.tanh(value / 25.0))


def swiglu_linear(layer, x_list, residual=None, residual_scale=None):
    if can_linear(layer, x_list[0]):
        return linear(layer, x_list, input_act="swiglu", residual=residual,
                      residual_scale=residual_scale, out=residual)
    output = layer(swiglu(x_list))
    if residual is not None:
        output.mul_(residual_scale).add_(residual)
    return output


def gated_residual_(residual, x_list, gate, *, tanh=False, fused=False, segments=None):
    """residual += gate * x in place, x handed off in x_list; gate broadcastable to x, or with segments [(start, stop, row)] over the
    rows of (batch * tokens) a table of gate rows (rows, dim) applied span by span, without a per-token gate tensor. tanh applies to
    the gate first; fused adds in one rounding (addcmul)."""
    x = take(x_list)
    gate = gate.tanh() if tanh else gate
    if segments is not None:
        rows, source = residual.view(-1, residual.shape[-1]), x.view(-1, x.shape[-1])
        for start, stop, row in segments:
            if stop > start:
                rows[start:stop].addcmul_(source[start:stop], gate[row])
        return residual
    if fused:
        return residual.addcmul_(x, gate)
    x.mul_(gate)
    return residual.add_(x)


def linear_residual_(layer, x_list, residual, scale=None):
    """residual += scale * layer(x) (scale: None for 1, a number, or one value per output channel), written into residual: the INT8
    backend's residual epilogue (one rounding) where the layer supports it, else the layer's output added in place."""
    x = take(x_list)
    if can_linear(layer, x) and residual.dtype == x.dtype and residual.is_contiguous():
        vector = (torch.full((layer.out_features,), scale, dtype=x.dtype, device=x.device) if isinstance(scale, (int, float)) else
                  None if scale is None else scale.reshape(-1).to(x.dtype))
        return linear(layer, [x], residual=residual, residual_scale=vector, out=residual)
    y = layer(x)
    del x
    if scale is None:
        return residual.add_(y)
    return residual.add_(y, alpha=scale) if isinstance(scale, (int, float)) else residual.addcmul_(y, scale.to(y.dtype))


def column_input(layer, rows, dtype, device, segments=None, residual=None):
    """For a layer whose input (rows, in_features) in dtype is computed by ranges of columns (an attention output by groups of
    heads: attention_kit out_proj=) and then read by project(layer, [input], residual, gate, segments): an object that takes each
    range as it is ready (append(x_list)) and keeps it as the layer's weight format reads it, smaller than the 16-bit input (INT8
    layers: half), LoRAs included. The layer's weight format declares it (the column_input of its quantization handler, through
    mmgp.quant_router.column_input); None when the layer cannot read its input that way."""
    return quant_router.column_input(layer, rows, dtype, device, segments, residual)


def project(layer, x_list, residual=None, gate=None, segments=None):
    """layer(x), x (..., in_features) handed off in x_list or the layer's column_input(): returned, (..., out_features) ((rows,
    out_features) from a column input); or, with residual (..., out_features), added to it and residual returned: linear_residual_
    with gate as its scale, or with segments [(start, stop, row)] over the rows gated_linear with the gate table."""
    x = take(x_list)
    if not torch.is_tensor(x):  # a column_input(): its handler's own projection
        return x.linear() if residual is None else x.linear_residual(residual, gate) if segments is None else x.gated_linear(residual, gate, segments)
    if residual is None:
        return layer(x)
    x_list, x = [x.reshape(*residual.shape[:-1], x.shape[-1])], None
    return linear_residual_(layer, x_list, residual, gate) if segments is None else gated_linear(layer, x_list, residual, gate, segments)


SEGMENT_GATED_GEMM = True  # comparison switch: False runs one linear per segment


def gated_linear(layer, x_list, residual, gate, segments):
    """residual[start:stop] += gate[row] * layer(x[start:stop]) for each segment (start, stop, row), written into residual: the INT8
    backend's residual epilogue (one rounding) where the layer supports it, else the layer's output scaled and added in place."""
    x = take(x_list)
    fused = can_linear(layer, x) and residual.dtype == x.dtype and residual.is_contiguous()
    if fused and SEGMENT_GATED_GEMM and segments and x.dim() == 2 and int8_backend.segment_gated_linear_supported(layer, x):  # one GEMM per chunk of rows
        from . import row_loop_triton
        return int8_backend.segment_gated_linear_(layer, x, residual, gate, row_loop_triton.segment_runs(segments))
    for start, stop, row in segments:
        if stop > start:
            target = residual[start:stop]
            if fused:
                out = linear(layer, [x[start:stop]], residual=target, residual_scale=gate[row].to(x.dtype), out=target)
                if out is not target:
                    target.copy_(out)
            else:
                target.addcmul_(layer(x[start:stop]), gate[row].to(residual.dtype))
    return residual


MLP_MIN_ROWS = 512  # fewer rows per chunk make the GEMMs less efficient (RTX 4090: 3744 rows within noise, 936 rows 3-5 % slower)
MLP_MIN_CHUNKS = 4  # an FFN with two expanded operands per chunk (gate | up, or activation | prefix) keeps its working set within a quarter of the whole expansion, as the models' former loops did; a single expansion follows the L2 budget alone (Qwen Image: 2304-row chunks, 6 % faster than 1024)
MLP_ROW_MULTIPLE = 256  # chunks of whole GEMM row tiles: 980 rows ran a Flux single block 6 % slower than 768 (a partial last tile)
MLP_MAX_CHUNKS = 16  # per call: every chunk adds its kernels to a block's CUDA graph
MLP_L2_FRACTION = 0.75  # of the device's L2 cache for a chunk's working set, so that the activation kernel reads it from L2
_l2_cache_bytes = {}


def mlp_chunk_rows(tokens, expanded_row_bytes, device, double=False):
    """Rows per chunk of mlp(): the chunk's working set (expanded_row_bytes per row: the fc1 output, plus the up projection or the
    concatenated fc2 input of a prefix MLP) within MLP_L2_FRACTION of the device's L2 cache, and, for those double working sets,
    within a MLP_MIN_CHUNKS-th of the tokens; never fewer than MLP_MIN_ROWS nor more than MLP_MAX_CHUNKS chunks per call; the whole
    input when it is that short. A larger floor (2048 rows) used to override the budget for wide FFNs and raised their transient above
    the former per-model loops (Flux single blocks: 113 MB against 43 MB at 1024x1024)."""
    index = torch.cuda.current_device() if device.index is None else device.index
    if index not in _l2_cache_bytes:
        _l2_cache_bytes[index] = torch.cuda.get_device_properties(index).L2_cache_size
    budget = int(_l2_cache_bytes[index] * MLP_L2_FRACTION) // expanded_row_bytes
    if double:
        budget = min(budget, -(-tokens // MLP_MIN_CHUNKS))
    rows = max(MLP_MIN_ROWS, budget // MLP_ROW_MULTIPLE * MLP_ROW_MULTIPLE, -(-tokens // MLP_MAX_CHUNKS))  # whole GEMM row tiles
    return min(tokens, rows)


MLP_PROBE = True  # comparison switch: False keeps mlp_chunk_rows' size
MLP_PROBED_CHUNKS = 64  # per call with a probed size: a size faster per row may take more chunks than MLP_MAX_CHUNKS
_mlp_probing = False


def _probed_chunk_rows(x, out, fc1, fc2, act, up, residual, scale, rows):
    """Rows per chunk of this FFN from the probe of its chunk sizes on this GPU (shared.kernels.chunk_rows: once per FFN and GPU, a
    second or so, on scratch copies of the call's first rows): the size, of at most the rule's `rows` (so never a larger working set),
    that computes the call's rows fastest, in at most MLP_PROBED_CHUNKS chunks. `rows` where nothing faster is known."""
    global _mlp_probing
    from . import chunk_rows

    def kind(layer):  # what changes the kernels a layer runs
        return f"{getattr(getattr(layer, 'weight_qtype', None), 'name', layer.weight.dtype)}{'+lora' if offload.lora_active(layer) else ''}"

    identity = (f"mlp {act} {fc1.in_features}x{fc1.out_features}x{fc2.out_features} {kind(fc1)} {kind(fc2)}{'' if up is None else ' up ' + kind(up)} "
                f"{str(x.dtype).rsplit('.', 1)[-1]} int8={int8_backend._backend} residual={'none' if residual is None else 'gated' if scale is not None else 'plain'}")

    def prepare(most):
        source = x.fill(0, most) if isinstance(x, offload.RowSource) else x[:most].clone()
        target = None if residual is None else out[:most].clone()

        def launch(count):
            global _mlp_probing
            _mlp_probing = True
            try:
                mlp([source[:count]], fc1, fc2, act, up=up, residual=None if target is None else target[:count], gate=scale)
            finally:
                _mlp_probing = False
        return launch

    total = x.rows() if isinstance(x, offload.RowSource) else x.shape[0]
    times = chunk_rows.curve(identity, x.device, "FFN", prepare, min(rows, total))
    return rows if times is None else chunk_rows.choose(times, total, rows, MLP_PROBED_CHUNKS) or rows


MLP_LOOP_COST = 0.001  # share of a turn's chunks' time its pause (as probed: a real body pays somewhat more) may cost: chunks per turn follow (cuda_graphs.loop_turn_size; LTX-2.5 720p FFN, ~1 ms chunks, ConvRot Triton: 1 per turn +0.9 %, 2-3 +0.5 %, 8 +0.3 %)
MLP_LOOP_CHUNKS_PER_TURN = None  # a fixed number of chunks per turn instead (comparisons, tests)
_LOOP_ACTIVATIONS = {"gelu": lambda h: torch.ops.aten.gelu_(h, approximate="tanh"), "gelu_erf": lambda h: torch.ops.aten.gelu_(h, approximate="none"),
                     "silu": torch.ops.aten.silu_}
mlp_loop_refusals = collections.Counter()  # why mlp() calls recorded in graphs ran without a loop ("looped": those with one)
cuda_graphs.register_key(lambda: (MLP_LOOP_COST, MLP_LOOP_CHUNKS_PER_TURN, SEGMENT_GATED_GEMM))  # (cuda_graphs.GRAPH_LOOPS is in the key: graphs with and without loops)
def _chunk_ms(fc2, rows):
    """The time of a chunk of `rows` rows of this FFN in its block's eager warmup (the call before its capture), None if not timed."""
    return cuda_graphs.loop_turn_ms(fc2, ("mlp chunk", rows))


def _mlp_graph_loop(x, out, fc1, fc2, act, residual, chunks, segments, up=None, chunk_fn=None):
    """While a CUDA graph is captured, runs mlp()'s chunks as graph loops (mmgp.cuda_graphs.loop): the evenly spaced chunks of one size (a
    position inside each frame when frames are cut in chunks of two sizes) take one loop, one turn of MLP_LOOP_CHUNKS_PER_TURN chunks recorded
    once instead of the nodes of every chunk. Two routes, both with the usual path's chunks, kernels and order of operations (the same bits):
    - kernels: the INT8 kernels find their rows through the loop's counter (ConvRot and Quanto INT8 linears on WGPKernels with row-offset
      entry points or on the Triton backend, LoRAs through their counter forms), an activation in place, one fc2 fuses into its quantizer
      (packed SwiGLU, WGPKernels) or a SwiGLU of a separate up projection: no copies (the Triton backend adds fc2's output to the
      residual rows by the counter, as the usual path's own addition);
    - generic (any other weight format and backend: Kitchen, FP8, 16-bit, GGUF): the turn's input rows, residual rows and segment
      gate gathered at fixed addresses (row_loop_triton), mlp()'s own chunk (chunk_fn) run on them, the output rows stored back: three
      copies of a chunk's rows; it needs the loop's allocations in the graph's pool (cuda_graphs.loop_allocations).
    One gate, or one per segment of equal size from row 0. x may be an offload.RowSource with fill_at, whose rows each turn then makes in
    the loop's body. Returns the chunks left for the usual path."""
    source = None if isinstance(x, torch.Tensor) else x
    sample = lambda rows, width: torch.empty((rows, width), dtype=x.dtype, device=x.device)  # shapes for the kernels' choices (no kernel runs)
    refusal = "no loops" if not cuda_graphs.loop_available(x.device) else "gate without residual" if residual is None and segments[0][2] is not None else None
    span, gate = 0, segments[0][2]
    if refusal is None and len(segments) > 1:  # a gate per segment: the kernels read the vector of their rows' segment
        span = segments[0][1] - segments[0][0]
        if any(start != index * span or stop != start + span for index, (start, stop, _) in enumerate(segments)):
            refusal = "uneven segments"
        gate = torch.stack([scale for *_, scale in segments])
    if refusal:
        mlp_loop_refusals[refusal] += 1
        return chunks
    from . import row_loop_triton
    generic = chunk_fn is not None and cuda_graphs.loop_allocations()
    fusable = act if up is None and act in ("gelu", "swiglu") else None
    fused = fusable if fusable and can_linear(fc2, sample(1, fc1.out_features), act=fusable) else None  # the usual path's choice
    lora1, lora2, lora_up = offload.lora_active(fc1), offload.lora_active(fc2), up is not None and offload.lora_active(up)
    kernel_refusal = ("SwiGLU of fc1's LoRA pass" if fusable == "swiglu" and fused is None
                      else "activation" if fused is None and act not in _LOOP_ACTIVATIONS and not (act == "swiglu" and up is not None)
                      else "LoRA without graph pool" if (lora1 or lora2 or lora_up) and not cuda_graphs.loop_allocations()
                      else "LoRA output" if lora2 and (residual is None or not _lora_residual_kernel.get(fc2, False)) else None)
    if kernel_refusal and not generic:
        mlp_loop_refusals[kernel_refusal] += 1
        return chunks
    left, groups = [], {}
    for chunk in chunks:
        groups.setdefault(chunk[1] - chunk[0], []).append(chunk)
    counter, activation = None, None if fused is not None or up is not None or act not in _LOOP_ACTIVATIONS else _LOOP_ACTIVATIONS[act]
    for size, members in groups.items():
        stride = members[1][0] - members[0][0] if len(members) > 1 else 0
        if len(members) < 2 or any(first != members[0][0] + index * stride for index, (first, *_) in enumerate(members)):
            left += members
            continue
        per_turn = (min(MLP_LOOP_CHUNKS_PER_TURN, len(members)) if MLP_LOOP_CHUNKS_PER_TURN
                    else cuda_graphs.loop_turn_size(x.device, _chunk_ms(fc2, size), len(members), MLP_LOOP_COST))
        turns = len(members) // per_turn
        if turns == 1:  # chunks too short for their pauses: one turn would hold them all, as many nodes as without a loop
            mlp_loop_refusals["short chunks"] += 1
            left += members
            continue
        base, step = members[0][0], per_turn * stride
        fills = None if source is None else [source.fill_at(base + index * stride, step, size) for index in range(per_turn)]
        if fills is not None and None in fills:
            mlp_loop_refusals["row source fill"] += 1
            left += members
            continue
        linears = None
        if kernel_refusal is None:
            h, value = sample(size, fc1.out_features), None if up is None else sample(size, up.out_features)
            # a linear with LoRAs runs as in the usual path (module hooks, cuBLAS) on rows at fixed addresses made in the body: fc1 (and up)
            # on the turn's rows gathered or filled there, fc2 on h, its LoRA output and residual update by the counter form of lora_triton.residual_
            x_rows = sample(size, fc1.in_features) if source is not None else x
            first_linear = None if lora1 else int8_backend.row_loop_linear(fc1, x_rows, size)
            up_linear = None if up is None or lora_up else int8_backend.row_loop_linear(up, x_rows, size)
            second_linear = None if lora2 else int8_backend.row_loop_linear(fc2, h, size, act=fused, out_at=True)
            if not (first_linear is None and not lora1 or second_linear is None and not lora2 or up is not None and up_linear is None and not lora_up):
                linears = first_linear, up_linear, second_linear
        if linears is None and not generic:
            mlp_loop_refusals["linears"] += 1
            left += members  # the groups looped already stay looped
            continue
        counter = torch.empty(1, dtype=torch.int64, device=x.device) if counter is None else counter

        def project(linear, module, lora, rows, target):  # module(the turn's rows): rows gathered or filled at fixed addresses, else by the counter
            if lora:
                return module(rows)
            linear(x, target, counter, x_at=(first, step)) if rows is None else linear(rows, target, counter)
            return target
        with cuda_graphs.loop(turns, counter):
            for index in range(per_turn):
                first = base + index * stride
                rows = None
                if linears is None:  # generic: mlp()'s own chunk on the turn's rows gathered at fixed addresses, its output rows stored back
                    rows = fills[index](counter) if source is not None else row_loop_triton.copy_rows_at(x, counter, first, step, size)
                    target = rows if residual is None else row_loop_triton.copy_rows_at(out, counter, first, step, size)  # out is x without residual
                    scale = gate if not span else row_loop_triton.segment_row_at(gate, counter, first, step, span)
                    chunk_fn(rows, target, scale)
                    row_loop_triton.store_rows_at(out, target, counter, first, step)
                    rows = target = scale = None
                    continue
                first_linear, up_linear, second_linear = linears
                if source is not None or lora1 or lora_up:  # the turn's input rows, made in the body (from the graph's pool)
                    rows = fills[index](counter) if source is not None else row_loop_triton.copy_rows_at(x, counter, first, step, size)
                h = project(first_linear, fc1, lora1, rows, h)
                if up is not None:  # silu(gate) * value in place on h, as the usual path
                    h = swiglu([h, project(up_linear, up, lora_up, rows, value)], packed=False)
                elif activation is not None:
                    activation(h)
                rows = None
                if lora2:
                    base_out, (y, factor, lora_scale, rank_first) = offload.linear_output_and_lora(fc2, h)
                    row_loop_triton.lora_residual_at(out, base_out, y, factor, lora_scale, rank_first, gate, counter, first, step, span)
                    del base_out, y
                else:
                    second_linear(h, out, counter, out_at=(first, step), residual=residual, residual_scale=gate, scale_span=span)
        mlp_loop_refusals["looped"] += 1
        mlp_loop_refusals["generic route" if linears is None else "kernel route"] += 1
        mlp_loop_refusals[f"{per_turn} per turn ({_chunk_ms(fc2, size) or 0:.2f} ms chunks)"] += 1
        left += members[turns * per_turn:]
    return left


def gate_segments(gate, batch, tokens):
    """mlp() segments for a gate of shape (batch or 1, groups, dim) over an input (batch, tokens, dim): groups is 1, or a number of
    frames that divides the tokens. A gate per token does not fit segments: apply it after mlp()."""
    gate = gate.reshape(gate.shape[0], -1, gate.shape[-1])
    groups, span = gate.shape[1], tokens // gate.shape[1]
    return [(b * tokens + g * span, b * tokens + (g + 1) * span, gate[b if gate.shape[0] > 1 else 0, g]) for b in range(batch) for g in range(groups)]


def mlp(x_list, fc1, fc2, act, *, up=None, prefix=None, residual=None, gate=None, segments=None):
    """fc2(act(fc1(x))) by chunks of tokens (mlp_chunk_rows): each chunk's output replaces its input rows, or is added to residual
    scaled by gate and written into residual. x, handed off in x_list, and residual have shape (..., dim) and are contiguous.
    act: 'gelu' (tanh approximation), 'gelu_erf', 'silu', 'situ' (fc1 rows value | gate), or 'swiglu' (fc1 rows gate | value).
    With up, fc1 is the gate and up is the value projection for situ/swiglu. prefix: rows
    (..., p) borrowed and concatenated before the activated fc1 output as fc2's input (fc2 reads p + the activation's width: the
    single-stream blocks of Flux, whose second linear takes the attention output and the MLP together). gate is one vector per channel
    for all tokens, or segments [(start, stop, vector)] over the flattened tokens (frames, spans, batch rows: gate_segments). Where
    the INT8 backend supports it, the activation is fused into fc2's kernel (gelu, packed swiglu, without prefix) and the gated residual
    into its epilogue, rounded once; otherwise the activation runs in place and the residual update follows. Returns x, or residual.
    x may be an mmgp offload.RowSource when residual is given: each chunk of rows is made (its fill) just before fc1 reads it, so that
    the input is never whole (a normalized and modulated copy of the residual: a chunk reads its own rows before its output is added)."""
    x = take(x_list)
    source = x if isinstance(x, offload.RowSource) else None
    shape, dim = x.shape, x.shape[-1]
    total = source.rows() if source is not None else x.numel() // dim
    if source is None:
        x = x.view(-1, dim)
    out = x if residual is None else residual.view(-1, dim)
    chunk_of = source.fill if source is not None else lambda first, last: x[first:last]
    prefix = None if prefix is None else prefix.view(-1, prefix.shape[-1])
    # working set per row: the fc1 (and up) output, plus the concatenated fc2 input that coexists with it for a prefix MLP
    width = fc1.out_features + (0 if up is None else up.out_features) + (0 if prefix is None else prefix.shape[-1] + fc1.out_features)
    rows = total if _mlp_probing else mlp_chunk_rows(total, width * x.dtype.itemsize, x.device, double=up is not None or prefix is not None)
    if (MLP_PROBE and not _mlp_probing and prefix is None and rows < total
            and cuda_inference(source.fill(0, 1) if source is not None else x, dtypes=_LORA_DTYPES)):
        # a chunk size that this GPU computes faster per row than the rule's, when its probe knows one
        rows = _probed_chunk_rows(x, out, fc1, fc2, act, up, residual, gate if segments is None else segments[0][2], rows)
    fusable = act if up is None and prefix is None and act in ("gelu", "swiglu") else None
    segments = [(start, stop, None if scale is None else scale.reshape(-1).to(out.dtype)) for start, stop, scale in ([(0, total, gate)] if segments is None else segments)]
    chunks = [(first, min(first + rows, stop), scale) for start, stop, scale in segments for first in range(start, stop, rows)]

    def chunk(rows_in, target, scale, prefix_rows=None):
        """One chunk: rows_in (the chunk's input rows) through the FFN into target (its output rows: residual rows updated, else written),
        gated by scale. CUDA graph loops run it on rows at fixed addresses (_mlp_graph_loop's generic route): the same calls."""
        # an activation that fc2's kernel cannot take (fc2 has a LoRA, whose product needs it) can still share fc1's pass over its rows
        h = _swiglu_of_linear(fc1, rows_in) if fusable == "swiglu" and not can_linear(fc2, rows_in, act=fusable) else None
        activated = h is not None
        h = h if activated else fc1(rows_in)
        fused = fusable if not activated and fusable and can_linear(fc2, h, act=fusable) else None
        if fused is None and not activated:
            h = (situ([h, up(rows_in)], packed=False) if act == "situ" and up is not None else situ([h]) if act == "situ"
                 else swiglu([h, up(rows_in)], packed=False) if up is not None else swiglu([h]) if act == "swiglu"
                 else torch.ops.aten.silu_(h) if act == "silu" else torch.ops.aten.gelu_(h, approximate="tanh" if act == "gelu" else "none"))
        rows_in = None
        if prefix_rows is not None:
            h = torch.cat((prefix_rows, h), dim=-1)
        if target.dtype == h.dtype and (fused is not None or (residual is not None and can_linear(fc2, h))):
            linear(fc2, [h], input_act=fused, residual=None if residual is None else target, residual_scale=scale, out=target)
            return
        if fused is None and residual is not None and _lora_linear_residual_(fc2, h, target, scale):
            return  # fc2's LoRA output and the residual update share one pass
        y = linear(fc2, [h], input_act=fused) if fused is not None else fc2(h)
        del h
        if residual is None:
            target.copy_(y)
        elif scale is None:
            target.add_(y)
        else:
            target.addcmul_(y, scale)

    with int8_backend.chunked_rows():  # the chunks are sized here: the INT8 kernels take each one whole
        loops = not _mlp_probing and cuda_graphs.recording() and cuda_graphs.loop_available(out.device)
        if loops and torch.cuda.is_current_stream_capturing():
            refusal = ("row source" if source is not None and (source.fill_at is None or not cuda_graphs.loop_allocations()) else "prefix" if prefix is not None
                       else "up projection" if up is not None and act not in ("swiglu", "situ") else None)
            if refusal:
                mlp_loop_refusals[refusal] += 1
            else:
                chunks = _mlp_graph_loop(source if source is not None else x, out, fc1, fc2, act, residual, chunks, segments, up, chunk)
        timed = loops and not torch.cuda.is_current_stream_capturing()  # the eager warmup: every chunk timed (the shortest of a size is kept)
        for first, last, scale in chunks:
            end = cuda_graphs.loop_turn_begin(fc2, ("mlp chunk", last - first)) if timed else None
            chunk(chunk_of(first, last), out[first:last], scale, None if prefix is None else prefix[first:last])
            if end is not None:
                end()
    return out.view(shape)


def prepare_split_half_rope(rope):
    if not cuda_inference(rope) or choose("Split-Half RMSNorm + RoPE", owned_triton=_triton is not None, kitchen=_kitchen_rms_rope is not None) != "kitchen":
        return rope
    c, s = rope[..., 0], rope[..., 1]
    return rope, torch.stack((c, -s, s, c), dim=-1).reshape(*c.shape, 2, 2)


def can_split_half_rms_rope(x, packed, q_norm, k_norm):
    return ((cuda_inference(x) and (packed is not None or _triton is not None) or _bf16_bits_eligible(x))
            and all(p.device == x.device and p.dtype == x.dtype for p in (q_norm.weight, k_norm.weight)))


def _native_split_half_norm_rope_(tensors, rope):
    for tensor, norm in tensors:
        rows = max(1, (256 << 20) // (tensor[0, 0].numel() * tensor.element_size()))
        for start in range(0, tensor.shape[1], rows):
            part = tensor[:, start:start + rows]
            part.copy_(norm(part))
    if rope is not None:
        pairs = rope.shape[-2]
        cosine, sine = rope[..., 0], rope[..., 1]
        scratch = None
        for tensor, _ in tensors:
            first, second = tensor[..., :pairs], tensor[..., pairs:2 * pairs]
            if scratch is None or scratch.shape != first.shape:  # q and k differ in heads under grouped-query attention
                scratch = torch.empty_like(first)
            scratch.copy_(first)
            first.mul_(cosine).addcmul_(second, sine, value=-1)
            second.mul_(cosine).addcmul_(scratch, sine)


def split_half_norm_rope_(q, k, rope, packed, q_norm, k_norm, *, allow_fusion=True):
    x = q if q is not None else k
    if rope is not None and allow_fusion and can_split_half_rms_rope(x, packed, q_norm, k_norm):
        return split_half_rms_rope_(q, k, rope, packed, q_norm, k_norm)
    _native_split_half_norm_rope_([(x, norm) for x, norm in ((q, q_norm), (k, k_norm)) if x is not None], rope)
    return q, k


def interleaved_norm_rope_(q, k, rope, q_norm, k_norm, *, rope_first=False):
    """In place per-head RMSNorm then the rotation of channel pairs (2p, 2p + 1) of q and / or k (batch, tokens, heads, head_dim; the
    other None) by rope (1 | batch, tokens, 1, pairs, 2) cos / sin, a complex frequency table viewed as real; rope None normalizes
    only. Norms: modules with .weight and .eps, or (weight, eps) tuples whose weight is (head_dim,) or one row per head (heads,
    head_dim), in the activation dtype or FP32; with tuples, rope may hold one table per head (1 | batch, tokens, heads, pairs, 2). The
    fused Triton kernel on CUDA inference tensors (FP32 statistics, the norm output rounded to the activation dtype, the rotation in
    FP32 with one rounding), else the norm and paired_rope_ by chunks of tokens, without complex arithmetic. rope_first (tuples): the
    rotation, then the RMSNorm of the rotated rows, both in FP32 and rounded once (Hi3D's and TripoSplat's order)."""
    for x, norm in ((q, q_norm), (k, k_norm)):
        if x is None:
            continue
        if isinstance(norm, tuple):
            weight, eps = norm
            if _triton is not None and cuda_inference(x, dtypes=(torch.bfloat16, torch.float16)) and weight.device == x.device:
                _triton.rms_rope_(x, weight, rope, eps, interleaved=True, rope_first=rope_first)
                continue
            rows = max(1, (64 << 20) // (x[0, 0].numel() * 4))
            for start in range(0, x.shape[1], rows):
                part = x[:, start:start + rows]
                y = part.float()
                if rope_first and rope is not None:
                    y = paired_rope_(y, rope[:, start:start + rows, ..., 0], rope[:, start:start + rows, ..., 1], interleaved=True)
                y = y.mul_(y.square().mean(-1, keepdim=True).add_(eps).rsqrt_()).mul_(weight)
                if not rope_first and rope is not None:  # the norm output rounded, then rotated in FP32
                    y = paired_rope_(y.to(x.dtype).float(), rope[:, start:start + rows, ..., 0], rope[:, start:start + rows, ..., 1], interleaved=True)
                part.copy_(y)
            continue
        if rope is not None and _triton is not None and cuda_inference(x) and norm.weight.device == x.device and norm.weight.dtype == x.dtype:
            _triton.rms_rope_(x, norm.weight, rope, norm.eps, interleaved=True)
            continue
        rows = max(1, (256 << 20) // (x[0, 0].numel() * x.element_size()))
        for start in range(0, x.shape[1], rows):
            part = x[:, start:start + rows]
            normalized = norm(part).to(x.dtype)
            part.copy_(normalized if rope is None else paired_rope_(normalized, rope[:, start:start + rows, ..., 0], rope[:, start:start + rows, ..., 1], interleaved=True))
    return q, k


def split_half_rms_rope_(q, k, rope, packed, q_norm, k_norm):
    """Borrow Q/K destinations. Either can be absent for staged attention."""
    if packed is not None and q is not None and k is not None:
        return _kitchen_rms_rope(q, k, packed, q_norm.weight, k_norm.weight,
                                 epsilon=q_norm.eps, rot_dim=packed.shape[-3] * 2)
    for x, norm in ((q, q_norm), (k, k_norm)):
        if x is None:
            continue
        if packed is not None and x.shape[2] > 1:
            half = x.shape[2] // 2
            _kitchen_rms_rope(x[:, :, :half], x[:, :, half:], packed, norm.weight, norm.weight,
                               epsilon=norm.eps, rot_dim=packed.shape[-3] * 2)
        elif _bf16_bits_eligible(x):  # below the BF16 floor: the kernel reading BF16 as bits
            _bits_triton.rms_rope_(x, norm.weight, rope, norm.eps)
        elif _triton is not None:
            _triton.rms_rope_(x, norm.weight, rope, norm.eps)
        else:
            # A one-head group cannot be divided for Kitchen's paired API.
            _native_split_half_norm_rope_([(x, norm)], rope)
    return q, k


def _norm_parameters(norm, tokens, device):
    """(weight, eps, weight_rows) of a per-head RMSNorm given as a module (.weight, .eps), as a (weight, eps) tuple, or as segments
    [(start, stop, module or tuple)] over the tokens with their own weights (stacked into (segments, D) rows, weight_rows the row of
    each token; one eps); (None, 0.0, None) for no norm."""
    if norm is None:
        return None, 0.0, None
    if isinstance(norm, list):
        parts = [_norm_parameters(segment, tokens, device) for _, _, segment in norm]
        if any(part[1] != parts[0][1] for part in parts):
            raise ValueError("Segments of a fused q/k norm must share their epsilon")
        rows = torch.zeros(tokens, dtype=torch.int32, device=device)
        for index, (start, stop, _) in enumerate(norm):
            rows[start:stop] = index
        return torch.stack([part[0] for part in parts]), parts[0][1], rows
    if isinstance(norm, tuple):
        return *norm, None
    return norm.weight, torch.finfo(torch.float32).eps if norm.eps is None else norm.eps, None


def norm_rope_query_sample(q, ids, rope, q_norm, core, *, interleaved=False, rope_first=False):
    """Apply the fused quantizer's per-head norm and the original positions' RoPE to sampled Q rows."""
    if isinstance(q_norm, list) or (rope is not None and (rope.ndim not in (5, 6) or rope.shape[3] > 64)):
        return None
    norm = _norm_parameters(q_norm, q.size(1), q.device)
    weight = norm[0]
    if weight is not None and (weight.device != q.device or weight.dtype not in (q.dtype, torch.float32)
                               or weight.ndim not in (1, 2) or weight.size(-1) != 128 or weight.stride(-1) != 1):
        return None
    return core.sample_projected_query(q, ids, rope, norm, interleaved=interleaved, rope_first=rope_first)


def split_half_norm_rope_quant_(q, k, rope, q_norm, k_norm, core, settings, *, interleaved=False, rope_first=False):
    """The staged SageAttention 2 quantization of q and / or k (batch, tokens, heads, head_dim; the other None) fused with their
    per-head RMSNorm (a module with .weight and .eps, a (weight, eps) tuple, None, or segments [(start, stop, module or tuple)] over
    the tokens, each with its own weights, in the activation dtype or FP32: a joint attention's streams; a tuple's weight may hold one
    row per head, (heads, head_dim)) and RoPE (rope: (1 | batch, tokens, 1 | heads, pairs, 2) cos / sin, one table for all heads or one per head,
    or (..., pairs, 2, 2) rotation matrices [[cos, -sin], [sin, cos]], read through their strides; None for no rotation; channel p
    with p + pairs, or interleaved 2p with 2p + 1): (q_quantized, k_quantized) as the core's staged_quantize_q / staged_quantize_k
    return them, quantized from the FP32 rotated rows (one rounding fewer than the in-place kernel followed by the core's
    quantizer), k's mean from FP32 partial sums. The CUDA per-warp query path preserves the rotation's 16-bit rounding.
    rope_first rotates before the norm (Hi3D), both in FP32. Padded heads get INT8 rows with zeros beyond head_dim.
    The caller releases k after packing. A core with recycle_query enabled may pack into the disposable q projection
    before staged_attention writes its output there. Other calls preserve q during packing. None where the fusion does not apply
    (a core without fused_qk_layouts or a quantization layout the kernels do not reproduce, no CUDA inference, attended head_dim not
    64 / 128, norm weights on another device / dtype): the caller runs norm_rope and the core's quantizers."""
    x = q if q is not None else k
    width = settings.get("padded_dim") or x.shape[-1]
    layouts = core.fused_qk_layouts(settings, x.shape[-1]) if _qk_quant is not None and hasattr(core, "fused_qk_layouts") else None
    # the kernel reads the weights in their own dtype (FP32 scales of some INT8 checkpoints included) and multiplies in FP32
    warp_query = getattr(core, "fused_warp_query", None) if layouts is None and k is None else None
    warp_key = getattr(core, "fused_warp_key", None) if layouts is None and q is None else None
    if (layouts is None and warp_query is None and warp_key is None) or width not in (64, 128) or not cuda_inference(x, dtypes=(torch.bfloat16, torch.float16)):
        return None
    q_weight, q_eps, q_rows = _norm_parameters(q_norm if q is not None else None, x.shape[1], x.device)
    k_weight, k_eps, k_rows = _norm_parameters(k_norm if k is not None else None, x.shape[1], x.device)
    if not all(w.device == x.device and w.dtype in (x.dtype, torch.float32) for w in (q_weight, k_weight) if w is not None):
        return None
    options = {"interleaved": interleaved, "rope_first": rope_first}
    if warp_key is not None:
        key = warp_key(k, rope, (k_weight, k_eps, k_rows), settings, **options)
        return None if key is None else (None, key)
    if warp_query is not None:
        query = warp_query(q, rope, (q_weight, q_eps, q_rows), settings, **options)
        return None if query is None else (query, None)
    quantized = [None, None]
    if k is not None:
        layout, block, sm_scale, scale_eps = layouts["k"]
        mean = settings.get("k_mean")  # a cached prefix's mean (staged_quantize_k), else k's own
        mean = _qk_quant.norm_rope_mean(k, k_weight, rope, k_eps, weight_rows=k_rows, **options) if mean is None else mean.expand(k.shape[0], -1, -1)
        quantized[1] = _qk_quant.norm_rope_quantize(k, k_weight, rope, k_eps, layout, block, mean=mean, sm_scale=sm_scale, scale_eps=scale_eps,
                                                    weight_rows=k_rows, width=width, **options)
        del mean
    if q is not None:
        layout, block, sm_scale, scale_eps = layouts["q"]
        recycle = hasattr(core, "fused_query_recycle") and core.fused_query_recycle(q, settings)
        quantized[0] = _qk_quant.norm_rope_quantize(q, q_weight, rope, q_eps, layout, block, sm_scale=sm_scale, scale_eps=scale_eps,
                                                    weight_rows=q_rows, width=width, recycle=recycle, **options)
    return tuple(quantized)


def sol_rms_rope_(x, weight, rope, eps):
    if choose(owned_triton=_triton is not None and cuda_inference(x)):
        return _triton.rms_rope_(x, weight, rope, eps)
    from shared.sol_attn import rms_norm_rope_
    rms_norm_rope_(x, weight, rope, eps)
    torch.autograd.graph.increment_version(x)
    return x


def compact_rope_(x, cache, layout, head_start=0, dim_head=None, token0=0, counter=None, step=0):
    """Try compact split RoPE without materializing token-sized tables.

    False leaves x untouched for the model's layout-specific PyTorch fallback.
    Norm over all heads and its prepared statistics are deliberately external.
    token0 (+ counter * step, a CUDA graph loop's turn: the native Triton kernel only): x's tokens are the layout's from that token,
    rotated with the layout's whole tables in one launch.
    """
    native = _triton is not None and cuda_inference(x, approximate=True)
    bits = not native and _bf16_bits_eligible(x, approximate=True)  # below the BF16 floor: x and BF16 tables as bits
    valid = ((native or bits)
             and x.stride(-1) == 1 and x.stride(1) == x.shape[2]
             and len(layout.axis_cos) in (1, 3) and 1 <= len(layout.grid_sizes) <= 3
             and 0 <= cache.pad_size < len(layout.axis_cos)
             and (x.shape[0] <= 1 or x.stride(0) >= x.shape[1] * x.shape[2]))
    dtype = torch.float32 if cache.use_fp32_freqs else x.dtype
    valid = valid and all(t.device == x.device and t.dtype == dtype and t.is_contiguous()
                          for t in (*layout.axis_cos, *layout.axis_sin))
    if (token0 or counter is not None) and bits or not choose("Compact Split RoPE", owned_triton=valid):
        return False
    if token0 or counter is not None:
        _triton.split_rope(x, cache, layout, head_start, dim_head, token0, counter, step)
    else:
        (_bits_triton if bits else _triton).split_rope(x, cache, layout, head_start, dim_head)
    torch.autograd.graph.increment_version(x)
    return True


def zero_center_rms_norm(x_list, weight, eps, *, offset_in_weight_dtype=False):
    """FP32 RMSNorm with the weight stored around zero: (weight + 1) added in FP32, or in the weight's dtype first (Krea 2's
    `scale + 1.0` on a BF16 parameter), then the product in FP32 and one final rounding. With the offset in the weight's dtype on a
    16-bit CUDA tensor, PyTorch's fused kernel does that arithmetic in FP32 without materializing FP32 copies of x (two of them cost
    a Krea 2 block 25 MiB of peak)."""
    x = take(x_list)
    if offset_in_weight_dtype and x.is_cuda and x.dtype in (torch.bfloat16, torch.float16) and weight.dtype == x.dtype:
        return F.rms_norm(x, (x.shape[-1],), weight + 1, eps)
    scale = (weight + 1).float() if offset_in_weight_dtype else weight.float() + 1
    return F.rms_norm(x.float(), (x.shape[-1],), scale, eps).to(x.dtype)


def sigmoid_gate_(x, gate_list):
    """x *= sigmoid(gate) in place, with PyTorch's native sigmoid rounding; the gate is consumed (its sigmoid is taken in place)."""
    gate = take(gate_list)
    return x.mul_(gate.sigmoid_())


def rms_norm_rounded(x_list, eps=1e-6, *, manual=False):
    """Unweighted FP32 RMSNorm, rounded before a caller's weight multiply.

    Flux uses explicit square/mean/rsqrt; SenseNova uses native rms_norm. Keep
    those reductions distinct, and leave weight promotion to the norm module.
    """
    x = take(x_list)
    dtype = x.dtype
    x = x.float()
    if manual:
        x = x * torch.rsqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    else:
        x = F.rms_norm(x, (x.shape[-1],), eps=eps)
    return x.to(dtype)


def matrix_rope(x_list, freqs):
    """FP32 matrix rotation with separate products, followed by a final cast."""
    x = take(x_list)
    shape, dtype = x.shape, x.dtype
    x = x.float().reshape(*shape[:-1], -1, 1, 2)
    output = freqs[..., 0] * x[..., 0]
    x = freqs[..., 1] * x[..., 1]
    return output.add_(x).reshape(shape).to(dtype)


def paired_rope_(x, cosine, sine, scratch=None, *, interleaved=False):
    """Native multiply then addcmul; borrow the destination and optional scratch."""
    if interleaved:
        first, second = x[..., 0::2], x[..., 1::2]
    else:
        first, second = x.chunk(2, dim=-1)
    if scratch is None:
        scratch = first.clone()
    else:
        scratch = scratch[tuple(slice(0, size) for size in first.shape)]
        scratch.copy_(first)
    first.mul_(cosine).addcmul_(second, sine, value=-1)
    second.mul_(cosine).addcmul_(scratch, sine)
    return x


def interleaved_rope(x_list, freqs):
    """Qwen Image's FP32 multiply/addcmul rotation after native norm rounding."""
    x = take(x_list)
    dtype = x.dtype
    x = x.float()
    cosine, sine = (freqs[..., i].unsqueeze(0).unsqueeze(2) for i in (0, 1))
    return paired_rope_(x, cosine, sine, interleaved=True).to(dtype)


def indexed_rope(x_list, cosine, sine, partner):
    """Native-dtype rotation with model-supplied partner indices and signs."""
    x = take(x_list)
    rotated = x[..., partner]
    return x.mul_(cosine).addcmul_(rotated, sine)


def complex_norm_rope(x_list, norm, freqs):
    x = take(x_list)
    x = norm(x).to(x.dtype)
    if freqs is None:
        return x
    rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    rotated.mul_(freqs.unsqueeze(1))
    return torch.view_as_real(rotated).flatten(3)


def matrix_rms_rope(x_list, weight, freqs, eps=1e-6):
    x = take(x_list)
    dtype = x.dtype
    x = x.float()
    x = x * x.square().mean(-1, keepdim=True).add(eps).rsqrt() * weight.float()
    pairs = x.unflatten(-1, (-1, 2)).unsqueeze(-2)
    return (freqs[..., 0] * pairs[..., 0] + freqs[..., 1] * pairs[..., 1]).flatten(-2).to(dtype)
