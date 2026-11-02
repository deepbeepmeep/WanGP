# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Activation operations shared by denoisers.

Owned inputs use x_list through the dispatch boundary. In-place operations borrow
their destination; residuals, weights and conditioning are always borrowed. Math
contracts are explicit: split-half RMS/RoPE rounds after normalization, matrix
RoPE keeps FP32 until its final cast, and complex RoPE rounds before rotation.
Whole-head normalization and compact rotary layouts remain model supplied.
"""
import torch
import torch.nn.functional as F
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
        gate, value = take(x_list).chunk(2, -1)
    else:
        gate, value = x_list
        x_list.clear()
    return F.silu(gate, inplace=in_place).mul_(value)


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


def gated_linear(layer, x_list, residual, gate, segments):
    """residual[start:stop] += gate[row] * layer(x[start:stop]) for each segment (start, stop, row), written into residual: the INT8
    backend's residual epilogue (one rounding) where the layer supports it, else the layer's output scaled and added in place."""
    x = take(x_list)
    fused = can_linear(layer, x) and residual.dtype == x.dtype and residual.is_contiguous()
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
    into its epilogue, rounded once; otherwise the activation runs in place and the residual update follows. Returns x, or residual."""
    x = take(x_list)
    shape, x = x.shape, x.view(-1, x.shape[-1])
    out = x if residual is None else residual.view(-1, x.shape[-1])
    prefix = None if prefix is None else prefix.view(-1, prefix.shape[-1])
    # working set per row: the fc1 (and up) output, plus the concatenated fc2 input that coexists with it for a prefix MLP
    width = fc1.out_features + (0 if up is None else up.out_features) + (0 if prefix is None else prefix.shape[-1] + fc1.out_features)
    rows = mlp_chunk_rows(x.shape[0], width * x.element_size(), x.device, double=up is not None or prefix is not None)
    fusable = act if up is None and prefix is None and act in ("gelu", "swiglu") else None
    for start, stop, scale in [(0, x.shape[0], gate)] if segments is None else segments:
        scale = None if scale is None else scale.reshape(-1).to(out.dtype)
        for first in range(start, stop, rows):
            last = min(first + rows, stop)
            h = fc1(x[first:last])
            fused = fusable if fusable and can_linear(fc2, h, act=fusable) else None
            if fused is None:
                h = (situ([h, up(x[first:last])], packed=False) if act == "situ" and up is not None else situ([h]) if act == "situ"
                     else swiglu([h, up(x[first:last])], packed=False) if up is not None else swiglu([h]) if act == "swiglu"
                     else torch.ops.aten.silu_(h) if act == "silu" else torch.ops.aten.gelu_(h, approximate="tanh" if act == "gelu" else "none"))
            if prefix is not None:
                h = torch.cat((prefix[first:last], h), dim=-1)
            target = out[first:last]
            if target.dtype == h.dtype and (fused is not None or (residual is not None and can_linear(fc2, h))):
                linear(fc2, [h], input_act=fused, residual=None if residual is None else target, residual_scale=scale, out=target)
                continue
            y = linear(fc2, [h], input_act=fused) if fused is not None else fc2(h)
            del h
            if residual is None:
                target.copy_(y)
            elif scale is None:
                target.add_(y)
            else:
                target.addcmul_(y, scale)
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


def split_half_norm_rope_quant_(q, k, rope, q_norm, k_norm, core, settings, *, interleaved=False, rope_first=False):
    """The staged SageAttention 2 quantization of q and / or k (batch, tokens, heads, head_dim; the other None) fused with their
    per-head RMSNorm (a module with .weight and .eps, a (weight, eps) tuple, None, or segments [(start, stop, module or tuple)] over
    the tokens, each with its own weights, in the activation dtype or FP32: a joint attention's streams; a tuple's weight may hold one
    row per head, (heads, head_dim)) and RoPE (rope: (1 | batch, tokens, 1 | heads, pairs, 2) cos / sin, one table for all heads or one per head,
    or (..., pairs, 2, 2) rotation matrices [[cos, -sin], [sin, cos]], read through their strides; None for no rotation; channel p
    with p + pairs, or interleaved 2p with 2p + 1): (q_quantized, k_quantized) as the core's staged_quantize_q / staged_quantize_k
    return them, quantized from the FP32 rotated rows (one rounding fewer than the in-place kernel followed by the core's
    quantizer), k's mean from FP32 partial sums. rope_first: the rotation before the norm (Hi3D), both in FP32. Heads that the core
    pads (settings["padded_dim"]: 96 attended as 128) get INT8 rows of the padded width, zeros beyond head_dim. q is left as it is
    (staged_attention writes the output into it) and k is not modified (the caller releases it). None where the fusion does not apply
    (a core without fused_qk_layouts or a quantization layout the kernels do not reproduce, no CUDA inference, attended head_dim not
    64 / 128, norm weights on another device / dtype): the caller runs norm_rope and the core's quantizers."""
    x = q if q is not None else k
    width = settings.get("padded_dim") or x.shape[-1]
    (q_weight, q_eps, q_rows), (k_weight, k_eps, k_rows) = (_norm_parameters(norm, x.shape[1], x.device) for norm in (q_norm, k_norm))
    layouts = core.fused_qk_layouts(settings, x.shape[-1]) if _qk_quant is not None and hasattr(core, "fused_qk_layouts") else None
    # the kernel reads the weights in their own dtype (FP32 scales of some INT8 checkpoints included) and multiplies in FP32
    if (layouts is None or width not in (64, 128) or not cuda_inference(x, dtypes=(torch.bfloat16, torch.float16))
            or not all(w.device == x.device and w.dtype in (x.dtype, torch.float32) for w in (q_weight, k_weight) if w is not None)):
        return None
    options = {"interleaved": interleaved, "rope_first": rope_first}
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
        quantized[0] = _qk_quant.norm_rope_quantize(q, q_weight, rope, q_eps, layout, block, sm_scale=sm_scale, scale_eps=scale_eps,
                                                    weight_rows=q_rows, width=width, **options)
    return tuple(quantized)


def sol_rms_rope_(x, weight, rope, eps):
    if choose(owned_triton=_triton is not None and cuda_inference(x)):
        return _triton.rms_rope_(x, weight, rope, eps)
    from shared.sol_attn import rms_norm_rope_
    rms_norm_rope_(x, weight, rope, eps)
    torch.autograd.graph.increment_version(x)
    return x


def compact_rope_(x, cache, layout, head_start=0, dim_head=None):
    """Try compact split RoPE without materializing token-sized tables.

    False leaves x untouched for the model's layout-specific PyTorch fallback.
    Norm over all heads and its prepared statistics are deliberately external.
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
    if not choose("Compact Split RoPE", owned_triton=valid):
        return False
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
