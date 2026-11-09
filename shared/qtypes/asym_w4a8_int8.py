# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import ast
import json
import os

import torch
from optimum.quanto import QModuleMixin
from optimum.quanto.tensor.qtensor import QTensor
from optimum.quanto.tensor.qtype import qtype as _quanto_qtype, qtypes as _quanto_qtypes
from torch.utils import _pytree as pytree

from .int8_convrot import _rotate_activation


HANDLER_NAME = "asym_w4a8_int8"
HANDLER_PRIORITY = 4

_QTYPE_NAME = "asym_w4a8_int8"
if _QTYPE_NAME not in _quanto_qtypes:
    _quanto_qtypes[_QTYPE_NAME] = _quanto_qtype(_QTYPE_NAME, is_floating_point=False, bits=4, dtype=torch.int8, qmin=0, qmax=15)
_W4A8_QTYPE = _quanto_qtypes[_QTYPE_NAME]

_FORMATS = {"asym_w4a8_int8", "w4a8_int8"}
_FUSED_SPLIT_MARKER_SUFFIX = ".qweight"
_KERNEL_LOGGED = False
_FALLBACK_LOGGED = False

try:
    from torch._subclasses.fake_tensor import is_fake as _torch_is_fake_tensor
except Exception:  # pragma: no cover
    _torch_is_fake_tensor = None

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda import libdevice as tl_libdevice

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    tl_libdevice = None
    _TRITON_AVAILABLE = False


def _is_fake_tensor(tensor):
    return bool(torch.is_tensor(tensor) and _torch_is_fake_tensor is not None and _torch_is_fake_tensor(tensor))


def _decode_metadata(metadata):
    if not isinstance(metadata, dict):
        return {}
    value = metadata.get("_quantization_metadata", {})
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value.get("layers", {}) if isinstance(value, dict) else {}


def _find_layer_config(layers, base):
    config = layers.get(base)
    if config is not None:
        return config
    return next((value for name, value in layers.items() if base.endswith(name) or name.endswith(base)), {})


def _is_tensor_like(value):
    return value is not None and hasattr(value, "dtype") and hasattr(value, "shape") and hasattr(value, "ndim")


def _collect_specs(state_dict, metadata=None):
    layers = _decode_metadata(metadata)
    specs = []
    for key, s_rel in state_dict.items():
        if not key.endswith(".weight_s_rel") or not _is_tensor_like(s_rel) or s_rel.ndim != 2:
            continue
        base = key.removesuffix(".weight_s_rel")
        weight = state_dict.get(base + ".weight")
        s_channel = state_dict.get(base + ".weight_s_channel")
        codebook = state_dict.get(base + ".weight_codebook")
        correction = state_dict.get(base + ".weight_correction")
        if not _is_tensor_like(weight) or weight.dtype != torch.int8 or weight.ndim != 2:
            continue
        if not _is_tensor_like(s_channel) or s_channel.numel() != weight.shape[0]:
            continue
        if codebook is not None and (not _is_tensor_like(codebook) or codebook.numel() != 16):
            continue
        logical_k = weight.shape[1] * 2
        if s_rel.shape[0] != weight.shape[0] or s_rel.shape[1] == 0 or logical_k % s_rel.shape[1] != 0:
            continue
        inferred_group_size = logical_k // s_rel.shape[1]
        config = _find_layer_config(layers, base)
        quant_format = config.get("format", HANDLER_NAME) if isinstance(config, dict) else HANDLER_NAME
        if quant_format not in _FORMATS:
            continue
        group_size = int(config.get("group_size", inferred_group_size))
        convrot_group_size = int(config.get("convrot_groupsize", 256))
        if group_size != inferred_group_size:
            raise ValueError(f"W4A8 group size mismatch for '{base}': metadata={group_size}, tensors={inferred_group_size}")
        if logical_k % group_size != 0 or logical_k % convrot_group_size != 0:  # K counts two 4-bit weights per byte
            raise ValueError(f"Unsupported quantized checkpoint: '{base}' is not in WanGP's asymmetric W4A8 layout (read as 4-bit weights: K={logical_k}, "
                             f"group={group_size}, ConvRot={convrot_group_size}). Other formats, such as W6A8, are not supported.")
        if correction is not None and tuple(correction.shape) != (logical_k // group_size, weight.shape[0]):
            raise ValueError(f"Invalid W4A8 correction shape for '{base}': {tuple(correction.shape)}")
        specs.append({"name": base, "weight": weight, "s_rel": s_rel, "s_channel": s_channel, "codebook": codebook,
                      "correction": correction, "group_size": group_size, "convrot_group_size": convrot_group_size})
    return specs


def detect(state_dict, verboseLevel=1, metadata=None):
    specs = _collect_specs(state_dict, metadata)
    return {"matched": bool(specs), "kind": HANDLER_NAME if specs else "none",
            "details": {"count": len(specs), "names": [spec["name"] for spec in specs[:8]]}}


def convert_to_quanto(state_dict, default_dtype, verboseLevel=1, detection=None, metadata=None):
    if detection is not None and not detection.get("matched", False):
        return {"state_dict": state_dict, "quant_map": {}}
    specs = _collect_specs(state_dict, metadata)
    quant_map = {}
    for spec in specs:
        base = spec["name"]
        state_dict[base + ".weight._data"] = state_dict.pop(base + ".weight")
        state_dict[base + ".weight._s_rel"] = state_dict.pop(base + ".weight_s_rel")
        state_dict[base + ".weight._s_channel"] = state_dict.pop(base + ".weight_s_channel")
        if spec["codebook"] is not None:
            state_dict[base + ".weight._codebook"] = state_dict.pop(base + ".weight_codebook")
        if spec["correction"] is not None:
            state_dict[base + ".weight._correction"] = state_dict.pop(base + ".weight_correction")
        device = spec["weight"].device
        state_dict[base + ".weight._group_size"] = torch.tensor(spec["group_size"], dtype=torch.int32, device=device)
        state_dict[base + ".weight._convrot_group_size"] = torch.tensor(spec["convrot_group_size"], dtype=torch.int32, device=device)
        state_dict[base + _FUSED_SPLIT_MARKER_SUFFIX] = torch.empty(0, dtype=torch.uint8, device=device)
        state_dict[base + ".input_scale"] = torch.ones((), dtype=torch.float32, device=device)
        state_dict[base + ".output_scale"] = torch.ones((), dtype=torch.float32, device=device)
        qconfig = {"weights": _QTYPE_NAME, "activations": "none"}
        quant_map[base] = qconfig
        quant_map[base + ".weight"] = qconfig
    return {"state_dict": state_dict, "quant_map": quant_map}


def split_fused_weights(state_dict, fused_split_map, quantization_map=None, allowed_bases=None, default_dtype=None, verboseLevel=1):
    from mmgp import offload

    state_dict, split_bases = offload.sd_split_linear(
        state_dict,
        fused_split_map,
        split_fields={"weight._data": 0, "weight._s_rel": 0, "weight._s_channel": 0, "weight._correction": 1, "bias": 0},
        share_fields=("weight._codebook", "weight._group_size", "weight._convrot_group_size", "input_scale", "output_scale"),
        verboseLevel=verboseLevel,
        allowed_bases=allowed_bases,
        return_split_bases=True,
    )
    for base in split_bases:
        state_dict.pop(base + _FUSED_SPLIT_MARKER_SUFFIX, None)
    return state_dict, split_bases


def apply_pre_quantization(model, state_dict, quantization_map, default_dtype=None, verboseLevel=1):
    for key in [key for key, value in state_dict.items() if key.endswith(_FUSED_SPLIT_MARKER_SUFFIX) and torch.is_tensor(value) and value.numel() == 0]:
        state_dict.pop(key)
    return quantization_map or {}, []


def detect_quantization_label_from_filename(filename, verboseLevel=0):
    name = os.path.basename(str(filename)).lower() if filename else ""
    return "W4A8 INT8 ConvRot" if "w4a8" in name else ""


if _TRITON_AVAILABLE:

    @triton.jit
    def _decode_w4a8_kernel(qdata, s_rel, codebook, output, rows, k_half, group_size: tl.constexpr,
                            stride_qn, stride_qk, stride_sn, stride_sg, stride_on, stride_ok,
                            has_codebook: tl.constexpr, block_n: tl.constexpr, block_kh: tl.constexpr):
        row_offsets = tl.program_id(0) * block_n + tl.arange(0, block_n)
        byte_offsets = tl.program_id(1) * block_kh + tl.arange(0, block_kh)
        mask = (row_offsets[:, None] < rows) & (byte_offsets[None, :] < k_half)
        packed = tl.load(qdata + row_offsets[:, None] * stride_qn + byte_offsets[None, :] * stride_qk, mask=mask, other=0).to(tl.int32) & 0xFF
        low = packed & 0xF
        high = (packed >> 4) & 0xF
        groups = (2 * byte_offsets) // group_size
        scale = tl.load(s_rel + row_offsets[:, None] * stride_sn + groups[None, :] * stride_sg, mask=mask, other=0.0).to(tl.float32)
        if has_codebook:
            low_value = tl.load(codebook + low).to(tl.float32)
            high_value = tl.load(codebook + high).to(tl.float32)
        else:
            low_value = (low - 8).to(tl.float32)
            high_value = (high - 8).to(tl.float32)
        low_value = tl.maximum(tl.minimum(tl_libdevice.rint(low_value * scale), 127.0), -127.0).to(tl.int8)
        high_value = tl.maximum(tl.minimum(tl_libdevice.rint(high_value * scale), 127.0), -127.0).to(tl.int8)
        tl.store(output + row_offsets[:, None] * stride_on + (2 * byte_offsets)[None, :] * stride_ok, low_value, mask=mask)
        tl.store(output + row_offsets[:, None] * stride_on + (2 * byte_offsets + 1)[None, :] * stride_ok, high_value, mask=mask)


def _decode_w4a8_triton(qdata, s_rel, codebook, group_size, output):
    rows, k_half = qdata.shape
    grid = (triton.cdiv(rows, 16), triton.cdiv(k_half, 128))
    # Older Triton versions cannot load E4M3 FP8 tensors on Ampere GPUs.
    s_rel = s_rel.float()
    codebook_arg = codebook.float() if codebook is not None else torch.empty(1, dtype=torch.float32, device=qdata.device)
    _decode_w4a8_kernel[grid](qdata, s_rel, codebook_arg, output, rows, k_half, group_size,
                              qdata.stride(0), qdata.stride(1), s_rel.stride(0), s_rel.stride(1), output.stride(0), output.stride(1),
                              has_codebook=codebook is not None, block_n=16, block_kh=128, num_warps=4)
    return output


def _decode_w4a8_torch(qdata, s_rel, codebook, group_size, output=None):
    rows, k_half = qdata.shape
    output = torch.empty((rows, k_half * 2), dtype=torch.int8, device=qdata.device) if output is None else output
    row_block = max(1, 2 * 1024 * 1024 // (k_half * 2))
    for start in range(0, rows, row_block):
        end = min(start + row_block, rows)
        packed = qdata[start:end].to(torch.uint8)
        indices = torch.empty((end - start, k_half * 2), dtype=torch.uint8, device=qdata.device)
        indices[:, 0::2] = packed & 0xF
        indices[:, 1::2] = packed >> 4
        if codebook is None:
            values = indices.to(torch.float32).sub_(8)
        else:
            values = codebook.to(device=qdata.device, dtype=torch.float32)[indices.to(torch.long)]
        values = values.view(end - start, -1, group_size)
        values.mul_(s_rel[start:end].to(torch.float32).unsqueeze(-1)).round_().clamp_(-127, 127)
        output[start:end].copy_(values.view(end - start, -1))
    return output


def _quantize_activation(input_2d):
    scale = input_2d.abs().amax(dim=-1, keepdim=True).float().div_(127).clamp_(min=1e-30)
    quantized = input_2d.div(scale.to(input_2d.dtype)).round_().clamp_(-128, 127).to(torch.int8)
    return quantized, scale.reshape(-1)


def _workspace_rows(k, n):
    workspace_mb = max(1, int(os.environ.get("WAN2GP_W4A8_WORKSPACE_MB", "64")))
    max_rows = max(256, int(os.environ.get("WAN2GP_W4A8_WORKSPACE_ROWS", "2048")))
    rows = max(256, workspace_mb * 1024 * 1024 // k)
    rows = max(256, rows // 256 * 256)
    return min(n, rows, max_rows)


def _kernel_backend(device):
    if not _TRITON_AVAILABLE or device.type != "cuda" or torch.version.cuda is None:
        return None
    from shared.kernels import quanto_int8_triton

    with torch.cuda.device(device):
        return quanto_int8_triton if quanto_int8_triton.is_available(device) else None


def _note_kernel(chunk_rows):
    global _KERNEL_LOGGED
    if not _KERNEL_LOGGED:
        print(f"W4A8 INT8: using chunked Triton decode + INT8 GEMM (workspace rows: {chunk_rows}).")
        _KERNEL_LOGGED = True


def _note_fallback(device):
    global _FALLBACK_LOGGED
    if not _FALLBACK_LOGGED:
        print(f"W4A8 INT8: Triton INT8 kernels are incompatible with {device}; using the portable chunked fallback.")
        _FALLBACK_LOGGED = True


def _portable_int_mm(a_int8, b_int8):
    # cuBLASLt INT8 GEMMs need more than 16 rows, rows of both operands aligned to 16 bytes and the weight read column-major
    rows, k = a_int8.shape
    if a_int8.device.type == "cpu" or (a_int8.device.type == "cuda" and rows > 16 and k % 16 == 0 and b_int8.shape[0] % 16 == 0):
        return torch._int_mm(a_int8, b_int8.t())
    return torch.nn.functional.linear(a_int8.to(torch.float32), b_int8.to(torch.float32))


def _prepare_w4a8_input(input, weight):
    """Rotate and quantize bounded token tiles; retain only the prepared INT8 input shared by projections."""
    input = input.reshape(-1, input.shape[-1])
    rows, k = input.shape
    activation = torch.empty((rows, k), dtype=torch.int8, device=input.device)
    activation_scale = torch.empty(rows, dtype=torch.float32, device=input.device)
    correction_input = input.new_empty((rows, k // weight._group_size)) if weight._correction is not None else None
    tile_rows = max(1, (1 << 24) // k)
    for start in range(0, rows, tile_rows):
        end = min(start + tile_rows, rows)
        rotated = _rotate_activation(input[start:end], weight._convrot_group_size)
        quantized, scale = _quantize_activation(rotated)
        activation[start:end].copy_(quantized)
        activation_scale[start:end].copy_(scale)
        if correction_input is not None:
            correction = quantized.view(end - start, -1, weight._group_size).sum(-1, dtype=torch.int32).to(input.dtype)
            correction.mul_(scale.to(input.dtype).unsqueeze(1))
            correction_input[start:end].copy_(correction)
            del correction
        del rotated, quantized, scale
    return activation, activation_scale, correction_input, input.dtype


def _w4a8_rows(prepared, weight, bias, first, last):
    activation, activation_scale, correction_input, dtype = prepared
    rows, k = activation.shape
    chunk_rows = _workspace_rows(k, last - first)
    output = torch.empty((rows, last - first), dtype=dtype, device=activation.device)
    backend = _kernel_backend(activation.device)
    if backend is not None:
        _note_kernel(chunk_rows)
        workspace = torch.empty((chunk_rows, k), dtype=torch.int8, device=activation.device)
    else:
        _note_fallback(activation.device)
        workspace = None

    for start in range(first, last, chunk_rows):
        end = min(start + chunk_rows, last)
        tile = output[:, start - first:end - first]
        qdata = weight._data[start:end]
        s_rel = weight._s_rel[start:end]
        if backend is not None:
            decoded = _decode_w4a8_triton(qdata, s_rel, weight._codebook, weight._group_size, workspace[:end - start])
            backend.scaled_int8_mm(activation, decoded, activation_scale, weight._s_channel[start:end], out_dtype=dtype, out=tile)
        else:
            decoded = _decode_w4a8_torch(qdata, s_rel, weight._codebook, weight._group_size)
            accum = _portable_int_mm(activation, decoded)
            tile.copy_((accum.float() * activation_scale.float().unsqueeze(1) * weight._s_channel[start:end].float().unsqueeze(0)).to(dtype))
        if correction_input is not None:
            tile.addmm_(correction_input, weight._correction[:, start:end].to(dtype))
        if bias is not None:
            tile.add_(bias[start:end].to(device=activation.device, dtype=dtype))
    return output


def _w4a8_linear(input, weight, bias=None):
    if _is_fake_tensor(input):
        return input.new_empty((*input.shape[:-1], weight.shape[0]))
    prepared = _prepare_w4a8_input(input, weight)
    return _w4a8_rows(prepared, weight, bias, 0, weight.shape[0]).reshape(*input.shape[:-1], weight.shape[0])


def _qfallback(callable, *args, **kwargs):
    args, kwargs = pytree.tree_map_only(AsymW4A8Int8WeightTensor, lambda tensor: tensor.dequantize(), (args, kwargs or {}))
    return callable(*args, **kwargs)


class AsymW4A8Int8WeightTensor(QTensor):
    @staticmethod
    def create(weight_packed, s_rel, s_channel, size, stride, dtype, group_size, convrot_group_size,
               codebook=None, correction=None, device=None, requires_grad=False, qtype=_W4A8_QTYPE, axis=0):
        device = weight_packed.device if device is None else torch.device(device)
        tensors = [weight_packed, s_rel, s_channel, codebook, correction]
        tensors = [tensor.to(device) if torch.is_tensor(tensor) and tensor.device != device else tensor for tensor in tensors]
        return AsymW4A8Int8WeightTensor(qtype, axis, size, stride, *tensors, dtype, group_size, convrot_group_size, requires_grad)

    @staticmethod
    def __new__(cls, qtype, axis, size, stride, weight_packed, s_rel, s_channel, codebook, correction, dtype,
                group_size, convrot_group_size, requires_grad=False):
        return torch.Tensor._make_wrapper_subclass(cls, size, strides=stride, dtype=dtype, device=weight_packed.device, requires_grad=requires_grad)

    def __init__(self, qtype, axis, size, stride, weight_packed, s_rel, s_channel, codebook, correction, dtype,
                 group_size, convrot_group_size, requires_grad=False):
        super().__init__(qtype, axis)
        self._data = weight_packed
        self._s_rel = s_rel
        self._s_channel = s_channel
        self._codebook = codebook
        self._correction = correction
        self._group_size = int(group_size)
        self._convrot_group_size = int(convrot_group_size)

    def __repr__(self):
        return f"AsymW4A8Int8WeightTensor(shape={tuple(self.shape)}, dtype={self.dtype}, device={self.device})"

    __str__ = __repr__

    def dequantize(self, dtype=None, device=None):
        dtype = self.dtype if dtype is None else dtype
        device = self.device if device is None else torch.device(device)
        packed = self._data if self._data.device == device else self._data.to(device)
        s_rel = self._s_rel if self._s_rel.device == device else self._s_rel.to(device)
        codebook = self._codebook if self._codebook is None or self._codebook.device == device else self._codebook.to(device)
        decoded = _decode_w4a8_torch(packed, s_rel, codebook, self._group_size).to(dtype)
        decoded.mul_(self._s_channel.to(device=device, dtype=dtype).unsqueeze(1))
        if self._correction is not None:
            decoded.view(decoded.shape[0], -1, self._group_size).add_(self._correction.to(device=device, dtype=dtype).t().unsqueeze(-1))
        return _rotate_activation(decoded, self._convrot_group_size)

    def get_quantized_subtensors(self):
        tensors = [("weight_packed", self._data), ("s_rel", self._s_rel), ("s_channel", self._s_channel)]
        if self._codebook is not None:
            tensors.append(("codebook", self._codebook))
        if self._correction is not None:
            tensors.append(("correction", self._correction))
        return tensors

    def set_quantized_subtensors(self, sub_tensors):
        values = dict(sub_tensors)
        self._data = values.get("weight_packed", self._data)
        self._s_rel = values.get("s_rel", self._s_rel)
        self._s_channel = values.get("s_channel", self._s_channel)
        self._codebook = values.get("codebook", self._codebook)
        self._correction = values.get("correction", self._correction)

    def __tensor_flatten__(self):
        tensors = ["_data", "_s_rel", "_s_channel"]
        if self._codebook is not None:
            tensors.append("_codebook")
        if self._correction is not None:
            tensors.append("_correction")
        return tensors, {"qtype": self._qtype.name, "axis": str(self._axis), "size": str(list(self.size())), "stride": str(list(self.stride())),
                         "dtype": str(self.dtype), "group_size": str(self._group_size), "convrot_group_size": str(self._convrot_group_size)}

    @staticmethod
    def __tensor_unflatten__(inner_tensors, metadata, outer_size, outer_stride):
        dtype = getattr(torch, metadata["dtype"].removeprefix("torch."))
        return AsymW4A8Int8WeightTensor.create(inner_tensors["_data"], inner_tensors["_s_rel"], inner_tensors["_s_channel"],
                                                ast.literal_eval(metadata["size"]), ast.literal_eval(metadata["stride"]), dtype,
                                                int(metadata["group_size"]), int(metadata["convrot_group_size"]),
                                                codebook=inner_tensors.get("_codebook"), correction=inner_tensors.get("_correction"),
                                                qtype=_quanto_qtypes[metadata["qtype"]], axis=ast.literal_eval(metadata["axis"]))

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.nn.functional.linear:
            input, weight = args[:2]
            bias = args[2] if len(args) > 2 else kwargs.get("bias")
            if isinstance(weight, AsymW4A8Int8WeightTensor):
                return _w4a8_linear(input, weight, bias)
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, op, types, args, kwargs=None):
        packet = op.overloadpacket
        kwargs = dict(kwargs or {})
        if packet is torch.ops.aten.detach:
            tensor = args[0]
            fields, metadata = tensor.__tensor_flatten__()
            return cls.__tensor_unflatten__({name: packet(getattr(tensor, name)) for name in fields}, metadata, tensor.size(), tensor.stride())
        if packet in (torch.ops.aten._to_copy, torch.ops.aten.to):
            tensor = args[0]
            dtype = kwargs.pop("dtype", tensor.dtype)
            device = torch.device(kwargs.pop("device", tensor.device))
            if dtype != tensor.dtype:
                raise ValueError("The dtype of a W4A8 weight tensor cannot be changed")
            if tensor.device.type == "cuda" and device.type == "cpu":
                kwargs["non_blocking"] = False
            fields, metadata = tensor.__tensor_flatten__()
            moved = {name: packet(getattr(tensor, name), device=device, **kwargs) for name in fields}
            return cls.__tensor_unflatten__(moved, metadata, tensor.size(), tensor.stride())
        return _qfallback(packet, *args, **kwargs)


class QLinearAsymW4A8Int8(QModuleMixin, torch.nn.Linear):
    @classmethod
    def qcreate(cls, module, weights, activations=None, optimizer=None, device=None):
        dtype = module.weight.dtype if torch.is_tensor(module.weight) and module.weight.dtype.is_floating_point else torch.float16
        return cls(module.in_features, module.out_features, module.bias is not None, device=device, dtype=dtype,
                   weights=weights, activations=activations, optimizer=optimizer, quantize_input=True)

    def __init__(self, in_features, out_features, bias=True, device=None, dtype=None, weights=None, activations=None,
                 optimizer=None, quantize_input=True):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype, weights=weights,
                         activations=activations, optimizer=optimizer, quantize_input=quantize_input)
        self._w4a8_default_dtype = dtype
        self._mm_requires_native_linear_forward = True

    def set_default_dtype(self, dtype):
        self._w4a8_default_dtype = dtype

    @property
    def qweight(self):
        return self.weight if self.weight_qtype == _W4A8_QTYPE else super().qweight

    def forward(self, input):
        return _w4a8_linear(input, self.qweight, self.bias)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if self.weight_qtype != _W4A8_QTYPE:
            return super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)
        names = ("_data", "_s_rel", "_s_channel", "_codebook", "_correction", "_group_size", "_convrot_group_size")
        values = {name: state_dict.pop(prefix + "weight." + name, None) for name in names}
        for required in ("_data", "_s_rel", "_s_channel", "_group_size", "_convrot_group_size"):
            if values[required] is None:
                missing_keys.append(prefix + "weight." + required)
        if all(values[name] is not None for name in ("_data", "_s_rel", "_s_channel", "_group_size", "_convrot_group_size")):
            dtype = self._w4a8_default_dtype or self.weight.dtype
            weight = AsymW4A8Int8WeightTensor.create(values["_data"], values["_s_rel"], values["_s_channel"], self.weight.size(),
                                                      self.weight.stride(), dtype, int(values["_group_size"].item()),
                                                      int(values["_convrot_group_size"].item()), codebook=values["_codebook"],
                                                      correction=values["_correction"], requires_grad=False)
            self.weight = torch.nn.Parameter(weight, requires_grad=False)
        bias = state_dict.pop(prefix + "bias", None)
        if bias is not None:
            self.bias = torch.nn.Parameter(bias.to(self._w4a8_default_dtype or bias.dtype), requires_grad=False)
        device = values["_data"].device if values["_data"] is not None else self.weight.device
        self.input_scale = state_dict.pop(prefix + "input_scale", torch.ones((), dtype=torch.float32, device=device))
        self.output_scale = state_dict.pop(prefix + "output_scale", torch.ones((), dtype=torch.float32, device=device))
        return


class _W4A8RowProjection:
    """MMGP row ranges without slicing/dequantizing the wrapper weight; LoRAs are applied by MMGP."""
    @staticmethod
    def supports(module, x):
        weight = getattr(module, "weight", None)
        # Slicing the optional floating-point correction GEMM can change its accumulation/rounding.
        return isinstance(weight, AsymW4A8Int8WeightTensor) and weight._correction is None and not _is_fake_tensor(x)

    @staticmethod
    def key(module):
        return HANDLER_NAME, module.weight._convrot_group_size

    @staticmethod
    def prepare(module, x):
        return _prepare_w4a8_input(x, module.weight)

    @staticmethod
    def rows(module, prepared, start, stop):
        return _w4a8_rows(prepared, module.weight, module.bias, start, stop)


from mmgp import offload

_W4A8_ROW_PROJECTION = _W4A8RowProjection()
offload.register_row_projection(_W4A8_ROW_PROJECTION)

__all__ = ["AsymW4A8Int8WeightTensor", "QLinearAsymW4A8Int8", "convert_to_quanto", "detect"]
