# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import ast
import json
import os

import torch
from torch._subclasses.fake_tensor import is_fake
from torch.utils import _pytree as pytree

from shared.kernels import fp8_backend

from optimum.quanto import QModuleMixin
from optimum.quanto.tensor.qtensor import QTensor
from optimum.quanto.tensor.qtype import qtype as _quanto_qtype, qtypes as _quanto_qtypes


HANDLER_NAME = "fp8"
HANDLER_PRIORITY = 10

_SCALED_FP8_E4M3_QTYPE_NAME = "scaled_float8_e4m3fn"
_SCALED_FP8_E5M2_QTYPE_NAME = "scaled_float8_e5m2"


def _register_fp8_qtype(name, dtype):
    if name not in _quanto_qtypes:
        _quanto_qtypes[name] = _quanto_qtype(
            name,
            is_floating_point=True,
            bits=8,
            dtype=dtype,
            qmin=float(torch.finfo(dtype).min),
            qmax=float(torch.finfo(dtype).max),
        )
    return _quanto_qtypes[name]


_SCALED_FP8_QTYPE_E4M3 = _register_fp8_qtype(_SCALED_FP8_E4M3_QTYPE_NAME, torch.float8_e4m3fn)
_SCALED_FP8_QTYPE_E5M2 = _register_fp8_qtype(_SCALED_FP8_E5M2_QTYPE_NAME, torch.float8_e5m2)

_SCALED_FP8_QTYPE_BY_DTYPE = {
    torch.float8_e4m3fn: _SCALED_FP8_QTYPE_E4M3,
    torch.float8_e5m2: _SCALED_FP8_QTYPE_E5M2,
}
_SCALED_FP8_QTYPES = set(_SCALED_FP8_QTYPE_BY_DTYPE.values())

_FP8_RANGE = {
    torch.float8_e4m3fn: (float(torch.finfo(torch.float8_e4m3fn).min), float(torch.finfo(torch.float8_e4m3fn).max)),
    torch.float8_e5m2: (float(torch.finfo(torch.float8_e5m2).min), float(torch.finfo(torch.float8_e5m2).max)),
}

_FP8_MM_SUPPORT = {
    torch.float8_e4m3fn: False,
    torch.float8_e5m2: False,
}
_FP8_MM_PROBED = False
_SCALED_FP8_DEFAULT_DTYPE = None

_SCALED_FP8_SPLIT_FIELDS = {
    "weight": 0,
    "bias": 0,
    "scale_weight": 0,
    "weight_scale": 0,
}
_SCALED_FP8_SHARE_FIELDS = ("input_scale", "output_scale")

def _is_float8_dtype(dtype):
    return dtype in _SCALED_FP8_QTYPE_BY_DTYPE


def _get_fp8_qtype(dtype):
    return _SCALED_FP8_QTYPE_BY_DTYPE.get(dtype, None)


def _set_default_dtype_from_loader(dtype):
    global _SCALED_FP8_DEFAULT_DTYPE
    if dtype is None:
        return
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return
    _SCALED_FP8_DEFAULT_DTYPE = dtype


def _normalize_default_dtype(dtype):
    if dtype is None:
        return _SCALED_FP8_DEFAULT_DTYPE or torch.bfloat16
    if dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        return _SCALED_FP8_DEFAULT_DTYPE or torch.bfloat16
    return dtype


def _decode_comfy_quant(tensor):
    if tensor is None:
        return {}
    return json.loads(bytes(tensor.tolist()).decode("utf-8"))


def _split_fp8_scale(src, *, dim, split_sizes, context):
    if src is None or not torch.is_tensor(src):
        return None
    total = sum(split_sizes)
    if src.numel() == 1:
        return [src] * len(split_sizes)
    if src.dim() > dim and src.size(dim) == total:
        return torch.split(src, split_sizes, dim=dim)
    if src.ndim > 1 and src.size(1) == total:
        return torch.split(src, split_sizes, dim=1)
    return [src] * len(split_sizes)


def split_fused_weights(state_dict, fused_split_map, quantization_map=None, allowed_bases=None, default_dtype=None, verboseLevel=1):
    from mmgp import offload
    return offload.sd_split_linear(
        state_dict,
        fused_split_map,
        split_fields=dict(_SCALED_FP8_SPLIT_FIELDS),
        share_fields=_SCALED_FP8_SHARE_FIELDS,
        split_handlers={
            "scale_weight": _split_fp8_scale,
            "weight_scale": _split_fp8_scale,
        },
        verboseLevel=verboseLevel,
        allowed_bases=allowed_bases,
        return_split_bases=True,
    )



def _scaled_mm_available(dtype):
    if os.environ.get("WAN2GP_FORCE_FP8_FALLBACK", "").strip().lower() in ("1", "true", "yes", "on"):
        return False
    return bool(_FP8_MM_SUPPORT.get(dtype, False))


def _reshape_scale(scale, weight):
    if scale.ndim == 0 or scale.numel() == 1:
        return scale
    if scale.ndim == 1 and scale.shape[0] == weight.shape[0]:
        return scale.view(weight.shape[0], *([1] * (weight.ndim - 1)))
    if scale.ndim == 2 and scale.shape[0] == weight.shape[0] and scale.shape[1] == 1:
        return scale.view(weight.shape[0], *([1] * (weight.ndim - 1)))
    return scale


def _normalize_scaled_mm_scale(scale):
    if not torch.is_tensor(scale):
        return None
    if scale.numel() != 1:
        return None
    if scale.ndim == 0:
        return scale
    return scale.reshape(())


def _scaled_mm_weight_scale(scale, weight):
    scale = scale if scale.device == weight.device else scale.to(weight.device)
    tensor_scale = _normalize_scaled_mm_scale(scale)
    if tensor_scale is not None:
        return tensor_scale, None
    if scale.ndim == 1 and scale.shape[0] == weight.shape[0]:
        return torch.ones((), device=weight.device, dtype=torch.float32), scale
    if scale.ndim == 2 and scale.shape[0] == weight.shape[0] and scale.shape[1] == 1:
        return torch.ones((), device=weight.device, dtype=torch.float32), scale.reshape(weight.shape[0])
    return None, None


_activation_kernel = None


def _get_activation_kernel():
    global _activation_kernel
    if _activation_kernel is None:
        try:
            from shared.kernels import fp8_activation_triton as kernel
        except ImportError:  # no Triton: PyTorch path
            kernel = False
        _activation_kernel = kernel
    return _activation_kernel


def _activation_quantizer(x, fp8_dtype):
    """(quantization of x's rows to FP8, whole or by slices, with one scale from the absmax of all of x: the values of one quantization,
    bytes it allocates per element)."""
    minv, maxv = _FP8_RANGE[fp8_dtype]
    kernel = _get_activation_kernel()
    if kernel and kernel.supports(x, fp8_dtype):
        partials, scale = kernel.absmax_partials(x), torch.empty((), device=x.device, dtype=torch.float32)
        return (lambda rows: (kernel.quantize(rows, partials, scale, fp8_dtype, minv, maxv), scale)), 1
    absmax = torch.linalg.vector_norm(x, float("inf")).float()
    scale = absmax / maxv
    scale = torch.where(absmax > 0, scale, torch.ones_like(scale))
    scale_x, scale = scale.to(dtype=x.dtype), scale.reshape(()).to(torch.float32)
    return (lambda rows: ((rows / scale_x).clamp_(minv, maxv).to(fp8_dtype), scale)), 1 + x.element_size()


def _quantize_activation(x, fp8_dtype):
    return _activation_quantizer(x, fp8_dtype)[0](x)


_ROW_SLICE_BYTES = 64 << 20  # quantization scratch of the activation rows multiplied at a time
_ROW_SLICE_MIN = 16384  # rows per FP8 GEMM: fewer lower the tensor core throughput of tall-K GEMMs on large GPUs


def _scaled_mm_rows(x, weight_t, scale_b, bias, out_dtype):
    """torch._scaled_mm of x (M, K) quantized to FP8 with one scale and weight_t (K, N), by slices of rows when x spans two or more:
    the FP8 copy of x never exceeds one slice, with the values of the whole-tensor quantization."""
    quantize, scratch = _activation_quantizer(x, weight_t.dtype)
    step = max(_ROW_SLICE_MIN, _ROW_SLICE_BYTES // (x.shape[1] * scratch) // 16 * 16)
    if x.shape[0] < 2 * step:
        x8, scale_a = quantize(x)
        return torch._scaled_mm(x8, weight_t, scale_a, scale_b, bias=bias, out_dtype=out_dtype)
    step = -(-x.shape[0] // (x.shape[0] // step * 16)) * 16  # slices of even size, none below the minimum
    out = torch.empty((x.shape[0], weight_t.shape[1]), dtype=out_dtype, device=x.device)
    for start in range(0, x.shape[0], step):
        x8, scale_a = quantize(x[start:start + step])
        torch._scaled_mm(x8, weight_t, scale_a, scale_b, bias=bias, out_dtype=out_dtype, out=out[start:start + step])
        del x8
    return out


def _scaled_mm_static_ok(weight, scale):
    if weight is None or scale is None:
        return False
    if not _is_float8_dtype(weight.dtype):
        return False
    if weight.ndim != 2:
        return False
    if not weight.is_contiguous():
        return False
    if weight.shape[0] % 16 != 0 or weight.shape[1] % 16 != 0:
        return False
    scale_b, output_scale = _scaled_mm_weight_scale(scale, weight)
    return scale_b is not None or output_scale is not None




def _init_scaled_mm_support():
    global _FP8_MM_PROBED
    if _FP8_MM_PROBED:
        return
    _FP8_MM_PROBED = True
    if not torch.cuda.is_available():
        return
    support = {torch.float8_e4m3fn: False, torch.float8_e5m2: False}
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        with torch.cuda.device(device):
            a = torch.randn(1, 16, device=device, dtype=torch.float16)
            b = torch.randn(16, 16, device=device, dtype=torch.float16)
            scale = torch.ones((), device=device, dtype=torch.float32)
            for fp8_dtype in support:
                try:
                    a_fp8 = a.to(fp8_dtype)
                    b_fp8 = b.to(fp8_dtype)
                    torch._scaled_mm(a_fp8, b_fp8.t(), scale, scale, out_dtype=torch.float16)
                    support[fp8_dtype] = True
                except Exception:
                    support[fp8_dtype] = False
    except Exception:
        support = {torch.float8_e4m3fn: False, torch.float8_e5m2: False}
    _FP8_MM_SUPPORT.update(support)


def _scaled_fp8_qfallback(callable, *args, **kwargs):
    args, kwargs = pytree.tree_map_only(ScaledFP8WeightTensor, lambda x: x.dequantize(), (args, kwargs or {}))
    return callable(*args, **kwargs)


class ScaledFP8WeightTensor(QTensor):
    @staticmethod
    def create(weight, scale, size, stride, dtype, device=None, requires_grad=False, force_full_precision_mm=False):
        if scale is None:
            scale = torch.ones((), device=weight.device, dtype=torch.float32)
        qtype = _get_fp8_qtype(weight.dtype)
        if qtype is None:
            raise TypeError(f"Scaled FP8 weight requires float8 dtype, got {weight.dtype}.")
        dtype = _normalize_default_dtype(dtype)
        return ScaledFP8WeightTensor(
            qtype=qtype,
            axis=0,
            size=size,
            stride=stride,
            weight=weight,
            scale=scale,
            dtype=dtype,
            requires_grad=requires_grad,
            force_full_precision_mm=force_full_precision_mm,
        )

    @staticmethod
    def __new__(cls, qtype, axis, size, stride, weight, scale, dtype, requires_grad=False, force_full_precision_mm=False):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            size,
            strides=stride,
            dtype=dtype,
            device=weight.device,
            requires_grad=requires_grad,
        )

    def __init__(self, qtype, axis, size, stride, weight, scale, dtype, requires_grad=False, force_full_precision_mm=False):
        super().__init__(qtype, axis)
        self._data = weight
        self._scale = scale
        self._force_full_precision_mm = force_full_precision_mm
        self._scaled_mm_static_ok = _scaled_mm_static_ok(self._data, self._scale)

    def __repr__(self):
        cls_name = self.__class__.__name__
        try:
            shape = tuple(self.shape)
        except Exception:
            shape = "<?>"
        try:
            dtype = str(self.dtype).replace("torch.", "")
        except Exception:
            dtype = "<?>"
        try:
            device = str(self.device)
        except Exception:
            device = "<?>"
        qtype = getattr(self, "_qtype", None)
        qtype_name = getattr(qtype, "name", None) or str(qtype) if qtype is not None else "<?>"
        return f"{cls_name}(shape={shape}, dtype={dtype}, device={device}, qtype={qtype_name})"

    __str__ = __repr__

    def linear(self, input, bias=None):
        return scaled_fp8_linear(self, input, bias)

    def dequantize(self, dtype=None, device=None):
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        data = self._data if self._data.device == device else self._data.to(device)
        scale = self._scale if self._scale.device == device else self._scale.to(device)
        out = data.to(dtype)
        if scale.numel() == 1:
            return out * scale.to(dtype)
        return out * _reshape_scale(scale.to(dtype), out)

    def _linear_fallback(self, input, bias=None):
        qweight= self
        target_type = _normalize_default_dtype(qweight.dtype)
        weights, output_scales = qweight._data, qweight._scale
        input = input.to(target_type)
        output_scales = output_scales.to(target_type)
        in_features = input.shape[-1]
        out_features = weights.shape[0]
        output_shape = input.shape[:-1] + (out_features,)
        weights = weights.to(target_type)
        output_scales = _reshape_scale(output_scales, weights)
        weights *= output_scales
        out = torch.matmul(input.reshape(-1, in_features), weights.t())
        out = out.reshape(output_shape)
        if bias is not None:
            out += bias
        return out      

    def _linear_scaled(self, input, bias=None):
        if not torch.is_tensor(input):
            return torch.nn.functional.linear(input, self.dequantize(), bias)
        if (
            not input.is_floating_point()
            or input.dtype not in (torch.float16, torch.bfloat16, torch.float32)
            or input.ndim < 2
            or input.device.type != "cuda"
            or input.device != self._data.device
            or input.shape[-1] != self._data.shape[1]
        ):
            return torch.nn.functional.linear(input, self.dequantize(dtype=input.dtype, device=input.device), bias)

        scale_b, output_scale = _scaled_mm_weight_scale(self._scale, self._data)
        if scale_b is None:
            return torch.nn.functional.linear(input, self.dequantize(dtype=input.dtype, device=input.device), bias)

        x2d = input.reshape(-1, input.shape[-1])
        if not x2d.is_contiguous():
            x2d = x2d.contiguous()

        if x2d.shape[1] % 16 != 0 or self._data.shape[0] % 16 != 0:
            return torch.nn.functional.linear(input, self.dequantize(dtype=input.dtype, device=input.device), bias)

        scale_b = scale_b.to(device=x2d.device, dtype=torch.float32)

        bias_arg = bias
        if bias_arg is not None:
            if bias_arg.device != x2d.device:
                bias_arg = bias_arg.to(x2d.device)
            if bias_arg.dtype != input.dtype:
                bias_arg = bias_arg.to(dtype=input.dtype)
        fused_bias = bias_arg if output_scale is None and input.dtype != torch.float32 else None

        out = _scaled_mm_rows(x2d, self._data.t(), scale_b, fused_bias, input.dtype)
        if output_scale is not None:
            out *= output_scale.to(device=out.device, dtype=out.dtype).view(1, -1)
        if bias_arg is not None and fused_bias is None:
            out += bias_arg.view(1, -1)

        return out.reshape(*input.shape[:-1], self._data.shape[0])

    def get_quantized_subtensors(self):
        return [("weight", self._data), ("scale", self._scale)]

    def set_quantized_subtensors(self, sub_tensors):
        if isinstance(sub_tensors, dict):
            sub_map = sub_tensors
        else:
            sub_map = {name: tensor for name, tensor in sub_tensors}
        data = sub_map.get("weight", sub_map.get("data", None))
        if data is not None:
            self._data = data
        scale = sub_map.get("scale", None)
        if scale is not None:
            self._scale = scale
        self._scaled_mm_static_ok = _scaled_mm_static_ok(self._data, self._scale)

    def __tensor_flatten__(self):
        inner_tensors = ["_data", "_scale"]
        meta = {
            "qtype": self._qtype.name,
            "axis": str(self._axis),
            "size": str(list(self.size())),
            "stride": str(list(self.stride())),
            "dtype": str(self.dtype),
            "force_full_precision_mm": str(self._force_full_precision_mm),
        }
        return inner_tensors, meta

    @staticmethod
    def __tensor_unflatten__(inner_tensors, meta, outer_size, outer_stride):
        qtype = _quanto_qtypes[meta["qtype"]]
        axis = ast.literal_eval(meta["axis"])
        size = ast.literal_eval(meta["size"])
        stride = ast.literal_eval(meta["stride"])
        dtype_str = meta.get("dtype", "torch.float16")
        if dtype_str.startswith("torch."):
            dtype_name = dtype_str.split(".", 1)[1]
            dtype = getattr(torch, dtype_name, torch.float16)
        else:
            dtype = getattr(torch, dtype_str, torch.float16)
        dtype = _normalize_default_dtype(dtype)
        return ScaledFP8WeightTensor(
            qtype=qtype,
            axis=axis,
            size=size,
            stride=stride,
            weight=inner_tensors["_data"],
            scale=inner_tensors["_scale"],
            dtype=dtype,
            force_full_precision_mm=meta.get("force_full_precision_mm", "False") == "True",
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.nn.functional.linear:
            input = args[0] if len(args) > 0 else kwargs.get("input", None)
            weight = args[1] if len(args) > 1 else kwargs.get("weight", None)
            bias = args[2] if len(args) > 2 else kwargs.get("bias", None)
            if isinstance(weight, ScaledFP8WeightTensor):
                return scaled_fp8_linear(weight, input, bias)
        with torch._C.DisableTorchFunctionSubclass():
            return func(*args, **kwargs)

    @classmethod
    def __torch_dispatch__(cls, op, types, args, kwargs=None):
        op = op.overloadpacket
        kwargs = kwargs or {}
        if op is torch.ops.aten.linear:
            input = args[0]
            weight = args[1]
            bias = args[2] if len(args) > 2 else None
            if isinstance(weight, ScaledFP8WeightTensor):
                return scaled_fp8_linear(weight, input, bias)
        if op is torch.ops.aten.detach:
            t = args[0]
            return ScaledFP8WeightTensor.create(
                weight=op(t._data),
                scale=op(t._scale),
                size=t.size(),
                stride=t.stride(),
                dtype=t.dtype,
                device=t.device,
                requires_grad=t.requires_grad,
                force_full_precision_mm=t._force_full_precision_mm,
            )
        if op in (torch.ops.aten._to_copy, torch.ops.aten.to):
            t = args[0]
            dtype = kwargs.pop("dtype", t.dtype) if kwargs else t.dtype
            device = kwargs.pop("device", t.device) if kwargs else t.device
            if dtype != t.dtype:
                return t.dequantize(dtype=dtype, device=device)
            out_data = op(t._data, device=device, **(kwargs or {}))
            out_scale = op(t._scale, device=device, **(kwargs or {}))
            return ScaledFP8WeightTensor.create(
                weight=out_data,
                scale=out_scale,
                size=t.size(),
                stride=t.stride(),
                dtype=t.dtype,
                device=device,
                requires_grad=t.requires_grad,
                force_full_precision_mm=t._force_full_precision_mm,
            )
        return _scaled_fp8_qfallback(op, *args, **(kwargs or {}))


def scaled_fp8_linear(weight, input, bias=None):
    """The linear of a scaled FP8 weight with the FP8 math kernels selected in the configuration (shared.kernels.fp8_backend): FP8
    tensor cores (cuda, except the layers kept in full precision), the Triton weight-only kernel (triton: the same weights as
    _linear_fallback, the FP8 values times the scale rounded to the compute dtype, never materialized), or the dequantized weight
    (pytorch). A plain function, not a method of the tensor subclass: torch.compile traces it (the kernels stay opaque custom ops)."""
    if is_fake(input) and not torch.compiler.is_compiling():
        # traced below Dynamo (a compiled F.linear(x, weight): Dynamo sees the weight as a plain tensor), in a graph that PyTorch caches
        # without the selected kernels: one op that chooses them when the graph runs
        return torch.ops.wan2gp_fp8.scaled_linear(input, weight._data, weight._scale, bias, weight.dtype, weight._force_full_precision_mm)
    backend = fp8_backend.scaled_backend
    if backend == "cuda" and not weight._force_full_precision_mm and weight._scaled_mm_static_ok and _scaled_mm_available(weight._data.dtype):
        if torch.compiler.is_compiling() and input.is_cuda and input.dtype in (torch.float16, torch.bfloat16):  # the same kernels as eager
            x = input.reshape(-1, input.shape[-1])
            out = fp8_backend.matmul(x, weight._data, weight._scale, None if bias is None else bias.to(x.dtype), True)
            return out.reshape(*input.shape[:-1], weight._data.shape[0])
        return ScaledFP8WeightTensor._linear_scaled(weight, input, bias)
    if (backend != "pytorch" and torch.is_tensor(input) and input.is_cuda and input.is_floating_point() and weight._data.ndim == 2
            and input.device == weight._data.device and weight._scale.numel() in (1, weight._data.shape[0])):
        dtype = _normalize_default_dtype(weight.dtype)
        x = input.reshape(-1, input.shape[-1]).to(dtype)
        out = fp8_backend.matmul(x, weight._data, weight._scale.to(device=x.device, dtype=dtype), bias, False)
        return out.reshape(*input.shape[:-1], weight._data.shape[0])
    return ScaledFP8WeightTensor._linear_fallback(weight, input, bias)


@torch.library.custom_op("wan2gp_fp8::scaled_linear", mutates_args=())
def _scaled_linear_op(input: torch.Tensor, data: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None, dtype: torch.dtype,
                      force_full_precision_mm: bool) -> torch.Tensor:
    weight = ScaledFP8WeightTensor.create(data, scale, data.shape, data.stride(), dtype, force_full_precision_mm=force_full_precision_mm)
    return scaled_fp8_linear(weight, input, bias).to(dtype)  # cuda computes in the input's dtype


@_scaled_linear_op.register_fake
def _(input, data, scale, bias, dtype, force_full_precision_mm):
    return input.new_empty((*input.shape[:-1], data.shape[0]), dtype=dtype)


class QLinearScaledFP8(QModuleMixin, torch.nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
        weights=None,
        activations=None,
        optimizer=None,
        quantize_input=True,
    ):
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype,
            weights=weights,
            activations=activations,
            optimizer=optimizer,
            quantize_input=quantize_input,
        )
        self._scaled_fp8_default_dtype = _normalize_default_dtype(dtype)

    @classmethod
    def qcreate(cls, module, weights, activations=None, optimizer=None, device=None):
        if torch.is_tensor(module.weight) and module.weight.dtype.is_floating_point:
            weight_dtype = module.weight.dtype
        elif torch.is_tensor(getattr(module, "bias", None)) and module.bias.dtype.is_floating_point:
            weight_dtype = module.bias.dtype
        else:
            weight_dtype = torch.float16
        weight_dtype = _normalize_default_dtype(weight_dtype)
        return cls(
            module.in_features,
            module.out_features,
            module.bias is not None,
            device=device,
            dtype=weight_dtype,
            weights=weights,
            activations=activations,
            optimizer=optimizer,
            quantize_input=True,
        )

    def set_default_dtype(self, dtype):
        self._scaled_fp8_default_dtype = _normalize_default_dtype(dtype)

    @property
    def qweight(self):
        if self.weight_qtype in _SCALED_FP8_QTYPES:
            return self.weight
        return super().qweight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        qweight = self.qweight
        if getattr(qweight, "_mm_cpu_compute", False) and qweight._data.device.type == "cpu":  # a block that MMGP leaves on the CPU (cpuCompute)
            from shared.kernels import cpu_linear
            return cpu_linear.linear(qweight, input, self.bias, input.dtype)
        return scaled_fp8_linear(qweight, input, self.bias)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if self.weight_qtype not in _SCALED_FP8_QTYPES:
            return super()._load_from_state_dict(
                state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
            )

        weight_key = prefix + "weight"
        scale_key = prefix + "scale_weight"
        alt_scale_key = prefix + "weight_scale"
        comfy_quant_key = prefix + "comfy_quant"
        bias_key = prefix + "bias"
        input_scale_key = prefix + "input_scale"
        output_scale_key = prefix + "output_scale"

        weight = state_dict.pop(weight_key, None)
        scale = state_dict.pop(scale_key, None)
        alt_scale = state_dict.pop(alt_scale_key, None)
        comfy_quant = _decode_comfy_quant(state_dict.pop(comfy_quant_key, None))
        if scale is None:
            scale = alt_scale
        bias = state_dict.pop(bias_key, None)
        input_scale = state_dict.pop(input_scale_key, None)
        output_scale = state_dict.pop(output_scale_key, None)
        # input_scale isn't used in FP8 inference; drop to avoid persisting it.
        input_scale = None

        if weight is None:
            missing_keys.append(weight_key)

        target_dtype = _normalize_default_dtype(self._scaled_fp8_default_dtype or self.weight.dtype)
        if weight is not None:
            qweight = ScaledFP8WeightTensor.create(
                weight=weight,
                scale=scale,
                size=self.weight.size(),
                stride=self.weight.stride(),
                dtype=target_dtype,
                device=weight.device,
                requires_grad=False,
                force_full_precision_mm=comfy_quant.get("full_precision_matrix_mult", False),
            )
            self.weight = torch.nn.Parameter(qweight, requires_grad=False)

        if bias is not None:
            if target_dtype is not None and bias.dtype != target_dtype:
                bias = bias.to(target_dtype)
            self.bias = torch.nn.Parameter(bias)

        if torch.is_tensor(weight):
            scale_device = weight.device
        elif torch.is_tensor(self.weight):
            scale_device = self.weight.device
        elif torch.is_tensor(bias):
            scale_device = bias.device
        else:
            scale_device = torch.device("cpu")

        if input_scale is not None:
            self.input_scale = input_scale.to(scale_device)
        else:
            if not hasattr(self, "input_scale") or self.input_scale.is_meta:
                scale_dtype = self.input_scale.dtype if hasattr(self, "input_scale") else torch.float32
                self.input_scale = torch.ones((), dtype=scale_dtype, device=scale_device)

        if output_scale is not None:
            self.output_scale = output_scale.to(scale_device)
        else:
            if not hasattr(self, "output_scale") or self.output_scale.is_meta:
                scale_dtype = self.output_scale.dtype if hasattr(self, "output_scale") else torch.float32
                self.output_scale = torch.ones((), dtype=scale_dtype, device=scale_device)

        return


def _collect_fp8_specs(state_dict):
    specs = []
    for key, tensor in state_dict.items():
        if not key.endswith(".weight"):
            continue
        if not _is_float8_dtype(tensor.dtype):
            continue
        base = key[:-7]
        specs.append(
            {
                "name": base,
                "weight": tensor,
            }
        )
    return specs


def cpu_compute_supported(tensor):
    """Whether linears with this weight compute on the CPU in its storage when MMGP leaves their block there (cpuCompute)."""
    from shared.kernels import cpu_linear
    return isinstance(tensor, ScaledFP8WeightTensor) and cpu_linear.supported(tensor)


def detect(state_dict, verboseLevel=1):
    specs = _collect_fp8_specs(state_dict)
    if not specs:
        return {"matched": False, "kind": "none", "details": {}}
    names = [spec["name"] for spec in specs][:8]
    return {"matched": True, "kind": "fp8", "details": {"count": len(specs), "names": names}}


def convert_to_quanto(state_dict, default_dtype, verboseLevel=1, detection=None):
    if detection is not None and not detection.get("matched", False):
        return {"state_dict": state_dict, "quant_map": {}}
    _set_default_dtype_from_loader(default_dtype)
    if "scaled_fp8" in state_dict:
        state_dict.pop("scaled_fp8", None)
    specs = _collect_fp8_specs(state_dict)
    if not specs:
        return {"state_dict": state_dict, "quant_map": {}}
    quant_map = {}
    for spec in specs:
        qtype = _get_fp8_qtype(spec["weight"].dtype)
        if qtype is None:
            continue
        quant_map[spec["name"]] = {"weights": qtype.name, "activations": "none"}
    return {"state_dict": state_dict, "quant_map": quant_map}


def _resolve_default_dtype(model, default_dtype):
    if default_dtype is not None:
        return default_dtype
    if model is not None:
        model_dtype = getattr(model, "_dtype", None) or getattr(model, "dtype", None)
        if isinstance(model_dtype, torch.dtype):
            return model_dtype
        for _, param in model.named_parameters():
            if torch.is_tensor(param) and param.dtype.is_floating_point:
                return param.dtype
    return torch.bfloat16


def _collect_linear_param_keys(model):
    if model is None:
        return set()
    keys = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            keys.add(f"{name}.weight")
            if module.bias is not None:
                keys.add(f"{name}.bias")
    return keys


def _cast_non_linear_float8_params(model, target_dtype):
    if model is None:
        return
    from mmgp.experts import excluded_names
    expert_modules = excluded_names(model)
    for name, module in model.named_modules():
        if name in expert_modules or isinstance(module, torch.nn.Linear) or isinstance(module, QModuleMixin):
            continue
        for name, param in list(module.named_parameters(recurse=False)):
            if torch.is_tensor(param) and _is_float8_dtype(param.dtype):
                module._parameters[name] = torch.nn.Parameter(
                    param.to(dtype=target_dtype),
                    requires_grad=False,
                )
        for name, buf in list(module.named_buffers(recurse=False)):
            if torch.is_tensor(buf) and _is_float8_dtype(buf.dtype):
                module._buffers[name] = buf.to(dtype=target_dtype)


def apply_pre_quantization(model, state_dict, quantization_map, default_dtype=None, verboseLevel=1):
    _set_default_dtype_from_loader(default_dtype)
    if not quantization_map:
        quantization_map = {}

    has_float8 = False
    for tensor in state_dict.values():
        if torch.is_tensor(tensor) and _is_float8_dtype(tensor.dtype):
            has_float8 = True
            break
    if not quantization_map and not has_float8:
        return quantization_map or {}, []

    target_dtype = _resolve_default_dtype(model, default_dtype)
    linear_param_keys = _collect_linear_param_keys(model)
    from mmgp.experts import excluded_names
    expert_modules = excluded_names(model)

    to_cast = []
    for key, tensor in state_dict.items():
        if not torch.is_tensor(tensor):
            continue
        if not _is_float8_dtype(tensor.dtype):
            continue
        if key.endswith(".weight") or key.endswith(".bias"):
            module_name = key.rsplit(".", 1)[0]
        else:
            continue
        if key in linear_param_keys or module_name in expert_modules:
            continue
        to_cast.append((key, module_name, tensor))

    for key, module_name, tensor in to_cast:
        state_dict[key] = tensor.to(dtype=target_dtype)
        state_dict.pop(module_name + ".scale_weight", None)
        state_dict.pop(module_name + ".weight_scale", None)
        state_dict.pop(module_name + ".input_scale", None)
        state_dict.pop(module_name + ".output_scale", None)
        if module_name in quantization_map:
            del quantization_map[module_name]

    post_load = []
    def _post_cast(model):
        cast_dtype = _resolve_default_dtype(model, default_dtype)
        _cast_non_linear_float8_params(model, cast_dtype)

    post_load.append(_post_cast)

    return quantization_map or {}, post_load


_init_scaled_mm_support()


from shared.kernels.cpu_experts import FP8_BLOCK, NativeExpertFormat


class Fp8BlockExperts(NativeExpertFormat):
    """Routed experts stored in FP8 E4M3FN with float32 scales per block x block tile (128 by default; w13 = gate then up rows, or up
    rows only without a gate)."""
    name, native, block = "fp8_block", FP8_BLOCK, 128

    def tensors(self, experts, hidden, intermediate, gated=True, blocks=None):
        (b13, b2), fp8 = blocks or (self.block, self.block), torch.float8_e4m3fn
        rows = (2 if gated else 1) * intermediate
        return {"w13_weight": ((experts, rows, hidden), fp8), "w2_weight": ((experts, hidden, intermediate), fp8),
                "w13_scale_inv": ((experts, rows // b13, hidden // b13), torch.float32), "w2_scale_inv": ((experts, hidden // b2, intermediate // b2), torch.float32)}

    def matches(self, tensors):
        return tensors["w13_weight"].dtype == torch.float8_e4m3fn and tensors["w13_scale_inv"].dtype == torch.float32

    def dequantize_matrix(self, weight, scale, block):
        rows, cols, b = *weight.shape, block
        return (weight.float().view(rows // b, b, cols // b, b) * scale[:, None, :, None]).reshape(rows, cols)


EXPERT_FORMATS = (Fp8BlockExperts(),)
