# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import ast
import ctypes
import functools
import os
import re
import struct
import sys
import numpy as np
from dataclasses import dataclass
from enum import IntEnum
from typing import List, Optional

import torch
from torch._subclasses.fake_tensor import is_fake
from torch.utils import _pytree as pytree

from optimum.quanto import QModuleMixin, register_qmodule
from optimum.quanto.tensor.qtensor import QTensor
from optimum.quanto.tensor.qtype import qtype as _quanto_qtype, qtypes as _quanto_qtypes
from collections import OrderedDict
from tqdm import tqdm
from mmgp import allocator as _allocator, cuda_graphs as _cuda_graphs
from mmgp.offload import QEmbedding as BaseQEmbedding
from shared.kernels.triton_tuner import LaunchTuner


HANDLER_NAME = "gguf"


class PrismQuantizationType(IntEnum):
    """Prism's extension IDs, without modifying the upstream gguf package."""
    PTQ1_0 = 143


class GgmlExtraQuantizationType(IntEnum):
    """ggml types newer than the released gguf package."""
    Q2_0 = 42


_EXTRA_QUANTIZATION_SIZES = {GgmlExtraQuantizationType.Q2_0: (64, 18)}


def _quantization_type(value):
    if value == 143:
        return PrismQuantizationType(value)
    return GgmlExtraQuantizationType(value) if value in _EXTRA_QUANTIZATION_SIZES else gguf.GGMLQuantizationType(value)


def _quantization_sizes(value):
    if value == PrismQuantizationType.PTQ1_0:
        return 128, 28
    return _EXTRA_QUANTIZATION_SIZES[value] if value in _EXTRA_QUANTIZATION_SIZES else gguf.GGML_QUANT_SIZES[value]

try:
    import gguf
except Exception:
    gguf = None


_GGUF_QTYPE_NAME = "gguf"
if _GGUF_QTYPE_NAME not in _quanto_qtypes:
    _quanto_qtypes[_GGUF_QTYPE_NAME] = _quanto_qtype(
        _GGUF_QTYPE_NAME,
        is_floating_point=False,
        bits=6,
        dtype=torch.uint8,
        qmin=-32.0,
        qmax=31.0,
    )
_GGUF_QTYPE = _quanto_qtypes[_GGUF_QTYPE_NAME]

_GGUF_DEFAULT_DTYPE = None
_GGUF_LABEL_CACHE = {}
_GGUF_METADATA_CACHE = {}
_GGUF_INDEX_CACHE = {}
_GGUF_RUNTIME_LOGGED = set()
_GGUF_DEQUANTIZE_BLOCK_CHUNK = 65536  # At most 16M decoded values per standard K/IQ chunk.
_GGUF_CUDA_KERNELS_ENV = "WGP_GGUF_LLAMACPP_CUDA"
_GGUF_CUDA_KERNELS_ENABLED_CACHE = None
_GGUF_CUDA_MODULE = None
_GGUF_CUDA_LOAD_ERROR = None
_GGUF_NATIVE_BYTE_ORDER = "<" if sys.byteorder == "little" else ">"
_GGUF_ORIG_SHAPE_PREFIX = "comfy.gguf.orig_shape."
_GGUF_FAST_METADATA_SCALARS = {
    None if gguf is None else gguf.GGUFValueType.UINT8: np.uint8,
    None if gguf is None else gguf.GGUFValueType.INT8: np.int8,
    None if gguf is None else gguf.GGUFValueType.UINT16: np.uint16,
    None if gguf is None else gguf.GGUFValueType.INT16: np.int16,
    None if gguf is None else gguf.GGUFValueType.UINT32: np.uint32,
    None if gguf is None else gguf.GGUFValueType.INT32: np.int32,
    None if gguf is None else gguf.GGUFValueType.FLOAT32: np.float32,
    None if gguf is None else gguf.GGUFValueType.BOOL: np.bool_,
    None if gguf is None else gguf.GGUFValueType.UINT64: np.uint64,
    None if gguf is None else gguf.GGUFValueType.INT64: np.int64,
    None if gguf is None else gguf.GGUFValueType.FLOAT64: np.float64,
}
_GGUF_FAST_METADATA_STRUCT_FORMATS = {
    None if gguf is None else gguf.GGUFValueType.UINT8: "B",
    None if gguf is None else gguf.GGUFValueType.INT8: "b",
    None if gguf is None else gguf.GGUFValueType.UINT16: "H",
    None if gguf is None else gguf.GGUFValueType.INT16: "h",
    None if gguf is None else gguf.GGUFValueType.UINT32: "I",
    None if gguf is None else gguf.GGUFValueType.INT32: "i",
    None if gguf is None else gguf.GGUFValueType.FLOAT32: "f",
    None if gguf is None else gguf.GGUFValueType.BOOL: "?",
    None if gguf is None else gguf.GGUFValueType.UINT64: "Q",
    None if gguf is None else gguf.GGUFValueType.INT64: "q",
    None if gguf is None else gguf.GGUFValueType.FLOAT64: "d",
}
_GGUF_TYPED_TENSOR_DTYPES = {
    None if gguf is None else gguf.GGMLQuantizationType.F16: np.float16,
    None if gguf is None else gguf.GGMLQuantizationType.F32: np.float32,
    None if gguf is None else gguf.GGMLQuantizationType.F64: np.float64,
    None if gguf is None else gguf.GGMLQuantizationType.I8: np.int8,
    None if gguf is None else gguf.GGMLQuantizationType.I16: np.int16,
    None if gguf is None else gguf.GGMLQuantizationType.I32: np.int32,
    None if gguf is None else gguf.GGMLQuantizationType.I64: np.int64,
}
_GGUF_TYPED_TENSOR_TORCH_DTYPES = {
    None if gguf is None else gguf.GGMLQuantizationType.F16: torch.float16,
    None if gguf is None else gguf.GGMLQuantizationType.BF16: torch.bfloat16,
    None if gguf is None else gguf.GGMLQuantizationType.F32: torch.float32,
    None if gguf is None else gguf.GGMLQuantizationType.F64: torch.float64,
    None if gguf is None else gguf.GGMLQuantizationType.I8: torch.int8,
    None if gguf is None else gguf.GGMLQuantizationType.I16: torch.int16,
    None if gguf is None else gguf.GGMLQuantizationType.I32: torch.int32,
    None if gguf is None else gguf.GGMLQuantizationType.I64: torch.int64,
}


@dataclass(frozen=True)
class _GGUFTensorInfo:
    name: str
    tensor_type: object
    raw_shape: tuple[int, ...]
    data_offset: int
    n_elements: int
    n_bytes: int


@dataclass(frozen=True)
class _GGUFParsedIndex:
    byte_order: str
    data_alignment: int
    data_offset: int
    tensor_infos: tuple[_GGUFTensorInfo, ...]
    config: object | None
    orig_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    prism_metadata: tuple = ()


def _normalize_gguf_path(file_path):
    try:
        return os.path.normcase(os.path.abspath(file_path))
    except Exception:
        return str(file_path).lower()


def normalize(file_path):
    return _normalize_gguf_path(file_path)


def _set_default_dtype_from_loader(dtype):
    global _GGUF_DEFAULT_DTYPE
    if dtype is None:
        return
    _GGUF_DEFAULT_DTYPE = dtype


def _resolve_default_dtype(dtype, fallback=None):
    if dtype is None:
        return _GGUF_DEFAULT_DTYPE or fallback
    return dtype


def _gguf_log_once(key, message):
    if key in _GGUF_RUNTIME_LOGGED:
        return
    _GGUF_RUNTIME_LOGGED.add(key)
    print(message)


def _gguf_cuda_env_enabled():
    raw = str(os.environ.get(_GGUF_CUDA_KERNELS_ENV, "1")).strip().lower()
    return raw in ("1", "true", "yes", "y", "on")


def _validate_gguf_cuda_module(module):
    import numpy as np

    if gguf is None:
        raise RuntimeError("gguf package is unavailable")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if not module.may_support_linear_qtype_name("Q4_0"):
        raise RuntimeError("Q4_0 linear is not supported by the installed GGUF CUDA package")

    weight = np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(1, 32)
    quantized_weight = gguf.quants.quantize(weight, gguf.GGMLQuantizationType.Q4_0)
    raw_weight = torch.from_numpy(quantized_weight).contiguous().cuda()
    input_tensor = torch.linspace(-0.5, 0.5, 64, dtype=torch.float32, device="cuda").reshape(2, 32)
    fast_out = module.linear(raw_weight, "Q4_0", [1, 32], input_tensor, None, torch.float32)
    dense_weight = torch.from_numpy(gguf.quants.dequantize(quantized_weight, gguf.GGMLQuantizationType.Q4_0)).to(dtype=torch.float32)
    # On the CPU: on the GPU, cuBLAS would keep a workspace for this thread and stream.
    ref_out = torch.nn.functional.linear(input_tensor.cpu(), dense_weight, None)
    fast_out = fast_out.cpu()
    if fast_out.shape != ref_out.shape:
        raise RuntimeError(f"Q4 self-test shape mismatch: {tuple(fast_out.shape)} vs {tuple(ref_out.shape)}")
    if not torch.allclose(fast_out, ref_out, atol=2e-2, rtol=2e-2):
        max_diff = float((fast_out - ref_out).abs().max().item())
        raise RuntimeError(f"Q4 self-test output mismatch (max diff {max_diff:.6f})")


def _probe_gguf_cuda_runtime(force=False):
    global _GGUF_CUDA_KERNELS_ENABLED_CACHE, _GGUF_CUDA_MODULE, _GGUF_CUDA_LOAD_ERROR
    if not force and not _gguf_cuda_env_enabled():
        _GGUF_CUDA_KERNELS_ENABLED_CACHE = False
        _GGUF_CUDA_MODULE = None
        _GGUF_CUDA_LOAD_ERROR = None
        return False
    try:
        import llamacpp_gguf_cuda as gguf_cuda_module
        _validate_gguf_cuda_module(gguf_cuda_module)
    except Exception as exc:
        _GGUF_CUDA_KERNELS_ENABLED_CACHE = False
        _GGUF_CUDA_MODULE = None
        _GGUF_CUDA_LOAD_ERROR = f"{type(exc).__name__}: {exc}"  # not the exception: its traceback would keep the callers' frames (the models being loaded) alive
        _gguf_log_once("gguf_cuda_probe_failed", f"[GGUF][llama.cpp CUDA] kernels unavailable, using fallback")
        return False
    _GGUF_CUDA_KERNELS_ENABLED_CACHE = True
    _GGUF_CUDA_MODULE = gguf_cuda_module
    _GGUF_CUDA_LOAD_ERROR = None
    _gguf_log_once("gguf_cuda_probe_ok", "[GGUF][llama.cpp CUDA] kernels available.")
    return True


def _gguf_cuda_kernels_enabled():
    global _GGUF_CUDA_KERNELS_ENABLED_CACHE
    if _GGUF_CUDA_KERNELS_ENABLED_CACHE is not None:
        return _GGUF_CUDA_KERNELS_ENABLED_CACHE
    return _probe_gguf_cuda_runtime()


def _gguf_cuda_module():
    if not _gguf_cuda_kernels_enabled():
        return None
    return _GGUF_CUDA_MODULE


def get_gguf_compute_dtype():
    module = _gguf_cuda_module()
    if module is None:
        return torch.bfloat16
    try:
        version = tuple(int(part) for part in module.__version__.split("+", 1)[0].split(".")[:3])
    except (AttributeError, TypeError, ValueError):
        return torch.float16
    return torch.bfloat16 if version >= (1, 0, 7) else torch.float16


def set_gguf_cuda_kernels_enabled(enabled=None):
    global _GGUF_CUDA_KERNELS_ENABLED_CACHE, _GGUF_CUDA_MODULE, _GGUF_CUDA_LOAD_ERROR
    if enabled is False:
        _GGUF_CUDA_KERNELS_ENABLED_CACHE = False
        _GGUF_CUDA_MODULE = None
        _GGUF_CUDA_LOAD_ERROR = None
        return
    _GGUF_CUDA_KERNELS_ENABLED_CACHE = None
    _GGUF_CUDA_MODULE = None
    _GGUF_CUDA_LOAD_ERROR = None
    if enabled is None:
        return _probe_gguf_cuda_runtime()
    return _probe_gguf_cuda_runtime(force=True)


def _gguf_read_array(data, offset, dtype, byte_order):
    dtype = np.dtype(dtype).newbyteorder(byte_order)
    value = np.frombuffer(data, dtype=dtype, count=1, offset=offset)
    return value, offset + int(value.nbytes)


def _gguf_read_string_fast(data, offset, byte_order):
    str_len_arr, offset = _gguf_read_array(data, offset, np.uint64, byte_order)
    str_len = int(str_len_arr[0])
    raw = bytes(data[offset : offset + str_len])
    return raw.decode("utf-8"), offset + str_len


def _gguf_decode_value_fast(data, offset, raw_type, byte_order):
    value_type = gguf.GGUFValueType(int(raw_type))
    scalar_dtype = _GGUF_FAST_METADATA_SCALARS.get(value_type)
    if scalar_dtype is not None:
        value_arr, offset = _gguf_read_array(data, offset, scalar_dtype, byte_order)
        value = value_arr[0]
        return (bool(value) if value_type == gguf.GGUFValueType.BOOL else value.item()), offset
    if value_type == gguf.GGUFValueType.STRING:
        return _gguf_read_string_fast(data, offset, byte_order)
    if value_type == gguf.GGUFValueType.ARRAY:
        item_type_arr, offset = _gguf_read_array(data, offset, np.uint32, byte_order)
        item_count_arr, offset = _gguf_read_array(data, offset, np.uint64, byte_order)
        item_type = int(item_type_arr[0])
        item_count = int(item_count_arr[0])
        items = []
        for _ in range(item_count):
            item, offset = _gguf_decode_value_fast(data, offset, item_type, byte_order)
            items.append(item)
        return items, offset
    raise ValueError(f"Unsupported GGUF metadata field type: {value_type}")


def _gguf_stream_read_exact(reader, size):
    data = reader.read(int(size))
    if len(data) != int(size):
        raise EOFError("Unexpected end of GGUF metadata header.")
    return data


def _gguf_stream_unpack(reader, fmt):
    return struct.unpack(fmt, _gguf_stream_read_exact(reader, struct.calcsize(fmt)))


def _gguf_read_string_stream(reader, byte_order):
    str_len = int(_gguf_stream_unpack(reader, byte_order + "Q")[0])
    return _gguf_stream_read_exact(reader, str_len).decode("utf-8")


def _gguf_decode_value_stream(reader, raw_type, byte_order):
    value_type = gguf.GGUFValueType(int(raw_type))
    scalar_fmt = _GGUF_FAST_METADATA_STRUCT_FORMATS.get(value_type)
    if scalar_fmt is not None:
        return _gguf_stream_unpack(reader, byte_order + scalar_fmt)[0]
    if value_type == gguf.GGUFValueType.STRING:
        return _gguf_read_string_stream(reader, byte_order)
    if value_type == gguf.GGUFValueType.ARRAY:
        item_type = int(_gguf_stream_unpack(reader, byte_order + "I")[0])
        item_count = int(_gguf_stream_unpack(reader, byte_order + "Q")[0])
        return [_gguf_decode_value_stream(reader, item_type, byte_order) for _ in range(item_count)]
    raise ValueError(f"Unsupported GGUF metadata field type: {value_type}")


def _gguf_stream_skip(reader, size):
    if int(size) <= 0:
        return
    reader.seek(int(size), os.SEEK_CUR)


def _gguf_skip_value_stream(reader, raw_type, byte_order):
    value_type = gguf.GGUFValueType(int(raw_type))
    scalar_fmt = _GGUF_FAST_METADATA_STRUCT_FORMATS.get(value_type)
    if scalar_fmt is not None:
        _gguf_stream_skip(reader, struct.calcsize(byte_order + scalar_fmt))
        return
    if value_type == gguf.GGUFValueType.STRING:
        str_len = int(_gguf_stream_unpack(reader, byte_order + "Q")[0])
        _gguf_stream_skip(reader, str_len)
        return
    if value_type == gguf.GGUFValueType.ARRAY:
        item_type = int(_gguf_stream_unpack(reader, byte_order + "I")[0])
        item_count = int(_gguf_stream_unpack(reader, byte_order + "Q")[0])
        item_value_type = gguf.GGUFValueType(item_type)
        item_scalar_fmt = _GGUF_FAST_METADATA_STRUCT_FORMATS.get(item_value_type)
        if item_scalar_fmt is not None:
            _gguf_stream_skip(reader, struct.calcsize(byte_order + item_scalar_fmt) * item_count)
            return
        if item_value_type == gguf.GGUFValueType.STRING:
            for _ in range(item_count):
                item_len = int(_gguf_stream_unpack(reader, byte_order + "Q")[0])
                _gguf_stream_skip(reader, item_len)
            return
        for _ in range(item_count):
            _gguf_skip_value_stream(reader, item_type, byte_order)
        return
    raise ValueError(f"Unsupported GGUF metadata field type: {value_type}")


def _gguf_quant_byte_shape(shape, tensor_type):
    if len(shape) == 0:
        return tuple(shape)
    block_size, type_size = _quantization_sizes(tensor_type)
    last_dim = int(shape[-1])
    if block_size <= 0 or last_dim % block_size != 0:
        raise ValueError(f"Invalid GGUF tensor shape {tuple(shape)} for tensor type {getattr(tensor_type, 'name', tensor_type)}")
    return tuple(int(dim) for dim in shape[:-1]) + (last_dim // block_size * type_size,)


def _gguf_cache_identity(file_path):
    try:
        stat = os.stat(file_path)
    except OSError:
        return None
    return (int(stat.st_size), int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9))))


def _gguf_parse_index(file_path):
    if gguf is None:
        raise RuntimeError("GGUF support requires the 'gguf' package.")
    with open(file_path, "rb") as reader:
        magic = int(_gguf_stream_unpack(reader, "<I")[0])
        if magic != int(gguf.GGUF_MAGIC):
            raise ValueError("GGUF magic invalid")
        version_le = int(_gguf_stream_unpack(reader, "<I")[0])
        byte_order = ">" if (version_le & 0xFFFF) == 0 else "<"
        tensor_count = int(_gguf_stream_unpack(reader, byte_order + "Q")[0])
        kv_count = int(_gguf_stream_unpack(reader, byte_order + "Q")[0])
        data_alignment = int(getattr(gguf, "GGUF_DEFAULT_ALIGNMENT", 32))
        config = None
        orig_shapes = {}
        prism_metadata = {}
        for _ in range(kv_count):
            key = _gguf_read_string_stream(reader, byte_order)
            raw_type = int(_gguf_stream_unpack(reader, byte_order + "I")[0])
            if key == "general.alignment":
                data_alignment = int(_gguf_decode_value_stream(reader, raw_type, byte_order))
                continue
            if key == "config":
                config = _gguf_decode_value_stream(reader, raw_type, byte_order)
                continue
            if key.startswith("prism.hadamard."):
                prism_metadata[key] = _gguf_decode_value_stream(reader, raw_type, byte_order)
                continue
            if key.startswith(_GGUF_ORIG_SHAPE_PREFIX):
                value = _gguf_decode_value_stream(reader, raw_type, byte_order)
                if isinstance(value, (list, tuple)):
                    tensor_name = key[len(_GGUF_ORIG_SHAPE_PREFIX):]
                    orig_shapes[tensor_name] = tuple(int(dim) for dim in value)
                continue
            _gguf_skip_value_stream(reader, raw_type, byte_order)

        tensor_infos = []
        for _ in range(tensor_count):
            name = _gguf_read_string_stream(reader, byte_order)
            n_dims = int(_gguf_stream_unpack(reader, byte_order + "I")[0])
            raw_shape = tuple(int(_gguf_stream_unpack(reader, byte_order + "Q")[0]) for _ in range(n_dims))
            tensor_type = _quantization_type(int(_gguf_stream_unpack(reader, byte_order + "I")[0]))
            rel_offset = int(_gguf_stream_unpack(reader, byte_order + "Q")[0])
            n_elements = 1
            for dim in raw_shape:
                n_elements *= int(dim)
            block_size, type_size = _quantization_sizes(tensor_type)
            n_bytes = int(n_elements * type_size // block_size)
            tensor_infos.append(
                _GGUFTensorInfo(
                    name=name,
                    tensor_type=tensor_type,
                    raw_shape=raw_shape,
                    data_offset=rel_offset,
                    n_elements=n_elements,
                    n_bytes=n_bytes,
                )
            )

        data_offset = int(reader.tell())
        padding = data_offset % data_alignment
        if padding != 0:
            data_offset += data_alignment - padding
        tensor_infos = tuple(
            _GGUFTensorInfo(
                name=info.name,
                tensor_type=info.tensor_type,
                raw_shape=info.raw_shape,
                data_offset=data_offset + info.data_offset,
                n_elements=info.n_elements,
                n_bytes=info.n_bytes,
            )
            for info in tensor_infos
        )
        return _GGUFParsedIndex(
            byte_order=byte_order,
            data_alignment=data_alignment,
            data_offset=data_offset,
            tensor_infos=tensor_infos,
            config=config,
            orig_shapes=tuple((name, shape) for name, shape in orig_shapes.items()),
            prism_metadata=tuple(prism_metadata.items()),
        )


def _gguf_get_index(file_path):
    cache_key = _normalize_gguf_path(file_path)
    identity = _gguf_cache_identity(file_path)
    cached = _GGUF_INDEX_CACHE.get(cache_key)
    if cached is not None and cached[0] == identity:
        return cached[1]
    parsed = _gguf_parse_index(file_path)
    _GGUF_INDEX_CACHE[cache_key] = (identity, parsed)
    return parsed


def _gguf_open_tensor_numpy(mmap_data, index, tensor_info):
    logical_shape = tuple(int(dim) for dim in reversed(tensor_info.raw_shape))
    tensor_type = tensor_info.tensor_type
    np_dtype = _GGUF_TYPED_TENSOR_DTYPES.get(tensor_type)
    if np_dtype is None:
        byte_shape = _gguf_quant_byte_shape(logical_shape, tensor_type)
        return np.ndarray(shape=byte_shape, dtype=np.uint8, buffer=mmap_data, offset=tensor_info.data_offset)
    dtype = np.dtype(np_dtype)
    if index.byte_order != _GGUF_NATIVE_BYTE_ORDER:
        dtype = dtype.newbyteorder(index.byte_order)
        return np.ndarray(shape=logical_shape, dtype=dtype, buffer=mmap_data, offset=tensor_info.data_offset).byteswap().newbyteorder()
    return np.ndarray(shape=logical_shape, dtype=dtype, buffer=mmap_data, offset=tensor_info.data_offset)


def _get_lightweight_gguf_metadata(file_path):
    try:
        parsed = _gguf_get_index(file_path)
    except Exception:
        return {}
    metadata = {}
    if parsed.config is not None:
        metadata["config"] = parsed.config
    return metadata


def ensure_gguf_handler_registered() -> None:
    try:
        from mmgp import quant_router
    except Exception:
        return
    quant_router.register_handler(__name__)
    quant_router.register_file_extension("gguf", sys.modules[__name__])


ensure_gguf_handler_registered()

def get_file_metadata(file_path):
    if gguf is None:
        raise RuntimeError("GGUF support requires the 'gguf' package.")
    cache_key = _normalize_gguf_path(file_path)
    cached = _GGUF_METADATA_CACHE.get(cache_key)
    if cached is not None:
        state_dict, metadata = cached
        return OrderedDict(state_dict), dict(metadata)
    from mmgp.safetensors2 import tensor_stub
    parsed = _gguf_get_index(file_path)
    orig_shapes = dict(parsed.orig_shapes)
    state_dict = OrderedDict((tensor.name, tensor_stub(_GGUF_TYPED_TENSOR_TORCH_DTYPES.get(tensor.tensor_type, torch.uint8),
                                                       orig_shapes.get(tensor.name, tuple(reversed(tensor.raw_shape)))))
                             for tensor in parsed.tensor_infos)
    metadata = {"config": parsed.config} if parsed.config is not None else {}
    result = (state_dict, metadata)
    _GGUF_METADATA_CACHE[cache_key] = (OrderedDict(result[0]), dict(result[1]))
    return OrderedDict(result[0]), dict(result[1])


def cpu_compute_supported(tensor):
    """Whether linears with this weight compute on the CPU in its storage when MMGP leaves their block there (cpuCompute)."""
    from shared.kernels import cpu_linear
    return isinstance(tensor, GGUFWeightTensor) and cpu_linear.supported(tensor)


def _filter_state_dict_basic(state_dict, base_model_prefix, keep_prefix=False):
    new_state_dict = {}
    start = -1
    if keep_prefix:
        for k, v in state_dict.items():
            if k.startswith(base_model_prefix):
                new_state_dict[k] = v
    else:
        for k, v in state_dict.items():
            if k.startswith(base_model_prefix):
                new_start = len(base_model_prefix)
            else:
                pos = k.find("." + base_model_prefix)
                if pos < 0:
                    continue
                new_start = pos + len(base_model_prefix) + 1
            if start != -1 and start != new_start:
                new_state_dict = state_dict
                break
            start = new_start
            new_state_dict[k[start:]] = v
    return new_state_dict


def _gguf_resolve_prefix(tensor_names, prefixes):
    for prefix in prefixes:
        if any(name.startswith(prefix) for name in tensor_names):
            return prefix
    return None


def load_gguf_state_dict(
    file_path,
    filters=None,
    keep_prefixes=False,
    writable_tensors=False,
    verboseLevel=1,
    default_dtype=None,
    pin_to_memory=False,
):
    if gguf is None:
        raise RuntimeError("GGUF support requires the 'gguf' package.")
    if pin_to_memory:
        raise Exception("Pinning to memory while loading GGUF files is not supported")
    _gguf_cuda_kernels_enabled() # probes the llama.cpp CUDA kernels on first GGUF use, before any forward or graph capture

    import warnings

    def _cast_plain_tensor(torch_tensor, tensor_type):
        if tensor_type == gguf.GGMLQuantizationType.F16:
            if torch_tensor.dtype in (torch.uint8, torch.uint16):
                torch_tensor = torch_tensor.view(torch.float16)
            elif torch_tensor.dtype != torch.float16:
                torch_tensor = torch_tensor.to(torch.float16)
        elif tensor_type == gguf.GGMLQuantizationType.BF16:
            if torch_tensor.dtype in (torch.uint8, torch.uint16):
                torch_tensor = torch_tensor.view(torch.bfloat16)
            elif torch_tensor.dtype != torch.bfloat16:
                torch_tensor = torch_tensor.to(torch.bfloat16)
        elif tensor_type == gguf.GGMLQuantizationType.F32:
            if torch_tensor.dtype in (torch.uint8, torch.uint16, torch.uint32):
                torch_tensor = torch_tensor.view(torch.float32)
            elif torch_tensor.dtype != torch.float32:
                torch_tensor = torch_tensor.to(torch.float32)
        return torch_tensor

    def _tensor_type_from_dtype(dtype):
        if dtype == torch.float16:
            return gguf.GGMLQuantizationType.F16
        if dtype == torch.bfloat16:
            return gguf.GGMLQuantizationType.BF16
        if dtype == torch.float32:
            return gguf.GGMLQuantizationType.F32
        return None

    parsed = _gguf_get_index(file_path)
    orig_shapes = dict(parsed.orig_shapes)
    mmap_data = np.memmap(file_path, mode="r")
    if verboseLevel >= 2:
        try:
            from mmgp import safetensors2
            safetensors2.verboseLevel = verboseLevel
            tracker = safetensors2.MmapTracker(file_path)
            tracker.register(mmap_data, 0, 0, int(mmap_data.nbytes))
        except Exception:
            tracker = None
    tensor_names = [tensor.name for tensor in parsed.tensor_infos]
    prefix = _gguf_resolve_prefix(tensor_names, ("model.diffusion_model.", "diffusion_model."))

    state_dict = {}
    qtype_counts = {}
    for tensor in parsed.tensor_infos:
        name = tensor.name
        if prefix and name.startswith(prefix):
            name = name[len(prefix):]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            torch_tensor = torch.from_numpy(_gguf_open_tensor_numpy(mmap_data, parsed, tensor))

        shape = orig_shapes.get(tensor.name, None)
        if shape is None:
            shape = torch.Size(tuple(int(v) for v in reversed(tensor.raw_shape)))
        else:
            shape = torch.Size(tuple(int(v) for v in shape))
        if tensor.tensor_type in (
            gguf.GGMLQuantizationType.F32,
            gguf.GGMLQuantizationType.F16,
            gguf.GGMLQuantizationType.BF16,
        ):
            torch_tensor = _cast_plain_tensor(torch_tensor, tensor.tensor_type)
            torch_tensor = torch_tensor.view(*shape)
        wrapped = GGUFSourceTensor.wrap(torch_tensor, tensor_type=tensor.tensor_type, tensor_shape=shape)
        if name.endswith(".bias"):
            wrapped._gguf_bias_orig_dtype = wrapped.dtype
            wrapped._gguf_bias_orig_tensor_type = tensor.tensor_type
        state_dict[name] = wrapped
        type_name = getattr(tensor.tensor_type, "name", str(tensor.tensor_type))
        qtype_counts[type_name] = qtype_counts.get(type_name, 0) + 1

    if verboseLevel >= 2 and qtype_counts:
        print("GGUF qtypes: " + ", ".join(f"{k} ({v})" for k, v in qtype_counts.items()))

    if filters is not None:
        if not isinstance(filters, list):
            filters = [filters]
        new_sd = {}
        for one_filter in filters:
            new_sd.update(_filter_state_dict_basic(state_dict, one_filter, keep_prefixes))
        state_dict = new_sd

    if state_dict and not parsed.prism_metadata:
        for name, bias in list(state_dict.items()):
            if not name.endswith(".bias") or not torch.is_tensor(bias):
                continue
            weight_name = name[:-5] + ".weight"
            weight = state_dict.get(weight_name)
            target_dtype = None
            weight_type = getattr(weight, "tensor_type", None) if torch.is_tensor(weight) else None
            if torch.is_tensor(weight) and weight_type is None:
                target_dtype = weight.dtype if weight.dtype.is_floating_point else default_dtype
            elif weight_type in (
                gguf.GGMLQuantizationType.F16,
                gguf.GGMLQuantizationType.BF16,
                gguf.GGMLQuantizationType.F32,
            ):
                target_dtype = weight.dtype
            else:
                target_dtype = default_dtype
            if target_dtype is None or bias.dtype == target_dtype:
                continue
            casted = bias.to(target_dtype)
            if isinstance(casted, GGUFSourceTensor):
                casted._gguf_bias_orig_dtype = getattr(bias, "_gguf_bias_orig_dtype", bias.dtype)
                casted._gguf_bias_orig_tensor_type = getattr(
                    bias, "_gguf_bias_orig_tensor_type", getattr(bias, "tensor_type", None)
                )
                new_tensor_type = _tensor_type_from_dtype(target_dtype)
                if new_tensor_type is not None:
                    casted.tensor_type = new_tensor_type
                casted.tensor_shape = getattr(bias, "tensor_shape", casted.shape)
            state_dict[name] = casted

    return state_dict, None, None


def load_state_dict(*args, **kwargs):
    return load_gguf_state_dict(*args, **kwargs)


class GGUFSourceTensor(torch.Tensor):
    @staticmethod
    def wrap(tensor, *, tensor_type, tensor_shape):
        wrapped = tensor.as_subclass(GGUFSourceTensor)
        wrapped.tensor_type = tensor_type
        wrapped.tensor_shape = tensor_shape
        return wrapped

    def to(self, *args, **kwargs):
        new = super().to(*args, **kwargs)
        new.tensor_type = getattr(self, "tensor_type", None)
        new.tensor_shape = getattr(self, "tensor_shape", new.shape)
        return new

    def clone(self, *args, **kwargs):
        cloned = super().clone(*args, **kwargs).as_subclass(GGUFSourceTensor)
        cloned.tensor_type = getattr(self, "tensor_type", None)
        cloned.tensor_shape = getattr(self, "tensor_shape", cloned.shape)
        return cloned

    def detach(self, *args, **kwargs):
        detached = super().detach(*args, **kwargs).as_subclass(GGUFSourceTensor)
        detached.tensor_type = getattr(self, "tensor_type", None)
        detached.tensor_shape = getattr(self, "tensor_shape", detached.shape)
        return detached

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.split:
            src = args[0]
            dim = args[2] if len(args) > 2 else kwargs.get("dim", 0)
            with torch._C.DisableTorchFunctionSubclass():
                chunks = func(*args, **kwargs)
            tensor_type = getattr(src, "tensor_type", None)
            tensor_shape = getattr(src, "tensor_shape", None)
            if tensor_type is None or tensor_shape is None:
                return chunks
            out = []
            for chunk in chunks:
                new_shape = list(tensor_shape)
                if dim < len(new_shape) and src.shape[dim] == tensor_shape[dim]:
                    new_shape[dim] = chunk.shape[dim]
                out.append(GGUFSourceTensor.wrap(chunk, tensor_type=tensor_type, tensor_shape=tuple(new_shape)))
            return tuple(out)
        return super().__torch_function__(func, types, args, kwargs)

    def get_quantized_subtensors(self):
        return [("data", self)]

    def set_quantized_subtensors(self, sub_tensors):
        if isinstance(sub_tensors, dict):
            data = sub_tensors.get("data")
        else:
            data = dict(sub_tensors).get("data")
        if data is None or data is self:
            return
        torch.utils.swap_tensors(self, data)


def materialize_source_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if not isinstance(tensor, GGUFSourceTensor):
        return tensor
    with torch._C.DisableTorchFunctionSubclass():
        plain = torch.empty_strided(
            tuple(tensor.shape),
            tuple(tensor.stride()),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        plain.copy_(tensor)
    return plain


def materialize_module_source_tensors(module: torch.nn.Module) -> int:
    converted = 0
    for submodule in module.modules():
        for name, param in list(submodule._parameters.items()):
            if param is None or not isinstance(param, GGUFSourceTensor):
                continue
            submodule._parameters[name] = torch.nn.Parameter(
                materialize_source_tensor(param),
                requires_grad=param.requires_grad,
            )
            converted += 1
        for name, buf in list(submodule._buffers.items()):
            if buf is None or not isinstance(buf, GGUFSourceTensor):
                continue
            submodule._buffers[name] = materialize_source_tensor(buf)
            converted += 1
    return converted


def _split_gguf_tensor(src, *, dim, split_sizes, context):
    if not torch.is_tensor(src):
        return None
    tensor_type = getattr(src, "tensor_type", None)
    if tensor_type is None:
        return None
    tensor_shape = getattr(src, "tensor_shape", None) or src.shape
    total = sum(split_sizes)
    if dim >= len(tensor_shape) or tensor_shape[dim] != total:
        return None
    chunks = torch.split(src, split_sizes, dim=dim)
    out = []
    for chunk, size in zip(chunks, split_sizes):
        new_shape = list(tensor_shape)
        new_shape[dim] = size
        wrapped = GGUFSourceTensor.wrap(chunk, tensor_type=tensor_type, tensor_shape=tuple(new_shape))
        if hasattr(src, "_gguf_bias_orig_dtype"):
            wrapped._gguf_bias_orig_dtype = getattr(src, "_gguf_bias_orig_dtype")
        if hasattr(src, "_gguf_bias_orig_tensor_type"):
            wrapped._gguf_bias_orig_tensor_type = getattr(src, "_gguf_bias_orig_tensor_type")
        out.append(wrapped)
    return out


def split_fused_weights(state_dict, fused_split_map, quantization_map=None, allowed_bases=None, default_dtype=None, verboseLevel=1):
    from mmgp import offload
    return offload.sd_split_linear(
        state_dict,
        fused_split_map,
        split_fields={"weight": 0, "bias": 0},
        split_handlers={"weight": _split_gguf_tensor, "bias": _split_gguf_tensor},
        verboseLevel=verboseLevel,
        allowed_bases=allowed_bases,
        return_split_bases=True,
    )


def _is_gguf_qtype(qtype_obj):
    if gguf is None:
        return False
    if qtype_obj is None:
        return False
    return qtype_obj not in (
        gguf.GGMLQuantizationType.F32,
        gguf.GGMLQuantizationType.F16,
        gguf.GGMLQuantizationType.BF16,
    )


def _gguf_qtype_name(qtype_obj):
    if qtype_obj is None:
        return None
    return getattr(qtype_obj, "name", None) or str(qtype_obj)


_LLAMACPP_SLICE_BYTES = 64 << 20  # scratch of one slice of rows: q8_1 input and FP32 product (the row loop's Q8 ring stays within it)
GGUF_CUDA_CHUNKING = True  # native row loop (GGUF 1.0.27+) where a one-time timing on this GPU shows it beats the slices

# Switching implementations must select a fresh graph instead of replaying the old one.
_cuda_graphs.register_key(lambda: GGUF_CUDA_CHUNKING)
_ROW_LOOP_TUNER = LaunchTuner("gguf_row_loop", 1, "GGUF")


def _typed_mmq(module, target_dtype):
    """Whether the package's MMQ writes target_dtype itself (GGUF 1.0.28+: BF16, no FP32 product nor conversion pass)."""
    typed = getattr(module, "_wangp_typed_mmq", None)
    if typed is None:
        typed = module._wangp_typed_mmq = callable(getattr(module, "has_typed_mmq", None)) and bool(module.has_typed_mmq())
    return typed and target_dtype == torch.bfloat16


def _mmq_product_bytes(module, target_dtype):
    return 2 if _typed_mmq(module, target_dtype) else 4


def _sliced_cuda_linear(module, raw, qtype_name, shape, x, bias, target_dtype, step):
    # Legacy MMQ needs a Q8 input copy and a product the size of the output (FP32, or the output type when the kernels write it):
    # bound that scratch with 128-row-aligned slices.
    product_dtype = target_dtype if _typed_mmq(module, target_dtype) else torch.float32
    out = torch.empty((x.shape[0], shape[0]), dtype=target_dtype, device=x.device)
    for start in range(0, x.shape[0], step):
        out[start:start + step].copy_(module.linear(raw, qtype_name, shape, x[start:start + step], None, product_dtype))
    if bias is not None:
        out.add_(bias.to(out.device, out.dtype))
    return out


def _row_loop_selected(module, raw, qtype_name, shape, x, bias, target_dtype, step, scratch):
    # One cooperative launch per linear (one CUDA graph node, no FP32 product) where the GPU and shape are structurally supported,
    # unless the slices measured faster on this GPU. The winner depends on the row count (few long tiles favour the slices' stream-K):
    # timed once per power-of-two range of rows, on the leading rows (read only, capped), cached across sessions.
    if not (GGUF_CUDA_CHUNKING and hasattr(module, "linear_chunked") and x.dtype == target_dtype and x.is_contiguous()
            and module.linear_chunked_supported(qtype_name, shape, target_dtype, bias is not None, x.device.index)):
        return False
    rows = min(x.shape[0], max(4 * step, 16384))
    tuned = []

    def prepare_scratch():
        tuned.append(True)
        sample = x[:rows]

        def launch(config):
            if config[0] == "row_loop":
                module.linear_chunked(raw, qtype_name, shape, sample, bias, target_dtype, scratch)
            else:
                _sliced_cuda_linear(module, raw, qtype_name, shape, sample, bias, target_dtype, step)
        return launch

    bucket = 1 << (rows.bit_length() - 1)
    identity = f"{module.__version__}|{qtype_name}|{shape[1]}->{shape[0]}|{str(target_dtype)[6:]}|bias={bias is not None}|rows>={bucket}"
    with _allocator.normal_allocations():  # timing outputs must not grow a CUDA graph's pool
        choice = _ROW_LOOP_TUNER.config("linear", identity, x.device, (("row_loop",), ("slices",)), prepare_scratch, True)[0]
    if tuned:
        tqdm.write(f"[GGUF][llama.cpp CUDA] {qtype_name} {shape[1]}->{shape[0]} linear, {bucket}+ rows: {choice.replace('_', ' ')} selected after timing on this GPU.")
    return choice == "row_loop"


def _llamacpp_cuda_linear(module, raw, qtype_name, shape, input_tensor, bias, target_dtype):
    input_tensor = input_tensor.contiguous()
    out_features, in_features = shape
    rows = input_tensor.numel() // in_features
    product_bytes = _mmq_product_bytes(module, target_dtype)
    step = max(128, _LLAMACPP_SLICE_BYTES // (product_bytes * out_features + in_features * 9 // 8) // 128 * 128)
    if rows < 2 * step or module._linear_mode() != "mmq":
        return module.linear(raw, qtype_name, shape, input_tensor, bias, target_dtype)
    step = -(-rows // (-(-rows // step) * 128)) * 128  # slices of even size
    # The row loop's Q8 ring stays one 2 MiB allocator page below a slice (here at least 32 MiB): its peak is never above the slices'.
    scratch = step * (product_bytes * out_features + in_features * 9 // 8) - (2 << 20)
    x = input_tensor.reshape(rows, in_features)
    if _row_loop_selected(module, raw, qtype_name, shape, x, bias, target_dtype, step, scratch):
        out = module.linear_chunked(raw, qtype_name, shape, x, bias, target_dtype, scratch)
    else:
        out = _sliced_cuda_linear(module, raw, qtype_name, shape, x, bias, target_dtype, step)
    return out.reshape(*input_tensor.shape[:-1], out_features)


_OVERREAD_QTYPES = ("Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "IQ4_NL")  # blocks of 32 values
_OVERREAD_ROWS = 2 if torch.version.hip else 9  # input rows from which the batched kernel serves these formats (CUDA: beyond the short-batch kernels)


def _linear_many_rows_narrow(module, weight_tensor, raw, qtype_name, shape, input_tensor, bias, target_dtype):
    """llamacpp_gguf_cuda up to 1.0.29: the batched kernel walks a weight row by 256 values, so for a format of 32-value blocks whose width is not a multiple of 256 it reads past the end
    of the last rows, and the memory beside the weight decides whether the last output features are finite (or whether the read faults). Those features (one for a width of 224 or more)
    are computed from their decoded weights here; the short-batch kernels are not concerned. Later packages serve these weights themselves (has_bounded_batched_reads)."""
    rows, columns = shape
    values, size = _quantization_sizes(weight_tensor._tensor_type)
    blocks = columns // values
    apart = min(rows, -(-(8 - blocks % 8) // blocks))
    raw = raw.reshape(rows, -1)
    last = _DEQUANTIZE_FUNCTIONS[_quantization_type(weight_tensor._tensor_type)](raw[rows - apart:].reshape(-1, size), values, size, torch.float32).reshape(apart, columns)
    tail = torch.nn.functional.linear(input_tensor.float(), last, None if bias is None else bias[rows - apart:].float()).to(target_dtype)
    if rows == apart:
        return tail
    head = _llamacpp_cuda_linear(module, raw[:rows - apart], qtype_name, (rows - apart, columns), input_tensor, None if bias is None else bias[:rows - apart], target_dtype)
    return torch.cat((head, tail.to(head.dtype)), -1)


def _try_llamacpp_cuda_linear(weight_tensor, input_tensor, bias, target_dtype):
    if not _gguf_cuda_kernels_enabled():
        return None
    if not torch.is_tensor(input_tensor) or input_tensor.device.type != "cuda":
        return None
    raw = getattr(weight_tensor, "_data", None)
    if not torch.is_tensor(raw) or raw.device.type != "cuda" or not raw.is_contiguous():
        return None
    qtype_name = _gguf_qtype_name(getattr(weight_tensor, "_tensor_type", None))
    gguf_llamacpp_cuda = _gguf_cuda_module()
    if gguf_llamacpp_cuda is None or not gguf_llamacpp_cuda.may_support_linear_qtype_name(qtype_name):
        return None
    shape = tuple(getattr(weight_tensor, "_tensor_shape", weight_tensor.shape))
    # Restored orig_shape rows can cross GGUF blocks. MMQ requires whole blocks
    # per row; the fused tile decoder addresses the original flattened stream.
    if shape[1] % _quantization_sizes(weight_tensor._tensor_type)[0] and gguf_llamacpp_cuda._linear_mode() == "mmq":
        select_kernel = getattr(gguf_llamacpp_cuda, 'packed_row_kernel', None)
        flat_linear = None if select_kernel is None else select_kernel(qtype_name, input_tensor.device.index, target_dtype)
        if flat_linear is None:
            return None
        return flat_linear(input_tensor, raw, list(shape), qtype_name, _maybe_cast_bias(bias, target_dtype), target_dtype)
    if qtype_name in _OVERREAD_QTYPES and shape[1] % 256 and input_tensor.numel() // shape[1] >= _OVERREAD_ROWS and not hasattr(gguf_llamacpp_cuda, "has_bounded_batched_reads"):
        return _linear_many_rows_narrow(gguf_llamacpp_cuda, weight_tensor, raw, qtype_name, shape, input_tensor, bias, target_dtype)
    fast_out = _llamacpp_cuda_linear(gguf_llamacpp_cuda, raw, qtype_name, shape, input_tensor, bias, target_dtype)
    _gguf_log_once(f"llamacpp_cuda_linear_active_{qtype_name}", f"[GGUF][llama.cpp CUDA] linear fast path active for {qtype_name}.")
    return fast_out


def _try_llamacpp_cuda_embedding(weight_tensor, index_tensor, target_dtype):
    if not _gguf_cuda_kernels_enabled():
        return None
    if not torch.is_tensor(index_tensor) or index_tensor.device.type != "cuda":
        return None
    raw = getattr(weight_tensor, "_data", None)
    if not torch.is_tensor(raw) or raw.device.type != "cuda" or not raw.is_contiguous():
        return None
    qtype_name = _gguf_qtype_name(getattr(weight_tensor, "_tensor_type", None))
    gguf_llamacpp_cuda = _gguf_cuda_module()
    if gguf_llamacpp_cuda is None or not gguf_llamacpp_cuda.may_support_embedding_qtype_name(qtype_name):
        return None
    fast_out = gguf_llamacpp_cuda.embedding(raw, qtype_name, tuple(getattr(weight_tensor, "_tensor_shape", weight_tensor.shape)), index_tensor, target_dtype)
    _gguf_log_once(f"llamacpp_cuda_embedding_active_{qtype_name}", f"[GGUF][llama.cpp CUDA] embedding fast path active for {qtype_name}.")
    return fast_out


def _may_try_llamacpp_cuda_linear(weight_tensor, input_tensor):
    if not _gguf_cuda_kernels_enabled():
        return False
    if not torch.is_tensor(input_tensor) or input_tensor.device.type != "cuda":
        return False
    raw = getattr(weight_tensor, "_data", None)
    if not torch.is_tensor(raw) or raw.device.type != "cuda" or not raw.is_contiguous():
        return False
    qtype_name = _gguf_qtype_name(getattr(weight_tensor, "_tensor_type", None))
    try:
        gguf_llamacpp_cuda = _gguf_cuda_module()
        return callable(getattr(gguf_llamacpp_cuda, "linear", None)) and gguf_llamacpp_cuda.may_support_linear_qtype_name(qtype_name)
    except Exception:
        return False


def _may_try_llamacpp_cuda_embedding(weight_tensor, index_tensor):
    if not _gguf_cuda_kernels_enabled():
        return False
    if not torch.is_tensor(index_tensor) or index_tensor.device.type != "cuda":
        return False
    raw = getattr(weight_tensor, "_data", None)
    if not torch.is_tensor(raw) or raw.device.type != "cuda" or not raw.is_contiguous():
        return False
    qtype_name = _gguf_qtype_name(getattr(weight_tensor, "_tensor_type", None))
    try:
        gguf_llamacpp_cuda = _gguf_cuda_module()
        return callable(getattr(gguf_llamacpp_cuda, "embedding", None)) and gguf_llamacpp_cuda.may_support_embedding_qtype_name(qtype_name)
    except Exception:
        return False


def _guess_variant_from_filename(filename):
    base = os.path.basename(str(filename))
    match = re.search(r"(?i)(?:^|[_-])(PTQ1_0|Q\d+_K|Q\d+_\d|Q\d+|IQ\d+_\w+)(?:$|[_.-])", base)
    if match:
        return match.group(1).upper()
    return None


def detect_gguf_quantization_variant(file_path, verboseLevel=1):
    if gguf is None:
        return None
    try:
        parsed = _gguf_get_index(file_path)
    except Exception:
        return None
    counts = {}
    for tensor in parsed.tensor_infos:
        qtype = getattr(tensor, "tensor_type", None)
        if qtype in (
            gguf.GGMLQuantizationType.F32,
            gguf.GGMLQuantizationType.F16,
            gguf.GGMLQuantizationType.BF16,
        ):
            continue
        name = _gguf_qtype_name(qtype)
        if not name:
            continue
        counts[name] = counts.get(name, 0) + 1
    if not counts:
        return None
    return max(counts, key=counts.get)


def detect_quantization_kind_for_file(file_path, verboseLevel=1):
    if not file_path or str(file_path).lower().endswith(".gguf") is False:
        return None
    if gguf is None:
        return None
    return "gguf"


def detect_quantization_label_from_filename(filename, verboseLevel=1):
    if not filename or str(filename).lower().endswith(".gguf") is False:
        return ""
    key = _normalize_gguf_path(filename)
    cached = _GGUF_LABEL_CACHE.get(key)
    if cached:
        return cached
    variant = _guess_variant_from_filename(filename)
    if not variant and os.path.isfile(filename):
        variant = detect_gguf_quantization_variant(filename, verboseLevel=verboseLevel)
    if variant:
        label = f"GGUF-{variant}"
    else:
        label = "GGUF"
    _GGUF_LABEL_CACHE[key] = label
    return label


def _gguf_qfallback(callable, *args, **kwargs):
    args, kwargs = pytree.tree_map_only(GGUFWeightTensor, lambda x: x.dequantize(), (args, kwargs or {}))
    return callable(*args, **kwargs)


def _reshape_scale(scale, weight):
    if scale.ndim == 0 or scale.numel() == 1:
        return scale
    if scale.ndim == 1 and scale.shape[0] == weight.shape[0]:
        return scale.view(weight.shape[0], *([1] * (weight.ndim - 1)))
    return scale


def _gguf_dequantize_tensor(raw, qtype_obj, oshape, dtype=None):
    if gguf is None:
        raise RuntimeError("gguf package is required to dequantize GGUF weights.")
    if qtype_obj in (
        gguf.GGMLQuantizationType.F32,
        gguf.GGMLQuantizationType.F16,
        gguf.GGMLQuantizationType.BF16,
    ):
        out = raw.view(*oshape)
        return out.to(dtype) if dtype is not None else out
    if qtype_obj not in _DEQUANTIZE_FUNCTIONS:
        out = gguf.quants.dequantize(raw.cpu().numpy(), qtype_obj)
        out = torch.from_numpy(out)
        return out.to(device=raw.device, dtype=dtype).reshape(oshape)
    block_size, type_size = _quantization_sizes(qtype_obj)
    dequantize_blocks = _DEQUANTIZE_FUNCTIONS[qtype_obj]
    rows = raw.reshape((-1, raw.shape[-1])).view(torch.uint8)
    n_blocks = rows.numel() // type_size
    blocks = rows.reshape((n_blocks, type_size))
    if n_blocks > _GGUF_DEQUANTIZE_BLOCK_CHUNK:
        output = torch.empty((n_blocks, block_size), device=raw.device, dtype=dtype or torch.float32)
        for start in range(0, n_blocks, _GGUF_DEQUANTIZE_BLOCK_CHUNK):
            end = min(start + _GGUF_DEQUANTIZE_BLOCK_CHUNK, n_blocks)
            output[start:end].copy_(dequantize_blocks(blocks[start:end], block_size, type_size, torch.float32))
        return output.reshape(oshape)
    blocks = dequantize_blocks(blocks, block_size, type_size, torch.float32)
    return blocks.reshape(oshape).to(dtype)


def _dequantize_blocks_PTQ1_0(blocks, block_size, type_size, dtype):
    # Prism's reference base-3 codec, including the four-trit tail bytes.
    scale = blocks[:, 26:28].contiguous().view(torch.float16).to(dtype)
    pieces = []
    for offset, width, count in ((0, 16, 5), (16, 8, 5), (24, 2, 4)):
        packed = blocks[:, offset:offset + width].to(torch.int32)
        for n in range(count):
            trit = (((packed * (3 ** n)) & 255) * 3 >> 8) - 1
            pieces.append(trit.to(dtype) * scale)
    return torch.cat(pieces, dim=1)


def _maybe_cast_bias(bias, target_dtype):
    if bias is None or not torch.is_tensor(bias) or target_dtype is None:
        return bias
    if bias.dtype == target_dtype:
        return bias
    if isinstance(bias, GGUFSourceTensor):
        tensor_type = getattr(bias, "tensor_type", None)
        tensor_shape = getattr(bias, "tensor_shape", bias.shape)
        if _is_gguf_qtype(tensor_type):
            return _gguf_dequantize_tensor(bias, tensor_type, tensor_shape, dtype=target_dtype)
    return bias.to(target_dtype)


def _to_uint32(x):
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1)


def _to_uint16(x):
    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8).unsqueeze(1)


def _const_like(ref, values, dtype):
    device = ref.device if torch.is_tensor(ref) else None
    count = len(values)
    if count == 0:
        return torch.empty((0,), device=device, dtype=dtype)
    if count == 1:
        return torch.full((1,), values[0], device=device, dtype=dtype)
    step = values[1] - values[0]
    if all(values[idx] - values[idx - 1] == step for idx in range(1, count)):
        end = values[0] + step * count
        return torch.arange(values[0], end, step, device=device, dtype=dtype)
    raise ValueError("Unsupported constant pattern for GGUF dequantization.")


def _split_block_dims(blocks, *args):
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)


def _dequantize_blocks_Q8_0(blocks, block_size, type_size, dtype=None):
    d, x = _split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    x = x.view(torch.int8)
    return d * x


def _dequantize_blocks_Q5_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, m, qh, qs = _split_block_dims(blocks, 2, 2, 4)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qh = _to_uint32(qh)
    qh = qh.reshape((n_blocks, 1)) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> _const_like(d, [0, 4], torch.uint8).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape((n_blocks, -1))
    qs = ql | (qh << 4)
    return d * qs + m


def _dequantize_blocks_Q5_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, qh, qs = _split_block_dims(blocks, 2, 4)
    d = d.view(torch.float16).to(dtype)
    qh = _to_uint32(qh)
    qh = qh.reshape(n_blocks, 1) >> torch.arange(32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape(n_blocks, -1, 1, block_size // 2) >> _const_like(d, [0, 4], torch.uint8).reshape(1, 1, 2, 1)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape(n_blocks, -1)
    qs = (ql | (qh << 4)).to(torch.int8) - 16
    return d * qs


def _dequantize_blocks_Q4_1(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, m, qs = _split_block_dims(blocks, 2, 2)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> _const_like(d, [0, 4], torch.uint8).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape((n_blocks, -1))
    return d * qs + m


def _dequantize_blocks_Q4_0(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, qs = _split_block_dims(blocks, 2)
    d = d.view(torch.float16).to(dtype)
    qs = qs.reshape((n_blocks, -1, 1, block_size // 2)) >> _const_like(d, [0, 4], torch.uint8).reshape(1, 1, 2, 1)
    qs = (qs & 0x0F).reshape((n_blocks, -1))
    qs = qs.to(torch.int8) - 8
    return d * qs


QK_K = 256
K_SCALE_SIZE = 12


def _get_scale_min(scales):
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8)
    scales = scales.reshape((n_blocks, 3, 4))
    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)
    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    mn = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)
    return sc.reshape((n_blocks, 8)), mn.reshape((n_blocks, 8))


def _dequantize_blocks_Q6_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    ql, qh, scales, d = _split_block_dims(blocks, QK_K // 2, QK_K // 4, QK_K // 16)
    scales = scales.view(torch.int8).to(dtype)
    d = d.view(torch.float16).to(dtype)
    d = (d * scales).reshape((n_blocks, QK_K // 16, 1))
    ql = ql.reshape((n_blocks, -1, 1, 64)) >> _const_like(d, [0, 4], torch.uint8).reshape((1, 1, 2, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> _const_like(d, [0, 2, 4, 6], torch.uint8).reshape((1, 1, 4, 1))
    qh = (qh & 0x03).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4)).to(torch.int8) - 32
    q = q.reshape((n_blocks, QK_K // 16, -1))
    return (d * q).reshape((n_blocks, QK_K))


def _dequantize_blocks_Q5_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, dmin, scales, qh, qs = _split_block_dims(blocks, 2, 2, K_SCALE_SIZE, QK_K // 8)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = _get_scale_min(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> _const_like(d, [0, 4], torch.uint8).reshape((1, 1, 2, 1))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> _const_like(d, list(range(8)), torch.uint8).reshape((1, 1, 8, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = (qh & 0x01).reshape((n_blocks, -1, 32))
    q = ql | (qh << 4)
    return (d * q - dm).reshape((n_blocks, QK_K))


def _dequantize_blocks_Q4_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    d, dmin, scales, qs = _split_block_dims(blocks, 2, 2, K_SCALE_SIZE)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = _get_scale_min(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))
    qs = qs.reshape((n_blocks, -1, 1, 32)) >> _const_like(d, [0, 4], torch.uint8).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1, 32))
    return (d * qs - dm).reshape((n_blocks, QK_K))


def _dequantize_blocks_Q3_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    hmask, qs, scales, d = _split_block_dims(blocks, QK_K // 8, QK_K // 4, 12)
    d = d.view(torch.float16).to(dtype)
    lscales, hscales = scales[:, :8], scales[:, 8:]
    lscales = lscales.reshape((n_blocks, 1, 8)) >> _const_like(d, [0, 4], torch.uint8).reshape((1, 2, 1))
    lscales = lscales.reshape((n_blocks, 16))
    hscales = hscales.reshape((n_blocks, 1, 4)) >> _const_like(d, [0, 2, 4, 6], torch.uint8).reshape((1, 4, 1))
    hscales = hscales.reshape((n_blocks, 16))
    scales = (lscales & 0x0F) | ((hscales & 0x03) << 4)
    scales = (scales.to(torch.int8) - 32)
    dl = (d * scales).reshape((n_blocks, 16, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> _const_like(d, [0, 2, 4, 6], torch.uint8).reshape((1, 1, 4, 1))
    qh = hmask.reshape(n_blocks, -1, 1, 32) >> _const_like(d, list(range(8)), torch.uint8).reshape((1, 1, 8, 1))
    ql = ql.reshape((n_blocks, 16, QK_K // 16)) & 3
    qh = (qh.reshape((n_blocks, 16, QK_K // 16)) & 1) ^ 1
    q = (ql.to(torch.int8) - (qh << 2).to(torch.int8))
    return (dl * q).reshape((n_blocks, QK_K))


def _dequantize_blocks_Q2_K(blocks, block_size, type_size, dtype=None):
    n_blocks = blocks.shape[0]
    scales, qs, d, dmin = _split_block_dims(blocks, QK_K // 16, QK_K // 4, 2)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    dl = (d * (scales & 0xF)).reshape((n_blocks, QK_K // 16, 1))
    ml = (dmin * (scales >> 4)).reshape((n_blocks, QK_K // 16, 1))
    shift = _const_like(d, [0, 2, 4, 6], torch.uint8).reshape((1, 1, 4, 1))
    qs = (qs.reshape((n_blocks, -1, 1, 32)) >> shift) & 3
    qs = qs.reshape((n_blocks, QK_K // 16, 16))
    qs = dl * qs - ml
    return qs.reshape((n_blocks, -1))


def _iq_grid(quant, blocks):
    quant.init_grid()
    # gguf owns the immutable CPU table for the lifetime of captured H2D copies.
    return torch.from_numpy(quant.grid).to(device=blocks.device, non_blocking=True).reshape(quant.grid_shape)


def _iq_signs(indices):
    bits = (indices.unsqueeze(-1) >> torch.arange(8, device=indices.device)) & 1
    bits[..., 7] = bits[..., :7].sum(-1) & 1
    return 1 - 2 * bits.float()


def _dequantize_blocks_IQ2_XXS(blocks, block_size, type_size, dtype=None):
    n = blocks.shape[0]
    d, qs = _split_block_dims(blocks, 2)
    words = _to_uint32(qs.reshape(-1, 4)).reshape(n, 8, 2)
    db = (d.view(torch.float16).float() * (.5 + ((words[..., 1] >> 28) & 15).float()) * .25).reshape(n, 8, 1, 1)
    signs = _iq_signs((words[..., 1, None] >> (torch.arange(4, device=blocks.device) * 7)) & 127)
    indices = qs.reshape(n, 8, 8)[..., :4].long()
    values = _iq_grid(gguf.quants.IQ2_XXS, blocks).index_select(0, indices.reshape(-1)).reshape(n, 8, 4, 8)
    return (db * values * signs).reshape(n, block_size).to(dtype)


def _dequantize_blocks_IQ2_XS(blocks, block_size, type_size, dtype=None):
    n = blocks.shape[0]
    d, qs, scales = _split_block_dims(blocks, 2, 64)
    words = _to_uint16(qs.reshape(-1, 2)).reshape(n, 32)
    scales = ((scales.unsqueeze(-1) >> (torch.arange(2, device=blocks.device) * 4)) & 15).reshape(n, 16)
    db = (d.view(torch.float16).float() * (.5 + scales.float()) * .25).reshape(n, 16, 1, 1)
    signs = _iq_signs(words >> 9).reshape(n, 16, 2, 8)
    values = _iq_grid(gguf.quants.IQ2_XS, blocks).index_select(0, (words & 511).reshape(-1).long()).reshape(n, 16, 2, 8)
    return (db * values * signs).reshape(n, block_size).to(dtype)


def _dequantize_blocks_IQ2_S(blocks, block_size, type_size, dtype=None):
    n = blocks.shape[0]
    d, qs, signs, qh, scales = _split_block_dims(blocks, 2, 32, 32, 8)
    shifts = torch.arange(8, device=blocks.device, dtype=torch.uint8)
    scales = (scales.unsqueeze(-1) >> (shifts[:2] * 4) & 15).reshape(n, 16)
    db = (d.view(torch.float16).float() * (scales.float() + .5) * .25).reshape(n, 16, 1, 1)
    signs = (1 - 2 * ((signs.unsqueeze(-1) >> shifts) & 1).float()).reshape(n, 16, 2, 8)
    qh = ((qh.unsqueeze(-1) >> (shifts[:4] * 2)) & 3).reshape(n, 32).int()
    indices = qs.int() | (qh << 8)
    values = _iq_grid(gguf.quants.IQ2_S, blocks).index_select(0, indices.reshape(-1).long()).reshape(n, 16, 2, 8)
    return (db * values * signs).reshape(n, block_size).to(dtype)


def _dequantize_blocks_IQ3_XXS(blocks, block_size, type_size, dtype=None):
    n = blocks.shape[0]
    d, qs, scales = _split_block_dims(blocks, 2, 64)
    words = _to_uint32(scales.reshape(-1, 4)).reshape(n, 8)
    db = (d.view(torch.float16).float() * (.5 + ((words >> 28) & 15).float()) * .5).reshape(n, 8, 1, 1)
    signs = _iq_signs((words.unsqueeze(-1) >> (torch.arange(4, device=blocks.device) * 7)) & 127)
    values = _iq_grid(gguf.quants.IQ3_XXS, blocks).index_select(0, qs.reshape(-1).long()).reshape(n, 8, 4, 8)
    return (db * values * signs).reshape(n, block_size).to(dtype)


def _dequantize_blocks_IQ3_S(blocks, block_size, type_size, dtype=None):
    n = blocks.shape[0]
    d, qs, qh, signs, scales = _split_block_dims(blocks, 2, 64, 8, 32)
    shifts = torch.arange(8, device=blocks.device, dtype=torch.uint8)
    scales = (scales.unsqueeze(-1) >> (shifts[:2] * 4) & 15).reshape(n, 8)
    db = (d.view(torch.float16).float() * (1 + 2 * scales)).reshape(n, 8, 1, 1)
    signs = (1 - 2 * ((signs.unsqueeze(-1) >> shifts) & 1).float()).reshape(n, 8, 4, 8)
    qh = ((qh.unsqueeze(-1) >> shifts) & 1).reshape(n, 64).int()
    indices = qs.int() | (qh << 8)
    values = _iq_grid(gguf.quants.IQ3_S, blocks).index_select(0, indices.reshape(-1).long()).reshape(n, 8, 4, 8)
    return (db * values * signs).reshape(n, block_size).to(dtype)


def _dequantize_blocks_IQ1_S(blocks, block_size, type_size, dtype=None):
    n = blocks.shape[0]
    d, qs, qh = _split_block_dims(blocks, 2, 32)
    words = _to_uint16(qh.reshape(-1, 2)).reshape(n, 8)
    dl = (d.view(torch.float16).float() * (2 * ((words >> 12) & 7) + 1)).reshape(n, 8, 1, 1)
    delta = torch.where((words & 32768) == 0, .125, -.125).reshape(n, 8, 1, 1)
    high = ((words.unsqueeze(-1) >> (torch.arange(4, device=blocks.device) * 3)) & 7).reshape(n, 32)
    indices = qs.int() | (high << 8)
    values = _iq_grid(gguf.quants.IQ1_S, blocks).index_select(0, indices.reshape(-1).long()).reshape(n, 8, 4, 8)
    return (dl * (values + delta)).reshape(n, block_size).to(dtype)


def _iq4_values(indices):
    values = _IQ4_VALUES.to(device=indices.device, non_blocking=True)
    return values[indices.long()]


def _dequantize_blocks_IQ4_NL(blocks, block_size, type_size, dtype=None):
    n = blocks.shape[0]
    d, qs = _split_block_dims(blocks, 2)
    qs = (qs.reshape(n, 1, 16) >> (torch.arange(2, device=blocks.device) * 4).reshape(1, 2, 1)) & 15
    return (d.view(torch.float16).float() * _iq4_values(qs).reshape(n, 32)).to(dtype)


def _dequantize_blocks_IQ4_XS(blocks, block_size, type_size, dtype=None):
    n = blocks.shape[0]
    d, scales_h, scales_l, qs = _split_block_dims(blocks, 2, 2, 4)
    low = ((scales_l.unsqueeze(-1) >> (torch.arange(2, device=blocks.device) * 4)) & 15).reshape(n, 8)
    high = (_to_uint16(scales_h) >> (torch.arange(8, device=blocks.device) * 2)) & 3
    dl = (d.view(torch.float16).float() * ((low | (high << 4)) - 32)).reshape(n, 8, 1)
    qs = (qs.reshape(n, 8, 1, 16) >> (torch.arange(2, device=blocks.device) * 4).reshape(1, 1, 2, 1)) & 15
    return (dl * _iq4_values(qs).reshape(n, 8, 32)).reshape(n, block_size).to(dtype)


def _dequantize_blocks_Q2_0(blocks, block_size, type_size, dtype=None):
    n = blocks.shape[0]
    d, qs = _split_block_dims(blocks, 2)
    codes = (qs.unsqueeze(-1) >> (torch.arange(4, device=blocks.device, dtype=torch.uint8) * 2)) & 3
    return (d.view(torch.float16).float() * (codes.reshape(n, block_size).float() - 1)).to(dtype)


if gguf is not None:
    _IQ4_VALUES = torch.tensor(gguf.quants.IQ4_NL.kvalues, device="cpu", dtype=torch.float32)
    _DEQUANTIZE_FUNCTIONS = {
        PrismQuantizationType.PTQ1_0: _dequantize_blocks_PTQ1_0,
        GgmlExtraQuantizationType.Q2_0: _dequantize_blocks_Q2_0,
        gguf.GGMLQuantizationType.Q8_0: _dequantize_blocks_Q8_0,
        gguf.GGMLQuantizationType.Q5_1: _dequantize_blocks_Q5_1,
        gguf.GGMLQuantizationType.Q5_0: _dequantize_blocks_Q5_0,
        gguf.GGMLQuantizationType.Q4_1: _dequantize_blocks_Q4_1,
        gguf.GGMLQuantizationType.Q4_0: _dequantize_blocks_Q4_0,
        gguf.GGMLQuantizationType.Q6_K: _dequantize_blocks_Q6_K,
        gguf.GGMLQuantizationType.Q5_K: _dequantize_blocks_Q5_K,
        gguf.GGMLQuantizationType.Q4_K: _dequantize_blocks_Q4_K,
        gguf.GGMLQuantizationType.Q3_K: _dequantize_blocks_Q3_K,
        gguf.GGMLQuantizationType.Q2_K: _dequantize_blocks_Q2_K,
        gguf.GGMLQuantizationType.IQ2_S: _dequantize_blocks_IQ2_S,
        gguf.GGMLQuantizationType.IQ3_S: _dequantize_blocks_IQ3_S,
        gguf.GGMLQuantizationType.IQ1_S: _dequantize_blocks_IQ1_S,
        gguf.GGMLQuantizationType.IQ2_XS: _dequantize_blocks_IQ2_XS,
        gguf.GGMLQuantizationType.IQ2_XXS: _dequantize_blocks_IQ2_XXS,
        gguf.GGMLQuantizationType.IQ3_XXS: _dequantize_blocks_IQ3_XXS,
        gguf.GGMLQuantizationType.IQ4_NL: _dequantize_blocks_IQ4_NL,
        gguf.GGMLQuantizationType.IQ4_XS: _dequantize_blocks_IQ4_XS,
    }
else:
    _DEQUANTIZE_FUNCTIONS = {}


class GGUFWeightTensor(QTensor):
    @staticmethod
    def create(raw_tensor, size, stride, dtype, device=None, requires_grad=False, tensor_type=None, tensor_shape=None):
        if tensor_type is None:
            tensor_type = getattr(raw_tensor, "tensor_type", None)
        if tensor_shape is None:
            tensor_shape = getattr(raw_tensor, "tensor_shape", None) or size
        device = raw_tensor.device if device is None else device
        if raw_tensor.device != device:
            raw_tensor = raw_tensor.to(device)
        return GGUFWeightTensor(
            qtype=_GGUF_QTYPE,
            axis=0,
            size=size,
            stride=stride,
            raw=raw_tensor,
            tensor_type=tensor_type,
            tensor_shape=tensor_shape,
            dtype=dtype,
            requires_grad=requires_grad,
        )

    @staticmethod
    def __new__(cls, qtype, axis, size, stride, raw, tensor_type, tensor_shape, dtype, requires_grad=False):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            size,
            strides=stride,
            dtype=dtype,
            device=raw.device,
            requires_grad=requires_grad,
        )

    def __init__(self, qtype, axis, size, stride, raw, tensor_type, tensor_shape, dtype, requires_grad=False):
        super().__init__(qtype, axis)
        self._data = raw
        self._tensor_type = tensor_type
        self._tensor_shape = torch.Size(tensor_shape)
        self._gguf_default_dtype = dtype

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
        tensor_type = _gguf_qtype_name(getattr(self, "_tensor_type", None)) or "<?>"
        return (
            f"{cls_name}(shape={shape}, dtype={dtype}, device={device}, "
            f"qtype={qtype_name}, tensor_type={tensor_type})"
        )

    __str__ = __repr__

    def dequantize(self, dtype=None, device=None):
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        raw = self._data if self._data.device == device else self._data.to(device)
        return _gguf_dequantize_tensor(raw, self._tensor_type, self._tensor_shape, dtype=dtype)

    def embedding(self, input, dtype=None):
        if dtype is None:
            dtype = self.dtype
        if len(self._tensor_shape) != 2:
            return None

        num_embeddings, embedding_dim = self._tensor_shape
        _gguf_log_once(f"torch_embedding_{self._tensor_type.name}", f"[GGUF] embedding backend=PyTorch packed-row dequantization for {self._tensor_type.name}.")
        raw = self._data.reshape(num_embeddings, -1)
        flat_input = input.reshape(-1)
        if flat_input.numel() == 0:
            return torch.empty((*input.shape, embedding_dim), dtype=dtype, device=input.device)

        # GGUF blocks are row-aligned, so gather each requested packed row before dequantizing it.
        source_input = flat_input if flat_input.device == raw.device else flat_input.to(raw.device)
        raw_rows = raw.index_select(0, source_input)
        del source_input

        if raw_rows.device != input.device:
            raw_rows = raw_rows.to(input.device)
        kernels = None
        if raw_rows.is_cuda and dtype in (torch.float32, torch.bfloat16):
            from shared.kernels import expert_gguf_groups as kernels
            if not kernels.decode_rows_supported(raw_rows.device, _quantization_type(self._tensor_type)):
                kernels = None
        if kernels is not None:
            dense_rows = kernels.decode_rows(raw_rows, _quantization_type(self._tensor_type), flat_input.numel(), embedding_dim, dtype)
        else:
            dense_rows = _gguf_dequantize_tensor(
                raw_rows,
                self._tensor_type,
                (flat_input.numel(), embedding_dim),
                dtype=dtype,
            )
        del raw_rows
        if dense_rows.device != input.device:
            dense_rows = dense_rows.to(input.device)

        return dense_rows.reshape(*input.shape, embedding_dim)

    def linear(self, input, bias=None):
        if torch.is_tensor(input):
            target_dtype = _resolve_default_dtype(self._gguf_default_dtype, fallback=input.dtype)
            target_device = input.device
        else:
            target_dtype = _resolve_default_dtype(self._gguf_default_dtype, fallback=self.dtype)
            target_device = self.device
        if self._data.device.type == "cpu" and getattr(self, "_mm_cpu_compute", False):  # a block that MMGP leaves on the CPU (cpuCompute)
            from shared.kernels import cpu_linear
            return cpu_linear.linear(self, input, bias, target_dtype)
        if self._tensor_type == PrismQuantizationType.PTQ1_0 and target_device.type == "cuda":
            from .prism import ptq_linear
            return ptq_linear(input, self._data, list(self._tensor_shape), bias, target_dtype)
        if _may_try_llamacpp_cuda_linear(self, input):
            fast_out = _try_llamacpp_cuda_linear(self, input, bias, target_dtype)
            if fast_out is not None:
                return fast_out
        _gguf_log_once(f"torch_linear_{self._tensor_type.name}", f"[GGUF] linear backend=PyTorch dense weight materialization for {self._tensor_type.name}.")
        weight = self.dequantize(dtype=target_dtype, device=target_device)
        if torch.is_tensor(input) and input.dtype != weight.dtype:
            input = input.to(weight.dtype)
        bias = _maybe_cast_bias(bias, weight.dtype)
        return torch.nn.functional.linear(input, weight, bias)

    def get_quantized_subtensors(self):
        raw = GGUFSourceTensor.wrap(self._data, tensor_type=self._tensor_type, tensor_shape=self._tensor_shape)
        return [("raw_tensor", raw)]

    def set_quantized_subtensors(self, sub_tensors):
        if isinstance(sub_tensors, dict):
            sub_map = sub_tensors
        else:
            sub_map = {name: tensor for name, tensor in sub_tensors}
        data = sub_map.get("raw_tensor", sub_map.get("data", None))
        if data is not None:
            tensor_type = getattr(data, "tensor_type", None)
            if tensor_type is not None:
                self._tensor_type = tensor_type
            tensor_shape = getattr(data, "tensor_shape", None)
            if tensor_shape is not None:
                self._tensor_shape = torch.Size(tensor_shape)
            old_data = self._data
            if torch.is_tensor(old_data):
                try:
                    torch.utils.swap_tensors(old_data, data)
                    self._data = old_data
                except Exception:
                    self._data = data
            else:
                self._data = data
            if hasattr(self, "_ggml_raw_cpu"):
                self._ggml_raw_cpu = None

    def __tensor_flatten__(self):
        inner_tensors = ["_data"]
        meta = {
            "qtype": self._qtype.name,
            "axis": str(self._axis),
            "size": str(list(self.size())),
            "stride": str(list(self.stride())),
            "dtype": str(self.dtype),
            "tensor_type": _gguf_qtype_name(self._tensor_type) or "",
            "tensor_shape": str(list(self._tensor_shape)),
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
        tensor_shape = ast.literal_eval(meta.get("tensor_shape", str(list(size))))
        tensor_type = None
        if gguf is not None and meta.get("tensor_type"):
            tensor_type = getattr(PrismQuantizationType, meta["tensor_type"], None) or getattr(gguf.GGMLQuantizationType, meta["tensor_type"], None)
        return GGUFWeightTensor(
            qtype=qtype,
            axis=axis,
            size=size,
            stride=stride,
            raw=inner_tensors["_data"],
            tensor_type=tensor_type,
            tensor_shape=tensor_shape,
            dtype=dtype,
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.nn.functional.linear:
            input = args[0] if len(args) > 0 else kwargs.get("input", None)
            weight = args[1] if len(args) > 1 else kwargs.get("weight", None)
            bias = args[2] if len(args) > 2 else kwargs.get("bias", None)
            if isinstance(weight, GGUFWeightTensor):
                return weight.linear(input, bias=bias)
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
            if isinstance(weight, GGUFWeightTensor):
                return weight.linear(input, bias=bias)
        if op is torch.ops.aten.detach:
            t = args[0]
            return GGUFWeightTensor.create(
                raw_tensor=op(t._data),
                size=t.size(),
                stride=t.stride(),
                dtype=t.dtype,
                device=t.device,
                requires_grad=t.requires_grad,
                tensor_type=getattr(t, "_tensor_type", None),
                tensor_shape=getattr(t, "_tensor_shape", None),
            )
        if op in (torch.ops.aten._to_copy, torch.ops.aten.to):
            t = args[0]
            dtype = kwargs.pop("dtype", t.dtype) if kwargs else t.dtype
            device = kwargs.pop("device", t.device) if kwargs else t.device
            if dtype != t.dtype:
                return t.dequantize(dtype=dtype, device=device)
            out_data = op(t._data, device=device, **(kwargs or {}))
            return GGUFWeightTensor.create(
                raw_tensor=out_data,
                size=t.size(),
                stride=t.stride(),
                dtype=t.dtype,
                device=device,
                requires_grad=t.requires_grad,
                tensor_type=getattr(t, "_tensor_type", None),
                tensor_shape=getattr(t, "_tensor_shape", None),
            )
        return _gguf_qfallback(op, *args, **(kwargs or {}))


class _GGUFRowProjection:
    @staticmethod
    def supports(module, x):
        weight = getattr(module, "weight", None)
        return (isinstance(weight, GGUFWeightTensor) and not is_fake(x) and len(weight._tensor_shape) == 2
                and weight.shape[1] % _quantization_sizes(weight._tensor_type)[0] == 0
                and not (getattr(module, "_use_optimized_kernels", False) and x.shape[0] <= 8)
                and not (getattr(module, "_prism_decode_enabled", False) and x.shape[0] == 1))

    @staticmethod
    def key(module):
        transform = getattr(module, "prism_transform", None)
        return ("prism_rows", transform) if transform is not None else None

    @staticmethod
    def prepare(module, x):
        return module.prism_transform(x)

    @staticmethod
    def rows(module, x, start, stop):
        source = module.weight
        k = source.shape[1]
        raw = source._data.reshape(source.shape[0], -1)[start:stop].reshape(-1)
        weight = GGUFWeightTensor.create(raw, (stop - start, k), (k, 1), source.dtype,
                                         tensor_type=source._tensor_type, tensor_shape=(stop - start, k))
        return weight.linear(x, None if module.bias is None else module.bias[start:stop])


from mmgp import offload

_GGUF_ROW_PROJECTION = _GGUFRowProjection()
offload.register_row_projection(_GGUF_ROW_PROJECTION)


class GGUFFirstRowsLinear(torch.nn.Module):
    """Zero-copy linear view over a contiguous prefix of GGUF weight rows."""

    def __init__(self, source: torch.nn.Module, row_count: int):
        super().__init__()
        self.register_parameter("weight", source.weight)
        self.row_count = int(row_count)
        object.__setattr__(self, "_cached_raw", None)
        object.__setattr__(self, "_cached_weight", None)

    def clear_cache(self) -> None:
        object.__setattr__(self, "_cached_raw", None)
        object.__setattr__(self, "_cached_weight", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        source_weight = self.weight
        if not isinstance(source_weight, GGUFWeightTensor):
            raise TypeError(f"GGUFFirstRowsLinear requires GGUFWeightTensor, got {type(source_weight).__name__}.")
        full_rows, in_features = map(int, source_weight._tensor_shape)
        if not 0 < self.row_count <= full_rows or source_weight._data.numel() % full_rows:
            raise RuntimeError(f"Cannot make a {self.row_count}-row view of GGUF weight {tuple(source_weight._tensor_shape)}.")
        source_raw = source_weight._data
        if self._cached_raw is not source_raw:
            row_bytes = source_raw.numel() // full_rows
            raw = source_raw.reshape(-1)[:self.row_count * row_bytes]
            weight = GGUFWeightTensor.create(raw, (self.row_count, in_features), (in_features, 1), source_weight.dtype, tensor_type=source_weight._tensor_type, tensor_shape=(self.row_count, in_features))
            object.__setattr__(self, "_cached_raw", source_raw)
            object.__setattr__(self, "_cached_weight", weight)
        return self._cached_weight.linear(input)


_DRAFT_HEAD_GROUPS, _DRAFT_HEAD_KEEP = 256, 4  # the rows scored exactly at each draft: the 4 best of each of 256 groups of the head's one-bit copy
_DRAFT_HEAD_FLOOR = -30000.0  # the logit of the other rows: no probability, and finite under every rule applied to logits
_DRAFT_HEAD_DISABLED = "draft_candidates" in os.environ.get("WGP_FN_DISABLE", "").split(",")  # the whole head at each draft, to compare


def draft_candidates_supported(source: torch.nn.Module, row_count: int) -> bool:
    """GGUFCandidateRowsLinear serves this head: the GGUF kernel package has its selection kernel for this GPU and the head's shape fits it."""
    module = _gguf_cuda_module() if _gguf_cuda_kernels_enabled() and torch.cuda.is_available() else None
    weight = getattr(source, "weight", None)
    if _DRAFT_HEAD_DISABLED or module is None or not hasattr(module, "head_group_bits") or not module.has_head_candidates(torch.cuda.current_device()) or not isinstance(weight, GGUFWeightTensor):
        return False
    columns = int(weight._tensor_shape[1])
    return row_count % _DRAFT_HEAD_GROUPS == 0 and _DRAFT_HEAD_KEEP <= row_count // _DRAFT_HEAD_GROUPS <= 1024 and columns % 128 == 0 and columns <= 8192


class GGUFCandidateRowsLinear(GGUFFirstRowsLinear):
    """A drafter's head over the first rows of a GGUF head, in two stages. A draft needs the head's few best tokens, and the head reads
    its whole vocabulary for them at every draft (Qwen3.8 27B: 98,304 rows of Q6_K, 412 MB, 247 us on an RTX 5090). Here a one-bit copy
    of those rows (their signs and a scale per row, 61 MB, dealt once into groups by a fixed shuffle) gives the 4 best rows of each of 256
    groups, and the head's own kernel scores those 1,024 rows exactly: 69 us in all. The other tokens get a logit that leaves them no
    probability. On 6,000 inputs of the drafter's head in real passes its best token is among the rows scored in 99.9 % of the cases, and
    the tokens left out hold 0.1 % of its probability at a temperature of 0.6 (0.4 % at 1);
    a token missed is a draft less likely to be accepted, never a different output: the verification does not depend on the drafts."""

    def __init__(self, source: torch.nn.Module, row_count: int):
        super().__init__(source, row_count)
        weight = source.weight
        values, size = _quantization_sizes(weight._tensor_type)
        full_rows, columns = map(int, weight._tensor_shape)
        raw = weight._data.reshape(full_rows, -1)
        dequantize, module = _DEQUANTIZE_FUNCTIONS[_quantization_type(weight._tensor_type)], _gguf_cuda_module()
        device = torch.device("cuda")  # built on the GPU by chunks of rows (about 100 MB at a time), kept as buffers that MMGP places with the model
        bits, scales = [], []
        for start in range(0, self.row_count, 1024):
            dense = dequantize(raw[start:min(start + 1024, self.row_count)].to(device).reshape(-1, size), values, size, torch.float32).reshape(-1, columns)
            chunk_bits, chunk_scale = module.head_sign_bits(dense)
            bits.append(chunk_bits.cpu())
            scales.append(chunk_scale.cpu())
        shuffle = torch.randperm(self.row_count, generator=torch.Generator(device="cpu").manual_seed(0), device="cpu")
        self.register_buffer("candidate_bits", module.head_group_bits(torch.cat(bits)[shuffle], _DRAFT_HEAD_GROUPS), persistent=False)  # by group and by part of a row: read as contiguous runs
        self.register_buffer("candidate_scale", torch.cat(scales)[shuffle].contiguous().view(torch.int32), persistent=False)  # float32 kept as its bits: no dtype conversion touches it
        self.register_buffer("candidate_ids", shuffle.to(torch.int32), persistent=False)
        # the candidates' quantized rows, gathered at each draft: one buffer for every graph that drafts, instead of a temporary in each graph's own memory
        self.register_buffer("candidate_rows", torch.empty((_DRAFT_HEAD_GROUPS * _DRAFT_HEAD_KEEP, raw.shape[1]), dtype=torch.uint8), persistent=False)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        source_weight, bits, gathered = self.weight, self.candidate_bits, self.candidate_rows
        source_raw = source_weight._data
        if not (input.is_cuda and bits.device == input.device and gathered.device == input.device and source_raw.device == input.device and input.numel() == input.shape[-1] and input.dtype in (torch.float16, torch.bfloat16, torch.float32)):
            return super().forward(input)  # several rows at once, or weights not placed with the input: the whole head
        module = _gguf_cuda_module()
        full_rows, in_features = map(int, source_weight._tensor_shape)
        count = _DRAFT_HEAD_GROUPS * _DRAFT_HEAD_KEEP
        index = module.head_candidates(input, bits, self.candidate_scale.view(torch.float32), self.candidate_ids, _DRAFT_HEAD_GROUPS, _DRAFT_HEAD_KEEP)[0].long()
        rows = GGUFWeightTensor.create(torch.index_select(source_raw.reshape(full_rows, -1), 0, index, out=gathered), (count, in_features), (in_features, 1), source_weight.dtype, tensor_type=source_weight._tensor_type, tensor_shape=(count, in_features))
        exact = rows.linear(input)
        _gguf_log_once("draft_candidates", f"[GGUF][llama.cpp CUDA] draft head: {count} candidate rows of {self.row_count} scored at each draft (one-bit copy of the head, {bits.numel() * 4 >> 20} MB).")
        logits = torch.full((*input.shape[:-1], self.row_count), _DRAFT_HEAD_FLOOR, dtype=exact.dtype, device=exact.device)
        return logits.scatter_(-1, index.view(*([1] * (input.dim() - 1)), -1), exact)


@torch.library.custom_op("wangp_gguf::linear_fused", mutates_args=())
def linear_fused(x: torch.Tensor, raw: torch.Tensor, qtype: str, shape: List[int], bias: Optional[torch.Tensor], dtype: torch.dtype, silu_mul: bool) -> torch.Tensor:
    return _gguf_cuda_module().linear(raw, qtype, shape, x, bias, dtype, fused_output=True, silu_mul=silu_mul)


@linear_fused.register_fake
def _linear_fake(x, raw, qtype, shape, bias, dtype, silu_mul):
    return x.new_empty((*x.shape[:-1], shape[0]), dtype=dtype)


class QLinearGGUF(QModuleMixin, torch.nn.Linear):
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
        self._gguf_default_dtype = dtype

    @classmethod
    def qcreate(cls, module, weights, activations=None, optimizer=None, device=None):
        if torch.is_tensor(module.weight) and module.weight.dtype.is_floating_point:
            weight_dtype = module.weight.dtype
        elif torch.is_tensor(getattr(module, "bias", None)) and module.bias.dtype.is_floating_point:
            weight_dtype = module.bias.dtype
        else:
            weight_dtype = torch.float16
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
        self._gguf_default_dtype = dtype

    @property
    def qweight(self):
        if self.weight_qtype == _GGUF_QTYPE:
            return self.weight
        return super().qweight

    def forward(self, input: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
        # The existing list handoff carries pre-activation FFN input. Keep the
        # normal module call so MMGP loads this projection before using weights.
        silu_mul = isinstance(input, list)
        if silu_mul:
            handoff = input
            input = handoff[0]
            handoff.clear()
        qweight = self.qweight
        optimized = getattr(self, "_use_optimized_kernels", False)
        # Decode and short speculative batches (up to 8 rows): typed outputs and
        # fused SiLU-multiply avoid separate conversion and activation kernels
        # (bit-identical results). Prefill keeps the established kernels.
        if optimized and input.numel() <= 8 * input.shape[-1] and input.is_cuda and torch.version.hip is None and isinstance(qweight, GGUFWeightTensor) and qweight._data.is_cuda:
            native = _gguf_cuda_module()
            supports = getattr(native, "supports_linear_fusions", None)
            if (callable(supports) and qweight.shape[1] % _quantization_sizes(qweight._tensor_type)[0] == 0
                    and supports(qweight._tensor_type.name, input.numel() // input.shape[-1], input.device.index)):
                dtype = _resolve_default_dtype(qweight._gguf_default_dtype, fallback=input.dtype)
                return linear_fused(input, qweight._data, qweight._tensor_type.name, list(qweight._tensor_shape), self.bias, dtype, silu_mul)
        if silu_mul:
            from shared.llm_engines.lmgp.layers.activation import silu_mul as apply_silu_mul
            input = apply_silu_mul(input, use_triton=optimized)
        if isinstance(qweight, GGUFWeightTensor):
            return qweight.linear(input, bias=self.bias)
        return torch.nn.functional.linear(input, qweight, bias=self.bias)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if self.weight_qtype != _GGUF_QTYPE:
            return super()._load_from_state_dict(
                state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
            )

        weight_key = prefix + "weight"
        bias_key = prefix + "bias"
        input_scale_key = prefix + "input_scale"
        output_scale_key = prefix + "output_scale"

        weight_raw = state_dict.pop(weight_key, None)
        bias = state_dict.pop(bias_key, None)
        input_scale = state_dict.pop(input_scale_key, None)
        output_scale = state_dict.pop(output_scale_key, None)

        if weight_raw is None:
            missing_keys.append(weight_key)

        target_dtype = _resolve_default_dtype(self._gguf_default_dtype, fallback=self.weight.dtype)
        if weight_raw is not None:
            gguf_weight = GGUFWeightTensor.create(
                raw_tensor=weight_raw,
                size=self.weight.size(),
                stride=self.weight.stride(),
                dtype=target_dtype,
                device=weight_raw.device,
                requires_grad=False,
            )
            self.weight = torch.nn.Parameter(gguf_weight, requires_grad=False)

        if bias is not None:
            self.bias = torch.nn.Parameter(bias, requires_grad=False)

        if torch.is_tensor(weight_raw):
            scale_device = weight_raw.device
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


@register_qmodule(torch.nn.Conv1d)
class QConv1dGGUF(QModuleMixin, torch.nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
        weights=None,
        activations=None,
        optimizer=None,
        quantize_input=True,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
            weights=weights,
            activations=activations,
            optimizer=optimizer,
            quantize_input=quantize_input,
        )
        self._gguf_default_dtype = dtype

    @classmethod
    def qcreate(cls, module, weights, activations=None, optimizer=None, device=None):
        if torch.is_tensor(module.weight) and module.weight.dtype.is_floating_point:
            weight_dtype = module.weight.dtype
        elif torch.is_tensor(getattr(module, "bias", None)) and module.bias.dtype.is_floating_point:
            weight_dtype = module.bias.dtype
        else:
            weight_dtype = torch.float16
        return cls(
            module.in_channels,
            module.out_channels,
            module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
            bias=module.bias is not None,
            padding_mode=module.padding_mode,
            device=device,
            dtype=weight_dtype,
            weights=weights,
            activations=activations,
            optimizer=optimizer,
            quantize_input=True,
        )

    def set_default_dtype(self, dtype):
        self._gguf_default_dtype = dtype

    @property
    def qweight(self):
        if self.weight_qtype == _GGUF_QTYPE:
            return self.weight
        return super().qweight

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        qweight = self.qweight
        if isinstance(qweight, GGUFWeightTensor):
            target_dtype = _resolve_default_dtype(self._gguf_default_dtype, fallback=input.dtype)
            weight = qweight.dequantize(dtype=target_dtype, device=input.device)
            bias = _maybe_cast_bias(self.bias, weight.dtype)
            if input.dtype != weight.dtype:
                input = input.to(weight.dtype)
            return torch.nn.functional.conv1d(
                input,
                weight,
                bias=bias,
                stride=self.stride,
                padding=self.padding,
                dilation=self.dilation,
                groups=self.groups,
            )
        return torch.nn.functional.conv1d(
            input,
            qweight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if self.weight_qtype != _GGUF_QTYPE:
            return super()._load_from_state_dict(
                state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
            )

        weight_key = prefix + "weight"
        bias_key = prefix + "bias"
        input_scale_key = prefix + "input_scale"
        output_scale_key = prefix + "output_scale"

        weight_raw = state_dict.pop(weight_key, None)
        bias = state_dict.pop(bias_key, None)
        input_scale = state_dict.pop(input_scale_key, None)
        output_scale = state_dict.pop(output_scale_key, None)

        if weight_raw is None:
            missing_keys.append(weight_key)

        target_dtype = _resolve_default_dtype(self._gguf_default_dtype, fallback=self.weight.dtype)
        if weight_raw is not None:
            gguf_weight = GGUFWeightTensor.create(
                raw_tensor=weight_raw,
                size=self.weight.size(),
                stride=self.weight.stride(),
                dtype=target_dtype,
                device=weight_raw.device,
                requires_grad=False,
            )
            self.weight = torch.nn.Parameter(gguf_weight, requires_grad=False)

        if bias is not None:
            self.bias = torch.nn.Parameter(bias, requires_grad=False)

        if torch.is_tensor(weight_raw):
            scale_device = weight_raw.device
        elif torch.is_tensor(self.weight):
            scale_device = self.weight.device
        elif torch.is_tensor(bias):
            scale_device = bias.device
        else:
            scale_device = torch.device("cpu")

        if input_scale is not None:
            self.input_scale = input_scale.to(scale_device)
        elif not hasattr(self, "input_scale") or self.input_scale.is_meta:
            scale_dtype = self.input_scale.dtype if hasattr(self, "input_scale") else torch.float32
            self.input_scale = torch.ones((), dtype=scale_dtype, device=scale_device)

        if output_scale is not None:
            self.output_scale = output_scale.to(scale_device)
        elif not hasattr(self, "output_scale") or self.output_scale.is_meta:
            scale_dtype = self.output_scale.dtype if hasattr(self, "output_scale") else torch.float32
            self.output_scale = torch.ones((), dtype=scale_dtype, device=scale_device)

        return


@register_qmodule(torch.nn.Embedding)
class QEmbedding(BaseQEmbedding):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        padding_idx=None,
        max_norm=None,
        norm_type=2.0,
        scale_grad_by_freq=False,
        sparse=False,
        device=None,
        dtype=None,
        weights=None,
        activations=None,
        optimizer=None,
        quantize_input=True,
    ):
        super().__init__(
            num_embeddings,
            embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            device=device,
            dtype=dtype,
            weights=weights,
            activations=activations,
            optimizer=optimizer,
            quantize_input=quantize_input,
        )
        self._gguf_default_dtype = dtype

    @classmethod
    def from_module(cls, module, weights=None, activations=None, optimizer=None):
        qmodule = super().from_module(module, weights, activations, optimizer)
        if "embed_scale" in module._buffers:
            qmodule.register_buffer("embed_scale", module._buffers["embed_scale"], persistent="embed_scale" not in module._non_persistent_buffers_set)
        elif hasattr(module, "embed_scale"):
            qmodule.embed_scale = module.embed_scale
        return qmodule

    def set_default_dtype(self, dtype):
        self._gguf_default_dtype = dtype

    @property
    def qweight(self):
        if self.weight_qtype == _GGUF_QTYPE:
            return self.weight
        return super().qweight

    def _apply_embed_scale(self, output):
        embed_scale = getattr(self, "embed_scale", None)
        if torch.is_tensor(embed_scale):
            embed_scale = embed_scale.to(device=output.device, dtype=output.dtype)
        return output if embed_scale is None else output * embed_scale

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        qweight = self.qweight
        if isinstance(qweight, GGUFWeightTensor):
            target_dtype = _resolve_default_dtype(self._gguf_default_dtype, fallback=torch.float16)
            if self.max_norm is None and not self.scale_grad_by_freq and not self.sparse:
                if _may_try_llamacpp_cuda_embedding(qweight, input):
                    fast_out = _try_llamacpp_cuda_embedding(qweight, input, target_dtype)
                    if fast_out is not None:
                        return self._apply_embed_scale(fast_out)
                output = qweight.embedding(input, dtype=target_dtype)
                if output is not None:
                    return self._apply_embed_scale(output)
            weight = qweight.dequantize(dtype=target_dtype, device=input.device)
            output = torch.nn.functional.embedding(
                input,
                weight,
                self.padding_idx,
                self.max_norm,
                self.norm_type,
                self.scale_grad_by_freq,
                self.sparse,
            )
            return self._apply_embed_scale(output)
        if (isinstance(qweight, QTensor) and getattr(qweight, "_data", None) is not None and qweight._data.dim() == 2
                and tuple(qweight._scale.shape) == (qweight._data.shape[0], 1) and self.max_norm is None):
            # Quanto per-row table: dequantize the looked-up rows only, as its dequantization does (scale * data), not the whole table
            # (a 248K-token INT8 vocabulary is 2 GB in BF16 per call)
            data = qweight._data[input]
            output = qweight._scale[input] * (data.to(qweight._scale.dtype) if qweight.qtype.is_floating_point else data)
            return self._apply_embed_scale(output)
        return super().forward(input)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        if self.weight_qtype != _GGUF_QTYPE:
            return super()._load_from_state_dict(
                state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
            )

        weight_key = prefix + "weight"
        weight_raw = state_dict.pop(weight_key, None)
        if weight_raw is None:
            missing_keys.append(weight_key)
            return

        target_dtype = _resolve_default_dtype(self._gguf_default_dtype, fallback=self.weight.dtype)
        gguf_weight = GGUFWeightTensor.create(
            raw_tensor=weight_raw,
            size=self.weight.size(),
            stride=self.weight.stride(),
            dtype=target_dtype,
            device=weight_raw.device,
            requires_grad=False,
        )
        self.weight = torch.nn.Parameter(gguf_weight, requires_grad=False)
        scale_device = weight_raw.device if torch.is_tensor(weight_raw) else torch.device("cpu")
        if not hasattr(self, "input_scale") or self.input_scale is None or self.input_scale.is_meta:
            self.input_scale = torch.ones((), dtype=torch.float32, device=scale_device)
        else:
            self.input_scale = self.input_scale.to(scale_device)
        if not hasattr(self, "output_scale") or self.output_scale is None or self.output_scale.is_meta:
            self.output_scale = torch.ones((), dtype=torch.float32, device=scale_device)
        else:
            self.output_scale = self.output_scale.to(scale_device)


def _collect_gguf_specs(state_dict):
    specs = []
    for key, tensor in state_dict.items():
        if not key.endswith(".weight"):
            continue
        if not _is_gguf_qtype(getattr(tensor, "tensor_type", None)):
            continue
        specs.append({"name": key[:-7], "tensor": tensor})
    return specs


def detect(state_dict, verboseLevel=1):
    if gguf is None:
        return {"matched": False, "kind": "none", "details": {"error": "gguf not installed"}}
    specs = _collect_gguf_specs(state_dict)
    if not specs:
        return {"matched": False, "kind": "none", "details": {}}
    names = [spec["name"] for spec in specs][:8]
    return {"matched": True, "kind": "gguf", "details": {"count": len(specs), "names": names}}


def convert_to_quanto(state_dict, default_dtype, verboseLevel=1, detection=None):
    if gguf is None:
        return {"state_dict": state_dict, "quant_map": {}}
    if detection is not None and not detection.get("matched", False):
        return {"state_dict": state_dict, "quant_map": {}}
    specs = _collect_gguf_specs(state_dict)
    if not specs:
        return {"state_dict": state_dict, "quant_map": {}}
    _set_default_dtype_from_loader(default_dtype)
    quant_map = {spec["name"]: {"weights": _GGUF_QTYPE_NAME, "activations": "none"} for spec in specs}
    return {"state_dict": state_dict, "quant_map": quant_map}


def _prepare_runtime_buffers(device):
    # Packages before 1.0.12 have no explicit runtime buffers: they allocate their scratch lazily.
    module = _gguf_cuda_module()
    if hasattr(module, "prepare_runtime_buffers"):
        module.prepare_runtime_buffers(device)


def _release_runtime_buffers():
    if hasattr(_GGUF_CUDA_MODULE, "release_runtime_buffers"):
        _GGUF_CUDA_MODULE.release_runtime_buffers()


def apply_pre_quantization(model, state_dict, quantization_map, default_dtype=None, verboseLevel=1):
    if model is not None and any(info.get("weights") == _GGUF_QTYPE_NAME for info in (quantization_map or {}).values()):
        from mmgp import runtime
        runtime.register_resource(model, _prepare_runtime_buffers, _release_runtime_buffers)
    if default_dtype is None or model is None or not quantization_map:
        return quantization_map or {}, []
    _set_default_dtype_from_loader(default_dtype)
    quantized = set(quantization_map.keys())
    for name, module in model.named_modules():
        if name in quantized and isinstance(module, torch.nn.Linear):
            module._router_default_dtype = default_dtype
    return quantization_map or {}, []


# ----------------------------------------------------------------------------------------------------------------------------------------
# Routed experts in GGUF storage: rows of ggml blocks, per expert, in the checkpoint's own tensors


from mmgp.expert_compute import ExpertFormat


class GgufExperts(ExpertFormat):
    """gate_exps and up_exps: uint8 [experts, intermediate, row bytes] rows of hidden values, down_exps: uint8 [experts, hidden, row bytes] rows of
    intermediate values, in the ggml block format of each tensor (a bank's blocks: the three ggml types, read from the row widths). The CPU kernels
    run in place (shared.kernels.cpu_linear), the GPU kernels read the same rows (shared.kernels.expert_gguf_triton)."""
    name = "gguf"
    reports_usage = True
    _TYPES = tuple(int(gguf.GGMLQuantizationType[name]) for name in ("IQ2_S", "IQ3_XXS", "IQ3_S", "IQ4_XS", "IQ4_NL")) + (int(GgmlExtraQuantizationType.Q2_0),) if gguf is not None else ()

    @staticmethod
    def _width(type, columns):
        values, size = _quantization_sizes(_quantization_type(type))
        return columns // values * size

    def tensors(self, experts, hidden, intermediate, gated=True, blocks=None):
        gate, up, down = blocks or (int(gguf.GGMLQuantizationType.IQ3_S), int(gguf.GGMLQuantizationType.IQ3_S), int(gguf.GGMLQuantizationType.IQ4_NL))
        return {"gate_exps": ((experts, intermediate, self._width(gate, hidden)), torch.uint8), "up_exps": ((experts, intermediate, self._width(up, hidden)), torch.uint8),
                "down_exps": ((experts, hidden, self._width(down, intermediate)), torch.uint8)}

    def matches(self, tensors):
        return all(tensor.dtype == torch.uint8 for tensor in tensors.values())

    def blocks(self, hidden, intermediate, tensors):
        def type_of(tensor, columns):
            for type in self._TYPES:
                if columns % _quantization_sizes(_quantization_type(type))[0] == 0 and self._width(type, columns) == tensor.shape[-1]:
                    return type
            raise ValueError(f"No GGUF expert format has rows of {tensor.shape[-1]} bytes for {columns} columns")
        return type_of(tensors["gate_exps"], hidden), type_of(tensors["up_exps"], hidden), type_of(tensors["down_exps"], intermediate)

    def _addresses(self, bank):
        return [getattr(bank, name).data_ptr() for name in bank.expert_tensor_names]

    def cpu_job(self, bank, payload, resident, out, tokens, topk, threads, share=None, share_rows=0):
        from shared.kernels import cpu_linear
        gate, up, down = self._addresses(bank)
        job = cpu_linear.MoeJob(gate=gate, up=up, down=down, payload=payload, resident=resident, usage=bank.usage.data_ptr(), out=out, gate_type=bank.blocks[0], up_type=bank.blocks[1], down_type=bank.blocks[2],
                                tokens=tokens, h=bank.hidden, i=bank.intermediate, experts=bank.num_experts, topk=topk, threads=threads, tail_rank=bank.tail_rank if tokens <= bank.TAIL_WINDOW else 0, status=0,
                                share=None if share is None else ctypes.pointer(share), share_rows=share_rows)
        return job, cpu_linear.moe_host_function()

    def cpu_share(self, fields):
        from shared.kernels import cpu_linear
        if not cpu_linear.moe_share_supported():
            return None
        share = cpu_linear.MoeShare()
        for name, value in fields.items():
            if name == "offsets":
                share.offsets[:] = value
            else:
                setattr(share, name, value)
        return share

    def share_ring(self, slot_bytes, slots, device):
        from shared.kernels import cpu_linear
        return cpu_linear.ShareRing(slot_bytes, slots, device)

    def share_sections(self):
        from shared.kernels import cpu_linear
        return cpu_linear.REMOTE_COPIES

    def cpu_forward(self, bank, x, ids, probabilities, resident, out, tokens, topk, threads):
        from shared.kernels import cpu_linear
        return cpu_linear.moe_forward(*self._addresses(bank), bank.blocks, x, ids, probabilities, resident, out, tokens, bank.hidden, bank.intermediate, bank.num_experts, topk, threads)

    def doorbell(self, device):
        from shared.kernels import cpu_linear, host_doorbell
        return (host_doorbell, cpu_linear.moe_server_function()) if host_doorbell.supported(device) else None

    def gpu_forward(self, bank, x, ids, probabilities, pointers, out=None, mid=None):
        from shared.kernels import expert_gguf_triton
        return expert_gguf_triton.forward(x, ids, probabilities, pointers, bank.intermediate, bank.blocks, out=out, mid=mid)

    def gpu_pass_supported(self, device):
        from shared.kernels import expert_gguf_groups
        return expert_gguf_groups.supported(device)

    def pass_shares(self):
        from shared.kernels import kernel_policy
        return kernel_policy.allow_approximate()

    def gpu_pass(self, bank, x, host_ids, probabilities, order, zero=False):
        from shared.kernels import expert_gguf_groups
        return expert_gguf_groups.prompt_pass(x, host_ids, probabilities, bank.intermediate, bank.blocks, order, zero)

    @staticmethod
    def _matrix(raw, type, rows, columns):
        type = _quantization_type(type)
        values, size = _quantization_sizes(type)
        return _DEQUANTIZE_FUNCTIONS[type](raw.reshape(-1, size), values, size, torch.float32).reshape(rows, columns)

    def dequantize(self, bank, tensors):
        gate = self._matrix(tensors["gate_exps"], bank.blocks[0], bank.intermediate, bank.hidden)
        up = self._matrix(tensors["up_exps"], bank.blocks[1], bank.intermediate, bank.hidden)
        return torch.cat((gate, up)), self._matrix(tensors["down_exps"], bank.blocks[2], bank.hidden, bank.intermediate)


EXPERT_FORMATS = (GgufExperts(),) if gguf is not None else ()
