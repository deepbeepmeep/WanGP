# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Streaming INT8 ConvRot checkpoint conversion used by ``--save-quantized --convrot``."""

import json
import math
import os
import struct
from collections import OrderedDict

import torch
from safetensors import safe_open

from shared.qtypes.int8_convrot import _regular_hadamard, convrot_group_size as _group_size


_SAFETENSORS_DTYPES = {
    torch.bfloat16: "BF16", torch.float16: "F16", torch.float32: "F32", torch.float64: "F64",
    torch.int8: "I8", torch.int16: "I16", torch.int32: "I32", torch.int64: "I64",
    torch.uint8: "U8", torch.uint16: "U16", torch.uint32: "U32", torch.uint64: "U64", torch.bool: "BOOL",
    torch.float8_e4m3fn: "F8_E4M3", torch.float8_e5m2: "F8_E5M2",
}


def _module_matches(name, prefixes):
    for wrapper in ("model.diffusion_model.", "diffusion_model."):
        if name.startswith(wrapper):
            name = name[len(wrapper):]
            break
    return any(name.startswith(prefix) for prefix in prefixes)


def _normalized_module_name(name):
    for wrapper in ("model.diffusion_model.", "diffusion_model."):
        if name.startswith(wrapper):
            return name[len(wrapper):]
    return name


def _tensor_bytes(tensor):
    tensor = tensor.detach().contiguous().cpu()
    if tensor.dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
        tensor = tensor.view(torch.uint8)
    return tensor.numpy().tobytes()


def quantize_weight(weight, group_size, device):
    weight = weight.to(device=device, dtype=torch.float32)
    hadamard = _regular_hadamard(group_size, device, torch.float32)
    rotated = torch.matmul(weight.reshape(weight.shape[0], -1, group_size), hadamard.T).reshape(weight.shape)
    scale = (rotated.abs().amax(dim=-1, keepdim=True).float() / 127.0).clamp_(min=1e-30)
    qweight = (rotated / scale.to(rotated.dtype)).round_().clamp_(-128, 127).to(torch.int8)
    return qweight.cpu(), scale.cpu()


def save_convrot_model(model, source_file, output_file, layout=None, verboseLevel=1, prefixes=None):
    """Stream a ConvRot checkpoint, optionally applying model-specific tensor layout transforms."""
    tower_prefixes = tuple(prefixes) if prefixes is not None else tuple(f"{name}." for name, module in model.named_modules() if isinstance(module, torch.nn.ModuleList) and len(module) >= 5)
    if not tower_prefixes:
        raise RuntimeError("ConvRot save requires a model with a repeated block tower")
    # Embedding lookups do not apply the inverse activation rotation used by
    # ConvRot Linear layers. Preserve their original weights, including relative
    # attention biases inside otherwise quantized transformer blocks.
    # A lock on a module holding no weights directly only keeps MMGP from converting its arguments: it says nothing about weights.
    locked_modules = tuple(name for name, module in model.named_modules() if isinstance(module, torch.nn.Embedding)
                           or getattr(module, "_lock_dtype", None) is not None and any(True for _ in module.parameters(recurse=False)))
    if layout is None:
        layout = {}
    source_file, output_file = os.path.abspath(source_file), os.path.abspath(output_file)
    temp_file = output_file + ".tmp"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with safe_open(source_file, framework="pt", device="cpu") as reader:
        source_keys = list(reader.keys())
        source_metadata = dict(reader.metadata() or {})
        quantized = {}
        for key in source_keys:
            tensor_slice = reader.get_slice(key)
            shape, dtype = tuple(tensor_slice.get_shape()), tensor_slice.get_dtype()
            if not key.endswith(".weight") or len(shape) != 2 or dtype not in ("BF16", "F16"):
                continue
            module_name = key[:-len(".weight")]
            if not _module_matches(module_name, tower_prefixes) or any(module_name == name or module_name.endswith(f".{name}") for name in locked_modules):
                continue
            if group_size := _group_size(shape[1]):
                quantized[key] = group_size

        header, output_entries, offset = OrderedDict(), [], 0

        def add_entry(key, dtype, shape, source_key=None, payload=None):
            nonlocal offset
            size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
            header[key] = {"dtype": _SAFETENSORS_DTYPES[dtype], "shape": list(shape), "data_offsets": [offset, offset + size]}
            output_entries.append((key, source_key, payload))
            offset += size

        for key in source_keys:
            if key in quantized:
                shape, group_size = tuple(reader.get_slice(key).get_shape()), quantized[key]
                base = key[:-len(".weight")]
                config = json.dumps({"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": group_size}).encode("utf-8")
                add_entry(key, torch.int8, shape, source_key=key, payload=group_size)
                add_entry(base + ".weight_scale", torch.float32, (shape[0], 1), payload="scale")
                add_entry(base + ".comfy_quant", torch.uint8, (len(config),), payload=config)
            else:
                tensor_slice = reader.get_slice(key)
                dtype_name = tensor_slice.get_dtype()
                dtype = next((one_dtype for one_dtype, safe_name in _SAFETENSORS_DTYPES.items() if safe_name == dtype_name), None)
                if dtype is None:
                    raise KeyError(f"Unsupported safetensors dtype '{dtype_name}' for '{key}'")
                add_entry(key, dtype, tuple(tensor_slice.get_shape()), source_key=key)

        source_metadata.update(quantization_format="int8_convrot", convrot_format="comfy_quant_v1")
        header["__metadata__"] = source_metadata
        header_bytes = json.dumps(header).encode("utf-8")
        if verboseLevel >= 1:
            print(f"ConvRot quantization of {len(quantized)} weights started on {device}")

        try:
            with open(temp_file, "wb") as writer:
                writer.write(struct.pack("<Q", len(header_bytes)))
                writer.write(header_bytes)
                scales = {}
                for index, (key, source_key, payload) in enumerate(output_entries, 1):
                    transform = layout.get(_normalized_module_name(source_key)) if source_key is not None else None
                    if source_key in quantized:
                        qweight, scale = quantize_weight(reader.get_tensor(source_key), payload, device)
                        if transform is not None:
                            qweight, scale = transform(qweight), transform(scale)
                        writer.write(_tensor_bytes(qweight))
                        scales[source_key[:-len(".weight")]] = scale
                    elif payload == "scale":
                        writer.write(_tensor_bytes(scales.pop(key[:-len(".weight_scale")])))
                    elif isinstance(payload, bytes):
                        writer.write(payload)
                    else:
                        tensor = reader.get_tensor(source_key)
                        if transform is not None:
                            tensor = transform(tensor)
                        writer.write(_tensor_bytes(tensor))
                    if verboseLevel >= 1 and index % 100 == 0:
                        print(f"ConvRot checkpoint: wrote {index}/{len(output_entries)} tensors")
            os.replace(temp_file, output_file)
        except BaseException:
            if os.path.isfile(temp_file):
                os.remove(temp_file)
            raise

    if verboseLevel >= 1:
        print(f"ConvRot checkpoint saved to '{output_file}'")
    return len(quantized)
