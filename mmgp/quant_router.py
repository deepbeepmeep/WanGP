# Copyright (c) 2024-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import importlib
import inspect
import os

import torch

from optimum.quanto import QModuleMixin, register_qmodule
from optimum.quanto.tensor.qtype import qtype as _quanto_qtype

from . import row_views, safetensors2


_QTYPE_QMODULE_CACHE = None
_QMODULE_BASE_ATTRS = None

_DEFAULT_KIND_PRIORITIES = {
    "nvfp4": 1,
    "nunchaku": 2,
    "nunchaku_int4": 2,
    "nunchaku_fp4": 2,
    "fp8": 10,
    "scaled_fp8": 10,
    "scaled_float8_e4m3fn": 10,
    "scaled_float8_e5m2": 10,
    "qfloat8": 10,
    "qfloat8_e4m3fn": 10,
    "qfloat8_e5m2": 10,
    "float8_e4m3fn": 10,
    "float8_e5m2": 10,
    "int8": 11,
    "qint8": 11,
}


def _extract_qtypes(handler):
    for obj in vars(handler).values():
        if isinstance(obj, _quanto_qtype):
            yield obj


def _extract_qmodule_classes(handler):
    for obj in vars(handler).values():
        if inspect.isclass(obj) and issubclass(obj, QModuleMixin) and issubclass(obj, torch.nn.Linear):
            if obj is QLinearQuantoRouter:
                continue
            yield obj


def _build_qmodule_cache():
    mapping = {}
    for handler in _load_handlers():
        qmodule_classes = list(_extract_qmodule_classes(handler))
        if len(qmodule_classes) != 1:
            continue
        qmodule_cls = qmodule_classes[0]
        for qt in _extract_qtypes(handler):
            mapping.setdefault(qt, qmodule_cls)
    return mapping


def _get_qmodule_base_attrs():
    global _QMODULE_BASE_ATTRS
    if _QMODULE_BASE_ATTRS is not None:
        return _QMODULE_BASE_ATTRS
    base = torch.nn.Linear(1, 1, bias=True)
    _QMODULE_BASE_ATTRS = set(base.__dict__.keys())
    _QMODULE_BASE_ATTRS.update({
        "_parameters",
        "_buffers",
        "_modules",
        "_non_persistent_buffers_set",
    })
    return _QMODULE_BASE_ATTRS


def _get_qmodule_for_qtype(qtype_obj):
    global _QTYPE_QMODULE_CACHE
    if qtype_obj is None:
        return None
    if _QTYPE_QMODULE_CACHE is None or qtype_obj not in _QTYPE_QMODULE_CACHE:
        _QTYPE_QMODULE_CACHE = _build_qmodule_cache()
    return _QTYPE_QMODULE_CACHE.get(qtype_obj)


def checkpoint_linear_classes():
    """Registered linear classes that load their checkpoint directly on the final module."""
    global _QTYPE_QMODULE_CACHE
    if _QTYPE_QMODULE_CACHE is None:
        _QTYPE_QMODULE_CACHE = _build_qmodule_cache()
    return {qt.name: cls for qt, cls in _QTYPE_QMODULE_CACHE.items()
            if getattr(cls, "direct_load_qtype", None) is qt}


def _load_with_qmodule(
    module,
    qmodule_cls,
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    device = module.weight.device if torch.is_tensor(module.weight) else None
    if torch.is_tensor(module.weight) and module.weight.dtype.is_floating_point:
        weight_dtype = module.weight.dtype
    elif torch.is_tensor(getattr(module, "bias", None)) and module.bias.dtype.is_floating_point:
        weight_dtype = module.bias.dtype
    else:
        weight_dtype = torch.float16
    tmp = qmodule_cls(
        module.in_features,
        module.out_features,
        bias=module.bias is not None,
        device=device,
        dtype=weight_dtype,
        weights=module.weight_qtype,
        activations=module.activation_qtype,
        optimizer=module.optimizer,
        quantize_input=True,
    )
    setter = getattr(tmp, "set_default_dtype", None)
    if callable(setter):
        setter(getattr(module, "_router_default_dtype", None) or module.weight.dtype)
    tmp._load_from_state_dict(
        state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    )

    module.weight = tmp.weight
    module.bias = tmp.bias
    module.input_scale = tmp.input_scale
    module.output_scale = tmp.output_scale

    ignore = set(_get_qmodule_base_attrs())
    ignore.update({
        "_quantize_hooks",
        "training",
        "_router_default_dtype",
    })
    for name, value in tmp.__dict__.items():
        if name in ignore:
            continue
        setattr(module, name, value)
    module._router_forward_impl = qmodule_cls.forward


@register_qmodule(torch.nn.Linear)
class QLinearQuantoRouter(QModuleMixin, torch.nn.Linear):
    @classmethod
    def qcreate(
        cls,
        module,
        weights,
        activations=None,
        optimizer=None,
        device=None,
    ):
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
        self._router_default_dtype = dtype

    def set_default_dtype(self, dtype):
        self._router_default_dtype = dtype

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        impl = getattr(self, "_router_forward_impl", None)
        if impl is not None:
            return impl(self, input)
        return torch.nn.functional.linear(input, self.qweight, bias=self.bias)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        qmodule_cls = _get_qmodule_for_qtype(self.weight_qtype)
        if qmodule_cls is not None and qmodule_cls is not QLinearQuantoRouter:
            return _load_with_qmodule(
                self, qmodule_cls, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
            )
        return super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )


_HANDLER_MODULES = []
_HANDLER_OBJECTS = []


def register_handler(handler):
    global _QTYPE_QMODULE_CACHE
    if isinstance(handler, str):
        if handler not in _HANDLER_MODULES:
            _HANDLER_MODULES.append(handler)
            _QTYPE_QMODULE_CACHE = None
        return handler
    if handler not in _HANDLER_OBJECTS:
        _HANDLER_OBJECTS.append(handler)
        _QTYPE_QMODULE_CACHE = None
    return handler


def unregister_handler(handler):
    global _QTYPE_QMODULE_CACHE
    removed = False
    if isinstance(handler, str):
        if handler in _HANDLER_MODULES:
            _HANDLER_MODULES.remove(handler)
            removed = True
    elif handler in _HANDLER_OBJECTS:
        _HANDLER_OBJECTS.remove(handler)
        removed = True
    if removed:
        _QTYPE_QMODULE_CACHE = None
    return removed


def _load_handlers():
    handlers = []
    for mod_path in _HANDLER_MODULES:
        module = importlib.import_module(mod_path, package=__package__)
        if not hasattr(module, "detect") or not hasattr(module, "convert_to_quanto"):
            raise RuntimeError(
                f"Quant handler '{mod_path}' must define detect() and convert_to_quanto() functions."
            )
        handlers.append(module)
    for handler in _HANDLER_OBJECTS:
        if not hasattr(handler, "detect") or not hasattr(handler, "convert_to_quanto"):
            raise RuntimeError(
                "Quant handler object must define detect() and convert_to_quanto() functions."
            )
        handlers.append(handler)
    register_qmodule(torch.nn.Linear)(QLinearQuantoRouter)
    return handlers


def _handler_name(handler):
    return getattr(handler, "HANDLER_NAME", handler.__name__.split(".")[-1])


def expert_formats():
    """Routed-expert storage formats (mmgp.expert_compute.ExpertFormat) of the registered handlers: their optional EXPERT_FORMATS."""
    return [format for handler in _load_handlers() for format in getattr(handler, "EXPERT_FORMATS", ())]

def cpu_compute_supported(tensor):
    """Whether a registered handler computes linears with this weight on the CPU, in its storage (its optional cpu_compute_supported(tensor)):
    the blocks that MMGP leaves on the CPU then compute it there (cpuCompute)."""
    return any(getattr(handler, "cpu_compute_supported", None) is not None and handler.cpu_compute_supported(tensor) for handler in _load_handlers())

def column_input(module, rows, dtype, device, segments=None, residual=None):
    """For a linear module whose input (rows, in_features) in dtype comes by ranges of columns and is then read once: the object of the
    first registered handler that keeps such an input in its weight format's own, smaller form (its optional column_input(module, rows,
    dtype, device, segments, residual), None for a module it cannot serve), or None. The object takes the ranges in order
    (append(x_list): a (..., columns) tensor handed off) and gives linear(), (rows, out_features); linear_residual(residual, scale),
    the output added to residual; gated_linear(residual, gate, segments), residual[start:stop] += gate[row] * output for each segment
    (start, stop, row) over the rows. segments and residual announce the read that will follow."""
    for handler in _load_handlers():
        factory = getattr(handler, "column_input", None)
        kept = None if factory is None else factory(module, rows, dtype, device, segments, residual)
        if kept is not None:
            return kept
    return None

_FILE_EXTENSION_HANDLERS = {}


def register_file_extension(extension, handler):
    if not extension or handler is None:
        return
    ext = str(extension).lower()
    if ext.startswith("."):
        ext = ext[1:]
    if not ext:
        return
    _FILE_EXTENSION_HANDLERS[ext] = handler


def get_extension_handler(file_path):
    if not isinstance(file_path, str):
        return None
    ext = os.path.splitext(file_path)[1].lower().lstrip(".")
    if not ext:
        return None
    return _FILE_EXTENSION_HANDLERS.get(ext)


def normalize_extension_path(file_path):
    handler = get_extension_handler(file_path)
    if handler is None:
        return file_path
    normalizer = getattr(handler, "normalize", None)
    if not callable(normalizer):
        ext = os.path.splitext(file_path)[1].lower().lstrip(".")
        raise Exception(f"Missing normalize for *.{ext} handler")
    return normalizer(file_path)

def _normalize_kind_key(value):
    if value is None:
        return ""
    if isinstance(value, _quanto_qtype):
        return value.name.lower()
    name = getattr(value, "name", None)
    if isinstance(name, str) and name:
        return name.lower()
    return str(value).lower()


def _priority_for_kind(kind):
    key = _normalize_kind_key(kind)
    if not key:
        return None
    if key in _DEFAULT_KIND_PRIORITIES:
        return _DEFAULT_KIND_PRIORITIES[key]
    if "nunchaku" in key:
        return _DEFAULT_KIND_PRIORITIES["nunchaku"]
    if "float8" in key or "fp8" in key:
        return _DEFAULT_KIND_PRIORITIES["fp8"]
    if "int8" in key:
        return _DEFAULT_KIND_PRIORITIES["int8"]
    return None


def _get_handler_priority(handler):
    for attr in ("HANDLER_PRIORITY", "PRIORITY", "priority"):
        value = getattr(handler, attr, None)
        if isinstance(value, (int, float)):
            return int(value)
    return _priority_for_kind(_handler_name(handler))


def _select_primary_kind(names, priority_map=None):
    if not names:
        return None
    best_name = None
    best_priority = None
    for name in names:
        priority = None
        if priority_map is not None:
            priority = priority_map.get(name)
        if priority is None:
            priority = _priority_for_kind(name)
        if priority is None:
            priority = 1000
        if best_priority is None or priority < best_priority:
            best_priority = priority
            best_name = name
    return best_name or names[0]


def _merge_quant_maps(target, source):
    if not source:
        return target
    if target is None:
        target = {}
    for key, cfg in source.items():
        if key not in target:
            target[key] = cfg
            continue
        if target[key] == cfg:
            continue
        current_priority = _priority_for_kind((target[key] or {}).get("weights")) if isinstance(target[key], dict) else None
        incoming_priority = _priority_for_kind((cfg or {}).get("weights")) if isinstance(cfg, dict) else None
        if current_priority is None:
            current_priority = 1000
        if incoming_priority is None:
            incoming_priority = 1000
        if incoming_priority < current_priority:
            target[key] = cfg
    return target


def _as_field_tuple(value):
    if not value:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _get_split_handler(info, field, default_handlers):
    handlers = info.get("split_handlers") or info.get("field_handlers") or {}
    if handlers:
        handler = handlers.get(field)
        if handler is not None:
            return handler
    if info.get("interleaved") and field in ("weight", "bias"):
        return _split_interleaved
    if default_handlers:
        return default_handlers.get(field)
    return None


def _split_interleaved(src, dim, split_sizes, context):
    # rows interleaved head by head, (heads, projections, head_dim): each projection's rows gathered into a copy
    return [row_views.gather(src, slot, len(split_sizes), context["info"]["attention_head_dim"]) for slot in range(len(split_sizes))]


def _get_split_base_fields(info, split_fields):
    base_fields = _as_field_tuple(info.get("base_fields") or info.get("base_field"))
    if base_fields:
        return base_fields
    if split_fields:
        return (next(iter(split_fields.keys())),)
    return ()


def _merge_share_fields(info, share_fields):
    info_fields = _as_field_tuple(info.get("share_fields") or info.get("shared_fields"))
    return tuple(sorted(set(info_fields).union(_as_field_tuple(share_fields))))


def _call_split_handler(handler, *, src, dim, split_sizes, context):
    if handler is None:
        return None
    try:
        chunks = handler(src=src, dim=dim, split_sizes=split_sizes, context=context)
    except Exception:
        return None
    if not isinstance(chunks, (list, tuple)) or len(chunks) != len(split_sizes):
        return None
    return chunks


def _fill_sub_maps(sub_maps, name, value):
    for sub_map in sub_maps:
        sub_map[name] = value


def _get_quantized_subtensors(p):
    getter = getattr(p, "get_quantized_subtensors", None)
    if getter is None:
        return None
    sub_tensors = getter()
    if not sub_tensors:
        return None
    if isinstance(sub_tensors, dict):
        sub_tensors = list(sub_tensors.items())
    out = []
    for name, tensor in sub_tensors:
        if tensor is None:
            continue
        if torch.is_tensor(tensor):
            out.append((name, tensor))
    return out if out else None


def sd_split_linear(
    state_dict,
    split_map,
    split_fields=None,
    share_fields=None,
    verboseLevel=1,
    split_handlers=None,
    allowed_bases=None,
    return_split_bases=False,
):
    if not split_map:
        return (state_dict, []) if return_split_bases else state_dict
    split_fields = split_fields or {}
    share_fields = share_fields or ()
    split_handlers = split_handlers or {}
    base_fields_by_suffix = {
        suffix: _get_split_base_fields(info or {}, split_fields)
        for suffix, info in split_map.items()
    }
    def _skip(msg):
        pass

    bases = {}
    for key in state_dict.keys():
        for suffix, base_fields in base_fields_by_suffix.items():
            for base_field in base_fields:
                suffix_token = f"{suffix}.{base_field}"
                if not key.endswith(suffix_token):
                    continue
                base = key[: -len("." + base_field)]
                if base.endswith(suffix):
                    bases[base] = suffix
                break

    if allowed_bases is not None:
        allowed_set = set(allowed_bases)
        bases = {base: suffix for base, suffix in bases.items() if base in allowed_set}

    if not bases:
        return (state_dict, []) if return_split_bases else state_dict

    split_bases = []

    for base, suffix in bases.items():
        info = split_map.get(suffix) or {}
        mapped = info.get("mapped_modules") or info.get("mapped_suffixes") or info.get("mapped") or []
        if not mapped:
            continue

        base_fields = base_fields_by_suffix.get(suffix) or _get_split_base_fields(info, split_fields)
        size_field = info.get("size_field") or (base_fields[0] if base_fields else None)
        size_tensor = state_dict.get(base + "." + size_field) if size_field else None
        split_dim = info.get("split_dim", 0)
        split_sizes = list(info.get("split_sizes") or [])
        if not split_sizes:
            if size_tensor is None:
                continue
            if size_tensor.dim() <= split_dim:
                _skip(f"{base}: dim={size_tensor.dim()} split_dim={split_dim}")
                continue
            out_dim = size_tensor.size(split_dim)
            if out_dim % len(mapped) != 0:
                _skip(f"{base}: out_dim={out_dim} not divisible by {len(mapped)}")
                continue
            split_sizes = [out_dim // len(mapped)] * len(mapped)
        elif None in split_sizes:
            if size_tensor is None:
                continue
            if size_tensor.dim() <= split_dim:
                _skip(f"{base}: dim={size_tensor.dim()} split_dim={split_dim}")
                continue
            known = sum(size for size in split_sizes if size is not None)
            none_count = split_sizes.count(None)
            remaining = size_tensor.size(split_dim) - known
            if remaining < 0 or remaining % none_count != 0:
                _skip(f"{base}: cannot resolve split sizes")
                continue
            fill = remaining // none_count
            split_sizes = [fill if size is None else size for size in split_sizes]

        total = sum(split_sizes)
        prefix = base[: -len(suffix)]
        target_bases = [prefix + name for name in mapped]
        added = 0

        field_tensors = {
            field: state_dict.get(base + "." + field)
            for field in set(split_fields.keys()).union(share_fields)
        }
        base_ctx = {
            "state_dict": state_dict,
            "base": base,
            "suffix": suffix,
            "split_sizes": split_sizes,
            "total": total,
            "mapped": mapped,
            "target_bases": target_bases,
            "verboseLevel": verboseLevel,
            "split_fields": split_fields,
            "share_fields": share_fields,
            "field_tensors": field_tensors,
            "size_field": size_field,
            "size_tensor": size_tensor,
            "split_dim": split_dim,
            "info": info,
        }
        fields_iter = list(split_fields.items()) + [(field, None) for field in share_fields]
        for field, dim in fields_iter:
            src = field_tensors.get(field)
            if src is None:
                continue
            if dim is None:
                for target_base in target_bases:
                    dest_key = target_base + "." + field
                    if dest_key not in state_dict:
                        state_dict[dest_key] = src
                        added += 1
                continue
            handler = _get_split_handler(info, field, split_handlers)
            chunks = _call_split_handler(
                handler,
                src=src,
                dim=dim,
                split_sizes=split_sizes,
                context=dict(base_ctx, field=field),
            )
            if chunks is None:
                if src.dim() <= dim:
                    _skip(f"{base}.{field}: dim={src.dim()} split_dim={dim}")
                    continue
                if src.size(dim) != total:
                    _skip(f"{base}.{field}: size({dim})={src.size(dim)} expected={total}")
                    continue
                chunks = torch.split(src, split_sizes, dim=dim)
            for target_base, chunk in zip(target_bases, chunks):
                if torch.is_tensor(chunk) and not chunk.is_contiguous():
                    chunk = chunk.contiguous()
                dest_key = target_base + "." + field
                if dest_key not in state_dict:
                    state_dict[dest_key] = chunk
                    added += 1

        if added:
            for field in list(split_fields.keys()) + list(share_fields):
                state_dict.pop(base + "." + field, None)
            split_bases.append(base)
    if return_split_bases:
        return state_dict, split_bases
    return state_dict


def split_linear_modules(model, map, split_handlers=None, share_fields=None):
    from accelerate import init_empty_weights

    split_handlers = split_handlers or {}
    share_fields = share_fields or ()

    modules_dict = { k: m for k, m in model.named_modules()}
    for module_suffix, split_info in map.items():
        mapped_modules = split_info["mapped_modules"]
        split_sizes = split_info["split_sizes"]
        split_share_fields = _merge_share_fields(split_info, share_fields)
        split_dims = split_info.get("split_dims") or {}
        for k, module in modules_dict.items():
            if k.endswith("." + module_suffix):
                parent_module = modules_dict[k[:len(k)-len(module_suffix)-1]]
                weight = module.weight
                bias = getattr(module, "bias", None)
                if isinstance(module, QModuleMixin):
                    out_features_total = weight.size(0)
                    if sum(split_sizes) != out_features_total:
                        raise ValueError(
                            f"Split sizes {split_sizes} do not match out_features {out_features_total} for '{k}'."
                        )
                    in_features = weight.size(1)
                    if bias is not None and bias.dim() > 0 and bias.size(0) == out_features_total:
                        sub_biases = torch.split(bias, split_sizes, dim=0)
                    else:
                        sub_biases = [bias] * len(split_sizes)

                    sub_tensors = _get_quantized_subtensors(weight)
                    if not sub_tensors:
                        raise ValueError(f"Unable to split quantized weight for '{k}'.")
                    sub_maps = [dict() for _ in split_sizes]
                    field_tensors = {name: tensor for name, tensor in sub_tensors}
                    base_ctx = {
                        "module": module,
                        "module_name": k,
                        "module_suffix": module_suffix,
                        "mapped_modules": mapped_modules,
                        "split_sizes": split_sizes,
                        "out_features": out_features_total,
                        "in_features": in_features,
                        "field_tensors": field_tensors,
                        "info": split_info,
                    }
                    for name, tensor in sub_tensors:
                        if tensor is None or name in split_share_fields or tensor.dim() <= 1:
                            _fill_sub_maps(sub_maps, name, tensor)
                            continue
                        split_dim = split_dims.get(name)
                        if split_dim is None:
                            if tensor.size(0) == out_features_total:
                                split_dim = 0
                            elif tensor.dim() > 1 and tensor.size(1) == out_features_total:
                                split_dim = 1
                            else:
                                split_dim = 0
                        handler = _get_split_handler(split_info, name, split_handlers)
                        chunks = _call_split_handler(
                            handler,
                            src=tensor,
                            dim=split_dim,
                            split_sizes=split_sizes,
                            context=dict(base_ctx, split_dim=split_dim),
                        )
                        if chunks is None:
                            if tensor.dim() <= split_dim or tensor.size(split_dim) != out_features_total:
                                got_size = "n/a" if tensor.dim() <= split_dim else tensor.size(split_dim)
                                raise ValueError(
                                    f"Cannot split '{k}' quantized tensor '{name}': "
                                    f"expected size({split_dim})={out_features_total}, got {got_size}."
                                )
                            chunks = torch.split(tensor, split_sizes, dim=split_dim)
                        for sub_map, chunk in zip(sub_maps, chunks):
                            sub_map[name] = chunk

                    create_fn = getattr(weight.__class__, "create", None)
                    if not callable(create_fn):
                        raise ValueError(f"Quantized weight class '{weight.__class__.__name__}' has no create()")
                    create_sig = inspect.signature(create_fn)
                    base_kwargs = {
                        "qtype": getattr(weight, "qtype", None),
                        "axis": getattr(weight, "axis", None),
                        "stride": weight.stride(),
                        "dtype": weight.dtype,
                        "activation_qtype": getattr(weight, "activation_qtype", None),
                        "requires_grad": weight.requires_grad,
                        "group_size": getattr(weight, "_group_size", None),
                        "device": weight.device,
                    }

                    qmodule_cls = module.__class__
                    for sub_name, sub_size, sub_map, sub_bias in zip(
                        mapped_modules, split_sizes, sub_maps, sub_biases
                    ):
                        with init_empty_weights():
                            sub_module = qmodule_cls(
                                in_features,
                                sub_size,
                                bias=bias is not None,
                                device="cpu",
                                dtype=weight.dtype,
                                weights=module.weight_qtype,
                                activations=module.activation_qtype,
                                optimizer=module.optimizer,
                                quantize_input=True,
                            )
                        size = list(weight.size())
                        if size:
                            size[0] = sub_size
                        base_kwargs["size"] = tuple(size)
                        create_kwargs = {}
                        missing = []
                        for name, param in create_sig.parameters.items():
                            if name == "self":
                                continue
                            if name in sub_map:
                                create_kwargs[name] = sub_map[name]
                            elif name in base_kwargs and base_kwargs[name] is not None:
                                create_kwargs[name] = base_kwargs[name]
                            elif param.default is param.empty:
                                missing.append(name)
                        if missing:
                            raise ValueError(
                                f"Unable to rebuild quantized weight for '{k}.{sub_name}': "
                                f"missing {missing}."
                            )
                        sub_weight = create_fn(**create_kwargs)
                        sub_module.weight = torch.nn.Parameter(sub_weight, requires_grad=weight.requires_grad)
                        if sub_bias is not None:
                            sub_module.bias = torch.nn.Parameter(sub_bias)
                        sub_module.optimizer = module.optimizer
                        sub_module.weight_qtype = module.weight_qtype
                        sub_module.activation_qtype = module.activation_qtype
                        setattr(parent_module, sub_name, sub_module)
                else:
                    sub_data = torch.split(weight, split_sizes, dim=0)
                    sub_bias = torch.split(bias, split_sizes, dim=0) if bias is not None else [None] * len(split_sizes)
                    for sub_name, sub_weight, sub_biases in zip(mapped_modules, sub_data, sub_bias):
                        sub_module = torch.nn.Linear(
                            module.in_features,
                            sub_weight.size(0),
                            bias=bias is not None,
                            device=weight.device,
                            dtype=weight.dtype,
                        )
                        sub_module.weight = torch.nn.Parameter(sub_weight)
                        if sub_biases is not None:
                            sub_module.bias = torch.nn.Parameter(sub_biases)
                        setattr(parent_module, sub_name, sub_module)
                delattr(parent_module, module_suffix)


def _call_quant_handler(fn, *args, metadata=None, **kwargs):
    if "metadata" in inspect.signature(fn).parameters:
        kwargs["metadata"] = metadata
    return fn(*args, **kwargs)


def detect_safetensors_format(state_dict, verboseLevel=1, metadata=None):
    matches = []
    details = {}
    priorities = {}
    for handler in _load_handlers():
        result = _call_quant_handler(handler.detect, state_dict, verboseLevel=verboseLevel, metadata=metadata)
        name = _handler_name(handler)
        details[name] = result
        if result.get("matched", False):
            matches.append(name)
            priorities[name] = _get_handler_priority(handler)
    kind = _select_primary_kind(matches, priorities) or "none"
    return {"kind": kind, "found": matches, "details": details, "mixed": len(matches) > 1}


def detect_and_convert(state_dict, default_dtype, verboseLevel=1, metadata=None):
    info = detect_safetensors_format(state_dict, verboseLevel=verboseLevel, metadata=metadata)
    kind = info.get("kind", "none")
    matches = info.get("found", []) or []
    if kind in ("none", "quanto") and not matches:
        return {"state_dict": state_dict, "quant_map": {}, "kind": kind, "details": info}

    handlers = _load_handlers()
    handler_map = {_handler_name(handler): handler for handler in handlers}
    if not matches:
        raise RuntimeError(f"Unsupported quantization format '{kind}'")

    if len(matches) == 1:
        handler = handler_map.get(matches[0])
        if handler is None:
            raise RuntimeError(f"Unsupported quantization format '{kind}'")
        detection = info.get("details", {}).get(matches[0], {})
        conv = _call_quant_handler(
            handler.convert_to_quanto,
            state_dict,
            default_dtype=default_dtype,
            verboseLevel=verboseLevel,
            detection=detection,
            metadata=metadata,
        )
        conv["kind"] = kind
        conv["details"] = info
        return conv

    def _match_priority(name):
        handler = handler_map.get(name)
        priority = _get_handler_priority(handler) if handler is not None else None
        if priority is None:
            priority = _priority_for_kind(name)
        if priority is None:
            priority = 1000
        return priority

    ordered_matches = sorted(matches, key=_match_priority)
    merged_state = state_dict
    merged_map = {}
    for name in ordered_matches:
        handler = handler_map.get(name)
        if handler is None:
            continue
        detection = info.get("details", {}).get(name, {})
        conv = _call_quant_handler(
            handler.convert_to_quanto,
            merged_state,
            default_dtype=default_dtype,
            verboseLevel=verboseLevel,
            detection=detection,
            metadata=metadata,
        )
        merged_state = conv.get("state_dict", merged_state)
        merged_map = _merge_quant_maps(merged_map, conv.get("quant_map", {}))
    return {"state_dict": merged_state, "quant_map": merged_map, "kind": kind, "details": info}


def get_available_qtypes():
    try:
        from optimum.quanto.tensor.qtype import qtypes as _quanto_qtypes
    except Exception:
        return []
    return sorted(_quanto_qtypes.keys())


def get_available_qtype_aliases():
    aliases = set()
    for name in get_available_qtypes():
        key = str(name).lower()
        aliases.add(key)
        if key.startswith("q") and len(key) > 1:
            aliases.add(key[1:])
        if "float8" in key:
            aliases.add("fp8")
    return aliases


def get_quantization_tokens(quantization):
    if quantization is None:
        return []
    key = str(quantization).lower()
    if len(key) == 0:
        return []
    aliases = get_available_qtype_aliases()
    if key not in aliases:
        return []
    tokens = {key}
    if key.startswith("q") and len(key) > 1:
        tokens.add(key[1:])
    if "float8" in key or key == "fp8":
        tokens.add("fp8")
    if "int4" in key:
        tokens.add("int4")
    if "int8" in key:
        tokens.add("int8")
    return sorted(tokens, key=len, reverse=True)


def get_quantization_label(quantization):
    if quantization is None:
        return ""
    key = str(quantization).lower()
    if key in ("", "none", "bf16", "fp16", "float16", "bfloat16"):
        return ""
    aliases = get_available_qtype_aliases()
    if key not in aliases:
        return ""
    if "float8" in key or key == "fp8":
        return "FP8"
    if key.startswith("q"):
        key = key[1:]
    return key.replace("_", " ").upper()


_quantization_filename_cache = {}


def _normalize_quant_file_key(file_path):
    try:
        return os.path.normcase(os.path.abspath(file_path))
    except Exception:
        return str(file_path).lower()


def get_cached_quantization_for_file(file_path):
    if not file_path:
        return None
    return _quantization_filename_cache.get(_normalize_quant_file_key(file_path))


def cache_quantization_for_file(file_path, kind):
    if not file_path or not kind:
        return
    key = _normalize_quant_file_key(file_path)
    if key not in _quantization_filename_cache:
        _quantization_filename_cache[key] = kind


def _detect_kind_from_handlers(file_path, verboseLevel=1):
    found = []
    for handler in _load_handlers():
        fn = getattr(handler, "detect_quantization_kind_for_file", None)
        if fn is None:
            continue
        try:
            kind = fn(file_path, verboseLevel=verboseLevel)
        except TypeError:
            kind = fn(file_path)
        if kind:
            found.append((kind, _get_handler_priority(handler)))
    if not found:
        return None
    found.sort(key=lambda entry: entry[1] if entry[1] is not None else 1000)
    return found[0][0]


def _detect_label_from_handlers(file_path, verboseLevel=0):
    for handler in _load_handlers():
        fn = getattr(handler, "detect_quantization_label_from_filename", None)
        if fn is None:
            continue
        try:
            label = fn(file_path, verboseLevel=verboseLevel)
        except TypeError:
            label = fn(file_path)
        if label:
            return label
    return ""


def _infer_qtype_from_quantization_map(quantization_map):
    if not quantization_map:
        return None
    counts = {}
    for entry in quantization_map.values():
        if not isinstance(entry, dict):
            continue
        weights = entry.get("weights")
        if not weights or weights == "none":
            continue
        key = _normalize_kind_key(weights)
        if not key:
            continue
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return None
    best_key = None
    best_priority = None
    best_count = None
    for key, count in counts.items():
        priority = _priority_for_kind(key)
        if priority is None:
            priority = 1000
        if (
            best_priority is None
            or priority < best_priority
            or (priority == best_priority and (best_count is None or count > best_count))
        ):
            best_priority = priority
            best_count = count
            best_key = key
    return best_key


def load_metadata_state_dict(file_path):

    if isinstance(file_path, str) and get_extension_handler(file_path) is not None:
        ext_handler = get_extension_handler(file_path)
        if hasattr(ext_handler, "get_file_metadata"):
            return ext_handler.get_file_metadata(file_path)
        else:
            return {}, {}
    state_dict, metadata = safetensors2.load_metadata_state_dict(file_path)
    return state_dict, metadata

def detect_quantization_kind_for_file(file_path, verboseLevel=1):
    cached = get_cached_quantization_for_file(file_path)
    if cached:
        return cached
    if not file_path or not os.path.isfile(file_path):
        return None
    if not (".safetensors" in file_path or ".sft" in file_path):
        kind = _detect_kind_from_handlers(file_path, verboseLevel=verboseLevel)
        if kind:
            cache_quantization_for_file(file_path, kind)
            return kind
        return None

    def _load_full():
        state_dict = {}
        with safetensors2.safe_open(
            file_path,
            framework="pt",
            device="cpu",
            writable_tensors=False,
        ) as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
            metadata = f.metadata()
        return state_dict, metadata

    def _try_detect(state_dict, metadata):
        try:
            info = detect_safetensors_format(state_dict, verboseLevel=verboseLevel, metadata=metadata)
            return info.get("kind"), True
        except Exception:
            return None, False

    metadata_only = False
    try:
        state_dict, metadata = load_metadata_state_dict(file_path)
        metadata_only = True
    except Exception:
        try:
            state_dict, metadata = _load_full()
        except Exception:
            return None

    kind, ok = _try_detect(state_dict, metadata)
    if metadata_only and not ok:
        try:
            state_dict, metadata = _load_full()
            kind, ok = _try_detect(state_dict, metadata)
        except Exception:
            kind = None

    if (not kind or kind == "none") and metadata is not None:
        inferred = _infer_qtype_from_quantization_map(metadata.get("quantization_map"))
        if inferred:
            kind = inferred

    cache_quantization_for_file(file_path, kind or "none")
    return kind


def detect_quantization_label_from_filename(filename):
    if not filename:
        return ""
    label = _detect_label_from_handlers(filename, verboseLevel=0)
    if label:
        return label
    cached = get_cached_quantization_for_file(filename)
    if cached:
        return get_quantization_label(cached)
    kind = detect_quantization_kind_for_file(filename, verboseLevel=0)
    if kind:
        label = get_quantization_label(kind)
        if label:
            return label
    base = os.path.basename(filename).lower()
    for token in sorted(get_available_qtype_aliases(), key=len, reverse=True):
        if token and token in base:
            return get_quantization_label(token)
    if "quanto" in base:
        return "QUANTO"
    return ""


def _get_qtype_name_from_quant_map(entry):
    if entry is None:
        return ""
    if isinstance(entry, dict):
        entry = entry.get("weights")
    return _normalize_kind_key(entry)


def _build_qtype_handler_map():
    mapping = {}
    for handler in _load_handlers():
        for qt in _extract_qtypes(handler):
            name = getattr(qt, "name", None)
            if isinstance(name, str) and name:
                mapping.setdefault(name.lower(), handler)
    return mapping


def _collect_fused_bases(state_dict, fused_split_map):
    if not fused_split_map or not state_dict:
        return {}
    suffixes = list(fused_split_map.keys())
    if not suffixes:
        return {}
    bases = {}
    for key in state_dict.keys():
        if key.endswith(".weight"):
            base = key[:-7]
        elif key.endswith(".qweight"):
            base = key[:-8]
        else:
            continue
        for suffix in suffixes:
            if base.endswith(suffix):
                bases[base] = suffix
                break
    return bases


def _remap_quantization_entries(quantization_map, base, targets):
    if not quantization_map or not targets:
        return
    for suffix in ("", ".weight"):
        key = base + suffix if suffix else base
        if key not in quantization_map:
            continue
        cfg = quantization_map.pop(key)
        for target in targets:
            quantization_map[target + suffix] = cfg


def split_fused_weights(state_dict, quantization_map, fused_split_map, default_dtype=None, verboseLevel=1, row_view_bases=None):
    """Splits the fused weights of fused_split_map. Unquantized weights whose entry declares "interleaved" rows are kept fused when the
    registered row view kernel computes their projections (appended to row_view_bases as (base, entry), see row_views.install)."""
    if not fused_split_map or not state_dict:
        return state_dict, quantization_map

    quantization_map = quantization_map or {}
    fused_bases = _collect_fused_bases(state_dict, fused_split_map)
    if not fused_bases:
        return state_dict, quantization_map

    handler_map = _build_qtype_handler_map()
    bases_info = {}
    for base, suffix in fused_bases.items():
        info = fused_split_map.get(suffix) or {}
        mapped = info.get("mapped_modules") or info.get("mapped_suffixes") or info.get("mapped") or []
        if not mapped:
            continue
        prefix = base[:-len(suffix)]
        targets = [prefix + name for name in mapped]
        qtype_name = _get_qtype_name_from_quant_map(quantization_map.get(base))
        if not qtype_name:
            qtype_name = _get_qtype_name_from_quant_map(quantization_map.get(base + ".weight"))
        if not qtype_name:
            weight = state_dict.get(base + ".weight")
            if torch.is_tensor(weight) and getattr(weight, "tensor_type", None) is not None:
                qtype_name = "gguf"
        bases_info[base] = {
            "suffix": suffix,
            "targets": targets,
            "qtype": qtype_name,
        }

    if not bases_info:
        return state_dict, quantization_map

    handler_bases = {}
    for base, info in bases_info.items():
        qtype_name = info["qtype"]
        if not qtype_name:
            continue
        if (fused_split_map.get(info["suffix"]) or {}).get("interleaved"):
            raise ValueError(f"'{base}': quantized weights with rows interleaved head by head are not supported; store them grouped")
        handler = handler_map.get(qtype_name)
        if handler is None:
            continue
        if not callable(getattr(handler, "split_fused_weights", None)):
            continue
        handler_bases.setdefault(handler, set()).add(base)

    handled_bases = set()
    for handler, bases in handler_bases.items():
        fn = getattr(handler, "split_fused_weights", None)
        if not callable(fn):
            continue
        result = fn(
            state_dict,
            fused_split_map,
            quantization_map=quantization_map,
            allowed_bases=bases,
            default_dtype=default_dtype,
            verboseLevel=verboseLevel,
        )
        if isinstance(result, tuple) and len(result) == 2:
            state_dict, split_bases = result
        else:
            state_dict = result or state_dict
            split_bases = []
        handled_bases.update(split_bases or [])

    default_bases = [
        base for base, info in bases_info.items()
        if not info["qtype"] and base not in handled_bases
    ]
    if row_view_bases is not None:
        for base in list(default_bases):
            info = fused_split_map[bases_info[base]["suffix"]]
            if info.get("interleaved") and row_views.available(state_dict[base + ".weight"].dtype, info["attention_head_dim"]):
                row_view_bases.append((base, info))
                default_bases.remove(base)
    if default_bases:
        state_dict, split_bases = sd_split_linear(
            state_dict,
            fused_split_map,
            split_fields={"weight": 0, "bias": 0},
            verboseLevel=verboseLevel,
            allowed_bases=default_bases,
            return_split_bases=True,
        )
        handled_bases.update(split_bases or [])

    if quantization_map and handled_bases:
        for base in handled_bases:
            info = bases_info.get(base)
            if not info:
                continue
            _remap_quantization_entries(quantization_map, base, info["targets"])

    return state_dict, quantization_map


def apply_pre_quantization(model, state_dict, quantization_map, default_dtype=None, verboseLevel=1):
    remaining = dict(quantization_map or {})
    post_load = []
    for handler in _load_handlers():
        fn = getattr(handler, "apply_pre_quantization", None)
        if fn is None:
            continue
        remaining, hooks = fn(
            model,
            state_dict,
            remaining,
            default_dtype=default_dtype,
            verboseLevel=verboseLevel,
        )
        if hooks:
            post_load.extend(hooks)
    return remaining, post_load

def _patch_marlin_fp8_bias():
    """
    Quanto's Marlin FP8 CUDA kernel currently ignores the bias argument.
    Add it back manually (in-place) so outputs stay correct on CUDA builds.
    """
    try:
        from optimum.quanto.tensor.weights.marlin.fp8 import qbits as marlin_fp8
    except Exception:
        return
    if getattr(marlin_fp8.MarlinF8QBytesLinearFunction, "_wan2gp_bias_patch", False):
        return

    orig_forward = marlin_fp8.MarlinF8QBytesLinearFunction.forward

    def forward_with_bias(ctx, input, other, bias=None):
        out = orig_forward(ctx, input, other, None)
        if bias is None:
            return out
        bias_to_add = bias
        if bias_to_add.device != out.device or bias_to_add.dtype != out.dtype:
            bias_to_add = bias_to_add.to(device=out.device, dtype=out.dtype)
        view_shape = [1] * out.ndim
        view_shape[-1] = bias_to_add.shape[0]
        bias_view = bias_to_add.view(*view_shape)
        out.add_(bias_view)
        return out

    marlin_fp8.MarlinF8QBytesLinearFunction.forward = staticmethod(forward_with_bias)  # type: ignore
    marlin_fp8.MarlinF8QBytesLinearFunction._wan2gp_bias_patch = True  # type: ignore
    marlin_fp8.MarlinF8QBytesLinearFunction._wan2gp_bias_orig = orig_forward  # type: ignore


_patch_marlin_fp8_bias()
