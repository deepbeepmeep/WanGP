"""Compatibility import for the packed kernels distributed in SparseGP."""
try:
    from sparsegp import packed_cuda as _packed
except (ImportError, OSError):
    _packed = None


def supported(*args, **kwargs):
    return _packed is not None and _packed.supported(*args, **kwargs)


def usable(*args, **kwargs):
    return _packed is not None and _packed.usable(*args, **kwargs)


def __getattr__(name):
    if name in ("allocate", "attention", "finish", "pack_keys", "pack_values", "prepare", "mixed_routing"):
        if _packed is None:
            raise ImportError("The packed Sol CUDA kernels require SparseGP")
        return getattr(_packed, "packed_routing" if name == "mixed_routing" else name)
    raise AttributeError(name)
