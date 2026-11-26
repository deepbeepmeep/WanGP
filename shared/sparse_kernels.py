"""Device operations shared by the WanGP Sparge and Sol attention cores."""
from functools import lru_cache

import torch
import torch.nn.functional as F


@lru_cache(None)
def native(device):
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    try:
        from sparsegp import native as kernels, sparge_cuda  # compiled-kernel wheel (1.1+)
    except (ImportError, OSError):
        return None
    if not kernels.image_path(device).is_file():
        return None
    try:
        kernels.module(device)
    except (kernels.ImageUnavailable, OSError):
        return None
    return kernels


@lru_cache(None)
def triton_available(device):
    if device.type != "cuda" or not torch.cuda.is_available():
        return False
    try:
        import triton
        from triton.backends import backends
    except (ImportError, OSError):
        return False
    if torch.version.hip:
        return "amd" in backends
    capability = torch.cuda.get_device_capability(device)
    version = tuple(int(part) for part in triton.__version__.split(".")[:2])
    return "nvidia" in backends and capability >= (7, 5) and (capability < (10, 0) or version >= (3, 3))


def mixed_engine(device):
    engine = native(device)
    if engine is None and triton_available(device):
        from shared.kernels import sol_mixed_triton as engine
    return engine


def padded_rows(x, rows):
    """Contiguous FP16 [batch * heads, padded tokens, width] from a strided NHD projection."""
    b, t, h, d = x.shape
    out = torch.empty((b, h, rows, d), dtype=torch.float16, device=x.device)
    out[:, :, :t].copy_(x.transpose(1, 2))
    if rows > t:
        out[:, :, t:].zero_()
    return out.view(b * h, rows, d)


def round_up(n, tile=64):
    return (n + tile - 1) // tile * tile


def dense(q, k, v, *, scale=None, causal=False, mask=None, dropout=0.0):
    """HND dense attention, including the exact conditioning queries in Sol3."""
    if (mask is None and not dropout and q.is_cuda and not torch.version.hip
            and q.dtype == k.dtype == v.dtype and q.dtype in (torch.bfloat16, torch.float16)
            and q.size(-1) in (64, 128) and q.size(-1) == v.size(-1)
            and (not causal or q.size(2) == k.size(2))
            and not (torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v)))
            and torch.cuda.get_device_capability(q.device) in ((7,5), (8,0), (8,6), (8,7), (8,9), (9,0), (10,0), (12,0), (12,1))):
        try:
            from sagegp import sageattn
        except ImportError:
            pass
        else:
            return sageattn(q, k, v, tensor_layout="HND", is_causal=causal, sm_scale=scale)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal, scale=scale,
                                          dropout_p=dropout, enable_gqa=q.size(1) != k.size(1))
