"""WanGP block-sparse attention. Callers supply block maps; the core owns packing and backend selection."""
from functools import lru_cache

import torch

from . import sparse_kernels as kernels

MODES = {"auto": "sparge", "sparge": "sparge_installed", "triton_sparse": "sparge_triton"}
BUNDLED = "bundled Triton Sparse Attention"


@lru_cache(None)
def _legacy(device, head_dim):
    if device.type != "cuda" or torch.version.hip:
        return None
    try:
        from . import spas_sage_attn_core as legacy
    except (ImportError, OSError, AttributeError, RuntimeError):
        return None
    settings = legacy.staged_settings(device, head_dim)
    if settings is not None:
        return legacy, settings


@lru_cache(None)
def staged_settings(device, head_dim, mode="sparge", block_size=(128, 64)):
    if head_dim not in (64, 128) or device.type != "cuda":
        return None
    if mode != "sparge_triton":
        native = kernels.native(device)
        if native is not None:
            from sparsegp import sparge_cuda
            cfg = sparge_cuda.settings(device,head_dim,block_size)
            if cfg is not None:
                return dict(compiled=sparge_cuda, name="SparseGP Sparge CUDA", block_q=block_size[0], block_k=block_size[1], **cfg)
            return dict(engine=native, name="SparseGP HIP" if torch.version.hip else "SparseGP CUDA",
                        block_q=block_size[0], block_k=block_size[1])
        legacy = _legacy(device, head_dim)
        if legacy is not None and block_size == (legacy[1]["block_q"], legacy[1]["block_k"]):
            return dict(legacy=legacy[0], name="SpargeAttention CUDA", **legacy[1])
        if kernels.triton_available(device):
            try:
                from sparsegp import triton_backend
            except (ImportError, OSError):
                pass
            else:
                return dict(engine=triton_backend, name="SparseGP Triton", block_q=block_size[0], block_k=block_size[1])
    if mode != "sparge_installed" and block_size == (128, 64) and kernels.triton_available(device):
        if torch.version.hip:
            # The portable tile kernel uses Radeon wave matrix operations (packed dot products on RDNA 2).
            from shared.kernels import sol_mixed_triton
            return dict(engine=sol_mixed_triton, name=BUNDLED, block_q=128, block_k=64)
        return dict(name=BUNDLED, block_q=128, block_k=64)
    return None


def radial_geometry(device, head_dim):
    if torch.version.hip is None and torch.cuda.get_device_capability(device) == (9, 0):
        if kernels.native(device) is not None:
            from sparsegp import sparge_cuda
            if sparge_cuda.settings(device, head_dim, (64,128)) is not None:
                return 64,128
        legacy = _legacy(device, head_dim)
        if legacy is not None:
            return legacy[1]["block_q"], legacy[1]["block_k"]
    return 128, 64


def require(device, head_dim=128, mode="sparge", block_size=(128, 64)):
    settings = staged_settings(device, head_dim, mode, block_size)
    if settings is None:
        raise RuntimeError(f"No compatible Sparge backend for {mode} on {device} with head width {head_dim} and block size {block_size}. See docs/INSTALLATION.md.")
    return settings


def prepare_mask(block_map, dtype):
    """Expose the batch and head axes for head splitting. Empty block rows stay empty."""
    while block_map.ndim < 4:
        block_map = block_map.unsqueeze(0)
    return block_map


@torch.compiler.disable
def staged_quantize_v(v_list, settings):
    if "compiled" in settings:
        return settings["compiled"].staged_quantize_v(v_list, settings)
    if "legacy" in settings:
        return settings["legacy"].staged_quantize_v(v_list, settings)
    v = v_list.pop()
    if "engine" in settings:
        return kernels.padded_rows(v, kernels.round_up(v.size(1)))
    return v.to(torch.float16)


@torch.compiler.disable
def staged_quantize_k(k_list, settings):
    if "compiled" in settings:
        return settings["compiled"].staged_quantize_k(k_list, settings)
    if "legacy" in settings:
        return settings["legacy"].staged_quantize_k(k_list, settings)
    k = k_list.pop()
    tokens = k.size(1)
    if "engine" in settings:
        rows = kernels.padded_rows(k, kernels.round_up(tokens))
        rows[:, :tokens].sub_(rows[:, :tokens].mean(1, keepdim=True))
        return (*settings["engine"].pack(rows), tokens)
    from shared.kernels.sparge_quant_per_block import quantize
    return quantize(k - k.mean(1, keepdim=True), 64, 1.0)


@torch.compiler.disable
def staged_quantize_q(q, settings):
    if "compiled" in settings:
        return settings["compiled"].staged_quantize_q(q, settings)
    if "legacy" in settings:
        return settings["legacy"].staged_quantize_q(q, settings)
    if "engine" in settings:
        return settings["engine"].pack(kernels.padded_rows(q, kernels.round_up(q.size(1))))
    from shared.kernels.sparge_quant_per_block import quantize
    return quantize(q, 128, q.size(-1) ** -0.5 * 1.44269504)


@torch.compiler.disable
def staged_attention(out, quantized_list, settings, attn_mask=None):
    if "compiled" in settings:
        return settings["compiled"].staged_attention(out, quantized_list, settings, attn_mask)
    if "legacy" in settings:
        qb, kb = settings["block_q"], settings["block_k"]
        keep = attn_mask.bool()
        if settings.get("causal"):
            # Legacy CUDA masks individual tokens only in the last tiles of each lookup list.
            q_end = (torch.arange(keep.size(-2), device=out.device) + 1) * qb
            k_start = torch.arange(keep.size(-1), device=out.device) * kb
            keep = keep & (k_start < q_end[:, None])
        settings["legacy"].staged_attention(out, quantized_list, settings, keep.contiguous().view(torch.int8))
        torch.autograd.graph.increment_version(out)
        # Legacy CUDA returns without writing a tile when its lookup list is empty.
        if settings.get("causal"):
            first = torch.where(keep, k_start, out.size(1)).amin(-1)
            rows = torch.arange(out.size(1), device=out.device)
            empty = rows < first.index_select(-1, rows // qb)
        else:
            empty = (~keep.any(-1)).repeat_interleave(qb, -1)[..., :out.size(1)]
        out.masked_fill_(empty.transpose(1, 2).unsqueeze(-1), 0)
        return out
    (q, qs), key, v = quantized_list
    quantized_list.clear()
    b, t, h, d = out.shape
    if "engine" in settings:
        k, ks, tokens = key
        qb, kb = settings["block_q"], settings["block_k"]
        keep = attn_mask.expand(b, h, (t + qb - 1) // qb, (tokens + kb - 1) // kb)
        engine = settings["engine"]
        if hasattr(engine, "attend_blocks"):
            result = engine.attend_blocks(q,k,v,qs,ks,keep,tokens,qb,kb,d**-0.5,settings.get("causal",False))
        else:
            keep = keep.flatten(0,1)
            if kb == 128:
                keep = keep.repeat_interleave(2, -1)[..., :(tokens + 63) // 64]
            bits = (-keep.bool().int()).unsqueeze(-1).expand(*keep.shape, 2).contiguous()
            result = engine.attend(q, k, v, qs, ks, bits, tokens, qb, d ** -0.5, False, settings.get("causal", False))
        out.copy_(result[:, :t].view(b, h, t, d).transpose(1, 2))
    else:
        from shared.kernels.sparge_sparse_int8_attn import forward
        k, ks = key
        mask = attn_mask.expand(b, h, (t+127)//128, (k.size(1)+63)//64).contiguous()
        forward(q, k, mask, v, qs, ks, tensor_layout="NHD", is_causal=settings.get("causal", False),
                output_dtype=out.dtype, out=out)
        torch.autograd.graph.increment_version(out)
    return out


@torch.compiler.disable
def attention(qkv_list, block_map, *, mode="sparge", block_size=(128, 64), recycle_q=False, causal=False):
    """HND explicit block map. Direct callers use the same projection stages as the kit."""
    q, k, v = qkv_list
    qkv_list.clear()
    if isinstance(block_map, list):
        block_map = block_map.pop() if block_map else None
    if block_map is None:
        return kernels.dense(q, k, v, causal=causal)
    settings = require(q.device, q.size(-1), mode, block_size)
    if causal:
        if q.size(2) != k.size(2):
            raise ValueError("Causal Sparge block maps require equal query and key lengths")
        settings = {**settings, "causal": True}
    q, k, v = (x.transpose(1, 2) for x in (q, k, v))
    b, t, h, d = q.shape
    block_map = block_map.expand(b, h, (t + block_size[0]-1)//block_size[0], (k.size(1) + block_size[1]-1)//block_size[1])
    with torch.cuda.device(q.device):
        values = staged_quantize_v([v], settings)
        del v
        keys = staged_quantize_k([k], settings)
        del k
        query = staged_quantize_q(q, settings)
        out = q if recycle_q else torch.empty_like(q)
        return staged_attention(out, [query, keys, values], settings, block_map).transpose(1, 2)
