# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Qwen partial RoPE and paged cache writes with original rounding boundaries."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit(do_not_specialize=("T", "QB", "QT", "QH", "KB", "KT", "KH", "VB", "VT", "VH", "CB", "CT", "HQ", "HK"),
            do_not_specialize_on_alignment=("T", "QB", "QT", "QH", "KB", "KT", "KH", "VB", "VT", "VH", "CB", "CT", "HQ", "HK"))
def _rope_cache_kernel(Q, K, V, COS, SIN, KC, VC, KS, VS, SLOTS,
                       T, HQ, HK,
                       D: tl.constexpr, R: tl.constexpr,
                       QB, QT, QH,
                       KB, KT, KH,
                       VB, VT, VH,
                       CB, CT,
                       WRITE_CACHE: tl.constexpr, INT8: tl.constexpr):
    token, head = tl.program_id(0), tl.program_id(1)
    batch, time = token // T, token % T
    is_query = head < HQ
    if is_query:
        ptr = Q + batch * QB + time * QT + head * QH
    else:
        ptr = K + batch * KB + time * KT + (head - HQ) * KH
    dims = tl.arange(0, D)
    value = tl.load(ptr + dims).to(tl.float32)
    partner_dim = tl.where(dims < R // 2, dims + R // 2, dims - R // 2)
    partner = tl.load(ptr + partner_dim, dims < R, 0).to(tl.float32)
    cos = tl.load(COS + batch * CB + time * CT + dims, dims < R, 1).to(tl.float32)
    sin = tl.load(SIN + batch * CB + time * CT + dims, dims < R, 0).to(tl.float32)
    # PyTorch rounds each multiplication before the in-place add/subtract.
    product = (value * cos).to(Q.dtype.element_ty).to(tl.float32)
    cross = (partner * sin).to(Q.dtype.element_ty).to(tl.float32)
    rotated = tl.where(dims < R // 2, product - cross, product + cross).to(Q.dtype.element_ty)
    result = tl.where(dims < R, rotated, value.to(Q.dtype.element_ty))
    # Partners may belong to another warp; finish all reads before overwriting.
    tl.debug_barrier()
    tl.store(ptr + dims, result, dims < R)
    if WRITE_CACHE:
        if not is_query:
            slot = tl.load(SLOTS + token)
            if slot >= 0:
                kv_head = head - HQ
                v = tl.load(V + batch * VB + time * VT + kv_head * VH + dims).to(tl.float32)
                offset = (slot * HK + kv_head) * D + dims
                if INT8:
                    k_blocks = result.to(tl.float32).reshape((D // 32, 32))
                    v_blocks = v.reshape((D // 32, 32))
                    k_scale = tl.maximum(tl.max(tl.abs(k_blocks), 1) / 127.0, 1e-8)
                    v_scale = tl.maximum(tl.max(tl.abs(v_blocks), 1) / 127.0, 1e-8)
                    k_quant = libdevice.nearbyint(k_blocks / k_scale[:, None]).to(tl.int8)
                    v_quant = libdevice.nearbyint(v_blocks / v_scale[:, None]).to(tl.int8)
                    tl.store(KC + offset, k_quant.reshape((D,)))
                    tl.store(VC + offset, v_quant.reshape((D,)))
                    scale_offset = (slot * HK + kv_head) * (D // 32) + tl.arange(0, D // 32)
                    tl.store(KS + scale_offset, k_scale)
                    tl.store(VS + scale_offset, v_scale)
                else:
                    tl.store(KC + offset, result)
                    tl.store(VC + offset, v)


from shared.kernels.dispatch import triton_dtype_supported  # after the kernels: their line numbers key Triton's cache


def supported(q, k, v, cos, sin):
    return (q.is_cuda and torch.version.hip is None and q.dtype in (torch.float16, torch.bfloat16, torch.float32) and triton_dtype_supported(q.device, q.dtype)
            and q.dtype == k.dtype == v.dtype == cos.dtype == sin.dtype
            and q.shape[:2] == k.shape[:2] == v.shape[:2] == cos.shape[:2]
            and q.shape[-1] == k.shape[-1] == v.shape[-1] and k.shape[2] == v.shape[2]
            and q.shape[-1] in (32, 64, 128, 256) and 0 < cos.shape[-1] <= q.shape[-1]
            and cos.shape[-1] % 2 == 0 and cos.shape == sin.shape and cos.stride() == sin.stride()
            and all(x.stride(-1) == 1 for x in (q, k, v, cos, sin)))


def apply_rope_cache(q, k, v, cos, sin, attention, slots):
    layout = getattr(attention, "kv_layout", None)
    # the fused write knows the BF16 and Q8 caches; a mixed-precision body (kv_formats) is written by the attention's own store
    write_cache = attention.k_cache.numel() > 0 and attention.v_cache.numel() > 0 and slots is not None and not (layout is not None and layout.mixed)
    _rope_cache_kernel[(q.shape[0] * q.shape[1], q.shape[2] + k.shape[2])](
        q, k, v, cos, sin, attention.k_cache, attention.v_cache, attention.k_scale, attention.v_scale, slots,
        q.shape[1], q.shape[2], k.shape[2], q.shape[3], cos.shape[-1],
        *q.stride()[:3], *k.stride()[:3], *v.stride()[:3], *cos.stride()[:2],
        write_cache, attention.k_cache.dtype == torch.int8, num_warps=4, enable_fp_fusion=False)
    return write_cache


from shared.kernels.triton_compilation_log import install_triton_compilation_logger
install_triton_compilation_logger()
