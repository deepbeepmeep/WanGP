# Modifications Copyright (c) 2025-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""
Copyright (c) 2024 by SageAttention team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sys
import torch

if sys.platform == 'darwin' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    raise ImportError("SageAttention is CUDA-only and is disabled on Apple Silicon MPS")

import torch.nn.functional as F
from mmgp import cuda_graphs

from shared.kernels.token_mean import token_mean

from .sageattention_patch import patch_sageattention
from .sage1_core import fp16_v, scale_output
try:  # the fused norm + RoPE + quantization kernels (fused_qk_layouts)
    from shared.kernels import qk_quant_triton as _qk_layouts
except (ImportError, OSError, RuntimeError):
    _qk_layouts = None

patch_sageattention()

from sageattention.triton.quant_per_block import per_block_int8 as per_block_int8_triton, quant_per_block_int8_kernel
from sageattention.triton.quant_per_block_varlen import per_block_int8 as per_block_int8_varlen_triton
import sageattention.triton.attn_qk_int8_per_block as attn_qk_int8_per_block
from sageattention.triton.attn_qk_int8_per_block import forward as attn_false
from sageattention.triton.attn_qk_int8_per_block_causal import forward as attn_true
from sageattention.triton.attn_qk_int8_block_varlen import forward as attn_false_varlen
from sageattention.triton.attn_qk_int8_per_block_causal_varlen import forward as attn_true_varlen

from sageattention.triton.quant_per_thread import per_thread_int8 as per_thread_int8_triton
from sageattention.triton.quant_per_thread import quant_query_per_thread_int8_kernel, quant_key_per_thread_int8_kernel

try:
    from sageattention import _fused
    if not hasattr(_fused, "transpose_pad_permute_cuda"):
        _fused = torch.ops.sageattention_fused
except:
    _fused = torch.ops.sageattention_fused

try:
    from sageattention import _qattn_sm80
    if not hasattr(_qattn_sm80, "qk_int8_sv_f16_accum_f32_attn"): 
        _qattn_sm80 = torch.ops.sageattention_qattn_sm80
    SM80_ENABLED = True
except:
    SM80_ENABLED = False

try:
    from sageattention import _qattn_sm89
    if not hasattr(_qattn_sm89, "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf"): 
        _qattn_sm89 = torch.ops.sageattention_qattn_sm89
    SM89_ENABLED = True
except:
    SM89_ENABLED = False

try:
    from sageattention import _qattn_sm90
    if not hasattr(_qattn_sm90, "qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf"): 
        _qattn_sm90 = torch.ops.sageattention_qattn_sm90
    SM90_ENABLED = True
except:
    SM90_ENABLED = False

from sageattention.quant import per_block_int8 as per_block_int8_cuda
from sageattention.quant import per_warp_int8 as per_warp_int8_cuda
from sageattention.quant import sub_mean
from sageattention.quant import per_channel_fp8

from typing import Any, List, Literal, Optional, Tuple, Union
import warnings
import os

def is_sage2_supported():
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        major, minor = torch.cuda.get_device_capability(i)
        if major < 8:
            return False
    return True

from importlib.metadata import version
sg2_version = version("sageattention")
sg2pp = sg2_version.startswith("2.2")

import subprocess
import re
import inspect
def get_cuda_version():
    try:
        output = subprocess.check_output(['nvcc', '--version']).decode()
        match = re.search(r'release (\d+)\.(\d+)', output)
        if match:
            major, minor = int(match.group(1)), int(match.group(2))
            return major, minor
    except Exception as e:
        print("Failed to get CUDA version:", e)
    return None, None

def get_cuda_arch_versions():
    cuda_archs = []
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        cuda_archs.append(f"sm{major}{minor}")
    return cuda_archs


def _device_shared_memory_limit(index: int) -> int:
    props = torch.cuda.get_device_properties(index)
    return getattr(props, "shared_memory_per_block_optin", getattr(props, "shared_memory_per_block", 0))


_CUDA_ARCHS = tuple(get_cuda_arch_versions())
_SINGLE_CUDA_DEVICE = torch.cuda.device_count() <= 1
_LOW_SHARED_MASKED_BLOCK_M = 64
_LOW_SHARED_MASKED_BLOCK_N = 64
# Upstream masked Triton with HEAD_DIM=128 asks for this much shared memory.
_UPSTREAM_MASKED_HEAD128_SHARED_BYTES = 157696
_LOW_SHARED_MASKED_TRITON_PATCH_PRINTED = False
_SHARED_MEMORY_LIMIT_BY_DEVICE = {}


def _get_device_index(device: torch.device) -> int:
    return torch.cuda.current_device() if device.index is None else device.index


def _get_cuda_arch(device: torch.device) -> str:
    idx = _get_device_index(device)
    if idx < len(_CUDA_ARCHS):
        return _CUDA_ARCHS[idx]
    return get_cuda_arch_versions()[idx]


def _get_shared_memory_limit(device: torch.device) -> int:
    idx = _get_device_index(device)
    if idx not in _SHARED_MEMORY_LIMIT_BY_DEVICE:
        _SHARED_MEMORY_LIMIT_BY_DEVICE[idx] = _device_shared_memory_limit(idx)
    return _SHARED_MEMORY_LIMIT_BY_DEVICE[idx]


def _maybe_set_device(device: torch.device):
    if _SINGLE_CUDA_DEVICE:
        return
    idx = _get_device_index(device)
    if idx != torch.cuda.current_device():
        torch.cuda.set_device(idx)


def _use_low_shared_masked_triton(device: torch.device) -> bool:
    return _get_shared_memory_limit(device) < _UPSTREAM_MASKED_HEAD128_SHARED_BYTES


def _attn_false_low_shared_masked(q, k, v, q_scale, k_scale, tensor_layout="HND", attn_mask=None, output_dtype=torch.float16, return_lse=False):
    global _LOW_SHARED_MASKED_TRITON_PATCH_PRINTED
    if not _LOW_SHARED_MASKED_TRITON_PATCH_PRINTED:
        print(f"[SageAttention] Using low-shared-memory masked Triton patch (BLOCK_M={_LOW_SHARED_MASKED_BLOCK_M}, BLOCK_N={_LOW_SHARED_MASKED_BLOCK_N}, GPU limit={_get_shared_memory_limit(q.device)} bytes).")
        _LOW_SHARED_MASKED_TRITON_PATCH_PRINTED = True

    o = torch.empty(q.shape, dtype=output_dtype, device=q.device)

    if tensor_layout == "HND":
        b, h_qo, qo_len, head_dim = q.shape
        _, h_kv, kv_len, _ = k.shape
        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(1), q.stride(2)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(1), k.stride(2)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(1), v.stride(2)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(1), o.stride(2)
    elif tensor_layout == "NHD":
        b, qo_len, h_qo, head_dim = q.shape
        _, kv_len, h_kv, _ = k.shape
        stride_bz_q, stride_h_q, stride_seq_q = q.stride(0), q.stride(2), q.stride(1)
        stride_bz_k, stride_h_k, stride_seq_k = k.stride(0), k.stride(2), k.stride(1)
        stride_bz_v, stride_h_v, stride_seq_v = v.stride(0), v.stride(2), v.stride(1)
        stride_bz_o, stride_h_o, stride_seq_o = o.stride(0), o.stride(2), o.stride(1)
    else:
        raise ValueError(f"tensor_layout {tensor_layout} not supported")

    stride_bz_mask, stride_h_mask, stride_m_mask, stride_n_mask = attn_mask.stride(0), attn_mask.stride(1), attn_mask.stride(2), attn_mask.stride(3)
    lse = torch.empty([b, h_qo, qo_len], dtype=torch.float32, device=q.device) if return_lse else torch.empty([0], dtype=torch.float32, device="cpu")
    grid = ((qo_len + _LOW_SHARED_MASKED_BLOCK_M - 1) // _LOW_SHARED_MASKED_BLOCK_M, h_qo, b)
    attn_qk_int8_per_block._attn_fwd[grid](
        q, k, v, q_scale, k_scale, o, attn_mask, lse,
        stride_bz_q, stride_h_q, stride_seq_q,
        stride_bz_k, stride_h_k, stride_seq_k,
        stride_bz_v, stride_h_v, stride_seq_v,
        stride_bz_o, stride_h_o, stride_seq_o,
        stride_bz_mask, stride_h_mask, stride_m_mask, stride_n_mask,
        qo_len, kv_len,
        h_qo, h_qo // h_kv,
        BLOCK_M=_LOW_SHARED_MASKED_BLOCK_M, BLOCK_N=_LOW_SHARED_MASKED_BLOCK_N, HEAD_DIM=head_dim,
        STAGE=1, RETURN_LSE=return_lse,
        num_warps=4,
        num_stages=3,
    )
    return o, lse


def sageattn_attention_mask_support_reason(qkv_list=None, attn_mask: torch.Tensor | None = None, device: torch.device | str | None = None, tensor_layout: str = "NHD") -> str | None:
    if qkv_list is not None:
        device = qkv_list[0].device
    if not torch.cuda.is_available():
        return "CUDA is unavailable"
    device = torch.device("cuda" if device is None else device)
    try:
        major, _ = torch.cuda.get_device_capability(_get_device_index(device))
        if major < 8:
            return f"CUDA architecture {_get_cuda_arch(device)} has no masked SageAttention path"
        if not hasattr(attn_qk_int8_per_block, "_attn_fwd"):
            return "SageAttention Triton kernel is unavailable"
        if "attn_mask" not in inspect.signature(attn_false).parameters:
            return "installed SageAttention does not expose attn_mask"
    except (TypeError, ValueError):
        return "unable to inspect installed SageAttention mask support"
    if qkv_list is None:
        return None

    q = qkv_list[0]
    if q.dtype not in (torch.float16, torch.bfloat16):
        return f"dtype {q.dtype} is unsupported"
    if q.shape[-1] > 128 and q.shape[-1] != 256:
        return f"head_dim {q.shape[-1]} is unsupported"
    return None


def sageattn_supports_attention_mask(device: torch.device | str | None = None, qkv_list=None, attn_mask: torch.Tensor | None = None, tensor_layout: str = "NHD") -> bool:
    return sageattn_attention_mask_support_reason(qkv_list, attn_mask, device, tensor_layout) is None

def sageattn(
    qkv_list,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: Optional[float] = None,
    return_lse: bool = False,
    recycle_q: bool = False,
    scale_v: bool = False,
    **kwargs: Any,
):
    """
    Automatically selects the appropriate implementation of the SageAttention kernel based on the GPU compute capability.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len.
        Default: False.

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    return_lse : bool
        Whether to return the log sum of the exponentiated attention weights. Used for cases like Ring Attention.
        Default: False.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    torch.Tensor
        The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        Shape: ``[batch_size, num_qo_heads, qo_len]``.
        Only returned if `return_lse` is True.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``.
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    - All tensors must be on the same cuda device.
    """
        
    attn_mask = kwargs.pop("attn_mask", None)
    arch = _get_cuda_arch(qkv_list[0].device)
    if attn_mask is not None:
        support_reason = sageattn_attention_mask_support_reason(qkv_list, attn_mask, tensor_layout=tensor_layout)
        if support_reason is not None:
            raise ValueError(f"Masked SageAttention is unsupported on CUDA architecture {arch}: {support_reason}")
    if qkv_list[0].shape[-1] == 256:
        if tensor_layout != "NHD" or is_causal or return_lse or sm_scale is not None:
            raise ValueError("SageAttention with head_dim 256 takes NHD tensors, without causal mask, softmax scale or LSE")
        return _sageattn_head256(qkv_list, attn_mask, scale_v, recycle_q)
    if attn_mask is not None or cuda_graphs.recording() and not cuda_kernels_capturable(qkv_list[0].device):
        return sageattn_qk_int8_pv_fp16_triton(qkv_list, tensor_layout=tensor_layout, is_causal=is_causal, sm_scale=sm_scale, return_lse=return_lse, attn_mask=attn_mask, scale_v=scale_v)
    if arch == "sm80":
        return sageattn_qk_int8_pv_fp16_cuda(qkv_list, tensor_layout=tensor_layout, is_causal=is_causal, sm_scale=sm_scale, return_lse=return_lse, pv_accum_dtype="fp32")
    elif arch == "sm86":
        return sageattn_qk_int8_pv_fp16_triton(qkv_list, tensor_layout=tensor_layout, is_causal=is_causal, sm_scale=sm_scale, return_lse=return_lse, scale_v=scale_v)
    elif arch in ("sm89", "sm120"):
        return sageattn_qk_int8_pv_fp8_cuda(qkv_list, tensor_layout=tensor_layout, is_causal=is_causal, sm_scale=sm_scale, return_lse=return_lse, recycle_q = recycle_q, **_fp8_pv_settings(arch))
    elif arch == "sm90":
        return sageattn_qk_int8_pv_fp8_cuda_sm90(qkv_list, tensor_layout=tensor_layout, is_causal=is_causal, sm_scale=sm_scale, return_lse=return_lse, pv_accum_dtype="fp32+fp32", recycle_q = recycle_q)
    else:
        raise ValueError(f"Unsupported CUDA architecture: {arch}")


def _sageattn_head256(qkv_list, attn_mask, scale_v, recycle_q):
    """sageattn of NHD q, k and v with head_dim 256 through the staged functions (no CUDA kernel, and sageattn_qk_int8_pv_fp16_triton's
    launches exceed the shared memory at 256). k is copied: staged_quantize_k subtracts its mean in place and callers may hold it."""
    q, k, v = qkv_list
    qkv_list.clear()
    settings = staged_settings(q.device, attn_mask is not None, scale_v, 256)
    quantized = [None, staged_quantize_k([k.clone()], settings), staged_quantize_v([v], settings)]
    k = v = None
    quantized[0] = staged_quantize_q(q, settings)
    return staged_attention(q if recycle_q else torch.empty_like(q, memory_format=torch.contiguous_format), quantized, settings, attn_mask)


_capturable = None


def cuda_kernels_capturable(device):
    """True when this SageAttention build launches its CUDA kernels on PyTorch's current stream: a CUDA graph can then capture them.
    Builds that launch them on the legacy default stream cannot be captured: blocks recorded for a CUDA graph use the Triton kernels."""
    global _capturable
    if _capturable is None:
        with torch.inference_mode():
            q = torch.zeros((1, 128, 1, 128), dtype=torch.float16, device=device)
            q_int8 = torch.full(q.shape, 7, dtype=torch.int8, device=device)
            q_scale, marker = torch.empty((1, 1, 4), dtype=torch.float32, device=device), torch.zeros(1, device=device)
            graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream(device)
            torch.cuda.synchronize(device)
            with torch.cuda.stream(stream):
                graph.capture_begin(capture_error_mode="relaxed")
                marker.fill_(1)
                _fused.quant_per_warp_int8_cuda(q, q_int8, q_scale, 128, 32, 0)
                graph.capture_end()
            torch.cuda.synchronize(device)
            _capturable = bool((q_int8 == 7).all())  # captured, not run: the kernel went to the capture stream
            del graph
        if not _capturable:
            print("[Sage] This SageAttention build cannot run its CUDA kernels in CUDA graphs: its slower Triton kernels are used in CUDA Graph mode. "
                  "SageAttention 2.2.0 post6 or later keeps the CUDA kernels (see docs/INSTALLATION.md).")
    return _capturable


def _fp8_pv_settings(arch):
    """sageattn's settings for the INT8 QK / FP8 PV kernels of sm89 and sm120."""
    if arch == "sm89":
        return {"qk_quant_gran": "per_thread", "pv_accum_dtype": "fp32+fp16" if sg2pp else "fp32+fp32", "smooth_v": False}
    # sm120 has accurate fp32 accumulator for fp8 mma and triton kernel is currently not usable on sm120.
    return {"qk_quant_gran": "per_warp", "pv_accum_dtype": "fp32+fp16" if sg2pp else "fp32", "smooth_v": not sg2pp}


# Staged quantization: a caller that computes q, k and v one after another quantizes each as soon as it is ready, so that at most one
# of them exists in 16 bits. Same kernels and values as sageattn (NHD layout, head_dim 64 or 128): INT8 q/k and FP8 v on sm89 and
# sm120; INT8 q/k per block and fp16 v with its Triton path (sm86, and masked attentions on every architecture: v keeps 16 bits),
# v scaled by powers of two with scale_v (shared.sage1_core.fp16_v). Heads of 65 to 127 channels are padded to 128 with zeros, as
# sageattn pads them (settings["padded_dim"]): q, k and v when they are quantized, the output in a buffer of its own; the softmax
# scale stays that of the real head_dim.
_PER_BLOCK_SETTINGS = {"qk_quant_gran": "per_block", "block_m": 128, "launch": None, "scale_v": False}
_LOW_SHARED_SETTINGS = {"qk_quant_gran": "per_block", "block_m": _LOW_SHARED_MASKED_BLOCK_M, "launch": (4, 3), "scale_v": False}
# head_dim 256 (no CUDA kernel): the same Triton kernel on every architecture, launches that fit 99 KiB of shared memory
_HEAD256_SETTINGS = {"qk_quant_gran": "per_block", "block_m": 128, "launch": (8, 1), "scale_v": False}
_MASKED_HEAD256_SETTINGS = {"qk_quant_gran": "per_block", "block_m": 64, "launch": (8, 2), "scale_v": False}
# The Triton attention of SageAttention 2.2 takes a mask and its strides, that of 2.1 does not (no masked path)
_TRITON_MASK_ARGUMENT = "mask" in attn_qk_int8_per_block._attn_fwd.arg_names


def staged_settings(device, masked=False, scale_v=False, head_dim=128):
    """sageattn's quantization settings for the staged_* functions on this device (with a mask: its masked Triton path, smaller
    blocks where shared memory is short), None where sageattn uses other kernels. A block recorded for a CUDA graph gets the Triton path
    when the CUDA kernels cannot be captured (cuda_kernels_capturable), as head_dim 256 always does. scale_v: v scaled by powers of two
    on the Triton path."""
    if 64 < head_dim < 128:
        settings = staged_settings(device, masked, scale_v)
        return None if settings is None else {**settings, "padded_dim": 128}
    if head_dim not in (64, 128, 256):
        return None
    if head_dim == 256:
        settings = _MASKED_HEAD256_SETTINGS if masked else _HEAD256_SETTINGS
    elif masked:
        settings = _LOW_SHARED_SETTINGS if _use_low_shared_masked_triton(device) else _PER_BLOCK_SETTINGS
    elif (arch := _get_cuda_arch(device)) == "sm86" or cuda_graphs.recording() and not cuda_kernels_capturable(device):
        settings = _PER_BLOCK_SETTINGS
    else:
        return _fp8_pv_settings(arch) if SM89_ENABLED and arch in ("sm89", "sm120") else None
    return {**settings, "scale_v": True} if scale_v else settings


def fused_qk_layouts(settings, head_dim):
    """Row layouts of these settings' staged q / k quantization for shared.kernels.qk_quant_triton (the norm and RoPE of q and k fused
    with their quantization): {"q": (layout, block, sm_scale, scale_eps), "k": (...)} as staged_quantize_q / staged_quantize_k
    quantize; None where the fused kernels do not reproduce the core's quantizers (the per-warp CUDA kernels of sm120 round
    differently by one LSB in places and are not validated there)."""
    if _qk_layouts is None:
        return None
    if settings["qk_quant_gran"] == "per_thread":
        return {"q": (_qk_layouts.LAYOUT_THREAD_Q, 32, 1.0, 1e-7), "k": (_qk_layouts.LAYOUT_THREAD_K, 64, 1.0, 1e-7)}
    if settings["qk_quant_gran"] == "per_block":
        return {"q": (_qk_layouts.LAYOUT_PER_BLOCK, settings["block_m"], 1.0 / head_dim ** 0.5 * 1.44269504, 0.0), "k": (_qk_layouts.LAYOUT_PER_BLOCK, 64, 1.0, 0.0)}
    return None


def _pad_dim(x_list, settings):
    """Replaces the tensor of x_list by a copy padded with zeros to settings["padded_dim"] channels, when the settings pad its heads."""
    padded = settings.get("padded_dim")
    if padded is not None and x_list[0].shape[-1] != padded:
        x_list[0] = F.pad(x_list[0], (0, padded - x_list[0].shape[-1]))


def _quant_per_block(x, x_int8, block, sm_scale):
    """per_block_int8's Triton quantization of x (NHD) into x_int8, returns the scales (batch, heads, blocks)."""
    b, tokens, heads, head_dim = x.shape
    blocks = (tokens + block - 1) // block
    scale = torch.empty((b, heads, blocks), dtype=torch.float32, device=x.device)
    quant_per_block_int8_kernel[(blocks, heads, b)](
        x, x_int8, scale, tokens, x.stride(0), x.stride(2), x.stride(1), x_int8.stride(0), x_int8.stride(2), x_int8.stride(1),
        scale.stride(0), scale.stride(1), sm_scale=sm_scale, C=head_dim, BLK=block)
    return scale


@torch.compiler.disable
def staged_quantize_v(v_list, settings):
    """FP8 v (transposed, padded, scaled per channel) with its scale and mean, or on the Triton path fp16 v with its scales (fp16_v) and
    None; v is released."""
    _pad_dim(v_list, settings)
    if settings["qk_quant_gran"] == "per_block":
        v = v_list[0]
        v_list.clear()
        return *fp16_v(v, settings["scale_v"]), None
    _maybe_set_device(v_list[0].device)
    scale_max = 2.25 if settings["pv_accum_dtype"] == "fp32+fp16" else 448.0
    return per_channel_fp8(v_list, tensor_layout="NHD", scale_max=scale_max, smooth_v=settings["smooth_v"])


@torch.compiler.disable
def staged_quantize_k(k_list, settings):
    """INT8 k minus its mean over the tokens, with its scales; k is released (the per thread and per block paths subtract the mean in
    place). settings["k_mean"] (batch or 1, heads, head_dim) FP32, when present, is the mean subtracted instead (that of a cached prefix
    whose keys are attended in a separate pass: shared.attention_kit)."""
    _pad_dim(k_list, settings)
    k = k_list[0]
    k_list.clear()
    _maybe_set_device(k.device)
    b, kv_len, heads, head_dim = k.shape
    km = settings.get("k_mean")
    km = token_mean(k, 1) if km is None else km.to(k.dtype).expand(b, -1, -1).unsqueeze(1)
    k_int8 = torch.empty(k.shape, dtype=torch.int8, device=k.device)
    blocks = (kv_len + 63) // 64
    if settings["qk_quant_gran"] == "per_warp":
        k_scale = torch.empty((b, heads, blocks), dtype=torch.float32, device=k.device)
        _fused.quant_per_block_int8_fuse_sub_mean_cuda(k, km.squeeze(1).contiguous(), k_int8, k_scale, 64, 0)
        return k_int8, k_scale
    k.sub_(km)
    if settings["qk_quant_gran"] == "per_block":
        return k_int8, _quant_per_block(k, k_int8, 64, 1.0)
    k_scale = torch.empty((b, heads, blocks * 4), dtype=torch.float32, device=k.device)
    quant_key_per_thread_int8_kernel[(blocks * 4, heads, b)](
        k, k_int8, k_scale, kv_len, k.stride(0), k.stride(2), k.stride(1), k_int8.stride(0), k_int8.stride(2), k_int8.stride(1),
        k_scale.stride(0), k_scale.stride(1), C=head_dim, BLK=64)
    return k_int8, k_scale


@torch.compiler.disable
def staged_quantize_q(q, settings):
    """INT8 q with its scales; q is not released (staged_attention can write the output into it)."""
    _maybe_set_device(q.device)
    head_dim = q.shape[-1]
    q_list = [q]
    _pad_dim(q_list, settings)  # a padded copy: q itself stays for the output
    q = q_list.pop()
    b, qo_len, heads, _ = q.shape
    q_int8 = torch.empty(q.shape, dtype=torch.int8, device=q.device)
    if settings["qk_quant_gran"] == "per_block":  # the softmax scale, in base 2, goes into q's scales
        return q_int8, _quant_per_block(q, q_int8, settings["block_m"], 1.0 / head_dim ** 0.5 * 1.44269504)
    warps = (qo_len + 127) // 128 * 4
    if settings["qk_quant_gran"] == "per_warp":
        q_scale = torch.empty((b, heads, warps), dtype=torch.float32, device=q.device)
        _fused.quant_per_warp_int8_cuda(q, q_int8, q_scale, 128, 32, 0)
    else:
        q_scale = torch.empty((b, heads, warps * 8), dtype=torch.float32, device=q.device)
        quant_query_per_thread_int8_kernel[(warps * 8, heads, b)](
            q, q_int8, q_scale, qo_len, q.stride(0), q.stride(2), q.stride(1), q_int8.stride(0), q_int8.stride(2), q_int8.stride(1),
            q_scale.stride(0), q_scale.stride(1), C=q.shape[-1], BLK=32)
    return q_int8, q_scale


@torch.compiler.disable
def staged_attention(out, quantized_list, settings, attn_mask=None, return_lse=False):
    """Attention of the staged [(q_int8, q_scale), (k_int8, k_scale), (v, v_scale, v_mean)], written into out and returned. out has
    q's shape with a contiguous last dimension: q itself, or a slice of heads of a larger output. attn_mask (per block settings
    only): SageAttention's mask layout (batch, heads, q, k), broadcastable. return_lse: (out, lse), lse (batch, heads, tokens) FP32 the
    base-2 log-sum-exp of each query's scores without the keys' smoothing mean (attentions merged by it must share that mean)."""
    (q_int8, q_scale), (k_int8, k_scale), (v, v_scale, v_mean) = quantized_list
    quantized_list.clear()
    _maybe_set_device(out.device)
    head_dim = out.shape[-1]
    if q_int8.shape[-1] != head_dim:  # padded heads: the kernels write the padded output, of which out takes the real channels
        padded = out.new_empty((*out.shape[:-1], q_int8.shape[-1]))
        result = staged_attention(padded, [(q_int8, q_scale), (k_int8, k_scale), (v, v_scale, v_mean)], {**settings, "softmax_dim": head_dim}, attn_mask, return_lse)
        del q_int8, k_int8, v, padded
        out.copy_((result[0] if return_lse else result)[..., :head_dim])
        return (out, result[1]) if return_lse else out
    if settings["qk_quant_gran"] == "per_block":  # attn_qk_int8_per_block.forward's launch, with out as its output
        b, qo_len, heads, head_dim = q_int8.shape
        kv_len = k_int8.shape[1]
        if attn_mask is not None:
            attn_mask = attn_mask.expand(b, heads, qo_len, kv_len)
        mask = ((attn_mask,), attn_mask.stride()) if attn_mask is not None else ((None,), (0, 0, 0, 0)) if _TRITON_MASK_ARGUMENT else ((), ())
        num_warps, num_stages = settings["launch"] or ((4, 3) if head_dim == 64 else (8, 4))
        lse = torch.empty((b, heads, qo_len), dtype=torch.float32, device=out.device) if return_lse else torch.empty([0], dtype=torch.float32, device="cpu")
        attn_qk_int8_per_block._attn_fwd[((qo_len + settings["block_m"] - 1) // settings["block_m"], heads, b)](
            q_int8, k_int8, v, q_scale, k_scale, out, *mask[0], lse,
            q_int8.stride(0), q_int8.stride(2), q_int8.stride(1), k_int8.stride(0), k_int8.stride(2), k_int8.stride(1),
            v.stride(0), v.stride(2), v.stride(1), out.stride(0), out.stride(2), out.stride(1), *mask[1],
            qo_len, kv_len, heads, heads // k_int8.shape[2], BLOCK_M=settings["block_m"], BLOCK_N=64, HEAD_DIM=head_dim, STAGE=1,
            RETURN_LSE=return_lse, num_warps=num_warps, num_stages=num_stages)
        out = out if v_scale is None else scale_output(out, v_scale, 2)
        return (out, lse) if return_lse else out
    inputs = (q_int8, k_int8, v, out, q_scale, k_scale, v_scale) + ((v_mean,) if settings["pv_accum_dtype"] == "fp32" and settings["smooth_v"] else ())
    lse = _cuda_kernel(settings)(*inputs, 0, 0, 3 if settings["qk_quant_gran"] == "per_thread" else 2, settings.get("softmax_dim", head_dim) ** -0.5, int(return_lse))
    return (out, lse) if return_lse else out


def _cuda_kernel(settings):
    if settings["pv_accum_dtype"] == "fp32+fp16":
        return _qattn_sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf
    if settings["pv_accum_dtype"] == "fp32+fp32":
        return _qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf
    if settings["smooth_v"]:
        return _qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn
    return _qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn


_TRITON_LSE_ARGUMENT = "RETURN_LSE" in attn_qk_int8_per_block._attn_fwd.arg_names
_lse_kernels = {}


def _returns_lse(kernel):
    """Whether a SageAttention CUDA kernel of this build takes return_lse: its operator schema, or a pybind signature naming it."""
    found = _lse_kernels.get(kernel)
    if found is None:
        schema = getattr(getattr(kernel, "default", None), "_schema", None)
        found = _lse_kernels[kernel] = any(argument.name == "return_lse" for argument in schema.arguments) if schema is not None else "return_lse" in (kernel.__doc__ or "")
    return found


# A cached prefix (shared.attention_kit.cache_prefix) kept in these settings' layout: k as staged_quantize_k makes it with the prefix's
# own mean (prefix_mean), which the keys attended with it subtract too; v in FP8 as staged_quantize_v makes it (CUDA kernels), else
# stored by the kit and quantized again for each attention (16-bit v of the Triton path). Its keys are attended in a pass of their own,
# merged with the others by log-sum-exp.
def prefix_supported(settings, device, dtype):
    """Whether this SageAttention build's kernels for the prefix's layout (prefix_settings) return the log-sum-exp (otherwise the kit
    keeps the prefix in q8 and joins it to the keys)."""
    if "padded_dim" in settings:  # a prefix of padded heads stays in q8
        return False
    settings = prefix_settings(settings, device, False)
    return _TRITON_LSE_ARGUMENT if settings["qk_quant_gran"] == "per_block" else _returns_lse(_cuda_kernel(settings))


def prefix_settings(settings, device, scale_v):
    """The settings with which a cached prefix is kept and every attention with it runs: these, or the Triton per-block path that blocks
    recorded for CUDA graphs use where the CUDA kernels cannot be captured (staged_settings), so that eager and recorded steps keep
    the prefix's layout."""
    if settings["qk_quant_gran"] == "per_block" or cuda_kernels_capturable(device):
        return settings
    return {**_PER_BLOCK_SETTINGS, "scale_v": True} if scale_v else _PER_BLOCK_SETTINGS


def prefix_fp8_v(settings):
    return settings["qk_quant_gran"] != "per_block"


def prefix_mean(k):
    """The key mean of a cached prefix k (batch, tokens, heads, head_dim) as settings["k_mean"] takes it: (batch, heads, head_dim) FP32,
    rounded to k's dtype so that the 16-bit and the fused FP32 quantizations subtract the same values."""
    return token_mean(k, 1)[:, 0].float()


def prefix_mean_heads(mean, heads):
    return mean[:, heads.start:heads.stop]


def prefix_quantize_k(k_list, settings, mean):
    """A prefix's k (released) quantized with mean, as staged_quantize_k makes it: (k_int8, k_scale)."""
    return staged_quantize_k(k_list, {**settings, "k_mean": mean})


def prefix_k(k_int8, k_scale, heads, batch):
    """The (k_int8, k_scale) of staged_attention for the heads of a cached prefix (k_int8 (batch, tokens, heads, head_dim), k_scale
    (batch, heads, scales)) and batch items (one prefix item serves them all)."""
    k_int8, k_scale = k_int8[:, :, heads.start:heads.stop], k_scale[:, heads.start:heads.stop]
    if k_int8.shape[0] != batch:
        k_int8, k_scale = k_int8.expand(batch, -1, -1, -1), k_scale.expand(batch, -1, -1)
    return k_int8.contiguous(), k_scale.contiguous()


def prefix_v(v, v_scale, v_mean, heads, batch):
    """The (v, v_scale, v_mean) of staged_attention for the heads of a cached prefix's FP8 v (batch, head_dim, heads, padded tokens)."""
    v, v_scale = v[:, :, heads.start:heads.stop], v_scale[:, heads.start:heads.stop]
    v_mean = None if v_mean is None else v_mean[:, heads.start:heads.stop]
    if v.shape[0] != batch:
        v, v_scale = v.expand(batch, -1, -1, -1), v_scale.expand(batch, -1, -1)
        v_mean = None if v_mean is None else v_mean.expand(batch, -1, -1)
    return v.contiguous(), v_scale.contiguous(), None if v_mean is None else v_mean.contiguous()

@torch.compiler.disable
def sageattn_qk_int8_pv_fp16_triton(
    qkv_list,
    # q: torch.Tensor, 
    # k: torch.Tensor, 
    # v: torch.Tensor, 
    tensor_layout: str = "HND",
    quantization_backend: str = "triton",
    is_causal: bool =False, 
    sm_scale: Optional[float] = None, 
    smooth_k: bool = True,
    return_lse: bool = False,
    attn_mask: Optional[torch.Tensor] = None,
    scale_v: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """
    SageAttention with per-block INT8 quantization for Q and K, FP16 PV with FP16 accumulation, implemented using Triton.
    scale_v: v scaled by powers of two (shared.sage1_core.fp16_v), for v large enough to overflow the FP16 accumulation.
    The FP16 accumulator is added to a FP32 buffer immediately after each iteration.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    quantization_backend : str
        The quantization backend, either "triton" or "cuda".
        "cuda" backend offers better performance due to kernel fusion.

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len.
        Default: False.

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    smooth_k : bool
        Whether to smooth the key tensor by subtracting the mean along the sequence dimension.
        Default: True.

    return_lse : bool
        Whether to return the log sum of the exponentiated attention weights. Used for cases like Ring Attention.
        Default: False.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    torch.Tensor
        The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        Shape: ``[batch_size, num_qo_heads, qo_len]``.
        Only returned if `return_lse` is True.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``. 
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16``, ``torch.bfloat16`` or ``torch.float32``.
    - All tensors must be on the same cuda device.
    - `smooth_k` will introduce slight overhead but will improve the accuracy under most circumstances.
    """
    q, k, v = qkv_list
    qkv_list.clear()
    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."
    if attn_mask is not None:
        assert not is_causal, "SageAttention does not support attn_mask with causal attention."
        assert attn_mask.dtype == torch.bool or attn_mask.dtype == dtype, "attn_mask must be bool or match q dtype."

    # FIXME(DefTruth): make sage attention work compatible with distributed 
    # env, for example, xDiT which launch by torchrun. Without this workaround, 
    # sage attention will run into illegal memory access error after first 
    # inference step in distributed env for multi gpus inference. This small
    # workaround also make sage attention work compatible with torch.compile
    # through non-fullgraph compile mode.
    _maybe_set_device(v.device)

    head_dim_og = q.size(-1)
    masked_low_shared = attn_mask is not None and _use_low_shared_masked_triton(q.device)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    # assert last dim is contiguous
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."

    seq_dim = 1 if tensor_layout == "NHD" else 2

    if smooth_k:
        km = token_mean(k, seq_dim)
        if return_lse:
            if tensor_layout == "NHD":
                lse_correction = torch.matmul(q.transpose(1, 2), km.transpose(1, 2).transpose(2, 3)).squeeze(-1).to(torch.float32)
            else:
                lse_correction = torch.matmul(q, km.transpose(2, 3)).squeeze(-1).to(torch.float32)
    else:
        km = None

    v, v_scale = fp16_v(v, scale_v, (0, seq_dim))

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim_og ** 0.5)

    if quantization_backend == "triton":
        if masked_low_shared:
            q_int8, q_scale, k_int8, k_scale = per_block_int8_triton(q, k, km=km, BLKQ=_LOW_SHARED_MASKED_BLOCK_M, BLKK=_LOW_SHARED_MASKED_BLOCK_N, sm_scale=sm_scale, tensor_layout=tensor_layout)
        else:
            q_int8, q_scale, k_int8, k_scale = per_block_int8_triton(q, k, km=km, sm_scale=sm_scale, tensor_layout=tensor_layout)
    elif quantization_backend == "cuda":
        q_int8, q_scale, k_int8, k_scale = per_block_int8_cuda(q, k, km=km, sm_scale=sm_scale, tensor_layout=tensor_layout)
    else:
        raise ValueError(f"Unsupported quantization backend: {quantization_backend}")
    del q,k, km

    if attn_mask is not None:
        target_shape = (
            (q_int8.shape[0], q_int8.shape[2], q_int8.shape[1], k_int8.shape[1])
            if tensor_layout == "NHD"
            else (q_int8.shape[0], q_int8.shape[1], q_int8.shape[2], k_int8.shape[2])
        )
        if attn_mask.shape != target_shape:
            attn_mask = attn_mask.expand(target_shape)

    if is_causal:
        o, lse = attn_true(q_int8, k_int8, v, q_scale, k_scale, tensor_layout=tensor_layout, output_dtype=dtype, return_lse=return_lse)
    elif masked_low_shared:
        o, lse = _attn_false_low_shared_masked(q_int8, k_int8, v, q_scale, k_scale, tensor_layout=tensor_layout, output_dtype=dtype, return_lse=return_lse, attn_mask=attn_mask)
    elif attn_mask is not None:
        o, lse = attn_false(q_int8, k_int8, v, q_scale, k_scale, tensor_layout=tensor_layout, output_dtype=dtype, return_lse=return_lse, attn_mask=attn_mask)
    else:
        o, lse = attn_false(q_int8, k_int8, v, q_scale, k_scale, tensor_layout=tensor_layout, output_dtype=dtype, return_lse=return_lse)
    if v_scale is not None:
        scale_output(o, v_scale, 3 - seq_dim)  # heads: dimension 2 in NHD, 1 in HND

    o = o[..., :head_dim_og]

    if return_lse:
        return o, lse / 1.44269504 + lse_correction * sm_scale if smooth_k else lse / 1.44269504
    else:
        return o

@torch.compiler.disable
def sageattn_varlen(
    q: torch.Tensor, 
    k: torch.Tensor, 
    v: torch.Tensor, 
    cu_seqlens_q: torch.Tensor, 
    cu_seqlens_k: torch.Tensor, 
    max_seqlen_q: int, 
    max_seqlen_k: int, 
    is_causal: bool = False,
    sm_scale: Optional[float] = None, 
    smooth_k: bool = True,
    **kwargs: Any,
) -> torch.Tensor:
    """

    Parameters
    ----------
    q : torch.Tensor
        The query tensor, shape: ``[cu_seqlens_q[-1], num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor, shape: ``[cu_seqlens_k[-1], num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor, shape: ``[cu_seqlens_k[-1], num_kv_heads, head_dim]``.

    cu_seqlens_q : torch.Tensor
        The cumulative sequence lengths for the query sequences in the batch, used to index into `q`. 
        Shape: ``[batch_size + 1]``, where each entry represents the cumulative length of sequences up to that batch index.

    cu_seqlens_k : torch.Tensor
        The cumulative sequence lengths for the key and value sequences in the batch, used to index into `k` and `v`. 
        Shape: ``[batch_size + 1]``, where each entry represents the cumulative length of sequences up to that batch index.

    max_seqlen_q : int
        The maximum sequence length for the query tensor in the batch.
    
    max_seqlen_k : int
        The maximum sequence length for the key and value tensors in the batch.

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len for each sequence.
        Default: False.
    
    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    smooth_k : bool
        Whether to smooth the key tensor by subtracting the mean along the sequence dimension.
        Default: True.

    Returns
    -------
    torch.Tensor
        The output tensor, shape: ``[cu_seqlens_q[-1], num_qo_heads, head_dim]``.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``.
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16``, ``torch.bfloat16`` or ``torch.float32``.
    - The tensors `cu_seqlens_q` and `cu_seqlens_k` must have the dtype ``torch.int32`` or ``torch.int64``.
    - All tensors must be on the same cuda device.
    - `smooth_k` will introduce slight overhead but will improve the accuracy under most circumstances.
    """
    
    dtype = q.dtype
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    # FIXME(DefTruth): make sage attention work compatible with distributed 
    # env, for example, xDiT which launch by torchrun. Without this workaround, 
    # sage attention will run into illegal memory access error after first 
    # inference step in distributed env for multi gpus inference. This small
    # workaround also make sage attention work compatible with torch.compile
    # through non-fullgraph compile mode.
    _maybe_set_device(v.device)

    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."
    assert cu_seqlens_q.is_contiguous() and cu_seqlens_k.is_contiguous(), "cu_seqlens_q and cu_seqlens_k must be contiguous."

    if dtype == torch.bfloat16 or dtype == torch.float32:
        v = v.to(torch.float16)

    if smooth_k:
        km = token_mean(k, 0) # ! km is calculated on the all the batches. Calculate over each individual sequence requires dedicated kernel.
        k = k - km

    if sm_scale is None:
        sm_scale = 1.0 / (head_dim_og ** 0.5)

    q_int8, q_scale, k_int8, k_scale, cu_seqlens_q_scale, cu_seqlens_k_scale = per_block_int8_varlen_triton(q, k, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, sm_scale=sm_scale)

    if is_causal:
        o = attn_true_varlen(q_int8, k_int8, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, q_scale, k_scale, cu_seqlens_q_scale, cu_seqlens_k_scale, output_dtype=dtype)
    else:
        o = attn_false_varlen(q_int8, k_int8, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, q_scale, k_scale, cu_seqlens_q_scale, cu_seqlens_k_scale, output_dtype=dtype)

    o = o[..., :head_dim_og]

    return o

@torch.compiler.disable
def sageattn_qk_int8_pv_fp16_cuda(
    qkv_list,
    # q: torch.Tensor, 
    # k: torch.Tensor, 
    # v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """
    SageAttention with INT8 quantization for Q and K, FP16 PV with FP16/FP32 accumulation, implemented using CUDA.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len.
        Default: False.

    qk_quant_gran : str
        The granularity of quantization for Q and K, either "per_warp" or "per_thread".
        Default: "per_thread".

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    pv_accum_dtype : str
        The dtype of the accumulation of the product of the value tensor and the attention weights, either "fp16", "fp16+fp32" or "fp32".
        - "fp16": PV accumulation is done in fully in FP16. This is the fastest option but may lead to numerical instability. `smooth_v` option will increase the accuracy in cases when the value tensor has a large bias (like in CogVideoX-2b).
        - "fp32": PV accumulation is done in FP32. This is the most accurate option but may be slower than "fp16" due to CUDA core overhead.
        - "fp16+fp32": PV accumulation is done in FP16, but added to a FP32 buffer every few iterations. This offers a balance between speed and accuracy.
        Default: "fp32".

    smooth_k : bool
        Whether to smooth the key tensor by subtracting the mean along the sequence dimension.
        Default: True.
    
    smooth_v : bool
        Whether to smooth the value tensor by subtracting the mean along the sequence dimension.
        smooth_v will be ignored if pv_accum_dtype is "fp32" or "fp16+fp32".
        Default: False.

    return_lse : bool
        Whether to return the log sum of the exponentiated attention weights. Used for cases like Ring Attention.
        Default: False.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    torch.Tensor
        The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        Shape: ``[batch_size, num_qo_heads, qo_len]``.
        Only returned if `return_lse` is True.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``. 
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    - All tensors must be on the same cuda device.
    - `smooth_k` will introduce slight overhead but will improve the accuracy under most circumstances.
    """
    q,k,v = qkv_list
    qkv_list.clear() 
    dtype = q.dtype
    assert SM80_ENABLED, "SM80 kernel is not available. make sure you GPUs with compute capability 8.0 or higher."
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert qk_quant_gran in ["per_warp", "per_thread"], "qk_quant_gran must be either 'per_warp' or 'per_thread'."
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    # FIXME(DefTruth): make sage attention work compatible with distributed 
    # env, for example, xDiT which launch by torchrun. Without this workaround, 
    # sage attention will run into illegal memory access error after first 
    # inference step in distributed env for multi gpus inference. This small
    # workaround also make sage attention work compatible with torch.compile
    # through non-fullgraph compile mode.
    _maybe_set_device(v.device)

    _tensor_layout = 0 if tensor_layout == "NHD" else 1
    _is_caual = 1 if is_causal else 0
    _qk_quant_gran = 3 if qk_quant_gran == "per_thread" else 2
    _return_lse = 1 if return_lse else 0

    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    # assert last dim is contiguous
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."

    if sm_scale is None:
        sm_scale = head_dim_og**-0.5

    seq_dim = 1 if _tensor_layout == 0 else 2

    if smooth_k:
        km = token_mean(k, seq_dim)
        if return_lse:
            if tensor_layout == "NHD":
                lse_correction = torch.matmul(q.transpose(1, 2), km.transpose(1, 2).transpose(2, 3)).squeeze(-1).to(torch.float32)
            else:
                lse_correction = torch.matmul(q, km.transpose(2, 3)).squeeze(-1).to(torch.float32)
    else:
        km = None

    if qk_quant_gran == "per_warp":
        q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(q, k, km, tensor_layout=tensor_layout, BLKQ=128, WARPQ=(16 if (q.size(-1) == 128 and pv_accum_dtype == "fp16+fp32") else 32), BLKK=64)
    elif qk_quant_gran == "per_thread":
        q_int8, q_scale, k_int8, k_scale = per_thread_int8_triton(q, k, km, tensor_layout=tensor_layout, BLKQ=128, WARPQ=(16 if (q.size(-1) == 128 and pv_accum_dtype == "fp16+fp32") else 32), BLKK=64, WARPK=64)

    q_size = q.size()
    q_device = q.device
    del q,k, km
    o = torch.empty(q_size, dtype=dtype, device=q_device)

    if pv_accum_dtype in ["fp32", "fp16+fp32"] and smooth_v:
        warnings.warn(f"pv_accum_dtype is {pv_accum_dtype}, smooth_v will be ignored.")
        smooth_v = False

    if pv_accum_dtype == 'fp32':
        v = v.to(torch.float16)
        lse = _qattn_sm80.qk_int8_sv_f16_accum_f32_attn(q_int8, k_int8, v, o, q_scale, k_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
    elif pv_accum_dtype == "fp16":
        if smooth_v:
            smoothed_v, vm = sub_mean(v, tensor_layout=tensor_layout)
            del v
            lse = _qattn_sm80.qk_int8_sv_f16_accum_f16_fuse_v_mean_attn(q_int8, k_int8, smoothed_v, o, q_scale, k_scale, vm, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
        else:
            v = v.to(torch.float16)
            lse = _qattn_sm80.qk_int8_sv_f16_accum_f16_attn(q_int8, k_int8, v, o, q_scale, k_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
    elif pv_accum_dtype == "fp16+fp32":
        v = v.to(torch.float16)
        lse = _qattn_sm80.qk_int8_sv_f16_accum_f16_attn_inst_buf(q_int8, k_int8, v, o, q_scale, k_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
    else:
        raise ValueError(f"Unsupported pv_accum_dtype: {pv_accum_dtype}")

    o = o[..., :head_dim_og]

    if return_lse:
        return o, lse / 1.44269504 + lse_correction * sm_scale if smooth_k else lse / 1.44269504
    else:
        return o

@torch.compiler.disable
def sageattn_qk_int8_pv_fp8_cuda(
    qkv_list,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = None,
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
    recycle_q: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    if pv_accum_dtype == None:
        pv_accum_dtype = "fp32+fp16" if sg2pp else "fp32+fp32"
        
    """
    SageAttention with INT8 quantization for Q and K, FP8 PV with FP32 accumulation, implemented using CUDA.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len.
        Default: False.

    qk_quant_gran : str
        The granularity of quantization for Q and K, either "per_warp" or "per_thread".
        Default: "per_thread".

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    pv_accum_dtype : str
        The dtype of the accumulation of the product of the value tensor and the attention weights, either "fp32" or "fp32+fp32".
        - "fp32": PV accumulation is done in fully in FP32. However, due to the hardware issue, there are only 22 valid bits in the FP32 accumulator.
        - "fp32+fp32": PV accumulation is done in FP32 (actually FP22), but added to a FP32 buffer every few iterations. This offers a balance between speed and accuracy.
        Default: "fp32+fp32".
        
    smooth_k : bool
        Whether to smooth the key tensor by subtracting the mean along the sequence dimension.
        Default: True.
    
    smooth_v : bool
        Whether to smooth the value tensor by subtracting the mean along the sequence dimension.
        smooth_v will be ignored if pv_accum_dtype is "fp32+fp32".
        Default: False.

    return_lse : bool
        Whether to return the log sum of the exponentiated attention weights. Used for cases like Ring Attention.
        Default: False.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

            torch.Tensor
        The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        Shape: ``[batch_size, num_qo_heads, qo_len]``.
        Only returned if `return_lse` is True.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``. 
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    - All tensors must be on the same cuda device.
    - `smooth_k` will introduce slight overhead but will improve the accuracy under most circumstances.
    """
    q, k, v = qkv_list
    qkv_list.clear()

    dtype = q.dtype
    assert SM89_ENABLED, "SM89 kernel is not available. Make sure you GPUs with compute capability 8.9."
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert qk_quant_gran in ["per_warp", "per_thread"], "qk_quant_gran must be either 'per_warp' or 'per_thread'."
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    # if sg2pp:
    #     cuda_major_version, cuda_minor_version = get_cuda_version()
    #     if(cuda_major_version, cuda_minor_version) < (12, 8) and pv_accum_dtype == 'fp32+fp16':
    #         warnings.warn("cuda version < 12.8, change pv_accum_dtype to 'fp32+fp32'")
    #         pv_accum_dtype = 'fp32+fp32'

    # FIXME(DefTruth): make sage attention work compatible with distributed 
    # env, for example, xDiT which launch by torchrun. Without this workaround, 
    # sage attention will run into illegal memory access error after first 
    # inference step in distributed env for multi gpus inference. This small
    # workaround also make sage attention work compatible with torch.compile
    # through non-fullgraph compile mode.
    _maybe_set_device(v.device)

    _tensor_layout = 0 if tensor_layout == "NHD" else 1
    _is_caual = 1 if is_causal else 0
    _qk_quant_gran = 3 if qk_quant_gran == "per_thread" else 2
    _return_lse = 1 if return_lse else 0

    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    # assert last dim is contiguous
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."

    if sm_scale is None:
        sm_scale = head_dim_og**-0.5
    seq_dim = 1 if _tensor_layout == 0 else 2

    if pv_accum_dtype == 'fp32+fp32' and smooth_v:
        warnings.warn("pv_accum_dtype is 'fp32+fp32', smooth_v will be ignored.")
        smooth_v = False

    # v_list = v
    v_list = [v]
    del v
    if sg2pp:
        if pv_accum_dtype == 'fp32+fp16' and smooth_v:
            warnings.warn("pv_accum_dtype is 'fp32+fp16', smooth_v will be ignored.")
            smooth_v = False

        quant_v_scale_max = 448.0
        if pv_accum_dtype == 'fp32+fp16':
            quant_v_scale_max = 2.25
        v_fp8, v_scale, vm = per_channel_fp8(v_list, tensor_layout=tensor_layout, scale_max=quant_v_scale_max, smooth_v=smooth_v)
    else:
        v_fp8, v_scale, vm = per_channel_fp8(v_list, tensor_layout=tensor_layout, smooth_v=smooth_v)
    # del v

    if smooth_k:
        km = token_mean(k, seq_dim)
        if return_lse:
            if tensor_layout == "NHD":
                lse_correction = torch.matmul(q.transpose(1, 2), km.transpose(1, 2).transpose(2, 3)).squeeze(-1).to(torch.float32)
            else:
                lse_correction = torch.matmul(q, km.transpose(2, 3)).squeeze(-1).to(torch.float32)
    else:
        km = None

    if qk_quant_gran == "per_warp":
        q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(q, k, km, tensor_layout=tensor_layout, BLKQ=128, WARPQ=32, BLKK=64)
    elif qk_quant_gran == "per_thread":
        q_int8, q_scale, k_int8, k_scale = per_thread_int8_triton(q, k, km, tensor_layout=tensor_layout, BLKQ=128, WARPQ=32, BLKK=64, WARPK=64)
    q_size = q.size()
    q_device = q.device
    if recycle_q:
        del k,km
        o = q
    else:
        del q,k,km
        o = torch.empty(q_size, dtype=dtype, device=q_device)
    if pv_accum_dtype == "fp32":
        if smooth_v:
            lse = _qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, vm, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
        else:
            lse = _qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
    elif pv_accum_dtype == "fp32+fp32":
        lse = _qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
    elif pv_accum_dtype == "fp32+fp16":
        lse = _qattn_sm89.qk_int8_sv_f8_accum_f16_fuse_v_scale_attn_inst_buf(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)


    o = o[..., :head_dim_og]

    if return_lse:
        return o, lse / 1.44269504 + lse_correction * sm_scale if smooth_k else lse / 1.44269504
    else:
        return o


@torch.compiler.disable
def sageattn_qk_int8_pv_fp8_window_cuda(
    qkv_list,
    # q: torch.Tensor, 
    # k: torch.Tensor, 
    # v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32+fp32",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
    window = -1,
    recycle_q: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """
    SageAttention with INT8 quantization for Q and K, FP8 PV with FP32 accumulation, implemented using CUDA.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len.
        Default: False.

    qk_quant_gran : str
        The granularity of quantization for Q and K, either "per_warp" or "per_thread".
        Default: "per_thread".

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    pv_accum_dtype : str
        The dtype of the accumulation of the product of the value tensor and the attention weights, either "fp32" or "fp32+fp32".
        - "fp32": PV accumulation is done in fully in FP32. However, due to the hardware issue, there are only 22 valid bits in the FP32 accumulator.
        - "fp32+fp32": PV accumulation is done in FP32 (actually FP22), but added to a FP32 buffer every few iterations. This offers a balance between speed and accuracy.
        Default: "fp32+fp32".
        
    smooth_k : bool
        Whether to smooth the key tensor by subtracting the mean along the sequence dimension.
        Default: True.
    
    smooth_v : bool
        Whether to smooth the value tensor by subtracting the mean along the sequence dimension.
        smooth_v will be ignored if pv_accum_dtype is "fp32+fp32".
        Default: False.

    return_lse : bool
        Whether to return the log sum of the exponentiated attention weights. Used for cases like Ring Attention.
        Default: False.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

            torch.Tensor
        The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        Shape: ``[batch_size, num_qo_heads, qo_len]``.
        Only returned if `return_lse` is True.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``. 
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    - All tensors must be on the same cuda device.
    - `smooth_k` will introduce slight overhead but will improve the accuracy under most circumstances.
    """
    q,k,v = qkv_list
    qkv_list.clear()
    dtype = q.dtype
    assert SM89_ENABLED, "SM89 kernel is not available. Make sure you GPUs with compute capability 8.9."
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert qk_quant_gran in ["per_warp", "per_thread"], "qk_quant_gran must be either 'per_warp' or 'per_thread'."
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    # FIXME(DefTruth): make sage attention work compatible with distributed 
    # env, for example, xDiT which launch by torchrun. Without this workaround, 
    # sage attention will run into illegal memory access error after first 
    # inference step in distributed env for multi gpus inference. This small
    # workaround also make sage attention work compatible with torch.compile
    # through non-fullgraph compile mode.
    _maybe_set_device(v.device)

    _tensor_layout = 0 if tensor_layout == "NHD" else 1
    _is_caual = 1 if is_causal else 0
    _qk_quant_gran = 3 if qk_quant_gran == "per_thread" else 2
    _return_lse = 1 if return_lse else 0

    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    # assert last dim is contiguous
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."

    if sm_scale is None:
        sm_scale = head_dim_og**-0.5

    seq_dim = 1 if _tensor_layout == 0 else 2

    if pv_accum_dtype == 'fp32+fp32' and smooth_v:
        warnings.warn("pv_accum_dtype is 'fp32+fp32', smooth_v will be ignored.")
        smooth_v = False

    v_list = [v]
    del v
    v_fp8, v_scale, vm = per_channel_fp8(v_list, tensor_layout=tensor_layout, smooth_v=smooth_v)

    if smooth_k:
        km = token_mean(k, seq_dim)
        if return_lse:
            if tensor_layout == "NHD":
                lse_correction = torch.matmul(q.transpose(1, 2), km.transpose(1, 2).transpose(2, 3)).squeeze(-1).to(torch.float32)
            else:
                lse_correction = torch.matmul(q, km.transpose(2, 3)).squeeze(-1).to(torch.float32)
    else:
        km = None

    if qk_quant_gran == "per_warp":
        q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(q, k, km, tensor_layout=tensor_layout, BLKQ=128, WARPQ=32, BLKK=64)
    elif qk_quant_gran == "per_thread":
        q_int8, q_scale, k_int8, k_scale = per_thread_int8_triton(q, k, km, tensor_layout=tensor_layout, BLKQ=128, WARPQ=32, BLKK=64, WARPK=64)

    q_size = q.size()
    q_device = q.device
    if recycle_q:
        del k
        o = q
    else:
        del q,k
        o = torch.empty(q_size, dtype=dtype, device=q_device)

    if pv_accum_dtype == "fp32":
        if smooth_v:
            lse = _qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_fuse_v_mean_attn(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, vm, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse, window)
        else:
            lse = _qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse,  window)
    elif pv_accum_dtype == "fp32+fp32":
        lse = _qattn_sm89.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse,  window)

    o = o[..., :head_dim_og]

    if return_lse:
        return o, lse / 1.44269504 + lse_correction * sm_scale if smooth_k else lse / 1.44269504
    else:
        return o

@torch.compiler.disable
def sageattn_qk_int8_pv_fp8_cuda_sm90(
    qkv_list,
    # q: torch.Tensor, 
    # k: torch.Tensor, 
    # v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32+fp32",
    smooth_k: bool = True,
    return_lse: bool = False,
    recycle_q: bool = False,
    **kwargs: Any,
) -> torch.Tensor:
    """
    SageAttention with INT8 quantization for Q and K, FP8 PV with FP32 accumulation, implemented using CUDA.

    Parameters
    ----------
    q : torch.Tensor
        The query tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

    k : torch.Tensor
        The key tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    v : torch.Tensor
        The value tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_kv_heads, kv_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, kv_len, num_kv_heads, head_dim]``.

    tensor_layout : str
        The tensor layout, either "HND" or "NHD".
        Default: "HND".

    is_causal : bool
        Whether to apply causal mask to the attention matrix. Only applicable when qo_len == kv_len.
        Default: False.

    qk_quant_gran : str
        The granularity of quantization for Q and K, either "per_warp" or "per_thread".
        Default: "per_thread".

    sm_scale : Optional[float]
        The scale used in softmax, if not provided, will be set to ``1.0 / sqrt(head_dim)``.

    pv_accum_dtype : str
        The dtype of the accumulation of the product of the value tensor and the attention weights, either "fp32" or "fp32+fp32".
        - "fp32": PV accumulation is done in fully in FP32. However, due to the hardware issue, there are only 22 valid bits in the FP32 accumulator.
        - "fp32+fp32": PV accumulation is done in FP32 (actually FP22), but added to a FP32 buffer every few iterations. This offers a balance between speed and accuracy.
        Default: "fp32+fp32".
        
    smooth_k : bool
        Whether to smooth the key tensor by subtracting the mean along the sequence dimension.
        Default: True.

    return_lse : bool
        Whether to return the log sum of the exponentiated attention weights. Used for cases like Ring Attention.
        Default: False.

    Returns
    -------
    torch.Tensor
        The output tensor. Shape:
        - If `tensor_layout` is "HND": ``[batch_size, num_qo_heads, qo_len, head_dim]``.
        - If `tensor_layout` is "NHD": ``[batch_size, qo_len, num_qo_heads, head_dim]``.

            torch.Tensor
        The logsumexp of each row of the matrix QK^T * scaling (e.g., log of the softmax normalization factor).
        Shape: ``[batch_size, num_qo_heads, qo_len]``.
        Only returned if `return_lse` is True.

    Note
    ----
    - ``num_qo_heads`` must be divisible by ``num_kv_heads``. 
    - The tensors `q`, `k`, and `v` must have the dtype ``torch.float16`` or ``torch.bfloat16``
    - All tensors must be on the same cuda device.
    - `smooth_k` will introduce slight overhead but will improve the accuracy under most circumstances.
    """
    q,k,v = qkv_list
    qkv_list.clear()
    dtype = q.dtype
    assert SM90_ENABLED, "SM90 kernel is not available. Make sure you GPUs with compute capability 9.0."
    assert q.is_cuda, "Input tensors must be on cuda."
    assert dtype in [torch.float16, torch.bfloat16], "Input tensors must be in dtype of torch.float16 or torch.bfloat16"
    assert qk_quant_gran in ["per_warp", "per_thread"], "qk_quant_gran must be either 'per_warp' or 'per_thread'."
    assert q.device == k.device == v.device, "All tensors must be on the same device."
    assert q.dtype == k.dtype == v.dtype, "All tensors must have the same dtype."

    _maybe_set_device(v.device)

    _tensor_layout = 0 if tensor_layout == "NHD" else 1
    _is_caual = 1 if is_causal else 0
    _qk_quant_gran = 3 if qk_quant_gran == "per_thread" else 2
    _return_lse = 1 if return_lse else 0

    head_dim_og = q.size(-1)

    if head_dim_og < 64:
        q = torch.nn.functional.pad(q, (0, 64 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 64 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 64 - head_dim_og))
    elif head_dim_og > 64 and head_dim_og < 128:
        q = torch.nn.functional.pad(q, (0, 128 - head_dim_og))
        k = torch.nn.functional.pad(k, (0, 128 - head_dim_og))
        v = torch.nn.functional.pad(v, (0, 128 - head_dim_og))
    elif head_dim_og > 128:
        raise ValueError(f"Unsupported head_dim: {head_dim_og}")

    # assert last dim is contiguous
    assert q.stride(-1) == 1 and k.stride(-1) == 1 and v.stride(-1) == 1, "Last dim of qkv must be contiguous."

    if sm_scale is None:
        sm_scale = head_dim_og**-0.5

    seq_dim = 1 if _tensor_layout == 0 else 2

    # pad v to multiple of 128

    # TODO: modify per_channel_fp8 kernel to handle this
    kv_len = k.size(seq_dim)
    v_pad_len = 128 - (kv_len % 128) if kv_len % 128 != 0 else 0
    if v_pad_len > 0:
        if tensor_layout == "HND":
            v = torch.cat([v, torch.zeros(v.size(0), v.size(1), v_pad_len, v.size(3), dtype=v.dtype, device=v.device)], dim=2)
        else:
            v = torch.cat([v, torch.zeros(v.size(0), v_pad_len, v.size(2), v.size(3), dtype=v.dtype, device=v.device)], dim=1)

    v_list = [v]
    del v
    v_fp8, v_scale, _ = per_channel_fp8(v_list, tensor_layout=tensor_layout, smooth_v=False)

    if smooth_k:
        km = token_mean(k, seq_dim)
        if return_lse:
            if tensor_layout == "NHD":
                lse_correction = torch.matmul(q.transpose(1, 2), km.transpose(1, 2).transpose(2, 3)).squeeze(-1).to(torch.float32)
            else:
                lse_correction = torch.matmul(q, km.transpose(2, 3)).squeeze(-1).to(torch.float32)
    else:
        km = None

    if qk_quant_gran == "per_warp":
        q_int8, q_scale, k_int8, k_scale = per_warp_int8_cuda(q, k, km, tensor_layout=tensor_layout, BLKQ=64, WARPQ=16, BLKK=128)
    elif qk_quant_gran == "per_thread":
        q_int8, q_scale, k_int8, k_scale = per_thread_int8_triton(q, k, km, tensor_layout=tensor_layout, BLKQ=64, WARPQ=16, BLKK=128, WARPK=128)

    q_size = q.size()
    q_device = q.device
    if recycle_q:
        del k
        o = q
    else:
        del q,k
        o = torch.empty(q_size, dtype=dtype, device=q_device)

    if pv_accum_dtype == "fp32":
        raise NotImplementedError("Please use pv_accum_dtype='fp32+fp32' for sm90.")
        lse = _qattn_sm90.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)
    elif pv_accum_dtype == "fp32+fp32":
        lse = _qattn_sm90.qk_int8_sv_f8_accum_f32_fuse_v_scale_attn_inst_buf(q_int8, k_int8, v_fp8, o, q_scale, k_scale, v_scale, _tensor_layout, _is_caual, _qk_quant_gran, sm_scale, _return_lse)

    o = o[..., :head_dim_og]

    if return_lse:
        return o, lse / 1.44269504 + lse_correction * sm_scale if smooth_k else lse / 1.44269504
    else:
        return o
    
_sage_per_channel_fp8 = per_channel_fp8


def xper_channel_fp8(
    v_or_list: Union[torch.Tensor, list],
    tensor_layout: str ="HND",
    scale_max: float = 448.0,
    smooth_v: bool = True
):
    _tensor_layout = 0 if tensor_layout == "NHD" else 1
    if isinstance(v_or_list, list):
        v = v_or_list[0]
        v_or_list.clear()
    else:
        v = v_or_list
    device = v.device
    if tensor_layout == "HND":
        b, h_kv, kv_len, head_dim = v.shape
        padded_len = (kv_len + 63) // 64 * 64
        v_transposed_permutted = torch.empty((b, h_kv, head_dim, padded_len), dtype=v.dtype, device=device)

    elif tensor_layout == "NHD":
        b, kv_len, h_kv, head_dim = v.shape
        padded_len = (kv_len + 63) // 64 * 64
        v_transposed_permutted = torch.empty((b, head_dim, h_kv, padded_len), dtype=v.dtype, device=device)
    
    _fused.transpose_pad_permute_cuda(v, v_transposed_permutted, _tensor_layout)
    del v
    v_fp8 = torch.empty(v_transposed_permutted.shape, dtype=torch.float8_e4m3fn, device=device)

    v_scale = torch.empty((b, h_kv, head_dim), dtype=torch.float32, device=device)
    vm = torch.empty((b, h_kv, head_dim), dtype=torch.float32, device=device)

    if smooth_v:
        _fused.mean_scale_fuse_quant_cuda(v_transposed_permutted, v_fp8, vm, v_scale, kv_len, scale_max, _tensor_layout)
        return v_fp8, v_scale, vm
    else:
        _fused.scale_fuse_quant_cuda(v_transposed_permutted, v_fp8, v_scale, kv_len, scale_max, _tensor_layout)
        return v_fp8, v_scale, None


def _install_per_channel_fp8_monkey_patch():
    global per_channel_fp8
    per_channel_fp8 = xper_channel_fp8
    sage_quant = sys.modules.get("sageattention.quant")
    if sage_quant is not None:
        sage_quant.per_channel_fp8 = xper_channel_fp8
    sage_core = sys.modules.get("sageattention.core")
    if sage_core is not None:
        sage_core.per_channel_fp8 = xper_channel_fp8


_install_per_channel_fp8_monkey_patch()
