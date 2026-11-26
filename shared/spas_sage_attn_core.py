# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""
Copyright (c) 2025 by SpargeAttn team.

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

import importlib

import torch
import triton
import triton.language as tl
from spas_sage_attn.utils import hyperparameter_check, get_block_map_meansim_fuse_quant, get_vanilla_qk_quant, block_map_lut_triton
from spas_sage_attn.triton_kernel_example import spas_sage_attn_meansim as spas_sage_attn_meansim_triton
from einops import rearrange


def _has_exports(module, names):
    return all(hasattr(module, name) for name in names)


def _load_qattn_kernel(compile_module, direct_module, attr_name, op_namespace, required_exports):
    try:
        compile_kernel = getattr(importlib.import_module(compile_module), attr_name)
        if _has_exports(compile_kernel, required_exports):
            return compile_kernel
    except ModuleNotFoundError as exc:
        if exc.name != compile_module:
            raise
    direct_kernel = importlib.import_module(direct_module)
    op_kernel = getattr(torch.ops, op_namespace)
    if _has_exports(op_kernel, required_exports):
        return op_kernel
    if _has_exports(direct_kernel, required_exports):
        return direct_kernel
    raise AttributeError(f"{direct_module} does not expose required kernels: {', '.join(required_exports)}")


def _load_fused_kernel():
    direct_kernel = importlib.import_module("spas_sage_attn._fused")
    op_kernel = torch.ops.spas_sage_attn_fused
    required_exports = ("transpose_pad_permute_cuda", "scale_fuse_quant_cuda")
    if _has_exports(op_kernel, required_exports):
        return op_kernel
    if _has_exports(direct_kernel, required_exports):
        return direct_kernel
    raise AttributeError(f"spas_sage_attn._fused does not expose required kernels: {', '.join(required_exports)}")


try:
    _qattn_sm80 = _load_qattn_kernel(
        "spas_sage_attn.sm80_compile",
        "spas_sage_attn._qattn_sm80",
        "_qattn_sm80",
        "spas_sage_attn_qattn_sm80",
        ("qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold",),
    )
    SM80_ENABLED = True
except:
    SM80_ENABLED = False

try:
    _qattn_sm89 = _load_qattn_kernel(
        "spas_sage_attn.sm89_compile",
        "spas_sage_attn._qattn_sm89",
        "_qattn_sm89",
        "spas_sage_attn_qattn_sm89",
        (
            "qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold",
            "qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold",
        ),
    )
    SM89_ENABLED = True
except:
    SM89_ENABLED = False

try:
    _qattn_sm90 = _load_qattn_kernel(
        "spas_sage_attn.sm90_compile",
        "spas_sage_attn._qattn_sm90",
        "_qattn_sm90",
        "spas_sage_attn_qattn_sm90",
        ("qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold_sm90",),
    )
    SM90_ENABLED = True
except:
    SM90_ENABLED = False

_fused = _load_fused_kernel()


def get_cuda_version():
    version = torch.version.cuda
    major, minor = version.split('.')
    return int(major), int(minor)


def get_cuda_arch_versions():
    cuda_archs = []
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        cuda_archs.append(f"sm{major}{minor}")
    return cuda_archs


# Currently get_cuda_arch_versions cannot be traced by torch.compile
_cuda_archs = get_cuda_arch_versions()


def _unpack_qkv(q, k=None, v=None):
    if k is None and v is None and isinstance(q, list):
        qkv_list = q
        q, k, v = qkv_list
        qkv_list.clear()
    return q, k, v


def _unpack_tensor(x):
    if isinstance(x, list):
        x_list = x
        x = x_list[0]
        x_list.clear()
    return x


def _contiguous_to_dtype(x, dtype):
    x = _unpack_tensor(x)
    if x.is_contiguous() and x.dtype == dtype:
        return x
    if x.dtype == dtype:
        return x.contiguous()
    out = torch.empty_like(x, dtype=dtype, memory_format=torch.contiguous_format)
    out.copy_(x)
    return out


def _quantize_v_fp8(v, arch):
    v = _unpack_tensor(v)
    b, h_kv, kv_len, head_dim = v.shape
    padded_len = (kv_len + 127) // 128 * 128
    v_transposed_permuted = torch.empty((b, h_kv, head_dim, padded_len), dtype=v.dtype, device=v.device)
    _fused.transpose_pad_permute_cuda(v, v_transposed_permuted, 1)
    del v
    v_fp8 = torch.empty(v_transposed_permuted.shape, dtype=torch.float8_e4m3fn, device=v_transposed_permuted.device)
    v_scale = torch.empty((b, h_kv, head_dim), dtype=torch.float32, device=v_transposed_permuted.device)
    scale_max = 2.25 if arch in {"sm89", "sm100", "sm120", "sm121"} else 448.0
    _fused.scale_fuse_quant_cuda(v_transposed_permuted, v_fp8, v_scale, kv_len, scale_max, 1)
    del v_transposed_permuted
    return v_fp8, v_scale


def spas_sage_attn_meansim(q, k=None, v=None, *args, **kwargs):
    q, k, v = _unpack_qkv(q, k, v)
    arch = _cuda_archs[q.device.index]
    if arch in {"sm70", "sm75"}:
        return spas_sage_attn_meansim_triton(q, k, v, *args, **kwargs)
    else:
        return spas_sage2_attn_meansim_cuda(q, k, v, *args, **kwargs)


def spas_sage2_attn_meansim_cuda(q, k=None, v=None, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, smooth_k=True, simthreshd1=0.6, cdfthreshd=0.98, pvthreshd=50, attention_sink=False, tensor_layout="HND", output_dtype=torch.float16, return_sparsity=False, recycle_q=False):
    q, k, v = _unpack_qkv(q, k, v)
    assert tensor_layout in ['HND', 'NHD']
    assert q.size(-2)>=128, "seq_len should be not less than 128."
    q_size = q.size()
    q_device = q.device
    q_dtype = q.dtype
    q_heads = q.size(-3)

    if smooth_k:
        km = k.mean(dim=-2, keepdim=True)
    else:
        km = None
    headdim = q.size(-1)

    arch = _cuda_archs[q.device.index]
    if arch == "sm90":
        lut, valid_block_num, q_int8, q_scale, k_int8, k_scale = get_block_map_meansim_fuse_quant(q, k, km, is_causal=is_causal, simthreshd1=simthreshd1, cdfthreshd=cdfthreshd, return_lut=True, attention_sink=attention_sink, BLKQ=64, BLKK=128)
    else:
        lut, valid_block_num, q_int8, q_scale, k_int8, k_scale = get_block_map_meansim_fuse_quant(q, k, km, is_causal=is_causal, simthreshd1=simthreshd1, cdfthreshd=cdfthreshd, return_lut=True, attention_sink=attention_sink, BLKQ=128, BLKK=64)
    del k, km

    if scale is None:
        scale = 1.0 / (headdim ** 0.5)

    assert headdim in [64, 128], "headdim should be in [64, 128]. For other headdim, you can use padding and specify the softmax scale."

    pvthreshd = hyperparameter_check(pvthreshd, q_heads, q_device)

    if arch in {"sm89", "sm90", "sm100", "sm120", "sm121"}:
        v_ref = [v]
        v = None
        v_fp8, v_scale = _quantize_v_fp8(v_ref, arch)

    _is_causal = 1 if is_causal else 0
    if recycle_q:
        o = q
    else:
        del q
        o = torch.empty(q_size, dtype=q_dtype, device=q_device)

    if arch in {"sm80", "sm86", "sm87"}:
        assert SM80_ENABLED
        _qattn_sm80.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(q_int8, k_int8, v, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, 1, _is_causal, 1, scale, 0)
        del v
    elif arch in {"sm89", "sm100", "sm120", "sm121"}:
        assert SM89_ENABLED
        if get_cuda_version() < (12, 8):
            f = _qattn_sm89.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold
        else:
            f = _qattn_sm89.qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold
        f(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, _is_causal, 1, scale, 0)
        del v_fp8, v_scale
    elif arch == "sm90":
        assert SM90_ENABLED
        _qattn_sm90.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold_sm90(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, _is_causal, 1, scale, 0)
        del v_fp8, v_scale
    else:
        raise ValueError(f"Unsupported CUDA architecture: {arch}")

    del q_int8, k_int8, q_scale, k_scale
    if tensor_layout == 'NHD':
        o = rearrange(o, '... H L D -> ... L H D')

    if return_sparsity:
        if is_causal is False:
            qk_sparsity = 1 - (valid_block_num.float().sum()) / (lut.size(3) * lut.size(2) * lut.size(0) * lut.size(1))
        else:
            qk_sparsity = 1 - (valid_block_num.float().sum()) / ((lut.size(3) + 2) // 2 * lut.size(2) * lut.size(0) * lut.size(1))
        del lut, valid_block_num
        return o, qk_sparsity.item()
    else:
        del lut, valid_block_num
        return o


def spas_sage2_attn_meansim_topk_cuda(q, k=None, v=None, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, smooth_k=True, simthreshd1=-0.1, cdfthreshd=None, topk=0.5, pvthreshd=50, attention_sink=False, tensor_layout="HND", output_dtype=torch.float16, return_sparsity=False, recycle_q=False):
    q, k, v = _unpack_qkv(q, k, v)
    assert tensor_layout in ['HND', 'NHD']
    assert q.size(-2)>=128, "seq_len should be not less than 128."
    q_size = q.size()
    q_device = q.device
    q_dtype = q.dtype
    q_heads = q.size(-3)

    if smooth_k:
        km = k.mean(dim=-2, keepdim=True)
    else:
        km = None
    headdim = q.size(-1)

    arch = _cuda_archs[q.device.index]
    if arch == "sm90":
        lut, valid_block_num, q_int8, q_scale, k_int8, k_scale = get_block_map_meansim_fuse_quant(q, k, km, is_causal=is_causal, simthreshd1=simthreshd1, cdfthreshd=cdfthreshd, topk=topk, return_lut=True, attention_sink=attention_sink, BLKQ=64, BLKK=128)
    else:
        lut, valid_block_num, q_int8, q_scale, k_int8, k_scale = get_block_map_meansim_fuse_quant(q, k, km, is_causal=is_causal, simthreshd1=simthreshd1, cdfthreshd=cdfthreshd, topk=topk, return_lut=True, attention_sink=attention_sink, BLKQ=128, BLKK=64)
    del k, km

    if scale is None:
        scale = 1.0 / (headdim ** 0.5)

    assert headdim in [64, 128], "headdim should be in [64, 128]. For other headdim, you can use padding and specify the softmax scale."

    pvthreshd = hyperparameter_check(pvthreshd, q_heads, q_device)

    if arch in {"sm89", "sm90", "sm100", "sm120", "sm121"}:
        v_ref = [v]
        v = None
        v_fp8, v_scale = _quantize_v_fp8(v_ref, arch)

    _is_causal = 1 if is_causal else 0
    if recycle_q:
        o = q
    else:
        del q
        o = torch.empty(q_size, dtype=q_dtype, device=q_device)

    if arch in {"sm80", "sm86", "sm87"}:
        assert SM80_ENABLED
        _qattn_sm80.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(q_int8, k_int8, v, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, 1, _is_causal, 1, scale, 0)
        del v
    elif arch in {"sm89", "sm100", "sm120", "sm121"}:
        assert SM89_ENABLED
        if get_cuda_version() < (12, 8):
            f = _qattn_sm89.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold
        else:
            f = _qattn_sm89.qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold
        f(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, _is_causal, 1, scale, 0)
        del v_fp8, v_scale
    elif arch == "sm90":
        assert SM90_ENABLED
        _qattn_sm90.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold_sm90(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, _is_causal, 1, scale, 0)
        del v_fp8, v_scale
    else:
        raise ValueError(f"Unsupported CUDA architecture: {arch}")

    del q_int8, k_int8, q_scale, k_scale
    if tensor_layout == 'NHD':
        o = rearrange(o, '... H L D -> ... L H D')

    if return_sparsity:
        if is_causal is False:
            qk_sparsity = 1 - (valid_block_num.float().sum()) / (lut.size(3) * lut.size(2) * lut.size(0) * lut.size(1))
        else:
            qk_sparsity = 1 - (valid_block_num.float().sum()) / ((lut.size(3) + 2) // 2 * lut.size(2) * lut.size(0) * lut.size(1))
        del lut, valid_block_num
        return o, qk_sparsity.item()
    else:
        del lut, valid_block_num
        return o


def block_sparse_attn_cuda(q, k=None, v=None, mask_id=None, dropout_p=0.0, scale=None, smooth_k=True, pvthreshd=50, attention_sink=False, tensor_layout="HND", output_dtype=torch.float16, return_sparsity=False, recycle_q=False):
    q, k, v = _unpack_qkv(q, k, v)
    mask_id = _unpack_tensor(mask_id)
    assert tensor_layout in ['HND', 'NHD']
    if tensor_layout == 'NHD':
        q = rearrange(q, '... L H D -> ... H L D')
        k = rearrange(k, '... L H D -> ... H L D')
        v = rearrange(v, '... L H D -> ... H L D')
    arch = _cuda_archs[q.device.index]
    v_ref = [v]
    v = None
    v = _contiguous_to_dtype(v_ref, torch.float16)
    if arch in {"sm89", "sm90", "sm100", "sm120", "sm121"}:
        v_ref = [v]
        v = None
        v_fp8, v_scale = _quantize_v_fp8(v_ref, arch)
    qk_dtype = torch.float16 if q.dtype in (torch.float32, torch.float16) else torch.bfloat16
    q_ref = [q]
    q = None
    q = _contiguous_to_dtype(q_ref, qk_dtype)
    k_ref = [k]
    k = None
    k = _contiguous_to_dtype(k_ref, qk_dtype)
    assert q.size(-2)>=128, "seq_len should be not less than 128."
    q_size = q.size()
    q_device = q.device
    q_dtype = q.dtype
    q_heads = q.size(-3)

    if smooth_k:
        km = k.mean(dim=-2, keepdim=True)
    else:
        km = None
    headdim = q.size(-1)
    
    if arch == "sm90":
        q_int8, q_scale, k_int8, k_scale = get_vanilla_qk_quant(q, k, km, 64, 128)
    else:
        q_int8, q_scale, k_int8, k_scale = get_vanilla_qk_quant(q, k, km, 128, 64)
    del k, km
    lut, valid_block_num = block_map_lut_triton(block_map=mask_id)
    del mask_id
    if scale is None:
        scale = 1.0 / (headdim ** 0.5)

    assert headdim in [64, 128], "headdim should be in [64, 128]. For other headdim, you can use padding and specify the softmax scale."

    pvthreshd = hyperparameter_check(pvthreshd, q_heads, q_device)



    if recycle_q:
        o = q
    else:
        del q
        o = torch.empty(q_size, dtype=q_dtype, device=q_device)

    if arch in {"sm80", "sm86", "sm87"}:
        assert SM80_ENABLED
        _qattn_sm80.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(q_int8, k_int8, v, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, 1, False, 1, scale, 0)
        del v
    elif arch in {"sm89", "sm100", "sm120", "sm121"}:
        assert SM89_ENABLED
        if get_cuda_version() < (12, 8):
            f = _qattn_sm89.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold
        else:
            f = _qattn_sm89.qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold
        f(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, False, 1, scale, 0)
        del v_fp8, v_scale
    elif arch == "sm90":
        assert SM90_ENABLED
        _qattn_sm90.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold_sm90(q_int8, k_int8, v_fp8, o, lut, valid_block_num, pvthreshd, q_scale, k_scale, v_scale, 1, False, 1, scale, 0)
        del v_fp8, v_scale
    else:
        raise ValueError(f"Unsupported CUDA architecture: {arch}")

    del q_int8, k_int8, q_scale, k_scale
    if tensor_layout == 'NHD':
        o = rearrange(o, '... H L D -> ... L H D')

    if return_sparsity:
        qk_sparsity = 1 - (valid_block_num.float().sum()) / (lut.size(3) * lut.size(2) * lut.size(0) * lut.size(1))
        del lut, valid_block_num
        return o, qk_sparsity.item()
    else:
        del lut, valid_block_num
        return o


# For compatibility. spas_sage2_attn_meansim_cuda and spas_sage2_attn_meansim_topk_cuda already support sm80/86/89/90/120
spas_sage_attn_meansim_cuda = spas_sage2_attn_meansim_cuda
spas_sage_attn_meansim_topk_cuda = spas_sage2_attn_meansim_topk_cuda


# ---- Staged quantization for the attention kit (shared.attention_kit): the q, k and v of a head group are quantized as soon as each
# is projected and the block-sparse kernel runs on the group's slice of the block map, so that no whole-sequence BF16 copies and no
# head-major (HND) copies exist. Same quantizers' arithmetic and same kernels as block_sparse_attn_cuda, on the kit's (batch, tokens,
# heads, head_dim) tensors through their strides.

_PV_THRESHOLD = 50
_pv_thresholds = {}


@triton.jit
def _quant_block_nhd(X, Mean, Xq, Scale, tokens, heads, sxb, sxn, sxh, sqb, sqn, sqh, ssb, ssh, D: tl.constexpr, BLK: tl.constexpr, SUB_MEAN: tl.constexpr):
    """qk_quantize's arithmetic (the mean subtracted in x's dtype, block scale max|x| / 127 + 1e-7, rounding half away from zero) on
    strided (batch, tokens, heads, D) tensors; Mean is (batch, 1, heads, D) contiguous in x's dtype."""
    block, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = block * BLK + tl.arange(0, BLK)
    cols = tl.arange(0, D)
    inside = rows[:, None] < tokens
    x = tl.load(X + batch * sxb + head * sxh + rows[:, None] * sxn + cols[None, :], inside, 0.0)
    if SUB_MEAN:
        x -= tl.load(Mean + (batch * heads + head) * D + cols)[None, :]
        x = tl.where(inside, x, 0.0)
    x = x.to(tl.float32)
    scale = tl.max(tl.abs(x)) / 127.0
    scale += 1e-7
    q = x / scale
    q += 0.5 * tl.where(q >= 0, 1, -1)
    tl.store(Xq + batch * sqb + head * sqh + rows[:, None] * sqn + cols[None, :], q.to(tl.int8), inside)
    tl.store(Scale + batch * ssb + head * ssh + block, scale)


def _quantize_block(x, mean, block):
    """(x_int8, scales (batch, heads, blocks) fp32) of x (batch, tokens, heads, head_dim), minus mean (batch, 1, heads, head_dim) when given."""
    b, tokens, heads, head_dim = x.shape
    blocks = (tokens + block - 1) // block
    x_int8 = torch.empty(x.shape, dtype=torch.int8, device=x.device)
    scale = torch.empty((b, heads, blocks), dtype=torch.float32, device=x.device)
    _quant_block_nhd[(blocks, heads, b)](x, mean if mean is not None else x, x_int8, scale, tokens, heads, x.stride(0), x.stride(1), x.stride(2),
                                          x_int8.stride(0), x_int8.stride(1), x_int8.stride(2), scale.stride(0), scale.stride(1),
                                          D=head_dim, BLK=block, SUB_MEAN=mean is not None)
    return x_int8, scale


def staged_settings(device, head_dim=128):
    """Settings of the staged_* functions on this device: the block sizes of the block map (query blocks x key blocks) and the P.V
    kernel block_sparse_attn_cuda runs there; None where its kernels are unavailable."""
    if head_dim not in (64, 128):
        return None
    arch = _cuda_archs[device.index if device.index is not None else torch.cuda.current_device()]
    if arch in {"sm80", "sm86", "sm87"} and SM80_ENABLED:
        return {"arch": arch, "block_q": 128, "block_k": 64, "fp8": False}
    if arch in {"sm89", "sm100", "sm120", "sm121"} and SM89_ENABLED:
        kernel = (_qattn_sm89.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold if get_cuda_version() < (12, 8)
                  else _qattn_sm89.qk_int8_sv_f8_accum_f16_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold)
        return {"arch": arch, "block_q": 128, "block_k": 64, "fp8": True, "scale_max": 2.25, "kernel": kernel}
    if arch == "sm90" and SM90_ENABLED:
        return {"arch": arch, "block_q": 64, "block_k": 128, "fp8": True, "scale_max": 448.0,
                "kernel": _qattn_sm90.qk_int8_sv_f8_accum_f32_block_sparse_attn_inst_buf_fuse_v_scale_with_pv_threshold_sm90}
    return None


@triton.jit
def _v_fp8_nhd(V, Low, High, V8, Scale, tokens, heads, svb, svn, svh, pad_len, scale_max, D: tl.constexpr, BLOCK: tl.constexpr):
    """scale_fuse_quant_cuda's FP8 V straight from (batch, tokens, heads, D), without its 16-bit transposed copy: per channel the scale
    amax / scale_max (amax from the channel's min and max over the tokens), the values times scale_max / amax rounded to e4m3, stored
    transposed (batch, D, heads, padded tokens) with the fp8 MMA permutation inside every 16 tokens, zeros beyond the tokens. The
    divisions are Triton's fp32 division, the instruction the CUDA kernel uses under fast math. The first block stores the scales."""
    block, head, batch = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    rows = block * BLOCK + tl.arange(0, BLOCK)
    cols = tl.arange(0, D)
    channel = (batch * heads + head) * D + cols
    amax = tl.maximum(-tl.load(Low + channel).to(tl.float32), tl.load(High + channel).to(tl.float32))
    if block == 0:
        tl.store(Scale + channel, amax / scale_max)
    x = tl.load(V + batch * svb + head * svh + rows[:, None] * svn + cols[None, :], rows[:, None] < tokens, 0.0).to(tl.float32)
    x = x * (scale_max / amax)[None, :]
    t = rows % 16
    col = (rows // 16) * 16 + (t // 8) * 2 + ((t // 2) % 4) * 4 + t % 2
    tl.store(V8 + ((batch * D + cols)[None, :] * heads + head) * pad_len + col[:, None], x.to(tl.float8e4nv))


@torch.compiler.disable
def staged_quantize_v(v_list, settings):
    """FP8 v (transposed, padded to 128, permuted, scaled per channel) with its scales, or on sm_80 fp16 v and None; v is released."""
    v = v_list[0]
    v_list.clear()
    if not settings["fp8"]:
        return v if v.dtype == torch.float16 else v.to(torch.float16), None
    b, kv_len, heads, head_dim = v.shape
    padded_len = (kv_len + 127) // 128 * 128
    low, high = torch.aminmax(v, dim=1)
    v_fp8 = torch.empty((b, head_dim, heads, padded_len), dtype=torch.float8_e4m3fn, device=v.device)
    v_scale = torch.empty((b, heads, head_dim), dtype=torch.float32, device=v.device)
    _v_fp8_nhd[(padded_len // 64, heads, b)](v, low, high, v_fp8, v_scale, kv_len, heads, v.stride(0), v.stride(1), v.stride(2), padded_len,
                                             settings["scale_max"], D=head_dim, BLOCK=64, num_warps=4, enable_fp_fusion=False)
    del v, low, high
    return v_fp8, v_scale


@torch.compiler.disable
def staged_quantize_k(k_list, settings):
    """INT8 k minus its mean over the tokens, in blocks of block_k, with its scales; k is released."""
    k = k_list[0]
    k_list.clear()
    km = k.mean(dim=1, keepdim=True)
    quantized = _quantize_block(k, km, settings["block_k"])
    del k, km
    return quantized


@torch.compiler.disable
def staged_quantize_q(q, settings):
    """INT8 q in blocks of block_q with its scales; q is not released (staged_attention can write the output into it)."""
    return _quantize_block(q, None, settings["block_q"])


def _pv_threshold(heads, device):
    key = (heads, device)
    if key not in _pv_thresholds:
        _pv_thresholds[key] = hyperparameter_check(_PV_THRESHOLD, heads, device)
    return _pv_thresholds[key]


@torch.compiler.disable
def staged_attention(out, quantized_list, settings, attn_mask=None):
    """Block-sparse attention of the staged [(q_int8, q_scale), (k_int8, k_scale), (v, v_scale)] over the block map attn_mask
    (batch, heads, query blocks of block_q, key blocks of block_k; nonzero = attended), written into out and returned. out has q's
    shape with a contiguous last dimension: q itself, or a slice of heads of a larger output."""
    (q_int8, q_scale), (k_int8, k_scale), (v, v_scale) = quantized_list
    quantized_list.clear()
    lut, valid_block_num = block_map_lut_triton(attn_mask if attn_mask.dtype == torch.int8 else attn_mask.to(torch.int8))
    options = (0, settings.get("causal", False), 1, out.shape[-1] ** -0.5, 0)  # NHD layout, per-block scales, no LSE
    pv_threshold = _pv_threshold(q_int8.shape[2], out.device)
    if not settings["fp8"]:
        _qattn_sm80.qk_int8_sv_f16_accum_f16_block_sparse_attn_inst_buf_with_pv_threshold(q_int8, k_int8, v, out, lut, valid_block_num, pv_threshold, q_scale, k_scale, *options)
    else:
        settings["kernel"](q_int8, k_int8, v, out, lut, valid_block_num, pv_threshold, q_scale, k_scale, v_scale, *options)
    return out


from shared.kernels.triton_compilation_log import install_triton_compilation_logger
install_triton_compilation_logger()
