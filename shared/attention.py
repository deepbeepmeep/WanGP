# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Modifications Copyright (c) 2025-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import functools
from importlib.util import find_spec
import os
import sys
from contextlib import contextmanager, nullcontext

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm
from mmgp import cuda_graphs, offload
import torch.nn.functional as F
import warnings
from importlib.metadata import version
from .sageattention_patch import sageattention_version

_is_mps = sys.platform == 'darwin' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()

major, minor = torch.cuda.get_device_capability(None) if torch.cuda.is_available() else (0, 0)
bfloat16_supported =  major >= 8
_HEAD_DIM_SDPA_WARNED = False
_MISSING = object()
from . import sol_core, sparge_core, sparse_kernels

sparsegp_backend = sol_core.backend_for(torch.device("cuda", torch.cuda.current_device())) if torch.cuda.is_available() else None


def sol3_mode(mode):
    if mode == "sol3":
        from shared.kernels import kernel_policy
        if kernel_policy.sol3_dense():
            return get_default_attention_mode()
    return mode

try:
    import triton
    triton_installed = True
except:
    triton_installed = False

# Select the SM120 dense kernels of SageGP from tensor metadata. Experimental asymmetric paths retain their opt-in.
_routed = None
_wide = None
_fp4_scores = None
_precise = None
if (major, minor) == (12, 0) and triton_installed:
    from triton import language as _tl
    if hasattr(_tl, "dot_scaled") and tuple(int(n) for n in triton.__version__.split(".")[:2]) >= (3, 6):
        try:
            from sagegp import hooks as _sagegp_hooks
            from sagegp.sm120 import wide as _wide, routed as _routed, fast as _fp4_scores, dense as _precise
        except ImportError:
            pass
        else:
            from shared.kernels.dispatch import choose as _choose
            _sagegp_hooks.register_constant = cuda_graphs.register_constant
            _sagegp_hooks.notice = lambda operation: _choose(operation, owned_triton=True)


# SageGP's RTX 50xx kernels are installed: they take priority over SageAttention 2 and 3 (sage_gp_redirect)
sagegp_ready = _fp4_scores is not None
_SAGE_GP_MODES = {"sage3": "fast", "sageadaptative": "adaptive"}


def sage_gp_redirect(attn):
    """(attention mode, Sage2 quality override) of a generation. With SageGP's RTX 50xx kernels, "sage3" runs SageGP Fast (every score
    in NVFP4, SageAttention 3 speed at a fraction of its error) and "sageadaptative" SageGP Adaptive, both through the Sage2 path, which
    then serves "sage2" with SageGP Precise."""
    if attn == "sol3":
        return attn, "sol3"
    return ("sage2", _SAGE_GP_MODES[attn]) if sagegp_ready and attn in _SAGE_GP_MODES else (attn, None)


def fp4_geometry(device, dtype, batch, queries, keys, heads, kv_heads, dim, force_attention=None):
    """Whether a direct kernel can consume these projections before the standard staged INT8 packing."""
    if sol3_mode(force_attention or offload.shared_state.get("_attention")) != "sage2":
        return False
    from shared.kernels import kernel_policy
    return (_wide is not None and kernel_policy.sage2_execution_quality() in ("adaptive", "fast")
            and _wide.preferred(device, dtype, batch, queries, keys, heads, kv_heads, dim))


def owned_query_staged(staged):
    if staged is not None and staged[0] in (sage1_core, sage2_core):
        return staged[0], {**staged[1], "recycle_query": True}
    return staged


def dense_staged(staged, device, dtype, batch, tokens, heads, kv_heads, dim, force_attention, *, recycle_query=False, split_keys=False):
    """split_keys: the call packs keys split by chunks of tokens (q / k norms over all heads), which the Fast core cannot: Adaptive
    serves it instead, the fastest core that can."""
    if staged is None or staged[0] is not sage2_core:
        return owned_query_staged(staged) if recycle_query else staged
    from shared.kernels import kernel_policy
    quality = kernel_policy.sage2_execution_quality()
    sage2 = sol3_mode(force_attention or offload.shared_state.get("_attention")) == "sage2"
    if (_fp4_scores is not None and quality == "fast" and not split_keys and sage2
            and _fp4_scores.eligible(device, dtype, batch, tokens, tokens, heads, kv_heads, dim)):
        return _fp4_scores, _fp4_scores.settings(recycle_query)
    # Precise: SageGP's INT8 kernel, about 8 % faster than the staged Sage2 core at its error (keys split by chunks of tokens keep
    # the staged Sage2 core)
    if (_precise is not None and quality == "regular" and not split_keys and sage2 and tokens > 8192 and _routed.supported(device, dtype)
            and _precise.eligible(batch, tokens, tokens, heads, kv_heads, dim, dtype)):
        return _precise, {"recycle_query": recycle_query, "int8_scores": True}
    # Adaptive also serves the calls Fast cannot
    if (_routed is not None and quality in ("adaptive", "fast") and sage2
            and _routed.eligible(device, dtype, batch, tokens, tokens, heads, kv_heads, dim)):
        return _routed, _routed.settings(recycle_query)
    return owned_query_staged(staged) if recycle_query else staged


def kv_first_staged(staged, tensors, force_attention):
    if staged is None:
        return staged
    q, k, v = tensors
    if (q.shape == k.shape == v.shape and q.dtype == k.dtype == v.dtype and q.device == k.device == v.device
            and all(x.stride(-1) == 1 for x in tensors)):
        return dense_staged(staged, q.device, q.dtype, *q.shape[:2], q.size(2), k.size(2), q.size(3), force_attention,
                            recycle_query=True)
    return owned_query_staged(staged)


# Model-specific attention modes exposed only by per-generation overrides.
# Keep these out of get_attention_modes() so they never appear in the main
# application attention configuration.
ATTENTION_MODE_AVAILABILITY = {
    "sol": {
        "installed": triton_installed or sparsegp_backend is not None,
        "supported": torch.cuda.is_available() and sol_core.classic_available(torch.device("cuda", torch.cuda.current_device())),
    },
    "veda": {
        "installed": triton_installed,
        "supported": triton_installed and major >= 8 and not torch.version.hip,
    },
    "vsa": {
        "installed": triton_installed,
        "supported": triton_installed and major >= 8 and not torch.version.hip,
    },
}


try:
    from xformers.ops import memory_efficient_attention
except ImportError:
    memory_efficient_attention = None

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False
    flash_attn = None

try:
    from sageattention import sageattn_varlen
    from . import sage1_core
    def sageattn_varlen_wrapper(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
            is_causal=False,
            sm_scale=None,
            scale_v=False,
        ):
        # scale_v: v divided by sage1_core.v_scales (exact in 16 bits) before sageattn_varlen converts it to fp16, the output multiplied back
        scale = sage1_core.v_scales(v, (0,)) if scale_v else None
        o = sageattn_varlen(
            q,
            k,
            v if scale is None else v / scale.to(v.dtype),
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
            is_causal=is_causal,
            sm_scale=sm_scale,
        )
        return o if scale is None else sage1_core.scale_output(o, scale, 1)
    
except ImportError:
    sageattn_varlen_wrapper = sage1_core = None

try:
    from .sage2_core import sageattn as sageattn2, is_sage2_supported, sageattn_attention_mask_support_reason
    from . import sage2_core
    sage2_supported =  is_sage2_supported()
except ImportError:
    sageattn2 = sage2_core = None
    sage2_supported = False
    sageattn_attention_mask_support_reason = lambda *args, **kwargs: "SageAttention 2 is unavailable"
    if not triton_installed: 
        try:
            sg2_version = sageattention_version()
            if not triton_installed: print("Sage Attention 2 has been detected but it won't work until Triton is installed.")
        except ImportError:
            pass

def sage_attention_mask(attention_mask, q_dtype=None, causal_mask=None):
    """Convert WanGP token masks to broadcastable (batch, heads, queries, keys) views."""
    # Keep implicit broadcasts compact through the reductions and bias conversion.
    if 0 in attention_mask.stride():
        attention_mask = attention_mask[tuple(slice(1) if stride == 0 else slice(None) for stride in attention_mask.stride())]
    if attention_mask.ndim == 4:
        mask = attention_mask.transpose(1, 2)
    elif attention_mask.ndim == 3:
        mask = attention_mask.unsqueeze(1)
    else:
        mask = attention_mask.reshape((1,) * (4 - attention_mask.ndim) + attention_mask.shape)
    if causal_mask is not None:
        return mask & causal_mask if mask.dtype == torch.bool else mask.masked_fill(~causal_mask, -torch.inf)
    return mask


@torch.compiler.disable()
def sageattn2_wrapper(
        qkv_list,
        attention_length,
        recycle_q = False,
        attention_mask = None,
        causal = False,
        scale_v = False,
    ):
    # no reference is kept here: the kernel takes q, k and v out of the list and frees k and v once quantized
    if attention_mask is not None:
        if causal:
            q, k = qkv_list[:2]
            causal_mask = torch.arange(k.size(1), device=q.device) <= torch.arange(q.size(1), device=q.device)[:, None]
            attention_mask = sage_attention_mask(attention_mask, causal_mask=causal_mask).transpose(1, 2)
        return _constrained_attention(qkv_list, attention_mask, False, (-1, -1), None, None, None, 0., "sage2", scale_v=scale_v)
    o = sageattn2(qkv_list, tensor_layout="NHD", is_causal=causal, recycle_q=recycle_q, attn_mask=attention_mask, scale_v=scale_v)
    qkv_list.clear()

    return o

try:
    from sageattn import sageattn_blackwell as sageattn3
    if not triton_installed:
        print("Sage Attention 3 is installed but it won't be supported until Triton is installed.")
except ImportError:
    sageattn3 = None
    if not triton_installed: 
        try:
            sg3_version = version("sageattn_blackwell")
            print("Sage Attention 3 has been detected but it won't work until Triton is installed.")
        except ImportError:
            pass

if sageattn3 is None:
    try:
        from sageattn3 import sageattn3_blackwell as sageattn3 #word0 windows version
    except ImportError:
        sageattn3 = None
        if not triton_installed: 
            try:
                sg3_version = version("sageattn3_blackwell")
                print("Sage Attention 3 has been detected but it won't work until Triton is installed.")
            except ImportError:
                pass

@torch.compiler.disable()
def sageattn3_wrapper(
        qkv_list,
        attention_length
    ):
    q,k, v = qkv_list
    # qkv_list = [q,k,v]
    # del q, k ,v
    # o = sageattn3(qkv_list, tensor_layout="NHD")
    q = q.transpose(1,2)
    k = k.transpose(1,2)
    v = v.transpose(1,2)
    o = sageattn3(q, k, v)
    o = o.transpose(1,2)
    qkv_list.clear()

    return o

     


# try:
# if True:
    # from .sage2_core import sageattn_qk_int8_pv_fp8_window_cuda
    # @torch.compiler.disable()
    # def sageattn_window_wrapper(
    #         qkv_list,
    #         attention_length,
    #         window
    #     ):
    #     q,k, v = qkv_list
    #     padding_length = q.shape[0] -attention_length
    #     q = q[:attention_length, :, : ].unsqueeze(0)
    #     k = k[:attention_length, :, : ].unsqueeze(0)
    #     v = v[:attention_length, :, : ].unsqueeze(0)
    #     qkvl_list = [q, k , v]
    #     del q, k ,v
    #     o = sageattn_qk_int8_pv_fp8_window_cuda(qkvl_list, tensor_layout="NHD", window = window).squeeze(0)
    #     qkv_list.clear()

    #     if padding_length > 0:
    #         o = torch.cat([o, torch.empty( (padding_length, *o.shape[-2:]), dtype= o.dtype, device=o.device  ) ], 0)

    #     return o
# except ImportError:
#     sageattn2 = sageattn_qk_int8_pv_fp8_window_cuda

def _sdpa_requires_math(query, attention_mask):
    # Masked 512-wide attention can return finite but incorrect results on these HIP architectures.
    if not torch.version.hip or query.device.type != "cuda" or query.shape[-1] != 512 or attention_mask is None or query.dtype not in (torch.bfloat16, torch.float32):
        return False
    return torch.cuda.get_device_properties(query.device).gcnArchName.split(":", 1)[0] in ("gfx1100", "gfx1151")


@torch.compiler.disable()
def sdpa_wrapper(
        qkv_list,
        attention_length,
        attention_mask = None,
        causal = False,
        softmax_scale = None,
        dropout_p = 0.,
    ):
    q, k, v = qkv_list

    q = q.transpose(1,2)
    k = k.transpose(1,2)
    v = v.transpose(1,2)
    if attention_mask != None:
        attention_mask = attention_mask.transpose(1,2)
    with sdpa_kernel(SDPBackend.MATH) if _sdpa_requires_math(q, attention_mask) else nullcontext():
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, is_causal=causal, scale=softmax_scale, dropout_p=dropout_p).transpose(1,2)
    del q, k ,v
    qkv_list.clear()

    return o


@functools.lru_cache(maxsize=None)
def _attention_modes():
    # Installed packages do not change while the server runs; the package
    # version lookups below read dist-info files on every call otherwise.
    ret = ["sdpa", "auto"]
    if flash_attn != None:
        ret.append("flash")
    if memory_efficient_attention != None:
        ret.append("xformers")
    if sageattn_varlen_wrapper != None:
        ret.append("sage")
    if sageattn2 != None and sageattention_version().startswith("2") :
        ret.append("sage2")
    if torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
        if sparge_core.staged_settings(device, 128, block_size=sparge_core.radial_geometry(device, 128)):
            ret.append("radial")

    if sageattn3 != None or sagegp_ready:  # with SageGP, its Fast mode serves sage3 (sage_gp_redirect)
        ret.append("sage3")
    if sagegp_ready or find_spec("sagegp") is not None:
        ret.append("sageadaptative")

    if sparsegp_backend is not None:
        ret.append("sol3")
    return tuple(ret)

def get_attention_modes():
    return list(_attention_modes())

def get_supported_attention_modes():
    return list(_supported_attention_modes())

@functools.lru_cache(maxsize=None)
def _supported_attention_modes():
    # CUDA and HIP backends both expose their GPU through torch.cuda.
    if _is_mps or not torch.cuda.is_available():
        return ("sdpa", "auto")
    ret = get_attention_modes()
    if "sageadaptative" in ret and not sagegp_ready:
        ret.remove("sageadaptative")
    if "sol3" in ret and sparsegp_backend is None:
        ret.remove("sol3")
    major, minor = torch.cuda.get_device_capability()
    if  major < 10 or not triton_installed:
        if "sage3" in ret:
            ret.remove("sage3")

    if not sage2_supported or not triton_installed:
        if "sage2" in ret:
            ret.remove("sage2")


    if major < 7 or not triton_installed:
        if "sage" in ret:
            ret.remove("sage")

    return tuple(ret)


def get_override_attention_modes():
    return get_attention_modes() + [mode for mode, status in ATTENTION_MODE_AVAILABILITY.items() if status["installed"]]


def get_supported_override_attention_modes():
    return get_supported_attention_modes() + [mode for mode, status in ATTENTION_MODE_AVAILABILITY.items() if status["supported"]]


def get_default_attention_mode():
    for attn in ("sage2", "sage", "sdpa"):
        if attn in get_supported_attention_modes():
            return attn
    return "sdpa"


def get_current_cuda_architecture(device=None):
    if _is_mps or not torch.cuda.is_available():
        return None
    major, minor = torch.cuda.get_device_capability(device)
    return major * 10 + minor


_SAGE_ATTENTION_MODES = {"sage", "sage2", "sage3", "sageadaptative", "radial"}


def resolve_attention_mode(attention_mode=None, disable_sage_pre_ada=False):
    attn = str(attention_mode or "auto").strip().lower()
    attn = get_default_attention_mode() if attn == "auto" else attn
    architecture = get_current_cuda_architecture() if disable_sage_pre_ada else None
    if architecture is not None and architecture < 89 and attn in _SAGE_ATTENTION_MODES:
        attn = "sdpa"
    if attn not in get_supported_attention_modes():
        raise ValueError(f"Attention mode '{attn}' is not installed or supported on this system.")
    return attn


@contextmanager
def attention_shared_state(default_attention=None):
    previous = offload.shared_state.get("_attention", _MISSING)
    if previous is _MISSING or previous == "auto":
        offload.shared_state["_attention"] = default_attention or get_default_attention_mode()
    else:
        offload.shared_state["_attention"] = previous
    try:
        yield
    finally:
        if previous is _MISSING:
            offload.shared_state.pop("_attention", None)
        else:
            offload.shared_state["_attention"] = previous


@contextmanager
def attention_config_shared_state(attention_mode=None, resolver=resolve_attention_mode, **resolver_kwargs):
    from shared.kernels import kernel_policy
    previous = offload.shared_state.get("_attention", _MISSING)
    previous_quality = kernel_policy.sage2_quality_override
    offload.shared_state["_attention"], kernel_policy.sage2_quality_override = sage_gp_redirect(resolver(attention_mode, **resolver_kwargs))
    try:
        yield offload.shared_state["_attention"]
    finally:
        kernel_policy.sage2_quality_override = previous_quality
        if previous is _MISSING:
            offload.shared_state.pop("_attention", None)
        else:
            offload.shared_state["_attention"] = previous

def _masked_staged(mode, device, dtype, head_dim, scale_v):
    """SageGP and vanilla Sage2 expose the same Sage2 kernel ABI. Its masked core
    tries the accelerated tile kernel before the installed masked Sage2 forward."""
    if mode == "sdpa" or torch.device(device).type != "cuda" or torch.version.hip or dtype not in (None, torch.float16, torch.bfloat16):
        return None
    sage1_width = sage1_core.padded_head_dim(head_dim, device) if sage1_core is not None and sage1_core.masks_supported(device) else None
    if mode != "sage" or sage1_width is None:
        if sageattn2 is not None and sageattn_attention_mask_support_reason(device=device) is None:
            settings = sage2_core.staged_settings(device, True, scale_v, head_dim)
            if settings is not None:
                return sage2_core, settings
    if sage1_width is not None:
        settings = sage1_core.SCALED_V_SETTINGS if scale_v else sage1_core.SETTINGS
        return sage1_core, settings if head_dim == sage1_width else {**settings, "padded_dim": sage1_width}
    return None


def sage_staged(device, head_dim, force_attention=None, masked=False, *, dtype=None, tokens=0, batch=1,
                sol3_prefix_tokens=0, sol3_suffix_tokens=0, sol_tau=None):
    """Projection stages for the selected attention. Sol needs the real geometry to choose its packed
    kernels or its dense short-sequence path. Sparge takes a per-group block map in HND layout."""
    attn = force_attention or offload.shared_state["_attention"]
    scale_v = offload.shared_state.get("_sage_v_scaling", False)
    if masked and attn not in sparge_core.MODES.values():
        return _masked_staged(attn, device, dtype, head_dim, scale_v)
    attn = sol3_mode(attn)
    if attn == "veda":
        attn = offload.shared_state.get("_attention_fallback") or get_default_attention_mode()
    if dtype is not None and not masked and (attn == "sol3" or attn == "sol" and batch == 1 and sol_tau is not None):
        settings = sol_core.staged_settings(device, head_dim, dtype, tokens, prefix=sol3_prefix_tokens, suffix=sol3_suffix_tokens,
                                            classic=attn == "sol", tau=sol_tau if attn == "sol" else sol_core.routing_tau())
        if settings is not None:
            return sol_core, settings
        attn = get_default_attention_mode()
    attn = get_default_attention_mode() if attn in ("sol", "vdn", "radial") else attn
    if attn in sparge_core.MODES.values():
        settings = sparge_core.staged_settings(device, head_dim, attn) if masked else None
        return None if settings is None else (sparge_core, settings)
    if attn == "sage":
        if sage1_core is None or head_dim not in sage1_core.head_dims(device):
            return None
        return sage1_core, sage1_core.SCALED_V_SETTINGS if scale_v else sage1_core.SETTINGS
    if attn != "sage2" or sageattn2 is None:
        return None
    settings = sage2_core.staged_settings(device, masked, scale_v, head_dim)
    return None if settings is None else (sage2_core, settings)


def indexed_window_plans(ring, steps, global_rows, max_global_keys, group_bytes):
    """Optional shared Sage2 plans over an interleaved (row, head, Q/K/V, channel) buffer, one per window step."""
    from shared.kernels import dispatch
    if ring.ndim != 4 or ring.shape[2] != 3 or not dispatch.cuda_inference(ring, dtypes=(torch.float16, torch.bfloat16), approximate=True):
        return None
    staged = sage_staged(ring.device, ring.shape[-1])
    prepare = getattr(staged[0], "indexed_window_plans", None) if staged is not None else None
    if prepare is not None:
        return prepare(staged[1], ring, steps, global_rows, max_global_keys, group_bytes)
    return None

__all__ = [
    'ATTENTION_MODE_AVAILABILITY',
    'attention_config_shared_state',
    'attention_shared_state',
    'get_current_cuda_architecture',
    'get_override_attention_modes',
    'get_supported_override_attention_modes',
    'resolve_attention_mode',
    'pay_attention',
    'sage_staged',
    'indexed_window_plans',
    'sage_attention_mask',
    'attention',
]

_WAVE_KEYS = 8192  # keys of the probe's launches: long enough for the attention kernel to outweigh what surrounds it


def launch_waves(device, head_dim, force_attention=None, masked=False):
    """shared.kernels.launch_waves.LaunchWaves of the attention kernel that the staged functions (sage_staged) or pay_attention would
    run on this device, with this force_attention and head_dim (a mask when masked): the query rows of one head that one of its
    programs covers (.rows), how many programs the device runs at once and what a part-filled last wave costs. Probed once per
    kernel and GPU (a second or two, kept under ~/.triton/autotune) by launching the attention over scratch tensors: the rows per
    program come from the staged core where the kernel's source gives them (its program_rows), else from the probe. None for the
    block-sparse attentions, and where the probe cannot run now or found no waves."""
    from shared.kernels import launch_waves as waves
    attn = force_attention or offload.shared_state["_attention"]
    if attn in sparge_core.MODES.values() or not masked and sol3_mode(attn) in ("sol", "sol3", "chipmunk", "veda", "vsa"):
        return None
    staged = sage_staged(device, head_dim, force_attention, masked)
    rows = None

    def scratch(length, width):  # only while probing (no generator may be made during CUDA graph recording); no FP32 copy of the whole: a block repeated
        generator = torch.Generator(device=device).manual_seed(length)
        return torch.randn((1, 1024, 1, width), generator=generator, device=device).to(torch.bfloat16).repeat(1, -(-length // 1024), 1, 1)[:, :length]

    if staged is not None:
        core, settings = staged
        width = settings.get("padded_dim") or head_dim
        rows = core.program_rows(settings, head_dim) if hasattr(core, "program_rows") else None
        identity = f"{core.__name__.rsplit('.', 1)[-1]} {' '.join(f'{key}={settings[key]}' for key in sorted(settings) if key != 'padded_dim')} d{width}"

        def prepare(most):
            q = scratch(most, width)
            mask = torch.ones((1, 1, 1, _WAVE_KEYS), device=device, dtype=torch.bool) if masked else None
            staged_k, staged_v = core.staged_quantize_k([scratch(_WAVE_KEYS, width)], settings), core.staged_quantize_v([scratch(_WAVE_KEYS, width)], settings)

            def stage(length):
                query = q[:, :length]
                staged_q = core.staged_quantize_q(query, settings)
                return lambda: core.staged_attention(query, [staged_q, staged_k, staged_v], settings, mask)
            return stage
    else:
        attn = get_default_attention_mode() if attn in ("auto", "sol", "vdn", "radial") else attn
        width, identity = head_dim, f"{attn} d{head_dim}"

        def prepare(most):
            q, k, v = scratch(most, width), scratch(_WAVE_KEYS, width), scratch(_WAVE_KEYS, width)
            mask = torch.ones((1, 1, 1, _WAVE_KEYS), device=device, dtype=torch.bool) if masked else None
            return lambda length: lambda: pay_attention([q[:, :length], k, v], force_attention=attn, attention_mask=mask)

    if masked:
        identity += " masked"
    model = waves.get(identity, device, f"Attention ({identity})", prepare, rows)
    if model is not None:
        model.keys = _WAVE_KEYS  # the probe's programs attend this many keys: a wave of a launch over other keys lasts in proportion
    return model


def get_cu_seqlens(batch_size, lens, max_len):
    # MPS compatibility: use dynamic device detection
    _cu_device = "mps" if _is_mps else "cuda"
    cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device=_cu_device)

    for i in range(batch_size):
        s = lens[i] 
        s1 = i * max_len + s
        s2 = (i + 1) * max_len
        cu_seqlens[2 * i + 1] = s1
        cu_seqlens[2 * i + 2] = s2

    return cu_seqlens

@torch.compiler.disable()
def _sage_rows_head_dim(q, softmax_scale):
    # the SageAttention (1) rows path (sage1_core.attention) for q's head_dim: those its kernels run, with the default softmax scale (as
    # before: with another scale they keep sageattn_varlen), and the others it pads with zeros (padded_head_dim), with any scale
    head_dim = q.shape[-1]
    return sage1_core.padded_head_dim(head_dim, q.device) is not None and (softmax_scale is None or head_dim not in sage1_core.head_dims(q.device))


def _constrained_attention(qkv, mask, causal, window, q_lens, k_lens, scale, dropout, mode, deterministic=False, *, scale_v=None):
    """Preserve WanGP's NHD masks and the end-aligned windows of variable-length attention."""
    q, k, v = (x.transpose(1, 2) for x in qkv)
    qkv.clear()
    batch, heads, nq, _ = q.shape
    nk, value_dim = k.size(2), v.size(-1)
    out_dtype = q.dtype
    q, k = q.to(v.dtype), k.to(v.dtype)
    if mask is not None:
        mask = sage_attention_mask(mask)
        if mask.is_floating_point() and mask.dtype not in (q.dtype, torch.float32):
            mask = mask.to(torch.float32 if q.dtype != torch.float64 else q.dtype)
    staged = None
    simple = q_lens is None and k_lens is None and window == (-1, -1)
    if ((mask is not None or causal or not simple) and not dropout and not deterministic and nq and nk and k.shape == v.shape
            and k.device == v.device == q.device and k.size(0) in (1, batch) and q.size(-1) == k.size(-1) and heads % k.size(1) == 0
            and (mask is None or mask.dtype == torch.bool or mask.is_floating_point())
            and not (torch.is_grad_enabled() and any(x is not None and x.requires_grad for x in (q, k, v, mask)))):
        if scale_v is None:
            scale_v = offload.shared_state.get("_sage_v_scaling", False)
        staged = _masked_staged(mode, q.device, q.dtype, q.size(-1), scale_v)
    if staged is not None:
        core, settings = staged
        # Staged K smoothing owns its input. These keys may be shared with another call or V.
        key = k.transpose(1, 2).expand(batch, -1, -1, -1).clone(memory_format=torch.contiguous_format)
        value = v.transpose(1, 2).expand(batch, -1, -1, -1).contiguous()
        packed_k = core.staged_quantize_k([key], settings)
        packed_v = core.staged_quantize_v([value], settings)
        del k, v, key, value

    def dense(query, mask=None, causal=False):
        if staged is not None:
            rows = query.transpose(1, 2).contiguous()
            packed_q = core.staged_quantize_q(rows, settings)
            if scale is not None:
                packed_q[1].mul_(scale * rows.size(-1) ** 0.5)
            return core.staged_attention(torch.empty_like(rows), [packed_q, packed_k, packed_v], settings, mask).transpose(1, 2)
        if mask is None and not causal and mode != "sdpa" and not deterministic:
            return sparse_kernels.dense(query, k, v, causal=causal, scale=scale, dropout=dropout)
        with sdpa_kernel(SDPBackend.MATH) if _sdpa_requires_math(query, mask) else nullcontext():
            return F.scaled_dot_product_attention(query, k, v, attn_mask=mask, is_causal=causal, scale=scale,
                                                  dropout_p=dropout, enable_gqa=heads != k.size(1))

    if simple and (not causal or staged is None):
        return dense(q, mask, causal).transpose(1, 2).to(out_dtype)

    qlen = torch.as_tensor(nq if q_lens is None else q_lens, device=q.device).reshape(-1, 1, 1, 1)
    klen = torch.as_tensor(nk if k_lens is None else k_lens, device=q.device).reshape(-1, 1, 1, 1)
    cols = torch.arange(nk, device=q.device)
    rows = torch.arange(nq, device=q.device)
    out = q.new_empty((batch, heads, nq, value_dim))
    for start in range(0, nq, 256):
        section = slice(start, start + 256)
        row = rows[section].view(1, 1, -1, 1)
        keep = (row < qlen) & (cols < klen)
        center = row if simple else row + klen - qlen
        if window[0] >= 0:
            keep = keep & (cols >= center - window[0])
        if window[1] >= 0:
            keep = keep & (cols <= center + window[1])
        if causal:
            keep = keep & (cols <= center)
        if mask is not None:
            part = mask if mask.ndim < 2 or mask.size(-2) == 1 else mask[..., section, :]
            keep = (keep & part) if part.dtype == torch.bool else torch.where(keep, part, -torch.inf)
        out[:, :, section] = dense(q[:, :, section], keep)
    return out.transpose(1, 2).to(out_dtype)


def pay_attention(
    qkv_list,
    dropout_p=0.,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    version=None,
    force_attention= None,
    attention_mask = None,
    recycle_q = False,
    q_lens = None,
    k_lens = None,
    sol3_prefix_tokens=0,
    sol3_suffix_tokens=0,
    sol_tau=None,
):
    global _HEAD_DIM_SDPA_WARNED
    # format : torch.Size([batches, tokens, heads, head_features])
    # assume if q_lens is non null, each q is padded up to lq (one q out of two will need to be discarded or ignored)
    # assume if k_lens is non null, each k is padded up to lk (one k out of two will need to be discarded or ignored)
    if attention_mask is not None and causal:
        raise ValueError("pay_attention received both attention_mask and causal=True; build a combined mask once and pass causal=False.")
    mode = force_attention or offload.shared_state["_attention"]
    if mode in sparge_core.MODES.values():
        if causal or window_size != (-1, -1) or dropout_p or q_lens is not None or k_lens is not None or softmax_scale is not None:
            if attention_mask is not None:
                raise ValueError("An explicit Sparge block map requires unwindowed, bidirectional attention at its default scale.")
            return _constrained_attention(qkv_list, None, causal, window_size, q_lens, k_lens, softmax_scale, dropout_p, mode, deterministic)
        hnd = [t.transpose(1, 2) for t in qkv_list]
        qkv_list.clear()
        return sparge_core.attention(hnd, attention_mask, mode=mode, recycle_q=recycle_q).transpose(1, 2)
    if (attention_mask is not None or window_size != (-1, -1) and mode != "flash"
            or causal and mode not in ("sage", "sage2", "sdpa", "flash")):
        return _constrained_attention(qkv_list, attention_mask, causal, window_size, q_lens, k_lens, softmax_scale, dropout_p, mode, deterministic)
    if mode == "sol3" or mode == "sol" and sol_tau is not None:
        q,k,v = qkv_list
        if ((mode == "sol" and q.size(0) == 1 or mode == "sol3" and sparsegp_backend and sol3_mode(mode) == mode) and not causal and not dropout_p and
                window_size == (-1,-1) and q_lens is None and k_lens is None and q.ndim == k.ndim == 4
                and k.shape == v.shape and q.shape[:2] == k.shape[:2] and q.size(-1) == k.size(-1) and q.device == k.device == v.device
                and q.size(2) > 0 and k.size(2) > 0 and q.size(2) % k.size(2) == 0 and (mode == "sol3" or q.size(2) == k.size(2)) and q.size(1) >= 1024 and q.size(-1) in (64,128) and q.is_cuda and
                q.dtype == k.dtype == v.dtype and q.dtype in (torch.float16,torch.bfloat16) and not deterministic):
            settings = sol_core.staged_settings(q.device, q.size(-1), q.dtype, q.size(1), scale=softmax_scale,
                                                prefix=sol3_prefix_tokens, suffix=sol3_suffix_tokens,
                                                classic=mode == "sol", tau=sol_tau if mode == "sol" else sol_core.routing_tau())
            if settings is not None:
                del q, k, v
                return sol_core.attention(qkv_list, settings, recycle_q=recycle_q)
        if (causal or dropout_p or softmax_scale is not None or deterministic or
                window_size != (-1,-1) or q_lens is not None or k_lens is not None or q.size(2)!=k.size(2)):
            return _constrained_attention(qkv_list, None, causal, window_size, q_lens, k_lens, softmax_scale, dropout_p, mode, deterministic)
        del q,k,v
        force_attention = get_default_attention_mode()
    attn = offload.shared_state["_attention"] if force_attention== None else force_attention
    if attn in ("veda", "vsa"):
        attn = offload.shared_state.get("_attention_fallback") or get_default_attention_mode()
    attn = get_default_attention_mode() if attn in ("sol", "vdn") else attn

    q,k,v = qkv_list
    qkv_list.clear()
    out_dtype = q.dtype
    # SageAttention on every batch item with all its tokens: its kernels without its device sync (CUDA graphs can record them)
    sage_rows = attn == "sage" and q_lens == None and k_lens == None and not causal and _sage_rows_head_dim(q, softmax_scale)
    if q.dtype == torch.bfloat16 and not bfloat16_supported and not sage_rows:  # sage1_core converts what its kernels need
        q = q.to(torch.float16)
        k = k.to(torch.float16)
        v = v.to(torch.float16)
    final_padding = 0
    b, lq, lk = q.size(0), q.size(1), k.size(1)

    q = q.to(v.dtype)
    k = k.to(v.dtype)
    batch = len(q)
    if len(k) != batch: k = k.expand(batch, -1, -1, -1)
    if len(v) != batch: v = v.expand(batch, -1, -1, -1)
    if q.device.type == "mps": q, k, v = q.contiguous(), k.contiguous(),v.contiguous()
    if attn == "chipmunk":
        from src.chipmunk.modules import SparseDiffMlp, SparseDiffAttn
        from src.chipmunk.util import LayerCounter, GLOBAL_CONFIG
    dense_call = (attention_mask is None and not causal and not dropout_p and softmax_scale is None
                  and window_size == (-1, -1) and q_lens is None and k_lens is None)
    if (attn == "sage2" and dense_call and sageattn2 is not None and k.shape == v.shape
            and q.device == k.device == v.device and q.shape[-1] == k.shape[-1]):
        core = None
        if _fp4_scores is not None and _fp4_scores.eligible(q.device, q.dtype, batch, lq, lk, q.size(2), k.size(2), q.size(3)):
            core = _fp4_scores
        elif _routed is not None and _routed.eligible(q.device, q.dtype, batch, lq, lk, q.size(2), k.size(2), q.size(3)):
            core = _routed
        elif (_wide is not None and q.shape[-1] == 256 and q.shape == k.shape and all(x.stride(-1) == 1 for x in (q, k, v))
              and _wide.preferred(q.device, q.dtype, batch, lq, lk, q.size(2), k.size(2), q.size(3))):
            core = _wide
        if core is not None:
            projections = [q, k, v]
            del q, k, v
            return core.attention_owned(projections, recycle_q=recycle_q).to(out_dtype)
    if attn == "radial": attn = get_default_attention_mode()
    # a block recorded for a CUDA graph: Sage 3 kernels, which are not known to run on PyTorch's current stream, give way to Sage 2
    if attn == "sage3" and sageattn2 is not None and cuda_graphs.recording(): attn = "sage2"
    attn = sage_gp_redirect(attn)[0]  # a mode set without the generation's redirect (its quality then stays the configured one)

    if b > 1 and k_lens != None and attn in ("sage2", "sage3", "sdpa"):
        assert attention_mask == None
        # Poor's man var k len attention
        assert q_lens == None
        chunk_sizes = []
        k_sizes = []
        current_size = k_lens[0]
        current_count= 1
        for k_len in k_lens[1:]:
            if k_len == current_size:
                current_count += 1
            else:
                chunk_sizes.append(current_count)
                k_sizes.append(current_size)
                current_count = 1
                current_size = k_len
        chunk_sizes.append(current_count)
        k_sizes.append(k_len)
        if len(chunk_sizes) > 1 or k_lens[0] != k.shape[1]:
            q_chunks =torch.split(q, chunk_sizes)
            k_chunks =torch.split(k, chunk_sizes)
            v_chunks =torch.split(v, chunk_sizes)
            q, k, v = None, None, None
            k_chunks = [ u[:, :sz] for u, sz in zip(k_chunks, k_sizes)]
            v_chunks = [ u[:, :sz] for u, sz in zip(v_chunks, k_sizes)]
            o = []
            for sub_q, sub_k, sub_v in zip(q_chunks, k_chunks, v_chunks): 
                qkv_list = [sub_q, sub_k, sub_v]
                sub_q, sub_k, sub_v = None, None, None
                o.append(pay_attention(qkv_list, dropout_p=dropout_p, softmax_scale=softmax_scale, causal=causal, force_attention=attn))
            q_chunks, k_chunks, v_chunks = None, None, None
            o = torch.cat(o, dim = 0)
            return o
    elif (q_lens != None or k_lens != None) and attn in ("sage2", "sage3", "sdpa"):
        assert b == 1
        szq = q_lens[0].item() if q_lens != None else lq
        szk = k_lens[0].item() if k_lens != None else lk
        final_padding = lq - szq
        q = q[:, :szq]
        k = k[:, :szk]
        v = v[:, :szk]

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    if attn == "sage" and not sage_rows and q.shape[-1] not in (64, 128):  # sageattn_varlen runs 64 and 128 only (sage_rows pads the others)
        if not _HEAD_DIM_SDPA_WARNED:
            tqdm.write(f"[WAN2GP] SageAttention does not support this attention (head_dim {q.shape[-1]}): it will use SDPA.")
            _HEAD_DIM_SDPA_WARNED = True
        attn = "sdpa"
    if attn=="sage" and not sage_rows or attn=="flash":
        if b != 1 :
            if k_lens == None:
                k_lens = torch.full((b,), lk, dtype=torch.int32, device=q.device)  # made on the GPU: no host copy (not allowed in CUDA graphs)
            if q_lens == None:
                q_lens = torch.full((b,), lq, dtype=torch.int32, device=q.device)
            k = k.reshape(-1, *k.shape[-2:])
            v = v.reshape(-1, *v.shape[-2:])
            q = q.reshape(-1, *q.shape[-2:])
            cu_seqlens_q=get_cu_seqlens(b, q_lens, lq) 
            cu_seqlens_k=get_cu_seqlens(b, k_lens, lk) 
        else:
            szq = q_lens[0].item() if q_lens != None else lq
            szk = k_lens[0].item() if k_lens != None else lk
            if szq != lq or szk != lk:
                cu_seqlens_q = torch.tensor([0, szq, lq], dtype=torch.int32, device=q.device)
                cu_seqlens_k = torch.tensor([0, szk, lk], dtype=torch.int32, device=q.device)
            else:
                cu_seqlens_q = torch.arange(0, 2 * lq, lq, dtype=torch.int32, device=q.device)  # [0, lq] made on the GPU: no host copy (not allowed in CUDA graphs)
                cu_seqlens_k = torch.arange(0, 2 * lk, lk, dtype=torch.int32, device=q.device)
            q = q.squeeze(0)
            k = k.squeeze(0)
            v = v.squeeze(0)


    # apply attention
    if sage_rows:
        qkv_list = [t.reshape(-1, *t.shape[2:]).view(t.shape) for t in (q, k, v)]  # batch items one after another (copies only otherwise)
        del q, k, v
        x = sage1_core.attention(qkv_list, recycle_q, offload.shared_state.get("_sage_v_scaling", False),
                                 None if attention_mask is None else sage_attention_mask(attention_mask, out_dtype), softmax_scale)
    elif attn=="sage":
        x = sageattn_varlen_wrapper(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q= cu_seqlens_q,
            cu_seqlens_kv= cu_seqlens_k,
            max_seqlen_q=lq,
            max_seqlen_kv=lk,
            is_causal=causal,
            sm_scale=softmax_scale,
            scale_v=offload.shared_state.get("_sage_v_scaling", False),
        ).unflatten(0, (b, lq))
    elif attn=="sage3":
        qkv_list = [q,k,v]
        del q,k,v
        x = sageattn3_wrapper(qkv_list, lq)
    elif attn=="sage2":
        qkv_list = [q,k,v]
        del q,k,v
        x = sageattn2_wrapper(qkv_list, lq, recycle_q=recycle_q, attention_mask=attention_mask, causal=causal, scale_v=offload.shared_state.get("_sage_v_scaling", False))
    elif attn=="sdpa":
        qkv_list = [q, k, v]
        del q ,k ,v
        x = sdpa_wrapper(qkv_list, lq, attention_mask=attention_mask, causal=causal, softmax_scale=softmax_scale, dropout_p=dropout_p)
    elif attn=="flash" and version == 3:
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q= cu_seqlens_q,
            cu_seqlens_k= cu_seqlens_k,
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0].unflatten(0, (b, lq))
    elif attn=="flash":
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q= cu_seqlens_q,
            cu_seqlens_k= cu_seqlens_k,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic).unflatten(0, (b, lq))

    # output

    elif attn=="xformers":
        from xformers.ops.fmha.attn_bias import BlockDiagonalPaddedKeysMask
        if k_lens == None and q_lens == None:
            x = memory_efficient_attention(q, k, v )
        elif k_lens != None and q_lens == None:
            attn_mask = BlockDiagonalPaddedKeysMask.from_seqlens([lq] * b , lk , list(k_lens) ) 
            x = memory_efficient_attention(q, k, v, attn_bias= attn_mask )
        elif b == 1:
            szq = q_lens[0].item() if q_lens != None else lq
            szk = k_lens[0].item() if k_lens != None else lk
            attn_mask = BlockDiagonalPaddedKeysMask.from_seqlens([szq, lq - szq ] , lk , [szk, 0] ) 
            x = memory_efficient_attention(q, k, v, attn_bias= attn_mask )
        else:
            assert False
    x = x.type(out_dtype)
    if final_padding > 0:
        x = torch.cat([x, torch.empty( (x.shape[0], final_padding, *x.shape[-2:]), dtype= x.dtype, device=x.device  ) ], 1)


    return x 
