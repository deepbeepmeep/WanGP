# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Optional INT8 backends shared by Quanto and ConvRot linear layers."""
from __future__ import annotations

import importlib
import os
from fnmatch import fnmatchcase
from types import MethodType

import torch

from shared.kernels import kernel_policy, quanto_int8_inject as triton


def _wgp_installed():
    """The WGPKernels package with its built INT8 library, checked without the GPU (the backend probe runs the kernels)."""
    try:
        from wgpkernels import int8
    except ImportError:
        return False
    return int8.default_library_path().is_file()


CHOICES = [("Disabled: PyTorch (slowest, most precise)", "disabled"), ("Auto (best available)", "auto"),
           (f"WGPKernels (recommended, {'installed' if _wgp_installed() else 'not installed'}, second fastest, precise)", "wgpkernels"),
           ("Triton (fast, less precise)", "triton"), ("Comfy Kitchen Kernels (fastest, least precise)", "kitchen")]
_LABELS = {"wgpkernels": "WGPKernels", "kitchen": "Comfy Kitchen", "triton": "Triton"}
_backend = "pytorch"
revision = 0
_wgp = None  # wgpkernels.int8 while it is the backend
_kitchen = None
_kitchen_hip = False
_direct_cutlass = False
_sm120_cutlass = False
_original_forward = None
_ops_registered = False
_fusion_logged = False
_compile_cache_root = None
_compile_cache_backend = None
_wide_convrot_triton = None
_blockwise_triton = None
_blockwise_required = False
_kitchen_dlpack_export = None  # (Kitchen module, its own DLPack exporter) while the direct exporter replaces it
_prepared_capture_streams = set()
# Bump when changes to INT8 custom operators invalidate compiled graphs.
_COMPILE_CACHE_VERSION = 4
# Includes row quantization, a possible INT32 GEMM result, and a temporary output.
# Never allocate an activation-sized quantization buffer for a whole video.
_SCRATCH_BYTES = 16 * 1024 * 1024
_WGP_BLOCK_ROWS = 65535  # rows of one WGPKernels per-block activation quantization call


def prepare_compile_cache(enabled):
    """Isolate compiled graphs by resolved backend; leave eager runs untouched."""
    global _compile_cache_root, _compile_cache_backend
    if not enabled or _compile_cache_backend == _backend:
        return
    from torch._inductor.runtime.runtime_utils import cache_dir

    if _compile_cache_root is None:
        _compile_cache_root = cache_dir()
    path = os.path.join(_compile_cache_root, f"wangp_int8_v{_COMPILE_CACHE_VERSION}", _backend)
    os.makedirs(path, exist_ok=True)
    # Reset in-memory graphs as well when switching backends in the same process.
    torch.compiler.reset()
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = path
    _compile_cache_backend = _backend
    print(f"[INT8] Compile Cache: {_backend} (v{_COMPILE_CACHE_VERSION}).")


def _export_dlpack(tensor):
    # Kitchen's DLPack export without the per-call checks of Tensor.__dlpack__, which always hold for the inference tensors it gets
    # (strided, on the current device, no stream synchronization): an INT8 linear exports a dozen tensors, so they add up on fast steps
    return torch._C._to_dlpack(tensor.detach() if tensor.requires_grad else tensor)


def _probe_kitchen():
    if not torch.cuda.is_available():
        return None, "requires a CUDA or ROCm GPU"
    hip = torch.version.hip is not None
    if not hip and torch.cuda.get_device_capability() < (7, 5):
        return None, "requires an NVIDIA GPU with compute capability 7.5 or newer"
    try:
        from importlib.metadata import version
        from packaging.version import Version

        if Version(version("comfy-kitchen")) < Version("0.2.35"):
            return None, "requires comfy-kitchen >= 0.2.35"
        module = importlib.import_module("comfy_kitchen.backends.hip" if hip else "comfy_kitchen.backends.cuda")
        if hip and not module.has_wmma():
            return None, "Kitchen HIP INT8 requires a working HIP extension and supported RDNA3/3.5/4 GPU"
        # Exercise both decode and GEMM at selection time, outside graph capture.
        with torch.inference_mode():
            device = torch.device("cuda", torch.cuda.current_device())
            w = torch.zeros((256, 256), dtype=torch.int8, device=device)
            s = torch.ones(256, dtype=torch.float32, device=device)
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                for rows in (1, 32):
                    x = torch.zeros((rows, 256), dtype=dtype, device=device)
                    for convrot in (False, True):
                        out = module.int8_linear(x, w, s, out_dtype=dtype, convrot=convrot)
                        if out.shape != x.shape or out.dtype != dtype or not torch.isfinite(out).all():
                            return None, "Kitchen INT8 compatibility probe failed"
            torch.cuda.synchronize(device)
        return module, "available"
    except Exception as exc:
        return None, str(exc)


def _probe_wgpkernels():
    if not torch.cuda.is_available():
        return None, "requires a GPU"
    if torch.version.hip is not None and not torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx1201"):
        return None, "requires an AMD RDNA4 GPU (gfx1201) with ROCm"
    # Turing (7.5) runs builds that include sm_75 code; on older builds the selftest below fails (no kernel image) and Auto moves on
    if torch.version.hip is None and torch.cuda.get_device_capability() < (7, 5):
        return None, "requires an NVIDIA GPU with compute capability 7.5 or newer"
    try:
        module = importlib.import_module("wgpkernels.int8")
        if module._lib is None:
            module.load_library()
        # Every kernel once with fixed configurations, at selection time and outside graph capture.
        with torch.inference_mode():
            failure = module.selftest()
        return (None, failure) if failure else (module, "available")
    except Exception as exc:
        return None, str(exc)


def resolve_backend(selection):
    """Validate user selection before changing the active runtime or config. The selected backend, else the Auto choice: WGPKernels,
    then Comfy Kitchen, then Triton, then PyTorch (kernel_policy.RANKING)."""
    if selection not in {value for _, value in CHOICES}:
        raise ValueError(f"Unknown INT8 kernel backend: {selection!r}")
    if selection == "disabled":
        return "pytorch", None
    probes = {"wgpkernels": _probe_wgpkernels, "kitchen": _probe_kitchen, "triton": triton._probe_triton_backend}
    if selection != "auto":
        module, reason = probes[selection]()
        if module is not None:
            return selection, module
        print(f"[INT8] {_LABELS[selection]} unavailable: {reason}. Using the Auto choice.")
    for backend in kernel_policy.RANKING[:3]:
        if backend != selection:
            module, reason = probes[backend]()
            if module is not None:
                return backend, module
    return "pytorch", None


def kitchen_enabled():
    return _backend == "kitchen"


def wgp_enabled():
    return _backend == "wgpkernels"


def _wgp_supports(input, weight):
    """Inputs of WGPKernels: BF16/FP16 activations on the GPU (plain, or traced by torch.compile, not Quanto activations), INT8
    weights with K a multiple of 16."""
    return ((type(input) is torch.Tensor or (triton._is_fake_tensor(input) and not triton._is_qbytes_tensor(input)))
            and input.is_cuda and input.dtype in (torch.bfloat16, torch.float16)
            and weight._data.is_cuda and weight._data.dtype == torch.int8 and input.shape[-1] % 16 == 0)


def _wgp_aligned(x):
    # the kernels load activation rows by 16 bytes
    return x.clone(memory_format=torch.contiguous_format) if x.stride(-1) != 1 or x.stride(0) % 8 or x.data_ptr() % 16 else x


def _wgp_run(x, weight, scale, bias, group, precise):
    return _wgp.linear(_wgp_aligned(x), weight, scale, bias, group, precise)


def wgp_linear(input, weight, bias=None, group=0, precise=False, act=None, act_weight=None, act_eps=0.0, residual=None, residual_scale=None,
               out=None):
    """INT8 linear with WGPKernels (ConvRot rotation `group`, 0 for plain INT8). The kernels choose, per GPU and shape, full
    precision activations or per-block INT8 activations; precise=True keeps full precision at every token count.
    Fusions (eager inference, see can_fuse_linear): act 'rms_norm' (act_weight, act_eps) or 'swiglu' (input rows: gate | value) applied in
    FP32 to the input, and residual + residual_scale * linear rounded once, written to out (which may be the residual)."""
    n = weight._data.shape[0]
    scale = _weight_scale(weight, n, input.device)
    x = input.reshape(-1, input.shape[-1])
    if bias is not None and bias.dtype != x.dtype:
        bias = bias.to(x.dtype)
    if torch.compiler.is_compiling() or triton._is_fake_tensor(input):
        return torch.ops.wan2gp_wgp.linear(x, weight._data, scale, bias, group, precise).reshape(*input.shape[:-1], n)
    if act is None and residual is None and out is None:
        return _wgp_run(x, weight._data, scale, bias, group, precise).reshape(*input.shape[:-1], n)
    if act_weight is not None and act_weight.dtype != x.dtype:
        act_weight = act_weight.to(x.dtype)
    if residual is not None:
        residual = residual.reshape(-1, n)
        if residual_scale is not None:
            residual_scale = residual_scale.reshape(-1).to(x.dtype)
    y = None
    if out is not None:
        if out.shape != (*input.shape[:-1], n) or out.dtype != x.dtype or not out.is_contiguous() or out.data_ptr() % 16:
            raise ValueError("Fused INT8 output must match the output shape and dtype, be contiguous and 16-byte aligned")
        y = out.view(-1, n)
    y = _wgp.linear(_wgp_aligned(x), weight._data, scale, bias, group, precise, act=act, act_weight=act_weight, eps=act_eps, residual=residual,
                    residual_scale=residual_scale, out=y)
    if out is None:
        return y.reshape(*input.shape[:-1], n)
    torch.autograd.graph.increment_version(out)  # written in place: MMGP CUDA graphs copy a mutated block argument back
    return out


def wgp_group_scales(input, weight):
    """Whether WGPKernels computes an INT8 weight with one FP32 scale per output channel and 128-column block (scale (N, K / 128)):
    the selected backend, with a library taking such scales, on BF16/FP16 CUDA activations (K a multiple of 16)."""
    return (_backend == "wgpkernels" and getattr(_wgp, "weight_groups_supported", lambda: False)() and input.is_cuda and input.dtype in (torch.bfloat16, torch.float16)
            and weight.is_cuda and input.shape[-1] % 16 == 0)


def wgp_group_linear(input, weight, scale, bias=None):
    """input @ (weight (N, K) INT8 * scale (N, K / 128) per 128-column block).T + bias with WGPKernels, which choose the path per GPU and
    shape as for the other INT8 weights."""
    x = input.reshape(-1, input.shape[-1])
    if bias is not None and bias.dtype != x.dtype:
        bias = bias.to(x.dtype)
    if torch.compiler.is_compiling() or triton._is_fake_tensor(input):
        return torch.ops.wan2gp_wgp.linear(x, weight, scale, bias, 0, False).reshape(*input.shape[:-1], weight.shape[0])
    return _wgp_run(x, weight, scale, bias, 0, False).reshape(*input.shape[:-1], weight.shape[0])


def linear_fused(input, weight, bias, group, input_act, act_weight, act_eps, residual, residual_scale, out=None):
    """linear_with_fusion's computation by the active backend (ConvRot layers, see can_fuse_linear)."""
    global _fusion_logged
    if _backend == "wgpkernels":
        if not _fusion_logged:
            print("[INT8] WGPKernels fused normalization/activation and residual linear kernels are being used.")
            _fusion_logged = True
        return wgp_linear(input, weight, bias, group, act=input_act, act_weight=act_weight, act_eps=act_eps, residual=residual,
                          residual_scale=residual_scale, out=out)
    return kitchen_linear_fused(input, weight, bias, input_act, act_weight, act_eps, residual, residual_scale, out)


def linear_prequantized(weight, bias, *prepared):
    """The output rows of a projection whose input linear_multi / linear_grouped prepared once (operands of the active backend)."""
    if _backend == "wgpkernels":
        prepared_input, out = prepared
        scale = _weight_scale(weight, weight._data.shape[0], out.device)
        return _wgp.linear_prepared(prepared_input, weight._data, scale, None if bias is None else bias.to(out.dtype), out=out)
    return kitchen_linear_prequantized(weight, bias, *prepared)


def require_int8_residual(model, reason):
    """Keep activation outliers from erasing small values in rowwise INT8 activations: Kitchen adds a second INT8 product of the
    discarded residual, WGPKernels keep full precision activations at every token count.

    Install before MMGP profiling. Other backends and weight formats retain their own forwards.
    """
    from optimum.quanto import qint8

    count = 0
    for module in model.modules():
        if (isinstance(module, torch.nn.Linear) and getattr(module, "weight_qtype", None) == qint8
                and not hasattr(module, "_int8_residual_forward")):
            module._int8_residual_forward = module.forward
            module.forward = MethodType(_residual_module_forward, module)
            count += 1
    if count:
        print(f"[INT8] Full Precision Activations Required for {count} Linear Layers - {reason}.")


def _residual_module_forward(module, input):
    if _backend == "wgpkernels" and _wgp_supports(input, module.qweight):
        return wgp_linear(input, module.qweight, module.bias, precise=True)
    if (_backend != "kitchen" or type(input) is not torch.Tensor or not input.is_cuda
            or input.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
        return module._int8_residual_forward(input)
    weight = module.qweight
    n = weight._data.shape[0]
    scale = _weight_scale(weight, n, input.device)
    x = input.reshape(-1, input.shape[-1])
    if torch.compiler.is_compiling() or triton._is_fake_tensor(input):
        out = torch.ops.wan2gp_kitchen.residual_linear(x, weight._data, scale, module.bias)
    else:
        out = _residual_linear_impl(x, weight._data, scale, module.bias)
    return out.reshape(*input.shape[:-1], n)


def _residual_linear_impl(x, weight, scale, bias):
    # W8A8's single row scale can destroy the signal of weight-only INT8 LMs.
    # Quantize the discarded residual separately and sum both INT8 products in
    # FP32, rounding only once. Never dequantize weights or retain a full input
    # residual; the bound includes contiguous input copies and cuBLAS INT32 output.
    m, k = x.shape
    n = weight.shape[0]
    rows = max(1, _SCRATCH_BYTES // (10 * k + 12 * n + 64))
    out = x.new_empty((m, n))
    for start in range(0, m, rows):
        stop = min(start + rows, m)
        tile = x[start:stop]
        q, qs = _kitchen.quantize_int8_rowwise(tile)
        residual = q.float().mul_(qs).neg_().add_(tile)
        del q, qs
        high = _kitchen.int8_linear(tile, weight, scale, out_dtype=torch.float32)
        low = _kitchen.int8_linear(residual, weight, scale, out_dtype=torch.float32)
        high.add_(low)
        if bias is not None:
            high.add_(bias)
        out[start:stop].copy_(high)
        del residual, high, low
    return out


def require_blockwise_int8(model, reason, module_patterns=("*",)):
    """Protect a model or subtree whose plain INT8 linears need finer activation scales.

    Install before MMGP profiling/LoRA hooks. Other qtypes and the user's Disabled,
    Triton or WGPKernels selections retain their normal forwards (WGPKernels
    use per-block activation scales at most); Kitchen uses Triton for these linears,
    or native Quanto when Triton is unavailable.
    """
    global _blockwise_required, _blockwise_triton
    from optimum.quanto import qint8

    count = 0
    for name, module in model.named_modules():
        if (isinstance(module, torch.nn.Linear) and getattr(module, "weight_qtype", None) == qint8
                and any(fnmatchcase(name, pattern) for pattern in module_patterns)
                and not hasattr(module, "_int8_precision_forward")):
            module._int8_precision_forward = module.forward
            module.forward = MethodType(_blockwise_module_forward, module)
            count += 1
    if count:
        _blockwise_required = True
        if kitchen_enabled():
            _blockwise_triton, unavailable = triton._probe_triton_backend()
            backend = "Triton" if _blockwise_triton is not None else f"PyTorch ({unavailable})"
            print(f"[INT8] Precision Rule: {backend} for {count} linear layers - {reason}.")


def _blockwise_linear_impl(x, weight, scale, bias):
    if _blockwise_triton is None:
        out = torch.ops.quanto.qbytes_mm(x, weight, scale)
    else:
        out = _blockwise_triton.fused_quant_scaled_mm(x, weight, scale, out_dtype=x.dtype)
    if bias is not None:
        out.add_(bias)
    return out


def _blockwise_module_forward(module, input):
    if (_backend != "kitchen" or type(input) is not torch.Tensor or not input.is_cuda
            or input.dtype not in (torch.float16, torch.bfloat16, torch.float32)):
        return module._int8_precision_forward(input)
    weight = module.qweight
    x = input.reshape(-1, input.shape[-1])
    if torch.compiler.is_compiling() or triton._is_fake_tensor(input):
        out = torch.ops.wan2gp_kitchen.blockwise_linear(x, weight._data, weight._scale, module.bias)
    else:
        out = _blockwise_linear_impl(x, weight._data, weight._scale, module.bias)
    return out.reshape(*input.shape[:-1], weight.shape[0])


def _weight_scale(weight, n, device):
    # float32 scales of an INT8 weight, made once per weight object: converting them at each call launches a kernel, which costs host
    # bound steps (image models run about a thousand INT8 linears per step)
    if torch.compiler.is_compiling():
        return triton._prepare_weight_scale(weight._scale, n, device)
    scale = weight.__dict__.get("_kitchen_scale", None)
    if scale is None:
        from mmgp import allocator
        # A converted weight scale outlives this block. A later block's warmup must not place it in scratch used by an earlier graph.
        with allocator.normal_allocations():
            scale = weight._kitchen_scale = triton._prepare_weight_scale(weight._scale, n, device)
    return scale


def _wide_convrot_linear(input, weight, bias=None):
    """Use Triton for ConvRot rows beyond Kitchen's graph-safe CUTLASS width."""
    from shared.qtypes.int8_convrot import _rotate_activation

    rotated = _rotate_activation(input, 256)
    x = rotated.reshape(-1, rotated.shape[-1])
    scale = _weight_scale(weight, weight.shape[0], input.device)
    out = _wide_convrot_triton.fused_quant_scaled_mm(x, weight._data, scale, out_dtype=input.dtype)
    if bias is not None:
        out += bias
    return out.reshape(*input.shape[:-1], weight.shape[0])


def can_fuse_linear(module, x, dtypes=(torch.bfloat16,), act=None):
    """Keep qtype/backend details out of model code; MMGP still owns module calls.
    Callers widen `dtypes` only for activation dtypes validated with their model; act is the input operation they want fused
    ('rms_norm', 'swiglu', or 'gelu', which only WGPKernels provide).
    WGPKernels fuse with the precision of their unfused linears (full precision or per-block activations, input operations in FP32,
    one rounding), whatever the kernel precision setting; Comfy Kitchen's rowwise fusions need the fast (approximate) setting."""
    if (torch.compiler.is_compiling() or not torch.is_inference_mode_enabled() or triton._is_fake_tensor(x) or not x.is_cuda
            or x.dtype not in dtypes or (getattr(module, '_mm_lora_old_forward', None) and getattr(module, '_mm_lora_data', None) != {})):
        return False
    if _backend == "wgpkernels":
        group = getattr(module, '_convrot_group_size', 0)
        quanto = group == 0 and getattr(getattr(module, 'weight_qtype', None), 'name', None) == "qint8"  # plain INT8 (Quanto files)
        return act in _wgp.ACTS and (group in (16, 64, 256) or quanto) and module.in_features % 16 == 0
    return (act != "gelu" and kernel_policy.allow_approximate() and kitchen_enabled() and _sm120_cutlass
            and getattr(module, '_convrot_group_size', 0) == 256
            and 256 <= module.in_features <= 16384 and module.in_features % 256 == 0
            and module.out_features % 8 == 0
            and _kitchen._convrot_fused_shared_memory_fits(x, module.in_features, 256))


def linear_with_fusion(module, x, *, input_act=None, act_weight=None, act_eps=0.0,
                       residual=None, residual_scale=None, out=None):
    # Quanto's router accepts only x. Keep its normal module/offload hooks while
    # handing the qtype the extra operands for this synchronous call only.
    previous = getattr(module, '_wangp_linear_fusion', None)
    fusion = module._wangp_linear_fusion = (input_act, act_weight, act_eps, residual, residual_scale, out)
    # Quanto's linear function receives the weight tensor, not the module: the operands ride on the weight for that call
    qweight = getattr(module, 'qweight', None) if getattr(getattr(module, 'weight_qtype', None), 'name', None) == "qint8" else None
    if qweight is not None:
        qweight._wangp_linear_fusion = fusion
    try:
        return module(x)
    finally:
        if previous is None:
            del module._wangp_linear_fusion
        else:
            module._wangp_linear_fusion = previous
        if qweight is not None:
            del qweight._wangp_linear_fusion


def _call_prepared(module, x, prepared):
    """module(x) with its input already prepared (and its output tile): the module's own hooks run, MMGP's weight loading included."""
    target = module.qweight if module.weight_qtype.name == "qint8" else module  # Quanto's forward receives the weight, not the module
    previous = getattr(target, '_wangp_prequantized_input', None)
    target._wangp_prequantized_input = prepared
    try:
        module(x)
    finally:
        if previous is None:
            del target._wangp_prequantized_input
        else:
            target._wangp_prequantized_input = previous


def _wgp_path(module, x, group, precise=False):
    """The WGPKernels path for x and this module's weight, used for every projection of an input prepared once."""
    weight = module.qweight
    return _wgp.choose_path(x, weight._data, _weight_scale(weight, weight._data.shape[0], x.device), None, group, precise)


def linear_multi(modules, x):
    """Reuse one bounded INT8 input tile across projections, retaining MMGP hooks."""
    if _backend == "wgpkernels":
        group = getattr(modules[0], '_convrot_group_size', 0)
        if any(getattr(module, '_convrot_group_size', 0) != group for module in modules):
            return [module(x) for module in modules]
        x = _wgp_aligned(x)
        outputs = [x.new_empty((x.shape[0], module.out_features)) for module in modules]
        rows = max(128, _SCRATCH_BYTES // (2 * x.shape[-1] + 4) // 128 * 128)  # exact activations: two INT8 values per input
        for start in range(0, x.shape[0], rows):
            tile = x[start:start + rows]
            prepared = _wgp.prepare(tile, group, _wgp_path(modules[0], tile, group))
            for module, output in zip(modules, outputs):
                _call_prepared(module, tile, (prepared, output[start:start + rows]))
            del prepared
        return outputs
    outputs = [x.new_empty((x.shape[0], module.out_features)) for module in modules]
    rows = max(32, (_SCRATCH_BYTES // (x.shape[-1] + 4)) // 32 * 32)
    for start in range(0, x.shape[0], rows):
        stop = min(start + rows, x.shape[0])
        tile = x[start:stop]
        q, scales = _kitchen.quantize_int8_rowwise_convrot64(tile, 256)
        for module, output in zip(modules, outputs):
            _call_prepared(module, tile, (q, scales, output[start:stop]))
        del q, scales
    return outputs


def kitchen_linear_prequantized(weight, bias, q, scales, out):
    scale = _weight_scale(weight, weight._data.shape[0], out.device)
    bias_arg = (_kitchen._gemm_vector_arg(bias, out.device, out.dtype) if bias is not None
                else _kitchen._empty_cuda_tensor(out.device, out.dtype))
    wrap = _kitchen._wrap_for_dlpack
    used = _kitchen._C.cutlass_int8_dequant(
        wrap(q), wrap(weight._data), wrap(scales), wrap(scale), wrap(bias_arg), wrap(out),
        _kitchen.DTYPE_TO_CODE[out.dtype], torch._C._cuda_getCurrentRawStream(out.device.index))
    if not used:
        raise RuntimeError("Comfy Kitchen rejected the shared-input INT8 output tile")
    return out


def linear_grouped(modules, x):
    """Independent small projections with batched input preparation and normal module hooks.

    Group-64 ConvRot normally launches rotation and quantization for each module.
    Kitchen retains its BF16 rotation boundary but prepares all input rows together;
    WGPKernels prepare all rows at once for their path (rows are independent). No
    weights are packed or cached, so MMGP retains their lifetime and residency.
    """
    group = getattr(modules[0], '_convrot_group_size', 0)
    common = (torch.is_inference_mode_enabled() and not torch.compiler.is_compiling() and not triton._is_fake_tensor(x) and x.is_cuda
              and x.ndim == 3 and x.is_contiguous() and len(modules) == x.shape[0]
              and all(getattr(module, '_convrot_group_size', 0) == group
                      and getattr(getattr(module, 'weight_qtype', None), 'name', None) == 'qint8_convrot'
                      and module.in_features == x.shape[2] and module.out_features == modules[0].out_features
                      and not getattr(module, '_wangp_linear_fusion', None)
                      and not getattr(module, '_wangp_prequantized_input', None)
                      and (not getattr(module, '_mm_lora_old_forward', None)
                           or getattr(module, '_mm_lora_data', None) == {}) for module in modules))
    if (_backend == "wgpkernels" and common and group in (16, 64, 256) and x.dtype in (torch.bfloat16, torch.float16) and x.shape[2] % 16 == 0
            and x.numel() * 2 <= _SCRATCH_BYTES and x.data_ptr() % 16 == 0):
        tokens = x.shape[1]
        prepared = _wgp.prepare(x.reshape(-1, x.shape[2]), group, _wgp_path(modules[0], x[0], group))
        output = x.new_empty((x.shape[0], tokens, modules[0].out_features))
        for index, (module, part, out) in enumerate(zip(modules, x.unbind(), output.unbind())):
            _call_prepared(module, part, (prepared.rows(index * tokens, (index + 1) * tokens), out))
        return output
    supported = (common and group == 64 and kernel_policy.allow_approximate() and kitchen_enabled() and _direct_cutlass
                 and x.dtype == torch.bfloat16 and x.shape[1] > 2 and x.shape[2] % 64 == 0
                 and x.numel() * 4 <= _SCRATCH_BYTES and modules[0].out_features % 8 == 0)
    if not supported:
        return torch.stack([module(part) for module, part in zip(modules, x.unbind())])
    from shared.qtypes.int8_convrot import _rotate_activation
    rotated = _rotate_activation(x, 64).reshape(-1, x.shape[-1])
    q, scales = _kitchen.quantize_int8_rowwise(rotated)
    del rotated
    q, scales = q.reshape_as(x), scales.reshape(x.shape[0], x.shape[1], -1)
    output = x.new_empty((x.shape[0], x.shape[1], modules[0].out_features))
    for module, part, quantized, scale, out in zip(modules, x.unbind(), q.unbind(), scales.unbind(), output.unbind()):
        _call_prepared(module, part, (quantized, scale, out))
    return output


def kitchen_linear_fused(input, weight, bias, input_act, act_weight, act_eps, residual, residual_scale, out=None):
    """Bound temporary quantization/output storage while retaining Kitchen fusions."""
    global _fusion_logged
    if not _fusion_logged:
        print("[INT8] Comfy Kitchen fused normalization/activation and residual linear kernels are being used.")
        _fusion_logged = True
    scale = _weight_scale(weight, weight._data.shape[0], input.device)
    x = input.reshape(-1, input.shape[-1])
    n, k = weight._data.shape
    if out is not None and (out.shape != (*input.shape[:-1], n) or out.dtype != input.dtype
                            or out.device != input.device or not out.is_contiguous()):
        raise ValueError("Fused INT8 output must match the output shape, dtype and device and be contiguous")
    supplied, out = out, input.new_empty((x.shape[0], n)) if out is None else out.reshape(-1, n)
    r = residual.reshape(-1, n) if residual is not None else None
    # The raw ConvRot quantizer requires packed rows, including fused SwiGLU input.
    row_bytes = k + 4 + (x.shape[-1] * x.element_size() if not x.is_contiguous() else 0)
    rows = max(32, (_SCRATCH_BYTES // row_bytes) // 32 * 32)
    wrap = _kitchen._wrap_for_dlpack
    stream = torch._C._cuda_getCurrentRawStream(input.device.index)
    bias_arg = (_kitchen._gemm_vector_arg(bias, input.device, input.dtype) if bias is not None
                else _kitchen._empty_cuda_tensor(input.device, input.dtype))
    act_arg = _kitchen._act_weight_arg(input_act, act_weight, input.device, input.dtype)
    residual_arg = (_kitchen._gemm_vector_arg(residual_scale, input.device, input.dtype)
                    if r is not None else None)
    for start in range(0, x.shape[0], rows):
        stop = min(start + rows, x.shape[0])
        tile = x[start:stop].contiguous()
        q = torch.empty((stop-start, k), device=input.device, dtype=torch.int8)
        qs = torch.empty((stop-start, 1), device=input.device, dtype=torch.float32)
        _kitchen._C.quantize_int8_rowwise_convrot64(
            wrap(tile), wrap(q), wrap(qs), 256, False,
            _kitchen._input_act_code(input_act), 0, wrap(act_arg), float(act_eps), stream)
        if r is not None:
            used = _kitchen._C.cutlass_int8_dequant_residual(
                wrap(q), wrap(weight._data), wrap(qs), wrap(scale), wrap(bias_arg),
                wrap(residual_arg), wrap(r[start:stop]), wrap(out[start:stop]),
                _kitchen.DTYPE_TO_CODE[input.dtype], stream)
        else:
            used = _kitchen._C.cutlass_int8_dequant(
                wrap(q), wrap(weight._data), wrap(qs), wrap(scale), wrap(bias_arg),
                wrap(out[start:stop]), _kitchen.DTYPE_TO_CODE[input.dtype], stream)
        if not used:
            raise RuntimeError("Comfy Kitchen rejected the fused SM120 INT8 output tile")
        del q, qs, tile
    if supplied is not None:
        torch.autograd.graph.increment_version(supplied)  # written in place: MMGP CUDA graphs copy a mutated block argument back
    return out.reshape(*input.shape[:-1], n)


def _linear_impl(x, weight, scale, bias, convrot):
    m, k = x.shape
    n = weight.shape[0]
    # Bound scratch even on the package's cuBLAS path, which has an INT32 output.
    row_bytes = k + n * (4 + x.element_size()) + 4
    if _kitchen_hip:
        # HIP may copy strided inputs and spill rotated rows plus group maxima.
        row_bytes += 2 * k * x.element_size() + 4 * (k // 256)
    rows = max(1, _SCRATCH_BYTES // row_bytes)
    if rows >= 32:
        rows = rows // 32 * 32
    if m <= rows:
        return _kitchen.int8_linear(x, weight, scale, bias=bias, out_dtype=x.dtype, convrot=convrot)
    if (_direct_cutlass and x.dtype in (torch.bfloat16, torch.float16)
            and x.is_contiguous() and weight.is_contiguous() and n % 8 == 0
            and k % 256 == 0 and 256 <= k <= 16384
            and (not convrot or _kitchen._convrot_fused_shared_memory_fits(x, k, 256))):
        return _cutlass_linear_chunked(x, weight, scale, bias, convrot)
    out = x.new_empty((m, n))
    for start in range(0, m, rows):
        stop = min(start + rows, m)
        out[start:stop].copy_(_kitchen.int8_linear(
            x[start:stop], weight, scale, bias=bias, out_dtype=x.dtype, convrot=convrot))
    return out


def _cutlass_linear_chunked(x, weight, scale, bias, convrot):
    """Write Kitchen's CUTLASS result directly into output slices on supported CUDA GPUs.

    Only quantized row tiles and their scales occupy scratch space. Avoiding a
    temporary output permits wide GEMMs without tiny, slow row chunks.
    """
    m, k = x.shape
    rows = max(32, (_SCRATCH_BYTES // (k + 4)) // 32 * 32)
    out = x.new_empty((m, weight.shape[0]))
    wrap = _kitchen._wrap_for_dlpack
    bias_arg = (_kitchen._gemm_vector_arg(bias, x.device, x.dtype) if bias is not None
                else _kitchen._empty_cuda_tensor(x.device, x.dtype))
    stream = torch._C._cuda_getCurrentRawStream(x.device.index)
    for start in range(0, m, rows):
        stop = min(start + rows, m)
        if convrot:
            q, scales = _kitchen.quantize_int8_rowwise_convrot64(x[start:stop], 256)
        else:
            q, scales = _kitchen.quantize_int8_rowwise(x[start:stop])
        used = _kitchen._C.cutlass_int8_dequant(
            wrap(q), wrap(weight), wrap(scales), wrap(scale), wrap(bias_arg),
            wrap(out[start:stop]), _kitchen.DTYPE_TO_CODE[x.dtype], stream)
        if not used:
            raise RuntimeError("Comfy Kitchen rejected the INT8 CUTLASS output tile")
        del q, scales
    return out


def prepare_capture_stream(stream):
    """Kitchen's stream-K GEMMs keep a workspace per stream that they allocate on first need and enlarge later: a CUDA graph capture
    cannot allocate it, and an enlargement would free the workspace of the graphs captured before. The capture stream gets the largest
    one once (a block per SM with the widest stream-K tile); a capture that would still need more fails instead of replaying stale memory."""
    key = (stream.cuda_stream, revision)
    if _backend != "kitchen" or _kitchen_hip or _kitchen._DISABLE_CUTLASS_INT8 or key in _prepared_capture_streams:
        return
    device = stream.device
    if torch.cuda.get_device_capability(device) >= (8, 0):
        sms = torch.cuda.get_device_properties(device).multi_processor_count
        m, k = 128 * (2 * sms - 1), 512  # one wave plus all the SMs but one: the most stream-K tiles and blocks
        wrap = _kitchen._wrap_for_dlpack
        with torch.inference_mode():
            a = torch.zeros((m, k), dtype=torch.int8, device=device)
            xs = torch.ones((m, 1), dtype=torch.float32, device=device)
            calls = [(config, torch.zeros((n, k), dtype=torch.int8, device=device), torch.ones(n, dtype=torch.float32, device=device),
                      torch.empty((m, n), dtype=torch.bfloat16, device=device)) for config, n in ((12, 128), (13, 256))]  # stream-K tiles
            stream.wait_stream(torch.cuda.current_stream(device))
            for config, b, ws, out in calls:
                if not _kitchen._C.cutlass_int8_dequant_config(wrap(a), wrap(b), wrap(xs), wrap(ws), wrap(out),
                                                               _kitchen.DTYPE_TO_CODE[torch.bfloat16], config, stream.cuda_stream):
                    raise RuntimeError(f"Comfy Kitchen could not prepare the stream-K workspace of tile config {config} for CUDA graphs")
            torch.cuda.current_stream(device).wait_stream(stream)
    _prepared_capture_streams.add(key)


@torch.inference_mode()
def _probe_direct_cutlass():
    """Check the actual output-buffer kernel at backend setup, never during capture."""
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        x = torch.arange(65 * 256, device=device, dtype=torch.float32).remainder_(251).sub_(125).div_(128).reshape(65, 256)
        weight = (torch.arange(264 * 256, device=device, dtype=torch.int32) % 255 - 127).to(torch.int8).reshape(264, 256)
        scale = torch.full((264,), 1 / 127, device=device, dtype=torch.float32)
        for dtype in (torch.float16, torch.bfloat16):
            inputs = x.to(dtype)
            for bias in (None, scale.to(dtype)):
                for convrot in (False, True):
                    expected = _kitchen.int8_linear(inputs, weight, scale, bias=bias, out_dtype=dtype, convrot=convrot)
                    actual = _cutlass_linear_chunked(inputs, weight, scale, bias, convrot)
                    if not torch.isfinite(actual).all() or not torch.equal(actual, expected):
                        raise RuntimeError("output-buffer kernel did not match Kitchen INT8")
        return True
    except RuntimeError as exc:
        print(f"[INT8] Direct Output Tiles Unavailable: {exc}. Using existing Kitchen batching.")
        return False


def _register_ops():
    global _ops_registered
    if _ops_registered:
        return

    # WGPKernels: the path (full precision or per-block activations) is chosen when the operator runs, never in a compiled graph.
    @torch.library.custom_op("wan2gp_wgp::linear", mutates_args=())
    def wgp_linear_op(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None,
                      group: int, precise: bool) -> torch.Tensor:
        return _wgp_run(x, weight, scale, bias, group, precise)

    @wgp_linear_op.register_fake
    def wgp_fake(x, weight, scale, bias, group, precise):
        return x.new_empty((x.shape[0], weight.shape[0]))

    @torch.library.custom_op("wan2gp_kitchen::linear", mutates_args=())
    def linear(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
               bias: torch.Tensor | None, convrot: bool) -> torch.Tensor:
        return _linear_impl(x, weight, scale, bias, convrot)

    @linear.register_fake
    def fake(x, weight, scale, bias, convrot):
        return x.new_empty((x.shape[0], weight.shape[0]))

    @torch.library.custom_op("wan2gp_kitchen::blockwise_linear", mutates_args=())
    def blockwise_linear(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
                         bias: torch.Tensor | None) -> torch.Tensor:
        return _blockwise_linear_impl(x, weight, scale, bias)

    @blockwise_linear.register_fake
    def blockwise_fake(x, weight, scale, bias):
        return x.new_empty((x.shape[0], weight.shape[0]))

    @torch.library.custom_op("wan2gp_kitchen::residual_linear", mutates_args=())
    def residual_linear(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor,
                        bias: torch.Tensor | None) -> torch.Tensor:
        return _residual_linear_impl(x, weight, scale, bias)

    @residual_linear.register_fake
    def residual_fake(x, weight, scale, bias):
        return x.new_empty((x.shape[0], weight.shape[0]))

    _ops_registered = True


def kitchen_linear(input, weight, bias=None, *, convrot=False):
    if convrot and _sm120_cutlass and input.shape[-1] > 16384:
        if (_wide_convrot_triton is not None and not torch.compiler.is_compiling()
                and not triton._is_fake_tensor(input)):
            return _wide_convrot_linear(input, weight, bias)
        # The wide cuBLAS path cannot be captured on validated SM120. Keep a
        # graph-safe Quanto fallback when Triton is unavailable.
        from shared.qtypes.int8_convrot import _rotate_activation
        return torch.nn.functional.linear(_rotate_activation(input, 256), weight, bias)
    n = weight._data.shape[0]  # the plain INT8 data: reading the quantized weight's shape costs a Python dispatch
    scale = _weight_scale(weight, n, input.device)
    x = input.reshape(-1, input.shape[-1])
    if torch.compiler.is_compiling() or triton._is_fake_tensor(input):
        out = torch.ops.wan2gp_kitchen.linear(x, weight._data, scale, bias, convrot)
    else:
        out = _linear_impl(x, weight._data, scale, bias, convrot)
    return out.reshape(*input.shape[:-1], n)


def _quanto_forward(ctx, input, weight, bias=None):
    if (type(input) is torch.Tensor and input.is_cuda
            and input.dtype in (torch.float16, torch.bfloat16, torch.float32)
            and weight._data.is_cuda and weight._data.dtype == torch.int8
            and (input.shape[-1] <= 16384 or not _sm120_cutlass)
            and weight._data.shape[-1] % (16 if _kitchen_hip else 4) == 0 and weight._data.shape[0] % 4 == 0):  # the kitchen GEMM rejects other widths
        ctx.save_for_backward(input, weight)
        return kitchen_linear(input, weight, bias)
    return _original_forward(ctx, input, weight, bias)


def _wgp_quanto_forward(ctx, input, weight, bias=None):
    if _wgp_supports(input, weight):
        ctx.save_for_backward(input, weight)
        prepared = weight.__dict__.get('_wangp_prequantized_input')
        if prepared is not None:
            return linear_prequantized(weight, bias, *prepared)
        fusion = weight.__dict__.get('_wangp_linear_fusion')  # set by linear_with_fusion for this call (plain INT8: rotation group 0)
        if fusion is None:
            return wgp_linear(input, weight, bias)
        input_act, act_weight, act_eps, residual, residual_scale, out = fusion
        return wgp_linear(input, weight, bias, act=input_act, act_weight=act_weight, act_eps=act_eps, residual=residual,
                          residual_scale=residual_scale, out=out)
    return _original_forward(ctx, input, weight, bias)


class _WgpRowProjection:
    """mmgp row projection (offload.linear_rows) of the INT8 ConvRot and Quanto INT8 layers computed with WGPKernels: the input is
    prepared once (rotated and quantized for the path chosen with the first module's weight), shared by the modules of the same rotation
    group and precision, and each range of output rows is the same linear on a view of the weight rows. The prepared input is at most as
    large as the input it replaces (prepare_linear_input releases it)."""
    @staticmethod
    def supports(module, x):
        qtype = getattr(module, "weight_qtype", None)
        if (_backend != "wgpkernels" or qtype is None or qtype.name not in ("qint8", "qint8_convrot") or torch.compiler.is_compiling()
                or triton._is_fake_tensor(x) or getattr(module, "activation_qtype", None) is not None):
            return False
        if qtype.name == "qint8" and getattr(module, "_router_forward_impl", None) is not None:
            return False
        weight = module.qweight
        return (_wgp_supports(x, weight) and x.is_contiguous() and x.data_ptr() % 16 == 0 and weight._data.is_contiguous()
                and (qtype.name == "qint8" or x.shape[-1] % module._convrot_group_size == 0))

    @staticmethod
    def key(module):
        group = module._convrot_group_size if module.weight_qtype.name == "qint8_convrot" else 0
        return "wgp_rows", group, hasattr(module, "_int8_residual_forward")

    @staticmethod
    def prepare(module, x):
        _, group, precise = _WgpRowProjection.key(module)
        path = _wgp_path(module, x, group, precise)
        if path != "block" or x.shape[0] <= _WGP_BLOCK_ROWS:
            return _wgp.prepare(x, group, path)
        # wq8_quantize_blocks puts the rows on the CUDA grid's y axis, at most 65535 per call (else CUDA error 1): longer inputs are
        # quantized by pieces; the GEMM that follows has no row limit
        parts = [_wgp.prepare(x[start:start + _WGP_BLOCK_ROWS], group, path) for start in range(0, x.shape[0], _WGP_BLOCK_ROWS)]
        return _wgp.Prepared(path, torch.cat([part.x for part in parts]), None, torch.cat([part.scale for part in parts]), x.dtype)

    @staticmethod
    def rows(module, prepared, start, stop):
        weight, bias = module.qweight, module.bias
        scale = _weight_scale(weight, weight._data.shape[0], weight._data.device)[start:stop]
        bias = None if bias is None else bias[start:stop].to(prepared.dtype)
        return _wgp.linear_prepared(prepared, weight._data[start:stop], scale, bias)


_WGP_ROW_PROJECTION = _WgpRowProjection()


class _Int8RowProjection:
    """Preserve PyTorch weight-only or Triton blockwise activation arithmetic for output row ranges."""
    @staticmethod
    def supports(module, x):
        qtype = getattr(module, "weight_qtype", None)
        if (_backend not in ("pytorch", "triton") or qtype is None or qtype.name not in ("qint8", "qint8_convrot")
                or triton._is_fake_tensor(x) or torch.compiler.is_compiling() or getattr(module, "activation_qtype", None) is not None):
            return False
        if qtype.name == "qint8" and getattr(module, "_router_forward_impl", None) is not None:
            return False
        if (_backend == "triton" and qtype.name == "qint8_convrot" and triton.FUSED_CONVROT_ENABLED
                and getattr(module, "_convrot_group_size", 0) == 256 and x.shape[0] <= 4):
            return False  # the separate tiny-M fused rotation has its own rounding
        return type(x) is torch.Tensor and module.qweight._data.dtype == torch.int8 and module.qweight.dtype == x.dtype

    @staticmethod
    def key(module):
        group = getattr(module, "_convrot_group_size", 0)
        return ("int8_rows", _backend, group, module.out_features) if _backend == "triton" or group else None

    @staticmethod
    def prepare(module, x):
        from shared.qtypes.int8_convrot import _rotate_activation
        x = _rotate_activation(x, getattr(module, "_convrot_group_size", 0))
        if (_backend == "triton" and x.is_cuda and not triton._prefer_native_quanto_path(x)):
            return triton._TRITON_MODULE.prepare_blockscale_input(x, module.out_features)
        return x

    @staticmethod
    def rows(module, prepared, start, stop):
        weight = module.qweight
        data = weight._data[start:stop]
        scale = weight._scale if weight._scale.numel() == 1 else weight._scale[start:stop]
        if isinstance(prepared, tuple):
            scale = triton._prepare_weight_scale(scale, stop - start, data.device)
            out = triton._TRITON_MODULE.prepared_blockscale_mm(prepared, data, scale)
            return out if module.bias is None else out.add_(module.bias[start:stop])
        elif prepared.device.type == "mps":
            out = prepared @ (data.to(prepared.dtype) * scale.to(prepared.dtype)).t()
        else:
            out = torch.ops.quanto.qbytes_mm(prepared, data, scale.to(prepared.dtype))
        return triton._add_bias_in_place_or_fallback(out, None if module.bias is None else module.bias[start:stop])


_INT8_ROW_PROJECTION = _Int8RowProjection()


class _KitchenRowProjection:
    """mmgp row projection (offload.linear_rows) of the INT8 ConvRot and Quanto INT8 layers that kitchen_linear computes with
    _cutlass_linear_chunked: the input is quantized once, each range of output rows is the same CUTLASS GEMM on a view of the weight
    rows, so the rows equal those of the layer's own output. The quantized input is as large as half the input it replaces: callers
    release the input (prepare_linear_input takes it)."""

    @staticmethod
    def supports(module, x):
        qtype = getattr(module, "weight_qtype", None)
        if (_kitchen is None or not _direct_cutlass or qtype is None or torch.compiler.is_compiling() or not x.is_cuda
                or hasattr(module, "_int8_residual_forward")
                or x.dtype not in (torch.bfloat16, torch.float16) or not x.is_contiguous()):
            return False
        convrot = qtype.name == "qint8_convrot"
        if convrot and (getattr(module, "_convrot_group_size", 0) != 256 or not kitchen_enabled()):
            return False
        if not convrot and (qtype.name != "qint8" or getattr(module, "_router_forward_impl", None) is not None):
            return False
        weight = module.qweight
        data = weight._data
        (m, k), n = x.shape, data.shape[0]
        rows = max(1, _SCRATCH_BYTES // (k + n * (4 + x.element_size()) + 4))  # _linear_impl's single call threshold
        rows = rows // 32 * 32 if rows >= 32 else rows
        return (m > rows and data.dtype == torch.int8 and data.is_cuda and data.is_contiguous() and weight.dtype == x.dtype
                and n % 8 == 0 and k % 256 == 0 and 256 <= k <= 16384 and (not convrot or _kitchen._convrot_fused_shared_memory_fits(x, k, 256)))

    @staticmethod
    def key(module):
        return "kitchen_int8_convrot" if module.weight_qtype.name == "qint8_convrot" else "kitchen_int8"

    @staticmethod
    def prepare(module, x):
        (m, k), convrot = x.shape, module.weight_qtype.name == "qint8_convrot"
        rows = max(32, (_SCRATCH_BYTES // (k + 4)) // 32 * 32)
        q, scales = torch.empty((m, k), dtype=torch.int8, device=x.device), None
        for start in range(0, m, rows):
            stop = min(start + rows, m)
            tile_q, tile_scales = _kitchen.quantize_int8_rowwise_convrot64(x[start:stop], 256) if convrot else _kitchen.quantize_int8_rowwise(x[start:stop])
            if scales is None:
                scales = torch.empty((m, *tile_scales.shape[1:]), dtype=tile_scales.dtype, device=x.device)
            q[start:stop].copy_(tile_q)
            scales[start:stop].copy_(tile_scales)
            del tile_q, tile_scales
        return q, scales, x.dtype

    @staticmethod
    def rows(module, prepared, start, stop):
        q, scales, dtype = prepared
        weight, bias = module.qweight, module.bias
        (m, k), device = q.shape, q.device
        data = weight._data[start:stop]
        scale = _weight_scale(weight, weight._data.shape[0], device)[start:stop]
        bias_arg = (_kitchen._gemm_vector_arg(bias[start:stop], device, dtype) if bias is not None
                    else _kitchen._empty_cuda_tensor(device, dtype))
        out = torch.empty((m, stop - start), dtype=dtype, device=device)
        wrap = _kitchen._wrap_for_dlpack
        stream = torch._C._cuda_getCurrentRawStream(device.index)
        tile = max(32, (_SCRATCH_BYTES // (k + 4)) // 32 * 32)
        for row in range(0, m, tile):
            end = min(row + tile, m)
            if not _kitchen._C.cutlass_int8_dequant(wrap(q[row:end]), wrap(data), wrap(scales[row:end]), wrap(scale), wrap(bias_arg),
                                                     wrap(out[row:end]), _kitchen.DTYPE_TO_CODE[dtype], stream):
                raise RuntimeError("Comfy Kitchen rejected the INT8 CUTLASS row range")
        return out


_KITCHEN_ROW_PROJECTION = _KitchenRowProjection()


def configure(selection, verbose_level=0, *, resolved=None):
    global _backend, _kitchen, _kitchen_hip, _original_forward, _direct_cutlass, _sm120_cutlass, _wide_convrot_triton, revision
    global _blockwise_triton, _kitchen_dlpack_export, _wgp
    backend, module = resolve_backend(selection) if resolved is None else resolved
    kernel_policy.configure_families(selection)
    previous_backend = _backend
    if _kitchen_dlpack_export is not None:
        _kitchen_dlpack_export[0]._wrap_for_dlpack = _kitchen_dlpack_export[1]
        _kitchen_dlpack_export = None
    if _original_forward is not None:
        from optimum.quanto.tensor.weights import qbytes
        qbytes.WeightQBytesLinearFunction.forward = staticmethod(_original_forward)
        _original_forward = None
    triton.disable_quanto_int8_kernel()
    os.environ["WAN2GP_QUANTO_INT8_KERNEL"] = "1" if backend == "triton" else "0"
    _backend, _kitchen, _wgp = "pytorch", None, None
    _kitchen_hip = False
    _direct_cutlass = False
    _sm120_cutlass = False
    _wide_convrot_triton = None
    if backend == "triton":
        if not triton.maybe_enable_quanto_int8_kernel(verbose_level):
            raise RuntimeError("Failed to enable Triton INT8 kernels")
    elif backend == "wgpkernels":
        from optimum.quanto.tensor.weights import qbytes
        _register_ops()
        _wgp = module
        _original_forward = qbytes.WeightQBytesLinearFunction.forward
        qbytes.WeightQBytesLinearFunction.forward = staticmethod(_wgp_quanto_forward)
    elif backend == "kitchen":
        from optimum.quanto.tensor.weights import qbytes
        _register_ops()
        _kitchen = module
        _kitchen_hip = torch.version.hip is not None
        if not _kitchen_hip:
            _kitchen_dlpack_export = (module, module._wrap_for_dlpack)
            module._wrap_for_dlpack = _export_dlpack
        if not _kitchen_hip and not module._DISABLE_CUTLASS_INT8:
            capability = torch.cuda.get_device_capability()
            _sm120_cutlass = capability == (12, 0)
            _direct_cutlass = _sm120_cutlass or (capability >= (8, 0) and _probe_direct_cutlass())
        if _sm120_cutlass:
            _wide_convrot_triton, _ = triton._probe_triton_backend()
        if _blockwise_required:
            _blockwise_triton, reason = triton._probe_triton_backend()
            label = "Triton" if _blockwise_triton is not None else f"PyTorch ({reason})"
            print(f"[INT8] Precision Rules: {label} for protected linear layers.")
        _original_forward = qbytes.WeightQBytesLinearFunction.forward
        qbytes.WeightQBytesLinearFunction.forward = staticmethod(_quanto_forward)
    _backend = backend
    from mmgp import cuda_graphs, offload
    from shared.qtypes import quanto_rows
    offload.register_row_projection(_INT8_ROW_PROJECTION)
    (offload.register_row_projection if backend == "kitchen" else offload.unregister_row_projection)(_KITCHEN_ROW_PROJECTION)
    (offload.register_row_projection if backend == "wgpkernels" else offload.unregister_row_projection)(_WGP_ROW_PROJECTION)
    cuda_graphs.register_capture_setup(prepare_capture_stream)
    if backend != previous_backend:
        revision += 1
    label = {"wgpkernels": "WGPKernels", "kitchen": "Comfy Kitchen HIP" if _kitchen_hip else "Comfy Kitchen CUDA", "triton": "Triton",
             "pytorch": "PyTorch"}[backend]
    print(f"[INT8] Backend: {label} (setting: {selection}).")
    return backend != "pytorch"
