# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Optional INT8 backends shared by Quanto and ConvRot linear layers."""
from __future__ import annotations

import contextlib
import contextvars
import ctypes
import importlib
import os
from concurrent.futures import ThreadPoolExecutor
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
_kitchen_triton = None  # Triton INT8 kernels for the layers the Kitchen GEMM cannot take, when Kitchen is the backend
_rotated_triton = None  # fused ConvRot Triton kernels for large inputs (Triton backend, or Kitchen's Triton route)
_column_triton = None  # the same kernels' quantizer writing in place into a ColumnInput, when WGPKernels is the backend
_blockwise_triton = None
_blockwise_required = False
_kitchen_dlpack_export = None  # (Kitchen module, its own DLPack exporter) while the direct exporter replaces it
_prepared_capture_streams = set()
# Bump when changes to INT8 custom operators invalidate compiled graphs.
_COMPILE_CACHE_VERSION = 4
# Includes row quantization, a possible INT32 GEMM result, and a temporary output.
# Never allocate an activation-sized quantization buffer for a whole video.
_SCRATCH_BYTES = 16 * 1024 * 1024


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
        # Every kernel once with fixed configurations, at selection time and outside graph capture. Its reference uses cuBLAS, whose workspaces
        # stay per thread and stream: on a thread of its own, the next thread inherits this one instead of allocating another.
        def selftest():
            with torch.inference_mode():
                return module.selftest()
        with ThreadPoolExecutor(1) as thread:
            failure = thread.submit(selftest).result()
        return (None, failure) if failure else (module, "available")
    except Exception as exc:
        return None, str(exc)


def resolve_backend(selection, precision):
    """Validate user selection before changing the active runtime or config. The selected backend, else the Auto choice: WGPKernels,
    then Comfy Kitchen (skipped with the "strict" kernel precision: less precise than PyTorch), then Triton, then PyTorch
    (kernel_policy.RANKING)."""
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
        if backend != selection and (backend != "kitchen" or precision != "strict"):
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


CHUNK_SCRATCH_BYTES = 64 * 1024 * 1024  # INT8 activations of one launch inside chunked_rows(); 0: the kernels' own 16 MiB (comparison switch)
_chunked_rows = contextvars.ContextVar("wan2gp_int8_chunked_rows", default=False)


@contextlib.contextmanager
def chunked_rows():
    """For a caller that already feeds its INT8 linears chunks of rows it sized itself (ops.mlp): within it, WGPKernels quantizes and
    multiplies up to CHUNK_SCRATCH_BYTES of INT8 activations per launch instead of cutting each chunk again by its 16 MiB: the same
    results (rows are quantized on their own, the products are exact integers) in a third of the kernels, which are nodes of a
    block's CUDA graph."""
    token = _chunked_rows.set(True)
    try:
        yield
    finally:
        _chunked_rows.reset(token)


def _wgp_linear(x, weight, scale, bias, group, precise, act=None, act_weight=None, eps=0.0, residual=None, residual_scale=None, out=None):
    x = _wgp_aligned(x)
    if not (_chunked_rows.get() and CHUNK_SCRATCH_BYTES):
        return _wgp.linear(x, weight, scale, bias, group, precise, act=act, act_weight=act_weight, eps=eps, residual=residual, residual_scale=residual_scale, out=out)
    path = _wgp.choose_path(x, weight, scale, bias, group, precise, act, act_weight, eps)
    (m, k), n = (x.shape[0], weight.shape[1]), weight.shape[0]
    row_bytes = (2 * k if path == "exact" else k) + 4 * (-(-k // 128) if path == "block" else 1)  # the kernels' own rule (_gemm_path)
    rows = max(128, CHUNK_SCRATCH_BYTES // row_bytes // 128 * 128)
    if path in ("lines", "small") or m <= max(128, _wgp.SCRATCH_BYTES // row_bytes // 128 * 128):
        return _wgp.linear(x, weight, scale, bias, group, precise, mode=path, act=act, act_weight=act_weight, eps=eps, residual=residual, residual_scale=residual_scale, out=out)
    out = torch.empty((m, n), dtype=x.dtype, device=x.device) if out is None else out
    for start in range(0, m, rows):
        stop = min(start + rows, m)
        prepared = _wgp.prepare(x[start:stop], group, path, act, act_weight, eps)
        _wgp.linear_prepared(prepared, weight, scale, bias, None if residual is None else residual[start:stop], residual_scale, out[start:stop])
        del prepared
    return out


def _wgp_run(x, weight, scale, bias, group, precise):
    return _wgp_linear(x, weight, scale, bias, group, precise)


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
    y = _wgp_linear(x, weight._data, scale, bias, group, precise, act=act, act_weight=act_weight, eps=act_eps, residual=residual,
                    residual_scale=residual_scale, out=y)
    if out is None:
        return y.reshape(*input.shape[:-1], n)
    torch.autograd.graph.increment_version(out)  # written in place: MMGP CUDA graphs copy a mutated block argument back
    return out


def _row_offsets(lib):
    """The WGPKernels entry points addressing rows through a device counter (CUDA graph loops), set up once; False without them."""
    if not hasattr(lib, "wq8_gemm_at"):
        return False
    if not getattr(lib, "_wangp_row_offsets", False):
        p, i, i64, f = ctypes.c_void_p, ctypes.c_int, ctypes.c_int64, ctypes.c_float
        lib.wq8_quantize_blocks_at.argtypes = [p, p, p, i, i, i64, i, i, i, p, f, p, p, i64, i64, p]
        lib.wq8_quantize_at.argtypes = [p, p, p, p, i, i, i64, i, i, i, p, f, p, i64, i64, p]
        lib.wq8_gemm_at.argtypes = [p, p, p, p, p, p, p, p, p, i, i, i, i, i, i, i, p, p, p, i64, i64, i64, i64, i64, i64, p]
        lib._wangp_row_offsets = True
    return True


_zeros = {}  # device -> a device int64 zero: the counter of row-offset launches outside graph loops


def _loop_group(module):
    """The rotation group the row-offset kernels run a layer with: a ConvRot INT8 weight's (0, 16, 64, 256), 0 for a plain Quanto INT8
    weight (the regular WGPKernels path's group 0: no rotation, the same quantizer and GEMM); None for another layer."""
    name = getattr(getattr(module, "weight_qtype", None), "name", None)
    if name == "qint8_convrot":
        return module._convrot_group_size if module._convrot_group_size in (0, 16, 64, 256) else None
    if (name == "qint8" and getattr(module, "activation_qtype", None) is None and getattr(module, "_router_forward_impl", None) is None
            and not hasattr(module, "_int8_residual_forward")):  # require_int8_residual: its own forward keeps full precision activations
        return 0
    return None


def segment_gated_linear_supported(module, x):
    """Whether segment_gated_linear_ takes this layer and input: WGPKernels with row-offset entry points, a ConvRot or Quanto INT8 weight
    without LoRA, a BF16 / FP16 CUDA input (what linear's fused residual path takes)."""
    from mmgp.offload import lora_active
    return (_backend == "wgpkernels" and _row_offsets(_wgp._lib) and _loop_group(module) is not None and not lora_active(module)
            and _wgp_supports(x, module.qweight) and not torch.compiler.is_compiling())


def _layer_operands(module, dtype, device):
    weight = module.qweight
    return weight._data, _loop_group(module), weight._data.shape[0], weight._data.shape[1], _weight_scale(weight, weight._data.shape[0], device),         None if module.bias is None else module.bias.to(dtype)


def _zero(device):
    zero = _zeros.get(device)
    if zero is None:
        from mmgp import allocator
        with allocator.normal_allocations():  # outlives the block: made during a warmup, it would keep that graph pool alive after its graphs
            zero = _zeros[device] = torch.zeros(1, dtype=torch.int64, device=device)
    return zero


def segment_run_path(module, x, start, stop, offset, span):
    """The path of segment_gated_linear_ for the run of x's rows [start, stop) (segments of span rows, the first offset rows in): full
    precision ('exact') where any of its segments' own calls would take it, else the block path."""
    w, group, _, _, scale, bias = _layer_operands(module, x.dtype, x.device)
    lengths = {min(stop - start, span - offset), min(span, stop - start), (stop - start - (span - offset)) % span or span}
    paths = {_wgp.choose_path(x[start:start + length], w, scale, bias, group, False) for length in lengths if 0 < length <= stop - start}
    return "exact" if paths - {"block", "row"} else "block" if "block" in paths else "row"


def _gated_gemms(module, x, path, rows_like, y_row0, gate_row0, gate_step, span, counter, first, step):
    """x's rows through module, added scaled by their segment's gate into the output rows first + r (+ counter * step) counted from the row
    y_row0 points to (whose segment's gate row gate_row0 points to, span rows a segment): per 16 MiB of rows, the prepared rows and one GEMM
    whose residual epilogue reads each row's gate. rows_like: a tensor of the output rows' shape and dtype (the GEMM plans' shapes)."""
    w, group, n, k, scale, bias = _layer_operands(module, x.dtype, x.device)
    lib, stream, sms, dtype = _wgp._lib, _wgp._stream(x.device), _wgp._sm_count(x.device), _wgp._DTYPES[x.dtype]
    row_bytes = (2 * k if path == "exact" else k) + 4 * (-(-k // 128) if path == "block" else 1)
    rows = max(128, _wgp.SCRATCH_BYTES // row_bytes // 128 * 128)
    for start in range(0, x.shape[0], rows):
        stop = min(x.shape[0], start + rows)
        prepared = _wgp.prepare(x[start:stop], group, path)
        variant, split, slot_bytes, counter_bytes = _wgp._plan(prepared.x, prepared.lo, prepared.scale, w, scale, bias, rows_like[:stop - start], path, True)
        slots = torch.empty(slot_bytes, dtype=torch.uint8, device=x.device) if slot_bytes else None
        counters = _wgp._zeroed_counters(x.device, counter_bytes, stream).data_ptr() if counter_bytes else None
        _wgp._check(lib.wq8_gemm_at(prepared.x.data_ptr(), _wgp._ptr(prepared.lo), prepared.scale.data_ptr(), w.data_ptr(), scale.data_ptr(), _wgp._ptr(bias), y_row0,
                                    gate_row0, y_row0, stop - start, n, k, dtype, variant, sms, split, _wgp._ptr(slots), counters, counter.data_ptr(),
                                    first + start, step, span, gate_step, 0, 0, stream))
        del prepared, slots


def segment_gated_linear_(module, x, residual, gate, runs):
    """residual += gate[row] * module(x) row by row, gate a table of rows (one per segment), for runs (start, stop, row0, offset, span) of
    segments (row_loop_triton.segment_runs: x's row r of a run takes gate row row0 + (offset + r - start) // span): one GEMM per 16 MiB of
    rows, the GEMM's residual epilogue reading each row's gate (one rounding), instead of one linear per segment (segment_run_path: never
    a less precise path than theirs; a chunk's GEMM may split K otherwise than one of a segment's rows)."""
    gate, x = gate.to(x.dtype), _wgp_aligned(x)
    for start, stop, row0, offset, span in runs:  # positions from the run's first segment's start: the GEMM adds them to its output rows
        base = residual[start].data_ptr() - offset * residual.shape[1] * residual.element_size()
        _gated_gemms(module, x[start:stop], segment_run_path(module, x, start, stop, offset, span), residual, base, gate[row0].data_ptr(), gate.stride(0), span,
                     _zero(x.device), offset, 0)
    torch.autograd.graph.increment_version(residual)
    return residual


def segment_gated_linear_at(module, x, residual, gate, span, path, counter, first, step):
    """segment_gated_linear_ of a CUDA graph loop's turn: x's rows (fixed) added into residual's rows first + counter * step, gate row
    (row // span) of the table (segments of span rows from residual's row 0), with the path segment_run_path gave the eager calls."""
    _gated_gemms(module, _wgp_aligned(x), path, residual, residual.data_ptr(), gate.to(x.dtype).data_ptr(), gate.stride(0), span, counter, first, step)
    torch.autograd.graph.increment_version(residual)
    return residual


def prepared_rows_linear_at(module, prepared, count, counter, first, step):
    """offload.linear_rows(module, rows, 0, out_features) of the prepared input's rows first + counter * step (a CUDA graph loop's turn), count
    of them (the GEMM of linear_prepared on those rows, its plan for that row count: the same bits): a new tensor (count, out_features)."""
    w, _, n, k, scale, _ = _layer_operands(module, prepared.dtype, prepared.x.device)
    bias = None if module.bias is None else module.bias.to(prepared.dtype)
    view = prepared.rows(0, count)
    out = torch.empty((count, n), dtype=prepared.dtype, device=prepared.x.device)
    variant, split, slot_bytes, counter_bytes = _wgp._plan(view.x, view.lo, view.scale, w, scale, bias, out, prepared.path, False)
    slots = torch.empty(slot_bytes, dtype=torch.uint8, device=out.device) if slot_bytes else None
    stream = _wgp._stream(out.device)
    counters = _wgp._zeroed_counters(out.device, counter_bytes, stream).data_ptr() if counter_bytes else None
    _wgp._check(_wgp._lib.wq8_gemm_at(prepared.x.data_ptr(), _wgp._ptr(prepared.lo), prepared.scale.data_ptr(), w.data_ptr(), scale.data_ptr(), _wgp._ptr(bias), None, None,
                                      out.data_ptr(), count, n, k, _wgp._DTYPES[prepared.dtype], variant, _wgp._sm_count(out.device), split, _wgp._ptr(slots), counters,
                                      counter.data_ptr(), 0, 0, 0, 0, first, step, stream))
    return out


def row_loop_linear(module, x, rows, act=None, out_at=False):
    """A RowLoopLinear for module over chunks of `rows` rows of inputs like x (act: the input operation fused into its quantizer, x's rows
    then gate | value for 'swiglu'; out_at: its calls will write their output rows through the counter), or None: a ConvRot or Quanto INT8
    weight without LoRA, BF16 / FP16 activations, eager inference, on WGPKernels with row-offset entry points or on the Triton backend's
    kernels (TritonRowLoopLinear, without act)."""
    from mmgp.offload import lora_active
    if lora_active(module) or not torch.is_inference_mode_enabled() or _loop_group(module) is None:
        return None
    if _backend == "triton":
        return TritonRowLoopLinear(module, x, rows, out_at) if act is None and _triton_loop_supported(module, x[:rows]) else None
    if _backend != "wgpkernels" or not _row_offsets(_wgp._lib) or not _wgp_supports(x, module.qweight):
        return None
    return RowLoopLinear(module, x, rows, act)


def _triton_loop_supported(module, x):
    """Whether the Triton backend runs this layer on x's rows with the kernels TritonRowLoopLinear reproduces: a ConvRot weight on the
    fused rotation kernels (rotated_triton_supported), a Quanto weight on the fused quantizing GEMM (quanto_int8_inject's dense path)."""
    weight = module.qweight
    if x.dtype not in (torch.bfloat16, torch.float16) or not x.is_cuda or not weight._data.is_cuda:
        return False
    if module.weight_qtype.name == "qint8_convrot":
        return rotated_triton_supported(x, weight, module._convrot_group_size)
    from shared.kernels import quanto_int8_inject as quanto
    return (quanto._PATCH_STATE.enabled and quanto._TRITON_MODULE is not None and quanto._TRITON_DIRECT_FUSED_READY and weight._data.dtype == torch.int8
            and not quanto._prefer_native_quanto_path(x))


class TritonRowLoopLinear:
    """RowLoopLinear for the Triton backend: a chunk of `rows` rows of a ConvRot INT8 linear on its fused rotation kernels (rotated_linear's
    launches, sub-chunks and configurations) or of a Quanto INT8 linear on the fused quantizing GEMM then its bias added in place (the dense
    path of quanto_int8_inject): the same bits. The input rows are read through the counter (x_at); the output goes to fixed rows, or with
    out_at to a fixed scratch whose rows are then stored or added (residual, row_loop_triton.residual_add_at: the usual path's add_ /
    addcmul_) through the counter. Every operand and scratch is made here, before the loop's body is recorded."""

    def __init__(self, module, x, rows, out_at=False):
        weight = module.qweight
        self.n, self.k, self.rows, self.bias = weight._data.shape[0], weight._data.shape[1], rows, module.bias
        self.y = torch.empty((rows, self.n), dtype=x.dtype, device=x.device) if out_at else None
        if module.weight_qtype.name == "qint8_convrot":
            self.group, self.w, self.scale = module._convrot_group_size, weight._data, _weight_scale(weight, self.n, x.device)
            sub = _rotated_triton.scratch_rows(self.k)
            self.parts = [(start, min(start + sub, rows)) for start in range(0, rows, sub)]
            self.q = torch.empty((min(sub, rows), self.k), dtype=torch.int8, device=x.device)
            self.scales = torch.empty((min(sub, rows), self.k // _rotated_triton.SCALE_BLOCK), dtype=torch.float32, device=x.device)
            self.configs = _rotated_triton.loop_configs(x.dtype, x.device, self.n, self.k, self.group, self.bias)
        else:
            from shared.kernels import quanto_int8_inject as quanto
            self.group, self.w = None, weight._data if weight._data.is_contiguous() else weight._data.contiguous()
            self.scale = quanto._prepare_weight_scale(weight._scale, self.n, x.device)
            self.params, self.kernel = quanto._fused_launch_params(rows, self.k, self.n, x.device), quanto._TRITON_MODULE._fused_dynamic_int8_blockscale_gemm_kernel

    def __call__(self, x, out, counter, x_at=None, out_at=None, residual=None, residual_scale=None, scale_span=0):
        """RowLoopLinear.__call__'s operands."""
        from shared.kernels import row_loop_triton
        y = out if out_at is None else self.y
        if self.group is None:
            bm, bn, bk, warps, stages, grid_m, grid_n = self.params
            self.kernel[(grid_m, grid_n)](x, self.w, self.scale, y, self.rows, self.n, self.k, x.stride(0), x.stride(1), self.w.stride(0), self.w.stride(1), y.stride(0),
                                          y.stride(1), block_m=bm, block_n=bn, block_k=bk, **({} if x_at is None else dict(counter=counter, first=x_at[0], step=x_at[1], AT=True)),
                                          num_warps=warps, num_stages=stages)
            if self.bias is not None:
                y += self.bias
        else:
            quantize_config, gemm_config = self.configs
            for start, stop in self.parts:
                count = stop - start
                if x_at is None:
                    _rotated_triton._quantize(x[start:stop], self.q, self.scales, count, self.k, self.group, quantize_config)
                else:
                    _rotated_triton._quantize(x, self.q, self.scales, count, self.k, self.group, quantize_config, (counter, x_at[0] + start, x_at[1]))
                _rotated_triton._gemm(self.q, self.scales, self.w, self.scale, self.bias, y[start:stop], count, self.n, self.k, gemm_config)
        if out_at is not None:
            if residual is None:
                row_loop_triton.store_rows_at(out, y, counter, *out_at)
            else:
                row_loop_triton.residual_add_at(out, y, residual_scale, counter, *out_at, scale_span)


def _convrot_qtype():
    from shared.qtypes.int8_convrot import _QINT8_CONVROT_QTYPE
    return _QINT8_CONVROT_QTYPE


class RowLoopLinear:
    """One chunk of `rows` rows of a WGPKernels ConvRot linear, as _wgp_linear runs it inside chunked_rows() (same path, sub-chunks, launch
    configurations: the same bits), with the chunk's input and / or output rows addressed through a device counter: rows
    first + counter * step of x or out, else the fixed rows passed. For CUDA graph loops (mmgp.cuda_graphs.loop): every operand and scratch
    is made here, before the loop's body is recorded, and a call allocates nothing."""

    def __init__(self, module, x, rows, act=None):
        weight = module.qweight
        self.w, self.group, self.n, self.k = weight._data, _loop_group(module), weight._data.shape[0], weight._data.shape[1]
        self.scale = _weight_scale(weight, self.n, x.device)
        self.bias = None if module.bias is None else module.bias.to(x.dtype)
        self.dtype, self.device = x.dtype, x.device
        self.zero = torch.zeros(1, dtype=torch.int64, device=self.device)  # the counter of fixed operands
        sample = x[:rows]
        self.act = act
        self.path = _wgp.choose_path(sample, self.w, self.scale, self.bias, self.group, False, act)
        if self.path not in ("block", "exact", "row"):
            raise ValueError(f"row loops take the INT8 GEMM paths, not {self.path}")
        row_bytes = (2 * self.k if self.path == "exact" else self.k) + 4 * (-(-self.k // 128) if self.path == "block" else 1)
        sub = rows if rows <= max(128, _wgp.SCRATCH_BYTES // row_bytes // 128 * 128) else max(128, CHUNK_SCRATCH_BYTES // row_bytes // 128 * 128)
        self.parts = [(start, min(start + sub, rows)) for start in range(0, rows, sub)]
        width = min(sub, rows)
        self.q = torch.empty((width, self.k), dtype=torch.int8, device=self.device)
        self.lo = torch.empty_like(self.q) if self.path == "exact" else None
        self.sa = torch.empty((width, -(-self.k // 128)) if self.path == "block" else (width,), dtype=torch.float32, device=self.device)
        self.plans, self.slots = [], []
        for start, stop in self.parts:
            count = stop - start
            out = torch.empty((count, self.n), dtype=self.dtype, device=self.device)
            plan = _wgp._plan(self.q[:count], None if self.lo is None else self.lo[:count], self.sa[:count], self.w, self.scale, self.bias, out, self.path, False)
            self.plans.append(plan)
            self.slots.append(torch.empty(plan[2], dtype=torch.uint8, device=self.device) if plan[2] else None)

    def __call__(self, x, out, counter, x_at=None, out_at=None, residual=None, residual_scale=None, scale_span=0):
        """x: the input rows (fixed), or the whole input with x_at = (first, step); out: the output rows (fixed), or the whole output with
        out_at = (first, step), residual (out itself, or None) added with residual_scale (one vector, or scale_span rows per vector)."""
        lib, stream, sms = _wgp._lib, _wgp._stream(self.device), _wgp._sm_count(self.device)
        dtype, act = _wgp._DTYPES[self.dtype], _wgp.ACTS[self.act]
        for (start, stop), (variant, split, _, counter_bytes), slots in zip(self.parts, self.plans, self.slots):
            count = stop - start
            q, lo, sa = self.q.data_ptr(), _wgp._ptr(self.lo), self.sa.data_ptr()
            if x_at is None:
                source, first, step, at = x[start:stop].data_ptr(), 0, 0, self.zero
            else:
                source, first, step, at = x.data_ptr(), x_at[0] + start, x_at[1], counter
            if self.path == "block":
                _wgp._check(lib.wq8_quantize_blocks_at(source, q, sa, count, self.k, x.stride(0), dtype, self.group, act, None, 0.0, None, at.data_ptr(), first, step, stream))
            else:
                _wgp._check(lib.wq8_quantize_at(source, q, lo, sa, count, self.k, x.stride(0), dtype, self.group, act, None, 0.0, at.data_ptr(), first, step, stream))
            if out_at is None:
                target, first, step, at = out[start:stop].data_ptr(), 0, 0, self.zero
            else:
                target, first, step, at = out.data_ptr(), out_at[0] + start, out_at[1], counter
            counters = _wgp._zeroed_counters(self.device, counter_bytes, stream).data_ptr() if counter_bytes else None
            _wgp._check(lib.wq8_gemm_at(q, lo, sa, self.w.data_ptr(), self.scale.data_ptr(), _wgp._ptr(self.bias), None if residual is None else target,
                                        _wgp._ptr(residual_scale), target, count, self.n, self.k, dtype, variant, sms, split, _wgp._ptr(slots), counters,
                                        at.data_ptr(), first, step, scale_span, self.n if scale_span else 0, 0, 0, stream))


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


def _kitchen_gemm_takes(weight):
    return weight._data.shape[-1] % (16 if _kitchen_hip else 4) == 0 and weight._data.shape[0] % 4 == 0  # the kitchen GEMM rejects other widths


def _probe_rotated_triton():
    """The fused large-input ConvRot Triton kernels, which need tl.split / tl.join / tl.permute."""
    language = importlib.import_module("triton.language")
    return importlib.import_module("shared.kernels.convrot_int8_triton") if all(hasattr(language, name) for name in ("split", "join", "permute")) else None


def rotated_triton_supported(input, weight, group):
    """Whether a ConvRot linear runs on the fused Triton kernels (rotation inside the activation quantization, rows in bounded chunks, then a
    block-scaled INT8 GEMM) instead of a rotated copy of its whole input: on the Triton backend, or under Kitchen for the weights its GEMM
    rejects; BF16 / FP16 CUDA inputs of more than 32 rows (smaller ones keep their decode paths), K a multiple of max(group, 64)."""
    return (_rotated_triton is not None and (_backend == "triton" or not _kitchen_gemm_takes(weight)) and type(input) is torch.Tensor
            and input.is_cuda and input.dtype in (torch.bfloat16, torch.float16) and group in (16, 64, 256) and input.shape[-1] % max(group, 64) == 0
            and input.numel() > 32 * input.shape[-1] and weight._data.is_contiguous() and not torch.compiler.is_compiling())


def rotated_triton_linear(input, weight, bias, group):
    x = input.reshape(-1, input.shape[-1])
    x = x if x.stride(-1) == 1 else x.contiguous()
    n = weight._data.shape[0]
    return _rotated_triton.rotated_linear(x, weight._data, _weight_scale(weight, n, input.device), bias, group).reshape(*input.shape[:-1], n)


def _triton_linear(input, weight, bias=None, convrot=False):
    """Kitchen mode: Triton for the INT8 layers the Kitchen GEMM cannot take (output widths it rejects, ConvRot rows beyond its
    graph-safe CUTLASS width). Its tuning never runs while a CUDA graph is captured."""
    if convrot and _rotated_triton is not None and input.dtype in (torch.bfloat16, torch.float16):
        return rotated_triton_linear(input, weight, bias, 256)
    if convrot:
        from shared.qtypes.int8_convrot import _rotate_activation
        input = _rotate_activation(input, 256)
    n = weight._data.shape[0]
    out = _kitchen_triton.fused_quant_scaled_mm(input.reshape(-1, input.shape[-1]), weight._data, _weight_scale(weight, n, input.device),
                                                out_dtype=input.dtype)
    if bias is not None:
        out += bias
    return out.reshape(*input.shape[:-1], n)


def can_fuse_linear(module, x, dtypes=(torch.bfloat16,), act=None):
    """Keep qtype/backend details out of model code; MMGP still owns module calls.
    Callers widen `dtypes` only for activation dtypes validated with their model; act is the input operation they want fused
    ('rms_norm', 'swiglu', or 'gelu', which only WGPKernels provide).
    WGPKernels fuse with the precision of their unfused linears (full precision or per-block activations, input operations in FP32,
    one rounding), whatever the kernel precision setting; Comfy Kitchen's rowwise fusions need the fast (approximate) setting."""
    from mmgp.offload import lora_active
    if (torch.compiler.is_compiling() or not torch.is_inference_mode_enabled() or triton._is_fake_tensor(x) or not x.is_cuda
            or x.dtype not in dtypes or lora_active(module)):
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
    from mmgp.offload import lora_active
    group = getattr(modules[0], '_convrot_group_size', 0)
    common = (torch.is_inference_mode_enabled() and not torch.compiler.is_compiling() and not triton._is_fake_tensor(x) and x.is_cuda
              and x.ndim == 3 and x.is_contiguous() and len(modules) == x.shape[0]
              and all(getattr(module, '_convrot_group_size', 0) == group
                      and getattr(getattr(module, 'weight_qtype', None), 'name', None) == 'qint8_convrot'
                      and module.in_features == x.shape[2] and module.out_features == modules[0].out_features
                      and not getattr(module, '_wangp_linear_fusion', None)
                      and not getattr(module, '_wangp_prequantized_input', None)
                      and not lora_active(module) for module in modules))
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
    # Kitchen's residual epilogue always takes one scale per output channel: no scale (linear_residual_, an ungated mlp) is 1
    residual_arg = (_kitchen._gemm_vector_arg(input.new_ones(n) if residual_scale is None else residual_scale, input.device, input.dtype)
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
        return supplied  # the tensor itself (an output that is an argument stays that argument in CUDA graphs)
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
        if (_kitchen_triton is not None and not torch.compiler.is_compiling()
                and not triton._is_fake_tensor(input)):
            return _triton_linear(input, weight, bias, convrot=True)
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
            and weight._data.is_cuda and weight._data.dtype == torch.int8):
        if (input.shape[-1] <= 16384 or not _sm120_cutlass) and _kitchen_gemm_takes(weight):
            ctx.save_for_backward(input, weight)
            return kitchen_linear(input, weight, bias)
        if _kitchen_triton is not None and not torch.compiler.is_compiling() and not triton._is_fake_tensor(input):
            ctx.save_for_backward(input, weight)
            return _triton_linear(input, weight, bias)
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
        return _wgp.prepare(x, group, _wgp_path(module, x, group, precise))

    @staticmethod
    def prepare_in_place(module, x):
        """(prepare() of x into x's own memory, the bytes of x left free) (offload.prepare_linear_input, when x is the input's only
        holder), for the INT8 paths of one byte per element ('block', 'row'): the INT8 rows take the first half of x's bytes, the scales
        a tensor of their own. Rows
        are quantized by ranges into a bounded scratch, then copied over 16-bit rows already read: row r's INT8 bytes [r K, (r + 1) K)
        lie in the 16-bit rows below r / 2 + 1, so a range [a, b) is safe once b <= 2 a (the first one from its scratch alone). The
        same values as prepare(); None (x untouched) for another path."""
        _, group, precise = _WgpRowProjection.key(module)
        path = _wgp_path(module, x, group, precise)
        if path not in ("block", "row") or not x.is_contiguous() or x.element_size() != 2:
            return None
        (m, k), step = x.shape, max(128, _SCRATCH_BYTES // x.shape[1] // 128 * 128)
        memory = x.view(torch.uint8).view(-1)
        into = memory[:m * k].view(torch.int8).view(m, k)
        scale, start = None, 0
        while start < m:
            stop = min(m, start + (step if start == 0 else min(step, start)))
            part = _wgp.prepare(x[start:stop], group, path)
            if scale is None:
                scale = part.scale.new_empty((m, *part.scale.shape[1:]))
            into[start:stop].copy_(part.x)
            scale[start:stop].copy_(part.scale)
            start, part = stop, None
        return _wgp.Prepared(path, into, None, scale, x.dtype), memory[m * k:]

    @staticmethod
    def prepare_into(module, prepared, x, start, rows):
        """prepare() of rows [start, start + len(x)) of an input of `rows` rows (an offload.RowSource), into prepared, None for the first
        range: its path, chosen on that range (the rows of a source span several thousands: the block path where m decides), serves
        every range. Rows are prepared on their own: the same as preparing the whole input."""
        _, group, precise = _WgpRowProjection.key(module)
        part = _wgp.prepare(x, group, _wgp_path(module, x, group, precise) if prepared is None else prepared.path)
        if prepared is None:
            whole = lambda t: None if t is None else t.new_empty((rows, *t.shape[1:]))
            prepared = _wgp.Prepared(part.path, whole(part.x), whole(part.lo), whole(part.scale), part.dtype)
        target = prepared.rows(start, start + x.shape[0])
        for into, value in ((target.x, part.x), (target.lo, part.lo), (target.scale, part.scale)):
            if into is not None:
                into.copy_(value)
        return prepared

    @staticmethod
    def block_form(module, rows):
        """The 'block' form (ConvRot group, 128 columns per scale) a range of `rows` rows of this module's input is prepared in, whatever its
        values (an outside quantizer may then make it: offload.prepare_rows), None where the path is chosen on the input."""
        _, group, precise = _WgpRowProjection.key(module)
        return (group, 128) if not precise and rows > _wgp.DECIDE_MAX_ROWS else None

    @staticmethod
    def block_prepared(module, q, scales, dtype):
        """The prepared input of INT8 rows q and their scales (rows, K / 128), made in the 'block' form by an outside quantizer."""
        return _wgp.Prepared("block", q, None, scales, dtype)

    @staticmethod
    def prepare_into_at(module, prepared, x, counter, first, step):
        """prepare_into of rows first + counter * step of the input (a CUDA graph loop's turn, the counter on the device), prepared the
        first range's way: x prepared on its own, then stored at those rows (the copies prepare_into makes)."""
        from mmgp import offload
        part = _wgp.prepare(x, _WgpRowProjection.key(module)[1], prepared.path)
        for into, value in ((prepared.x, part.x), (prepared.lo, part.lo), (prepared.scale, part.scale)):
            if into is not None:
                offload._row_store_at(into, value, counter, first, step)

    @staticmethod
    def rows(module, prepared, start, stop):
        weight, bias = module.qweight, module.bias
        scale = _weight_scale(weight, weight._data.shape[0], weight._data.device)[start:stop]
        bias = None if bias is None else bias[start:stop].to(prepared.dtype)
        return _wgp.linear_prepared(prepared, weight._data[start:stop], scale, bias)

    @staticmethod
    def slice_input(prepared, start, stop):  # rows of the input (offload.linear_input_rows); the GEMM may round a few values otherwise for fewer rows
        return prepared.rows(start, stop)


_WGP_ROW_PROJECTION = _WgpRowProjection()
COLUMN_INPUT = True  # comparison switch: False keeps the 16-bit input of a linear fed by ranges of columns
COLUMN_QUANTIZE_IN_PLACE = True  # comparison switch: False quantizes each range with WGPKernels, then copies it into the kept input
COLUMN_ROWS = 16384  # rows per GEMM of a kept input: the kernels' top tuning bucket (one tile configuration from there on) and a bounded scratch output while they tune


class ColumnInput:
    """The input of an INT8 linear received by ranges of columns (an attention output computed by groups of heads) and kept as its
    kernels quantize it, one byte per element instead of the 16-bit input: the WGPKernels block path, or the Triton backend's
    ConvRot kernels. Rotation groups and activation scales cover at most `unit` columns, so every range is quantized as the whole
    input would be, and the layer's output has the precision of its own forward: the same results with the Triton kernels; with
    WGPKernels the GEMMs run on more rows per call (another tile configuration: a few results per million one unit in the last
    place away). Columns that end inside a unit wait for the next range. The rows of segments short enough for another path of the
    layer stay in 16 bits and go through the layer itself. LoRAs: mmgp LoraColumns, their input products summed range by range.
    ConvRot layers are quantized by the Triton kernel in both cases, straight into the kept input (for WGPKernels with its 128-column
    scales: one kernel per range instead of a quantization and two copies; it divides by the scale where WGPKernels multiply by its
    inverse, another rounding of the same precision)."""
    def __init__(self, module, rows, dtype, segments, loras, fused):
        self.module, self.rows, self.dtype, self.loras, self.fused = module, rows, dtype, loras, fused
        self.group, self.triton = getattr(module, "_convrot_group_size", 0), _backend == "triton"
        self.block = _rotated_triton.SCALE_BLOCK if self.triton else 128  # columns per activation scale
        self.quantizer = _rotated_triton if self.triton else _column_triton if COLUMN_QUANTIZE_IN_PLACE and self.group in (16, 64, 256) else None
        self.unit, self.done, self.carry, self.q, self.scale = max(self.block, self.group), 0, None, None, None
        self.parts = None  # [(first row, stop row, INT8 rows)]: the kept input in parts of memory (reuse), else one tensor (self.q)
        # short segments keep their 16-bit rows where the layer's own call would treat them apart: WGPKernels on its fused path (off it,
        # with a LoRA, the layer runs all its rows through the block path in one call), the Triton kernels' few-row paths
        short = segments if fused or self.triton else ()
        self.wide = {(start, stop): None for start, stop, _ in short or () if 0 < stop - start <= _column_short_rows()}

    def reuse(self, memory):
        """Keeps the INT8 input in memory (bytes, a 1-D uint8 tensor on the device, 16-byte aligned) instead of a tensor of its own:
        memory a caller has no more use for (the free half of an input prepared in place). Memory smaller than the input keeps its
        first rows (a multiple of 128), the others get a tensor of their own (an attention wider than the model: its kept input is
        larger than the free half). Returns whether it keeps any."""
        width = self.module.in_features
        fit = self.rows if memory is None or memory.numel() >= self.rows * width else memory.numel() // width // 128 * 128
        if self.q is not None or self.parts is not None or memory is None or not fit or memory.data_ptr() % 16:
            return False
        head = memory[:fit * width].view(torch.int8).view(fit, width)
        if fit == self.rows:
            self.q = head
        else:
            self.parts = [(0, fit, head), (fit, self.rows, None)]  # the tail's tensor is made with the scales, at the first range
        return True

    def _q_rows(self, first, last):
        """The kept INT8 rows [first, last): a view of the part holding them, or, for the rows of a GEMM that cross from one part to the
        next (one per pass at most), a copy of them (the GEMM reads the same rows as from one tensor: the same results)."""
        if self.parts is None:
            return self.q[first:last]
        pieces = [q[max(first, start) - start:min(last, stop) - start] for start, stop, q in self.parts if start < last and stop > first]
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces)

    def append(self, x_list):
        """The next columns, (..., columns) in 16 bits, handed off in x_list."""
        x = x_list[0]
        x_list.clear()
        x = x.reshape(self.rows, -1)
        if self.scale is None:
            if self.parts is not None:
                first, stop, _ = self.parts[1]
                self.parts[1] = (first, stop, torch.empty((stop - first, self.module.in_features), dtype=torch.int8, device=x.device))
            elif self.q is None:
                self.q = torch.empty((self.rows, self.module.in_features), dtype=torch.int8, device=x.device)
            self.scale = torch.empty((self.rows, self.module.in_features // self.block), dtype=torch.float32, device=x.device)
            self.wide = {(start, stop): x.new_empty((stop - start, self.module.in_features)) for start, stop in self.wide}
        if self.loras is not None:
            self.loras.append(x)
        if self.carry is not None:  # one quantization per range: fewer kernels than the waiting columns on their own
            x, self.carry = torch.cat((self.carry, x), dim=1), None
        ready = x.shape[1] // self.unit * self.unit
        if ready:
            first, last = self.done, self.done + ready
            for start, stop, kept in self.parts or [(0, self.rows, self.q)]:  # each part of the kept rows quantized on its own (rows are independent)
                if self.quantizer is not None:
                    self.quantizer.quantize_columns(x[start:stop, :ready], kept, self.scale[start:stop], first, self.group, self.block)
                else:
                    q, scale = _wgp.quantize_blocks(_wgp_aligned(x[start:stop, :ready]), self.group)
                    kept.view(torch.int64)[:, first // 8:last // 8].copy_(q.view(torch.int64))  # by 8 bytes: a third faster than the copy of single bytes
                    self.scale[start:stop, first // 128:last // 128].copy_(scale)
                    del q, scale
            for (start, stop), wide in self.wide.items():
                wide[:, first:last].copy_(x[start:stop, :ready])
            self.done = last
        if ready < x.shape[1]:
            self.carry = x[:, ready:].clone()

    def _spans(self, start, stop):
        """[start, stop) by equal parts of at most COLUMN_ROWS rows; the rows of a short segment together."""
        if (start, stop) in self.wide:
            return [(start, stop)]
        rows = -(-(stop - start) // -(-(stop - start) // COLUMN_ROWS) // 128) * 128
        return [(first, min(first + rows, stop)) for first in range(start, stop, rows)]

    def _rows(self, first, last, out=None, residual_scale=None, fused=False):
        """The layer's output for the rows [first, last) of one span, LoRAs included, written into out when given. fused (no LoRA): out
        holds the residual and becomes residual + residual_scale * output, rounded once."""
        module = self.module
        wide = self.wide.get((first, last))
        if wide is not None:
            if fused:
                return linear_with_fusion(module, wide, residual=out, residual_scale=residual_scale, out=out)
            return module(wide) if out is None else out.copy_(module(wide))
        weight = module.qweight
        scale = _weight_scale(weight, weight._data.shape[0], self.scale.device)
        if out is None:
            out = torch.empty((last - first, module.out_features), dtype=self.dtype, device=self.scale.device)
        q = self._q_rows(first, last)
        if self.triton:
            _rotated_triton.prepared_linear(q, self.scale[first:last], weight._data, scale, module.bias, out)
        else:
            bias = None if module.bias is None else module.bias.to(self.dtype)
            _wgp.linear_prepared(_wgp.Prepared("block", q, None, self.scale[first:last], self.dtype), weight._data, scale, bias,
                                 out if fused else None, residual_scale if fused else None, out)
        del q
        if self.loras is not None:
            self.loras.add_to(out, first, last)
        return out

    def linear(self, start=0, stop=None):
        """The layer's output for the rows [start, stop) (all by default), (stop - start, out_features)."""
        stop = self.rows if stop is None else stop
        out = torch.empty((stop - start, self.module.out_features), dtype=self.dtype, device=self.scale.device)
        for first, last in self._spans(start, stop):
            self._rows(first, last, out[first - start:last - start])
        return out

    def _add(self, start, stop, residual, scale):
        """residual[start:stop] += scale * output (None for 1, a number, or one value per channel in the residual's dtype), a part of
        rows at a time: off the fused path (a LoRA) each part's output is added as soon as it is computed, never the whole output
        beside the residual."""
        for first, last in self._spans(start, stop):
            target = residual[first:last]
            if self.fused:
                self._rows(first, last, target, scale, True)
            elif scale is None:
                target.add_(self._rows(first, last))
            elif isinstance(scale, (int, float)):
                target.add_(self._rows(first, last), alpha=scale)
            else:
                target.addcmul_(self._rows(first, last), scale)

    def linear_residual(self, residual, scale=None):
        """residual += scale * layer(input) (scale: None for 1, a number, or one value per output channel), as ops.linear_residual_."""
        rows = residual.view(-1, residual.shape[-1])
        if isinstance(scale, (int, float)) and self.fused:
            scale = torch.full((self.module.out_features,), scale, dtype=rows.dtype, device=rows.device)
        self._add(0, self.rows, rows, scale if scale is None or isinstance(scale, (int, float)) else scale.reshape(-1).to(rows.dtype))
        if self.fused:
            torch.autograd.graph.increment_version(residual)  # written in place: MMGP CUDA graphs copy a mutated block argument back
        return residual

    def gated_linear(self, residual, gate, segments):
        """residual[start:stop] += gate[row] * layer(input[start:stop]) for each segment (start, stop, row), as ops.gated_linear."""
        rows = residual.view(-1, residual.shape[-1])
        for start, stop, row in segments:
            if stop > start:
                self._add(start, stop, rows, gate[row].reshape(-1).to(rows.dtype))
        if self.fused:
            torch.autograd.graph.increment_version(residual)  # written in place: MMGP CUDA graphs copy a mutated block argument back
        return residual


def _column_short_rows():
    """Rows up to which the layer may take another path than the one a ColumnInput keeps its input for: the full precision paths of
    WGPKernels (tuned per shape), the few-row paths of the Triton backend's ConvRot layers (rotated_triton_supported)."""
    return 32 if _backend == "triton" else _wgp.DECIDE_MAX_ROWS


def column_input(module, rows, dtype, device, segments=None, residual=None):
    """A ColumnInput for module's input of `rows` rows in `dtype`, to be read through its linear(), linear_residual() or gated_linear()
    (residual: where the output will be added; segments: gated_linear's), or None when this layer cannot read its input by ranges of
    columns: a backend or weight format whose kernels read the 16-bit input or scale it over the whole row, full precision
    activations required, a LoRA that is not low rank, or too few rows on the kept path to save memory."""
    from mmgp import offload
    qtype = getattr(module, "weight_qtype", None)
    if (not COLUMN_INPUT or qtype is None or qtype.name not in ("qint8", "qint8_convrot") or torch.compiler.is_compiling()
            or device.type != "cuda" or dtype not in (torch.bfloat16, torch.float16) or getattr(module, "activation_qtype", None) is not None
            or hasattr(module, "_int8_residual_forward") or qtype.name == "qint8" and getattr(module, "_router_forward_impl", None) is not None):
        return None
    data, group = module.qweight._data, module._convrot_group_size if qtype.name == "qint8_convrot" else 0
    if _backend == "triton":  # its fused ConvRot kernels (rotated_triton_supported): a scale per row and 64 columns
        if _rotated_triton is None or group not in (16, 64, 256) or module.in_features % max(group, 64) or not data.is_contiguous():
            return None
    elif _backend != "wgpkernels" or group not in (0, 16, 64, 256) or module.in_features % max(128, group):
        return None
    if (not data.is_cuda or data.dtype != torch.int8
            or 4 * sum(stop - start for start, stop, _ in segments or () if stop - start <= _column_short_rows()) > rows):
        return None  # mostly short segments (one per frame): little to keep in INT8
    loras = None
    if offload.lora_active(module):
        if not offload.linear_columns_supported(module):
            return None
        loras = offload.LoraColumns(module)
    # the fused residual linear where the layer's own call would use it: otherwise its output, then the residual update
    fused = residual is not None and loras is None and can_fuse_linear(module, residual) and residual.dtype == dtype and residual.is_contiguous()
    return ColumnInput(module, rows, dtype, segments, loras, fused)


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
        group = getattr(module, "_convrot_group_size", 0)
        if (_backend == "triton" and x.is_cuda and not triton._prefer_native_quanto_path(x)):  # rotated chunk by chunk, never whole in 16 bits
            return triton._TRITON_MODULE.prepare_blockscale_input(x, module.out_features, (lambda part: _rotate_activation(part, group)) if group else None)
        return _rotate_activation(x, group)

    @staticmethod
    def block_form(module, rows):
        """With the Triton backend, the 'block' form (ConvRot group, the K tile of its GEMM: columns per scale) that prepare() gives `rows`
        rows of this module's input (prepare_blockscale_input), so that an outside quantizer may make it (offload.prepare_rows: the norm,
        modulation and quantization of shared.kernels.norm_quant_triton in one pass, the rotation in FP32 rather than 16 bits). None with
        PyTorch, rows the native path takes, or a group or tile that quantizer does not make."""
        group = getattr(module, "_convrot_group_size", 0)
        if (_backend != "triton" or triton._TRITON_MODULE is None or rows <= triton._NATIVE_FALLBACK_MAX_M or group not in (0, 16, 64, 256)
                or module.in_features % 256):
            return None
        block = triton._TRITON_MODULE._select_triton_int8_config(rows, module.in_features, module.out_features, device=module.qweight._data.device, kernel_kind="fused")[2]
        return (group, block) if block in (64, 128, 256) else None

    @staticmethod
    def block_prepared(module, q, scales, dtype):
        """prepare()'s form of INT8 rows q and their scales (rows, K / the K tile), made in the 'block' form by an outside quantizer."""
        m, k = q.shape
        return q, scales, dtype, triton._TRITON_MODULE._select_triton_int8_config(m, k, module.out_features, device=q.device, kernel_kind="fused")

    @staticmethod
    def slice_input(prepared, start, stop):  # rows of the input (offload.linear_input_rows): every row is computed on its own
        return (prepared[0][start:stop], prepared[1][start:stop], *prepared[2:]) if isinstance(prepared, tuple) else prepared[start:stop]

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
    def block_form(module, rows):
        """The 'block' form (ConvRot group, 0: one FP32 scale per row) of prepare(), so that an outside quantizer may make it
        (offload.prepare_rows: shared.kernels.norm_quant_triton, the norm, modulation, rotation and quantization in one pass, in FP32).
        The rows count only through supports(), which chose this projection."""
        convrot = module.weight_qtype.name == "qint8_convrot"
        return (256 if convrot else 0, 0)

    @staticmethod
    def block_prepared(module, q, scales, dtype):
        """prepare()'s form of INT8 rows q and their scales (rows, 1), made in the 'block' form by an outside quantizer."""
        return q, scales, dtype

    @staticmethod
    def slice_input(prepared, start, stop):  # rows of the input (offload.linear_input_rows)
        q, scales, dtype = prepared
        return q[start:stop], scales[start:stop], dtype

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
    global _backend, _kitchen, _kitchen_hip, _original_forward, _direct_cutlass, _sm120_cutlass, _kitchen_triton, _rotated_triton, revision
    global _blockwise_triton, _kitchen_dlpack_export, _wgp, _column_triton
    backend, module = resolve_backend(selection, kernel_policy.precision) if resolved is None else resolved
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
    _kitchen_triton = None
    _rotated_triton = _column_triton = None
    if backend == "triton":
        if not triton.maybe_enable_quanto_int8_kernel(verbose_level):
            raise RuntimeError("Failed to enable Triton INT8 kernels")
        _rotated_triton = _probe_rotated_triton()
    elif backend == "wgpkernels":
        from optimum.quanto.tensor.weights import qbytes
        _register_ops()
        _wgp = module
        _original_forward = qbytes.WeightQBytesLinearFunction.forward
        qbytes.WeightQBytesLinearFunction.forward = staticmethod(_wgp_quanto_forward)
        if torch.version.hip is None and torch.cuda.get_device_capability() >= (8, 0):  # the BF16 floor of the Triton kernels
            try:
                _column_triton = _probe_rotated_triton()
            except (ImportError, OSError, RuntimeError):
                _column_triton = None
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
        _kitchen_triton, _ = triton._probe_triton_backend()
        if _kitchen_triton is not None:
            _rotated_triton = _probe_rotated_triton()
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
