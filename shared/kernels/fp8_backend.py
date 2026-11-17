# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""FP8 math kernels of FP8 checkpoints (Quanto qfloat8 weights and shared.qtypes.scaled_fp8 weights), chosen in the configuration
(fp8_kernels):
- pytorch: each weight is dequantized whole (copies of its size in 16 bits) before a 16-bit matmul;
- triton: weight-only Triton matmul (fp8_w8a16_triton): FP8 weight tiles decoded in registers, no copy of the weight, same values as
  pytorch up to the order of the sums;
- cuda: FP8 tensor cores (torch._scaled_mm, cuBLASLt, RTX 40xx and later): the activations are also quantized to FP8 (one scale per
  call), fastest, lower precision;
- auto: the fastest available (cuda, then triton, then pytorch), as WanGP's FP8 checkpoints used FP8 tensor cores before.
Block-scaled FP8 weights (shared.qtypes.block_scaled: one scale per 128 x 128 block, or per output channel and 128-element block of K)
run on WGPKernels' FP8 GEMMs with E4M3 activations (quantized per row and 128 columns, approximate) under the cuda choice, else with
16-bit activations (exact, the weights decoded in registers): WGPKernels' W8A16 kernels when its library has them, else the Triton block
kernel (shared.kernels.fp8_block_triton), on every GPU including those without FP8 tensor cores; without a precise kernel, the E4M3
kernels when the precision setting allows approximate kernels; the pytorch choice dequantizes them per call.
A choice this GPU cannot run gives way to the next one (cuda -> triton -> pytorch), and the kernels in use are printed."""
import torch
from torch._subclasses.fake_tensor import is_fake

CHOICES = [("Auto (fastest available)", "auto"), ("PyTorch (slowest, precise)", "pytorch"), ("Triton (fast, precise)", "triton"),
           ("CUDA FP8 (fastest, least precise)", "cuda")]
_LABELS = {"pytorch": "PyTorch", "triton": "Triton", "cuda": "CUDA FP8 Tensor Cores"}
FP8_DTYPES = (torch.float8_e4m3fn, torch.float8_e5m2)
backend = "pytorch"  # kernels of the Quanto qfloat8 weights
scaled_backend = "pytorch"  # kernels of the scaled FP8 weights (shared.qtypes.scaled_fp8)
revision = 0  # part of the CUDA graph keys: graphs record the kernels of the backend
_ops_registered = False
_cuda_skips_noted = False
_wgp_fp8 = None  # wgpkernels.fp8 while a kernel choice is active and its library runs on this GPU (block-scaled FP8 weights)
_wgp_a16 = False  # that library computes 16-bit activations (W8A16)
_triton_block = False  # shared.kernels.fp8_block_triton imports
_selection = "auto"
BLOCK_LABELS = {"wgp_w8a8": "WGPKernels (E4M3 activations)", "wgp_a16": "WGPKernels (16-bit activations)", "triton_a16": "Triton (16-bit activations)"}


def _available(selection):
    """(kernels this GPU runs for a pytorch / triton / cuda selection, reason when they differ)."""
    reason = None
    if selection == "cuda":
        from shared.qtypes import scaled_fp8
        scaled_fp8._init_scaled_mm_support()
        if scaled_fp8._scaled_mm_available(torch.float8_e4m3fn):
            return "cuda", None
        selection, reason = "triton", "FP8 tensor cores need an RTX 40xx or later GPU"
    if selection == "triton":
        try:
            from shared.kernels import fp8_w8a16_triton  # noqa: F401
            return "triton", reason
        except ImportError as exc:
            return "pytorch", f"Triton is unavailable ({exc})"
    return "pytorch", reason


def resolve_backend(selection):
    """((kernels of the Quanto qfloat8 weights, kernels of the scaled FP8 weights), reason when they differ from the selection)."""
    if selection not in {value for _, value in CHOICES}:
        raise ValueError(f"Unknown FP8 kernel backend: {selection!r}")
    resolved, reason = _available("cuda" if selection == "auto" else selection)
    return (resolved, resolved), reason


def _probe_wgp_fp8():
    """wgpkernels.fp8 when installed with its library and its selftest passes on this GPU, else None."""
    import importlib
    try:
        module = importlib.import_module("wgpkernels.fp8")
        if not module.default_library_path().is_file():
            return None
        if module._lib is None:
            module.load_library()
        with torch.inference_mode():
            return None if module.selftest() else module
    except Exception:
        return None


def configure(selection, resolved=None):
    global backend, scaled_backend, revision, _wgp_fp8, _wgp_a16, _triton_block, _selection
    (quanto, scaled), reason = resolve_backend(selection) if resolved is None else resolved
    if (quanto, scaled, selection) != (backend, scaled_backend, _selection):
        revision += 1
    backend, scaled_backend, _selection = quanto, scaled, selection
    _wgp_fp8 = _probe_wgp_fp8() if scaled != "pytorch" else None
    _wgp_a16 = _wgp_fp8 is not None and getattr(_wgp_fp8, "a16_supported", lambda: False)()
    _triton_block = scaled != "pytorch" and _probe_triton_block()
    if (quanto, scaled) != ("pytorch", "pytorch"):
        _register_ops()
    from mmgp import offload
    offload.register_row_projection(_FP8_ROW_PROJECTION)
    kernels = _LABELS[quanto] if quanto == scaled else f"{_LABELS[quanto]}, {_LABELS[scaled]} for scaled FP8 checkpoints"
    block = _block_route()
    print(f"[FP8] Backend: {kernels}{'; block-scaled weights: ' + BLOCK_LABELS[block] if block else ''} (setting: {selection}{'; ' + reason if reason else ''}).")
    return quanto, scaled


def _probe_triton_block():
    try:
        from shared.kernels import fp8_block_triton  # noqa: F401
        return True
    except Exception:
        return False


def _block_route():
    """Kernels of the block-scaled FP8 weights for the active choice: 'wgp_w8a8' (WGPKernels, E4M3 activations) under the cuda choice,
    else the exact 16-bit-activation kernels 'wgp_a16' (WGPKernels) or 'triton_a16', else 'wgp_w8a8' when approximate kernels are allowed,
    else None (dequantized per call)."""
    if scaled_backend == "pytorch":
        return None
    if _wgp_fp8 is not None and (scaled_backend == "cuda" or _selection == "cuda"):
        return "wgp_w8a8"
    if _wgp_a16:
        return "wgp_a16"
    if _triton_block:
        return "triton_a16"
    from shared.kernels import kernel_policy
    return "wgp_w8a8" if _wgp_fp8 is not None and kernel_policy.allow_approximate() else None


def _cuda_runs(weight, x):
    from shared.qtypes import scaled_fp8
    return scaled_fp8._scaled_mm_available(weight.dtype) and weight.shape[0] % 16 == 0 and weight.shape[1] % 16 == 0 and weight.is_contiguous()


def _cuda_linear(x, weight, scale, bias):
    """FP8 tensor cores: x quantized to FP8 with one scale, the weight scale applied by cuBLASLt (one value) or to the output."""
    from shared.qtypes import scaled_fp8
    scale_b, output_scale = scaled_fp8._scaled_mm_weight_scale(scale.reshape(-1) if scale.numel() > 1 else scale, weight)
    out = scaled_fp8._scaled_mm_rows(x.contiguous(), weight.t(), scale_b.to(torch.float32), bias if output_scale is None else None, x.dtype)
    if output_scale is not None:
        out.mul_(output_scale.to(out.dtype).view(1, -1))
        if bias is not None:
            out.add_(bias)
    return out


def _linear_impl(x, weight, scale, bias, cuda):
    global _cuda_skips_noted
    if cuda:
        if _cuda_runs(weight, x):
            return _cuda_linear(x, weight, scale, bias)
        if not _cuda_skips_noted:
            _cuda_skips_noted = True
            print(f"[FP8] CUDA FP8 kernels cannot compute some layers (e.g. {weight.dtype}, {tuple(weight.shape)}): Triton computes them.")
    from shared.kernels import fp8_w8a16_triton
    return fp8_w8a16_triton.linear(x, weight, scale, bias, x.new_empty((x.shape[0], weight.shape[0])))


def _register_ops():
    global _ops_registered
    if _ops_registered:
        return

    @torch.library.custom_op("wan2gp_fp8::linear", mutates_args=())
    def linear_op(x: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor | None, cuda: bool) -> torch.Tensor:
        return _linear_impl(x, weight, scale, bias, cuda)

    @linear_op.register_fake
    def fake(x, weight, scale, bias, cuda):
        return x.new_empty((x.shape[0], weight.shape[0]))

    _ops_registered = True


def matmul(x, weight, scale, bias, cuda):
    """x (M, K) @ (weight (N, K) float8 * scale (1 or N values, in x's dtype))^T + bias with the selected kernels (FP8 tensor cores
    when cuda, else the Triton weight-only kernel), in x's dtype."""
    if torch.compiler.is_compiling() or is_fake(x):  # tracing (Dynamo also runs tensor methods on fake tensors): the opaque op
        return torch.ops.wan2gp_fp8.linear(x, weight, scale, bias, cuda)
    return _linear_impl(x, weight, scale, bias, cuda)


def block_mode(input, weight):
    """The kernels (a BLOCK_LABELS key) computing a block-scaled FP8 weight for this input (BF16/FP16 CUDA activations, K a multiple of 16),
    or None for the per-call dequantization."""
    if not (input.is_cuda and input.dtype in (torch.bfloat16, torch.float16) and weight.is_cuda and input.shape[-1] % 16 == 0):
        return None
    return _block_route()


def block_linear(input, weight, scale, bias=None, mode="wgp_w8a8"):
    """input @ (weight E4M3 * scale per 128 x 128 block, or per channel and 128-element block of K).T + bias with the kernels of mode
    (block_mode()): WGPKernels with E4M3 activations ('wgp_w8a8') or 16-bit ones ('wgp_a16'), or the Triton block kernel ('triton_a16')."""
    x = input.reshape(-1, input.shape[-1])
    if x.stride(-1) != 1 or x.stride(0) % 8 or x.data_ptr() % 16:  # rows loaded by 16 bytes
        x = x.contiguous()
    if bias is not None and bias.dtype != x.dtype:
        bias = bias.to(x.dtype)
    if mode == "triton_a16":
        from shared.kernels import fp8_block_triton
        out = fp8_block_triton.linear(x, weight, scale, bias)
    else:
        out = _wgp_fp8.linear(x, weight, scale, bias, precise=mode == "wgp_a16") if _wgp_a16 else _wgp_fp8.linear(x, weight, scale, bias)
    return out.reshape(*input.shape[:-1], weight.shape[0])


def linear(input, weight, bias=None):
    """input @ weight^T + bias for a Quanto qfloat8 weight with the selected kernels, computed as Quanto does in the dtype of the
    weight's scale."""
    data, scale = weight._data, weight._scale
    x = input.reshape(-1, input.shape[-1])
    if x.dtype != scale.dtype:
        x = x.to(scale.dtype)
    fused_bias = bias if bias is not None and bias.dtype == scale.dtype else None
    out = matmul(x, data, scale, fused_bias, backend == "cuda").reshape(*input.shape[:-1], data.shape[0])
    return out + bias if bias is not None and fused_bias is None else out


class _FP8RowProjection:
    @staticmethod
    def supports(module, x):
        qtype = getattr(module, "weight_qtype", None)
        if qtype is None or qtype.name not in ("qfloat8_e4m3fn", "qfloat8_e4m3fnuz", "qfloat8_e5m2", "scaled_float8_e4m3fn", "scaled_float8_e5m2"):
            return False
        if is_fake(x) or torch.compiler.is_compiling() or getattr(module, "activation_qtype", None) is not None:
            return False
        weight = module.qweight
        return (weight.dtype == x.dtype and weight._scale.numel() in (1, weight.shape[0])
                and (module.bias is None or module.bias.dtype == x.dtype))

    @staticmethod
    def key(module):
        from shared.qtypes.scaled_fp8 import ScaledFP8WeightTensor
        weight = module.qweight
        scaled = isinstance(weight, ScaledFP8WeightTensor)
        mode = scaled_backend if scaled else backend
        if not weight.is_cuda:
            mode = "pytorch"
        elif mode == "cuda" and ((scaled and weight._force_full_precision_mm) or not _cuda_runs(weight._data, weight._data)):
            mode = "triton"
        return "fp8_rows", mode, weight.dtype, weight._data.dtype

    @staticmethod
    def prepare(module, x):
        from shared.qtypes.scaled_fp8 import _quantize_activation
        _, mode, dtype, fp8_dtype = _FP8RowProjection.key(module)
        x = x.to(dtype).contiguous()
        return (*_quantize_activation(x, fp8_dtype), dtype) if mode == "cuda" else x

    @staticmethod
    def rows(module, prepared, start, stop):
        from shared.qtypes.scaled_fp8 import _scaled_mm_weight_scale
        weight = module.qweight
        data = weight._data[start:stop]
        scale = weight._scale if weight._scale.numel() == 1 else weight._scale.reshape(-1)[start:stop]
        bias = None if module.bias is None else module.bias[start:stop]
        mode = _FP8RowProjection.key(module)[1]
        if mode == "cuda":
            x, xs, dtype = prepared
            count = stop - start
            ws, output_scale = _scaled_mm_weight_scale(scale, data)
            # cuBLASLt requires an aligned N; pad only the requested row range.
            if count % 16:
                padded = data.new_zeros(((count + 15) // 16 * 16, data.shape[1]))
                padded[:count].copy_(data)
                data = padded
            fused_bias = bias if output_scale is None and dtype != torch.float32 and count % 16 == 0 else None
            out = torch._scaled_mm(x, data.t(), xs, ws.to(torch.float32), bias=fused_bias, out_dtype=dtype)[:, :count]
            if output_scale is not None:
                out.mul_(output_scale.to(dtype).view(1, -1))
            if bias is not None and fused_bias is None:
                out.add_(bias)
            return out
        if mode == "triton":
            return matmul(prepared, data, scale.to(prepared.dtype), bias, False)
        dense = data.to(prepared.dtype) * scale.to(prepared.dtype).reshape(-1, 1)
        out = prepared @ dense.t()
        if bias is not None:
            out.add_(bias)
        return out

    @staticmethod
    def slice_input(prepared, start, stop):  # rows of the input (offload.linear_input_rows): one activation scale for the whole input
        return (prepared[0][start:stop], *prepared[1:]) if isinstance(prepared, tuple) else prepared[start:stop]


_FP8_ROW_PROJECTION = _FP8RowProjection()
