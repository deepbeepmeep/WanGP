# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Common eligibility and family ordering for activation kernels.

Dispatch never caches tensors, modules, bound methods or shape-specific choices.
An operation supplies only the implementations that satisfy its arithmetic and
layout contract; the existing user policy orders those implementations.

Device eligibility lives here, once: a WanGP-owned kernel runs on every NVIDIA GPU
whose instructions its arithmetic needs (DTYPE_FLOORS), not only where it was first
timed. Triton compiles each kernel for the running GPU; the kernels carry no
architecture-specific code, and their precision against PyTorch was measured per
layer and end to end on compute capabilities 8.9 and 12.0 (specs:
H3_FL2VA_ACCELERATION_AUDIT_20261021*.md, H3_VAE_SPEED_20261009.md). An operation
whose kernel needs more than its dtype floor says so (`capability=`); one validated
on HIP admits it (`hip=True`); PACKED_FMA_CAPABILITY marks the one architecture whose
compiler changes a bit-exact kernel's rounding. Performance-only restrictions (a kernel
that runs everywhere but only beats cuDNN on RTX 50) stay with the kernel that measured
them, built on `capability()`.
"""
import functools
import torch
from tqdm import tqdm
from mmgp import cuda_graphs
from . import kernel_policy

_announced = set()

# NVIDIA compute capability from which a kernel computing in this dtype runs: BF16 conversions are native instructions from Ampere
# (Triton emulates nothing below), FP16 and FP32 from Turing, the oldest architecture WanGP supports.
DTYPE_FLOORS = {torch.bfloat16: (8, 0), torch.float16: (7, 5), torch.float32: (7, 5)}
# GeForce Blackwell: ptxas lowers a packed f32x2 multiply followed by an add to one FMA even with enable_fp_fusion=False, so an
# elementwise kernel written to reproduce PyTorch's separate roundings rounds once less there (specs/3D_OPT_ROUND3_20261009.md); on
# every other architecture it is bit-identical (H3 norm-modulate: H3_FL2VA_ACCELERATION_AUDIT_20261021_DENOISING.md).
PACKED_FMA_CAPABILITY = (12, 0)


@functools.cache
def capability(device_index):
    """(major, minor) of an NVIDIA CUDA device, None on HIP. Device capabilities are immutable; cache metadata, never activations."""
    if torch.version.hip is not None:
        return None
    return torch.cuda.get_device_capability(torch.cuda.current_device() if device_index is None else device_index)


def device_supported(device, dtype, *, capability_floor=None, hip=False):
    """Whether a WanGP kernel computing in dtype runs on device: NVIDIA GPUs from the dtype's floor, or from the kernel's own
    `capability_floor` when it states one (its arithmetic, e.g. a BF16 kernel reading bits on Turing); HIP devices only when the
    operation admits them."""
    if device.type != "cuda":
        return False
    found = capability(device.index)
    if found is None:
        return hip
    floor = DTYPE_FLOORS.get(dtype, (8, 0)) if capability_floor is None else capability_floor
    return found >= floor


@functools.cache
def triton_dtype_supported(device, dtype):
    """BF16 arithmetic needs native conversion instructions on NVIDIA GPUs (HIP: Triton's own conversions). Self-contained, so that
    clearing its cache alone re-reads the device (the LM engine and FlashVSR tests patch the capability around it)."""
    return dtype != torch.bfloat16 or torch.version.hip is not None or torch.cuda.get_device_capability(device) >= DTYPE_FLOORS[torch.bfloat16]


@functools.cache
def packed_fma(device_index):
    """Whether the compiler of this device fuses a packed multiply and add despite enable_fp_fusion=False (PACKED_FMA_CAPABILITY): a
    kernel bit-identical to PyTorch elsewhere is one rounding apart here, so it counts as approximate on this device only."""
    found = capability(device_index)
    return found is not None and found[0] == PACKED_FMA_CAPABILITY[0]


def take(x_list):
    """Consume an owned activation. Callers must drop their other references.

    Residuals, parameters, rotary tables and caches are borrowed explicitly, not
    put in this holder. In-place APIs ending in '_' borrow their destination.
    """
    if not isinstance(x_list, list) or len(x_list) != 1:
        raise ValueError("Expected a one-element activation holder")
    return x_list.pop()


def choose(operation=None, *, wgpkernels=False, kitchen=False, triton=False, owned_triton=False):
    # WanGP-owned Triton implementations occupy the WanGP slot unless a native
    # implementation exists. Disabled concerns INT8 linears, not these fusions.
    family = kernel_policy.first({"wgpkernels": wgpkernels or owned_triton,
                                  "kitchen": kitchen, "triton": triton or owned_triton})
    if operation is not None and not torch.compiler.is_compiling():
        implementation = "WanGP Triton" if family == "wgpkernels" and not wgpkernels else family or "pytorch"
        key = operation, implementation, kernel_policy.precision
        if key not in _announced:
            _announced.add(key)
            tqdm.write(f"[Kernels] {operation}: {implementation} ({kernel_policy.precision}).")
    return family


def cuda_inference(x, *, dtypes=(torch.bfloat16,), approximate=False, capability_floor=None, hip=False):
    """Whether a WanGP kernel may replace the PyTorch path for x: no autograd (inference mode, or grad disabled as wgp.py runs every
    generation: the kernels never touch autograd, and pipelines under no_grad alone, such as Flux, Qwen Image and Wan, were running
    the PyTorch paths while their blocks measured the kernels under inference mode), no compilation, a CUDA tensor of one of dtypes
    on a device the kernel supports (device_supported), and the user's precision policy when the kernel is approximate (a bool,
    possibly decided per device: see packed_fma)."""
    if (torch.compiler.is_compiling() or torch.is_grad_enabled()
            or type(x) is not torch.Tensor or not x.is_cuda or x.dtype not in dtypes
            or (approximate and not kernel_policy.allow_approximate())):
        return False
    return device_supported(x.device, x.dtype, capability_floor=capability_floor, hip=hip)


# A preferred family can change while the resolved INT8 backend stays the same.
# Include the preference itself, including for standalone managed components.
cuda_graphs.register_key(lambda: (kernel_policy.precision, kernel_policy.families, kernel_policy.sage2_execution_quality()))
cuda_graphs.register_eager(kernel_policy.first_step_precise)  # one use: no graph for that step
