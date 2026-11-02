# Copyright (c) 2026 DeepBeepMeep - WanGP Community License 2.0 (see LICENSE.txt)
"""Ahead-of-time CPU kernels of routed expert banks (experts.cpp: SwiGLU or squared ReLU), used by the expert formats of WanGP's qtypes.

Weights stay in their mmap storage: only the selected experts are read, and no dequantized copy is kept. Importing or using this module
never invokes a compiler: python -m shared.kernels.cpu_experts.build rebuilds the libraries."""
import ctypes
import os
from pathlib import Path
import platform

import torch

from mmgp.expert_compute import ExpertFormat

FP8_BLOCK, INT4_GROUP, INT8_ROWS = 0, 1, 2  # experts.cpp storage formats
SWIGLU, RELU2 = 0, 1  # experts.cpp activations
ACTIVATIONS = {"swiglu": SWIGLU, "relu2": RELU2}

_library = None
_attempted = False


def library_path():
    return Path(__file__).with_name("cpu_experts_" + ("win_amd64.dll" if os.name == "nt" else "linux_x86_64.so"))


def library():
    global _library, _attempted
    if not _attempted:
        _attempted = True
        path = library_path()
        if platform.machine().lower() in ("amd64", "x86_64") and path.is_file():
            lib = ctypes.CDLL(str(path))
            lib.moe_supported.restype = ctypes.c_int
            if lib.moe_supported():
                lib.moe_forward.argtypes = [ctypes.c_int] * 3 + [ctypes.c_void_p] * 9 + [ctypes.c_int] * 8
                lib.moe_forward.restype = ctypes.c_int
                lib.moe_host_job.argtypes = [ctypes.c_void_p]
                lib.moe_host_job.restype = None
                _library = lib
        if _library is None:
            raise RuntimeError("The CPU expert kernels require an x86-64 CPU with AVX2/FMA/F16C and their library (python -m shared.kernels.cpu_experts.build).")
    if _library is None:
        raise RuntimeError("The CPU expert kernels are unavailable")
    return _library


class Job(ctypes.Structure):
    """One routed call run by a CUDA host function (experts.cpp MoeJob): the addresses stay fixed while graphs replay it."""
    _fields_ = [(name, ctypes.c_void_p) for name in ("w13", "w2", "s13", "s2", "payload", "resident", "out")] + \
               [(name, ctypes.c_int32) for name in ("format", "rotation", "act", "batch", "h", "i", "e", "topk", "block", "block2", "threads", "status")]


def host_function():
    """Address of the host function running a Job."""
    return ctypes.cast(library().moe_host_job, ctypes.c_void_p).value


def run(format, rotation, act, w13, w2, s13, s2, x, ids, probabilities, resident, out, batch, hidden, intermediate, experts, topk, block, block2, threads):
    """Addresses of contiguous CPU buffers: weights and scales of the format, float32 x/probabilities/out, int32 ids and a uint8
    resident flag per expert (or 0). Pairs whose id is -1 or resident are skipped. Returns how many pairs were computed; out is left
    unwritten when none were."""
    count = library().moe_forward(format, rotation, act, w13, w2, s13, s2, x, ids, probabilities, resident, out, batch, hidden, intermediate, experts, topk, block, block2, max(1, threads))
    if count < 0:
        raise RuntimeError(f"CPU expert kernel failed ({count})")
    return count


class NativeExpertFormat(ExpertFormat):
    """An expert format computed by this library on the CPU and by shared.kernels.expert_triton on the GPU. Subclasses declare the
    tensors (input projection weights, down weights, input projection scales, down scales, in this order), the native format id
    (FP8_BLOCK, INT4_GROUP, INT8_ROWS) and the default scale block, and dequantize_matrix() for the PyTorch path. A bank's scale blocks
    follow from the widths of its scale tensors."""
    native = block = None

    def blocks(self, hidden, intermediate, tensors):
        s13, s2 = list(tensors.values())[2:]
        return hidden // s13.shape[-1], intermediate // s2.shape[-1]

    def _addresses(self, bank):
        return [getattr(bank, name).data_ptr() for name in bank.expert_tensor_names]

    def cpu_job(self, bank, payload, resident, out, tokens, topk, threads):
        w13, w2, s13, s2 = self._addresses(bank)
        job = Job(w13=w13, w2=w2, s13=s13, s2=s2, payload=payload, resident=resident, out=out, format=self.native, rotation=bank.rotation, act=ACTIVATIONS[bank.activation],
                  batch=tokens, h=bank.hidden, i=bank.intermediate, e=bank.num_experts, topk=topk, block=bank.blocks[0], block2=bank.blocks[1], threads=threads, status=0)
        return job, host_function()

    def cpu_forward(self, bank, x, ids, probabilities, resident, out, tokens, topk, threads):
        return run(self.native, bank.rotation, ACTIVATIONS[bank.activation], *self._addresses(bank), x, ids, probabilities, resident, out, tokens, bank.hidden,
                   bank.intermediate, bank.num_experts, topk, *bank.blocks, threads)

    def gpu_forward(self, bank, x, ids, probabilities, pointers, out=None, mid=None):
        from shared.kernels import expert_triton
        return expert_triton.forward(self.name, x, ids, probabilities, pointers, bank.intermediate, bank.blocks, out=out, mid=mid, rotation=bank.rotation, gated=bank.gated)

    def dequantize(self, bank, tensors):
        weights = [self.dequantize_matrix(tensors[weight], tensors[scale], block)
                   for weight, scale, block in zip(bank.expert_tensor_names[:2], bank.expert_tensor_names[2:], bank.blocks)]
        if bank.rotation:
            # ConvRot weights were quantized as W @ H.T by groups of input columns: W = W' @ H (H is symmetric and orthogonal)
            from shared.qtypes.int8_convrot import _regular_hadamard
            h = _regular_hadamard(bank.rotation, weights[0].device, torch.float32)
            weights = [torch.matmul(w.view(w.shape[0], -1, bank.rotation), h).view(w.shape) for w in weights]
        return weights

    def dequantize_matrix(self, weight, scale, block):
        raise NotImplementedError
