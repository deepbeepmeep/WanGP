# Copyright (c) 2026 DeepBeepMeep - WanGP Community License 2.0 (see LICENSE.txt)
"""Ahead-of-time CPU kernels of routed expert banks (experts.cpp: SwiGLU, SwiGLU with soft caps or squared ReLU), used by the expert formats of
WanGP's qtypes.

Weights stay in their mmap storage: only the selected experts are read, and no dequantized copy is kept. Importing or using this module
never invokes a compiler: python -m shared.kernels.cpu_experts.build rebuilds the libraries."""
import ctypes
import os
from pathlib import Path
import platform

import torch

from mmgp.expert_compute import ExpertFormat

FP8_BLOCK, INT4_GROUP, INT8_ROWS, NVFP4_ROWS, MXFP8_ROWS = 0, 1, 2, 3, 4  # experts.cpp storage formats
SWIGLU, RELU2, SWIGLU_SOFTCAP = 0, 1, 2  # experts.cpp activations
ACTIVATIONS = {"swiglu": SWIGLU, "relu2": RELU2, "swiglu_softcap": SWIGLU_SOFTCAP}

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
                lib.moe_forward.argtypes = [ctypes.c_int] * 3 + [ctypes.c_void_p] * 9 + [ctypes.c_int] * 8 + [ctypes.c_void_p] * 2 + [ctypes.c_float] * 2
                lib.moe_forward.restype = ctypes.c_int
                lib.moe_host_job.argtypes = [ctypes.c_void_p]
                lib.moe_host_job.restype = None
                lib.moe_server_job.argtypes = [ctypes.c_void_p]
                lib.moe_server_job.restype = ctypes.c_int
                _library = lib
        if _library is None:
            raise RuntimeError("The CPU expert kernels require an x86-64 CPU with AVX2/FMA/F16C and their library (python -m shared.kernels.cpu_experts.build).")
    if _library is None:
        raise RuntimeError("The CPU expert kernels are unavailable")
    return _library


class Job(ctypes.Structure):
    """One routed call run by a CUDA host function (experts.cpp MoeJob): the addresses stay fixed while graphs replay it."""
    _fields_ = [(name, ctypes.c_void_p) for name in ("w13", "w2", "s13", "s2", "payload", "resident", "usage", "out")] + \
               [(name, ctypes.c_int32) for name in ("format", "rotation", "act", "batch", "h", "i", "e", "topk", "block", "block2", "threads", "status")] + \
               [(name, ctypes.c_void_p) for name in ("f13", "f2")] + [(name, ctypes.c_float) for name in ("cap_gate", "cap_up")]


def host_function():
    """Address of the host function running a Job."""
    return ctypes.cast(library().moe_host_job, ctypes.c_void_p).value


def server_function():
    """The job function of the host server (shared.kernels.host_doorbell): returns whether it wrote a result."""
    return ctypes.cast(library().moe_server_job, ctypes.c_void_p).value


def run(format, rotation, act, w13, w2, s13, s2, x, ids, probabilities, resident, out, batch, hidden, intermediate, experts, topk, block, block2, threads, f13=0, f2=0, caps=(0.0, 0.0)):
    """Addresses of contiguous CPU buffers: weights and scales of the format (f13, f2: its float32 factors per expert, if any), float32
    x/probabilities/out, int32 ids and a uint8 resident flag per expert (or 0). caps: the soft caps of SWIGLU_SOFTCAP. Pairs whose id is
    -1 or resident are skipped. Returns how many pairs were computed; out is left unwritten when none were."""
    count = library().moe_forward(format, rotation, act, w13, w2, s13, s2, x, ids, probabilities, resident, out, batch, hidden, intermediate, experts, topk, block, block2, max(1, threads), f13, f2, *caps)
    if count < 0:
        raise RuntimeError(f"CPU expert kernel failed ({count})")
    return count


class NativeExpertFormat(ExpertFormat):
    """An expert format computed by this library on the CPU and by shared.kernels.expert_triton on the GPU. Subclasses declare the
    tensors (input projection weights, down weights, input projection scales, down scales, in this order, then the float32 factors per
    expert of both projections for the formats that have them), the native format id (FP8_BLOCK, INT4_GROUP, INT8_ROWS) and the default
    scale block, and dequantize_matrix() for the PyTorch path. A bank's scale blocks follow from the widths of its scale tensors."""
    native = block = None
    reports_usage = True

    def blocks(self, hidden, intermediate, tensors):
        s13, s2 = list(tensors.values())[2:4]
        return hidden // s13.shape[-1], intermediate // s2.shape[-1]

    def _addresses(self, bank):
        """w13, w2, s13, s2, f13, f2 (0 without factors)."""
        return ([getattr(bank, name).data_ptr() for name in bank.expert_tensor_names] + [0, 0])[:6]

    def cpu_job(self, bank, payload, resident, out, tokens, topk, threads, share=None, share_rows=0):
        w13, w2, s13, s2, f13, f2 = self._addresses(bank)  # share: always None (cpu_share() gives none)
        cap_gate, cap_up = bank.caps or (0.0, 0.0)
        job = Job(w13=w13, w2=w2, s13=s13, s2=s2, payload=payload, resident=resident, usage=bank.usage.data_ptr(), out=out, format=self.native, rotation=bank.rotation, act=ACTIVATIONS[bank.activation],
                  batch=tokens, h=bank.hidden, i=bank.intermediate, e=bank.num_experts, topk=topk, block=bank.blocks[0], block2=bank.blocks[1], threads=threads, status=0,
                  f13=f13, f2=f2, cap_gate=cap_gate, cap_up=cap_up)
        return job, host_function()

    def doorbell(self, device):
        from shared.kernels import host_doorbell
        return (host_doorbell, server_function()) if host_doorbell.supported(device) else None

    def cpu_forward(self, bank, x, ids, probabilities, resident, out, tokens, topk, threads):
        w13, w2, s13, s2, f13, f2 = self._addresses(bank)
        return run(self.native, bank.rotation, ACTIVATIONS[bank.activation], w13, w2, s13, s2, x, ids, probabilities, resident, out, tokens, bank.hidden,
                   bank.intermediate, bank.num_experts, topk, *bank.blocks, threads, f13, f2, bank.caps or (0.0, 0.0))

    def gpu_forward(self, bank, x, ids, probabilities, pointers, out=None, mid=None):
        from shared.kernels import expert_triton
        if bank.caps is not None or pointers.shape[1] != 4:
            raise NotImplementedError(f"The GPU kernels of the {self.name} experts have no soft caps and read tables of 4 addresses per expert")
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


class RowScaledExpertFormat(NativeExpertFormat):
    """Expert formats whose rows carry their own block scales (NVFP4 rows, MXFP8 rows), computed by this library on the CPU and by
    shared.kernels.expert_rows_triton on the GPU (gated activations only). The rows of w13 alternate gate and up (row 2i: gate i, row
    2i + 1: up i). Subclasses declare the tensors, the native format id, the kernels' kind and dequantize_matrix(weight, scale, factor)."""
    kind = None  # shared.kernels.expert_rows_triton: NVFP4 or MXFP8

    def blocks(self, hidden, intermediate, tensors):
        if hidden % 128 or intermediate % 128:
            raise ValueError(f"The {self.name} experts take widths that are multiples of 128, not {hidden} and {intermediate}")
        return super().blocks(hidden, intermediate, tensors)

    def count_pairs(self, ids, counts):
        from shared.kernels import expert_rows_triton
        expert_rows_triton.count_pairs(ids, counts)

    def gpu_forward(self, bank, x, ids, probabilities, pointers, out=None, mid=None):
        from shared.kernels import expert_rows_triton
        if bank.rotation or not bank.gated:
            raise NotImplementedError(f"The GPU kernels of the {self.name} experts take a gated activation and no ConvRot rotation")
        return expert_rows_triton.forward(self.kind, x, ids, probabilities, pointers, bank.intermediate, bank.caps, out=out, mid=mid)

    def gpu_pass_supported(self, device):
        from shared.kernels import expert_rows_triton
        return expert_rows_triton.pass_supported(device)

    def pass_shares(self):
        return True  # the CPU kernels compute the exact rows in float32: no less precise than the GPU's pass

    def cpu_threads(self, threads):
        # on Windows the library lets these formats' workers run on both threads of the performance cores (experts.cpp, wide_workers): twice the workers, measured with
        # 8 and 16 on 8 such cores: 16 GB card 63-71 -> 68-72 tokens/s, 12 GB 45-48 -> 50-54, 8 GB 27-29 -> 33-36
        return min(2 * threads, os.cpu_count() or threads) if os.name == "nt" else threads

    def gpu_pass(self, bank, x, host_ids, probabilities, order, zero=False):
        from shared.kernels import expert_rows_triton
        return expert_rows_triton.GroupedPass(self.kind, x, host_ids, probabilities, bank.intermediate, bank.caps, order, zero=zero)

    def dequantize(self, bank, tensors):
        w13, w2, s13, s2, f13, f2 = ([tensors[name] for name in bank.expert_tensor_names] + [None, None])[:6]
        w13 = self.dequantize_matrix(w13, s13, f13)
        return torch.cat((w13[0::2], w13[1::2])) if bank.gated else w13, self.dequantize_matrix(w2, s2, f2)  # gate rows then up rows, as the PyTorch path reads them
