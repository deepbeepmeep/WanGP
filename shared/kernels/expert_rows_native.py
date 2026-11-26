# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Prompt passes of the row-scaled expert formats (NVFP4 / MXFP8 rows, shared/kernels/expert_rows_triton.py) on Blackwell's block-scaled
tensor cores, with the activation quantization the Inkling 2 checkpoint specifies (hf_quant_config.json "required_activation_quantization":
NVFP4 "fouroversix_mae" per token / per (token, expert) row, MXFP8 per 32 values; gate / up, the activation and each expert's output
rounded to BF16, the top-k outputs summed in float32): LightX2V's moe_ops operators (torch.ops.lightx2v_moe, lightx2v_kernel.moe_ops).
Binning, quantizers, both grouped GEMMs and the slot sum stay on the device; a token's result does not depend on the other rows of the
pass. Opt-in (WGP_INKLING2_ENABLE=fp4_prefill): the arithmetic differs from the float-activation path of the other kernels.
Nemotron Image single-token passes use native FP4 unpacking with floating-point activations and squared ReLU."""
import importlib
import os
from functools import cache

import torch

from shared.kernels import kernel_policy

NVFP4, MXFP8 = 0, 1
ENABLED = "fp4_prefill" in os.environ.get("WGP_INKLING2_ENABLE", "").split(",")
MIN_ROWS = 704  # prompt passes of more tokens than this (measured on the model, RTX 5090: slower than the Triton grouped pass at 640 rows, faster from 768; decoding and draft verification never reach it)
HIDDEN, INTERMEDIATE = 2304, 1152  # the widths the operators are built for (Inkling 2)


@cache
def _library():
    """The operators: lightx2v_kernel.moe_ops when the installed package has it (or already loaded), else None."""
    if not hasattr(torch.ops.lightx2v_moe, "inkling2_prefill"):
        try:
            importlib.import_module("lightx2v_kernel.moe_ops")
        except (ImportError, OSError):
            return None
    return torch.ops.lightx2v_moe if hasattr(torch.ops.lightx2v_moe, "inkling2_prefill") else None


@cache
def supported(device):
    """Compute capability 12.0 / 12.1 (NVIDIA) and a library built with their architecture-specific code."""
    if torch.version.hip is not None or _library() is None:
        return False
    index = device.index if device.index is not None else torch.cuda.current_device()
    return torch.cuda.get_device_capability(index) in ((12, 0), (12, 1)) and bool(_library().supported(index))


def eligible(bank, x):
    """A prompt pass of a bank these operators compute: switch on, approximate kernels allowed (FP4 / FP8 activations: Preserve Precision keeps the float path), BF16 rows beyond
    MIN_ROWS outside graph capture, Inkling 2's widths and soft caps."""
    return (ENABLED and kernel_policy.allow_approximate() and x.is_cuda and x.dtype == torch.bfloat16 and x.shape[0] > MIN_ROWS and bank.hidden == HIDDEN and bank.intermediate == INTERMEDIATE and bank.caps is not None
            and bank.format.name in ("nvfp4_rows", "mxfp8_rows") and not torch.cuda.is_current_stream_capturing() and supported(x.device))


@cache
def _relu2_decode_operator(device):
    if torch.version.hip is None and torch.cuda.get_device_capability(device) in ((12, 0), (12, 1)) and supported(device):
        return getattr(_library(), "nvfp4_relu2_decode", None)
    return None


def relu2_decode(x, ids, probabilities, pointers, intermediate, out, mid):
    """One resident-expert decoding pass with native FP4 unpacking. Older LightX2V installations use the portable kernel."""
    if (x.shape != (1, 2688) or intermediate != 1856 or ids.shape != (1, 6) or x.dtype != torch.bfloat16
            or probabilities.dtype != torch.float32 or ids.dtype not in (torch.int32, torch.int64) or mid.data_ptr() % 16):
        return False
    op = _relu2_decode_operator(x.device)
    if op is None:
        return False
    op(out, mid, x, ids, probabilities, pointers)
    return True


def forward(kind, x, ids, weights, pointers, caps):
    """float32 [tokens, 2304]: the sum over the routed experts having an address in pointers (int64 [experts, 6 (NVFP4) or 4 or more], MMGP's slots: 16-byte aligned)."""
    out = torch.empty(x.shape[0], HIDDEN, dtype=torch.float32, device=x.device)
    _library().inkling2_prefill(out, x.contiguous(), (ids if ids.dtype in (torch.int32, torch.int64) else ids.int()).contiguous(), weights.float().contiguous(), pointers, kind, caps[0], caps[1], 0, False)
    return out
