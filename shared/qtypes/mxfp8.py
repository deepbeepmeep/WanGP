# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""MXFP8 weights (FP8 E4M3 values with a power-of-two scale 2^(byte - 127) per 32 values of a row), as routed expert banks. Linear
layers in this format are not handled."""
import torch

from shared.kernels.cpu_experts import MXFP8_ROWS, RowScaledExpertFormat

HANDLER_NAME = "mxfp8"


class Mxfp8RowExperts(RowScaledExpertFormat):
    """Routed experts: w13_weight E4M3 [experts, 2 * intermediate, hidden], rows alternating gate and up (up rows only without a gate), and w2_weight E4M3 [experts,
    hidden, intermediate]; w13_weight_scale and w2_weight_scale uint8 [experts, rows, columns / 32]. A weight is its E4M3 value times
    2^(scale - 127)."""
    name, native, kind, block = "mxfp8_rows", MXFP8_ROWS, 1, 32

    def tensors(self, experts, hidden, intermediate, gated=True, blocks=None):
        rows, fp8 = (2 if gated else 1) * intermediate, torch.float8_e4m3fn
        return {"w13_weight": ((experts, rows, hidden), fp8), "w2_weight": ((experts, hidden, intermediate), fp8),
                "w13_weight_scale": ((experts, rows, hidden // 32), torch.uint8), "w2_weight_scale": ((experts, hidden, intermediate // 32), torch.uint8)}

    def matches(self, tensors):
        weight, scale = tensors["w13_weight"], tensors["w13_weight_scale"]
        return weight.dtype == torch.float8_e4m3fn and scale.dtype == torch.uint8 and weight.shape[-1] == 32 * scale.shape[-1]

    def dequantize_matrix(self, weight, scale, factor):
        return (weight.float().view(weight.shape[0], -1, 32) * torch.exp2(scale.float() - 127)[:, :, None]).view(weight.shape)


EXPERT_FORMATS = (Mxfp8RowExperts(),)


def detect(state_dict, verboseLevel=1, metadata=None):
    return {"matched": False, "kind": "none", "details": {}}


def convert_to_quanto(state_dict, default_dtype, verboseLevel=1, detection=None, metadata=None):
    return {"state_dict": state_dict, "quant_map": {}}
