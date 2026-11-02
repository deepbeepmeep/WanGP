# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Symmetric INT4 weights with FP16 scales per group of 128 input columns (GPTQ without activation order, zero point 8), as routed
expert banks. Each byte holds two columns, the even one in its low nibble: the rows of a GPTQ qweight [in / 8, out] int32 tensor,
transposed and read as bytes. The banks optionally apply a ConvRot rotation (the weights quantized after a regular Hadamard
transform of groups of input columns, see int8_convrot). Linear layers in this format are not handled."""
import torch

from shared.kernels.cpu_experts import INT4_GROUP, NativeExpertFormat

HANDLER_NAME = "gptq_int4"


class Int4GroupExperts(NativeExpertFormat):
    """Routed experts: w13 = gate then up rows [2 * intermediate, hidden / 2] (up rows only [intermediate, hidden / 2] without a gate),
    w2 = down rows [hidden, intermediate / 2], uint8; FP16 scales per group of input columns (128 by default; each projection may use
    another multiple of 32 dividing its input width)."""
    name, native, block = "int4_group", INT4_GROUP, 128

    def tensors(self, experts, hidden, intermediate, gated=True, blocks=None):
        g13, g2 = blocks or (self.block, self.block)
        rows = (2 if gated else 1) * intermediate
        return {"w13_weight": ((experts, rows, hidden // 2), torch.uint8), "w2_weight": ((experts, hidden, intermediate // 2), torch.uint8),
                "w13_scales": ((experts, rows, hidden // g13), torch.float16), "w2_scales": ((experts, hidden, intermediate // g2), torch.float16)}

    def matches(self, tensors):
        return tensors["w13_weight"].dtype == torch.uint8 and tensors["w13_scales"].dtype == torch.float16

    def dequantize_matrix(self, weight, scale, block):
        rows, half = weight.shape
        values = torch.stack((weight & 15, weight >> 4), -1).view(rows, 2 * half).float() - 8
        return (values.view(rows, -1, block) * scale.float()[:, :, None]).view(rows, 2 * half)


EXPERT_FORMATS = (Int4GroupExperts(),)


def detect(state_dict, verboseLevel=1, metadata=None):
    return {"matched": False, "kind": "none", "details": {}}


def convert_to_quanto(state_dict, default_dtype, verboseLevel=1, detection=None, metadata=None):
    return {"state_dict": state_dict, "quant_map": {}}
