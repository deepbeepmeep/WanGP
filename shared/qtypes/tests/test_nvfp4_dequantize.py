# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""NVFP4 weight dequantization (tensor-core layout) gives the decoded FP4 values times their block scales and the global factor, ROCm included.

On ROCm, Comfy Kitchen's CUDA extension imports too, but its dequantize_nvfp4 rejects ROCm tensors
("incompatible function arguments"): the NVFP4 text encoder of MiniMax H3 failed to load. ROCm must use the PyTorch path.
Run from the repository root: python -m unittest shared.qtypes.tests.test_nvfp4_dequantize
"""
from __future__ import annotations

import unittest

import torch

FP4_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA or ROCm GPU")
class Nvfp4DequantizeTests(unittest.TestCase):
    def test_kitchen_cuda_backend_is_off_on_rocm(self):
        from shared.qtypes import nvfp4
        if torch.version.hip is not None:
            self.assertFalse(nvfp4._ck_cuda_available)

    def test_tensorcore_weight_matches_reference(self):
        from shared.qtypes import nvfp4
        gen = torch.Generator(device="cuda").manual_seed(7)
        rows, k = 256, 256  # 128-row tiles of 64-value scale groups: two of each, so the deswizzle is exercised
        weight = torch.randint(0, 256, (rows, k // 2), device="cuda", dtype=torch.uint8, generator=gen)
        block_scales = (torch.rand(rows, k // 16, device="cuda", generator=gen) * 4 + 0.25).to(torch.float8_e4m3fn)
        alpha = torch.tensor(0.75, device="cuda")
        out = nvfp4._dequantize_nvfp4_weight(weight, block_scales, torch.tensor(1.0, device="cuda"), alpha, torch.bfloat16,
                                             torch.device("cuda"), layout=nvfp4._NVFP4_LAYOUT_TENSORCORE)
        lut = torch.tensor(FP4_E2M1, device="cuda", dtype=torch.float32)
        values = torch.stack((lut[(weight >> 4).long()], lut[(weight & 15).long()]), dim=-1).reshape(rows, k)  # high nibble first
        scales = nvfp4._deswizzle_nvfp4_scale(block_scales, k).float().repeat_interleave(16, dim=1)
        reference = (values * scales * 0.75).to(torch.bfloat16)
        self.assertEqual(out.shape, (rows, k))
        self.assertTrue(torch.equal(out, reference))


if __name__ == "__main__":
    unittest.main()
