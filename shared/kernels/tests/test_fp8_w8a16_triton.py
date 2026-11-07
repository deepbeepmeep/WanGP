# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""The weight-only FP8 Triton matmul matches dequantization + matmul on the running GPU, ROCm included.

On ROCm the kernel used to ask Triton for tf32 input precision, which the AMD backend rejects when it compiles the kernel.
Run from the repository root: python -m unittest shared.kernels.tests.test_fp8_w8a16_triton
"""
from __future__ import annotations

import os
import unittest

import torch

os.environ.setdefault("WAN2GP_FP8_TRITON_AUTOTUNE", "0")  # static configurations: no tuning runs, no cache file written

CASES = [  # (tokens M, out features N, in features K, weight dtype, one scale per output row, bias)
    (8, 256, 512, torch.float8_e4m3fn, True, False),
    (77, 384, 640, torch.float8_e4m3fn, False, True),
    (300, 512, 1000, torch.float8_e4m3fn, True, True),  # K is not a multiple of the tile: masked rows
    (64, 256, 512, torch.float8_e5m2, True, False),
]


def _reference(x, weight, scale, bias):
    w = (weight.to(torch.float32) * scale.to(torch.float32).view(-1, 1)).to(x.dtype)  # rounded to the compute dtype, as Quanto does
    out = (x.to(torch.float32) @ w.to(torch.float32).t()).to(x.dtype)
    return out if bias is None else (out.to(torch.float32) + bias.to(torch.float32)).to(x.dtype)


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA or ROCm GPU")
class Fp8W8A16TritonTests(unittest.TestCase):
    def test_matches_dequantized_matmul(self):
        from shared.kernels import fp8_w8a16_triton
        for m, n, k, wdtype, per_row, with_bias in CASES:
            with self.subTest(m=m, n=n, k=k, weight=str(wdtype), per_row=per_row, bias=with_bias):
                gen = torch.Generator(device="cuda").manual_seed(m * 7919 + n + k)
                x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, generator=gen)
                weight = (torch.randn(n, k, device="cuda", generator=gen) * 2).to(wdtype)
                scale = (torch.rand(n if per_row else 1, device="cuda", generator=gen) * 0.05 + 0.01).to(torch.bfloat16)
                bias = torch.randn(n, device="cuda", dtype=torch.bfloat16, generator=gen) if with_bias else None
                out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
                fp8_w8a16_triton.linear(x, weight, scale, bias, out)
                ref = _reference(x, weight, scale, bias).float()
                self.assertTrue(torch.isfinite(out).all().item())
                # same products, sums in another order: only BF16 rounding differs
                self.assertLessEqual((out.float() - ref).abs().max().item(), 0.02 * ref.abs().max().item())


if __name__ == "__main__":
    unittest.main()
