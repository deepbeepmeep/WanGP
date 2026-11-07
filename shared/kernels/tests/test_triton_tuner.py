# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""LaunchTuner skips the candidates a GPU cannot run instead of failing the call.

AMD RDNA GPUs have 64 KiB of shared memory per workgroup: some candidates of the row-view GEMM (MiniMax H3 attention projections) need
72 KiB and stopped every H3 generation with "out of resource: shared memory" during tuning.
Run from the repository root: python -m unittest shared.kernels.tests.test_triton_tuner
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import torch


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA or ROCm GPU")
class LaunchTunerTests(unittest.TestCase):
    def _tuner(self, folder):
        from shared.kernels.triton_tuner import LaunchTuner
        tuner = LaunchTuner("test_launch_tuner", 1, "Test")
        tuner.path = Path(folder) / "test_launch_tuner.json"  # never the user's cache
        return tuner

    def test_skips_configurations_out_of_resources(self):
        from triton.runtime.errors import OutOfResources
        scratch = torch.zeros(1024, device="cuda")

        def prepare():
            def launch(config):
                if config == "too_big":
                    raise OutOfResources(73728, 65536, "shared memory")
                scratch.add_(1)
            return launch

        with tempfile.TemporaryDirectory() as folder:
            config = self._tuner(folder).config("kernel", "identity", scratch.device, ["too_big", "fits"], prepare, True)
        self.assertEqual(config, "fits")

    def test_no_runnable_configuration_raises(self):
        from triton.runtime.errors import OutOfResources

        def prepare():
            def launch(config):
                raise OutOfResources(73728, 65536, "shared memory")
            return launch

        with tempfile.TemporaryDirectory() as folder, self.assertRaises(RuntimeError):
            self._tuner(folder).config("kernel", "identity", torch.device("cuda"), ["a", "b"], prepare, True)

    def test_rowmap_gemm_tunes_on_this_gpu(self):
        from shared.kernels import rowmap_triton
        with tempfile.TemporaryDirectory() as folder:
            rowmap_triton._TUNER = self._tuner(folder)  # tune now, in a scratch cache
            heads, slots, head_dim, width = 4, 3, 128, 512
            gen = torch.Generator(device="cuda").manual_seed(3)
            x = torch.randn(rowmap_triton._TUNE_ROWS, width, device="cuda", dtype=torch.bfloat16, generator=gen)
            weight = torch.randn(heads * slots * head_dim, width, device="cuda", dtype=torch.bfloat16, generator=gen) * 0.05
            out = rowmap_triton.rows(x, weight, None, 1, slots, head_dim, 0, heads * head_dim)
            rows = torch.cat([torch.arange(h * slots * head_dim + head_dim, h * slots * head_dim + 2 * head_dim) for h in range(heads)])
            reference = (x.float() @ weight[rows.to(x.device)].float().t()).to(torch.bfloat16)
            torch.testing.assert_close(out, reference, atol=0.05, rtol=0.02)


if __name__ == "__main__":
    unittest.main()
