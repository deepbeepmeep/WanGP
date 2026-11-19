"""Run separately: the MPS compatibility layer changes process-wide torch APIs."""
import os
import subprocess
import sys
import unittest
from pathlib import Path

import torch


@unittest.skipUnless(torch.backends.mps.is_available(), "Requires Apple Silicon MPS")
class StagedTransferTests(unittest.TestCase):
    def test_reused_cpu_slots_do_not_corrupt_gpu_weights(self):
        script = '''
from shared.mps import device_patch
import torch
from mmgp.offload import _PinnedStager

torch.set_num_threads(1)
stager = _PinnedStager()
try:
    sources = [torch.full((2048, 2048), float(i + 1), dtype=torch.bfloat16)
               for i in range(8)]
    outputs = [stager.to_gpu(source) for source in sources]
    torch.mps.synchronize()
    for i, (source, output) in enumerate(zip(sources, outputs)):
        assert torch.equal(source, output.cpu()), f"Staged transfer {i} corrupted"
finally:
    torch.mps.synchronize()
    stager.release()
'''
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1",
                   PYTORCH_MPS_HIGH_WATERMARK_RATIO="0.35",
                   PYTORCH_MPS_LOW_WATERMARK_RATIO="0.3")
        result = subprocess.run([sys.executable, "-B", "-c", script],
                                cwd=Path(__file__).resolve().parents[2], env=env,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
