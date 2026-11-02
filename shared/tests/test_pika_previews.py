import importlib.util
from pathlib import Path
import unittest

import torch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("pika_sampler", ROOT / "models/minimax_h3/pika.py")
PIKA = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PIKA)


class PikaPreviewTests(unittest.TestCase):
    def test_denoised_preview_preserves_renoise_behavior_and_rng(self):
        for final in (False, True):
            with self.subTest(final=final):
                sample = torch.tensor([[[[[0.25, -0.5], [1.0, 0.75]]]]])
                velocity = torch.tensor([[[[[0.5, 0.25], [-0.5, 1.0]]]]])
                sigma, sigma_next = 0.4, 0.2
                expected_clean = velocity * sigma + sample

                baseline, preview = sample.clone(), sample.clone()
                default_rng = torch.Generator(device="cpu").manual_seed(73)
                preview_rng = torch.Generator(device="cpu").manual_seed(73)
                default_result = PIKA.renoise_(baseline, [velocity.clone()], sigma, sigma_next, default_rng, final=final)
                denoised = PIKA.renoise_(preview, [velocity.clone()], sigma, sigma_next, preview_rng, final=final, return_denoised=True)

                self.assertIsNone(default_result)
                torch.testing.assert_close(denoised, expected_clean)
                torch.testing.assert_close(preview, baseline)
                self.assertTrue(torch.equal(preview_rng.get_state(), default_rng.get_state()))


if __name__ == "__main__":
    unittest.main()
