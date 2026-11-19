"""Run with Metal access; the regression is in mixed CPU-scalar/MPS arithmetic."""
import unittest

import torch

from shared.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


@unittest.skipUnless(torch.backends.mps.is_available(), "Requires Apple Silicon MPS")
class UniPCScalarTests(unittest.TestCase):
    def test_model_output_uses_the_selected_sigma(self):
        sample = torch.tensor([1., -2., .3])
        output = torch.tensor([.5, .7, -.2])
        for predict_x0 in [True, False]:
            scheduler = FlowUniPCMultistepScheduler(predict_x0=predict_x0)
            scheduler.set_timesteps(50, device="mps", shift=5)
            for index in [0, 1, 20, 49]:
                with self.subTest(predict_x0=predict_x0, index=index):
                    scheduler._step_index = index
                    sigma = scheduler.sigmas[index].item()
                    expected = sample - (sigma if predict_x0 else 1-sigma)*output
                    actual = scheduler.convert_model_output(output.to("mps"), sample=sample.to("mps"))
                    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-6)

    def test_fifty_step_smooth_trajectory_matches_cpu(self):
        results = []
        for device in ["cpu", "mps"]:
            scheduler = FlowUniPCMultistepScheduler()
            scheduler.set_timesteps(50, device=device, shift=5)
            generator = torch.Generator(device="cpu").manual_seed(9)
            sample = torch.randn((1, 48, 3, 8, 8), generator=generator).to(device)
            for timestep in scheduler.timesteps:
                flow = .5*sample + .1*torch.sin(sample)
                sample = scheduler.step(flow, timestep, sample)[0]
            results.append(sample.cpu())
        torch.testing.assert_close(results[0], results[1], rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
