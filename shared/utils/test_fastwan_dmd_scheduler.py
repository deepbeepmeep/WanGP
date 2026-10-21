import unittest

import torch

from shared.utils.fastwan_dmd_scheduler import FastWanDMDScheduler


class FastWanDMDTests(unittest.TestCase):
    def setUp(self):
        self.scheduler = FastWanDMDScheduler()
        self.scheduler.set_timesteps(3, device="cpu")

    def test_published_timestep_and_training_sigma_schedule(self):
        self.assertEqual(self.scheduler.timesteps.tolist(), [1000, 757, 522])
        torch.testing.assert_close(self.scheduler.sigmas, torch.tensor([1., 28/37, 12/23]))

    def test_rejects_untrained_step_counts(self):
        for steps in [0, 1, 2, 4, 50]:
            with self.subTest(steps=steps), self.assertRaisesRegex(ValueError, "exactly 3"):
                self.scheduler.set_timesteps(steps)

    def test_clean_prediction_is_renoised_at_each_nonfinal_step(self):
        sample = torch.tensor([1.5, -2.5])
        flow = torch.tensor([.25, -.5])
        for timestep, sigma, sigma_next in [(1000, 1., 28/37), (757, 28/37, 12/23)]:
            with self.subTest(timestep=timestep):
                generator = torch.Generator().manual_seed(73)
                expected_noise = torch.randn(sample.shape, generator=torch.Generator().manual_seed(73))
                expected = (1-sigma_next)*(sample-sigma*flow) + sigma_next*expected_noise
                actual = self.scheduler.step(flow, timestep, sample, generator=generator)[0]
                torch.testing.assert_close(actual, expected)

    def test_final_step_returns_clean_sample_without_consuming_rng(self):
        generator = torch.Generator().manual_seed(13)
        state = generator.get_state().clone()
        sample = torch.tensor([2., -1.])
        flow = torch.tensor([.5, .25])
        actual = self.scheduler.step(flow, 522, sample, generator=generator, return_dict=False)
        self.assertIsInstance(actual, tuple)
        torch.testing.assert_close(actual[0], sample-(12/23)*flow)
        self.assertTrue(torch.equal(generator.get_state(), state))

    def test_seed_reproduces_the_complete_denoising_sequence(self):
        def run(seed):
            generator = torch.Generator().manual_seed(seed)
            sample = torch.ones(2, 3, 4, 4)
            for timestep in self.scheduler.timesteps:
                sample = self.scheduler.step(sample*.2, timestep, sample, generator=generator)[0]
            return sample
        torch.testing.assert_close(run(9), run(9), rtol=0, atol=0)
        self.assertFalse(torch.equal(run(9), run(10)))

    def test_trimmed_schedule_keeps_correct_final_sigma(self):
        self.scheduler.timesteps = self.scheduler.timesteps[2:]
        self.scheduler.sigmas = self.scheduler.sigmas[2:]
        sample = torch.ones(2)
        result = self.scheduler.step(torch.ones(2), 522, sample)[0]
        torch.testing.assert_close(result, torch.full((2,), 1-12/23))

    def test_low_precision_latents_keep_dtype_with_float32_accumulation(self):
        for dtype in [torch.float16, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                sample = torch.tensor([.5, -1.], dtype=dtype)
                flow = torch.tensor([.3, .6], dtype=dtype)
                actual = self.scheduler.step(flow, 522, sample)[0]
                expected = (sample.float()-self.scheduler.sigmas[-1]*flow.float()).to(dtype)
                self.assertEqual(actual.dtype, dtype)
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
