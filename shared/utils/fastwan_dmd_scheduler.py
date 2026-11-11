"""Three-step DMD sampling for the FastWan 2.2 5B FullAttn accelerator."""
import torch
from diffusers.schedulers.scheduling_utils import SchedulerOutput


class FastWanDMDScheduler:
    is_stateful = False

    def set_timesteps(self, num_inference_steps, device=None):
        if num_inference_steps != 3:
            raise ValueError("FastWan DMD requires exactly 3 inference steps.")
        self.timesteps = torch.tensor([1000, 757, 522], device=device)
        # FastVideo's DMD pipeline uses the nearest values in a 1000-step,
        # shift-8 training schedule, independently of the UI flow shift.
        # https://github.com/hao-ai-lab/FastVideo/blob/main/fastvideo/pipelines/basic/wan/wan_dmd_pipeline.py
        training_sigmas = torch.linspace(1000, 1, 1000, dtype=torch.float32) / 1000
        training_sigmas = 8 * training_sigmas / (1 + 7 * training_sigmas)
        indices = (training_sigmas[:, None] * 1000 - self.timesteps.cpu()).abs().argmin(dim=0)
        self.sigmas = training_sigmas[indices].to(device)

    def step(self, model_output, timestep, sample, generator=None, return_dict=True):
        index = int((self.timesteps - timestep).abs().argmin().item())
        # DMD predicts a clean sample, then adds fresh noise for the next
        # timestep. UniPC's multistep ODE update is not interchangeable.
        # Accumulate in float32 because Metal does not support float64.
        clean = sample.float() - self.sigmas[index] * model_output.float()
        if index + 1 < len(self.timesteps):
            noise = torch.randn(sample.shape, device=sample.device,
                                dtype=sample.dtype, generator=generator)
            sigma_next = self.sigmas[index + 1]
            previous = (1 - sigma_next) * clean + sigma_next * noise.float()
        else:
            previous = clean
        previous = previous.to(sample.dtype)
        if not return_dict:
            return (previous,)
        return SchedulerOutput(prev_sample=previous)
