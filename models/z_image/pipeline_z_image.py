# Copyright 2025 Alibaba Z-Image Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modifications Copyright (c) 2025-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)

from shared.utils.phase_progress import text_encoding_progress
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer, PreTrainedModel

from diffusers.image_processor import VaeImageProcessor
from diffusers.loaders import FromSingleFileMixin
from diffusers.models.autoencoders import AutoencoderKL
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from mmgp import offload
from .z_image_transformer2d import ZImageTransformer2DModel
from .unified_sampler import UnifiedSampler
from .pipeline_output import ZImagePipelineOutput
from shared.utils.utils import get_outpainting_frame_location, resize_lanczos, calculate_new_dimensions, convert_image_to_tensor, fit_image_into_canvas
from shared.utils.loras_mutipliers import update_loras_slists
from shared.utils.text_encoder_cache import TextEncoderCache


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import ZImagePipeline

        >>> pipe = ZImagePipeline.from_pretrained("Z-a-o/Z-Image-Turbo", torch_dtype=torch.bfloat16)
        >>> pipe.to("cuda")

        >>> # Optionally, set the attention backend to flash-attn 2 or 3, default is SDPA in PyTorch.
        >>> # (1) Use flash attention 2
        >>> # pipe.transformer.set_attention_backend("flash")
        >>> # (2) Use flash attention 3
        >>> # pipe.transformer.set_attention_backend("_flash_3")

        >>> prompt = "一幅为名为“造相「Z-IMAGE-TURBO」”的项目设计的创意海报。画面巧妙地将文字概念视觉化：一辆复古蒸汽小火车化身为巨大的拉链头，正拉开厚厚的冬日积雪，展露出一个生机盎然的春天。"
        >>> image = pipe(
        ...     prompt,
        ...     height=1024,
        ...     width=1024,
        ...     num_inference_steps=9,
        ...     guidance_scale=0.0,
        ...     generator=torch.Generator("cuda").manual_seed(42),
        ... ).images[0]
        >>> image.save("zimage.png")
        ```
"""

# Default negative prompt for Z-Image when user input is empty.
DEFAULT_NEGATIVE_PROMPT = (
    "low quality, worst quality, blurry, pixelated, noisy, artifacts, watermark, text, logo, "
    "bad anatomy, bad hands, extra limbs"
)


# Copied from diffusers.pipelines.flux.pipeline_flux.calculate_shift
def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


_UNIFIED_SOLVERS = {"unified", "unified_2s", "unified_4s", "twinflow"}
_UNIFIED_PRESET_GAP = {
    "unified_2s": [0.001, 0.6],
    "unified_4s": [0.001, 0.5],
    "unified_mul": [0.001, 0.0],
}


class _UnifiedSamplerInterrupted(RuntimeError):
    pass


def _is_unified_solver(sample_solver: Optional[str]) -> bool:
    solver = (sample_solver or "").strip().lower()
    return solver in _UNIFIED_SOLVERS or solver.startswith("unified_")


def _resolve_unified_sampler_config(
    sample_solver: Optional[str],
    num_inference_steps: int,
    unified_sampler_config: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    solver = (sample_solver or "").strip().lower()
    sampling_steps = max(int(num_inference_steps), 1)
    if solver == "unified_2s":
        sampling_steps = 2
    elif solver == "unified_4s":
        sampling_steps = 4

    if sampling_steps <= 2:
        preset_key = "unified_2s"
        sampling_style = "few"
    elif sampling_steps <= 4:
        preset_key = "unified_4s"
        sampling_style = "any"
    else:
        preset_key = "unified_mul"
        sampling_style = "mul"

    rfba_gap_steps = _UNIFIED_PRESET_GAP[preset_key]
    config = {
        "sampling_steps": sampling_steps,
        "stochast_ratio": 1.0,
        "extrapol_ratio": 0.0,
        "sampling_order": 1,
        "sampling_style": sampling_style,
        "time_dist_ctrl": [1.0, 1.0, 1.0],
        "rfba_gap_steps": list(rfba_gap_steps),
    }
    if unified_sampler_config:
        config.update(unified_sampler_config)
    return config


class ZImagePipeline(DiffusionPipeline, FromSingleFileMixin):
    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _optional_components = []
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: PreTrainedModel,
        tokenizer: AutoTokenizer,
        transformer: ZImageTransformer2DModel,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            transformer=transformer,
        )
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if hasattr(self, "vae") and self.vae is not None else 8
        )
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.text_encoder_cache = TextEncoderCache()

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 512,
        lora_scale: Optional[float] = None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if do_classifier_free_guidance:
            if negative_prompt is None:
                negative_prompt = [DEFAULT_NEGATIVE_PROMPT for _ in prompt]
            elif isinstance(negative_prompt, str):
                negative_prompt = [DEFAULT_NEGATIVE_PROMPT for _ in prompt] if not negative_prompt.strip() else [negative_prompt]
            elif len(negative_prompt) == len(prompt):
                negative_prompt = [DEFAULT_NEGATIVE_PROMPT if p is None or (isinstance(p, str) and not p.strip()) else p for p in negative_prompt]
            assert len(prompt) == len(negative_prompt)
        pending_positive = prompt if prompt_embeds is None else []
        pending_negative = negative_prompt if do_classifier_free_guidance and negative_prompt_embeds is None else []
        encoded = self._encode_prompt(pending_positive + pending_negative, device=device, dtype=dtype, num_images_per_prompt=num_images_per_prompt, max_sequence_length=max_sequence_length)
        if prompt_embeds is None:
            prompt_embeds = encoded[:len(pending_positive)]
        if not do_classifier_free_guidance:
            negative_prompt_embeds = []
        elif negative_prompt_embeds is None:
            negative_prompt_embeds = encoded[len(pending_positive):]
        return prompt_embeds, negative_prompt_embeds

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        max_sequence_length: int = 512,
    ) -> List[torch.FloatTensor]:
        assert num_images_per_prompt == 1
        device = device or self._execution_device

        if prompt_embeds is not None:
            return prompt_embeds

        if isinstance(prompt, str):
            prompt = [prompt]

        def encode_fn(prompts):
            formatted_prompts = []
            for prompt_item in prompts:
                messages = [
                    {"role": "user", "content": prompt_item},
                ]
                formatted_prompts.append(
                    self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=True,
                    )
                )
            text_inputs = self.tokenizer(
                formatted_prompts,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(device)
            prompt_masks = text_inputs.attention_mask.to(device).bool()
            with text_encoding_progress(self.text_encoder.model.layers, prompt_count=len(prompts)):
                prompt_embeds = self.text_encoder.model(input_ids=text_input_ids, attention_mask=prompt_masks, output_hidden_states=True).hidden_states[-2]
            embeddings_list = []
            for i in range(len(prompt_embeds)):
                embeddings_list.append(prompt_embeds[i][prompt_masks[i]])
            return embeddings_list

        cache_keys = [(max_sequence_length, p) for p in prompt]
        return self.text_encoder_cache.encode(encode_fn, prompt, device=device, cache_keys=cache_keys)

    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents=None,
    ):
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, height, width)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        sample_solver: str = "default",
        unified_sampler_config: Optional[Dict[str, Any]] = None,
        guidance_scale: float = 5.0,
        cfg_normalization: bool = False,
        cfg_truncation: float = 1.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        negative_prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        callback=None,
        pipeline=None,
        control_image: Optional[torch.Tensor] = None,
        inpaint_mask: Optional[torch.Tensor] = None,
        control_context_scale: float = 0.75,
        input_ref_images = None,
        NAG_scale: float = 1.0,
        NAG_tau: float = 3.5,
        NAG_alpha: float = 0.5,
        loras_slists = None,
        vae_upsampler = None,
        vae_upsampler_seed: int = 0,
        vae_upsampler_progress_callback = None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, *optional*, defaults to 1024):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 1024):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            sample_solver (`str`, *optional*, defaults to `"default"`):
                Sampler selection. Use `"unified"` for the TwinFlow unified sampler (2-step vs 4-step preset is chosen
                from `num_inference_steps`). `"unified_2s"`/`"unified_4s"` are still accepted for compatibility.
            unified_sampler_config (`dict`, *optional*):
                Optional overrides for unified sampler parameters.
            guidance_scale (`float`, *optional*, defaults to 5.0):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            cfg_normalization (`bool`, *optional*, defaults to False):
                Whether to apply configuration normalization.
            cfg_truncation (`float`, *optional*, defaults to 1.0):
                The truncation value for configuration.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`List[torch.FloatTensor]`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`List[torch.FloatTensor]`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.ZImagePipelineOutput`] instead of a plain
                tuple.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, *optional*, defaults to 512):
                Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.z_image.ZImagePipelineOutput`] or `tuple`: [`~pipelines.z_image.ZImagePipelineOutput`] if
            `return_dict` is True, otherwise a `tuple`. When returning a tuple, the first element is a list with the
            generated images.
        """
        height = height or 1024
        width = width or 1024

        vae_scale = self.vae_scale_factor * 2
        if height % vae_scale != 0:
            raise ValueError(
                f"Height must be divisible by {vae_scale} (got {height}). "
                f"Please adjust the height to a multiple of {vae_scale}."
            )
        if width % vae_scale != 0:
            raise ValueError(
                f"Width must be divisible by {vae_scale} (got {width}). "
                f"Please adjust the width to a multiple of {vae_scale}."
            )

        self._guidance_scale = guidance_scale
        kwargs = {}
        NAG = None
        # NAG is only safe without CFG; skip it when guidance > 1.
        if NAG_scale > 1 and guidance_scale <= 1:
            NAG = { "scale": NAG_scale, "tau": NAG_tau, "alpha": NAG_alpha, }

        dtype = (
            self.text_encoder.dtype
            if prompt_embeds is None
            else (prompt_embeds[0].dtype if isinstance(prompt_embeds, list) else prompt_embeds.dtype)
        )
        device = self._execution_device
        self.vae.enable_slicing()
        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False
        self._cfg_normalization = cfg_normalization
        self._cfg_truncation = cfg_truncation
        self.transformer.loras_slists = loras_slists
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = len(prompt_embeds)

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )

        # If prompt_embeds is provided and prompt is None, skip encoding
        if prompt_embeds is not None and prompt is None:
            if self.do_classifier_free_guidance and negative_prompt_embeds is None:
                raise ValueError(
                    "When `prompt_embeds` is provided without `prompt`, "
                    "`negative_prompt_embeds` must also be provided for classifier-free guidance."
                )
        else:
            (
                prompt_embeds,
                negative_prompt_embeds,
            ) = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance or NAG is not None,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                dtype=dtype,
                device=device,
                num_images_per_prompt=1,
                max_sequence_length=max_sequence_length,
                lora_scale=lora_scale,
            )
        neg_embeds_list = negative_prompt_embeds if isinstance(negative_prompt_embeds, list) else []
        if NAG is not None and len(neg_embeds_list) > 0:
            NAG["neg_feats"] = neg_embeds_list[0]
            kwargs["NAG"] = NAG

        batch_size = num_images_per_prompt
        num_images_per_prompt = 1
        if batch_size > 1:
            prompt_embeds = prompt_embeds * batch_size
            if len(negative_prompt_embeds) > 0:
                negative_prompt_embeds = negative_prompt_embeds * batch_size

        # 4. Prepare latent variables

        num_channels_latents = self.transformer.in_channels

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            torch.float32,
            device,
            generator,
            latents,
        )
        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)

        use_unified = _is_unified_solver(sample_solver)
        if not use_unified:
            # 5. Prepare timesteps
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 4096),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 1.15),
            )
            self.scheduler.sigma_min = 0.0
            scheduler_kwargs = {"mu": mu}
            timesteps, num_inference_steps = retrieve_timesteps(
                self.scheduler,
                num_inference_steps,
                device,
                sigmas=sigmas,
                **scheduler_kwargs,
            )
            num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
            self._num_timesteps = len(timesteps)

        # Encode control image if provided and transformer supports it
        control_latent = control_image_tensor = None
        control_in_dim = self.transformer.control_in_dim
        if control_image is not None and hasattr(self.transformer, 'has_control') and self.transformer.has_control:
            # input_frames comes in video format [C, F, H, W] - convert to image format [B, C, H, W]
            control_image_tensor = control_image
            if control_image_tensor.dim() == 4 and control_image_tensor.shape[1] == 1:
                # Shape is [C, F=1, H, W] - squeeze frame dim and add batch dim
                control_image_tensor = control_image_tensor.squeeze(1).unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
            elif control_image_tensor.dim() == 3:
                # Shape is [C, H, W] - just add batch dim
                control_image_tensor = control_image_tensor.unsqueeze(0)  # [1, C, H, W]
            control_image_tensor = control_image_tensor.to(device=device, dtype=self.vae.dtype)
            if inpaint_mask is None:
                control_latent = self.vae.encode(control_image_tensor).latent_dist.mode()
                control_latent = (control_latent - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        # Build control_latent_input outside the loop (doesn't change per step)
        control_latent_input = None
        if control_image_tensor is not None:
            # For v2 control (control_in_dim=33): concat [control_latent(16), keep_mask(1), inpaint_latent(16)]
            if control_in_dim is not None and control_in_dim > num_channels_latents:
                if inpaint_mask is not None:
                    inpaint_mask = inpaint_mask.to(device)
                    inpaint_image = control_image_tensor * (1-inpaint_mask) #+ inpaint_mask
                    inpaint_latent = self.vae.encode(inpaint_image).latent_dist.mode()
                    inpaint_latent = (inpaint_latent - self.vae.config.shift_factor) * self.vae.config.scaling_factor

                    if input_ref_images is not None and len(input_ref_images):
                        control_image_tensor = convert_image_to_tensor(input_ref_images[0]).unsqueeze(0) 
                    
                    control_image_tensor = control_image_tensor.to(device=device, dtype=self.vae.dtype)
                    control_latent = self.vae.encode(control_image_tensor).latent_dist.mode()
                    control_latent = (control_latent - self.vae.config.shift_factor) * self.vae.config.scaling_factor

                    mask = inpaint_mask
                    if mask.dim() == 3:
                        mask = mask.unsqueeze(0)  # Add batch dim
                    if mask.dim() == 4 and mask.shape[1] > 1:
                        mask = mask[:, :1]  # Take first channel only
                    if mask.shape[-2:] != inpaint_latent.shape[-2:]:
                        mask = torch.nn.functional.interpolate(
                            mask.float(), size=inpaint_latent.shape[-2:], mode='nearest'
                        )
                    mask = mask.to(device=control_latent.device, dtype=inpaint_latent.dtype)
                    keep_mask = 1.0 - mask

                    control_context = torch.cat([control_latent, keep_mask, inpaint_latent], dim=1)
                else:
                    keep_mask = torch.zeros(
                        (control_latent.shape[0], 1, control_latent.shape[2], control_latent.shape[3]),
                        device=control_latent.device, dtype=control_latent.dtype
                    )
                    inpaint_latent = torch.zeros_like(control_latent)
                    control_context = torch.cat([control_latent, keep_mask, inpaint_latent], dim=1)
                control_latent_input = control_context.unsqueeze(2)
            else:
                # v1 control: just use the control latent directly (16 channels)
                control_latent_input = control_latent.unsqueeze(2)

        if use_unified:
            unified_cfg = _resolve_unified_sampler_config(sample_solver, num_inference_steps, unified_sampler_config)
            sampler = UnifiedSampler()
            sampling_steps = int(unified_cfg.get("sampling_steps", num_inference_steps))
            sampling_order = int(unified_cfg.get("sampling_order", 1))
            sampling_style = str(unified_cfg.get("sampling_style", "few")).lower()
            stochast_ratio = unified_cfg.get("stochast_ratio", 0.0)
            extrapol_ratio = unified_cfg.get("extrapol_ratio", 0.0)
            time_dist_ctrl = unified_cfg.get("time_dist_ctrl", [1.0, 1.0, 1.0])
            rfba_gap_steps = unified_cfg.get("rfba_gap_steps", [0.0, 0.0])
            if sampling_style not in {"few", "mul", "any"}:
                raise ValueError(f"Unknown unified sampling_style '{sampling_style}'")

            num_steps = (sampling_steps + 1) // 2 if sampling_order == 2 else sampling_steps
            if (rfba_gap_steps[1] - 0.0) == 0.0:
                num_steps += 1
            t_steps = torch.linspace(
                rfba_gap_steps[0],
                1.0 - rfba_gap_steps[1],
                num_steps,
                dtype=torch.float64,
                device=latents.device,
            ).to(latents)
            if (rfba_gap_steps[1] - 0.0) == 0.0:
                t_steps = t_steps[:-1]
            t_steps = sampler.kumaraswamy_transform(t_steps, *time_dist_ctrl)
            t_steps = torch.cat([(1 - t_steps), torch.zeros_like(t_steps[:1])])
            total_steps = max(int(t_steps.numel() - 1), 0)
            self._num_timesteps = total_steps

            callback(-1, None, True, total_steps)
            if hasattr(self.transformer, 'loras_slists') and self.transformer.loras_slists is not None:
                update_loras_slists(self.transformer, self.transformer.loras_slists, total_steps)

            def model_fn(x_t, t, tt=None):
                if self._interrupt:
                    raise _UnifiedSamplerInterrupted()

                t = t.flatten()
                t_sign = t.sign()
                t_abs = t.abs()
                t_mapped = t_sign * (1.0 - t_abs)
                timestep = t_mapped.expand(x_t.shape[0]).to(torch.float32)
                t_norm = timestep[0].item()
                target_timestep = None
                if tt is not None:
                    tt = tt.flatten()
                    tt_sign = tt.sign()
                    tt_abs = tt.abs()
                    tt_mapped = tt_sign * (1.0 - tt_abs)
                    target_timestep = tt_mapped.expand(x_t.shape[0]).to(torch.float32)

                current_guidance_scale = self.guidance_scale
                if (
                    self.do_classifier_free_guidance
                    and self._cfg_truncation is not None
                    and float(self._cfg_truncation) <= 1
                ):
                    if t_norm > self._cfg_truncation:
                        current_guidance_scale = 0.0

                apply_cfg = self.do_classifier_free_guidance and current_guidance_scale > 0
                latent_model_input = x_t if x_t.dtype == dtype else x_t.to(dtype)
                latent_model_input = latent_model_input.unsqueeze(2)

                if apply_cfg:
                    model_out_list = self.transformer(
                        [latent_model_input, latent_model_input],
                        timestep,
                        [prompt_embeds[0], negative_prompt_embeds[0]],
                        control_context_list=[control_latent_input, control_latent_input] if control_latent_input is not None else None,
                        control_context_scale=control_context_scale,
                        target_timestep=target_timestep,
                        callback=callback,
                        pipeline=self,
                        **kwargs,
                    )

                    if model_out_list is None:
                        raise _UnifiedSamplerInterrupted()

                    pos, neg = model_out_list
                    model_out_list = None
                    pos, neg = pos.float(), neg.float()

                    noise_pred = pos + current_guidance_scale * (pos - neg)

                    if self._cfg_normalization and float(self._cfg_normalization) > 0.0:
                        ori_pos_norm = torch.linalg.vector_norm(pos)
                        new_pos_norm = torch.linalg.vector_norm(noise_pred)
                        max_new_norm = ori_pos_norm * float(self._cfg_normalization)
                        if new_pos_norm > max_new_norm:
                            noise_pred = noise_pred * (max_new_norm / new_pos_norm)
                    pos = neg = None
                else:
                    model_out_list = self.transformer(
                        [latent_model_input],
                        timestep,
                        [prompt_embeds[0]],
                        control_context_list=[control_latent_input] if control_latent_input is not None else None,
                        control_context_scale=control_context_scale,
                        target_timestep=target_timestep,
                        callback=callback,
                        pipeline=self,
                        **kwargs,
                    )

                    if model_out_list is None:
                        raise _UnifiedSamplerInterrupted()
                    noise_pred = model_out_list[0]
                    model_out_list = None

                noise_pred = noise_pred.squeeze(2)
                noise_pred = -noise_pred
                return noise_pred

            input_dtype = latents.dtype
            x_cur = latents.to(torch.float64)
            x_hats, z_hats = [], []
            buffer_freq = 1
            final_x_hat = None

            with self.progress_bar(total=total_steps) as progress_bar:
                for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
                    offload.set_step_no_for_lora(self.transformer, i)
                    if self.interrupt:
                        break

                    if sampling_style == "few":
                        t_tgt = torch.zeros_like(t_cur)
                    elif sampling_style == "mul":
                        t_tgt = t_cur
                    else:
                        t_tgt = t_next

                    try:
                        x_hat, z_hat, _, _ = sampler.forward(
                            model_fn,
                            x_cur.to(input_dtype),
                            t_cur.to(input_dtype),
                            t_tgt.to(input_dtype),
                        )
                    except _UnifiedSamplerInterrupted:
                        self._interrupt = True
                        break

                    final_x_hat = x_hat.to(torch.float32)
                    x_hat, z_hat = x_hat.to(torch.float64), z_hat.to(torch.float64)

                    if buffer_freq > 0 and extrapol_ratio > 0:
                        z_hats.append(z_hat)
                        x_hats.append(x_hat)
                        if i > buffer_freq:
                            z_hat = z_hat + extrapol_ratio * (z_hat - z_hats[-buffer_freq - 1])
                            x_hat = x_hat + extrapol_ratio * (x_hat - x_hats[-buffer_freq - 1])
                            z_hats.pop(0)
                            x_hats.pop(0)

                    if stochast_ratio == "SDE":
                        stochast_ratio = (
                            torch.sqrt((t_next - t_cur).abs())
                            * torch.sqrt(2 * sampler.alpha_in(t_cur))
                            / sampler.alpha_in(t_next)
                        )
                        stochast_ratio = torch.clamp(stochast_ratio ** (1 / 0.50), min=0, max=1)
                        noi = torch.randn(x_cur.size(), device=x_cur.device).to(x_cur)
                    else:
                        noi = torch.randn(x_cur.size(), device=x_cur.device).to(x_cur) if stochast_ratio > 0 else 0.0
                    x_next = sampler.gamma_in(t_next) * x_hat + sampler.alpha_in(t_next) * (
                        z_hat * ((1 - stochast_ratio) ** 0.5) + noi * (stochast_ratio**0.5)
                    )

                    if sampling_order == 2 and i < num_steps - 1:
                        if sampling_style == "few":
                            t_tgt = torch.zeros_like(t_next)
                        elif sampling_style == "mul":
                            t_tgt = t_next
                        else:
                            t_tgt = t_next
                        x_pri, z_pri, _, _ = sampler.forward(
                            model_fn,
                            x_next.to(input_dtype),
                            t_next.to(input_dtype),
                            t_tgt.to(input_dtype),
                        )
                        x_pri, z_pri = x_pri.to(torch.float64), z_pri.to(torch.float64)

                        x_next = x_cur * sampler.gamma_in(t_next) / sampler.gamma_in(t_cur) + (
                            sampler.alpha_in(t_next)
                            - sampler.gamma_in(t_next) * sampler.alpha_in(t_cur) / sampler.gamma_in(t_cur)
                        ) * (0.5 * z_hat + 0.5 * z_pri)

                    x_cur = x_next

                    if self._interrupt:
                        break
                    if callback is not None and final_x_hat is not None:
                        latents_preview = final_x_hat.unsqueeze(2)
                        if len(latents_preview) > 1:
                            latents_preview = latents_preview.transpose(0, 2)
                        callback(i, latents_preview[0], False)
                        latents_preview = None

            if self._interrupt or final_x_hat is None:
                return None

            latents = final_x_hat
        else:
            callback(-1, None, True, len(timesteps))

            if hasattr(self.transformer, 'loras_slists') and self.transformer.loras_slists is not None:
                update_loras_slists(self.transformer, self.transformer.loras_slists, len(timesteps))

            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    offload.set_step_no_for_lora(self.transformer, i)
                    if self.interrupt:
                        break

                    timestep = t.expand(latents.shape[0])
                    timestep = (1000 - timestep) / 1000
                    t_norm = timestep[0].item()

                    current_guidance_scale = self.guidance_scale
                    if (
                        self.do_classifier_free_guidance
                        and self._cfg_truncation is not None
                        and float(self._cfg_truncation) <= 1
                    ):
                        if t_norm > self._cfg_truncation:
                            current_guidance_scale = 0.0

                    apply_cfg = self.do_classifier_free_guidance and current_guidance_scale > 0

                    latent_model_input = latents if latents.dtype == dtype else latents.to(dtype)
                    latent_model_input = latent_model_input.unsqueeze(2)

                    if apply_cfg:
                        model_out_list = self.transformer(
                            [latent_model_input, latent_model_input],
                            timestep,
                            [prompt_embeds[0], negative_prompt_embeds[0]] ,
                            control_context_list=[control_latent_input, control_latent_input] if control_latent_input is not None else None,
                            control_context_scale=control_context_scale,
                            callback=callback,
                            pipeline=self,
                            **kwargs,
                        )

                        if model_out_list is None:
                            return None

                        pos, neg = model_out_list
                        model_out_list = None
                        pos, neg = pos.float(), neg.float()

                        noise_pred = pos + current_guidance_scale * (pos - neg)

                        if self._cfg_normalization and float(self._cfg_normalization) > 0.0:
                            ori_pos_norm = torch.linalg.vector_norm(pos)
                            new_pos_norm = torch.linalg.vector_norm(noise_pred)
                            max_new_norm = ori_pos_norm * float(self._cfg_normalization)
                            if new_pos_norm > max_new_norm:
                                noise_pred = noise_pred * (max_new_norm / new_pos_norm)
                        pos = neg = None
                    else:
                        model_out_list = self.transformer(
                            [latent_model_input],
                            timestep,
                            [prompt_embeds[0]],
                            control_context_list=[control_latent_input] if control_latent_input is not None else None,
                            control_context_scale=control_context_scale,
                            callback=callback,
                            pipeline=self,
                            **kwargs,
                        )

                        if model_out_list is None:
                            return None
                        noise_pred = model_out_list[0]
                        model_out_list = None

                    noise_pred = noise_pred.squeeze(2)
                    noise_pred = -noise_pred

                    preview_x0 = None
                    if callback is not None and getattr(callback, "tiny_vae", False):
                        if self.scheduler.step_index is None:
                            self.scheduler._init_step_index(t)
                        preview_x0 = latents - self.scheduler.sigmas[self.scheduler.step_index].to(latents.device) * noise_pred.float()
                    latents = self.scheduler.step(noise_pred.to(torch.float32), t, latents, return_dict=False)[0]
                    assert latents.dtype == torch.float32

                    if self._interrupt:
                        del preview_x0
                        break
                    if callback is not None:
                        latents_preview = (preview_x0 if preview_x0 is not None else latents).unsqueeze(2)
                        if len(latents_preview) > 1:
                            latents_preview = latents_preview.transpose(0, 2)
                        callback(i, latents_preview[0], False)
                        latents_preview = None
                    del preview_x0


        if self._interrupt:
            return None
        
        latents = latents.to(dtype)
        if output_type == "latent":
            image = latents
        else:
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor

            image = self.vae.decode(latents, return_dict=False)[0]
            if vae_upsampler is not None:
                if vae_upsampler_progress_callback is not None:
                    vae_upsampler_progress_callback("VAE")
                image_ref, latents_ref = [image], [latents]
                image = latents = None
                image = vae_upsampler.decode_inputs(
                    image_ref,
                    latents_ref,
                    prompt=prompt,
                    seed=vae_upsampler_seed,
                    abort_callback=lambda: self._interrupt,
                    progress_callback=vae_upsampler_progress_callback,
                )
                if image is None:
                    return None
            if image.dtype == torch.uint8 and output_type == "pt":
                pass
            elif image.dtype == torch.uint8:
                image = image.float().div_(127.5).sub_(1.0)
                image = self.image_processor.postprocess(image, output_type=output_type)
            elif output_type == "pt":
                image = self.image_processor.postprocess(
                    image, output_type=output_type, do_denormalize=[False] * image.shape[0]
                )
            else:
                image = self.image_processor.postprocess(image, output_type=output_type)

        return image

