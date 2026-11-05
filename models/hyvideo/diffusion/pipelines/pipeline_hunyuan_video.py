# Copyright 2024 The HuggingFace Team. All rights reserved.
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
# ==============================================================================
#
# Modified from diffusers==0.29.2
#
# ==============================================================================
# Modifications Copyright (c) 2025-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from shared.utils.phase_progress import text_encoding_prompts, text_encoding_progress
import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import torch
import torch.distributed as dist
import numpy as np
from dataclasses import dataclass
from packaging import version
import re
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import VaeImageProcessor
from diffusers.utils import BaseOutput
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL
from diffusers.models.lora import adjust_lora_scale_text_encoder
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
)
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.utils import BaseOutput

from ...constants import PRECISION_TO_TYPE
from ...vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from ...text_encoder import TextEncoder
from ...modules import HYVideoDiffusionTransformer
from mmgp import offload
from ...utils.data_utils import black_image
from einops import rearrange
from shared.utils.text_encoder_cache import TextEncoderCache

EXAMPLE_DOC_STRING = """"""


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(
        dim=list(range(1, noise_pred_text.ndim)), keepdim=True
    )
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = (
        guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    )
    return noise_cfg


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
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
        raise ValueError(
            "Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values"
        )
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(
            inspect.signature(scheduler.set_timesteps).parameters.keys()
        )
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


@dataclass
class HunyuanVideoPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class HunyuanVideoPipeline(DiffusionPipeline):
    r"""
    Pipeline for text-to-video generation using HunyuanVideo.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines (downloading, saving, running on a particular device, etc.).

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) model to encode and decode images to and from latent representations.
        text_encoder ([`TextEncoder`]):
            Frozen text-encoder.
        text_encoder_2 ([`TextEncoder`]):
            Frozen text-encoder_2.
        transformer ([`HYVideoDiffusionTransformer`]):
            A `HYVideoDiffusionTransformer` to denoise the encoded video latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
    """

    model_cpu_offload_seq = "text_encoder->text_encoder_2->transformer->vae"
    _optional_components = ["text_encoder_2"]
    _exclude_from_cpu_offload = ["transformer"]
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: TextEncoder,
        transformer: HYVideoDiffusionTransformer,
        scheduler: KarrasDiffusionSchedulers,
        text_encoder_2: Optional[TextEncoder] = None,
        byt5_model= None,
        byt5_tokenizer= None,
        byt5_max_length= 256,
        vision_encoder=None,
        upsampler=None,
        prompt_format=None,
        hunyuan_1_5 = None,
        progress_bar_config: Dict[str, Any] = None,
        args=None,
    ):
        super().__init__()

        # ==========================================================================================
        if progress_bar_config is None:
            progress_bar_config = {}
        if not hasattr(self, "_progress_bar_config"):
            self._progress_bar_config = {}
        self._progress_bar_config.update(progress_bar_config)

        self.args = args
        # ==========================================================================================

        if (
            hasattr(scheduler.config, "steps_offset")
            and scheduler.config.steps_offset != 1
        ):
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate(
                "steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if (
            hasattr(scheduler.config, "clip_sample")
            and scheduler.config.clip_sample is True
        ):
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate(
                "clip_sample not set", "1.0.0", deprecation_message, standard_warn=False
            )
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
            text_encoder_2=text_encoder_2,
            byt5_model= byt5_model,
            byt5_tokenizer= byt5_tokenizer,
            vision_encoder=vision_encoder,
            upsampler=upsampler,
        )
        self.byt5_max_length = byt5_max_length
        self.prompt_format= prompt_format
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.noise_pertub = 0
        self.hunyuan_1_5 = hunyuan_1_5
        self.target_dtype = torch.bfloat16 
        self.text_encoder_cache = TextEncoderCache()

    def encode_prompt_1_5(
        self,
        prompt,
        name,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        clip_skip: Optional[int] = None,
        text_encoder: Optional[TextEncoder] = None,
        data_type: Optional[str] = "image",
        pixel_value_llava: Optional[torch.Tensor] = None,
        uncond_pixel_value_llava: Optional[torch.Tensor] = None,                
        lora_scale: Optional[float] = None,
        semantic_images=None
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_videos_per_prompt (`int`):
                number of videos that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the video generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            attention_mask (`torch.Tensor`, *optional*):
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_attention_mask (`torch.Tensor`, *optional*):
                Attention mask for negative prompt embeddings.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            text_encoder (TextEncoder, *optional*):
                Text encoder to use. If None, uses the pipeline's default text encoder.
            data_type (`str`, *optional*):
                Type of data being encoded. Defaults to "image".
        """
        if text_encoder is None:
            text_encoder = self.text_encoder
        positive = [] if prompt_embeds is not None else [prompt] if isinstance(prompt, str) else list(prompt)
        negative = [] if not do_classifier_free_guidance or negative_prompt_embeds is not None else [negative_prompt or ""] if isinstance(negative_prompt, str) or negative_prompt is None else list(negative_prompt)
        prompt_count = len(positive) if False else sum((id(text_encoder), name, data_type, clip_skip, text_encoder.max_length, p) not in self.text_encoder_cache._entries for p in dict.fromkeys(positive))
        prompt_count += len(negative) if False else sum(("uncond", id(text_encoder), name, data_type, text_encoder.max_length, p) not in self.text_encoder_cache._entries for p in dict.fromkeys(negative))
        with text_encoding_prompts(prompt_count):

            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            if prompt_embeds is None:

                def encode_fn(prompts):
                    text_inputs = text_encoder.text2tokens(prompts, data_type=data_type, max_length=text_encoder.max_length)
                    if clip_skip is None:
                        prompt_outputs = text_encoder.encode(text_inputs, data_type=data_type, device=device)
                        prompt_embeds = prompt_outputs.hidden_state
                    else:
                        prompt_outputs = text_encoder.encode(text_inputs, output_hidden_states=True, data_type=data_type, device=device)
                        prompt_embeds = prompt_outputs.hidden_states_list[-(clip_skip + 1)]
                        prompt_embeds = text_encoder.model.text_model.final_layer_norm(prompt_embeds)
                    attention_mask = prompt_outputs.attention_mask
                    if attention_mask is None:
                        return [(prompt_embeds[i], None) for i in range(prompt_embeds.shape[0])]
                    return list(zip(prompt_embeds, attention_mask))

                prompt_list = prompt if isinstance(prompt, list) else [prompt]
                cache_keys = [(id(text_encoder), name, data_type, clip_skip, text_encoder.max_length, text) for text in prompt_list]
                prompt_contexts = self.text_encoder_cache.encode(encode_fn, prompt_list, device=device, cache_keys=cache_keys)
                prompt_embeds = torch.stack([ctx[0] for ctx in prompt_contexts], dim=0)
                attention_mask = prompt_contexts[0][1]
                if attention_mask is not None:
                    attention_mask = torch.stack([ctx[1] for ctx in prompt_contexts], dim=0)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                    bs_embed, seq_len = attention_mask.shape
                    attention_mask = attention_mask.repeat(1, num_videos_per_prompt)
                    attention_mask = attention_mask.view(bs_embed * num_videos_per_prompt, seq_len)

            if text_encoder is not None:
                prompt_embeds_dtype = text_encoder.dtype
            elif self.transformer is not None:
                prompt_embeds_dtype = self.transformer.dtype
            else:
                prompt_embeds_dtype = prompt_embeds.dtype

            prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            if prompt_embeds.ndim == 2:
                bs_embed, _ = prompt_embeds.shape
                # duplicate text embeddings for each generation per prompt, using mps friendly method
                prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
                prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, -1)
            else:
                bs_embed, seq_len, _ = prompt_embeds.shape
                # duplicate text embeddings for each generation per prompt, using mps friendly method
                prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
                prompt_embeds = prompt_embeds.view(
                    bs_embed * num_videos_per_prompt, seq_len, -1
                )

            # get unconditional embeddings for classifier free guidance
            if do_classifier_free_guidance and negative_prompt_embeds is None:
                uncond_tokens: List[str]
                if negative_prompt is None:
                    uncond_tokens = [""] * batch_size
                elif prompt is not None and type(prompt) is not type(negative_prompt):
                    raise TypeError(
                        f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                        f" {type(prompt)}."
                    )
                elif isinstance(negative_prompt, str):
                    uncond_tokens = [negative_prompt]
                elif batch_size != len(negative_prompt):
                    raise ValueError(
                        f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                        f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                        " the batch size of `prompt`."
                    )
                else:
                    uncond_tokens = negative_prompt

                def encode_uncond_fn(prompts):
                    uncond_input = text_encoder.text2tokens(prompts, data_type=data_type, max_length=text_encoder.max_length)
                    negative_prompt_outputs = text_encoder.encode(uncond_input, data_type=data_type, is_uncond=True)
                    negative_prompt_embeds = negative_prompt_outputs.hidden_state
                    negative_attention_mask = negative_prompt_outputs.attention_mask
                    if negative_attention_mask is None:
                        return [(negative_prompt_embeds[i], None) for i in range(negative_prompt_embeds.shape[0])]
                    return list(zip(negative_prompt_embeds, negative_attention_mask))

                cache_keys = [("uncond", id(text_encoder), name, data_type, text_encoder.max_length, text) for text in uncond_tokens]
                negative_contexts = self.text_encoder_cache.encode(encode_uncond_fn, uncond_tokens, device=device, cache_keys=cache_keys)
                negative_prompt_embeds = torch.stack([ctx[0] for ctx in negative_contexts], dim=0)
                negative_attention_mask = negative_contexts[0][1]
                if negative_attention_mask is not None:
                    negative_attention_mask = torch.stack([ctx[1] for ctx in negative_contexts], dim=0)
                if negative_attention_mask is not None:
                    negative_attention_mask = negative_attention_mask.to(device)
                    _, seq_len = negative_attention_mask.shape
                    negative_attention_mask = negative_attention_mask.repeat(1, num_videos_per_prompt)
                    negative_attention_mask = negative_attention_mask.view(batch_size * num_videos_per_prompt, seq_len)

            if do_classifier_free_guidance:
                # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
                seq_len = negative_prompt_embeds.shape[1]

                negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

                if negative_prompt_embeds.ndim == 2:
                    negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt)
                    negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, -1)
                else:
                    negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
                    negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

            return (
                prompt_embeds,
                negative_prompt_embeds,
                attention_mask,
                negative_attention_mask,
            )


    def encode_prompt(
        self,
        prompt,
        name,
        device,
        num_videos_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        pixel_value_llava: Optional[torch.Tensor] = None,
        uncond_pixel_value_llava: Optional[torch.Tensor] = None,                
        prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
        text_encoder: Optional[TextEncoder] = None,
        data_type: Optional[str] = "image",
        semantic_images=None
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_videos_per_prompt (`int`):
                number of videos that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the video generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            pixel_value_llava (`torch.Tensor`, *optional*):
                The image tensor for llava. 
            uncond_pixel_value_llava (`torch.Tensor`, *optional*):
                The image tensor for llava.  Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).                
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            attention_mask (`torch.Tensor`, *optional*):
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            negative_attention_mask (`torch.Tensor`, *optional*):
            lora_scale (`float`, *optional*):
                A LoRA scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            text_encoder (TextEncoder, *optional*):
            data_type (`str`, *optional*):
        """
        if text_encoder is None:
            text_encoder = self.text_encoder
        positive = [] if prompt_embeds is not None else [prompt] if isinstance(prompt, str) else list(prompt)
        negative = [] if not do_classifier_free_guidance or negative_prompt_embeds is not None else [negative_prompt or ""] if isinstance(negative_prompt, str) or negative_prompt is None else list(negative_prompt)
        prompt_count = len(positive) if pixel_value_llava is not None or semantic_images is not None else sum((id(text_encoder), name, data_type, clip_skip, lora_scale, p) not in self.text_encoder_cache._entries for p in dict.fromkeys(positive))
        prompt_count += len(negative) if uncond_pixel_value_llava is not None or semantic_images is not None else sum(("uncond", id(text_encoder), name, data_type, lora_scale, p) not in self.text_encoder_cache._entries for p in dict.fromkeys(negative))
        with text_encoding_prompts(prompt_count):

            # set lora scale so that monkey patched LoRA
            # function of text encoder can correctly access it
            if lora_scale is not None and isinstance(self, LoraLoaderMixin):
                self._lora_scale = lora_scale

                # dynamically adjust the LoRA scale
                if not USE_PEFT_BACKEND:
                    adjust_lora_scale_text_encoder(text_encoder.model, lora_scale)
                else:
                    scale_lora_layers(text_encoder.model, lora_scale)

            if prompt is not None and isinstance(prompt, str):
                batch_size = 1
            elif prompt is not None and isinstance(prompt, list):
                batch_size = len(prompt)
            else:
                batch_size = prompt_embeds.shape[0]

            if prompt_embeds is None:
                # textual inversion: process multi-vector tokens if necessary
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, text_encoder.tokenizer)

                use_cache = pixel_value_llava is None and semantic_images is None
                if use_cache:
                    def encode_fn(prompts):
                        text_inputs = text_encoder.text2tokens(prompts, data_type=data_type, name=name)
                        if clip_skip is None:
                            prompt_outputs = text_encoder.encode(text_inputs, data_type=data_type, semantic_images=None, device=device)
                            prompt_embeds = prompt_outputs.hidden_state
                        else:
                            prompt_outputs = text_encoder.encode(text_inputs, output_hidden_states=True, data_type=data_type, semantic_images=None, device=device)
                            prompt_embeds = prompt_outputs.hidden_states_list[-(clip_skip + 1)]
                            prompt_embeds = text_encoder.model.text_model.final_layer_norm(prompt_embeds)
                        attention_mask = prompt_outputs.attention_mask
                        if attention_mask is None:
                            return [(prompt_embeds[i], None) for i in range(prompt_embeds.shape[0])]
                        return list(zip(prompt_embeds, attention_mask))

                    prompt_list = prompt if isinstance(prompt, list) else [prompt]
                    cache_keys = [(id(text_encoder), name, data_type, clip_skip, lora_scale, text) for text in prompt_list]
                    prompt_contexts = self.text_encoder_cache.encode(encode_fn, prompt_list, device=device, cache_keys=cache_keys)
                    prompt_embeds = torch.stack([ctx[0] for ctx in prompt_contexts], dim=0)
                    attention_mask = prompt_contexts[0][1]
                    if attention_mask is not None:
                        attention_mask = torch.stack([ctx[1] for ctx in prompt_contexts], dim=0)
                else:
                    text_inputs = text_encoder.text2tokens(prompt, data_type=data_type, name=name)

                    if pixel_value_llava is not None:
                        text_inputs['pixel_value_llava'] = pixel_value_llava
                        text_inputs['attention_mask'] = torch.cat([text_inputs['attention_mask'], torch.ones((1, 575 * len(pixel_value_llava))).to(text_inputs['attention_mask'])], dim=1)

                    if clip_skip is None:
                        prompt_outputs = text_encoder.encode(text_inputs, data_type=data_type, semantic_images=semantic_images, device=device)
                        prompt_embeds = prompt_outputs.hidden_state
                    else:
                        prompt_outputs = text_encoder.encode(text_inputs, output_hidden_states=True, data_type=data_type, semantic_images=semantic_images, device=device)
                        prompt_embeds = prompt_outputs.hidden_states_list[-(clip_skip + 1)]
                        prompt_embeds = text_encoder.model.text_model.final_layer_norm(prompt_embeds)

                    attention_mask = prompt_outputs.attention_mask

                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                    bs_embed, seq_len = attention_mask.shape
                    attention_mask = attention_mask.repeat(1, num_videos_per_prompt)
                    attention_mask = attention_mask.view(bs_embed * num_videos_per_prompt, seq_len)

            if text_encoder is not None:
                prompt_embeds_dtype = text_encoder.dtype
            elif self.transformer is not None:
                prompt_embeds_dtype = self.transformer.dtype
            else:
                prompt_embeds_dtype = prompt_embeds.dtype

            prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            if prompt_embeds.ndim == 2:
                bs_embed, _ = prompt_embeds.shape
                # duplicate text embeddings for each generation per prompt, using mps friendly method
                prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
                prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, -1)
            else:
                bs_embed, seq_len, _ = prompt_embeds.shape
                # duplicate text embeddings for each generation per prompt, using mps friendly method
                prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
                prompt_embeds = prompt_embeds.view(
                    bs_embed * num_videos_per_prompt, seq_len, -1
                )

            # get unconditional embeddings for classifier free guidance
            if do_classifier_free_guidance and negative_prompt_embeds is None:
                uncond_tokens: List[str]
                if negative_prompt is None:
                    uncond_tokens = [""] * batch_size
                elif prompt is not None and type(prompt) is not type(negative_prompt):
                    raise TypeError(
                        f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                        f" {type(prompt)}."
                    )
                elif isinstance(negative_prompt, str):
                    uncond_tokens = [negative_prompt]
                elif batch_size != len(negative_prompt):
                    raise ValueError(
                        f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                        f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                        " the batch size of `prompt`."
                    )
                else:
                    uncond_tokens = negative_prompt

                # textual inversion: process multi-vector tokens if necessary
                if isinstance(self, TextualInversionLoaderMixin):
                    uncond_tokens = self.maybe_convert_prompt(
                        uncond_tokens, text_encoder.tokenizer
                    )

                use_cache = uncond_pixel_value_llava is None and semantic_images is None
                if use_cache:
                    def encode_uncond_fn(prompts):
                        uncond_input = text_encoder.text2tokens(prompts, data_type=data_type, name=name)
                        negative_prompt_outputs = text_encoder.encode(uncond_input, data_type=data_type, semantic_images=None, device=device)
                        negative_prompt_embeds = negative_prompt_outputs.hidden_state
                        negative_attention_mask = negative_prompt_outputs.attention_mask
                        if negative_attention_mask is None:
                            return [(negative_prompt_embeds[i], None) for i in range(negative_prompt_embeds.shape[0])]
                        return list(zip(negative_prompt_embeds, negative_attention_mask))

                    cache_keys = [("uncond", id(text_encoder), name, data_type, lora_scale, text) for text in uncond_tokens]
                    negative_contexts = self.text_encoder_cache.encode(encode_uncond_fn, uncond_tokens, device=device, cache_keys=cache_keys)
                    negative_prompt_embeds = torch.stack([ctx[0] for ctx in negative_contexts], dim=0)
                    negative_attention_mask = negative_contexts[0][1]
                    if negative_attention_mask is not None:
                        negative_attention_mask = torch.stack([ctx[1] for ctx in negative_contexts], dim=0)
                else:
                    uncond_input = text_encoder.text2tokens(uncond_tokens, data_type=data_type, name=name)

                    if semantic_images is not None:
                        uncond_image = [black_image(img.size[0], img.size[1]) for img in semantic_images]
                    else:
                        uncond_image = None

                    if uncond_pixel_value_llava is not None:
                        uncond_input['pixel_value_llava'] = uncond_pixel_value_llava
                        uncond_input['attention_mask'] = torch.cat([uncond_input['attention_mask'], torch.ones((1, 575 * len(uncond_pixel_value_llava))).to(uncond_input['attention_mask'])], dim=1)

                    negative_prompt_outputs = text_encoder.encode(uncond_input, data_type=data_type, semantic_images=uncond_image, device=device)
                    negative_prompt_embeds = negative_prompt_outputs.hidden_state
                    negative_attention_mask = negative_prompt_outputs.attention_mask

                if negative_attention_mask is not None:
                    negative_attention_mask = negative_attention_mask.to(device)
                    _, seq_len = negative_attention_mask.shape
                    negative_attention_mask = negative_attention_mask.repeat(1, num_videos_per_prompt)
                    negative_attention_mask = negative_attention_mask.view(batch_size * num_videos_per_prompt, seq_len)

            if do_classifier_free_guidance:
                # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
                seq_len = negative_prompt_embeds.shape[1]

                negative_prompt_embeds = negative_prompt_embeds.to(
                    dtype=prompt_embeds_dtype, device=device
                )

                if negative_prompt_embeds.ndim == 2:
                    negative_prompt_embeds = negative_prompt_embeds.repeat(
                        1, num_videos_per_prompt
                    )
                    negative_prompt_embeds = negative_prompt_embeds.view(
                        batch_size * num_videos_per_prompt, -1
                    )
                else:
                    negative_prompt_embeds = negative_prompt_embeds.repeat(
                        1, num_videos_per_prompt, 1
                    )
                    negative_prompt_embeds = negative_prompt_embeds.view(
                        batch_size * num_videos_per_prompt, seq_len, -1
                    )

            if text_encoder is not None:
                if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
                    # Retrieve the original scale by scaling back the LoRA layers
                    unscale_lora_layers(text_encoder.model, lora_scale)

            return (
                prompt_embeds,
                negative_prompt_embeds,
                attention_mask,
                negative_attention_mask,
            )

    def decode_latents(self, latents, enable_tiling=True):
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        if enable_tiling:
            self.vae.enable_tiling()
            image = self.vae.decode(latents, return_dict=False)[0]
        else:
            image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        if image.ndim == 4:
            image = image.cpu().permute(0, 2, 3, 1).float()
        else:
            image = image.cpu().float()
        return image

    def prepare_extra_func_kwargs(self, func, kwargs):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        extra_step_kwargs = {}

        for k, v in kwargs.items():
            accepts = k in set(inspect.signature(func).parameters.keys())
            if accepts:
                extra_step_kwargs[k] = v
        return extra_step_kwargs

    def check_inputs(
        self,
        prompt,
        height,
        width,
        video_length,
        callback_steps,
        pixel_value_llava=None,
        uncond_pixel_value_llava=None,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        vae_ver="88-4c-sd",
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if video_length is not None:
            if "884" in vae_ver:
                if video_length != 1 and (video_length - 1) % 4 != 0:
                    raise ValueError(
                        f"`video_length` has to be 1 or a multiple of 4 but is {video_length}."
                    )
            elif "888" in vae_ver:
                if video_length != 1 and (video_length - 1) % 8 != 0:
                    raise ValueError(
                        f"`video_length` has to be 1 or a multiple of 8 but is {video_length}."
                    )

        if callback_steps is not None and (
            not isinstance(callback_steps, int) or callback_steps <= 0
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs
            for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (
            not isinstance(prompt, str) and not isinstance(prompt, list)
        ):
            raise ValueError(
                f"`prompt` has to be of type `str` or `list` but is {type(prompt)}"
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        
        if pixel_value_llava is not None and uncond_pixel_value_llava is not None:
            if len(pixel_value_llava) != len(uncond_pixel_value_llava):
                raise ValueError(
                    "`pixel_value_llava` and `uncond_pixel_value_llava` must have the same length when passed directly, but"
                    f" got: `pixel_value_llava` {len(pixel_value_llava)} != `uncond_pixel_value_llava`"
                    f" {len(uncond_pixel_value_llava)}."
                )
            
        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start * self.scheduler.order :]
        if hasattr(self.scheduler, "set_begin_index"):
            self.scheduler.set_begin_index(t_start * self.scheduler.order)

        return timesteps.to(device), num_inference_steps - t_start


    def prepare_latents(
        self,
        batch_size,
        num_channels_latents,
        num_inference_steps,        
        height,
        width,
        video_length,
        dtype,
        device,
        timesteps,
        generator,
        latents=None,
        denoise_strength=1.0,
        img_latents=None,
        i2v_mode=False,
        i2v_condition_type=None,
        i2v_stability=True,
    ):
        shape = (
            batch_size,
            num_channels_latents,
            video_length,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if i2v_mode and i2v_stability:
            if img_latents.shape[2] == 1:
                img_latents = img_latents.repeat(1, 1, video_length, 1, 1)
            x0 = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            x1 = img_latents

            t = torch.tensor([0.999]).to(device=device)
            latents = x0 * t + x1 * (1 - t)
            latents = latents.to(dtype=dtype)

        if denoise_strength == 0:
            if latents is None:
                latents = randn_tensor(
                    shape, generator=generator, device=device, dtype=dtype
                )
            else:
                latents = latents.to(device)   
            original_latents = None
            noise = None
            timesteps = timesteps
        else:
            noise = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, denoise_strength, device)

            if latents is None:
                latents = noise 
                original_latents = None
            else:
                latents = latents.to(device)
                latent_timestep = timesteps[:1]
                frames_needed = noise.shape[2]
                current_frames = latents.shape[2]
                
                if frames_needed > current_frames:
                    repeat_factor = frames_needed - current_frames
                    additional_frame = torch.randn((latents.size(0), latents.size(1),repeat_factor, latents.size(3), latents.size(4)), dtype=latents.dtype, device=latents.device)
                    latents = torch.cat((additional_frame, latents), dim=2)
                    self.additional_frames = repeat_factor
                elif frames_needed < current_frames:
                    latents = latents[:, :, :frames_needed, :, :]
                
                original_latents = latents.clone()
                latents = latents * (1 - latent_timestep / 1000) + latent_timestep / 1000 * noise
                print(f'debug:latent_timestep={latent_timestep}, latents-size={latents.shape}')
       
        # Check existence to make it compatible with FlowMatchEulerDiscreteScheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
        return latents, original_latents, noise, timesteps

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(
        self,
        w: torch.Tensor,
        embedding_dim: int = 512,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        See https://github.com/google-research/vdm/blob/dc27b98a554f65cdc654b800da5aa1846545d41b/model_vdm.py#L298

        Args:
            w (`torch.Tensor`):
                Generate embedding vectors with a specified guidance scale to subsequently enrich timestep embeddings.
            embedding_dim (`int`, *optional*, defaults to 512):
                Dimension of the embeddings to generate.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                Data type of the generated embeddings.

        Returns:
            `torch.Tensor`: Embedding vectors with shape `(len(w), embedding_dim)`.
        """
        assert len(w.shape) == 1
        w = w * 1000.0

        half_dim = embedding_dim // 2
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=dtype) * -emb)
        emb = w.to(dtype)[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:  # zero pad
            emb = torch.nn.functional.pad(emb, (0, 1))
        assert emb.shape == (w.shape[0], embedding_dim)
        return emb

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def guidance_rescale(self):
        return self._guidance_rescale

    @property
    def clip_skip(self):
        return self._clip_skip

    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    @property
    def do_classifier_free_guidance(self):
        # return self._guidance_scale > 1 and self.transformer.config.time_cond_proj_dim is None
        return self._guidance_scale > 1

    @property
    def cross_attention_kwargs(self):
        return self._cross_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    @staticmethod
    def get_byt5_text_tokens(byt5_tokenizer, byt5_max_length, text_prompt):
        """
        Tokenize text prompt for byT5 model.

        Args:
            byt5_tokenizer: The byT5 tokenizer.
            byt5_max_length: Maximum sequence length for tokenization.
            text_prompt: Text prompt string to tokenize.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - input_ids: Tokenized input IDs.
                - attention_mask: Attention mask tensor.
        """
        byt5_text_inputs = byt5_tokenizer(
            text_prompt,
            padding="max_length",
            max_length=byt5_max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        return byt5_text_inputs.input_ids, byt5_text_inputs.attention_mask

    def _extract_glyph_texts(self, prompt):
        """
        Extract glyph texts from prompt using regex pattern.

        Args:
            prompt: Input prompt string containing quoted text.

        Returns:
            List[str]: List of extracted glyph texts (deduplicated if multiple).
        """
        pattern = r'\"(.*?)\"|“(.*?)”'
        matches = re.findall(pattern, prompt)
        result = [match[0] or match[1] for match in matches]
        result = list(dict.fromkeys(result)) if len(result) > 1 else result
        return result

    def _process_single_byt5_prompt(self, prompt_text, device):
        """
        Process a single prompt for byT5 encoding.

        Args:
            prompt_text: The prompt text to process.
            device: Target device for tensors.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - byt5_embeddings: Encoded embeddings tensor.
                - byt5_mask: Attention mask tensor.
        """
        byt5_embeddings = torch.zeros((1, self.byt5_max_length, 1472), device=device)
        byt5_mask = torch.zeros((1, self.byt5_max_length), device=device, dtype=torch.int64)
        
        glyph_texts = self._extract_glyph_texts(prompt_text)
        
        if len(glyph_texts) > 0:
            text_styles = [{'color': None, 'font-family': None} for _ in range(len(glyph_texts))]
            formatted_text = self.prompt_format.format_prompt(glyph_texts, text_styles)
            
            text_ids, text_mask = self.get_byt5_text_tokens(
                self.byt5_tokenizer, self.byt5_max_length, formatted_text
            )
            text_ids = text_ids.to(device=device)
            text_mask = text_mask.to(device=device)
            
            with text_encoding_progress(self.byt5_model.encoder.block, prompt_count=text_ids.shape[0]):
                byt5_outputs = self.byt5_model(text_ids, attention_mask=text_mask.float())
            byt5_embeddings = byt5_outputs[0]
            byt5_mask = text_mask
            
        return byt5_embeddings, byt5_mask

    def _prepare_byt5_embeddings(self, prompts, device):
        """
        Prepare byT5 embeddings for both positive and negative prompts.

        Args:
            prompts: List of prompt strings or single prompt string.
            device: Target device for tensors.

        Returns:
            dict: Dictionary containing:
                - "byt5_text_states": Combined embeddings tensor.
                - "byt5_text_mask": Combined attention mask tensor.
                Returns empty dict if glyph_byT5_v2 is disabled.
        """
        # if not self.config.glyph_byT5_v2:
        #     return {}
            
        if isinstance(prompts, str):
            prompt_list = [prompts]
        elif isinstance(prompts, list):
            prompt_list = prompts
        else:
            raise ValueError("prompts must be str or list of str")

        positive_embeddings = []
        positive_masks = []
        negative_embeddings = []
        negative_masks = []

        for prompt in prompt_list:
            pos_emb, pos_mask = self._process_single_byt5_prompt(prompt, device)
            positive_embeddings.append(pos_emb)
            positive_masks.append(pos_mask)

            if self.do_classifier_free_guidance:
                neg_emb, neg_mask = self._process_single_byt5_prompt("", device)
                negative_embeddings.append(neg_emb)
                negative_masks.append(neg_mask)

        byt5_positive = torch.cat(positive_embeddings, dim=0)
        byt5_positive_mask = torch.cat(positive_masks, dim=0)
        
        if self.do_classifier_free_guidance:
            byt5_negative = torch.cat(negative_embeddings, dim=0)
            byt5_negative_mask = torch.cat(negative_masks, dim=0)
            
            byt5_embeddings = torch.cat([byt5_negative, byt5_positive], dim=0)
            byt5_masks = torch.cat([byt5_negative_mask, byt5_positive_mask], dim=0)
        else:
            byt5_embeddings = byt5_positive
            byt5_masks = byt5_positive_mask

        return byt5_embeddings, byt5_masks 


    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: int,
        width: int,
        video_length: int,
        name: Union[str, List[str]] = None,        
        data_type: str = "video",
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        pixel_value_ref=None,
        # ref_latents: Optional[torch.Tensor] = None,
        # uncond_ref_latents: Optional[torch.Tensor] = None,
        pixel_value_llava: Optional[torch.Tensor] = None,
        uncond_pixel_value_llava: Optional[torch.Tensor] = None,
        bg_latents: Optional[torch.Tensor] = None,
        audio_prompts: Optional[torch.Tensor] = None,
        ip_cfg_scale: float = 0.0,
        audio_strength: float = 1.0,
        use_deepcache: int = 1,
        num_videos_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_attention_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        clip_skip: Optional[int] = None,
        callback_on_step_end: Optional[
            Union[
                Callable[[int, int, Dict], None],
                PipelineCallback,
                MultiPipelineCallbacks,
            ]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        vae_ver: str = "88-4c-sd",
        enable_tiling: bool = False,
        n_tokens: Optional[int] = None,
        video_val_flag: bool=False,
        denoise_strength: float = 1.0,
        mask = None,        
        embedded_guidance_scale: Optional[float] = None,
        i2v_mode: bool = False,
        i2v_condition_type: str = None,
        i2v_stability: bool = True,
        img_latents: Optional[torch.Tensor] = None,
        semantic_images=None,
        joint_pass = False,
        cfg_star_rescale = False,
        callback = None,
        **kwargs,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`):
                The height in pixels of the generated image.
            width (`int`):
                The width in pixels of the generated image.
            video_length (`int`):
                The number of frames in the generated video.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            timesteps (`List[int]`, *optional*):
                Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
                in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
                passed will be used. Must be in descending order.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            ref_latents (`torch.Tensor`, *optional*):
                The image tensor for time-concat.
            uncond_ref_latents (`torch.Tensor`, *optional*):
                The image tensor for time-concat. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            pixel_value_llava (`torch.Tensor`, *optional*):
                The image tensor for llava. 
            uncond_pixel_value_llava (`torch.Tensor`, *optional*):
                The image tensor for llava.  Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
                
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`HunyuanVideoPipelineOutput`] instead of a
                plain tuple.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.0):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            clip_skip (`int`, *optional*):
                Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
                the output of the pre-final layer will be used for computing the prompt embeddings.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.

        Examples:

        Returns:
            [`~HunyuanVideoPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`HunyuanVideoPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        callback_steps = kwargs.pop("callback_steps", None)

        # if callback is not None:
        #     deprecate(
        #         "callback",
        #         "1.0.0",
        #         "Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
        #     )
        # if callback_steps is not None:
        #     deprecate(
        #         "callback_steps",
        #         "1.0.0",
        #         "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
        #     )


        if self._interrupt:
            return [None]
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        if pixel_value_ref != None:
            pixel_value_ref = pixel_value_ref * 2 - 1.
            pixel_value_ref_for_vae = rearrange(pixel_value_ref,"b c h w -> b c 1 h w")

            ref_latents = self.vae.encode(pixel_value_ref_for_vae.clone()).latent_dist.sample()
            uncond_ref_latents = self.vae.encode(torch.ones_like(pixel_value_ref_for_vae)).latent_dist.sample()
            ref_latents.mul_(self.vae.config.scaling_factor)
            uncond_ref_latents.mul_(self.vae.config.scaling_factor)
        else:
            ref_latents = None
            uncond_ref_latents = None



        # 0. Default height and width to unet
        # height = height or self.transformer.config.sample_size * self.vae_scale_factor
        # width = width or self.transformer.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks
        trans = self.transformer
        skip_steps_cache = trans.cache
        if skip_steps_cache != None:
            cache_type = skip_steps_cache.cache_type 
            if  cache_type == "tea":
                teacache_multiplier = skip_steps_cache.multiplier
                skip_steps_cache.accumulated_rel_l1_distance = 0
                skip_steps_cache.rel_l1_thresh = 0.1 if teacache_multiplier < 2 else 0.15
            elif cache_type== "mag":
                trans.compute_magcache_threshold(skip_steps_cache.start_step, num_inference_steps, skip_steps_cache.multiplier)
                skip_steps_cache.accumulated_err, skip_steps_cache.accumulated_steps, skip_steps_cache.accumulated_ratio  = 0, 0, 1.0
            else:
                trans.cache = None
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            video_length,
            callback_steps,
            negative_prompt,
            pixel_value_llava,
            uncond_pixel_value_llava,            
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            vae_ver=vae_ver,
        )

        self._guidance_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs


        vae_precision = "fp16" # torch.float16
        precision = "bf16" # torch.bfloat16

        disable_autocast =  True
        target_dtype = PRECISION_TO_TYPE[precision]
        autocast_enabled = target_dtype != torch.float32 and not disable_autocast
        vae_dtype = self.vae._model_dtype # PRECISION_TO_TYPE[vae_precision]
        vae_autocast_enabled = vae_dtype != torch.float32 and False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = torch.device(f"cuda:{dist.get_rank()}") if dist.is_initialized() else self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None)
            if self.cross_attention_kwargs is not None
            else None
        )
        hunyuan_1_5= self.hunyuan_1_5
        encode_prompt = self.encode_prompt_1_5 if hunyuan_1_5 else self.encode_prompt
        (
            prompt_embeds,
            negative_prompt_embeds,
            prompt_mask,
            negative_prompt_mask,
        ) = encode_prompt(
            prompt,
            name,            
            device,
            num_videos_per_prompt,
            self.do_classifier_free_guidance,
            negative_prompt,
            pixel_value_llava=pixel_value_llava,
            uncond_pixel_value_llava=uncond_pixel_value_llava,            
            prompt_embeds=prompt_embeds,
            attention_mask=attention_mask,
            negative_prompt_embeds=negative_prompt_embeds,
            negative_attention_mask=negative_attention_mask,
            lora_scale=lora_scale,
            clip_skip=self.clip_skip,
            data_type=data_type,
            semantic_images=semantic_images
        )
        if self.text_encoder_2 is not None:
            (
                prompt_embeds_2,
                negative_prompt_embeds_2,
                prompt_mask_2,
                negative_prompt_mask_2,
            ) = encode_prompt(
                prompt,
                name,
                device,
                num_videos_per_prompt,
                self.do_classifier_free_guidance,
                negative_prompt,
                prompt_embeds=None,
                attention_mask=None,
                negative_prompt_embeds=None,
                negative_attention_mask=None,
                lora_scale=lora_scale,
                clip_skip=self.clip_skip,
                text_encoder=self.text_encoder_2,
                data_type=data_type,
            )
        else:
            prompt_embeds_2 = None
            negative_prompt_embeds_2 = None
            prompt_mask_2 = None
            negative_prompt_mask_2 = None

        if self.byt5_model is not None:
            byt5_embeddings, byt5_masks = self._prepare_byt5_embeddings(prompt, device)
        else:
            byt5_embeddings = byt5_masks = None


        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
            if prompt_mask is not None:
                prompt_mask = torch.cat([negative_prompt_mask, prompt_mask])
            if prompt_embeds_2 is not None:
                prompt_embeds_2 = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
            if prompt_mask_2 is not None:
                prompt_mask_2 = torch.cat([negative_prompt_mask_2, prompt_mask_2])

        if self.do_classifier_free_guidance:
            if ref_latents is not None:
                ref_latents = torch.cat([ref_latents, ref_latents], dim=0)
                if prompt_mask[0].sum() > 575:
                    prompt_mask[0] = torch.cat([torch.ones((1, prompt_mask[0].sum() - 575)).to(prompt_mask), 
                                                torch.zeros((1, prompt_mask.shape[1] - prompt_mask[0].sum() + 575)).to(prompt_mask)], dim=1)

            if bg_latents is not None and self.transformer.custom:
                bg_latents = torch.cat([bg_latents, bg_latents], dim=0)

            if audio_prompts is not None:
                audio_prompts = torch.cat([torch.zeros_like(audio_prompts), audio_prompts], dim=0)

        if ip_cfg_scale>0:
            prompt_embeds = torch.cat([prompt_embeds, prompt_embeds[1:]])
            prompt_embeds_2 = torch.cat([prompt_embeds_2, prompt_embeds_2[1:]])
            prompt_mask = torch.cat([prompt_mask, prompt_mask[1:]], dim=0)
            ref_latents = torch.cat([uncond_ref_latents, uncond_ref_latents, ref_latents[1:]], dim=0)


        # 4. Prepare timesteps
        extra_set_timesteps_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.set_timesteps, {"n_tokens": n_tokens}
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            timesteps,
            sigmas,
            **extra_set_timesteps_kwargs,
        )

        if "884" in vae_ver:
            video_length = (video_length - 1) // 4 + 1
        elif "888" in vae_ver:
            video_length = (video_length - 1) // 8 + 1
        else:
            video_length = video_length

        if self.transformer.mixed_precision:
            latent_dtype = torch.float32
        else:
            latent_dtype = torch.bfloat16
        if prompt_embeds != None:
            prompt_embeds = prompt_embeds.to(torch.bfloat16)
        if prompt_embeds_2 != None:
            prompt_embeds_2 = prompt_embeds_2.to(torch.bfloat16)

        if self.upsampler is not None:
            # lq_latents = rearrange(bg_latents, "b c f h w -> (b f) c h w")
            # bg_latents = None
            # lq_latents = torch.nn.functional.interpolate(lq_latents, size=[height // 16, width // 16], mode="bilinear", align_corners=False)
            # lq_latents = rearrange(lq_latents, "(b f) c h w -> b c f h w", b=1)            
            # lq_latents = self.upsampler(lq_latents.to(target_dtype))
            lq_mask_latents = torch.ones_like(bg_latents[:, :1])
            extra_latent_frames = video_length  - bg_latents.shape[2]
            if extra_latent_frames:
                bg_latents = torch.cat([bg_latents, torch.zeros_like(bg_latents[:, :, :extra_latent_frames])], dim=2)
                lq_mask_latents = torch.cat([lq_mask_latents, torch.zeros_like(bg_latents[:, :1, :extra_latent_frames])], dim=2)
            lq_latents = self.upsampler(bg_latents.to(target_dtype))
            noise_scale = 0.7
            lq_latents = (1-noise_scale) * lq_latents + noise_scale * torch.randn_like(lq_latents)
            lq_latents =  torch.cat( [lq_latents, lq_mask_latents ], dim=1 )
            zero_lq_latents =  torch.zeros_like(lq_latents)
                
        # 5. Prepare latent variables

        num_channels_latents  = self.transformer.config.out_channels
        latents, original_latents, noise, timesteps = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            num_inference_steps,            
            height,
            width,
            video_length,
            latent_dtype, #prompt_embeds.dtype,
            device,
            timesteps,            
            generator,
            latents,
            denoise_strength,            
            img_latents=img_latents,
            i2v_mode=i2v_mode,
            i2v_condition_type=i2v_condition_type,
            i2v_stability=i2v_stability
        )

        if i2v_condition_type == "latent_concat" :
            if i2v_mode:
                if img_latents.shape[2] == 1:
                    img_latents_concat = img_latents.repeat(1, 1, video_length, 1, 1)
                else:
                    img_latents_concat = img_latents
                img_latents_concat[:, :, 1:] = 0
                mask_concat = torch.ones(img_latents_concat.shape[0], 1, *img_latents_concat.shape[2:], device=img_latents.device)
                mask_concat[:, :, 1:] = 0
            else:
                img_latents_concat = torch.zeros(latents.shape[0], latents.shape[1], *latents.shape[2:], device=latents.device)
                mask_concat = torch.zeros(img_latents_concat.shape[0], 1, *latents.shape[2:], device=latents.device)


        vision_states = None
        if semantic_images is not None and self.vision_encoder is not None:
            vision_states = self.vision_encoder.encode_images(np.array(semantic_images))
            vision_states = vision_states.last_hidden_state.to(device=device, dtype=target_dtype)


        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step,
            {"generator": generator, "eta": eta},
        )

            

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        start_scale = ip_cfg_scale  #  3.0
        end_scale = 1.0
        step_scale = (start_scale - end_scale) / (self._num_timesteps - 1 + 1e-3)

        # print('sigmas used in generation:', self.scheduler.sigmas)
        # print('inference timesteps used in generation:', timesteps)


        # 8. Mask latents
        mask_latents = None
        if mask is not None:
            target_video_length = mask.shape[0]
            target_height = mask.shape[1]
            target_width = mask.shape[2]

            mask_length = (target_video_length - 1) // 4 + 1
            mask_height = target_height // 8
            mask_width = target_width // 8

            mask = mask[...,0:1]
            mask = mask.unsqueeze(0)
            mask = rearrange(mask, "b t h w c -> b c t h w")
            
            mask_latents = torch.nn.functional.interpolate(mask, size=(mask_length, mask_height, mask_width))
            mask_latents = mask_latents.to(device)

        if mask_latents is not None:
            mask_latents_model_input = (
                torch.cat([mask_latents] * 2)
                if self.do_classifier_free_guidance
                else mask_latents
            )
            print(f'maskinfo, mask={mask.shape}, mask_latents_model_input={mask_latents_model_input.shape} ')


        if callback != None:
            callback(-1, None, True)

        load_latent = True
        load_latent = False

        multi_passes_free_guidance = not joint_pass
        if load_latent:
            timesteps = []

        latent_items = 2 if self.do_classifier_free_guidance else 1
        if ip_cfg_scale>0:
            latent_items += 1

        if skip_steps_cache != None:
            skip_steps_cache.previous_residual = [None] * latent_items

        # if is_progress_bar:
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                offload.set_step_no_for_lora(self.transformer, i)
                if self.interrupt:
                    continue

                timesteps_r = None
                if self.upsampler is not None:
                    if i == len(timesteps) - 1:
                        timesteps_r = torch.tensor([0.0], device=self.device)
                    else:
                        timesteps_r = timesteps[i + 1]
                    timesteps_r = timesteps_r.repeat(latents.shape[0])

                if i2v_condition_type == "latent_concat":
                    if self.upsampler is not None:
                        latent_model_input = torch.concat([latents, img_latents_concat, mask_concat, zero_lq_latents if t < 1000 * noise_scale else lq_latents], dim=1)
                    else:
                        latent_model_input = torch.concat([latents, img_latents_concat, mask_concat], dim=1)
                elif i2v_condition_type == "token_replace":
                    latent_model_input = torch.concat([img_latents, latents[:, :, 1:, :, :]], dim=2)
                else:
                    latent_model_input = latents

                latent_model_input =  torch.cat([latent_model_input] * latent_items) if latent_items > 1 else latent_model_input                     
 
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                if mask_latents is not None:
                    original_latents_noise = original_latents * (1 - t / 1000.0) + t / 1000.0 * noise
                    original_latent_noise_model_input = (
                        torch.cat([original_latents_noise] * 2)
                        if self.do_classifier_free_guidance
                        else original_latents_noise
                    )
                    original_latent_noise_model_input = self.scheduler.scale_model_input(original_latent_noise_model_input, t)
                    latent_model_input = mask_latents_model_input * latent_model_input + (1 - mask_latents_model_input) * original_latent_noise_model_input

                t_expand = t.repeat(latent_model_input.shape[0])
                guidance_expand = (
                    torch.tensor(
                        [embedded_guidance_scale] * latent_model_input.shape[0],
                        dtype=torch.float32,
                        device=device,
                    ).to(latent_dtype)
                    * 1000.0
                    if embedded_guidance_scale is not None
                    else None
                )
 
                # predict the noise residual
                with torch.autocast(
                    device_type="cuda", dtype=target_dtype, enabled=autocast_enabled
                ):
                    
                    if self.do_classifier_free_guidance and multi_passes_free_guidance:
                        for j in range(len(latent_model_input)):
                            ret = self.transformer(  # For an input image (129, 192, 336) (1, 256, 256)
                                latent_model_input[j].unsqueeze(0),  # [2, 16, 33, 24, 42]
                                t_expand[j].unsqueeze(0),  # [2]
                                text_states=prompt_embeds[j].unsqueeze(0),  # [2, 256, 4096]
                                text_mask=prompt_mask[j].unsqueeze(0),  # [2, 256]
                                text_states_2=None if prompt_embeds_2 is None else prompt_embeds_2[j].unsqueeze(0) ,  # [2, 768]
                                ref_latents=ref_latents[j].unsqueeze(0),
                                freqs_cos=freqs_cis[0],  # [seqlen, head_dim]
                                freqs_sin=freqs_cis[1],  # [seqlen, head_dim]
                                guidance=guidance_expand,
                                pipeline=self,
                                x_id=j,
                                step_no=i,
                                bg_latents=bg_latents[j].unsqueeze(0) if bg_latents!=None else None,
                                audio_prompts=audio_prompts[j].unsqueeze(0) if audio_prompts!=None else None,
                                audio_strength=audio_strength,
                                vision_states= vision_states,
                                byt5_text_states=byt5_embeddings[j].unsqueeze(0),
                                byt5_text_mask=byt5_masks[j].unsqueeze(0),
                                timesteps_r=timesteps_r,
                                callback = callback,
                            )
                            if self._interrupt:
                                return [None]
                            if j==0:
                                noise_pred_uncond= ret[0]
                            elif j==1:
                                noise_pred_text= ret[0]
                            else:
                                noise_pred_ip = ret[0]
                            ret = None
                    else:
                        ret = self.transformer(  # For an input image (129, 192, 336) (1, 256, 256)
                            latent_model_input,  # [2, 16, 33, 24, 42]
                            t_expand,  # [2]
                            text_states=prompt_embeds,  # [2, 256, 4096]
                            text_mask=prompt_mask,  # [2, 256]
                            text_states_2=prompt_embeds_2,  # [2, 768]
                            ref_latents=ref_latents,
                            freqs_cos=freqs_cis[0],  # [seqlen, head_dim]
                            freqs_sin=freqs_cis[1],  # [seqlen, head_dim]
                            guidance=guidance_expand,
                            pipeline=self,
                            step_no=i,
                            bg_latents=bg_latents,
                            audio_prompts=audio_prompts,
                            audio_strength=audio_strength,
                            vision_states= vision_states,
                            byt5_text_states=byt5_embeddings,
                            byt5_text_mask=byt5_masks,
                            timesteps_r=timesteps_r,
                            callback = callback,
                        )
                        if self._interrupt:
                            return [None]
                        if self.do_classifier_free_guidance :
                            if ip_cfg_scale > 0:
                                noise_pred_uncond, noise_pred_text, noise_pred_ip = ret
                            else:
                                noise_pred_uncond, noise_pred_text = noise_pred = ret
                        else:
                            noise_pred = ret[0]

                # perform guidance
                if self.do_classifier_free_guidance:
                    if cfg_star_rescale:
                        batch_size = 1 
                        positive_flat = noise_pred_text.view(batch_size, -1)
                        negative_flat = noise_pred_uncond.view(batch_size, -1)
                        dot_product = torch.sum(
                            positive_flat * negative_flat, dim=1, keepdim=True
                        )
                        squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
                        positive_flat, negative_flat = None, None
                        alpha = dot_product / squared_norm
                        noise_pred_uncond *= alpha 

                    if ip_cfg_scale > 0:
                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond) + start_scale * (noise_pred_ip-noise_pred_text)
                        start_scale -= step_scale
                        if i==0:
                            print(f'i={i}, noise_pred shape={noise_pred.shape}')
                    else:
                        noise_pred = noise_pred_uncond + self.guidance_scale * ( noise_pred_text - noise_pred_uncond)

                        
                    if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg( noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale, )

                # compute the previous noisy sample x_t -> x_t-1
                preview_latents = None
                if getattr(callback, "tiny_vae", False):
                    preview_step_index = self.scheduler.step_index
                    if preview_step_index is None:
                        preview_step_index = self.scheduler.begin_index
                    if preview_step_index is None:
                        preview_step_index = self.scheduler.index_for_timestep(t)
                    preview_sigma = self.scheduler.sigmas[preview_step_index]
                    if i2v_mode and i2v_condition_type == "token_replace":
                        preview_latents = torch.cat([img_latents, latents[:, :, 1:].float() - preview_sigma * noise_pred.unsqueeze(0)[:, :, 1:].float()], dim=2)
                    else:
                        preview_latents = latents.float() - preview_sigma * noise_pred.float()
                if i2v_mode and i2v_condition_type == "token_replace":
                    noise_pred = noise_pred.unsqueeze(0)
                    latents = self.scheduler.step(
                        noise_pred[:, :, 1:, :, :], t, latents[:, :, 1:, :, :], **extra_step_kwargs, return_dict=False
                    )[0]
                    latents = torch.concat(
                        [img_latents, latents], dim=2
                    )
                else:
                    latents = self.scheduler.step(
                        noise_pred, t, latents, **extra_step_kwargs, return_dict=False
                    )[0]


                noise_pred_uncond, noise_pred_text, noise_pred, noise_pred_ip, ret = None, None, None, None, None

                if callback is not None:
                    callback(i, (preview_latents if preview_latents is not None else latents).squeeze(0), False)
                    preview_latents = None

        if self.interrupt:
            return [None]
        
        if load_latent:
            latents = torch.load("latent.pt")
        else:
            torch.save(latents, "latent.pt")


        if mask_latents is not None:
            latents = mask_latents * latents + (1 - mask_latents) * original_latents

        if not output_type == "latent":
            expand_temporal_dim = False
            if len(latents.shape) == 4:
                if isinstance(self.vae, AutoencoderKLCausal3D):
                    latents = latents.unsqueeze(2)
                    expand_temporal_dim = True
            elif len(latents.shape) == 5:
                pass
            else:
                raise ValueError(
                    f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), but got {latents.shape}."
                )

            if (
                hasattr(self.vae.config, "shift_factor")
                and self.vae.config.shift_factor
            ):
                latents = (
                    latents / self.vae.config.scaling_factor
                    + self.vae.config.shift_factor
                )
            else:
                latents = latents / self.vae.config.scaling_factor

            with torch.autocast(
                device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled
            ):
                if enable_tiling:
                    self.vae.enable_tiling()
                    image = self.vae.decode(
                        latents, return_dict=False, generator=generator
                    )[0]
                else:
                    image = self.vae.decode(
                        latents, return_dict=False, generator=generator
                    )[0]

            if expand_temporal_dim or image.shape[2] == 1:
                image = image.squeeze(2)

        else:
            image = latents

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().float()

        # if i2v_mode and i2v_condition_type == "latent_concat":
        #     image = image[:, :, 4:, :, :]


        if not return_dict:
            return image

        return HunyuanVideoPipelineOutput(videos=image)
