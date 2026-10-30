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
from shared.utils.phase_progress import text_encoding_prompts
import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
import numpy as np
import torch
from packaging import version
from diffusers.utils import BaseOutput
from dataclasses import dataclass
from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.configuration_utils import FrozenDict
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.loaders import LoraLoaderMixin, TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL, ImageProjection
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

from ...constants import PRECISION_TO_TYPE
from ...vae.autoencoder_kl_causal_3d import AutoencoderKLCausal3D
from ...text_encoder import TextEncoder
from einops import rearrange
from ...modules import HYVideoDiffusionTransformer
from shared.utils.text_encoder_cache import TextEncoderCache

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

EXAMPLE_DOC_STRING = """"""


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
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

@dataclass
class HunyuanVideoPipelineOutput(BaseOutput):
    videos: Union[torch.Tensor, np.ndarray]


class HunyuanVideoAudioPipeline(DiffusionPipeline):
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
        progress_bar_config: Dict[str, Any] = None,
        args=None,
    ):
        super().__init__()

        # ==========================================================================================
        if progress_bar_config is None:
            progress_bar_config = {}
        if not hasattr(self, '_progress_bar_config'):
            self._progress_bar_config = {}
        self._progress_bar_config.update(progress_bar_config)

        self.args = args
        # ==========================================================================================

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            transformer=transformer,
            scheduler=scheduler,
            text_encoder_2=text_encoder_2
        )
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.text_encoder_cache = TextEncoderCache()
        
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
    ):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            device: (`torch.device`):
                torch device
            num_videos_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
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
        """
        if text_encoder is None:
            text_encoder = self.text_encoder
        positive = [] if prompt_embeds is not None else [prompt] if isinstance(prompt, str) else list(prompt)
        negative = [] if not do_classifier_free_guidance or negative_prompt_embeds is not None else [negative_prompt or ""] if isinstance(negative_prompt, str) or negative_prompt is None else list(negative_prompt)
        prompt_count = len(positive) if pixel_value_llava is not None else sum((id(text_encoder), name, data_type, clip_skip, lora_scale, p) not in self.text_encoder_cache._entries for p in dict.fromkeys(positive))
        prompt_count += len(negative) if uncond_pixel_value_llava is not None else sum(("uncond", id(text_encoder), name, data_type, lora_scale, p) not in self.text_encoder_cache._entries for p in dict.fromkeys(negative))
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

                use_cache = pixel_value_llava is None
                if use_cache:
                    def encode_fn(prompts):
                        text_inputs = text_encoder.text2tokens(prompts, data_type=data_type, name=name)
                        if clip_skip is None:
                            prompt_outputs = text_encoder.encode(text_inputs, data_type=data_type)
                            prompt_embeds = prompt_outputs.hidden_state
                        else:
                            prompt_outputs = text_encoder.encode(text_inputs, output_hidden_states=True, data_type=data_type)
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
                        prompt_outputs = text_encoder.encode(text_inputs, data_type=data_type)
                        prompt_embeds = prompt_outputs.hidden_state
                    else:
                        prompt_outputs = text_encoder.encode(text_inputs, output_hidden_states=True, data_type=data_type)
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
                prompt_embeds = prompt_embeds.view(bs_embed * num_videos_per_prompt, seq_len, -1)

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
                    uncond_tokens = self.maybe_convert_prompt(uncond_tokens, text_encoder.tokenizer)

                use_cache = uncond_pixel_value_llava is None
                if use_cache:
                    def encode_uncond_fn(prompts):
                        uncond_input = text_encoder.text2tokens(prompts, data_type=data_type)
                        negative_prompt_outputs = text_encoder.encode(uncond_input, data_type=data_type)
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
                    uncond_input = text_encoder.text2tokens(uncond_tokens, data_type=data_type)
                    if uncond_pixel_value_llava is not None:
                        uncond_input['pixel_value_llava'] = uncond_pixel_value_llava
                        uncond_input['attention_mask'] = torch.cat([uncond_input['attention_mask'], torch.ones((1, 575 * len(uncond_pixel_value_llava))).to(uncond_input['attention_mask'])], dim=1)

                    negative_prompt_outputs = text_encoder.encode(uncond_input, data_type=data_type)
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

                negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

                if negative_prompt_embeds.ndim == 2:
                    negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt)
                    negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, -1)
                else:
                    negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_videos_per_prompt, 1)
                    negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

            if text_encoder is not None:
                if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
                    # Retrieve the original scale by scaling back the LoRA layers
                    unscale_lora_layers(text_encoder.model, lora_scale)

            return prompt_embeds, negative_prompt_embeds, attention_mask, negative_attention_mask

    def encode_prompt_audio_text_base(
        self, 
        prompt, 
        uncond_prompt, 
        pixel_value_llava, 
        uncond_pixel_value_llava, 
        device,
        num_images_per_prompt,
        do_classifier_free_guidance,
        negative_prompt=None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        lora_scale: Optional[float] = None,
        clip_skip: Optional[int] = None,
        text_encoder: Optional[TextEncoder] = None,
        data_type: Optional[str] = "image",
        name = "person"
    ):
        if text_encoder is None:
            text_encoder = self.text_encoder
        positive = [] if prompt_embeds is not None else [prompt] if isinstance(prompt, str) else list(prompt)
        negative = [] if not do_classifier_free_guidance or negative_prompt_embeds is not None else [negative_prompt or ""] if isinstance(negative_prompt, str) or negative_prompt is None else list(negative_prompt)
        prompt_count = len(positive) if pixel_value_llava is not None else sum((id(text_encoder), name, data_type, clip_skip, lora_scale, p) not in self.text_encoder_cache._entries for p in dict.fromkeys(positive))
        prompt_count += len(negative) if uncond_pixel_value_llava is not None else sum(("uncond", id(text_encoder), name, data_type, lora_scale, p) not in self.text_encoder_cache._entries for p in dict.fromkeys(negative))
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
            
            prompt_embeds = None
        
            if prompt_embeds is None:
                # textual inversion: process multi-vector tokens if necessary
                if isinstance(self, TextualInversionLoaderMixin):
                    prompt = self.maybe_convert_prompt(prompt, text_encoder.tokenizer)

                use_cache = pixel_value_llava is None
                if use_cache:
                    def encode_fn(prompts):
                        text_inputs = text_encoder.text2tokens(prompts, data_type=data_type, name=name)
                        if clip_skip is None:
                            prompt_outputs = text_encoder.encode(text_inputs, data_type=data_type)
                            prompt_embeds = prompt_outputs.hidden_state
                        else:
                            prompt_outputs = text_encoder.encode(text_inputs, output_hidden_states=True, data_type=data_type)
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
                    text_inputs = text_encoder.text2tokens(prompt, data_type=data_type, name=name) # data_type: video, text_inputs: {'input_ids', 'attention_mask'}

                    text_keys = ['input_ids', 'attention_mask']

                    if pixel_value_llava is not None:
                        text_inputs['pixel_value_llava'] = pixel_value_llava
                        text_inputs['attention_mask'] = torch.cat([text_inputs['attention_mask'], torch.ones((1, 575)).to(text_inputs['attention_mask'])], dim=1)

                    if clip_skip is None:
                        prompt_outputs = text_encoder.encode(text_inputs, data_type=data_type)
                        prompt_embeds = prompt_outputs.hidden_state
                    else:
                        prompt_outputs = text_encoder.encode(text_inputs, output_hidden_states=True, data_type=data_type)
                        prompt_embeds = prompt_outputs.hidden_states_list[-(clip_skip + 1)]
                        prompt_embeds = text_encoder.model.text_model.final_layer_norm(prompt_embeds)

                    attention_mask = prompt_outputs.attention_mask

                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                    bs_embed, seq_len = attention_mask.shape
                    attention_mask = attention_mask.repeat(1, num_images_per_prompt)
                    attention_mask = attention_mask.view(bs_embed * num_images_per_prompt, seq_len)

            if text_encoder is not None:
                prompt_embeds_dtype = text_encoder.dtype
            elif self.unet is not None:
                prompt_embeds_dtype = self.unet.dtype
            else:
                prompt_embeds_dtype = prompt_embeds.dtype

            prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

            if prompt_embeds.ndim == 2:
                bs_embed, _ = prompt_embeds.shape
                # duplicate text embeddings for each generation per prompt, using mps friendly method
                prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
                prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, -1)
            else:
                bs_embed, seq_len, _ = prompt_embeds.shape
                # duplicate text embeddings for each generation per prompt, using mps friendly method
                prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
                prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

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
                    uncond_tokens = self.maybe_convert_prompt(uncond_tokens, text_encoder.tokenizer)

                use_cache = uncond_pixel_value_llava is None
                if use_cache:
                    def encode_uncond_fn(prompts):
                        uncond_input = text_encoder.text2tokens(prompts, data_type=data_type, name=name)
                        negative_prompt_outputs = text_encoder.encode(uncond_input, data_type=data_type)
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

                    if uncond_pixel_value_llava is not None:
                        uncond_input['pixel_value_llava'] = uncond_pixel_value_llava
                        uncond_input['attention_mask'] = torch.cat([uncond_input['attention_mask'], torch.ones((1, 575)).to(uncond_input['attention_mask'])], dim=1)

                    negative_prompt_outputs = text_encoder.encode(uncond_input, data_type=data_type)
                    negative_prompt_embeds = negative_prompt_outputs.hidden_state
                    negative_attention_mask = negative_prompt_outputs.attention_mask

                if negative_attention_mask is not None:
                    negative_attention_mask = negative_attention_mask.to(device)
                    _, seq_len = negative_attention_mask.shape
                    negative_attention_mask = negative_attention_mask.repeat(1, num_images_per_prompt)
                    negative_attention_mask = negative_attention_mask.view(batch_size * num_images_per_prompt, seq_len)

            if do_classifier_free_guidance:
                # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
                seq_len = negative_prompt_embeds.shape[1]

                negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)

                if negative_prompt_embeds.ndim == 2:
                    negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt)
                    negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, -1)
                else:
                    negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
                    negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            if text_encoder is not None:
                if isinstance(self, LoraLoaderMixin) and USE_PEFT_BACKEND:
                    # Retrieve the original scale by scaling back the LoRA layers
                    unscale_lora_layers(text_encoder.model, lora_scale)

            return prompt_embeds, negative_prompt_embeds, attention_mask, negative_attention_mask

    def decode_latents(self, latents, enable_tiling=True):
        deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        if enable_tiling:
            self.vae.enable_tiling()
            image = self.vae.decode(latents, return_dict=False)[0]
            self.vae.disable_tiling()
        else:
            image = self.vae.decode(latents, return_dict=False)[0]
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        if image.ndim==4: image = image.cpu().permute(0, 2, 3, 1).float()
        else: image = image.cpu().float()
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
        frame,
        callback_steps,
        pixel_value_llava=None,
        uncond_pixel_value_llava=None,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        vae_ver='88-4c-sd'
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if frame is not None:
            if '884' in vae_ver:
                if frame!=1 and (frame-1)%4!=0:
                    raise ValueError(f'`frame` has to be 1 or a multiple of 4 but is {frame}.')
            elif '888' in vae_ver:
                if frame!=1 and (frame-1)%8!=0:
                    raise ValueError(f'`frame` has to be 1 or a multiple of 8 but is {frame}.')

        if callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
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
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

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
    
    def prepare_latents(self, batch_size, num_channels_latents, height, width, frame, dtype, device, generator, latents=None, ref_latents=None, timestep=None):
        shape = (
            batch_size,
            num_channels_latents,
            frame,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)


        if timestep is not None:
            init_latents = ref_latents.clone().repeat(1,1,frame,1,1).to(device).to(dtype)
            latents = latents

        # Check existence to make it compatible with FlowMatchEulerDiscreteScheduler
        if hasattr(self.scheduler, "init_noise_sigma"):
            latents = latents * self.scheduler.init_noise_sigma

        return latents

    # Copied from diffusers.pipelines.latent_consistency_models.pipeline_latent_consistency_text2img.LatentConsistencyModelPipeline.get_guidance_scale_embedding
    def get_guidance_scale_embedding(
        self, w: torch.Tensor, embedding_dim: int = 512, dtype: torch.dtype = torch.float32
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

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]],                              
 
        ref_latents: Union[torch.Tensor],                            # [1, 16, 1, h//8, w//8]
        # uncond_ref_latents: Union[torch.Tensor],
        pixel_value_llava: Union[torch.Tensor],                # [1, 3, 336, 336]
        uncond_pixel_value_llava: Union[torch.Tensor],
        pixel_value_ref: Union[torch.Tensor],
        face_masks: Union[torch.Tensor],                              # [b f h w]
        audio_prompts: Union[torch.Tensor], 
        uncond_audio_prompts: Union[torch.Tensor], 
        motion_exp: Union[torch.Tensor], 
        motion_pose: Union[torch.Tensor], 
        fps: Union[torch.Tensor],
      
        height: int,
        width: int,
        video_length: int,
        data_type: str = "video",
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        sigmas: List[float] = None,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
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
        embedded_guidance_scale: Optional[float] = None,
        joint_pass = False,
        cfg_star_rescale = False,
        name = None,
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

        if self._interrupt:
            return [None]

        callback = kwargs.pop("callback", None)
        callback_steps = kwargs.pop("callback_steps", None)
        if callback_steps is not None:
            deprecate(
                "callback_steps",
                "1.0.0",
                "Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
            )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs


        # num_inference_steps =  50

        # 0. Default height and width to transformer
        # height = height or self.transformer.config.sample_size * self.vae_scale_factor
        # width = width or self.transformer.config.sample_size * self.vae_scale_factor
        # to deal with lora scaling and other possible forward hooks

        transformer = self.transformer

        skip_steps_cache = transformer.cache
        cache_type = None
        if skip_steps_cache != None:
            cache_type = skip_steps_cache.cache_type 
            if cache_type == "tea":
                teacache_multiplier = skip_steps_cache.multiplier
                skip_steps_cache.accumulated_rel_l1_distance = 0
                skip_steps_cache.rel_l1_thresh = 0.1 if teacache_multiplier < 2 else 0.15
            elif cache_type == "mag":
                transformer.compute_magcache_threshold(skip_steps_cache.start_step, num_inference_steps, skip_steps_cache.multiplier)
                skip_steps_cache.accumulated_err, skip_steps_cache.accumulated_steps, skip_steps_cache.accumulated_ratio  = 0, 0, 1.0
            else:
                transformer.cache = None
                cache_type = None

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            video_length,
            callback_steps,
            pixel_value_llava,
            uncond_pixel_value_llava,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            vae_ver=vae_ver
        )

        self._guidance_scale = guidance_scale
        self.start_cfg_scale = guidance_scale
        self._guidance_rescale = guidance_rescale
        self._clip_skip = clip_skip
        self._cross_attention_kwargs = cross_attention_kwargs

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        lora_scale = (
            self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
        )

 
        # ========== Encode text prompt (image prompt) ==========
        prompt_embeds, negative_prompt_embeds, prompt_mask, negative_prompt_mask = \
            self.encode_prompt_audio_text_base(
                prompt=prompt,
                uncond_prompt=negative_prompt,
                pixel_value_llava=pixel_value_llava,
                uncond_pixel_value_llava=uncond_pixel_value_llava,
                device=device,
                num_images_per_prompt=num_videos_per_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                lora_scale=lora_scale,
                clip_skip=self.clip_skip,
                text_encoder=self.text_encoder,
                data_type=data_type, 
                name= name,
                # **kwargs
            )
        if self.text_encoder_2 is not None:
            prompt_embeds_2, negative_prompt_embeds_2, prompt_mask_2, negative_prompt_mask_2 = \
                self.encode_prompt_audio_text_base(
                    prompt=prompt,
                    uncond_prompt=negative_prompt,
                    pixel_value_llava=None,
                    uncond_pixel_value_llava=None,
                    device=device,
                    num_images_per_prompt=num_videos_per_prompt,
                    do_classifier_free_guidance=self.do_classifier_free_guidance,
                    negative_prompt=negative_prompt,
                    prompt_embeds=None,
                    negative_prompt_embeds=None,
                    lora_scale=lora_scale,
                    clip_skip=self.clip_skip,
                    text_encoder=self.text_encoder_2,
                    # **kwargs
                )
        else:
            prompt_embeds_2 = None
            negative_prompt_embeds_2 = None
            prompt_mask_2 = None
            negative_prompt_mask_2 = None

        if self.transformer.mixed_precision:
            latent_dtype = torch.float32
        else:
            latent_dtype = torch.bfloat16
        if prompt_embeds != None:
            prompt_embeds = prompt_embeds.to(torch.bfloat16)
        if negative_prompt_embeds != None:
            negative_prompt_embeds = negative_prompt_embeds.to(torch.bfloat16)
        if prompt_embeds_2 != None:
            prompt_embeds_2 = prompt_embeds_2.to(torch.bfloat16)
        if negative_prompt_embeds_2 != None:
            negative_prompt_embeds_2 = negative_prompt_embeds_2.to(torch.bfloat16)
        if audio_prompts != None:
            audio_prompts = audio_prompts.to(torch.bfloat16)
        if face_masks!= None:
            face_masks = face_masks.to(torch.bfloat16)
        if ref_latents != None:
            ref_latents = ref_latents.to(torch.bfloat16)

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if self.do_classifier_free_guidance:
            prompt_embeds_input = torch.cat([negative_prompt_embeds, prompt_embeds])
            if prompt_mask is not None:
                prompt_mask_input = torch.cat([negative_prompt_mask, prompt_mask])
            if prompt_embeds_2 is not None:
                prompt_embeds_2_input = torch.cat([negative_prompt_embeds_2, prompt_embeds_2])
            if prompt_mask_2 is not None:
                prompt_mask_2_input = torch.cat([negative_prompt_mask_2, prompt_mask_2])
            
        if self.do_classifier_free_guidance and ref_latents != None:
            ref_latents = torch.cat([ref_latents, ref_latents], dim=0)


        # 4. Prepare timesteps
        extra_set_timesteps_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.set_timesteps, {"n_tokens": n_tokens}
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas, **extra_set_timesteps_kwargs,
        )

        video_length = audio_prompts.shape[1] // 4 * 4 + 1
        if "884" in vae_ver:
            video_length = (video_length - 1) // 4 + 1
        elif "888" in vae_ver:
            video_length = (video_length - 1) // 8 + 1
        else:
            video_length = video_length


        # 5. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels
        infer_length = (audio_prompts.shape[1] // 128 + 1) * 32 + 1
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            infer_length,
            latent_dtype, #prompt_embeds.dtype,
            device,
            generator,
            latents,
            ref_latents[-1:] if ref_latents != None else None,
            timesteps[:1]
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_func_kwargs(
            self.scheduler.step, {"generator": generator, "eta": eta},
        )

        vae_precision = "fp16" # torch.float16
        precision = "bf16" # torch.bfloat16
        disable_autocast =  True

        target_dtype = PRECISION_TO_TYPE[precision]
        autocast_enabled = (target_dtype != torch.float32) and not disable_autocast
        vae_dtype = self.vae._model_dtype #PRECISION_TO_TYPE[vae_precision]
        vae_autocast_enabled = (vae_dtype != torch.float32) and not disable_autocast

        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        latents_all = latents.clone()
        pad_audio_length = (audio_prompts.shape[1] // 128 + 1) * 128 + 4 - audio_prompts.shape[1]
        audio_prompts_all = torch.cat([audio_prompts, torch.zeros_like(audio_prompts[:, :pad_audio_length])], dim=1)


        shift = 0
        shift_offset = 10
        frames_per_batch = 33
        self.cache_tensor = None

        """ If the total length is shorter than 129, shift is not required """
        if video_length == 33 or infer_length == 33:
            infer_length = 33
            shift_offset = 0
            latents_all = latents_all[:, :, :33]
            audio_prompts_all = audio_prompts_all[:, :132]
        joint_pass = joint_pass or not self.do_classifier_free_guidance

        if callback != None:
            callback(-1, None, True, override_num_inference_steps = num_inference_steps)

        latent_items = 2 if self.do_classifier_free_guidance else 1

        fps = torch.from_numpy(np.array(fps)).unsqueeze(0).to(dtype=torch.float16)

        if self._interrupt:
            return [None]

        if cache_type == "tea":
            cache_size = round( infer_length / frames_per_batch )
            skip_steps_cache.previous_residual = [None] * latent_items
            cache_all_previous_residual =  [None] * latent_items
            cache_all_previous_modulated_input = None
            cache_should_calc = [True] * cache_size 
            cache_accumulated_rel_l1_distance = [0.] * cache_size
            cache_teacache_skipped_steps = [0] * cache_size
        elif cache_type == "mag":
            skip_steps_cache.previous_residual = [None] * latent_items


        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # init 
                pred_latents = torch.zeros_like(
                    latents_all,
                    dtype=latents_all.dtype,
                )
                counter = torch.zeros(
                    (latents_all.shape[0], latents_all.shape[1], infer_length, 1, 1),
                    dtype=latents_all.dtype,
                ).to(device=latents_all.device)

                preview_latents = torch.zeros_like(pred_latents) if getattr(callback, "tiny_vae", False) else None
                cache_slot_no = 0
                for index_start in range(0, infer_length, frames_per_batch):
                    self.scheduler._step_index = None

                    index_start = index_start - shift
                    idx_list = [ii % latents_all.shape[2] for ii in range(index_start, index_start + frames_per_batch)]
                    latents = latents_all[:, :, idx_list].clone()

                    idx_list_audio = [ii % audio_prompts_all.shape[1] for ii in range(index_start * 4, (index_start + frames_per_batch) * 4 - 3)]
                    audio_prompts = audio_prompts_all[:, idx_list_audio].clone()

                    # expand the latents if we are doing classifier free guidance
                    if self.do_classifier_free_guidance:
                        latent_model_input = torch.cat([latents] * 2)
                    else:
                        latent_model_input = latents 
                        
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    embedded_hw = (latent_model_input.shape[-1] // 2) * (latent_model_input.shape[-2] // 2) * 3072
                    img_ref_len = (latent_model_input.shape[-1] // 2)  * (latent_model_input.shape[-2] // 2) * ( 1) 
                    img_all_len = (latents_all.shape[-1] // 2)  * (latents_all.shape[-2] // 2) * latents_all.shape[-3]

                    if cache_type == "tea" and cache_size > 1:
                        for l in range(latent_items):
                            if cache_all_previous_residual[l] != None:
                                bsz = cache_all_previous_residual[l].shape[0]
                                skip_steps_cache.previous_residual[l][:, img_ref_len:] = cache_all_previous_residual[l].reshape(1, -1, embedded_hw) [:, idx_list].reshape(1, -1, 3072)
                        if cache_all_previous_modulated_input != None:
                            skip_steps_cache.previous_modulated_input[:, img_ref_len:] = cache_all_previous_modulated_input.reshape(1, -1, embedded_hw) [:, idx_list].reshape(1, -1, 3072)
                        skip_steps_cache.should_calc = cache_should_calc[cache_slot_no]  
                        skip_steps_cache.accumulated_rel_l1_distance = cache_accumulated_rel_l1_distance[cache_slot_no]
                        skip_steps_cache.teacache_skipped_steps = cache_teacache_skipped_steps[cache_slot_no]


                    if self.do_classifier_free_guidance:
                        if i < num_inference_steps * 0.2 :
                            self._guidance_scale = (1 - i / len(timesteps)) * (self.start_cfg_scale - 2) + 2
                            audio_prompts_input = torch.cat([uncond_audio_prompts, audio_prompts], dim=0)
                            face_masks_input = torch.cat([face_masks * 0.6] * 2, dim=0)
                        else:
                            # define 10-50 step cfg
                            self._guidance_scale = (1 - i / len(timesteps)) * (6.5 - 3.5) + 3.5  # 5-2 +2

                            prompt_embeds_input = torch.cat([prompt_embeds, prompt_embeds])
                            if prompt_mask is not None:
                                prompt_mask_input = torch.cat([prompt_mask, prompt_mask])
                            if prompt_embeds_2 is not None:
                                prompt_embeds_2_input = torch.cat([prompt_embeds_2, prompt_embeds_2])
                            if prompt_mask_2 is not None:
                                prompt_mask_2_input = torch.cat([prompt_mask_2, prompt_mask_2])
                            audio_prompts_input = torch.cat([uncond_audio_prompts, audio_prompts], dim=0)
                            face_masks_input = torch.cat([face_masks] * 2, dim=0)

                        motion_exp_input = torch.cat([motion_exp] * 2, dim=0)
                        motion_pose_input = torch.cat([motion_pose] * 2, dim=0)
                        fps_input = torch.cat([fps] * 2, dim=0)
    
                    else:
                        audio_prompts_input = audio_prompts
                        face_masks_input = face_masks
                        motion_exp_input = motion_exp
                        motion_pose_input = motion_pose
                        fps_input = fps

                    t_expand = t.repeat(latent_model_input.shape[0])
                    guidance_expand = None

                    with torch.autocast(device_type="cuda", dtype=target_dtype, enabled=autocast_enabled):                        
                        additional_kwargs = {
                            "pipeline": self,
                            "step_no": i,
                        }
                        if joint_pass:
                            additional_kwargs.update({
                                "motion_exp": motion_exp_input,        
                                "motion_pose": motion_pose_input,     
                                "fps": fps_input,                       
                                "audio_prompts": audio_prompts_input,   
                                "face_mask": face_masks_input         
                            })
                            noise_pred = self.transformer(latent_model_input, t_expand, ref_latents=ref_latents, text_states=prompt_embeds_input, text_mask=prompt_mask_input, text_states_2=prompt_embeds_2_input, freqs_cos=freqs_cis[0], freqs_sin=freqs_cis[1], guidance=guidance_expand, **additional_kwargs,)
                            if self._interrupt:
                                return [None]
                            if self.do_classifier_free_guidance:
                                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        else:                                
                            additional_kwargs.update({
                                "motion_exp": motion_exp_input[:1],        
                                "motion_pose": motion_pose_input[:1],      
                                "fps": fps_input[:1],                       
                                "audio_prompts": audio_prompts_input[:1],  
                                "face_mask": face_masks_input[:1]        
                            })
                            noise_pred_uncond = self.transformer(latent_model_input[:1], t_expand[:1], ref_latents=ref_latents[:1], text_states=prompt_embeds_input[:1], text_mask=prompt_mask_input[:1], text_states_2=prompt_embeds_2_input[:1], freqs_cos=freqs_cis[0], freqs_sin=freqs_cis[1], guidance=guidance_expand, x_id = 0, **additional_kwargs,)
                            if self._interrupt:
                                return [None]
                            noise_pred_uncond = noise_pred_uncond[0]
                            additional_kwargs.update({
                                "motion_exp": motion_exp_input[1:],        
                                "motion_pose": motion_pose_input[1:],      
                                "fps": fps_input[1:],                       
                                "audio_prompts": audio_prompts_input[1:],  
                                "face_mask": face_masks_input[1:]        
                            })
                            noise_pred_text = self.transformer(latent_model_input[1:], t_expand[1:], ref_latents=ref_latents[1:], text_states=prompt_embeds_input[1:], text_mask=prompt_mask_input[1:], text_states_2=prompt_embeds_2_input[1:], freqs_cos=freqs_cis[0], freqs_sin=freqs_cis[1], guidance=guidance_expand, x_id = 1, **additional_kwargs,)
                            if self._interrupt:
                                return [None]
                            noise_pred_text = noise_pred_text[0]
                            
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

                        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
                      
                    if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
                        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)
                    noise_pred_text, noise_pred_uncond = None, None
                    # compute the previous noisy sample x_t -> x_t-1
                    preview_chunk = None
                    if preview_latents is not None:
                        preview_step_index = self.scheduler.step_index
                        if preview_step_index is None:
                            preview_step_index = self.scheduler.begin_index
                        if preview_step_index is None:
                            preview_step_index = self.scheduler.index_for_timestep(t)
                        preview_chunk = latents.float() - self.scheduler.sigmas[preview_step_index] * noise_pred.float()
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                    noise_pred = None

                    latents = latents.to(torch.bfloat16)
                    for iii in range(frames_per_batch):
                        p = (index_start + iii) % pred_latents.shape[2]
                        pred_latents[:, :, p] += latents[:, :, iii]
                        counter[:, :, p] += 1
                        if preview_chunk is not None:
                            preview_latents[:, :, p] += preview_chunk[:, :, iii]
                    preview_chunk = None

                    if cache_type == "tea" and cache_size > 1:
                        for l in range(latent_items):
                            if skip_steps_cache.previous_residual[l] != None:
                                bsz = skip_steps_cache.previous_residual[l].shape[0]
                                if cache_all_previous_residual[l] == None:
                                    cache_all_previous_residual[l] = torch.zeros((bsz, img_all_len, 3072 ), device=skip_steps_cache.previous_residual[l].device, dtype=skip_steps_cache.previous_residual[l].dtype)
                                cache_all_previous_residual[l].reshape(bsz, -1, embedded_hw)[:, idx_list] = skip_steps_cache.previous_residual[l][:, img_ref_len:].reshape(bsz, -1, embedded_hw)

                        if skip_steps_cache.previous_modulated_input  != None:
                            if cache_all_previous_modulated_input == None:
                                cache_all_previous_modulated_input = torch.zeros((1, img_all_len, 3072 ), device=skip_steps_cache.previous_modulated_input.device, dtype=skip_steps_cache.previous_modulated_input.dtype)
                            cache_all_previous_modulated_input.reshape(1, -1, embedded_hw)[:, idx_list] = skip_steps_cache.previous_modulated_input[:, img_ref_len:].reshape(1, -1, embedded_hw)
                        cache_should_calc[cache_slot_no]  = skip_steps_cache.should_calc 
                        cache_accumulated_rel_l1_distance[cache_slot_no]  = skip_steps_cache.accumulated_rel_l1_distance
                        cache_teacache_skipped_steps[cache_slot_no]  = skip_steps_cache.teacache_skipped_steps

                    cache_slot_no += 1

                shift += shift_offset
                shift = shift % frames_per_batch  
                pred_latents  = pred_latents / counter
                latents_all = pred_latents     

                if callback is not None:
                    callback(i, (preview_latents / counter if preview_latents is not None else latents_all)[0, :, :video_length], False)
                    preview_latents = None

        latents = latents_all.float()[:, :, :video_length] 

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
                    f"Only support latents with shape (b, c, h, w) or (b, c, f, h, w), but got {latents.shape}.")

            if hasattr(self.vae.config, 'shift_factor') and self.vae.config.shift_factor:
                latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
            else:
                latents = latents / self.vae.config.scaling_factor

            with torch.autocast(device_type="cuda", dtype=vae_dtype, enabled=vae_autocast_enabled):
                image = self.vae.decode(latents, return_dict=False, generator=generator)[0]
            if image is None:
                return (None, )

            if expand_temporal_dim or image.shape[2] == 1:
                image = image.squeeze(2)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().float()

        # Offload all models
        self.maybe_free_model_hooks()
        
        if not return_dict:
            return image
        
        return HunyuanVideoPipelineOutput(videos=image)
