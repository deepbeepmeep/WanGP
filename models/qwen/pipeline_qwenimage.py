# Copyright 2025 Qwen-Image Team and The HuggingFace Team. All rights reserved.
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
from shared.utils.phase_progress import text_encoding_prompts, text_encoding_progress
from mmgp import offload
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch, json
import math
from diffusers.image_processor import VaeImageProcessor
from .transformer_qwenimage import QwenImageTransformer2DModel

from diffusers.utils import logging
from diffusers.utils.torch_utils import randn_tensor
from diffusers import FlowMatchEulerDiscreteScheduler
from PIL import Image
from shared.utils.utils import calculate_new_dimensions, convert_image_to_tensor, convert_tensor_to_image
from shared.utils.text_encoder_cache import TextEncoderCache

XLA_AVAILABLE = False

PREFERRED_QWENIMAGE_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


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

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")

class QwenImagePipeline(): #DiffusionPipeline
    r"""
    The QwenImage pipeline for text-to-image generation.

    Args:
        transformer ([`QwenImageTransformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`Qwen2.5-VL-7B-Instruct`]):
            [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), specifically the
            [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) variant.
        tokenizer (`QwenTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds"]

    def __init__(
        self,
        vae,
        text_encoder,
        tokenizer,
        transformer,
        processor,
    ):
        
        self.vae=vae
        self.text_encoder=text_encoder
        self.tokenizer=tokenizer
        self.transformer=transformer
        self.processor = processor

        self.latent_channels = self.vae.z_dim if getattr(self, "vae", None) else 16
        self.vae_scale_factor = 8 # 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
        # QwenImage latents are turned into 2x2 patches and packed. This means the latent width and height has to be divisible
        # by the patch size. So the vae scale factor is multiplied by the patch size to account for this
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        self.tokenizer_max_length = 1024
        self.text_encoder_cache = TextEncoderCache()
        if processor is not None:
            # self.prompt_template_encode = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n"
            self.prompt_template_encode = "<|im_start|>system\nDescribe the key features of the input image (color, shape, size, texture, objects, background), then explain how the user's text instruction should alter or modify the image. Generate a new image that meets the user's requirements while maintaining consistency with the original input where appropriate.<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
            self.prompt_template_encode_start_idx = 64
        else:
            self.prompt_template_encode = "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
            self.prompt_template_encode_start_idx = 34
        self.default_sample_size = 128

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        split_result = torch.split(selected, valid_lengths.tolist(), dim=0)

        return split_result

    def _get_qwen_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        image: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        template = self.prompt_template_encode
        drop_idx = self.prompt_template_encode_start_idx

        if self.processor is not None and image is not None and len(image) > 0:
            img_prompt_template = "Picture {}: <|vision_start|><|image_pad|><|vision_end|>"
            if isinstance(image, list):
                base_img_prompt = ""
                for i, img in enumerate(image):
                    base_img_prompt += img_prompt_template.format(i + 1)
            elif image is not None:
                base_img_prompt = img_prompt_template.format(1)
            else:
                base_img_prompt = ""

            template = self.prompt_template_encode

            drop_idx = self.prompt_template_encode_start_idx
            txt = [template.format(base_img_prompt + e) for e in prompt]

            model_inputs = self.processor(
                text=txt,
                images=image,
                padding=True,
                return_tensors="pt",
                device=device,
            ).to(device)

            with text_encoding_progress(self.text_encoder.model.language_model.layers, prompt_count=len(prompt)):
                outputs = self.text_encoder.model(input_ids=model_inputs.input_ids, attention_mask=model_inputs.attention_mask, pixel_values=model_inputs.pixel_values, image_grid_thw=model_inputs.image_grid_thw, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            split_hidden_states = self._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
            split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
            attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        else:
            def encode_fn(prompts):
                txt = [template.format(p) for p in prompts]
                txt_tokens = self.tokenizer(
                    txt,
                    max_length=self.tokenizer_max_length + drop_idx,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                with text_encoding_progress(self.text_encoder.model.language_model.layers, prompt_count=len(prompts)):
                    hidden_states = self.text_encoder.model(input_ids=txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask, output_hidden_states=True).hidden_states[-1]
                split_hidden_states = self._extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
                split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
                attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
                return list(zip(split_hidden_states, attn_mask_list))
            contexts = self.text_encoder_cache.encode(encode_fn, prompt, device=device)
            split_hidden_states = [ctx[0] for ctx in contexts]
            attn_mask_list = [ctx[1] for ctx in contexts]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds, encoder_attention_mask

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        image: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        max_sequence_length: int = 1024,
    ):
        r"""

        Args:
            prompt (`str` or `List[str]`, *optional*):
                prompt to be encoded
            image (`torch.Tensor`, *optional*):
                image to be encoded
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(prompt, image, device)

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
        prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)

        return prompt_embeds, prompt_embeds_mask


    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        prompt_embeds_mask=None,
        negative_prompt_embeds_mask=None,
        callback_on_step_end_tensor_inputs=None,
        max_sequence_length=None,
    ):
        if height % (self.vae_scale_factor * 2) != 0 or width % (self.vae_scale_factor * 2) != 0:
            logger.warning(
                f"`height` and `width` have to be divisible by {self.vae_scale_factor * 2} but are {height} and {width}. Dimensions will be resized accordingly"
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

        if prompt_embeds is not None and prompt_embeds_mask is None:
            raise ValueError(
                "If `prompt_embeds` are provided, `prompt_embeds_mask` also have to be passed. Make sure to generate `prompt_embeds_mask` from the same text encoder that was used to generate `prompt_embeds`."
            )
        if negative_prompt_embeds is not None and negative_prompt_embeds_mask is None:
            raise ValueError(
                "If `negative_prompt_embeds` are provided, `negative_prompt_embeds_mask` also have to be passed. Make sure to generate `negative_prompt_embeds_mask` from the same text encoder that was used to generate `negative_prompt_embeds`."
            )

        if max_sequence_length is not None and max_sequence_length > 1024:
            raise ValueError(f"`max_sequence_length` cannot be greater than 1024 but is {max_sequence_length}")

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3, device=device)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height, device=device)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width, device=device)[None, :]

        latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

        latent_image_ids = latent_image_ids.reshape(
            latent_image_id_height * latent_image_id_width, latent_image_id_channels
        )

        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def _pack_latents(latents):
        batch_size, num_channels_latents, _, height, width = latents.shape 
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        batch_size, num_patches, channels = latents.shape

        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))

        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)

        latents = latents.reshape(batch_size, channels // (2 * 2), 1, height, width)

        return latents


    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(self.vae.encode(image[i : i + 1]), generator=generator[i], sample_mode="argmax")
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator, sample_mode="argmax")
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean, device=image_latents.device, dtype=image_latents.dtype)
            .view(1, self.latent_channels, 1, 1, 1)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std, device=image_latents.device, dtype=image_latents.dtype)
            .view(1, self.latent_channels, 1, 1, 1)
        )
        image_latents = (image_latents - latents_mean) / latents_std

        return image_latents
    def enable_vae_slicing(self):
        r"""
        Enable sliced VAE decoding. When this option is enabled, the VAE will split the input tensor in slices to
        compute decoding in several steps. This is useful to save some memory and allow larger batch sizes.
        """
        self.vae.enable_slicing()

    def disable_vae_slicing(self):
        r"""
        Disable sliced VAE decoding. If `enable_vae_slicing` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_slicing()

    def enable_vae_tiling(self):
        r"""
        Enable tiled VAE decoding. When this option is enabled, the VAE will split the input tensor into tiles to
        compute decoding and encoding in several steps. This is useful for saving a large amount of memory and to allow
        processing larger images.
        """
        self.vae.enable_tiling()

    def disable_vae_tiling(self):
        r"""
        Disable tiled VAE decoding. If `enable_vae_tiling` was previously enabled, this method will go back to
        computing decoding in one step.
        """
        self.vae.disable_tiling()

    def prepare_latents(
        self,
        images,
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        num_layers: int = 1,
        latents=None,
        tile_size=0,
    ):
        # VAE applies 8x compression on images but we must also account for packing which requires
        # latent height and width to be divisible by 2.
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))

        shape = (batch_size, num_channels_latents, 1, height, width)

        image_latents = None
        if images is not None and len(images ) > 0:
            if not isinstance(images, list):
                images = [images]
            all_image_latents = []
            for image in images:
                if image.shape[1] != self.latent_channels:
                    image = image.to(device=device, dtype=self.vae.dtype)
                    image_latents = self._encode_vae_image(image=image, generator=generator)
                else:
                    image_latents = image
                image_latents = image_latents.to(device=device, dtype=dtype)
                if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
                    # expand init_latents for batch_size
                    additional_image_per_prompt = batch_size // image_latents.shape[0]
                    image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
                elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
                    raise ValueError(
                        f"Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts."
                    )
                else:
                    image_latents = torch.cat([image_latents], dim=0)

                image_latents = self._pack_latents(image_latents)
                all_image_latents.append(image_latents)
            image_latents = torch.cat(all_image_latents, dim=1)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            if num_layers > 1:
                layer_shape = (batch_size * num_layers, num_channels_latents, 1, height, width)
                latents = randn_tensor(layer_shape, generator=generator, device=device, dtype=dtype)
                latents = self._pack_latents(latents)
                seq_len = latents.shape[1]
                dim = latents.shape[2]
                latents = latents.view(batch_size, num_layers, seq_len, dim)
                latents = latents.reshape(batch_size, num_layers * seq_len, dim)
            else:
                latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
                latents = self._pack_latents(latents)
        else:
            latents = latents.to(device=device, dtype=dtype)
            if num_layers > 1 and latents.shape[1] % num_layers != 0:
                raise ValueError("Provided latents length is not divisible by the number of layers.")

        return latents, image_latents

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        true_cfg_scale: float = 4.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        num_images_per_prompt: int = 1,
        layers: int = 1,
        cfg_normalize: bool = True,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        image = None,
        image_mask = None,
        denoising_strength = 0,
        masking_strength = 1,
        callback=None,
        pipeline=None,
        loras_slists=None,
        joint_pass= True,
        model_mode = 0,
        outpainting_dims = None,
        qwen_edit_plus = False,
        VAE_tile_size = 0,
        vae_upsampler=None,
        vae_upsampler_seed: int = 0,
        vae_upsampler_progress_callback=None,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
                not greater than `1`).
            true_cfg_scale (`float`, *optional*, defaults to 1.0):
                When > 1.0 and a provided `negative_prompt`, enables true classifier-free guidance.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.qwenimage.QwenImagePipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
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
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.qwenimage.QwenImagePipelineOutput`] or `tuple`:
            [`~pipelines.qwenimage.QwenImagePipelineOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is a list with the generated images.
        """

        model_mode_int = None
        if model_mode is not None:
            try:
                model_mode_int = int(model_mode)
            except (TypeError, ValueError):
                model_mode_int = None
        lora_inpaint = image_mask is not None and model_mode_int == 1
        lanpaint_enabled = image_mask is not None and model_mode_int in (2, 3, 4, 5)

        kwargs = {'pipeline': pipeline, 'callback': callback}
        if callback != None:
            callback(-1, None, True)

        vae_z_dim = self.vae.z_dim
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        device = "cuda"

        num_layers = max(1, int(layers) if layers is not None else 1)
        effective_batch_size = batch_size * num_images_per_prompt
        vae_input_channels = getattr(self.vae.config, "input_channels", 3)

        condition_images = []
        vae_image_sizes = []
        vae_images = []
        image_mask_latents = None
        image_mask_rebuilt = None
        ref_size = 1024
        ref_text_encoder_size = 384 if qwen_edit_plus else 1024
        if image is not None:
            if not isinstance(image, list): image = [image]
            if height * width < ref_size * ref_size: ref_size =  round(math.sqrt(height * width))  
            for ref_no, img in enumerate(image):
                image_width, image_height = img.size
                any_mask = num_layers == 1 and ref_no == 0 and image_mask is not None
                if (not qwen_edit_plus) and (image_height * image_width > ref_size * ref_size) and not any_mask:
                    vae_height, vae_width =calculate_new_dimensions(ref_size, ref_size, image_height, image_width, False, block_size=multiple_of)
                else:
                    vae_height, vae_width = image_height, image_width 
                    vae_width = vae_width // multiple_of * multiple_of
                    vae_height = vae_height // multiple_of * multiple_of
                vae_image_sizes.append((vae_width, vae_height))
                condition_height, condition_width =calculate_new_dimensions(ref_text_encoder_size, ref_text_encoder_size, image_height, image_width, False, block_size=multiple_of)
                condition_img = img
                if getattr(condition_img, "mode", None) != "RGB":
                    condition_img = condition_img.convert("RGB")
                condition_images.append(condition_img.resize((condition_width, condition_height), resample=Image.Resampling.LANCZOS) )

                vae_img = img
                if vae_input_channels == 4:
                    if getattr(vae_img, "mode", None) != "RGBA":
                        vae_img = vae_img.convert("RGBA")
                else:
                    if getattr(vae_img, "mode", None) != "RGB":
                        vae_img = vae_img.convert("RGB")
                if vae_img.size != (vae_width, vae_height):
                    vae_img = vae_img.resize((vae_width, vae_height), resample=Image.Resampling.LANCZOS) 
                if any_mask :
                    if lora_inpaint:
                        image_mask_rebuilt = torch.where(convert_image_to_tensor(image_mask)>-0.5, 1., 0. )[0:1]
                        vae_tensor = convert_image_to_tensor(vae_img)
                        green = torch.tensor([-1.0, 1.0, -1.0], device=vae_tensor.device).to(vae_tensor)
                        green_image = green[:, None, None] .expand_as(vae_tensor)
                        vae_tensor = torch.where(image_mask_rebuilt > 0, green_image, vae_tensor)
                        vae_img = convert_tensor_to_image(vae_tensor)
                    else:
                        image_mask_latents = convert_image_to_tensor(image_mask.resize((vae_width // 8, vae_height // 8), resample=Image.Resampling.LANCZOS))
                        image_mask_latents = torch.where(image_mask_latents>-0.5, 1., 0. )[0:1]
                        image_mask_rebuilt = image_mask_latents.repeat_interleave(8, dim=-1).repeat_interleave(8, dim=-2).unsqueeze(0)
                        # convert_tensor_to_image( image_mask_rebuilt.squeeze(0).repeat(3,1,1)).save("mmm.png")
                        image_mask_latents = image_mask_latents.to(device).unsqueeze(0).unsqueeze(0).repeat(1,16,1,1,1)
                        image_mask_latents = self._pack_latents(image_mask_latents)
                # img.save("nnn.png")
                vae_images.append( convert_image_to_tensor(vae_img).unsqueeze(0).unsqueeze(2) )


        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        pending_prompts = [] if prompt_embeds is not None else [prompt] if isinstance(prompt, str) else list(prompt)
        if do_true_cfg and negative_prompt_embeds is None:
            pending_prompts += [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
        prompt_count = len(pending_prompts) if condition_images else sum(p not in self.text_encoder_cache._entries for p in dict.fromkeys(pending_prompts))
        with text_encoding_prompts(prompt_count):
            prompt_embeds, prompt_embeds_mask = self.encode_prompt(
                image=condition_images,
                prompt=prompt,
                prompt_embeds=prompt_embeds,
                prompt_embeds_mask=prompt_embeds_mask,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )
            if do_true_cfg:
                negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                    image=condition_images,
                    prompt=negative_prompt,
                    prompt_embeds=negative_prompt_embeds,
                    prompt_embeds_mask=negative_prompt_embeds_mask,
                    device=device,
                    num_images_per_prompt=num_images_per_prompt,
                    max_sequence_length=max_sequence_length,
                )
        additional_t_cond = None
        if getattr(self.transformer, "use_additional_t_cond", False):
            add_value = 1 if image is not None and len(condition_images) > 0 else 0
            additional_t_cond = torch.full(
                (effective_batch_size,),
                add_value,
                device=device,
                dtype=torch.long,
            )
        dtype = torch.bfloat16
        prompt_embeds = prompt_embeds.to(dtype)        
        if do_true_cfg:
            negative_prompt_embeds = negative_prompt_embeds.to(dtype)        

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.in_channels // 4
        latents, image_latents = self.prepare_latents(
            vae_images,
            effective_batch_size,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            num_layers=num_layers,
            latents=latents,
            tile_size=VAE_tile_size
        )

        if image is not None:
            output_shape = (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2)
            output_shapes = [output_shape] * num_layers if num_layers > 1 else [output_shape]
            condition_shapes = [
                (1, vae_height // self.vae_scale_factor // 2, vae_width // self.vae_scale_factor // 2)
                for vae_width, vae_height in vae_image_sizes
            ]
            if lanpaint_enabled:
                condition_shapes = condition_shapes[1:]
            img_shapes = [output_shapes + condition_shapes] * effective_batch_size
        else:
            output_shape = (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2)
            if num_layers > 1:
                img_shapes = [[output_shape] * num_layers] * effective_batch_size
            else:
                img_shapes = [output_shape] * effective_batch_size

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)
        original_timesteps = timesteps

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
        )
        morph, first_step = False, 0
        lanpaint_proc = original_image_latents = None
        if image_mask_latents is not None:
            original_image_latents =  image_latents[:, :latents.shape[1]].clone() 
            if lanpaint_enabled:
                if image_latents.shape[1]==latents.shape[1]:
                    image_latents = None
                else:
                    image_latents = image_latents[:, latents.shape[1]:] 

                from shared.lanpaint import LanPaint
                lanpaint_steps = {2: 2, 3: 5, 4: 10, 5: 15}.get(model_mode_int, 5)
                lanpaint_proc = LanPaint(NSteps=lanpaint_steps)
                denoising_strength = 1.
                masking_strength = 1.
            randn = torch.randn_like(original_image_latents)
            if denoising_strength < 1.:
                first_step = int(len(timesteps) * (1. - denoising_strength))
            masked_steps = math.ceil(len(timesteps) * masking_strength)                
            if not morph:
                latent_noise_factor = timesteps[first_step]/1000
                # latents  = original_image_latents  * (1.0 - latent_noise_factor) + torch.randn_like(original_image_latents) * latent_noise_factor 
                latents  = original_image_latents  * (1.0 - latent_noise_factor) + randn * latent_noise_factor 
                timesteps = timesteps[first_step:]
                self.scheduler.timesteps = timesteps
                self.scheduler.sigmas= self.scheduler.sigmas[first_step:]
        # 6. Denoising loop
        self.scheduler.set_begin_index(0)
        updated_num_steps= len(timesteps)
        if callback != None:
            from shared.utils.loras_mutipliers import update_loras_slists
            update_loras_slists(self.transformer, loras_slists, len(original_timesteps))
            callback(-1, None, True, override_num_inference_steps = updated_num_steps)


        for i, t in enumerate(timesteps):
            offload.set_step_no_for_lora(self.transformer, first_step  + i)
            if self.interrupt:
                continue
            self._current_timestep = t
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            if image_mask_latents is not None and denoising_strength <1. and i == first_step and morph:
                latent_noise_factor = t/1000
                latents  = original_image_latents  * (1.0 - latent_noise_factor) + latents * latent_noise_factor 

            latents_dtype = latents.dtype

            # latent_model_input = latents
            def denoise(latent_model_input, true_cfg_scale):
                if image_latents is not None:
                    latent_model_input = torch.cat([latent_model_input, image_latents], dim=1)
                do_true_cfg = true_cfg_scale > 1
                if do_true_cfg and joint_pass:
                    noise_pred, neg_noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        encoder_hidden_states_mask_list=[prompt_embeds_mask,negative_prompt_embeds_mask],
                        encoder_hidden_states_list=[prompt_embeds, negative_prompt_embeds],
                        img_shapes=img_shapes,
                        txt_seq_lens_list=[txt_seq_lens, negative_txt_seq_lens],
                        attention_kwargs=self.attention_kwargs,
                        additional_t_cond=additional_t_cond,
                        **kwargs
                    )
                    if noise_pred == None: return None, None
                    noise_pred = noise_pred[:, : latents.size(1)]
                    neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                else:
                    neg_noise_pred = None
                    noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        encoder_hidden_states_mask_list=[prompt_embeds_mask],
                        encoder_hidden_states_list=[prompt_embeds],
                        img_shapes=img_shapes,
                        txt_seq_lens_list=[txt_seq_lens],
                        attention_kwargs=self.attention_kwargs,
                        additional_t_cond=additional_t_cond,
                        **kwargs
                    )[0]
                    if noise_pred == None: return None, None
                    noise_pred = noise_pred[:, : latents.size(1)]

                    if do_true_cfg:
                        neg_noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            encoder_hidden_states_mask_list=[negative_prompt_embeds_mask],
                            encoder_hidden_states_list=[negative_prompt_embeds],
                            img_shapes=img_shapes,
                            txt_seq_lens_list=[negative_txt_seq_lens],
                            attention_kwargs=self.attention_kwargs,
                            additional_t_cond=additional_t_cond,
                            **kwargs
                        )[0]
                        if neg_noise_pred == None: return None, None
                        neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                return noise_pred, neg_noise_pred
            def cfg_predictions( noise_pred, neg_noise_pred, guidance, t):
                if do_true_cfg:
                    comb_pred = neg_noise_pred + guidance * (noise_pred - neg_noise_pred)
                    if comb_pred == None: return None
                    if cfg_normalize:
                        cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                        noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                        noise_pred = comb_pred * (cond_norm / noise_norm)
                    else:
                        noise_pred = comb_pred

                return noise_pred


            if lanpaint_proc is not None and i < updated_num_steps - 1:
                latents = lanpaint_proc(denoise, cfg_predictions, true_cfg_scale, 1., latents, original_image_latents, randn, t/1000, image_mask_latents, height=height , width= width, vae_scale_factor= 8)
                if latents is None: return None

            noise_pred, neg_noise_pred = denoise(latents, true_cfg_scale)
            if noise_pred == None: return None
            noise_pred = cfg_predictions(noise_pred, neg_noise_pred, true_cfg_scale, t)
            neg_noise_pred = None
            preview_x0 = None
            if callback is not None and getattr(callback, "tiny_vae", False):
                if self.scheduler.step_index is None:
                    self.scheduler._init_step_index(t)
                preview_x0 = latents.float() - self.scheduler.sigmas[self.scheduler.step_index].to(latents.device) * noise_pred.float()
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            noise_pred = None

            if image_mask_latents is not None and i < masked_steps -1 :
                next_t = timesteps[i+1] if i<len(timesteps)-1 else 0
                latent_noise_factor = next_t / 1000
                noisy_image  = original_image_latents  * (1.0 - latent_noise_factor) + randn * latent_noise_factor 
                latents  =  noisy_image * (1-image_mask_latents)  + image_mask_latents * latents
                noisy_image = None
                if preview_x0 is not None:
                    preview_x0 = original_image_latents * (1-image_mask_latents) + image_mask_latents * preview_x0

            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

            if callback is not None:
                preview_latents = preview_x0 if preview_x0 is not None else latents
                if num_layers > 1:
                    seq_len = latents.shape[1] // num_layers
                    preview_latents = preview_latents[:, :seq_len, :]
                preview = self._unpack_latents(preview_latents, height, width, self.vae_scale_factor)
                preview = preview.transpose(0,2).squeeze(0)
                callback(i, preview, False)
                del preview, preview_latents
            del preview_x0


        self._current_timestep = None
        if output_type == "latent":
            output_image = latents
        else:
            latents_to_decode = latents
            if num_layers > 1:
                seq_len = latents.shape[1] // num_layers
                dim = latents.shape[2]
                latents_to_decode = latents.view(latents.shape[0], num_layers, seq_len, dim)
                latents_to_decode = latents_to_decode.reshape(latents.shape[0] * num_layers, seq_len, dim)
            latents_to_decode = self._unpack_latents(latents_to_decode, height, width, self.vae_scale_factor)
            latents_to_decode = latents_to_decode.to(self.vae.dtype)
            pid_latents = latents_to_decode[:, :, 0] if vae_upsampler is not None else None
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean, device=latents_to_decode.device)
                .view(1, vae_z_dim, 1, 1, 1)
                .to(latents_to_decode.device, latents_to_decode.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std, device=latents_to_decode.device).view(1, vae_z_dim, 1, 1, 1).to(
                latents_to_decode.device, latents_to_decode.dtype
            )
            latents_to_decode = latents_to_decode / latents_std + latents_mean
            output_image = self.vae.decode_to_cpu_uint8(latents_to_decode)[:, :, 0]
            if vae_upsampler is not None:
                if vae_upsampler_progress_callback is not None:
                    vae_upsampler_progress_callback("VAE")
                image_ref, latent_ref = [output_image], [pid_latents]
                output_image = latents_to_decode = None
                output_image = vae_upsampler.decode_inputs(image_ref, latent_ref, prompt=prompt, seed=vae_upsampler_seed, abort_callback=lambda: self._interrupt, progress_callback=vae_upsampler_progress_callback)
                if output_image is None:
                    return None
            # looks worse
            # if (
            #     num_layers == 1
            #     and image_mask_rebuilt is not None
            #     and not lora_inpaint
            #     and self.vae.upsampling_set is None
            #     and masking_strength == 10
            # ):
            #     output_image = vae_images[0].squeeze(2) * (1 - image_mask_rebuilt) + output_image.to(vae_images[0]  ) * image_mask_rebuilt 


        return output_image
