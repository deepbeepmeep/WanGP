"""Single-GPU WanGP pipeline for Ming Image Design and Design-Layer.

Weights remain on CPU after loading; MMGP owns residency and layer offload.
"""

import hashlib
import inspect
import json
import math
import re
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import torch
from accelerate import init_empty_weights
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.normalization import RMSNorm
from PIL import Image
from mmgp import runtime
from torch import nn
from torch.nn import functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

from shared.utils import files_locator as fl
from shared.utils.phase_progress import (
    check_abort, generation_progress, set_phase_status, text_encoding_progress,
)
from shared.utils.text_encoder_cache import TextEncoderCache
from shared.utils.utils import convert_tensor_to_image
from .ming_handler import ENCODER, LAYER_ARCHITECTURE, LAYER_PROJECT, PROJECT, SHARED_PROJECT, TOKENIZER_FILES, conditioner_file
from .prompt_format import strip_comment_lines
from .vae import MingImageVAE
from .upstream.bailingmm_utils import process_ratio
from .upstream.configuration_bailingmm2 import BailingMM2Config
from .upstream.diffusion.generator import ConditionedTransformer
from .upstream.diffusion.pipeline import ImageGenerationPipeline
from .upstream.diffusion.transformer import DiffusionTransformer
from .upstream.inference_profile import GENERATION_PROFILE, LAYER_PROFILE, InferenceProfile
from .upstream.modeling_bailingmm2 import BailingMM2NativeForConditionalGeneration
from .upstream.processing_bailingmm2 import load_bailingmm2_processor


def _text_conditioning_cache_key(encoder, vision_encoder, processor, text, task, bucket, width, height, image_key):
    key = (
        id(encoder), getattr(encoder, "_model_dtype", None),
        id(vision_encoder), getattr(vision_encoder, "_model_dtype", None),
        id(processor), text, task,
    )
    # Without a reference, Bailing's hidden-state extraction uses the tokenized
    # prompt only. Output geometry is consumed later by the diffusion pipeline.
    # A reference image is cropped and resized for the requested geometry.
    if task != "text-to-image":
        key += (bucket, width, height, image_key)
    return key


def _layer_count(prompt):
    match = re.search(r"decompose this image into\s+(\d+)\s+layers?", prompt, re.I)
    if match:
        return int(match.group(1))
    for line in prompt.splitlines():
        if line.strip().lower().startswith("number of layers:"):
            return int(line.split(":", 1)[1].strip())
    return 5


def _config(name, project=PROJECT):
    path = fl.locate_file(f"{project}/{name}")
    with open(path, encoding="utf-8") as reader:
        return json.load(reader)


def _kwargs(cls, config):
    supported = inspect.signature(cls.__init__).parameters
    return {key: value for key, value in config.items() if key in supported}


def _profile(transformer_config, vae_config, layer=False):
    profile = InferenceProfile(
        schema_version=1,
        inference_profile=LAYER_PROFILE if layer else GENERATION_PROFILE,
        alignment_padding_mode=transformer_config["alignment_padding_mode"],
        multi_frame_output=transformer_config["multi_frame_output"],
        vae_input_channels=vae_config["input_channels"],
        vae_sample_mode="argmax",
    )
    profile.validate()
    return profile


class MingVisionEncoder(nn.Module):
    def __init__(self, vision, linear_proj):
        super().__init__()
        self.vision = vision
        self.linear_proj = linear_proj

    def forward(self, pixel_values, grid_thw):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            image_embeds = self.vision(pixel_values, grid_thw=grid_thw)
        return F.normalize(self.linear_proj(image_embeds), dim=-1)


def _make_text_encoder(mllm_config, connector_config, mlp_config):
    config = BailingMM2Config.from_json_file(mllm_config)
    config._attn_implementation = "eager"
    config.llm_config._attn_implementation = "eager"
    connector_cfg = Qwen2Config(**connector_config)
    connector_cfg._attn_implementation = "sdpa"
    with torch.device("cpu"), init_empty_weights(include_buffers=True):
        encoder = BailingMM2NativeForConditionalGeneration(config)
        encoder.connector = Qwen2ForCausalLM(connector_cfg)
        encoder.connector.lm_head = nn.Identity()
        encoder.query_tokens_dict = nn.ParameterDict({
            f"{scale}x{scale}": nn.Parameter(torch.empty(
                scale * scale, config.llm_config.hidden_size, device="meta"
            )) for scale in mlp_config["img_gen_scales"]
        })
        encoder.proj_in = nn.Linear(config.llm_config.hidden_size, connector_cfg.hidden_size)
        encoder.proj_out = nn.Linear(connector_cfg.hidden_size, mlp_config["diffusion_c_input_dim"])
        direct_dim = config.llm_config.hidden_size * len(mlp_config["selected_hidden_states_layers"])
        encoder.proj_directvlm = nn.Sequential(
            RMSNorm(direct_dim, eps=1e-5),
            nn.Linear(direct_dim, mlp_config["diffusion_inner_dim"], bias=True),
        )
    # Rotary frequencies are non-persistent buffers and therefore absent from
    # the checkpoint.  Materialize them on CPU before MMGP verifies the model.
    for module in encoder.modules():
        frequency = getattr(module, "inv_freq", None)
        if frequency is None or not frequency.is_meta:
            continue
        if hasattr(module, "rope_init_fn"):
            module.inv_freq, module.attention_scaling = module.rope_init_fn(
                module.config, device="cpu"
            )
        else:
            positions = torch.arange(0, module.dim, 2, dtype=torch.float32, device="cpu")
            module.inv_freq = 1.0 / (module.theta ** (positions / module.dim))
        if hasattr(module, "original_inv_freq"):
            module.original_inv_freq = module.inv_freq
    for layer in encoder.connector.model.layers:
        layer.self_attn.is_causal = False
    encoder.img_gen_scales = mlp_config["img_gen_scales"]
    encoder.scale_indices = []
    count = 0
    for scale in encoder.img_gen_scales:
        count += scale * scale
        encoder.scale_indices.append(count)
    encoder.connector_norm = mlp_config["connector_norm"]
    encoder.use_learnable_token_condition = mlp_config["use_learnable_token_condition"]
    encoder.use_vlm_directvlm_condition = mlp_config["use_vlm_directvlm_condition"]
    encoder.selected_hidden_states_layers = mlp_config["selected_hidden_states_layers"]
    encoder.diffusion_inner_dim = mlp_config["diffusion_inner_dim"]
    encoder.loaded_image_gen_modules = True
    vision_encoder = MingVisionEncoder(encoder.vision, encoder.linear_proj)
    encoder.vision = None
    encoder.linear_proj = None
    return encoder, vision_encoder


def _preserve_dtype(module):
    module.eval().requires_grad_(False)
    module._convertWeightsFloatTo = None
    for child in module.modules():
        child._lock_dtype = None
    return module


def load_components(transformer_filename, text_encoder_filename, model_type, base_model_type,
                    save_quantized, quantize_transformer, model_def):
    from mmgp import offload

    if text_encoder_filename is None:
        raise ValueError("Ming Image requires its BailingMM2 text encoder checkpoint")
    layer = base_model_type == LAYER_ARCHITECTURE
    project = LAYER_PROJECT if layer else PROJECT
    encoder_folder = ENCODER
    transformer_config = _config("transformer_config.json", project)
    connector_config = _config("connector_config.json", project)
    mlp_config = _config("mlp_config.json", project)
    vae_config = _config("vae_config.json")
    scheduler_config = _config("scheduler_config.json", project)
    profile = _profile(transformer_config, vae_config, layer)

    for name in TOKENIZER_FILES:
        fl.locate_file(f"{encoder_folder}/{name}")
    tokenizer_dir = Path(fl.locate_file(f"{encoder_folder}/tokenizer.json")).parent

    set_phase_status("Loading - Transformer Model")
    with torch.device("cpu"), init_empty_weights(include_buffers=True):
        dit = DiffusionTransformer(**_kwargs(DiffusionTransformer, transformer_config))
        transformer = ConditionedTransformer(
            dit, vision_dim=mlp_config["diffusion_c_input_dim"],
            use_identity_mlp=mlp_config["use_identity_mlp"],
            text_encoder_norm=mlp_config["text_encoder_norm"],
        )
    offload.load_model_data(transformer, transformer_filename,
                            writable_tensors=False, default_dtype=torch.bfloat16)
    if save_quantized:
        from wgp import save_quantized_model
        save_quantized_model(transformer, model_type, transformer_filename,
                             torch.bfloat16, fl.locate_file(f"{project}/transformer_config.json"))
    _preserve_dtype(transformer)

    set_phase_status("Loading - Bailing Text Encoder")
    encoder, vision_encoder = _make_text_encoder(str(tokenizer_dir / "config.json"), connector_config, mlp_config)
    if model_def.get("ming_release") == "0.2":
        # This connector is always called with inputs_embeds from proj_in.
        encoder.connector.model.embed_tokens = None
    encoder.inference_profile = profile
    variant = model_def.get("ming_encoder_variant", "bf16")
    expected_suffix = f"_{variant}.safetensors"
    if not Path(text_encoder_filename).name.endswith(expected_suffix):
        raise ValueError("Ming encoder core and companion checkpoint precisions do not match")
    conditioner_filename = fl.locate_file(f"{project}/{conditioner_file(model_def)}")
    vision_filename = fl.locate_file(f"{SHARED_PROJECT}/vision_encoder{expected_suffix}")
    offload.load_model_data(encoder, [text_encoder_filename, conditioner_filename],
                            writable_tensors=False, default_dtype=torch.bfloat16)
    _preserve_dtype(encoder)
    set_phase_status("Loading - Bailing Vision Encoder")
    offload.load_model_data(vision_encoder, vision_filename,
                            writable_tensors=False, default_dtype=torch.bfloat16)
    _preserve_dtype(vision_encoder)
    processor = load_bailingmm2_processor(tokenizer_dir)

    set_phase_status("Loading - Image VAE")
    with torch.device("cpu"), init_empty_weights(include_buffers=True):
        vae = MingImageVAE(**_kwargs(MingImageVAE, vae_config))
    vae.register_to_config(scaling_factor=vae_config["scaling_factor"],
                           shift_factor=vae_config["shift_factor"])
    vae.input_channels = vae_config["input_channels"]
    offload.load_model_data(vae, fl.locate_file(f"{PROJECT}/vae.safetensors"),
                            writable_tensors=False, default_dtype=None)
    _preserve_dtype(vae)

    scheduler_config["use_dynamic_shifting"] = True
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
    pipeline = MingImagePipeline(encoder, vision_encoder, transformer, vae, processor, scheduler, profile,
                                 design_release=model_def.get("ming_release", "0.1"))
    return pipeline, {"transformer": transformer, "text_encoder": encoder,
                      "vision_encoder": vision_encoder, "vae": vae,
                      "tokenizer": processor.tokenizer}


class MingImagePipeline:
    def __init__(self, encoder, vision_encoder, transformer, vae, processor, scheduler, profile, design_release="0.1"):
        self.text_encoder = encoder
        self.vision_encoder = vision_encoder
        self.transformer = transformer
        self.vae = vae
        self.processor = processor
        self.profile = profile
        self.design_release = design_release
        self.text_encoder_cache = TextEncoderCache()
        self.diffusion = ImageGenerationPipeline(
            scheduler=scheduler, vae=vae, transformer=transformer,
            text_encoder=None, tokenizer=None,
        )

    @property
    def _interrupt(self):
        return getattr(self.diffusion, "_interrupt", False)

    @_interrupt.setter
    def _interrupt(self, value):
        self.diffusion._interrupt = bool(value)

    def prepare_preview_payload(self, latents, preview_meta=None):
        # WanGP builds its callback around the transformer, so convert here
        # before calling it. The scheduler supplies [B, C, H, W]; WanGP's
        # image preview expects [C, T, H, W]. Copy to CPU synchronously.
        composite = latents[0, :, 0] if latents.ndim == 5 else latents[0]
        return {"latents": composite.unsqueeze(1).detach().to(device="cpu")}

    @generation_progress
    @torch.inference_mode()
    def generate(self, input_prompt, seed=42, sampling_steps=12, guide_scale=1.0,
                 width=1024, height=1024, batch_size=1, input_ref_images=None,
                 input_frames=None, callback=None, set_progress_status=None,
                 VAE_tile_size=None, custom_settings=None, **kwargs):
        input_prompt = strip_comment_lines(input_prompt)
        if not input_prompt:
            raise ValueError("Ming Image prompt contains no content after removing comment lines")
        device = runtime.execution_device(self.transformer)
        reference = None
        images = list(input_ref_images or [])
        if input_frames is not None:
            images.insert(0, input_frames)
        if len(images) > 1:
            raise ValueError("Ming Image accepts one reference image")
        if images:
            reference = convert_tensor_to_image(images[0]) if torch.is_tensor(images[0]) else images[0]
        if self.profile.inference_profile == LAYER_PROFILE:
            task = "layer-decompose"
            num_layers = _layer_count(input_prompt)
            self.profile.validate_task(task, has_reference_image=reference is not None, num_layers=num_layers)
            bucket = 512 if max(width, height) < 768 else 1024
        else:
            task = "image-edit" if reference is not None else "text-to-image"
            num_layers = 1
            self.profile.validate_task(task, has_reference_image=reference is not None)
            if task == "image-edit":
                bucket = 1024
            elif self.design_release == "0.2":
                # Upstream chooses the closest tier by area, breaking ties toward 1024.
                bucket = 1024 if round(math.sqrt(width * height)) <= 1536 else 2048
            else:
                bucket = 1024 if max(width, height) < 1536 else 2048
        self.vae.use_slicing = True
        self.vae.configure_tiling(VAE_tile_size)

        content = []
        if reference is not None:
            content.append({"type": "image", "image": reference})
        content.append({"type": "text", "text": input_prompt})
        messages = [{"role": "HUMAN", "content": content}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        image_inputs, _, _ = self.processor.process_vision_info(messages)
        ref_rgba = reference.convert("RGBA") if reference is not None else None
        set_phase_status("Preparing Image Conditioning")
        processor_inputs = self.processor(
            text=[text], images=image_inputs, return_tensors="pt",
            image_gen_highres=bucket, image_gen_aspect_ratio=width / height,
            image_gen_ref_images=ref_rgba, image_gen_input_channels=4,
        ).to(device)
        for key, value in processor_inputs.items():
            if isinstance(value, torch.Tensor) and torch.is_floating_point(value):
                processor_inputs[key] = value.to(torch.bfloat16)
        ref_pixels = (processor_inputs.get("image_gen_pixel_values_reference")
                      if reference is not None else None)
        image_key = None if ref_rgba is None else (
            ref_rgba.size, hashlib.sha256(ref_rgba.tobytes()).digest()
        )
        cache_key = _text_conditioning_cache_key(
            self.text_encoder, self.vision_encoder, self.processor, text, task,
            bucket, width, height, image_key,
        )

        def encode_conditioning(_prompts):
            check_abort()
            encoder_layers = list(self.text_encoder.model.model.layers)
            # Whole-model MMGP profiles can bypass the connector's child hooks.
            # Count its transformer layers when the enclosing forward completes.
            encoder_layers.append((self.text_encoder.connector.model,
                                   len(self.text_encoder.connector.model.layers)))
            pixel_values = processor_inputs.pop("pixel_values", None)
            image_embeds = None
            if pixel_values is not None:
                set_phase_status("Encoding Reference Image")
                with text_encoding_progress(
                    self.vision_encoder.vision.blocks, title="Encoding Reference Image",
                    prompt_count=0, next_status="Encoding Text Prompt",
                    complete_on_success=True, completion_hold=0.35,
                ):
                    image_embeds = self.vision_encoder(pixel_values, processor_inputs["image_grid_thw"])
                del pixel_values
            set_phase_status("Encoding Text Prompt")
            with text_encoding_progress(encoder_layers, next_status="Preparing Image Conditioning",
                                        complete_on_success=True, completion_hold=0.35):
                positive, _, direct, _ = self.text_encoder.generate(
                    **processor_inputs, precomputed_image_embeds=image_embeds,
                    image_gen=True, image_gen_task=task,
                    image_gen_steps=sampling_steps, image_gen_cfg=guide_scale,
                    image_gen_only_extract_hidden_states=True,
                )
            check_abort()
            return [(positive, direct)]

        check_abort()
        positive, direct = self.text_encoder_cache.encode(
            encode_conditioning, [input_prompt], device=device, cache_keys=[cache_key]
        )[0]
        check_abort()
        del processor_inputs
        generated_h, generated_w = process_ratio(height, width, bucket)[0]
        if ref_pixels is not None:
            generated_h, generated_w = ref_pixels.shape[-2:]
        positive = [part for part in positive.unbind(0)] if positive is not None else None
        direct = [part for part in direct.unbind(0)] if direct is not None else None
        negative = [torch.zeros_like(part) for part in positive] if positive is not None else None
        negative_direct = [torch.zeros_like(part) for part in direct] if direct is not None else None
        def on_denoising_start(total_steps):
            check_abort()
            set_phase_status("Denoising")
            if callback is not None:
                callback(-1, None, True, override_num_inference_steps=total_steps)

        def on_step_end(pipe, step, timestep, payload):
            check_abort()
            if callback is not None:
                callback(int(step), self.prepare_preview_payload(payload["latents"]), False)
            return payload
        frames_per_image = num_layers + 1 if task == "layer-decompose" else 1
        # The pipeline creates latents in frame-major order for layer sets.
        # Separate generators give each output a stable, distinct noise seed.
        generator = torch.Generator(device=device).manual_seed(int(seed))
        if batch_size > 1:
            generator = [
                torch.Generator(device=device).manual_seed(int(seed) + index)
                for index in range(batch_size * frames_per_image)
            ]
        result = self.diffusion(
            prompt_embeds=positive, negative_prompt_embeds=negative,
            prompt_embeds_2=direct, negative_prompt_embeds_2=negative_direct,
            guidance_scale=guide_scale, num_inference_steps=sampling_steps,
            num_images_per_prompt=batch_size,
            generator=generator,
            height=generated_h, width=generated_w, device=device,
            ref_hidden_states=ref_pixels, sample_mode="argmax",
            num_frames_per_prompt=frames_per_image,
            callback_on_denoising_start=on_denoising_start,
            callback_on_step_end=on_step_end,
        ).images
        if task == "layer-decompose":
            expected = batch_size * (num_layers + 1)
            if len(result) != expected:
                raise RuntimeError(f"Design-Layer returned {len(result)} images; expected {expected}")
            layers = []
            with BytesIO() as archive_buffer:
                with ZipFile(archive_buffer, "w", compression=ZIP_DEFLATED) as archive:
                    # Diffusion returns all batch composites, then each layer's
                    # batch. Present complete layer sets in gallery order.
                    for batch_index in range(batch_size):
                        for index in range(1, num_layers + 1):
                            check_abort()
                            image = result[index * batch_size + batch_index]
                            image = image.resize((width, height), Image.Resampling.LANCZOS).convert("RGBA")
                            layers.append(torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1))
                            with BytesIO() as image_buffer:
                                image.save(image_buffer, format="PNG")
                                name = f"layer_{index:02d}.png"
                                if batch_size > 1:
                                    name = f"set_{batch_index + 1:02d}/{name}"
                                archive.writestr(name, image_buffer.getvalue())
                return {"x": torch.stack(layers, dim=1), "side_files": {".zip": archive_buffer.getvalue()}}
        keep_alpha = (custom_settings or {}).get("rgba", "Disabled") == "Enabled"
        output = []
        for image in result:
            check_abort()
            image = image.resize((width, height), Image.Resampling.LANCZOS)
            image = image.convert("RGBA" if keep_alpha else "RGB")
            output.append(torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1))
        return torch.stack(output, dim=0).transpose(0, 1)
