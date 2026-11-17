"""Chain-of-Zoom runtime (https://github.com/bryanswkim/Chain-of-Zoom).

optimized by DeepBeepMeep
"""

from __future__ import annotations
from shared.utils.media_control import checkpoint_modules
from shared.utils.media_control import controlled_model_loading, loading_callback

import gc
import math
import os
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from mmgp import offload
from shared.utils import offload_registry


COZ_LATENT_TILE_SIZE = 64
COZ_LATENT_TILE_OVERLAP = 16
COZ_STEP_FACTORS = {2.0: (2,), 4.0: (4,), 8.0: (4, 2), 16.0: (4, 4)}
COZ_VLM_CONTEXT_SIZE = 512
COZ_VLM_MAX_NEW_TOKENS = 16
COZ_VLM_VISION_BATCH = 8  # tile image pairs encoded per vision tower pass
COZ_VLM_PROMPT_BATCH = 16  # tile prompts decoded concurrently by the LM engine
COZ_VLM_PROCESSOR_USE_FAST = True  # fast (torchvision) image preprocessing; cheaper CPU prep, slightly different pixel values
COZ_VLM_MESSAGE = "The second image is a zoom-in of the first image. Based on this knowledge, what is in the second image? Give me a set of English words."
COZ_CLIP_MAX_LENGTH = 77
COZ_T5_MAX_LENGTH = 512
COZ_CLIP_TEXT_BATCH_SIZE = 1
COZ_T5_TEXT_BATCH_SIZE = 1
COZ_TIMESTEP = 1000.0

_CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")


@dataclass
class CoZPaths:
    transformer: str
    vae: str
    clip_l: str
    clip_tokenizer_dir: str  # CLIP-L tokenizer files; CLIP-G uses the same vocab with pad token "!"
    clip_g: str
    t5: str
    vlm: str
    vlm_dir: str  # folder holding the VLM config/processor/tokenizer files


def _config_path(filename: str) -> str:
    return os.path.join(_CONFIGS_DIR, filename)


def _abort_requested(abort_callback) -> bool:
    return callable(abort_callback) and abort_callback()


def _report_progress(progress_callback, phase: str, current_step: int | None = None, total_steps: int | None = None, *, label_step: int | None = None, label_total: int | None = None) -> None:
    if callable(progress_callback):
        del label_step, label_total
        progress_callback(phase, current_step, total_steps)


def _coz_phase(phase: str, step_label: str) -> str:
    return f"CoZ {phase} {step_label}"


def _batch_count(total_items: int, batch_size: int) -> int:
    return (int(total_items) + int(batch_size) - 1) // int(batch_size)


def _grid_positions(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    positions = list(range(0, length - tile + 1, stride))
    if positions[-1] != length - tile:
        positions.append(length - tile)
    return positions


def _gaussian_weights(width: int, height: int, device) -> torch.Tensor:
    var = 0.01
    midpoint_x = (width - 1) / 2
    x_probs = [math.exp(-(x - midpoint_x) * (x - midpoint_x) / (width * width) / (2 * var)) / math.sqrt(2 * math.pi * var) for x in range(width)]
    midpoint_y = height / 2
    y_probs = [math.exp(-(y - midpoint_y) * (y - midpoint_y) / (height * height) / (2 * var)) / math.sqrt(2 * math.pi * var) for y in range(height)]
    return torch.tensor(np.outer(y_probs, x_probs), device=device, dtype=torch.float32)


class ChainOfZoomRuntime:
    def __init__(self) -> None:
        self.dtype = torch.bfloat16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.transformer = None
        self.vae = None
        self.clip_l = None
        self.clip_g = None
        self.t5 = None
        self.vlm = None
        self.vlm_visual = None
        self.vlm_text = None
        self.vlm_engine = None
        self.vlm_config = None
        self.vlm_generation_config = None
        self.tokenizer_l = None
        self.tokenizer_g = None
        self.tokenizer_t5 = None
        self.vlm_processor = None
        self.vlm_llm = None
        self.offloadobj = None
        self.profile = None
        self.paths: CoZPaths | None = None
        self.use_lm_engine = False
        self.vlm_lm_features = None
        self.vlm_vision_batch = COZ_VLM_VISION_BATCH
        self.vlm_prompt_batch = COZ_VLM_PROMPT_BATCH

    @controlled_model_loading
    def load(self, paths: CoZPaths, profile, init_pipe, lm_features, vlm_vision_batch: int = COZ_VLM_VISION_BATCH, vlm_prompt_batch: int = COZ_VLM_PROMPT_BATCH) -> None:
        vlm_vision_batch = int(vlm_vision_batch)
        vlm_prompt_batch = int(vlm_prompt_batch)
        if self.offloadobj is not None and self.paths == paths and self.profile == profile and self.vlm_lm_features == lm_features and self.vlm_vision_batch == vlm_vision_batch and self.vlm_prompt_batch == vlm_prompt_batch:
            return
        self.release()
        from accelerate import init_empty_weights
        from shared.image_vae.autoencoder_kl import AutoencoderKL  # diffusers' AutoencoderKL with lower VRAM peaks (large convolutions by bands of rows)
        from transformers import AutoConfig, AutoProcessor, CLIPTextModelWithProjection, CLIPTokenizer, GenerationConfig, Qwen2_5_VLForConditionalGeneration, T5Tokenizer
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
        from shared.llm_engines.lmgp.models.qwen2_5_vl import Qwen2_5_VLForCausalLM
        from shared.llm_engines.lmgp.vllm_support import TextEngine, configure_engine_kernels
        from postprocessing.chain_of_zoom.sd3_transformer import SD3Transformer, SD3_MEDIUM_CONFIG

        with init_empty_weights(include_buffers=True):
            self.transformer = SD3Transformer(**SD3_MEDIUM_CONFIG)
        self.transformer.init_runtime_buffers()
        offload.load_model_data(self.transformer, paths.transformer, writable_tensors=False, default_dtype=self.dtype, ignore_unused_weights=True, verboseLevel=-1)
        self.vae = offload.fast_load_transformers_model(paths.vae, modelClass=AutoencoderKL, defaultConfigPath=_config_path("sd3_vae_config.json"), writable_tensors=False)
        self.vae.enable_tiling()
        self.clip_l = offload.fast_load_transformers_model(paths.clip_l, modelClass=CLIPTextModelWithProjection, forcedConfigPath=os.path.join(os.path.dirname(paths.clip_l), "config.json"), ignore_unused_weights=True, writable_tensors=False)
        self.clip_g = offload.fast_load_transformers_model(paths.clip_g, modelClass=CLIPTextModelWithProjection, forcedConfigPath=os.path.join(os.path.dirname(paths.clip_g), "config.json"), writable_tensors=False)
        self.t5 = offload.fast_load_transformers_model(paths.t5, writable_tensors=False)
        self.vlm_config = AutoConfig.from_pretrained(paths.vlm_dir)
        self.vlm_generation_config = GenerationConfig.from_pretrained(paths.vlm_dir)
        self.vlm_generation_config.do_sample = False
        self.vlm_generation_config.temperature = None
        self.use_lm_engine = lm_features.graphs  # the LM engine's decoder (MMGP graphs), otherwise transformers
        self.vlm_lm_features = lm_features
        self.vlm_vision_batch = vlm_vision_batch
        self.vlm_prompt_batch = vlm_prompt_batch
        print(f"[Chain-of-Zoom] VLM LM: {lm_features.describe()}")
        print(f"[Chain-of-Zoom] VLM tile batch sizes: vision tower={self.vlm_vision_batch}, decoder={self.vlm_prompt_batch}")
        vision_attn = "flash_attention_2" if lm_features.flash else None
        if self.use_lm_engine:
            with init_empty_weights(include_buffers=True):
                self.vlm = Qwen2_5_VLForConditionalGeneration(self.vlm_config)
                self.vlm_text = Qwen2_5_VLForCausalLM(self.vlm_config.text_config)
            with init_empty_weights(include_buffers=False):
                if vision_attn:
                    self.vlm_visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(self.vlm_config.vision_config, attn_implementation=vision_attn, torch_dtype=self.dtype)
                else:
                    self.vlm_visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(self.vlm_config.vision_config)
            offload.load_model_data(self.vlm_visual, paths.vlm, modelPrefix="model.visual", writable_tensors=False, default_dtype=self.dtype, ignore_unused_weights=True, verboseLevel=-1)
            offload.load_model_data(self.vlm_text, paths.vlm, modelPrefix="model.language_model", writable_tensors=False, default_dtype=self.dtype, ignore_unused_weights=True, verboseLevel=-1)
            configure_engine_kernels(self.vlm_text, lm_features)
        else:
            self.vlm = offload.fast_load_transformers_model(paths.vlm, modelClass=Qwen2_5_VLForConditionalGeneration, defaultConfigPath=os.path.join(paths.vlm_dir, "config.json"), writable_tensors=False)
            self.vlm.tie_weights()
            self.vlm.generation_config = self.vlm_generation_config
            if vision_attn:
                self.vlm.model.visual.config._attn_implementation = vision_attn
        for model in (self.transformer, self.vae, self.clip_l, self.clip_g, self.t5, self.vlm_visual if self.use_lm_engine else self.vlm, self.vlm_text):
            if model is None:
                continue
            model.eval().requires_grad_(False)
        self.tokenizer_l = CLIPTokenizer.from_pretrained(paths.clip_tokenizer_dir)
        # SD3's tokenizer_2 is the same CLIP vocab with pad token "!" (id 0)
        self.tokenizer_g = CLIPTokenizer.from_pretrained(paths.clip_tokenizer_dir, pad_token="!")
        self.tokenizer_t5 = T5Tokenizer.from_pretrained(os.path.dirname(paths.t5))
        self.vlm_processor = AutoProcessor.from_pretrained(paths.vlm_dir, use_fast=COZ_VLM_PROCESSOR_USE_FAST)
        if self.use_lm_engine:
            self.vlm_engine = TextEngine(model=self.vlm_text, model_path=paths.vlm_dir, tokenizer=self.vlm_processor.tokenizer, enforce_eager=False)
            self.vlm_llm = self.vlm_text
        else:
            self.vlm_llm = torch.nn.ModuleDict({"language_model": self.vlm.model.language_model, "lm_head": self.vlm.lm_head})

        pipe = {
            "transformer": self.transformer,
            "vae": self.vae,
            "clip_l": self.clip_l,
            "clip_g": self.clip_g,
            "t5": self.t5,
            "vlm_visual": self.vlm_visual if self.use_lm_engine else self.vlm.model.visual,
            "vlm_llm": self.vlm_llm,
        }
        kwargs = {}
        profile_no = init_pipe(pipe, kwargs, profile)
        kwargs.setdefault("budgets", {})["clip_l"] = 0
        kwargs["budgets"]["clip_g"] = 0
        kwargs["budgets"]["vlm_visual"] = 0
        if self.use_lm_engine:
            kwargs["budgets"]["vlm_llm"] = 0
        kwargs.setdefault("coTenantsMap", {}).update({"clip_l": ["clip_g"], "clip_g": ["clip_l"]})
        kwargs["pinnedMemory"] = False
        self.offloadobj = offload.profile(pipe, loading_callback=loading_callback(), profile_no=profile_no, quantizeTransformer=False, convertWeightsFloatTo=self.dtype, verboseLevel=-1, **kwargs)
        offload_registry.register_offloadobj("Chain-of-Zoom", self.offloadobj, self.release)
        self.profile = profile
        self.paths = paths

    def _unload_mmgp(self) -> None:
        if self.vlm_engine is not None:
            self.vlm_engine.release_runtime_allocations()
        if self.offloadobj is not None:
            self.offloadobj.unload_all()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _current_vlm_engine_label(self) -> str | None:
        if self.vlm_lm_features is None:
            return None
        return "LM engine" if self.use_lm_engine else "transformers decoder"

    def release(self) -> None:
        vlm_engine_label = self._current_vlm_engine_label()
        if vlm_engine_label is not None:
            print(f"[Chain-of-Zoom] Closing VLM {vlm_engine_label}")
        # CUDA graphs hold model/KV pointers; clear them before MMGP releases model storage.
        if self.vlm_engine is not None:
            self.vlm_engine.close()
            self.vlm_engine = None
        if self.offloadobj is not None:
            offload_registry.unregister_offloadobj("Chain-of-Zoom", self.offloadobj)
            self.offloadobj.release()
            self.offloadobj = None
        self.transformer = None
        self.vae = None
        self.clip_l = None
        self.clip_g = None
        self.t5 = None
        self.vlm = None
        self.vlm_visual = None
        self.vlm_text = None
        self.vlm_config = None
        self.vlm_generation_config = None
        self.tokenizer_l = None
        self.tokenizer_g = None
        self.tokenizer_t5 = None
        self.vlm_processor = None
        self.vlm_llm = None
        self.use_lm_engine = False
        self.vlm_lm_features = None
        self.vlm_vision_batch = COZ_VLM_VISION_BATCH
        self.vlm_prompt_batch = COZ_VLM_PROMPT_BATCH
        self.profile = None
        self.paths = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _vlm_inputs(self, context_image: Image.Image, tile_image: Image.Image):
        messages = [
            {"role": "system", "content": COZ_VLM_MESSAGE},
            {"role": "user", "content": [{"type": "image"}, {"type": "image"}]},
        ]
        text = self.vlm_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.vlm_processor(text=[text], images=[context_image, tile_image], padding=True, return_tensors="pt")

    def _vlm_visual_embeds(self, inputs_batch: list) -> list[torch.Tensor]:
        """One vision tower pass over the images of several tile jobs; returns per-job packed embeds."""
        pixel_values = torch.cat([inputs["pixel_values"] for inputs in inputs_batch], dim=0).to(self.device)
        image_grid_thw = torch.cat([inputs["image_grid_thw"] for inputs in inputs_batch], dim=0).to(self.device)
        if self.use_lm_engine:
            pixel_values = pixel_values.type(self.vlm_visual.dtype)
            with checkpoint_modules(self.vlm_visual.blocks):
                image_embeds = self.vlm_visual(pixel_values, grid_thw=image_grid_thw)
            merge_size = self.vlm_visual.spatial_merge_size
        else:
            with checkpoint_modules(self.vlm.model.visual.blocks):
                image_embeds = torch.cat(self.vlm.model.get_image_features(pixel_values, image_grid_thw), dim=0)
            merge_size = self.vlm.model.visual.spatial_merge_size
        tokens_per_image = (image_grid_thw.prod(-1) // merge_size**2).tolist()
        job_sizes = []
        image_index = 0
        for inputs in inputs_batch:
            image_count = int(inputs["image_grid_thw"].shape[0])
            job_sizes.append(int(sum(tokens_per_image[image_index:image_index + image_count])))
            image_index += image_count
        image_embeds = image_embeds.to(self.dtype).cpu()
        del pixel_values, image_grid_thw
        return list(torch.split(image_embeds, job_sizes, dim=0))

    def _vlm_prompts_from_embeds_lm_engine(self, jobs: list[dict]) -> list[str]:
        """Decode the prompts of several tile jobs concurrently through the LM engine."""
        requests = []
        for job in jobs:
            input_ids = job["input_ids"].to(self.device)
            attention_mask = job.get("attention_mask")
            attention_mask = attention_mask.to(self.device) if attention_mask is not None else torch.ones_like(input_ids, device=self.device)
            image_grid_thw = job["image_grid_thw"].to(self.device)
            image_embeds = job["image_embeds"]
            inputs_embeds = self.vlm_text.embed_tokens(input_ids)
            image_mask = input_ids == self.vlm_config.image_token_id
            if int(image_mask.sum()) != image_embeds.shape[0]:
                raise ValueError(f"VLM image token/features mismatch: tokens={int(image_mask.sum())}, features={image_embeds.shape[0]}")
            image_embeds = image_embeds.to(device=self.device, dtype=inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask.unsqueeze(-1).expand_as(inputs_embeds), image_embeds)
            position_ids, rope_deltas = self.vlm.model.get_rope_index(input_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask)
            active_mask = attention_mask[0].bool()
            requests.append({
                "prompt_token_ids": [int(token_id) for token_id in input_ids[0][active_mask].tolist()],
                "prompt_embeds": inputs_embeds[0, active_mask].contiguous().cpu(),
                "prompt_position_ids": position_ids[:, 0, active_mask].contiguous().cpu(),
                "position_offset": int(rope_deltas.reshape(-1)[0].item()) if torch.is_tensor(rope_deltas) and rope_deltas.numel() > 0 else 0,
            })
            del input_ids, attention_mask, image_grid_thw, inputs_embeds, image_embeds, position_ids, rope_deltas
        results = self.vlm_engine.generate_embedded_batch(
            requests,
            max_tokens=COZ_VLM_MAX_NEW_TOKENS,
            temperature=None,
            top_p=None,
            top_k=1,
            cfg_scale=1.0,
            seed=None,
            use_tqdm=False,
            release_vram_after=False,
            ignore_eos=False,
            repetition_penalty=float(self.vlm_generation_config.repetition_penalty or 1.0),
        )
        requests.clear()
        token_id_lists = [[] if result is None else result.get("token_ids", []) for result in results or []]
        return [prompt.strip() for prompt in self.vlm_processor.batch_decode(token_id_lists, skip_special_tokens=True, clean_up_tokenization_spaces=False)]

    def _vlm_prompts_from_embeds_legacy(self, jobs: list[dict]) -> list[str]:
        input_ids = torch.cat([job["input_ids"] for job in jobs], dim=0).to(self.device)
        attention_mask = torch.cat([
            job["attention_mask"] if job.get("attention_mask") is not None else torch.ones_like(job["input_ids"])
            for job in jobs
        ], dim=0).to(self.device)
        image_grid_thw = torch.cat([job["image_grid_thw"] for job in jobs], dim=0).to(self.device)
        image_embeds = torch.cat([job["image_embeds"] for job in jobs], dim=0)
        inputs_embeds = self.vlm.model.get_input_embeddings()(input_ids)
        image_mask = input_ids == self.vlm.config.image_token_id
        if int(image_mask.sum()) != image_embeds.shape[0]:
            raise ValueError(f"VLM image token/features mismatch: tokens={int(image_mask.sum())}, features={image_embeds.shape[0]}")
        image_embeds = image_embeds.to(device=self.device, dtype=inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask.unsqueeze(-1).expand_as(inputs_embeds), image_embeds)
        self.vlm.model.rope_deltas = None
        position_ids, rope_deltas = self.vlm.model.get_rope_index(input_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask)
        self.vlm.model.rope_deltas = rope_deltas
        eos_token_ids = self.vlm_generation_config.eos_token_id
        eos_token_ids = {eos_token_ids} if isinstance(eos_token_ids, int) else set(eos_token_ids or [])
        forced_eos_token_id = int(next(iter(eos_token_ids))) if eos_token_ids else None
        repetition_penalty = float(self.vlm_generation_config.repetition_penalty or 1.0)
        generated = []
        finished = torch.zeros((input_ids.shape[0], 1), device=self.device, dtype=torch.bool)
        past_key_values = None
        current_input_ids = input_ids
        current_inputs_embeds = inputs_embeds
        current_position_ids = position_ids
        cache_position = torch.arange(input_ids.shape[1], device=self.device, dtype=torch.long)
        with checkpoint_modules(self.vlm.model.language_model.layers):
            for _ in range(COZ_VLM_MAX_NEW_TOKENS):
                output = self.vlm(
                    input_ids=current_input_ids,
                    inputs_embeds=current_inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=current_position_ids,
                    image_grid_thw=image_grid_thw,
                    past_key_values=past_key_values,
                    use_cache=True,
                    cache_position=cache_position,
                    logits_to_keep=1,
                )
                logits = output.logits[:, -1, :]
                if repetition_penalty != 1.0:
                    previous_tokens = torch.cat([input_ids] + generated, dim=1) if generated else input_ids
                    for row_no in range(input_ids.shape[0]):
                        previous_ids = torch.unique(previous_tokens[row_no])
                        previous_scores = logits[row_no, previous_ids]
                        logits[row_no, previous_ids] = torch.where(previous_scores < 0, previous_scores * repetition_penalty, previous_scores / repetition_penalty)
                next_token = logits.argmax(dim=-1, keepdim=True)
                if forced_eos_token_id is not None:
                    next_token = torch.where(finished, torch.full_like(next_token, forced_eos_token_id), next_token)
                generated.append(next_token)
                past_key_values = output.past_key_values
                del output, logits
                if eos_token_ids:
                    token_finished = torch.zeros_like(finished)
                    for eos_token_id in eos_token_ids:
                        token_finished |= next_token == int(eos_token_id)
                    finished |= token_finished
                    if bool(finished.all()):
                        break
                attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=1)
                cache_position = cache_position[-1:] + 1
                current_input_ids = next_token
                current_inputs_embeds = None
                current_position_ids = cache_position.view(1, 1, -1).expand(3, input_ids.shape[0], -1) + rope_deltas.view(1, input_ids.shape[0], 1).to(self.device)
        generated_ids = torch.cat(generated, dim=1)
        prompts = self.vlm_processor.batch_decode(generated_ids.cpu(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
        del input_ids, attention_mask, image_grid_thw, inputs_embeds, image_embeds, position_ids, rope_deltas, generated_ids, generated, finished, past_key_values
        return [prompt.strip() for prompt in prompts]

    def _encode_clip_prompts(self, text_encoder, tokenizer, prompts: list[str], phase: str, progress_start: int, progress_total: int, abort_callback=None, progress_callback=None) -> tuple[torch.Tensor, torch.Tensor] | None:
        hidden_batches = []
        pooled_batches = []
        total = len(prompts)
        for batch_no, start in enumerate(range(0, total, COZ_CLIP_TEXT_BATCH_SIZE)):
            if _abort_requested(abort_callback):
                return None
            current_step = progress_start + batch_no + 1
            _report_progress(progress_callback, phase, current_step, progress_total)
            ids = tokenizer(prompts[start:start + COZ_CLIP_TEXT_BATCH_SIZE], padding="max_length", max_length=COZ_CLIP_MAX_LENGTH, truncation=True, return_tensors="pt").input_ids
            with checkpoint_modules(text_encoder.text_model.encoder.layers):
                output = text_encoder(ids.to(self.device), output_hidden_states=True)
            hidden_batches.append(output.hidden_states[-2].to(self.dtype).cpu())
            pooled_batches.append(output[0].to(self.dtype).cpu())
            del ids, output
        return torch.cat(hidden_batches), torch.cat(pooled_batches)

    def _encode_t5_prompts(self, prompts: list[str], phase: str, progress_start: int, progress_total: int, abort_callback=None, progress_callback=None) -> torch.Tensor | None:
        batches = []
        total = len(prompts)
        for batch_no, start in enumerate(range(0, total, COZ_T5_TEXT_BATCH_SIZE)):
            if _abort_requested(abort_callback):
                return None
            current_step = progress_start + batch_no + 1
            _report_progress(progress_callback, phase, current_step, progress_total)
            ids = self.tokenizer_t5(prompts[start:start + COZ_T5_TEXT_BATCH_SIZE], padding="max_length", max_length=COZ_T5_MAX_LENGTH, truncation=True, add_special_tokens=True, return_tensors="pt").input_ids
            with checkpoint_modules(self.t5.encoder.block):
                batches.append(self.t5(ids.to(self.device))[0].to(self.dtype).cpu())
            del ids
        return torch.cat(batches)

    def _encode_prompts(self, prompts: list[str], step_label: str, abort_callback=None, progress_callback=None) -> dict[str, tuple[torch.Tensor, torch.Tensor]] | None:
        unique_prompts = list(dict.fromkeys(prompts))
        clip_batches = _batch_count(len(unique_prompts), COZ_CLIP_TEXT_BATCH_SIZE)
        t5_batches = _batch_count(len(unique_prompts), COZ_T5_TEXT_BATCH_SIZE)
        text_progress_total = clip_batches * 2 + t5_batches
        clip1 = self._encode_clip_prompts(self.clip_l, self.tokenizer_l, unique_prompts, _coz_phase("Text Encoding CLIP-L", step_label), 0, text_progress_total, abort_callback=abort_callback, progress_callback=progress_callback)
        if clip1 is None:
            return None
        clip2 = self._encode_clip_prompts(self.clip_g, self.tokenizer_g, unique_prompts, _coz_phase("Text Encoding CLIP-G", step_label), clip_batches, text_progress_total, abort_callback=abort_callback, progress_callback=progress_callback)
        if clip2 is None:
            return None
        t5_emb = self._encode_t5_prompts(unique_prompts, _coz_phase("Text Encoding T5", step_label), clip_batches * 2, text_progress_total, abort_callback=abort_callback, progress_callback=progress_callback)
        if t5_emb is None:
            return None
        embeds_cache = {}
        clip1_emb, pool1 = clip1
        clip2_emb, pool2 = clip2
        for prompt_no, prompt in enumerate(unique_prompts):
            clip_emb = torch.cat([clip1_emb[prompt_no:prompt_no + 1], clip2_emb[prompt_no:prompt_no + 1]], dim=-1)
            clip_emb = torch.nn.functional.pad(clip_emb, (0, t5_emb.shape[-1] - clip_emb.shape[-1]))
            prompt_emb = torch.cat([clip_emb, t5_emb[prompt_no:prompt_no + 1]], dim=-2)
            pooled_emb = torch.cat([pool1[prompt_no:prompt_no + 1], pool2[prompt_no:prompt_no + 1]], dim=-1)
            embeds_cache[prompt] = (prompt_emb, pooled_emb)
        return embeds_cache

    def _sr_step(self, image: Image.Image, context_image: Image.Image, out_width: int, out_height: int, seed: int, step_label: str, abort_callback=None, progress_callback=None) -> Image.Image | None:
        up_image = image.resize((out_width, out_height), Image.LANCZOS)
        latent_height, latent_width = out_height // 8, out_width // 8
        tile_h = min(COZ_LATENT_TILE_SIZE, latent_height)
        tile_w = min(COZ_LATENT_TILE_SIZE, latent_width)
        stride_h = max(1, tile_h - COZ_LATENT_TILE_OVERLAP)
        stride_w = max(1, tile_w - COZ_LATENT_TILE_OVERLAP)
        positions = [(y0, x0) for y0 in _grid_positions(latent_height, tile_h, stride_h) for x0 in _grid_positions(latent_width, tile_w, stride_w)]
        total_tiles = len(positions)

        # phase 1: multi-scale aware prompt per tile (original image as zoom-out context,
        # tile crop of the pre-upsampled image as zoom-in; equivalent to upstream's decoded
        # latent patch, without the extra per-tile VAE decode). Split Qwen2.5-VL into a
        # visual pass over every tile, then an LLM pass over cached visual embeddings.
        # every tile shares the same chat template and image sizes, so the prompt token ids
        # and the context image preprocessing are computed once; per tile only the crop
        # itself goes through the (CPU bound) image processor, on a small thread pool
        from concurrent.futures import ThreadPoolExecutor
        from shared.utils.utils import get_default_workers

        first_inputs = self._vlm_inputs(context_image, up_image.crop((positions[0][1] * 8, positions[0][0] * 8, (positions[0][1] + tile_w) * 8, (positions[0][0] + tile_h) * 8)))
        # the fast image processor may return CUDA tensors; jobs are staged on CPU
        first_inputs = {key: value.cpu() if torch.is_tensor(value) else value for key, value in first_inputs.items()}
        context_patches = int(first_inputs["image_grid_thw"][0].prod())
        template = {
            "input_ids": first_inputs["input_ids"].cpu(),
            "attention_mask": first_inputs["attention_mask"].cpu() if "attention_mask" in first_inputs else None,
            "context_grid": first_inputs["image_grid_thw"][:1].cpu(),
            "context_pixel_values": first_inputs["pixel_values"][:context_patches].cpu(),
            "tile_grid": first_inputs["image_grid_thw"][1:].cpu(),
        }

        def tile_inputs(tile_image: Image.Image) -> dict:
            tile_features = self.vlm_processor.image_processor(images=[tile_image], return_tensors="pt")
            tile_grid = tile_features["image_grid_thw"].cpu()
            if not torch.equal(tile_grid, template["tile_grid"]):
                raise RuntimeError(f"VLM tile grid mismatch: {tile_grid.tolist()} vs {template['tile_grid'].tolist()}")
            return {
                "input_ids": template["input_ids"],
                "attention_mask": template["attention_mask"],
                "image_grid_thw": torch.cat([template["context_grid"], tile_grid], dim=0),
                "pixel_values": torch.cat([template["context_pixel_values"], tile_features["pixel_values"].cpu()], dim=0),
            }

        vlm_jobs = []
        vision_phase = _coz_phase("VLM Vision", step_label)
        with ThreadPoolExecutor(max_workers=max(1, min(int(get_default_workers()), self.vlm_vision_batch))) as executor:
            for batch_start in range(0, total_tiles, self.vlm_vision_batch):
                if _abort_requested(abort_callback):
                    return None
                batch_positions = positions[batch_start:batch_start + self.vlm_vision_batch]
                _report_progress(progress_callback, vision_phase, batch_start + len(batch_positions), total_tiles)
                if batch_start == 0:
                    batch_positions = batch_positions[1:]
                tile_images = [up_image.crop((x0 * 8, y0 * 8, (x0 + tile_w) * 8, (y0 + tile_h) * 8)) for y0, x0 in batch_positions]
                inputs_batch = ([first_inputs] if batch_start == 0 else []) + list(executor.map(tile_inputs, tile_images))
                del tile_images
                image_embeds_batch = self._vlm_visual_embeds(inputs_batch)
                for inputs, image_embeds in zip(inputs_batch, image_embeds_batch):
                    attention_mask = inputs.get("attention_mask")
                    vlm_jobs.append({
                        "input_ids": inputs["input_ids"].cpu(),
                        "attention_mask": attention_mask.cpu() if attention_mask is not None else None,
                        "image_grid_thw": inputs["image_grid_thw"].cpu(),
                        "image_embeds": image_embeds,
                    })
                del inputs_batch, image_embeds_batch
        first_inputs = template = None

        prompts = []
        text_phase = _coz_phase("VLM Text", step_label)
        if self.use_lm_engine and vlm_jobs:
            max_prompt_len = max(int(job["attention_mask"].sum().item()) if job["attention_mask"] is not None else int(job["input_ids"].shape[-1]) for job in vlm_jobs)
            self.vlm_engine.reserve_runtime(prompt_len=max_prompt_len, max_tokens=COZ_VLM_MAX_NEW_TOKENS, cfg_scale=1.0, num_seqs=min(self.vlm_prompt_batch, len(vlm_jobs)))
            for batch_start in range(0, len(vlm_jobs), self.vlm_prompt_batch):
                if _abort_requested(abort_callback):
                    return None
                batch_jobs = vlm_jobs[batch_start:batch_start + self.vlm_prompt_batch]
                _report_progress(progress_callback, text_phase, batch_start + len(batch_jobs), total_tiles)
                for tile_offset, prompt in enumerate(self._vlm_prompts_from_embeds_lm_engine(batch_jobs)):
                    prompts.append(prompt)
                    print(f"[Chain-of-Zoom] {step_label} VLM tile {batch_start + tile_offset + 1}/{total_tiles} prompt: {prompt}")
                for job in batch_jobs:
                    job.clear()
        else:
            for batch_start in range(0, len(vlm_jobs), self.vlm_prompt_batch):
                if _abort_requested(abort_callback):
                    return None
                batch_jobs = vlm_jobs[batch_start:batch_start + self.vlm_prompt_batch]
                _report_progress(progress_callback, text_phase, batch_start + len(batch_jobs), total_tiles)
                for tile_offset, prompt in enumerate(self._vlm_prompts_from_embeds_legacy(batch_jobs)):
                    prompts.append(prompt)
                    print(f"[Chain-of-Zoom] {step_label} VLM tile {batch_start + tile_offset + 1}/{total_tiles} prompt: {prompt}")
                for job in batch_jobs:
                    job.clear()
        vlm_jobs.clear()
        if self.use_lm_engine and self.vlm_engine is not None:
            # VLM phase over: free the engine KV cache + CUDA graph pool before the
            # text-encode/VAE/diffusion phases. MMGP will move the weights anyway, which
            # invalidates the captured graphs; the engine rebuilds them next VLM phase.
            self.vlm_engine.release_runtime_allocations()

        # phase 2: text embeddings (cached per unique prompt, stored on CPU)
        embeds_cache = self._encode_prompts(prompts, step_label, abort_callback=abort_callback, progress_callback=progress_callback)
        if embeds_cache is None:
            return None

        # phase 3: VAE encode (OSEDiff convention: scaling factor only, no shift)
        if _abort_requested(abort_callback):
            return None
        _report_progress(progress_callback, _coz_phase("VAE Encode", step_label))
        x_full = torch.from_numpy(np.array(up_image)).permute(2, 0, 1).unsqueeze(0).to(device=self.device, dtype=self.dtype).div_(127.5).sub_(1.0)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        with checkpoint_modules(module for module in self.vae.modules() if not module._modules):
            z_full = self.vae.encode(x_full).latent_dist.sample(generator) * self.vae.config.scaling_factor
        del x_full

        # phase 4: one-step OSEDiff on each tile, Gaussian-blended accumulation
        weights = _gaussian_weights(tile_w, tile_h, self.device)
        z_acc = torch.zeros_like(z_full, dtype=torch.float32)
        norm = torch.zeros_like(z_full, dtype=torch.float32)
        timestep = torch.tensor([COZ_TIMESTEP], device=self.device, dtype=self.dtype)
        diffusion_phase = _coz_phase("Diffusion", step_label)
        _report_progress(progress_callback, diffusion_phase, 0, total_tiles)
        for tile_no, (y0, x0) in enumerate(positions):
            if _abort_requested(abort_callback):
                return None
            prompt_emb, pooled_emb = embeds_cache[prompts[tile_no]]
            patch = z_full[:, :, y0:y0 + tile_h, x0:x0 + tile_w]
            pred_v = self.transformer(patch, timestep, prompt_emb.to(device=self.device, dtype=self.dtype), pooled_emb.to(device=self.device, dtype=self.dtype))
            tile_out = (patch - pred_v).float()
            z_acc[:, :, y0:y0 + tile_h, x0:x0 + tile_w] += tile_out * weights
            norm[:, :, y0:y0 + tile_h, x0:x0 + tile_w] += weights
            del patch, pred_v, tile_out
            _report_progress(progress_callback, diffusion_phase, tile_no + 1, total_tiles)
        embeds_cache.clear()
        z_full = (z_acc / (norm + 1e-10)).to(self.dtype)
        del z_acc, norm

        # phase 5: VAE decode
        if _abort_requested(abort_callback):
            return None
        _report_progress(progress_callback, _coz_phase("VAE Decode", step_label))
        with checkpoint_modules(module for module in self.vae.modules() if not module._modules):
            output = self.vae.decode(z_full / self.vae.config.scaling_factor, return_dict=False)[0].clamp(-1, 1)
        del z_full
        output = output[0].float().add_(1.0).mul_(127.5).round_().clamp_(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        return Image.fromarray(output)

    @torch.inference_mode()
    def upscale(self, sample: torch.Tensor, scale: float, *, seed: int = 0, abort_callback=None, progress_callback=None) -> torch.Tensor | None:
        if self.device.type != "cuda":
            raise RuntimeError("Chain-of-Zoom requires CUDA.")
        factors = COZ_STEP_FACTORS[float(scale)]
        frame = sample[:, 0]
        if frame.dtype == torch.uint8:
            frame_np = frame.permute(1, 2, 0).cpu().numpy()
        else:
            frame_np = frame.permute(1, 2, 0).float().add(1.0).mul(127.5).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        if frame_np.shape[2] == 1:
            frame_np = np.repeat(frame_np, 3, axis=2)
        image = Image.fromarray(frame_np)
        target_w = int(round(image.width * scale / 16) * 16)
        target_h = int(round(image.height * scale / 16) * 16)
        # constant zoom-out context across the chain: the original input resized to 512 min side
        context_scale = COZ_VLM_CONTEXT_SIZE / min(image.width, image.height)
        context_image = image.resize((max(1, int(image.width * context_scale)), max(1, int(image.height * context_scale))), Image.LANCZOS)

        print(f"[Chain-of-Zoom] x{scale:g} upsampling in {len(factors)} step(s): input={image.width}x{image.height}, output={target_w}x{target_h}")
        for step_no, factor in enumerate(factors):
            if step_no == len(factors) - 1:
                step_w, step_h = target_w, target_h
            else:
                step_w = int(round(image.width * factor / 16) * 16)
                step_h = int(round(image.height * factor / 16) * 16)
            image = self._sr_step(image, context_image, step_w, step_h, seed, f"Step {step_no + 1}/{len(factors)}", abort_callback=abort_callback, progress_callback=progress_callback)
            if image is None:
                return None
        output = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(1)
        return output


_RUNTIME = ChainOfZoomRuntime()


def load_models(paths: CoZPaths, *, init_pipe, profile, lm_features, vlm_vision_batch: int = COZ_VLM_VISION_BATCH, vlm_prompt_batch: int = COZ_VLM_PROMPT_BATCH, progress_callback=None) -> None:
    _report_progress(progress_callback, "CoZ Loading Models")
    _RUNTIME.load(paths, profile=profile, init_pipe=init_pipe, lm_features=lm_features, vlm_vision_batch=vlm_vision_batch, vlm_prompt_batch=vlm_prompt_batch)


def upscale_image(sample: torch.Tensor, scale: float, *, seed: int = 0, abort_callback=None, progress_callback=None) -> torch.Tensor | None:
    return _RUNTIME.upscale(sample, scale, seed=seed, abort_callback=abort_callback, progress_callback=progress_callback)


def release_models() -> None:
    _RUNTIME.release()
