# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import math
from contextlib import contextmanager, nullcontext
from typing import Any

import torch
from PIL import Image

from shared.deepy.image_channels import upright
from shared.llm_engines.lmgp.models.qwen3_5 import Qwen3_5ForCausalLM
from shared.llm_engines.lmgp.models.qwen4_exp import Qwen4ExpForCausalLM
from shared.prompt_enhancer.qwen35_vl import TEXT_MODEL_ID as _TEXT_MODEL_ID, VISION_MODEL_ID as _VISION_MODEL_ID, _prepare_multimodal_vllm_prompt, vision_beside_text
from shared.utils.video_decode import decode_video_frame_indices_ffmpeg


VISION_MAX_IMAGES = 8
VISION_REMOTE_MAX_IMAGES = 10
VISION_MAX_VISUAL_TOKENS_PER_IMAGE = 1024
VISION_ANSWER_MAX_NEW_TOKENS = 1024
VISION_LOCAL_ANSWER_MAX_NEW_TOKENS = 4096
VISION_MIN_MODEL_LEN = 16384
VISION_REMOTE_MAX_IMAGE_EDGE = 1024
VISION_VIDEO_MAX_IMAGES = 128
VISION_VIDEO_REMOTE_MAX_IMAGES = 160
VISION_VIDEO_MAX_IMAGE_EDGE = 256
VISION_VIDEO_MID_RES_MAX_IMAGE_EDGE = 512
VISION_VIDEO_MID_RES_SAMPLE_DIVISOR = 4
VISION_VIDEO_MAX_SAMPLES_PER_SECOND = 2
VISION_VIDEO_SHORT_RANGE_SECONDS, VISION_VIDEO_SHORT_RANGE_SAMPLES_PER_SECOND = 3.0, 8  # joins and cuts need dense frames
VISION_VIDEO_DECODE_BATCH_SIZE = 8
VISION_QA_SYSTEM_PROMPT = "Answer the user's question about the labeled visual inputs accurately and concisely. Inputs may be images or ordered frames from one or more videos. If the answer is uncertain, say so."


def can_keep_text_resident(runtime, manager) -> bool:
    if runtime is None or manager is None or not isinstance(runtime.model, (Qwen3_5ForCausalLM, Qwen4ExpForCausalLM)):
        return False
    if manager.models.get(_TEXT_MODEL_ID) is not runtime.model or _TEXT_MODEL_ID not in manager.active_models_ids:
        return False
    from mmgp import runtime as mmgp_runtime
    if mmgp_runtime.dynamic_preload(runtime.model):
        # its dynamic preload releases only the blocks the vision tower needs beyond the lent cache, and keeps them again afterwards
        return runtime._get_live_llm() is not None
    prefix = _TEXT_MODEL_ID + "/"
    preloaded = manager.preloaded_blocks_per_model[_TEXT_MODEL_ID]
    if not all(name[len(prefix):] in preloaded for name in manager.blocks_of_modules if name.startswith(prefix)) or not all(parameter.is_cuda for parameter in runtime.model.parameters()):
        return False
    runner = runtime._get_live_llm().model_runner
    cache_bytes = runner.kv_cache.nbytes
    if hasattr(runner, "kv_cache_scales"):
        cache_bytes += runner.kv_cache_scales.nbytes
    vision_prefix = _VISION_MODEL_ID + "/"
    vision_bytes = sum(size for name, size in manager.blocks_of_modules_sizes.items() if name == _VISION_MODEL_ID or name.startswith(vision_prefix))
    # Small Q8 caches (e.g. 9B at 32K) cannot even cover the tower's weights.
    # Keep the established unload path instead of knowingly increasing its footprint.
    return cache_bytes >= vision_bytes


def engine_idle(runtime):
    """While the vision tower encodes, the assistant's engine is not used: MMGP may release its KV cache if VRAM runs short (Deepy saves its
    context first, restored at its next segment)."""
    engine = None if runtime is None else getattr(runtime.model, "_prompt_enhancer_vllm_engine", None)
    return nullcontext() if engine is None else engine.idle()


@contextmanager
def resident_inspection(runtime, caption_model, manager):
    """The vision tower beside the assistant's language model, whose weights and context stay. With the dynamic preload (MMGP VRAM
    allocator) the tower takes only the VRAM it needs: the idle model's blocks, then its KV cache (engine_idle), and the answer runs beside the
    live context when the free pages of the cache hold it (Qwen35AssistantRuntime.generate_embedded_answer). Without it, the cache's memory
    is lent to the tower: the context goes to RAM and comes back afterwards."""
    from mmgp import runtime as mmgp_runtime
    from shared.prompt_enhancer.qwen35_assistant_runtime import _ASSISTANT_PREFILL_CHUNK_TOKENS

    model = runtime.model
    lend = not mmgp_runtime.dynamic_preload(model)
    snapshot = runtime.snapshot_context() if lend else None
    min_model_len = model._prompt_enhancer_min_model_len_hint
    prefill_chunk_tokens = model.__dict__.get("_prefill_chunk_tokens")
    try:
        if lend:
            # Teardown synchronizes in-flight graphs and releases their KV/state pointers.
            # The tower has no decoder cache; allocate the smaller QA cache after encoding.
            model._prompt_enhancer_vllm_engine.close()
            model._prompt_enhancer_min_model_len_hint = VISION_MIN_MODEL_LEN
        model._prefill_chunk_tokens = _ASSISTANT_PREFILL_CHUNK_TOKENS
        with vision_beside_text(caption_model, manager) as unload_vision:
            runtime._log("Inspection: keeping Qwen weights resident; " + ("lending assistant cache memory to vision (and Qwen blocks if it needs more)." if lend else "the vision tower takes Qwen blocks, then its cache, only if VRAM runs short."))
            yield unload_vision
    finally:
        if lend:
            model._prompt_enhancer_vllm_engine.close()
        model._prompt_enhancer_min_model_len_hint = min_model_len
        if prefill_chunk_tokens is None:
            del model._prefill_chunk_tokens
        else:
            model._prefill_chunk_tokens = prefill_chunk_tokens
        if snapshot is not None:
            runtime.restore_snapshot(snapshot)


def normalize_inspection_bbox(bbox: Any) -> list[int] | None:
    if bbox is None:
        return None
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("bbox must be [x_min, y_min, x_max, y_max].")
    try:
        values = [int(value) for value in bbox]
        if any(isinstance(value, bool) or float(value) != parsed for value, parsed in zip(bbox, values)):
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("bbox values must be integers from 0 to 1000.") from exc
    x_min, y_min, x_max, y_max = values
    if not all(0 <= value <= 1000 for value in values):
        raise ValueError("bbox values must be integers from 0 to 1000.")
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("bbox=[x_min,y_min,x_max,y_max] uses integers 0..1000 relative to the full image; x_max must be greater than x_min and y_max greater than y_min.")
    return values


def prepare_inspection_image(image: Any, max_edge: int | None = None, max_pixels: int | None = None, bbox: list[int] | None = None) -> Image.Image:
    prepared = upright(image).convert("RGB")
    if bbox is not None:
        x_min, y_min, x_max, y_max = bbox
        width, height = prepared.size
        left = min(width - 1, math.floor(x_min * width / 1000))
        top = min(height - 1, math.floor(y_min * height / 1000))
        right = max(left + 1, min(width, math.ceil(x_max * width / 1000)))
        bottom = max(top + 1, min(height, math.ceil(y_max * height / 1000)))
        prepared = prepared.crop((left, top, right, bottom))
    if max_pixels is not None and prepared.width * prepared.height > int(max_pixels):
        scale = math.sqrt(int(max_pixels) / (prepared.width * prepared.height))
        prepared = prepared.resize((max(1, math.floor(prepared.width * scale)), max(1, math.floor(prepared.height * scale))), Image.Resampling.LANCZOS)
    elif max_edge is not None:
        prepared.thumbnail((int(max_edge), int(max_edge)), Image.Resampling.LANCZOS)
    return prepared


def resize_inspection_image(image: Any, max_edge: int) -> Image.Image:
    return prepare_inspection_image(image, max_edge=max_edge)


def decode_inspection_video_frames(path: str, frame_indices: list[int], max_edge: int | None = None, max_pixels: int | None = None, bboxes: list[list[int] | None] | None = None) -> list[Image.Image]:
    if bboxes is not None and len(bboxes) != len(frame_indices):
        raise ValueError("Video frame bboxes must match the frame count.")
    images = []
    for offset in range(0, len(frame_indices), VISION_VIDEO_DECODE_BATCH_SIZE):
        current_indices = frame_indices[offset:offset + VISION_VIDEO_DECODE_BATCH_SIZE]
        frames = decode_video_frame_indices_ffmpeg(path, current_indices, bridge="numpy")
        if len(frames) != len(current_indices):
            raise RuntimeError(f"Video decoder returned {len(frames)} of {len(current_indices)} requested frames.")
        for frame_index, frame in enumerate(frames):
            bbox = None if bboxes is None else bboxes[offset + frame_index]
            images.append(prepare_inspection_image(Image.fromarray(frame), max_edge=max_edge, max_pixels=max_pixels, bbox=bbox))
    return images


def video_inspection_sample_count(*, remote: bool, mid_res_sampling: bool) -> int:
    base_count = VISION_VIDEO_REMOTE_MAX_IMAGES if remote else VISION_VIDEO_MAX_IMAGES
    return base_count // VISION_VIDEO_MID_RES_SAMPLE_DIVISOR if mid_res_sampling else base_count


def _inspection_image_size(processor: Any, max_pixels_per_image: int | None = None) -> tuple[dict[str, int], int, int]:
    image_processor = processor.image_processor
    merge_size = int(image_processor.merge_size)
    token_edge = int(image_processor.patch_size) * merge_size
    token_budget_pixels = VISION_MAX_VISUAL_TOKENS_PER_IMAGE * token_edge * token_edge
    max_pixels = token_budget_pixels if max_pixels_per_image is None else min(token_budget_pixels, int(max_pixels_per_image))
    min_pixels = min(int(image_processor.size.get("shortest_edge", max_pixels)), max_pixels) if max_pixels_per_image is None else min(token_edge * token_edge, max_pixels)
    return {"shortest_edge": min_pixels, "longest_edge": max_pixels}, merge_size, min(VISION_MAX_VISUAL_TOKENS_PER_IMAGE, math.ceil(max_pixels / (token_edge * token_edge)))


def build_image_question_prompt(caption_model: Any, processor: Any, image: Any, question: str, system_prompt: str | None = None, image_labels: list[str] | None = None, *, max_images: int = VISION_MAX_IMAGES, max_pixels_per_image: int | None = None, resident: bool = False):
    question = str(question or "").strip()
    if len(question) == 0:
        raise ValueError("Vision question is empty.")
    images = list(image) if isinstance(image, (list, tuple)) else [image]
    if not 1 <= len(images) <= int(max_images):
        raise ValueError(f"Vision inspection requires between 1 and {int(max_images)} images.")
    if image_labels is not None and len(image_labels) != len(images):
        raise ValueError("Vision input labels must match the image count.")
    messages = []
    system_prompt = str(system_prompt or VISION_QA_SYSTEM_PROMPT).strip()
    if len(system_prompt) > 0:
        messages.append({"role": "system", "content": system_prompt})
    content = []
    for index, current_image in enumerate(images):
        if image_labels is not None:
            content.append({"type": "text", "text": str(image_labels[index]).strip()})
        content.append({"type": "image", "image": current_image})
    content.append({"type": "text", "text": question})
    messages.append({"role": "user", "content": content})
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    image_size, merge_size, max_visual_tokens = _inspection_image_size(processor, max_pixels_per_image=max_pixels_per_image)
    model_inputs = processor(
        text=[text],
        images=images,
        return_tensors="pt",
        padding=True,
        return_mm_token_type_ids=True,
        images_kwargs={"size": image_size},
    )
    image_grid_thw = model_inputs.get("image_grid_thw")
    image_grids = image_grid_thw.tolist() if hasattr(image_grid_thw, "tolist") else image_grid_thw
    if image_grids is None or len(image_grids) != len(images):
        raise RuntimeError("Vision processor returned an unexpected image grid count.")
    if any(int(grid[0]) * int(grid[1]) * int(grid[2]) // (merge_size * merge_size) > max_visual_tokens for grid in image_grids):
        raise RuntimeError("Vision processor exceeded the per-image visual token limit.")
    image_features = None
    if resident:
        # Each image is independent in the tower. Bound its activation workspace by
        # the existing single-image token budget, including video inspection frames.
        pixel_values = model_inputs.pop("pixel_values")
        image_features = []
        patch_limit = VISION_MAX_VISUAL_TOKENS_PER_IMAGE * merge_size * merge_size
        first_image = first_patch = batch_patches = 0
        with torch.inference_mode():
            for index, grid in enumerate(image_grids):
                patch_count = math.prod(grid)
                if batch_patches and batch_patches + patch_count > patch_limit:
                    output = caption_model.model.get_image_features(pixel_values[first_patch:first_patch + batch_patches], image_grid_thw[first_image:index], return_dict=True)
                    image_features.extend(output.pooler_output)
                    del output
                    first_image, first_patch, batch_patches = index, first_patch + batch_patches, 0
                batch_patches += patch_count
            output = caption_model.model.get_image_features(pixel_values[first_patch:first_patch + batch_patches], image_grid_thw[first_image:], return_dict=True)
            image_features.extend(output.pooler_output)
            del output
    return _prepare_multimodal_vllm_prompt(caption_model, model_inputs, image_features=image_features)


__all__ = [
    "VISION_ANSWER_MAX_NEW_TOKENS", "VISION_MAX_IMAGES", "VISION_MAX_VISUAL_TOKENS_PER_IMAGE", "VISION_QA_SYSTEM_PROMPT",
    "VISION_REMOTE_MAX_IMAGES", "VISION_REMOTE_MAX_IMAGE_EDGE", "VISION_VIDEO_MAX_IMAGE_EDGE", "VISION_VIDEO_MAX_IMAGES",
    "VISION_VIDEO_MAX_SAMPLES_PER_SECOND", "VISION_VIDEO_SHORT_RANGE_SAMPLES_PER_SECOND", "VISION_VIDEO_SHORT_RANGE_SECONDS", "VISION_VIDEO_MID_RES_MAX_IMAGE_EDGE", "VISION_VIDEO_MID_RES_SAMPLE_DIVISOR", "VISION_VIDEO_REMOTE_MAX_IMAGES", "build_image_question_prompt",
    "decode_inspection_video_frames", "normalize_inspection_bbox", "prepare_inspection_image", "resize_inspection_image", "video_inspection_sample_count",
]
