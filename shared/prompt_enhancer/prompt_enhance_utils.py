# Prompt enhancement instructions adapted from LTX-Video (https://github.com/Lightricks/LTX-Video, ltx_video/utils/prompt_enhance_utils.py),
# Copyright (c) Lightricks Ltd., Apache License 2.0 (see docs/third_party_licenses/Apache-2.0.txt). Modified and extended for WanGP.
# Modifications Copyright (c) 2025-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import logging
import time
from typing import Union, List, Optional
from contextlib import nullcontext

import torch
from PIL import Image

from shared.llm_io import known_token_ids, llm_io_enabled, log_llm_io, media_descriptor
from shared.prompt_enhancer.templates import render_system_prompt, uses_placeholder
from shared.utils.cancellation import check_cancelled
from shared.prompt_enhancer.streaming import ThrottledStreamEmitter


def _check_generation_cancelled(*args, **kwargs):
    check_cancelled()
    return False

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

T2V_CINEMATIC_PROMPT = """You are an expert cinematic director with many award winning movies, When writing prompts based on the user input, focus on detailed, chronological descriptions of actions and scenes.
Include specific movements, appearances, camera angles, and environmental details - all in a single flowing paragraph.
Start directly with the action, and keep descriptions literal and precise.
Think like a cinematographer describing a shot list.
Do not change the user input intent, just enhance it.
Keep within 150 words.
For best results, build your prompts using this structure:
Start with main action in a single sentence
Add specific details about movements and gestures
Describe character/object appearances precisely
Include background and environment details
Specify camera angles and movements
Describe lighting and colors
Note any changes or sudden events
Do not exceed the 150 word limit!
Output the enhanced prompt only.
"""
T2I_VISUAL_PROMPT = """You are an expert visual artist and photographer with award-winning compositions. When writing prompts based on the user input, focus on detailed, precise descriptions of visual elements and composition.
Include specific poses, appearances, framing, and environmental details - all in a single flowing paragraph.
Start directly with the main subject, and keep descriptions literal and precise.
Think like a photographer describing the perfect shot.
Do not change the user input intent, just enhance it.
Keep within 150 words.
For best results, build your prompts using this structure:
Start with main subject and pose in a single sentence
Add specific details about expressions and positioning
Describe character/object appearances precisely
Include background and environment details
Specify framing, composition and perspective
Describe lighting, colors, and mood
Note any atmospheric or stylistic elements
Do not exceed the 150 word limit!
Output the enhanced prompt only.
"""

T2T_TEXT_PROMPT= """You are an expert speechwriter who crafts compelling, audience-appropriate speeches that effectively communicate the speaker's message while maintaining authenticity and impact.
Do not exceed the 150 word limit!
Output the enhanced prompt only.
"""

QWEN35_THINKING_SUPER_SYSTEM_PROMPT = """Use private reasoning to solve the task as well as possible.
Keep your thinking hidden and do not output any reasoning, chain-of-thought, or thinking process.
Always review your final answer to ensure it meets the user request.
"""

IT2V_CINEMATIC_PROMPT = """You are an expert cinematic director with many award winning movies.
You have the following information:
1. The user provides a general text input about its scenes expectations 
2. The user provides a caption of an image of a subject that relates to the scene
When writing prompts based on the user input, focus on detailed, chronological descriptions of actions and scenes.
Include specific movements, appearances, camera angles, and environmental details - all in a single flowing paragraph.
Start directly with the action, and keep descriptions literal and precise.
Think like a cinematographer describing a shot list.
Keep within 150 words.
For best results, build your prompts using this structure:
Describe the inital scene first using the image caption of the subject and then describe how the scene evolves by following the user text input. Image description should be in first priority! Align to the image caption if it contradicts the user text input.
Start with main action in a single sentence
Add specific details about movements and gestures
Describe character/object appearances precisely
Include background and environment details
Specify camera angles and movements
Describe lighting and colors
Note any changes or sudden events
Align to the image caption if it contradicts the user text input.
Do not exceed the 150 word limit!
Output the enhanced prompt only.
"""

I2V_CINEMATIC_PROMPT = """You are an expert cinematic director with many award winning movies.
You have been provided with a caption of an image of a subject that relates to the scene to film.
Focus on detailed, chronological descriptions of actions and scenes.
Include specific movements, appearances, camera angles, and environmental details - all in a single flowing paragraph.
Start directly with the action, and keep descriptions literal and precise.
Think like a cinematographer describing a shot list.
Keep within 150 words.
For best results, build your prompts using this structure:
Describe the inital scene first using the image caption of the subject and then describe how the scene should naturally evolves.
Start with main action in a single sentence
Add specific details about movements and gestures
Describe character/object appearances precisely
Include background and environment details
Specify camera angles and movements
Describe lighting and colors
Note any changes or sudden events
Do not exceed the 150 word limit!
Output the enhanced prompt only.
"""

IT2I_VISUAL_PROMPT = """You are an expert visual artist and photographer with award-winning compositions. When writing prompts based on the user input, focus on detailed, precise descriptions of visual elements and composition.
Include specific poses, appearances, framing, and environmental details - all in a single flowing paragraph.
You have the following information:
1. The user provides a general text input about the expected photography 
2. The user provides a caption of an image of a subject he wants to be represented in the photography
Start directly with the main subject, and keep descriptions literal and precise.
Think like a photographer describing the perfect shot.
Do not change the user input intent, just enhance it.
Keep within 150 words.
For best results, build your prompts using this structure:
Using the image caption start with main subject and pose in a single sentence
Add specific details about expressions and positioning
Describe character/object appearances precisely
Include background and environment details
Specify framing, composition and perspective
Describe lighting, colors, and mood
Note any atmospheric or stylistic elements
Do not exceed the 150 word limit!
Output the enhanced prompt only.
"""

I2I_VISUAL_PROMPT = """You are an expert visual artist and photographer with award-winning compositions. 
You have been provided with a caption of an image of a subject to be represented in the photography.
Focus on detailed, descriptions of actions that are happening in the photography.
Include specific poses, appearances, framing, and environmental details - all in a single flowing paragraph.
Start directly with the main subject, and keep descriptions literal and precise.
Think like a photographer describing the perfect shot.
Do not change the user input intent, just enhance it.
Keep within 150 words.
For best results, build your prompts using this structure:
Using the image caption start with main subject and pose in a single sentence
Add specific details about expressions and positioning
Describe character/object appearances precisely
Include background and environment details
Specify framing, composition and perspective
Describe lighting, colors, and mood
Note any atmospheric or stylistic elements
Do not exceed the 150 word limit!
Output the enhanced prompt only.
"""

def tensor_to_pil(tensor):
    # Ensure tensor is in range [-1, 1]
    assert tensor.min() >= -1 and tensor.max() <= 1

    # Convert from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2

    # Rearrange from [C, H, W] to [H, W, C]
    tensor = tensor.permute(1, 2, 0)

    # Convert to numpy array and then to uint8 range [0, 255]
    numpy_image = (tensor.cpu().numpy() * 255).astype("uint8")

    # Convert to PIL Image
    return Image.fromarray(numpy_image)


def _use_qwen35_thinking_prompt(prompt_enhancer_model, thinking_enabled: Optional[bool] = None) -> bool:
    if thinking_enabled is not None:
        return bool(thinking_enabled)
    return bool(getattr(prompt_enhancer_model, "_prompt_enhancer_enable_thinking", False))


def _split_prompt_enhancer_system_suffix(prompt_enhancer_model, prompt: str) -> tuple[str, str, bool]:
    del prompt_enhancer_model
    prompt = str(prompt or "").strip()
    prompt_body, separator, system_suffix = prompt.partition("@@")
    if separator == "@@":
        return prompt_body.strip(), system_suffix.strip(), True
    prompt_body, separator, system_suffix = prompt.partition("@")
    if separator == "":
        return prompt, "", False
    return prompt_body.strip(), system_suffix.strip(), False


def _merge_prompt_enhancer_system_prompt(prompt_enhancer_model, system_prompt: str, system_suffix: str, replace_system_prompt: bool = False, thinking_enabled: Optional[bool] = None, duration_instructions="") -> str:
    system_prompt = str(system_prompt or "")
    system_suffix = str(system_suffix or "").strip()
    effective_template = system_suffix if replace_system_prompt and system_suffix else system_prompt + "\n" + system_suffix
    if duration_instructions and not uses_placeholder(effective_template, "duration_seconds"):
        system_prompt += duration_instructions
    system_prompt = system_prompt.rstrip()
    if len(system_suffix) == 0:
        merged_prompt = system_prompt
    elif replace_system_prompt:
        merged_prompt = system_suffix
    else:
        merged_prompt = f"{system_prompt}\nFollow these additional user instructions with higher priority if they conflict with the guidance above:\n{system_suffix}"
    if not _use_qwen35_thinking_prompt(prompt_enhancer_model, thinking_enabled=thinking_enabled):
        return merged_prompt
    if len(merged_prompt) == 0:
        return QWEN35_THINKING_SUPER_SYSTEM_PROMPT.strip()
    return f"{QWEN35_THINKING_SUPER_SYSTEM_PROMPT.rstrip()}\n\n{merged_prompt}"


def _format_prompt_enhancer_user_content(prompt_enhancer_model, prompt: str, image_caption: Optional[str] = None, thinking_enabled: Optional[bool] = None) -> str:
    prompt, _system_suffix, _replace_system_prompt = _split_prompt_enhancer_system_suffix(prompt_enhancer_model, prompt)
    if not _use_qwen35_thinking_prompt(prompt_enhancer_model, thinking_enabled=thinking_enabled):
        if image_caption is None:
            return f"user_prompt: {prompt}"
        return f"user_prompt: {prompt}\nimage_caption: {image_caption}"
    if image_caption is None:
        return prompt
    image_caption = str(image_caption or "").strip()
    if len(prompt) == 0:
        return f"image_caption:\n{image_caption}"
    return f"{prompt}\n\nimage_caption:\n{image_caption}"


def generate_cinematic_prompt(
    image_caption_model,
    image_caption_processor,
    prompt_enhancer_model,
    prompt_enhancer_tokenizer,
    prompt: Union[str, List[str]],
    images: Optional[List] = None,
    video_prompt= True,
    text_prompt = False,
    max_new_tokens: int = 512,
    prompt_enhancer_instructions = None,
    do_sample: bool = True,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
    post_image_caption_hook = None,
    thinking_enabled: Optional[bool] = None,
    generation_callbacks=None,
    image_contexts=None,
    offload_manager=None,
    prompt_durations=None,
    template_context=None,
    inject_duration=True,
    duration_instructions="",
    vision=None,
) -> List[str]:
    """vision: a model's own vision stage for an enhancer mode (shared.prompt_enhancer.images.vision_settings), which replaces the image
    captions: its instructions, given each request, the answer's token budget and the reading resolution."""
    prompts = [prompt] if isinstance(prompt, str) else prompt
    check_cancelled()

    if images is None and image_contexts is None:
        if prompt_enhancer_instructions is None:
            prompt_enhancer_instructions = T2T_TEXT_PROMPT if text_prompt else (T2V_CINEMATIC_PROMPT if video_prompt else T2I_VISUAL_PROMPT)
        prompts = _generate_t2v_prompt(
            prompt_enhancer_model,
            prompt_enhancer_tokenizer,
            prompts,
            max_new_tokens,
            prompt_enhancer_instructions,
            do_sample,
            temperature,
            top_p,
            top_k,
            seed,
            thinking_enabled,
            generation_callbacks=generation_callbacks,
            prompt_durations=prompt_durations,
            template_context=template_context,
            inject_duration=inject_duration,
            duration_instructions=duration_instructions,
        )
    else:
        if prompt_enhancer_instructions is None:
            prompt_enhancer_instructions = IT2V_CINEMATIC_PROMPT if video_prompt else IT2I_VISUAL_PROMPT

        prompts = _generate_i2v_prompt(
            image_caption_model,
            image_caption_processor,
            prompt_enhancer_model,
            prompt_enhancer_tokenizer,
            prompts,
            images,
            max_new_tokens,
            prompt_enhancer_instructions,
            do_sample,
            temperature,
            top_p,
            top_k,
            seed,
            post_image_caption_hook=post_image_caption_hook,
            thinking_enabled=thinking_enabled,
            generation_callbacks=generation_callbacks,
            image_contexts=image_contexts,
            offload_manager=offload_manager,
            prompt_durations=prompt_durations,
            template_context=template_context,
            inject_duration=inject_duration,
            duration_instructions=duration_instructions,
            vision=vision,
        )

    return prompts


def _get_first_frames_from_conditioning_item(conditioning_item) -> List[Image.Image]:
    frames_tensor = conditioning_item.media_item
    return [
        tensor_to_pil(frames_tensor[i, :, 0, :, :])
        for i in range(frames_tensor.shape[0])
    ]


def _generate_t2v_prompt(
    prompt_enhancer_model,
    prompt_enhancer_tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    system_prompt: str,
    do_sample: bool,
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    seed: Optional[int],
    thinking_enabled: Optional[bool],
    generation_callbacks=None,
    prompt_durations=None,
    template_context=None,
    inject_duration=True,
    duration_instructions="",
) -> List[str]:
    messages = []
    for index, prompt in enumerate(prompts):
        prompt_body, system_suffix, replace_system_prompt = _split_prompt_enhancer_system_suffix(prompt_enhancer_model, prompt)
        message_system_prompt = _merge_prompt_enhancer_system_prompt(prompt_enhancer_model, system_prompt, system_suffix, replace_system_prompt,
            thinking_enabled=thinking_enabled, duration_instructions=duration_instructions if inject_duration else "")
        duration = prompt_durations[index] if prompt_durations is not None else None
        message_system_prompt = render_system_prompt(message_system_prompt, template_context, duration, inject_duration=inject_duration)
        messages.append(
            [
                {"role": "system", "content": message_system_prompt},
                {"role": "user", "content": _format_prompt_enhancer_user_content(prompt_enhancer_model, prompt_body, thinking_enabled=thinking_enabled)},
            ]
        )

    if hasattr(prompt_enhancer_model, "generate_messages"):
        if llm_io_enabled():
            log_llm_io("OUT", "local-prompt-enhancer", "messages", {"messages": messages, "generation": {"max_new_tokens": max_new_tokens, "do_sample": do_sample, "temperature": temperature, "top_p": top_p, "top_k": top_k, "seed": seed, "thinking_enabled": thinking_enabled}})
        outputs = prompt_enhancer_model.generate_messages(
            messages,
            max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            thinking_enabled=thinking_enabled,
            **(generation_callbacks or {}),
        )
        log_llm_io("IN", "local-prompt-enhancer", "messages", {"text": outputs})
        return outputs

    texts = [
        prompt_enhancer_tokenizer.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True
        )
        for m in messages
    ]

    out_prompts = []
    for idx, text in enumerate(texts):
        if (generation_callbacks or {}).get("enhancement_progress") is not None:
            generation_callbacks["enhancement_progress"].prompt(idx, len(texts), max_new_tokens)
        model_inputs = prompt_enhancer_tokenizer(text, return_tensors="pt").to(
            prompt_enhancer_model.device
        )
        prompt_seed = None if seed is None else int(seed) + idx
        out_prompts.append(
            _generate_and_decode_prompts(
                prompt_enhancer_model,
                prompt_enhancer_tokenizer,
                model_inputs,
                max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=prompt_seed,
                generation_callbacks=generation_callbacks,
            )[0]
        )

    return out_prompts

def _generate_i2v_prompt(
    image_caption_model,
    image_caption_processor,
    prompt_enhancer_model,
    prompt_enhancer_tokenizer,
    prompts: List[str],
    first_frames: List[Image.Image],
    max_new_tokens: int,
    system_prompt: str,
    do_sample: bool,
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    seed: Optional[int],
    post_image_caption_hook = None,
    thinking_enabled: Optional[bool] = None,
    generation_callbacks=None,
    image_contexts=None,
    offload_manager=None,
    prompt_durations=None,
    template_context=None,
    inject_duration=True,
    duration_instructions="",
    vision=None,
) -> List[str]:
    if vision is not None and hasattr(image_caption_model, "generate_image_captions"):
        # the model's vision stage reads the images for each request (its answer takes the caption's place)
        image_captions = ["\n\n".join(image_caption_model.generate_image_captions(first_frames or [], image_contexts=None if image_contexts is None else [image_contexts[index]], offload_manager=offload_manager,
                          stop_requested=(generation_callbacks or {}).get("stop_requested"), enhancement_progress=(generation_callbacks or {}).get("enhancement_progress"), vision=dict(vision, request=prompt)))
                          for index, prompt in enumerate(prompts)]
    elif hasattr(image_caption_model, "generate_image_captions"):
        image_captions = image_caption_model.generate_image_captions(first_frames or [], image_contexts=image_contexts, offload_manager=offload_manager, stop_requested=(generation_callbacks or {}).get("stop_requested"), enhancement_progress=(generation_callbacks or {}).get("enhancement_progress"))
    else:
        image_captions = _generate_image_captions(
            image_caption_model, image_caption_processor, first_frames, enhancement_progress=(generation_callbacks or {}).get("enhancement_progress")
        )
    if callable(post_image_caption_hook) and not hasattr(image_caption_model, "generate_image_captions"):
        if bool(getattr(prompt_enhancer_model, "_prompt_enhancer_use_vllm", False)):
            unload_runtime = getattr(prompt_enhancer_model, "unload", None)
            if callable(unload_runtime):
                unload_runtime()
        post_image_caption_hook()
    if len(image_captions) == 1 and len(image_captions) < len(prompts):
        image_captions *= len(prompts)
    messages = []
    for index, (prompt, image_caption) in enumerate(zip(prompts, image_captions)):
        prompt_body, system_suffix, replace_system_prompt = _split_prompt_enhancer_system_suffix(prompt_enhancer_model, prompt)
        message_system_prompt = _merge_prompt_enhancer_system_prompt(prompt_enhancer_model, system_prompt, system_suffix, replace_system_prompt,
            thinking_enabled=thinking_enabled, duration_instructions=duration_instructions if inject_duration else "")
        if image_contexts is not None:
            message_system_prompt += "\n\nThe labeled image descriptions belong only to this generation window. Respect the start and end anchors and the times of injected frames. Reference images guide appearance, not chronological order. Keep these roles distinct and do not invent absent anchors."
        duration = prompt_durations[index] if prompt_durations is not None else image_contexts[index].duration_seconds if image_contexts is not None else None
        message_system_prompt = render_system_prompt(message_system_prompt, template_context, duration, inject_duration=inject_duration)
        messages.append(
            [
                {"role": "system", "content": message_system_prompt},
                {"role": "user", "content": _format_prompt_enhancer_user_content(prompt_enhancer_model, prompt_body, image_caption=image_caption, thinking_enabled=thinking_enabled)},
            ]
        )

    if hasattr(prompt_enhancer_model, "generate_messages"):
        if llm_io_enabled():
            log_llm_io("OUT", "local-prompt-enhancer", "messages", {"messages": messages, "images": [media_descriptor(image) for image in (first_frames or [])], "image_captions": image_captions, "generation": {"max_new_tokens": max_new_tokens, "do_sample": do_sample, "temperature": temperature, "top_p": top_p, "top_k": top_k, "seed": seed, "thinking_enabled": thinking_enabled}})
        outputs = prompt_enhancer_model.generate_messages(
            messages,
            max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            thinking_enabled=thinking_enabled,
            **(generation_callbacks or {}),
        )
        log_llm_io("IN", "local-prompt-enhancer", "messages", {"text": outputs})
        return outputs

    texts = [
        prompt_enhancer_tokenizer.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True
        )
        for m in messages
    ]
    out_prompts = []
    for idx, text in enumerate(texts):
        if (generation_callbacks or {}).get("enhancement_progress") is not None:
            generation_callbacks["enhancement_progress"].prompt(idx, len(texts), max_new_tokens)
        model_inputs = prompt_enhancer_tokenizer(text, return_tensors="pt").to(
            prompt_enhancer_model.device
        )
        prompt_seed = None if seed is None else int(seed) + idx
        out_prompts.append(
            _generate_and_decode_prompts(
                prompt_enhancer_model,
                prompt_enhancer_tokenizer,
                model_inputs,
                max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=prompt_seed,
                generation_callbacks=generation_callbacks,
            )[0]
        )

    return out_prompts


def _generate_image_captions(
    image_caption_model,
    image_caption_processor,
    images: List[Image.Image],
    system_prompt: str = "<DETAILED_CAPTION>",
    enhancement_progress=None,
) -> List[str]:
    if enhancement_progress is not None:
        enhancement_progress.caption(0, 1, 1024)
    image_caption_prompts = [system_prompt] * len(images)
    inputs = image_caption_processor(
        image_caption_prompts, images, return_tensors="pt"
    ).to(image_caption_model.device)

    bad_words_ids = None
    bos_id = getattr(image_caption_processor.tokenizer, "bos_token_id", None)
    if bos_id is not None:
        bad_words_ids = [[int(bos_id)]]

    with torch.inference_mode():
        if llm_io_enabled():
            log_llm_io("OUT", "local-image-captioner", "generation", {
                "prompts": image_caption_prompts,
                "images": [media_descriptor(image) for image in images],
                "input_token_ids": inputs["input_ids"].tolist(),
                "known_token_ids": known_token_ids(image_caption_processor.tokenizer),
                "generation": {"max_new_tokens": 1024, "do_sample": False, "num_beams": 3, "bad_words_ids": bad_words_ids},
            })
        generated_ids = image_caption_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
            bad_words_ids=bad_words_ids,
            stopping_criteria=[_check_generation_cancelled],
        )
    decoded = image_caption_processor.batch_decode(generated_ids, skip_special_tokens=True)
    log_llm_io("IN", "local-image-captioner", "generation", {"text": decoded, "output_token_ids": generated_ids.tolist()})
    return decoded


def _generate_and_decode_prompts(
    prompt_enhancer_model,
    prompt_enhancer_tokenizer,
    model_inputs,
    max_new_tokens: int,
    do_sample: bool = True,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
    generation_callbacks=None,
) -> List[str]:
    callbacks = generation_callbacks or {}
    stop_requested, stream_callback = callbacks.get("stop_requested"), callbacks.get("stream_callback")
    if callbacks.get("enhancement_progress") is not None:
        stream_callback = callbacks["enhancement_progress"].tokens
    emitter = ThrottledStreamEmitter(1 / 3)
    started = time.perf_counter()
    prefill_seconds = None
    prompt_length = model_inputs.input_ids.shape[-1]

    def report_tokens(input_ids, scores=None, *, is_final=False):
        nonlocal prefill_seconds
        if callable(stop_requested) and stop_requested():
            raise InterruptedError("Prompt Enhancement Stopped")
        check_cancelled()
        count = input_ids.shape[-1] - prompt_length
        elapsed = time.perf_counter() - started
        if count and prefill_seconds is None:
            prefill_seconds = elapsed
        if stream_callback is not None and (is_final or emitter.is_due()):
            speed = max(0, count - 1) / max(elapsed - (prefill_seconds or elapsed), 1e-6)
            emitter.emit(stream_callback, raw_text=prompt_enhancer_tokenizer.decode(input_ids[0, prompt_length:], skip_special_tokens=True), token_count=count, max_tokens=max_new_tokens, prefill_seconds=prefill_seconds or 0.0, tokens_per_second=speed, stop_reason="completed" if is_final else None, is_final=is_final, force=is_final)
        return False

    device = "cuda"
    if seed is None:
        rng_context = nullcontext()
    else:
        devices = []
        if isinstance(device, torch.device) and device.type == "cuda":
            devices = [device.index or 0]
        rng_context = torch.random.fork_rng(devices=devices) if devices else torch.random.fork_rng()
    with rng_context, torch.inference_mode():
        if seed is not None:
            torch.manual_seed(int(seed))
            if isinstance(device, torch.device) and device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(int(seed))
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if temperature is not None:
            gen_kwargs["temperature"] = float(temperature)
        if top_p is not None:
            gen_kwargs["top_p"] = float(top_p)
        if top_k is not None:
            gen_kwargs["top_k"] = int(top_k)
        if llm_io_enabled():
            log_llm_io("OUT", "local-prompt-enhancer", "generation", {
                "input_token_ids": model_inputs.input_ids.tolist(),
                "known_token_ids": known_token_ids(prompt_enhancer_tokenizer),
                "generation": {**gen_kwargs, "seed": seed},
            })
        report_tokens(model_inputs.input_ids)
        outputs = prompt_enhancer_model.generate(
            **model_inputs,
            **gen_kwargs,
            stopping_criteria=[report_tokens],
        )
        report_tokens(outputs, is_final=True)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, outputs)
        ]
        decoded_prompts = prompt_enhancer_tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )
    log_llm_io("IN", "local-prompt-enhancer", "generation", {"text": decoded_prompts, "output_token_ids": [token_ids.tolist() for token_ids in generated_ids]})
    return decoded_prompts
