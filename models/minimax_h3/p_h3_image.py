# Copyright 2026 The HuggingFace Team. All rights reserved.
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
#
# WanGP adaptation of the P-H3-Image image sizing (diffusers).
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""P-H3-Image by Pruna AI: still images and edits on MiniMax H3's FL2VA transformer, a Qwen3-VL-4B text encoder and a single-frame VAE decoder."""

import math

import torch

from shared.utils.hf import build_hf_url


P_H3_IMAGE_ARCHITECTURE = "minimax_p_h3_image"
P_H3_IMAGE_REPO_ID = "DeepBeepMeep/MiniMax-H3"
P_H3_IMAGE_TEXT_ENCODER_FOLDER = "P-H3-Image-Qwen3-VL-4B"
P_H3_IMAGE_TEXT_ENCODER_BF16 = "P-H3-Image-Qwen3-VL-4B_bf16.safetensors"
P_H3_IMAGE_TEXT_ENCODER_INT8 = "P-H3-Image-Qwen3-VL-4B_int8_convrot.safetensors"
P_H3_IMAGE_FOLDER = "p_h3_image"
P_H3_IMAGE_VAE_FILE = f"{P_H3_IMAGE_FOLDER}/p_h3_image_vae_fp16.safetensors"
P_H3_IMAGE_VAE_INT8_FILE = f"{P_H3_IMAGE_FOLDER}/p_h3_image_vae_int8_convrot.safetensors"
P_H3_IMAGE_SIGMAS = (1.0, 0.984753, 0.967905, 0.951688, 0.925372, 0.881923, 0.810083, 0.528718)
P_H3_IMAGE_STEPS = len(P_H3_IMAGE_SIGMAS)
P_H3_IMAGE_SHIFT = 12.0
P_H3_IMAGE_SWITCH_THRESHOLD = 0.7  # the detail-pass start of the pDMD 2-step recipe
P_H3_IMAGE_MAX_IMAGES = 8
P_H3_IMAGE_REFERENCE_MIN_PIXELS = 1024**2
P_H3_IMAGE_REFERENCE_MAX_PIXELS = 8 * 1024**2
P_H3_IMAGE_VISION_PIXELS = 1_000_000


def size_at_area(width, height, pixels=P_H3_IMAGE_VISION_PIXELS, multiple=32):
    """(width, height) at about ``pixels``, its aspect ratio kept and both sides multiples of ``multiple``; sides already aligned within 20 % of the area keep their size."""
    if width % multiple == 0 and height % multiple == 0 and 0.8 * pixels <= width * height <= 1.2 * pixels:
        return width, height
    ratio = width / height
    return max(multiple, round(math.sqrt(pixels * ratio) / multiple) * multiple), max(multiple, round(math.sqrt(pixels / ratio) / multiple) * multiple)


def p_h3_image_sigmas(steps):
    """The noise levels of ``steps`` denoising steps, terminal 0 left out: the checkpoint's own grid at 8 steps, otherwise H3's
    ``linspace(1, 0, steps + 1)`` grid at shift 12, the grid few-step LoRAs such as pDMD are trained on."""
    if steps == P_H3_IMAGE_STEPS:
        return P_H3_IMAGE_SIGMAS
    base = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32, device="cpu")[:-1]
    return tuple((P_H3_IMAGE_SHIFT * base / (1 + (P_H3_IMAGE_SHIFT - 1) * base)).tolist())


def reference_size(width, height, multiple=32):
    """An image keeps its size unless its area is outside the accepted range, both sides rounded to multiples of ``multiple``."""
    area = width * height
    scale = math.sqrt(min(max(area, P_H3_IMAGE_REFERENCE_MIN_PIXELS), P_H3_IMAGE_REFERENCE_MAX_PIXELS) / area)
    return tuple(max(multiple, round(side * scale / multiple) * multiple) for side in (width, height))


P_H3_IMAGE_INFOS = """## P-H3-Image

Generates a single high-resolution image from text, or edits an image with up to 8 input images. It does not generate video or audio. It runs 8 steps by default, without guidance or a negative prompt.

### Text to Image

Describe the finished image. **8 MP sizes work best**: faces, hands and fine details break more often at 4 MP and below. WanGP lists the recommended 8 MP and 4 MP sizes first. Sides must be multiples of 32.

### Editing

Choose an option under **Images to Edit**, then add 1 to 8 images. The first image is the one edited; the others are references, named `<Picture 2>`, `<Picture 3>` and so on.

- **Edit the First Image at its Own Proportions:** the first image defines the output proportions, within the selected resolution. Choose this for most edits.
- **Edit with the Selected Output Size:** the canvas keeps the selected resolution, which can differ from the first image's proportions.

Edits work best at 4 or 8 MP. Each image is used at its own size between 1 and 8 MP, so several large images use much more memory and time: resize large references when memory is short.

### Prompts

This model is sensitive to prompting and renders full, specific prompts much better than short requests. The optional **Write** button turns a short request into a full prompt or edit instruction. See the prompt help for how to write them by hand.

### Fewer Steps

The **pDMD** accelerator profiles add a LoRA that finishes the image in fewer steps:

- **pDMD 2 Steps:** the fastest, about 2.5 times faster than 8 steps for a whole image. Sharp and coherent, but graphic prompts can invent pseudo-text.
- **pDMD 4 Steps:** about 1.6 times faster, with more detail than 2 steps. Choose it for quoted text and flat graphics.

3 steps is a middle ground. Both profiles also work for edits. For the same seed they give a different composition than 8 steps. Without an accelerator, fewer than 8 steps leaves the image blurred and unfinished. At 8 steps the model uses its own tuned schedule; other step counts use the standard H3 schedule the pDMD LoRA is trained on. **P-H3-Image 8 Steps (Reset)** goes back to the default settings and keeps your other LoRAs.

### Two Phases (Detail Pass)

With 2 phases, phase 1 composes the image at half the width and height, WanGP enlarges it twice, and phase 2 redraws the details at full size from **Phase 2 Noise Level Start**. Two presets set this up and keep your other LoRAs:

- **pDMD 2 Steps + Regenerate 2K Detail:** a 2-step draft, then 3 detail steps with the Open-H3 Regenerate 2K LoRA.
- **Regenerate 2K Detail (8 + 3 Steps):** the normal 8-step draft, then the same detail pass.

Both start phase 2 at 0.7: composition and style hold, people can change. A lower start keeps faces closer: enter it in **Custom Sigmas** (see below). Without an accelerator that has a phase-2 multiplier, phase 2 uses the FL2VA LoRA Accelerator (Lightx2v 4 Steps).

**Custom Sigmas** sets the noise levels of each phase, phases separated by `;`. A blank phase keeps its default schedule. The detail presets use `; 0.7, 0.659794, 0.513173`; for `0.5`, scale all three values by 0.5 / 0.7.

### LoRAs

This model shares the MiniMax H3 LoRA list. LoRAs made for FL2VA apply to it; Ref2VA LoRAs are listed too, but this model is built on FL2VA. For the same reason Open-H3 Regenerate 2K IC, a Ref2VA LoRA, is refused here: use the Regenerate 2K presets. Accelerators made for H3 video have not been checked on this model.

### Several Images

**Number of Images** generates several variations of the same prompt together, faster than one after the other; the first keeps the composition a single generation with the same seed gives, with slightly different fine details. It works with one phase: the two-phase Regenerate presets make one image at a time.

### Limits

Hands and small people in the background can break, and rendering at 8 MP helps. Short quoted text renders well; small or dense text turns into pseudo-text, and longer passages can get single letters wrong.
"""

P_H3_IMAGE_PROMPT_INFOS = """### Text to Image

1. **Say what the image is.** Open with one sentence naming the real object the image is: `A photograph by an experienced street photographer`, `A frame from a Japanese anime TV series`, `A two-color risograph print`. This sets the look more reliably than style words.
2. **Describe one scene.** A viewpoint, one main subject caught in a moment, a few supporting elements, and light with a source and a direction.
3. **Spell out every word.** Put each piece of text in quotes, keep it short, say where it sits and end with `no other text`. If you want no text, do not mention signs or labels at all.
4. **Leave out hype words.** Stunning, epic, cinematic, vibrant, hyper-detailed, 8k or masterpiece push the image toward a glossy, airbrushed look.

Very long prompts also make images worse.

Styles that work especially well: documentary and editorial photography, portraits with real skin texture, film frames, TV-frame anime, ink and charcoal drawings, screen-printed posters and flat graphic design with short quoted text.

`A photograph by an experienced street photographer of an old fishmonger in a Lisbon market, seen from counter height as she wraps a sardine in paper. Crates of ice behind her, a hanging bulb lighting her hands from the left. A hand-painted sign above her reads "PEIXE FRESCO", no other text.`

### Editing

`Edit <Picture 1>. <The change, described as the finished result>. Keep <everything that must stay> exactly as it is.`

- **Point to the exact thing:** `the old man's faded blue cap`, not `his hat`.
- **Describe the result, not the action,** and give new things a size next to something already there: `about the size of the cat's head`.
- **Say what stays:** faces, poses, the background, the light. Without this list the model tends to redraw the whole image.
- **Quote new text exactly** and name the text around it that must stay.

`Edit <Picture 1>. The pickup truck is now glossy red. Keep the driver, the road, the trees and the afternoon light exactly as they are.`

With several images, name them: `Edit <Picture 1>. The woman wears the straw hat from <Picture 2>. Keep her face, pose, clothes and the background exactly as they are.`

WanGP adds `Edit <Picture 1>.` when an edit prompt does not start with it.
"""

P_H3_IMAGE_DEEPY_INFOS = """Single images from `prompt`, or edits with 1 to 8 ordered `image_refs`; no video or audio. Euler, no guidance or negative prompt. `num_inference_steps` defaults to 8 (the checkpoint's own schedule); other counts use H3's shift-12 schedule and need an accelerator: the `pDMD 2 Steps` / `pDMD 4 Steps` accelerator profiles add the pDMD LoRA to `activated_loras`. Whole image about 2.5x faster at 2 steps, 1.6x at 4; 2 steps can invent pseudo-text on graphic prompts, 4 steps is the choice for quoted text and flat graphics; both work for edits and change the composition against 8 steps at the same seed; without the LoRA, fewer than 8 steps gives blurred, unfinished images. Profile `P-H3-Image 8 Steps (Reset)` clears accelerators (`loras_multipliers` `|`), 8 steps, one phase. `guidance_phases` 2: phase 1 at half width and height, H3 latent upscaler x2, phase 2 at full size from `switch_threshold` (min 0.7) for `num_inference_steps2` steps; the built-in P-H3-Image LoRA applies to both phases; without a selected accelerator with a nonzero phase-2 multiplier, the FL2VA LoRA Accelerator (Lightx2v 4 Steps) is added at `0;1`; accelerators count only before `|` in `loras_multipliers` (profiles and presets store `1;0 0;1|`), so without the bar that accelerator stacks on Regenerate. Presets (not accelerator profiles) `pDMD 2 Steps + Regenerate 2K Detail` (pDMD `1;0`, `open_h3_regenerate_2k_lora.safetensors` `0;1`, 2 + 3 steps) and `Regenerate 2K Detail (8 + 3 Steps)` use `custom_sigmas` `; 0.7, 0.659794, 0.513173` (Regenerate's 0.909375 / 0.857143 / 0.666667 grid scaled to start 0.7; scale to 0.5 for portraits). `custom_sigmas`: commas between descending sigmas below 1, `;` between exactly `guidance_phases` slots, blank slot = default schedule. Best at 8 MP (`2912x2912`, `3360x2496`, `2496x3360`, `3552x2368`, `2368x3552`, `3872x2176`, `2176x3872`); edits at 4 or 8 MP; artifacts in faces and hands grow below that. `video_prompt_type`: empty = text to image; `KI` = edit, first image defines output proportions within `resolution` (usual choice); `I` = edit on the exact `resolution`. The first image is edited, the others are references `<Picture 2>` and so on. Each image is used at its own size between 1 and 8 MP, so many large images need much more memory and time. Prompt enhancer is off by default; short prompts render weakly, so write full prompts. Weak on hands, small background people and dense text; short quoted text renders well. `activated_loras` come from the MiniMax H3 list: use FL2VA LoRAs, since this model is built on FL2VA; `open_h3_regenerate_2k_ic_lora_bf16.safetensors` (Ref2VA) is refused; H3 video accelerators other than pDMD are unchecked here. `batch_size` up to 16 images denoised together, with `guidance_phases` 1 only (image 0 has the composition of the single image of that seed, fine details differ slightly)."""

P_H3_IMAGE_DEEPY_PROMPT_INFOS = """Text to image: open with one sentence naming what the image physically is (`A photograph by...`, `A frame from a Japanese anime TV series`, `A two-color risograph print`), then one scene: viewpoint, one subject in a moment, a few supporting elements, light with a source and direction. Quote every visible word, keep strings short, say where they sit, end with `no other text`; without text, do not mention signs or labels. No hype words (stunning, epic, cinematic, vibrant, 8k, masterpiece); no very long prompts. Edit: `Edit <Picture 1>. <the change as the finished result>. Keep <everything that must stay> exactly as it is.` Name the exact object and its place, give new things a size relative to something present, list faces, poses, background and light to keep, quote new text. Other images are `<Picture 2>`, `<Picture 3>`."""

P_H3_IMAGE_TEXT_SYSTEM_PROMPT = """You write prompts for P-H3-Image, a high-resolution image model. Rewrite the user's request as one complete prompt for the finished image.

- Open with one sentence that names the real object the image is, such as "A photograph by an experienced street photographer", "A frame from a Japanese anime TV series", "A two-color risograph print" or "A flat scan of a handwritten recipe card". Honor a style the user asks for.
- Then describe one scene: a viewpoint, one main subject caught in a moment, a few supporting elements, and light with a source and a direction.
- Put every word visible in the image in double quotes, exactly as the user wrote it. Keep each string short, say where it sits and end with "no other text". If a design needs copy the user left open, write short final copy. If the image has no text, do not mention signs, labels or captions at all.
- Never use hype words such as stunning, epic, cinematic, vibrant, hyper-detailed, 8k or masterpiece.
- Keep the user's subject, language of quoted text and every explicit exclusion. Do not invent reference images.
- Write one paragraph of about 60 to 130 words. Return only the prompt, without commentary, Markdown or labels.

Example: A photograph by an experienced street photographer of an old fishmonger in a Lisbon market, seen from counter height as she wraps a sardine in paper. Crates of ice stand behind her and a hanging bulb lights her hands from the left. A hand-painted sign above her reads "PEIXE FRESCO", no other text.
"""

P_H3_IMAGE_EDIT_SYSTEM_PROMPT = """You write edit instructions for P-H3-Image, a high-resolution image model. The user supplies a request and images: <Picture 1> is the image to edit, and any others are references named <Picture 2>, <Picture 3> and so on. Write one instruction in this form:

Edit <Picture 1>. <The change, described as the finished result>. Keep <everything that must stay> exactly as it is.

- Start with exactly "Edit <Picture 1>." and include a sentence starting with "Keep".
- Point to the exact thing: name the object, person or text and where it is, such as "the old man's faded blue cap" rather than "his hat". The image descriptions are evidence, not text to copy.
- Describe how the edited part looks in the finished image, not the action. Give new things a size next to something already there, such as "about the size of the cat's head".
- Say what each reference supplies, such as "the straw hat from <Picture 2>".
- List what must not change, such as faces, poses, the background and the light. Without this list the model redraws the whole image.
- Put new text in double quotes, exactly as the user wrote it, and name the text around it that must stay.
- Never use hype words such as stunning, epic, cinematic, vibrant or masterpiece. Keep every explicit exclusion.
- Write one paragraph. Return only the instruction, without commentary, Markdown or labels.

Example: Edit <Picture 1>. The woman wears the straw hat from <Picture 2>, resting slightly tilted on her head. Keep her face, pose, clothes, the background and the afternoon light exactly as they are.
"""


def query_p_h3_image_def(denoising_priority_infos):
    """The model definition: a still-image workflow, none of the video, audio or sampling choices of the H3 video models."""
    return {
        "dtype": "bf16",
        "device_explicit": True,
        "image_outputs": True,
        "profile_type": "image",
        "cuda_graphs": ["transformer"],
        "tiny_vae_architecture": "minimax_h3_ref2va",
        "accelerated": "native",
        "specialities": [
            {"name": "8 megapixel images", "aliases": ["high resolution", "high resolution photorealism"], "description": "Renders 8 MP photographs, film frames and graphic styles in 8 steps, or 2 to 4 with the pDMD accelerator."},
            {"name": "multi-image editing", "aliases": ["image composition"], "description": "Edits one image and combines elements from up to 7 further images."},
        ],
        "block_size": 32,
        "vae_block_size": 32,
        "resolution_budget": "2912x2912",
        "guidance_max_phases": 2,
        "visible_phases": 0,
        "lora_multiplier_phases": 2,
        "phase_2_spatial_tiling": False,
        "switch_threshold": {"label": "Phase 2 Noise Level Start", "type": "number", "min": 0.7, "max": 1.0, "step": 0.0001},
        "custom_sigmas": True,
        "custom_sigmas_schedulers": ["euler"],
        "inference_steps": True,
        "inference_steps2": True,
        "flow_shift": False,
        "sample_solvers": [("Euler", "euler")],
        "no_negative_prompt": True,
        "custom_settings": [],
        "fit_into_canvas_image_refs": 0,
        "no_background_removal": True,
        "image_ref_choices": {
            "choices": [("Generate without Images", ""),
                        ("Edit the First Image at its Own Proportions, any Images after it are References", "KI"),
                        ("Edit with the Selected Output Size, any Images after the First are References", "I")],
            "letters_filter": "KI",
            "default": "",
            "label": "Images to Edit",
        },
        "infos": P_H3_IMAGE_INFOS,
        "prompt_infos": P_H3_IMAGE_PROMPT_INFOS,
        "deepy_infos": P_H3_IMAGE_DEEPY_INFOS,
        "deepy_prompt_infos": P_H3_IMAGE_DEEPY_PROMPT_INFOS,
        "prompt_enhancer_button_label": "Write",
        "prompt_enhancer_def": {
            "selection": ["T", "TI"],
            "labels": {"T": "An Image Prompt from Text", "TI": "An Edit Instruction from Text + {image_inputs}"},
            "default": "",
        },
        "prompt_enhancer_picture_labels": True,
        "text_prompt_enhancer_instructions": P_H3_IMAGE_TEXT_SYSTEM_PROMPT,
        "image_prompt_enhancer_instructions": P_H3_IMAGE_EDIT_SYSTEM_PROMPT,
        "text_prompt_enhancer_max_tokens": 512,
        "image_prompt_enhancer_max_tokens": 512,
        "profiles_dir": [P_H3_IMAGE_ARCHITECTURE],
        "preset_profiles_dir": [P_H3_IMAGE_ARCHITECTURE + "_presets"],
        "qkv_splitting": True,
        "qkv_layout": "interleaved",
        "text_encoder_folder": P_H3_IMAGE_TEXT_ENCODER_FOLDER,
        "text_encoder_URLs": [build_hf_url(P_H3_IMAGE_REPO_ID, P_H3_IMAGE_TEXT_ENCODER_FOLDER, filename) for filename in (P_H3_IMAGE_TEXT_ENCODER_BF16, P_H3_IMAGE_TEXT_ENCODER_INT8)],
        "system_configs": {},
        "system_configs2": {
            "_name": "VAE",
            "_default_label": "Auto",
            "fp16": {"name": "FP16", "video_vae_file": P_H3_IMAGE_VAE_FILE},
            "int8_convrot": {"name": "INT8 ConvRot Decoder", "video_vae_file": P_H3_IMAGE_VAE_INT8_FILE},
        },
        "system_configs3": {
            "_name": "DiT Denoising Priority",
            "_default_label": "Lower VRAM",
            "_infos": denoising_priority_infos,
            "lower_ram": {"name": "Lower RAM", "qkv_splitting": False},
        },
    }


__all__ = ["P_H3_IMAGE_ARCHITECTURE", "P_H3_IMAGE_FOLDER", "P_H3_IMAGE_MAX_IMAGES", "P_H3_IMAGE_REPO_ID", "P_H3_IMAGE_SHIFT", "P_H3_IMAGE_SIGMAS", "P_H3_IMAGE_STEPS",
           "P_H3_IMAGE_SWITCH_THRESHOLD", "P_H3_IMAGE_TEXT_ENCODER_BF16",
           "P_H3_IMAGE_TEXT_ENCODER_FOLDER", "P_H3_IMAGE_TEXT_ENCODER_INT8", "P_H3_IMAGE_VAE_FILE", "P_H3_IMAGE_VAE_INT8_FILE", "p_h3_image_sigmas", "query_p_h3_image_def",
           "reference_size", "size_at_area"]
