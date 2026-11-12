# Copyright 2026 The MiniMax and HuggingFace Teams. All rights reserved.
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
# WanGP adaptation of the MiniMax-H3-Image sizing and sampling grid (diffusers modular pipeline, huggingface/diffusers#15230).
# Modifications Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""MiniMax-H3-Image: MiniMax's own image model on the H3 architecture without its audio stream, sampled in 6 steps with one output head
per step, conditioned by H3's Qwen3-VL text encoder and decoded by a 2D ViT decoder in H3's latent space."""

import math

import torch


H3_IMAGE_ARCHITECTURE = "minimax_h3_image"
H3_IMAGE_PRUNED_ARCHITECTURE = "minimax_h3_image_pruned"
H3_IMAGE_ARCHITECTURES = (H3_IMAGE_ARCHITECTURE, H3_IMAGE_PRUNED_ARCHITECTURE)
H3_IMAGE_MODEL = "h3_image"  # the pipeline variant; P-H3-Image is "p_h3_image"
H3_IMAGE_FAMILY = "h3_image"  # the UI family of the H3 image models
H3_IMAGE_FOLDER = "minimax_h3_image"
H3_IMAGE_VAE_FILE = f"{H3_IMAGE_FOLDER}/minimax_h3_image_vae_fp16.safetensors"
# The grid the 6 output heads were distilled on, each sigma rounded through its float32 time 1 - sigma as the released sampler does
H3_IMAGE_SIGMAS = tuple((1.0 - torch.tensor([1.0 - sigma for sigma in (1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132)],
                                            dtype=torch.float32, device="cpu")).tolist())
H3_IMAGE_STEPS = len(H3_IMAGE_SIGMAS)
H3_IMAGE_MIN_AREA = 2048 * 2048  # smaller requests are generated on a canvas of this area and resized back
H3_IMAGE_MAX_IMAGES = 16
H3_IMAGE_REFERENCE_SHORT_EDGE = (1024, 2048)  # a reference keeps its size when its short edge is within these bounds
H3_IMAGE_VISION_SHORT_EDGE = 1024  # what the text encoder sees of each reference
H3_IMAGE_PICTURE_LABEL = "Picture {}: "  # the label of a reference in the text encoder input; prompts refer to it as <Picture N>


def _round(value, multiple=32):
    return max(multiple, round(value / multiple) * multiple)


def canvas_size(width, height, multiple=32):
    """The (width, height) an image is generated at: scaled up to H3_IMAGE_MIN_AREA at the same proportions when smaller, sides rounded."""
    scale = max(1.0, math.sqrt(H3_IMAGE_MIN_AREA / (width * height)))
    return _round(width * scale, multiple), _round(height * scale, multiple)


def bound_short_edge(width, height, low, high, multiple=32):
    """A size whose short edge is within [low, high] is kept; otherwise it is scaled to the nearer bound, sides rounded to ``multiple``."""
    short = min(width, height)
    if low <= short <= high:
        return width, height
    scale = (low if short < low else high) / short
    return _round(width * scale, multiple), _round(height * scale, multiple)


H3_IMAGE_SHIFT = 12.0  # the grid of other step counts: H3's shift-12 linspace, the one few-step LoRAs such as pDMD are trained on


def h3_image_sigmas(steps=H3_IMAGE_STEPS):
    """The noise levels of ``steps`` denoising steps, terminal 0 left out: the checkpoint's own grid at 6 steps, otherwise H3's
    ``linspace(1, 0, steps + 1)`` grid at shift 12."""
    if steps == H3_IMAGE_STEPS:
        return H3_IMAGE_SIGMAS
    base = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float32, device="cpu")[:-1]
    return tuple((H3_IMAGE_SHIFT * base / (1 + (H3_IMAGE_SHIFT - 1) * base)).tolist())


def h3_image_head_plans(sigmas):
    """(steps, 6) weights of the per-step output heads for a step schedule (sigmas with their terminal 0): a step mixes the heads of the
    checkpoint's grid intervals it overlaps, in proportion to the overlap, which is one head per step on the checkpoint's own grid."""
    grid = torch.tensor([*H3_IMAGE_SIGMAS, 0.0], dtype=torch.float64, device="cpu")
    sigmas = torch.as_tensor(sigmas).flatten().to(device="cpu", dtype=torch.float64)
    plans = []
    for high, low in zip(sigmas[:-1], sigmas[1:]):
        overlap = (torch.minimum(grid[:-1], high) - torch.maximum(grid[1:], low)).clamp_min_(0.0)
        plans.append(overlap / overlap.sum())
    return torch.stack(plans).float()


H3_IMAGE_INFOS = """## MiniMax H3-Image

Generates a single image from text, or edits and combines up to 16 input images, in 6 steps without guidance or a negative prompt, or in 2 to 4 steps with its pDMD accelerator profiles. It is MiniMax's image model on the H3 architecture: no video or audio.

### Text to Image

Describe the whole finished image in detail. **About 16 MP works best**: WanGP lists the recommended 16 MP sizes first, then 4 MP sizes. Requests below 4 MP are generated at 4 MP with the same proportions and resized, so smaller sizes save no time. Sides are multiples of 32.

### Editing

Choose an option under **Images to Edit**, then add 1 to 16 images. The prompt refers to them as `<Picture 1>`, `<Picture 2>` and so on, in the order you add them.

- **Edit the First Image at its Own Proportions:** the first image sets the output proportions, within the selected resolution. Choose this for most edits.
- **Edit with the Selected Output Size:** the output keeps the selected resolution.

Each image is used at its own size and only resized when its short side is under 1024 or over 2048 pixels, never to the output size. Many large images use much more memory and time.

### Prompts

The model follows long, complete descriptions much better than short requests. The optional **Write** button turns a short request into the detailed description the model expects: a full caption for text to image, or for edits a description of what stays, what changes and how the result looks.

### Pruned Version

**H3-Image Pruned** is smaller (about 40 GB in BF16 instead of 66 GB) and loads faster, for nearly identical images.

### Fewer Steps

The **pDMD** accelerator profiles add a LoRA that finishes the image in fewer steps:

- **pDMD 2 Steps:** the fastest: denoising takes less than half the time of 6 steps. Reading the prompt and decoding the image take as long as before, so a whole 4 MP image is about 25 % faster.
- **pDMD 4 Steps:** crisper detail than 2 steps, for a little more time.

For the same seed they give a different picture than 6 steps. Without the LoRA, fewer than 6 steps leaves the image soft and unfinished. At 6 steps the model uses its own tuned schedule; other step counts use the schedule the pDMD LoRA is trained on. **H3-Image 6 Steps (Reset)** goes back to the default settings and keeps your other LoRAs.

### Several Images

**Number of Images** generates several variations of the same prompt together, faster than one after the other. The first image keeps the composition a single generation with the same seed gives, with slightly different fine details. Each extra image needs more memory during generation.

### LoRAs

H3-Image shares the MiniMax H3 LoRA list. It is a heavily retrained H3, so style, realism or character LoRAs made for the video models can carry over, often with a weaker or different effect: try them at a few strengths. LoRAs for video motion, control or fewer steps do not suit it, except the pDMD accelerator of its profiles. The sampler and guidance are fixed.
"""

H3_IMAGE_PROMPT_INFOS = """### Text to Image

Write one detailed English paragraph that describes the whole image, the way a designer would brief it:

1. **Say what the image is:** `The image is an overhead photograph of...`, `The image is a two-color risograph poster...`.
2. **Place everything:** what sits in the center, on the left, at the upper right; sizes, materials, light with its source and direction.
3. **Spell out every word** in double quotes, exactly, and say where it sits.
4. **For posters, pages, charts and interfaces**, describe each region in reading order: title, subtitle, main visual, panels, labels, footer.
5. **Use concrete visual cues** rather than hype words such as stunning, epic or masterpiece.

`The image is an overhead, top-down photograph of a rustic farm-to-table feast on a weathered dark wooden table. In the center, a woven basket holds a round sourdough loaf with a cracked golden crust. To the left, a dark ceramic plate carries orange carrots, heirloom tomatoes and kale; on the right, a wooden board holds aged cheddar, brie and walnuts. Warm natural light comes from the upper left and casts long, soft shadows to the lower right.`

### Editing

Name the parts of the images to keep, then the change, then describe the result. MiniMax's own expander writes edits in this form, which the **Write** button reproduces:

`The man on the left in a dark sweater in <Picture 1> is <Subject 1>. The podcast studio in <Picture 1> is <Subject 2>. The target image shows <Subject 1> without his headphones in <Subject 2>. <Subject 1> is partially preserved; his face, sweater and pose remain, while the headphones are removed and his hair is filled in where the band flattened it. <Subject 2> is fully preserved. His short dark curls now cover the top of his head naturally, with no dent.`

A plain instruction also works: `Keep everything in <Picture 1> as it is, but remove the headphones and have the two hosts fist bump.` With several images, say what each one supplies: `the straw hat from <Picture 2>`. Quote new text exactly.
"""

H3_IMAGE_DEEPY_INFOS = """Single images from `prompt`, or edits with 1 to 16 ordered `image_refs` (`<Picture 1>`, `<Picture 2>` in the prompt); no video or audio. Euler, no guidance, no negative prompt, one phase, no step skipping. `num_inference_steps` defaults to 6 (the checkpoint's own schedule); other counts use H3's shift-12 schedule and need the pDMD LoRA: accelerator profiles `pDMD 2 Steps` / `pDMD 4 Steps` add `minimax_h3_pdmd_2nfe_bf16.safetensors` to `activated_loras` (denoising more than 2x faster at 2 steps, a whole 4 MP image about 25 % faster; 4 steps is crisper; a different picture than 6 steps at the same seed; without the LoRA fewer than 6 steps is soft and unfinished). Profile `H3-Image 6 Steps (Reset)` clears accelerators (`loras_multipliers` `|`). Best near 16 MP (`4096x4096`, `4736x3552`, `3552x4736`, `5024x3360`, `3360x5024`, `5472x3072`, `3072x5472`, `6272x2688`, `2688x6272`); requests below 4 MP are generated at 4 MP and resized (no time saved). `video_prompt_type`: empty = text to image; `KI` = edit, the first image sets the output proportions within `resolution` (usual choice); `I` = edit on the exact `resolution`. References keep their own size (resized only when the short side is outside 1024-2048 px), so many large references cost much more memory and time. Long, complete English descriptions work much better than short requests; the prompt enhancer is off by default. `minimax_h3_image_pruned` is the lighter variant (40 GB BF16 vs 66 GB) with nearly identical images. Shares the MiniMax H3 LoRA folder: H3 video style, realism or character LoRAs may carry over with a weaker or different effect (heavily retrained H3; not validated); video motion, control and step LoRAs do not apply, except pDMD. `batch_size` up to 16: several images of one prompt denoised together (faster than one by one; image 0 has the composition of the single image of that seed, fine details differ slightly; more VRAM)."""

H3_IMAGE_DEEPY_PROMPT_INFOS = """Text to image: one detailed English paragraph describing the whole image, usually starting `The image is a/an <medium>...`: subject identity, pose, positions (center, left, upper right), materials, light source and direction, lens or medium; quote every visible word in double quotes with its position; for posters, pages, charts and UIs describe each region in reading order with its exact text. No hype words. Edit: name kept source elements first (`The <cues> in <Picture 1> is <Subject 1>.`), then `The target image shows ...`, then one clause per subject (`<Subject N> is fully preserved.` or `<Subject N> is partially preserved; <what stays>, while <what changes>.`), then a concrete description of only the changed or new content; quote edited text exactly in single quotes. A plain instruction naming what to keep and what to change also works."""

H3_IMAGE_TEXT_SYSTEM_PROMPT = """You expand image requests for MiniMax-H3-Image, a high-resolution text-to-image model that works best from one long, detailed English description of the whole image. Rewrite the user's request as one complete caption of the finished image, written like an expert visual designer and photographer, so that every important part of the picture can be inspected. The goal is not more words: every sentence must describe something visible.

The image is {{width}}x{{height}} pixels: compose for this canvas, spreading content sideways when it is wide and vertically when it is tall, but never mention the canvas, its size or its aspect ratio. A poster, page, card, cover or screen fills the whole canvas edge to edge with the canvas's own proportions, unless the user asks to show it as an object inside a scene: never describe it as vertical, horizontal or square against the canvas.

First decide what kind of image it is.
- A depicted scene or photograph: describe subject identity, pose and contact, scale, environment, light with its source and direction, lens or medium, and keep everything physically plausible.
- A communication artifact (poster, cover, packaging, card, web page, user interface, chart, dashboard, infographic, technical diagram, printed page, product board, multi-panel composite): build its finished layout in reading order, region by region (title, subtitle, main visual, panels, cards, chart area, legend, footer), giving each region its position, its exact visible text, what it shows and how it is styled. Never use vague fillers such as "icons", "details", "information" or "annotations" without saying what they show. Charts need a title, axis labels, units, legend entries and values placed consistently.

Rules:
1. Keep every hard fact of the request exactly: requested text, names, counts, numbers, formulas, data values, style, medium, orientation and left, right, top and bottom order.
2. Required content must be visible, not implied. Put every visible word in double quotes, exactly as the user wrote it and in its original language and script, and say where it sits. When a design needs copy the user left open, write short, complete final copy yourself; never leave placeholders. Add no writing to an image that needs none.
3. For people in action, fix the scene and one camera first, then state each body's orientation to the camera (front-facing, back-facing, left or right profile, three-quarter) and its direction of motion in the frame, frozen in one moment: a limb cannot be in two places at once.
4. For hands, grips, hugs and held objects, say which hand touches what and frame the shot so the contact is readable; prefer a medium shot when hands or faces matter.
5. Use world knowledge only as visible evidence: real materials, tools, clothing, architecture, scientific or cultural details, units and labels. Add no backstory that cannot be seen.
6. Keep the number of main subjects the user asked for; background people stay small and secondary.
7. Content first, then style: replace mood words and generic adjectives (stunning, epic, cinematic, vibrant, masterpiece, 8k) with concrete cues of light, color, texture and lens.

Write one fluent English paragraph that starts with "The image is", usually 150 to 350 words and longer only for dense designs, and finish it within the response. Return only the caption, without commentary, Markdown, labels or surrounding quotes.

Example: The image is an overhead, top-down photograph of a rustic farm-to-table feast spread across a weathered dark wooden table with visible grain and knots. In the center, a woven basket holds a round sourdough loaf with a cracked golden crust and a dusting of flour. To the left, a dark ceramic plate carries bright orange carrots, deep red heirloom tomatoes and a mound of green kale. On the right, a wooden board displays a wedge of aged cheddar, a round of brie and a few scattered walnuts. At the upper right, a glass jar of golden olive oil stands beside a jar of honey with a wooden dipper. Warm natural light comes from the upper left and casts soft, long shadows to the lower right across the bread and vegetables.
"""

H3_IMAGE_EDIT_SYSTEM_PROMPT = """You expand image-editing requests for MiniMax-H3-Image. The user gives a short instruction and one or more input images, labeled <Picture 1>, <Picture 2> and so on in their order; what you are told about the images is evidence of what they show. Write one rich English description of the intended finished image, called the target image, as one continuous paragraph in three parts, in this order. The target image is {{width}}x{{height}} pixels; if this changes the framing of a source picture, treat the reframing as a change.

Part 1, the subjects. Name every source element that stays visible in the target image or lends it a property, with its picture and a few unambiguous visible cues: "The man on the left in a dark sweater in <Picture 1> is <Subject 1>." Number subjects from 1; the same person or object seen in several pictures keeps one number, and each definition names its picture explicitly. Define a part, a text block, a color, a pose, the background or the layout on its own only when it changes or is used on its own. Transcribe the readable text of a defined subject once, exactly, in single quotes: "The wooden desk sign reading 'MONEY TALKS' in <Picture 1> is <Subject 2>." Never guess unreadable text or invent source details. Anything removed or replaced, and anything new, gets no subject number.

Part 2, the relationships. One sentence starting with "The target image" that sums up the result with the subject tokens. Then one sentence per subject:
- "<Subject N> is fully preserved." when nothing about it changes; write exactly this and never describe that subject again.
- "<Subject N> is partially preserved; ..." saying what stays (identity, face, clothing, shape) and what changes, from what to what.
- For a property taken from a source, say which subject supplies which property to which target: "<Subject 1> supplies the exact color transferred to <Subject 3>."
- For a loose style or palette reference, say what is borrowed and what the new design may vary.
A subject is not fully preserved when anything inside it changes, including its pose, position, lighting or text.

Part 3, the changes. Describe richly and concretely only what changes or is new: the final look of changed parts, new objects, people, text, backgrounds and layouts, and how they fit in (size compared with something present, position, contact, occlusion, perspective, shadows, reflections, light direction). A newly requested scene, design or style may be described completely and vividly. Use bare subject tokens only as anchors, as in "a lantern now hangs beside <Subject 2>", without describing preserved subjects again. Write all new or edited text in full and exactly, in single quotes; when the user asks for copy without giving the words, write the final words yourself. For grids, state the rows and columns explicitly, such as "2 rows and 3 columns".

Interpret terse or informal requests sensibly and complete them, keep every explicit constraint and exclusion, leave unmentioned content as it is and never present an invented detail as a source fact. Write the description in English, while quoted text keeps its own language. Refer to the result only as "the target image" and never narrate the instruction or use commands such as "edit" or "generate". Use no headings, bullets, line breaks, Markdown or commentary: return only the paragraph.

Example: The black horse in <Picture 1> is <Subject 1>. The meadow at dusk in <Picture 1> is <Subject 2>. The target image shows <Subject 1> standing in <Subject 2> beside a newly added luminous pearl. <Subject 1> is fully preserved. <Subject 2> is fully preserved. A pearl about the size of an apple rests in the grass just in front of <Subject 1>, glowing a cool blue-white that tints the nearby blades of grass and casts a faint pale reflection on the ground.
"""


def query_h3_image_def(pruned, text_encoder_def, denoising_priority_infos):
    """The model definition: a still-image workflow on a fixed 6-step recipe, none of the video, audio or sampling choices of the H3 video models."""
    return {
        "dtype": "bf16",
        "device_explicit": True,
        "image_outputs": True,
        "profile_type": "image",
        "cuda_graphs": ["transformer"],
        "tiny_vae_architecture": "minimax_h3_ref2va",  # H3's latent space
        "accelerated": "native",
        "size": "lighter" if pruned else "large",
        "block_size": 32,
        "vae_block_size": 32,
        "resolution_budget": "4480x4480",
        "guidance_max_phases": 1,
        "lock_guidance_phases": True,
        "visible_phases": 0,
        "lora_multiplier_phases": 1,
        "inference_steps": True,
        "flow_shift": False,
        "custom_sigmas": False,
        "spectrum_cache": False,
        "first_block_cache": False,
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
        "infos": H3_IMAGE_INFOS,
        "prompt_infos": H3_IMAGE_PROMPT_INFOS,
        "deepy_infos": H3_IMAGE_DEEPY_INFOS,
        "deepy_prompt_infos": H3_IMAGE_DEEPY_PROMPT_INFOS,
        "prompt_enhancer_button_label": "Write",
        "prompt_enhancer_def": {
            "selection": ["T", "TI"],
            "labels": {"T": "An Image Description from Text", "TI": "An Edit Description from Text + {image_inputs}"},
            "default": "",
        },
        "prompt_enhancer_picture_labels": True,
        "text_prompt_enhancer_instructions": H3_IMAGE_TEXT_SYSTEM_PROMPT,
        "image_prompt_enhancer_instructions": H3_IMAGE_EDIT_SYSTEM_PROMPT,
        "text_prompt_enhancer_max_tokens": 1024,
        "image_prompt_enhancer_max_tokens": 1536,
        "profiles_dir": [H3_IMAGE_ARCHITECTURE],
        "qkv_splitting": True,
        "qkv_layout": "interleaved",
        **text_encoder_def,
        "system_configs2": {
            "_name": "DiT Denoising Priority",
            "_default_label": "Lower VRAM",
            "_infos": denoising_priority_infos,
            "lower_ram": {"name": "Lower RAM", "qkv_splitting": False},
        },
    }


__all__ = ["H3_IMAGE_ARCHITECTURE", "H3_IMAGE_ARCHITECTURES", "H3_IMAGE_FAMILY", "H3_IMAGE_FOLDER", "H3_IMAGE_MAX_IMAGES", "H3_IMAGE_MIN_AREA", "H3_IMAGE_MODEL",
           "H3_IMAGE_PICTURE_LABEL", "H3_IMAGE_PRUNED_ARCHITECTURE", "H3_IMAGE_REFERENCE_SHORT_EDGE", "H3_IMAGE_SIGMAS", "H3_IMAGE_STEPS", "H3_IMAGE_VAE_FILE",
           "H3_IMAGE_SHIFT", "H3_IMAGE_VISION_SHORT_EDGE", "bound_short_edge", "canvas_size", "h3_image_head_plans", "h3_image_sigmas", "query_h3_image_def"]
