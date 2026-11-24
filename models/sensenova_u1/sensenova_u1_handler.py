# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)

import numpy as np
import torch
from PIL import Image

from shared.utils.hf import build_hf_url
from .prompt_enhancers import SENSENOVA_GENERIC_PROMPT, SENSENOVA_GENERIC_REFERENCE_PROMPT, SENSENOVA_INFOGRAPHIC_PROMPT, SENSENOVA_INFOGRAPHIC_REFERENCE_PROMPT, SENSENOVA_TEXT_FIX_PROMPT, SENSENOVA_TEXT_FIX_VISION_PROMPT


_VERSIONS = {"sensenova_u1_5_8b_mot": "1.5", "sensenova_u1_6_8b_mot": "1.6"}
_PROJECT_REPO = "DeepBeepMeep/SenseNova"
_PROJECT_FOLDER = "sensenova_u1_5"  # tokenizer and config files, identical for U1.6
_LORA_FOLDER = "sensenova_u1_5_8b_mot"  # shared by U1.5 and U1.6 (same weight structure)

SENSENOVA_INFOS = """## SenseNova-U{version}

SenseNova-U{version} is a unified model for text-to-image generation and image editing. Generate from text with no reference image, or provide one or more ordered reference images for editing and reference-guided generation.

### Practical Settings

- **Standard quality:** use `50` steps, **Guidance Scale** `4.0`, and **Flow Shift** `3.0`, matching the upstream reference settings.
- **Fast generation:** {accelerators}{self_correction}
- **Resolution:** `2048x2048` is a good general-purpose choice. Native 4K improves space for detail and typography but requires substantially more time and memory. WanGP keeps both dimensions aligned to the model's 32-pixel image grid.
- **Reference mode:** the default treats the first image as the main subject or landscape and derives output dimensions from it. Select **Use Reference Images** instead when every image is an ordinary reference and the chosen output resolution should be kept.
- **Reference quality:** use clean, high-resolution source images when possible. WanGP sends multiple references in their displayed order, so describe each image's role in the prompt.

### Current Limitations

- Dense, lengthy, very small, or mixed Chinese-English text can still contain spelling errors.
- Exact counts, alignment, and hierarchy may drift in heavily constrained layouts.
- Small faces, hands, limbs, and fine object structures can be unstable.
- Broad or complex multi-reference edits may alter content that was meant to stay unchanged; state preservation requirements explicitly.
- If details or colors become excessive, try lowering **Guidance Scale**.

See the [official model card](https://huggingface.co/sensenova/SenseNova-U{version}-8B-MoT) and [SenseNova-U1.5 cookbook](https://github.com/OpenSenseNova/SenseNova-U1/blob/main/docs/u1.5_best_practices.md) for upstream examples and guidance."""

_ACCELERATORS = {
    "1.5": "load the **Official 8-Step Accelerator** profile, which selects the matching LoRA and step count.",
    "1.6": "load the **Official 4-Step Accelerator** or **Official 8-Step Accelerator** profile, which selects the matching LoRA, step count and **Guidance Scale** `1`. The 4-step profile is about twice as fast as the 8-step one.",
}
SELF_CORRECTION_SETTING = {
    "id": "self_correction",
    "name": "Self-Correction",
    "label": "Self-Correction",
    "type": "dropdown",
    "default": "Disabled",
    "choices": [("Disabled", "Disabled"), ("Enabled", "Enabled"), ("Enabled, Keep the Draft", "Keep Draft"), ("Always Review", "Always"), ("Always Review, Keep the Draft", "Always Keep Draft")],
    "info": "The model draws the image, reviews its lettering against the prompt, writes the fixes it finds and draws a corrected version, which it may review again (up to 3 fixes). Best for text-rich posters and infographics; each round takes as long as a generation. The review is saved beside the image as a _critique.txt file. With Enabled the model may decide there is nothing to fix and keep its first version, sometimes despite visible errors (seen with the accelerator profiles at native 4K); Always Review makes it write its review every time. Keep the Draft also outputs the first version: it keeps the usual file name and the corrected versions get _fixed, _fixed2...; otherwise only the last version is saved.",
}
_TEXT_FIX_INFOS = {
    "1.5": "",
    "1.6": "\n- **Text Fix Edit** (with the image as the first reference): reads the lettering of the image, finds the misspelled, garbled or gibberish lines (or those your text asks to change) and writes box-located fixes in the form the model applies precisely, for example `in the right area [616, 433, 834, 459], replace \"Stage separaton\" with \"Stage separation\"`. Write a short request such as `Fix all the spelling mistakes`, or the prompt that produced the image so that its exact wording is used. Reading small lettering needs a capable vision model: a larger local enhancer (such as Qwen3.8 27B) or a remote one; small local enhancers may miss lines or misplace boxes. Proofread the instruction before generating.",
}
_SELF_CORRECTION_INFOS = {
    "1.5": "",
    "1.6": "\n- **Self-Correction:** the model draws the image, reviews its lettering against your prompt (typos, garbled or gibberish text), writes the fixes it finds and draws a corrected version, which it may review again (up to 3 fixes). It suits text-rich posters and infographics; each round takes as long as a generation; the review is saved as a `_critique.txt` file beside the image. With **Enabled** the model may judge its first version fine and stop there (the `_critique.txt` then says so), sometimes despite visible errors, notably with the accelerator profiles at native 4K; **Always Review** makes it write its review every time. The **Keep the Draft** choices also output the first version for comparison: it keeps the usual file name and the corrected versions are saved as `_fixed`, `_fixed2`...",
}
_DEEPY_SELF_CORRECTION = {"1.5": "", "1.6": " custom_settings.self_correction='Enabled' / 'Always' ('Keep Draft' / 'Always Keep Draft' also output the draft: files name, name_fixed, name_fixed2...; up to 3 fixes, the model decides when to stop): the model reviews its own lettering, writes box-located fixes (side file _critique.txt) and redraws; for text-heavy posters/infographics, each round costs a generation. 'Enabled' may stop after the draft despite errors (accelerated 4K drafts); 'Always' forces the review."}
_DEEPY_ACCELERATORS = {"1.5": "the official 8-step profile is a separate accelerated choice", "1.6": "the official 4-step and 8-step profiles (LoRA, guidance 1) are accelerated choices"}

SENSENOVA_PROMPT_INFOS = """## SenseNova-U{version} Prompt Guide

### Text-to-Image

Direct natural language works well for a clear subject with few constraints. Describe the **subject**, **setting**, **composition**, **style**, **palette**, **materials**, **lighting**, and **camera or viewpoint** in the order that matters most.

- State exact object counts and spatial relationships instead of relying on implication.
- For visible text, put every literal string in double quotes and specify its language, placement, size, and typographic hierarchy.
- For posters and infographics, organize the prompt into sections such as title, layout, panels, exact copy, icons, palette, typography, and constraints.
- Use prompt enhancement for a short creative brief that needs a full design plan. Skip enhancement when the prompt is already long, structured, and production-ready.

### Prompt Enhancer Modes

- **General Image Prompt:** expands a short brief into a balanced visual description without overloading a simple request.
- **Infographic Prompt:** builds a visually led information design with semantic icons, illustrations, charts, diagrams, compact copy, and explicit typography.
- Each mode also has a **First Reference Image** variant for edits or reference-guided subject, composition, layout, palette, and style reuse.{text_fix}

### Editing and References

Put the requested change first, then state what must remain unchanged.

- Name the edit target, location, quantity, color, material, size, or orientation precisely.
- For text replacement, write both strings exactly: `Replace "OLD TEXT" with "NEW TEXT"`.
- With multiple references, assign each a numbered role: `Use Image 1 as the base composition, Image 2 for the subject, and Image 3 for the color palette.`
- For a local edit, explicitly preserve identity, pose, framing, lighting, background, and every untouched region that matters.
- For a style reference, describe which visible traits to transfer—palette, texture, typography, lighting, spacing, or composition—rather than saying only `use this style`.

### Prompt Shapes

- **Generation:** `A cinematic mountain lake at sunrise, realistic photography, calm reflections, natural mist, balanced wide composition.`
- **Editing:** `Change the jacket to cobalt blue. Preserve the face, pose, background, lighting, and framing.`
- **Infographic:** `Create a vertical infographic titled "CITY TREES". Use three clearly separated panels with the exact headings "COOLER STREETS", "CLEANER AIR", and "WILDLIFE", crisp sans-serif typography, restrained green palette, and no additional text.`

Upstream recommends prompt enhancement especially for short infographic prompts because additional structure improves hierarchy, typography, and information density. See the [prompt-enhancement guide](https://github.com/OpenSenseNova/SenseNova-U1/blob/main/docs/prompt_enhancement.md)."""


class family_handler:
    @staticmethod
    def query_model_def(base_model_type, model_def):
        version = _VERSIONS[base_model_type]
        return {
            "image_outputs": True,
            "device_explicit": True,
            "no_model_swap": True,  # one managed model: kept loaded between samples and queued tasks of the same model
            "cuda_graphs": ["transformer"],
            "dtype": "bf16",
            "no_negative_prompt": True,
            "no_background_removal": True,
            "guidance_max_phases": 1,
            "inference_steps": True,
            "flow_shift": True,
            "fit_into_canvas_image_refs": 0,
            "image_ref_choices": {
                "choices": [
                    ("Generate without Reference Images", ""),
                    ("First Reference Image is the Main Subject / Landscape, defines Output Dimensions, and may be followed by other Reference Images", "KI"),
                    ("Use Reference Images", "I"),
                ],
                "letters_filter": "KI",
                "default": "KI",
                "label": "Reference Images",
            },
            "at_least_one_image_ref_needed": False,
            "image_prompt_types_allowed": "S",
            "infos": SENSENOVA_INFOS.format(version=version, accelerators=_ACCELERATORS[version], self_correction=_SELF_CORRECTION_INFOS[version]),
            "specialities": [{"name": "infographics", "aliases": ["information graphics", "information posters"]}, {"name": "text rendering", "description": "Chinese and English lettering within images."}, {"name": "precise layouts"}],
            "prompt_infos": SENSENOVA_PROMPT_INFOS.format(version=version, text_fix=_TEXT_FIX_INFOS[version]),
            "deepy_infos": f"### SenseNova-U{version}\n\nText-to-image and editing with ordered references. First-reference main-scene mode derives dimensions from that image; ordinary-reference mode keeps chosen resolution. Start at 50 steps, guidance 4, shift 3, 2048px; {_DEEPY_ACCELERATORS[version]}. Strengths: infographics, Chinese/English text, layouts. Proofread text, counts and placement; complex edits can affect untouched content.{_DEEPY_SELF_CORRECTION[version]}",
            "deepy_prompt_infos": '### Prompt\n\nDescribe subject, composition, style, palette and light. For infographics specify layout/panels, hierarchy and every exact string in quotes. For edits, state change and preservation constraints; number reference roles. Example: `Replace "CLOSED" with "OPEN", retaining the sign material and perspective.` Enhancement is optional; use it for short briefs needing design detail.',
            "preview_all_images": True,
            "prompt_enhancer_button_label": "Enhance",
            "prompt_enhancer_def": {
                "selection": ["T", "TI", "T1", "TI1"] + (["TI2"] if version == "1.6" else []),
                "labels": {
                    "T": "A General Image Prompt using existing Text Prompt",
                    "TI": "A General Image Prompt using existing Text Prompt and {image_inputs}",
                    "T1": "An Infographic Prompt using existing Text Prompt",
                    "TI1": "An Infographic Prompt using existing Text Prompt and {image_inputs}",
                    "TI2": "A Text Fix Edit of {image_inputs} using existing Text Prompt",
                },
                "default": "",
            },
            "text_prompt_enhancer_instructions": SENSENOVA_GENERIC_PROMPT,
            "image_prompt_enhancer_instructions": SENSENOVA_GENERIC_REFERENCE_PROMPT,
            "text_prompt_enhancer_instructions1": SENSENOVA_INFOGRAPHIC_PROMPT,
            "image_prompt_enhancer_instructions1": SENSENOVA_INFOGRAPHIC_REFERENCE_PROMPT,
            "text_prompt_enhancer_max_tokens": 1024,
            "image_prompt_enhancer_max_tokens": 1024,
            "text_prompt_enhancer_max_tokens1": 1536,
            "image_prompt_enhancer_max_tokens1": 1536,
            # Text Fix Edit (U1.6): box-located text fixes, the syntax of the model's own critiques, which its edit mode applies. Local
            # enhancers find them with their vision stage (high resolution, to read small lettering), remote engines see the image
            **({"image_prompt_enhancer_instructions2": SENSENOVA_TEXT_FIX_PROMPT, "image_prompt_enhancer_max_tokens2": 1536,
                "image_prompt_enhancer_vision_instructions2": SENSENOVA_TEXT_FIX_VISION_PROMPT, "image_prompt_enhancer_vision_max_tokens2": 3072,
                "image_prompt_enhancer_vision_highres2": True} if version == "1.6" else {}),
            "profiles_dir": [base_model_type],
            "resolution_budget": "4096x4096",
            "skip_prompt_template": True,
            "vae_block_size": 32,
            "vision_rope_cpu_tables": version == "1.6",  # upstream 925e7addc7: patch RoPE tables built on the CPU like the checkpoint's training; U1.5 keeps its outputs
            **({"custom_settings": [SELF_CORRECTION_SETTING]} if version == "1.6" else {}),
        }

    @staticmethod
    def query_supported_types():
        return list(_VERSIONS)

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "sensenova"

    @staticmethod
    def query_family_infos():
        return {"sensenova": (1170, "SenseNova")}

    @staticmethod
    def get_lora_dir(base_model_type):
        return _LORA_FOLDER

    @staticmethod
    def preview_latents(base_model_type, latents, meta):
        if not torch.is_tensor(latents) or latents.dim() != 4 or latents.shape[0] != 3:
            return None
        # with NumPy, on this thread alone: PyTorch's CPU ops wake its pool of threads, which then spin and slow the next denoising step
        image = latents.detach().float().cpu().numpy()
        channels, frames, height, width = image.shape
        image = image.transpose(0, 2, 1, 3).reshape(channels, height, frames * width)
        image = ((np.clip(image, -1, 1) + 1) * 127.5).clip(0, 255).astype(np.uint8)
        preview = Image.fromarray(np.ascontiguousarray(image.transpose(1, 2, 0)))
        if preview.height > 0:
            scale = 200 / preview.height
            resampling_module = getattr(Image, "Resampling", Image)
            preview = preview.resize((max(1, int(round(preview.width * scale))), 200), resample=getattr(resampling_module, "BILINEAR", Image.BILINEAR))
        return preview

    @staticmethod
    def query_retired_files(model_type):
        """Checkpoints that earlier releases of a model type downloaded and its current definition no longer names (Models Manager
        lists them with the model, to delete them)."""
        return {"sensenova_u1_5_8b_mot": ["SenseNova_U1.5_8B_MoT_int8_convrot.safetensors"]}.get(model_type, [])

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return [{
            "repoId": _PROJECT_REPO,
            "sourceFolderList": [_PROJECT_FOLDER],
            "fileList": [[
                "added_tokens.json",
                "config.json",
                "merges.txt",
                "special_tokens_map.json",
                "tokenizer_config.json",
                "vocab.json",
            ]],
        }]

    @staticmethod
    def load_model(
        model_filename,
        model_type=None,
        base_model_type=None,
        model_def=None,
        quantizeTransformer=False,
        text_encoder_quantization=None,
        dtype=torch.bfloat16,
        VAE_dtype=torch.float32,
        mixed_precision_transformer=False,
        save_quantized=False,
        submodel_no_list=None,
        text_encoder_filename=None,
        **kwargs,
    ):
        from .sensenova_u1_main import model_factory

        processor = model_factory(
            model_filename=model_filename,
            model_type=model_type,
            base_model_type=base_model_type,
            dtype=dtype,
            save_quantized=save_quantized,
            vision_rope_cpu_tables=model_def.get("vision_rope_cpu_tables", False),
        )
        return processor, {"transformer": processor.transformer}

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({
            "image_mode": 1,
            "num_inference_steps": 50,
            "guidance_scale": 4.0,
            "flow_shift": 3.0,
            "batch_size": 1,
            "video_prompt_type": "",
        })
