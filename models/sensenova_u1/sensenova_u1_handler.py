# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)

import torch
from PIL import Image

from shared.utils.hf import build_hf_url
from .prompt_enhancers import SENSENOVA_GENERIC_PROMPT, SENSENOVA_GENERIC_REFERENCE_PROMPT, SENSENOVA_INFOGRAPHIC_PROMPT, SENSENOVA_INFOGRAPHIC_REFERENCE_PROMPT


_ARCHITECTURE = "sensenova_u1_5_8b_mot"
_PROJECT_REPO = "DeepBeepMeep/SenseNova"
_PROJECT_FOLDER = "sensenova_u1_5"
_PROFILE_FOLDER = _ARCHITECTURE

SENSENOVA_INFOS = """## SenseNova-U1.5

SenseNova-U1.5 is a unified model for text-to-image generation and image editing. Generate from text with no reference image, or provide one or more ordered reference images for editing and reference-guided generation.

### Practical Settings

- **Standard quality:** use `50` steps, **Guidance Scale** `4.0`, and **Flow Shift** `3.0`, matching the upstream reference settings.
- **Fast generation:** load the **Official 8-Step Accelerator** profile, which selects the matching LoRA and step count.
- **Resolution:** `2048x2048` is a good general-purpose choice. Native 4K improves space for detail and typography but requires substantially more time and memory. WanGP keeps both dimensions aligned to the model's 32-pixel image grid.
- **Reference mode:** the default treats the first image as the main subject or landscape and derives output dimensions from it. Select **Use Reference Images** instead when every image is an ordinary reference and the chosen output resolution should be kept.
- **Reference quality:** use clean, high-resolution source images when possible. WanGP sends multiple references in their displayed order, so describe each image's role in the prompt.

### Current Limitations

- Dense, lengthy, very small, or mixed Chinese-English text can still contain spelling errors.
- Exact counts, alignment, and hierarchy may drift in heavily constrained layouts.
- Small faces, hands, limbs, and fine object structures can be unstable.
- Broad or complex multi-reference edits may alter content that was meant to stay unchanged; state preservation requirements explicitly.
- If details or colors become excessive, try lowering **Guidance Scale**.

See the [official model card](https://huggingface.co/sensenova/SenseNova-U1.5-8B-MoT) and [SenseNova-U1.5 cookbook](https://github.com/OpenSenseNova/SenseNova-U1/blob/refs/heads/feat/u1.5/docs/u1.5_best_practices.md) for upstream examples and guidance."""

SENSENOVA_PROMPT_INFOS = """## SenseNova-U1.5 Prompt Guide

### Text-to-Image

Direct natural language works well for a clear subject with few constraints. Describe the **subject**, **setting**, **composition**, **style**, **palette**, **materials**, **lighting**, and **camera or viewpoint** in the order that matters most.

- State exact object counts and spatial relationships instead of relying on implication.
- For visible text, put every literal string in double quotes and specify its language, placement, size, and typographic hierarchy.
- For posters and infographics, organize the prompt into sections such as title, layout, panels, exact copy, icons, palette, typography, and constraints.
- Use prompt enhancement for a short creative brief that needs a full design plan. Skip enhancement when the prompt is already long, structured, and production-ready.

### Prompt Enhancer Modes

- **General Image Prompt:** expands a short brief into a balanced visual description without overloading a simple request.
- **Infographic Prompt:** builds a visually led information design with semantic icons, illustrations, charts, diagrams, compact copy, and explicit typography.
- Each mode also has a **First Reference Image** variant for edits or reference-guided subject, composition, layout, palette, and style reuse.

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
            "infos": SENSENOVA_INFOS,
            "specialities": [{"name": "infographics", "aliases": ["information graphics", "information posters"]}, {"name": "text rendering", "description": "Chinese and English lettering within images."}, {"name": "precise layouts"}],
            "prompt_infos": SENSENOVA_PROMPT_INFOS,
            "deepy_infos": "### SenseNova-U1.5\n\nText-to-image and editing with ordered references. First-reference main-scene mode derives dimensions from that image; ordinary-reference mode keeps chosen resolution. Start at 50 steps, guidance 4, shift 3, 2048px; the official 8-step profile is a separate accelerated choice. Strengths: infographics, Chinese/English text, layouts. Proofread text, counts and placement; complex edits can affect untouched content.",
            "deepy_prompt_infos": '### Prompt\n\nDescribe subject, composition, style, palette and light. For infographics specify layout/panels, hierarchy and every exact string in quotes. For edits, state change and preservation constraints; number reference roles. Example: `Replace "CLOSED" with "OPEN", retaining the sign material and perspective.` Enhancement is optional; use it for short briefs needing design detail.',
            "preview_all_images": True,
            "prompt_enhancer_button_label": "Enhance",
            "prompt_enhancer_def": {
                "selection": ["T", "TI", "T1", "TI1"],
                "labels": {
                    "T": "A General Image Prompt using existing Text Prompt",
                    "TI": "A General Image Prompt using existing Text Prompt and {image_inputs}",
                    "T1": "An Infographic Prompt using existing Text Prompt",
                    "TI1": "An Infographic Prompt using existing Text Prompt and {image_inputs}",
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
            "profiles_dir": [_PROFILE_FOLDER],
            "resolution_budget": "4096x4096",
            "skip_prompt_template": True,
            "vae_block_size": 32,
        }

    @staticmethod
    def query_supported_types():
        return [_ARCHITECTURE]

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
        return _PROFILE_FOLDER

    @staticmethod
    def preview_latents(base_model_type, latents, meta):
        if not torch.is_tensor(latents) or latents.dim() != 4 or latents.shape[0] != 3:
            return None
        image = latents.detach().float().cpu().clamp(-1, 1)
        channels, frames, height, width = image.shape
        image = image.permute(0, 2, 1, 3).reshape(channels, height, frames * width)
        image = image.add(1).mul(127.5).clamp(0, 255).to(torch.uint8)
        preview = Image.fromarray(image.permute(1, 2, 0).numpy())
        if preview.height > 0:
            scale = 200 / preview.height
            resampling_module = getattr(Image, "Resampling", Image)
            preview = preview.resize((max(1, int(round(preview.width * scale))), 200), resample=getattr(resampling_module, "BILINEAR", Image.BILINEAR))
        return preview

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
