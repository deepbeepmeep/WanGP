# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from models.help_shared import EDIT_PROMPT, EDIT_PROMPT_SHORT, IMAGE_PROMPT, IMAGE_PROMPT_SHORT, REFERENCE_PROMPT, help_text, specialities


def flux_help(architecture, definition):
    info = "Generate an image from a text description. Use the selected variant's defaults: fast and distilled checkpoints need their matching step count and guidance. More steps are not automatically better for a distilled model."
    compact = "Text-to-image. Use this variant's sampling defaults; distilled and standard checkpoints use different schedules."
    prompt, short_prompt = IMAGE_PROMPT, IMAGE_PROMPT_SHORT
    strengths = []
    sources = ["https://github.com/black-forest-labs/flux"]
    title = "FLUX Image Generation"
    if architecture.startswith("flux2") or architecture == "pi_flux2" or "kontext" in architecture:
        title = "FLUX Reference Editing"
        info += "\n\n### Reference Images\n\nGenerate from text alone, or select **Reference Images** for editing and composition. In main-scene mode (`KI`), the first reference is the scene to edit; subsequent images supply subjects. In subject mode (`I`), describe the desired scene and the role of each selected image. Keep background removal off when the whole reference matters."
        compact += " References: KI uses scene first, then subjects; I uses subjects. Number references in selection order and assign roles."
        prompt += "\n\n" + EDIT_PROMPT + "\n\n" + REFERENCE_PROMPT
        short_prompt += " " + EDIT_PROMPT_SHORT + " " + REFERENCE_PROMPT
        strengths += ["image editing", "identity preservation"]
        if architecture.startswith("flux2") or architecture == "pi_flux2":
            sources = ["https://github.com/black-forest-labs/flux2"]
        if "dreamomni2" in architecture:
            info += "\n\n**DreamOmni2** uses references for unified image editing and subject/style-guided composition. Describe the requested concept concretely and identify what each reference contributes."
            compact += " DreamOmni2: assign subject/style roles explicitly."
            strengths += ["style transfer"]
            sources.append("https://github.com/dvlab-research/DreamOmni2")
    elif architecture == "flux_dev_umo":
        info = "Use **Reference Images** to place one or several subjects in a new composition. Describe their relationships and actions, and identify each subject by its visible appearance. The references supply subjects rather than a first frame to reproduce."
        compact = "UMO: ordered subject references + description of the new composition, relationships and actions."
        prompt += "\n\nFor UMO, prefer a description of the finished scene over an editing command. Start around 768–1024 pixels."
        short_prompt += " UMO: describe the finished scene rather than writing an edit command; start around 768–1024px."
        strengths += ["identity preservation"]
        sources = ["https://github.com/bytedance/UMO"]
    elif architecture == "flux_dev_uso":
        title = "USO Subject and Style Transfer"
        info = "Choose **Reference Images / Style Images** carefully: in subject-and-style mode (`KI`), the first image supplies the subject and the next one or two supply style. In style-only mode (`KIJ`), up to two images supply style. Describe the new composition in the prompt. Reference Relative Size adjusts how much image detail is supplied."
        compact = "USO: KI = subject first, then up to two styles; KIJ = up to two styles only. Prompt describes the new composition."
        prompt += '\n\nFor a new scene, describe the subject acting in that scene. For a style change that retains composition, use an instruction such as `Transform the image into a watercolor style; preserve its layout.` Match the reference framing to the desired result.'
        short_prompt += " USO: descriptive prompt for a new scene; a style-transform instruction for preserving layout. Match reference framing."
        strengths += ["style transfer", "identity preservation"]
        sources = ["https://github.com/bytedance/USO"]
    elif architecture.startswith("flux_chroma"):
        title = "Chroma Image Generation"
        info += "\n\nUse the **Negative Prompt** for unwanted visual content. Chroma Radiance is a separate text-to-image variant; it does not expose Kontext's reference-image editing."
        compact += " Chroma supports a negative prompt. Radiance is text-to-image; do not assume Kontext reference editing."
        sources = ["https://huggingface.co/lodestones/Chroma1-HD", "https://huggingface.co/lodestones/Chroma1-Radiance"]
    if architecture == "flux2_klein_9b_kv":
        title = "FLUX.2 Klein 9B KV"
        info = "Generate images from text or edit with one or more reference images. Start with **4 steps** around **1024 pixels**. This variant is designed for fast editing with several references; it does not accelerate text-only generation.\n\n### Reference Images\n\n- **Main Image Ref** (`KI`): the first image sets the output dimensions through the selected resolution and canvas fitting; later images supply additional subjects or styles.\n- **Image References** (`I`): keep the selected output dimensions and describe how each reference should contribute.\n\nNumber references in selection order, give a precise edit, and state what to preserve. Keep background removal off.\n\n### Practical Limits\n\nMore steps are not automatically better. Negative prompts and guidance controls are not used by this distilled workflow. Exact lettering and complex changes can require another attempt."
        compact = "flux2_klein_9b_kv: text-to-image and ordered image_refs; num_inference_steps=4, guidance_scale=1, prompt_enhancer='' by default. video_prompt_type='KI' lets the first reference determine the fitted output dimensions; 'I' retains resolution. Reference K/V are cached within each generation only; no reference-cache benefit for text-only calls. No negative prompt or NAG. Keep remove_background_images_ref=0."
        sources = ["https://huggingface.co/black-forest-labs/FLUX.2-klein-9b-kv", "https://github.com/black-forest-labs/flux2/blob/main/docs/flux2_klein_kv_cache.md"]
    if definition.get("inpaint_support"):
        info += "\n\n### Local Edits\n\n**Image Inpainting** combines a source image, mask and prompt. Describe what should replace the masked region. Where offered, **Masked Denoising** can reuse source detail, while **LanPaint** spends more time refining the edit."
        compact += " Inpainting: source + mask + replacement description; LanPaint costs more time."
        strengths += ["image inpainting"]
    result = help_text(title, info, prompt, compact, short_prompt, sources)
    result["specialities"] = specialities(*strengths)
    return result
