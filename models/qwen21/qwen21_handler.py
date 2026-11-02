# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import os
from functools import lru_cache
from urllib.parse import urlsplit

import torch

from shared.utils import files_locator as fl
from shared.utils.hf import build_hf_url
from shared.utils.gguf_mapping import has_standard_gguf_tensor_names, remap_state_dict_triplet
from .enhancer import GENERATION, EDITING


ENCODER = "Qwen3-VL-8B-Instruct"
PROJECT = "qwen_image_21"
REPO = "DeepBeepMeep/Qwen_image_2"
ENCODER_REPO = "DeepBeepMeep/Ideogram4"
VISION_FILE = ENCODER + "_vision_bf16.safetensors"
PROCESSOR_FILES = ["config_legacy.json", "tokenizer_legacy.json", "tokenizer_config_legacy.json", "preprocessor_config.json", "chat_template.jinja", "merges.txt", "vocab.json", "added_tokens.json", "special_tokens_map.json", "video_preprocessor_config.json"]


@lru_cache(maxsize=1)
def encoder_gguf_name_map():
    import json
    from accelerate import init_empty_weights
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from transformers.modeling_gguf_pytorch_utils import get_gguf_hf_weights_map

    with open(fl.locate_file(os.path.join(ENCODER, "config_legacy.json")), encoding="utf-8") as reader:
        text_config = json.load(reader)["text_config"]
    # Qwen3-VL's text layers use Qwen3's GGUF names and tensor layout.
    config = Qwen3Config(**{key: text_config[key] for key in (
        "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
        "num_attention_heads", "num_key_value_heads", "head_dim")})
    with torch.device("cpu"), init_empty_weights():
        model = Qwen3ForCausalLM(config)
    return {name: target.replace("model.", "model.language_model.", 1)
            for name, target in get_gguf_hf_weights_map(model).items()}


def encoder_state_dict(state_dict, quantization_map=None, tied_weights_map=None):
    """Accept GGUF, shared encoder-only and conditional-generation layouts."""
    if has_standard_gguf_tensor_names(state_dict):
        state_dict, quantization_map, tied_weights_map = remap_state_dict_triplet(
            state_dict, quantization_map, tied_weights_map, encoder_gguf_name_map())

    def model_name(key):
        key = key.removeprefix("model.")
        if key.partition(".")[0] in ("embed_tokens", "layers", "norm"):
            key = "language_model." + key
        return "model." + key

    def remap(mapping):
        if mapping is None:
            return None
        return {model_name(key): ([model_name(item) for item in value if item != "lm_head.weight"] if isinstance(value, list) else value)
                for key, value in mapping.items() if key != "lm_head.weight" and key != "lm_head"}
    return remap(state_dict), remap(quantization_map), remap(tied_weights_map)


class family_handler:
    @staticmethod
    def query_supported_types():
        return ["qwen_image_21_7B"]

    @staticmethod
    def query_model_family():
        return "qwen"

    @staticmethod
    def query_family_infos():
        return {"qwen": (1110, "Qwen")}

    @staticmethod
    def get_lora_dir(base_model_type):
        return "qwen21"

    @staticmethod
    def query_model_def(base_model_type, model_def):
        definition = {
            "image_outputs": True,
            "device_explicit": True,
            "cuda_graphs": ["transformer"],
            "resolution_budget": "2400x1792",
            "custom_settings": [{
                "id": "qwen21_kv_cache",
                "name": "KV Cache",
                "label": "KV Cache",
                "type": "dropdown",
                "default": "Disabled",
                "choices": [("Disabled (Slower but lower VRAM/RAM)", "Disabled"), ("Enabled (Fastest, best quality but requires more VRAM/RAM)", "Enabled"), ("INT8 (Fastest, about half the memory of Enabled)", "INT8"), ("INT8 in RAM (Faster, lowest VRAM)", "INT8_RAM")],
                "info": "Disabled recomputes conditioning each denoising step without retaining per-layer K/V. Enabled caches conditioning K/V for faster denoising, in VRAM as far as it fits beside the model; when VRAM runs short, the layers that do not fit stay in system RAM and stream to the GPU as they run. INT8 caches them in about half the memory: with SageAttention the images match Enabled closely, with other attention modes they differ very slightly. INT8 in RAM keeps them in system RAM and streams each layer to the GPU as it runs, for the lowest VRAM. The CPU text-embedding cache remains active in every mode.",
            }, {
                "id": "rgba",
                "name": "RGBA",
                "label": "RGBA",
                "type": "dropdown",
                "default": "Disabled",
                "choices": [("Disabled", "Disabled"), ("Enabled", "Enabled")],
                "info": "Disabled saves RGB in your selected image format. Enabled preserves the generated alpha channel and saves RGBA PNG. Request transparency in the prompt when needed.",
            }],
            "specialities": [
                {"name": "alpha output", "aliases": ["RGBA", "transparent background"], "description": "Set custom_settings.rgba to Enabled and request an RGBA image with transparent background to preserve generated alpha in PNG. Turbo adapter RGBA quality is not validated."},
            ],
            "prompt_enhancer_def": {"selection": ["T", "TI", "T1", "TI1"], "labels": {"T": "A New Image Description using existing Text Prompt", "TI": "An Image Edit using existing Text Prompt and {image_inputs}", "T1": "An Editing Instruction using existing Text Prompt", "TI1": "A New Composition using existing Text Prompt and {image_inputs}"}, "default": ""},
            "text_prompt_enhancer_instructions": GENERATION,
            "image_prompt_enhancer_instructions": EDITING,
            "text_prompt_enhancer_instructions1": EDITING,
            "image_prompt_enhancer_instructions1": EDITING,
            "text_prompt_enhancer_max_tokens": 768,
            "image_prompt_enhancer_max_tokens": 768,
            "text_prompt_enhancer_max_tokens1": 768,
            "image_prompt_enhancer_max_tokens1": 768,
            "vae_block_size": 32,
            "sample_solvers": [("FlowMatch Euler", "default"), ("Qwen Image 2.1 Turbo", "qwen21_turbo"), ("Viggle Turbo", "viggle_v02"), ("Pruna", "pruna")],
            "guidance_max_phases": 1,
            "embedded_guidance": False,
            "NAG": False,
            "fit_into_canvas_image_refs": 0,
            "profiles_dir": ["qwen21"],
            "inpaint_support": True,
            "inpaint_video_prompt_type": "VAG",
            "inpaint_with_image_ref": True,
            "image_ref_inpaint": True,
            "mask_strength_always_enabled": True,
            "inpaint_color": "FF0000",
            "guide_inpaint_color": "FF0000",
            "video_guide_outpainting": [1, 2],
            "outpainting_quantize_margins": 32,
            "guide_preprocessing": {"selection": ["", "PV", "DV", "SV", "CV", "V"], "labels": {"V": "Control Image"}},
            "mask_preprocessing": {"selection": ["", "A"], "visible": True},
            "model_modes": {"choices": [("Masked Denoising", 0), ("LanPaint (2 steps)", 2), ("LanPaint (5 steps)", 3), ("LanPaint (10 steps)", 4), ("LanPaint (15 steps)", 5)], "default": 0, "label": "Inpainting Method", "image_modes": [2]},
            "text_encoder_folder": ENCODER,
            "text_encoder_URLs": [build_hf_url(ENCODER_REPO, ENCODER, ENCODER + suffix) for suffix in ("_bf16.safetensors", "_int8_convrot.safetensors")],
            "image_ref_choices": {"choices": [("None", ""), ("First Image Is the Main Subject or Landscape", "KI"), ("Reference Images Are People or Objects", "I")], "letters_filter": "KI"},
            "infos": "**Qwen Image 2.1 (7B)** generates images from text and edits images using up to **10 reference images**. Select reference images in the order you will describe them. Use a control image and a mask for a local edit. For outpainting, select Control Image under Control Image Process (or select a main reference), then enable Spatial Outpainting on Control Image. The same checkbox is available in Image Inpainting. Outpainting replaces red margins with a continuation of the scene. When combined with inpainting, denoising strength applies only to the painted interior. Masked Denoising offers strength controls; LanPaint spends additional time refining masked regions.\n\nFor the base model, start with **40 steps**, **guidance 4** and a square image around **1024 pixels**. Qwen Image 2.1 is native 2K: its recommended sizes are 2048x2048, 2400x1792, 2528x1696, 2752x1536 and their portrait versions, up to about 4.3 MP. Larger images require more memory and time; VAE decoding presets use 1024px tiles for 16GB+, 512px for 8GB+, and 256px for 6GB+, with 25% overlap. Auto chooses by GPU capacity. KV Cache defaults to Disabled to reduce VRAM; enable it for faster denoising, or choose INT8 for the same speedup in about half the memory. The cache stays in VRAM as far as it fits; the rest stays in system RAM and streams to the GPU. The CPU text-embedding cache stays active. The VAE runs in BF16. Text-rendering quality is moderate: prefer short text and proofread spelling. Dense infographics are not a recommended use. RGBA defaults to Disabled: outputs are RGB and use your selected image format. For transparent cutouts or stickers, set RGBA to Enabled and explicitly request an RGBA image, an alpha channel and a transparent background; WanGP saves RGBA output as PNG to preserve transparency. This uses model-generated alpha, not a background-removal toggle. Prompt enhancement is optional and disabled by default.\n\nNAG controls are temporarily hidden. Use guidance above 1 for standard CFG and negative prompts.\n\nViggle Turbo profiles use LoRA strength 1 and guidance 1: v0.1 runs 4 steps, v0.2 runs 5, and the recommended v0.2.1 runs 6. Each profile selects the Viggle Turbo scheduler, which applies the matching sigma nodes and disables terminal shift. The built-in Deepy image-generation and editing recipes use the v0.2.1 six-step settings. Viggle calls these adapters previews; complex edits remain weaker than the base model, and masks, RGBA, more than three references and 2K output have not been validated for them.\n\nPruna v0.1 has separate 5-step and 8-step profiles; 8 steps is the recommended starting point, while 5 steps is faster with lower visual quality. Both use guidance 1, LoRA strength 1, and the Pruna scheduler with the matching sigma nodes and no dynamic or terminal shift. Pruna trained these adapters around 1024x1024 for text-to-image and edits with up to three references; higher resolutions and more references have not been validated. The built-in Deepy Pruna recipes use the recommended 8-step settings.\n\nUse **Qwen Image 2.1 7B LoRAs** in the dedicated LoRA folder. Older Qwen Image/Edit 20B adapters are incompatible even when their filenames use the same convention. Diffusers/PEFT and Kohya naming are supported.",
            "prompt_infos": "Describe the finished image, including the subject, composition, lighting and style. Put any exact visible text in quotes. Text-rendering quality is moderate. Prefer short quoted text and proofread spelling and layout; avoid dense infographics.\n\n**Generation:** A red ceramic teapot on a wooden table, soft window light, detailed product photograph.\n\n**Editing:** Change the cat's fur to orange. Keep its pose and the green background unchanged.\n\n**Multiple references:** Put the person from `<image1>` in the jacket from `<image2>`. Keep the person's face and pose. Number references in their selected order.\n\n**Masked editing:** Select the region to replace, then describe what belongs there: Replace the background with a bright blue sky.\n\n**Transparency:** Set RGBA to Enabled, then prompt: This is an RGBA image with transparency. A cartoon dragon sticker. The image has alpha channel and the background is transparent. WanGP saves RGBA output as PNG automatically to retain alpha; JPEG cannot store transparency.\n\nUse the negative prompt to describe unwanted content when guidance is above 1. Keep editing instructions specific about both the requested change and details to preserve.",
            "deepy_infos": "Generate images from text or edit with up to 10 ordered reference images. Use the main/control image plus a mask for local edits; outpainting extends the canvas. Outpainting uses red canvas margins; combined inpainting applies denoising strength only to painted interior pixels. Base model: start at 40 steps, guidance 4 and roughly 1024 pixels. Viggle Turbo profiles use guidance 1, LoRA strength 1 and the Viggle Turbo scheduler: v0.1 uses 4 steps, v0.2 uses 5, and the recommended v0.2.1 uses 6. The built-in Deepy image-generation and editing recipes use the recommended v0.2.1 six-step settings. The scheduler selects the matching nodes and disables terminal shift. Viggle adapters are previews best suited to text-to-image and simple edits with up to 3 references. Complex editing, masks, RGBA and 2K output are not validated for them. Pruna v0.1 profiles use a separate Pruna scheduler at 5 or 8 steps, guidance 1 and LoRA strength 1; 8 steps is recommended for better quality and is available in Deepy generation and editing recipes. Pruna trained at roughly 1024 pixels with up to three references; higher resolutions and more references are outside its training coverage. LanPaint adds refinement steps for masked editing. KV Cache defaults to Disabled for lower VRAM; custom_settings.qwen21_kv_cache='Enabled' speeds up denoising, 'INT8' gives the same speedup in about half the memory (near-identical images with SageAttention); layers that do not fit in VRAM stay in RAM and stream, 'INT8_RAM' always streams them (lowest VRAM). VAE Auto tiles larger images; The VAE runs in BF16. Text-rendering quality is moderate, best with short copy; proofread spelling and layout. Do not recommend this model for dense infographics. RGBA defaults to Disabled (RGB in the selected image format). Enable RGBA and explicitly request transparency to preserve alpha in PNG output. Prompt enhancement defaults off. Only Qwen Image 2.1 7B LoRAs are compatible; old Qwen 20B adapters are not.",
            "deepy_prompt_infos": "For generation, describe subject, composition, lighting, style and quoted visible text. Text-rendering quality is moderate: prefer short quoted copy and proofread it; avoid dense infographics. For editing, specify what changes and what stays: 'Change the cat fur to orange; keep its pose and background.' Refer to ordered references as `<image1>`, `<image2>`, etc. For masked edits, describe the desired replacement. For transparency, set custom_settings.rgba to Enabled and explicitly request an RGBA image, alpha channel and transparent background. Negative prompts apply with guidance above 1 (CFG). NAG controls are temporarily hidden.",
        }
        if model_def.get("qwen21_variant") == "turbo":
            definition.update(
                sample_solvers=[("Qwen Image 2.1 Turbo", "qwen21_turbo"), ("FlowMatch Euler", "default")],
                profiles_dir=[],
                infos="**Qwen Image 2.1 Turbo (7B)** generates images from text and edits images using ordered references. This is Qwen's full distilled checkpoint; no acceleration LoRA is needed. Start with **8 steps**, **guidance 1** and the **Qwen Image 2.1 Turbo** scheduler. Its eight released sigma values are fixed, without dynamic or terminal shifting. FlowMatch Euler remains available for experiments, but vanilla sigmas and other step counts have not been quality-validated for this checkpoint. Negative prompts have no effect at guidance 1. Do not apply the base model's Viggle or Pruna acceleration profiles.\n\nDescribe the desired result, quote exact visible text, and refer to ordered inputs as `<image1>`, `<image2>`, etc. The shared pipeline accepts up to 10 reference images, control-image masks and outpainting. Those inherited editing controls still need Turbo-specific validation. Upstream examples use roughly 2K output; 1024 pixels is a practical smaller starting point. KV Cache defaults to Disabled to reduce memory; enable it to reuse text and reference conditioning across steps, or choose INT8 to do so in about half the memory; the cache stays in VRAM as far as it fits and the rest streams from system RAM. VAE tiling and the shared BF16 VAE remain available.\n\nRGBA defaults to Disabled. To preserve generated transparency, enable RGBA and request an RGBA image with an alpha channel and transparent background; output is saved as PNG. Proofread rendered text. Prompt enhancement remains optional and disabled by default. Qwen Image 2.1 7B adapters share the tensor layout; older Qwen 20B LoRAs are incompatible.",
                deepy_infos="Qwen's full distilled Qwen Image 2.1 Turbo 7B checkpoint, for text-to-image and reference editing. Default: 8 steps, guidance 1, sample_solver=qwen21_turbo. The named Turbo scheduler uses the released fixed sigmas, with no dynamic/terminal shifting; it requires 8 steps. No acceleration LoRA is needed. Do not use the base model's Viggle/Pruna profiles. Vanilla FlowMatch Euler remains experimental for this checkpoint. Negative prompts are inactive at CFG 1. The shared pipeline accepts up to 10 ordered references, masks and outpainting; inherited controls need Turbo-specific quality validation. Upstream examples use roughly 2K; defaults start at 1024. KV Cache (qwen21_kv_cache Enabled, or INT8 at about half the memory; layers that do not fit in VRAM stream from RAM, INT8_RAM always streams) and RGBA default off. Enable RGBA and explicitly request transparency to retain alpha in PNG. Prompt enhancement defaults off; proofread rendered text.",
            )
        return definition

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        encoder_files = PROCESSOR_FILES.copy()
        if model_def is not None and any(urlsplit(url.split("|", 1)[0]).path.lower().endswith(".gguf")
                                         for url in model_def.get("text_encoder_URLs", [])):
            encoder_files.append(VISION_FILE)
        return [{"repoId": REPO, "sourceFolderList": [PROJECT], "fileList": [["qwen_image_21_vae.safetensors", "vae_config.json", "scheduler_config.json"]]},
                {"repoId": ENCODER_REPO, "sourceFolderList": [ENCODER], "fileList": [encoder_files]}]

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update(image_mode=1, num_inference_steps=40, guidance_scale=4.0, sample_solver="default", video_prompt_type="", prompt_enhancer="", remove_background_images_ref=0, model_mode=0, denoising_strength=1.0)
        if model_def.get("qwen21_variant") == "turbo":
            ui_defaults.update(num_inference_steps=8, guidance_scale=1.0, sample_solver="qwen21_turbo")

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        custom = ui_defaults.get("custom_settings")
        if isinstance(custom, dict) and "outpainting_borders" in custom:
            ui_defaults["custom_settings"] = {key: value for key, value in custom.items() if key != "outpainting_borders"}
        # Early defaults inherited video mode from the shared settings. This
        # image-only architecture must retain either generation or inpainting.
        if ui_defaults.get("image_mode") in (None, 0, "0"):
            ui_defaults["image_mode"] = 1

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, text_encoder_filename=None, save_quantized=False, quantizeTransformer=False, **kwargs):
        from mmgp import offload, quant_router, runtime
        from .text_encoder import Qwen3VLForConditionalGeneration
        from .pipeline import Qwen21Pipeline, load_processor
        from .transformer import QwenImage21Transformer2DModel
        from .vae import AutoencoderKLQwenImage21

        config = os.path.join(os.path.dirname(__file__), "configs", "qwen_image_21_7B.json")
        transformer = offload.fast_load_transformers_model(model_filename[0], writable_tensors=False, modelClass=QwenImage21Transformer2DModel, defaultConfigPath=config, do_quantize=quantizeTransformer and not save_quantized)
        if save_quantized:
            from wgp import save_quantized_model
            save_quantized_model(transformer, model_type, model_filename[0], torch.bfloat16, config)
        processor_paths = {name: fl.locate_file(os.path.join(ENCODER, name)) for name in PROCESSOR_FILES}
        processor = load_processor(os.path.dirname(processor_paths["tokenizer_legacy.json"]), processor_paths["preprocessor_config.json"],
                                   tokenizer_path=processor_paths["tokenizer_legacy.json"], tokenizer_config_path=processor_paths["tokenizer_config_legacy.json"])
        import json
        from accelerate import init_empty_weights
        with open(processor_paths["config_legacy.json"], encoding="utf-8") as reader:
            encoder_config = json.load(reader)
        with torch.device("cpu"), init_empty_weights():
            text_encoder = Qwen3VLForConditionalGeneration(encoder_config)
        text_encoder._config = encoder_config
        encoder_files = [text_encoder_filename]
        encoder_keys, _ = quant_router.load_metadata_state_dict(text_encoder_filename)
        if not any(key.startswith(("visual.", "model.visual.")) for key in encoder_keys):
            encoder_files.append(fl.locate_file(os.path.join(ENCODER, VISION_FILE)))
        offload.load_model_data(text_encoder, encoder_files, writable_tensors=False, preprocess_sd=encoder_state_dict)
        runtime.lookup_on_cpu(text_encoder.model.language_model.embed_tokens)  # the 1.2 GB token table stays in RAM: mmgp looks up the prompt's rows on the CPU
        vae = offload.fast_load_transformers_model(fl.locate_file(PROJECT + "/qwen_image_21_vae.safetensors"), writable_tensors=False, modelClass=AutoencoderKLQwenImage21, defaultConfigPath=fl.locate_file(PROJECT + "/vae_config.json"), default_dtype=torch.float32)
        text_encoder.eval().requires_grad_(False)
        pipe = {"transformer": transformer, "text_encoder": text_encoder.model.language_model,
                "vision_encoder": text_encoder.model.visual, "vae": vae}
        for component in pipe.values():
            component.eval().requires_grad_(False)
            component._convertWeightsFloatTo = None
            component._model_dtype = next(component.parameters()).dtype
            for module in component.modules():
                module._lock_dtype = None
        text_encoder._model_dtype = pipe["text_encoder"]._model_dtype
        if kwargs.get("VAE_dtype", torch.float32) != torch.float32:
            # User-requested 16-bit execution: BF16 is validated; FP16
            # overflows on real image latents. MMGP owns the conversion.
            vae._convertWeightsFloatTo = torch.bfloat16
            vae._model_dtype = torch.bfloat16
            for module in vae.modules():
                del module._lock_dtype
        vae.upsampling_set = None
        pipeline = Qwen21Pipeline(transformer, text_encoder, vae, processor, fl.locate_file(PROJECT + "/scheduler_config.json"))
        return pipeline, {**pipe, "tokenizer": processor.tokenizer}
