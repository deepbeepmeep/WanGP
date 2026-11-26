# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""WanGP integration for the inclusionAI Ming Image family.

The checkpoint URLs describe the prepared DeepBeepMeep layout.  The
upstream code and checkpoint provenance are recorded in README.md.
"""

from pathlib import Path

import torch

from shared.utils import files_locator as fl
from shared.utils.hf import build_hf_url

from .prompt_enhancers import CLASSIC_IMAGE, CLASSIC_TEXT, JSON_IMAGE, JSON_TEXT, LAYER_PLAN


REPO = "DeepBeepMeep/MingImage"
PROJECT = "ming_image"
ENCODER = "BailingMM2-Ming-Image"
ENCODER_CORE = "BailingMM2-Ming-Image-Core"
SHARED_PROJECT = "ming_image_shared"
ARCHITECTURE = "ming_image_0_1_design"
TRANSFORMER = "Ming-Image-0.1-Design"
LAYER_REPO = REPO
LAYER_PROJECT = "ming_image_layer"
LAYER_ARCHITECTURE = "ming_image_0_1_design_layer"
LAYER_TRANSFORMER = "Ming-Image-0.1-Design-Layer"
TOKENIZER_FILES = (
    "config.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

STRUCTURED_PROMPT_EXAMPLE = """{
  "canvas_settings": {
    "aspect_ratio": "1:1, 2048 × 2048 px",
    "ambient_lighting": "Soft, even studio light",
    "image_style": "Clean modern editorial poster"
  },
  "layers": [
    {
      "description": "Full-canvas deep blue background with subtle paper texture.",
      "coordinates": "cx: 0.500, cy: 0.500, w: 1.000, h: 1.000",
      "hierarchy_and_relation": "Backmost layer covering the canvas.",
      "color_specs": ["#14213D"]
    },
    {
      "description": "Large centered white title \\"OPEN HOUSE\\" in bold sans-serif lettering.",
      "coordinates": "cx: 0.500, cy: 0.250, w: 0.700, h: 0.160",
      "hierarchy_and_relation": "Top layer centered above the main artwork.",
      "color_specs": ["#FFFFFF"]
    }
  ]
}"""


def conditioner_file(model_def):
    stem = "conditioning_0_2" if model_def.get("ming_release") == "0.2" else "conditioning"
    return f"{stem}_{model_def.get('ming_encoder_variant', 'bf16')}.safetensors"


class family_handler:
    @staticmethod
    def resolve_runtime_model_def(model_def, runtime_context):
        variant = "int8_convrot" if runtime_context.get("text_encoder_quantization") == "int8" else "bf16"
        return {**model_def, "ming_encoder_variant": variant}

    @staticmethod
    def render_prompt_helper(model_type, model_def, prompt_id, popup_id, prompt_elem_id, resolution_elem_id):
        if model_type == LAYER_ARCHITECTURE:
            return ""
        from .prompt_helper import render_prompt_helper

        return render_prompt_helper(model_type, model_def, prompt_id, popup_id, prompt_elem_id, resolution_elem_id)

    @staticmethod
    def get_prompt_helper_css():
        from .prompt_helper import get_prompt_helper_css

        return get_prompt_helper_css()

    @staticmethod
    def get_prompt_helper_javascript():
        from .prompt_helper import get_prompt_helper_javascript

        return get_prompt_helper_javascript()

    @staticmethod
    def query_supported_types():
        return [ARCHITECTURE, LAYER_ARCHITECTURE]

    @staticmethod
    def query_model_family():
        return "ming_image"

    @staticmethod
    def query_family_infos():
        return {"ming_image": (1160, "Ming Image")}

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def get_lora_dir(base_model_type):
        return LAYER_PROJECT if base_model_type == LAYER_ARCHITECTURE else PROJECT

    @staticmethod
    def get_rgb_factors(base_model_type):
        from .rgb_factors import RGB_BIAS, RGB_FACTORS

        return RGB_FACTORS, RGB_BIAS

    @staticmethod
    def query_model_def(base_model_type, model_def):
        if base_model_type == LAYER_ARCHITECTURE:
            return {
                "image_outputs": True,
                "device_explicit": True,
                "skip_prompt_template": True,
                "embedded_guidance": False,
                "guidance_max_phases": 1,
                "NAG": False,
                "profiles_dir": [LAYER_ARCHITECTURE],
                "resolution_budget": "1024x1024",
                "text_encoder_folder": ENCODER,
                "text_encoder_URLs": [
                    build_hf_url(REPO, ENCODER, ENCODER_CORE + suffix)
                    for suffix in ("_bf16.safetensors", "_int8_convrot.safetensors")
                ],
                "image_ref_choices": {
                    "choices": [("Decompose the Reference Image", "I")],
                    "letters_filter": "I",
                    "default": "I",
                },
                "one_image_ref_needed": True,
                "one_image_ref_only": True,
                "no_background_removal": True,
                "prompt_enhancer_def": {
                    "selection": ["TI"],
                    "labels": {"TI": "Build a Layer Plan from Text Prompt and {image_inputs}"},
                    "default": "",
                },
                "image_prompt_enhancer_instructions": LAYER_PLAN,
                "image_prompt_enhancer_max_tokens": 1024,
                "specialities": [{
                    "name": "layer decomposition",
                    "aliases": ["transparent design layers"],
                    "description": "Separates one flattened design into ordered transparent PNG layers.",
                }, {"name": "alpha output", "aliases": ["RGBA", "transparent layers"], "description": "Separate a supplied design into transparent PNG layers."}],
                "infos": (
                    "**Ming Image Design-Layer** turns a finished poster, infographic, card, or similar design "
                    "into separate transparent image layers. Select one Reference Image, then describe the pieces "
                    "you want to separate in Text Prompt, starting with the frontmost. For example, put the words "
                    "on one layer, the panel behind them on another, the illustration on a third, and the "
                    "background last. Describe only elements that are visible in the image.\n\n"
                    "WanGP adds every layer to the gallery and saves them together in a ZIP. You can hide or "
                    "rearrange the layers in an image editor; text remains part of the image rather than editable "
                    "type. Start with 12 steps, guidance 2, and 1024x1024 for a detailed square result. "
                    "Choose 512x512 when speed matters more than detail."
                ),
                "prompt_infos": (
                    "Write `Decompose this image into N layers with the following specifications:` followed by "
                    "`Number of layers: N` and one `Layer i:` line for each layer. Order layers front to back; "
                    "put readable text on a front layer, any card or panel directly behind it on its own layer, "
                    "the main subject on its own layer, and the background last. Quote actual visible words "
                    "exactly and do not invent unreadable lettering. For a quick request, `Decompose this image "
                    "into 4 layers.` also works.\n\n"
                    "**Example:**\n```text\nDecompose this image into 4 layers with the following specifications:\n"
                    "Number of layers: 4\nLayer 1: Exact visible headline and subtitle on transparent pixels.\n"
                    "Layer 2: The colored title card directly behind the text.\n"
                    "Layer 3: The main foreground illustration.\n"
                    "Layer 4: The complete background and remaining supporting shapes.\n```"
                ),
                "deepy_infos": (
                    "Ming Image Design-Layer separates one supplied design image into transparent layers. "
                    "Ask for the visible pieces from front to back, with the background last. Each layer appears "
                    "in the gallery and in a ZIP. Lettering remains image pixels. Start with 12 steps, guidance 2, "
                    "and 1024x1024; choose 512x512 for a faster draft."
                ),
                "deepy_prompt_infos": (
                    "Ask for `Decompose this image into N layers` or supply `Number of layers: N` and exact "
                    "`Layer 1:` through `Layer N:` descriptions. The first layer is topmost and the last is "
                    "the background. Ground every described object and quoted string in the reference image. "
                    "Place text, its supporting panel, the main subject, and the background on separate layers "
                    "when present."
                ),
            }
        return {
            "image_outputs": True,
            "device_explicit": True,
            "skip_prompt_template": True,
            "prompt_helper_popup_dims": [86, 94],
            "prompt_enhancer_def": {
                "selection": ["T", "TI", "T1", "TI1"],
                "labels": {
                    "T": "A Classic Image Prompt using existing Text Prompt",
                    "TI": "A Classic Image Edit Prompt using existing Text Prompt and {image_inputs}",
                    "T1": "A Ming JSON Image Prompt using existing Text Prompt",
                    "TI1": "A Ming JSON Image Edit Prompt using existing Text Prompt and {image_inputs}",
                },
                "default": "",
            },
            "text_prompt_enhancer_instructions": CLASSIC_TEXT,
            "image_prompt_enhancer_instructions": CLASSIC_IMAGE,
            "text_prompt_enhancer_instructions1": JSON_TEXT,
            "image_prompt_enhancer_instructions1": JSON_IMAGE,
            "text_prompt_enhancer_max_tokens": 1024,
            "image_prompt_enhancer_max_tokens": 1024,
            "text_prompt_enhancer_max_tokens1": 4096,
            "image_prompt_enhancer_max_tokens1": 4096,
            "embedded_guidance": False,
            "NAG": False,
            "profiles_dir": [ARCHITECTURE],
            "resolution_budget": "2048x2048",
            "text_encoder_folder": ENCODER,
            "text_encoder_URLs": [
                build_hf_url(REPO, ENCODER, ENCODER_CORE + suffix)
                for suffix in ("_bf16.safetensors", "_int8_convrot.safetensors")
            ],
            "image_ref_choices": {
                "choices": [
                    ("None", ""),
                    ("First Reference Defines Output Dimensions (Main Image Ref)", "KI"),
                    ("Edit the Reference Image at the Selected Size", "I"),
                ],
                "letters_filter": "KI",
            },
            "one_image_ref_only": True,
            "no_background_removal": True,
            "custom_settings": [{
                "id": "rgba",
                "name": "RGBA",
                "label": "RGBA Output",
                "type": "dropdown",
                "default": "Disabled",
                "choices": [("Disabled", "Disabled"), ("Enabled", "Enabled")],
                "info": "Preserve the model-generated alpha channel in PNG. Transparency depends on the prompt and is not guaranteed.",
            }],
            "specialities": [
                {"name": "alpha output", "aliases": ["RGBA", "transparent background"], "description": "Set custom_settings.rgba to Enabled and request a transparent background to preserve model-generated alpha in PNG. Transparency depends on the prompt."},
                {"name": "visual design", "aliases": ["UI design", "poster design"], "description": "Creates text-rich visual designs, posters, and interface mockups."},
            ],
            "infos": (
                "**Ming Image Design** creates posters, infographics, interface mockups, and other visual designs "
                "from text. You can also select one Reference Image and describe a change while saying what to "
                "keep. Main Image Ref takes the canvas dimensions from the reference. Edit at the Selected Size uses "
                "the chosen canvas. For a simple image, write a normal prompt. For a precise layout with titles and labels, "
                "use the Ming JSON prompt enhancer or arrange the design in Prompt Helper. Put any words that "
                "must appear exactly in quotes.\n\n"
                "Start with 12 steps and guidance 1. The default 2048x2048 size gives more detail; choose "
                "1024x1024 to save time and memory. Reference edits work best at 1024x1024; larger edit "
                "outputs are resized. To request transparency, enable RGBA Output and ask for a transparent "
                "background, though the result may still be opaque. For separating an existing design into "
                "individual layers, choose Ming Image Design-Layer instead."
            ),
            "prompt_infos": (
                "For text-to-image designs with exact layout, paste one JSON object into Prompt. It has exactly two top-level "
                "keys: `canvas_settings` (`aspect_ratio`, `ambient_lighting`, `image_style`) and `layers` (backmost to "
                "frontmost). Each layer has `description`, `coordinates`, `hierarchy_and_relation`, and `color_specs` "
                "(an array of hex colors). Write `coordinates` as one normalized string: "
                "`cx: 0.500, cy: 0.500, w: 1.000, h: 1.000`. Keep each box inside the canvas, describe one complete "
                "visible group per layer, and quote every exact rendered string once in its owning layer's description. "
                "Match the JSON aspect ratio to the selected WanGP resolution. Use Prompt Helper to draw or edit layer boxes; "
                "it updates the size field from the selected resolution.\n\n"
                "**Example:**\n```json\n" + STRUCTURED_PROMPT_EXAMPLE + "\n```\n\n"
                "Plain-language prompts also work. For image editing, describe the change and what to preserve, for "
                "example: Change the headline to \"OPEN STUDIO\" while keeping the layout and colors. For transparency, "
                "enable RGBA Output and begin a plain-language prompt with `RGBA, 4-channel, transparent background`."
            ),
            "deepy_infos": (
                "Ming Image Design creates posters, infographics, and other visual designs from text or edits "
                "one reference image. Use Main Image Ref to take its dimensions or Edit at the Selected Size to use "
                "the chosen canvas. Use a plain prompt for simple work or structured JSON for precise "
                "layouts. Quote exact visible words. For an edit, name the change and what "
                "to preserve. Start with 12 steps and guidance 1; 2048x2048 gives more detail, while "
                "1024x1024 is faster and suits edits. For transparency, enable RGBA Output and request a "
                "transparent background. Use Design-Layer to separate an existing image into layers."
            ),
            "deepy_prompt_infos": (
                "For a text-to-image design needing exact layout or lettering, write one JSON object with exactly "
                "`canvas_settings` (`aspect_ratio`, `ambient_lighting`, `image_style`) and `layers` in back-to-front order. "
                "Each visible semantic layer contains exactly `description`, `coordinates`, `hierarchy_and_relation`, "
                "and `color_specs` (hex-color array). `coordinates` is one normalized string in the form "
                "`cx: 0.500, cy: 0.500, w: 1.000, h: 1.000`; boxes stay inside the canvas. Keep people and objects "
                "intact, use the fewest groups that preserve the layout, and put each exact visible string once, quoted "
                "verbatim, in its owning description. Use relationships for alignment, containment, stacking and "
                "occlusion, with no duplicate text or invisible layers. Match `aspect_ratio` to the chosen WanGP output "
                "size. For simple requests, a concrete plain-language prompt is valid. For reference editing, use a "
                "specific plain-language change plus what must remain unchanged. Do not request layer decomposition."
            ),
        }

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        variant = (model_def or {}).get("ming_encoder_variant", "bf16")
        vision_file = f"vision_encoder_{variant}.safetensors"
        conditioning = conditioner_file(model_def or {})
        if base_model_type == LAYER_ARCHITECTURE:
            return [
                {"repoId": LAYER_REPO, "sourceFolderList": [LAYER_PROJECT], "fileList": [[
                    "transformer_config.json", "connector_config.json", "mlp_config.json",
                    "scheduler_config.json", conditioning,
                ]]},
                {"repoId": REPO, "sourceFolderList": [SHARED_PROJECT], "fileList": [[vision_file]]},
                {"repoId": REPO, "sourceFolderList": [ENCODER], "fileList": [list(TOKENIZER_FILES)]},
                {"repoId": REPO, "sourceFolderList": [PROJECT], "fileList": [[
                    "vae_config.json", "vae.safetensors",
                ]]},
            ]
        return [
            {"repoId": REPO, "sourceFolderList": [PROJECT], "fileList": [[
                "transformer_config.json", "connector_config.json", "mlp_config.json",
                "vae_config.json", "vae.safetensors", "scheduler_config.json", conditioning,
            ]]},
            {"repoId": REPO, "sourceFolderList": [SHARED_PROJECT], "fileList": [[vision_file]]},
            {"repoId": REPO, "sourceFolderList": [ENCODER], "fileList": [list(TOKENIZER_FILES)]},
        ]

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        if base_model_type == LAYER_ARCHITECTURE:
            ui_defaults.update(image_mode=1, resolution="1024x1024",
                               num_inference_steps=12, guidance_scale=2.0,
                               batch_size=1, prompt_enhancer="", video_prompt_type="I",
                               remove_background_images_ref=0)
            return
        ui_defaults.update(image_mode=1, resolution="2048x2048",
                           num_inference_steps=12, guidance_scale=1.0,
                           batch_size=1, prompt_enhancer="", video_prompt_type="",
                           remove_background_images_ref=0)

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def,
                   text_encoder_filename=None, save_quantized=False,
                   quantizeTransformer=False, **kwargs):
        from .ming_pipeline import load_components

        return load_components(
            transformer_filename=model_filename[0],
            text_encoder_filename=text_encoder_filename,
            model_type=model_type,
            base_model_type=base_model_type,
            save_quantized=save_quantized,
            quantize_transformer=quantizeTransformer,
            model_def=model_def,
        )
