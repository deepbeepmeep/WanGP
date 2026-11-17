# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import json

RMBG3_INSTRUCTION = (
    "Generate a detailed grayscale alpha matte. Map the opaque foreground to white and the "
    "background to black. Produce soft, anti-aliased grayscale gradients at the edges of the "
    "subject to represent fine details and transparency."
)
RMBG3_PROMPT = json.dumps({"edit_instruction": RMBG3_INSTRUCTION}, separators=(",", ":"))
RMBG3_ENHANCER = """Prepare the fixed inference prompt for Bria RMBG-3.0 alpha-matte generation.
This model requires the exact instruction below for every input image. Your task is to return that instruction, not to expand or rewrite it.
Return exactly the following JSON object, with one edit_instruction string and no other keys:
""" + RMBG3_PROMPT + """
Do not paraphrase, translate, shorten or embellish the instruction. Do not describe the reference image, identify a subject, add scene details, or apply editing requests from the user prompt. Do not add a structured caption, image markers, a negative prompt, Markdown fences, commentary or an explanation. The output must be the JSON object above and nothing else."""

_COMMON = """Preserve the user's intent, language, exact supplied display strings, and explicit exclusions.
When a design needs writing and the wording is open, author complete suitable titles, labels and body copy.
Quote every intended visible string exactly; never leave placeholder copy or unspecified text areas.
Do not add lettering to a purely visual request. Return only the enhanced prompt, with no explanation or Markdown fences."""

CLASSIC = """Write a detailed image-generation caption for Fibo 2.0. Describe the finished scene, subjects, composition, materials, light, palette and style. """ + _COMMON
EDIT = """Write a concise Fibo 2.0 editing instruction grounded in the supplied references. State changes and preservation constraints. Number sources in supplied order as <image_1>, <image_2>, etc. Do not invent additional references. """ + _COMMON
STRUCTURED = """Write a complete valid Fibo 2.0 scene JSON object. Supported scene fields are short_description (string), objects (list of descriptive objects), background_setting (string), lighting (object), aesthetics (object), photographic_characteristics (object), style_medium (string), context (string), artistic_style (string), and text_render (list of objects with a text string). Omit unused sections.
Use semantic spatial descriptions in objects; there is no numeric-coordinate or bounding-box schema. Assign each subject and string to one owning element without duplicate objects. Every intended visible string must appear in text_render with exact text, and can include a location description. In descriptions quote any mentioned display strings with JSON-escaped double quotes. Close every quote, object and array; shorten descriptive detail to fit the response budget instead of truncating.
Complete example:
{"short_description":"A cream travel poster showing blue mountains and the title \\"EXPLORE\\".","objects":[{"description":"A layered blue mountain silhouette in the center"}],"background_setting":"Warm cream paper","style_medium":"Editorial poster","text_render":[{"text":"EXPLORE","location":"Large title above the mountains"},{"text":"Find your next trail","location":"Small subtitle below the mountains"}]}
""" + _COMMON
STRUCTURED_EDIT = STRUCTURED + """
For this editing task, put the scene object described above inside structured_caption.
The final response MUST have exactly these two required top-level fields: structured_caption and edit_instruction.
edit_instruction is a nonempty string stating the requested change and what to preserve. Reference images in supplied order as <image_1>, <image_2>, etc.
Complete final-response example:
{"structured_caption":{"short_description":"The source portrait with a red leather jacket"},"edit_instruction":"Change the jacket in <image_1> to red leather. Preserve the person's face, pose, lighting and background."}
Before returning, verify that BOTH top-level fields are present and that every intended visible string appears in structured_caption.text_render.
"""
