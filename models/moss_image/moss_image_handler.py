# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
import torch
from shared.utils.hf import build_hf_url
from .prompts import CLASSIC, STRUCTURED, EDITING

REPO = 'DeepBeepMeep/Moss'
TEXT_FOLDER = 'Qwen3.5-9B'


class family_handler:
    @staticmethod
    def query_supported_types():
        return ['moss_image']

    @staticmethod
    def query_model_family():
        return 'moss_image'

    @staticmethod
    def query_family_infos():
        return {'moss_image': (1161, 'Moss')}

    @staticmethod
    def query_family_maps():
        return {}, {'moss_image': ['moss_image']}

    @staticmethod
    def get_lora_dir(base_model_type):
        return 'moss_image'

    @staticmethod
    def render_prompt_helper(model_type, model_def, prompt_id, popup_id, prompt_elem_id, resolution_elem_id):
        if model_def.get('moss_edit', False):
            return ''
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
    def query_model_def(base_model_type, model_def):
        definition = {
            'image_outputs': True, 'device_explicit': True,
            'group': 'moss_image', 'profiles_dir': ['moss_image'], 'inference_steps': True,
            'guidance_max_phases': 1, 'embedded_guidance': False, 'no_negative_prompt': False,
            'no_background_removal': True, 'vae_block_size': 16, 'resolution_budget': '2048x2048',
            'tiny_vae_architecture': 'flux2_klein_4b',
            'skip_prompt_template': True, 'preserve_empty_prompt_lines': True,
            'text_encoder_folder': TEXT_FOLDER,
            'text_encoder_URLs': [build_hf_url(REPO, TEXT_FOLDER, f'qwen3_5_9b_text_{fmt}.safetensors') for fmt in ('bf16', 'int8_convrot')],
            'infos': 'Generate images from natural-language prompts, tag sequences or experimental JSON layouts. Base and SFT use the same 5B architecture; SFT is the curated-data post-trained release. Both start at 50 steps, CFG 3 and shift 5, with layer 22 skipped in the negative branch. Try CFG 3â€“6 for short prompts. Upstream recommends about 1â€“2 megapixels and describes extrapolation from 512Ã—512 to about 4 megapixels. Dimensions must be multiples of 16. Batch Size generates images together; larger batches use more VRAM. Layout edits change the prompt for a new generation; these checkpoints do not accept reference images or pixel masks. Prompt enhancement is optional and off by default.',
            'prompt_infos': 'Use Prompt Helper to draw, move and resize items and edit their exact visible text. It recovers common JSON punctuation mistakes and reports repairs before saving; unrepairable input stays available in the text editor. Save preserves comments and extra fields. The layout enhancer uses the selected output resolution. Re-enhance after changing resolution, or use Fit Boxes to Canvas to move existing boxes inside while preserving their size where possible. Describe the subject, composition, lighting, style and exact visible words. English has the most training coverage; Chinese and other languages are supported. Quote lettering and use explicit line breaks when needed, e.g. `A cream travel poster with blue mountains, title "EXPLORE", subtitle "Find your next trail".` Tags can follow quality, score, meta, year, character, series, artist and general content. JSON layouts use `summary.caption` and `items`, with `label`, `caption`, optional `text` and `bbox:[y_min,x_min,y_max,x_max]`. Set `bbox_format:"yxyx"` and `bbox_coord_space:"abs_centralized"`. Coordinates are pixel corners measured from the canvas center, X right and Y down: a 1024Ã—1024 canvas runs from -512 to 512 on both axes. Keep boxes inside the requested dimensions. Text and placement are guidance, not exact guarantees. Whole comment lines beginning with # are ignored during encoding.',
            'deepy_infos': 'Text-to-image only: `prompt` as natural language, tags or JSON layout; no image_refs, audio inputs or masks. Both releases default to num_inference_steps=50, guidance_scale=3, flow_shift=5, resolution="1024x1024", batch_size=1, video_prompt_type="", prompt_enhancer="". batch_size>1 denoises and decodes all images together with seeds seed, seed+1, etc.; larger batches use more VRAM. Negative-branch layer 22 skipping is built in. Dimensions must be multiples of 16; budget about 4 megapixels, recommended 1â€“2 megapixels. Base and SFT share architecture and adapters; neither is a step-distilled release.',
            'deepy_prompt_infos': 'Optional prompt_enhancer="T1" produces a layout using the selected resolution and centered pixel bounds; "T1K" adds Think. "T" produces a classic caption. Re-enhance layouts after changing resolution. The UI Prompt Helper edits centered-pixel boxes, reports punctuation repairs and preserves comments/extra fields on save. Use multi_prompts_gen_type="FG" for multiline captions/JSON. Quote exact text, specify line breaks and describe full scene/style. English recommended; preserve requested language. Optional layout: {summary:{meta:["poster"],canvas:"single",caption:"scene"},items:[{label:"text",caption:"appearance",text:"EXACT WORDS",bbox:[ymin,xmin,ymax,xmax]}],bbox_coord_space:"abs_centralized",bbox_format:"yxyx"}. Serialize as valid JSON. Integer pixel corners, origin canvas center, X right/Y down; bounds X +/-width/2, Y +/-height/2. Every visible element belongs to one item. No reference editing. Comments whose first nonspace character is # are ignored in encoding.',
            'text_prompt_enhancer_instructions': CLASSIC, 'text_prompt_enhancer_instructions1': STRUCTURED,
            'text_prompt_enhancer_max_tokens1': 4096,
            'prompt_enhancer_def': {'selection': ['T', 'T1'], 'labels': {'T': 'A Detailed Image Caption', 'T1': 'An Image Layout with Bounding Boxes'}, 'default': ''},
        }
        if model_def.get('moss_edit', False):
            definition.update({
                'at_least_one_image_ref_needed': True,
                'image_ref_choices': {'choices': [('Main Image Ref - First Reference Sets Output Dimensions (up to 4 Images)', 'KI'),
                                                   ('Ordered Reference Images (up to 4)', 'I')], 'letters_filter': 'KI', 'default': 'KI'},
                'fit_into_canvas_image_refs': 0,
                'custom_settings': [{'id': 'moss_image_kv_cache', 'name': 'KV Cache', 'label': 'KV Cache', 'type': 'dropdown', 'default': 'Disabled',
                                     'choices': [('Disabled (Slower but Lower VRAM/RAM)', 'Disabled'), ('Enabled (Fastest, Best Quality but Requires More VRAM/RAM)', 'Enabled'), ('INT8 (Fastest, About Half the Memory of Enabled)', 'INT8'), ('INT8 in RAM (Faster, Lowest VRAM)', 'INT8_RAM')],
                                     'info': 'Cache reference keys and values during this generation, in VRAM as far as they fit; when VRAM runs short, the layers that do not fit stay in system RAM and stream to the GPU as they run. Disabled recomputes them each step. INT8 caches them in about half the memory: with SageAttention the images match Enabled closely, with other attention modes they differ very slightly. INT8 in RAM keeps them in system RAM and streams each layer to the GPU as it runs, for the lowest VRAM. Text embeddings are cached in every mode.'}],
                'infos': '## Edit an Image\nDescribe a change to one reference image, such as replacing a background, changing clothing or adjusting lighting. MOSS Image Edit uses the SFT model with its editing adapter. Start at 50 steps and guidance 4.\n\n## Combine References\nProvide up to four ordered images and explain each image\'s role. Main Image Ref uses the first reference to set output dimensions through the selected resolution and canvas fitting; later images remain additional references. Ordered Reference Images keeps the selected output dimensions. Reference editing can change surrounding details; it does not lock pixels or provide a mask workflow. Multi-image quality depends on the requested combination.\n\n## Memory and Quality\nKV Cache is disabled by default. Enable it for faster denoising with extra memory (in VRAM as far as it fits, the rest streamed from system RAM), or choose INT8 for the same speedup in about half that memory; larger or more numerous references increase its cost. References are fitted by WanGP to the selected resolution. Start at 512 pixels when memory is limited, then increase the resolution. Batch Size generates several edited outputs from the same references. Prompt enhancement is optional and off by default.',
                'prompt_infos': '## Editing Instructions\nSay what to change and what to keep. Example: `Change the background to a sunny beach. Preserve the person, clothing and pose.`\n\n## Multiple Images\nIdentify references in their input order using ordinary language, for example: `Keep the person and pose from image 1, dress them in the jacket from image 2, and use the room from image 3 as the background.` These are descriptions, not special tokens.\n\n## Written Content\nQuote exact visible wording, for example: `Replace the sign text with "WELCOME" and preserve its lettering style.` Preserve existing text unless it should change. English and Chinese instructions are supported. Use natural-language edit instructions; the text-to-image layout editor is not an editing mask.',
                'deepy_infos': 'Reference image editing with MOSS Image SFT plus its built-in adapter. Require 1-4 ordered image_refs. video_prompt_type="KI" makes image_refs[0] determine output dimensions via resolution and fit_canvas; "I" keeps the selected dimensions. Additional references remain ordered. Defaults: num_inference_steps=50, guidance_scale=4, flow_shift=5, resolution="1024x1024", batch_size=1, prompt_enhancer="", custom_settings.moss_image_kv_cache="Disabled". Enable that dropdown with "Enabled" to retain reference K/V for this run (in VRAM as far as it fits, the rest streamed from RAM), "INT8" to retain them in about half the memory, or "INT8_RAM" to keep them in RAM streamed per layer (lowest VRAM); text is timestep-dependent and is not in the K/V cache. Step-varying adapters recompute reference K/V. No pixel masks, video, audio or alpha workflow. Dimensions are multiples of 16; reduce resolution for memory. Multi-reference edits may alter composition or unrequested details. Base/SFT text-to-image defaults are separate workflows.',
                'deepy_prompt_infos': 'Use concise natural-language instructions describing the change and retained details. Name image 1, image 2, etc. in reference order and explain their roles; no special image tags or vision-token prefix. Example: "Keep the person from image 1 and give them the jacket from image 2; preserve their face and pose." Quote exact visible text. English/Chinese supported. prompt_enhancer="T" refines the instruction, "TI" also consults reference images; enhancement remains opt-in. Do not substitute a complete unrelated scene description for a bounded edit. No structured edit-mask syntax is supported.',
                'prompt_enhancer_def': {'selection': ['T', 'TI'], 'labels': {'T': 'An Image Editing Instruction', 'TI': 'An Image Edit using {image_inputs}'}, 'default': ''},
                'text_prompt_enhancer_instructions': EDITING, 'image_prompt_enhancer_instructions': EDITING,
                'text_prompt_enhancer_max_tokens': 768, 'image_prompt_enhancer_max_tokens': 768,
            })
            if model_def.get('moss_edit_variant') == 'hq_v4':
                definition.update({
                    'infos': '## Edit an Image\nHQ v4 is a community editing adapter for MOSS Image SFT, trained on single-reference edits with instructions to preserve surrounding details. Start with one source image, 50 steps and guidance 3. Raise guidance to 3.5-4 if the requested change is too weak.\n\n## Reference Images\nMain Image Ref uses the first reference to set output dimensions through the selected resolution and canvas fitting. Ordered Reference Images keeps the selected output dimensions. Up to four ordered references are accepted, but combining people or objects across photos is limited: v4 was trained primarily for single-image edits. Describe what to change and what to retain; unchanged pixels are not guaranteed and there is no mask workflow.\n\n## Memory and Quality\nThe same SFT base, text encoder and VAE are reused, with a larger editing adapter. KV Cache is optional and disabled by default; HQ needs separate caches for its two guidance passes, so enabling it uses more memory than Edit SFT. INT8 reduces the cache size; INT8 in RAM keeps it in system memory. References follow WanGP\'s resolution fitting. Start around 1024 pixels, or reduce resolution and batch size when memory is limited. Prompt enhancement is optional and off by default.',
                    'deepy_infos': 'Community MOSS Image Edit HQ v4 (armV4) on the SFT base. Require 1-4 ordered image_refs; prefer one source image because v4 training uses single-reference edits and multi-image compositing is limited. video_prompt_type="KI" uses image_refs[0] for output dimensions via resolution and fit_canvas; "I" keeps the selected dimensions. Defaults: num_inference_steps=50, guidance_scale=3, flow_shift=5, resolution="1024x1024", batch_size=1, prompt_enhancer="", custom_settings.moss_image_kv_cache="Disabled". Try guidance_scale=3.5-4 for weak edits. Uses the released ai-toolkit reference positioning and negative-branch layer-22 skip; reference K/V must be separate per guidance branch. Cache choices: Disabled, Enabled, INT8, INT8_RAM; the last streams cached reference K/V from RAM. Step-varying adapters invalidate both caches. No masks, video, audio or alpha workflow; surrounding details can change. Same SFT BF16/ConvRot base weights, text encoder, VAE, LoRA folder and profiles directory as the MOSS family; the HQ adapter replaces the original editing adapter.',
                })
                definition['prompt_infos'] += '\n\n## HQ v4\nPrefer a bounded edit to one source image, for example: `Change only the jacket to dark blue. Preserve the person\'s face, hair, pose, background and lighting.` Additional references are accepted but complex compositing is less reliable.'
                definition['deepy_prompt_infos'] += ' HQ v4: prefer single-source bounded edits and explicit preservation instructions; do not promise exact pixel preservation or reliable multi-image compositing.'
        return definition

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        files = [{'repoId': REPO, 'sourceFolderList': [TEXT_FOLDER, 'moss_image'],
                  'fileList': [['config.json', 'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja'], ['flux2_small_decoder_bf16.safetensors']]}]
        if model_def.get('moss_edit', False):
            files.append({'repoId': 'DeepBeepMeep/Flux2', 'sourceFolderList': [''], 'fileList': [['flux2_vae.safetensors']]})
        return files

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, text_encoder_filename=None, save_quantized=False, dtype=torch.bfloat16, **kwargs):
        from .pipeline import load_pipeline
        pipeline = load_pipeline(model_filename, text_encoder_filename, model_def, model_type, save_quantized, dtype)
        return pipeline, {'transformer': pipeline.transformer, 'text_encoder': pipeline.text_encoder, 'vae': pipeline.vae}

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update({'image_mode': 1, 'resolution': '1024x1024', 'batch_size': 1, 'num_inference_steps': 50,
                            'guidance_scale': 3, 'flow_shift': 5, 'prompt_enhancer': '', 'video_prompt_type': '', 'multi_prompts_gen_type': 'FG'})
        if model_def.get('moss_edit', False):
            ui_defaults.update(guidance_scale=3 if model_def.get('moss_edit_variant') == 'hq_v4' else 4, video_prompt_type='KI', remove_background_images_ref=0,
                               custom_settings={'moss_image_kv_cache': 'Disabled'})

    @staticmethod
    def get_rgb_factors(base_model_type):
        from shared.RGB_factors import get_rgb_factors
        return get_rgb_factors('flux', sub_family='flux2')
