# Copyright (c) 2026 DeepBeepMeep - WanGP Community License 2.0 (see LICENSE.txt)
import os
from pathlib import Path
import torch
from shared.utils import files_locator as fl
from .help import INFOS, PROMPT_INFOS, DEEPY_INFOS, DEEPY_PROMPT_INFOS, ENHANCER, DIALOGUE_ENHANCER, DEFAULT_PROMPT

ARCHITECTURE = "fish_drama3"
ASSETS = ["tokenizer.json", "Fish_Drama3_codec_bf16.safetensors", "fish_audio_silence_probe.safetensors"]
# Optional cuts of long scripts at whole lines; each part is prompted with the last parts (text and generated audio) as history.
DIALOGUE_CUT_LINES = {"id": "dialogue_cut_lines", "label": "Cut Script Every N Lines (0 = One Pass)", "name": "Cut Every N Lines", "type": "int",
                      "default": 0, "min": 0, "max": 50, "inc": 1}
DIALOGUE_HISTORY_SECONDS = {"id": "dialogue_history_seconds", "label": "History Kept at Each Cut (seconds)", "name": "Cut History", "type": "int",
                            "default": 20, "min": 0, "max": 120, "inc": 5}


class family_handler:
    @staticmethod
    def query_supported_types():
        return [ARCHITECTURE]

    @staticmethod
    def query_family_maps():
        return {}, {}

    @staticmethod
    def query_model_family():
        return "tts"

    @staticmethod
    def query_family_infos():
        return {"tts": (2200, "TTS")}

    @staticmethod
    def get_lora_dir(base_model_type):
        return ARCHITECTURE

    @staticmethod
    def query_model_def(base_model_type, model_def):
        return {
            "group": "tts", "audio_only": True, "image_outputs": False, "sliding_window": False,
            "device_explicit": True, "dtype": "bf16", "auto_quantize": False,
            "no_negative_prompt": True, "inference_steps": False, "guidance_max_phases": 0,
            "temperature": True, "compile": False, "cuda_graphs": True, "supports_early_stop": True,
            "profiles_dir": [ARCHITECTURE], "lm_decoder": True,
            "image_prompt_types_allowed": "", "any_audio_prompt": True, "audio_prompt_choices": True,
            "audio_prompt_type_sources": {"selection": ["", "A", "AB", "ABD"], "default": "", "letters_filter": "ABD",
                "labels": {"": "New Voice", "A": "Voice Reference", "AB": "Voice References for Speakers 1 and 2", "ABD": "Voice References for Speakers 1, 2 and 3"}},
            "audio_guide_label": "Speaker 1 Voice Reference", "audio_guide2_label": "Speaker 2 Voice Reference", "audio_guide3_label": "Speaker 3 Voice Reference",
            "alt_prompt": {"label": "Reference Transcripts", "name": "Reference Transcripts", "lines": 3, "audio_prompt_type": "A",
                           "placeholder": "Exact words of each voice reference, one 'Speaker N: ...' line per reference. Leave empty to transcribe them with Whisper."},
            "custom_settings": [dict(DIALOGUE_CUT_LINES), dict(DIALOGUE_HISTORY_SECONDS)],
            "duration_slider": {"label": "Maximum Speech Duration (seconds)", "name": "Maximum Speech Duration", "min": 1, "max": 300, "increment": 1, "default": 60},
            "prompt_class": "Speech", "preserve_empty_prompt_lines": True,
            "prompt_enhancer_def": {"selection": ["T", "T1"], "labels": {"T": "A Speech based on current Prompt", "T1": "A Dialogue based on current Prompt"}, "default": ""},
            "prompt_enhancer_button_label": "Write",
            "text_prompt_enhancer_instructions": ENHANCER, "text_prompt_enhancer_max_tokens": 2048,
            "text_prompt_enhancer_instructions1": DIALOGUE_ENHANCER, "text_prompt_enhancer_max_tokens1": 2048,
            "infos": INFOS, "prompt_infos": PROMPT_INFOS, "deepy_infos": DEEPY_INFOS, "deepy_prompt_infos": DEEPY_PROMPT_INFOS,
        }

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return [{"repoId": "DeepBeepMeep/TTS", "sourceFolderList": ["fish_audio"], "fileList": [ASSETS]}]

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, save_quantized=False, lm_features=None, profile=0, **kwargs):
        from .pipeline import FishAudioPipeline
        paths = {name: fl.locate_file(str(Path("fish_audio") / name)) for name in ASSETS}
        backbone, codebooks = model_filename
        experts = fl.locate_file(os.path.basename(model_def["preload_URLs"][0]))  # routed experts, in their own format (FP8 or INT4)
        pipeline = FishAudioPipeline([backbone, experts], codebooks, paths[ASSETS[1]], paths[ASSETS[0]], paths[ASSETS[2]], lm_features)
        if save_quantized:
            from wgp import save_quantized_model
            config = str(Path(__file__).with_name("fish_drama3.json"))
            save_quantized_model(pipeline.transformer, model_type, backbone, torch.bfloat16, config, submodel_no=1, convrot_prefixes=("",))
            save_quantized_model(pipeline.transformer2, model_type, codebooks, torch.bfloat16, config, submodel_no=2, convrot_prefixes=("",))
        offload_def = {"pipe": {"transformer": pipeline.transformer, "transformer2": pipeline.transformer2, "vae": pipeline.vae}, "cudaGraphs": lm_features.graphs,
            "coTenantsMap": {"transformer": ["transformer2"], "transformer2": ["transformer"]}}
        if int(profile) in (0, 2, 4, 5):
            # Both stages run at every audio frame: streaming them would cost a full transfer of their weights per frame. They stay
            # whole in VRAM; their routed experts take what a VRAM Preload leaves beyond them, or the free VRAM with a dynamic preload.
            offload_def["budgets"] = {"transformer": 0, "transformer2": 0}
        return pipeline, offload_def

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update(prompt=DEFAULT_PROMPT, alt_prompt="", audio_prompt_type="",
            duration_seconds=60, temperature=0.8, top_k=30, top_p=0.95, prompt_enhancer="", guidance_scale=1.0,
            num_inference_steps=1, video_length=0, repeat_generation=1, multi_prompts_gen_type="FG")

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        pass  # New architecture; no older saved settings to migrate.

    @staticmethod
    def validate_generative_prompt(base_model_type, model_def, inputs, one_prompt):
        from .prompt import check_paired_tags, reference_transcripts, speaker_lines_to_tags, validate_drama3_text
        try:
            validate_drama3_text(speaker_lines_to_tags(one_prompt), "Speech Script")
            check_paired_tags(one_prompt, "Speech Script")
            references = [letter for letter in "ABD" if letter in inputs.get("audio_prompt_type", "")]
            # A missing transcript is made by Whisper at generation; a supplied one must be valid.
            for index, text in enumerate(reference_transcripts(inputs.get("alt_prompt", ""), len(references)), 1):
                if text:
                    validate_drama3_text(text, f"Reference Transcript {index}")
        except ValueError as error:
            return str(error)
        return None
