# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from pathlib import Path

import torch
from shared.utils import files_locator as fl
from .help import INFOS, PROMPT_INFOS, DEEPY_INFOS, DEEPY_PROMPT_INFOS, DEFAULT_PROMPT, ENHANCER, DIALOGUE_ENHANCER

ARCHITECTURE = "zonos3_mini"
FOLDER = "zonos3"
ASSETS = ["ZAC1_keep2_decoder.safetensors", "MECT_ownmect_s2.safetensors", "voices.safetensors", "superbpe8192.json", "steering.safetensors"]
PRESETS = "zonos3_mini_presets"


def _languages():
    from .loading import CONFIG
    return [(language["label"], language["code"]) for language in CONFIG["languages"]]


def _custom_settings():
    from .voices import DEFAULT_VOICE, VOICES
    return [
        {"id": "voice", "name": "Voice", "label": "Voice", "type": "dropdown", "default": DEFAULT_VOICE, "audio_prompt_type_not": "A",
         "choices": [(f"{name} ({language}, {style})", key) for key, (name, language, style) in VOICES.items()]},
        {"id": "dialogue_voices", "name": "Other Speakers' Voices", "label": "Other Speakers' Voices (e.g. Rupert, Willow; empty: the next voices of the list)", "type": "text",
         "default": "", "audio_prompt_type_not": "A"},
        {"id": "language", "name": "Language", "label": "Language (reading of numbers, dates and symbols)", "type": "dropdown", "default": "en_us", "choices": _languages()},
        {"id": "speaking_rate", "name": "Speaking Rate", "label": "Speaking Rate", "type": "dropdown", "default": "auto",
         "choices": [("Auto", "auto"), ("Very Slow", "very_slow"), ("Slow", "slow"), ("Normal", "normal"), ("Fast", "fast"), ("Very Fast", "very_fast")]},
    ]


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
            "profiles_dir": [ARCHITECTURE], "preset_profiles_dir": [PRESETS], "lm_decoder": True, "pause_between_sentences": True,
            "image_prompt_types_allowed": "", "any_audio_prompt": True, "audio_prompt_choices": True,
            "audio_prompt_type_sources": {"selection": ["", "A", "AB", "ABD"], "default": "", "letters_filter": "ABD",
                "labels": {"": "Built-in Voice", "A": "Voice Reference", "AB": "Voice References for Speakers 1 and 2", "ABD": "Voice References for Speakers 1, 2 and 3"}},
            "audio_guide_label": "Speaker 1 Voice Reference", "audio_guide2_label": "Speaker 2 Voice Reference", "audio_guide3_label": "Speaker 3 Voice Reference",
            "alt_prompt": {"label": "Voice Controls (optional, filled by the Presets)", "name": "Voice Controls", "lines": 1,
                           "placeholder": "e.g. happy=3, pace=-1, gender=1, deep=2, accent:irish=1.5 (see the help)"},
            "custom_settings": _custom_settings(),
            "duration_slider": {"label": "Maximum Speech Duration (seconds)", "name": "Maximum Speech Duration", "min": 1, "max": 600, "increment": 1, "default": 60},
            "prompt_class": "Speech", "preserve_empty_prompt_lines": True,
            "prompt_enhancer_def": {"selection": ["T", "T1"], "labels": {"T": "A Speech based on current Prompt", "T1": "A Dialogue between two People based on current Prompt"}, "default": ""},
            "prompt_enhancer_button_label": "Write",
            "text_prompt_enhancer_instructions": ENHANCER, "text_prompt_enhancer_max_tokens": 1024,
            "text_prompt_enhancer_instructions1": DIALOGUE_ENHANCER, "text_prompt_enhancer_max_tokens1": 1536,
            "infos": INFOS, "prompt_infos": PROMPT_INFOS, "deepy_infos": DEEPY_INFOS, "deepy_prompt_infos": DEEPY_PROMPT_INFOS,
        }

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return [{"repoId": "DeepBeepMeep/TTS", "sourceFolderList": [FOLDER], "fileList": [ASSETS]}]

    @staticmethod
    def load_model(model_filename, model_type, base_model_type, model_def, save_quantized=False, lm_features=None, profile=0, **kwargs):
        from .pipeline import Zonos3Pipeline
        lm_file, = model_filename
        paths = [fl.locate_file(str(Path(FOLDER) / name)) for name in ASSETS]
        pipeline = Zonos3Pipeline(lm_file, *paths, lm_features)
        if save_quantized:
            from wgp import save_quantized_model
            save_quantized_model(pipeline.transformer, model_type, lm_file, torch.bfloat16, str(Path(__file__).with_name("zonos3_mini.json")), convrot_prefixes=("",))
        # The LM runs once per 1/15 s of speech: streamed to the GPU at every frame it decoded 9.5 frames/s (measured with the dynamic
        # preload of profile 3), slower than real time. It stays whole in VRAM (1.7 GB) with every profile.
        offload_def = {"pipe": {"transformer": pipeline.transformer, "vae": pipeline.vae, "speaker_encoder": pipeline.speaker_encoder}, "cudaGraphs": lm_features.graphs,
                       "budgets": {"transformer": 0}}
        return pipeline, offload_def

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        ui_defaults.update(prompt=DEFAULT_PROMPT, alt_prompt="", audio_prompt_type="", duration_seconds=60, pause_seconds=0.3,
            temperature=0.75, top_k=80, top_p=0.69, prompt_enhancer="", guidance_scale=1.0, num_inference_steps=1, video_length=0,
            repeat_generation=1, multi_prompts_gen_type="FG")

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        pass  # New architecture; no older saved settings to migrate.

    @staticmethod
    def validate_generative_prompt(base_model_type, model_def, inputs, one_prompt):
        from shared.utils.speech_dialogue import SPEAKER_LINE_PATTERN, parse_dialogue
        from .pipeline import DEFAULT_VOICE, check_cues, speaker_voices, voice_controls
        settings = inputs.get("custom_settings") or {}
        audio_prompt_type = inputs.get("audio_prompt_type", "")
        references = sum(letter in audio_prompt_type for letter in "ABD")
        labels = {int(match.group(2)) for match in SPEAKER_LINE_PATTERN.finditer(one_prompt)}
        try:
            voice_controls(inputs.get("alt_prompt", ""))
            check_cues(one_prompt)
            # built-in voices: the speakers without a named voice take the next voices of the list
            voices = references or len(speaker_voices(settings.get("voice", DEFAULT_VOICE), settings.get("dialogue_voices") or "", len(labels)))
        except ValueError as error:
            return str(error)
        if labels:
            try:
                parse_dialogue(one_prompt, len(labels), line_labels=True, canonical_ids=True)
            except ValueError as error:
                return str(error)
            if len(labels) > voices:
                return f"The script has {len(labels)} speakers but only {voices} voice{'s' if voices > 1 else ''}: " + \
                    ("add a voice reference per speaker." if references else "at most one speaker per built-in voice.")
        elif references > 1:
            return "With several voice references, start each line with its speaker (Speaker 1:, Speaker 2:...)."
        return None
