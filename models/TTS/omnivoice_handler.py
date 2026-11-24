# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from models.help_shared import merge_help
from .help import OMNIVOICE_HELP
import os
import re

import torch

from shared.deepy.assets import whisper_download_def
from shared.mps import mps_device_or
from shared.utils import files_locator as fl

from .omnivoice.settings import (
    OMNIVOICE_ASSET_DIR,
    OMNIVOICE_AUDIO_TOKENIZER_DIR,
    OMNIVOICE_AUDIO_TOKENIZER_WEIGHTS,
    OMNIVOICE_AUTO_END_TRIM_FLAG,
    OMNIVOICE_ADJUST_SPEED_SETTING_ID,
    OMNIVOICE_AUTO_SPLIT_MAX_SECONDS,
    OMNIVOICE_AUTO_SPLIT_MIN_SECONDS,
    OMNIVOICE_AUTO_SPLIT_SETTING_ID,
    OMNIVOICE_CONFIG_NAME,
    OMNIVOICE_DEFAULT_VOICE_INSTRUCTION,
    is_omnivoice_voice_instruction,
    normalize_omnivoice_voice_instruction,
)
from .omnivoice.utils.voice_design import _INSTRUCT_VALID_EN, _INSTRUCT_VALID_ZH, _ZH_RE
from .prompt_enhancers import TTS_MONOLOGUE_PROMPT, TTS_QWEN3_DIALOGUE_PROMPT


OMNIVOICE_REPO_ID = "DeepBeepMeep/TTS"
OMNIVOICE_MAIN_FILENAME = "omnivoice_bf16.safetensors"
OMNIVOICE_QUANT_FILENAME = "omnivoice_quanto_bf16_int8.safetensors"
OMNIVOICE_TOKENIZER_FILES = [
    OMNIVOICE_CONFIG_NAME,
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
]
OMNIVOICE_AUDIO_TOKENIZER_FILES = [
    "config.json",
    "preprocessor_config.json",
    OMNIVOICE_AUDIO_TOKENIZER_WEIGHTS,
]
OMNIVOICE_LANGUAGE_CHOICES = [
    ("Auto", "auto"),
    ("English", "english"),
    ("Chinese", "chinese"),
    ("French", "french"),
    ("German", "german"),
    ("Italian", "italian"),
    ("Japanese", "japanese"),
    ("Korean", "korean"),
    ("Portuguese", "portuguese"),
    ("Spanish", "spanish"),
    ("Arabic", "arabic"),
    ("Hindi", "hindi"),
    ("Russian", "russian"),
]
OMNIVOICE_DURATION_SLIDER = {
    "label": "Max duration (seconds, 0 = auto)",
    "name": "Max Duration",
    "min": 0,
    "max": 600,
    "increment": 1,
    "default": 0,
}
OMNIVOICE_AUDIO_PROMPT_TYPE_SOURCES = {
    "selection": ["", "A", "AB", "ABD"],
    "labels": {
        "": "Voice design",
        "A": "Voice cloning (1 reference audio)",
        "AB": "Voice cloning dialogue (Speaker 1 and Speaker 2)",
        "ABD": "Voice cloning dialogue (Speakers 1, 2 and 3)",
    },
    "letters_filter": "ABD",
    "default": "",
}
OMNIVOICE_AUDIO_PROMPT_TYPE_CUSTOM_OPTION = {
    "label": "Auto Detect Segment End",
    "flag": OMNIVOICE_AUTO_END_TRIM_FLAG,
}
OMNIVOICE_CUSTOM_SETTINGS = [
    {
        "id": OMNIVOICE_AUTO_SPLIT_SETTING_ID,
        "label": "Auto Split s (0 = Auto, 5-90)",
        "name": "Auto Split Every s",
        "type": "float",
        "default": 0.0,
        "min": 0.0,
        "max": OMNIVOICE_AUTO_SPLIT_MAX_SECONDS,
        "inc": 1.0,
    },
    {
        "id": OMNIVOICE_ADJUST_SPEED_SETTING_ID,
        "label": "Adjust Speed to Max duration",
        "name": "Adjust Speed",
        "type": "dropdown",
        "choices": ["No", "Yes"],
        "default": "No",
    },
]
OMNIVOICE_PROMPT_SPECIAL_TAGS = [
    "[laughter]",
    "[sigh]",
    "[confirmation-en]",
    "[question-en]",
    "[question-ah]",
    "[question-oh]",
    "[question-ei]",
    "[question-yi]",
    "[surprise-ah]",
    "[surprise-oh]",
    "[surprise-wa]",
    "[surprise-yo]",
    "[dissatisfaction-hnn]",
]


def _format_markdown_items(items):
    return ", ".join(f"`{item}`" for item in sorted(items))


def _read_omnivoice_text_input(value):
    if value is None:
        return ""
    if isinstance(value, str) and os.path.isfile(value):
        with open(value, "r", encoding="utf-8") as reader:
            return reader.read()
    return str(value)


def _validate_omnivoice_instruction(instruction, target_text):
    from .omnivoice.modeling_omnivoice import _resolve_instruct  # here: the model code is heavy to import
    try:
        _resolve_instruct(instruction, use_zh=bool(target_text and _ZH_RE.search(target_text)))
    except ValueError as error:
        return f"Invalid OmniVoice voice instruction:\n{error}"
    return None


OMNIVOICE_INFOS = f"""
## Prompt special tags

These tags can be inserted directly in the main prompt text:
{", ".join(f"`{tag}`" for tag in OMNIVOICE_PROMPT_SPECIAL_TAGS)}

## Voice instruction / reference transcript(s)

Use this field differently depending on the selected voice mode.

### Voice Design

Leave it blank for Auto Voice. To design a voice, enter comma-separated voice tags.

Valid English tags:
{_format_markdown_items(_INSTRUCT_VALID_EN)}

Valid Chinese tags:
{_format_markdown_items(_INSTRUCT_VALID_ZH)}

Examples:

```text
female, young adult, low pitch, british accent
male, middle-aged, whisper
```

Use only valid tags here. Square-bracket non-verbal tags such as `[laughter]` belong in the main prompt, not in this field.

### Voice cloning

Upload a reference audio file and either leave this field blank or enter the exact transcript of the reference audio. If left blank, WanGP transcribes the reference with Whisper.

The transcript must describe the reference audio, not the target prompt. For best results, use a clean 3-10 second reference clip, preferably in the same language as the text you want to generate.

If this field contains only valid voice tags such as `female` or `male, british accent`, WanGP treats it as a voice instruction rather than a reference transcript.

### Two- or three-speaker cloning

Upload the two or three reference voices and provide transcripts like this, or leave blank for Whisper transcription:

```text
Speaker 1: Exact words spoken in the first reference audio.
Speaker 2: Exact words spoken in the second reference audio.
Speaker 3: Exact words spoken in the third reference audio.
```
"""


def _get_omnivoice_model_def():
    return {
        "audio_only": True,
        "image_outputs": False,
        "sliding_window": False,
        "guidance_max_phases": 1,
        "no_negative_prompt": True,
        "inference_steps": True,
        "temperature": False,
        "image_prompt_types_allowed": "",
        "supports_early_stop": True,
        "profiles_dir": ["omnivoice"],
        "duration_slider": dict(OMNIVOICE_DURATION_SLIDER),
        "infos": OMNIVOICE_INFOS,
        "model_modes": {
            "choices": list(OMNIVOICE_LANGUAGE_CHOICES),
            "default": "auto",
            "label": "Language",
        },
        "alt_prompt": {
            "label": "Voice instruction / reference transcript(s)",
            "placeholder": "Voice Design: optional voice tags such as female\nVoice clone: optional transcript; blank uses Whisper to autotranscribe",
            "lines": 4,
        },
        "preserve_empty_prompt_lines": True,
        "pause_between_sentences": True,
        "any_audio_prompt": True,
        "audio_prompt_choices": True,
        "audio_prompt_type_sources": dict(OMNIVOICE_AUDIO_PROMPT_TYPE_SOURCES),
        "audio_prompt_type_custom_option": dict(OMNIVOICE_AUDIO_PROMPT_TYPE_CUSTOM_OPTION),
        "custom_settings": [one.copy() for one in OMNIVOICE_CUSTOM_SETTINGS],
        "audio_guide_label": "Speaker 1 reference voice",
        "audio_guide2_label": "Speaker 2 reference voice",
        "audio_guide3_label": "Speaker 3 reference voice",
        "text_prompt_enhancer_instructions": TTS_MONOLOGUE_PROMPT,
        "text_prompt_enhancer_instructions1": TTS_QWEN3_DIALOGUE_PROMPT,
        "text_prompt_enhancer_max_tokens": 512,
        "text_prompt_enhancer_max_tokens1": 512,
        "prompt_enhancer_def": {
            "selection": ["T", "T1"],
            "labels": {
                "T": "A Speech based on current Prompt",
                "T1": "A Dialogue between two People based on current Prompt",
            },
            "default": "T",
        },
        "prompt_enhancer_button_label": "Write",
        "compile": False,
    }


def _get_omnivoice_download_def():
    return [
        {
            "repoId": OMNIVOICE_REPO_ID,
            "sourceFolderList": [OMNIVOICE_ASSET_DIR, OMNIVOICE_AUDIO_TOKENIZER_DIR],
            "fileList": [OMNIVOICE_TOKENIZER_FILES, OMNIVOICE_AUDIO_TOKENIZER_FILES],
        },
        whisper_download_def(),
    ]


class family_handler:
    @staticmethod
    def query_supported_types():
        return ["omnivoice"]

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
        return "omnivoice"

    @staticmethod
    def query_model_def(base_model_type, model_def):
        return merge_help({**_get_omnivoice_model_def(), "device_explicit": True}, OMNIVOICE_HELP)

    @staticmethod
    def query_model_files(computeList, base_model_type, model_def=None):
        return _get_omnivoice_download_def()

    @staticmethod
    def load_model(
        model_filename,
        model_type,
        base_model_type,
        model_def,
        quantizeTransformer=False,
        text_encoder_quantization=None,
        dtype=None,
        VAE_dtype=None,
        mixed_precision_transformer=False,
        save_quantized=False,
        submodel_no_list=None,
        text_encoder_filename=None,
        profile=0,
        **kwargs,
    ):
        from .omnivoice.pipeline import OmniVoicePipeline

        weights_path = model_filename[0] if isinstance(model_filename, (list, tuple)) else model_filename
        pipeline = OmniVoicePipeline(
            model_weights_path=weights_path,
            ckpt_root=fl.get_download_location(),
            device=mps_device_or(torch.device("cpu")),
            dtype=dtype or torch.bfloat16,
        )
        pipe = {
            "transformer": pipeline.model,
            "audio_tokenizer": pipeline.audio_tokenizer,
        }

        if save_quantized and weights_path:
            from wgp import save_quantized_model

            config_path = fl.locate_file(os.path.join(OMNIVOICE_ASSET_DIR, OMNIVOICE_CONFIG_NAME))
            save_quantized_model(pipeline.model, model_type, weights_path, dtype or torch.bfloat16, config_path)

        return pipeline, pipe

    @staticmethod
    def fix_settings(base_model_type, settings_version, model_def, ui_defaults):
        ui_defaults.setdefault("audio_prompt_type", "")
        ui_defaults.setdefault("model_mode", "auto")
        ui_defaults.setdefault("alt_prompt", "")
        ui_defaults["alt_prompt"] = normalize_omnivoice_voice_instruction(str(ui_defaults.get("alt_prompt") or ""))
        ui_defaults.setdefault("pause_seconds", 0.2)

    @staticmethod
    def update_default_settings(base_model_type, model_def, ui_defaults):
        duration_def = model_def.get("duration_slider", {})
        ui_defaults.update(
            {
                "audio_prompt_type": "",
                "model_mode": "auto",
                "prompt": "The lights are already on, so we can start whenever you are ready.",
                "alt_prompt": OMNIVOICE_DEFAULT_VOICE_INSTRUCTION,
                "repeat_generation": 1,
                "duration_seconds": duration_def.get("default", 0),
                "pause_seconds": 0.2,
                "video_length": 0,
                "num_inference_steps": 32,
                "negative_prompt": "",
                "temperature": 0.1,
                "guidance_scale": 2.0,
                "multi_prompts_gen_type": "FG",
            }
        )

    @staticmethod
    def validate_generative_prompt(base_model_type, model_def, inputs, one_prompt):
        if one_prompt is None or len(str(one_prompt).strip()) == 0:
            return "Prompt text cannot be empty for OmniVoice."
        audio_prompt_type = str(inputs.get("audio_prompt_type", "") or "").upper()
        text = str(one_prompt)
        instruction_or_ref = normalize_omnivoice_voice_instruction(_read_omnivoice_text_input(inputs.get("alt_prompt", ""))).strip()
        if instruction_or_ref and ("A" not in audio_prompt_type or is_omnivoice_voice_instruction(instruction_or_ref)):
            instruction_error = _validate_omnivoice_instruction(instruction_or_ref, text)
            if instruction_error is not None:
                return instruction_error
        has_speaker_syntax = re.search(r"Speaker\s*\d+\s*:", text, flags=re.IGNORECASE) is not None
        if "A" in audio_prompt_type and "B" not in audio_prompt_type and inputs.get("audio_guide") is None:
            return "OmniVoice voice cloning requires a reference audio file."
        if "B" in audio_prompt_type:
            speaker_count = 3 if "D" in audio_prompt_type else 2
            if any(inputs.get(key) is None for key in ("audio_guide", "audio_guide2", "audio_guide3")[:speaker_count]):
                return f"OmniVoice dialogue mode requires {speaker_count} reference audio files."
            speaker_matches = list(re.finditer(r"Speaker\s*(\d+)\s*:", text, flags=re.IGNORECASE))
            if not speaker_matches:
                return f"OmniVoice dialogue mode requires prompt lines using Speaker 1: to Speaker {speaker_count}:."
            speaker_ids = sorted({int(m.group(1)) for m in speaker_matches})
            if len(speaker_ids) != speaker_count:
                return f"OmniVoice dialogue mode requires exactly {speaker_count} speaker IDs. Use Speaker 1: to Speaker {speaker_count}:."
        elif has_speaker_syntax:
            return "Speaker-tag dialogue requires an OmniVoice dialogue mode."
        return None

    @staticmethod
    def validate_generative_settings(base_model_type, model_def, inputs):
        custom_settings = inputs.get("custom_settings", None)
        if custom_settings is None:
            return None
        if not isinstance(custom_settings, dict):
            return "Custom settings must be a dictionary."

        raw_value = custom_settings.get(OMNIVOICE_AUTO_SPLIT_SETTING_ID, None)
        if raw_value is None:
            return None
        if isinstance(raw_value, str):
            raw_value = raw_value.strip()
            if len(raw_value) == 0:
                custom_settings.pop(OMNIVOICE_AUTO_SPLIT_SETTING_ID, None)
                inputs["custom_settings"] = custom_settings if len(custom_settings) > 0 else None
                return None

        try:
            if isinstance(raw_value, bool):
                raise ValueError()
            auto_split_seconds = float(raw_value)
        except Exception:
            return (
                f"Auto Split Every s must be a number between "
                f"{int(OMNIVOICE_AUTO_SPLIT_MIN_SECONDS)} and {int(OMNIVOICE_AUTO_SPLIT_MAX_SECONDS)} seconds."
            )

        if auto_split_seconds == 0.0:
            custom_settings.pop(OMNIVOICE_AUTO_SPLIT_SETTING_ID, None)
            inputs["custom_settings"] = custom_settings if len(custom_settings) > 0 else None
            return None
        if auto_split_seconds < OMNIVOICE_AUTO_SPLIT_MIN_SECONDS or auto_split_seconds > OMNIVOICE_AUTO_SPLIT_MAX_SECONDS:
            return (
                f"Auto Split Every s must be between "
                f"{int(OMNIVOICE_AUTO_SPLIT_MIN_SECONDS)} and {int(OMNIVOICE_AUTO_SPLIT_MAX_SECONDS)} seconds."
            )

        custom_settings[OMNIVOICE_AUTO_SPLIT_SETTING_ID] = auto_split_seconds
        inputs["custom_settings"] = custom_settings
        return None
