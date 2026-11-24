# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""OmniVoice's file names, setting ids and voice-instruction helpers: read by the handler without importing the model."""
import re

from .utils.voice_design import _INSTRUCT_VALID_EN, _INSTRUCT_VALID_ZH

OMNIVOICE_ASSET_DIR = "omnivoice"
OMNIVOICE_AUDIO_TOKENIZER_DIR = "omnivoice/audio_tokenizer"
OMNIVOICE_CONFIG_NAME = "config.json"
OMNIVOICE_AUDIO_TOKENIZER_WEIGHTS = "audio_tokenizer_bf16.safetensors"
OMNIVOICE_DEFAULT_VOICE_INSTRUCTION = ""
OMNIVOICE_LEGACY_DEFAULT_VOICE_INSTRUCTION = "female, warm tone, clear articulation"
OMNIVOICE_AUTO_END_TRIM_FLAG = "E"
OMNIVOICE_AUTO_SPLIT_SETTING_ID = "auto_split_every_s"
OMNIVOICE_ADJUST_SPEED_SETTING_ID = "adjust_speed_to_max_duration"
OMNIVOICE_AUTO_SPLIT_MIN_SECONDS = 5.0
OMNIVOICE_AUTO_SPLIT_MAX_SECONDS = 90.0

def normalize_omnivoice_voice_instruction(text: str) -> str:
    if text.strip().lower() == OMNIVOICE_LEGACY_DEFAULT_VOICE_INSTRUCTION:
        return ""
    return text


def is_omnivoice_voice_instruction(text: str) -> bool:
    parts = [part.strip() for part in re.split(r"[,\uff0c]", text or "") if part.strip()]
    if not parts:
        return False
    return all(part.lower() in _INSTRUCT_VALID_EN or part in _INSTRUCT_VALID_ZH for part in parts)
