# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Prime's advanced transcription (toolbox action advanced_transcription): a WanGP generation of the transcription model that the user
enabled in the configuration, read back from the transcript it saves beside its output audio and from its Info."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from shared.deepy.config import DEEPY_CONTEXT_TOKENS_DEFAULT, DEEPY_CONTEXT_TOKENS_KEY, advanced_transcription_model, advanced_transcription_terms, get_deepy_config_value, normalize_deepy_context_tokens

TASKS = ("transcribe", "translate")  # the model's model_mode values 0 and 1
STYLES = ("verbatim", "readable")
TIMESTAMPS = {"turn": 1, "word": 2, "off": 0}  # the model's word_timestamps values: a timed line per turn, per word, or plain text
OUTPUTS = ("audio", "video")  # video: the source video with the subtitles as a subtitle track (a video source only)
SIDE_FILES = {"transcript": ".txt", "subtitles": ".srt"}
INFO_FIELDS = {"speakers": "Speakers", "translated_to": "Translated To", "notes": "Transcription Notes"}  # Info entries returned as they are


def enabled() -> bool:
    return bool(advanced_transcription_model())


def merged_terms(*lists: str) -> str:
    """Comma- or line-separated term lists joined once each (case-insensitive), in order; "#" comment lines dropped."""
    terms = {}
    for term in (term.strip() for text in lists for term in re.split(r"[,\n]", text)):
        if term and not term.startswith("#"):
            terms.setdefault(term.casefold(), term)
    return ", ".join(terms.values())


def build_task(source_media: dict[str, Any], *, client_id: str, task: str, target_language: str, speakers: bool, timestamps: str, style: str, context_terms: str,
               speaker_names: str = "", output: str = "audio") -> dict[str, Any]:
    """The generation settings: an audio file is the audio_guide, a video the Control Video (its first audio track is transcribed); the
    user's configured names and jargon come before the call's context terms."""
    path = str(source_media["path"])
    source = {"audio_prompt_type": "A", "audio_guide": path} if source_media["media_type"] == "audio" else {"audio_prompt_type": "K", "video_prompt_type": "V", "video_guide": path}
    return {"model_type": advanced_transcription_model(), "client_id": client_id, "prompt": merged_terms(advanced_transcription_terms(), context_terms), "model_mode": TASKS.index(task), **source,
            "custom_settings": {"speakers": int(speakers), "speaker_names": speaker_names, "word_timestamps": TIMESTAMPS[timestamps], "style": style,
                                "target_language": target_language or "English", "output": output}}


def _budget() -> int:
    """Characters of transcript returned inline: a share of the result budget that Deepy Prime accepts from a tool."""
    return int(0.8 * max(8000, min(normalize_deepy_context_tokens(get_deepy_config_value(DEEPY_CONTEXT_TOKENS_KEY, DEEPY_CONTEXT_TOKENS_DEFAULT)), 100000)))


def read_result(output_file: str, info: dict[str, Any], virtualize) -> dict[str, Any]:
    """The transcript of a finished generation (one turn, or one word, per line: "[start - end] Speaker N: text"), its beginning when longer
    than the inline budget, with the speakers and notes of the generation's Info."""
    output = Path(output_file)
    files = {name: output.with_name(output.stem + suffix) for name, suffix in SIDE_FILES.items()}
    text = files["transcript"].read_text(encoding="utf-8").strip()
    budget = _budget()
    shown = text if len(text) <= budget else text[:max(text.rfind("\n", 0, budget), budget // 2)].rstrip()
    result = {key: info[label] for key, label in INFO_FIELDS.items() if label in info}
    result.update(text=shown, text_complete=len(shown) == len(text), line_count=text.count("\n") + 1)
    result["files"] = {name: virtualize(str(path)).replace("\\", "/") for name, path in files.items() if path.is_file()}
    if not result["text_complete"]:
        result["note"] = (f"Only the beginning is shown: read the rest with get_media_details on media_id, side_file .txt and start_line {shown.count(chr(10)) + 2} "
                          "(one turn, or word, per line).")
    return result
