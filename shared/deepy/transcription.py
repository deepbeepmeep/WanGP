# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import gc
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

import torch
import whisper
from torch.utils._python_dispatch import TorchDispatchMode
from mmgp import offload

from shared.deepy import video_tools as deepy_video_tools
from shared.deepy.audio_inspection import speech_intervals
from shared.utils.audio_video import extract_audio_track_to_wav
from shared.utils.virtual_media import media_source_exists, parse_virtual_media_path
from shared.deepy.assets import (
    WHISPER_MEDIUM_CONFIG_FILENAME,
    WHISPER_MEDIUM_DOWNLOAD_DEF,
    WHISPER_MEDIUM_FOLDER,
    WHISPER_MEDIUM_REQUIRED_FILES,
    WHISPER_MEDIUM_WEIGHTS_FILENAME,
    WHISPER_LARGE_V3_DOWNLOAD_DEF,
    WHISPER_LARGE_V3_FOLDER,
    whisper_model_name,
)
from shared.ffmpeg_setup import download_ffmpeg
from shared.utils import files_locator as fl


_TEMP_ROOT = Path(__file__).resolve().parents[2] / "_temp_codex" / "deepy_transcribe"
_TIMESTAMP_TYPE_ALIASES = {
    "none": None,
    "off": None,
    "disabled": None,
    "segment": "segment",
    "segments": "segment",
    "word": "word",
    "words": "word",
}
_WHISPER_MEDIUM_REQUIRED_FILES = WHISPER_MEDIUM_REQUIRED_FILES


def normalize_timestamp_type(value: Any) -> str | None:
    normalized = str(value or "").strip().lower()
    if len(normalized) == 0:
        return "segment"
    if normalized not in _TIMESTAMP_TYPE_ALIASES:
        raise ValueError("timestamp_type must be 'segment', 'word', or 'none'.")
    return _TIMESTAMP_TYPE_ALIASES[normalized]


def _whisper_medium_files_present(model_dir: Path | None) -> bool:
    if model_dir is None or not model_dir.is_dir():
        return False
    return all((model_dir / filename).is_file() for filename in _WHISPER_MEDIUM_REQUIRED_FILES)


def _ensure_whisper_medium_assets(model_dir: Path | None = None, gen=None) -> None:
    if _whisper_medium_files_present(model_dir):
        return
    from shared.utils.download import process_files_def
    process_files_def(**WHISPER_MEDIUM_DOWNLOAD_DEF, gen=gen)


def _whisper_medium_dir(gen=None) -> Path:
    located = fl.locate_folder(WHISPER_MEDIUM_FOLDER, error_if_none=False)
    located_path = None if located is None else Path(located).resolve()
    _ensure_whisper_medium_assets(located_path, gen)
    located = fl.locate_folder(WHISPER_MEDIUM_FOLDER, error_if_none=False)
    if located is not None:
        resolved = Path(located).resolve()
        if _whisper_medium_files_present(resolved):
            return resolved
    raise FileNotFoundError(
        f"Unable to locate the Whisper medium folder '{WHISPER_MEDIUM_FOLDER}' in the configured checkpoints paths."
    )


def load_whisper(model_name: str, device: torch.device, **kwargs) -> whisper.Whisper:
    """A Whisper model by name (whisper_model_name() gives the configured one); kwargs: check_cancelled, gen."""
    return (_load_whisper_medium if model_name == "medium" else _load_whisper_large_v3)(device, **kwargs)


def _load_whisper_medium(device: torch.device, *, check_cancelled=lambda: None, gen=None) -> whisper.Whisper:
    check_cancelled()
    model_dir = _whisper_medium_dir(gen)
    config_path = model_dir / WHISPER_MEDIUM_CONFIG_FILENAME
    weights_path = model_dir / WHISPER_MEDIUM_WEIGHTS_FILENAME
    if not config_path.is_file():
        raise FileNotFoundError(f"Whisper config file not found: {config_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(f"Whisper weights file not found: {weights_path}")
    with config_path.open("r", encoding="utf-8") as reader:
        config = json.load(reader)
    alignment_heads = str(config.get("alignment_heads", "") or "").strip()
    return _load_whisper(config["dims"], weights_path, device, alignment_heads.encode("ascii") if alignment_heads else None, check_cancelled=check_cancelled)


class _SkipRandomInitialization(TorchDispatchMode):
    """Skip overwritten random weights in this thread, leaving buffers intact."""
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in (torch.ops.aten.uniform_.default, torch.ops.aten.normal_.default):
            return args[0]
        return func(*args, **(kwargs or {}))


def _load_whisper(dims, weights_path, device, alignment_heads, preprocess_sd=None, check_cancelled=lambda: None):
    check_cancelled()
    with _SkipRandomInitialization():
        model = whisper.model.Whisper(whisper.model.ModelDimensions(**dims))
    check_cancelled()
    # The FP16 checkpoints keep their projections in FP16: Whisper casts them to the input dtype at each call, as it would FP32 copies
    # of the same values. LayerNorm computes in FP32 with FP32 weights.
    offload.load_model_data(model, str(weights_path), writable_tensors=False, default_dtype=None, preprocess_sd=preprocess_sd)
    check_cancelled()
    for module in model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            module.float()
    if alignment_heads:
        model.set_alignment_heads(alignment_heads)
    model.eval()
    return model.to(device=device)


def _whisper_on(model, device):
    """Whisper ready to run on `device`: its FP16 weights on CUDA; FP32 on the CPU, converted once instead of at every call."""
    return model.to(device=device) if device.type == "cuda" else model.to(device=device, dtype=torch.float32)


def _large_v3_state_dict(state_dict):
    replacements = (
        ("model.", ""), (".layers.", ".blocks."), (".self_attn_layer_norm.", ".attn_ln."), (".encoder_attn_layer_norm.", ".cross_attn_ln."),
        (".self_attn.", ".attn."), (".encoder_attn.", ".cross_attn."), (".q_proj.", ".query."), (".k_proj.", ".key."), (".v_proj.", ".value."), (".out_proj.", ".out."),
        (".fc1.", ".mlp.0."), (".fc2.", ".mlp.2."), (".final_layer_norm.", ".mlp_ln."), ("encoder.layer_norm.", "encoder.ln_post."), ("decoder.layer_norm.", "decoder.ln."),
        (".embed_positions.weight", ".positional_embedding"), (".embed_tokens.", ".token_embedding."),
    )
    converted = {}
    for name, tensor in state_dict.items():
        for old, new in replacements:
            name = name.replace(old, new)
        converted[name] = tensor
    return converted


def _load_whisper_large_v3(device, *, check_cancelled=lambda: None, gen=None):
    from shared.utils.download import process_files_def

    check_cancelled()
    folder = fl.locate_folder(WHISPER_LARGE_V3_FOLDER, error_if_none=False)
    if not _whisper_medium_files_present(Path(folder) if folder else None):
        process_files_def(**WHISPER_LARGE_V3_DOWNLOAD_DEF, gen=gen)
    check_cancelled()
    config_path = fl.locate_file(f"{WHISPER_LARGE_V3_FOLDER}/config.json")
    weights_path = fl.locate_file(f"{WHISPER_LARGE_V3_FOLDER}/model.safetensors")
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    dims = dict(
        n_mels=config["num_mel_bins"], n_audio_ctx=config["max_source_positions"], n_audio_state=config["d_model"], n_audio_head=config["encoder_attention_heads"], n_audio_layer=config["encoder_layers"],
        n_vocab=config["vocab_size"], n_text_ctx=config["max_target_positions"], n_text_state=config["d_model"], n_text_head=config["decoder_attention_heads"], n_text_layer=config["decoder_layers"],
    )
    return _load_whisper(dims, weights_path, device, whisper._ALIGNMENT_HEADS["large-v3"], _large_v3_state_dict, check_cancelled)


def _make_temp_audio_path() -> Path:
    _TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    return (_TEMP_ROOT / f"{uuid.uuid4().hex}.wav").resolve()


def _prepare_audio_input(source_path: str, audio_track_no: int | None = None, duration_seconds: float | None = None) -> tuple[str, list[Path]]:
    download_ffmpeg()
    temp_audio_path = _make_temp_audio_path()
    if parse_virtual_media_path(source_path) is None:
        deepy_video_tools.extract_audio(source_path, str(temp_audio_path), audio_track_no=audio_track_no, audio_codec="wav", duration=duration_seconds)
        return str(temp_audio_path), [temp_audio_path]
    part_path = _make_temp_audio_path()  # a part of a file (a view, or a |start_s=... suffix): its own sound first
    extract_audio_track_to_wav(source_path, str(part_path))
    deepy_video_tools.extract_audio(str(part_path), str(temp_audio_path), audio_codec="wav", duration=duration_seconds)
    return str(temp_audio_path), [temp_audio_path, part_path]


def _round_timestamp(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return round(float(value), 3)
    except Exception:
        return None


def _tighten(start: float | None, end: float | None, speech: list[tuple[float, float]], min_pause: float = 0.15) -> tuple[float | None, float | None]:
    """Whisper attaches a pause to the following word: move each bound past a pause of at least min_pause inside the span."""
    inside = [(max(a, start), min(b, end)) for a, b in speech if start is not None and end is not None and b > start and a < end]
    if not inside:
        return start, end
    return (inside[0][0] if inside[0][0] - start >= min_pause else start), (inside[-1][1] if end - inside[-1][1] >= min_pause else end)


def segments_to_srt(segments: list[dict[str, Any]]) -> str:
    """SubRip subtitles from serialized transcription segments (seconds)."""
    stamp = lambda seconds: (lambda ms: f"{ms // 3600000:02}:{ms // 60000 % 60:02}:{ms // 1000 % 60:02},{ms % 1000:03}")(round(float(seconds) * 1000))
    cues = [segment for segment in segments if segment["text"] and segment["start"] is not None and segment["end"] is not None]
    return "\n".join(f"{index}\n{stamp(cue['start'])} --> {stamp(cue['end'])}\n{cue['text']}\n" for index, cue in enumerate(cues, 1))


def _serialize_segments(segments: list[dict[str, Any]], timestamp_type: str | None, speech: list[tuple[float, float]]) -> list[dict[str, Any]]:
    serialized = []
    include_words = timestamp_type == "word"
    for segment in segments:
        start, end = _tighten(_round_timestamp(segment.get("start", None)), _round_timestamp(segment.get("end", None)), speech)
        item = {
            "start": start,
            "end": end,
            "text": str(segment.get("text", "") or "").strip(),
        }
        if include_words:
            words = []
            for word in list(segment.get("words", []) or []):
                word_start, word_end = _tighten(_round_timestamp(word.get("start", None)), _round_timestamp(word.get("end", None)), speech)
                words.append(
                    {
                        "word": str(word.get("word", "") or ""),
                        "start": word_start,
                        "end": word_end,
                        "probability": None if word.get("probability", None) is None else round(float(word["probability"]), 4),
                    }
                )
            if len(words) > 0:
                item["words"] = words
        serialized.append(item)
    return serialized


def transcribe_media(source_path: str, *, timestamp_type: str | None = None, audio_track_no: int | None = None, device: str | None = None, model_name: str | None = None, language: str | None = None, prepared_model: list | None = None, check_cancelled=lambda: None, duration_seconds: float | None = None, condition_on_previous_text: bool = True) -> dict[str, Any]:
    normalized_timestamp_type = normalize_timestamp_type(timestamp_type)
    source_path = str(source_path or "").strip()
    if len(source_path) == 0 or not media_source_exists(source_path):
        raise FileNotFoundError(f"Media file not found: {source_path}")
    device = torch.device(device if device is not None else "cuda" if torch.cuda.is_available() else "cpu")
    model_name = model_name or whisper_model_name()
    started = time.perf_counter()
    audio_path, temporary_paths = _prepare_audio_input(source_path, audio_track_no=audio_track_no, duration_seconds=duration_seconds)
    prepared = time.perf_counter()
    model = None
    cancellation_hooks = []
    try:
        check_cancelled()
        if prepared_model is None:
            model = load_whisper(model_name, device)
        else:
            model = prepared_model.pop()
            check_cancelled()
        model = _whisper_on(model, device)
        check_cancelled()
        if prepared_model is not None:
            # Check at encoder/decoder boundaries, including each decoded token.
            # Hooks belong only to this recording's model, never to TTS models.
            for module in (model.encoder, model.decoder):
                cancellation_hooks.append(module.register_forward_pre_hook(lambda module, inputs: check_cancelled()))
        loaded = time.perf_counter()
        print(f"[Whisper {model_name}] Loaded on {model.device}; transcribing...", flush=True)
        raw_result = model.transcribe(
            audio_path,
            verbose=None,
            fp16=device.type == "cuda",
            word_timestamps=normalized_timestamp_type == "word",
            language=language,
            condition_on_previous_text=condition_on_previous_text,
        )
        check_cancelled()
        speech = speech_intervals(audio_path) if normalized_timestamp_type is not None else []
        transcribed = time.perf_counter()
        print(f"[Whisper {model_name}] Audio preparation {prepared - started:.2f}s; model loading {loaded - prepared:.2f}s; transcription {transcribed - loaded:.2f}s.", flush=True)
    finally:
        for hook in cancellation_hooks:
            hook.remove()
        for temporary_path in temporary_paths:
            try:
                temporary_path.unlink(missing_ok=True)
            except Exception:
                pass
        if model is not None:
            del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    segments = list(raw_result.get("segments", []) or [])
    payload = {
        "text": str(raw_result.get("text", "") or "").strip(),
        "language": str(raw_result.get("language", "") or "").strip(),
        "segment_count": int(len(segments)),
    }
    if normalized_timestamp_type is not None:
        payload["timestamp_type"] = normalized_timestamp_type
        payload["segments"] = _serialize_segments(segments, normalized_timestamp_type, speech)
        payload["speech_intervals"] = [list(span) for span in speech[:300]]
    return payload
