import copy
import html
import os
import random
import re
import unicodedata
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np
import torch
import torchaudio
from accelerate import init_empty_weights
from mmgp import offload
from mmgp import offload as mmgp_offload
from tqdm import tqdm

from shared.utils import files_locator as fl
from shared.utils.audio_cleaning import ensure_trailing_silence, mute_isolated_transient_noise, trim_after_silence_boundary, trim_leading_noise_before_speech, trim_leading_transient_noise, trim_trailing_transient_noise
from shared.utils.text_encoder_cache import TextEncoderCache

from .ltx2 import _VAEContainer, _load_config_from_checkpoint, _make_sd_postprocess, _make_vae_postprocess
from .ltx_core.components.diffusion_steps import EulerDiffusionStep
from .ltx_core.components.noisers import GaussianNoiser
from .ltx_core.conditioning import AudioConditionByReferenceLatent
from .ltx_core.model.audio_vae import (
    VOCODER_COMFY_KEYS_FILTER,
    AudioDecoderConfigurator,
    AudioEncoderConfigurator,
    AudioProcessor,
    VocoderConfigurator,
    decode_audio,
)
from .ltx_core.model.transformer import LTXAudioOnlyModelConfigurator, LTXV_MODEL_COMFY_RENAMING_MAP, X0Model
from .ltx_core.text_encoders.gemma import (
    TEXT_EMBEDDING_PROJECTION_KEY_OPS,
    TEXT_EMBEDDINGS_CONNECTOR_KEY_OPS,
    GemmaTextEmbeddingsConnectorModelConfigurator,
    build_gemma_text_encoder,
    encode_text,
    postprocess_text_embeddings,
    resolve_text_connectors,
)
from .ltx_core.text_encoders.gemma.feature_extractor import GemmaFeaturesExtractorProjLinear
from .ltx_core.tools import AudioLatentTools
from .ltx_core.types import AudioLatentShape, VideoPixelShape
from .ltx_audio_tts import LTXAudioTTSPipelineBase
from .ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE, DISTILLED_SIGMA_VALUES
from .ltx_pipelines.utils.helpers import (
    _prepare_conditioning_context,
    modality_from_latent_state,
    post_process_latent,
    state_with_conditionings,
)
from .ltx_pipelines.utils.types import PipelineComponents


SCENEMA_DEFAULT_VOICE = "Natural expressive voice"
SCENEMA_DEFAULT_SCENE = "a person speaking to camera"
SCENEMA_MAX_DURATION_SECONDS = 20.0
SCENEMA_DEFAULT_TOTAL_DURATION_SECONDS = 120.0
SCENEMA_MAX_TOTAL_DURATION_SECONDS = 30.0 * 60.0
SCENEMA_MAX_CHUNK_DURATION_SECONDS = 15.0
SCENEMA_MAX_REF_SECONDS = 121.0 / 25.0
SCENEMA_FPS = 24.0
SCENEMA_DEFAULT_PACE = 1.5
SCENEMA_ACTION_DURATION_SECONDS = 1.5
SCENEMA_FALLBACK_WORDS_PER_SECOND = 2.2
SCENEMA_REF_TAIL_SECONDS = 3.0
SCENEMA_BOUNDARY_PUNCTUATION = ".!?"
SCENEMA_TRIM_EXTRA_WORDS = True
SCENEMA_DEBUG_PROMPT = False
SCENEMA_ALIGNMENT_SILENCE_THRESHOLD = 0.015
SCENEMA_TRANSIENT_SILENCE_THRESHOLD = 0.006
SCENEMA_ISOLATED_TRANSIENT_THRESHOLD = 0.01
SCENEMA_TRANSIENT_MAX_SECONDS = 0.18
SCENEMA_LEADING_TRANSIENT_MAX_SECONDS = 0.30
SCENEMA_CHUNK_TAIL_SILENCE_SECONDS = 0.5
SCENEMA_LEADING_SPEECH_THRESHOLD = 0.03
SCENEMA_CHUNK_DURATION_HEADROOM_SECONDS = 4.0
SCENEMA_CHUNK_BOUNDARY_WORD_PADDING_SECONDS = 0.25
SCENEMA_CHUNK_BOUNDARY_MIN_SILENCE_SECONDS = 0.18
SCENEMA_CHUNK_BOUNDARY_KEEP_SILENCE_SECONDS = 0.12
SCENEMA_KOKORO_FOLDER = "kokoro"
SCENEMA_KOKORO_VOICE = "af_heart.pt"


@dataclass
class _CompiledPrompt:
    prompt: str
    speech_text: str
    voice: str
    scene: str | None
    language: str
    gender: str
    shot: str
    explicit_gender: bool


@dataclass
class _ChunkSpec:
    compiled_prompt: str
    duration_s: float
    seed: int
    expected_text: str
    speaker: int = 1
    language: str = "en"


@dataclass
class _PromptBlock:
    xml_text: str
    speaker: int = 1


@dataclass
class _TextBlock:
    text: str


@dataclass
class _ActionBlock:
    text: str


@dataclass
class _SoundBlock:
    text: str


def _read_text_or_file(value, label: str) -> str:
    if value is None:
        return ""
    text = os.fspath(value) if isinstance(value, os.PathLike) else str(value)
    if os.path.isfile(text) and os.path.splitext(text)[1].lower() in {".txt", ".xml"}:
        with open(text, "r", encoding="utf-8") as reader:
            return reader.read()
    return text


def _clean_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _ensure_sentence(text: str) -> str:
    text = _clean_spaces(text)
    if text and text[-1] not in ".!?\"'":
        return text + "."
    return text


def _voice_with_explicit_gender(voice: str, gender: str, explicit_gender: bool) -> str:
    voice = _clean_spaces(voice)
    if not explicit_gender:
        return voice
    gender = _clean_spaces(gender).lower()
    if gender == "female":
        pattern = r"\b(female|woman|women|girl|she|her)\b"
        prefix = "Female voice"
    elif gender == "male":
        pattern = r"\b(male|man|men|boy|he|his)\b"
        prefix = "Male voice"
    else:
        return voice
    if re.search(pattern, voice, flags=re.IGNORECASE):
        return voice
    return _clean_spaces(f"{prefix}, {voice}") if voice else prefix


def _speaker_intro_for_gender(gender: str) -> str:
    gender = _clean_spaces(gender).lower()
    if gender == "female":
        return "A woman says"
    if gender == "male":
        return "A man says"
    return "The speaker says"


def _normalize_speaker_id(value) -> int:
    try:
        text = str(value if value is not None else "1")
        match = re.search(r"\d+", text)
        return max(1, int(match.group(0))) if match else 1
    except Exception:
        return 1


def _escape_xml_text(text: str) -> str:
    return html.escape(str(text or ""), quote=False)


def _format_xml_attrs(attrs: dict) -> str:
    parts = []
    for key, value in attrs.items():
        value = _clean_spaces(value)
        if value:
            parts.append(f'{key}="{html.escape(value, quote=True)}"')
    return " ".join(parts)


def _parse_speaker_options(raw_options: str | None) -> dict:
    if not raw_options:
        return {}
    text = raw_options.strip()
    if text.startswith("{") and text.endswith("}"):
        text = text[1:-1]
    options = {}
    pattern = re.compile(r"([A-Za-z_][\w-]*)\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^,\s}]+))")
    for match in pattern.finditer(text):
        key = match.group(1).strip().lower()
        if key not in {"voice", "gender", "scene", "shot", "language"}:
            continue
        value = next(group for group in match.groups()[1:] if group is not None)
        options[key] = _clean_spaces(value)
    return options


def _speaker_section_to_xml(section_text: str, attrs: dict, speaker: int) -> str:
    xml_attrs = {"speaker": str(speaker)}
    for key in ("voice", "gender", "scene", "shot", "language"):
        if _clean_spaces(attrs.get(key) or ""):
            xml_attrs[key] = attrs[key]
    pieces = [f"<speak {_format_xml_attrs(xml_attrs)}>"]
    split_parts = re.split(r"\[([^\]]+)\]", str(section_text or ""))
    for index, part in enumerate(split_parts):
        part = _clean_spaces(part)
        if not part:
            continue
        if index % 2:
            pieces.append(f"<action>{_escape_xml_text(part)}</action>")
        else:
            pieces.append(_escape_xml_text(part))
    pieces.append("</speak>")
    return " ".join(pieces)


def _looks_like_wangp_speaker_prompt(text: str) -> bool:
    return re.search(r"(?im)^\s*Speaker\s*\d+\s*(?:\{[^\n{}]*\})?\s*:", text or "") is not None


def _parse_wangp_speaker_prompt(text: str, voice_instruction: str | None = None) -> list[_PromptBlock]:
    header_pattern = re.compile(r"(?im)^\s*Speaker\s*(\d+)\s*(\{[^\n{}]*\})?\s*:\s*")
    matches = list(header_pattern.finditer(text or ""))
    if not matches:
        return []
    speaker_attrs: dict[int, dict] = {}
    blocks: list[_PromptBlock] = []
    for idx, match in enumerate(matches):
        speaker = _normalize_speaker_id(match.group(1))
        section_start = match.end()
        section_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        section = text[section_start:section_end].strip()
        if not section:
            continue
        attrs = speaker_attrs.setdefault(speaker, {})
        parsed_options = _parse_speaker_options(match.group(2))
        if parsed_options:
            attrs.update(parsed_options)
        blocks.append(_PromptBlock(xml_text=_speaker_section_to_xml(section, attrs, speaker), speaker=speaker))
    return blocks


def _element_to_speak_xml(root: ET.Element, voice_instruction: str | None = None) -> _PromptBlock:
    speaker = _normalize_speaker_id(root.get("speaker"))
    element = copy.deepcopy(root)
    element.attrib.pop("speaker", None)
    return _PromptBlock(xml_text=ET.tostring(element, encoding="unicode"), speaker=speaker)


def _parse_xml_prompt_blocks(text: str, voice_instruction: str | None = None) -> list[_PromptBlock]:
    raw = str(text or "").strip()
    if "<speak" not in raw.lower():
        return []
    try:
        root = ET.fromstring(raw)
    except ET.ParseError:
        root = ET.fromstring(f"<dialogue>{raw}</dialogue>")
    if root.tag.lower() == "speak":
        return [_element_to_speak_xml(root, voice_instruction)]
    blocks = [_element_to_speak_xml(child, voice_instruction) for child in root if child.tag.lower() == "speak"]
    if not blocks:
        raise ValueError("Scenema XML dialogue prompts must contain at least one <speak> block.")
    return blocks


def _plain_text_to_speak_xml(text: str, voice_instruction: str | None = None) -> str:
    attrs = {
        "voice": _clean_spaces(voice_instruction or "") or SCENEMA_DEFAULT_VOICE,
        "scene": SCENEMA_DEFAULT_SCENE,
        "shot": "closeup",
        "language": "en",
    }
    return _speaker_section_to_xml(text, attrs, speaker=1)


def _prepare_prompt_blocks(text: str, voice_instruction: str | None = None) -> list[_PromptBlock]:
    raw = str(text or "").strip()
    if _looks_like_wangp_speaker_prompt(raw):
        blocks = _parse_wangp_speaker_prompt(raw, voice_instruction)
        if blocks:
            return blocks
    xml_blocks = _parse_xml_prompt_blocks(raw, voice_instruction)
    if xml_blocks:
        return xml_blocks
    return [_PromptBlock(xml_text=_plain_text_to_speak_xml(raw, voice_instruction), speaker=1)]


def _strip_speaker_attr(xml_text: str) -> str:
    root = ET.fromstring(xml_text)
    if root.tag.lower() == "speak" and "speaker" in root.attrib:
        root.attrib.pop("speaker", None)
        return ET.tostring(root, encoding="unicode")
    return xml_text


def _extract_speak_blocks(root: ET.Element):
    blocks = []
    if root.text and root.text.strip():
        blocks.append(_TextBlock(_clean_spaces(root.text)))
    for child in root:
        tag = child.tag.lower()
        if child.text and child.text.strip():
            if tag == "action":
                blocks.append(_ActionBlock(_clean_spaces(child.text)))
            elif tag == "sound":
                blocks.append(_SoundBlock(_clean_spaces(child.text)))
        if child.tail and child.tail.strip():
            blocks.append(_TextBlock(_clean_spaces(child.tail)))
    return blocks


def _compile_speak_xml(xml_text: str, voice_override: str | None = None) -> _CompiledPrompt:
    root = ET.fromstring(xml_text)
    if root.tag.lower() != "speak":
        raise ValueError("Scenema XML prompts must use a <speak> root element.")

    explicit_gender = root.get("gender") is not None
    scene = _clean_spaces(root.get("scene") or SCENEMA_DEFAULT_SCENE)
    gender = _clean_spaces(root.get("gender") or "male").lower()
    voice = _voice_with_explicit_gender(root.get("voice") or voice_override or SCENEMA_DEFAULT_VOICE, gender, explicit_gender)
    shot = _clean_spaces(root.get("shot") or "closeup").lower()
    shot_prefix = {"closeup": "Close-up in", "wide": "Wide shot of", "scene": ""}.get(shot, "Close-up in")
    scene_mode = shot in {"wide", "scene"}
    pronoun = "She" if gender == "female" else "He"

    parts = [f"{shot_prefix} {scene}." if shot_prefix else f"{scene}."]
    blocks = _extract_speak_blocks(root)
    has_action = any(isinstance(block, _ActionBlock) for block in blocks)
    speaker_intro = _speaker_intro_for_gender(gender) if explicit_gender and not scene_mode else ""
    first_speech = True
    speech_texts = []
    for block in blocks:
        if isinstance(block, _SoundBlock):
            parts.append(_ensure_sentence(block.text))
        elif isinstance(block, _ActionBlock):
            parts.append(_clean_spaces(block.text) + ":" if scene_mode else _ensure_sentence(block.text))
        else:
            speech = _ensure_sentence(block.text)
            speech_texts.append(speech)
            if scene_mode and first_speech and not has_action:
                parts.append(f'{pronoun} speaks: "{speech}"')
            elif speaker_intro and first_speech:
                parts.append(f'{speaker_intro}: "{speech}"')
            else:
                parts.append(f'"{speech}"')
            first_speech = False

    parts.append(_ensure_sentence(voice))
    if scene_mode and scene:
        parts.append(_ensure_sentence(scene))
    return _CompiledPrompt(
        prompt=" ".join(part for part in parts if part),
        speech_text=" ".join(speech_texts),
        voice=voice,
        scene=scene,
        language=_clean_spaces(root.get("language") or "en"),
        gender=gender,
        shot=shot,
        explicit_gender=explicit_gender,
    )


def compile_scenema_prompt(text: str, voice_instruction: str | None = None) -> _CompiledPrompt:
    raw = str(text or "").strip()
    voice = _clean_spaces(voice_instruction or "")
    if raw.lstrip().startswith("<speak"):
        return _compile_speak_xml(raw, voice_override=voice)

    speech = _ensure_sentence(raw)
    voice = voice or SCENEMA_DEFAULT_VOICE
    prompt = f'Close-up in {SCENEMA_DEFAULT_SCENE}. "{speech}" {_ensure_sentence(voice)}'
    return _CompiledPrompt(
        prompt=prompt,
        speech_text=speech,
        voice=voice,
        scene=SCENEMA_DEFAULT_SCENE,
        language="en",
        gender="male",
        shot="closeup",
        explicit_gender=False,
    )


def _kokoro_voice_path() -> str:
    return fl.locate_file(os.path.join(SCENEMA_KOKORO_FOLDER, "voices", SCENEMA_KOKORO_VOICE))


def _kokoro_duration(text: str, kokoro_pipeline=None) -> float | None:
    text = _clean_spaces(text)
    if not text:
        return 0.0
    if kokoro_pipeline is None:
        return None
    voice_path = _kokoro_voice_path()
    try:
        total_frames = 0
        for result in kokoro_pipeline(text, voice=voice_path):
            audio = getattr(result, "audio", None)
            if audio is None and isinstance(result, (tuple, list)) and len(result) >= 3:
                audio = result[2]
            if audio is not None:
                total_frames += int(len(audio))
        return total_frames / 24000.0
    except Exception:
        return None


def _split_into_sentences(text: str) -> list[str]:
    sentences = []
    current = ""
    for char in str(text or ""):
        current += char
        if char in SCENEMA_BOUNDARY_PUNCTUATION:
            stripped = current.strip()
            if stripped:
                sentences.append(stripped)
            current = ""
    if current.strip():
        sentences.append(current.strip())
    return sentences


def _estimate_sentence_durations(sentences: list[str], kokoro_pipeline=None) -> list[float]:
    durations = []
    for sentence in sentences:
        duration = _kokoro_duration(sentence, kokoro_pipeline)
        if duration is None:
            duration = len(sentence.split()) / SCENEMA_FALLBACK_WORDS_PER_SECOND + 0.3
        durations.append(duration)
    return durations


def _split_text_by_duration(text: str, multiplier: float, max_duration: float = SCENEMA_MAX_CHUNK_DURATION_SECONDS, kokoro_pipeline=None) -> list[tuple[str, float]]:
    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    expanded = []
    for sentence in sentences:
        duration = _estimate_sentence_durations([sentence], kokoro_pipeline)[0]
        if duration * multiplier > max_duration and "," in sentence:
            clauses = [clause.strip() for clause in sentence.split(",") if clause.strip()]
            clause_durations = _estimate_sentence_durations(clauses, kokoro_pipeline)
            sub_texts = []
            sub_duration = 0.0
            for clause, clause_duration in zip(clauses, clause_durations):
                if sub_texts and (sub_duration + clause_duration) * multiplier > max_duration:
                    expanded.append(", ".join(sub_texts))
                    sub_texts = []
                    sub_duration = 0.0
                sub_texts.append(clause)
                sub_duration += clause_duration
            if sub_texts:
                expanded.append(", ".join(sub_texts))
        else:
            expanded.append(sentence)

    durations = _estimate_sentence_durations(expanded, kokoro_pipeline)
    chunks = []
    current_texts = []
    current_duration = 0.0
    for sentence, duration in zip(expanded, durations):
        if current_texts and (current_duration + duration) * multiplier > max_duration:
            chunks.append((" ".join(current_texts), min(current_duration * multiplier, max_duration)))
            current_texts = []
            current_duration = 0.0
        current_texts.append(sentence)
        current_duration += duration
    if current_texts:
        chunks.append((" ".join(current_texts), min(current_duration * multiplier, max_duration)))
    return chunks


def _extract_sentence_actions(xml_text: str) -> dict[int, list[str]]:
    root = ET.fromstring(xml_text)
    blocks = _extract_speak_blocks(root)
    sentence_actions = {}
    pending_actions = []
    sentence_idx = 0
    for block in blocks:
        if isinstance(block, _ActionBlock):
            pending_actions.append(block.text)
        elif isinstance(block, _TextBlock):
            sentences = _split_into_sentences(block.text)
            if pending_actions and sentences:
                sentence_actions[sentence_idx] = pending_actions.copy()
                pending_actions.clear()
            sentence_idx += len(sentences)
    return sentence_actions


def _compile_chunk_prompt(
    speech_text: str,
    voice: str,
    scene: str | None = None,
    actions_before: list[str] | None = None,
    gender: str = "male",
    shot: str = "closeup",
    explicit_gender: bool = False,
) -> str:
    attrs = {
        "voice": voice or SCENEMA_DEFAULT_VOICE,
        "scene": scene or SCENEMA_DEFAULT_SCENE,
        "shot": shot or "closeup",
        "language": "en",
    }
    if explicit_gender:
        attrs["gender"] = gender or "male"
    pieces = [f"<speak {_format_xml_attrs(attrs)}>"]
    for action in actions_before or []:
        pieces.append(f"<action>{_escape_xml_text(action)}</action>")
    pieces.append(_escape_xml_text(speech_text))
    pieces.append("</speak>")
    return _compile_speak_xml(" ".join(pieces)).prompt


def _plan_xml_chunks(xml_text: str, speaker: int, base_seed: int, pace: float, kokoro_pipeline=None) -> list[_ChunkSpec]:
    xml_text = _strip_speaker_attr(xml_text)
    compiled = compile_scenema_prompt(xml_text)
    if not compiled.speech_text.strip():
        raise ValueError("Scenema <speak> blocks must contain speech text.")

    kokoro_duration = _kokoro_duration(compiled.speech_text, kokoro_pipeline)
    if kokoro_duration is not None:
        total_duration = kokoro_duration * pace
    else:
        total_duration = (len(compiled.speech_text.split()) / SCENEMA_FALLBACK_WORDS_PER_SECOND + 0.5) * pace

    if total_duration <= SCENEMA_MAX_CHUNK_DURATION_SECONDS:
        return [
            _ChunkSpec(
                compiled_prompt=compiled.prompt,
                duration_s=max(0.5, min(total_duration, SCENEMA_MAX_CHUNK_DURATION_SECONDS)),
                seed=base_seed,
                expected_text=compiled.speech_text,
                speaker=speaker,
                language=compiled.language,
            )
        ]

    sentence_action_map = _extract_sentence_actions(xml_text)
    text_chunks = _split_text_by_duration(compiled.speech_text, multiplier=pace, kokoro_pipeline=kokoro_pipeline)
    global_sentence_idx = 0
    specs = []
    for chunk_idx, (chunk_text, chunk_duration) in enumerate(text_chunks):
        actions_before = sentence_action_map.get(global_sentence_idx)
        chunk_prompt = _compile_chunk_prompt(
            speech_text=chunk_text,
            voice=compiled.voice,
            scene=compiled.scene,
            actions_before=actions_before,
            gender=compiled.gender,
            shot=compiled.shot,
            explicit_gender=compiled.explicit_gender,
        )
        specs.append(
            _ChunkSpec(
                compiled_prompt=chunk_prompt,
                duration_s=max(0.5, min(chunk_duration, SCENEMA_MAX_CHUNK_DURATION_SECONDS)),
                seed=base_seed + chunk_idx * 1000,
                expected_text=chunk_text,
                speaker=speaker,
                language=compiled.language,
            )
        )
        global_sentence_idx += len(_split_into_sentences(chunk_text))
    return specs


def _plan_prompt_chunks(text: str, voice_instruction: str | None, base_seed: int, pace: float, debug_prompt: bool = False, kokoro_pipeline=None) -> list[_ChunkSpec]:
    blocks = _prepare_prompt_blocks(text, voice_instruction)
    chunks = []
    for block_index, block in enumerate(blocks):
        compiler_xml = _strip_speaker_attr(block.xml_text)
        if debug_prompt:
            print(f"[Scenema Audio] XML block {block_index + 1}/{len(blocks)} (Speaker {block.speaker}):\n{compiler_xml}")
        planned = _plan_xml_chunks(compiler_xml, block.speaker, base_seed + len(chunks) * 1000, pace, kokoro_pipeline=kokoro_pipeline)
        if debug_prompt:
            for chunk_index, chunk in enumerate(planned):
                print(f"[Scenema Audio] Compiled chunk {chunk_index + 1}/{len(planned)} (Speaker {chunk.speaker}, {chunk.duration_s:.2f}s): {chunk.compiled_prompt}")
                print(f"[Scenema Audio] Expected speech: {chunk.expected_text}")
        chunks.extend(planned)
    return chunks


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _coerce_duration(duration_seconds: Optional[float]) -> float:
    try:
        duration = float(duration_seconds) if duration_seconds is not None else SCENEMA_DEFAULT_TOTAL_DURATION_SECONDS
    except (TypeError, ValueError):
        duration = SCENEMA_DEFAULT_TOTAL_DURATION_SECONDS
    return max(1.0, min(SCENEMA_MAX_TOTAL_DURATION_SECONDS, duration))


def _limit_chunks_to_duration(chunks: list[_ChunkSpec], duration_seconds: Optional[float]) -> list[_ChunkSpec]:
    duration_limit = _coerce_duration(duration_seconds)
    limited = []
    elapsed = 0.0
    for chunk in chunks:
        remaining = duration_limit - elapsed
        if remaining <= 0:
            break
        if chunk.duration_s <= remaining:
            limited.append(chunk)
            elapsed += chunk.duration_s
        elif not limited:
            limited.append(replace(chunk, duration_s=max(0.5, remaining)))
            break
        else:
            break
    return limited


def _duration_to_frames(duration: float, fps: float = SCENEMA_FPS) -> int:
    return ((int(duration * fps) + 7) // 8) * 8 + 1


def _model_device_dtype(module: torch.nn.Module, default_device: torch.device, default_dtype: torch.dtype):
    param = next(module.parameters(), None)
    if param is None:
        return default_device, default_dtype
    return param.device, param.dtype


def _audio_tensor_to_numpy(audio: torch.Tensor) -> np.ndarray:
    audio = audio.detach().cpu().float()
    if audio.ndim == 3:
        audio = audio.squeeze(0)
    if audio.ndim == 1:
        return audio.numpy()
    return audio.T.numpy()


def _numpy_to_audio_tensor(audio_np: np.ndarray) -> torch.Tensor:
    audio_np = np.asarray(audio_np, dtype=np.float32)
    if audio_np.ndim == 1:
        return torch.from_numpy(audio_np).unsqueeze(0)
    return torch.from_numpy(audio_np.T.copy())


def _normalize_alignment_words(text: str) -> list[str]:
    text = unicodedata.normalize("NFD", str(text or "").lower())
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = re.sub(r"[^\w\s]", "", text)
    return [word for word in text.split() if word]


def _fuzzy_alignment_match(left: str, right: str) -> bool:
    if left == right:
        return True
    if not left or not right or len(left) < 4 or len(right) < 4:
        return False
    distances = list(range(len(right) + 1))
    for row, left_char in enumerate(left, 1):
        previous = distances[0]
        distances[0] = row
        for col, right_char in enumerate(right, 1):
            current = distances[col]
            distances[col] = previous if left_char == right_char else 1 + min(previous, distances[col], distances[col - 1])
            previous = current
    return 1 - distances[-1] / max(len(left), len(right)) >= 0.5


def _alignment_score(left: str, right: str) -> int:
    return 2 if _fuzzy_alignment_match(left, right) else -1


def _label_transcribed_words(transcribed: list[str], expected: list[str]) -> list[str]:
    gap_score = -1
    rows = len(transcribed)
    cols = len(expected)
    dp = [[0] * (cols + 1) for _ in range(rows + 1)]
    for row in range(1, rows + 1):
        dp[row][0] = dp[row - 1][0] + gap_score
    for col in range(1, cols + 1):
        dp[0][col] = dp[0][col - 1] + gap_score
    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            match = dp[row - 1][col - 1] + _alignment_score(transcribed[row - 1], expected[col - 1])
            delete = dp[row - 1][col] + gap_score
            insert = dp[row][col - 1] + gap_score
            dp[row][col] = max(match, delete, insert)
    labels = []
    row, col = rows, cols
    while row > 0 or col > 0:
        if row > 0 and col > 0 and dp[row][col] == dp[row - 1][col - 1] + _alignment_score(transcribed[row - 1], expected[col - 1]):
            labels.append("match" if _alignment_score(transcribed[row - 1], expected[col - 1]) == 2 else "substitution")
            row -= 1
            col -= 1
        elif row > 0 and dp[row][col] == dp[row - 1][col] + gap_score:
            labels.append("insertion")
            row -= 1
        else:
            col -= 1
    labels.reverse()
    return labels


def _transcribe_words(alignment_whisper: torch.nn.Module, audio_np: np.ndarray, sample_rate: int, language: str) -> list[dict]:
    mono = audio_np.mean(axis=1) if audio_np.ndim == 2 else audio_np
    mono_tensor = torch.from_numpy(mono.astype(np.float32))
    if int(sample_rate) != 16000:
        mono_tensor = torchaudio.functional.resample(mono_tensor.unsqueeze(0), int(sample_rate), 16000).squeeze(0)
    model_dtype = getattr(alignment_whisper, "_model_dtype", next(alignment_whisper.parameters()).dtype)
    with torch.device("cuda" if torch.cuda.is_available() else "cpu"):  # openai-whisper creates its tensors on the default device
        result = alignment_whisper.transcribe(mono_tensor.numpy(), language=language, word_timestamps=True, fp16=model_dtype == torch.float16, verbose=None)
    words = []
    for segment in result.get("segments", []):
        for word in segment.get("words", []) or []:
            normalized = _normalize_alignment_words(word.get("word", ""))
            if normalized:
                words.append({"word": normalized[0], "start": float(word.get("start", 0.0)), "end": float(word.get("end", 0.0))})
    return words


def _find_silence_boundary(audio: np.ndarray, sample_rate: int, center_sample: int, direction: str) -> int:
    hop = max(1, int(0.01 * sample_rate))
    window = int(0.3 * sample_rate)
    positions = range(center_sample, max(0, center_sample - window), -hop) if direction == "left" else range(center_sample, min(len(audio), center_sample + window), hop)
    for position in positions:
        chunk = audio[max(0, position - hop // 2) : min(len(audio), position + hop // 2)]
        if len(chunk) > 0 and np.sqrt(np.mean(chunk.astype(np.float64) ** 2)) < SCENEMA_ALIGNMENT_SILENCE_THRESHOLD:
            return position
    return center_sample


def _trim_leading_extra_words(alignment_whisper: torch.nn.Module, audio_np: np.ndarray, sample_rate: int, expected_text: str, language: str, debug_prompt: bool = False, label: str = "Scenema Audio") -> np.ndarray:
    expected_words = _normalize_alignment_words(expected_text)
    if not expected_words:
        return audio_np
    transcribed = _transcribe_words(alignment_whisper, audio_np, sample_rate, language)
    transcribed_words = [word["word"] for word in transcribed]
    if not transcribed_words:
        return audio_np
    labels = _label_transcribed_words(transcribed_words, expected_words)
    leading_insertions = 0
    for label in labels:
        if label != "insertion":
            break
        leading_insertions += 1
    if leading_insertions == 0:
        return audio_np
    mono = audio_np.mean(axis=1) if audio_np.ndim == 2 else audio_np
    trim_end = _find_silence_boundary(mono, sample_rate, int(transcribed[leading_insertions - 1]["end"] * sample_rate), "right")
    if trim_end <= 0 or trim_end >= len(audio_np):
        return audio_np
    if debug_prompt:
        removed_words = " ".join(word["word"] for word in transcribed[:leading_insertions])
        print(f"[{label}] Trimmed leading extra words ({trim_end / sample_rate:.2f}s): {removed_words}")
    return audio_np[trim_end:]


def _trim_leading_extra_words_tensor(alignment_whisper: torch.nn.Module, audio: torch.Tensor, sample_rate: int, expected_text: str, language: str, debug_prompt: bool = False, label: str = "Scenema Audio") -> torch.Tensor:
    original_device = audio.device
    original_dtype = audio.dtype
    trimmed = _trim_leading_extra_words(alignment_whisper, _audio_tensor_to_numpy(audio), sample_rate, expected_text, language, debug_prompt=debug_prompt, label=label)
    return _numpy_to_audio_tensor(trimmed).to(device=original_device, dtype=original_dtype)


def _trim_chunk_tail_to_silence(alignment_whisper: torch.nn.Module, audio_np: np.ndarray, sample_rate: int, expected_text: str, language: str, debug_prompt: bool = False) -> np.ndarray:
    expected_words = _normalize_alignment_words(expected_text)
    if not expected_words:
        return audio_np
    transcribed = _transcribe_words(alignment_whisper, audio_np, sample_rate, language)
    transcribed_words = [word["word"] for word in transcribed]
    if not transcribed_words:
        return audio_np
    labels = _label_transcribed_words(transcribed_words, expected_words)
    expected_count = 0
    final_expected_index = None
    for idx, label in enumerate(labels):
        if label != "insertion":
            expected_count += 1
            if expected_count == len(expected_words):
                final_expected_index = idx
                break
    if final_expected_index is None:
        if debug_prompt:
            print("[Scenema Audio] Kept chunk tail: final expected word was not confidently located")
        return audio_np
    trailing_words = transcribed[final_expected_index + 1 :]
    if not trailing_words:
        return audio_np
    final_word_end = float(transcribed[final_expected_index]["end"])
    trimmed = trim_after_silence_boundary(
        audio_np,
        sample_rate,
        final_word_end + SCENEMA_CHUNK_BOUNDARY_WORD_PADDING_SECONDS,
        search_seconds=SCENEMA_CHUNK_DURATION_HEADROOM_SECONDS,
        min_silence_seconds=SCENEMA_CHUNK_BOUNDARY_MIN_SILENCE_SECONDS,
        keep_silence_seconds=SCENEMA_CHUNK_BOUNDARY_KEEP_SILENCE_SECONDS,
        threshold=SCENEMA_TRANSIENT_SILENCE_THRESHOLD,
        debug=debug_prompt,
        label="Scenema Audio",
    )
    if debug_prompt and len(trimmed) < len(audio_np):
        trailing_text = " ".join(word["word"] for word in trailing_words)
        if trailing_text:
            print(f"[Scenema Audio] Trimmed trailing extra words: {trailing_text}")
    return trimmed


def _trim_chunk_tail_to_silence_tensor(alignment_whisper: torch.nn.Module, audio: torch.Tensor, sample_rate: int, expected_text: str, language: str, debug_prompt: bool = False) -> torch.Tensor:
    original_device = audio.device
    original_dtype = audio.dtype
    trimmed = _trim_chunk_tail_to_silence(alignment_whisper, _audio_tensor_to_numpy(audio), sample_rate, expected_text, language, debug_prompt=debug_prompt)
    return _numpy_to_audio_tensor(trimmed).to(device=original_device, dtype=original_dtype)


def _trim_silence(audio_np: np.ndarray, sample_rate: int, max_silence: float = 0.5, threshold_db: float = -40.0) -> np.ndarray:
    threshold = 10 ** (threshold_db / 20.0)
    max_silent_samples = int(max_silence * sample_rate)
    window = max(1, int(0.02 * sample_rate))
    mono = audio_np.mean(axis=1) if audio_np.ndim == 2 else audio_np
    if len(mono) < window:
        return audio_np
    energy = np.array([np.abs(mono[i : i + window]).max() for i in range(0, len(mono) - window, window)])
    voiced = np.where(energy > threshold)[0]
    if len(voiced) == 0:
        return audio_np
    first_voiced = max(0, voiced[0] * window - max_silent_samples)
    last_voiced = min(len(audio_np), (voiced[-1] + 1) * window + max_silent_samples)
    return audio_np[first_voiced:last_voiced]


def _normalize_volume(audio_np: np.ndarray, target_lufs: float = -23.0) -> np.ndarray:
    mono = audio_np.mean(axis=1) if audio_np.ndim == 2 else audio_np
    rms = float(np.sqrt(np.mean(mono**2))) if len(mono) > 0 else 0.0
    if rms < 1e-8:
        return audio_np
    current_lufs = 20 * np.log10(rms) - 0.691
    gain = 10 ** ((target_lufs - current_lufs) / 20.0)
    result = audio_np * max(0.1, min(float(gain), 10.0))
    peak = float(np.abs(result).max(initial=0.0))
    if peak > 0.99:
        result = result * (0.99 / peak)
    return result


def _shorten_long_silence(
    audio_np: np.ndarray,
    sample_rate: int,
    max_duration: float = 1.0,
    target_duration: float = 0.3,
    threshold_db: float = -35.0,
) -> np.ndarray:
    threshold = 10 ** (threshold_db / 20.0)
    window = max(1, int(0.02 * sample_rate))
    max_samples = int(max_duration * sample_rate)
    target_samples = int(target_duration * sample_rate)
    mono = audio_np.mean(axis=1) if audio_np.ndim == 2 else audio_np
    if len(mono) < window:
        return audio_np
    energy = np.array([np.abs(mono[i : i + window]).max() for i in range(0, len(mono) - window, window)])
    is_silent = energy < threshold
    silence_regions = []
    in_silence = False
    start = 0
    for idx, silent in enumerate(is_silent):
        if silent and not in_silence:
            start = idx * window
            in_silence = True
        elif not silent and in_silence:
            end = idx * window
            if end - start > max_samples:
                silence_regions.append((start, end))
            in_silence = False
    if in_silence:
        end = len(mono)
        if end - start > max_samples:
            silence_regions.append((start, end))
    if not silence_regions:
        return audio_np
    parts = []
    previous_end = 0
    for start, end in silence_regions:
        parts.append(audio_np[previous_end:start])
        parts.append(audio_np[start : start + target_samples])
        previous_end = end
    parts.append(audio_np[previous_end:])
    return np.concatenate(parts, axis=0)


def _concatenate_audio_chunks(chunks: list[torch.Tensor], sample_rate: int, pace: float, debug_prompt: bool = False) -> torch.Tensor:
    processed = []
    for audio in chunks:
        audio_np = _audio_tensor_to_numpy(audio)
        audio_np = trim_leading_transient_noise(audio_np, sample_rate, max_transient_seconds=SCENEMA_LEADING_TRANSIENT_MAX_SECONDS, threshold=SCENEMA_TRANSIENT_SILENCE_THRESHOLD, debug=debug_prompt, label="Scenema Audio")
        audio_np = trim_trailing_transient_noise(audio_np, sample_rate, max_transient_seconds=SCENEMA_TRANSIENT_MAX_SECONDS, threshold=SCENEMA_TRANSIENT_SILENCE_THRESHOLD, debug=debug_prompt, label="Scenema Audio")
        audio_np = _trim_silence(audio_np, sample_rate, max_silence=0.5)
        audio_np = _normalize_volume(audio_np)
        audio_np = trim_leading_transient_noise(audio_np, sample_rate, max_transient_seconds=SCENEMA_LEADING_TRANSIENT_MAX_SECONDS, threshold=SCENEMA_TRANSIENT_SILENCE_THRESHOLD, debug=debug_prompt, label="Scenema Audio")
        audio_np = trim_leading_noise_before_speech(audio_np, sample_rate, speech_threshold=SCENEMA_LEADING_SPEECH_THRESHOLD, max_leading_seconds=SCENEMA_CHUNK_DURATION_HEADROOM_SECONDS, debug=debug_prompt, label="Scenema Audio")
        audio_np = trim_trailing_transient_noise(audio_np, sample_rate, max_transient_seconds=SCENEMA_TRANSIENT_MAX_SECONDS, threshold=SCENEMA_TRANSIENT_SILENCE_THRESHOLD, debug=debug_prompt, label="Scenema Audio")
        audio_np = mute_isolated_transient_noise(audio_np, sample_rate, max_transient_seconds=SCENEMA_TRANSIENT_MAX_SECONDS, threshold=SCENEMA_TRANSIENT_SILENCE_THRESHOLD, debug=debug_prompt, label="Scenema Audio")
        audio_np = mute_isolated_transient_noise(audio_np, sample_rate, max_transient_seconds=SCENEMA_TRANSIENT_MAX_SECONDS, threshold=SCENEMA_ISOLATED_TRANSIENT_THRESHOLD, debug=debug_prompt, label="Scenema Audio")
        audio_np = ensure_trailing_silence(audio_np, sample_rate, SCENEMA_CHUNK_TAIL_SILENCE_SECONDS, threshold=SCENEMA_ISOLATED_TRANSIENT_THRESHOLD)
        audio_np = mute_isolated_transient_noise(audio_np, sample_rate, max_transient_seconds=SCENEMA_TRANSIENT_MAX_SECONDS, threshold=SCENEMA_ISOLATED_TRANSIENT_THRESHOLD, debug=debug_prompt, label="Scenema Audio")
        processed.append(audio_np)
    if not processed:
        raise ValueError("No Scenema Audio chunks were generated.")
    audio_np = np.concatenate(processed, axis=0)
    max_silence = min(0.5 * float(pace), 1.5)
    audio_np = mute_isolated_transient_noise(audio_np, sample_rate, max_transient_seconds=SCENEMA_TRANSIENT_MAX_SECONDS, threshold=SCENEMA_ISOLATED_TRANSIENT_THRESHOLD, debug=debug_prompt, label="Scenema Audio")
    audio_np = _shorten_long_silence(audio_np, sample_rate, max_duration=max_silence, target_duration=max_silence * 0.6, threshold_db=-30.0)
    audio_np = mute_isolated_transient_noise(audio_np, sample_rate, max_transient_seconds=SCENEMA_TRANSIENT_MAX_SECONDS, threshold=SCENEMA_ISOLATED_TRANSIENT_THRESHOLD, debug=debug_prompt, label="Scenema Audio")
    audio_np = trim_trailing_transient_noise(audio_np, sample_rate, max_transient_seconds=SCENEMA_TRANSIENT_MAX_SECONDS, threshold=SCENEMA_TRANSIENT_SILENCE_THRESHOLD, debug=debug_prompt, label="Scenema Audio")
    return _numpy_to_audio_tensor(audio_np).clamp_(-1.0, 1.0)


class ScenemaAudioPipeline(LTXAudioTTSPipelineBase):
    def __init__(
        self,
        model_weights_path: str,
        audio_vae_path: str,
        vocoder_path: str,
        text_projection_path: str,
        text_connector_path: str,
        gemma_path: str,
        config_path: str | None = None,
        alignment_whisper: torch.nn.Module | None = None,
        kokoro_pipeline=None,
        seedvc=None,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.alignment_whisper = alignment_whisper
        self.kokoro_pipeline = kokoro_pipeline
        self.seedvc = seedvc
        super().__init__(
            model_weights_path=model_weights_path,
            audio_vae_path=audio_vae_path,
            vocoder_path=vocoder_path,
            text_projection_path=text_projection_path,
            text_connector_path=text_connector_path,
            gemma_path=gemma_path,
            config_path=config_path,
            device=device,
            dtype=dtype,
        )

    @torch.inference_mode()
    def _generate_audio(self, audio_context: torch.Tensor, duration: float, seed: int, ref_latent=None, callback=None, status_extra: str = "", set_progress_status=None):
        sigmas = torch.tensor(DISTILLED_SIGMA_VALUES, dtype=torch.float32, device=self.device)
        audio_state, audio_tools = self._build_audio_state(duration, SCENEMA_FPS, sigmas, seed, ref_latent=ref_latent, reference_conditioner=AudioConditionByReferenceLatent)
        audio_state = self._generate_audio_euler(audio_context, sigmas, audio_state, audio_tools, callback=callback, status_extra=status_extra, set_progress_status=set_progress_status)
        if audio_state is None:
            return None
        return self._decode_audio_state(audio_state, set_progress_status=set_progress_status, status_extra=status_extra)

    def _reference_tail_waveform(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        return super()._reference_tail_waveform(waveform, sample_rate, SCENEMA_REF_TAIL_SECONDS)

    def _encode_reference(self, input_waveform, input_waveform_sample_rate, audio_guide: str | None):
        return super()._encode_reference(input_waveform, input_waveform_sample_rate, audio_guide, tail_seconds=SCENEMA_REF_TAIL_SECONDS, max_seconds=SCENEMA_MAX_REF_SECONDS)

    def _encode_tail_reference(self, audio: torch.Tensor, sample_rate: int):
        return super()._encode_tail_reference(audio, sample_rate, SCENEMA_REF_TAIL_SECONDS)

    @staticmethod
    def _generation_duration(planned_duration: float) -> float:
        return min(SCENEMA_MAX_DURATION_SECONDS, float(planned_duration) + SCENEMA_CHUNK_DURATION_HEADROOM_SECONDS)

    def _convert_repeated_speaker_chunks(
        self,
        generated_chunks: list[tuple[_ChunkSpec, torch.Tensor]],
        sample_rate: int,
        vc_steps: int,
        vc_cfg_rate: float,
        set_progress_status=None,
        speaker_ref_waveforms: dict[int, tuple[torch.Tensor, int]] | None = None,
        convert_reference_speakers: bool = False,
        convert_generated_repeats: bool = True,
    ) -> list[torch.Tensor]:
        speaker_ref_waveforms = speaker_ref_waveforms or {}
        generated_refs: dict[int, torch.Tensor] = {}
        speaker_seen: dict[int, int] = {}
        conversion_count = 0
        dry_seen: dict[int, int] = {}
        for chunk, _ in generated_chunks:
            dry_seen[chunk.speaker] = dry_seen.get(chunk.speaker, 0) + 1
            if convert_reference_speakers and chunk.speaker in speaker_ref_waveforms:
                conversion_count += 1
            elif convert_generated_repeats and dry_seen[chunk.speaker] > 1:
                conversion_count += 1
        if conversion_count and self.seedvc is None:
            raise RuntimeError("SeedVC is required to stabilize repeated Scenema Audio speaker segments.")
        converted_chunks = []
        converted_count = 0
        for chunk, audio in generated_chunks:
            speaker_seen[chunk.speaker] = speaker_seen.get(chunk.speaker, 0) + 1
            reference_waveform, reference_rate = speaker_ref_waveforms.get(chunk.speaker, (None, 0))
            if convert_reference_speakers and reference_waveform is not None and reference_rate > 0:
                converted_count += 1
                if set_progress_status is not None:
                    set_progress_status(f"Applying SeedVC Speaker {chunk.speaker} ({converted_count}/{conversion_count})")
                converted_chunks.append(self.seedvc.convert_tensor(audio, sample_rate, reference_waveform, reference_rate, sample_rate, diffusion_steps=vc_steps, cfg_rate=vc_cfg_rate))
                continue
            reference_audio = generated_refs.setdefault(chunk.speaker, audio.detach().cpu().float())
            if not convert_generated_repeats or speaker_seen[chunk.speaker] == 1:
                converted_chunks.append(audio)
                continue
            converted_count += 1
            if set_progress_status is not None:
                set_progress_status(f"Applying SeedVC Speaker {chunk.speaker} Segment {speaker_seen[chunk.speaker]} ({converted_count}/{conversion_count})")
            converted_chunks.append(self.seedvc.convert_tensor(audio, sample_rate, reference_audio, sample_rate, sample_rate, diffusion_steps=vc_steps, cfg_rate=vc_cfg_rate))
        return converted_chunks

    def generate(
        self,
        input_prompt: str,
        audio_guide: Optional[str] = None,
        *,
        alt_prompt: Optional[str] = None,
        seed: int = -1,
        callback=None,
        input_waveform=None,
        input_waveform_sample_rate=None,
        audio_guide2: Optional[str] = None,
        audio_guide3: Optional[str] = None,
        audio_prompt_type: str = "",
        custom_settings=None,
        duration_seconds: Optional[float] = None,
        set_progress_status=None,
        offloadobj=None,
        **kwargs,
    ) -> Optional[dict]:
        self._interrupt = False
        self._early_stop = False
        text = _read_text_or_file(input_prompt, "Prompt")
        if not text.strip():
            raise ValueError("Prompt text cannot be empty for Scenema Audio.")
        seed = random.randrange(0, 2**31) if seed is None or int(seed) < 0 else int(seed)
        _seed_everything(seed)

        audio_prompt_type = str(audio_prompt_type or "").upper()
        voice_instruction = _read_text_or_file(alt_prompt, "Voice instruction")
        pace = self._custom_float(custom_settings, "pace", SCENEMA_DEFAULT_PACE)
        vc_steps = self._custom_int(custom_settings, "vc_steps", 25)
        vc_cfg_rate = self._custom_float(custom_settings, "vc_cfg_rate", 0.5)
        debug_prompt = SCENEMA_DEBUG_PROMPT
        seedvc_enabled = "2" in audio_prompt_type

        if set_progress_status is not None:
            set_progress_status("Planning Audio Chunks")
        chunks = _limit_chunks_to_duration(_plan_prompt_chunks(text, voice_instruction, seed, pace, debug_prompt=debug_prompt, kokoro_pipeline=self.kokoro_pipeline), duration_seconds)
        if not chunks:
            raise ValueError("Scenema Audio prompt produced no chunks.")

        speaker_ref_latents = {}
        speaker_ref_waveforms = {}
        if ("A" in audio_prompt_type or "B" in audio_prompt_type) and (audio_guide or input_waveform is not None):
            if "B" not in audio_prompt_type and not audio_guide and input_waveform is None:
                raise ValueError("Scenema Audio reference voice mode requires a reference audio file.")
            if set_progress_status is not None:
                set_progress_status("Encoding Speaker 1 Reference")
            reference_waveform, reference_rate = self._waveform_from_input(input_waveform, input_waveform_sample_rate, audio_guide)
            if reference_waveform is None or reference_rate <= 0:
                raise ValueError("Scenema Audio could not encode the reference audio.")
            speaker_ref_waveforms[1] = (reference_waveform, reference_rate)
            if not seedvc_enabled:
                speaker_ref_latents[1] = self._encode_reference_waveform(self._reference_tail_waveform(reference_waveform, reference_rate), reference_rate)
        if "B" in audio_prompt_type and audio_guide2:
            if set_progress_status is not None:
                set_progress_status("Encoding Speaker 2 Reference")
            reference_waveform, reference_rate = self._waveform_from_input(None, None, audio_guide2)
            if reference_waveform is None or reference_rate <= 0:
                raise ValueError("Scenema Audio could not encode the second reference audio.")
            speaker_ref_waveforms[2] = (reference_waveform, reference_rate)
            if not seedvc_enabled:
                speaker_ref_latents[2] = self._encode_reference_waveform(self._reference_tail_waveform(reference_waveform, reference_rate), reference_rate)
        if "D" in audio_prompt_type and audio_guide3:
            if set_progress_status is not None:
                set_progress_status("Encoding Speaker 3 Reference")
            reference_waveform, reference_rate = self._waveform_from_input(None, None, audio_guide3)
            if reference_waveform is None or reference_rate <= 0:
                raise ValueError("Scenema Audio could not encode the third reference audio.")
            speaker_ref_waveforms[3] = (reference_waveform, reference_rate)
            if not seedvc_enabled:
                speaker_ref_latents[3] = self._encode_reference_waveform(self._reference_tail_waveform(reference_waveform, reference_rate), reference_rate)

        if self._interrupt:
            return None

        output_audio_sampling_rate = int(getattr(self.vocoder, "output_sampling_rate", AUDIO_SAMPLE_RATE))
        generated_chunks = []
        speaker_active_latents = dict(speaker_ref_latents)
        anchored_ref_speakers = set(speaker_ref_latents)
        try:
            for chunk_index, chunk in enumerate(chunks):
                if self._interrupt:
                    return None
                if self._early_stop_requested() and generated_chunks:
                    break
                if set_progress_status is not None:
                    set_progress_status(f"Encoding Prompt {chunk_index + 1}/{len(chunks)}")
                audio_context = self._encode_prompt(chunk.compiled_prompt)
                if set_progress_status is not None:
                    set_progress_status("Generating Audio")
                audio = self._generate_audio(audio_context, self._generation_duration(chunk.duration_s), chunk.seed, ref_latent=speaker_active_latents.get(chunk.speaker), callback=callback, status_extra=f"Chunk {chunk_index + 1}/{len(chunks)}", set_progress_status=set_progress_status)
                if audio is None:
                    if self._early_stop_requested() and generated_chunks:
                        break
                    return None
                if self._interrupt:
                    return None
                if SCENEMA_TRIM_EXTRA_WORDS:
                    if set_progress_status is not None:
                        set_progress_status(f"Trimming Extra Words {chunk_index + 1}/{len(chunks)}")
                    audio = _trim_leading_extra_words_tensor(self.alignment_whisper, audio, output_audio_sampling_rate, chunk.expected_text, chunk.language, debug_prompt=debug_prompt)
                audio = _trim_chunk_tail_to_silence_tensor(self.alignment_whisper, audio, output_audio_sampling_rate, chunk.expected_text, chunk.language, debug_prompt=debug_prompt)
                generated_chunks.append((chunk, audio))
                if self._early_stop_requested():
                    break
                if chunk_index < len(chunks) - 1 and chunk.speaker not in anchored_ref_speakers:
                    speaker_active_latents[chunk.speaker] = self._encode_tail_reference(audio, output_audio_sampling_rate)
        finally:
            if offloadobj is not None:
                offloadobj.unload_all()  # as before: the generation models and the alignment Whisper leave the VRAM before voice conversion

        if not generated_chunks:
            return None

        if seedvc_enabled and self.seedvc is None:
            raise RuntimeError("SeedVC voice conversion is not available.")

        speaker_ids = {chunk.speaker for chunk, _ in generated_chunks}
        if len(speaker_ids) > 1:
            audio = _concatenate_audio_chunks(
                self._convert_repeated_speaker_chunks(
                    generated_chunks,
                    output_audio_sampling_rate,
                    vc_steps,
                    vc_cfg_rate,
                    set_progress_status,
                    speaker_ref_waveforms=speaker_ref_waveforms,
                    convert_reference_speakers=seedvc_enabled,
                    convert_generated_repeats=True,
                ),
                output_audio_sampling_rate,
                pace,
                debug_prompt=debug_prompt,
            )
        elif seedvc_enabled:
            if self.seedvc is None:
                raise RuntimeError("SeedVC voice conversion is not available.")
            speaker_id = next(iter(speaker_ids))
            if speaker_id in speaker_ref_waveforms:
                audio = _concatenate_audio_chunks([audio for _, audio in generated_chunks], output_audio_sampling_rate, pace, debug_prompt=debug_prompt)
                reference_waveform, reference_rate = speaker_ref_waveforms[speaker_id]
                if reference_waveform is None or reference_rate <= 0:
                    raise ValueError("SeedVC voice conversion requires a reference audio file.")
                if set_progress_status is not None:
                    set_progress_status("Applying SeedVC")
                audio = self.seedvc.convert_tensor(audio, output_audio_sampling_rate, reference_waveform, reference_rate, output_audio_sampling_rate, diffusion_steps=vc_steps, cfg_rate=vc_cfg_rate)
                if set_progress_status is not None:
                    set_progress_status("Cleaning SeedVC Output")
                audio = _concatenate_audio_chunks([audio], output_audio_sampling_rate, pace, debug_prompt=debug_prompt)
            else:
                audio = _concatenate_audio_chunks(
                    self._convert_repeated_speaker_chunks(generated_chunks, output_audio_sampling_rate, vc_steps, vc_cfg_rate, set_progress_status),
                    output_audio_sampling_rate,
                    pace,
                    debug_prompt=debug_prompt,
                )
        elif not speaker_ref_waveforms:
            audio = _concatenate_audio_chunks(self._convert_repeated_speaker_chunks(generated_chunks, output_audio_sampling_rate, vc_steps, vc_cfg_rate, set_progress_status), output_audio_sampling_rate, pace, debug_prompt=debug_prompt)
        else:
            audio = _concatenate_audio_chunks([audio for _, audio in generated_chunks], output_audio_sampling_rate, pace, debug_prompt=debug_prompt)
        return {"x": audio, "audio_sampling_rate": output_audio_sampling_rate}
