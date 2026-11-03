# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Prompts of Nemotron 3.5 Transcribe: instruction wording from the model card's prompt guide, the checkpoint's chat template rendered with
reasoning off as the reference processor does (`apply_transcription_request`), the audio placeholder expansion, decoding and the model card's
generation-loop guardrail."""
import json
import re
from pathlib import Path

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment
from transformers import AutoTokenizer

# the model card's prompt guide (exact wording)
VERBATIM_PROMPT = "Audio transcription. Language: as spoken. Wording: verbatim, including fillers and repetitions. Formatting: punctuation and capitalization. Output: transcript only."
READABLE_PROMPT = ("Audio transcription. Language: as spoken. Wording: omit fillers and accidental repetitions; keep the remaining spoken wording unchanged. "
                   "Formatting: punctuation and capitalization. Output: transcript only.")
SPEAKERS_PROMPT = ("Transcribe this audio with punctuation and capitalization. Include all spoken words and verbal fillers. Write numbers in their standard "
                   "written form. Put speaker tags <spk:0> <spk:1> <spk:2> <spk:3> on the transcription.")
TRANSLATION_PROMPT = "Please translate the given speech to {language}."
CONTEXT_TERMS = "Context terms: {terms}."
# not in the guide: the speaker-attributed prompt with the readable wording clause of READABLE_PROMPT
SPEAKERS_READABLE_WORDING = "Omit fillers and accidental repetitions; keep the remaining spoken wording unchanged."

SPEAKER_TAG_PATTERN = re.compile(r"<spk:(\d+)>", flags=re.IGNORECASE)
TAG_PATTERN = re.compile(r"(?i:<spk:\d+>)|<(?:ev|sty):[^<>]*>")  # speaker, event and style tags
EOS_TOKEN = "<|im_end|>"


def build_instruction(task="transcribe", speakers=False, verbatim=True, context_terms="", target_language="", instructions=""):
    """User instruction for a transcription or a translation (into `target_language`). Speaker tags request `<spk:N>` turns (the prompt the
    word timestamps and the diarization mapping expect); `context_terms` (a string or a list of names and terms) bias the vocabulary;
    `instructions` are appended as given."""
    if task == "translate":
        prompt = TRANSLATION_PROMPT.format(language=target_language.strip())
    elif speakers:
        prompt = SPEAKERS_PROMPT if verbatim else SPEAKERS_PROMPT.replace("Include all spoken words and verbal fillers.", SPEAKERS_READABLE_WORDING)
    else:
        prompt = VERBATIM_PROMPT if verbatim else READABLE_PROMPT
    if not isinstance(context_terms, str):
        context_terms = ", ".join(term.strip() for term in context_terms if term.strip())
    context_terms = context_terms.strip().rstrip(".")
    if context_terms:
        prompt += " " + CONTEXT_TERMS.format(terms=context_terms)
    if instructions.strip():
        prompt += " " + instructions.strip()
    return prompt


# Adapted by NVIDIA from the Qwen3-ASR repetition fix (https://github.com/QwenLM/Qwen3-ASR/blob/7c6daf77a2421100f5fb066495372c00129d39ff/qwen_asr/inference/utils.py,
# Copyright 2026 The Alibaba Qwen team, Apache License 2.0), as given in the Nemotron 3.5 Transcribe model card: blocks of up to 4,096 characters.
CHAR_RUN = re.compile(r"(.)\1{20,}", re.DOTALL)
BLOCK_RUN = re.compile(r"(.{1,4096}?)\1{19,}", re.DOTALL)


def cleanup_generation_loops(text: str) -> str:
    # Collapse a character repeated ≥21 times to one copy.
    text = CHAR_RUN.sub(lambda m: m.group(1), text)

    # Collapse an exact 1–4096-character block repeated ≥20 times.
    return BLOCK_RUN.sub(lambda m: m.group(1), text)


def render_speakers(text: str) -> str:
    """One "Speaker N: ..." line per `<spk:N>` turn, speakers numbered from 1 (tag 0 is Speaker 1); the text before the first tag on its own line.
    Event and style tags are kept, everything from the end-of-text token on dropped."""
    parts = SPEAKER_TAG_PATTERN.split(text.split(EOS_TOKEN)[0])
    lines = [f"Speaker {int(speaker) + 1}: {content.strip()}" for speaker, content in zip(parts[1::2], parts[2::2]) if content.strip()]
    return "\n".join(([parts[0].strip()] if parts[0].strip() else []) + lines)


class NemotronPrompt:
    """Tokenizer and chat template of the checkpoint (`assets_dir`: tokenizer.json, tokenizer_config.json, chat_template.jinja, config.json),
    `processor_config`: processor_config.json."""

    def __init__(self, assets_dir, processor_config):
        assets_dir = Path(assets_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(str(assets_dir))
        self.eos_token_id = self.tokenizer.eos_token_id
        self.audio_token = processor_config["audio_token"]
        self.audio_token_id = self.tokenizer.convert_tokens_to_ids(self.audio_token)
        self.default_prompt = processor_config["default_transcription_prompt"]
        # the audio token is appended after the language model's vocabulary: embedded as token 0, its rows replaced by the audio
        self.vocab_size = json.loads((assets_dir / "config.json").read_text(encoding="utf-8"))["text_config"]["vocab_size"]
        environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=[jinja2.ext.loopcontrols])
        environment.filters["tojson"] = lambda x, ensure_ascii=False, indent=None, separators=None, sort_keys=False: json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
        environment.globals["raise_exception"] = self._raise_exception
        self.template = environment.from_string((assets_dir / "chat_template.jinja").read_text(encoding="utf-8"))

    @staticmethod
    def _raise_exception(message):
        raise jinja2.exceptions.TemplateError(message)

    def render(self, instruction=None, system_prompt=None, enable_thinking=False):
        """The prompt text with one audio locator: the user turn is the instruction followed by the audio, as `apply_transcription_request`."""
        messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
        messages.append({"role": "user", "content": [{"type": "text", "text": self.default_prompt if instruction is None else instruction}, {"type": "audio"}]})
        return self.template.render(messages=messages, add_generation_prompt=True, enable_thinking=enable_thinking, **self.tokenizer.special_tokens_map)

    def encode(self, instruction, num_audio_tokens, system_prompt=None, enable_thinking=False):
        """`(prompt_ids, audio_positions)`: the token ids of the rendered prompt with its audio locator expanded to `num_audio_tokens` audio
        tokens (ids `audio_token_id`), and the slice of their positions, the rows the audio embeddings replace."""
        prompt_ids = self.tokenizer(self.render(instruction, system_prompt, enable_thinking))["input_ids"]
        start = prompt_ids.index(self.audio_token_id)
        return prompt_ids[:start] + [self.audio_token_id] * num_audio_tokens + prompt_ids[start + 1 :], slice(start, start + num_audio_tokens)

    def embedding_ids(self, prompt_ids):
        """The ids to look up in the language model's embeddings: the audio token past the vocabulary becomes 0 (its rows take the audio)."""
        return [0 if token == self.audio_token_id and token >= self.vocab_size else token for token in prompt_ids]

    def decode(self, token_ids) -> str:
        """Generated text with its special tokens (speaker, event and style tags) and the padding token removed, as the model card decodes."""
        return self.tokenizer.decode(token_ids, skip_special_tokens=False).replace(self.tokenizer.pad_token, "")

    def strip_special(self, text: str) -> str:
        """`text` without its speaker, event and style tags and its end-of-text token."""
        text = TAG_PATTERN.sub(" ", text.split(EOS_TOKEN)[0])
        return re.sub(r"[ \t]+", " ", re.sub(r" *\n *", "\n", text)).strip()
