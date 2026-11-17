# Adapted from Zyphra/ZONOS3-Mini (model/src/zonos3/tts/superbpe.py, tts/prompt.py, tts/plan.py, tokenizer/textnorm.py), commit
# 9fd14e51. Copyright (c) 2026 Zyphra, Inc. MIT License, see LICENSE_ZONOS3. WanGP additions: the [pause] markup, inline cues and dialogue turns.
"""Prompt rows of ZONOS3-Mini, in the training layout: [speaker row | reference prefix][condition rows] BOS text EOS [silence rows].
Each row holds the 10 codec codes of a frame (codebook j delayed by j rows; audio padding where none) and one text token."""
import bisect
import re

import torch
from tokenizers import Tokenizer

PHONEME_OPEN, PHONEME_CLOSE, START_MARKER = "", "", ""
CUE_MARKER = ""  # where an inline cue starts: kept through normalization, never tokenized
PAUSE_MARKER_RE = re.compile("[-]")
PRIVATE_MARKER_RE = re.compile("[-]")
PAUSE_TAG_RE = re.compile(r"\s*\[\s*pause(?:\s+(\d+(?:\.\d+)?)\s*s?)?\s*\]", re.IGNORECASE)
BRACKET_RE = re.compile(r"\[([^\[\]]*)\][ \t]*")
PAUSE_EDGES = (0.35, 0.7, 1.4)  # bucket boundaries in seconds between the 0.25 / 0.5 / 1 / 2 s pause tokens
DEFAULT_PAUSE = 0.5


def pause_markers(seconds, markers):
    """Pause markers for a silence of about `seconds`: the nearest bucket, several 2 s ones for a longer pause."""
    out = ""
    while seconds > 2.8:
        out, seconds = out + markers[-1], seconds - 2.0
    return out + markers[bisect.bisect_right(PAUSE_EDGES, seconds)]


def split_cues(text):
    """Inline cues `[happy]` -> a cue marker where each starts (the text is otherwise unchanged) and the text inside each one;
    [pause] tags stay. Cues with no words after them are dropped. Raises ValueError for an unmatched bracket."""
    cues = []

    def cue(match):
        if PAUSE_TAG_RE.fullmatch(match.group().rstrip()):
            return match.group()
        cues.append(match.group(1).strip())
        return CUE_MARKER
    text = BRACKET_RE.sub(cue, text)
    rest = PAUSE_TAG_RE.sub("", text)
    if "[" in rest or "]" in rest:
        raise ValueError("Unmatched '[' or ']' in the text: cues are written [happy], pauses [pause] or [pause 1s].")
    text = text.rstrip()
    while text.endswith(CUE_MARKER):
        text, cues = text[:-1].rstrip(), cues[:-1]
    return text, cues


def apply_pause_tags(text, markers):
    """[pause] / [pause 1.5s] -> the model's pause markers, glued to the word before them as trained ("slid,<pause> on")."""
    return PAUSE_TAG_RE.sub(lambda match: pause_markers(float(match.group(1)) if match.group(1) else DEFAULT_PAUSE, markers), text).strip()


def shear(x, pad):
    """Delay codebook c of frames [T, C] by c rows: row t holds codebook c of frame t - c (pad before)."""
    frames, codebooks = x.shape
    padded = x.new_full((codebooks - 1 + frames, codebooks), pad)
    padded[codebooks - 1:] = x
    rows = (codebooks - 1) + torch.arange(frames, device=x.device)[:, None] - torch.arange(codebooks, device=x.device)
    return padded.gather(0, rows)


def undelay(rows, pad):
    """Delayed rows [R, C] -> frames [R - C + 1, C]: frame f takes codebook c from row f + c."""
    count, codebooks = rows.shape
    frames = count - codebooks + 1
    return torch.stack([rows[c:c + frames, c] for c in range(codebooks)], dim=1)


class TextEncoder:
    """Text normalization (zyphra-textnorm: Rust ports of the NeMo grammars) and SuperBPE ids with the start and pause tokens."""

    def __init__(self, tokenizer_path, text, languages):
        self.tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self.offset, self.bos, self.eos = text["offset"], text["bos"], text["eos"]
        self.start_id = text["start"]["id"]
        self.pause_ids = dict(zip(text["pauses"]["markers"], text["pauses"]["ids"]))
        self.pause_markers = text["pauses"]["markers"]
        self.nemo = {language["code"]: language["nemo"] for language in languages}
        self._normalizer = None

    def normalize(self, text, language):
        """Written -> spoken form ("$5.32" -> "five dollars thirty two cents") in the grammar of the language. The start / pause / cue
        markers go through as one-character groups the grammar keeps verbatim."""
        import zyphra_textnorm
        if self._normalizer is None:
            self._normalizer = zyphra_textnorm.Normalizer()
        wrapped = re.sub("[-]", lambda match: PHONEME_OPEN + match.group() + PHONEME_CLOSE, text)
        return re.sub(PHONEME_OPEN + "([-])" + PHONEME_CLOSE, r"\1", self._normalizer.normalize(wrapped, self.nemo[language]))

    def payload_ids(self, text):
        if text.startswith(START_MARKER):
            return [self.start_id] + self.payload_ids(text[1:])
        ids, position = [], 0
        for match in PAUSE_MARKER_RE.finditer(text):
            if match.start() > position:
                ids += self._bpe(text[position:match.start()])
            ids.append(self.pause_ids[match.group()])
            position = match.end()
        return ids + (self._bpe(text[position:]) if position < len(text) else [])

    def _bpe(self, text):
        return [index + self.offset for index in self.tokenizer.encode(text, add_special_tokens=False).ids]

    def spoken(self, text, language, normalize=True):
        """The text as it is read, (normalized) with its pause markers and a cue marker where each inline cue starts, and the text
        inside each cue."""
        import zyphra_textnorm
        text, cues = split_cues(PRIVATE_MARKER_RE.sub("", text))
        text = zyphra_textnorm.collapse_line_breaks(apply_pause_tags(text, self.pause_markers))
        return (self.normalize(text, language) if normalize else text), cues

    def spoken_ids(self, spoken, start=False):
        """The prompt's text ids of a spoken text: BOS, the text with its pause markers, EOS (cue markers are not read)."""
        return [self.bos] + self.payload_ids((START_MARKER if start else "") + spoken.replace(CUE_MARKER, "")) + [self.eos]

    def ids(self, text, language, normalize=True, start=False):
        return self.spoken_ids(self.spoken(text, language, normalize)[0], start)


def prompt_rows(text_ids, *, n_codebooks, audio_pad_id, text_vocab, condition_count, speaker_row, reference_codes=None, silence_frame=None, silence_frames=0):
    """The prompt rows [R, n_codebooks + 1] (int64, on the CPU), the speaker row index (-1 without) and the first condition row."""
    def plain(count):
        rows = torch.full((count, n_codebooks + 1), audio_pad_id, dtype=torch.int64, device="cpu")
        rows[:, n_codebooks] = text_vocab
        return rows

    parts = []
    if speaker_row:
        parts.append(plain(1))
    if reference_codes is not None:
        # each codebook delayed by its index, the tail flushed with audio padding, no separator
        codes = torch.cat((reference_codes.long().cpu(), torch.full((n_codebooks - 1, n_codebooks), audio_pad_id, dtype=torch.int64, device="cpu")))
        parts.append(torch.cat((shear(codes, audio_pad_id), torch.full((len(codes), 1), text_vocab, dtype=torch.int64, device="cpu")), dim=1))
    condition_start = sum(len(part) for part in parts)
    parts.append(plain(condition_count))
    text = plain(len(text_ids))
    text[:, n_codebooks] = torch.tensor(text_ids, device="cpu")
    parts.append(text)
    if silence_frames:
        # the codec's silence frame as the first audio rows, delayed, without the flushed tail: generation continues the pattern
        silence = shear(torch.tensor(silence_frame, dtype=torch.int64, device="cpu").repeat(silence_frames, 1), audio_pad_id)[:silence_frames]
        parts.append(torch.cat((silence, torch.full((silence_frames, 1), text_vocab, dtype=torch.int64, device="cpu")), dim=1))
    return torch.cat(parts), 0 if speaker_row else -1, condition_start


def condition_bucket(value, edges):
    """A raw condition value -> its bucket (the training loader's bisect over the bucket edges)."""
    return bisect.bisect_right(edges, value)
