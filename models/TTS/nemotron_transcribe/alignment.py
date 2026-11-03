# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0. Ported from Hugging Face Transformers (commit a37e25d9, models/nemotron3_5_transcribe/processing_nemotron3_5_transcribe.py:
# `align_word_timestamps`, a port of NVIDIA NeMo's `MultiSpeakerSOTWordTimestampAligner`, and `extract_speaker_dict`).
# The CTC forced alignment reproduces torchaudio.functional.forced_align and merge_tokens (BSD 2-Clause License, Copyright (c) 2017 Facebook Inc.).
# Modified for WanGP.
"""Word timestamps and speaker attribution of a Nemotron 3.5 Transcribe transcript: the words of each `<spk:N>` tag are force-aligned on the
frames of the word timestamp head where the diarization speaker matched to the tag is active, then snapped to that speaker's segments."""
import re
import unicodedata
from bisect import bisect_right

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

SPEAKER_TAG_PATTERN = re.compile(r"<spk:(\d+)>", flags=re.IGNORECASE)
# A "frame" is a frame of the word timestamp head, a "tag" a `<spk:N>` speaker tag, a "speaker" a diarization speaker.
KEPT_TAG_PATTERN = re.compile(r"(?i:<spk:\d+>)|<(?:ev|sty):[^<>]*>")  # speaker, event and style tags
# the language model tokenizer's special tokens (bos, eos, unk = pad, audio), replaced by spaces before the alignment
SPECIAL_TOKENS = ("<s>", "<|im_end|>", "<unk>", "<|audio|>")
EOS_TOKEN = "<|im_end|>"
# the scripts that set the language of a transcript, by priority, as Unicode character name prefixes
SCRIPT_LANGUAGES = [("ko", "HANGUL"), ("ja", "HIRAGANA"), ("ja", "KATAKANA"), ("zh", "CJK UNIFIED"), ("zh", "CJK COMPATIBILITY"), ("th", "THAI")]
VITERBI_SEGMENT_FRAMES = 1024  # frames between the checkpointed Viterbi scores: the backpointers of one segment exist at a time


def _lexical_characters(text):
    """Letters, marks and numbers of `text`, NFC-normalized."""
    return [char for char in unicodedata.normalize("NFC", text) if unicodedata.category(char)[0] in "LMN"]


def _character_token_ids(tokenizer, character):
    """A character is its own piece when the vocabulary has one, else its pieces without the bare word boundary."""
    token_ids = [tokenizer.piece_to_id(character)]
    if token_ids == [tokenizer.unk_id()]:
        word_boundary_id = tokenizer.piece_to_id("▁")
        token_ids = [i for i in tokenizer.encode(character) if i != word_boundary_id]
    if tokenizer.unk_id() in token_ids:
        raise ValueError(f"Unrepresentable character: {character!r}")
    return token_ids


def _tokenize_transcript(tokenizer, text, language, blank_id):
    """`(word, token ids, turn)` of the words of each tag of a transcript (`turn` the index of the text part between tags): the words before the
    first `<spk:N>` tag have tag `None`, a transcript without tags is tag 0. `tokenizer` is the SentencePiece model, whose dummy prefix is the
    leading space of each word. Chinese and Japanese (`language`, detected from the script when `None`) are aligned character by character."""
    parts = SPEAKER_TAG_PATTERN.split(text.replace("—", " "))  # [text, tag, text, tag, text, ...]
    tags = [0 if len(parts) == 1 else None] + [int(tag) for tag in parts[1::2]]
    words = [(word, tag, turn) for turn, (tag, part) in enumerate(zip(tags, parts[::2])) for word in part.split()]
    if language is None:
        names = [unicodedata.name(char, "") for word, _, _ in words if not KEPT_TAG_PATTERN.fullmatch(word) for char in word]
        language = next((lang for lang, prefix in SCRIPT_LANGUAGES if any(n.startswith(prefix) for n in names)), None)
    if language == "th":
        raise ValueError("Word timestamps are not supported for Thai, which needs the `pythainlp` word segmenter.")
    streams = {}
    for word, tag, turn in words:
        if language in ("zh", "ja") and not KEPT_TAG_PATTERN.fullmatch(word):
            units = [(char, _character_token_ids(tokenizer, char), turn) for char in _lexical_characters(word)]
        else:
            units = [(word, tokenizer.encode(word), turn)]
        if any(not token_ids or max(token_ids) >= blank_id for _, token_ids, _ in units):
            raise ValueError("Tokenizer produced IDs outside the non-blank CTC vocabulary.")
        streams.setdefault(tag, []).extend(units)
    return streams


def _match_tags_to_speakers(class_log_probs, classes, streams, speaker_probabilities, threshold, blank_id, vocab_size):
    """Diarization speaker of each tag, matched one to one on the tokens they share: a speaker collects the greedy tokens decoded while it is
    active, a tag the tokens of its words, and the IDF-weighted cosine similarity of their counts, penalized when their amounts of speech differ,
    is maximized by linear assignment. Unmatched tags get `None`. `class_log_probs` holds the log-probabilities of `classes` (the blank, then
    the transcript's tokens, sorted): the greedy decoding chooses among them, the blank first on ties."""
    num_speakers = speaker_probabilities.shape[1]
    targets = [[token for _, token_ids, _ in words for token in token_ids] for words in streams.values()]
    greedy = classes[class_log_probs.argmax(dim=1)]
    is_onset = (greedy != blank_id) & torch.cat([torch.tensor([True], device="cpu"), greedy[1:] != greedy[:-1]])
    tag_counts = torch.stack([torch.bincount(torch.tensor(target, device="cpu"), minlength=vocab_size) for target in targets])
    speaker_counts = torch.stack([torch.bincount(greedy[is_onset & (active >= threshold)], minlength=vocab_size) for active in speaker_probabilities.T])
    tag_counts, speaker_counts = tag_counts.double(), speaker_counts.double()

    idf = torch.log1p(num_speakers / (1 + (speaker_counts > 0).sum(dim=0).double()))
    tag_vectors, speaker_vectors = (tag_counts * idf).sqrt(), (speaker_counts * idf).sqrt()
    tag_norms, speaker_norms = (tag_vectors**2).sum(dim=1).sqrt(), (speaker_vectors**2).sum(dim=1).sqrt()
    similarity = tag_vectors @ speaker_vectors.T / (tag_norms[:, None] * speaker_norms[None, :]).clamp(min=1e-8)
    similarity -= 0.1 * ((tag_counts.sum(dim=1)[:, None] + 1) / (speaker_counts.sum(dim=1)[None, :] + 1)).log().abs()

    tag_to_speaker = dict.fromkeys(streams)
    for tag, speaker in zip(*linear_sum_assignment(-similarity.numpy())):
        tag_to_speaker[list(streams)[tag]] = int(speaker)
    return tag_to_speaker


def _forced_align(emissions, targets, blank):
    """Label of every frame on the most likely CTC path of `targets` through `emissions` [frames, labels] (float32): the recursion, band and
    tie-breaking of torchaudio's `forced_align`, with the scores checkpointed every VITERBI_SEGMENT_FRAMES frames and the backpointers recomputed
    one segment at a time during the traceback."""
    num_frames, num_targets = emissions.shape[0], len(targets)
    targets = np.asarray(targets, dtype=np.int64)
    num_states = 2 * num_targets + 1
    labels = np.full(num_states, blank, dtype=np.int64)
    labels[1::2] = targets
    can_skip = np.zeros(num_states, dtype=bool)
    can_skip[3::2] = targets[1:] != targets[:-1]
    num_repeats = int((targets[1:] == targets[:-1]).sum())
    # the states reachable at each frame
    bands = np.empty((num_frames, 2), dtype=np.int64)
    start, end = (0 if num_frames - (num_targets + num_repeats) > 0 else 1), (1 if num_states == 1 else 2)
    bands[0] = start, end
    for t in range(1, num_frames):
        if num_frames - t <= num_targets + num_repeats:
            if start % 2 == 1 and start // 2 + 1 < num_targets and targets[start // 2] != targets[start // 2 + 1]:
                start += 1
            start += 1
        if t <= num_targets + num_repeats:
            if end % 2 == 0 and end < 2 * num_targets and targets[end // 2 - 1] != targets[end // 2]:
                end += 1
            end += 1
        bands[t] = start, end
    neg_inf = np.float32(-np.inf)

    def step(alpha, t, backpointers):
        # `alpha`: the scores of frame t - 1 behind two -inf pads; returns frame t's
        start, end = bands[t]
        scores = np.full(num_states + 2, neg_inf, dtype=np.float32)
        if start == 0:
            scores[2] = alpha[2] + emissions[t, blank]
            if backpointers is not None:
                backpointers[0] = 0
            start = 1
        if start < end:
            stay, advance = alpha[start + 2 : end + 2], alpha[start + 1 : end + 1]
            skip = np.where(can_skip[start:end], alpha[start:end], neg_inf)
            is_skip = (skip > advance) & (skip > stay)
            is_advance = ~is_skip & (advance > stay) & (advance > skip)
            scores[start + 2 : end + 2] = np.where(is_skip, skip, np.where(is_advance, advance, stay)) + emissions[t, labels[start:end]]
            if backpointers is not None:
                backpointers[start:end] = np.where(is_skip, 2, np.where(is_advance, 1, 0))
        return scores

    alpha = np.full(num_states + 2, neg_inf, dtype=np.float32)
    alpha[bands[0, 0] + 2 : bands[0, 1] + 2] = emissions[0, labels[bands[0, 0] : bands[0, 1]]]
    checkpoints = {0: alpha}
    for t in range(1, num_frames):
        if t % VITERBI_SEGMENT_FRAMES == 0:
            checkpoints[t] = alpha  # the scores of frame t - 1, from which the segment starting at t is recomputed
        alpha = step(alpha, t, None)
    state = num_states - 1 if alpha[num_states + 1] > alpha[num_states] else num_states - 2
    path = np.empty(num_frames, dtype=np.int64)
    for first in reversed(range(0, num_frames, VITERBI_SEGMENT_FRAMES)):
        last = min(first + VITERBI_SEGMENT_FRAMES, num_frames)
        backpointers = np.full((last - first, num_states), -1, dtype=np.int8)
        alpha = checkpoints[first]
        for t in range(max(first, 1), last):
            alpha = step(alpha, t, backpointers[t - first])
        for t in range(last - 1, first - 1, -1):
            path[t] = labels[state]
            state -= int(backpointers[t - first, state])
    return path


def _token_spans(path, blank):
    """`(start, end)` frames of each run of a non-blank label (torchaudio's `merge_tokens`)."""
    changes = np.flatnonzero(np.diff(path, prepend=-1, append=-1) != 0)
    return [(int(start), int(end)) for start, end in zip(changes[:-1], changes[1:]) if path[start] != blank]


def _align_words(class_log_probs, column_of, words, probabilities, frame_duration, max_word_duration):
    """Start and end of each `(word, token ids, turn)`, force-aligned on the frames where their speaker has a `probabilities` of at least 0.01,
    its log-probability added to the token log-probabilities (column 0 is the blank). A word spans the frames of its tokens, for at most
    `max_word_duration`. `None` when the words do not fit on those frames."""
    target = [column_of[token] for _, token_ids, _ in words for token in token_ids]
    frames = (probabilities >= 0.01).nonzero().flatten()
    num_repeats = sum(token == next_token for token, next_token in zip(target, target[1:]))
    if len(frames) < len(target) + num_repeats:  # a blank must separate repeated tokens
        return None
    emissions = class_log_probs[frames] + probabilities[frames].clamp(min=1e-6).log()[:, None]
    emissions[:, 0] = class_log_probs[frames, 0]
    spans = iter(_token_spans(_forced_align(emissions.numpy(), target, 0), 0))  # one span per token, in order
    frames = frames.tolist()
    timestamps = []
    for word, token_ids, turn in words:
        word_spans = [next(spans) for _ in token_ids]
        start = frames[word_spans[0][0]] * frame_duration
        end = min((frames[word_spans[-1][1] - 1] + 1) * frame_duration, start + max_word_duration)
        timestamps.append({"word": word, "start": start, "end": end, "turn": turn})
    return timestamps


def _refine_word_timestamps(words, segments, frame_duration, max_word_frames):
    """Snaps the words of one speaker, in time order, to its diarization `segments`: the first word of a segment starts with it; a word ends where
    the next word of its segment starts, else with its segment; a word outside segments ends where the next word starts. Shifts and durations
    are capped at `max_word_frames` frames of at most 80 ms."""
    limit = max_word_frames * min(frame_duration, 0.08)
    segments = sorted(segments)
    starts = [start for start, _ in segments]
    # the segment containing the start of each word, `None` outside segments
    segment_of = []
    for word in words:
        index = bisect_right(starts, word["start"]) - 1
        segment_of.append(index if index >= 0 and word["start"] < segments[index][1] else None)
    for i, (word, segment) in enumerate(zip(words, segment_of)):
        is_first_of_segment = segment is not None and segment_of[i - 1 : i] != [segment]
        if is_first_of_segment and 0 < word["start"] - segments[segment][0] <= limit:
            word["start"] = segments[segment][0]
    for i, (word, segment) in enumerate(zip(words, segment_of)):
        if segment is not None and segment_of[i + 1 : i + 2] != [segment]:
            end = segments[segment][1]
        else:
            end = words[i + 1]["start"] if i + 1 < len(words) else word["end"]
        word["end"] = max(word["start"], min(end, word["start"] + limit))
    return words


def _resample_frames(probabilities, num_frames):
    """Linear interpolation of `probabilities` `(frames, speakers)` onto `num_frames` frames, first and last aligned."""
    if probabilities.shape[0] == num_frames:
        return probabilities
    positions = torch.linspace(0, probabilities.shape[0] - 1, num_frames, dtype=torch.float32, device=probabilities.device)
    lower, upper = positions.floor().long(), positions.ceil().long()
    fraction = (positions - lower)[:, None]
    return probabilities[lower] * (1 - fraction) + probabilities[upper] * fraction


def _diarization_segments(is_active, frame_duration, audio_duration):
    """Each run of activity of `is_active` `(speakers, frames)` as `{"speaker", "start", "end"}`, sorted by time."""
    transitions = torch.nn.functional.pad(is_active.to(torch.int8), (1, 1)).diff(dim=1)
    starts, ends = (transitions == 1).nonzero().tolist(), (transitions == -1).nonzero()[:, 1].tolist()
    segments = [{"speaker": speaker, "start": start * frame_duration, "end": min(end * frame_duration, audio_duration)} for (speaker, start), end in zip(starts, ends)]
    segments = [segment for segment in segments if segment["end"] > segment["start"]]
    return sorted(segments, key=lambda segment: (segment["start"], segment["end"], segment["speaker"]))


def alignment_classes(text, ctc_tokenizer, language=None):
    """The classes the alignment of `text` reads (the CTC blank, then the transcript's tokens, sorted), for
    `NemotronPerception.word_timestamp_logits(..., columns=...)`; `None` when `text` cannot be aligned."""
    blank_id = ctc_tokenizer.get_piece_size()
    for token in SPECIAL_TOKENS:
        text = text.replace(token, " ")
    try:
        streams = _tokenize_transcript(ctc_tokenizer, text, language, blank_id)
    except ValueError:
        return None
    return [blank_id] + sorted({token for words in streams.values() for _, token_ids, _ in words for token in token_ids}) if streams else None


def align_words(text, word_timestamp_logits, mask, speaker_logits, config, ctc_tokenizer, columns=None, language=None, speaker_activity_threshold=0.5,
                max_word_duration=1.0, max_word_frames=12):
    """Times and speaker of each word of a generated transcript (decoded with its special tokens), as the reference `align_word_timestamps`.

    `word_timestamp_logits`: the raw head logits [1, frames, vocab] of `NemotronPerception.word_timestamp_logits`, or, with `columns` (from
    `alignment_classes`), its float32 log-probabilities [valid frames, len(columns)] of those classes. `mask` [1, spectrogram frames] the
    feature mask, `speaker_logits` [1, spectrogram frames, speakers] from `NemotronPerception.forward` (or `None`: one speaker, no
    diarization), `config` the checkpoint's config.json, `ctc_tokenizer` the alignment SentencePiece model. `language`: "zh" and "ja" are aligned
    character by character, detected from the script when `None`; Thai is not supported.

    Returns `words` ([{"word", "start", "end", "speaker", "turn"}] in seconds, sorted by time, `speaker` the N of the word's `<spk:N>` tag
    (`None` before the first tag, 0 without tags), `turn` its speaker turn for `speaker_turns`; `None` when the text cannot be aligned),
    `diarization` ([{"speaker", "start", "end"}] activity segments of the diarization speakers), `speaker_tag_to_diarization_speaker`,
    `failed_speakers` (the tags whose words could not be aligned, missing from `words`) and `error` (`None` on success)."""
    preprocessor = config["perception"]["preprocessor"]
    sampling_rate, factor, blank_id = preprocessor["sample_rate"], config["audio_config"]["asr_config"]["subsampling_factor"], ctc_tokenizer.get_piece_size()
    hop_length = int(preprocessor["window_stride"] * sampling_rate)  # as the processor's conversion
    num_spectrogram_frames = int(mask.sum())
    num_encoder_frames = -(-num_spectrogram_frames // factor)
    # the head's frames per encoder frame: full logits span the encoder frames of the padded spectrogram, log-probabilities the valid ones
    upsampling_factor = word_timestamp_logits.shape[-2] // (-(-mask.shape[-1] // factor) if columns is None else num_encoder_frames)
    num_frames = num_encoder_frames * upsampling_factor
    # the recording's duration to the spectrogram hop below, which NeMo takes from the audio samples
    audio_duration = num_spectrogram_frames * hop_length / sampling_rate
    frame_duration = audio_duration / num_frames

    speaker_probabilities, diarization = None, []
    if speaker_logits is not None:
        logits = speaker_logits[0].detach().float().cpu()
        # as the model: the mean of the probabilities over the spectrogram frames of each encoder frame
        frame_logits = logits[: num_encoder_frames * factor].unflatten(0, (num_encoder_frames, -1))
        speaker_probabilities = _resample_frames(frame_logits.sigmoid().mean(1), num_frames)
        is_active = logits[:num_spectrogram_frames].sigmoid() > speaker_activity_threshold
        diarization = _diarization_segments(is_active.T, hop_length / sampling_rate, audio_duration)
    result = {"words": None, "diarization": diarization, "speaker_tag_to_diarization_speaker": {}, "failed_speakers": [], "error": None}

    for token in SPECIAL_TOKENS:
        text = text.replace(token, " ")
    try:
        streams = _tokenize_transcript(ctc_tokenizer, text, language, blank_id)
    except ValueError as error:
        return {**result, "error": str(error)}
    if not streams:
        return {**result, "error": "The transcript has no word to align."}
    classes = torch.tensor([blank_id] + sorted({token for words in streams.values() for _, token_ids, _ in words for token in token_ids}), device="cpu")
    column_of = {token: column for column, token in enumerate(classes.tolist())}
    if columns is None:
        class_log_probs = word_timestamp_logits[0, :num_frames].detach().log_softmax(-1).float().cpu()[:, classes]
    else:
        position_of = {column: position for position, column in enumerate(columns)}
        class_log_probs = word_timestamp_logits[:num_frames, [position_of[column] for column in classes.tolist()]].float().cpu()

    tag_to_speaker = dict.fromkeys(streams)
    if speaker_probabilities is not None:
        tag_to_speaker = _match_tags_to_speakers(class_log_probs, classes, streams, speaker_probabilities, speaker_activity_threshold, blank_id, blank_id + 1)
    words, failed_tags = [], []
    for tag, stream in streams.items():
        speaker = tag_to_speaker[tag]
        if speaker_probabilities is None:
            probabilities = torch.ones(num_frames, device="cpu")
        elif speaker is None:
            probabilities = torch.zeros(num_frames, device="cpu")  # an unmatched tag has no frame to align on
        else:
            probabilities = speaker_probabilities[:, speaker]
        timestamps = _align_words(class_log_probs, column_of, stream, probabilities, frame_duration, max_word_duration)
        if timestamps is None:
            failed_tags.append(tag)
            continue
        segments = [(s["start"], s["end"]) for s in diarization if s["speaker"] == speaker]
        timestamps = _refine_word_timestamps(timestamps, segments, frame_duration, max_word_frames * upsampling_factor)
        words += [{"word": word["word"], "start": word["start"], "end": word["end"], "speaker": tag, "turn": word["turn"]} for word in timestamps]
    error = f"The words of the speaker tags {failed_tags} could not be aligned." if failed_tags else None
    return {**result, "words": sorted(words, key=lambda word: (word["start"], word["end"])), "speaker_tag_to_diarization_speaker": tag_to_speaker,
            "failed_speakers": failed_tags, "error": error}


def speaker_turns(text, words=None):
    """The speaker turns of a transcript decoded with its special tokens, as the reference `extract_speaker_dict` (everything from the end-of-text
    token on dropped): `{"speaker", "text", "start", "end"}` per `<spk:N>` turn, the text before the first tag (or a text without tags) a turn
    of speaker `None`. `start` / `end` span the turn's aligned `words` (from `align_words`), `None` without them."""
    parts = SPEAKER_TAG_PATTERN.split(text.split(EOS_TOKEN)[0])
    turns = [(turn, int(speaker), content.strip()) for turn, (speaker, content) in enumerate(zip(parts[1::2], parts[2::2]), start=1)]
    if parts[0].strip() or not turns:
        turns.insert(0, (0, None, parts[0].strip()))
    spans = {}
    for word in words or ():
        start, end = spans.get(word["turn"], (word["start"], word["end"]))
        spans[word["turn"]] = min(start, word["start"]), max(end, word["end"])
    return [{"speaker": speaker, "text": content, "start": spans.get(turn, (None, None))[0], "end": spans.get(turn, (None, None))[1]} for turn, speaker, content in turns]
