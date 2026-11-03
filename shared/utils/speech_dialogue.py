# Copyright (c) 2026 DeepBeepMeep - WanGP Community License 2.0 (see LICENSE.txt)
"""Speaker labels shared by WanGP speech pipelines; model-specific cues stay opaque."""
import re


SPEAKER_LINE_PATTERN = re.compile(r"^([ \t]*)Speaker[ \t]*(\d+)[ \t]*:[ \t]*", re.IGNORECASE | re.MULTILINE)
SPEAKER_PATTERN = re.compile(r"Speaker\s*(\d+)\s*:\s*", re.IGNORECASE)


def parse_speaker_names(text):
    """The names given to Speaker 1, 2... in that order, separated by commas or lines; an empty entry keeps "Speaker N"."""
    return [name.replace(":", "").strip() for name in re.split(r"[,\n]", text)]


def speaker_label(number, names):
    """The label of Speaker `number` (from 1): its name when one was given."""
    return names[number - 1] if number <= len(names) and names[number - 1] else f"Speaker {number}"


def parse_dialogue(text, speaker_count, *, line_labels=False, canonical_ids=False):
    """Return zero-based (speaker, text) turns. Continuations belong to the preceding label.

    The historical parser accepts inline labels and maps sorted numeric IDs to
    reference order. New consumers can require line labels and canonical 1..N IDs.
    """
    pattern = SPEAKER_LINE_PATTERN if line_labels else SPEAKER_PATTERN
    group = 2 if line_labels else 1
    matches = list(pattern.finditer(text))
    if not matches:
        raise ValueError(f"{speaker_count}-speaker mode requires Speaker 1: to Speaker {speaker_count}: labels.")
    ids = sorted({int(match.group(group)) for match in matches})
    if len(ids) != speaker_count or (canonical_ids and ids != list(range(1, speaker_count + 1))):
        raise ValueError(f"{speaker_count}-speaker mode requires exactly Speaker 1: to Speaker {speaker_count}:.")
    if canonical_ids and text[:matches[0].start()].strip():
        raise ValueError("Put the first Speaker label before the dialogue text.")
    mapping = {speaker: index for index, speaker in enumerate(ids)}
    turns = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        words = text[match.end():end].strip()
        if words:
            turns.append((mapping[int(match.group(group))], words))
        elif canonical_ids:
            raise ValueError(f"Speaker {match.group(group)} has an empty turn.")
    if not turns:
        raise ValueError("No dialogue text found after Speaker labels.")
    return turns
