# Adapted from Zyphra/ZONOS3-Mini (model/src/zonos3/tts/steering.py, studio/src/state/engine.ts), commit 9fd14e51.
# Copyright (c) 2026 Zyphra, Inc. MIT License, see LICENSE_ZONOS3. WanGP additions: the Voice Controls syntax, the styles and the
# inline cues.
"""Steering controls of ZONOS3-Mini: measured direction vectors, no model change. A control value v uses its map's "+" doses times v
(the "-" doses times |v| below 0). Speaker-embedding ("raw") directions move the speaker embedding, their sum rescaled to its norm
(speaker-embedding conditioning only); hidden-state directions add to a Full AttnRes source on every generated frame, full strength
for 4 s then held at 35 % (steer then hold: a constant push drifts over long lines), except the pace and whisper directions.
Inline cues change the hidden-state controls within a take, from where their text is read."""
import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file

META = json.loads(Path(__file__).with_name("steering.json").read_text(encoding="utf-8"))
RAW, HIDDEN = "raw", "attnres_source"
CONTROLS = {control["id"]: control for control in META["controls"]}
ACCENTS = {accent["id"]: accent for accent in META["accents"]}
# Styles, written as inline cues ([happy], [sad 1.5]) or in Voice Controls: control values, scaled by the strength (1 = tested).
STYLES = {
    "neutral": ("Neutral", {}),
    "happy": ("Happy", {"happy": 4}), "sad": ("Sad", {"sad": 6}), "angry": ("Angry", {"angry": 4}), "calm": ("Calm", {"calm": 4}),
    "scared": ("Scared", {"scared": 10}), "confused": ("Confused", {"confused": 3}),
    "excited": ("Excited", {"arousal": 3, "valence": 2}), "gloomy": ("Gloomy", {"valence": -3, "arousal": -2}),
    "whisper": ("Whispered", {"projection": -1.2}), "loud": ("Loud", {"projection": 1.5}),
    "chatty": ("Chatty", {"chatty": 3}), "reading": ("Careful Reading", {"chatty": -3}),
    "lively": ("Lively", {"lively": 6}), "flat": ("Flat", {"lively": -6}), "hesitant": ("Hesitant", {"hesitant": 6}),
    "slow": ("Slow", {"pace": -1}), "fast": ("Fast", {"pace": 1}),
}
STYLE_WORDS = {**{key: key for key in STYLES}, **{label.lower(): key for key, (label, _) in STYLES.items()}, "whispering": "whisper"}
MAX_STRENGTH = 2.0
# Emotion controls: a cue's emotion replaces the emotion of the Voice Controls instead of mixing with it.
MOODS = {name for name, control in CONTROLS.items() if all(direction.split(".")[0] in ("mood", "emotions") for side in control["map"].values() for direction in side)}
_ITEM = re.compile(r"^\s*([a-z_]+(?::[a-z_]+)?)\s*[=:]\s*([+-]?\d+(?:\.\d+)?)\s*$", re.IGNORECASE)
_STYLE = re.compile(r"^([a-z][a-z ]*?)(?:\s+(\d+(?:\.\d+)?))?$")
OPTIONS = {"conditioning": ("embedding", "prefix"), "normalize": ("on", "off")}


def _clip(name, value):
    low, high = (ACCENTS[name[len("accent:"):]] if name.startswith("accent:") else CONTROLS[name])["range"]
    return min(max(value, low), high)


def parse_controls(text, cue=False):
    """Voice Controls text, or the inside of an inline cue (cue=True) -> ({control id or accent:<id>: value}, {option: value}). Items
    are separated by commas: styles with an optional strength (`happy`, `sad 1.5`), control values (`happy=3, pace=-1,
    accent:irish=1.5`) and, in Voice Controls, the options `conditioning=embedding`, `normalize=off`. Values add up, clipped to their
    tested range. Cues take only the controls that can change within a take. Raises ValueError for anything else."""
    where = f"Cue [{text}]" if cue else "Voice Controls"
    if cue and not text.strip():
        raise ValueError("Empty cue []: write a style such as [happy], or [neutral] to go back to the Voice Controls.")
    values, options = {}, {}
    for item in re.split(r"[,;\n]", text or ""):
        item = item.strip()
        if not item:
            continue
        key, equal, value = item.partition("=")
        key, value = key.strip().lower(), value.strip().lower()
        if equal and key in OPTIONS and not cue:
            if value not in OPTIONS[key]:
                raise ValueError(f"Voice Controls: {key} must be {' or '.join(OPTIONS[key])}")
            options[key] = value
            continue
        match = _ITEM.match(item)
        if match is not None:
            name = match.group(1).lower()
            if (ACCENTS.get(name[len("accent:"):]) if name.startswith("accent:") else CONTROLS.get(name)) is None:
                raise ValueError(f"{where}: unknown control '{name}'. Controls: {', '.join(CONTROLS)}; accents: "
                                 + ", ".join(f"accent:{key}" for key in ACCENTS))
            if cue and not CONTROLS.get(name, {}).get("spannable"):
                raise ValueError(f"{where}: {name} changes the voice itself and cannot change within a take: set it in Voice Controls.")
            added = {name: float(match.group(2))}
        else:
            style = _STYLE.match(item.lower())
            key = STYLE_WORDS.get(style.group(1)) if style else None
            if key is None:
                raise ValueError(f"{where}: unknown {'cue' if cue else 'item'} '{item}'. Styles: {', '.join(STYLES)}, with an optional "
                                 f"strength ([sad 1.5]); or control values such as {'[happy=3, pace=-1]' if cue else 'happy=3, pace=-1'} (see the help).")
            strength = float(style.group(2) or 1)
            if strength > MAX_STRENGTH:
                raise ValueError(f"{where}: the strength of {key} goes up to {MAX_STRENGTH:g}.")
            added = {name: number * strength for name, number in STYLES[key][1].items()}
        for name, number in added.items():
            values[name] = values.get(name, 0.0) + number
    return {name: _clip(name, number) for name, number in values.items()}, options


def cue_values(default, cue):
    """The control values of a cued stretch, as (those kept from the Voice Controls, the cue's own): the cue's controls replace the
    same ones of the Voice Controls, and a cued emotion replaces their emotion."""
    if any(name in MOODS for name in cue):
        default = {name: value for name, value in default.items() if name not in MOODS}
    return {name: value for name, value in default.items() if name not in cue}, cue



def doses(values):
    """Control values -> direction doses ({"<set>.<direction>": dose})."""
    out = {}
    for name, value in values.items():
        if name.startswith("accent:"):
            mapping, scale = ACCENTS[name[len("accent:"):]]["map"], value
        else:
            mapping, scale = CONTROLS[name]["map"].get("+" if value > 0 else "-"), abs(value)
        for direction, weight in (mapping or {}).items():
            out[direction] = out.get(direction, 0.0) + weight * scale
    return {direction: dose for direction, dose in out.items() if dose}


def is_raw(direction):
    return META["sets"][direction.split(".")[0]]["space"] == RAW


class Steering:
    def __init__(self, vectors_path):
        self.vectors = load_file(str(vectors_path))
        self.sources = sorted({entry["source_index"] for entry in META["sets"].values() if entry["space"] == HIDDEN})  # steered bank sources
        release = META["steer_release"]
        self.exempt = set(release["exempt"])
        self.onset_seconds, self.ramp_seconds, self.hold = release["onset_seconds"], release["ramp_seconds"], release["hold"]

    def move_embedding(self, embedding, dose_map):
        """The speaker embedding moved by the speaker-embedding directions, rescaled to its norm."""
        raw = {direction: dose for direction, dose in dose_map.items() if is_raw(direction)}
        if not raw:
            return embedding
        base = embedding.float().reshape(-1)
        moved = base + sum(dose * self.vectors[direction] for direction, dose in raw.items())
        return moved * (base.norm() / moved.norm().clamp_min(1e-8))

    def hidden(self, dose_map):
        """(constant, released) deltas [n_sources, hidden] of the hidden-state directions, None when absent."""
        def delta(selected):
            if not selected:
                return None
            out = torch.zeros(len(self.sources), self.vectors[next(iter(selected))].numel(), device="cpu")
            for direction, dose in selected.items():
                source = META["sets"][direction.split(".")[0]]["source_index"]
                out[self.sources.index(source)] += dose * self.vectors[direction]
            return out
        hidden = {direction: dose for direction, dose in dose_map.items() if not is_raw(direction)}
        return (delta({d: v for d, v in hidden.items() if d in self.exempt}), delta({d: v for d, v in hidden.items() if d not in self.exempt}))

    def release_weight(self, frame, frame_rate):
        """Weight of the released controls at generated frame `frame`: 1, then a linear ramp to the hold value."""
        onset, ramp = round(self.onset_seconds * frame_rate), round(self.ramp_seconds * frame_rate)
        if frame < onset:
            return 1.0
        if frame >= onset + ramp:
            return self.hold
        return 1.0 + (self.hold - 1.0) * (frame - onset) / ramp
