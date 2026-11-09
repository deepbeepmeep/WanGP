# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Exploration instructions of world prompts, shared by the world generators: after the world description, one line per move of the
camera through the world, as the Worlds gallery's explorer makes them with the keyboard.

    A narrow cobblestone street in an old town on a sunny afternoon.
    [Forward]
    [Forward 2s]
    [Forward+Look_Left]
    [Look_Up]
    [Wait]
    [Event] a cat jumps onto the fountain
    [Forward] [Event] the street lamps light up

- Moves are relative to the view: [Forward], [Backward], [Left] and [Right] walk and step sideways, [Look_Left], [Look_Right],
  [Look_Up] and [Look_Down] turn the camera, [Wait] keeps it still. Commands joined by + or written on the same line act together.
- A move lasts one explorer move (about 1 s) unless it gives its length in seconds: [Forward 2s], [Look_Left 1.5].
- [Event] makes the rest of its line happen during the move (an action of the world's subject when it names it, otherwise a change of
  the scene); a move with an event lasts long enough for it to happen.
- Names ignore case, spaces and dashes ([look up], [Look-Up]). A line whose first bracket holds anything else than a command name, such
  as a JSON array, belongs to the description.

The commands a world model understands are those of the explorer controls its handler names (query_world_explorer "controls"), from
the shared catalogue CONTROLS: "walk" and "look" give the moves above, "event" the text events, and game models add their player's actions
([Jump], [Sprint], [Crouch], [Interact], [Attack]). [Wait] is always understood. Generators map the moves to their own inputs through the
explorer's key names: w / s walk, a / d step sideways, the arrows turn, then the actions' own names.
"""
from __future__ import annotations

import re

# The Worlds explorer's shared controls: each gives its commands in world prompts (command -> explorer key name, None: no key), the keys
# that make them (KeyboardEvent.code -> key name, by position: the same physical keys on every layout; letters are labelled as the
# keyboard prints them, other keys by "labels") and its part of the explorer's help and of its "Hold ..." hint ({key}: that key's label)
CONTROLS = {
    "walk": {"commands": {"forward": "w", "backward": "s", "left": "a", "right": "d"}, "keys": {"KeyW": "w", "KeyA": "a", "KeyS": "s", "KeyD": "d"},
             "help": "{w} {a} {s} {d}: Move", "hint": "{w} {a} {s} {d} to move"},
    "look": {"commands": {"look_up": "up", "look_down": "down", "look_left": "left", "look_right": "right"},
             "keys": {"ArrowUp": "up", "ArrowDown": "down", "ArrowLeft": "left", "ArrowRight": "right"},
             "labels": {"up": "↑", "down": "↓", "left": "←", "right": "→"}, "help": "Arrows: Look", "hint": "the arrows to look"},
    "jump": {"commands": {"jump": "jump"}, "keys": {"Space": "jump"}, "labels": {"jump": "Space"}, "help": "Space: Jump", "hint": "Space to jump"},
    "sprint": {"commands": {"sprint": "sprint"}, "keys": {"ShiftLeft": "sprint", "ShiftRight": "sprint"}, "labels": {"sprint": "Shift"}, "help": "Shift: Sprint"},
    "crouch": {"commands": {"crouch": "crouch"}, "keys": {"KeyC": "crouch"}, "help": "{crouch}: Crouch"},
    "interact": {"commands": {"interact": "interact"}, "keys": {"KeyE": "interact"}, "help": "{interact}: Interact"},
    "attack": {"commands": {"attack": "attack"}, "keys": {"KeyF": "attack"}, "help": "{attack}: Attack"},
    "event": {"commands": {"event": None}, "help": "Enter: Event", "hint": "or type an event"},  # the rest of the line happens during the move
}
CAMERA_CONTROLS = ("walk", "look", "event")  # a walking camera with text events
_ALIASES = {"back": "backward"}
_ALWAYS = {"wait": None}
_LINE = re.compile(r"\[\s*[A-Za-z]")  # an instruction line opens with a bracketed name
_BRACKET = re.compile(r"\[([^\]]*)\]\s*")
_PART = re.compile(r"\s*([A-Za-z][A-Za-z _-]*?)\s*(?:(\d+(?:\.\d+)?)\s*s?)?\s*")


def is_instruction(line):
    return _LINE.match(line.strip()) is not None


def _command(part, line, commands):
    match = _PART.fullmatch(part)
    name = None if match is None else re.sub(r"[\s-]+", "_", match.group(1).strip().lower())
    name = _ALIASES.get(name, name)
    if name not in commands:
        raise ValueError(f"Unknown world instruction [{part.strip()}] in '{line.strip()}' for this world model: use " + ", ".join(f"[{name.title()}]" for name in commands))
    return name, None if match.group(2) is None else float(match.group(2))


def commands_of(controls):
    """The commands of world prompts for the explorer controls `controls` (names of CONTROLS), mapped to their key names."""
    return {**_ALWAYS, **{command: key for name in controls for command, key in CONTROLS[name]["commands"].items()}}


def parse_instruction(line, controls=CAMERA_CONTROLS):
    """{"keys": explorer key names held during the move, "seconds": its length or None (the generator's default), "event": text or "",
    "text": the line} of one instruction line, for a world model with the explorer controls `controls`."""
    commands = commands_of(controls)
    rest, keys, seconds, event = line.strip(), [], None, None
    while rest.startswith("["):
        match = _BRACKET.match(rest)
        if match is None:
            raise ValueError(f"Unclosed world instruction: '{line.strip()}'")
        rest = rest[match.end():]
        for part in match.group(1).split("+"):
            name, length = _command(part, line, commands)
            if name == "event":
                event = ""
            elif commands[name] is not None and commands[name] not in keys:
                keys.append(commands[name])
            seconds = length if seconds is None or (length is not None and length > seconds) else seconds
        if event is not None:
            break
    if event is None and rest:
        if rest.lstrip().startswith("+"):  # [Left] + [Look_Right]: moves held together share one bracket
            joined = "[" + "+".join(part.strip() for part in re.findall(r"\[([^\]]*)\]", line)) + "]"
            raise ValueError(f"Moves made together go in one bracket, {joined}, not '{line.strip()}'")
        raise ValueError(f"Text after a world instruction needs an [Event] before it: '{line.strip()}'")
    return {"keys": keys, "seconds": seconds, "event": (rest if event is not None else "").strip(), "text": line.strip()}


def split_instructions(prompt, controls=CAMERA_CONTROLS):
    """(description, instructions) of a world prompt for a world model with the explorer controls `controls` (its handler's
    query_world_explorer "controls"): the lines that are not instructions, joined, and the parsed instruction lines in order."""
    description, instructions = [], []
    for line in prompt.splitlines():
        if is_instruction(line):
            instructions.append(parse_instruction(line, controls))
        else:
            description.append(line)
    return "\n".join(description).strip(), instructions
