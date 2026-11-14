# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Child processes: WanGP never hands its terminal to the helper programs it starts.

Every `subprocess` launch in WanGP that does not say otherwise gives the child the terminal, keyboard
included, as its standard input. Most helpers ignore it. ffmpeg does not: its own documentation says
to start it with `-nostdin`, or with its input sent to `/dev/null`, precisely so it never reads the
keyboard -- "useful if you launch ffmpeg in the background", which is how WanGP runs it. A tool that
does read the keyboard (ffmpeg, a TTS binary, a GGUF llama binary) switches echo, line editing and
Ctrl+C off on the shared terminal, and WanGP's children live in their own process groups (upstream's
`shared/imageio_ffmpeg_patch.py` enforces that for ffmpeg, to keep Ctrl+C from hitting a half-written
file), so an interrupt never reaches them: they are killed hard, and the terminal stays broken.

`child_kwargs` is the one answer, for the launches WanGP makes itself -- the same shape imageio-ffmpeg
uses for the video writing it drives, and the same shape the ffmpeg launches in
`shared/deepy/audio_inspection.py` and `shared/utils/video_decode.py` already ask for with `-nostdin`.

It is only for children that take no keyboard input, so a site that already sets `stdin=` -- the ones
feeding ffmpeg raw frames (`plugins/media_flow/media_io.py`, `postprocessing/dlss5/runtime.py`,
`shared/utils/video_decode.py`) -- is left exactly as it is, and a site that passes data with
`input=` / `communicate()` keeps its pipe.
"""
from __future__ import annotations

import os
import subprocess

_IS_WINDOWS = os.name == "nt"
_CREATE_NO_WINDOW = 0x08000000                       # CREATE_NO_WINDOW: no console window for the child
_CREATE_NEW_PROCESS_GROUP = 0x00000200               # CREATE_NEW_PROCESS_GROUP: Ctrl+C is not theirs


def child_kwargs(**overrides):
    """The `subprocess` options for a helper program that must not own the terminal.

    No keyboard input, and its own process group: an interrupt typed at the terminal goes to WanGP,
    which is the one that knows how to clean up. Call it last in the argument list
    (`subprocess.run(command, **child_kwargs())`), and pass `overrides` to adjust a specific launch.
    """
    kwargs = {"stdin": subprocess.DEVNULL}
    if _IS_WINDOWS:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", _CREATE_NO_WINDOW)
        flags |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", _CREATE_NEW_PROCESS_GROUP)
        kwargs["creationflags"] = flags
    else:
        kwargs["start_new_session"] = True
    kwargs.update(overrides)
    return kwargs


def keeps_terminal(kwargs):
    """True when the options a launch site already passes keep the terminal -- leave such a site alone."""
    return any(name in kwargs for name in ("stdin", "preexec_fn", "start_new_session", "creationflags"))


def no_stdin_args(*binaries):
    """ffmpeg's own `-nostdin`, for the commands built as a list: ffprobe has no such option."""
    args = []
    for binary in binaries:
        name = os.path.basename(str(binary)).strip().lower()
        if name.startswith("ffmpeg"):
            args.append("-nostdin")
    return args


def reads_stdin(command):
    """True when a command list takes its input from a pipe (`pipe:0`, `-i -`): keep its stdin open."""
    parts = [str(one) for one in (command or [])]
    for index, part in enumerate(parts):
        if part.startswith("pipe:") or part == "-":
            return True
        if part in ("-i", "--input") and index + 1 < len(parts) and parts[index + 1] in ("-", "pipe:0"):
            return True
    return False
