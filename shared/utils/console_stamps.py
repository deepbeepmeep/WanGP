# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)

"""Console stamps: a wall-clock stamp at the start of every line the app prints.

Console only: nothing is written to a file and no folder is created. The stamps ride on stdout, so
``python wgp.py ... 2>&1 | tee job.log`` stores them, and ``> job.log`` sends them to a file.

Progress bars stay intact. The app produces three kinds of console traffic: plain print lines,
carriage-return status lines that overwrite themselves in a terminal (the CLI queue loop,
``process_tasks_cli``) and bar redraws (``tqdm`` in ``shared/utils/phase_progress.py`` and the
download bars in ``shared/utils/download.py``). A line is stamped once, at its start, and every
byte of a bar's redraws is echoed untouched -- colour codes included, a terminal is what reads
this output.

Output written straight to the descriptors -- ffmpeg's own stderr, a C library warning -- never
reaches the wrapper, so it keeps the console's own shape: no stamp, no reordering.

A line that carries text is stamped at its start; a line carrying nothing -- a bare ``print()``, a
padded status line closing on a carriage return, a line of spaces -- is echoed unstamped. In a
terminal, the newline written when an interrupt arrives puts the shutdown lines on a line of their
own: the terminal echoes ``^C`` on the very line the app is about to print on, and a stamp written
there would sit inside that echo.

Invisible bytes carry no text. A bar lives on control codes -- ``\x1b[K``, ``\x1b[A``,
``\x1b[?25h``, the ``\x1b]8;;`` link strings, the ``\x03`` an interrupt leaves behind -- the terminal
consumes them and draws nothing, so a line built only of them is echoed unstamped. Counting them as
text stamped a line that printed as nothing: a timestamp with no message after it.

A rule of stars is a header, not a message. The app centres its banners in a field of stars -- the
WanGP version line, the mmgp welcome line, the mmgp memory profile line -- so the line is decoration
around the words inside it, and such a line is echoed unstamped. A stamp in front of it takes the
rule past the width it was built with and breaks the framing the rule exists for, and a terminal
under that width wraps the tail of the rule onto a line of its own. A single star is a bullet --
``* Running on local URL:`` -- and two stars open a bold line, so both keep their stamp.

The last line is closed on the way out. A bar's closing write ends on a carriage return, not a
newline, so the stamp written on that line stayed open and the shell prompt arrived right after it
(``2026-10-22 18:51:17 | user@host:~$``). The stamped streams write one newline at exit -- terminals
only, and only on a line they stamped and never saw close.

A ``Traceback (most recent call last)`` line is the last line a stream is stamped on: the
crash dump under it, and the console prompt that returns after the crash, reach the console
exactly as they are -- a stamped backtrace is noise, and the prompt after a crash wants a
clean line of its own.

The default format is the fixed-width wall clock (``2026-10-12 15:06:41``) followed by a
pipe, because WanGP already uses square brackets for its own tags (``[50/50]``, ``[ERROR]``,
``[INFO]``, tqdm's ``[00:41<00:00, 1.21it/s]``) and syslog reads a bracketed number as a
process id. The frame is just text: the format string is rendered literally, so brackets are
available by writing them.

Off by default, in every mode: a run without the flag prints exactly what it printed before, with
no header line, no stamps and no file. ``--log-timestamp=on`` turns the stamps on for that run,
``WAN2GP_LOG_TIMESTAMP=1`` turns them on for every run; ``--log-timestamp=off`` /
``WAN2GP_LOG_TIMESTAMP=0`` puts the original streams back, byte for byte. The one mode that
refuses them is the MCP server over stdio: its stdout carries JSON-RPC frames, so the feature is
simply not installed there.
"""
from __future__ import annotations

import atexit
import os
import re
import signal
import sys
import threading
import time

_LOCK = threading.RLock()

FORMAT_PRESETS = {
    "full": "%Y-%m-%d %H:%M:%S",
    "clock": "%H:%M:%S",
    "date": "%Y-%m-%d",
    "ms": "%H:%M:%S.{ms}",
    "uptime": "+{uptime}",
    "clock+uptime": "{time} +{uptime}",
}
SEPARATOR_PRESETS = {"pipe": " | ", "space": " ", "dash": " - ", "none": " "}

_TOKEN_RE = re.compile(r"\{(uptime|elapsed|time|date|ms|pid)\}")

DEFAULT_FORMAT = "full"
DEFAULT_SEPARATOR = "pipe"

_STATE = {
    "installed": False,
    "enabled": False,
    "format": DEFAULT_FORMAT,
    "separator": DEFAULT_SEPARATOR,
    "separator_resolved": " | ",
    "stdout": None,
    "stderr": None,
    "streams": [],
    "started": 0.0,
    "interrupt_handler": None,
    "exit_flush": False,
}

def _safe_write(stream, text):
    if stream is None or not text:
        return None
    try:
        return stream.write(text)
    except Exception:
        return None

def _safe_flush(stream):
    if stream is None:
        return
    try:
        stream.flush()
    except Exception:
        return

def _format_duration(seconds):
    """The same human-readable duration shape as the app's own ``format_time``."""
    seconds = max(0.0, float(seconds))
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if seconds >= 60:
        return f"{minutes}m {secs:02d}s"
    return f"{seconds:.1f}s"

def _uptime_text():
    started = _STATE.get("started") or 0.0
    if started <= 0:
        return "0:00:00"
    return _format_duration(time.time() - started).replace(" ", "")

def resolve_format(value):
    fmt = str(value if value is not None else DEFAULT_FORMAT).strip()
    if len(fmt) == 0:
        fmt = DEFAULT_FORMAT
    return FORMAT_PRESETS.get(fmt, fmt)

def resolve_separator(value):
    raw = str(value if value is not None else DEFAULT_SEPARATOR)
    preset = SEPARATOR_PRESETS.get(raw.strip().lower())
    if preset is not None:
        return preset                                        # presets are exact: 'none' means a single space
    separator = raw.strip()
    if not separator.endswith(" "):
        separator = f"{separator} "
    return separator

_STAMP_CACHE = [-1, "", ""]

_LINE_BLANK = " \t\r\x08\n"

# Escape sequences move the terminal, they draw nothing: skip them whole, so a bar's own codes are
# not counted as the text of a line. ``( # % lead a fixed 3-byte charset designator, ESC [ a CSI
# ending on 0x40..0x7E, ESC ] P X ^ _ a string (OSC / DCS / SOS / PM / APC) ending on BEL or ST.
_ESCAPE_FIXED_LEN = {"(": 3, ")": 3, "#": 3, "%": 3}
_ESCAPE_STRINGS = ("]", "P", "X", "^", "_")

_TRACEBACK_MARK = "Traceback (most recent call last)"

def _is_csi_final(one):
    return 0x40 <= ord(one) <= 0x7E

def _escape_span(text, start, stop):
    """How many characters the escape sequence at ``start`` takes up."""
    head = text[start + 1] if start + 1 < stop else ""
    if head in _ESCAPE_FIXED_LEN:
        return _ESCAPE_FIXED_LEN[head]                                 # ESC ( B, ESC ) 0, ESC # 3 ...
    if head == "[":
        index = start + 2                                              # CSI: parameters, then a final byte
        while index < stop:
            if _is_csi_final(text[index]):
                return index + 1 - start
            index += 1
        return stop - start                                   # a CSI split across writes: this chunk is all of it
    if head in _ESCAPE_STRINGS:
        index = start + 2
        while index < stop:
            one = text[index]
            if one == "\x07" or ord(one) == 0x9c:                       # BEL, or the 8-bit ST
                return index + 1 - start
            if one == "\x1b" and index + 1 < stop and text[index + 1] == "\\":
                return index + 2 - start                              # ST: ESC \
            index += 1
        return stop - start
    return 2                                            # a lone ESC, or a pair (ESC \, ESC 7): nothing draws

def _first_drawn(text, start, stop):
    """The index of the first character of the line the terminal draws, ``stop`` when none does.

    Spaces, tabs, carriage returns, backspaces and the closing newline carry no text, and neither do
    the control codes a progress bar is made of: the cursor moves and colour codes, the erase-to-end,
    the show/hide cursor, the OSC link strings, and the bare ``\\x03`` an interrupt leaves on a stream.
    """
    index = start
    stop = min(stop, len(text))
    while index < stop:
        one = text[index]
        if one in _LINE_BLANK:
            index += 1
            continue
        code = ord(one)
        if one == "\x1b":
            index += _escape_span(text, index, stop)
            continue
        if code == 0x9b:                                             # the 8-bit CSI: skip to its final byte
            index += 1
            while index < stop and not _is_csi_final(text[index]):
                index += 1
            index += 1
            continue
        if code < 0x20 or code == 0x7f or (0x80 <= code < 0xa0):
            index += 1                                               # C0 / DEL / C1 controls draw nothing
            continue
        return index                                                   # one drawn character is enough
    return stop

def _carries_text(text, start, stop):
    """True when this line has a character the terminal draws: one is enough to ask for a stamp."""
    return _first_drawn(text, start, stop) < min(stop, len(text))

_STAR_RULE_RUN = 3

def _opens_with_stars(text, start, stop):
    """True when the line opens with a rule of stars: a header, not a message.

    The app centres its banners in a field of stars, so the stars frame the words inside them and the
    line is decoration: it is echoed unstamped and keeps the width it was built with. Three stars is
    the shortest run that is a rule -- the thinnest banner in the app has six on each side. A single
    star is the bullet the server prints, and two stars open a bold line, and both keep their stamp.
    """
    index = _first_drawn(text, start, stop)
    stop = min(stop, len(text))
    run = 0
    while index + run < stop and text[index + run] == "*":
        run += 1
    return run >= _STAR_RULE_RUN

def render_stamp(fmt=None):
    """One rendered stamp, cached per wall-clock second (bars write many lines per second)."""
    fmt = resolve_format(fmt) if fmt is not None else _STATE["format"]
    fmt = resolve_format(fmt)
    if fmt in ("off", "none", "none", ""):
        return ""
    now = int(time.time())
    if _STAMP_CACHE[0] == now and _STAMP_CACHE[1] == fmt:
        return _STAMP_CACHE[2]
    values = {
        "time": time.strftime("%H:%M:%S"),
        "date": time.strftime("%Y-%m-%d"),
        "ms": f"{int((time.time() % 1) * 1000):03d}",
        "uptime": _uptime_text(),
        "elapsed": _uptime_text(),
        "pid": str(os.getpid()),
    }
    text = time.strftime(fmt)
    if "{" in fmt:
        text = _TOKEN_RE.sub(lambda match: values[match.group(1)], text)   # a token mix is strftime'd first
    _STAMP_CACHE[0], _STAMP_CACHE[1], _STAMP_CACHE[2] = now, fmt, text
    return text


class _LineStamper:
    """Write a stamp at the start of every line of one console stream.

    ``write`` chunks are echoed verbatim, so a bar keeps redrawing exactly as it does today, its
    ``ljust`` padding intact, and the terminal receives every byte it expects -- colour codes
    included. One stamp is written per line, at its start: a carriage return continues the same
    line, so a bar is stamped once, and a closing newline is what asks for the next stamp. A line
    that carries nothing -- a bare ``print()``, a padded status line closing on a carriage return,
    a line of spaces -- is echoed unstamped, and a line split across writes keeps its one stamp.
    A line that opens with a rule of stars -- a banner -- is a header, and it too is echoed
    unstamped, so its rule keeps the width the app built it with.
    A ``Traceback (most recent call last)`` line is the last stamp the stream writes: a crash
    dump and the prompt that returns after it read clean.
    """

    def __init__(self, console):
        self._console = console
        self._need_stamp = True
        self._stamps_stopped = False
        self._stamp_open = False        # a stamp was written and its line has not closed yet

    def feed(self, text):
        if not text:
            return
        position = 0
        while position < len(text):
            index = text.find("\n", position)
            stop = index if index >= 0 else len(text)
            if (self._need_stamp and not self._stamps_stopped and _carries_text(text, position, stop)
                    and not _opens_with_stars(text, position, stop)):
                stamp = render_stamp()
                if stamp:
                    _safe_write(self._console, f"{stamp}{_STATE['separator_resolved']}")
                    self._stamp_open = True
                self._need_stamp = False
                if text[position:stop].lstrip(" \t").startswith(_TRACEBACK_MARK):
                    self._stamps_stopped = True   # the last stamp this stream writes
            _safe_write(self._console, text[position:stop + (1 if index >= 0 else 0)])
            if index < 0:
                return
            position = index + 1
            self._need_stamp = True
            self._stamp_open = False                    # the line closed: the stamp is on a finished line

    def close_open_line(self):
        """Close the line this stream left open under a stamp: the prompt must start on its own line."""
        if not self._stamp_open:
            return False
        self._stamp_open = False
        _safe_write(self._console, "\n")
        _safe_flush(self._console)
        return True

    def flush(self):
        _safe_flush(self._console)



class ConsoleStampStream:
    """A ``sys.stdout`` / ``sys.stderr`` replacement: the console, stamped at line starts.

    The stream contract the app relies on is preserved: ``encoding``, ``errors``, ``isatty``,
    ``fileno``, ``buffer`` and friends pass through to the wrapped stream (``shared/deepy/
    engine.py`` and ``shared/prompt_enhancer/qwen35_text.py`` probe ``encoding``, and tqdm
    decides how to redraw from ``isatty``). ``fileno`` keeps returning the *real* descriptor,
    so a subprocess started without redirection still writes to the real console.
    """

    def __init__(self, console):
        self._stamper = _LineStamper(console)

    def write(self, data):
        if not data:
            return 0
        text = data.decode("utf-8", "replace") if isinstance(data, (bytes, bytearray)) else str(data)
        with _LOCK:
            self._stamper.feed(text)
        return len(text)

    def flush(self):
        with _LOCK:
            self._stamper.flush()

    def close_open_line(self):
        with _LOCK:
            return self._stamper.close_open_line()

    @property
    def encoding(self):
        return getattr(self._stamper._console, "encoding", "utf-8")

    @property
    def errors(self):
        return getattr(self._stamper._console, "errors", "strict")

    def isatty(self):
        return bool(getattr(self._stamper._console, "isatty", lambda: False)())

    def writable(self):
        return True

    def fileno(self):
        return getattr(self._stamper._console, "fileno")()

    def __getattr__(self, name):
        return getattr(self._stamper._console, name)


def _argv_value(argv, name):
    argv = [str(one) for one in (argv or [])]
    prefix = f"{name}="
    for index, one in enumerate(argv):
        if one == name and index + 1 < len(argv):
            return argv[index + 1]
        if one.startswith(prefix):
            return one[len(prefix):]
    return None


ON_OFF_TRUE = ("on", "yes", "true", "1")
ON_OFF_FALSE = ("off", "no", "false", "0")


def _on_off_value(raw):
    value = str(raw).strip().lower()
    if value in ON_OFF_TRUE:
        return True
    if value in ON_OFF_FALSE:
        return False
    return None


def _argv_on_off(argv, name):
    """True or False when the command line sets the option, None when it is left out.

    The flag on its own means on; the value form (--name=off, or --name with the value next to
    it) decides both ways.
    """
    argv = [str(one) for one in (argv or [])]
    prefix = f"{name}="
    for index, one in enumerate(argv):
        text = one.strip()
        if text.startswith(prefix):
            return _on_off_value(text[len(prefix):])
        if text == name:
            following = argv[index + 1].strip() if index + 1 < len(argv) else ""
            value = _on_off_value(following)
            return True if value is None else value                  # a flag on its own is an on
    return None


def _uses_stdio_protocol(argv):
    """MCP stdio carries JSON-RPC on stdout: that stream must never be stamped."""
    argv = [str(one) for one in (argv or [])]
    if not any(one == "--mcp" for one in argv):
        return False
    transport = _argv_value(argv, "--mcp-transport") or "stdio"
    return transport.strip().lower() == "stdio"


def console_stamps_enabled():
    return bool(_STATE["installed"] and _STATE["enabled"])


def _wrap_streams():
    """Wrap the live ``sys.stdout`` / ``sys.stderr`` with the stamping streams."""
    streams = [one for one in (_STATE.get("streams") or []) if isinstance(one, ConsoleStampStream)]
    for name in ("stdout", "stderr"):
        original = getattr(sys, name)
        if original is None or isinstance(original, ConsoleStampStream):
            continue
        if _STATE.get(name) is None:
            _STATE[name] = original
        wrapped = ConsoleStampStream(original)
        setattr(sys, name, wrapped)
        streams.append(wrapped)
    _STATE["streams"] = streams
    return streams


def _restore_streams():
    for name in ("stdout", "stderr"):
        original = _STATE.get(name)
        if original is not None:
            setattr(sys, name, original)
    _STATE["streams"] = []


def close_open_lines():
    """One newline per stamped console stream that ends its life on an open line.

    A bar's closing write ends on a carriage return, so the stamp written on that line stayed open and
    the shell prompt printed right after it. The newline goes to the real console, and only for a line
    this stream stamped: a run that ends mid-bar, a pipe, a file: nothing is added.
    """
    closed = False
    with _LOCK:
        for stream in list(_STATE.get("streams") or []):
            if not isinstance(stream, ConsoleStampStream):
                continue
            try:
                if not bool(stream.isatty()):
                    continue
            except Exception:
                continue
            closed = bool(stream.close_open_line()) or closed
    return closed


def _arm_exit_flush():
    """Register the closing newline once, when the streams are stamped (atexit runs it last)."""
    if _STATE.get("exit_flush") or not _STATE["streams"]:
        return False
    _STATE["exit_flush"] = True
    atexit.register(close_open_lines)
    return True


_INTERRUPT_NEWLINE = b"\n"

def _interrupt_newline():
    """One newline straight to the terminal when the interrupt lands.

    Ctrl+C is echoed by the terminal itself, on the line the app is about to print on, so a
    shutdown message starts glued to it (``^C2026-10-13 15:50:10 | Keyboard interruption ...``)
    and the stamp ends up inside that echo. A newline written to the descriptor -- not to the
    stamped stream, a stamped empty line would defeat the point -- puts the message on its own
    line. A process with no parent of its own (the forked probes) has no terminal to tidy.
    """
    try:
        if os.getpid() == os.getppid():
            return
    except OSError:
        return
    for descriptor in (1, 2):
        try:
            if os.isatty(descriptor):
                os.write(descriptor, _INTERRUPT_NEWLINE)
        except OSError:
            continue
        break            # one newline is enough: the first console descriptor that is a terminal

def _chain_interrupt(signum, frame, previous):
    """Write the newline, then run the handler the interrupt had before the stamps existed."""
    _interrupt_newline()
    if previous is signal.SIG_DFL:
        raise KeyboardInterrupt                       # what the default handler means to Python
    if previous is signal.SIG_IGN:
        return
    if callable(previous):
        previous(signum, frame)
        return
    raise KeyboardInterrupt

def _arm_interrupt_newline():
    """Chain onto the SIGINT handler in force, terminals only; called when the stamps install."""
    previous = signal.getsignal(signal.SIGINT)
    if callable(previous) and getattr(previous, "__wgp_interrupt_chain__", False):
        return                                        # the stamps are already chained
    def handler(signum, frame):
        _chain_interrupt(signum, frame, previous)
    handler.__wgp_interrupt_chain__ = True
    try:
        signal.signal(signal.SIGINT, handler)
    except (ValueError, OSError, RuntimeError, AttributeError):
        return                                        # not the main thread, or no signal handling
    _STATE["interrupt_handler"] = handler
    return handler


def _apply_environment(argv):
    env_stamp = _on_off_value(os.environ.get("WAN2GP_LOG_TIMESTAMP", ""))
    return {
        "enabled": env_stamp is True,
        "env_stamp_set": env_stamp is not None,
        "format": os.environ.get("WAN2GP_LOG_TIMESTAMP_FORMAT") or DEFAULT_FORMAT,
        "separator": os.environ.get("WAN2GP_LOG_TIMESTAMP_SEP") or DEFAULT_SEPARATOR,
    }


def install_console_stamps(argv=None, *, fmt=None, separator=None, quiet=False):
    """Wrap sys.stdout / sys.stderr so every line the app prints starts with a stamp.

    No file is opened and no folder is created: the stamps go to the console and nowhere else.
    Called from ``wgp.py`` before the heavy imports, so model loading output is stamped too --
    the options are read from the raw command line and the environment here, and the parsed
    ``argparse`` values are applied afterwards by :func:`configure_console_stamps`.
    Stamping is off unless ``--log-timestamp=on`` (or ``WAN2GP_LOG_TIMESTAMP=1``) asks for it, so a
    plain run keeps the legacy output. Returns True when the streams are stamped.
    """
    if _STATE["installed"]:
        return console_stamps_enabled()
    argv = list(sys.argv[1:] if argv is None else argv)
    env = _apply_environment(argv)
    # A flag given on the command line takes effect from the very first printed line: this runs
    # before the heavy imports, while the parsed argparse values land later, in
    # configure_console_stamps. Explicit keyword arguments still win over the flags.
    if fmt is None:
        fmt = _argv_value(argv, "--log-timestamp-format")
    if separator is None:
        separator = _argv_value(argv, "--log-timestamp-sep")
    requested = _argv_on_off(argv, "--log-timestamp")
    if requested is not None:
        enabled = requested                                             # the flag wins over everything
    elif env["env_stamp_set"]:
        enabled = env["enabled"]                                        # WAN2GP_LOG_TIMESTAMP sets the default
    else:
        enabled = False                                                 # the default is the legacy plain output
    enabled = bool(enabled) and not _uses_stdio_protocol(argv)
    fmt = fmt if fmt is not None else env["format"]
    separator = separator if separator is not None else env["separator"]
    _STATE.update(
        enabled=enabled,
        format=resolve_format(fmt),
        separator=resolve_separator(separator),
        separator_resolved=resolve_separator(separator),
        started=time.time(),
    )
    _STATE["installed"] = True
    if not enabled:
        return False

    _wrap_streams()
    _arm_exit_flush()                # a stamp is never left open when the process ends
    if _STATE["streams"] and all(one.isatty() for one in _STATE["streams"]):
        _arm_interrupt_newline()       # terminals only: the interrupt gets a clean line
    if not quiet:
        mode = str(_STATE.get("format") or DEFAULT_FORMAT)
        detail = ("time of day" if "uptime" not in mode
                  else "time of day + uptime")
        print(f"WanGP console log start: PID {os.getpid()}, python {sys.version.split()[0]}, "
              f"stamps {detail} at the start of every line, format '{mode}'")
    return True


def configure_console_stamps(args, argv=None):
    """Apply the parsed command-line options (called right after ``parse_wgp_args``).

    A flag left out keeps the value coming from the environment and the defaults, so
    ``WAN2GP_LOG_TIMESTAMP``, ``WAN2GP_LOG_TIMESTAMP_FORMAT`` & co. work without a flag.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    get = (lambda key, default=None: getattr(args, key, default))
    stamps = _argv_on_off(argv, "--log-timestamp")
    if stamps is False:
        return disable_console_stamps()
    if stamps is True and not console_stamps_enabled():
        install_console_stamps(argv)
    if not console_stamps_enabled():
        return ""                                # the modes that print as before: nothing to apply

    fmt = get("log_timestamp_format", None)
    separator = get("log_timestamp_sep", None)

    with _LOCK:
        if fmt is not None:
            _STATE["format"] = resolve_format(fmt)
        if separator is not None:
            _STATE["separator_resolved"] = resolve_separator(separator)
            _STATE["separator"] = _STATE["separator_resolved"]
    return ""


def disable_console_stamps():
    """``--log-timestamp=off``: put the original streams back, output is byte-identical to a plain run."""
    _STATE["enabled"] = False
    close_open_lines()                        # never hand the console over with a stamp open on it
    _restore_streams()
    if _STATE.get("exit_flush"):
        _STATE["exit_flush"] = False
        atexit.unregister(close_open_lines)
    return ""


def console_stamp_state():
    return {
        "enabled": console_stamps_enabled(),
        "format": _STATE.get("format"),
        "separator": _STATE.get("separator_resolved"),
    }
