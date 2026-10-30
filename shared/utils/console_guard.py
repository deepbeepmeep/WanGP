# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Console guard: WanGP hands the terminal back in the state it received it.

WanGP starts helper programs -- ffmpeg, ffprobe, the DLSS runtime, the TTS and GGUF binaries -- and
they all inherit the terminal as their standard input. A native tool that reads the keyboard takes
the terminal over: echo, line editing and Ctrl+C are switched off on the shared terminal, and when
that tool is killed -- WanGP's children live in their own process groups, so Ctrl+C never reaches
them and they are killed without a chance to clean up -- the terminal stays in that state. The shell
prompt stops answering: the user types blind, and `reset` is the only way back. Nothing in WanGP ever
looked at the terminal, so nothing ever put it back.

The guard does the two things a terminal application owes its user:

* At startup it reads the settings of the terminal it runs on (fd 0, then 1, then 2). If the terminal
  arrives broken -- a previous run left it raw -- it repairs the flags a terminal needs to be typed
  in, so WanGP starts on a working console. The repaired settings are what it remembers.
* On the way out it writes those settings back: at exit, on Ctrl+C (SIGINT), on SIGTERM, and from the
  app's own shutdown paths (`quit_application`, `restart_application`, which ends in `os._exit` and
  so skips the exit handlers). On an interrupt it restores first, then runs the handler that was
  installed before it, so a crash, a graceful shutdown and a restart all restore the terminal.

Invisible when there is nothing to guard: no terminal (output to a pipe, a file), `--console-guard=off`,
`WAN2GP_CONSOLE_GUARD=0`, the MCP server over stdio, or no terminal access. No key is ever written:
the guard only touches the settings, on the terminal WanGP was started on.
"""
from __future__ import annotations

import atexit
import os
import signal
import sys

_POSIX = os.name == "posix"
_WINDOWS = os.name == "nt"
if _POSIX:
    import termios

_STATE = {
    "installed": False,
    "enabled": False,
    "report": False,
    "repaired": False,
    "fd": None,               # POSIX: the terminal descriptor the settings were read from
    "saved": None,            # POSIX: the termios settings to write back
    "handles": [],            # Windows: (handle, mode) pairs to write back
    "handlers": [],
}

# The settings a terminal needs before a person can type in it. Ctrl+C is a signal (iflags), the rest
# are the line discipline and the output fixes (oflags / lflags); everything the user set is left as is.
_GUARD_IFLAGS = termios.ISIG if _POSIX else 0
_GUARD_OFLAGS = termios.OPOST if _POSIX else 0
_GUARD_LFLAGS = (termios.ICANON | termios.ECHO | termios.ECHOCTL | termios.ECHOE | termios.ECHOKE |
                 termios.IEXTEN) if _POSIX else 0

_READ_IFLAGS = ("BRKINT", "ICRNL", "INPCK", "ISTRIP", "IXON", "IXANY", "IXOFF", "IMAXBEL", "IUTF8")
_READ_OFLAGS = ("OPOST", "ONLCR", "OCRNL", "ONOCR", "ONLSTRIP")
_READ_LFLAGS = ("ICANON", "ECHO", "ECHOE", "ECHOK", "ECHOCTL", "ECHOKE", "NOFLSH", "TOSTOP", "IEXTEN")


def _flag_text(attrs):
    """The terminal settings as one readable line, for the log."""
    if attrs is None:
        return "n/a"
    on = []
    for name in _READ_IFLAGS:
        if attrs[0] & getattr(termios, name, 0):
            on.append(name)
    if attrs[0] & termios.ISIG:
        on.append("ISIG")
    for name in _READ_OFLAGS:
        if attrs[1] & getattr(termios, name, 0):
            on.append(name)
    for name in _READ_LFLAGS:
        if attrs[3] & getattr(termios, name, 0):
            on.append(name)
    return " ".join(sorted(on))


def _read_flags():
    """The settings of the terminal WanGP runs on: (fd, attrs), or (None, None) when there is none."""
    for fd in (0, 1, 2):
        try:
            if not os.isatty(fd):
                continue
            return fd, list(termios.tcgetattr(fd))
        except (OSError, termios.error):
            continue
    return None, None


def _typable(attrs):
    """The same settings with the flags a terminal needs to be typed in switched back on."""
    attrs = list(attrs)
    attrs[0] = attrs[0] | _GUARD_IFLAGS
    attrs[1] = attrs[1] | _GUARD_OFLAGS
    attrs[3] = attrs[3] | _GUARD_LFLAGS
    return attrs


# ── Windows console parity ────────────────────────────────────────────────────────────────────
_ENABLE_PROCESSED_INPUT = 0x0001
_ENABLE_LINE_INPUT = 0x0002
_ENABLE_ECHO_INPUT = 0x0004
_ENABLE_INSERT_MODE = 0x0020
_ENABLE_QUICK_EDIT_MODE = 0x0040
_ENABLE_EXTENDED_FLAGS = 0x0080
_TYPABLE_MODES = (_ENABLE_PROCESSED_INPUT | _ENABLE_LINE_INPUT | _ENABLE_ECHO_INPUT |
                  _ENABLE_INSERT_MODE | _ENABLE_QUICK_EDIT_MODE | _ENABLE_EXTENDED_FLAGS)


def _windows_kernel32():
    import ctypes
    return ctypes.windll.kernel32


def _install_windows():
    """Remember the console input mode, repairing it when a program left it stripped."""
    import ctypes
    from ctypes import wintypes
    kernel32 = _windows_kernel32()
    invalid = ctypes.c_void_p(-1).value
    for std_handle in (-10, -11, -12):                      # STD_INPUT_HANDLE first: the typed keys
        handle = kernel32.GetStdHandle(std_handle)
        if not handle or handle == invalid:
            continue
        mode = wintypes.DWORD()
        if not kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            continue                                        # redirected: nothing to guard
        wanted = mode.value | _TYPABLE_MODES
        if wanted != mode.value and kernel32.SetConsoleMode(handle, wanted):
            _STATE["repaired"] = True
            _STATE["handles"] = [(handle, wanted)]
            return True
        _STATE["handles"] = [(handle, mode.value)]
        return True
    return False


def _restore_windows():
    if not _STATE["handles"]:
        return False
    from ctypes import wintypes
    kernel32 = _windows_kernel32()
    restored = False
    for handle, mode in _STATE["handles"]:
        try:
            restored = bool(kernel32.SetConsoleMode(handle, wintypes.DWORD(mode))) or restored
        except Exception:
            continue
    return restored


def _windows_flag_text():
    if not _STATE["handles"]:
        return "n/a"
    import ctypes
    from ctypes import wintypes
    kernel32 = _windows_kernel32()
    mode = wintypes.DWORD()
    if not kernel32.GetConsoleMode(_STATE["handles"][0][0], ctypes.byref(mode)):
        return "n/a"
    names = ((_ENABLE_PROCESSED_INPUT, "PROCESSED_INPUT"), (_ENABLE_LINE_INPUT, "LINE_INPUT"),
             (_ENABLE_ECHO_INPUT, "ECHO_INPUT"), (_ENABLE_INSERT_MODE, "INSERT_MODE"),
             (_ENABLE_QUICK_EDIT_MODE, "QUICK_EDIT"), (_ENABLE_EXTENDED_FLAGS, "EXTENDED_FLAGS"))
    return " ".join(name for bit, name in names if mode.value & bit)


# ── The two operations ────────────────────────────────────────────────────────────────────────
def restore_console():
    """Write the remembered settings back. Safe to call from every exit path, as often as needed."""
    if not _STATE["enabled"]:
        return False
    if _WINDOWS:
        return _restore_windows()
    fd, saved = _STATE.get("fd"), _STATE.get("saved")
    if fd is None or saved is None:
        return False
    if _STATE["report"]:
        _fd, now = _read_flags()
        if now is not None and list(now) != list(saved):
            print(f"Console guard: the terminal was left in another state ({_flag_text(now)}); "
                  "it has been restored.")
    try:
        termios.tcsetattr(fd, termios.TCSAFLUSH, saved)
        termios.tcflush(fd, termios.TCIFLUSH)               # drop the keys typed blind while it was broken
    except (OSError, termios.error, ValueError):
        return False
    return True


def console_guard_state():
    """One readable line describing the terminal right now, for the log and for tests."""
    if _WINDOWS:
        return _windows_flag_text()
    _fd, attrs = _read_flags()
    return _flag_text(attrs)


def console_guard_enabled():
    return bool(_STATE["enabled"])


# ── The interrupt, chained: restore first, then do what was going to happen ───────────────────
def _chain_interrupt(signum, frame, previous):
    restore_console()
    if previous is signal.SIG_DFL:
        raise KeyboardInterrupt                             # what the default handler means to Python
    if previous is signal.SIG_IGN:
        return
    if callable(previous):
        previous(signum, frame)
        return
    raise KeyboardInterrupt


def _arm_signals():
    """Chain onto the handlers in force; never replace them, and never fail when signals are off."""
    for name in ("SIGINT", "SIGTERM"):
        signum = getattr(signal, name, None)
        if signum is None:
            continue
        previous = signal.getsignal(signum)
        if callable(previous) and getattr(previous, "__wgp_console_guard__", False):
            continue                                        # the guard is already chained
        def handler(signum, frame, previous=previous):
            _chain_interrupt(signum, frame, previous)
        handler.__wgp_console_guard__ = True
        try:
            signal.signal(signum, handler)
        except (ValueError, OSError, RuntimeError, AttributeError):
            continue                                        # not the main thread, or no signal handling
        _STATE["handlers"].append(handler)

# ── Command line and environment ──────────────────────────────────────────────────────────────
_ON_TRUE = ("on", "yes", "true", "1")
_ON_FALSE = ("off", "no", "false", "0")


def _on_off(raw):
    value = str(raw).strip().lower()
    if value in _ON_TRUE:
        return True
    if value in _ON_FALSE:
        return False
    return None


def _argv_on_off(argv, name):
    """True / False when the command line sets the option, None when it is left out."""
    argv = [str(one) for one in (argv or [])]
    prefix = f"{name}="
    for index, one in enumerate(argv):
        text = one.strip()
        if text.startswith(prefix):
            return _on_off(text[len(prefix):])
        if text == name:
            following = argv[index + 1].strip() if index + 1 < len(argv) else ""
            value = _on_off(following)
            return True if value is None else value                     # a flag on its own is an on
    return None


def _uses_stdio_protocol(argv):
    """MCP stdio: WanGP's stdout carries JSON-RPC frames, so the guard keeps out of it."""
    argv = [str(one) for one in (argv or [])]
    if not any(one == "--mcp" for one in argv):
        return False
    for index, one in enumerate(argv):
        text = str(one)
        if text.startswith("--mcp-transport="):
            return text[len("--mcp-transport="):].strip().lower() == "stdio"
        if one == "--mcp-transport" and index + 1 < len(argv):
            return str(argv[index + 1]).strip().lower() == "stdio"
    return True


def install_console_guard(argv=None, *, report=None):
    """Remember the terminal, repair it when it arrives broken, and put it back on the way out.

    On by default -- a healthy run never notices it -- with `--console-guard=off` /
    `WAN2GP_CONSOLE_GUARD=0` turning it off. Returns True when the terminal is guarded: there is a
    terminal to guard, and the option is on. A pipe, a file, the MCP stdio server: nothing is touched.
    """
    if _STATE["installed"]:
        return console_guard_enabled()
    _STATE["installed"] = True
    argv = list(sys.argv[1:] if argv is None else argv)
    setting = _on_off(os.environ.get("WAN2GP_CONSOLE_GUARD", ""))
    flag = _argv_on_off(argv, "--console-guard")
    if flag is not None:
        setting = flag
    if _uses_stdio_protocol(argv):
        return False
    env_report = _on_off(os.environ.get("WAN2GP_CONSOLE_GUARD_LOG", ""))
    _STATE["report"] = bool(report if report is not None else (True if env_report is None else env_report))
    if setting is False:
        _STATE["installed"] = False
        return False
    try:
        if _POSIX:
            fd, attrs = _read_flags()
            if fd is None or attrs is None:
                _STATE["installed"] = False
                return False                                # a pipe or a file: nothing to guard
            wanted = _typable(attrs)
            if wanted != attrs:
                termios.tcsetattr(fd, termios.TCSANOW, wanted)
                _STATE["repaired"] = True
            _STATE["fd"], _STATE["saved"] = fd, wanted
        elif _WINDOWS:
            if not _install_windows():
                _STATE["installed"] = False
                return False
        else:
            _STATE["installed"] = False
            return False
    except Exception:
        _STATE["installed"] = False
        _STATE["fd"], _STATE["saved"], _STATE["handles"] = None, None, []
        return False
    _STATE["enabled"] = True
    atexit.register(restore_console)
    _arm_signals()
    if setting is None and _on_off(os.environ.get("WAN2GP_CONSOLE_GUARD_LOG", "")):
        print(f"Console guard: terminal settings at start: {console_guard_state()}")
    if _STATE["repaired"]:
        print("Console guard: the terminal was left in a state you cannot type in; it has been repaired. "
              "--console-guard=off turns the guard off.")
    return True


def configure_console_guard(args, argv=None):
    """Apply the parsed command line: `--console-guard=off` puts the guard out of the way."""
    value = getattr(args, "console_guard", None) if args is not None else None
    argv = list(sys.argv[1:] if argv is None else argv)
    if value is None:
        value = _argv_on_off(argv, "--console-guard")
    if value is False:
        return disable_console_guard()
    if value is True and not _STATE["enabled"]:
        return install_console_guard(argv)
    return console_guard_enabled()


def disable_console_guard():
    """Stop guarding: the terminal is left exactly as it is, and no handler of ours is left running."""
    _STATE["enabled"] = False
    try:
        atexit.unregister(restore_console)
    except Exception:
        pass
    return False

    return bool(_STATE["handlers"])
