# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Keep the original launch alive across application restarts, including Manager jobs."""

import os
import signal
import subprocess
import sys
import threading

_CHILD = "_WANGP_LAUNCH_CHILD"
_restart_lock = threading.Lock()
_restart_timer = None


def supervise():
    """Run before application imports. Only the child initializes WanGP.

    The parent retains the original argv, cwd and environment, and waits for
    each child to exit before starting another. Keeping that parent alive also
    keeps Manager's process job, console pipes and installation lease alive.
    """
    launch = os.environ.pop(_CHILD, None)
    if launch is not None:
        return launch == "restart"
    command = [sys.executable, *sys.orig_argv[1:]]
    env = dict(os.environ, **{_CHILD: "initial"})
    while True:
        with subprocess.Popen(command, env=env) as child:
            # Console Ctrl+C reaches both processes. Let WanGP save its queue
            # and shut down; the lightweight parent must keep waiting for it.
            previous = signal.signal(signal.SIGINT, lambda *_: None)
            previous_term = signal.signal(signal.SIGTERM, lambda *_: child.terminate())
            try:
                code = child.wait()
            finally:
                signal.signal(signal.SIGINT, previous)
                signal.signal(signal.SIGTERM, previous_term)
        if code != 42:
            raise SystemExit(code)
        env[_CHILD] = "restart"


def request_restart():
    """Let the UI receive its acknowledgement before the launcher restarts us."""
    global _restart_timer
    with _restart_lock:
        if _restart_timer is None:
            def finish():
                sys.stdout.flush()
                sys.stderr.flush()
                os._exit(42)
            _restart_timer = threading.Timer(2, finish)
            _restart_timer.daemon = True
            _restart_timer.start()
