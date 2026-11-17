# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from contextvars import ContextVar
from contextlib import contextmanager

_abort = ContextVar('background_removal_abort', default=None)


def check_abort():
    callback = _abort.get()
    if callback is not None:
        callback()


@contextmanager
def abort_scope(callback):
    token = _abort.set(callback)
    try:
        yield
    finally:
        _abort.reset(token)
