# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Saving the videos of a generation while it goes on: the next window, sample or queued task is generated while the previous output is
encoded, joined, muxed with its sound and recorded with its metadata and side files.

Jobs run one after the other on a thread of their own, in the order they were given. A job that writes an output waits until no other
output is waiting or being written: the frames of one output at most are held in RAM beside the generation's own. Other jobs (removing a
generation's working files once its outputs are written, starting a new sample) only queue."""
from __future__ import annotations

import contextvars
import os
import threading
import traceback
from collections import deque
from functools import partial

from tqdm import tqdm


def _normalized(path):
    return os.path.normcase(os.path.abspath(path))


class BackgroundSaver:
    def __init__(self):
        self._condition = threading.Condition()
        self._jobs, self._outputs, self._thread = deque(), 0, None
        self._owner, self._error, self._failed_owner = 0, None, None
        self._reserved = set()

    def _work(self):
        while True:
            with self._condition:
                if not self._jobs:
                    self._thread = None
                    self._condition.notify_all()
                    return
                job, path, output, owner = self._jobs[0]
                skipped = output and owner == self._failed_owner  # the outputs after a failed one would follow a video that is not there
            try:
                if not skipped:
                    job()
            except BaseException as error:
                tqdm.write(f"[Background Save] {'Saving ' + os.path.basename(path) if path is not None else 'A save job'} failed:\n{traceback.format_exc()}")
                with self._condition:
                    self._failed_owner = owner
                    if owner == self._owner:
                        self._error = error
            finally:
                with self._condition:
                    self._jobs.popleft()
                    self._outputs -= output
                    if path is not None:
                        self._reserved.discard(path)
                    self._condition.notify_all()

    def _raise(self):
        error, self._error = self._error, None
        if error is not None:
            raise error

    def reserve(self, path):
        """path, or when an output still being written has that name, the first free one after it numbered as get_available_filename does
        ("name(2).mp4")."""
        name, extension = os.path.splitext(path)
        candidate, counter = path, 2
        with self._condition:
            while candidate in self._reserved or candidate != path and os.path.exists(candidate):
                candidate, counter = f"{name}({counter}){extension}", counter + 1
        return candidate

    def writes_any(self, values):
        """Whether one of values (paths, or lists of paths) is an output still being written."""
        with self._condition:
            if not self._reserved:
                return False
            reserved = {_normalized(path) for path in self._reserved}
        paths = [item for value in values for item in (value if isinstance(value, (list, tuple)) else [value]) if isinstance(item, str) and item]
        return any(_normalized(path) in reserved for path in paths)

    def submit(self, job, path=None, output=True):
        """Run job() after the jobs given before it. path is the output it writes; output: it holds an output's frames, and waits until no
        other output is waiting or being written. job runs in the caller's context (its workspace and gallery)."""
        with self._condition:
            while output and self._outputs > 0:
                self._condition.wait()
            self._raise()
            self._jobs.append((partial(contextvars.copy_context().run, job), path, output, self._owner))
            self._outputs += output
            if path is not None:
                self._reserved.add(path)
            if self._thread is None:
                self._thread = threading.Thread(target=self._work, name="WanGP Background Save")
                self._thread.start()

    def detach(self):
        """The generation that gave the jobs queued so far ends without waiting for them: their errors are only reported in the console,
        and the next generation starts with a clean state."""
        with self._condition:
            self._owner += 1
            self._error = None

    def pending(self):
        with self._condition:
            return bool(self._jobs)

    def wait(self, raise_error=True):
        """Wait for every job given so far; the error of one of the current generation is raised here (only reported when raise_error is False)."""
        with self._condition:
            while self._jobs:
                self._condition.wait()
            self._failed_owner = None  # the outputs given from now on follow no failed one
            if raise_error:
                self._raise()
            self._error = None
