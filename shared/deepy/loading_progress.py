# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Forward Deepy's first-use downloads and model loading to its chat status."""

import time
from tqdm import tqdm


class DeepyLoadingProgress:
    def __init__(self, publish, cancelled):
        self.publish = publish
        self.gen = {"download_progress_callback": self.download, "abort_callback": cancelled}
        self.last_update = self.last_console = 0.0
        self.phase = self.filename = None
        self.console_text = None

    def _status(self, text, phase, finished=False):
        now = time.monotonic()
        if phase != self.phase or finished or now - self.last_update >= 1 / 3:
            self.publish(text)
            self.last_update, self.phase = now, phase

    def __call__(self, value, desc="Loading Deepy", **_kwargs):
        text = desc
        finished = False
        if isinstance(value, tuple):
            completed, total = value
            text += f" - {completed:,}/{total:,}"
            finished = completed == total
        self._status(text, desc, finished)

    def scope(self, show_status, on_phase):
        """MMGP's loading context around the preparation of a model call: each step after the load (weights moved to the GPU, graph captures,
        experts placed, the instructions read) goes to on_phase(phase), and to the status, as the load's do, when show_status (a first call).
        Without cancellation: these steps run inside the model's calls."""
        from mmgp import offload

        def report(phase, completed, total, _model_id):
            on_phase(phase)
            if not show_status:
                return
            text = f"Loading - {phase} Deepy"
            self._status(f"{text} - {completed:,}/{total:,}" if total else f"{text} ({completed:,} done)" if completed else text, text, total > 0 and completed == total)

        return offload.loading_context(offload.LoadingCallback(None, report))

    def download(self, report):
        if report is None:
            self.filename = None
            self._status("Loading Deepy", "loading")
            return
        filename = report.get("filename") or "Model Files"
        completed, total = report.get("completed", 0), report.get("total")
        text = f"Downloading Deepy - {filename}"
        if total:
            text += f" - {completed / total:.0%} ({completed / 1024**2:,.0f}/{total / 1024**2:,.0f} MiB)"
        finished = total is not None and completed == total
        self._status(text, ("download", filename), finished)
        now = time.monotonic()
        if text != self.console_text and (filename != self.filename or now - self.last_console >= 5 or finished):
            tqdm.write(text)
            self.last_console = now
            self.console_text = text
        self.filename = filename
