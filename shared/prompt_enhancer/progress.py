# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Progress for enhancer phases, shared by manual and automatic enhancement."""

from contextlib import contextmanager
from time import monotonic

from tqdm.auto import tqdm


class EnhancementProgress:
    def __init__(self, callback=None, stream_callback=None, prompt_offset=0, prompt_total=None, failures=None):
        self.callback = callback
        self.stream_callback = stream_callback
        self.prompt_offset = prompt_offset
        self.prompt_total = prompt_total
        self.title = "Enhancing Prompt"
        self.next_update = 0
        self.failures = [] if failures is None else failures  # why enhanced prompts came back without a prompt, for the user's notice (shared by for_prompt)

    def for_prompt(self, index, total):
        return EnhancementProgress(self.callback, self.stream_callback, index, total, self.failures)

    def start(self, title, index=0, count=1, *, steps=0, unit="tokens"):
        self.title = title + (f" {index + 1}/{count}" if count > 1 else "")
        self.update(0, steps, unit, force=True)

    def prompt(self, index, count, max_tokens):
        self.start("Enhancing Prompt", self.prompt_offset + index, self.prompt_total or count, steps=max_tokens)

    def caption(self, index, count, max_tokens):
        self.start("Captioning Images", self.prompt_offset + index, self.prompt_total or count, steps=max_tokens)

    def update(self, completed, total, unit, *, force=False):
        now = monotonic()
        if force or now >= self.next_update:
            self.next_update = now + 1 / 3
            if self.callback is not None:
                self.callback((completed, total), desc=self.title, total=total, unit=unit)

    def tokens(self, **event):
        if self.stream_callback is not None:
            self.stream_callback(description=self.title, **event)
        else:
            self.update(event["token_count"], event["max_tokens"], "tokens", force=event.get("is_final", False))

    @contextmanager
    def vision(self, layers, index, count):
        self.start("Encoding Image", index, count, steps=len(layers), unit="layers")
        handles = []
        with tqdm(total=len(layers), desc=self.title, unit="layers", mininterval=1 / 3, leave=False) as bar:
            def completed(*_):
                bar.update()
                self.update(bar.n, len(layers), "layers", force=bar.n == len(layers))

            try:
                handles = [layer.register_forward_hook(completed) for layer in layers]
                yield
            finally:
                for handle in handles:
                    handle.remove()
