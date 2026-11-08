# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Launch configurations of WanGP Triton kernels, tuned once per kernel, tuning identity and GPU.

A call that cannot be tuned (too small to time, during CUDA graph recording or PyTorch compilation) uses the static configuration, the
first candidate, without caching it. The first call that can be tuned times every candidate on scratch work prepared by the caller (never
in place on live tensors), skipping those that need more shared memory than the GPU has, and keeps the winner in a versioned cache file
shared by the sessions, so later sessions compile only it."""
import functools
import json
import time
from pathlib import Path

import torch


@functools.cache
def _device_name(device):
    return torch.cuda.get_device_name(device)


class LaunchTuner:
    def __init__(self, cache_name, version, label):
        """cache_name: file under ~/.triton/autotune; version: bumped when a kernel, its candidates or the benchmark change."""
        self.path = Path.home() / ".triton" / "autotune" / f"{cache_name}.json"
        self.version, self.label = version, label
        self._tuned = None

    def _cache(self):
        if self._tuned is None:
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                self._tuned = payload["configs"] if payload.get("version") == self.version else {}
            except (OSError, ValueError, KeyError):
                self._tuned = {}
        return self._tuned

    def config(self, name, identity, device, candidates, prepare_scratch, tunable):
        """Launch configuration of kernel `name` for `identity` (what changes its best configuration, such as a model width): the cached
        winner, else tuned now when `tunable`. prepare_scratch() returns launch(config) running the kernel on scratch work."""
        key = f"{name}|{identity}|{_device_name(device)}"
        config = self._cache().get(key)
        if config is not None:
            return tuple(config)
        if not tunable or torch.cuda.is_current_stream_capturing() or torch.compiler.is_compiling():
            return candidates[0]
        from triton.runtime.errors import OutOfResources
        started, timings = time.time(), []
        launch = prepare_scratch()
        for config in candidates:
            try:
                launch(config)  # compilation: Triton refuses a configuration needing more shared memory than the GPU has (64 KiB on AMD RDNA)
            except OutOfResources as error:
                refused = error
                continue
            launch(config)
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(10):
                launch(config)
            end.record()
            end.synchronize()
            timings.append((start.elapsed_time(end) / 10, config))
        if not timings:
            raise refused
        best_ms, best = min(timings)
        self._tuned[key] = list(best)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"version": self.version, "configs": self._tuned}, indent=1), encoding="utf-8")
        except OSError as error:
            print(f"[{self.label}] Could not save the Triton tuning cache {self.path}: {error}")
        return best


from shared.kernels.triton_compilation_log import install_triton_compilation_logger, log_compilation_tuning
install_triton_compilation_logger()
LaunchTuner.config = log_compilation_tuning(LaunchTuner.config)
