# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

from typing import Any
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from time import perf_counter


_LOGGER_MARKER = "_wangp_triton_compilation_logger"
_TUNING = ContextVar("wangp_triton_tuning", default=None)


class _CompilationGroup:
    def __init__(self):
        self.started = {}
        self.winner = None

    def start(self, name):
        if name not in self.started:
            self.started[name] = perf_counter()
            print(f"[WanGP][Triton] Preparing {name} (compiling and tuning, please wait)...", flush=True)

    def finish(self, failed=False):
        for name, started in self.started.items():
            elapsed = perf_counter() - started
            if failed:
                print(f"[WanGP][Triton] Compilation/tuning failed for {name} after {elapsed:.2f} s.", flush=True)
            else:
                print(f"[WanGP][Triton] Compiled {name} in {elapsed:.2f} s; winner: {self.winner}.", flush=True)


@contextmanager
def compilation_tuning():
    """Group candidate compilations; a fully cached search remains silent."""
    group = _CompilationGroup()
    token = _TUNING.set(group)
    try:
        yield group
    except BaseException:
        group.finish(failed=True)
        raise
    else:
        group.finish()
    finally:
        _TUNING.reset(token)


def log_compilation_tuning(function):
    """Wrap a custom tuning search which returns its selected configuration."""
    @wraps(function)
    def wrapped(*args, **kwargs):
        with compilation_tuning() as group:
            result = function(*args, **kwargs)
            group.winner = result
            return result
    return wrapped


def _install_autotune_logger(triton):
    autotuner = getattr(getattr(getattr(triton, "runtime", None), "autotuner", None), "Autotuner", None)
    if autotuner is None or not hasattr(autotuner, "run") or getattr(autotuner.run, _LOGGER_MARKER, False):
        return
    original = autotuner.run

    @wraps(original)
    def run(self, *args, **kwargs):
        with compilation_tuning() as group:
            result = original(self, *args, **kwargs)
            group.winner = self.best_config
            return result

    setattr(run, _LOGGER_MARKER, True)
    autotuner.run = run


def _compile_duration_ms(times: Any) -> float:
    lowering_stages = getattr(times, "lowering_stages", ())
    total_us = int(getattr(times, "ir_initialization", 0)) + int(getattr(times, "store_results", 0))
    total_us += sum(int(duration) for _stage, duration in lowering_stages)
    return total_us / 1000.0


def _compile_variant(source: Any) -> str:
    parts = []
    if hasattr(source, "fn"):
        dtypes = dict.fromkeys(value.lstrip("*") for value in source.signature.values() if isinstance(value, str) and value.startswith("*"))
        if dtypes:
            parts.append("/".join(dtypes))
        for index, name in enumerate(source.fn.arg_names):
            if name.upper() in {"B", "T", "D", "H", "W", "C", "M", "N", "K", "SX0", "BM", "BN", "BK", "BT", "BW", "BD"} or name.upper().startswith("BLOCK_"):
                value = source.constants.get((index,), "dynamic")
                if isinstance(value, int) and not isinstance(value, bool) or value == "dynamic":
                    parts.append(f"{name}={value}")
    return " ".join(parts)


def install_triton_compilation_logger() -> bool:
    try:
        import triton
    except Exception:
        return False

    knobs = getattr(triton, "knobs", None)
    compilation = getattr(knobs, "compilation", None)
    compiler = getattr(getattr(triton, "compiler", None), "compiler", None)
    if compilation is None or compiler is None or not hasattr(compilation, "listener") or not hasattr(compiler, "get_cache_manager"):
        return False

    previous_listener = compilation.listener
    _install_autotune_logger(triton)
    if not getattr(previous_listener, _LOGGER_MARKER, False):
        def listener(**event):
            if previous_listener is not None:
                previous_listener(**event)
            source = event.get("src")
            kernel_name = str(getattr(source, "name", "") or type(source).__name__)
            if event.get("cache_hit", True):
                return
            group = _TUNING.get()
            if group is not None:
                group.start(kernel_name)
                return
            duration_ms = _compile_duration_ms(event.get("times"))
            variant = _compile_variant(source)
            details = f" [{variant}]" if variant else ""
            metadata = event.get("metadata") or {}
            launch = " ".join(f"{key}={metadata[key]}" for key in ("num_warps", "num_stages") if key in metadata)
            selected = f"; configuration: {launch}" if launch else ""
            print(f"[WanGP][Triton] Compiled {kernel_name}{details} in {duration_ms:.0f} ms{selected}.", flush=True)

        setattr(listener, _LOGGER_MARKER, True)
        compilation.listener = listener

    previous_get_cache_manager = compiler.get_cache_manager
    if not getattr(previous_get_cache_manager, _LOGGER_MARKER, False):
        def get_cache_manager(key):
            cache_manager = previous_get_cache_manager(key)

            class CompilationCacheManager:
                def __getattr__(self, name):
                    return getattr(cache_manager, name)

                def get_group(self, filename):
                    group = cache_manager.get_group(filename)
                    if filename.endswith(".json") and (compilation.always_compile or not group or group.get(filename) is None):
                        tuning = _TUNING.get()
                        if tuning is not None:
                            tuning.start(filename[:-5])
                        else:
                            print(f"[WanGP][Triton] Preparing {filename[:-5]} (compiling, please wait)...", flush=True)
                    return group

            return CompilationCacheManager()

        setattr(get_cache_manager, _LOGGER_MARKER, True)
        compiler.get_cache_manager = get_cache_manager
    return True
