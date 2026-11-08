# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Choose the GGUF short-batch linear kernel per shape before CUDA graph capture.

Speculative verification runs Q4_K and PTQ1_0 linears on 2-8 activation rows.
The native package offers llama.cpp MMVQ and an INT8 tensor-core path; both
quantize activations to 8 bits and form exact integer dot products. Compute
capability 12.0 defaults to tensor cores, where they were validated; other GPUs
(compute capability 8.0+) default to MMVQ. Before graph capture, both kernels
are timed on the model's own weights of each shape in rotation (cache-cold),
and the default is kept unless the other kernel is at least 5% faster. Choices
are cached on disk per GPU, driver and package.
"""
import json
import time
from pathlib import Path

import torch
from mmgp import runtime

QTYPES = ("Q4_K", "PTQ1_0")
ROWS = range(2, 9)
METHOD_VERSION = 4
MARGIN = 0.95
MAX_WEIGHTS = 64
# Tensor cores versus single-row MMVQ (FP32 output) differ by ~1e-3, as MMVQ keeps
# activation block scales in FP16; a broken kernel differs by O(1).
AGREEMENT = 1e-2
_choices = {}
_disk = None


def _cache_path():
    return Path.home() / ".triton" / "autotune" / "wan2gp_gguf_short_batch.json"


def _cold_bytes(device):
    """Rotation size that keeps timed weights out of L2, as in a real verification pass."""
    return 4 * torch.cuda.get_device_properties(device).L2_cache_size


def _driver_version():
    try:
        import pynvml
        pynvml.nvmlInit()
        return str(pynvml.nvmlSystemGetDriverVersion())
    except Exception:
        return ""


def _device_key(device, native):
    props = torch.cuda.get_device_properties(device)
    return "|".join((props.name, str(getattr(props, "uuid", "")), f"{props.major}.{props.minor}", str(props.multi_processor_count),
                     _driver_version(), native.__version__, str(torch.version.cuda), f"method{METHOD_VERSION}"))


def _load_disk():
    global _disk
    if _disk is None:
        try:
            _disk = json.loads(_cache_path().read_text(encoding="utf-8"))["entries"]
        except Exception:
            _disk = {}
    return _disk


def _save_disk():
    path = _cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"version": METHOD_VERSION, "entries": _load_disk()}, indent=1, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
    except OSError as exc:
        print(f"[GGUF] Short-Batch choice cache not saved: {exc}")


def _measure(native, qtype, shape, peers, x, dtype, warm_seconds=0., pool=None):
    """Per-call time (ms) of both kernels over the weights in rotation, or None if tensor cores disagree
    with single-row MMVQ (the decode kernel; batched native may be MMQ, which is itself less accurate)."""
    times, graphs = {"native": [], "mma": []}, {}
    try:
        native.set_short_batch_mode("native")
        reference = torch.cat([native.linear(peers[0], qtype, shape, row, None, torch.float32) for row in x.split(1)])
        native.set_short_batch_mode("mma")
        batched = native.linear(peers[0], qtype, shape, x, None, torch.float32)
        error = float((batched - reference).norm() / reference.norm().clamp_min(1e-30))
        if not error < AGREEMENT:
            return None, error
        for mode in times:
            native.set_short_batch_mode(mode)
            native.linear(peers[0], qtype, shape, x, None, dtype)  # load the timed variant before capture
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, capture_error_mode="thread_local"):
                for raw in peers:
                    native.linear(raw, qtype, shape, x, None, dtype)
            graphs[mode] = graph
        native.set_short_batch_mode("auto")
        # Untimed, memory-bound replays: an idle GPU raises its memory clock and Windows makes
        # fresh allocations resident on first touch.
        warm_start = time.perf_counter()
        while True:
            for graph in graphs.values():
                graph.replay()
            torch.cuda.synchronize()
            if time.perf_counter() - warm_start >= warm_seconds:
                break
        for round_index in range(9):
            for mode in (("native", "mma") if round_index % 2 == 0 else ("mma", "native")):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                graphs[mode].replay()
                end.record()
                end.synchronize()
                if round_index:
                    times[mode].append(start.elapsed_time(end) / len(peers))
        # Other GPU clients (desktop compositor, browsers) can take time slices mid-measurement;
        # interference only adds time, so the fastest alternating round is the kernel's cost.
        return {mode: min(values) for mode, values in times.items()}, error
    finally:
        native.set_short_batch_mode("auto")
        for graph in graphs.values():
            graph.reset()
        graphs.clear()


def _gguf_weights(model):
    """Resident Q4_K/PTQ1_0 weights grouped by (device, qtype, shape), de-duplicated by storage: (raw, dtype, mapped), mapped for a weight that
    the GPU reads in place from RAM (MMGP map_in_ram), counted as resident and timed from a temporary VRAM copy."""
    groups = {}
    for module in model.modules():
        for tensor in module._parameters.values():
            tensor_type, raw = getattr(tensor, "_tensor_type", None), getattr(tensor, "_data", None)
            if tensor_type is None or tensor_type.name not in QTYPES or not torch.is_tensor(raw) or not raw.is_cuda or not raw.is_contiguous():
                continue
            shape = tuple(int(size) for size in tensor._tensor_shape)
            if shape[1] % 256 == 0:
                groups.setdefault((raw.device, tensor_type.name, shape), {}).setdefault(raw.data_ptr(), (raw, tensor._gguf_default_dtype, runtime.is_mapped(tensor)))
    return groups


def prepare(model, native):
    """Record short-batch kernel choices for the model's Q4_K/PTQ1_0 weights. Call before CUDA graph capture."""
    if native is None or not getattr(native, "has_short_batch_policy", lambda: False)() or native.short_batch_mode() != "auto":
        return
    measured, lines, started = 0, [], time.perf_counter()
    pools = {}  # one graph pool per device for every timing capture: each new pool reserves GPU address space (as large as the card with MMGP's allocator)
    for (device, qtype, shape), weights in _gguf_weights(model).items():
        props = torch.cuda.get_device_properties(device)
        if props.major < 8:
            return
        if all((device, qtype, rows, shape) in _choices for rows in ROWS):
            continue
        default_mma = (props.major, props.minor) == (12, 0)
        device_key = _device_key(device, native)
        entries = _load_disk().setdefault(device_key, {})
        peers, total, dtype = [], 0, next(iter(weights.values()))[1]
        for raw, _, mapped in weights.values():
            if len(peers) == MAX_WEIGHTS or total >= _cold_bytes(device):
                break
            peers.append((raw, mapped))
            total += raw.numel()
        if total < _cold_bytes(device):
            # Too few bytes of this shape to time cache-cold: keep the architecture default (not cached,
            # another model may hold enough weights of this shape).
            for rows in ROWS:
                _choices[(device, qtype, rows, shape)] = default_mma
            continue
        speedups, enabled, timed = [], [], None
        for rows in ROWS:
            shape_key = f"{qtype}|{rows}|{shape[0]}|{shape[1]}"
            record = entries.get(shape_key)
            if record is None:
                if timed is None:  # the weights read from RAM are timed from VRAM copies, as when they were resident
                    timed = [raw.clone() if mapped else raw for raw, mapped in peers]
                if device not in pools:
                    pools[device] = torch.cuda.graph_pool_handle()
                with torch.inference_mode(), torch.cuda.device(device):
                    x = torch.randn((rows, shape[1]), device=device, dtype=dtype)
                    timings, error = _measure(native, qtype, shape, timed, x, dtype, warm_seconds=0. if measured else .05, pool=pools[device])
                measured += 1
                if timings is None:
                    record = {"mma": False, "error": error}
                    print(f"[GGUF] Short-Batch {qtype} {shape[1]}->{shape[0]}, {rows} Rows: Tensor Cores Disagree With MMVQ (Relative Error {error:.1e}) -> Keeping MMVQ.")
                else:
                    speedup = timings["native"] / timings["mma"]
                    record = {"mma": speedup >= 1 / MARGIN if not default_mma else speedup > MARGIN, "speedup": speedup, "error": error, "weights": len(peers)}
                    speedups.append(f"{speedup:.2f}")
                entries[shape_key] = record
            native.set_short_batch_decision(qtype, rows, shape[0], shape[1], record["mma"])
            _choices[(device, qtype, rows, shape)] = record["mma"]
            if record["mma"]:
                enabled.append(rows)
        if speedups:
            lines.append(f"[GGUF] Short-Batch {qtype} {shape[1]}->{shape[0]} ({len(peers)} Weights): Tensor Cores vs MMVQ {'/'.join(speedups)}x "
                         f"for Rows 2-8 -> Tensor Cores for Rows {','.join(map(str, enabled)) or 'None'}.")
    if measured:
        _save_disk()
        print(f"[GGUF] Short-Batch Kernel Selection on {torch.cuda.get_device_name()}: {measured} Shape/Row Combinations Measured in {time.perf_counter() - started:.1f}s (Cached for Later Loads).")
        for line in lines:
            print(line)
