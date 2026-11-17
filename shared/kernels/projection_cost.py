# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""What a group of heads adds to an attention split by groups (shared.attention_kit), probed once per GPU and weight format.

Each group projects its q, k and v with GEMMs of its own over the whole prepared input of the projections: every launch reads that input
again, from VRAM once it is larger than the L2 cache, so a narrow launch costs more than its share of a wide one, and the more the larger
the input. Each group also runs a few small kernels of its own (quantizations, norm, the attention launch, the output's columns).

The probe times, on the call's own prepared input and projection (no other input in VRAM), NARROW launches of one head's columns
against one launch of as many columns, at two row counts: the extra time of one more launch as a + b x (rows x in_features), a line
through both points (GEMMs of tens of microseconds and more: eager launches with events time them, a block's recording warmup
included). And the time of a small kernel inside a CUDA graph, measured outside MMGP's recording (KERNEL_MS until then). Kept per GPU,
weight format and input width under ~/.triton/autotune; nothing is measured while a CUDA graph is captured."""
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

from mmgp import cuda_graphs

VERSION = 1  # bumped when the benchmark or the model change
PROBE_ROWS = 32768  # rows of the larger point (beyond the L2 cache at usual widths), a quarter of them for the other
NARROW = 8  # launches of one head's columns timed against one of NARROW heads
KERNELS_PER_GROUP = 8  # small kernels a group runs beside its GEMMs and its attention (quantizations, norm and RoPE, gate, columns)
KERNEL_MS = 0.003  # a small kernel inside a CUDA graph until measured (RTX 5090: about 3 us)
_PATH = Path.home() / ".triton" / "autotune" / "wan2gp_projection_cost.json"
_saved = None
_costs = {}
_kernels = {}  # device name -> measured small kernel (ms)


class ProjectionCost:
    """launch_ms(elements): the extra time of one more GEMM launch over a prepared input of `elements` (rows x in_features); kernel_ms:
    one small kernel inside a CUDA graph."""
    def __init__(self, a, b, kernel_ms, elements=None):
        self.a, self.b, self.kernel_ms, self.elements = a, b, kernel_ms, elements  # elements: the largest input probed when below PROBE_ROWS

    def launch_ms(self, elements):
        return self.a + self.b * elements


def _load():
    global _saved
    if _saved is None:
        try:
            payload = json.loads(_PATH.read_text(encoding="utf-8"))
            _saved = payload["costs"] if payload.get("version") == VERSION else {}
        except (OSError, ValueError, KeyError):
            _saved = {}
    return _saved


def _save():
    try:
        _PATH.parent.mkdir(parents=True, exist_ok=True)
        _PATH.write_text(json.dumps({"version": VERSION, "costs": _saved}, indent=1), encoding="utf-8")
    except OSError as error:
        tqdm.write(f"[Kernels] Could not save the projection probe {_PATH}: {error}")


def _time(launch):
    launch()
    best = float("inf")
    for _ in range(4):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        launch()
        end.record()
        end.synchronize()
        best = min(best, start.elapsed_time(end))
    return best


def _kernel_ms(device, name):
    """One small kernel inside a CUDA graph (64 of them replayed), measured once per device outside MMGP's recording (a graph of its own
    is captured), else KERNEL_MS."""
    if name in _kernels:
        return _kernels[name]
    saved = _load().get(f"kernel|{name}")
    if saved is not None:
        _kernels[name] = saved
        return saved
    if cuda_graphs.recording() or cuda_graphs.capturing() or torch.cuda.is_current_stream_capturing():
        return KERNEL_MS
    x = torch.zeros(4096, device=device)
    graph, stream = torch.cuda.CUDAGraph(), torch.cuda.Stream(device)
    with torch.cuda.stream(stream):
        x.add_(1)
    torch.cuda.synchronize(device)
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(64):
            x.add_(1)
    ms = _time(graph.replay) / 64
    del graph
    _kernels[name] = _saved[f"kernel|{name}"] = ms
    _save()
    return ms


def get(identity, device, rows_of, width):
    """The ProjectionCost of weight format `identity` (its row projection's key and the input width) on this device: from this session,
    the cache file, or probed now. rows_of(count) returns project(columns, first) -> None over the first `count` rows of the prepared
    input (at most its rows; None when it cannot give them, or the module has fewer than NARROW x width outputs), with .rows and
    .in_features. None during a CUDA graph capture or compilation, or when nothing could be measured."""
    index = torch.cuda.current_device() if device.index is None else device.index
    name = torch.cuda.get_device_name(index)
    key = f"{identity}|{name}"
    cost = _costs.get(key)
    if cost is not None and cost.elements is not None:  # probed on a shorter input than PROBE_ROWS: again on a longer one
        longer = rows_of(PROBE_ROWS)
        if longer is not None and longer.rows * longer.in_features > cost.elements:
            del _costs[key]
        del longer
    if key in _costs:
        cost = _costs[key]
        if cost is not None:
            cost.kernel_ms = _kernel_ms(device, name)
        return cost
    saved = _load().get(key)
    if saved is not None:
        cost = _costs[key] = ProjectionCost(saved["a"], saved["b"], _kernel_ms(device, name))
        return cost
    if torch.cuda.is_current_stream_capturing() or torch.compiler.is_compiling():
        return None
    started, points = time.time(), []
    for count in (PROBE_ROWS // 4, PROBE_ROWS):
        project = rows_of(count)
        if project is None or points and project.rows * project.in_features <= points[-1][0]:  # an input shorter than both points: one
            continue
        narrow = _time(lambda: [project(width, first * width) for first in range(NARROW)])
        wide = _time(lambda: project(NARROW * width, 0))
        points.append((project.rows * project.in_features, max(0.0, (narrow - wide) / (NARROW - 1))))
        project_width = project.in_features
        del project
    if not points:
        _costs[key] = None
        return None
    if len(points) == 2:
        b = max(0.0, (points[1][1] - points[0][1]) / (points[1][0] - points[0][0]))
        a = max(0.0, points[1][1] - b * points[1][0])
    else:
        a, b = 0.0, points[0][1] / points[0][0]
    complete = points[-1][0] >= PROBE_ROWS * project_width
    cost = _costs[key] = ProjectionCost(a, b, _kernel_ms(device, name), None if complete else points[-1][0])
    tqdm.write(f"[Kernels] Head groups' projections probed ({identity}): {cost.launch_ms(points[-1][0]):.3f} ms per launch at {points[-1][0] / 1e6:.0f} M "
               f"input elements ({time.time() - started:.1f} s).")
    if complete:  # kept for the next sessions only when probed at its full size
        _saved[key] = {"a": a, "b": b}
        _save()
    return cost
