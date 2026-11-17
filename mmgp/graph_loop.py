# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Loops inside CUDA graphs: a conditional WHILE node whose body is recorded once and run by the GPU itself a given number of times, so
that a loop over chunks of rows adds a few nodes to a graph instead of the nodes of every chunk (the driver keeps a few KB of VRAM per
node of an instantiated graph, and a long graph takes longer to capture and instantiate).

    with graph_loop.loop(iterations, counter):
        ...kernels of one turn...

runs during a capture: the counter (one int64 on the device) is zeroed, the body is recorded on a stream of its own into the WHILE node's
graph, and a step kernel ends each turn (counter += 1, another turn while counter < iterations). Kernels of the body address their turn's
data through the counter (row offsets read on the device); their other pointers are fixed. The body runs once at least: iterations >= 1.
The body allocates nothing: allocators do not see its capture, so a tensor allocated there could be reused while the graph still uses it.
Requires an NVIDIA driver of CUDA 12.3 or later (available()); the driver API is reached through ctypes and the step kernel is a
fatbin for sm_75 to sm_120 (graph_loop_step.cu), so nothing is compiled."""
import contextlib
import ctypes
import json
import os
import time
from pathlib import Path

import torch
from tqdm import tqdm

_CONDITIONAL, _WHILE, _ASSIGN_DEFAULT, _SET_DEPENDENCIES, _RELAXED, _CAPTURING = 13, 1, 1, 1, 2, 1
_state = None  # None: not loaded yet, False: unavailable, else the driver library
_modules = {}  # CUcontext -> the step kernel's CUfunction in that context
_streams = {}  # device index -> the stream bodies are recorded on
_available = {}  # device index -> whether loops can be recorded there
stats = {"loops": 0, "body_nodes": 0}  # loops recorded and the nodes of their bodies (a graph's node count leaves them out)


class _ConditionalParams(ctypes.Structure):
    _fields_ = [("handle", ctypes.c_uint64), ("type", ctypes.c_int), ("size", ctypes.c_uint), ("graph_out", ctypes.POINTER(ctypes.c_void_p)),
                ("ctx", ctypes.c_void_p)]


class _NodeParams(ctypes.Structure):  # CUgraphNodeParams: type, reserved, a union of 29 long longs, reserved
    _fields_ = [("type", ctypes.c_int), ("reserved0", ctypes.c_int * 3), ("union", ctypes.c_longlong * 29), ("reserved2", ctypes.c_longlong)]


def _check(code, call):
    if code:
        raise RuntimeError(f"[mmgp] CUDA graph loop: {call} failed (CUDA driver error {code})")


def _driver():
    global _state
    if _state is None:
        _state = False
        if torch.version.cuda is None or not torch.cuda.is_available():
            return False
        try:
            lib = ctypes.WinDLL("nvcuda.dll") if os.name == "nt" else ctypes.CDLL("libcuda.so.1")
        except OSError:
            return False
        version = ctypes.c_int()
        if lib.cuDriverGetVersion(ctypes.byref(version)) or version.value < 12030:
            return False
        p, u64, size, u = ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_uint
        signatures = {"cuStreamGetCaptureInfo_v3": [p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(u64), ctypes.POINTER(p), ctypes.POINTER(p), ctypes.POINTER(p), ctypes.POINTER(size)],
                      "cuGraphConditionalHandleCreate": [ctypes.POINTER(u64), p, p, u, u],
                      "cuGraphAddNode_v2": [ctypes.POINTER(p), p, p, p, size, ctypes.POINTER(_NodeParams)],
                      "cuStreamUpdateCaptureDependencies_v2": [p, ctypes.POINTER(p), p, size, u],
                      "cuStreamBeginCaptureToGraph": [p, p, p, p, size, ctypes.c_int], "cuStreamEndCapture": [p, ctypes.POINTER(p)],
                      "cuGraphGetNodes": [p, p, ctypes.POINTER(size)],
                      "cuCtxGetCurrent": [ctypes.POINTER(p)], "cuModuleLoadData": [ctypes.POINTER(p), p], "cuModuleGetFunction": [ctypes.POINTER(p), p, ctypes.c_char_p],
                      "cuLaunchKernel": [p, u, u, u, u, u, u, u, p, ctypes.POINTER(p), p]}
        try:
            for name, arguments in signatures.items():
                getattr(lib, name).argtypes = arguments
        except AttributeError:  # a driver without one of these entry points
            return False
        _state = lib
    return _state


def _step_function(lib):
    context = ctypes.c_void_p()
    _check(lib.cuCtxGetCurrent(ctypes.byref(context)), "cuCtxGetCurrent")
    function = _modules.get(context.value)
    if function is None:
        image = (Path(__file__).with_name("graph_loop_step.fatbin")).read_bytes()
        module, function = ctypes.c_void_p(), ctypes.c_void_p()
        buffer = ctypes.create_string_buffer(image, len(image))
        if lib.cuModuleLoadData(ctypes.byref(module), buffer):  # no code for this GPU in the fatbin
            function = False
        else:
            _check(lib.cuModuleGetFunction(ctypes.byref(function), module, b"mmgp_loop_step"), "cuModuleGetFunction")
        _modules[context.value] = function
    return function


def available(device=None):
    """Whether loops can be recorded on this device: an NVIDIA GPU, a driver of CUDA 12.3 or later, and step kernel code for the GPU."""
    lib = _driver()
    if not lib:
        return False
    index = torch.cuda.current_device() if device is None or torch.device(device).index is None else torch.device(device).index
    known = _available.get(index)
    if known is None:
        with torch.cuda.device(index):
            torch.cuda.synchronize()  # the device's context exists (PyTorch creates it lazily)
            known = _available[index] = bool(_step_function(lib))
    return known


_PAUSE_PATH = Path.home() / ".triton" / "autotune" / "mmgp_graph_loop.json"
_PAUSE_VERSION = 2  # 2: bodies of several kernels of real size (a body of one tiny kernel measured 3.0 us for 4.4 to 5.2 with real ones)
_pauses = None  # "GPU name | driver version" -> microseconds per turn


def _pause_key(index):
    version = ctypes.c_int()
    _driver().cuDriverGetVersion(ctypes.byref(version))
    return f"{torch.cuda.get_device_name(index)} | driver {version.value}"


def pause_us(device):
    """The time a loop turn adds to its body's kernels on this GPU, in microseconds (the WHILE node's decision and the step kernel): measured
    once per GPU and driver (a few milliseconds, never while a stream records) and kept in ~/.triton/autotune/mmgp_graph_loop.json; None
    while unknown during a recording."""
    global _pauses
    index = torch.device(device).index if torch.device(device).index is not None else torch.cuda.current_device()
    if _pauses is None:
        try:
            payload = json.loads(_PAUSE_PATH.read_text(encoding="utf-8"))
            _pauses = payload["pauses"] if payload.get("version") == _PAUSE_VERSION else {}
        except (OSError, ValueError, KeyError):
            _pauses = {}
    key = _pause_key(index)
    if key not in _pauses:
        if torch.cuda.is_current_stream_capturing() or not available(index):
            return None
        started = time.time()
        _pauses[key] = _measure_pause(index)
        try:
            _PAUSE_PATH.parent.mkdir(parents=True, exist_ok=True)
            _PAUSE_PATH.write_text(json.dumps({"version": _PAUSE_VERSION, "pauses": _pauses}, indent=1), encoding="utf-8")
        except OSError as error:
            tqdm.write(f"[mmgp] Could not save the CUDA graph loop timing {_PAUSE_PATH}: {error}")
        tqdm.write(f"[mmgp] CUDA graph loops: {_pauses[key]:.2f} us per turn on this GPU (measured in {time.time() - started:.2f} s).")
    return _pauses[key]


def _measure_pause(index, turns=64, kernels=4):
    """A loop of `turns` turns of `kernels` kernels over 8 MiB against the same kernels unrolled, replays alternated: the extra time per turn
    (bodies of real kernels pay more than one tiny kernel: their launch cannot hide behind the previous one)."""
    device = torch.device("cuda", index)
    with torch.cuda.device(index), torch.inference_mode():
        x, counter = torch.zeros(2 << 20, device=device), torch.zeros(1, dtype=torch.int64, device=device)
        stream(device)
        unrolled, looped = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(unrolled):
            for _ in range(turns * kernels):
                x.add_(1)
        with torch.cuda.graph(looped):
            with loop(turns, counter):
                for _ in range(kernels):
                    x.add_(1)
        best = {"unrolled": float("inf"), "looped": float("inf")}
        for _ in range(5):
            for name, graph in (("unrolled", unrolled), ("looped", looped)):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                best[name] = min(best[name], start.elapsed_time(end))
        del unrolled, looped
    return max(0.0, (best["looped"] - best["unrolled"]) * 1000 / turns)


def stream(device):
    """The stream loop bodies of a device are recorded on: create and prepare it outside capture (the library setups of capture streams,
    such as split-K counters, apply to it too), mmgp.cuda_graphs.capture_stream() does."""
    index = torch.device(device).index if torch.device(device).index is not None else torch.cuda.current_device()
    body = _streams.get(index)
    if body is None:
        body = _streams[index] = torch.cuda.Stream(index)
    return body


@contextlib.contextmanager
def loop(iterations, counter, pool=None):
    """Records the body of a loop of `iterations` turns (>= 1) into the capture of the current stream; counter: one int64 on the device,
    zeroed here and incremented after each turn. The body is recorded on stream(device); its allocations come from `pool` (the memory pool
    of the capture, as given to capture_begin), and without one it must allocate nothing. Its memory then stays the body stream's until
    the capture ends, as for any stream that a capture forks to."""
    if iterations < 1:
        raise ValueError("a CUDA graph loop runs its body once at least")
    lib = _driver()
    outer = torch.cuda.current_stream(counter.device)
    counter.zero_()
    status, capture_id, graph, size = ctypes.c_int(), ctypes.c_uint64(), ctypes.c_void_p(), ctypes.c_size_t()
    dependencies, edges = ctypes.c_void_p(), ctypes.c_void_p()
    _check(lib.cuStreamGetCaptureInfo_v3(outer.cuda_stream, ctypes.byref(status), ctypes.byref(capture_id), ctypes.byref(graph), ctypes.byref(dependencies),
                                         ctypes.byref(edges), ctypes.byref(size)), "cuStreamGetCaptureInfo")
    if status.value != _CAPTURING:
        raise RuntimeError("[mmgp] a CUDA graph loop is recorded during a capture")
    context, handle = ctypes.c_void_p(), ctypes.c_uint64()
    _check(lib.cuCtxGetCurrent(ctypes.byref(context)), "cuCtxGetCurrent")
    _check(lib.cuGraphConditionalHandleCreate(ctypes.byref(handle), graph, context, 1, _ASSIGN_DEFAULT), "cuGraphConditionalHandleCreate")
    params = _NodeParams(type=_CONDITIONAL)
    conditional = _ConditionalParams.from_buffer(params.union)
    conditional.handle, conditional.type, conditional.size, conditional.ctx = handle.value, _WHILE, 1, context.value
    node = ctypes.c_void_p()
    _check(lib.cuGraphAddNode_v2(ctypes.byref(node), graph, dependencies, edges, size, ctypes.byref(params)), "cuGraphAddNode")
    _check(lib.cuStreamUpdateCaptureDependencies_v2(outer.cuda_stream, ctypes.byref(node), None, 1, _SET_DEPENDENCIES), "cuStreamUpdateCaptureDependencies")
    body_graph = conditional.graph_out[0]
    body = stream(counter.device)
    function = _step_function(lib)
    device = counter.device.index
    with torch.cuda.stream(body):
        _check(lib.cuStreamBeginCaptureToGraph(body.cuda_stream, body_graph, None, None, 0, _RELAXED), "cuStreamBeginCaptureToGraph")
        if pool is not None:  # the body's allocations: the capture's pool, as those of the streams it forks to
            torch._C._cuda_beginAllocateCurrentStreamToPool(device, pool)
        try:
            yield
            arguments = (ctypes.c_void_p(counter.data_ptr()), ctypes.c_longlong(iterations), ctypes.c_uint64(handle.value))
            pointers = (ctypes.c_void_p * 3)(*(ctypes.cast(ctypes.byref(a), ctypes.c_void_p) for a in arguments))
            _check(lib.cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, body.cuda_stream, pointers, None), "cuLaunchKernel")
        finally:
            if pool is not None:  # as torch.cuda.use_mem_pool: the use this routing took is given back (the capture keeps its own)
                torch._C._cuda_endAllocateToPool(device, pool)
                torch._C._cuda_releasePool(device, pool)
            ended, nodes = ctypes.c_void_p(), ctypes.c_size_t()
            _check(lib.cuStreamEndCapture(body.cuda_stream, ctypes.byref(ended)), "cuStreamEndCapture")
            _check(lib.cuGraphGetNodes(body_graph, None, ctypes.byref(nodes)), "cuGraphGetNodes")
            stats["loops"] += 1
            stats["body_nodes"] += nodes.value
