# Copyright (c) 2024-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""VRAM debug mode of the mmgp allocator: which tensors fill the VRAM at its peak, and where in the code they come from.

    from mmgp.allocator import debug
    debug.start(min_mb=16)          # the allocator must be installed (mode "vmm")
    ...                             # run the workload
    debug.report("vram.json")       # JSON for an agent, and vram.md, a summary

Every allocation of min_mb or more is recorded with its origin: the innermost module running (its path in the models that mmgp's
offload loads, or in those given to name_models), a tag, and the Python stack (raw frames, formatted in the report: about 8 us per
recorded allocation; max_stacks_per_s limits them). Tags: `x._allocator_tag = "kv cache"` names the allocation of a tensor (views share
it, the first tag is kept), `with debug.tag("decode tile"):` names the allocations of a block of code. The allocator copies the records
at the peak of each phase (one phase per model that mmgp loads to the GPU, or debug.mark("name")) and when an allocation fails, and totals
the allocations of each origin (count, bytes, lifetime of the freed ones: scratch buffers). The report lists, for each phase, the
tensors alive at its peak grouped by origin, with their sizes, ages at the peak, tags and stacks; the recorded allocations still alive
when it is written; the totals of each origin. Heap breakdowns distinguish unused whole chunks, occupied-block padding, and unused
space in partially occupied chunks. Allocated and reserved peaks are recorded separately, with simultaneous values at each peak.
Tracked padding covers allocations above the recording threshold; occupied blocks include pending GPU users. An idle graph's
reserved scratch mappings must remain available for replay and are not, by themselves, evidence of a leak.
Without debug mode the tags cost nothing (a plain attribute, an empty context).
The origin is asked from inside the allocation, which takes the GIL: a thread holding the GIL while waiting for a lock that the
allocating thread holds would deadlock (PyTorch's own memory history has the same constraint).

Chunk log, for memory that two owners share (a different question: not what fills the VRAM, but who wrote over what). MMGP_VRAM_CHUNK_LOG=<file> starts it
with the process; or start_chunk_log(path). It keeps the history of every chunk (up to 6 M events, about 430 MB of RAM), and writes to the file, with the Python
stack, when a chunk gets two owners, with the chunk's history and the pools (the first REPORTED_VIOLATIONS times). MMGP_VRAM_WATCH_SIZES=4325376,... adds a line with
thread, stream and stack for allocations of exactly these sizes (4325376 bytes: a cuBLAS workspace). chunk_report(tensor) writes what the allocator knows of a tensor's
chunk, at the moment damage is seen; pools_report(label) lists the pools and their live blocks. Off, each event costs one relaxed load.

Command line: python -m mmgp.allocator.debug summary report.json | diff before.json after.json"""
import contextlib
import ctypes
import json
import os
import re
import sys
import threading
import time
import traceback

import torch
from tqdm import tqdm

from . import events as _events

KINDS = ("chunk range", "small pool", "mid-size pool", "spilled", "graph pool", "driver spill")
GB, MB = 2 ** 30, 2 ** 20


class _Entry(ctypes.Structure):
    _fields_ = [("ptr", ctypes.c_uint64), ("size", ctypes.c_int64), ("origin", ctypes.c_int64), ("seq", ctypes.c_int64), ("kind", ctypes.c_int64), ("time", ctypes.c_double)]


class _Header(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int64) for name in ("phase", "allocated", "reserved", "spilled", "chunks", "small_pool", "mid_pool", "graph_pools",
                                                    "cached_ranges", "spare", "small_cached", "request", "entries")] + [("time", ctypes.c_double)]


_ORIGIN = ctypes.CFUNCTYPE(ctypes.c_int64, ctypes.c_size_t, ctypes.c_int)
_local = threading.local()
_rec = None  # the recording in progress


class _Recording:
    def __init__(self, min_mb, stacks, max_stacks_per_s):
        self.threshold = int(min_mb * MB)
        self.min_mb = min_mb
        self.stacks = stacks
        self.stack_interval = 1 / max_stacks_per_s if max_stacks_per_s else 0
        self.last_stack = 0.0
        self.origin_ids, self.origin_keys = {}, []  # (module, tag, stack id) <-> origin id
        self.stack_ids, self.stack_frames = {}, []  # raw frames <-> stack id
        self.phase_ids, self.phase_names = {}, []
        self.phase = None
        self.tags_by_seq = {}
        self.module_names = {}  # id(module) -> path
        self.named_models = set()
        self.patched = []
        self.t0 = 0.0  # the library's recording clock starts with start()
        self.event_cursor = self.release_cursor = 0  # the MMGP events (events.py) and releases already reported


def _library():
    from mmgp import allocator
    if allocator.active != "vmm":
        raise RuntimeError("The VRAM debug mode needs the mmgp allocator: allocator.install('vmm') before CUDA starts.")
    lib = allocator._lib
    if not getattr(lib, "_debug_ready", False):
        lib.vmm_debug_start.argtypes = (ctypes.c_size_t, _ORIGIN)
        lib.vmm_debug_mark.argtypes = (ctypes.c_int64,)
        lib.vmm_debug_seq.argtypes = (ctypes.c_void_p,)
        lib.vmm_debug_seq.restype = ctypes.c_int64
        lib.vmm_debug_phases.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64)
        lib.vmm_debug_phases.restype = ctypes.c_int64
        lib.vmm_debug_snapshot.argtypes = (ctypes.c_int, ctypes.c_int64, ctypes.POINTER(_Header), ctypes.POINTER(_Entry), ctypes.c_int64)
        lib.vmm_debug_snapshot.restype = ctypes.c_int64
        if hasattr(lib, "vmm_debug_heaps"):
            lib.vmm_debug_reserved_snapshot.argtypes = lib.vmm_debug_snapshot.argtypes
            lib.vmm_debug_reserved_snapshot.restype = ctypes.c_int64
            lib.vmm_debug_heaps.argtypes = (ctypes.c_int, ctypes.c_int64, ctypes.c_int, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64)
            lib.vmm_debug_heaps.restype = ctypes.c_int64
        lib.vmm_debug_live.argtypes = (ctypes.c_int, ctypes.POINTER(_Entry), ctypes.c_int64)
        lib.vmm_debug_live.restype = ctypes.c_int64
        lib.vmm_debug_origins.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64)
        lib.vmm_debug_origins.restype = ctypes.c_int64
        lib._debug_ready = True
    return lib


def _origin(size, device):  # called by the allocator for each allocation of the threshold and more
    rec = _rec
    if rec is None:
        return -1
    try:
        modules, tags = getattr(_local, "modules", None), getattr(_local, "tags", None)
        stack_id = -1
        if rec.stacks:
            now = time.perf_counter()
            if now - rec.last_stack >= rec.stack_interval:
                rec.last_stack = now
                frame, frames = sys._getframe().f_back, []
                while frame is not None:
                    frames.append((frame.f_code, frame.f_lineno))
                    frame = frame.f_back
                frames = tuple(frames)
                stack_id = rec.stack_ids.get(frames)
                if stack_id is None:
                    stack_id = rec.stack_ids[frames] = len(rec.stack_frames)
                    rec.stack_frames.append(frames)
        key = (modules[-1] if modules else "", tags[-1] if tags else "", stack_id)
        origin = rec.origin_ids.get(key)
        if origin is None:
            origin = rec.origin_ids[key] = len(rec.origin_keys)
            rec.origin_keys.append(key)
        return origin
    except Exception:
        return -1


_origin_callback = _ORIGIN(_origin)  # kept for the life of the process: the library may call it while a recording stops


def _module_entered(module, args):
    modules = getattr(_local, "modules", None)
    if modules is None:
        modules = _local.modules = []
    rec = _rec
    name = rec.module_names.get(id(module)) if rec is not None else None
    modules.append(name or (modules[-1] + "/" if modules else "") + type(module).__name__)


def _module_left(module, args, output):
    modules = getattr(_local, "modules", None)
    if modules:
        modules.pop()


_module_hooks, _module_hook_users = [], 0  # shared with the RAM debug mode (ram_debug), installed once while either records


def _use_module_hooks():
    global _module_hook_users
    if _module_hook_users == 0:
        _module_hooks[:] = [torch.nn.modules.module.register_module_forward_pre_hook(_module_entered),
                            torch.nn.modules.module.register_module_forward_hook(_module_left, always_call=True)]
    _module_hook_users += 1


def _release_module_hooks():
    global _module_hook_users
    _module_hook_users -= 1
    if _module_hook_users == 0:
        for hook in _module_hooks:
            hook.remove()
        _module_hooks.clear()


def _tag_get(tensor):
    return tensor.__dict__.get("_allocator_tag")


def _tag_set(tensor, name):
    tensor.__dict__["_allocator_tag"] = name
    rec = _rec
    if rec is not None and tensor.is_cuda:
        seq = _library().vmm_debug_seq(tensor.untyped_storage().data_ptr())
        if seq >= 0:
            rec.tags_by_seq.setdefault(seq, name)


def _watch_mmgp(rec):  # a phase per model that mmgp's offload loads to the GPU, whose modules get their paths
    from mmgp import offload
    for attribute in ("gpu_load", "gpu_load_blocks"):
        original = getattr(offload.offload, attribute)

        def wrapped(self, model_id, *args, _original=original, **kwargs):
            if model_id != rec.phase:
                if model_id not in rec.named_models and model_id in self.models:
                    name_models({model_id: self.models[model_id]})
                mark(model_id)
            return _original(self, model_id, *args, **kwargs)
        setattr(offload.offload, attribute, wrapped)
        rec.patched.append((offload.offload, attribute, original))


_CHUNK_CALLBACK = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_uint64, ctypes.c_uint64)
REPORTED_VIOLATIONS = 3
_chunk_log = None
_chunk_violations = 0


def start_chunk_log(path, watch_sizes=()):
    """See the module's description. The callback runs from inside the allocation, without the allocator's lock."""
    global _chunk_log
    lib = _library()
    lib.vmm_chunk_log_report.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p)
    lib.vmm_chunk_log_pools.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p)

    @_CHUNK_CALLBACK
    def event(reason, pointer, size):
        global _chunk_violations
        reason = reason.decode()
        stack = " < ".join(f"{os.path.basename(frame.filename)}:{frame.lineno} {frame.name}" for frame in reversed(traceback.extract_stack(limit=16)[:-1]))
        with open(path, "a", encoding="utf-8") as out:
            out.write(f"{reason}: allocation at {pointer:#x}, size {size}, thread {threading.current_thread().name}, stream {torch.cuda.current_stream().cuda_stream:#x}, from {stack}\n")
        if reason.startswith("watched"):
            return
        _chunk_violations += 1
        if _chunk_violations <= REPORTED_VIOLATIONS:
            tqdm.write(f"[MMGP] VRAM chunk log: {reason}, written to {path}")
            lib.vmm_chunk_log_report(None, _device(), os.fsencode(path))
            lib.vmm_chunk_log_pools(_device(), os.fsencode(path), f"at violation {_chunk_violations}".encode())

    watched = (ctypes.c_size_t * len(watch_sizes))(*watch_sizes)
    _chunk_log = event, path
    lib.vmm_chunk_log_start.argtypes = (_CHUNK_CALLBACK, ctypes.POINTER(ctypes.c_size_t), ctypes.c_int)
    lib.vmm_chunk_log_start(event, watched, len(watch_sizes))


def chunk_report(target, path=None):
    """What the allocator knows of the chunk behind a tensor or an address: its owners and mappings now, its history, every violation and the end of the log."""
    pointer = target if isinstance(target, int) else target.untyped_storage().data_ptr()
    _library().vmm_chunk_log_report(ctypes.c_void_p(pointer), _device(), os.fsencode(path or _chunk_log[1]))


def pools_report(label, path=None):
    """The allocator's pools now: their chunks, and the sizes of their live blocks."""
    _library().vmm_chunk_log_pools(_device(), os.fsencode(path or _chunk_log[1]), label.encode())


def start(min_mb=16, stacks=True, max_stacks_per_s=0, watch_mmgp=True):
    """Starts recording the allocations of min_mb or more (a new recording if one is in progress)."""
    global _rec
    if _rec is not None:
        stop()
    lib = _library()
    rec = _Recording(min_mb, stacks, max_stacks_per_s)
    _use_module_hooks()
    torch.Tensor._allocator_tag = property(_tag_get, _tag_set)
    _events.start()
    rec.event_cursor, rec.release_cursor = len(_events.events), len(_events.releases)
    if watch_mmgp:
        _watch_mmgp(rec)
    _rec = rec
    rec.t0 = time.perf_counter()
    lib.vmm_debug_start(rec.threshold, _origin_callback)
    mark("start")


def stop():
    global _rec
    rec = _rec
    if rec is None:
        return
    _library().vmm_debug_stop()
    _rec = None
    _release_module_hooks()
    for owner, attribute, original in rec.patched:
        setattr(owner, attribute, original)
    _events.stop()
    del torch.Tensor._allocator_tag


def active():
    return _rec is not None


def mark(name):
    """The allocations that follow count for the peak of phase `name` (a phase seen before continues its peak)."""
    rec = _rec
    if rec is None:
        return
    if name not in rec.phase_ids:
        rec.phase_ids[name] = len(rec.phase_names)
        rec.phase_names.append(name)
    rec.phase = name
    _library().vmm_debug_mark(rec.phase_ids[name])


@contextlib.contextmanager
def tag(name):
    """Tags the allocations made in the block (the innermost tag wins)."""
    tags = getattr(_local, "tags", None)
    if tags is None:
        tags = _local.tags = []
    tags.append(name)
    try:
        yield
    finally:
        tags.pop()


def name_models(models):
    """Names the modules of {name: model} by their paths, e.g. transformer.blocks.12.attn.to_q."""
    rec = _rec
    if rec is None:
        return
    for model_name, model in models.items():
        if isinstance(model, torch.nn.Module):
            rec.named_models.add(model_name)
            for path, module in model.named_modules():
                rec.module_names[id(module)] = f"{model_name}.{path}" if path else model_name


def reset():
    """New peaks and totals (between two generations); the allocations alive stay recorded. The allocator's peaks restart too: the header of
    the next report covers the same period as its phases."""
    rec = _rec
    if rec is None:
        return
    lib = _library()
    lib.vmm_debug_reset()
    lib.vmm_reset_peaks(_device())
    rec.phase = None
    mark("start")


def _device():
    return torch.cuda.current_device()


def _snapshot(phase, reserved=False):
    lib, header = _library(), _Header()
    read = lib.vmm_debug_reserved_snapshot if reserved else lib.vmm_debug_snapshot
    count = read(_device(), phase, ctypes.byref(header), None, 0)
    buffer = (_Entry * count)()
    count = min(count, read(_device(), phase, ctypes.byref(header), buffer, count))
    return header, list(buffer[:count])


def _heaps(phase, reserved):
    lib = _library()
    if not hasattr(lib, "vmm_debug_heaps"):
        return []
    fields = ("pool_first", "pool_second", "kind", "reserved", "unused", "occupied_blocks", "tracked_requests", "tracked_padding",
              "large_padding", "pending_blocks", "cached_blocks", "small_arena_blocks", "mid_arena_blocks")
    count = lib.vmm_debug_heaps(_device(), phase, reserved, None, 0)
    buffer = (ctypes.c_int64 * (len(fields) * count))()
    count = min(count, lib.vmm_debug_heaps(_device(), phase, reserved, buffer, count))
    result = []
    for index in range(count):
        row = dict(zip(fields, buffer[index * len(fields):(index + 1) * len(fields)]))
        row["kind"] = ("ordinary", "graph", "driver small", "driver mid")[row["kind"]]
        row["occupied_chunk_slack"] = row["reserved"] - row["unused"] - row["occupied_blocks"] if row["kind"] in ("ordinary", "graph") else None
        result.append(row)
    return result


_INTERNAL = (os.sep + "site-packages" + os.sep, os.path.join("mmgp", "offload.py"), os.path.join("mmgp", "allocator"), "<frozen")


def _frame_text(code, line):
    path = code.co_filename
    try:
        path = os.path.relpath(path)
    except ValueError:
        pass
    if path.startswith(".."):
        index = path.find("site-packages")
        path = path[index:] if index >= 0 else path
    return f"{path.replace(os.sep, '/')}:{line} in {code.co_name}"


_WEIGHT_LOADERS = {"to_gpu", "_take_slot", "_move_block", "copy_block", "cpu_to_gpu"}  # mmgp offload functions that allocate model weights


def _where(rec, stack_id):  # the innermost frame of the application's own code, or mmgp's when it loads model weights
    if stack_id < 0:
        return ""
    for code, line in rec.stack_frames[stack_id]:  # library frames may come first (e.g. PyTorch's default device mode)
        if code.co_name in _WEIGHT_LOADERS and code.co_filename.endswith(os.path.join("mmgp", "offload.py")):
            return _frame_text(code, line) + " (model weights loaded by mmgp)"
        if not any(part in code.co_filename for part in _INTERNAL):
            return _frame_text(code, line)
    return _frame_text(*rec.stack_frames[stack_id][0])


def _normalized(module):
    return re.sub(r"\.\d+(?=\.|/|$)", ".*", module)


def _groups(rec, entries, now):  # entries grouped by origin, the largest first
    groups = {}
    for entry in entries:
        module, region, stack_id = rec.origin_keys[entry.origin] if 0 <= entry.origin < len(rec.origin_keys) else ("", "", -1)
        name = rec.tags_by_seq.get(entry.seq, "")
        key = (_normalized(module), region, name, stack_id)
        group = groups.get(key)
        if group is None:
            group = groups[key] = {"mb": 0.0, "count": 0, "sizes_mb": [], "module": key[0], "tag": region, "tensor_tag": name, "stack": stack_id,
                                   "where": _where(rec, stack_id), "kinds": set(), "age_ms": [], "modules": set()}
        group["mb"] += entry.size / MB
        group["count"] += 1
        group["sizes_mb"].append(round(entry.size / MB, 1))
        group["kinds"].add(KINDS[entry.kind])
        group["age_ms"].append(round((now - entry.time) * 1000, 1))
        if len(group["modules"]) < 4:
            group["modules"].add(module)
    result = sorted(groups.values(), key=lambda group: -group["mb"])
    for group in result:
        group["mb"] = round(group["mb"], 1)
        group["sizes_mb"] = sorted(group["sizes_mb"], reverse=True)[:12]
        group["kinds"], group["modules"] = sorted(group["kinds"]), sorted(group["modules"])
        ages = group.pop("age_ms")
        group["age_ms"] = [min(ages), max(ages)]
    return result


def _phase_report(rec, header, entries, name, reserved=False):
    large = sum(entry.size for entry in entries)
    return {"name": name, "allocated_gb": round(header.allocated / GB, 3), "reserved_gb": round(header.reserved / GB, 3), "spilled_gb": round(header.spilled / GB, 3),
            "large_gb": round(large / GB, 3), "smaller_gb": round((header.allocated - large) / GB, 3), "time_s": round(header.time, 3),
            "request_mb": round(header.request / MB, 1) if header.request else None,
            "reserved_detail_gb": {key: round(getattr(header, key) / GB, 3) for key in ("chunks", "small_pool", "mid_pool", "graph_pools", "cached_ranges", "spare", "small_cached")},
            "allocated_bytes": header.allocated, "reserved_bytes": header.reserved,
            "heaps": _heaps(-1 if header.request else header.phase, reserved) if header.time > 0 else [], "groups": _groups(rec, entries, header.time)}


def report(path=None):
    """The report (a dict), also written to path (.json) with a summary (.md) when path is given."""
    rec = _rec
    if rec is None:
        raise RuntimeError("No VRAM debug recording in progress: debug.start() first.")
    lib = _library()
    phase_ids = (ctypes.c_int64 * 1024)()
    phases, reserved_peaks = [], []
    for phase in phase_ids[:lib.vmm_debug_phases(_device(), phase_ids, 1024)]:
        header, entries = _snapshot(phase)
        phases.append(_phase_report(rec, header, entries, rec.phase_names[phase] if phase < len(rec.phase_names) else str(phase)))
        if hasattr(lib, "vmm_debug_heaps"):
            header, entries = _snapshot(phase, reserved=True)
            if header.time > 0:
                reserved_peaks.append(_phase_report(rec, header, entries, rec.phase_names[phase] if phase < len(rec.phase_names) else str(phase), reserved=True))
    header, entries = _snapshot(-1)
    oom = _phase_report(rec, header, entries, rec.phase_names[header.phase] if header.phase < len(rec.phase_names) else "") if header.time > 0 else None
    count = lib.vmm_debug_live(_device(), None, 0)
    buffer = (_Entry * count)()
    live_entries = list(buffer[:min(count, lib.vmm_debug_live(_device(), buffer, count))])
    totals = (ctypes.c_int64 * (6 * 65536))()
    merged = {}  # the origins of the same module path (all blocks) and allocation site together
    for row in range(min(65536, lib.vmm_debug_origins(_device(), totals, 65536))):
        origin, count, total, largest, freed, life_us = totals[6 * row: 6 * row + 6]
        module, region, stack_id = rec.origin_keys[origin] if 0 <= origin < len(rec.origin_keys) else ("", "", -1)
        where = _where(rec, stack_id)
        entry = merged.setdefault((_normalized(module), region, where), {"module": _normalized(module), "tag": region, "stack": stack_id, "where": where,
                                                                        "count": 0, "total": 0, "max": 0, "freed": 0, "life_us": 0})
        entry["count"] += count
        entry["total"] += total
        entry["max"] = max(entry["max"], largest)
        entry["freed"] += freed
        entry["life_us"] += life_us
    origins = [{"module": entry["module"], "tag": entry["tag"], "stack": entry["stack"], "where": entry["where"], "count": entry["count"],
                "total_gb": round(entry["total"] / GB, 3), "max_mb": round(entry["max"] / MB, 1), "freed": entry["freed"],
                "mean_life_ms": round(entry["life_us"] / entry["freed"] / 1000, 2) if entry["freed"] else None} for entry in merged.values()]
    origins.sort(key=lambda origin: -origin["total_gb"])
    live = _groups(rec, live_entries, time.perf_counter() - rec.t0)
    used = {group["stack"] for phase in phases + reserved_peaks + ([oom] if oom else []) for group in phase["groups"]} | {group["stack"] for group in live} | {origin["stack"] for origin in origins[:200]}
    entries, rec.event_cursor = _events.since(rec.event_cursor)
    releases = []  # the models released since the previous report, and the recorded VRAM their modules still hold
    for release in _events.releases[rec.release_cursor:]:
        models = set(release["models"])
        held = [group for group in live if group["module"] and group["module"].split(".")[0] in models]
        releases.append({"time_s": round(release["t"] - rec.t0, 2), "models": sorted(models), "alive": sorted(model_id for model_id, info in release["models"].items() if info["ref"]() is not None),
                         "files": {model_id: info["files"] for model_id, info in release["models"].items() if info["files"]},
                         "vram_still_held_gb": round(sum(group["mb"] for group in held) / 1024, 3), "groups": held})
    rec.release_cursor = len(_events.releases)
    from . import stats
    allocator = {key: round(value / GB, 3) if key not in ("chunks", "live_large", "recoveries") else value for key, value in stats().items()}
    result = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "threshold_mb": rec.min_mb, "allocator": allocator, "phases": phases, "reserved_peaks": reserved_peaks, "out_of_memory": oom, "live_now": live, "origins": origins[:200],
              "releases": releases, "events": _events.as_json(entries, rec.t0),
              "stacks": {str(stack_id): [_frame_text(code, line) for code, line in rec.stack_frames[stack_id][:40]] for stack_id in sorted(used) if stack_id >= 0}}
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as writer:
            json.dump(result, writer, indent=1)
        with open(os.path.splitext(path)[0] + ".md", "w", encoding="utf-8") as writer:
            writer.write(summary(result))
    return result


def _table(groups, top):
    lines = ["| MB | count | module | allocated at | tag | age at peak (ms) |", "|---:|---:|---|---|---|---|"]
    for group in groups[:top]:
        tag_text = " / ".join(filter(None, (group["tag"], group["tensor_tag"])))
        age = group["age_ms"]
        lines.append(f"| {group['mb']:.0f} | {group['count']} | {group['module'] or '-'} | {group['where'] or '-'} | {tag_text or '-'} | "
                     f"{age[0]:.0f}{'' if age[0] == age[1] else f'-{age[1]:.0f}'} |")
    rest = groups[top:]
    if rest:
        lines.append(f"| {sum(group['mb'] for group in rest):.0f} | {sum(group['count'] for group in rest)} | {len(rest)} more origins | | | |")
    return "\n".join(lines)


def _phase_summary(phase, title):
    detail = phase["reserved_detail_gb"]
    heaps = ""
    if phase.get("heaps"):
        heaps = "\n\n| Heap | Reserved MiB | Unused Chunks / Driver Capacity MiB | Occupied Blocks MiB | Occupied Chunk Slack MiB | Tracked Padding MiB | Pending Blocks MiB | Cached Blocks MiB |\n|---|---:|---:|---:|---:|---:|---:|---:|\n"
        for heap in phase["heaps"]:
            if not heap["reserved"]:
                continue
            identity = f" ({heap['pool_first']}, {heap['pool_second']})" if heap["kind"] in ("ordinary", "graph") else ""
            keys = ("reserved", "unused", "occupied_blocks", "occupied_chunk_slack", "tracked_padding", "pending_blocks", "cached_blocks")
            heaps += "| " + heap["kind"] + identity + " | " + " | ".join("-" if heap.get(key) is None else f"{heap[key] / MB:.2f}" for key in keys) + " |\n"
        heaps += "\nTracked Padding covers recorded allocations only. Occupied Blocks include pending GPU users; padding is part of those blocks. "
        heaps += "Occupied Chunk Slack is free space inside chunks that still contain occupied blocks. Driver capacity and cached blocks are separate."
    return (f"## {title}\n\nIn use {phase['allocated_gb']:.2f} GiB: {phase['large_gb']:.2f} in the tensors below, {phase['smaller_gb']:.2f} in smaller ones. "
            f"Reserved {phase['reserved_gb']:.2f} GiB (chunks {detail['chunks']:.2f}, of which cached ranges {detail['cached_ranges']:.2f} and spare "
            f"{detail['spare']:.2f}; small pool {detail['small_pool']:.2f}; mid-size pool {detail['mid_pool']:.2f}; graph pools {detail['graph_pools']:.2f}), "
            f"spilled {phase['spilled_gb']:.2f} GiB.{heaps}\n\n{_table(phase['groups'], 25)}\n\n")


def _coverage_warnings(result):
    """What the phases below do not show: a peak of the period above all of theirs (reached without a phase snapshot), and memory in
    allocations below the recording threshold (counted, not itemized)."""
    warnings, allocator = [], result.get("allocator") or {}
    top = max((phase["allocated_gb"] for phase in result["phases"]), default=0.0)
    if allocator and allocator["peak_allocated"] > top + 0.02:
        warnings.append(f"**The peak in use of this period ({allocator['peak_allocated']:.2f} GiB) is above every phase peak below ({top:.2f} GiB)**: "
                        "it was reached outside them (before the recording or between two reports), its tensors are not listed.\n\n")
    smaller = [phase for phase in result["phases"] if phase["smaller_gb"] >= max(0.25, 0.2 * phase["allocated_gb"])]
    if smaller:
        warnings.append("**Allocations below the recording threshold weigh "
                        + ", ".join(f"{phase['smaller_gb']:.2f} GiB at the peak of {phase['name']}" for phase in smaller)
                        + f"**: not itemized at {result['threshold_mb']} MB, a lower threshold (e.g. 1 MB) lists them.\n\n")
    return warnings


def summary(result):
    """Markdown summary of a report."""
    text = [f"# VRAM debug report, {result['created']}: allocations of {result['threshold_mb']} MB and more\n\n"]
    allocator = result.get("allocator")
    if allocator:
        text.append(f"VRAM allocator now: in use {allocator['allocated']:.2f} GiB, reserved {allocator['reserved']:.2f} GiB (graph pools {allocator['graph_pools']:.2f}); "
                    f"peaks since the last reset: in use {allocator['peak_allocated']:.2f}, reserved {allocator['peak_reserved']:.2f} GiB; spilled to RAM {allocator['spilled']:.2f} GiB "
                    f"pinned, {allocator['driver_spilled']:.2f} GiB through the driver; {allocator['recoveries']} recoveries (VRAM short).\n\n")
    for release in result.get("releases", []):
        files = release.get("files", {})
        names = ", ".join(model_id + (" (" + ", ".join(files[model_id]) + ")" if model_id in files else "") for model_id in release["models"])
        text.append(f"Release of {names} at {release['time_s']:.1f} s: "
                    + (f"**models still alive: {', '.join(release['alive'])}**; " if release["alive"] else "")
                    + (f"**{release['vram_still_held_gb']:.2f} GiB of recorded VRAM still held by their modules**:\n\n{_table(release['groups'], 15)}\n\n" if release["groups"]
                       else "no recorded VRAM left by their modules.\n\n"))
    text.extend(_coverage_warnings(result))
    for phase in result["phases"]:
        text.append(_phase_summary(phase, f"Peak of phase {phase['name']} ({phase['time_s']:.1f} s)"))
    for phase in result.get("reserved_peaks", []):
        text.append(_phase_summary(phase, f"Reserved peak of phase {phase['name']} ({phase['time_s']:.1f} s)"))
    if result["out_of_memory"]:
        oom = result["out_of_memory"]
        text.append(_phase_summary(oom, f"Out of memory in phase {oom['name']} ({oom['time_s']:.1f} s), {oom['request_mb']} MB requested"))
    text.append("## Totals by origin (scratch buffers: many allocations, short lives)\n\n| allocations | total GiB | max MB | mean life (ms) | module | allocated at | tag |\n|---:|---:|---:|---:|---|---|---|\n")
    for origin in result["origins"][:25]:
        life = origin["mean_life_ms"]
        text.append(f"| {origin['count']} | {origin['total_gb']:.2f} | {origin['max_mb']:.0f} | {'-' if life is None else f'{life:.1f}'} | "
                    f"{origin['module'] or '-'} | {origin['where'] or '-'} | {origin['tag'] or '-'} |\n")
    text.append(f"\n## Recorded allocations alive when the report was written\n\n{_table(result['live_now'], 25)}\n\n")
    if "events" in result:
        text.append("## MMGP memory events since the previous report\n\n" + _events.timeline_json(result["events"]))
    text.append("Stacks (innermost frame first) are in the JSON report under `stacks`, by the `stack` id of each group.\n")
    return "".join(text)


def diff(before, after):
    """Markdown comparison of two reports: the peaks of the phases, and the origins whose memory at the peak changed."""
    lines = ["# VRAM debug comparison\n"]
    phases_before = {phase["name"]: phase for phase in before["phases"]}
    for phase in after["phases"]:
        old = phases_before.get(phase["name"])
        if old is None:
            lines.append(f"## {phase['name']}: only in the second report, {phase['allocated_gb']:.2f} GiB\n")
            continue
        lines.append(f"## {phase['name']}: peak in use {old['allocated_gb']:.2f} -> {phase['allocated_gb']:.2f} GiB "
                     f"({phase['allocated_gb'] - old['allocated_gb']:+.2f}), reserved {old['reserved_gb']:.2f} -> {phase['reserved_gb']:.2f}; "
                     f"below the threshold {old['smaller_gb']:.2f} -> {phase['smaller_gb']:.2f} GiB\n")
        # net by model first: the same weights allocated at another site or under another module path (whole model / blocks) cancel out
        by_model = {}
        for index, report_phase in enumerate((old, phase)):
            for group in report_phase["groups"]:
                by_model.setdefault((group["module"] or "-").split(".")[0], [0.0, 0.0])[index] += group["mb"]
        net = sorted(((model, mb) for model, mb in by_model.items() if abs(mb[1] - mb[0]) >= 1), key=lambda item: -abs(item[1][1] - item[1][0]))
        if net:
            lines.append("| net change MB | model (- : no module recorded) |\n|---:|---|")
            lines.extend(f"| {new_mb - old_mb:+.0f} | {module} |" for module, (old_mb, new_mb) in net[:12])
            lines.append("")
        key = lambda group: (group["module"], re.sub(r":\d+ in ", " in ", group["where"]), group["tag"], group["tensor_tag"])  # line numbers move with edits
        sizes = {}
        for index, report_phase in enumerate((old, phase)):
            for group in report_phase["groups"]:
                sizes.setdefault(key(group), [0.0, 0.0])[index] += group["mb"]
        changes = sorted(sizes.items(), key=lambda item: -abs(item[1][1] - item[1][0]))
        lines.append("| MB before | MB after | change | module | allocated at |\n|---:|---:|---:|---|---|")
        for (module, where, region, name), (old_mb, new_mb) in changes[:20]:
            if abs(new_mb - old_mb) >= 1:
                lines.append(f"| {old_mb:.0f} | {new_mb:.0f} | {new_mb - old_mb:+.0f} | {module or '-'} | {where or '-'} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def _main(argv):
    if len(argv) == 2 and argv[0] == "summary":
        with open(argv[1], encoding="utf-8") as reader:
            print(summary(json.load(reader)))
    elif len(argv) == 3 and argv[0] == "diff":
        with open(argv[1], encoding="utf-8") as first, open(argv[2], encoding="utf-8") as second:
            print(diff(json.load(first), json.load(second)))
    else:
        print("usage: python -m mmgp.allocator.debug summary report.json | diff before.json after.json")


if __name__ == "__main__":
    _main(sys.argv[1:])
