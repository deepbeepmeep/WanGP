# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""How a kernel made of equal, long programs fills a GPU, probed once per kernel and GPU.

A device runs a fixed number of such programs at once (a few per multiprocessor). The programs of one launch last the same time, so a
launch runs as waves of that many programs and its last wave leaves the device part empty; launches do not overlap. Splitting one
launch into several (attention by groups of heads) therefore costs time when the smaller launches end on emptier waves than the whole
one did.

The probe needs one thing from the kernel: a way to launch it over a given number of rows (query rows of one head, for an attention).
What a program covers (its block of rows) is declared by the kernel's wrapper when its source gives it, else found: while a launch
has at most one program per multiprocessor its time does not depend on its rows, and the largest block before the time rises is the
program's. The probe then times one launch at each program count of a grid (an eighth of the multiprocessors apart), finds the wave
(the smallest whole number of programs per multiprocessor after which the times repeat) and keeps the measured length of a
part-filled wave, for the first wave and for the following ones. Saved in a versioned cache file under ~/.triton/autotune, shared by
the sessions."""
import json
import statistics
import time
from pathlib import Path

import torch
from tqdm import tqdm

VERSION = 3  # bumped when the benchmark or the model change
GRID_PER_MULTIPROCESSOR = 8  # program counts measured per multiprocessor count
GRID_MULTIPROCESSORS = 12  # the grid ends at this many programs per multiprocessor
MAX_PER_MULTIPROCESSOR = 4  # waves of up to this many programs per multiprocessor are recognized (three periods on the grid)
MAX_ERROR = 0.12  # of a wave: median distance between the times one wave apart and a wave, above which nothing is kept (RTX 5090: 0.03)
PROGRAM_ROWS = (32, 64, 128, 256)  # blocks of rows looked for when the wrapper declares none
FLAT = 1.25  # a launch of at most one program per multiprocessor lasts no more than this many times the shortest one
_PATH = Path.home() / ".triton" / "autotune" / "wan2gp_launch_waves.json"
_saved = None
_models = {}


class LaunchWaves:
    """rows: rows per program. slots: programs the device runs at once. first / tail: length, in waves, of a wave filled up to
    (i + 1) / len of its slots, for a launch's only wave and for the last wave of a longer launch. wave_ms: the time of a whole
    wave of the probe's launches (what a program does there: the caller's to know)."""
    def __init__(self, slots, first, tail, rows=None, wave_ms=None):
        self.slots, self.first, self.tail, self.rows, self.wave_ms = slots, first, tail, rows, wave_ms

    def cost(self, programs):
        """Share of a launch's time lost to its last wave: its time over that of as many programs on always full waves, minus one
        (the probe's last waves at every depth: a flat loss beyond its launches chose slower groups at 9 to 11 waves, H3 720p)."""
        full, rest = divmod(programs, self.slots)
        if rest == 0:
            return 0.0
        table = self.tail if full else self.first
        return (full + table[-(-rest * len(table) // self.slots) - 1]) * self.slots / programs - 1


def _load():
    global _saved
    if _saved is None:
        try:
            payload = json.loads(_PATH.read_text(encoding="utf-8"))
            _saved = payload["waves"] if payload.get("version") == VERSION else {}
        except (OSError, ValueError, KeyError):
            _saved = {}
    return _saved


def fit(times, multiprocessors, rows=None):
    """LaunchWaves from times[i], the time of one launch of grid(multiprocessors)[i] programs; None when they do not repeat wave after
    wave (a busy device, a kernel that is not made of equal programs)."""
    step = GRID_PER_MULTIPROCESSOR
    candidates = []
    for per in range(1, MAX_PER_MULTIPROCESSOR + 1):
        period = per * step  # grid points per wave
        waves = len(times) // period
        wave = times[waves * period - 1] / waves
        candidates.append((statistics.median(abs(times[i] - times[i - period] - wave) for i in range(period, len(times))) / wave, per, wave))
    least = min(candidates)[0]
    # a multiple of the wave repeats as well, and a little better (the first wave is shorter than the next ones): the smallest wins
    error, per, wave = next(candidate for candidate in candidates if candidate[0] <= 2 * least + 0.02)
    if error > MAX_ERROR or wave <= 0:
        return None
    period = per * step
    first = [min(1.0, max(0.0, times[i] / wave)) for i in range(period)]
    tail = []
    for i in range(period):
        lengths = [(times[k * period + i] - times[k * period - 1]) / wave for k in range(1, len(times) // period) if k * period + i < len(times)]
        tail.append(min(1.0, max(0.0, sum(lengths) / len(lengths))))
    for table in (first, tail):  # a fuller wave is never shorter, and a full one is a wave
        table[-1] = 1.0
        for i in range(len(table) - 2, -1, -1):
            table[i] = min(table[i], table[i + 1])
    return LaunchWaves(per * multiprocessors, first, tail, rows, wave)


def grid(multiprocessors):
    return [max(1, round(multiprocessors * (i + 1) / GRID_PER_MULTIPROCESSOR)) for i in range(GRID_PER_MULTIPROCESSOR * GRID_MULTIPROCESSORS)]


def _time(launch):
    launch()
    best = float("inf")
    for _ in range(5):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(3):
            launch()
        end.record()
        end.synchronize()
        best = min(best, start.elapsed_time(end) / 3)
    return best


def program_rows(times):
    """Rows per program from times {block: time of a launch of block x multiprocessors rows}: the largest block of PROGRAM_ROWS
    whose launch lasts no longer than one of fewer rows (at most one program per multiprocessor), when the next one does (two
    programs on some of them). None when the times rise from the start or never: not a kernel of equal programs of such a block."""
    shortest = min(times.values())
    flat = [block for block in sorted(times) if times[block] <= FLAT * shortest]
    block = flat[-1]
    return block if block in PROGRAM_ROWS and block != max(times) and flat == sorted(times)[:len(flat)] else None


def get(identity, device, label, prepare, rows=None):
    """The LaunchWaves of kernel `identity` on this device: from this session, the cache file, or probed now. prepare(most rows)
    returns stage(rows) -> launch(), which runs one launch over that many rows on scratch tensors of its own. rows: the rows a
    program covers when the kernel's source gives them, else found by the probe. None when it cannot be probed (CUDA graph
    recording, PyTorch compilation: asked again later) or was probed without a result (kept for the session)."""
    index = torch.cuda.current_device() if device.index is None else device.index
    key = f"{identity}|{torch.cuda.get_device_name(index)}"
    if key in _models:
        return _models[key]
    saved = _load().get(key)
    if saved is not None:
        model = _models[key] = LaunchWaves(saved["slots"], saved["first"], saved["tail"], saved["rows"], saved["wave_ms"])
        return model
    if torch.cuda.is_current_stream_capturing() or torch.compiler.is_compiling():
        return None
    started = time.time()
    multiprocessors = torch.cuda.get_device_properties(index).multi_processor_count
    counts = grid(multiprocessors)
    declared = rows is not None
    stage = prepare(counts[-1] * (rows or PROGRAM_ROWS[-1]))
    model = None
    if not declared:
        rows = program_rows({block: _time(stage(multiprocessors * block)) for block in (PROGRAM_ROWS[0] // 2, *PROGRAM_ROWS, PROGRAM_ROWS[-1] * 2)})
    if rows is not None:
        model = fit([_time(stage(programs * rows)) for programs in counts], multiprocessors, rows)
    del stage
    _models[key] = model
    if model is None:
        tqdm.write(f"[Kernels] {label} launches could not be probed on this device (another program using it, or a kernel that does not run by waves): "
                   "Head Split keeps equal groups; asked again next session.")
        return None
    tqdm.write(f"[Kernels] {label} launches probed: programs of {rows} rows{'' if declared else ' (found)'}, {model.slots} at once ({time.time() - started:.1f} s).")
    _saved[key] = {"rows": rows, "slots": model.slots, "wave_ms": round(model.wave_ms, 5), "first": [round(v, 4) for v in model.first], "tail": [round(v, 4) for v in model.tail]}
    try:
        _PATH.parent.mkdir(parents=True, exist_ok=True)
        _PATH.write_text(json.dumps({"version": VERSION, "waves": _saved}, indent=1), encoding="utf-8")
    except OSError as error:
        tqdm.write(f"[Kernels] Could not save the launch probe {_PATH}: {error}")
    return model
