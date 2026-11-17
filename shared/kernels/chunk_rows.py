# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Rows per chunk of an operation computed by chunks of rows (an FFN), probed once per operation and GPU.

The time per row of a chunk depends on its rows: too few and the kernels' launches end on part-empty waves or partial tiles, too many
and the chunk's working set leaves the cache; in between some sizes are better than others (RTX 5090, the FFN of a 5376-wide
transformer with a LoRA: 2,048 and 3,072 rows are 7 % faster per row than 2,688). The probe times one chunk at each size of a grid
(multiples of STEP rows up to the chunk the caller's own rule gives, MOST_ROWS at most) and at the rule's own size, on scratch rows, and
choose() then takes the size that computes the caller's rows fastest: the rule's own when nothing measured is faster. Saved in a versioned cache file under ~/.triton/autotune, shared by the sessions."""
import json
import time
from pathlib import Path

import torch
from tqdm import tqdm

VERSION = 1  # bumped when the benchmark changes
STEP = 256  # rows between two sizes of the grid: whole GEMM row tiles
LEAST_ROWS, MOST_ROWS = 512, 4096  # of the grid
GAIN = 0.005  # of the operation's time: what a smaller chunk (one more chunk and its kernels) has to save
_PATH = Path.home() / ".triton" / "autotune" / "wan2gp_chunk_rows.json"
_saved = None


def _load():
    global _saved
    if _saved is None:
        try:
            payload = json.loads(_PATH.read_text(encoding="utf-8"))
            _saved = payload["curves"] if payload.get("version") == VERSION else {}
        except (OSError, ValueError, KeyError):
            _saved = {}
    return _saved


def curve(identity, device, label, prepare, most):
    """{rows: milliseconds per row of one chunk of that many rows} of operation `identity` on this device, for the grid's sizes up to
    `most` rows and for `most` itself: from the cache file, completed now where sizes are missing. prepare(rows) returns launch(rows), which computes one
    chunk of that many rows on scratch tensors of its own. None below two sizes; what is known (maybe nothing) while the probe
    cannot run (CUDA graph recording, PyTorch compilation)."""
    index = torch.cuda.current_device() if device.index is None else device.index
    key = f"{identity}|{torch.cuda.get_device_name(index)}"
    sizes = sorted({*range(LEAST_ROWS, min(most, MOST_ROWS) + 1, STEP), most})  # the caller's own size is one of the candidates
    known = _load().setdefault(key, {})
    missing = [rows for rows in sizes if str(rows) not in known]
    if len(sizes) < 2:
        return None
    if missing and not (torch.cuda.is_current_stream_capturing() or torch.compiler.is_compiling()):
        started = time.time()
        launch = prepare(missing[-1])
        for rows in missing:
            launch(rows)
            best = float("inf")
            for _ in range(5):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(3):
                    launch(rows)
                end.record()
                end.synchronize()
                best = min(best, start.elapsed_time(end) / 3)
            known[str(rows)] = round(best / rows, 7)
        del launch
        fastest = min(sizes, key=lambda rows: known[str(rows)])
        tqdm.write(f"[Kernels] {label} chunks probed up to {sizes[-1]} rows: fastest per row at {fastest} ({time.time() - started:.1f} s).")
        try:
            _PATH.parent.mkdir(parents=True, exist_ok=True)
            _PATH.write_text(json.dumps({"version": VERSION, "curves": _saved}, indent=1), encoding="utf-8")
        except OSError as error:
            tqdm.write(f"[Kernels] Could not save the chunk probe {_PATH}: {error}")
    return {rows: known[str(rows)] for rows in sizes if str(rows) in known} or None


def choose(times, tokens, most_rows, most_chunks):
    """Rows per chunk for `tokens` rows from times {rows: time per row}: the size, of at most most_rows rows and giving at most
    most_chunks chunks, that computes them fastest (whole chunks at their own time per row, the last, shorter one at that of the next
    size up); among the sizes within GAIN of it, the largest (fewer chunks). None when no probed size qualifies."""
    sizes = sorted(times)

    def total(rows):
        full, rest = divmod(tokens, rows)
        return full * rows * times[rows] + rest * times[next((size for size in sizes if size >= rest), sizes[-1])]

    allowed = [rows for rows in sizes if rows <= most_rows and -(-tokens // rows) <= most_chunks]
    if not allowed:
        return None
    least = min(total(rows) for rows in allowed)
    return max(rows for rows in allowed if total(rows) <= least * (1 + GAIN))
