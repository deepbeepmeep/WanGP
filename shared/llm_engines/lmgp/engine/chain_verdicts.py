# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Whether the long draft chain pays on a machine, kept from one session to the next: one entry per GPU, checkpoint and KV cache layout,
beside the kernels' tuning caches. The runner measures what a pass costs with the usual drafts and with the long chain's
(draft_sources.WindowCosts) and records the verdict once; a loader that finds a verdict against the long chain leaves it out, with the
CUDA graphs of its longer windows. An entry is measured again when the GPU, the checkpoint, the layout, the draft counts, the GGUF kernel
package or VERSION change."""
import importlib.util
import json
import time
from pathlib import Path

import torch

VERSION = 1  # of the rule and of what it measures
SHORT_CONTEXT = 20480  # a verdict against the long chain counts only when measured with at most this many tokens of context: beyond, the verification attention weighs on the long pass
_entries = None


def _path():
    return Path.home() / ".triton" / "autotune" / "wan2gp_long_chain.json"


def _load():
    global _entries
    if _entries is None:
        try:
            payload = json.loads(_path().read_text(encoding="utf-8"))
            _entries = payload["entries"] if payload.get("version") == VERSION else {}
        except (OSError, ValueError, KeyError):
            _entries = {}
    return _entries


def _kernels() -> str:
    if importlib.util.find_spec("llamacpp_gguf_cuda") is None:
        return "no gguf kernels"
    import llamacpp_gguf_cuda
    return f"gguf {llamacpp_gguf_cuda.__version__}" + (f" r{llamacpp_gguf_cuda.iq_rows_revision()}" if hasattr(llamacpp_gguf_cuda, "iq_rows_revision") else "")


def key(device, checkpoint: str, kv_layout: str, usual: int, long: int) -> str:
    props = torch.cuda.get_device_properties(device)
    return "|".join((props.name, f"{props.major}.{props.minor}", f"{props.total_memory >> 20} MiB", Path(checkpoint).name, kv_layout or "bf16", f"{usual}+{long} drafts", _kernels(), f"cuda {torch.version.cuda}"))


def verdict(entry_key: str):
    """{"pays": bool, "ratio": the long pass's time against the usual one's, "context": tokens, "date": ...}, or None when never measured."""
    return _load().get(entry_key)


def record(entry_key: str, ratio: float, pays: bool, context: int) -> None:
    entries = _load()
    entries[entry_key] = {"pays": bool(pays), "ratio": round(float(ratio), 3), "context": int(context), "date": time.strftime("%Y-%m-%d")}
    path = _path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps({"version": VERSION, "entries": entries}, indent=1, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
    except OSError as error:
        print(f"[Deepy][Speculative] Long chain verdict not saved: {error}")
