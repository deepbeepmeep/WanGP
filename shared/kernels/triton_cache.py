# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""One-shot Triton cache maintenance, before any startup probe or loaded kernel can reuse it."""
import os
import re
import shutil
from pathlib import Path

from tqdm import tqdm

_REQUEST = ".purge-triton-cache"
# Triton versions encode their cache keys as base32, URL-safe base64 or hex.
_CACHE_KEY = re.compile(r"[A-Z2-7]{52}|[A-Za-z0-9_-]{43}|[0-9a-f]{64}|[0-9a-f]{32}")
_ARTIFACT_SUFFIXES = (".cubin", ".hsaco", ".ttir", ".ttgir", ".autotune.json", ".so", ".pyd")
_PROBE_FILES = ("WAN2GP_FP8_BLOCK_TRITON_CACHE", "WAN2GP_FP8_TRITON_CACHE", "WAN2GP_NVFP4_TRITON_CACHE",
                "WAN2GP_QUANTO_INT8_AUTOTUNE_CACHE")
last_purge = ""


def request_purge(config_dir):
    (Path(config_dir) / _REQUEST).touch()


def _kernel_caches():
    triton_home = Path(os.environ.get("TRITON_HOME") or Path.home())
    yield Path(os.environ.get("TRITON_CACHE_DIR") or triton_home / ".triton" / "cache")
    # torch.compile can redirect Triton to a per-device cache, including WanGP's per-INT8-backend folders.
    from torch._inductor.runtime.runtime_utils import default_cache_dir
    inductor = Path(os.environ.get("TORCHINDUCTOR_CACHE_DIR") or default_cache_dir())
    for pattern in ("triton/[0-9]*", "wangp_int8_v*/*/triton/[0-9]*"):
        yield from (path for path in inductor.glob(pattern) if path.name.isdecimal())


def _cache_entries():
    for cache in set(_kernel_caches()):
        if not cache.is_dir():
            continue
        # A custom TRITON_CACHE_DIR can be a shared folder. Remove only compiler-owned entries.
        for entry in cache.iterdir():
            if _CACHE_KEY.fullmatch(entry.name) and entry.is_dir():
                if any(artifact.name.endswith(_ARTIFACT_SUFFIXES) for artifact in entry.iterdir()):
                    yield entry, True

    probes = set((Path.home() / ".triton" / "autotune").glob("*.json"))
    probes.update(Path(os.environ[key]).expanduser() for key in _PROBE_FILES if os.environ.get(key))
    if os.environ.get("WGPKERNELS_OPS_CACHE"):
        probes.add(Path(os.environ["WGPKERNELS_OPS_CACHE"]))  # WGPKernels uses this path literally.
    yield from ((path, False) for path in probes if path.is_file())


def _purge():
    failure = None
    for path, kernel_dir in _cache_entries():
        try:
            if kernel_dir and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
        except FileNotFoundError:
            continue
        except OSError as exc:
            failure = exc  # A helper locked by another process should not prevent the other kernels and probes from resetting.
    if failure is not None:
        raise failure


def purge_if_requested(config_dir):
    global last_purge
    try:
        try:
            (Path(config_dir) / _REQUEST).unlink()
        except FileNotFoundError:
            return
        tqdm.write("Clearing Triton kernel cache and performance probe results...")
        _purge()
    except OSError as exc:
        last_purge = "Triton cache purge incomplete. Check the console for the affected path and retry."
        tqdm.write(f"{last_purge} {exc}")
    else:
        last_purge = "Triton cache and performance probes cleared at startup."
        tqdm.write(last_purge)
