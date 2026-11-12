# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice as tl_libdevice

    _TRITON_AVAILABLE = True
except Exception:  # pragma: no cover
    triton = None  # type: ignore
    tl = None  # type: ignore
    tl_libdevice = None  # type: ignore
    _TRITON_AVAILABLE = False


_ENV_ENABLE = "WAN2GP_QUANTO_INT8_TRITON"
_ENV_AUTOTUNE_ENABLE = "WAN2GP_QUANTO_INT8_AUTOTUNE"
_ENV_AUTOTUNE_DEBUG = "WAN2GP_QUANTO_INT8_AUTOTUNE_DEBUG"
_ENV_AUTOTUNE_MAX_M = "WAN2GP_QUANTO_INT8_AUTOTUNE_MAX_M"
_ENV_AUTOTUNE_MAX_SHAPES = "WAN2GP_QUANTO_INT8_AUTOTUNE_MAX_SHAPES"
_ENV_AUTOTUNE_WARMUP = "WAN2GP_QUANTO_INT8_AUTOTUNE_WARMUP"
_ENV_AUTOTUNE_ITERS = "WAN2GP_QUANTO_INT8_AUTOTUNE_ITERS"
_ENV_AUTOTUNE_MIN_SPEEDUP = "WAN2GP_QUANTO_INT8_AUTOTUNE_MIN_SPEEDUP"
_ENV_AUTOTUNE_CACHE = "WAN2GP_QUANTO_INT8_AUTOTUNE_CACHE"
_ENV_AUTOTUNE_VALIDATE = "WAN2GP_QUANTO_INT8_AUTOTUNE_VALIDATE"
_ENV_AUTOTUNE_MAX_ABS_ERR = "WAN2GP_QUANTO_INT8_AUTOTUNE_MAX_ABS_ERR"
_ENV_AUTOTUNE_MAX_REL_ERR = "WAN2GP_QUANTO_INT8_AUTOTUNE_MAX_REL_ERR"
_ENV_AUTOTUNE_LOCK_FUSED_BLOCK_K = "WAN2GP_QUANTO_INT8_AUTOTUNE_LOCK_FUSED_BLOCK_K"
_IS_AVAILABLE = {}
_CONFIG_LEN = 5
_CONVROT_KINDS = ("convrot", "convrot_bias")
_AUTOTUNE_CACHE_VERSION = 3
_AUTOTUNE_CACHE_LOADED = False
_AUTOTUNE_CACHE_DIRTY = False
_AUTOTUNE_CONFIG_CACHE: dict[str, tuple[int, int, int, int, int]] = {}
_AUTOTUNE_SESSION_CACHE: dict[tuple[int, str, str], tuple[int, int, int, int, int]] = {}
_AUTOTUNE_SEEN_SLOTS: set[tuple[int, str, str]] = set()
_AUTOTUNE_SLOTS_TUNED = 0
_AUTOTUNE_DEBUG_OVERRIDE: Optional[bool] = None

# Triton reports this staged large-M int8 tile as requiring 102400 bytes of
# shared memory on SM120, while some cards expose only 101376 bytes per block.
_HIGH_SHARED_MEMORY_CONFIG = (64, 256, 64, 8, 4)
_HIGH_SHARED_MEMORY_CONFIG_BYTES = 102400
_LOW_SHARED_MEMORY_CONFIG = (64, 256, 64, 8, 3)

# Tuned decode-time configs reused from the LM engine's int8 kernels.
_TRITON_SMALL_M_CONFIGS = {
    (2048, 4096): (2, 32, 256, 8, 5),
    (2048, 2048): (1, 32, 64, 2, 4),
    (2048, 12288): (8, 64, 256, 8, 4),
    (6144, 2048): (1, 32, 512, 4, 5),
}
_TRITON_TINY_M_SHAPE_CONFIGS = {
    (2, 3072, 3072): (2, 128, 64, 8, 4),
    (4, 3072, 3072): (4, 256, 64, 4, 4),
    (2, 3072, 1024): (2, 256, 128, 8, 4),
    (4, 3072, 1024): (2, 256, 128, 8, 4),
    (2, 3072, 1536): (2, 64, 64, 4, 4),
    (4, 3072, 1536): (2, 256, 128, 8, 4),
    (2, 3072, 8192): (2, 128, 64, 8, 4),
    (4, 3072, 8192): (2, 128, 64, 8, 4),
    (2, 8192, 3072): (4, 128, 64, 4, 4),
    (4, 8192, 3072): (2, 128, 128, 8, 4),
}
_TRITON_TINY_M_RUNTIME_CONFIGS = {}
_TRITON_TINY_M_PAIR_CONFIGS = {
    (3072, 3072): (2, 128, 64, 8, 4),
    (3072, 1024): (2, 256, 128, 8, 4),
    (3072, 1536): (2, 64, 64, 4, 4),
    (3072, 8192): (2, 128, 64, 8, 4),
    (8192, 3072): (4, 128, 64, 4, 4),
}
_TRITON_SMALL_M_DEFAULT = (4, 256, 64, 4, 4)
_TRITON_SMALL_M_K3072_DEFAULT = (2, 64, 64, 4, 4)
_TRITON_MID_M_DEFAULT = (32, 128, 64, 8, 4)
_TRITON_LARGE_M_DEFAULT = (64, 128, 64, 8, 4)
_TRITON_LARGE_M_SHAPE_CONFIGS = {
    # Hot LTX2 distilled fused-int8 shapes profiled on RTX 50xx.
    (3840, 2048): (64, 256, 64, 8, 4),
    (3840, 15360): (64, 256, 64, 8, 4),
    (3840, 4096): (64, 256, 64, 8, 4),
    (4096, 3840): (64, 256, 64, 8, 4),
    (15360, 3840): (64, 256, 64, 8, 4),
    # Hot WAN2 I2V enhanced-lightning fused-int8 shapes (M ~= 512).
    (4096, 4096): (64, 256, 64, 8, 4),
    (4096, 10240): (64, 256, 64, 8, 4),
    (10240, 4096): (64, 256, 64, 8, 4),
}

_AUTOTUNE_SLOT_REPS = {
    "tiny_k3072_default": ((2, 3072, 2048), (4, 3072, 4096)),
    "tiny_default": ((2, 4096, 1536), (4, 4096, 1536)),
    "mid_default": ((32, 2048, 4096), (32, 3072, 3072), (32, 4096, 4096)),
    "large_n_ge_2048": ((512, 4096, 4096), (3840, 3840, 4096)),
    "large_index_gt_2g": ((512, 4096, 4096), (3840, 3840, 4096)),
    "large_default": ((128, 4096, 1024), (192, 3072, 1536)),
}
_RUNTIME_PROBE_MAX_N = 4096
_INT32_LINEAR_INDEX_LIMIT = 2**31


def _env_flag(name: str, default: str = "1") -> bool:
    val = os.environ.get(name, default)
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def _parse_version(ver: str) -> tuple[int, int]:
    try:
        parts = ver.split(".")
        return int(parts[0]), int(parts[1])
    except Exception:
        return (0, 0)


def _is_stream_capturing() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        return bool(torch.cuda.is_current_stream_capturing())
    except Exception:
        return False


def _autotune_is_blocked() -> bool:
    return torch.compiler.is_compiling() or _is_stream_capturing()


def _use_cg_autotune(device_index: int) -> bool:
    props = torch.cuda.get_device_properties(device_index)
    # Benchmark the same graph-replay regime on all supported NVIDIA tensor-core
    # targets. The device fingerprint separates these results from event timing.
    return torch.version.hip is None and props.major >= 8


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


def _autotune_debug(msg: str) -> None:
    if _AUTOTUNE_DEBUG_OVERRIDE is None:
        debug_on = _env_flag(_ENV_AUTOTUNE_DEBUG, "0")
    else:
        debug_on = bool(_AUTOTUNE_DEBUG_OVERRIDE)
    if debug_on:
        print(f"[WAN2GP][INT8][autotune] {msg}")


def set_autotune_debug(enabled: Optional[bool] = None) -> None:
    global _AUTOTUNE_DEBUG_OVERRIDE
    _AUTOTUNE_DEBUG_OVERRIDE = None if enabled is None else bool(enabled)


def _runtime_compatible(device=None) -> bool:
    if not (_TRITON_AVAILABLE and torch.cuda.is_available()):
        return False
    try:
        cc_major, _ = torch.cuda.get_device_capability(device)
    except Exception:
        return False

    # Triton int8 dot kernels require tensor-core generation GPUs.
    if cc_major < 8:
        return False

    # Keep SM120 safe on older Triton builds that abort at compile time.
    triton_ver = _parse_version(getattr(triton, "__version__", "0.0"))
    if cc_major >= 12 and triton_ver < (3, 6):
        return False

    return True


def is_available(device=None) -> bool:
    if not torch.cuda.is_available():
        return False
    device_index = torch.cuda.current_device() if device is None else torch.device(device).index
    device_index = torch.cuda.current_device() if device_index is None else device_index
    if device_index not in _IS_AVAILABLE:
        _IS_AVAILABLE[device_index] = bool(_runtime_compatible(device) and _env_flag(_ENV_ENABLE, "1"))
    return _IS_AVAILABLE[device_index]


def configure_tiny_m_shape_overrides(configs=None) -> None:
    global _TRITON_TINY_M_RUNTIME_CONFIGS
    configs = configs or {}
    parsed = {}
    for shape, config in configs.items():
        if not isinstance(shape, (tuple, list)) or len(shape) != 3 or not isinstance(config, (tuple, list)) or len(config) != _CONFIG_LEN:
            raise ValueError("Triton tiny-M overrides require (M, K, N) keys and five-value configs")
        shape, config = tuple(int(value) for value in shape), tuple(int(value) for value in config)
        if shape[0] > 4 or min(shape) <= 0 or min(config) <= 0:
            raise ValueError(f"Invalid Triton tiny-M override: shape={shape}, config={config}")
        parsed[shape] = config
    _TRITON_TINY_M_RUNTIME_CONFIGS = parsed


def _tiny_m_shape_config(m: int, k: int, n: int):
    return _TRITON_TINY_M_RUNTIME_CONFIGS.get((m, k, n)) or _TRITON_TINY_M_SHAPE_CONFIGS.get((m, k, n))


def _select_static_triton_int8_config(m: int, k: int, n: int) -> tuple[int, int, int, int, int]:
    if m <= 4:
        cfg = _tiny_m_shape_config(m, k, n)
        if cfg is not None:
            return cfg
        cfg = _TRITON_TINY_M_PAIR_CONFIGS.get((k, n))
        if cfg is not None:
            return cfg
        cfg = _TRITON_SMALL_M_CONFIGS.get((k, n))
        if cfg is not None:
            return cfg
        if k == 3072:
            return _TRITON_SMALL_M_K3072_DEFAULT
        return _TRITON_SMALL_M_DEFAULT
    if m < 64:
        return _TRITON_MID_M_DEFAULT
    if m >= 256:
        cfg = _TRITON_LARGE_M_SHAPE_CONFIGS.get((k, n))
        if cfg is not None:
            return cfg
        if n >= 2048:
            return (64, 256, 64, 8, 4)
    return _TRITON_LARGE_M_DEFAULT


def _dedup_shapes(shapes: tuple[tuple[int, int, int], ...]) -> tuple[tuple[int, int, int], ...]:
    out: list[tuple[int, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for shape in shapes:
        if not isinstance(shape, (list, tuple)) or len(shape) != 3:
            continue
        try:
            m, k, n = (int(shape[0]), int(shape[1]), int(shape[2]))
        except Exception:
            continue
        if m <= 0 or k <= 0 or n <= 0:
            continue
        key = (m, k, n)
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    return tuple(out)


def _resolve_autotune_slot(m: int, k: int, n: int) -> tuple[str, tuple[tuple[int, int, int], ...]]:
    baseline = _select_static_triton_int8_config(m, k, n)
    if m <= 4:
        if _tiny_m_shape_config(m, k, n) is not None:
            slot_id = f"tiny_shape|m={m}|k={k}|n={n}"
            reps = ((m, k, n),)
        elif (k, n) in _TRITON_TINY_M_PAIR_CONFIGS:
            slot_id = f"tiny_pair|k={k}|n={n}"
            reps = ((2, k, n), (4, k, n))
        elif (k, n) in _TRITON_SMALL_M_CONFIGS:
            slot_id = f"tiny_small_pair|k={k}|n={n}"
            reps = ((2, k, n), (4, k, n))
        elif k == 3072:
            slot_id = "tiny_k3072_default"
            reps = _AUTOTUNE_SLOT_REPS[slot_id]
        else:
            slot_id = "tiny_default"
            reps = _AUTOTUNE_SLOT_REPS[slot_id]
    elif m < 64:
        slot_id = "mid_default"
        reps = _AUTOTUNE_SLOT_REPS[slot_id]
    elif max(m * k, n * k, m * n) > _INT32_LINEAR_INDEX_LIMIT:
        slot_id = "large_index_gt_2g"
        reps = _AUTOTUNE_SLOT_REPS[slot_id]
    elif m >= 256 and (k, n) in _TRITON_LARGE_M_SHAPE_CONFIGS:
        slot_id = f"large_hot_pair|k={k}|n={n}"
        reps = ((512, k, n), (3840, k, n))
    elif m >= 256 and n >= 2048:
        slot_id = "large_n_ge_2048"
        reps = _AUTOTUNE_SLOT_REPS[slot_id]
    else:
        slot_id = "large_default"
        reps = _AUTOTUNE_SLOT_REPS[slot_id]

    filtered = [shape for shape in _dedup_shapes(reps) if _select_static_triton_int8_config(shape[0], shape[1], shape[2]) == baseline]
    if len(filtered) == 0:
        filtered = [(m, k, n)]
    return slot_id, tuple(filtered)


def _normalize_config(cfg) -> Optional[tuple[int, int, int, int, int]]:
    if not isinstance(cfg, (list, tuple)) or len(cfg) != _CONFIG_LEN:
        return None
    try:
        c0, c1, c2, c3, c4 = (int(v) for v in cfg)
    except Exception:
        return None
    if c0 <= 0 or c1 <= 0 or c2 <= 0 or c3 <= 0 or c4 <= 0:
        return None
    return (c0, c1, c2, c3, c4)


def _autotune_cache_path() -> Path:
    default_path = str(Path.home() / ".triton" / "autotune" / "wan2gp_int8_autotune_cache.json")
    return Path(os.environ.get(_ENV_AUTOTUNE_CACHE, default_path)).expanduser()


def _load_autotune_cache() -> None:
    global _AUTOTUNE_CACHE_LOADED, _AUTOTUNE_CONFIG_CACHE
    if _AUTOTUNE_CACHE_LOADED:
        return
    _AUTOTUNE_CACHE_LOADED = True
    cache_path = _autotune_cache_path()
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception:
        return
    entries = payload.get("entries", {})
    if not isinstance(entries, dict):
        return
    parsed = {}
    for key, raw_cfg in entries.items():
        if not isinstance(key, str):
            continue
        cfg = _normalize_config(raw_cfg)
        if cfg is not None:
            parsed[key] = cfg
    _AUTOTUNE_CONFIG_CACHE = parsed


def _save_autotune_cache() -> None:
    global _AUTOTUNE_CACHE_DIRTY
    if not _AUTOTUNE_CACHE_DIRTY:
        return
    cache_path = _autotune_cache_path()
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = Path(f"{cache_path}.tmp")
        payload = {"version": _AUTOTUNE_CACHE_VERSION, "entries": {key: list(cfg) for key, cfg in _AUTOTUNE_CONFIG_CACHE.items()}}
        tmp_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        tmp_path.replace(cache_path)
        _AUTOTUNE_CACHE_DIRTY = False
    except Exception as exc:
        _autotune_debug(f"cache write failed: {exc}")


def _device_index(device: Optional[torch.device]) -> int:
    if device is not None and device.type == "cuda" and device.index is not None:
        return int(device.index)
    return int(torch.cuda.current_device())


def _device_fingerprint(device_index: int) -> str:
    props = torch.cuda.get_device_properties(device_index)
    triton_ver = getattr(triton, "__version__", "0.0")
    fingerprint = (
        f"{props.name}|cc={props.major}.{props.minor}|sm={props.multi_processor_count}|"
        f"torch={torch.__version__}|triton={triton_ver}|wan2gp_int8_cache_v={_AUTOTUNE_CACHE_VERSION}"
    )
    return fingerprint + ("|bench=cg-v1" if _use_cg_autotune(device_index) else "")


def _autotune_slot_cache_key(device_index: int, kernel_kind: str, slot_id: str) -> str:
    return f"{_device_fingerprint(device_index)}|{kernel_kind}|slot={slot_id}"


def _autotune_legacy_shape_cache_key(device_index: int, kernel_kind: str, m: int, k: int, n: int) -> str:
    return f"{_device_fingerprint(device_index)}|{kernel_kind}|{m}|{k}|{n}"


def _get_cached_config(device_index: int, kernel_kind: str, slot_id: str, m: int, k: int, n: int) -> Optional[tuple[int, int, int, int, int]]:
    global _AUTOTUNE_CACHE_DIRTY
    _load_autotune_cache()
    slot_key = _autotune_slot_cache_key(device_index, kernel_kind, slot_id)
    cfg = _AUTOTUNE_CONFIG_CACHE.get(slot_key)
    if cfg is not None:
        return cfg
    legacy_key = _autotune_legacy_shape_cache_key(device_index, kernel_kind, m, k, n)
    legacy_cfg = _AUTOTUNE_CONFIG_CACHE.get(legacy_key)
    if legacy_cfg is None:
        return None
    _AUTOTUNE_CONFIG_CACHE[slot_key] = legacy_cfg
    _AUTOTUNE_CACHE_DIRTY = True
    return legacy_cfg


def _set_cached_config(device_index: int, kernel_kind: str, slot_id: str, cfg: tuple[int, int, int, int, int], *, overwrite: bool = True) -> bool:
    global _AUTOTUNE_CACHE_DIRTY
    _load_autotune_cache()
    key = _autotune_slot_cache_key(device_index, kernel_kind, slot_id)
    existing = _AUTOTUNE_CONFIG_CACHE.get(key)
    if existing == cfg:
        return True
    if existing is not None and not overwrite:
        _autotune_debug(f"keeping cached config {existing} for {kernel_kind} slot={slot_id}; refusing recovery overwrite with {cfg}")
        return False
    _AUTOTUNE_CONFIG_CACHE[key] = cfg
    _AUTOTUNE_CACHE_DIRTY = True
    _save_autotune_cache()
    return True


def _drop_cached_config(device_index: int, kernel_kind: str, slot_id: str, m: int, k: int, n: int) -> None:
    global _AUTOTUNE_CACHE_DIRTY
    _load_autotune_cache()
    keys = (
        _autotune_slot_cache_key(device_index, kernel_kind, slot_id),
        _autotune_legacy_shape_cache_key(device_index, kernel_kind, m, k, n),
    )
    removed = False
    for key in keys:
        if key in _AUTOTUNE_CONFIG_CACHE:
            del _AUTOTUNE_CONFIG_CACHE[key]
            removed = True
    if removed:
        _AUTOTUNE_CACHE_DIRTY = True


def _config_compatible_with_baseline(
    kind: str,
    baseline: tuple[int, int, int, int, int],
    cfg: tuple[int, int, int, int, int],
) -> bool:
    if kind in _CONVROT_KINDS:
        return cfg[2] == 64
    if baseline == _LOW_SHARED_MEMORY_CONFIG and cfg == _HIGH_SHARED_MEMORY_CONFIG:
        return False
    if kind == "fused" and _env_flag(_ENV_AUTOTUNE_LOCK_FUSED_BLOCK_K, "1"):
        # Fused blockscale kernel computes row scales per K-chunk; changing block_k changes numerics.
        return int(cfg[2]) == int(baseline[2])
    return True


def _runtime_probe_shape(
    kind: str,
    m: int,
    k: int,
    n: int,
    baseline: tuple[int, int, int, int, int],
) -> tuple[int, int, int, tuple[int, int, int, int, int]]:
    if kind in _CONVROT_KINDS:
        return m, k, min(n, _RUNTIME_PROBE_MAX_N), baseline
    if n <= _RUNTIME_PROBE_MAX_N:
        return m, k, n, baseline
    probe_candidates = []
    for probe_n in (_RUNTIME_PROBE_MAX_N, 3072, 2048, 1536, 1024, 768, 512, 256, 128, 64):
        if probe_n >= n:
            continue
        if probe_n % 8 != 0:
            continue
        probe_candidates.append(probe_n)
    for probe_n in probe_candidates:
        probe_baseline = _select_static_triton_int8_config(m, k, probe_n)
        if not _config_compatible_with_baseline(kind, baseline, probe_baseline):
            continue
        if not _config_compatible_with_baseline(kind, probe_baseline, baseline):
            continue
        return m, k, probe_n, probe_baseline
    return m, k, n, baseline


def _candidate_configs(
    baseline: tuple[int, int, int, int, int],
    m: int,
    k: int,
    n: int,
    *,
    kind: str,
) -> list[tuple[int, int, int, int, int]]:
    out = [baseline]
    if m <= 4:
        out.extend(
            [
                (1, 64, 64, 2, 4),
                (1, 128, 64, 4, 4),
                (2, 64, 64, 4, 4),
                (2, 128, 64, 4, 4),
                (2, 128, 128, 8, 4),
                (2, 256, 64, 8, 4),
                (4, 128, 64, 4, 4),
                (4, 256, 64, 4, 4),
                (8, 128, 64, 4, 4),
            ]
        )
        shape_cfg = _tiny_m_shape_config(m, k, n)
        if shape_cfg is not None:
            out.append(shape_cfg)
        pair_cfg = _TRITON_TINY_M_PAIR_CONFIGS.get((k, n))
        if pair_cfg is not None:
            out.append(pair_cfg)
    elif m <= 16:
        out.extend([(8, 128, 64, 4, 4), (8, 256, 64, 8, 4), (16, 128, 64, 8, 4), (16, 256, 64, 8, 4), (32, 128, 64, 8, 4)])
    if kind in _CONVROT_KINDS:
        out.extend((bm, bn, bk, warps, stages) for bm, bn, bk, warps, _ in tuple(out) for stages in (1, 2))
    dedup: list[tuple[int, int, int, int, int]] = []
    seen = set()
    for cfg in out:
        norm = _normalize_config(cfg)
        if norm is None or norm in seen:
            continue
        if not _config_compatible_with_baseline(kind, baseline, norm):
            continue
        seen.add(norm)
        dedup.append(norm)
    return dedup


def _looks_like_unsupported_dot_tile(cfg: tuple[int, int, int, int, int]) -> bool:
    block_m, block_n, block_k, _, _ = cfg
    # Triton int8 dot kernels can reject tiny tiles on some runtimes (e.g. decode-time M<=4).
    return block_m < 16 or block_n < 16 or block_k < 32


def _compile_recovery_candidates(
    kind: str,
    baseline: tuple[int, int, int, int, int],
    preferred: tuple[int, int, int, int, int],
    m: int,
    k: int,
    n: int,
) -> list[tuple[int, int, int, int, int]]:
    block_k = max(32, int(baseline[2]))
    if block_k % 32 != 0:
        block_k = ((block_k + 31) // 32) * 32
    conservative_large_tiles = [
        (16, 32, block_k, 4, 4),
        (16, 64, block_k, 4, 4),
        (16, 128, block_k, 4, 4),
        (32, 32, block_k, 4, 4),
        (32, 64, block_k, 8, 4),
        (32, 128, block_k, 8, 4),
        (64, 64, block_k, 8, 4),
        (64, 128, block_k, 8, 4),
    ]
    raw = [preferred]
    raw.extend(_candidate_configs(baseline, m, k, n, kind=kind))
    raw.extend(conservative_large_tiles)
    # Larger tiles need fewer buffered stages on GPUs with limited shared memory.
    raw.extend((bm, bn, bk, warps, stages) for bm, bn, bk, warps, _ in conservative_large_tiles for stages in (1, 2))

    dedup: list[tuple[int, int, int, int, int]] = []
    seen = set()
    for cfg in raw:
        norm = _normalize_config(cfg)
        if norm is None or norm in seen:
            continue
        if not _config_compatible_with_baseline(kind, baseline, norm):
            continue
        seen.add(norm)
        dedup.append(norm)

    if baseline not in dedup:
        dedup.append(baseline)

    if len(dedup) <= 1:
        return dedup

    head = dedup[0]
    tail = dedup[1:]
    non_tiny = [cfg for cfg in tail if not _looks_like_unsupported_dot_tile(cfg)]
    tiny = [cfg for cfg in tail if _looks_like_unsupported_dot_tile(cfg)]
    return [head, *non_tiny, *tiny]


def _launch_candidate(kind: str, cfg: tuple[int, int, int, int, int], tensors: tuple[torch.Tensor, ...], m: int, n: int, k: int) -> None:
    block_m, block_n, block_k, num_warps, num_stages = cfg
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    if kind in _CONVROT_KINDS:
        from shared.kernels.convrot_int8_triton import convrot_int8_mm
        x, weight, scale, bias, out = tensors
        convrot_int8_mm(x, weight, scale, out, bias, cfg)
        return
    if kind == "fused":
        x_mm_c, qweight_c, b_scale_c, out = tensors
        _fused_dynamic_int8_blockscale_gemm_kernel[grid](
            x_mm_c,
            qweight_c,
            b_scale_c,
            out,
            m,
            n,
            k,
            x_mm_c.stride(0),
            x_mm_c.stride(1),
            qweight_c.stride(0),
            qweight_c.stride(1),
            out.stride(0),
            out.stride(1),
            block_m=block_m,
            block_n=block_n,
            block_k=block_k,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return
    a_int8_c, b_int8_c, a_scale_c, b_scale_c, out = tensors
    _scaled_int8_gemm_kernel[grid](
        a_int8_c,
        b_int8_c,
        a_scale_c,
        b_scale_c,
        out,
        m,
        n,
        k,
        a_int8_c.stride(0),
        a_int8_c.stride(1),
        b_int8_c.stride(0),
        b_int8_c.stride(1),
        out.stride(0),
        out.stride(1),
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )


def _create_bench_tensors(kind: str, device: torch.device, m: int, k: int, n: int) -> tuple[torch.Tensor, ...]:
    if kind == "fused" or kind in _CONVROT_KINDS:
        x_mm_c = torch.randn((m, k), device=device, dtype=torch.bfloat16)
        qweight_c = torch.randint(-128, 128, (n, k), device=device, dtype=torch.int8)
        b_scale_c = torch.rand((n,), device=device, dtype=torch.float32).add_(1e-4)
        out = torch.empty((m, n), device=device, dtype=torch.bfloat16)
        if kind in _CONVROT_KINDS:
            bias = torch.randn(n, device=device, dtype=torch.bfloat16) if kind == "convrot_bias" else None
            return (x_mm_c, qweight_c, b_scale_c, bias, out)
        return (x_mm_c, qweight_c, b_scale_c, out)
    a_int8_c = torch.randint(-128, 128, (m, k), device=device, dtype=torch.int8)
    b_int8_c = torch.randint(-128, 128, (n, k), device=device, dtype=torch.int8)
    a_scale_c = torch.rand((m,), device=device, dtype=torch.float32).add_(1e-4)
    b_scale_c = torch.rand((n,), device=device, dtype=torch.float32).add_(1e-4)
    out = torch.empty((m, n), device=device, dtype=torch.bfloat16)
    return (a_int8_c, b_int8_c, a_scale_c, b_scale_c, out)


def _run_candidate_once_with_error(
    kind: str,
    cfg: tuple[int, int, int, int, int],
    tensors: tuple[torch.Tensor, ...],
    m: int,
    k: int,
    n: int,
) -> tuple[Optional[torch.Tensor], Optional[Exception]]:
    try:
        if kind in _CONVROT_KINDS:
            x, weight, scale, bias, _ = tensors
            out = torch.empty((m, n), device=x.device, dtype=torch.bfloat16)
            _launch_candidate(kind, cfg, (x, weight, scale, bias, out), m, n, k)
            torch.cuda.synchronize(x.device)
            return out, None
        if kind == "fused":
            x_mm_c, qweight_c, b_scale_c, _ = tensors
            out = torch.empty((m, n), device=x_mm_c.device, dtype=torch.bfloat16)
            _launch_candidate(kind, cfg, (x_mm_c, qweight_c, b_scale_c, out), m, n, k)
            torch.cuda.synchronize(x_mm_c.device)
            return out, None
        a_int8_c, b_int8_c, a_scale_c, b_scale_c, _ = tensors
        out = torch.empty((m, n), device=a_int8_c.device, dtype=torch.bfloat16)
        _launch_candidate(kind, cfg, (a_int8_c, b_int8_c, a_scale_c, b_scale_c, out), m, n, k)
        torch.cuda.synchronize(a_int8_c.device)
        return out, None
    except Exception as exc:
        _autotune_debug(f"single-run failed for {kind} shape=({m},{k},{n}) cfg={cfg}: {exc}")
        return None, exc


def _run_candidate_once(kind: str, cfg: tuple[int, int, int, int, int], tensors: tuple[torch.Tensor, ...], m: int, k: int, n: int) -> Optional[torch.Tensor]:
    out, _ = _run_candidate_once_with_error(kind, cfg, tensors, m, k, n)
    return out


def _ensure_compile_compatible_config(
    kind: str,
    device_index: int,
    slot_id: str,
    preferred: tuple[int, int, int, int, int],
    baseline: tuple[int, int, int, int, int],
    m: int,
    k: int,
    n: int,
    rep_shapes: tuple[tuple[int, int, int], ...],
) -> tuple[tuple[int, int, int, int, int], Optional[Exception]]:
    if _autotune_is_blocked():
        return preferred, None
    device = torch.device("cuda", device_index)
    _ = rep_shapes
    probe_m, probe_k, probe_n, _ = _runtime_probe_shape(kind, m, k, n, baseline)
    # Probe a reduced runtime-compatible shape when possible so compile validation does
    # not allocate fake giant heads (for example lm_head with 248k vocab rows).
    probe_shapes = ((probe_m, probe_k, probe_n),)
    tensors_by_shape = {shape: _create_bench_tensors(kind, device, *shape) for shape in probe_shapes}
    candidates = _compile_recovery_candidates(kind, baseline, preferred, m, k, n)
    last_error: Optional[Exception] = None

    for cfg in candidates:
        all_ok = True
        for probe_m, probe_k, probe_n in probe_shapes:
            _, probe_err = _run_candidate_once_with_error(
                kind,
                cfg,
                tensors_by_shape[(probe_m, probe_k, probe_n)],
                probe_m,
                probe_k,
                probe_n,
            )
            if probe_err is not None:
                all_ok = False
                last_error = probe_err
                break
        if all_ok:
            if cfg != preferred:
                _autotune_debug(
                    f"compile recovery picked {cfg} for {kind} slot={slot_id} shape=({m},{k},{n}) "
                    f"instead of {preferred}"
                )
            return cfg, None

    if kind in _CONVROT_KINDS:
        raise RuntimeError(f"No compatible Triton {kind} configuration for shape=({m},{k},{n})") from last_error
    if last_error is not None:
        _autotune_debug(
            f"compile recovery failed for {kind} slot={slot_id} shape=({m},{k},{n}); "
            f"keeping {preferred}. last_error={last_error}"
        )
    return preferred, last_error


def _candidate_matches_baseline(
    baseline_out: torch.Tensor,
    candidate_out: torch.Tensor,
    *,
    max_abs_limit: float,
    rel_limit: float,
) -> tuple[bool, float, float]:
    if not torch.isfinite(candidate_out).all().item():
        return False, float("inf"), float("inf")
    base_f = baseline_out.float()
    cand_f = candidate_out.float()
    diff = (base_f - cand_f).abs()
    max_abs = float(diff.max().item())
    denom = base_f.abs().mean().clamp_min(1e-6)
    rel = float((diff.mean() / denom).item())
    return (max_abs <= max_abs_limit and rel <= rel_limit), max_abs, rel


def _validate_config(
    kind: str,
    device: torch.device,
    m: int,
    k: int,
    n: int,
    baseline: tuple[int, int, int, int, int],
    cfg: tuple[int, int, int, int, int],
) -> bool:
    if not _config_compatible_with_baseline(kind, baseline, cfg):
        return False
    if cfg == baseline:
        return True
    if not _env_flag(_ENV_AUTOTUNE_VALIDATE, "1"):
        return True
    max_abs_limit = max(0.0, _env_float(_ENV_AUTOTUNE_MAX_ABS_ERR, 0.25))
    rel_limit = max(0.0, _env_float(_ENV_AUTOTUNE_MAX_REL_ERR, 0.001))
    tensors = _create_bench_tensors(kind, device, m, k, n)
    baseline_out = _run_candidate_once(kind, baseline, tensors, m, k, n)
    candidate_out = _run_candidate_once(kind, cfg, tensors, m, k, n)
    if baseline_out is None or candidate_out is None:
        return False
    ok, max_abs, rel = _candidate_matches_baseline(
        baseline_out,
        candidate_out,
        max_abs_limit=max_abs_limit,
        rel_limit=rel_limit,
    )
    if not ok:
        _autotune_debug(
            f"rejecting config {cfg} for {kind} shape=({m},{k},{n}) "
            f"vs baseline {baseline}: max_abs={max_abs:.6f}, rel={rel:.6f}"
        )
    return ok


def _benchmark_config_ms(kind: str, cfg: tuple[int, int, int, int, int], tensors: tuple[torch.Tensor, ...], device: torch.device, m: int, k: int, n: int) -> Optional[float]:
    if _autotune_is_blocked():
        return None
    warmup = max(1, _env_int(_ENV_AUTOTUNE_WARMUP, 2))
    iters = max(1, _env_int(_ENV_AUTOTUNE_ITERS, 5))
    graph = None
    try:
        with torch.cuda.device(device):
            for _ in range(warmup):
                _launch_candidate(kind, cfg, tensors, m, n, k)
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            if _use_cg_autotune(_device_index(device)):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                    for _ in range(iters):
                        _launch_candidate(kind, cfg, tensors, m, n, k)
                graph.replay()
                torch.cuda.synchronize(device)
                timings = []
                for _ in range(3):
                    start.record()
                    graph.replay()
                    end.record()
                    end.synchronize()
                    timings.append(float(start.elapsed_time(end)) / iters)
                return sorted(timings)[1]
            start.record()
            for _ in range(iters):
                _launch_candidate(kind, cfg, tensors, m, n, k)
            end.record()
            end.synchronize()
            return float(start.elapsed_time(end)) / float(iters)
    except Exception as exc:
        _autotune_debug(f"benchmark failed for {kind} shape=({m},{k},{n}) cfg={cfg}: {exc}")
        return None
    finally:
        if graph is not None:
            graph.reset()


def _can_tune_slot(slot_key: tuple[int, str, str]) -> bool:
    global _AUTOTUNE_SLOTS_TUNED
    if slot_key in _AUTOTUNE_SEEN_SLOTS:
        return True
    max_shapes = max(0, _env_int(_ENV_AUTOTUNE_MAX_SHAPES, 32))
    if _AUTOTUNE_SLOTS_TUNED >= max_shapes:
        return False
    _AUTOTUNE_SEEN_SLOTS.add(slot_key)
    _AUTOTUNE_SLOTS_TUNED += 1
    return True


def _benchmark_slot_config_ms(
    kind: str,
    cfg: tuple[int, int, int, int, int],
    device: torch.device,
    rep_shapes: tuple[tuple[int, int, int], ...],
    baseline: tuple[int, int, int, int, int],
) -> Optional[float]:
    total = 0.0
    count = 0
    for rep_m, rep_k, rep_n in rep_shapes:
        rep_baseline = baseline if kind in _CONVROT_KINDS else _select_static_triton_int8_config(rep_m, rep_k, rep_n)
        if not _validate_config(kind, device, rep_m, rep_k, rep_n, rep_baseline, cfg):
            return None
        tensors = _create_bench_tensors(kind, device, rep_m, rep_k, rep_n)
        ms = _benchmark_config_ms(kind, cfg, tensors, device, rep_m, rep_k, rep_n)
        if ms is None:
            return None
        total += ms
        count += 1
    if count == 0:
        return None
    return total / float(count)


def _autotune_config(
    kind: str,
    device_index: int,
    m: int,
    k: int,
    n: int,
    baseline: tuple[int, int, int, int, int],
    slot_id: str,
    rep_shapes: tuple[tuple[int, int, int], ...],
) -> tuple[int, int, int, int, int]:
    if _autotune_is_blocked():
        return baseline
    device = torch.device("cuda", device_index)
    cached = _get_cached_config(device_index, kind, slot_id, m, k, n)
    if cached is not None:
        probe_m, probe_k, probe_n, probe_baseline = _runtime_probe_shape(kind, m, k, n, baseline)
        if _validate_config(kind, device, probe_m, probe_k, probe_n, probe_baseline, cached):
            return cached
        _drop_cached_config(device_index, kind, slot_id, m, k, n)
    slot_key = (device_index, kind, slot_id)
    if not _can_tune_slot(slot_key):
        _autotune_debug(f"slot budget reached; keeping baseline for {kind} slot={slot_id} shape=({m},{k},{n})")
        return baseline

    rep_m, rep_k, rep_n = rep_shapes[0]
    rep_baseline = baseline if kind in _CONVROT_KINDS else _select_static_triton_int8_config(rep_m, rep_k, rep_n)
    candidate_seed = rep_baseline if _config_compatible_with_baseline(kind, baseline, rep_baseline) else baseline
    candidates = _candidate_configs(candidate_seed, rep_m, rep_k, rep_n, kind=kind)
    if _use_cg_autotune(device_index) and rep_m <= 4 and baseline[2] == 64:
        candidates.append((4, 64, 64, 4, 4))
    if baseline not in candidates:
        candidates = [baseline, *candidates]

    results: dict[tuple[int, int, int, int, int], float] = {}
    for cfg in candidates:
        ms = _benchmark_slot_config_ms(kind, cfg, device, rep_shapes, baseline)
        if ms is not None:
            results[cfg] = ms
    baseline_ms = results.get(baseline)
    if baseline_ms is None:
        if len(results) > 0:
            recovered_cfg, recovered_ms = min(results.items(), key=lambda item: item[1])
            _set_cached_config(device_index, kind, slot_id, recovered_cfg)
            _autotune_debug(
                f"baseline config failed for {kind} slot={slot_id} shape=({m},{k},{n}); "
                f"using first compilable cfg={recovered_cfg} (ms={recovered_ms:.4f})"
            )
            return recovered_cfg
        _set_cached_config(device_index, kind, slot_id, baseline)
        _autotune_debug(
            f"no compilable configs found during autotune for {kind} slot={slot_id} shape=({m},{k},{n}); "
            f"keeping baseline {baseline}"
        )
        return baseline
    best_cfg, best_ms = min(results.items(), key=lambda item: item[1])
    min_speedup = max(1.0, _env_float(_ENV_AUTOTUNE_MIN_SPEEDUP, 1.02))
    use_best = best_cfg != baseline and best_ms > 0.0 and (baseline_ms / best_ms) >= min_speedup
    picked = best_cfg if use_best else baseline
    probe_m, probe_k, probe_n, probe_baseline = _runtime_probe_shape(kind, m, k, n, baseline)
    if not _validate_config(kind, device, probe_m, probe_k, probe_n, probe_baseline, picked):
        picked = baseline
    _set_cached_config(device_index, kind, slot_id, picked)
    if use_best:
        _autotune_debug(
            f"picked {picked} over baseline {baseline} for {kind} slot={slot_id} shape=({m},{k},{n}), "
            f"baseline_ms={baseline_ms:.4f}, tuned_ms={best_ms:.4f}, speedup={baseline_ms / best_ms:.3f}x"
        )
    else:
        _autotune_debug(
            f"kept baseline {baseline} for {kind} slot={slot_id} shape=({m},{k},{n}), "
            f"baseline_ms={baseline_ms:.4f}, best_cfg={best_cfg}, best_ms={best_ms:.4f}"
        )
    return picked


def _select_triton_int8_config(
    m: int,
    k: int,
    n: int,
    *,
    device: Optional[torch.device] = None,
    kernel_kind: str = "fused",
) -> tuple[int, int, int, int, int]:
    baseline = _select_static_triton_int8_config(m, k, n)
    convrot = kernel_kind in _CONVROT_KINDS
    if convrot:
        baseline = (*baseline[:2], 64, *baseline[3:])
    if torch.compiler.is_compiling():
        return _LOW_SHARED_MEMORY_CONFIG if baseline == _HIGH_SHARED_MEMORY_CONFIG else baseline
    if not convrot and (m, k, n) in _TRITON_TINY_M_RUNTIME_CONFIGS:
        return baseline
    if not is_available() or not torch.cuda.is_available():
        return baseline
    try:
        device_index = _device_index(device)
    except Exception:
        return baseline
    if baseline == _HIGH_SHARED_MEMORY_CONFIG:
        props = torch.cuda.get_device_properties(device_index)
        limit = int(getattr(props, "shared_memory_per_block_optin", getattr(props, "shared_memory_per_block", 0)) or 0)
        if limit > 0 and limit < _HIGH_SHARED_MEMORY_CONFIG_BYTES:
            baseline = _LOW_SHARED_MEMORY_CONFIG
    slot_id, rep_shapes = _resolve_autotune_slot(m, k, n)
    if convrot:
        slot_id = f"shape_{m}_{k}_{n}"
        rep_shapes = ((m, k, min(n, _RUNTIME_PROBE_MAX_N)),)
    session_key = (device_index, kernel_kind, slot_id)
    cached = _AUTOTUNE_SESSION_CACHE.get(session_key)
    if cached is not None:
        if _config_compatible_with_baseline(kernel_kind, baseline, cached):
            return cached
        _AUTOTUNE_SESSION_CACHE.pop(session_key, None)
    if convrot and _autotune_is_blocked():
        raise RuntimeError("ConvRot INT8 requires eager warmup before CUDA graph capture")
    cached_cfg = _get_cached_config(device_index, kernel_kind, slot_id, m, k, n)
    if cached_cfg is not None and _config_compatible_with_baseline(kernel_kind, baseline, cached_cfg):
        if not _autotune_is_blocked():
            compile_safe, _ = _ensure_compile_compatible_config(
                kernel_kind,
                device_index,
                slot_id,
                cached_cfg,
                baseline,
                m,
                k,
                n,
                rep_shapes,
            )
            if compile_safe != cached_cfg:
                _set_cached_config(device_index, kernel_kind, slot_id, compile_safe)
            _AUTOTUNE_SESSION_CACHE[session_key] = compile_safe
            return compile_safe
        _AUTOTUNE_SESSION_CACHE[session_key] = cached_cfg
        return cached_cfg
    if _autotune_is_blocked():
        # During graph capture we must avoid autotune/probing allocations. Do not populate
        # session cache with the baseline fallback, so a later non-capture call can autotune.
        return baseline

    if convrot:
        # The ordinary INT8 baseline may not fit ConvRot. Validate the actual
        # kernel before using its output as the reference for candidate timing.
        baseline, _ = _ensure_compile_compatible_config(kernel_kind, device_index, slot_id, baseline, baseline, m, k, n, rep_shapes)

    autotune_enabled = _env_flag(_ENV_AUTOTUNE_ENABLE, "1")
    max_m = _env_int(_ENV_AUTOTUNE_MAX_M, -1)
    if autotune_enabled and not (max_m >= 0 and m > max_m):
        preferred = _autotune_config(kernel_kind, device_index, m, k, n, baseline, slot_id, rep_shapes)
    else:
        preferred = baseline

    compile_safe, compile_err = _ensure_compile_compatible_config(
        kernel_kind,
        device_index,
        slot_id,
        preferred,
        baseline,
        m,
        k,
        n,
        rep_shapes,
    )

    picked = compile_safe
    if compile_safe != preferred:
        _set_cached_config(device_index, kernel_kind, slot_id, compile_safe)
    elif compile_err is not None:
        _autotune_debug(
            f"compile probe could not find an alternative for {kernel_kind} slot={slot_id} "
            f"shape=({m},{k},{n}); will keep {preferred}"
        )
    _AUTOTUNE_SESSION_CACHE[session_key] = picked
    return picked


atexit.register(_save_autotune_cache)


if _TRITON_AVAILABLE:

    @triton.jit
    def _fused_dynamic_int8_gemm_kernel(
        a_ptr,
        b_ptr,
        s_ptr,
        c_ptr,
        m,
        n,
        k,
        stride_am,
        stride_ak,
        stride_bn,
        stride_bk,
        stride_cm,
        stride_cn,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = (pid_m * block_m + tl.arange(0, block_m)).to(tl.int64)  # 64 bits: rows x width passes 2^31 on long high resolution videos
        offs_n = pid_n * block_n + tl.arange(0, block_n)
        offs_k = tl.arange(0, block_k)

        # Pass 1: rowwise absmax for dynamic symmetric int8 activation quantization.
        row_amax = tl.zeros((block_m,), dtype=tl.float32)
        for k0 in range(0, k, block_k):
            kk = k0 + offs_k
            a = tl.load(
                a_ptr + offs_m[:, None] * stride_am + kk[None, :] * stride_ak,
                mask=(offs_m[:, None] < m) & (kk[None, :] < k),
                other=0,
            ).to(tl.float32)
            row_amax = tl.maximum(row_amax, tl.max(tl.abs(a), axis=1))

        row_scale = row_amax / 127.0
        row_scale = tl.where(row_scale > 0.0, row_scale, 1.0)
        row_inv_scale = 1.0 / row_scale

        # Pass 2: quantize activations on the fly + int8 dot.
        acc = tl.zeros((block_m, block_n), dtype=tl.int32)
        for k0 in range(0, k, block_k):
            kk = k0 + offs_k
            a = tl.load(
                a_ptr + offs_m[:, None] * stride_am + kk[None, :] * stride_ak,
                mask=(offs_m[:, None] < m) & (kk[None, :] < k),
                other=0,
            ).to(tl.float32)
            a = a * row_inv_scale[:, None]
            # Match torch.round behavior (ties-to-even) used by quanto::quantize_symmetric.
            a = tl_libdevice.nearbyint(a)
            a = tl.maximum(tl.minimum(a, 127.0), -128.0).to(tl.int8)

            # Weight is [N, K]; load as [K, N] tile for dot.
            b = tl.load(
                b_ptr + offs_n[None, :] * stride_bn + kk[:, None] * stride_bk,
                mask=(offs_n[None, :] < n) & (kk[:, None] < k),
                other=0,
            ).to(tl.int8)
            acc += tl.dot(a, b)

        scales = tl.load(s_ptr + offs_n, mask=offs_n < n, other=0).to(tl.float32)
        out = acc.to(tl.float32) * row_scale[:, None] * scales[None, :]
        tl.store(
            c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
            out,
            mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
        )

    @triton.jit
    def _fused_dynamic_int8_blockscale_gemm_kernel(
        a_ptr,
        b_ptr,
        s_ptr,
        c_ptr,
        m,
        n,
        k,
        stride_am,
        stride_ak,
        stride_bn,
        stride_bk,
        stride_cm,
        stride_cn,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = (pid_m * block_m + tl.arange(0, block_m)).to(tl.int64)  # 64 bits: rows x width passes 2^31 on long high resolution videos
        offs_n = pid_n * block_n + tl.arange(0, block_n)
        offs_k = tl.arange(0, block_k)

        acc = tl.zeros((block_m, block_n), dtype=tl.float32)
        for k0 in range(0, k, block_k):
            kk = k0 + offs_k
            a = tl.load(
                a_ptr + offs_m[:, None] * stride_am + kk[None, :] * stride_ak,
                mask=(offs_m[:, None] < m) & (kk[None, :] < k),
                other=0,
            ).to(tl.float32)
            row_amax = tl.max(tl.abs(a), axis=1)
            row_scale = row_amax / 127.0
            row_scale = tl.where(row_scale > 0.0, row_scale, 1.0)
            a = a / row_scale[:, None]
            a = tl_libdevice.nearbyint(a)
            a = tl.maximum(tl.minimum(a, 127.0), -128.0).to(tl.int8)

            b = tl.load(
                b_ptr + offs_n[None, :] * stride_bn + kk[:, None] * stride_bk,
                mask=(offs_n[None, :] < n) & (kk[:, None] < k),
                other=0,
            ).to(tl.int8)

            dot_i32 = tl.dot(a, b)
            acc += dot_i32.to(tl.float32) * row_scale[:, None]

        scales = tl.load(s_ptr + offs_n, mask=offs_n < n, other=0).to(tl.float32)
        out = acc * scales[None, :]
        tl.store(
            c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
            out,
            mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
        )

    @triton.jit
    def _scaled_int8_gemm_kernel(
        a_ptr,
        b_ptr,
        a_scales_ptr,
        b_scales_ptr,
        c_ptr,
        m,
        n,
        k,
        stride_am,
        stride_ak,
        stride_bn,
        stride_bk,
        stride_cm,
        stride_cn,
        block_m: tl.constexpr,
        block_n: tl.constexpr,
        block_k: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = (pid_m * block_m + tl.arange(0, block_m)).to(tl.int64)  # 64 bits: rows x width passes 2^31 on long high resolution videos
        offs_n = pid_n * block_n + tl.arange(0, block_n)
        offs_k = tl.arange(0, block_k)

        acc = tl.zeros((block_m, block_n), dtype=tl.int32)
        for k0 in range(0, k, block_k):
            kk = k0 + offs_k
            a = tl.load(
                a_ptr + offs_m[:, None] * stride_am + kk[None, :] * stride_ak,
                mask=(offs_m[:, None] < m) & (kk[None, :] < k),
                other=0,
            ).to(tl.int8)
            # Weight is [N, K]; load as [K, N] tile for dot.
            b = tl.load(
                b_ptr + offs_n[None, :] * stride_bn + kk[:, None] * stride_bk,
                mask=(offs_n[None, :] < n) & (kk[:, None] < k),
                other=0,
            ).to(tl.int8)
            acc += tl.dot(a, b)

        a_scales = tl.load(a_scales_ptr + offs_m, mask=offs_m < m, other=1).to(tl.float32)
        b_scales = tl.load(b_scales_ptr + offs_n, mask=offs_n < n, other=1).to(tl.float32)
        out = acc.to(tl.float32) * a_scales[:, None] * b_scales[None, :]
        tl.store(
            c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
            out,
            mask=(offs_m[:, None] < m) & (offs_n[None, :] < n),
        )


if _TRITON_AVAILABLE:
    @triton.jit
    def _prepare_blockscale_kernel(x, q, scales, m, k, blocks: tl.constexpr, stride_m, stride_k, bm: tl.constexpr, bk: tl.constexpr):
        rows = (tl.program_id(0) * bm + tl.arange(0, bm)).to(tl.int64)
        block = tl.program_id(1)
        cols = block * bk + tl.arange(0, bk)
        values = tl.load(x + rows[:, None] * stride_m + cols[None, :] * stride_k,
                         (rows[:, None] < m) & (cols[None, :] < k), 0).to(tl.float32)
        scale = tl.max(tl.abs(values), 1) / 127.0
        scale = tl.where(scale > 0.0, scale, 1.0)
        values = tl_libdevice.nearbyint(values / scale[:, None])
        values = tl.maximum(tl.minimum(values, 127.0), -128.0).to(tl.int8)
        tl.store(q + rows[:, None] * k + cols[None, :], values, (rows[:, None] < m) & (cols[None, :] < k))
        tl.store(scales + rows * blocks + block, scale, rows < m)

    @triton.jit
    def _prepared_blockscale_gemm_kernel(q, b, scales, ws, out, m, n, k, blocks: tl.constexpr, stride_bn, stride_bk,
                                          bm: tl.constexpr, bn: tl.constexpr, bk: tl.constexpr):
        rows = (tl.program_id(0) * bm + tl.arange(0, bm)).to(tl.int64)
        cols = tl.program_id(1) * bn + tl.arange(0, bn)
        offsets = tl.arange(0, bk)
        acc = tl.zeros((bm, bn), tl.float32)
        for block in range(blocks):
            kk = block * bk + offsets
            a = tl.load(q + rows[:, None] * k + kk[None, :], (rows[:, None] < m) & (kk[None, :] < k), 0)
            w = tl.load(b + cols[None, :] * stride_bn + kk[:, None] * stride_bk, (cols[None, :] < n) & (kk[:, None] < k), 0)
            scale = tl.load(scales + rows * blocks + block, rows < m, 1)
            acc += tl.dot(a, w).to(tl.float32) * scale[:, None]
        weight_scale = tl.load(ws + cols, cols < n, 0).to(tl.float32)
        tl.store(out + rows[:, None] * n + cols[None, :], acc * weight_scale[None, :], (rows[:, None] < m) & (cols[None, :] < n))


def prepare_blockscale_input(x, out_features):
    """Keep the ordinary fused kernel's K blocks, rounding and launch configuration."""
    m, k = x.shape
    config = _select_triton_int8_config(m, k, out_features, device=x.device, kernel_kind="fused")
    bm, _, bk, warps, _ = config
    blocks = triton.cdiv(k, bk)
    q = torch.empty((m, k), dtype=torch.int8, device=x.device)
    scales = torch.empty((m, blocks), dtype=torch.float32, device=x.device)
    _prepare_blockscale_kernel[(triton.cdiv(m, bm), blocks)](x, q, scales, m, k, blocks, *x.stride(), bm, bk, num_warps=warps)
    return q, scales, x.dtype, config


def prepared_blockscale_mm(prepared, weight, scale):
    q, scales, dtype, (bm, bn, bk, warps, stages) = prepared
    m, k = q.shape
    n = weight.shape[0]
    out = torch.empty((m, n), dtype=dtype, device=q.device)
    _prepared_blockscale_gemm_kernel[(triton.cdiv(m, bm), triton.cdiv(n, bn))](
        q, weight, scales, scale, out, m, n, k, scales.shape[1], *weight.stride(), bm, bn, bk, num_warps=warps, num_stages=stages)
    return out


def _flatten_scale(scale: torch.Tensor) -> torch.Tensor:
    if scale.ndim == 2 and scale.shape[1] == 1:
        return scale.view(-1)
    if scale.ndim == 1:
        return scale
    return scale.reshape(-1)


def _expand_or_validate_scale(scale: torch.Tensor, expected: int) -> torch.Tensor:
    scale = _flatten_scale(scale)
    if scale.numel() == 1:
        return scale.reshape(1).expand(expected)
    if scale.numel() != expected:
        raise RuntimeError(f"Scale length mismatch: expected {expected}, got {scale.numel()}")
    return scale


def _fused_quant_scaled_mm_common(
    x2d: torch.Tensor,
    qweight: torch.Tensor,
    b_scale: torch.Tensor,
    *,
    k: int,
    n: int,
    stride_bn: int,
    stride_bk: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    m = x2d.shape[0]
    out = torch.empty((m, n), device=x2d.device, dtype=out_dtype)
    x_mm_c = x2d if x2d.is_contiguous() else x2d.contiguous()
    b_scale_c = b_scale if b_scale.is_contiguous() else b_scale.contiguous()

    block_m, block_n, block_k, num_warps, num_stages = _select_triton_int8_config(m, k, n, device=x2d.device, kernel_kind="fused")
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _fused_dynamic_int8_blockscale_gemm_kernel[grid](
        x_mm_c,
        qweight,
        b_scale_c,
        out,
        m,
        n,
        k,
        x_mm_c.stride(0),
        x_mm_c.stride(1),
        stride_bn,
        stride_bk,
        out.stride(0),
        out.stride(1),
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


def fused_quant_scaled_mm(
    x2d: torch.Tensor,
    qweight: torch.Tensor,
    qweight_scale: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if not is_available():
        raise RuntimeError("Triton backend not available")
    if x2d.ndim != 2:
        raise RuntimeError("x2d must be 2D")
    if qweight.ndim != 2:
        raise RuntimeError("qweight must be 2D [N, K]")
    if x2d.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise RuntimeError("x2d must be bf16/fp16/fp32")
    if qweight.dtype != torch.int8:
        raise RuntimeError("qweight must be int8")
    if not x2d.is_cuda or not qweight.is_cuda:
        raise RuntimeError("fused_quant_scaled_mm requires CUDA tensors")

    m, k = x2d.shape
    n, k2 = qweight.shape
    if k != k2:
        raise RuntimeError(f"Triton int8 GEMM shape mismatch: x={x2d.shape}, w={qweight.shape}")

    b_scale = _expand_or_validate_scale(qweight_scale, n)
    if b_scale.device != x2d.device or b_scale.dtype != torch.float32:
        b_scale = b_scale.to(device=x2d.device, dtype=torch.float32)
    elif not b_scale.is_contiguous():
        b_scale = b_scale.contiguous()
    if x2d.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise RuntimeError(f"Unsupported activation dtype for fused path: {x2d.dtype}")

    out_dtype = out_dtype or x2d.dtype
    qweight_c = qweight if qweight.is_contiguous() else qweight.contiguous()
    return _fused_quant_scaled_mm_common(
        x2d,
        qweight_c,
        b_scale,
        k=k,
        n=n,
        stride_bn=qweight_c.stride(0),
        stride_bk=qweight_c.stride(1),
        out_dtype=out_dtype,
    )


def fused_quant_scaled_mm_transposed(
    x2d: torch.Tensor,
    qweight_t: torch.Tensor,
    qweight_scale: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if not is_available():
        raise RuntimeError("Triton backend not available")
    if x2d.ndim != 2:
        raise RuntimeError("x2d must be 2D")
    if qweight_t.ndim != 2:
        raise RuntimeError("qweight_t must be 2D [K, N]")
    if x2d.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise RuntimeError("x2d must be bf16/fp16/fp32")
    if qweight_t.dtype != torch.int8:
        raise RuntimeError("qweight_t must be int8")
    if not x2d.is_cuda or not qweight_t.is_cuda:
        raise RuntimeError("fused_quant_scaled_mm_transposed requires CUDA tensors")

    m, k = x2d.shape
    k2, n = qweight_t.shape
    if k != k2:
        raise RuntimeError(f"Triton int8 GEMM shape mismatch: x={x2d.shape}, w_t={qweight_t.shape}")

    b_scale = _expand_or_validate_scale(qweight_scale, n)
    if b_scale.device != x2d.device or b_scale.dtype != torch.float32:
        b_scale = b_scale.to(device=x2d.device, dtype=torch.float32)
    elif not b_scale.is_contiguous():
        b_scale = b_scale.contiguous()
    if x2d.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise RuntimeError(f"Unsupported activation dtype for fused path: {x2d.dtype}")

    out_dtype = out_dtype or x2d.dtype
    qweight_t_c = qweight_t if qweight_t.is_contiguous() else qweight_t.contiguous()
    return _fused_quant_scaled_mm_common(
        x2d,
        qweight_t_c,
        b_scale,
        k=k,
        n=n,
        stride_bn=qweight_t_c.stride(1),
        stride_bk=qweight_t_c.stride(0),
        out_dtype=out_dtype,
    )


def scaled_int8_mm(
    a_int8: torch.Tensor,
    b_int8: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out_dtype: Optional[torch.dtype] = None,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if not is_available():
        raise RuntimeError("Triton backend not available")
    if a_int8.ndim != 2:
        raise RuntimeError("a_int8 must be 2D")
    if b_int8.ndim != 2:
        raise RuntimeError("b_int8 must be 2D [N, K]")
    if a_int8.dtype != torch.int8 or b_int8.dtype != torch.int8:
        raise RuntimeError("scaled_int8_mm requires int8 activations and int8 weights")
    if not a_int8.is_cuda or not b_int8.is_cuda:
        raise RuntimeError("scaled_int8_mm requires CUDA tensors")

    m, k = a_int8.shape
    n, k2 = b_int8.shape
    if k != k2:
        raise RuntimeError(f"Triton int8 GEMM shape mismatch: a={a_int8.shape}, w={b_int8.shape}")

    a_scale = _expand_or_validate_scale(a_scale, m)
    b_scale = _expand_or_validate_scale(b_scale, n)
    if a_scale.device != a_int8.device or a_scale.dtype != torch.float32:
        a_scale = a_scale.to(device=a_int8.device, dtype=torch.float32)
    elif not a_scale.is_contiguous():
        a_scale = a_scale.contiguous()
    if b_scale.device != a_int8.device or b_scale.dtype != torch.float32:
        b_scale = b_scale.to(device=a_int8.device, dtype=torch.float32)
    elif not b_scale.is_contiguous():
        b_scale = b_scale.contiguous()

    out_dtype = out_dtype or torch.bfloat16
    if out is None:
        out = torch.empty((m, n), device=a_int8.device, dtype=out_dtype)
    elif out.shape != (m, n) or out.device != a_int8.device or out.dtype != out_dtype:
        raise RuntimeError(f"Invalid scaled_int8_mm output tensor: expected {(m, n)} {out_dtype} on {a_int8.device}, got {tuple(out.shape)} {out.dtype} on {out.device}")
    a_int8_c = a_int8 if a_int8.is_contiguous() else a_int8.contiguous()
    b_int8_c = b_int8 if b_int8.is_contiguous() else b_int8.contiguous()
    a_scale_c = a_scale if a_scale.is_contiguous() else a_scale.contiguous()
    b_scale_c = b_scale if b_scale.is_contiguous() else b_scale.contiguous()

    block_m, block_n, block_k, num_warps, num_stages = _select_triton_int8_config(m, k, n, device=a_int8.device, kernel_kind="scaled")
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_n))
    _scaled_int8_gemm_kernel[grid](
        a_int8_c,
        b_int8_c,
        a_scale_c,
        b_scale_c,
        out,
        m,
        n,
        k,
        a_int8_c.stride(0),
        a_int8_c.stride(1),
        b_int8_c.stride(0),
        b_int8_c.stride(1),
        out.stride(0),
        out.stride(1),
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out


from shared.kernels.triton_compilation_log import install_triton_compilation_logger, log_compilation_tuning
install_triton_compilation_logger()
_autotune_config = log_compilation_tuning(_autotune_config)
