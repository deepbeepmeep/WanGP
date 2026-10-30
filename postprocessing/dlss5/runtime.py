from __future__ import annotations

from dataclasses import dataclass
import collections
import argparse
import json
import math
import os
import platform
import re
import struct
import subprocess
import sys
import threading
import time
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
from mmgp import offload
from shared.utils import offload_registry
from shared.utils.media_control import controlled_model_loading, loading_callback, checkpoint_modules


RUNTIME = Path(__file__).resolve().parents[2] / "dlss5"
HOST_DIR = RUNTIME / "host"
DLSS_DIR = RUNTIME / "dlss"
DLSSG_DIR = RUNTIME / "dlssg"
DLSSG_NATIVE_DIR = RUNTIME / "dlssg-native"
USE_DEPTH_GUIDE = True
DEPTH_MODEL_VARIANT = "vitl"
LEGACY_NR_WORKER = HOST_DIR / "nvngx.dll"
DEPTH_NR_WORKER = HOST_DIR / "nr-depth-worker.exe"
NR_WORKER = DEPTH_NR_WORKER if USE_DEPTH_GUIDE else LEGACY_NR_WORKER
DLSSG_LINUX = sys.platform == "linux"
NR_WSL = DLSSG_LINUX and "microsoft" in platform.release().lower()
DLSSG_MULTIPLIERS = (2.0, 3.0, 4.0, 5.0, 6.0)
DLSSG_LINUX_RUNTIME = "libnvidia-ngx-dlssg.so.310.9.16"
DLSSG_LINUX_RUNTIME_DEV = "libnvidia-ngx-dlssg.so.310.7.0"
DLSSG_NATIVE_METHOD = "dlssg_native"
DLSSG_NATIVE_NAME = "DLSS FG Linux"


@dataclass(frozen=True)
class DlssgVariant:
    """One Frame Generation worker plus the NVIDIA runtime installed next to it.

    The Linux worker resolves its own directory through /proc/self/exe and
    loads the NGX runtime from there (scripts/dlssg_worker_linux), so every
    variant keeps its own directory: dlss5/dlssg is the upstream layout (x2
    only on native Linux), dlss5/dlssg-native the multi-multiplier worker from
    scripts/build_dlssg_worker_linux.sh. Both variants are offered in the UI at
    the same time and the multiplier range shown next to each name is the one
    the installed worker and GPU report, so the choice is made per generation
    with no restart and no environment variable.
    """

    method: str
    name: str
    dir: Path
    worker: Path
    runtime: Path
    native_linux: bool = False

    @property
    def files(self) -> tuple[Path, Path]:
        return (self.runtime, self.worker)

    @property
    def label_path(self) -> str:
        try:
            return str(self.worker.relative_to(RUNTIME))
        except ValueError:
            return str(self.worker)


DLSSG_UPSTREAM = DlssgVariant(
    method="dlssg",
    name="DLSS Frame Generation",
    dir=DLSSG_DIR,
    worker=DLSSG_DIR / ("dlssg-worker" if DLSSG_LINUX else "dlssg-worker.exe"),
    runtime=DLSSG_DIR / ("nvngx_dlssg.dll" if os.name == "nt" else DLSSG_LINUX_RUNTIME),
)
# Aliases for the Neural Rendering path and the spatial upsampler, which always
# run on the upstream worker.
DLSSG_WORKER = DLSSG_UPSTREAM.worker
DLSSG_RUNTIME = DLSSG_UPSTREAM.runtime

NR_FILES = (
    HOST_DIR / "dxgi.dll",
    HOST_DIR / "renodx-dlss5.addon64",
    HOST_DIR / "nvngx_dlssnr.dll",
    DLSS_DIR / "nvngx_dlss.dll",
    NR_WORKER,
)
DLSSG_FILES = (DLSSG_RUNTIME, DLSSG_WORKER)

NR_MODES = {
    1.0: ("DLAA", 5),
    1.5: ("Quality", 2),
    1.724: ("Balanced", 1),
    2.0: ("Performance", 0),
    3.0: ("Ultra Performance", 3),
}

DEPTH_RESOLUTION_DIVISORS = {"full": 1, "half": 2, "quarter": 4}
MOTION_VECTOR_METHODS = ("original", "raft")
RAFT_ITERATIONS = 20
_RAFT_MODELS = {}
_DEPTH_MODELS = {}
_GUIDE_OFFLOADS = {}
_POSTPROCESSOR_CONFIG = {}


def configure_depth_estimator(server_config):
    global DEPTH_MODEL_VARIANT, _POSTPROCESSOR_CONFIG
    DEPTH_MODEL_VARIANT = server_config.get("depth_anything_v2_variant", "vitl")
    _POSTPROCESSOR_CONFIG = server_config


def _offload_guide_model(name, model):
    from postprocessing.memory import configure

    kwargs = {}
    profile = configure(kwargs, _POSTPROCESSOR_CONFIG)
    offloadobj = offload.profile({"model": model}, loading_callback=loading_callback(), profile_no=profile, quantizeTransformer=False, convertWeightsFloatTo=None, pinnedMemory=False, verboseLevel=-1, **kwargs)
    _GUIDE_OFFLOADS[name] = offloadobj
    offload_registry.register_offloadobj(name, offloadobj, release_flow_model)


@controlled_model_loading
def _raft_model(device: torch.device):
    key = str(device)
    if key not in _RAFT_MODELS:
        from preprocessing.raft.raft import RAFT
        from shared.utils import files_locator as fl

        model = RAFT(argparse.Namespace(small=False, mixed_precision=False, alternate_corr=False))
        weights = torch.load(fl.locate_file("flow/raft-things.pth"), map_location="cpu", weights_only=True)
        model.load_state_dict({name.removeprefix("module."): value for name, value in weights.items()})
        _RAFT_MODELS[key] = model.eval()
        if device.type == "cuda":
            _offload_guide_model("DLSS Motion Vectors", model)
    return _RAFT_MODELS[key]


def release_flow_model():
    for name, offloadobj in _GUIDE_OFFLOADS.items():
        offload_registry.unregister_offloadobj(name, offloadobj)
        offloadobj.release()
    _GUIDE_OFFLOADS.clear()
    for annotator in _DEPTH_MODELS.values():
        if hasattr(annotator, "close"):
            annotator.close()
        else:
            annotator.model.to("cpu")
    _DEPTH_MODELS.clear()
    _RAFT_MODELS.clear()


@controlled_model_loading
def _depth_model(device: torch.device):
    variant = DEPTH_MODEL_VARIANT
    key = (variant, str(device))
    if key not in _DEPTH_MODELS:
        from shared.utils import files_locator as fl

        if variant == "da3_metric_large":
            from preprocessing.depth_anything_v3.depth import DepthV3VideoAnnotator

            _DEPTH_MODELS[key] = DepthV3VideoAnnotator({"PRETRAINED_MODEL": fl.locate_file("depth/depth_anything_v3_metric_large_bf16.safetensors"), "MODEL_NAME": "da3metric-large", "PROCESS_RES": 0, "CHUNK_SIZE": 1, "CHUNK_OVERLAP": 8}, device=torch.device("cpu"))
        else:
            from preprocessing.depth_anything_v2.depth import DepthV2Annotator

            filename = f"depth/depth_anything_v2_{variant}.pth"
            _DEPTH_MODELS[key] = DepthV2Annotator({"PRETRAINED_MODEL": fl.locate_file(filename), "MODEL_VARIANT": variant}, device=torch.device("cpu"))
        if device.type == "cuda":
            _offload_guide_model(f"DLSS Depth ({variant})", _DEPTH_MODELS[key].model)
    return _DEPTH_MODELS[key]


def _missing(files: tuple[Path, ...]) -> list[Path]:
    return [path for path in files if not path.is_file()]


_DLSSG_PROBES: dict[tuple[str, int], dict] = {}
_DLSSG_PROBE_GPUS: dict[str, tuple[tuple[int, str], ...]] = {}
_DLSSG_GPU_PINS: dict[str, int | None] = {}


def _run_probe(variant: DlssgVariant, gpu_index: int) -> dict:
    """One --probe run, remembered per GPU it was pinned to.

    Unpinned, the worker prints "Device <i>: <name> ..." for every Vulkan GPU it
    sees before picking one. That listing is what lets WanGP pin the worker to
    the GPU WanGP runs on, because the Vulkan ICD ignores CUDA_VISIBLE_DEVICES.
    """
    key = (variant.method, gpu_index)
    if key in _DLSSG_PROBES:
        return _DLSSG_PROBES[key]
    capabilities: dict = {}
    devices: tuple[tuple[int, str], ...] = ()
    if not _missing(variant.files) and (os.name == "nt" or DLSSG_LINUX):
        command = [str(variant.worker), "--probe"] + (["--gpu", str(gpu_index)] if gpu_index >= 0 else [])
        try:
            result = subprocess.run(command, cwd=str(variant.dir), capture_output=True, text=True, timeout=15, check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            parsed = json.loads(result.stdout.strip())
            capabilities = parsed if isinstance(parsed, dict) and "available" in parsed else {}
            if gpu_index < 0:
                devices = tuple((int(index), name) for index, name in re.findall(r"^Device (\d+): (.+?) \(", result.stderr, re.MULTILINE))
        except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
            capabilities = {}
    _DLSSG_PROBE_GPUS[variant.method] = devices or _DLSSG_PROBE_GPUS.get(variant.method, ())
    _DLSSG_PROBES[key] = capabilities
    return capabilities


def _probe_gpu_name(capabilities: dict) -> str:
    """The GPU the probe used, read out of its own report."""
    gpu = str(capabilities.get("gpu", "") or "").strip()
    if gpu:
        return gpu
    match = re.search(r"completed on (.+?)\.?$", str(capabilities.get("detail", "") or ""))
    return match.group(1).strip() if match else ""


def _pin_worker_gpu(variant: DlssgVariant, report: dict) -> int | None:
    """The --gpu index putting the worker on WanGP's GPU, when there is one to pin.

    None when the worker already chose that GPU, when WanGP is not restricted to
    a GPU nvidia-smi can name, or when the worker's listing does not contain it.
    """
    if variant.method in _DLSSG_GPU_PINS:
        return _DLSSG_GPU_PINS[variant.method]
    wanted = _selected_gpu_names()[:1]
    used = _probe_gpu_name(report)
    pin = None
    if wanted and used and used != wanted[0]:
        for index, name in _DLSSG_PROBE_GPUS.get(variant.method, ()):
            if name == wanted[0]:
                pin = index
                break
    _DLSSG_GPU_PINS[variant.method] = pin
    return pin


def dlssg_capabilities(variant: DlssgVariant = DLSSG_UPSTREAM) -> dict:
    """What the worker reports on the GPU WanGP runs on, not on the one it picks."""
    report = _run_probe(variant, -1)
    if not report:
        return {}
    pin = _pin_worker_gpu(variant, report)
    return (_run_probe(variant, pin) or report) if pin is not None else report


def dlssg_gpu_args(variant: DlssgVariant = DLSSG_UPSTREAM) -> list[str]:
    """The worker arguments pinning it to the GPU WanGP runs on."""
    pin = _pin_worker_gpu(variant, _run_probe(variant, -1))
    return [] if pin is None else ["--gpu", str(pin)]


def dlssg_gpu_note(variant: DlssgVariant = DLSSG_UPSTREAM) -> str:
    """The GPU Frame Generation runs on, for the log line; empty when unknown."""
    pin = _pin_worker_gpu(variant, _run_probe(variant, -1))
    if pin is None:
        return _probe_gpu_name(dlssg_capabilities(variant))
    for index, name in _DLSSG_PROBE_GPUS.get(variant.method, ()):
        if index == pin:
            return f"{name} (GPU {index})"
    return ""


def _worker_version(variant: DlssgVariant) -> int:
    try:
        return int(dlssg_capabilities(variant).get("worker_version", 0) or 0)
    except (TypeError, ValueError):
        return 0


def _linux_runtime_version(name: str) -> tuple[int, ...]:
    return tuple(int(part) for part in name.removeprefix("libnvidia-ngx-dlssg.so.").split(".") if part.isdigit())


def _newest_linux_runtime(directory: Path) -> Path:
    # The worker loads the runtime sitting next to it, so the glob picks the
    # newest version in the worker's own directory and falls back to the file
    # scripts/build_dlssg_worker_linux.sh links in, for a clear "missing" report.
    installed = [path for path in directory.glob("libnvidia-ngx-dlssg.so.*") if path.is_file()]
    if not installed:
        return directory / DLSSG_LINUX_RUNTIME_DEV
    return max(installed, key=lambda path: _linux_runtime_version(path.name))


def _native_variant() -> DlssgVariant | None:
    if not DLSSG_LINUX:
        return None
    worker = DLSSG_NATIVE_DIR / "dlssg-worker"
    directory = DLSSG_NATIVE_DIR
    if not worker.is_file():
        # No dedicated directory: dlss5/dlssg may hold a native worker built by
        # scripts/build_dlssg_worker_linux.sh. The upstream Linux release is an
        # x2-only worker, so sharing its directory requires a v2+ (multi-frame)
        # worker; the probe reports that, and its result is cached.
        worker, directory = DLSSG_DIR / "dlssg-worker", DLSSG_DIR
        if not worker.is_file():
            return None
        candidate = DlssgVariant(DLSSG_NATIVE_METHOD, DLSSG_NATIVE_NAME, directory, worker, _newest_linux_runtime(directory), native_linux=True)
        return candidate if _worker_version(candidate) >= 2 else None
    return DlssgVariant(DLSSG_NATIVE_METHOD, DLSSG_NATIVE_NAME, directory, worker, _newest_linux_runtime(directory), native_linux=True)


@lru_cache(maxsize=1)
def dlssg_variants() -> tuple[DlssgVariant, ...]:
    """Every Frame Generation worker installed on this host, in UI order."""
    native = _native_variant()
    return (DLSSG_UPSTREAM,) + ((native,) if native is not None else ())


def _visible_cuda_devices() -> list[str] | None:
    """The GPUs WanGP may use, as CUDA_VISIBLE_DEVICES lists them (indices or UUIDs).

    None when the variable is unset, meaning every GPU of the host is fair game.
    WanGP pins a GPU with --default-device-id, which ends up here, so a multi-GPU
    host has one GPU to tier Frame Generation on.
    """
    raw = str(os.environ.get("CUDA_VISIBLE_DEVICES", "")).strip()
    if not raw:
        return None
    return [token.strip() for token in raw.split(",") if token.strip()]


def _nvidia_smi_gpus(nvidia_smi: str) -> tuple[tuple[str, str, str], ...]:
    """(index, uuid, name) of every GPU nvidia-smi lists; () when it cannot answer."""
    try:
        result = subprocess.run([nvidia_smi, "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5, check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.TimeoutExpired):
        return ()
    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) == 3 and parts[0].isdigit():
            rows.append((parts[0], parts[1], parts[2]))
    return tuple(rows)


def _selected_gpu_names() -> list[str]:
    """The GPUs WanGP can run on, in CUDA_VISIBLE_DEVICES order; the first is the one it uses.

    Empty when the host cannot be queried, which leaves the tiering at 0.
    """
    rows = _nvidia_smi_gpus("/usr/lib/wsl/lib/nvidia-smi" if NR_WSL else "nvidia-smi")
    if not rows:
        return []
    names = [name for _index, _uuid, name in rows]
    visible = _visible_cuda_devices()
    if visible is None:
        return names
    by_token = {index: name for index, _uuid, name in rows}
    by_token.update({uuid: name for _index, uuid, name in rows})
    chosen = [by_token[token] for token in visible if token in by_token]
    return chosen or names


def _gpu_tier(name: str) -> int:
    """The RTX series of one GPU name; 50 is what unlocks x5/x6, 0 means unknown."""
    match = re.search(r"GeForce\s+RTX\s+(\d{2})\d{2}", name, re.IGNORECASE)
    if match:
        return int(match.group(1))
    # Workstation / datacenter cards carry no GeForce tag (e.g. RTX PRO
    # 5000 Blackwell, RTX 5000 Ada, RTX A6000, L40S), so the GeForce-only
    # match above reads them as series 0 and they falsely report
    # "RTX 30+/40+ required". Tier them by architecture instead; GeForce
    # behavior is unchanged.
    up = name.upper()
    if any(key in up for key in ("PRO 6000", "PRO 5000", "PRO 4000", "B100", "B200", "GB100", "H100", "H200")):
        return 50
    if "ADA" in up or "L40" in up or " L4" in up:
        return 40
    if "RTX A" in up or any(key in up for key in (" A40", " A30", " A16", " A10", " A80")):
        return 30
    if "QUADRO" in up:
        return 20
    match = re.search(r"RTX\D*?(\d{2})\d{2}", up)
    return int(match.group(1)) if match else 0


@lru_cache(maxsize=None)
def _gpu_series(variant: DlssgVariant = DLSSG_UPSTREAM) -> int:
    # nvidia-smi is available on Windows, Linux and WSL; the query is a no-op on
    # hosts without an NVIDIA GPU (no series found). Upstream tiers only
    # Windows / WSL hosts: the native Linux variant tiers its GPU as well.
    if os.name != "nt" and not NR_WSL and not variant.native_linux:
        return 0
    # Frame Generation runs on the GPU WanGP runs on, so the ceiling is the
    # weakest GPU WanGP may pick: a 5090 + 3090 host pinned to the 3090 tops out
    # at x4, and a host pinning nothing is capped by its oldest card, because
    # WanGP is free to run the job on that one.
    tiers = [tier for tier in (_gpu_tier(name) for name in _selected_gpu_names()) if tier]
    return min(tiers, default=0)


def is_rtx_50_series(variant: DlssgVariant = DLSSG_UPSTREAM) -> bool:
    return _gpu_series(variant) == 50


def hardware_limit(variant: DlssgVariant = DLSSG_UPSTREAM) -> int:
    # Upstream treats native Linux as an x2-only path; the native variant is
    # tiered exactly like the Windows path.
    if DLSSG_LINUX and not variant.native_linux:
        return 2
    return 6 if is_rtx_50_series(variant) else 4


def multiplier_range(variant: DlssgVariant = DLSSG_UPSTREAM) -> tuple[float, float]:
    """The xN..xM range this worker, runtime and GPU currently support."""
    capabilities = dlssg_capabilities(variant)
    worker_version = int(capabilities.get("worker_version", 0) or 0)
    maximum = int(capabilities.get("multi_frame_count_max", 1) or 1)
    cap = min(hardware_limit(variant), maximum + 1, 6 if worker_version >= 2 else 2)
    return 2.0, float(min(6, max(2, cap)))


def multipliers_for(variant: DlssgVariant = DLSSG_UPSTREAM) -> tuple[float, ...]:
    low, high = multiplier_range(variant)
    return tuple(scale for scale in DLSSG_MULTIPLIERS if low <= scale <= high)


def nominal_multiplier_range(variant: DlssgVariant = DLSSG_UPSTREAM) -> tuple[float, float]:
    """The xN..xM ceiling for this GPU; computable without a working probe."""
    return 2.0, float(min(6, max(2, hardware_limit(variant))))


def format_multiplier_range(variant: DlssgVariant = DLSSG_UPSTREAM, nominal: bool = False) -> str:
    low, high = nominal_multiplier_range(variant) if nominal else multiplier_range(variant)
    return f"x{low:g}" if high <= low else f"x{low:g}-x{high:g}"


@lru_cache(maxsize=1)
def _hags_enabled() -> bool | None:
    if os.name != "nt":
        return None
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\GraphicsDrivers") as key:
            value, _kind = winreg.QueryValueEx(key, "HwSchMode")
        return {1: False, 2: True}.get(int(value))
    except (OSError, ValueError):
        return None


def unavailable_reason(*, temporal: bool, variant: DlssgVariant | None = None, short: bool = False) -> str:
    if not temporal and os.name != "nt":
        return "Windows only: ReShade/RenoDX"
    variant = variant or DLSSG_UPSTREAM
    missing = _missing(variant.files if temporal else NR_FILES)
    if missing:
        return f"missing {missing[0].name}"
    if temporal and DLSSG_LINUX:
        capabilities = dlssg_capabilities(variant)
        if not capabilities:
            return f"Linux probe failed, run {variant.label_path} --probe" if short else f"Linux DLSS Frame Generation probe failed (run {variant.label_path} --probe)"
        if capabilities["available"]:
            return ""
        detail = capabilities.get("reason") or capabilities.get("detail") or ""
        if short:
            return (detail.split(";")[0].strip() if detail else "") or "Linux Frame Generation unavailable"
        return detail or "NVIDIA Vulkan Frame Generation unavailable"
    if not temporal and NR_WSL:
        interop = Path("/proc/sys/fs/binfmt_misc/WSLInterop")
        if not interop.is_file() or not interop.read_text().startswith("enabled\n"):
            return "WSL Windows interoperability must be enabled for Neural Rendering"
    elif os.name != "nt":
        return "Windows 11 required"
    series = _gpu_series(variant)
    minimum = 40 if temporal else 30
    if series < minimum:
        return f"RTX {minimum}+ required"
    if temporal:
        hags_enabled = _hags_enabled()
        if hags_enabled is False:
            return "HAGS disabled"
        capabilities = dlssg_capabilities(variant)
        if capabilities and not capabilities["available"]:
            return "DLSS Frame Generation unavailable (check HAGS and NVIDIA driver)"
    return ""


def require_runtime(*, temporal: bool, variant: DlssgVariant | None = None) -> None:
    reason = unavailable_reason(temporal=temporal, variant=variant)
    if reason:
        raise RuntimeError(f"DLSS {'Frame Generation' if temporal else 'Neural Rendering'} is unavailable: {reason}. See docs/DLSS5.md.")


def _read_exact(stream, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        block = stream.read(size - len(data))
        if not block:
            raise RuntimeError(f"DLSS worker closed after {len(data)} of {size} output bytes")
        data.extend(block)
    return bytes(data)


def _read_array(stream, shape: tuple[int, ...], output: np.ndarray | None = None) -> np.ndarray:
    output = np.empty(shape, dtype=np.uint8) if output is None else output
    view = memoryview(output).cast("B")
    offset = 0
    while offset < len(view):
        count = stream.readinto(view[offset:])
        if not count:
            raise RuntimeError(f"DLSS worker closed after {offset} of {len(view)} output bytes")
        offset += count
    return output


class Worker:
    def __init__(self, command: list[str], cwd: Path):
        self.logs: collections.deque[str] = collections.deque(maxlen=100)
        self.process = subprocess.Popen(command, cwd=str(cwd), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self._log_thread = threading.Thread(target=self._read_logs, daemon=True)
        self._log_thread.start()

    def _read_logs(self):
        assert self.process.stderr is not None
        for raw in iter(self.process.stderr.readline, b""):
            self.logs.append(raw.decode("utf-8", "replace").rstrip())

    def error(self) -> str:
        return "\n".join(self.logs)

    def read_exact(self, size: int, operation: str) -> bytes:
        assert self.process.stdout is not None
        try:
            return _read_exact(self.process.stdout, size)
        except (OSError, RuntimeError) as error:
            self.close(abort=True)
            code = self.process.returncode
            code_text = "unknown" if code is None else f"{code} (0x{code & 0xFFFFFFFF:08X})"
            details = self.error()
            if code is not None and (code & 0xFFFFFFFF) == 0xC0000135:
                details = f"Windows could not load a DLL dependency. Install the current DLSS 5 worker bundle.\n{details}".rstrip()
            elif not details:
                details = "The worker produced no diagnostic output. Reinstall the current DLSS 5 worker bundle and verify dlss5/host/ReShade.log."
            raise RuntimeError(f"{operation}: {Path(self.process.args[0]).name} exited before replying (exit code {code_text}).\n{details}") from error

    def close(self, *, abort: bool = False):
        process = self.process
        if process.poll() is None:
            try:
                if abort:
                    process.terminate()
                elif process.stdin and not process.stdin.closed:
                    process.stdin.close()
                process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
                process.wait(timeout=5)
        self._log_thread.join(timeout=1)
        if not abort and process.returncode:
            raise RuntimeError(f"DLSS worker exited with code {process.returncode}.\n{self.error()}")


class FlowGuides:
    def __init__(self, width: int, height: int, motion_vector: str):
        self.width, self.height = width, height
        self.use_raft = motion_vector == "raft"
        scale = min(1.0, 640 / width)
        self.flow_width = max(64, round(width * scale / 2) * 2)
        self.flow_height = max(64, round(height * scale / 2) * 2)
        self.previous = None
        self.zero = np.zeros((height, width, 2), dtype=np.float16)
        if self.use_raft:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.flow = _raft_model(self.device)
        else:
            self.flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            self.flow.setUseSpatialPropagation(True)
            self.flow.setFinestScale(1)

    def _raft(self, current: np.ndarray) -> np.ndarray:
        from preprocessing.raft.utils.utils import InputPadder

        current_tensor = torch.from_numpy(current[..., :3]).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        previous_tensor = torch.from_numpy(self.previous[..., :3]).permute(2, 0, 1).float().unsqueeze(0).to(self.device)
        padder = InputPadder(current_tensor.shape)
        current_tensor, previous_tensor = padder.pad(current_tensor, previous_tensor)
        with torch.inference_mode(), checkpoint_modules(module for module in self.flow.modules() if not module._modules):
            _low, motion = self.flow(current_tensor, previous_tensor, iters=RAFT_ITERATIONS, test_mode=True)
        return padder.unpad(motion)[0].permute(1, 2, 0).cpu().numpy()

    def process(self, rgba: np.ndarray) -> tuple[np.ndarray, bool]:
        resized = cv2.resize(rgba, (self.flow_width, self.flow_height), interpolation=cv2.INTER_AREA)
        current = resized if self.use_raft else cv2.cvtColor(resized, cv2.COLOR_RGBA2GRAY)
        if self.previous is None:
            motion, reset = self.zero, True
        else:
            score = float(np.mean(cv2.absdiff(current, self.previous))) / 255.0
            duplicate = score < 0.0005
            reset = score > 0.24
            if reset or duplicate:
                motion = self.zero
            else:
                low_resolution_motion = self._raft(current) if self.use_raft else self.flow.calc(current, self.previous, None)
                motion = cv2.resize(low_resolution_motion, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                motion[..., 0] *= self.width / self.flow_width
                motion[..., 1] *= self.height / self.flow_height
                finite = np.isfinite(motion).all(axis=2)
                motion[~finite] = 0
                reset = float(np.mean(finite)) < 0.98
                if reset:
                    motion = self.zero
                motion = np.ascontiguousarray(motion, dtype=np.float16)
        self.previous = current
        return motion, reset


class DepthGuides:
    def __init__(self, width: int, height: int, depth_resolution: str):
        self.width, self.height = width, height
        divisor = DEPTH_RESOLUTION_DIVISORS[depth_resolution]
        self.inference_width = width // divisor
        self.inference_height = height // divisor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.variant = DEPTH_MODEL_VARIANT
        self.estimator = _depth_model(self.device)
        self.low = self.high = None

    def process(self, rgba: np.ndarray, reset: bool) -> np.ndarray:
        inference_frame = rgba if (self.inference_width, self.inference_height) == (self.width, self.height) else cv2.resize(rgba, (self.inference_width, self.inference_height), interpolation=cv2.INTER_AREA)
        with checkpoint_modules(module for module in self.estimator.model.modules() if not module._modules):
            if self.variant == "da3_metric_large":
                from preprocessing.depth_anything_v3.depth import _run_da3_depth_prediction

                metric_depth = _run_da3_depth_prediction(self.estimator.model, inference_frame[None, ..., :3], self.inference_width, chunk_size=1)[0]
                disparity = 1.0 / np.maximum(metric_depth, 1e-6)
                del metric_depth
            else:
                bgr = cv2.cvtColor(inference_frame, cv2.COLOR_RGBA2BGR)
                disparity = self.estimator.model.infer_image(bgr, input_size=min(self.inference_width, self.inference_height))
                del bgr
        del inference_frame
        low, high = (float(value) for value in np.nanpercentile(disparity, (2, 98)))
        if reset or self.low is None:
            self.low, self.high = low, high
        else:
            self.low = 0.95 * self.low + 0.05 * low
            self.high = 0.95 * self.high + 0.05 * high
        disparity -= self.low
        disparity /= max(self.high - self.low, 1e-6)
        np.clip(disparity, 0, 1, out=disparity)
        depth = disparity
        if disparity.shape != (self.height, self.width):
            depth = cv2.resize(disparity, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            del disparity
        return np.ascontiguousarray(depth, dtype=np.float32)


def _sample_info(sample: torch.Tensor) -> tuple[torch.dtype, torch.device, int, int, int, int]:
    if sample.ndim != 4 or sample.shape[0] not in (3, 4):
        raise ValueError("DLSS expects a [3|4, frames, height, width] tensor")
    channels, frames, height, width = sample.shape
    if frames < 1:
        raise ValueError("DLSS requires at least one frame")
    return sample.dtype, sample.device, channels, frames, height, width


def _scaled_dimension(size: int, scale: float) -> int:
    return max(2, math.floor(size * scale / 2 + 0.5) * 2)


def _nr_canvas_size(width: int, height: int, scale: float) -> tuple[int, int, int, int, float]:
    """Align the RenoDX NR output texture width while preserving the requested crop."""
    target_width, target_height = _scaled_dimension(width, scale), _scaled_dimension(height, scale)
    if target_width % 64 == 0:
        return target_width, target_height, width, height, scale
    output_width = math.ceil(target_width / 64) * 64
    render_width = round(output_width * width / target_width)
    worker_scale = output_width / render_width
    render_height = height
    while _scaled_dimension(render_height, worker_scale) < target_height:
        render_height += 1
    return target_width, target_height, render_width, render_height, worker_scale


def _frame_to_rgba(sample: torch.Tensor, index: int, width: int | None = None, height: int | None = None) -> np.ndarray:
    channels = sample.shape[0]
    frame = sample[:, index].detach()
    if frame.dtype != torch.uint8:
        frame = frame.add(1).mul(127.5).clamp(0, 255).to(torch.uint8)
    data = frame.to(device="cpu").permute(1, 2, 0).contiguous().numpy()
    source_height, source_width = data.shape[:2]
    width, height = width or source_width, height or source_height
    if channels == 4 and (width, height) == (source_width, source_height):
        return data
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[:source_height, :source_width, :channels] = data
    if channels == 3:
        rgba[..., 3] = 255
    if width > source_width:
        rgba[:source_height, source_width:] = rgba[:source_height, source_width - 1:source_width]
    if height > source_height:
        rgba[source_height:] = rgba[source_height - 1:source_height]
    return rgba


def _from_rgba_frames(data: np.ndarray, dtype: torch.dtype, device: torch.device, channels: int) -> torch.Tensor:
    output = torch.from_numpy(data).permute(3, 0, 1, 2)
    if channels == 3:
        output = output[:3]
    if device.type != "cpu":
        output = output.to(device=device)
    if dtype != torch.uint8:
        output = output.to(dtype=dtype).div_(127.5).sub_(1)
    return output


class NeuralRenderingSession(Worker):
    VIDEO_MAGIC = 0x34563544
    SETUP_MAGIC = 0x34505553
    DEPTH_VIDEO_MAGIC = 0x3144574E
    DEPTH_SETUP_MAGIC = 0x3152574E
    FRAME_MAGIC = 0x314D5246
    OUT_MAGIC = 0x3154554F

    def __init__(self, width: int, height: int, frames: int, scale: float, intensity: float = 1.0, quality_scale: float | None = None):
        output_width = _scaled_dimension(width, scale)
        output_height = _scaled_dimension(height, scale)
        if max(output_width, output_height) > 7680 or min(output_width, output_height) > 4320:
            raise ValueError(f"DLSS output {output_width}x{output_height} exceeds the 7680x4320 limit")
        intensity = max(0.0, min(2.0, float(intensity)))
        command = [str(NR_WORKER), "--nr-intensity", f"{intensity:.4f}", "--wangp-video"] if USE_DEPTH_GUIDE else [str(NR_WORKER), "--video"]
        super().__init__(command, HOST_DIR)
        self.output_width, self.output_height = output_width, output_height
        assert self.process.stdin is not None and self.process.stdout is not None
        if USE_DEPTH_GUIDE:
            self.process.stdin.write(struct.pack("<10I", self.DEPTH_VIDEO_MAGIC, width, height, output_width, output_height, frames, 0, 60, 0, 0))
        else:
            _name, perf_quality = NR_MODES[scale if quality_scale is None else quality_scale]
            self.process.stdin.write(struct.pack("<14I4f", self.VIDEO_MAGIC, width, height, output_width, output_height, 0, frames, perf_quality, 0, 0, 0, 0, 0, 0, intensity, 1.0, 1.0, -1.0))
        self.process.stdin.flush()
        if USE_DEPTH_GUIDE:
            response = struct.unpack("<6I", self.read_exact(struct.calcsize("<6I"), "DLSS Neural Rendering setup"))
            if response[0] != self.DEPTH_SETUP_MAGIC or response[1]:
                self.close(abort=True)
                raise RuntimeError(f"DLSS Neural Rendering depth setup failed (NGX 0x{response[1]:08X}).\n{self.error()}")
            self.render_width, self.render_height = response[2], response[3]
            negotiated_output = response[4], response[5]
        else:
            response = struct.unpack("<12I", self.read_exact(struct.calcsize("<12I"), "DLSS Neural Rendering setup"))
            if response[0] != self.SETUP_MAGIC or not response[1]:
                self.close(abort=True)
                raise RuntimeError(f"DLSS Neural Rendering setup failed (NGX 0x{response[2]:08X}).\n{self.error()}")
            self.render_width, self.render_height = response[3], response[4]
            negotiated_output = response[5], response[6]
        if negotiated_output != (output_width, output_height):
            self.close(abort=True)
            raise RuntimeError("DLSS worker negotiated unexpected output dimensions")

    def process_frame(self, index: int, rgba: np.ndarray, motion: np.ndarray, reset: bool, output: np.ndarray | None = None, *, depth: np.ndarray | None = None) -> np.ndarray:
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(struct.pack("<4Iq", self.FRAME_MAGIC, index, int(reset), 0, index))
        self.process.stdin.write(memoryview(np.ascontiguousarray(rgba, dtype=np.uint8)).cast("B"))
        self.process.stdin.write(memoryview(np.ascontiguousarray(motion, dtype=np.float16)).cast("B"))
        if USE_DEPTH_GUIDE:
            if depth is None:
                raise ValueError("DLSS Neural Rendering depth mode requires a depth guide")
            self.process.stdin.write(memoryview(np.ascontiguousarray(depth, dtype=np.float32)).cast("B"))
        self.process.stdin.flush()
        magic, out_index, ok, byte_count, ngx_result, _pts = struct.unpack("<5Iq", self.read_exact(struct.calcsize("<5Iq"), f"DLSS Neural Rendering frame {index}"))
        expected = self.output_width * self.output_height * 4
        if magic != self.OUT_MAGIC or out_index != index or not ok or byte_count != expected or ngx_result != 1:
            raise RuntimeError(f"DLSS Neural Rendering failed on frame {index} (NGX 0x{ngx_result:08X}).\n{self.error()}")
        return _read_array(self.process.stdout, (self.output_height, self.output_width, 4), output)


def neural_render(sample: torch.Tensor, scale: float, *, still_image: bool, depth_resolution: str, motion_vector: str, intensity: float = 1.0, abort_callback=None, progress_callback=None) -> torch.Tensor | None:
    require_runtime(temporal=False)
    dtype, device, channels, frame_count, height, width = _sample_info(sample)
    output_width, output_height, render_width, render_height, worker_scale = _nr_canvas_size(width, height, scale)
    session = NeuralRenderingSession(render_width, render_height, frame_count, worker_scale, intensity, scale)
    completed = False
    try:
        output = np.empty((frame_count, output_height, output_width, 4), dtype=np.uint8)
        work_output = None if (session.output_width, session.output_height) == (output_width, output_height) else np.empty((session.output_height, session.output_width, 4), dtype=np.uint8)
        if (render_width, render_height) != (width, height):
            print(f"[DLSS 5] Edge padding render canvas {width}x{height} -> {render_width}x{render_height}; output canvas {session.output_width}x{session.output_height} is cropped to {output_width}x{output_height}")
        flow_guides = None if still_image else FlowGuides(session.render_width, session.render_height, motion_vector)
        zero_motion = np.zeros((session.render_height, session.render_width, 2), dtype=np.float16) if still_image else None
        depth_guides = DepthGuides(session.render_width, session.render_height, depth_resolution) if USE_DEPTH_GUIDE else None
        for index in range(frame_count):
            if abort_callback is not None and abort_callback():
                return None
            render_frame = _frame_to_rgba(sample, index, session.render_width, session.render_height)
            motion, reset = (zero_motion, True) if still_image else flow_guides.process(render_frame)
            depth = depth_guides.process(render_frame, reset) if depth_guides is not None else None
            if abort_callback is not None and abort_callback():
                return None
            processed = session.process_frame(index, render_frame, motion, reset, output[index] if work_output is None else work_output, depth=depth)
            if channels == 4:
                processed[..., 3] = cv2.resize(render_frame[..., 3], (session.output_width, session.output_height), interpolation=cv2.INTER_LANCZOS4)
            if work_output is not None:
                output[index] = processed[:output_height, :output_width]
            if progress_callback is not None:
                progress_callback("DLSS 5 Neural Rendering", index + 1, frame_count)
            del render_frame, motion, depth, processed
        if abort_callback is not None and abort_callback():
            return None
        completed = True
    finally:
        try:
            session.close(abort=not completed)
        finally:
            offload_registry.unload_vram(list(_GUIDE_OFFLOADS))
    return _from_rgba_frames(output, dtype, device, channels)


class FrameGenerationSession(Worker):
    SETUP_MAGIC = 0x31534746
    SETUP_OUT_MAGIC = 0x31524746
    FRAME_MAGIC = 0x31464746
    FRAME_OUT_MAGIC = 0x314F4746

    def __init__(self, width: int, height: int, frames: int, generated_count: int, variant: DlssgVariant = DLSSG_UPSTREAM):
        super().__init__([str(variant.worker), "--serve", *dlssg_gpu_args(variant)], variant.dir)
        self.width, self.height = width, height
        self.generated_count = generated_count
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(struct.pack("<5I", self.SETUP_MAGIC, width, height, max(1, frames), generated_count))
        self.process.stdin.flush()
        magic, status, maximum, _reserved = struct.unpack("<4I", self.read_exact(struct.calcsize("<4I"), "DLSS Frame Generation setup"))
        if magic != self.SETUP_OUT_MAGIC or status or maximum < generated_count:
            self.close(abort=True)
            raise RuntimeError(f"DLSS Frame Generation {generated_count + 1}x setup failed (status {status}, runtime maximum {maximum + 1}x).\n{self.error()}")

    def process_frame(self, index: int, rgba: np.ndarray, motion: np.ndarray, timestamp: Fraction, reset: bool, output: np.ndarray | None = None) -> int:
        assert self.process.stdin is not None and self.process.stdout is not None
        self.process.stdin.write(struct.pack("<4I2q", self.FRAME_MAGIC, index, int(reset), 0, timestamp.numerator, timestamp.denominator))
        self.process.stdin.write(memoryview(np.ascontiguousarray(rgba, dtype=np.uint8)).cast("B"))
        self.process.stdin.write(memoryview(np.ascontiguousarray(motion, dtype=np.float16)).cast("B"))
        self.process.stdin.flush()
        magic, status, generated, disabled = struct.unpack("<4I", self.read_exact(struct.calcsize("<4I"), f"DLSS Frame Generation frame {index}"))
        if magic != self.FRAME_OUT_MAGIC or status:
            raise RuntimeError(f"DLSS Frame Generation failed on frame {index} (status {status}).\n{self.error()}")
        if disabled or generated == 0:
            return 0
        if generated != self.generated_count:
            raise RuntimeError(f"DLSS Frame Generation returned {generated} frames when {self.generated_count} were requested")
        if output is None:
            raise RuntimeError("DLSS Frame Generation returned a frame without a destination buffer")
        for index in range(generated):
            _read_array(self.process.stdout, (self.height, self.width, 4), output[index])
        return generated


def _interpolate(frame_count: int, width: int, height: int, frame_getter, fps: float, scale: int, motion_vector: str, abort_callback=None, progress_callback=None, variant: DlssgVariant = DLSSG_UPSTREAM) -> np.ndarray | None:
    generated_count = scale - 1
    session = FrameGenerationSession(width, height, frame_count, generated_count, variant)
    gpu_note = dlssg_gpu_note(variant)
    print(f"[DLSS 5] Frame Generation: {frame_count} input frames at {width}x{height}, x{scale} ({generated_count} generated frame(s) per pair) using {variant.worker.name} from {variant.dir.name}" + (f" on {gpu_note}" if gpu_note else ""))
    start = time.perf_counter()
    completed = False
    try:
        guides = FlowGuides(width, height, motion_vector)
        output = np.empty(((frame_count - 1) * scale + 1, height, width, 4), dtype=np.uint8)
        for index in range(frame_count):
            if abort_callback is not None and abort_callback():
                return None
            frame_start = time.perf_counter()
            frame = frame_getter(index)
            motion, reset = guides.process(frame)
            target_index = index * scale
            output[target_index] = frame
            generated = 0
            if index == 0:
                generated = session.process_frame(index, frame, motion, Fraction(index, 1) / Fraction(str(fps)), reset)
            else:
                generated_output = output[target_index - generated_count:target_index]
                generated = session.process_frame(index, frame, motion, Fraction(index, 1) / Fraction(str(fps)), reset, generated_output)
                if reset or not generated:
                    generated_output[:] = output[target_index - scale]
            if reset:
                status = "reset"
            elif generated:
                status = f"generated {generated} frame(s)"
            else:
                status = "disabled, carried previous frame forward"
            print(f"[DLSS 5] Frame Generation {index + 1}/{frame_count}: {status} in {time.perf_counter() - frame_start:.2f} s")
            if progress_callback is not None:
                progress_callback("DLSS Frame Generation", index + 1, frame_count)
            del frame, motion
        if abort_callback is not None and abort_callback():
            return None
        completed = True
        elapsed = time.perf_counter() - start
    finally:
        try:
            session.close(abort=not completed)
        finally:
            offload_registry.unload_vram(list(_GUIDE_OFFLOADS))
    if completed:
        print(f"[DLSS 5] Frame Generation finished: {frame_count} input frames -> {output.shape[0]} output frames in {elapsed:.1f} s ({frame_count / elapsed:.1f} input fps -> {output.shape[0] / elapsed:.1f} output fps)")
        for line in session.logs:
            print(f"[DLSS 5] worker: {line}")
    return output


def frame_generate(sample: torch.Tensor, previous_last_frame: torch.Tensor | None, fps: float, scale: int, *, motion_vector: str, abort_callback=None, progress_callback=None, variant: DlssgVariant | None = None) -> tuple[torch.Tensor | None, torch.Tensor, float]:
    variant = variant or DLSSG_UPSTREAM
    require_runtime(temporal=True, variant=variant)
    dtype, device, channels, frame_count, height, width = _sample_info(sample)
    next_previous = sample[:, -1:].clone()
    has_previous = previous_last_frame is not None
    input_count = frame_count + int(has_previous)

    def input_frame(index):
        return _frame_to_rgba(previous_last_frame if has_previous and index == 0 else sample, index - 1 if has_previous else index)

    output = _interpolate(input_count, width, height, input_frame, fps, scale, motion_vector, abort_callback=abort_callback, progress_callback=progress_callback, variant=variant)
    if output is None:
        return None, next_previous, fps * scale
    if has_previous:
        output = output[1:]
    result = _from_rgba_frames(output, dtype, device, channels)
    return result, next_previous, fps * scale
