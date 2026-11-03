# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""CUDA memory settings applied when WanGP starts, before anything initializes CUDA: an emulated smaller GPU (--max-vram), a RAM budget (--max-ram), a
simulated PC with less RAM or GPU under real pressure (--ram-stress / --vram-stress GB[:OTHERS]), the VRAM allocator
(Configuration > RAM/VRAM Management or --vram-allocator), its debug mode (--vram-debug) and the per-thread stack reserve of the CUDA context; the RAM debug
mode (--ram-debug) and the MMGP RAM allocator (Configuration > RAM/VRAM Management or --ram-allocator); and the folder of the driver's
compute cache (Configuration > Performance)."""
import ctypes
import json
import os
import sys
import time

VRAM_ALLOCATOR_KEY = "vram_allocator"
VRAM_ALLOCATOR_CHOICES = ("default", "vmm", "vmm_spill")
RAM_ALLOCATOR_KEY = "ram_allocator"
RAM_ALLOCATOR_CHOICES = ("default", "mmgp")
RAM_ALLOCATOR_DEFAULT = "mmgp"
_vram_debug = _ram_debug = False
VRAM_ALLOCATOR = "default"  # the VRAM allocator running in this process
# The CUDA context reserves this much local memory for the call stack of every thread the GPU can run at once: with the driver's 1 KiB,
# about 0.25 GiB of VRAM on a large GPU. WanGP's kernels need less; the driver raises the reserve when a kernel needs more.
# WANGP_CUDA_STACK_BYTES overrides it (0 keeps the driver's default).
CUDA_STACK_BYTES = 256
COMPUTE_CACHE_KEY, COMPUTE_CACHE_PATH_KEY, COMPUTE_CACHE_PATH_DEFAULT = "compute_cache", "compute_cache_path", "ComputeCache"
COMPUTE_CACHE_MAX_BYTES = 4 << 30  # the driver's maximum
WANGP_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def compute_cache_folder(path):
    """The folder of the compute cache: a relative path is in the WanGP folder."""
    return os.path.join(WANGP_ROOT, path.strip() or COMPUTE_CACHE_PATH_DEFAULT)


def mmgp_vram_allocator():
    """The MMGP VRAM allocator runs: without the fragmentation the VAE tiling presets were made against, VAEs offer two levels,
    Default (fast, about 2-3 GB of VRAM) and Low VRAM (Configuration > RAM/VRAM Management > VAE Tiling, values 0 and 3)."""
    return VRAM_ALLOCATOR in ("vmm", "vmm_spill")


def compute_cache_useful():
    """NVIDIA Blackwell GPUs (RTX 50xx, B200...): libraries ship their code in a form that the driver optimizes for the exact GPU when it
    loads it, about 15 s per launch, and caches; older GPUs run the shipped code as is. ROCm reports AMD architectures as capabilities."""
    import torch
    return torch.version.hip is None and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


def _argv_value(argv, option):
    for index, arg in enumerate(argv):
        if arg == option and index + 1 < len(argv):
            return argv[index + 1]
        if arg.startswith(option + "="):
            return arg.split("=", 1)[1]
    return None


def _config_value(argv, config_filename, key, default=None):
    # wgp.py reads its configuration after CUDA has started: the choices needed before are read from the same file first
    config_dir = _argv_value(argv, "--config")
    for path in ([os.path.join(os.path.abspath(config_dir), config_filename)] if config_dir else []) + [config_filename]:
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as reader:
                return json.load(reader).get(key, default)
    return default


def _requested(argv, config_filename, option, key):
    value = _argv_value(argv, option)
    return _config_value(argv, config_filename, key) if value is None else value


def _apply_compute_cache(argv, config_filename):
    # The NVIDIA driver keeps the GPU code it optimizes for WanGP in this folder rather than in the one it shares with every program, which
    # programs using the default size trim back to 1 GiB: with up to 4 GiB the code of every model is optimized once and no longer rewritten.
    # The driver reads these variables when CUDA starts.
    if not _config_value(argv, config_filename, COMPUTE_CACHE_KEY, False):
        return
    path = compute_cache_folder(_config_value(argv, config_filename, COMPUTE_CACHE_PATH_KEY, COMPUTE_CACHE_PATH_DEFAULT))
    try:
        os.makedirs(path, exist_ok=True)
    except OSError as error:  # a folder that is no longer available (removed drive): the driver's own cache is used
        print(f"[CUDA] Compute Cache folder {path} not available ({error}): the NVIDIA driver's default cache is used")
        return
    os.environ["CUDA_CACHE_PATH"], os.environ["CUDA_CACHE_MAXSIZE"] = path, str(COMPUTE_CACHE_MAX_BYTES)
    print(f"[CUDA] Compute Cache: {path}")


def requested_vram_allocator(argv, config_filename):
    return _requested(argv, config_filename, "--vram-allocator", VRAM_ALLOCATOR_KEY) or "vmm_spill"


def _stress(argv, option):
    """--ram-stress / --vram-stress GB[:OTHERS]: the simulated machine's size and, optionally, what the system and other programs use of it."""
    value = _argv_value(argv, option)
    if value is None:
        return None
    size, _, others = value.partition(":")
    return float(size), float(others) if others else None


def apply_startup_settings(argv, config_filename):
    ram_stress_value, max_ram = _stress(argv, "--ram-stress"), _argv_value(argv, "--max-ram")
    max_vram, vram_stress_value = _argv_value(argv, "--max-vram"), _stress(argv, "--vram-stress")
    if max_vram is not None and vram_stress_value is not None:
        raise ValueError("--vram-stress emulates its own GPU: use it without --max-vram")
    if ram_stress_value is not None:  # before mmgp.offload reads the RAM at import; a budget then applies within the simulated PC
        from mmgp import ram_stress
        ram_stress.start(*ram_stress_value)
    if max_ram is not None:
        from mmgp import virtual_gpu
        virtual_gpu.install_ram(float(max_ram))
    import torch
    ram_allocator = _requested(argv, config_filename, "--ram-allocator", RAM_ALLOCATOR_KEY) or RAM_ALLOCATOR_DEFAULT
    if ram_allocator not in RAM_ALLOCATOR_CHOICES:
        raise ValueError(f"Unknown RAM allocator {ram_allocator!r}: expected one of {RAM_ALLOCATOR_CHOICES}")
    if ram_allocator == "mmgp":
        from mmgp.allocator import ram
        try:
            ram.install()
        except RuntimeError as error:  # a platform or PyTorch it does not support: the default must not stop WanGP
            print(f"[RAM] MMGP RAM Allocator not available ({error}): PyTorch's CPU allocator is used")
        else:
            print("[RAM] MMGP RAM Allocator: the RAM of freed CPU tensors goes back to the system")
    # before the VRAM allocator, which takes the emulated card's limit when it installs, and before WanGP allocates VRAM
    if max_vram is not None and torch.cuda.is_available():
        from mmgp import virtual_gpu
        virtual_gpu.install(float(max_vram))
    if vram_stress_value is not None and torch.cuda.is_available():
        from mmgp import vram_stress
        gpu = _argv_value(argv, "--gpu") or ""
        vram_stress.start(*vram_stress_value, device=int(gpu.split(":")[1]) if ":" in gpu else 0)
    if torch.version.hip is None:
        _apply_compute_cache(argv, config_filename)
    if torch.version.hip is None and torch.cuda.is_available():
        _apply_cuda_settings(argv, config_filename, torch)
    if (max_vram is not None or vram_stress_value is not None) and torch.cuda.is_available() and VRAM_ALLOCATOR not in ("vmm", "vmm_spill"):
        from mmgp import virtual_gpu
        virtual_gpu.limit_torch_allocator()
    debug_value = _argv_value(argv, "--ram-debug")
    debug_mb = float(debug_value) if debug_value and not debug_value.startswith("-") else 16 if "--ram-debug" in argv else 0
    if debug_mb > 0:
        from mmgp.allocator import ram_debug
        ram_debug.start(min_mb=debug_mb)
        global _ram_debug
        _ram_debug = True
        print(f"[RAM] Debug mode: a report of where the RAM goes is written after each generation and after each model release in <outputs>/ram_debug "
              f"(allocations of {debug_mb:g} MB and more traced to their source)")


def _apply_cuda_settings(argv, config_filename, torch):
    allocator = requested_vram_allocator(argv, config_filename)
    if allocator not in VRAM_ALLOCATOR_CHOICES:
        raise ValueError(f"Unknown VRAM allocator {allocator!r}: expected one of {VRAM_ALLOCATOR_CHOICES}")
    if allocator in ("vmm", "vmm_spill"):
        from mmgp import allocator as vram_allocator
        try:
            vram_allocator.install(spill=allocator == "vmm_spill")
        except RuntimeError as error:  # a GPU, driver, platform or PyTorch it does not support: the default must not stop WanGP
            print(f"[VRAM] MMGP Optimized VRAM Allocator not available ({error}): PyTorch's allocator is used")
            allocator = "default"
        else:
            print("[VRAM] MMGP Optimized VRAM Allocator" + (", spilling into RAM when VRAM is full" if allocator == "vmm_spill" else ""))
    global VRAM_ALLOCATOR
    VRAM_ALLOCATOR = allocator
    debug_mb = float(_argv_value(argv, "--vram-debug") or 0)
    if debug_mb > 0:
        if allocator not in ("vmm", "vmm_spill"):
            raise ValueError("--vram-debug needs the MMGP allocator: --vram-allocator vmm or vmm_spill")
        from mmgp.allocator import debug
        debug.start(min_mb=debug_mb)
        global _vram_debug
        _vram_debug = True
        print(f"[VRAM] Debug mode: allocations of {debug_mb:g} MB and more recorded, a report is written after each generation in <outputs>/vram_debug")
    stack_bytes = int(os.environ.get("WANGP_CUDA_STACK_BYTES", CUDA_STACK_BYTES))
    if stack_bytes > 0:
        torch.zeros(1, device="cuda")  # creates the context, made current in this thread
        driver = ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" else ctypes.CDLL("libcuda.so.1")
        result = driver.cuCtxSetLimit(0, ctypes.c_size_t(stack_bytes))  # CU_LIMIT_STACK_SIZE
        if result != 0:
            raise RuntimeError(f"cuCtxSetLimit(CU_LIMIT_STACK_SIZE, {stack_bytes}) failed with CUDA error {result}")


def write_vram_debug_report(output_dir, label):
    """With --vram-debug: the report of the generation that just ended, in output_dir/vram_debug; then new peaks for the next one."""
    if not _vram_debug:
        return
    from mmgp.allocator import debug
    path = os.path.join(output_dir, "vram_debug", f"{time.strftime('%Y-%m-%d-%Hh%Mm%Ss')}_{label or 'generation'}.json")
    debug.report(path)
    debug.reset()
    print(f"[VRAM debug] Report: {path} (summary: {os.path.splitext(path)[0]}.md)")


def release_ram_cache():
    """The end of a queue: the freed blocks the MMGP RAM allocator kept for reuse go back to the system (between generations of the same model
    they are kept: the next one reuses them)."""
    from mmgp.allocator import ram
    ram.release()


def ram_debug_mark(name):
    """With --ram-debug: the RAM that follows counts for the peak of phase name."""
    if _ram_debug:
        from mmgp.allocator import ram_debug
        ram_debug.mark(name)


def write_ram_debug_report(output_dir, label, census_label="after generation"):
    """With --ram-debug: a census of the RAM now (census_label) and the peaks of the phases since the previous report, in output_dir/ram_debug."""
    if not _ram_debug:
        return
    from mmgp.allocator import ram_debug
    path = os.path.join(output_dir, "ram_debug", f"{time.strftime('%Y-%m-%d-%Hh%Mm%Ss')}_{label or 'generation'}.json")
    ram_debug.report(path, label=label, snapshot_label=census_label)
    ram_debug.reset()
    print(f"[RAM debug] Report: {path} (summary: {os.path.splitext(path)[0]}.md); the latest report includes the RAM after each report since the start")
