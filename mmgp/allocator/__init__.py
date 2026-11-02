# Copyright (c) 2024-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""MMGP Optimized VRAM Allocator for PyTorch: recycles more efficiently the VRAM no longer used (Windows and Linux, NVIDIA GPUs,
PyTorch 2.3+).

    from mmgp import allocator
    allocator.install()          # before anything initializes CUDA

install(spill=True) places what VRAM has no room for in system RAM instead of raising an out of memory error (slower). torch.cuda memory
statistics and empty_cache are answered by the allocator. set_pressure_callback, mark_reset / mark_peak, room and stats serve the automatic
VRAM preload; set_vram_limit emulates a smaller GPU for tests."""
import ctypes
import contextlib
import os
import platform
import sys
import threading

import torch

STATS = ("allocated", "reserved", "peak_allocated", "peak_reserved", "chunks", "cached_ranges", "stale_ranges", "live_large", "small_cached", "graph_pools", "spilled", "recoveries", "driver_spilled")
_LIBRARIES = {("win32", "amd64"): "vmm_alloc_win_amd64.dll", ("linux", "x86_64"): "vmm_alloc_linux_x86_64.so", ("linux", "aarch64"): "vmm_alloc_linux_aarch64.so"}
_CHECK_ERRORS = {
    1: "the NVIDIA driver library could not be loaded",
    2: "the NVIDIA driver is too old (CUDA 11.2 or newer is needed)",
    3: "the NVIDIA driver failed to initialize",
    4: "no CUDA device was found",
    5: "the GPU does not support CUDA virtual memory management",
    6: "the GPU does not support CUDA memory pools",
}
_lib = None
active = None  # "vmm" or "expandable" once installed
_PRESSURE_CALLBACK = ctypes.CFUNCTYPE(ctypes.c_int64, ctypes.c_size_t, ctypes.c_int)
_OOM_ERROR_LIBRARIES = {("win32", "amd64"): "oom_error_win_amd64.dll", ("linux", "x86_64"): "oom_error_linux_x86_64.so"}
# PyTorch versions whose c10::Error (members, virtual functions, constructors) was checked identical to the headers the companion was built
# with (2.10): 2.6 to 2.14 and the development branch 2.15 of 2026-10. Before using another one, compare its c10/util/Exception.h.
OOM_ERROR_TORCH_VERSIONS = ((2, 6), (2, 15))
_oom_error = None  # the companion library, kept loaded while the allocator may call it
_pressure_callback = None  # the ctypes callback, kept alive while the library may call it
_warmup = threading.local()


def library_path():
    name = _LIBRARIES.get((sys.platform, platform.machine().lower()))
    if name is None:
        raise RuntimeError(f"The mmgp VRAM allocator has no library for {sys.platform} {platform.machine()}.")
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), name)


def install(mode="vmm", large_mb=256, chunk_mb=32, spill=False):
    """Replaces PyTorch's CUDA allocator. Must run before CUDA is initialized: PyTorch cannot swap an allocator it has started.
    mode "vmm": this allocator (Windows and Linux); "expandable": PyTorch's own expandable segments (Linux only).
    spill (vmm): tensors that VRAM has no room for go to pinned system RAM, mapped into the GPU's address space and read
    over PCIe, instead of raising an out of memory error. Slow, but lets a generation slightly too large for the VRAM finish. When RAM
    is too short to pin them, the driver allocates them as for PyTorch's allocator (on Windows, in shared GPU memory)."""
    global _lib, active
    if active == mode:
        return
    if active is not None:
        raise RuntimeError(f"The {active} VRAM allocator is already installed.")
    if torch.version.hip is not None or not torch.cuda.is_available():
        raise RuntimeError("The mmgp VRAM allocator needs an NVIDIA GPU.")
    if torch.cuda.is_initialized():
        raise RuntimeError("The VRAM allocator must be installed before CUDA is initialized.")
    if mode == "expandable":
        if sys.platform == "win32":
            raise RuntimeError("PyTorch's expandable segments are not available on Windows: use mode='vmm'.")
        settings = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments" not in settings:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(filter(None, (settings, "expandable_segments:True")))
        active = mode
        return
    if mode != "vmm":
        raise ValueError(f"Unknown VRAM allocator mode {mode!r}: expected 'vmm' or 'expandable'.")
    path = library_path()
    if not os.path.isfile(path):
        raise RuntimeError(f"{os.path.basename(path)} is missing: build it with python -m mmgp.allocator.build")
    lib = ctypes.CDLL(path)
    lib.vmm_check.argtypes = (ctypes.c_int,)
    status = lib.vmm_check(0)  # driver API only: no CUDA context is created
    if status:
        raise RuntimeError(f"The mmgp VRAM allocator cannot be used: {_CHECK_ERRORS.get(status, f'error {status}')}.")
    allocator = torch.cuda.memory.CUDAPluggableAllocator(path, "vmm_alloc", "vmm_free")
    hooks = allocator._allocator
    address = lambda name: ctypes.cast(getattr(lib, name), ctypes.c_void_p).value
    if not hasattr(hooks, "set_begin_allocate_to_pool"):  # the CUDA graph pool hooks of PyTorch 2.3+
        raise RuntimeError(f"The mmgp VRAM allocator requires PyTorch 2.3 or newer (installed: {torch.__version__}).")
    hooks.set_begin_allocate_to_pool(address("vmm_begin_allocate_to_pool"))
    hooks.set_end_allocate_to_pool_fn(address("vmm_end_allocate_to_pool"))
    hooks.set_release_pool(address("vmm_release_pool"))
    hooks.set_record_stream_fn(address("vmm_record_stream"))
    hooks.set_reset_fn(address("vmm_empty_cache"))  # every entry point (torch.cuda / torch.accelerator empty_cache) releases the cache
    torch.cuda.memory.change_current_allocator(allocator)
    lib.vmm_configure.argtypes = (ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int)
    lib.vmm_stats.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_int64))
    lib.vmm_reset_peaks.argtypes = (ctypes.c_int,)
    lib.vmm_set_pressure_callback.argtypes = (_PRESSURE_CALLBACK,)
    lib.vmm_set_vram_limit.argtypes = (ctypes.c_int64,)
    for name in ("vmm_mark_reset", "vmm_mark_peak", "vmm_room"):
        getattr(lib, name).argtypes = (ctypes.c_int,)
    lib.vmm_mark_peak.restype = lib.vmm_room.restype = ctypes.c_int64
    lib.vmm_configure(large_mb << 20, chunk_mb << 20, 3 if spill else 0)  # spill modes: pinned system RAM, then the driver's allocation
    lib.vmm_set_oom_thrower.argtypes = (ctypes.c_void_p,)
    lib.vmm_set_ram_release.argtypes = (ctypes.c_void_p,)
    lib.vmm_begin_warmup.argtypes = (ctypes.c_int, ctypes.c_ulonglong, ctypes.c_ulonglong, ctypes.c_void_p)
    lib.vmm_begin_warmup.restype = ctypes.c_ulonglong
    lib.vmm_end_warmup.argtypes = (ctypes.c_int, ctypes.c_ulonglong, ctypes.c_ulonglong, ctypes.c_ulonglong)
    lib.vmm_pause_warmup.argtypes = (ctypes.c_int,)
    lib.vmm_pause_warmup.restype = ctypes.c_int
    lib.vmm_prepare_reuse.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int)
    lib.vmm_prepare_reuse.restype = ctypes.c_int
    lib.vmm_lifetime_wrap.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
    lib.vmm_lifetime_wrap.restype = ctypes.c_void_p
    lib.vmm_lifetime_end.argtypes = (ctypes.c_void_p,)
    lib.vmm_lifetime_end.restype = ctypes.c_int
    lib.vmm_lifetime_available.argtypes = (ctypes.c_void_p, ctypes.c_int)
    lib.vmm_lifetime_available.restype = ctypes.c_int
    lib.vmm_lifetime_discard.argtypes = (ctypes.c_void_p,)
    lib.vmm_output_plan_begin.argtypes = (ctypes.c_int, ctypes.c_ulonglong, ctypes.c_ulonglong)
    lib.vmm_output_plan_begin.restype = ctypes.c_void_p
    lib.vmm_output_plan_pause.argtypes = ()
    lib.vmm_output_plan_bind.argtypes = (ctypes.c_void_p, ctypes.c_void_p)
    lib.vmm_output_plan_capture.argtypes = (ctypes.c_void_p,)
    lib.vmm_output_plan_finish.argtypes = (ctypes.c_void_p,)
    lib.vmm_output_plan_finish.restype = ctypes.c_int
    lib.vmm_output_plan_delete.argtypes = (ctypes.c_void_p,)
    lib.vmm_same_graph_storage.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int)
    lib.vmm_same_graph_storage.restype = ctypes.c_int
    lib.vmm_graph_storage_overlaps.argtypes = (ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int)
    lib.vmm_graph_storage_overlaps.restype = ctypes.c_int
    lib.vmm_pool_live.argtypes = (ctypes.c_int, ctypes.c_ulonglong, ctypes.c_ulonglong)
    lib.vmm_pool_live.restype = ctypes.c_int64
    lib.vmm_in_pool.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_ulonglong, ctypes.c_ulonglong)
    lib.vmm_in_pool.restype = ctypes.c_int
    _install_oom_error(lib)
    _lib, active = lib, mode
    _redirect_torch_memory_functions()
    from .. import virtual_gpu
    if virtual_gpu.state:  # an emulated smaller card: what it leaves the process
        set_vram_limit(virtual_gpu.state["limit"])
    from . import ram
    ram._link_vram_allocator()  # the RAM allocator, when installed first, gives its cache back before a spill is refused


def _install_oom_error(lib):
    """Failed allocations raise PyTorch's own out of memory error when the companion library suits this PyTorch, a RuntimeError with the same
    message otherwise (said once)."""
    global _oom_error
    version = tuple(int(part) for part in torch.__version__.split(".")[:2])
    name = _OOM_ERROR_LIBRARIES.get((sys.platform, platform.machine().lower()))
    path = None if name is None else os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    if not OOM_ERROR_TORCH_VERSIONS[0] <= version <= OOM_ERROR_TORCH_VERSIONS[1]:
        reason = f"PyTorch {torch.__version__} is not among the versions checked for its companion library (2.6 to 2.15)"
    elif sys.platform.startswith("linux") and not torch.compiled_with_cxx11_abi():
        reason = "this PyTorch build uses the old C++ ABI"
    elif path is None or not os.path.isfile(path):
        reason = f"its companion library {name or ''} is missing"
    else:
        try:
            _oom_error = ctypes.CDLL(path)
        except OSError as error:
            reason = f"its companion library could not be loaded ({error})"
        else:
            lib.vmm_set_oom_thrower(ctypes.cast(_oom_error.mmgp_throw_out_of_memory, ctypes.c_void_p))
            return
    print(f"[mmgp] VRAM allocator: out of memory errors are raised as RuntimeError, not torch.OutOfMemoryError: {reason}")


def _device_index(device=None):
    return torch.cuda.current_device() if device is None else torch.cuda._get_device_index(device, optional=True)


@contextlib.contextmanager
def graph_warmup(pool):
    """Run warmup on the graph's storage, including its capture. Native allocation paths outside this scope are unchanged."""
    if active != "vmm":
        yield
        return
    device = torch.cuda.current_device()
    stream = torch.cuda.current_stream(device)
    token = _lib.vmm_begin_warmup(device, *pool, stream.cuda_stream)
    _warmup.depth = getattr(_warmup, "depth", 0) + 1
    try:
        yield
    finally:
        _warmup.depth -= 1
        _lib.vmm_end_warmup(device, *pool, token)


def pool_live(pool, device=None):
    """Number of allocations alive in a graph pool."""
    return _lib.vmm_pool_live(_device_index(device), *pool)


def in_pool(tensor, pool):
    """Whether the storage of tensor is an allocation of this graph pool."""
    return bool(_lib.vmm_in_pool(tensor.untyped_storage().data_ptr(), tensor.device.index, *pool))


def in_graph_warmup():
    return getattr(_warmup, "depth", 0) > 0


def owns_tensor(tensor):
    return active == "vmm" and bool(_lib.vmm_prepare_reuse(tensor.untyped_storage().data_ptr(), None, 0))


def prepare_reuse(tensor):
    """Order reuse of an adopted live buffer after its recorded consumers, without waiting on the CPU."""
    stream = torch.cuda.current_stream(tensor.device)
    if not _lib.vmm_prepare_reuse(tensor.untyped_storage().data_ptr(), stream.cuda_stream, 1):
        raise RuntimeError("CUDA graph tried to reuse a buffer no longer owned by the MMGP allocator")


_capsule_get = ctypes.pythonapi.PyCapsule_GetPointer
_capsule_get.argtypes = (ctypes.py_object, ctypes.c_char_p)
_capsule_get.restype = ctypes.c_void_p
_capsule_name = ctypes.pythonapi.PyCapsule_SetName
_capsule_name.argtypes = (ctypes.py_object, ctypes.c_char_p)
_capsule_name.restype = ctypes.c_int
_capsule_new = ctypes.pythonapi.PyCapsule_New
_capsule_new.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p)
_capsule_new.restype = ctypes.py_object
_capsule_valid = ctypes.pythonapi.PyCapsule_IsValid
_capsule_valid.argtypes = (ctypes.py_object, ctypes.c_char_p)
_capsule_valid.restype = ctypes.c_int


def lifetime_available(tensor):
    """Other-stream consumers keep their input protected instead of adding a dependency for memory reuse."""
    return bool(_lib.vmm_lifetime_available(tensor.untyped_storage().data_ptr(), tensor.device.index))


def lifetime_view(tensor):
    """An execution-owned view; native storage destruction detects the last alias without Python dispatch hooks."""
    if active != "vmm":
        return tensor, None
    capsule = torch.utils.dlpack.to_dlpack(tensor)
    token = _lib.vmm_lifetime_wrap(_capsule_get(capsule, b"dltensor"), tensor.numel() * tensor.element_size())
    if token is None:
        return tensor, None
    _capsule_name(capsule, b"used_dltensor")  # ownership passes to the wrapping DLPack storage
    wrapped = _capsule_new(token, b"dltensor", None)
    try:
        with torch.inference_mode(False):
            view = torch.utils.dlpack.from_dlpack(wrapped)
    except BaseException:
        _lib.vmm_lifetime_end(token)
        if _capsule_valid(wrapped, b"dltensor"):
            _lib.vmm_lifetime_discard(token)
        raise
    return view, token


def end_lifetime(token):
    return bool(_lib.vmm_lifetime_end(token))


class OutputPlan:
    """Bind eligible captured outputs to warmup storage at their observed allocation, without a GPU copy."""
    def __init__(self, pool):
        self.token = _lib.vmm_output_plan_begin(torch.cuda.current_device(), *pool) if active == "vmm" else None

    def pause(self):
        if self.token is not None:
            _lib.vmm_output_plan_pause()

    def bind(self, tensor):
        if self.token is not None:
            _lib.vmm_output_plan_bind(self.token, tensor.untyped_storage().data_ptr())

    def capture(self):
        if self.token is not None:
            _lib.vmm_output_plan_capture(self.token)

    def finish(self):
        return self.token is None or bool(_lib.vmm_output_plan_finish(self.token))

    def close(self):
        if self.token is not None:
            self.pause()
            _lib.vmm_output_plan_delete(self.token)
            self.token = None


def graph_storage_overlaps(first, second):
    """Capture planning only: escaped outputs must not alias physical pages loaned by static inputs."""
    return (active == "vmm" and first.is_cuda and first.device == second.device and
            bool(_lib.vmm_graph_storage_overlaps(first.untyped_storage().data_ptr(), second.untyped_storage().data_ptr(), first.device.index)))


def same_graph_storage(first, second):
    return (active == "vmm" and first.is_cuda and first.device == second.device and first.storage_offset() == second.storage_offset() == 0 and
            first.shape == second.shape and first.stride() == second.stride() and first.dtype == second.dtype and
            bool(_lib.vmm_same_graph_storage(first.data_ptr(), second.data_ptr(), first.device.index)))


@contextlib.contextmanager
def normal_allocations():
    """Persistent warmup caches need independent storage: earlier graphs can overwrite their pool's scratch on replay."""
    if not in_graph_warmup():
        yield
        return
    previous = _lib.vmm_pause_warmup(1)
    try:
        yield
    finally:
        _lib.vmm_pause_warmup(previous)


def stats(device=None):
    out = (ctypes.c_int64 * len(STATS))()
    _lib.vmm_stats(_device_index(device), out)
    return dict(zip(STATS, out))


def set_pressure_callback(fn):
    """fn(size, device) -> bytes it freed, None removes it. Called when an allocation finds the VRAM short, before it fails (or spills),
    from the allocating thread without the allocator's lock: fn may free tensors, then the allocation is tried again."""
    global _pressure_callback
    _pressure_callback = None if fn is None else _PRESSURE_CALLBACK(lambda size, device: int(fn(size, device)))
    _lib.vmm_set_pressure_callback(_pressure_callback or _PRESSURE_CALLBACK())  # a prototype called without argument is a null pointer


def set_vram_limit(nbytes):
    """The VRAM this process may reserve on each device (0: no limit but the GPU's): emulates a smaller GPU, for tests. Unlike another process
    holding VRAM, which Windows pages out of the GPU while it is idle, it is a hard limit: allocations beyond it fail as they would on that GPU."""
    _lib.vmm_set_vram_limit(int(nbytes))


def mark_reset(device=None):
    """Starts a phase whose peak of allocated memory mark_peak returns (torch.cuda.reset_peak_memory_stats does not reset it)."""
    _lib.vmm_mark_reset(_device_index(device))


def mark_peak(device=None):
    return _lib.vmm_mark_peak(_device_index(device))


def room(device=None):
    """VRAM the allocator may still take, in bytes: this process' free budget (DXGI on Windows, the driver's free memory elsewhere), or NVML's
    free memory of the whole GPU if smaller."""
    return _lib.vmm_room(_device_index(device))


def _redirect_torch_memory_functions():
    # PyTorch's memory statistics do not reach a pluggable allocator: they are answered by this one
    def memory_allocated(device=None):
        return stats(device)["allocated"]

    def memory_reserved(device=None):
        return stats(device)["reserved"]

    def max_memory_allocated(device=None):
        return stats(device)["peak_allocated"]

    def max_memory_reserved(device=None):
        return stats(device)["peak_reserved"]

    def reset_peak_memory_stats(device=None):
        _lib.vmm_reset_peaks(_device_index(device))

    def memory_stats(device=None):
        s = stats(device)
        return {"allocated_bytes.all.current": s["allocated"], "allocated_bytes.all.peak": s["peak_allocated"],
                "reserved_bytes.all.current": s["reserved"], "reserved_bytes.all.peak": s["peak_reserved"], "num_alloc_retries": 0}

    driver_mem_get_info = torch.cuda.memory.mem_get_info

    def mem_get_info(device=None):  # free: the room of this allocator (the driver's figure misses VRAM on Windows, see room_now), total: the driver's
        return room(device), driver_mem_get_info(device)[1]

    for module in (torch.cuda, torch.cuda.memory):
        for fn in (memory_allocated, memory_reserved, max_memory_allocated, max_memory_reserved, reset_peak_memory_stats, memory_stats, mem_get_info):
            setattr(module, fn.__name__, fn)
