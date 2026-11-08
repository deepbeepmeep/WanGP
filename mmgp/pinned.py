# Copyright (c) 2024-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Pinned host memory of MMGP, in one place.

Every pinned byte is normal RAM locked in place (cudaHostRegister):
- exact sizes, rounded to the page;
- no driver allocation (cudaHostAlloc, ~7-15 GB/s and serialized in the driver);
- counted in total_bytes, by kind (stats()), against the reserved RAM budget (fits());
- given back when its owner is closed.

PyTorch's pinned allocator (pin_memory=True) rounds each block up to a power of two and keeps freed blocks until
torch._C._host_emptyCache(), outside any budget. MMGP uses it only for small transient buffers (activations, step inputs).

Unlocking (cudaHostUnregister) waits for all the work queued on the GPU, so registrations are closed at quiet points: when their owner
goes there (a model released, the pool's entries leaving), or retire() now and collect() at the next quiet point.
- Registration: a locked range and what keeps its RAM alive.
- block(size): page-aligned normal RAM of its own, not yet locked, for callers that fill it before locking it.
- lock(...): registers a range from the registration thread, so that a failure leaves no CUDA error pending in the caller's thread.
- empty(nbytes, kind): a locked uint8 tensor of nbytes, None if the budget or the driver refuses it.
- Buffer: a reusable locked buffer that grows to the largest request; the outgrown one is retired.
"""
import collections
import ctypes
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy
import torch

PAGE = 4096
LOCKS_RAM = True # pinned tensors live in normal RAM locked in place once filled (cudaHostRegister, about 1 ms per 256 MB), instead of
                 # pinned memory allocated by the CUDA driver (cudaHostAlloc, used if False: tens of ms per 256 MB, or seconds per GB)
total_bytes = 0 # every byte locked by a Registration
limit_bytes = 0 # the total at which a lock or an allocation was refused (0: none yet): nothing more is attempted beyond it
_bytes_lock = threading.Lock()
_by_kind = collections.Counter()
_failures = {} # keep backing allocations alive after an unregister/synchronization failure
_retired = [] # owners whose registrations collect() closes at the next quiet point
_registrar = None


class Registration:
    """Own the backing RAM until CUDA has finished using it and successfully unregistered it. flags: cudaHostRegister's (2 maps the RAM in
    the GPU's address space, mapped.py). kind: what it is for, in stats()."""
    def __init__(self, buffer, pointer, size, device, flags=0, kind="other"):
        self.buffer, self.pointer, self.size, self.device, self.flags, self.kind = buffer, pointer, size, device, flags, kind
        self.registered, self.error = False, None
        self.guard = threading.Lock()
        self.failures, self.is_finalizing = _failures, sys.is_finalizing

    def register(self):
        global total_bytes
        with self.guard:
            if self.registered:
                raise RuntimeError("Host memory is already registered")
            error = torch.cuda.cudart().cudaHostRegister(self.pointer, self.size, self.flags)
            if int(error) == 0:
                self.registered = True
                with _bytes_lock:
                    total_bytes += self.size
                    _by_kind[self.kind] += self.size
            return error

    def close(self):
        global total_bytes
        with self.guard:
            if not self.registered:
                return True
            if self.error is not None:
                return False # never retry a failed driver operation from a finalizer
            try:
                if self.is_finalizing():
                    raise RuntimeError("Python is shutting down")
                with torch.cuda.device(self.device):
                    torch.cuda.synchronize(self.device)
                    error = torch.cuda.cudart().cudaHostUnregister(self.pointer)
                    if int(error) != 0:
                        raise RuntimeError(f"cudaHostUnregister returned {error}")
            except Exception as error:
                self.error = str(error)
                self.failures[self.pointer] = self # retain RAM and its accounting until process exit
                try:
                    print(f"[mmgp] Unable to release {self.size / 2**20:.2f} MB of registered RAM: {error}. Memory retained for safety; restart the process before further loading.", file=sys.stderr)
                except Exception:
                    pass # the output stream may already be closed during interpreter shutdown
                return False
            self.registered = False
            with _bytes_lock:
                total_bytes -= self.size
                _by_kind[self.kind] -= self.size
            self.buffer = None
            return True

    def __del__(self):
        self.close()


class LockedRAM(numpy.ndarray):
    # torch.from_numpy retains this exact owner, including its registration, until the last storage view dies.
    pass


def block(size):
    """A block for pinned tensors and its owner (None for driver pinned memory, LOCKS_RAM False): page-aligned RAM of its own, not locked."""
    if not LOCKS_RAM:
        return torch.empty(size, dtype=torch.uint8, pin_memory=True, device="cpu"), None
    ram = numpy.empty(size + PAGE, dtype=numpy.uint8)
    start = -ram.ctypes.data % PAGE # page aligned, like driver pinned memory
    owner = ram[start:start + size].view(LockedRAM)
    return torch.from_numpy(owner), owner


class _Section:
    """A section of the paging file mapped here (Windows): unmapped and closed with its last reference."""
    def __init__(self, nbytes):
        import ctypes.wintypes as wt
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.CreateFileMappingW.argtypes, kernel32.CreateFileMappingW.restype = [wt.HANDLE, ctypes.c_void_p, wt.DWORD, wt.DWORD, wt.DWORD, ctypes.c_wchar_p], wt.HANDLE
        kernel32.MapViewOfFile.argtypes, kernel32.MapViewOfFile.restype = [wt.HANDLE, wt.DWORD, wt.DWORD, wt.DWORD, ctypes.c_size_t], ctypes.c_void_p
        kernel32.UnmapViewOfFile.argtypes, kernel32.CloseHandle.argtypes = [ctypes.c_void_p], [wt.HANDLE]
        self.kernel32 = kernel32
        self.handle = kernel32.CreateFileMappingW(wt.HANDLE(-1), None, 0x04, nbytes >> 32, nbytes & 0xFFFFFFFF, None)  # PAGE_READWRITE
        self.view = kernel32.MapViewOfFile(self.handle, 0x0002 | 0x0004, 0, 0, nbytes) if self.handle else None  # FILE_MAP_WRITE | FILE_MAP_READ
        if not self.view:
            if self.handle:
                kernel32.CloseHandle(self.handle)
            raise MemoryError(f"A section of {nbytes} bytes was refused ({ctypes.get_last_error()})")
        self.sections = _sections  # held: at the interpreter's exit the module's globals are gone before the last sections
        self.sections[self.view] = self.handle

    def __del__(self):
        self.sections.pop(self.view, None)
        self.kernel32.UnmapViewOfFile(self.view)
        self.kernel32.CloseHandle(self.handle)


_sections = {}  # view address -> handle of the sections of shared_block()


def shared_block(size):
    """As block(), in a section of the paging file that another process may map (the helper that copies the GPU's share of experts): its handle is
    section_of(the block's address). Windows."""
    section = _Section(size)
    owner = numpy.frombuffer((ctypes.c_uint8 * size).from_address(section.view), dtype=numpy.uint8).view(LockedRAM)
    owner.section = section
    return torch.from_numpy(owner), owner


def section_of(address):
    """The handle of the section of a shared_block() starting at address, None for other RAM."""
    return _sections.get(address)


def fits(nbytes, cap=0):
    """Whether nbytes more stay within cap (the reserved RAM budget, 0: none) and under the total at which a lock was refused."""
    return not (cap > 0 and total_bytes + nbytes > cap or 0 < limit_bytes < total_bytes + nbytes)


def refused(nbytes=0):
    """A lock or an allocation of nbytes failed: nothing more is attempted beyond the current total (fits())."""
    global limit_bytes
    limit_bytes = total_bytes + nbytes if limit_bytes <= 0 else min(limit_bytes, total_bytes + nbytes)


def account(nbytes, kind):
    """Counts (negative: uncounts) driver pinned memory that no Registration owns (LOCKS_RAM False)."""
    global total_bytes
    with _bytes_lock:
        total_bytes += nbytes
        _by_kind[kind] += nbytes


def lock(buffer, pointer, size, kind, flags=0):
    """The registration of size bytes at pointer (buffer keeps them alive), made on the registration thread; None if refused."""
    global _registrar
    if _registrar is None:
        _registrar = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mmgp_pinned", initializer=torch.cuda.set_device, initargs=(torch.cuda.current_device(),))
    registration = Registration(buffer, pointer, size, torch.cuda.current_device(), flags, kind)
    return registration if int(_registrar.submit(registration.register).result()) == 0 else None


def empty(nbytes, kind, cap=0, flags=0):
    """A uint8 tensor of nbytes in locked RAM of its own, locked as long as a view of it lives; None if cap or the driver refuses it."""
    size = -(-nbytes // PAGE) * PAGE
    if not fits(size, cap):
        return None
    try:
        tensor, owner = block(size)
    except (MemoryError, RuntimeError):
        refused(size)
        return None
    if owner is None:
        return tensor[:nbytes]
    owner.registration = lock(owner.base, tensor.data_ptr(), size, kind, flags)
    if owner.registration is None:
        refused(size)
        return None
    return tensor[:nbytes]


def retire(tensor):
    """tensor (or any owner of registrations) is no longer used: its registrations close at the next quiet point (collect())."""
    if tensor is not None:
        _retired.append(tensor)


def collect():
    """A quiet point: the registrations retired since the last one close (waiting for the GPU once)."""
    if _retired:
        torch.cuda.synchronize()
        _retired.clear()


class Buffer:
    """A reusable buffer of locked RAM that grows to the largest request (rounded up to granularity). The outgrown buffer is retired:
    transfers may still read it until the next quiet point."""
    def __init__(self, kind, granularity=2**20, flags=0):
        self.kind, self.granularity, self.flags, self.tensor = kind, granularity, flags, None

    def get(self, nbytes, cap=0):
        """At least nbytes of locked RAM, None if refused (the previous buffer stays retired)."""
        if self.tensor is None or self.tensor.numel() < nbytes:
            retire(self.tensor)
            self.tensor = empty(-(-nbytes // self.granularity) * self.granularity, self.kind, cap, self.flags)
        return self.tensor

    def release(self):
        retire(self.tensor)
        self.tensor = None


def stats():
    """{kind: bytes locked} and the total."""
    with _bytes_lock:
        return {kind: nbytes for kind, nbytes in _by_kind.items() if nbytes}, total_bytes
