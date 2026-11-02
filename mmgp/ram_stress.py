# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""A PC with less RAM simulated under real memory pressure: a helper process locks RAM until the system has only what such a PC would leave
available (what the system and other programs use beside this process when it starts: RAM_OVERHEAD_GB unless given), and psutil reports
that PC's total RAM, as MMGP reads it at import: MMGP plans as there, and the system trims and pages as there. The helper keeps its RAM
locked until this process ends (it waits for the end of its stdin, a pipe from this process). The other programs have less RAM too: while
the available RAM stays under SAFETY_GB for 10 s, the helper gives back 1 GiB.

    from mmgp import ram_stress; ram_stress.start(32)       # before mmgp.offload is imported (WanGP: --ram-stress 32)
    from mmgp import ram_stress; ram_stress.start(64, 2.5)  # 2.5 GB used by the system and other programs (WanGP: --ram-stress 64:2.5)

Linux needs the right to lock that much memory (ulimit -l)."""
import ctypes
import math
import os
import subprocess
import sys
import threading
import time

GB = 2**30
RAM_OVERHEAD_GB = 6 if os.name == "nt" else 2  # the RAM the system, the desktop and a few programs use on such a PC when the process starts
SAFETY_GB = 1
state = {}  # total (bytes), helper (process) once started


def start(ram_gb, others_gb=None):
    """Simulates a PC with ram_gb (GiB) of RAM, of which the system and other programs use others_gb (RAM_OVERHEAD_GB by default), before
    mmgp.offload reads the RAM at import."""
    import psutil
    if state:
        raise RuntimeError("The RAM stress is already started.")
    others = RAM_OVERHEAD_GB if others_gb is None else float(others_gb)
    total = int(float(ram_gb) * GB)
    target = total - int(others * GB) - psutil.Process().memory_info().rss
    print(f"[MMGP] RAM stress: locking {max(0, psutil.virtual_memory().available - target) / GB:.0f} GiB of RAM for a {float(ram_gb):g} GB PC "
          f"({others:g} GB used by the system and other programs)...", flush=True)
    helper = hold(target)
    state.update(total=total, helper=helper)
    real_virtual_memory = psutil.virtual_memory

    def virtual_memory():
        real = real_virtual_memory()
        available = min(total, real.available)
        return real._replace(total=total, available=available, used=total - available, free=min(real.free, available), percent=round(100 * (total - available) / total, 1))
    psutil.virtual_memory = virtual_memory
    print(f"[MMGP] RAM stress: a {float(ram_gb):g} GB PC simulated, {helper.locked / GB:.0f} GiB locked by a helper process, {virtual_memory().available / GB:.1f} GiB available", flush=True)
    if helper.locked < helper.wanted:
        print(f"[MMGP] RAM stress: only {helper.locked / GB:.0f} of {helper.wanted / GB:.0f} GiB could be locked: more RAM stays available than on that PC", flush=True)


def hold(available):
    """A helper process that locks RAM until the system has only `available` bytes available (nothing if it has less), returned once it is
    done, with .locked and .wanted (bytes): less is locked when the system refuses more. Locked pages are never paged out. The RAM is freed
    when its stdin closes (release(), or this process ending)."""
    helper = subprocess.Popen([sys.executable, "-I", os.path.abspath(__file__), str(int(available))], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)  # -I: not the modules of its folder
    answer = helper.stdout.readline().split()
    if answer[:1] != ["locked"]:
        helper.wait()
        raise RuntimeError(f"The RAM stress helper could not lock the RAM: {' '.join(answer) or 'no answer'}")
    helper.locked, helper.wanted = int(answer[1]), int(answer[2])
    return helper


def release(helper):
    helper.stdin.close()
    helper.wait()


def _lock(target):
    # in the helper process: 1 GiB chunks locked until the system has only target bytes available, reported on stdout, held until stdin ends
    import psutil
    wanted = max(0, math.ceil((psutil.virtual_memory().available - target) / GB))
    if os.name == "nt":
        import ctypes.wintypes as w
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.GetCurrentProcess.restype = w.HANDLE
        k32.VirtualAlloc.restype = ctypes.c_void_p
        k32.VirtualAlloc.argtypes = (ctypes.c_void_p, ctypes.c_size_t, w.DWORD, w.DWORD)
        k32.VirtualLock.argtypes = k32.VirtualUnlock.argtypes = (ctypes.c_void_p, ctypes.c_size_t)
        k32.VirtualFree.argtypes = (ctypes.c_void_p, ctypes.c_size_t, w.DWORD)
        k32.SetProcessWorkingSetSizeEx.argtypes = (w.HANDLE, ctypes.c_size_t, ctypes.c_size_t, w.DWORD)
        # a process may lock no more than its minimum working set (a few MB by default): raised first, which any user may do
        if wanted and not k32.SetProcessWorkingSetSizeEx(k32.GetCurrentProcess(), (wanted + 1) * GB, (wanted + 2) * GB, 0):
            print(f"failed: the working set could not be raised to {wanted + 1} GiB (error {ctypes.get_last_error()})", flush=True)
            return

        def lock():
            address = k32.VirtualAlloc(None, GB, 0x3000, 0x04)  # MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE
            if address and k32.VirtualLock(address, GB):  # locking makes the pages resident
                return address
            if address:
                k32.VirtualFree(address, 0, 0x8000)
            return None

        def unlock(address):
            k32.VirtualUnlock(address, GB)
            k32.VirtualFree(address, 0, 0x8000)  # MEM_RELEASE
        error = ctypes.get_last_error
    else:
        import mmap
        libc = ctypes.CDLL(None, use_errno=True)
        libc.mmap.restype = ctypes.c_void_p
        libc.mmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_long)
        libc.mlock.argtypes = libc.munmap.argtypes = (ctypes.c_void_p, ctypes.c_size_t)

        def lock():
            address = libc.mmap(None, GB, mmap.PROT_READ | mmap.PROT_WRITE, mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS, -1, 0)
            if address in (None, ctypes.c_void_p(-1).value):
                return None
            if libc.mlock(address, GB) == 0:  # locking makes the pages resident
                return address
            libc.munmap(address, GB)
            return None

        def unlock(address):
            libc.munmap(address, GB)  # unlocks too
        error = ctypes.get_errno
    chunks = []
    while len(chunks) < wanted and (address := lock()) is not None:
        chunks.append(address)
    if wanted and not chunks:
        print(f"failed: no memory could be locked (error {error()})", flush=True)
        return
    print(f"locked {len(chunks) * GB} {wanted * GB}", flush=True)
    ended = threading.Event()
    threading.Thread(target=lambda: (sys.stdin.read(), ended.set()), daemon=True).start()
    short_since = None
    while not ended.wait(1):
        if not chunks or psutil.virtual_memory().available >= SAFETY_GB * GB:
            short_since = None
        elif short_since is None:
            short_since = time.monotonic()
        elif time.monotonic() - short_since > 10:
            unlock(chunks.pop())
            short_since = None
            print(f"[MMGP] RAM stress: the available RAM stayed under {SAFETY_GB} GB for 10 s, 1 GiB given back ({len(chunks)} GiB still locked)", file=sys.stderr, flush=True)


if __name__ == "__main__":
    _lock(int(sys.argv[1]))
