# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""A smaller GPU simulated under real pressure, as ram_stress simulates a smaller PC: the card is emulated (mmgp.virtual_gpu: every VRAM
figure is its own), and a helper process holds VRAM until the GPU has only what such a card leaves free beside the desktop, a browser and a
few programs (VRAM_OVERHEAD_GB unless given), the VRAM other programs already use counting. The helper's memory is written (NVML counts
memory once written) and stays idle, as the desktop's: MMGP sees it in the whole GPU's free VRAM (NVML), and on Windows the driver may page
it out to RAM when an allocation still finds the VRAM short after MMGP's own recoveries, as it would the desktop's. An emulated card alone
counts the other programs only when they leave less than the card. Its limit (card - virtual_gpu.LIMIT_OVERHEAD_GB) still applies: the
other programs count when they leave less. The helper holds its VRAM until this process ends (it waits for the end of its stdin, a pipe
from this process).

    from mmgp import vram_stress; vram_stress.start(16)       # before the process reads or allocates VRAM (WanGP: --vram-stress 16)
    from mmgp import vram_stress; vram_stress.start(16, 1.9)  # 1.9 GB used by other programs (WanGP: --vram-stress 16:1.9)

The helper uses the CUDA driver and NVML libraries installed with the NVIDIA driver, neither PyTorch nor pynvml."""
import ctypes
import os
import subprocess
import sys

GB = 2**30
VRAM_OVERHEAD_GB = 1.5 if os.name == "nt" else 0.5  # the VRAM the desktop, a browser and a few programs use on such a card
CHUNK = 256 << 20  # allocations of the helper, written one after the other
ALIGN = 2 << 20
state = {}  # helper (process), held (bytes) once started


def start(card_gb, others_gb=None, device=0):
    """Simulates a card of card_gb (GiB) of which other programs use others_gb (VRAM_OVERHEAD_GB by default), on CUDA device `device`,
    before the process reads or allocates VRAM."""
    from . import virtual_gpu
    if state:
        raise RuntimeError("The VRAM stress is already started.")
    others = VRAM_OVERHEAD_GB if others_gb is None else float(others_gb)
    virtual_gpu.install(card_gb)
    helper = subprocess.Popen([sys.executable, "-I", os.path.abspath(__file__), str(int(others * GB)), str(int(float(card_gb) * GB)), str(int(device))], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)  # -I: not the modules of its folder
    answer = helper.stdout.readline().split()
    if answer[:1] != ["held"]:
        helper.wait()
        raise RuntimeError(f"The VRAM stress helper could not hold the VRAM: {' '.join(answer) or 'no answer'}")
    held, free = map(int, answer[1:])
    card = int(float(card_gb) * GB)
    state.update(helper=helper, held=held)
    print(f"[MMGP] VRAM stress: a {float(card_gb):g} GB GPU simulated, {(card - free) / GB:.1f} GiB of it used by other programs ({held / GB:.1f} GiB held by a "
          f"helper process), {free / GB:.1f} GiB free", flush=True)
    if free > card - others * GB + CHUNK:
        print(f"[MMGP] VRAM stress: only {held / GB:.1f} GiB could be held: more VRAM stays free than on that GPU", flush=True)


def release():
    helper = state.pop("helper")
    state.clear()
    helper.stdin.close()
    helper.wait()


class _Memory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


def _hold(used, card, ordinal):
    # in the helper process: VRAM allocated and written until the GPU has only card - used free, reported on stdout, held until stdin ends
    cuda, nvml = (ctypes.WinDLL("nvcuda.dll"), ctypes.WinDLL("nvml.dll")) if os.name == "nt" else (ctypes.CDLL("libcuda.so.1"), ctypes.CDLL("libnvidia-ml.so.1"))
    cuda.cuDeviceGet.argtypes = (ctypes.POINTER(ctypes.c_int), ctypes.c_int)
    cuda.cuDeviceGetPCIBusId.argtypes = (ctypes.c_char_p, ctypes.c_int, ctypes.c_int)
    cuda.cuDevicePrimaryCtxRetain.argtypes = (ctypes.POINTER(ctypes.c_void_p), ctypes.c_int)
    cuda.cuCtxSetCurrent.argtypes = (ctypes.c_void_p,)
    cuda.cuMemAlloc_v2.argtypes = (ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t)
    cuda.cuMemsetD8_v2.argtypes = (ctypes.c_uint64, ctypes.c_ubyte, ctypes.c_size_t)
    nvml.nvmlDeviceGetHandleByPciBusId_v2.argtypes = (ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p))
    nvml.nvmlDeviceGetMemoryInfo.argtypes = (ctypes.c_void_p, ctypes.POINTER(_Memory))

    def check(name, result):
        if result != 0:
            print(f"failed: {name} returned error {result}", flush=True)
            sys.exit(1)
    device, bus, context, handle, memory = ctypes.c_int(), ctypes.create_string_buffer(64), ctypes.c_void_p(), ctypes.c_void_p(), _Memory()
    check("cuInit", cuda.cuInit(0))
    check("cuDeviceGet", cuda.cuDeviceGet(ctypes.byref(device), ordinal))
    check("cuDeviceGetPCIBusId", cuda.cuDeviceGetPCIBusId(bus, len(bus), device))
    check("cuDevicePrimaryCtxRetain", cuda.cuDevicePrimaryCtxRetain(ctypes.byref(context), device))
    check("cuCtxSetCurrent", cuda.cuCtxSetCurrent(context))  # the helper's context counts among the other programs' VRAM
    check("nvmlInit", nvml.nvmlInit_v2())
    check("nvmlDeviceGetHandleByPciBusId", nvml.nvmlDeviceGetHandleByPciBusId_v2(bus.value, ctypes.byref(handle)))

    def free():  # the whole GPU's, every process counting
        check("nvmlDeviceGetMemoryInfo", nvml.nvmlDeviceGetMemoryInfo(handle, ctypes.byref(memory)))
        return memory.free
    need, held, pointers = free() - (card - used), 0, []  # (a card larger than the real one: nothing held)
    while need - held >= ALIGN:
        size, pointer = min(CHUNK, (need - held) // ALIGN * ALIGN), ctypes.c_uint64()
        if cuda.cuMemAlloc_v2(ctypes.byref(pointer), size) != 0:  # the GPU refused more: less is held
            break
        check("cuMemsetD8", cuda.cuMemsetD8_v2(pointer, 1, size))
        pointers.append(pointer)
        held += size
    check("cuCtxSynchronize", cuda.cuCtxSynchronize())
    print(f"held {held} {free()}", flush=True)
    sys.stdin.read()  # the end of this process frees the VRAM


if __name__ == "__main__":
    _hold(*map(int, sys.argv[1:]))
