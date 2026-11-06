# Copyright (c) 2024-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Read ahead of mapped weights that a drive must read, before copies that split them across threads (Windows): cold_ranges finds them,
prefetch_ranges reads them with large requests."""
import ctypes
import mmap
import sys
import time

from tqdm import tqdm


MAX_RANGES = 128  # ranges per prefetch call, and per pinning batch
COLD_SEGMENT = 16 * 2**20  # cold_ranges checks one page of each segment of this size
COLD_TOUCH_NS = 40_000  # a first touch slower than this was read from a drive (soft faults: a few microseconds)
COLD_PIECE = 64 * 2**20  # cold copies read ahead one piece of this size at a time


class _Range(ctypes.Structure):
    _fields_ = [('address', ctypes.c_void_p), ('size', ctypes.c_size_t)]


def _windows_prefetch():
    if sys.platform != 'win32':
        return None
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    try:
        hint = kernel.PrefetchVirtualMemory
    except AttributeError:
        return None
    hint.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(_Range), ctypes.c_ulong]
    hint.restype = ctypes.c_int
    def prefetch(ranges):
        array = (_Range * len(ranges))(*(_Range(address, size) for address, size in ranges))
        if not hint(ctypes.c_void_p(-1), len(array), array, 0):
            raise ctypes.WinError(ctypes.get_last_error())
    return prefetch


class _WorkingSetEntry(ctypes.Structure):
    _fields_ = [('address', ctypes.c_void_p), ('attributes', ctypes.c_size_t)]


def _windows_cold_hints():
    # resident: in this process' working set (QueryWorkingSetEx). A page outside it may still be in RAM (standby list, e.g. read by an
    # earlier process): on_disk touches a byte of the first and last such pages, a soft fault takes a few microseconds, a read from a drive
    # 100 us and more, so that a prefetch is only asked for data that a drive must read (on standby pages it costs 40 us per MB for nothing)
    prefetch = _windows_prefetch()
    if prefetch is None:
        return None
    query = ctypes.WinDLL('kernel32', use_last_error=True).K32QueryWorkingSetEx
    query.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    query.restype = ctypes.c_int
    def resident(addresses):
        entries = (_WorkingSetEntry * len(addresses))(*(_WorkingSetEntry(address, 0) for address in addresses))
        if not query(ctypes.c_void_p(-1), entries, ctypes.sizeof(entries)):
            raise ctypes.WinError(ctypes.get_last_error())
        return [entry.attributes & 1 for entry in entries]
    def on_disk(addresses):
        for address in addresses:
            start = time.perf_counter_ns()
            ctypes.c_char.from_address(address).value
            if time.perf_counter_ns() - start > COLD_TOUCH_NS:
                return True
        return False
    return resident, prefetch, on_disk


_cold_hints = None  # (resident, prefetch, on_disk) of this system, False when it has none or after an error

# Windows: threads faulting in the same mapped file read it 64 KiB at a time, ten times slower than one sequential reader on a hard disk and
# twice on an NVMe drive, which made the weights that RAM cannot keep (a model larger than the RAM left to it) slow to stage at each step.
# Copies split across threads first read what a drive must read with prefetch_ranges (large requests), then the threads copy from RAM. Where
# RAM is short, the copy follows the prefetch closely (pieces of COLD_PIECE): data read too far ahead would lose its pages before being copied.
# Linux reads around each fault (several hundred KiB per request with the same three threads) and keeps its own behavior.


def _hints():
    global _cold_hints
    if _cold_hints is None:
        _cold_hints = (_windows_cold_hints() if sys.platform == 'win32' else None) or False
    return _cold_hints


def _disable(error):  # an optional hint: the copies remain valid without it
    global _cold_hints
    _cold_hints = False
    tqdm.write(f"[MMGP] Prefetch of the weights not in RAM disabled after an error of the system: {error}")


def cold_ranges(ranges):
    """The parts of ranges ((address, byte count), in copy order) that a drive must read, as page-aligned [start, end] lists: empty when all
    of them are in RAM or the system offers no hints. Costs a query of one page per COLD_SEGMENT, and one first touch when some are not
    in the working set (Windows)."""
    hints = _hints()
    if not hints:
        return []
    page = mmap.PAGESIZE
    segments = []
    for address, size in ranges:
        start, end = address // page * page, address + size
        for segment in range(start, end, COLD_SEGMENT):
            segments.append((segment, min(end, segment + COLD_SEGMENT)))
    if not segments:
        return []
    resident, _, on_disk = hints
    try:
        cold = []
        for (start, end), present in zip(segments, resident([start for start, _ in segments])):
            if present:
                continue
            end = (end + page - 1) // page * page
            if cold and start <= cold[-1][1]:
                cold[-1][1] = max(cold[-1][1], end)
            else:
                cold.append([start, end])
        return cold if cold and on_disk({cold[0][0], cold[-1][1] - page}) else []
    except OSError as error:
        _disable(error)
        return []


def ram_short():
    """Whether the pages of mapped weights should leave the working set: the available RAM is under the floor of the pinned pool. Elsewhere than on Windows the available RAM already counts them."""
    if sys.platform != "win32":
        return False
    import psutil
    from . import offload
    return psutil.virtual_memory().available < offload.POOL_RAM_FLOOR


def drop_behind(ranges):
    """Mapped weights (address, byte count) just copied elsewhere and not read again soon, when RAM is short: their pages go from the working set to the standby list, available to the system and to the next read without
    the drive. Without this they stay, counted as used, until Windows trims the working set, recent pages included. About 75 ms per GiB, on the thread that copied them."""
    if ram_short():
        from . import offload
        offload._drop_mapped_pages(offload._mapped_pages(ranges))


def prefetch_ranges(ranges):
    """Reads ranges ([start, end] address pairs) into RAM with large requests."""
    hints = _hints()
    if not hints or not ranges:
        return
    page = mmap.PAGESIZE
    ranges = [(start // page * page, (end + page - 1) // page * page) for start, end in ranges]
    try:
        for index in range(0, len(ranges), MAX_RANGES):
            hints[1]([(start, end - start) for start, end in ranges[index:index + MAX_RANGES]])
    except OSError as error:
        _disable(error)
