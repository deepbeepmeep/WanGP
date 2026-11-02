# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Coordinate reclaimable VRAM: optional expert copies precede streamed blocks.

The clients release VRAM only on the thread that runs their blocks. A helper thread (the prefetch thread, whose staged transfers allocate
the blocks they copy) hands its shortages to the thread waiting for its work (wait), which releases on its behalf.

Parked offload objects (offload.park: their models stay loaded while another one uses the GPU) give their VRAM first, from any thread:
expert copies and state kept in VRAM, preloaded blocks, the least recently used models first (stamp), then whole models or their owner's
release. The running objects come next."""
import itertools
import threading
import weakref

from . import allocator

_clients = weakref.WeakKeyDictionary()
HEADROOM = 256 << 20  # the allocator leaves this much free beyond a new allocation (vram_headroom)
_helpers = set()  # idents of the helper threads
_requests = []  # [size, device, done, freed] of the helpers' shortages, released by the thread waiting for their work
_ready = threading.Condition()
_stamps = itertools.count(1)
_parked = weakref.WeakSet()  # the offload objects parked


def stamp():
    """A process-wide use counter: the models record it when they run, the least recently used give their VRAM first."""
    return next(_stamps)


def _manager(client):
    manager = getattr(client, "manager", None)
    return manager() if isinstance(manager, weakref.ref) else manager


def clients_of(manager):
    """The clients of an offload object, in their priority order."""
    return [client for client, _ in sorted(list(_clients.items()), key=lambda item: item[1]) if _manager(client) is manager]


def parked():
    return [manager for manager in list(_parked) if manager.parked]


def reclaimable(exclude=None):
    """VRAM the parked offload objects (but exclude) would give back: what the automatic preloads may count as theirs."""
    return sum(manager.parked_bytes() for manager in parked() if manager is not exclude)


def _reclaim(size, device):
    # what the allocation of size bytes misses, the VRAM it may still take deducted (its size when the room did not refuse it)
    if threading.get_ident() in _helpers:
        request = [size, device, threading.Event(), 0]
        with _ready:
            _requests.append(request)
            _ready.notify_all()
        request[2].wait()
        return request[3]
    room = allocator.room(device)
    size = size + HEADROOM - room if room < size + HEADROOM else size
    freed = 0
    for manager in sorted(parked(), key=lambda manager: manager.last_used()):
        freed += manager.reclaim_parked(size - freed)
        if freed >= size:
            return freed
    # then the running objects: by priority, the least recently used object first in each (an idle one before the one allocating)
    running = [(priority, client, _manager(client)) for client, priority in list(_clients.items())]
    for _, client, manager in sorted(running, key=lambda item: (item[0], 0 if item[2] is None else item[2].last_used())):
        if manager is not None and manager.parked:
            continue
        freed += client.on_pressure(max(0, size - freed))
        if freed >= size:
            break
    return freed


def helper_thread():
    """Marks the calling thread as a helper (a thread pool's initializer)."""
    _helpers.add(threading.get_ident())


def wait(future):
    """future.result() of a helper's work, releasing meanwhile the VRAM its allocations find short (on this thread, which runs the blocks)."""
    future.add_done_callback(lambda _: _notify())
    while True:
        with _ready:
            _ready.wait_for(lambda: future.done() or _requests)
            requests = _requests[:]
            _requests.clear()
        for request in requests:
            try:
                request[3] = _reclaim(request[0], request[1])
            finally:
                request[2].set()
        if future.done() and not requests:
            return future.result()


def _notify():
    with _ready:
        _ready.notify_all()


def _update_callback():
    if allocator.active == "vmm":
        allocator.set_pressure_callback(_reclaim if _clients or parked() else None)


def register(client, priority=1):
    _clients[client] = priority
    _update_callback()


def unregister(client):
    _clients.pop(client, None)
    _update_callback()


def park(manager):
    _parked.add(manager)
    _update_callback()


def unpark(manager):
    _parked.discard(manager)
    _update_callback()
