# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""CPU-backed expert banks with independently reclaimable GPU expert copies.

An ExpertBank registers checkpoint tensors whose first dimension is the expert
index. MMGP leaves these tensors on pageable CPU memory, outside normal block
swapping/pinning. Its residency controller owns every optional GPU copy. Compute
backends use resident_tensors() and lease(); they never transfer model weights.
"""
import bisect
import collections
import contextlib
import ctypes
import functools
import itertools
import os
import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from . import allocator, cuda_graphs, memory_pressure, pinned, runtime
from . import expert_stream
from .expert_stream import ExpertStream, REGION_BYTES
from .read_ahead import cold_ranges, drop_behind, prefetch_ranges, ram_short

MARGIN = 512 << 20
UPLOAD_STAGING = 32 << 20  # bytes of each of the two pinned buffers through which new GPU copies go
ADAPT_PASSES = 16  # decoding passes between two adaptations of the placement to the routing
ADAPT_BYTES_PER_PAIR = 16 << 10  # what an adaptation may upload per (token, slot) pair routed since the last one: a pass of a token over 48 layers lets about 8 MiB in
ADAPT_MAX_BYTES = 2 << 30  # at once, after a long prompt
ELASTIC = "placement" not in os.environ.get("WGP_FN_DISABLE", "").split(",")  # the elastic placement of the banks of models following their routing (_lm_engine_adaptive_experts), until proven
ELASTIC_EVERY = 4  # decoding passes between two rounds, the next pass after a round that took all its swaps: a topic that changes. Each round stops the GPU for about a millisecond, and a text staying on its topic moves few experts
ELASTIC_SWAPS = 32  # experts copied per round at most
ELASTIC_FILL = int(os.environ.get("WGP_FN_FILL", 256))  # experts copied per round at most while free VRAM takes them with no other leaving: after a prompt, the first tokens of an answer are the slowest. First answer after 8K tokens at 32, 128, 512: 76, 94, 100 tok/s
ELASTIC_SLAB = 256 << 20  # GPU copies live in slabs of slots of one size, given back whole: experts of many sizes freed at random in the allocator's small blocks leave holes it cannot give back (3.7 GiB seen on a 12 GB card), and of slabs of 64 MiB in its mid-size pool a third stayed there. From 256 MiB its allocations are chunks it can give back
ELASTIC_REBALANCE = 32  # rounds between two moves of a slab from the size whose experts are worth least to the one whose missing experts are worth most
ELASTIC_DECAY = float(os.environ.get("WGP_FN_DECAY", 0.9)), 0.995  # per pass, of the two routing counts of an expert: the topic of the last passes, what this text always reads
ELASTIC_SLOW_SHARE = float(os.environ.get("WGP_FN_SLOW", 0.2))  # the second count's weight in the value of an expert. Replays of real routing, a text staying on its topic: at 15 GiB of experts 25 misses per token against 107 with 0.02 and a gain of 5; texts changing topics 187 against 206
ELASTIC_PROMPT_SLOW = int("prompt_slow" in os.environ.get("WGP_FN_ENABLE", "").split(","))  # a prompt's routing counts reach the topic's count only, not the one of what the text always reads: a system prompt of tools before a story kept 4-6 CPU pairs per token more, 134 -> 142 tok/s (WGP_FN_ENABLE=prompt_slow: as before)
ELASTIC_TIE = 1e-3  # what the ranking weighs beside the routing counts: it decides between experts of equal counts
ELASTIC_GAIN = float(os.environ.get("WGP_FN_GAIN", 3.0))  # an expert takes the place of another when it is worth this many pairs more (of an average expert)
ELASTIC_KEEP = 1 << 30  # VRAM the placement leaves free when it grows: what the allocator keeps free beyond its allocations, and small allocations of the driver
ELASTIC_LOW = 640 << 20  # below this it gives VRAM back, whatever took it: another program, the graphs of a new shape
ELASTIC_FIRST_MARGIN = 1 << 30  # VRAM the first placement leaves free for the first passes
ELASTIC_TRIM = 128 << 20  # VRAM the allocator holds unused from which it is asked to give it back, when the room is short
ELASTIC_TRIM_ROUNDS = 64  # rounds between two such requests at most
LEVEL2 = "level2" in os.environ.get("WGP_FN_ENABLE", "").split(",")  # experts out of VRAM in locked RAM (pinned pool entries), and the GPU's share of each layer's misses copied from there by the host jobs, until proven
# level 2 and the share only where VRAM holds less than this part of the experts' bytes: with 47 % (Flash-Next, 32 GB card) a layer misses 0.4 expert
# on average, nothing for the copy engine to take beside the CPU, and the share's wait for each job's rows and level 2's rounds cost 25 % of decoding;
# with 12 % (16 GB card) a layer misses 5-7 of them
LEVEL2_VRAM_BELOW = float(os.environ.get("WGP_FN_LEVEL2_BELOW", 0.33))
LEVEL2_BYTES = int(float(os.environ.get("WGP_FN_LEVEL2_GIB", 8)) * 2**30)  # what they lock at most, within the pinned pool's room
LEVEL2_ENTRY = 128 << 20  # an entry of the pinned pool: slots of one size, in the layout of a VRAM slot
LEVEL2_ROUND = int(os.environ.get("WGP_FN_LEVEL2_ROUND", 16))  # experts copied into level 2 per round at most (each about 2 MiB, by a helper process)
LEVEL2_SWAPS = 4  # copies a full level 2 replaces per round at most
SHARE_PERCENT = int(os.environ.get("WGP_FN_SHARE", 50))  # of a layer's experts missing VRAM, the share the GPU computes at most: those with a locked copy, the most read first
SHARE_SLOTS = 16  # of the VRAM ring the share is copied into
SHARE_MODE = os.environ.get("WGP_FN_SHARE_MODE", "copies")  # copies: the copy engine brings the share into a ring of VRAM; mapped: the GPU reads it in place in mapped RAM, over the bus (no copy engine)


class ExpertBank(nn.Module):
    def __init__(self, num_experts, tensor_names):
        super().__init__()
        self.num_experts = int(num_experts)
        self.expert_tensor_names = tuple(tensor_names)
        self._residency = None
        self._resident = {}
        self.residency_version = 0
        self._leases = 0
        self.keep_resident = False  # a placement that follows the routing does not move these experts out for others
        # 1 for the experts with GPU copies, read by the CPU backend; a plain tensor, outside the module's state
        self.resident_flags = torch.zeros(self.num_experts, dtype=torch.uint8, device="cpu")
        self._flags = self.resident_flags.numpy()
        # Optional placement priority per expert, higher first and comparable across banks (e.g. measured pick frequencies); set by
        # the model before MMGP takes it over. None keeps the default order.
        self.expert_priority = None
        # Pairs routed to each expert, decaying, when the format's compute backend counts them (reports_usage): the routing follows the
        # topic of the text, which a fixed ranking cannot know. A plain tensor, outside the module's state.
        self.reports_usage = False
        self.usage = torch.zeros(self.num_experts, dtype=torch.uint32, device="cpu")
        self._usage = self.usage.numpy()
        self.gpu_branch = False  # graphs run its GPU branch even when it has no GPU copy at capture: the elastic placement may give it some later, in place

    def _apply(self, fn, recurse=True):
        # the checkpoint tensors stay in RAM in their stored dtypes whatever moves or casts the model (to("cuda"), half()): the residency owns the GPU copies
        return self

    @property
    def expert_nbytes(self):
        return sum(getattr(self, name)[0].numel() * getattr(self, name).element_size() for name in self.expert_tensor_names)

    def resident_tensors(self):
        """Expert id to checkpoint-name -> GPU tensor; callers must not mutate it."""
        return self._resident

    def release_execution_cache(self):
        """Optional backend scratch; called when the owning model is unloaded."""

    @contextlib.contextmanager
    def lease(self):
        """Keep this bank's copies alive while its compute backend submits work."""
        self._leases += 1
        try:
            yield self._resident
        finally:
            self._leases -= 1

    def _place(self, expert, tensors):
        self._resident[expert] = tensors
        self._flags[expert] = 1
        self.residency_version += 1
        if self.gpu_branch:
            self._residency().write_row(self, expert, tensors)

    def _evict(self, expert):
        del self._resident[expert]
        self._flags[expert] = 0
        self.residency_version += 1
        if self.gpu_branch:
            self._residency().write_row(self, expert, None)

    def _swap(self, old, new):
        """The GPU copy of old now holds new (the same memory, rewritten)."""
        self._resident[new] = self._resident.pop(old)
        self._flags[old], self._flags[new] = 0, 1
        self.residency_version += 1

    def add_usage(self, ids):
        """Counts a pass that its compute backend does not count: a prompt chunk, whose routing announces what decoding will read."""
        self._usage += np.bincount(ids.reshape(-1).cpu().numpy(), minlength=self.num_experts).astype(np.uint32)
        residency = None if self._residency is None else self._residency()
        if residency is not None:
            residency.adapt_due = True


def banks(model):
    return {name: module for name, module in model.named_modules() if isinstance(module, ExpertBank)}


def excluded_names(model):
    prefixes = tuple(banks(model))
    return {name for name, _ in model.named_modules() if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)}


_memcpy = None


def _stream_handle(stream):
    """The handle of a stream for direct copies from locked RAM (a round makes dozens, and PyTorch's copy_ costs the decoding thread its views and dispatch), None on HIP, where they go through PyTorch."""
    global _memcpy
    if torch.version.hip is not None:
        return None
    if _memcpy is None:
        _memcpy = ctypes.CDLL("nvcuda.dll" if os.name == "nt" else "libcuda.so.1").cuMemcpyHtoDAsync_v2
        _memcpy.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        _memcpy.restype = ctypes.c_int
    return stream.cuda_stream


def _copy_moves(base, moves):
    """In a helper thread: (offset in the buffer, source address, bytes). Sources that a drive must read are read first with large requests (touching them would read 64 KiB at a time)."""
    cold = cold_ranges([(source, nbytes) for _, source, nbytes in moves])
    if cold:
        prefetch_ranges(cold)
    for offset, source, nbytes in moves:
        ctypes.memmove(base + offset, source, nbytes)
    drop_behind([(source, nbytes) for _, source, nbytes in moves])  # the GPU's copies replace them


def _efficient_cores():
    """On a hybrid CPU (Windows), its threads go to the least efficient cores, where no CPU expert worker waits for them: the workers' pool waits for its
    slowest worker at each phase, and a copy taking a performance core from one held a whole job (0.3 -> 1.7 ms per job at 160 GB)."""
    if os.name != "nt":
        return
    kernel32 = ctypes.WinDLL("kernel32")
    size = ctypes.c_uint32(0)
    kernel32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))  # RelationProcessorCore
    buffer = ctypes.create_string_buffer(size.value)
    if not size.value or not kernel32.GetLogicalProcessorInformationEx(0, buffer, ctypes.byref(size)):
        return
    raw, masks, offset = buffer.raw, {}, 0
    while offset < size.value:
        record = int.from_bytes(raw[offset + 4:offset + 8], "little")
        efficiency, groups = raw[offset + 9], int.from_bytes(raw[offset + 30:offset + 32], "little")
        for group in range(groups):
            at = offset + 32 + 16 * group
            if int.from_bytes(raw[at + 8:at + 10], "little") == 0:
                masks[efficiency] = masks.get(efficiency, 0) | int.from_bytes(raw[at:at + 8], "little")
        offset += record
    if len(masks) > 1:
        kernel32.GetCurrentThread.restype = ctypes.c_void_p
        kernel32.SetThreadAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        kernel32.SetThreadAffinityMask(kernel32.GetCurrentThread(), masks[min(masks)])


class _Level2:
    """Copies of experts out of VRAM in locked RAM (mmgp level 2), filled as the routing reads them: entries of the pinned pool of one slot size each,
    empty when made, whose slots the experts worth most that are neither in VRAM nor here take, a few per round. With a copy helper (sections of level 2)
    the helper copies them from their files: its page faults and drive reads hold nothing of this process, and no thread of this process runs for them (a
    copying thread here cost about 8 % of decoding at 160 GB, where nothing faults: the GIL, working-set queries). Otherwise a thread of its own copies them,
    on the efficient cores of a hybrid CPU, their checkpoint pages then dropped when RAM is short. One copy per expert: an expert placed in VRAM, or worth much
    less than one waiting, leaves, after a round's synchronization: nothing in flight reads a slot then. Per expert, read by the host jobs: the address of its
    copy (sources), its section and offset for a copy helper (remote), its device address for a GPU reading it in place (device), 0 without."""
    def __init__(self, residency, total):
        self.residency = weakref.ref(residency)
        self.sources, self.remote, self.device = (np.zeros(total, dtype=np.uint64) for _ in range(3))
        self.slot_of = np.full(total, -1, dtype=np.int64)
        self.pending = set()  # experts whose copy is on its way
        self.slots, self.free = [], [[] for _ in residency.slot_bytes]  # slot -> [entry name, address, offset, owner]; free slots of each size
        self.entries, self.making, self.names = {}, set(), 0  # entry name -> (class, its slots, section, device base); the sizes whose entry is being made
        self.guard = threading.Lock()
        self.copier = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mmgp_level2", initializer=_efficient_cores)
        self.filling, self.files = [], {}  # the helper's fills on their way: (sequence number, expert, slot); per bank, its parts' (offset, file mapping, offset in the file, bytes), or None
        self.counts = dict(rounds=0, copies=0, left=0, entries=0, refused=0, copy_seconds=0.0, fills=0, failed=0)

    def nbytes(self):
        residency = self.residency()
        return sum(len(slots) * residency.slot_bytes[klass] for klass, slots, _, _ in self.entries.values())

    def round(self, value, sync=False):
        """At a round: the experts worth most out of VRAM and out of here take free slots, LEVEL2_ROUND at most; an entry is asked of the pool when a size has
        no free slot left, and when the pool holds no more (sync: after a synchronization of the GPU, nothing in flight reads a slot) a copy of an expert in
        VRAM, else one worth much less, makes room. Copies of experts placed in VRAM stay until then: the rounds move experts in and out of VRAM all the
        time, and each copy that left with its expert was copied again from the file when it came back (at 64 GB, 56K copies in 13 min, 170 MB/s of
        drive reads, the CPU's experts faulting behind them). Those worth enough to take a place in VRAM at the next two rounds (the first
        2 x ELASTIC_SWAPS missing ones) are left to them."""
        residency = self.residency()
        self.counts["rounds"] += 1
        with self.guard:  # the ring read under it: once _free_share has settled, no round asks a closed ring
            share = residency.share
            ring = share[0] if share is not None and share[0] is not None and share[0].remote else None
            if ring is not None:
                self._settle(ring)
            missing = np.flatnonzero((residency.all_flags == 0) & (value > 0))
            missing = missing[np.argsort(-value[missing], kind="stable")][2 * ELASTIC_SWAPS:]  # the next two rounds' promotions left to them
            candidates = missing[self.slot_of[missing] < 0][:LEVEL2_ROUND]
            if not len(candidates):
                return
            copies, replaced = [], 0
            for index in candidates.tolist():
                if index in self.pending:
                    continue
                klass = int(residency.class_of[index])
                if not self.free[klass]:
                    if klass not in self.making and self.nbytes() + LEVEL2_ENTRY * (1 + len(self.making)) <= LEVEL2_BYTES and self._make_entry(klass):
                        continue  # its slots come with the entry
                    if not sync or replaced >= LEVEL2_SWAPS:
                        continue
                    held = np.flatnonzero((self.slot_of >= 0) & (self.sources != 0) & (residency.class_of == klass))
                    if not len(held):
                        continue
                    # the copy worth least makes room for one worth much more, a copy of an expert in VRAM counting half (read once it comes back): full, it
                    # replaced up to 16 copies per round, which came back from the file soon after (31K copies in 3 answers at 64 GB)
                    worth = value[held] * np.where(residency.all_flags[held] != 0, 0.5, 1.0)
                    victim = int(np.argmin(worth))
                    if value[index] <= 1.5 * worth[victim] + 1:
                        continue
                    self._leave(int(held[victim]))
                    replaced += 1
                slot = self.free[klass].pop()
                self.slots[slot][3] = index
                self.slot_of[index] = slot
                self.pending.add(index)
                copies.append((index, slot))
            if ring is not None:
                copies = self._fill(ring, copies)
        for index, slot in copies:
            self.copier.submit(self._copy, index, slot)

    def _file_moves(self, bank):
        # its parts as offsets in their files, None unless each is in a view of a whole file (gguf)
        if bank not in self.files:
            from . import offload
            residency = self.residency()
            moves = [(start, offload.file_mapping(source), nbytes) for start, source, nbytes in residency.sources[bank]]
            self.files[bank] = None if any(found is None for _, found, _ in moves) else [(start, found[0], found[1], nbytes) for start, found, nbytes in moves]
        return self.files[bank]

    def _fill(self, ring, copies):
        """Under the guard: the copies the helper makes from the files, asked of it; returns the others (for the thread of this process)."""
        residency, fills, posted, others = self.residency(), [], [], []
        drop = ram_short()
        for index, slot in copies:
            bank = residency.banks[residency.bank_of[index]]
            moves, name = self._file_moves(bank), self.slots[slot][0]
            section = self.entries[name][2]
            if moves is None or section is None:
                others.append((index, slot))
                continue
            expert = index - residency.first[bank]
            fills.append((section, self.slots[slot][2], [(mapping, offset + expert * nbytes, start, nbytes) for start, mapping, offset, nbytes in moves], drop))
            posted.append((index, slot))
        first = ring.fill(fills) if fills else None
        if first is None:  # its queue is full: free slots again, the next rounds ask again
            for index, slot in posted:
                self._release(index, slot)
            return others
        self.filling.extend((first + n, index, slot) for n, (index, slot) in enumerate(posted))
        return others

    def _settle(self, ring, abandon=False):
        """Under the guard: the helper's fills done are published (the slot still theirs), those that failed free their slot; abandon: those still on their way too
        (their ring closes)."""
        if not self.filling:
            return
        first = self.filling[0][0]
        done, results, (seconds, _) = ring.filled(first, self.filling[-1][0] + 1 - first)
        self.counts["copy_seconds"] = seconds
        waiting = []
        for sequence, index, slot in self.filling:
            if sequence >= done:
                if abandon:
                    self._release(index, slot)
                else:
                    waiting.append((sequence, index, slot))
                continue
            if results[sequence - first] != 0:
                self.counts["failed"] += 1
                self._release(index, slot)
            elif self.slots[slot][3] == index and self.slots[slot][0] in self.entries:
                self.pending.discard(index)
                self.counts["fills"] += 1
                self._publish(index, slot)
        self.filling = waiting

    def settle(self, ring):
        """Before the helper's ring closes: its fills done published, the others given up."""
        with self.guard:
            self._settle(ring, abandon=True)
        self.files.clear()  # the mappings go with the ring (offload.close_file_mappings)

    def _release(self, index, slot):
        self.pending.discard(index)
        if self.slots[slot][3] == index and self.slots[slot][0] in self.entries:
            self.slot_of[index] = -1
            self.slots[slot][3] = -1
            self.free[self.entries[self.slots[slot][0]][0]].append(slot)

    def _publish(self, index, slot):
        _, address, offset, _ = self.slots[slot]
        _, _, section, device = self.entries[self.slots[slot][0]]
        if section is not None:
            self.remote[index] = (section << 32) | offset
        if device is not None:
            self.device[index] = device + offset
        self.sources[index] = address + offset  # last: a job takes it once all its addresses are there

    def _copy(self, index, slot):
        residency = self.residency()
        if residency is None:
            return
        bank = residency.banks[residency.bank_of[index]]
        expert = index - residency.first[bank]
        name, address, offset, _ = self.slots[slot]
        started = time.perf_counter()
        _copy_moves(address + offset, [(start, source + expert * nbytes, nbytes) for start, source, nbytes in residency.sources[bank]])
        self.counts["copies"] += 1
        self.counts["copy_seconds"] += time.perf_counter() - started
        with self.guard:
            self.pending.discard(index)
            if self.slots[slot][3] != index or name not in self.entries:  # its entry left meanwhile
                return
            self._publish(index, slot)

    def _leave(self, index):
        self.counts["left"] += 1
        slot = int(self.slot_of[index])
        self.sources[index] = self.remote[index] = self.device[index] = 0
        self.slot_of[index] = -1
        self.slots[slot][3] = -1
        self.free[self.entries[self.slots[slot][0]][0]].append(slot)

    def _make_entry(self, klass):
        """An entry of slots of a size asked of the pool: False if it takes no more until the next quiet point."""
        residency = self.residency()
        manager, size = residency.manager(), residency.slot_bytes[klass]
        sections = pinned.shared_block if SHARE_MODE == "copies" and residency.banks[0].format.share_sections() else None  # a helper process copies them
        flags = 2 if SHARE_MODE == "mapped" else 0  # mapped for the GPU, which reads them in place
        self.names += 1
        name = f"{residency.model_id}/level2/{self.names}"
        count = LEVEL2_ENTRY // size
        # empty when made: its slots take the place of checkpoint pages as they fill, which leave the working set (added 0)
        future = manager.pool_experts(residency.model_id, name, count * size, lambda buffer, helpers: None, functools.partial(self._entry_out, name), 0, sections, flags)
        if future is None:
            self.counts["refused"] += 1
            return False
        self.making.add(klass)
        future.add_done_callback(functools.partial(self._entry_in, name, klass, count, size, flags == 2))
        return True

    def _entry_in(self, name, klass, count, size, mapped_ram, future):
        # on the pool's thread
        residency = self.residency()
        with self.guard:
            self.making.discard(klass)
            if not future.result() or residency is None:
                self.counts["refused"] += 1
                return
            self.counts["entries"] += 1
            base = residency.manager().pool_source(name).data_ptr()
            section = pinned.section_of(base)
            device = None
            if mapped_ram:
                from . import mapped
                device = mapped._device_pointer(base)
            first = len(self.slots)
            self.slots.extend([name, base, n * size, -1] for n in range(count))
            self.entries[name] = (klass, list(range(first, first + count)), section, device)
            self.free[klass].extend(range(first, first + count))
        if section is not None and residency.share is not None and residency.share[0] is not None:
            residency.share[0].prepare([section])  # the helper maps it now, before its first copy

    def _entry_out(self, name):
        # at a quiet point, before the entry's RAM goes: its experts are read in the checkpoint again, and a copy helper lets its section go
        residency = self.residency()
        with self.guard:
            if name not in self.entries:
                return
            klass, slots, section, _ = self.entries.pop(name)
            for slot in slots:
                index = self.slots[slot][3]
                if index >= 0:
                    self.sources[index] = self.remote[index] = self.device[index] = 0
                    self.slot_of[index] = -1
                    self.pending.discard(index)
                self.slots[slot] = [None, 0, 0, -1]
            self.free[klass] = [slot for slot in self.free[klass] if self.slots[slot][0] is not None]
        if section is not None and residency is not None and residency.share is not None and residency.share[0] is not None:
            residency.share[0].forget(section)


_Batch = collections.namedtuple("_Batch", "future plan removals staging slot")  # experts on their way to a locked buffer: their placement waits for the next round


def _smallest(values, count):
    """The places of the count smallest values, in increasing order, without sorting them all."""
    if values.size <= count:
        return np.argsort(values, kind="stable")
    places = np.argpartition(values, count - 1)[:count]
    return places[np.argsort(values[places], kind="stable")]


class _Slab:
    """GPU memory for experts of one size. nbytes: what its slots may use, a little less than the tensor."""
    __slots__ = ("tensor", "base", "nbytes", "size", "members", "id")

    def __init__(self, size, identifier):
        self.tensor = torch.empty(ELASTIC_SLAB, dtype=torch.uint8, device="cuda")
        self.base, self.nbytes, self.size, self.members, self.id = self.tensor.data_ptr(), ELASTIC_SLAB - 4096, size, set(), identifier


PAGES_RELEASED = 512  # about 1 GiB: fewer experts are not worth a call


class _Slot:
    """The GPU copies of an expert in a slab: a mapping of its tensor names to GPU tensors, made when asked (decoding reads addresses)."""
    __slots__ = ("slab", "offset", "parts")

    def __init__(self, slab, offset, parts):
        self.slab, self.offset, self.parts = slab, offset, parts

    def __getitem__(self, name):
        for part, start, nbytes, dtype, shape in self.parts:
            if part == name:
                return self.slab.tensor[self.offset + start:self.offset + start + nbytes].view(dtype).view(shape)
        raise KeyError(name)

    def __iter__(self):
        return (part[0] for part in self.parts)

    def keys(self):
        return list(self)

    def values(self):
        return [self[name] for name in self]

    def items(self):
        return [(name, self[name]) for name in self]


class Residency:
    def __init__(self, manager, model_id, expert_banks, budget, dynamic):
        self.manager = weakref.ref(manager)
        self.model_id = model_id
        self.banks = list(expert_banks.values())
        self.sizes = {bank: bank.expert_nbytes for bank in self.banks}
        self.budget = budget
        self.dynamic = dynamic
        self.thread = None
        self.changing = False
        self.loaded = {}  # (bank, expert) with a GPU copy, in ranking order
        self.nbytes = 0
        self.applied = None  # number of ranked experts of the placement prepare() applied; None once the placement changed otherwise
        self.started = False
        self.pressure_ceiling = None
        self.pinned = {}  # (bank, expert) -> checkpoint name -> pinned copy, the source of its uploads (pin())
        self._pinned_holder = None
        self.upload_staging = [pinned.Buffer("experts") for _ in range(2)]  # the two locked buffers of _upload, kept until close()
        self.stream, self.stream_staging = None, [pinned.Buffer("experts") for _ in range(2)]  # the pass being read through an ExpertStream, its two locked buffers
        self.pool_parts, self.pool_names, self.pool_seen = {}, 0, {}  # bank -> [(entry name in the pinned pool, experts, future of its entry)]: the batches it streamed that stay locked; entry name -> (the share of it read by a short pass, the tokens of that pass)
        self.pages_released, self.pages_checked, self.pages_thread = set(), 0.0, None  # release_copied_pages: the experts done, when RAM was last checked, the thread at work
        self.adapt_calls, self.adapt_due, self.adapt_counted, self.swapped = 0, False, 0, 0
        self.rank_of = None
        self.elastic = ELASTIC and all(bank.reports_usage for bank in self.banks) and getattr(manager.models[model_id], "_lm_engine_adaptive_experts", False)
        # Every prefix spreads individual experts across layers; no whole-bank minimum. With priorities, the most used experts of all
        # layers come first (equal priorities keep this order).
        self.ranking = [(bank, expert) for expert in range(max(bank.num_experts for bank in self.banks)) for bank in self.banks if expert < bank.num_experts]
        if any(bank.expert_priority is not None for bank in self.banks):
            order = {item: position for position, item in enumerate(self.ranking)}
            priority = lambda item: 0.0 if item[0].expert_priority is None else float(item[0].expert_priority[item[1]])
            self.ranking.sort(key=lambda item: (-priority(item), order[item]))
        self.cumulative = list(itertools.accumulate(self.sizes[bank] for bank, _ in self.ranking))
        for bank in self.banks:
            bank._residency = weakref.ref(self)
        if self.elastic:
            self._init_elastic()
        memory_pressure.register(self, priority=0)

    def close(self):
        if self.pages_thread is not None:
            self.pages_thread.join()
        self.unload()
        self.pinned, self._pinned_holder, self.pool_parts = {}, None, {}
        for buffer in self.upload_staging + self.stream_staging:
            buffer.release()
        memory_pressure.unregister(self)
        if self.elastic:
            allocator.before_graph_warmup(self._room_for_capture, remove=True)
            if self.round_thread is not None:
                self.round_thread.shutdown()
        for bank in self.banks:
            bank._residency = None
            bank.gpu_branch = False

    def _init_elastic(self):
        """The banks of a model that follows its routing: one array of each kind for all of them, which the banks and their compute backends read through views, so that a round works on all the
        banks at once; one table of GPU addresses, which graphs read in place."""
        counts = [bank.num_experts for bank in self.banks]
        self.first = dict(zip(self.banks, itertools.accumulate([0] + counts)))
        total = sum(counts)
        self.all_flags, self.all_usage = np.zeros(total, dtype=np.uint8), np.zeros(total, dtype=np.uint32)
        self.seen, self.heat, self.slow = np.zeros(total, dtype=np.uint32), np.zeros(total, dtype=np.float32), np.zeros(total, dtype=np.float32)
        self.bank_of = np.repeat(np.arange(len(self.banks)), counts)
        self.size_of = np.repeat(np.array([self.sizes[bank] for bank in self.banks], dtype=np.int64), counts)
        self.scale = (self.size_of.mean() / self.size_of).astype(np.float32)  # counts per byte, in units of pairs of an average expert: a place is bytes
        self.prior = np.zeros(total, dtype=np.float32)  # the ranking, between 0 and 1: what decides between experts of equal counts
        self._read_ranking()
        for bank, count in zip(self.banks, counts):
            start = self.first[bank]
            bank._flags, bank._usage = self.all_flags[start:start + count], self.all_usage[start:start + count]
            bank.resident_flags, bank.usage = torch.from_numpy(bank._flags), torch.from_numpy(bank._usage)
            bank.gpu_branch = True
        self.table = self.table_host = self.table_memory = None
        self.table_views, self.table_dirty, self.out_of_room = {}, False, False
        self.trimmed = -ELASTIC_TRIM_ROUNDS
        self.last_round, self.next_round, self.plan_due, self.round_room, self.arrivals = 0, 0, False, 0, None
        self.round_seconds, self.rounds_done, self.moved_slabs = 0.0, 0, 0  # what rounds cost on the thread that runs them; slabs moved between sizes
        self.batch, self.round_thread, self.round_events, self.pass_done = None, None, [None, None], None
        # one size of slot per size of expert: tensors one after the other, aligned
        self.parts, self.sources, sizes = {}, {}, []
        for bank in self.banks:
            parts, sources, start = [], [], 0
            for name in bank.expert_tensor_names:
                tensor = getattr(bank, name)
                nbytes = tensor[0].numel() * tensor.element_size()
                parts.append((name, start, nbytes, tensor.dtype, tensor.shape[1:]))
                sources.append((start, tensor.data_ptr(), nbytes))
                start += -(-nbytes // 256) * 256
            self.parts[bank], self.sources[bank] = parts, sources
            sizes.append(start)
        self.slot_bytes = sorted(set(sizes))
        self.slot_size = dict(zip(self.banks, sizes))
        self.level2 = LEVEL2 and torch.version.hip is None
        self.level2_store = _Level2(self, total) if self.level2 else None
        if self.level2:  # read by the host jobs: see _Level2
            self.level2_sources, self.level2_remote, self.level2_device = self.level2_store.sources, self.level2_store.remote, self.level2_store.device
        self.share, self.share_percent = None, None
        self.class_of = np.repeat(np.array([self.slot_bytes.index(size) for size in sizes]), counts)
        self.class_members = [np.flatnonzero(self.class_of == klass) for klass in range(len(self.slot_bytes))]
        self.capacity = [(ELASTIC_SLAB - 4096) // size for size in self.slot_bytes]
        self.placed_once = False
        self.block_sizes = None
        self.kept_sizes = {size for bank, size in zip(self.banks, sizes) if bank.keep_resident}  # slots of the banks whose experts stay in VRAM
        self.slabs, self.free_slots, self.slab_of, self.next_slab = {}, [[] for _ in self.slot_bytes], np.full(total, -1, dtype=np.int64), 0
        self.round_staging = [pinned.Buffer("experts", granularity=16 << 20) for _ in range(2)]
        allocator.before_graph_warmup(self._room_for_capture)

    def _read_ranking(self):
        for position, (bank, expert) in enumerate(self.ranking):
            self.prior[self.first[bank] + expert] = 1 - position / len(self.ranking)

    def pointers(self, bank, device):
        """The bank's rows of the table of GPU addresses (int64 [experts, tensors], zero without a GPU copy). The table stays at its address while the model is loaded: graphs read the placement there."""
        if self.table is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("The GPU expert pointer table must be prepared before graph capture")
            names = max(len(item.expert_tensor_names) for item in self.banks)
            rows = len(self.all_flags)
            memory = pinned.empty(rows * names * 8, "experts")
            if memory is None:
                raise RuntimeError(f"Unable to lock {rows * names * 8 / 2**20:.1f} MB of RAM for the expert table of {self.model_id}")
            self.table_memory = memory
            with torch.inference_mode(False):
                self.table_host = memory.view(torch.int64).view(rows, names)
                with allocator.normal_allocations():
                    self.table = torch.empty(rows, names, dtype=torch.int64, device=device)
                self.table_views = {item: self.table[self.first[item]:self.first[item] + item.num_experts] for item in self.banks}
            self.table_rows = self.table_host.numpy()
            self.table_rows[:] = 0
            for (item, expert) in self.loaded:
                self.write_row(item, expert, item._resident[expert])
            self.table_dirty = True
            self.flush_table()
            if self.level2 and self._expert_room() < LEVEL2_VRAM_BELOW * self.cumulative[-1]:
                self._init_share(device)
        return self.table_views[bank]

    def _init_share(self, device):
        """The GPU's share of the misses: a ring of VRAM slots that the format's library fills with experts of locked RAM for the host jobs (format.share_ring: a helper process on Windows, a
        thread elsewhere), and per bank a table of the addresses of the experts the GPU computes from there, written by the GPU from the rows the job leaves in mapped RAM. Made with the address
        table, outside graph capture, and freed with it: the doorbell jobs made from then on hold them."""
        total, format = len(self.all_flags), self.banks[0].format
        with torch.inference_mode(False), allocator.normal_allocations():  # read by the graphs, not graph scratch
            tables = torch.zeros(total, 3, dtype=torch.int64, device=device)
        ring = format.share_ring(max(self.slot_bytes), SHARE_SLOTS, device) if SHARE_MODE == "copies" else None
        if ring is not None:  # the sections of level 2 mapped by the helper now, while the prompt is read, before the first copies need them
            ring.prepare([section for _, _, section, _ in self.level2_store.entries.values() if section is not None])
        self.share, self.share_percent = (ring, tables), SHARE_PERCENT
        for bank in self.banks:
            first = self.first[bank]
            bank.cpu_share = bank.format.cpu_share(dict(sources=self.level2_sources.ctypes.data + 8 * first, offsets=[part[1] for part in self.parts[bank]], slot_bytes=self.slot_size[bank],
                                                        ring=None if ring is None else ring.handle, fraction=SHARE_PERCENT / 100, experts=bank.num_experts,
                                                        remote_sources=self.level2_remote.ctypes.data + 8 * first, device_sources=self.level2_device.ctypes.data + 8 * first))
            bank.gpu_share = None if bank.cpu_share is None else tables[first:first + bank.num_experts]
            bank.share_early = ring is None or ring.remote

    def _pause_share(self, paused):
        """A GPU with one copy engine (WDDM) serves the placement's copies and the share's in turn, by slices of about a millisecond: the share asks for
        none while the placement's copies are on their way, the GPU would wait for them in every layer."""
        if self.share is not None and self.share[0] is not None:
            self.share[0].pause(paused)

    def _free_share(self):
        """Once the doorbell jobs are released and nothing in flight asks for copies."""
        if self.share is not None:
            for bank in self.banks:
                bank.cpu_share = bank.gpu_share = None
            ring, self.share = self.share[0], None  # the rounds from now on ask no ring
            if ring is not None:
                if ring.remote and self.level2_store is not None:
                    self.level2_store.settle(ring)  # its fills done published, the others given up
                ring.close()
                if ring.remote:
                    from . import offload
                    offload.close_file_mappings()  # no helper maps them now

    def share_stats(self):
        """(experts in locked RAM, their bytes, copies asked for, landed in time, experts missing VRAM, host jobs, first failure) since the share was made."""
        shares = [bank.cpu_share for bank in self.banks if bank.cpu_share is not None]
        sizes = np.asarray(self.slot_bytes, dtype=np.int64)[self.class_of]
        held = self.level2_sources != 0
        return (int(held.sum()), int(sizes[held].sum()), sum(share.asked for share in shares), sum(share.landed for share in shares), sum(share.missing for share in shares),
                sum(share.jobs for share in shares), next((share.status for share in shares if share.status), 0))

    def write_row(self, bank, expert, tensors):
        if self.table_host is not None:
            self.table_rows[self.first[bank] + expert] = 0 if tensors is None else [tensors.slab.base + tensors.offset + part[1] for part in tensors.parts]
            self.table_dirty = True

    def flush_table(self):
        """To the GPU, on this thread's stream: what follows on it reads the new placement. Whoever writes the rows has synchronized the GPU before, nothing in flight reads them."""
        if self.table_dirty:
            with torch.inference_mode():
                self.table.copy_(self.table_host, non_blocking=True)
            self.table_dirty = False

    def _clear_graphs(self):
        manager = self.manager()
        graphs = manager.cuda_graphs.get(self.model_id)
        if graphs is not None:
            graphs.clear()

    def _nonexpert_bytes(self):
        manager = self.manager()
        return sum(size for name, size in manager.blocks_of_modules_sizes.items() if name == self.model_id or name.startswith(self.model_id + "/"))

    def placeable(self):
        """The ranked experts this residency could ever place: within its budget, or (dynamic) within the whole VRAM beside the ordinary
        blocks of the model and its co-tenants."""
        return self.ranking[:bisect.bisect_right(self.cumulative, self._expert_room())]

    def _expert_room(self):
        if self.dynamic:
            manager = self.manager()
            cotenants = manager.cotenants_map.get(self.model_id, ())
            model_ids = set(manager.models if cotenants == "*" else (self.model_id, *cotenants))
            ordinary = sum(size for name, size in manager.blocks_of_modules_sizes.items() if name.split("/", 1)[0] in model_ids)
            return torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory - ordinary - MARGIN
        return max(0, self.budget - self._nonexpert_bytes()) if self.budget else 0

    def pin(self, pin_module):
        """Pinned copies of the experts this residency could place, in ranking order and as far as the reserved RAM left by the models
        allows (pin_module(module, name) pins a module's parameters and stops there): their uploads become plain transfers. The banks
        keep their pageable sources, which the CPU backend reads."""
        experts = self.placeable()
        if not experts:
            return
        holder = nn.Module()
        for index, (bank, expert) in enumerate(experts):
            for name in bank.expert_tensor_names:
                holder.register_parameter(f"{index}_{name}", nn.Parameter(getattr(bank, name)[expert], requires_grad=False))
        pin_module(holder, f"{self.model_id} experts")
        for index, item in enumerate(experts):
            tensors = {name: getattr(holder, f"{index}_{name}").data for name in item[0].expert_tensor_names}
            tensors = {name: tensor for name, tensor in tensors.items() if tensor.is_pinned()}
            if tensors:
                self.pinned[item] = tensors
        self._pinned_holder = holder

    @contextlib.contextmanager
    def streaming(self, banks, tokens, nbytes=0):
        """A pass of many tokens over banks, in their order of use, whose experts out of VRAM stream through a ring (mmgp.expert_stream)."""
        device = torch.device("cuda", torch.cuda.current_device())
        if not all(bank.format.gpu_pass_supported(device) for bank in banks):
            yield None
            return
        self.release_copied_pages()
        with torch.inference_mode(False):  # reused by the passes that follow, whatever their inference mode
            staging = [buffer.get(REGION_BYTES) for buffer in self.stream_staging]
        if None in staging:  # the reserved RAM refuses them: each bank uploads its experts for itself
            yield None
            return
        free = runtime.free_vram() - nbytes
        if self.elastic and free < REGION_BYTES * expert_stream.MIN_RING_REGIONS:
            free += self._give_back(REGION_BYTES * expert_stream.MIN_RING_REGIONS - free)
        if free < REGION_BYTES * expert_stream.MIN_RING_REGIONS:  # no room for the ring: as above
            yield None
            return
        pool = self.elastic and self.model_id in self.manager().pool_models
        regions, kept, share = min(expert_stream.RING_REGIONS, free // REGION_BYTES), self._pool_batches if pool else None, expert_stream.CPU_SHARE if tokens < expert_stream.AHEAD_TOKENS and all(bank.format.pass_shares() for bank in banks) else 0.0
        sent = self._pool_ahead(banks, tokens) if pool and tokens < expert_stream.AHEAD_TOKENS else None
        note = (lambda bank, routed, routing: self._pool_note(bank, routed, tokens)) if pool and tokens < expert_stream.AHEAD_TOKENS else None
        self.stream = expert_stream._Lanes(device, banks, tokens, staging, regions, kept, share, sent, note) if sent else ExpertStream(device, banks, tokens, staging, regions, kept, share)
        self.stream.note = note
        try:
            yield self.stream
            if pool and tokens >= expert_stream.POOL_TOKENS:
                self._pool_enter(banks)
        finally:
            stream, self.stream = self.stream, None
            stream.close()

    def release_copied_pages(self, wait=False):
        """RAM short: the checkpoint pages of the experts having a GPU copy leave the working set, first of all (nothing reads them while the copy lives), so that the RAM goes to the experts the CPU reads and to the pinned pool. The pages of the experts out of VRAM stay:
        the CPU reads them, and read again they fault at about 1 GB/s. Once per expert, by groups of at least PAGES_RELEASED experts (those a round brings drop their pages as they copy them), on a thread of its own (about 75 ms per GiB), RAM checked at most once per second;
        wait: at once, on this thread. On the standby list they count as available and come back without the drive if the expert returns before the system reuses them."""
        now = time.monotonic()
        if not wait and now - self.pages_checked < 1:
            return
        self.pages_checked = now
        if self.pages_thread is not None and self.pages_thread.is_alive():
            if wait:
                self.pages_thread.join()
            return
        if not ram_short():
            return
        loaded = set(self.loaded)
        items = loaded - self.pages_released
        if wait and items or len(items) >= PAGES_RELEASED:
            self.pages_released = loaded
            if wait:
                self._drop_copied_pages(items)
            else:
                self.pages_thread = threading.Thread(target=self._drop_copied_pages, args=(items,), name="mmgp_expert_pages", daemon=True)
                self.pages_thread.start()

    def _drop_copied_pages(self, items):
        from . import offload
        sources = {bank: [(getattr(bank, name).data_ptr(), getattr(bank, name)[0].numel() * getattr(bank, name).element_size()) for name in bank.expert_tensor_names] for bank in {bank for bank, _ in items}}
        before = offload.psutil.virtual_memory().available
        offload._drop_mapped_pages(offload._mapped_pages([(address + expert * nbytes, nbytes) for bank, expert in items for address, nbytes in sources[bank]]))
        if self.manager() is not None and self.manager().verboseLevel >= 1:
            tqdm.write(f"RAM Short - {self.model_id}: Pages of {len(items)} Experts on GPU Released, {(offload.psutil.virtual_memory().available - before) / 2**30:.1f} GiB More Available")

    def _pool_ahead(self, banks, tokens):
        """The locked batches a short pass should send ahead: those that the routing of the last short pass read enough of, for what this one's tokens read. {bank: [(experts, locked copy)]}, None when there are none."""
        manager, sent = self.manager(), {}
        for bank in banks:
            for name, members, future in self.pool_parts.get(bank, ()):
                seen = self.pool_seen.get(name)
                if seen is not None and future.done() and future.result() and (source := manager.pool_source(name)) is not None and 1 - (1 - seen[0]) ** (tokens / seen[1]) >= expert_stream.AHEAD_SHARE:
                    sent.setdefault(bank, []).append((members, source))
        return sent or None

    def _pool_note(self, bank, routed, tokens):
        # of each of its batches in the pool, the share this pass reads
        for name, members, _ in self.pool_parts.get(bank, ()):
            self.pool_seen[name] = np.count_nonzero(routed[members] & (bank._flags[members] == 0)) / len(members), tokens

    def _pool_batches(self, bank, experts):
        """For a stream: its batches in the pinned pool that it sends whole (two thirds of their experts wanted at least, the others cross for nothing: one transfer against one per run of experts, at 24 GB/s, against 16 with the CPU copy) and the experts left."""
        wanted = np.zeros(bank.num_experts, dtype=bool)
        wanted[experts] = True
        whole = []
        for name, members, future in self.pool_parts.get(bank, ()):
            if future.done() and future.result() and (source := self.manager().pool_source(name)) is not None and 3 * np.count_nonzero(wanted[members]) >= 2 * len(members):
                whole.append((members, source))
                wanted[members] = False
        return whole, np.flatnonzero(wanted)

    def _pool_enter(self, banks):
        """After a pass that read whole layers, the placement known: the experts that crossed and no entry holds enter the pinned pool on the pool's thread, for the next passes, by batches of the size of the stream's, in the same layout. The last
        if at least half of one: a few experts out of an entry cross with the staged ones."""
        if self.share is not None and self.level2_store.nbytes() + LEVEL2_ENTRY <= LEVEL2_BYTES:
            # level 2 first: both take the same reserved RAM, which these batches filled before level 2's first entry at 64 GB (decoding reads level 2 at
            # each pass, these batches serve the prompts); they take what it leaves
            return
        manager, value = self.manager(), self.prior + self._value()
        for bank in banks:
            parts = [part for part in self.pool_parts.get(bank, ()) if not part[2].done() or (part[2].result() and manager.pool_source(part[0]) is not None)]
            held = np.zeros(bank.num_experts, dtype=bool)
            for _, members, _ in parts:
                held[members] = True
            experts, count = np.flatnonzero((bank._flags == 0) & ~held), expert_stream.batch_experts(bank)
            experts = experts[np.argsort(-value[self.first[bank] + experts], kind="stable")]  # the most read first: a batch is read whole or hardly at all
            for start in range(0, len(experts), count):
                members = experts[start:start + count]
                if 2 * len(members) < count:
                    continue
                batch = expert_stream._Batch(bank, members, 0)
                self.pool_names += 1
                name = f"{self.model_id}/experts/{self.pool_names}"
                future = manager.pool_experts(self.model_id, name, batch.nbytes, lambda buffer, helpers, batch=batch: expert_stream._copy_batch(buffer, batch, helpers))
                if future is None:  # the pool takes no more until the next quiet point
                    break
                parts.append((name, members, future))
            self.pool_parts[bank] = parts

    def pool_bytes(self):
        """The bytes of its streamed experts that the pinned pool holds or is getting, and those refused or gone: what it would take back. A model that never streamed asks for nothing."""
        manager, held, short = self.manager(), 0, 0
        for bank, parts in self.pool_parts.items():
            for name, members, future in parts:
                size = len(members) * self.sizes[bank]
                if not future.done() or (future.result() and manager.pool_source(name) is not None):
                    held += size
                else:
                    short += size
        return held, short

    def _nonexperts_resident(self):
        manager = self.manager()
        cotenants = manager.cotenants_map.get(self.model_id, ())
        model_ids = manager.models if cotenants == "*" else (self.model_id, *cotenants)
        for model_id in model_ids:
            if model_id not in manager.models:
                continue  # Co-tenant declarations can include optional pipeline components.
            if model_id not in manager.active_models_ids:
                return False
            preloaded = manager.preloaded_blocks_per_model[model_id]
            if any(name.split("/", 1)[1] not in preloaded for name, size in manager.blocks_of_modules_sizes.items() if name.startswith(model_id + "/") and size):
                return False
        return True

    def place(self):
        """Placement requested outside graph recording (runtime.place_experts): grows the GPU copies to what the budget or the free VRAM
        allows without waiting for a measuring cycle, never shrinks them (pressure does). False while the ordinary blocks are not all
        resident yet."""
        if not self._nonexperts_resident():
            return False
        self.prepare(grow_only=True)
        return True

    def prepare(self, grow_only=False):
        """At a model/cycle boundary, after the ordinary preload has priority."""
        self.thread = threading.get_ident()
        self.release_copied_pages()
        if self.changing or cuda_graphs.recording() or torch.cuda.is_current_stream_capturing():
            return  # never inside a graph capture (a decoding pass of a model called as a module): its next eager call places them
        if not self._nonexperts_resident():
            if self.elastic and self.placed_once and self.model_id in self.manager().active_models_ids:
                return  # ordinary blocks released under pressure: the rounds make their room, the experts stay
            self.unload(reset=False)
            return
        if self.dynamic and not self.started and not grow_only:
            # First call measures/allocates KV, recurrence and other working storage.
            self.started = True
            return
        if self.elastic and self.placed_once:
            return  # placed: its rounds follow the routing and the free VRAM, in place (adapt)
        target = max(0, self.budget - self._nonexpert_bytes()) if self.budget else 0
        if self.dynamic:
            if allocator.active == "vmm":
                state = allocator.stats()
                free = allocator.room() + state["reserved"] - state["allocated"]
            else:
                free = torch.cuda.mem_get_info()[0]
            # the model's _mm_expert_margin: what its later work needs beyond the free VRAM's usual margin (CUDA graphs still to capture, which no placement can make room for)
            # (elastic: its rounds give the rest to experts, and a capture asks for its room)
            target = max(0, self.nbytes + self._room() - ELASTIC_FIRST_MARGIN - len(self.slot_bytes) * ELASTIC_SLAB) if self.elastic else max(0, self.nbytes + free - getattr(self.manager().models[self.model_id], "_mm_expert_margin", MARGIN) - int(0.05 * torch.cuda.memory_allocated()))
        if self.pressure_ceiling is not None:
            target = min(target, self.pressure_ceiling)
        if self.elastic:
            self.placed_once = True
        # Called at every cycle (each decoding step): the ranked prefix that fits is found without walking the ranking.
        count = bisect.bisect_right(self.cumulative, target)
        if count == self.applied or grow_only and count <= len(self.loaded):
            return
        if self.dynamic and self.loaded and abs((self.cumulative[count - 1] if count else 0) - self.nbytes) < (64 << 20):
            return  # Allocation noise must not repeatedly invalidate live graphs.
        wanted = self.ranking[:count]
        if len(wanted) == len(self.loaded) and all(item in self.loaded for item in wanted):
            self.applied = count
            return
        self.changing = True
        try:
            self._clear_graphs()
            torch.cuda.synchronize()
            keep = set(wanted)
            for item in [item for item in reversed(self.loaded) if item not in keep]:
                self._drop(*item)
            additions = [item for item in wanted if item not in self.loaded]
            if additions:
                self._upload(additions)
            if self.elastic:
                self.flush_table()  # the flags already say these experts are in VRAM: kernels must not read a table without their addresses
            self.loaded = dict.fromkeys(wanted)
            self.applied = count
            if self.manager().verboseLevel >= 1:
                tqdm.write(f"Expert Residency - {self.model_id}: {len(self.loaded)}/{len(self.ranking)} Experts on GPU ({self.nbytes / (1 << 20):.0f} MiB)" + ("; CPU Experts Are Not Pinned" if self.model_id not in self.manager().pool_models else ""))
        finally:
            self.changing = False
            self._placement_changed()

    @torch.inference_mode()
    def adapt(self):
        """Between decoding passes, outside graph recording: experts the routing keeps asking for take the GPU copies of the experts of their bank that it does not read, in place
        and at the same addresses: no VRAM moves, and the graphs read the new placement through the pointer table and the resident flags, which they do not capture. Bounded by the
        routing seen since the last time, so that a change of topic costs a few short pauses. The ranking follows, so that a later placement keeps what was learned."""
        self.adapt_calls += 1
        if self.share is not None and self.share_percent != SHARE_PERCENT:
            self.share_percent = SHARE_PERCENT
            for bank in self.banks:
                if bank.cpu_share is not None:
                    bank.cpu_share.fraction = SHARE_PERCENT / 100
        if self.elastic:
            return self._adapt_elastic()
        if not self.adapt_due and self.adapt_calls % ADAPT_PASSES or self.changing or not self.loaded or cuda_graphs.recording() or torch.cuda.is_current_stream_capturing():
            return
        self.adapt_due = False
        plan, total = [], 0
        for bank in self.banks:
            if not bank.reports_usage:
                continue
            usage, flags = bank._usage, bank._flags
            total += int(usage.sum())
            missing = np.nonzero((flags == 0) & (usage >= 2))[0]
            residents = np.nonzero(flags)[0]
            if not len(missing) or not len(residents):
                continue
            missing = missing[np.argsort(-usage[missing].astype(np.int64), kind="stable")]
            priority = np.zeros(bank.num_experts) if bank.expert_priority is None else np.asarray(bank.expert_priority, dtype=np.float64)
            victims = residents[np.lexsort((priority[residents], usage[residents]))]
            for new, old in zip(missing.tolist(), victims.tolist()):
                gain = int(usage[new]) - int(usage[old])
                if gain < 2:
                    break
                plan.append((gain, id(bank), bank, old, new))
        budget = min(ADAPT_MAX_BYTES, max(0, total - self.adapt_counted) * ADAPT_BYTES_PER_PAIR)
        chosen = []
        for _, _, bank, old, new in sorted(plan, key=lambda item: -item[0]):
            if budget < self.sizes[bank]:
                break
            budget -= self.sizes[bank]
            chosen.append(((bank, new), old))
        if chosen:
            self.changing = True
            try:
                torch.cuda.synchronize()  # nothing in flight reads these copies or the flags
                self._upload([item for item, _ in chosen], dict(chosen))
                self.swapped += len(chosen)
            finally:
                self.changing = False
        for bank in self.banks:  # about half of a count left after 5 adaptations
            if bank.reports_usage:
                bank._usage -= bank._usage >> 3
        self.adapt_counted = sum(int(bank._usage.sum()) for bank in self.banks if bank.reports_usage)

    def _adapt_elastic(self):
        """A round between decoding passes, outside graph recording: the routing counts decay and take the new pairs; experts the routing asks for take the place of the experts it reads least, of
        any bank (a place is bytes: GPU copies are freed and allocated, the same shapes reuse their memory), and the placement grows into free VRAM or gives it back. Everything in place: the
        graphs read the placement through the table and the flags."""
        if self.changing or not self.placed_once or cuda_graphs.recording() or torch.cuda.is_current_stream_capturing():
            return
        self._land()
        self.plan_round()  # asked for during the last pass by a path that does not call it
        batch = self.batch
        if self.adapt_calls < self.next_round and not self.adapt_due and (batch is None or self.arrivals is not None or not batch.future.done()):
            return  # a batch ready goes at once, whatever the time of the next plan
        started = time.perf_counter()
        try:
            self._round()
        finally:
            self.round_seconds += time.perf_counter() - started

    def _round(self):
        """Between two passes: what touches the placement. The experts to bring come after, from plan_round()."""
        self.thread = threading.get_ident()
        if self.arrivals is not None:  # their copies are still on their way: the counts go on
            self.next_round = self.adapt_calls + 1
            return
        batch = self.batch
        if batch is not None:
            if not batch.future.done():  # still copying from the drive or the mapped file: the counts go on
                self.next_round = self.adapt_calls + 1
                return
            self.batch = None
            self._commit(batch)
            return
        room = self._room()
        blocks_out = self._blocks_out()
        if room < ELASTIC_LOW + blocks_out:  # ordinary blocks come before experts: what they need of the VRAM to come back stays free
            room = self._room(trim=True)  # chunks the allocator keeps count as used: no expert leaves for them
            if room < ELASTIC_LOW + blocks_out:
                self._give_back(ELASTIC_KEEP + blocks_out - room)
                room = self._room(trim=True)
        self.round_room = room - blocks_out
        if self.adapt_calls % ELASTIC_REBALANCE == 0:
            self._rebalance(self._value())
        self.plan_due = True

    def plan_round(self):
        """The experts to bring next: numbers on the CPU and a job for a helper thread, nothing the GPU waits for. A decoding pass calls it once its verification is launched, when the host would only wait; otherwise the next pass does."""
        self.release_copied_pages()
        if not self.plan_due:
            return
        started = time.perf_counter()
        try:
            self._plan()
        finally:
            self.round_seconds += time.perf_counter() - started
            self.rounds_done += 1

    def _plan(self):
        self.plan_due = False
        if self.level2_store is not None and self.share is not None:  # level 2 is read by the share's jobs only
            self.level2_store.round(self.prior + self._value())
        passes, self.last_round = self.adapt_calls - self.last_round, self.adapt_calls
        flags, new_pairs = self.all_flags, self.all_usage - self.seen
        self.seen[:] = self.all_usage
        prompt = self.adapt_due
        if prompt:  # a prompt read since the last round: thousands of pairs per expert would weigh on the counts for dozens of passes
            self.adapt_due = False
            new_pairs = np.where(new_pairs > 16, 16 + np.sqrt(np.maximum(new_pairs, 16).astype(np.float32) - 16), new_pairs)
        self.heat = self.heat * ELASTIC_DECAY[0] ** passes + new_pairs
        self.slow = self.slow * ELASTIC_DECAY[1] ** passes + (0 if prompt and not ELASTIC_PROMPT_SLOW else new_pairs)
        room = self.round_room
        value = self._value()
        missing = np.flatnonzero((flags == 0) & (value >= ELASTIC_GAIN))
        room_left = max(0, room - ELASTIC_KEEP)
        if not self.dynamic:
            room_left = min(room_left, max(0, self.budget - self._nonexpert_bytes() - self.nbytes))
        available = [len(slots) for slots in self.free_slots]
        additions, removals, victims = [], [], {}
        swaps = ELASTIC_FILL if room_left >= ELASTIC_SLAB or sum(available) >= ELASTIC_SWAPS else ELASTIC_SWAPS

        def take(new):
            klass = self.class_of[new]
            if available[klass]:
                available[klass] -= 1
                return True
            nonlocal room_left
            if room_left >= ELASTIC_SLAB:
                room_left -= ELASTIC_SLAB
                available[klass] += self.capacity[klass] - 1
                return True
            return False

        if missing.size:
            for new in missing[np.argsort(-value[missing], kind="stable")][:swaps].tolist():
                if take(new):
                    additions.append(new)
                    continue
                klass = self.class_of[new]  # a place of its size, from what the routing reads least
                if klass not in victims:
                    residents = self.class_members[klass][flags[self.class_members[klass]] != 0]
                    victims[klass] = residents[_smallest(value[residents] + self.prior[residents] * ELASTIC_TIE, swaps)].tolist()[::-1]
                if victims[klass] and value[victims[klass][-1]] <= value[new] - ELASTIC_GAIN:
                    removals.append(victims[klass].pop())
                    additions.append(new)
        if len(additions) < swaps and (room_left >= ELASTIC_SLAB or any(available)):
            # places that nothing asks for: the experts most worth having among those left, of the sizes that have places
            planned = set(additions)
            rest = np.flatnonzero(flags == 0)
            if room_left < ELASTIC_SLAB:
                rest = rest[np.asarray(available)[self.class_of[rest]] > 0]
            for new in rest[_smallest(-(value[rest] + self.prior[rest] * ELASTIC_TIE), 2 * swaps)].tolist():
                if new not in planned and take(new):
                    additions.append(new)
                    if len(additions) >= swaps:
                        break
                elif not any(available) and room_left < ELASTIC_SLAB:
                    break
        if additions:
            self._stage(additions, removals)
        self.next_round = self.adapt_calls + (1 if len(additions) >= ELASTIC_SWAPS else ELASTIC_EVERY)

    def _blocks_out(self):
        """The bytes of the ordinary blocks of the model that its preload lost under pressure."""
        manager = self.manager()
        preloaded = manager.preloaded_blocks_per_model.get(self.model_id, ())
        if self.block_sizes is None:
            prefix = self.model_id + "/"
            self.block_sizes = {name[len(prefix):]: size for name, size in manager.blocks_of_modules_sizes.items() if size and name.startswith(prefix)}
        return sum(size for name, size in self.block_sizes.items() if name not in preloaded)

    def _item(self, index):
        bank = self.banks[self.bank_of[index]]
        return bank, index - self.first[bank]

    def _rebalance(self, value):
        """A slab moves from the size whose coldest slab is worth least to the size whose missing experts are worth most, when those are worth much more. Its experts leave at once, those of the other size come by the rounds that follow."""
        flags = self.all_flags
        placed = np.flatnonzero((flags != 0) & (self.slab_of >= 0))
        if not placed.size:
            return
        worth = np.bincount(self.slab_of[placed], weights=value[placed], minlength=self.next_slab)
        best, best_worth = None, 0.0
        for klass, members in enumerate(self.class_members):
            out = value[members[flags[members] == 0]]
            slots = self.capacity[klass]
            if out.size >= slots:
                top = float(np.partition(out, out.size - slots)[out.size - slots:].sum())
                if top > best_worth:
                    best, best_worth = klass, top
        if best is None:
            return
        worst = min((slab for slab in self.slabs.values() if slab.size != self.slot_bytes[best] and slab.size not in self.kept_sizes), key=lambda slab: worth[slab.id], default=None)
        if worst is None or best_worth <= 2 * worth[worst.id] + 8 * ELASTIC_GAIN or self.slot_bytes[best] > worst.nbytes:
            return
        self.changing = True
        try:
            torch.cuda.synchronize()
            for index in list(worst.members):
                self._drop(*self._item(index))
            self.flush_table()
            old = self.slot_bytes.index(worst.size)
            self.free_slots[old] = [slot for slot in self.free_slots[old] if slot[0] is not worst]
            worst.size = self.slot_bytes[best]
            self.free_slots[best] += [(worst, offset) for offset in range(0, worst.nbytes - worst.size + 1, worst.size)]
            self.moved_slabs += 1
        finally:
            self.changing = False

    def _take_slot(self, index):
        """A free slot for an expert, in a new slab if none is free."""
        klass = self.class_of[index]
        slots = self.free_slots[klass]
        if not slots:
            size = self.slot_bytes[klass]
            slab = _Slab(size, self.next_slab)
            self.next_slab += 1
            self.slabs[slab.id] = slab
            self.nbytes += ELASTIC_SLAB
            slots += [(slab, offset) for offset in range(0, slab.nbytes - size + 1, size)]
        slab, offset = slots.pop()
        slab.members.add(index)
        self.slab_of[index] = slab.id
        return _Slot(slab, offset, self.parts[self.banks[self.bank_of[index]]])

    def _free_slot(self, slot, index):
        slot.slab.members.discard(index)
        self.slab_of[index] = -1
        self.free_slots[self.slot_bytes.index(slot.slab.size)].append((slot.slab, slot.offset))

    def _release_slab(self, slab):
        klass = self.slot_bytes.index(slab.size)
        self.free_slots[klass] = [slot for slot in self.free_slots[klass] if slot[0] is not slab]
        del self.slabs[slab.id]
        self.nbytes -= ELASTIC_SLAB
        slab.tensor = None

    def _stage(self, additions, removals):
        """The bytes of the experts chosen go to a locked buffer on a helper thread, reading what a drive must read first: decoding goes on meanwhile, its CPU experts reading the file as before. The next round finding them
        there commits them."""
        moves, plan, offset = [], [], 0
        for index in additions:
            bank, expert = self._item(index)
            for start, address, nbytes in self.sources[bank]:
                moves.append((offset + start, address + expert * nbytes, nbytes))
            plan.append((index, offset))
            offset += int(self.slot_bytes[self.class_of[index]])
        slot = next((slot for slot in range(2) if self.staging_ready(slot)), None)
        if slot is None:
            return  # a transfer still reads both buffers: next round
        size = max(ELASTIC_SWAPS, len(additions)) * self.slot_bytes[-1]
        with torch.inference_mode(False):
            staging = self.round_staging[slot].get(size)
        if staging is None:
            raise RuntimeError(f"Unable to lock {size / 2**20:.0f} MB of RAM for the expert copies of {self.model_id}")
        if self.round_thread is None:
            self.round_thread = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mmgp_expert_rounds")
        self.batch = _Batch(self.round_thread.submit(_copy_moves, staging.data_ptr(), moves), plan, removals, staging, slot)

    def staging_ready(self, slot):
        event = self.round_events[slot]
        return event is None or event.query()

    def _commit(self, batch):
        """The experts staged replace those that leave, in place: between two passes, nothing in flight reads those. Their copies run on the copy stream beside the next passes, which go on without them (the CPU computes them): _land() places them once arrived."""
        batch.future.result()
        before = self.nbytes
        self.changing = True
        try:
            self._passes_done()
            if self.level2_store is not None and self.share is not None:  # nothing in flight reads level 2 either: its experts placed now, or worth much less than others, leave
                self.level2_store.round(self.prior + self._value(), sync=True)
            copies = expert_stream.copy_stream(torch.device("cuda", torch.cuda.current_device()))
            base, stream, arrivals = batch.staging.data_ptr(), _stream_handle(copies), []
            for index in batch.removals:
                if self.all_flags[index]:
                    self._drop(*self._item(index))
            try:
                for index, offset in batch.plan:
                    if self.all_flags[index]:
                        continue
                    bank, expert = self._item(index)
                    slot = self._take_slot(index)
                    nbytes = int(self.slot_bytes[self.class_of[index]])
                    if stream is None:
                        with torch.cuda.stream(copies):
                            slot.slab.tensor[slot.offset:slot.offset + nbytes].copy_(batch.staging[offset:offset + nbytes], non_blocking=True)
                    else:
                        status = _memcpy(slot.slab.base + slot.offset, base + offset, nbytes, stream)
                        if status:
                            raise RuntimeError(f"cuMemcpyHtoDAsync failed ({status})")
                    arrivals.append((index, slot))
            except torch.OutOfMemoryError:  # the room read at the plan is gone: another program took it. Next round reads it again
                if not self.out_of_room:
                    self.out_of_room = True
                    tqdm.write(f"Expert Residency - {self.model_id}: VRAM taken while experts were placed, {self.nbytes / (1 << 20):.0f} MiB on GPU")
            event = self.round_events[batch.slot] = self.round_events[batch.slot] or torch.cuda.Event()
            event.record(copies)
            self.arrivals = arrivals, event
            self._pause_share(True)  # the copy engine is this round's until its copies land
            self.flush_table()
            if self.nbytes != before:
                self._placement_changed()
        finally:
            self.changing = False

    def _passes_done(self):
        """Waits for the passes queued on this thread's stream: the copies, the flags and the table change once nothing in flight reads them. A blocking
        wait: where a pass still runs at the round (Fish decodes a frame ahead), its CPU experts are computing, and a spinning wait took their cores
        (Fish: 48 ms per round)."""
        self.pass_done = self.pass_done or torch.cuda.Event(blocking=True)
        self.pass_done.record()
        self.pass_done.synchronize()

    def _land(self, wait=False):
        """The experts whose copies have arrived take their place, between two passes: what follows reads them."""
        if self.arrivals is None:
            return
        arrivals, event = self.arrivals
        if wait:
            event.synchronize()
        elif not event.query():
            return
        self.arrivals = None
        self._pause_share(False)
        self._passes_done()  # a pass in flight would find their flags set and their table rows not yet there: computed by neither
        for index, slot in arrivals:
            bank, expert = self._item(index)
            bank._place(expert, slot)
            self.loaded[(bank, expert)] = None
            self.swapped += 1
        self.flush_table()

    def _finish_batch(self):
        """The batch on its way, if any, is dropped: its experts stay out."""
        self._land(wait=True)
        batch, self.batch = self.batch, None
        if batch is not None:
            batch.future.exception()  # waits for its thread

    def _value(self):
        """What a place is worth for each expert, in pairs of an average expert: places are bytes."""
        return (self.heat + ELASTIC_SLOW_SHARE * self.slow) * self.scale

    def _room(self, trim=False):
        """The VRAM new memory may take. Short of what a round wants, or asked to, the allocator gives back what it holds unused, from time to time (the driver pools keep what is freed until then)."""
        if allocator.active != "vmm":
            return torch.cuda.mem_get_info()[0]
        room = allocator.room()
        if trim or room < ELASTIC_KEEP + ELASTIC_SLAB and self.adapt_calls - self.trimmed >= ELASTIC_TRIM_ROUNDS:
            state = allocator.stats()
            if trim or state["reserved"] - state["allocated"] - state["graph_pools"] >= ELASTIC_TRIM:
                torch.cuda.synchronize()  # the frees are seen: the pools give them back
                torch.cuda.empty_cache()
                self.trimmed = self.adapt_calls
                room = allocator.room()
        return room

    def _room_for_capture(self):
        """Before the warmup of a graph capture: what it allocates cannot make experts leave, so that they leave first. Rounds give them their place back."""
        if not self.loaded or self.changing or torch.cuda.is_current_stream_capturing():
            return
        room = self._room()
        if room < ELASTIC_KEEP:
            room = self._room(trim=True)
            if room < ELASTIC_KEEP:
                self._give_back(ELASTIC_KEEP - room)
                self._room(trim=True)

    def _give_back(self, need):
        """Frees need bytes of GPU copies: the slabs whose experts the routing reads least go whole, what a slab frees being memory the allocator can give back (a slab with an expert of a bank in use by its backend stays), in
        place: the graphs stay valid. Returns the bytes freed."""
        if not self.slabs:
            return 0
        self._land(wait=True)
        value = self._value()
        order = sorted(self.slabs.values(), key=lambda slab: (slab.size in self.kept_sizes, sum(value[index] for index in slab.members)))
        self.changing = True
        try:
            torch.cuda.synchronize()
            freed = 0
            for slab in order:
                items = [self._item(index) for index in slab.members]
                if any(bank._leases for bank, _ in items):
                    continue
                for bank, expert in items:
                    self._drop(bank, expert)
                self._release_slab(slab)
                freed += ELASTIC_SLAB
                if freed >= need:
                    break
            self.flush_table()
            torch.cuda.synchronize()  # the next user of the table may be on another stream (a parked model's)
            self._placement_changed()
            return freed
        finally:
            self.changing = False

    def _keep_learned(self):
        """Before the copies go: what is placed comes first in the ranking, for the next placement, the rest by what the routing last read and the old ranking."""
        order = np.lexsort((self.prior, self._value(), self.all_flags))[::-1]
        by_index = {self.first[bank] + expert: (bank, expert) for bank, expert in self.ranking}
        self.ranking = [by_index[index] for index in order.tolist()]
        self.cumulative = list(itertools.accumulate(self.sizes[bank] for bank, _ in self.ranking))
        self.rank_of, self.applied = None, None
        self._read_ranking()

    def _upload_slots(self, additions):
        """A first placement in slabs, by batches through two locked buffers of its own (a prompt's stream may be running, on its first block), kept for the time of the placement: the CPU copies on the stream's helper threads while the transfer of the previous
        batch runs, and the slots of a batch, neighbours in their slab, one transfer: 3 per expert hold the bus to 6 GB/s. False if the reserved RAM refuses the buffers."""
        with torch.inference_mode(False):
            staging = [buffer.get(REGION_BYTES) for buffer in self.upload_staging]
        if None in staging:
            return False
        by_size = {}
        for bank, expert in additions:
            by_size.setdefault(int(self.slot_bytes[self.class_of[self.first[bank] + expert]]), []).append((bank, expert))
        transfers = [torch.cuda.Event(), torch.cuda.Event()]
        slot, armed, pool, staged, sent = 0, [False, False], expert_stream._helper_pool(), None, [0]
        from .offload import report_progress

        def send(chosen, size, futures, slot):
            """A staged batch to its slots: the call holds this thread for most of the transfer, while the CPU copies of the next batch run."""
            for future in futures:
                future.result()
            first = 0
            for place in range(1, len(chosen) + 1):
                if place == len(chosen) or chosen[place][0].slab is not chosen[first][0].slab or chosen[place][0].offset != chosen[place - 1][0].offset + size:
                    target = chosen[first][0]
                    target.slab.tensor[target.offset:target.offset + (place - first) * size].copy_(staging[slot][first * size:place * size], non_blocking=True)
                    first = place
            for target, bank, expert in chosen:
                bank._place(expert, target)
                self.loaded[(bank, expert)] = None
            transfers[slot].record()
            armed[slot] = True
            sent[0] += len(chosen)
            report_progress("Preloading Experts", sent[0], len(additions))

        with tqdm(total=len(additions), desc=f"Preloading Experts - {self.model_id}", unit="expert", mininterval=1 / 3, disable=self.manager().verboseLevel < 1) as progress:
            try:
                for size, members in by_size.items():
                    count = REGION_BYTES // size
                    for start in range(0, len(members), count):
                        runtime.checkpoint()
                        slot = 1 - slot
                        if armed[slot]:
                            transfers[slot].synchronize()  # the transfers that read this buffer last have completed
                        chosen = sorted(((self._take_slot(self.first[bank] + expert), bank, expert) for bank, expert in members[start:start + count]), key=lambda item: (item[0].slab.id, item[0].offset))
                        base = staging[slot].data_ptr()
                        moves = [(offset + place * size, address + expert * nbytes, nbytes) for place, (_, bank, expert) in enumerate(chosen) for offset, address, nbytes in self.sources[bank]]
                        parts = -(-len(moves) // expert_stream.COPY_THREADS)
                        futures = [pool.submit(_copy_moves, base, moves[first:first + parts]) for first in range(0, len(moves), parts)]
                        if staged is not None:
                            send(*staged)
                            progress.update(len(staged[0]))
                        staged = chosen, size, futures, slot
                if staged is not None:
                    send(*staged)
                    progress.update(len(staged[0]))
            finally:
                for index in range(2):
                    if armed[index]:
                        transfers[index].synchronize()
                    self.upload_staging[index].release()
        return True

    def _upload(self, additions, slots=None):
        """GPU copies of new experts, from their pinned copies (pin()) or else through two pinned buffers: copying the next experts from
        their pageable (mmap) sources overlaps the transfer of the previous ones. These copies are only made when the placement changes.
        Sources that a drive must read are read ahead in large requests, the next expert while the current one is copied (the parallel CPU
        copy would fault them in 64 KiB reads). slots: (bank, expert) -> the expert of that bank whose GPU copy it takes over (adapt())."""
        if self.elastic and slots is None and not self.pinned and self._upload_slots(additions):
            return
        size = max(UPLOAD_STAGING, 2 * max(self.sizes.values()))
        with torch.inference_mode(False):  # reused by the uploads that follow, whatever their inference mode
            staging = [buffer.get(size) for buffer in self.upload_staging]
        if None in staging:
            raise RuntimeError(f"Unable to lock {2 * size / 2**20:.0f} MB of RAM for the expert uploads of {self.model_id}")
        transfers = [torch.cuda.Event(), torch.cuda.Event()]
        slot, offset = 0, 0

        def sources(index):
            bank, expert = additions[index]
            return [(tensor.data_ptr(), tensor.numel() * tensor.element_size()) for name in bank.expert_tensor_names
                    if name not in self.pinned.get((bank, expert), {}) for tensor in (getattr(bank, name)[expert],)]

        from .offload import report_progress
        if additions:
            prefetch_ranges(cold_ranges(sources(0)))
        try:
            for index, (bank, expert) in enumerate(tqdm(additions, desc=f"Preloading Experts - {self.model_id}", unit="expert", mininterval=1 / 3, disable=self.manager().verboseLevel < 1 or slots is not None)):
                runtime.checkpoint()
                if slots is None:  # a placement, not the swaps of adapt() between decoding passes
                    report_progress("Preloading Experts", index, len(additions))
                if index + 1 < len(additions):
                    prefetch_ranges(cold_ranges(sources(index + 1)))
                if offset + self.sizes[bank] + 64 * len(bank.expert_tensor_names) > size:
                    transfers[slot].record()
                    slot, offset = 1 - slot, 0
                    transfers[slot].synchronize()  # the transfers that read this buffer last have completed
                tensors = {}
                kept = self.pinned.get((bank, expert), {})
                old = None if slots is None else slots[(bank, expert)]
                target = self._take_slot(self.first[bank] + expert) if self.elastic else None if old is None else bank._resident[old]
                for name in bank.expert_tensor_names:
                    if name in kept:
                        tensors[name] = (torch.empty_like(kept[name], device="cuda") if target is None else target[name]).copy_(kept[name], non_blocking=True)
                        continue
                    source = getattr(bank, name)[expert]
                    nbytes = source.numel() * source.element_size()
                    staged = staging[slot][offset:offset + nbytes].view(source.dtype).view(source.shape)
                    staged.copy_(source)
                    tensors[name] = (torch.empty_like(source, device="cuda") if target is None else target[name]).copy_(staged, non_blocking=True)
                    offset += -(-nbytes // 64) * 64
                if self.elastic:
                    bank._place(expert, target)
                    self.loaded[(bank, expert)] = None
                    continue
                if target is not None:
                    self._take_over(bank, old, expert)
                    continue
                bank._place(expert, tensors)
                self.loaded[(bank, expert)] = None
                self.nbytes += self.sizes[bank]
            if slots is None:
                report_progress("Preloading Experts", len(additions), len(additions))
        finally:
            transfers[slot].record()
            transfers[slot].synchronize()  # also orders the other buffer's transfers, queued before on the same stream
        if self.elastic and slots is None:
            drop_behind([span for index in range(len(additions)) for span in sources(index)])  # the GPU's copies replace them

    def _take_over(self, bank, old, new):
        bank._swap(old, new)
        del self.loaded[(bank, old)]
        self.loaded[(bank, new)] = None
        if self.rank_of is None:
            self.rank_of = {item: position for position, item in enumerate(self.ranking)}
        # same bank, same size: the cumulative sizes hold
        low, high = self.rank_of[(bank, new)], self.rank_of[(bank, old)]
        self.ranking[low], self.ranking[high] = (bank, old), (bank, new)
        self.rank_of[(bank, new)], self.rank_of[(bank, old)] = high, low

    def _drop(self, bank, expert):
        if self.elastic:
            self._free_slot(bank._resident[expert], self.first[bank] + expert)
        else:
            self.nbytes -= self.sizes[bank]
        bank._evict(expert)
        del self.loaded[(bank, expert)]
        self.applied = None

    def _placement_changed(self):
        auto_preload = self.manager().auto_preload
        if auto_preload is not None:
            auto_preload.on_expert_change()

    def on_pressure(self, need, parked=False):
        # parked: its offload object is parked (offload.park), any thread may release its copies
        if self.elastic and not (need <= 0 or self.changing or not parked and self.thread != threading.get_ident() or torch.cuda.is_current_stream_capturing()):
            return self._give_back(need + MARGIN)  # in place: the graphs stay valid
        if need <= 0 or self.changing or not parked and self.thread != threading.get_ident() or cuda_graphs.recording() or torch.cuda.is_current_stream_capturing():
            return 0
        victims = [(bank, expert) for bank, expert in reversed(self.loaded) if not bank._leases]
        if not victims:
            return 0
        self.changing = True
        try:
            self._clear_graphs()
            torch.cuda.synchronize()
            freed = 0
            for bank, expert in victims:
                freed += self.sizes[bank]
                self._drop(bank, expert)
                if freed >= need + MARGIN:
                    break
            self.pressure_ceiling = self.nbytes
            self._placement_changed()
            return freed
        finally:
            self.changing = False

    def unload(self, reset=True):
        self.pages_released = set()
        if self.elastic:
            self._finish_batch()
        if self.loaded:
            if self.elastic:
                self._keep_learned()
            self._clear_graphs()
            torch.cuda.synchronize()
            for item in list(reversed(self.loaded)):
                self._drop(*item)
            if self.elastic:
                self.flush_table()
            self._placement_changed()
        if self.elastic:
            for slab in list(self.slabs.values()):
                self._release_slab(slab)
            self.placed_once = False
        if reset:
            self.started = False
            self.pressure_ceiling = None
            for bank in self.banks:
                bank.release_execution_cache()
            if self.elastic and self.table is not None:
                torch.cuda.synchronize()
                if self.share is not None and self.manager().verboseLevel >= 1:
                    held, nbytes, asked, landed, missing, jobs, status = self.share_stats()
                    ring = self.share[0].stats() if self.share[0] is not None else dict(wait_nanoseconds=0, issued=0, part_nanoseconds=0, parts=0, call_nanoseconds=0, call_max_nanoseconds=0,
                                                                                        calls_under={}, wake_nanoseconds=0, wakes=0, status=0,
                                                                                        fraction=float(np.mean([bank.cpu_share.live for bank in self.banks if bank.cpu_share is not None])))
                    tqdm.write(f"Expert Level 2 - {self.model_id}: {held} Experts in Locked RAM ({nbytes / 2**30:.1f} GiB); of {missing} Experts Missing VRAM over {jobs} Jobs, {asked} Copied, "
                               f"{landed} in Time for the GPU; a copy waits {ring['wait_nanoseconds'] / max(ring['issued'], 1) / 1e3:.0f} us to be issued, a CPU part lasts "
                               f"{ring['part_nanoseconds'] / max(ring['parts'], 1) / 1e3:.0f} us; copy call {ring['call_nanoseconds'] / max(ring['issued'], 1) / 1e3:.0f} us (longest "
                               f"{ring['call_max_nanoseconds'] / 1e6:.1f} ms; calls under {ring['calls_under']}), wake-up {ring['wake_nanoseconds'] / max(ring['wakes'], 1) / 1e3:.0f} us, share now {ring['fraction']:.2f}; "
                               f"level 2 {self.level2_store.counts}" + (f", Failure {status or ring['status']}" if status or ring['status'] else ""))
                self._free_share()
                pinned.retire(self.table_memory)
                self.table = self.table_host = self.table_memory = None
                self.table_views, self.table_dirty = {}, False
                for buffer in self.round_staging:
                    buffer.release()
                self.round_events = [None, None]
