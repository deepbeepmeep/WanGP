# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
# With ideas of Strata (MIT License, https://github.com/Niko1221/Strata) for reading prompts.
"""A pass of many tokens over banks of experts mostly out of VRAM (prompt reading): the experts without a GPU copy cross the bus once per pass, for all their rows, in a fixed order and ahead of their use, so that the copy engine keeps working while
the GPU computes (the next layer's experts arrive during the current layer's attention). One thread staging and one issuing the transfers: helper threads do the CPU copies from the pageable checkpoint into two pinned staging buffers (Residency keeps them)
while the other buffer's transfer runs, whose call holds its thread for most of the transfer. Each batch of experts takes one region of a VRAM ring, reused once the kernels that read it have run. Batches of a model of the pinned pool stay in it (Residency):
the next passes send them from there, without the CPU copies, whole since their experts lie one after the other."""
import collections
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from .read_ahead import cold_ranges, prefetch_ranges

ENABLED = "prefill" not in os.environ.get("WGP_FN_DISABLE", "").split(",")  # proven: the switch goes with the others at the end
REGION_BYTES = 128 << 20  # a batch of experts: one region of the ring, one transfer, one launch of the kernels
RING_REGIONS = 12  # about one layer and a half of the largest experts: the next layer's can arrive while this one's are read
MIN_RING_REGIONS = 8  # with less free VRAM: a longer chunk reads each expert for more tokens, a longer ring only overlaps more of them
AHEAD_TOKENS = 4096  # from this many tokens nearly every expert of a layer is routed: the experts stream before their routing is known; below, each layer's routed experts once it is
COPY_THREADS = 4  # the CPU copies, beside the transfers that read what they wrote: 13 GiB/s with 3 threads, 16 with 4, no more beyond
TAKE_EXPERTS = 16  # experts per job of CPU copies
TRANSFER_STREAMS = int(os.environ.get("WGP_FN_TRANSFER_STREAMS", 2))  # transfers side by side, on as many copy engines: 5-10% on prompts of 1K to 3K tokens, none from 8K, none from a third
CPU_SHARE = float(os.environ.get("WGP_FN_CPU_SHARE", -1))  # of the experts a short prompt would stream, the share the CPU takes; below 0: where both sides end together (1K tokens, in one process: 581-733 -> 692-792 tok/s; 2K about level; a fixed share of 0.25 or more loses)
CPU_SHARE_ROWS = 8  # an expert goes to the CPU for so few rows at most: its time there grows with them, a transfer's does not
STREAM_RATE = 16e9  # bytes per second of the experts that stream, CPU copies included
AHEAD_SHARE = float(os.environ.get("WGP_FN_AHEAD_SHARE", 0.6))  # a short pass sends ahead the locked batches of which it should read at least this share (above 1: none). 3K tokens, against none: 1300 -> 2000 tok/s
POOL_TOKENS = 1024  # a pass of this many tokens at least sends the experts it streams to the pinned pool
pair_seconds = [100e-6]  # what a (token, expert) pair takes on the CPU pool, measured by the passes that share

_copy_streams = {}
_helpers = None


def copy_stream(device, number=0):
    """A stream of the transfers that run beside the GPU's work."""
    if (device, number) not in _copy_streams:
        _copy_streams[(device, number)] = torch.cuda.Stream(device)
    return _copy_streams[(device, number)]


def _helper_pool():
    global _helpers
    if _helpers is None:
        _helpers = ThreadPoolExecutor(max_workers=COPY_THREADS, thread_name_prefix="mmgp_expert_copy")
    return _helpers


def batch_experts(bank):
    """How many experts of a bank a batch holds."""
    return max(1, REGION_BYTES // (bank.expert_nbytes + 64 * len(bank.expert_tensor_names)))


def _take(job):
    source, experts, target = job
    np.take(source, experts, axis=0, out=target, mode="clip")


def _copy_batch(staged, batch, pool=None):
    """The batch's experts into a locked buffer on the helpers, a call per job: each holds the GIL once, and the thread that launches the kernels loses it less often (memmoves per run: 2100 tokens/s at 8K, these: 2770). What a drive must read is read first,
    in large requests (the threads would fault it in 64 KiB reads)."""
    cold = cold_ranges([(source, nbytes) for source, nbytes, _ in batch.runs])
    if cold:
        prefetch_ranges(cold)
    memory = staged.numpy()
    jobs = []
    for name, (offset, size) in zip(batch.bank.expert_tensor_names, batch.sections):
        source = getattr(batch.bank, name).numpy()
        for start in range(0, len(batch.experts), TAKE_EXPERTS):
            part = batch.experts[start:start + TAKE_EXPERTS]
            jobs.append((source, part, memory[offset + start * size:offset + (start + len(part)) * size].reshape(len(part), *source.shape[1:])))
    pool = pool or _helper_pool()
    for future in [pool.submit(_take, job) for job in jobs]:
        future.result()


class _Batch:
    """Experts of a bank in one region: their tensors one after the other, each kind in its section. runs: (source address, bytes, offset in the region). pinned: a locked copy of this layout, which crosses the bus without a CPU copy."""
    __slots__ = ("bank", "experts", "sections", "nbytes", "runs", "index", "issued", "started", "pinned", "stream")

    def __init__(self, bank, experts, index, pinned=None, stream=None):
        self.bank, self.index, self.pinned, self.stream = bank, index, pinned, stream  # index: its place in the order of the transfers
        self.issued, self.started = threading.Event(), False
        self.layout(experts)

    def layout(self, experts):
        self.experts = experts
        self.sections, self.runs, offset = [], [], 0
        breaks = np.nonzero(np.diff(experts) != 1)[0] + 1
        starts = np.concatenate(([0], breaks))[:len(breaks) + (len(experts) > 0)]
        stops = np.concatenate((breaks, [len(experts)]))[:len(starts)]
        for name in self.bank.expert_tensor_names:
            tensor = getattr(self.bank, name)
            size = tensor[0].numel() * tensor.element_size()
            address = tensor.data_ptr()
            self.sections.append((offset, size))
            self.runs.extend((address + int(experts[start]) * size, int(stop - start) * size, offset + int(start) * size) for start, stop in zip(starts, stops))
            offset += -(-len(experts) * size // 64) * 64
        self.nbytes = offset

    def addresses(self, base):
        """int64 [experts, tensors]: where each of its experts lands, from base (the address of its region)."""
        index = np.arange(len(self.experts), dtype=np.int64)[:, None]
        return base + np.array([section for section, _ in self.sections], dtype=np.int64)[None, :] + index * np.array([size for _, size in self.sections], dtype=np.int64)[None, :]


class _Lanes:
    """A short pass with locked batches its routing should read: these leave ahead of it, beside the kernels of the layers before, on a stream of their own; the experts left wait for their layer's routing, on another (in a ring, the order of the transfers has to be that of the kernels)."""
    def __init__(self, device, banks, tokens, staging, regions, kept, share, sent, note):
        ahead = max(2, regions * 2 // 3)
        self.first = ExpertStream(device, banks, tokens, [], ahead, None, 0.0, sent)
        self.second = ExpertStream(device, banks, tokens, staging, max(2, regions - ahead), kept, share, lane=1)  # a stream of its own: late, it passes the transfers of the first
        self.ahead, self.regions, self.on_cpu, self.note = False, regions, None, note

    def take(self, bank, routed, routing=None):
        self.note(bank, routed, routing)
        with self.first.cond:
            first = self.first.plan.pop(bank, [])
            for batch in first:
                if not batch.started and not routed[batch.experts].any():  # none of it is read
                    batch.layout(batch.experts[:0])
        rest = routed
        if first:
            rest = routed.copy()
            for batch in first:
                rest[batch.experts] = False
        batches = first + self.second.take(bank, rest, routing)
        self.on_cpu = self.second.on_cpu
        return batches

    def addresses(self, bank, batches):
        return self.second.addresses(bank, batches)

    def wait(self, batch):
        batch.stream.wait(batch)

    def release(self, batch):
        batch.stream.release(batch)

    def close(self):
        self.first.close()
        self.second.close()


class ExpertStream:
    """One pass over banks (in their order of use). take() gives the batches of the bank now running, wait() queues the wait for a batch's transfer on the current stream, release() gives its region back once the kernels reading it are queued."""
    def __init__(self, device, banks, tokens, staging, regions=RING_REGIONS, kept=None, share=0.0, sent=None, lane=0):
        self.device, self.region_bytes, self.regions = device, REGION_BYTES, regions
        self.staging = staging
        self.kept = kept  # (bank, experts) -> [(experts, locked copy)] sent whole, the experts left
        self.share, self.on_cpu = share, None  # a pass of few tokens: of the experts it would stream, the share the CPU takes; those of the last bank taken
        self.ring = None  # allocated by the first batches
        self.base = 0
        self.copy_streams = [copy_stream(device, lane + number) for number in range(TRANSFER_STREAMS)]
        self.note = None  # (bank, routed, routing), for each bank taken from its routing
        self.copied = [torch.cuda.Event() for _ in range(regions)]
        self.used = [torch.cuda.Event() for _ in range(regions)]
        self.transfers = [torch.cuda.Event(blocking=True) for _ in staging]
        self.pending = collections.deque()
        self.staged, self.free_buffers, self.armed = collections.deque(), collections.deque(range(len(staging))), [False] * len(staging)
        self.cond = threading.Condition()
        self.planned_count = self.issued_count = self.released_count = 0
        self.stop = False
        self.error = None
        self.plan, self.batches = {}, []
        self.ahead = tokens >= AHEAD_TOKENS
        if self.ahead:
            for bank in banks:
                self.plan[bank] = self._batches(bank, np.nonzero(bank._flags == 0)[0])
                self.pending.extend(self.plan[bank])
            self._allocate()
        elif sent is not None:  # locked batches sent before their routing is known: the lane of a short pass that runs ahead (_Lanes)
            for bank, parts in sent.items():
                self.plan[bank] = [_Batch(bank, members, self.planned_count + number, pinned, self) for number, (members, pinned) in enumerate(parts)]
                self.planned_count += len(parts)
                self.batches.extend(self.plan[bank])
                self.pending.extend(self.plan[bank])
            self.ring = torch.empty(min(self.regions, self.planned_count) * self.region_bytes, dtype=torch.uint8, device=self.device)
            self.base = self.ring.data_ptr()
            self.regions = self.ring.numel() // self.region_bytes
        self.threads = [threading.Thread(target=self._stage, name="mmgp_expert_stage", daemon=True)] + [threading.Thread(target=self._transfer, args=(stream,), name="mmgp_expert_transfer", daemon=True) for stream in self.copy_streams]
        for thread in self.threads:
            thread.start()

    def _allocate(self):
        if self.ring is None and self.planned_count:
            self.ring = torch.empty(min(self.regions, self.planned_count) * self.region_bytes if self.ahead else self.regions * self.region_bytes, dtype=torch.uint8, device=self.device)
            self.base = self.ring.data_ptr()
            self.regions = self.ring.numel() // self.region_bytes

    def _batches(self, bank, experts, routing=None):
        count = batch_experts(bank)
        whole, experts = ([], experts) if self.kept is None else self.kept(bank, experts)
        if routing is not None and self.share:  # those routed by the fewest tokens: they cross the bus whole for little work, the CPU reads their rows where they are
            few = experts[routing[experts] <= CPU_SHARE_ROWS]
            few = few[np.argsort(routing[few], kind="stable")]
            if self.share < 0:  # where both sides end together: each expert the CPU takes costs its rows there, and spares the bus its bytes
                bus = (len(experts) - np.arange(1, len(few) + 1)) * bank.expert_nbytes / STREAM_RATE + sum(part.numel() for _, part in whole) / STREAM_RATE
                self.on_cpu = few[:int(np.searchsorted(np.cumsum(routing[few]) * pair_seconds[0] - bus, 0))]
            else:
                self.on_cpu = few[:round(self.share * len(experts))]
            if len(self.on_cpu):
                kept = np.ones(bank.num_experts, dtype=bool)
                kept[self.on_cpu] = False
                experts = experts[kept[experts]]
        parts = whole + [(experts[start:start + count], None) for start in range(0, len(experts), count)]
        batches = [_Batch(bank, part, self.planned_count + number, pinned, self) for number, (part, pinned) in enumerate(parts)]
        self.planned_count += len(batches)
        self.batches.extend(batches)
        return batches

    def _fail(self, error):
        """Reported to the thread that waits for the batches."""
        self.error = error
        with self.cond:
            self.pending.clear()
        for batch in self.batches:
            batch.issued.set()

    def _stage(self):
        try:
            with torch.inference_mode():
                while True:
                    with self.cond:
                        while not (self.pending and (self.free_buffers or self.pending[0].pinned is not None)) and not self.stop:
                            self.cond.wait()
                        if self.stop:
                            return
                        batch = self.pending.popleft()
                        batch.started = True
                        if batch.pinned is not None:  # nothing to copy: its transfer takes its turn
                            self.staged.append((batch, None))
                            self.cond.notify_all()
                            continue
                        slot = self.free_buffers.popleft()
                    if self.armed[slot]:
                        self.transfers[slot].synchronize()  # the transfer that read this buffer last is complete
                    _copy_batch(self.staging[slot], batch)
                    with self.cond:
                        self.staged.append((batch, slot))
                        self.cond.notify_all()
        except BaseException as error:
            self._fail(error)

    def _transfer(self, copies):
        try:
            with torch.inference_mode():
                while True:
                    with self.cond:
                        while not (self.staged and self.staged[0][0].index < self.released_count + self.regions) and not self.stop:
                            self.cond.wait()
                        if self.stop:
                            return
                        batch, slot = self.staged.popleft()
                    region = batch.index % self.regions
                    with torch.cuda.stream(copies):
                        if batch.index >= self.regions:
                            copies.wait_event(self.used[region])
                        start = region * self.region_bytes
                        self.ring[start:start + batch.nbytes].copy_((batch.pinned if slot is None else self.staging[slot])[:batch.nbytes], non_blocking=True)
                        self.copied[region].record(copies)
                        if slot is not None:
                            self.transfers[slot].record(copies)
                            self.armed[slot] = True
                    with self.cond:
                        self.issued_count += 1
                        if slot is not None:
                            self.free_buffers.append(slot)
                        self.cond.notify_all()
                    batch.issued.set()
        except BaseException as error:
            self._fail(error)

    def take(self, bank, routed, routing=None):
        """The batches of bank in their order: planned ahead, or now from its routing (bool [experts]; routing: its tokens per expert). on_cpu then gives the experts left to the CPU."""
        self.on_cpu = None
        batches = self.plan.pop(bank, None)
        if batches is not None:
            with self.cond:
                for batch in batches:
                    if not batch.started and (batch.pinned is None or not routed[batch.experts].any()):  # what the routing does not read does not cross the bus (a locked copy crosses whole: one transfer)
                        batch.layout(batch.experts[routed[batch.experts]])
        else:
            if self.note is not None:
                self.note(bank, routed, routing)
            batches = self._batches(bank, np.nonzero(routed & (bank._flags == 0))[0], routing)
            self._allocate()
            with self.cond:
                self.pending.extend(batches)
                self.cond.notify_all()
        return batches

    def addresses(self, bank, batches):
        """int64 [experts, tensors] on the GPU: where the experts of the batches of a bank land, zero for the others."""
        table = np.zeros((bank.num_experts, len(bank.expert_tensor_names)), dtype=np.int64)
        for batch in batches:
            table[batch.experts] = batch.addresses(batch.stream.base + batch.index % batch.stream.regions * batch.stream.region_bytes)
        return torch.from_numpy(table).to(self.device)

    def wait(self, batch):
        """What is queued next on the current stream runs after the batch's transfer."""
        batch.issued.wait()
        if self.error is not None:
            raise self.error
        torch.cuda.current_stream().wait_event(self.copied[batch.index % self.regions])

    def release(self, batch):
        self.used[batch.index % self.regions].record(torch.cuda.current_stream())
        with self.cond:
            self.released_count += 1
            self.cond.notify_all()

    def close(self):
        with self.cond:
            self.stop = True
            self.cond.notify_all()
        for thread in self.threads:
            thread.join()
        if self.ring is not None:  # kernels and copies of this pass may still be queued: the next pass overwrites its memory from another stream
            torch.cuda.current_stream().synchronize()
            for copies in self.copy_streams:
                copies.synchronize()
        self.ring = None
        self.staging = self.pending = self.staged = None
