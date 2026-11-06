# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""State rings: per-block tensors a model computes once and reads at every pass (a cached attention prefix of reference images or of a
history), kept in RAM and streamed to the GPU block by block, as MMGP streams swapped weights: the VRAM they hold is that of two blocks.

The model stores each block's tensors once (store: a blocking copy into RAM; with current=True also into the block's slot, for the pass
that computed them): its GPU tensors can then go. A tensor may be stored as a list of row pieces (joined along its first dimension, each
piece copied into the record: no joined copy first). The ring is attached to the blocks (the model's tower, a ModuleList): before each block
runs at every pass, its forward hook fetches its tensors (waits for their copy) and queues those of the next block on MMGP's transfer
stream (the stream of the weight transfers), so the copy runs while the block computes. Block i always reads the slot i % slots, separate
allocations of each tensor: fixed addresses, which CUDA graphs adopt as the static buffers of the block's arguments (no copy at replay).
The blocks must be called one by one (not replayed as one graph spanning several blocks).

The records and the slots keep their memory (and their locks) across reset(): blocks stored again with fewer rows (the first dimension of
each tensor, the others unchanged) use views of them, and only the rows stored are copied (to the slots, and through MMGP's staging ring);
more rows make them again. A ring made with resident=True keeps the state of some blocks in VRAM instead of streaming it (no copy at each
pass), when the model's memory profile preloads blocks (a VRAM preload budget, a dynamic preload): only blocks whose weights MMGP keeps in
VRAM, in groups of `slots` consecutive blocks (the streamed blocks keep alternating their slots), as far as the room the preload leaves
once the weights have theirs (offload.state_room minus the other resident rings of the model: weights first). The VRAM this state holds
counts as the model's weights for the dynamic preload's measures; it is released first under VRAM pressure (the bytes really freed are
reported, and they shrink the room of the next plans: offload.state_pressure), when the room shrinks (trimmed before the next pass; it grows
back only when blocks are stored again), when the weights of its block leave the VRAM and when the model is unloaded, the block then
streaming its record again. Resident tensors have a spare row, so that CUDA graphs never adopt them as their static storage (they copy them
into their own: dropping them frees their memory). Its caller reads the list store() returns at each pass (the ring rebinds its items: the
block's slot, or its resident tensors).

The ring follows the schedule MMGP decided for the tower's blocks (when its blocks belong to a managed model): it fetches as many blocks
ahead as MMGP's prefetch window for the model (window + 1 slots), and locks the records (cudaHostRegister, asynchronous transfers) of the
blocks whose weights MMGP streams from pinned RAM first, then of the blocks it keeps in VRAM, as far as the reserved RAM left by the offload
object allows (its pinned budget, shared with the pinned weights and pool), spread over them as MMGP spreads its pinned blocks; the records
of the blocks whose weights cross MMGP's staging ring, and those left without room, stay in pageable RAM and are moved in advance by a copier
thread into locked staging buffers, from which they are transferred asynchronously too (without room for the staging buffers: synchronously
for the host). With MMGP's staging ring, the pageable records of the blocks whose weights it streams cross it with them, and the records
of the blocks whose weights it keeps in VRAM, which it never stages, are locked first (the ring's own staging buffers move the others
ahead, as the blocks MMGP keeps in VRAM change under a dynamic preload). For a model MMGP does not pin (profile 5), the ring locks only the
records of the blocks whose weights smartPinning pins and has no staging buffers of its own: the other records cross MMGP's staging ring
with their blocks (synchronously without it). The
placement is decided when the first block is stored (after the pass that computed them, MMGP's plans made). release()
frees the slots and the RAM; unlocking waits for the GPU (cudaHostUnregister), so call it at a quiet point (the end of a generation).

A ring made with resident="dynamic" (a cache the model computes on the GPU: needs MMGP's VRAM allocator) keeps its state in VRAM first and
in RAM only for the blocks that do not stay there. The blocks stored with current=True keep their state in VRAM, by groups of `count`
consecutive blocks, while the VRAM left beside the peak of the pass that computes them allows it (the dynamic preload's margin), without
any copy to RAM; the others go to RAM and stream as above. The dynamic preload of the model plans the groups kept in VRAM with its weights
at each cycle (mmgp.auto_preload, plan()): a group that leaves the VRAM is written back to RAM first (once per store), one that comes back
is filled when its blocks would have been streamed anyway (from their slot, or their record instead of their slot), whatever the blocks
whose weights MMGP keeps in VRAM. VRAM pressure writes groups back and streams them (no plan: the groups stay where they are until then).
Its records are allocated when the first block is stored, unlocked (RAM only committed until written), and locked at the start of a pass
once written, as far as the reserved RAM allows; its slots are allocated when a block first streams. They count with the state kept in VRAM
as the model's weights. Its tensors in VRAM have their exact shapes: CUDA graphs adopt them as the static buffers of the blocks' arguments,
as they adopt a cache kept on the GPU (no copy at each replay, a graph per guidance branch), and let them go when they leave the VRAM.

A block whose state changes by rows at each pass (a cache extended or rolled: the keys of a window of frames) replaces them with
update(block, rows) during its pass, after its fetch: the rows are written into the GPU tensors the pass reads (the list store() returned),
by update or by the caller itself, and only those rows reach its record: from locked RAM asynchronously, on a stream of the ring's own
(the transfers reading the record, and the writes of the GPU tensors it reads from, wait for that copy), blocking into pageable RAM. The
copies of the record staged before (by the ring's copier or by MMGP's staging ring with the block's weights) are not transferred: a record
staged for an older revision of the block is read again. A dynamic ring keeping the block in VRAM marks its record stale instead (written
back whole when its group leaves the VRAM). skip (store or update): rows of the block that its next passes replace before reading them,
which its fetches do not bring from RAM (its slot keeps whatever it held there). Rings without updates behave as before.

Pageable records that no staging buffer of the ring's own and no staging by MMGP with their block's weights would move (a record too large
to cross MMGP's staging ring with its block, a model whose ring is made later) cross the lane (_Lane): two staging rings of MMGP's own kind
with RAM, copier threads and a stream of their own, shared by the rings of the process, apart from the staging ring of the weights. A record
goes to the GPU in pieces, each copied into the lane ahead and transferred as soon as it is there, while the previous block computes; the
rows an update changes go to the lane on the save stream, then by the lane's copier into the record, without the GPU waiting. Without the
lane (its RAM refused), these copies are synchronous, as they were.

reserve(block, shapes) gives a pass the GPU tensors it fills with a block's state before storing it (store(block, those tensors,
current=True, rows=...)): its slot, or the tensors a dynamic ring keeps it in. store(rows=) writes only those rows of the record (the
others are not read before being replaced). A ring of blocks spaced out in their model (blocks of one kind among others) has as many slots
as it has blocks among window + 1 consecutive blocks of the model: one when they are further apart, its next block being prefetched when the
model's block before it starts (lead()).

A minimal ring (minimal=True, the least VRAM) has one slot, made when its block starts and released once the block has run (after the
copies reading it): the GPU holds the state of the block running only, nothing between passes; its blocks are not prefetched (each record
transferred when its block starts, from locked RAM or through the lane; the updated rows go back without the GPU waiting)."""
import ctypes
import functools
import itertools
import threading
from concurrent.futures import ThreadPoolExecutor

import torch
from tqdm import tqdm

from . import allocator, auto_preload, cuda_graphs, memory_pressure, offload, pinned

SLOTS = 2  # GPU slots without a managed model: the block running and the next one (window + 1 with MMGP's prefetch window)
STAGING = 2  # locked buffers of pageable records (window + 1 with MMGP), one per record transferred or being copied ahead
RESIDENT = True  # rings made with resident=True keep state in VRAM where the preload leaves room; False streams every block (comparisons)
LANE_BYTES = (192 << 20, 64 << 20)  # locked RAM of the lane: its ring towards the GPU, its ring towards RAM
LANE_PIECE = 32 << 20  # bytes of a record moved at once through the lane
MMGP_STAGES_RECORDS = True  # comparison switch: False moves every pageable record through the lane, none through MMGP's staging ring
_ALIGN = offload.BLOCK_ALIGN
_serials = itertools.count()  # the order the rings were made: the dynamic preload ranks their groups in a stable order
_lane, _lane_guard = None, threading.Lock()


def _locked(size, device):
    """size bytes of RAM of its own, locked, and its registration (None when it could not be locked)."""
    ram, owner = pinned.block(size)
    registration = pinned.Registration(owner, ram.data_ptr(), size, device, kind="state")
    return (ram, registration) if int(registration.register()) == 0 else (ram, None)


def _run_rings(block, previous, *args, **kwargs):
    rings = block.__dict__.get("_mm_state_rings", ())
    for ring, index in rings:
        ring.fetch(index)
    for ring, index in block.__dict__.get("_mm_state_leads", ()):
        ring.lead(index)
    result = previous(*args, **kwargs)
    for ring, index in rings:
        if ring.minimal:
            ring.end(index)
    return result


def _hook(module, kind, entry):
    """Attaches (ring, index) to module's list kind (_mm_state_rings: fetched before it runs, _mm_state_leads: prefetched before it runs),
    its forward then running the rings first."""
    if "_mm_state_rings" not in module.__dict__ and "_mm_state_leads" not in module.__dict__:
        module._mm_state_previous = module.__dict__.get("forward")
        module.forward = functools.partial(_run_rings, module, module.forward)
    module.__dict__.setdefault(kind, []).append(entry)


def _unhook(module, ring):
    """Detaches ring from module, whose forward is restored when no ring is left."""
    for kind in ("_mm_state_rings", "_mm_state_leads"):
        entries = module.__dict__.get(kind)
        if entries is not None:
            entries[:] = [entry for entry in entries if entry[0] is not ring]
            if not entries:
                del module.__dict__[kind]
    if "_mm_state_rings" not in module.__dict__ and "_mm_state_leads" not in module.__dict__ and "_mm_state_previous" in module.__dict__:
        previous = module.__dict__.pop("_mm_state_previous")
        if previous is None:
            del module.forward
        else:
            module.forward = previous


class _Lane:
    """The rings' path for pageable records: two staging rings of MMGP's own kind (offload._StagingRing), with RAM, copier threads, helpers
    and a stream of their own, apart from the staging ring of the weights, its copier and the transfer stream. Shared by the rings of the
    process (_acquire_lane), closed with the last of them (a quiet point)."""

    def __init__(self, device):
        self.up = offload._StagingRing(LANE_BYTES[0])
        try:
            self.down = offload._StagingRing(LANE_BYTES[1])
        except BaseException:
            self.up.close()
            raise
        self.stream, self.users = torch.cuda.Stream(device), 0

    def close(self):
        self.up.close()
        self.down.close()

    def upload(self, pieces, waits, ready, after=None):
        """pieces: [(address in RAM, bytes, target: a uint8 GPU tensor of that many bytes)], copied in turn into the up ring by its copier and
        helpers (after the future `after`, a copy into the same record), each transferred on the lane's stream after the events `waits`
        (recorded already); `ready` is recorded after the last. Returns the future of the job, done once every transfer is queued."""
        ring, stream = self.up, self.stream

        def job():
            if after is not None:
                after.result()
            with torch.inference_mode():
                for event in waits:
                    stream.wait_event(event)
                for address, nbytes, target in pieces:
                    region = ring._reserve(nbytes)  # waits for the transfer that read the oldest region
                    offload._copy_runs(ring.buffer.data_ptr() + region[0], [(address, nbytes, 0, None)], ring.helpers)
                    with torch.cuda.stream(stream):
                        target.copy_(ring.buffer[region[0]:region[0] + nbytes], non_blocking=True)
                    event = torch.cuda.Event()
                    event.record(stream)
                    ring.release(region, event)
                ready.record(stream)

        return ring.copier.submit(job)

    def download(self, stream, pieces):
        """pieces: [(source: a uint8 GPU tensor, bytes, address in RAM)], each copied on stream (after the work queued on it) into the down
        ring (waiting while its copier empties it), then by that copier into RAM. Returns (the event of the last copy on stream, the future
        of the last copy into RAM: the copier takes them in order)."""
        ring, event, future = self.down, None, None
        for source, nbytes, address in pieces:
            region = ring._reserve(nbytes)
            with torch.cuda.stream(stream):
                ring.buffer[region[0]:region[0] + nbytes].copy_(source, non_blocking=True)  # into locked RAM, read once the event is reached
            event = torch.cuda.Event()
            event.record(stream)
            future = ring.copier.submit(self._unload, region, event, nbytes, address)
        return event, future

    def _unload(self, region, event, nbytes, address):
        event.synchronize()
        offload._copy_runs(address, [(self.down.buffer.data_ptr() + region[0], nbytes, 0, None)], self.down.helpers)
        self.down.release(region, torch.cuda.Event())  # an event never recorded counts as completed: the region is free


def _acquire_lane(device, budget):
    """The lane, made by its first user if the reserved RAM (budget, 0: none) and the driver allow it; None otherwise."""
    global _lane
    with _lane_guard:
        if _lane is None:
            if not pinned.fits(sum(LANE_BYTES), budget):
                return None
            try:
                _lane = _Lane(device)
            except (RuntimeError, MemoryError):
                return None
        _lane.users += 1
        return _lane


def _release_lane(lane):
    global _lane
    with _lane_guard:
        lane.users -= 1
        if lane.users == 0:
            if lane is _lane:
                _lane = None
            lane.close()


def _flat(tensor):
    """The bytes of a contiguous tensor, as a flat uint8 tensor."""
    return tensor.reshape(-1).view(torch.uint8)


def _joined(tensor):
    """(shape, dtype) of a stored tensor, given whole or as a list of row pieces."""
    if isinstance(tensor, (list, tuple)):
        return (sum(piece.shape[0] for piece in tensor), *tensor[0].shape[1:]), tensor[0].dtype
    return tuple(tensor.shape), tensor.dtype


def _copy_rows(target, tensor):
    if isinstance(tensor, (list, tuple)):
        start = 0
        for piece in tensor:
            target[start:start + piece.shape[0]].copy_(piece)
            start += piece.shape[0]
    else:
        target.copy_(tensor)


def _allocated(device):
    # the bytes the active allocator has handed out on the device (MMGP's VMM allocator, whose pressure callback calls rings, or PyTorch's)
    return allocator.stats(device)["allocated"] if allocator.active == "vmm" else torch.cuda.memory_allocated(device)


def _packed(shapes):
    """[(shape, dtype, offset, nbytes)] of tensors packed in a record, and its bytes."""
    offset, layout = 0, []
    for shape, dtype in shapes:
        nbytes = torch.Size(shape).numel() * dtype.itemsize
        layout.append((tuple(shape), dtype, offset, nbytes))
        offset += -(-nbytes // _ALIGN) * _ALIGN
    return layout, offset


class StateRing:
    def __init__(self, blocks, device, manager=None, name="state", modules=None, model_id=None, entries=None, resident=False, minimal=False):
        """blocks: number of blocks whose state is stored (indices 0 .. blocks - 1, fetched in that order at each pass); modules: the
        blocks themselves (a ModuleList), whose forward then fetches their state; model_id / entries: their model and block entries in the
        offload object (manager), whose schedule the ring follows; resident: blocks may keep their state in VRAM (see the module), "dynamic":
        in VRAM first, placed by the model's dynamic preload; minimal: one slot, held only while its block runs (see the module)."""
        if minimal and resident:
            raise ValueError("A minimal state ring keeps no state in VRAM")
        self.minimal = minimal
        self.blocks, self.device, self.manager, self.name = blocks, torch.device(device), manager, name
        self.model_id, self.entries, self.count, self.staging_count = model_id, entries, SLOTS, STAGING
        self.capacity = None  # [(shape, dtype)] of the tensors the records and the slots can hold (the largest stored since they were made)
        self.layout = None  # [(shape, dtype, offset, nbytes)] of a block's tensors stored now, packed in its record
        self.record = 0  # bytes of a block's record (its capacity)
        self.full = {}  # block -> (RAM tensor of its whole record, locked)
        self.records = {}  # block -> (RAM tensor of the bytes it stores now, locked): what MMGP's staging ring copies
        self.registrations, self.memory, self.staging, self.copier = [], [], [], None
        self.slots, self.views, self.held, self.ready, self.stored = None, None, [None] * self.count, [None] * self.count, set()
        self.handles = {}  # block -> the list store() returned: the views of its slot, or of its resident tensors
        self.resident = {}  # block -> its own GPU tensors (capacity shapes) holding its state
        self.resident_ready = {}  # block -> event of the copy filling its resident tensors on the transfer stream (fetch waits for it)
        self.allow_resident = bool(resident) and manager is not None and entries is not None
        self.dynamic = self.allow_resident and resident == "dynamic"
        self.serial = next(_serials)
        self.written = set()  # dynamic: blocks whose record holds their state (the others are only in VRAM)
        self.wanted = None  # dynamic: the groups the dynamic preload keeps in VRAM (None: no plan, they stay where they are)
        self.promote = set()  # dynamic: blocks of wanted groups, filled in VRAM when they would be streamed
        self.deferred = set()  # dynamic: groups to leave the VRAM at the next pass (the block running was one of theirs)
        self.refused = set()  # dynamic: groups not kept in VRAM by the pass storing them (no room left beside its peak)
        self.admitted = set()  # dynamic: groups the pass storing them keeps in VRAM
        self.reported = None
        self.storing = False  # dynamic: the pass storing the blocks runs (what it keeps in VRAM counts as its activations: it is measured)
        self.last = None  # dynamic: the block last stored or fetched
        self.planning, self.thread = False, None  # residency being planned (its allocations); the thread running the blocks
        self.staged = {}  # block -> (staging buffer index, future of its copy)
        self.mmgp_staged = set()  # pageable records staged in MMGP's staging ring with their blocks (offload._schedule_ring / _transfer)
        self.generation = 0  # bumped when the stored records change: a record MMGP staged before then is not transferred
        self.current = None  # the block last fetched: its compute is not queued yet when MMGP transfers the next blocks
        self.locked_bytes = self.pageable_bytes = 0
        self.owner, self.stride = None, 0  # dynamic: the owner of the records' RAM (locked in place once written), a record's pages
        self.skips = {}  # block -> (start, stop): rows its next passes replace before reading them (update), left out of its fetches
        self.revisions = {}  # block -> updates of its record: what MMGP's staging ring staged of an older revision is not transferred
        self.saving = {}  # block -> event of the copy of its updated rows into its locked record (on save_stream)
        self.slot_saving = [None] * self.count  # slot -> event of the last copy of updated rows read from it
        self.save_stream = None
        self.lane, self.lane_refused = None, False  # the lane (_Lane) moving the pageable records nothing else moves ahead
        self.pending = {}  # slot -> future of the lane's job filling it (its transfers queued once done)
        self.writebacks = {}  # block -> future of the lane's copy of its updated rows into its pageable record
        self.prepared = False  # the records, slots and layout of the blocks stored since the last reset are made
        self.leads = []  # the modules whose forward prefetches a block of this ring (blocks spaced out in their model)
        self.modules = []
        for index, module in enumerate(modules or ()):
            _hook(module, "_mm_state_rings", (self, index))
            self.modules.append(module)
        if self.allow_resident:
            manager.state_rings.add(self)

    @property
    def locked(self):
        return self.layout is not None and self.pageable_bytes == 0

    @property
    def resident_bytes(self):
        return sum(t.untyped_storage().nbytes() for tensors in self.resident.values() for t in tensors)

    @property
    def vram_bytes(self):
        """The VRAM counted with the model's weights: the state kept in VRAM, and the slots of a dynamic ring."""
        slots = sum(t.untyped_storage().nbytes() for slot in self.slots for t in slot) if self.dynamic and self.slots is not None else 0
        return self.resident_bytes + slots

    @property
    def groups(self):
        """Number of whole groups of `count` consecutive blocks, the units kept in VRAM."""
        return self.blocks // self.count

    @property
    def group_bytes(self):
        """VRAM of a group kept in VRAM (a dynamic ring's tensors have their exact shapes)."""
        return self.count * _packed([((shape[0] + (not self.dynamic), *shape[1:]), dtype) for shape, dtype in self.capacity])[1]

    @property
    def slot_bytes(self):
        return self.count * self.record

    def stages(self, block):
        """Whether MMGP's staging ring may move the block's record with its weights (offload._schedule_ring): stored in RAM, not locked, not
        being written by the lane."""
        writing = self.writebacks.get(block)
        return (block in self.stored and (writing is None or writing.done())
                and (not self.dynamic or block in self.written and block not in self.resident and not self.records[block][1]))

    def _room(self):
        """The VRAM the model's preload leaves to the state this ring keeps in VRAM: its room, less the state the model's other rings keep."""
        return self.manager.state_room(self.model_id) - (self.manager.state_bytes(self.model_id) - self.resident_bytes)

    def _stream(self):
        stream = getattr(self.manager, "transfer_stream", None)
        if stream is None:
            stream = self.__dict__.setdefault("own_stream", torch.cuda.Stream(self.device))
        return stream

    def _lock_groups(self):
        """The blocks whose records are locked first, in groups, following MMGP's schedule for the tower: the blocks whose weights it streams
        from pinned RAM, then those it keeps in VRAM (the other way round with MMGP's staging ring, which stages the records of the blocks it
        streams, but not of those it keeps in VRAM); the records of the blocks whose weights cross its staging ring are staged. Without a
        managed model, all the blocks."""
        manager = self.manager
        if self.entries is not None and hasattr(manager, "needs_staging"):
            preloaded = manager.preloaded_blocks_per_model.get(self.model_id, ())
            resident = [block for block, entry in enumerate(self.entries) if entry not in manager.blocks_of_modules or entry.split("/", 1)[1] in preloaded]
            groups = ([block for block, entry in enumerate(self.entries) if block not in resident and not manager.needs_staging(entry)], resident)
            if self.model_id not in manager.pinned_models:  # a model MMGP does not pin (profile 5): only the blocks smartPinning n pins
                groups = groups[:1]
            elif getattr(manager, "ring", None) is not None:
                groups = groups[::-1]
        else:
            groups = (list(range(self.blocks)),)
        return groups

    def _plan(self, room, own_staging):
        """The blocks whose records are locked (_lock_groups), as many as room allows, spread over each group as MMGP spreads its pinned
        blocks."""
        groups = self._lock_groups()
        candidates = sum(len(group) for group in groups)
        lockable = room // self.record
        if lockable < self.blocks and own_staging:  # some records staged by the ring itself: room for its staging buffers first
            lockable = min(candidates, max(0, room - self.staging_count * self.record) // self.record)
        locked = []
        for group in groups:
            take = min(len(group), lockable - len(locked))
            if take > 0:
                locked += [group[int(name[1:])] for name in offload._spread([f"b{position}" for position in range(len(group))], take / len(group))]
        return set(locked)

    def _allocate(self, shapes):
        self.capacity = list(shapes)
        _, self.record = _packed(shapes)
        if self.minimal:  # one slot, made when a block starts
            self.count = self.staging_count = 1
            self.held, self.ready, self.slot_saving = [None], [None], [None]
        elif self.model_id is not None and hasattr(self.manager, "model_prefetch_window"):  # as many blocks ahead as MMGP's weights
            window = max(1, self.manager.model_prefetch_window(self.model_id))
            # slots: block i reads slot i % count; a count dividing the blocks keeps the blocks sharing a slot count blocks apart across the
            # wrap to the next pass too (the last block and the first ones of the next pass, which MMGP prefetches during it)
            self.count = next((count for count in range(window + 1, 2 * window + 3) if self.blocks % count == 0), window + 1)
            if self.entries is not None and self._spaced(window):  # one slot, the next block prefetched when the model's block before it starts
                self.count = 1
            self.staging_count = self.count
            self.held, self.ready, self.slot_saving = [None] * self.count, [None] * self.count, [None] * self.count
        budget = getattr(self.manager, "max_reservable_memory", 0)
        room = budget - pinned.total_bytes - offload._pool.reserved() if budget > 0 else 0
        # with MMGP's staging ring, the pageable records cross it with their blocks' weights while MMGP streams them (fetch decides per block),
        # if a record fits in it with the largest block of the ring; the others cross the lane
        mmgp_ring = getattr(self.manager, "ring", None) if self.entries is not None and MMGP_STAGES_RECORDS and not self.minimal else None
        largest = max((self.manager.blocks_of_modules_sizes.get(entry, 0) for entry in self.entries), default=0) if mmgp_ring is not None else 0
        mmgp_staging = mmgp_ring is not None and self.record + largest <= mmgp_ring.size
        # buffers of the ring's own for the records MMGP does not stage, but for a model MMGP does not pin (profile 5): without MMGP's ring
        # its records then cross synchronously, as its weights
        own_staging = self.entries is None or self.model_id in self.manager.pinned_models
        if self.dynamic:  # every record in RAM of its own pages, unlocked until written (_lock_records); no slot until a block streams
            self.stride = -(-self.record // pinned.PAGE) * pinned.PAGE
            ram, self.owner = pinned.block(self.stride * self.blocks)
            self.memory.append(ram)
            for block in range(self.blocks):
                self.full[block] = (ram[block * self.stride:block * self.stride + self.record], self.owner is None)  # None: driver pinned
            if self.owner is None:
                self.locked_bytes = self.record * self.blocks
            else:
                self.pageable_bytes = self.record * self.blocks
                if mmgp_staging:
                    for block in range(self.blocks):
                        self.manager.state_records.setdefault(self.entries[block], []).append((self, block))
                    self.mmgp_staged = set(range(self.blocks))
            return
        locked = self._plan(room, own_staging)
        if locked:
            ram, registration = _locked(self.record * len(locked), self.device)
            if registration is None:
                locked = set()
            else:
                self.registrations.append(registration)
                self.memory.append(ram)
                for position, block in enumerate(sorted(locked)):
                    self.full[block] = (ram[position * self.record:(position + 1) * self.record], True)
                self.locked_bytes = ram.numel()
        pageable = [block for block in range(self.blocks) if block not in locked]
        if pageable:
            ram = torch.empty(self.record * len(pageable), dtype=torch.uint8, device="cpu")
            self.memory.append(ram)
            for position, block in enumerate(pageable):
                self.full[block] = (ram[position * self.record:(position + 1) * self.record], False)
            self.pageable_bytes = ram.numel()
            if mmgp_staging:
                for block in pageable:
                    self.manager.state_records.setdefault(self.entries[block], []).append((self, block))
                self.mmgp_staged = set(pageable)
            if own_staging and room >= (len(locked) + self.staging_count) * self.record:
                self._make_staging()
            lane = self._lane() if not mmgp_staging and self.copier is None else None
            tqdm.write(f"[mmgp] {self.name}: {self.locked_bytes / 2 ** 20:.0f} MB locked, {self.pageable_bytes / 2 ** 20:.0f} MB in pageable RAM "
                       + ("moved ahead with their blocks through MMGP's staging ring" + (", or through locked staging buffers" if self.copier else "") if mmgp_staging
                          else "moved ahead through locked staging buffers" if self.copier
                          else "moved ahead in pieces through the state lane" if lane is not None
                          else "transferred synchronously (no reserved RAM left)" if own_staging else "transferred synchronously, as the weights of its model"))
        if not self.minimal:
            self._ensure_slots()

    def _spaced(self, window):
        """Whether the ring's blocks are more than window blocks apart in their model's tower (blocks of one kind among others): one slot then
        holds their state, each block prefetched when the tower's block before it starts (lead). Attaches those leads."""
        names = [entry.rsplit(".", 1) for entry in self.entries]
        if len(names) < 2 or any(len(name) != 2 or not name[1].isdigit() for name in names) or len({prefix for prefix, _ in names}) != 1:
            return False
        prefix = names[0][0] + "."
        size = sum(1 for entry in self.manager.blocks_of_modules if entry.startswith(prefix) and entry[len(prefix):].isdigit())
        positions = [int(index) for _, index in names]
        members = set(positions)
        if size <= window + 1 or any((position + step) % size in members for position in positions for step in range(1, window + 1)):
            return False
        model, name = self.manager.models[self.model_id], prefix.split("/", 1)[1]
        for block, position in enumerate(positions):
            module = model.get_submodule(name + str((position - 1) % size))
            _hook(module, "_mm_state_leads", (self, block))
            self.leads.append(module)
        return True

    def _lane(self):
        """The lane for the ring's pageable records (acquired at its first need), None if its RAM was refused."""
        if self.lane is None and not self.lane_refused:
            self.lane = _acquire_lane(self.device, getattr(self.manager, "max_reservable_memory", 0))
            self.lane_refused = self.lane is None
        return self.lane

    def _make_staging(self):
        """The locked buffers into which the copier thread moves pageable records ahead of their transfer."""
        for _ in range(self.staging_count):
            buffer, registration = _locked(self.record, self.device)
            if registration is None:
                break
            self.registrations.append(registration)
            self.staging.append([buffer, None])  # [locked buffer, event of the transfer that reads it]
        if len(self.staging) == self.staging_count:
            self.copier = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mmgp_state_ring")
        else:
            self._unlock(self.staging)

    def _ensure_slots(self):
        """The GPU slots of the streamed blocks (a dynamic ring makes them when a block first streams), the streamed blocks' views of them."""
        if self.slots is not None:
            return
        self.slots = [[torch.empty(shape, dtype=dtype, device=self.device) for shape, dtype in self.capacity] for _ in range(self.count)]
        self.ready, self.held, self.slot_saving = [torch.cuda.Event() for _ in range(self.count)], [None] * self.count, [None] * self.count
        if self.layout is not None:
            self.views = [[tensor[:shape[0]] for tensor, (shape, _, _, _) in zip(slot, self.layout)] for slot in self.slots]
            for block in self.stored - set(self.resident):
                self.handles[block][:] = self.views[block % self.count]
        if self.dynamic and not self.storing:
            self.manager.state_changed(self.model_id)  # the slots count with the model's weights

    def _fits(self, shapes):
        return len(shapes) == len(self.capacity) and all(dtype == held_dtype and len(shape) == len(held) and shape[1:] == held[1:] and shape[0] <= held[0]
                                                         for (shape, dtype), (held, held_dtype) in zip(shapes, self.capacity))

    def _set_layout(self, shapes):
        """The tensors stored from now on (after a reset): their packing in the records, the views of the slots and of the resident tensors."""
        self.layout, used = _packed(shapes)
        self.records = {block: (ram[:used], locked) for block, (ram, locked) in self.full.items()}
        self.views = None if self.slots is None else [[tensor[:shape[0]] for tensor, (shape, _) in zip(slot, shapes)] for slot in self.slots]
        for block in range(self.blocks):
            self.handles.setdefault(block, [])[:] = (self._resident_views(block) if block in self.resident
                                                     else [] if self.views is None else self.views[block % self.count])

    def _resident_views(self, block):
        return [tensor[:shape[0]] for tensor, (shape, _, _, _) in zip(self.resident[block], self.layout)]

    def _unlock(self, staging):
        for buffer, _ in staging:
            for registration in [r for r in self.registrations if r.pointer == buffer.data_ptr()]:
                registration.close()
                self.registrations.remove(registration)
        staging.clear()

    def _views(self, ram):
        return [ram[offset:offset + nbytes].view(dtype).view(shape) for shape, dtype, offset, nbytes in self.layout]

    def _weights_resident(self, block):
        """Whether MMGP keeps the weights of the block in VRAM now (its base, or a preloaded block)."""
        entry, manager = self.entries[block], self.manager
        return entry not in manager.blocks_of_modules or entry.split("/", 1)[1] in manager.preloaded_blocks_per_model.get(self.model_id, ())

    def _mmgp_stages(self, block):
        """Whether MMGP's staging ring moves the block's record with its weights: those it streams, not those it keeps in VRAM now."""
        return block in self.mmgp_staged and not self._weights_resident(block)

    def _plan_resident(self):
        """The blocks keeping their state in VRAM, decided when blocks are stored after a reset: whole groups of `count` consecutive blocks
        whose weights MMGP keeps in VRAM, spread over the tower, as many as the room the preload leaves to the state (offload.state_room)."""
        per_group = self.count * _packed([((shape[0] + 1, *shape[1:]), dtype) for shape, dtype in self.capacity])[1]  # with their spare row
        groups = [range(start, start + self.count) for start in range(0, self.blocks - self.count + 1, self.count)]
        eligible = [group for group in groups if all(self._weights_resident(block) for block in group)]
        take = min(len(eligible), max(0, int(self._room() // per_group))) if RESIDENT else 0
        chosen = {block for name in (offload._spread([f"g{position}" for position in range(len(eligible))], take / len(eligible)) if take else ())
                  for block in eligible[int(name[1:])]}
        changed = False
        self.planning = True
        try:
            for block in [block for block in self.resident if block not in chosen]:
                self._drop(block)
                changed = True
            for block in sorted(chosen - set(self.resident)):  # a spare row: no CUDA graph adopts them as its static storage
                self.resident[block] = [torch.empty((shape[0] + 1, *shape[1:]), dtype=dtype, device=self.device)[:shape[0]] for shape, dtype in self.capacity]
                changed = True
        finally:
            self.planning = False
        if changed:
            self.manager.state_changed(self.model_id)
            memory_pressure.register(self, 0) if self.resident else memory_pressure.unregister(self)  # released before the weights

    def _drop(self, block):
        """The block's state leaves the VRAM: it streams its record again from now on."""
        for event in (self.resident_ready.pop(block, None), self.saving.get(block)):
            if event is not None:  # the memory is reused after the copy that fills it, and after an update's copy reading it
                torch.cuda.current_stream().wait_event(event)
        tensors = self.resident.pop(block)
        if self.dynamic:  # CUDA graphs adopted them as the static buffers of the block's arguments: they let them go
            self._forget(tensors)
        self.handles[block][:] = [] if self.views is None else self.views[block % self.count]  # a dynamic ring's slots come at its fetch

    def _group(self, group):
        return range(group * self.count, (group + 1) * self.count)

    def _forget(self, tensors):
        graphs = getattr(self.manager, "cuda_graphs", {}).get(self.model_id)
        if graphs is not None:
            graphs.forget_storage(tensors)

    def _new_resident(self, blocks, adopted=None):
        """GPU tensors of their own for the blocks' state (or adopted: the tensors a store handed over), which CUDA graphs adopt as the static
        buffers of the block's arguments, as they adopt a cache kept on the GPU (no copy at each replay, a graph per guidance branch); they
        let them go when they leave the VRAM."""
        self.planning = True  # their allocation does not release the state of this ring
        try:
            for block in blocks:
                self.resident[block] = list(adopted) if adopted is not None else [torch.empty(shape, dtype=dtype, device=self.device) for shape, dtype in self.capacity]
                if self.layout is not None:
                    self.handles.setdefault(block, [])[:] = self._resident_views(block)
        finally:
            self.planning = False
        memory_pressure.register(self, 1.5 if auto_preload.STATE_FIRST else 0)  # after the preloaded blocks when the state ranks before them
        if not self.storing:  # the cycle storing them measures them as its activations, as a cache the model would keep on the GPU
            self.manager.state_changed(self.model_id)

    def _admit(self, group):
        """Dynamic, a group stored by the pass computing its state: kept in VRAM if the peak of the pass so far, raised by the group, stays
        within the VRAM by the dynamic preload's margin (no VRAM pressure: the weights the preload keeps stay); otherwise stored in RAM."""
        if group in self.admitted:
            return True
        if group >= self.groups or group in self.refused:
            return False
        if allocator.mark_peak(self.device) + self.group_bytes + auto_preload.MARGIN > allocator.stats(self.device)["reserved"] + allocator.room(self.device):
            self.refused.add(group)
            return False
        self.admitted.add(group)  # its blocks get their tensors as they are stored
        return True

    def _write_back(self, block):
        """A block's state kept only in VRAM is copied to its record (blocking), before it leaves the VRAM."""
        self._settle(block)
        event = self.resident_ready.pop(block, None)
        if event is not None:
            torch.cuda.current_stream().wait_event(event)
        for view, tensor in zip(self._views(self.full[block][0]), self._resident_views(block)):
            view.copy_(tensor)
        self.written.add(block)

    def _demote(self, group):
        """A group of a dynamic ring leaves the VRAM, written back to RAM first: its blocks stream their records from now on."""
        blocks = [block for block in self._group(group) if block in self.resident]
        for block in blocks:
            if block in self.stored and block not in self.written:
                self._write_back(block)
            self._drop(block)
        self.promote.difference_update(blocks)
        if blocks and not self.resident:
            memory_pressure.unregister(self)
        return bool(blocks)

    def plan(self, groups):
        """Dynamic: the groups the model's dynamic preload keeps in VRAM, at the start of its cycle (on the thread running the blocks). The
        others leave now (the group of the block running at the next pass); the groups planned come back as their blocks are streamed
        (fetch), without other transfers."""
        self.wanted = set(groups)
        current = None if self.current is None else self.current // self.count
        changed = False
        for group in sorted({block // self.count for block in self.resident} - self.wanted):
            if group == current:
                self.deferred.add(group)
            else:
                changed |= self._demote(group)
        self.promote = {block for group in self.wanted for block in self._group(group) if block in self.stored and block not in self.resident}
        if changed:
            self.manager.state_changed(self.model_id)

    def _promote(self, block):
        """Dynamic, before a block planned in VRAM runs: its state fills tensors of its own from its slot (prefetched there) or its record,
        if the VRAM has room for them without pressure (otherwise it streams again at this pass)."""
        if cuda_graphs.recording() or torch.cuda.is_current_stream_capturing():
            return
        stats = allocator.stats(self.device)
        if stats["reserved"] - stats["allocated"] + allocator.room(self.device) < self.group_bytes // self.count + memory_pressure.HEADROOM:
            return
        self.promote.discard(block)
        slot = block % self.count
        prefetched = self.slots is not None and self.held[slot] == block
        self._new_resident([block])
        if prefetched:  # the copy into its slot, then from there
            self._slot_filled(slot, torch.cuda.current_stream())
            torch.cuda.current_stream().wait_event(self.ready[slot])
            for target, source in zip(self.handles[block], self.views[slot]):
                target.copy_(source)
            self.held[slot] = None
        else:
            ram, locked = self.records[block]
            stream = self._stream()
            stream.wait_stream(torch.cuda.current_stream())
            if block in self.saving:  # its record still being written by an update
                stream.wait_event(self.saving[block])
            with torch.cuda.stream(stream):
                self._fill(block, self.handles[block], self._views(ram), locked)
                self.resident_ready.setdefault(block, torch.cuda.Event()).record(stream)

    def _lock_records(self):
        """Dynamic, at the start of a pass: the records written since are locked (asynchronous transfers), in the order of _lock_groups, as
        far as the reserved RAM allows; a locked record leaves MMGP's staging ring. Then the ring's own staging buffers, if records it moves
        itself stay pageable."""
        budget = getattr(self.manager, "max_reservable_memory", 0)
        waiting = {block for block in self.written if not self.full[block][1]}
        if budget > 0 and waiting and self.owner is not None:
            for group in self._lock_groups():
                for block in [group[position] for position in auto_preload.even_order(len(group))]:
                    if block not in waiting:
                        continue
                    if budget - pinned.total_bytes - offload._pool.reserved() < self.stride or not pinned.fits(self.stride, budget):
                        break
                    ram = self.full[block][0]
                    registration = pinned.lock(self.owner, ram.data_ptr(), self.stride, "state")
                    if registration is None:
                        pinned.refused(self.stride)
                        break
                    self.registrations.append(registration)
                    self.full[block] = (ram, True)
                    self.records[block] = (self.records[block][0], True)
                    self.locked_bytes, self.pageable_bytes = self.locked_bytes + self.record, self.pageable_bytes - self.record
                    if block in self.mmgp_staged:
                        self.mmgp_staged.discard(block)
                        records = self.manager.state_records.get(self.entries[block], [])
                        records[:] = [(ring, index) for ring, index in records if not (ring is self and index == block)]
                        if not records:
                            self.manager.state_records.pop(self.entries[block], None)
        own_staging = self.model_id in self.manager.pinned_models
        if (own_staging and self.copier is None and budget > 0 and budget - pinned.total_bytes - offload._pool.reserved() >= self.staging_count * self.record
                and any(not self.full[block][1] and block not in self.mmgp_staged for block in self.written)):
            self._make_staging()

    def _pass_start(self):
        """Dynamic, before the first block of a pass: the groups the plan let go while one of their blocks ran leave the VRAM, the records
        written since are locked."""
        self.storing, changed = False, False
        for group in self.deferred:
            if self.wanted is None or group not in self.wanted:
                changed |= self._demote(group)
        self.deferred = set()
        if changed:
            self.manager.state_changed(self.model_id)
        self._lock_records()
        self._report()

    def _report(self):
        report = (len(self.resident), self.locked_bytes // 2 ** 20, self.pageable_bytes // 2 ** 20)
        if report != self.reported and self.stored:
            self.reported = report
            streamed = len(self.stored - set(self.resident))
            tqdm.write(f"[mmgp] {self.name}: {len(self.resident)} of {self.blocks} blocks in VRAM" + (
                f", {streamed} streamed from RAM ({report[1]} MB of their records locked, {report[2]} MB pageable)" if streamed else ""))

    def _trim(self):
        """Before a pass, on the thread running the blocks: whole groups leave the VRAM while the state kept there exceeds the room (the
        preload's target shrank); the room grows back only when blocks are stored again (_plan_resident), so that the preload measures."""
        room = self._room()
        if self.resident_bytes <= room:
            return
        self._stream().wait_stream(torch.cuda.current_stream())  # the work queued so far may still read it
        for start in sorted({block - block % self.count for block in self.resident}, reverse=True):
            for block in range(start, start + self.count):
                if block in self.resident:  # a block whose weights left the VRAM dropped its state alone
                    self._drop(block)
            if self.resident_bytes <= room:
                break
        if not self.resident:
            memory_pressure.unregister(self)
        self.manager.state_changed(self.model_id)

    def drop_resident(self):
        """All the state kept in VRAM leaves it (the model is unloaded: a quiet point), written back to RAM first by a dynamic ring."""
        if self.resident:
            for block in list(self.resident):
                if self.dynamic and block in self.stored and block not in self.written:
                    self._write_back(block)
                self._drop(block)
            self.promote.clear()
            for block in range(self.blocks):
                self.handles[block][:] = self.views[block % self.count] if self.views is not None else []
            memory_pressure.unregister(self)
            self.manager.state_changed(self.model_id)

    def on_pressure(self, need, parked=False):
        """The allocator found the VRAM short: state kept in VRAM leaves it first (before the weights), except that of the block running;
        only on the thread running the blocks (any thread while its offload object is parked: offload.park), not while a CUDA graph is
        captured. Returns the bytes the allocator got back (another holder of a tensor keeps it), which shrink the room of the next plans
        (offload.state_pressure)."""
        if (need <= 0 or self.planning or not self.resident or not parked and threading.get_ident() != self.thread or cuda_graphs.recording()
                or torch.cuda.is_current_stream_capturing()):
            return 0
        self._stream().wait_stream(torch.cuda.current_stream())  # the work queued so far may still read it
        before = _allocated(self.device)
        freed = 0
        if self.dynamic:  # whole groups, written back, the last of the plan's order first; the group running stays
            running = None if parked or self.current is None else self.current // self.count
            for group in reversed(auto_preload.even_order(self.groups)):
                if freed >= need:
                    break
                if group != running and self._demote(group):
                    if self.wanted is not None:
                        self.wanted.discard(group)  # not filled again before the next plan
                    freed = before - _allocated(self.device)
        for block in [] if self.dynamic else sorted(self.resident, reverse=True):
            if freed >= need:
                break
            if block != self.current:
                self._drop(block)
                freed = before - _allocated(self.device)
        if not self.resident:
            memory_pressure.unregister(self)
        self.manager.state_changed(self.model_id)
        if freed > 0:
            if not parked:  # parked: what other models take does not count against its workload
                self.manager.state_pressure(self.model_id, freed, dynamic=True) if self.dynamic else self.manager.state_pressure(self.model_id, freed)
            tqdm.write(f"[mmgp] {self.name}: VRAM short, {freed / 2 ** 20:.0f} MB of state kept in VRAM released")
        return max(0, freed)

    def _prepare(self, block, shapes):
        """Before a block is stored (or reserved): the records, the slots, the residency plan and the layout, made with the first block
        stored after a reset."""
        if self.capacity is not None and not self.prepared and not self._fits(shapes):  # more rows after a reset: made again
            self._free()
        if self.capacity is None:
            self._allocate(shapes)
        if not self.prepared:
            if self.dynamic:
                self._begin_store()
            elif self.allow_resident:
                self._plan_resident()
            self._set_layout(shapes)
            self.prepared = True
        elif [(shape, dtype) for shape, dtype, _, _ in self.layout] != shapes:
            raise ValueError(f"{self.name}: block {block} stores tensors of another layout than the first block's")

    def reserve(self, block, shapes):
        """The GPU tensors (of shapes: [(shape, dtype)], as stored) that the pass computing block's state fills before storing them
        (store(block, these tensors, current=True, rows=...)), instead of tensors of its own: the block's slot, or the tensors that keep its
        state in VRAM (a dynamic ring keeping its group there, a resident block). The same list store() returns."""
        self.thread = threading.get_ident()
        self._settle(block)
        self._prepare(block, [(tuple(shape), dtype) for shape, dtype in shapes])
        if block in self.resident or self.dynamic and self._admit(block // self.count):
            if block not in self.resident:
                self._new_resident([block])
            return self.handles[block]
        self._ensure_slots()
        slot, current = block % self.count, torch.cuda.current_stream()
        self._slot_filled(slot, current)
        for event in (self.slot_saving[slot], self.ready[slot]):  # copies reading or filling the slot
            if event is not None:
                current.wait_event(event)
        self.held[slot] = None
        self.handles[block][:] = self.views[slot]
        return self.handles[block]

    def store(self, block, tensors, current=False, adopt=False, skip=None, rows=None):
        """Keeps block's tensors (GPU or CPU tensors, or lists of row pieces; the same shapes and dtypes for every block) in RAM, a blocking
        copy: the caller may then free them. Returns the list of GPU tensors the block reads (read it at each pass: the ring rebinds its
        items when the block's state enters or leaves the VRAM), which hold these values already with current=True (the pass computing them
        reads them) or for a block whose state is kept in VRAM, and from its next fetch otherwise. After a reset, blocks may store fewer rows
        than before (views of the same memory) or more (the records and the slots are made again). adopt: a dynamic ring keeping the block in
        VRAM may keep these GPU tensors themselves (whole tensors of the stored shapes, that the caller no longer uses), without a copy.
        skip: rows (a slice of the first dimension) that the block's next passes replace (update) before reading them. tensors may be the
        list reserve() returned (filled by the pass). rows: only these rows hold state (a slice of the first dimension of whole tensors): the
        others are not stored, nor read before being replaced."""
        shapes = [_joined(tensor) for tensor in tensors]
        self.thread = threading.get_ident()  # the thread storing the blocks runs them
        self._settle(block)  # an update's copy into its record, or reading its tensors, still in flight
        self._skip(block, skip)
        own = block in self.handles and len(tensors) == len(self.handles[block]) and all(tensor is handle for tensor, handle in zip(tensors, self.handles[block]))
        self._prepare(block, shapes)
        ram, locked = self.full[block]
        # a dynamic ring keeps a block computed by this pass in VRAM if its group is kept there, without any copy to RAM
        in_vram = self.dynamic and (block in self.resident or current and self._admit(block // self.count))
        if in_vram:
            self.written.discard(block)
        else:
            for view, tensor in zip(self._views(ram), tensors):
                _copy_rows(view, tensor) if rows is None else view[rows].copy_(tensor[rows])  # blocking (device to host)
            self.written.add(block)
        if own:
            tensors = ()  # the pass filled the tensors the block reads: nothing to copy there
        if block in self.stored:  # stored again: what MMGP staged of it before is stale
            self.generation += 1
        self.stored.add(block)
        slot = block % self.count
        self.staged.pop(block, None)
        if self.dynamic:
            self.last = block
        if self.dynamic and not in_vram:
            self._ensure_slots()
            self.handles[block][:] = self.views[slot]
        if in_vram and block not in self.resident:  # the tensors handed over, or tensors of its own
            adopted = adopt and all(isinstance(tensor, torch.Tensor) and tensor.is_cuda and (tuple(tensor.shape), tensor.dtype) == (tuple(shape), dtype)
                                    and tensor.is_contiguous() and tensor.storage_offset() == 0 and tensor.untyped_storage().nbytes() == tensor.nbytes
                                    for tensor, (shape, dtype) in zip(tensors, self.capacity))
            self._new_resident([block], tensors if adopted else None)
            if adopted:
                tensors = ()
        if block in self.resident and (current or in_vram):  # filled once here, not at each pass
            event = self.resident_ready.pop(block, None)
            if event is not None:  # a copy of its former state in flight
                torch.cuda.current_stream().wait_event(event)
            for target, tensor in zip(self.handles[block], tensors):
                _copy_rows(target, tensor)
        elif block in self.resident:  # from its record on the transfer stream (asynchronous from locked RAM), which fetch waits for; the
            # record is written again only after a reset, which waits for the transfer stream
            stream = self._stream()
            stream.wait_stream(torch.cuda.current_stream())  # the work queued so far may still read its former state
            with torch.cuda.stream(stream):
                for target, source in zip(self.handles[block], self._views(ram)):
                    target.copy_(source, non_blocking=locked)
                self.resident_ready.setdefault(block, torch.cuda.Event()).record(stream)
        elif current:
            self._ensure_slots()
            self._slot_filled(slot, torch.cuda.current_stream())
            if self.slot_saving[slot] is not None:  # an update's copy still reading the slot
                torch.cuda.current_stream().wait_event(self.slot_saving[slot])
            for target, tensor in zip(self.views[slot], tensors):
                _copy_rows(target, tensor)
            self.ready[slot].record()
            self.held[slot] = block
        elif self.held[slot] == block:  # an earlier copy of this block in its slot is stale
            self.held[slot] = None
        return self.handles[block]

    def _begin_store(self):
        """Dynamic: blocks stored after a reset (the pass computing them). Without a plan of the dynamic preload, the peak of this pass is
        measured from here (the cycles other models left open end first, as when a model loads)."""
        self.wanted, self.promote, self.deferred, self.refused, self.admitted, self.storing = None, set(), set(), set(), set(), True
        preload = getattr(self.manager, "auto_preload", None)
        if preload is None or self.model_id not in preload.plans:
            auto_preload.end_open_cycles(self.manager, self.model_id)
            allocator.mark_reset(self.device)

    def _stage(self, block):
        """The copier thread copies block's pageable record into a free staging buffer (after the transfer that last read it)."""
        if block in self.staged or block not in self.stored or self.records[block][1] or self.copier is None:
            return
        used = {index for index, _ in self.staged.values()}
        free = next((index for index in range(self.staging_count) if index not in used), None)
        if free is None:
            return
        buffer, event = self.staging[free]
        source, writing = self.records[block][0], self.writebacks.get(block)

        def copy():
            if writing is not None:  # the lane's copy of updated rows into the record
                writing.result()
            if event is not None:
                event.synchronize()
            ctypes.memmove(buffer.data_ptr(), source.data_ptr(), source.numel())  # releases the GIL, no OpenMP threads

        self.staged[block] = (free, self.copier.submit(copy))

    def _copy(self, block):
        slot = block % self.count
        ram, locked = self.records[block]
        staged = None if locked else self.staged.pop(block, None)
        if staged is None and not locked and self.copier is not None:
            self._stage(block)
            staged = self.staged.pop(block, None)
        if staged is None and not locked and self._lane() is not None:  # in pieces through the lane, while the GPU computes
            self._lane_copy(block, slot)
            return
        if staged is not None:
            staged[1].result()
            ram = self.staging[staged[0]][0]
        elif block in self.writebacks:  # the record read here: the lane's copy of updated rows into it done first
            self.writebacks.pop(block).result()
        stream = self._stream()
        self._slot_filled(slot, stream)
        stream.wait_stream(torch.cuda.current_stream())  # the work queued so far is done with the slot (the block that read it)
        for event in (self.saving.get(block) if staged is None else None, self.slot_saving[slot]):
            if event is not None:  # an update still writing the record, or reading the slot
                stream.wait_event(event)
        with torch.cuda.stream(stream):
            self._fill(block, self.views[slot], self._views(ram), locked or staged is not None)
            self.ready[slot].record(stream)
        if staged is not None:
            event = torch.cuda.Event()
            event.record(stream)
            self.staging[staged[0]][1] = event
        self.held[slot] = block

    def transfer_staged(self, block, generation, source, stream):
        """MMGP's transfer of block's weights on stream (offload._transfer): its record, staged with them in MMGP's ring (source, a view of
        the ring), copied into its slot after them."""
        slot = block % self.count
        if (self.slots is None or block not in self.stored or block in self.resident or generation != self.stamp(block)
                or self.held[slot] == self.current != block):
            return  # stale, kept in VRAM, or the slot of the block about to run (fetched again when its turn comes)
        self._slot_filled(slot, stream)
        stream.wait_stream(torch.cuda.current_stream())  # the work queued so far is done with the slot
        if self.slot_saving[slot] is not None:  # an update's copy still reading the slot
            stream.wait_event(self.slot_saving[slot])
        with torch.cuda.stream(stream):
            self._fill(block, self.views[slot], self._views(source), True)
            self.ready[slot].record(stream)
        self.held[slot] = block

    def _ranges(self, block, rows=None):
        """[(tensor index, first byte, end byte)] of the record to move: rows (start, stop) of each tensor, or all its rows but those the
        block skips (left out of its fetches)."""
        start, stop = rows if rows is not None else self.skips.get(block, (0, 0))
        ranges = []
        for index, (shape, dtype, offset, nbytes) in enumerate(self.layout):
            length = shape[0]
            row = nbytes // length if length else 0
            spans = [(start, min(stop, length))] if rows is not None else [(0, length)] if start >= stop else [(0, min(start, length)), (min(stop, length), length)]
            ranges += [(index, first * row, last * row) for first, last in spans if last > first]
        return ranges

    def _lane_copy(self, block, slot):
        """block's pageable record into its slot through the lane: in pieces copied ahead by the lane's copier, transferred on the lane's
        stream once the work queued so far is done with the slot (and an update's copy read it), after the lane's copy of updated rows into
        the record; fetch waits for the job (the transfers queued), then for ready[slot]."""
        free = torch.cuda.Event()
        free.record(torch.cuda.current_stream())
        waits = [free] + ([self.slot_saving[slot]] if self.slot_saving[slot] is not None else [])
        address, pieces = self.records[block][0].data_ptr(), []
        for index, first, end in self._ranges(block):
            target, offset = _flat(self.views[slot][index]), self.layout[index][2]
            pieces += [(address + offset + start, min(end, start + LANE_PIECE) - start, target[start:min(end, start + LANE_PIECE)]) for start in range(first, end, LANE_PIECE)]
        self.pending[slot] = self.lane.upload(pieces, waits, self.ready[slot], self.writebacks.get(block))
        self.held[slot] = block

    def _slot_filled(self, slot, stream):
        """Before the slot is read or written again: the lane's transfers into it queued, and stream waiting for them."""
        future = self.pending.pop(slot, None)
        if future is not None:
            future.result()
            stream.wait_event(self.ready[slot])

    def end(self, block):
        """A minimal ring, once its block ran: its slot is released, after the copies into it and from it (the GPU reuses its memory in
        order); the next block makes it again."""
        if self.slots is None:
            return
        current = torch.cuda.current_stream()
        self._slot_filled(0, current)
        if self.slot_saving[0] is not None:  # the copy of updated rows read from the slot
            current.wait_event(self.slot_saving[0])
        self.slots, self.views, self.held, self.slot_saving = None, None, [None], [None]
        for handles in self.handles.values():
            handles.clear()

    def lead(self, block):
        """The model's block before `block` starts (a ring of blocks spaced out in their model): block's state is copied into its slot while
        that block computes, the ring's block that last read the slot having its work queued."""
        if block not in self.stored or block in self.resident or self.slots is None:
            return
        self.current = None  # the block last fetched has its work queued: MMGP's transfers may fill its slot
        if self.held[block % self.count] != block and not self._mmgp_stages(block):
            self._copy(block)

    def stamp(self, block):
        """The revision of block's record that MMGP's staging ring stages (offload._schedule_ring): transfer_staged copies a staged record
        only if it is still the current one."""
        return self.generation, self.revisions.get(block, 0)

    def fetch(self, block):
        """Before block runs: its slot holds its tensors once the current stream reaches this point; the copies of the blocks of the
        prefetch window after it are queued and the pageable record after them moved ahead. A block whose state is kept in VRAM copies
        nothing (its state leaves the VRAM when its weights did)."""
        if block not in self.stored:
            return
        self.thread = threading.get_ident()
        if self.minimal:
            self._ensure_slots()
        if self.dynamic:
            if block == min(self.stored) and self.last != block:  # a pass starts (not another guidance branch calling the same block)
                self._pass_start()
            self.last = block
            if block in self.promote:
                self._promote(block)
            if self.slots is None and any(stored not in self.resident for stored in self.stored):
                self._ensure_slots()
        else:
            if self.resident and block == min(self.stored):  # a pass starts: the state kept in VRAM within the room
                self._trim()
            if block in self.resident and not self._weights_resident(block):
                self._stream().wait_stream(torch.cuda.current_stream())
                self._drop(block)
                if not self.resident:
                    memory_pressure.unregister(self)
                self.manager.state_changed(self.model_id)
        slot, self.current = block % self.count, block
        if block not in self.resident:
            if self.held[slot] != block:
                self._copy(block)
            self._slot_filled(slot, torch.cuda.current_stream())
            torch.cuda.current_stream().wait_event(self.ready[slot])
        elif block in self.resident_ready:  # its state filled after a store
            torch.cuda.current_stream().wait_event(self.resident_ready.pop(block))
        saving = self.slot_saving[slot] if block not in self.resident else self.saving.get(block)
        if saving is not None:  # an update's copy still reading what the block may update at this pass
            torch.cuda.current_stream().wait_event(saving)
        # the next blocks streamed (those kept in VRAM skipped: whole groups of `count` blocks, so that the streamed ones keep alternating slots)
        following, ahead = block, []
        while len(ahead) < self.count and len(ahead) < self.blocks - len(self.resident):
            following = (following + 1) % self.blocks
            if following not in self.resident:
                ahead.append(following)
        for following in ahead[:self.count - 1]:
            if (following % self.count != slot and following in self.stored and self.held[following % self.count] != following
                    and not self._mmgp_stages(following)):  # MMGP transfers those with their block's weights
                self._copy(following)
        if len(ahead) == self.count and not self._mmgp_stages(ahead[-1]):
            self._stage(ahead[-1])

    def update(self, block, rows, tensors=None, skip=None):
        """Replaces rows (a slice of the first dimension of the stored tensors) of block's state, during its pass after its fetch: tensors
        (the rows, of the stored dtypes and trailing shapes) are written into the GPU tensors the pass reads (the list store() returned);
        without tensors, the caller wrote these rows there itself. Its record follows (see the module); skip: rows its next passes replace
        before reading them, left out of its fetches (None: every row is fetched). Outside its pass (its slot holds another block), tensors
        are written into its record only (blocking)."""
        if block not in self.stored:
            raise ValueError(f"{self.name}: block {block} is updated before it is stored")
        if cuda_graphs.recording() or torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"{self.name}: a state update cannot be recorded in a CUDA graph")
        self.thread = threading.get_ident()
        slot, resident = block % self.count, block in self.resident
        if not resident and self.held[slot] != block:
            if tensors is None:
                raise ValueError(f"{self.name}: block {block} is updated outside its pass without its rows")
            self._settle(block)
            self._unstage(block)
            for view, tensor in zip(self._views(self.records[block][0]), tensors):
                view[rows].copy_(tensor)  # blocking (device to host)
            self.revisions[block] = self.revisions.get(block, 0) + 1
            self._skip(block, skip)
            return
        if tensors is not None:
            for target, tensor in zip(self.handles[block], tensors):
                target[rows].copy_(tensor)
        if resident and self.dynamic:  # kept in VRAM: its record is stale, written back whole if its group leaves the VRAM
            self.written.discard(block)
            self.revisions[block] = self.revisions.get(block, 0) + 1
        else:
            self._save(block, rows, None if resident else slot)
        self._skip(block, skip)

    def _save(self, block, rows, slot):
        """The updated rows of block (in its slot, or in its tensors kept in VRAM) copied into its record: asynchronously on the ring's save
        stream into locked RAM (its event guards the record and the source), through the lane into pageable RAM (the event guards the
        source, the future the record), blocking into pageable RAM without the lane."""
        ram, locked = self.records[block]
        self._unstage(block)
        lane = None if locked else self._lane()
        if locked or lane is not None:
            if self.save_stream is None:
                self.save_stream = torch.cuda.Stream(self.device)
            self.save_stream.wait_stream(torch.cuda.current_stream())  # the rows written by the pass
        if locked:
            with torch.cuda.stream(self.save_stream):
                for target, source in zip([view[rows] for view in self._views(ram)], [tensor[rows] for tensor in self.handles[block]]):
                    target.copy_(source, non_blocking=True)  # into locked RAM, which every reader reaches through this event
            event = torch.cuda.Event()
            event.record(self.save_stream)
        elif lane is not None:
            pieces = []
            for index, first, end in self._ranges(block, (rows.start or 0, rows.stop)):
                source, address = _flat(self.handles[block][index]), ram.data_ptr() + self.layout[index][2]
                pieces += [(source[start:min(end, start + LANE_PIECE)], min(end, start + LANE_PIECE) - start, address + start) for start in range(first, end, LANE_PIECE)]
            event, future = lane.download(self.save_stream, pieces)
            if future is not None:
                self.writebacks[block] = future
        else:
            for target, source in zip([view[rows] for view in self._views(ram)], [tensor[rows] for tensor in self.handles[block]]):
                target.copy_(source)  # blocking (device to host)
            event = None
        if event is not None:
            self.saving[block] = event
            if slot is not None:
                self.slot_saving[slot] = event
        self.revisions[block] = self.revisions.get(block, 0) + 1

    def _skip(self, block, skip):
        if skip is None:
            self.skips.pop(block, None)
        else:
            self.skips[block] = (skip.start or 0, skip.stop)

    def _fill(self, block, targets, sources, non_blocking):
        """Copies block's record (sources) into GPU tensors (targets), but for the rows its pass replaces (skip)."""
        start, stop = self.skips.get(block, (0, 0))
        for target, source in zip(targets, sources):
            if start >= stop:
                target.copy_(source, non_blocking=non_blocking)
                continue
            if start > 0:
                target[:start].copy_(source[:start], non_blocking=non_blocking)
            if stop < target.shape[0]:
                target[stop:].copy_(source[stop:], non_blocking=non_blocking)

    def _settle(self, block):
        """Waits for the copy of block's updated rows into its record (the host writes the record next)."""
        event, future = self.saving.pop(block, None), self.writebacks.pop(block, None)
        if event is not None:
            event.synchronize()
        if future is not None:
            future.result()

    def _unstage(self, block):
        """Before its pageable record changes: the copies reading it finish (the ring's copier, MMGP's copier staging it with the block's
        weights), and what the ring's copier staged is dropped (MMGP's staged copy is refused at its transfer: its revision is older)."""
        staged = self.staged.pop(block, None)
        if staged is not None:
            staged[1].result()
        mmgp_ring = getattr(self.manager, "ring", None)
        if mmgp_ring is not None and block in self.mmgp_staged:
            future = mmgp_ring.jobs.get(self.entries[block])
            if future is not None:
                future.result()

    def nbytes(self):
        return self.record * self.blocks

    def _quiet(self):
        mmgp_ring = getattr(self.manager, "ring", None) if self.mmgp_staged else None
        if mmgp_ring is not None:  # the copies of MMGP's copier reading the records
            entries = {self.entries[block] for block in self.mmgp_staged}
            for name, future in list(mmgp_ring.jobs.items()):
                if name in entries:
                    future.result()
        if self.copier is not None:
            for _, future in self.staged.values():
                future.result()
        for future in [*self.pending.values(), *self.writebacks.values()]:  # the lane's jobs for this ring
            future.result()
        if self.slots is not None:
            self._stream().synchronize()
        if self.lane is not None:
            self.lane.stream.synchronize()
        if self.save_stream is not None:  # the updates' copies into the records
            self.save_stream.synchronize()
        self.staged, self.saving, self.slot_saving, self.pending, self.writebacks = {}, {}, [None] * self.count, {}, {}

    def reset(self):
        """Forgets the stored blocks, keeping the RAM and the slots (and the state kept in VRAM) for blocks stored again."""
        self._quiet()
        self.held, self.stored, self.prepared = [None] * self.count, set(), False
        self.generation += 1
        self.written, self.promote, self.deferred, self.skips = set(), set(), set(), {}  # the records are stale

    def _free(self):
        """Frees the slots, the state kept in VRAM and the RAM (waits for the GPU); the ring stays attached to the blocks."""
        self._quiet()
        if self.copier is not None:
            self.copier.shutdown()
        if self.lane is not None:
            _release_lane(self.lane)
        self.lane, self.lane_refused, self.prepared = None, False, False
        for module in self.leads:  # made again with the slots
            _unhook(module, self)
        self.leads = []
        self.saving, self.slot_saving, self.skips = {}, [None] * self.count, {}
        self.stored, self.written, self.promote, self.deferred, self.refused, self.wanted, self.reported = set(), set(), set(), set(), set(), None, None
        if self.resident:  # nothing stored: nothing written back
            self.drop_resident()
        if self.dynamic and self.slots is not None:  # the graphs that adopted the slots let them go
            self._forget([tensor for slot in self.slots for tensor in slot])
        self.slots, self.views, self.held, self.ready, self.stored, self.full, self.records, self.memory, self.staging, self.copier = \
            None, None, [None] * self.count, [None] * self.count, set(), {}, {}, [], [], None
        for block in self.mmgp_staged:
            records = self.manager.state_records.get(self.entries[block], [])
            records[:] = [(ring, index) for ring, index in records if ring is not self]
            if not records:
                self.manager.state_records.pop(self.entries[block], None)
        self.mmgp_staged, self.generation = set(), self.generation + 1
        self.capacity, self.layout, self.locked_bytes, self.pageable_bytes, self.owner, self.stride = None, None, 0, 0, None, 0
        for registration in self.registrations:
            registration.close()
        self.registrations = []
        for handle in self.handles.values():
            handle.clear()

    def release(self):
        """Frees the slots and the RAM (waits for the GPU) and detaches the ring from the blocks."""
        self._quiet()
        self._free()
        if self.allow_resident:
            self.manager.state_rings.discard(self)
        for module in self.modules:
            _unhook(module, self)
        self.modules = []
