# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""CPU-backed expert banks with independently reclaimable GPU expert copies.

An ExpertBank registers checkpoint tensors whose first dimension is the expert
index. MMGP leaves these tensors on pageable CPU memory, outside normal block
swapping/pinning. Its residency controller owns every optional GPU copy. Compute
backends use resident_tensors() and lease(); they never transfer model weights.
"""
import bisect
import contextlib
import itertools
import threading
import weakref

import torch
from torch import nn
from tqdm import tqdm

from . import allocator, cuda_graphs, memory_pressure, pinned, runtime
from .read_ahead import cold_ranges, prefetch_ranges

MARGIN = 512 << 20
UPLOAD_STAGING = 32 << 20  # bytes of each of the two pinned buffers through which new GPU copies go


class ExpertBank(nn.Module):
    def __init__(self, num_experts, tensor_names):
        super().__init__()
        self.num_experts = int(num_experts)
        self.expert_tensor_names = tuple(tensor_names)
        self._residency = None
        self._resident = {}
        self.residency_version = 0
        self._leases = 0
        # 1 for the experts with GPU copies, read by the CPU backend; a plain tensor, outside the module's state
        self.resident_flags = torch.zeros(self.num_experts, dtype=torch.uint8, device="cpu")
        self._flags = self.resident_flags.numpy()
        # Optional placement priority per expert, higher first and comparable across banks (e.g. measured pick frequencies); set by
        # the model before MMGP takes it over. None keeps the default order.
        self.expert_priority = None

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

    def _evict(self, expert):
        del self._resident[expert]
        self._flags[expert] = 0
        self.residency_version += 1


def banks(model):
    return {name: module for name, module in model.named_modules() if isinstance(module, ExpertBank)}


def excluded_names(model):
    prefixes = tuple(banks(model))
    return {name for name, _ in model.named_modules() if any(name == prefix or name.startswith(prefix + ".") for prefix in prefixes)}


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
        memory_pressure.register(self, priority=0)

    def close(self):
        self.unload()
        self.pinned, self._pinned_holder = {}, None
        for buffer in self.upload_staging:
            buffer.release()
        memory_pressure.unregister(self)
        for bank in self.banks:
            bank._residency = None

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
        if self.dynamic:
            manager = self.manager()
            cotenants = manager.cotenants_map.get(self.model_id, ())
            model_ids = set(manager.models if cotenants == "*" else (self.model_id, *cotenants))
            ordinary = sum(size for name, size in manager.blocks_of_modules_sizes.items() if name.split("/", 1)[0] in model_ids)
            target = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory - ordinary - MARGIN
        else:
            target = max(0, self.budget - self._nonexpert_bytes()) if self.budget else 0
        return self.ranking[:bisect.bisect_right(self.cumulative, target)]

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
        if self.changing or cuda_graphs.recording() or torch.cuda.is_current_stream_capturing():
            return  # never inside a graph capture (a decoding pass of a model called as a module): its next eager call places them
        if not self._nonexperts_resident():
            self.unload(reset=False)
            return
        if self.dynamic and not self.started and not grow_only:
            # First call measures/allocates KV, recurrence and other working storage.
            self.started = True
            return
        target = max(0, self.budget - self._nonexpert_bytes()) if self.budget else 0
        if self.dynamic:
            if allocator.active == "vmm":
                state = allocator.stats()
                free = allocator.room() + state["reserved"] - state["allocated"]
            else:
                free = torch.cuda.mem_get_info()[0]
            target = max(0, self.nbytes + free - MARGIN - int(0.05 * torch.cuda.memory_allocated()))
        if self.pressure_ceiling is not None:
            target = min(target, self.pressure_ceiling)
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
            self.loaded = dict.fromkeys(wanted)
            self.applied = count
            if self.manager().verboseLevel >= 1:
                tqdm.write(f"Expert Residency - {self.model_id}: {len(self.loaded)}/{len(self.ranking)} Experts on GPU ({self.nbytes / (1 << 20):.0f} MiB); CPU Experts Are Not Pinned")
        finally:
            self.changing = False
            self._placement_changed()

    def _upload(self, additions):
        """GPU copies of new experts, from their pinned copies (pin()) or else through two pinned buffers: copying the next experts from
        their pageable (mmap) sources overlaps the transfer of the previous ones. These copies are only made when the placement changes.
        Sources that a drive must read are read ahead in large requests, the next expert while the current one is copied (the parallel CPU
        copy would fault them in 64 KiB reads)."""
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

        if additions:
            prefetch_ranges(cold_ranges(sources(0)))
        try:
            for index, (bank, expert) in enumerate(tqdm(additions, desc=f"Preloading Experts - {self.model_id}", unit="expert", mininterval=1 / 3, disable=self.manager().verboseLevel < 1)):
                runtime.checkpoint()
                if index + 1 < len(additions):
                    prefetch_ranges(cold_ranges(sources(index + 1)))
                if offset + self.sizes[bank] + 64 * len(bank.expert_tensor_names) > size:
                    transfers[slot].record()
                    slot, offset = 1 - slot, 0
                    transfers[slot].synchronize()  # the transfers that read this buffer last have completed
                tensors = {}
                kept = self.pinned.get((bank, expert), {})
                for name in bank.expert_tensor_names:
                    if name in kept:
                        tensors[name] = torch.empty_like(kept[name], device="cuda").copy_(kept[name], non_blocking=True)
                        continue
                    source = getattr(bank, name)[expert]
                    nbytes = source.numel() * source.element_size()
                    staged = staging[slot][offset:offset + nbytes].view(source.dtype).view(source.shape)
                    staged.copy_(source)
                    tensors[name] = torch.empty_like(source, device="cuda").copy_(staged, non_blocking=True)
                    offset += -(-nbytes // 64) * 64
                bank._place(expert, tensors)
                self.loaded[(bank, expert)] = None
                self.nbytes += self.sizes[bank]
        finally:
            transfers[slot].record()
            transfers[slot].synchronize()  # also orders the other buffer's transfers, queued before on the same stream

    def _drop(self, bank, expert):
        bank._evict(expert)
        del self.loaded[(bank, expert)]
        self.nbytes -= self.sizes[bank]
        self.applied = None

    def _placement_changed(self):
        auto_preload = self.manager().auto_preload
        if auto_preload is not None:
            auto_preload.on_expert_change()

    def on_pressure(self, need, parked=False):
        # parked: its offload object is parked (offload.park), any thread may release its copies
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
        if self.loaded:
            self._clear_graphs()
            torch.cuda.synchronize()
            for item in list(reversed(self.loaded)):
                self._drop(*item)
            self._placement_changed()
        if reset:
            self.started = False
            self.pressure_ceiling = None
            for bank in self.banks:
                bank.release_execution_cache()
