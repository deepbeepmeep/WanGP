# Copyright (c) 2024-2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Automatic VRAM preload of the models processed block by block (offload.all / offload.profile(..., autoPreload=True or a dict)):
each such model keeps in VRAM as many of its blocks as the VRAM left by its current workload allows, instead of a fixed preload. Needs the
MMGP Optimized VRAM Allocator (mmgp.allocator)."""
import dataclasses
import threading
import weakref

import torch
from tqdm import tqdm

from . import allocator, cuda_graphs, memory_pressure

ONE_MB = 1 << 20
MARGIN = 512 * ONE_MB  # left free beyond the measured peak: the allocator's headroom (256 MB) and its fragmentation
MARGIN_SHARE = 0.05  # and this share of the peak, for variations between the cycles of a workload
GROW_MIN = 512 * ONE_MB  # a plan grows by at least this much (or 2 blocks), not to change for small variations of the free VRAM
PENALTY_PATIENCE = 8  # cycles without VRAM shortage or allocator recovery after which the VRAM a workload leaves free for them halves
PENALTY_PATIENCE_MAX = 1024  # a shortage after such a decrease doubles the cycles waited before the next one, up to this
PENALTY_MIN = 64 * ONE_MB  # a smaller penalty is dropped when it halves
_process_measures = {}  # with autoPreload=True: (model class, model id, structure, workload) -> peak of the activations measured, and
# (..., workload, "penalty") -> its penalty (_penalty), (..., workload, "unmeasured") -> True while its measure is the 0 set by a shortage
END_OTHER_CYCLES = True  # comparison switch: another model's load or cycle ends the cycles left open (end_open_cycles)
FIRST_MEASURE_CLEARS_PENALTY = True  # comparison switch: a workload's first measure drops the penalty of the shortages met before it (_end_cycle)
HELD_APART = True  # comparison switch: a measure leaves out the VRAM held beside the model, the plans add the VRAM held now (_planned)
STATE_FIRST = True  # comparison switch: the groups of the dynamic state rings of a model (a cache of its blocks) rank before the blocks of its
# towers (False: after them). Every byte either streams once per pass, but a block streams in the shadow of its compute: its state kept in
# VRAM leaves the transfer of its weights alone, rather than adding to it, and more blocks hide it
_planners = weakref.WeakSet()  # the AutoPreload objects of the process


def end_open_cycles(manager=None, model_id=None):
    """A model is loaded or starts a cycle: the cycles the other models left open end here (all offload objects). A cycle's measure is
    the process' allocation peak since it started, and a measure only grows: left open, the VRAM of the next model's work (a
    generation beside a language model kept loaded, a vision tower) would count as their activations for good, and their next plans
    would keep fewer blocks."""
    if not END_OTHER_CYCLES:
        return
    for planner in list(_planners):
        for plan in list(planner.plans.values()):
            if plan.workload is not None and not (planner.manager is manager and plan.model_id == model_id):
                planner._end_cycle(plan)


def _van_der_corput(n):
    x, f = 0.0, 0.5
    while n:
        if n & 1:
            x += f
        n >>= 1
        f /= 2
    return x


def even_order(count):
    """0..count-1 in an order whose every prefix is spread evenly over the range (van der Corput positions, nearest free index)."""
    order, free, n = [], list(range(count)), 0
    while free:
        target = _van_der_corput(n) * count
        best = min(free, key=lambda i: (abs(i - target), i))
        free.remove(best)
        order.append(best)
        n += 1
    return order


def _interleave(towers):
    """The blocks of the towers ([(name, size), ...] in the order they run), each tower in its even order, the towers sharing the ranks in
    proportion to their sizes."""
    orders = [[floors[i] for i in even_order(len(floors))] for floors in towers]
    totals = [sum(size for _, size in floors) or 1 for floors in towers]
    taken, positions, ranking = [0] * len(towers), [0] * len(towers), []
    while True:
        candidates = [t for t in range(len(towers)) if positions[t] < len(orders[t])]
        if not candidates:
            return ranking
        t = min(candidates, key=lambda t: ((taken[t] + orders[t][positions[t]][1]) / totals[t], t))
        name, size = orders[t][positions[t]]
        positions[t] += 1
        taken[t] += size
        ranking.append(name)


def ranked_blocks(towers, slow=()):
    """towers: [[(block name, size), ...] in the order they run, ...]. The block names in the order they are kept in VRAM: the slow ones first
    (not pinned), then the others, each group spread evenly across the towers in proportion to their sizes."""
    return [name for in_slow in (True, False) for name in _interleave([[(name, size) for name, size in floors if (name in slow) == in_slow] for floors in towers])]


def workload(args, kwargs):
    """Number of elements of the largest input tensor of a block (one level of lists, tuples, dicts and dataclasses): the workload of a
    cycle."""
    largest = 0
    for value in (*args, *kwargs.values()):
        if isinstance(value, (list, tuple)):
            items = value
        elif isinstance(value, dict):
            items = value.values()
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):  # e.g. the arguments of each modality of LTX-2's blocks
            items = [getattr(value, field.name) for field in dataclasses.fields(value)]
        else:
            items = (value,)
        for item in items:
            if torch.is_tensor(item):
                largest = max(largest, item.numel())
    return largest


class _Plan:
    """The blocks of a model and what it keeps in VRAM."""
    def __init__(self, model_id, model_class, towers, base, others=()):
        self.model_id, self.towers, self.base, self.others = model_id, towers, base, list(others)
        self.sizes = {name: size for _, floors in towers for name, size in floors}
        self.largest = max(self.sizes.values())  # the slots of the streamed blocks are sized for the towers (larger blocks get buffers of their own)
        self.sizes.update(self.others)
        self.ranking = self.ranked()
        self.structure = (model_class, model_id, len(self.sizes), sum(self.sizes.values()))  # another quantization or model is measured again
        self.first = towers[0][1][0][0]  # entering the first block of the first tower starts a cycle
        self.final = towers[-1][1][-1][0]  # once the last block of the last tower ran, the pass is over (done) until the next cycle starts
        self.done = False
        self.floor = 0  # number of ranked blocks preloaded for an unknown workload (the model's budget)
        self.pending = set()  # blocks of the plan that become preloaded when they have been transferred
        self.reported = None  # the plan last printed: blocks that never run (e.g. an encoder unused by this workload) stay pending
        self.previous = set()  # the blocks kept in VRAM when the model was last unloaded: they come back with it as far as they fit
        self.last_work = None  # the workload of its last cycle
        self.last = None  # block that ran last
        self.workload = None  # of the cycle running, None between cycles
        self.cycles = (0, 0)  # use stamps (memory_pressure.stamp) at the start of its last two cycles, the previous one first
        self.held = 0  # VRAM allocated beside its weights and the reclaimable VRAM when its cycle started: other models, caches, the pipeline's tensors
        self.min_weights = 0  # least VRAM held by the model's weights during the cycle
        self.grew = False  # blocks were kept during the cycle: its peak is not a measure of the activations
        self.short = False  # the VRAM ran short during the cycle: the pressure callback released blocks of the plan
        self.relink = False  # blocks were released by the pressure callback: to chain again with the streamed blocks
        self.slow = set()  # blocks not pinned: staged through RAM at each transfer, the first to keep in VRAM
        self.recoveries = None  # the allocator's recoveries when a cycle with blocks kept in VRAM started (None: no block kept)
        # the cycle running measures a new workload with the budget's preload: the next one may keep other blocks. True from its load until its
        # first cycle: a graph recorded before it (a warmup) would run with the plan of its last workload, without its hooks starting the cycle
        self.measuring = True
        self.workload_of = workload  # or the model's _auto_preload_workload(args, kwargs): its own grouping of the inputs into workloads
        self.cap = None  # or the model's _auto_preload_max: the most VRAM its weights may hold (base, blocks kept, slots of the streamed ones)
        self.margin = MARGIN  # or the model's _auto_preload_margin: left free beyond its measured peaks (e.g. passes replayed by CUDA graphs vary less)
        self.state_room = 0  # VRAM left to the state its rings keep in VRAM (mmgp.state_ring) once every block is kept, nothing otherwise
        self.state_plan = {}  # its dynamic state rings -> the groups of blocks they keep in VRAM, planned with its blocks (_fit)
        self.state_short = False  # the state its rings kept in VRAM was released under pressure during the cycle

    def ranked(self, slow=()):
        """The order the blocks are kept in VRAM: those outside the towers first (embeddings, heads, short stacks: each runs at every pass as a
        block of the towers does, and keeping it frees no slot), then those of the towers (ranked_blocks)."""
        return [name for name, _ in self.others] + ranked_blocks([floors for _, floors in self.towers], slow)


class AutoPreload:
    def __init__(self, manager, model_ids, verbose_level=1, measures=None, fixed_ids=()):
        """manager: the offload object, once its models are hooked and their budgets planned (tune_preloading). The preload of each model
        becomes the same number of ranked blocks, kept for the workloads not measured yet. measures: the dictionary that keeps the measures
        (None: kept for the process). fixed_ids: models whose budget's preload stays as planned, except under VRAM pressure: their preloaded
        blocks then leave the VRAM rather than the allocation spilling into RAM, until the model is unloaded."""
        self.manager, self.verbose = manager, verbose_level
        self.measures = _process_measures if measures is None else measures
        self.plans = {}
        self.fixed = {}  # the plans of the fixed preloads (manual budgets): plan.manual is their preload, restored at each load
        self.thread = None  # the thread running the blocks, the only one that may change the plans
        self.dropping = False
        self.window = manager.prefetch_window if manager.prefetch_window is not None else 2  # fixed: the slots do not change with the plans
        for model_id in model_ids:
            plan = self._new_plan(model_id)
            if plan is not None:
                self.plans[model_id] = plan
                plan.floor = len(manager.preloaded_blocks_per_model.get(model_id, {}))
                self._set(plan, plan.ranking[:plan.floor])
        for model_id in fixed_ids:
            plan = self._new_plan(model_id)
            if plan is not None and manager.preloaded_blocks_per_model.get(model_id):
                plan.manual = dict(manager.preloaded_blocks_per_model[model_id])
                self.fixed[model_id] = plan
        _planners.add(self)
        if self.plans or self.fixed:
            memory_pressure.register(self)
            if self.plans and self.verbose >= 1:
                print(f"Automatic VRAM preload of {', '.join(repr(model_id) for model_id in self.plans)}: the blocks kept in VRAM follow the VRAM left by each workload")

    def close(self):
        memory_pressure.unregister(self)

    def after_pinning(self):
        """The RAM is pinned: the blocks that are not pinned are the first to keep in VRAM (those of the models of the pinned pool are pinned
        as they stream). Nothing is in VRAM yet."""
        m = self.manager
        for plan in self.plans.values():
            plan.slow = set() if plan.model_id in m.pool_models else {name for name in plan.sizes if m._block_items(plan.model_id + "/" + name)[2] is None}
            plan.ranking = plan.ranked(plan.slow)
            self._set(plan, plan.ranking[:plan.floor])

    def _new_plan(self, model_id):
        m = self.manager
        towers = []
        for tower_name in m.towers_per_model.get(model_id) or ():
            prefix = model_id + "/" + tower_name
            floors = sorted((int(name[len(prefix):]), name[len(model_id) + 1:], size) for name, size in m.blocks_of_modules_sizes.items() if name.startswith(prefix) and name[len(prefix):].isdigit())
            if floors:
                towers.append((tower_name, [(name, size) for _, name, size in floors]))
        if not towers:
            return None
        # the other blocks (a Sequential, a list of fewer layers than a tower), but those the model loads only while they run
        excluded = {name for _, floors in towers for name, _ in floors} | set(getattr(m.models[model_id], "_offload_separate_blocks", ()))
        others = [(name[len(model_id) + 1:], size) for name, size in m.blocks_of_modules_sizes.items() if name.startswith(model_id + "/") and size and name[len(model_id) + 1:] not in excluded]
        plan = _Plan(model_id, type(m.models[model_id]).__name__, towers, m.blocks_of_modules_sizes.get(model_id, 0), others)
        plan.workload_of = getattr(m.models[model_id], "_auto_preload_workload", None) or workload
        plan.cap = getattr(m.models[model_id], "_auto_preload_max", None)
        plan.margin = getattr(m.models[model_id], "_auto_preload_margin", MARGIN)
        return plan

    def _penalty(self, key):
        """[bytes, clean cycles, patience, probing] of a workload, kept with its measures: the VRAM its plans leave free beyond the measured
        peak after VRAM shortages (the blocks they released) and allocator recoveries (a block each), which halves after `patience` cycles without
        either; a shortage after such a decrease (probing) doubles the patience. The measured peak itself only grows, from real peaks."""
        return self.measures.setdefault((*key, "penalty"), [0, 0, PENALTY_PATIENCE, False])

    def _state_penalty(self, key):
        """[bytes, clean cycles] of a workload, kept with its measures once the state of the model's rings was released under pressure: the
        VRAM that state leaves free since, which halves after PENALTY_PATIENCE cycles without such a release; None before any."""
        return self.measures.get((*key, "state_penalty"))

    def _planned(self, key, held=0):
        # the VRAM a workload needs beside the weights (None until measured): its measured activations, the VRAM held beside the model now
        # (other models' bases and caches, e.g. a parked language model's, the pipeline's tensors: a model loaded or gone since the measure
        # changes it, which its plans then follow rather than finding it through VRAM shortages) and its penalty
        peak = self.measures.get(key)
        return None if peak is None else peak + (held if HELD_APART else 0) + self.measures.get((*key, "penalty"), (0,))[0]

    def _weights(self, plan, preloaded):
        """VRAM of the model's weights with these blocks preloaded: its base, the blocks and the slots of the blocks streamed, and the state
        its rings keep in VRAM (mmgp.state_ring), which the measures of its activations leave out as they leave out its weights."""
        streamed = len(preloaded) < len(plan.sizes)
        experts = sum(residency.nbytes for residency in self.manager.expert_residency.values())
        return (plan.base + sum(plan.sizes[name] for name in preloaded) + ((self.window + 1) * plan.largest if streamed else 0) + experts
                + self.manager.state_bytes(plan.model_id))

    def _state_units(self, plan):
        """The dynamic state rings of the model that hold state (mmgp.state_ring, resident="dynamic"), and their groups of blocks in the
        order they are kept in VRAM: each ring's groups spread evenly, the rings (the branches of a guidance) sharing the ranks in proportion
        to their sizes."""
        rings = sorted((ring for ring in self.manager.state_rings if ring.dynamic and ring.model_id == plan.model_id and ring.stored), key=lambda ring: ring.serial)
        return rings, _interleave([[((ring, group), ring.group_bytes) for group in range(ring.groups)] for ring in rings])

    def _fit(self, plan, peak, work):
        """The longest prefix of the ranking that fits beside a peak of activations, with the base and the slots of the streamed blocks. The
        groups of the model's dynamic state rings are planned with the blocks (plan.state_plan), ranked before the blocks of its towers
        (STATE_FIRST) or after them, with the slots of a ring while any of its groups streams. The state of its other rings gets what remains
        once everything fits (plan.state_room), less its own penalty for the workload (_state_penalty): a shortage of that state shrinks only
        its room."""
        stats = allocator.stats()
        target = stats["reserved"] + allocator.room() - peak - plan.margin - int(MARGIN_SHARE * peak)
        if plan.cap is not None:
            target = min(target, plan.cap)
        whole = plan.base + sum(plan.sizes.values())
        rings, units = self._state_units(plan)
        state = sum(ring.group_bytes for ring, _ in units)
        plan.state_room = max(0, target - whole - state - (self._state_penalty((*plan.structure, work)) or (0,))[0])
        plan.state_plan = {ring: set() for ring in rings}
        if whole + state <= target:
            for ring, group in units:
                plan.state_plan[ring].add(group)
            return list(plan.ranking)
        shuttle, used, count, blocks_left = (self.window + 1) * plan.largest, plan.base, 0, len(plan.ranking)
        left = {ring: ring.groups for ring in rings}
        blocks = [(None, name) for name in plan.ranking]
        for owner, name in units + blocks if STATE_FIRST else blocks + units:
            size = plan.sizes[name] if owner is None else owner.group_bytes
            slots = sum(ring.slot_bytes for ring in rings if left[ring] - (ring is owner) > 0 or ring.slots is not None)
            if used + size + (shuttle if blocks_left - (owner is None) else 0) + slots > target:
                break
            used += size
            if owner is None:
                count, blocks_left = count + 1, blocks_left - 1
            else:
                left[owner] -= 1
                plan.state_plan[owner].add(name)
        return plan.ranking[:count]

    def settled(self, model_id):
        """False while the model runs a cycle measuring a new workload: the blocks it keeps in VRAM (where its weights are) may change at the
        next cycle."""
        plan = self.plans.get(model_id)
        return plan is None or not plan.measuring

    def own_buffer(self, model_id, block_name):
        """A block transferred into a buffer of its own rather than a shared slot: a block to keep in VRAM once transferred, or any block while
        the plan keeps all of its own (the slots, sized for the blocks of the towers, would only hold the blocks loaded while they run)."""
        plan = self.plans.get(model_id)
        return plan is not None and (block_name in plan.pending or len(self.manager.preloaded_blocks_per_model[model_id]) + len(plan.pending) == len(plan.sizes))

    def keeps_all(self, model_id, block_name):
        """block_name is transferred to stay in VRAM, as every block of the model: an idle slot of the streamed blocks may become its buffer
        (the slots serve no streamed block any more)."""
        plan = self.plans.get(model_id)
        return plan is not None and block_name in plan.pending and len(self.manager.preloaded_blocks_per_model[model_id]) + len(plan.pending) == len(plan.sizes)

    def keep(self, model_id, block_name):
        """Called when the block that ran last is about to be unloaded: True keeps it in VRAM, preloaded from now on (it was transferred
        into a buffer of its own; one that got a shared slot before it was planned keeps that slot when every block stays in VRAM, otherwise
        it is kept when it is transferred again)."""
        plan = self.plans.get(model_id)
        if plan is None or block_name not in plan.pending:
            return False
        m, entry = self.manager, model_id + "/" + block_name
        if any(slot[1] == entry for slot in m.block_slots.get(model_id, ())) and (not self.keeps_all(model_id, block_name) or m.adopt_slot(model_id, plan.sizes[block_name], entry) is None):
            return False
        plan.pending.discard(block_name)
        m.preloaded_blocks_per_model[model_id][block_name] = plan.sizes[block_name]
        self._link(plan)
        self._complete_loras(model_id, entry)
        plan.grew = True
        return True

    def on_block(self, model_id, block_name, args, kwargs):
        """Called before each module of a block of the model runs: entering the first block starts a cycle. Not while a graph stage
        is prepared or captured: the weights stay where the stage found them (a stage may span several blocks)."""
        plan = self.plans.get(model_id)
        if plan is None:
            fixed = self.fixed.get(model_id)
            if fixed is not None and not cuda_graphs.recording():
                self.thread, fixed.last = threading.get_ident(), block_name
                if fixed.relink:  # blocks released under pressure: chained again with the streamed ones, outside the allocation that released them
                    self._drop(fixed, [])
            return
        if cuda_graphs.recording():
            return
        last, plan.last, plan.done = plan.last, block_name, False
        if block_name != plan.first or last == block_name:
            return
        self.thread = threading.get_ident()
        end_open_cycles(self.manager, model_id)
        self._end_cycle(plan)
        plan.cycles = (plan.cycles[1], memory_pressure.stamp())
        loaded = self.manager.loaded_blocks[model_id]
        if loaded is not None and self.keep(model_id, loaded):  # the last block streamed to grow a plan: no streamed block retires it any more
            self.manager.loaded_blocks[model_id] = None
        work = plan.workload_of(args, kwargs)
        plan.held = self._held(plan)
        peak = self._planned((*plan.structure, work), plan.held)
        wanted = plan.ranking[:plan.floor] if peak is None else self._fit(plan, peak, work)
        current = self.manager.preloaded_blocks_per_model[model_id]
        experts = self.manager.expert_residency.get(model_id)
        if len(current) < len(plan.sizes) or len(wanted) < len(plan.sizes):
            for residency in self.manager.expert_residency.values():
                residency.unload(reset=False)
        missing = [name for name in wanted if name not in current]
        dropped = [name for name in current if name not in wanted]
        if dropped or plan.relink:
            self._drop(plan, dropped)
        if peak is not None:  # the groups of the dynamic state rings planned with the blocks: those that leave go now, before blocks are kept
            for ring, groups in plan.state_plan.items():
                ring.plan(groups)
            plan.state_plan = {}
        # small growths are skipped (variations of the free VRAM), not that of blocks not pinned (left out when the pressure callback stopped a
        # growth): each one streamed costs a copy through RAM at every step
        if plan.pending or len(wanted) == len(plan.sizes) or sum(plan.sizes[name] for name in missing) >= max(GROW_MIN, 2 * plan.largest) or not plan.slow.isdisjoint(missing):
            plan.pending = set(missing)
        report = (len(current) + len(plan.pending), None if peak is None else round(peak / ONE_MB))
        if self.verbose >= 1 and (dropped or plan.pending) and (dropped or report != plan.reported):
            plan.reported = report
            penalty = self.measures.get((*plan.structure, work, "penalty"), (0,))[0]
            measured = "the budget's preload until measured" if peak is None else f"{(peak - penalty) / ONE_MB:0.0f} MB of activations measured" + \
                       (f", {penalty / ONE_MB:0.0f} MB more left free after VRAM shortages" if penalty else "")
            tqdm.write(f"Automatic VRAM preload of '{model_id}': {len(current) + len(plan.pending)} of its {len(plan.sizes)} blocks to keep in VRAM ({measured})")
        if len(current) + len(plan.pending) == len(plan.sizes):
            self._free_slots(plan)
        if experts is not None:
            experts.prepare()
        plan.workload, plan.last_work, plan.grew, plan.measuring = work, work, False, peak is None and plan.floor < len(plan.sizes)
        plan.min_weights = self._weights(plan, current)
        plan.recoveries = allocator.stats()["recoveries"] if current or plan.pending else None
        allocator.mark_reset()

    def after_block(self, model_id, block_name):
        """Called after each block of the model ran: after the last block of its last tower, its pass is over until its next cycle starts."""
        plan = self.plans.get(model_id)
        if plan is not None and not cuda_graphs.recording():
            plan.done = block_name == plan.final

    def _between_passes(self, plan):
        """A VRAM shortage between the model's passes comes from other work (the pipeline's tensors of a next generation while the model
        stays loaded): it ends the cycle left open, unmeasured, and costs the workload no penalty."""
        if plan.done and plan.workload is not None:
            plan.grew = True
            self._end_cycle(plan)
        return plan.done

    def on_load(self, model_id):
        """The model comes back to VRAM: the blocks it kept when it left come back with it, at once as its base does, as far as they fit
        beside the activations measured for its last workload (its next cycle adjusts them to its workload). Otherwise they would stream at
        its first step and be kept one by one as they run, which also leaves its CUDA graph stages waiting for them."""
        self.thread = threading.get_ident()  # the thread loading the model runs its blocks (another one may have run them before)
        plan = self.plans.get(model_id)
        if plan is None:
            return
        plan.measuring = True  # until its first cycle
        if not plan.previous:
            return
        peak = None if plan.last_work is None else self._planned((*plan.structure, plan.last_work), self._held(plan))
        if peak is not None:
            fit, floor = set(self._fit(plan, peak, plan.last_work)), set(plan.ranking[:plan.floor])
            self._set(plan, [name for name in plan.ranking if name in floor or name in plan.previous and name in fit])
        plan.previous = set()

    def on_unload(self, model_ids):
        """The models left the VRAM: their measures are kept, their preload is back to their budget's for the next load."""
        for model_id in model_ids:
            plan = self.plans.get(model_id)
            if plan is not None:
                self._end_cycle(plan)
                plan.last, plan.pending, plan.previous = None, set(), set(self.manager.preloaded_blocks_per_model[model_id])
                self._set(plan, plan.ranking[:plan.floor])
            fixed = self.fixed.get(model_id)
            if fixed is not None and len(self.manager.preloaded_blocks_per_model[model_id]) != len(fixed.manual):  # released under pressure
                fixed.last, fixed.relink = None, False
                self.manager.preloaded_blocks_per_model[model_id] = dict(fixed.manual)  # the whole manual preload at the next load
                self._link(fixed)

    def on_state_pressure(self, plan, released, dynamic=False):
        """The state the model's rings kept in VRAM was released under pressure: its room shrinks by as much for the workload (not the
        weights' penalty), and the cycle does not measure the activations. dynamic: state planned with the blocks, whose penalty grows.
        Between the model's passes, no penalty (_between_passes)."""
        if self._between_passes(plan):
            return
        work = plan.workload if plan.workload is not None else plan.last_work
        if dynamic:
            if work is not None:
                key = (*plan.structure, work)
                if key not in self.measures:  # measured from now on: not the budget's preload again
                    self.measures[key], self.measures[(*key, "unmeasured")] = 0, True
                self._penalty(key)[0] += released
            plan.short = plan.grew = True
            return
        if work is not None:
            self.measures.setdefault((*plan.structure, work, "state_penalty"), [0, 0])[0] += released
        plan.state_room, plan.state_short, plan.grew = max(0, plan.state_room - released), True, True

    def on_state_change(self, model_id):
        """The state the model's rings keep in VRAM changed (mmgp.state_ring): the cycle running does not measure its activations."""
        plan = self.plans.get(model_id)
        if plan is not None and plan.workload is not None:
            plan.grew = True
            plan.min_weights = min(plan.min_weights, self._weights(plan, self.manager.preloaded_blocks_per_model[model_id]))

    def on_expert_change(self):
        # The allocator peak is shared by co-tenants. A cycle spanning a change
        # in expert residency cannot measure activation memory independently.
        for model_id, plan in self.plans.items():
            if plan.workload is not None:
                plan.grew = True
                plan.min_weights = self._weights(plan, self.manager.preloaded_blocks_per_model[model_id])

    def on_pressure(self, need, parked=False):
        """The allocator found the VRAM short (without its lock): the plans stop growing, and the preloaded blocks of the models loaded leave
        the VRAM, the least recently used models first (memory_pressure.stamp: an idle co-tenant before the model running), the lowest ranked
        first in a model, except the block running. They are streamed again when they run (chained again at the next cycle), and the penalty
        of the workload grows by what was released (_penalty): its next plans leave that memory free instead of growing back into the same
        shortage at every step. A fixed preload keeps that much less until the model is unloaded. parked: the offload object is parked
        (offload.park): called from any thread, nothing of its models runs, and its measures and penalties stay as they are."""
        m, freed = self.manager, 0
        if not parked and self.thread is None and any(model_id in self.plans or model_id in self.fixed for model_id in m.active_models_ids):
            self.thread = threading.get_ident()  # no cycle ran yet: the thread loading the models is the one that runs their blocks
        # nor while any CUDA graph is captured on this thread's stream (a model's own graphs): unloading blocks would invalidate the capture (a
        # parked object's blocks are not part of the graph recorded)
        if self.dropping or need <= 0 or torch.cuda.is_current_stream_capturing() or not parked and (threading.get_ident() != self.thread or cuda_graphs.recording()):
            return 0
        need += MARGIN
        for model_id in sorted(m.active_models_ids, key=lambda model_id: m.used.get(model_id, 0)):
            plan = self.plans.get(model_id) or self.fixed.get(model_id)
            if plan is None:
                continue
            self._between_passes(plan)
            plan.pending = set()
            preloaded = m.preloaded_blocks_per_model[model_id]
            victims = [name for name in reversed(plan.ranking) if name in preloaded and (parked or name != plan.last and model_id + "/" + name not in m.prefetch_futures)]
            if not victims:
                continue
            if not parked:  # (parked: its work finished when it was parked)
                m.transfer_stream.wait_stream(torch.cuda.current_stream())  # their memory may still be read by the work queued
            released = 0
            for name in victims:  # the blocks of a model being loaded that are not transferred yet free nothing: they all stay out
                in_vram = m.block_in_vram(model_id, name)
                if in_vram and freed + released >= need:
                    continue
                m.gpu_unload_blocks(model_id, name)
                del preloaded[name]
                released += plan.sizes[name] if in_vram else 0
            freed += released
            plan.min_weights = min(plan.min_weights, self._weights(plan, preloaded))
            plan.relink = True  # back in the chain of the streamed blocks at the next cycle, not inside an allocation
            if plan.workload is not None:  # the plan was too large for this workload: its next plans leave the memory released free
                key = (*plan.structure, plan.workload)
                if key not in self.measures:  # measured from now on: not the budget's preload again (its cycles keep blocks, unmeasured)
                    self.measures[key], self.measures[(*key, "unmeasured")] = 0, True
                self._penalty(key)[0] += released
                plan.short = True
            if freed >= need:
                break
        if freed:
            tqdm.write(f"VRAM preload: VRAM short, {freed / ONE_MB:0.0f} MB of preloaded blocks released" + (" (parked models)" if parked else ""))
        return freed

    def on_park(self):
        """The offload object is parked (offload.park): the cycles of its models end here, measured; what other models take from them while
        parked does not count against their workloads."""
        for plan in list(self.plans.values()):
            self._end_cycle(plan)

    def _held(self, plan):
        """VRAM allocated beside the model's weights now in VRAM (its base, its blocks, the slots allocated, expert copies, the state its rings
        keep in VRAM) and beside the reclaimable VRAM: other models, caches, the pipeline's tensors."""
        m, model_id = self.manager, plan.model_id
        base = m.blocks_of_modules.get(model_id, ())[-1:]  # in VRAM once loaded (not yet when the model is about to load)
        weights = (plan.base if base and getattr(base[0][0], base[0][1]).is_cuda else 0) + sum(plan.sizes[name] for name in m.preloaded_blocks_per_model[model_id] if m.block_in_vram(model_id, name))
        weights += sum(slot[2] for slot in m.block_slots.get(model_id, ()) if slot[0] is not None) + m.state_bytes(model_id)
        weights += sum(residency.nbytes for residency in m.expert_residency.values())
        return max(0, allocator.stats()["allocated"] - weights - self._reclaimable(plan))

    def _reclaimable(self, plan):
        """VRAM held beside this model that the pressure callback gives it when it needs it: what parked offload objects hold
        (memory_pressure.reclaimable), and the preloaded blocks of the models of this object that did not run since its previous cycle
        started (an idle co-tenant, e.g. a vision tower after its encoding). Its measures leave it out, so that its plans may take it; the
        blocks of co-tenants that run between its cycles stay in its measures (they share the VRAM as before rather than evict each other at
        every step)."""
        m, idle = self.manager, 0
        for model_id in m.active_models_ids:
            if model_id != plan.model_id and m.used.get(model_id, 0) < plan.cycles[0] and (model_id in self.plans or model_id in self.fixed):
                idle += sum(size for name, size in m.preloaded_blocks_per_model[model_id].items() if m.block_in_vram(model_id, name))
        return idle + memory_pressure.reclaimable(exclude=m)

    def _end_cycle(self, plan):
        if plan.workload is not None:
            key = (*plan.structure, plan.workload)
            state_penalty = self._state_penalty(key)
            if state_penalty is None:
                pass
            elif plan.state_short:
                state_penalty[1] = 0
            elif state_penalty[0]:  # cycles without a shortage of the state: its room may grow back into half of that memory
                state_penalty[1] += 1
                if state_penalty[1] >= PENALTY_PATIENCE:
                    state_penalty[0] = 0 if state_penalty[0] < 2 * PENALTY_MIN else state_penalty[0] // 2
                    state_penalty[1] = 0
            plan.state_short = False
            penalty = self._penalty(key)
            if not plan.grew and not plan.short:  # a cycle that ran short peaked with the blocks it released still counted (the penalty has it)
                measure = allocator.mark_peak() - plan.min_weights - self._reclaimable(plan) - (plan.held if HELD_APART else 0)  # the activations
                self.measures[key] = max(self.measures.get(key, 0), measure)
                if self.measures.pop((*key, "unmeasured"), False) and FIRST_MEASURE_CLEARS_PENALTY:
                    # the shortages before it came from plans that left no room for the activations, which this measure now leaves: their
                    # penalty would count them twice (e.g. a 27B LLM decoding at 11 GiB kept 53 instead of 60 of its 64 blocks)
                    penalty[0], penalty[1], penalty[3] = 0, 0, False
            recovered = plan.recoveries is not None and allocator.stats()["recoveries"] > plan.recoveries
            if recovered:
                # the allocator found the VRAM short and waited for other streams or synchronized the device to reuse memory, which stalls the
                # transfers (H3 at 720p under 11 GB: 14 instead of 12.6 s per step): the next plans leave a block more free
                penalty[0] += plan.largest
            if recovered or plan.short:
                if penalty[3]:  # the last decrease was too much: wait twice as long before the next one
                    penalty[2] = min(2 * penalty[2], PENALTY_PATIENCE_MAX)
                penalty[1], penalty[3] = 0, False
            elif penalty[0]:
                penalty[1] += 1
                if penalty[1] >= penalty[2]:  # enough cycles without shortage: the plans may grow back into half of that memory
                    penalty[0] = 0 if penalty[0] < 2 * PENALTY_MIN else penalty[0] // 2
                    penalty[1], penalty[3] = 0, True
        plan.workload, plan.short = None, False

    def _free_slots(self, plan):
        """No block of the plan is streamed any more: the VRAM slots of the model are freed, with the block loaded in one of them (a block
        loaded while it runs, loaded again into a buffer of its own: the slots are sized again for the blocks streamed when there are). They
        would otherwise hold VRAM, measured as activations. A slot still used by a block in flight is freed at the next cycle."""
        m, model_id = self.manager, plan.model_id
        slots = m.block_slots.get(model_id)
        if not slots:
            return
        m.transfer_stream.wait_stream(torch.cuda.current_stream())  # their memory may still be read by the work queued
        loaded = m.loaded_blocks[model_id]
        if loaded is not None and any(slot[1] == model_id + "/" + loaded for slot in slots):
            m.gpu_unload_blocks(model_id, loaded)
        for slot_no, slot in enumerate(slots):
            if slot[1] is None and slot[0] is not None:
                slot[0] = None
                m.slot_views = {key: views for key, views in m.slot_views.items() if key[1] != slot_no or not key[0].startswith(model_id + "/")}
        if all(slot[1] is None for slot in slots):
            del m.block_slots[model_id]

    def _set(self, plan, names):
        """Records the blocks preloaded and links the others (no VRAM change)."""
        m = self.manager
        m.preloaded_blocks_per_model[plan.model_id] = {name: plan.sizes[name] for name in names}
        m.planned_windows[plan.model_id] = self.window
        m.prefetch_windows.pop(plan.model_id, None)
        m.shuttle_sizes[plan.model_id] = plan.largest
        self._link(plan)

    def _link(self, plan):
        """Chains the streamed blocks as tune_preloading does: in the order they run, a single tower in a cycle (its last streamed block
        prefetches the first one), the preloaded blocks out of the chain."""
        m, model_id = self.manager, plan.model_id
        preloaded = m.preloaded_blocks_per_model[model_id]
        for entry, target in m.next_blocks_names.items():
            # a link learned while the block was streamed (e.g. from a block outside the towers that runs before it) would transfer it again
            # once it is preloaded: its parameters would then point to a buffer still being copied while it runs without waiting for it
            if target is not None and target.startswith(model_id + "/") and target[len(model_id) + 1:] in preloaded:
                m.next_blocks_names[entry] = None
        for name, _ in plan.others:  # out of the chain while preloaded; streamed again, their links are learned as they run
            if name in preloaded:
                entry = model_id + "/" + name
                m.learned_links.discard(entry)
                m.next_blocks_names[entry] = m.prev_blocks_names[entry] = None
        for _, floors in plan.towers:
            first = prev = None
            for name, _ in floors:
                entry = model_id + "/" + name
                m.learned_links.discard(entry)
                if name in preloaded:
                    m.next_blocks_names[entry] = m.prev_blocks_names[entry] = None
                    continue
                if prev is None:
                    first = entry
                else:
                    m.next_blocks_names[prev] = entry
                m.prev_blocks_names[entry] = prev
                prev = entry
            if prev is not None:
                m.next_blocks_names[prev] = None
                if len(plan.towers) == 1:
                    m.next_blocks_names[prev] = first
                    m.prev_blocks_names[first] = prev
                    m.learned_links.add(prev)

    def _complete_loras(self, model_id, entry):
        """A streamed block got the LoRA copies of the adapters active at this step: a preloaded block needs those of every step."""
        m = self.manager
        model = m.models[model_id]
        adapters, data = getattr(model, "_loras_active_adapters", None), getattr(model, "_loras_model_data", None)
        if adapters and data is not None:
            modules = {parent: data[parent] for parent, _, _, _, _ in m.blocks_of_modules[entry] if parent in data}
            if modules:
                m._move_loras(adapters, modules, True, model)

    def _drop(self, plan, names):
        """Preloaded blocks leave the VRAM at the start of a cycle and are streamed again (none: the blocks the pressure callback released
        are chained again)."""
        m, model_id = self.manager, plan.model_id
        preloaded = m.preloaded_blocks_per_model[model_id]
        self.dropping = True
        try:
            if model_id in m.active_models_ids:
                if m.ring is not None:  # the staging ring follows the chain of the streamed blocks: emptied before the chain changes
                    m.wait_prefetch()
                    torch.cuda.synchronize()
                    m.ring.reset()
                    for entry in [entry for entry in m.inflight if entry.startswith(model_id + "/")]:
                        m.gpu_unload_blocks(model_id, entry[len(model_id) + 1:])
                else:
                    m.transfer_stream.wait_stream(torch.cuda.current_stream())  # their memory may still be read by the work queued
                for name in names:
                    m.gpu_unload_blocks(model_id, name)
            for name in names:
                del preloaded[name]
            self._link(plan)
            plan.relink = False
        finally:
            self.dropping = False
