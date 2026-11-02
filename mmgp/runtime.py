# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Model-facing execution services, independent of MMGP's offload/compilation hooks."""
import weakref
from contextlib import contextmanager
from contextvars import ContextVar

import torch


_owners = weakref.WeakKeyDictionary()
_resources = weakref.WeakKeyDictionary()
_resource_users = {}
_execution_checkpoint = ContextVar("mmgp_execution_checkpoint", default=None)


@contextmanager
def execution_scope(checkpoint):
    """Supply a local cancellation check for long MMGP residency operations."""
    token = _execution_checkpoint.set(checkpoint)
    try:
        yield
    finally:
        _execution_checkpoint.reset(token)


def checkpoint():
    callback = _execution_checkpoint.get()
    if callback is not None:
        callback()


def register_resource(module, prepare, release):
    """Tie a shared execution resource to a model's GPU residency.

    prepare(device) runs before its first owner loads, outside graph warmup;
    release() runs after its last owner unloads and its graphs are destroyed.
    Register before profiling, with the same callback pair for models sharing the resource. Neither
    callback may retain the model or allocate resources at registration time.
    """
    _resources.setdefault(module, set()).add((prepare, release))


def _load_resources(manager, model_id):
    model = manager.models[model_id]
    for resource in manager.runtime_resources[model_id]:
        users = _resource_users.setdefault(resource, weakref.WeakSet())
        first = not users
        users.add(model)  # an interrupted preparation is cleaned up by unload_all too
        if first:
            from . import allocator
            with allocator.normal_allocations():
                resource[0](torch.device("cuda", torch.cuda.current_device()))


def _unload_resources(manager, model_ids):
    for model_id in model_ids:
        model = manager.models[model_id]
        for resource in manager.runtime_resources[model_id]:
            users = _resource_users.get(resource)
            if users is None:
                continue
            users.discard(model)
            if not users:
                resource[1]()
                del _resource_users[resource]


def _register(manager, models):
    owners = {model_id: (weakref.ref(manager), model_id) for model_id in models}
    manager.runtime_resources = {}
    for model_id, model in models.items():
        resources = manager.runtime_resources[model_id] = set()
        for module in model.modules():
            _owners[module] = owners[model_id]
            resources.update(_resources.get(module, ()))
    # A separately managed root owns itself even when another model contains it.
    for model_id, model in models.items():
        _owners[model] = owners[model_id]


def _unregister(manager):
    for module, (reference, _) in list(_owners.items()):
        if reference() is manager:
            del _owners[module]


def _owner(module):
    entry = _owners.get(module)
    if entry is None:
        return None
    manager = entry[0]()
    return None if manager is None else (manager, entry[1])


def execution_device(module):
    """Return the execution device, even while managed weights are on CPU.

    An unmanaged module uses its first parameter/buffer's device, or CPU if empty.
    Registration is removed when its offload manager is released.
    """
    from .experts import ExpertBank
    if isinstance(module, ExpertBank):
        return torch.device("cpu")
    if _owner(module) is not None:
        return torch.device("cuda")
    tensor = next(module.parameters(), None)
    if tensor is None:
        tensor = next(module.buffers(), None)
    return torch.device("cpu") if tensor is None else tensor.device


def lookup_on_cpu(*modules):
    """Mark modules that MMGP computes on the CPU, their weights kept in RAM, never in VRAM: a token table only gathers the rows of a
    prompt, the same values as on the GPU, without moving the whole table (2 GB for an umT5 or a Gemma 3) at each load. Their tensor
    inputs go to the CPU and their output comes back to the execution device. Call it on the loaded modules, before the offload
    profile: a quantized table (a module whose tensors are not plain) keeps the GPU path of its qtype. A module tied to a marked one
    (a head sharing the table) must be marked too, with unused when it never runs."""
    for module in modules:
        module._mm_on_cpu = True


def map_in_ram(*modules):
    """Mark modules whose parameters MMGP keeps in reserved RAM mapped in the GPU's address space, never in VRAM: their kernels read the
    rows they need in place, over the bus, inside CUDA graphs too and with indices computed on the GPU (a token table read for a few tokens
    at each step). The parameters stay the modules' own, as CUDA views of their RAM, outside the blocks, base and budgets; the RAM is
    charged to the reserved RAM while it exists. Parameters of other modules sharing their tensors follow them: never mark a table that a
    module reads whole at each call (a head tied to it), which would read it over the bus. Call it on the loaded modules, before the offload
    profile; without CUDA, or without reserved RAM left, the modules keep their usual placement."""
    for module in modules:
        module._mm_mapped = True


def is_mapped(tensor):
    """True for a parameter kept in mapped RAM (map_in_ram): a CUDA view of RAM, whose reads cross the bus."""
    from .mapped import is_mapped
    return is_mapped(tensor)


def unused(*modules):
    """Mark modules that never run, such as the output head of a language model whose hidden states condition a generator (called through
    its base model): MMGP keeps their weights in RAM, out of VRAM, pinning and budgets, quantized or not, and raises if one runs. Call it on
    the loaded modules, before the offload profile."""
    for module in modules:
        module._mm_unused = True


def ensure_loaded(module):
    """Load a managed model's base/preloaded weights; leave block swapping to MMGP.

    Use before custom entry points that read weights without calling a hooked
    forward. Unmanaged modules are unchanged. This does not pin a model in VRAM.
    """
    owner = _owner(module)
    if owner is not None:
        manager, model_id = owner
        manager.ensure_model_loaded(model_id)


def state_ring(blocks, name="state", resident=False):
    """A state ring (mmgp.state_ring.StateRing) for tensors a model computes once per block and reads at every pass (a cached attention
    prefix), blocks: the model's blocks (a ModuleList, called one by one). Kept in RAM and streamed block by block on the transfer stream of
    the model's offload object, with the weights, following MMGP's schedule for these blocks (its prefetch window; records locked for the
    blocks whose weights are pinned, then, for a model MMGP pins, for those kept in VRAM, as far as the reserved RAM allows; the others moved
    ahead through MMGP's staging ring with their blocks): each block's forward fetches its state. The model calls store(block, tensors) once per block, and release() at the end.
    resident: blocks whose weights MMGP keeps in VRAM may keep their state there too, in the room their preload leaves (see mmgp.state_ring;
    the model then reads the lists store() returns at each pass). resident="dynamic" (where dynamic_state(blocks)): a cache the model
    computes on the GPU stays in VRAM first, the model's dynamic preload planning it with its weights, and goes to RAM only for the blocks
    that do not stay there."""
    from .state_ring import StateRing
    owner = _owner(blocks)
    if owner is None:
        return StateRing(len(blocks), torch.device("cuda", torch.cuda.current_device()), None, name, modules=blocks)
    manager, model_id = owner
    prefix = next(name for name, module in manager.models[model_id].named_modules() if module is blocks) + "."
    entries = [f"{model_id}/{prefix}{index}" for index in range(len(blocks))]  # the blocks' entries: the ring follows their schedule
    return StateRing(len(blocks), torch.device("cuda", torch.cuda.current_device()), manager, name, modules=blocks, model_id=model_id, entries=entries,
                     resident=resident)


def dynamic_state(module):
    """Whether a state ring of these blocks can place its state dynamically (state_ring(blocks, resident="dynamic")): their model is managed
    by MMGP with its VRAM allocator, which reports the VRAM shortages."""
    from . import allocator
    return _owner(module) is not None and allocator.active == "vmm"


def weights_resident(module):
    """Whether the weights of a managed module are all in VRAM now (its model loaded, none of the blocks under the module streamed; expert
    banks aside): an MMGP graph stage that runs this module may then span several of its blocks (see cuda_graphs.run). For a submodule
    (e.g. the language model of a larger model), only its own blocks count: blocks the stage never runs may stay out of VRAM. Check it at
    each call: a dynamic preload can release blocks between calls. Unmanaged modules: True."""
    owner = _owner(module)
    if owner is None:
        return True
    manager, model_id = owner
    if model_id not in manager.active_models_ids:
        return False
    names = _block_names.get(manager)
    if names is None:  # the blocks of a model are those of its profile
        names = _block_names[manager] = {}
    model = manager.models[model_id]
    blocks = names.get((model_id, id(module)))
    if blocks is None:
        prefix = model_id + "/" + ("" if module is model else next(name for name, m in model.named_modules() if m is module) + ".")
        blocks = names[(model_id, id(module))] = [name[len(model_id) + 1:] for name, size in manager.blocks_of_modules_sizes.items() if size and name.startswith(prefix)]
    preloaded = manager.preloaded_blocks_per_model[model_id]
    return all(name in preloaded for name in blocks)


_block_names = weakref.WeakKeyDictionary()  # an offload manager -> (model id, id of a module of the model) -> the names of its blocks
_tower_entries = weakref.WeakKeyDictionary()  # a tower (ModuleList) -> (model id, the entry name of each of its blocks)


def resident_runs(module, blocks):
    """The blocks of a tower of a managed model (a ModuleList of its blocks) as runs (start, stop, resident) that cover them in order:
    the blocks of a resident run are kept in VRAM now, the others are streamed when they run (all are resident while their model is
    loaded whole). The first block of a tower where an automatic preload starts its cycles is a run of its own, not resident, as are
    the blocks of a model not loaded yet. Query it at each pass: a dynamic preload changes the blocks kept between calls."""
    owner = _owner(module)
    if owner is None:
        return ((0, len(blocks), True),)
    manager, model_id = owner
    entries = _tower_entries.get(blocks)
    if entries is None or entries[0] != (id(manager), model_id):
        model = manager.models[model_id]
        prefix = next(name for name, submodule in model.named_modules() if submodule is blocks) + "."
        names = [prefix + str(index) for index in range(len(blocks))]
        entries = _tower_entries[blocks] = ((id(manager), model_id), [name if manager.blocks_of_modules_sizes.get(model_id + "/" + name) else None for name in names])
    names = entries[1]
    if model_id not in manager.active_models_ids:
        return ((0, len(blocks), False),)
    preloaded = manager.preloaded_blocks_per_model[model_id]
    flags = [name is None or name in preloaded for name in names]
    plan = None if manager.auto_preload is None else manager.auto_preload.plans.get(model_id)
    if plan is not None and names and plan.first == names[0]:
        flags[0] = False
    runs, start = [], 0
    for index in range(1, len(flags) + 1):
        if index == len(flags) or flags[index] != flags[start] or not flags[start]:
            runs.append((start, index, flags[start]))
            start = index
    return tuple(runs)


def place_experts(module):
    """Place a managed model's optional GPU expert copies now, outside graph recording (e.g. at the first step of a decoding loop replayed
    as one graph, which MMGP's hooks never see): as many as its budget or the dynamic preload allows, ordinary blocks first; never
    shrinks them. Returns False while its ordinary blocks are not all resident yet (call again at a later step). Models without
    experts: True."""
    owner = _owner(module)
    if owner is None:
        return True
    manager, model_id = owner
    residency = manager.expert_residency.get(model_id)
    return residency is None or residency.place()


def dynamic_preload(module):
    """Whether an automatic VRAM preload manages this model's blocks: it keeps as many as fit and releases some when another allocation
    needs VRAM (a co-tenant loaded beside it then takes only what it needs), keeping them again later as they run."""
    owner = _owner(module)
    if owner is None:
        return False
    manager, model_id = owner
    return manager.auto_preload is not None and model_id in manager.auto_preload.plans


def requires_layer_offload(module):
    """Whether this owner's weights may move during execution or include CPU experts.

    Whole-model external CUDA graphs need stable weight addresses. A dynamic
    preload remains reclaimable even when it currently happens to fit entirely.
    MMGP's own block graphs support these models and share their release lifecycle.
    """
    owner = _owner(module)
    if owner is None:
        return False
    manager, model_id = owner
    if model_id in manager.expert_residency:
        return True
    if manager.auto_preload is not None and model_id in manager.auto_preload.plans:
        return True
    return any(name.startswith(model_id + "/") and size for name, size in manager.blocks_of_modules_sizes.items())


DISPOSABLE_PRIORITY = 2  # in the VRAM pressure order (mmgp.memory_pressure): after expert copies and ring state (0) and preloaded blocks (1)


class Disposable:
    """VRAM held by a managed model's own code beside its weights (a cache, a workspace) that it can do without and rebuild: see
    disposable()."""

    def __init__(self, manager, model_id, name, release):
        self.manager, self.model_id, self.name, self._release = manager, model_id, name, release
        self.released, self.idle_thread, self.idle_depth = False, None, 0
        if manager is not None:
            from . import memory_pressure
            memory_pressure.register(self, DISPOSABLE_PRIORITY)

    def close(self):
        """The asset is gone (freed by its owner): no more release."""
        self.released = True
        if self.manager is not None:
            from . import memory_pressure
            memory_pressure.unregister(self)

    @contextmanager
    def idle(self):
        """Its owner does not use it meanwhile (code on this thread that runs other models): MMGP may release it if VRAM runs short."""
        import threading
        self.idle_thread, self.idle_depth = threading.get_ident(), self.idle_depth + 1
        try:
            yield
        finally:
            self.idle_depth -= 1

    def on_pressure(self, need, parked=False):
        """mmgp.memory_pressure: released when VRAM runs short, while its model's offload object is parked (any thread) or while its owner
        declared it idle (its thread), never during a CUDA graph capture. Returns the bytes the allocator got back."""
        import threading
        from . import allocator
        if need <= 0 or self.released or torch.cuda.is_current_stream_capturing() or not (parked or self.idle_depth and threading.get_ident() == self.idle_thread):
            return 0
        before = allocator.stats()["allocated"]
        self.close()
        try:
            self._release()
        except Exception as exc:  # the owner's code, inside an allocation: the allocation goes on with what was freed
            from tqdm import tqdm
            tqdm.write(f"VRAM short: the release of {self.name} failed: {exc}")
        freed = max(0, before - allocator.stats()["allocated"])
        if freed:
            from tqdm import tqdm
            tqdm.write(f"VRAM short, {freed / 2**20:0.0f} MB of {self.name} released")
        return freed


def disposable(module, name, release):
    """Declare VRAM that the code of a managed model holds beside its weights and can do without (e.g. the KV cache of a language model, a
    workspace): MMGP releases it, calling release(), only when an allocation finds the VRAM short (the automatic preloads of other models do
    not plan with it), in its pressure order: after the expert copies, the state kept in VRAM and the preloaded blocks, and only while it is
    not in use, i.e. while the model's offload object is parked (offload.park) or inside the returned Disposable's idle() scope on the
    owner's thread. release() frees the asset (it may copy it to RAM; it must not allocate VRAM) and runs at most once: the owner checks
    `released` before its next use and rebuilds it (declaring it again). close() when the owner frees it itself. Unmanaged modules get a
    Disposable that is never released."""
    owner = _owner(module)
    return Disposable(None, None, name, release) if owner is None else Disposable(owner[0], owner[1], name, release)
