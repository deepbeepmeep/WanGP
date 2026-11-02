"""Automatic CUDA graphs of the blocks of the towers (the repeated layers of a transformer): the host work of a block call (Python,
kernel launches) is recorded once and replayed, for models whose steps are short enough for the host to be the bottleneck.

Inside `scope()` (the denoising loop of a model; elsewhere blocks run as usual), each call of a block is keyed by the tensor objects of
its weights, its active LoRAs and their scalings, the shapes of its tensor arguments and extra inputs, its other arguments and the
registered global keys: the first call of a key runs eagerly (it compiles and tunes the kernels, makes the caches of the weights), then
is captured into a CUDA graph that the next calls replay. Blocks run eagerly, without captures, while the placement of their weights is not
settled (`ready()` False: e.g. an automatic preload measuring a new workload, which keeps other blocks in VRAM at the next step). A block thus has a graph per location of its weights (kept in VRAM, or each
slot of the swapped blocks), and a model loaded again captures its graphs again. The tensor arguments, extra inputs (tensors that come
and go with a block, e.g. its LoRA copies) and outputs go through static buffers shared by the graphs, the arguments a block modifies in
place are copied back when needed, and an output that is an argument is that argument. A block keeps its `MAX_KEYS` most recently used graphs (a new
resolution replaces them); all the graphs go with the model or `invalidate()`. With MMGP's VMM allocator, eager preparation and capture
share a pool. Suitable live arguments and first-call outputs supply the static storage; retained caller outputs are protected before
another call can overwrite them. A large argument that only its caller holds (a tower's hidden state, new at each step) moves into its
static buffer instead of being copied there: the caller's tensor takes the buffer's storage and frees its own, so the graphs keep no
second copy of it and the next blocks copy nothing; while the caller holds it, other calls use another buffer. Eager stages beside the
blocks (input preparation, output head) can run on the pool's idle memory with `pooled()`. Graph warmup is scoped to its manager's pool.
A consumed single-tensor list can transfer its input's execution lifetime: after its last execution view dies, later temporaries may use
its physical pages while every graph address stays fixed. Caller references, storage aliases and recorded stream consumers prevent this
transfer. Large outputs can reuse their warmup storage at the corresponding captured allocation; a changed allocation trace is recaptured
without that output plan. These plans add no replay-time mapping or synchronization, and do not participate in ordinary, non-graph
allocation.

Contract of a block call in a scope: the block reads only its weights, its tensor arguments (in lists, tuples, dicts and dataclasses),
the state registered for its model (register_state) and tensors it creates; its effects are its outputs and in-place updates of its
tensor arguments or of that state (Python side effects are not replayed, and the capture runs the Python of the block once more); no
synchronization or CPU round trip. The capture checks the reads and raises an error naming any other tensor.

A model's own loop over a tower (run_blocks) replays each run of consecutive blocks kept in VRAM as one graph, while the blocks streamed
in between go through the hooks that transfer them.
"""
import contextlib
import dataclasses
import enum
import sys
import traceback
import weakref

import torch
from tqdm import tqdm
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode
from . import runtime

MAX_KEYS = 8  # graphs kept per block, the least recently used dropped first: weight locations, CFG branches of other lengths, resolutions
MOVE_BYTES = 32 << 20  # arguments from this size move into their static buffer when only their caller holds them
ESCAPE_WARMUP = True  # comparison switch: the result of a key's first call leaves the graph pool (_escape)
RECAPTURE_HELD = True  # comparison switch: a graph whose replay may overwrite a buffer held since a later capture is captured again
_SCALARS = (bool, int, float, str, type(None), torch.dtype, torch.device)
_depth = 0
_recording = 0
_key_providers = []
_gates = []
_blocked = None  # why the blocks of the scope running are not graphed (a gate's reason), or None
_reported = set()
_capture_setups = []
_constants = []  # tensors that blocks may read besides their weights and arguments: kept for the life of the process
_capture_streams = {}
_blas_workspaces = set()
_registered = set()
_managers = []
_capturing = False  # a block's capture is running (its forward is recorded, nothing runs on the GPU)
_autotune_guarded = False
_static_storages = weakref.WeakValueDictionary()  # storage address -> a static buffer of some graph manager (`_mm_graphs`)


def _guard_triton_autotune():
    """Triton's autotuners (@triton.autotune) benchmark their configurations on a cache miss: inside a capture, the benchmark launches
    would be recorded in the graph and CUDA refuses its timing synchronizations. The eager call that precedes each capture tunes the keys
    a block uses (sizes and dtypes); a miss that still happens in a capture (a key that differs there) raises an error naming the kernel."""
    global _autotune_guarded
    _autotune_guarded = True
    try:
        from triton.runtime import autotuner
    except ImportError:
        return
    bench = autotuner.Autotuner._bench

    def guarded_bench(self, *args, **kwargs):
        if _capturing:
            fn = getattr(self, "base_fn", self.fn)
            raise RuntimeError(f"the Triton kernel {getattr(fn, '__name__', fn)} would tune itself (@triton.autotune) during a CUDA graph capture, "
                               f"for a key the eager call before the capture did not use")
        return bench(self, *args, **kwargs)

    autotuner.Autotuner._bench = guarded_bench


@contextlib.contextmanager
def scope(enabled=True, gated=True):
    """Block calls in this scope use CUDA graphs when the mode is on for their model (a no-op otherwise). gated=False: the registered
    gates do not apply (blocks that never use what they check, e.g. an LLM with its own attention kernels)."""
    global _depth, _blocked
    if enabled and _depth == 0:
        _blocked = next((reason for reason in (gate() for gate in _gates) if reason), None) if gated else None
        if _blocked is not None and _blocked not in _reported:
            _reported.add(_blocked)
            tqdm.write(f"[MMGP] CUDA graphs are off: {_blocked}.")
    _depth += bool(enabled)
    try:
        yield
    finally:
        _depth -= bool(enabled)


def recording():
    """True while a block of a graphed model runs to be captured (its first, eager call and its capture): a kernel launched outside
    PyTorch's current stream (e.g. on the legacy default stream) would be missing from its graph, so code with such kernels uses others."""
    return _recording > 0


def _graphs(module):
    owner = runtime._owner(module)
    if owner is None:
        return None
    manager, model_id = owner
    return manager.cuda_graphs.get(model_id)


def is_enabled(module):
    """Whether a managed module can use graphs in the current scope.

    Includes mode selection, registered gates and weight-residency readiness.
    Use only to skip preparation specific to a graph stage; run() also enforces
    these conditions. Unmanaged/released modules return False.
    """
    graphs = _graphs(module) if _depth else None
    return graphs is not None and _blocked is None and graphs.disabled is None and graphs.ready()


def run(module, *args, owner=None, method="forward", enabled=True, **kwargs):
    """Execute an extra stage using its owner's existing graph cache and lifetime.

    Call inside scope(). An unregistered stage may supply a managed owner whose
    weights it uses; its parameters need not be registered a second time. The
    selected method must satisfy the same graph contract as a tower block.
    Outside a scope, with enabled=False, or without an active graph manager,
    execute the method normally. Gates/readiness and cache invalidation/release
    remain MMGP-owned. PyTorch compilation does not enable these CUDA graphs.
    A stage may span several blocks of its owner (e.g. a whole decoding step) only while
    runtime.weights_resident(owner): graphed blocks it calls are then captured inline.
    """
    forward = getattr(module, method)
    graphs = _graphs(module if owner is None else owner) if enabled and _depth else None
    return forward(*args, **kwargs) if graphs is None else graphs.call(module, forward, args, kwargs)


def pooled(owner, fn, *args, **kwargs):
    """Run an eager stage beside the graphed blocks of `owner`'s model (its input preparation, its output head) on the idle memory of the
    model's graph pool, where its temporaries would otherwise add to the pool's memory.

    Call inside scope(); fn runs normally while the model's graphs are off. The stage keeps nothing but its
    outputs (no cache filled, no tensor stored elsewhere: checked). Its tensor outputs leave the pool before any graph replays: a
    large one (a hidden state) moves into the free static buffer of its signature, which the next block then reads without a copy,
    the others move to ordinary memory.
    """
    from . import allocator
    graphs = _graphs(owner) if _depth and not _recording and allocator.active == "vmm" else None
    if graphs is None or _blocked is not None or graphs.disabled is not None or not graphs.ready():
        return fn(*args, **kwargs)
    return graphs.pooled(fn, args, kwargs)


def register_state(owner, name, tensors, holders=()):
    """Tensors that the graphs of owner's model read or update in place besides its weights and arguments (caches, recurrent states,
    buffers its loop fills before each call), at fixed addresses until release_state: never copied, whatever the allocator, and allowed
    by the read check. holders: objects holding these tensors that blocks take as arguments (e.g. a transformers StaticCache passed to
    every layer), keyed by their identity and passed as they are. A new registration under the same name replaces the previous one and
    drops the model's graphs."""
    graphs = _graphs(owner)
    if graphs is None:
        return
    if _recording:
        raise RuntimeError("CUDA graph state must be registered before its graphs are prepared or captured")
    tensors = tuple(t for t in tensors if type(t) is torch.Tensor and t.is_cuda)  # plain tensors: subclasses (weights) are not state
    holders = tuple(holders)
    previous, previous_holders = graphs.state.get(name), graphs.holders.get(name, ())
    if (previous is not None and len(previous) == len(tensors) and all(a is b for a, b in zip(previous, tensors))
            and len(previous_holders) == len(holders) and all(a is b for a, b in zip(previous_holders, holders))):
        return
    if previous is not None:
        graphs.clear()
    graphs.state[name], graphs.holders[name] = tensors, holders
    graphs.state_ids = {id(t) for state in graphs.state.values() for t in state}
    graphs.holder_ids = {id(h) for state in graphs.holders.values() for h in state}


def release_state(owner, name=None):
    """Forgets the state registered under name (all of owner's when None) and drops the graphs of its model, which read it."""
    graphs = _graphs(owner)
    if graphs is None or not graphs.state:
        return
    if name is None:
        graphs.state.clear()
        graphs.holders.clear()
    elif graphs.state.pop(name, None) is None:
        return
    graphs.holders.pop(name, None)
    graphs.state_ids = {id(t) for state in graphs.state.values() for t in state}
    graphs.holder_ids = {id(h) for state in graphs.holders.values() for h in state}
    graphs.clear()


def run_blocks(owner, blocks, step, carry, *shared):
    """Runs the blocks of a tower of owner's model (a ModuleList) in order, carry = step(index, block, carry, *shared), and returns the last
    carry: a list of tensors that each step takes over (it may empty it to free them early) and returns anew. Inside scope(), each run of
    consecutive blocks kept in VRAM is one graph, the others go through MMGP's hooks (transfers, block graphs); the first block of a tower
    whose automatic preload starts its cycles there runs through its hooks. The step of a run follows the contract of a block call (no
    Python side effect)."""
    graphs = _graphs(owner) if _depth and not _recording else None
    if graphs is None or _blocked is not None or graphs.disabled is not None or not graphs.ready():
        for index, block in enumerate(blocks):
            carry = step(index, block, carry, *shared)
        return carry
    index, runs, current = 0, runtime.resident_runs(owner, blocks), True
    while index < len(blocks):
        _, stop, resident = next(run for run in runs if run[0] <= index < run[1])
        if resident and stop - index > 1 and not current:
            # The hooks of the blocks run since may have changed the preload plan or released blocks under memory pressure: a span is
            # chosen from the residency after them, once per span rather than once per block
            runs, current = runtime.resident_runs(owner, blocks), True
            continue
        if resident and stop - index > 1:
            span = graphs.span(blocks, index, stop, step, runs)
            carry = graphs.call(span, span.forward, (carry, shared), {})
            index = stop
        else:
            carry = step(index, blocks[index], carry, *shared)
            index, current = index + 1, False
    return carry


class _Span(torch.nn.Module):
    """Consecutive blocks of a tower kept in VRAM: one graph stage of run_blocks, keyed and read-checked by their weights."""

    def __init__(self, blocks, start, stop, step, max_keys):
        super().__init__()
        self.blocks, self.start, self._mm_graph_keys = torch.nn.ModuleList(list(blocks)[start:stop]), start, max_keys
        self.__dict__["step"] = step  # a bound method of the model: not one of its modules

    def forward(self, carry, shared):
        for offset, block in enumerate(self.blocks):
            carry = self.step(self.start + offset, block, carry, *shared)
        return carry


def register_key(provider):
    """provider() -> hashable: a global state that changes what blocks compute (attention mode, kernel options), part of every key."""
    if provider not in _key_providers:
        _key_providers.append(provider)


def register_gate(gate):
    """gate() -> None, or the reason why the blocks must run eagerly (e.g. a kernel that cannot be captured): checked when a scope starts."""
    if gate not in _gates:
        _gates.append(gate)


def register_constant(tensor):
    """A tensor that blocks may read and that is never freed nor replaced (a process-wide cache, e.g. a rotation matrix): graphs read it
    at its address."""
    from . import allocator
    if allocator.in_graph_warmup():
        if _capturing:
            raise RuntimeError("A persistent CUDA graph constant must be prepared during warmup, before capture")
        with allocator.normal_allocations():
            tensor = tensor.clone()
    _constants.append(tensor)
    return tensor


def register_capture_setup(setup):
    """setup(stream) is called before each capture on the capture stream, the one stream of the captures of a device: it prepares what a
    library would allocate there on first use, which a capture cannot do (per-stream workspaces), and returns at once when it is ready."""
    if setup not in _capture_setups:
        _capture_setups.append(setup)


def _prepare_blas_stream(stream):
    # PyTorch owns these workspaces per handle/stream, beyond the life of a model's graphs. Initialize both cuBLAS and cuBLASLt
    # before warmup/capture can put them in graph scratch (which would also pin that entire pool after the model is released).
    from . import allocator
    with allocator.normal_allocations(), torch.cuda.stream(stream), torch.inference_mode():
        key = (torch.cuda.current_blas_handle(), stream.cuda_stream)
        if key not in _blas_workspaces:
            matrix = torch.zeros((16, 16), dtype=torch.float16, device=stream.device)
            torch.addmm(matrix[0], matrix, matrix)
            _blas_workspaces.add(key)


def _capture_stream(device):
    stream = _capture_streams.get(device, None)
    if stream is None:
        stream = _capture_streams[device] = torch.cuda.Stream(device)
    for setup in _capture_setups:
        setup(stream)
    return stream


_BLAS_OPS = {getattr(torch.ops.aten, name) for name in ("mm", "addmm", "bmm", "baddbmm", "addbmm", "mv", "addmv", "dot", "vdot", "_addmm_activation", "_int_mm",
                                                         "_scaled_mm", "_scaled_mm_v2", "_grouped_mm") if hasattr(torch.ops.aten, name)}


class _BlasUse(TorchDispatchMode):
    """The eager warmup of a graph: its first cuBLAS operation prepares the stream's cuBLAS workspaces outside the graph pool, and the capture
    stream gets them only then. A model computing without cuBLAS (e.g. with its own kernels) keeps no workspace for the capture stream."""

    def __init__(self):
        super().__init__()
        self.used = False

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if not self.used and func.overloadpacket in _BLAS_OPS:
            self.used = True
            _prepare_blas_stream(torch.cuda.current_stream())
        return func(*args, **(kwargs or {}))


def invalidate():
    """Drops every graph, e.g. after a change of the settings that the graphs depend on."""
    for manager in _managers:
        manager.clear()


def release(managers):
    """Drops the graphs of these managers for good (their offload object is released)."""
    for manager in list(managers):
        manager.clear()
        if manager in _managers:
            _managers.remove(manager)


def _register_dataclass(cls):
    fields = [field.name for field in dataclasses.fields(cls)]
    pytree.register_pytree_node(cls, lambda obj: ([getattr(obj, name) for name in fields], None),
                                lambda values, _: cls(**dict(zip(fields, values))), serialized_type_name=f"{cls.__module__}.{cls.__qualname__}")
    _registered.add(cls)


class _OverlappingArgument(Exception):
    def __init__(self, index):
        self.index = index


def _argument_path(tree, target, path):
    """Where `target` sits in a block call's arguments (lists, tuples, dicts, dataclasses), for error messages."""
    if tree is target:
        return path
    if dataclasses.is_dataclass(tree) and not isinstance(tree, type):
        items = [(f".{field.name}", getattr(tree, field.name)) for field in dataclasses.fields(tree)]
    elif isinstance(tree, dict):
        items = [(f"[{key!r}]", value) for key, value in tree.items()]
    elif isinstance(tree, (list, tuple)):
        items = [(f"[{index}]", value) for index, value in enumerate(tree)]
    else:
        return None
    return next((found for suffix, value in items if (found := _argument_path(value, target, path + suffix)) is not None), None)


def _flatten(tree):
    while True:
        leaves, spec = pytree.tree_flatten(tree)
        new = {type(leaf) for leaf in leaves if dataclasses.is_dataclass(leaf) and not isinstance(leaf, type) and type(leaf) not in _registered}
        if not new:
            return leaves, spec
        for cls in new:
            _register_dataclass(cls)


def _signature(leaves, holders=frozenset()):
    signature = []
    for leaf in leaves:
        if isinstance(leaf, torch.Tensor):
            signature.append((tuple(leaf.shape), leaf.stride(), leaf.dtype, leaf.device))
        elif isinstance(leaf, _SCALARS) or isinstance(leaf, enum.Enum):
            signature.append(leaf)
        elif isinstance(leaf, slice):
            signature.append(("slice", leaf.start, leaf.stop, leaf.step))
        elif id(leaf) in holders:  # an object holding registered state (register_state): the same object, its tensors in place
            signature.append(("state", id(leaf)))
        else:
            raise TypeError(f"CUDA graph block call: an argument of type {type(leaf).__name__} is not supported, pass its tensors as arguments")
    return tuple(signature)


def _storage(t):
    return t.untyped_storage().data_ptr()


def _same_buffer(a, b):
    return b is not None and a.data_ptr() == b.data_ptr() and a.shape == b.shape and a.stride() == b.stride()


def _unshared(tensor):
    # The static TensorImpl and this temporary Storage wrapper are the only owners. Protect detached views and raw Storage holders too.
    storage = tensor.untyped_storage()
    return torch._C._storage_Use_Count(storage._cdata) == 2 and sys.getrefcount(storage) == 3


def _adoptable(tensor, owner):
    # Another manager's static buffer (an output it returned) is never adopted: sharing its storage would keep that output retained,
    # and its graphs would be evicted and captured again at each of their calls.
    from . import allocator
    static = _static_storages.get(_storage(tensor)) if tensor.is_cuda else None
    return (type(tensor) is torch.Tensor and tensor.is_cuda and not tensor.requires_grad and tensor.storage_offset() == 0 and
            torch._debug_has_internal_overlap(tensor) == 0 and tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size() and
            allocator.owns_tensor(tensor) and (static is None or static._mm_graphs() is owner))


def _movable(leaf, static):
    # The caller's tensor alone holds its storage (no view, no other holder), and the static buffer is an allocation of the VMM allocator,
    # which orders its reuse after the streams that used it.
    from . import allocator
    return (allocator.active == "vmm" and type(leaf) is torch.Tensor and not leaf.requires_grad and leaf.numel() * leaf.element_size() >= MOVE_BYTES
            and (torch.is_inference_mode_enabled() or not leaf.is_inference()) and _unshared(leaf) and allocator.owns_tensor(static))


def _buffer_alias(tensor):
    # A non-inference TensorImpl tracks in-place updates during capture without allocating another activation.
    with torch.inference_mode(False):
        return torch.empty(0, dtype=tensor.dtype, device=tensor.device).set_(tensor.untyped_storage(), tensor.storage_offset(), tensor.shape, tensor.stride())


class _ReadCheck(TorchDispatchMode):
    """Records, during a capture, the CUDA tensors read by the operations that are neither allowed nor created by the capture."""

    def __init__(self, allowed):
        super().__init__()
        self.allowed, self.unknown = allowed, []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        for leaf in pytree.tree_leaves((args, kwargs)):
            if type(leaf) is torch.Tensor and leaf.is_cuda and _storage(leaf) not in self.allowed and len(self.unknown) < 8:
                self.unknown.append(f"{func} reads a tensor {tuple(leaf.shape)} {leaf.dtype}\n" + "".join(traceback.format_stack(limit=12)[:-2]))
        out = func(*args, **kwargs)
        for leaf in pytree.tree_leaves(out):
            if isinstance(leaf, torch.Tensor) and leaf.is_cuda:
                self.allowed.add(_storage(leaf))
        return out


class _Graph:
    __slots__ = ("graph", "positions", "static", "extra_static", "static_key", "mutated", "out_leaves", "out_spec", "outputs", "lifetimes", "emptied", "seq")


def _handoffs(args, kwargs):
    """Single-tensor lists can transfer ownership to a block. Other argument containers retain their normal lifetime."""
    found, seen = {}, set()
    pending = [args, kwargs]
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value, list) and len(value) == 1 and isinstance(value[0], torch.Tensor):
            found[id(value[0])] = value
        elif isinstance(value, (list, tuple)):
            pending.extend(value)
        elif isinstance(value, dict):
            pending.extend(value.values())
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            pending.extend(getattr(value, field.name) for field in dataclasses.fields(value))
    return found


def _execute_lifetimes(forward, args, kwargs, positions, leaves):
    """Plan consumed input storage during warmup/capture; no operation dispatch interception or replay-time remapping."""
    if not positions:
        return forward(*args, **kwargs), ()
    from . import allocator
    containers = _handoffs(args, kwargs)
    bindings = []
    released = []
    try:
        for index in positions:
            source = leaves[index]
            container = containers.get(id(source))
            if container is None:
                continue
            view, token = allocator.lifetime_view(source)
            if token is not None:
                bindings.append((index, container, source, weakref.ref(view), token, view._version))
                container[0] = view
            del view
        out = forward(*args, **kwargs)
        # Rebind returned input views to the ordinary owner. Escaped results must own their storage beyond this execution.
        flat, spec = _flatten(out)
        by_storage = {_storage(source): source for _, _, source, _, _, _ in bindings}
        for index, value in enumerate(flat):
            if isinstance(value, torch.Tensor) and (source := by_storage.get(_storage(value))) is not None:
                if _same_buffer(source, value) and source.dtype == value.dtype:
                    flat[index] = source
                else:
                    with torch.inference_mode(False):
                        flat[index] = torch.empty(0, dtype=value.dtype, device=value.device).set_(
                            source.untyped_storage(), value.storage_offset(), value.shape, value.stride())
        out = pytree.tree_unflatten(flat, spec)
    finally:
        for index, container, source, ref, token, version in bindings:
            if allocator.end_lifetime(token):
                released.append(index)
            view = ref()
            if view is not None:
                if view._version != version:
                    torch.autograd.graph.increment_version(source)
                if len(container) == 1 and container[0] is view:
                    container[0] = source
    return out, tuple(released)


class BlockGraphs:
    """CUDA graphs of the tower blocks of one model."""

    def __init__(self, name, weights_key, weight_tensors, extra_inputs, ready, forget=None):
        # extra_inputs(block) -> [(list, index)]: tensors read by the block that are neither weights nor arguments, read by its graphs from
        # static buffers (their list items are these buffers during a capture); ready() -> False while the weights may move; forget(): the
        # graphs are gone, so are the records that weights_key keeps on the blocks
        self.name, self.weights_key, self.weight_tensors, self.extra_inputs, self.ready = name, weights_key, weight_tensors, extra_inputs, ready
        self.forget = forget
        self.disabled = None  # the reason why this model's blocks run eagerly (e.g. weights whose kernels cannot be captured), or None
        self.reported = False
        self.checked = set()  # (block, method, signature) whose capture checked the reads
        self.blas = {}  # (block, method, signature) -> whether its eager warmup used cuBLAS (its capture stream then needs the workspaces)
        self.state = {}  # name -> tensors that the graphs read besides weights and arguments (register_state): kept across clear()
        self.state_ids = set()  # their ids
        self.holders = {}  # name -> the objects holding these tensors that blocks take as arguments
        self.holder_ids = set()  # their ids
        self.clear()
        _managers.append(self)

    def clear(self):
        # entries: key -> graph; keys: block -> its keys, least recently used first; static: buffers shared by the graphs
        self.entries, self.keys, self.static, self.pool = {}, {}, {}, None
        self.static_by_buffer, self.static_by_base = {}, {}
        self.borrowed = set()
        self._tensor_refs = {}
        self.spans, self.span_runs = {}, {}  # (tower, start, stop, step) -> _Span; (tower, step) -> the runs its spans were made for
        self.memo = {}  # (module, its first argument) -> its last call replayed through the fast path of call()
        # output buffers that a capture put in the pool, by the order of that capture: the scratch of the graphs captured before may lie
        # over them (the pool shares it in the order of the captures), which their replays overwrite (_held_later)
        self.captures, self.later = 0, {}
        _blas_workspaces.clear()  # library caches may be released between models too; prepare them again before a new capture
        if self.forget is not None:
            self.forget()

    def span(self, blocks, start, stop, step, runs):
        # the stage of a run of blocks kept in VRAM; when the runs change, the stages of the runs gone leave with their graphs
        func = getattr(step, "__func__", step)
        if self.span_runs.get((id(blocks), func)) != runs:
            self.span_runs[(id(blocks), func)] = runs
            current = {(id(blocks), a, b, func) for a, b, resident in runs if resident and b - a > 1}
            for key in [key for key in self.spans if key[0] == id(blocks) and key[3] is func and key not in current]:
                for entry in list(self.keys.pop(id(self.spans.pop(key)), {})):
                    self._forget(self.entries.pop(entry))
            self._prune()
        key = (id(blocks), start, stop, func)
        span = self.spans.get(key)
        if span is None:
            span = self.spans[key] = _Span(blocks, start, stop, step, getattr(blocks[start], "_mm_graph_keys", MAX_KEYS))
        return span

    def wrap(self, module, forward):
        def graphed_forward(*args, **kwargs):
            if _depth == 0:
                return forward(*args, **kwargs)
            return self.call(module, forward, args, kwargs)
        return graphed_forward

    def call(self, module, forward, args, kwargs):
        global _recording
        # A block called by a stage being prepared or captured (a stage spanning several blocks) runs inside that stage's graph.
        if _recording or _blocked is not None or self.disabled is not None or not self.ready():
            return forward(*args, **kwargs)
        method = getattr(forward, "__func__", forward)
        # The same call as the last one of this module on registered state only (an engine calling a stage on its persistent buffers): the
        # same argument objects at the same addresses, a graph with no output buffer, no LoRA copy nor lifetime transfer. Only the keys that
        # may change are read. Arguments that are not registered state are never kept (they would stay alive, and their moves rely on counts).
        memo = self.memo.get((id(module), id(args[0]) if args else None)) if not kwargs else None
        if (memo is not None and memo[0] is method and len(memo[1]) == len(args) and all(a is b for a, b in zip(args, memo[1]))
                and all(p is None or a.data_ptr() == p for a, p in zip(args, memo[2]))):
            weights, providers = self.weights_key(module), tuple(provider() for provider in _key_providers)
            entry = memo[5]
            # its weights key is the same tuple object while the weights stay where they are (_graph_weights_key); an evicted graph leaves memo
            if (weights[0] is memo[3][0] and weights[1:] == memo[3][1:] and providers == memo[4] and not self.extra_inputs(module)
                    and not self._held_later(entry)):
                copies, borrowed, mutated, result, streams = memo[6]
                for static, leaf in copies:
                    static.copy_(leaf)
                entry.graph.replay()
                stream = torch.cuda.current_stream()
                if stream not in streams:  # recording a stream again on the same buffers would change nothing
                    streams.append(stream)
                    for static in borrowed:
                        static.record_stream(stream)
                for leaf, static in mutated:
                    if _same_buffer(leaf, static):
                        torch.autograd.graph.increment_version(leaf)
                    else:
                        leaf.copy_(static)
                return result
        leaves, spec = _flatten((args, kwargs))
        signature = _signature(leaves, self.holder_ids)
        extras = self.extra_inputs(module)
        extra_signature = _signature([holder[index] for holder, index in extras])
        positions = [index for index, leaf in enumerate(leaves) if isinstance(leaf, torch.Tensor)]
        input_signatures = [signature[index] for index in positions] + list(extra_signature)
        try:
            static_key, buffers = self._buffers(input_signatures, [leaves[index] for index in positions] + [None] * len(extras), arguments=True)
        except _OverlappingArgument as overlap:
            leaf = leaves[positions[overlap.index]]
            name = _argument_path(args, leaf, "args") or _argument_path(kwargs, leaf, "kwargs")
            raise ValueError(f"CUDA graph block call of '{self.name}' ({type(module).__name__}): argument {name} is a view whose elements share memory "
                             f"(shape {tuple(leaf.shape)}, stride {leaf.stride()}, e.g. from expand), which a graph's static buffer cannot hold. Pass the "
                             f"unexpanded tensor and expand it inside the block's forward (no cost), or make it contiguous before the call.") from None
        key = (id(module), spec, signature, extra_signature, self.weights_key(module), tuple(provider() for provider in _key_providers), tuple(static_key), method)
        keys = self.keys.setdefault(id(module), {})
        entry = self.entries.get(key, None)
        lifetimes = None
        if entry is not None and entry.lifetimes:
            lifetimes = self._consumable(args, kwargs, leaves, positions, buffers)
            if not set(entry.lifetimes).issubset(lifetimes):
                key = (*key, "retained-inputs")
                entry = self.entries.get(key)
                lifetimes = ()
        retained_output = entry is not None and any(k in self.borrowed and not _unshared(self.static[k]) for k in entry.static_key[len(buffers):])
        if retained_output or entry is not None and self._held_later(entry):  # its replay would overwrite a buffer the caller holds
            self._evict(key, static_key)
            entry = None
        if entry is None:  # first call of this key: eager (it compiles and tunes the kernels, makes the caches of the weights), then captured
            if lifetimes is None:
                lifetimes = self._consumable(args, kwargs, leaves, positions, buffers)
            if len(keys) >= getattr(module, "_mm_graph_keys", MAX_KEYS):
                self._evict(next(iter(keys)), static_key)
            for ordinal, index in enumerate(positions):  # before the warmup, which then runs on the moved arguments
                if not _same_buffer(buffers[ordinal], leaves[index]) and _movable(leaves[index], buffers[ordinal]):
                    self._move(static_key[ordinal], leaves[index])
            _recording += 1
            try:
                from . import allocator
                if self.pool is None:
                    self.pool = torch.cuda.graph_pool_handle()
                with allocator.graph_warmup(self.pool):
                    plan = allocator.OutputPlan(self.pool)
                    try:
                        blas_key = (id(module), method, None if module.__dict__.get("_mm_graph_check_once", False) else signature)
                        if blas_key in self.blas:  # decided at an earlier capture of the same call: its workspaces are kept
                            if self.blas[blas_key]:
                                _prepare_blas_stream(torch.cuda.current_stream())
                            out, consumed = _execute_lifetimes(forward, args, kwargs, lifetimes, leaves)
                        else:
                            blas = _BlasUse()
                            with blas:
                                out, consumed = _execute_lifetimes(forward, args, kwargs, lifetimes, leaves)
                            self.blas[blas_key] = blas.used
                        plan.pause()
                        if consumed:
                            # An escaped warmup result must not retain pages loaned by a static input: the preceding graph may
                            # write that input again before this graph can protect its retained outputs. Stabilize only overlapping
                            # results, after input claims were restored, so capture copies to independent output storage as usual.
                            flat, out_spec = _flatten(out)
                            for index, value in enumerate(flat):
                                if isinstance(value, torch.Tensor) and value.is_cuda and any(
                                        allocator.graph_storage_overlaps(value, leaves[position]) for position in consumed):
                                    flat[index] = value.clone()
                            out = pytree.tree_unflatten(flat, out_spec)
                            value = None
                            del flat
                        # Returned views keep execution storage alive: capture them with the ordinary owner's version counter.
                        self._capture_lifetimes, self._capture_output_plan = consumed, plan
                        entry = self._capture(module, forward, leaves, spec, signature, extras, extra_signature, out, static_key, buffers, not retained_output, self.blas[blas_key])
                        if entry is None:  # a different allocation trace must not replay the proposed aliases
                            self._capture_output_plan = None
                            entry = self._capture(module, forward, leaves, spec, signature, extras, extra_signature, out, static_key, buffers, not retained_output, self.blas[blas_key])
                        self.entries[key] = entry
                    finally:
                        self.__dict__.pop('_capture_lifetimes', None)
                        self.__dict__.pop('_capture_output_plan', None)
                        plan.close()
            finally:
                _recording -= 1
                self._prune()
            keys[key] = None
            return self._escape(out)
        keys[key] = keys.pop(key)  # most recently used
        if (self.state_ids and not kwargs and not extras and not entry.lifetimes and not entry.emptied and len(entry.static_key) == len(entry.static)
                and len(leaves) == len(args) and all(leaf is arg and (not isinstance(arg, torch.Tensor) or id(arg) in self.state_ids) for leaf, arg in zip(leaves, args))):
            result = self._replay(entry, leaves, extras)
            # the replay of these arguments, as _replay does it: the copies of the inputs that are not their static buffers, the static
            # buffers to record on the stream, the arguments updated in place, and the outputs (arguments or constants: no output buffer)
            copies = [(static, leaves[index]) for index, static in zip(entry.positions, entry.static) if not _same_buffer(static, leaves[index])]
            borrowed = [self.static[k] for k in set(entry.static_key) if k in self.borrowed]
            mutated = [(leaves[entry.positions[index]], entry.static[index]) for index in entry.mutated]
            self.memo[(id(module), id(args[0]) if args else None)] = (method, args, tuple(arg.data_ptr() if isinstance(arg, torch.Tensor) else None for arg in args), key[4], key[5], entry,
                                     (copies, borrowed, mutated, result, [torch.cuda.current_stream()]))
            return result
        result = self._replay(entry, leaves, extras)
        if entry.emptied:
            handoffs = _handoffs(args, kwargs)
            for index in entry.emptied:
                container = handoffs.get(id(leaves[index]))
                if container is not None:
                    container.clear()
        return result

    def _move(self, key, leaf):
        # The caller's tensor takes the storage of its static buffer, which then serves no other call while the caller holds it (borrowed).
        static = self.static[key]
        static.copy_(leaf)
        leaf.record_stream(torch.cuda.current_stream())
        leaf.set_(static.untyped_storage(), static.storage_offset(), static.shape, static.stride())
        self.borrowed.add(key)

    def _free_static(self, leaf):
        # the static buffer that a block call would give this tensor (first of its signature), when no caller holds it
        for key in self.static_by_base.get((_signature((leaf,))[0], 0), ()):
            if (key not in self.borrowed or _unshared(self.static[key])) and _movable(leaf, self.static[key]):
                return key
        return None

    def pooled(self, fn, args, kwargs):
        from . import allocator
        if self.pool is None:  # a stage before the first captures: its temporaries give the pool its first memory, which they would keep from it
            self.pool = torch.cuda.graph_pool_handle()
        live = allocator.pool_live(self.pool)
        with allocator.graph_warmup(self.pool):
            out = fn(*args, **kwargs)
        holders = {}
        for leaf in _flatten(out)[0]:
            if isinstance(leaf, torch.Tensor) and leaf.is_cuda and allocator.in_pool(leaf, self.pool):
                holders.setdefault(_storage(leaf), {})[id(leaf)] = leaf
        for tensors in holders.values():
            tensors = list(tensors.values())
            leaf = tensors[0]
            key = self._free_static(leaf) if len(tensors) == 1 else None
            if key is not None:
                if key in self.borrowed:
                    allocator.prepare_reuse(self.static[key])
                self._move(key, leaf)
                continue
            storage = leaf.untyped_storage()
            if torch._C._storage_Use_Count(storage._cdata) != len(tensors) + 1:
                raise RuntimeError(f"A pooled stage of '{self.name}' returned a tensor whose graph-pool storage is also held elsewhere (a view kept by the "
                                   f"stage, a cache): the next graph replay would overwrite it.")
            copy = torch.UntypedStorage(storage.nbytes(), device=leaf.device)
            copy.copy_(storage)
            for tensor in tensors:
                tensor.set_(copy, tensor.storage_offset(), tensor.shape, tensor.stride())
            del storage
        left = allocator.pool_live(self.pool) - live
        if left:
            raise RuntimeError(f"A pooled stage of '{self.name}' kept {left} allocation(s) in the graph pool beyond its outputs (a cache filled or a tensor "
                               f"stored elsewhere), which the next graph replay would overwrite: fill such caches outside the stage.")
        return out

    def _consumable(self, args, kwargs, leaves, positions, buffers):
        from . import allocator
        if allocator.active != "vmm":
            return ()
        handoffs = _handoffs(args, kwargs)
        eligible = []
        for ordinal, index in enumerate(positions):
            source, buffer = leaves[index], buffers[ordinal]
            if (id(source) not in handoffs or not source.is_cuda or source.numel() * source.element_size() < 32 * 1024 * 1024 or
                    not _same_buffer(source, buffer) or sum(value is source for value in leaves) != 1):
                continue
            # Account only for MMGP's references, this call's flattened arguments/buffers, and the transferring list.
            # Any additional Python owner or storage/view owner keeps the input protected.
            expected = self._tensor_refs.get(id(source), 0) + 4 + sum(value is source for value in buffers) + int(source is buffer)
            if sys.getrefcount(source) != expected or source._use_count() != 1:
                continue
            storage = source.untyped_storage()
            if (source.storage_offset() != 0 or storage.nbytes() != source.numel() * source.element_size() or
                    torch._C._storage_Use_Count(storage._cdata) != 2 + int(source is not buffer) or sys.getrefcount(storage) != 3):
                continue
            if not allocator.lifetime_available(source):
                continue
            eligible.append(index)
        return tuple(eligible)

    def _outputs(self, leaves, out):
        # per output leaf: ("argument", index of the argument it is), its signature, or None if it is not a tensor
        by_pointer = {leaf.data_ptr(): (index, leaf) for index, leaf in enumerate(leaves) if isinstance(leaf, torch.Tensor)}
        outputs = []
        for leaf in _flatten(out)[0]:  # an output that is an argument (its storage, shape and strides) is returned as the caller's tensor
            if isinstance(leaf, torch.Tensor):
                index, argument = by_pointer.get(leaf.data_ptr(), (None, None))
                same = argument is not None and leaf.dtype == argument.dtype and leaf.shape == argument.shape and leaf.stride() == argument.stride()
                outputs.append(("argument", index) if same else _signature((leaf,))[0])
            else:
                outputs.append(None)
        return outputs

    def _evict(self, key, keep=()):
        del self.keys[key[0]][key]
        entry = self.entries.pop(key)
        self._forget(entry)
        self._prune(keep)

    def _forget(self, *entries):
        # the fast path of call() replays only graphs still in entries
        for memo_key in [memo_key for memo_key, memo in self.memo.items() if memo[5] in entries]:
            del self.memo[memo_key]

    def _escape(self, out):
        """The result of a key's first call (eager, in the graph pool) leaves the pool: the scratch of the graphs captured before may lie
        there, which their replays overwrite while the caller still holds it (the output of a block kept while the block runs for another
        guidance branch). Its tensors in the pool that are no static buffer are copied to ordinary memory, once per capture."""
        from . import allocator
        if self.pool is None or allocator.active != "vmm" or not ESCAPE_WARMUP:
            return out
        flat, spec = _flatten(out)
        escaped = [leaf.clone() if isinstance(leaf, torch.Tensor) and leaf.is_cuda and _static_storages.get(_storage(leaf)) is None and allocator.in_pool(leaf, self.pool)
                   else leaf for leaf in flat]
        return pytree.tree_unflatten(escaped, spec)

    def _held_later(self, entry):
        """Whether a caller still holds an output buffer that a capture after this graph's put in the pool: its replay writes its scratch,
        which may lie over it (the output of a block kept by the caller while the block runs for another guidance branch, whose graph an
        earlier pass captured). The graph is then captured again, its scratch avoiding the buffers alive (no copy of them)."""
        return RECAPTURE_HELD and bool(self.later) and any(seq > entry.seq and not _unshared(self.static[key]) for key, seq in self.later.items())

    def forget_storage(self, tensors):
        """The graphs whose static buffers adopted these tensors' storage go, with those buffers: their owner drops them (a cache leaving the
        VRAM) and gets the memory back. The blocks capture their graphs again at their next calls."""
        pointers = {tensor.untyped_storage().data_ptr() for tensor in tensors}
        keys = {key for key, buffer in self.static.items() if buffer.untyped_storage().data_ptr() in pointers}
        if keys:
            for key in [key for key, entry in self.entries.items() if not keys.isdisjoint(entry.static_key)]:
                self._evict(key)
            self._prune()

    def _prune(self, keep=()):
        used = set(keep) | {k for entry in self.entries.values() for k in entry.static_key}
        if any(k not in used for k in self.static):
            self.static = {k: buffer for k, buffer in self.static.items() if k in used}
            self.later = {k: seq for k, seq in self.later.items() if k in used}
            self.static_by_buffer, self.static_by_base = {}, {}
            for key, buffer in self.static.items():
                self._index_buffer(key, buffer)
        self.borrowed.intersection_update(used)
        refs = {}
        for buffer in self.static.values():
            refs[id(buffer)] = refs.get(id(buffer), 0) + 1
        for entry in self.entries.values():
            for buffer in (*entry.static, *entry.extra_static, *entry.outputs):
                if isinstance(buffer, torch.Tensor):
                    refs[id(buffer)] = refs.get(id(buffer), 0) + 1
        self._tensor_refs = refs

    def _index_buffer(self, key, buffer):
        self.static_by_buffer.setdefault((key[0], buffer.data_ptr()), []).append(key)
        self.static_by_base.setdefault(key[:2], []).append(key)

    def _buffers(self, signatures, sources, counts=None, arguments=False):
        # static buffers, shared by the graphs (they run one after the other) for each signature and occurrence in a call; not inference
        # tensors, whose version counters show the in-place updates of the arguments. An argument whose elements share memory (an expanded
        # view) cannot be copied into a buffer of its layout: arguments=True reports it (checked only when a buffer is created).
        counts, keys, buffers = {} if counts is None else counts, [], []
        for signature, source in zip(signatures, sources):
            base = (signature, counts.setdefault(signature, 0))
            counts[signature] += 1
            # Prefix tensors can keep hundreds of adopted buffers alive. A replay usually names an existing buffer exactly;
            # look it up directly instead of scanning every other layer's buffers for every argument.
            matching = None if source is None else next((k for k in self.static_by_buffer.get((signature, source.data_ptr()), ())
                                                        if _same_buffer(self.static[k], source)), None)
            if matching is not None:
                key = matching
            else:
                candidates = [k for k in self.static_by_base.get(base, ()) if k not in keys]
                key = next((k for k in candidates if k not in self.borrowed or _unshared(self.static[k])), None)
            if key in self.borrowed and matching is None:
                from . import allocator
                allocator.prepare_reuse(self.static[key])
            if key is None:
                key = (*base, 1 + max((k[2] for k in candidates), default=-1))
                shape, stride, dtype, device = signature
                if arguments and source is not None and torch._debug_has_internal_overlap(source) == 1:
                    raise _OverlappingArgument(len(keys))
                if source is not None and _adoptable(source, self):
                    self.static[key] = _buffer_alias(source)
                    self.borrowed.add(key)
                else:
                    with torch.inference_mode(False):
                        self.static[key] = torch.empty_strided(shape, stride, dtype=dtype, device=device)
                self.static[key]._mm_graphs = weakref.ref(self)
                _static_storages[_storage(self.static[key])] = self.static[key]
                self._index_buffer(key, self.static[key])
                buffer_id = id(self.static[key])
                self._tensor_refs[buffer_id] = self._tensor_refs.get(buffer_id, 0) + 1
            keys.append(key)
            buffers.append(self.static[key])
        return keys, buffers

    def _capture(self, module, forward, leaves, spec, signature, extras, extra_signature, warm_output, static_key, buffers, adopt_outputs, blas=False):
        # blas: the call uses cuBLAS, whose workspaces the capture stream then needs before the capture (they cannot be made during it)
        from . import allocator
        entry = _Graph()
        self.captures += 1
        entry.seq = self.captures
        outputs = self._outputs(leaves, warm_output)
        entry.positions = [index for index, leaf in enumerate(leaves) if isinstance(leaf, torch.Tensor)]
        out_signatures = [output for output in outputs if output is not None and output[0] != "argument"]
        counts = {}
        for sig in [signature[index] for index in entry.positions] + list(extra_signature):
            counts[sig] = counts.get(sig, 0) + 1
        out_sources = [leaf if adopt_outputs else None for leaf, output in zip(_flatten(warm_output)[0], outputs) if output is not None and output[0] != "argument"]
        known = set(self.static)
        output_keys, output_buffers = self._buffers(out_signatures, out_sources, counts)
        for key in output_keys:
            if key not in known and self.pool is not None and allocator.active == "vmm" and allocator.in_pool(self.static[key], self.pool):
                self.later[key] = entry.seq
        plan = self._capture_output_plan
        if plan is not None:
            for source, buffer in zip(out_sources, output_buffers):
                if source is not None and _same_buffer(source, buffer) and _adoptable(source, self):
                    plan.bind(buffer)
        entry.static_key = static_key + output_keys
        buffers = buffers + output_buffers
        inputs = len(entry.positions)
        entry.static, entry.extra_static = buffers[:inputs], buffers[inputs:inputs + len(extras)]
        out_buffers = iter(buffers[inputs + len(extras):])
        static_leaves = list(leaves)
        for index, static in zip(entry.positions, entry.static):  # a capture runs nothing: the buffers need no values
            static_leaves[index] = static
        args, kwargs = pytree.tree_unflatten(static_leaves, spec)
        versions = [static._version for static in entry.static]
        # a module whose calls read the same tensors whatever their shapes (_mm_graph_check_once, e.g. an LLM's decoding passes of any length)
        # has its reads checked at its first capture only
        checked_key = (id(module), getattr(forward, "__func__", forward), None if module.__dict__.get("_mm_graph_check_once", False) else signature)
        if checked_key in self.checked:
            check = contextlib.nullcontext()
        else:  # costly (each operation goes through Python): once per code path, which other weights capture again
            allowed = {_storage(t) for t in buffers} | {_storage(t) for t in _constants}
            allowed.update(_storage(t) for tensors in self.state.values() for t in tensors)
            allowed.update(_storage(t) for t in self.weight_tensors(module) if t.is_cuda)
            check = _ReadCheck(allowed)
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        entry.graph = torch.cuda.CUDAGraph()
        failed = f"CUDA graph capture of a block of '{self.name}' ({type(module).__name__}) failed"
        # torch.cuda.graph would synchronize the device and empty the allocator cache before each capture, which no capture needs
        originals = [holder[index] for holder, index in extras]
        for (holder, index), static in zip(extras, entry.extra_static):
            holder[index] = static
        if not _autotune_guarded:
            _guard_triton_autotune()
        global _capturing
        stream = _capture_stream(torch.cuda.current_device())
        if blas:
            _prepare_blas_stream(stream)
        with torch.cuda.stream(stream):
            entry.graph.capture_begin(self.pool, capture_error_mode="thread_local")
            _capturing = True
            if plan is not None:
                plan.capture()
            plan_valid = True
            try:
                with check:
                    handoffs = _handoffs(args, kwargs) if self._capture_lifetimes else {}
                    output, entry.lifetimes = _execute_lifetimes(forward, args, kwargs, self._capture_lifetimes, static_leaves)
                    entry.emptied = tuple(index for index in entry.lifetimes if not handoffs[id(static_leaves[index])])
                    out_leaves, entry.out_spec = _flatten(output)
                    del output
                    entry.outputs = []
                    for leaf, output in zip(out_leaves, outputs):  # the other outputs are copied to static buffers, which leaves the memory of
                        if output is None:  # the tensors of the block to the next captures
                            entry.outputs.append(None)
                        elif output[0] == "argument":
                            entry.outputs.append(entry.positions.index(output[1]))
                        else:
                            static = next(out_buffers)
                            if not allocator.same_graph_storage(leaf, static):
                                static.copy_(leaf)
                            entry.outputs.append(static)
                    entry.out_leaves = [None if isinstance(leaf, torch.Tensor) else leaf for leaf in out_leaves]
                    del out_leaves
                    leaf = None
            except BaseException:
                with contextlib.suppress(Exception):  # the stream leaves the capture, which the error has invalidated
                    entry.graph.capture_end()
                raise
            finally:
                if plan is not None:
                    plan_valid = plan.finish()
                _capturing = False
                for (holder, index), original in zip(extras, originals):
                    holder[index] = original
            try:
                entry.graph.capture_end()
            except Exception as error:
                raise RuntimeError(f"{failed}: {error}") from error
        if not plan_valid:
            return None
        if isinstance(check, _ReadCheck):
            if check.unknown:
                raise RuntimeError(f"CUDA graph capture of a block of '{self.name}' ({type(module).__name__}): it reads tensors that are neither "
                                   f"its weights nor its arguments, whose addresses may change between replays:\n" + "\n".join(check.unknown))
            self.checked.add(checked_key)
        entry.mutated = [index for index, (static, version) in enumerate(zip(entry.static, versions)) if static._version != version]
        if not self.reported:
            tqdm.write(f"[MMGP] CUDA graphs of the blocks of '{self.name}' are being captured.")
            self.reported = True
        return entry

    def _replay(self, entry, leaves, extras):
        from . import allocator
        for key in entry.static_key[len(entry.static) + len(entry.extra_static):]:
            if key in self.borrowed:
                allocator.prepare_reuse(self.static[key])
        for ordinal, (index, static) in enumerate(zip(entry.positions, entry.static)):
            if _same_buffer(static, leaves[index]):
                continue
            if _movable(leaves[index], static):
                self._move(entry.static_key[ordinal], leaves[index])
            else:
                static.copy_(leaves[index])
        if extras:
            torch._foreach_copy_(entry.extra_static, [holder[index] for holder, index in extras])
        entry.graph.replay()
        stream = torch.cuda.current_stream()
        for key in set(entry.static_key):
            if key in self.borrowed:
                self.static[key].record_stream(stream)
        for index in entry.mutated:
            if not _same_buffer(leaves[entry.positions[index]], entry.static[index]):
                leaves[entry.positions[index]].copy_(entry.static[index])
            else:
                torch.autograd.graph.increment_version(leaves[entry.positions[index]])
        output_keys = iter(entry.static_key[len(entry.static) + len(entry.extra_static):])
        out = [leaf if output is None else leaves[entry.positions[output]] if isinstance(output, int) else
               _buffer_alias(output) if next(output_keys) in self.borrowed else output.clone()
               for leaf, output in zip(entry.out_leaves, entry.outputs)]
        return pytree.tree_unflatten(out, entry.out_spec)
