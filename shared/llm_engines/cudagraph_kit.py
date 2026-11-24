# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

import torch

from mmgp import cuda_graphs


@dataclass
class GraphKVIndexWriter:
    """Writes decode K/V values at the absolute token position provided by input_pos."""

    def write(self, k_cache: torch.Tensor, v_cache: torch.Tensor, k_val: torch.Tensor, v_val: torch.Tensor, input_pos: torch.Tensor) -> None:
        index = input_pos.reshape(-1)[:1]
        if index.dtype != torch.long:
            index = index.to(dtype=torch.long)
        if index.device != k_cache.device:
            index = index.to(device=k_cache.device)
        k_cache[: k_val.shape[0]].index_copy_(1, index, k_val)
        v_cache[: v_val.shape[0]].index_copy_(1, index, v_val)


@dataclass
class GraphKVFlashState:
    """Marks a cache as using flash_attn_with_kvcache-compatible graph decode."""

    flash_kvcache: bool = True


class CUDAGraphRunner:
    """Captures and replays a fixed-shape CUDA callable."""

    def __init__(self, name: str, fn: Callable[..., object]) -> None:
        self._name = name
        self._fn = fn
        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_inputs: list[torch.Tensor] = []
        self._output: object = None

    def _capture(self, *inputs: torch.Tensor) -> object:
        if not inputs:
            raise ValueError(f"[{self._name}] CUDAGraphRunner requires at least one tensor input.")
        for tensor in inputs:
            if not torch.is_tensor(tensor):
                raise TypeError(f"[{self._name}] CUDAGraphRunner accepts tensor inputs only.")
            if tensor.device.type != "cuda":
                raise ValueError(f"[{self._name}] CUDAGraphRunner requires CUDA tensors.")
        self._static_inputs = [tensor.detach().clone() for tensor in inputs]
        self._graph = cuda_graphs.Graph()
        try:
            with torch.inference_mode():
                with torch.cuda.graph(self._graph, capture_error_mode="thread_local"):
                    self._output = self._fn(*self._static_inputs)
            # The capture pass records kernels but does not populate outputs for immediate use.
            # Replay once so the first caller receives valid results.
            with torch.inference_mode():
                self._graph.replay()
        except Exception:
            self.release()
            raise
        return self._output

    def _copy_inputs(self, *inputs: torch.Tensor) -> None:
        if len(inputs) != len(self._static_inputs):
            raise ValueError(f"[{self._name}] Expected {len(self._static_inputs)} inputs, got {len(inputs)}.")
        for static_tensor, tensor in zip(self._static_inputs, inputs):
            if tensor.shape != static_tensor.shape or tensor.dtype != static_tensor.dtype or tensor.device != static_tensor.device:
                raise ValueError(
                    f"[{self._name}] Static shape mismatch: expected {tuple(static_tensor.shape)} {static_tensor.dtype} on {static_tensor.device}, "
                    f"got {tuple(tensor.shape)} {tensor.dtype} on {tensor.device}."
                )
            static_tensor.copy_(tensor)

    def run(self, *inputs: torch.Tensor) -> object:
        if self._graph is None:
            return self._capture(*inputs)
        self._copy_inputs(*inputs)
        with torch.inference_mode():
            self._graph.replay()
        return self._output

    def release(self) -> None:
        self._graph = None
        self._static_inputs = []
        self._output = None


def _cuda_tensors(value, depth: int = 0) -> list[torch.Tensor]:
    """The plain CUDA tensors held by a cache or state object (in lists, tuples, dicts or object attributes)."""
    if type(value) is torch.Tensor:
        return [value] if value.is_cuda else []
    if depth > 3 or isinstance(value, torch.nn.Module):
        return []
    if isinstance(value, (list, tuple)):
        items = value
    elif isinstance(value, dict):
        items = value.values()
    elif hasattr(value, "__dict__"):
        items = vars(value).values()
    else:
        return []
    return [tensor for item in items for tensor in _cuda_tensors(item, depth + 1)]


class _Stage:
    """A decoding step of the kit as an MMGP graph stage of its owner's model. MMGP keys a stage's graphs by the module (the owner, or the
    part of it that the steps run: the weights the step reads) and the method: one function per runner name, kept as an attribute of that
    module while the kit lives."""

    def __init__(self, owner, attribute: str) -> None:
        self.fn: Callable[..., object] | None = None
        self.attribute = attribute

        def step(*inputs: torch.Tensor) -> object:  # its own function object: the method identity of this step's graphs
            return self.fn(*inputs)
        owner.__dict__[attribute] = step


class AutoRegressiveCudaGraphKit:
    """Reusable helper to attach graph-safe KV writers and replay decode callables.

    With an owner that MMGP manages with CUDA graphs (its handler lists it in cudaGraphs), a decoding step is an MMGP graph stage: one
    graph while the owner's weights are all in VRAM, otherwise the step runs with each layer replayed from its own MMGP graph, whether
    the layer is kept in VRAM or streamed. The caches the steps read go through register_state. Other owners keep the kit's own graphs,
    which need the weights in place. resident: the module the steps run when it is only a part of the owner (e.g. its language model):
    its weights alone key the graphs (the owner's other blocks may stream meanwhile) and its blocks in VRAM are enough for a single graph
    per step."""

    def __init__(self, name: str, owner=None, resident=None) -> None:
        self._name = name
        self._owner = owner
        self._resident = owner if resident is None else resident
        if owner is not None:  # only MMGP moves its weights: the weights key of a whole-step graph is checked in constant time
            self._resident._mm_graph_placement_key = True
        self._runners: dict[str, CUDAGraphRunner] = {}
        self._stages: dict[str, _Stage] = {}  # MMGP owners: one graph stage per runner name
        self._state_registered = False
        self._attached_caches: list[object] = []
        self._attached_cache_ids: set[int] = set()

    def register_state(self, *holders: object) -> None:
        """The CUDA tensors (KV caches, static buffers) that the steps read or update besides the owner's weights and the step inputs,
        at fixed addresses until release(): MMGP owners only (a no-op otherwise). Registering new tensors drops the graphs. The cache
        objects among the holders (e.g. a transformers StaticCache) may be passed to the owner's layers."""
        if self._owner is None:
            return
        from mmgp import cuda_graphs
        tensors = [tensor for holder in holders for tensor in _cuda_tensors(holder)]
        cuda_graphs.register_state(self._owner, self._name, tensors, holders=[holder for holder in holders if not isinstance(holder, (torch.Tensor, list, tuple, dict))])
        self._state_registered = True

    def attach_attention_modules(self, attention_modules: Iterable[object], max_tokens: int, graph_mode: str = "index") -> int:
        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0.")
        if graph_mode not in ("index", "flash_kvcache"):
            raise ValueError(f"Unsupported graph_mode '{graph_mode}'.")
        final_capacity = 0
        for module in attention_modules:
            kv_cache = getattr(module, "kv_cache", None)
            if kv_cache is None:
                raise RuntimeError("KV cache must be initialized before CUDA graph attachment.")
            kv_cache.ensure_capacity(max_tokens)
            final_capacity = max(final_capacity, int(kv_cache.capacity))
            cache_id = id(kv_cache)
            if cache_id not in self._attached_cache_ids:
                if graph_mode == "flash_kvcache":
                    kv_cache.graph_state = GraphKVFlashState()
                else:
                    kv_cache.graph_state = GraphKVIndexWriter()
                self._attached_cache_ids.add(cache_id)
                self._attached_caches.append(kv_cache)
        return final_capacity

    def run(self, runner_name: str, fn: Callable[..., object], *inputs: torch.Tensor) -> object:
        if self._owner is not None:
            from mmgp import cuda_graphs, runtime
            with cuda_graphs.scope(gated=False):
                if cuda_graphs.is_enabled(self._owner):
                    stage = self._stages.get(runner_name)
                    if stage is None:
                        stage = self._stages[runner_name] = _Stage(self._resident, f"_mm_kit_{self._name}_{runner_name}")
                    stage.fn = fn  # the eager calls (first call, capture, streamed layers) run the caller's current function
                    return cuda_graphs.run(self._resident, *inputs, method=stage.attribute, enabled=runtime.weights_resident(self._resident))
            if runtime.requires_layer_offload(self._owner):
                # Keep the caller's KV cache and sequence state while MMGP streams layers.
                for runner in self._runners.values():
                    runner.release()
                self._runners.clear()
                return fn(*inputs)
        runner = self._runners.get(runner_name)
        if runner is None:
            runner = CUDAGraphRunner(f"{self._name}:{runner_name}", fn)
            self._runners[runner_name] = runner
        return runner.run(*inputs)

    def release(self) -> None:
        for runner in self._runners.values():
            runner.release()
        self._runners.clear()
        for stage in self._stages.values():
            self._resident.__dict__.pop(stage.attribute, None)
            stage.fn = None  # MMGP may keep the step function as a key (per-method caches): its caller's state must not stay reachable
        self._stages.clear()
        if self._state_registered:
            from mmgp import cuda_graphs
            cuda_graphs.release_state(self._owner, self._name)  # drops the owner's graphs, which read the released caches
            self._state_registered = False
        for kv_cache in self._attached_caches:
            if hasattr(kv_cache, "graph_state"):
                kv_cache.graph_state = None
        self._attached_caches.clear()
        self._attached_cache_ids.clear()
