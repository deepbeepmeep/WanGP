# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""
Execution Adapter Registry and Dispatcher for Wan2GP.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Type, Union

from shared.execution.base import ExecutionAdapter
from shared.execution.cloud import CloudExecutionAdapter
from shared.execution.local import LocalExecutionAdapter
from shared.execution.types import ExecutionMode, ExecutionResult, GenerationRequest

_ADAPTER_REGISTRY: Dict[str, Type[ExecutionAdapter]] = {
    ExecutionMode.LOCAL.value: LocalExecutionAdapter,
    ExecutionMode.CLOUD.value: CloudExecutionAdapter,
}

_ADAPTER_INSTANCES: Dict[str, ExecutionAdapter] = {}


def register_adapter(mode_name: str, adapter_class: Type[ExecutionAdapter]) -> None:
    """Register a custom execution adapter class."""
    _ADAPTER_REGISTRY[mode_name.strip().lower()] = adapter_class
    _ADAPTER_INSTANCES.pop(mode_name.strip().lower(), None)


def get_execution_adapter(
    mode: Optional[Union[str, ExecutionMode]] = None,
    generation_fn: Optional[Callable[..., Any]] = None,
    **kwargs: Any,
) -> ExecutionAdapter:
    """
    Get or create an ExecutionAdapter instance for the specified mode.

    Args:
        mode: "local" or "cloud" (defaults to ExecutionMode.LOCAL).
        generation_fn: Optional custom generation function for LocalExecutionAdapter.
        kwargs: Configuration arguments passed to adapter constructors.
    """
    normalized_mode = ExecutionMode.from_value(mode).value

    # If custom options/callbacks provided, create a fresh instance
    if generation_fn is not None or kwargs:
        adapter_cls = _ADAPTER_REGISTRY.get(normalized_mode, LocalExecutionAdapter)
        if normalized_mode == ExecutionMode.LOCAL.value:
            return adapter_cls(generation_fn=generation_fn, **kwargs)
        return adapter_cls(**kwargs)

    if normalized_mode not in _ADAPTER_INSTANCES:
        adapter_cls = _ADAPTER_REGISTRY.get(normalized_mode, LocalExecutionAdapter)
        _ADAPTER_INSTANCES[normalized_mode] = adapter_cls()

    return _ADAPTER_INSTANCES[normalized_mode]


def dispatch_generation(
    task: Dict[str, Any],
    send_cmd: Callable[[str, Any], None],
    state: Any = None,
    filtered_params: Optional[Dict[str, Any]] = None,
    plugin_data: Optional[Dict[str, Any]] = None,
    local_generator_fn: Optional[Callable[..., Any]] = None,
    execution_mode: Optional[str] = None,
) -> ExecutionResult:
    """
    Unified entry point for executing a Wan2GP task.

    Normalizes the incoming task into a GenerationRequest, resolves the appropriate
    ExecutionAdapter (Local GPU vs Cloud GPU), and executes it with streaming event callbacks.

    Args:
        task: Wan2GP queue task dict containing 'params', 'id', etc.
        send_cmd: Event stream callback function (status, progress, preview, output, error).
        state: Gradio state dictionary.
        filtered_params: Validated parameters filtered against generate_media signature.
        plugin_data: Optional plugin dictionary.
        local_generator_fn: Callable for local execution (defaults to wgp.generate_media).
        execution_mode: Override for execution mode ('local' | 'cloud').

    Returns:
        ExecutionResult containing status, output files, metadata, and error details.
    """
    request = GenerationRequest.from_task(
        task=task,
        state=state,
        plugin_data=plugin_data,
        filtered_params=filtered_params,
        execution_mode=execution_mode,
    )

    adapter = get_execution_adapter(
        mode=request.execution_mode,
        generation_fn=local_generator_fn,
    )

    return adapter.execute(request, send_cmd=send_cmd)
