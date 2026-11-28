# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""
Local GPU Execution Adapter for Wan2GP.

Executes generation requests directly on the local machine using Wan2GP's
existing high-performance pipelines, memory offloading (mmgp), and attention kernels.
"""
from __future__ import annotations

import time
import traceback
from typing import Any, Callable, Dict, List, Optional

from shared.execution.base import ExecutionAdapter
from shared.execution.types import ExecutionMode, ExecutionResult, GenerationRequest


class LocalExecutionAdapter(ExecutionAdapter):
    """
    Executes generation requests on the local GPU.

    Delegates directly to Wan2GP's `generate_media(...)` pipeline, ensuring
    100% backward compatibility with upstream behavior, offloading, and Gradio events.
    """

    def __init__(self, generation_fn: Optional[Callable[..., Any]] = None):
        """
        Args:
            generation_fn: Callable implementing local generation (typically `wgp.generate_media`).
                           If omitted, imported lazily from `wgp`.
        """
        self._generation_fn = generation_fn

    @property
    def name(self) -> str:
        return "Local GPU Adapter"

    @property
    def mode(self) -> ExecutionMode:
        return ExecutionMode.LOCAL

    def _get_generation_fn(self) -> Callable[..., Any]:
        if self._generation_fn is not None:
            return self._generation_fn
        try:
            import wgp
            return wgp.generate_media
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(f"Could not resolve local generation pipeline: {exc}") from exc

    def execute(
        self,
        request: GenerationRequest,
        send_cmd: Optional[Callable[[str, Any], None]] = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        """
        Execute the normalized request locally.

        Args:
            request: The normalized GenerationRequest.
            send_cmd: Event streaming callback (status, progress, preview, output, error).
            kwargs: Extra parameters passed through if provided.
        """
        fn_send_cmd = send_cmd if send_cmd is not None else (lambda cmd, data=None: None)
        generation_fn = self._get_generation_fn()

        raw_task = getattr(request, "_raw_task", None)
        raw_params = getattr(request, "_raw_params", None)
        raw_plugin_data = getattr(request, "_raw_plugin_data", None)

        task = raw_task if raw_task is not None else request.to_task()
        params = dict(raw_params) if raw_params is not None else request.to_params(include_runtime_state=True)
        plugin_data = raw_plugin_data if raw_plugin_data is not None else dict(request.plugin_data)

        # Track baseline output count if gen_info is accessible
        state = request._runtime_state or params.get("state")
        initial_file_count = 0
        gen_info = None
        if isinstance(state, dict) and "gen" in state:
            gen_info = state["gen"]
            initial_file_count = len(gen_info.get("file_list", []))

        start_time = time.time()
        # Execute existing pipeline, ensuring required signature args are present and extra args filtered
        import inspect
        call_kwargs = {k: v for k, v in params.items() if k not in ("task", "send_cmd", "plugin_data")}
        try:
            sig = inspect.signature(generation_fn)
            params_list = list(sig.parameters.values())
            positional_skip = {
                param.name for param in params_list[:2]
                if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            }
            excluded = positional_skip | {"task", "send_cmd", "plugin_data"}
            call_kwargs = {k: v for k, v in call_kwargs.items() if k not in excluded}
            has_var_kw = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in params_list)
            if not has_var_kw:
                call_kwargs = {k: v for k, v in call_kwargs.items() if k in sig.parameters}
            for arg_name, param in sig.parameters.items():
                if arg_name not in excluded and param.kind in (
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    inspect.Parameter.KEYWORD_ONLY,
                ) and arg_name not in call_kwargs and param.default is inspect.Parameter.empty:
                    call_kwargs[arg_name] = None
        except (TypeError, ValueError):
            pass

        # Propagate exceptions (e.g., DownloadCancelled, gr.Error, pipeline errors) identically to direct generate_media()
        success = generation_fn(
            task,
            fn_send_cmd,
            plugin_data=plugin_data,
            **call_kwargs,
        )

        execution_time = time.time() - start_time
        is_successful = bool(success is True or success is None)

        # Discover output files
        output_files: List[str] = []
        if gen_info and "file_list" in gen_info:
            current_files = gen_info["file_list"]
            if len(current_files) > initial_file_count:
                output_files = list(current_files[initial_file_count:])

        status = "completed" if is_successful else "failed"
        if gen_info and gen_info.get("abort", False):
            status = "aborted"

        return ExecutionResult(
            success=is_successful,
            status=status,
            task_id=request.task_id,
            output_files=output_files,
            metadata={
                "adapter": self.name,
                "execution_mode": self.mode.value,
                "model_type": request.model_type,
            },
            generation_info={"execution_time_seconds": execution_time},
            execution_time=execution_time,
        )
