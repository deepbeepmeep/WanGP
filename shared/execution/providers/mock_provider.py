# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""
Mock Cloud Provider for Wan2GP testing and development.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Optional

from shared.execution.providers.base import BaseCloudProvider
from shared.execution.types import ExecutionResult, GenerationRequest


class MockCloudProvider(BaseCloudProvider):
    """
    Simulates remote cloud execution for offline tests and development.
    Does not require network access, API keys, or GPU resources.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    @property
    def name(self) -> str:
        return "mock"

    def is_configured(self) -> bool:
        return True

    def get_configuration_help(self) -> str:
        return "Mock provider is always configured and ready for simulation."

    def execute(
        self,
        request: GenerationRequest,
        send_cmd: Optional[Callable[[str, Any], None]] = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        fn_send_cmd = send_cmd if send_cmd is not None else (lambda cmd, data=None: None)
        start_time = time.time()
        job_id = f"mock_{int(time.time() * 1000)}"

        fn_send_cmd("status", f"Cloud (Mock): Job {job_id} submitted to simulated cloud runtime...")
        time.sleep(0.05)

        fn_send_cmd("progress", [(1, 4), "Cloud (Mock): Provisioning cloud GPU worker..."])
        time.sleep(0.05)

        fn_send_cmd("progress", [(2, 4), "Cloud (Mock): Streaming model weights to VRAM..."])
        time.sleep(0.05)

        fn_send_cmd("progress", [(3, 4), f"Cloud (Mock): Generating '{request.prompt[:30]}...'"])
        time.sleep(0.05)

        fn_send_cmd("progress", [(4, 4), "Cloud (Mock): Encoding video frames..."])
        fn_send_cmd("status", "Cloud (Mock): Generation completed successfully")
        fn_send_cmd("output", None)

        exec_time = time.time() - start_time
        return ExecutionResult(
            success=True,
            status="completed",
            task_id=request.task_id,
            job_id=job_id,
            output_files=[f"/tmp/{job_id}.mp4"],
            execution_time=exec_time,
            metadata={"simulated": True, "provider": "mock"},
        )

    def cancel(self, task_id: Optional[Any] = None) -> bool:
        return True
