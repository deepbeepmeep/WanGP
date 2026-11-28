# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""
Base Execution Adapter interface for Wan2GP.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from shared.execution.types import ExecutionMode, ExecutionResult, GenerationRequest


class ExecutionAdapter(ABC):
    """
    Abstract interface for executing Wan2GP generation requests.

    Concrete implementations:
    - LocalExecutionAdapter: Executes on the local GPU via Wan2GP's existing pipelines.
    - CloudExecutionAdapter: Serializes and dispatches requests to a remote Cloud Runtime.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable adapter name."""
        pass

    @property
    @abstractmethod
    def mode(self) -> ExecutionMode:
        """Execution mode associated with this adapter."""
        pass

    @abstractmethod
    def execute(
        self,
        request: GenerationRequest,
        send_cmd: Optional[Callable[[str, Any], None]] = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        """
        Execute a normalized generation request.

        Args:
            request: The normalized GenerationRequest containing all settings and inputs.
            send_cmd: Callback used by Wan2GP UI and runners to stream status,
                      progress, previews, outputs, and errors.
                      Signature: `send_cmd(command_name: str, payload: Any)`

        Returns:
            ExecutionResult with success flag, status, outputs, and metadata.
        """
        pass

    def cancel(self, task_id: Optional[Any] = None) -> bool:
        """
        Attempt to cancel an ongoing generation task.
        Returns True if cancellation was acknowledged.
        """
        return False
