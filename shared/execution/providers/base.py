# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""
Base interface for Wan2GP Cloud Providers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

from shared.execution.types import ExecutionResult, GenerationRequest


class BaseCloudProvider(ABC):
    """
    Abstract interface for a Cloud Execution Provider (Modal, RunPod, HTTP, Mock).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name (e.g. 'modal', 'runpod', 'http', 'mock')."""
        pass

    @abstractmethod
    def is_configured(self) -> bool:
        """Check if provider credentials/endpoints are configured."""
        pass

    @abstractmethod
    def get_configuration_help(self) -> str:
        """Human-readable instructions on how to configure this provider."""
        pass

    @abstractmethod
    def execute(
        self,
        request: GenerationRequest,
        send_cmd: Optional[Callable[[str, Any], None]] = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute the generation request on the cloud provider."""
        pass

    def cancel(self, task_id: Optional[Any] = None) -> bool:
        """Cancel an ongoing generation task on the provider."""
        return False
