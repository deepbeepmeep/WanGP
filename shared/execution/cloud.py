# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""
Cloud Execution Adapter for Wan2GP.

Coordinates generation requests with Cloud Execution Providers (Modal, RunPod, HTTP REST, Mock).
"""
from __future__ import annotations

import os
from typing import Any, Callable, Optional, Union

from shared.execution.base import ExecutionAdapter
from shared.execution.providers import (
    BaseCloudProvider,
    get_cloud_provider,
)
from shared.execution.types import (
    CloudConfigurationError,
    ExecutionMode,
    ExecutionResult,
    GenerationRequest,
)


class CloudExecutionAdapter(ExecutionAdapter):
    """
    Adapter that executes GenerationRequests on a remote Cloud Runtime via a configured provider.

    Supported Providers:
    - modal: Executes on Modal (via web endpoint or modal SDK)
    - runpod: Executes on RunPod Serverless endpoints
    - http: Executes on any custom REST API implementing /v1/jobs
    - mock: Executes simulated cloud generations for tests/development
    """

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        api_key: Optional[str] = None,
        provider: Optional[Union[str, BaseCloudProvider]] = None,
        timeout: Optional[float] = None,
        poll_interval: float = 1.0,
        mock_mode: Optional[bool] = None,
    ):
        if timeout is None:
            timeout = float(os.environ.get("WAN2GP_CLOUD_TIMEOUT", "3600.0"))
        if isinstance(provider, BaseCloudProvider):
            self.provider = provider
        else:
            resolved_provider = provider
            if mock_mode or os.environ.get("WAN2GP_CLOUD_MOCK", "").lower() in ("1", "true", "yes"):
                resolved_provider = "mock"
            self.provider = get_cloud_provider(
                provider_name=resolved_provider,
                endpoint_url=endpoint_url,
                api_key=api_key,
                timeout=timeout,
                poll_interval=poll_interval,
            )

    @property
    def name(self) -> str:
        return f"Cloud GPU Adapter ({self.provider.name})"

    @property
    def mode(self) -> ExecutionMode:
        return ExecutionMode.CLOUD

    def is_configured(self) -> bool:
        return self.provider.is_configured()

    def execute(
        self,
        request: GenerationRequest,
        send_cmd: Optional[Callable[[str, Any], None]] = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        """Execute request using the selected cloud provider."""
        provider = self.provider

        # If adapter is configured in mock mode, always preserve mock provider
        if provider.name != "mock":
            explicit_provider = request.metadata.get("provider") or request.settings.get("provider")
            if (
                explicit_provider
                and str(explicit_provider).strip().lower() not in ("auto", "", "local")
                and str(explicit_provider).strip().lower() != provider.name.lower()
            ):
                provider = get_cloud_provider(provider_name=str(explicit_provider).strip().lower())

        return provider.execute(request, send_cmd=send_cmd, **kwargs)

    def cancel(self, task_id: Optional[Any] = None) -> bool:
        return self.provider.cancel(task_id)
