# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""
Wan2GP Cloud Providers package.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Type

from shared.execution.providers.base import BaseCloudProvider
from shared.execution.providers.http_provider import HTTPCloudProvider
from shared.execution.providers.modal_provider import ModalCloudProvider
from shared.execution.providers.mock_provider import MockCloudProvider
from shared.execution.providers.runpod_provider import RunPodCloudProvider

_PROVIDERS: Dict[str, Type[BaseCloudProvider]] = {
    "http": HTTPCloudProvider,
    "modal": ModalCloudProvider,
    "runpod": RunPodCloudProvider,
    "mock": MockCloudProvider,
}


def get_cloud_provider(
    provider_name: Optional[str] = None,
    **kwargs: Any,
) -> BaseCloudProvider:
    """
    Get or resolve a Cloud Provider instance.

    If provider_name is 'auto' or None, automatically inspects environment variables:
    1. If WAN2GP_CLOUD_MOCK is set -> 'mock'
    2. If MODAL_ENDPOINT or MODAL_TOKEN_ID is set -> 'modal'
    3. If RUNPOD_ENDPOINT_ID or RUNPOD_API_KEY is set -> 'runpod'
    4. Default -> 'http'
    """
    name = (provider_name or os.environ.get("WAN2GP_CLOUD_PROVIDER", "auto")).strip().lower()

    if name in ("auto", ""):
        if os.environ.get("WAN2GP_CLOUD_MOCK", "").lower() in ("1", "true", "yes"):
            name = "mock"
        elif os.environ.get("MODAL_ENDPOINT") or os.environ.get("MODAL_TOKEN_ID"):
            name = "modal"
        elif os.environ.get("RUNPOD_ENDPOINT_ID") or os.environ.get("RUNPOD_API_KEY"):
            name = "runpod"
        else:
            name = "http"

    provider_cls = _PROVIDERS.get(name, HTTPCloudProvider)
    return provider_cls(**kwargs)


__all__ = [
    "BaseCloudProvider",
    "HTTPCloudProvider",
    "ModalCloudProvider",
    "RunPodCloudProvider",
    "MockCloudProvider",
    "get_cloud_provider",
]
