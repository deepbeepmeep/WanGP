# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""
Wan2GP Execution Architecture Package.

Provides a clean execution abstraction separating generation configuration
from the execution runtime (Local GPU vs Cloud GPU).
"""
import os
from pathlib import Path

def _load_local_env():
    env_file = Path(__file__).resolve().parent.parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("\"'")
                if k and k not in os.environ:
                    os.environ[k] = v

_load_local_env()

from shared.execution.base import ExecutionAdapter
from shared.execution.cloud import CloudExecutionAdapter
from shared.execution.local import LocalExecutionAdapter
from shared.execution.registry import (
    dispatch_generation,
    get_execution_adapter,
    register_adapter,
)
from shared.execution.packaging import package_generation_request_to_zip
from shared.execution.types import (
    ATTACHMENT_KEYS,
    CloudConfigurationError,
    ExecutionMode,
    ExecutionResult,
    GenerationRequest,
)

__all__ = [
    "ExecutionMode",
    "GenerationRequest",
    "ExecutionResult",
    "ExecutionAdapter",
    "LocalExecutionAdapter",
    "CloudExecutionAdapter",
    "get_execution_adapter",
    "dispatch_generation",
    "register_adapter",
    "CloudConfigurationError",
    "ATTACHMENT_KEYS",
    "package_generation_request_to_zip",
]
