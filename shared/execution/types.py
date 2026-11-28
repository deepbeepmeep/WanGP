# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""
Wan2GP Execution Types and Data Structures.

Defines the normalized GenerationRequest and ExecutionResult contracts
used across Local and Cloud execution adapters.
"""
from __future__ import annotations

import copy
import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


# Media attachment keys recognized by Wan2GP for inputs
ATTACHMENT_KEYS = [
    "image_start",
    "image_end",
    "image_refs",
    "image_guide",
    "image_mask",
    "video_guide",
    "video_guide2",
    "video_guide3",
    "video_mask",
    "video_source",
    "audio_guide",
    "audio_guide2",
    "audio_guide3",
    "audio_source",
    "replace_voice_sample",
    "replace_voice_sample2",
    "custom_guide",
]

# Keys that are ephemeral UI/runtime state and should not be serialized
EPHEMERAL_KEYS = {
    "state",
    "start_image_data",
    "end_image_data",
    "start_image_labels",
    "end_image_labels",
    "start_image_data_base64",
    "end_image_data_base64",
    "lset_name",
}


class ExecutionMode(str, Enum):
    """Execution targets supported by Wan2GP."""
    LOCAL = "local"
    CLOUD = "cloud"

    @classmethod
    def from_value(cls, val: Any) -> "ExecutionMode":
        if isinstance(val, cls):
            return val
        s = str(val or "").strip().lower()
        if "cloud" in s:
            return cls.CLOUD
        return cls.LOCAL


class CloudConfigurationError(RuntimeError):
    """Raised when Cloud execution is requested but the cloud runtime is unconfigured."""
    pass


def _parse_dimensions_from_resolution(resolution: Any) -> Tuple[Optional[int], Optional[int]]:
    """Extract width and height integers from resolution string like '1280x720 (16:9)'."""
    if not resolution or not isinstance(resolution, str):
        return None, None
    clean = resolution.strip().split(" ")[0]
    match = re.match(r"^(\d+)\s*[xX*]\s*(\d+)$", clean)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


@dataclass
class GenerationRequest:
    """
    Normalized, serializable representation of a Wan2GP generation task.

    Decouples WHAT should be generated from WHERE it runs (Local GPU vs Cloud GPU).
    Compatible with future external consumers such as WanStudio and Serverless Runtimes.
    """
    engine: str = "wan2gp"
    version: str = "1.0"
    task_id: Optional[Union[str, int]] = None
    execution_mode: str = "local"
    model: str = ""
    model_type: str = ""
    base_model_type: str = ""
    settings: Dict[str, Any] = field(default_factory=dict)
    inputs: Dict[str, Any] = field(default_factory=dict)
    plugin_data: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    _runtime_state: Any = field(default=None, repr=False, compare=False)
    _raw_task: Optional[Dict[str, Any]] = field(default=None, repr=False, compare=False)
    _raw_params: Optional[Dict[str, Any]] = field(default=None, repr=False, compare=False)
    _raw_plugin_data: Optional[Dict[str, Any]] = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        # Normalize execution mode
        self.execution_mode = ExecutionMode.from_value(self.execution_mode).value

        # Infer width and height into settings if resolution is available
        res = self.settings.get("resolution")
        if res and ("width" not in self.settings or "height" not in self.settings):
            w, h = _parse_dimensions_from_resolution(res)
            if w is not None and h is not None:
                self.settings.setdefault("width", w)
                self.settings.setdefault("height", h)

        # Set default creation timestamp if missing
        if "created_at" not in self.metadata:
            self.metadata["created_at"] = datetime.now().isoformat()

    # --- Convenience Accessors for Common Settings ---

    @property
    def prompt(self) -> str:
        return str(self.settings.get("prompt", "") or "")

    @property
    def negative_prompt(self) -> str:
        return str(self.settings.get("negative_prompt", "") or "")

    @property
    def resolution(self) -> str:
        return str(self.settings.get("resolution", "") or "")

    @property
    def width(self) -> Optional[int]:
        return self.settings.get("width")

    @property
    def height(self) -> Optional[int]:
        return self.settings.get("height")

    @property
    def frames(self) -> Optional[int]:
        return self.settings.get("video_length") or self.settings.get("frames")

    @property
    def fps(self) -> Optional[Union[int, float]]:
        return self.settings.get("force_fps") or self.settings.get("fps")

    @property
    def steps(self) -> Optional[int]:
        return self.settings.get("num_inference_steps") or self.settings.get("steps")

    @property
    def guidance(self) -> Optional[float]:
        return self.settings.get("guidance_scale") or self.settings.get("guidance")

    @property
    def seed(self) -> Optional[int]:
        return self.settings.get("seed")

    @property
    def loras(self) -> List[Any]:
        return self.settings.get("activated_loras", [])

    @property
    def loras_multipliers(self) -> List[Any]:
        return self.settings.get("loras_multipliers", [])

    @property
    def provider(self) -> str:
        return str(
            self.metadata.get("provider")
            or self.settings.get("provider")
            or os.environ.get("WAN2GP_CLOUD_PROVIDER")
            or ""
        ).strip().lower()

    # --- Conversion and Serialization ---

    def to_params(self, include_runtime_state: bool = True) -> Dict[str, Any]:
        """
        Recombine settings and inputs back into a full Wan2GP params dictionary.
        Guarantees 100% fidelity with existing generation pipelines.
        """
        params = copy.deepcopy(self.settings)
        # Restore media attachments from inputs
        for k, v in self.inputs.items():
            params[k] = v

        if self.model_type and "model_type" not in params:
            params["model_type"] = self.model_type
        if self.base_model_type and "base_model_type" not in params:
            params["base_model_type"] = self.base_model_type

        params["execution_mode"] = self.execution_mode

        if include_runtime_state and self._runtime_state is not None:
            params["state"] = self._runtime_state

        return params

    def to_task(self) -> Dict[str, Any]:
        """Convert back to the Wan2GP queue task format."""
        params = self.to_params(include_runtime_state=True)
        return {
            "id": self.task_id,
            "params": params,
            "plugin_data": copy.deepcopy(self.plugin_data),
            "execution_mode": self.execution_mode,
            "prompt": self.prompt,
            "length": self.frames or 0,
            "steps": self.steps or 0,
            "repeats": self.settings.get("repeat_generation", 1),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Return a clean JSON-serializable dictionary representation."""
        def _clean_val(v: Any) -> Any:
            # Handle non-serializable objects gracefully
            if hasattr(v, "isoformat"):
                return v.isoformat()
            if hasattr(v, "tolist"):
                return v.tolist()
            if isinstance(v, (str, int, float, bool)) or v is None:
                return v
            if isinstance(v, list):
                return [_clean_val(item) for item in v]
            if isinstance(v, dict):
                return {str(k): _clean_val(sub_v) for k, sub_v in v.items()}
            # Fallback to string representation for complex objects
            return str(v)

        return {
            "engine": self.engine,
            "version": self.version,
            "task_id": self.task_id,
            "execution_mode": self.execution_mode,
            "model": self.model,
            "model_type": self.model_type,
            "base_model_type": self.base_model_type,
            "settings": _clean_val(self.settings),
            "inputs": _clean_val(self.inputs),
            "plugin_data": _clean_val(self.plugin_data),
            "metadata": _clean_val(self.metadata),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize request to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationRequest":
        """Reconstruct GenerationRequest from dictionary."""
        settings = dict(data.get("settings", {}))
        inputs = dict(data.get("inputs", {}))
        return cls(
            engine=str(data.get("engine", "wan2gp")),
            version=str(data.get("version", "1.0")),
            task_id=data.get("task_id"),
            execution_mode=str(data.get("execution_mode", "local")),
            model=str(data.get("model", "")),
            model_type=str(data.get("model_type", "") or settings.get("model_type", "")),
            base_model_type=str(data.get("base_model_type", "") or settings.get("base_model_type", "")),
            settings=settings,
            inputs=inputs,
            plugin_data=dict(data.get("plugin_data", {})),
            metadata=dict(data.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "GenerationRequest":
        """Reconstruct GenerationRequest from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_task(
        cls,
        task: Dict[str, Any],
        state: Any = None,
        plugin_data: Optional[Dict[str, Any]] = None,
        filtered_params: Optional[Dict[str, Any]] = None,
        execution_mode: Optional[str] = None,
    ) -> "GenerationRequest":
        """
        Build a normalized GenerationRequest from a Wan2GP task dict.
        Extracts media inputs vs settings and strips ephemeral UI state.
        """
        raw_params = filtered_params if filtered_params is not None else task.get("params", {})
        params_copy = dict(raw_params)

        # Resolve execution mode
        exec_mode = (
            execution_mode
            or task.get("execution_mode")
            or params_copy.get("execution_mode")
            or (state.get("execution_mode") if isinstance(state, dict) else None)
            or ("cloud" if os.environ.get("WAN2GP_MODE") == "cloud" else "local")
        )
        if os.environ.get("WAN2GP_MODE") == "cloud":
            exec_mode = "cloud"

        model_type = str(params_copy.get("model_type", "") or "")
        base_model_type = str(params_copy.get("base_model_type", "") or "")
        model_name = str(task.get("model", "") or model_type)

        # Separate inputs (media attachments) from settings
        inputs = {}
        for key in ATTACHMENT_KEYS:
            if key in params_copy:
                val = params_copy.pop(key)
                if val is not None:
                    inputs[key] = val

        # Clean ephemeral / runtime keys from settings
        for key in list(params_copy.keys()):
            if key in EPHEMERAL_KEYS or key.startswith("_"):
                params_copy.pop(key, None)

        # Extract dimensions if present
        res = params_copy.get("resolution")
        if res and ("width" not in params_copy or "height" not in params_copy):
            w, h = _parse_dimensions_from_resolution(res)
            if w is not None and h is not None:
                params_copy["width"] = w
                params_copy["height"] = h

        explicit_provider = (
            task.get("provider")
            or params_copy.pop("provider", None)
            or params_copy.pop("cloud_provider", None)
            or (state.get("cloud_provider") if isinstance(state, dict) else None)
            or os.environ.get("WAN2GP_CLOUD_PROVIDER")
        )

        resolved_plugin_data = plugin_data if plugin_data is not None else task.get("plugin_data", {})

        metadata = {
            "source": "wan2gp",
            "task_id": task.get("id"),
            "client_id": params_copy.get("client_id", ""),
            "created_at": datetime.now().isoformat(),
        }
        if explicit_provider:
            metadata["provider"] = str(explicit_provider).strip().lower()

        return cls(
            task_id=task.get("id"),
            execution_mode=exec_mode,
            model=model_name,
            model_type=model_type,
            base_model_type=base_model_type,
            settings=params_copy,
            inputs=inputs,
            plugin_data=copy.deepcopy(resolved_plugin_data or {}),
            metadata=metadata,
            _runtime_state=state,
            _raw_task=task,
            _raw_params=filtered_params if filtered_params is not None else task.get("params"),
            _raw_plugin_data=resolved_plugin_data,
        )


@dataclass
class ExecutionResult:
    """Normalized result returned from an ExecutionAdapter."""
    success: bool
    status: str = "completed"  # "completed" | "failed" | "aborted" | "skipped"
    task_id: Optional[Union[str, int]] = None
    output_files: List[str] = field(default_factory=list)
    job_id: Optional[str] = None
    media_type: str = "video"  # "video" | "image" | "audio"
    metadata: Dict[str, Any] = field(default_factory=dict)
    generation_info: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status,
            "task_id": self.task_id,
            "output_files": list(self.output_files),
            "media_type": self.media_type,
            "metadata": dict(self.metadata),
            "generation_info": dict(self.generation_info),
            "error": self.error,
            "execution_time": self.execution_time,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExecutionResult":
        return cls(
            success=bool(data.get("success", False)),
            status=str(data.get("status", "completed")),
            task_id=data.get("task_id"),
            output_files=list(data.get("output_files", []) or []),
            media_type=str(data.get("media_type", "video")),
            metadata=dict(data.get("metadata", {}) or {}),
            generation_info=dict(data.get("generation_info", {}) or {}),
            error=data.get("error"),
            execution_time=float(data.get("execution_time", 0.0)),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "ExecutionResult":
        return cls.from_dict(json.loads(json_str))
