# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""
Generic HTTP REST Cloud Provider for Wan2GP.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, List, Optional

from shared.execution.providers.base import BaseCloudProvider
from shared.execution.types import ExecutionResult, GenerationRequest


class HTTPCloudProvider(BaseCloudProvider):
    """
    Dispatches generation requests to any standard REST endpoint implementing /v1/jobs.
    """

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 300.0,
        poll_interval: float = 1.0,
    ):
        self.endpoint_url = (endpoint_url or os.environ.get("WAN2GP_CLOUD_ENDPOINT", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("WAN2GP_CLOUD_API_KEY", "")
        self.timeout = float(os.environ.get("WAN2GP_CLOUD_TIMEOUT", str(timeout)))
        self.poll_interval = poll_interval

    @property
    def name(self) -> str:
        return "http"

    def is_configured(self) -> bool:
        return bool(self.endpoint_url)

    def get_configuration_help(self) -> str:
        return (
            "HTTP Cloud Provider configuration:\n"
            "- Set WAN2GP_CLOUD_ENDPOINT (e.g. export WAN2GP_CLOUD_ENDPOINT=http://your-server:8000)\n"
            "- (Optional) Set WAN2GP_CLOUD_API_KEY for authorization"
        )

    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Wan2GP-CloudAdapter/1.0",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def execute(
        self,
        request: GenerationRequest,
        send_cmd: Optional[Callable[[str, Any], None]] = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        fn_send_cmd = send_cmd if send_cmd is not None else (lambda cmd, data=None: None)
        start_time = time.time()

        if not self.is_configured():
            msg = f"Cloud Runtime is not configured (HTTP Cloud Provider).\n{self.get_configuration_help()}"
            fn_send_cmd("error", msg)
            return ExecutionResult(success=False, status="failed", task_id=request.task_id, error=msg)

        try:
            fn_send_cmd("status", f"Cloud (HTTP): Submitting request to {self.endpoint_url}...")
            url = f"{self.endpoint_url}/v1/jobs"
            payload = json.dumps(request.to_dict()).encode("utf-8")
            req = urllib.request.Request(url, data=payload, headers=self._get_headers(), method="POST")

            with urllib.request.urlopen(req, timeout=30.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            job_id = data.get("job_id") or data.get("id")
            if not job_id:
                raise RuntimeError(f"HTTP runtime did not return a job_id: {data}")

            fn_send_cmd("status", f"Cloud (HTTP): Job {job_id} submitted. Monitoring...")

            deadline = time.time() + self.timeout
            while time.time() < deadline:
                poll_url = f"{self.endpoint_url}/v1/jobs/{urllib.parse.quote(str(job_id))}"
                poll_req = urllib.request.Request(poll_url, headers=self._get_headers(), method="GET")

                with urllib.request.urlopen(poll_req, timeout=30.0) as resp:
                    status_data = json.loads(resp.read().decode("utf-8"))

                state = str(status_data.get("status", "running")).lower()
                msg = status_data.get("status_message") or status_data.get("message")
                if msg:
                    fn_send_cmd("status", f"Cloud (HTTP): {msg}")

                prog = status_data.get("progress")
                if prog and isinstance(prog, (list, tuple)) and len(prog) >= 2:
                    fn_send_cmd("progress", [prog, msg or "Generating in Cloud..."])

                if state == "completed":
                    outputs = status_data.get("output_files", []) or status_data.get("outputs", [])
                    fn_send_cmd("status", "Cloud (HTTP): Generation completed successfully")
                    fn_send_cmd("output", None)
                    return ExecutionResult(
                        success=True,
                        status="completed",
                        task_id=request.task_id or job_id,
                        output_files=list(outputs),
                        metadata={"provider": "http", "endpoint": self.endpoint_url, "job_id": job_id},
                        execution_time=time.time() - start_time,
                    )

                if state == "failed":
                    err = status_data.get("error", "Remote HTTP generation failure.")
                    fn_send_cmd("error", f"Cloud generation failed: {err}")
                    return ExecutionResult(
                        success=False,
                        status="failed",
                        task_id=request.task_id or job_id,
                        error=str(err),
                        execution_time=time.time() - start_time,
                    )

                time.sleep(self.poll_interval)

            timeout_err = f"Cloud generation timed out after {self.timeout}s"
            fn_send_cmd("error", timeout_err)
            return ExecutionResult(success=False, status="failed", task_id=request.task_id, error=timeout_err, execution_time=time.time() - start_time)

        except urllib.error.URLError as exc:
            err_msg = f"HTTP Cloud Provider connection error: {exc.reason}"
            fn_send_cmd("error", err_msg)
            return ExecutionResult(success=False, status="failed", task_id=request.task_id, error=err_msg, execution_time=time.time() - start_time)
        except Exception as exc:
            err_msg = f"HTTP Cloud Provider error: {exc}"
            fn_send_cmd("error", err_msg)
            return ExecutionResult(success=False, status="failed", task_id=request.task_id, error=err_msg, execution_time=time.time() - start_time)
