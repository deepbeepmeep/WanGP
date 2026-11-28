# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""
RunPod Serverless Cloud Provider for Wan2GP.

Dispatches generation requests to RunPod Serverless endpoints using the standard
RunPod REST API (or the runpod Python SDK if installed).
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, Optional

from shared.execution.providers.base import BaseCloudProvider
from shared.execution.types import ExecutionResult, GenerationRequest


class RunPodCloudProvider(BaseCloudProvider):
    """
    Executes generation on RunPod Serverless (https://runpod.io).

    Endpoints:
    - Submit: POST https://api.runpod.ai/v2/{endpoint_id}/run
    - Poll: GET https://api.runpod.ai/v2/{endpoint_id}/status/{job_id}
    """

    def __init__(
        self,
        endpoint_id: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 600.0,
        poll_interval: float = 2.0,
    ):
        self.endpoint_id = endpoint_id or os.environ.get("RUNPOD_ENDPOINT_ID", "")
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY", "")
        self.timeout = float(os.environ.get("WAN2GP_CLOUD_TIMEOUT", str(timeout)))
        self.poll_interval = poll_interval

    @property
    def name(self) -> str:
        return "runpod"

    def is_configured(self) -> bool:
        return bool(self.endpoint_id and self.api_key)

    def get_configuration_help(self) -> str:
        return (
            "RunPod Serverless Provider configuration:\n"
            "- Set RUNPOD_API_KEY (export RUNPOD_API_KEY=your_key)\n"
            "- Set RUNPOD_ENDPOINT_ID (export RUNPOD_ENDPOINT_ID=your_endpoint_id)"
        )

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "Wan2GP-RunPod/1.0",
        }

    def execute(
        self,
        request: GenerationRequest,
        send_cmd: Optional[Callable[[str, Any], None]] = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        fn_send_cmd = send_cmd if send_cmd is not None else (lambda cmd, data=None: None)
        start_time = time.time()

        if not self.is_configured():
            msg = f"RunPod Serverless Provider is not configured.\n\n{self.get_configuration_help()}"
            fn_send_cmd("error", msg)
            return ExecutionResult(success=False, status="failed", task_id=request.task_id, error=msg)

        try:
            from shared.execution.packaging import package_generation_request_to_zip
            from pathlib import Path
            import base64

            fn_send_cmd("status", "RunPod [Beta]: Packaging generation bundle and media attachments...")
            zip_bytes = package_generation_request_to_zip(request)
            bundle_b64 = base64.b64encode(zip_bytes).decode("utf-8")

            fn_send_cmd("status", f"RunPod [Beta]: Submitting job to endpoint '{self.endpoint_id}'...")
            run_url = f"https://api.runpod.ai/v2/{self.endpoint_id}/run"
            payload_dict = {
                "input": {
                    "bundle_base64": bundle_b64,
                    "request": request.to_dict()
                }
            }
            payload = json.dumps(payload_dict).encode("utf-8")

            req = urllib.request.Request(run_url, data=payload, headers=self._get_headers(), method="POST")
            with urllib.request.urlopen(req, timeout=30.0) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))

            job_id = resp_data.get("id")
            if not job_id:
                raise RuntimeError(f"RunPod did not return a job ID: {resp_data}")

            fn_send_cmd("status", f"RunPod [Beta]: Job {job_id} queued on Serverless worker...")

            deadline = time.time() + self.timeout
            while time.time() < deadline:
                status_url = f"https://api.runpod.ai/v2/{self.endpoint_id}/status/{job_id}"
                status_req = urllib.request.Request(status_url, headers=self._get_headers(), method="GET")

                with urllib.request.urlopen(status_req, timeout=30.0) as resp:
                    status_data = json.loads(resp.read().decode("utf-8"))

                runpod_status = status_data.get("status", "IN_QUEUE")
                fn_send_cmd("status", f"RunPod [Beta]: Job status is {runpod_status}...")

                # Surface progress message if provided by worker
                progress_msg = status_data.get("progress")
                if progress_msg:
                    fn_send_cmd("status", f"RunPod [Beta] GPU: {str(progress_msg)[:120]}")

                if runpod_status == "IN_PROGRESS":
                    fn_send_cmd("progress", [(1, 2), "RunPod [Beta]: Worker generating on cloud GPU..."])

                elif runpod_status == "COMPLETED":
                    output_data = status_data.get("output", {})
                    raw_files = output_data.get("output_files", []) if isinstance(output_data, dict) else [str(output_data)]
                    
                    # Save files into local output folder
                    save_dir_setting = os.environ.get("WAN2GP_OUTPUT_DIR") or (request.settings.get("save_path") if hasattr(request, "settings") and isinstance(request.settings, dict) else None) or "outputs"
                    out_dir = Path(save_dir_setting)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    downloaded_paths = []

                    for f_entry in raw_files:
                        if isinstance(f_entry, dict) and "data_b64" in f_entry:
                            fname = f_entry.get("filename", "output.mp4")
                            dest_path = out_dir / fname
                            dest_path.write_bytes(base64.b64decode(f_entry["data_b64"]))
                            downloaded_paths.append(str(dest_path.resolve()))
                        elif isinstance(f_entry, str):
                            downloaded_paths.append(f_entry)

                    fn_send_cmd("status", "RunPod: Job completed successfully")
                    fn_send_cmd("output", None)
                    return ExecutionResult(
                        success=True,
                        status="completed",
                        task_id=request.task_id or job_id,
                        output_files=downloaded_paths or list(raw_files),
                        metadata={"provider": "runpod", "endpoint_id": self.endpoint_id, "job_id": job_id},
                        execution_time=time.time() - start_time,
                    )

                elif runpod_status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                    err = status_data.get("error", f"RunPod job terminated with status: {runpod_status}")
                    fn_send_cmd("error", f"RunPod error: {err}")
                    return ExecutionResult(
                        success=False,
                        status="failed",
                        task_id=request.task_id or job_id,
                        error=str(err),
                        execution_time=time.time() - start_time,
                    )

                time.sleep(self.poll_interval)

            timeout_err = f"RunPod generation timed out after {self.timeout}s"
            fn_send_cmd("error", timeout_err)
            return ExecutionResult(success=False, status="failed", task_id=request.task_id, error=timeout_err, execution_time=time.time() - start_time)

        except urllib.error.URLError as exc:
            err_msg = f"RunPod connection error: {exc.reason}"
            fn_send_cmd("error", err_msg)
            return ExecutionResult(success=False, status="failed", task_id=request.task_id, error=err_msg, execution_time=time.time() - start_time)
        except Exception as exc:
            err_msg = f"RunPod error: {exc}"
            fn_send_cmd("error", err_msg)
            return ExecutionResult(success=False, status="failed", task_id=request.task_id, error=err_msg, execution_time=time.time() - start_time)
