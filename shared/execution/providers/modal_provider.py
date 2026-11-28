# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""
Modal Cloud Provider for Wan2GP.

Dispatches generation requests to a remote Modal GPU worker.
Supports both Modal Web Endpoints (HTTP webhook) and direct Modal Python SDK function invocation.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, Optional

from shared.execution.packaging import package_generation_request_to_zip
from shared.execution.providers.base import BaseCloudProvider
from shared.execution.types import ExecutionResult, GenerationRequest


class ModalCloudProvider(BaseCloudProvider):
    """
    Executes generation on Modal (https://modal.com).

    Supports:
    1. Modal HTTP Webhook: POST to MODAL_ENDPOINT or WAN2GP_CLOUD_ENDPOINT
    2. Modal SDK: modal.Function.lookup(app_name, function_name)
    """

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        app_name: Optional[str] = None,
        function_name: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 3600.0,
        poll_interval: float = 2.0,
        **kwargs: Any,
    ):
        self.endpoint_url = (endpoint_url or os.environ.get("MODAL_ENDPOINT") or os.environ.get("WAN2GP_CLOUD_ENDPOINT", "")).rstrip("/")
        self.app_name = app_name or os.environ.get("MODAL_APP_NAME", "wan2gp-cloud")
        self.function_name = function_name or os.environ.get("MODAL_FUNCTION_NAME", "generate")
        self.api_key = api_key or os.environ.get("MODAL_API_KEY") or os.environ.get("WAN2GP_CLOUD_API_KEY", "")
        self.timeout = float(os.environ.get("WAN2GP_CLOUD_TIMEOUT", str(timeout)))
        self.poll_interval = float(poll_interval)
        self._active_job_id: Optional[str] = None

    def cancel(self, task_id: Optional[Any] = None) -> bool:
        """Cancel an ongoing generation task on Modal."""
        target_id = self._active_job_id or (str(task_id) if task_id else "")
        if not target_id:
            return False
        return self._stop_job_http(target_id)

    def _stop_job_http(self, job_id: str) -> bool:
        if not self.endpoint_url or not job_id:
            return False
        try:
            url = f"{self.endpoint_url}/stop/{job_id}"
            headers = {"Content-Type": "application/json", "User-Agent": "Wan2GP-Modal/1.0"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
                headers["X-API-Key"] = self.api_key
            req = urllib.request.Request(url, data=b"{}", headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                return resp.status == 200
        except Exception:
            return False

    @property
    def name(self) -> str:
        return "modal"

    def is_configured(self) -> bool:
        if bool(self.endpoint_url):
            return True
        # Check if Modal SDK is available and credentials are set
        has_modal_auth = bool(os.environ.get("MODAL_TOKEN_ID") and os.environ.get("MODAL_TOKEN_SECRET"))
        if has_modal_auth:
            return True
        # Check standard ~/.modal.toml
        modal_config_file = os.path.expanduser("~/.modal.toml")
        return os.path.isfile(modal_config_file)

    def get_configuration_help(self) -> str:
        return (
            "Modal Cloud Provider configuration:\n"
            "Option A (Recommended Web Endpoint):\n"
            "  - Set MODAL_ENDPOINT (e.g. export MODAL_ENDPOINT=https://your-workspace--wan2gp.modal.run)\n"
            "Option B (Modal SDK):\n"
            "  - Set MODAL_TOKEN_ID and MODAL_TOKEN_SECRET\n"
            "  - Set MODAL_APP_NAME (default: wan2gp-cloud)\n"
            "  - Or run 'modal setup' on your system"
        )

    def _resolve_endpoint_url(self) -> str:
        """Auto-discover the deployed Modal FastAPI URL when MODAL_ENDPOINT is not set."""
        if self.endpoint_url:
            return self.endpoint_url
        try:
            import modal
            candidates = []
            for name in (self.app_name, "wan2gp-cloud"):
                if name and name not in candidates:
                    candidates.append(name)
            for app_name in candidates:
                try:
                    fn = modal.Function.from_name(app_name, "fastapi_app")
                    web_url = fn.get_web_url()
                    if web_url:
                        self.endpoint_url = str(web_url).rstrip("/")
                        self.app_name = app_name
                        return self.endpoint_url
                except Exception:
                    continue
        except ImportError:
            pass
        return ""

    @staticmethod
    def _stream_download_file(dl_url: str, dest_file: Path, headers: Dict[str, str], chunk_size: int = 2 * 1024 * 1024) -> None:
        """Stream a remote file to disk in chunks to handle >16 MB videos without RAM spikes."""
        dl_req = urllib.request.Request(dl_url, headers=headers, method="GET")
        tmp_file = dest_file.with_name(f".{dest_file.name}.part")
        try:
            with urllib.request.urlopen(dl_req, timeout=300.0) as dl_resp:
                with open(tmp_file, "wb") as f_out:
                    while True:
                        chunk = dl_resp.read(chunk_size)
                        if not chunk:
                            break
                        f_out.write(chunk)
            os.replace(tmp_file, dest_file)
        except Exception:
            if tmp_file.exists():
                try:
                    tmp_file.unlink()
                except OSError:
                    pass
            raise

    def execute(
        self,
        request: GenerationRequest,
        send_cmd: Optional[Callable[[str, Any], None]] = None,
        **kwargs: Any,
    ) -> ExecutionResult:
        fn_send_cmd = send_cmd if send_cmd is not None else (lambda cmd, data=None: None)
        start_time = time.time()

        if not self.is_configured():
            msg = f"Modal Cloud Provider is not configured.\n\n{self.get_configuration_help()}"
            fn_send_cmd("error", msg)
            return ExecutionResult(success=False, status="failed", task_id=request.task_id, error=msg)

        # 1. Prefer HTTP Web Endpoint (auto-discovered from Modal app if MODAL_ENDPOINT is unset)
        if self._resolve_endpoint_url():
            return self._execute_http(request, fn_send_cmd, start_time)

        # 2. Fallback to Modal SDK
        return self._execute_sdk(request, fn_send_cmd, start_time)

    def execute_batch(
        self,
        batch_zip_bytes: bytes,
        send_cmd: Optional[Callable[[str, Any], None]] = None,
        gpu_type: Optional[str] = None,
        save_path: Optional[str] = None,
    ) -> ExecutionResult:
        """
        Execute a bundled multi-task batch on 1 warm GPU container sequentially.
        Progressive downloads stream each completed video immediately to disk.
        """
        fn_send_cmd = send_cmd if send_cmd is not None else (lambda cmd, data=None: None)
        start_time = time.time()
        if not self._resolve_endpoint_url():
            msg = f"Modal Cloud Provider endpoint not found.\n\n{self.get_configuration_help()}"
            fn_send_cmd("error", msg)
            return ExecutionResult(success=False, status="failed", error=msg)
        resolved_gpu = gpu_type or os.environ.get("MODAL_DEFAULT_GPU", "L40S")
        resolved_save = save_path or os.environ.get("WAN2GP_OUTPUT_DIR") or "outputs"
        return self._execute_http_payload(
            zip_payload=batch_zip_bytes,
            fn_send_cmd=fn_send_cmd,
            start_time=start_time,
            gpu_type=resolved_gpu,
            save_path=resolved_save,
            task_id="batch",
        )

    def _execute_http(
        self,
        request: GenerationRequest,
        fn_send_cmd: Callable[[str, Any], None],
        start_time: float,
    ) -> ExecutionResult:
        gpu_type = os.environ.get("MODAL_DEFAULT_GPU", "L40S")
        if isinstance(request.settings, dict):
            gpu_type = request.settings.get("gpu_type") or gpu_type

        # Package request and any media inputs into a self-contained ZIP bundle
        fn_send_cmd("status", "Modal: Packaging generation bundle and media attachments...")
        zip_payload = package_generation_request_to_zip(request)
        save_dir_setting = os.environ.get("WAN2GP_OUTPUT_DIR") or (request.settings.get("save_path") if hasattr(request, "settings") and isinstance(request.settings, dict) else None) or "outputs"

        return self._execute_http_payload(
            zip_payload=zip_payload,
            fn_send_cmd=fn_send_cmd,
            start_time=start_time,
            gpu_type=gpu_type,
            save_path=save_dir_setting,
            task_id=request.task_id,
        )

    def _execute_http_payload(
        self,
        zip_payload: bytes,
        fn_send_cmd: Callable[[str, Any], None],
        start_time: float,
        gpu_type: str,
        save_path: str,
        task_id: Any = None,
    ) -> ExecutionResult:
        job_id = None
        try:
            payload_size_mb = len(zip_payload) / (1024 * 1024)

            fn_send_cmd("status", f"Modal: Submitting bundle ({payload_size_mb:.2f} MB) to {self.endpoint_url} (GPU: {gpu_type})...")
            url = f"{self.endpoint_url}/generate?gpu_type={gpu_type}"
            headers = {"Content-Type": "application/zip", "User-Agent": "Wan2GP-Modal/1.0"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
                headers["X-API-Key"] = self.api_key

            req = urllib.request.Request(url, data=zip_payload, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))

            job_id = resp_data.get("job_id") or resp_data.get("id")
            self._active_job_id = job_id
            status = str(resp_data.get("status", "queued")).lower()

            # If already completed or failed synchronously
            if status == "completed":
                outputs = resp_data.get("output_files", []) or resp_data.get("outputs", [])
                fn_send_cmd("status", "Modal: Remote generation completed successfully")
                fn_send_cmd("output", None)
                return ExecutionResult(
                    success=True,
                    status="completed",
                    task_id=task_id or job_id,
                    output_files=list(outputs),
                    metadata={"provider": "modal", "endpoint": self.endpoint_url},
                    generation_info=resp_data.get("generation_info", {}),
                    execution_time=time.time() - start_time,
                )

            # Asynchronous execution polling
            if status in ("queued", "running") and job_id:
                fn_send_cmd("status", f"Modal: Job {job_id} submitted to cloud GPU ({resp_data.get('gpu_selected', 'auto')}). Polling progress...")
                poll_interval = 2.0
                deadline = time.time() + self.timeout
                last_log_len = 0

                out_dir = Path(save_path)
                out_dir.mkdir(parents=True, exist_ok=True)
                downloaded_paths = []
                downloaded_fnames = set()

                while time.time() < deadline:
                    time.sleep(poll_interval)
                    status_url = f"{self.endpoint_url}/status/{job_id}"
                    status_req = urllib.request.Request(status_url, headers=headers, method="GET")
                    try:
                        with urllib.request.urlopen(status_req, timeout=30.0) as s_resp:
                            s_data = json.loads(s_resp.read().decode("utf-8"))
                    except Exception:
                        continue

                    state = str(s_data.get("status", "running")).lower()

                    # Progressive downloads: download completed videos as soon as they finish
                    current_remote_files = s_data.get("files", [])
                    for fname in current_remote_files:
                        if fname in ("input.zip", "status.json", "run.log", "STOP", "results.json") or fname.endswith(".tmp") or "_tmp." in fname or fname in downloaded_fnames:
                            continue
                        dest_file = out_dir / fname
                        dl_url = f"{self.endpoint_url}/download/{job_id}/{urllib.parse.quote(fname)}"
                        try:
                            self._stream_download_file(dl_url, dest_file, headers)
                            resolved_str = str(dest_file.resolve())
                            downloaded_paths.append(resolved_str)
                            downloaded_fnames.add(fname)
                            fn_send_cmd("status", f"Modal: Progressive download ready -> {fname}")
                            fn_send_cmd("output", resolved_str)
                        except Exception as dl_err:
                            fn_send_cmd("status", f"Modal: Warning downloading {fname}: {dl_err}")

                    # Fetch real-time log lines for progress feedback
                    try:
                        logs_url = f"{self.endpoint_url}/logs/{job_id}"
                        l_req = urllib.request.Request(logs_url, headers=headers, method="GET")
                        with urllib.request.urlopen(l_req, timeout=10.0) as l_resp:
                            logs_text = l_resp.read().decode("utf-8", errors="replace")
                            if len(logs_text) > last_log_len:
                                new_logs = logs_text[last_log_len:].strip()
                                last_log_len = len(logs_text)
                                if new_logs:
                                    import re
                                    for line in new_logs.splitlines():
                                        clean_l = line.strip()
                                        if clean_l:
                                            print(f"[Modal] {clean_l}", flush=True)
                                            m_prog = re.search(r"\[(\d+)/(\d+)\]\s+([^|\n]+)", clean_l)
                                            if m_prog:
                                                curr_step = int(m_prog.group(1))
                                                total_steps = int(m_prog.group(2))
                                                phase_name = m_prog.group(3).strip()
                                                if total_steps > 0:
                                                    fn_send_cmd("progress", [(curr_step, total_steps), f"Cloud GPU: {phase_name}", total_steps])
                                            else:
                                                m_pct = re.search(r"(\d+)%\|.*\|\s*(\d+)/(\d+)", clean_l)
                                                if m_pct:
                                                    curr_step = int(m_pct.group(2))
                                                    total_steps = int(m_pct.group(3))
                                                    if total_steps > 0:
                                                        fn_send_cmd("progress", [(curr_step, total_steps), f"Cloud GPU: Denoising ({m_pct.group(1)}%)", total_steps])

                                    last_line = new_logs.splitlines()[-1]
                                    fn_send_cmd("status", f"Modal GPU: {last_line[:120]}")
                    except Exception:
                        pass

                    if state == "completed":
                        fn_send_cmd("status", "Modal: Cloud generation completed. Downloading output files...")
                        output_files = s_data.get("files", [])
                        if not output_files:
                            try:
                                dl_list_url = f"{self.endpoint_url}/download/{job_id}"
                                dl_list_req = urllib.request.Request(dl_list_url, headers=headers, method="GET")
                                with urllib.request.urlopen(dl_list_req, timeout=15.0) as dl_list_resp:
                                    dl_list_data = json.loads(dl_list_resp.read().decode("utf-8"))
                                    output_files = dl_list_data.get("files", [])
                            except Exception:
                                pass

                        for fname in output_files:
                            if fname in ("input.zip", "status.json", "run.log", "STOP", "results.json") or fname.endswith(".tmp") or "_tmp." in fname or fname in downloaded_fnames:
                                continue
                            dest_file = out_dir / fname
                            dl_url = f"{self.endpoint_url}/download/{job_id}/{urllib.parse.quote(fname)}"
                            try:
                                self._stream_download_file(dl_url, dest_file, headers)
                                downloaded_paths.append(str(dest_file.resolve()))
                                downloaded_fnames.add(fname)
                                fn_send_cmd("output", str(dest_file.resolve()))
                            except Exception as dl_err:
                                fn_send_cmd("status", f"Modal: Warning downloading {fname}: {dl_err}")

                        total_exec_time = time.time() - start_time
                        gpu_chosen = str(resp_data.get("gpu_selected", "L40S")).upper()
                        modal_rates = {
                            "T4": 0.59,
                            "L4": 0.80,
                            "L40S": 1.95,
                            "A100_40GB": 2.10,
                            "A100_80GB": 2.80,
                            "H100": 4.50,
                        }
                        rate_key = "L40S"
                        for k in modal_rates:
                            if k in gpu_chosen.replace("-", "_"):
                                rate_key = k
                                break
                        hourly_rate = modal_rates.get(rate_key, 1.95)
                        cost_usd = (total_exec_time / 3600.0) * hourly_rate

                        cost_summary = f"${cost_usd:.3f} USD ({rate_key} @ ${hourly_rate:.2f}/hr, {int(total_exec_time)}s)"
                        print(f"[Modal Cost] {cost_summary}", flush=True)
                        fn_send_cmd("status", f"Modal: Done in {int(total_exec_time)}s on {rate_key} | Estimated Cost: ${cost_usd:.3f}")
                        fn_send_cmd("output", downloaded_paths[0] if downloaded_paths else None)
                        return ExecutionResult(
                            success=True,
                            status="completed",
                            task_id=task_id or job_id,
                            output_files=downloaded_paths,
                            metadata={
                                "provider": "modal",
                                "endpoint": self.endpoint_url,
                                "job_id": job_id,
                                "gpu": rate_key,
                                "cost_usd": round(cost_usd, 4),
                                "hourly_rate": hourly_rate
                            },
                            execution_time=total_exec_time,
                        )

                    if state in ("failed", "error", "stopped"):
                        self._stop_job_http(job_id)
                        err_msg = s_data.get("error") or f"Job {job_id} {state} on Modal"
                        fn_send_cmd("error", f"Modal generation error: {err_msg}")
                        return ExecutionResult(
                            success=False,
                            status="failed",
                            task_id=task_id or job_id,
                            error=str(err_msg),
                            execution_time=time.time() - start_time,
                        )

                # Deadline reached
                if job_id:
                    self._stop_job_http(job_id)
                timeout_msg = f"Modal generation timed out after {self.timeout}s"
                fn_send_cmd("error", timeout_msg)
                return ExecutionResult(
                    success=False,
                    status="failed",
                    task_id=task_id or job_id,
                    error=timeout_msg,
                    execution_time=time.time() - start_time,
                )

            err = resp_data.get("error", "Modal execution returned non-success.")
            if job_id:
                self._stop_job_http(job_id)
            fn_send_cmd("error", f"Modal generation error: {err}")
            return ExecutionResult(
                success=False,
                status="failed",
                task_id=task_id or job_id,
                error=str(err),
                execution_time=time.time() - start_time,
            )

        except urllib.error.URLError as exc:
            if job_id:
                try:
                    self._stop_job_http(job_id)
                except Exception:
                    pass
            err_msg = f"Modal connection error: {exc.reason}"
            fn_send_cmd("error", err_msg)
            return ExecutionResult(success=False, status="failed", task_id=task_id or job_id, error=err_msg, execution_time=time.time() - start_time)
        except (KeyboardInterrupt, GeneratorExit):
            if job_id:
                try:
                    self._stop_job_http(job_id)
                except Exception:
                    pass
            err_msg = "Generation cancelled / disconnected."
            fn_send_cmd("error", err_msg)
            return ExecutionResult(success=False, status="failed", task_id=task_id or job_id, error=err_msg, execution_time=time.time() - start_time)
        except Exception as e:
            if job_id:
                try:
                    self._stop_job_http(job_id)
                except Exception:
                    pass
            err_msg = f"Modal execution failed: {e}"
            fn_send_cmd("error", err_msg)
            return ExecutionResult(
                success=False,
                status="failed",
                task_id=task_id or job_id,
                error=err_msg,
                execution_time=time.time() - start_time,
            )
        finally:
            self._active_job_id = None

    def _execute_sdk(
        self,
        request: GenerationRequest,
        fn_send_cmd: Callable[[str, Any], None],
        start_time: float,
    ) -> ExecutionResult:
        try:
            import modal
        except ImportError:
            msg = (
                "Modal Python SDK is not installed.\n"
                "Install it using: pip install modal\n"
                "Or configure MODAL_ENDPOINT to use HTTP webhook mode."
            )
            fn_send_cmd("error", msg)
            return ExecutionResult(success=False, status="failed", task_id=request.task_id, error=msg)

        try:
            fn_send_cmd("status", f"Modal SDK: Looking up function '{self.function_name}' in app '{self.app_name}'...")
            lookup_fn = getattr(modal.Function, "from_name", None) or getattr(modal.Function, "lookup")
            remote_fn = lookup_fn(self.app_name, self.function_name)

            fn_send_cmd("status", "Modal SDK: Spawning remote cloud GPU execution...")
            fn_send_cmd("progress", [(1, 3), "Modal: Invoking remote worker..."])

            payload = request.to_dict()
            result_payload = remote_fn.remote(payload)

            fn_send_cmd("progress", [(3, 3), "Modal: Remote execution completed."])
            fn_send_cmd("status", "Modal: Generation completed successfully")
            fn_send_cmd("output", None)

            outputs = result_payload.get("output_files", []) if isinstance(result_payload, dict) else [str(result_payload)]
            return ExecutionResult(
                success=True,
                status="completed",
                task_id=request.task_id,
                output_files=list(outputs),
                metadata={"provider": "modal", "app_name": self.app_name, "function": self.function_name},
                execution_time=time.time() - start_time,
            )
        except Exception as exc:
            err_msg = f"Modal SDK execution failed: {exc}"
            fn_send_cmd("error", err_msg)
            return ExecutionResult(success=False, status="failed", task_id=request.task_id, error=err_msg, execution_time=time.time() - start_time)


# Backwards-compatible alias — callers may import either ModalProvider or ModalCloudProvider.
ModalProvider = ModalCloudProvider
