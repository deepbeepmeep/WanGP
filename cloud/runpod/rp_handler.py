# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""
Production RunPod Serverless Worker for Wan2GP
Supports:
  1. Zero-Cost Ephemeral Mode: Uses the 70 GB container disk (/workspace/ckpts) with $0 idle cost.
  2. Persistent Volume Mode: Automatically detects /runpod-volume if attached.
"""

import os
import sys
import json
import base64
import time
import shutil
import tempfile
import subprocess
from pathlib import Path
import runpod

# Storage resolution: detect optional network volume or fallback to 70GB container disk
NETWORK_VOLUME = Path("/runpod-volume")
WAN2GP_DIR = Path("/workspace/Wan2GP")

if NETWORK_VOLUME.exists():
    STORAGE_MODE = "persistent_network_volume"
    CKPTS_DIR = NETWORK_VOLUME / "ckpts"
else:
    STORAGE_MODE = "ephemeral_container_disk"
    CKPTS_DIR = Path("/workspace/ckpts")

CKPTS_DIR.mkdir(parents=True, exist_ok=True)


def setup_worker_environment():
    """Symlink persistent or container checkpoints to Wan2GP ckpts directory."""
    target_ckpts = WAN2GP_DIR / "ckpts"
    if target_ckpts.exists() and not target_ckpts.is_symlink():
        shutil.rmtree(target_ckpts, ignore_errors=True)
    if not target_ckpts.exists():
        try:
            os.symlink(CKPTS_DIR, target_ckpts)
            print(f"[Worker] Symlinked {CKPTS_DIR} -> {target_ckpts} (Mode: {STORAGE_MODE})")
        except Exception as e:
            print(f"[Worker] Warning setting up ckpts symlink: {e}")


def handler(job):
    """
    RunPod Serverless Job Handler.
    Receives GenerationRequest or packaged ZIP bundle, executes headless Wan2GP,
    and returns generated output files with base64 data.
    """
    job_id = job.get("id", "job_unknown")
    job_input = job.get("input", {})
    start_time = time.time()

    runpod.serverless.progress_update(job, "Wan2GP Worker: Initializing job on cloud GPU...")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        output_dir = temp_path / "outputs"
        output_dir.mkdir(parents=True, exist_ok=True)

        bundle_b64 = job_input.get("bundle_base64")
        if bundle_b64:
            zip_path = temp_path / "input.zip"
            zip_path.write_bytes(base64.b64decode(bundle_b64))
            run_cmd = [
                sys.executable, str(WAN2GP_DIR / "wgp.py"),
                "--process", str(zip_path),
                "--output-dir", str(output_dir)
            ]
        else:
            settings_json = temp_path / "settings.json"
            settings_json.write_text(json.dumps(job_input.get("request", {})), encoding="utf-8")
            run_cmd = [
                sys.executable, str(WAN2GP_DIR / "wgp.py"),
                "--process", str(settings_json),
                "--output-dir", str(output_dir)
            ]

        runpod.serverless.progress_update(job, "Wan2GP Worker: Running diffusion inference...")

        env = os.environ.copy()
        env["WAN2GP_MODE"] = "local"  # Runs locally inside the GPU container
        env["HF_HOME"] = str(CKPTS_DIR / "hf_cache")

        proc = subprocess.Popen(
            run_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(WAN2GP_DIR),
            env=env
        )

        for line in proc.stdout:
            line_str = line.strip()
            if "%|" in line_str or "Step" in line_str:
                runpod.serverless.progress_update(job, line_str)

        proc.wait()

        if proc.returncode != 0:
            return {
                "status": "FAILED",
                "error": f"Wan2GP generation process exited with code {proc.returncode}"
            }

        # Gather output files
        output_files = []
        for file_path in output_dir.glob("*.*"):
            if file_path.suffix.lower() in (".mp4", ".png", ".jpg", ".flac", ".wav"):
                file_bytes = file_path.read_bytes()
                output_files.append({
                    "filename": file_path.name,
                    "mime_type": "video/mp4" if file_path.suffix == ".mp4" else "image/png",
                    "data_b64": base64.b64encode(file_bytes).decode("utf-8"),
                    "size_bytes": len(file_bytes)
                })

        return {
            "status": "COMPLETED",
            "output_files": output_files,
            "storage_mode": STORAGE_MODE,
            "execution_time_seconds": time.time() - start_time
        }


if __name__ == "__main__":
    setup_worker_environment()
    runpod.serverless.start({"handler": handler})
