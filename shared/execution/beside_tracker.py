# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""
Sideways / Beside Job Tracker for Wan2GP.

Tracks jobs running concurrently / beside the main queue across parallel Cloud GPU containers.
Shared between UI (wgp.py) and Deepy / MCP execution layers.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
import threading
import time
from typing import Any, Dict, List, Optional


@dataclass
class BesideJobInfo:
    job_id: str
    prompt: str
    model_type: str
    status: str = "queued"  # queued, packaging, running, downloading, completed, failed, cancelled
    gpu_type: str = "L40S"
    start_time: float = field(default_factory=time.time)
    finish_time: Optional[float] = None
    output_files: List[str] = field(default_factory=list)
    error: Optional[str] = None
    origin: str = "ui"  # "ui" or "deepy"


class BesideJobTracker:
    def __init__(self) -> None:
        self._jobs: Dict[str, BesideJobInfo] = {}
        self._lock = threading.Lock()

    def register_job(
        self,
        job_id: str,
        prompt: str,
        model_type: str,
        gpu_type: str = "L40S",
        origin: str = "ui",
    ) -> BesideJobInfo:
        with self._lock:
            info = BesideJobInfo(
                job_id=job_id,
                prompt=prompt,
                model_type=model_type,
                gpu_type=gpu_type,
                origin=origin,
            )
            self._jobs[job_id] = info
            return info

    def update_job(self, job_id: str, **kwargs: Any) -> None:
        with self._lock:
            if job_id in self._jobs:
                job = self._jobs[job_id]
                for k, v in kwargs.items():
                    if hasattr(job, k):
                        setattr(job, k, v)
                if kwargs.get("status") in ("completed", "failed", "cancelled") and not job.finish_time:
                    job.finish_time = time.time()

    def get_active_jobs(self) -> List[BesideJobInfo]:
        with self._lock:
            now = time.time()
            return [
                j for j in self._jobs.values()
                if j.status in ("queued", "packaging", "running", "downloading")
                or (j.finish_time and now - j.finish_time < 90)
            ]

    def has_running_jobs(self) -> bool:
        with self._lock:
            return any(
                j.status in ("queued", "packaging", "running", "downloading")
                for j in self._jobs.values()
            )

    def clear(self) -> None:
        with self._lock:
            self._jobs.clear()


beside_tracker = BesideJobTracker()


def render_beside_jobs_html() -> str:
    """Renders the HTML status block for sideways/parallel cloud jobs."""
    active = beside_tracker.get_active_jobs()
    if not active:
        return ""

    rows = []
    for job in active:
        elapsed = int(time.time() - job.start_time)
        mins, secs = divmod(elapsed, 60)
        time_str = f"{mins}m {secs}s" if mins else f"{secs}s"

        status_color = "#38bdf8"
        if job.status == "completed":
            status_color = "#4ade80"
        elif job.status in ("failed", "cancelled"):
            status_color = "#f87171"
        elif job.status in ("downloading", "packaging"):
            status_color = "#fbbf24"

        badge = (
            f'<span style="background: {status_color}22; color: {status_color}; '
            f'border: 1px solid {status_color}66; padding: 2px 8px; border-radius: 9999px; '
            f'font-size: 11px; font-weight: 600; text-transform: uppercase;">{job.status}</span>'
        )
        origin_badge = (
            '<span style="background: rgba(168, 85, 247, 0.2); color: #c084fc; '
            'border: 1px solid rgba(168, 85, 247, 0.4); padding: 1px 6px; '
            'border-radius: 4px; font-size: 10px; font-weight: 600;">Deepy</span>'
            if job.origin == "deepy"
            else '<span style="background: rgba(56, 189, 248, 0.2); color: #7dd3fc; '
            'border: 1px solid rgba(56, 189, 248, 0.4); padding: 1px 6px; '
            'border-radius: 4px; font-size: 10px; font-weight: 600;">UI</span>'
        )
        prompt_snippet = (job.prompt[:55] + "...") if len(job.prompt) > 55 else (job.prompt or "Generation")

        rows.append(f"""
            <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.06);">
                <td style="padding: 7px 10px; font-family: monospace; font-size: 11px; color: #94a3b8;">#{job.job_id[:8]}</td>
                <td style="padding: 7px 10px; font-size: 12px; color: #e2e8f0;">{prompt_snippet}</td>
                <td style="padding: 7px 10px; font-size: 11px; color: #cbd5e1; text-align: center;">{job.model_type}</td>
                <td style="padding: 7px 10px; font-size: 11px; color: #94a3b8; text-align: center;">{job.gpu_type}</td>
                <td style="padding: 7px 10px; text-align: center;">{origin_badge}</td>
                <td style="padding: 7px 10px; font-size: 11px; color: #94a3b8; text-align: center;">{time_str}</td>
                <td style="padding: 7px 10px; text-align: center;">{badge}</td>
            </tr>
        """)

    max_concurrent = os.environ.get("MAX_CONCURRENT_CLOUD_JOBS", "3")
    return f"""
    <div class="wangp-sideways-status" style="margin-bottom: 12px; background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(56, 189, 248, 0.45); border-radius: 8px; padding: 12px 14px; box-shadow: 0 4px 14px rgba(0,0,0,0.3);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; border-bottom: 1px solid rgba(56, 189, 248, 0.25); padding-bottom: 6px;">
            <div style="display: flex; align-items: center; gap: 8px;">
                <span style="display: inline-block; width: 9px; height: 9px; background: #38bdf8; border-radius: 50%; box-shadow: 0 0 8px #38bdf8;"></span>
                <span style="font-size: 13px; font-weight: 700; color: #38bdf8; letter-spacing: 0.5px;">SIDEWAYS / PARALLEL CLOUD CONTAINERS</span>
                <span style="font-size: 11px; background: rgba(56, 189, 248, 0.2); color: #7dd3fc; padding: 1px 7px; border-radius: 4px; font-weight: 600;">{len(active)} Running</span>
            </div>
            <div style="font-size: 11px; color: #94a3b8;">Max Containers: <b style="color: #38bdf8;">{max_concurrent}</b></div>
        </div>
        <table style="width: 100%; border-collapse: collapse; text-align: left;">
            <thead>
                <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.1); color: #64748b; font-size: 11px; text-transform: uppercase;">
                    <th style="padding: 4px 10px;">ID</th>
                    <th style="padding: 4px 10px;">Prompt</th>
                    <th style="padding: 4px 10px; text-align: center;">Model</th>
                    <th style="padding: 4px 10px; text-align: center;">GPU</th>
                    <th style="padding: 4px 10px; text-align: center;">Source</th>
                    <th style="padding: 4px 10px; text-align: center;">Time</th>
                    <th style="padding: 4px 10px; text-align: center;">Status</th>
                </tr>
            </thead>
            <tbody>
                {"".join(rows)}
            </tbody>
        </table>
    </div>
    """
