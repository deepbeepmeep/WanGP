# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
from __future__ import annotations

import html
import time

from shared.api import extract_status_phase_label


def phase_label_from_status(status: str = "") -> str:
    return extract_status_phase_label(status)


def phase_label_from_update(update=None, *, status: str = "", phase: str = "", raw_phase: str = "") -> str:
    status_phase = phase_label_from_status(status or getattr(update, "status", ""))
    raw_phase_text = str(raw_phase or getattr(update, "raw_phase", "") or phase or "").strip()
    if len(status_phase) > 0:
        return status_phase
    return raw_phase_text


def _format_elapsed(seconds: float | None) -> str:
    if seconds is None:
        return "--"
    total_seconds = int(round(float(seconds)))
    if total_seconds < 0:
        total_seconds = 0
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds_only = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds_only:02d}" if hours > 0 else f"{minutes:02d}:{seconds_only:02d}"


class ChunkCallbacks:
    def __init__(self) -> None:
        self.phase_label = "Queued in WanGP..."
        self.status_text = "Queued in WanGP..."
        self.current_step = None
        self.total_steps = None
        self._last_explicit_status_at = 0.0

    def on_status(self, status):
        status_text = str(status or "").strip()
        if len(status_text) == 0:
            return
        self.status_text = status_text
        status_phase = phase_label_from_status(self.status_text)
        if len(status_phase) > 0:
            if status_phase != self.phase_label:
                self.current_step = None
                self.total_steps = None
            self.phase_label = status_phase
            self._last_explicit_status_at = time.time()

    def on_progress(self, update):
        incoming_status = str(getattr(update, "status", "") or "").strip()
        incoming_phase = phase_label_from_update(update, status=incoming_status or self.status_text)
        incoming_step = getattr(update, "current_step", None)
        incoming_total = getattr(update, "total_steps", None)
        if time.time() - self._last_explicit_status_at <= 1.0 and len(self.phase_label) > 0 and len(incoming_phase) > 0 and incoming_phase.lower() != self.phase_label.lower() and incoming_step is None:
            return
        if len(incoming_status) > 0:
            self.status_text = incoming_status
        if len(incoming_phase) > 0:
            self.phase_label = incoming_phase
        self.current_step = incoming_step
        self.total_steps = incoming_total


class StatusHtml(str):
    """A rendered status that keeps its values in fields, for agents following a run without the page."""
    fields: dict


def _status(text: str, **fields) -> StatusHtml:
    status = StatusHtml(text)
    status.fields = fields
    return status


def _tone(phase: str) -> str:
    phase = phase.strip().lower()
    if phase in ("", "idle"):
        return "idle"
    if "completed" in phase:
        return "done"
    if phase == "error":
        return "error"
    if phase in ("stopped", "stopping", "info", "merge pending"):
        return "warn"
    return "running"


def _meter(ratio: float | None, kind: str = "") -> str:
    width = f"{100.0 * min(1.0, max(0.0, ratio or 0.0)):.2f}%"
    return f"<div class='mf-meter{' ' + kind if kind else ''}'><span style='width:{width}'></span></div>"


def _status_card(raw_phase_text: str, raw_status_text: str, body: str, *, phase_current_step, phase_total_steps, elapsed_seconds, eta_seconds, prefer_status_phase: bool) -> str:
    """The pill of the phase with the times, the given meters, then the phase's step meter and the status line."""
    phase_html = html.escape(raw_phase_text or "Queued in WanGP...")
    has_phase_progress = phase_current_step is not None and phase_total_steps is not None and phase_total_steps > 0
    normalized_phase = raw_phase_text.lower()
    show_status_line = (not prefer_status_phase) and len(raw_status_text) > 0 and (len(normalized_phase) == 0 or normalized_phase not in raw_status_text.lower())
    tone = _tone(raw_phase_text)
    times = f"<span class='mf-times'><span>Elapsed <b>{html.escape(_format_elapsed(elapsed_seconds))}</b></span><span>ETA <b>{html.escape(_format_elapsed(eta_seconds))}</b></span></span>" if tone != "idle" else ""
    steps = ""
    if has_phase_progress:
        steps = f"<div class='mf-meter-row'><span>Progress</span><b>{phase_current_step} / {phase_total_steps}</b></div>{_meter(float(phase_current_step) / float(phase_total_steps), 'thin')}"
    status_line = f"<div class='mf-status-text{' muted' if tone == 'idle' else ''}'>{html.escape(raw_status_text)}</div>" if show_status_line else ""
    return f"<div class='mf-status' data-tone='{tone}'><div class='mf-status-head'><span class='mf-pill'><i></i>{phase_html}</span>{times}</div>{body}{steps}{status_line}</div>"


def render_chunk_status_html(total_chunks: int, completed_chunks: int, current_chunk: int, phase_label: str, status_text: str, *, continued: bool = False, phase_current_step=None, phase_total_steps=None, elapsed_seconds: float | None = None, eta_seconds: float | None = None, prefer_status_phase: bool = False) -> str:
    total_chunks = int(total_chunks)
    completed_chunks = int(completed_chunks)
    current_chunk = int(current_chunk)
    top_ratio = completed_chunks / total_chunks if total_chunks > 0 else 0.0
    chunks_text = f"{completed_chunks} / {total_chunks}" if total_chunks > 0 else "- / -"
    raw_status_text = status_text.strip()
    raw_phase_text = phase_label.strip()
    if prefer_status_phase:
        derived_phase = phase_label_from_status(raw_status_text)
        if len(derived_phase) > 0:
            raw_phase_text = derived_phase
    continued_html = " <em>Continued</em>" if continued else ""
    body = f"<div class='mf-meter-row'><span>Chunks processed{continued_html}</span><b>{chunks_text}</b></div>{_meter(top_ratio)}"
    return _status(
        _status_card(raw_phase_text, raw_status_text, body, phase_current_step=phase_current_step, phase_total_steps=phase_total_steps, elapsed_seconds=elapsed_seconds, eta_seconds=eta_seconds, prefer_status_phase=prefer_status_phase),
        total_chunks=total_chunks, completed_chunks=completed_chunks, current_chunk=current_chunk, phase=raw_phase_text, status=raw_status_text, step=phase_current_step, steps=phase_total_steps, eta_s=eta_seconds,
    )


def render_process_status_html(phase_label: str, status_text: str, *, phase_current_step=None, phase_total_steps=None, elapsed_seconds: float | None = None, eta_seconds: float | None = None, prefer_status_phase: bool = False) -> str:
    raw_status_text = status_text.strip()
    raw_phase_text = phase_label.strip()
    if prefer_status_phase:
        derived_phase = phase_label_from_status(raw_status_text)
        if len(derived_phase) > 0:
            raw_phase_text = derived_phase
    return _status(
        _status_card(raw_phase_text, raw_status_text, "", phase_current_step=phase_current_step, phase_total_steps=phase_total_steps, elapsed_seconds=elapsed_seconds, eta_seconds=eta_seconds, prefer_status_phase=prefer_status_phase),
        phase=raw_phase_text, status=raw_status_text, step=phase_current_step, steps=phase_total_steps, eta_s=eta_seconds,
    )


def render_batch_status_html(total_files: int, completed_files: int, current_index: int, current_file: str, status_text: str, failed_files: int = 0) -> str:
    total_files = max(0, int(total_files or 0))
    completed_files = max(0, int(completed_files or 0))
    failed_files = max(0, int(failed_files or 0))
    current_index = max(0, int(current_index or 0))
    ratio = float(completed_files) / float(total_files) if total_files > 0 else 0.0
    current_name = html.escape(str(current_file or ""))
    status_html = html.escape(str(status_text or "").strip())
    failures_html = f" <em style='color:var(--mf-red)'>{failed_files} {'failure' if failed_files == 1 else 'failures'}</em>" if failed_files > 0 else ""
    current_line = f"<div class='mf-current'>Current file <b>{current_name}</b></div>" if current_name else ""
    return _status(
        "<div class='mf-status batch'>"
        f"<div class='mf-meter-row'><span>Batch files processed{failures_html}</span><b>{completed_files} / {total_files}</b></div>"
        f"{_meter(ratio, 'batch')}{current_line}<div class='mf-status-text'>{status_html}</div>"
        "</div>",
        total_files=total_files, completed_files=completed_files, failed_files=failed_files, current_index=current_index, current_file=str(current_file or ""), status=str(status_text or "").strip(),
    )


def render_output_file_html(output_path: str) -> str:
    value = html.escape(output_path, quote=False)
    filled = " set" if len(output_path.strip()) > 0 else ""
    return (
        "<div class='mf-output'><div class='mf-output-label'>Output File</div>"
        f"<div class='mf-output-box{filled}'><svg viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round' aria-hidden='true'>"
        "<path d='M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8Z'/><path d='M14 3v5h5'/></svg>"
        "<textarea readonly onclick='this.select()' spellcheck='false' rows='1' placeholder='The path of the result appears here when the process starts.'>"
        f"{value}</textarea></div></div>"
    )
