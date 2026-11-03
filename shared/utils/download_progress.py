# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Task-scoped download progress/cancellation and process-wide Ctrl+C shutdown."""

import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps


class DownloadCancelled(Exception):
    pass


_sigint_requested = False


def _install_sigint_handler():
    # WanGP imports this module on the main thread before starting its workers.
    # Downloads may begin on any thread, where signal.signal is not permitted.
    if threading.current_thread() is not threading.main_thread():
        return
    previous = signal.getsignal(signal.SIGINT)
    if not callable(previous) or getattr(previous, "_wangp_download_sigint", False):
        return

    def interrupt(signum, frame):
        global _sigint_requested
        # No locks, I/O or joins in a signal handler. Workers observe this flag
        # and close their own transfers. Keep it set so queued files cannot start
        # while the main thread and thread pools are shutting down.
        _sigint_requested = True
        previous(signum, frame)

    interrupt._wangp_download_sigint = True
    signal.signal(signal.SIGINT, interrupt)


def check_download_cancelled(gen):
    if _sigint_requested:
        raise DownloadCancelled("Download cancelled by Ctrl+C")
    from shared.utils.media_control import current_control
    control = current_control()
    if control is not None and control.gen is gen:
        control.checkpoint(unload=False)
    if gen is not None and (gen.get("abort", False) or (gen.get("abort_callback") is not None and gen["abort_callback"]())):
        raise DownloadCancelled("Download cancelled")


_current_download = ContextVar("wangp_download", default=None)
_operation_download_gen = ContextVar("wangp_download_gen", default=None)


def resolve_download_gen(gen=None):
    return _operation_download_gen.get() if gen is None else gen


@contextmanager
def download_operation(gen):
    """Share progress and cancellation with nested asset consumers in this worker."""
    token = _operation_download_gen.set(gen)
    try:
        yield
    finally:
        _operation_download_gen.reset(token)


class _DownloadProgress:
    def __init__(self, gen, filename, show_filename, file_index, file_count):
        self.gen = gen
        self.filename = os.path.basename(filename) if show_filename else None
        self.file_index, self.file_count = file_index, file_count
        self.started = self.updated = time.monotonic()
        self.initial = None

    def update(self, completed, total, force=False):
        check_download_cancelled(self.gen)
        if self.gen is None:
            return
        now = time.monotonic()
        if self.initial is None:
            self.initial = completed
        if force or now - self.updated >= 0.1:
            self.gen["download_progress"] = {"filename": self.filename, "completed": completed, "total": total, "speed": (completed - self.initial) / max(now - self.started, 0.001), "file_index": self.file_index, "file_count": self.file_count}
            callback = self.gen.get("download_progress_callback")
            if callback is not None:
                callback(self.gen["download_progress"])
            self.updated = now


@contextmanager
def download_context(gen, filename, show_filename=True, file_index=1, file_count=1):
    check_download_cancelled(gen)
    progress = _DownloadProgress(gen, filename, show_filename, file_index, file_count)
    token = _current_download.set(progress)
    try:
        progress.update(0, None, force=True)
        yield
        check_download_cancelled(gen)
        if gen is not None:
            report = gen["download_progress"]
            if (report["total"] is not None and report["completed"] == report["total"]) or (report["total"] is None and report["completed"] > 0):
                gen["download_progress_completed"] = dict(report, total=report["completed"], finished_at=time.monotonic())
                callback = gen.get("download_progress_callback")
                if callback is not None:
                    callback(gen["download_progress_completed"])
    finally:
        _current_download.reset(token)
        if gen is not None:
            gen.pop("download_progress", None)
            callback = gen.get("download_progress_callback")
            if callback is not None:
                callback(None)


def _http_get(url, temp_file, *, proxies=None, resume_size=0, headers=None, expected_size=None, displayed_filename=None, _nb_retries=5, _tqdm_bar=None):
    import requests
    from huggingface_hub import file_download as hf

    progress = _current_download.get()
    completed = resume_size
    progress.initial = completed
    progress.update(completed, expected_size, force=True)
    while True:
        check_download_cancelled(progress.gen)
        if expected_size is not None and completed == expected_size:
            return
        request_headers = dict(headers or {})
        if completed:
            request_headers["Range"] = hf._adjust_range_header(request_headers.get("Range"), completed)
        try:
            with hf._request_wrapper(method="GET", url=url, stream=True, proxies=proxies, headers=request_headers, timeout=hf.constants.HF_HUB_DOWNLOAD_TIMEOUT) as response:
                check_download_cancelled(progress.gen)
                if completed and response.status_code == 200:
                    temp_file.seek(0)
                    temp_file.truncate()
                    completed = progress.initial = 0
                content_length = hf._get_file_length_from_http_response(response)
                total = expected_size if expected_size is not None else content_length
                progress.update(completed, total, force=True)
                with hf._get_progress_bar_context(desc=displayed_filename or progress.filename or "Downloading", log_level=hf.logger.getEffectiveLevel(), total=total, initial=completed, name="huggingface_hub.http_get", _tqdm_bar=_tqdm_bar) as bar:
                    for chunk in response.iter_content(chunk_size=256 * 1024):
                        check_download_cancelled(progress.gen)
                        if chunk:
                            temp_file.write(chunk)
                            completed += len(chunk)
                            bar.update(len(chunk))
                            progress.update(completed, total)
                            _nb_retries = 5
                check_download_cancelled(progress.gen)
            break
        except (requests.ConnectionError, requests.ReadTimeout):
            check_download_cancelled(progress.gen)
            if _nb_retries <= 0:
                raise
            _nb_retries -= 1
            hf.logger.warning("Download connection interrupted; retrying from byte %s", completed)
            for _ in range(10):
                time.sleep(0.1)
                check_download_cancelled(progress.gen)
            hf.reset_sessions()
    if expected_size is not None and completed != expected_size:
        raise OSError(f"Download size mismatch: expected {expected_size} bytes, received {completed}")
    progress.update(completed, total, force=True)


def _xet_get(*, incomplete_path, xet_file_data, headers, expected_size=None, displayed_filename=None, _tqdm_bar=None):
    try:
        from hf_xet import XetFileInfo, XetSession
    except ImportError as error:
        raise RuntimeError('Cancellable Xet downloads require hf_xet>=1.5.2. Run: pip install "hf_xet>=1.5.2"') from error
    from huggingface_hub import file_download as hf

    progress = _current_download.get()
    check_download_cancelled(progress.gen)
    session = XetSession()
    waiter = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wangp-xet-download")
    try:
        connection = hf.refresh_xet_connection_info(file_data=xet_file_data, headers=headers)
        check_download_cancelled(progress.gen)
        group = session.new_file_download_group(endpoint=connection.endpoint, token=connection.access_token, token_expiry_unix_secs=connection.expiration_unix_epoch, token_refresh_url=xet_file_data.refresh_route, token_refresh_headers=headers)
        check_download_cancelled(progress.gen)
        handle = group.start_download_file(XetFileInfo(xet_file_data.file_hash, expected_size), str(incomplete_path.absolute()))
        # The blocking wait finalizes native task results; try_result alone does not.
        finished = waiter.submit(group.wait_to_finish)
        with hf._get_progress_bar_context(desc=displayed_filename or progress.filename or "Downloading", log_level=hf.logger.getEffectiveLevel(), total=expected_size, initial=0, name="huggingface_hub.xet_get", _tqdm_bar=_tqdm_bar) as bar:
            completed = 0
            while True:
                check_download_cancelled(progress.gen)
                report = handle.progress()
                if report is not None:
                    bar.update(report.bytes_completed - completed)
                    completed = report.bytes_completed
                    progress.update(completed, expected_size)
                if finished.done():
                    finished.result()
                    break
                time.sleep(0.1)
            check_download_cancelled(progress.gen)
            size = incomplete_path.stat().st_size
            if expected_size is not None and size != expected_size:
                raise OSError(f"Download size mismatch: expected {expected_size} bytes, received {size}")
            bar.update(size - completed)
            progress.update(size, expected_size, force=True)
    finally:
        # This session belongs to this file only. Shutdown joins native workers before
        # the caller closes/unlinks the incomplete file, including on cancellation.
        # No Python progress callbacks run on those workers (we poll above).
        try:
            session.sigint_abort()
        finally:
            waiter.shutdown(wait=True)


def install_hf_download_patch():
    from huggingface_hub import file_download as hf

    _install_sigint_handler()
    if getattr(hf.http_get, "_wangp_download_patch", False):
        return
    original_http_get = hf.http_get
    original_xet_get = hf.xet_get
    original_download = hf._download_to_tmp_and_move

    @wraps(original_http_get)
    def http_get(*args, **kwargs):
        if _current_download.get() is None:
            return original_http_get(*args, **kwargs)
        return _http_get(*args, **kwargs)

    @wraps(original_xet_get)
    def xet_get(*args, **kwargs):
        if _current_download.get() is None:
            return original_xet_get(*args, **kwargs)
        return _xet_get(*args, **kwargs)

    @wraps(original_download)
    def download_to_tmp_and_move(incomplete_path, destination_path, *args, **kwargs):
        # HF snapshot workers and callers without a generation task need an owned,
        # cancellable Xet session too, rather than HF's blocking legacy downloader.
        context = nullcontext() if _current_download.get() is not None else download_context(None, kwargs.get("filename") or destination_path.name)
        with context:
            check_download_cancelled(_current_download.get().gen)
            try:
                return original_download(incomplete_path, destination_path, *args, **kwargs)
            except (DownloadCancelled, KeyboardInterrupt):
                # The original function has closed the file; its caller still owns the HF lock.
                incomplete_path.unlink(missing_ok=True)
                raise

    http_get._wangp_download_patch = True
    hf.http_get = http_get
    hf.xet_get = xet_get
    hf._download_to_tmp_and_move = download_to_tmp_and_move


_install_sigint_handler()
