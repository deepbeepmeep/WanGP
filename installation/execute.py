"""Bounded subprocess output and cancellation of the owned process tree."""

import codecs
import ctypes
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time

from .state import InstallError, plain_path


class Cancelled(InstallError):
    def __init__(self):
        super().__init__("cancelled", "Operation cancelled")


class WindowsJob:
    def __init__(self, process):
        from ctypes import wintypes

        class BasicLimit(ctypes.Structure):
            _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong), ("flags", wintypes.DWORD), ("min_working", ctypes.c_size_t), ("max_working", ctypes.c_size_t), ("active", wintypes.DWORD), ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD), ("scheduling", wintypes.DWORD)]

        class IoCounters(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]

        class ExtendedLimit(ctypes.Structure):
            _fields_ = [("basic", BasicLimit), ("io", IoCounters), ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t), ("peak_process", ctypes.c_size_t), ("peak_job", ctypes.c_size_t)]

        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimit()
        limits.basic.flags = 0x2000
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)) or not self.kernel.AssignProcessToJobObject(self.handle, process._handle):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def terminate(self):
        self.kernel.TerminateJobObject(self.handle, 1)

    def close(self):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


class ProcessRunner:
    def __init__(self, log_path=None, cancelled=None, events=None, max_log_bytes=4 * 1024 * 1024):
        self.log_path = plain_path(log_path) if log_path else None
        self.cancelled = cancelled or (lambda: False)
        self.events = events or (lambda event: None)
        self.max_log_bytes = max_log_bytes

    def check_cancelled(self):
        if self.cancelled():
            raise Cancelled()

    def run(self, arguments, *, env=None, cwd=None, timeout=1800, check=True, capture=False, on_start=None, emit_output=True):
        self.check_cancelled()
        job = None
        options = {"env": env, "cwd": cwd, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE if capture else subprocess.STDOUT}
        if os.name == "nt":
            # The helper cannot start the requested command until assigned to its job.
            wrapper = "import json,subprocess,sys; sys.exit(subprocess.call(json.load(sys.stdin)))"
            options.update(stdin=subprocess.PIPE, creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
            process = subprocess.Popen([sys.executable, "-I", "-u", "-c", wrapper], **options)
            try:
                job = WindowsJob(process)
                process.stdin.write(json.dumps(list(map(str, arguments))).encode())
                process.stdin.close()
            except BaseException:
                process.kill()
                process.wait()
                if job:
                    job.close()
                raise
        else:
            process = subprocess.Popen(list(map(str, arguments)), start_new_session=True, **options)
        output = queue.Queue(maxsize=128)

        def read(stream, is_stdout):
            try:
                while chunk := os.read(stream.fileno(), 4096):
                    output.put((is_stdout, chunk))
            finally:
                stream.close()
                output.put((is_stdout, None))

        streams = [threading.Thread(target=read, args=(process.stdout, True), daemon=True)]
        if capture:
            streams.append(threading.Thread(target=read, args=(process.stderr, False), daemon=True))
        for thread in streams:
            thread.start()
        log = None
        captured, tail, closed = bytearray(), bytearray(), 0
        start, last_event = time.monotonic(), 0
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        pending_text = ""
        def flush_events():
            nonlocal pending_text, last_event
            if pending_text:
                if emit_output:
                    self.events({"type": "log", "text": pending_text})
                pending_text = ""
                last_event = time.monotonic()
        try:
            if on_start:
                on_start(process.pid)
            if self.log_path:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                log = open(self.log_path, "ab")
            written = log.tell() if log else 0
            while closed < len(streams) or process.poll() is None:
                self.check_cancelled()
                if timeout is not None and time.monotonic() - start > timeout:
                    raise InstallError("process_timeout", f"Command exceeded {timeout} seconds")
                if process.poll() is not None:
                    if job:
                        job.terminate()
                    else:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                try:
                    is_stdout, chunk = output.get(timeout=0.1)
                except queue.Empty:
                    if time.monotonic() - last_event >= 1 / 3:
                        flush_events()
                    continue
                if chunk is None:
                    closed += 1
                    continue
                tail = (tail + chunk)[-8192:]
                if capture and is_stdout:
                    captured.extend(chunk)
                    if len(captured) > 2 * 1024 * 1024:
                        raise InstallError("output_limit", "Command result exceeded the 2 MiB capture limit")
                if log and written < self.max_log_bytes:
                    piece = chunk[:self.max_log_bytes - written]
                    log.write(piece)
                    written += len(piece)
                pending_text += decoder.decode(chunk)
                if time.monotonic() - last_event >= 1 / 3 or len(pending_text) >= 32768:
                    flush_events()
            code = process.wait()
            if check and code:
                raise InstallError("command_failed", f"Command exited with {code}: {tail.decode('utf-8', errors='replace')}")
            return subprocess.CompletedProcess(arguments, code, captured.decode("utf-8", errors="replace") if capture else "")
        finally:
            pending_text += decoder.decode(b"", final=True)
            flush_events()
            if job:
                job.terminate()
                job.close()
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            process.wait()
            # Drain readers after termination; never leave a bounded queue blocking them.
            while any(thread.is_alive() for thread in streams):
                try:
                    output.get(timeout=0.1)
                except queue.Empty:
                    pass
            if log:
                log.flush()
                log.close()
