"""Bounded loopback readiness checks tied to the launched Windows process tree."""

import ctypes
from http.client import HTTPConnection, HTTPException
import os
import socket
import struct
import time

from .state import InstallError


def require_free_port(port):
    if not isinstance(port, int) or not 1024 <= port <= 65535:
        raise InstallError("invalid_port", "Choose a port between 1024 and 65535")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        if os.name == "nt":
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError as exc:
            raise InstallError("port_busy", f"Port {port} is already in use. Choose another port; the existing process was left alone") from exc


def owned_listener(root_pid, port):
    if os.name != "nt":
        return None
    from ctypes import wintypes

    class ProcessEntry(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("usage", wintypes.DWORD), ("pid", wintypes.DWORD), ("heap", ctypes.c_size_t), ("module", wintypes.DWORD), ("threads", wintypes.DWORD), ("parent", wintypes.DWORD), ("priority", wintypes.LONG), ("flags", wintypes.DWORD), ("exe", wintypes.WCHAR * 260)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    snapshot = kernel.CreateToolhelp32Snapshot(2, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    parents = {}
    try:
        entry = ProcessEntry()
        entry.size = ctypes.sizeof(entry)
        available = kernel.Process32FirstW(snapshot, ctypes.byref(entry))
        while available:
            parents[entry.pid] = entry.parent
            available = kernel.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        kernel.CloseHandle(snapshot)
    owned = {root_pid}
    while additions := {pid for pid, parent in parents.items() if parent in owned} - owned:
        owned.update(additions)
    iphelper = ctypes.WinDLL("iphlpapi")
    size = wintypes.DWORD()
    iphelper.GetExtendedTcpTable.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD), wintypes.BOOL, wintypes.ULONG, ctypes.c_int, wintypes.ULONG]
    iphelper.GetExtendedTcpTable(None, ctypes.byref(size), False, 2, 3, 0)
    buffer = ctypes.create_string_buffer(size.value)
    if iphelper.GetExtendedTcpTable(buffer, ctypes.byref(size), False, 2, 3, 0):
        return None
    count = struct.unpack_from("<I", buffer)[0]
    for index in range(count):
        state, address, local_port, remote, remote_port, pid = struct.unpack_from("<IIIIII", buffer, 4 + index * 24)
        if socket.ntohs(local_port & 65535) == port and pid in owned and address in (0, 0x0100007F):
            return pid
    return None


def wait_for_server(root_pid, port, finished, emit, timeout=180):
    url = f"http://127.0.0.1:{port}"
    # Gradio renders its homepage for its own startup check. Ask for the
    # lightweight app ID so Manager does not trigger competing page renders.
    path = "/gradio_api/app_id"
    deadline = time.monotonic() + timeout
    while not finished.wait(0.4) and time.monotonic() < deadline:
        try:
            if (pid := owned_listener(root_pid, port)) is None:
                continue
            conn = HTTPConnection("127.0.0.1", port, timeout=2)
            try:
                conn.request("GET", path)
                status = conn.getresponse().status
            finally:
                conn.close()
            if status in (404, 405) and path != "/":
                path = "/"  # Non-Gradio sample servers and older installations.
                continue
            if 200 <= status < 400 or status == 401:
                emit({"type": "server_ready", "url": url, "pid": pid, "validation": "owned_listener_http", "http_status": status})
                return
        except (OSError, HTTPException):
            continue
    if not finished.is_set():
        emit({"type": "notice", "title": "Still Starting", "message": "The process is running, but its own web server has not become ready. Check the logs or stop and retry."})
