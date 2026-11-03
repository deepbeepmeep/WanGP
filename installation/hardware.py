"""Bounded host probes and conservative hardware-profile selection."""

import csv
import io
import json
import platform
import re
import subprocess
import os

from .catalog import os_key


def profile_for_device(device):
    vendor = device["vendor"]
    if vendor == "NVIDIA":
        capability = device.get("compute_capability")
        if capability is not None:
            return {"6.1": "GTX_10", "7.5": "RTX_20", "8.0": "RTX_30", "8.6": "RTX_30", "8.9": "RTX_40", "12.0": "RTX_50"}.get(capability)
        return None
    if vendor == "AMD":
        return "AMD" if re.fullmatch(r"gfx[0-9a-f]+", device.get("architecture", "")) else None
    if vendor == "APPLE":
        return "MPS" if device.get("architecture") == "arm64" else None
    return None


def profile_for_name(name, vendor):
    """Compatibility recommendation for the legacy CLI's name-only probe."""
    if len(name.strip().splitlines()) != 1:
        return None
    if vendor == "NVIDIA":
        match = re.search(r"\b(?:GEFORCE\s+)?RTX\s+(20|30|40|50)\d{2}\b", name.upper())
        if match:
            return "RTX_" + match[1]
        if re.search(r"\bGTX\s+10\d{2}\b", name.upper()):
            return "GTX_10"
    if vendor == "APPLE" and re.search(r"\bAPPLE M\d", name.upper()):
        return "MPS"
    if vendor == "AMD":
        return "AMD"
    return None


def parse_nvidia(text):
    devices = []
    for row in csv.reader(io.StringIO(text)):
        if not row:
            continue
        if len(row) != 6:
            raise ValueError("Unexpected nvidia-smi GPU inventory fields")
        index, uuid, bus, name, memory, capability = (value.strip() for value in row)
        devices.append({"id": uuid, "index": int(index), "pci_bus": bus, "name": name, "vendor": "NVIDIA", "memory_bytes": int(float(memory) * 1024**2), "memory_kind": "dedicated", "compute_capability": capability if re.fullmatch(r"\d+\.\d+", capability) else None})
    return devices


def parse_amd(text):
    devices, is_gpu, board = [], False, ""
    for line in text.splitlines():
        key, _, value = line.strip().partition(":")
        value = value.strip()
        if key == "Device Type":
            is_gpu, board = "CL_DEVICE_TYPE_GPU" in value, ""
        elif is_gpu and key == "Board name":
            board = value
        elif is_gpu and key == "Name" and re.fullmatch(r"gfx[0-9a-f]+", value):
            devices.append({"id": f"amd-probe-{len(devices)}", "stable_id": False, "name": board or value, "vendor": "AMD", "architecture": value, "memory_bytes": None, "memory_kind": "unknown"})
            is_gpu = False
    return devices


def detect_hardware(timeout=8):
    devices, errors = [], []

    def probe(label, command):
        try:
            return subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, check=True, **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {})).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append({"probe": label, "error": str(exc)})
            return None

    system = os_key()
    if system == "macos":
        text = probe("apple_gpu", ["system_profiler", "SPDisplaysDataType", "-json"])
        if text is not None:
            try:
                for index, entry in enumerate(json.loads(text).get("SPDisplaysDataType", [])):
                    name = entry.get("sppci_model", entry.get("_name", "Unknown"))
                    devices.append({"id": f"apple-probe-{index}", "stable_id": False, "name": name, "vendor": "APPLE" if name.startswith("Apple") else "UNKNOWN", "architecture": platform.machine(), "memory_bytes": None, "memory_kind": "unified" if name.startswith("Apple") else "unknown"})
            except (TypeError, ValueError) as exc:
                errors.append({"probe": "apple_gpu", "error": str(exc)})
    else:
        text = probe("nvidia", ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,name,memory.total,compute_cap", "--format=csv,noheader,nounits"])
        if text is not None:
            try:
                devices.extend(parse_nvidia(text))
            except ValueError as exc:
                errors.append({"probe": "nvidia", "error": str(exc)})
        text = probe("amd", ["clinfo"])
        if text is not None:
            devices.extend(parse_amd(text))
        if system == "win":
            text = probe("display_adapters", ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object Name,PNPDeviceID | ConvertTo-Json -Compress"])
            if text and text.strip():
                try:
                    entries = json.loads(text)
                    for entry in entries if isinstance(entries, list) else [entries]:
                        name = entry["Name"]
                        if any(device["name"] == name for device in devices):
                            continue
                        vendor = next((vendor for needle, vendor in (("NVIDIA", "NVIDIA"), ("AMD", "AMD"), ("Radeon", "AMD"), ("Intel", "INTEL")) if needle in name), "UNKNOWN")
                        if vendor == "AMD" and any(device["vendor"] == "AMD" for device in devices):
                            continue
                        devices.append({"id": entry["PNPDeviceID"], "name": name, "vendor": vendor, "memory_bytes": None, "memory_kind": "unknown"})
                except (KeyError, TypeError, ValueError) as exc:
                    errors.append({"probe": "display_adapters", "error": str(exc)})
    ram_bytes = None
    try:
        if system == "win":
            import ctypes
            from ctypes import wintypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD), *[(name, ctypes.c_ulonglong) for name in ("total_phys", "avail_phys", "total_page", "avail_page", "total_virtual", "avail_virtual", "avail_extended")]]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                raise OSError("GlobalMemoryStatusEx failed")
            ram_bytes = status.total_phys
        elif system == "linux":
            ram_bytes = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        else:
            text = probe("host_memory", ["sysctl", "-n", "hw.memsize"])
            ram_bytes = int(text.strip()) if text is not None else None
    except (OSError, ValueError) as exc:
        errors.append({"probe": "host_memory", "error": str(exc)})
    for device in devices:
        device["profile"] = profile_for_device(device)
    return {"platform": system, "machine": platform.machine(), "ram_bytes": ram_bytes, "devices": devices, "probe_errors": errors}


def select_device(inventory, identity=None):
    devices = inventory["devices"]
    if identity is not None:
        matches = [device for device in devices if device["id"] == identity]
        if len(matches) != 1:
            raise ValueError(f"Unknown or ambiguous device ID: {identity}")
        return matches[0]
    if len(devices) != 1:
        raise ValueError("Select an explicit device ID or hardware profile; hardware detection is empty or ambiguous")
    return devices[0]
