"""Detect GPU memory and system RAM."""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass

GiB = 1024**3


@dataclass
class Device:
    name: str
    total: int
    free: int


def nvidia_gpus() -> list[Device]:
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.rsplit(",", 2)]
        if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
            gpus.append(Device(parts[0], int(parts[1]) * 1024**2, int(parts[2]) * 1024**2))
    return gpus


def system_ram() -> tuple[int, int]:
    """(total, available) bytes of system memory."""
    system = platform.system()
    if system == "Windows":

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
        return status.ullTotalPhys, status.ullAvailPhys
    if system == "Darwin":
        total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout or 0)
        return total, total
    info = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key, value = line.split(":", 1)
                info[key] = int(value.split()[0]) * 1024
    except OSError:
        pages = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        return pages, pages
    return info.get("MemTotal", 0), info.get("MemAvailable", info.get("MemFree", 0))


def apple_unified_gpu(total_ram: int) -> Device | None:
    """Apple Silicon: Metal may use about 2/3 of RAM up to 36 GB, 3/4 above (macOS default limit)."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None
    share = 0.75 if total_ram > 36 * GiB else 2 / 3
    limit = int(total_ram * share)
    return Device("Apple GPU (unified memory)", limit, limit)


def detect() -> tuple[list[Device], int, int]:
    """(GPUs, total RAM, available RAM)."""
    total, available = system_ram()
    gpus = nvidia_gpus()
    if not gpus:
        apple = apple_unified_gpu(total)
        gpus = [apple] if apple else []
    return gpus, total, available
