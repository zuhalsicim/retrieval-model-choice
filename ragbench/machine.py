"""What the machine is, how long things take, and how much memory a process held at most."""

import platform
import sys
import time
from importlib import metadata

import numpy as np
import psutil


def peak_memory_bytes() -> int:
    """The most RAM this process has held so far (peak working set on Windows, VmHWM on Linux)."""
    info = psutil.Process().memory_info()
    if hasattr(info, "peak_wset"):  # Windows
        return int(info.peak_wset)
    import resource  # Linux and macOS; ru_maxrss is in kilobytes on Linux

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def cpu_name() -> str:
    if sys.platform == "win32":
        import winreg

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def describe() -> dict:
    import torch

    versions = {}
    for package in ("torch", "transformers", "sentence-transformers", "numpy", "huggingface_hub"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "cpu": cpu_name(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch_threads": torch.get_num_threads(),
        "gpu": False,
        "packages": versions,
    }


def latency_summary(seconds: list[float]) -> dict:
    """Median, 95th percentile and mean of a list of times, in milliseconds."""
    ms = np.asarray(seconds) * 1000
    return {
        "count": int(ms.size),
        "p50_ms": float(np.percentile(ms, 50)),
        "p95_ms": float(np.percentile(ms, 95)),
        "mean_ms": float(ms.mean()),
    }


class Timer:
    """with Timer() as t: ...; t.seconds"""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self._start
        return False
