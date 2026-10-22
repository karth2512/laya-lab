"""Device selection (MPS first, CPU fallback), dtype policy and memory probes."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from typing import Any

import psutil


def _torch() -> Any:
    try:
        import torch
    except ImportError:
        return None
    return torch


def select_device(pref: str = "auto") -> str:
    """`LAYA_BENCH_DEVICE` env wins, then `pref`; `auto`/`mps` fall back to cpu when MPS is absent."""
    pref = os.environ.get("LAYA_BENCH_DEVICE", pref)
    if pref == "cpu":
        return "cpu"
    torch = _torch()
    if torch is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@contextlib.contextmanager
def autocast(device: str, enabled: bool = True) -> Iterator[None]:
    """bf16 autocast on MPS (REQUIREMENTS §8.1); a no-op elsewhere or when disabled."""
    torch = _torch()
    if torch is None or not enabled or device != "mps":
        yield
        return
    with torch.autocast("mps", dtype=torch.bfloat16):
        yield


def memory_now() -> dict[str, float]:
    """Current memory in MB: process RSS always, MPS tensor/driver memory when on MPS."""
    out = {"rss_mb": psutil.Process().memory_info().rss / 2**20}
    torch = _torch()
    if torch is not None and torch.backends.mps.is_available():
        out["device_mb"] = torch.mps.current_allocated_memory() / 2**20
        out["driver_mb"] = torch.mps.driver_allocated_memory() / 2**20
    return out
