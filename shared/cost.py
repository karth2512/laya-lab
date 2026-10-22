"""Cost tracking: parameter counts, wall-clock, peak memory (sampled) and latency."""

from __future__ import annotations

import threading
import time
from types import TracebackType
from typing import Any

import numpy as np

from shared.device import memory_now


def count_params(model: Any) -> tuple[int, int]:
    """(trainable, total) parameter counts of a torch module."""
    params = list(model.parameters())
    trainable = sum(p.numel() for p in params if p.requires_grad)
    return int(trainable), int(sum(p.numel() for p in params))


class Timer:
    def __enter__(self) -> Timer:
        self.start = time.perf_counter()
        self.seconds = 0.0
        return self

    def __exit__(self, *exc: object) -> None:
        self.seconds = time.perf_counter() - self.start


class MemorySampler:
    """Samples memory in a background thread; MPS has no peak counter (REQUIREMENTS §8.2)."""

    def __init__(self, interval_s: float = 0.2) -> None:
        self.interval_s = interval_s
        self.peaks: dict[str, float] = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _sample(self) -> None:
        for k, v in memory_now().items():
            self.peaks[k] = max(self.peaks.get(k, 0.0), v)

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_s):
            self._sample()

    def __enter__(self) -> MemorySampler:
        self._sample()
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop.set()
        self._thread.join()
        self._sample()

    def metrics(self) -> dict[str, float]:
        names = {"device_mb": "device_mem", "driver_mb": "driver_mem", "rss_mb": "rss"}
        return {f"cost/peak_{names[k]}_mb": v for k, v in self.peaks.items()}


def latency_stats(latency_ms: np.ndarray) -> dict[str, float]:
    return {
        "latency/p50_ms": float(np.percentile(latency_ms, 50)),
        "latency/p95_ms": float(np.percentile(latency_ms, 95)),
    }
