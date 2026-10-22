"""The contract every arm implements (REQUIREMENTS §4.3). No arm-specific logic here."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

Stage = Literal["dev", "final"]


@dataclass
class Split:
    ids: list[str]
    texts: list[str]
    labels: np.ndarray  # int in [0, 4]; for the test split at stage=dev this is never built

    def __len__(self) -> int:
        return len(self.ids)

    def subset(self, idx: list[int] | np.ndarray) -> Split:
        idx = list(idx)
        return Split(
            ids=[self.ids[i] for i in idx],
            texts=[self.texts[i] for i in idx],
            labels=self.labels[idx],
        )


@dataclass
class Predictions:
    ids: list[str]
    probs: np.ndarray  # (N, 5), rows sum to 1
    logits: np.ndarray | None = None  # (N, 5) when the arm has them; enables temperature scaling
    latency_ms: np.ndarray | None = None  # per-example, when measured by the arm

    def validate(self, expected_ids: list[str], n_classes: int = 5) -> None:
        if self.ids != expected_ids:
            raise ValueError("prediction ids do not match the split ids")
        if self.probs.shape != (len(expected_ids), n_classes):
            want = (len(expected_ids), n_classes)
            raise ValueError(f"probs shape {self.probs.shape} != {want}")
        if not np.all(np.isfinite(self.probs)) or not np.allclose(self.probs.sum(1), 1, atol=1e-4):
            raise ValueError("probs must be finite and rows must sum to 1")
        if self.logits is not None and self.logits.shape != self.probs.shape:
            raise ValueError("logits shape must match probs")


@dataclass
class FitReport:
    trainable_params: int | None = None
    total_params: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunContext:
    """Everything an arm may touch during a run. Arms log only through these callables."""

    cfg: dict[str, Any]  # resolved config: base.yaml + arm config.yaml + overrides
    arm: str
    stage: Stage
    seed: int
    fraction: float
    device: str  # "mps" or "cpu", chosen by shared.device
    scratch_dir: Path
    log_metric: Callable[..., None]  # (name, value, step=None)
    log_param: Callable[[str, Any], None]
    log_artifact: Callable[[str | Path], None]
    log_dict: Callable[[dict[str, Any], str], None]  # (data, artifact_file)
    set_tag: Callable[[str, Any], None]

    @property
    def arm_cfg(self) -> dict[str, Any]:
        """The arm's own section of the resolved config (keys from its config.yaml)."""
        return self.cfg.get("arm", {})


class Arm(ABC):
    name: str

    @abstractmethod
    def setup(self, ctx: RunContext) -> None:
        """Load models, clients, caches. No data access."""

    @abstractmethod
    def fit(self, train: Split, val: Split, ctx: RunContext) -> FitReport:
        """Train or select settings using train/val ONLY (a1/a5 may use val to pick wording)."""

    @abstractmethod
    def predict(self, split: Split, ctx: RunContext) -> Predictions: ...

    def teardown(self, ctx: RunContext) -> None:  # noqa: B027 - optional hook
        pass
