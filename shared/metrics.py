"""Classification metrics (checked against scikit-learn in tests/test_metrics.py)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

from shared.calibration import brier, ece

EPS = 1e-7  # Jev can return exact 0.0 probabilities, so log loss clips


def compute_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    class_names: Sequence[str] | None = None,
    n_bins: int = 15,
) -> dict[str, float]:
    """Quality + calibration metrics, names without a split prefix. Raises on NaN/inf."""
    if class_names is None:
        from shared.data import CLASSES

        class_names = CLASSES
    y_true = np.asarray(y_true)
    k = len(class_names)
    labels = list(range(k))
    y_pred = probs.argmax(axis=1)
    clipped = np.clip(probs, EPS, 1.0)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)

    out: dict[str, float] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "log_loss": log_loss(y_true, clipped, labels=labels),
        "brier": brier(y_true, probs),
        "ece": ece(y_true, probs, n_bins),
        "mean_confidence": probs.max(axis=1).mean(),
    }
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    for i, name in enumerate(class_names):
        out[f"precision/{name}"] = p[i]
        out[f"recall/{name}"] = r[i]
        out[f"f1/{name}"] = f[i]

    result = {key: float(v) for key, v in out.items()}
    bad = [key for key, v in result.items() if not math.isfinite(v)]
    if bad:
        raise ValueError(f"non-finite metrics: {bad}")
    return result
