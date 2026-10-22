"""Calibration: ECE, Brier score and temperature scaling."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    out: np.ndarray = e / e.sum(axis=1, keepdims=True)
    return out


def ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error of the top-1 confidence, equal-width bins on (0, 1]."""
    y_true = np.asarray(y_true)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.searchsorted(edges, conf, side="left") - 1, 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(total)


def brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    """Multi-class Brier score: mean over examples of the squared error summed over classes."""
    onehot = np.eye(probs.shape[1])[np.asarray(y_true)]
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def fit_temperature(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Temperature T > 0 minimizing NLL of softmax(logits / T) on the given (val) set."""
    y_true = np.asarray(y_true)
    idx = np.arange(len(y_true))

    def nll(log_t: float) -> float:
        p = softmax(logits, float(np.exp(log_t)))
        return float(-np.log(np.clip(p[idx, y_true], 1e-12, None)).mean())

    res = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(res.x))
