"""Bootstrap confidence intervals and McNemar's test."""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest, chi2
from sklearn.metrics import accuracy_score, f1_score

EXACT_BELOW = 25  # discordant pairs below this use the exact binomial test


def _metric(name: str, y: np.ndarray, pred: np.ndarray, k: int) -> float:
    if name == "accuracy":
        return float(accuracy_score(y, pred))
    if name == "macro_f1":
        return float(f1_score(y, pred, labels=list(range(k)), average="macro", zero_division=0))
    raise ValueError(f"unsupported bootstrap metric: {name}")


def bootstrap_ci(
    y_true: np.ndarray,
    probs: np.ndarray,
    metric: str = "macro_f1",
    n: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for `accuracy` or `macro_f1` over examples."""
    y_true = np.asarray(y_true)
    pred = probs.argmax(axis=1)
    k = probs.shape[1]
    rng = np.random.default_rng(seed)
    scores = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(y_true), len(y_true))
        scores[i] = _metric(metric, y_true[idx], pred[idx], k)
    low, high = np.quantile(scores, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> tuple[float, float]:
    """McNemar's test on paired per-example correctness. Returns (statistic, p-value).

    Exact binomial (statistic = min discordant count) when b + c < EXACT_BELOW,
    otherwise chi-squared with continuity correction.
    """
    a = np.asarray(correct_a, dtype=bool)
    b_ = np.asarray(correct_b, dtype=bool)
    if a.shape != b_.shape:
        raise ValueError("paired arrays must have the same shape")
    b = int((a & ~b_).sum())
    c = int((~a & b_).sum())
    if b + c == 0:
        return 0.0, 1.0
    if b + c < EXACT_BELOW:
        return float(min(b, c)), float(binomtest(min(b, c), b + c, 0.5).pvalue)
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return float(stat), float(chi2.sf(stat, 1))
