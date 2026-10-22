import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, log_loss, matthews_corrcoef

from shared.calibration import brier, ece, fit_temperature, softmax
from shared.metrics import compute_metrics
from shared.stats import bootstrap_ci, mcnemar

NAMES = ["a", "b", "c", "d", "e"]


def _data(n: int = 200, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 5, n)
    logits = rng.normal(size=(n, 5))
    logits[np.arange(n), y] += 1.5
    return y, softmax(logits)


def test_sklearn_parity() -> None:
    y, p = _data()
    pred = p.argmax(1)
    m = compute_metrics(y, p, class_names=NAMES)
    assert m["accuracy"] == pytest.approx(accuracy_score(y, pred))
    assert m["macro_f1"] == pytest.approx(f1_score(y, pred, average="macro"))
    assert m["weighted_f1"] == pytest.approx(f1_score(y, pred, average="weighted"))
    assert m["mcc"] == pytest.approx(matthews_corrcoef(y, pred))
    assert m["log_loss"] == pytest.approx(log_loss(y, p), rel=1e-5)
    assert m["f1/c"] == pytest.approx(f1_score(y, pred, average=None)[2])
    assert {"precision/a", "recall/e", "brier", "ece", "mean_confidence"} <= m.keys()


def test_log_loss_clips_exact_zero() -> None:
    y = np.array([0, 1])
    p = np.array([[0.0, 1.0, 0, 0, 0], [0.0, 1.0, 0, 0, 0]])
    assert np.isfinite(compute_metrics(y, p, class_names=NAMES)["log_loss"])


def test_nan_raises() -> None:
    y = np.array([0, 1])
    p = np.full((2, 5), np.nan)
    with pytest.raises(ValueError):
        compute_metrics(y, p, class_names=NAMES)


def test_brier_hand_computed() -> None:
    y = np.array([0, 1])
    p = np.array([[0.8, 0.2], [0.6, 0.4]])
    # (0.04 + 0.04) and (0.36 + 0.36) -> mean 0.4
    assert brier(y, p) == pytest.approx(0.4)


def test_ece_hand_computed() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([[0.9, 0.1], [0.9, 0.1], [0.6, 0.4], [0.6, 0.4]])
    # bin conf .9: acc 1 -> gap .1; bin conf .6: acc 0 -> gap .6; ECE = .5*.1 + .5*.6
    assert ece(y, p, n_bins=10) == pytest.approx(0.35)
    perfect = np.eye(2)[y]
    assert ece(y, perfect) == pytest.approx(0.0)


def test_temperature_recovers_known_t() -> None:
    rng = np.random.default_rng(1)
    logits = rng.normal(scale=3.0, size=(5000, 5))
    true_t = 2.5
    p = softmax(logits, true_t)
    y = np.array([rng.choice(5, p=row) for row in p])
    assert fit_temperature(logits, y) == pytest.approx(true_t, rel=0.1)


def test_bootstrap_ci_brackets_point_estimate() -> None:
    y, p = _data(300)
    low, high = bootstrap_ci(y, p, "accuracy", n=300, seed=0)
    acc = accuracy_score(y, p.argmax(1))
    assert low < acc < high
    assert bootstrap_ci(y, p, "macro_f1", n=50, seed=3) == bootstrap_ci(
        y, p, "macro_f1", n=50, seed=3
    )


def test_mcnemar() -> None:
    a = np.ones(100, dtype=bool)
    assert mcnemar(a, a) == (0.0, 1.0)
    b = a.copy()
    b[:40] = False  # 40 discordant all one way -> chi2 path, tiny p
    stat, p = mcnemar(a, b)
    assert stat == pytest.approx(39**2 / 40)
    assert p < 1e-6
    c = a.copy()
    c[:3] = False  # exact path
    _, p_small = mcnemar(a, c)
    assert p_small == pytest.approx(0.25)
