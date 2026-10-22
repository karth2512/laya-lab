"""aggregate.py and error_analysis.py on synthetic runs and prediction CSVs (no MLflow)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.aggregate import build_report, mcnemar_matrix, normalize_runs, results_table
from scripts.error_analysis import build_error_report, confusion_pairs

CLASSES = ["A", "B", "C", "D", "E"]
N = 40


def _preds(n_wrong: int) -> pd.DataFrame:
    """N examples, the first n_wrong predicted as class (y + 1) % 5 with p=0.6."""
    y = np.arange(N) % 5
    pred = y.copy()
    pred[:n_wrong] = (y[:n_wrong] + 1) % 5
    probs = np.full((N, 5), 0.1)
    probs[np.arange(N), pred] = 0.6
    df = pd.DataFrame({"id": [f"test-{i}" for i in range(N)], "true_label": y, "pred_label": pred})
    for c, name in enumerate(CLASSES):
        df[f"prob_{name}"] = probs[:, c]
    return df


def _run(run_id: str, arm: str, seed: int, fraction: float, f1: float, stage: str = "final"):
    return {
        "run_id": run_id,
        "tags.arm": arm,
        "tags.stage": stage,
        "params.seed": str(seed),
        "params.fraction": str(fraction),
        "metrics.test/macro_f1": f1,
        "metrics.test/accuracy": f1,
        "metrics.cost/train_seconds": 10.0 * (seed + 1),
        "metrics.latency/p50_ms": 5.0,
    }


def _fixture() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = [
        _run("lora-s0", "a4_laya_lora", 0, 1.0, 0.8),
        _run("lora-s1", "a4_laya_lora", 1, 1.0, 0.6),
        _run("lora-s0-f01", "a4_laya_lora", 0, 0.1, 0.5),
        _run("zs", "a1_laya_zeroshot", 0, 1.0, 0.4),
        {"run_id": "parent", "tags.arm": "a4_laya_lora"},  # parent run: no seed/fraction
        {
            **_run("jev-dev", "a5_jev_api", 0, 1.0, 0.0, stage="dev"),
            "metrics.wording/plain/macro_f1": 0.5,
            "metrics.wording/short/macro_f1": 0.3,
        },
    ]
    preds = {
        "lora-s0": _preds(0),
        "lora-s1": _preds(8),
        "lora-s0-f01": _preds(20),
        "zs": _preds(30),
    }
    return pd.DataFrame(rows), preds


def test_normalize_drops_parent_runs() -> None:
    runs, _ = _fixture()
    norm = normalize_runs(runs)
    assert "parent" not in set(norm["run_id"])
    assert set(norm["method"]) == {"a4_laya_lora", "a1_laya_zeroshot", "a5_jev_api"}


def test_results_table_mean_std_and_ci() -> None:
    runs, preds = _fixture()
    final = normalize_runs(runs).query("stage == 'final'")
    table = results_table(final, preds.get, n_boot=200).set_index(["method", "fraction"])
    row = table.loc[("a4_laya_lora", 1.0)]
    assert row["seeds"] == 2
    assert row["test macro_f1"] == f"{0.7:.4f} ± {np.std([0.8, 0.6], ddof=1):.4f}"
    low, high = (float(x) for x in row["accuracy 95% CI"].strip("[]").split(", "))
    assert low <= 72 / 80 <= high  # pooled accuracy: 80 examples, 8 wrong


def test_mcnemar_matrix_pairs_zero_shot_with_every_seed() -> None:
    runs, preds = _fixture()
    final = normalize_runs(runs).query("stage == 'final'")
    mat = mcnemar_matrix(final, preds.get)
    p = float(mat.loc["a4_laya_lora", "a1_laya_zeroshot"])
    assert p < 0.001  # 22 vs 60 errors on 80 paired examples
    assert (
        mat.loc["a4_laya_lora", "a1_laya_zeroshot"] == mat.loc["a1_laya_zeroshot", "a4_laya_lora"]
    )


def test_build_report_writes_markdown_and_figures(tmp_path: Path) -> None:
    runs, preds = _fixture()
    path = build_report(runs, preds.get, tmp_path, n_boot=100)
    text = path.read_text()
    for section in ("Results table", "McNemar", "Cost", "Label-wording spread"):
        assert section in text
    assert "end-to-end" in text and "same" in text
    assert "| a5_jev_api | 2 | plain | 0.5000 | 0.2000 |" in text
    assert (tmp_path / "figures" / "learning_curves.png").stat().st_size > 0
    assert (tmp_path / "figures" / "calibration.png").stat().st_size > 0


def test_confusion_pairs_and_error_report(tmp_path: Path) -> None:
    df = _preds(10)
    y = df["true_label"].to_numpy()
    pairs = confusion_pairs(y, df["pred_label"].to_numpy(), CLASSES)
    assert pairs["count"].sum() == 10
    assert set(pairs["involves catch-all"]) == {""}  # catch-all name is not in these classes

    runs, preds = _fixture()
    texts = {f"test-{i}": "an abstract about tumors" for i in range(N)}
    path = build_error_report(runs, preds.get, tmp_path, texts)
    text = path.read_text()
    assert "## a4_laya_lora" in text and "## a1_laya_zeroshot" in text
    assert "an abstract about tumors" in text
    assert (tmp_path / "figures" / "error_confidence_a1_laya_zeroshot.png").exists()
