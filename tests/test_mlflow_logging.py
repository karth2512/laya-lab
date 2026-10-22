"""A temp-backend run logs the expected params, tags, metrics and artifacts (§10.6)."""

from __future__ import annotations

from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient

from shared import runner
from shared.mlflow_utils import ChildRun
from tests.runtime_helpers import STUB_ARM, make_project

TAGS = [
    "arm", "stage", "git_sha", "git_dirty", "data_hash", "hostname", "device_name",
    "macos_version", "torch_backend", "mps_fallback_used", "on_ac_power", "python_version",
    "library_versions",
]  # fmt: skip
METRICS = [
    "val/macro_f1", "val/accuracy", "val/ece", "val/brier", "val/temperature", "val/ece_ts",
    "val/f1/Neoplasms", "test/macro_f1", "test/log_loss", "test/mcc", "test/ece_ts",
    "test/bootstrap_ci_low/accuracy", "test/bootstrap_ci_high/macro_f1", "test/mean_confidence",
    "cost/train_seconds", "cost/trainable_params", "cost/total_params", "cost/peak_rss_mb",
    "latency/p50_ms", "latency/p95_ms", "throughput/examples_per_sec", "data/train_size",
    "data/truncated_pct", "train/loss",
]  # fmt: skip
ARTIFACTS = [
    "predictions_val.csv", "predictions_test.csv", "classification_report.json",
    "confusion_matrix.png", "calibration_curve.png", "resolved_config.yaml", "pip_freeze.txt",
    "split_manifest_hash.txt", "fit_report.json",
]  # fmt: skip


def test_final_run_logs_everything(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    overrides = make_project(tmp_path, monkeypatch)
    runner.run(STUB_ARM, "final", 1, 0.5, overrides, variant="v1")

    client = MlflowClient(f"sqlite:///{tmp_path / 'mlflow.db'}")
    exp = client.get_experiment_by_name("laya-medabs-bench")
    assert exp is not None
    runs = {r.info.run_name: r for r in client.search_runs([exp.experiment_id])}
    child = runs[f"{STUB_ARM}/final/s1/f0.5/v1"]
    parent = runs[STUB_ARM]
    assert child.data.tags["mlflow.parentRunId"] == parent.info.run_id
    assert child.info.status == "FINISHED"

    assert not set(TAGS) - set(child.data.tags)
    assert not set(METRICS) - set(child.data.metrics)
    params = child.data.params
    assert params["seed"] == "1" and params["fraction"] == "0.5" and params["stage"] == "final"
    assert params["split_hash"] == child.data.tags["data_hash"]
    artifacts = {f.path for f in client.list_artifacts(child.info.run_id)}
    assert not set(ARTIFACTS) - artifacts

    # A second run of the same arm reuses the parent and is found as completed.
    runner.run(STUB_ARM, "dev", 0, 1.0, overrides)
    tracker = runner.tracker_from_cfg(runner.load_config(STUB_ARM, overrides))
    assert tracker.completed_run_names(STUB_ARM) == {
        f"{STUB_ARM}/final/s1/f0.5/v1",
        f"{STUB_ARM}/dev/s0/f1",
    }
    assert tracker.parent_run_id(STUB_ARM) == parent.info.run_id


def test_non_finite_metric_raises() -> None:
    class Dummy:
        def log_metric(self, *a: object, **k: object) -> None:
            raise AssertionError("must not reach mlflow")

    child = ChildRun(Dummy(), "run")  # type: ignore[arg-type]
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="not finite"):
            child.log_metric("val/x", bad)
