"""A throwaway project (raw CSVs, manifest, MLflow backend) and a stub arm for runner tests."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from shared import data, splits
from shared.interface import Arm, FitReport, Predictions, RunContext, Split
from tests.helpers import synthetic_split

STUB_ARM = "zz_stub_arm"


class StubArm(Arm):
    """Keyword-free 'classifier': random logits seeded per id, plus the fit-time class prior."""

    name = STUB_ARM

    def setup(self, ctx: RunContext) -> None:
        self.prior = np.zeros(5)

    def fit(self, train: Split, val: Split, ctx: RunContext) -> FitReport:
        self.prior = np.bincount(train.labels, minlength=5) / len(train)
        ctx.log_metric("train/loss", 1.0, step=0)
        return FitReport(trainable_params=5, total_params=5, extra={"note": "stub"})

    def predict(self, split: Split, ctx: RunContext) -> Predictions:
        rng = np.random.default_rng(len(split))
        logits = rng.normal(size=(len(split), 5)) + np.log(self.prior + 1e-3)
        probs = np.exp(logits) / np.exp(logits).sum(1, keepdims=True)
        return Predictions(ids=list(split.ids), probs=probs, logits=logits)


def install_stub_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType(f"arms.{STUB_ARM}.arm")
    module.build = lambda cfg: StubArm()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, f"arms.{STUB_ARM}.arm", module)


def make_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Write raw CSVs + manifest under tmp_path; return config overrides pointing at them."""
    raw = tmp_path / "raw"
    raw.mkdir()
    for name, n, seed in (("train", 100, 0), ("test", 25, 1)):
        s = synthetic_split(n, seed=seed)
        pd.DataFrame({"condition_label": s.labels + 1, "medical_abstract": s.texts}).to_csv(
            raw / f"{name}.csv", index=False
        )
    train_df, test_df = data.load_frames(raw, include_test=True)
    assert test_df is not None
    manifest = splits.make_manifest(
        train_df["id"].tolist(), train_df["label"].to_numpy(), test_df["id"].tolist()
    )
    splits.save_manifest(manifest, tmp_path / "split.json")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    install_stub_arm(monkeypatch)
    return [
        f"dataset.raw_dir={raw}",
        f"dataset.split_manifest={tmp_path / 'split.json'}",
        f"mlflow.tracking_uri=sqlite:///{tmp_path / 'mlflow.db'}",
        f"mlflow.artifact_location={tmp_path / 'mlartifacts'}",
        "eval.latency_examples=3",
        "eval.bootstrap_resamples=50",
        "device=cpu",
    ]
