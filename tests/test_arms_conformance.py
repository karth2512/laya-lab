"""Interface conformance: CPU-loadable arms run fit/predict and meet the Predictions contract."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from tests.helpers import RecordingContext, check_arm, synthetic_split

CPU_ARMS = ["a0_majority", "a0_tfidf_logreg"]


def _build(name: str) -> tuple[object, dict]:
    arm_cfg = yaml.safe_load((Path("arms") / name / "config.yaml").read_text()) or {}
    module = importlib.import_module(f"arms.{name}.arm")
    return module.build({"arm": arm_cfg}), arm_cfg


@pytest.mark.parametrize("name", CPU_ARMS)
def test_arm_conformance(name: str, tmp_path: Path) -> None:
    arm, arm_cfg = _build(name)
    rec = RecordingContext()
    preds = check_arm(arm, rec.ctx(tmp_path, arm=name, arm_cfg=arm_cfg))  # type: ignore[arg-type]
    assert preds.probs.shape == (20, 5)


def test_majority_predicts_train_majority(tmp_path: Path) -> None:
    arm, _ = _build("a0_majority")
    train = synthetic_split(50)
    train.labels[:] = 3
    ctx = RecordingContext().ctx(tmp_path)
    arm.fit(train, synthetic_split(10, prefix="v"), ctx)  # type: ignore[attr-defined]
    preds = arm.predict(synthetic_split(10, prefix="v"), ctx)  # type: ignore[attr-defined]
    assert (preds.probs.argmax(1) == 3).all()


def test_tfidf_learns_and_logs_trials(tmp_path: Path) -> None:
    arm, arm_cfg = _build("a0_tfidf_logreg")
    rec = RecordingContext()
    preds = check_arm(arm, rec.ctx(tmp_path, arm_cfg=arm_cfg))  # type: ignore[arg-type]
    val = synthetic_split(20, seed=1, prefix="va")
    assert (preds.probs.argmax(1) == val.labels).mean() > 0.8
    assert preds.logits is not None
    trials = [m for m in rec.metrics if m[0] == "hp/val_macro_f1"]
    assert 1 <= len(trials) <= len(arm_cfg["trials"]) == 6
    assert np.isfinite([m[1] for m in trials]).all()
