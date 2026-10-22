"""Test helpers: a recording RunContext, synthetic splits and the Predictions check."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from shared.interface import Arm, Predictions, RunContext, Split

_WORDS = [
    ["tumor", "carcinoma", "chemotherapy", "metastasis"],
    ["liver", "colon", "gastric", "pancreatitis"],
    ["brain", "seizure", "neuropathy", "dementia"],
    ["heart", "artery", "hypertension", "infarction"],
    ["syndrome", "infection", "patients", "inflammation"],
]


def synthetic_split(n: int = 50, seed: int = 0, prefix: str = "x") -> Split:
    """Balanced, learnable 5-class split with short pseudo-abstracts."""
    rng = np.random.default_rng(seed)
    labels = np.arange(n) % 5
    texts = [
        " ".join(rng.choice(_WORDS[y], size=8).tolist() + ["study", "results", "clinical"])
        for y in labels
    ]
    return Split(ids=[f"{prefix}-{i}" for i in range(n)], texts=texts, labels=labels)


class RecordingContext:
    """Builds a RunContext whose log calls are recorded in memory."""

    def __init__(self) -> None:
        self.metrics: list[tuple[str, float, int | None]] = []
        self.params: dict[str, Any] = {}
        self.artifacts: list[Path] = []
        self.dicts: dict[str, dict[str, Any]] = {}
        self.tags: dict[str, Any] = {}

    def ctx(
        self, tmp_path: Path, arm: str = "test_arm", arm_cfg: dict[str, Any] | None = None
    ) -> RunContext:
        return RunContext(
            cfg={"arm": arm_cfg or {}, "eval": {"ece_bins": 15}},
            arm=arm,
            stage="dev",
            seed=0,
            fraction=1.0,
            device="cpu",
            scratch_dir=tmp_path,
            log_metric=lambda name, value, step=None: self.metrics.append(
                (name, float(value), step)
            ),
            log_param=lambda k, v: self.params.__setitem__(k, v),
            log_artifact=lambda p: self.artifacts.append(Path(p)),
            log_dict=lambda d, name: self.dicts.__setitem__(name, d),
            set_tag=lambda k, v: self.tags.__setitem__(k, v),
        )


def check_arm(
    arm: Arm, ctx: RunContext, train: Split | None = None, val: Split | None = None
) -> Predictions:
    """Run setup/fit/predict/teardown and assert the Predictions contract on val."""
    train = train or synthetic_split(50, seed=0, prefix="tr")
    val = val or synthetic_split(20, seed=1, prefix="va")
    arm.setup(ctx)
    arm.fit(train, val, ctx)
    preds = arm.predict(val, ctx)
    preds.validate(val.ids)
    arm.teardown(ctx)
    return preds
