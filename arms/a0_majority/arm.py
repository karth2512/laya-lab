"""a0 majority-class predictor: always predicts the most frequent training class."""

from __future__ import annotations

from typing import Any

import numpy as np

from shared.interface import Arm, FitReport, Predictions, RunContext, Split

N_CLASSES = 5


class MajorityArm(Arm):
    name = "a0_majority"

    def __init__(self) -> None:
        self.majority: int | None = None

    def setup(self, ctx: RunContext) -> None:
        pass

    def fit(self, train: Split, val: Split, ctx: RunContext) -> FitReport:
        self.majority = int(np.bincount(train.labels, minlength=N_CLASSES).argmax())
        return FitReport(
            trainable_params=0, total_params=0, extra={"majority_class": self.majority}
        )

    def predict(self, split: Split, ctx: RunContext) -> Predictions:
        assert self.majority is not None, "fit() must run before predict()"
        probs = np.zeros((len(split), N_CLASSES))
        probs[:, self.majority] = 1.0
        return Predictions(ids=list(split.ids), probs=probs)


def build(cfg: dict[str, Any]) -> Arm:
    return MajorityArm()
