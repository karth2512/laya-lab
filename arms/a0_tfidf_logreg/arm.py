"""a0 TF-IDF + logistic regression, HPs picked on val from a fixed trial list."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline, make_pipeline

from shared.interface import Arm, FitReport, Predictions, RunContext, Split

N_CLASSES = 5


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class TfidfLogRegArm(Arm):
    name = "a0_tfidf_logreg"

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.model: Pipeline | None = None

    def _pipeline(self, trial: dict[str, Any], seed: int) -> Pipeline:
        return make_pipeline(
            TfidfVectorizer(
                ngram_range=(1, int(trial["ngram_max"])),
                min_df=int(trial["min_df"]),
                sublinear_tf=bool(self.cfg.get("sublinear_tf", True)),
            ),
            LogisticRegression(
                C=float(trial["C"]), max_iter=int(self.cfg.get("max_iter", 2000)), random_state=seed
            ),
        )

    def setup(self, ctx: RunContext) -> None:
        pass

    def fit(self, train: Split, val: Split, ctx: RunContext) -> FitReport:
        if len(np.unique(train.labels)) != N_CLASSES:
            raise ValueError("training subset must contain all 5 classes")
        best: tuple[float, int, Pipeline] | None = None
        for i, trial in enumerate(self.cfg["trials"]):
            try:
                model = self._pipeline(trial, ctx.seed).fit(train.texts, train.labels)
            except ValueError:  # e.g. min_df removed every term on a tiny fraction
                continue
            score = float(f1_score(val.labels, model.predict(val.texts), average="macro"))
            ctx.log_metric("hp/val_macro_f1", score, step=i)
            if best is None or score > best[0]:
                best = (score, i, model)
        if best is None:
            raise RuntimeError("no TF-IDF trial could be fitted")
        score, i, self.model = best
        clf: LogisticRegression = self.model[-1]
        n_params = int(clf.coef_.size + clf.intercept_.size)
        return FitReport(
            trainable_params=n_params,
            total_params=n_params,
            extra={"best_trial": i, **self.cfg["trials"][i], "hp_val_macro_f1": score},
        )

    def predict(self, split: Split, ctx: RunContext) -> Predictions:
        assert self.model is not None, "fit() must run before predict()"
        logits = np.asarray(self.model.decision_function(split.texts), dtype=np.float64)
        return Predictions(ids=list(split.ids), probs=_softmax(logits), logits=logits)


def build(cfg: dict[str, Any]) -> Arm:
    return TfidfLogRegArm(cfg.get("arm", {}))
