"""a3 Laya full fine-tuning: every parameter of the Laya-initialised classifier is trainable."""

from __future__ import annotations

from typing import Any

import torch
from safetensors.torch import save_file

from shared.calibration import softmax
from shared.cost import count_params
from shared.interface import Arm, FitReport, Predictions, RunContext, Split
from shared.laya_backbone import load_classifier
from shared.train_loop import TrainConfig, fit_classifier, predict_classifier


class FullFineTuneArm(Arm):
    name = "a3_laya_full_ft"

    def setup(self, ctx: RunContext) -> None:
        self.tc = TrainConfig.from_cfg(ctx.cfg)
        self.model, self.tok = load_classifier(ctx.cfg, ctx.device)

    def fit(self, train: Split, val: Split, ctx: RunContext) -> FitReport:
        for p in self.model.parameters():
            p.requires_grad = True
        extra = fit_classifier(self.model, self.tok, train, val, self.tc, ctx)
        if ctx.cfg.get("eval", {}).get("save_model", False):
            path = ctx.scratch_dir / "model.safetensors"
            save_file({k: v.contiguous() for k, v in self.model.state_dict().items()}, str(path))
            ctx.log_artifact(path)
        trainable, total = count_params(self.model)
        return FitReport(trainable_params=trainable, total_params=total, extra=extra)

    def predict(self, split: Split, ctx: RunContext) -> Predictions:
        logits = predict_classifier(self.model, self.tok, split, self.tc, ctx.device)
        return Predictions(ids=list(split.ids), probs=softmax(logits), logits=logits)

    def teardown(self, ctx: RunContext) -> None:
        del self.model
        if ctx.device == "mps":
            torch.mps.empty_cache()


def build(cfg: dict[str, Any]) -> Arm:
    return FullFineTuneArm()
