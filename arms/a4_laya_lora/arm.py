"""a4 Laya + LoRA: low-rank adapters on Wqkv/Wo/Wi plus a fully trained classification head."""

from __future__ import annotations

from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model

from shared.calibration import softmax
from shared.cost import count_params
from shared.interface import Arm, FitReport, Predictions, RunContext, Split
from shared.laya_backbone import load_classifier
from shared.train_loop import TrainConfig, fit_classifier, predict_classifier


class LoraArm(Arm):
    name = "a4_laya_lora"

    def setup(self, ctx: RunContext) -> None:
        self.tc = TrainConfig.from_cfg(ctx.cfg)
        base, self.tok = load_classifier(ctx.cfg, ctx.device)
        lora = ctx.arm_cfg["lora"]
        self.model = get_peft_model(
            base,
            LoraConfig(
                task_type=TaskType.SEQ_CLS,
                r=int(lora["r"]),
                lora_alpha=int(lora["alpha"]),
                lora_dropout=float(lora["dropout"]),
                target_modules=list(lora["target_modules"]),
                modules_to_save=list(lora["modules_to_save"]),
            ),
        )

    def fit(self, train: Split, val: Split, ctx: RunContext) -> FitReport:
        extra = fit_classifier(self.model, self.tok, train, val, self.tc, ctx)
        adapter_dir = ctx.scratch_dir / "lora_adapter"
        self.model.save_pretrained(str(adapter_dir))
        files = [p for p in adapter_dir.rglob("*") if p.is_file()]
        for p in files:
            ctx.log_artifact(p)
        extra["adapter_mb"] = sum(p.stat().st_size for p in files) / 2**20
        ctx.log_metric("cost/adapter_mb", extra["adapter_mb"])
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
    return LoraArm()
