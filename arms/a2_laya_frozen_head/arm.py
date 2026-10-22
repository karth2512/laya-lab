"""a2 frozen Laya encoder + trained head: pooled features are cached, then only the head trains."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from shared.calibration import softmax
from shared.cost import count_params
from shared.device import autocast
from shared.interface import Arm, FitReport, Predictions, RunContext, Split
from shared.laya_backbone import head_logits, load_classifier, pool
from shared.train_loop import Batch, TrainConfig, encode, fit, predict_logits, token_batches


class FrozenHeadArm(Arm):
    name = "a2_laya_frozen_head"

    def setup(self, ctx: RunContext) -> None:
        self.tc = TrainConfig.from_cfg(ctx.cfg)
        self.model, self.tok = load_classifier(ctx.cfg, ctx.device)
        for p in self.model.model.parameters():
            p.requires_grad = False

    def _features(self, split: Split, ctx: RunContext) -> tuple[torch.Tensor, float]:
        """Pooled encoder features (fp32, CPU) in split order, plus the truncation rate."""
        enc = encode(self.tok, split, self.tc.max_length)
        feats = torch.zeros(len(split), self.model.config.hidden_size)
        self.model.eval()
        with torch.no_grad(), autocast(ctx.device, self.tc.bf16):
            for b in token_batches(enc, self.tc.eval_max_tokens_per_batch, self.tok.pad_token_id):
                inputs = {k: v.to(ctx.device) for k, v in b.inputs.items()}
                hidden = self.model.model(**inputs)[0]
                feats[b.idx] = pool(self.model, hidden, inputs["attention_mask"]).float().cpu()
        return feats, enc.truncated_pct

    def _batches(
        self, feats: torch.Tensor, labels: np.ndarray, rng: np.random.Generator | None
    ) -> list[Batch]:
        order = rng.permutation(len(labels)) if rng is not None else np.arange(len(labels))
        size = self.tc.effective_batch_size
        return [
            Batch(idx, {"features": feats[idx]}, torch.as_tensor(labels[idx], dtype=torch.long))
            for idx in (order[i : i + size] for i in range(0, len(order), size))
        ]

    def _forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        return head_logits(self.model, inputs["features"])

    def fit(self, train: Split, val: Split, ctx: RunContext) -> FitReport:
        train_x, train_trunc = self._features(train, ctx)
        val_x, val_trunc = self._features(val, ctx)
        ctx.log_metric("data/truncated_pct", train_trunc)
        ctx.log_metric("data/truncated_pct_val", val_trunc)
        train_y, val_y = np.asarray(train.labels), np.asarray(val.labels)
        extra = fit(
            self.model,
            self._forward,
            lambda epoch: self._batches(train_x, train_y, np.random.default_rng([ctx.seed, epoch])),
            self._batches(val_x, val_y, None),
            val_y,
            self.tc,
            ctx,
        )
        trainable, total = count_params(self.model)
        return FitReport(trainable_params=trainable, total_params=total, extra=extra)

    def predict(self, split: Split, ctx: RunContext) -> Predictions:
        feats, _ = self._features(split, ctx)
        labels = np.zeros(len(split), dtype=int)  # unused; predict never needs labels
        logits = predict_logits(
            self.model, self._forward, self._batches(feats, labels, None), ctx.device, False
        )
        return Predictions(ids=list(split.ids), probs=softmax(logits), logits=logits)

    def teardown(self, ctx: RunContext) -> None:
        del self.model
        if ctx.device == "mps":
            torch.mps.empty_cache()


def build(cfg: dict[str, Any]) -> Arm:
    return FrozenHeadArm()
