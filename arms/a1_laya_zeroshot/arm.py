"""a1: Laya zero-shot, one typed choice question over the 5 classes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml
from sklearn.metrics import f1_score

from shared.interface import Arm, FitReport, Predictions, RunContext, Split

REPO_ROOT = Path(__file__).resolve().parents[2]
QID = "category"


def load_variants(path: str | Path, names: list[str] | None = None) -> dict[str, dict[str, Any]]:
    """Wording variants: name -> {question, options (5 strings in class-id order)}."""
    p = Path(path)
    variants: dict[str, dict[str, Any]] = yaml.safe_load(
        (p if p.is_absolute() else REPO_ROOT / p).read_text()
    )["variants"]
    if names:
        variants = {n: variants[n] for n in names}
    for name, v in variants.items():
        if len(v["options"]) != 5 or len(set(v["options"])) != 5:
            raise ValueError(f"variant {name!r} needs 5 distinct options")
    return variants


class LayaZeroShot(Arm):
    name = "a1_laya_zeroshot"

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.arm_cfg: dict[str, Any] = cfg.get("arm", {})
        self.variants = load_variants(
            self.arm_cfg.get("labels_file", "configs/prompts/labels.yaml"),
            self.arm_cfg.get("variants"),
        )
        self.selected: str | None = None
        self.agent: Any = None

    def setup(self, ctx: RunContext) -> None:
        import laya

        laya_cfg = self.cfg.get("laya", {})
        self.agent = laya.load(
            laya_cfg.get("hub_id", "convaiinnovations/laya"),
            device=ctx.device,
            revision=laya_cfg.get("revision"),
        )

    def _question(self, variant: str, reverse: bool = False) -> dict[str, Any]:
        v = self.variants[variant]
        q: dict[str, Any] = {
            "type": "choice",
            "instructions": v["question"],
            "criteria": v["options"],
        }
        if reverse:
            q["option_order"] = list(range(len(v["options"])))[::-1]
        return {QID: q}

    def _score(self, texts: list[str], variant: str, reverse: bool = False) -> np.ndarray:
        """(N, 5) probabilities in class-id order, rows renormalised."""
        options = self.variants[variant]["options"]
        results = self.agent.predict_batch(
            texts,
            self._question(variant, reverse),
            batch_size=int(self.arm_cfg.get("batch_size", 16)),
            max_len=int(self.arm_cfg.get("max_len", 512)),
        )
        probs = np.array(
            [[r["answers"][QID]["probabilities"][o] for o in options] for r in results],
            dtype=np.float64,
        )
        # The runtime rounds to 4 decimals; fall back to uniform for an all-zero row.
        probs[probs.sum(1) <= 0] = 1.0
        return probs / probs.sum(1, keepdims=True)

    def _truncated_pct(self, texts: list[str], variant: str) -> float | None:
        """% of texts longer than the state room the question leaves (about 472 tokens)."""
        try:
            from laya.common import state_room

            tok = self.agent.tok
            v = self.variants[variant]
            room = state_room(
                tok,
                {"t": "choice", "ins": v["question"], "crit": {o: None for o in v["options"]}},
                int(self.arm_cfg.get("max_len", 512)),
                int(self.agent.cfg.get("head_max_len", 192)),
            )
            lengths = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts]
        except Exception:  # truncation stats are best-effort; a stub or API change skips them
            return None
        return 100.0 * float(np.mean([n > room for n in lengths]))

    def fit(self, train: Split, val: Split, ctx: RunContext) -> FitReport:
        scores: dict[str, float] = {}
        for variant in self.variants:
            preds = self._score(val.texts, variant).argmax(1)
            scores[variant] = float(f1_score(val.labels, preds, average="macro"))
            ctx.log_metric(f"wording/{variant}/val_macro_f1", scores[variant])
        self.selected = max(scores, key=lambda k: scores[k])
        spread = max(scores.values()) - min(scores.values())
        ctx.log_metric("wording/spread_val_macro_f1", spread)
        ctx.log_param("wording/selected", self.selected)
        extra: dict[str, Any] = {"selected_wording": self.selected, "wording_val_macro_f1": scores}
        if self.arm_cfg.get("option_order_check", True):
            rev = self._score(val.texts, self.selected, reverse=True).argmax(1)
            rev_f1 = float(f1_score(val.labels, rev, average="macro"))
            ctx.log_metric("wording/order_reversed_val_macro_f1", rev_f1)
            extra["order_reversed_val_macro_f1"] = rev_f1
        ctx.log_dict(extra, "wording_selection.json")
        # No training data, so truncation is reported on val (predict is also called per example).
        truncated = self._truncated_pct(val.texts, self.selected)
        if truncated is not None:
            ctx.log_metric("data/truncated_pct", truncated)
        model = getattr(self.agent, "model", None)
        total = sum(p.numel() for p in model.parameters()) if model is not None else None
        return FitReport(trainable_params=0, total_params=total, extra=extra)

    def predict(self, split: Split, ctx: RunContext) -> Predictions:
        if self.selected is None:
            raise RuntimeError("fit() must select a wording before predict()")
        return Predictions(ids=list(split.ids), probs=self._score(split.texts, self.selected))

    def teardown(self, ctx: RunContext) -> None:
        self.agent = None


def build(cfg: dict[str, Any]) -> Arm:
    return LayaZeroShot(cfg)
