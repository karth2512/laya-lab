"""a1 with a stubbed `laya.load`: conformance, wording selection and option mapping."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from arms.a1_laya_zeroshot.arm import build, load_variants
from tests.helpers import _WORDS, RecordingContext, check_arm, synthetic_split


class StubAgent:
    """Answers by keyword; only the 'plain' wording is informative, others are uniform."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def predict_batch(self, states: list[str], questions: dict[str, Any], **kw: Any) -> list[Any]:
        self.calls.append(questions)
        ((qid, q),) = questions.items()
        options = list(q["criteria"])
        informative = q["instructions"].startswith("Which category")
        out = []
        for s in states:
            p = np.full(5, 0.2)
            if informative:
                hits = np.array([sum(s.count(w) for w in ws) for ws in _WORDS], dtype=float)
                p = hits / hits.sum()
            # Keys are labels, so the arm must map back to class-id order itself.
            probs = {o: round(float(v), 4) for o, v in zip(options, p, strict=True)}
            out.append({"answers": {qid: {"type": "choice", "probabilities": probs}}})
        return out


@pytest.fixture
def stub_laya(monkeypatch: pytest.MonkeyPatch) -> StubAgent:
    agent = StubAgent()
    mod = types.ModuleType("laya")
    mod.load = lambda *a, **kw: agent  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "laya", mod)
    return agent


def test_conformance_and_wording_selection(stub_laya: StubAgent, tmp_path: Path) -> None:
    rec = RecordingContext()
    arm = build({"arm": {}, "laya": {"hub_id": "stub", "revision": None}})
    preds = check_arm(arm, rec.ctx(tmp_path, arm="a1_laya_zeroshot"))
    assert rec.params["wording/selected"] == "plain"
    names = {m[0] for m in rec.metrics}
    for v in load_variants("configs/prompts/labels.yaml"):
        assert f"wording/{v}/val_macro_f1" in names
    assert {"wording/spread_val_macro_f1", "wording/order_reversed_val_macro_f1"} <= names
    assert preds.logits is None
    # Keyword stub is perfect on synthetic data, so class-id order must be preserved.
    val = synthetic_split(20, seed=1, prefix="va")
    assert (preds.probs.argmax(1) == val.labels).all()
    assert any("option_order" in q["category"] for q in stub_laya.calls)


def test_option_order_check_can_be_disabled(stub_laya: StubAgent, tmp_path: Path) -> None:
    rec = RecordingContext()
    arm = build({"arm": {"option_order_check": False, "variants": ["plain", "short"]}})
    check_arm(arm, rec.ctx(tmp_path))
    assert "wording/order_reversed_val_macro_f1" not in {m[0] for m in rec.metrics}
    assert len(stub_laya.calls) == 3  # 2 variants on val + predict


def test_predict_requires_fit(stub_laya: StubAgent, tmp_path: Path) -> None:
    arm = build({"arm": {}})
    ctx = RecordingContext().ctx(tmp_path)
    arm.setup(ctx)
    with pytest.raises(RuntimeError):
        arm.predict(synthetic_split(5), ctx)
