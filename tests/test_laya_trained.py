"""a2/a3/a4 on a tiny random ModernBERT (CPU, offline): contract, logging and batching."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

pytest.importorskip("torch")
pytest.importorskip("transformers")

from shared.train_loop import Encoded, token_batches  # noqa: E402
from tests.helpers import RecordingContext, check_arm, synthetic_split  # noqa: E402

ARMS = ["a2_laya_frozen_head", "a3_laya_full_ft", "a4_laya_lora"]


def _tiny_cfg(arm: str) -> dict[str, Any]:
    cfg: dict[str, Any] = yaml.safe_load(Path(f"arms/{arm}/config.yaml").read_text())
    cfg["backbone"]["tiny_random"] = True
    cfg["train"].update(
        epochs=2,
        min_steps=2,
        patience=5,
        effective_batch_size=8,
        max_tokens_per_batch=1024,
        lr=1e-3,
    )
    if "lora" in cfg:
        cfg["lora"]["r"] = 4
    return cfg


@pytest.mark.parametrize("arm_name", ARMS)
def test_arm_contract(arm_name: str, tmp_path: Path) -> None:
    pytest.importorskip("peft")
    rec = RecordingContext()
    ctx = rec.ctx(tmp_path, arm=arm_name, arm_cfg=_tiny_cfg(arm_name))
    arm = importlib.import_module(f"arms.{arm_name}.arm").build(ctx.cfg)
    preds = check_arm(arm, ctx)
    assert preds.logits is not None
    names = {m[0] for m in rec.metrics}
    assert {"train/loss", "train/lr", "val/loss", "val/macro_f1", "data/truncated_pct"} <= names
    epochs = [s for n, _, s in rec.metrics if n == "val/macro_f1"]
    assert epochs == list(range(len(epochs))) and len(epochs) >= 2
    if arm_name == "a4_laya_lora":
        assert any(p.name.endswith(".safetensors") for p in rec.artifacts)


def test_param_counts(tmp_path: Path) -> None:
    pytest.importorskip("peft")
    counts = {}
    for arm_name in ARMS:
        ctx = RecordingContext().ctx(tmp_path, arm=arm_name, arm_cfg=_tiny_cfg(arm_name))
        arm = importlib.import_module(f"arms.{arm_name}.arm").build(ctx.cfg)
        arm.setup(ctx)
        report = arm.fit(synthetic_split(20), synthetic_split(10, seed=1, prefix="v"), ctx)
        counts[arm_name] = (report.trainable_params, report.total_params)
    a2, a3, a4 = (counts[a] for a in ARMS)
    assert a3[0] == a3[1]  # everything trains
    assert 0 < a2[0] < a4[0] < a3[0]  # head < LoRA + head < all


def test_token_batches_budget_and_padding() -> None:
    rng = np.random.default_rng(0)
    lengths = rng.integers(5, 300, size=200)
    enc = Encoded([[1] * int(n) for n in lengths], np.zeros(200, dtype=int), 0.0)
    batches = token_batches(enc, max_tokens=1024, pad_id=0, rng=rng)
    seen = np.concatenate([b.idx for b in batches])
    assert sorted(seen.tolist()) == list(range(200))
    for b in batches:
        n, width = b.inputs["input_ids"].shape
        assert width % 64 == 0 and (n == 1 or n * width <= 1024)
        assert b.inputs["attention_mask"].sum(1).tolist() == [int(lengths[i]) for i in b.idx]
