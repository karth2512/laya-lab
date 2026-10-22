"""Test guard (REQUIREMENTS §4.4, §10.5): a dev run cannot load test labels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from shared import data, runner
from tests.runtime_helpers import STUB_ARM, make_project


def test_dev_run_never_reads_test_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    overrides = make_project(tmp_path, monkeypatch)
    (tmp_path / "raw" / "test.csv").unlink()  # any attempt to read it would fail

    calls: list[bool] = []
    real = data.load_frames

    def spy(raw_dir: Any, include_test: bool = False) -> Any:
        calls.append(include_test)
        return real(raw_dir, include_test=include_test)

    monkeypatch.setattr(data, "load_frames", spy)
    metrics = runner.run(STUB_ARM, "dev", 0, 1.0, overrides)
    assert calls == [False]
    assert not any(k.startswith("test/") for k in metrics)


def test_final_run_evaluates_test(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    overrides = make_project(tmp_path, monkeypatch)
    metrics = runner.run(STUB_ARM, "final", 0, 0.5, overrides)
    assert "test/macro_f1" in metrics
    assert "test/bootstrap_ci_low/macro_f1" in metrics


def test_fraction_subset_is_nested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    overrides = make_project(tmp_path, monkeypatch)
    cfg = runner.load_config(STUB_ARM, overrides)
    small, val, test, _ = runner.load_splits(cfg, "dev", 0, 0.25)
    big, _, _, _ = runner.load_splits(cfg, "dev", 0, 0.5)
    assert test is None
    assert set(small.ids) < set(big.ids)
    assert not set(big.ids) & set(val.ids)
