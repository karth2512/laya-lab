"""a5 Jev client: mocked HTTP, cache hit/miss, retries, key never leaks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
import yaml

from arms.a5_jev_api.arm import JevArm
from shared.cache import ResponseCache
from tests.helpers import RecordingContext, check_arm, synthetic_split

KEY = "sk-test-SECRET-123"
ARM_CFG: dict[str, Any] = yaml.safe_load(
    (Path(__file__).resolve().parents[1] / "arms/a5_jev_api/config.yaml").read_text()
)


def _handler(fail_first: int = 0, status: int = 503):
    calls: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if len(calls) <= fail_first:
            return httpx.Response(status)
        body = json.loads(request.content)
        options = body["question"]["options"]
        # Pick the option whose index matches the hash of the text, to give varied answers.
        probs = {o: 0.0 for o in options}
        probs[options[len(body["state"]) % len(options)]] = 1.0
        return httpx.Response(200, json={"probabilities": probs})

    return handle, calls


def _arm(tmp_path: Path, handler: Any, **overrides: Any) -> JevArm:
    cfg = {**ARM_CFG, "cache_file": str(tmp_path / "cache.jsonl"), "backoff_s": 0.0}
    cfg.update(overrides)
    arm = JevArm({"arm": cfg}, transport=httpx.MockTransport(handler))
    arm.sleep = lambda s: None
    return arm


@pytest.fixture(autouse=True)
def _key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", KEY)


def test_cache_roundtrip(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "c.jsonl")
    k = cache.key("text", "q", "a|b", "jev-1.13.0")
    assert cache.get(k) is None
    cache.put(k, {"probs": [1.0]})
    assert ResponseCache(tmp_path / "c.jsonl").get(k) == {"probs": [1.0]}
    assert (cache.hits, cache.misses) == (0, 1)


def test_conformance_and_cache_hits(tmp_path: Path) -> None:
    handler, calls = _handler()
    rec = RecordingContext()
    preds = check_arm(_arm(tmp_path, handler), rec.ctx(tmp_path, arm="a5_jev_api"))
    assert preds.logits is None and preds.latency_ms is not None
    first_calls = len(calls)
    assert rec.tags["latency_kind"] == "end_to_end"
    assert "selected_variant" in rec.params
    # Second run over the same data is served fully from the cache.
    rec2 = RecordingContext()
    check_arm(_arm(tmp_path, handler), rec2.ctx(tmp_path, arm="a5_jev_api"))
    assert len(calls) == first_calls
    metrics = {name: value for name, value, _ in rec2.metrics}
    assert metrics["jev/api_calls"] == 0 and metrics["jev/cache_hit_rate"] == 1.0


def test_retry_then_success(tmp_path: Path) -> None:
    handler, calls = _handler(fail_first=2, status=429)
    arm = _arm(tmp_path, handler, variants=["plain"])
    rec = RecordingContext()
    ctx = rec.ctx(tmp_path)
    arm.setup(ctx)
    split = synthetic_split(3)
    arm.fit(split, split, ctx)
    assert len(calls) == 3 + 2
    np.testing.assert_allclose(arm.predict(split, ctx).probs.sum(1), 1.0)


def test_gives_up_and_never_leaks_key(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    handler, _ = _handler(fail_first=100)
    arm = _arm(tmp_path, handler, max_retries=2)
    rec = RecordingContext()
    ctx = rec.ctx(tmp_path)
    arm.setup(ctx)
    with pytest.raises(RuntimeError) as err:
        arm.fit(synthetic_split(2), synthetic_split(2), ctx)
    arm.teardown(ctx)
    logged = repr(rec.metrics) + repr(rec.params) + repr(rec.tags) + repr(rec.dicts)
    out = capsys.readouterr()
    for text in (str(err.value), logged, out.out, out.err):
        assert KEY not in text


def test_refuses_moving_alias_at_final(tmp_path: Path) -> None:
    handler, _ = _handler()
    ctx = RecordingContext().ctx(tmp_path)
    ctx.stage = "final"
    with pytest.raises(ValueError):
        _arm(tmp_path, handler, model="jev-latest").setup(ctx)


def test_missing_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPESAFE_API_KEY")
    handler, _ = _handler()
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        _arm(tmp_path, handler).setup(RecordingContext().ctx(tmp_path))
