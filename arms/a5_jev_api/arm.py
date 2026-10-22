"""a5: Jev via the typesafe API, zero-shot, with wording selection on val and a response cache."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import yaml
from sklearn.metrics import f1_score

from shared.cache import ResponseCache
from shared.interface import Arm, FitReport, Predictions, RunContext, Split

REPO_ROOT = Path(__file__).resolve().parents[2]
MOVING_ALIASES = {"jev-latest", "jev-preview"}
RETRY_STATUS = {429, 500, 502, 503, 504}


# --- Wire format: unconfirmed until account access is granted (see README) -------------------


def _request_body(model: str, question: str, options: list[str], text: str) -> dict[str, Any]:
    return {
        "model": model,
        "question": {"type": "choice", "instructions": question, "options": options},
        "state": text,
    }


def _parse_probs(payload: dict[str, Any], options: list[str]) -> np.ndarray:
    """Per-option probabilities, as a dict keyed by option or a list in option order."""
    raw = payload["probabilities"]
    values = [raw[o] for o in options] if isinstance(raw, dict) else list(raw)
    probs = np.clip(np.asarray(values, dtype=float), 0.0, None)
    if probs.shape != (len(options),) or not np.all(np.isfinite(probs)):
        raise ValueError(f"unexpected probabilities in response: {raw!r}")
    total = probs.sum()
    return probs / total if total > 0 else np.full(len(options), 1.0 / len(options))


# ---------------------------------------------------------------------------------------------


class JevArm(Arm):
    name = "a5_jev_api"

    def __init__(self, cfg: dict[str, Any], transport: httpx.BaseTransport | None = None) -> None:
        self.cfg = cfg["arm"]
        self.transport = transport
        self.sleep = time.sleep  # swapped out in tests
        prompts = Path(self.cfg["prompts_path"])
        prompts = prompts if prompts.is_absolute() else REPO_ROOT / prompts
        variants = yaml.safe_load(prompts.read_text())["variants"]
        wanted = self.cfg.get("variants") or list(variants)
        self.variants: dict[str, dict[str, Any]] = {v: variants[v] for v in wanted}
        self.selected: str | None = None
        self.api_calls = 0
        self.input_chars = 0

    def setup(self, ctx: RunContext) -> None:
        model = self.cfg["model"]
        if ctx.stage == "final" and model in MOVING_ALIASES:
            raise ValueError(f"pin an exact Jev version for final runs, not {model!r}")
        self._key = os.environ.get(self.cfg["api_key_env"], "")
        if not self._key:
            raise RuntimeError(f"set the {self.cfg['api_key_env']} environment variable")
        cache_file = Path(self.cfg["cache_file"])
        self.cache = ResponseCache(
            cache_file if cache_file.is_absolute() else REPO_ROOT / cache_file
        )
        self.client = httpx.Client(
            transport=self.transport,
            timeout=self.cfg["timeout_s"],
            headers={"Authorization": f"Bearer {self._key}"},
        )
        self.deadline = time.monotonic() + self.cfg["run_timeout_s"]
        self.min_interval = 1.0 / self.cfg["rate_limit_rps"]
        self.last_call = 0.0
        ctx.log_param("jev_model_version", model)
        ctx.set_tag("latency_kind", "end_to_end")

    def _redact(self, msg: str) -> str:
        return msg.replace(self._key, "***") if self._key else msg

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        delay = self.cfg["backoff_s"]
        for attempt in range(self.cfg["max_retries"] + 1):
            if time.monotonic() > self.deadline:
                raise TimeoutError("a5 run timeout exceeded")
            wait = self.last_call + self.min_interval - time.monotonic()
            if wait > 0:
                self.sleep(wait)
            self.last_call = time.monotonic()
            self.api_calls += 1
            try:
                resp = self.client.post(self.cfg["url"], json=body)
                if resp.status_code not in RETRY_STATUS:
                    resp.raise_for_status()
                    result: dict[str, Any] = resp.json()
                    return result
                error = f"HTTP {resp.status_code}"
            except httpx.TimeoutException as e:
                error = type(e).__name__
            except httpx.HTTPError as e:
                raise RuntimeError(self._redact(f"Jev request failed: {e}")) from None
            if attempt < self.cfg["max_retries"]:
                self.sleep(delay)
                delay *= 2
        raise RuntimeError(self._redact(f"Jev request failed after retries: {error}"))

    def _classify(self, text: str, variant: str) -> tuple[np.ndarray, float]:
        v = self.variants[variant]
        key = self.cache.key(text, v["question"], "|".join(v["options"]), self.cfg["model"])
        hit = self.cache.get(key)
        if hit is not None:
            return np.asarray(hit["probs"]), float(hit["latency_ms"])
        body = _request_body(self.cfg["model"], v["question"], v["options"], text)
        start = time.perf_counter()
        payload = self._post(body)
        latency_ms = (time.perf_counter() - start) * 1000
        self.input_chars += len(text) + len(v["question"]) + sum(len(o) for o in v["options"])
        probs = _parse_probs(payload, v["options"])
        self.cache.put(key, {"probs": probs.tolist(), "latency_ms": latency_ms})
        return probs, latency_ms

    def _predict_variant(self, split: Split, variant: str) -> Predictions:
        rows = [self._classify(t, variant) for t in split.texts]
        return Predictions(
            ids=list(split.ids),
            probs=np.stack([p for p, _ in rows]),
            latency_ms=np.array([ms for _, ms in rows]),
        )

    def fit(self, train: Split, val: Split, ctx: RunContext) -> FitReport:
        scores: dict[str, float] = {}
        for variant in self.variants:
            preds = self._predict_variant(val, variant)
            scores[variant] = float(f1_score(val.labels, preds.probs.argmax(1), average="macro"))
            ctx.log_metric(f"wording/{variant}/val_macro_f1", scores[variant])
        self.selected = max(scores, key=lambda k: scores[k])
        ctx.log_metric("wording/val_macro_f1_spread", max(scores.values()) - min(scores.values()))
        ctx.log_param("selected_variant", self.selected)
        return FitReport(extra={"selected_variant": self.selected, "variant_val_macro_f1": scores})

    def predict(self, split: Split, ctx: RunContext) -> Predictions:
        if self.selected is None:
            raise RuntimeError("fit() must select a wording variant before predict()")
        return self._predict_variant(split, self.selected)

    def teardown(self, ctx: RunContext) -> None:
        lookups = self.cache.hits + self.cache.misses
        ctx.log_metric("jev/api_calls", self.api_calls)
        ctx.log_metric("jev/cache_hits", self.cache.hits)
        ctx.log_metric("jev/cache_hit_rate", self.cache.hits / lookups if lookups else 0.0)
        tokens = self.input_chars / 4
        ctx.log_metric("jev/est_input_tokens", tokens)
        ctx.log_metric("jev/est_cost_usd", tokens / 1e6 * self.cfg["usd_per_m_input_tokens"])
        if self.cache.path.exists():
            ctx.log_artifact(self.cache.path)
        self.client.close()


def build(cfg: dict[str, Any]) -> Arm:
    return JevArm(cfg)
