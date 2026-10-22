# a5: Jev API (zero-shot)

Closed-model reference. No training: `fit` scores every label-wording variant from
`configs/prompts/labels.yaml` (shared with a1) on val and keeps the best by macro-F1; `predict`
uses that variant. Test is run once, by the runner, at `stage=final`.

- **Deps:** `uv sync --extra jev` (httpx).
- **Key:** read from the env var named by `api_key_env` (default `TYPESAFE_API_KEY`, or `.env`).
  Never logged, printed or stored; it is redacted from error messages.
- **Model pin:** `jev-1.13.0`. `jev-latest` / `jev-preview` are refused at `stage=final`.
- **Cache:** `shared/cache.py`, keyed by `sha256(text | question | options | model)`. Hit rate,
  API calls and estimated cost (`$0.042 / 1M input tokens`, tokens ≈ chars / 4) are logged and the
  cache file is logged as an artifact. Re-runs are free.
- **Robustness:** per-request timeout, whole-run timeout, exponential backoff on 429/5xx/timeouts,
  a simple requests-per-second limiter. Requests are sequential.
- **Outputs:** per-option probabilities, renormalized; no logits, so calibration is raw only.
  Probabilities can be exactly 0, so log loss is clipped in `shared/metrics.py`.
- **Latency** is end-to-end (network and queueing) and tagged `latency_kind=end_to_end`; it is
  not comparable to local arms. Cache hits replay the latency measured on the original call.

## Known quirk: request/response schema not yet confirmed

Account access is still waitlisted, so the JSON shapes are a best guess from the public docs.
They live in two small functions in `arm.py`, `_request_body` and `_parse_probs`. Confirm both
against a real response before the first real run.
