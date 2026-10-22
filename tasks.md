# Tasks: Laya Classification Benchmark

Ordered plan to complete `REQUIREMENTS.md`. Tasks in the same **wave** are independent and run in
parallel; a wave starts when the waves it depends on are done. Status: `[ ]` todo · `[~]` in progress ·
`[x]` done · `[!]` blocked (reason given).

**Where work runs.** Waves 0–3 are code and run anywhere (CPU, tests on a ~50-example fixture).
Waves 4–6 need the M5 Mac (MPS, AC power) and, for a5, a Jev API key.

## Simplifications (vs. REQUIREMENTS)

- **No HF Trainer, so no `shared/hf_callbacks.py`.** a2–a4 share one plain PyTorch loop
  (`shared/train_loop.py`): token-budget batching, pad-to-64, bf16 autocast, grad accumulation,
  early stopping, logs through `ctx`. §4.1 only requires the callback for framework loops.
- **One Laya loader** `shared/laya_backbone.py` (ADR 0001 path A) used by a2/a3/a4; arms differ only in
  what is trainable.
- **Nested fractions by construction:** one stratified permutation per (seed, class); fraction `f`
  takes the first `ceil(f · n_class)` IDs, so nesting holds without extra checks.
- **Label mapping in one place:** `shared/data.CLASSES`; IDs are `train-<row>` / `test-<row>`.
- **a2 caches encoder features once per run** and trains the head on them (ADR 0001).
- **LoRA rank tuned as an HP inside a4's 6-trial budget** (the §11 #8 default) unless the owner decides
  otherwise.
- **Added deps:** `mlflow`, `matplotlib`, `scipy` (base dependencies; `uv.lock` updated).

## Module contracts (fixed in wave 0 so wave 1 can run in parallel)

| Module | Public API |
|---|---|
| `shared/interface.py` | `Split`, `Predictions(.validate)`, `FitReport`, `RunContext(.arm_cfg)`, `Arm` |
| `shared/data.py` | `CLASSES`, `download(raw_dir)`, `load_frames(raw_dir) -> (train_df, test_df)` with cols `id,text,label`, `to_split(df) -> Split`, `audit(...)` |
| `shared/splits.py` | `make_manifest(...)`, `save_manifest`, `load_manifest(path)` (verifies hash), `manifest_hash`, `fraction_ids(pool_ids, pool_labels, seed, fraction)` |
| `shared/metrics.py` | `compute_metrics(y_true, probs) -> dict` (`accuracy`, `macro_f1`, …, `f1/<class>`; log loss clipped) |
| `shared/calibration.py` | `softmax`, `ece`, `brier`, `fit_temperature(logits, y) -> float` |
| `shared/stats.py` | `bootstrap_ci(y, probs, metric, n, seed)`, `mcnemar(correct_a, correct_b) -> (stat, p)` |
| `shared/cost.py` | `count_params(model)`, `MemorySampler` (bg thread; tensor/driver/RSS peaks), `Timer`, `latency_stats` |
| `shared/device.py` | `select_device(pref)`, `autocast(device, enabled)`, `memory_now()` |
| `shared/seed.py` | `seed_everything(seed, deterministic) -> bool` |
| `shared/config.py` | `load_config(arm, overrides) -> dict` (pydantic base schema, arm cfg under `arm:`), `flatten` |
| `shared/cache.py` | `ResponseCache(path)`: `key(*parts)`, `get`, `put`, `hits`, `misses` |
| `shared/mlflow_utils.py` | only `mlflow` importer: parent/child runs, NaN guard, `env_tags()`, `completed_run_names()` |
| `shared/runner.py` | `run(arm, stage, seed, fraction, overrides, variant=None) -> dict` (§4.4 flow + test guard) |

---

## Wave 0: Foundation (sequential, blocks everything)

- [x] 0.1 Add `mlflow`, `matplotlib`, `scipy` with `uv add`; `uv.lock` refreshed
- [x] 0.2 Package skeleton: `shared/`, `arms/`, `scripts/`, `tests/` with `__init__.py`
- [x] 0.3 `shared/interface.py` (contract above)
- [x] 0.4 `configs/base.yaml`, `configs/prompts/labels.yaml` (4 wording variants)

## Wave 1: Build in parallel (each stream owns its files; needs wave 0)

| Stream | Tasks |
|---|---|
| **A. Data** | [~] A1 `shared/data.py` (download from HF, label map, hygiene: exact + near-dup across splits, token lengths/% truncated, class balance) · [~] A2 `shared/splits.py` · [~] A3 `scripts/prepare_data.py` → `data/splits/split_v1.json` + `reports/data_audit.md` · [~] A4 `tests/test_splits.py` + 50-example fixture `tests/fixtures/` |
| **B. Metrics** | [~] B1 `shared/metrics.py` · [~] B2 `shared/calibration.py` · [~] B3 `shared/stats.py` · [~] B4 `tests/test_metrics.py` (sklearn parity, hand-computed ECE/Brier) |
| **C. Core runtime** | [~] C1 `shared/config.py`, `seed.py`, `device.py`, `cost.py` · [~] C2 `shared/mlflow_utils.py` · [~] C3 `shared/runner.py` · [~] C4 `scripts/run.py`, `scripts/sweep.py` (resumable via MLflow) · [~] C5 tests: test guard, MLflow temp-backend logging, isolation (arms ↛ arms, only `mlflow_utils` imports mlflow) |
| **D. Baselines** | [x] D1 `arms/a0_majority` · [x] D2 `arms/a0_tfidf_logreg` (TF-IDF settings tuned on val, 6 trials) · [x] D3 `tests/test_arms_conformance.py` (parametrized over every CPU-loadable arm, stubs for heavy ones) |
| **E. Jev API** | [x] E1 `shared/cache.py` · [x] E2 `arms/a5_jev_api` (pinned `jev-1.13.0`, httpx, retries/backoff, rate limit, timeouts, `TYPESAFE_API_KEY`, wording select on val, e2e latency tag, cost estimate) · [x] E3 `tests/test_jev.py` (mocked HTTP, cache hit/miss, retry, key never logged) |
| **F. Laya zero-shot** | [~] F1 `arms/a1_laya_zeroshot` (`laya.load` pinned revision, typed choice, wording sweep on val, spread + option-order check) · [~] F2 conformance via stubbed `laya` |
| **G. Laya trained** | [~] G1 `shared/laya_backbone.py` · [~] G2 `shared/train_loop.py` · [~] G3 `arms/a2_laya_frozen_head` · [~] G4 `arms/a3_laya_full_ft` (ckpt on, `--save-model` opt-in) · [~] G5 `arms/a4_laya_lora` (r∈{4,8,16,32}, `Wqkv/Wo/Wi`, adapter saved) · [~] G6 tiny-random-ModernBERT CPU test |
| **H. Reporting** | [~] H1 `scripts/aggregate.py` (mean ± std, bootstrap CI, learning curves with a1/a5 lines, McNemar matrix, calibration plots, cost table → `reports/results.md`, `reports/figures/`) · [~] H2 `scripts/error_analysis.py` · [~] H3 tests on synthetic MLflow runs |

Every arm folder: `arm.py` (`build(cfg) -> Arm`), `config.yaml`, `README.md`.

Cross-stream conventions:
- Raw data: `data/raw/{train,test}.csv` with columns `condition_label` (1–5), `medical_abstract`.
  Test fixture uses the same layout in `tests/fixtures/raw/` (~40 train / 10 test rows).
- `load_frames(raw_dir, include_test=False)`: at `stage=dev` the runner never reads the test file.
- Manifest JSON: `{version, val_fraction, split_seed, train_pool: [ids], val: [ids], test: [ids], hash}`.
- `build(cfg)` gets the full resolved config; the arm's own settings are `cfg["arm"]` / `ctx.arm_cfg`.
- Arm tests use `tests/helpers.py` (`RecordingContext`, `synthetic_split`, `check_arm`).

## Wave 2: Integrate (needs wave 1)

- [ ] 2.1 `make check` green (ruff, mypy, pytest on CPU)
- [ ] 2.2 `make data` on real data; record split hash here
- [ ] 2.3 `make run ARM=a0_majority` and `a0_tfidf_logreg` (dev + final) on real data; MLflow has all §5 groups/artifacts/tags
- [ ] 2.4 CPU smoke runs of a1–a4 on a tiny fraction (`--set` caps) through `scripts/run.py`
- [ ] 2.5 `make aggregate` on the runs above produces a report skeleton
- [x] 2.6 README (usage, arm table, CC BY-SA 3.0 attribution, limitations §11 #5–7)

## Wave 3: Acceptance review (needs wave 2)

- [ ] 3.1 Walk the §10 acceptance checklist; every Makefile target exists and works
- [ ] 3.2 Fresh-clone check: `make setup && make data` reproduces the split hash

## Wave 4: M2–M4 on the M5 Mac (owner; needs wave 3)

- [ ] 4.1 a1 wording sweep on val (MPS) — can run in parallel with 4.2 only if not sharing the GPU; otherwise sequential
- [ ] 4.2 a5 wording sweep on val — [!] blocked on Jev waitlist access (`TYPESAFE_API_KEY`); network-bound, can overlap MPS runs
- [ ] 4.3 HP tuning on val at fraction 1.0, 6 trials each: a0_tfidf (CPU, can overlap), a2, a4 (incl. rank), a3 — one MPS run at a time
- [ ] 4.4 Verify a4 no-checkpointing stays under 12 GB driver on real batches; otherwise turn ckpt on and log it
- [ ] 4.5 Owner decision: LoRA rank scope (§11 #8)

## Wave 5: M5 final sweep (owner, M5 Mac, ~45 h, one MPS run at a time)

- [ ] 5.1 `make sweep STAGE=final` for a0 (CPU, can overlap), a1, a2, a4, a3; a5 when unblocked (overlaps)
- [ ] 5.2 Apply §8.3 scope cuts if time runs out; document them

## Wave 6: M6 report (needs wave 5)

- [ ] 6.1 `make aggregate` + `scripts/error_analysis.py` → `reports/results.md`
- [ ] 6.2 Write up limitations (contamination, label noise, HP parity, same-machine timing caveat)

---

## Progress log

- Wave 0 done: deps added, skeleton, interface, configs and `tests/helpers.py` in place.
- Wave 1 started: streams A–H running in parallel.
- Stream D done: a0_majority, a0_tfidf_logreg (6-trial val tuning, logits), conformance test; 4 tests pass, lint/types clean.
- Stream E done: `shared/cache.py`, a5_jev_api (pinned model, retries, rate limit, cache, key redaction); 6 tests pass. Request/response JSON shape is a placeholder until API access.
