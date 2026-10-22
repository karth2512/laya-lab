# Laya Classification Benchmark: Requirements

Benchmark Laya (`convaiinnovations/laya`, ModernBERT-large encoder, 421M params, 512-token context) as a text classifier on the Medical Abstracts TC Corpus. Compare zero-shot, frozen-head, full fine-tuning and LoRA adaptation against classical floors and one closed-model API reference (Jev). Every metric is logged to MLflow.

---

## 1. Goals

1. Measure how much accuracy Laya gains from each adaptation method, and at what cost (params, time, memory, latency).
2. Measure how much labeled data each method needs (learning curves over training fractions).
3. Place Laya against a closed reference model (Jev API, zero-shot) and against classical floors.
4. Produce results that are reproducible, statistically defensible and fully traceable in MLflow.

### Non-goals
- LayaStudio or any no-code tooling. Custom Python only.
- Other model families (GLiClass, NLI encoders, OpenJev variants, domain BERTs).
- Multi-label classification, deployment, serving or productionization.
- CUDA/ROCm-specific optimizations. Everything targets a single M5 Mac with 16GB unified memory (MPS).

---

## 2. Dataset and splits

| Item | Requirement |
|---|---|
| Dataset | Medical Abstracts TC Corpus (HF: `TimSchopf/medical_abstracts`, or the GitHub CSVs `medical_tc_train.csv` / `medical_tc_test.csv`) |
| Classes (5) | Neoplasms, Digestive system diseases, Nervous system diseases, Cardiovascular diseases, General pathological conditions |
| Official split | 11,550 train / 2,888 test. Use it as-is. Do not re-split train/test. |
| Validation | Stratified 10% of train (about 1,155), fixed seed, carved once. Remaining about 10,395 is the **training pool**. |
| Split manifest | `data/splits/split_v1.json`: example IDs per split, plus a SHA-256 hash. The hash is logged on every run. |
| Fractions | Training fractions {1%, 5%, 10%, 25%, 100%} of the training pool. Stratified, **nested** (the 1% set is a subset of the 5% set, and so on) per seed. |
| Label mapping | Numeric labels 1–5 mapped to the names above in one place (`shared/data.py`). |
| Majority baseline | "General pathological conditions" is 961 of 2,888 test examples, so the floor is about 33.3% accuracy. |
| Hygiene checks | (a) duplicate and near-duplicate detection across train/val/test, (b) token-length distribution with the Laya tokenizer, with % truncated at 512 tokens, (c) class distribution per split. Report all three in `reports/data_audit.md`. |
| Truncation | Truncate to 512 tokens (Laya's limit). The truncation rate is logged per run. |
| License | Dataset is CC BY-SA 3.0. Keep attribution in the README and any report. |

---

## 3. Arms

Every arm implements the same interface (Section 4) and lives in its own folder.

| ID | Folder | Description | Fit? | Fractions? |
|---|---|---|---|---|
| a0 | `arms/a0_majority` | Majority-class predictor | trivial | no |
| a0 | `arms/a0_tfidf_logreg` | TF-IDF + logistic regression | yes | yes |
| a1 | `arms/a1_laya_zeroshot` | Laya zero-shot, typed choice question over 5 options | no (wording selection only) | no |
| a2 | `arms/a2_laya_frozen_head` | Frozen Laya encoder + trained classification head | head only | yes |
| a3 | `arms/a3_laya_full_ft` | Laya full fine-tuning | all params | yes |
| a4 | `arms/a4_laya_lora` | Laya + LoRA, rank sweep r ∈ {4, 8, 16, 32} | adapters + head | yes |
| a5 | `arms/a5_jev_api` | Jev via API, zero-shot, no training | no (wording selection only) | no |

Zero-shot arms (a1, a5) appear as horizontal reference lines on learning-curve plots.

---

## 4. Architecture

### 4.1 Principles
- **One experiment interface.** Every arm implements the same `Arm` contract and is launched through one runner and one CLI.
- **Independent arm logic.** Each arm's code lives in its own folder. **Arms must not import from other arms.** A test enforces this.
- **Shared code is imported, not copied.** Data, splits, metrics, calibration, stats, MLflow logging, cost tracking and the runner live in `shared/`.
- **Arms never call `mlflow` directly.** They log through the context object (`ctx.log_metric`, `ctx.log_param`, `ctx.log_artifact`). Training loops that use framework logging (for example HF Trainer) must set `report_to="none"` and use a shared callback that forwards to `ctx`.
- **Python 3.11+**, type hints throughout, `ruff` and `mypy` clean.

### 4.2 Repository layout

```
laya-bench/
├── pyproject.toml              # extras per arm: [baselines] [laya] [jev]; dev dependency group
├── uv.lock                     # committed lockfile (Section 8.4)
├── .python-version             # pinned interpreter for uv
├── Makefile                    # entry point for every routine task (Section 8.4)
├── README.md
├── REQUIREMENTS.md
├── configs/
│   ├── base.yaml               # dataset, seeds, mlflow, device defaults
│   └── prompts/labels.yaml     # label-wording variants shared by a1 and a5
├── shared/                     # importable package, no arm-specific logic
│   ├── __init__.py
│   ├── interface.py            # Arm ABC, Split, Predictions, RunContext dataclasses
│   ├── config.py               # pydantic-validated YAML loading and CLI overrides
│   ├── data.py                 # download, load, label mapping, hygiene checks
│   ├── splits.py               # val carve-out, nested stratified fractions, manifest hash
│   ├── metrics.py              # classification metrics (sklearn-checked)
│   ├── calibration.py          # ECE, Brier, temperature scaling
│   ├── stats.py                # bootstrap CIs, McNemar
│   ├── cost.py                 # param counts, wall-clock, peak memory, latency timers
│   ├── cache.py                # disk cache keyed by content hash (used by API arms)
│   ├── device.py               # MPS-first selection (CPU fallback), dtype policy, memory probes
│   ├── seed.py                 # seeding for python, numpy, torch
│   ├── mlflow_utils.py         # the only module that imports mlflow
│   ├── hf_callbacks.py         # TrainerCallback forwarding to ctx
│   └── runner.py               # orchestrates a single run (Section 4.4)
├── arms/
│   ├── __init__.py
│   ├── a0_majority/            { __init__.py, arm.py, config.yaml, README.md }
│   ├── a0_tfidf_logreg/
│   ├── a1_laya_zeroshot/
│   ├── a2_laya_frozen_head/
│   ├── a3_laya_full_ft/
│   ├── a4_laya_lora/
│   └── a5_jev_api/
├── scripts/
│   ├── prepare_data.py         # download, audit, write split manifest
│   ├── run.py                  # single run: one arm, one seed, one fraction
│   ├── sweep.py                # grid of runs, creates parent runs per arm
│   ├── aggregate.py            # pulls from MLflow, builds tables and plots
│   └── error_analysis.py       # top confusions, misclassified samples
├── spikes/m0/                  # M0 spike scripts and raw results (not imported by shared/ or arms/)
├── docs/adr/                   # architecture decision records
├── reports/                    # generated: data_audit.md, results.md, figures/
├── tests/
└── data/                       # gitignored: raw data, splits, caches
```

Each arm folder contains:
- `arm.py`, which exposes `build(cfg) -> Arm`
- `config.yaml`, the arm's defaults (hyperparameters, search space)
- `README.md`, a short description, the deps it needs and known quirks

### 4.3 The shared interface

```python
# shared/interface.py (contract; exact signatures may be refined)
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

@dataclass
class Split:
    ids: list[str]
    texts: list[str]
    labels: np.ndarray              # int in [0, 4]

@dataclass
class Predictions:
    ids: list[str]
    probs: np.ndarray               # (N, 5), rows sum to 1, REQUIRED
    logits: np.ndarray | None       # (N, 5) if the arm has them; enables temperature scaling
    latency_ms: np.ndarray | None   # per-example, if measured by the arm

@dataclass
class FitReport:
    trainable_params: int | None
    total_params: int | None
    extra: dict                     # arm-specific values to log (selected wording, best HPs, ...)

class Arm(ABC):
    name: str

    @abstractmethod
    def setup(self, ctx: "RunContext") -> None: ...
    """Load models, clients, caches. No data access."""

    @abstractmethod
    def fit(self, train: Split, val: Split, ctx: "RunContext") -> FitReport: ...
    """Train or select settings using train/val ONLY. No-op for pure zero-shot arms
    (a1/a5 may use val to pick label wording)."""

    @abstractmethod
    def predict(self, split: Split, ctx: "RunContext") -> Predictions: ...

    def teardown(self, ctx: "RunContext") -> None: ...
```

`RunContext` provides: resolved config, seed, fraction, stage, device, `log_metric(name, value, step=None)`, `log_param`, `log_artifact(path)`, `log_dict`, and a scratch directory.

### 4.4 Runner flow (`shared/runner.py`)

1. Load and validate config (`base.yaml` + arm `config.yaml` + CLI overrides). Seed everything.
2. Load data and the split manifest. Verify the hash. Build the nested training subset for (seed, fraction).
3. Start the MLflow child run (nested under the arm's parent run). Log params, tags and environment (Section 5).
4. `arm.setup(ctx)`, then `arm.fit(train, val, ctx)` (timed, memory-tracked).
5. `arm.predict(val)` and compute val metrics. If logits are present, fit temperature on val.
6. **Only if `stage=final`:** `arm.predict(test)` and compute test metrics. Test is touched once per final run.
7. Compute metrics, cost numbers and artifacts. Log everything. `arm.teardown(ctx)`.

**Test-set guard.** `stage=dev` never loads test labels into the process. Arms receive only `train` and `val` in `fit`. `predict(test)` is only called by the runner, after `fit` returns.

### 4.5 CLI

```
python -m scripts.run   --arm a4_laya_lora --stage dev   --seed 0 --fraction 0.1 --set lora.r=16
python -m scripts.run   --arm a4_laya_lora --stage final --seed 0 --fraction 0.1 --set lora.r=16
python -m scripts.sweep --arm a4_laya_lora --stage final --seeds 0 1 2 --fractions 0.01 0.05 0.1 0.25 1.0
python -m scripts.aggregate --experiment laya-medabs-bench --out reports/
```

Arms are discovered by folder name. The runner uses `importlib.import_module(f"arms.{name}.arm")`.

---

## 5. MLflow requirements

| Item | Requirement |
|---|---|
| Tracking URI | From `MLFLOW_TRACKING_URI`; default is a local SQLite backend (`sqlite:///mlflow.db`) with artifacts under `./mlartifacts` |
| Experiment | `laya-medabs-bench` (configurable) |
| Run hierarchy | **Parent run per arm** (`arm=<id>`), **child run per (seed, fraction, variant, stage)** using nested runs |
| Run naming | `<arm>/<stage>/s<seed>/f<fraction>[/<variant>]` |
| Only `shared/mlflow_utils.py` imports `mlflow` | Enforced by a test |

### Params (flattened resolved config)
Arm, stage, seed, fraction, dataset version, split hash, max length, device, dtype, and all arm hyperparameters (LR, batch size, epochs, LoRA rank/alpha/dropout/target modules, label-wording variant, Jev model version, and so on).

### Tags
`arm`, `stage`, `git_sha`, `git_dirty`, `data_hash`, `hostname`, `device_name`, `macos_version`, `torch_backend` (mps/cpu), `mps_fallback_used`, `on_ac_power`, `python_version`, `library_versions` (torch, transformers, peft, laya, mlflow).

### Metrics (names use `/`; step = epoch or optimizer step where applicable)

| Group | Metrics |
|---|---|
| Training curves | `train/loss`, `train/lr`, `val/loss`, `val/macro_f1` per epoch |
| Quality, `val/*` and `test/*` | `accuracy`, `macro_f1` (**primary**), `weighted_f1`, `balanced_accuracy`, `mcc`, `log_loss`, per-class `precision/recall/f1` (`<metric>/<class_name>`) |
| Calibration | `brier`, `ece` (15 bins), plus `ece_ts` and `temperature` when logits are available and temperature scaling was fitted on val |
| Uncertainty summary | `mean_confidence`, `bootstrap_ci_low/high` for `macro_f1` and `accuracy` (test, final stage) |
| Cost | `cost/trainable_params`, `cost/total_params`, `cost/train_seconds`, `cost/peak_device_mem_mb` (MPS tensor memory), `cost/peak_driver_mem_mb` (MPS driver memory), `cost/peak_rss_mb`, `latency/p50_ms`, `latency/p95_ms` (batch size 1), `throughput/examples_per_sec` (best batch size) |
| Data | `data/truncated_pct`, `data/train_size` |

NaN or inf metrics must raise, not be logged.

### Artifacts (per final run)
- `predictions_{val,test}.csv`: id, true label, predicted label, per-class probabilities
- `classification_report.json`
- `confusion_matrix.png`
- `calibration_curve.png`
- `resolved_config.yaml` and `pip_freeze.txt`
- `split_manifest_hash.txt`
- Model weights: **opt-in only**. For a4 save the LoRA adapter. For a3 save weights only when `--save-model` is set.

---

## 6. Evaluation protocol

| Item | Requirement |
|---|---|
| Primary metric | Macro-F1 on the official test set |
| Seeds | At least 3 (0, 1, 2) for trained arms. Optionally 5 for fractions ≤ 10%. |
| Reporting | Mean ± std across seeds, plus bootstrap 95% CI (≥ 1,000 resamples) on pooled test predictions |
| Paired comparison | McNemar's test between arms on test predictions (seed-matched where applicable). Report the full pairwise matrix. |
| HP tuning | **Equal budget per trained method** (default N = 6 trials, random or grid), tuned on val at fraction 1.0. Selected config reused across fractions. Document any deviation. |
| Step floor | At small fractions, scale epochs so every run gets at least a fixed minimum number of optimizer steps. |
| Test discipline | Test is evaluated only at `stage=final`, after all choices are fixed from val. |
| Error analysis | `scripts/error_analysis.py`: top confusion pairs, especially against "General pathological conditions"; sampled misclassified abstracts per class; the confidence distribution of errors |

---

## 7. Arm-specific requirements

### a0: baselines
- Majority class and TF-IDF + LogReg (sklearn). Same interface. CPU only.
- TF-IDF settings are tuned on val within the same trial budget.

### a1: Laya zero-shot
- Typed choice question over the 5 class options via the `laya` package (`laya.load("convaiinnovations/laya")`).
- **Label-wording sweep:** 3–5 phrasings defined in `configs/prompts/labels.yaml`, shared with a5. Evaluate all on val and select the best by val macro-F1. Report the **spread** across variants on val. Only the selected variant is run on test.
- Log the selected variant, per-variant val metrics and any option-order effects.

### a2: Laya frozen encoder + head
- Encoder frozen, a linear (or small MLP) head trained.
- Logs trainable params (head only) and the training curve.

### a3: Laya full fine-tune
- All parameters trainable, early stopping on val macro-F1.
- Must fit the Section 8.2 memory budget on the M5 Mac: gradient checkpointing on by default, small micro-batch with gradient accumulation, dynamic padding, bf16 only if validated on MPS.
- If it cannot fit after all memory levers, follow the Section 8.2 fallback ladder (partial fine-tune as a separately named arm, or an off-machine run flagged in the report). Never change the method silently.

### a4: Laya + LoRA
- `peft` LoRA with rank sweep r ∈ {4, 8, 16, 32}. Target modules, alpha and dropout in `config.yaml`. LoRA LR is tuned separately from full-FT LR.
- Logs trainable-param count and adapter size.
- Plain LoRA only (no QLoRA, since `bitsandbytes` is unavailable on macOS). Same memory levers as a3, but expected to fit well within budget.

### a5: Jev API
- **Pin the model version.** Do not use a moving alias such as `jev-latest` for final runs. Log the exact version as a param.
- **Response cache** (`shared/cache.py`): keyed by `sha256(text | question | options | model_version)`. Cache hit rate, number of API calls and estimated cost are logged. The cache file is logged as an artifact.
- Retries with exponential backoff, rate limiting and per-run timeouts.
- API key from an environment variable (name defined in the arm config). It is never logged, committed or printed.
- Reuse the same label-wording variants as a1. Select on val, then run test once.
- Convert Jev's per-option probabilities into the shared `Predictions` format. Logits are absent, so calibration metrics are reported raw only.
- Latency is logged as **end-to-end** (network and queueing included) and tagged as such. It is not directly comparable to local arms.
- Early-access limits and pricing must be confirmed before the sweep (see Section 11).

---

## 8. Environment and compute (single M5 Mac, 16GB unified memory)

All training, tuning and evaluation run on **one Apple-silicon Mac (M5, 16GB unified memory)** using PyTorch's **MPS** backend. There is no CUDA/ROCm path. The 16GB is shared between the OS, Python and the GPU, so memory, not compute, is the binding constraint.

### 8.1 Platform requirements
- macOS version and PyTorch build that support MPS (and bf16 on MPS). **Pinned by M0:** macOS 26.6, Python 3.12.15, torch 2.14.1, transformers 5.18.0 (≥ 5 required, see ADR 0001), peft 0.21.2, laya 0.3.28, Laya Hub revision `7b928d828b7b0e022f929d9bd2e44165aa270148`. Upgrades go through `make lock` and a re-run of the affected `make spike` steps.
- `shared/device.py` selects `mps` first and falls back to `cpu`. Arms must not hard-code a device. `PYTORCH_ENABLE_MPS_FALLBACK=1` is allowed so unsupported ops fall back to CPU. Any run that triggers a fallback logs a tag so it can be spotted.
- **No CUDA-only dependencies.** No `bitsandbytes` (so no 8-bit optimizers or QLoRA), no `flash-attn`, and no assumption that `torch.compile` works. Attention uses the SDPA implementation.
- fp16 mixed precision with a grad scaler is avoided on MPS. **M0 confirmed bf16 autocast is stable and 1.35–1.55× faster**, so a2–a4 train and evaluate under `torch.autocast("mps", dtype=torch.bfloat16)` with fp32 weights and optimizer state. a1 keeps the laya runtime's own MPS precision policy (fp16 autocast, identical argmax to fp32 in M0).
- Dependencies pinned via `pyproject.toml` and a committed `uv.lock` (see Section 8.4). Arm-specific dependencies sit behind extras so one arm's dependency does not break another's install.

### 8.2 Memory budget
- **Budget (set by M0):** peak MPS **tensor** memory ≤ 9 GB and peak MPS **driver** memory ≤ 12 GB per run. Every MPS run sets `PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.1` (about 13 GB on this machine) and `PYTORCH_MPS_LOW_WATERMARK_RATIO=0.99`, so an over-budget run fails with an OOM instead of swapping macOS.
- Measured in M0 (`docs/m0_spike.md`): full fine-tuning (a3) with AdamW, gradient checkpointing and bf16 peaks at 6.1 GB tensors / 8.0 GB driver at fixed shapes. With real variable-length batches the driver figure grew to 11.3 GB, because the MPS allocator caches blocks per distinct shape. **Batches are therefore padded to a multiple of 64 tokens and formed by a token budget** (`max_tokens_per_batch` in each arm's config, default 2,048 for a3 and a4), not by a fixed micro-batch size.
- Every run logs `cost/peak_device_mem_mb` (MPS tensor memory, `torch.mps.current_allocated_memory`), `cost/peak_driver_mem_mb` (`torch.mps.driver_allocated_memory`) and `cost/peak_rss_mb`. MPS has no peak counter, so `shared/cost.py` samples in a background thread.
- Memory levers, exposed in each arm's `config.yaml`:
  1. Small micro-batch with gradient accumulation to reach the effective batch size.
  2. Gradient checkpointing (on by default for a3).
  3. Dynamic padding and length-sorted batching instead of padding everything to 512.
  4. bf16 weights and activations.
  5. A memory-lean optimizer (for example Adafactor) if AdamW state does not fit.
- **Fallback ladder for a3 (full fine-tune).** If levers 1–5 are not enough, a3 is marked infeasible locally. In order: (a) freeze the embeddings and lower layers and document it as a **deviation** (`a3_laya_partial_ft`, logged as its own arm), (b) run a3 alone on a rented or free GPU (for example Kaggle T4) with the `device_name` tag set so the report flags the different hardware. Do not silently change the method.
- a2 (frozen encoder) and a4 (LoRA) are expected to fit comfortably.
- **M0 outcome: a3 is a go locally.** No fallback-ladder step is needed. AdamW fits, so Adafactor is not used (it saves about 3 GB but is about 30% slower). a4 runs **without** gradient checkpointing by default (6.6 vs 4.1 ex/s); with the token budget above it peaks at 5.2 GB driver at 256 tokens and 9.2 GB at 512. If real-data runs breach the driver budget, a4 turns checkpointing on and logs it.

### 8.3 Run scheduling and time budget
- **One run at a time.** The single device cannot run sweeps in parallel.
- M0 measures training throughput (examples/sec) per arm at the chosen micro-batch size and extrapolates total sweep time. If the total exceeds the available time, scope is reduced in this order: (1) fewer HP trials, (2) 2 seeds at fraction 1.0 (keep 3 at smaller fractions), (3) drop the 25% fraction, (4) cap epochs.
- Sweeps must be resumable (Section 9) because laptop sessions get interrupted. The sweep script runs under `caffeinate` to prevent sleep.
- Thermal throttling and battery state affect wall-clock. M0 found sustained-load throughput 21–27% below a rested machine, on AC both times (likely thermal), so timing comparisons only hold between runs made under the same conditions. **AC power is required** for multi-hour runs: `make sweep` and `make spike` refuse to start on battery, and `on_ac_power` is logged as a tag. Wall-clock, latency and throughput comparisons are only valid within this machine and are labeled as such in the report.
- **M0 estimate:** about 45 h of machine time at sustained-load speed for the full protocol with the LoRA rank tuned inside a4's HP budget (a3 ≈ 27 h, a4 ≈ 17 h, everything else < 2 h). A full rank × fraction grid for a4 adds about 25 h. Assumptions and the per-arm breakdown are in `docs/m0_spike.md`. M5 is planned as several overnight sessions.
- Unit tests, data prep and baselines (a0) also run on this machine (CPU is fine for a0 and tests).

### 8.4 Project tooling: uv and Makefile
- **uv is the only environment and dependency manager.** No `pip install`, `conda` or hand-made virtualenvs. The Python version is pinned in `.python-version` (3.11 or 3.12, whichever M0 confirms has working torch/MPS wheels), and `requires-python` in `pyproject.toml` matches it.
- `pyproject.toml` declares the extras from Section 4.2 (`baselines`, `laya`, `jev`) and a `dev` dependency group (pytest, ruff, mypy). `uv.lock` is committed and kept current. CI and every Makefile target use `uv sync --locked` / `uv run`, so a stale lockfile fails rather than silently re-resolving.
- Every command in this document is run through `uv run` (for example `uv run python -m scripts.run ...`). Bare `python` is only used inside the uv-managed `.venv`.
- **A top-level `Makefile` is the entry point for every routine task.** Targets call `uv run` and contain no logic that is not also reachable from the Python CLIs. Required targets:

| Target | Does |
|---|---|
| `make help` | Lists targets with one-line descriptions (default target) |
| `make setup` | Installs uv-managed Python if needed, `uv sync --locked --all-extras --group dev` |
| `make lock` | `uv lock`, the only sanctioned way to change pins |
| `make lint` / `make format` | `ruff check` and `ruff format --check` / `ruff format` and `ruff check --fix` |
| `make typecheck` | `mypy shared arms scripts` |
| `make test` | `pytest` on CPU (forces `device=cpu`) |
| `make check` | `lint` + `typecheck` + `test`; what CI runs |
| `make data` | `scripts.prepare_data`: download, audit, write split manifest |
| `make run ARM=... STAGE=... SEED=... FRACTION=... SET="..."` | One run through `scripts.run` |
| `make sweep ARM=... STAGE=...` | `scripts.sweep` wrapped in `caffeinate -dims` |
| `make aggregate` | `scripts.aggregate` into `reports/` |
| `make mlflow-ui` | MLflow UI against the configured tracking URI |
| `make spike` | Re-runs the M0 spike scripts in `spikes/m0/` |
| `make clean` | Removes caches and build artifacts. Never deletes `data/`, `mlflow.db` or `mlartifacts/` |

- Make variables have defaults from `configs/base.yaml` so `make run ARM=a0_majority` works with nothing else set. Targets that need MPS set `PYTORCH_ENABLE_MPS_FALLBACK=1`. Secrets are read from `.env` by the Python code, never echoed by make.

---

## 9. Reproducibility and security

- Fixed seeds everywhere. Deterministic data loading and split manifests. Log whether deterministic kernels were enabled.
- Every run records git SHA, dirty flag, data hash and the full resolved config.
- Secrets only via environment variables or a gitignored `.env`. Never in config files, MLflow params or logs.
- Sweeps are resumable: skip completed (arm, seed, fraction, variant, stage) combinations by querying MLflow.

---

## 10. Testing and acceptance criteria

### Tests (`tests/`, run on CPU in CI against a small fixture of about 50 examples)
1. **Metrics:** match scikit-learn for accuracy, F1, MCC and log loss. ECE and Brier verified on hand-computed cases.
2. **Splits:** deterministic given a seed. Fractions are nested and stratified. Train, val and test are disjoint. The manifest hash is stable.
3. **Interface conformance:** a parametrized test runs every arm that can load on CPU (using stubs or tiny models) through `fit` and `predict`, and checks the `Predictions` contract (shape, probabilities sum to 1, IDs align).
4. **Isolation:** no arm imports another arm. Only `shared/mlflow_utils.py` imports `mlflow`.
5. **Test guard:** a `dev` run cannot load test labels.
6. **MLflow logging:** a temp-backend run produces the expected params, tags, metrics and artifacts.
7. **Jev client:** mocked HTTP, cache hit and miss behaviour, retry logic, key never appears in logs.

### Acceptance checklist
- [ ] All 7 arms run end-to-end through `scripts/run.py` with the same CLI.
- [ ] A fresh clone plus `make setup` plus `make data` reproduces the split hash.
- [ ] `make check` passes from a fresh clone, and every Section 8.4 Makefile target exists and works.
- [ ] Each arm folder can be read and understood without opening any other arm folder.
- [ ] Peak device memory for every arm stays within the Section 8.2 budget, or the deviation is documented.
- [ ] MLflow contains, for every final run: all metric groups in Section 5, the artifacts, and the tags.
- [ ] `scripts/aggregate.py` produces the results table (mean ± std, CI), learning curves with a1 and a5 reference lines, the McNemar matrix, calibration plots and the cost table.
- [ ] `reports/data_audit.md` exists (duplicates, truncation, class balance).
- [ ] All tests pass on CPU. Lint and type checks are clean.

---

## 11. Open questions and risks

| # | Item | Resolution |
|---|---|---|
| 1 | **Laya integration path:** can the encoder be wrapped in a standard `AutoModelForSequenceClassification`-style head (enables HF Trainer and PEFT), or must fine-tuning go through Laya's typed-decision heads? | **Resolved in M0.** Yes: Laya's encoder loads strictly into `ModernBertForSequenceClassification` and gives identical hidden states. a2–a4 use that path; see `docs/adr/0001-laya-integration.md`. |
| 2 | **Jev early-access limits, pricing and version naming** | **Answered from docs in M0 (2026-10-05), see `docs/m0_spike.md`.** Pin `jev-1.13.0` (`jev-latest`/`jev-preview` are aliases of it). $0.042 per 1M input tokens, output free; ~$0.15 per full a5 run. Limits 100K tok/s, 80 req/s, adjusting during early access. Key env var `TYPESAFE_API_KEY`. Probabilities can be exactly 0, so log loss must clip. **Still open:** waitlist access for this account. |
| 3 | **MPS compatibility:** operator coverage, bf16 stability and `peft` behaviour on MPS for Laya's ModernBERT backbone | **Resolved in M0.** All ops run natively (no CPU fallback needed), PEFT LoRA trains on MPS, and bf16 autocast is stable and faster (§8.1). |
| 3b | **Memory ceiling on 16GB unified memory:** full fine-tuning (a3) may not fit | **Resolved in M0.** a3 fits (§8.2); the budget is restated in tensor and driver terms. |
| 3c | **Sweep time on a single laptop** and thermal throttling | **Resolved in M0:** about 45 h at sustained-load speed (§8.3). Sustained load ran ~25% slower than a rested machine; throttling over multi-hour runs is still unmeasured, so watch `cost/train_seconds` drift in M4. |
| 4 | **LoRA target module names** for Laya's ModernBERT backbone | **Resolved in M0:** `["Wqkv", "Wo", "Wi"]` (all attention and MLP linears, 28 layers); embeddings frozen. |
| 8 | **LoRA rank sweep scope:** tune r as an HP, or report the full rank curve | Owner decision (`docs/m0_spike.md`). Default until decided: rank tuned inside a4's 6-trial budget at fraction 1.0. |
| 5 | **Pretraining contamination:** whether Laya or Jev saw this corpus is unknowable | State as a limitation in the report. |
| 6 | **Label noise ceiling:** the corpus labels are noisy, and the "general pathological conditions" class is a catch-all | Error analysis. Interpret small differences cautiously. |
| 7 | **HP budget parity** between methods with very different cost | Document trial counts. Report cost columns alongside quality. |

---

## 12. Milestones

| M | Deliverable |
|---|---|
| M0 ✅ | **Done 2026-10-06** (`docs/m0_spike.md`, ADR 0001). Spike on the M5 Mac: load Laya on MPS, check the integration path (open question 1), LoRA target modules, bf16 support, **peak memory and throughput for a2/a3/a4 at several micro-batch sizes**, sweep time estimate. Update the memory budget and the a3 go/no-go. ADR written. |
| M1 | `shared/` core: interface, config, data, splits, metrics, calibration, MLflow utils, runner. Tests green. `prepare_data.py` and the data audit. |
| M2 | a0 baselines and a1 Laya zero-shot, including the label-wording sweep. First MLflow runs. |
| M3 | a5 Jev API arm with cache. Zero-shot reference lines on val. |
| M4 | a2, a3, a4 implemented. HP tuning on val. |
| M5 | Full sweep: seeds × fractions at `stage=final`. |
| M6 | `aggregate.py`, error analysis, results report. |
