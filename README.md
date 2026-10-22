# laya-bench

Benchmark of Laya (`convaiinnovations/laya`, ModernBERT-large encoder) as a text classifier on the
Medical Abstracts TC Corpus: zero-shot, frozen head, full fine-tuning and LoRA, against classical
floors and the Jev API. Every run is logged to MLflow. The full spec is `REQUIREMENTS.md`; the work plan
and progress are in `tasks.md`.

## Quick start

```bash
make setup                     # uv-managed Python + locked env (all extras + dev)
make data                      # download, audit, write data/splits/split_v1.json
make run ARM=a0_tfidf_logreg   # one dev run (seed 0, fraction 1.0)
make run ARM=a4_laya_lora STAGE=dev FRACTION=0.1 SET="arm.lora.r=16"
make sweep ARM=a4_laya_lora STAGE=final   # seeds × fractions, resumable, needs AC power
make aggregate                 # tables and plots into reports/
make mlflow-ui
make check                     # lint + typecheck + tests (CPU)
```

`make help` lists every target. All commands run through `uv`; never `pip install`.

## Arms

| ID | Folder | Method |
|---|---|---|
| a0 | `arms/a0_majority` | Majority class |
| a0 | `arms/a0_tfidf_logreg` | TF-IDF + logistic regression |
| a1 | `arms/a1_laya_zeroshot` | Laya typed-choice zero-shot, label-wording sweep |
| a2 | `arms/a2_laya_frozen_head` | Frozen Laya encoder + trained head |
| a3 | `arms/a3_laya_full_ft` | Full fine-tuning |
| a4 | `arms/a4_laya_lora` | LoRA, r ∈ {4, 8, 16, 32} |
| a5 | `arms/a5_jev_api` | Jev API zero-shot (needs `TYPESAFE_API_KEY` in `.env`) |

Each arm folder has its own `README.md`. Arms share code only through `shared/` and never import
each other. Only `shared/mlflow_utils.py` imports `mlflow`.

## Layout

`shared/` (interface, data, splits, metrics, runner, …), `arms/`, `scripts/` (CLIs), `configs/`,
`tests/`, `docs/` (M0 spike report, ADRs), `reports/` (generated). `data/`, `mlflow.db` and
`mlartifacts/` are gitignored.

## Hardware

Targets one Apple M5 Mac with 16 GB unified memory (PyTorch MPS, CPU fallback). Timing and memory
numbers are only comparable between runs on the same machine under the same conditions.

## Limitations

- Whether Laya or Jev saw this corpus during pretraining is unknowable (possible contamination).
- Labels are noisy and "General pathological conditions" is a catch-all, so small differences
  should be read cautiously.
- Methods have very different costs; HP budgets are equal in trial count, and cost is reported next to
  quality.

## Data and license

Medical Abstracts TC Corpus by Tim Schopf et al. (`TimSchopf/medical_abstracts`), licensed
CC BY-SA 3.0. See: Schopf, Braun, Matthes, "Evaluating Unsupervised Text Classification:
Zero-shot and Similarity-based Approaches", NLPIR 2022.
