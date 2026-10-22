.DEFAULT_GOAL := help
SHELL := /bin/bash

UV        ?= uv
RUN       := $(UV) run
MPS_ENV   := PYTORCH_ENABLE_MPS_FALLBACK=1 PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.1 \
             PYTORCH_MPS_LOW_WATERMARK_RATIO=0.99 USE_TF=0 TOKENIZERS_PARALLELISM=false

ARM       ?= a0_majority
STAGE     ?= dev
SEED      ?= 0
FRACTION  ?= 1.0
SET       ?=
SEEDS     ?= 0 1 2
FRACTIONS ?= 0.01 0.05 0.1 0.25 1.0
EXPERIMENT ?= laya-medabs-bench
MLFLOW_TRACKING_URI ?= sqlite:///mlflow.db
export MLFLOW_TRACKING_URI

.PHONY: help setup lock lint format typecheck test check data run tune sweep aggregate mlflow-ui spike clean

help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n",$$1,$$2}'

setup: ## Install Python (if needed) and sync the locked environment with all extras + dev
	$(UV) python install
	$(UV) sync --locked --all-extras --group dev

lock: ## Re-resolve and update uv.lock
	$(UV) lock

lint: ## ruff check + format check
	$(RUN) ruff check .
	$(RUN) ruff format --check .

format: ## Apply ruff formatting and autofixes
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

typecheck: ## mypy over shared, arms, scripts
	$(RUN) mypy $(wildcard shared arms scripts)

test: ## pytest on CPU
	LAYA_BENCH_DEVICE=cpu $(RUN) pytest -q

check: lint typecheck test ## What CI runs

data: ## Download, audit, write split manifest
	$(RUN) python -m scripts.prepare_data

run: ## One run: make run ARM=a4_laya_lora STAGE=dev SEED=0 FRACTION=0.1 SET="lora.r=16"
	$(MPS_ENV) $(RUN) python -m scripts.run --arm $(ARM) --stage $(STAGE) --seed $(SEED) \
		--fraction $(FRACTION) $(if $(SET),--set $(SET))

tune: ## HP tuning on val at fraction 1.0 (6 trials): make tune ARM=a4_laya_lora
	@pmset -g batt | grep -q "AC Power" || { echo "plug in to AC power first (multi-hour MPS runs)"; exit 1; }
	$(MPS_ENV) caffeinate -dims $(RUN) python -m scripts.tune --arm $(ARM) $(if $(SET),--set $(SET))

sweep: ## Grid of runs under caffeinate (AC power required): make sweep ARM=a4_laya_lora STAGE=final
	@pmset -g batt | grep -q "AC Power" || { echo "plug in to AC power first (multi-hour MPS runs)"; exit 1; }
	$(MPS_ENV) caffeinate -dims $(RUN) python -m scripts.sweep --arm $(ARM) --stage $(STAGE) \
		--seeds $(SEEDS) --fractions $(FRACTIONS)

aggregate: ## Build tables and plots from MLflow into reports/
	$(RUN) python -m scripts.aggregate --experiment $(EXPERIMENT) --out reports/

mlflow-ui: ## MLflow UI on the configured tracking URI
	$(RUN) mlflow ui --backend-store-uri $(MLFLOW_TRACKING_URI)

spike: ## Re-run the M0 spike on AC power (~1.5 h; bench resumes from results/bench.jsonl)
	@pmset -g batt | grep -q "AC Power" || { echo "plug in to AC power first (multi-hour MPS runs)"; exit 1; }
	$(MPS_ENV) $(RUN) python -m spikes.m0.inspect_model
	$(MPS_ENV) $(RUN) python -m spikes.m0.data_lengths
	$(MPS_ENV) $(RUN) python -m spikes.m0.zeroshot_smoke
	$(MPS_ENV) caffeinate -dims $(RUN) python -m spikes.m0.bench
	$(MPS_ENV) caffeinate -dims $(RUN) python -m spikes.m0.precision_check
	$(RUN) python -m spikes.m0.estimate

clean: ## Remove caches (never data/, mlflow.db or mlartifacts/)
	rm -rf .pytest_cache .mypy_cache .ruff_cache build dist
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
