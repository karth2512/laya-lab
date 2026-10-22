# ADR 0001: Laya integration path for the trained arms (a2, a3, a4)

- **Status:** Accepted (M0, 2026-10-06)
- **Answers:** open question 1 (and 4, LoRA targets), REQUIREMENTS.md §11
- **Evidence:** `spikes/m0/inspect_model.py` → `spikes/m0/results/inspect.json`

## Context

Laya (`convaiinnovations/laya`, laya 0.3.28) is not a `transformers` model class. `laya.load()` returns an `Agent` wrapping `laya.common.DecisionModel`, which is:

| Part | Params | What it is |
|---|---|---|
| `encoder` | 394.8M | a stock `transformers` `ModernBertModel` (28 layers, d=1024, SDPA attention), fully fine-tuned by Laya's RLCD training |
| `head` | 25.2M | 2 × `nn.TransformerEncoderLayer` (uses `nn.MultiheadAttention`) |
| `scorer`, `type_emb`, `act_head` | 1.3M | scores each option at its own `[MASK]` marker; act/escalate head |
| **total** | **421.3M** | |

Input format for the typed-decision head: `[CLS] choice question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 … [SEP] <state> [SEP]`. The question head takes part of the 512-token budget (40 tokens for our 5-option question, leaving 472 for the abstract).

We had two options for a2–a4:

- **A. Standard classifier on Laya's encoder.** Build `transformers.ModernBertForSequenceClassification` from the encoder config, load Laya's encoder weights into `.model`, then train a new mean-pooled head over 5 labels (1.05M params).
- **B. Fine-tune Laya's own typed-decision head.** Keep `DecisionModel`, express each label as a typed `choice` option, and train with `laya.train` (soft-CE or RLCD over the option markers).

## What M0 measured

- **A loads exactly.** `clf.model.load_state_dict(decision_model.encoder.state_dict())`: 0 missing and 0 unexpected keys. Hidden states match Laya's encoder with max abs diff 0.0 on MPS (fp32).
- **A works with PEFT and HF tooling.** LoRA wraps it with `task_type="SEQ_CLS"` and runs forward and backward on MPS with finite grads on all 224 LoRA tensors. LoRA and full-FT training steps also ran with `PYTORCH_ENABLE_MPS_FALLBACK` unset, with no fallback warnings (`spikes/m0/bench.py`).
- **B also accepts PEFT** (it is a plain `nn.Module`; r=16 gives 8.25M trainable params). The head's `nn.MultiheadAttention` packs Q/K/V into `in_proj_weight`, though, so LoRA can't target the head's attention without PEFT's experimental MHA support. Training B also means building on `laya.train` internals (`encode_item`, `collate_items`, `_forward`, RLCD loss), which are not a public, versioned API.

## Decision

**Use path A for a2, a3 and a4.** One code path, `ModernBertForSequenceClassification` initialised from Laya's encoder, shared through `shared/` (a small `laya_backbone` loader). Arms differ only in what is trainable:

| Arm | Trainable |
|---|---|
| a2 frozen head | new `head` + `classifier` (1.05M); encoder features cached once |
| a3 full FT | everything (395.8M) |
| a4 LoRA | LoRA on `Wqkv`, `Wo`, `Wi` + `head`/`classifier` via `modules_to_save` |

Weights are always taken from the Laya checkpoint via `laya.train.load_checkpoint`, pinned to Hub revision `7b928d828b7b0e022f929d9bd2e44165aa270148`. We never use `answerdotai/ModernBERT-large`, so every arm measures *Laya's* encoder.

**LoRA targets (question 4):** `target_modules=["Wqkv", "Wo", "Wi"]`, where `"Wo"` matches both `attn.Wo` and `mlp.Wo` (28 each). Embeddings stay frozen. Trainable params, including the 1.05M head:

| r | 4 | 8 | 16 | 32 |
|---|---|---|---|---|
| trainable | 2.85M (0.72%) | 4.65M (1.16%) | 8.25M (2.04%) | 15.45M (3.76%) |
| adapter, fp32 | 10.9 MB | 17.8 MB | 31.5 MB | 58.9 MB |

## Rationale

1. **Comparability.** a2/a3/a4 share the same head, pooling, loss (plain CE) and input format, so differences between them come from the adaptation method alone.
2. **Tooling.** HF Trainer, PEFT `SEQ_CLS` and `save_pretrained` for adapters work unmodified (REQUIREMENTS §4.1 and §5 artifacts).
3. **Stability of dependencies.** Path B would tie the arms to private functions in `laya.train`, which changes often (0.3.20 → 0.3.28 in weeks).
4. **The a1 arm still covers Laya's head.** Zero-shot a1 goes through `laya.load(...).predict_batch` with the typed-choice question, so the benchmark still measures Laya's native decision path.

## Consequences

- We discard the 26.5M-param decision head and its zero-shot prior for the trained arms. At the 1%–5% fractions, a fine-tuned typed head (path B) might beat a freshly initialised linear head. That is a legitimate question but **out of scope**. If it is wanted later, it is a new, separately named arm (for example `a6_laya_typed_head_ft`), never a silent change to a3/a4.
- Pooling is ModernBERT's default `classifier_pooling="mean"`. Changing it to `cls` is a hyperparameter in the arm configs, not a code change.
- `transformers>=5` is required (Laya encoder configs use `rope_parameters`; the laya runtime remaps them for 4.x, our path A does not).
