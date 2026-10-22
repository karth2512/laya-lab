# a3: Laya full fine-tune

Laya's encoder loaded into `ModernBertForSequenceClassification` (ADR 0001, path A) with a new
mean-pooled head; **all 395.8M parameters train**. Early stopping on val macro-F1.

- **Deps:** `laya` extra (torch, transformers ≥ 5, laya).
- **Memory (M0):** gradient checkpointing on, token-budget batches of 2,048 tokens padded to a
  multiple of 64, bf16 autocast on MPS, AdamW. Peaks about 6.1 GB tensors / 8.0 GB driver; real
  variable-length batches reached 11.3 GB driver before token budgeting.
- **Weights** are saved only with `eval.save_model: true` (`--save-model`).
- **Fallback ladder (§8.2):** if it ever stops fitting, the partial fine-tune is a separate arm
  (`a3_laya_partial_ft`), never a silent change here.
