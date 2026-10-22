# a1: Laya zero-shot

Asks Laya (`laya.load("convaiinnovations/laya")`, pinned to `laya.revision` in `configs/base.yaml`)
one typed `choice` question per abstract, with the 5 classes as options. No training.

- **Wording sweep.** `fit` scores every variant in `configs/prompts/labels.yaml` (shared with a5) on
  val, logs `wording/<variant>/val_macro_f1` and `wording/spread_val_macro_f1` (max − min), and keeps
  the best by val macro-F1. Only the selected variant is used by `predict`.
- **Option-order check.** The selected variant is re-scored on val with the options in reverse
  order (laya's `option_order`), logged as `wording/order_reversed_val_macro_f1`.
- **Outputs.** Laya's per-option probabilities (already temperature-scaled by the runtime, rounded to
  4 decimals), mapped to class-id order and renormalised. The runtime exposes no raw logits, so
  `logits=None` and calibration is reported raw only.
- **Truncation.** The question head leaves about 472 of the 512 tokens for the abstract;
  `data/truncated_pct` is logged per scored split.

Deps: the `laya` extra (torch, transformers, laya). Precision follows laya's own MPS policy (fp16
autocast); the device comes from `ctx.device`.
