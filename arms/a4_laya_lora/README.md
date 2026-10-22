# a4: Laya + LoRA

Laya's encoder in `ModernBertForSequenceClassification` (ADR 0001, path A), wrapped with PEFT LoRA
(`SEQ_CLS`) on `Wqkv`, `Wo`, `Wi` in all 28 layers. The new head and classifier are fully trained
(`modules_to_save`). Rank r ∈ {4, 8, 16, 32} is part of the HP space.

- **Deps:** `laya` extra (torch, transformers ≥ 5, peft, laya). No QLoRA: `bitsandbytes` has no
  macOS build.
- **Trainable params (M0):** r=4/8/16/32 → 2.85M / 4.65M / 8.25M / 15.45M.
- **Memory:** gradient checkpointing **off** by default (faster; peaks 5.2 GB driver at 256 tokens,
  9.2 GB at 512). If real batches breach the 12 GB driver budget, set
  `train.gradient_checkpointing: true` and it is logged with the config.
- **Artifacts:** the LoRA adapter is always saved and logged; `cost/adapter_mb` records its size.
- `lora.alpha_ratio` in `hp_space` means `alpha = ratio * r`; the tuner resolves it.
