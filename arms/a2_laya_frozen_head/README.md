# a2: frozen Laya encoder + head

Laya's encoder in `ModernBertForSequenceClassification` (ADR 0001, path A) with the encoder frozen.
Pooled encoder features are computed **once per run** for train/val (and per predicted split), then
only `head` + `classifier` (1.05M params) train on them. Pooling (`mean`/`cls`) is a config value.

- **Deps:** `laya` extra (torch, transformers ≥ 5, laya).
- **Memory:** inference only on the encoder; M0 peaked at about 3 GB driver.
- Because the encoder is frozen and dropout sits after pooling, cached features are exact: the
  result equals training the full module with the encoder frozen.
