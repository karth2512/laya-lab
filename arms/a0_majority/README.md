# a0_majority

Predicts the most frequent class in the training subset for every example (one-hot probabilities).
It is the accuracy floor (~33.3% on test, "General pathological conditions").

- Deps: base only. CPU.
- Quirks: probabilities are exact 0/1, so log loss relies on the shared clipping. No logits, so no
  temperature scaling.
