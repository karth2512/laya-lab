# a0_tfidf_logreg

TF-IDF features + multinomial logistic regression (scikit-learn), CPU only.

- `fit` tries every entry in `trials` (config.yaml, 6 = the shared HP budget): fit on train, score
  macro-F1 on val, keep the best, log each trial as `hp/val_macro_f1` (step = trial index).
- The chosen settings are reported in `FitReport.extra`; trainable params = coefficients + intercepts.
- `predict` returns logits (`decision_function`), so the runner can fit temperature on val.
- Deps: base only (`scikit-learn`).
- Quirks: `min_df` > 1 can drop every token on tiny fractions; such trials are skipped.
