# Data audit

Split manifest hash: `84fda3400fd151ed62e04fd048cd8503eed40c05854d2dfd49051abfa4b2a1a0`

## Class distribution

| Class | train_pool | val | test |
|---|---|---|---|
| Neoplasms | 2277 (21.9%) | 253 (21.9%) | 633 (21.9%) |
| Digestive system diseases | 1075 (10.3%) | 120 (10.4%) | 299 (10.4%) |
| Nervous system diseases | 1386 (13.3%) | 154 (13.3%) | 385 (13.3%) |
| Cardiovascular diseases | 2197 (21.1%) | 244 (21.1%) | 610 (21.1%) |
| General pathological conditions | 3460 (33.3%) | 384 (33.2%) | 961 (33.3%) |
| **Total** | 10395 | 1155 | 2888 |

## Token lengths

Counted with: Laya tokenizer (`convaiinnovations/laya`@`7b928d828b7b0e022f929d9bd2e44165aa270148`). Truncation limit: 512 tokens.

| Split | p50 | mean | p95 | p99 | max | % > 512 |
|---|---|---|---|---|---|---|
| train_pool | 240 | 248 | 433 | 541 | 790 | 1.55% |
| val | 243 | 249 | 432 | 531 | 646 | 1.56% |
| test | 242 | 250 | 431 | 559 | 650 | 1.90% |

## Duplicates

`exact` = identical text; `near` = identical after lowercasing and stripping punctuation; `near_prefix` = identical first 30 normalized words. `a~a` counts repeats within a split, `a~b` counts shared texts across splits. `texts_with_conflicting_labels` counts distinct normalized texts that appear with more than one label (the corpus repeats multi-topic abstracts once per label).

| Check | Count |
|---|---|
| near/texts_with_conflicting_labels | 2929 |
| exact/train_pool~train_pool | 1722 |
| exact/train_pool~val | 359 |
| exact/train_pool~test | 900 |
| exact/val~val | 24 |
| exact/val~test | 110 |
| exact/test~test | 118 |
| near/train_pool~train_pool | 1723 |
| near/train_pool~val | 359 |
| near/train_pool~test | 900 |
| near/val~val | 24 |
| near/val~test | 110 |
| near/test~test | 118 |
| near_prefix/train_pool~train_pool | 1723 |
| near_prefix/train_pool~val | 360 |
| near_prefix/train_pool~test | 900 |
| near_prefix/val~val | 24 |
| near_prefix/val~test | 110 |
| near_prefix/test~test | 118 |

## License

Data: Medical Abstracts TC Corpus (Schopf et al., `TimSchopf/medical_abstracts`), licensed CC BY-SA 3.0.
