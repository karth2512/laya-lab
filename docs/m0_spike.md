# M0 spike report

**Date:** 2026-10-05 to 2026-10-06 · **Machine:** Apple M5, 16 GB unified memory, macOS 26.6 · **Stack:** Python 3.12.15, torch 2.14.1 (MPS), transformers 5.18.0, peft 0.21.2, laya 0.3.28 · **Model:** `convaiinnovations/laya` @ `7b928d828b7b0e022f929d9bd2e44165aa270148`

Scripts are in `spikes/m0/`, raw results in `spikes/m0/results/`, and `make spike` re-runs everything (about 1.5 h, AC power required). The official **test split was never touched**: every quality number below comes from samples of the official train split. Those quality numbers are smoke checks, not results.

## Answers to the open questions

| # | Question | Answer | Status |
|---|---|---|---|
| 1 | Laya integration path | Laya's encoder is a stock `ModernBertModel`. It loads into `ModernBertForSequenceClassification` with 0 missing or unexpected keys and identical hidden states. **Path A is chosen for a2–a4** (ADR 0001). | Closed |
| 2 | Jev limits, pricing, versions | Pin `jev-1.13.0`, the only version (`jev-latest` and `jev-preview` are aliases of it). The model is set by a `model` field in the body. $0.042 per 1M input tokens, output free, so a full a5 run (5 wordings on val + test, about 2.8M tokens) costs about $0.12. Limits are 100K tok/s and 80 req/s, "adjusting dynamically". Max 64K tokens per request and 255 options. `POST https://api.typesafe.ai/v1/systemone`, `Authorization: Bearer`, env var `TYPESAFE_API_KEY`. Responses carry per-option `probabilities` and can contain exact 0.0, so log loss needs clipping. | Closed, except **account access**: early access is waitlist-only |
| 3 | MPS compatibility, bf16, peft | Every a2/a3/a4 op runs natively on MPS: no fallback warnings in 68 configs, and runs pass with `PYTORCH_ENABLE_MPS_FALLBACK` unset. SDPA attention is used. PEFT LoRA trains on MPS. **bf16 autocast is stable** (details below) and **1.35–1.55× faster** than fp32. | Closed |
| 3b | Memory ceiling, a3 go/no-go | **a3 is a go locally.** Full FT with AdamW and gradient checkpointing peaks at 8.0 GB driver / 6.1 GB tensors (bf16, 256 tok, micro-batch 16) and 8.1 GB at 512 tok with micro-batch 8. No fallback-ladder step is needed. | Closed (see the memory caveat) |
| 3c | Sweep time | About **45 h** of machine time for the full protocol at sustained-load speed, if the LoRA rank is tuned inside the HP budget (breakdown below). | Closed; scope choice needed (below) |
| 4 | LoRA target modules | `["Wqkv", "Wo", "Wi"]`, covering all attention and MLP linears in the 28 layers. r=4/8/16/32 gives 0.72/1.16/2.04/3.76% trainable. | Closed |
| 5–7 | Contamination, label noise, HP parity | Not spike items. Carried into the report as limitations. | Unchanged |

## Findings in detail

### Data (`data_lengths.json`)
- Official split 11,550 train / 2,888 test. Columns are `condition_label` (1–5) and `medical_abstract`, and the class counts match REQUIREMENTS §2.
- Laya tokenizer lengths (train): p50 238, mean 246, p95 431, p99 539, max 788.
- Truncation: 1.5% (train) and 1.9% (test) at 512 for path A. With the zero-shot typed-choice layout it's 2.8% and 2.6%, because the question head uses 40 tokens and leaves 472.
- Padding matters a lot. Random batches of 16 pad to a mean of 445 tokens, but length-bucketed batches padded to a multiple of 64 average about 280. In the real-data check, unbucketed training ran at about half the speed of the fixed-length benchmark.

### Zero-shot a1 smoke (`zeroshot_smoke.json`, n=250 balanced train sample)
- `laya.load(..., device="mps")` and `predict_batch` work. Accuracy is 0.504 and macro-F1 0.460, against 0.20 for chance on this balanced sample. Predictions are skewed toward "Digestive system diseases" (91/250).
- fp16, bf16 and fp32 give identical argmax on 250/250 examples. Max prob drift is 0.015 for fp16 and 0.025 for bf16. Throughput at batch 16 is 25.9 ex/s (fp16), 24.4 (bf16) and 11.7 (fp32). Laya's own MPS default is fp16 autocast from 5 rows up.
- Batch-1 latency is p50 88 ms and p95 145 ms.
- The checkpoint ships a `choice:11+` temperature of 0.10, which the runtime clamps to 0.5 with a warning. That doesn't affect our 5-option questions.

### Memory and throughput (`bench.jsonl`, `bench_ac_recheck.jsonl`)
Each config ran in a fresh process with synthetic tokens at a fixed length and the MPS cap at 1.1 × recommended max (13.0 GB). "Driver" is `torch.mps.driver_allocated_memory()`; "tensors" is `current_allocated_memory()`. Selected rows, measured mid-grid under sustained load:

| Arm | Config | ex/s @256 | Peak driver / tensors |
|---|---|---|---|
| a2 frozen | bf16, batch 64 | 23.2 | 3.0 / 2.0 GB |
| a3 full FT | fp32, ckpt, mb 16 | 2.67 | 8.0 / 6.1 GB |
| a3 full FT | **bf16, ckpt, mb 16** | **4.12** | 8.0 / 6.1 GB |
| a4 LoRA r16 | fp32, ckpt, mb 8 | 2.78 | 3.3 / 2.0 GB |
| a4 LoRA r16 | bf16, ckpt, mb 8 | 4.11 | 3.2 / 2.0 GB |
| a4 LoRA r16 | **bf16, no ckpt, mb 4** | **6.59** | 5.2 / 4.9 GB |

The a3 fp32 rows ran on battery in the first 8 minutes of the grid; every other row ran on AC. The a4 no-checkpointing row was measured separately, right after 30 minutes of load.

- OOM at 13 GB: no-checkpointing at 512 tok with micro-batch 8, for both a3 and a4 in both precisions.
- At 512 tokens, a3 fits with checkpointing up to micro-batch 16 (9.1 GB). a4 without checkpointing fits up to micro-batch 4 (9.2 GB).
- With checkpointing on, LoRA is barely faster than full FT. Backprop still runs through all 28 frozen layers, and checkpointing recomputes them. LoRA's speed advantage only shows with checkpointing off, which its small optimizer state makes affordable.
- LoRA rank barely affects cost: r=32 vs r=16 is within 5% on speed and memory.
- Adafactor (lever 5) cuts a3 memory by about 3 GB but is about 30% slower. It isn't needed.
- **Sustained load vs rested machine:** re-running four configs on a rested machine (`bench_ac_recheck.jsonl`) gave 21–27% more throughput than the same configs mid-grid. Two of the pairs were on AC both times (a4 fp32 2.78 → 3.51 ex/s; a2 fp32 12.6 → 16.0), so power state does not explain the gap. Thermal throttling under sustained load is the likely cause, but it was not measured directly. The estimate below uses the sustained-load numbers. **The battery-vs-AC effect was not isolated.** AC power is still required for sweeps because a 45 h protocol cannot run on battery.

### bf16 training stability (`precision_check.json`)
Real abstracts: 400-example stratified train sample, 2 epochs, micro-batch 8, checkpointing on, same seed and data order; eval on 250 disjoint train examples.

| Arm | Precision | Loss first→last 10 steps | Eval macro-F1 | ex/s |
|---|---|---|---|---|
| a4 r16 | fp32 | 3.16 → 0.88 | 0.568 | 2.02 |
| a4 r16 | bf16 | 3.12 → 0.87 | 0.580 | 2.71 |
| a3 | fp32 | 1.72 → 0.46 | 0.595 | 1.52 |
| a3 | bf16 | 1.70 → 0.43 | 0.586 | 2.20 |

The loss curves track each other step for step and no loss was non-finite. The F1 gaps are well inside the sampling noise at n=250 (SE about 0.03). **Decision: bf16 autocast for a2–a4 (weights and optimizer state stay fp32). fp16 with a grad scaler is not used.**

**Memory caveat found here:** with real, variable-length batches, a3's *driver* memory reached 11.3 GB while tensor memory stayed at 6.3 GB. The MPS allocator caches one block set per distinct shape. The fixed-shape benchmark could not show this. Mitigations, adopted below: pad to multiples of 64, which gives at most 8 shapes; batch by a **token budget** rather than a fixed micro-batch; and keep the 1.1 high-watermark cap so a run fails fast instead of swapping.

### Sweep-time estimate (`estimate.json`)
Assumptions, all constants in `estimate.py`: sustained-load throughput; length-bucketed batches (mean 280 tok); 3 epochs at fraction 1.0; a floor of 200 optimizer steps at effective batch 16; 6 HP trials at fraction 1.0; 3 seeds × 5 fractions at final; val eval each epoch.

| Arm | Config | One run @ f=1.0 | HP tuning | Final sweep | Total |
|---|---|---|---|---|---|
| a0, a1, a2 | | | | | < 0.5 h |
| a3 | bf16, ckpt | 2.4 h | 14.1 h | 12.5 h | **26.6 h** |
| a4 | bf16, no ckpt, rank tuned as an HP | 1.5 h | 8.9 h | 8.5 h | **17.4 h** |
| a4 | every rank × every fraction × 3 seeds | | 8.9 h | 33.9 h | 42.8 h |
| a5 | API, network-bound | | | | < 1 h |
| **Total** | rank tuned as HP | | | | **≈ 45 h** |
| | plus the rank curve at fraction 1.0 only | | | | ≈ 59 h |
| | full rank × fraction grid | | | | ≈ 71 h |
| | for reference, fp32 everywhere | | | | ≈ 80 h |

Risks to the estimate: early stopping will shorten some runs; the a4 no-checkpointing config still has to be verified on real variable-length batches (if it has to fall back to checkpointing, a4 takes about 1.5× longer); and thermal throttling over multi-hour runs was not measured. The longest single run was about 15 min.

## Changes made to REQUIREMENTS.md
- §8.1: versions pinned; bf16 autocast confirmed.
- §8.2: memory budget restated in driver and tensor terms; token-budget batching and pad-to-64 added; a3 marked go.
- §8.3: AC power is now required (enforced in `make spike` / `make sweep`), and the measured estimate is recorded.
- §11: questions 1–4 resolved, with pointers here.

## Decisions for the project owner
1. **LoRA rank sweep scope.** Either tune r ∈ {4, 8, 16, 32} inside a4's HP budget (≈ 45 h total), or report the full rank curve, only at fraction 1.0 (≈ 59 h) or at every fraction (≈ 71 h).
2. **Jev access.** Early access is waitlist-only, so a5 is blocked until the account has a key.
