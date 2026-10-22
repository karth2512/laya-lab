"""Plain PyTorch training loop shared by a2/a3/a4 (no HF Trainer, so no framework logging).

Batches are formed by a token budget over length-sorted examples, padded to a multiple of 64
(REQUIREMENTS §8.2: bounds the number of distinct shapes the MPS allocator caches). Optimizer
steps accumulate micro-batches up to the effective batch size; epochs are raised so every run gets
at least `min_steps` optimizer steps (§6 step floor). Early stopping on val macro-F1 restores the
best trainable weights. Everything is logged through `ctx`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

from shared.device import autocast
from shared.interface import RunContext, Split

PAD_MULTIPLE = 64


@dataclass
class TrainConfig:
    lr: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 3
    min_steps: int = 200
    effective_batch_size: int = 16
    max_tokens_per_batch: int = 2048
    eval_max_tokens_per_batch: int = 8192
    warmup_ratio: float = 0.06
    optimizer: str = "adamw"  # or "adafactor" (memory lever 5)
    max_grad_norm: float = 1.0
    patience: int = 2
    gradient_checkpointing: bool = False
    bf16: bool = True  # bf16 autocast on MPS only; weights and optimizer state stay fp32
    max_length: int = 512

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any]) -> TrainConfig:
        """From the resolved config: `cfg["arm"]["train"]`, max_length from `cfg["dataset"]`."""
        section = dict(cfg["arm"].get("train", {}))
        section.setdefault("max_length", cfg.get("dataset", {}).get("max_length", 512))
        out = cls()
        for f in fields(cls):
            if f.name in section:
                setattr(out, f.name, type(getattr(out, f.name))(section[f.name]))
        return out


@dataclass
class Batch:
    idx: np.ndarray  # positions in the source split
    inputs: dict[str, torch.Tensor]
    labels: torch.Tensor


@dataclass
class Encoded:
    input_ids: list[list[int]]
    labels: np.ndarray
    truncated_pct: float


def encode(tokenizer: Any, split: Split, max_length: int) -> Encoded:
    """Tokenize once; truncate to `max_length` keeping the final [SEP]."""
    ids = tokenizer(split.texts, add_special_tokens=True)["input_ids"]
    sep = tokenizer.sep_token_id
    truncated = [len(x) > max_length for x in ids]
    ids = [x[: max_length - 1] + [sep] if t else x for x, t in zip(ids, truncated, strict=True)]
    labels = np.asarray(split.labels) if len(split.labels) == len(ids) else np.zeros(len(ids), int)
    return Encoded(ids, labels, 100.0 * float(np.mean(truncated)))


def token_batches(
    enc: Encoded, max_tokens: int, pad_id: int, rng: np.random.Generator | None = None
) -> list[Batch]:
    """Length-sorted batches whose padded size (multiple of 64) stays within `max_tokens`."""
    order = np.argsort([len(x) for x in enc.input_ids], kind="stable")
    groups: list[list[int]] = []
    current: list[int] = []
    for i in order:
        padded = _pad_len(max(len(enc.input_ids[j]) for j in [*current, i]))
        if current and padded * (len(current) + 1) > max_tokens:
            groups.append(current)
            current = []
        current.append(int(i))
    if current:
        groups.append(current)
    if rng is not None:
        rng.shuffle(groups)  # type: ignore[arg-type]
    return [_collate(enc, g, pad_id) for g in groups]


def _pad_len(n: int) -> int:
    return math.ceil(n / PAD_MULTIPLE) * PAD_MULTIPLE


def _collate(enc: Encoded, group: list[int], pad_id: int) -> Batch:
    width = _pad_len(max(len(enc.input_ids[i]) for i in group))
    ids = torch.full((len(group), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(group), width), dtype=torch.long)
    for row, i in enumerate(group):
        seq = enc.input_ids[i]
        ids[row, : len(seq)] = torch.tensor(seq)
        mask[row, : len(seq)] = 1
    labels = torch.as_tensor(enc.labels[group], dtype=torch.long)
    return Batch(np.asarray(group), {"input_ids": ids, "attention_mask": mask}, labels)


Forward = Callable[[dict[str, torch.Tensor]], torch.Tensor]  # inputs on device -> logits


def predict_logits(
    model: torch.nn.Module, forward: Forward, batches: list[Batch], device: str, bf16: bool
) -> np.ndarray:
    """fp32 logits in source order."""
    model.eval()
    n = sum(len(b.idx) for b in batches)
    out = np.zeros((n, 0), dtype=np.float32)
    with torch.no_grad(), autocast(device, bf16):
        for b in batches:
            logits = forward({k: v.to(device) for k, v in b.inputs.items()}).float().cpu().numpy()
            if out.shape[1] == 0:
                out = np.zeros((n, logits.shape[1]), dtype=np.float32)
            out[b.idx] = logits
    return out


def _steps_per_epoch(batches: list[Batch], effective_batch_size: int) -> int:
    steps, acc = 0, 0
    for b in batches:
        acc += len(b.idx)
        if acc >= effective_batch_size:
            steps, acc = steps + 1, 0
    return steps + (acc > 0)


def _optimizer(params: list[torch.nn.Parameter], tc: TrainConfig) -> torch.optim.Optimizer:
    if tc.optimizer == "adafactor":
        from transformers.optimization import Adafactor

        return Adafactor(
            params, lr=tc.lr, weight_decay=tc.weight_decay, scale_parameter=False,
            relative_step=False,
        )
    return torch.optim.AdamW(params, lr=tc.lr, weight_decay=tc.weight_decay)


def fit(
    model: torch.nn.Module,
    forward: Forward,
    train_batches: Callable[[int], list[Batch]],
    val_batches: list[Batch],
    val_labels: np.ndarray,
    tc: TrainConfig,
    ctx: RunContext,
) -> dict[str, Any]:
    """Train the parameters of `model` that require grad. `train_batches(epoch)` reshuffles."""
    device = ctx.device
    params = [p for p in model.parameters() if p.requires_grad]
    first = train_batches(0)
    per_epoch = _steps_per_epoch(first, tc.effective_batch_size)
    epochs = max(tc.epochs, math.ceil(tc.min_steps / max(per_epoch, 1)))
    total_steps = epochs * per_epoch
    opt = _optimizer(params, tc)
    warmup = int(tc.warmup_ratio * total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (s + 1) / max(warmup, 1)
        if s < warmup
        else max(0.0, (total_steps - s) / max(total_steps - warmup, 1)),
    )

    best_f1, best_epoch, bad, step = -1.0, -1, 0, 0
    best_state: dict[str, torch.Tensor] = {}
    for epoch in range(epochs):
        model.train()
        batches = first if epoch == 0 else train_batches(epoch)
        loss_sum, seen, acc = 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        for i, b in enumerate(batches):
            inputs = {k: v.to(device) for k, v in b.inputs.items()}
            with autocast(device, tc.bf16):
                logits = forward(inputs)
            loss = F.cross_entropy(logits.float(), b.labels.to(device), reduction="sum")
            (loss / tc.effective_batch_size).backward()
            loss_sum += float(loss.detach())
            seen += len(b.idx)
            acc += len(b.idx)
            if acc >= tc.effective_batch_size or i == len(batches) - 1:
                torch.nn.utils.clip_grad_norm_(params, tc.max_grad_norm)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                acc, step = 0, step + 1
        if not math.isfinite(loss_sum):
            raise FloatingPointError(f"non-finite training loss at epoch {epoch}")

        val_logits = predict_logits(model, forward, val_batches, device, tc.bf16)
        val_loss = float(F.cross_entropy(torch.from_numpy(val_logits), torch.from_numpy(val_labels)))
        val_f1 = float(f1_score(val_labels, val_logits.argmax(1), average="macro"))
        ctx.log_metric("train/loss", loss_sum / max(seen, 1), step=epoch)
        ctx.log_metric("train/lr", sched.get_last_lr()[0], step=epoch)
        ctx.log_metric("val/loss", val_loss, step=epoch)
        ctx.log_metric("val/macro_f1", val_f1, step=epoch)

        if val_f1 > best_f1:
            best_f1, best_epoch, bad = val_f1, epoch, 0
            best_state = {
                n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad
            }
        else:
            bad += 1
            if bad >= tc.patience:
                break

    model.load_state_dict(best_state, strict=False)
    return {
        "best_val_macro_f1": best_f1,
        "best_epoch": best_epoch,
        "epochs_planned": epochs,
        "epochs_run": epoch + 1,
        "optimizer_steps": step,
    }


def classifier_forward(model: torch.nn.Module) -> Forward:
    return lambda inputs: model(**inputs).logits


def fit_classifier(
    model: torch.nn.Module, tokenizer: Any, train: Split, val: Split, tc: TrainConfig,
    ctx: RunContext,
) -> dict[str, Any]:
    """Token-level training of a sequence classifier (a3, a4)."""
    pad = tokenizer.pad_token_id
    enc_train, enc_val = encode(tokenizer, train, tc.max_length), encode(tokenizer, val, tc.max_length)
    ctx.log_metric("data/truncated_pct", enc_train.truncated_pct)
    ctx.log_metric("data/truncated_pct_val", enc_val.truncated_pct)
    if tc.gradient_checkpointing:
        model.gradient_checkpointing_enable(  # type: ignore[operator]
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    def train_batches(epoch: int) -> list[Batch]:
        rng = np.random.default_rng([ctx.seed, epoch])
        return token_batches(enc_train, tc.max_tokens_per_batch, pad, rng)

    val_batches = token_batches(enc_val, tc.eval_max_tokens_per_batch, pad)
    return fit(model, classifier_forward(model), train_batches, val_batches, enc_val.labels, tc, ctx)


def predict_classifier(
    model: torch.nn.Module, tokenizer: Any, split: Split, tc: TrainConfig, device: str
) -> np.ndarray:
    enc = encode(tokenizer, split, tc.max_length)
    batches = token_batches(enc, tc.eval_max_tokens_per_batch, tokenizer.pad_token_id)
    return predict_logits(model, classifier_forward(model), batches, device, tc.bf16)

