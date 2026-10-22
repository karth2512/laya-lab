"""Orchestrates a single run (REQUIREMENTS §4.4): one arm, one stage, one seed, one fraction."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import os
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import classification_report, confusion_matrix

from shared import data, splits
from shared.calibration import ece, fit_temperature, softmax
from shared.config import flatten, load_config
from shared.cost import MemorySampler, Timer, latency_stats
from shared.device import select_device
from shared.interface import Arm, Predictions, RunContext, Split, Stage
from shared.metrics import compute_metrics
from shared.mlflow_utils import ChildRun, Tracker, env_tags
from shared.seed import seed_everything
from shared.stats import bootstrap_ci

MPS_FALLBACK_TEXT = "fall back to run on the CPU"


def run_name(arm: str, stage: str, seed: int, fraction: float, variant: str | None = None) -> str:
    name = f"{arm}/{stage}/s{seed}/f{fraction:g}"
    return f"{name}/{variant}" if variant else name


def tracker_from_cfg(cfg: dict[str, Any]) -> Tracker:
    uri = os.environ.get("MLFLOW_TRACKING_URI", cfg["mlflow"]["tracking_uri"])
    return Tracker(uri, cfg["experiment"], cfg["mlflow"]["artifact_location"])


def _cap(split: Split, n: int) -> Split:
    """Evenly spaced subset of at most n examples (keeps a class mix in id-ordered splits)."""
    return split.subset(np.linspace(0, len(split) - 1, min(n, len(split))).astype(int))


def load_splits(
    cfg: dict[str, Any], stage: Stage, seed: int, fraction: float
) -> tuple[Split, Split, Split | None, str]:
    """(train subset, val, test or None, manifest hash). At stage=dev the test file is not read."""
    ds = cfg["dataset"]
    manifest = splits.load_manifest(ds["split_manifest"])
    train_df, test_df = data.load_frames(ds["raw_dir"], include_test=stage == "final")
    train_df = train_df.set_index("id", drop=False)
    pool = data.to_split(train_df.loc[manifest["train_pool"]])
    train_ids = splits.fraction_ids(pool.ids, pool.labels, seed, fraction)
    train = data.to_split(train_df.loc[train_ids])
    val = data.to_split(train_df.loc[manifest["val"]])
    test = None
    if test_df is not None:
        test = data.to_split(test_df.set_index("id", drop=False).loc[manifest["test"]])
    cap = cfg["eval"].get("max_examples")  # smoke runs only; never set for real results
    if cap:
        train, val = _cap(train, cap), _cap(val, cap)
        test = _cap(test, cap) if test is not None else None
    return train, val, test, str(manifest["hash"])


def _context(
    cfg: dict[str, Any],
    arm: str,
    stage: Stage,
    seed: int,
    fraction: float,
    device: str,
    scratch: Path,
    child: ChildRun,
) -> RunContext:
    return RunContext(
        cfg=cfg,
        arm=arm,
        stage=stage,
        seed=seed,
        fraction=fraction,
        device=device,
        scratch_dir=scratch,
        log_metric=child.log_metric,
        log_param=child.log_param,
        log_artifact=child.log_artifact,
        log_dict=child.log_dict,
        set_tag=child.set_tag,
    )


def _measure_latency(arm: Arm, split: Split, ctx: RunContext, n: int) -> np.ndarray:
    """Per-example latency at batch size 1 over the first n examples."""
    out = []
    for i in range(min(n, len(split))):
        one = split.subset([i])
        start = time.perf_counter()
        arm.predict(one, ctx)
        out.append((time.perf_counter() - start) * 1000)
    return np.asarray(out)


def _evaluate(
    prefix: str, split: Split, preds: Predictions, temperature: float | None, cfg: dict[str, Any]
) -> dict[str, float]:
    n_bins = cfg["eval"]["ece_bins"]
    metrics = compute_metrics(split.labels, preds.probs, n_bins=n_bins)
    if temperature is not None and preds.logits is not None:
        metrics["temperature"] = temperature
        metrics["ece_ts"] = ece(split.labels, softmax(preds.logits, temperature), n_bins)
    return {f"{prefix}/{k}": v for k, v in metrics.items()}


def _bootstrap(
    split: Split, preds: Predictions, cfg: dict[str, Any], seed: int
) -> dict[str, float]:
    out = {}
    for metric in ("macro_f1", "accuracy"):
        low, high = bootstrap_ci(
            split.labels, preds.probs, metric, n=cfg["eval"]["bootstrap_resamples"], seed=seed
        )
        out[f"test/bootstrap_ci_low/{metric}"] = low
        out[f"test/bootstrap_ci_high/{metric}"] = high
    return out


def _write_artifacts(
    out: Path, split_name: str, split: Split, preds: Predictions, child: ChildRun
) -> None:
    """predictions_<split>.csv: id, true label, predicted label, per-class probabilities."""
    df = pd.DataFrame({"id": split.ids, "true_label": split.labels})
    df["pred_label"] = preds.probs.argmax(1)
    for i, name in enumerate(data.CLASSES):
        df[f"prob_{name}"] = preds.probs[:, i]
    df.to_csv(out / f"predictions_{split_name}.csv", index=False)
    child.log_artifact(out / f"predictions_{split_name}.csv")


def _write_report_artifacts(out: Path, split: Split, preds: Predictions, child: ChildRun) -> None:
    """classification_report.json, confusion_matrix.png and calibration_curve.png."""
    labels = list(range(len(data.CLASSES)))
    y_pred = preds.probs.argmax(1)
    report = classification_report(
        split.labels,
        y_pred,
        labels=labels,
        target_names=list(data.CLASSES),
        output_dict=True,
        zero_division=0,
    )
    (out / "classification_report.json").write_text(json.dumps(report, indent=1))

    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(split.labels, y_pred, labels=labels)
    ax.imshow(cm, cmap="Blues")
    for (r, c), v in np.ndenumerate(cm):
        ax.text(c, r, str(v), ha="center", va="center", fontsize=8)
    ax.set(xlabel="predicted", ylabel="true", xticks=labels, yticks=labels)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=100)
    plt.close(fig)

    conf = preds.probs.max(1)
    correct = y_pred == split.labels
    edges = np.linspace(0, 1, 16)
    bins = np.clip(np.digitize(conf, edges) - 1, 0, 14)
    xs = [conf[bins == b].mean() for b in range(15) if (bins == b).any()]
    ys = [correct[bins == b].mean() for b in range(15) if (bins == b).any()]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.plot(xs, ys, "o-")
    ax.set(xlabel="confidence", ylabel="accuracy", xlim=(0, 1), ylim=(0, 1))
    fig.tight_layout()
    fig.savefig(out / "calibration_curve.png", dpi=100)
    plt.close(fig)

    for name in ("classification_report.json", "confusion_matrix.png", "calibration_curve.png"):
        child.log_artifact(out / name)


def _pip_freeze() -> str:
    dists = {(d.metadata["Name"] or "?", d.version) for d in importlib.metadata.distributions()}
    return "\n".join(f"{n}=={v}" for n, v in sorted(dists)) + "\n"


def run(
    arm: str,
    stage: Stage,
    seed: int,
    fraction: float,
    overrides: list[str] | None = None,
    variant: str | None = None,
    save_model: bool = False,
) -> dict[str, float]:
    """Run one (arm, stage, seed, fraction[, variant]) and log it to MLflow. Returns metrics."""
    # 1. config + seeding
    cfg = load_config(arm, overrides)
    cfg.update(stage=stage, seed=seed, fraction=fraction, variant=variant)
    cfg["eval"]["save_model"] = save_model or cfg["eval"]["save_model"]
    deterministic = seed_everything(seed, bool(cfg.get("deterministic", False)))
    device = select_device(cfg["device"])

    # 2. data (test only at stage=final)
    train, val, test, split_hash = load_splits(cfg, stage, seed, fraction)

    # 3. MLflow child run
    tracker = tracker_from_cfg(cfg)
    with (
        tracker.child(arm, run_name(arm, stage, seed, fraction, variant)) as child,
        tempfile.TemporaryDirectory() as tmp,
        MemorySampler() as memory,
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        out = Path(tmp)
        child.set_tags(
            {
                "arm": arm,
                "stage": stage,
                "data_hash": split_hash,
                **env_tags(device),
                "deterministic": str(deterministic).lower(),
            }
        )
        child.log_params(
            {**flatten(cfg), "arm_name": arm, "split_hash": split_hash, "device_used": device}
        )
        ctx = _context(cfg, arm, stage, seed, fraction, device, out / "scratch", child)
        ctx.scratch_dir.mkdir()

        # 4. setup + fit (timed, memory-tracked)
        model: Arm = importlib.import_module(f"arms.{arm}.arm").build(cfg)
        model.setup(ctx)
        with Timer() as fit_timer:
            report = model.fit(train, val, ctx)
        metrics: dict[str, float] = {
            "cost/train_seconds": fit_timer.seconds,
            "data/train_size": float(len(train)),
        }
        if report.trainable_params is not None:
            metrics["cost/trainable_params"] = float(report.trainable_params)
        if report.total_params is not None:
            metrics["cost/total_params"] = float(report.total_params)
        if report.extra:
            child.log_dict(report.extra, "fit_report.json")

        # 5. val metrics + temperature scaling
        with Timer() as val_timer:
            val_preds = model.predict(val, ctx)
        val_preds.validate(val.ids)
        metrics["throughput/examples_per_sec"] = len(val) / max(val_timer.seconds, 1e-9)
        temperature = None
        if val_preds.logits is not None:
            temperature = fit_temperature(val_preds.logits, val.labels)
        metrics.update(_evaluate("val", val, val_preds, temperature, cfg))
        _write_artifacts(out, "val", val, val_preds, child)

        latency = val_preds.latency_ms
        if latency is None:
            latency = _measure_latency(model, val, ctx, cfg["eval"]["latency_examples"])
        metrics.update(latency_stats(latency))

        # 6. test, only at stage=final
        report_split, report_preds = val, val_preds
        if stage == "final":
            assert test is not None
            test_preds = model.predict(test, ctx)
            test_preds.validate(test.ids)
            metrics.update(_evaluate("test", test, test_preds, temperature, cfg))
            metrics.update(_bootstrap(test, test_preds, cfg, seed))
            _write_artifacts(out, "test", test, test_preds, child)
            report_split, report_preds = test, test_preds

        # 7. remaining metrics + artifacts, teardown
        model.teardown(ctx)
        _write_report_artifacts(out, report_split, report_preds, child)
        (out / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        (out / "pip_freeze.txt").write_text(_pip_freeze())
        (out / "split_manifest_hash.txt").write_text(split_hash + "\n")
        for name in ("resolved_config.yaml", "pip_freeze.txt", "split_manifest_hash.txt"):
            child.log_artifact(out / name)

        if "data/truncated_pct" not in child.logged:
            metrics["data/truncated_pct"] = 0.0  # arms that tokenize log their own value
        metrics.update(memory.metrics())
        child.log_metrics(metrics)
        fallback = any(MPS_FALLBACK_TEXT in str(w.message) for w in caught)
        child.set_tag("mps_fallback_used", str(fallback).lower())
    return metrics
