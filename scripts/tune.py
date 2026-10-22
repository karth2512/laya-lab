"""HP tuning on val at fraction 1.0 with an equal trial budget per arm (REQUIREMENTS §6).

Draws `hp_trials` points from the arm's `hp_space` (full grid if it is small enough, else a seeded
random sample), runs each as a dev run with variant `hp<i>`, and writes the best by val macro-F1 to
`configs/tuned/<arm>.yaml`, which `load_config` applies to every later run. Resumable via MLflow.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from shared.config import ROOT, load_config
from shared.mlflow_utils import load_runs
from shared.runner import run, run_name


def draw_trials(space: dict[str, list[Any]], n: int, seed: int = 0) -> list[dict[str, Any]]:
    grid = [dict(zip(space, values, strict=True)) for values in itertools.product(*space.values())]
    if len(grid) <= n:
        return grid
    idx = np.random.default_rng(seed).choice(len(grid), size=n, replace=False)
    return [grid[i] for i in sorted(idx)]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = p.parse_args(argv)

    cfg = load_config(args.arm, args.set, use_tuned=False)
    space = cfg["arm"].get("hp_space")
    if not space:
        raise SystemExit(f"{args.arm} has no hp_space (it tunes or selects inside fit)")
    trials = draw_trials(space, int(cfg["arm"].get("hp_trials", 6)))

    runs = load_runs(cfg["experiment"])
    done: dict[str, float] = {}
    if not runs.empty and "metrics.val/macro_f1" in runs:
        names, scores = runs["tags.mlflow.runName"], runs["metrics.val/macro_f1"]
        done = dict(zip(names, scores, strict=True))
    results: list[dict[str, Any]] = []
    for i, trial in enumerate(trials):
        overrides = [f"{k}={json.dumps(v)}" for k, v in trial.items()]
        name = run_name(args.arm, "dev", args.seed, 1.0, f"hp{i}")
        if name in done:
            score = float(done[name])
        else:
            metrics = run(args.arm, "dev", args.seed, 1.0, [*args.set, *overrides], f"hp{i}")
            score = float(metrics["val/macro_f1"])
        print(f"{name}: val/macro_f1={score:.4f} {trial}", flush=True)
        results.append({"variant": f"hp{i}", "val_macro_f1": score, "params": trial})

    best = max(results, key=lambda r: r["val_macro_f1"])
    out = Path(ROOT / "configs" / "tuned" / f"{args.arm}.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    best_overrides = [f"{k}={json.dumps(v)}" for k, v in best["params"].items()]
    doc = {"overrides": best_overrides, "best": best["variant"], "trials": results}
    out.write_text(f"# Written by scripts.tune; applied by load_config.\n{yaml.safe_dump(doc)}")
    print(f"best {best['variant']} ({best['val_macro_f1']:.4f}) -> {out}")


if __name__ == "__main__":
    main()
