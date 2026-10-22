"""Pull runs from MLflow and build the results report (REQUIREMENTS §6, §10).

Writes `reports/results.md` and `reports/figures/*.png`: results table (mean ± std across seeds,
bootstrap CI on pooled test predictions), learning curves with zero-shot reference lines,
McNemar matrix, reliability diagrams, cost table and the label-wording spread for a1/a5.

The reporting functions take a DataFrame of runs (MLflow `search_runs` layout: `run_id`,
`metrics.*`, `params.*`, `tags.*`) and a `fetch_predictions(run_id) -> DataFrame | None` callable,
so they run without MLflow in tests.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from shared.stats import bootstrap_ci, mcnemar  # noqa: E402

FetchPredictions = Callable[[str], pd.DataFrame | None]

ZERO_SHOT_ARMS = ("a1_laya_zeroshot", "a5_jev_api")
END_TO_END_LATENCY_ARMS = ("a5_jev_api",)
PRIMARY = "macro_f1"


# ---------------------------------------------------------------------------- loading


def normalize_runs(raw: pd.DataFrame) -> pd.DataFrame:
    """Add `arm, stage, seed, fraction, variant, method` columns; drop parent runs."""

    def col(name: str) -> pd.Series:
        return raw[name] if name in raw.columns else pd.Series([None] * len(raw), index=raw.index)

    df = raw.copy()
    df["arm"] = col("tags.arm").fillna(col("params.arm"))
    df["stage"] = col("tags.stage").fillna(col("params.stage"))
    df["seed"] = pd.to_numeric(col("params.seed"), errors="coerce")
    df["fraction"] = pd.to_numeric(col("params.fraction"), errors="coerce")
    df["variant"] = col("params.variant").fillna("").astype(str)
    df = df[df["arm"].notna() & df["seed"].notna() & df["fraction"].notna()].copy()
    df["seed"] = df["seed"].astype(int)
    df["method"] = np.where(df["variant"] == "", df["arm"], df["arm"] + "/" + df["variant"])
    return df.reset_index(drop=True)


def read_predictions(df: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (ids, y_true, probs) from a predictions CSV frame.

    Expected columns: `id`, `true_label` (class id or name), and one `prob_<class>` column per
    class in class-id order.
    """
    prob_cols = [c for c in df.columns if c.startswith("prob_")]
    classes = [c.removeprefix("prob_") for c in prob_cols]
    y = df["true_label"]
    if y.dtype == object:
        y = y.map({name: i for i, name in enumerate(classes)})
    return df["id"].astype(str).tolist(), y.to_numpy(dtype=int), df[prob_cols].to_numpy(float)


def class_names(df: pd.DataFrame) -> list[str]:
    return [c.removeprefix("prob_") for c in df.columns if c.startswith("prob_")]


# ---------------------------------------------------------------------------- tables


def _pm(values: pd.Series) -> str:
    values = values.dropna()
    if values.empty:
        return "–"
    std = values.std(ddof=1) if len(values) > 1 else 0.0
    return f"{values.mean():.4f} ± {std:.4f}"


def pooled_predictions(
    runs: pd.DataFrame, fetch: FetchPredictions
) -> tuple[np.ndarray, np.ndarray] | None:
    """Concatenate test predictions over the given runs (one per seed)."""
    ys, ps = [], []
    for run_id in runs["run_id"]:
        pred = fetch(run_id)
        if pred is not None:
            _, y, p = read_predictions(pred)
            ys.append(y)
            ps.append(p)
    if not ys:
        return None
    return np.concatenate(ys), np.concatenate(ps)


def results_table(final: pd.DataFrame, fetch: FetchPredictions, n_boot: int = 1000) -> pd.DataFrame:
    rows = []
    for (method, fraction), g in final.groupby(["method", "fraction"], sort=True):
        row = {
            "method": method,
            "fraction": fraction,
            "seeds": len(g),
            "test macro_f1": _pm(g.get("metrics.test/macro_f1", pd.Series(dtype=float))),
            "test accuracy": _pm(g.get("metrics.test/accuracy", pd.Series(dtype=float))),
        }
        pooled = pooled_predictions(g, fetch)
        for metric in ("macro_f1", "accuracy"):
            if pooled is None:
                row[f"{metric} 95% CI"] = "–"
            else:
                low, high = bootstrap_ci(pooled[0], pooled[1], metric=metric, n=n_boot)
                row[f"{metric} 95% CI"] = f"[{low:.4f}, {high:.4f}]"
        rows.append(row)
    return pd.DataFrame(rows)


def cost_table(final: pd.DataFrame) -> pd.DataFrame:
    """Mean cost metrics per method at fraction 1.0."""
    cols = [
        c
        for c in final.columns
        if c.startswith(("metrics.cost/", "metrics.latency/", "metrics.throughput/"))
    ]
    full = final[final["fraction"] == 1.0]
    if full.empty or not cols:
        return pd.DataFrame()
    table = full.groupby("method")[cols].mean().reset_index()
    table.columns = [c.removeprefix("metrics.") for c in table.columns]
    table["latency kind"] = [
        "end-to-end (network)" if m.split("/")[0] in END_TO_END_LATENCY_ARMS else "local"
        for m in table["method"]
    ]
    return table


def _seed_runs(runs: pd.DataFrame, method: str) -> dict[int, str]:
    sub = runs[runs["method"] == method]
    return dict(zip(sub["seed"], sub["run_id"], strict=True))


def _correct(fetch: FetchPredictions, run_id: str) -> pd.Series | None:
    pred = fetch(run_id)
    if pred is None:
        return None
    ids, y, p = read_predictions(pred)
    return pd.Series(p.argmax(1) == y, index=ids)


def mcnemar_matrix(final: pd.DataFrame, fetch: FetchPredictions) -> pd.DataFrame:
    """Pairwise McNemar p-values at fraction 1.0, seed-matched.

    Arms with a single run (zero-shot, majority) are paired with every seed of the other arm.
    """
    full = final[final["fraction"] == 1.0]
    methods = sorted(full["method"].unique())
    out = pd.DataFrame(index=methods, columns=methods, dtype=object)
    cache: dict[str, pd.Series | None] = {}

    def correct(run_id: str) -> pd.Series | None:
        if run_id not in cache:
            cache[run_id] = _correct(fetch, run_id)
        return cache[run_id]

    for a in methods:
        for b in methods:
            if a == b:
                out.loc[a, b] = "–"
                continue
            ra, rb = _seed_runs(full, a), _seed_runs(full, b)
            if len(ra) == 1 or len(rb) == 1:
                seeds = sorted(set(ra) | set(rb))
            else:
                seeds = sorted(set(ra) & set(rb))
            xa, xb = [], []
            for s in seeds:
                ca = correct(ra.get(s, next(iter(ra.values()))))
                cb = correct(rb.get(s, next(iter(rb.values()))))
                if ca is None or cb is None:
                    continue
                common = ca.index.intersection(cb.index)
                xa.append(ca[common].to_numpy())
                xb.append(cb[common].to_numpy())
            if not xa:
                out.loc[a, b] = "–"
                continue
            _, p = mcnemar(np.concatenate(xa), np.concatenate(xb))
            out.loc[a, b] = f"{p:.3g}"
    return out


def wording_spread(runs: pd.DataFrame) -> pd.DataFrame:
    """Val macro-F1 per label-wording variant for a1/a5.

    Reads either per-variant dev runs (`variant` param set) or metrics named
    `wording/<variant>/macro_f1` logged on a single run.
    """
    rows = []
    zs = runs[runs["arm"].isin(ZERO_SHOT_ARMS)]
    for arm, g in zs.groupby("arm"):
        scores: dict[str, float] = {}
        for c in g.columns:
            if c.startswith("metrics.wording/") and c.endswith("/macro_f1"):
                variant = c.removeprefix("metrics.wording/").removesuffix("/macro_f1")
                val = g[c].dropna()
                if not val.empty:
                    scores[variant] = float(val.iloc[-1])
        if not scores and "metrics.val/macro_f1" in g.columns:
            dev = g[(g["stage"] == "dev") & (g["variant"] != "")]
            for variant, v in dev.groupby("variant")["metrics.val/macro_f1"]:
                scores[str(variant)] = float(v.mean())
        if scores:
            vals = np.array(list(scores.values()))
            rows.append(
                {
                    "arm": arm,
                    "variants": len(scores),
                    "best": max(scores, key=lambda k: scores[k]),
                    "best val macro_f1": f"{vals.max():.4f}",
                    "spread (max-min)": f"{vals.max() - vals.min():.4f}",
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------- figures


def plot_learning_curves(final: pd.DataFrame, path: Path) -> None:
    metric = f"metrics.test/{PRIMARY}"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    if metric in final.columns:
        for method, g in final.groupby("method"):
            arm = str(method).split("/")[0]
            stats = g.groupby("fraction")[metric].agg(["mean", "std"]).fillna(0.0)
            if arm in ZERO_SHOT_ARMS or len(stats) == 1:
                ax.axhline(stats["mean"].iloc[0], linestyle="--", linewidth=1, label=method)
            else:
                ax.errorbar(
                    stats.index,
                    stats["mean"],
                    yerr=stats["std"],
                    marker="o",
                    capsize=3,
                    label=method,
                )
    ax.set_xscale("log")
    ax.set_xlabel("training fraction")
    ax.set_ylabel(f"test {PRIMARY}")
    ax.set_title("Learning curves (mean ± std over seeds)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def reliability(y: np.ndarray, probs: np.ndarray, bins: int = 15) -> tuple[np.ndarray, ...]:
    """Per-bin (mean confidence, accuracy, count) over max-probability confidence."""
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, bins - 1)
    counts = np.bincount(idx, minlength=bins)
    safe = np.maximum(counts, 1)
    mean_conf = np.bincount(idx, weights=conf, minlength=bins) / safe
    acc = np.bincount(idx, weights=correct, minlength=bins) / safe
    return mean_conf, acc, counts


def plot_calibration(final: pd.DataFrame, fetch: FetchPredictions, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], color="grey", linewidth=1, linestyle=":")
    full = final[final["fraction"] == 1.0]
    for method, g in full.groupby("method"):
        pooled = pooled_predictions(g, fetch)
        if pooled is None:
            continue
        conf, acc, counts = reliability(*pooled)
        keep = counts > 0
        ax.plot(conf[keep], acc[keep], marker="o", markersize=3, label=method)
    ax.set_xlabel("confidence")
    ax.set_ylabel("accuracy")
    ax.set_title("Reliability (test, fraction 1.0, pooled seeds)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------- report


def markdown_table(df: pd.DataFrame, index: bool = False) -> str:
    if df.empty:
        return "_No data._\n"
    frame = df.reset_index() if index else df
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    sep = "|" + "---|" * len(frame.columns)
    body = ["| " + " | ".join(_fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, sep, *body]) + "\n"


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def build_report(
    raw_runs: pd.DataFrame, fetch: FetchPredictions, out: Path, n_boot: int = 1000
) -> Path:
    runs = normalize_runs(raw_runs)
    final = runs[runs["stage"] == "final"]
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    plot_learning_curves(final, figures / "learning_curves.png")
    plot_calibration(final, fetch, figures / "calibration.png")
    mcn = mcnemar_matrix(final, fetch)
    mcn.index.name = "method"

    text = "\n".join(
        [
            "# Results: Laya on the Medical Abstracts TC Corpus",
            "",
            "Generated by `scripts/aggregate.py`. Primary metric: test macro-F1. Mean ± std across",
            "seeds; 95% CI from a bootstrap over pooled test predictions.",
            "Dataset: Medical Abstracts TC Corpus (Schopf et al.), CC BY-SA 3.0.",
            "",
            "## Results table (stage=final)",
            "",
            markdown_table(results_table(final, fetch, n_boot)),
            "## Learning curves",
            "",
            "Zero-shot arms (a1, a5) are dashed horizontal reference lines.",
            "",
            "![learning curves](figures/learning_curves.png)",
            "",
            "## McNemar p-values (test, fraction 1.0, seed-matched)",
            "",
            markdown_table(mcn, index=True),
            "## Calibration",
            "",
            "![calibration](figures/calibration.png)",
            "",
            "## Cost (fraction 1.0, mean over seeds)",
            "",
            "Wall-clock, latency and throughput are only comparable between runs on the same",
            "machine (M5, 16 GB) under the same conditions. a5 latency is end-to-end",
            "(network and queueing included) and not comparable to local arms.",
            "",
            markdown_table(cost_table(final)),
            "## Label-wording spread (val)",
            "",
            markdown_table(wording_spread(runs)),
        ]
    )
    path = out / "results.md"
    path.write_text(text)
    return path


def mlflow_fetcher(cache_dir: Path) -> FetchPredictions:
    from shared.mlflow_utils import download_artifact

    def fetch(run_id: str) -> pd.DataFrame | None:
        try:
            path = download_artifact(run_id, "predictions_test.csv", cache_dir / run_id)
        except Exception:
            return None
        return pd.read_csv(path)

    return fetch


def main(argv: list[str] | None = None) -> None:
    from shared.mlflow_utils import load_runs

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="laya-medabs-bench")
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)

    runs = load_runs(args.experiment)
    fetch = mlflow_fetcher(Path("data/cache/artifacts"))
    path = build_report(runs, fetch, args.out, args.bootstrap)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
