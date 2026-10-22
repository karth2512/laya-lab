"""Error analysis on final test predictions (REQUIREMENTS §6).

Per method at fraction 1.0 (seed 0 by default): top confusion pairs, with pairs involving
"General pathological conditions" flagged; sampled misclassified abstracts per class; and the
confidence distribution of errors vs correct predictions. Writes `reports/error_analysis.md`
and `reports/figures/error_confidence_<method>.png`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.aggregate import (  # noqa: E402
    FetchPredictions,
    class_names,
    markdown_table,
    mlflow_fetcher,
    normalize_runs,
    read_predictions,
)

CATCH_ALL = "General pathological conditions"


def confusion_pairs(
    y: np.ndarray, pred: np.ndarray, classes: list[str], top: int = 10
) -> pd.DataFrame:
    """Most frequent (true, predicted) error pairs."""
    wrong = y != pred
    pairs = pd.DataFrame({"true": y[wrong], "pred": pred[wrong]}).value_counts().head(top)
    rows = [
        {
            "true": classes[t],
            "predicted": classes[p],
            "count": int(n),
            "share of errors": f"{n / max(wrong.sum(), 1):.1%}",
            "involves catch-all": "yes" if CATCH_ALL in (classes[t], classes[p]) else "",
        }
        for (t, p), n in pairs.items()
    ]
    return pd.DataFrame(rows)


def sample_errors(
    ids: list[str],
    y: np.ndarray,
    probs: np.ndarray,
    classes: list[str],
    texts: dict[str, str],
    per_class: int = 3,
    seed: int = 0,
) -> list[str]:
    """Markdown bullets of sampled misclassified examples, grouped by true class."""
    rng = np.random.default_rng(seed)
    pred = probs.argmax(1)
    lines = []
    for c, name in enumerate(classes):
        idx = np.flatnonzero((y == c) & (pred != c))
        lines.append(f"#### True: {name} ({len(idx)} errors)\n")
        for i in rng.choice(idx, size=min(per_class, len(idx)), replace=False):
            text = texts.get(ids[i], "(text unavailable)")
            snippet = text[:300] + ("…" if len(text) > 300 else "")
            lines.append(
                f"- `{ids[i]}` → **{classes[pred[i]]}** (p={probs[i].max():.2f}): {snippet}"
            )
        lines.append("")
    return lines


def plot_confidence(y: np.ndarray, probs: np.ndarray, title: str, path: Path) -> None:
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(0, 1, 21)
    ax.hist(conf[correct], bins=bins, alpha=0.6, label="correct", density=True)
    ax.hist(conf[~correct], bins=bins, alpha=0.6, label="error", density=True)
    ax.set_xlabel("max probability")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def load_texts(raw_dir: Path) -> dict[str, str]:
    """Test texts by id, or {} if the raw data is not available."""
    try:
        from shared.data import load_frames

        _, test = load_frames(raw_dir, include_test=True)
    except Exception:
        return {}
    if test is None:
        return {}
    return dict(zip(test["id"].astype(str), test["text"], strict=True))


def build_error_report(
    raw_runs: pd.DataFrame,
    fetch: FetchPredictions,
    out: Path,
    texts: dict[str, str],
    seed: int = 0,
    per_class: int = 3,
) -> Path:
    runs = normalize_runs(raw_runs)
    final = runs[(runs["stage"] == "final") & (runs["fraction"] == 1.0)]
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Error analysis (test, fraction 1.0)",
        "",
        f"Seed {seed} where available, otherwise the method's first run. Labels in this corpus",
        f'are noisy and "{CATCH_ALL}" is a catch-all class, so read small gaps cautiously.',
        "",
    ]
    for method, g in final.groupby("method"):
        match = g[g["seed"] == seed]
        run = (match if not match.empty else g).iloc[0]
        pred_df = fetch(run["run_id"])
        if pred_df is None:
            continue
        ids, y, probs = read_predictions(pred_df)
        classes = class_names(pred_df)
        pairs = confusion_pairs(y, probs.argmax(1), classes)
        slug = str(method).replace("/", "_")
        fig_name = f"error_confidence_{slug}.png"
        plot_confidence(y, probs, f"{method}: confidence of errors vs correct", figures / fig_name)
        lines += [f"## {method}", "", "### Top confusion pairs", ""]
        lines.append(markdown_table(pairs))
        lines += [f"![confidence](figures/{fig_name})", "", "### Sampled misclassified", ""]
        lines += sample_errors(ids, y, probs, classes, texts, per_class, seed)
    path = out / "error_analysis.md"
    path.write_text("\n".join(lines))
    return path


def main(argv: list[str] | None = None) -> None:
    from shared.mlflow_utils import load_runs

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="laya-medabs-bench")
    parser.add_argument("--out", type=Path, default=Path("reports"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--per-class", type=int, default=3)
    args = parser.parse_args(argv)

    fetch = mlflow_fetcher(Path("data/cache/artifacts"))
    path = build_error_report(
        load_runs(args.experiment),
        fetch,
        args.out,
        load_texts(args.raw_dir),
        args.seed,
        args.per_class,
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
