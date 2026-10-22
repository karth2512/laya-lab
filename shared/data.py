"""Download, load and audit the Medical Abstracts TC Corpus. The label mapping lives here only."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from shared.interface import Split

HF_DATASET = "TimSchopf/medical_abstracts"

# Label id i (0-based) is numeric label i+1 in the corpus.
CLASSES: tuple[str, ...] = (
    "Neoplasms",
    "Digestive system diseases",
    "Nervous system diseases",
    "Cardiovascular diseases",
    "General pathological conditions",
)

LICENSE_NOTE = (
    "Data: Medical Abstracts TC Corpus (Schopf et al., `TimSchopf/medical_abstracts`), "
    "licensed CC BY-SA 3.0."
)


def download(raw_dir: str | Path, hf_id: str = HF_DATASET) -> None:
    """Write `train.csv` and `test.csv` (condition_label, medical_abstract) into raw_dir."""
    from huggingface_hub import hf_hub_download

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        out = raw_dir / f"{split}.csv"
        if out.exists():
            continue
        path = hf_hub_download(hf_id, f"data/{split}-00000-of-00001.parquet", repo_type="dataset")
        pd.read_parquet(path)[["condition_label", "medical_abstract"]].to_csv(out, index=False)


def _read(path: Path, prefix: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    labels = raw["condition_label"].astype(int) - 1
    if not labels.between(0, len(CLASSES) - 1).all():
        raise ValueError(f"{path}: condition_label outside 1..{len(CLASSES)}")
    return pd.DataFrame(
        {
            "id": [f"{prefix}-{i}" for i in range(len(raw))],
            "text": raw["medical_abstract"].astype(str),
            "label": labels,
        }
    )


def load_frames(
    raw_dir: str | Path, include_test: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """Return (train_df, test_df) with columns id, text, label. The test file is not read
    unless include_test is True, so stage=dev never holds test labels."""
    raw_dir = Path(raw_dir)
    train = _read(raw_dir / "train.csv", "train")
    test = _read(raw_dir / "test.csv", "test") if include_test else None
    return train, test


def to_split(df: pd.DataFrame) -> Split:
    return Split(
        ids=df["id"].tolist(),
        texts=df["text"].tolist(),
        labels=df["label"].to_numpy(dtype=np.int64),
    )


# ---------------------------------------------------------------- hygiene audit


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def find_duplicates(splits: dict[str, Split], prefix_words: int = 30) -> dict[str, int]:
    """Count exact and near duplicates within and across splits.

    exact: identical raw text. near: identical after lowercasing/stripping punctuation, or
    identical first `prefix_words` normalized words (catches re-posted abstracts with edits
    at the end). Counts are pairs of splits, e.g. "near/train_pool~test".
    """
    keys: dict[str, Callable[[str], str]] = {
        "exact": lambda t: _digest(t),
        "near": lambda t: _digest(_normalize(t)),
        "near_prefix": lambda t: _digest(" ".join(_normalize(t).split()[:prefix_words])),
    }
    names = list(splits)
    out: dict[str, int] = {}
    for kind, fn in keys.items():
        hashed = {n: [fn(t) for t in splits[n].texts] for n in names}
        for i, a in enumerate(names):
            ha = hashed[a]
            out[f"{kind}/{a}~{a}"] = len(ha) - len(set(ha))
            for b in names[i + 1 :]:
                out[f"{kind}/{a}~{b}"] = len(set(ha) & set(hashed[b]))
    return out


def token_lengths(
    texts: list[str], hub_id: str | None = None, revision: str | None = None
) -> tuple[np.ndarray, str]:
    """Token counts with the Laya tokenizer when transformers can load it, else whitespace."""
    if hub_id:
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(hub_id, revision=revision)
            ids = tok(texts, add_special_tokens=True, truncation=False)["input_ids"]
            return np.array([len(x) for x in ids]), f"Laya tokenizer (`{hub_id}`@`{revision}`)"
        except Exception as exc:  # noqa: BLE001 - any failure falls back to a rough count
            note = f"whitespace words (Laya tokenizer unavailable: {type(exc).__name__})"
            return np.array([len(t.split()) for t in texts]), note
    return np.array([len(t.split()) for t in texts]), "whitespace words"


def audit(
    splits: dict[str, Split],
    max_length: int = 512,
    hub_id: str | None = None,
    revision: str | None = None,
    split_hash: str | None = None,
) -> str:
    """Markdown report: duplicates, token lengths / truncation, class balance per split."""
    lines = ["# Data audit", ""]
    if split_hash:
        lines += [f"Split manifest hash: `{split_hash}`", ""]

    lines += ["## Class distribution", ""]
    lines += ["| Class | " + " | ".join(splits) + " |", "|---|" + "---|" * len(splits)]
    for c, name in enumerate(CLASSES):
        cells = []
        for s in splits.values():
            n = int((s.labels == c).sum())
            cells.append(f"{n} ({100 * n / max(len(s), 1):.1f}%)")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines.append("| **Total** | " + " | ".join(str(len(s)) for s in splits.values()) + " |")

    lines += ["", "## Token lengths", ""]
    method = ""
    rows = []
    for name, s in splits.items():
        lens, method = token_lengths(s.texts, hub_id, revision)
        p50, p95, p99 = np.percentile(lens, [50, 95, 99])
        trunc = 100 * float((lens > max_length).mean())
        rows.append(
            f"| {name} | {p50:.0f} | {lens.mean():.0f} | {p95:.0f} | {p99:.0f} | "
            f"{lens.max()} | {trunc:.2f}% |"
        )
    lines += [
        f"Counted with: {method}. Truncation limit: {max_length} tokens.",
        "",
        f"| Split | p50 | mean | p95 | p99 | max | % > {max_length} |",
        "|---|---|---|---|---|---|---|",
        *rows,
    ]

    lines += [
        "",
        "## Duplicates",
        "",
        "`exact` = identical text; `near` = identical after lowercasing and stripping "
        "punctuation; `near_prefix` = identical first 30 normalized words. "
        "`a~a` counts repeats within a split, `a~b` counts shared texts across splits.",
        "",
        "| Check | Count |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in find_duplicates(splits).items()]
    lines += ["", "## License", "", LICENSE_NOTE, ""]
    return "\n".join(lines)
