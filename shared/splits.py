"""Validation carve-out, nested stratified training fractions and the split manifest."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

_HASHED_KEYS = ("version", "val_fraction", "split_seed", "train_pool", "val", "test")


def manifest_hash(manifest: dict[str, Any]) -> str:
    """SHA-256 over the canonical JSON of the manifest's content (not its `hash` field)."""
    body = {k: manifest[k] for k in _HASHED_KEYS}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def make_manifest(
    train_ids: list[str],
    train_labels: np.ndarray,
    test_ids: list[str],
    val_fraction: float = 0.10,
    split_seed: int = 1234,
    version: str = "v1",
) -> dict[str, Any]:
    """Carve a stratified validation set out of the official train split, once."""
    pool, val = train_test_split(
        list(train_ids),
        test_size=val_fraction,
        stratify=np.asarray(train_labels),
        random_state=split_seed,
    )
    manifest: dict[str, Any] = {
        "version": version,
        "val_fraction": val_fraction,
        "split_seed": split_seed,
        "train_pool": sorted(pool, key=_id_order),
        "val": sorted(val, key=_id_order),
        "test": list(test_ids),
    }
    manifest["hash"] = manifest_hash(manifest)
    return manifest


def _id_order(id_: str) -> tuple[str, int]:
    prefix, _, num = id_.rpartition("-")
    return prefix, int(num)


def save_manifest(manifest: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1) + "\n")


def load_manifest(path: str | Path) -> dict[str, Any]:
    """Load the manifest and verify its hash."""
    manifest: dict[str, Any] = json.loads(Path(path).read_text())
    if manifest.get("hash") != manifest_hash(manifest):
        raise ValueError(f"{path}: split manifest hash mismatch")
    return manifest


def fraction_ids(
    pool_ids: list[str], pool_labels: np.ndarray, seed: int, fraction: float
) -> list[str]:
    """Stratified training subset for (seed, fraction), nested across fractions.

    One permutation per (seed, class); a fraction takes the first ceil(f * n_class) IDs of
    each class, so the 1% set is a prefix of the 5% set and so on. Returned in pool order.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    labels = np.asarray(pool_labels)
    if len(labels) != len(pool_ids):
        raise ValueError("pool_ids and pool_labels differ in length")
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        perm = rng.permutation(idx)
        keep.extend(perm[: math.ceil(fraction * len(idx))].tolist())
    return [pool_ids[i] for i in sorted(keep)]
