"""Download the corpus, write the split manifest and the data audit (`make data`)."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from shared.data import audit, download, load_frames, to_split
from shared.splits import load_manifest, make_manifest, save_manifest


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--audit-out", default="reports/data_audit.md")
    ap.add_argument("--no-tokenizer", action="store_true", help="count whitespace words instead")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    ds, laya = cfg["dataset"], cfg["laya"]
    download(ds["raw_dir"], ds["hf_id"])
    train_df, test_df = load_frames(ds["raw_dir"], include_test=True)
    assert test_df is not None

    path = Path(ds["split_manifest"])
    if path.exists():
        manifest = load_manifest(path)  # carved once; never re-split
    else:
        manifest = make_manifest(
            train_df["id"].tolist(),
            train_df["label"].to_numpy(),
            test_df["id"].tolist(),
            val_fraction=ds["val_fraction"],
            split_seed=ds["split_seed"],
            version=ds["version"],
        )
        save_manifest(manifest, path)

    train = to_split(train_df)
    pos = {id_: i for i, id_ in enumerate(train.ids)}
    splits = {
        "train_pool": train.subset([pos[i] for i in manifest["train_pool"]]),
        "val": train.subset([pos[i] for i in manifest["val"]]),
        "test": to_split(test_df),
    }
    report = audit(
        splits,
        max_length=ds["max_length"],
        hub_id=None if args.no_tokenizer else laya["hub_id"],
        revision=laya["revision"],
        split_hash=manifest["hash"],
    )
    out = Path(args.audit_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    sizes = ", ".join(f"{k}={len(v)}" for k, v in splits.items())
    print(f"split hash {manifest['hash']} ({sizes}); audit -> {out}")


if __name__ == "__main__":
    main()
