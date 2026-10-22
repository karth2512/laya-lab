from pathlib import Path

import numpy as np
import pytest

from shared.data import CLASSES, audit, find_duplicates, load_frames, to_split
from shared.splits import fraction_ids, load_manifest, make_manifest, save_manifest

RAW = Path(__file__).parent / "fixtures" / "raw"
FRACTIONS = [0.01, 0.05, 0.1, 0.25, 1.0]


@pytest.fixture
def frames():
    train, test = load_frames(RAW, include_test=True)
    assert test is not None
    return train, test


def _manifest(frames, seed=1234):
    train, test = frames
    return make_manifest(
        train["id"].tolist(), train["label"].to_numpy(), test["id"].tolist(), 0.25, seed
    )


def test_load_frames_maps_labels_and_skips_test():
    train, test = load_frames(RAW)
    assert test is None
    assert set(train["label"]) == set(range(len(CLASSES)))
    assert train["id"].iloc[0] == "train-0"


def test_manifest_deterministic_disjoint_and_stable(frames, tmp_path):
    m1, m2 = _manifest(frames), _manifest(frames)
    assert m1 == m2
    pool, val, test = set(m1["train_pool"]), set(m1["val"]), set(m1["test"])
    assert not (pool & val) and not (pool & test) and not (val & test)
    assert len(pool) + len(val) == len(frames[0])
    save_manifest(m1, tmp_path / "m.json")
    assert load_manifest(tmp_path / "m.json")["hash"] == m1["hash"]
    assert _manifest(frames, seed=7)["hash"] != m1["hash"]


def test_manifest_hash_is_verified(frames, tmp_path):
    m = _manifest(frames)
    m["val"] = m["val"][:-1]
    save_manifest(m, tmp_path / "m.json")
    with pytest.raises(ValueError, match="hash"):
        load_manifest(tmp_path / "m.json")


def test_val_is_stratified(frames):
    train, _ = frames
    m = _manifest(frames)
    labels = train.set_index("id").loc[m["val"], "label"]
    assert labels.value_counts().tolist() == [2] * 5  # 25% of 8 per class


def test_fractions_nested_stratified_deterministic():
    rng = np.random.default_rng(0)
    labels = rng.choice(5, size=1000, p=[0.4, 0.1, 0.2, 0.2, 0.1])
    ids = [f"train-{i}" for i in range(1000)]
    for seed in (0, 1):
        sets = [fraction_ids(ids, labels, seed, f) for f in FRACTIONS]
        assert sets == [fraction_ids(ids, labels, seed, f) for f in FRACTIONS]
        for small, big in zip(sets, sets[1:], strict=False):
            assert set(small) < set(big)
        assert sets[-1] == ids
        sub = labels[[int(i.split("-")[1]) for i in sets[2]]]  # 10%
        for c in range(5):
            assert abs((sub == c).sum() - 0.1 * (labels == c).sum()) <= 1
    assert fraction_ids(ids, labels, 0, 0.1) != fraction_ids(ids, labels, 1, 0.1)
    with pytest.raises(ValueError):
        fraction_ids(ids, labels, 0, 0.0)


def test_audit_reports_duplicates(frames):
    train, test = frames
    tr, te = to_split(train), to_split(test)
    dup = te.subset([0])
    dup.texts = [tr.texts[0].upper() + "!"]
    counts = find_duplicates({"train": tr, "test": dup})
    assert counts["exact/train~test"] == 0 and counts["near/train~test"] == 1
    report = audit({"train": tr, "test": te})
    assert "CC BY-SA 3.0" in report and "Class distribution" in report
