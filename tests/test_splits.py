"""Tests for leakage-safe splitting (src/data/splits.py). Synthetic data only."""

import pandas as pd
import pytest

from src.data import splits as sp
from src.utils.config import REPO_ROOT, load_config

RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}


@pytest.fixture(scope="module")
def frame():
    n = 2000
    df = pd.DataFrame({
        "sample_id": [f"s{i:04d}" for i in range(n)],
        "image_sha256": [f"h{i:04d}" for i in range(n)],
        "duplicate_group": [""] * n,
        "text_raw": [f"caption number {i} here" for i in range(n)],
        "text_hinglish_normalized": [f"caption number {i} here" for i in range(n)],
        "target_label": [str(i % 2) for i in range(n)],
    })
    df.loc[1, "image_sha256"] = df.loc[0, "image_sha256"]            # identical image pair: s000, s001
    df.loc[[10, 11, 12], "duplicate_group"] = "txt_a"                # substantive shared caption
    df.loc[[10, 11, 12], "text_raw"] = "the very same long caption"
    df.loc[[20, 21], "duplicate_group"] = "txt_b"                    # trivial 1-word caption
    df.loc[[20, 21], "text_raw"] = "lol"
    return df


@pytest.fixture(scope="module")
def near():
    return pd.DataFrame({"sample_id_a": ["s0030", "s0031"], "sample_id_b": ["s0031", "s0032"]})  # chain


def run(df, near, seed=42):
    groups, links = sp.build_groups(df, near)
    split = sp.assign_splits(df["target_label"].astype(int), groups, RATIOS, seed, n_folds=20)
    return groups, links, split


@pytest.fixture(scope="module")
def result(frame, near):
    """The seed-42 split of the fixture, computed once (fixtures are never mutated)."""
    return run(frame, near)


def test_groups_formed_for_each_reason(frame, near, result):
    groups, links, _ = result
    g = dict(zip(frame["sample_id"], groups))
    assert g["s0000"] == g["s0001"]                                   # identical image
    assert g["s0010"] == g["s0011"] == g["s0012"]                      # substantive caption
    assert g["s0030"] == g["s0031"] == g["s0032"]                      # near-duplicate chain
    assert g["s0020"] != g["s0021"]                                   # trivial caption NOT grouped
    assert links["reason"].str.startswith("NOT_GROUPED_trivial").sum() == 1
    assert g["s0100"] != g["s0101"]                                   # unrelated samples apart


def test_valid_split_names_and_full_coverage(frame, near, result):
    _, _, split = result
    assert set(split) == set(sp.SPLITS)
    assert split.notna().all() and len(split) == len(frame)


def test_ratios_close_to_target(frame, near, result):
    _, _, split = result
    for name, ratio in RATIOS.items():
        assert abs((split == name).mean() - ratio) < 0.01


def test_class_stratification(frame, near, result):
    _, _, split = result
    labels = frame["target_label"].astype(int)
    for name in sp.SPLITS:
        assert abs(labels[split == name].mean() - labels.mean()) < 0.02


def test_groups_never_cross_splits(frame, near, result):
    groups, _, split = result
    assert (pd.DataFrame({"g": groups, "s": split}).groupby("g")["s"].nunique() == 1).all()


def test_identical_image_hashes_never_cross(frame, near, result):
    _, _, split = result
    assert (pd.DataFrame({"h": frame["image_sha256"], "s": split}).groupby("h")["s"].nunique() == 1).all()


def test_deterministic_and_seed_dependent(frame, near):
    _, _, a = run(frame, near, seed=42)
    _, _, b = run(frame, near, seed=42)
    _, _, c = run(frame, near, seed=7)
    assert a.equals(b)
    assert not a.equals(c)


def test_split_does_not_depend_on_text_variant(frame, near, result):
    _, _, a = result
    changed = frame.assign(text_hinglish_normalized="something else entirely")
    _, _, b = run(changed, near)
    assert a.equals(b)


def test_audit_passes_on_valid_split(frame, near, result):
    groups, _, split = result
    audit = sp.leakage_audit(frame, split, groups, near)
    assert audit.loc[audit["type"] == "blocking", "passed"].all()


def test_audit_detects_leakage(frame, near, result):
    groups, _, split = result
    broken = split.copy()
    broken[frame["sample_id"] == "s0001"] = "test" if broken[0] != "test" else "train"   # separate identical images
    audit = sp.leakage_audit(frame, broken, groups, near).set_index("check")
    assert not audit.loc["identical_images_not_crossing", "passed"]
    assert not audit.loc["leakage_groups_not_crossing", "passed"]


def test_audit_detects_lost_and_duplicated_samples(frame, near, result):
    groups, _, split = result
    duplicated = pd.concat([frame, frame.iloc[[5]]], ignore_index=True)
    audit = sp.leakage_audit(duplicated, pd.concat([split, split.iloc[[5]]], ignore_index=True),
                             pd.concat([groups, groups.iloc[[5]]], ignore_index=True), near).set_index("check")
    assert not audit.loc["sample_id_unique", "passed"]
    lost = split.copy()
    lost.iloc[3] = None
    audit = sp.leakage_audit(frame, lost, groups, near).set_index("check")
    assert not audit.loc["every_sample_assigned", "passed"] and not audit.loc["union_equals_dataset", "passed"]


def test_folds_per_split_validation():
    assert sp.folds_per_split(RATIOS, 20) == {"train": 14, "validation": 3, "test": 3}
    with pytest.raises(ValueError):
        sp.folds_per_split({"train": 0.7, "validation": 0.2, "test": 0.15}, 20)
    with pytest.raises(ValueError):
        sp.folds_per_split(RATIOS, 7)


def test_distribution_table(frame, near, result):
    _, _, split = result
    dist = sp.distribution(frame, split, {0: "neg", 1: "pos"}).set_index("split")
    assert dist.loc["all", "samples"] == 2000
    assert dist.loc[list(sp.SPLITS), "samples"].sum() == 2000


# ---------------- canonical artifacts (skipped if not generated) ----------------

def _canonical_available():
    cfg = load_config()["splitting"]
    return (REPO_ROOT / cfg["assignments_file"]).exists() and (REPO_ROOT / cfg["manifest_file"]).exists()


@pytest.mark.skipif(not _canonical_available(), reason="canonical split not generated on this machine")
def test_canonical_assignments_match_manifest():
    table = sp.load_split_assignments()
    assert table["sample_id"].is_unique and set(table["split"]) == set(sp.SPLITS)


def test_test_split_is_locked_by_default():
    with pytest.raises(PermissionError):
        sp.load_split("test")
    with pytest.raises(ValueError):
        sp.load_split("holdout")
