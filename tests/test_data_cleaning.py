"""Tests for the cleaning pipeline (src/data/cleaning.py) on synthetic data.

The fixture writes a tiny dataset (CSV + generated images) into a temp
folder that exercises every cleaning rule. No real dataset is used.
"""

import csv
import importlib.util

import pandas as pd
import pytest
from PIL import Image

from src.data.cleaning import POLICY, clean_dataset, make_sample_id, read_records, validate_cleaned
from src.data.datasets import DatasetSpec
from src.utils.config import REPO_ROOT

MAPPING = {"0": 0, "1": 1}
HEADER = ["FileName", "ExtractedText", "Misogyny", "Objectification", "Prejudice", "Humiliation"]
ROWS = [
    ["a.jpg", "hello", "1", "1", "0", "0"],          # kept (first of identical pair)
    ["b.jpg", "hello", "1", "0", "1", "0"],          # identical image+text to a -> removed; sub-labels differ
    ["c.jpg", "same caption", "0", "0", "0", "0"],   # duplicate text, different image ...
    ["d.jpg", "Same  caption", "1", "1", "0", "0"],  # ... with a different label -> kept, flagged
    ["e.jpg", "x", "0", "0", "0", "0"],              # identical image+text with ...
    ["f.jpg", "x", "1", "1", "0", "0"],              # ... conflicting label -> both removed
    ["missing.jpg", "t", "0", "0", "0", "0"],        # image not on disk -> removed
    ["bad.jpg", "t2", "0", "0", "0", "0"],           # unreadable image -> removed
    ["", "t3", "0", "0", "0", "0"],                  # missing ID -> removed
    ["g.jpg", "t4", "", "0", "0", "0"],              # missing label -> removed
    ["h.jpg", "t5", "2", "0", "0", "0"],             # invalid label -> removed
    ["c.jpg", "same caption", "0", "0", "0", "0"],   # exact repeat of row 3 -> removed
    ["i.jpg", "  ", "0", "0", "0", "0"],             # no text -> kept, has_text False
    ["K.JPG", "t6", "1", "0", "1", "0"],             # file on disk is k.jpg -> path repaired
    ["m.jpg", "t7", "1", "0", "0", "0"],             # positive, no sub-label -> flagged
    ["n.jpg", "t8", "0", "x", "0", "0"],             # invalid sub-label -> kept, flagged
]
MALFORMED = ["z.jpg", "too", "many", "fields", "0", "0", "0", "0"]


def _jpeg(path, color, size=(16, 16)):
    Image.new("RGB", size, color).save(path, format="JPEG")


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "data" / "raw" / "toy"
    files = root / "Files"
    files.mkdir(parents=True)
    _jpeg(files / "a.jpg", "red")
    (files / "b.jpg").write_bytes((files / "a.jpg").read_bytes())    # byte-identical
    _jpeg(files / "c.jpg", "blue")
    _jpeg(files / "d.jpg", "green")
    _jpeg(files / "e.jpg", "yellow")
    (files / "f.jpg").write_bytes((files / "e.jpg").read_bytes())
    (files / "bad.jpg").write_bytes(b"not an image")
    for name, color in [("g.jpg", "gray"), ("h.jpg", "pink"), ("i.jpg", "white"),
                        ("k.jpg", "black"), ("m.jpg", "purple"), ("n.jpg", "orange")]:
        _jpeg(files / name, color)
    with open(root / "toy.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(HEADER)
        writer.writerows(ROWS[:5])
        writer.writerow(MALFORMED)                   # source_row 6
        writer.writerows(ROWS[5:])
    spec = DatasetSpec(
        name="TOY", directory=str(root), annotation_file="toy.csv", id_column="FileName",
        text_column="ExtractedText", image_column="FileName", primary_label="Misogyny",
        secondary_labels=["Objectification", "Prejudice", "Humiliation"], positive_value="1",
        image_directory="Files",
    )
    return spec, tmp_path


def run(dataset):
    spec, repo_root = dataset
    return clean_dataset(spec, "toy", MAPPING, repo_root)


def by_ref(result):
    return result.cleaned.set_index("image_reference")


def test_raw_files_never_modified(dataset):
    spec, _ = dataset
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in spec.root.rglob("*") if p.is_file()}
    run(dataset)
    after = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in spec.root.rglob("*") if p.is_file()}
    assert before == after


def test_every_issue_excluded_with_logged_reason(dataset):
    result = run(dataset)
    got = {(e["image_reference"], e["issue_category"]) for e in result.excluded}
    assert got == {
        ("b.jpg", "duplicate_image_and_text"),
        ("e.jpg", "conflicting_labels_identical_sample"),
        ("f.jpg", "conflicting_labels_identical_sample"),
        ("missing.jpg", "missing_image"),
        ("bad.jpg", "corrupt_image"),
        ("", "missing_id"),
        ("g.jpg", "missing_label"),
        ("h.jpg", "invalid_label"),
        ("c.jpg", "duplicate_row"),
        ("", "malformed_record"),
    }
    assert all(e["exclusion_reason"] and e["action"] for e in result.excluded)
    # counts reconcile: nothing removed silently
    assert result.counts["retained"] + result.counts["excluded"] == result.counts["original_records"] == 17


def test_excluded_log_contains_no_text(dataset):
    result = run(dataset)
    texts = {r[1] for r in ROWS if len(r[1].strip()) > 2}
    assert not any(t in str(e.values()) for e in result.excluded for t in texts)


def test_malformed_record_detected(dataset):
    spec, _ = dataset
    df, malformed, header = read_records(spec.annotation_path)
    assert malformed == [{"source_row": 6, "field_count": 8}]
    assert header == HEADER and len(df) == len(ROWS)


def test_required_columns_validated(dataset):
    spec, repo_root = dataset
    broken = DatasetSpec(**{**spec.__dict__, "primary_label": "NotAColumn"})
    with pytest.raises(ValueError, match="missing required columns"):
        clean_dataset(broken, "toy", MAPPING, repo_root)


def test_stable_sample_ids(dataset):
    result = run(dataset)
    ids = by_ref(result)["sample_id"]
    assert ids["a.jpg"] == make_sample_id("toy", "a.jpg")
    assert make_sample_id("toy", "a.jpg") == make_sample_id("toy", "a.jpg")
    assert make_sample_id("toy", "a.jpg") != make_sample_id("toy", "c.jpg")
    assert not any("_M_" in s or ".jpg" in s for s in ids)       # raw file name not leaked
    assert result.cleaned["sample_id"].is_unique


def test_sample_ids_independent_of_row_order(dataset):
    spec, repo_root = dataset
    first = run(dataset).cleaned.set_index("image_reference")["sample_id"]
    lines = spec.annotation_path.read_text(encoding="utf-8").splitlines()
    spec.annotation_path.write_text("\n".join([lines[0], *reversed(lines[1:])]) + "\n", encoding="utf-8")
    second = clean_dataset(spec, "toy", MAPPING, repo_root).cleaned.set_index("image_reference")["sample_id"]
    common = first.index.intersection(second.index)
    assert len(common) >= 6
    assert (first[common] == second[common]).all()


def test_label_mapping_preserves_original(dataset):
    c = run(dataset).cleaned
    assert (c["original_label"].map(MAPPING) == c["target_label"]).all()
    assert set(c["target_label"]) <= {0, 1}
    assert c["original_label"].map(type).eq(str).all()


def test_original_text_preserved_exactly(dataset):
    c = by_ref(run(dataset))
    assert c.loc["d.jpg", "original_text"] == "Same  caption"   # case + double space kept
    assert c.loc["i.jpg", "original_text"] == "  "


def test_missing_text_kept_and_marked(dataset):
    c = by_ref(run(dataset))
    assert not c.loc["i.jpg", "has_text"]
    assert c.loc["a.jpg", "has_text"]


def test_duplicate_handling(dataset):
    result = run(dataset)
    c = by_ref(result)
    # identical image+text, same label: first kept, flagged because sub-labels differ
    assert "secondary_label_conflict" in c.loc["a.jpg", "review_flags"]
    b = next(e for e in result.excluded if e["image_reference"] == "b.jpg")
    assert b["kept_representative"] == c.loc["a.jpg", "sample_id"]
    # same text on different images is KEPT, grouped and flagged when labels differ
    assert c.loc["c.jpg", "duplicate_group"] == c.loc["d.jpg", "duplicate_group"] != ""
    assert "text_label_conflict" in c.loc["d.jpg", "review_flags"]
    types = sorted(d["group_type"] for d in result.duplicate_decisions)
    assert types == ["duplicate_text_different_images", "identical_image_and_text", "identical_image_and_text"]


def test_image_path_repaired_and_flagged(dataset):
    c = by_ref(run(dataset))
    assert c.loc["K.JPG", "image_path"].endswith("Files/k.jpg")
    assert "image_path_repaired" in c.loc["K.JPG", "review_flags"]


def test_secondary_label_flags(dataset):
    c = by_ref(run(dataset))
    assert "no_secondary_label" in c.loc["m.jpg", "review_flags"]
    assert pd.isna(c.loc["n.jpg", "Objectification"])
    assert "invalid_secondary_label" in c.loc["n.jpg", "review_flags"]


def test_deterministic_output(dataset):
    r1, r2 = run(dataset), run(dataset)
    pd.testing.assert_frame_equal(r1.cleaned, r2.cleaned)
    assert r1.excluded == r2.excluded and r1.duplicate_decisions == r2.duplicate_decisions
    assert r1.cleaned.to_csv(index=False) == r2.cleaned.to_csv(index=False)


def test_validate_cleaned(dataset):
    _, repo_root = dataset
    c = run(dataset).cleaned
    assert all(validate_cleaned(c, MAPPING, repo_root).values())
    bad = pd.concat([c, c.iloc[[0]]], ignore_index=True)
    bad.loc[len(bad) - 1, "target_label"] = 7
    checks = validate_cleaned(bad, MAPPING, repo_root)
    assert not checks["sample_ids_unique"] and not checks["target_labels_valid"]


def test_policy_actions_are_known():
    allowed = {"KEEP", "REMOVE", "REPAIR METADATA", "FLAG FOR REVIEW"}
    for action, reason in POLICY.values():
        assert set(a.strip() for a in action.split("+")) <= allowed and reason


def test_clean_script_fails_safely_without_dataset(monkeypatch):
    spec = importlib.util.spec_from_file_location("clean_dataset", REPO_ROOT / "scripts" / "clean_dataset.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    missing = DatasetSpec(name="X", directory="does_not_exist", annotation_file="a.csv",
                          id_column="id", text_column=None, image_column=None, primary_label="y")
    monkeypatch.setattr(script, "DATASETS", {"mimic": missing})
    out = REPO_ROOT / "data" / "processed" / "mimic_cleaned.csv"
    before = out.stat().st_mtime_ns if out.exists() else None
    assert script.main() == 1
    assert (out.stat().st_mtime_ns if out.exists() else None) == before
