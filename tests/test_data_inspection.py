"""Tests for the read-only dataset inspection code (synthetic data only)."""

import pandas as pd
import pytest
from PIL import Image

from src.data import inspection as ins
from src.data.datasets import DATASETS


@pytest.fixture
def table():
    return pd.DataFrame({
        "id": ["1_M_pic_1.jpg", "2_NM_pic.jpg", "3_M_pic_2.jpg", "4_NM_pic.jpg", "5_NM_pic.jpg"],
        "text": ["Ye kya hai yaar", "the cat is here", "  ", "The cat is HERE", "नमस्ते dost @user #tag"],
        "label": ["1", "0", "1", "1", "0"],
        "sub_a": ["1", "0", "0", "0", "0"],
        "sub_b": ["1", "0", "0", "0", "1"],
    })


def test_read_table_keeps_raw_strings(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("a,b\n01,\nNA,x\n", encoding="utf-8")
    df = ins.read_table(path)
    assert df["a"].tolist() == ["01", "NA"]  # no int conversion, "NA" not turned into NaN
    assert df["b"].tolist() == ["", "x"]


def test_schema_summary(table):
    s = ins.schema_summary(table)
    assert s["rows"] == 5 and s["column_count"] == 5
    assert s["columns"]["text"]["missing"] == 1  # whitespace-only counts as missing
    assert s["columns"]["label"]["inferred_type"] == "numeric-like"


def test_label_summary(table):
    s = ins.label_summary(table["label"])
    assert s["counts"] == {"0": 2, "1": 3}
    assert s["percentages"] == {"0": 40.0, "1": 60.0}
    assert s["type"] == "binary" and s["missing"] == 0


def test_sublabel_consistency(table):
    s = ins.sublabel_consistency(table, "label", ["sub_a", "sub_b"], "1", "id")
    assert s["rows_primary_not_positive_but_sublabel_positive"] == 1   # row 5
    assert s["ids_primary_not_positive_but_sublabel_positive"] == ["5_NM_pic.jpg"]
    assert s["rows_primary_positive_with_no_sublabel"] == 2            # rows 3, 4
    assert s["rows_with_multiple_sublabels"] == 1
    assert s["sublabel_combinations_when_primary_positive"]["sub_a+sub_b"] == 1


def test_text_summary(table):
    s = ins.text_summary(table["text"])
    assert s["non_empty"] == 4 and s["missing_or_empty"] == 1
    assert s["samples_with_mention"] == 1 and s["samples_with_hashtag"] == 1
    lang = s["language_indicator_heuristic"]
    assert lang["mixed_script"] == 1
    assert lang["latin_hindi_markers_only"] == 1
    assert lang["latin_english_markers_only"] == 2


def test_duplicate_summary_finds_case_duplicates_with_conflicting_labels(table):
    s = ins.duplicate_summary(table, "id", "text", "label")
    assert s["duplicate_ids"] == 0 and s["duplicate_full_rows"] == 0
    assert s["rows_in_exact_duplicate_text_groups"] == 0
    assert s["rows_in_case_whitespace_duplicate_groups"] == 2   # "the cat is here" vs "The cat is HERE"
    assert s["duplicate_text_groups_with_conflicting_labels"] == 1
    assert s["conflicting_label_group_ids"] == [["2_NM_pic.jpg", "4_NM_pic.jpg"]]


def test_duplicate_ids_detected():
    df = pd.DataFrame({"id": ["a", "a", ""], "text": ["x", "y", "z"]})
    s = ins.duplicate_summary(df, "id", "text", None)
    assert s["duplicate_ids"] == 1 and s["missing_ids"] == 1


def test_filename_label_association(table):
    s = ins.filename_label_association(table["id"], table["label"])
    assert s["pattern_label_counts"]["#_M_pic_#.jpg"] == {"1": 2}
    # 4_NM_pic.jpg has label 1, so the pattern does not perfectly predict the label
    assert s["pattern_perfectly_predicts_label"] is False
    clean = table.assign(label=["1", "0", "1", "0", "0"])
    assert ins.filename_label_association(clean["id"], clean["label"])["pattern_perfectly_predicts_label"]


def _make_image(path, size, color):
    Image.new("RGB", size, color).save(path, format="JPEG")


def test_image_summary(tmp_path):
    folder = tmp_path / "Files"
    folder.mkdir()
    _make_image(folder / "a.jpg", (100, 50), "red")
    _make_image(folder / "b.jpg", (100, 50), "red")      # byte-identical to a.jpg
    _make_image(folder / "c.jpg", (40, 80), "blue")
    (folder / "bad.jpg").write_bytes(b"not an image")
    Image.new("RGB", (10, 10)).save(folder / "png_named.jpg", format="PNG")
    _make_image(folder / "extra.jpg", (5, 5), "green")   # not referenced

    refs = pd.Series(["a.jpg", "b.jpg", "c.jpg", "bad.jpg", "png_named.jpg", "missing.jpg"])
    s = ins.image_summary(tmp_path, refs)
    assert s["status"] == "INSPECTED"
    assert s["found"] == 5 and s["missing"] == 1 and s["unreferenced_files_on_disk"] == 1
    assert s["corrupt_or_unreadable"] == 1 and s["corrupt_files"] == ["bad.jpg"]
    assert s["extension_format_mismatch"] == 1
    assert s["exact_duplicate_image_names"] == [["a.jpg", "b.jpg"]]
    assert s["width"]["max"] == 100 and s["height"]["max"] == 80


def test_image_summary_without_images(tmp_path):
    s = ins.image_summary(tmp_path, pd.Series(["a.jpg"]))
    assert s["status"] == "NO_IMAGES_FOUND" and s["missing"] == 1


def test_inspection_does_not_modify_files(tmp_path, table):
    csv_path = tmp_path / "data.csv"
    table.to_csv(csv_path, index=False)
    _make_image(tmp_path / "1_M_pic_1.jpg", (20, 20), "red")
    before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in tmp_path.iterdir()}
    df = ins.read_table(csv_path)
    ins.image_summary(tmp_path, df["id"])
    ins.file_inventory(tmp_path)
    after = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in tmp_path.iterdir()}
    assert before == after


def test_mimic_spec_points_inside_data_raw():
    spec = DATASETS["mimic"]
    assert spec.annotation_path.parts[-3:] == ("raw", "mimic", "MIMIC2024.csv")
    assert spec.documented["license"] == "TO_BE_VERIFIED"
    assert spec.documented["label_value_meaning"] == "TO_BE_VERIFIED"


# ---------------- scripts/inspect_datasets.py ----------------

def _load_script():
    import importlib.util
    from src.utils.config import REPO_ROOT

    spec = importlib.util.spec_from_file_location(
        "inspect_datasets", REPO_ROOT / "scripts" / "inspect_datasets.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_missing_dataset_never_overwrites_existing_report(tmp_path, monkeypatch):
    from src.data.datasets import DatasetSpec

    script = _load_script()
    missing = DatasetSpec(name="X", directory="does_not_exist", annotation_file="a.csv",
                          id_column="id", text_column=None, image_column=None, primary_label="y")
    monkeypatch.setattr(script, "DATASETS", {"x": missing})
    monkeypatch.setattr(script, "RAW_DIR", tmp_path)
    monkeypatch.setattr(script, "METRICS", tmp_path / "metrics")
    monkeypatch.setattr(script, "TABLES", tmp_path / "tables")
    existing = tmp_path / "metrics" / "dataset_inspection_x.json"
    existing.parent.mkdir()
    existing.write_text('{"real": "measurements"}', encoding="utf-8")

    assert script.main() == 1
    assert existing.read_text(encoding="utf-8") == '{"real": "measurements"}'
    assert not (tmp_path / "tables").exists()


def test_write_report_skips_timestamp_only_changes(tmp_path):
    script = _load_script()
    out = tmp_path / "r.json"
    assert script.write_report(out, {"generated_at": "t1", "rows": 5}) == "written"
    assert script.write_report(out, {"generated_at": "t2", "rows": 5}) == "unchanged"
    assert '"t1"' in out.read_text(encoding="utf-8")
    assert script.write_report(out, {"generated_at": "t3", "rows": 6}) == "written"
