"""Tests for EDA statistics and plotting (synthetic data only)."""

import ast
import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from src.data import eda, eda_plots
from src.utils.config import REPO_ROOT

NAMES = {0: "non_misogynistic", 1: "misogynistic"}


@pytest.fixture
def df():
    return pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(6)],
        "original_text": ["Ye kya hai @page1", "the girl is here!!!", "", "नमस्ते दोस्त 123", "LOOK AT THIS NOW",
                          "soooo funny #tag @page1"],
        "has_text": [True, True, False, True, True, True],
        "target_label": [1, 1, 1, 0, 0, 0],
        "image_reference": ["1_M_pic_1.jpg", "2_M_pic_2.jpg", "3_M_pic_1.jpg", "4_NM_pic.jpg", "5_NM_pic.jpg",
                            "6_NM_pic.jpg"],
        "duplicate_group": ["", "", "", "", "", ""],
        "Objectification": pd.array([1, 0, 1, 0, 0, 0], dtype="Int64"),
        "Prejudice": pd.array([1, 1, 0, 0, 0, 0], dtype="Int64"),
    })


def test_eda_tokens_only_letters_and_devanagari():
    assert eda.eda_tokens("Hai! kya 123 है।") == ["hai", "kya", "है"]
    assert eda.eda_tokens("") == []


def test_eda_does_not_modify_input(df):
    before = df.copy(deep=True)
    eda.text_features(df)
    eda.top_terms(df, NAMES)
    eda.distinctive_terms(df, NAMES, min_count=1)
    pd.testing.assert_frame_equal(df, before)


def test_text_features(df):
    f = eda.text_features(df).set_index("sample_id")
    assert f.loc["s0", "has_mention"] and f.loc["s5", "has_hashtag"]
    assert f.loc["s1", "has_repeated_punctuation"] and f.loc["s5", "has_repeated_characters"]
    assert f.loc["s4", "is_uppercase_heavy"] and not f.loc["s1", "is_uppercase_heavy"]
    assert f.loc["s3", "has_digit"] and f.loc["s3", "script_category"] == "devanagari_only"
    assert f.loc["s2", "script_category"] == "empty"
    assert f.loc["s0", "has_romanized_hindi_marker"] and f.loc["s1", "has_english_marker"]
    assert f.loc["s1", "char_count"] == len("the girl is here!!!") and f.loc["s1", "word_count"] == 4


def test_class_distribution(df):
    table, ratio = eda.class_distribution(df, NAMES)
    assert table["count"].tolist() == [3, 3] and table["percent"].tolist() == [50.0, 50.0]
    assert ratio["imbalance_ratio_majority_to_minority"] == 1.0


def test_describe_percentiles():
    d = eda.describe(pd.Series(range(1, 101)))
    assert d["count"] == 100 and d["min"] == 1 and d["max"] == 100
    assert d["median"] == 50.5 and d["p90"] == pytest.approx(90.1)


def test_length_statistics_excludes_empty(df):
    stats = eda.length_statistics(eda.text_features(df), NAMES)
    row = stats[(stats.metric == "char_count") & (stats.group == "misogynistic")].iloc[0]
    assert row["count"] == 2   # the empty text is not counted


def test_secondary_label_stats(df):
    s = eda.secondary_label_stats(df, ["Objectification", "Prejudice"])
    per = s["per_label"].set_index("label")
    assert per.loc["Objectification", "count_within_positive_class"] == 2
    assert s["cooccurrence_within_positive"].loc["Objectification", "Prejudice"] == 1
    assert set(s["combinations"]["combination"]) == {"Objectification+Prejudice", "Prejudice", "Objectification"}


def test_distinctive_terms_direction():
    texts = ["girl girl girl x"] * 5 + ["boy boy boy x"] * 5
    d = pd.DataFrame({"original_text": texts, "target_label": [1] * 5 + [0] * 5})
    table = eda.distinctive_terms(d, NAMES, min_count=1, top=1)
    assert table.iloc[0]["term"] == "girl" and table.iloc[0]["z_score"] > 0
    assert table.iloc[1]["term"] == "boy" and table.iloc[1]["z_score"] < 0


def test_feature_by_class_and_handles(df):
    social = eda.feature_by_class(eda.text_features(df), NAMES).set_index("feature")
    assert social.loc["has_mention", "count_misogynistic"] == 1
    assert social.loc["has_mention", "percent_misogynistic"] == 50.0   # 1 of 2 non-empty texts
    handles = eda.mention_handles(df, NAMES, min_samples=2)
    assert handles.iloc[0]["handle"] == "@page1" and handles.iloc[0]["max_class_share"] == 0.5


def test_id_pattern_association(df):
    assert eda.id_pattern_association(df)["pattern_perfectly_predicts_label"]


def _img(path, array):
    Image.fromarray(array.astype(np.uint8)).save(path, format="PNG")


def test_images_resolutions_near_duplicates_and_outliers(tmp_path, df):
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (64, 64, 3))
    arrays = [base, base, rng.integers(0, 255, (64, 64, 3)), rng.integers(0, 255, (40, 160, 3)),
              rng.integers(0, 255, (64, 64, 3)), rng.integers(0, 255, (64, 64, 3))]
    for i, a in enumerate(arrays):
        _img(tmp_path / f"{i}.png", a)
    df = df.assign(image_path=[f"{i}.png" for i in range(6)])
    meta = eda.image_metadata(df, repo_root=tmp_path)
    assert meta.loc[3, "aspect_ratio"] == 4.0 and (meta["format"] == "PNG").all()
    res = eda.common_resolutions(meta, NAMES)
    assert res.iloc[0]["resolution"] == "64x64" and res.iloc[0]["samples"] == 5

    near = eda.near_duplicate_images(df, max_distance=0, repo_root=tmp_path)
    assert len(near) == 1 and set(near.iloc[0][["sample_id_a", "sample_id_b"]]) == {"s0", "s1"}
    assert not near.iloc[0]["cross_class"] and not near.iloc[0]["same_text"]

    out = eda.outliers(eda.text_features(df), meta)
    assert set(out.loc[out.category == "aspect_ratio_extreme", "sample_id"]) == {"s3"}
    assert set(out.loc[out.category == "image_small", "sample_id"]) == set(df.sample_id)


def test_plots_write_png(tmp_path, df):
    table, _ = eda.class_distribution(df, NAMES)
    paths = [
        eda_plots.class_distribution(table, tmp_path / "a.png"),
        eda_plots.length_by_class(eda.text_features(df), NAMES, tmp_path / "b.png"),
        eda_plots.histogram(pd.Series([1, 2, 3]), tmp_path / "c.png", "t", "x", note="a long note " * 30),
    ]
    for p in paths:
        assert p.exists() and p.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plots_are_deterministic(tmp_path, df):
    table, _ = eda.class_distribution(df, NAMES)
    a = eda_plots.class_distribution(table, tmp_path / "a.png").read_bytes()
    b = eda_plots.class_distribution(table, tmp_path / "b.png").read_bytes()
    assert a == b


def test_notebook_is_valid_and_output_free():
    nb = json.loads((REPO_ROOT / "notebooks" / "02_eda.ipynb").read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert code and all(not c.get("outputs") for c in code)   # no stored outputs (no dataset text)
    for c in code:
        ast.parse("".join(c["source"]))
