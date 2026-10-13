"""Tests for category feasibility helpers and config/categories.yaml (Task 8.5)."""

import pytest
import yaml

from src.data.categories import CATEGORIES_PATH, enabled_tasks, feasibility_tier, load_categories
from src.utils.config import load_config


@pytest.mark.parametrize("train, val, expected", [
    (1756, 375, "STRONG"), (1049, 214, "STRONG"), (642, 152, "MODERATE"),
    (262, 55, "WEAK"), (1200, 90, "WEAK"), (150, 30, "INSUFFICIENT"), (0, 0, "INSUFFICIENT"),
])
def test_feasibility_tiers(train, val, expected):
    assert feasibility_tier(train, val) == expected


def test_shipped_category_config_is_valid():
    cfg = load_config()
    columns = ["FileName", "ExtractedText", "Misogyny", *cfg["dataset"]["secondary_label_columns"]]
    categories = load_categories(dataset_columns=columns)
    # only the original binary task is enabled; sub-categories await a decision
    assert enabled_tasks(categories) == ["misogyny_detection"]
    sub = categories["tasks"]["misogyny_subcategories"]
    assert sub["task_type"] == "multilabel" and not sub["enabled"]
    assert {c["source_column"] for c in sub["categories"].values()} == set(cfg["dataset"]["secondary_label_columns"])
    for name in ("SAFE", "HATE", "ABUSE", "THREAT", "OBSCENE", "SARCASM"):
        assert name in categories["unsupported"]


def _write(tmp_path, data):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _base():
    return {"tasks": {"t": {"task_type": "binary", "enabled": True, "categories": {
        "m": {"canonical": "MISOGYNY", "source_column": "Misogyny", "status": "VERIFIED_DATASET_LABEL"}}}},
        "unsupported": {"HATE": "no annotation"}}


def test_unsupported_category_cannot_be_configured(tmp_path):
    data = _base()
    data["tasks"]["t"]["categories"]["h"] = {"canonical": "HATE", "source_column": "Misogyny",
                                             "status": "VERIFIED_DATASET_LABEL"}
    with pytest.raises(ValueError, match="unsupported"):
        load_categories(_write(tmp_path, data))


def test_unverified_label_rejected(tmp_path):
    data = _base()
    data["tasks"]["t"]["categories"]["m"]["status"] = "PROPOSED_MAPPING"
    with pytest.raises(ValueError, match="verified"):
        load_categories(_write(tmp_path, data))


def test_missing_source_column_rejected(tmp_path):
    with pytest.raises(ValueError, match="not in dataset"):
        load_categories(_write(tmp_path, _base()), dataset_columns=["Other"])


def test_category_file_location():
    assert CATEGORIES_PATH.name == "categories.yaml" and CATEGORIES_PATH.parent.name == "config"
