"""Tests for config.yaml and the config loader (src/utils/config.py)."""

import copy
import re

import pytest

from src.utils.config import DEFAULT_CONFIG_PATH, load_config, validate_config


@pytest.fixture
def config():
    return load_config()


def test_config_loads_and_has_required_sections(config):
    # load_config already validates; this checks the real file passes.
    assert config["project"]["name"]


def test_random_seed_default(config):
    assert config["random_seed"] == 42


def test_split_ratios_sum_to_one(config):
    s = config["splitting"]
    assert abs(s["train"] + s["validation"] + s["test"] - 1.0) < 1e-6


def test_tuning_never_on_test_set(config):
    assert config["evaluation"]["tune_on"] == "validation"


def test_dataset_schema_not_assumed(config):
    # Dataset details must stay undecided until the inspection task.
    ds = config["dataset"]
    for key in ("text_column", "image_column", "label_column", "labels", "positive_label"):
        assert ds[key] is None, f"dataset.{key} should be null until verified"


def test_database_config_holds_only_env_var_names(config):
    db = config["database"]
    assert db["uri_env_var"] == "MONGODB_URI"
    assert "uri" not in db and "password" not in db


def test_config_file_contains_no_credentials():
    text = DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    assert not re.search(r"mongodb(\+srv)?://", text), "connection string found"
    assert not re.search(r"\bhf_[A-Za-z0-9]{20,}", text), "HF token found"


def test_validate_rejects_bad_split(config):
    bad = copy.deepcopy(config)
    bad["splitting"]["test"] = 0.30  # sums to 1.15
    with pytest.raises(ValueError):
        validate_config(bad)


def test_validate_rejects_missing_section(config):
    bad = copy.deepcopy(config)
    del bad["database"]
    with pytest.raises(ValueError):
        validate_config(bad)


def test_validate_rejects_non_integer_seed(config):
    bad = copy.deepcopy(config)
    bad["random_seed"] = "42"
    with pytest.raises(ValueError):
        validate_config(bad)
