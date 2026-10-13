"""Tests for TF-IDF baselines (synthetic data; no dataset needed)."""

import joblib
import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import bootstrap_macro_f1, classification_metrics
from src.models.text_baselines import TextPreprocessor, build_pipeline
from src.preprocessing import preprocess
from src.training import text_baselines as tb
from src.utils.config import load_config

NAMES = {0: "non_misogynistic", 1: "misogynistic"}


@pytest.fixture(scope="module")
def config():
    return load_config()


def make_split(n, seed, offset=0):
    rng = np.random.default_rng(seed)
    pos = ["ladki kr rha hai @page", "she is so dumb nhi", "women ko kitchen bhejo", "beti ko mat padhao"]
    neg = ["bhai exam kal hai", "mom ne khana banaya", "teacher class me late", "friend ke sath movie"]
    rows = []
    for i in range(n):
        label = int(rng.integers(0, 2))
        text = (pos if label else neg)[int(rng.integers(0, 4))] + f" extra{int(rng.integers(0, 5))}"
        rows.append({"sample_id": f"s{offset + i}", "text_raw": text, "target_label": str(label),
                     "review_flags": ""})
    df = pd.DataFrame(rows)
    df["text_basic"] = [preprocess(t, "basic") for t in df["text_raw"]]
    df["text_hinglish_normalized"] = [preprocess(t, "hinglish_normalized") for t in df["text_raw"]]
    return df


@pytest.fixture(scope="module")
def data():
    return make_split(120, 1), make_split(60, 2, offset=1000)


# ---------------- pipeline / preprocessing column ----------------

@pytest.mark.parametrize("mode, column", [("raw", "text_raw"), ("basic", "text_basic"),
                                          ("hinglish_normalized", "text_hinglish_normalized")])
def test_pipeline_applies_selected_preprocessing(config, data, mode, column):
    train, _ = data
    pipe = build_pipeline("logistic_regression", mode, config, 42).fit(train["text_raw"].tolist(),
                                                                       train["target_label"].astype(int))
    assert tb.check_preprocessing_matches_task6(pipe, train, mode)
    assert pipe.named_steps["preprocess"].transform(train["text_raw"].tolist()) == train[column].tolist()


def test_tfidf_vocabulary_learned_from_training_text_only(config, data):
    train, val = data
    val = val.assign(text_raw=val["text_raw"] + " zzvalidationonlyword")
    r = tb.run_experiment("T", "logistic_regression", "basic", train, val, config, NAMES)
    vocab = r.pipeline.named_steps["tfidf"].vocabulary_
    assert not any("zzvalidationonlyword" in term for term in vocab)


def test_devanagari_words_kept_whole(config):
    pipe = build_pipeline("logistic_regression", "basic", config, 42)
    pipe.fit(["नहीं पता ladki", "मत करो bhai", "नहीं ladki", "करो bhai"], [1, 0, 1, 0])
    vocab = pipe.named_steps["tfidf"].vocabulary_
    assert "नहीं" in vocab and "नह" not in vocab


@pytest.mark.parametrize("classifier", ["logistic_regression", "linear_svm"])
def test_valid_predictions_and_scores(config, data, classifier):
    train, val = data
    r = tb.run_experiment("T", classifier, "hinglish_normalized", train, val, config, NAMES)
    assert set(np.unique(r.val_pred)) <= {0, 1} and len(r.val_pred) == len(val)
    assert len(r.val_scores) == len(val) and r.converged
    assert r.metrics["macro_f1"] > 0.9            # trivially separable synthetic data


def test_deterministic_training(config, data):
    train, val = data
    a = tb.run_experiment("T", "linear_svm", "basic", train, val, config, NAMES)
    b = tb.run_experiment("T", "linear_svm", "basic", train, val, config, NAMES)
    assert np.array_equal(a.val_pred, b.val_pred) and np.allclose(a.val_scores, b.val_scores)


def test_serialization_roundtrip(config, data, tmp_path):
    train, val = data
    r = tb.run_experiment("T", "logistic_regression", "hinglish_normalized", train, val, config, NAMES)
    path = tmp_path / "m.joblib"
    joblib.dump(r.pipeline, path)
    reloaded = joblib.load(path)
    assert np.array_equal(reloaded.predict(val["text_raw"].tolist()), r.val_pred)
    # the saved pipeline accepts raw app-style input directly
    assert reloaded.predict(["Ye LADKI nhi samjhegi @someone"]).shape == (1,)


def test_preprocessor_rejects_version_mismatch():
    pre = TextPreprocessor(mode="basic", options={"version": "1.0"}, preprocessing_version="9.9")
    with pytest.raises(RuntimeError, match="preprocessing"):
        pre.transform(["x"])


def test_preprocessor_is_stateless():
    pre = TextPreprocessor(mode="basic")
    assert pre.fit(["a"]).transform(["kr  @u"]) == pre.fit(["completely", "different"]).transform(["kr  @u"])


def test_unknown_classifier_rejected(config):
    with pytest.raises(ValueError):
        build_pipeline("random_forest", "raw", config, 42)


# ---------------- metrics ----------------

def test_metrics_conventions():
    y_true = [0, 0, 0, 1, 1, 1, 1, 1]
    y_pred = [0, 0, 1, 1, 1, 1, 0, 0]
    m = classification_metrics(y_true, y_pred, NAMES)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (2, 1, 2, 3)
    assert m["precision"] == pytest.approx(3 / 4, abs=1e-4) and m["recall"] == pytest.approx(3 / 5, abs=1e-4)
    # class 1: P = 3/4, R = 3/5 ; class 0: P = 2/4 (predicted 0 four times, 2 correct), R = 2/3
    f1_pos, f1_neg = 2 * 0.75 * 0.6 / 1.35, 2 * 0.5 * (2 / 3) / (0.5 + 2 / 3)
    assert m["macro_f1"] == pytest.approx((f1_pos + f1_neg) / 2, abs=1e-4)
    assert m["weighted_f1"] == pytest.approx((3 * f1_neg + 5 * f1_pos) / 8, abs=1e-4)
    assert m["support_misogynistic"] == 5 and m["accuracy"] == pytest.approx(5 / 8, abs=1e-4)


def test_bootstrap_interval_contains_point_estimate_and_is_deterministic():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    p = np.where(rng.random(300) < 0.7, y, 1 - y)
    lo, hi = bootstrap_macro_f1(y, p, 300, seed=42)
    point = classification_metrics(y, p, NAMES)["macro_f1"]
    assert lo <= point <= hi and (lo, hi) == bootstrap_macro_f1(y, p, 300, seed=42)


# ---------------- test-set protection ----------------

def test_development_loader_never_requests_test(monkeypatch):
    requested = []
    fake = pd.DataFrame({"sample_id": ["a"], "target_label": ["1"]})

    def fake_load_split(name, allow_test=False, config=None):
        requested.append((name, allow_test))
        return fake.assign(sample_id=[name])

    assignments = pd.DataFrame({"sample_id": ["train", "validation", "x"], "split": ["train", "validation", "test"]})
    monkeypatch.setattr(tb, "load_split", fake_load_split)
    monkeypatch.setattr(tb, "load_split_assignments", lambda config=None: assignments)
    train, val = tb.load_development_data()
    assert [n for n, _ in requested] == ["train", "validation"] and not any(a for _, a in requested)


def test_training_script_does_not_reference_test_split():
    from src.utils.config import REPO_ROOT
    source = (REPO_ROOT / "scripts" / "train_text_baselines.py").read_text(encoding="utf-8")
    source += (REPO_ROOT / "src" / "training" / "text_baselines.py").read_text(encoding="utf-8")
    assert 'load_split("test"' not in source and "allow_test=True" not in source


# ---------------- diagnostics ----------------

def test_feature_flags():
    assert tb.feature_flag("doublemeaning in") == "watermark_handle"
    assert tb.feature_flag("user user") == "mention_placeholder"
    assert tb.feature_flag("8") == "number_only"
    assert tb.feature_flag("ladki") == ""


def test_error_table_has_no_text_and_flags(config, data):
    train, val = data
    val = val.copy()
    val.loc[val.index[0], "text_raw"] = ""
    results = [tb.run_experiment(f"B0{i}", c, m, train, val, config, NAMES)
               for i, (c, m) in enumerate([("logistic_regression", "raw"), ("linear_svm", "basic")], start=1)]
    errors = tb.error_table(val, results, "B01")
    assert "text_raw" not in errors.columns
    assert set(errors["error_type"]) <= {"false_positive", "false_negative"}
