"""Tests for Task 12 tuning code: search spaces, trial logging, selection, parameter application,
early stopping, serialization, registry safety and the test-access audit. No real search is run."""

import csv
import importlib.util
import json

import joblib
import numpy as np
import pytest

from src.models import fusion
from src.models.text_baselines import build_pipeline
from src.training import tuning
from src.utils.config import REPO_ROOT, load_config


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------- search spaces ----------------

def test_grid_and_random_configs():
    space = {"a": [1, 2, 3], "b": ["x", "y"]}
    assert tuning.grid_size(space) == 6 and len(tuning.grid(space)) == 6
    assert tuning.grid(space)[0] == {"a": 1, "b": "x"}
    r1, r2 = tuning.random_configs(space, 4, seed=7), tuning.random_configs(space, 4, seed=7)
    assert r1 == r2 and len(r1) == 4 and len({json.dumps(c) for c in r1}) == 4          # reproducible, distinct
    assert {"a": 1, "b": "x"} not in tuning.random_configs(space, 5, seed=7, exclude=[{"a": 1, "b": "x"}])
    assert len(tuning.random_configs(space, 99, seed=1)) == 6                         # capped at the grid


def test_configured_search_spaces_are_compact_and_contain_the_baselines():
    c = load_config()
    t = c["tuning"]
    assert t["selection_metric"] == "macro_f1" and t["primary_seed"] == 42 and t["robustness_seeds"] == [42, 123, 2026]
    assert tuning.grid_size(t["traditional_text"]["space"]) == 90
    assert sum(len(v["C"]) for v in t["image"]["space"].values()) == 16
    assert tuning.grid_size(t["multimodal"]["space"]) == 486 and t["multimodal"]["n_trials"] == 30
    base = c["text_model"]["baselines"]
    b03 = {"C": base["logistic_regression"]["C"], "min_df": base["tfidf"]["min_df"],
           "ngram_range": base["tfidf"]["ngram_range"], "sublinear_tf": base["tfidf"]["sublinear_tf"]}
    assert b03 in tuning.grid(t["traditional_text"]["space"])
    assert 1.0 in t["image"]["space"]["logistic_regression"]["C"]                       # I01 setting
    m02 = {k: c["multimodal_model"]["fusion_mlp"][k] for k in t["multimodal"]["space"]}
    assert m02 in tuning.grid(t["multimodal"]["space"])
    assert t["multimodal"]["early_stopping"] == {"monitor": "validation macro_f1", "patience": 5,
                                                 "min_improvement": 0.0, "max_epochs": 50}


def test_params_json_is_canonical():
    assert tuning.params_json({"b": 1, "a": (1, 2)}) == '{"a": [1, 2], "b": 1}'


# ---------------- trial logging and selection ----------------

def test_trial_log_records_success_and_failure(tmp_path):
    log = tuning.TrialLog()
    ok = log.run("X-001", "s", "m", "text", {"C": 1}, 42, lambda: {"validation_macro_f1": 0.7, "validation_accuracy": 0.71})
    bad = log.run("X-002", "s", "m", "text", {"C": 2}, 42, lambda: 1 / 0)
    assert ok["status"] == "COMPLETED" and ok["training_time_s"] is not None
    assert bad["status"] == "FAILED" and "ZeroDivisionError" in bad["note"]                 # not dropped
    log.write(tmp_path / "t.csv")
    rows = list(csv.DictReader(open(tmp_path / "t.csv", encoding="utf-8")))
    assert [r["trial_id"] for r in rows] == ["X-001", "X-002"] and list(rows[0]) == tuning.TRIAL_COLUMNS


def test_select_best_uses_validation_metric_ties_and_skips_failed():
    rows = [{"trial_id": "a", "status": "COMPLETED", "validation_macro_f1": 0.70},
            {"trial_id": "b", "status": "FAILED", "validation_macro_f1": None},
            {"trial_id": "c", "status": "EARLY_STOPPED", "validation_macro_f1": 0.75},
            {"trial_id": "d", "status": "COMPLETED", "validation_macro_f1": 0.75},       # tie -> earlier (c) wins
            {"trial_id": "e", "status": "FAILED", "validation_macro_f1": 0.99}]          # failed never selected
    assert tuning.select_best(rows)["trial_id"] == "c"
    with pytest.raises(ValueError):
        tuning.select_best([{"trial_id": "x", "status": "FAILED", "validation_macro_f1": None}])


def test_seed_summary():
    s = tuning.seed_summary([0.80, 0.78, 0.79])
    assert s["n_seeds"] == 3 and s["mean"] == 0.79 and s["std"] == 0.01 and s["min"] == 0.78


# ---------------- parameter application ----------------

def test_tfidf_parameters_are_applied():
    tm = load_script("tune_models")
    p = tm.tfidf_lr({"ngram_range": [1, 3], "min_df": 3, "sublinear_tf": False, "C": 0.3},
                    {"max_df": 1.0, "max_features": None, "class_weight": None}, r"(?u)\w+", 42)
    vec, clf = p.steps[0][1], p.steps[1][1]
    assert vec.ngram_range == (1, 3) and vec.min_df == 3 and vec.sublinear_tf is False and clf.C == 0.3


def test_fusion_parameters_and_early_stopping_config_are_applied():
    tm = load_script("tune_models")
    cfg = tm.fusion_cfg(load_config(), {"projection_dim": 128, "hidden_dim": 512, "dropout": 0.5, "learning_rate": 3e-4,
                                        "weight_decay": 0.1, "batch_size": 32}, seed=123)
    assert (cfg.projection_dim, cfg.hidden_dim, cfg.dropout, cfg.learning_rate, cfg.weight_decay, cfg.batch_size, cfg.seed) == \
        (128, 512, 0.5, 3e-4, 0.1, 32, 123)
    assert (cfg.max_epochs, cfg.early_stopping_patience, cfg.min_improvement) == (50, 5, 0.0)


def test_min_improvement_controls_early_stopping():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 120)
    text, image = rng.normal(size=(120, 6)).astype(np.float32), rng.normal(size=(120, 5)).astype(np.float32)
    text[:, 0] += 2 * y
    cfg = fusion.FusionConfig(projection_dim=4, hidden_dim=4, max_epochs=20, early_stopping_patience=2,
                              min_improvement=1.0, learning_rate=1e-2)     # nothing can improve by > 1.0
    _, _, res = fusion.train_fusion(text[:90], image[:90], y[:90], text[90:], image[90:], y[90:], cfg, log=lambda *_: None)
    assert len(res.history) == 3 and res.best_epoch == 1


def test_tuned_text_pipeline_serialization(tmp_path):
    config = load_config()
    texts = ["ladki kitchen me", "bhai exam hai", "she is dumb", "movie dekhi", "ladki dumb", "exam kal hai"] * 4
    y = np.array([1, 0, 1, 0, 1, 0] * 4)
    pipe = build_pipeline("logistic_regression", "hinglish_normalized", config, 42)
    pipe.set_params(tfidf__ngram_range=(1, 3), tfidf__min_df=1, clf__C=0.5)
    pipe.fit(texts, y)
    joblib.dump(pipe, tmp_path / "b.joblib")
    again = joblib.load(tmp_path / "b.joblib")
    assert again.named_steps["tfidf"].ngram_range == (1, 3) and again.named_steps["clf"].C == 0.5
    np.testing.assert_allclose(again.predict_proba(texts), pipe.predict_proba(texts))


# ---------------- registry safety and test-set protection ----------------

def test_registry_update_is_append_only(tmp_path):
    rt = load_script("report_tuning")
    path = tmp_path / "reg.json"
    path.write_text(json.dumps({"models": {"B03": {"kind": "tfidf_logreg"}}}), encoding="utf-8")
    rt.update_registry(path, {"B07": {"kind": "tfidf_logreg", "artifact": "x"}})
    reg = json.loads(path.read_text(encoding="utf-8"))["models"]
    assert reg["B03"] == {"kind": "tfidf_logreg"} and "B07" in reg
    with pytest.raises(RuntimeError):
        rt.update_registry(path, {"B03": {"kind": "changed"}})


def test_test_access_audit_detects_test_metrics(tmp_path, monkeypatch):
    rt = load_script("report_tuning")
    monkeypatch.setattr(rt, "TUNE_M", tmp_path)
    monkeypatch.setattr(rt, "TABLES", tmp_path)
    (tmp_path / "B07.json").write_text(json.dumps({"test_set": "NOT EVALUATED"}), encoding="utf-8")
    audit = rt.test_access_audit(load_config())
    assert audit["passed"].all()
    (tmp_path / "x.csv").write_text("model,test_macro_f1\nB07,0.7\n", encoding="utf-8")
    assert not rt.test_access_audit(load_config())["passed"].all()


def test_task12_sources_never_touch_test_split():
    for rel in ("scripts/tune_models.py", "scripts/report_tuning.py", "src/training/tuning.py"):
        source = (REPO_ROOT / rel).read_text(encoding="utf-8")
        assert 'load_split("test"' not in source and "allow_test=True" not in source, rel
