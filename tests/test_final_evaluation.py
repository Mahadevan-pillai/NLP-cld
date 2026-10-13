"""Tests for Task 13 final-evaluation code: checksums, statistics, consistency, freeze manifest, test-split access.

No model is run and no data file is needed (the manifest test only reads the committed JSON).
"""

import importlib.util
import json

import numpy as np
import pytest
from scipy.stats import binomtest
from sklearn.metrics import f1_score

from src.evaluation import final
from src.evaluation.metrics import classification_metrics
from src.utils.config import REPO_ROOT, load_config


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------- checksums ----------------

def test_sha256_path_file_and_folder(tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "d" / "b.txt").write_text("y", encoding="utf-8")
    h_file, h_dir = final.sha256_path(tmp_path / "d" / "a.txt"), final.sha256_path(tmp_path / "d")
    assert h_file == "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881"     # sha256("x")
    (tmp_path / "d" / "b.txt").write_text("changed", encoding="utf-8")
    assert final.sha256_path(tmp_path / "d") != h_dir
    with pytest.raises(FileNotFoundError):
        final.sha256_path(tmp_path / "missing")


# ---------------- metrics and statistics ----------------

def test_macro_f1_fast_matches_sklearn():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y, p = rng.integers(0, 2, 200), rng.integers(0, 2, 200)
        assert abs(final.macro_f1_fast(y, p) - f1_score(y, p, average="macro", zero_division=0)) < 1e-12
    assert final.macro_f1_fast(np.array([1, 1]), np.array([1, 1])) == 0.5          # absent class scores 0 (zero_division=0)


def test_bootstrap_is_reproducible_and_paired():
    rng = np.random.default_rng(1)
    y, a = rng.integers(0, 2, 300), rng.integers(0, 2, 300)
    idx1, idx2 = final.bootstrap_indices(300, 500, seed=42), final.bootstrap_indices(300, 500, seed=42)
    assert np.array_equal(idx1, idx2)
    ci = final.bootstrap_ci(y, a, idx1)
    assert ci["ci_low"] <= ci["macro_f1"] <= ci["ci_high"]
    same = final.paired_bootstrap_difference(y, a, a, idx1)
    assert same["difference"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0
    better = np.where(rng.random(300) < 0.9, y, a)                                  # mostly correct
    d = final.paired_bootstrap_difference(y, better, a, idx1)
    assert d["difference"] > 0 and d["ci_low"] > 0 and d["share_of_resamples_a_better"] > 0.95


def test_mcnemar_exact_and_holm():
    y = np.array([1] * 20)
    a = np.array([1] * 15 + [0] * 5)          # a right on 15
    b = np.array([1] * 5 + [0] * 15)          # b right on 5 (all of which a also gets right)
    r = final.mcnemar_exact(y, a, b)
    assert (r["only_a_correct"], r["only_b_correct"], r["discordant"]) == (10, 0, 10)
    assert abs(r["p_value"] - binomtest(10, 10, 0.5).pvalue) < 1e-12
    assert final.mcnemar_exact(y, a, a)["p_value"] == 1.0
    adj = final.holm({"x": 0.01, "y": 0.04, "z": 0.03})
    assert adj == {"x": 0.03, "z": 0.06, "y": 0.06}                                  # monotone Holm step-down


def test_confusion_consistency():
    names = {0: "non_misogynistic", 1: "misogynistic"}
    y, p = np.array([0, 0, 1, 1, 1]), np.array([0, 1, 1, 0, 1])
    m = classification_metrics(y, p, names)
    assert final.confusion_consistent(m, 5)
    assert not final.confusion_consistent(m, 6)
    assert not final.confusion_consistent({**m, "tp": m["tp"] + 1}, 5)


# ---------------- frozen configuration and plan ----------------

def test_evaluation_scope_and_finals_match_config():
    fe = load_script("final_evaluation")
    c = load_config()
    assert fe.FINALS == c["final_models"] == {"FINAL_TEXT_MODEL": "T05", "FINAL_IMAGE_MODEL": "I03",
                                               "FINAL_MULTIMODAL_MODEL": "M04"}
    assert set(fe.FINALS.values()) <= set(fe.MODELS)
    assert [list(v) for v in fe.PLAN["primary_comparisons"].values()] == c["final_evaluation"]["primary_comparisons"]
    assert fe.PLAN["bootstrap"]["resamples"] == c["final_evaluation"]["bootstrap"]["resamples"] == 10000
    for fam in fe.PLAN["rq1_comparisons"].values():
        assert set(fam) <= set(fe.MODELS)


def test_freeze_manifest_is_complete_when_present():
    path = REPO_ROOT / "results/final/model_freeze_manifest.json"
    if not path.exists():
        pytest.skip("freeze manifest not created yet")
    m = json.loads(path.read_text(encoding="utf-8"))
    fe = load_script("final_evaluation")
    assert m["finals"] == fe.FINALS and set(m["models"]) == set(fe.MODELS)
    assert all(v["validation_reproduced_by_evaluation_code"] for v in m["models"].values())
    assert all(f"artifact:{k}" in m["checksums"] for k in fe.MODELS)
    for key in ("config.yaml", "split_assignments.csv", "hinglish_rules", "src/preprocessing", "muril_weights", "siglip_weights"):
        assert len(m["checksums"][key]) == 64
    assert m["evaluation_plan"]["primary_comparisons"] == fe.PLAN["primary_comparisons"]


def test_evaluation_refuses_without_freeze(tmp_path, monkeypatch):
    fe = load_script("final_evaluation")
    monkeypatch.setattr(fe, "MANIFEST", tmp_path / "missing.json")
    with pytest.raises(SystemExit):
        fe.verify_freeze(load_config())


def test_only_the_final_evaluation_script_opens_the_test_split():
    needle = "load_split(" + '"test", allow_test=True'
    users = sorted(p.relative_to(REPO_ROOT).as_posix() for folder in ("src", "scripts", "app")
                   for p in (REPO_ROOT / folder).rglob("*.py") if needle in p.read_text(encoding="utf-8"))
    assert users == ["scripts/final_evaluation.py"]
